"""Build the windowed training dataset from the SMART-PDM washing machine recordings.

Each cycle folder ``<begin>_<end>`` holds ``slow.csv`` (1 Hz power meter) and
``fast.csv`` (2048 Hz current and vibration). The output has one row per
sliding window of ``window`` seconds, every ``stride`` seconds, for each cycle
with a label in ``washing_machine_metadata.csv``. ``TIMESTAMP`` is the exclusive
window end in unix seconds: the window covers ``[TIMESTAMP - window, TIMESTAMP)``.

    python -m data_prep.build_dataset <raw_dir> <out_csv> [--window 60] [--stride 10]
        [--workers N] [--with-slow]

By default only fast.csv features are used: the slow stream is unreliable (see
below) and in the Motor cycles it only shows an idle meter, which would teach a
model a recording artifact. ``--with-slow`` adds the slow features.

Cycles are skipped, with the reason printed, when they have no label, a
missing or unreadable fast.csv (or slow.csv with ``--with-slow``), a fast.csv
shorter than one window, missing values or signals that are not ADC counts in
fast.csv, or a fast.csv byte-identical to another cycle's (all copies are
dropped when their labels differ, otherwise one is kept).

How fast.csv lines up with slow.csv (checked on all 96 cycles):

- The fast timestamp is not a clock reading. It starts at 0 and equals
  ``floor(sample_index * 1e6 / 2048)`` on every row, so it only counts samples.
  In three December 2021 files it even jumps backwards every 102000 rows. The
  sample index is used as the fast time axis instead.
- Fast and slow recordings start and stop at different moments. Matching the
  per-second fast Current spread against slow ``A`` gives offsets from about
  -3000 s to +4000 s, and fast often runs long after slow ends. Neither
  ``timestamp_begin`` nor the first slow ``Ts`` marks the fast start.
- That match is clear (correlation >= 0.8) in 30 of the 83 usable cycles. In all
  Motor cycles and three Bearings cycles slow ``ActP`` is 0 and ``A`` is flat (the
  meter did not see the machine run), so nothing can be matched. Some fast files
  also lose samples, which shortens their time axis.
- Most December 2021 fast files hold less than 90 s, and three hold Current in
  volts instead of ADC counts.

With ``--with-slow``, ``align_offset`` estimates the offset per cycle. Below
``MIN_ALIGN_CORR`` the fast recording is assumed to start with the slow one. The
label belongs to the whole cycle, so a wrong offset pairs slow and fast windows
from different moments of the same cycle but never mixes labels.
"""
import argparse
import hashlib
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

METADATA_FILE = 'washing_machine_metadata.csv'
METADATA_COLUMNS = ['begin_end', 'timestamp_begin', 'timestamp_end', 'brand', 'model',
                    'program', 'temperature', 'spin', 'load', 'failure']
META_COLUMNS = ['TIMESTAMP', 'DateTime', 'target', 'cycle_id', 'brand', 'model']
# ActE is a cumulative energy counter (identifies the meter and the time); Fr and V
# describe the grid, not the machine
SLOW_SIGNALS = ['ActP', 'RctP', 'AppP', 'PF', 'A']
FAST_SIGNALS = ['Current', 'Vibration']
FAST_RATE = 2048
# Roughly log-spaced spectral bands in Hz; the last one ends at Nyquist
BAND_EDGES = [1, 6, 32, 180, 1024]
# Share of the seconds of a slow window that must be present
MIN_SLOW_COVERAGE = 0.8
MIN_ALIGN_CORR = 0.8


def load_metadata(raw_dir):
    """Cycle labels, one row per ``begin_end`` folder name."""
    meta = pd.read_csv(os.path.join(raw_dir, METADATA_FILE), header=None,
                       usecols=range(len(METADATA_COLUMNS)), names=METADATA_COLUMNS)
    meta = meta.dropna(subset=['failure']).drop_duplicates('begin_end')
    return meta.reset_index(drop=True)


def slow_features(slow, window=60, stride=10):
    """Rolling mean/std/min/max of the slow signals, indexed by the last covered Ts.

    Windows end at ``first Ts + window - 1 + k * stride``; windows with fewer
    than ``MIN_SLOW_COVERAGE`` of their seconds present are dropped.
    """
    signals = slow.groupby('Ts')[SLOW_SIGNALS].mean()
    start, end = int(signals.index.min()), int(signals.index.max())
    signals = signals.reindex(range(start, end + 1))
    rolling = signals.rolling(window, min_periods=int(np.ceil(MIN_SLOW_COVERAGE * window)))
    stats = {'mean': rolling.mean(), 'std': rolling.std(), 'min': rolling.min(),
             'max': rolling.max()}
    features = pd.concat({f'{sig}_{name}': df[sig] for sig in SLOW_SIGNALS
                          for name, df in stats.items()}, axis=1)
    ends = range(start + window - 1, end + 1, stride)
    return features.reindex(ends).dropna()


def _second_stats(block):
    """Moment sums, extremes and band powers of each row (one second) of ``block``.

    Each second is Hann-tapered before the FFT to limit spectral leakage.
    """
    tapered = (block - block.mean(axis=1, keepdims=True)) * np.hanning(FAST_RATE)
    spectrum = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FAST_RATE, 1 / FAST_RATE)
    stats = {f's{k}': (block ** k).sum(axis=1) for k in (1, 2, 3, 4)}
    stats.update(min=block.min(axis=1), max=block.max(axis=1))
    for i, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        in_band = (freqs >= lo) & ((freqs < hi) | (hi == BAND_EDGES[-1]))
        stats[f'band{i}'] = spectrum[:, in_band].sum(axis=1)
    return stats


def fast_seconds(fast_csv, chunk_seconds=256):
    """Per-second statistics of ``fast.csv``, read in chunks.

    Row ``k`` covers samples ``k * 2048`` to ``(k + 1) * 2048 - 1``; a trailing
    partial second is dropped. Moment sums are taken around the mean of the
    first second, which keeps them precise and cancels out in the features.
    """
    reader = pd.read_csv(fast_csv, usecols=FAST_SIGNALS, dtype='float64',
                         chunksize=chunk_seconds * FAST_RATE)
    parts, center = [], None
    for chunk in reader:
        values = chunk[FAST_SIGNALS].values
        if np.isnan(values).any():
            raise ValueError('missing values in fast.csv')
        if not np.array_equal(values, np.round(values)):
            raise ValueError('fast.csv signals are not integer ADC counts')
        n_sec = len(values) // FAST_RATE
        if n_sec == 0:
            break
        blocks = values[:n_sec * FAST_RATE].reshape(n_sec, FAST_RATE, len(FAST_SIGNALS))
        if center is None:
            center = blocks[0].mean(axis=0)
        part = {}
        for j, sig in enumerate(FAST_SIGNALS):
            for name, col in _second_stats(blocks[:, :, j] - center[j]).items():
                part[f'{sig}_{name}'] = col
        parts.append(pd.DataFrame(part))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fast_features(seconds, window=60):
    """Window features from ``fast_seconds`` output, indexed by the window's last second.

    ``_kurtosis`` is the excess kurtosis (0 for a flat signal) and ``_band*``
    are the shares of the spectral energy in ``BAND_EDGES``. Each window only
    uses its own samples.
    """
    n = window * FAST_RATE
    sums = seconds.rolling(window).sum()
    features = {}
    for sig in FAST_SIGNALS:
        m1, m2, m3, m4 = (sums[f'{sig}_s{k}'] / n for k in (1, 2, 3, 4))
        var = (m2 - m1 ** 2).clip(lower=0)
        central4 = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4).clip(lower=0)
        features[f'{sig}_std'] = np.sqrt(var * n / (n - 1))
        features[f'{sig}_p2p'] = (seconds[f'{sig}_max'].rolling(window).max()
                                  - seconds[f'{sig}_min'].rolling(window).min())
        # A flat signal has no defined kurtosis or spectrum: report 0
        features[f'{sig}_kurtosis'] = (central4 / var ** 2 - 3).where(var > 1e-9, 0.0)
        bands = sums[[f'{sig}_band{i}' for i in range(len(BAND_EDGES) - 1)]]
        total = bands.sum(axis=1)
        for i, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
            share = bands[f'{sig}_band{i}'] / total
            features[f'{sig}_band_{lo}_{hi}hz'] = share.where(total > 0, 0.0)
    return pd.DataFrame(features).iloc[window - 1:]


def align_offset(slow, seconds, min_overlap=120):
    """Offset ``d`` such that fast second ``k`` is slow ``Ts = first Ts + d + k``.

    Correlates the per-second fast Current spread with slow ``A`` at every
    shift that overlaps at least half of the shorter recording. Returns
    ``(offset, correlation)``; the correlation is nan when no shift qualifies
    or slow ``A`` hardly varies.
    """
    amps = slow.groupby('Ts')['A'].mean()
    start = int(amps.index.min())
    slow_a = amps.reindex(range(start, int(amps.index.max()) + 1)).values
    # A flat A (meter not seeing the machine run) only gives chance correlations
    if not np.nanstd(slow_a) >= 0.1 * abs(np.nanmean(slow_a)):
        return 0, np.nan
    m1 = seconds['Current_s1'] / FAST_RATE
    fast_i = np.sqrt((seconds['Current_s2'] / FAST_RATE - m1 ** 2).clip(lower=0)).values
    need = max(min_overlap, min(len(fast_i), len(slow_a)) // 2)
    best = (0, np.nan)
    for lag in range(-len(fast_i) + need, len(slow_a) - need + 1):
        k0, k1 = max(0, -lag), min(len(fast_i), len(slow_a) - lag)
        x, y = fast_i[k0:k1], slow_a[k0 + lag:k1 + lag]
        ok = ~np.isnan(y)
        if ok.sum() < need or x[ok].std() == 0 or y[ok].std() == 0:
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        if not r <= best[1]:
            best = (lag, r)
    return best


def build_cycle(cycle_dir, labels, window=60, stride=10, with_slow=False):
    """Windowed rows of one cycle folder.

    ``labels`` holds failure, brand, model and timestamp_begin. Windows are cut
    on the fast time axis. ``TIMESTAMP`` is the exclusive window end in unix
    seconds: the window covers ``[TIMESTAMP - window, TIMESTAMP)``. Fast-only, it
    is ``timestamp_begin`` plus the window end on the fast axis; the real fast
    start time is unknown, so it is only nominal. ``with_slow`` adds the slow
    features of the aligned slow windows and then follows their ``Ts`` (last
    covered second + 1). Raises ``ValueError`` when the cycle is unusable.
    """
    cycle_id = os.path.basename(os.path.normpath(cycle_dir))
    seconds = fast_seconds(os.path.join(cycle_dir, 'fast.csv'))
    if len(seconds) < window:
        raise ValueError('fast.csv shorter than one window')
    features = fast_features(seconds, window)
    info = ''
    if with_slow:
        slow = pd.read_csv(os.path.join(cycle_dir, 'slow.csv'), usecols=['Ts'] + SLOW_SIGNALS)
        offset, corr = align_offset(slow, seconds, min_overlap=2 * window)
        if not corr >= MIN_ALIGN_CORR:
            offset = 0
        # Fast second k is slow Ts = first Ts + offset + k
        features.index = features.index + int(slow['Ts'].min()) + offset
        features = slow_features(slow, window, stride).join(features, how='inner')
        features.index = features.index + 1
        info = f', fast offset {offset:+d} s (corr {corr:.2f})'
    else:
        features = features.iloc[::stride]
        features.index = features.index + 1 + int(labels['timestamp_begin'])
    if features.empty:
        raise ValueError('no window with both slow and fast data')
    rows = pd.DataFrame({
        'TIMESTAMP': features.index.astype('int64'),
        'DateTime': pd.to_datetime(features.index, unit='s'),
        'target': labels['failure'],
        'cycle_id': cycle_id,
        'brand': labels['brand'],
        'model': labels['model'],
    })
    rows = pd.concat([rows, features.reset_index(drop=True)], axis=1)
    print(f'{cycle_id}: {len(rows)} windows, {labels["failure"]}{info}', flush=True)
    return rows


def _build_task(args):
    try:
        return build_cycle(*args), None
    except (ValueError, OSError) as err:
        return args[0], str(err)


def _md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def duplicate_fast_files(raw_dir, folders):
    """Groups of folders whose fast.csv files are byte-identical (same size, then md5)."""
    by_size, by_hash = {}, {}
    for d in folders:
        by_size.setdefault(os.path.getsize(os.path.join(raw_dir, d, 'fast.csv')), []).append(d)
    for same_size in by_size.values():
        if len(same_size) > 1:
            for d in same_size:
                by_hash.setdefault(_md5(os.path.join(raw_dir, d, 'fast.csv')), []).append(d)
    return [sorted(group) for group in by_hash.values() if len(group) > 1]


def build_dataset(raw_dir, out_csv, window=60, stride=10, workers=None, with_slow=False):
    meta = load_metadata(raw_dir).set_index('begin_end')
    folders = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))
    required = ['fast.csv', 'slow.csv'] if with_slow else ['fast.csv']
    skipped = {}
    for d in folders:
        missing = [f for f in required if not os.path.isfile(os.path.join(raw_dir, d, f))]
        if d not in meta.index:
            skipped[d] = 'no label in the metadata'
        elif missing:
            skipped[d] = f'missing {", ".join(missing)}'
    folders = [d for d in folders if d not in skipped]
    for group in duplicate_fast_files(raw_dir, folders):
        if meta.loc[group, 'failure'].nunique() > 1:
            for d in group:
                skipped[d] = f'fast.csv identical in {group} with different labels'
        else:
            for d in group[1:]:
                skipped[d] = f'fast.csv identical to {group[0]}'
    tasks = [(os.path.join(raw_dir, d), meta.loc[d].to_dict(), window, stride, with_slow)
             for d in folders if d not in skipped]
    if with_slow:
        print('WARNING: slow.csv features carry recording artifacts (the meter reads zero '
              'power in every Motor cycle) and are often misaligned with fast.csv')
    workers = workers or max(1, cpu_count() - 1)
    started = time.time()
    parts = []
    if not tasks:
        raise ValueError(f'No usable cycle in {raw_dir}')
    with Pool(min(workers, len(tasks))) as pool:
        for rows, reason in pool.imap_unordered(_build_task, tasks):
            if reason is None:
                parts.append(rows)
            else:
                skipped[os.path.basename(rows)] = reason
                print(f'{os.path.basename(rows)}: skipped, {reason}', flush=True)
    print(f'Skipped {len(skipped)} folders:')
    for d, reason in sorted(skipped.items()):
        print(f'  {d}: {reason}')
    if not parts:
        raise ValueError(f'No cycle in {raw_dir} produced a window')
    df = pd.concat(parts, ignore_index=True).sort_values(['cycle_id', 'TIMESTAMP'])
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f'NaN values in {nan_cols}')
    df.to_csv(out_csv, index=False, float_format='%.6g')
    print(f'Wrote {out_csv}: {df.shape[0]} rows, {df.shape[1]} columns, '
          f'{df["cycle_id"].nunique()} cycles in {time.time() - started:.0f} s')
    print(df.groupby('target')['cycle_id'].nunique().to_string())
    return df


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'must be > 0, got {value}')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the windowed washing machine dataset')
    parser.add_argument('raw_dir', help='folder with the cycle folders and the metadata CSV')
    parser.add_argument('out_csv')
    parser.add_argument('--window', type=_positive_int, default=60,
                        help='window length in seconds')
    parser.add_argument('--stride', type=_positive_int, default=10,
                        help='seconds between windows')
    parser.add_argument('--workers', type=int, help='processes (default: CPUs - 1)')
    parser.add_argument('--with-slow', action='store_true',
                        help='add slow.csv features, aligned to the fast windows')
    args = parser.parse_args(argv)
    build_dataset(args.raw_dir, args.out_csv, args.window, args.stride, args.workers,
                  args.with_slow)


if __name__ == '__main__':
    main()
