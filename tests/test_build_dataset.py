import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from data_prep.build_dataset import (
    FAST_RATE,
    META_COLUMNS,
    align_offset,
    fast_features,
    fast_seconds,
    load_metadata,
    main,
)

SECONDS = 80
CYCLES = {
    '2022-01-01_10.00.00_2022-01-01_10.01.20': 'Working',
    '2022-01-02_10.00.00_2022-01-02_10.01.20': 'Bearings',
    '2022-01-03_10.00.00_2022-01-03_10.01.20': 'Motor',
}
UNLABELED = '2022-01-04_10.00.00_2022-01-04_10.01.20'


def write_cycle(folder, t0, seed, fast_delay=0):
    """slow.csv and fast.csv whose current follows the same random on/off profile."""
    rng = np.random.default_rng(seed)
    amps = rng.choice([50, 400, 900], SECONDS + fast_delay)
    folder.mkdir()
    pd.DataFrame({
        'Ts': t0 + np.arange(SECONDS), 'ActE': 1000 + np.arange(SECONDS) // 10,
        'ActP': amps[:SECONDS] * 2, 'RctP': rng.integers(-20, 20, SECONDS),
        'AppP': amps[:SECONDS] * 2 + 5, 'Fr': 5000, 'PF': rng.integers(500, 1000, SECONDS),
        'V': rng.integers(22000, 24000, SECONDS), 'A': amps[:SECONDS] * 10,
    }).to_csv(folder / 'slow.csv', index=False)
    n = SECONDS * FAST_RATE
    t = np.arange(n) / FAST_RATE
    amp = np.repeat(amps[fast_delay:fast_delay + SECONDS], FAST_RATE)
    pd.DataFrame({
        'UnixTimestamp (us)': (np.arange(n) * 1e6 / FAST_RATE).astype(int),
        'Current': np.round(1850 + amp * np.sin(2 * np.pi * 50 * t)).astype(int),
        'Vibration': np.round(1900 + 300 * np.sin(2 * np.pi * 400 * t)
                              + rng.normal(0, 50, n)).astype(int),
    }).to_csv(folder / 'fast.csv', index=False)


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = []
    for i, (name, failure) in enumerate(CYCLES.items()):
        t0 = 1640995200 + i * 86400
        write_cycle(raw / name, t0, seed=i)
        rows.append(f'{name},{t0},{t0 + SECONDS},kunft,KWM5317,Cotton,40,800,Full,'
                    f'{failure},SummerTime')
    write_cycle(raw / UNLABELED, 1641254400, seed=9)
    rows.append(rows[0])  # duplicate row
    (raw / 'washing_machine_metadata.csv').write_text('\n'.join(rows) + '\n')
    return raw


def build(raw_dir, tmp_path, *extra):
    out = tmp_path / 'out.csv'
    main([str(raw_dir), str(out), '--window', '20', '--stride', '5', '--workers', '2', *extra])
    return pd.read_csv(out)


def add_cycle(raw_dir, name, failure, copy_fast_from=None, seed=7):
    t0 = 1641513600
    write_cycle(raw_dir / name, t0, seed=seed)
    if copy_fast_from:
        shutil.copy(raw_dir / copy_fast_from / 'fast.csv', raw_dir / name / 'fast.csv')
    with open(raw_dir / 'washing_machine_metadata.csv', 'a') as f:
        f.write(f'{name},{t0},{t0 + SECONDS},kunft,KWM5317,Cotton,40,800,Full,{failure},\n')


def test_load_metadata_drops_duplicates_and_extra_column(raw_dir):
    meta = load_metadata(raw_dir)
    assert len(meta) == 3
    assert meta['begin_end'].is_unique
    assert 'SummerTime' not in meta.values
    assert meta.columns[-1] == 'failure'


def test_build_dataset_fast_only_by_default(raw_dir, tmp_path):
    df = build(raw_dir, tmp_path)

    assert list(df.columns[:len(META_COLUMNS)]) == META_COLUMNS
    assert set(df['cycle_id']) == set(CYCLES)
    assert dict(df.groupby('cycle_id')['target'].first()) == CYCLES
    # 80 s of data, windows of 20 s every 5 s
    assert (df.groupby('cycle_id').size() == 13).all()
    # TIMESTAMP is timestamp_begin + window end in seconds
    first = df.groupby('cycle_id')['TIMESTAMP'].min()
    assert first[list(CYCLES)[1]] == 1640995200 + 86400 + 20
    assert (df.groupby('cycle_id')['TIMESTAMP'].diff().dropna() == 5).all()
    assert (pd.to_datetime(df['DateTime']) == pd.to_datetime(df['TIMESTAMP'], unit='s')).all()

    features = df.drop(columns=META_COLUMNS)
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features)
    assert not df.isna().any().any()
    assert all(c.startswith(('Current_', 'Vibration_')) for c in features)
    for name in ('Current_rms', 'Current_std', 'Vibration_kurtosis', 'Vibration_p2p',
                 'Vibration_band_180_1024hz'):
        assert name in features
    bands = features.filter(regex='^Vibration_band_')
    assert np.allclose(bands.sum(axis=1), 1)
    # 400 Hz tone
    assert (bands['Vibration_band_180_1024hz'] > 0.9).all()


def test_build_dataset_with_slow(raw_dir, tmp_path):
    df = build(raw_dir, tmp_path, '--with-slow')
    features = df.drop(columns=META_COLUMNS).columns
    assert set(df['cycle_id']) == set(CYCLES)
    assert (df.groupby('cycle_id').size() == 13).all()
    assert not df.isna().any().any()
    for name in ('ActP_mean', 'A_std', 'V_min', 'PF_max', 'Current_rms', 'Vibration_std'):
        assert name in features
    assert not any(c.startswith(('ActE', 'Fr_')) for c in features)


def test_duplicate_fast_files_with_conflicting_labels_are_dropped(raw_dir, tmp_path):
    working, bearings = list(CYCLES)[:2]
    add_cycle(raw_dir, '2022-01-05_10.00.00_2022-01-05_10.01.20', 'Motor',
              copy_fast_from=working)
    df = build(raw_dir, tmp_path)
    assert set(df['cycle_id']) == {bearings, list(CYCLES)[2]}


def test_duplicate_fast_files_with_same_label_keep_one(raw_dir, tmp_path):
    working = list(CYCLES)[0]
    copy = '2022-01-05_10.00.00_2022-01-05_10.01.20'
    add_cycle(raw_dir, copy, 'Working', copy_fast_from=working)
    df = build(raw_dir, tmp_path)
    assert set(df['cycle_id']) == set(CYCLES)


def test_short_fast_file_is_skipped(raw_dir, tmp_path):
    short = '2022-01-05_10.00.00_2022-01-05_10.01.20'
    add_cycle(raw_dir, short, 'Heating')
    fast = pd.read_csv(raw_dir / short / 'fast.csv')
    fast.iloc[:10 * FAST_RATE].to_csv(raw_dir / short / 'fast.csv', index=False)
    df = build(raw_dir, tmp_path)
    assert short not in set(df['cycle_id'])
    assert set(df['cycle_id']) == set(CYCLES)


def test_fast_features_match_direct_computation(tmp_path):
    folder = tmp_path / 'cycle'
    write_cycle(folder, 0, seed=3)
    window = 20
    feats = fast_features(fast_seconds(folder / 'fast.csv', chunk_seconds=7), window)
    assert list(feats.index) == list(range(window - 1, SECONDS))

    signal = pd.read_csv(folder / 'fast.csv')['Vibration'].values.astype(float)
    end = 42
    x = signal[(end - window + 1) * FAST_RATE:(end + 1) * FAST_RATE]
    row = feats.loc[end]
    assert row['Vibration_std'] == pytest.approx(x.std(ddof=1))
    assert row['Vibration_p2p'] == x.max() - x.min()
    assert row['Vibration_kurtosis'] == pytest.approx(kurtosis(x), abs=1e-6)
    dc = np.median(signal[:SECONDS * FAST_RATE].reshape(SECONDS, FAST_RATE).mean(axis=1))
    assert row['Vibration_rms'] == pytest.approx(np.sqrt(np.mean((x - dc) ** 2)))


def test_align_offset_recovers_late_fast_start(tmp_path):
    folder = tmp_path / 'cycle'
    write_cycle(folder, 1000, seed=4, fast_delay=7)
    slow = pd.read_csv(folder / 'slow.csv')
    offset, corr = align_offset(slow, fast_seconds(folder / 'fast.csv'), min_overlap=20)
    assert offset == 7
    assert corr > 0.99
