import pandas as pd

from components.preprocess import preprocess


def run_preprocess(raw_csv, tmp_path, **kwargs):
    paths = [tmp_path / f'{name}.csv' for name in ('train', 'val', 'test')]
    preprocess(str(raw_csv), *map(str, paths), **kwargs)
    return [pd.read_csv(p, parse_dates=['DateTime']) for p in paths]


def test_preprocess_cleans_and_splits_chronologically(raw_csv, tmp_path):
    train, val, test = run_preprocess(raw_csv, tmp_path,
                                      val_size=0.2, test_size=0.2, gap=10)

    for df in (train, val, test):
        assert not any(c.startswith('Unnamed:') for c in df.columns)
        assert 'constant' not in df.columns
        assert {'TIMESTAMP', 'DateTime', 'target'} <= set(df.columns)
    # 600 rows: val starts at 360, test at 480, 10 rows dropped before each
    assert (len(train), len(val), len(test)) == (350, 110, 120)
    assert train['DateTime'].max() < val['DateTime'].min()
    assert val['DateTime'].max() < test['DateTime'].min()


def test_preprocess_gap_separates_splits(raw_csv, tmp_path):
    train, val, test = run_preprocess(raw_csv, tmp_path, gap=30)
    # Rows are one minute apart: the gap leaves 30 unused minutes
    minute = pd.Timedelta(minutes=1)
    assert val['DateTime'].min() - train['DateTime'].max() == 31 * minute
    assert test['DateTime'].min() - val['DateTime'].max() == 31 * minute


def test_preprocess_drops_missing_targets_and_duplicates(raw_csv, tmp_path):
    df = pd.read_csv(raw_csv)
    df.loc[0, 'target'] = None
    df = pd.concat([df, df.iloc[[5]]])
    dirty = tmp_path / 'dirty.csv'
    df.to_csv(dirty, index=False)

    parts = run_preprocess(dirty, tmp_path, gap=0)
    assert sum(len(p) for p in parts) == 599
