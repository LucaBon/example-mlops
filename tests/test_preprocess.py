import pandas as pd

from components.preprocess import preprocess


def test_preprocess_cleans_and_splits_chronologically(raw_csv, tmp_path):
    train_path, test_path = tmp_path / 'train.csv', tmp_path / 'test.csv'
    preprocess(str(raw_csv), str(train_path), str(test_path), test_size=0.25)

    train = pd.read_csv(train_path, parse_dates=['DateTime'])
    test = pd.read_csv(test_path, parse_dates=['DateTime'])

    for df in (train, test):
        assert not any(c.startswith('Unnamed:') for c in df.columns)
        assert 'constant' not in df.columns
        assert {'TIMESTAMP', 'DateTime', 'target'} <= set(df.columns)
    assert len(train) == 450 and len(test) == 150
    assert train['DateTime'].max() < test['DateTime'].min()


def test_preprocess_drops_missing_targets_and_duplicates(raw_csv, tmp_path):
    df = pd.read_csv(raw_csv)
    df.loc[0, 'target'] = None
    df = pd.concat([df, df.iloc[[5]]])
    dirty = tmp_path / 'dirty.csv'
    df.to_csv(dirty, index=False)

    train_path, test_path = tmp_path / 'train.csv', tmp_path / 'test.csv'
    preprocess(str(dirty), str(train_path), str(test_path), test_size=0.25)

    total = len(pd.read_csv(train_path)) + len(pd.read_csv(test_path))
    assert total == 599
