import pandas as pd
import pytest

from components.preprocess import preprocess
from tests.conftest import make_dataset

META = {'TIMESTAMP', 'DateTime', 'target', 'cycle_id', 'brand', 'model'}


def run_preprocess(raw_csv, tmp_path, **kwargs):
    paths = [tmp_path / f'{name}.csv' for name in ('train', 'test')]
    preprocess(str(raw_csv), *map(str, paths), **kwargs)
    return [pd.read_csv(p, dtype={'cycle_id': str}) for p in paths]


def cycles_by_class(df):
    return df.groupby('target')['cycle_id'].unique().apply(set).to_dict()


def test_preprocess_cleans_and_splits_by_cycle(raw_csv, tmp_path):
    train, test = run_preprocess(raw_csv, tmp_path)

    for df in (train, test):
        assert not any(c.startswith('Unnamed:') for c in df.columns)
        assert 'constant' not in df.columns
        assert META <= set(df.columns)
    assert not set(train['cycle_id']) & set(test['cycle_id'])
    # 20 / 10 / 10 cycles, 25% of each class (rounded) in test
    test_cycles = {c: len(v) for c, v in cycles_by_class(test).items()}
    assert test_cycles == {'Working': 5, 'Heating': 2, 'Motor': 2}
    assert len(train) + len(test) == 600


def test_every_class_with_two_cycles_is_in_test(tmp_path):
    # 2 Heating cycles and 1 Motor cycle: round(0.25 * 2) = 0, raised to 1
    raw = tmp_path / 'small.csv'
    make_dataset(n_cycles=23, class_probs=(20 / 23, 2 / 23, 1 / 23)).to_csv(raw)
    train, test = run_preprocess(raw, tmp_path)
    assert set(test['target']) == {'Working', 'Heating'}
    assert set(train['target']) == {'Working', 'Heating', 'Motor'}


def test_split_is_stable_when_cycles_are_added(tmp_path):
    old, new = tmp_path / 'old.csv', tmp_path / 'new.csv'
    base = make_dataset(n_cycles=40)
    base.to_csv(old, index=False)
    pd.concat([base, make_dataset(n_cycles=8, seed=3, first_cycle=40)]).to_csv(new, index=False)

    old_test = cycles_by_class(run_preprocess(old, tmp_path)[1])
    new_test = cycles_by_class(run_preprocess(new, tmp_path)[1])
    old_ids = set(base['cycle_id'])
    for cls, before in old_test.items():
        after = new_test[cls] & old_ids
        # Ranked by hash: the old test cycles of a class stay a prefix of the
        # ranking, only the boundary moves when the class quota changes
        assert before & after
        assert before <= after or after <= before


@pytest.mark.parametrize('column', ['cycle_id', 'TIMESTAMP', 'DateTime'])
def test_preprocess_requires_columns(raw_csv, tmp_path, column):
    df = pd.read_csv(raw_csv).drop(columns=[column])
    path = tmp_path / 'missing.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        run_preprocess(path, tmp_path)


@pytest.mark.parametrize('column, value', [('target', 'Motor'), ('model', 'Z0')])
def test_preprocess_rejects_mixed_cycles(raw_csv, tmp_path, column, value):
    df = pd.read_csv(raw_csv)
    first = df['cycle_id'] == 'cycle_0000'
    df.loc[df.index[first][0], column] = value
    path = tmp_path / 'mixed.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        run_preprocess(path, tmp_path)


def test_preprocess_drops_missing_targets_and_duplicates(raw_csv, tmp_path):
    df = pd.read_csv(raw_csv)
    df.loc[0, 'target'] = None
    df = pd.concat([df, df.iloc[[5]]])
    dirty = tmp_path / 'dirty.csv'
    df.to_csv(dirty, index=False)

    parts = run_preprocess(dirty, tmp_path)
    assert sum(len(p) for p in parts) == 599
