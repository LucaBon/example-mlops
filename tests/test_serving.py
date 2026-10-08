import pytest

from components.preprocess import preprocess
from components.train import train
from serving.drift import detect_drift
from serving.predict import build_payload, model_feature_names, parse_response, prediction_url
from tests.conftest import make_dataset


def one_cycle(n_windows):
    return make_dataset(n_cycles=1, windows_per_cycle=n_windows, class_probs=(1, 0, 0))


def test_payload_contains_only_features():
    df = one_cycle(5)
    payload = build_payload(df)
    names = payload['data']['names']
    assert names == [f'feature_{i}' for i in range(8)] + ['constant']
    assert len(payload['data']['ndarray']) == 5
    assert len(payload['data']['ndarray'][0]) == len(names)


def test_payload_uses_model_features_in_order():
    df = one_cycle(3)
    payload = build_payload(df, ['feature_2', 'feature_0'])
    assert payload['data']['names'] == ['feature_2', 'feature_0']
    assert payload['data']['ndarray'][0] == df.loc[0, ['feature_2', 'feature_0']].tolist()


def test_payload_rejects_missing_model_features():
    with pytest.raises(ValueError, match='feature_9'):
        build_payload(one_cycle(3), ['feature_0', 'feature_9'])


def test_model_feature_names_excludes_dropped_columns(mlflow_store, raw_csv, tmp_path):
    train_path, test_path = (tmp_path / f'{name}.csv' for name in ('train', 'test'))
    preprocess(str(raw_csv), str(train_path), str(test_path))
    out = train(str(train_path), n_estimators=5, cv_repeats=1)

    names = model_feature_names(f'runs:/{out.run_id}/model')
    assert names == [f'feature_{i}' for i in range(8)]
    # 'constant' is in the raw data but was dropped by preprocess; meta columns
    # such as cycle_id or the machine model are never features
    payload = build_payload(one_cycle(2), names)
    assert 'constant' not in payload['data']['names']


def test_parse_response_formats():
    assert parse_response({'data': {'names': [], 'ndarray': [1, 0]}}) == [1, 0]
    assert parse_response({'data': {'tensor': {'shape': [2], 'values': [2, 1]}}}) == [2, 1]


def test_prediction_url():
    assert (prediction_url('http://gw/', 'ns', 'wm')
            == 'http://gw/seldon/ns/wm/api/v1.0/predictions')


def test_no_drift_on_same_distribution():
    report = detect_drift(make_dataset(seed=0), make_dataset(seed=1))
    assert not report['dataset_drift']


def test_drift_on_shifted_features():
    report = detect_drift(make_dataset(seed=0), make_dataset(seed=1, shift=2.0))
    assert report['dataset_drift']
    assert 'feature_0' in report['drifted_features']
