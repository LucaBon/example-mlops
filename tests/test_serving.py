from serving.drift import detect_drift
from serving.predict import build_payload, parse_response, prediction_url
from tests.conftest import make_dataset


def test_payload_contains_only_features():
    df = make_dataset(n_rows=5)
    payload = build_payload(df)
    names = payload['data']['names']
    assert 'target' not in names and 'DateTime' not in names
    assert len(payload['data']['ndarray']) == 5
    assert len(payload['data']['ndarray'][0]) == len(names)


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
