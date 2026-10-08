import json

import pytest
from mlflow.tracking import MlflowClient

from components.evaluate import evaluate
from components.preprocess import preprocess
from components.promote import promote
from components.train import train


@pytest.fixture
def splits(raw_csv, tmp_path):
    train_path, test_path = tmp_path / 'train.csv', tmp_path / 'test.csv'
    preprocess(str(raw_csv), str(train_path), str(test_path), test_size=0.25)
    return str(train_path), str(test_path)


def production_versions():
    return [str(v.version) for v in
            MlflowClient().get_latest_versions('WashingMachineModel', ['Production'])]


def run_cycle(splits, tmp_path, min_accuracy, tag):
    train_path, test_path = splits
    out = train(train_path, n_estimators=20, max_depth=0, random_state=0)
    metrics_path = tmp_path / f'metrics_{tag}.json'
    ui_metrics_path = tmp_path / f'ui_metrics_{tag}.json'
    (decision,) = evaluate(test_path, out.run_id, out.model_version,
                           str(metrics_path), str(ui_metrics_path),
                           min_accuracy=min_accuracy)
    return out, decision, json.loads(metrics_path.read_text())


def test_train_registers_model_version(mlflow_store, splits):
    out = train(splits[0], n_estimators=10, max_depth=3, random_state=0)
    assert out.model_version == '1'
    assert out.model_uri.endswith('/model')
    run = MlflowClient().get_run(out.run_id)
    assert run.data.params['n_estimators'] == '10'
    assert 'importance_feature_0' in run.data.metrics


def test_evaluate_approves_good_model_without_promoting(mlflow_store, splits, tmp_path):
    _, decision, metrics = run_cycle(splits, tmp_path, 0.8, 'good')
    assert decision == 'deploy'
    assert metrics['candidate']['test_accuracy'] >= 0.8
    assert metrics['champion'] is None
    # Promotion is left to the post-deploy step
    assert production_versions() == []
    ui = json.loads((tmp_path / 'ui_metrics_good.json').read_text())
    assert {m['name'] for m in ui['metrics']} == {'test-accuracy', 'test-f1-macro'}


def test_evaluate_skips_below_threshold(mlflow_store, splits, tmp_path):
    _, decision, _ = run_cycle(splits, tmp_path, 1.01, 'bad')
    assert decision == 'skip'


def test_promote_archives_previous_production(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 0.8, 'first')
    promote(first.model_version)
    assert production_versions() == [first.model_version]

    second, decision, metrics = run_cycle(splits, tmp_path, 0.8, 'second')
    assert metrics['champion_version'] == first.model_version
    # Same data and seed: candidate ties the champion, which is enough to deploy
    assert decision == 'deploy'
    promote(second.model_version)
    assert production_versions() == [second.model_version]
    archived = MlflowClient().get_model_version('WashingMachineModel', first.model_version)
    assert archived.current_stage == 'Archived'
