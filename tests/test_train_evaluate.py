import json

import pytest
from mlflow.tracking import MlflowClient

from components.evaluate import evaluate
from components.preprocess import preprocess
from components.promote import promote
from components.train import train
from tests.conftest import make_dataset


def split(raw_csv, tmp_path):
    paths = [str(tmp_path / f'{name}.csv') for name in ('train', 'val', 'test')]
    preprocess(str(raw_csv), *paths, val_size=0.2, test_size=0.25, gap=10)
    return paths


@pytest.fixture
def splits(raw_csv, tmp_path):
    return split(raw_csv, tmp_path)


def production_versions():
    return [str(v.version) for v in
            MlflowClient().get_latest_versions('WashingMachineModel', ['Production'])]


def run_cycle(splits, tmp_path, tag, **gates):
    train_path, val_path, test_path = splits
    out = train(train_path, val_path, n_estimators=20, max_depth=0, random_state=0)
    metrics_path = tmp_path / f'metrics_{tag}.json'
    ui_metrics_path = tmp_path / f'ui_metrics_{tag}.json'
    ui_metadata_path = tmp_path / f'ui_metadata_{tag}.json'
    (decision,) = evaluate(test_path, out.run_id, out.model_version,
                           str(metrics_path), str(ui_metrics_path),
                           str(ui_metadata_path), n_bootstrap=200, block_size=10,
                           **gates)
    return out, decision, json.loads(metrics_path.read_text())


def test_train_registers_model_version(mlflow_store, splits):
    out = train(splits[0], splits[1], n_estimators=10, max_depth=3, random_state=0)
    assert out.model_version == '1'
    assert out.model_uri.endswith('/model')
    run = MlflowClient().get_run(out.run_id)
    assert run.data.params['n_estimators'] == '10'
    assert run.data.params['n_val_rows'] == '110'
    assert 'importance_feature_0' in run.data.metrics
    assert run.data.metrics['val_f1_macro'] > 0.8
    assert 'training_accuracy' not in run.data.metrics


def test_evaluate_approves_good_model_without_promoting(mlflow_store, splits, tmp_path):
    _, decision, metrics = run_cycle(splits, tmp_path, 'good')
    assert decision == 'deploy'
    assert all(metrics['checks'].values())
    candidate = metrics['candidate']
    assert candidate['test_f1_macro'] >= 0.8
    assert candidate['test_f1_macro_ci_low'] <= candidate['test_f1_macro'] \
        <= candidate['test_f1_macro_ci_high']
    assert metrics['champion'] is None
    # Promotion is left to the post-deploy step
    assert production_versions() == []
    ui = json.loads((tmp_path / 'ui_metrics_good.json').read_text())
    assert {m['name'] for m in ui['metrics']} == {
        'test-accuracy', 'test-f1-macro', 'test-min-class-recall'}
    ui_metadata = json.loads((tmp_path / 'ui_metadata_good.json').read_text())
    matrix = ui_metadata['outputs'][0]
    assert matrix['type'] == 'confusion_matrix'
    assert matrix['labels'] == ['0', '1', '2']
    assert len(matrix['source'].splitlines()) == 9


def test_evaluate_skips_below_threshold(mlflow_store, splits, tmp_path):
    _, decision, metrics = run_cycle(splits, tmp_path, 'bad', min_f1_macro=1.01)
    assert decision == 'skip'
    assert metrics['checks']['min_f1_macro'] is False


def test_evaluate_rejects_majority_class_model(mlflow_store, tmp_path):
    # Noise features, 90% one class: the model mostly predicts the majority
    raw = tmp_path / 'imbalanced.csv'
    make_dataset(n_rows=1000, signal=0.0, class_probs=[0.9, 0.05, 0.05]).to_csv(raw)
    _, decision, metrics = run_cycle(split(raw, tmp_path), tmp_path, 'majority',
                                     min_f1_macro=0.0)
    # A plain accuracy gate at 0.8 would have deployed it
    assert metrics['candidate']['test_accuracy'] >= 0.8
    assert metrics['checks']['min_class_recall'] is False
    assert decision == 'skip'


def test_tie_with_champion_is_not_deployed(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    assert production_versions() == [first.model_version]

    second, decision, metrics = run_cycle(splits, tmp_path, 'second')
    assert metrics['champion_version'] == first.model_version
    # Same data and seed: identical model, so no reason to redeploy
    assert metrics['comparison'] == {'improvement': 0.0, 'prob_better': 0.0}
    assert metrics['checks']['beats_champion'] is False
    assert decision == 'skip'


def test_promote_archives_previous_production(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    second, _, _ = run_cycle(splits, tmp_path, 'second')
    promote(second.model_version)
    assert production_versions() == [second.model_version]
    archived = MlflowClient().get_model_version('WashingMachineModel', first.model_version)
    assert archived.current_stage == 'Archived'
