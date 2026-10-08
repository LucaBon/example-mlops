import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from components.evaluate import evaluate
from components.preprocess import preprocess
from components.promote import promote
from components.train import train
from tests.conftest import make_dataset


class SpyForest(RandomForestClassifier):
    """Records the rows each forest is fit on and predicts."""
    calls = []

    def fit(self, X, y, sample_weight=None):
        SpyForest.calls.append(('fit', set(X.index)))
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        SpyForest.calls.append(('predict', set(X.index)))
        return super().predict(X)


def split(raw_csv, tmp_path):
    paths = [str(tmp_path / f'{name}.csv') for name in ('train', 'test')]
    preprocess(str(raw_csv), *paths)
    return paths


@pytest.fixture
def splits(raw_csv, tmp_path):
    return split(raw_csv, tmp_path)


def production_versions():
    return [str(v.version) for v in
            MlflowClient().get_latest_versions('WashingMachineModel', ['Production'])]


def run_cycle(splits, tmp_path, tag, **gates):
    train_path, test_path = splits
    out = train(train_path, n_estimators=20, max_depth=0, random_state=0, cv_repeats=1)
    metrics_path = tmp_path / f'metrics_{tag}.json'
    ui_metrics_path = tmp_path / f'ui_metrics_{tag}.json'
    ui_metadata_path = tmp_path / f'ui_metadata_{tag}.json'
    (decision,) = evaluate(test_path, out.run_id, out.model_version,
                           str(metrics_path), str(ui_metrics_path),
                           str(ui_metadata_path), n_bootstrap=200, **gates)
    return out, decision, json.loads(metrics_path.read_text())


def test_train_logs_cycle_cv_and_registers_model(mlflow_store, splits):
    out = train(splits[0], n_estimators=10, max_depth=3, random_state=0,
                cv_folds=5, cv_repeats=2, class_weight='balanced')
    assert out.model_version == '1'
    assert out.model_uri.endswith('/model')
    run = MlflowClient().get_run(out.run_id)
    assert run.data.params['n_estimators'] == '10'
    assert run.data.params['class_weight'] == 'balanced'
    assert run.data.params['n_train_cycles'] == '31'
    metrics = run.data.metrics
    assert metrics['cv_f1_macro_mean'] > 0.8
    assert metrics['cv_f1_macro_min'] <= metrics['cv_f1_macro_mean']
    assert {'cv_f1_macro_std', 'cv_accuracy_mean', 'cv_recall_Working_mean',
            'cv_recall_Heating_mean', 'cv_recall_Motor_mean',
            'importance_feature_0'} <= set(metrics)
    # Meta columns are never model features
    assert not any(k.startswith('importance_') and k[11:] in
                   ('cycle_id', 'brand', 'model', 'TIMESTAMP') for k in metrics)
    assert 'val_f1_macro' not in metrics
    train_cycles = json.loads(Path(MlflowClient().download_artifacts(
        out.run_id, 'train_cycles.json', str(mlflow_store))).read_text())
    assert train_cycles == sorted(pd.read_csv(splits[0])['cycle_id'].unique())


def test_cv_folds_hold_out_whole_cycles_of_every_class(mlflow_store, splits, monkeypatch):
    monkeypatch.setattr('sklearn.ensemble.RandomForestClassifier', SpyForest)
    SpyForest.calls = []
    train(splits[0], n_estimators=5, cv_folds=5, cv_repeats=2)

    df = pd.read_csv(splits[0], dtype={'cycle_id': str})
    fits = [rows for kind, rows in SpyForest.calls if kind == 'fit'][:-1]
    held = [rows for kind, rows in SpyForest.calls if kind == 'predict'][:len(fits)]
    assert len(fits) == 10
    for fit_rows, held_rows in zip(fits, held):
        fit_cycles = set(df.loc[sorted(fit_rows), 'cycle_id'])
        held_part = df.loc[sorted(held_rows)]
        assert not fit_cycles & set(held_part['cycle_id'])
        # Every class has at least 5 train cycles
        assert set(held_part['target']) == {'Working', 'Heating', 'Motor'}
    # Each repeat holds out every train cycle once
    assert set().union(*held[:5]) == set(df.index)


def test_train_clips_folds_to_smallest_class(mlflow_store, splits, capsys):
    out = train(splits[0], n_estimators=5, cv_folds=20, cv_repeats=1)
    # 8 Heating and Motor cycles in train
    assert MlflowClient().get_run(out.run_id).data.params['cv_folds'] == '8'
    assert 'clipped' in capsys.readouterr().out


def test_evaluate_approves_good_model_without_promoting(mlflow_store, splits, tmp_path):
    _, decision, metrics = run_cycle(splits, tmp_path, 'good')
    assert decision == 'deploy'
    assert set(metrics['checks']) == {'cv_min_f1_macro', 'cv_min_class_recall',
                                      'beats_majority_baseline', 'beats_champion'}
    assert all(metrics['checks'].values())
    candidate = metrics['candidate']
    assert metrics['n_test_cycles'] == 9
    assert candidate['test_cycle_f1_macro'] >= 0.8
    assert 'test_window_f1_macro' in candidate
    assert candidate['test_cycle_f1_macro_ci_low'] <= candidate['test_cycle_f1_macro'] \
        <= candidate['test_cycle_f1_macro_ci_high']
    assert metrics['champion'] is None
    # Promotion is left to the post-deploy step
    assert production_versions() == []
    ui = json.loads((tmp_path / 'ui_metrics_good.json').read_text())
    assert {m['name'] for m in ui['metrics']} == {
        'cv-f1-macro-mean', 'test-cycle-accuracy', 'test-cycle-f1-macro'}
    matrix, table = json.loads((tmp_path / 'ui_metadata_good.json').read_text())['outputs']
    assert matrix['type'] == 'confusion_matrix'
    assert matrix['labels'] == ['Heating', 'Motor', 'Working']
    # One count per cycle
    assert sum(int(line.split(',')[2]) for line in matrix['source'].splitlines()) == 9
    assert table['type'] == 'markdown'
    assert '| X9 |' in table['source']


def test_evaluate_reports_per_machine_model(mlflow_store, splits, tmp_path):
    out, _, metrics = run_cycle(splits, tmp_path, 'machines')
    assert set(metrics['per_machine']) == {'A1', 'B2', 'X9'}
    assert sum(m['n_cycles'] for m in metrics['per_machine'].values()) == 9
    # Motor cycles only come from X9: it is left out of the multi-class subset
    assert set(metrics['per_machine']['X9']['recall']) == {'Motor'}
    assert 'X9' not in metrics['multi_class_models']['models']
    logged = MlflowClient().get_run(out.run_id).data.metrics
    assert {'test_machine_X9_accuracy', 'test_machine_X9_recall_Motor',
            'test_multi_class_models_f1_macro'} <= set(logged)


def test_evaluate_skips_below_threshold(mlflow_store, splits, tmp_path):
    _, decision, metrics = run_cycle(splits, tmp_path, 'bad', min_f1_macro=1.01)
    assert decision == 'skip'
    assert metrics['checks']['cv_min_f1_macro'] is False


def test_evaluate_rejects_majority_class_model(mlflow_store, tmp_path):
    # Noise features, 80% of cycles in one class: the model mostly predicts it
    raw = tmp_path / 'imbalanced.csv'
    make_dataset(n_cycles=60, signal=0.0, class_probs=(0.8, 0.1, 0.1)).to_csv(raw)
    _, decision, metrics = run_cycle(split(raw, tmp_path), tmp_path, 'majority',
                                     min_f1_macro=0.0)
    assert metrics['checks']['cv_min_class_recall'] is False
    assert decision == 'skip'


def test_tie_with_champion_is_not_deployed(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    assert production_versions() == [first.model_version]

    second, decision, metrics = run_cycle(splits, tmp_path, 'second')
    assert metrics['champion_version'] == first.model_version
    # Same data and seed: identical model, so no reason to redeploy
    assert metrics['comparison']['n_cycles'] == 9
    assert metrics['comparison']['improvement'] == 0.0
    assert metrics['comparison']['prob_better'] == 0.0
    # The tie falls back to CV, where both have the same score
    assert metrics['comparison']['tie_decided_by_cv']
    assert metrics['checks']['beats_champion'] is False
    assert decision == 'skip'


def test_tie_with_champion_is_broken_by_better_cv(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    MlflowClient().log_metric(first.run_id, 'cv_f1_macro_mean', 0.1, step=1)

    _, decision, metrics = run_cycle(splits, tmp_path, 'second')
    assert metrics['champion']['cv_f1_macro_mean'] == 0.1
    assert metrics['comparison']['improvement'] == 0.0
    assert metrics['checks']['beats_champion'] is True
    assert decision == 'deploy'


def test_champion_without_train_cycles_is_not_compared(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    # As for a model registered before train_cycles.json was logged
    artifacts = MlflowClient().get_run(first.run_id).info.artifact_uri
    (Path(artifacts.replace('file://', '')) / 'train_cycles.json').unlink()

    _, decision, metrics = run_cycle(splits, tmp_path, 'second')
    assert metrics['champion_version'] == first.model_version
    assert metrics['comparison'] is None
    assert 'train_cycles.json' in metrics['champion_note']
    assert metrics['checks']['beats_champion'] is True
    assert decision == 'deploy'


def test_promote_archives_previous_production(mlflow_store, splits, tmp_path):
    first, _, _ = run_cycle(splits, tmp_path, 'first')
    promote(first.model_version)
    second, _, _ = run_cycle(splits, tmp_path, 'second')
    promote(second.model_version)
    assert production_versions() == [second.model_version]
    archived = MlflowClient().get_model_version('WashingMachineModel', first.model_version)
    assert archived.current_stage == 'Archived'
