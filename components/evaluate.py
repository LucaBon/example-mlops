from typing import NamedTuple

from kfp.components import InputPath, OutputPath


def evaluate(test_path: InputPath('CSV'),
             run_id: str,
             model_version: str,
             metrics_path: OutputPath('JSON'),
             mlpipeline_metrics_path: OutputPath('Metrics'),
             mlpipeline_ui_metadata_path: OutputPath(),
             min_f1_macro: float = 0.8,
             min_class_recall: float = 0.5,
             min_improvement: float = 0.0,
             min_prob_better: float = 0.9,
             block_size: int = 60,
             n_bootstrap: int = 1000,
             registered_model_name: str = 'WashingMachineModel',
             ) -> NamedTuple('Outputs', [('decision', str)]):
    """Score the candidate model on the test split and gate deployment.

    Returns ``deploy`` when the candidate
      * reaches ``min_f1_macro``,
      * recalls every test class at least ``min_class_recall``,
      * is more accurate than always predicting the majority class, and
      * beats the current Production model: macro-F1 higher by more than
        ``min_improvement`` and higher in at least ``min_prob_better`` of the
        block-bootstrap resamples of the test set,
    and ``skip`` otherwise. Test rows are autocorrelated, so the bootstrap
    resamples contiguous blocks of ``block_size`` rows. Promotion happens in
    ``promote``, after a successful deploy.
    """
    import json
    import re
    from collections import namedtuple

    import mlflow
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    time_col_name = 'TIMESTAMP'
    datetime_col_name = 'DateTime'
    target_col_name = 'target'

    df = pd.read_csv(test_path)
    x = df.drop(columns=[target_col_name, time_col_name, datetime_col_name])
    y = df[target_col_name].to_numpy()
    test_classes = np.unique(y)

    def predict(model_uri):
        return np.asarray(mlflow.pyfunc.load_model(model_uri).predict(x))

    def f1_macro(y_true, y_pred):
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    candidate_pred = predict(f'runs:/{run_id}/model')
    report = classification_report(y, candidate_pred, zero_division=0)
    print(report)
    class_recall = dict(zip(test_classes, recall_score(
        y, candidate_pred, labels=test_classes, average=None, zero_division=0)))
    candidate = {
        'test_accuracy': float(accuracy_score(y, candidate_pred)),
        'test_f1_macro': f1_macro(y, candidate_pred),
        'test_min_class_recall': float(min(class_recall.values())),
    }
    baseline_accuracy = float(pd.Series(y).value_counts(normalize=True).iloc[0])

    # Moving-block bootstrap: the same resampled rows score both models
    rng = np.random.default_rng(0)
    n_rows = len(y)
    block = max(1, min(block_size, n_rows))
    n_blocks = -(-n_rows // block)
    offsets = np.arange(block)
    resamples = [
        (rng.integers(0, n_rows - block + 1, n_blocks)[:, None] + offsets)
        .ravel()[:n_rows] for _ in range(n_bootstrap)]
    candidate_boot = np.array([f1_macro(y[i], candidate_pred[i]) for i in resamples])
    candidate['test_f1_macro_ci_low'] = float(np.quantile(candidate_boot, 0.05))
    candidate['test_f1_macro_ci_high'] = float(np.quantile(candidate_boot, 0.95))

    client = MlflowClient()
    production = [v for v in client.get_latest_versions(
        registered_model_name, stages=['Production'])
        if str(v.version) != str(model_version)]
    champion, champion_version, comparison = None, None, None
    if production:
        champion_pred = predict(f'models:/{registered_model_name}/Production')
        champion_version = str(production[0].version)
        champion = {
            'test_accuracy': float(accuracy_score(y, champion_pred)),
            'test_f1_macro': f1_macro(y, champion_pred),
        }
        champion_boot = np.array([f1_macro(y[i], champion_pred[i]) for i in resamples])
        comparison = {
            'improvement': candidate['test_f1_macro'] - champion['test_f1_macro'],
            'prob_better': float(np.mean(candidate_boot > champion_boot)),
        }

    checks = {
        'min_f1_macro': candidate['test_f1_macro'] >= min_f1_macro,
        'min_class_recall': candidate['test_min_class_recall'] >= min_class_recall,
        'beats_majority_baseline': candidate['test_accuracy'] > baseline_accuracy,
        'beats_champion': (comparison is None
                           or (comparison['improvement'] > min_improvement
                               and comparison['prob_better'] >= min_prob_better)),
    }
    decision = 'deploy' if all(checks.values()) else 'skip'

    labels = [str(c) for c in sorted(set(y) | set(candidate_pred))]
    matrix = confusion_matrix(y.astype(str), candidate_pred.astype(str), labels=labels)
    matrix_csv = '\n'.join(f'{t},{p},{matrix[i][j]}'
                           for i, t in enumerate(labels)
                           for j, p in enumerate(labels))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(candidate)
        mlflow.log_metric('baseline_test_accuracy', baseline_accuracy)
        mlflow.log_metrics({
            'test_recall_' + re.sub(r'[^\w.\- /]', '_', str(c)): float(r)
            for c, r in class_recall.items()})
        mlflow.log_text(report, 'classification_report.txt')
        mlflow.log_text('target,predicted,count\n' + matrix_csv,
                        'confusion_matrix.csv')
        if champion:
            mlflow.log_metrics({f'champion_{k}': v for k, v in champion.items()})
            mlflow.log_metrics({f'champion_{k}': v for k, v in comparison.items()})
        mlflow.set_tag('evaluation_decision', decision)

    summary = {'candidate_version': model_version, 'candidate': candidate,
               'class_recall': {str(c): float(r) for c, r in class_recall.items()},
               'baseline_accuracy': baseline_accuracy,
               'champion_version': champion_version, 'champion': champion,
               'comparison': comparison, 'checks': checks, 'decision': decision}
    print(json.dumps(summary, indent=2))
    with open(metrics_path, 'w') as f:
        json.dump(summary, f)
    with open(mlpipeline_metrics_path, 'w') as f:
        json.dump({'metrics': [
            {'name': name.replace('_', '-'), 'numberValue': candidate[name],
             'format': 'RAW'}
            for name in ('test_accuracy', 'test_f1_macro', 'test_min_class_recall')]}, f)
    with open(mlpipeline_ui_metadata_path, 'w') as f:
        json.dump({'outputs': [{
            'type': 'confusion_matrix', 'format': 'csv', 'storage': 'inline',
            'schema': [{'name': 'target', 'type': 'CATEGORY'},
                       {'name': 'predicted', 'type': 'CATEGORY'},
                       {'name': 'count', 'type': 'NUMBER'}],
            'source': matrix_csv, 'labels': labels}]}, f)

    outputs = namedtuple('Outputs', ['decision'])
    return outputs(decision)
