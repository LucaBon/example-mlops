from typing import NamedTuple

from kfp.components import InputPath, OutputPath


def evaluate(test_path: InputPath('CSV'),
             run_id: str,
             model_version: str,
             metrics_path: OutputPath('JSON'),
             mlpipeline_metrics_path: OutputPath('Metrics'),
             min_accuracy: float = 0.8,
             registered_model_name: str = 'WashingMachineModel',
             ) -> NamedTuple('Outputs', [('decision', str)]):
    """Score the candidate model on the test split and gate deployment.

    Returns ``deploy`` when the candidate reaches ``min_accuracy`` and its
    macro-F1 is not worse than the current Production model; ``skip``
    otherwise. Promotion happens in ``promote``, after a successful deploy.
    """
    import json
    from collections import namedtuple

    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    time_col_name = 'TIMESTAMP'
    datetime_col_name = 'DateTime'
    target_col_name = 'target'

    df = pd.read_csv(test_path)
    x = df.drop(columns=[target_col_name, time_col_name, datetime_col_name])
    y = df[target_col_name]

    def score(model_uri):
        y_pred = mlflow.pyfunc.load_model(model_uri).predict(x)
        return {
            'test_accuracy': float(accuracy_score(y, y_pred)),
            'test_f1_macro': float(f1_score(y, y_pred, average='macro')),
        }, classification_report(y, y_pred, zero_division=0)

    candidate, report = score(f'runs:/{run_id}/model')
    print(report)

    client = MlflowClient()
    production = [v for v in client.get_latest_versions(
        registered_model_name, stages=['Production'])
        if str(v.version) != str(model_version)]
    if production:
        champion, _ = score(f'models:/{registered_model_name}/Production')
        champion_version = str(production[0].version)
    else:
        champion, champion_version = None, None

    meets_threshold = candidate['test_accuracy'] >= min_accuracy
    beats_champion = (champion is None
                      or candidate['test_f1_macro'] >= champion['test_f1_macro'])
    decision = 'deploy' if meets_threshold and beats_champion else 'skip'

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(candidate)
        mlflow.log_text(report, 'classification_report.txt')
        if champion:
            mlflow.log_metrics({f'champion_{k}': v for k, v in champion.items()})
        mlflow.set_tag('evaluation_decision', decision)

    summary = {'candidate_version': model_version, 'candidate': candidate,
               'champion_version': champion_version, 'champion': champion,
               'min_accuracy': min_accuracy, 'decision': decision}
    print(json.dumps(summary, indent=2))
    with open(metrics_path, 'w') as f:
        json.dump(summary, f)
    with open(mlpipeline_metrics_path, 'w') as f:
        json.dump({'metrics': [
            {'name': name.replace('_', '-'), 'numberValue': value,
             'format': 'RAW'} for name, value in candidate.items()]}, f)

    outputs = namedtuple('Outputs', ['decision'])
    return outputs(decision)
