from typing import NamedTuple

from kfp.components import InputPath


def train(train_path: InputPath('CSV'),
          n_estimators: int = 300,
          max_depth: int = 0,
          random_state: int = 42,
          registered_model_name: str = 'WashingMachineModel',
          ) -> NamedTuple('Outputs', [('run_id', str),
                                      ('model_uri', str),
                                      ('model_version', str)]):
    """Train a RandomForest on the training split and register it in MLflow.

    ``max_depth=0`` means no depth limit.
    """
    import os
    from collections import namedtuple

    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier

    time_col_name = 'TIMESTAMP'
    datetime_col_name = 'DateTime'
    target_col_name = 'target'

    # MinIO does not auto-create the artifact bucket; skip for non-S3 stores
    s3_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    if s3_endpoint:
        import boto3
        object_storage = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            config=boto3.session.Config(signature_version='s3v4'),
        )
        default_bucket_name = 'mlflow'
        buckets = object_storage.list_buckets()['Buckets']
        if not any(b['Name'] == default_bucket_name for b in buckets):
            object_storage.create_bucket(Bucket=default_bucket_name)

    df = pd.read_csv(train_path)
    x = df.drop(columns=[target_col_name, time_col_name, datetime_col_name])
    y = df[target_col_name]

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth or None,
        'random_state': random_state,
    }

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_jobs=-1, **params)
        model.fit(x, y)

        mlflow.log_params({**params, 'n_features': x.shape[1],
                           'n_train_rows': x.shape[0]})
        mlflow.log_metric('training_accuracy', model.score(x, y))
        mlflow.log_metrics({f'importance_{col}': imp for col, imp
                            in zip(x.columns, model.feature_importances_)})

        mlflow.sklearn.log_model(
            model, 'model',
            registered_model_name=registered_model_name,
            signature=infer_signature(x, model.predict(x.head(100))))

        run_id = run.info.run_id
        model_uri = f'{mlflow.get_artifact_uri()}/model'

    versions = MlflowClient().search_model_versions(f"run_id='{run_id}'")
    model_version = str(max(int(v.version) for v in versions))
    print(f'Run {run_id} registered {registered_model_name} '
          f'v{model_version} at {model_uri}')

    outputs = namedtuple('Outputs', ['run_id', 'model_uri', 'model_version'])
    return outputs(run_id, model_uri, model_version)
