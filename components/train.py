from typing import NamedTuple

from kfp.components import InputPath


def train(train_path: InputPath('CSV'),
          n_estimators: int = 300,
          max_depth: int = 0,
          random_state: int = 42,
          cv_folds: int = 5,
          cv_repeats: int = 2,
          class_weight: str = '',
          registered_model_name: str = 'WashingMachineModel',
          ) -> NamedTuple('Outputs', [('run_id', str),
                                      ('model_uri', str),
                                      ('model_version', str)]):
    """Cross-validate a RandomForest by cycle, then fit it on all train and register it.

    Windows of one cycle are near-duplicates and a cycle has a single class,
    so the cross-validation is a repeated ``StratifiedKFold`` over cycles (all
    windows of a cycle land in the same fold) and is scored per cycle: the
    out-of-fold window predictions of a cycle are reduced to a majority vote.
    The ``cv_*`` metrics are the main quality signal; compare runs on them
    when tuning, never on test. ``max_depth=0`` means no depth limit,
    ``class_weight`` is ``''`` or ``'balanced'``.
    """
    import os
    import re
    from collections import namedtuple

    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    target_col_name = 'target'
    cycle_col_name = 'cycle_id'
    meta_cols = ['TIMESTAMP', 'DateTime', target_col_name, cycle_col_name,
                 'brand', 'model']

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

    df = pd.read_csv(train_path, dtype={cycle_col_name: str})
    x = df.drop(columns=[c for c in meta_cols if c in df.columns])
    y = df[target_col_name].astype(str)
    groups = df[cycle_col_name]

    def majority(s):
        # Most frequent label; ties go to the first label in sorted order
        counts = s.value_counts()
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def sanitize(name):
        return re.sub(r'[^\w.\- ]', '_', str(name))

    cycle_y = y.groupby(groups).agg(majority)
    classes = sorted(cycle_y.unique())

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth or None,
        'random_state': random_state,
        'class_weight': class_weight or None,
    }

    min_class_cycles = int(cycle_y.value_counts().min())
    n_splits = max(2, min(cv_folds, min_class_cycles))
    if n_splits != cv_folds:
        print(f'WARNING: cv_folds={cv_folds} clipped to {n_splits}: the smallest '
              f'class has {min_class_cycles} cycles in train')

    scores = []
    for r in range(cv_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        oof = pd.Series(index=y.index, dtype=object)
        for _, held_idx in cv.split(cycle_y.index.to_numpy(), cycle_y):
            held = groups.isin(cycle_y.index[held_idx])
            fold_model = RandomForestClassifier(n_jobs=-1, **params)
            fold_model.fit(x[~held], y[~held])
            oof[held] = fold_model.predict(x[held])
        true, pred = cycle_y, oof.groupby(groups).agg(majority)[cycle_y.index]
        recalls = recall_score(true, pred, labels=classes, average=None, zero_division=0)
        scores.append({
            'f1_macro': f1_score(true, pred, labels=classes, average='macro',
                                 zero_division=0),
            'accuracy': accuracy_score(true, pred),
            **{f'recall_{c}': rec for c, rec in zip(classes, recalls)},
        })
    scores = pd.DataFrame(scores)
    cv_metrics = {
        'cv_f1_macro_mean': scores['f1_macro'].mean(),
        'cv_f1_macro_std': scores['f1_macro'].std(ddof=0),
        'cv_f1_macro_min': scores['f1_macro'].min(),
        'cv_accuracy_mean': scores['accuracy'].mean(),
        **{f'cv_recall_{sanitize(c)}_mean': scores[f'recall_{c}'].mean() for c in classes},
    }
    cv_metrics = {k: float(v) for k, v in cv_metrics.items()}
    print(f'Cycle-level CV ({cv_repeats} x {n_splits} folds, {len(cycle_y)} cycles):')
    print(pd.Series(cv_metrics).to_string())

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_jobs=-1, **params)
        model.fit(x, y)

        mlflow.log_params({**params, 'n_features': x.shape[1],
                           'n_train_rows': x.shape[0],
                           'n_train_cycles': len(cycle_y),
                           'cv_folds': n_splits, 'cv_repeats': cv_repeats})
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics({f'importance_{sanitize(col)}': imp for col, imp
                            in zip(x.columns, model.feature_importances_)})
        # Lets evaluate score a later candidate against this model on unseen cycles only
        mlflow.log_dict(sorted(cycle_y.index), 'train_cycles.json')

        mlflow.sklearn.log_model(
            model, 'model',
            registered_model_name=registered_model_name,
            signature=infer_signature(x, np.asarray(model.predict(x.head(100)))))

        run_id = run.info.run_id
        model_uri = f'{mlflow.get_artifact_uri()}/model'

    versions = MlflowClient().search_model_versions(f"run_id='{run_id}'")
    model_version = str(max(int(v.version) for v in versions))
    print(f'Run {run_id} registered {registered_model_name} '
          f'v{model_version} at {model_uri}')

    outputs = namedtuple('Outputs', ['run_id', 'model_uri', 'model_version'])
    return outputs(run_id, model_uri, model_version)
