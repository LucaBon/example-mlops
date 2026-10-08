# Washing machine MLOps pipeline

Kubeflow pipeline (kfp v1 SDK) that trains a washing-machine cycle classifier
on windowed sensor signals and serves it with Seldon Core.

```
load_data ─► preprocess ─► train ─► evaluate ─► [decision == deploy] ─► deploy ─► promote
  (DVC)       clean +        RandomForest:  test-set gates +         SeldonDeployment  version ->
              chronological  val metrics,   champion comparison      (MLFLOW_SERVER)   Production
              train/val/test refit on
              split with gap train+val,
                             registered
```

### Evaluation

- `preprocess` sorts rows by time and splits them into train, validation and test
  (`val_size`, `test_size`). It drops `gap` rows before validation and before test so
  that overlapping windows never sit on both sides of a boundary. Set `gap` to at
  least the window overlap in rows.
- `train` fits on train and logs `val_accuracy` / `val_f1_macro`. Compare runs on
  these when tuning hyperparameters, never on the test metrics. The registered model
  is then refit on train + validation.
- `evaluate` uses the test split once. It returns `deploy` only when the candidate
  reaches `min_f1_macro`, recalls every test class at least `min_class_recall`, is
  more accurate than always predicting the majority class, and beats the Production
  model. Beating the Production model means a macro-F1 higher by more than
  `min_improvement` and higher in at least `min_prob_better` of the block-bootstrap
  resamples of the test set (blocks of `gap` rows). A tie never redeploys.
  Per-class recall, a bootstrap CI for macro-F1 and the confusion matrix are logged
  to MLflow. The confusion matrix also appears in the KFP UI.

The registry's Production stage only changes after Seldon reports the deployment
`Available`, so it always matches the model being served.

| Path | What |
| --- | --- |
| `pipeline.py` | Pipeline definition and CLI (`compile`, `run`, `schedule`) |
| `components/` | Lightweight components (`load_data`, `preprocess`, `train`, `evaluate`, `promote`) and the `deploy` container component |
| `serving/predict.py` | Client for the deployed model (Seldon v1 protocol) |
| `serving/drift.py` | Feature drift check (KS test + PSI) between reference and recent data |
| `k8s/` | RBAC so pipeline steps can create SeldonDeployments |
| `admin-integrations.sh` | One-off setup of MLflow/MinIO PodDefaults, Seldon secret and RBAC in the user namespace |

## Cluster prerequisites

Kubeflow with MLflow (`mlflow-server.kubeflow:5000`), MinIO and Seldon Core v1.

1. Set `USER_NAMESPACE` in `admin-integrations.sh` and run it. To let the cluster pull
   the private deploy image, export `GHCR_USER` (your GitHub user) and `GHCR_TOKEN`
   (a token with only the `read:packages` scope) first. The script then creates the
   `ghcr-pull-secret` that every pipeline pod uses.
2. Only if the pipeline reads data from AWS (`dvc_remote=s3`): create the `aws-secret`
   (keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) in the user namespace. With
   the default `dvc_remote=minio`, `load_data` uses the existing
   `mlpipeline-minio-artifact` secret.

## Data storage

The dataset is versioned with DVC. `.dvc/config` defines three remotes:

| Remote | Where | Used by |
| --- | --- | --- |
| `local` (default) | `.dvc-storage/` in the repo, gitignored | development on your machine |
| `minio` | `s3://dvc/dvc_remote` on the in-cluster MinIO | the pipeline (default `dvc_remote`) |
| `s3` | `s3://mlops-remote-storage/dvc_remote` on AWS, `eu-south-1` | the pipeline with `dvc_remote=s3` |

Credentials never go in the repo. For `s3`, DVC uses your AWS CLI profile or the
`AWS_*` variables. For `minio`, put them in the git-ignored `.dvc/config.local`:

```
uv tool install "dvc[s3]==3.51.2" --with pathspec==0.12.1   # or pipx; clashes with requirements.txt pins
dvc remote modify --local local url /path/to/storage           # optional: another folder
kubectl port-forward -n kubeflow svc/minio 9000:9000            # to reach MinIO from a laptop
dvc remote modify --local minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id <accesskey>
dvc remote modify --local minio secret_access_key <secretkey>
```

The MinIO keys are in the `mlpipeline-minio-artifact` secret. The `dvc` bucket must
exist before the first push.

To publish a new version of the data:

```
cp /path/to/data/signal_cycles_train_win_60_data.csv .
dvc add signal_cycles_train_win_60_data.csv
dvc push                     # local remote
dvc push -r minio            # what the pipeline reads by default
git add signal_cycles_train_win_60_data.csv.dvc && git commit -m "data: ..." && git push
```

Pipeline pods clone `repo_url` and read the `.dvc` file from its default branch, so the
`.dvc` change must be pushed. The `local` remote can't be reached from the cluster.

## Build the deploy component image

The image is published to GitHub Container Registry as
`ghcr.io/lucabon/washing-machine-deploy-model`:

```
gh auth refresh -s write:packages                       # once
gh auth token | docker login ghcr.io -u <github-user> --password-stdin
cd components/deploy && ./build_image.sh
```

The package is private; pipeline pods pull it with `ghcr-pull-secret` (see
cluster prerequisites). To use another registry, change `REPO` in
`build_image.sh` and the image in `components/deploy/component.yaml`.

## Compile, run, schedule

```
pip install -r requirements.txt
python pipeline.py compile                      # -> generated/washing_machine-pipeline.yaml
python pipeline.py run --host <KFP_URL> --arg min_f1_macro=0.9 --arg n_estimators=500
python pipeline.py schedule --host <KFP_URL> --cron "0 0 3 * * 1"   # weekly retraining
```

You can also upload `generated/washing_machine-pipeline.yaml` through the UI. Its parameters are
`repo_url`, `filename`, `dvc_remote` (`minio` or `s3`), `val_size`, `test_size`, `gap`, `n_estimators`, `max_depth`
(0 = unlimited), `random_state`, `min_f1_macro`, `min_class_recall`, `min_improvement`,
`min_prob_better`, `namespace` and `deployment_name`.

## Use the model

```
python -m serving.predict data.csv --host http://<istio-ingress> --namespace kubeflow-user-example-com --rows 5 \
    --model_uri models:/WashingMachineModel/Production
python -m serving.drift reference_train.csv recent_inputs.csv   # exit code 1 on drift
```

With `--model_uri` (needs `MLFLOW_TRACKING_URI` and artifact store credentials), the client
sends exactly the model's input columns in training order. Without it, it sends every
non-metadata column of the CSV.

Add `logger: {mode: all}` to the predictor in
`components/deploy/templates/deploy-manifest.j2` to collect live requests for drift checks.

## Development

```
pip install -r requirements.txt -r requirements-dev.txt
ruff check . && pytest
```

The tests run every component locally against a SQLite MLflow store, so no cluster is needed.
