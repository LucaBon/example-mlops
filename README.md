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
| `data_prep/build_dataset.py` | Builds the windowed training CSV from the raw SMART-PDM recordings |
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

## Dataset

The model is trained on the washing machine recordings of the SMART-PDM Appliance
Dataset: T. Fonseca, L.L. Ferreira, P. Chaves, B. Cabral, P. Costa, "SMART-PDM
Appliance Dataset", Zenodo, 2022,
[doi:10.5281/zenodo.7245198](https://doi.org/10.5281/zenodo.7245198), described in
T. Fonseca et al., "Dataset for identifying maintenance needs of home appliances using
artificial intelligence", Data in Brief 48 (2023) 109068,
[doi:10.1016/j.dib.2023.109068](https://doi.org/10.1016/j.dib.2023.109068). It is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The derived
`wm_cycles_win60.csv` falls under the same license: keep the attribution in
[DATA_LICENSE.md](DATA_LICENSE.md) wherever you share it or data built from it.

Each row is a 60 s window, taken every 10 s, of one washing cycle. `target` is the
cycle's label (`Working`, `Heating`, `Bearings` or `Motor`). `TIMESTAMP` (cycle start
from the metadata plus the window end in seconds), `DateTime`, `cycle_id`, `brand` and
`model` are metadata. All other columns are statistics and spectral band shares of the
2048 Hz current and vibration signals (`fast.csv`).

The 1 Hz power meter (`slow.csv`) is left out by default: its clock does not match
the fast recording, and in every Motor cycle it reads zero power, so a model would
learn a recording artifact. Cycles are skipped, with the reason printed, when they
have no label, less than one window of fast data, fast signals in volts instead of
ADC counts (three December 2021 cycles), or a `fast.csv` identical to another
cycle's. The current build keeps 83 of 96 cycles.

To rebuild it, download the dataset from the Zenodo record above, unpack it (about
22 GB) and run:

```
python -m data_prep.build_dataset /path/to/SMART-PDM-Dataset/2-washing_machines wm_cycles_win60.csv
```

Options: `--window` and `--stride` in seconds, `--workers` (default: CPUs - 1) and
`--with-slow` (add power meter features). The docstring of `data_prep/build_dataset.py`
explains how the two sensor streams are aligned. Then `dvc add` and `dvc push` the
file as shown below.

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
python -m data_prep.build_dataset /path/to/2-washing_machines wm_cycles_win60.csv
dvc add wm_cycles_win60.csv
dvc push                     # local remote
dvc push -r minio            # what the pipeline reads by default
git add wm_cycles_win60.csv.dvc && git commit -m "data: ..." && git push
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
