# Washing machine MLOps pipeline

Kubeflow pipeline (kfp v1 SDK) that trains a washing-machine cycle classifier
on windowed sensor signals and serves it with Seldon Core.

```
load_data ─► preprocess ─► train ─► evaluate ─► [decision == deploy] ─► deploy ─► promote
  (DVC)       clean +        grouped CV     CV gates, cycle-level    SeldonDeployment  version ->
              stratified     by cycle,      test vs majority and     (MLFLOW_SERVER)   Production
              train/test     fit on all     champion, per-machine
              split by       train cycles,  report
              cycle_id       registered
```

### Evaluation

The washing cycle is the unit of evaluation. Each cycle becomes many 60 s windows
with a 10 s stride, so windows of one cycle are near-duplicates: scoring windows
would count the same cycle dozens of times, and splitting windows would leak a cycle
into both train and test.

- `preprocess` splits by `cycle_id`, stratified by class. Within each class, cycles
  are ranked by the md5 of their `cycle_id` and the lowest `test_size` share goes to
  test (at least one cycle when the class has two or more). The ranking does not
  depend on row order, so reruns give the same split and new cycles only move the
  train/test boundary of their class. `cycle_id`, `brand` and `model` are kept as
  metadata and are never model features.
- `train` runs `cv_repeats` x `cv_folds` stratified cross-validation over the train
  cycles (`StratifiedKFold` on one label per cycle, so all windows of a cycle land in
  the same fold; folds are capped by the smallest class). Out-of-fold window predictions are reduced to one label per cycle
  by majority vote, and the cycle-level `cv_f1_macro_mean/std/min`,
  `cv_accuracy_mean` and `cv_recall_<class>_mean` are logged to MLflow. Compare runs
  on these when tuning. The registered model is then fit on all train cycles, and
  the list of its train cycles is logged as `train_cycles.json`.
- `evaluate` returns `deploy` only when
  - `cv_f1_macro_mean` reaches `min_f1_macro` and every `cv_recall_<class>_mean`
    reaches `min_class_recall`. With only 6-7 cycles per fault class, a single
    holdout has one or two cycles of each fault and its scores jump with each one,
    so the repeated CV over all train cycles is the main quality gate;
  - test cycle accuracy beats always predicting the majority class;
  - the candidate beats the Production model (the champion): cycle-level macro-F1
    higher by more than `min_improvement` and higher in at least `min_prob_better`
    of a paired bootstrap that resamples cycles within each class. Both models are
    scored on the test cycles missing from the champion's `train_cycles.json`, since
    a split rerun on more data can move a cycle the champion trained on into test.
    If the test macro-F1 is tied (for example both perfect), the candidate must
    instead have a `cv_f1_macro_mean` higher than the champion's by more than
    `min_improvement`. Retraining the same model on the same data is therefore never
    redeployed. When no comparison is possible (the champion has no
    `train_cycles.json`, cannot score the test set, or has seen every test cycle),
    `evaluate` prints a warning, records the reason as `champion_note`, and only
    the other gates apply.

  Cycle-level and window-level test metrics, per-class recall and the cycle-level
  confusion matrix are logged to MLflow; the confusion matrix also appears in the
  KFP UI.
- Per-machine report (information, not a gate). Fault classes are concentrated on
  one machine model (all Bearings and Motor cycles come from the same model) and the
  features alone identify the machine model of most cycles. A classifier can
  therefore score well by recognising the machine rather than the fault. `evaluate`
  reports test accuracy and per-class recall for each `model`, plus the metrics on
  `multi_class_models`, the machine models with at least two classes in the test
  set, where machine identity alone cannot give the answer. The table appears in the
  KFP UI next to the confusion matrix and as `test_machine_*` and
  `test_multi_class_models_*` metrics in MLflow.

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
| `k8s/local/` | Setup of a local k3s cluster with standalone KFP, MLflow and Seldon (see its README) |
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
from the metadata plus the exclusive window end in seconds), `DateTime`, `cycle_id`,
`brand` and `model` are metadata. All other columns are statistics and spectral band
shares of the 2048 Hz current and vibration signals (`fast.csv`).

The 1 Hz power meter (`slow.csv`) is left out by default: its clock does not match
the fast recording, and in every Motor cycle it reads zero power, so a model would
learn a recording artifact. Cycles are skipped, with the reason printed, when they
have no label, a missing `fast.csv`, less than one window of fast data, missing
values, fast signals in volts instead of ADC counts (three December 2021 cycles), or
a `fast.csv` identical to another cycle's. The current build keeps 83 of 96 cycles.

The fault classes are concentrated on one machine: all Bearings and Motor cycles come
from the Indesit BWE 101484X, so the features may partly encode machine identity. The
evaluation therefore also reports per-machine results.

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

Pipeline pods clone `repo_url` and read the `.dvc` file at `data_rev` (a branch, tag or
commit; empty means the default branch), so the `.dvc` change must be pushed. Pin
`data_rev` to a tag or commit to make a run reproducible. The `local` remote can't be reached from the cluster.

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
`repo_url`, `filename`, `dvc_remote` (`minio` or `s3`), `data_rev`, `test_size`, `n_estimators`, `max_depth`
(0 = unlimited), `random_state`, `cv_folds`, `cv_repeats`, `class_weight` (empty or
`balanced`), `min_f1_macro`, `min_class_recall`, `min_improvement`, `min_prob_better`,
`namespace` and `deployment_name`.

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
