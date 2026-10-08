# Washing machine MLOps pipeline

Kubeflow pipeline (kfp v1 SDK) that trains a washing-machine cycle classifier
on windowed sensor signals and serves it with Seldon Core.

```
load_data ─► preprocess ─► train ─► evaluate ─► [decision == deploy] ─► deploy ─► promote
  (DVC/S3)    clean +        RandomForest   test metrics vs          SeldonDeployment  version ->
              chronological  logged to      min_accuracy and         (MLFLOW_SERVER)   Production
              train/test     MLflow +       current Production
              split          registry
```

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

1. Set `USER_NAMESPACE` in `admin-integrations.sh` and run it.
2. Create the `aws-secret` (keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
   in the user namespace. It gives the `load_data` step access to the DVC
   remote `s3://mlops-remote-storage`.

## Build the deploy component image

```
cd components/deploy && ./build_image.sh
```

Update the image in `components/deploy/component.yaml` if you push to your own registry.

## Compile, run, schedule

```
pip install -r requirements.txt
python pipeline.py compile                      # -> generated/washing_machine-pipeline.yaml
python pipeline.py run --host <KFP_URL> --arg min_accuracy=0.9 --arg n_estimators=500
python pipeline.py schedule --host <KFP_URL> --cron "0 0 3 * * 1"   # weekly retraining
```

You can also upload `generated/washing_machine-pipeline.yaml` through the UI. Its parameters are
`repo_url`, `filename`, `test_size`, `n_estimators`, `max_depth` (0 = unlimited),
`random_state`, `min_accuracy`, `namespace` and `deployment_name`.

## Use the model

```
python -m serving.predict data.csv --host http://<istio-ingress> --namespace kalpa-k8 --rows 5 \
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
