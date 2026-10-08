# Local cluster (k3s)

Runs the whole pipeline on a single-node k3s (or any recent Kubernetes) without
full Kubeflow:

| Component | Version | Namespace |
| --- | --- | --- |
| Kubeflow Pipelines, standalone (Argo 3.4, MySQL, MinIO) | 2.4.1 | `kubeflow` |
| MinIO (`bitnamilegacy/minio`, replaces the unpublished KFP image) | 2024.11.7 | `kubeflow` |
| MLflow tracking server + registry (SQLite on a PVC) | 1.30.0 | `kubeflow` |
| Seldon Core v1 operator (no Istio/Ambassador) | 1.17.1 | `seldon-system` |
| Deploy namespace: rclone secret, RBAC for `kubeflow:pipeline-runner` | | `kubeflow-user-example-com` |

The KFP 2.x backend runs the Argo YAML compiled by the kfp 1.8 SDK through the
v1beta1 API, so `pipeline.py` works unchanged. The services match what the pipeline
expects: `mlflow-server.kubeflow:5000`, `minio.kubeflow:9000` (an alias of KFP's
`minio-service`) and the `mlpipeline-minio-artifact` secret. MinIO holds the
`mlpipeline`, `mlflow` and `dvc` buckets.

| Path | What |
| --- | --- |
| `setup.sh` | Installs or updates everything, then waits until it is ready |
| `kfp-cluster/`, `kfp/` | KFP kustomizations: upstream manifests, MinIO and argoexec images, `minio` service, smaller requests |
| `mlflow/` | MLflow server, PVC and service |
| `seldon/values.yaml` | Values for the `seldon-core-operator` chart |
| `user-namespace/rbac.yaml` | Lets pipeline steps create SeldonDeployments in the deploy namespace |

## Prerequisites

- `kubectl` pointing at the cluster, with a default StorageClass (k3s: `local-path`)
- `openssl`, and `helm` or Docker (the chart is rendered with `alpine/helm` if helm is missing)
- About 3 GB of free memory, plus 4 GB for the `train` step
- Internet access from the pods: pipeline steps install packages from PyPI, and the
  Seldon MLflow server builds a conda environment from the model

## Setup

```
k8s/local/setup.sh            # USER_NAMESPACE=... to deploy elsewhere
```

The first run generates random MinIO keys into the `mlpipeline-minio-artifact`
secret; later runs reuse them. KFP, MLflow, the pipeline steps and Seldon all read
this secret (Seldon through `seldon-init-container-secret`, which `setup.sh` derives
from it).

The deploy step image `ghcr.io/lucabon/washing-machine-deploy-model` is public. If it
is ever private, create the pull secret the pipeline references (without it, pods
only log a warning):

```
read -rs GHCR_TOKEN   # token with only read:packages
kubectl create secret docker-registry ghcr-pull-secret -n kubeflow \
  --docker-server=ghcr.io --docker-username=<github-user> --docker-password="$GHCR_TOKEN"
```

## Access

```
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80     # KFP UI and API
kubectl port-forward -n kubeflow svc/mlflow-server 5001:5000     # MLflow UI
kubectl port-forward -n kubeflow svc/minio 9102:9000             # MinIO S3 API
```

Point DVC at the cluster MinIO without printing the keys (MinIO port-forward running):

```
key() { kubectl get secret -n kubeflow mlpipeline-minio-artifact -o "jsonpath={.data.$1}" | base64 -d; }
dvc remote modify --local minio endpointurl http://localhost:9102
dvc remote modify --local minio access_key_id "$(key accesskey)"
dvc remote modify --local minio secret_access_key "$(key secretkey)"
dvc push -r minio && dvc status -c -r minio
```

## Run the pipeline

With the KFP port-forward running, after pushing the data and the `.dvc` file:

```
python pipeline.py run --host http://localhost:8888 --arg data_rev=develop
```

Follow the run in the KFP UI and the metrics in MLflow. To use MLflow from the
host, export `MLFLOW_TRACKING_URI=http://localhost:5001`,
`MLFLOW_S3_ENDPOINT_URL=http://localhost:9102` and the MinIO keys as
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.

## Call the model

Without Istio, port-forward the predictor service that Seldon creates
(`<deployment>-<predictor>`, here `washing-machine-default`), and use port 8000
(executor, REST):

```
kubectl port-forward -n kubeflow-user-example-com svc/washing-machine-default 9001:8000
curl -s -H 'Content-Type: application/json' \
  -d '{"data": {"names": [...], "ndarray": [[...]]}}' \
  http://localhost:9001/api/v1.0/predictions
```

`serving/predict.py` builds the URL for the Istio gateway
(`/seldon/<namespace>/<deployment>/api/v1.0/predictions`), so use its
`build_payload` with curl or `requests` against the URL above.

## Teardown

```
kubectl delete sdep --all -n kubeflow-user-example-com
kubectl delete namespace kubeflow-user-example-com seldon-system kubeflow
kubectl delete -k k8s/local/kfp-cluster
kubectl delete crd seldondeployments.machinelearning.seldon.io
kubectl delete validatingwebhookconfiguration seldon-validating-webhook-configuration
kubectl delete mutatingwebhookconfiguration cache-webhook-kubeflow
```

Deleting the `kubeflow` namespace deletes the MinIO, MySQL and MLflow volumes, so
all data, runs, models and the MinIO keys.

## Known limitations

- Not production-grade: MySQL has no root password, SQLite backs MLflow, and there
  is no authentication on any UI or API (reach them only through port-forwards).
- MinIO uses the frozen `bitnamilegacy/minio` image, since MinIO no longer
  publishes public images. It gets no security updates. Argo's executor comes from
  `quay.io/argoproj/argoexec`, as the KFP build of it is gone too.
- MLflow stays on 1.30: the pipeline's mlflow 1.24 client calls the model registry
  at `/api/2.0/preview/mlflow/...`, which MLflow 2.x servers no longer serve (404).
  Moving the server to 2.x requires upgrading the client in `pipeline.py` first.
- The MLflow server installs `boto3` from PyPI at every start (the official image
  lacks it), so it needs internet access to come up.
- `seldonio/mlflowserver` creates a conda environment from the model's `conda.yaml`
  (`conda-forge`, `python=3.9.x`) every time the predictor starts. Its conda 4.10
  loads the whole conda-forge index for that, which takes more than 2.3 GB of memory
  and several minutes, and needs internet access. With less free memory on the node
  the container is OOM-killed while "Collecting package metadata" and the
  SeldonDeployment never becomes `Available`.
- The operator creates its own `seldon-config` (`managerCreateResources`), which sets
  the storage initializer to `seldonio/rclone-storage-initializer:1.14.1`.
- `serving/predict.py` assumes the Istio gateway URL (see above).
