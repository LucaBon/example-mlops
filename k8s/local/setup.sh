#!/bin/bash
# Installs standalone Kubeflow Pipelines, MLflow and Seldon Core on the cluster
# of the current kubectl context (tested on single-node k3s). Safe to rerun:
# the MinIO credentials are generated on the first run and reused afterwards.
#
#   USER_NAMESPACE=kubeflow-user-example-com k8s/local/setup.sh
set -euo pipefail

USER_NAMESPACE="${USER_NAMESPACE:-kubeflow-user-example-com}"
SELDON_NAMESPACE=seldon-system
SELDON_CHART_VERSION=1.17.1
SELDON_CHART_REPO=https://storage.googleapis.com/seldon-charts
HELM_IMAGE=alpine/helm:3.16.2
TIMEOUT=900s
DIR="$(cd "$(dirname "$0")" && pwd)"

apply() { kubectl apply --server-side --force-conflicts "$@"; }

echo "==> Kubeflow Pipelines: cluster-scoped resources"
apply -k "$DIR/kfp-cluster"
kubectl wait --for=condition=established --timeout=120s \
  crd/applications.app.k8s.io crd/workflows.argoproj.io crd/scheduledworkflows.kubeflow.org

echo "==> MinIO credentials (secret kubeflow/mlpipeline-minio-artifact)"
if kubectl get secret -n kubeflow mlpipeline-minio-artifact >/dev/null 2>&1; then
  echo "    exists, reusing it"
else
  # printf is a builtin: the keys never appear in a process list or the output
  kubectl create secret generic mlpipeline-minio-artifact -n kubeflow \
    --from-env-file=<(printf 'accesskey=%s\nsecretkey=%s\n' \
      "$(openssl rand -hex 10)" "$(openssl rand -hex 20)")
  kubectl label secret -n kubeflow mlpipeline-minio-artifact \
    application-crd-id=kubeflow-pipelines
fi

echo "==> Kubeflow Pipelines 2.4.1"
apply -k "$DIR/kfp"

echo "==> MLflow"
apply -k "$DIR/mlflow"

echo "==> Seldon Core operator $SELDON_CHART_VERSION"
kubectl create namespace "$SELDON_NAMESPACE" --dry-run=client -o yaml | apply -f -
helm_args=(template seldon-core seldon-core-operator --repo "$SELDON_CHART_REPO"
           --version "$SELDON_CHART_VERSION" --namespace "$SELDON_NAMESPACE")
if command -v helm >/dev/null 2>&1; then
  helm "${helm_args[@]}" -f "$DIR/seldon/values.yaml" | apply -f -
else
  docker run --rm -v "$DIR/seldon:/values:ro" "$HELM_IMAGE" \
    "${helm_args[@]}" -f /values/values.yaml | apply -f -
fi

echo "==> Deploy namespace $USER_NAMESPACE"
kubectl create namespace "$USER_NAMESPACE" --dry-run=client -o yaml | apply -f -
apply -n "$USER_NAMESPACE" -f "$DIR/user-namespace/rbac.yaml"
key() {
  kubectl get secret -n kubeflow mlpipeline-minio-artifact -o "jsonpath={.data.$1}" | base64 -d
}
# rclone settings for the Seldon storage initializer that downloads s3:// models
kubectl create secret generic seldon-init-container-secret -n "$USER_NAMESPACE" \
  --from-env-file=<(printf '%s\n' \
    RCLONE_CONFIG_S3_TYPE=s3 \
    RCLONE_CONFIG_S3_PROVIDER=Minio \
    RCLONE_CONFIG_S3_ENV_AUTH=false \
    "RCLONE_CONFIG_S3_ACCESS_KEY_ID=$(key accesskey)" \
    "RCLONE_CONFIG_S3_SECRET_ACCESS_KEY=$(key secretkey)" \
    RCLONE_CONFIG_S3_ENDPOINT=http://minio.kubeflow.svc.cluster.local:9000) \
  --dry-run=client -o yaml | apply -f -

echo "==> Waiting for deployments (first run pulls images, takes a few minutes)"
kubectl rollout status -n "$SELDON_NAMESPACE" deploy/seldon-controller-manager --timeout="$TIMEOUT"
for d in $(kubectl get deploy -n kubeflow -o name); do
  kubectl rollout status -n kubeflow "$d" --timeout="$TIMEOUT"
done

echo "==> MinIO buckets"
kubectl exec -n kubeflow deploy/minio -- \
  mc mb --ignore-existing local/mlpipeline local/mlflow local/dvc

echo "Done. Port-forward commands are in k8s/local/README.md."
