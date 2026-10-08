# Deploy component

Container component that renders a Seldon Core `SeldonDeployment` for an MLflow
model (`templates/deploy-manifest.j2`), applies it with `kubectl` and waits
until it is `Available`.

Build and push the image with `./build_image.sh`, then update the image in
`component.yaml`. Preview the manifest locally with:

    python src/deploy.py --model_uri s3://mlflow/0/<run>/artifacts/model --render_only
