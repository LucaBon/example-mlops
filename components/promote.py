def promote(model_version: str,
            registered_model_name: str = 'WashingMachineModel'):
    """Move a deployed model version to Production, archiving the previous one.

    Runs after the deploy step so the registry only marks a version as
    Production once Seldon is actually serving it.
    """
    from mlflow.tracking import MlflowClient

    MlflowClient().transition_model_version_stage(
        registered_model_name, model_version, 'Production',
        archive_existing_versions=True)
    print(f'{registered_model_name} v{model_version} promoted to Production')
