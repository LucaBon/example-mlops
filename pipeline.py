# use virtual env from requirements.txt

import argparse
import os
from pathlib import Path

import kfp
from kfp import dsl
from kfp.aws import use_aws_secret
from kfp.onprem import use_k8s_secret
from kubernetes.client.models import V1EnvVar, V1LocalObjectReference

from components.evaluate import evaluate
from components.load_data import get_data_from_dvc
from components.preprocess import preprocess
from components.promote import promote
from components.train import train

OUTPUT_DIRECTORY = 'generated'
PROJECT_ROOT = Path(__file__).absolute().parent
PIPELINE_FILE = os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                             'washing_machine-pipeline.yaml')

MLFLOW_TRACKING_URI = 'http://mlflow-server.kubeflow.svc.cluster.local:5000'
MLFLOW_S3_ENDPOINT_URL = 'http://minio.kubeflow.svc.cluster.local:9000'
# docker-registry secret for ghcr.io, created by admin-integrations.sh
IMAGE_PULL_SECRET = 'ghcr-pull-secret'
BASE_IMAGE = 'python:3.9'
ML_PACKAGES = ['pandas==1.4.2', 'scikit-learn==1.0.2', 'mlflow==1.24.0',
               'boto3==1.21.32', 'protobuf==3.20.0', 'setuptools<70']

DEFAULT_ARGUMENTS = {
    'repo_url': 'https://github.com/LucaBon/washingmachine-mlops.git',
    'filename': 'signal_cycles_train_win_60_data.csv',
}


def _component(func, name, packages):
    return kfp.components.create_component_from_func(
        func=func,
        # This is optional. It saves the component spec for future use.
        output_component_file=os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                                           f'{name}-component.yaml'),
        base_image=BASE_IMAGE,
        packages_to_install=packages)


load_data_op = _component(get_data_from_dvc, 'load_data', ['dvc', 'dvc-s3'])
preprocess_op = _component(preprocess, 'preprocess', ['pandas==1.4.2'])
training_op = _component(train, 'train', ML_PACKAGES)
evaluate_op = _component(evaluate, 'evaluate', ML_PACKAGES)
promote_op = _component(promote, 'promote', ML_PACKAGES)
deploy_op = kfp.components.load_component_from_file(
    os.path.join(PROJECT_ROOT, 'components', 'deploy', 'component.yaml'))


def with_mlflow_env(task):
    """Point a task at the in-cluster MLflow server and MinIO artifact store."""
    return (task
            .add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                       value=MLFLOW_TRACKING_URI))
            .add_env_variable(V1EnvVar(name='MLFLOW_S3_ENDPOINT_URL',
                                       value=MLFLOW_S3_ENDPOINT_URL))
            # https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.extensions.html#kfp.onprem.use_k8s_secret
            .apply(use_k8s_secret(secret_name='mlpipeline-minio-artifact',
                                  k8s_secret_key_to_env={
                                      'accesskey': 'AWS_ACCESS_KEY_ID',
                                      'secretkey': 'AWS_SECRET_ACCESS_KEY',
                                  })))


@dsl.pipeline(
    name="washing_machine_pipeline",
    description="WASHING MACHINE pipeline",
)
def washing_machine_pipeline(
        repo_url: str = DEFAULT_ARGUMENTS['repo_url'],
        filename: str = DEFAULT_ARGUMENTS['filename'],
        test_size: float = 0.33,
        n_estimators: int = 300,
        max_depth: int = 0,
        random_state: int = 42,
        min_accuracy: float = 0.8,
        namespace: str = 'kubeflow-user-example-com',
        deployment_name: str = 'washing-machine'):
    # The deploy component image is private on ghcr.io
    dsl.get_pipeline_conf().set_image_pull_secrets(
        [V1LocalObjectReference(name=IMAGE_PULL_SECRET)])

    load_data_task = load_data_op(repo_url, filename).apply(
        use_aws_secret(secret_name='aws-secret',
                       aws_access_key_id_name='AWS_ACCESS_KEY_ID',
                       aws_secret_access_key_name='AWS_SECRET_ACCESS_KEY',
                       aws_region='eu-south-1'))

    preprocess_task = preprocess_op(file=load_data_task.outputs['data'],
                                    test_size=test_size)

    train_task = with_mlflow_env(training_op(
        train=preprocess_task.outputs['train'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state))

    evaluate_task = with_mlflow_env(evaluate_op(
        test=preprocess_task.outputs['test'],
        run_id=train_task.outputs['run_id'],
        model_version=train_task.outputs['model_version'],
        min_accuracy=min_accuracy))

    with dsl.Condition(evaluate_task.outputs['decision'] == 'deploy'):
        deploy_task = deploy_op(model_uri=train_task.outputs['model_uri'],
                                namespace=namespace,
                                deployment_name=deployment_name)
        with_mlflow_env(promote_op(
            model_version=train_task.outputs['model_version'])).after(deploy_task)


def _client(host):
    return kfp.Client(host=host) if host else kfp.Client()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Washing machine pipeline')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('compile', help='compile the pipeline to YAML (default)')
    for name, help_text in (('run', 'submit a single run'),
                            ('schedule', 'create a recurring retraining run')):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--host', help='KFP API endpoint (defaults to in-cluster)')
        p.add_argument('--experiment', default='washing-machine')
        p.add_argument('--arg', action='append', default=[],
                       metavar='KEY=VALUE', help='pipeline argument override')
        if name == 'schedule':
            p.add_argument('--cron', default='0 0 3 * * 1',
                           help='KFP 6-field cron (default: Mondays 03:00)')
    args = parser.parse_args(argv)

    if args.command in (None, 'compile'):
        kfp.compiler.Compiler().compile(washing_machine_pipeline, PIPELINE_FILE)
        print(f'Generated the washing machine pipeline definition: {PIPELINE_FILE}')
        return

    arguments = dict(DEFAULT_ARGUMENTS)
    arguments.update(kv.split('=', 1) for kv in args.arg)
    client = _client(args.host)

    if args.command == 'run':
        client.create_run_from_pipeline_func(
            washing_machine_pipeline, arguments=arguments,
            experiment_name=args.experiment)
    else:
        kfp.compiler.Compiler().compile(washing_machine_pipeline, PIPELINE_FILE)
        experiment = client.create_experiment(args.experiment)
        client.create_recurring_run(
            experiment_id=experiment.id,
            job_name='washing-machine-retraining',
            cron_expression=args.cron,
            pipeline_package_path=PIPELINE_FILE,
            params=arguments)


if __name__ == '__main__':
    main()
