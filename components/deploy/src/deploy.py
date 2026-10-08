import argparse
import os
import subprocess
import sys
import tempfile

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, 'templates')


def render_manifest(model_uri: str, deployment_name: str) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR),
                      trim_blocks=True, lstrip_blocks=True)
    template = env.get_template('deploy-manifest.j2')
    return template.render(model_uri=model_uri,
                           deployment_name=deployment_name)


def deploy(model_uri: str, namespace: str, deployment_name: str,
           timeout: str = '600s') -> int:
    rendered = render_manifest(model_uri, deployment_name)
    print("Rendered Manifest:")
    print(rendered)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        f.write(rendered)
        f.flush()
        result = subprocess.call(
            ['kubectl', 'apply', '-f', f.name, '-n', namespace])
    if result != 0:
        print(f'kubectl apply failed with exit code {result}', file=sys.stderr)
        return result

    return subprocess.call(
        ['kubectl', 'wait', f'sdep/{deployment_name}', '-n', namespace,
         '--for=jsonpath={.status.state}=Available', f'--timeout={timeout}'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy an MLflow model with Seldon Core')
    parser.add_argument('--model_uri', required=True, help='Model URI')
    parser.add_argument('--namespace', default='kalpa-k8')
    parser.add_argument('--deployment_name', default='washing-machine')
    parser.add_argument('--render_only', action='store_true',
                        help='print the manifest without applying it')

    args = parser.parse_args()

    if args.render_only:
        print(render_manifest(args.model_uri, args.deployment_name))
        sys.exit(0)
    sys.exit(deploy(args.model_uri, args.namespace, args.deployment_name))
