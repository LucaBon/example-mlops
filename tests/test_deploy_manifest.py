import re

import yaml

from deploy import render_manifest

DNS_1123 = re.compile(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')


def test_manifest_is_valid_seldon_deployment():
    uri = 's3://mlflow/0/abc123/artifacts/model'
    manifest = yaml.safe_load(render_manifest(uri, 'washing-machine'))

    assert manifest['apiVersion'] == 'machinelearning.seldon.io/v1'
    assert manifest['kind'] == 'SeldonDeployment'
    assert manifest['metadata']['name'] == 'washing-machine'
    predictor = manifest['spec']['predictors'][0]
    assert predictor['graph']['modelUri'] == uri
    assert predictor['graph']['implementation'] == 'MLFLOW_SERVER'
    for name in (manifest['metadata']['name'], manifest['spec']['name'],
                 predictor['name'], predictor['graph']['name']):
        assert DNS_1123.match(name), name
    image = predictor['componentSpecs'][0]['spec']['containers'][0]['image']
    assert not image.endswith('-dev')
