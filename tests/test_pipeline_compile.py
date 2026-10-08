import os

import yaml


def test_pipeline_compiles_with_gated_deploy(monkeypatch):
    monkeypatch.delenv('KF_PIPELINES_ENDPOINT', raising=False)
    import pipeline  # importing must not contact a KFP host

    pipeline.main(['compile'])
    assert os.path.exists(pipeline.PIPELINE_FILE)
    with open(pipeline.PIPELINE_FILE) as f:
        workflow = yaml.safe_load(f)

    templates = {t['name']: t for t in workflow['spec']['templates']}
    assert {'get-data-from-dvc', 'preprocess', 'train', 'evaluate',
            'deploy-model', 'promote'} <= set(templates)
    dag_tasks = [task for t in workflow['spec']['templates'] if 'dag' in t
                 for task in t['dag']['tasks']]
    condition = next(t for t in dag_tasks if t['name'].startswith('condition'))
    assert 'deploy' in condition['when']

    condition_dag = templates[condition['template']]['dag']['tasks']
    promote = next(t for t in condition_dag if t['name'] == 'promote')
    assert 'deploy-model' in promote['dependencies']
