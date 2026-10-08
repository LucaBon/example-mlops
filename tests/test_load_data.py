import sys
import types

from components.load_data import get_data_from_dvc


def fake_dvc(monkeypatch, calls):
    class FakeFileSystem:
        def __init__(self, url, **kwargs):
            calls.append((url, kwargs))

        def get_file(self, filename, path):
            calls.append((filename, path))

    api = types.ModuleType('dvc.api')
    api.DVCFileSystem = FakeFileSystem
    monkeypatch.setitem(sys.modules, 'dvc', types.ModuleType('dvc'))
    monkeypatch.setitem(sys.modules, 'dvc.api', api)


def test_load_data_reads_requested_revision_and_remote(monkeypatch):
    calls = []
    fake_dvc(monkeypatch, calls)
    monkeypatch.setenv('MINIO_ACCESS_KEY_ID', 'key')
    monkeypatch.setenv('MINIO_SECRET_ACCESS_KEY', 'secret')
    get_data_from_dvc('repo', 'data.csv', 'out.csv', dvc_remote='minio', data_rev='develop')
    assert calls[0] == ('repo', {'rev': 'develop', 'remote': 'minio', 'remote_config': {
        'access_key_id': 'key', 'secret_access_key': 'secret'}})
    assert calls[1] == ('data.csv', 'out.csv')


def test_load_data_defaults_to_default_branch(monkeypatch):
    calls = []
    fake_dvc(monkeypatch, calls)
    monkeypatch.delenv('MINIO_ACCESS_KEY_ID', raising=False)
    get_data_from_dvc('repo', 'data.csv', 'out.csv', dvc_remote='s3')
    assert calls[0] == ('repo', {'rev': None, 'remote': 's3', 'remote_config': {}})
