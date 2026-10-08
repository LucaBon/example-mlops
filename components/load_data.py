from kfp.components import OutputPath


def get_data_from_dvc(repo_url: str, filename: str, data_path: OutputPath('CSV'),
                      dvc_remote: str = 'minio', data_rev: str = ''):
    """Download a DVC-tracked file from one of the remotes in ``.dvc/config``.

    ``dvc_remote`` is ``minio`` (in-cluster), ``s3`` (AWS) or ``local`` (a
    folder, only reachable from a developer machine). MinIO credentials come
    from ``MINIO_ACCESS_KEY_ID``/``MINIO_SECRET_ACCESS_KEY``, AWS credentials
    from the standard ``AWS_*`` variables. ``data_rev`` is the git branch,
    tag or commit of ``repo_url`` whose ``.dvc`` file is read; empty means the
    default branch.
    """
    import os

    from dvc.api import DVCFileSystem

    remote_config = {}
    if dvc_remote == 'minio' and os.getenv('MINIO_ACCESS_KEY_ID'):
        remote_config = {
            'access_key_id': os.environ['MINIO_ACCESS_KEY_ID'],
            'secret_access_key': os.environ['MINIO_SECRET_ACCESS_KEY'],
        }
    fs = DVCFileSystem(repo_url, rev=data_rev or None, remote=dvc_remote,
                       remote_config=remote_config)
    print(f'Downloading {filename} at {data_rev or "default branch"} '
          f'from DVC remote "{dvc_remote}"...')
    fs.get_file(filename, data_path)
