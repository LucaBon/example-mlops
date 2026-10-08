from kfp.components import OutputPath


def get_data_from_dvc(repo_url: str, filename: str, data_path: OutputPath('CSV')):
    from dvc.api import DVCFileSystem
    fs = DVCFileSystem(repo_url)
    print('Downloading data from DVC...')
    fs.get_file(filename, data_path)
