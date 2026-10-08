import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'components' / 'deploy' / 'src'))


def make_dataset(n_rows=600, seed=0, shift=0.0, signal=1.0, class_probs=None):
    """Synthetic data with the real schema: features separate 3 classes.

    ``signal=0`` makes the features pure noise; ``class_probs`` sets the
    class balance.
    """
    rng = np.random.default_rng(seed)
    target = rng.choice(3, n_rows, p=class_probs)
    df = pd.DataFrame({
        'TIMESTAMP': np.arange(n_rows) * 60,
        'DateTime': pd.date_range('2022-01-01', periods=n_rows, freq='min'),
    })
    for i in range(8):
        df[f'feature_{i}'] = signal * target * (i % 3) + rng.normal(shift, 0.5, n_rows)
    df['constant'] = 1.0
    df['target'] = target
    return df


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / 'raw.csv'
    # Shuffled rows + written index: preprocess must sort and drop it
    make_dataset().sample(frac=1, random_state=1).to_csv(path)
    return path


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # default artifact root ./mlruns
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/mlflow.db')
    monkeypatch.delenv('MLFLOW_S3_ENDPOINT_URL', raising=False)
    import mlflow
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/mlflow.db')
    yield tmp_path
