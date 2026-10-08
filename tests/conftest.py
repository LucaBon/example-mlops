import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'components' / 'deploy' / 'src'))

CLASSES = ['Working', 'Heating', 'Motor']


def make_dataset(n_cycles=40, windows_per_cycle=15, seed=0, shift=0.0, signal=1.0,
                 class_probs=(0.5, 0.25, 0.25), first_cycle=0):
    """Synthetic data with the real schema: windows grouped in washing cycles.

    Each cycle has one class and ``windows_per_cycle`` rows; the class counts
    follow ``class_probs`` exactly. ``signal=0`` makes the features pure noise.
    'Motor' cycles all come from one machine model, as in the real data.
    ``first_cycle`` offsets the cycle ids, to append new cycles to a dataset.
    """
    rng = np.random.default_rng(seed)
    counts = np.round(np.asarray(class_probs) * n_cycles).astype(int)
    counts[0] = n_cycles - counts[1:].sum()
    cycle_class = rng.permutation(np.repeat(np.arange(len(CLASSES)), counts))
    target = np.repeat(cycle_class, windows_per_cycle)
    cycle = np.repeat(np.arange(first_cycle, first_cycle + n_cycles), windows_per_cycle)
    n_rows = len(target)
    machine = np.where(cycle_class == 2, 'X9', np.where(np.arange(n_cycles) % 2, 'A1', 'B2'))
    df = pd.DataFrame({
        'TIMESTAMP': (np.arange(n_rows) + first_cycle * windows_per_cycle) * 10,
        'DateTime': pd.date_range('2022-01-01', periods=n_rows, freq='10s')
        + pd.Timedelta(seconds=10 * first_cycle * windows_per_cycle),
        'cycle_id': [f'cycle_{c:04d}' for c in cycle],
        'brand': 'Acme',
        'model': np.repeat(machine, windows_per_cycle),
    })
    for i in range(8):
        cycle_effect = np.repeat(rng.normal(0, 0.3, n_cycles), windows_per_cycle)
        df[f'feature_{i}'] = (signal * target * (i % 3) + cycle_effect
                              + rng.normal(shift, 0.5, n_rows))
    df['constant'] = 1.0
    df['target'] = np.asarray(CLASSES)[target]
    return df


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / 'raw.csv'
    # Shuffled rows + written index: preprocess must drop it
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
