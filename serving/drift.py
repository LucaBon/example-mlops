"""Detect feature drift between a reference dataset and a recent batch.

Exits with code 1 when the share of drifted features exceeds the threshold,
so it can gate a retraining job.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NON_FEATURE_COLUMNS = ['TIMESTAMP', 'DateTime', 'target', 'cycle_id', 'brand', 'model']


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index over reference-quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac = np.clip(ref_frac, 1e-6, None)
    cur_frac = np.clip(cur_frac, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 p_value: float = 0.01, psi_threshold: float = 0.2,
                 share_threshold: float = 0.3) -> dict:
    columns = [c for c in reference.select_dtypes('number').columns
               if c not in NON_FEATURE_COLUMNS and c in current.columns]
    features = {}
    for col in columns:
        ref, cur = reference[col].dropna().values, current[col].dropna().values
        ks = ks_2samp(ref, cur)
        col_psi = psi(ref, cur)
        features[col] = {
            'ks_p_value': float(ks.pvalue),
            'psi': col_psi,
            'drift': bool(ks.pvalue < p_value and col_psi > psi_threshold),
        }
    drifted = [c for c, r in features.items() if r['drift']]
    share = len(drifted) / len(columns) if columns else 0.0
    return {'drifted_features': drifted, 'drift_share': share,
            'dataset_drift': share > share_threshold, 'features': features}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', help='reference CSV (e.g. training split)')
    parser.add_argument('current', help='recent inputs CSV')
    parser.add_argument('--p_value', type=float, default=0.01)
    parser.add_argument('--psi_threshold', type=float, default=0.2)
    parser.add_argument('--share_threshold', type=float, default=0.3)
    args = parser.parse_args(argv)

    report = detect_drift(pd.read_csv(args.reference), pd.read_csv(args.current),
                          args.p_value, args.psi_threshold, args.share_threshold)
    print(json.dumps(report, indent=2))
    return 1 if report['dataset_drift'] else 0


if __name__ == '__main__':
    sys.exit(main())
