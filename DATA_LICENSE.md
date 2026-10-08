# Data license and attribution

The training data `wm_cycles_win60.csv` (versioned with DVC, not stored in git) is
derived from:

- **Dataset:** T. Fonseca, L.L. Ferreira, P. Chaves, B. Cabral, P. Costa,
  "SMART-PDM Appliance Dataset", Zenodo, 2022.
  [doi:10.5281/zenodo.7245198](https://doi.org/10.5281/zenodo.7245198)
- **Paper:** T. Fonseca, P. Chaves, L.L. Ferreira, N. Gouveia, D. Costa, A. Oliveira,
  J. Landeck, "Dataset for identifying maintenance needs of home appliances using
  artificial intelligence", Data in Brief 48 (2023) 109068.
  [doi:10.1016/j.dib.2023.109068](https://doi.org/10.1016/j.dib.2023.109068)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The original authors do not endorse this project or the derived data.

## Changes

- Only the washing machine part (`2-washing_machines`) is used.
- `washing_machine_metadata.csv` is deduplicated (one row per cycle) and only the
  cycle label (`failure`), `brand`, `model` and start time (`timestamp_begin`) are used.
- Each cycle is cut into sliding windows (60 s long, every 10 s). Per window the CSV
  holds statistics and spectral band shares of the `fast.csv` signals (`Current`,
  `Vibration`). `TIMESTAMP` is the cycle start from the metadata plus the window end.
  `slow.csv` is not used by default; with the `--with-slow` option its summary
  statistics are added and the two recordings are time-aligned per cycle as
  described in `data_prep/build_dataset.py`.
- Cycles are left out for quality reasons: no label in the metadata, a `fast.csv`
  shorter than one window (most December 2021 cycles and one empty file), `fast.csv`
  signals in volts instead of ADC counts (three December 2021 cycles), or a
  `fast.csv` byte-identical to another cycle's (all copies dropped if their labels
  differ, otherwise one kept). The current build keeps 83 of the 96 cycles.
- Nothing else from the dataset is redistributed: the raw recordings, the metadata
  and `WM_ExtractedFeatures.csv` are not in this repository or its DVC remotes.

`data_prep/build_dataset.py` rebuilds the derived CSV from the raw data. Anyone who
shares `wm_cycles_win60.csv` or data derived from it must keep this attribution
and the license notice.
