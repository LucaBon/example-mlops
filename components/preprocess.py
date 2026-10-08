from kfp.components import InputPath, OutputPath


def preprocess(file_path: InputPath('CSV'),
               train_path: OutputPath('CSV'),
               test_path: OutputPath('CSV'),
               test_size: float = 0.25):
    """Clean the raw windowed-signal dataset and split it by washing cycle.

    Windows of the same cycle overlap and are near-duplicates, so the cycle
    is the sample: every cycle goes entirely to train or to test. Within each
    class, cycles are ranked by the md5 of their ``cycle_id`` and the lowest
    ``round(test_size * n_cycles)`` go to test (at least one when the class
    has two or more cycles, none when it has one). The hash ranking does not
    depend on row order or on the other cycles, so adding cycles only moves
    the train/test boundary of a class by the change in its test quota.
    """
    import hashlib

    import pandas as pd

    time_col_name = 'TIMESTAMP'
    datetime_col_name = 'DateTime'
    target_col_name = 'target'
    cycle_col_name = 'cycle_id'
    meta_cols = [time_col_name, datetime_col_name, target_col_name, cycle_col_name,
                 'brand', 'model']

    df = pd.read_csv(file_path, dtype={cycle_col_name: str})
    print(f'Raw dataset shape: {df.shape}')
    missing = [c for c in (time_col_name, datetime_col_name, target_col_name, cycle_col_name)
               if c not in df.columns]
    if missing:
        raise ValueError(f'Input is missing required columns {missing}; '
                         f'{cycle_col_name!r} is needed to split by washing cycle')

    # Index columns written by a previous to_csv without index=False
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed:')])
    df[datetime_col_name] = pd.to_datetime(df[datetime_col_name])
    df = df.dropna(subset=[target_col_name, cycle_col_name]).drop_duplicates()

    meta_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.select_dtypes('number').columns
                    if c not in meta_cols]
    df = df[meta_cols + feature_cols].dropna(subset=feature_cols)
    df = df.sort_values([cycle_col_name, time_col_name]).reset_index(drop=True)

    # A cycle is one sample: one class and one machine model
    for col in (target_col_name, 'model'):
        if col in df.columns:
            n_values = df.groupby(cycle_col_name)[col].nunique()
            mixed = sorted(n_values[n_values > 1].index)
            if mixed:
                raise ValueError(f'Cycles with more than one {col!r} value: {mixed[:10]}')
    cycle_class = df.groupby(cycle_col_name)[target_col_name].first()
    test_cycles = set()
    for _, cycles in cycle_class.groupby(cycle_class):
        ranked = sorted(cycles.index,
                        key=lambda c: hashlib.md5(c.encode()).hexdigest())
        if len(ranked) >= 2:
            n_test = min(max(int(round(test_size * len(ranked))), 1), len(ranked) - 1)
            test_cycles.update(ranked[:n_test])

    is_test = df[cycle_col_name].isin(test_cycles)
    train_df, test_df = df[~is_test], df[is_test]
    if train_df.empty or test_df.empty:
        raise ValueError(f'Empty split with {len(cycle_class)} cycles, '
                         f'test_size={test_size}')

    # Decided on train only, so the test data does not shape the features
    constant_cols = [c for c in feature_cols
                     if train_df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f'Dropping constant columns: {constant_cols}')
    keep = meta_cols + [c for c in feature_cols if c not in constant_cols]
    train_df, test_df = train_df[keep], test_df[keep]

    print(f'Clean dataset shape: {df.shape}, features: {len(keep) - len(meta_cols)}')
    print(f'Train rows: {len(train_df)}, test rows: {len(test_df)}')
    print('Cycles per class:')
    print(pd.DataFrame({
        name: part.groupby(target_col_name)[cycle_col_name].nunique()
        for name, part in (('train', train_df), ('test', test_df))
    }).fillna(0).astype(int).to_string())
    for name, part in (('train', train_df), ('test', test_df)):
        missing = sorted(set(cycle_class) - set(part[target_col_name]))
        if missing:
            print(f'WARNING: classes missing from {name}: {missing}')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
