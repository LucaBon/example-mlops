from kfp.components import InputPath, OutputPath


def preprocess(file_path: InputPath('CSV'),
               train_path: OutputPath('CSV'),
               val_path: OutputPath('CSV'),
               test_path: OutputPath('CSV'),
               val_size: float = 0.2,
               test_size: float = 0.2,
               gap: int = 60):
    """Clean the raw windowed-signal dataset and split it chronologically.

    Consecutive windows overlap, so a random split would leak test
    information into training. Rows are sorted by ``DateTime`` and split into
    train / validation / test, oldest to newest. ``gap`` rows are dropped
    before the validation and the test part so that windows on either side of
    a boundary do not share samples; set it to at least the window overlap.
    """
    import pandas as pd

    time_col_name = 'TIMESTAMP'
    datetime_col_name = 'DateTime'
    target_col_name = 'target'
    meta_cols = [time_col_name, datetime_col_name, target_col_name]

    df = pd.read_csv(file_path)
    print(f'Raw dataset shape: {df.shape}')

    # Index columns written by a previous to_csv without index=False
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed:')])
    df[datetime_col_name] = pd.to_datetime(df[datetime_col_name])
    df = df.dropna(subset=[target_col_name]).drop_duplicates()

    feature_cols = [c for c in df.select_dtypes('number').columns
                    if c not in meta_cols]
    df = df[meta_cols + feature_cols].dropna()
    df = df.sort_values(datetime_col_name).reset_index(drop=True)

    n_rows = len(df)
    test_start = int(n_rows * (1 - test_size))
    val_start = int(n_rows * (1 - test_size - val_size))
    train_df = df.iloc[:max(val_start - gap, 0)]
    val_df = df.iloc[val_start:max(test_start - gap, val_start)]
    test_df = df.iloc[test_start:]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(f'Empty split with {n_rows} rows, val_size={val_size}, '
                         f'test_size={test_size}, gap={gap}')

    # Decided on train only, so the test data does not shape the features
    constant_cols = [c for c in feature_cols
                     if train_df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f'Dropping constant columns: {constant_cols}')
    keep = meta_cols + [c for c in feature_cols if c not in constant_cols]
    train_df, val_df, test_df = train_df[keep], val_df[keep], test_df[keep]

    print(f'Clean dataset shape: {df.shape}, features: {len(keep) - len(meta_cols)}')
    dropped = n_rows - len(train_df) - len(val_df) - len(test_df)
    print(f'Train rows: {len(train_df)}, validation rows: {len(val_df)}, '
          f'test rows: {len(test_df)}, gap rows dropped: {dropped}')
    for name, part in (('Train', train_df), ('Validation', val_df), ('Test', test_df)):
        print(f'{name} class balance:')
        print(part[target_col_name].value_counts(normalize=True).to_string())
    unseen = set(test_df[target_col_name]) - set(train_df[target_col_name])
    if unseen:
        print(f'WARNING: test classes missing from train: {sorted(unseen)}')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
