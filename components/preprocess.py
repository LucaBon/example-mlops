from kfp.components import InputPath, OutputPath


def preprocess(file_path: InputPath('CSV'),
               train_path: OutputPath('CSV'),
               test_path: OutputPath('CSV'),
               test_size: float = 0.33):
    """Clean the raw windowed-signal dataset and split it chronologically.

    Consecutive windows overlap, so a random split would leak test
    information into training: the last ``test_size`` fraction (by
    ``DateTime``) becomes the test set.
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
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f'Dropping constant columns: {constant_cols}')
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    df = df[meta_cols + feature_cols].dropna()

    df = df.sort_values(datetime_col_name).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    print(f'Clean dataset shape: {df.shape}, features: {len(feature_cols)}')
    print(f'Train rows: {len(train_df)}, test rows: {len(test_df)}')
    print('Train class balance:')
    print(train_df[target_col_name].value_counts(normalize=True).to_string())

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
