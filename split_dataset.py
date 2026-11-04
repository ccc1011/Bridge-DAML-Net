import pandas as pd
import numpy as np
import os


def split_dataset_by_report(csv_path, output_root, train_ratio=0.7, val_ratio=0.15,
                            test_ratio=0.15, random_state=42, group_col='source'):
    """
    Split the dataset by inspection report.

    All records from one report are assigned to the same subset, which prevents
    data leakage between subsets.

    Args:
        csv_path: Input CSV file with description, component, rating and group_col
        output_root: Output folder. The subsets are written to its multi/ subfolder
        train_ratio, val_ratio, test_ratio: Target ratios of reports
        random_state: Random seed
        group_col: Column that identifies the report
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "The ratios must sum to 1"

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {df[group_col].nunique()} reports")

    # Remove ratings 0-3
    filtered_ratings = [0, 1, 2, 3]
    original_len = len(df)
    df = df[~df['rating'].isin(filtered_ratings)].reset_index(drop=True)
    removed = original_len - len(df)
    print(f"Removed {removed} records with ratings {filtered_ratings}: "
          f"{len(df)} records from {df[group_col].nunique()} reports remain")

    # Split the reports so that each one appears in a single subset
    groups = df[group_col].unique()
    n_groups = len(groups)
    rng = np.random.RandomState(random_state)
    shuffled_groups = rng.permutation(groups)

    n_train_g = int(round(n_groups * train_ratio))
    n_val_g = int(round(n_groups * val_ratio))

    train_groups = set(shuffled_groups[:n_train_g])
    val_groups = set(shuffled_groups[n_train_g:n_train_g + n_val_g])
    test_groups = set(shuffled_groups[n_train_g + n_val_g:])

    # Confirm that no report appears in two subsets
    assert not (train_groups & val_groups), "train and val share reports"
    assert not (train_groups & test_groups), "train and test share reports"
    assert not (val_groups & test_groups), "val and test share reports"

    def assign_split(source_id):
        if source_id in train_groups:
            return 'train'
        elif source_id in val_groups:
            return 'val'
        else:
            return 'test'

    df['__split__'] = pd.Categorical(df[group_col].map(assign_split), categories=['train', 'val', 'test'])

    train_df = df[df['__split__'] == 'train'].drop(columns=['__split__'])
    val_df = df[df['__split__'] == 'val'].drop(columns=['__split__'])
    test_df = df[df['__split__'] == 'test'].drop(columns=['__split__'])

    # Shuffle the records within each subset
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    total_records = len(train_df) + len(val_df) + len(test_df)
    print(f"\nSplit by '{group_col}' with random_state={random_state}")
    print(f"Reports: train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}, "
          f"total={len(train_groups) + len(val_groups) + len(test_groups)}")
    print(f"Records: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, total={total_records}")
    print(f"Record share (%): train={len(train_df) / total_records * 100:.2f}, "
          f"val={len(val_df) / total_records * 100:.2f}, test={len(test_df) / total_records * 100:.2f}")

    print("\nRating distribution:")
    print(pd.crosstab(df['rating'], df['__split__']))

    print("\nComponent distribution:")
    print(pd.crosstab(df['component'], df['__split__']))

    output_cols = ['description', 'component', 'rating']

    output_dir = os.path.join(output_root, 'multi')
    os.makedirs(output_dir, exist_ok=True)

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df[output_cols].to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)

    print(f"\nSaved {output_dir}/{{train,val,test}}.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    split_dataset_by_report(
        csv_path="dataset/vtrans_bridge_inspection_results.csv",
        output_root="dataset",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        group_col='source'
    )
