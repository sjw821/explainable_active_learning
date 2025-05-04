import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def dataset_load_all(name, max_sample=5000, seed=0):
    data_df = pd.read_csv(f"./datasets/{name}.csv", header=None)
    data = data_df.values
    X, y = data[:, :-1], data[:, -1]
    if X.shape[0] > max_sample:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_sample, stratify=y, random_state=seed
        )
    return X, y

def noise_injection_uni_simple(
        X, missing_rate_row=30, missing_rate_col=50,
        missing_col_num=None, seed=0
    ):
    np.random.seed(seed)
    X_noised = X.astype(float).copy()
    n_rows, n_cols = X.shape
    if missing_col_num is None:
        missing_col_num = int(n_cols * missing_rate_col / 100)
    n_noise_rows = int(n_rows * missing_rate_row / 100)

    row_idx = np.random.choice(n_rows, n_noise_rows, replace=False)
    col_idx = np.random.choice(n_cols, missing_col_num, replace=False)

    for j in col_idx:
        col_vals = X[:, j]
        min_v, max_v = col_vals.min(), col_vals.max()
        noise = np.random.uniform(min_v, max_v, size=n_noise_rows)
        X_noised[row_idx, j] = noise

    return X_noised, col_idx
