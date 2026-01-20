from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from src.config import Config


@dataclass
class PreprocessOutput:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def _train_test_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    n_test = int(len(df) * test_size)
    df_test = df.iloc[:n_test].copy()
    df_train = df.iloc[n_test:].copy()
    return df_train, df_test


def preprocess_house_prices(cfg: Config) -> PreprocessOutput:
    if not cfg.data_raw.exists():
        raise FileNotFoundError(f"Не найден {cfg.data_raw}. Положи train.csv в data/raw/train.csv")

    df = pd.read_csv(cfg.data_raw)

    if "SalePrice" not in df.columns:
        raise ValueError("В train.csv должен быть столбец SalePrice")

    y = df["SalePrice"].astype(float).values
    if cfg.log_target:
        y = np.log1p(y)

    X_df = df.drop(columns=["SalePrice"])

    full = X_df.copy()
    full["__y__"] = y
    train_df, test_df = _train_test_split(full, cfg.test_size, cfg.random_state)

    y_train = train_df["__y__"].values.astype(float)
    y_test = test_df["__y__"].values.astype(float)

    X_train_df = train_df.drop(columns=["__y__"])
    X_test_df = test_df.drop(columns=["__y__"])

    num_cols = X_train_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train_df.columns if c not in num_cols]

    for c in num_cols:
        med = X_train_df[c].median()
        X_train_df[c] = X_train_df[c].fillna(med)
        X_test_df[c] = X_test_df[c].fillna(med)

    for c in cat_cols:
        X_train_df[c] = X_train_df[c].astype("object").fillna("Unknown")
        X_test_df[c] = X_test_df[c].astype("object").fillna("Unknown")

    combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    combined_ohe = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    X_train_ohe = combined_ohe.iloc[:len(X_train_df)].copy()
    X_test_ohe = combined_ohe.iloc[len(X_train_df):].copy()

    X_train_np = X_train_ohe.values.astype(float)
    X_test_np = X_test_ohe.values.astype(float)

    mu = X_train_np.mean(axis=0)
    sigma = X_train_np.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train_np = (X_train_np - mu) / sigma
    X_test_np = (X_test_np - mu) / sigma

    feature_names = list(X_train_ohe.columns)

    return PreprocessOutput(
        X_train=X_train_np,
        X_test=X_test_np,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
