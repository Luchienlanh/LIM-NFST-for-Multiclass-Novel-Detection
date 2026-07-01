from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from limnfst.sampling import stratified_min_per_class_sample


def make_scaler(name: str, random_state: int = 42, n_samples: int | None = None):
    key = name.lower()
    if key in ("standard", "standardscaler"):
        return StandardScaler()
    if key in ("minmax", "minmaxscaler"):
        return MinMaxScaler()
    if key in ("robust", "robustscaler"):
        return RobustScaler()
    if key == "normalizer":
        return Normalizer()
    if key in ("quantile", "quantiletransformer"):
        n_quantiles = 1000 if n_samples is None else min(1000, n_samples)
        return QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=random_state,
        )
    raise ValueError(f"Unknown scaler: {name}")


def split(X, y, test_size=0.2, random_state=42, stratify=True):
    stratify_arg = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


def _to_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def _encode_features(X_train, X_test, categorical="onehot"):
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    parts_train = []
    parts_test = []
    artifacts = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": None,
        "cat_imputer": None,
        "encoder": None,
    }

    if num_cols:
        X_train_num = X_train[num_cols].replace([np.inf, -np.inf], np.nan)
        X_test_num = X_test[num_cols].replace([np.inf, -np.inf], np.nan)

        num_imputer = SimpleImputer(strategy="mean")
        X_train_num = num_imputer.fit_transform(X_train_num)
        X_test_num = num_imputer.transform(X_test_num)

        parts_train.append(X_train_num)
        parts_test.append(X_test_num)
        artifacts["num_imputer"] = num_imputer

    if cat_cols:
        X_train_cat = X_train[cat_cols].astype("object")
        X_test_cat = X_test[cat_cols].astype("object")
        X_train_cat = X_train_cat.where(~X_train_cat.isna(), np.nan)
        X_test_cat = X_test_cat.where(~X_test_cat.isna(), np.nan)

        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = cat_imputer.fit_transform(X_train_cat)
        X_test_cat = cat_imputer.transform(X_test_cat)
        X_train_cat = X_train_cat.astype(str)
        X_test_cat = X_test_cat.astype(str)

        if categorical == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif categorical == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            raise ValueError("categorical must be 'onehot' or 'ordinal'")

        X_train_cat = encoder.fit_transform(X_train_cat)
        X_test_cat = encoder.transform(X_test_cat)

        parts_train.append(X_train_cat)
        parts_test.append(X_test_cat)
        artifacts["cat_imputer"] = cat_imputer
        artifacts["encoder"] = encoder

    if not parts_train:
        raise ValueError("X has no usable columns")

    X_train_out = np.hstack(parts_train).astype(np.float64)
    X_test_out = np.hstack(parts_test).astype(np.float64)
    return X_train_out, X_test_out, artifacts


def preprocess(
    X,
    y,
    *,
    scaler="quantile",
    test_size=0.2,
    random_state=42,
    stratify=True,
    sample_train=False,
    n_train_samples=None,
    min_per_class=1,
    categorical="onehot",
    return_artifacts=False,
):
    X = _to_dataframe(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if sample_train:
        if n_train_samples is None:
            raise ValueError("n_train_samples is required when sample_train=True")
        X_train, y_train, sample_idx = stratified_min_per_class_sample(
            X_train,
            y_train,
            n_samples=n_train_samples,
            min_per_class=min_per_class,
            random_state=random_state,
            return_indices=True,
        )
    else:
        sample_idx = None

    X_train, X_test, artifacts = _encode_features(X_train, X_test, categorical=categorical)

    sc = make_scaler(scaler, random_state=random_state, n_samples=len(X_train))
    X_train = np.nan_to_num(sc.fit_transform(X_train), nan=0.0)
    X_test = np.nan_to_num(sc.transform(X_test), nan=0.0)

    if return_artifacts:
        artifacts["scaler"] = sc
        artifacts["sample_idx"] = sample_idx
        return X_train, X_test, y_train, y_test, artifacts

    return X_train, X_test, y_train, y_test
