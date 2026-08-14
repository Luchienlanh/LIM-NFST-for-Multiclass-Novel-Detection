from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


SCALER_NAMES = (
    "QuantileTransformer",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "None",
)
SCALERS = SCALER_NAMES[:-1]


def make_scaler(scaler_name, random_state):
    scalers = {
        "QuantileTransformer": QuantileTransformer(
            output_distribution="normal",
            random_state=random_state,
        ),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
    }

    if scaler_name not in scalers:
        raise ValueError(
            f"Unknown scaler {scaler_name!r}. Valid values: {SCALER_NAMES}."
        )
    return scalers[scaler_name]


def remove_training_outliers(X_train, y_train, contamination=0.05):
    """Apply CPAI's class-wise LOF rule to training data only."""
    kept_X = []
    kept_y = []

    for class_label in np.unique(y_train):
        class_mask = y_train == class_label
        class_X = X_train[class_mask]
        class_y = y_train[class_mask]

        # CPAI keeps encoded class zero unchanged.
        if class_label == 0 or len(class_X) < 3:
            kept_X.append(class_X)
            kept_y.append(class_y)
            continue

        number_of_neighbors = min(20, len(class_X) - 1)
        detector = LocalOutlierFactor(
            n_neighbors=number_of_neighbors,
            contamination=contamination,
        )
        is_inlier = detector.fit_predict(class_X) != -1
        kept_X.append(class_X[is_inlier])
        kept_y.append(class_y[is_inlier])

    return np.vstack(kept_X), np.concatenate(kept_y)


def preprocess_data(
    dataframe,
    dataset_name,
    scaler_name="QuantileTransformer",
    random_state=42,
    test_size=0.20,
    return_preprocessing=False,
):
    """Run the established poly=-1 preprocessing path for multiclass LIM."""
    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    raw_y = data[:, -1]

    imputer = None
    if dataset_name == "IoTID20":
        X[np.isinf(X)] = np.nan
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

    X_train, X_test, raw_y_train, raw_y_test = train_test_split(
        X,
        raw_y,
        test_size=test_size,
        stratify=raw_y,
        random_state=random_state,
    )

    order = raw_y_train.argsort()
    X_train = X_train[order]
    raw_y_train = raw_y_train[order]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(raw_y_train)
    y_test = label_encoder.transform(raw_y_test)

    if scaler_name == "None":
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    else:
        scaler = make_scaler(scaler_name, random_state)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.nan_to_num(
        X_train_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    X_test_scaled = np.nan_to_num(
        X_test_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_train_clean, y_train_clean = remove_training_outliers(
        X_train_scaled,
        y_train,
    )
    result = (
        X_train_clean,
        y_train_clean,
        X_test_scaled,
        y_test,
        label_encoder,
    )
    if not return_preprocessing:
        return result

    preprocessing = {
        "imputer": imputer,
        "scaler": scaler,
        "scaler_name": scaler_name,
        "feature_names": list(dataframe.columns[:-1]),
    }
    return (*result, preprocessing)
