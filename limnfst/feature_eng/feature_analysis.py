from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


def get_label_related_cols(df, label_col):
    label_names = {"attack", "category", "subcategory", "label", "class", "target"}
    cols = []
    for col in df.columns:
        if col == label_col:
            continue
        if str(col).lower() in label_names:
            cols.append(col)
    return cols


def _missing_rate(s):
    missing = s.isna()
    if pd.api.types.is_numeric_dtype(s):
        missing = missing | np.isinf(pd.to_numeric(s, errors="coerce"))
    return float(missing.mean())


def _top_freq(s):
    if len(s) == 0:
        return 0.0
    return float(s.value_counts(dropna=False, normalize=True).iloc[0])


def _variance(s):
    if not pd.api.types.is_numeric_dtype(s):
        return np.nan
    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.var()) if values.notna().any() else np.nan


def _flag_feature(col, n_unique, unique_ratio, top_freq, label_related_cols):
    name = str(col).lower()
    flags = []

    if col in label_related_cols:
        flags.append("label_related")
    if name in ("id", "pkseqid", "seq", "flow_id") or name.endswith("_id"):
        flags.append("id_like")
    if name in ("ts", "stime", "ltime", "time", "timestamp") or "timestamp" in name:
        flags.append("time_like")
    if name in ("saddr", "daddr", "src_ip", "dst_ip", "source_ip", "destination_ip"):
        flags.append("ip_like")
    if name in ("sport", "dport", "src_port", "dst_port", "source_port", "destination_port"):
        flags.append("port_like")
    if n_unique <= 1 or top_freq >= 0.99:
        flags.append("near_constant")
    if n_unique > 50 and unique_ratio >= 0.8:
        flags.append("high_cardinality")

    if not flags:
        flags.append("safe_candidate")
    return ",".join(flags)


def basic_feature_report(df, label_col, label_related_cols=None):
    if label_related_cols is None:
        label_related_cols = get_label_related_cols(df, label_col)

    rows = []
    n = len(df)
    for col in df.columns:
        if col == label_col:
            continue

        s = df[col]
        n_unique = int(s.nunique(dropna=False))
        unique_ratio = float(n_unique / n) if n else 0.0
        top_freq = _top_freq(s)
        flag = _flag_feature(col, n_unique, unique_ratio, top_freq, label_related_cols)

        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "missing_rate": _missing_rate(s),
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
                "top_freq": top_freq,
                "variance": _variance(s),
                "flag": flag,
            }
        )

    return pd.DataFrame(rows)


def _encode_for_importance(X):
    parts = []
    discrete = []

    for col in X.columns:
        s = X[col]

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            values = values.to_numpy(dtype=np.float64)
            mean = np.nanmean(values)
            if np.isnan(mean):
                mean = 0.0
            values[np.isnan(values)] = mean
            parts.append(values.reshape(-1, 1))
            discrete.append(False)
        else:
            values = s.astype("object")
            values = values.where(~values.isna(), "__missing__").astype(str)
            codes, _ = pd.factorize(values, sort=True)
            parts.append(codes.astype(np.float64).reshape(-1, 1))
            discrete.append(True)

    if not parts:
        return np.empty((len(X), 0)), np.array([], dtype=bool)

    return np.hstack(parts), np.array(discrete, dtype=bool)


def feature_importance_report(
    df,
    label_col,
    *,
    label_related_cols=None,
    test_size=0.2,
    random_state=42,
    stratify=True,
    max_rows=20000,
    n_estimators=200,
):
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"{label_col} is not in dataframe")

    if label_related_cols is None:
        label_related_cols = get_label_related_cols(df, label_col)

    stratify_arg = df[label_col] if stratify else None
    train_df, _ = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    if max_rows is not None and len(train_df) > max_rows:
        stratify_arg = train_df[label_col] if stratify else None
        train_df, _ = train_test_split(
            train_df,
            train_size=max_rows,
            random_state=random_state,
            stratify=stratify_arg,
        )

    report = basic_feature_report(train_df, label_col, label_related_cols)
    report["used_for_importance"] = ~report["flag"].str.contains("label_related")
    report["mutual_info"] = np.nan
    report["tree_importance"] = np.nan

    used_cols = report.loc[report["used_for_importance"], "feature"].tolist()
    if len(used_cols) == 0:
        return report

    X_imp, discrete = _encode_for_importance(train_df[used_cols])
    y_train = train_df[label_col].to_numpy()

    mi = mutual_info_classif(
        X_imp,
        y_train,
        discrete_features=discrete,
        random_state=random_state,
    )

    tree = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    tree.fit(X_imp, y_train)

    for col, mi_score, tree_score in zip(used_cols, mi, tree.feature_importances_):
        mask = report["feature"] == col
        report.loc[mask, "mutual_info"] = float(mi_score)
        report.loc[mask, "tree_importance"] = float(tree_score)

    report = report.sort_values(
        ["tree_importance", "mutual_info"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return report


def print_top_features(report, n=20):
    cols = [
        "feature",
        "dtype",
        "flag",
        "missing_rate",
        "n_unique",
        "mutual_info",
        "tree_importance",
    ]
    print(report[cols].head(n).to_string(index=False))
