from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from limnfst.dataloader.common import (
    DatasetBundle,
    concat_bundles,
    dataset_root,
    iter_csv_limited,
    make_bundle,
    normalize_label,
    require_dir,
    require_files,
    resolve_files,
    sorted_csv_files,
)
from limnfst.dataloader.labels import botiot_label, validate_label_granularity


DEFAULT_ROOT = dataset_root("BoT_IOT", "OneDrive_1_6-11-2026", "Entire Dataset")
FEATURE_NAMES_FILE = "UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv"
LABEL_COLUMNS = ("attack", "category", "subcategory")
METADATA_COLUMNS = (
    "pkSeqID",
    "stime",
    "ltime",
    "proto",
    "saddr",
    "sport",
    "daddr",
    "dport",
    "attack",
    "category",
    "subcategory",
)


def botiot_feature_names(root: str | Path | None = None) -> list[str]:
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    path = data_root / FEATURE_NAMES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Feature names file not found: {path}")
    return [col.strip() for col in pd.read_csv(path, nrows=0).columns]


def botiot_files(root: str | Path | None = None) -> list[Path]:
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    files = sorted_csv_files(
        data_root,
        "UNSW_2018_IoT_Botnet_Dataset_*.csv",
        exclude={FEATURE_NAMES_FILE},
    )
    return require_files(files, data_root)


def _multiclass_labels(df: pd.DataFrame, label_granularity: str = "paper") -> pd.Series:
    label_granularity = validate_label_granularity(label_granularity)
    if {"category", "subcategory"}.issubset(df.columns):
        return pd.Series(
            (
                botiot_label(category, subcategory, label_granularity)
                for category, subcategory in zip(df["category"], df["subcategory"])
            ),
            index=df.index,
        )
    if "category" in df.columns:
        return df["category"].map(normalize_label)
    return pd.Series(["unknown"] * len(df), index=df.index)


def iter_botiot_chunks(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    drop_columns: Sequence[str] = LABEL_COLUMNS,
    label_granularity: str = "paper",
) -> Iterator[DatasetBundle]:
    label_granularity = validate_label_granularity(label_granularity)
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    paths = resolve_files(data_root, files, botiot_files(data_root))
    names = botiot_feature_names(data_root)

    for path, chunk in iter_csv_limited(
        paths,
        max_rows=max_rows,
        chunksize=chunksize,
        header=None,
        names=names,
        skipinitialspace=True,
        low_memory=False,
    ):
        if "attack" not in chunk.columns:
            raise ValueError(f"'attack' column not found in {path}")

        yield make_bundle(
            "BoT_IOT",
            chunk,
            y_binary=chunk["attack"],
            y_multiclass=_multiclass_labels(chunk, label_granularity),
            drop_columns=drop_columns,
            metadata_columns=METADATA_COLUMNS,
            source_file=path.name,
        )


def load_botiot(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    drop_columns: Sequence[str] = LABEL_COLUMNS,
    label_granularity: str = "paper",
) -> DatasetBundle:
    return concat_bundles(
        "BoT_IOT",
        iter_botiot_chunks(
            root,
            max_rows=max_rows,
            chunksize=chunksize,
            files=files,
            drop_columns=drop_columns,
            label_granularity=label_granularity,
        ),
    )


load_dataset = load_botiot
iter_chunks = iter_botiot_chunks
