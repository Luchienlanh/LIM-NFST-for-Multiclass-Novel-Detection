from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


DEFAULT_MISSING_VALUES = ("", " ", "-", "?", "nan", "NaN", "None", "null", "NULL")


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y_binary: pd.Series
    y_multiclass: pd.Series
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_columns: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.X)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dataset_root(*parts: str) -> Path:
    return project_root().joinpath("Datasets", *parts)


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def sorted_csv_files(root: Path, pattern: str = "*.csv", exclude: Iterable[str] = ()) -> list[Path]:
    excluded = set(exclude)
    return sorted(
        (path for path in root.glob(pattern) if path.name not in excluded),
        key=natural_key,
    )


def require_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")
    return path


def require_files(paths: Sequence[Path], root: Path) -> list[Path]:
    files = list(paths)
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root}")
    return files


def resolve_files(root: Path, files: Sequence[str | Path] | None, default_files: Sequence[Path]) -> list[Path]:
    if files is None:
        return list(default_files)

    resolved = []
    for file_path in files:
        path = Path(file_path)
        resolved.append(path if path.is_absolute() else root / path)
    return require_files(resolved, root)


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")


def binary_from_label(value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return "benign" if float(value) == 0.0 else "attack"

    label = normalize_label(value)
    if label in {"0", "0.0", "false", "benign", "normal", "normaltraffic", "benigntraffic"}:
        return "benign"
    return "attack"


def normalize_missing_values(
    df: pd.DataFrame,
    missing_values: Iterable[object] = DEFAULT_MISSING_VALUES,
) -> pd.DataFrame:
    return df.replace(list(missing_values), pd.NA)


def coerce_numeric_columns(df: pd.DataFrame, min_success_rate: float = 0.95) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_object_dtype(out[col]):
            continue

        source = out[col]
        non_missing = source.notna()
        if not non_missing.any():
            continue

        converted = pd.to_numeric(source, errors="coerce")
        success_rate = float(converted[non_missing].notna().mean())
        if success_rate >= min_success_rate:
            out[col] = converted

    return out


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def make_bundle(
    name: str,
    df: pd.DataFrame,
    *,
    y_binary: Sequence[object] | pd.Series,
    y_multiclass: Sequence[object] | pd.Series,
    drop_columns: Iterable[str] = (),
    metadata_columns: Iterable[str] = (),
    source_file: str | None = None,
    coerce_numeric: bool = True,
) -> DatasetBundle:
    frame = normalize_missing_values(df.copy())

    y_binary_series = pd.Series(y_binary, name="label_binary", index=frame.index).map(binary_from_label)
    y_multiclass_series = pd.Series(
        y_multiclass,
        name="label_multiclass",
        index=frame.index,
    ).map(normalize_label)

    meta_cols = existing_columns(frame, metadata_columns)
    metadata = frame[meta_cols].copy() if meta_cols else pd.DataFrame(index=frame.index)
    metadata["label_binary"] = y_binary_series
    metadata["label_multiclass"] = y_multiclass_series
    if source_file is not None:
        metadata["source_file"] = source_file

    drop_cols = existing_columns(frame, drop_columns)
    X = frame.drop(columns=drop_cols, errors="ignore")
    X = normalize_missing_values(X)
    if coerce_numeric:
        X = coerce_numeric_columns(X)

    return DatasetBundle(
        name=name,
        X=X.reset_index(drop=True),
        y_binary=y_binary_series.reset_index(drop=True),
        y_multiclass=y_multiclass_series.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        dropped_columns=tuple(drop_cols),
    )


def concat_bundles(name: str, bundles: Iterable[DatasetBundle]) -> DatasetBundle:
    parts = list(bundles)
    if not parts:
        raise ValueError(f"No chunks loaded for {name}")

    return DatasetBundle(
        name=name,
        X=pd.concat([part.X for part in parts], ignore_index=True),
        y_binary=pd.concat([part.y_binary for part in parts], ignore_index=True),
        y_multiclass=pd.concat([part.y_multiclass for part in parts], ignore_index=True),
        metadata=pd.concat([part.metadata for part in parts], ignore_index=True),
        dropped_columns=parts[0].dropped_columns,
    )


def iter_csv_limited(
    files: Sequence[Path],
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    **read_csv_kwargs,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    remaining = max_rows
    for path in files:
        if remaining is not None and remaining <= 0:
            break

        if chunksize is None:
            nrows = remaining if remaining is not None else None
            chunk = pd.read_csv(path, nrows=nrows, **read_csv_kwargs)
            if len(chunk) == 0:
                continue
            yield path, chunk
            if remaining is not None:
                remaining -= len(chunk)
            continue

        effective_chunksize = chunksize
        if remaining is not None:
            effective_chunksize = min(chunksize, remaining)

        reader = pd.read_csv(path, chunksize=effective_chunksize, **read_csv_kwargs)
        for chunk in reader:
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            if len(chunk) == 0:
                continue
            yield path, chunk
            if remaining is not None:
                remaining -= len(chunk)


def combine_label_parts(*parts: object) -> str:
    labels = [normalize_label(part) for part in parts if normalize_label(part) not in {"", "unknown"}]
    if not labels:
        return "unknown"
    if len(set(labels)) == 1:
        return labels[0]
    return ".".join(labels)
