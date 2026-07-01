from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from limnfst.dataloader.common import (
    DatasetBundle,
    concat_bundles,
    dataset_root,
    iter_csv_limited,
    make_bundle,
    natural_key,
    normalize_label,
    require_dir,
    require_files,
    resolve_files,
)
from limnfst.dataloader.labels import nbaiot_label, validate_label_granularity


DEFAULT_ROOT = dataset_root("N_BaloT", "archive")
SKIP_FILES = {"features.csv", "device_info.csv", "data_summary.csv"}
METADATA_COLUMNS = ("device_id", "device_name", "raw_label", "paper_label")
LABEL_COLUMNS = ("raw_label", "paper_label")


def nbaiot_files(root: str | Path | None = None) -> list[Path]:
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    files = [
        path
        for path in data_root.glob("*.csv")
        if path.name not in SKIP_FILES and path.name.count(".") >= 1
    ]
    return require_files(sorted(files, key=natural_key), data_root)


def _read_device_names(root: Path) -> dict[str, str]:
    path = root / "device_info.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"DeviceID", "DeviceName"}.issubset(df.columns):
        return {}
    return dict(zip(df["DeviceID"].astype(str), df["DeviceName"].astype(str)))


def _file_labels(path: Path) -> tuple[str, str]:
    parts = path.stem.split(".")
    device_id = parts[0]
    label = ".".join(parts[1:]) if len(parts) > 1 else "unknown"
    return device_id, normalize_label(label)


def iter_nbaiot_chunks(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    label_granularity: str = "paper",
) -> Iterator[DatasetBundle]:
    label_granularity = validate_label_granularity(label_granularity)
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    paths = resolve_files(data_root, files, nbaiot_files(data_root))
    device_names = _read_device_names(data_root)

    for path, chunk in iter_csv_limited(paths, max_rows=max_rows, chunksize=chunksize):
        device_id, label = _file_labels(path)
        mapped_label = nbaiot_label(label, label_granularity)
        frame = chunk.copy()
        frame["device_id"] = device_id
        frame["device_name"] = device_names.get(device_id, "")
        frame["raw_label"] = label
        frame["paper_label"] = mapped_label

        yield make_bundle(
            "N_BaIoT",
            frame,
            y_binary=frame["paper_label"],
            y_multiclass=frame["paper_label"],
            drop_columns=LABEL_COLUMNS,
            metadata_columns=METADATA_COLUMNS,
            source_file=path.name,
        )


def load_nbaiot(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    label_granularity: str = "paper",
) -> DatasetBundle:
    return concat_bundles(
        "N_BaIoT",
        iter_nbaiot_chunks(
            root,
            max_rows=max_rows,
            chunksize=chunksize,
            files=files,
            label_granularity=label_granularity,
        ),
    )


load_dataset = load_nbaiot
iter_chunks = iter_nbaiot_chunks
