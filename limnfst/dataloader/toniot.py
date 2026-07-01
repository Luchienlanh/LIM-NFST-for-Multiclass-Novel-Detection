from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from limnfst.dataloader.common import (
    DatasetBundle,
    concat_bundles,
    dataset_root,
    iter_csv_limited,
    make_bundle,
    require_dir,
    require_files,
    resolve_files,
    sorted_csv_files,
)
from limnfst.dataloader.labels import validate_label_granularity


DEFAULT_ROOT = dataset_root("ToN_IoT", "Processed_Network_dataset", "Processed_Network_dataset")
LABEL_COLUMN = "label"
TYPE_COLUMN = "type"
LABEL_COLUMNS = ("label", "type")
METADATA_COLUMNS = (
    "ts",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "proto",
    "service",
    "label",
    "type",
)


def toniot_files(root: str | Path | None = None) -> list[Path]:
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    return require_files(sorted_csv_files(data_root, "Network_dataset_*.csv"), data_root)


def iter_toniot_chunks(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    drop_columns: Sequence[str] = LABEL_COLUMNS,
    label_granularity: str = "paper",
) -> Iterator[DatasetBundle]:
    validate_label_granularity(label_granularity)
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    paths = resolve_files(data_root, files, toniot_files(data_root))

    for path, chunk in iter_csv_limited(paths, max_rows=max_rows, chunksize=chunksize, low_memory=False):
        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"{LABEL_COLUMN!r} column not found in {path}")
        if TYPE_COLUMN not in chunk.columns:
            raise ValueError(f"{TYPE_COLUMN!r} column not found in {path}")

        yield make_bundle(
            "ToN_IoT",
            chunk,
            y_binary=chunk[LABEL_COLUMN],
            y_multiclass=chunk[TYPE_COLUMN],
            drop_columns=drop_columns,
            metadata_columns=METADATA_COLUMNS,
            source_file=path.name,
        )


def load_toniot(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    drop_columns: Sequence[str] = LABEL_COLUMNS,
    label_granularity: str = "paper",
) -> DatasetBundle:
    return concat_bundles(
        "ToN_IoT",
        iter_toniot_chunks(
            root,
            max_rows=max_rows,
            chunksize=chunksize,
            files=files,
            drop_columns=drop_columns,
            label_granularity=label_granularity,
        ),
    )


load_dataset = load_toniot
iter_chunks = iter_toniot_chunks
