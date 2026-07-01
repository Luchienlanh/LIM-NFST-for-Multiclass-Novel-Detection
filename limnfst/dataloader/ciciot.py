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
from limnfst.dataloader.labels import map_ciciot_labels, validate_label_granularity


DEFAULT_ROOT = dataset_root("CIC_IOT", "MERGED_CSV", "MERGED_CSV")
LABEL_COLUMN = "Label"
DROP_COLUMNS = (LABEL_COLUMN,)
METADATA_COLUMNS = (LABEL_COLUMN,)


def ciciot_files(root: str | Path | None = None) -> list[Path]:
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    return require_files(sorted_csv_files(data_root, "Merged*.csv"), data_root)


def iter_ciciot_chunks(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    label_granularity: str = "paper",
) -> Iterator[DatasetBundle]:
    label_granularity = validate_label_granularity(label_granularity)
    data_root = require_dir(Path(root) if root is not None else DEFAULT_ROOT)
    paths = resolve_files(data_root, files, ciciot_files(data_root))

    for path, chunk in iter_csv_limited(paths, max_rows=max_rows, chunksize=chunksize):
        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"{LABEL_COLUMN!r} column not found in {path}")
        y_multiclass = map_ciciot_labels(chunk[LABEL_COLUMN], label_granularity)

        yield make_bundle(
            "CIC_IOT",
            chunk,
            y_binary=chunk[LABEL_COLUMN],
            y_multiclass=y_multiclass,
            drop_columns=DROP_COLUMNS,
            metadata_columns=METADATA_COLUMNS,
            source_file=path.name,
        )


def load_ciciot(
    root: str | Path | None = None,
    *,
    max_rows: int | None = None,
    chunksize: int | None = 100_000,
    files: Sequence[str | Path] | None = None,
    label_granularity: str = "paper",
) -> DatasetBundle:
    return concat_bundles(
        "CIC_IOT",
        iter_ciciot_chunks(
            root,
            max_rows=max_rows,
            chunksize=chunksize,
            files=files,
            label_granularity=label_granularity,
        ),
    )


load_dataset = load_ciciot
iter_chunks = iter_ciciot_chunks
