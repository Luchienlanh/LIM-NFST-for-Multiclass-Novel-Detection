"""Evaluate and tune RFF-REF-LIM for multiclass novel detection.

For each outer novel class, the grid excludes that class completely. It uses
only the remaining outer-training classes in an inner LOCO + StratifiedKFold
selection loop, then saves one best configuration for that outer class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))

from limnfst.datasets import DATASETS, DEFAULT_LIMITS, load_dataset
from limnfst.models import LIM_NFST
from limnfst.preprocessing import SCALERS, make_scaler, remove_training_outliers

TEST_SIZE = 0.20
NOVEL_LABEL = -1
MODEL_NAME = "RFF-REF-LIM"
SCRIPT_NAME = "lim_nfst_rff_mnd_nested_loco_v1"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_GRID_SCALERS = list(SCALERS)
DEFAULT_GRID_REFERENCE_SIZES = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_GRID_NEIGHBORS = [1, 3, 5, 7, 9]
DEFAULT_GRID_RFF_COMPONENTS = [128, 256, 512]
DEFAULT_GRID_RFF_GAMMA_MULTIPLIERS = [0.10, 1.0, 10.0]
DEFAULT_GRID_NOVELTY_QUANTILES = [0.90, 0.95, 0.99]

METRIC_COLUMNS = [
    "accuracy_with_novel",
    "accuracy_closed_world",
    "balanced_accuracy_with_novel",
    "balanced_accuracy_closed_world",
    "novel_detection_precision",
    "novel_detection_recall",
    "novel_detection_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc_with_novel",
]
FIXED_RUN_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "novel_class",
    "test_set",
    *METRIC_COLUMNS,
]
GRID_RUN_COLUMNS = [
    "dataset",
    "model",
    "outer_novel_class",
    "inner_novel_class",
    "seed",
    "fold",
    "test_set",
    *METRIC_COLUMNS,
]
SUMMARY_COLUMNS = ["dataset", "model", "test_set", *METRIC_COLUMNS]
GRID_RESULT_COLUMNS = [
    "dataset",
    "outer_novel_class",
    "model",
    "test_set",
    *METRIC_COLUMNS,
]
CANDIDATE_COLUMNS = [
    "dataset",
    "outer_novel_class",
    "model",
    "configuration",
    "scaler",
    "reference_size",
    "neighbors",
    "epsilon",
    "use_rff",
    "rff_components",
    "rff_gamma_mode",
    "rff_gamma_multiplier",
    "threshold_mode",
    "novelty_quantile",
    "selection_metric",
    "selection_metric_std",
    "selection_validation_runs",
    *METRIC_COLUMNS,
]
BEST_PARAMETER_COLUMNS = [
    "dataset",
    "outer_novel_class",
    "scaler",
    "reference_size",
    "neighbors",
    "epsilon",
    "use_rff",
    "rff_components",
    "rff_gamma_mode",
    "rff_gamma_multiplier",
    "threshold_mode",
    "novelty_quantile",
    "selection_metric",
    "validation_score",
    "validation_score_std",
    "cross_validation",
    "cv_folds",
    "cv_repeats",
    "mnd_selection_protocol",
    "lim_code_version",
]


def get_version() -> str:
    source_files = [
        Path(__file__),
        CLASSIFIER_ROOT / "limnfst" / "models.py",
        CLASSIFIER_ROOT / "limnfst" / "nfst.py",
        CLASSIFIER_ROOT / "limnfst" / "novelty.py",
        CLASSIFIER_ROOT / "limnfst" / "mapping.py",
        CLASSIFIER_ROOT / "limnfst" / "datasets.py",
        CLASSIFIER_ROOT / "limnfst" / "preprocessing.py",
    ]
    code_hash = hashlib.sha256()
    for source_file in source_files:
        code_hash.update(source_file.name.encode("utf-8"))
        code_hash.update(source_file.read_bytes())
    return f"{SCRIPT_NAME}_{code_hash.hexdigest()[:8]}"


def number_to_name(value: float) -> str:
    return format(float(value), ".6g").replace(".", "p")


def save_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate or tune RFF-REF-LIM for MND/LOCO."
    )
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="BoT_IoT")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--novel-class",
        default=None,
        help="Held-out raw label or zero-based index; omit for full LOCO.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    scaler_group = parser.add_mutually_exclusive_group()
    scaler_group.add_argument("--scaler", choices=SCALERS)
    scaler_group.add_argument(
        "--no-scaler", action="store_const", const="None", dest="scaler"
    )
    parser.set_defaults(scaler="StandardScaler")

    parser.add_argument("--reference-size", type=float, default=0.20)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--rff-components", type=int, default=256)
    parser.add_argument("--rff-gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--novelty-quantile", type=float, default=0.95)

    parser.add_argument(
        "--grid-scalers",
        nargs="+",
        choices=[*SCALERS, "None"],
        default=DEFAULT_GRID_SCALERS,
    )
    parser.add_argument(
        "--grid-reference-sizes",
        nargs="+",
        type=float,
        default=DEFAULT_GRID_REFERENCE_SIZES,
    )
    parser.add_argument(
        "--grid-neighbors",
        nargs="+",
        type=int,
        default=DEFAULT_GRID_NEIGHBORS,
    )
    parser.add_argument(
        "--grid-rff-components",
        nargs="+",
        type=int,
        default=DEFAULT_GRID_RFF_COMPONENTS,
    )
    parser.add_argument(
        "--grid-rff-gamma-multipliers",
        nargs="+",
        type=float,
        default=DEFAULT_GRID_RFF_GAMMA_MULTIPLIERS,
    )
    parser.add_argument(
        "--grid-novelty-quantiles",
        nargs="+",
        type=float,
        default=DEFAULT_GRID_NOVELTY_QUANTILES,
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=METRIC_COLUMNS,
        default="novel_detection_f1",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def get_datasets_and_seeds(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], bool]:
    run_all = args.all_datasets
    if run_all and args.limit is not None:
        raise ValueError("--limit can only be used with one --dataset.")

    datasets = list(DATASETS) if run_all else [args.dataset]
    if args.seeds is not None:
        seeds = args.seeds
    elif args.grid_search:
        seeds = [args.seed]
    elif run_all:
        seeds = DEFAULT_SEEDS
    else:
        seeds = [args.seed]
    return datasets, seeds, run_all


def resolve_novel_class(y_raw: np.ndarray, requested_class: object) -> object:
    classes = np.unique(y_raw)
    if requested_class is None:
        return classes[0]

    matching_labels = [
        label for label in classes if str(label) == str(requested_class)
    ]
    if matching_labels:
        return matching_labels[0]
    return classes[int(requested_class)]


def resolve_novel_classes(
    dataframe: pd.DataFrame,
    requested_class: object,
) -> list[object]:
    labels = dataframe.iloc[:, -1].to_numpy()
    if requested_class is not None:
        return [resolve_novel_class(labels, requested_class)]
    return np.unique(labels).tolist()


def raw_dataset_arrays(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    data = dataframe.to_numpy()
    return data[:, :-1].astype(np.float64), data[:, -1]


def split_outer_data(
    dataframe: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y_raw = raw_dataset_arrays(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_raw,
        test_size=TEST_SIZE,
        stratify=y_raw,
        random_state=seed,
    )
    return X_train, y_train, X_test, y_test


def split_outer_mnd_data(
    dataframe: pd.DataFrame,
    novel_class: object,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
]:
    X_train, y_train, X_test, y_test = split_outer_data(dataframe, seed)
    train_known_mask = y_train != novel_class
    test_known_mask = y_test != novel_class
    return (
        X_train[train_known_mask],
        y_train[train_known_mask],
        X_test[test_known_mask],
        y_test[test_known_mask],
        X_test[~test_known_mask],
        novel_class,
    )


def sanitize_features(values: np.ndarray) -> np.ndarray:
    clean_values = np.asarray(values, dtype=np.float64).copy()
    clean_values[np.isinf(clean_values)] = np.nan
    return clean_values


def preprocess_mnd_split(
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_known_test: np.ndarray,
    y_known_test_raw: np.ndarray,
    X_novel_test: np.ndarray,
    scaler: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = sanitize_features(X_train)
    X_known_test = sanitize_features(X_known_test)
    X_novel_test = sanitize_features(X_novel_test)

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_known_test = imputer.transform(X_known_test)
    X_novel_test = imputer.transform(X_novel_test)
    if scaler != "None":
        fitted_scaler = make_scaler(scaler, random_state=seed)
        X_train = fitted_scaler.fit_transform(X_train)
        X_known_test = fitted_scaler.transform(X_known_test)
        X_novel_test = fitted_scaler.transform(X_novel_test)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_known_test = np.nan_to_num(X_known_test, nan=0.0)
    X_novel_test = np.nan_to_num(X_novel_test, nan=0.0)
    order = y_train_raw.argsort()
    X_train = X_train[order]
    y_train_raw = y_train_raw[order]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    X_train, y_train = remove_training_outliers(X_train, y_train)
    y_known_test = encoder.transform(y_known_test_raw)
    return X_train, y_train, X_known_test, y_known_test, X_novel_test


def prepare_inner_loco_folds(
    X_selection: np.ndarray,
    y_selection_raw: np.ndarray,
    inner_novel_class: object,
    scaler: str,
    seed: int,
    cv_folds: int,
) -> list[dict[str, object]]:
    known_mask = y_selection_raw != inner_novel_class
    X_known = X_selection[known_mask]
    y_known_raw = y_selection_raw[known_mask]
    X_inner_novel = X_selection[~known_mask]
    _, class_counts = np.unique(y_known_raw, return_counts=True)
    smallest_class = int(class_counts.min())
    if cv_folds > smallest_class:
        raise ValueError(
            f"cv_folds={cv_folds} is larger than the smallest known class "
            f"({smallest_class} samples) for inner class {inner_novel_class}."
        )

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    folds = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X_known, y_known_raw),
        start=1,
    ):
        prepared = preprocess_mnd_split(
            X_known[train_indices],
            y_known_raw[train_indices],
            X_known[validation_indices],
            y_known_raw[validation_indices],
            X_inner_novel,
            scaler,
            seed,
        )
        folds.append(
            {
                "fold": fold,
                "X_train": prepared[0],
                "y_train": prepared[1],
                "X_known_test": prepared[2],
                "y_known_test": prepared[3],
                "X_novel_test": prepared[4],
            }
        )
    return folds


def prepare_outer_class_grid(
    X_outer_train: np.ndarray,
    y_outer_train_raw: np.ndarray,
    outer_novel_class: object,
    scaler: str,
    seed: int,
    cv_folds: int,
) -> dict[str, list[dict[str, object]]]:
    selection_mask = y_outer_train_raw != outer_novel_class
    X_selection = X_outer_train[selection_mask]
    y_selection_raw = y_outer_train_raw[selection_mask]
    inner_novel_classes = np.unique(y_selection_raw)
    if len(inner_novel_classes) < 3:
        raise ValueError(
            "Strict nested LOCO grid search needs at least four classes: "
            "one outer novel class, one inner novel class, and at least two "
            "inner known classes."
        )
    return {
        str(inner_novel_class): prepare_inner_loco_folds(
            X_selection,
            y_selection_raw,
            inner_novel_class,
            scaler,
            seed,
            cv_folds,
        )
        for inner_novel_class in inner_novel_classes
    }


def make_model(configuration: dict[str, object], seed: int) -> LIM_NFST:
    return LIM_NFST(
        epsilon=float(configuration["epsilon"]),
        reference_size=float(configuration["reference_size"]),
        number_of_neighbors=int(configuration["neighbors"]),
        novelty_quantile=float(configuration["novelty_quantile"]),
        random_state=seed,
        use_rff=True,
        rff_components=int(configuration["rff_components"]),
        rff_gamma_multiplier=float(configuration["rff_gamma_multiplier"]),
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_open: np.ndarray,
    y_pred_closed: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred_open = np.asarray(y_pred_open, dtype=int)
    y_pred_closed = np.asarray(y_pred_closed, dtype=int)
    known_mask = y_true != NOVEL_LABEL
    closed_accuracy = (
        accuracy_score(y_true[known_mask], y_pred_closed[known_mask])
        if known_mask.any()
        else 0.0
    )
    closed_balanced_accuracy = (
        balanced_accuracy_score(y_true[known_mask], y_pred_closed[known_mask])
        if known_mask.any()
        else 0.0
    )

    is_novel_true = (y_true == NOVEL_LABEL).astype(int)
    is_novel_pred = (y_pred_open == NOVEL_LABEL).astype(int)
    novel_precision, novel_recall, novel_f1, _ = (
        precision_recall_fscore_support(
            is_novel_true,
            is_novel_pred,
            average="binary",
            zero_division=0,
        )
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred_open,
            average="macro",
            zero_division=0,
        )
    )
    mcc = (
        matthews_corrcoef(y_true, y_pred_open)
        if np.unique(y_true).size > 1
        else 0.0
    )
    return {
        "accuracy_with_novel": accuracy_score(y_true, y_pred_open),
        "accuracy_closed_world": closed_accuracy,
        "balanced_accuracy_with_novel": recall_score(
            y_true,
            y_pred_open,
            labels=np.unique(y_true),
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy_closed_world": closed_balanced_accuracy,
        "novel_detection_precision": novel_precision,
        "novel_detection_recall": novel_recall,
        "novel_detection_f1": novel_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mcc_with_novel": mcc,
    }


def evaluate_model(
    model: LIM_NFST,
    X_known_test: np.ndarray,
    y_known_test: np.ndarray,
    X_novel_test: np.ndarray,
) -> list[dict[str, object]]:
    y_novel = np.full(len(X_novel_test), NOVEL_LABEL, dtype=int)
    evaluation_sets = [
        ("known", X_known_test, y_known_test),
        ("novel", X_novel_test, y_novel),
        (
            "combined",
            np.vstack([X_known_test, X_novel_test]),
            np.concatenate([y_known_test, y_novel]),
        ),
    ]
    return [
        {
            "test_set": test_set,
            **calculate_metrics(
                y_true,
                model.predict_open(X_test, novel_label=NOVEL_LABEL),
                model.predict_closed(X_test),
            ),
        }
        for test_set, X_test, y_true in evaluation_sets
    ]


def evaluate_configuration(
    configuration: dict[str, object],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_known_test: np.ndarray,
    y_known_test: np.ndarray,
    X_novel_test: np.ndarray,
) -> tuple[list[dict[str, object]], float]:
    model = make_model(configuration, seed).fit(X_train, y_train)
    return (
        evaluate_model(model, X_known_test, y_known_test, X_novel_test),
        float(model.rff_gamma_),
    )


def fixed_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "scaler": args.scaler,
        "reference_size": args.reference_size,
        "neighbors": args.neighbors,
        "epsilon": args.epsilon,
        "use_rff": True,
        "rff_components": args.rff_components,
        "rff_gamma_mode": "scale_times_multiplier",
        "rff_gamma_multiplier": args.rff_gamma_multiplier,
        "threshold_mode": "reference_cloud_quantile",
        "novelty_quantile": args.novelty_quantile,
    }


def grid_configurations(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "scaler": scaler,
            "reference_size": reference_size,
            "neighbors": neighbors,
            "epsilon": args.epsilon,
            "use_rff": True,
            "rff_components": rff_components,
            "rff_gamma_mode": "scale_times_multiplier",
            "rff_gamma_multiplier": gamma_multiplier,
            "threshold_mode": "reference_cloud_quantile",
            "novelty_quantile": novelty_quantile,
        }
        for (
            scaler,
            reference_size,
            neighbors,
            rff_components,
            gamma_multiplier,
            novelty_quantile,
        ) in product(
            args.grid_scalers,
            args.grid_reference_sizes,
            args.grid_neighbors,
            args.grid_rff_components,
            args.grid_rff_gamma_multipliers,
            args.grid_novelty_quantiles,
        )
    ]


def configuration_name(configuration: dict[str, object]) -> str:
    return (
        f"scaler={configuration['scaler']}"
        f"__reference_size={number_to_name(configuration['reference_size'])}"
        f"__neighbors={configuration['neighbors']}"
        f"__rff_components={configuration['rff_components']}"
        "__rff_gamma_multiplier="
        f"{number_to_name(configuration['rff_gamma_multiplier'])}"
        "__novelty_quantile="
        f"{number_to_name(configuration['novelty_quantile'])}"
    )


def configuration_id(configuration: dict[str, object]) -> str:
    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def summarize_results(run_rows: list[dict[str, object]]) -> pd.DataFrame:
    runs = pd.DataFrame(run_rows, columns=FIXED_RUN_COLUMNS)
    if runs.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return (
        runs.groupby(["dataset", "model", "test_set"], sort=False)[
            METRIC_COLUMNS
        ]
        .mean()
        .reset_index()
        .reindex(columns=SUMMARY_COLUMNS)
    )


def summarize_grid_runs(
    run_frame: pd.DataFrame,
    dataset: str,
    outer_novel_class: object,
    configuration: dict[str, object],
) -> pd.DataFrame:
    model_name = f"{MODEL_NAME}[{configuration_name(configuration)}]"
    if run_frame.empty:
        return pd.DataFrame(columns=GRID_RESULT_COLUMNS)
    return (
        run_frame.groupby(
            ["dataset", "outer_novel_class", "model", "test_set"],
            sort=False,
        )[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .reindex(columns=GRID_RESULT_COLUMNS)
    )


def fixed_output_dir(
    args: argparse.Namespace,
    datasets: list[str],
    run_all: bool,
    version: str,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    dataset_name = "all" if run_all else datasets[0]
    configuration = fixed_configuration(args)
    return (
        WORKSPACE_ROOT
        / "results"
        / "lim-models_task2"
        / version
        / (
            f"evaluation__dataset={dataset_name}"
            f"__scaler={configuration['scaler']}"
            "__reference_size="
            f"{number_to_name(configuration['reference_size'])}"
            f"__neighbors={configuration['neighbors']}"
            f"__rff_components={configuration['rff_components']}"
            "__rff_gamma_multiplier="
            f"{number_to_name(configuration['rff_gamma_multiplier'])}"
            "__novelty_quantile="
            f"{number_to_name(configuration['novelty_quantile'])}"
        )
    )


def run_fixed_experiment(
    args: argparse.Namespace,
    datasets: list[str],
    seeds: list[int],
    run_all: bool,
) -> int:
    version = get_version()
    configuration = fixed_configuration(args)
    output_dir = fixed_output_dir(args, datasets, run_all, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "parameters.json",
        {
            "mode": "fixed RFF-REF-LIM MND evaluation",
            "lim_code_version": version,
            "datasets": datasets,
            "seeds": seeds,
            "test_size": TEST_SIZE,
            "mnd_protocol": "outer_LOCO_holdout_evaluation",
            **configuration,
        },
    )

    run_rows = []
    gamma_by_run = {}
    novel_classes_by_dataset = {}
    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)
        novel_classes = resolve_novel_classes(dataframe, args.novel_class)
        novel_classes_by_dataset[dataset] = [str(label) for label in novel_classes]

        for seed in seeds:
            for novel_class in novel_classes:
                (
                    X_train_raw,
                    y_train_raw,
                    X_known_test_raw,
                    y_known_test_raw,
                    X_novel_test_raw,
                    resolved_class,
                ) = split_outer_mnd_data(dataframe, novel_class, seed)
                prepared = preprocess_mnd_split(
                    X_train_raw,
                    y_train_raw,
                    X_known_test_raw,
                    y_known_test_raw,
                    X_novel_test_raw,
                    str(configuration["scaler"]),
                    seed,
                )
                evaluation_rows, rff_gamma = evaluate_configuration(
                    configuration,
                    seed,
                    *prepared,
                )
                for row in evaluation_rows:
                    run_rows.append(
                        {
                            "dataset": dataset,
                            "model": MODEL_NAME,
                            "seed": seed,
                            "novel_class": str(resolved_class),
                            **row,
                        }
                    )
                gamma_by_run[
                    f"{dataset}__novel_class={resolved_class}__seed={seed}"
                ] = rff_gamma

    results_by_seed = pd.DataFrame(run_rows, columns=FIXED_RUN_COLUMNS)
    results_by_seed.to_csv(
        output_dir / "results_by_seed.csv",
        index=False,
        float_format="%.8f",
    )
    summarize_results(run_rows).to_csv(
        output_dir / "results.csv",
        index=False,
        float_format="%.8f",
    )
    save_json(output_dir / "novel_classes.json", novel_classes_by_dataset)
    save_json(output_dir / "rff_gamma_by_run.json", gamma_by_run)

    print(f"Output: {output_dir.resolve()}")
    return 0


def grid_output_dir(args: argparse.Namespace, version: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return (
        WORKSPACE_ROOT
        / "results"
        / "lim-models_task2"
        / version
        / "rff"
    )


def save_grid_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    datasets: list[str],
    seeds: list[int],
    configurations: list[dict[str, object]],
    version: str,
) -> None:
    save_json(
        output_dir / "grid_parameters.json",
        {
            "mode": "RFF-REF-LIM MND strict nested LOCO grid search",
            "lim_code_version": version,
            "datasets": datasets,
            "seeds": seeds,
            "limit_override": args.limit,
            "test_size": TEST_SIZE,
            "use_rff": True,
            "scalers": args.grid_scalers,
            "reference_sizes": args.grid_reference_sizes,
            "neighbors": args.grid_neighbors,
            "epsilon": args.epsilon,
            "rff_components": args.grid_rff_components,
            "rff_gamma_mode": "scale_times_multiplier",
            "rff_gamma_multipliers": args.grid_rff_gamma_multipliers,
            "threshold_mode": "reference_cloud_quantile",
            "novelty_quantiles": args.grid_novelty_quantiles,
            "selection_metric": args.selection_metric,
            "cross_validation": "strict_nested_LOCO_StratifiedKFold",
            "cv_folds": args.cv_folds,
            "cv_repeats": len(seeds),
            "configurations_per_outer_novel_class": len(configurations),
            "outer_test_used_for_selection": False,
            "outer_novel_class_excluded_from_all_grid_selection": True,
            "mnd_selection_protocol": (
                "outer_novel_class_excluded_from_all_grid_selection"
            ),
        },
    )


def prepare_grid_fold_cache(
    dataframe: pd.DataFrame,
    outer_novel_class: object,
    scaler: str,
    seeds: list[int],
    cv_folds: int,
) -> dict[tuple[int, str], list[dict[str, object]]]:
    fold_cache = {}
    for seed in seeds:
        X_outer_train, y_outer_train, _, _ = split_outer_data(dataframe, seed)
        inner_folds = prepare_outer_class_grid(
            X_outer_train,
            y_outer_train,
            outer_novel_class,
            scaler,
            seed,
            cv_folds,
        )
        fold_cache.update(
            {
                (seed, inner_novel_class): folds
                for inner_novel_class, folds in inner_folds.items()
            }
        )
    return fold_cache


def evaluate_grid_configuration(
    dataset: str,
    outer_novel_class: object,
    configuration: dict[str, object],
    fold_cache: dict[tuple[int, str], list[dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    model_name = f"{MODEL_NAME}[{configuration_name(configuration)}]"
    run_rows = []
    gamma_by_run = {}
    for (seed, inner_novel_class), folds in fold_cache.items():
        for fold_data in folds:
            evaluation_rows, rff_gamma = evaluate_configuration(
                configuration,
                seed,
                fold_data["X_train"],
                fold_data["y_train"],
                fold_data["X_known_test"],
                fold_data["y_known_test"],
                fold_data["X_novel_test"],
            )
            for row in evaluation_rows:
                run_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "outer_novel_class": str(outer_novel_class),
                        "inner_novel_class": inner_novel_class,
                        "seed": seed,
                        "fold": fold_data["fold"],
                        **row,
                    }
                )
            gamma_by_run[
                "seed="
                f"{seed}__inner_novel_class={inner_novel_class}"
                f"__fold={fold_data['fold']}"
            ] = rff_gamma
    return run_rows, gamma_by_run


def save_configuration_artifacts(
    output_dir: Path,
    dataset: str,
    outer_novel_class: object,
    configuration: dict[str, object],
    run_frame: pd.DataFrame,
    gamma_by_run: dict[str, float],
    args: argparse.Namespace,
    version: str,
) -> None:
    configuration_dir = (
        output_dir
        / dataset
        / f"outer_novel_class={outer_novel_class}"
        / f"cfg_{configuration_id(configuration)}"
    )
    configuration_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        configuration_dir / "parameters.json",
        {
            "dataset": dataset,
            "outer_novel_class": str(outer_novel_class),
            "model": MODEL_NAME,
            "configuration": configuration_name(configuration),
            "configuration_id": configuration_id(configuration),
            "lim_code_version": version,
            "selection_metric": args.selection_metric,
            "cross_validation": "strict_nested_LOCO_StratifiedKFold",
            "cv_folds": args.cv_folds,
            "outer_test_used_for_selection": False,
            "outer_novel_class_excluded_from_all_grid_selection": True,
            "mnd_selection_protocol": (
                "outer_novel_class_excluded_from_all_grid_selection"
            ),
            **configuration,
        },
    )
    run_frame.to_csv(
        configuration_dir / "cv_results.csv",
        index=False,
        float_format="%.8f",
    )
    summary = (
        run_frame.groupby("test_set", sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(column).rstrip("_")
        for column in summary.columns
    ]
    summary.insert(0, "model", f"{MODEL_NAME}[{configuration_name(configuration)}]")
    summary.insert(0, "outer_novel_class", str(outer_novel_class))
    summary.insert(0, "dataset", dataset)
    summary.to_csv(
        configuration_dir / "cv_summary.csv",
        index=False,
        float_format="%.8f",
    )
    save_json(configuration_dir / "rff_gamma_by_cv_run.json", gamma_by_run)


def candidate_from_runs(
    dataset: str,
    outer_novel_class: object,
    configuration: dict[str, object],
    run_frame: pd.DataFrame,
    selection_metric: str,
) -> dict[str, object]:
    combined_runs = run_frame[run_frame["test_set"] == "combined"]
    metric_means = combined_runs[METRIC_COLUMNS].mean()
    selection_values = combined_runs[selection_metric]
    return {
        "dataset": dataset,
        "outer_novel_class": str(outer_novel_class),
        "model": MODEL_NAME,
        "configuration": configuration_name(configuration),
        **configuration,
        "selection_metric": selection_metric,
        "selection_metric_std": selection_values.std(ddof=1),
        "selection_validation_runs": len(combined_runs),
        **metric_means.to_dict(),
    }


def select_best_parameters(
    candidate_frame: pd.DataFrame,
    selection_metric: str,
    cv_folds: int,
    cv_repeats: int,
    version: str,
) -> pd.DataFrame:
    best_parameters = []
    ranking_columns = [
        selection_metric,
        "selection_metric_std",
        "reference_size",
        "neighbors",
        "rff_components",
        "rff_gamma_multiplier",
        "novelty_quantile",
        "scaler",
    ]
    ranking_ascending = [False, True, True, True, True, True, True, True]
    for (dataset, outer_novel_class), candidates in candidate_frame.groupby(
        ["dataset", "outer_novel_class"],
        sort=False,
    ):
        best = candidates.sort_values(
            ranking_columns,
            ascending=ranking_ascending,
            kind="stable",
        ).iloc[0]
        best_parameters.append(
            {
                "dataset": dataset,
                "outer_novel_class": outer_novel_class,
                "scaler": best["scaler"],
                "reference_size": best["reference_size"],
                "neighbors": int(best["neighbors"]),
                "epsilon": best["epsilon"],
                "use_rff": True,
                "rff_components": int(best["rff_components"]),
                "rff_gamma_mode": "scale_times_multiplier",
                "rff_gamma_multiplier": best["rff_gamma_multiplier"],
                "threshold_mode": "reference_cloud_quantile",
                "novelty_quantile": best["novelty_quantile"],
                "selection_metric": selection_metric,
                "validation_score": best[selection_metric],
                "validation_score_std": best["selection_metric_std"],
                "cross_validation": "strict_nested_LOCO_StratifiedKFold",
                "cv_folds": cv_folds,
                "cv_repeats": cv_repeats,
                "mnd_selection_protocol": (
                    "outer_novel_class_excluded_from_all_grid_selection"
                ),
                "lim_code_version": version,
            }
        )
    return pd.DataFrame(best_parameters, columns=BEST_PARAMETER_COLUMNS)


def report_grid_progress(
    dataset: str,
    outer_novel_class: object,
    configuration_index: int,
    configuration_total: int,
) -> None:
    print(
        f"Tuning dataset={dataset} "
        f"outer_novel_class={outer_novel_class} "
        f"configuration={configuration_index}/{configuration_total}",
        flush=True,
    )


def run_grid_search(
    args: argparse.Namespace,
    datasets: list[str],
    seeds: list[int],
    run_all: bool,
) -> int:
    version = get_version()
    configurations = grid_configurations(args)
    configurations_by_scaler = {
        scaler: [
            configuration
            for configuration in configurations
            if configuration["scaler"] == scaler
        ]
        for scaler in args.grid_scalers
    }
    output_dir = grid_output_dir(args, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_grid_manifest(
        output_dir,
        args,
        datasets,
        seeds,
        configurations,
        version,
    )
    print(
        f"Grid search: {len(configurations)} configurations per "
        "outer novel class.",
        flush=True,
    )

    grid_result_rows = []
    candidates = []
    novel_classes_by_dataset = {}
    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)
        outer_novel_classes = resolve_novel_classes(dataframe, args.novel_class)
        novel_classes_by_dataset[dataset] = [
            str(label) for label in outer_novel_classes
        ]

        for outer_novel_class in outer_novel_classes:
            configuration_index = 0
            for scaler, scaler_configurations in configurations_by_scaler.items():
                fold_cache = prepare_grid_fold_cache(
                    dataframe,
                    outer_novel_class,
                    scaler,
                    seeds,
                    args.cv_folds,
                )
                for configuration in scaler_configurations:
                    configuration_index += 1
                    if (
                        configuration_index == 1
                        or configuration_index % 100 == 0
                        or configuration_index == len(configurations)
                    ):
                        report_grid_progress(
                            dataset,
                            outer_novel_class,
                            configuration_index,
                            len(configurations),
                        )
                    run_rows, gamma_by_run = evaluate_grid_configuration(
                        dataset,
                        outer_novel_class,
                        configuration,
                        fold_cache,
                    )
                    run_frame = pd.DataFrame(
                        run_rows,
                        columns=GRID_RUN_COLUMNS,
                    )
                    save_configuration_artifacts(
                        output_dir,
                        dataset,
                        outer_novel_class,
                        configuration,
                        run_frame,
                        gamma_by_run,
                        args,
                        version,
                    )
                    grid_result_rows.extend(
                        summarize_grid_runs(
                            run_frame,
                            dataset,
                            outer_novel_class,
                            configuration,
                        ).to_dict("records")
                    )
                    candidates.append(
                        candidate_from_runs(
                            dataset,
                            outer_novel_class,
                            configuration,
                            run_frame,
                            args.selection_metric,
                        )
                    )
            print(
                f"Completed dataset={dataset} "
                f"outer_novel_class={outer_novel_class} "
                f"configurations={len(configurations)}",
                flush=True,
            )

    grid_results = pd.DataFrame(grid_result_rows, columns=GRID_RESULT_COLUMNS)
    grid_results.to_csv(
        output_dir / "grid_search_results.csv",
        index=False,
        float_format="%.8f",
    )
    candidate_frame = pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)
    candidate_frame.to_csv(
        output_dir / "grid_candidates.csv",
        index=False,
        float_format="%.8f",
    )
    best_parameters = select_best_parameters(
        candidate_frame,
        args.selection_metric,
        args.cv_folds,
        len(seeds),
        version,
    )
    best_parameters.to_csv(
        output_dir / "best_parameters.csv",
        index=False,
        float_format="%.8f",
    )
    save_json(output_dir / "novel_classes.json", novel_classes_by_dataset)

    print(f"Output: {output_dir.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    datasets, seeds, run_all = get_datasets_and_seeds(args)
    if args.grid_search:
        return run_grid_search(args, datasets, seeds, run_all)
    return run_fixed_experiment(args, datasets, seeds, run_all)


if __name__ == "__main__":
    raise SystemExit(main())
