"""Run and tune LIM-NFST for multiclass novelty detection (MND)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent

sys.path.insert(0, str(CLASSIFIER_ROOT))

from limnfst.datasets import DATASETS, load_dataset
from limnfst.preprocessing import (
    SCALERS,
    make_scaler,
    remove_training_outliers,
)
from limnfst.models import LIM_NFST


TEST_SIZE = 0.20
NOVEL_LABEL = -1
LIM_VERSION_NAME = "rff_ref_lim_multiclass_novelty_v1"
RFF_REF_LIM_NAME = "RFF-REF-LIM"

DEFAULT_LIMITS = {
    "BoT_IoT": 1000,
    "CIC_IoT2023": 1000,
    "ToN_IoT": 1000,
    "UNSW_NB15": 1000,
    "IoTID20": 2000,
    "N_BaIoT": 1000,
    "Edge_IIoTset": 1000,
    "5G_NIDD": 1000,
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_GRID_SCALERS = list(SCALERS)
DEFAULT_GRID_REFERENCE_SIZES = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_GRID_NEIGHBORS = [1, 3, 5, 7, 9]
DEFAULT_GRID_NOVELTY_QUANTILES = [0.90, 0.95, 0.99]
DEFAULT_GRID_RFF_COMPONENTS = [128, 256, 512]
DEFAULT_GRID_RFF_GAMMA_MULTIPLIERS = [0.10, 1.0, 10.0]

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
RESULT_COLUMNS = ["dataset", "model", "test_set", *METRIC_COLUMNS]
SELECTION_METRICS = [
    "novel_detection_f1",
    "macro_f1",
    "accuracy_with_novel",
    "balanced_accuracy_with_novel",
    "mcc_with_novel",
]


def get_lim_version():
    source_files = [
        CLASSIFIER_ROOT / "limnfst" / "models.py",
        CLASSIFIER_ROOT / "limnfst" / "nfst.py",
        CLASSIFIER_ROOT / "limnfst" / "novelty.py",
        CLASSIFIER_ROOT / "limnfst" / "datasets.py",
        CLASSIFIER_ROOT / "limnfst" / "preprocessing.py",
    ]
    code_hash = hashlib.sha256()
    for source_file in source_files:
        code_hash.update(source_file.name.encode("utf-8"))
        code_hash.update(source_file.read_bytes())
    return f"{LIM_VERSION_NAME}_{code_hash.hexdigest()[:8]}"


def number_to_name(value):
    return format(float(value), ".6g").replace(".", "p")


def safe_name(value):
    return "".join(
        char if char.isalnum() or char in "-_" else "-"
        for char in str(value)
    )


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def repository_results_root():
    """Use the existing results collection inside this repository."""
    standard_root = CLASSIFIER_ROOT / "results"
    imported_root = CLASSIFIER_ROOT / "limnfst" / "results" / "results"
    if standard_root.exists():
        return standard_root
    if imported_root.exists():
        return imported_root
    return standard_root


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate or tune RFF-REF-LIM for LOCO multiclass novelty "
            "detection. RFF is enabled by default."
        )
    )
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--novel-class",
        default=None,
        help="Held-out label or zero-based class index (default: first class).",
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
    parser.add_argument("--novelty-quantile", type=float, default=0.95)
    rff_group = parser.add_mutually_exclusive_group()
    rff_group.add_argument(
        "--use-rff",
        action="store_true",
        dest="use_rff",
        help="Use RFF-REF-LIM (default).",
    )
    rff_group.add_argument(
        "--no-rff",
        action="store_false",
        dest="use_rff",
        help="Run the non-RFF LIM ablation.",
    )
    parser.set_defaults(use_rff=True)
    parser.add_argument("--rff-components", type=int, default=256)
    parser.add_argument("--rff-gamma-multiplier", type=float, default=1.0)

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
        "--grid-novelty-quantiles",
        nargs="+",
        type=float,
        default=DEFAULT_GRID_NOVELTY_QUANTILES,
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
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="novel_detection_f1",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoints and rerun every requested configuration.",
    )
    return parser.parse_args()


def get_datasets_and_seeds(args):
    run_all = args.all_datasets or args.dataset is None
    if run_all and args.limit is not None:
        raise ValueError("--limit can only be used with one --dataset.")
    datasets = list(DATASETS) if run_all else [args.dataset]
    if args.seeds is not None:
        seeds = args.seeds
    elif run_all:
        seeds = DEFAULT_SEEDS
    else:
        seeds = [args.seed]
    return datasets, seeds, run_all


def calculate_metrics(y_true, y_pred_open, y_pred_closed):
    """Calculate open-world, closed-world, and binary novelty metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_open = np.asarray(y_pred_open, dtype=int)
    y_pred_closed = np.asarray(y_pred_closed, dtype=int)
    known_mask = y_true != NOVEL_LABEL

    if known_mask.any():
        closed_accuracy = accuracy_score(
            y_true[known_mask], y_pred_closed[known_mask]
        )
        closed_balanced_accuracy = balanced_accuracy_score(
            y_true[known_mask], y_pred_closed[known_mask]
        )
    else:
        closed_accuracy = 0.0
        closed_balanced_accuracy = 0.0

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
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_open)) > 1:
        mcc = matthews_corrcoef(y_true, y_pred_open)
    else:
        mcc = 0.0

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


def average_results(run_results, include_overall=True):
    """Average seed results in the same style as lim-models.py."""
    if not run_results:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    runs = pd.DataFrame(run_results)
    results = (
        runs.groupby(["dataset", "model", "test_set"], sort=False)[
            METRIC_COLUMNS
        ]
        .mean()
        .reset_index()
    )
    if include_overall:
        overall = (
            results.groupby(["model", "test_set"], sort=False)[METRIC_COLUMNS]
            .mean()
            .reset_index()
        )
        overall.insert(0, "dataset", "OVERALL")
        results = pd.concat([results, overall], ignore_index=True)
    return results[RESULT_COLUMNS]


def resolve_novel_class(y_raw, requested_class):
    """Resolve an exact raw label first, then a zero-based class index."""
    classes = np.unique(y_raw)
    if len(classes) < 2:
        raise ValueError("MND requires at least two classes.")
    if requested_class is None:
        return classes[0]
    matches = [label for label in classes if str(label) == str(requested_class)]
    if matches:
        return matches[0]
    try:
        class_index = int(requested_class)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Unknown novel class {requested_class!r}. "
            f"Available classes: {classes.tolist()}"
        ) from error
    if not 0 <= class_index < len(classes):
        raise ValueError(
            f"Novel class index {class_index} is outside "
            f"[0, {len(classes) - 1}]."
        )
    return classes[class_index]


def split_loco_data(dataframe, requested_novel_class):
    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]
    novel_class = resolve_novel_class(y_raw, requested_novel_class)
    novel_mask = y_raw == novel_class
    return X[~novel_mask], y_raw[~novel_mask], X[novel_mask], novel_class


def fit_preprocessor(X_train, evaluation_sets, scaler, seed):
    """Fit imputation and scaling on known training samples only."""
    X_train = np.asarray(X_train, dtype=np.float64).copy()
    transformed_sets = [
        np.asarray(values, dtype=np.float64).copy()
        for values in evaluation_sets
    ]
    X_train[np.isinf(X_train)] = np.nan
    for values in transformed_sets:
        values[np.isinf(values)] = np.nan

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    transformed_sets = [imputer.transform(values) for values in transformed_sets]
    if scaler != "None":
        fitted_scaler = make_scaler(scaler, random_state=seed)
        X_train = fitted_scaler.fit_transform(X_train)
        transformed_sets = [
            fitted_scaler.transform(values) for values in transformed_sets
        ]

    X_train = np.nan_to_num(X_train, nan=0.0)
    transformed_sets = [
        np.nan_to_num(values, nan=0.0) for values in transformed_sets
    ]
    return X_train, transformed_sets


def encode_and_clean_training(X_train, y_train_raw):
    order = y_train_raw.argsort()
    X_train = X_train[order]
    y_train_raw = y_train_raw[order]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    X_train, y_train = remove_training_outliers(X_train, y_train)
    return X_train, y_train, encoder


def prepare_fixed_data(dataframe, requested_novel_class, scaler, seed):
    X_known, y_known_raw, X_novel, novel_class = split_loco_data(
        dataframe, requested_novel_class
    )
    X_train, X_test_known, y_train_raw, y_test_known_raw = train_test_split(
        X_known,
        y_known_raw,
        test_size=TEST_SIZE,
        stratify=y_known_raw,
        random_state=seed,
    )
    X_train, (X_test_known, X_test_novel) = fit_preprocessor(
        X_train, [X_test_known, X_novel], scaler, seed
    )
    X_train, y_train, encoder = encode_and_clean_training(X_train, y_train_raw)
    return (
        X_train,
        y_train,
        X_test_known,
        encoder.transform(y_test_known_raw),
        X_test_novel,
        novel_class,
    )


def prepare_grid_data(
    dataframe, requested_novel_class, scaler, seed, validation_size
):
    """Build inner validation sets while leaving outer test data unused."""
    X_known, y_known_raw, X_novel, novel_class = split_loco_data(
        dataframe, requested_novel_class
    )
    X_outer_train, _, y_outer_train, _ = train_test_split(
        X_known,
        y_known_raw,
        test_size=TEST_SIZE,
        stratify=y_known_raw,
        random_state=seed,
    )
    X_train, X_val_known, y_train_raw, y_val_known_raw = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=validation_size,
        stratify=y_outer_train,
        random_state=seed,
    )
    X_val_novel, _ = train_test_split(
        X_novel, train_size=validation_size, random_state=seed
    )
    X_train, (X_val_known, X_val_novel) = fit_preprocessor(
        X_train, [X_val_known, X_val_novel], scaler, seed
    )
    X_train, y_train, encoder = encode_and_clean_training(X_train, y_train_raw)
    return (
        X_train,
        y_train,
        X_val_known,
        encoder.transform(y_val_known_raw),
        X_val_novel,
        novel_class,
    )


def make_model(
    args,
    seed,
    reference_size,
    neighbors,
    novelty_quantile,
    rff_components=None,
    gamma_multiplier=None,
):
    return LIM_NFST(
        epsilon=args.epsilon,
        reference_size=reference_size,
        number_of_neighbors=neighbors,
        novelty_quantile=novelty_quantile,
        random_state=seed,
        use_rff=args.use_rff,
        rff_components=(
            rff_components if args.use_rff else args.rff_components
        ),
        rff_gamma_multiplier=(
            gamma_multiplier if args.use_rff else args.rff_gamma_multiplier
        ),
    )


def predict_open_and_closed(model, X):
    """Predict with the threshold of the predicted class, not a global max."""
    scores = model.reference_scores(X)
    predicted_indices = np.argmin(scores, axis=1)
    rows = np.arange(len(scores))
    y_pred_closed = np.asarray(model.classes_[predicted_indices], dtype=int)
    is_novel = (
        scores[rows, predicted_indices]
        > model.reference_thresholds_[predicted_indices]
    )
    y_pred_open = np.where(is_novel, NOVEL_LABEL, y_pred_closed)
    return y_pred_open.astype(int), y_pred_closed


def evaluate_model(
    model,
    dataset,
    model_name,
    seed,
    X_known,
    y_known,
    X_novel,
):
    y_known_open, y_known_closed = predict_open_and_closed(model, X_known)
    X_combined = np.vstack([X_known, X_novel])
    y_combined = np.concatenate(
        [y_known, np.full(len(X_novel), NOVEL_LABEL, dtype=int)]
    )
    y_combined_open, y_combined_closed = predict_open_and_closed(
        model, X_combined
    )
    rows = []
    for test_set, y_true, y_open, y_closed in [
        ("known_only", y_known, y_known_open, y_known_closed),
        (
            "with_novel_class",
            y_combined,
            y_combined_open,
            y_combined_closed,
        ),
    ]:
        row = {
            "dataset": dataset,
            "model": model_name,
            "test_set": test_set,
            "seed": seed,
        }
        row.update(calculate_metrics(y_true, y_open, y_closed))
        rows.append(row)
    return rows


def validate_fixed_arguments(args):
    if not 0.0 < args.reference_size <= 0.30:
        raise ValueError("--reference-size must be in (0, 0.30].")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least one.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than zero.")
    if not 0.0 < args.novelty_quantile < 1.0:
        raise ValueError("--novelty-quantile must be between zero and one.")
    if args.use_rff and args.rff_components < 1:
        raise ValueError("--rff-components must be at least one.")
    if args.use_rff and args.rff_gamma_multiplier <= 0.0:
        raise ValueError("--rff-gamma-multiplier must be greater than zero.")


def fixed_output_dir(args, datasets, run_all, version):
    if args.output_dir is not None:
        return args.output_dir
    dataset_name = "all" if run_all else datasets[0]
    folder_name = (
        f"evaluation__dataset={dataset_name}__scaler={args.scaler}"
        f"__reference_size={number_to_name(args.reference_size)}"
        f"__neighbors={args.neighbors}"
        f"__novelty_quantile={number_to_name(args.novelty_quantile)}"
    )
    if args.novel_class is not None:
        folder_name += f"__novel_class={safe_name(args.novel_class)}"
    if args.use_rff:
        folder_name += (
            f"__rff_components={args.rff_components}"
            f"__rff_gamma_multiplier="
            f"{number_to_name(args.rff_gamma_multiplier)}"
        )
    return (
        repository_results_root()
        / "lim-models-multiclass-novelty"
        / version
        / folder_name
    )


def run_fixed_experiment(args, datasets, seeds, run_all):
    validate_fixed_arguments(args)
    version = get_lim_version()
    output_dir = fixed_output_dir(args, datasets, run_all, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "RFF_LIM_NFST" if args.use_rff else "LIM_NFST"
    manifest = {
        "mode": f"fixed {model_name} MND evaluation",
        "model_display_name": (
            RFF_REF_LIM_NAME if args.use_rff else "LIM-NFST"
        ),
        "model_implementation": "LIM_NFST",
        "lim_code_version": version,
        "datasets": datasets,
        "seeds": seeds,
        "novel_class": args.novel_class,
        "novel_class_default": "first sorted class",
        "scaler": args.scaler,
        "reference_size": args.reference_size,
        "neighbors": args.neighbors,
        "epsilon": args.epsilon,
        "novelty_quantile": args.novelty_quantile,
        "use_rff": args.use_rff,
        "rff_components": args.rff_components if args.use_rff else None,
        "rff_gamma_mode": "scale_times_multiplier" if args.use_rff else None,
        "rff_gamma_multiplier": (
            args.rff_gamma_multiplier if args.use_rff else None
        ),
    }
    save_json(output_dir / "parameters.json", manifest)

    print(f"Mode       : fixed {model_name} MND evaluation")
    print(
        "Main model : "
        f"{RFF_REF_LIM_NAME if args.use_rff else 'LIM-NFST ablation'}"
    )
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Novel class: {args.novel_class or 'first sorted class'}")
    print(f"Scaler     : {args.scaler}")
    print(f"Reference  : {args.reference_size}")
    print(f"Neighbors  : {args.neighbors}")
    print(f"Quantile   : {args.novelty_quantile}")
    print(f"Use RFF    : {args.use_rff}")
    if args.use_rff:
        print(f"RFF dims   : {args.rff_components}")
        print(f"Gamma mult : {args.rff_gamma_multiplier}")
    print(f"Output     : {output_dir.resolve()}")

    run_results = []
    errors = []
    gamma_by_run = {}
    resolved_novel_classes = {}
    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)
        for seed in seeds:
            try:
                prepared = prepare_fixed_data(
                    dataframe, args.novel_class, args.scaler, seed
                )
                X_train, y_train, X_known, y_known, X_novel, novel_class = (
                    prepared
                )
                resolved_novel_classes[dataset] = novel_class
                model = make_model(
                    args,
                    seed,
                    args.reference_size,
                    args.neighbors,
                    args.novelty_quantile,
                    args.rff_components,
                    args.rff_gamma_multiplier,
                )
                model.fit(X_train, y_train)
                result_rows = evaluate_model(
                    model,
                    dataset,
                    model_name,
                    seed,
                    X_known,
                    y_known,
                    X_novel,
                )
                run_results.extend(result_rows)
                if args.use_rff:
                    gamma_by_run[f"{dataset}__seed={seed}"] = model.rff_gamma_
                combined = result_rows[1]
                print(
                    f"dataset={dataset} seed={seed} model={model_name} "
                    f"novel_class={novel_class} "
                    f"accuracy={combined['accuracy_with_novel']:.4f} "
                    f"novel_f1={combined['novel_detection_f1']:.4f}"
                )
            except Exception as error:
                errors.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"FAILED dataset={dataset} seed={seed}: {error}")

    results = average_results(run_results)
    results.to_csv(
        output_dir / "results.csv", index=False, float_format="%.8f"
    )
    save_json(output_dir / "novel_classes.json", resolved_novel_classes)
    if errors:
        save_json(output_dir / "errors.json", errors)
    if gamma_by_run:
        save_json(output_dir / "rff_gamma_by_run.json", gamma_by_run)
    print("\nFinal results:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


def validate_grid_arguments(args):
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between zero and one.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than zero.")
    if any(
        value <= 0.0 or value > 0.30
        for value in args.grid_reference_sizes
    ):
        raise ValueError("Grid reference sizes must be in (0, 0.30].")
    if any(value < 1 for value in args.grid_neighbors):
        raise ValueError("Grid neighbors must be at least one.")
    if any(
        value <= 0.0 or value >= 1.0
        for value in args.grid_novelty_quantiles
    ):
        raise ValueError("Grid novelty quantiles must be between zero and one.")
    if args.use_rff and any(
        value < 1 for value in args.grid_rff_components
    ):
        raise ValueError("Grid RFF components must be at least one.")
    if args.use_rff and any(
        value <= 0.0 for value in args.grid_rff_gamma_multipliers
    ):
        raise ValueError(
            "Grid RFF gamma multipliers must be greater than zero."
        )


def configuration_name(config, use_rff):
    scaler, reference_size, neighbors, quantile, components, gamma = config
    name = (
        f"scaler={scaler}"
        f"__reference_size={number_to_name(reference_size)}"
        f"__neighbors={neighbors}"
        f"__novelty_quantile={number_to_name(quantile)}"
    )
    if use_rff:
        name += (
            f"__rff_components={components}"
            f"__rff_gamma_multiplier={number_to_name(gamma)}"
        )
    return name


def configuration_payload(
    config,
    use_rff,
    epsilon,
    novel_class,
    validation_size,
):
    scaler, reference_size, neighbors, quantile, components, gamma = config
    return {
        "scaler": scaler,
        "reference_size": float(reference_size),
        "neighbors": int(neighbors),
        "novelty_quantile": float(quantile),
        "epsilon": float(epsilon),
        "use_rff": bool(use_rff),
        "rff_components": int(components) if use_rff else None,
        "rff_gamma_multiplier": float(gamma) if use_rff else None,
        "novel_class": None if novel_class is None else str(novel_class),
        "validation_size": float(validation_size),
    }


def configuration_id(
    config,
    use_rff,
    epsilon,
    novel_class,
    validation_size,
):
    payload = configuration_payload(
        config,
        use_rff,
        epsilon,
        novel_class,
        validation_size,
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"cfg_{digest}"


def config_from_parameters(parameters):
    return (
        parameters["scaler"],
        float(parameters["reference_size"]),
        int(parameters["neighbors"]),
        float(parameters["novelty_quantile"]),
        (
            int(parameters["rff_components"])
            if parameters.get("use_rff")
            else None
        ),
        (
            float(parameters["rff_gamma_multiplier"])
            if parameters.get("use_rff")
            else None
        ),
    )


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_without_overwrite(source, destination):
    if destination.exists():
        if file_digest(source) != file_digest(destination):
            raise FileExistsError(
                f"Refusing to overwrite different file: {destination}"
            )
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_digest(source) != file_digest(destination):
        destination.unlink(missing_ok=True)
        raise OSError(f"Verification failed while copying {source}")
    return True


def migrate_legacy_grid_results(legacy_dir, output_dir, validation_size):
    """Copy legacy long-name configs into short, collision-safe directories."""
    legacy_dir = legacy_dir.resolve()
    output_dir = output_dir.resolve()
    if not legacy_dir.exists() or legacy_dir == output_dir:
        return {"configurations": 0, "files_copied": 0}

    configurations = 0
    files_copied = 0
    for parameter_path in legacy_dir.glob("*/*/parameters.json"):
        parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
        config = config_from_parameters(parameters)
        resolved_novel_class = parameters.get("resolved_novel_class")
        effective_validation_size = float(
            parameters.get("validation_size", validation_size)
        )
        short_name = configuration_id(
            config,
            bool(parameters.get("use_rff")),
            float(parameters["epsilon"]),
            resolved_novel_class,
            effective_validation_size,
        )
        destination = output_dir / parameters["dataset"] / short_name
        destination.mkdir(parents=True, exist_ok=True)

        augmented = dict(parameters)
        augmented["configuration_id"] = short_name
        augmented["configuration_name"] = parameter_path.parent.name
        augmented["validation_size"] = effective_validation_size
        destination_parameters = destination / "parameters.json"
        if destination_parameters.exists():
            existing = json.loads(
                destination_parameters.read_text(encoding="utf-8")
            )
            if existing.get("configuration_id") != short_name:
                raise FileExistsError(
                    f"Configuration collision: {destination_parameters}"
                )
        else:
            save_json(destination_parameters, augmented)
            files_copied += 1

        for source_file in parameter_path.parent.iterdir():
            if source_file.is_file() and source_file.name != "parameters.json":
                if copy_without_overwrite(
                    source_file,
                    destination / source_file.name,
                ):
                    files_copied += 1
        configurations += 1

    summary = {
        "source": str(legacy_dir),
        "destination": str(output_dir),
        "configurations": configurations,
        "files_copied": files_copied,
    }
    save_json(output_dir / "legacy_import.json", summary)
    return summary


def grid_configurations(args):
    rff_settings = (
        list(
            product(
                args.grid_rff_components,
                args.grid_rff_gamma_multipliers,
            )
        )
        if args.use_rff
        else [(None, None)]
    )
    return [
        (scaler, reference_size, neighbors, quantile, components, gamma)
        for scaler, reference_size, neighbors, quantile, (components, gamma)
        in product(
            args.grid_scalers,
            args.grid_reference_sizes,
            args.grid_neighbors,
            args.grid_novelty_quantiles,
            rff_settings,
        )
    ]


SEED_RESULT_COLUMNS = ["dataset", "model", "test_set", "seed", *METRIC_COLUMNS]


def save_csv_atomic(frame, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, float_format="%.8f")
    temporary_path.replace(path)


def summarize_configuration(
    results,
    dataset,
    model_name,
    display_name,
    novel_class,
    config,
    args,
):
    combined = results[results["test_set"] == "with_novel_class"]
    if combined.empty:
        return None, []
    scaler, reference_size, neighbors, quantile, components, gamma = config
    metrics = combined.iloc[0]
    candidate = {
        "dataset": dataset,
        "novel_class": novel_class,
        "scaler": scaler,
        "reference_size": reference_size,
        "neighbors": neighbors,
        "novelty_quantile": quantile,
        "epsilon": args.epsilon,
        "use_rff": args.use_rff,
        "rff_components": components,
        "rff_gamma_multiplier": gamma,
        **{metric: metrics[metric] for metric in METRIC_COLUMNS},
    }
    report_rows = []
    for _, result in results.iterrows():
        report_rows.append(
            {
                "dataset": dataset,
                "model": f"{model_name}[{display_name}]",
                "test_set": result["test_set"],
                **{metric: result[metric] for metric in METRIC_COLUMNS},
            }
        )
    return candidate, report_rows


def load_completed_configuration(config_dir, seeds):
    results_path = config_dir / "results.csv"
    parameters_path = config_dir / "parameters.json"
    if not results_path.exists() or not parameters_path.exists():
        return None
    if (config_dir / "errors.json").exists():
        return None
    try:
        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        stored_seeds = [int(seed) for seed in parameters.get("seeds", [])]
        if stored_seeds != [int(seed) for seed in seeds]:
            return None
        results = pd.read_csv(results_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    required_columns = set(RESULT_COLUMNS)
    if not required_columns.issubset(results.columns):
        return None
    test_sets = set(results["test_set"].astype(str))
    if test_sets != {"known_only", "with_novel_class"}:
        return None
    if results[METRIC_COLUMNS].isna().any().any():
        return None
    return results[RESULT_COLUMNS]


def load_seed_checkpoint(config_dir, dataset, model_name, seeds):
    checkpoint_path = config_dir / "seed_results.csv"
    if not checkpoint_path.exists():
        return [], set()
    try:
        checkpoint = pd.read_csv(checkpoint_path)
    except (OSError, ValueError):
        return [], set()
    if not set(SEED_RESULT_COLUMNS).issubset(checkpoint.columns):
        return [], set()
    checkpoint = checkpoint[
        (checkpoint["dataset"] == dataset)
        & (checkpoint["model"] == model_name)
        & checkpoint["seed"].isin(seeds)
    ].copy()
    completed_seeds = set()
    for seed in seeds:
        seed_rows = checkpoint[checkpoint["seed"] == seed]
        if set(seed_rows["test_set"].astype(str)) == {
            "known_only",
            "with_novel_class",
        }:
            completed_seeds.add(int(seed))
    checkpoint = checkpoint[checkpoint["seed"].isin(completed_seeds)]
    return checkpoint[SEED_RESULT_COLUMNS].to_dict("records"), completed_seeds


def save_configuration_result(
    args,
    dataset,
    model_name,
    version,
    seeds,
    config,
    output_dir,
    prepared_by_seed,
    novel_class,
    all_errors,
):
    scaler, reference_size, neighbors, quantile, components, gamma = config
    display_name = configuration_name(config, args.use_rff)
    short_name = configuration_id(
        config,
        args.use_rff,
        args.epsilon,
        novel_class,
        args.validation_size,
    )
    config_dir = output_dir / dataset / short_name
    config_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "configuration_id": short_name,
        "configuration_name": display_name,
        "dataset": dataset,
        "model": model_name,
        "novel_class": args.novel_class,
        "resolved_novel_class": novel_class,
        "scaler": scaler,
        "reference_size": reference_size,
        "neighbors": neighbors,
        "novelty_quantile": quantile,
        "epsilon": args.epsilon,
        "use_rff": args.use_rff,
        "rff_components": components,
        "rff_gamma_mode": "scale_times_multiplier" if args.use_rff else None,
        "rff_gamma_multiplier": gamma,
        "seeds": seeds,
        "validation_size": args.validation_size,
        "selection_metric": args.selection_metric,
        "selection_test_set": "with_novel_class",
        "lim_code_version": version,
        "outer_test_used_for_selection": False,
    }

    if not args.force:
        completed_results = load_completed_configuration(config_dir, seeds)
        if completed_results is not None:
            print(f"RESUME skip dataset={dataset} configuration={short_name}")
            return summarize_configuration(
                completed_results,
                dataset,
                model_name,
                display_name,
                novel_class,
                config,
                args,
            )

    save_json(config_dir / "parameters.json", parameters)
    runs, completed_seeds = (
        ([], set())
        if args.force
        else load_seed_checkpoint(config_dir, dataset, model_name, seeds)
    )
    gamma_path = config_dir / "rff_gamma_by_seed.json"
    if not args.force and gamma_path.exists():
        try:
            gamma_by_seed = json.loads(gamma_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            gamma_by_seed = {}
    else:
        gamma_by_seed = {}
    errors = []

    for seed in seeds:
        if int(seed) in completed_seeds:
            print(
                f"RESUME seed dataset={dataset} configuration={short_name} "
                f"seed={seed}"
            )
            continue
        prepared = prepared_by_seed.get(seed)
        if prepared is None:
            errors.append({"seed": seed, "error": "Preprocessing failed."})
            continue
        X_train, y_train, X_known, y_known, X_novel, _ = prepared
        try:
            model = make_model(
                args,
                seed,
                reference_size,
                neighbors,
                quantile,
                components,
                gamma,
            )
            model.fit(X_train, y_train)
            runs.extend(
                evaluate_model(
                    model,
                    dataset,
                    model_name,
                    seed,
                    X_known,
                    y_known,
                    X_novel,
                )
            )
            completed_seeds.add(int(seed))
            save_csv_atomic(
                pd.DataFrame(runs, columns=SEED_RESULT_COLUMNS),
                config_dir / "seed_results.csv",
            )
            if args.use_rff:
                gamma_by_seed[str(seed)] = model.rff_gamma_
                save_json(gamma_path, gamma_by_seed)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            error_row = {"seed": seed, "error": message}
            errors.append(error_row)
            all_errors.append(
                {
                    "dataset": dataset,
                    "configuration_id": short_name,
                    "scaler": scaler,
                    "reference_size": reference_size,
                    "neighbors": neighbors,
                    "novelty_quantile": quantile,
                    "rff_components": components,
                    "rff_gamma_multiplier": gamma,
                    **error_row,
                }
            )

    results = average_results(runs, include_overall=False)
    save_csv_atomic(results, config_dir / "results.csv")
    errors_path = config_dir / "errors.json"
    if errors:
        save_json(errors_path, errors)
    elif errors_path.exists():
        errors_path.unlink()
    if completed_seeds != {int(seed) for seed in seeds}:
        print(f"FAILED dataset={dataset} configuration={short_name}")
        return None, []

    candidate, report_rows = summarize_configuration(
        results,
        dataset,
        model_name,
        display_name,
        novel_class,
        config,
        args,
    )
    message = (
        f"dataset={dataset} config={short_name} scaler={scaler} "
        f"ref={reference_size:g} k={neighbors} quantile={quantile:g}"
    )
    if args.use_rff:
        message += f" rff={components} gamma_mult={gamma:g}"
    if candidate is not None:
        message += (
            f" {args.selection_metric}="
            f"{candidate[args.selection_metric]:.4f}"
        )
    print(message)
    return candidate, report_rows


def select_best_parameters(args, datasets, candidates, version):
    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        return pd.DataFrame()
    best_parameters = []
    descending_metrics = set(METRIC_COLUMNS)
    ranking_columns = list(
        dict.fromkeys(
            [
                args.selection_metric,
                "accuracy_with_novel",
                "mcc_with_novel",
                "reference_size",
                "neighbors",
                "novelty_quantile",
                "rff_components" if args.use_rff else None,
            ]
        )
    )
    ranking_columns = [column for column in ranking_columns if column]
    for dataset in datasets:
        choices = candidate_frame[candidate_frame["dataset"] == dataset].copy()
        if choices.empty:
            continue
        choices = choices.sort_values(
            ranking_columns,
            ascending=[
                column not in descending_metrics for column in ranking_columns
            ],
            kind="stable",
        )
        best = choices.iloc[0]
        best_parameters.append(
            {
                "dataset": dataset,
                "novel_class": best["novel_class"],
                "scaler": best["scaler"],
                "reference_size": best["reference_size"],
                "neighbors": int(best["neighbors"]),
                "novelty_quantile": best["novelty_quantile"],
                "epsilon": float(best["epsilon"]),
                "use_rff": args.use_rff,
                "rff_components": (
                    int(best["rff_components"]) if args.use_rff else None
                ),
                "rff_gamma_mode": (
                    "scale_times_multiplier" if args.use_rff else None
                ),
                "rff_gamma_multiplier": (
                    float(best["rff_gamma_multiplier"])
                    if args.use_rff
                    else None
                ),
                "selection_metric": args.selection_metric,
                "selection_test_set": "with_novel_class",
                "validation_score": best[args.selection_metric],
                "lim_code_version": version,
            }
        )
    return pd.DataFrame(best_parameters)


def run_grid_search(args, datasets, seeds, run_all):
    validate_grid_arguments(args)
    version = get_lim_version()
    grid_mode = "rff" if args.use_rff else "no-rff"
    output_dir = args.output_dir or (
        repository_results_root()
        / "lim-models-multiclass-novelty"
        / version
        / grid_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "RFF_LIM_NFST" if args.use_rff else "LIM_NFST"
    configurations = grid_configurations(args)
    manifest = {
        "mode": f"{model_name} MND grid search",
        "model_display_name": (
            RFF_REF_LIM_NAME if args.use_rff else "LIM-NFST"
        ),
        "model_implementation": "LIM_NFST",
        "lim_code_version": version,
        "datasets": datasets,
        "seeds": seeds,
        "novel_class": args.novel_class,
        "novel_class_default": "first sorted class",
        "scalers": args.grid_scalers,
        "reference_sizes": args.grid_reference_sizes,
        "neighbors": args.grid_neighbors,
        "novelty_quantiles": args.grid_novelty_quantiles,
        "epsilon": args.epsilon,
        "use_rff": args.use_rff,
        "rff_components": args.grid_rff_components if args.use_rff else None,
        "rff_gamma_mode": "scale_times_multiplier" if args.use_rff else None,
        "rff_gamma_multipliers": (
            args.grid_rff_gamma_multipliers if args.use_rff else None
        ),
        "selection_metric": args.selection_metric,
        "selection_test_set": "with_novel_class",
        "validation_size": args.validation_size,
        "configurations_per_dataset": len(configurations),
        "outer_test_used_for_selection": False,
        "resume_enabled": not args.force,
        "configuration_directory_schema": "cfg_<sha256-prefix-12>",
        "seed_checkpoint_file": "seed_results.csv",
    }
    save_json(output_dir / "grid_parameters.json", manifest)

    print(f"Mode       : {model_name} MND grid search")
    print(
        "Main model : "
        f"{RFF_REF_LIM_NAME if args.use_rff else 'LIM-NFST ablation'}"
    )
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Novel class: {args.novel_class or 'first sorted class'}")
    print(f"Scalers    : {args.grid_scalers}")
    print(f"Ref sizes  : {args.grid_reference_sizes}")
    print(f"Neighbors  : {args.grid_neighbors}")
    print(f"Quantiles  : {args.grid_novelty_quantiles}")
    print(f"Use RFF    : {args.use_rff}")
    if args.use_rff:
        print(f"RFF dims   : {args.grid_rff_components}")
        print(f"Gamma mult : {args.grid_rff_gamma_multipliers}")
    print(f"Configs/data: {len(configurations)}")
    print(f"Metric     : {args.selection_metric}")
    print(f"Resume     : {not args.force}")
    print(f"Output     : {output_dir.resolve()}")

    if args.output_dir is None and not args.force:
        legacy_dir = (
            WORKSPACE_ROOT
            / "results"
            / "lim-models-multiclass-novelty"
            / version
            / grid_mode
        )
        import_marker = output_dir / "legacy_import.json"
        if import_marker.exists():
            print(f"Legacy import already complete: {import_marker}")
        else:
            migration = migrate_legacy_grid_results(
                legacy_dir,
                output_dir,
                args.validation_size,
            )
            print(
                "Imported legacy results: "
                f"{migration['configurations']} configurations, "
                f"{migration['files_copied']} new files"
            )

    candidates = []
    report_rows = []
    all_errors = []
    resolved_novel_classes = {}
    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)
        for scaler in args.grid_scalers:
            prepared_by_seed = {}
            for seed in seeds:
                try:
                    prepared = prepare_grid_data(
                        dataframe,
                        args.novel_class,
                        scaler,
                        seed,
                        args.validation_size,
                    )
                    prepared_by_seed[seed] = prepared
                    resolved_novel_classes[dataset] = prepared[-1]
                except Exception as error:
                    prepared_by_seed[seed] = None
                    all_errors.append(
                        {
                            "dataset": dataset,
                            "scaler": scaler,
                            "seed": seed,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

            scaler_configs = [
                config for config in configurations if config[0] == scaler
            ]
            for config in scaler_configs:
                candidate, config_rows = save_configuration_result(
                    args,
                    dataset,
                    model_name,
                    version,
                    seeds,
                    config,
                    output_dir,
                    prepared_by_seed,
                    resolved_novel_classes.get(dataset),
                    all_errors,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    report_rows.extend(config_rows)

    grid_results = pd.DataFrame(report_rows, columns=RESULT_COLUMNS)
    grid_results.to_csv(
        output_dir / "grid_search_results.csv",
        index=False,
        float_format="%.8f",
    )
    best_frame = select_best_parameters(args, datasets, candidates, version)
    best_frame.to_csv(
        output_dir / "best_parameters.csv",
        index=False,
        float_format="%.8f",
    )
    save_json(output_dir / "novel_classes.json", resolved_novel_classes)
    if all_errors:
        save_json(output_dir / "errors.json", all_errors)
    print("\nBest parameters:")
    print(best_frame.to_string(index=False))
    return 0 if len(best_frame) == len(datasets) else 2


def main():
    args = parse_args()
    datasets, seeds, run_all = get_datasets_and_seeds(args)
    if args.grid_search:
        return run_grid_search(args, datasets, seeds, run_all)
    return run_fixed_experiment(args, datasets, seeds, run_all)


if __name__ == "__main__":
    raise SystemExit(main())
