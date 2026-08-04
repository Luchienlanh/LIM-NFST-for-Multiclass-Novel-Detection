"""Run LIM experiments and tune ReferenceLIM with a simple grid search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent
CPAI_ROOT = WORKSPACE_ROOT / "CPAI-main" / "CPAI-main" / "code"
REFERENCE_ROOT = WORKSPACE_ROOT / "limnfst-reference-classifier"

sys.path.insert(0, str(CLASSIFIER_ROOT))
sys.path.insert(0, str(CPAI_ROOT))
sys.path.insert(0, str(REFERENCE_ROOT))

from cpai.datasets import DATASETS, load_dataset
from cpai.preprocessing import (
    SCALERS,
    _remove_outliers_lof,
    _scaler as make_scaler,
    preprocess as cpai_preprocess,
)
from limnfst.models import LIM_NFST
from reference_lim import ReferenceLIM, ReferenceResidualLIM, ResidualLIM


CPAI_POLY = -1
CPAI_KERNEL = None
CPAI_TEST_SIZE = 0.20
LIM_VERSION_NAME = "lim_reference_v1"

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

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
]
RESULT_COLUMNS = ["dataset", "model", *METRIC_COLUMNS]


def get_lim_version():
    """Create a short version from the current ReferenceLIM source files."""
    source_files = [
        REFERENCE_ROOT / "reference_lim" / "reference_model.py",
        REFERENCE_ROOT / "reference_lim" / "nfst.py",
        REFERENCE_ROOT / "reference_lim" / "mapping.py",
    ]

    code_hash = hashlib.sha256()
    for source_file in source_files:
        code_hash.update(source_file.name.encode("utf-8"))
        code_hash.update(source_file.read_bytes())

    return f"{LIM_VERSION_NAME}_{code_hash.hexdigest()[:8]}"


def number_to_name(value):
    return format(float(value), ".6g").replace(".", "p")


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def average_results(run_results, include_overall=True):
    """Average seed results and keep only model metrics in the CSV."""
    if not run_results:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    runs = pd.DataFrame(run_results)
    results = (
        runs.groupby(["dataset", "model"], sort=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )

    if include_overall:
        overall = (
            results.groupby("model", sort=False)[METRIC_COLUMNS]
            .mean()
            .reset_index()
        )
        overall.insert(0, "dataset", "OVERALL")
        results = pd.concat([results, overall], ignore_index=True)

    return results[RESULT_COLUMNS]


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LIM models or tune ReferenceLIM."
    )
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    scaler_group = parser.add_mutually_exclusive_group()
    scaler_group.add_argument("--scaler", choices=SCALERS)
    scaler_group.add_argument(
        "--no-scaler",
        action="store_const",
        const="None",
        dest="scaler",
    )
    parser.set_defaults(scaler="StandardScaler")

    parser.add_argument("--reference-size", type=float, default=0.20)
    parser.add_argument("--neighbors", type=int, default=5)

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
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "balanced_accuracy", "accuracy", "mcc"],
        default="macro_f1",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
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


def preprocess_fixed_experiment(dataframe, dataset, scaler, seed):
    """Use the normal CPAI train/test preprocessing pipeline."""
    if scaler != "None":
        return cpai_preprocess(
            dataframe,
            dataset_name=dataset,
            poly=CPAI_POLY,
            kernel=CPAI_KERNEL,
            scaler=scaler,
            seed=seed,
        )

    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]

    if dataset == "IoTID20":
        X[np.isinf(X)] = np.nan
        X = SimpleImputer(strategy="mean").fit_transform(X)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        test_size=CPAI_TEST_SIZE,
        stratify=y_raw,
        random_state=seed,
    )

    order = y_train_raw.argsort()
    X_train = X_train[order]
    y_train_raw = y_train_raw[order]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_train, y_train = _remove_outliers_lof(X_train, y_train)

    return X_train, y_train, X_test, y_test, encoder


def prepare_grid_data(dataframe, dataset, scaler, seed, validation_size):
    """Create an inner validation split without touching the outer test set."""
    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]

    X_outer_train, _, y_outer_train, _ = train_test_split(
        X,
        y_raw,
        test_size=CPAI_TEST_SIZE,
        stratify=y_raw,
        random_state=seed,
    )

    X_train, X_validation, y_train_raw, y_validation_raw = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=validation_size,
        stratify=y_outer_train,
        random_state=seed,
    )

    order = y_train_raw.argsort()
    X_train = X_train[order]
    y_train_raw = y_train_raw[order]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_validation = encoder.transform(y_validation_raw)

    if dataset == "IoTID20":
        X_train[np.isinf(X_train)] = np.nan
        X_validation[np.isinf(X_validation)] = np.nan
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_validation = imputer.transform(X_validation)

    if scaler == "None":
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_validation = np.nan_to_num(X_validation, nan=0.0)
    else:
        fitted_scaler = make_scaler(scaler, random_state=seed)
        X_train = fitted_scaler.fit_transform(X_train)
        X_validation = fitted_scaler.transform(X_validation)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_validation = np.nan_to_num(X_validation, nan=0.0)

    X_train, y_train = _remove_outliers_lof(X_train, y_train)
    return X_train, y_train, X_validation, y_validation


def run_fixed_experiment(args, datasets, seeds, run_all):
    if not 0.0 < args.reference_size <= 0.30:
        raise ValueError("--reference-size must be in (0, 0.30].")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least one.")

    version = get_lim_version()
    dataset_name = "all" if run_all else datasets[0]

    if args.output_dir is None:
        folder_name = (
            f"evaluation__dataset={dataset_name}__scaler={args.scaler}"
            f"__reference_size={number_to_name(args.reference_size)}"
            f"__neighbors={args.neighbors}"
        )
        output_dir = (
            WORKSPACE_ROOT
            / "results"
            / "lim-models"
            / version
            / folder_name
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "fixed LIM comparison",
        "lim_code_version": version,
        "datasets": datasets,
        "seeds": seeds,
        "scaler": args.scaler,
        "reference_size": args.reference_size,
        "neighbors": args.neighbors,
    }
    save_json(output_dir / "parameters.json", manifest)

    print("Mode       : fixed LIM comparison")
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Scaler     : {args.scaler}")
    print(f"Reference  : {args.reference_size}")
    print(f"Neighbors  : {args.neighbors}")
    print(f"Output     : {output_dir.resolve()}")

    run_results = []
    errors = []

    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)

        for seed in seeds:
            try:
                X_train, y_train, X_test, y_test, _ = (
                    preprocess_fixed_experiment(
                        dataframe,
                        dataset,
                        args.scaler,
                        seed,
                    )
                )

                original_model = LIM_NFST()
                reference_model = ReferenceLIM(
                    reference_size=args.reference_size,
                    number_of_neighbors=args.neighbors,
                    random_state=seed,
                )
                residual_model = ResidualLIM()
                combined_model = ReferenceResidualLIM(
                    reference_size=args.reference_size,
                    number_of_neighbors=args.neighbors,
                    random_state=seed,
                )

                models = [
                    ("LIM_NFST", original_model),
                    ("LIM_Reference", reference_model),
                    ("LIM_Residual", residual_model),
                    ("LIM_Reference_Residual", combined_model),
                ]

                for model_name, model in models:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_closed(X_test)

                    row = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                    }
                    row.update(calculate_metrics(y_test, y_pred))
                    run_results.append(row)

                    print(
                        f"dataset={dataset} seed={seed} model={model_name} "
                        f"accuracy={row['accuracy']:.4f} "
                        f"macro_f1={row['macro_f1']:.4f}"
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
        output_dir / "results.csv",
        index=False,
        float_format="%.8f",
    )

    if errors:
        save_json(output_dir / "errors.json", errors)

    print("\nFinal results:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


def run_grid_search(args, datasets, seeds, run_all):
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between zero and one.")
    if any(
        value <= 0.0 or value > 0.30
        for value in args.grid_reference_sizes
    ):
        raise ValueError("Grid reference sizes must be in (0, 0.30].")
    if any(value < 1 for value in args.grid_neighbors):
        raise ValueError("Grid neighbors must be at least one.")

    version = get_lim_version()
    output_dir = args.output_dir or (
        WORKSPACE_ROOT / "results" / "lim-models" / version
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "ReferenceLIM grid search",
        "lim_code_version": version,
        "datasets": datasets,
        "seeds": seeds,
        "scalers": args.grid_scalers,
        "reference_sizes": args.grid_reference_sizes,
        "neighbors": args.grid_neighbors,
        "selection_metric": args.selection_metric,
        "validation_size": args.validation_size,
        "outer_test_used_for_selection": False,
    }
    save_json(output_dir / "grid_parameters.json", manifest)

    print("Mode       : ReferenceLIM grid search")
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Scalers    : {args.grid_scalers}")
    print(f"Ref sizes  : {args.grid_reference_sizes}")
    print(f"Neighbors  : {args.grid_neighbors}")
    print(f"Metric     : {args.selection_metric}")
    print(f"Output     : {output_dir.resolve()}")

    all_grid_results = []
    candidates = []
    all_errors = []

    for dataset in datasets:
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )
        dataframe, _ = load_dataset(dataset, limit)

        # Cache preprocessing once for each scaler and seed.
        validation_data = {}
        for scaler in args.grid_scalers:
            for seed in seeds:
                try:
                    validation_data[(scaler, seed)] = prepare_grid_data(
                        dataframe,
                        dataset,
                        scaler,
                        seed,
                        args.validation_size,
                    )
                except Exception as error:
                    validation_data[(scaler, seed)] = None
                    all_errors.append(
                        {
                            "dataset": dataset,
                            "scaler": scaler,
                            "seed": seed,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

        for scaler in args.grid_scalers:
            for reference_size in args.grid_reference_sizes:
                for neighbors in args.grid_neighbors:
                    configuration_name = (
                        f"scaler={scaler}"
                        f"__reference_size={number_to_name(reference_size)}"
                        f"__neighbors={neighbors}"
                    )
                    configuration_dir = (
                        output_dir / dataset / configuration_name
                    )
                    configuration_dir.mkdir(parents=True, exist_ok=True)

                    parameters = {
                        "dataset": dataset,
                        "model": "LIM_Reference",
                        "scaler": scaler,
                        "reference_size": reference_size,
                        "neighbors": neighbors,
                        "seeds": seeds,
                        "selection_metric": args.selection_metric,
                        "lim_code_version": version,
                        "outer_test_used_for_selection": False,
                    }
                    save_json(configuration_dir / "parameters.json", parameters)

                    configuration_runs = []
                    configuration_errors = []

                    for seed in seeds:
                        prepared_data = validation_data[(scaler, seed)]
                        if prepared_data is None:
                            configuration_errors.append(
                                {
                                    "seed": seed,
                                    "error": "Preprocessing failed.",
                                }
                            )
                            continue

                        X_train, y_train, X_validation, y_validation = (
                            prepared_data
                        )

                        try:
                            model = ReferenceLIM(
                                reference_size=reference_size,
                                number_of_neighbors=neighbors,
                                random_state=seed,
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict_closed(X_validation)

                            row = {
                                "dataset": dataset,
                                "model": "LIM_Reference",
                                "seed": seed,
                            }
                            row.update(
                                calculate_metrics(y_validation, y_pred)
                            )
                            configuration_runs.append(row)
                        except Exception as error:
                            configuration_errors.append(
                                {
                                    "seed": seed,
                                    "error": (
                                        f"{type(error).__name__}: {error}"
                                    ),
                                }
                            )

                    configuration_result = average_results(
                        configuration_runs,
                        include_overall=False,
                    )
                    configuration_result.to_csv(
                        configuration_dir / "results.csv",
                        index=False,
                        float_format="%.8f",
                    )

                    if configuration_errors:
                        save_json(
                            configuration_dir / "errors.json",
                            configuration_errors,
                        )

                    complete = len(configuration_runs) == len(seeds)
                    if complete and not configuration_result.empty:
                        metric_values = configuration_result.iloc[0].to_dict()

                        report_row = {
                            "dataset": dataset,
                            "model": (
                                "LIM_Reference"
                                f"[scaler={scaler},"
                                f"reference_size={reference_size:g},"
                                f"neighbors={neighbors}]"
                            ),
                        }
                        for metric in METRIC_COLUMNS:
                            report_row[metric] = metric_values[metric]
                        all_grid_results.append(report_row)

                        candidate = {
                            "dataset": dataset,
                            "scaler": scaler,
                            "reference_size": reference_size,
                            "neighbors": neighbors,
                        }
                        for metric in METRIC_COLUMNS:
                            candidate[metric] = metric_values[metric]
                        candidates.append(candidate)

                        print(
                            f"dataset={dataset} scaler={scaler} "
                            f"ref={reference_size:g} k={neighbors} "
                            f"{args.selection_metric}="
                            f"{candidate[args.selection_metric]:.4f}"
                        )
                    else:
                        print(
                            f"FAILED dataset={dataset} scaler={scaler} "
                            f"ref={reference_size:g} k={neighbors}"
                        )

    grid_results = pd.DataFrame(all_grid_results, columns=RESULT_COLUMNS)
    grid_results.to_csv(
        output_dir / "grid_search_results.csv",
        index=False,
        float_format="%.8f",
    )

    best_parameters = []
    candidate_frame = pd.DataFrame(candidates)

    for dataset in datasets:
        dataset_candidates = candidate_frame[
            candidate_frame["dataset"] == dataset
        ].copy()

        if dataset_candidates.empty:
            continue

        ranking_columns = list(
            dict.fromkeys(
                [
                    args.selection_metric,
                    "accuracy",
                    "mcc",
                    "reference_size",
                    "neighbors",
                ]
            )
        )
        descending_metrics = {
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
        }
        dataset_candidates = dataset_candidates.sort_values(
            ranking_columns,
            ascending=[
                column not in descending_metrics
                for column in ranking_columns
            ],
            kind="stable",
        )
        best = dataset_candidates.iloc[0]

        best_parameters.append(
            {
                "dataset": dataset,
                "scaler": best["scaler"],
                "reference_size": best["reference_size"],
                "neighbors": int(best["neighbors"]),
                "selection_metric": args.selection_metric,
                "validation_score": best[args.selection_metric],
                "lim_code_version": version,
            }
        )

    best_frame = pd.DataFrame(best_parameters)
    best_frame.to_csv(
        output_dir / "best_parameters.csv",
        index=False,
        float_format="%.8f",
    )

    if all_errors:
        save_json(output_dir / "errors.json", all_errors)

    print("\nBest parameters:")
    print(best_frame.to_string(index=False))
    return 0 if len(best_parameters) == len(datasets) else 2


def main():
    args = parse_args()
    datasets, seeds, run_all = get_datasets_and_seeds(args)

    if args.grid_search:
        return run_grid_search(args, datasets, seeds, run_all)

    return run_fixed_experiment(args, datasets, seeds, run_all)


if __name__ == "__main__":
    raise SystemExit(main())
