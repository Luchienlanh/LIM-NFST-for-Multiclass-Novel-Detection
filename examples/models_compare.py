"""Compare LIM models with kNFST, LDA, and classical classifiers."""

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent
CPAI_ROOT = WORKSPACE_ROOT / "CPAI-main" / "CPAI-main" / "code"

sys.path.insert(0, str(CLASSIFIER_ROOT))
sys.path.insert(0, str(CPAI_ROOT))

from cpai.datasets import DATASETS, load_dataset
from cpai.models import KNFST
from cpai.preprocessing import (
    SCALERS,
    _remove_outliers_lof,
    preprocess as cpai_preprocess,
)
from limnfst.mapping import center_and_normalize as center_normalize
from limnfst.models import LIM_NFST


CPAI_POLY = -1
CPAI_KERNEL = None
CPAI_TEST_SIZE = 0.20
LIM_VERSION_NAME = "lim_nfst_rff_v1"

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

MODEL_NAMES = [
    "LIM_NFST",
    "RFF_LIM_NFST",
    "kNFST",
    "LDA",
    "GaussianNB",
    "KNeighbors",
    "NearestCentroid",
    "RadiusNeighbors",
    "NuSVC",
    "SGD",
]

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
    source_files = [
        CLASSIFIER_ROOT / "limnfst" / "models.py",
        CLASSIFIER_ROOT / "limnfst" / "nfst.py",
        CLASSIFIER_ROOT / "limnfst" / "mapping.py",
    ]

    code_hash = hashlib.sha256()
    for source_file in source_files:
        code_hash.update(source_file.name.encode("utf-8"))
        code_hash.update(source_file.read_bytes())

    return f"{LIM_VERSION_NAME}_{code_hash.hexdigest()[:8]}"


def number_to_name(value):
    return format(float(value), ".6g").replace(".", "p")


def find_default_best_parameters_file(parameter_source, current_version):
    grid_mode = "no-rff" if parameter_source == "lim_ref" else "rff"
    root = WORKSPACE_ROOT / "results" / "lim-models"
    exact_file = root / current_version / grid_mode / "best_parameters.csv"
    if exact_file.exists():
        return exact_file

    available_files = list(root.glob(f"*/{grid_mode}/best_parameters.csv"))
    if parameter_source == "lim_ref":
        available_files.extend(root.glob("*/best_parameters.csv"))
    else:
        legacy_root = WORKSPACE_ROOT / "results" / "rff-ref-lim"
        available_files.extend(legacy_root.glob("*/best_parameters.csv"))
    if available_files:
        return max(available_files, key=lambda path: path.stat().st_mtime)
    return exact_file


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


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LIM with kNFST, LDA, and classical models."
    )
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
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

    parser.add_argument(
        "--normalization-mode",
        choices=["cpai_only", "common_lim"],
        default="cpai_only",
    )
    parser.add_argument("--reference-size", type=float, default=0.20)
    parser.add_argument("--reference-neighbors", type=int, default=5)
    parser.add_argument(
        "--lim-parameter-mode",
        choices=["best", "fixed"],
        default="best",
    )
    parser.add_argument(
        "--best-parameter-source",
        choices=["lim_ref", "rff_ref"],
        default="lim_ref",
        help=(
            "lim_ref: use the no-RFF grid from lim-models.py; "
            "rff_ref: use the RFF grid from lim-models.py."
        ),
    )
    parser.add_argument("--best-parameters-file", type=Path, default=None)
    parser.add_argument("--rff-components", type=int, default=256)
    parser.add_argument("--rff-gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def preprocess_data(dataframe, dataset, scaler, seed):
    """Run the same CPAI preprocessing used by lim-models.py."""
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


def normalize_data(X_train, X_test, normalization_mode):
    if normalization_mode == "cpai_only":
        return X_train, X_test

    with np.errstate(divide="ignore", invalid="ignore"):
        X_train = center_normalize(X_train)
        X_test = center_normalize(X_test)

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(
            "center_normalize produced NaN or Inf because a row has "
            "zero centered norm."
        )

    return X_train, X_test


def load_best_parameters(
    file_path,
    datasets,
    current_version,
    parameter_source,
):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {file_path}. Run lim-models.py --grid-search first."
        )

    parameters = pd.read_csv(file_path)
    required_columns = {
        "dataset",
        "scaler",
        "reference_size",
        "neighbors",
        "lim_code_version",
    }
    if parameter_source == "rff_ref":
        required_columns.update(
            {
                "rff_components",
                "rff_gamma_mode",
                "rff_gamma_multiplier",
            }
        )
    missing_columns = required_columns - set(parameters.columns)
    if missing_columns:
        raise ValueError(
            f"best_parameters.csv is missing: {sorted(missing_columns)}"
        )
    if parameters["dataset"].duplicated().any():
        raise ValueError("best_parameters.csv contains duplicate datasets.")

    result = {}
    for dataset in datasets:
        selected = parameters[parameters["dataset"] == dataset]
        if selected.empty:
            raise ValueError(f"No best parameters found for {dataset}.")

        row = selected.iloc[0]
        stored_version = str(row["lim_code_version"])
        if stored_version != current_version:
            print(
                f"WARNING: {dataset} parameters use {stored_version}, "
                f"current LIM version is {current_version}."
            )

        scaler = str(row["scaler"])
        reference_size = float(row["reference_size"])
        neighbors = int(row["neighbors"])

        if scaler not in [*SCALERS, "None"]:
            raise ValueError(f"Invalid scaler for {dataset}: {scaler}")
        if not 0.0 < reference_size <= 0.30:
            raise ValueError(
                f"Invalid reference size for {dataset}: {reference_size}"
            )
        if neighbors < 1:
            raise ValueError(f"Invalid neighbors for {dataset}: {neighbors}")

        dataset_parameters = {
            "scaler": scaler,
            "reference_size": reference_size,
            "neighbors": neighbors,
            "epsilon": float(row.get("epsilon", 1e-4)),
            "use_rff": parameter_source == "rff_ref",
            "rff_components": None,
            "rff_gamma_multiplier": None,
        }

        if parameter_source == "rff_ref":
            if "use_rff" in parameters.columns:
                stored_use_rff = str(row["use_rff"]).lower()
                if stored_use_rff not in {"true", "1"}:
                    raise ValueError(
                        f"RFF parameters for {dataset} have use_rff="
                        f"{row['use_rff']}"
                    )
            rff_gamma_mode = str(row["rff_gamma_mode"])
            if rff_gamma_mode != "scale_times_multiplier":
                raise ValueError(
                    f"Unsupported RFF gamma mode for {dataset}: "
                    f"{rff_gamma_mode}"
                )

            rff_components = int(row["rff_components"])
            gamma_multiplier = float(row["rff_gamma_multiplier"])
            if rff_components < 1:
                raise ValueError(
                    f"Invalid RFF components for {dataset}: "
                    f"{rff_components}"
                )
            if gamma_multiplier <= 0.0:
                raise ValueError(
                    f"Invalid RFF gamma multiplier for {dataset}: "
                    f"{gamma_multiplier}"
                )

            dataset_parameters["rff_components"] = rff_components
            dataset_parameters["rff_gamma_multiplier"] = gamma_multiplier

        result[dataset] = dataset_parameters

    return result


def create_model(
    model_name,
    seed,
    reference_size,
    neighbors,
    epsilon,
    rff_components,
    rff_gamma_multiplier,
):
    """Create one model. Kept explicit so each configuration is visible."""
    if model_name == "LIM_NFST":
        return LIM_NFST(
            epsilon=epsilon,
            reference_size=reference_size,
            number_of_neighbors=neighbors,
            random_state=seed,
        )
    if model_name == "RFF_LIM_NFST":
        return LIM_NFST(
            epsilon=epsilon,
            reference_size=reference_size,
            number_of_neighbors=neighbors,
            random_state=seed,
            use_rff=True,
            rff_components=rff_components,
            rff_gamma_multiplier=rff_gamma_multiplier,
        )
    if model_name == "kNFST":
        return KNFST(kernel="rbf")
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver="svd")
    if model_name == "GaussianNB":
        return GaussianNB()
    if model_name == "KNeighbors":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "NearestCentroid":
        return NearestCentroid(metric="euclidean")
    if model_name == "RadiusNeighbors":
        return RadiusNeighborsClassifier(
            radius=1.0,
            outlier_label="most_frequent",
        )
    if model_name == "NuSVC":
        return NuSVC(nu=0.2, kernel="rbf", random_state=seed)
    if model_name == "SGD":
        return SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def summarize_results(run_results, datasets, seeds):
    """Average complete seed runs and return only names plus metrics."""
    if not run_results:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    runs = pd.DataFrame(run_results)
    run_counts = (
        runs.groupby(["dataset", "model"], sort=False)
        .size()
        .rename("completed_seeds")
        .reset_index()
    )
    means = (
        runs.groupby(["dataset", "model"], sort=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    means = means.merge(run_counts, on=["dataset", "model"])
    means = means[means["completed_seeds"] == len(seeds)]
    results = means[RESULT_COLUMNS].copy()

    overall_rows = []
    for model_name in results["model"].unique():
        model_results = results[results["model"] == model_name]
        if set(model_results["dataset"]) != set(datasets):
            continue

        overall = {"dataset": "OVERALL", "model": model_name}
        for metric in METRIC_COLUMNS:
            overall[metric] = model_results[metric].mean()
        overall_rows.append(overall)

    if overall_rows:
        results = pd.concat(
            [results, pd.DataFrame(overall_rows)],
            ignore_index=True,
        )

    return results[RESULT_COLUMNS]


def main():
    args = parse_args()
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

    version = get_lim_version()
    if args.best_parameters_file is None:
        args.best_parameters_file = find_default_best_parameters_file(
            args.best_parameter_source,
            version,
        )

    if args.lim_parameter_mode == "best":
        parameters_by_dataset = load_best_parameters(
            args.best_parameters_file,
            datasets,
            version,
            args.best_parameter_source,
        )
        parameter_name = (
            f"best-grid-parameters__source={args.best_parameter_source}"
        )
    else:
        if not 0.0 < args.reference_size <= 0.30:
            raise ValueError("--reference-size must be in (0, 0.30].")
        if args.reference_neighbors < 1:
            raise ValueError("--reference-neighbors must be at least one.")
        if args.best_parameter_source == "rff_ref":
            if args.rff_components < 1:
                raise ValueError("--rff-components must be at least one.")
            if args.rff_gamma_multiplier <= 0.0:
                raise ValueError(
                    "--rff-gamma-multiplier must be greater than zero."
                )

        parameters_by_dataset = {
            dataset: {
                "scaler": args.scaler,
                "reference_size": args.reference_size,
                "neighbors": args.reference_neighbors,
                "epsilon": 1e-4,
                "use_rff": args.best_parameter_source == "rff_ref",
                "rff_components": (
                    args.rff_components
                    if args.best_parameter_source == "rff_ref"
                    else None
                ),
                "rff_gamma_multiplier": (
                    args.rff_gamma_multiplier
                    if args.best_parameter_source == "rff_ref"
                    else None
                ),
            }
            for dataset in datasets
        }
        parameter_name = (
            f"fixed__source={args.best_parameter_source}"
            f"__scaler={args.scaler}"
            f"__reference_size={number_to_name(args.reference_size)}"
            f"__neighbors={args.reference_neighbors}"
        )
        if args.best_parameter_source == "rff_ref":
            parameter_name += (
                f"__rff_components={args.rff_components}"
                "__rff_gamma_multiplier="
                f"{number_to_name(args.rff_gamma_multiplier)}"
            )

    if args.models is None:
        model_names = [
            "LIM_NFST",
        ]
        if args.best_parameter_source == "rff_ref":
            model_names.append("RFF_LIM_NFST")
        model_names.extend(
            [
                "kNFST",
                "LDA",
                "GaussianNB",
                "KNeighbors",
                "NearestCentroid",
                "RadiusNeighbors",
                "NuSVC",
                "SGD",
            ]
        )
    else:
        model_names = args.models

    if "RFF_LIM_NFST" in model_names:
        missing_rff_parameters = [
            dataset
            for dataset, parameters in parameters_by_dataset.items()
            if parameters["rff_components"] is None
            or parameters["rff_gamma_multiplier"] is None
        ]
        if missing_rff_parameters:
            raise ValueError(
                "RFF_LIM_NFST needs RFF parameters. Select "
                "--best-parameter-source rff_ref. Missing datasets: "
                f"{missing_rff_parameters}"
            )

    if args.output_dir is None:
        output_dir = (
            WORKSPACE_ROOT
            / "results"
            / "models-compare"
            / version
            / f"{parameter_name}__normalization={args.normalization_mode}"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        output_dir / "parameters.json",
        {
            "lim_code_version": version,
            "datasets": datasets,
            "models": model_names,
            "seeds": seeds,
            "normalization_mode": args.normalization_mode,
            "lim_parameter_mode": args.lim_parameter_mode,
            "best_parameter_source": args.best_parameter_source,
            "best_parameters_file": (
                str(args.best_parameters_file.resolve())
                if args.lim_parameter_mode == "best"
                else None
            ),
            "parameters_by_dataset": parameters_by_dataset,
        },
    )

    print("Task       : closed-set multiclass classification")
    print(f"Datasets   : {datasets}")
    print(f"Models     : {model_names}")
    print(f"Seeds      : {seeds}")
    print(f"Best source: {args.best_parameter_source}")
    print(f"Output     : {output_dir.resolve()}")

    run_results = []
    errors = []
    gamma_by_run = {}

    for dataset in datasets:
        parameters = parameters_by_dataset[dataset]
        limit = DEFAULT_LIMITS[dataset] if run_all else (
            args.limit or DEFAULT_LIMITS[dataset]
        )

        try:
            dataframe, _ = load_dataset(dataset, limit)
        except Exception as error:
            errors.append(
                {
                    "dataset": dataset,
                    "seed": None,
                    "model": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"FAILED loading {dataset}: {error}")
            continue

        for seed in seeds:
            try:
                X_train, y_train, X_test, y_test, _ = preprocess_data(
                    dataframe,
                    dataset,
                    parameters["scaler"],
                    seed,
                )
                X_train, X_test = normalize_data(
                    X_train,
                    X_test,
                    args.normalization_mode,
                )
            except Exception as error:
                errors.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"FAILED preprocessing {dataset}, seed={seed}: {error}")
                continue

            for model_name in model_names:
                try:
                    model = create_model(
                        model_name,
                        seed,
                        parameters["reference_size"],
                        parameters["neighbors"],
                        parameters["epsilon"],
                        parameters["rff_components"],
                        parameters["rff_gamma_multiplier"],
                    )

                    model.fit(X_train, y_train)

                    if model_name == "RFF_LIM_NFST":
                        gamma_by_run[f"{dataset}__seed={seed}"] = (
                            model.rff_gamma_
                        )

                    if model_name in [
                        "LIM_NFST",
                        "RFF_LIM_NFST",
                    ]:
                        y_pred = model.predict_closed(X_test)
                    else:
                        y_pred = model.predict(X_test)

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
                            "model": model_name,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    print(
                        f"FAILED dataset={dataset} seed={seed} "
                        f"model={model_name}: {error}"
                    )

    results = summarize_results(run_results, datasets, seeds)
    results.to_csv(
        output_dir / "results.csv",
        index=False,
        float_format="%.8f",
    )

    if errors:
        save_json(output_dir / "errors.json", errors)
    if gamma_by_run:
        save_json(output_dir / "rff_gamma_by_run.json", gamma_by_run)

    print("\nFinal results:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
