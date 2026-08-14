"""Compare LIM models with kNFST, LDA, and classical classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import pairwise_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.svm import NuSVC


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent

sys.path.insert(0, str(CLASSIFIER_ROOT))

from limnfst.competitors import KNFST
from limnfst.datasets import DATASETS, load_dataset
from limnfst.preprocessing import (
    SCALERS,
    preprocess_data as shared_preprocess_data,
)
from limnfst.mapping import center_and_normalize as center_normalize
from limnfst.metrics import (
    SUMMARY_METRIC_COLUMNS,
    calculate_summary_metrics,
    distances_to_scores,
    measure_single_sample_latency,
    save_confusion_matrix_plot,
    save_curve_plot,
    save_evaluation_artifacts,
    save_training_history_plot,
    scores_to_probabilities,
    summarize_error_pairs,
)
from limnfst.models import LIM_NFST


LIM_VERSION_NAME = "lim_nfst_rff_cv_v2"

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

METRIC_COLUMNS = list(SUMMARY_METRIC_COLUMNS)
RESULT_COLUMNS = ["dataset", "model", *METRIC_COLUMNS]


def get_lim_version():
    source_files = [
        CLASSIFIER_ROOT / "limnfst" / "models.py",
        CLASSIFIER_ROOT / "limnfst" / "nfst.py",
        CLASSIFIER_ROOT / "limnfst" / "mapping.py",
        CLASSIFIER_ROOT / "limnfst" / "metrics.py",
        CLASSIFIER_ROOT / "limnfst" / "datasets.py",
        CLASSIFIER_ROOT / "limnfst" / "preprocessing.py",
        CLASSIFIER_ROOT / "examples" / "lim-models.py",
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
        choices=["preprocess_only", "common_lim"],
        default="preprocess_only",
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
    parser.add_argument(
        "--single-sample-repeats",
        type=int,
        default=30,
        help="Number of timed predictions used for one-sample latency.",
    )
    parser.add_argument(
        "--single-sample-warmup",
        type=int,
        default=3,
        help="Warm-up predictions before measuring one-sample latency.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="create_plots",
        help="Save CSV artifacts but skip PNG confusion/ROC/PR plots.",
    )
    parser.set_defaults(create_plots=True)
    parser.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="Do not save the fitted primary LIM model bundle.",
    )
    parser.set_defaults(save_model=True)
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help=(
            "Rebuild aggregate CSV/PNG artifacts from completed per-seed "
            "artifacts without fitting models again."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def preprocess_data(dataframe, dataset, scaler, seed):
    """Run the shared preprocessing owned by limnfst-classifier."""
    return shared_preprocess_data(
        dataframe,
        dataset_name=dataset,
        scaler_name=scaler,
        random_state=seed,
        test_size=0.20,
        return_preprocessing=True,
    )


def normalize_data(X_train, X_test, normalization_mode):
    if normalization_mode == "preprocess_only":
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


def align_class_columns(values, source_labels, labels, fill_value):
    """Place model score/probability columns in the common label order."""
    values = np.asarray(values, dtype=np.float64)
    source_labels = np.asarray(source_labels)
    labels = np.asarray(labels)

    if values.ndim != 2:
        raise ValueError("Class score values must be a two-dimensional array.")
    if values.shape[1] != len(source_labels):
        raise ValueError(
            "The number of score columns does not match source labels."
        )

    aligned = np.full(
        (values.shape[0], len(labels)),
        fill_value,
        dtype=np.float64,
    )
    label_positions = {
        label: position for position, label in enumerate(labels)
    }
    for source_position, label in enumerate(source_labels):
        if label in label_positions:
            aligned[:, label_positions[label]] = values[:, source_position]
    return aligned


def make_predict_function(model, model_name, training_labels):
    """Return one prediction function with encoded labels for every model."""
    if model_name in {"LIM_NFST", "RFF_LIM_NFST"}:
        return model.predict_closed

    if model_name == "kNFST":
        training_labels = np.asarray(training_labels)

        def predict_knfst(X):
            class_indices = np.asarray(model.predict(X), dtype=np.int64)
            return training_labels[class_indices]

        return predict_knfst

    return model.predict


def get_scores_and_probabilities(
    model,
    model_name,
    X_test,
    labels,
    training_labels,
):
    """Return comparable class scores, probabilities and their provenance.

    ROC and PR only require ranking scores. Models with real ``predict_proba``
    keep those probabilities. Distance and decision-function models use a
    softmax conversion, which is explicitly marked as uncalibrated.
    """
    labels = np.asarray(labels)
    training_labels = np.asarray(training_labels)

    if model_name in {"LIM_NFST", "RFF_LIM_NFST"}:
        distance_matrix = model.reference_scores(X_test)
        score_matrix = distances_to_scores(distance_matrix)
        probability_matrix = scores_to_probabilities(score_matrix)
        return (
            score_matrix,
            probability_matrix,
            "negative_reference_distance",
            "softmax_uncalibrated",
        )

    if model_name == "NearestCentroid":
        distance_matrix = pairwise_distances(
            X_test,
            model.centroids_,
            metric="euclidean",
        )
        source_labels = model.classes_
        minimum_score = -float(np.max(distance_matrix)) - 1.0
        score_matrix = align_class_columns(
            distances_to_scores(distance_matrix),
            source_labels,
            labels,
            minimum_score,
        )
        probability_matrix = scores_to_probabilities(score_matrix)
        return (
            score_matrix,
            probability_matrix,
            "negative_centroid_distance",
            "softmax_uncalibrated",
        )

    if hasattr(model, "predict_proba"):
        probability_values = np.asarray(
            model.predict_proba(X_test),
            dtype=np.float64,
        )
        source_labels = getattr(model, "classes_", training_labels)
        probability_matrix = align_class_columns(
            probability_values,
            source_labels,
            labels,
            0.0,
        )
        row_sums = probability_matrix.sum(axis=1, keepdims=True)
        empty_rows = row_sums[:, 0] <= 0.0
        if empty_rows.any():
            probability_matrix[empty_rows] = 1.0 / len(labels)
            row_sums = probability_matrix.sum(axis=1, keepdims=True)
        probability_matrix = probability_matrix / row_sums
        probability_kind = (
            "pseudo_probability"
            if model_name == "kNFST"
            else "model_probability"
        )
        return (
            probability_matrix.copy(),
            probability_matrix,
            "predict_proba",
            probability_kind,
        )

    if hasattr(model, "decision_function"):
        score_values = np.asarray(
            model.decision_function(X_test),
            dtype=np.float64,
        )
        source_labels = np.asarray(model.classes_)
        if score_values.ndim == 1:
            if len(source_labels) != 2:
                raise ValueError(
                    "A one-dimensional decision function requires two classes."
                )
            score_values = np.column_stack([-score_values, score_values])
        minimum_score = float(np.min(score_values)) - 1.0
        score_matrix = align_class_columns(
            score_values,
            source_labels,
            labels,
            minimum_score,
        )
        probability_matrix = scores_to_probabilities(score_matrix)
        return (
            score_matrix,
            probability_matrix,
            "decision_function",
            "softmax_uncalibrated",
        )

    raise ValueError(
        f"Cannot obtain class scores for model {model_name}."
    )


def summarize_per_class_results(per_class_runs):
    """Average per-class metrics over seeds for each dataset and model."""
    if not per_class_runs:
        return pd.DataFrame()

    runs = pd.concat(per_class_runs, ignore_index=True)
    identity_columns = [
        "dataset",
        "model",
        "class_index",
        "label",
        "class_name",
    ]
    numeric_columns = [
        column
        for column in runs.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "class_index", "label"}
    ]
    averaged = (
        runs.groupby(identity_columns, sort=False, dropna=False)[
            numeric_columns
        ]
        .mean()
        .reset_index()
    )
    return averaged


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


def normalize_confusion_matrix(matrix, mode):
    """Normalize a confusion matrix without warnings for empty rows/columns."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if mode == "true":
        denominator = matrix.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denominator = matrix.sum(axis=0, keepdims=True)
    elif mode == "all":
        denominator = np.asarray([[matrix.sum()]])
    else:
        raise ValueError("mode must be true, pred, or all.")
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator != 0,
    )


def average_curve_files(files, curve_type):
    """Average saved per-seed curves on a shared 501-point x-axis."""
    if curve_type == "roc":
        x_column = "false_positive_rate"
        y_column = "true_positive_rate"
    elif curve_type == "pr":
        x_column = "recall"
        y_column = "precision"
    else:
        raise ValueError("curve_type must be roc or pr.")

    tables = [pd.read_csv(file_path) for file_path in files]
    labels = []
    for table in tables:
        for label in table["label"].astype(str):
            if label not in labels:
                labels.append(label)

    common_x = np.linspace(0.0, 1.0, 501)
    rows = []
    for label in labels:
        interpolated = []
        for table in tables:
            selected = table[table["label"].astype(str) == label]
            selected = selected[[x_column, y_column]].dropna()
            if selected.empty:
                continue
            selected = (
                selected.groupby(x_column, as_index=False)[y_column]
                .mean()
                .sort_values(x_column)
            )
            interpolated.append(
                np.interp(
                    common_x,
                    selected[x_column].to_numpy(),
                    selected[y_column].to_numpy(),
                )
            )
        if not interpolated:
            continue
        mean_y = np.mean(np.vstack(interpolated), axis=0)
        for x_value, y_value in zip(common_x, mean_y):
            row = {
                "curve": "mean_across_seeds",
                "class_index": -1,
                "label": label,
                "threshold": np.nan,
                x_column: x_value,
                y_column: y_value,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def rebuild_aggregate_artifacts(output_dir, create_plots=True):
    """Rebuild aggregate reports from existing per-seed CSV artifacts."""
    output_dir = Path(output_dir)
    artifact_root = output_dir / "artifacts"
    if not artifact_root.exists():
        raise FileNotFoundError(
            f"Cannot find completed per-seed artifacts: {artifact_root}"
        )

    results_file = output_dir / "results.csv"
    per_class_file = output_dir / "per_class_results.csv"
    results = pd.read_csv(results_file) if results_file.exists() else None
    per_class = (
        pd.read_csv(per_class_file) if per_class_file.exists() else None
    )

    rebuilt = []
    for dataset_dir in sorted(
        [path for path in artifact_root.iterdir() if path.is_dir()]
    ):
        seed_dirs = sorted(
            [path for path in dataset_dir.iterdir() if path.is_dir()]
        )
        model_names = sorted(
            {
                model_dir.name
                for seed_dir in seed_dirs
                for model_dir in seed_dir.iterdir()
                if model_dir.is_dir()
            }
        )

        for model_name in model_names:
            run_dirs = [
                seed_dir / model_name
                for seed_dir in seed_dirs
                if (seed_dir / model_name).is_dir()
            ]
            raw_files = [
                run_dir / "confusion_matrix_raw.csv"
                for run_dir in run_dirs
                if (run_dir / "confusion_matrix_raw.csv").exists()
            ]
            if not raw_files:
                continue

            raw_tables = [
                pd.read_csv(file_path, index_col=0)
                for file_path in raw_files
            ]
            class_names = list(raw_tables[0].columns)
            raw = np.sum(
                [table.to_numpy(dtype=np.int64) for table in raw_tables],
                axis=0,
            )
            normalized_true = normalize_confusion_matrix(raw, "true")
            normalized_pred = normalize_confusion_matrix(raw, "pred")
            normalized_all = normalize_confusion_matrix(raw, "all")

            aggregate_dir = (
                output_dir / "aggregate" / dataset_dir.name / model_name
            )
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            matrices = {
                "raw": raw,
                "normalized_true": normalized_true,
                "normalized_pred": normalized_pred,
                "normalized_all": normalized_all,
            }
            for matrix_name, matrix in matrices.items():
                pd.DataFrame(
                    matrix,
                    index=class_names,
                    columns=class_names,
                ).to_csv(
                    aggregate_dir / f"confusion_matrix_{matrix_name}.csv",
                    float_format="%.10f",
                )

            one_vs_rest_rows = []
            total = raw.sum()
            for class_index, class_name in enumerate(class_names):
                true_positive = int(raw[class_index, class_index])
                false_negative = int(
                    raw[class_index, :].sum() - true_positive
                )
                false_positive = int(
                    raw[:, class_index].sum() - true_positive
                )
                true_negative = int(
                    total
                    - true_positive
                    - false_negative
                    - false_positive
                )
                one_vs_rest_rows.append(
                    {
                        "class_index": class_index,
                        "class_name": class_name,
                        "true_negative": true_negative,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                        "true_positive": true_positive,
                    }
                )
                if create_plots:
                    save_confusion_matrix_plot(
                        np.asarray(
                            [
                                [true_negative, false_positive],
                                [false_negative, true_positive],
                            ]
                        ),
                        [f"not_{class_name}", class_name],
                        aggregate_dir / "cm_class" / f"c{class_index:02d}.png",
                        title=f"One-vs-rest confusion matrix: {class_name}",
                    )
            pd.DataFrame(one_vs_rest_rows).to_csv(
                aggregate_dir / "confusion_matrix_one_vs_rest.csv",
                index=False,
            )

            if results is not None:
                summary = results[
                    (results["dataset"] == dataset_dir.name)
                    & (results["model"] == model_name)
                ]
                summary.to_csv(
                    aggregate_dir / "summary_metrics.csv",
                    index=False,
                )
            if per_class is not None:
                selected_classes = per_class[
                    (per_class["dataset"] == dataset_dir.name)
                    & (per_class["model"] == model_name)
                ]
                selected_classes.to_csv(
                    aggregate_dir / "per_class_metrics.csv",
                    index=False,
                )

            error_tables = []
            for run_dir in run_dirs:
                error_file = run_dir / "misclassified_samples.csv"
                if not error_file.exists() or error_file.stat().st_size == 0:
                    continue
                try:
                    error_table = pd.read_csv(error_file)
                except pd.errors.EmptyDataError:
                    continue
                if not error_table.empty:
                    error_table.insert(0, "run", run_dir.parent.name)
                    error_tables.append(error_table)
            all_errors = (
                pd.concat(error_tables, ignore_index=True)
                if error_tables
                else pd.DataFrame()
            )
            all_errors.to_csv(
                aggregate_dir / "misclassified_samples.csv",
                index=False,
            )
            summarize_error_pairs(all_errors).to_csv(
                aggregate_dir / "error_pair_summary.csv",
                index=False,
            )

            roc_files = [
                run_dir / "roc_curve_points.csv"
                for run_dir in run_dirs
                if (run_dir / "roc_curve_points.csv").exists()
            ]
            pr_files = [
                run_dir / "pr_curve_points.csv"
                for run_dir in run_dirs
                if (run_dir / "pr_curve_points.csv").exists()
            ]
            if roc_files:
                roc_points = average_curve_files(roc_files, "roc")
                roc_points.to_csv(
                    aggregate_dir / "roc_curve_points.csv",
                    index=False,
                )
                if create_plots:
                    save_curve_plot(
                        roc_points,
                        aggregate_dir / "roc_curves.png",
                        "roc",
                        title="Mean one-vs-rest ROC curves across seeds",
                    )
            if pr_files:
                pr_points = average_curve_files(pr_files, "pr")
                pr_points.to_csv(
                    aggregate_dir / "pr_curve_points.csv",
                    index=False,
                )
                if create_plots:
                    save_curve_plot(
                        pr_points,
                        aggregate_dir / "precision_recall_curves.png",
                        "pr",
                        title=(
                            "Mean one-vs-rest precision-recall curves "
                            "across seeds"
                        ),
                    )

            if create_plots:
                save_confusion_matrix_plot(
                    raw,
                    class_names,
                    aggregate_dir / "confusion_matrix_raw.png",
                )
                save_confusion_matrix_plot(
                    normalized_true,
                    class_names,
                    aggregate_dir / "confusion_matrix_normalized_true.png",
                    title="Confusion matrix normalized by true class",
                    value_format=".2f",
                )

            rebuilt.append(
                {
                    "dataset": dataset_dir.name,
                    "model": model_name,
                    "completed_seed_artifacts": len(run_dirs),
                    "output": str(aggregate_dir.resolve()),
                }
            )

    save_json(output_dir / "artifact_rebuild.json", rebuilt)
    return rebuilt


def main():
    args = parse_args()
    run_all = args.all_datasets or args.dataset is None

    if args.single_sample_repeats < 1:
        raise ValueError("--single-sample-repeats must be at least one.")
    if args.single_sample_warmup < 0:
        raise ValueError("--single-sample-warmup cannot be negative.")

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
        result_root = WORKSPACE_ROOT / "results" / "models-compare"
        experiment_folder = (
            f"{parameter_name}__normalization={args.normalization_mode}"
        )
        output_dir = (
            result_root
            / version
            / experiment_folder
        )
        if args.artifacts_only and not (output_dir / "artifacts").exists():
            completed_runs = [
                candidate
                for candidate in result_root.glob(f"*/{experiment_folder}")
                if (candidate / "artifacts").exists()
            ]
            if completed_runs:
                output_dir = max(
                    completed_runs,
                    key=lambda path: path.stat().st_mtime,
                )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.artifacts_only:
        rebuilt = rebuild_aggregate_artifacts(
            output_dir,
            create_plots=args.create_plots,
        )
        print(f"Rebuilt aggregate artifacts for {len(rebuilt)} model runs.")
        print(f"Output: {(output_dir / 'aggregate').resolve()}")
        return 0

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
            "single_sample_repeats": args.single_sample_repeats,
            "single_sample_warmup": args.single_sample_warmup,
            "create_plots": args.create_plots,
            "save_model": args.save_model,
            "metric_columns": METRIC_COLUMNS,
            "score_note": (
                "ROC/PR use class ranking scores. Log loss and Brier use "
                "native probabilities when available; distance and decision "
                "scores are converted with an uncalibrated softmax."
            ),
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

    primary_lim_model = (
        "RFF_LIM_NFST"
        if args.best_parameter_source == "rff_ref"
        else "LIM_NFST"
    )
    run_results = []
    per_class_runs = []
    errors = []
    gamma_by_run = {}
    score_methods = {}
    pooled_predictions = {}
    saved_models = {}

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
                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    encoder,
                    preprocessing,
                ) = preprocess_data(
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

            labels = np.arange(len(encoder.classes_), dtype=np.int64)
            label_names = [str(name) for name in encoder.classes_]
            training_labels = np.unique(y_train)

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

                    fit_started = perf_counter()
                    model.fit(X_train, y_train)
                    fit_seconds = perf_counter() - fit_started

                    if model_name == "RFF_LIM_NFST":
                        gamma_by_run[f"{dataset}__seed={seed}"] = (
                            model.rff_gamma_
                        )

                    predict_function = make_predict_function(
                        model,
                        model_name,
                        training_labels,
                    )
                    predict_started = perf_counter()
                    y_pred = np.asarray(predict_function(X_test))
                    predict_seconds = perf_counter() - predict_started

                    (
                        y_score,
                        y_proba,
                        score_method,
                        probability_method,
                    ) = get_scores_and_probabilities(
                        model,
                        model_name,
                        X_test,
                        labels,
                        training_labels,
                    )

                    single_sample_times = measure_single_sample_latency(
                        predict_function,
                        X_test,
                        repeats=args.single_sample_repeats,
                        warmup=args.single_sample_warmup,
                    )

                    summary = calculate_summary_metrics(
                        y_test,
                        y_pred,
                        y_score=y_score,
                        y_proba=y_proba,
                        labels=labels,
                        fit_seconds=fit_seconds,
                        predict_seconds=predict_seconds,
                        single_sample_times=single_sample_times,
                    )

                    row = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                    }
                    row.update(summary)
                    run_results.append(row)

                    run_key = (
                        f"{dataset}__seed={seed}__model={model_name}"
                    )
                    score_methods[run_key] = {
                        "score_method": score_method,
                        "probability_method": probability_method,
                    }
                    pool_key = (dataset, model_name)
                    if pool_key not in pooled_predictions:
                        pooled_predictions[pool_key] = {
                            "y_true": [],
                            "y_pred": [],
                            "y_score": [],
                            "y_proba": [],
                            "sample_indices": [],
                            "labels": labels,
                            "label_names": label_names,
                        }
                    pool = pooled_predictions[pool_key]
                    pool["y_true"].append(np.asarray(y_test))
                    pool["y_pred"].append(np.asarray(y_pred))
                    pool["y_score"].append(np.asarray(y_score))
                    pool["y_proba"].append(np.asarray(y_proba))
                    pool["sample_indices"].append(
                        np.asarray(
                            [
                                f"seed={seed}:test_index={sample_index}"
                                for sample_index in range(len(y_test))
                            ],
                            dtype=object,
                        )
                    )

                    artifact_dir = (
                        output_dir
                        / "artifacts"
                        / dataset
                        / f"seed_{seed}"
                        / model_name
                    )
                    artifact_result = save_evaluation_artifacts(
                        artifact_dir,
                        y_test,
                        y_pred,
                        y_score=y_score,
                        y_proba=y_proba,
                        labels=labels,
                        label_names=label_names,
                        summary_extra={
                            "dataset": dataset,
                            "model": model_name,
                            "seed": seed,
                            "score_method": score_method,
                            "probability_method": probability_method,
                            **summary,
                        },
                        sample_indices=np.arange(len(y_test)),
                        create_plots=False,
                    )
                    per_class = artifact_result["per_class"].copy()
                    per_class.insert(0, "seed", seed)
                    per_class.insert(0, "model", model_name)
                    per_class.insert(0, "dataset", dataset)
                    per_class_runs.append(per_class)

                    training_metadata = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "fit_seconds": fit_seconds,
                        "closed_form": model_name
                        in {"LIM_NFST", "RFF_LIM_NFST", "LDA", "kNFST"},
                        "iterations": (
                            np.asarray(model.n_iter_).tolist()
                            if hasattr(model, "n_iter_")
                            else None
                        ),
                        "loss_history_available": hasattr(
                            model,
                            "loss_curve_",
                        ),
                    }
                    save_json(
                        artifact_dir / "training_metadata.json",
                        training_metadata,
                    )
                    if hasattr(model, "loss_curve_"):
                        loss_values = np.asarray(model.loss_curve_)
                        save_training_history_plot(
                            {
                                "epoch": np.arange(1, len(loss_values) + 1),
                                "loss": loss_values,
                            },
                            artifact_dir / "training_loss.png",
                            title=f"Training loss: {model_name}",
                        )

                    if (
                        args.save_model
                        and model_name == primary_lim_model
                        and seed == seeds[0]
                    ):
                        model_file = (
                            output_dir
                            / "saved_models"
                            / dataset
                            / f"{model_name}__seed={seed}.joblib"
                        )
                        model_file.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(
                            {
                                "dataset": dataset,
                                "model_name": model_name,
                                "model": model,
                                "seed": seed,
                                "lim_code_version": version,
                                "best_parameters": parameters,
                                "normalization_mode": (
                                    args.normalization_mode
                                ),
                                "preprocessing": preprocessing,
                                "label_encoder": encoder,
                                "test_metrics": summary,
                                "input_note": (
                                    "Apply imputer, scaler, then optional "
                                    "common_lim normalization before predict. "
                                    "LOF is training-only."
                                ),
                            },
                            model_file,
                        )
                        saved_models[dataset] = str(model_file.resolve())

                    print(
                        f"dataset={dataset} seed={seed} model={model_name} "
                        f"accuracy={row['accuracy']:.4f} "
                        f"macro_f1={row['macro_f1']:.4f} "
                        f"fit={fit_seconds:.4f}s "
                        f"predict={predict_seconds:.4f}s"
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

    run_results_frame = pd.DataFrame(run_results)
    run_results_frame.to_csv(
        output_dir / "results_by_seed.csv",
        index=False,
        float_format="%.10f",
    )

    results = summarize_results(run_results, datasets, seeds)
    results.to_csv(
        output_dir / "results.csv",
        index=False,
        float_format="%.8f",
    )

    if per_class_runs:
        per_class_by_seed = pd.concat(per_class_runs, ignore_index=True)
        per_class_by_seed.to_csv(
            output_dir / "per_class_results_by_seed.csv",
            index=False,
            float_format="%.10f",
        )
        summarize_per_class_results(per_class_runs).to_csv(
            output_dir / "per_class_results.csv",
            index=False,
            float_format="%.10f",
        )

    for (dataset, model_name), pool in pooled_predictions.items():
        pooled_y_true = np.concatenate(pool["y_true"])
        pooled_y_pred = np.concatenate(pool["y_pred"])
        pooled_y_score = np.vstack(pool["y_score"])
        pooled_y_proba = np.vstack(pool["y_proba"])
        pooled_indices = np.concatenate(pool["sample_indices"])
        save_evaluation_artifacts(
            output_dir / "aggregate" / dataset / model_name,
            pooled_y_true,
            pooled_y_pred,
            y_score=pooled_y_score,
            y_proba=pooled_y_proba,
            labels=pool["labels"],
            label_names=pool["label_names"],
            summary_extra={
                "dataset": dataset,
                "model": model_name,
                "aggregation": "pooled_test_predictions_across_seeds",
                "completed_seeds": len(pool["y_true"]),
            },
            sample_indices=pooled_indices,
            create_plots=args.create_plots,
        )

    save_json(output_dir / "errors.json", errors)
    save_json(output_dir / "rff_gamma_by_run.json", gamma_by_run)
    save_json(output_dir / "score_methods.json", score_methods)
    save_json(output_dir / "saved_models.json", saved_models)

    print("\nFinal results:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
