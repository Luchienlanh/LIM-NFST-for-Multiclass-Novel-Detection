"""Run LIM experiments and tune LIM-NFST with Stratified K-fold CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLASSIFIER_ROOT.parent

sys.path.insert(0, str(CLASSIFIER_ROOT))

from limnfst.datasets import DATASETS, load_dataset
from limnfst.preprocessing import (
    SCALERS,
    make_scaler,
    preprocess_data,
    remove_training_outliers,
)
from limnfst.metrics import (
    CORE_METRIC_COLUMNS,
    CURVE_METRIC_COLUMNS,
    PROBABILITY_METRIC_COLUMNS,
    calculate_summary_metrics,
    distances_to_scores,
    save_cross_validation_ranking_plots,
    scores_to_probabilities,
)
from limnfst.models import LIM_NFST


TEST_SIZE = 0.20
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
DEFAULT_GRID_SCALERS = list(SCALERS)
DEFAULT_GRID_REFERENCE_SIZES = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_GRID_NEIGHBORS = [1, 3, 5, 7, 9]
DEFAULT_GRID_RFF_COMPONENTS = [128, 256, 512]
DEFAULT_GRID_RFF_GAMMA_MULTIPLIERS = [0.10, 1.0, 10.0]

GRID_TIMING_COLUMNS = [
    "fit_seconds",
    "predict_test_seconds",
    "predict_per_sample_seconds",
    "predict_throughput_samples_per_second",
]
METRIC_COLUMNS = [
    *CORE_METRIC_COLUMNS,
    *CURVE_METRIC_COLUMNS,
    *PROBABILITY_METRIC_COLUMNS,
    *GRID_TIMING_COLUMNS,
]
RESULT_COLUMNS = ["dataset", "model", *METRIC_COLUMNS]


def get_lim_version():
    """Create a short version from the current LIM-NFST source files."""
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


def calculate_metrics(
    y_true,
    y_pred,
    y_score=None,
    fit_seconds=None,
    predict_seconds=None,
):
    y_proba = (
        scores_to_probabilities(y_score)
        if y_score is not None
        else None
    )
    metrics = calculate_summary_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        y_proba=y_proba,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )
    return {
        metric: metrics[metric]
        for metric in METRIC_COLUMNS
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


def summarize_cross_validation(run_results):
    """Return mean and standard deviation across completed CV folds."""
    if not run_results:
        columns = ["dataset", "model", "completed_folds"]
        for metric in METRIC_COLUMNS:
            columns.extend([f"{metric}_mean", f"{metric}_std"])
        return pd.DataFrame(columns=columns)

    runs = pd.DataFrame(run_results)
    row = {
        "dataset": runs.iloc[0]["dataset"],
        "model": runs.iloc[0]["model"],
        "completed_folds": len(runs),
    }
    for metric in METRIC_COLUMNS:
        values = runs[metric]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=1)
    return pd.DataFrame([row])


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate or tune LIM-NFST with optional RFF."
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
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--use-rff", action="store_true")
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
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Number of StratifiedKFold splits used to score every grid "
            "configuration. Every configuration runs on every fold."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "macro_f1",
            "weighted_f1",
            "micro_f1",
            "balanced_accuracy",
            "accuracy",
            "mcc",
            "cohen_kappa",
            "roc_auc_ovr_macro",
            "average_precision_macro",
            "pr_auc_macro",
        ],
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
    elif args.grid_search:
        # K-fold already repeats every configuration. Use one CV seed by
        # default; --seeds explicitly enables repeated K-fold evaluation.
        seeds = [args.seed]
    elif run_all:
        seeds = DEFAULT_SEEDS
    else:
        seeds = [args.seed]

    return datasets, seeds, run_all


def preprocess_fixed_experiment(dataframe, dataset, scaler, seed):
    """Use the shared preprocessing owned by limnfst-classifier."""
    return preprocess_data(
        dataframe,
        dataset_name=dataset,
        scaler_name=scaler,
        random_state=seed,
        test_size=TEST_SIZE,
    )


def predict_closed_with_scores(model_name, model, X):
    """Return hard predictions and class scores where larger is better."""
    if model_name not in {"LIM_NFST", "RFF_LIM_NFST"}:
        raise ValueError(f"Unsupported LIM model: {model_name}")
    distances = model.reference_scores(X)
    y_score = distances_to_scores(distances)
    predicted_indices = np.argmax(y_score, axis=1)
    return model.classes_[predicted_indices], y_score


def preprocess_grid_fold(
    X_train,
    y_train,
    X_validation,
    y_validation,
    dataset,
    scaler,
    seed,
):
    """Fit preprocessing on one fold's training rows only."""
    X_train = X_train.copy()
    X_validation = X_validation.copy()
    y_train = y_train.copy()
    y_validation = y_validation.copy()

    order = y_train.argsort()
    X_train = X_train[order]
    y_train = y_train[order]

    if dataset == "IoTID20":
        X_train[np.isinf(X_train)] = np.nan
        X_validation[np.isinf(X_validation)] = np.nan
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_validation = imputer.transform(X_validation)

    if scaler == "None":
        X_train = np.nan_to_num(
            X_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        X_validation = np.nan_to_num(
            X_validation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        fitted_scaler = make_scaler(scaler, random_state=seed)
        X_train = fitted_scaler.fit_transform(X_train)
        X_validation = fitted_scaler.transform(X_validation)
        X_train = np.nan_to_num(
            X_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        X_validation = np.nan_to_num(
            X_validation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    X_train, y_train = remove_training_outliers(X_train, y_train)
    return X_train, y_train, X_validation, y_validation


def prepare_grid_folds(dataframe, dataset, scaler, seed, cv_folds):
    """Create leak-free StratifiedKFold data from outer training rows."""
    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]

    X_outer_train, _, y_outer_train, _ = train_test_split(
        X,
        y_raw,
        test_size=TEST_SIZE,
        stratify=y_raw,
        random_state=seed,
    )

    encoder = LabelEncoder()
    y_outer_encoded = encoder.fit_transform(y_outer_train)

    _, class_counts = np.unique(y_outer_encoded, return_counts=True)
    smallest_class = int(class_counts.min())
    if cv_folds > smallest_class:
        raise ValueError(
            f"cv_folds={cv_folds} is larger than the smallest outer-train "
            f"class ({smallest_class} samples)."
        )

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    prepared_folds = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(X_outer_train, y_outer_encoded),
        start=1,
    ):
        prepared = preprocess_grid_fold(
            X_outer_train[train_indices],
            y_outer_encoded[train_indices],
            X_outer_train[validation_indices],
            y_outer_encoded[validation_indices],
            dataset,
            scaler,
            seed,
        )
        prepared_folds.append(
            {
                "fold": fold_index,
                "X_train": prepared[0],
                "y_train": prepared[1],
                "X_validation": prepared[2],
                "y_validation": prepared[3],
            }
        )

    return prepared_folds


def run_fixed_experiment(args, datasets, seeds, run_all):
    if not 0.0 < args.reference_size <= 0.30:
        raise ValueError("--reference-size must be in (0, 0.30].")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least one.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than zero.")
    if args.use_rff and args.rff_components < 1:
        raise ValueError("--rff-components must be at least one.")
    if args.use_rff and args.rff_gamma_multiplier <= 0.0:
        raise ValueError("--rff-gamma-multiplier must be greater than zero.")

    version = get_lim_version()
    dataset_name = "all" if run_all else datasets[0]

    if args.output_dir is None:
        folder_name = (
            f"evaluation__dataset={dataset_name}__scaler={args.scaler}"
            f"__reference_size={number_to_name(args.reference_size)}"
            f"__neighbors={args.neighbors}"
        )
        if args.use_rff:
            folder_name += (
                f"__rff_components={args.rff_components}"
                "__rff_gamma_multiplier="
                f"{number_to_name(args.rff_gamma_multiplier)}"
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
        "epsilon": args.epsilon,
        "use_rff": args.use_rff,
        "rff_components": args.rff_components if args.use_rff else None,
        "rff_gamma_mode": (
            "scale_times_multiplier" if args.use_rff else None
        ),
        "rff_gamma_multiplier": (
            args.rff_gamma_multiplier if args.use_rff else None
        ),
        "classification_score": "negative_class_distance",
        "probability_mode": "softmax_score_uncalibrated",
    }
    save_json(output_dir / "parameters.json", manifest)

    print("Mode       : fixed LIM comparison")
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Scaler     : {args.scaler}")
    print(f"Reference  : {args.reference_size}")
    print(f"Neighbors  : {args.neighbors}")
    print(f"Use RFF    : {args.use_rff}")
    if args.use_rff:
        print(f"RFF dims   : {args.rff_components}")
        print(f"Gamma mult : {args.rff_gamma_multiplier}")
    print(f"Output     : {output_dir.resolve()}")

    run_results = []
    errors = []
    gamma_by_run = {}

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

                lim_model = LIM_NFST(
                    epsilon=args.epsilon,
                    reference_size=args.reference_size,
                    number_of_neighbors=args.neighbors,
                    random_state=seed,
                )
                models = [("LIM_NFST", lim_model)]

                if args.use_rff:
                    rff_model = LIM_NFST(
                        epsilon=args.epsilon,
                        reference_size=args.reference_size,
                        number_of_neighbors=args.neighbors,
                        random_state=seed,
                        use_rff=True,
                        rff_components=args.rff_components,
                        rff_gamma_multiplier=args.rff_gamma_multiplier,
                    )
                    models.insert(1, ("RFF_LIM_NFST", rff_model))

                for model_name, model in models:
                    fit_started = perf_counter()
                    model.fit(X_train, y_train)
                    fit_seconds = perf_counter() - fit_started

                    predict_started = perf_counter()
                    y_pred, y_score = predict_closed_with_scores(
                        model_name,
                        model,
                        X_test,
                    )
                    predict_seconds = perf_counter() - predict_started

                    if model_name == "RFF_LIM_NFST":
                        gamma_by_run[f"{dataset}__seed={seed}"] = (
                            model.rff_gamma_
                        )

                    row = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                    }
                    row.update(
                        calculate_metrics(
                            y_test,
                            y_pred,
                            y_score=y_score,
                            fit_seconds=fit_seconds,
                            predict_seconds=predict_seconds,
                        )
                    )
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
    if gamma_by_run:
        save_json(output_dir / "rff_gamma_by_run.json", gamma_by_run)

    print("\nFinal results:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


def run_grid_search(args, datasets, seeds, run_all):
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least two.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than zero.")
    if any(
        value <= 0.0 or value > 0.30
        for value in args.grid_reference_sizes
    ):
        raise ValueError("Grid reference sizes must be in (0, 0.30].")
    if any(value < 1 for value in args.grid_neighbors):
        raise ValueError("Grid neighbors must be at least one.")
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

    version = get_lim_version()
    grid_mode = "rff" if args.use_rff else "no-rff"
    output_dir = args.output_dir or (
        WORKSPACE_ROOT / "results" / "lim-models" / version / grid_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_rff:
        rff_settings = [
            (components, multiplier)
            for components in args.grid_rff_components
            for multiplier in args.grid_rff_gamma_multipliers
        ]
    else:
        rff_settings = [(None, None)]

    number_of_configurations = (
        len(args.grid_scalers)
        * len(args.grid_reference_sizes)
        * len(args.grid_neighbors)
        * len(rff_settings)
    )
    model_name = "RFF_LIM_NFST" if args.use_rff else "LIM_NFST"

    manifest = {
        "mode": f"{model_name} grid search",
        "lim_code_version": version,
        "datasets": datasets,
        "seeds": seeds,
        "scalers": args.grid_scalers,
        "reference_sizes": args.grid_reference_sizes,
        "neighbors": args.grid_neighbors,
        "epsilon": args.epsilon,
        "use_rff": args.use_rff,
        "rff_components": (
            args.grid_rff_components if args.use_rff else None
        ),
        "rff_gamma_mode": (
            "scale_times_multiplier" if args.use_rff else None
        ),
        "rff_gamma_multipliers": (
            args.grid_rff_gamma_multipliers if args.use_rff else None
        ),
        "selection_metric": args.selection_metric,
        "classification_score": "negative_reference_distance",
        "probability_mode": "softmax_score_uncalibrated",
        "cross_validation": "StratifiedKFold",
        "cv_folds": args.cv_folds,
        "cv_repeats": len(seeds),
        "evaluations_per_configuration": args.cv_folds * len(seeds),
        "configurations_per_dataset": number_of_configurations,
        "total_model_fits_per_dataset": (
            number_of_configurations * args.cv_folds * len(seeds)
        ),
        "outer_test_used_for_selection": False,
    }
    save_json(output_dir / "grid_parameters.json", manifest)

    print(f"Mode       : {model_name} grid search")
    print(f"Version    : {version}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Scalers    : {args.grid_scalers}")
    print(f"Ref sizes  : {args.grid_reference_sizes}")
    print(f"Neighbors  : {args.grid_neighbors}")
    print(f"Use RFF    : {args.use_rff}")
    if args.use_rff:
        print(f"RFF dims   : {args.grid_rff_components}")
        print(f"Gamma mult : {args.grid_rff_gamma_multipliers}")
    print(f"Configs/data: {number_of_configurations}")
    print(f"CV folds   : {args.cv_folds}")
    print(f"CV repeats : {len(seeds)}")
    print(
        "Fits/config: "
        f"{args.cv_folds * len(seeds)} "
        "(number of folds x number of seeds)"
    )
    print(
        "Fits/data  : "
        f"{number_of_configurations * args.cv_folds * len(seeds)}"
    )
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

        for scaler in args.grid_scalers:
            # Keep only one scaler's folds in memory. Every configuration
            # using this scaler receives exactly the same prepared folds.
            cross_validation_data = {}
            for seed in seeds:
                try:
                    cross_validation_data[seed] = prepare_grid_folds(
                        dataframe,
                        dataset,
                        scaler,
                        seed,
                        args.cv_folds,
                    )
                except Exception as error:
                    cross_validation_data[seed] = None
                    all_errors.append(
                        {
                            "dataset": dataset,
                            "scaler": scaler,
                            "seed": seed,
                            "stage": "cross_validation_preprocessing",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

            for rff_components, gamma_multiplier in rff_settings:
                for reference_size in args.grid_reference_sizes:
                    for neighbors in args.grid_neighbors:
                        configuration_name = (
                            f"scaler={scaler}"
                            f"__reference_size="
                            f"{number_to_name(reference_size)}"
                            f"__neighbors={neighbors}"
                        )
                        if args.use_rff:
                            configuration_name += (
                                f"__rff_components={rff_components}"
                                "__rff_gamma_multiplier="
                                f"{number_to_name(gamma_multiplier)}"
                            )
                        configuration_dir = (
                            output_dir / dataset / configuration_name
                        )
                        configuration_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        parameters = {
                            "dataset": dataset,
                            "model": model_name,
                            "scaler": scaler,
                            "reference_size": reference_size,
                            "neighbors": neighbors,
                            "epsilon": args.epsilon,
                            "use_rff": args.use_rff,
                            "rff_components": rff_components,
                            "rff_gamma_mode": (
                                "scale_times_multiplier"
                                if args.use_rff
                                else None
                            ),
                            "rff_gamma_multiplier": gamma_multiplier,
                            "seeds": seeds,
                            "cross_validation": "StratifiedKFold",
                            "cv_folds": args.cv_folds,
                            "cv_repeats": len(seeds),
                            "selection_metric": args.selection_metric,
                            "classification_score": (
                                "negative_reference_distance"
                            ),
                            "probability_mode": (
                                "softmax_score_uncalibrated"
                            ),
                            "lim_code_version": version,
                            "outer_test_used_for_selection": False,
                        }
                        save_json(
                            configuration_dir / "parameters.json",
                            parameters,
                        )

                        configuration_runs = []
                        configuration_errors = []
                        gamma_by_fold = {}

                        for seed in seeds:
                            prepared_folds = cross_validation_data[seed]
                            if prepared_folds is None:
                                configuration_errors.append(
                                    {
                                        "seed": seed,
                                        "fold": None,
                                        "error": "Preprocessing failed.",
                                    }
                                )
                                continue

                            for fold_data in prepared_folds:
                                fold_index = fold_data["fold"]
                                X_train = fold_data["X_train"]
                                y_train = fold_data["y_train"]
                                X_validation = fold_data["X_validation"]
                                y_validation = fold_data["y_validation"]

                                try:
                                    model = LIM_NFST(
                                        epsilon=args.epsilon,
                                        reference_size=reference_size,
                                        number_of_neighbors=neighbors,
                                        random_state=seed,
                                        use_rff=args.use_rff,
                                        rff_components=(
                                            rff_components
                                            if args.use_rff
                                            else 256
                                        ),
                                        rff_gamma_multiplier=(
                                            gamma_multiplier
                                            if args.use_rff
                                            else 1.0
                                        ),
                                    )
                                    fit_started = perf_counter()
                                    model.fit(X_train, y_train)
                                    fit_seconds = (
                                        perf_counter() - fit_started
                                    )

                                    predict_started = perf_counter()
                                    distance_matrix = (
                                        model.reference_scores(
                                            X_validation
                                        )
                                    )
                                    predict_seconds = (
                                        perf_counter() - predict_started
                                    )
                                    y_score = distances_to_scores(
                                        distance_matrix
                                    )
                                    predicted_indices = np.argmax(
                                        y_score,
                                        axis=1,
                                    )
                                    y_pred = model.classes_[
                                        predicted_indices
                                    ]

                                    if args.use_rff:
                                        gamma_key = (
                                            f"seed={seed}__fold={fold_index}"
                                        )
                                        gamma_by_fold[gamma_key] = (
                                            model.rff_gamma_
                                        )

                                    row = {
                                        "dataset": dataset,
                                        "model": model_name,
                                        "seed": seed,
                                        "fold": fold_index,
                                    }
                                    row.update(
                                        calculate_metrics(
                                            y_validation,
                                            y_pred,
                                            y_score=y_score,
                                            fit_seconds=fit_seconds,
                                            predict_seconds=predict_seconds,
                                        )
                                    )
                                    configuration_runs.append(row)
                                except Exception as error:
                                    message = (
                                        f"{type(error).__name__}: {error}"
                                    )
                                    configuration_errors.append(
                                        {
                                            "seed": seed,
                                            "fold": fold_index,
                                            "error": message,
                                        }
                                    )
                                    all_errors.append(
                                        {
                                            "dataset": dataset,
                                            "scaler": scaler,
                                            "reference_size": (
                                                reference_size
                                            ),
                                            "neighbors": neighbors,
                                            "rff_components": (
                                                rff_components
                                            ),
                                            "rff_gamma_multiplier": (
                                                gamma_multiplier
                                            ),
                                            "seed": seed,
                                            "fold": fold_index,
                                            "error": message,
                                        }
                                    )

                        if gamma_by_fold:
                            save_json(
                                configuration_dir
                                / "rff_gamma_by_fold.json",
                                gamma_by_fold,
                            )

                        pd.DataFrame(configuration_runs).to_csv(
                            configuration_dir / "fold_results.csv",
                            index=False,
                            float_format="%.8f",
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
                        cv_summary = summarize_cross_validation(
                            configuration_runs
                        )
                        cv_summary.to_csv(
                            configuration_dir / "cv_summary.csv",
                            index=False,
                            float_format="%.8f",
                        )

                        if configuration_errors:
                            save_json(
                                configuration_dir / "errors.json",
                                configuration_errors,
                            )

                        expected_runs = len(seeds) * args.cv_folds
                        complete = (
                            len(configuration_runs) == expected_runs
                        )
                        if complete and not configuration_result.empty:
                            metric_values = (
                                configuration_result.iloc[0].to_dict()
                            )
                            metric_standard_deviation = cv_summary.iloc[
                                0
                            ][f"{args.selection_metric}_std"]

                            report_row = {
                                "dataset": dataset,
                                "model": (
                                    f"{model_name}[{configuration_name}]"
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
                                "epsilon": args.epsilon,
                                "use_rff": args.use_rff,
                                "rff_components": rff_components,
                                "rff_gamma_multiplier": gamma_multiplier,
                                "selection_metric_std": (
                                    metric_standard_deviation
                                ),
                            }
                            for metric in METRIC_COLUMNS:
                                candidate[metric] = metric_values[metric]
                            candidates.append(candidate)

                            message = (
                                f"dataset={dataset} scaler={scaler} "
                                f"ref={reference_size:g} k={neighbors}"
                            )
                            if args.use_rff:
                                message += (
                                    f" rff={rff_components} "
                                    f"gamma_mult={gamma_multiplier:g}"
                                )
                            message += (
                                f" {args.selection_metric}="
                                f"{candidate[args.selection_metric]:.4f}"
                            )
                            print(message)
                        else:
                            print(
                                f"FAILED dataset={dataset} "
                                f"configuration={configuration_name}"
                            )

    grid_results = pd.DataFrame(all_grid_results, columns=RESULT_COLUMNS)
    grid_results.to_csv(
        output_dir / "grid_search_results.csv",
        index=False,
        float_format="%.8f",
    )

    best_parameters = []
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(
        output_dir / "grid_candidates.csv",
        index=False,
        float_format="%.8f",
    )

    if candidate_frame.empty:
        if all_errors:
            save_json(output_dir / "errors.json", all_errors)
        print("No grid configuration completed every CV fold.")
        return 2

    try:
        save_cross_validation_ranking_plots(
            candidate_frame,
            args.selection_metric,
            output_dir / "plots",
        )
    except Exception as error:
        all_errors.append(
            {
                "stage": "cross_validation_plot",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        print(f"WARNING: could not create CV ranking plots: {error}")

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
                    "selection_metric_std",
                    "accuracy",
                    "mcc",
                    "reference_size",
                    "neighbors",
                    "rff_components" if args.use_rff else None,
                ]
            )
        )
        ranking_columns = [
            column for column in ranking_columns if column is not None
        ]
        descending_metrics = set(
            [*CORE_METRIC_COLUMNS, *CURVE_METRIC_COLUMNS]
        )
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
                "epsilon": float(best["epsilon"]),
                "use_rff": args.use_rff,
                "rff_components": (
                    int(best["rff_components"])
                    if args.use_rff
                    else None
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
                "validation_score": best[args.selection_metric],
                "validation_score_std": best[
                    "selection_metric_std"
                ],
                "cross_validation": "StratifiedKFold",
                "cv_folds": args.cv_folds,
                "cv_repeats": len(seeds),
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
