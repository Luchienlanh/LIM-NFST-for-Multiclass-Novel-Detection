"""Run one fixed RFF-REF-LIM configuration for MND/LOCO.

There is deliberately no grid search: each invocation evaluates one
configuration. The threshold modes are paper (tau = (delta / 2)^2) and
quantile (leave-one-out calibration per known class).
"""

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
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLASSIFIER_ROOT))

from limnfst.datasets import DATASETS, DEFAULT_LIMITS, load_dataset
from limnfst.models import LIM_NFST
from limnfst.preprocessing import (
    SCALERS,
    make_scaler,
    remove_training_outliers,
)

TEST_SIZE = 0.20
NOVEL_LABEL = -1
MODEL_NAME = "RFF-REF-LIM"
SCRIPT_NAME = "rff_ref_lim_mnd_fixed_v1"
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
RUN_COLUMNS = ["dataset", "model", "novel_class", "test_set", *METRIC_COLUMNS]
SUMMARY_COLUMNS = ["dataset", "model", "test_set", *METRIC_COLUMNS]


def get_version() -> str:
    source_files = [
        Path(__file__),
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
    return f"{SCRIPT_NAME}_{code_hash.hexdigest()[:8]}"


def save_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed RFF-REF-LIM MND/LOCO configuration."
    )
    parser.add_argument("--dataset", choices=DATASETS, default="BoT_IoT")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Row limit; defaults to the dataset's standard limit.",
    )
    parser.add_argument(
        "--novel-class",
        default=None,
        help="Held-out raw label or zero-based index; omit for full LOCO.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--threshold-mode",
        choices=["paper", "quantile"],
        default="paper",
        help="paper: tau=(delta/2)^2; quantile: adaptive class threshold.",
    )
    parser.add_argument("--novelty-quantile", type=float, default=0.95)
    parser.add_argument(
        "--delta",
        "--paper-delta",
        dest="paper_delta",
        type=float,
        default=2.0,
        help="Paper parameter delta, where tau=(delta/2)^2.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least one.")
    if not 0.0 < args.reference_size <= 0.30:
        raise ValueError("--reference-size must be in (0, 0.30].")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least one.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than zero.")
    if args.rff_components < 1:
        raise ValueError("--rff-components must be at least one.")
    if args.rff_gamma_multiplier <= 0.0:
        raise ValueError("--rff-gamma-multiplier must be greater than zero.")
    if args.threshold_mode == "quantile" and not (
        0.0 < args.novelty_quantile < 1.0
    ):
        raise ValueError("--novelty-quantile must be between zero and one.")
    if args.threshold_mode == "paper" and args.paper_delta <= 0.0:
        raise ValueError("--delta must be greater than zero.")


def resolve_novel_class(y_raw: np.ndarray, requested_class: object):
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
            f"Novel class index {class_index} is outside [0, {len(classes) - 1}]."
        )
    return classes[class_index]


def resolve_novel_classes(
    dataframe: pd.DataFrame,
    requested_class: object,
) -> list[object]:
    y_raw = dataframe.iloc[:, -1].to_numpy()
    if requested_class is not None:
        return [resolve_novel_class(y_raw, requested_class)]
    classes = np.unique(y_raw)
    if len(classes) < 2:
        raise ValueError("MND requires at least two classes.")
    return classes.tolist()


def prepare_mnd_data(
    dataframe: pd.DataFrame,
    requested_novel_class: object,
    scaler: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    data = dataframe.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]
    novel_class = resolve_novel_class(y_raw, requested_novel_class)
    novel_mask = y_raw == novel_class
    X_known = X[~novel_mask]
    y_known_raw = y_raw[~novel_mask]
    X_novel = X[novel_mask]

    X_train, X_known_test, y_train_raw, y_known_test_raw = train_test_split(
        X_known,
        y_known_raw,
        test_size=TEST_SIZE,
        stratify=y_known_raw,
        random_state=seed,
    )
    X_train = np.asarray(X_train, dtype=np.float64).copy()
    X_known_test = np.asarray(X_known_test, dtype=np.float64).copy()
    X_novel = np.asarray(X_novel, dtype=np.float64).copy()
    for values in (X_train, X_known_test, X_novel):
        values[np.isinf(values)] = np.nan

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_known_test = imputer.transform(X_known_test)
    X_novel = imputer.transform(X_novel)
    if scaler != "None":
        fitted_scaler = make_scaler(scaler, random_state=seed)
        X_train = fitted_scaler.fit_transform(X_train)
        X_known_test = fitted_scaler.transform(X_known_test)
        X_novel = fitted_scaler.transform(X_novel)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_known_test = np.nan_to_num(X_known_test, nan=0.0)
    X_novel = np.nan_to_num(X_novel, nan=0.0)
    order = y_train_raw.argsort()
    X_train, y_train_raw = X_train[order], y_train_raw[order]
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    X_train, y_train = remove_training_outliers(X_train, y_train)
    y_known_test = label_encoder.transform(y_known_test_raw)
    return X_train, y_train, X_known_test, y_known_test, X_novel, novel_class


def make_model(args: argparse.Namespace) -> LIM_NFST:
    model_threshold_mode = (
        "quantile" if args.threshold_mode == "quantile" else "delta"
    )
    return LIM_NFST(
        epsilon=args.epsilon,
        reference_size=args.reference_size,
        number_of_neighbors=args.neighbors,
        novelty_quantile=args.novelty_quantile,
        threshold_mode=model_threshold_mode,
        novelty_delta=args.paper_delta,
        random_state=args.seed,
        use_rff=True,
        rff_components=args.rff_components,
        rff_gamma_multiplier=args.rff_gamma_multiplier,
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
    novel_precision, novel_recall, novel_f1, _ = precision_recall_fscore_support(
        is_novel_true,
        is_novel_pred,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred_open,
        average="macro",
        zero_division=0,
    )
    mcc = (
        matthews_corrcoef(y_true, y_pred_open)
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_open)) > 1
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
    dataset: str,
    novel_class: object,
    X_known: np.ndarray,
    y_known: np.ndarray,
    X_novel: np.ndarray,
) -> list[dict[str, object]]:
    y_novel = np.full(len(X_novel), NOVEL_LABEL, dtype=int)
    evaluation_sets = [
        ("known", X_known, y_known),
        ("novel", X_novel, y_novel),
        (
            "combined",
            np.vstack([X_known, X_novel]),
            np.concatenate([y_known, y_novel]),
        ),
    ]
    rows = []
    for test_set, X_test, y_true in evaluation_sets:
        y_open = model.predict_open(X_test, novel_label=NOVEL_LABEL)
        y_closed = model.predict_closed(X_test)
        row = {
            "dataset": dataset,
            "model": MODEL_NAME,
            "novel_class": str(novel_class),
            "test_set": test_set,
        }
        row.update(calculate_metrics(y_true, y_open, y_closed))
        rows.append(row)
    return rows


def output_dir(args: argparse.Namespace, version: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    configuration = {
        "limit": args.limit or DEFAULT_LIMITS[args.dataset],
        "novel_class": args.novel_class,
        "seed": args.seed,
        "scaler": args.scaler,
        "reference_size": args.reference_size,
        "neighbors": args.neighbors,
        "epsilon": args.epsilon,
        "rff_components": args.rff_components,
        "rff_gamma_multiplier": args.rff_gamma_multiplier,
        "threshold_mode": args.threshold_mode,
        "novelty_quantile": (
            args.novelty_quantile if args.threshold_mode == "quantile" else None
        ),
        "paper_delta": args.paper_delta if args.threshold_mode == "paper" else None,
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return (
        CLASSIFIER_ROOT
        / "results"
        / "mnd-rff-ref-lim"
        / version
        / args.dataset
        / f"cfg_{configuration_id}"
    )


def parameters_payload(args: argparse.Namespace, version: str) -> dict[str, object]:
    paper_threshold = 0.25 * args.paper_delta**2
    return {
        "mode": "fixed RFF-REF-LIM MND evaluation",
        "model_display_name": MODEL_NAME,
        "version": version,
        "dataset": args.dataset,
        "limit": args.limit or DEFAULT_LIMITS[args.dataset],
        "seed": args.seed,
        "novel_class": args.novel_class,
        "scaler": args.scaler,
        "test_size": TEST_SIZE,
        "reference_size": args.reference_size,
        "neighbors": args.neighbors,
        "epsilon": args.epsilon,
        "use_rff": True,
        "rff_components": args.rff_components,
        "rff_gamma": "scale_times_multiplier",
        "rff_gamma_multiplier": args.rff_gamma_multiplier,
        "threshold_mode": args.threshold_mode,
        "novelty_quantile": (
            args.novelty_quantile if args.threshold_mode == "quantile" else None
        ),
        "paper_delta": args.paper_delta if args.threshold_mode == "paper" else None,
        "paper_threshold_tau": (
            paper_threshold if args.threshold_mode == "paper" else None
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    version = get_version()
    results_dir = output_dir(args, version)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "parameters.json", parameters_payload(args, version))

    row_limit = args.limit or DEFAULT_LIMITS[args.dataset]
    dataframe, _ = load_dataset(args.dataset, row_limit)
    novel_classes = resolve_novel_classes(dataframe, args.novel_class)
    save_json(
        results_dir / "novel_classes.json",
        {args.dataset: [str(novel_class) for novel_class in novel_classes]},
    )
    print("Mode       : fixed RFF-REF-LIM MND evaluation")
    print(f"Dataset    : {args.dataset} (limit={row_limit})")
    print(f"Seed       : {args.seed}")
    print(f"Novel class: {args.novel_class or 'all classes (full LOCO)'}")
    print(f"Scaler     : {args.scaler}")
    print(f"Reference  : {args.reference_size}")
    print(f"Neighbors  : {args.neighbors}")
    print(f"RFF dims   : {args.rff_components}")
    print(f"Gamma mult : {args.rff_gamma_multiplier}")
    if args.threshold_mode == "quantile":
        print(f"Threshold  : quantile={args.novelty_quantile}")
    else:
        threshold = 0.25 * args.paper_delta**2
        print(
            "Threshold  : paper "
            f"delta={args.paper_delta}, tau=(delta/2)^2={threshold}"
        )
    print(f"Output     : {results_dir.resolve()}")

    run_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    rff_gamma_by_novel_class: dict[str, float] = {}
    for novel_class in novel_classes:
        try:
            prepared = prepare_mnd_data(
                dataframe,
                novel_class,
                args.scaler,
                args.seed,
            )
            X_train, y_train, X_known, y_known, X_novel, resolved_class = prepared
            model = make_model(args)
            model.fit(X_train, y_train)
            class_rows = evaluate_model(
                model,
                args.dataset,
                resolved_class,
                X_known,
                y_known,
                X_novel,
            )
            run_rows.extend(class_rows)
            rff_gamma_by_novel_class[str(resolved_class)] = float(model.rff_gamma_)
            combined = next(
                row for row in class_rows if row["test_set"] == "combined"
            )
            print(
                f"novel_class={resolved_class} "
                f"accuracy={combined['accuracy_with_novel']:.4f} "
                f"novel_f1={combined['novel_detection_f1']:.4f}"
            )
        except Exception as error:
            errors.append(
                {
                    "dataset": args.dataset,
                    "novel_class": str(novel_class),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"FAILED novel_class={novel_class}: {error}")

    class_results = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    if class_results.empty:
        results = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        results = (
            class_results.groupby(
                ["dataset", "model", "test_set"], as_index=False
            )[METRIC_COLUMNS]
            .mean()
            .reindex(columns=SUMMARY_COLUMNS)
        )
    results.to_csv(results_dir / "results.csv", index=False, float_format="%.8f")
    class_results.to_csv(
        results_dir / "results_by_novel_class.csv",
        index=False,
        float_format="%.8f",
    )
    save_json(results_dir / "rff_gamma_by_novel_class.json", rff_gamma_by_novel_class)
    if errors:
        save_json(results_dir / "errors.json", errors)

    print("\nFinal mean results across held-out classes:")
    print(results.to_string(index=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
