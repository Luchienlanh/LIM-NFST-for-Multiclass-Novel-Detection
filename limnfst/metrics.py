"""Metrics, curves, timings and error-analysis helpers for LIM experiments.

All quality metrics use the 0-to-1 scale. PR AUC is trapezoidal area under
the precision-recall curve; Average Precision is also reported because it is
a different, commonly used summary of the same ranking scores.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


AVERAGES = ("micro", "macro", "weighted")

CORE_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_specificity",
    "weighted_specificity",
    "mcc",
    "cohen_kappa",
]

CURVE_METRIC_COLUMNS = [
    "roc_auc_ovr_micro",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "average_precision_micro",
    "average_precision_macro",
    "average_precision_weighted",
    "pr_auc_micro",
    "pr_auc_macro",
    "pr_auc_weighted",
]

PROBABILITY_METRIC_COLUMNS = [
    "log_loss",
    "multiclass_brier_score",
]

TIMING_METRIC_COLUMNS = [
    "fit_seconds",
    "predict_test_seconds",
    "predict_per_sample_seconds",
    "predict_throughput_samples_per_second",
    "single_sample_mean_seconds",
    "single_sample_median_seconds",
    "single_sample_p95_seconds",
]

SUMMARY_METRIC_COLUMNS = [
    *CORE_METRIC_COLUMNS,
    *CURVE_METRIC_COLUMNS,
    *PROBABILITY_METRIC_COLUMNS,
    *TIMING_METRIC_COLUMNS,
]


def _as_one_dimensional(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def _resolve_labels(y_true, y_pred, labels=None):
    if labels is None:
        return np.unique(np.concatenate([y_true, y_pred]))

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if len(labels) < 2:
        raise ValueError("At least two labels are required.")
    return labels


def _binarize_labels(y_true, labels):
    binary = label_binarize(y_true, classes=labels)

    # sklearn returns one column for binary classification. The rest of this
    # module always uses one score column per class.
    if len(labels) == 2 and binary.shape[1] == 1:
        binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])

    return binary.astype(np.int64, copy=False)


def _validate_score_matrix(y_score, number_of_samples, labels, name):
    if y_score is None:
        return None

    score_matrix = np.asarray(y_score, dtype=np.float64)
    expected_shape = (number_of_samples, len(labels))
    if score_matrix.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"got {score_matrix.shape}."
        )
    if not np.isfinite(score_matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return score_matrix


def distances_to_scores(distance_matrix):
    """Convert class distances into scores where larger means more likely."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be two-dimensional.")
    if not np.isfinite(distances).all():
        raise ValueError("distance_matrix contains NaN or infinite values.")
    return -distances


def scores_to_probabilities(score_matrix):
    """Apply a stable softmax to class scores.

    This creates probability-like values for log loss and Brier score. They
    are not calibrated probabilities unless calibration is done separately.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional.")
    if not np.isfinite(scores).all():
        raise ValueError("score_matrix contains NaN or infinite values.")

    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    row_sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / row_sums


def _safe_per_class_curve_metrics(y_binary, score_matrix):
    number_of_classes = y_binary.shape[1]
    roc_auc_by_class = np.full(number_of_classes, np.nan)
    average_precision_by_class = np.full(number_of_classes, np.nan)
    pr_auc_by_class = np.full(number_of_classes, np.nan)

    for class_index in range(number_of_classes):
        binary_target = y_binary[:, class_index]
        class_score = score_matrix[:, class_index]

        if len(np.unique(binary_target)) < 2:
            continue

        roc_auc_by_class[class_index] = roc_auc_score(
            binary_target,
            class_score,
        )
        average_precision_by_class[class_index] = average_precision_score(
            binary_target,
            class_score,
        )
        precision, recall, _ = precision_recall_curve(
            binary_target,
            class_score,
        )
        pr_auc_by_class[class_index] = auc(recall, precision)

    return (
        roc_auc_by_class,
        average_precision_by_class,
        pr_auc_by_class,
    )


def _weighted_nanmean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def calculate_per_class_metrics(
    y_true,
    y_pred,
    y_score=None,
    labels=None,
    label_names=None,
):
    """Return one row of metrics for every class."""
    y_true = _as_one_dimensional(y_true, "y_true")
    y_pred = _as_one_dimensional(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    labels = _resolve_labels(y_true, y_pred, labels)
    score_matrix = _validate_score_matrix(
        y_score,
        len(y_true),
        labels,
        "y_score",
    )

    if label_names is None:
        label_names = [str(label) for label in labels]
    if len(label_names) != len(labels):
        raise ValueError("label_names must have one value per label.")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()

    if score_matrix is not None:
        y_binary = _binarize_labels(y_true, labels)
        (
            roc_auc_by_class,
            average_precision_by_class,
            pr_auc_by_class,
        ) = _safe_per_class_curve_metrics(y_binary, score_matrix)
    else:
        roc_auc_by_class = np.full(len(labels), np.nan)
        average_precision_by_class = np.full(len(labels), np.nan)
        pr_auc_by_class = np.full(len(labels), np.nan)

    rows = []
    for class_index, (label, label_name) in enumerate(
        zip(labels, label_names)
    ):
        true_positive = int(matrix[class_index, class_index])
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        true_negative = int(
            total - true_positive - false_negative - false_positive
        )
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive

        precision = (
            true_positive / predicted_count if predicted_count else 0.0
        )
        recall = true_positive / support if support else 0.0
        specificity_denominator = true_negative + false_positive
        specificity = (
            true_negative / specificity_denominator
            if specificity_denominator
            else 0.0
        )
        negative_predictive_denominator = true_negative + false_negative
        negative_predictive_value = (
            true_negative / negative_predictive_denominator
            if negative_predictive_denominator
            else 0.0
        )
        false_positive_rate = (
            false_positive / specificity_denominator
            if specificity_denominator
            else 0.0
        )
        false_negative_rate = (
            false_negative / support if support else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        one_vs_rest_accuracy = (
            (true_positive + true_negative) / total if total else 0.0
        )

        rows.append(
            {
                "class_index": class_index,
                "label": label,
                "class_name": label_name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "negative_predictive_value": negative_predictive_value,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "f1": f1,
                "one_vs_rest_accuracy": one_vs_rest_accuracy,
                "roc_auc_ovr": roc_auc_by_class[class_index],
                "average_precision": average_precision_by_class[class_index],
                "pr_auc": pr_auc_by_class[class_index],
            }
        )

    return pd.DataFrame(rows)


def calculate_summary_metrics(
    y_true,
    y_pred,
    y_score=None,
    y_proba=None,
    labels=None,
    fit_seconds=None,
    predict_seconds=None,
    single_sample_times=None,
):
    """Calculate global multiclass metrics.

    Values are returned on the 0-to-1 scale. Timing values use seconds.
    """
    y_true = _as_one_dimensional(y_true, "y_true")
    y_pred = _as_one_dimensional(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("At least one sample is required.")

    labels = _resolve_labels(y_true, y_pred, labels)
    score_matrix = _validate_score_matrix(
        y_score,
        len(y_true),
        labels,
        "y_score",
    )
    probability_matrix = _validate_score_matrix(
        y_proba,
        len(y_true),
        labels,
        "y_proba",
    )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }

    for average in AVERAGES:
        results[f"{average}_precision"] = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
        )
        results[f"{average}_recall"] = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
        )
        results[f"{average}_f1"] = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
        )

    per_class = calculate_per_class_metrics(
        y_true,
        y_pred,
        y_score=score_matrix,
        labels=labels,
    )
    class_support = per_class["support"].to_numpy(dtype=np.float64)
    class_specificity = per_class["specificity"].to_numpy(dtype=np.float64)
    results["macro_specificity"] = float(class_specificity.mean())
    results["weighted_specificity"] = _weighted_nanmean(
        class_specificity,
        class_support,
    )

    for metric_name in CURVE_METRIC_COLUMNS:
        results[metric_name] = np.nan

    ranking_scores = (
        score_matrix if score_matrix is not None else probability_matrix
    )
    if ranking_scores is not None:
        y_binary = _binarize_labels(y_true, labels)
        (
            roc_auc_by_class,
            average_precision_by_class,
            pr_auc_by_class,
        ) = _safe_per_class_curve_metrics(y_binary, ranking_scores)

        results["roc_auc_ovr_macro"] = float(
            np.nanmean(roc_auc_by_class)
        )
        results["roc_auc_ovr_weighted"] = _weighted_nanmean(
            roc_auc_by_class,
            class_support,
        )
        results["average_precision_macro"] = float(
            np.nanmean(average_precision_by_class)
        )
        results["average_precision_weighted"] = _weighted_nanmean(
            average_precision_by_class,
            class_support,
        )
        results["pr_auc_macro"] = float(np.nanmean(pr_auc_by_class))
        results["pr_auc_weighted"] = _weighted_nanmean(
            pr_auc_by_class,
            class_support,
        )

        flattened_target = y_binary.ravel()
        flattened_scores = ranking_scores.ravel()
        if len(np.unique(flattened_target)) == 2:
            results["roc_auc_ovr_micro"] = roc_auc_score(
                flattened_target,
                flattened_scores,
            )
            results["average_precision_micro"] = average_precision_score(
                flattened_target,
                flattened_scores,
            )
            micro_precision, micro_recall, _ = precision_recall_curve(
                flattened_target,
                flattened_scores,
            )
            results["pr_auc_micro"] = auc(
                micro_recall,
                micro_precision,
            )

    results["log_loss"] = np.nan
    results["multiclass_brier_score"] = np.nan
    if probability_matrix is not None:
        if (probability_matrix < 0.0).any():
            raise ValueError("y_proba cannot contain negative values.")
        row_sums = probability_matrix.sum(axis=1, keepdims=True)
        if (row_sums <= 0.0).any():
            raise ValueError("Every y_proba row must have a positive sum.")
        probability_matrix = probability_matrix / row_sums

        results["log_loss"] = log_loss(
            y_true,
            probability_matrix,
            labels=labels,
        )
        y_binary = _binarize_labels(y_true, labels)
        results["multiclass_brier_score"] = float(
            np.mean(np.sum((probability_matrix - y_binary) ** 2, axis=1))
        )

    for metric_name in TIMING_METRIC_COLUMNS:
        results[metric_name] = np.nan

    if fit_seconds is not None:
        results["fit_seconds"] = float(fit_seconds)
    if predict_seconds is not None:
        predict_seconds = float(predict_seconds)
        results["predict_test_seconds"] = predict_seconds
        results["predict_per_sample_seconds"] = predict_seconds / len(y_true)
        if predict_seconds > 0.0:
            results["predict_throughput_samples_per_second"] = (
                len(y_true) / predict_seconds
            )

    if single_sample_times is not None:
        times = np.asarray(single_sample_times, dtype=np.float64)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError(
                "single_sample_times must be a non-empty 1D array."
            )
        if not np.isfinite(times).all() or (times < 0.0).any():
            raise ValueError("single_sample_times contains invalid values.")
        results["single_sample_mean_seconds"] = float(times.mean())
        results["single_sample_median_seconds"] = float(np.median(times))
        results["single_sample_p95_seconds"] = float(
            np.quantile(times, 0.95)
        )

    return results


def calculate_confusion_matrices(y_true, y_pred, labels=None):
    """Return raw, normalized and one-vs-rest confusion matrices."""
    y_true = _as_one_dimensional(y_true, "y_true")
    y_pred = _as_one_dimensional(y_pred, "y_pred")
    labels = _resolve_labels(y_true, y_pred, labels)

    raw = confusion_matrix(y_true, y_pred, labels=labels)
    normalized_true = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )
    normalized_pred = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="pred",
    )
    normalized_all = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="all",
    )

    one_vs_rest = {}
    total = raw.sum()
    for class_index, label in enumerate(labels):
        true_positive = raw[class_index, class_index]
        false_negative = raw[class_index, :].sum() - true_positive
        false_positive = raw[:, class_index].sum() - true_positive
        true_negative = (
            total - true_positive - false_negative - false_positive
        )
        one_vs_rest[str(label)] = np.asarray(
            [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            dtype=np.int64,
        )

    return {
        "labels": labels,
        "raw": raw,
        "normalized_true": normalized_true,
        "normalized_pred": normalized_pred,
        "normalized_all": normalized_all,
        "one_vs_rest": one_vs_rest,
    }


def calculate_roc_curve_points(y_true, y_score, labels):
    """Return long-form ROC curve points for each class and micro average."""
    y_true = _as_one_dimensional(y_true, "y_true")
    labels = np.asarray(labels)
    scores = _validate_score_matrix(
        y_score,
        len(y_true),
        labels,
        "y_score",
    )
    y_binary = _binarize_labels(y_true, labels)

    rows = []
    class_curves = []
    for class_index, label in enumerate(labels):
        binary_target = y_binary[:, class_index]
        if len(np.unique(binary_target)) < 2:
            continue
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            binary_target,
            scores[:, class_index],
        )
        class_curves.append(
            (
                false_positive_rate,
                true_positive_rate,
                int(binary_target.sum()),
            )
        )
        for point_index in range(len(thresholds)):
            rows.append(
                {
                    "curve": "class",
                    "class_index": class_index,
                    "label": label,
                    "threshold": thresholds[point_index],
                    "false_positive_rate": false_positive_rate[point_index],
                    "true_positive_rate": true_positive_rate[point_index],
                }
            )

    if class_curves:
        common_false_positive_rate = np.linspace(0.0, 1.0, 501)
        interpolated_rates = np.vstack(
            [
                np.interp(
                    common_false_positive_rate,
                    false_positive_rate,
                    true_positive_rate,
                )
                for false_positive_rate, true_positive_rate, _ in class_curves
            ]
        )
        supports = np.asarray(
            [support for _, _, support in class_curves],
            dtype=np.float64,
        )
        average_curves = {
            "MACRO": interpolated_rates.mean(axis=0),
            "WEIGHTED": np.average(
                interpolated_rates,
                axis=0,
                weights=supports,
            ),
        }
        for average_name, average_rate in average_curves.items():
            for point_index in range(len(common_false_positive_rate)):
                rows.append(
                    {
                        "curve": average_name.lower(),
                        "class_index": -1,
                        "label": average_name,
                        "threshold": np.nan,
                        "false_positive_rate": (
                            common_false_positive_rate[point_index]
                        ),
                        "true_positive_rate": average_rate[point_index],
                    }
                )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_binary.ravel(),
        scores.ravel(),
    )
    for point_index in range(len(thresholds)):
        rows.append(
            {
                "curve": "micro",
                "class_index": -1,
                "label": "MICRO",
                "threshold": thresholds[point_index],
                "false_positive_rate": false_positive_rate[point_index],
                "true_positive_rate": true_positive_rate[point_index],
            }
        )

    return pd.DataFrame(rows)


def calculate_pr_curve_points(y_true, y_score, labels):
    """Return long-form precision-recall points per class and micro."""
    y_true = _as_one_dimensional(y_true, "y_true")
    labels = np.asarray(labels)
    scores = _validate_score_matrix(
        y_score,
        len(y_true),
        labels,
        "y_score",
    )
    y_binary = _binarize_labels(y_true, labels)

    rows = []
    class_curves = []
    for class_index, label in enumerate(labels):
        binary_target = y_binary[:, class_index]
        if len(np.unique(binary_target)) < 2:
            continue
        precision, recall, thresholds = precision_recall_curve(
            binary_target,
            scores[:, class_index],
        )
        class_curves.append(
            (
                recall,
                precision,
                int(binary_target.sum()),
            )
        )
        thresholds = np.append(thresholds, np.nan)
        for point_index in range(len(precision)):
            rows.append(
                {
                    "curve": "class",
                    "class_index": class_index,
                    "label": label,
                    "threshold": thresholds[point_index],
                    "recall": recall[point_index],
                    "precision": precision[point_index],
                }
            )

    if class_curves:
        common_recall = np.linspace(0.0, 1.0, 501)
        interpolated_precisions = np.vstack(
            [
                np.interp(
                    common_recall,
                    recall[::-1],
                    precision[::-1],
                )
                for recall, precision, _ in class_curves
            ]
        )
        supports = np.asarray(
            [support for _, _, support in class_curves],
            dtype=np.float64,
        )
        average_curves = {
            "MACRO": interpolated_precisions.mean(axis=0),
            "WEIGHTED": np.average(
                interpolated_precisions,
                axis=0,
                weights=supports,
            ),
        }
        for average_name, average_precision in average_curves.items():
            for point_index in range(len(common_recall)):
                rows.append(
                    {
                        "curve": average_name.lower(),
                        "class_index": -1,
                        "label": average_name,
                        "threshold": np.nan,
                        "recall": common_recall[point_index],
                        "precision": average_precision[point_index],
                    }
                )

    precision, recall, thresholds = precision_recall_curve(
        y_binary.ravel(),
        scores.ravel(),
    )
    thresholds = np.append(thresholds, np.nan)
    for point_index in range(len(precision)):
        rows.append(
            {
                "curve": "micro",
                "class_index": -1,
                "label": "MICRO",
                "threshold": thresholds[point_index],
                "recall": recall[point_index],
                "precision": precision[point_index],
            }
        )

    return pd.DataFrame(rows)


def build_error_analysis(
    y_true,
    y_pred,
    y_score=None,
    labels=None,
    sample_indices=None,
):
    """Return only misclassified samples with confidence and score margin."""
    y_true = _as_one_dimensional(y_true, "y_true")
    y_pred = _as_one_dimensional(y_pred, "y_pred")
    labels = _resolve_labels(y_true, y_pred, labels)
    score_matrix = _validate_score_matrix(
        y_score,
        len(y_true),
        labels,
        "y_score",
    )

    if sample_indices is None:
        sample_indices = np.arange(len(y_true))
    sample_indices = np.asarray(sample_indices)
    if len(sample_indices) != len(y_true):
        raise ValueError("sample_indices must match y_true length.")

    label_to_index = {
        label: class_index
        for class_index, label in enumerate(labels)
    }
    wrong_indices = np.flatnonzero(y_true != y_pred)
    rows = []

    for row_index in wrong_indices:
        row = {
            "sample_index": sample_indices[row_index],
            "true_label": y_true[row_index],
            "predicted_label": y_pred[row_index],
            "error_pair": (
                f"{y_true[row_index]} -> {y_pred[row_index]}"
            ),
        }

        if score_matrix is not None:
            predicted_index = label_to_index[y_pred[row_index]]
            true_index = label_to_index[y_true[row_index]]
            sample_scores = score_matrix[row_index]
            descending = np.sort(sample_scores)[::-1]
            probabilities = scores_to_probabilities(
                sample_scores[np.newaxis, :]
            )[0]

            row["predicted_score"] = sample_scores[predicted_index]
            row["true_score"] = sample_scores[true_index]
            row["score_margin_top1_top2"] = (
                descending[0] - descending[1]
                if len(descending) > 1
                else np.nan
            )
            row["predicted_probability_like"] = probabilities[
                predicted_index
            ]
            row["true_probability_like"] = probabilities[true_index]

        rows.append(row)

    errors = pd.DataFrame(rows)
    if (
        not errors.empty
        and "predicted_probability_like" in errors.columns
    ):
        errors = errors.sort_values(
            "predicted_probability_like",
            ascending=False,
            kind="stable",
        )
    return errors.reset_index(drop=True)


def summarize_error_pairs(error_rows):
    """Count which true classes are most often confused with which labels."""
    errors = pd.DataFrame(error_rows)
    columns = [
        "true_label",
        "predicted_label",
        "error_count",
        "share_of_all_errors",
    ]
    if errors.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        errors.groupby(
            ["true_label", "predicted_label"],
            sort=False,
        )
        .size()
        .rename("error_count")
        .reset_index()
    )
    summary["share_of_all_errors"] = (
        summary["error_count"] / summary["error_count"].sum()
    )
    return summary.sort_values(
        "error_count",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def measure_single_sample_latency(
    predict_function,
    X_test,
    repeats=30,
    warmup=3,
    sample_index=0,
):
    """Measure standalone latency for one sample.

    This is different from batch_time / n_test, which is amortized latency.
    """
    X_test = np.asarray(X_test)
    if X_test.ndim != 2 or len(X_test) == 0:
        raise ValueError("X_test must be a non-empty two-dimensional matrix.")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative.")
    if not 0 <= sample_index < len(X_test):
        raise ValueError("sample_index is outside X_test.")

    one_sample = X_test[sample_index : sample_index + 1]
    for _ in range(warmup):
        predict_function(one_sample)

    timings = []
    for _ in range(repeats):
        started = perf_counter()
        predict_function(one_sample)
        timings.append(perf_counter() - started)
    return np.asarray(timings, dtype=np.float64)


def save_confusion_matrix_plot(
    matrix,
    labels,
    output_file,
    title="Confusion matrix",
    value_format="g",
):
    """Save one confusion matrix heatmap. Matplotlib is imported lazily."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure_size = max(7.0, min(18.0, 0.65 * len(labels) + 4.0))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=labels,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format=value_format,
        colorbar=True,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_curve_plot(
    curve_points,
    output_file,
    curve_type,
    title=None,
):
    """Save ROC or precision-recall curves from the long-form tables."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if curve_type not in {"roc", "pr"}:
        raise ValueError("curve_type must be 'roc' or 'pr'.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    for label, rows in curve_points.groupby("label", sort=False):
        if curve_type == "roc":
            axis.plot(
                rows["false_positive_rate"],
                rows["true_positive_rate"],
                label=str(label),
            )
        else:
            axis.plot(
                rows["recall"],
                rows["precision"],
                label=str(label),
            )

    if curve_type == "roc":
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        default_title = "One-vs-rest ROC curves"
    else:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        default_title = "One-vs-rest precision-recall curves"

    axis.set_title(title or default_title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_artifacts(
    output_dir,
    y_true,
    y_pred,
    y_score=None,
    y_proba=None,
    labels=None,
    label_names=None,
    summary_extra=None,
    sample_indices=None,
    create_plots=True,
):
    """Save summary, class metrics, matrices, curves and error rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = _as_one_dimensional(y_true, "y_true")
    y_pred = _as_one_dimensional(y_pred, "y_pred")
    labels = _resolve_labels(y_true, y_pred, labels)
    if label_names is None:
        label_names = [str(label) for label in labels]

    summary = calculate_summary_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        y_proba=y_proba,
        labels=labels,
    )
    if summary_extra:
        summary.update(summary_extra)
    pd.DataFrame([summary]).to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
        float_format="%.10f",
    )

    per_class = calculate_per_class_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        labels=labels,
        label_names=label_names,
    )
    per_class.to_csv(
        output_dir / "per_class_metrics.csv",
        index=False,
        float_format="%.10f",
    )

    matrices = calculate_confusion_matrices(y_true, y_pred, labels=labels)
    matrix_names = [
        "raw",
        "normalized_true",
        "normalized_pred",
        "normalized_all",
    ]
    for matrix_name in matrix_names:
        pd.DataFrame(
            matrices[matrix_name],
            index=label_names,
            columns=label_names,
        ).to_csv(
            output_dir / f"confusion_matrix_{matrix_name}.csv",
            float_format="%.10f",
        )

    one_vs_rest_rows = []
    for label, matrix in matrices["one_vs_rest"].items():
        one_vs_rest_rows.append(
            {
                "label": label,
                "true_negative": matrix[0, 0],
                "false_positive": matrix[0, 1],
                "false_negative": matrix[1, 0],
                "true_positive": matrix[1, 1],
            }
        )
    pd.DataFrame(one_vs_rest_rows).to_csv(
        output_dir / "confusion_matrix_one_vs_rest.csv",
        index=False,
    )

    errors = build_error_analysis(
        y_true,
        y_pred,
        y_score=y_score,
        labels=labels,
        sample_indices=sample_indices,
    )
    errors.to_csv(
        output_dir / "misclassified_samples.csv",
        index=False,
        float_format="%.10f",
    )
    summarize_error_pairs(errors).to_csv(
        output_dir / "error_pair_summary.csv",
        index=False,
        float_format="%.10f",
    )

    if y_score is not None:
        roc_points = calculate_roc_curve_points(y_true, y_score, labels)
        pr_points = calculate_pr_curve_points(y_true, y_score, labels)
        roc_points.to_csv(
            output_dir / "roc_curve_points.csv",
            index=False,
            float_format="%.10f",
        )
        pr_points.to_csv(
            output_dir / "pr_curve_points.csv",
            index=False,
            float_format="%.10f",
        )
    else:
        roc_points = None
        pr_points = None

    if create_plots:
        save_confusion_matrix_plot(
            matrices["raw"],
            label_names,
            output_dir / "confusion_matrix_raw.png",
        )
        save_confusion_matrix_plot(
            matrices["normalized_true"],
            label_names,
            output_dir / "confusion_matrix_normalized_true.png",
            title="Confusion matrix normalized by true class",
            value_format=".2f",
        )
        # Keep this path deliberately short. Deep experiment folders can hit
        # the legacy 260-character Windows path limit used by Pillow.
        class_matrix_dir = output_dir / "cm_class"
        for class_index, (label, label_name) in enumerate(
            zip(labels, label_names)
        ):
            safe_name = "".join(
                character
                if character.isalnum() or character in {"-", "_"}
                else "_"
                for character in str(label_name)
            )
            save_confusion_matrix_plot(
                matrices["one_vs_rest"][str(label)],
                [f"not_{label_name}", str(label_name)],
                class_matrix_dir / f"c{class_index:02d}_{safe_name[:20]}.png",
                title=f"One-vs-rest confusion matrix: {label_name}",
            )
        if roc_points is not None:
            save_curve_plot(
                roc_points,
                output_dir / "roc_curves.png",
                curve_type="roc",
            )
            save_curve_plot(
                pr_points,
                output_dir / "precision_recall_curves.png",
                curve_type="pr",
            )

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrices": matrices,
        "errors": errors,
    }


def save_training_history_plot(
    history,
    output_file,
    x_name="epoch",
    title="Training history",
):
    """Plot iterative training history when a model actually exposes it.

    LIM-NFST is closed-form and has no epoch/loss history, so this helper is
    intended for iterative competitors only.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = pd.DataFrame(history)
    if x_name not in history.columns:
        raise ValueError(f"history must contain the {x_name!r} column.")
    metric_columns = [
        column for column in history.columns if column != x_name
    ]
    if not metric_columns:
        raise ValueError("history has no metric columns to plot.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for metric in metric_columns:
        axis.plot(history[x_name], history[metric], label=metric)
    axis.set_xlabel(x_name)
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_cross_validation_ranking_plots(
    candidate_results,
    metric,
    output_dir,
    top_n=20,
    standard_deviation_column="selection_metric_std",
):
    """Plot the best grid configurations and their CV standard deviation."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    results = pd.DataFrame(candidate_results)
    required_columns = {"dataset", metric}
    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f"candidate_results is missing columns: {sorted(missing)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_columns = [
        "scaler",
        "reference_size",
        "neighbors",
        "rff_components",
        "rff_gamma_multiplier",
    ]

    output_files = []
    for dataset, dataset_results in results.groupby(
        "dataset",
        sort=False,
    ):
        best_rows = dataset_results.sort_values(
            metric,
            ascending=False,
            kind="stable",
        ).head(top_n)

        configuration_labels = []
        for _, row in best_rows.iterrows():
            parts = []
            for column in parameter_columns:
                if column in best_rows.columns and pd.notna(row[column]):
                    parts.append(f"{column}={row[column]}")
            configuration_labels.append(", ".join(parts))

        positions = np.arange(len(best_rows))
        errors = (
            best_rows[standard_deviation_column].to_numpy()
            if standard_deviation_column in best_rows.columns
            else None
        )

        figure_height = max(5.0, 0.45 * len(best_rows) + 2.0)
        figure, axis = plt.subplots(figsize=(11, figure_height))
        axis.errorbar(
            best_rows[metric],
            positions,
            xerr=errors,
            fmt="o",
            capsize=3,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(configuration_labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel(f"Cross-validation {metric}")
        axis.set_title(f"{dataset}: top {len(best_rows)} configurations")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()

        safe_dataset = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in str(dataset)
        )
        output_file = (
            output_dir / f"{safe_dataset}__cv_{metric}_ranking.png"
        )
        figure.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(figure)
        output_files.append(output_file)

    return output_files


def evaluate(
    y_pred,
    y_true,
    y_proba=None,
    y_score=None,
    labels=None,
):
    """Backward-compatible wrapper using the old argument order.

    Unlike the old implementation, values stay on the 0-to-1 scale.
    """
    return calculate_summary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        y_proba=y_proba,
        labels=labels,
    )
