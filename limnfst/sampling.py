from __future__ import annotations

import numpy as np


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def stratified_min_per_class_indices(
    y,
    n_samples,
    min_per_class=1,
    random_state=None,
    shuffle=True,
    replace=False,
):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D label array")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if min_per_class < 0:
        raise ValueError("min_per_class must be non-negative")

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_samples < n_classes * min_per_class:
        raise ValueError("n_samples is too small for min_per_class")
    if not replace and n_samples > len(y):
        raise ValueError("n_samples cannot exceed len(y) when replace=False")
    if not replace and np.any(counts < min_per_class):
        raise ValueError("some classes have fewer rows than min_per_class")

    allocation = np.full(n_classes, min_per_class, dtype=int)
    target = n_samples * counts / len(y)

    while allocation.sum() < n_samples:
        capacity_left = np.full(n_classes, np.inf)
        if not replace:
            capacity_left = counts - allocation

        candidates = np.where(capacity_left > 0)[0]
        if len(candidates) == 0:
            raise ValueError("not enough rows to sample without replacement")

        gap = target - allocation
        best_gap = np.max(gap[candidates])

        if best_gap > 0:
            best_candidates = candidates[gap[candidates] == best_gap]
        else:
            best_candidates = candidates[capacity_left[candidates] == np.max(capacity_left[candidates])]

        chosen_class = best_candidates[0]
        allocation[chosen_class] += 1

    rng = np.random.default_rng(random_state)
    selected = []
    for cls, class_n in zip(classes, allocation):
        class_indices = np.flatnonzero(y == cls)
        selected.append(rng.choice(class_indices, size=class_n, replace=replace))

    indices = np.concatenate(selected)
    if shuffle:
        rng.shuffle(indices)
    return indices


def stratified_min_per_class_sample(
    X,
    y,
    n_samples,
    min_per_class=1,
    random_state=None,
    shuffle=True,
    replace=False,
    return_indices=False,
):
    indices = stratified_min_per_class_indices(
        y=y,
        n_samples=n_samples,
        min_per_class=min_per_class,
        random_state=random_state,
        shuffle=shuffle,
        replace=replace,
    )

    X_sample = _take_rows(X, indices)
    y_sample = _take_rows(y, indices)

    if return_indices:
        return X_sample, y_sample, indices
    return X_sample, y_sample
