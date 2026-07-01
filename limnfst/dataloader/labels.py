from __future__ import annotations

import pandas as pd

from limnfst.dataloader.common import normalize_label


LABEL_GRANULARITIES = ("paper", "raw")


def validate_label_granularity(value: str) -> str:
    key = normalize_label(value)
    if key not in LABEL_GRANULARITIES:
        raise ValueError(f"Unknown label granularity: {value!r}")
    return key


def nbaiot_label(value: object, granularity: str = "paper") -> str:
    label = normalize_label(value)
    if validate_label_granularity(granularity) == "raw":
        return label
    if label in {"benign", "normal"}:
        return "normal"
    if "." in label:
        return label.rsplit(".", 1)[-1]
    return label


def botiot_label(category: object, subcategory: object, granularity: str = "paper") -> str:
    category_label = normalize_label(category)
    subcategory_label = normalize_label(subcategory)
    if validate_label_granularity(granularity) == "raw":
        return _join_label_parts(category_label, subcategory_label)
    return _join_label_parts(category_label, subcategory_label)


def ciciot_label(value: object, granularity: str = "paper") -> str:
    label = normalize_label(value)
    if validate_label_granularity(granularity) == "raw":
        return label
    if label in {"benign", "normal"}:
        return "normal"
    if label.startswith("ddos-") or label.startswith("dos-"):
        return "ddos_dos"
    if label.startswith("recon-") or label == "vulnerabilityscan":
        return "scan"
    if label in {
        "browserhijacking",
        "commandinjection",
        "dictionarybruteforce",
        "sqlinjection",
        "uploading_attack",
        "xss",
    }:
        return "web"
    if label in {"dns_spoofing", "mitm-arpspoofing"}:
        return "spoofing"
    if label.startswith("mirai-") or label == "backdoor_malware":
        return "botnet"
    return label


def map_ciciot_labels(values: pd.Series, granularity: str = "paper") -> pd.Series:
    return values.map(lambda value: ciciot_label(value, granularity))


def _join_label_parts(*parts: str) -> str:
    labels = [part for part in parts if part not in {"", "unknown"}]
    if not labels:
        return "unknown"
    if "normal" in labels or "benign" in labels:
        return "normal"
    if len(set(labels)) == 1:
        return labels[0]
    return ".".join(labels)
