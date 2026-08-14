"""Dataset loading for the self-contained LIM-NFST experiments.

The balanced CSV files live in ``limnfst-classifier/data``.  The loader keeps
the label selection, category remapping and leakage-column removal previously
used by the CPAI experiment code, so examples no longer need CPAI on sys.path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CLASSIFIER_ROOT / "data"

DATASETS = [
    "BoT_IoT",
    "ToN_IoT",
    "N_BaIoT",
    "UNSW_NB15",
    "CIC_IoT2023",
    "IoTID20",
    "Edge_IIoTset",
    "5G_NIDD",
]

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

DATASET_FILES = {
    ("BoT_IoT", 1000): "BoT_IoT_1000.csv",
    ("CIC_IoT2023", 1000): "CIC_IoT2023_1000.csv",
    ("ToN_IoT", 1000): "ToN_IoT_1000.csv",
    ("ToN_IoT", 2000): "ToN_IoT_2000.csv",
    ("UNSW_NB15", 1000): "UNSW_NB15_1000.csv",
    ("IoTID20", 1000): "iotid20_1000.csv",
    ("IoTID20", 2000): "iotid20_2000.csv",
    ("N_BaIoT", 1000): "N_BaIoT_1000.csv",
    # This local file has the same content/hash as the Edge file used by the
    # completed CPAI-backed grid search.  The two Edge filenames were swapped
    # earlier, so selecting it explicitly preserves experiment consistency.
    ("Edge_IIoTset", 1000): (
        "edge_iiotset_1000_no_conflicting_labels.csv"
    ),
    ("5G_NIDD", 1000): "5G_NIDD_1000.csv",
}

LABEL_COLUMNS = {
    "BoT_IoT": "subcategory",
    "CIC_IoT2023": "Label",
    "ToN_IoT": "type",
    "UNSW_NB15": "attack_cat",
    "IoTID20": "Target",
    "N_BaIoT": "Names Atk",
    "Edge_IIoTset": "Attack_type",
    "5G_NIDD": "Attack_Type",
}

CATEGORY_MAPS = {
    "BoT_IoT": {
        "0Normal": "0Normal",
        "Data_Exfiltration": "theft",
        "HTTP": "HTTP",
        "Keylogging": "theft",
        "OS_Fingerprint": "scan",
        "Service_Scan": "scan",
        "TCP": "TCP",
        "UDP": "UDP",
    },
    "ToN_IoT": {
        "0Normal": "0Normal",
        "backdoor": "Malware",
        "ransomware": "Malware",
        "scanning": "Scan",
        "password": "BruteForce",
        "ddos": "DDoS",
        "xss": "WebAttack",
        "injection": "WebAttack",
        "dos": "DoS",
        "mitm": "MITM",
    },
    "UNSW_NB15": {
        "0Normal": "0Normal",
        "Exploits": "Exploits",
        "Shellcode": "Shellcode",
        "Backdoor": "Backdoor",
        "Worms": "Worms",
        "DoS": "Fuzzers",
        "Reconnaissance": "Reconnaissance",
        "Analysis": "Analysis",
    },
    "CIC_IoT2023": {
        "0Normal": "0Normal",
        "DDoS-UDP_Flood": "DDoS/DoS",
        "DDoS-ICMP_Flood": "DDoS/DoS",
        "DDoS-TCP_Flood": "DDoS/DoS",
        "MITM-ArpSpoofing": "Spoofing",
        "DoS-TCP_Flood": "DDoS/DoS",
        "DoS-UDP_Flood": "DDoS/DoS",
        "VulnerabilityScan": "Scan",
        "Backdoor_Malware": "Web",
        "Mirai-udpplain": "Mirai",
    },
}

DROP_COLUMNS = {
    "attack",
    "category",
    "subcategory",
    "type",
    "Label",
    "label",
    "Target",
    "saddr",
    "daddr",
    "sport",
    "dport",
    "src_ip",
    "dst_ip",
    "srcip",
    "dstip",
    "Binary_dtloader",
    "Category_dtloader",
    "Category",
    "Binary",
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Timestamp",
    "attack_cat",
}

BENIGN_LABELS = {
    "BENIGN",
    "Normal",
    "normal",
    "Benign",
    "BenignTraffic",
    "Normal_Normal",
}


def clean_dataset(dataframe, label_column="Label"):
    """Remove leakage/id columns and leave one numeric feature matrix."""
    dataframe = dataframe.copy()
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])

    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in dataframe.columns and column != label_column
    ]
    if columns_to_drop:
        dataframe = dataframe.drop(columns=columns_to_drop)

    for column in dataframe.select_dtypes(include="object").columns:
        if column != label_column:
            dataframe[column] = dataframe[column].astype("category").cat.codes

    all_nan_columns = [
        column
        for column in dataframe.columns
        if column != label_column and dataframe[column].isna().all()
    ]
    if all_nan_columns:
        dataframe = dataframe.drop(columns=all_nan_columns)

    feature_columns = [
        column for column in dataframe.columns if column != label_column
    ]
    return dataframe[[*feature_columns, label_column]]


def load_dataset(name, limit=None):
    """Load one balanced dataset; the final column is always ``Label``."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Valid values: {DATASETS}.")

    selected_limit = limit or DEFAULT_LIMITS[name]
    key = (name, selected_limit)
    if key not in DATASET_FILES:
        available = sorted(
            candidate_limit
            for dataset_name, candidate_limit in DATASET_FILES
            if dataset_name == name
        )
        raise FileNotFoundError(
            f"No file for {name} at limit={selected_limit}. "
            f"Available limits: {available}."
        )

    file_path = DATA_DIR / DATASET_FILES[key]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    dataframe = pd.read_csv(file_path, low_memory=False)
    raw_label_column = LABEL_COLUMNS[name]
    if raw_label_column not in dataframe.columns:
        raise ValueError(
            f"{name}: expected label column {raw_label_column!r}, "
            f"got {list(dataframe.columns)[:10]}."
        )

    if raw_label_column != "Label" and "Label" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Label"])
    dataframe = dataframe.rename(columns={raw_label_column: "Label"})
    dataframe["Label"] = dataframe["Label"].apply(
        lambda value: "0Normal" if value in BENIGN_LABELS else value
    )

    category_map = CATEGORY_MAPS.get(name)
    if category_map is not None:
        dataframe = dataframe[
            dataframe["Label"].isin(category_map)
        ].copy()
        dataframe["Label"] = dataframe["Label"].map(category_map)

    return clean_dataset(dataframe), selected_limit


def split_features_labels(dataframe):
    """Return numeric features and the final label column."""
    values = dataframe.to_numpy()
    return values[:, :-1].astype(np.float64), values[:, -1]
