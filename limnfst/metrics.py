from __future__ import annotations
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef,
)

def evaluate(
    y_pred: np.ndarray,
    y_true: np.ndarray, 
    y_proba: np.ndarray | None = None
) -> dict:
    
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    
    results = {
        "MCC": mcc, "ACC": accuracy, "PRC": precision, 
        "RC_macro": recall, "F1_macro": f1,
        "Confusion_matrix": cfm,
    }
    
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            results['AUC_ROC_OVR'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            ) * 100
        except Exception:
            pass
        
    return results
        