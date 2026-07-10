from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a / b)


def hprs(precision: float, recall: float, specificity: float) -> float:
    if precision <= 0 or recall <= 0 or specificity <= 0:
        return 0.0
    return float(3.0 / ((1.0 / precision) + (1.0 / recall) + (1.0 / specificity)))


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    specificity = _safe_div(tn, tn + fp)

    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "hprs": hprs(float(precision), float(recall), float(specificity)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_frames": int(y_true.shape[0]),
    }


def compute_score_curves(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, list]:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float32)

    try:
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    except ValueError:
        precision = np.asarray([], dtype=np.float32)
        recall = np.asarray([], dtype=np.float32)
        pr_thresholds = np.asarray([], dtype=np.float32)

    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    except ValueError:
        fpr = np.asarray([], dtype=np.float32)
        tpr = np.asarray([], dtype=np.float32)
        roc_thresholds = np.asarray([], dtype=np.float32)

    return {
        "pr_curve": {
            "precision": precision.astype(float).tolist(),
            "recall": recall.astype(float).tolist(),
            "thresholds": pr_thresholds.astype(float).tolist(),
        },
        "roc_curve": {
            "fpr": fpr.astype(float).tolist(),
            "tpr": tpr.astype(float).tolist(),
            "thresholds": roc_thresholds.astype(float).tolist(),
        },
    }


def best_threshold_by_hprs(y_true: np.ndarray, y_score: np.ndarray, steps: int = 300) -> float:
    if y_score.size == 0:
        return 0.0
    lo = float(np.percentile(y_score, 50))
    hi = float(np.percentile(y_score, 99.9))
    if hi <= lo:
        return float(np.median(y_score))

    best_t = lo
    best_m = -1.0
    for t in np.linspace(lo, hi, steps):
        m = compute_binary_metrics(y_true, y_score, float(t))["hprs"]
        if m > best_m:
            best_m = m
            best_t = float(t)
    return best_t
