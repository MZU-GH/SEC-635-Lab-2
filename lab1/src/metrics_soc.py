"""
SOC Metrics and Evaluation Module for SEC-635 Lab 1.
Implements standard ML metrics, False Positive Rate (FPR) extrapolation to 24-hour SOC volume,
and analyst triage operational burden calculations.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from src.config import CAPTURE_DURATION_HOURS, DAY_HOURS, ANALYST_TRIAGE_MINUTES

def compute_soc_metrics(y_true, y_pred, y_prob=None,
                        test_duration_hours=None,
                        full_day_benign_volume=None,
                        model_name="Model",
                        dataset_name="Dataset"):
    """
    Computes standard ML metrics and translates False Positives into SOC operational burden.

    Args:
        y_true (np.ndarray or pd.Series): Ground truth binary labels (0 = Benign, 1 = Attack).
        y_pred (np.ndarray or pd.Series): Predicted binary labels (0 = Benign, 1 = Attack).
        y_prob (np.ndarray, optional): Predicted probabilities for the Attack class.
        test_duration_hours (float, optional): Duration in hours spanned by the test split.
        full_day_benign_volume (int or float, optional): Expected 24-hour total benign flow volume.
        model_name (str): Name of the evaluated model.
        dataset_name (str): Name of the dataset/day.

    Returns:
        dict: Comprehensive dictionary of ML and SOC operational metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification Metrics on the minority Attack class
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # False Positive Rate on benign traffic: FPR = FP / (FP + TN)
    benign_total = tn + fp
    fpr = fp / benign_total if benign_total > 0 else 0.0

    # Operational SOC Alert Volume Extrapolation:
    # Option 1: Based on test split capture duration (if test spans T_test hours)
    if test_duration_hours and test_duration_hours > 0:
        alerts_per_hour = fp / test_duration_hours
        daily_false_alerts = alerts_per_hour * DAY_HOURS
    # Option 2: Based on full-day benign traffic volume
    elif full_day_benign_volume and full_day_benign_volume > 0:
        daily_false_alerts = fpr * full_day_benign_volume
    else:
        # Default extrapolation: 12-hour capture rate scaled to 24 hours
        # In a 12-hour capture, test is ~30% duration (~3.6 hours)
        est_test_hours = CAPTURE_DURATION_HOURS * (len(y_true) / (len(y_true) * 1.0))
        alerts_per_hour = fp / max(est_test_hours, 1.0)
        daily_false_alerts = alerts_per_hour * DAY_HOURS

    # Operational analyst impact
    analyst_minutes_per_day = daily_false_alerts * ANALYST_TRIAGE_MINUTES
    analyst_hours_per_day = analyst_minutes_per_day / 60.0
    analyst_shifts_needed = analyst_hours_per_day / 8.0  # standard 8-hour shift

    # ROC / PR AUC if probabilities are provided
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass

    # Mandatory hand-in sentence formulation
    acceptable = daily_false_alerts < 50.0  # A realistic SOC threshold (~1 analyst workload)
    if acceptable:
        verdict = f"this model raises {daily_false_alerts:.1f} false alerts per day, and here is why that is acceptable: it requires ~{analyst_hours_per_day:.1f} analyst hours/day, within the capacity of a Tier-1 SOC queue."
    else:
        verdict = f"this model raises {daily_false_alerts:.1f} false alerts per day, and here is why that is not acceptable: it would inundate analysts with ~{analyst_hours_per_day:.1f} hours/day of false positive triage, causing severe alert fatigue."

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "daily_false_alerts": float(daily_false_alerts),
        "analyst_hours_per_day": float(analyst_hours_per_day),
        "analyst_shifts_needed": float(analyst_shifts_needed),
        "verdict_sentence": verdict
    }

def print_soc_report(metrics_dict):
    """Prints a structured summary of SOC evaluation metrics."""
    m = metrics_dict
    print("=" * 65)
    print(f"SOC EVALUATION SUMMARY: {m['model_name']} on {m['dataset_name']}")
    print("=" * 65)
    print(f"Confusion Matrix: [TP: {m['tp']:,} | FP: {m['fp']:,} | FN: {m['fn']:,} | TN: {m['tn']:,}]")
    print(f"Precision:        {m['precision']:.4f}")
    print(f"Recall:           {m['recall']:.4f}")
    print(f"F1-Score:         {m['f1']:.4f}")
    print(f"FPR:              {m['fpr']:.6f} ({m['fpr']*100:.4f}%)")
    if m['roc_auc'] is not None:
        print(f"ROC-AUC:          {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f}")
    print("-" * 65)
    print(f"Daily False Alerts:       {m['daily_false_alerts']:,.1f} alerts/day")
    print(f"Analyst Workload:         {m['analyst_hours_per_day']:.2f} hours/day ({m['analyst_shifts_needed']:.2f} 8-hr shifts)")
    print("-" * 65)
    print("Hand-in Statement:")
    print(f'"{m["verdict_sentence"]}"')
    print("=" * 65)
