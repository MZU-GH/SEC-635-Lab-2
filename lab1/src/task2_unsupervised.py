"""
SEC-635 Lab 1: Task 2 - Unsupervised Attack Detection & Stability Sweeps.

This module evaluates two unsupervised anomaly detectors with fundamentally different
geometric / cluster assumptions on network flow traffic without labels:
  1. Tree Space Partitioning: Isolation Forest (orthogonal recursive cuts).
     - Geometric assumption: anomalies are "few and different", susceptible to early
       isolation along axis-aligned hyperplanes near tree roots.
  2. Subspace / Manifold Projection: PCA Reconstruction Error.
     - Geometric assumption: benign network traffic lies along a low-dimensional linear
       hyperplane capturing primary cross-feature correlations; anomalous flows deviate
       into orthogonal residual space, producing large reconstruction errors ||x - x_hat||^2.

Datasets evaluated:
  - Wednesday-14-02-2018 (FTP / SSH Brute Force)
  - Thursday-01-03-2018 (Infiltration)

Evaluations performed:
  - Strictly unsupervised model fitting (all labels stripped during training).
  - Parameter Sensitivity Sweeps across:
      * Isolation Forest: contamination in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.28, 0.36]
      * Isolation Forest: n_estimators in [50, 100, 150, 200]
      * PCA Reconstruction: n_components in [2, 5, 10, 15, 20, 30, 40]
      * PCA Reconstruction: percentile threshold in [70, 75, 80, 85, 90, 95, 98, 99]
  - Evaluation of Precision, Recall, F1, FPR, and PR-AUC against held-out ground truth.
  - Qualitative Breakdown: Found vs Missed attacks (FTP vs SSH, Infiltration by hour/characteristics).
  - Detection Overlap & Exclusive Detections Analysis.
  - Generation of publication-quality figures and metric tables.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# Internal project imports
from src.config import (
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    WEDNESDAY_PARQUET,
    THURSDAY_PARQUET,
    RANDOM_STATE,
    CAPTURE_DURATION_HOURS,
    DAY_HOURS,
    ANALYST_TRIAGE_MINUTES,
    LEAKAGE_COLUMNS,
)
from src.metrics_soc import compute_soc_metrics

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("task2_unsupervised")

# Set visualization styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def load_unsupervised_data(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads dataset, strictly separates labels and leakage features, and scales features.

    Returns:
        X_scaled (np.ndarray): Standardized feature matrix (labels completely stripped).
        y_binary (np.ndarray): Ground truth binary labels (0 = Benign, 1 = Attack) held aside.
        y_raw (np.ndarray): Original string labels held aside for qualitative analysis.
        meta_df (pd.DataFrame): Metadata dataframe containing Timestamps and raw labels.
    """
    logger.info(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} flows with {df.shape[1]} columns.")

    assert "Label" in df.columns, "Label column required for held-out evaluation!"
    y_raw = df["Label"].to_numpy()
    y_binary = (y_raw != "Benign").astype(int)

    meta_cols = ["Label"]
    if "Timestamp" in df.columns:
        meta_cols.append("Timestamp")
    meta_df = df[meta_cols].copy()

    # Defend against leakage: drop static identifiers, timestamps, target port shortcuts, and target
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns] + ["Label"]
    logger.info(f"Dropping static leakage features and target: {cols_to_drop}")
    X_df = df.drop(columns=cols_to_drop)
    feature_names = X_df.columns.tolist()
    logger.info(f"Feature matrix dimension: {X_df.shape[0]:,} rows x {X_df.shape[1]} features.")

    logger.info("Applying StandardScaler to unsupervised feature space...")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float32))
    logger.info(f"Feature scaling completed in {time.time() - t0:.2f}s.")

    return X_scaled, y_binary, y_raw, meta_df


def run_isolation_forest_sweeps(
    X_scaled: np.ndarray,
    y_binary: np.ndarray,
    dataset_name: str,
    n_estimators_list: List[int] = [50, 100, 150, 200],
    contaminations: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.28, 0.36],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Executes parameter sensitivity sweep for Isolation Forest across n_estimators and contamination.
    """
    logger.info(f"--- Running Isolation Forest Sweeps on {dataset_name} ---")
    sweep_records = []
    scores_by_n_est = {}

    for n_est in n_estimators_list:
        logger.info(f"Fitting IsolationForest(n_estimators={n_est}, random_state={RANDOM_STATE})...")
        t0 = time.time()
        iso = IsolationForest(
            n_estimators=n_est,
            max_samples=256,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        iso.fit(X_scaled)
        fit_time = time.time() - t0

        # Continuous anomaly score: -score_samples(x) so higher score = more anomalous
        scores = -iso.score_samples(X_scaled)
        scores_by_n_est[n_est] = scores
        pr_auc = float(average_precision_score(y_binary, scores))
        roc_auc = float(roc_auc_score(y_binary, scores))
        logger.info(f"IF (n_est={n_est}) fit in {fit_time:.2f}s | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

        # Sweep contamination thresholds
        for cont in contaminations:
            th = np.percentile(scores, 100.0 * (1.0 - cont))
            y_pred = (scores >= th).astype(int)

            cm = confusion_matrix(y_binary, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            prec = precision_score(y_binary, y_pred, zero_division=0)
            rec = recall_score(y_binary, y_pred, zero_division=0)
            f1 = f1_score(y_binary, y_pred, zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Daily false alerts extrapolated based on 12-hour capture rate
            daily_fa = (fp / CAPTURE_DURATION_HOURS) * DAY_HOURS

            sweep_records.append({
                "dataset": dataset_name,
                "detector": "IsolationForest",
                "n_estimators": n_est,
                "contamination": cont,
                "n_components": np.nan,
                "percentile_threshold": (1.0 - cont) * 100.0,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "daily_false_alerts": float(daily_fa),
            })

    sweep_df = pd.DataFrame(sweep_records)
    logger.info(f"Isolation Forest sweeps complete. Generated {len(sweep_df)} parameter evaluations.")
    return sweep_df, scores_by_n_est


def run_pca_sweeps(
    X_scaled: np.ndarray,
    y_binary: np.ndarray,
    dataset_name: str,
    n_components_list: List[int] = [2, 5, 10, 15, 20, 30, 40],
    percentiles: List[float] = [70, 75, 80, 85, 90, 95, 98, 99],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], PCA]:
    """
    Executes parameter sensitivity sweep for PCA Reconstruction Error across n_components and percentile threshold.
    """
    logger.info(f"--- Running PCA Reconstruction Sweeps on {dataset_name} ---")
    sweep_records = []
    errors_by_n_comp = {}

    max_comp = max(n_components_list)
    logger.info(f"Fitting master PCA(n_components={max_comp}, random_state={RANDOM_STATE})...")
    t0 = time.time()
    master_pca = PCA(n_components=max_comp, random_state=RANDOM_STATE)
    X_master_transformed = master_pca.fit_transform(X_scaled)
    logger.info(f"Master PCA fit in {time.time() - t0:.2f}s.")

    total_dim = X_scaled.shape[1]

    for k in n_components_list:
        # Exact reconstruction from first k principal components
        X_k = X_master_transformed[:, :k]
        comp_k = master_pca.components_[:k, :]
        X_rec = X_k @ comp_k + master_pca.mean_

        # Reconstruction error: mean squared error across all original feature dimensions
        rec_err = np.mean((X_scaled - X_rec) ** 2, axis=1)
        errors_by_n_comp[k] = rec_err

        pr_auc = float(average_precision_score(y_binary, rec_err))
        roc_auc = float(roc_auc_score(y_binary, rec_err))
        logger.info(f"PCA (n_comp={k:2d}) | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

        for pct in percentiles:
            th = np.percentile(rec_err, pct)
            y_pred = (rec_err >= th).astype(int)

            cm = confusion_matrix(y_binary, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            prec = precision_score(y_binary, y_pred, zero_division=0)
            rec = recall_score(y_binary, y_pred, zero_division=0)
            f1 = f1_score(y_binary, y_pred, zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            daily_fa = (fp / CAPTURE_DURATION_HOURS) * DAY_HOURS

            sweep_records.append({
                "dataset": dataset_name,
                "detector": "PCAReconstruction",
                "n_estimators": np.nan,
                "contamination": (100.0 - pct) / 100.0,
                "n_components": k,
                "percentile_threshold": pct,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "daily_false_alerts": float(daily_fa),
            })

    sweep_df = pd.DataFrame(sweep_records)
    logger.info(f"PCA sweeps complete. Generated {len(sweep_df)} parameter evaluations.")
    return sweep_df, errors_by_n_comp, master_pca


def analyze_qualitative_breakdown(
    y_raw: np.ndarray,
    meta_df: pd.DataFrame,
    scores_if: np.ndarray,
    scores_pca: np.ndarray,
    dataset_name: str,
    ref_cont_if: float,
    ref_pct_pca: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs qualitative breakdown of found vs missed attacks, exclusive detections, and flow categories.
    """
    logger.info(f"--- Performing Qualitative Breakdown on {dataset_name} ---")
    y_binary = (y_raw != "Benign").astype(int)

    th_if = np.percentile(scores_if, 100.0 * (1.0 - ref_cont_if))
    pred_if = scores_if >= th_if

    th_pca = np.percentile(scores_pca, ref_pct_pca)
    pred_pca = scores_pca >= th_pca

    both = pred_if & pred_pca
    if_only = pred_if & (~pred_pca)
    pca_only = (~pred_if) & pred_pca
    neither = (~pred_if) & (~pred_pca)

    breakdown_records = []

    if dataset_name == "Wednesday":
        # Analyze Benign, FTP-BruteForce, SSH-Bruteforce
        unique_labels = ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]
        for lbl in unique_labels:
            mask = (y_raw == lbl)
            n_total = int(mask.sum())
            n_if = int((pred_if & mask).sum())
            n_pca = int((pred_pca & mask).sum())
            n_both = int((both & mask).sum())
            n_if_only = int((if_only & mask).sum())
            n_pca_only = int((pca_only & mask).sum())
            n_missed = int((neither & mask).sum())

            breakdown_records.append({
                "dataset": dataset_name,
                "category": lbl,
                "total_flows": n_total,
                "if_detected": n_if,
                "if_recall": n_if / n_total if n_total > 0 else 0.0,
                "pca_detected": n_pca,
                "pca_recall": n_pca / n_total if n_total > 0 else 0.0,
                "both_detected": n_both,
                "both_pct": n_both / n_total if n_total > 0 else 0.0,
                "if_exclusive": n_if_only,
                "if_exclusive_pct": n_if_only / n_total if n_total > 0 else 0.0,
                "pca_exclusive": n_pca_only,
                "pca_exclusive_pct": n_pca_only / n_total if n_total > 0 else 0.0,
                "missed_by_both": n_missed,
                "missed_by_both_pct": n_missed / n_total if n_total > 0 else 0.0,
            })

    elif dataset_name == "Thursday":
        # Analyze Benign, Overall Infiltration, and Infiltration broken down by Hour
        timestamps = pd.to_datetime(meta_df["Timestamp"])
        hours = timestamps.dt.hour.to_numpy()

        # Benign and Total Infiltration
        for lbl in ["Benign", "Infilteration"]:
            mask = (y_raw == lbl)
            n_total = int(mask.sum())
            n_if = int((pred_if & mask).sum())
            n_pca = int((pred_pca & mask).sum())
            n_both = int((both & mask).sum())
            n_if_only = int((if_only & mask).sum())
            n_pca_only = int((pca_only & mask).sum())
            n_missed = int((neither & mask).sum())

            breakdown_records.append({
                "dataset": dataset_name,
                "category": f"All_{lbl}",
                "total_flows": n_total,
                "if_detected": n_if,
                "if_recall": n_if / n_total if n_total > 0 else 0.0,
                "pca_detected": n_pca,
                "pca_recall": n_pca / n_total if n_total > 0 else 0.0,
                "both_detected": n_both,
                "both_pct": n_both / n_total if n_total > 0 else 0.0,
                "if_exclusive": n_if_only,
                "if_exclusive_pct": n_if_only / n_total if n_total > 0 else 0.0,
                "pca_exclusive": n_pca_only,
                "pca_exclusive_pct": n_pca_only / n_total if n_total > 0 else 0.0,
                "missed_by_both": n_missed,
                "missed_by_both_pct": n_missed / n_total if n_total > 0 else 0.0,
            })

        # Infiltration hourly breakdown (Hours 2, 3, 9, 10)
        for h in [2, 3, 9, 10]:
            mask = (y_raw == "Infilteration") & (hours == h)
            n_total = int(mask.sum())
            n_if = int((pred_if & mask).sum())
            n_pca = int((pred_pca & mask).sum())
            n_both = int((both & mask).sum())
            n_if_only = int((if_only & mask).sum())
            n_pca_only = int((pca_only & mask).sum())
            n_missed = int((neither & mask).sum())

            breakdown_records.append({
                "dataset": dataset_name,
                "category": f"Infiltration_Hour_{h:02d}",
                "total_flows": n_total,
                "if_detected": n_if,
                "if_recall": n_if / n_total if n_total > 0 else 0.0,
                "pca_detected": n_pca,
                "pca_recall": n_pca / n_total if n_total > 0 else 0.0,
                "both_detected": n_both,
                "both_pct": n_both / n_total if n_total > 0 else 0.0,
                "if_exclusive": n_if_only,
                "if_exclusive_pct": n_if_only / n_total if n_total > 0 else 0.0,
                "pca_exclusive": n_pca_only,
                "pca_exclusive_pct": n_pca_only / n_total if n_total > 0 else 0.0,
                "missed_by_both": n_missed,
                "missed_by_both_pct": n_missed / n_total if n_total > 0 else 0.0,
            })

    breakdown_df = pd.DataFrame(breakdown_records)

    overlap_summary = {
        "pred_if": pred_if,
        "pred_pca": pred_pca,
        "both": both,
        "if_only": if_only,
        "pca_only": pca_only,
        "neither": neither,
    }
    return breakdown_df, overlap_summary


def plot_parameter_sweeps(
    all_sweeps_df: pd.DataFrame,
    output_path: Path,
):
    """
    Generates high-quality 2x2 parameter sweep stability curves for both detectors on both days.
    """
    logger.info(f"Generating parameter sweep visualization -> {output_path}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # 1. Wednesday Isolation Forest: F1 vs Contamination across n_estimators
    ax1 = axes[0, 0]
    sub1 = all_sweeps_df[(all_sweeps_df["dataset"] == "Wednesday") & (all_sweeps_df["detector"] == "IsolationForest")]
    palette_if = sns.color_palette("mako", n_colors=sub1["n_estimators"].nunique())
    for (n_est, grp), col in zip(sub1.groupby("n_estimators"), palette_if):
        ax1.plot(grp["contamination"], grp["f1"], marker="o", linewidth=2.0, label=f"n_est = {n_est}", color=col)
    ax1.axvline(0.3646, color="red", linestyle="--", alpha=0.8, label="True Attack Prior (36.5%)")
    ax1.set_title("A. Wednesday: Isolation Forest Stability\n(Tree Partitioning vs Contamination)", fontweight="bold")
    ax1.set_xlabel("Contamination Parameter")
    ax1.set_ylabel("F1 Score (Minority Attack Class)")
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_ylim(-0.02, 0.60)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Wednesday PCA: F1 vs Percentile Threshold across n_components
    ax2 = axes[0, 1]
    sub2 = all_sweeps_df[(all_sweeps_df["dataset"] == "Wednesday") & (all_sweeps_df["detector"] == "PCAReconstruction")]
    palette_pca = sns.color_palette("viridis", n_colors=sub2["n_components"].nunique())
    for (n_comp, grp), col in zip(sub2.groupby("n_components"), palette_pca):
        ax2.plot(grp["percentile_threshold"], grp["f1"], marker="s", linewidth=2.0, label=f"k = {int(n_comp)}", color=col)
    ax2.axvline(63.54, color="red", linestyle="--", alpha=0.8, label="True Anomaly Pct (63.5%)")
    ax2.set_title("B. Wednesday: PCA Reconstruction Stability\n(Linear Subspace vs Percentile Cutoff)", fontweight="bold")
    ax2.set_xlabel("Percentile Threshold Cutoff (%)")
    ax2.set_ylabel("F1 Score (Minority Attack Class)")
    ax2.legend(loc="upper right", frameon=True, ncol=2)
    ax2.set_ylim(-0.02, 0.60)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Thursday Isolation Forest: F1 vs Contamination across n_estimators
    ax3 = axes[1, 0]
    sub3 = all_sweeps_df[(all_sweeps_df["dataset"] == "Thursday") & (all_sweeps_df["detector"] == "IsolationForest")]
    for (n_est, grp), col in zip(sub3.groupby("n_estimators"), palette_if):
        ax3.plot(grp["contamination"], grp["f1"], marker="o", linewidth=2.0, label=f"n_est = {n_est}", color=col)
    ax3.axvline(0.2816, color="red", linestyle="--", alpha=0.8, label="True Attack Prior (28.2%)")
    ax3.set_title("C. Thursday: Isolation Forest Stability\n(Infiltration vs Contamination)", fontweight="bold")
    ax3.set_xlabel("Contamination Parameter")
    ax3.set_ylabel("F1 Score (Minority Attack Class)")
    ax3.legend(loc="upper left", frameon=True)
    ax3.set_ylim(-0.02, 0.35)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Thursday PCA: F1 vs Percentile Threshold across n_components
    ax4 = axes[1, 1]
    sub4 = all_sweeps_df[(all_sweeps_df["dataset"] == "Thursday") & (all_sweeps_df["detector"] == "PCAReconstruction")]
    for (n_comp, grp), col in zip(sub4.groupby("n_components"), palette_pca):
        ax4.plot(grp["percentile_threshold"], grp["f1"], marker="s", linewidth=2.0, label=f"k = {int(n_comp)}", color=col)
    ax4.axvline(71.84, color="red", linestyle="--", alpha=0.8, label="True Anomaly Pct (71.8%)")
    ax4.set_title("D. Thursday: PCA Reconstruction Stability\n(Infiltration vs Percentile Cutoff)", fontweight="bold")
    ax4.set_xlabel("Percentile Threshold Cutoff (%)")
    ax4.set_ylabel("F1 Score (Minority Attack Class)")
    ax4.legend(loc="upper right", frameon=True, ncol=2)
    ax4.set_ylim(-0.02, 0.35)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {output_path}")


def plot_cluster_projections(
    wed_X_scaled: np.ndarray,
    wed_y_raw: np.ndarray,
    wed_pred_if: np.ndarray,
    wed_pred_pca: np.ndarray,
    thu_X_scaled: np.ndarray,
    thu_y_raw: np.ndarray,
    thu_pred_if: np.ndarray,
    thu_pred_pca: np.ndarray,
    output_path: Path,
    n_sample: int = 15000,
):
    """
    Generates 2D PCA projection scatter plots (2x3 grid) visualizing:
      - Row 1: Wednesday (Ground Truth Classes, IF Flagged Outliers, PCA Flagged Outliers)
      - Row 2: Thursday (Ground Truth Classes, IF Flagged Outliers, PCA Flagged Outliers)
    """
    logger.info(f"Generating cluster projection scatter plots -> {output_path}")

    # Subsample Wednesday
    np.random.seed(RANDOM_STATE)
    idx_wed = np.random.choice(len(wed_X_scaled), size=min(n_sample, len(wed_X_scaled)), replace=False)
    pca_wed_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(wed_X_scaled[idx_wed])

    # Subsample Thursday
    idx_thu = np.random.choice(len(thu_X_scaled), size=min(n_sample, len(thu_X_scaled)), replace=False)
    pca_thu_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(thu_X_scaled[idx_thu])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # 1. Wednesday: Ground Truth
    ax = axes[0, 0]
    w_labels = wed_y_raw[idx_wed]
    palette_w = {"Benign": "#4575b4", "FTP-BruteForce": "#d73027", "SSH-Bruteforce": "#fc8d59"}
    for lbl in ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]:
        m = (w_labels == lbl)
        ax.scatter(pca_wed_2d[m, 0], pca_wed_2d[m, 1], s=8, alpha=0.45, label=lbl, c=palette_w[lbl])
    ax.set_title("Wednesday: Ground Truth Labels\n(PC1 vs PC2 Subspace)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    # 2. Wednesday: Isolation Forest Detections
    ax = axes[0, 1]
    w_if = wed_pred_if[idx_wed]
    ax.scatter(pca_wed_2d[~w_if, 0], pca_wed_2d[~w_if, 1], s=8, alpha=0.3, label="Normal (Inlier)", c="#91bfdb")
    ax.scatter(pca_wed_2d[w_if, 0], pca_wed_2d[w_if, 1], s=12, alpha=0.7, label="IF Flagged Outlier", c="#d73027")
    ax.set_title("Wednesday: Isolation Forest Outliers\n(Tree Partition Cuts)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    # 3. Wednesday: PCA Reconstruction Detections
    ax = axes[0, 2]
    w_pca = wed_pred_pca[idx_wed]
    ax.scatter(pca_wed_2d[~w_pca, 0], pca_wed_2d[~w_pca, 1], s=8, alpha=0.3, label="Normal (Inlier)", c="#91bfdb")
    ax.scatter(pca_wed_2d[w_pca, 0], pca_wed_2d[w_pca, 1], s=12, alpha=0.7, label="PCA Flagged Outlier", c="#762a83")
    ax.set_title("Wednesday: PCA Reconstruction Outliers\n(Linear Residual Error)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    # 4. Thursday: Ground Truth
    ax = axes[1, 0]
    t_labels = thu_y_raw[idx_thu]
    palette_t = {"Benign": "#4575b4", "Infilteration": "#d73027"}
    for lbl in ["Benign", "Infilteration"]:
        m = (t_labels == lbl)
        ax.scatter(pca_thu_2d[m, 0], pca_thu_2d[m, 1], s=8, alpha=0.45, label=lbl, c=palette_t[lbl])
    ax.set_title("Thursday: Ground Truth Labels\n(PC1 vs PC2 Subspace)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    # 5. Thursday: Isolation Forest Detections
    ax = axes[1, 1]
    t_if = thu_pred_if[idx_thu]
    ax.scatter(pca_thu_2d[~t_if, 0], pca_thu_2d[~t_if, 1], s=8, alpha=0.3, label="Normal (Inlier)", c="#91bfdb")
    ax.scatter(pca_thu_2d[t_if, 0], pca_thu_2d[t_if, 1], s=12, alpha=0.7, label="IF Flagged Outlier", c="#d73027")
    ax.set_title("Thursday: Isolation Forest Outliers\n(Tree Partition Cuts)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    # 6. Thursday: PCA Reconstruction Detections
    ax = axes[1, 2]
    t_pca = thu_pred_pca[idx_thu]
    ax.scatter(pca_thu_2d[~t_pca, 0], pca_thu_2d[~t_pca, 1], s=8, alpha=0.3, label="Normal (Inlier)", c="#91bfdb")
    ax.scatter(pca_thu_2d[t_pca, 0], pca_thu_2d[t_pca, 1], s=12, alpha=0.7, label="PCA Flagged Outlier", c="#762a83")
    ax.set_title("Thursday: PCA Reconstruction Outliers\n(Linear Residual Error)", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=2.5)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 35)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {output_path}")


def plot_detection_overlap(
    wed_breakdown_df: pd.DataFrame,
    thu_breakdown_df: pd.DataFrame,
    output_path: Path,
):
    """
    Visualizes detection overlap and exclusive detections (Both, IF Only, PCA Only, Missed by Both).
    """
    logger.info(f"Generating detection overlap visualization -> {output_path}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # 1. Wednesday Attack Detection Breakdown
    ax1 = axes[0]
    w_sub = wed_breakdown_df[wed_breakdown_df["category"].isin(["FTP-BruteForce", "SSH-Bruteforce"])].copy()

    categories_w = w_sub["category"].tolist()
    x = np.arange(len(categories_w))
    width = 0.2

    ax1.bar(x - 1.5 * width, w_sub["both_pct"] * 100, width, label="Caught by Both", color="#2ca02c")
    ax1.bar(x - 0.5 * width, w_sub["if_exclusive_pct"] * 100, width, label="Isolation Forest Only", color="#1f77b4")
    ax1.bar(x + 0.5 * width, w_sub["pca_exclusive_pct"] * 100, width, label="PCA Reconstruction Only", color="#9467bd")
    ax1.bar(x + 1.5 * width, w_sub["missed_by_both_pct"] * 100, width, label="Missed by Both (FN)", color="#d62728")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories_w, fontweight="bold", fontsize=11)
    ax1.set_ylabel("Detection Proportion (% of Category)")
    ax1.set_title("A. Wednesday: Attack Detection Overlap & Misses\n(FTP-BruteForce vs SSH-Bruteforce)", fontweight="bold")
    ax1.legend(loc="upper center", frameon=True)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Add text labels on bars
    for idx, row in w_sub.reset_index().iterrows():
        ax1.text(idx - 1.5 * width, row["both_pct"] * 100 + 1.5, f"{row['both_pct']*100:.1f}%", ha="center", fontsize=8)
        ax1.text(idx - 0.5 * width, row["if_exclusive_pct"] * 100 + 1.5, f"{row['if_exclusive_pct']*100:.1f}%", ha="center", fontsize=8)
        ax1.text(idx + 0.5 * width, row["pca_exclusive_pct"] * 100 + 1.5, f"{row['pca_exclusive_pct']*100:.1f}%", ha="center", fontsize=8)
        ax1.text(idx + 1.5 * width, row["missed_by_both_pct"] * 100 + 1.5, f"{row['missed_by_both_pct']*100:.1f}%", ha="center", fontsize=8)

    # 2. Thursday Infiltration Hourly Breakdown
    ax2 = axes[1]
    t_sub = thu_breakdown_df[thu_breakdown_df["category"].str.startswith("Infiltration_Hour")].copy()
    categories_t = [c.replace("Infiltration_", "") for c in t_sub["category"].tolist()]
    x2 = np.arange(len(categories_t))

    ax2.bar(x2 - 1.5 * width, t_sub["both_pct"] * 100, width, label="Caught by Both", color="#2ca02c")
    ax2.bar(x2 - 0.5 * width, t_sub["if_exclusive_pct"] * 100, width, label="Isolation Forest Only", color="#1f77b4")
    ax2.bar(x2 + 0.5 * width, t_sub["pca_exclusive_pct"] * 100, width, label="PCA Reconstruction Only", color="#9467bd")
    ax2.bar(x2 + 1.5 * width, t_sub["missed_by_both_pct"] * 100, width, label="Missed by Both (FN)", color="#d62728")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories_t, fontweight="bold", fontsize=11)
    ax2.set_ylabel("Detection Proportion (% of Category)")
    ax2.set_title("B. Thursday: Infiltration Detection Overlap by Time Window\n(Hours 02, 03, 09, 10)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle=":", alpha=0.6)

    for idx, row in t_sub.reset_index().iterrows():
        ax2.text(idx - 1.5 * width, row["both_pct"] * 100 + 1.5, f"{row['both_pct']*100:.1f}%", ha="center", fontsize=8)
        ax2.text(idx - 0.5 * width, row["if_exclusive_pct"] * 100 + 1.5, f"{row['if_exclusive_pct']*100:.1f}%", ha="center", fontsize=8)
        ax2.text(idx + 0.5 * width, row["pca_exclusive_pct"] * 100 + 1.5, f"{row['pca_exclusive_pct']*100:.1f}%", ha="center", fontsize=8)
        ax2.text(idx + 1.5 * width, row["missed_by_both_pct"] * 100 + 1.5, f"{row['missed_by_both_pct']*100:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {output_path}")


def main():
    """
    Main orchestration routine for Task 2 (Unsupervised Attack Detection & Stability Sweeps).
    """
    logger.info("=================================================================")
    logger.info("Starting SEC-635 Lab 1 Task 2: Unsupervised Attack Detection")
    logger.info("=================================================================")

    start_time = time.time()

    # Define Parameter Sweep Grids
    n_estimators_if = [50, 100, 150, 200]
    contaminations_if = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.28, 0.36]
    n_components_pca = [2, 5, 10, 15, 20, 30, 40]
    percentiles_pca = [70, 75, 80, 85, 90, 95, 98, 99]

    # -------------------------------------------------------------
    # 1. Process Wednesday (FTP / SSH Brute Force)
    # -------------------------------------------------------------
    wed_X, wed_y, wed_y_raw, wed_meta = load_unsupervised_data(WEDNESDAY_PARQUET)

    wed_if_sweeps, wed_if_scores = run_isolation_forest_sweeps(
        wed_X, wed_y, "Wednesday", n_estimators_if, contaminations_if
    )
    wed_pca_sweeps, wed_pca_scores, wed_pca_model = run_pca_sweeps(
        wed_X, wed_y, "Wednesday", n_components_pca, percentiles_pca
    )

    # Reference evaluation points for Wednesday:
    # Contamination = 0.28 (comparable to Thursday) and 0.36 (close to true attack prior)
    ref_cont_if_w = 0.28
    ref_pct_pca_w = 72.0  # 100 - 28 = 72%
    wed_breakdown, wed_overlap = analyze_qualitative_breakdown(
        wed_y_raw, wed_meta,
        wed_if_scores[100], wed_pca_scores[10],
        "Wednesday", ref_cont_if_w, ref_pct_pca_w
    )

    # -------------------------------------------------------------
    # 2. Process Thursday (Infiltration)
    # -------------------------------------------------------------
    thu_X, thu_y, thu_y_raw, thu_meta = load_unsupervised_data(THURSDAY_PARQUET)

    thu_if_sweeps, thu_if_scores = run_isolation_forest_sweeps(
        thu_X, thu_y, "Thursday", n_estimators_if, contaminations_if
    )
    thu_pca_sweeps, thu_pca_scores, thu_pca_model = run_pca_sweeps(
        thu_X, thu_y, "Thursday", n_components_pca, percentiles_pca
    )

    # Reference evaluation points for Thursday:
    ref_cont_if_t = 0.28
    ref_pct_pca_t = 72.0
    thu_breakdown, thu_overlap = analyze_qualitative_breakdown(
        thu_y_raw, thu_meta,
        thu_if_scores[100], thu_pca_scores[10],
        "Thursday", ref_cont_if_t, ref_pct_pca_t
    )

    # -------------------------------------------------------------
    # 3. Combine Tables and Save Artifacts
    # -------------------------------------------------------------
    all_sweeps_df = pd.concat([wed_if_sweeps, wed_pca_sweeps, thu_if_sweeps, thu_pca_sweeps], ignore_index=True)
    sweeps_csv_path = TABLES_DIR / "task2_parameter_sweeps.csv"
    all_sweeps_df.to_csv(sweeps_csv_path, index=False)
    logger.info(f"Saved complete parameter sweeps table ({len(all_sweeps_df)} rows) -> {sweeps_csv_path}")

    all_breakdown_df = pd.concat([wed_breakdown, thu_breakdown], ignore_index=True)
    breakdown_csv_path = TABLES_DIR / "task2_attack_breakdown.csv"
    all_breakdown_df.to_csv(breakdown_csv_path, index=False)
    logger.info(f"Saved attack breakdown table ({len(all_breakdown_df)} rows) -> {breakdown_csv_path}")

    # Build High-Level Metrics Comparison Table for Key Operating Points
    metrics_comp_rows = []
    operating_points = [
        # Wednesday
        ("Wednesday", "IsolationForest", 100, 0.10, np.nan, 90.0, "IF @ cont=0.10 (Standard 10% Cutoff)"),
        ("Wednesday", "IsolationForest", 100, 0.28, np.nan, 72.0, "IF @ cont=0.28 (Operational Baseline)"),
        ("Wednesday", "IsolationForest", 100, 0.36, np.nan, 64.0, "IF @ cont=0.36 (Aligned with True Prior 36.5%)"),
        ("Wednesday", "PCAReconstruction", np.nan, 0.10, 10, 90.0, "PCA (k=10) @ 90th Pct (10% Cutoff)"),
        ("Wednesday", "PCAReconstruction", np.nan, 0.20, 10, 80.0, "PCA (k=10) @ 80th Pct (20% Cutoff)"),
        ("Wednesday", "PCAReconstruction", np.nan, 0.30, 10, 70.0, "PCA (k=10) @ 70th Pct (30% Cutoff)"),
        # Thursday
        ("Thursday", "IsolationForest", 100, 0.01, np.nan, 99.0, "IF @ cont=0.01 (Low False Alarm Setting)"),
        ("Thursday", "IsolationForest", 100, 0.10, np.nan, 90.0, "IF @ cont=0.10 (Standard 10% Cutoff)"),
        ("Thursday", "IsolationForest", 100, 0.28, np.nan, 72.0, "IF @ cont=0.28 (Aligned with True Prior 28.2%)"),
        ("Thursday", "PCAReconstruction", np.nan, 0.01, 10, 99.0, "PCA (k=10) @ 99th Pct (Low False Alarm)"),
        ("Thursday", "PCAReconstruction", np.nan, 0.10, 10, 90.0, "PCA (k=10) @ 90th Pct (10% Cutoff)"),
        ("Thursday", "PCAReconstruction", np.nan, 0.28, 10, 72.0, "PCA (k=10) @ 72nd Pct (Aligned with True Prior 28.2%)"),
    ]

    for d_name, det, n_est, cont, n_comp, pct, desc in operating_points:
        if det == "IsolationForest":
            match = all_sweeps_df[
                (all_sweeps_df["dataset"] == d_name)
                & (all_sweeps_df["detector"] == det)
                & (all_sweeps_df["n_estimators"] == n_est)
                & (np.isclose(all_sweeps_df["contamination"], cont))
            ]
        else:
            match = all_sweeps_df[
                (all_sweeps_df["dataset"] == d_name)
                & (all_sweeps_df["detector"] == det)
                & (all_sweeps_df["n_components"] == n_comp)
                & (np.isclose(all_sweeps_df["percentile_threshold"], pct))
            ]

        if not match.empty:
            r = match.iloc[0]
            daily_fa = r["daily_false_alerts"]
            hrs = (daily_fa * ANALYST_TRIAGE_MINUTES) / 60.0
            verdict = "Deployable (<50 alerts/day)" if daily_fa < 50 else f"Infeasible alert fatigue ({daily_fa:,.0f} alerts/day)"
            metrics_comp_rows.append({
                "dataset": d_name,
                "detector": det,
                "config_description": desc,
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "fpr": r["fpr"],
                "pr_auc": r["pr_auc"],
                "daily_false_alerts": daily_fa,
                "analyst_hours_per_day": hrs,
                "operational_verdict": verdict,
            })

    metrics_comp_df = pd.DataFrame(metrics_comp_rows)
    comp_csv_path = TABLES_DIR / "task2_metrics_comparison.csv"
    metrics_comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved metrics comparison table ({len(metrics_comp_df)} rows) -> {comp_csv_path}")

    # -------------------------------------------------------------
    # 4. Generate Visualizations
    # -------------------------------------------------------------
    sweeps_fig_path = FIGURES_DIR / "task2_parameter_sweeps.png"
    plot_parameter_sweeps(all_sweeps_df, sweeps_fig_path)

    cluster_fig_path = FIGURES_DIR / "task2_cluster_projections.png"
    plot_cluster_projections(
        wed_X, wed_y_raw,
        wed_overlap["pred_if"], wed_overlap["pred_pca"],
        thu_X, thu_y_raw,
        thu_overlap["pred_if"], thu_overlap["pred_pca"],
        cluster_fig_path,
    )

    overlap_fig_path = FIGURES_DIR / "task2_detection_overlap.png"
    plot_detection_overlap(wed_breakdown, thu_breakdown, overlap_fig_path)

    total_elapsed = time.time() - start_time
    logger.info(f"Task 2 Execution completed successfully in {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} minutes).")


if __name__ == "__main__":
    main()
