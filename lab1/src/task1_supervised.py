"""
SEC-635 Lab 1: Task 1 - Supervised Classifier Evaluation & SOC Operational Framing.

This module trains and evaluates three distinct classifier families:
  1. Gradient Boosted Trees: LightGBM (LGBMClassifier)
  2. Linear / Margin-based: Logistic Regression via SGDClassifier (loss='log_loss', penalty='l2')
  3. Neural Architecture: Multi-Layer Perceptron (MLPClassifier)

Datasets evaluated:
  - Wednesday-14-02-2018 (FTP / SSH Brute Force - "The Easy Day")
  - Thursday-01-03-2018 (Infiltration - "The Hard Day")

Evaluations performed:
  - Honest Temporal Split (First 70% chronological flows -> Train, Last 30% -> Test)
  - Random Stratified Split (70/30 split to demonstrate data leakage / over-optimism)
  - Operational SOC Alert Framing (FPR -> Daily False Alerts -> Analyst Hours at 12 min/alert)

Artifacts generated:
  - Confusion matrices: outputs/figures/confusion_matrices_task1.png
  - ROC & PR curves: outputs/figures/roc_pr_curves_task1.png
  - Temporal vs. Random split comparison: outputs/figures/leakage_comparison_task1.png
  - Metrics tables: outputs/tables/task1_metrics_comparison.csv & .json
  - Serialized model artifacts: outputs/models/*.joblib
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)
import lightgbm as lgb

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
    TEMPORAL_TRAIN_FRACTION,
)
from src.metrics_soc import compute_soc_metrics, print_soc_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("task1_supervised")


def load_dataset(parquet_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads preprocessed dataset, constructs binary label, and strips leakage features.

    Returns:
        X (pd.DataFrame): Cleaned feature matrix without leakage features.
        y (pd.Series): Binary labels (0 = Benign, 1 = Attack).
        timestamps (pd.Series): Chronologically ordered Timestamp series.
    """
    logger.info(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} flows with {df.shape[1]} raw columns.")

    assert "Timestamp" in df.columns, "Timestamp column is required for temporal splitting!"
    timestamps = df["Timestamp"].copy()

    # Binary label mapping: 0 for Benign, 1 for Attack
    y = (df["Label"] != "Benign").astype(int)
    attack_ratio = y.mean() * 100
    logger.info(
        f"Label distribution: Benign={(y == 0).sum():,} ({100 - attack_ratio:.2f}%), "
        f"Attack={(y == 1).sum():,} ({attack_ratio:.2f}%)"
    )

    # Defend against leakage: drop static identity / shortcut columns
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns] + ["Label"]
    logger.info(f"Dropping static leakage features and target: {cols_to_drop}")
    X = df.drop(columns=cols_to_drop)
    logger.info(f"Final feature space dimension: {X.shape[1]} features.")

    return X, y, timestamps


def get_models() -> Dict[str, Any]:
    """
    Instantiates three distinct classifier families with fixed random state.
    """
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
        ),
        "SGD_Logistic": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "MLP_NeuralNet": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=25,
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                ),
            ),
        ]),
    }
    return models


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    dataset_name: str,
    split_type: str,
    full_day_benign_volume: float,
) -> Dict[str, Any]:
    """
    Trains a model and records timing, latency, standard metrics, and SOC impact.
    """
    logger.info(f"--- Training {model_name} on {dataset_name} [{split_type} Split] ---")

    # 1. Measure Training Time
    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train_end = time.perf_counter()
    train_time_sec = t_train_end - t_train_start
    logger.info(f"  Training finished in {train_time_sec:.2f} seconds.")

    # 2. Measure Inference Latency (Microseconds per flow)
    # Using full test set for stable latency measurement
    t_inf_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_inf_end = time.perf_counter()
    inf_time_sec = t_inf_end - t_inf_start
    latency_us_per_flow = (inf_time_sec / len(X_test)) * 1_000_000
    logger.info(
        f"  Inference finished in {inf_time_sec:.3f} s "
        f"({latency_us_per_flow:.2f} µs/flow across {len(X_test):,} flows)."
    )

    # 3. Predict Probabilities for the Attack class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Normalize decision function via sigmoid to [0, 1]
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_prob = None

    # 4. Compute SOC Operational & ML Metrics
    soc_results = compute_soc_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        test_duration_hours=None,  # Use full_day_benign_volume for exact FPR * DayVolume formula
        full_day_benign_volume=full_day_benign_volume,
        model_name=model_name,
        dataset_name=f"{dataset_name} ({split_type})",
    )

    # 5. Enrich result dictionary with metadata and runtime statistics
    result = {
        **soc_results,
        "dataset_key": dataset_name,
        "split_type": split_type,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_time_seconds": round(train_time_sec, 3),
        "inference_latency_us": round(latency_us_per_flow, 2),
        "daily_benign_volume": full_day_benign_volume,
    }

    # Store predicted probabilities and predictions for figure generation (removed before JSON serialization)
    result["_y_test"] = y_test.values
    result["_y_pred"] = y_pred
    result["_y_prob"] = y_prob

    print_soc_report(soc_results)
    return result


def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    output_path: Path,
):
    """
    Generates a 2x3 grid of normalized and raw confusion matrices for the 3 models on both days.
    """
    logger.info(f"Generating confusion matrices plot at: {output_path}")
    days = ["Wednesday", "Thursday"]
    model_keys = ["LightGBM", "SGD_Logistic", "MLP_NeuralNet"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Task 1: Supervised Classifier Confusion Matrices (Honest Temporal Split)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for r_idx, day in enumerate(days):
        for c_idx, m_name in enumerate(model_keys):
            ax = axes[r_idx, c_idx]
            match = [
                res for res in results_list
                if res["dataset_key"] == day
                and res["model_name"] == m_name
                and res["split_type"] == "Temporal"
            ]
            if not match:
                continue
            res = match[0]
            cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
            cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

            # Annotations with counts and row percentages
            annot = np.empty_like(cm, dtype=object)
            for i in range(2):
                for j in range(2):
                    annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.2%})"

            sns.heatmap(
                cm_norm,
                annot=annot,
                fmt="",
                cmap="Blues",
                cbar=True,
                ax=ax,
                vmin=0.0,
                vmax=1.0,
                annot_kws={"fontsize": 11, "fontweight": "medium"},
            )

            day_title = "Wednesday (Brute Force)" if day == "Wednesday" else "Thursday (Infiltration)"
            ax.set_title(
                f"{day_title}\n{m_name}\nF1: {res['f1']:.4f} | FPR: {res['fpr']:.4%}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Predicted Class", fontsize=10)
            ax.set_ylabel("True Class", fontsize=10)
            ax.set_xticklabels(["Benign", "Attack"], fontsize=10)
            ax.set_yticklabels(["Benign", "Attack"], fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrices figure to: {output_path}")


def plot_roc_pr_curves(
    results_list: List[Dict[str, Any]],
    output_path: Path,
):
    """
    Generates a 2x2 grid comparing ROC and Precision-Recall curves across models.
    """
    logger.info(f"Generating ROC and PR curves plot at: {output_path}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        "Task 1: ROC and Precision-Recall Curves (Honest Temporal Split)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "LightGBM": "#1f77b4",
        "SGD_Logistic": "#2ca02c",
        "MLP_NeuralNet": "#d62728",
    }
    linestyles = {
        "LightGBM": "-",
        "SGD_Logistic": "--",
        "MLP_NeuralNet": "-.",
    }

    days = ["Wednesday", "Thursday"]
    for idx, day in enumerate(days):
        ax_roc = axes[idx, 0]
        ax_pr = axes[idx, 1]
        day_title = "Wednesday: FTP/SSH Brute Force" if day == "Wednesday" else "Thursday: Infiltration"

        # Baseline random lines
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")

        day_results = [
            res for res in results_list
            if res["dataset_key"] == day and res["split_type"] == "Temporal"
        ]

        if day_results:
            # Baseline PR line is prevalence = positive / total
            y_test_example = day_results[0]["_y_test"]
            prevalence = np.mean(y_test_example)
            ax_pr.plot(
                [0, 1],
                [prevalence, prevalence],
                "k--",
                alpha=0.5,
                label=f"Prevalence Baseline ({prevalence:.2%})",
            )

        for res in day_results:
            m_name = res["model_name"]
            y_test = res["_y_test"]
            y_prob = res["_y_prob"]

            if y_prob is None:
                continue

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr,
                tpr,
                color=colors.get(m_name, "black"),
                linestyle=linestyles.get(m_name, "-"),
                linewidth=2.2,
                label=f"{m_name} (AUC = {roc_auc:.4f})",
            )

            # PR Curve
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(rec, prec)
            ax_pr.plot(
                rec,
                prec,
                color=colors.get(m_name, "black"),
                linestyle=linestyles.get(m_name, "-"),
                linewidth=2.2,
                label=f"{m_name} (PR-AUC = {pr_auc:.4f})",
            )

        # Style ROC Subplot
        ax_roc.set_title(f"{day_title} — ROC Curves", fontsize=12, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        ax_roc.grid(True, linestyle=":", alpha=0.6)
        ax_roc.legend(loc="lower right", fontsize=10)

        # Style PR Subplot
        ax_pr.set_title(f"{day_title} — Precision-Recall Curves", fontsize=12, fontweight="bold")
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.02])
        ax_pr.grid(True, linestyle=":", alpha=0.6)
        ax_pr.legend(loc="lower left", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC & PR curves figure to: {output_path}")


def plot_split_comparison(
    results_list: List[Dict[str, Any]],
    output_path: Path,
):
    """
    Generates a comparison chart proving data leakage / over-optimism of random split vs temporal split.
    """
    logger.info(f"Generating split comparison plot at: {output_path}")

    # Prepare DataFrame for seaborn barplot
    data_records = []
    for res in results_list:
        data_records.append({
            "Dataset": res["dataset_key"],
            "Model": res["model_name"],
            "Split": res["split_type"],
            "F1_Score": res["f1"],
            "Recall": res["recall"],
            "Precision": res["precision"],
            "FPR_Pct": res["fpr"] * 100.0,
            "Daily_False_Alerts": res["daily_false_alerts"],
        })
    df_plot = pd.DataFrame(data_records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Empirical Proof of Data Leakage: Honest Temporal Split vs. Random Stratified Split",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    palette = {"Temporal": "#1f77b4", "Random": "#ff7f0e"}

    # Left: F1-Score Comparison
    ax1 = axes[0]
    sns.barplot(
        data=df_plot,
        x="Model",
        y="F1_Score",
        hue="Split",
        palette=palette,
        ax=ax1,
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_title("F1-Score: Attack Class Generalization", fontsize=12, fontweight="bold")
    ax1.set_ylabel("F1-Score", fontsize=11)
    ax1.set_xlabel("Model Family", fontsize=11)
    ax1.set_ylim([0, 1.08])
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right: False Positive Rate (FPR %)
    ax2 = axes[1]
    sns.barplot(
        data=df_plot,
        x="Model",
        y="FPR_Pct",
        hue="Split",
        palette=palette,
        ax=ax2,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_title("False Positive Rate (FPR % on Benign Flows)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("FPR (%)", fontsize=11)
    ax2.set_xlabel("Model Family", fontsize=11)
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(
                f"{h:.2f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved leakage comparison figure to: {output_path}")


def main():
    logger.info("=================================================================")
    logger.info("Starting Phase 2 / Task 1: Supervised Classifier Evaluation")
    logger.info("=================================================================")

    datasets_to_process = [
        {
            "name": "Wednesday",
            "path": WEDNESDAY_PARQUET,
            "description": "FTP & SSH Brute Force (The Easy Day)",
        },
        {
            "name": "Thursday",
            "path": THURSDAY_PARQUET,
            "description": "Infiltration (The Hard Day)",
        },
    ]

    all_results: List[Dict[str, Any]] = []

    for d_meta in datasets_to_process:
        d_name = d_meta["name"]
        d_path = d_meta["path"]
        d_desc = d_meta["description"]

        logger.info(f"\n{'='*70}\nProcessing Dataset: {d_name} — {d_desc}\n{'='*70}")

        X, y, timestamps = load_dataset(d_path)
        total_rows = len(X)
        total_benign_flows = (y == 0).sum()

        # Compute full-day benign flow volume extrapolated to 24-hour SOC baseline:
        # 12-hour capture volume * (24.0 / 12.0)
        extrapolation_factor = DAY_HOURS / CAPTURE_DURATION_HOURS
        full_day_benign_volume = float(total_benign_flows * extrapolation_factor)
        logger.info(
            f"Capture benign flows: {total_benign_flows:,} in {CAPTURE_DURATION_HOURS} hours. "
            f"Extrapolated 24-hr benign volume: {full_day_benign_volume:,.1f} flows/day."
        )

        # -------------------------------------------------------------
        # 1. HONEST TEMPORAL SPLIT (First 70% -> Train, Last 30% -> Test)
        # -------------------------------------------------------------
        split_idx = int(total_rows * TEMPORAL_TRAIN_FRACTION)
        X_train_temp = X.iloc[:split_idx]
        y_train_temp = y.iloc[:split_idx]
        X_test_temp = X.iloc[split_idx:]
        y_test_temp = y.iloc[split_idx:]

        min_ts = timestamps.min()
        max_ts = timestamps.max()
        train_end_ts = timestamps.iloc[split_idx - 1]
        test_start_ts = timestamps.iloc[split_idx]

        logger.info(f"--- Temporal Split Boundary Details for {d_name} ---")
        logger.info(f"  Total Rows: {total_rows:,} | Train Split (70%): {len(X_train_temp):,} | Test Split (30%): {len(X_test_temp):,}")
        logger.info(f"  Capture Window: {min_ts} -> {max_ts}")
        logger.info(f"  Train Window:   {min_ts} -> {train_end_ts}")
        logger.info(f"  Test Window:    {test_start_ts} -> {max_ts}")
        logger.info(f"  Train Attack %: {y_train_temp.mean()*100:.2f}% | Test Attack %: {y_test_temp.mean()*100:.2f}%")

        # -------------------------------------------------------------
        # 2. RANDOM STRATIFIED SPLIT (70/30 split to prove data leakage)
        # -------------------------------------------------------------
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            X,
            y,
            test_size=(1.0 - TEMPORAL_TRAIN_FRACTION),
            stratify=y,
            random_state=RANDOM_STATE,
        )
        logger.info(f"--- Stratified Random Split Details for {d_name} ---")
        logger.info(f"  Train: {len(X_train_rand):,} rows (Attack {y_train_rand.mean()*100:.2f}%)")
        logger.info(f"  Test:  {len(X_test_rand):,} rows (Attack {y_test_rand.mean()*100:.2f}%)")

        # -------------------------------------------------------------
        # 3. TRAIN & EVALUATE ON HONEST TEMPORAL SPLIT
        # -------------------------------------------------------------
        temp_models = get_models()
        for m_name, model in temp_models.items():
            res = evaluate_model(
                model=model,
                X_train=X_train_temp,
                y_train=y_train_temp,
                X_test=X_test_temp,
                y_test=y_test_temp,
                model_name=m_name,
                dataset_name=d_name,
                split_type="Temporal",
                full_day_benign_volume=full_day_benign_volume,
            )
            # Add split metadata
            res["split_boundary_timestamp"] = str(test_start_ts)
            res["train_end_timestamp"] = str(train_end_ts)
            res["capture_min_timestamp"] = str(min_ts)
            res["capture_max_timestamp"] = str(max_ts)
            all_results.append(res)

            # Serialize trained model artifact
            model_artifact_path = MODELS_DIR / f"task1_{d_name.lower()}_{m_name.lower()}_temporal.joblib"
            logger.info(f"Saving serialized model artifact to: {model_artifact_path}")
            joblib.dump(model, model_artifact_path)

        # -------------------------------------------------------------
        # 4. TRAIN & EVALUATE ON RANDOM SPLIT (LEAKAGE DEFENSE CHECK)
        # -------------------------------------------------------------
        rand_models = get_models()
        for m_name, model in rand_models.items():
            res = evaluate_model(
                model=model,
                X_train=X_train_rand,
                y_train=y_train_rand,
                X_test=X_test_rand,
                y_test=y_test_rand,
                model_name=m_name,
                dataset_name=d_name,
                split_type="Random",
                full_day_benign_volume=full_day_benign_volume,
            )
            all_results.append(res)

    # -----------------------------------------------------------------
    # 5. VISUALIZATIONS
    # -----------------------------------------------------------------
    cm_fig_path = FIGURES_DIR / "confusion_matrices_task1.png"
    plot_confusion_matrices(all_results, cm_fig_path)

    roc_fig_path = FIGURES_DIR / "roc_pr_curves_task1.png"
    plot_roc_pr_curves(all_results, roc_fig_path)

    split_fig_path = FIGURES_DIR / "leakage_comparison_task1.png"
    plot_split_comparison(all_results, split_fig_path)

    # -----------------------------------------------------------------
    # 6. SERIALIZE METRICS TABLE (CSV & JSON)
    # -----------------------------------------------------------------
    # Strip numpy arrays used for plotting
    serializable_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if not k.startswith("_")}
        serializable_results.append(clean_r)

    df_metrics = pd.DataFrame(serializable_results)

    # Order columns logically
    ordered_cols = [
        "dataset_key",
        "split_type",
        "model_name",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "pr_auc",
        "daily_false_alerts",
        "analyst_hours_per_day",
        "analyst_shifts_needed",
        "train_time_seconds",
        "inference_latency_us",
        "verdict_sentence",
    ]
    remaining_cols = [c for c in df_metrics.columns if c not in ordered_cols]
    df_metrics = df_metrics[ordered_cols + remaining_cols]

    csv_path = TABLES_DIR / "task1_metrics_comparison.csv"
    json_path = TABLES_DIR / "task1_metrics_comparison.json"

    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics CSV table to: {csv_path}")

    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved metrics JSON to: {json_path}")

    # Print final summary table
    print("\n" + "=" * 100)
    print("FINAL SUMMARY: TASK 1 SUPERVISED EVALUATION & SOC METRICS COMPARISON")
    print("=" * 100)
    summary_cols = [
        "dataset_key",
        "split_type",
        "model_name",
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "daily_false_alerts",
        "analyst_hours_per_day",
        "train_time_seconds",
        "inference_latency_us",
    ]
    print(df_metrics[summary_cols].to_string(index=False))
    print("=" * 100)
    logger.info("Task 1 execution completed successfully!")


if __name__ == "__main__":
    main()
