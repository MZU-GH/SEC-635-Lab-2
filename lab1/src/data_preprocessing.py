"""
CICIDS-2018 Data Preprocessing and Ingestion Script
SEC-635 Lab 1: AI-Driven Detection

This module cleans and transforms raw CICIDS-2018 CSV flow logs into
reproducible, chronologically-sorted Parquet datasets ready for downstream ML.

Key operations:
1. Strips repeated header rows embedded inside CSV files.
2. Parses timestamps (%d/%m/%Y %H:%M:%S) and filters out clock anomalies (e.g., 1970 records).
3. Handles NaN and Inf values (records exact counts and applies drop/impute strategy).
4. Chronologically sorts flows by timestamp to enable honest temporal splits without data leakage.
5. Emits optimized Parquet files and detailed JSON/log metrics.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cicids_preprocessor")


def clean_dataset(
    csv_path: Path,
    nan_inf_strategy: str = "drop",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a raw CICIDS-2018 CSV dataset.

    Parameters:
        csv_path: Path to raw CSV file.
        nan_inf_strategy: Strategy for handling NaN/Inf ('drop' or 'impute').

    Returns:
        Tuple of (cleaned_df, metrics_dict).
    """
    logger.info(f"Loading raw CSV: {csv_path} (size: {csv_path.stat().st_size / (1024 * 1024):.2f} MB)")
    df = pd.read_csv(csv_path, low_memory=False)
    raw_row_count = len(df)
    logger.info(f"Loaded {raw_row_count:,} raw rows with {df.shape[1]} columns.")

    # 1. Clean repeated header rows
    header_mask = (df["Timestamp"] == "Timestamp") | (df["Dst Port"] == "Dst Port")
    header_rows_removed = int(header_mask.sum())
    if header_rows_removed > 0:
        logger.info(f"Detected and removing {header_rows_removed} embedded header rows.")
        df = df[~header_mask].copy()
    else:
        logger.info("No embedded header rows detected.")

    # 2. Parse timestamps and remove anomalies (e.g. 1970 clock errors)
    ts_parsed = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    invalid_ts_mask = ts_parsed.isna()
    y1970_mask = ts_parsed.dt.year == 1970
    other_year_mask = (ts_parsed.dt.year != 2018) & (~y1970_mask) & (~invalid_ts_mask)

    y1970_rows_dropped = int(y1970_mask.sum())
    invalid_ts_dropped = int(invalid_ts_mask.sum())
    other_year_dropped = int(other_year_mask.sum())
    total_ts_anomalies = y1970_rows_dropped + invalid_ts_dropped + other_year_dropped

    if total_ts_anomalies > 0:
        logger.info(
            f"Filtering timestamp anomalies: 1970_rows={y1970_rows_dropped}, "
            f"invalid_format={invalid_ts_dropped}, non_2018={other_year_dropped}"
        )
        valid_ts_mask = (ts_parsed.dt.year == 2018) & (~invalid_ts_mask)
        df = df[valid_ts_mask].copy()
        df["Timestamp"] = ts_parsed[valid_ts_mask]
    else:
        df["Timestamp"] = ts_parsed

    # 3. Numeric conversion of feature columns
    feature_cols = [c for c in df.columns if c not in ["Timestamp", "Label"]]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Detect and handle NaN / Inf values
    feat_df = df[feature_cols]
    nan_mask = feat_df.isna()
    inf_mask = np.isinf(feat_df)
    
    col_nan_counts = {col: int(cnt) for col, cnt in nan_mask.sum().items() if cnt > 0}
    col_inf_counts = {col: int(cnt) for col, cnt in inf_mask.sum().items() if cnt > 0}

    row_has_nan = nan_mask.any(axis=1)
    row_has_inf = inf_mask.any(axis=1)
    row_has_nan_or_inf = row_has_nan | row_has_inf

    rows_with_nan = int(row_has_nan.sum())
    rows_with_inf = int(row_has_inf.sum())
    rows_with_nan_or_inf = int(row_has_nan_or_inf.sum())

    logger.info(
        f"NaN/Inf analysis: rows_with_nan={rows_with_nan:,}, rows_with_inf={rows_with_inf:,}, "
        f"total_affected_rows={rows_with_nan_or_inf:,} ({rows_with_nan_or_inf / len(df):.4%})"
    )
    if col_nan_counts:
        logger.info(f"  Columns with NaNs: {col_nan_counts}")
    if col_inf_counts:
        logger.info(f"  Columns with Infs: {col_inf_counts}")

    nan_inf_breakdown_by_label = (
        df.loc[row_has_nan_or_inf, "Label"].value_counts().to_dict()
    )

    if nan_inf_strategy == "drop":
        logger.info(f"Strategy 'drop': Dropping {rows_with_nan_or_inf:,} rows containing NaN or Inf.")
        df = df[~row_has_nan_or_inf].copy()
    elif nan_inf_strategy == "impute":
        logger.info("Strategy 'impute': Imputing Inf with col max/min finite values, NaN with median.")
        for col in feature_cols:
            is_inf_pos = df[col] == np.inf
            is_inf_neg = df[col] == -np.inf
            is_nan = df[col].isna()
            
            finite_mask = ~(is_inf_pos | is_inf_neg | is_nan)
            if is_inf_pos.any() or is_inf_neg.any() or is_nan.any():
                col_max = df.loc[finite_mask, col].max() if finite_mask.any() else 0.0
                col_min = df.loc[finite_mask, col].min() if finite_mask.any() else 0.0
                col_med = df.loc[finite_mask, col].median() if finite_mask.any() else 0.0
                
                df.loc[is_inf_pos, col] = col_max
                df.loc[is_inf_neg, col] = col_min
                df.loc[is_nan, col] = col_med
    else:
        raise ValueError(f"Unknown nan_inf_strategy: {nan_inf_strategy}")

    # 5. Chronological sort by Timestamp
    logger.info("Sorting chronologically by Timestamp (stable merge sort)...")
    df = df.sort_values(by="Timestamp", kind="mergesort").reset_index(drop=True)
    assert df["Timestamp"].is_monotonic_increasing, "Dataset must be monotonically non-decreasing by Timestamp!"

    final_row_count = len(df)
    min_ts = df["Timestamp"].min()
    max_ts = df["Timestamp"].max()
    time_span = max_ts - min_ts
    
    label_counts = df["Label"].value_counts().to_dict()
    label_pcts = (df["Label"].value_counts(normalize=True) * 100).round(4).to_dict()

    metrics = {
        "dataset_filename": csv_path.name,
        "raw_row_count": raw_row_count,
        "header_rows_removed": header_rows_removed,
        "1970_rows_dropped": y1970_rows_dropped,
        "invalid_ts_dropped": invalid_ts_dropped,
        "other_year_dropped": other_year_dropped,
        "nan_inf_strategy": nan_inf_strategy,
        "rows_with_nan": rows_with_nan,
        "rows_with_inf": rows_with_inf,
        "rows_with_nan_or_inf_handled": rows_with_nan_or_inf,
        "nan_inf_breakdown_by_label": nan_inf_breakdown_by_label,
        "columns_with_nan": col_nan_counts,
        "columns_with_inf": col_inf_counts,
        "final_row_count": final_row_count,
        "retention_rate_pct": round(final_row_count / raw_row_count * 100, 4),
        "min_timestamp": str(min_ts),
        "max_timestamp": str(max_ts),
        "time_span_seconds": int(time_span.total_seconds()),
        "time_span_formatted": str(time_span),
        "is_chronologically_sorted": bool(df["Timestamp"].is_monotonic_increasing),
        "class_distribution_counts": label_counts,
        "class_distribution_pct": label_pcts,
        "feature_count": len(feature_cols),
    }

    logger.info(
        f"Cleaned {csv_path.name}: {final_row_count:,} rows retained "
        f"({metrics['retention_rate_pct']}%). Range: {min_ts} -> {max_ts}"
    )
    logger.info(f"Class distribution: {label_counts}")

    return df, metrics


def save_dataset(df: pd.DataFrame, parquet_path: Path) -> int:
    """
    Saves a cleaned DataFrame to Parquet with Snappy compression and verifies integrity.
    """
    logger.info(f"Writing Parquet file: {parquet_path}...")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    file_size_bytes = parquet_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info(f"Parquet saved successfully. Size: {file_size_mb:.2f} MB.")
    
    # Quick sanity read
    verify_df = pd.read_parquet(parquet_path, columns=["Timestamp", "Label"])
    assert len(verify_df) == len(df), f"Verification row count mismatch: {len(verify_df)} vs {len(df)}"
    assert verify_df["Timestamp"].is_monotonic_increasing, "Saved Parquet is not monotonically sorted!"
    logger.info("Verification passed: row count and temporal order validated.")
    return file_size_bytes


def main():
    parser = argparse.ArgumentParser(description="Clean and ingest CICIDS-2018 datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing raw CSV datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory to save cleaned parquet files and summary.",
    )
    parser.add_argument(
        "--nan-inf-strategy",
        type=str,
        choices=["drop", "impute"],
        default="drop",
        help="Handling strategy for NaN/Inf values (default: drop).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    strategy = args.nan_inf_strategy

    datasets = [
        {
            "name": "wednesday",
            "csv": data_dir / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
            "parquet": output_dir / "cleaned_wednesday_14-02-2018.parquet",
        },
        {
            "name": "thursday",
            "csv": data_dir / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "parquet": output_dir / "cleaned_thursday_01-03-2018.parquet",
        },
    ]

    all_metrics = {}

    for d in datasets:
        csv_path = d["csv"]
        parquet_path = d["parquet"]

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing raw CSV file: {csv_path}")

        logger.info(f"\n{'='*70}\nProcessing dataset: {csv_path.name}\n{'='*70}")
        cleaned_df, metrics = clean_dataset(csv_path, nan_inf_strategy=strategy)
        saved_bytes = save_dataset(cleaned_df, parquet_path)
        metrics["parquet_filename"] = parquet_path.name
        metrics["parquet_size_mb"] = round(saved_bytes / (1024 * 1024), 2)
        all_metrics[d["name"]] = metrics

    # Save metrics summary JSON
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nPreprocessing metrics summary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("PHASE 1 PREPROCESSING SUMMARY TABLE")
    print("=" * 90)
    for day_name, m in all_metrics.items():
        print(f"\n--- Dataset: {m['dataset_filename']} ({day_name.upper()}) ---")
        print(f"  Raw Rows:                      {m['raw_row_count']:,}")
        print(f"  Header Rows Removed:           {m['header_rows_removed']:,}")
        print(f"  1970 Corrupted Rows Dropped:   {m['1970_rows_dropped']:,}")
        print(f"  Invalid / Non-2018 Dropped:    {m['invalid_ts_dropped'] + m['other_year_dropped']:,}")
        print(f"  NaN / Inf Rows Handled:        {m['rows_with_nan_or_inf_handled']:,} ({m['nan_inf_strategy']})")
        print(f"  Final Retained Rows:           {m['final_row_count']:,} ({m['retention_rate_pct']}%)")
        print(f"  Chronological Span:            {m['min_timestamp']} to {m['max_timestamp']} ({m['time_span_formatted']})")
        print(f"  Chronologically Sorted:        {m['is_chronologically_sorted']}")
        print(f"  Parquet Output:                {m['parquet_filename']} ({m['parquet_size_mb']} MB)")
        print("  Class Distribution:")
        for label, cnt in m["class_distribution_counts"].items():
            pct = m["class_distribution_pct"][label]
            print(f"    - {label:<20}: {cnt:>10,} ({pct:6.2f}%)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
