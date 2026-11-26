"""Audit the Lab 1 split that Lab 2 attacks, against the clock-repaired split of the submitted Lab 1.

The CICIDS-2018 CSVs write the time on a 12-hour clock without AM/PM: the afternoon
(13:00-17:59) appears as 01:00-05:59. The Lab 1 code copied into lab1/ (the model Lab 2
attacks) sorts by that raw timestamp and cuts 70/30 by row order, so its "later" test rows are
not the latest flows. The submitted Lab 1 (github.com/MZU-GH/SEC-635-Lab-1) repairs the clock
first (+12 h for raw hours 01-07), drops constant columns and retrains.

This script measures both facts from the data and records the submitted Lab 1's own numbers
(read from its results files when that repository is present, LAB1_REPO, default ../../lab1;
otherwise the stored values are checked for internal consistency only).

    python3 scripts/split_audit.py            # write results/split_audit.json
    python3 scripts/split_audit.py --verify   # recompute and compare with the stored file
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.dataset import THURSDAY_PARQUET, WEDNESDAY_PARQUET, build_split  # noqa: E402

OUT = Path(os.environ.get("SPLIT_AUDIT_JSON", ROOT / "results" / "split_audit.json"))
LAB1_REPO = Path(os.environ.get("LAB1_REPO", ROOT.parent.parent / "lab1"))
LAB1_COMMIT = "72e026b"


def repair(ts: pd.Series) -> pd.Series:
    """Submitted Lab 1's repair_clock: raw hours 01-07 are afternoon, add 12 h."""
    moved = (ts.dt.hour >= 1) & (ts.dt.hour <= 7)
    return ts + pd.to_timedelta(np.where(moved, 12, 0), unit="h")


def day(parquet: Path, name: str) -> dict:
    sp = build_split(parquet, name)
    raw = pd.to_datetime(sp.timestamps).reset_index(drop=True)
    true = repair(raw)
    n_tr = len(sp.X_train)
    tr_true, te_true = true.iloc[:n_tr], true.iloc[n_tr:]
    labels = pd.read_parquet(parquet, columns=["Label"])["Label"].reset_index(drop=True)
    const = [f for f, s in zip(sp.feature_names, sp.X_train.std(axis=0)) if s == 0]
    return {
        "rows_train": int(n_tr),
        "rows_test": int(len(sp.X_test)),
        "n_features": len(sp.feature_names),
        "rows_sorted_by_raw_timestamp": bool(raw.is_monotonic_increasing),
        "test_true_clock": [str(te_true.min()), str(te_true.max())],
        "train_true_clock": [str(tr_true.min()), str(tr_true.max())],
        "train_rows_after_test_window": int((tr_true > te_true.max()).sum()),
        "train_after_window_true_clock": [str(tr_true[tr_true > te_true.max()].min()),
                                          str(tr_true[tr_true > te_true.max()].max())],
        "train_labels": {k: int(v) for k, v in labels.iloc[:n_tr].value_counts().items()},
        "test_labels": {k: int(v) for k, v in labels.iloc[n_tr:].value_counts().items()},
        "constant_in_train": const,
    }


def lab1_submitted() -> dict:
    files = {r: LAB1_REPO / r for r in ("results/task1_results.json", "results/prep_wed.json")}
    if not all(p.exists() for p in files.values()):
        return {}
    t = json.loads(files["results/task1_results.json"].read_text())
    p = json.loads(files["results/prep_wed.json"].read_text())
    w = t["splits"]["wed70"]
    mlp = w["models"]["MLP"]
    return {
        "source": f"github.com/MZU-GH/SEC-635-Lab-1 commit {LAB1_COMMIT}",
        "source_sha256": {r: hashlib.sha256(f.read_bytes()).hexdigest() for r, f in files.items()},
        "wed_n_features": p["n_features"],
        "wed_constant_columns_dropped": p["constant_columns"],
        "wed_flows_moved_by_clock_fix": p["flows_moved_by_clock_fix"],
        "wed70_cut_time": w["cut_time"],
        "wed70_train_labels": w["train_labels"],
        "wed70_test_labels": w["test_labels"],
        "wed70_mlp_hidden": w["chosen"]["MLP"]["hidden"],
        "wed70_mlp_tp": mlp["tp"],
        "wed70_mlp_recall": mlp["recall"],
        "wed70_mlp_f1": mlp["f1"],
        "unrepaired_clock_demo_test_labels": t["unrepaired_clock"]["wed"]["test_labels"],
        "unrepaired_clock_demo_train_labels": t["unrepaired_clock"]["wed"]["train_labels"],
    }


def build() -> dict:
    wed = day(WEDNESDAY_PARQUET, "Wednesday")
    out = {
        "what": "Lab 2's attacked split versus the submitted Lab 1's clock-repaired split",
        "wednesday": wed,
        "thursday": day(THURSDAY_PARQUET, "Thursday"),
    }
    sub = lab1_submitted()
    if sub:
        out["lab1_submitted"] = sub
        out["checks"] = {
            "our_split_equals_lab1_unrepaired_demo": (
                wed["test_labels"] == sub["unrepaired_clock_demo_test_labels"]
                and wed["train_labels"] == sub["unrepaired_clock_demo_train_labels"]),
            "our_constant_set_equals_lab1_dropped": (
                sorted(wed["constant_in_train"]) == sorted(sub["wed_constant_columns_dropped"])),
        }
    return out


def main() -> None:
    new = build()
    if "--verify" in sys.argv:
        old = json.loads(OUT.read_text())
        if "lab1_submitted" not in new:  # Lab 1 repo absent: keep its stored record
            new["lab1_submitted"], new["checks"] = old["lab1_submitted"], old["checks"]
            print("split audit: Lab 1 repository not present, its stored values kept")
        if new != old:
            raise SystemExit("split audit: recomputed values differ from results/split_audit.json")
        if not all(old["checks"].values()):
            raise SystemExit(f"split audit: a check is false: {old['checks']}")
        print("split audit: PASS")
        return
    OUT.write_text(json.dumps(new, indent=2) + "\n")
    print(json.dumps({k: v for k, v in new.items() if k != "thursday"}, indent=1))


if __name__ == "__main__":
    main()
