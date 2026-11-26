"""Numbers quoted by the Recommendations section (stdlib only).

Reads results/task1_attacker_budget.json and results/task2_summary.json, derives the
detector-API alert threshold by a stated rule, prints every value, and writes
results/task5_recommendation_controls.json.

Rule (a policy choice, derived, never typed): alert threshold = floor(ALERT_FRACTION x the
cheapest measured black-box attack, in server-measured queries per flow), ALERT_FRACTION = 1/2.
Run: .venv/bin/python scripts/recommendation_controls.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALERT_FRACTION = 0.5


def main() -> int:
    bu = json.loads((ROOT / "results" / "task1_attacker_budget.json").read_text())
    sm = json.loads((ROOT / "results" / "task2_summary.json").read_text())
    bb = bu["black_box_methods"]
    fd_min = bb["fd_fgsm"]["server_measured_queries_per_flow_min"]
    rs_min = bb["random_search"]["server_measured_queries_per_flow_min"]
    rs_max = bb["random_search"]["server_measured_queries_per_flow_max"]
    cheapest = min(fd_min, rs_min)
    out = {
        "source_files": ["results/task1_attacker_budget.json", "results/task2_summary.json"],
        "fd_fgsm_queries_per_flow_min": fd_min,
        "random_search_queries_per_flow_min": rs_min,
        "random_search_queries_per_flow_max": rs_max,
        "cheapest_attack_queries_per_flow": cheapest,
        "alert_fraction": ALERT_FRACTION,
        "alert_rule": "alert_threshold = floor(alert_fraction * cheapest_attack_queries_per_flow)",
        "alert_threshold_near_identical_vectors": math.floor(ALERT_FRACTION * cheapest),
        "guard_baseline_unspecified_or_errored_records":
            sm["guard_evidence"]["records_with_unspecified_empty_category_or_error"],
    }
    for k, v in out.items():
        print(f"{k}: {v}")
    (ROOT / "results" / "task5_recommendation_controls.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
