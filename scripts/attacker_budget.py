"""What a black-box attacker needs: queries per flow and what must be known.

Everything is computed from results/*.json and detector/features.py; writes
results/task1_attacker_budget.json. Run: .venv/bin/python scripts/attacker_budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402

from detector.dataset import LABEL_COL, LEAKAGE_COLUMNS, WEDNESDAY_PARQUET  # noqa: E402
from detector.features import classify  # noqa: E402


def main() -> int:
    ev = json.loads((ROOT / "results" / "task1b_evasion_wednesday.json").read_text())
    names = [c for c in pq.read_schema(WEDNESDAY_PARQUET).names
             if c not in LEAKAGE_COLUMNS and c != LABEL_COL]
    mut = classify(names)
    n = ev["n_flows"]

    per_method = {}
    for method in ("random_search", "fd_fgsm"):
        rows = []
        for cell in ev["curve"]:
            if cell["eps_std"] == 0.0:
                continue
            a = cell["attacks"][method]
            r = {"eps_std": cell["eps_std"],
                 "client_budget_model_calls": a["budget"]["model_calls"],
                 "client_budget_queries_per_flow": a["budget"]["model_calls"] / n,
                 "server_measured_rows_scored": a["server_measured"]["rows_scored"],
                 "server_measured_requests": a["server_measured"]["requests"],
                 "server_measured_queries_per_flow": a["server_measured"]["rows_scored"] / n}
            rows.append(r)
        per_method[method] = {
            "per_eps": rows,
            "server_measured_queries_per_flow_min": min(r["server_measured_queries_per_flow"] for r in rows),
            "server_measured_queries_per_flow_max": max(r["server_measured_queries_per_flow"] for r in rows),
            "client_budget_queries_per_flow_min": min(r["client_budget_queries_per_flow"] for r in rows),
            "client_budget_queries_per_flow_max": max(r["client_budget_queries_per_flow"] for r in rows),
            "note": "server_measured counts every row the API scored, including the "
                    "unperturbed-flow query (random search); the client budget omits it for "
                    "random search and counts one unqueried FGSM step for FD-FGSM",
            "structure": ("<= 200 candidate queries per flow (stops at first success) "
                          "plus one query of the unperturbed flow"
                          if method == "random_search" else
                          "2 queries per mutable feature (central differences) "
                          "= 2 * n_mutable, then one FGSM step"),
            "n_mutable_x2": 2 * int(mut.sum()),
        }
    mem_src = (ROOT / "detector" / "membership.py").read_text().splitlines()
    cite = [f"detector/membership.py:{i+1}: {l.strip()}" for i, l in enumerate(mem_src)
            if "p_true = np.where(y == 1" in l or "def fit_threshold" in l
            or "tau = fit_threshold(s_cal[key], cal_member, above)" in l]
    out = {
        "source_files": ["results/task1b_evasion_wednesday.json", "detector/features.py",
                         "detector/membership.py"],
        "n_flows": n,
        "n_mutable_features": int(mut.sum()),
        "n_immutable_features": int((~mut).sum()),
        "n_features_total": len(names),
        "api_returns": "a score only: P(attack) (/score also returns the 0.5-threshold label "
                       "and the logit margin; /score_batch returns scores and labels); no "
                       "gradients, weights, training-set size or attribution. /stats returns "
                       "request counters only.",
        "black_box_methods": per_method,
        "pgd": "white-box: 40 gradient computations per flow with the model weights "
               "in-process; not an attacker capability under the API-only threat model",
        "attacker_must_know": {
            "feature_schema": "the 77 feature names and order (GET /features)",
            "which_features_are_mutable": "the attacker must control the mutable subset",
            "membership_inference_requires_true_label": (
                "the 1A threshold signals are computed from P(true class), i.e. the attacker "
                "must already know each queried flow's true label, and needs a labelled "
                "calibration pool with known membership to fit the threshold tau"),
            "citations": cite,
        },
    }
    p = ROOT / "results" / "task1_attacker_budget.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
