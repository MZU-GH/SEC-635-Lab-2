# usage: install_1b.py NEW_HTTP.json OLD_STORED.json OUT.json
# Merge a reproduced 1B curve into the stored file: keep every old top-level key, replace
# the curve, add transport and a list of the old values of every cell that changed.
import json, sys
new = json.load(open(sys.argv[1])); old = json.load(open(sys.argv[2]))
out = dict(old)
sup = []
for co, cn in zip(old["curve"], new["curve"]):
    for k, ao in co["attacks"].items():
        for f in ("detection_rate", "evasion_rate", "budget", "median_l2_raw_of_successful_change",
                  "median_features_changed", "example_raw_shifts"):
            if ao[f] != cn["attacks"][k][f]:
                sup.append({"eps_std": co["eps_std"], "attack": k, "field": f,
                            "old_value": ao[f], "new_value": cn["attacks"][k][f]})
out["curve"] = new["curve"]
out["transport"] = new["transport"]
out["transport"]["server_stats_note"] = ("server_stats_final is cumulative since the API "
    "process started and includes a 1525-row timing run; use the per-cell "
    "server_measured fields")
out["superseded_cells"] = {
    "reason": "stored values not reproducible by current code; current code is deterministic "
              "and HTTP == in-process (scripts/audit/diag_random.py, scripts/audit/diag_scores.py)",
    "n_cells": len(sup), "cells": sup}
json.dump(out, open(sys.argv[3], "w"), indent=2)
print("superseded cells:", len(sup), "| top-level keys:", sorted(out))
