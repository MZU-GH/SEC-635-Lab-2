# usage: compare_1a.py NEW.json SERVER_ROWS_BASELINE_WED [OLD.json]
# Accuracy / CI / AUC / counts must be exactly equal; tau and mean_gap may differ by float noise.
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
new = json.load(open(sys.argv[1])); base = int(sys.argv[2])
old = json.load(open(sys.argv[3] if len(sys.argv) > 3 else ROOT / "results/task1a_membership.json"))
cells = same = 0; exact_fail = []; maxd = 0.0
SOFT = ("tau", "mean_gap")
def cmp(a, b, path):
    global cells, same, maxd
    if isinstance(a, dict):
        for k in a: cmp(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)): cmp(x, y, path + f"[{i}]")
    else:
        cells += 1
        if a == b: same += 1
        elif path.split("/")[-1] in SOFT and isinstance(a, float):
            maxd = max(maxd, abs(a - b))
        else: exact_fail.append((path, a, b))
for do, dn in zip(old["days"], new["days"]):
    for key in ("day", "split", "generalisation_gap", "duplicate_control"): cmp(do[key], dn[key], do["day"] + "/" + key)
    for ao, an in zip(do["arms"], dn["arms"]): cmp(ao["attacks"], an["attacks"], ao["arm"])
print(f"{cells} cells compared, {same} identical, {cells-same-len(exact_fail)} differ only in tau/mean_gap float noise (max abs diff {maxd:.3e}), {len(exact_fail)} real differences")
for d in exact_fail: print("DIFF", d)
t = new["transport"]; print(t["wednesday_target_arm"], t["thursday_target_arm"], t["client_rows_sent"])
print("server rows delta wednesday", t["server_stats_final"]["wednesday"]["rows_scored"] - base)
