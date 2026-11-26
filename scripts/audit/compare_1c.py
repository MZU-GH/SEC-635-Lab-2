# usage: compare_1c.py NEW_EVAL.json [STORED_task1c_defense.json]
# Cell-by-cell comparison of curve_after, and of the "before"/undefended curves held in
# task1c_defense.json / task1c_ablation.json against the installed 1B JSON.
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
new = json.load(open(sys.argv[1]))
old = json.load(open(sys.argv[2] if len(sys.argv) > 2 else ROOT / "results/task1c_defense.json"))
cells = same = 0; diffs = []
for co, cn in zip(old["curve_after"], new["curve_after"]):
    for k in co["attacks"]:
        for f in ("detection_rate", "evasion_rate"):
            cells += 1
            if co["attacks"][k][f] == cn["attacks"][k][f]: same += 1
            else: diffs.append((co["eps_std"], k, f, co["attacks"][k][f], cn["attacks"][k][f]))
print(f"curve_after: {cells} cells compared, {same} identical, {len(diffs)} differ")
for d in diffs: print("DIFF", d)
print("n_eval_flows", old["n_eval_flows"], new["n_eval_flows"])
