# usage: compare_1b.py NEW.json SERVER_ROWS_BASELINE [OLD.json]
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import json, sys
new = json.load(open(sys.argv[1])); old = json.load(open(sys.argv[3] if len(sys.argv) > 3 else str(ROOT)+"/results/task1b_evasion_wednesday.json"))
BASE = int(sys.argv[2])  # server rows before the full run (timing run)
cells = same = 0; diffs = []
for co, cn in zip(old["curve"], new["curve"]):
    assert co["eps_std"] == cn["eps_std"]
    for k in co["attacks"]:
        for f in ("detection_rate", "evasion_rate", "budget", "median_l2_raw_of_successful_change",
                  "median_features_changed", "example_raw_shifts"):
            cells += 1
            if co["attacks"][k][f] == cn["attacks"][k][f]: same += 1
            else: diffs.append((co["eps_std"], k, f, co["attacks"][k][f], cn["attacks"][k][f]))
print(f"{cells} cells compared, {same} identical, {len(diffs)} differ")
for d in diffs: print("DIFF", d)
print("n_flows", old["n_flows"], new["n_flows"], "| provenance sha", old["eval_set_provenance"]["sha256_of_flow_matrix"], new["eval_set_provenance"]["sha256_of_flow_matrix"],
      "equal:", old["eval_set_provenance"]["sha256_of_flow_matrix"] == new["eval_set_provenance"]["sha256_of_flow_matrix"])
print("--- server vs client")
tot_srv = tot_cli = tot_bud = 0
for c in new["curve"]:
    for k in ("random_search", "fd_fgsm"):
        a = c["attacks"][k]
        if "server_measured" not in a: continue
        s, cl, b = a["server_measured"], a["client_measured"], a["budget"]["model_calls"]
        tot_srv += s["rows_scored"]; tot_cli += cl["rows"]; tot_bud += b
        print(f"eps={c['eps_std']:<6} {k:14s} server_rows={s['rows_scored']:6d} server_req={s['requests']:6d} client_rows={cl['rows']:6d} client_req={cl['requests']:6d} budget_calls={b:6d} server==client:{s['rows_scored']==cl['rows'] and s['requests']==cl['requests']}")
fin = new["transport"]["server_stats_final"]
print("TOTALS per-cell: server_rows", tot_srv, "client_rows", tot_cli, "budget", tot_bud)
print("server final /stats", fin, "minus timing-run baseline", BASE, "=", fin["rows_scored"] - BASE)
