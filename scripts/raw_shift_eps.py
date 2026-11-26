"""Fix 11: raw-unit size of an eps=0.10 perturbation, same code path as Task 1B.

Task 1B's `example_raw_shifts` is `raw_change_summary(...)["largest_raw_shifts"]` of the
FIRST successful FD-FGSM evasion in `evaluation_flows()` order. Here:
  * raw radius = raw_radius(original Lab 1 scaler_.scale_, eps, mutable)  (no attack needed)
  * the same example flow (first FD-FGSM evasion at eps=0.25) is attacked at eps=0.10 with
    attack_fd_fgsm and the shifts of the report's two features read off
  * the first FD-FGSM evasion at eps=0.10 is also reported and compared with the stored
    example_raw_shifts of results/task1b_evasion_wednesday.json
Runs on the GPU server (loads the parquet split).
"""
import json, sys, hashlib
from datetime import date
from pathlib import Path
import numpy as np, joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.dataset import build_split
from detector.features import classify
from detector.evasion import (evaluation_flows, raw_radius, Scorer, attack_fd_fgsm,
                              raw_change_summary, SEED)

FEATS = ["Fwd IAT Tot", "Flow IAT Max"]   # the two features in the report's realism sentence
EPS_LO, EPS_REF = 0.10, 0.25

split = build_split()
names = split.feature_names
mutable = classify(names)
base = joblib.load(ROOT / "lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib")
scale = base.named_steps["scaler"].scale_
X = evaluation_flows(split, base, n=300)
sha = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()
stored = json.loads((ROOT / "results/task1b_evasion_wednesday.json").read_text())
prov = stored["eval_set_provenance"]["sha256_of_flow_matrix"]
assert sha[:16] == prov, (sha[:16], prov)
scorer = Scorer(base, split)
grid = [r["eps_std"] for r in stored["curve"]]


def stored_example(eps):
    r = next(r for r in stored["curve"] if abs(r["eps_std"] - eps) < 1e-9)
    return r["attacks"]["fd_fgsm"]["example_raw_shifts"]


def run(xi, eps):
    rad = raw_radius(scale, eps, mutable)
    adv, _ = attack_fd_fgsm(scorer, xi, rad)
    ev = bool(base.predict_proba(adv[None, :])[0][1] < 0.5)
    return adv, ev, raw_change_summary(xi, adv, names, mutable)


def first_evasion(eps):
    for i, xi in enumerate(X):
        adv, ev, ch = run(xi, eps)
        if ev:
            return i, ch
    return None, None


def close(a, b):
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= 1e-6 * max(1, abs(b[k])) for k in a)


i_ref, ch_ref = first_evasion(EPS_REF)
assert close(ch_ref["largest_raw_shifts"], stored_example(EPS_REF)), "0.25 example not reproduced"
print(f"reproduced stored eps=0.25 example (flow index {i_ref}): ok")
i_lo, ch_lo = first_evasion(EPS_LO)
lo_matches = close(ch_lo["largest_raw_shifts"], stored_example(EPS_LO))
print(f"first FD-FGSM evasion at eps=0.10: flow index {i_lo}; matches stored example: {lo_matches}")
same_adv, same_ev, same_ch = run(X[i_ref], EPS_LO)
fi = [names.index(f) for f in FEATS]
rad_lo, rad_ref = raw_radius(scale, EPS_LO, mutable), raw_radius(scale, EPS_REF, mutable)
# median over all successful FD-FGSM evasions at eps=0.10 of the two features' |shift|
sh = []
for xi in X:
    adv, ev, ch = run(xi, EPS_LO)
    if ev:
        sh.append([abs(float(adv[j] - xi[j])) for j in fi])
sh = np.array(sh)
out = {
    "eps_grid_of_1B": grid, "eps_0.10_in_1B_grid": any(abs(g - EPS_LO) < 1e-9 for g in grid),
    "units": "raw feature units of the CIC flow export; for IAT/duration/idle features this is microseconds "
             "(the report divides by 1e6 to print seconds)",
    "scaler": "ORIGINAL Lab 1 model StandardScaler scale_ (raw_radius(scale_, eps, mutable))",
    "features": FEATS,
    "raw_radius_eps_0.10": {f: float(rad_lo[j]) for f, j in zip(FEATS, fi)},
    "raw_radius_eps_0.25": {f: float(rad_ref[j]) for f, j in zip(FEATS, fi)},
    "raw_radius_eps_0.10_seconds": {f: float(rad_lo[j]) / 1e6 for f, j in zip(FEATS, fi)},
    "same_example_flow": {
        "flow_index_in_evaluation_flows": int(i_ref),
        "definition": "first FD-FGSM evasion at eps=0.25 (the flow behind the report's realism sentence)",
        "eps_0.10_fd_fgsm_evaded": same_ev,
        "eps_0.10_shift": {f: float(same_adv[j] - X[i_ref][j]) for f, j in zip(FEATS, fi)},
        "eps_0.10_shift_seconds": {f: float(same_adv[j] - X[i_ref][j]) / 1e6 for f, j in zip(FEATS, fi)},
        "eps_0.10_n_features_changed": same_ch["n_features_changed"],
    },
    "first_evasion_at_eps_0.10": {
        "flow_index": int(i_lo), "matches_stored_example_raw_shifts": lo_matches,
        "shifts": {f: ch_lo["largest_raw_shifts"].get(f) for f in FEATS},
        "shifts_seconds": {f: (ch_lo["largest_raw_shifts"].get(f) or 0) / 1e6 for f in FEATS},
        "n_features_changed": ch_lo["n_features_changed"],
    },
    "n_successful_fd_fgsm_evasions_eps_0.10": int(len(sh)),
    "median_abs_shift_over_successful_evasions_eps_0.10": {f: float(np.median(sh[:, k])) for k, f in enumerate(FEATS)},
    "median_abs_shift_seconds": {f: float(np.median(sh[:, k])) / 1e6 for k, f in enumerate(FEATS)},
    "eval_set_provenance": {**stored["eval_set_provenance"], "sha256_full_recomputed": sha},
    "seed": SEED,
    "code": {"script": "scripts/raw_shift_eps.py", "date": date.today().isoformat()},
}
Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/task1b_raw_shift_eps010.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("eps_0.10_in_1B_grid", "raw_radius_eps_0.10", "raw_radius_eps_0.10_seconds",
      "same_example_flow", "first_evasion_at_eps_0.10", "n_successful_fd_fgsm_evasions_eps_0.10",
      "median_abs_shift_seconds")}, indent=1))
