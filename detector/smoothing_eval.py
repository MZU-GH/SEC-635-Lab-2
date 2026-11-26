"""Fix 12: detection rate of the randomized-smoothing defended detector under attack.

Smoothed classifier = the DEFENDED model (as in detector/defend.py) with Gaussian noise,
sigma standardized and converted per feature via the defended model's scale_ (exactly what
defend.certify does), n votes, Clopper-Pearson alpha. Per flow: top class by majority vote;
ABSTAIN when the CP lower bound p_lo <= 0.5 (certified radius <= 0, defend.certify's rule).
  detected          = top class is attack and not abstained
  flagged_fail_closed = detected or abstained (abstain treated as an alert)
Adversarial flows come from the existing attacks (detector.evasion.attack_fd_fgsm /
attack_pgd) on evaluation_flows(); eps is in the ORIGINAL Lab 1 model's standardized units.
Sources of the adversarial examples:
  defended_base  : crafted against the defended base classifier (what is being smoothed), transfer to smoothed
  original_lab1  : crafted against the original Lab 1 model (the Task 1B examples), transfer
  eot_pgd        : PGD (attack_pgd, unchanged) on the mean logit of the defended model over noise
                   draws (expectation over transformation) with the same sigma.
Run on the GPU server.
"""
import argparse, hashlib, json, sys
from datetime import date
from pathlib import Path
import numpy as np, joblib, torch
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.dataset import build_split
from detector.features import classify
from detector.evasion import evaluation_flows, raw_radius, Scorer, attack_fd_fgsm, attack_pgd, SEED
from detector.defend import certify, to_torch
from detector.torch_model import load_lab1_mlp

ap = argparse.ArgumentParser()
ap.add_argument("--sigma", type=float, default=0.25)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.10, 0.15, 0.25])
ap.add_argument("--eot-draws", type=int, default=32)
ap.add_argument("--out", default=str(ROOT / "results/task1c_smoothing_detection.json"))
args = ap.parse_args()
SIGMA, N, ALPHA = args.sigma, args.n, args.alpha

split = build_split()
names = split.feature_names
mutable = classify(names)
base = joblib.load(ROOT / "lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib")
defended = joblib.load(ROOT / "models/task1c_defended.joblib")
stored = json.loads((ROOT / "results/task1c_defense.json").read_text())
scale_orig = base.named_steps["scaler"].scale_
scale_def = defended.named_steps["scaler"].scale_
X = evaluation_flows(split, base, n=300)
sha = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()
assert sha[:16] == stored["eval_set_provenance"]["sha256_of_flow_matrix"], "eval set differs from 1B/1C"
print(f"eval set: {len(X)} flows, sha256 {sha}")
sigma_raw = SIGMA * scale_def


def votes_for(model, Xr, rng, batch=200):
    """Same sampling loop as defend.certify; returns per-row attack-vote counts."""
    v = np.zeros(len(Xr), dtype=int)
    for i in range(len(Xr)):
        for s in range(0, N, batch):
            k = min(batch, N - s)
            noise = rng.normal(0.0, 1.0, size=(k, Xr.shape[1])) * sigma_raw
            v[i] += int((model.predict_proba(Xr[i] + noise)[:, 1] >= 0.5).sum())
    return v


def summarize(v):
    top_attack = v >= N - v
    k_hi = np.maximum(v, N - v)
    p_lo = np.array([0.0 if k == 0 else min(float(beta.ppf(ALPHA / 2, k, N - k + 1)), 1 - 1e-12) for k in k_hi])
    abstain = (SIGMA * norm.ppf(p_lo)) <= 0
    det = top_attack & ~abstain
    n = len(v)
    return {"n_flows": n, "detection_rate": float(det.mean()), "abstain_rate": float(abstain.mean()),
            "flagged_fail_closed_rate": float((det | abstain).mean()),
            "majority_attack_rate_ignoring_abstain": float(top_attack.mean()),
            "n_detected": int(det.sum()), "n_abstain": int(abstain.sum()),
            "median_attack_vote_share": float(np.median(v / N))}


class SmoothedLogit:
    """EOT surrogate: mean base logit over fresh Gaussian draws; plugs into attack_pgd unchanged."""
    def __init__(self, base_t, m, seed):
        self.b, self.m = base_t, m
        self.g = torch.Generator().manual_seed(seed)
        self.sig = torch.tensor(sigma_raw, dtype=torch.float32)

    def logit(self, x):
        z = torch.randn((self.m, x.shape[-1]), generator=self.g) * self.sig
        return self.b.logit(x + z).mean()


t_def, t_orig = to_torch(defended), load_lab1_mlp()
sc_def, sc_orig = Scorer(defended, split), Scorer(base, split)
mut = mutable
out_curve = []
for ei, eps in enumerate(args.eps):
    rad = raw_radius(scale_orig, eps, mut)       # ORIGINAL scaler, as in 1B and 1C
    cell = {"eps_std": eps, "sources": {}}
    gens = {
        "defended_base": {"fd_fgsm": lambda x: attack_fd_fgsm(sc_def, x, rad)[0],
                          "pgd": lambda x: attack_pgd(t_def, x, rad, feature_names=names)[0]},
        "original_lab1": {"fd_fgsm": lambda x: attack_fd_fgsm(sc_orig, x, rad)[0],
                          "pgd": lambda x: attack_pgd(t_orig, x, rad, feature_names=names)[0]},
        "eot_pgd": {"pgd_vs_smoothed": None},
    }
    for si, (src, atks) in enumerate(gens.items()):
        cell["sources"][src] = {}
        for ai, (an, fn) in enumerate(atks.items()):
            if eps == 0.0:
                Xa = X.copy()
            elif src == "eot_pgd":
                sm = SmoothedLogit(t_def, args.eot_draws, SEED + 1000 + ei)
                Xa = np.stack([attack_pgd(sm, x, rad, feature_names=names)[0] for x in X])
            else:
                Xa = np.stack([fn(x) for x in X])
            rng = np.random.default_rng(np.random.SeedSequence([SEED, ei, si, ai]))
            r = summarize(votes_for(defended, Xa, rng))
            r["unsmoothed_defended_detection_rate"] = float((defended.predict_proba(Xa)[:, 1] >= 0.5).mean())
            cell["sources"][src][an] = r
            print(f"eps={eps:5.3f} {src:14s} {an:16s} smoothed det={r['detection_rate']:.4f} "
                  f"abstain={r['abstain_rate']:.4f} failclosed={r['flagged_fail_closed_rate']:.4f} "
                  f"unsmoothed det={r['unsmoothed_defended_detection_rate']:.4f}", flush=True)
            if eps == 0.0:
                break
        if eps == 0.0:
            break
    out_curve.append(cell)

# ---------------- sanity checks at eps=0 against task1c_defense.json ----------------
checks = []
def chk(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ": " + detail)

sm = stored["smoothing"]
chk("sigma/n/alpha equal stored", (SIGMA, N, ALPHA) == (sm["sigma_std"], sm["n_samples"], sm["alpha"]),
    f"run=({SIGMA},{N},{ALPHA}) stored=({sm['sigma_std']},{sm['n_samples']},{sm['alpha']})")
# Re-run defend.certify on the stored recipe (--only-smoothing path: fresh rng(SEED) picks 200 test rows).
Xte, yte = split.X_test, split.y_test
idx = np.random.default_rng(SEED).choice(len(Xte), size=200, replace=False)
rc = certify(defended, Xte[idx], sigma=SIGMA, n=N, scale=scale_def, alpha=ALPHA)
keys = ["n_certified", "n_abstain", "radius_median", "radius_p25", "radius_p75", "radius_mean", "median_vote_share"]
diffs = {k: (rc[k], sm[k]) for k in keys if abs(rc[k] - sm[k]) > 1e-9}
chk("certify() on 200 test rows reproduces stored smoothing block", not diffs,
    "all 7 fields equal" if not diffs else f"differs (recomputed, stored): {diffs}")
v_te = votes_for(defended, Xte[idx], np.random.default_rng(SEED))
s_te = summarize(v_te)
acc_s = float(((v_te >= N - v_te) == (yte[idx] == 1)).mean())
acc_stored = stored["clean_accuracy"]["after"]
chk("smoothed clean accuracy on the 200 test rows >= stored clean accuracy - 0.01", acc_s >= acc_stored - 0.01,
    f"smoothed={acc_s:.6f} stored_defended_clean={acc_stored:.6f} abstain_rate={s_te['abstain_rate']:.4f}")
e0 = out_curve[0]["sources"]["defended_base"]["fd_fgsm"] if args.eps[0] == 0.0 else None
st0 = next(r for r in stored["curve_after"] if r["eps_std"] == 0.0)["attacks"]["pgd"]["detection_rate"]
if e0:
    chk("eps=0 smoothed detection on eval flows within 0.01 of stored defended eps=0 detection",
        abs(e0["detection_rate"] - st0) <= 0.01,
        f"smoothed={e0['detection_rate']:.4f} stored_unsmoothed={st0:.4f} abstain={e0['abstain_rate']:.4f}")
# Attack reproduction: unsmoothed defended detection on defended_base examples must equal the stored 1C curve.
for cell in out_curve:
    if cell["eps_std"] == 0.0:
        continue
    row = next(r for r in stored["curve_after"] if abs(r["eps_std"] - cell["eps_std"]) < 1e-9)
    for an in ("fd_fgsm", "pgd"):
        got = cell["sources"]["defended_base"][an]["unsmoothed_defended_detection_rate"]
        want = row["attacks"][an]["detection_rate"]
        chk(f"unsmoothed defended detection reproduces task1c_defense curve_after eps={cell['eps_std']} {an}",
            abs(got - want) < 1e-9, f"got={got:.6f} stored={want:.6f}")
ok = all(c["pass"] for c in checks)
if not ok:
    print("SANITY CHECK FAILED: not writing output", file=sys.stderr)
    raise SystemExit(2)

payload = {
    "day": "Wednesday", "n_flows": int(len(X)), "seeds": {"base": SEED, "smoothing_noise": "SeedSequence([42, eps_idx, source_idx, attack_idx])",
                                                        "eot": "torch.Generator seed 42+1000+eps_idx", "attack_internal": "SEED=42 (attack_pgd)"},
    "smoothing": {"smoothed_model": "defended model (models/task1c_defended.joblib), as in defend.py",
                  "sigma_std": SIGMA, "n_samples": N, "alpha": ALPHA,
                  "sigma_units": "standardized; converted per feature via the DEFENDED model's scale_ (as defend.certify)",
                  "abstain_rule": "CP lower bound (alpha/2, as certify) on top-class vote share <= 0.5, i.e. certified radius <= 0"},
    "eps_units": "standardized by the ORIGINAL Lab 1 scaler (raw_radius with base scale_), same as 1B/1C",
    "eot": {"draws_per_step": args.eot_draws, "objective": "mean base logit over noise draws; attack_pgd unchanged (40 steps, 2.5*rad/40, 1 restart)"},
    "curve": out_curve,
    "sanity_checks": checks,
    "recomputed_certify": rc,
    "eval_set_provenance": {"n": int(len(X)), "sha256_of_flow_matrix": sha[:16], "sha256_full": sha,
                            "selector": "evaluation_flows(split, ORIGINAL Lab1 model, n=300, seed=42)",
                            "equals_task1c_defense_json": True},
    "code": {"script": "detector/smoothing_eval.py", "date": date.today().isoformat()},
}
Path(args.out).write_text(json.dumps(payload, indent=2))
print("wrote", args.out)
