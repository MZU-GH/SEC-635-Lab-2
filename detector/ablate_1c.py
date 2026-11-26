"""Task 1C ablation: why did full-data adversarial training lower detection?

The diagnosis that motivated this script (measured, not assumed):

  10 of the 77 features are *identically constant* in Lab 1's clean Wednesday training
  set, so sklearn's StandardScaler gives them scale_ = 1.0 (the zero-variance
  convention). Five of them are numeric-looking counters that the attacker-capability
  classifier marks MUTABLE: Fwd URG Flags, CWE Flag Count, Fwd Byts/b Avg,
  Fwd Pkts/b Avg, Fwd Blk Rate Avg. Adversarial training perturbs them, so in the mixed
  training set they acquire std 0.125 and the refit scaler divides by 0.125 instead of
  1.0. The attack box is unchanged in raw units, but in the defended model's own
  standardized space the same raw move is 8x larger - the defense manufactured a lever
  out of features that are zero in every real flow, in training and in test alike.

  Restricted attack on the full-data defended model (evasion rate, 300 flows):
      eps=0.15  all mutable 0.650   degenerate-only 0.017   non-degenerate 0.120
      eps=0.25  all mutable 1.000   degenerate-only 0.143   non-degenerate 0.650
  Neither group alone reproduces the full attack, so the lever is the degenerate
  features *plus* the ordinary ones; the question the ablation answers is whether
  refusing to perturb the constant features restores the defense.

Four configs, each answering one question:

  full_2r            the headline full-data run (2 rounds) - reuses the cached refit
  full_1r            is the regression caused by the second round?
  sub100k_2r         does the 100k-subsample result survive the eps-units fix?
  full_2r_frozen     as full_2r but the 10 constant features are frozen in BOTH the
                     training-time and the eval-time attack, which is what a threat
                     model written against real traffic should say anyway.

Writes results/task1c_ablation.json, checkpointing after each config. Nothing here is
imported by the other modules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import build_split  # noqa: E402
from detector.defend import (AT_EPS_STD, adversarial_train, to_torch)  # noqa: E402
from detector.evasion import (Scorer, attack_fd_fgsm, attack_pgd, attack_random,  # noqa: E402
                              evaluation_flows, raw_radius)
from detector.features import classify  # noqa: E402

SEED = 42
# random_search is dropped from the grid: it is the weakest attack and the question
# here is about gradient attacks exploiting the refit scaler.
EPS_GRID = [0.05, 0.10, 0.15, 0.25, 0.50]
ATTACKS = ("fd_fgsm", "pgd")


def detection_curve(model, base_scale, mutable, X, split, feature_names):
    """Detection rate per eps per attack, eps in the ORIGINAL model's units."""
    rng = np.random.default_rng(SEED)
    scorer = Scorer(model, split)
    torch_model = to_torch(model)
    curve = []
    for eps in EPS_GRID:
        rad = raw_radius(base_scale, eps, mutable)
        row = {"eps_std": eps, "attacks": {}}
        for name in ATTACKS:
            evaded = 0
            for xi in X:
                if name == "fd_fgsm":
                    x_adv, _ = attack_fd_fgsm(scorer, xi, rad)
                else:
                    x_adv, _ = attack_pgd(torch_model, xi, rad,
                                          feature_names=feature_names)
                if model.predict_proba(x_adv[None, :])[0][1] < 0.5:
                    evaded += 1
            row["attacks"][name] = {"detection_rate": 1.0 - evaded / len(X),
                                    "evasion_rate": evaded / len(X)}
        curve.append(row)
        print(f"    eps={eps:5.3f} " + "  ".join(
            f"{k}={v['detection_rate']:.3f}" for k, v in row["attacks"].items()),
            flush=True)
    return curve


def run_config(name, split, base, X, mutable, frozen, n_rows, rounds, log,
               cached=None):
    """Train one config and evaluate it. `frozen` features get radius 0 everywhere."""
    t0 = time.time()
    train_mutable = mutable.astype(float).copy()
    train_mutable[frozen] = 0.0
    eval_mutable = train_mutable.copy()

    if cached is not None:
        log(f"  [{name}] reusing cached refit {cached}")
        defended = joblib.load(cached)
        meta = {"loaded_from_cache": str(Path(cached).resolve().relative_to(ROOT))}
    else:
        defended, meta = adversarial_train(split, base, split.feature_names,
                                           eps_std=AT_EPS_STD, rounds=rounds,
                                           n_rows=n_rows, log=log,
                                           mutable=train_mutable)

    log(f"  [{name}] curve:")
    curve = detection_curve(defended, base.named_steps["scaler"].scale_, eval_mutable,
                            X, split, split.feature_names)
    Xte, yte = split.X_test, split.y_test
    out = {"config": name, "n_rows": n_rows, "rounds": rounds,
           "n_frozen_features": int(frozen.sum()),
           "adversarial_training": meta,
           "clean_accuracy": float((defended.predict(Xte) == yte).mean()),
           "curve_after": curve, "seconds": round(time.time() - t0, 1)}
    log(f"  [{name}] done in {out['seconds']}s clean_acc={out['clean_accuracy']:.6f}")
    return out


def main() -> int:
    split = build_split()
    names = split.feature_names
    base = joblib.load(ROOT / "lab1" / "outputs" / "models"
                       / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    mutable = classify(names).astype(float)
    X = evaluation_flows(split, base, n=300)

    train_std = np.std(split.X_train, axis=0)
    base_scale = base.named_steps["scaler"].scale_
    degenerate = (train_std == 0) & (base_scale == 1.0)
    print(f"constant-in-train features: {int(degenerate.sum())}, of which mutable: "
          f"{int((degenerate & (mutable > 0)).sum())}")
    print("names:", [names[i] for i in np.where(degenerate)[0]], flush=True)

    zero = np.zeros(len(names), bool)
    configs = [
        ("full_1r", None, 1, zero, None),
        ("sub100k_2r", 100_000, 2, zero, None),
        ("full_2r_frozen", None, 2, degenerate, None),
    ]
    results = {"day": "Wednesday", "seed": SEED, "n_eval_flows": int(len(X)),
               "eps_units": "standardized by the ORIGINAL Lab 1 scaler",
               "attacks": list(ATTACKS),
               "degenerate_features": [names[i] for i in np.where(degenerate)[0]],
               "configs": []}

    # full_2r is the headline run already in results/task1c_defense.json; re-evaluate
    # it here on this grid so all four rows of the table come from one code path.
    cache = ROOT / "models" / "task1c_defended.joblib"
    print("\n=== full_2r (cached refit, 2 rounds, all rows) ===", flush=True)
    results["configs"].append(
        run_config("full_2r", split, base, X, mutable, zero, None, 2, print,
                   cached=cache))
    out_path = ROOT / "results" / "task1c_ablation.json"
    out_path.write_text(json.dumps(results, indent=2))

    for name, n_rows, rounds, frozen, cached in configs:
        print(f"\n=== {name} (rows={n_rows or 'all'}, rounds={rounds}, "
              f"frozen={int(frozen.sum())}) ===", flush=True)
        results["configs"].append(
            run_config(name, split, base, X, mutable, frozen, n_rows, rounds, print,
                       cached=cached))
        out_path.write_text(json.dumps(results, indent=2))

    print("\nwrote results/task1c_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
