"""Is the FD-FGSM eps=0.15 gap between the recommended defense and the undefended model noise?

The compared rates (same 300 evaluation flows, so the outcomes are paired):
  A = undefended Lab 1 model, FD-FGSM at eps=0.15       (results/task1b_evasion_wednesday.json)
  B = recommended defense (1 round, constant features frozen, config full_1r_frozen),
      FD-FGSM at eps=0.15                               (results/task1c_ablation_rounds.json)

Two stages.
  --recompute-perflow  (GPU server only; retrains the 1-round frozen defense, ~10+ min)
        stores the per-flow detected/evaded outcome of both models in
        results/noise_check_perflow.json and checks the rates equal the stored ones.
  default (stdlib + scipy)
        count difference, Wilson and Clopper-Pearson 95% intervals for each rate, exact
        McNemar test on the paired outcomes; writes results/noise_check.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PERFLOW = RES / "noise_check_perflow.json"
EPS = 0.15


def recompute_perflow() -> None:
    sys.path.insert(0, str(ROOT))
    import joblib
    import numpy as np
    from detector.dataset import build_split
    from detector.defend import AT_EPS_STD, adversarial_train, to_torch  # noqa: F401
    from detector.evasion import Scorer, attack_fd_fgsm, evaluation_flows, raw_radius
    from detector.features import classify

    split = build_split()
    base = joblib.load(ROOT / "lab1" / "outputs" / "models"
                       / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    mutable = classify(split.feature_names).astype(float)
    X = evaluation_flows(split, base, n=300)
    train_std = np.std(split.X_train, axis=0)
    scale = base.named_steps["scaler"].scale_
    degenerate = (train_std == 0) & (scale == 1.0)
    frozen_mut = mutable.copy()
    frozen_mut[degenerate] = 0.0

    defended, _ = adversarial_train(split, base, split.feature_names, rounds=1,
                                    n_rows=None, mutable=frozen_mut)

    def detected(model, mut):
        rad = raw_radius(scale, EPS, mut)
        sc = Scorer(model, split)
        out = []
        for xi in X:
            x_adv, _ = attack_fd_fgsm(sc, xi, rad)
            out.append(int(model.predict_proba(x_adv[None, :])[0][1] >= 0.5))
        return out

    a = detected(base, mutable)
    b = detected(defended, frozen_mut)
    PERFLOW.write_text(json.dumps({
        "eps_std": EPS, "attack": "fd_fgsm", "n_flows": len(X),
        "undefended_detected": a, "recommended_defense_detected": b,
        "code": {"script": "scripts/noise_check.py --recompute-perflow",
                 "date": str(date.today())}}, indent=1))
    print("wrote", PERFLOW.name, "undefended", sum(a), "defense", sum(b))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main() -> int:
    from scipy.stats import beta, binomtest
    b1 = json.loads((RES / "task1b_evasion_wednesday.json").read_text())
    ab = json.loads((RES / "task1c_ablation_rounds.json").read_text())
    ra = next(r for r in b1["curve"] if r["eps_std"] == EPS)["attacks"]["fd_fgsm"]["detection_rate"]
    cfg = next(c for c in ab["configs"] if c["config"] == "full_1r_frozen")
    rb = next(r for r in cfg["curve_after"] if r["eps_std"] == EPS)["attacks"]["fd_fgsm"]["detection_rate"]
    n = b1["n_eval_flows"]
    assert n == ab["n_eval_flows"] == 300
    ka, kb = round(ra * n), round(rb * n)
    assert abs(ka / n - ra) < 1e-9 and abs(kb / n - rb) < 1e-9

    def cp(k):
        return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
                1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))
    out = {"eps_std": EPS, "attack": "fd_fgsm", "n_flows": n,
           "A_undefended": {"detected": ka, "rate": ka / n, "wilson95": wilson(ka, n),
                            "clopper_pearson95": cp(ka)},
           "B_recommended_defense_1r_frozen": {"detected": kb, "rate": kb / n,
                            "wilson95": wilson(kb, n), "clopper_pearson95": cp(kb)},
           "count_difference_A_minus_B": ka - kb,
           "rate_difference_A_minus_B": (ka - kb) / n}
    out["one_flow_equals"] = 1 / n
    if PERFLOW.exists():
        pf = json.loads(PERFLOW.read_text())
        a, b = pf["undefended_detected"], pf["recommended_defense_detected"]
        assert sum(a) == ka and sum(b) == kb, "per-flow recompute disagrees with stored rates"
        n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # detected only by A
        n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # detected only by B
        p = binomtest(min(n10, n01), n10 + n01, 0.5, alternative="two-sided").pvalue if n10 + n01 else 1.0
        out["mcnemar_exact"] = {"A_only_detected": n10, "B_only_detected": n01,
                                "both_detected": sum(1 for x, y in zip(a, b) if x and y),
                                "neither_detected": sum(1 for x, y in zip(a, b) if not x and not y),
                                "p_value_two_sided": float(p),
                                "significant_at_0.05": bool(p < 0.05)}
    else:
        out["mcnemar_exact"] = None
    out["code"] = {"script": "scripts/noise_check.py", "date": str(date.today())}
    (RES / "noise_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    if "--recompute-perflow" in sys.argv:
        recompute_perflow()
        raise SystemExit(0)
    raise SystemExit(main())
