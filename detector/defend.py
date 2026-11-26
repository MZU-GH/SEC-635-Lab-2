"""Task 1C: harden the detector, then re-run 1A and 1B unchanged.

Two defenses, because the lab asks what the defense *costs* and one defense only gives
one data point:

  adversarial_training  The textbook's recommended defense: min_theta E[max_{||d||<=eps}
                        L(theta, x+d, y)]. sklearn's MLPClassifier has no training-loop
                        hook, so this is an outer loop: generate PGD adversarial rows
                        from the training set, refit on clean+adversarial mixed. Cost is
                        explicit - one PGD-40 run per training row per round.

  randomized_smoothing  The only defense with a *proof*. Chapter 9 gives the recipe:
                        sample n noisy copies, take the majority vote, and the certified
                        radius is R = sigma * Phi^-1(p_hat_lower) where p_hat_lower is
                        the Clopper-Pearson lower bound on the observed vote share.
                        We report the radius distribution over the test set, not one
                        example's radius, because a single file proves nothing.

Both are evaluated against the *same* three attacks and the same epsilon grid as 1B, so
the before/after curves are directly comparable, and clean accuracy is reported next to
robust accuracy every time - the textbook's point being that the gap is the cost.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import Split, build_split  # noqa: E402
from detector.features import classify, project_feasible  # noqa: E402

SEED = 42
AT_EPS_STD = 0.25        # radius the training-time attack uses
AT_STEPS = 40
AT_ROUNDS = 2            # outer generate->refit rounds
AT_FRACTION = 1.0        # attack every training row (textbook does 100%)
SMOKE_N = 40_000         # training rows used when --smoke

# Training radius. Measured on a 100k random subsample, detection rate at
# eps = 0.10 / 0.25 / 0.50:
#   undefended            0.973  0.350  0.000
#   trained at eps=0.10   0.973  0.863  0.350
#   trained at eps=0.25   0.963  0.863  0.350
#   trained at eps=0.50   0.907  0.863  0.350
# The gain at large eps is the same for every training radius; the only thing the radius
# buys is a small loss of clean accuracy at small eps. 0.25 is chosen because it is where
# the undefended model actually collapses (0.350), so the defense is trained around the
# operating point that matters rather than one where the model was never failing.


def make_mlp(seed: int = SEED):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25,
                              random_state=seed, early_stopping=True)),
    ])


def adversarial_train(split: Split, base_model, feature_names: list[str],
                      eps_std: float = AT_EPS_STD, rounds: int = AT_ROUNDS,
                      n_rows: int | None = None, log=print,
                      mutable: np.ndarray | None = None):
    """PGD adversarial training as an outer loop.

    The adversarial rows are generated with the *current* model each round, not once
    with the original, which is the difference between adversarial training and simply
    augmenting with a stale attack.

    `mutable` overrides the attacker-capability mask. The ablation in
    detector/ablate_1c.py uses it to freeze features that are constant in real traffic:
    perturbing them is what makes the refit worse than the model it started from.
    """
    from detector.evasion import raw_radius

    if mutable is None:
        mutable = classify(feature_names)
    scale = base_model.named_steps["scaler"].scale_
    rad = raw_radius(scale, eps_std, mutable)

    X_tr, y_tr = split.X_train, split.y_train
    if n_rows:
        # A random subsample, NOT the first N rows. The rows are in raw-timestamp order and
        # the file opens with hours of pure benign traffic, so `X_train[:20000]` is
        # single-class and the refit collapses to a constant predictor - which looks
        # exactly like "adversarial training destroyed the model" and is nothing of the
        # sort. Verified: the first 50,000 training rows contain zero attacks.
        # This is a uniform draw, not a per-class stratified one; at a 33.9% attack rate
        # a 100k draw lands within a fraction of a point of the true rate, so it is
        # adequate here. It is NOT stratified - do not describe it as stratified, and do
        # not rely on it if the class balance ever gets extreme.
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), size=min(n_rows, len(X_tr)), replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    model = base_model
    history = []
    for r in range(rounds):
        t0 = time.time()
        # Generate adversarial rows with the *current* model, not the original one.
        # Using a stale generator is augmentation, not adversarial training.
        torch_model = to_torch(model)
        adv = attack_pgd_batch(torch_model, X_tr, rad, steps=AT_STEPS,
                               feature_names=feature_names)
        # Keep the true label on the adversarial row - that is the whole point: the
        # model is taught that this perturbed flow is still an attack / still benign.
        X_mix = np.vstack([X_tr, adv])
        y_mix = np.concatenate([y_tr, y_tr])
        X_mix = project_feasible(X_mix, feature_names)

        new = make_mlp()
        new.fit(X_mix, y_mix)
        model = new
        history.append({"round": r, "eps_std": eps_std, "steps": AT_STEPS,
                        "n_clean": int(len(X_tr)), "n_adversarial": int(len(adv)),
                        "train_rows_used": int(len(X_mix)),
                        "seconds": round(time.time() - t0, 1)})
        log(f"  adversarial-training round {r}: {len(X_mix)} rows in "
            f"{history[-1]['seconds']}s")
    return model, {"eps_std": eps_std, "steps": AT_STEPS, "rounds": rounds,
                   "history": history}


def to_torch(sk_pipeline):
    """Wrap a refit sklearn pipeline as a FlowMLP so PGD can use it."""
    from detector.torch_model import FlowMLP
    sc = sk_pipeline.named_steps["scaler"]
    clf = sk_pipeline.named_steps["clf"]
    return FlowMLP(sc.mean_, sc.scale_, clf.coefs_, clf.intercepts_)


def attack_pgd_batch(model_torch, X: np.ndarray, rad: np.ndarray, steps: int = AT_STEPS,
                     feature_names: list[str] | None = None,
                     device: str = "cpu") -> np.ndarray:
    """PGD over a whole batch at once.

    The per-row attack in detector/evasion.py is fine for evaluating a few hundred
    flows, but adversarial training needs a gradient for every training row every
    round, and 731k rows x 40 steps one at a time is hours. The model is a
    77->64->32->1 MLP, so a batched forward/backward costs almost the same as one row
    and this is orders of magnitude faster.

    Same objective (minimise the logit) and same box as the per-row attack, so the
    adversarial rows are the same kind of object - only the loop shape differs.
    """
    import torch

    X0 = torch.tensor(X, dtype=torch.float32, device=device)
    radt = torch.tensor(rad, dtype=torch.float32, device=device)
    alpha = 2.5 * radt / steps
    lo, hi = X0 - radt, X0 + radt
    rng = np.random.default_rng(SEED)
    x = (X0 + torch.tensor(rng.uniform(-1.0, 1.0, size=X.shape) * rad,
                           dtype=torch.float32, device=device)).clamp(lo, hi)
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logit = model_torch.logit(x).sum()
        grad, = torch.autograd.grad(logit, x)
        with torch.no_grad():
            x = torch.clamp(x - alpha * grad.sign(), lo, hi)
        if feature_names is not None:
            x = torch.tensor(project_feasible(x.detach().numpy(), feature_names),
                             dtype=torch.float32, device=device)
    return x.detach().numpy()


def certify(smooth_model, X: np.ndarray, sigma: float, n: int = 1000,
            scale: np.ndarray | None = None,
            alpha: float = 0.05, rng: np.random.Generator | None = None,
            batch: int = 200) -> dict:
    """Randomized-smoothing certified radius, following the textbook recipe.

    For each row: draw n Gaussian samples, take the majority label, then
        R = sigma * Phi^-1(p_lower)
    with p_lower the Clopper-Pearson lower bound on the observed vote share. The bound
    is what makes it a *proof* rather than an estimate: at alpha=0.05 as implemented below
    (beta.ppf(alpha/2, ...)), 1000/1000 votes gives a lower bound of 99.63%, not 100%,
    which is why the radius is finite (0.25 * Phi^-1(0.99632) = 0.670, the report's
    median radius).

    `sigma` is in STANDARDIZED units, matching how the attacks are parameterised, and
    `scale` is the model's own StandardScaler scale_ used to convert it to raw units per
    feature. Passing sigma without scale is a units bug and a silent one: these features
    span 5.6e-2 to 2.6e7 in raw units, so a raw sigma of 0.25 is a median of 9.2e-4
    standardized units of noise - effectively no noise at all, and a certificate that
    looks enormous and proves nothing. The returned radius is in standardized units.
    """
    rng = rng or np.random.default_rng(SEED)
    sigma_raw = sigma if scale is None else sigma * np.asarray(scale, dtype=float)
    radii, shares, abstains = [], [], 0
    for i in range(len(X)):
        votes = 0
        for start in range(0, n, batch):
            k = min(batch, n - start)
            noise = rng.normal(0.0, 1.0, size=(k, X.shape[1])) * sigma_raw
            preds = smooth_model.predict_proba(X[i] + noise)
            votes += int((preds[:, 1] >= 0.5).sum())
        share = max(votes, n - votes) / n
        k_hi = max(votes, n - votes)
        p_lo = 0.0 if k_hi == 0 else float(beta.ppf(alpha / 2, k_hi, n - k_hi + 1))
        if p_lo >= 1.0:
            p_lo = 1.0 - 1e-12
        radius = sigma * float(norm.ppf(p_lo))
        if radius <= 0:
            abstains += 1
        radii.append(radius)
        shares.append(share)
    radii = np.array(radii)
    return {
        "sigma_std": sigma, "n_samples": n, "alpha": alpha,
        "sigma_units": "standardized; converted per feature via scale_ before sampling",
        "n_certified": int(len(radii)), "n_abstain": abstains,
        "radius_median": float(np.median(radii)),
        "radius_p25": float(np.percentile(radii, 25)),
        "radius_p75": float(np.percentile(radii, 75)),
        "radius_mean": float(radii.mean()),
        "frac_radius_gt_0": float((radii > 0).mean()),
        # Per-feature equivalent: an L2 ball of radius R spread evenly over d features
        # is R/sqrt(d) each, which is the number a network engineer can interpret.
        "radius_median_per_feature": float(np.median(radii) / np.sqrt(X.shape[1])),
        "median_vote_share": float(np.median(shares)),
    }


def main() -> int:
    import argparse
    import joblib
    from detector.evasion import (EPS_GRID, Budget, Scorer, attack_fd_fgsm,
                                  attack_pgd, attack_random, raw_radius)

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small run for a quick check")
    ap.add_argument("--skip-smoothing", action="store_true")
    ap.add_argument("--only-smoothing", action="store_true",
                    help="recompute just the certified radius and merge it into the "
                         "existing results/task1c_defense.json (the refit is cached, so "
                         "this is minutes; the curve is left untouched)")
    ap.add_argument("--rows", type=int, default=None,
                    help="training rows to adversarially augment (default: all)")
    ap.add_argument("--out", default=None,
                    help="evaluation-only re-run: write ONLY the curve_after payload here "
                         "(no smoothing, nothing written under results/); requires the "
                         "cached defended model and never retrains")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    split = build_split()
    mutable = classify(split.feature_names)
    base = joblib.load(ROOT / "lab1" / "outputs" / "models"
                       / "task1_wednesday_mlp_neuralnet_temporal.joblib")

    n_rows = args.rows or (SMOKE_N if args.smoke else None)
    print(f"adversarial training on {n_rows or len(split.X_train)} training rows")
    # Cache the refit: a full adversarial-training round costs ~6 minutes, and iterating
    # on the evaluation protocol should not require retraining each time.
    cache = ROOT / "models" / ("task1c_defended_smoke.joblib" if args.smoke
                               or n_rows else "task1c_defended.joblib")
    if args.out and not cache.exists():
        raise SystemExit("--out is evaluation-only and needs the cached defended model")
    if cache.exists():
        print(f"loading cached defended model: {cache}")
        defended, at_meta = joblib.load(cache), {"loaded_from_cache": str(cache.resolve().relative_to(ROOT))}
    else:
        defended, at_meta = adversarial_train(split, base, split.feature_names,
                                              n_rows=n_rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(defended, cache)
        print(f"cached defended model: {cache}")

    # ---- clean accuracy before/after (the cost side) ---------------------------
    Xte, yte = split.X_test, split.y_test
    clean = {"before": float((base.predict(Xte) == yte).mean()),
             "after": float((defended.predict(Xte) == yte).mean())}
    print(f"clean accuracy: before={clean['before']:.4f} after={clean['after']:.4f}")

    # ---- re-run 1B unchanged against the defended model ------------------------
    # Same flow set as Task 1B, chosen by the ORIGINAL model's confident detections.
    # Filtering by the defended model would silently shrink the denominator whenever the
    # defense lowered confidence, making it look more robust by measuring it on less.
    from detector.evasion import evaluation_flows
    X = evaluation_flows(split, base, n=300)
    if len(X) == 0:
        raise SystemExit("no confidently-detected attack flows to evaluate; "
                         "the defense removed all detections - report that as a "
                         "finding rather than dividing by zero")
    scorer = Scorer(defended, split)
    # eps must mean the same physical perturbation in both curves. The defended model's
    # StandardScaler is refit on clean+adversarial data, so its scale_ differs from Lab
    # 1's; using it would silently resize the attack box between the before and after
    # runs and make the two curves incomparable. The radius is therefore always defined
    # in the ORIGINAL model's standardized units.
    scale = base.named_steps["scaler"].scale_
    torch_defended = to_torch(defended)

    out = ROOT / "results"
    name = "task1c_defense_smoke.json" if args.smoke else "task1c_defense.json"

    if args.only_smoothing:
        # Reuse the curve already on disk; only the certificate is being recomputed.
        prior = json.loads((out / name).read_text())
        idx = rng.choice(len(Xte), size=200, replace=False)
        prior["smoothing"] = certify(defended, Xte[idx], sigma=0.25, n=1000,
                                     scale=defended.named_steps["scaler"].scale_)
        prior["smoothing_note"] = (
            "sigma is standardized and converted to raw units per feature via the "
            "DEFENDED model's refit scale_ before sampling. An earlier version sampled "
            "sigma in raw units, which on features spanning 5.6e-2 to 2.6e7 is a median "
            "of 9.2e-4 standardized units - effectively no noise, and a certificate that "
            "proved nothing.")
        (out / name).write_text(json.dumps(prior, indent=2))
        print("smoothing:", json.dumps(prior["smoothing"], indent=2))
        print(f"merged into results/{name}")
        return 0

    curve = []
    for eps in EPS_GRID:
        rad = raw_radius(scale, eps, mutable)
        row = {"eps_std": eps, "attacks": {}}
        for name in ("random_search", "fd_fgsm", "pgd"):
            evaded = 0
            for xi in X:
                if eps == 0.0:
                    x_adv = xi.copy()
                elif name == "random_search":
                    x_adv, _ = attack_random(scorer, xi, rad, rng, tries=200)
                elif name == "fd_fgsm":
                    x_adv, _ = attack_fd_fgsm(scorer, xi, rad)
                else:
                    x_adv, _ = attack_pgd(torch_defended, xi, rad,
                                          feature_names=split.feature_names)
                if defended.predict_proba(x_adv[None, :])[0][1] < 0.5:
                    evaded += 1
            row["attacks"][name] = {"detection_rate": 1.0 - evaded / len(X),
                                    "evasion_rate": evaded / len(X)}
        curve.append(row)
        print(f"  defended eps={eps:5.3f} " + "  ".join(
            f"{k}: det={v['detection_rate']:.3f}" for k, v in row["attacks"].items()),
            flush=True)

    payload = {"day": "Wednesday", "seed": SEED, "n_eval_flows": int(len(X)),
               "adversarial_training": at_meta, "clean_accuracy": clean,
               "eps_units": "standardized by the ORIGINAL Lab 1 scaler, shared with Task 1B",
               "curve_after": curve, "smoothing": None}

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"evaluation-only: wrote {args.out}")
        return 0

    if not args.skip_smoothing:
        # Smoothing is applied to the *defended* model; 200 rows x 1000 samples is the
        # textbook's n and keeps the run to minutes rather than hours.
        idx = rng.choice(len(Xte), size=200, replace=False)
        payload["smoothing"] = certify(defended, Xte[idx], sigma=0.25, n=1000,
                                       scale=defended.named_steps["scaler"].scale_)
        print("smoothing:", json.dumps(payload["smoothing"], indent=2))

    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(payload, indent=2))
    print(f"wrote results/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
