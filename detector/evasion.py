"""Task 1B: evasion - make attack flows score benign with the smallest change.

Three attacks of deliberately different strength, so the report can show that a weak
attack understates risk (the textbook's warning: "a detector that resists PGD-6 but
fails against PGD-40 is not robust: it was tested too weakly").

  random-search   black-box, labels/scores only. Weakest. Budget: N API queries.
  fd-fgsm         black-box, gradient by central finite differences. Middle.
                  Budget: 2 * 52 * d model calls per gradient.
  pgd             white-box, exact autograd gradients. Strongest; the ceiling.
                  Budget: 40 gradient computations.

Every attack is constrained in two ways that a naive implementation is not:
  1. only the 52 attacker-mutable features move (detector/features.py);
  2. `project_feasible` re-imposes physical consistency after every step, so the
     adversarial flow is still a flow a sensor could emit.

Perturbation size is reported in standardized units (the epsilon the attack optimises)
AND in raw units per feature, because "epsilon = 0.10" is meaningless to a network
engineer but "Flow Duration changed by 40 ms" is not.

Detection-rate denominator follows the textbook convention: only attack flows the
detector already flags confidently (P > 0.9) are counted, so an evasion rate is not
flattered by flows that were never detected in the first place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import Split, build_split  # noqa: E402
from detector.features import classify, project_feasible  # noqa: E402

SEED = 42
EPS_GRID = [0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.25, 0.5]  # standardized units
DETECT_THRESHOLD = 0.5
CONFIDENT = 0.9


def evaluation_flows(split: Split, reference_model, n: int = 300,
                     seed: int = SEED) -> np.ndarray:
    """The attack-flow set both Task 1B and Task 1C measure on.

    Shared and deterministic on purpose. The lab asks for "the same curves again" after
    the defense, and two curves computed on different flow sets are not comparable -
    a defended model that happens to score below 0.9 on some flows would otherwise be
    measured on a smaller, easier set and look more robust than it is.

    The filter uses `reference_model` (always the ORIGINAL Lab 1 model), never the model
    under test, for the same reason.
    """
    rng = np.random.default_rng(seed)
    attack_rows = np.where(split.y_test == 1)[0]
    sample = rng.choice(attack_rows, size=min(n, len(attack_rows)), replace=False)
    X = split.X_test[sample]
    keep = reference_model.predict_proba(X)[:, 1] > CONFIDENT
    return X[keep]


@dataclass
class Budget:
    """Query accounting. The threat model cares how expensive an attack is."""

    model_calls: int = 0
    gradient_computations: int = 0

    def __repr__(self) -> str:
        return f"calls={self.model_calls} grads={self.gradient_computations}"


class Scorer:
    """Black-box access to the detector.

    `via_http=True` routes through the FastAPI service so the query budget is real and
    the attack demonstrably never touches the weights. The in-process path exists only
    to keep the finite-difference sweep tractable; the report states which was used.

    `margin()` matters: this detector's calibrated probability is exactly 1.0 in
    float32 for confidently-attack flows, so differencing the probability gives a
    gradient of zero and the attack cannot move. The logit is the continuous signal a
    real detector would log, and the API returns it for this purpose.
    """

    def __init__(self, model, split: Split, via_http: bool = False,
                 base_url: str = "http://127.0.0.1:8055") -> None:
        self.model = model
        self.split = split
        self.via_http = via_http
        self.base_url = base_url
        self.calls = 0   # HTTP requests / model invocations made by the client
        self.rows = 0    # flows scored by the client

    def p(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.rows += len(X)
        if self.via_http:
            import requests
            r = requests.post(f"{self.base_url}/score_batch",
                              json={"features": X.tolist()}, timeout=600)
            r.raise_for_status()
            return np.asarray(r.json()["scores"], dtype=float)
        return self.model.predict_proba(X)[:, 1]

    def margin(self, X: np.ndarray) -> np.ndarray:
        """logit(P(attack)); the unbounded score the finite-difference attack follows."""
        self.calls += 1
        self.rows += len(X)
        if self.via_http:
            import requests
            r = requests.post(f"{self.base_url}/score_batch",
                              json={"features": X.tolist()}, timeout=600)
            r.raise_for_status()
            p = np.clip(np.asarray(r.json()["scores"], dtype=float), 1e-12, 1 - 1e-12)
            return np.log(p / (1 - p))
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


def raw_radius(scale: np.ndarray, eps_std: float, mutable: np.ndarray) -> np.ndarray:
    """Convert a standardized-unit epsilon into a per-feature raw-unit radius.

    This is the step a naive implementation skips and it changes the result completely.
    A single scalar epsilon applied in raw units means "change Flow Duration by 0.1
    microseconds" (nothing) and "change Protocol by 0.1" (a different protocol) at the
    same time. The textbook's epsilon is defined on standardized features
    (`z = (value - mean) / std`), so the honest box is eps_std * std_j for feature j.

    Immutable features get radius zero, which is how the threat model is enforced.
    """
    return eps_std * scale * mutable


def attack_random(scorer: Scorer, X0: np.ndarray, rad: np.ndarray,
                  rng: np.random.Generator, tries: int = 400) -> tuple[np.ndarray, Budget]:
    """Random search inside the feasible box. Black-box, no gradient, weakest attack.

    Draws candidate perturbations uniformly inside the radius box and keeps the first
    that flips the verdict. Included precisely because it is weak: it is what an
    attacker who knows nothing about the model can do.
    """
    budget = Budget()
    best = X0.copy()
    best_p = scorer.p(X0[None, :])[0]
    for _ in range(tries):
        cand = np.clip(X0 + rng.uniform(-1.0, 1.0, size=X0.shape) * rad,
                       X0 - rad, X0 + rad)
        cand = project_feasible(cand[None, :], scorer.split.feature_names)[0]
        p = scorer.p(cand[None, :])[0]
        budget.model_calls += 1
        if p < DETECT_THRESHOLD:
            return cand, budget
        if p < best_p:
            best, best_p = cand, p
    return best, budget


def finite_diff_grad(scorer: Scorer, X0: np.ndarray, rad: np.ndarray,
                     h_frac: float = 0.05) -> tuple[np.ndarray, Budget]:
    """Central-difference gradient of the logit w.r.t. each mutable feature.

    Two things this has to get right, and a naive version gets wrong:
      * it differences the **logit**, not the probability. This detector outputs exactly
        1.0 in float32 for confident flows, so a probability gradient is identically
        zero and the attack cannot move.
      * the step h is per-feature, a fraction of that feature's own radius. A fixed
        h=1e-3 is below float noise for Flow Duration and enormous for a flag count.

    Cost is 2 model calls per mutable feature per gradient, which is exactly why this
    attack sits between random search and PGD: the gradient is real but expensive, so
    you get one step rather than forty.
    """
    budget = Budget()
    g = np.zeros_like(X0)
    idx = np.where(rad > 0)[0]
    for j in idx:
        h = max(h_frac * rad[j], 1e-9)
        up = X0.copy(); up[j] += h
        dn = X0.copy(); dn[j] -= h
        g[j] = (scorer.margin(up[None, :])[0] - scorer.margin(dn[None, :])[0]) / (2 * h)
        budget.model_calls += 2
    budget.gradient_computations += 1
    return g, budget


def attack_fd_fgsm(scorer: Scorer, X0: np.ndarray, rad: np.ndarray,
                   h_frac: float = 0.05) -> tuple[np.ndarray, Budget]:
    """FGSM using a finite-difference gradient: one gradient, one full-budget step."""
    g, budget = finite_diff_grad(scorer, X0, rad, h_frac)
    # sign() descends the logit; multiply by rad so each feature moves its own full step.
    step = X0 - rad * np.sign(g)
    step = np.clip(step, X0 - rad, X0 + rad)
    step = project_feasible(step[None, :], scorer.split.feature_names)[0]
    budget.model_calls += 1
    return step, budget


def attack_pgd(model_torch, X0: np.ndarray, rad: np.ndarray,
               steps: int = 40, restarts: int = 1,
               feature_names: list[str] | None = None,
               device: str = "cpu") -> tuple[np.ndarray, Budget]:
    """PGD with exact gradients: the strong attack, and the honest ceiling.

    Settings follow the textbook recipe so the numbers are comparable to published
    ones: 40 steps, step size 2.5*radius/40 per feature (so ~2.5 full-radius traversals
    fit in the steps), one random start. The objective is the logit, not cross-entropy:
    cross-entropy saturates to a flat (or infinite) gradient on a model this confident
    and the attack would not move at all.
    """
    import torch

    budget = Budget(gradient_computations=steps * max(1, restarts))
    X0t = torch.tensor(X0, dtype=torch.float32, device=device)
    radt = torch.tensor(rad, dtype=torch.float32, device=device)
    alpha = 2.5 * radt / steps
    lo, hi = X0t - radt, X0t + radt

    best_x = X0t.clone()
    with torch.no_grad():
        best_score = float(model_torch.logit(best_x[None, :]))
    rng = np.random.default_rng(SEED)
    for _ in range(max(1, restarts)):
        jitter = rng.uniform(-1.0, 1.0, size=X0.shape) * rad
        x = (X0t + torch.tensor(jitter, dtype=torch.float32,
                                device=device)).clamp(lo, hi)
        for _ in range(steps):
            x = x.clone().detach().requires_grad_(True)
            # Minimise the logit directly. It is unbounded below, so the gradient stays
            # informative even when P(attack) is 1.0 to float precision; cross-entropy
            # on this model saturates to -inf and PGD would not move at all.
            logit = model_torch.logit(x[None, :]).sum()
            grad, = torch.autograd.grad(logit, x)
            with torch.no_grad():
                x = x - alpha * grad.sign()
                x = torch.clamp(x, lo, hi)
            if feature_names is not None:
                # Feasibility projection in numpy; cheap and keeps the attack honest.
                x_np = project_feasible(x.detach().numpy()[None, :], feature_names)[0]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            score = float(model_torch.logit(x[None, :]))
        if score < best_score:
            best_x, best_score = x.detach().clone(), score
    return best_x.detach().numpy(), budget


def raw_change_summary(X0: np.ndarray, X1: np.ndarray, names: list[str],
                       mutable: np.ndarray) -> dict:
    """Describe a perturbation in raw units, which is the only way a network engineer
    can judge whether it is realistic. A standardized L-inf of 0.10 tells you nothing;
    "Flow Duration moved by 40 ms" does."""
    d = X1 - X0
    moved = np.abs(d) > 1e-9
    per_feature = {names[j]: float(d[j]) for j in np.where(moved)[0]}
    return {
        "l_inf_raw": float(np.max(np.abs(d))) if d.size else 0.0,
        "l2_raw": float(np.linalg.norm(d)),
        "l1_raw": float(np.sum(np.abs(d))),
        "n_features_changed": int(moved.sum()),
        "largest_raw_shifts": dict(sorted(per_feature.items(),
                                          key=lambda kv: -abs(kv[1]))[:8]),
    }


def _server_stats(base_url: str) -> dict:
    import requests
    r = requests.get(f"{base_url}/stats", timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> int:
    import joblib
    from detector.torch_model import FlowMLP, load_lab1_mlp

    ap = argparse.ArgumentParser()
    ap.add_argument("--via-http", action="store_true",
                    help="route the black-box attacks (random search, FD-FGSM) through "
                         "the HTTP API; PGD stays in-process (white-box by design)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8055")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default results/task1b_evasion_wednesday.json)")
    ap.add_argument("--limit-flows", type=int, default=None,
                    help="timing/smoke only: use the first N evaluation flows")
    ap.add_argument("--eps", type=float, nargs="*", default=None,
                    help="timing/smoke only: subset of the eps grid")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    split = build_split()
    mutable = classify(split.feature_names)

    sk_model = joblib.load(ROOT / "lab1" / "outputs" / "models"
                           / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    torch_model = load_lab1_mlp()

    # Denominator convention: only confidently-detected attack flows.
    X = evaluation_flows(split, sk_model, n=300)
    X_full = X
    if args.limit_flows:
        X = X[:args.limit_flows]
    attack_rows = np.where(split.y_test == 1)[0]
    print(f"evaluating evasion on {len(X)} attack flows with P(attack) > {CONFIDENT} "
          f"(of {len(attack_rows)} attack test flows)")

    scorer = Scorer(sk_model, split, via_http=args.via_http, base_url=args.base_url)
    if args.via_http:
        print("server health:", __import__("requests").get(
            f"{args.base_url}/health", timeout=30).json())
    scale = sk_model.named_steps["scaler"].scale_
    results = []
    for eps in (args.eps if args.eps is not None else EPS_GRID):
        row = {"eps_std": eps, "attacks": {}}
        rad = raw_radius(scale, eps, mutable)
        for name in ("random_search", "fd_fgsm", "pgd"):
            t0 = time.time()
            c0, r0 = scorer.calls, scorer.rows
            s0 = _server_stats(args.base_url) if args.via_http else None
            evaded, budget = 0, Budget()
            changes = []
            for xi in X:
                if eps == 0.0:
                    x_adv, b = xi.copy(), Budget()
                elif name == "random_search":
                    x_adv, b = attack_random(scorer, xi, rad, rng, tries=200)
                elif name == "fd_fgsm":
                    x_adv, b = attack_fd_fgsm(scorer, xi, rad)
                else:
                    x_adv, b = attack_pgd(torch_model, xi, rad,
                                          feature_names=split.feature_names)
                budget.model_calls += b.model_calls
                budget.gradient_computations += b.gradient_computations
                if sk_model.predict_proba(x_adv[None, :])[0][1] < DETECT_THRESHOLD:
                    evaded += 1
                    changes.append(raw_change_summary(xi, x_adv, split.feature_names,
                                                      mutable))
            row["attacks"][name] = {
                "detection_rate": 1.0 - evaded / len(X),
                "evasion_rate": evaded / len(X),
                "budget": {"model_calls": budget.model_calls,
                           "gradient_computations": budget.gradient_computations},
                "median_l2_raw_of_successful_change": (
                    float(np.median([c["l2_raw"] for c in changes])) if changes else None),
                "median_features_changed": (
                    float(np.median([c["n_features_changed"] for c in changes]))
                    if changes else None),
                "example_raw_shifts": (changes[0]["largest_raw_shifts"] if changes else {}),
                "seconds": round(time.time() - t0, 1),
            }
            if args.via_http and name != "pgd":
                s1 = _server_stats(args.base_url)
                row["attacks"][name]["client_measured"] = {
                    "requests": scorer.calls - c0, "rows": scorer.rows - r0}
                row["attacks"][name]["server_measured"] = {
                    "requests": s1["requests"] - s0["requests"],
                    "rows_scored": s1["rows_scored"] - s0["rows_scored"]}
        results.append(row)
        line = f"eps={eps:5.3f}  " + "  ".join(
            f"{k}: det={v['detection_rate']:.3f}" for k, v in row["attacks"].items())
        print(line, flush=True)

    out_path = Path(args.out) if args.out else ROOT / "results" / "task1b_evasion_wednesday.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(np.ascontiguousarray(X_full, dtype=np.float64).tobytes()
                         ).hexdigest()[:16]
    payload = {
        "day": "Wednesday", "seed": SEED, "n_flows": int(len(X)),
        "confident_threshold": CONFIDENT, "detect_threshold": DETECT_THRESHOLD,
        "n_mutable_features": int(mutable.sum()),
        "pgd": {"steps": 40, "alpha_rule": "2.5*eps/40", "restarts": 1},
        "curve": results,
        "eval_set_provenance": {
            "n": int(len(X_full)),
            "selector": "evaluation_flows(split, ORIGINAL Lab1 model, n=300, seed=42)",
            "filter": "P(attack) > 0.9 under the original model",
            "sha256_of_flow_matrix": sha,
        },
        "n_eval_flows": int(len(X)),
    }
    if args.via_http:
        payload["transport"] = {
            "random_search": "http", "fd_fgsm": "http",
            "pgd": "in-process white-box",
            "success_check": "in-process (experimenter ground truth, not an attacker query)",
            "base_url": args.base_url,
            "server_stats_final": _server_stats(args.base_url),
        }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
