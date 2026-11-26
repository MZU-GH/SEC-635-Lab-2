"""Task 1C, second ablation: which round breaks it, and what should be shipped?

results/task1c_ablation.json established two separate effects:

  * the epsilon=0.15 collapse (0.897 -> 0.000) is the degenerate-scaler lever; freezing
    the 10 constant-in-train features restores 0.870/0.880 at 0.15;
  * the epsilon=0.25 collapse is a *round-count* effect that only appears at full data:
    one round gives 0.777, two rounds give 0.000, while two rounds on a 100k subsample
    still give 0.777.

This script answers the two questions that follow:

  full_1r_frozen   the model the evidence actually recommends shipping - one round,
                   constant features frozen.
  crossing fraction, per round, at full data: what share of the generated adversarial
                   rows actually cross the *generating* model's boundary. Adversarial
                   training only learns from rows that cross; if round 2's generator is
                   already robust at the training radius, round 2 is refitting on hard
                   negatives rather than on failures, which is the mechanism that would
                   explain the regression.

Writes results/task1c_ablation_rounds.json.
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

from detector.ablate_1c import detection_curve, run_config  # noqa: E402
from detector.dataset import build_split  # noqa: E402
from detector.defend import AT_EPS_STD, AT_STEPS, make_mlp, to_torch  # noqa: E402
from detector.evasion import raw_radius  # noqa: E402
from detector.features import classify  # noqa: E402

SEED = 42


def crossing_by_round(split, base, mutable, eps_std, rounds, log):
    """Adversarial-train while recording, per round, the fraction of generated rows that
    the *generating* model misclassifies - i.e. rows that actually crossed."""
    from detector.defend import attack_pgd_batch

    scale = base.named_steps["scaler"].scale_
    rad = raw_radius(scale, eps_std, mutable)
    X_tr, y_tr = split.X_train, split.y_train
    model = base
    out = []
    for r in range(rounds):
        t0 = time.time()
        adv = attack_pgd_batch(to_torch(model), X_tr, rad, steps=AT_STEPS,
                               feature_names=split.feature_names)
        pred_adv = model.predict_proba(adv)[:, 1]
        crossed = float(((pred_adv < 0.5) & (y_tr == 1)).sum() +
                        ((pred_adv >= 0.5) & (y_tr == 0)).sum())
        mixed = np.vstack([X_tr, adv])
        y_mix = np.concatenate([y_tr, y_tr])
        new = make_mlp()
        new.fit(mixed, y_mix)
        model = new
        out.append({"round": r, "n_rows": int(len(X_tr)),
                    "frac_crossed_by_generator": round(crossed / len(X_tr), 6),
                    "seconds": round(time.time() - t0, 1)})
        log(f"  round {r}: generator crossed {crossed/len(X_tr)*100:.3f}% of rows "
            f"in {out[-1]['seconds']}s")
    return model, out


def main() -> int:
    split = build_split()
    names = split.feature_names
    base = joblib.load(ROOT / "lab1" / "outputs" / "models"
                       / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    from detector.evasion import evaluation_flows
    X = evaluation_flows(split, base, n=300)
    mutable = classify(names).astype(float)
    train_std = np.std(split.X_train, axis=0)
    degenerate = (train_std == 0) & (base.named_steps["scaler"].scale_ == 1.0)

    results = {"day": "Wednesday", "seed": SEED, "n_eval_flows": int(len(X)),
               "eps_units": "standardized by the ORIGINAL Lab 1 scaler",
               "configs": [], "crossing_by_round": None}
    out_path = ROOT / "results" / "task1c_ablation_rounds.json"

    print("=== full_1r_frozen (1 round, all rows, constant features frozen) ===",
          flush=True)
    results["configs"].append(
        run_config("full_1r_frozen", split, base, X, mutable, degenerate,
                   None, 1, print))
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== crossing fraction per round (full data, frozen, eps=0.25) ===",
          flush=True)
    train_mutable = mutable.copy()
    train_mutable[degenerate] = 0.0
    _, cross = crossing_by_round(split, base, train_mutable, AT_EPS_STD, 2, print)
    results["crossing_by_round"] = cross
    out_path.write_text(json.dumps(results, indent=2))
    print("\nwrote results/task1c_ablation_rounds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
