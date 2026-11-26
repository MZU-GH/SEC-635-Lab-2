"""Recompute the report's derived claims from the raw data and the model weights.

verify_report_numbers.py only proves the report agrees with results/*.json. If a JSON
is itself wrong, that gate passes happily. This script recomputes the load-bearing
claims from source instead: the feature classifier, the training/test matrices, the two
model pipelines, and the Lab 1 metrics. It is the check that would have caught the two
mechanism errors found earlier (the invented "8x scaler" story and the raw-vs-standardized
smoothing sigma) because both are arithmetic, not judgement.

    .venv/bin/python scripts/verify_claims.py

Exit code non-zero if any claim fails. Loads the full split; takes a couple of minutes.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.dataset import build_split  # noqa: E402
from detector.evasion import evaluation_flows, raw_radius  # noqa: E402
from detector.features import classify, project_feasible  # noqa: E402

FAILS: list[str] = []
N = 0


def claim(label: str, ok: bool, detail: str = "") -> None:
    global N
    N += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    split = build_split()
    names = split.feature_names
    base = joblib.load(ROOT / "lab1" / "outputs" / "models"
                       / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    defd_path = ROOT / "models" / "task1c_defended.joblib"
    mutable = classify(names)

    print("feature contract")
    claim("52 mutable / 25 immutable of 77",
          int(mutable.sum()) == 52 and int((~mutable).sum()) == 25,
          f"{int(mutable.sum())}/{int((~mutable).sum())}/{len(names)}")

    print("\ndegenerate (constant-in-train) features")
    tr_std = np.std(split.X_train, axis=0)
    te_std = np.std(split.X_test, axis=0)
    sb = base.named_steps["scaler"].scale_
    degen = (tr_std == 0) & (sb == 1.0)
    claim("exactly 10 features constant in the clean training set",
          int(degen.sum()) == 10, f"found {int(degen.sum())}")
    claim("all 10 also constant in the test set", bool((te_std[degen] == 0).all()))
    claim("exactly 5 of them are attacker-mutable",
          int((degen & mutable).sum()) == 5, f"found {int((degen & mutable).sum())}")

    print("\nstandardized move the defended model sees under the REAL attack box")
    # The attack box of every defended curve is raw_radius(sb, eps, mutable) with the
    # ORIGINAL scaler sb (detector/defend.py "scale = base..."), while the defended pipeline
    # standardizes with its own refit scaler. The move it sees is measured by pushing real
    # rows and the box through each pipeline's scaler (not by the formula eps*sb/sd).
    defd = joblib.load(defd_path) if defd_path.exists() else None
    if defd is not None and os.environ.get("GATE_DEFENDED_IS_BASE"):
        defd = base                      # negative test only: substitute the base pipeline
        print("  (GATE_DEFENDED_IS_BASE set: defended pipeline replaced by the base one)")
    sd = defd.named_steps["scaler"].scale_ if defd else None
    eps = 0.15
    Xs = evaluation_flows(split, base, n=300)
    box = raw_radius(sb, eps, mutable.astype(float))

    def seen(model):
        sc = model.named_steps["scaler"]
        return (sc.transform(Xs + box[None, :]) - sc.transform(Xs)).mean(axis=0)

    mv_b = seen(base)
    claim("standardized move == eps under the ORIGINAL scaler (real box)",
          np.allclose(mv_b[mutable], eps, rtol=0, atol=1e-6))
    dgm = degen & mutable
    if sd is not None:
        mv_d = seen(defd)
        claim("the two scalers DO differ on the 5 degenerate features",
              bool((sd[dgm] != sb[dgm]).all()),
              f"base={sb[dgm][:3]} defended={sd[dgm][:3]}")
        claim("defended model sees standardized move eps*sb/sd on the 5 features under the real "
              "box, which is NOT eps (measured through the defended scaler)",
              bool(np.allclose(mv_d[dgm], eps * sb[dgm] / sd[dgm], rtol=1e-6)
                   and not np.allclose(mv_d[dgm], eps, rtol=1e-3)),
              f"seen={np.round(mv_d[dgm], 4)} eps={eps}")

    print("\nthe REAL mechanism: dead weights become live weights")
    import torch
    from detector.defend import to_torch
    X = evaluation_flows(split, base, n=300)
    dg = np.where(degen & mutable)[0]

    def sensitivity(model, cols):
        # move = the REAL attack box (original scaler sb), as in every defended curve
        t = to_torch(model)
        sc = sb
        Xt = torch.tensor(X, dtype=torch.float32)
        out = []
        for j in cols:
            x = Xt.clone()
            x[:, j] = x[:, j] + eps * sc[j] * float(mutable[j])
            with torch.no_grad():
                out.append(float((t.logit(x) - t.logit(Xt)).abs().mean()))
        return np.array(out)

    sens_b = sensitivity(base, dg)
    claim("undefended model has EXACTLY ZERO logit sensitivity to the 5 features",
          bool((sens_b == 0).all()), f"{np.round(sens_b, 4)}")
    if defd is not None:
        sens_d = sensitivity(defd, dg)
        claim("defended model has LIVE weights on them under the real box (all > 0.1)",
              bool((sens_d > 0.1).all()), f"{np.round(sens_d, 3)}")

    print("\nsmoothing sigma must be standardized, not raw")
    if sd is not None:
        eff = 0.25 / sd          # raw sigma 0.25 expressed in standardized units
        claim("a raw sigma of 0.25 is ~1e-3 standardized (i.e. no noise) - the bug",
              np.median(eff) < 1e-2, f"median effective sigma_std = {np.median(eff):.2e}")

    print("\nevaluation-set identity across Task 1")
    Xe = evaluation_flows(split, base, n=300)
    h = hashlib.sha256(np.ascontiguousarray(Xe, dtype=np.float64).tobytes()).hexdigest()[:16]
    import json
    for f in ("task1b_evasion_wednesday.json", "task1c_defense.json",
              "task1c_ablation.json", "task1c_ablation_rounds.json"):
        p = ROOT / "results" / f
        if not p.exists():
            continue
        prov = json.loads(p.read_text()).get("eval_set_provenance", {})
        claim(f"{f} provenance sha matches recomputed", prov.get("sha256_of_flow_matrix") == h,
              f"{prov.get('sha256_of_flow_matrix')} vs {h}")

    print("\nLab 1 reproduction (report claims 6-dp agreement)")
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    pred = base.predict(split.X_test)
    prob = base.predict_proba(split.X_test)[:, 1]
    y = split.y_test
    claim("precision 0.999992", round(precision_score(y, pred), 6) == 0.999992,
          f"{precision_score(y, pred):.6f}")
    claim("recall 1.000000", round(recall_score(y, pred), 6) == 1.000000,
          f"{recall_score(y, pred):.6f}")
    claim("F1 0.999996", round(f1_score(y, pred), 6) == 0.999996, f"{f1_score(y, pred):.6f}")
    claim("ROC-AUC 1.000000", round(roc_auc_score(y, prob), 6) == 1.000000,
          f"{roc_auc_score(y, prob):.6f}")

    print("\nfeasibility projection")
    claim("project_feasible is an exact no-op on clean rows",
          bool(np.array_equal(split.X_test[:50000],
                              project_feasible(split.X_test[:50000], names))),
          "50,000 clean test rows")

    print("\nKB size")
    try:
        import chromadb
        c = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
        col = c.get_collection("sec635_kb")
        claim("2498 documents in the collection", col.count() == 2498, f"{col.count()}")
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] chroma not available: {type(e).__name__}: {e}")

    print(f"\n{N} claims checked, {len(FAILS)} failures")
    if FAILS:
        print("FAILED:", "; ".join(FAILS))
        return 1
    print("all derived claims verified from source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
