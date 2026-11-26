"""Numbers the report prints that no other stored result backs (number-coverage audit).

Each block states what is measured, then stores it under results/report_extras.json so
that gate 1 (scripts/verify_report_numbers.py) can compare the report against a file
instead of against a docstring or a memory. Needs torch, sklearn and the full Wednesday
split: run it on the GPU server, never on the analysis host.

    .venv/bin/python -u scripts/report_extras.py [--skip-radius]
    .venv/bin/python -u scripts/report_extras.py --only-sensitivity   # refresh just that block

Blocks
  torch_port        max |p_torch - p_sklearn| over 5,000 held-out rows (detector/torch_model.py)
  split             where the 70/30 split falls in raw-timestamp order (timestamp of the last train row)
  scalers           refit scaler vs Lab 1 scaler: median ratio of scale_, ratio on the five
                    constant-in-training mutable features
  sensitivity       mean |delta logit| from pushing one feature to its eps=0.15 box corner
                    (the definition used by scripts/verify_claims.py), original and defended
  radius_010        eps=0.10 and eps=0.25 PGD-40 on every training row against the Lab 1
                    model: mean logit of the attack rows before/after, and the fraction of
                    rows (all rows / attack rows only) whose predicted class flips
  restricted_pgd    evasion rate of per-row PGD-40 against the shipped two-round defended
                    model when only some feature groups may move (300 evaluation flows)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from detector.dataset import TEMPORAL_TRAIN_FRACTION, build_split  # noqa: E402
from detector.defend import attack_pgd_batch, to_torch  # noqa: E402
from detector.evasion import attack_pgd, evaluation_flows, raw_radius  # noqa: E402
from detector.features import classify  # noqa: E402
from detector.torch_model import LAB1_MLP, verify_equivalence  # noqa: E402

OUT = ROOT / "results" / "report_extras.json"


def save(res: dict) -> None:
    OUT.write_text(json.dumps(res, indent=2))


def sensitivity_block(split, names, base, defd, mutable, degen, dg_mut, res, final=True) -> int:
    X = evaluation_flows(split, base, n=300)
    eps = 0.15

    def sensitivity(model):
        t = to_torch(model)
        sc = model.named_steps["scaler"].scale_
        Xt = torch.tensor(X, dtype=torch.float32)
        out = np.zeros(len(names))
        for j in range(len(names)):
            if not mutable[j]:
                continue
            x = Xt.clone()
            x[:, j] = x[:, j] + eps * sc[j]
            with torch.no_grad():
                out[j] = float((t.logit(x) - t.logit(Xt)).abs().mean())
        return out

    sb_s, sd_s = sensitivity(base), sensitivity(defd)
    real = mutable & ~degen
    top5_real = np.sort(sd_s[real])[::-1][:5]
    res["sensitivity"] = {
        "eps_std": eps,
        "undefended_on_degenerate_mutable": [float(x) for x in sb_s[dg_mut]],
        "defended_on_degenerate_mutable": [float(x) for x in sd_s[dg_mut]],
        "defended_degenerate_min": float(sd_s[dg_mut].min()),
        "defended_degenerate_max": float(sd_s[dg_mut].max()),
        "defended_real_mutable_median_all": float(np.median(sd_s[real])),
        "defended_real_mutable_top5": [float(x) for x in top5_real],
        "defended_real_mutable_top5_median": float(np.median(top5_real)),
        "defended_real_mutable_top10_median": float(np.median(np.sort(sd_s[real])[::-1][:10])),
        "n_real_mutable": int(real.sum())}
    sens = res["sensitivity"]
    dnames = [names[i] for i in np.where(dg_mut)[0]]
    sens["degenerate_mutable_features"] = dnames
    sens["undefended_by_feature"] = {n: float(v) for n, v in zip(dnames, sb_s[dg_mut])}
    sens["defended_by_feature"] = {n: float(v) for n, v in zip(dnames, sd_s[dg_mut])}
    sens["undefended_all_exactly_zero"] = bool((sb_s[dg_mut] == 0).all())
    sens["n_real_mutable_above_degenerate_max"] = int((sd_s[real] > sd_s[dg_mut].max()).sum())
    sens["n_real_mutable_below_degenerate_min"] = int((sd_s[real] < sd_s[dg_mut].min()).sum())
    sens["measurement"] = ("per feature j: mean over the 300 evaluation flows of |logit(x with "
                           "x_j + 0.15*scale_j) - logit(x)|, one feature at a time, upper box "
                           "corner, scale_ and logit of the model under test (undefended Lab 1 "
                           "model, or the shipped two-round defended refit); real mutable = the "
                           "47 mutable features that are not constant in training")
    sens["formatted"] = {"degenerate_min_2dp": f"{sens['defended_degenerate_min']:.2f}",
                         "degenerate_max_2dp": f"{sens['defended_degenerate_max']:.2f}",
                         "real_median_2dp": f"{sens['defended_real_mutable_median_all']:.2f}",
                         "real_top5_median_1dp": f"{sens['defended_real_mutable_top5_median']:.1f}"}
    save(res)
    if final:
        print(json.dumps(res["sensitivity"], indent=1))
        print("wrote", OUT)
    return 0


def main() -> int:
    skip_radius = "--skip-radius" in sys.argv
    only_sens = "--only-sensitivity" in sys.argv
    split = build_split()
    names = split.feature_names
    base = joblib.load(LAB1_MLP)
    defd = joblib.load(ROOT / "models" / "task1c_defended.joblib")
    mutable = classify(names)
    sb = base.named_steps["scaler"].scale_
    sd = defd.named_steps["scaler"].scale_
    train_std = np.std(split.X_train, axis=0)
    degen = (train_std == 0) & (sb == 1.0)
    dg_mut = degen & mutable
    res: dict = json.loads(OUT.read_text()) if only_sens else {}
    res.update({"day": "Wednesday",
                 "definitions": {k: v for k, v in (
                     ("sensitivity", "mean over the 300 evaluation flows of |logit(x with "
                      "feature j moved by +eps*scale_j) - logit(x)|, eps=0.15, "
                      "scale_ of the model under test (same as scripts/verify_claims.py)"),
                     ("radius_010", "PGD-40 batch attack (detector.defend.attack_pgd_batch) on "
                      "every training row, all mutable features, radius eps*scale_ of the ORIGINAL "
                      "model; 'crossed' = predicted class flips (as in ablate_1c_rounds.py)"),
                     ("restricted_pgd", "per-row detector.evasion.attack_pgd against the shipped "
                      "defended model, radius from the ORIGINAL scaler, 300 evaluation flows; "
                      "evasion = P(attack) < 0.5 afterwards"))}})

    # ---- torch port
    if only_sens:
        return sensitivity_block(split, names, base, defd, mutable, degen, dg_mut, res)
    rows = np.random.default_rng(0).choice(len(split.X_test), 5000, replace=False)
    res["torch_port"] = verify_equivalence(split.X_test[rows], split.y_test[rows])
    save(res)

    # ---- split position
    n_tr = len(split.X_train)
    ts = split.timestamps
    res["split"] = {"train_fraction": TEMPORAL_TRAIN_FRACTION, "n_train": int(n_tr),
                    "n_test": int(len(split.X_test)),
                    "last_train_timestamp": str(ts.iloc[n_tr - 1]),
                    "first_test_timestamp": str(ts.iloc[n_tr])}
    save(res)

    # ---- scalers
    ratio = sd / sb
    res["scalers"] = {"median_ratio_all_77": float(np.median(ratio)),
                      "n_features": int(len(sb)),
                      "degenerate_mutable_features": [names[i] for i in np.where(dg_mut)[0]],
                      "original_scale_on_them": [float(x) for x in sb[dg_mut]],
                      "defended_scale_on_them": [float(x) for x in sd[dg_mut]],
                      "ratio_on_them": [float(x) for x in ratio[dg_mut]]}
    save(res)

    sensitivity_block(split, names, base, defd, mutable, degen, dg_mut, res, final=False)

    # ---- restricted PGD on the shipped defended model
    t_def = to_torch(defd)
    mut_f = mutable.astype(float)
    groups = {"all_mutable": mut_f,
              "degenerate_only": mut_f * degen,
              "non_degenerate_only": mut_f * (~degen)}
    res["restricted_pgd"] = {"n_flows": int(len(X)), "curve": []}
    for e in (0.15, 0.25):
        row = {"eps_std": e}
        for g, m in groups.items():
            rad = raw_radius(sb, e, m)
            ev = 0
            for xi in X:
                xa, _ = attack_pgd(t_def, xi, rad, feature_names=names)
                ev += int(defd.predict_proba(xa[None, :])[0][1] < 0.5)
            row[g] = {"evasion_rate": ev / len(X), "n_features_allowed": int((m > 0).sum())}
            print(f"restricted eps={e} {g}: evasion {ev / len(X):.3f}", flush=True)
        res["restricted_pgd"]["curve"].append(row)
        save(res)

    # ---- eps = 0.10 / 0.25 training-radius probe (the slow block)
    if not skip_radius:
        t_base = to_torch(base)
        Xtr, ytr = split.X_train, split.y_train
        res["radius_probe"] = []
        for e in (0.10, 0.25):
            t0 = time.time()
            rad = raw_radius(sb, e, mut_f)
            adv = attack_pgd_batch(t_base, Xtr, rad, feature_names=names)
            p_adv = base.predict_proba(adv)[:, 1]
            pred_adv = (p_adv >= 0.5).astype(int)
            att = ytr == 1
            with torch.no_grad():
                lg_clean = t_base.logit(torch.tensor(Xtr, dtype=torch.float32)).numpy()
                lg_adv = t_base.logit(torch.tensor(adv, dtype=torch.float32)).numpy()
            res["radius_probe"].append({
                "eps_std": e, "n_rows": int(len(Xtr)), "n_attack_rows": int(att.sum()),
                "mean_logit_attack_rows_clean": float(lg_clean[att].mean()),
                "mean_logit_attack_rows_adv": float(lg_adv[att].mean()),
                "mean_logit_all_rows_clean": float(lg_clean.mean()),
                "mean_logit_all_rows_adv": float(lg_adv.mean()),
                "frac_all_rows_flipped": float((pred_adv != ytr).mean()),
                "frac_attack_rows_flipped": float((pred_adv[att] == 0).mean()),
                "seconds": round(time.time() - t0, 1)})
            print("radius probe", res["radius_probe"][-1], flush=True)
            save(res)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
