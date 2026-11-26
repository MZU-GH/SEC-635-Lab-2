"""Does the original-unit attack box give the defended model a different standardized move?

The attack box for every defended-model curve is raw_radius(sb, eps, mutable) with sb the
ORIGINAL Lab 1 scaler (detector/defend.py, detector/ablate_1c.py, detector/smoothing_eval.py),
but the defended pipeline standardizes with its own refit scaler sd. This script measures,
by pushing real rows and the real box through the defended pipeline's scaler, what
standardized move the defended model sees, and how the per-feature logit sensitivity
changes between a move of eps*sd_j (the defended model's own units) and the real box eps*sb_j.

    .venv/bin/python -u scripts/scale_effect.py        # writes results/task1c_scale_effect.json
    .venv/bin/python scripts/scale_effect.py --augment # only (re)computes the derived blocks

Needs torch and the full Wednesday split: run on the GPU server. `--augment` reads the stored
JSONs only (numpy, seconds, any machine) and rewrites the two derived blocks `l2_extent` and
`evasion_reference`, which the report quotes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from detector.dataset import build_split  # noqa: E402
from detector.defend import to_torch  # noqa: E402
from detector.evasion import attack_pgd, evaluation_flows, raw_radius  # noqa: E402
from detector.features import classify  # noqa: E402
from detector.torch_model import LAB1_MLP  # noqa: E402

OUT = ROOT / "results" / "task1c_scale_effect.json"
EPS = 0.15


def derived_blocks(res: dict, b1: dict, df: dict) -> dict:
    """Numbers the report quotes that are differences, sums or comparisons of stored values.

    l2_extent: the L2 length of the attack's change in the DEFENDED model's standardized units,
    sqrt(sum_j m_j^2) over the 52 mutable features with m_j the mean standardized move that
    the defended pipeline's own scaler sees under the real box (per_feature), against the
    naive eps*sqrt(52) that assumes the box is eps in every coordinate.
    evasion_reference: PGD-40 evasion on the shipped defended model with the real box and with
    the five degenerate features at their own-unit radius, against the undefended model
    (1 - detection rate of task1b), and the share of the excess evasion that the own-unit
    radius removes. `mostly_scale_effect` is that share > 0.5.
    """
    pf = res["per_feature"]
    eps = res["eps_std"]
    n = len(pf)
    l2_def = float(np.sqrt(sum(v["std_move_defended"] ** 2 for v in pf.values())))
    l2_und = float(np.sqrt(sum(v["std_move_undefended"] ** 2 for v in pf.values())))
    naive = float(eps * np.sqrt(n))
    cert = df["smoothing"]["radius_median"]
    l2 = {"n_mutable": n, "eps_std": eps, "l2_defended_units": l2_def, "l2_undefended_units": l2_und,
          "l2_naive_eps_sqrt_n": naive, "certified_radius_median": cert,
          "certificate_below_defended_extent": bool(cert < l2_def),
          "certificate_below_naive_extent": bool(cert < naive),
          "definition": "sqrt(sum over the mutable features of (mean standardized move)^2), "
                        "per_feature.std_move_defended; naive = eps*sqrt(n_mutable)"}
    und = {c["eps_std"]: c["attacks"]["pgd"] for c in b1["curve"]}
    ref = []
    for row in res["pgd_scale_test"]["curve"]:
        e = row["eps_std"]
        u = 1.0 - und[e]["detection_rate"]
        real = row["all_mutable_real_box"]["evasion_rate"]
        own = row["all_mutable_five_own_units"]["evasion_rate"]
        ex_real, ex_own = real - u, own - u
        share = (ex_real - ex_own) / ex_real if ex_real > 0 else None
        ref.append({"eps_std": e, "undefended_pgd_evasion": u,
                    "defended_real_box_evasion": real, "defended_five_own_units_evasion": own,
                    "excess_over_undefended_real_box": ex_real,
                    "excess_over_undefended_five_own_units": ex_own,
                    "share_of_excess_removed_by_own_units": share,
                    "scale_independent": bool(real == own),
                    "mostly_scale_effect": bool(share is not None and share > 0.5)})
    return {"l2_extent": l2, "evasion_reference": ref}


def degenerate_training_value(dnames: list) -> dict:
    """The value each of the five constant-in-training features takes: Lab 1's fitted scaler
    mean (its variance is 0, so the mean is that constant). Reads only the small model file
    and the Parquet schema for the feature order, so it needs no data and no GPU."""
    import pyarrow.parquet as pq
    from detector.dataset import LEAKAGE_COLUMNS, LABEL_COL, WEDNESDAY_PARQUET
    from detector.torch_model import LAB1_MLP
    names = [c for c in pq.read_schema(WEDNESDAY_PARQUET).names
             if c not in LEAKAGE_COLUMNS and c != LABEL_COL]
    sc = joblib.load(LAB1_MLP).named_steps["scaler"]
    assert len(names) == len(sc.mean_) and float(np.max(sc.var_[[names.index(n) for n in dnames]])) == 0.0
    return {n: float(sc.mean_[names.index(n)]) for n in dnames}


def augment() -> int:
    res = json.loads(OUT.read_text())
    b1 = json.loads((ROOT / "results" / "task1b_evasion_wednesday.json").read_text())
    df = json.loads((ROOT / "results" / "task1c_defense.json").read_text())
    res.update(derived_blocks(res, b1, df))
    res["degenerate_training_value"] = degenerate_training_value(res["degenerate_mutable_features"])
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["l2_extent"], indent=1))
    print("training mean (value of the constant) of the five:", res["degenerate_training_value"])
    for r in res["evasion_reference"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", OUT)
    return 0


def stats(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def main() -> int:
    if "--augment" in sys.argv[1:]:
        return augment()
    split = build_split()
    names = split.feature_names
    base = joblib.load(LAB1_MLP)
    defd = joblib.load(ROOT / "models" / "task1c_defended.joblib")
    mutable = classify(names)
    sb = base.named_steps["scaler"].scale_
    sd = defd.named_steps["scaler"].scale_
    degen = (np.std(split.X_train, axis=0) == 0) & (sb == 1.0)
    dg = np.where(degen & mutable)[0]
    rl = np.where(mutable & ~degen)[0]
    assert len(dg) == 5 and len(rl) == 47, (len(dg), len(rl))
    X = evaluation_flows(split, base, n=300)

    # ---- a. standardized move seen by each pipeline under the REAL attack box
    box = raw_radius(sb, EPS, mutable.astype(float))      # the repo's own function, ORIGINAL scaler

    def seen(model):
        sc = model.named_steps["scaler"]
        d = sc.transform(X + box[None, :]) - sc.transform(X)   # all mutable features at once
        return d                                              # (300, 77)

    d_def, d_base = seen(defd), seen(base)
    mv_def = d_def.mean(axis=0)
    mv_base = d_base.mean(axis=0)
    spread_def = float((d_def.max(axis=0) - d_def.min(axis=0))[mutable].max())
    assert np.allclose(d_base[:, mutable], EPS, rtol=0, atol=1e-6), "undefended move != eps"
    ratio = sb / sd
    per_feature = {names[j]: {"sb": float(sb[j]), "sd": float(sd[j]), "sb_over_sd": float(ratio[j]),
                              "std_move_defended": float(mv_def[j]),
                              "std_move_undefended": float(mv_base[j]),
                              "kind": "degenerate" if degen[j] else "real"}
                   for j in np.where(mutable)[0]}
    cancels_degen = bool(np.allclose(mv_def[dg], EPS, rtol=1e-4))
    cancels_real = bool(np.allclose(mv_def[rl], EPS, rtol=1e-4))

    # ---- b/c. logit sensitivity, one feature at a time, upper (and lower) box corner
    Xt = torch.tensor(X, dtype=torch.float32)

    def sens(model, delta):
        t = to_torch(model)
        with torch.no_grad():
            l0 = t.logit(Xt)
            up = np.zeros(len(names)); sg = np.zeros(len(names)); fneg = np.zeros(len(names))
            lo_sg = np.zeros(len(names))
            for j in np.where(mutable)[0]:
                x = Xt.clone(); x[:, j] = x[:, j] + float(delta[j])
                dl = t.logit(x) - l0
                up[j] = float(dl.abs().mean()); sg[j] = float(dl.mean())
                fneg[j] = float((dl < 0).float().mean())
                x = Xt.clone(); x[:, j] = x[:, j] - float(delta[j])
                lo_sg[j] = float((t.logit(x) - l0).mean())
        return up, sg, fneg, lo_sg

    sens_def_i = sens(defd, EPS * sd)       # (i) defended model's own units
    sens_def_ii = sens(defd, EPS * sb)      # (ii) real attack box
    sens_base_i = sens(base, EPS * sb)      # undefended: its own scale == the box
    sens_base_ii = sens(base, EPS * sb)
    for a, b in zip(sens_base_i, sens_base_ii):
        assert np.array_equal(a, b), "undefended (i) != (ii)"
    assert (sens_base_i[0][dg] == 0).all(), "undefended sensitivity on the five is not exactly 0"

    def summ(s):
        up, sg, fneg, lo = s
        top5 = np.sort(up[rl])[::-1][:5]
        return {"degenerate_by_feature": {names[j]: float(up[j]) for j in dg},
                "degenerate_min": float(up[dg].min()), "degenerate_max": float(up[dg].max()),
                "real_median": float(np.median(up[rl])), "real_top5_median": float(np.median(top5)),
                "signed_mean_upper_by_feature": {names[j]: float(sg[j]) for j in dg},
                "frac_flows_logit_decreases_upper_by_feature": {names[j]: float(fneg[j]) for j in dg},
                "signed_mean_lower_by_feature": {names[j]: float(lo[j]) for j in dg},
                "real_signed_mean_upper_median": float(np.median(sg[rl])),
                "n_degenerate_upper_decreases_logit": int((sg[dg] < 0).sum()),
                "n_degenerate_lower_decreases_logit": int((lo[dg] < 0).sum())}

    res = {
        "day": "Wednesday", "eps_std": EPS, "n_flows": int(len(X)),
        "definitions": {
            "box": "detector.evasion.raw_radius(sb, 0.15, mutable): the attack box of every "
                   "defended curve (defend.py, ablate_1c.py, smoothing_eval.py), ORIGINAL scaler",
            "std_move": "mean over the 300 evaluation flows of "
                        "scaler.transform(x+box)[j] - scaler.transform(x)[j], the pipeline's own "
                        "scaler, all mutable features moved at once",
            "sensitivity_i": "mean |logit(x_j + 0.15*sd_j) - logit(x)| (defended model's own units)",
            "sensitivity_ii": "same with x_j + 0.15*sb_j (the real attack box); the undefended "
                              "model's box is its own scale, so (i)==(ii) for it (asserted)",
            "degenerate": "the five mutable features constant in Lab 1 training",
            "real": "the 47 mutable features not constant in training"},
        "degenerate_mutable_features": [names[j] for j in dg],
        "per_feature": per_feature,
        "summary_degenerate": {"sb": stats(sb[dg]), "sd": stats(sd[dg]), "sb_over_sd": stats(ratio[dg]),
                               "std_move_defended": stats(mv_def[dg])},
        "summary_real": {"n": int(len(rl)), "sb": stats(sb[rl]), "sd": stats(sd[rl]),
                         "sb_over_sd": stats(ratio[rl]), "std_move_defended": stats(mv_def[rl])},
        "std_move_across_flows_spread_max": spread_def,
        "scale_cancels_on_degenerate": cancels_degen,
        "scale_cancels_on_real": cancels_real,
        "undefended_std_move_equals_eps": True,
        "sensitivity_defended_i_own_units": summ(sens_def_i),
        "sensitivity_defended_ii_real_box": summ(sens_def_ii),
        "sensitivity_undefended": summ(sens_base_i),
        "undefended_i_equals_ii": True,
        "verdict": ("CANCELS" if cancels_degen else "DOES NOT CANCEL")
                   + " on the five degenerate features under the real attack box",
    }
    # ---- d. attack measurement: does the 8x larger standardized move matter to the attack?
    # Per-row PGD (detector.evasion.attack_pgd, as in report_extras.py restricted_pgd) against
    # the shipped defended model. "real box": radius eps*sb_j on every allowed feature.
    # "own-units box": the five degenerate features get radius eps*sd_j instead (so the
    # defended model sees a standardized move of exactly eps on them); the rest is unchanged.
    t_def = to_torch(defd)
    mut_f = mutable.astype(float)
    five = np.zeros(len(names)); five[dg] = 1.0
    res["pgd_scale_test"] = {"n_flows": int(len(X)), "curve": []}
    for e in (0.15, 0.25):
        r_real_all = raw_radius(sb, e, mut_f)
        r_own_all = r_real_all.copy(); r_own_all[dg] = e * sd[dg]
        cfgs = {"five_real_box": r_real_all * five, "five_own_units_box": r_own_all * five,
                "all_mutable_real_box": r_real_all, "all_mutable_five_own_units": r_own_all}
        row = {"eps_std": e}
        for g, rad in cfgs.items():
            ev = 0
            for xi in X:
                xa, _ = attack_pgd(t_def, xi, rad, feature_names=names)
                ev += int(defd.predict_proba(xa[None, :])[0][1] < 0.5)
            row[g] = {"evasion_rate": ev / len(X), "n_evaded": ev}
            print(f"pgd eps={e} {g}: evasion {ev / len(X):.3f}", flush=True)
        res["pgd_scale_test"]["curve"].append(row)
    res.update(derived_blocks(res, json.loads((ROOT / "results" / "task1b_evasion_wednesday.json").read_text()),
                              json.loads((ROOT / "results" / "task1c_defense.json").read_text())))
    OUT.write_text(json.dumps(res, indent=2))
    sdg, srl = res["summary_degenerate"], res["summary_real"]
    print("degenerate sb/sd:", sdg["sb_over_sd"], " std move seen by defended:", sdg["std_move_defended"])
    print("real       sb/sd:", srl["sb_over_sd"], " std move seen by defended:", srl["std_move_defended"])
    for k in ("sensitivity_defended_i_own_units", "sensitivity_defended_ii_real_box", "sensitivity_undefended"):
        s = res[k]
        print(k, "deg min/max", round(s["degenerate_min"], 4), round(s["degenerate_max"], 4),
              "real median", round(s["real_median"], 4), "top5 median", round(s["real_top5_median"], 4))
        print("   signed mean upper (five):", {a: round(b, 4) for a, b in s["signed_mean_upper_by_feature"].items()})
        print("   signed mean lower (five):", {a: round(b, 4) for a, b in s["signed_mean_lower_by_feature"].items()})
        print("   frac flows logit decreases (upper):", {a: round(b, 3) for a, b in s["frac_flows_logit_decreases_upper_by_feature"].items()})
    print("VERDICT:", res["verdict"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
