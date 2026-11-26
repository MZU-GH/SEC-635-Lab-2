"""Task 1 extras (follow-up experiments): three small experiments on the EXISTING attack/defense code.

Blocks (run one per process; each writes a part file, `assemble` merges them):

  harness_check   re-runs the 1B flow on Wednesday through this module's curve driver and
                  compares every cell with results/task1b_evasion_wednesday.json (proves the
                  driver used for the two new directions equals the original loop).
  thursday        1B (random search, FD-FGSM, PGD-40, 1B eps grid) on the Thursday model/data.
  benign          the same three attacks pushing benign Wednesday flows toward "attack".
  recommended     retrain the 1C recommended config (1 round, 10 constant features frozen, with
                  detector.defend.adversarial_train exactly as detector/ablate_1c_rounds.py),
                  verify it against results/task1c_ablation_rounds.json, then run the 1A
                  membership re-run (same code path as detector/membership_after.py) and
                  randomized smoothing (same recipe as detector/smoothing_eval.py).
  assemble        merge run-1 parts, compare with run-2 parts, add input sha256s.

Nothing existing is edited; attack code is imported from detector/evasion.py. For the
benign->attack direction the same attack functions are driven through a score-flipped view
of the detector (FlippedScorer / FlippedTorch), so the objective is reversed and nothing else
changes. Run on the GPU server mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import THURSDAY_PARQUET, WEDNESDAY_PARQUET, build_split  # noqa: E402
from detector.evasion import (CONFIDENT, EPS_GRID, SEED, Budget, Scorer,  # noqa: E402
                              attack_fd_fgsm, attack_pgd, attack_random,
                              evaluation_flows, raw_radius)
from detector.features import classify  # noqa: E402

WED_MODEL = ROOT / "lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib"
THU_MODEL = ROOT / "lab1/outputs/models/task1_thursday_mlp_neuralnet_temporal.joblib"
METHODS = ("random_search", "fd_fgsm", "pgd")


def rate(k: int, n: int) -> dict:
    return {"count": int(k), "n": int(n), "fraction": f"{int(k)}/{int(n)}",
            "value": (k / n) if n else None}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_matrix(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# reversed-objective views, so the unchanged attacks push benign flows toward "attack"
# --------------------------------------------------------------------------------------
class FlippedScorer(Scorer):
    """Black-box view whose 'detected' (score < 0.5) means the real detector says attack.

    p()      = 1 - P(attack)               (random search succeeds when this drops below 0.5)
    margin() = -logit(P(attack))           (FD-FGSM descends this = ascends the real logit)
    The real logit is computed from P(attack) directly with a 1e-300 clip, not 1e-12 as in
    Scorer.margin: benign flows sit near P ~ 1e-11, and clipping at 1e-12 would zero the
    finite-difference gradient and understate the attack.
    """

    def p(self, X):
        return 1.0 - super().p(X)

    def margin(self, X):
        self.calls += 1
        self.rows += len(X)
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-300, 1 - 1e-12)
        return -(np.log(p) - np.log1p(-p))


class FlippedTorch:
    """attack_pgd only calls .logit(); minimising the negated logit raises P(attack)."""

    def __init__(self, m):
        self.m = m

    def logit(self, x):
        return -self.m.logit(x)


def attack_curve(sk, torch_m, X, scale, mutable, split, flip: bool, eps_grid, log):
    """Mirror of the loop in detector/evasion.py::main (same rng use, same call order).

    flip=False: count flows still detected (P>=0.5) after the attack.
    flip=True : X are benign flows; count flows scored >= 0.5 (false alarms) after the attack.
    """
    rng = np.random.default_rng(SEED)
    scorer = (FlippedScorer if flip else Scorer)(sk, split)
    tmodel = FlippedTorch(torch_m) if flip else torch_m
    curve = []
    for eps in eps_grid:
        rad = raw_radius(scale, eps, mutable)
        cell = {"eps_std": eps, "methods": {}}
        for name in METHODS:
            t0 = time.time()
            budget = Budget()
            hits = 0  # detected (forward) / false alarm (flip)
            for xi in X:
                if eps == 0.0:
                    x_adv, b = xi.copy(), Budget()
                elif name == "random_search":
                    x_adv, b = attack_random(scorer, xi, rad, rng, tries=200)
                elif name == "fd_fgsm":
                    x_adv, b = attack_fd_fgsm(scorer, xi, rad)
                else:
                    x_adv, b = attack_pgd(tmodel, xi, rad, feature_names=split.feature_names)
                budget.model_calls += b.model_calls
                budget.gradient_computations += b.gradient_computations
                pa = sk.predict_proba(x_adv[None, :])[0][1]
                hits += int(pa >= 0.5)
            cell["methods"][name] = {
                ("false_alarm" if flip else "detection"): rate(hits, len(X)),
                "budget": {"model_calls": budget.model_calls,
                           "gradient_computations": budget.gradient_computations},
                "seconds": round(time.time() - t0, 1)}
        curve.append(cell)
        log(f"eps={eps:5.3f} " + "  ".join(
            f"{k}={v[('false_alarm' if flip else 'detection')]['fraction']}"
            for k, v in cell["methods"].items()))
        sys.stdout.flush()
    return curve


def torch_equivalence(sk, torch_m, X):
    import torch
    with torch.no_grad():
        pt = torch_m(torch.tensor(X, dtype=torch.float32)).numpy()
    return float(np.max(np.abs(pt - sk.predict_proba(X)[:, 1])))


# --------------------------------------------------------------------------------------
def block_harness_check(args, log):
    import joblib
    from detector.defend import to_torch
    split = build_split()
    sk = joblib.load(WED_MODEL)
    X = evaluation_flows(split, sk, n=300)
    mutable = classify(split.feature_names)
    curve = attack_curve(sk, to_torch(sk), X, sk.named_steps["scaler"].scale_, mutable,
                         split, False, EPS_GRID, log)
    stored = json.loads((ROOT / "results/task1b_evasion_wednesday.json").read_text())
    cells, ok = [], True
    for row, srow in zip(curve, stored["curve"]):
        assert row["eps_std"] == srow["eps_std"]
        for m in METHODS:
            d = row["methods"][m]["detection"]
            want = srow["attacks"][m]["detection_rate"]
            # stored value is 1 - evaded/n (float); compare the exact count it implies
            want_count = int(round(want * d["n"]))
            eq = (d["count"] == want_count and abs(want_count / d["n"] - want) < 1e-12)
            ok &= eq
            cells.append({"eps_std": row["eps_std"], "method": m, "mine": d["fraction"],
                          "stored_rate": want, "stored_implied_count": want_count,
                          "equal": eq})
    return {"method": "Wednesday 1B flow re-run through attack_curve(); every cell compared "
                      "with results/task1b_evasion_wednesday.json (stored rate must equal count/n to 1e-12)",
            "n_flows": int(len(X)), "all_equal": bool(ok), "cells": cells, "curve": curve}


def block_thursday(args, log):
    import joblib
    from detector.defend import to_torch
    split = build_split(THURSDAY_PARQUET, "Thursday")
    wed_names = build_split().feature_names
    assert split.feature_names == wed_names, "Thursday/Wednesday feature order differs"
    sk = joblib.load(THU_MODEL)
    mutable = classify(split.feature_names)
    scale = sk.named_steps["scaler"].scale_
    tm = to_torch(sk)
    attack_rows = np.where(split.y_test == 1)[0]
    p_att = sk.predict_proba(split.X_test[attack_rows])[:, 1]
    pool = attack_rows[p_att > CONFIDENT]
    # variant A: the 1A/1B/1C convention (draw 300 attack test flows, then keep P>0.9)
    XA = evaluation_flows(split, sk, n=300)
    # variant B: filter the whole attack test set to P>0.9 first, then draw 300 (all if fewer)
    rng = np.random.default_rng(SEED)
    selB = rng.choice(pool, size=min(300, len(pool)), replace=False)
    XB = split.X_test[selB]
    log(f"Thursday: A={len(XA)} flows; pool P>{CONFIDENT}={len(pool)} of {len(attack_rows)}; B={len(XB)}")
    out = {
        "method": "1B on the Thursday model/data: random search (200 tries), FD-FGSM, PGD-40 "
                  "exactly as detector/evasion.py (imported); eps in the Thursday model's own "
                  "scaler units; same 52-feature mutable mask; detection = still scored >= 0.5; "
                  "black-box attacks in-process. Variant A uses evaluation_flows(Thursday split, "
                  "Thursday model, n=300, seed=42) (draw 300 attack test flows, keep P>0.9: the "
                  "1B/1C convention). Variant B filters all attack test flows to P>0.9 first and "
                  "draws 300 from that pool with default_rng(42) (all of them if fewer than 300)",
        "seed": SEED, "eps_grid": EPS_GRID, "n_mutable_features": int(mutable.sum()),
        "n_attack_test_flows": int(len(attack_rows)),
        "n_attack_test_flows_with_P_gt_0.9": rate(len(pool), len(attack_rows)),
        "n_attack_test_flows_with_P_ge_0.5": rate(int((p_att >= 0.5).sum()), len(attack_rows)),
        "variants": {}}
    for tag, X in (("A_sample300_then_filter", XA), ("B_filter_then_sample300", XB)):
        eq = torch_equivalence(sk, tm, X)
        curve = attack_curve(sk, tm, X, scale, mutable, split, False, EPS_GRID, log)
        out["variants"][tag] = {"n_flows": int(len(X)),
                                "torch_vs_sklearn_max_abs_prob_diff": eq,
                                "flow_matrix_sha256": sha_matrix(X), "curve": curve}
    return out


def block_benign(args, log):
    import joblib
    from detector.defend import to_torch
    split = build_split()
    sk = joblib.load(WED_MODEL)
    mutable = classify(split.feature_names)
    benign_rows = np.where(split.y_test == 0)[0]
    p_all = sk.predict_proba(split.X_test[benign_rows])[:, 1]
    pool = benign_rows[p_all < 0.1]
    rng = np.random.default_rng(SEED)
    sel = rng.choice(pool, size=min(300, len(pool)), replace=False)
    X = split.X_test[sel]
    p0 = sk.predict_proba(X)[:, 1]
    log(f"benign pool P<0.1: {len(pool)} of {len(benign_rows)}; selected {len(X)}")
    tm = to_torch(sk)
    eq = torch_equivalence(sk, tm, X)
    curve = attack_curve(sk, tm, X, sk.named_steps["scaler"].scale_, mutable, split, True,
                         EPS_GRID, log)
    return {
        "method": "1B reversed on Wednesday benign flows: pool = benign test flows with "
                  "P(attack)<0.1 under the Lab 1 model; 300 drawn with default_rng(42).choice "
                  "(no replacement); the unchanged evasion.py attacks are driven through a "
                  "score-flipped view (maximise P(attack)); eps in the Lab 1 scaler units; same "
                  "52-feature mutable mask; false alarm = P(attack) >= 0.5 after the attack",
        "seed": SEED, "eps_grid": EPS_GRID, "n_flows": int(len(X)),
        "pool_size": int(len(pool)), "n_benign_test_flows": int(len(benign_rows)),
        "n_mutable_features": int(mutable.sum()),
        "clean_false_alarm_before_attack": rate(int((p0 >= 0.5).sum()), len(X)),
        "max_clean_P_attack": float(p0.max()),
        "torch_vs_sklearn_max_abs_prob_diff": eq,
        "flow_matrix_sha256": sha_matrix(X), "curve": curve}


# --------------------------------------------------------------------------------------
def smoothing_cells(model, base, X, split, mask, sigma, n, alpha, eps_list, only_idx, log):
    """Copy of detector/smoothing_eval.py's procedure (that file is a script, not importable).

    Per-cell seeds, sources, attack call signatures, vote loop and abstain rule are identical;
    `only_idx` restricts to some positions of eps_list while keeping the original seed index.
    """
    import torch
    from scipy.stats import beta, norm
    from detector.defend import to_torch
    from detector.torch_model import load_lab1_mlp
    names = split.feature_names
    scale_orig = base.named_steps["scaler"].scale_
    sigma_raw = sigma * model.named_steps["scaler"].scale_
    N = n

    def votes_for(Xr, rng, batch=200):
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
        p_lo = np.array([0.0 if k == 0 else min(float(beta.ppf(alpha / 2, k, N - k + 1)),
                                                1 - 1e-12) for k in k_hi])
        abstain = (sigma * norm.ppf(p_lo)) <= 0
        det = top_attack & ~abstain
        nn_ = len(v)
        return {"detected": rate(int(det.sum()), nn_), "abstain": rate(int(abstain.sum()), nn_),
                "flagged_fail_closed": rate(int((det | abstain).sum()), nn_),
                "majority_attack": rate(int(top_attack.sum()), nn_),
                "detection_rate": float(det.mean()), "abstain_rate": float(abstain.mean()),
                "median_attack_vote_share": float(np.median(v / N))}

    class SmoothedLogit:
        def __init__(self, base_t, m, seed):
            self.b, self.m = base_t, m
            self.g = torch.Generator().manual_seed(seed)
            self.sig = torch.tensor(sigma_raw, dtype=torch.float32)

        def logit(self, x):
            z = torch.randn((self.m, x.shape[-1]), generator=self.g) * self.sig
            return self.b.logit(x + z).mean()

    t_def, t_orig = to_torch(model), load_lab1_mlp()
    sc_def, sc_orig = Scorer(model, split), Scorer(base, split)
    cells = []
    for ei, eps in enumerate(eps_list):
        if ei not in only_idx:
            continue
        rad = raw_radius(scale_orig, eps, mask)
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
                    sm = SmoothedLogit(t_def, 32, SEED + 1000 + ei)
                    Xa = np.stack([attack_pgd(sm, x, rad, feature_names=names)[0] for x in X])
                else:
                    Xa = np.stack([fn(x) for x in X])
                rng = np.random.default_rng(np.random.SeedSequence([SEED, ei, si, ai]))
                r = summarize(votes_for(Xa, rng))
                u = int((model.predict_proba(Xa)[:, 1] >= 0.5).sum())
                r["unsmoothed_detected"] = rate(u, len(Xa))
                r["unsmoothed_detection_rate"] = u / len(Xa)
                cell["sources"][src][an] = r
                log(f"  smooth eps={eps:5.3f} {src:14s} {an:16s} det={r['detected']['fraction']} "
                    f"abstain={r['abstain']['fraction']} unsmoothed={r['unsmoothed_detected']['fraction']}")
                sys.stdout.flush()
                if eps == 0.0:
                    break
            if eps == 0.0:
                break
        cells.append(cell)
    return cells


def block_recommended(args, log):
    import joblib
    import requests
    from scipy.stats import beta  # noqa: F401  (import check)
    from detector.ablate_1c import detection_curve
    from detector.defend import AT_EPS_STD, adversarial_train, certify
    from detector.membership import (HttpModel, evaluate, generalisation_gap,
                                     index_provenance, sample_positions, server_stats)

    t_start = time.time()
    split = build_split()
    names = split.feature_names
    base = joblib.load(WED_MODEL)
    mutable = classify(names).astype(float)
    X = evaluation_flows(split, base, n=300)
    train_std = np.std(split.X_train, axis=0)
    base_scale = base.named_steps["scaler"].scale_
    degenerate = (train_std == 0) & (base_scale == 1.0)
    log(f"constant features: {int(degenerate.sum())}: {[names[i] for i in np.where(degenerate)[0]]}")

    # ---- train exactly as ablate_1c.run_config / ablate_1c_rounds.py (full_1r_frozen) ----
    t0 = time.time()
    train_mutable = mutable.copy()
    train_mutable[degenerate] = 0.0
    model, meta = adversarial_train(split, base, names, eps_std=AT_EPS_STD, rounds=1,
                                    n_rows=None, log=log, mutable=train_mutable)
    train_seconds = round(time.time() - t0, 1)
    curve = detection_curve(model, base_scale, train_mutable.copy(), X, split, names)
    clean = float((model.predict(split.X_test) == split.y_test).mean())

    # ---- verify against the stored 1-round frozen run -------------------------------
    stored = json.loads((ROOT / "results/task1c_ablation_rounds.json").read_text())
    sc = next(c for c in stored["configs"] if c["config"] == "full_1r_frozen")
    checks = [{"what": "clean_accuracy", "mine": clean, "stored": sc["clean_accuracy"],
               "equal": clean == sc["clean_accuracy"]}]
    for row, srow in zip(curve, sc["curve_after"]):
        assert row["eps_std"] == srow["eps_std"]
        for a in ("fd_fgsm", "pgd"):
            mine, want = row["attacks"][a]["detection_rate"], srow["attacks"][a]["detection_rate"]
            checks.append({"what": f"{a}_detection_eps_{row['eps_std']}", "mine": mine,
                           "stored": want, "equal": mine == want})
    all_eq = all(c["equal"] for c in checks)
    for c in checks:
        log(f"  repro {c['what']:28s} mine={c['mine']!r} stored={c['stored']!r} equal={c['equal']}")
    out = {
        "method": "full_1r_frozen: detector.defend.adversarial_train(split, Lab1 model, eps_std=0.25, "
                  "rounds=1, all rows, mutable mask with the 10 constant-in-train features frozen) "
                  "-- identical calls to detector/ablate_1c_rounds.py; curve by "
                  "detector.ablate_1c.detection_curve with the frozen mask",
        "stored_reference": "results/task1c_ablation_rounds.json config full_1r_frozen "
                            "(the one-round frozen config is stored there; task1c_ablation.json "
                            "holds only the two-round frozen config)",
        "n_frozen_features": int(degenerate.sum()),
        "frozen_features": [names[i] for i in np.where(degenerate)[0]],
        "adversarial_training": meta, "train_seconds": train_seconds,
        "clean_accuracy": clean,
        "curve_after": curve,
        "reproduction": {"all_equal": bool(all_eq), "checks": checks},
    }
    if not all_eq:
        out["stopped"] = "retrained model does not equal the stored values; nothing else run"
        return out

    # ---- save (determinism is judged by comparing run 1 and run 2 file hashes) ------
    mpath = Path(args.model_out)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, mpath)
    clf = model.named_steps["clf"]
    wsha = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in
                                   [*clf.coefs_, *clf.intercepts_,
                                    model.named_steps["scaler"].mean_,
                                    model.named_steps["scaler"].scale_])).hexdigest()
    out["saved_model"] = {"path": str(mpath), "sha256": sha256_file(mpath), "weights_sha256": wsha}

    # ---- (a) 1A membership re-run, same code path as membership_after.py ------------
    port = args.port
    env = dict(os.environ, SEC635_MODEL=str(mpath))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "detector.api:app", "--host",
                             "127.0.0.1", "--port", str(port)], cwd=str(ROOT), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                health = requests.get(f"{base_url}/health", timeout=3).json()
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("API did not start")
        if health["model"] != mpath.name:
            raise SystemExit(f"API serves {health['model']} not {mpath.name}")
        ref = json.loads((ROOT / "results/task1a_membership.json").read_text())
        wed = next(d for d in ref["days"] if d["day"] == "Wednesday")
        ref_target = next(a for a in wed["arms"] if a["arm"] == "Wednesday_target_model")
        ref_blind = next(a for a in wed["arms"] if a["arm"] == "Wednesday_blind_control")
        after = json.loads((ROOT / "results/task1c_membership_after.json").read_text())
        rng = np.random.default_rng(SEED)
        mem, non, cal = sample_positions(split, rng)
        prov = index_provenance(mem, non, cal)
        repro = evaluate("lab1_reproduction", base, split, mem, non, cal)
        repro_ok = {k: (repro["attacks"][k]["accuracy"] == v["accuracy"]
                        and repro["attacks"][k]["tau"] == v["tau"])
                    for k, v in ref_target["attacks"].items()}
        s0 = server_stats(base_url)
        http = HttpModel(base_url)
        arm = evaluate("Wednesday_recommended_target_model", http, split, mem, non, cal)
        s1 = server_stats(base_url)
        inproc = evaluate("check", model, split, mem, non, cal)
        same = all(arm["attacks"][k]["accuracy"] == inproc["attacks"][k]["accuracy"]
                   for k in arm["attacks"])

        def gap_for(m):
            r = np.random.default_rng(SEED)
            sample_positions(split, r)
            return generalisation_gap(m, split, r)
        gap_rec = gap_for(model)
        for k, v in arm["attacks"].items():
            log(f"  membership recommended {k:22s} acc={v['accuracy']!r} ci={v['ci95']} auc={v['auc']!r}")
        out["membership"] = {
            "method": "same procedure as detector/membership_after.py: positions from "
                      "membership.sample_positions(default_rng(42)); thresholds fitted on the "
                      "disjoint calibration pool; target arm scored over HTTP from a local API "
                      "serving the recommended model; Lab 1 model reproduces the original 1A "
                      "numbers first",
            "seed": SEED, "eval_index_provenance": prov,
            "index_provenance_equals_task1a_json": prov == wed["eval_index_provenance"],
            "lab1_reproduces_original_target_arm": repro_ok,
            "recommended_target_arm": arm["attacks"],
            "http_equals_in_process": same,
            "generalisation_gap_recommended": gap_rec,
            "references": {
                "undefended_lab1_target_arm": ref_target["attacks"],
                "blind_control": ref_blind["attacks"],
                "two_round_defended_target_arm": after["defended_target_arm"]},
            "transport": {"client_requests": http.requests_made, "client_rows": http.rows_sent,
                          "server_requests_delta": s1["requests"] - s0["requests"],
                          "server_rows_delta": s1["rows_scored"] - s0["rows_scored"]},
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    # ---- (b) randomized smoothing, same sigma/n/alpha/certify ------------------------
    sd = json.loads((ROOT / "results/task1c_defense.json").read_text())["smoothing"]
    SIGMA, NS, ALPHA = sd["sigma_std"], sd["n_samples"], sd["alpha"]
    Xte = split.X_test
    idx = np.random.default_rng(SEED).choice(len(Xte), size=200, replace=False)
    rc = certify(model, Xte[idx], sigma=SIGMA, n=NS, scale=model.named_steps["scaler"].scale_,
                 alpha=ALPHA)
    rc["n_abstain_rate"] = rate(rc["n_abstain"], rc["n_certified"])
    log(f"  certify: median radius={rc['radius_median']!r} abstain={rc['n_abstain']}/{rc['n_certified']}")
    eps_list = [0.0, 0.10, 0.15, 0.25]
    frozen_mask = train_mutable.astype(bool)
    full_mask = classify(names)
    sm_frozen = smoothing_cells(model, base, X, split, frozen_mask, SIGMA, NS, ALPHA,
                                eps_list, {0, 1, 2, 3}, log)
    sm_unfrozen = smoothing_cells(model, base, X, split, full_mask, SIGMA, NS, ALPHA,
                                  eps_list, {2, 3}, log)

    # reproduction of my smoothing copy on the stored 2-round defended model
    stored_sm = json.loads((ROOT / "results/task1c_smoothing_detection.json").read_text())
    d2 = joblib.load(ROOT / "models/task1c_defended.joblib")
    sm_chk = smoothing_cells(d2, base, X, split, full_mask, SIGMA, NS, ALPHA, eps_list,
                             {2, 3}, log)
    rep_cells = []
    for cell in sm_chk:
        scell = next(c for c in stored_sm["curve"] if c["eps_std"] == cell["eps_std"])
        for src, atks in cell["sources"].items():
            for an, r in atks.items():
                s = scell["sources"][src][an]
                rep_cells.append({"eps_std": cell["eps_std"], "source": src, "attack": an,
                                  "mine": [r["detected"]["count"], r["abstain"]["count"]],
                                  "stored": [s["n_detected"], s["n_abstain"]],
                                  "equal": [r["detected"]["count"], r["abstain"]["count"]]
                                  == [s["n_detected"], s["n_abstain"]]})
    out["smoothing"] = {
        "method": "randomized smoothing of the recommended model: sigma/n/alpha read from "
                  "results/task1c_defense.json smoothing block; sigma standardized and "
                  "converted with the model's own scale_; certify() on the same 200 test rows "
                  "(default_rng(42).choice); detection under smoothing by the detector/"
                  "smoothing_eval.py procedure (majority vote, Clopper-Pearson abstain rule, "
                  "same SeedSequence seeds, sources defended_base/original_lab1/eot_pgd)",
        "sigma_std": SIGMA, "n_samples": NS, "alpha": ALPHA,
        "certify_200_test_rows": rc,
        "detection_frozen_mask": {
            "attacker_mask": "52 mutable features minus the 10 constant ones (the recommended "
                             "config's threat model; same mask the ablation evaluates it with)",
            "eps_list": eps_list, "cells": sm_frozen},
        "detection_unfrozen_mask": {
            "attacker_mask": "all 52 mutable features including the 5 mutable constant ones "
                             "(no range check on the always-zero features)",
            "eps_list": eps_list, "cells_eps_0.15_0.25": sm_unfrozen},
        "copy_reproduces_stored_two_round_cells": {
            "all_equal": all(c["equal"] for c in rep_cells), "cells": rep_cells},
        "references": {"two_round_defended_stored": {
            "source": "results/task1c_smoothing_detection.json",
            "certify_block": json.loads((ROOT / "results/task1c_defense.json").read_text())["smoothing"],
            "curve": stored_sm["curve"]}},
    }
    # unsmoothed detection on the frozen-mask defended_base examples must equal stored curve
    out["smoothing"]["unsmoothed_equals_stored_curve"] = []
    for cell in sm_frozen:
        if cell["eps_std"] in (0.15, 0.25):
            row = next(r for r in curve if r["eps_std"] == cell["eps_std"])
            for an in ("fd_fgsm", "pgd"):
                out["smoothing"]["unsmoothed_equals_stored_curve"].append({
                    "eps_std": cell["eps_std"], "attack": an,
                    "mine": cell["sources"]["defended_base"][an]["unsmoothed_detected"]["fraction"],
                    "stored_rate": row["attacks"][an]["detection_rate"],
                    "equal": abs(cell["sources"]["defended_base"][an]["unsmoothed_detection_rate"]
                                 - row["attacks"][an]["detection_rate"]) < 1e-12})
    out["seconds_total"] = round(time.time() - t_start, 1)
    return out


# --------------------------------------------------------------------------------------
def strip_seconds(o):
    if isinstance(o, dict):
        return {k: strip_seconds(v) for k, v in o.items()
                if k not in ("seconds", "seconds_total", "train_seconds", "seconds_block", "path")}
    if isinstance(o, list):
        return [strip_seconds(v) for v in o]
    return o


def assemble(args):
    wd = Path(args.workdir)
    out = {"experiment": "SEC-635 Lab 2 Task 1 extras (follow-up experiments)",
           "seeds": {"global": SEED, "attack_flow_selection": "default_rng(42)",
                     "random_search": "default_rng(42), one stream across eps/methods as in evasion.py",
                     "pgd_start": "default_rng(42) inside attack_pgd",
                     "smoothing_noise": "SeedSequence([42, eps_idx, source_idx, attack_idx])",
                     "eot": "torch.Generator seed 42+1000+eps_idx"},
           "script": "scripts/detector_extras.py", "blocks": {}, "determinism": {}}
    names = {"harness_check": "harness_check", "thursday_evasion": "thursday",
             "benign_to_attack": "benign", "recommended_config": "recommended"}
    for key, part in names.items():
        p1, p2 = wd / f"{part}_run1.json", wd / f"{part}_run2.json"
        if not p1.exists():
            continue
        b1 = json.loads(p1.read_text())
        out["blocks"][key] = b1
        if p2.exists():
            b2 = json.loads(p2.read_text())
            out["determinism"][key] = {"run1_equals_run2_ignoring_timings_and_paths":
                                       strip_seconds(b1) == strip_seconds(b2)}
    rec = out["blocks"].get("recommended_config")
    if rec and "saved_model" in rec and (wd / "recommended_run2.json").exists():
        r2 = json.loads((wd / "recommended_run2.json").read_text())
        out["determinism"]["recommended_config"]["saved_file_sha256_equal"] = (
            rec["saved_model"]["sha256"] == r2["saved_model"]["sha256"])
        out["determinism"]["recommended_config"]["weights_sha256_equal"] = (
            rec["saved_model"]["weights_sha256"] == r2["saved_model"]["weights_sha256"])
    files = [WED_MODEL, THU_MODEL, WEDNESDAY_PARQUET, THURSDAY_PARQUET,
             ROOT / "models/task1c_defended.joblib",
             ROOT / "results/task1b_evasion_wednesday.json",
             ROOT / "results/task1a_membership.json",
             ROOT / "results/task1c_ablation.json",
             ROOT / "results/task1c_ablation_rounds.json",
             ROOT / "results/task1c_defense.json",
             ROOT / "results/task1c_membership_after.json",
             ROOT / "results/task1c_smoothing_detection.json"]
    out["input_sha256"] = {str(f.relative_to(ROOT)): sha256_file(f) for f in files}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["harness_check", "thursday", "benign", "recommended",
                                      "assemble"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=str(ROOT / "work_e1"))
    ap.add_argument("--model-out", default=str(ROOT / "models/task1c_recommended.joblib"))
    ap.add_argument("--port", type=int, default=8058)
    args = ap.parse_args()
    if args.block == "assemble":
        assemble(args)
        return 0
    fn = {"harness_check": block_harness_check, "thursday": block_thursday,
          "benign": block_benign, "recommended": block_recommended}[args.block]
    t0 = time.time()
    res = fn(args, print)
    res["seconds_block"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)
    return 3 if res.get("stopped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
