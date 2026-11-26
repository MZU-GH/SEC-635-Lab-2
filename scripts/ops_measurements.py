"""Operational measurements for the report: detector latency, confidence bands, copilot latency.

Writes results/ops_measurements.json with three blocks. Run on the lab GPU server (it loads
the models and the full Wednesday parquet):

    .venv/bin/python -u scripts/ops_measurements.py [--out results/ops_measurements.json]

Block 1  detector_latency   per-flow inference time of the deployed Lab 1 MLP and of the
                            2-round adversarially trained model, batch throughput, the
                            observed Wednesday test-window traffic rate, headroom, and the
                            HTTP /score round trip on localhost.
Block 2  confidence_bands   score distribution of the deployed model on clean test flows and
                            on regenerated 1B evasion flows, per fixed alert band. The 1B
                            flows are regenerated with the existing attack code and seed;
                            detection at 0.5 must equal the stored 1B results exactly, or the
                            script records the mismatch and exits non-zero.
Block 3  copilot_latency    from recorded timings in results/task2_redteam.json only
                            (no model calls). Parts with no recorded timing are null + reason.

Deterministic parts (bands, traffic rate, copilot) are identical between runs; timings vary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import WEDNESDAY_PARQUET, build_split  # noqa: E402
from detector.evasion import (SEED, Scorer, attack_fd_fgsm, attack_pgd,  # noqa: E402
                              evaluation_flows, raw_radius)
from detector.features import classify  # noqa: E402

MACHINE = "lab GPU server (CPU inference)"
LAB1_MODEL = ROOT / "lab1" / "outputs" / "models" / "task1_wednesday_mlp_neuralnet_temporal.joblib"
DEFENDED_MODEL = ROOT / "models" / "task1c_defended.joblib"
EVASION_JSON = ROOT / "results" / "task1b_evasion_wednesday.json"
REDTEAM_JSON = ROOT / "results" / "task2_redteam.json"
SUMMARY_JSON = ROOT / "results" / "task2_summary.json"
DEFENSE_JSON = ROOT / "results" / "task1c_defense.json"

# Band edges are fixed by design (not tuned on these data).
EDGE_AUTO, EDGE_L2, EDGE_L1 = 0.99, 0.5, 0.01
SENSITIVITY_LOWER_EDGES = [0.001, 0.1]
EPS_LIST = [0.15, 0.25]


def sha256_file(p: Path) -> dict:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p.relative_to(ROOT)), "sha256": h.hexdigest()}


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q))


# ---------------------------------------------------------------------------------------
# Block 1
# ---------------------------------------------------------------------------------------
def thread_info() -> dict:
    info = {"os_cpu_count": os.cpu_count(),
            "sched_getaffinity": len(os.sched_getaffinity(0)),
            "env": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS")}}
    try:
        import torch
        info["torch_num_threads"] = torch.get_num_threads()
    except Exception as e:  # pragma: no cover
        info["torch_num_threads"] = f"unavailable: {e}"
    try:
        from threadpoolctl import threadpool_info
        info["blas_threadpools"] = [{k: v for k, v in d.items()
                                     if k in ("internal_api", "num_threads", "version")}
                                    for d in threadpool_info()]
    except Exception as e:  # pragma: no cover
        info["blas_threadpools"] = f"threadpoolctl unavailable: {e}"
    try:
        with open("/proc/cpuinfo") as f:
            names = sorted({l.split(":", 1)[1].strip() for l in f if l.startswith("model name")})
        info["cpu_model"] = names
    except Exception:
        info["cpu_model"] = None
    return info


def time_single(model, rows: np.ndarray, n_warm: int = 200) -> dict:
    """One predict_proba call per row; wall time of the call only."""
    for i in range(n_warm):
        model.predict_proba(rows[i % len(rows)][None, :])
    ts = []
    for i in range(len(rows)):
        x = rows[i][None, :]
        t0 = time.perf_counter_ns()
        model.predict_proba(x)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    return {"n_calls": len(ts), "n_warmup": n_warm,
            "median_ms": float(statistics.median(ts)), "p95_ms": pct(ts, 95),
            "p99_ms": pct(ts, 99), "mean_ms": float(np.mean(ts)),
            "single_flow_throughput_flows_per_s": float(1000.0 / statistics.median(ts))}


def time_batch(model, X: np.ndarray, batch: int = 1000, reps: int = 30) -> dict:
    xb = X[:batch]
    for _ in range(5):
        model.predict_proba(xb)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict_proba(xb)
        ts.append(time.perf_counter() - t0)
    thr = [batch / t for t in ts]
    return {"batch_size": batch, "reps": reps, "median_batch_seconds": float(statistics.median(ts)),
            "throughput_flows_per_s_median": float(statistics.median(thr)),
            "throughput_flows_per_s_min": float(min(thr)),
            "throughput_flows_per_s_max": float(max(thr))}


def traffic_rate(split) -> dict:
    import pandas as pd
    ts = pd.to_datetime(split.timestamps.iloc[len(split.X_train):])
    n_test = int(len(split.X_test))
    assert len(ts) == n_test
    t_min, t_max = ts.min(), ts.max()
    dur = float((t_max - t_min).total_seconds())
    per_sec = ts.dt.floor("s").value_counts()
    # active seconds: seconds that contain at least one flow start
    return {
        "source": "Timestamp column of data/processed/cleaned_wednesday_14-02-2018.parquet, "
                  "rows from the Lab 1 temporal split test partition (last 30% of rows); "
                  "flows/s = test rows / (max Timestamp - min Timestamp)",
        "n_test_rows": n_test,
        "test_window_start": str(t_min), "test_window_end": str(t_max),
        "test_window_seconds": dur,
        "mean_flows_per_s": float(n_test / dur),
        "busiest_second_flows": int(per_sec.max()),
        "n_seconds_with_flows": int(len(per_sec)),
        "mean_flows_per_s_over_active_seconds": float(n_test / len(per_sec)),
    }


def http_timing(rows: np.ndarray, port: int, n_calls: int = 300, n_fresh: int = 200) -> dict:
    import requests
    env = dict(os.environ)
    env.pop("SEC635_MODEL", None)  # default = deployed Lab 1 Wednesday model
    log = open(ROOT / f"ops_uvicorn_{port}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "detector.api:app", "--host", "127.0.0.1",
         "--port", str(port)], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if requests.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("uvicorn did not come up")
        payloads = [{"features": r.tolist()} for r in rows[:max(n_calls, n_fresh)]]
        s = requests.Session()
        for p in payloads[:30]:
            s.post(base + "/score", json=p, timeout=30).raise_for_status()
        ka = []
        for p in payloads[:n_calls]:
            t0 = time.perf_counter_ns()
            r = s.post(base + "/score", json=p, timeout=30)
            ka.append((time.perf_counter_ns() - t0) / 1e6)
            r.raise_for_status()
        fresh = []
        for p in payloads[:n_fresh]:
            t0 = time.perf_counter_ns()
            r = requests.post(base + "/score", json=p, timeout=30)
            fresh.append((time.perf_counter_ns() - t0) / 1e6)
            r.raise_for_status()
        return {
            "endpoint": "POST /score (one flow, 77 features), uvicorn on 127.0.0.1, deployed Lab 1 model",
            "keepalive_session": {"n_calls": len(ka), "n_warmup": 30,
                                  "median_ms": float(statistics.median(ka)), "p95_ms": pct(ka, 95)},
            "new_connection_per_call": {"n_calls": len(fresh),
                                        "median_ms": float(statistics.median(fresh)),
                                        "p95_ms": pct(fresh, 95)},
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def block_detector_latency(split, base, defended, port: int) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(split.X_test), size=1200, replace=False)
    rows = split.X_test[idx]
    X_batch = split.X_test[rng.choice(len(split.X_test), size=1000, replace=False)]
    out = {"machine": MACHINE, "threads": thread_info(),
           "method": "single-flow: time of one sklearn Pipeline.predict_proba call on a (1,77) row "
                     "(scaler + MLP, no HTTP), 1200 distinct random Wednesday test rows (seed 42), "
                     "after 200 warm-up calls; median and p95 over the 1200 timed calls. "
                     "Batch: predict_proba on 1000 rows, 30 repetitions after 5 warm-ups, "
                     "throughput = 1000 / seconds, median over repetitions. Traffic rate and "
                     "headroom from the Timestamp column. HTTP: POST /score to a uvicorn "
                     "started by this script on 127.0.0.1, 300 keep-alive calls (30 warm-ups) "
                     "and 200 new-connection calls, deployed Lab 1 model.",
           "models": {}}
    for name, m in (("deployed_lab1", base), ("adv_trained_2round_1c", defended)):
        s1 = time_single(m, rows)
        b = time_batch(m, X_batch)
        out["models"][name] = {"single_flow": s1, "batch_1000": b}
        # single-thread variant for the thread-setting question
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                out["models"][name]["single_flow_1_thread"] = time_single(m, rows)
                out["models"][name]["batch_1000_1_thread"] = time_batch(m, X_batch)
        except Exception as e:
            out["models"][name]["single_thread_variant"] = f"not run: {e}"
    tr = traffic_rate(split)
    out["traffic"] = tr
    out["headroom"] = {}
    for name, m in out["models"].items():
        thr = m["batch_1000"]["throughput_flows_per_s_median"]
        thr1 = m["single_flow"]["single_flow_throughput_flows_per_s"]
        out["headroom"][name] = {
            "definition": "throughput / traffic rate (mean test-window flows/s)",
            "batch_1000_over_mean_rate": thr / tr["mean_flows_per_s"],
            "single_flow_over_mean_rate": thr1 / tr["mean_flows_per_s"],
            "batch_1000_over_busiest_second": thr / tr["busiest_second_flows"],
            "single_flow_over_busiest_second": thr1 / tr["busiest_second_flows"],
        }
    out["http_score"] = http_timing(rows, port)
    return out


# ---------------------------------------------------------------------------------------
# Block 2
# ---------------------------------------------------------------------------------------
def band_counts(scores: np.ndarray, lower: float = EDGE_L1) -> dict:
    s = np.asarray(scores, dtype=float)
    n = int(len(s))
    c = {"auto_alert": int((s >= EDGE_AUTO).sum()),
         "L2_review": int(((s >= EDGE_L2) & (s < EDGE_AUTO)).sum()),
         "L1_review": int(((s >= lower) & (s < EDGE_L2)).sum()),
         "pass": int((s < lower).sum())}
    assert sum(c.values()) == n
    return {"n": n, "lower_edge": lower, "counts": c,
            "fractions": {k: (v / n if n else None) for k, v in c.items()}}


def block_confidence_bands(split, base) -> tuple[dict, bool]:
    t0 = time.time()
    scale = base.named_steps["scaler"].scale_
    mutable = classify(split.feature_names)
    from detector.torch_model import load_lab1_mlp
    tmodel = load_lab1_mlp()

    p_test = base.predict_proba(split.X_test)[:, 1]
    y = split.y_test
    groups_scores = {"clean_test_benign": p_test[y == 0], "clean_test_attack": p_test[y == 1]}

    X = evaluation_flows(split, base, n=300)
    h = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()[:16]
    stored = json.loads(EVASION_JSON.read_text())
    stored_h = stored["eval_set_provenance"]["sha256_of_flow_matrix"]
    stored_curve = {r["eps_std"]: r["attacks"] for r in stored["curve"]}

    scorer = Scorer(base, split)  # in-process; stored 1B notes HTTP == in-process
    repro = []
    adv_scores, adv_evaded_scores = {}, {}
    for eps in EPS_LIST:
        rad = raw_radius(scale, eps, mutable)
        for atk in ("fd_fgsm", "pgd"):
            advs = []
            for xi in X:
                if atk == "fd_fgsm":
                    xa, _ = attack_fd_fgsm(scorer, xi, rad)
                else:
                    xa, _ = attack_pgd(tmodel, xi, rad, feature_names=split.feature_names)
                advs.append(xa)
            sc = base.predict_proba(np.asarray(advs))[:, 1]
            evaded = int((sc < 0.5).sum())
            det = 1.0 - evaded / len(X)
            ev = evaded / len(X)
            st = stored_curve[eps][atk]
            ok = (det == st["detection_rate"]) and (ev == st["evasion_rate"])
            repro.append({"eps_std": eps, "attack": atk,
                          "detection_rate_recomputed": det, "detection_rate_stored": st["detection_rate"],
                          "evasion_rate_recomputed": ev, "evasion_rate_stored": st["evasion_rate"],
                          "n_evaded_recomputed": evaded, "exact_match": bool(ok)})
            key = f"{atk}_eps{eps}"
            adv_scores[key] = sc
            adv_evaded_scores[key] = sc[sc < 0.5]
            print(f"[repro] {key}: det {det!r} stored {st['detection_rate']!r} "
                  f"evasion {ev!r} stored {st['evasion_rate']!r} match={ok}", flush=True)
    all_ok = bool(all(r["exact_match"] for r in repro)) and (h == stored_h)

    all_groups = {**groups_scores, **adv_scores}
    bands = {k: band_counts(v) for k, v in all_groups.items()}
    sens = {}
    for lo in SENSITIVITY_LOWER_EDGES:
        sens[str(lo)] = {k: band_counts(v, lo) for k, v in all_groups.items()}
    evasion_l1 = {}
    for k, sc in adv_scores.items():
        ev = sc[sc < 0.5]
        e = {"n_evasion_flows": int(len(ev)), "n_total_flows": int(len(sc))}
        for lo in [EDGE_L1] + SENSITIVITY_LOWER_EDGES:
            n_l1 = int(((ev >= lo)).sum())
            e[f"lower_edge_{lo}"] = {"in_L1_review": n_l1, "pass_unreviewed": int(len(ev) - n_l1),
                                     "fraction_of_evasion_flows_in_L1": (n_l1 / len(ev) if len(ev) else None),
                                     "fraction_of_all_300_flows_in_L1": n_l1 / len(sc)}
        evasion_l1[k] = e
    qs = {k: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
          for k, v in all_groups.items()}
    block = {
        "method": "Deployed Lab 1 Wednesday MLP scored in-process. Clean groups: all Wednesday "
                  "temporal-split test rows split by ground-truth label. Evasion groups: the 1B "
                  "evaluation flows (evaluation_flows(split, Lab 1 model, n=300, seed=42), "
                  "P>0.9) re-attacked with the existing attack code (FD-FGSM in-process via "
                  "Scorer, PGD-40 via load_lab1_mlp, same raw_radius/eps) at eps 0.15 and 0.25, "
                  "then scored by the deployed model. Bands: auto-alert >= 0.99; L2 review "
                  "[0.5, 0.99); L1 review [0.01, 0.5); pass < 0.01. 'Evasion flows' are those "
                  "with score < 0.5. Sensitivity moves only the lower edge (L1/pass) to 0.001 "
                  "and 0.1; the L1/L2 edge stays 0.5.",
        "band_edges_fixed_by_design": {"auto_alert_ge": EDGE_AUTO, "L2_review": [EDGE_L2, EDGE_AUTO],
                                       "L1_review": [EDGE_L1, EDGE_L2], "pass_lt": EDGE_L1},
        "input_files": [sha256_file(p) for p in (LAB1_MODEL, WEDNESDAY_PARQUET, EVASION_JSON)],
        "evaluation_flow_matrix_sha256_16": {"recomputed": h, "stored_in_1b_json": stored_h,
                                             "equal": h == stored_h},
        "reproduction_of_1b": {"all_exact_match": all_ok, "cells": repro,
                               "note": "detection/evasion at 0.5 compared with == against "
                                       "results/task1b_evasion_wednesday.json"},
        "bands": bands,
        "score_min_median_max": qs,
        "evasion_flows_in_L1_review": evasion_l1,
        "sensitivity_lower_edge": sens,
        "seconds": round(time.time() - t0, 1),
    }
    return block, all_ok


# ---------------------------------------------------------------------------------------
# Block 3
# ---------------------------------------------------------------------------------------
def block_copilot_latency() -> dict:
    r = json.loads(REDTEAM_JSON.read_text())
    summ = json.loads(SUMMARY_JSON.read_text())
    inputs = [sha256_file(REDTEAM_JSON), sha256_file(SUMMARY_JSON)]
    # (a) benign questions
    b_keys = set()
    for arm in ("before", "after"):
        for q in r["benign"][arm]["per_question"]:
            b_keys |= {k for k in q if k in ("seconds", "latency", "elapsed")}
            for s in q["samples"]:
                b_keys |= {k for k in s if k in ("seconds", "latency", "elapsed")}
    a = None
    a_reason = None
    if not b_keys:
        a_reason = ("results/task2_redteam.json records no timing for the 5 benign questions "
                    "(per_question entries and samples carry only counts, text and retrieved ids); "
                    "the only timings are one 'seconds' value per attack block")
    # (b) attack blocks answered in both arms
    before = {x["key"]: x for x in r["before"]}
    after = {x["key"]: x for x in r["after"]}
    rows = {}
    for k in before:
        b, af = before[k], after[k]
        n_b = b["runs"] - b["errors"]
        n_a = af["runs"] - af["errors"]
        called_b = n_b  # unguarded arm always calls the model
        called_a = n_a - af["blocked_by_input"]  # trials that reached the model in the guarded arm
        rows[k] = {
            "before_block_seconds": b["seconds"], "after_block_seconds": af["seconds"],
            "n_before": n_b, "n_after": n_a,
            "after_blocked_by_input": af["blocked_by_input"],
            "after_blocked_by_output": af["blocked_by_output"],
            "after_trials_reaching_model": called_a,
            "before_seconds_per_trial_mean": b["seconds"] / n_b,
            "after_seconds_per_trial_mean": af["seconds"] / n_a,
        }
    # an attack block counts as "answered in both arms" if no trial was stopped before the model
    model_called = [k for k, v in rows.items()
                    if v["after_trials_reaching_model"] == v["n_after"]]
    released = [k for k in model_called if rows[k]["after_blocked_by_output"] == 0]

    def pooled(keys):
        if not keys:
            return None
        sb = sum(rows[k]["before_block_seconds"] for k in keys)
        sa = sum(rows[k]["after_block_seconds"] for k in keys)
        nb = sum(rows[k]["n_before"] for k in keys)
        na = sum(rows[k]["n_after"] for k in keys)
        per_b = [rows[k]["before_seconds_per_trial_mean"] for k in keys]
        per_a = [rows[k]["after_seconds_per_trial_mean"] for k in keys]
        return {"attacks": keys, "n_trials_before": nb, "n_trials_after": na,
                "pooled_mean_seconds_per_trial_before": sb / nb,
                "pooled_mean_seconds_per_trial_after": sa / na,
                "ratio_after_over_before": (sa / na) / (sb / nb),
                "per_attack_mean_seconds_per_trial_before": dict(zip(keys, per_b)),
                "per_attack_mean_seconds_per_trial_after": dict(zip(keys, per_a)),
                "median_of_attack_means_before": float(statistics.median(per_b)),
                "median_of_attack_means_after": float(statistics.median(per_a))}

    b_block = {
        "method": "The only recorded timing is one wall-clock 'seconds' value per attack block of "
                  "10 trials per arm (rounded to 0.1 s), so this is a per-trial MEAN (block "
                  "seconds / trials), not a per-call median or p95 (null: per-call timings were "
                  "not recorded). Attacks counted are those where no guarded-arm trial was "
                  "stopped before the model was called; the unguarded arm always calls the model.",
        "median_per_call": None, "p95_per_call": None,
        "median_p95_reason": "per-call timings are not recorded in results/task2_redteam.json",
        "model_called_in_both_arms_all_trials": pooled(model_called),
        "answer_released_in_both_arms_all_trials": pooled(released),
        "note_on_difference": "attacks in the first set but not the second had guarded-arm trials "
                              "where the model answered and the output guard then withheld the "
                              "answer: " + json.dumps({k: rows[k]["after_blocked_by_output"]
                                                       for k in model_called
                                                       if k not in released}),
        "per_attack_all_nine": rows,
    }
    c_reason = ("guard time is not recorded separately: results/task2_redteam.json has one "
                "'seconds' per attack block covering the whole pipeline (input filter, input "
                "guard, retrieval, model, output guard) and no per-stage timings. "
                "The only inferable bound is for input-blocked trials, where no model call "
                "happens: whole-pipeline per-trial mean for those attack blocks is given below.")
    inblock = [k for k, v in rows.items() if v["after_blocked_by_input"] == v["n_after"]]
    c_block = {"share_of_guarded_latency_in_guard_calls": None, "reason": c_reason,
               "input_blocked_blocks_after_arm_seconds_per_trial_mean":
                   {k: rows[k]["after_seconds_per_trial_mean"] for k in inblock}}
    return {
        "method": "Recorded timings only, no model calls. results/task2_summary.json "
                  "(latency block) summarises results/task2_redteam.json; this block "
                  "recomputes from the raw run file and checks that the summary's "
                  "per_attack_seconds_per_trial equal the recomputed means.",
        "input_files": inputs,
        "summary_consistency": {k: {"summary_before": v["before"], "summary_after": v["after"],
                                    "recomputed_before": rows[k]["before_seconds_per_trial_mean"],
                                    "recomputed_after": rows[k]["after_seconds_per_trial_mean"]}
                                for k, v in summ["latency"]["per_attack_seconds_per_trial"].items()},
        "a_normal_questions_unguarded_vs_guarded": {"value": a, "reason": a_reason},
        "b_attack_trials_answered_in_both_arms": b_block,
        "c_guard_share_of_guarded_latency": c_block,
    }


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ops_measurements.json"))
    ap.add_argument("--port", type=int, default=8073)
    args = ap.parse_args()

    import joblib
    split = build_split()
    base = joblib.load(LAB1_MODEL)
    defended = joblib.load(DEFENDED_MODEL)
    d = json.loads(DEFENSE_JSON.read_text())

    t0 = time.time()
    bands, ok = block_confidence_bands(split, base)
    lat = block_detector_latency(split, base, defended, args.port)
    lat["input_files"] = [sha256_file(p) for p in (LAB1_MODEL, DEFENDED_MODEL, WEDNESDAY_PARQUET,
                                                    DEFENSE_JSON)]
    lat["adv_trained_model_note"] = (
        "models/task1c_defended.joblib, the cached 1C refit written by detector/defend.py; "
        f"results/task1c_defense.json records adversarial_training.rounds = "
        f"{d['adversarial_training']['rounds']}")
    cop = block_copilot_latency()
    payload = {
        "generated_by": "scripts/ops_measurements.py",
        "machine": MACHINE,
        "python": platform.python_version(),
        "versions": {m: __import__(m).__version__ for m in ("numpy", "sklearn", "torch")},
        "detector_latency": lat,
        "confidence_bands": bands,
        "copilot_latency": cop,
        "total_seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}; 1B reproduction exact: {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
