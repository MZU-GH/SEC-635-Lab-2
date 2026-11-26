"""Task 1C requirement: re-run Task 1A (membership inference) against the DEFENDED model.

Black-box: the target arm scores through detector/api.py started with
SEC635_MODEL=models/task1c_defended.joblib. The member / non-member / calibration
positions are the ORIGINAL 1A ones: same seeded generator, same code path
(`membership.sample_positions`), and the original Lab 1 Wednesday accuracies are
reproduced from that path first as proof.

Members are the original 731,322-style clean training rows (the defended model's training
set is those clean rows PLUS adversarial copies; the adversarial rows exist only inside its
training set and are not scored here). The blind-control reference is read from
results/task1a_membership.json, not recomputed.

Run: SEC635_MODEL=models/task1c_defended.joblib uvicorn detector.api:app --port 8056 &
     .venv/bin/python -u detector/membership_after.py --base-url http://127.0.0.1:8056
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import WEDNESDAY_PARQUET, build_split  # noqa: E402
from detector.membership import (SEED, HttpModel, evaluate, generalisation_gap,  # noqa: E402
                                 index_provenance, sample_positions, server_stats)

LAB1 = ROOT / "lab1" / "outputs" / "models" / "task1_wednesday_mlp_neuralnet_temporal.joblib"
DEFENDED = ROOT / "models" / "task1c_defended.joblib"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    import joblib
    import requests

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True,
                    help="API serving models/task1c_defended.joblib")
    ap.add_argument("--out", default=str(ROOT / "results" / "task1c_membership_after.json"))
    args = ap.parse_args()

    health = requests.get(f"{args.base_url}/health", timeout=30).json()
    print("server health:", health)
    if health["model"] != DEFENDED.name:
        raise SystemExit(f"API is serving {health['model']}, not {DEFENDED.name}")

    split = build_split(WEDNESDAY_PARQUET, "Wednesday")
    lab1 = joblib.load(LAB1)
    defended = joblib.load(DEFENDED)
    ref = json.loads((ROOT / "results" / "task1a_membership.json").read_text())
    wed = next(d for d in ref["days"] if d["day"] == "Wednesday")
    ref_target = next(a for a in wed["arms"] if a["arm"] == "Wednesday_target_model")
    ref_blind = next(a for a in wed["arms"] if a["arm"] == "Wednesday_blind_control")

    # Same seeded draw as the original run: Wednesday is processed first, from a fresh
    # default_rng(SEED), and the position draw consumes the first two permutations.
    rng = np.random.default_rng(SEED)
    mem, non, cal = sample_positions(split, rng)
    prov = index_provenance(mem, non, cal)

    # Proof the indices are the originals: the Lab 1 model reproduces the original numbers.
    repro = evaluate("Wednesday_target_model_lab1_reproduction", lab1, split, mem, non, cal)
    repro_ok = {k: (repro["attacks"][k]["accuracy"] == v["accuracy"]
                    and repro["attacks"][k]["tau"] == v["tau"])
                for k, v in ref_target["attacks"].items()}
    print("reproduction of original Wednesday target arm (accuracy and tau identical):",
          repro_ok)
    print("reproduced confidence_threshold accuracy:",
          repro["attacks"]["confidence_threshold"]["accuracy"])
    if not all(repro_ok.values()):
        raise SystemExit("could not reproduce original 1A numbers; indices not proven")

    # Generalisation gap, same sampling as the original (generator state after the two
    # permutations; the original gap for Lab 1 is reproduced as a check on that state).
    def gap_for(model):
        r = np.random.default_rng(SEED)
        sample_positions(split, r)
        return generalisation_gap(model, split, r)
    gap_lab1, gap_def = gap_for(lab1), gap_for(defended)
    print("lab1 gap reproduces original:", gap_lab1 == wed["generalisation_gap"])

    # The defended target arm, over HTTP.
    s0 = server_stats(args.base_url)
    http = HttpModel(args.base_url)
    arm = evaluate("Wednesday_defended_target_model", http, split, mem, non, cal)
    s1 = server_stats(args.base_url)
    # Sanity: the server really serves the defended file (compare with in-process).
    inproc = evaluate("check", defended, split, mem, non, cal)
    same_as_inproc = all(arm["attacks"][k]["accuracy"] == inproc["attacks"][k]["accuracy"]
                         for k in arm["attacks"])
    for k, v in arm["attacks"].items():
        print(f"defended {k:22s} acc={v['accuracy']:.4f} "
              f"CI=[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] auc={v['auc']:.4f} n={v['n']}")
    print("http result equals in-process defended result:", same_as_inproc)

    payload = {
        "seed": SEED, "day": "Wednesday",
        "source_model": {"path": str(DEFENDED.relative_to(ROOT)),
                         "sha256": sha256_file(DEFENDED)},
        "reference_model": {"path": str(LAB1.relative_to(ROOT)), "sha256": sha256_file(LAB1)},
        "members_note": "members are the original clean Wednesday training rows (positions "
                        "< n_train); adversarial rows exist only in the defended model's "
                        "training set and are not scored",
        "n_train_clean": int(len(split.X_train)), "n_test": int(len(split.X_test)),
        "eval_index_provenance": prov,
        "indices_match_original_1a": {
            "provenance_equals_task1a_json": (prov == wed["eval_index_provenance"]
                                                  if "eval_index_provenance" in wed else None),
            "lab1_model_reproduces_original_target_arm": repro_ok,
            "lab1_reproduced_accuracy": {k: v["accuracy"]
                                         for k, v in repro["attacks"].items()},
            "lab1_original_accuracy": {k: v["accuracy"]
                                       for k, v in ref_target["attacks"].items()},
            "lab1_gap_reproduces_original": gap_lab1 == wed["generalisation_gap"],
        },
        "defended_target_arm": arm["attacks"],
        "defended_equals_in_process_scoring": same_as_inproc,
        "generalisation_gap_defended": gap_def,
        "generalisation_gap_lab1": gap_lab1,
        "blind_control_reference": {"source": "results/task1a_membership.json "
                                              "Wednesday_blind_control",
                                    "attacks": ref_blind["attacks"]},
        "original_target_reference": ref_target["attacks"],
        "transport": {"target_arm": "http", "base_url": args.base_url,
                      "gap_and_reproduction_checks": "in-process (experimenter diagnostics)",
                      "client_requests": http.requests_made, "client_rows": http.rows_sent,
                      "server_requests_delta": s1["requests"] - s0["requests"],
                      "server_rows_delta": s1["rows_scored"] - s0["rows_scored"]},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
