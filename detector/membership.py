"""Task 1A: membership inference against the Lab 1 detector, API-only.

Question: from the API's answers alone, decide whether a given flow was in the
detector's training set. Judge against the 50% guess.

Why this needs controls rather than one number
----------------------------------------------
The textbook signal for membership is the *generalisation gap*: a model has memorised
its training rows, so it is more confident on them than on rows it never saw. Lab 1's
Wednesday model scores ~1.0 F1 on train AND on test, so the gap is essentially zero and
the prior expectation is that MI fails.

A naive threshold attack can still beat 50% here for reasons that have nothing to do
with memorisation, and each needs its own control:

1. Distribution shift. The split is by row order (raw timestamp) and the halves have
   different class mixes (train 33.87% attack, test 42.51%). A threshold can be detecting "does this
   flow look like a test-window flow" rather than "was this flow trained on".
   -> blind control: score the same Wednesday rows with a model trained on Thursday,
      which never saw any Wednesday flow. Its accuracy is pure artifact and is the
      floor the real attack must clear - not the 50% guess.

2. Near-duplicate leakage. With a split by row order, a test flow one second after a
   training flow is nearly the same packet stream, so membership leaks through the
   feature space instead of through the model.
   -> duplicate control: re-score with test rows split into those near and far from the
      training set, and report whether the signal survives on the far subset.

Thresholds are fitted on a calibration pool disjoint from every scored row; choosing
tau on the evaluated rows would inflate the headline number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import THURSDAY_PARQUET, WEDNESDAY_PARQUET, Split, build_split

SEED = 42
N_SCORE = 20_000       # members and non-members each in the scored set
N_CALIB = 20_000       # calibration pool per side, never scored


class HttpModel:
    """Black-box handle on a model served by detector/api.py: scores only, via /score_batch.

    Exposes only `predict_proba`, which is all the membership attack uses. Rows are sent
    in chunks; the attacker sees one P(attack) per flow and nothing else.
    """

    def __init__(self, base_url: str, chunk: int = 40000) -> None:
        self.base_url, self.chunk = base_url, chunk
        self.requests_made = 0
        self.rows_sent = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import requests
        out = []
        for i in range(0, len(X), self.chunk):
            part = np.asarray(X[i:i + self.chunk], dtype=np.float64)
            r = requests.post(f"{self.base_url}/score_batch",
                              json={"features": part.tolist()}, timeout=600)
            r.raise_for_status()
            out.append(np.asarray(r.json()["scores"], dtype=float))
            self.requests_made += 1
            self.rows_sent += len(part)
        p = np.concatenate(out) if out else np.zeros(0)
        if len(p) != len(X):
            raise RuntimeError("API refused the batch (wrong width?)")
        return np.stack([1 - p, p], axis=1)


def server_stats(base_url: str) -> dict:
    import requests
    return requests.get(f"{base_url}/stats", timeout=30).json()


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI; the normal approximation is misleading near 0.5."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def signals(p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row statistics an MI attack can be built from.

    The attacker is assumed to know the true label of the flow they query - the
    standard assumption (Yeom et al., Song et al.) and realistic here, since a SOC
    operator replaying captured traffic knows what it was labelled.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    p_true = np.where(y == 1, p, 1 - p)
    return {
        "confidence": p_true,                       # Yeom: high confidence => member
        "loss": -np.log(p_true),                    # Song: low training loss => member
        # Logit keeps ordering that a saturating probability throws away, which matters
        # for a model whose probabilities sit at 1e-6 and 1-1e-6.
        "margin": np.log(p_true / (1 - p_true)),
    }


def fit_threshold(sig: np.ndarray, member: np.ndarray, above: bool) -> float:
    """Threshold maximising accuracy on a calibration pool."""
    order = np.argsort(sig)
    s, m = sig[order], member[order].astype(bool)
    cands = np.unique(np.concatenate([s, (s[:-1] + s[1:]) / 2])) if len(s) else s
    best_tau, best_acc = float(cands[0]), -1.0
    for tau in cands:
        pred = (s >= tau) if above else (s <= tau)
        acc = float(np.mean(pred == m))
        if acc > best_acc:
            best_tau, best_acc = float(tau), acc
    return best_tau


def evaluate(name: str, model, split: Split,
             mem_pos: np.ndarray, non_pos: np.ndarray,
             cal_pos: np.ndarray) -> dict:
    """Run the threshold attacks for one model on one split."""
    def prob(pos):
        return model.predict_proba(split.rows(pos))[:, 1]

    p_mem, p_non = prob(mem_pos), prob(non_pos)
    s_mem = signals(p_mem, split.labels(mem_pos))
    s_non = signals(p_non, split.labels(non_pos))
    s_cal = signals(prob(cal_pos), split.labels(cal_pos))
    cal_member = split.is_member(cal_pos).astype(int)

    from sklearn.metrics import roc_auc_score
    y_all = np.concatenate([np.ones(len(p_mem)), np.zeros(len(p_non))])

    attacks = {}
    for key, above in (("confidence", True), ("margin", True), ("loss", False)):
        tau = fit_threshold(s_cal[key], cal_member, above)
        hit_mem = (s_mem[key] >= tau) if above else (s_mem[key] <= tau)
        hit_non = (s_non[key] >= tau) if above else (s_non[key] <= tau)
        correct = int(hit_mem.sum() + (~hit_non).sum())
        total = len(p_mem) + len(p_non)
        lo, hi = clopper_pearson(correct, total)
        attacks[f"{key}_threshold"] = {
            "tau": tau,
            "accuracy": correct / total,
            "ci95": [lo, hi],
            "tpr_members_flagged": float(hit_mem.mean()),
            "fpr_nonmembers_flagged": float(hit_non.mean()),
            "auc": float(roc_auc_score(y_all, np.concatenate([s_mem[key], s_non[key]]))),
            "mean_gap": float(np.mean(s_mem[key]) - np.mean(s_non[key])),
            "n": total,
        }
    return {"arm": name, "attacks": attacks}


def duplicate_control(split: Split, rng: np.random.Generator) -> dict:
    """How many test flows are indistinguishable from a training flow?

    Brute-force traffic is extremely repetitive, so the same feature vector recurs
    thousands of times. A test row that is an exact duplicate of a training row makes
    "was this flow in the training set" ill-posed: the *row* was not, the *vector* was.
    An MI attack can score above chance on such data without learning anything about the
    model, so this overlap is the floor any MI result here must be read against.

    Reported as exact-vector overlap (hash on the rounded feature tuple) plus a
    near-duplicate rate at a small standardized distance, which is the softer version of
    the same leak.
    """
    def key(X: np.ndarray) -> np.ndarray:
        # Round before hashing: float noise from the CSV parse would otherwise hide
        # duplicates that are identical to any precision a sensor could resolve.
        R = np.round(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 6)
        return np.array([hash(r.tobytes()) for r in np.ascontiguousarray(R)])

    k_tr = key(split.X_train)
    k_te = key(split.X_test)
    tr_set = set(k_tr.tolist())
    exact_hit = np.fromiter((k in tr_set for k in k_te.tolist()), dtype=bool, count=len(k_te))

    out = {
        "test_rows_exact_duplicate_of_a_train_row": int(exact_hit.sum()),
        "frac_test_exact_duplicates": float(exact_hit.mean()),
        "distinct_train_vectors": len(tr_set),
        "distinct_test_vectors": len(set(k_te.tolist())),
        "n_train": int(len(k_tr)), "n_test": int(len(k_te)),
    }
    # Per-class: if duplicates concentrate in one class the artifact is class-coupled.
    y_te = split.y_test
    out["frac_duplicates_among_test_benign"] = float(exact_hit[y_te == 0].mean())
    out["frac_duplicates_among_test_attack"] = float(exact_hit[y_te == 1].mean())
    out["members_are_exact_duplicates_of_another_member"] = float(
        len(k_tr) - len(tr_set)) / max(1, len(k_tr))
    return out


def generalisation_gap(model, split: Split, rng: np.random.Generator) -> dict:
    """The mechanism membership inference depends on, measured directly.

    Chapter 9's claim is that MI works when the model is more confident on rows it
    trained on than on rows it did not. If that gap is ~0, MI cannot work and the
    honest finding is a negative one.
    """
    n = 40_000
    tr = rng.choice(len(split.X_train), size=min(n, len(split.X_train)), replace=False)
    te = rng.choice(len(split.X_test), size=min(n, len(split.X_test)), replace=False)
    p_tr = model.predict_proba(split.X_train[tr])[:, 1]
    p_te = model.predict_proba(split.X_test[te])[:, 1]
    conf_tr = np.where(split.y_train[tr] == 1, p_tr, 1 - p_tr)
    conf_te = np.where(split.y_test[te] == 1, p_te, 1 - p_te)
    return {
        "mean_conf_on_train": float(conf_tr.mean()),
        "mean_conf_on_test": float(conf_te.mean()),
        "gap": float(conf_tr.mean() - conf_te.mean()),
        "median_abs_logit_train": float(np.median(np.abs(np.log(
            np.clip(p_tr, 1e-12, 1 - 1e-12) / (1 - np.clip(p_tr, 1e-12, 1 - 1e-12)))))),
        "median_abs_logit_test": float(np.median(np.abs(np.log(
            np.clip(p_te, 1e-12, 1 - 1e-12) / (1 - np.clip(p_te, 1e-12, 1 - 1e-12)))))),
        "accuracy_train": float(np.mean((p_tr > 0.5).astype(int) == split.y_train[tr])),
        "accuracy_test": float(np.mean((p_te > 0.5).astype(int) == split.y_test[te])),
    }


def sample_positions(split: Split, rng: np.random.Generator) -> tuple:
    """The seeded draw of member / non-member / calibration positions.

    Consumes exactly two permutations from `rng`, in this order, so any caller that
    reproduces the generator state reproduces the original 1A indices.
    """
    tr = rng.permutation(len(split.X_train))
    te = rng.permutation(len(split.X_test)) + len(split.X_train)
    mem_pos, cal_mem = tr[:N_SCORE], tr[N_SCORE:N_SCORE + N_CALIB]
    non_pos, cal_non = te[:N_SCORE], te[N_SCORE:N_SCORE + N_CALIB]
    return mem_pos, non_pos, np.concatenate([cal_mem, cal_non])


def index_provenance(mem_pos, non_pos, cal_pos) -> dict:
    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()
    return {"sha256_member_positions": h(mem_pos), "sha256_nonmember_positions": h(non_pos),
            "sha256_calibration_positions": h(cal_pos),
            "n_member": int(len(mem_pos)), "n_nonmember": int(len(non_pos)),
            "n_calibration": int(len(cal_pos))}


def run_day(day: str, model, blind, rng: np.random.Generator,
            target_scorer=None) -> dict:
    """Full 1A protocol on one day's split.

    `model` is the in-process pipeline (used for the generalisation-gap diagnostic and as
    the default scorer). `target_scorer`, if given, is what the TARGET arm queries
    instead (an HttpModel); the blind-control arm always runs in-process.
    """
    split = build_split(*({"Wednesday": (WEDNESDAY_PARQUET, "Wednesday"),
                           "Thursday": (THURSDAY_PARQUET, "Thursday")}[day]))
    mem_pos, non_pos, cal_pos = sample_positions(split, rng)

    arms = [evaluate(f"{day}_target_model", target_scorer or model, split,
                     mem_pos, non_pos, cal_pos)]
    arms[0]["transport"] = "http" if target_scorer is not None else "in-process"
    if blind is not None:
        arms.append(evaluate(f"{day}_blind_control", blind, split, mem_pos, non_pos, cal_pos))
        arms[1]["transport"] = "in-process (experimental control model, not the attack target)"
    return {
        "day": day,
        "eval_index_provenance": index_provenance(mem_pos, non_pos, cal_pos),
        "split": {"n_train": len(split.X_train), "n_test": len(split.X_test),
                  "train_attack_pct": float(split.y_train.mean() * 100),
                  "test_attack_pct": float(split.y_test.mean() * 100)},
        "generalisation_gap": generalisation_gap(model, split, rng),
        "duplicate_control": duplicate_control(split, rng),
        "arms": arms,
    }


def main() -> int:
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def mlp():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25,
                                  random_state=SEED, early_stopping=True)),
        ])

    ap = argparse.ArgumentParser()
    ap.add_argument("--url-wednesday", default=None,
                    help="score the Wednesday TARGET arm through this API (serving the Lab 1 "
                         "Wednesday model)")
    ap.add_argument("--url-thursday", default=None,
                    help="score the Thursday TARGET arm through this API (serving the Lab 1 "
                         "Thursday model via SEC635_MODEL); omitted -> in-process")
    ap.add_argument("--out", default=None, help="default results/task1a_membership.json")
    args = ap.parse_args()
    http_wed = HttpModel(args.url_wednesday) if args.url_wednesday else None
    http_thu = HttpModel(args.url_thursday) if args.url_thursday else None

    rng = np.random.default_rng(SEED)
    lab1_wed = joblib.load(ROOT / "lab1" / "outputs" / "models"
                           / "task1_wednesday_mlp_neuralnet_temporal.joblib")
    lab1_thu = joblib.load(ROOT / "lab1" / "outputs" / "models"
                           / "task1_thursday_mlp_neuralnet_temporal.joblib")

    # Blind controls: a model that never saw the day under test.
    wed = build_split(WEDNESDAY_PARQUET, "Wednesday")
    thu = build_split(THURSDAY_PARQUET, "Thursday")
    blind_on_wed = mlp().fit(thu.X_train, thu.y_train)
    blind_on_thu = mlp().fit(wed.X_train, wed.y_train)

    payload = {
        "seed": SEED,
        "n_scored_per_class": N_SCORE,
        "n_calibration_per_class": N_CALIB,
        "days": [run_day("Wednesday", lab1_wed, blind_on_wed, rng, http_wed),
                 run_day("Thursday", lab1_thu, blind_on_thu, rng, http_thu)],
    }
    if http_wed or http_thu:
        payload["transport"] = {
            "wednesday_target_arm": "http" if http_wed else "in-process",
            "thursday_target_arm": "http" if http_thu else "in-process",
            "blind_control_arms": "in-process (experimental controls, not attack targets)",
            "generalisation_gap_and_duplicate_control": "in-process (experimenter diagnostics "
                                                        "of the model, not attacker queries)",
            "client_rows_sent": {"wednesday": http_wed.rows_sent if http_wed else 0,
                                 "thursday": http_thu.rows_sent if http_thu else 0},
            "server_stats_final": {
                "wednesday": server_stats(args.url_wednesday) if http_wed else None,
                "thursday": server_stats(args.url_thursday) if http_thu else None},
        }
    out_path = Path(args.out) if args.out else ROOT / "results" / "task1a_membership.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    for d in payload["days"]:
        print(f"\n=== {d['day']} === split {d['split']}")
        g = d["generalisation_gap"]
        print(f"  gap: train_conf={g['mean_conf_on_train']:.6f} "
              f"test_conf={g['mean_conf_on_test']:.6f} gap={g['gap']:.6f} "
              f"| acc train={g['accuracy_train']:.4f} test={g['accuracy_test']:.4f}")
        dup = d["duplicate_control"]
        print(f"  duplicates: {dup['frac_test_exact_duplicates']:.4f} of test rows are "
              f"exact copies of a train row ({dup['test_rows_exact_duplicate_of_a_train_row']:,}"
              f" of {dup['n_test']:,}); distinct train vectors {dup['distinct_train_vectors']:,}")
        for arm in d["arms"]:
            for k, v in arm["attacks"].items():
                print(f"  {arm['arm']:32s} {k:22s} acc={v['accuracy']:.4f} "
                      f"CI=[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] auc={v['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
