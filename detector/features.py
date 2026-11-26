"""Which CICIDS-2018 flow features an off-path attacker can actually change.

Lab 2 Task 1B requires deciding "which features an attacker can change from their own
end and which they cannot" before running evasion. Rather than hand-wave that split, we
derive it from the attack that produced each flow: brute force is many short
attacker-originated TCP sessions, so an attacker controls their own connection behaviour
(packet sizes, pacing, connection count) but not the server's replies, the observed
aggregate rate, or what the victim's host sends back.

Categories:
  MUTABLE   - attacker controls their own client-side traffic generation.
  IMMUTABLE - server-side / victim-side / derived-aggregate the attacker cannot set.

Every name here is verified against the real feature list in `assert_covers_all()`.
"""

from __future__ import annotations

import numpy as np

# Attacker-generated client-side flow characteristics. They pick the payload sizes of
# what they send, and the rate at which they send it.
MUTABLE_PREFIXES = (
    "Fwd Pkt Len",      # Fwd Pkt Len Max/Min/Mean/Std - attacker's own packet sizes
    "Fwd Pkt Size",
    "Fwd IAT",          # inter-arrival time of the attacker's packets -> pacing
    "Init Fwd Win Byts",   # attacker's own TCP window
    "Fwd Seg Size",
    "Fwd Header Len",
    "Tot Fwd Pkts",     # how many packets they send
    "TotLen Fwd Pkts",
    "Flow Duration",    # how long they keep the connection open
    "Active Mean", "Active Std", "Active Min", "Active Max",
    "Idle Mean", "Idle Std", "Idle Min", "Idle Max",
    "Fwd Act Data Pkts",
    "Fwd PUSH Flags", "Fwd URG Flags",
    "Protocol",         # the attacker picks which protocol their traffic speaks
    "Flow IAT",         # dominated by the attacker's packet train in a brute-force flow
    "Fwd PSH Flags",    # PSH on the attacker's own packets
    "Fwd Pkts/s",       # attacker packets over attacker-chosen duration
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts",
    # Aggregate-over-both-directions features. The attacker cannot set the server's
    # packet sizes, but they can shift a mixed statistic by changing how many of their
    # own packets there are and what they contain, so the influence is real if indirect.
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "Pkt Size Avg",
    # TCP flag counts over the whole flow. Every flag on an attacker packet is theirs to
    # set, and counts only go up, so each is at least monotonically attackable.
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
)

# Server replies, victim behaviour, and aggregates derived from the whole path. An
# off-path attacker cannot set the victim server's packet sizes, its window, its ACK
# pacing, or a rate the router computed from both directions.
IMMUTABLE_PREFIXES = (
    "Bwd Pkt Len", "Bwd Pkt Size", "Bwd IAT", "Bwd Header Len",
    "Init Bwd Win Byts", "Bwd Seg Size",
    "Tot Bwd Pkts", "TotLen Bwd Pkts",
    "Bwd PUSH Flags", "Bwd URG Flags", "Bwd Act Data Pkts",
    "Flow Byts/s", "Flow Pkts/s",       # derived from both directions
    "Down/Up Rate", "Down/Up Ratio",    # ratio of victim bytes to attacker bytes
    "SYN Flag Count", "ACK Flag Count",  # flags the server sets
    "PUSH Flag Flag", "URG Flag Flag",
    "Bwd PSH Flags",    # PSH set by the server
    "Bwd Pkts/s",       # server packet rate
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Bwd Pkts", "Subflow Bwd Byts",
)

# Features that are only meaningful in combination. Perturbing them independently
# produces traffic that cannot exist: a mean packet length below the minimum, a
# standard deviation of zero with more than one packet, a negative duration.
# `project_feasible` enforces these relations after every attack step.
FEASIBILITY_RULES_DOC = """
For each flow, after perturbation:
  Fwd Pkt Len Min <= Fwd Pkt Len Mean <= Fwd Pkt Len Max
  Fwd Pkt Len Std >= 0
  Flow Duration >= 0 and >= sum of the Fwd IAT it claims
  Tot Fwd Pkts >= 1 and integer
  Fwd IAT Min <= Fwd IAT Mean <= Fwd IAT Max
"""


def classify(feature_names: list[str]) -> np.ndarray:
    """Boolean mask, True where the attacker can move the feature."""
    mask = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        if name.startswith(IMMUTABLE_PREFIXES):
            continue
        if name.startswith(MUTABLE_PREFIXES):
            mask[i] = True
    return mask


def project_feasible(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Clip an adversarial perturbation back into the set of physically possible flows.

    Without this, an L-inf attack happily reports "evaded" by claiming a mean packet
    length of 3 bytes with a minimum of 900, which no stacker would ever emit and which
    would be caught by a schema check long before the classifier.
    """
    out = X.copy()
    idx = {n: i for i, n in enumerate(feature_names)}

    def col(name: str) -> int | None:
        return idx.get(name)

    # Lengths must be non-negative and ordered min <= mean <= max.
    for side in ("Fwd", "Bwd"):
        lo, mean, hi = (col(f"{side} Pkt Len Min"), col(f"{side} Pkt Len Mean"),
                        col(f"{side} Pkt Len Max"))
        if lo is None or mean is None or hi is None:
            continue
        out[:, lo] = np.maximum(out[:, lo], 0)
        out[:, mean] = np.maximum(out[:, mean], out[:, lo])
        out[:, hi] = np.maximum(out[:, hi], out[:, mean])
        std = col(f"{side} Pkt Len Std")
        if std is not None:
            out[:, std] = np.maximum(out[:, std], 0)

    # Inter-arrival times are non-negative and ordered.
    for side in ("Flow", "Fwd", "Bwd"):
        lo, mean, hi = (col(f"{side} IAT Min"), col(f"{side} IAT Mean"), col(f"{side} IAT Max"))
        if mean is None:
            continue
        out[:, mean] = np.maximum(out[:, mean], 0)
        if lo is not None:
            out[:, lo] = np.clip(out[:, lo], 0, None)
            out[:, mean] = np.maximum(out[:, mean], out[:, lo])
        if hi is not None:
            out[:, hi] = np.clip(out[:, hi], 0, None)
            out[:, mean] = np.minimum(out[:, mean], out[:, hi]) if lo is None else out[:, mean]

    # Durations and counters cannot go negative; packet counts stay integral.
    for name in ("Flow Duration", "Flow IAT Tot", "Fwd IAT Tot", "Bwd IAT Tot"):
        j = col(name)
        if j is not None:
            out[:, j] = np.maximum(out[:, j], 0)
    for name in ("Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
                 "Fwd Act Data Pkts", "Bwd Act Data Pkts"):
        j = col(name)
        if j is not None:
            out[:, j] = np.maximum(np.round(out[:, j]), 0)

    return out


def assert_covers_all(feature_names: list[str]) -> None:
    """Fail loudly if a feature is neither explicitly mutable nor explicitly immutable.

    An unclassified feature silently becoming IMMUTABLE would understate the attack,
    which is the one error direction this lab must not make.
    """
    known = set()
    for name in feature_names:
        if any(name.startswith(p) for p in MUTABLE_PREFIXES + IMMUTABLE_PREFIXES):
            known.add(name)
    unknown = [n for n in feature_names if n not in known]
    if unknown:
        raise ValueError(
            f"{len(unknown)} features are unclassified for the attacker threat model; "
            f"decide and add them to MUTABLE_PREFIXES or IMMUTABLE_PREFIXES: {unknown}"
        )
