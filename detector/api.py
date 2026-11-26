"""HTTP API around the Lab 1 detector - the black-box attack surface for Task 1A/1B.

Lab 2 gives Task 1A and 1B "the API only". So this service deliberately exposes the
minimum: a score per flow and nothing else. No gradients, no training-set size, no
per-feature attribution. White-box access for 1C is a separate in-process path
(`detector/torch_model.py`), not an endpoint.

Two score views are offered because they have different leakage profiles:
  score  = P(attack), what a production detector would log.
  margin = logit(P(attack)), which is strictly more informative than a rounded score
           and is what a C&W attack wants. Exposing it is a design choice we test.

SEC635_MODEL=<path> serves a different joblib pipeline (default: the Lab 1 Wednesday
model), e.g. models/task1c_defended.joblib for the 1C re-run. GET /stats reports the
server-side request and row counters, so query budgets are measured at the server.

Run: .venv/bin/uvicorn detector.api:app --host 127.0.0.1 --port 8055
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.dataset import LABEL_COL, LEAKAGE_COLUMNS, WEDNESDAY_PARQUET  # noqa: E402

app = FastAPI(title="SEC635 Lab2 detector API", version="1.0")

# Loaded once at import. The API never reveals which rows it was trained on.
_MODEL = None
_META: dict | None = None  # feature names only; the API never loads training rows
_DEFAULT_MODEL = ROOT / "lab1" / "outputs" / "models" / "task1_wednesday_mlp_neuralnet_temporal.joblib"
_MODEL_PATH = Path(os.environ.get("SEC635_MODEL", str(_DEFAULT_MODEL)))
if not _MODEL_PATH.is_absolute():
    _MODEL_PATH = ROOT / _MODEL_PATH

# Server-side query accounting. A "request" is one HTTP call to a scoring endpoint; a
# "row" is one flow scored. Counters are read-only over /stats.
_LOCK = threading.Lock()
_STATS = {"requests": 0, "rows_scored": 0,
          "score_requests": 0, "score_batch_requests": 0}


def _count(endpoint: str, rows: int) -> None:
    with _LOCK:
        _STATS["requests"] += 1
        _STATS[endpoint + "_requests"] += 1
        _STATS["rows_scored"] += rows


def _ensure_loaded():
    global _MODEL, _META
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(_MODEL_PATH)
        import pyarrow.parquet as pq
        names = [c for c in pq.read_schema(WEDNESDAY_PARQUET).names
                 if c not in LEAKAGE_COLUMNS and c != LABEL_COL]
        _META = {"feature_names": names, "n_features": len(names)}
    return _MODEL, _META


class ScoreRequest(BaseModel):
    """One flow as the 77 Lab 1 features, in Lab 1's column order."""

    features: list[float] = Field(..., description="77 numeric flow features")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="P(attack)")
    label: int = Field(..., description="1 = attack, 0 = benign at threshold 0.5")
    margin: float | None = Field(None, description="logit(P(attack)); only if asked for")


class HealthResponse(BaseModel):
    status: str
    model: str
    n_features: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model, _ = _ensure_loaded()
    return HealthResponse(status="ok",
                          model=_MODEL_PATH.name,
                          n_features=model.named_steps["scaler"].mean_.shape[0])


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a single flow. This is the only endpoint a black-box attacker needs."""
    model, meta = _ensure_loaded()
    _count("score", 1)
    x = np.asarray(req.features, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != meta['n_features']:
        # A wrong width is an attacker error, not a server error; say so precisely so
        # the harness can tell a malformed payload apart from a real response.
        return ScoreResponse(score=-1.0, label=-1,
                             margin=None)  # type: ignore[arg-type]
    p = float(model.predict_proba(x)[0][1])
    margin = float(np.log(max(p, 1e-12) / max(1 - p, 1e-12)))
    return ScoreResponse(score=p, label=int(p >= 0.5), margin=margin)


class BatchRequest(BaseModel):
    features: list[list[float]]


class BatchResponse(BaseModel):
    scores: list[float]
    labels: list[int]


@app.post("/score_batch", response_model=BatchResponse)
def score_batch(req: BatchRequest) -> BatchResponse:
    """Batch scoring. Exists only to make the *research* harness fast.

    A real deployment would not offer this, and the report notes that it is a
    throughput advantage handed to the attacker: membership inference and random-search
    evasion both become cheap when one call scores thousands of flows.
    """
    model, meta = _ensure_loaded()
    X = np.asarray(req.features, dtype=np.float64)
    _count("score_batch", len(X) if X.ndim == 2 else 0)
    if X.ndim != 2 or X.shape[1] != meta['n_features']:
        return BatchResponse(scores=[], labels=[])
    probs = model.predict_proba(X)[:, 1]
    return BatchResponse(scores=[float(v) for v in probs],
                         labels=[int(v >= 0.5) for v in probs])


@app.get("/stats")
def stats() -> dict:
    """Server-side counters (requests, rows scored) since process start. Counts only."""
    with _LOCK:
        return dict(_STATS)


@app.get("/features")
def features() -> dict:
    """Feature names and order.

    Exposed on purpose: the attacker in our threat model is assumed to know the feature
    schema (they operate the sensor), but not the weights or the training rows.
    """
    _, meta = _ensure_loaded()
    return {"features": meta["feature_names"]}
