"""Port the Lab 1 sklearn MLP to torch so Task 1C gets exact gradients.

Lab 1's detector is `Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(64,32)))`.
sklearn exposes the fitted weights but not gradients, so white-box attacks (PGD, C&W)
need an equivalent graph. sklearn's forward pass for a binary MLPClassifier is

    a1 = relu(W0 @ x + b0); a2 = relu(W1 @ a1 + b1); p = sigmoid(W2 @ a2 + b2)

with `coefs_[i]` shaped (n_in, n_out) and `intercepts_[i]` shaped (n_out,). We copy those
tensors verbatim rather than retraining, so the port is the same model, not a lookalike.
`verify_equivalence` asserts that numerically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB1_MLP = ROOT / "lab1" / "outputs" / "models" / "task1_wednesday_mlp_neuralnet_temporal.joblib"


class FlowMLP(nn.Module):
    """The Lab 1 MLP plus its StandardScaler folded into the graph.

    Keeping the scaler inside the module matters for two reasons: gradients must flow
    through the standardization for white-box attacks to be correct, and the attack
    budget is measured in raw feature units, not standardized ones.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, coefs: list[np.ndarray],
                 intercepts: list[np.ndarray]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.linears = nn.ModuleList(
            [nn.Linear(c.shape[0], c.shape[1]) for c in coefs]
        )
        for lin, c, b in zip(self.linears, coefs, intercepts):
            with torch.no_grad():
                # sklearn stores (n_in, n_out); torch Linear stores (n_out, n_in).
                lin.weight.copy_(torch.tensor(c.T, dtype=torch.float32))
                lin.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(attack). Raw feature units in, probability out."""
        h = (x - self.mean) / self.scale
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = torch.relu(h)
        return torch.sigmoid(h).squeeze(-1)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        """Signed margin before the sigmoid. Positive means 'attack'.

        Attacks optimise this instead of the probability. On a model whose calibrated
        output is exactly 1.0 in float32, cross-entropy saturates: -log(p/(1-p)) hits
        division by zero and the gradient is worthless. The logit is linear in the last
        layer, so its gradient never vanishes, which is what makes PGD able to move at
        all against a detector this confident.
        """
        h = (x - self.mean) / self.scale
        for lin in self.linears:
            h = lin(h)
        return h.squeeze(-1)

    @property
    def n_features(self) -> int:
        return int(self.mean.numel())


def load_lab1_mlp(path: Path = LAB1_MLP) -> FlowMLP:
    """Build a FlowMLP carrying the exact weights Lab 1 shipped."""
    pipe = joblib.load(path)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    return FlowMLP(scaler.mean_, scaler.scale_, clf.coefs_, clf.intercepts_)


def decision_function(model: FlowMLP, X: np.ndarray) -> np.ndarray:
    """Signed margin in standardized units: logit(p). Positive means 'attack'.

    C&W minimizes |F(x')|, so it needs the logit rather than the probability.
    """
    with torch.no_grad():
        p = model(torch.tensor(np.asarray(X, dtype=np.float32))).clamp(1e-12, 1 - 1e-12)
    return torch.log(p / (1 - p)).numpy()


def verify_equivalence(X: np.ndarray, y: np.ndarray, path: Path = LAB1_MLP,
                       tol: float = 1e-5) -> dict:
    """Assert the torch port matches sklearn on the same rows."""
    from sklearn.metrics import accuracy_score

    pipe = joblib.load(path)
    model = load_lab1_mlp(path)
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        p_torch = model(Xt).numpy()
    p_sk = pipe.predict_proba(X)[:, 1]
    max_abs = float(np.max(np.abs(p_torch - p_sk)))
    return {
        "max_abs_prob_diff": max_abs,
        "passes": bool(max_abs < tol),
        "sklearn_accuracy": float(accuracy_score(y, pipe.predict(X))),
        "torch_accuracy": float(accuracy_score(y, (p_torch > 0.5).astype(int))),
        "n_checked": int(len(X)),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from detector.dataset import build_split

    split = build_split()
    rows = np.random.default_rng(0).choice(len(split.X_test), 5000, replace=False)
    report = verify_equivalence(split.X_test[rows], split.y_test[rows])
    for k, v in report.items():
        print(f"{k}: {v}")
    sys.exit(0 if report["passes"] else 1)
