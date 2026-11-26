"""Fix 10: exact Clopper-Pearson upper bounds for Task 2 cells that observed 0 successes.

Counts are read from results/task2_summary.json (never typed). For k successes of n:
  one_sided_95_upper   = beta.ppf(0.95,  k+1, n-k)   (one-sided 95% bound)
  two_sided_95_upper   = beta.ppf(0.975, k+1, n-k)   (upper end of the two-sided 95% CI)
"""
import json, sys
from datetime import date
from pathlib import Path
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "task2_summary.json").read_text())


def ub(k, n):
    # k < n always here (k == 0)
    return {"successes": k, "n": n,
            "one_sided_95_upper": float(beta.ppf(0.95, k + 1, n - k)),
            "two_sided_95_upper": float(beta.ppf(0.975, k + 1, n - k))}


cells = []
for p in S["per_attack"]:
    for arm in ("before", "after"):
        k, n = p[f"{arm}_successes"], p[f"n_{arm}"]
        if k == 0:
            cells.append({"cell": f"{p['key']}/{arm}", "arm": arm, **ub(k, n)})
benign = []
for arm in ("before", "after"):
    b = S["benign"][arm]
    benign.append({"cell": f"benign guardrail refusals/{arm}", "arm": arm,
                   **ub(b["guardrail_refusals"], b["total_runs"])})
out = {
    "method": "exact Clopper-Pearson via scipy.stats.beta",
    "one_sided_95_upper_formula": "beta.ppf(0.95, k+1, n-k)",
    "two_sided_95_upper_formula": "beta.ppf(0.975, k+1, n-k)",
    "source": "results/task2_summary.json",
    "code": {"script": "scripts/zero_rate_bounds.py", "date": date.today().isoformat()},
    "attack_cells_with_zero_successes": cells,
    "benign_guardrail_refusals": benign,
}
(ROOT / "results" / "task2_zero_rate_bounds.json").write_text(json.dumps(out, indent=2))
for c in cells + benign:
    print(f"{c['cell']}: {c['successes']}/{c['n']} one-sided95={c['one_sided_95_upper']:.6f} "
          f"two-sided95-upper={c['two_sided_95_upper']:.6f}")
