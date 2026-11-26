"""Re-score results/task2_redteam.json from the stored responses, no LLM re-run.

The success criteria are pure functions of the response text, and every trial already
stores its full `response`, so fixing a scorer (e.g. the sentinel-refusal false positive
in chatbot/attacks.py) does not require re-running the model. This recomputes each
trial's `success` with the CURRENT attack definitions and rolls the per-attack counts
back up. It refuses to run if a stored response is missing, rather than silently
scoring it as a failure.

    .venv/bin/python scripts/rescore_task2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.attacks import ATTACKS  # noqa: E402

PATH = ROOT / "results" / "task2_redteam.json"
BY_KEY = {a.key: a for a in ATTACKS}


def main() -> int:
    d = json.loads(PATH.read_text())
    changes = []
    for side in ("before", "after"):
        for rec in d[side]:
            atk = BY_KEY[rec["key"]]
            n = 0
            for tr in rec["trials"]:
                if "response" not in tr or tr["response"] is None:
                    raise SystemExit(f"{rec['key']} run {tr['run']} has no stored "
                                     f"response; cannot re-score without re-running")
                new = bool(atk.success(tr["response"]))
                if new != tr["success"]:
                    changes.append(f"{side}/{rec['key']} run {tr['run']}: "
                                   f"{tr['success']} -> {new}")
                    tr["success"] = new
                if new:
                    n += 1
            rec["successes"] = n
            eff = rec["effective_runs"]
            rec["success_rate"] = (n / eff) if eff else 0.0
    d["rescore_note"] = ("success re-derived from stored responses with the current "
                         "chatbot/attacks.py criteria (sentinel-refusal false positive "
                         "fixed); responses themselves are unchanged, no LLM re-run.")
    PATH.write_text(json.dumps(d, indent=2))
    print(f"re-scored; {len(changes)} trial(s) changed")
    for c in changes:
        print("  ", c)
    for side in ("before", "after"):
        print(side, {r["key"]: r["successes"] for r in d[side]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
