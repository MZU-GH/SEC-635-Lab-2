#!/usr/bin/env python3
"""Re-file the stored OWASP category on the Task 2 trial records (stdlib only, no network).

The category is defined in chatbot/attacks.py (source of truth). results/task2_redteam.json
also carries it per record (read by the gate to confirm both arms ran the same attacks).
This script rewrites ONLY that `owasp` field to match attacks.py, records the old->new
mapping under a top-level "owasp_remap" key, and touches no response, score or seed.

    python3 scripts/remap_owasp.py [--date 2026-11-25]
"""
import argparse, json, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_stub = types.ModuleType("chatbot.llm")   # no network code can run
_stub.GUARD_MODEL = _stub.GUARD_URL = ""
_stub.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network disabled"))
sys.modules["chatbot.llm"] = _stub
from chatbot.attacks import ATTACKS  # noqa: E402

PATH = ROOT / "results" / "task2_redteam.json"
REASON = ("OWASP LLM Top 10 2025 treats jailbreaking as a form of prompt injection (LLM01); "
          "LLM05 covers model output consumed unsafely by downstream code, which this "
          "chatbot does not do.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-25")
    a = ap.parse_args()
    want = {x.key: x.owasp for x in ATTACKS}
    d = json.loads(PATH.read_text())
    mapping = {}
    for side in ("before", "after"):
        for rec in d[side]:
            new = want[rec["key"]]
            if rec["owasp"] != new:
                mapping.setdefault(rec["owasp"], {"to": new, "attacks": []})
                if rec["key"] not in mapping[rec["owasp"]]["attacks"]:
                    mapping[rec["owasp"]]["attacks"].append(rec["key"])
                rec["owasp"] = new
    if mapping:
        d["owasp_remap"] = {"date": a.date, "reason": REASON,
                            "old_to_new": mapping}
        PATH.write_text(json.dumps(d, indent=2))
    print(json.dumps(mapping, indent=1) if mapping else "nothing to remap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
