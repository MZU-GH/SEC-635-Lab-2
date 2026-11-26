"""Record the lab-brief sentences the report cites, so the audit works without the brief.

number_coverage.py verifies three citations against the assignment .docx (the 697 ATT&CK
techniques, the "(1,700 entries)" KEV figure and the activity number the brief names).
The .docx is course material and is not part of the published tree, so this script copies
the one brief sentence that carries them, verbatim, plus the .docx sha256, into
results/brief_citations.json. number_coverage.py uses the .docx when present (and checks
that this JSON still matches it) and the JSON otherwise. Stdlib only.

    python3 scripts/make_brief_citations.py [--docx PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCX = ROOT / "docs" / "labs" / "SEC635Lab2IntegratedLabII.docx"
DEFAULT_OUT = ROOT / "results" / "brief_citations.json"

# (id, fragment that must occur in the quoted sentence, what the report cites)
CITED = (
    ("attack_techniques", "697 ATT&CK techniques", "ATT&CK technique count"),
    ("kev_entries", "(1,700 entries)", "CISA KEV entry count"),
    ("activity_number", "Activity 4.4", "course activity the brief names"),
)


def brief_text(docx: Path) -> str:
    """Whitespace-normalised plain text of the .docx body, XML entities decoded."""
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode()
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
    return html.unescape(txt).strip()


def sentence_containing(text: str, fragment: str) -> str:
    for s in re.split(r"(?<=[.)])\s+(?=[A-Z])", text):
        if fragment in s:
            return s
    raise SystemExit(f"fragment {fragment!r} not found in the brief")


def build(docx: Path) -> dict:
    text = brief_text(docx)
    sentences, checks = {}, []
    for cid, frag, what in CITED:
        s = sentence_containing(text, frag)
        key = next((k for k, v in sentences.items() if v == s), f"s{len(sentences) + 1}")
        sentences[key] = s
        checks.append({"id": cid, "what": what, "fragment": frag, "sentence": key})
    return {"source": "assignment brief (.docx, not published)",
            "docx_sha256": hashlib.sha256(docx.read_bytes()).hexdigest(),
            "sentences": sentences, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    data = build(a.docx)
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} (docx sha256 {data['docx_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
