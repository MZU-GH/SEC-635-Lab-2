"""Record the textbook sentences the report cites, so the audit works without the textbook.

The report quotes figures from the course textbook (chapters/course9.txt: Chapter 9, the
detector chapter): the adversarial-attack budget "0.10" it contrasts with its own training
radius, and the randomized-smoothing radius 0.616 (sigma 0.25, n = 1,000). chapters/ is
course material and is not part of the published tree. This script copies the one sentence
that carries each figure, verbatim (whitespace normalised), plus the sha256 of the chapter
file, into results/textbook_citations.json. verify_report_numbers.py then checks

  * each cited number against its stored sentence (and the textbook's own arithmetic,
    R = sigma x z, recomputed here), always; and
  * each stored sentence against chapters/course9.txt, and the stored sha256, when the
    chapter file is present (otherwise it says so and relies on the stored sentences).

    python3 scripts/make_textbook_citations.py [--course9 PATH] [--out PATH]
    python3 scripts/make_textbook_citations.py --verify [--course9 PATH] [--json PATH]

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TXT = ROOT / "chapters" / "course9.txt"
DEFAULT_JSON = ROOT / "results" / "textbook_citations.json"

# (id, fragment that must occur in the quoted sentence, the number the report cites, what it is)
CITED = (
    ("attack_budget", "budget ε = 0.10", "0.10",
     "the textbook's worked PGD attack budget (report 1C: 'not the textbook's 0.10')"),
    ("smoothing_radius_formula", "R = 0.25 × 2.463 = 0.616", "0.616",
     "textbook certified radius, worked example (sigma 0.25, 1,000 unanimous copies)"),
    ("smoothing_radius_summary", "proven L2 radius of 0.616 at σ = 0.25 and n = 1,000", "0.616",
     "textbook summary of the same radius, with sigma and n"),
)


def norm(text: str) -> str:
    """Whitespace-normalised text (the form in which sentences are stored and compared)."""
    return re.sub(r"\s+", " ", text).strip()


def sentence_containing(text: str, fragment: str) -> str:
    for line in text.splitlines():
        line = norm(line)
        if fragment in line:
            for s in re.split(r"(?<=[.)])\s+(?=[A-Z])", line):
                if fragment in s:
                    return s
    raise SystemExit(f"fragment {fragment!r} not found in the textbook text")


def build(txt: Path) -> dict:
    text = txt.read_text()
    sentences, checks = {}, []
    for cid, frag, value, what in CITED:
        s = sentence_containing(text, frag)
        key = next((k for k, v in sentences.items() if v == s), f"s{len(sentences) + 1}")
        sentences[key] = s
        checks.append({"id": cid, "what": what, "fragment": frag, "value": value, "sentence": key})
    return {"source": "course textbook, chapter 9 (chapters/course9.txt, not published)",
            "source_sha256": hashlib.sha256(txt.read_bytes()).hexdigest(),
            "sentences": sentences, "checks": checks}


def verify(data: dict, txt: Path | None) -> tuple[list[str], str]:
    """(problems, how): the stored citations must be self-consistent (fragment and number in
    their sentence, the radius arithmetic right) and, when `txt` exists, match it (same
    sha256, every sentence verbatim). Never raises on a malformed JSON."""
    out: list[str] = []
    sents = data.get("sentences") if isinstance(data, dict) else None
    checks = data.get("checks") if isinstance(data, dict) else None
    if not isinstance(sents, dict) or not sents or not isinstance(checks, list) or not checks:
        return ["textbook citations: no sentences or checks stored"], "stored JSON only"
    for c in checks:
        cid, s = c.get("id"), sents.get(c.get("sentence"), "")
        frag, val = c.get("fragment", ""), str(c.get("value", ""))
        if not frag or frag not in s:
            out.append(f"textbook citations: {cid}: fragment {frag!r} not in its stored sentence")
        if not val or val not in frag:
            out.append(f"textbook citations: {cid}: cited number {val!r} not in its fragment {frag!r}")
        m = re.search(r"R = ([\d.]+) × ([\d.]+) = ([\d.]+)", frag)
        if m:   # the textbook's own arithmetic: radius = sigma x z, at the printed rounding
            sig, z, r = (float(g) for g in m.groups())
            if round(sig * z, len(m.group(3).split(".")[1])) != r:
                out.append(f"textbook citations: {cid}: {sig} x {z} does not round to {r}")
    how = "stored JSON only (chapters/course9.txt not present)"
    if txt is not None and txt.exists():
        how = f"checked against {txt.name}"
        plain = norm(txt.read_text())
        if hashlib.sha256(txt.read_bytes()).hexdigest() != data.get("source_sha256"):
            out.append(f"textbook citations: source_sha256 does not match {txt.name}")
        for k, s in sents.items():
            if s not in plain:
                out.append(f"textbook citations: sentence {k} is not verbatim in {txt.name}")
    return out, how


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course9", type=Path, default=DEFAULT_TXT)
    ap.add_argument("--out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="file to check with --verify")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        problems, how = verify(json.loads(a.json.read_text()), a.course9)
        print(f"textbook citations: {how}")
        for p in problems:
            print("FAIL:", p)
        print("PASS" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    data = build(a.course9)
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} (course9 sha256 {data['source_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
