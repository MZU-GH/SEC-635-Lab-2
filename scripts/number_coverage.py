"""Number coverage audit (gate 1b): is EVERY number in the report backed by executed code?

Gate 1 (verify_report_numbers.py) checks the numbers someone thought to check. This audit
closes the other half of the question: it extracts every numeric token from report.tex and
the files it \\input's, runs gate 1 with --dump-coverage (each helper that matches a number
registers the document line it matched on), and classifies every token as

  COVERED      a gate check verified that literal on that very line (one check covers one
               token occurrence, so a second, unchecked copy of the same number on the same
               line is NOT covered);
  ALLOWLISTED  a non-measurement (section/figure refs, years, model names, layout, attack
               labels, configuration values ...), matched by an entry of
               scripts/number_allowlist.json whose regex must contain the token in context;
  UNBACKED     anything else.

Exit status is non-zero if any token is UNBACKED, if gate 1 fails, or if the allowlist is
malformed. Stdlib only.

    .venv/bin/python scripts/number_coverage.py [--list-all] [--show-gate]

GATE_TEX / GATE_ROLLUP / GATE_PAYLOADS / GATE_RESULTS point both this audit and gate 1 at
copies (used by the negative tests); the allowlist is never changed by them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = Path(os.environ.get("NUMBER_ALLOWLIST", ROOT / "scripts" / "number_allowlist.json"))
GATE = ROOT / "scripts" / "verify_report_numbers.py"
# The assignment brief (.docx) is course material and is not published; its cited sentences
# are kept in results/brief_citations.json (written by scripts/make_brief_citations.py).
# BRIEF_CITATIONS / BRIEF_DOCX redirect them for negative tests.
BRIEF_JSON = Path(os.environ.get("BRIEF_CITATIONS", ROOT / "results" / "brief_citations.json"))
BRIEF_DOCX_OVERRIDE = os.environ.get("BRIEF_DOCX")

CATEGORIES = (
    "section/figure/table/appendix refs",
    "years and dates",
    "versions and model names",
    "citations",
    "course and lab identifiers",
    "layout",
    "OWASP IDs",
    "CVE and ATT&CK IDs",
    "attack labels",
    "hyperparameter settings",
    "file and artifact names",
    "definitions",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_source   # noqa: E402  (shared expander: report.tex + sections/*.tex)

DOC_PATHS = {
    "report": Path(os.environ.get("GATE_TEX", ROOT / "report" / "report.tex")),
    "rollup": Path(os.environ.get("GATE_ROLLUP",
                                  ROOT / "report" / "generated" / "owasp_rollup.tex")),
    "payloads": Path(os.environ.get("GATE_PAYLOADS",
                                    ROOT / "report" / "generated" / "payloads.tex")),
}
INPUT_DOC = {"generated/owasp_rollup": "rollup", "generated/payloads": "payloads"}

# One numeric atom. Order matters (longest forms first).
ATOM = re.compile(r"""
    (?P<sci>\d+(?:\.\d+)?\s*\\times\s*10\^\{?[-+]?\d+\}?)
  | (?P<sci2>\d+(?:\.\d+)?[eE][-+]?\d+(?![A-Za-z]))
  | (?P<thou>\d{1,3}(?:\{,\}\d{3})+(?!\d)|\d{1,3}(?:,\d{3})+(?![\d,]))
  | (?P<frac>\d+/\d+)
  | (?P<dec>\d*\.\d+)
  | (?P<int>\d+)
""", re.X)


def strip_comment(line: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", line)


def canon(text: str) -> str:
    t = text.replace("{,}", "").replace("\\%", "").replace("%", "")
    t = re.sub(r"\s+", "", t)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\\times10\^\{?([-+]?\d+)\}?", t)
    if m:
        return f"{m.group(1)}e{int(m.group(2))}"
    t = t.replace(",", "")
    if re.fullmatch(r"0\.\d+", t):
        t = t[1:]
    return t


class Token:
    __slots__ = ("doc", "line", "start", "end", "text", "canon", "ctx", "cls", "cat",
                 "entry", "signed", "aend", "need", "entry_cand")

    def __init__(self, doc, line, start, end, text, ctx, signed, aend):
        self.doc, self.line, self.start, self.end = doc, line, start, end
        self.aend = aend
        self.entry_cand = None
        self.need = text.replace('\\%', '').split('/') if re.fullmatch(r'\d+/\d+', text) else None
        self.text, self.canon, self.ctx, self.signed = text, canon(text), ctx, signed
        self.cls = "UNBACKED"
        self.cat = self.entry = None


def extract(doc: str, text: str) -> list[Token]:
    toks = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        for m in ATOM.finditer(line):
            a, b = m.span()
            # absorb a trailing percent sign into the token text
            tail = re.match(r"\\?%", line[b:])
            end = b + (tail.end() if tail else 0)
            pre = line[a - 1] if a else ""
            pre2 = line[a - 2] if a > 1 else ""
            signed = pre in "+-" and (a < 2 or pre2 in "$( {~\t&") and pre2 != pre
            toks.append(Token(doc, i, a, end, line[a:end], line, signed, b))
    return toks


def load_docs() -> dict[str, str]:
    docs = {"report": report_source.read_expanded(DOC_PATHS["report"])}
    for ln in docs["report"].splitlines():
        for m in re.finditer(r"\\input\{([^}]+)\}", strip_comment(ln)):
            name = INPUT_DOC.get(m.group(1))
            if name is None and not m.group(1).startswith("generated/"):
                continue   # style.tex holds layout only (no report numbers); sections/ already expanded
            if name is None:
                p = DOC_PATHS["report"].parent / (m.group(1) + ".tex")
                name = m.group(1)
                DOC_PATHS[name] = p
            docs[name] = DOC_PATHS[name].read_text()
    return docs


def where(doc: str, line: int) -> str:
    """file:line a rewriter can open: 'sections/02_task1.tex:45' for the report body (the
    expanded line is mapped back to its section file), 'generated/payloads.tex:7' otherwise."""
    if doc == "report":
        return report_source.locate(report_source.origins(DOC_PATHS["report"]), line)
    p = DOC_PATHS.get(doc)
    return f"{p.parent.name}/{p.name}:{line}" if p else f"{doc}:{line}"


# ------------------------------------------------------------------ allowlist
def _docx_path(default: Path) -> Path:
    return Path(BRIEF_DOCX_OVERRIDE) if BRIEF_DOCX_OVERRIDE else default


def _docx_text(path: Path) -> str:
    """Whitespace-normalised body text of the .docx with XML entities still encoded."""
    import zipfile
    xml = zipfile.ZipFile(path).read("word/document.xml").decode()
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))


def brief_texts(docx: Path) -> tuple[list[str], str]:
    """Texts a brief-citation pattern is searched in (entity-encoded and decoded forms),
    and a label. The .docx when it exists, else the sentences in results/brief_citations.json."""
    import html
    docx = _docx_path(docx)
    if docx.exists():
        t = _docx_text(docx)
        return [t, html.unescape(t)], str(docx)
    sents = " ".join(json.loads(BRIEF_JSON.read_text())["sentences"].values())
    return [sents, sents.replace("&", "&amp;")], f"{BRIEF_JSON.name} (brief .docx not present)"


def check_brief_citations(docx: Path) -> list[str]:
    """The stored citations must be self-consistent and, when the .docx is present, match
    it: same sha256, and every stored sentence occurs verbatim in the .docx text."""
    import hashlib
    import html
    try:
        data = json.loads(BRIEF_JSON.read_text())
    except (OSError, ValueError) as ex:
        return [f"brief citations {BRIEF_JSON}: unreadable: {ex}"]
    out = []
    sents = data.get("sentences", {})
    if not sents or not data.get("checks"):
        out.append("brief citations: no sentences or checks stored")
    for c in data.get("checks", []):
        if c.get("fragment") not in sents.get(c.get("sentence"), ""):
            out.append(f"brief citations: {c.get('id')}: fragment {c.get('fragment')!r} "
                       f"not in its stored sentence")
    docx = _docx_path(docx)
    if docx.exists():
        if hashlib.sha256(docx.read_bytes()).hexdigest() != data.get("docx_sha256"):
            out.append(f"brief citations: docx_sha256 does not match {docx.name}")
        plain = html.unescape(_docx_text(docx))
        for k, s in sents.items():
            if s not in plain:
                out.append(f"brief citations: sentence {k} is not verbatim in {docx.name}")
        print(f"brief citations: checked against {docx.name} and results/{BRIEF_JSON.name}")
    else:
        print(f"brief citations: brief .docx not present; using results/{BRIEF_JSON.name}")
    return out


def verify_entry(e: dict) -> str | None:
    """Return None when the entry's verifier passes (or it has none), else a reason."""
    v = e.get("verify")
    if not v:
        return None
    kind = v["type"]
    if kind == "grep":
        txt = (ROOT / v["file"]).read_text()
        return None if re.search(v["pattern"], txt, re.S) else \
            f"pattern {v['pattern']!r} not found in {v['file']}"
    if kind == "json":
        d = json.loads((ROOT / v["file"]).read_text())
        for k in v["path"].split("."):
            d = d[int(k)] if isinstance(d, list) else d[k]
        return None if d == v["equals"] else f"{v['file']}:{v['path']} is {d!r}"
    if kind == "docx":
        texts, src = brief_texts(ROOT / v["file"])
        return None if any(re.search(v["pattern"], t) for t in texts) else \
            f"{v['pattern']!r} not found in {src}"
    if kind == "file_exists":
        return None   # verified per match, see match_ok()
    return f"unknown verifier {kind}"


def match_ok(e: dict, text: str) -> bool:
    """Per-match verifier: a file name must exist in the repository (glob allowed)."""
    v = e.get("verify")
    if not v or v["type"] != "file_exists":
        return True
    name = text.replace("\\allowbreak{}", "").replace("\\_", "_").replace("results/", "")
    base = name.split("/")[-1]
    skip = {".venv", "chroma", "__pycache__", "node_modules"}
    return any(not (skip & set(p.parts)) for p in ROOT.rglob(base))


def load_allowlist() -> tuple[list[dict], list[str]]:
    problems, entries = [], json.loads(ALLOWLIST.read_text())["entries"]
    problems += check_brief_citations(ROOT / "docs" / "labs" / "SEC635Lab2IntegratedLabII.docx")
    seen = set()
    for e in entries:
        for k in ("id", "category", "regex", "reason"):
            if not e.get(k):
                problems.append(f"allowlist entry {e.get('id', '?')}: missing '{k}'")
        if e.get("category") not in CATEGORIES:
            problems.append(f"allowlist entry {e.get('id')}: category {e.get('category')!r} "
                            f"not in the fixed set")
        if e.get("id") in seen:
            problems.append(f"allowlist entry {e.get('id')}: duplicate id")
        seen.add(e.get("id"))
        try:
            e["_re"] = re.compile(e["regex"])
        except re.error as ex:
            problems.append(f"allowlist entry {e.get('id')}: bad regex: {ex}")
        why = verify_entry(e) if "_re" in e else None
        e["_verify_fail"] = why
        if why:
            problems.append(f"allowlist entry {e['id']}: verifier failed: {why}")
    return entries, problems


# ------------------------------------------------------------------ main
def check_generated() -> dict[str, bool]:
    """Which generated documents equal a fresh regeneration from the stored data
    (scripts/make_task2_evidence.py --check-report, pointed at the audited copies)."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for doc, name in (("rollup", "owasp_rollup.tex"), ("payloads", "payloads.tex")):
            if doc in DOC_PATHS and DOC_PATHS[doc].exists():
                (Path(td) / name).write_text(DOC_PATHS[doc].read_text())
        R = Path(os.environ.get("GATE_RESULTS", ROOT / "results"))
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_task2_evidence.py"),
                            "--check-report", "--out-dir", td,
                            "--report-tex", str(DOC_PATHS["report"]),
                            "--redteam", str(R / "task2_redteam.json"),
                            "--summary", str(R / "task2_summary.json")],
                           capture_output=True, text=True)
        txt = r.stdout
        for doc, name in (("rollup", "owasp_rollup.tex"), ("payloads", "payloads.tex")):
            out[doc] = f"OK   {name} on disk equals regenerated" in txt
    print("generated documents equal their regeneration:",
          ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in out.items()))
    return out


def run_gate(dump: Path, show: bool, baseline: bool = False) -> tuple[int, str]:
    env = os.environ.copy()
    if baseline:
        env["GATE_SKIP_CLOSURE"] = "1"
    r = subprocess.run([sys.executable, str(GATE), "--dump-coverage", str(dump)],
                       capture_output=True, text=True, env=env)
    if show:
        print(r.stdout, end="")
    return r.returncode, r.stdout


def main() -> int:
    list_all = "--list-all" in sys.argv
    docs = load_docs()
    tokens = {d: extract(d, t) for d, t in docs.items()}
    entries, problems = load_allowlist()

    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "coverage.json"
        baseline = "--baseline" in sys.argv
        rc, out = run_gate(dump, "--show-gate" in sys.argv, baseline)
        summary = [l for l in out.splitlines() if "checks run" in l or l.startswith("report")]
        cov = json.loads(dump.read_text()) if dump.exists() else {"entries": []}
        if baseline:   # only what the original literal-matching helpers registered
            cov["entries"] = [e for e in cov["entries"] if e["kind"] in ("check", "int_cell")]
    print("gate 1:", "; ".join(summary) if summary else "no summary", f"(exit {rc})")

    # ---- allowlist candidates (a token is a candidate when an entry's regex, matched on
    # its own line, contains the token and the entry's verifier passes)
    used = Counter()
    for d, ts in tokens.items():
        for t in ts:
            for e in entries:
                if "_re" not in e or e["_verify_fail"]:
                    continue
                if e.get("doc", "*") not in (d, "*"):
                    continue
                if e.get("anchor") and e["anchor"] not in t.ctx:
                    continue
                if e.get("line") and e["line"] != t.line:
                    continue
                if any(m.start() <= t.start and t.aend <= m.end() and match_ok(e, m.group(0))
                       for m in e["_re"].finditer(t.ctx)):
                    t.entry_cand = e
                    break

    # ---- COVERED: each registry entry consumes at most one token per literal atom, on one
    # of the lines it was matched on; tokens that are not allowlist candidates are preferred
    # so that "PGD-40" (a name) is not consumed by the check of "40-step" (a setting)
    by_line = defaultdict(list)
    for d, ts in tokens.items():
        for t in ts:
            by_line[(d, t.line)].append(t)

    def pick(e, pred):
        cands = [t for ln in sorted(e["lines"]) for t in by_line.get((e["doc"], ln), [])
                 if t.cls == "UNBACKED" and pred(t)]
        cands.sort(key=lambda t: t.entry_cand is not None)
        return cands[0] if cands else None

    unlocated = unmatched = 0
    for e in cov["entries"]:
        atoms = [canon(m.group(0)) for m in ATOM.finditer(e["literal"])]
        if not e["doc"] or not e["lines"]:
            unlocated += 1
            continue
        for atom in atoms:
            hit = pick(e, lambda t: t.canon == atom)
            if hit is None:   # one component of a fraction like 0/5 verified separately
                part = pick(e, lambda t: t.need and atom in t.need)
                if part is not None:
                    part.need.remove(atom)
                    if not part.need:
                        part.cls, part.entry = "COVERED", "gate"
                    continue
            if hit is None and any(t.canon == atom for ln in e["lines"]
                                   for t in by_line.get((e["doc"], ln), [])):
                continue   # the same token was already verified by an earlier check
            if hit is None:
                unmatched += 1
                if "--debug" in sys.argv:
                    print("  unmatched registry literal:", atom, e["doc"], e["lines"], e["kind"])
            else:
                hit.cls, hit.entry = "COVERED", "gate"

    # ---- generated documents: byte-equal to a regeneration from the stored data
    gen_ok = {} if baseline else check_generated()
    for d, ts in tokens.items():
        for t in ts:
            if t.cls != "UNBACKED":
                continue
            e = getattr(t, "entry_cand", None)
            if e is not None and (not e.get("max") or used[e["id"]] < e["max"]):
                t.cls, t.cat, t.entry = "ALLOWLISTED", e["category"], e["id"]
                used[e["id"]] += 1
            elif gen_ok.get(d):
                t.cls, t.entry = "COVERED", "generated"

    all_t = [t for ts in tokens.values() for t in ts]
    cls = Counter(t.cls for t in all_t)
    cat = Counter(t.cat for t in all_t if t.cls == "ALLOWLISTED")
    print(f"\nnumeric tokens: {len(all_t)}  "
          f"({', '.join(f'{d} {len(ts)}' for d, ts in tokens.items())})")
    for c in ("COVERED", "ALLOWLISTED", "UNBACKED"):
        print(f"  {c:<12} {cls[c]}")
    src = Counter(t.entry for t in all_t if t.cls == "COVERED")
    print(f"  covered by gate 1 checks: {src['gate']}; by generated-file regeneration: "
          f"{src['generated']}")
    print("allowlisted by category:")
    for c in CATEGORIES:
        print(f"  {c:<38} {cat[c]}")
    print(f"gate registry: {len(cov['entries'])} entries, {unlocated} without a source "
          f"location, {unmatched} literal(s) with no matching token on their line(s)")
    stale = [e["id"] for e in entries if not used[e["id"]]]
    if stale:
        print("allowlist entries that matched nothing (stale?):", ", ".join(stale))
    if "--emit-tokens" in sys.argv:   # used by the mutation sweep in the audit notes
        out = Path(sys.argv[sys.argv.index("--emit-tokens") + 1])
        out.write_text(json.dumps([{"doc": t.doc, "line": t.line, "start": t.start,
                                    "end": t.aend, "where": where(t.doc, t.line), "text": t.text, "cls": t.cls,
                                    "entry": t.entry, "cat": t.cat} for t in all_t]))
    if list_all:
        for t in all_t:
            print(f"  {t.cls[:4]} {where(t.doc, t.line):<34} {t.text:<22} {t.cat or ''} {t.entry or ''}")
    bad = [t for t in all_t if t.cls == "UNBACKED"]
    if bad:
        print(f"\nUNBACKED ({len(bad)}):")
        for t in bad:
            lo = max(0, t.start - 50)
            print(f"  {where(t.doc, t.line)}: {t.text!r}  ...{t.ctx[lo:t.end + 40].strip()}...")
    for p in problems:
        print("PROBLEM:", p)
    if rc != 0:
        print("\nFAIL: gate 1 itself failed")
    print(f"\nnumber coverage: {len(bad)} UNBACKED")
    return 1 if (bad or problems or rc != 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
