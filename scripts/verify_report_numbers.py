"""Verify every number printed in report/report.tex against the results/*.json files.

Prose and tables can drift from the stored results, and reading them by eye is not a
reliable way to catch that. Every
check below parses the number out of the LaTeX and compares it to the JSON the run
produced. Exit code is non-zero if any claim does not match, so this is a gate, not a
report.

    .venv/bin/python scripts/verify_report_numbers.py

It checks the numbers the report *prints*. It cannot check a claim that has no number
(e.g. "the ordering is the result"), and it cannot detect a number that is correct in
the JSON but wrong in the science - that is what detector/ tests and the ablations are
for.
"""

from __future__ import annotations

import hashlib
import json
import re as _re_real
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_source   # noqa: E402  (shared expander: report.tex + sections/*.tex)

TEX_PATH = Path(os.environ.get("GATE_TEX", ROOT / "report" / "report.tex"))
TEX = report_source.read_expanded(TEX_PATH)     # every gate below sees the EXPANDED report
STYLE_TEX = (TEX_PATH.parent / "style.tex").read_text() if (TEX_PATH.parent / "style.tex").exists() else ""
ROLL = Path(os.environ.get("GATE_ROLLUP",
                           ROOT / "report" / "generated" / "owasp_rollup.tex")).read_text()
R = Path(os.environ.get("GATE_RESULTS", ROOT / "results"))
PAYL = Path(os.environ.get("GATE_PAYLOADS",
                           ROOT / "report" / "generated" / "payloads.tex")).read_text()

FAILS: list[str] = []
_CURRENT = ["(startup)"]     # name of the check function now running
NCHECK = 0

# ---------------------------------------------------------------------------------------
# Coverage instrumentation (scripts/number_coverage.py reads the dump).
#
# Every string the gate slices out of report.tex / the generated files is a `Src`: a str
# that remembers which document and which source lines it came from. Slicing, stripping,
# splitting, regex groups and concatenation keep that provenance. When a check verifies a
# literal (check / int_cell_check / cover), the literal and its source lines are appended
# to REGISTRY. The registry is only a record; it never changes a pass/fail decision.
# ---------------------------------------------------------------------------------------
REGISTRY: list[dict] = []


class Src(str):
    """str that remembers its source document and lines. `lmap` (one source line per
    character) is kept through slicing and strip when known; any other operation falls
    back to the union of lines of the string it came from."""
    doc: str = ""
    _lines: frozenset = frozenset()
    lmap = None

    @property
    def lines(self) -> frozenset:
        return frozenset(self.lmap) if self.lmap is not None else self._lines

    def _w(self, s):
        return _mk(s, self.doc, self.lines)

    def _cut(self, a, b):
        if self.lmap is not None:
            return _mk(str.__getitem__(self, slice(a, b)), self.doc, (), self.lmap[a:b])
        return self._w(str.__getitem__(self, slice(a, b)))

    def strip(self, *c):
        s = str(self)
        a = len(s) - len(s.lstrip(*c))
        return self._cut(a, len(s.rstrip(*c)) if s.strip(*c) else a)

    def lstrip(self, *c):
        s = str(self)
        return self._cut(len(s) - len(s.lstrip(*c)), len(s))

    def rstrip(self, *c):
        s = str(self)
        return self._cut(0, len(s.rstrip(*c)))

    def replace(self, *a):
        return self._w(str.replace(self, *a))

    def lower(self):
        return self._w(str.lower(self))

    def __getitem__(self, k):
        if isinstance(k, slice) and k.step in (None, 1) and self.lmap is not None:
            a, b, _ = k.indices(len(self))
            return self._cut(a, b)
        return self._w(str.__getitem__(self, k))

    def split(self, *a):
        return [self._w(x) for x in str.split(self, *a)]

    def splitlines(self, *a):
        return [self._w(x) for x in str.splitlines(self, *a)]

    def __add__(self, o):
        return self._w(str.__add__(self, str(o)))

    def __radd__(self, o):
        return self._w(str(o) + str(self))


def _mk(s, doc, lines=(), lmap=None):
    o = Src(s)
    o.doc = doc
    o._lines = frozenset(lines)
    o.lmap = tuple(lmap) if lmap is not None else None
    return o


def _prov(x):
    return (x.doc, sorted(x.lines)) if isinstance(x, Src) else (None, [])


def cover(scope, literals, kind="cover") -> None:
    """Record that `literals` (strings or one string) were verified inside `scope`.
    Call ONLY next to a real comparison that fails the gate if the value is wrong."""
    doc, lines = _prov(scope)
    if isinstance(literals, str):
        literals = [literals]
    for lit in literals:
        d, ls = _prov(lit)
        REGISTRY.append({"doc": d or doc, "lines": ls or lines, "literal": str(lit),
                         "kind": kind})


class _M:
    def __init__(self, m):
        self._m = m

    def group(self, *a):
        s = self._m.string
        gs = a or (0,)

        def one(g):
            r = self._m.group(g)
            if isinstance(s, Src) and r is not None:
                return s[self._m.start(g):self._m.end(g)]
            return r

        out = tuple(one(g) for g in gs)
        return out if len(out) > 1 else out[0]

    def __getattr__(self, k):
        return getattr(self._m, k)


def _wrap_like(src, x):
    return _mk(x, src.doc, src.lines) if isinstance(src, Src) and isinstance(x, str) else x


class _Re:
    """`re` look-alike whose results keep the provenance of the subject string."""

    def __getattr__(self, k):
        return getattr(_re_real, k)

    def search(self, pat, s, *a):
        m = _re_real.search(pat, s, *a)
        return _M(m) if m else None

    def match(self, pat, s, *a):
        m = _re_real.match(pat, s, *a)
        return _M(m) if m else None

    def fullmatch(self, pat, s, *a):
        m = _re_real.fullmatch(pat, s, *a)
        return _M(m) if m else None

    def finditer(self, pat, s, *a):
        return (_M(m) for m in _re_real.finditer(pat, s, *a))

    def findall(self, pat, s, *a):
        return [_wrap_like(s, x) for x in _re_real.findall(pat, s, *a)]

    def sub(self, pat, rep, s, *a, **k):
        return _wrap_like(s, _re_real.sub(pat, rep, s, *a, **k))

    def split(self, pat, s, *a, **k):
        return [_wrap_like(s, x) for x in _re_real.split(pat, s, *a, **k)]


re = _Re()
_DOCS = {"report": TEX, "rollup": ROLL, "payloads": PAYL}


def _doc_lines(doc: str) -> list:
    """Every line of a document as a Src carrying its own line number."""
    return [_mk(ln, doc, {i}) for i, ln in enumerate(_DOCS[doc].splitlines(), 1)]


def _paras(doc: str = "report") -> list:
    """Paragraphs (split on blank lines), whitespace-squashed, as Src with their lines."""
    text, out = _DOCS[doc], []
    cuts = [(m.start(), m.end()) for m in _re_real.finditer(r"\n\s*\n", text)]
    starts = [0] + [e for _, e in cuts]
    ends = [s for s, _ in cuts] + [len(text)]
    for a, b in zip(starts, ends):
        l0 = text.count("\n", 0, a) + 1
        l1 = l0 + text.count("\n", a, b)
        chars, lm = [], []
        for tk in _re_real.finditer(r"\s+|\S+", text[a:b]):
            ln = l0 + text.count("\n", a, a + tk.start())
            if tk.group().isspace():
                chars.append(" ")
                lm.append(ln)
            else:
                chars.append(tk.group())
                lm.extend([l0 + text.count("\n", a, a + tk.start() + i)
                           for i in range(len(tk.group()))])
        out.append(_mk("".join(chars), doc, (), lm))
    return out


def _slice(a_str: str, b_str: str, doc: str = "report", length: int | None = None) -> "Src":
    """text[index(a_str):index(b_str)] (or `length` chars) as a Src with its lines."""
    text = _DOCS[doc]
    global NCHECK
    NCHECK += 1
    a = text.find(a_str)
    if a < 0:
        FAILS.append(f"{_CURRENT[0]}: slice start anchor '{a_str}' not found in the report")
        return _mk("", doc, ())
    if length is not None:
        b = a + length
    else:
        b = text.find(b_str, a)
        if b < 0:
            FAILS.append(f"{_CURRENT[0]}: slice end anchor '{b_str}' not found after '{a_str}'")
            return _mk("", doc, ())
    l0 = text.count("\n", 0, a) + 1
    return _mk(text[a:b], doc, range(l0, l0 + text.count("\n", a, b) + 1))


def check(label: str, tex_str: str, json_val: float, tol: float | None = None,
          scope: str | None = None) -> None:
    """Assert the literal `tex_str` appears in the report and rounds to `json_val`.

    `scope` is the text to search (a row or sentence located by an anchor). When it is
    None the WHOLE document is searched, which lets the same literal elsewhere mask a
    corrupted copy here; prefer a scope (see anchored_line / table_rows).

    Comparison is at the precision the report prints: "56.5" is checked as
    round(json_val, 1) == 56.5, not as an absolute tolerance, so a correctly rounded
    figure never fails and a wrong digit always does.
    """
    global NCHECK
    NCHECK += 1
    _register(tex_str, scope)
    present = tex_str in (TEX if scope is None else scope)
    plain = tex_str.replace(",", "").replace("_", "")
    decimals = len(plain.split(".")[1]) if "." in plain else 0
    try:
        agree = abs(round(float(json_val), decimals) - float(plain)) < 10 ** -(decimals + 4)
        if tol is not None:
            agree = abs(float(plain) - float(json_val)) <= tol
    except ValueError:
        agree = False
    if not present:
        FAILS.append(f"{label}: literal '{tex_str}' not found in report.tex")
    elif not agree:
        FAILS.append(f"{label}: report says {tex_str}, JSON says {json_val}")


def _register(lit, scope=None, kind="check") -> None:
    src = lit if isinstance(lit, Src) and lit.lines else scope
    doc, lines = _prov(src)
    REGISTRY.append({"doc": doc, "lines": lines, "literal": str(lit), "kind": kind})


def _squash(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def anchored_line(label: str, anchor: str) -> str | None:
    """The single report line containing `anchor` (whitespace-squashed). Zero or several
    matches is a FAILURE (returns None), never a silent pass."""
    global NCHECK
    NCHECK += 1
    hits = [ln for ln in _doc_lines("report")
            if not ln.lstrip().startswith("%") and anchor in _squash(ln)]
    if len(hits) != 1:
        FAILS.append(f"{label}: anchor '{anchor}' matched {len(hits)} report lines, "
                     f"expected exactly 1")
        return None
    return _squash(hits[0])


def _strip_bold(t: str) -> str:
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", t)
    return t


def table_rows(label: str, marker: str) -> list[str] | None:
    """Logical rows (a row may span lines, ends at `\\\\`) of the one tabular that
    contains `marker`, from the marker line to \\end{tabular}. Marker must be unique."""
    global NCHECK
    NCHECK += 1
    lines = _doc_lines("report")
    starts = [i for i, ln in enumerate(lines) if marker in ln]
    if len(starts) != 1:
        FAILS.append(f"{label}: table marker '{marker}' matched {len(starts)} lines, "
                     f"expected exactly 1")
        return None
    rows, cur, cur_l = [], "", set()
    for ln in lines[starts[0]:]:
        if "\\end{tabular}" in ln:
            break
        if ln.lstrip().startswith(("%", "\\midrule", "\\toprule", "\\cmidrule",
                                   "\\bottomrule")):
            continue
        cur += " " + ln.strip()
        cur_l |= ln.lines
        if cur.rstrip().endswith("\\\\"):
            rows.append(_mk(_squash(cur).strip(), "report", cur_l))
            cur, cur_l = "", set()
    return rows


def table_row(label: str, rows: list[str] | None, prefix: str) -> list[str] | None:
    """Cells of the single row starting with `prefix` (bold stripped); FAIL if absent
    or duplicated."""
    global NCHECK
    NCHECK += 1
    if rows is None:
        return None
    hits = [r for r in rows if _strip_bold(r).startswith(prefix)]
    if len(hits) != 1:
        FAILS.append(f"{label}: row anchor '{prefix}' matched {len(hits)} rows, "
                     f"expected exactly 1")
        return None
    body = _strip_bold(hits[0])
    body = re.sub(r"\\\\\s*$", "", body)
    return [c.strip() for c in re.split(r"(?<!\\)&", body)]


def cell_check(label: str, cell_txt: str, json_val: float, tol: float | None = None) -> None:
    """Check ONE table cell (already isolated from its row) at its printed precision."""
    check(label, cell_txt, json_val, tol, scope=cell_txt)


def int_cell_check(label: str, cell_txt: str, json_val: int) -> None:
    global NCHECK
    NCHECK += 1
    _register(cell_txt, kind="int_cell")
    plain = cell_txt.replace("{,}", "").replace(",", "")
    if not plain.isdigit() or int(plain) != int(json_val):
        FAILS.append(f"{label}: report says {cell_txt}, JSON says {json_val}")


def row_len(label: str, cells: list[str] | None, n: int) -> list[str] | None:
    if cells is not None and len(cells) != n:
        FAILS.append(f"{label}: row has {len(cells)} cells, expected {n}: {cells}")
        return None
    return cells


def scoped_num(label: str, anchor: str, items) -> None:
    """Locate ONE report line by `anchor`, pull each number out of it with a regex
    (group 1) and check it against the JSON. Missing anchor / pattern is a FAILURE."""
    ln = anchored_line(label, anchor)
    if ln is None:
        return
    for lab, pat, val, tol in items:
        m = re.search(pat, ln)
        if not m:
            FAILS.append(f"{lab}: pattern '{pat}' not found in the '{anchor}' line")
            continue
        cell_check(lab, m.group(1), val, tol)


def cell(curve, eps, attack, key="detection_rate"):
    for r in curve:
        if abs(r["eps_std"] - eps) < 1e-9:
            return r["attacks"][attack][key]
    raise KeyError(f"eps={eps} attack={attack} not in curve")


def load(name):
    return json.loads((R / name).read_text())


def task1a():
    d = load("task1a_membership.json")
    days = {x["day"]: x for x in d["days"]}
    for day, acc, blind in (("Wednesday", "0.5352", "0.5472"),
                            ("Thursday", "0.5280", "0.5703")):
        x = days[day]
        a = x["arms"][0]["attacks"]["confidence_threshold"]["accuracy"]
        b = x["arms"][1]["attacks"]["confidence_threshold"]["accuracy"]
        ln = anchored_line(f"1A {day} table row", f"{day} & confidence / margin / loss")
        if ln is None:
            continue
        cells = [_strip_bold(c).strip() for c in ln.replace("\\\\", "").split("&")]
        if len(cells) != 5:
            FAILS.append(f"1A {day}: row has {len(cells)} cells, expected 5")
            continue
        cell_check(f"1A {day} MI accuracy", cells[2], a)
        cell_check(f"1A {day} blind control", cells[4], b)
        # the report's central claim: the control beats the target model
        if not b > a:
            FAILS.append(f"1A {day}: blind control {b} is NOT above target {a}")


def task1b():
    """1B body (cost line, Thursday sample, tables, benign-to-attack, realism, attacker
    knowledge) is checked sentence- and row-wise in task1_text()."""
    d = load("task1b_evasion_wednesday.json")
    if d.get("n_eval_flows") != 300:
        FAILS.append(f"1B n_eval_flows is {d.get('n_eval_flows')}, expected 300")
    NCHECK_inc()
    if d.get("n_mutable_features") != 52:
        FAILS.append(f"1B n_mutable_features {d.get('n_mutable_features')} != 52")
    NCHECK_inc()


def NCHECK_inc():
    global NCHECK
    NCHECK += 1


def task1c_defense():
    b = load("task1b_evasion_wednesday.json")
    a = load("task1c_defense.json")
    rows = table_rows("1C table", "Before (Lab~1 model)")
    for eps in (0.05, 0.10, 0.15, 0.25, 0.50):
        lab = f"1C defense eps={eps:.2f}"
        c = row_len(lab, table_row(lab, rows, f"{eps:.3f} &"), 7)
        if c is None:
            continue
        atts = ("random_search", "fd_fgsm", "pgd")
        cell_check(f"{lab} eps cell", c[0], eps)
        for k, x in enumerate(atts):
            cell_check(f"{lab} before {x}", c[1 + k], cell(b["curve"], eps, x))
            cell_check(f"{lab} after {x}", c[4 + k], cell(a["curve_after"], eps, x))
    # (the 1C sentences are checked in task1c_text(); the invariant below stays)
    s = a["smoothing"]
    nm = load("task1b_evasion_wednesday.json").get("n_mutable_features", 52)
    NCHECK_inc()
    if not s["radius_median"] < 0.15 * (nm ** 0.5):
        FAILS.append("1C: certificate is NOT below the attack's L2 extent; "
                     "the report's conclusion would be wrong")


def task2():
    d = load("task2_redteam.json")
    before = {x["key"]: x for x in d["before"]}
    after = {x["key"]: x for x in d["after"]}
    # the per-attack table is the generated roll-up that report.tex \\input's
    NCHECK_inc()
    if "\\input{generated/owasp_rollup}" not in TEX:
        FAILS.append("2: report.tex does not \\input the generated OWASP roll-up table")
    NCHECK_inc()
    if "\\input{generated/payloads}" not in TEX:
        FAILS.append("2: report.tex does not \\input the generated payload appendix")
    norm_lines = [l.replace("\\textbf{", "").replace("}", "").replace("\\quad ", "")
                  for l in _doc_lines("rollup")]
    # The report's own numbers are the source of truth for what it claims; the JSON is
    # the source of truth for what happened. Both must agree, and the counts must be
    # internally consistent with the per-trial records. The comparison is ROW-SCOPED:
    # searching the whole document for "10/10 & 0/10" lets one row satisfy another row's
    # check, which silently passed a deliberately corrupted kb-override count.
    row_label = {
        "2A-direct-override": "Direct instruction override",
        "2A-direct-secret": "Direct secret extraction",
        "2A-direct-prompt-exfil": "Direct prompt exfiltration",
        "2A-kb-override": "KB-planted override",
        "2A-kb-secret": "KB-planted secret theft",
        "2A-kb-misinfo": "Fabricated technique",
        "2B-roleplay": "Role-play exploit code",
        "2B-base64": "Base64 smuggle",
        "2B-manyshot": "Many-shot jailbreak",
    }
    plain_rows = [ln for ln in norm_lines
                  if "&" in ln and "/10" in ln and not ln.startswith("%")]
    for key in before:
        NCHECK_inc()
        b, a = before[key]["successes"], after[key]["successes"]
        label = row_label[key]
        row = next((ln for ln in plain_rows if ln.startswith(label)), None)
        if row is None:
            FAILS.append(f"2 {key}: no report row starting '{label}'")
        elif f"{b}/10 $\\to$ {a}/10" not in row:
            FAILS.append(f"2 {key}: JSON says {b}/10 before, {a}/10 after but the "
                         f"'{label}' row reads: {row.strip()[:90]}")
        for side, rec in (("before", before[key]), ("after", after[key])):
            recount = sum(1 for tr in rec["trials"] if tr["success"])
            if recount != rec["successes"]:
                FAILS.append(f"2 {key} {side}: successes={rec['successes']} but "
                             f"{recount} trials flagged success")
            if rec["effective_runs"] != rec["runs"] - rec["errors"]:
                FAILS.append(f"2 {key} {side}: errors not excluded from denominator")
            # a success must be backed by a non-refused, non-blocked trial
            for tr in rec["trials"]:
                if tr["success"] and (tr["refusal"] or tr["blocked_input"]
                                      or tr["blocked_output"]):
                    FAILS.append(f"2 {key} {side} run {tr['run']}: counted success "
                                 f"while refused/blocked")
    ben = d["benign"]["after"]
    if ben["guardrail_refusal_rate"] != 0.0:
        FAILS.append(f"2 benign guardrail refusals {ben['guardrail_refusal_rate']} "
                     f"but report claims 0/50")
    # Attribution: the report names the layer that stopped each attack. That claim was
    # wrong for four of nine rows once already, so both sides are checked: what the JSON's
    # block_layers actually say, AND that the report's "Stopped by" cell says the same
    # thing. Checking only the JSON makes the gate vacuous - it passes whatever the report
    # claims. (row label in report.tex, expected phrase, json key)
    attribution = [
        ("Direct instruction override", "input filter", "2A-direct-override"),
        ("Direct secret extraction", "input filter", "2A-direct-secret"),
        ("Direct prompt exfiltration", "input filter", "2A-direct-prompt-exfil"),
        ("KB-planted override", "no layer fired", "2A-kb-override"),
        ("KB-planted secret theft", "output filter", "2A-kb-secret"),
        ("Role-play exploit code", "Llama Guard", "2B-roleplay"),
        ("Base64 smuggle", "input filter", "2B-base64"),
        ("Many-shot jailbreak", "Llama Guard", "2B-manyshot"),
    ]
    plain_lines = norm_lines
    for label, want, key in attribution:
        NCHECK_inc()
        layers = [t.get("block_layers") or [] for t in after[key]["trials"]]
        blocked = [l for l in layers if l]
        # (a) what the JSON says
        if want == "no layer fired":
            json_ok = (not blocked) and after[key]["successes"] == 0
        elif want == "Llama Guard":
            json_ok = bool(blocked) and all(
                any("llama_guard" in x for x in l) and not any("input_filter" in x for x in l)
                for l in blocked)
        elif want == "input filter":
            json_ok = bool(blocked) and all(any("input_filter" in x for x in l)
                                            for l in blocked)
        else:
            json_ok = bool(blocked)
        if not json_ok:
            FAILS.append(f"2 {key}: JSON block_layers do not support '{want}': {layers[:2]}")
        # (b) that the report's own cell says it
        row = next((ln for ln in plain_lines if ln.startswith(label)), None)
        if row is None:
            FAILS.append(f"2 {key}: report row '{label}' not found")
        elif want not in row:
            FAILS.append(f"2 {key}: report says the wrong blocker for '{label}' "
                         f"(expected '{want}' in the Stopped-by column)")
    # kb-secret is split across two layers; the report must show the split, not one layer
    NCHECK_inc()
    n_out = sum(1 for t in after["2A-kb-secret"]["trials"]
                if any("secret_code" in x for x in (t.get("block_layers") or [])))
    krow = next((ln for ln in plain_lines
                 if ln.startswith("KB-planted secret theft")), "")
    if f"output filter ({n_out}/10)" not in krow:
        FAILS.append(f"2 2A-kb-secret: {n_out}/10 blocked by the output filter; the "
                     f"report must state that split rather than a single layer")
    # F3 is the report's headline unmitigated finding; verify it is still true
    NCHECK_inc()
    if not (before["2A-kb-misinfo"]["successes"] == 10
            and after["2A-kb-misinfo"]["successes"] == 10):
        FAILS.append("2: kb-misinfo is no longer 10/10 before AND after; the report's "
                     "F3 'not mitigated' claim would be wrong")


def cross_file():
    """The before/after curves are only comparable if they share one flow set."""
    prov = {}
    for name in ("task1b_evasion_wednesday.json", "task1c_defense.json",
                 "task1c_ablation.json", "task1c_ablation_rounds.json"):
        p = load(name).get("eval_set_provenance")
        if p is None:
            FAILS.append(f"provenance: {name} has no eval_set_provenance")
            continue
        prov[name] = (p["sha256_of_flow_matrix"], p["n"])
    vals = set(prov.values())
    NCHECK_inc()
    if len(vals) > 1:
        FAILS.append(f"provenance: Task 1 JSONs disagree on the eval set: {prov}")
    if prov and list(prov.values())[0][1] != 300:
        FAILS.append("provenance: eval set is not 300 flows")


def scale_ref():
    """Recompute, from the raw stored values, what scripts/scale_effect.py --augment stores in
    task1c_scale_effect.json under `l2_extent` and `evasion_reference`; FAIL if the stored
    blocks differ. Returns the recomputed numbers. Used by every sentence that quotes them."""
    import math
    se = load("task1c_scale_effect.json")
    b1 = load("task1b_evasion_wednesday.json")
    df = load("task1c_defense.json")
    pf = se["per_feature"]
    n = len(pf)
    out = {"n_mutable": n,
           "l2_def": math.sqrt(sum(v["std_move_defended"] ** 2 for v in pf.values())),
           "l2_naive": se["eps_std"] * math.sqrt(n),
           "cert": df["smoothing"]["radius_median"], "ref": {}}
    for row in se["pgd_scale_test"]["curve"]:
        e = row["eps_std"]
        u = 1.0 - cell(b1["curve"], e, "pgd")
        real, own = row["all_mutable_real_box"]["evasion_rate"], row["all_mutable_five_own_units"]["evasion_rate"]
        out["ref"][e] = {"und": u, "real": real, "own": own,
                         "share": (real - own) / (real - u) if real > u else None,
                         "mostly": real > u and (real - own) / (real - u) > 0.5,
                         "indep": real == own}
    NCHECK_inc()
    st = {r["eps_std"]: r for r in se.get("evasion_reference", [])}
    ok = (abs(se.get("l2_extent", {}).get("l2_defended_units", -1) - out["l2_def"]) < 1e-9
          and abs(se["l2_extent"]["l2_naive_eps_sqrt_n"] - out["l2_naive"]) < 1e-9
          and set(st) == set(out["ref"]) and all(
              abs(st[e]["undefended_pgd_evasion"] - r["und"]) < 1e-9 and st[e]["mostly_scale_effect"] == r["mostly"]
              and st[e]["scale_independent"] == r["indep"] for e, r in out["ref"].items()))
    if not ok:
        FAILS.append("scale_effect.json: stored l2_extent/evasion_reference disagree with the values recomputed from per_feature/pgd_scale_test/task1b")
    return out


# =======================================================================================
# Executive summary (sections/01_summary.tex) and risk assessment (sections/04_risk.tex).
# Every number is compared with results/*.json at the precision the text prints; derived
# values (severity, likelihood, residual rating, priority order, counts) are recomputed
# here from the stored results and from the rule the report states, never typed in.
# =======================================================================================
EXEC_HDR = "\\textbf{Finding} & \\textbf{Severity} & \\textbf{Status"
RISK_HDR = "\\textbf{Priority} & \\textbf{Finding}"
FKEY = {"F1": "2A-kb-override", "F2": "2A-kb-secret", "F3": "2A-kb-misinfo", "F4": "2B-roleplay"}
FTAGS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
RISK7 = {}            # kept for older callers: tag -> parsed risk row (dict) once risk_table() ran
SCALE = {"H": 3, "M": 2, "L": 1}
_RANK_SEV = {"High": 0, "Medium": 1, "Low": 2}
_RANK_STATUS = {"Not mitigated": 0, "Partly": 1, "Mitigated": 2, "n/a": 3}
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_T: dict = {}


def risk_rule(impact: str, likelihood: str) -> str:
    """Severity from impact x likelihood: High for H/H, H/M, M/H; Medium for M/M, H/L, L/H;
    Low for M/L, L/M, L/L (score H=3, M=2, L=1; sum >= 5 High, 4 Medium, else Low)."""
    t = SCALE[impact] + SCALE[likelihood]
    return "High" if t >= 5 else "Medium" if t == 4 else "Low"


def _sec_no(prefix: str):
    titles = re.findall(r"\\section\{([^}]*)\}", str(TEX))
    for i, t in enumerate(titles, 1):
        if t.startswith(prefix):
            return i
    return None


def _frac(label: str, lit, k, n) -> None:
    """The literal 'k/n' (a table or sentence token) must equal the stored counts."""
    NCHECK_inc()
    if str(lit) != f"{k}/{n}":
        FAILS.append(f"{label}: report says {lit}, results say {k}/{n}")
    else:
        cover(lit, [lit], "frac")


def _eps_lit(label: str, lit, grid) -> None:
    """A change-size label printed in the text must be a point of the stored epsilon grid."""
    NCHECK_inc()
    try:
        ok = any(abs(float(lit) - g) < 1e-9 for g in grid)
    except ValueError:
        ok = False
    if not ok:
        FAILS.append(f"{label}: {lit} is not a point of the stored grid {sorted(grid)}")
    else:
        cover(lit, [lit], "eps")


def _agree(lit, val) -> bool:
    """Same rounding the text uses, no registration (second reading of an already checked token)."""
    s = str(lit).replace(",", "")
    dec = len(s.split(".")[1]) if "." in s else 0
    try:
        return abs(round(float(val), dec) - float(s)) < 10 ** -(dec + 4)
    except ValueError:
        return False


def _line(label: str, anchor: str):
    return anchored_line(label, anchor)


def _rx(label: str, pat: str, text):
    """Search `pat` in `text`; a miss is a FAIL naming the check, never a skip."""
    NCHECK_inc()
    if text is None:
        return None
    m = re.search(pat, text)
    if not m:
        FAILS.append(f"{label}: sentence not in the expected form (pattern '{pat[:70]}...')")
    return m


def _truth() -> dict:
    """Everything the checks compare against, read or computed from results/*.json."""
    if _T:
        return _T
    import math
    sm, zb = load("task2_summary.json"), load("task2_zero_rate_bounds.json")
    rt = load("task2_redteam.json")
    b1, df = load("task1b_evasion_wednesday.json"), load("task1c_defense.json")
    abl = {c["config"]: c for c in load("task1c_ablation.json")["configs"]
           + load("task1c_ablation_rounds.json")["configs"]}
    pr, ctl, bud = load("task2_retrieval_probe.json"), load("task5_recommendation_controls.json"), \
        load("task1_attacker_budget.json")
    ex, ops = load("task1_extras.json"), load("ops_measurements.json")
    d1a = {x["day"]: x for x in load("task1a_membership.json")["days"]}["Wednesday"]["arms"]
    t = {"sm": sm, "zb": zb, "b1": b1, "df": df, "abl": abl, "pr": pr, "ctl": ctl, "bud": bud, "ops": ops,
         "pa": {p["key"]: p for p in sm["per_attack"]},
         "zcell": {x["cell"]: x for x in zb["attack_cells_with_zero_successes"]},
         "channel": {r["key"]: r["channel"] for r in rt["before"]},
         "runs": rt["runs_per_attack"], "grid": {r["eps_std"] for r in b1["curve"]},
         "thu": ex["blocks"]["thursday_evasion"]["variants"]["B_filter_then_sample300"],
         "mi_attack": d1a[0]["attacks"]["confidence_threshold"]["accuracy"],
         "mi_control": d1a[1]["attacks"]["confidence_threshold"]["accuracy"]}
    t["rec"] = abl["full_1r_frozen"]
    fd_min, thr = ctl["cheapest_attack_queries_per_flow"], ctl["alert_threshold_near_identical_vectors"]
    t["fd_min"], t["thr"], t["rs_max"] = fd_min, thr, ctl["random_search_queries_per_flow_max"]
    t["thr_calc"] = math.floor(ctl["alert_fraction"] * fd_min)
    # likelihood from the stored data, using the rule the report states
    t["lik"] = {}
    for tag, key in FKEY.items():
        p = t["pa"][key]
        t["lik"][tag] = ("H", p["before_successes"] / p["n_before"])    # threshold applied in risk_rule_checks
    hits = pr["b2_fixed_kev_probe"]["n_exact_doc_in_top5"] + pr["b1_benign_questions_naming_ids"]["n_exact_doc_in_top5"]
    nprb = pr["b2_fixed_kev_probe"]["n_probes"] + pr["b1_benign_questions_naming_ids"]["n_naming_an_id"]
    t["f7_fail_rate"] = 1 - hits / nprb
    t["det_lik"] = "H" if fd_min < thr else "M"
    t["f6_lik"] = "L" if t["mi_attack"] <= t["mi_control"] else "M"
    _T.update(t)
    return _T


def _cells_of(row):
    body = re.sub(r"\\\\\s*$", "", _strip_bold(row))
    return [c.strip() for c in re.split(r"(?<!\\)&", body)]


def _exec_cells() -> dict:
    """tag -> the four cells of the executive-summary table (Finding, Severity, Status, Basis)."""
    if "exec" in _T:
        return _T["exec"]
    rows = table_rows("exec table", EXEC_HDR)
    out = {}
    for r in (rows or [])[1:]:
        c = _cells_of(r)
        m = re.match(r"(F\d) ", c[0])
        if len(c) != 4 or not m:
            FAILS.append(f"exec table: row not parseable (need 4 cells starting 'F<n> '): {str(r)[:60]}")
            continue
        out[m.group(1)] = c
    NCHECK_inc()
    if rows is not None and sorted(out) != list(FTAGS):
        FAILS.append(f"exec table: findings present {sorted(out)}, expected {list(FTAGS)}")
    _T["exec"] = out
    return out


def _risk_cells() -> dict:
    """tag -> {prio, sev, imp, lik, res, rating} parsed from the section-4 table."""
    if "risk" in _T:
        return _T["risk"]
    rows = table_rows("risk table", RISK_HDR)
    out = {}
    for r in (rows or [])[1:]:
        c = _cells_of(r)
        m = re.match(r"(F\d) ", c[1]) if len(c) == 6 else None
        if not m or not c[0].isdigit():
            FAILS.append(f"risk table: row not parseable (need 6 cells: priority, F<n> name, severity, I, L, "
                         f"residual): {str(r)[:60]}")
            continue
        rm = re.match(r"(High|Medium|Low)\b", c[5])
        out[m.group(1)] = {"prio": c[0], "sev": c[2], "imp": c[3], "lik": c[4], "res": c[5],
                           "rating": rm.group(1) if rm else None}
    NCHECK_inc()
    if rows is not None and sorted(out) != list(FTAGS):
        FAILS.append(f"risk table: findings present {sorted(out)}, expected {list(FTAGS)}")
    _T["risk"] = out
    RISK7.clear()
    RISK7.update(out)
    return out


def risk_table_rows():
    """[(finding id, severity, priority)] of the section-4 table, for other checks."""
    return [(t, r["sev"], int(r["prio"])) for t, r in _risk_cells().items()]


def _status_truth() -> dict:
    """Status word of each finding, derived from the stored results (never from the text)."""
    T = _truth()
    st = {}
    for tag, key in FKEY.items():
        st[tag] = "Mitigated" if T["pa"][key]["after_successes"] == 0 else "Not mitigated"
    pr = T["pr"]
    pb = T["sm"]["benign"]
    f7_unfixed = (pr["b2_fixed_kev_probe"]["n_exact_doc_in_top5"] == 0
                  and pr["b1_benign_questions_naming_ids"]["n_exact_doc_in_top5"] == 0
                  and pb["after"]["unblocked_hedges"] >= pb["before"]["unblocked_hedges"])
    st["F7"] = "Not mitigated" if f7_unfixed else "Partly"
    cur, rec = T["b1"]["curve"], T["rec"]["curve_after"]
    improved = [cell(rec, 0.25, a) > cell(cur, 0.25, a) for a in ("fd_fgsm", "pgd")]
    st["F5"] = "Mitigated" if all(improved) else "Partly" if any(improved) else "Not mitigated"
    shipped = T["df"]["curve_after"]
    better = [cell(shipped, e, a) > cell(cur, e, a) for e in (0.15, 0.25) for a in ("fd_fgsm", "pgd")]
    st["F8"] = "Not mitigated" if not any(better) else "Partly"
    st["F6"] = "n/a" if T["mi_attack"] <= T["mi_control"] else "Not mitigated"
    return st


def _cop_thr() -> float:
    """The copilot 'High' likelihood threshold, read from the report's own words ('at least four
    runs in five'); 0.8 only if the sentence is missing (risk_rule_checks then FAILS)."""
    if "cop_thr" not in _T:
        m = re.search(r"H = works in at least (\w+) runs in (\w+),", str(TEX))
        _T["cop_thr"] = (_WORDNUM[m.group(1)] / _WORDNUM[m.group(2)]) if m and m.group(1) in _WORDNUM \
            and m.group(2) in _WORDNUM else 0.8
    return _T["cop_thr"]


def _expected_levels() -> dict:
    """tag -> (impact, likelihood, severity, residual rating, status) with the likelihood taken
    from the stored data and the impact from the report's own table (a judgement, range-checked)."""
    T, rk, stt = _truth(), _risk_cells(), _status_truth()
    out = {}
    for tag in FTAGS:
        r = rk.get(tag)
        if r is None or r["imp"] not in SCALE:
            continue
        if tag in FKEY:
            _, rate = T["lik"][tag]
            lik = "H" if rate >= _cop_thr() else "M" if rate > 0 else "L"
        elif tag == "F7":
            lik = "H" if T["f7_fail_rate"] >= _cop_thr() else "M" if T["f7_fail_rate"] > 0 else "L"
        elif tag in ("F5", "F8"):
            lik = T["det_lik"]
        else:
            lik = T["f6_lik"]
        sev = risk_rule(r["imp"], lik)
        resid = risk_rule(r["imp"], "L") if stt[tag] == "Mitigated" else sev
        out[tag] = (r["imp"], lik, sev, resid, stt[tag])
    return out


# ---------------------------------------------------------------------------------------
def exec_summary():
    """Section 1: findings count, posture, top recommendation, deploy verdict, the two result
    bullets, the findings table and the key-terms box, each against the stored results."""
    T, ex, rk = _truth(), _exec_cells(), _risk_cells()
    lv = _expected_levels()
    pa, grid = T["pa"], T["grid"]
    sec = _slice("\\section{Executive summary}", "\\section{Task 1")
    sq = _squash(sec)
    # ---- order of the opening lines (count, posture, top recommendation, deploy), then bullets, table
    NCHECK_inc()
    marks = [sq.find(m) for m in ("\\lead{Findings.}", "\\lead{Overall risk posture.}", "\\lead{Top recommendation.}",
                                  "\\lead{Would we deploy?}", "\\begin{itemize}", "\\begin{tabular}", "\\begin{termsbox}")]
    if -1 in marks or marks != sorted(marks):
        FAILS.append(f"exec order: findings, posture, top recommendation, deploy verdict, bullets, table, key terms "
                     f"are not in that order (positions {marks})")
    NCHECK_inc()
    for term, defn in (("KB", "knowledge base (KB"), ("KEV", "(Known Exploited Vulnerabilities)")):
        i = sq.find(term)
        if i >= 0 and defn not in sq[:i + len(term) + 40]:
            FAILS.append(f"exec order: '{term}' is used before it is defined")
    # ---- findings line
    ln = _line("exec findings line", "\\lead{Findings.}")
    m = _rx("exec findings line", r"(\d+) findings \(F1--F(\d+)\): (\d+) High, (\d+) Medium, (\d+) Low, "
                                  r"ranked in \\S(\d+)", ln)
    if m:
        sevs = [v[2] for v in lv.values()]
        for lab, got, want in (("count", m.group(1), len(rk)), ("last id F-n", m.group(2), len(rk)),
                               ("High", m.group(3), sevs.count("High")), ("Medium", m.group(4), sevs.count("Medium")),
                               ("Low", m.group(5), sevs.count("Low"))):
            int_cell_check(f"exec findings {lab} (counted from impact x likelihood)", got, want)
        NCHECK_inc()
        if sorted(rk) != [f"F{i}" for i in range(1, len(rk) + 1)] or int(m.group(6)) != _sec_no("Risk assessment"):
            FAILS.append(f"exec findings line: ids {sorted(rk)} are not F1..F{len(rk)} or \\S{m.group(6)} is not the "
                         f"Risk assessment section (section {_sec_no('Risk assessment')})")
    # ---- posture line: copilot posture is the worst copilot residual, detector one per model
    ln = _line("exec posture line", "\\lead{Overall risk posture.}")
    m = _rx("exec posture", r"\\emph\{Copilot: (\w+)\.\} F3 .*?\\emph\{Detector: (\w+)\} \(F5\).*?It is "
                            r"\\emph\{(\w+)\} \(F8\)", ln)
    if m:
        order = {"High": 2, "Medium": 1, "Low": 0}
        cop = max((t for t in ("F1", "F2", "F3", "F4", "F7") if t in lv), key=lambda t: order[lv[t][3]])
        NCHECK_inc()
        if m.group(1).lower() != lv[cop][3].lower() or cop != "F3":
            FAILS.append(f"exec posture: copilot '{m.group(1)}' (driven by F3) but the worst copilot residual is "
                         f"{lv[cop][3]} ({cop})")
        NCHECK_inc()
        mt = re.search(r"if we keep the Lab~1 model \(the first-version model we attacked, \\S(\d+)\) behind a server-side", ln)
        if not mt or int(mt.group(1)) != _sec_no("Task 1"):
            FAILS.append("exec posture: the first-version-model reference is missing or is not the Task 1 section")
        NCHECK_inc()
        if m.group(2).lower() != lv["F5"][3].lower() or m.group(3).lower() != lv["F8"][3].lower():
            FAILS.append(f"exec posture: detector '{m.group(2)}' (F5) / '{m.group(3)}' (F8) but residuals are "
                         f"{lv['F5'][3]} / {lv['F8'][3]}")
    # ---- top recommendation: one label; the range it names is the KB-channel findings and holds priority 1
    ln = _line("exec top recommendation", "\\lead{Top recommendation.}")
    NCHECK_inc()
    if len(re.findall(r"Top recommendation", sq)) != 1:
        FAILS.append("exec top recommendation: not exactly one 'Top recommendation' label")
    m = _rx("exec top recommendation", r"stops F(\d)--F(\d) at the source", ln)
    if m and rk:
        named = {f"F{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)}
        kb = {t for t, k in FKEY.items() if T["channel"][k] == "kb"}
        top = min(rk, key=lambda t: int(rk[t]["prio"]))
        NCHECK_inc()
        if named != kb or top not in named:
            FAILS.append(f"exec top recommendation: names {sorted(named)}; knowledge-base-channel findings are "
                         f"{sorted(kb)}; priority-1 finding is {top}")
    # ---- deploy verdict follows the residual ratings
    ln = _line("exec deploy verdict", "\\lead{Would we deploy?} \\emph{Copilot:}")
    m = _rx("exec deploy verdict", r"\\emph\{Detector:\} yes for the Lab~1 model behind the check, but only after a re-test on the clock-repaired "
                                    r"split \(\\S(\d+)\); no for the "
                                    r"two-round model; the one-round candidate waits for a pre-release evasion test", ln)
    if m:
        NCHECK_inc()
        if int(m.group(1)) != _sec_no("Task 1"):
            FAILS.append("exec deploy verdict: the clock-repaired-split reference is not the Task 1 section")
        NCHECK_inc()
        mm = re.search(r"\(\\S(\d+), \\S(\d+)\)\. Our reasons for accepting Medium are in \\S(\d+)\.", ln)
        if not mm or (int(mm.group(1)), int(mm.group(2)), int(mm.group(3))) != (
                _sec_no("Task 1"), _sec_no("Recommendations"), _sec_no("Risk assessment")):
            FAILS.append("exec deploy verdict: its section references (Task 1, Recommendations, Risk assessment) are wrong")
        NCHECK_inc()
        if not (lv["F5"][3] != "High" and lv["F8"][3] == "High"
                and T["df"]["adversarial_training"]["rounds"] == 2 and T["rec"]["rounds"] == 1):
            FAILS.append("exec deploy verdict: yes/no does not follow the residuals (F5 not High, F8 High) or the "
                         "round counts of the two-round model and the one-round candidate")
    # ---- copilot bullet
    ln = _line("exec copilot bullet", "\\textbf{Copilot result.}")
    m = _rx("exec copilot bullet", r"Without guardrails, (\d+) of (\d+) attacks worked \((\d+/\d+) trials\)\. With them, "
                                   r"only the fabricated-technique attack \(F3\) still worked \((\d+/\d+) trials\).*?"
                                   r"Each attack ran (\d+) times per arm", ln)
    if m:
        worked = [k for k, p in pa.items() if p["before_successes"] > 0]
        left = [k for k, p in pa.items() if p["after_successes"] > 0]
        trials = T["sm"]["totals"]["trials_per_arm"]
        int_cell_check("exec copilot attacks that worked", m.group(1), len(worked))
        int_cell_check("exec copilot attacks", m.group(2), len(pa))
        _frac("exec copilot before trials", m.group(3), sum(pa[k]["before_successes"] for k in pa), trials)
        _frac("exec copilot after trials", m.group(4), sum(pa[k]["after_successes"] for k in pa), trials)
        NCHECK_inc()
        if left != [FKEY["F3"]]:
            FAILS.append(f"exec copilot bullet: only F3 should still work after the guardrails; stored results say {left}")
        NCHECK_inc()
        if not all(p["n_before"] == p["n_after"] == T["runs"] for p in pa.values()):
            FAILS.append("exec copilot bullet: not every attack ran the same number of times in both arms")
        int_cell_check("exec runs per arm", m.group(5), T["runs"])
    # ---- detector bullet
    ln = _line("exec detector bullet", "\\textbf{Detector result.}")
    m = _rx("exec detector bullet", r"\\vtheta=([\d.]+)\$ is a quarter of one\. We tested Wednesday's data and a "
                                    r"(\d+)-flow Thursday sample\..*?clean accuracy stayed at ([\d.]+) \(F8\)", ln)
    if m:
        _eps_lit("exec detector eps example", m.group(1), grid)
        NCHECK_inc()
        if abs(float(m.group(1)) - 1 / 4) > 1e-12:
            FAILS.append("exec detector bullet: the epsilon example is not a quarter")
        int_cell_check("exec Thursday sample flows", m.group(2), T["thu"]["n_flows"])
        cell_check("exec clean accuracy", m.group(3), T["df"]["clean_accuracy"]["after"])
        NCHECK_inc()
        if round(T["df"]["clean_accuracy"]["before"], 4) != round(T["df"]["clean_accuracy"]["after"], 4):
            FAILS.append("exec detector bullet: clean accuracy did not stay the same to 4 decimals")
        NCHECK_inc()
        if lv["F8"][4] != "Not mitigated" or lv["F6"][4] != "n/a":
            FAILS.append("exec detector bullet: 'easier to fool' (F8) / 'no leak' (F6) not supported by the stored results")
    # ---- findings table: Basis column
    cur, shp, rec = T["b1"]["curve"], T["df"]["curve_after"], T["rec"]["curve_after"]
    for tag in ("F1", "F2", "F3", "F4"):
        c = ex.get(tag)
        if c is None:
            continue
        p = pa[FKEY[tag]]
        m = _rx(f"exec {tag} basis", r"(\d+/\d+) (?:worked )?before, (\d+/\d+) after$", c[3])
        if m:
            _frac(f"exec {tag} before", m.group(1), p["before_successes"], p["n_before"])
            _frac(f"exec {tag} after", m.group(2), p["after_successes"], p["n_after"])
    c = ex.get("F8")
    if c:
        m = _rx("exec F8 basis", r"PGD-40\): detection ([\d.]+) \$\\to\$ ([\d.]+) at \$\\vtheta=([\d.]+)\$, and "
                                 r"([\d.]+) \$\\to\$ ([\d.]+) at ([\d.]+)$", c[3])
        if m:
            e1, e2 = float(m.group(3)), float(m.group(6))
            _eps_lit("exec F8 eps 1", m.group(3), grid)
            _eps_lit("exec F8 eps 2", m.group(6), grid)
            cell_check("exec F8 undefended PGD @eps1", m.group(1), cell(cur, e1, "pgd"))
            cell_check("exec F8 two-round PGD @eps1", m.group(2), cell(shp, e1, "pgd"))
            cell_check("exec F8 undefended PGD @eps2", m.group(4), cell(cur, e2, "pgd"))
            cell_check("exec F8 two-round PGD @eps2", m.group(5), cell(shp, e2, "pgd"))
    c = ex.get("F7")
    if c:
        m = _rx("exec F7 basis", r"(\d+/\d+) KEV-ID probes and (\d+/\d+) ID question found it$", c[3])
        if m:
            b2, b1q = T["pr"]["b2_fixed_kev_probe"], T["pr"]["b1_benign_questions_naming_ids"]
            _frac("exec F7 probes", m.group(1), b2["n_exact_doc_in_top5"], b2["n_probes"])
            _frac("exec F7 ID question", m.group(2), b1q["n_exact_doc_in_top5"], b1q["n_naming_an_id"])
    c = ex.get("F5")
    if c:
        m = _rx("exec F5 basis", r"PGD-40 detection at \$\\vtheta=([\d.]+)\$: Lab~1 model ([\d.]+), one-round "
                                 r"candidate ([\d.]+)$", c[3])
        if m:
            e = float(m.group(1))
            _eps_lit("exec F5 eps", m.group(1), grid)
            cell_check("exec F5 Lab 1 PGD", m.group(2), cell(cur, e, "pgd"))
            cell_check("exec F5 one-round candidate PGD", m.group(3), cell(rec, e, "pgd"))
            NCHECK_inc()
            if T["rec"]["rounds"] != 1 or T["rec"]["n_frozen_features"] < 1:
                FAILS.append("exec F5 basis: 'one-round candidate' is not a one-round config with frozen features")
    c = ex.get("F6")
    if c:
        m = _rx("exec F6 basis", r"Score ([\d.]+) against blind control ([\d.]+)$", c[3])
        if m:
            cell_check("exec F6 attack score", m.group(1), T["mi_attack"])
            cell_check("exec F6 blind control", m.group(2), T["mi_control"])
    # ---- Severity / Status columns follow from the stored data and from section 4
    vocab = re.compile(r"(Mitigated|Partly|Not mitigated|n/a) \((High|Medium|Low)(?: if shipped)?\)$")
    for tag in FTAGS:
        c = ex.get(tag)
        if c is None or tag not in lv:
            continue
        NCHECK_inc()
        if c[1] != lv[tag][2] or (tag in rk and c[1] != rk[tag]["sev"]):
            FAILS.append(f"exec {tag} severity: table says '{c[1]}', impact {lv[tag][0]} x likelihood {lv[tag][1]} "
                         f"gives {lv[tag][2]} (section 4 says {rk[tag]['sev']})")
        m = vocab.match(str(c[2]))
        NCHECK_inc()
        if not m:
            FAILS.append(f"exec {tag} status: '{c[2]}' not 'Mitigated|Partly|Not mitigated|n/a (High|Medium|Low)'")
            continue
        NCHECK_inc()
        if m.group(1) != lv[tag][4] or m.group(2) != lv[tag][3] or m.group(2) != rk[tag]["rating"]:
            FAILS.append(f"exec {tag} status: '{c[2]}' but the stored results say {lv[tag][4]} with residual "
                         f"{lv[tag][3]} (section 4 residual: {rk[tag]['rating']})")
    # ---- key terms box
    ln = _line("exec term PGD-40", "\\term{PGD-40}")
    m = _rx("exec term PGD-40", r"(\d+) gradient steps", ln)
    if m:
        int_cell_check("exec term PGD-40 steps", m.group(1), T["b1"]["pgd"]["steps"])
    ln = _line("exec term random search", "\\term{Random search}")
    m = _rx("exec term random search", r"(\d+) random changes", ln)
    if m:
        per = [r["client_budget_queries_per_flow"] for r in T["bud"]["black_box_methods"]["random_search"]["per_eps"]]
        NCHECK_inc()
        if "tries=200" not in (ROOT / "detector" / "evasion.py").read_text() or max(per) != 200:
            FAILS.append(f"exec term random search: the 200-try budget is not what detector/evasion.py runs "
                         f"(stored per-flow budgets: max {max(per)})")
        int_cell_check("exec term random search tries", m.group(1), int(max(per)))
    ln = _line("exec term release gate", "\\term{Evasion-curve release gate}")
    m = _rx("exec term release gate", r"\\vtheta=([\d.]+)\$ and ([\d.]+) must detect", ln)
    if m:
        _eps_lit("exec release gate eps 1", m.group(1), grid)
        _eps_lit("exec release gate eps 2", m.group(2), grid)
    ln = _line("exec term Clopper-Pearson", "\\term{Clopper-Pearson bound}")
    m = _rx("exec term Clopper-Pearson", r"a count like (\d+/\d+) allows \((\d+\\%) confidence\)", ln)
    if m:
        z = [x for x in T["zb"]["attack_cells_with_zero_successes"] if f"{x['successes']}/{x['n']}" == str(m.group(1))]
        NCHECK_inc()
        if not z:
            FAILS.append(f"exec term Clopper-Pearson: no stored zero-success cell {m.group(1)}")
        else:
            cover(m.group(1), [m.group(1)], "frac")
        NCHECK_inc()
        if "beta.ppf(0.95," not in T["zb"]["one_sided_95_upper_formula"] or str(m.group(2)) != "95\\%":
            FAILS.append("exec term Clopper-Pearson: 95% is not the confidence level of the stored bound")
        else:
            cover(m.group(2), [m.group(2)], "level")


# ---------------------------------------------------------------------------------------
def risk_table():
    """Section 4: every number inside the risk table's residual cells and the two paragraphs
    after it (overall residual, why Medium), each cell-scoped, against the stored results."""
    T, rk = _truth(), _risk_cells()
    lv = _expected_levels()
    grid, pa = T["grid"], T["pa"]
    cur, shp, rec = T["b1"]["curve"], T["df"]["curve_after"], T["rec"]["curve_after"]
    res = {t: r["res"] for t, r in rk.items()}
    # F3: unchanged, k/n after
    if "F3" in res:
        m = _rx("risk F3 residual", r"^High, unchanged\. (\d+/\d+) after;", res["F3"])
        if m:
            p = pa[FKEY["F3"]]
            _frac("risk F3 after", m.group(1), p["after_successes"], p["n_after"])
    # F1, F2, F4: k/n after and the exact one-sided upper bound
    for tag in ("F1", "F2", "F4"):
        if tag not in res:
            continue
        key = FKEY[tag]
        m = _rx(f"risk {tag} residual", r"(\d+/\d+) after \((?:95\\% )?upper bound ([\d.]+)\)", res[tag])
        if m:
            p = pa[key]
            z = T["zcell"].get(key + "/after")
            _frac(f"risk {tag} after", m.group(1), p["after_successes"], p["n_after"])
            NCHECK_inc()
            if z is None or z["successes"] != 0:
                FAILS.append(f"risk {tag}: no stored zero-success bound for {key}/after")
            else:
                cell_check(f"risk {tag} upper bound", m.group(2), z["one_sided_95_upper"])
                NCHECK_inc()
                if abs(z["one_sided_95_upper"] - (1 - 0.05 ** (1 / z["n"]))) > 1e-9:
                    FAILS.append(f"risk {tag}: stored bound is not the exact k=0 Clopper-Pearson bound for n={z['n']}")
    # F8: fell at the two change sizes under all three attacks, clean accuracy held, candidate not below at the larger size
    if "F8" in res:
        m = _rx("risk F8 residual", r"Detection fell at \$\\vtheta=([\d.]+)\$ and ([\d.]+) while clean accuracy held",
                res["F8"])
        if m:
            for lit in (m.group(1), m.group(2)):
                _eps_lit("risk F8 eps", lit, grid)
                NCHECK_inc()
                for a in ("random_search", "fd_fgsm", "pgd"):
                    if not cell(shp, float(lit), a) < cell(cur, float(lit), a):
                        FAILS.append(f"risk F8: detection did not fall at {lit} under {a} "
                                     f"(shipped {cell(shp, float(lit), a)}, undefended {cell(cur, float(lit), a)})")
            NCHECK_inc()
            if round(T["df"]["clean_accuracy"]["before"], 4) != round(T["df"]["clean_accuracy"]["after"], 4):
                FAILS.append("risk F8: clean accuracy did not hold to 4 decimals")
        if m:
            e2 = float(m.group(2))
            NCHECK_inc()
            if not all(cell(rec, e2, a) >= cell(cur, e2, a) for a in ("fd_fgsm", "pgd")) or T["rec"]["rounds"] != 1:
                FAILS.append(f"risk F8: the one-round candidate is below the undefended model at {e2}")
    # F7: fails every time; hedges before/after
    if "F7" in res:
        m = _rx("risk F7 residual", r"Fails every time \(unblocked hedges (\d+/\d+) before, (\d+/\d+) after, \\S(\d+)\)",
                res["F7"])
        if m:
            b = T["sm"]["benign"]
            _frac("risk F7 hedges before", m.group(1), b["before"]["unblocked_hedges"], b["before"]["total_runs"])
            _frac("risk F7 hedges after", m.group(2), b["after"]["unblocked_hedges"], b["after"]["total_runs"])
            NCHECK_inc()
            if T["f7_fail_rate"] != 1.0 or int(m.group(3)) != _sec_no("Task 2"):
                FAILS.append(f"risk F7: 'fails every time' but the stored failure rate is {T['f7_fail_rate']}, or "
                             f"\\S{m.group(3)} is not the Task 2 section ({_sec_no('Task 2')})")
    # F5: Lab 1 model, one-round candidate, Thursday sample
    if "F5" in res:
        m = _rx("risk F5 residual", r"Lab~1 model: detection ([\d.]+) at \$\\vtheta=([\d.]+)\$ and ([\d.]+) at ([\d.]+)\. "
                                    r"One-round candidate: FD-FGSM stays ([\d.]+) at ([\d.]+) \(PGD-40 rises to ([\d.]+)\)\. "
                                    r"Evasion also works on a Thursday sample \(FD-FGSM (\d+/\d+) detected at ([\d.]+)\)",
                res["F5"])
        if m:
            e1, e2, e3, e4 = (float(m.group(i)) for i in (2, 4, 6, 9))
            for i in (2, 4, 6, 9):
                _eps_lit("risk F5 eps", m.group(i), grid)
            cell_check("risk F5 Lab 1 FD-FGSM @eps1", m.group(1), cell(cur, e1, "fd_fgsm"))
            NCHECK_inc()
            if not _agree(m.group(1), cell(cur, e1, "pgd")):
                FAILS.append("risk F5: Lab 1 PGD-40 detection differs from the printed value at the first epsilon")
            cell_check("risk F5 Lab 1 FD-FGSM @eps2", m.group(3), cell(cur, e2, "fd_fgsm"))
            NCHECK_inc()
            if not _agree(m.group(3), cell(cur, e2, "pgd")):
                FAILS.append("risk F5: Lab 1 PGD-40 detection differs from the printed value at the second epsilon")
            cell_check("risk F5 candidate FD-FGSM", m.group(5), cell(rec, e3, "fd_fgsm"))
            NCHECK_inc()
            if cell(rec, e3, "fd_fgsm") != cell(cur, e3, "fd_fgsm"):
                FAILS.append("risk F5: 'FD-FGSM stays' but the candidate differs from the Lab 1 model at that epsilon")
            cell_check("risk F5 candidate PGD-40", m.group(7), cell(rec, e3, "pgd"))
            NCHECK_inc()
            if not cell(rec, e3, "pgd") > cell(cur, e3, "pgd"):
                FAILS.append("risk F5: 'PGD-40 rises' but the candidate is not above the Lab 1 model")
            d = next(r for r in T["thu"]["curve"] if abs(r["eps_std"] - e4) < 1e-9)["methods"]["fd_fgsm"]["detection"]
            _frac("risk F5 Thursday FD-FGSM", m.group(8), d["count"], d["n"])
            NCHECK_inc()
            if T["thu"]["n_flows"] != d["n"]:
                FAILS.append("risk F5: Thursday sample size differs from the stored detection denominator")
    # F6
    if "F6" in res:
        m = _rx("risk F6 residual", r"\(([\d.]+) against control ([\d.]+)\)", res["F6"])
        if m:
            cell_check("risk F6 attack accuracy", m.group(1), T["mi_attack"])
            cell_check("risk F6 control accuracy", m.group(2), T["mi_control"])
    # ---- overall residual paragraph
    ln = _line("risk overall residual", "\\lead{Overall residual risk.}")
    m = _rx("risk overall residual", r"One High remains in the copilot \(F3\)\. In the detector, F8 stays High only if "
                                     r"the two-round model ships; .*?exposure is F5 \((\w+)\)\. F1 and F2 drop to "
                                     r"residual (\w+); a (\d+/\d+) result bounds the true success rate only at ([\d.]+)"
                                     r".*?Verdict: \\S(\d+)\.", ln)
    if m and lv:
        copilot_high = [t for t in ("F1", "F2", "F3", "F4", "F7") if lv[t][3] == "High"]
        NCHECK_inc()
        if copilot_high != ["F3"] or lv["F8"][3] != "High":
            FAILS.append(f"risk overall: copilot residual Highs are {copilot_high} (expected only F3) or F8 is not High")
        NCHECK_inc()
        if m.group(1) != lv["F5"][3] or m.group(2) != lv["F1"][3] or lv["F1"][3] != lv["F2"][3]:
            FAILS.append(f"risk overall: text says F5 {m.group(1)}, F1/F2 {m.group(2)}; rule gives F5 {lv['F5'][3]}, "
                         f"F1 {lv['F1'][3]}, F2 {lv['F2'][3]}")
        z = [x for x in T["zb"]["attack_cells_with_zero_successes"] if f"{x['successes']}/{x['n']}" == str(m.group(3))]
        NCHECK_inc()
        if not z:
            FAILS.append(f"risk overall: no stored zero-success cell {m.group(3)}")
        else:
            cover(m.group(3), [m.group(3)], "frac")
            cell_check("risk overall upper bound", m.group(4), z[0]["one_sided_95_upper"])
        NCHECK_inc()
        if int(m.group(5)) != _sec_no("Executive summary"):
            FAILS.append(f"risk overall: 'Verdict: \\S{m.group(5)}' is not the Executive summary section")
    # ---- why Medium is acceptable
    ln = _line("risk why medium", "\\lead{Why Medium is acceptable for F5.}")
    m = _rx("risk why medium", r"The Lab~1 model still detects ([\d.]+) of attack flows at \$\\vtheta=([\d.]+)\$\. "
                               r".*?reaches (\d+) of the (\d+) flows that slip under the alert at that size", ln)
    if m:
        e = float(m.group(2))
        _eps_lit("risk why medium eps", m.group(2), grid)
        for pat, want in ((r"timing shifts of seconds \(\\S(\d+)\)", "Task 1"), (r"at that size \(\\S(\d+)\)", "Recommendations")):
            NCHECK_inc()
            mm = re.search(pat, ln)
            if not mm or int(mm.group(1)) != _sec_no(want):
                FAILS.append(f"risk why medium: section reference for '{want}' missing or wrong (pattern {pat})")
        cell_check("risk why medium Lab 1 detection (FD-FGSM)", m.group(1), cell(cur, e, "fd_fgsm"))
        NCHECK_inc()
        if not _agree(m.group(1), cell(cur, e, "pgd")):
            FAILS.append("risk why medium: Lab 1 PGD-40 detection differs from the printed value")
        band = T["ops"]["confidence_bands"]["evasion_flows_in_L1_review"]
        bands = T["ops"]["confidence_bands"]["bands"]
        for att in ("fd_fgsm", "pgd"):
            k = f"{att}_eps{e}"
            NCHECK_inc()
            b, c = band.get(k), bands.get(k)
            if not b or not c or b["n_evasion_flows"] != c["counts"]["L1_review"] + c["counts"]["pass"] \
                    or b["n_evasion_flows"] != round(b["n_total_flows"] * (1 - cell(cur, e, att))) \
                    or b["lower_edge_0.01"]["in_L1_review"] != c["counts"]["L1_review"]:
                FAILS.append(f"risk why medium: stored band counts for {k} are inconsistent")
        k = f"fd_fgsm_eps{e}"
        if k in band:
            int_cell_check("risk why medium flows in the L1 band", m.group(3), band[k]["lower_edge_0.01"]["in_L1_review"])
            int_cell_check("risk why medium flows slipping under the alert", m.group(4), band[k]["n_evasion_flows"])
        else:
            FAILS.append(f"risk why medium: no stored band counts for {k}")


# ---------------------------------------------------------------------------------------
def _parse_rule(txt: str) -> dict:
    out = {}
    for sev, body in re.findall(r"(High|Medium|Low) = ([^.]*?)(?:\.|$)", txt):
        out[sev] = {tuple(p.replace("$\\times$", "x").split("x")) for p in body.split(", ")}
    return out


def risk_rule_checks():
    """Section 4: the stated rule equals risk_rule(); every row's severity, likelihood, residual
    rating, status and priority follow from the rule and the stored results; the detector
    likelihood criterion quotes the stored attacker-budget numbers."""
    T, rk = _truth(), _risk_cells()
    lv = _expected_levels()
    # ---- the rule as stated in the text equals the rule the checks apply
    ln = _line("risk rule bullet", "\\item \\textbf{Rule:}")
    NCHECK_inc()
    if ln is not None:
        stated = _parse_rule(str(ln).split("Rule:}", 1)[1].split("Residual risk")[0])
        want = {s: {(i, l) for i in "HML" for l in "HML" if risk_rule(i, l) == s} for s in ("High", "Medium", "Low")}
        if stated != want:
            FAILS.append(f"risk rule bullet: the stated rule {stated} differs from impact x likelihood as applied {want}")
    # ---- copilot likelihood threshold quoted in words equals the threshold applied
    ln = _line("risk copilot likelihood", "\\item \\textbf{Likelihood (L), copilot:}")
    m = _rx("risk copilot likelihood", r"H = works in at least (\w+) runs in (\w+),", ln)
    if m:
        NCHECK_inc()
        if m.group(1) not in _WORDNUM or m.group(2) not in _WORDNUM or \
                _WORDNUM[m.group(1)] / _WORDNUM[m.group(2)] != _cop_thr():
            FAILS.append("risk copilot likelihood: 'at least X runs in Y' is not the threshold the checks apply")
    # ---- detector likelihood criterion quotes the stored attacker-budget numbers
    ln = _line("risk detector likelihood", "\\item \\textbf{Likelihood (L), detector,}")
    m = _rx("risk detector likelihood", r"The alert \(\\S(\d+)\) fires at (\d+) queries from one client, half of the (\d+) "
                                        r"queries per flow that FD-FGSM, our cheapest attack, needs\. M = white-box "
                                        r"access, or at least that many queries \(random search needs up to (\d+)\)", ln)
    if m:
        NCHECK_inc()
        if int(m.group(1)) != _sec_no("Recommendations"):
            FAILS.append(f"risk detector likelihood: alert section \\S{m.group(1)} is not Recommendations")
        int_cell_check("risk alert threshold", m.group(2), T["thr"])
        int_cell_check("risk FD-FGSM queries per flow", m.group(3), int(T["fd_min"]))
        int_cell_check("risk random-search queries per flow (max)", m.group(4), int(T["rs_max"]))
        NCHECK_inc()
        if T["thr"] != T["thr_calc"] or T["ctl"]["alert_fraction"] != 0.5 or T["thr"] * 2 != T["fd_min"]:
            FAILS.append(f"risk detector likelihood: 'half of the cheapest attack' not supported: threshold {T['thr']}, "
                         f"floor(fraction x cheapest) = {T['thr_calc']}, fraction {T['ctl']['alert_fraction']}")
        NCHECK_inc()
        bm = T["bud"]["black_box_methods"]
        if not (T["fd_min"] == bm["fd_fgsm"]["server_measured_queries_per_flow_max"] <= T["ctl"]["random_search_queries_per_flow_min"]):
            FAILS.append("risk detector likelihood: FD-FGSM is not the cheapest black-box attack in the stored budgets")
    # ---- per-row checks
    for tag in FTAGS:
        r = rk.get(tag)
        if r is None or tag not in lv:
            NCHECK_inc()
            FAILS.append(f"risk {tag}: row missing or its impact is outside H/M/L")
            continue
        imp, lik, sev, resid, status = lv[tag]
        NCHECK_inc()
        if r["lik"] not in SCALE or r["lik"] != lik:
            FAILS.append(f"risk {tag}: likelihood '{r['lik']}' but the stored results and the stated criterion give '{lik}'")
        NCHECK_inc()
        if r["sev"] != risk_rule(imp, r["lik"]) or r["sev"] != sev:
            FAILS.append(f"risk {tag}: severity '{r['sev']}' contradicts the rule: impact {imp} x likelihood {r['lik']} "
                         f"= {risk_rule(imp, r['lik'])}")
        NCHECK_inc()
        if r["rating"] != resid:
            FAILS.append(f"risk {tag}: residual rating '{r['rating']}' but the rule with the post-defense likelihood "
                         f"({'L' if status == 'Mitigated' else lik}) gives '{resid}'")
        NCHECK_inc()
        if tag in ("F3", "F7") and not re.match(rf"{sev}, (unchanged|unaddressed)\.", r["res"]):
            FAILS.append(f"risk {tag}: an unmitigated finding keeps its severity '{sev}' as residual: {r['res'][:50]}")
    # ---- priority: 1..n, severity first, then findings the defenses did not mitigate; ties are cost (not measured)
    NCHECK_inc()
    if rk and len(rk) == len(FTAGS) and all(t in lv for t in FTAGS):
        order = sorted(rk, key=lambda t: int(rk[t]["prio"]))
        keys = [(_RANK_SEV[lv[t][2]], _RANK_STATUS[lv[t][4]]) for t in order]
        if [int(rk[t]["prio"]) for t in order] != list(range(1, len(rk) + 1)):
            FAILS.append("risk table: priorities are not 1..n")
        elif keys != sorted(keys):
            FAILS.append(f"risk table: priority order {order} is not severity first, then not-mitigated before "
                         f"partly before mitigated (keys {keys})")
        for t in order:
            cover(rk[t]["prio"], [rk[t]["prio"]], "priority")


def sentence_at(label: str, anchor: str, also: str = "") -> str:
    """The sentence (squashed, from the sentence containing `anchor` to its end) of the
    single paragraph holding `anchor`; FAIL (returns '') on zero or several paragraphs."""
    global NCHECK
    NCHECK += 1
    paras = [p for p in _paras() if anchor in p and also in p]
    if len(paras) != 1:
        FAILS.append(f"{label}: anchor '{anchor}' matched {len(paras)} paragraphs, expected 1")
        return ""
    t = paras[0]
    a = t.index(anchor)
    b = re.search(r"\.(?:\s|$)", t[a:])
    return t[a:a + b.start()] if b else t[a:]


def chroma_vendors(ids):
    """Vendor of each KEV doc id, read straight from the stored collection (read-only)."""
    import sqlite3
    db = ROOT / "data" / "chroma" / "chroma.sqlite3"
    if not db.is_file():
        return None
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for i in ids:
        r = c.execute("select e.id from embeddings e where e.embedding_id=?", (i,)).fetchone()
        v = c.execute("select string_value from embedding_metadata where id=? and key='vendor'",
                      (r[0],)).fetchone() if r else None
        out.append(v[0] if v else "?")
    return out


def task2_text():
    """Section 6 / Fig. 1 prose numbers that live in task2_text (guard evidence, Limitation, Fig. 1
    caption). The Task 2 section itself is checked in task2_section()."""
    sm = load("task2_summary.json")
    # --- guard availability (section 6): the Limitation paragraph, recomputed from the raw trials
    ge = sm["guard_evidence"]
    d = load("task2_redteam.json")
    p = _para1("guard evidence", "\\lead{Limitation.}")
    m = _rx("guard evidence", r"The stored records show the guard was serving\. In the after arm, (\d+) of (\d+) trials carry a "
            r"real category verdict, (\d+) passed with no blocking layer, and (\d+) records are errored or unspecified\.", p)
    if m:
        tr = [t for r in d["after"] for t in r["trials"]]
        verdict = sum(1 for t in tr if any(l.startswith("llama_guard:") and l != "llama_guard:unspecified" for l in t["block_layers"]))
        passed = sum(1 for t in tr if not t["block_layers"])
        bad = sum(1 for t in tr for l in t["block_layers"]
                  if l.startswith("llama_guard") and (l in ("llama_guard", "llama_guard:unspecified", "llama_guard:[]") or t.get("errored")))
        NCHECK_inc()
        if not (verdict == ge["after_trials_with_any_llama_guard_verdict"] and len(tr) == ge["after_trials_total"]
                and passed == ge["after_trials_passing_with_no_block_layer"]):
            FAILS.append("guard evidence: task2_summary.json guard_evidence disagrees with the raw after-arm trials")
        for lab, got, w in (("guard verdicts", m.group(1), verdict),
                            ("guard trials", m.group(2), len(tr)),
                            ("after-arm passes", m.group(3), passed),
                            ("errored/unspecified", m.group(4), ge["records_with_unspecified_empty_category_or_error"])):
            int_cell_check(lab, got, w)
        NCHECK_inc()
        if ge["errored_trials_all_arms"] != 0 or bad != 0 or any(t.get("errored") for r in d["before"] + d["after"] for t in r["trials"]):
            FAILS.append("guard evidence: errored or unspecified guard records exist, so 'the guard was serving' is unsupported")
    # the Limitation paragraph: one paragraph, the fail-closed premise and the three guard sentences
    NCHECK_inc()
    if p is not None and not ("fails closed and an outage would read as a perfect defense" in p
                              and len(re.findall(r"\.(?:\s|$)", re.sub(r"\d\.\d", "", p))) == 3):
        FAILS.append("Limitation paragraph is not the three expected sentences (fail-closed premise, stored-records, counts)")
    # --- Fig. 1 caption n
    ln = sentence_at("fig1 caption", "Detection rate against perturbation size (")
    m = re.search(r"\$n=(\d+)\$ flows per point", ln)
    n_eval = load("task1b_evasion_wednesday.json")["eval_set_provenance"]["n"]
    int_cell_check("fig1 n", m.group(1) if m else "?", n_eval)


_NUMW = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10}


def _word_is(label: str, word, n) -> None:
    """A number written as a word ('Five of the nine') must equal the count computed from the JSON."""
    NCHECK_inc()
    if _NUMW.get(str(word).capitalize()) != n:
        FAILS.append(f"{label}: the text says '{word}', the results say {n}")


def _file_has(label: str, relfile: str, pat: str) -> bool:
    """A configuration value quoted in the text must still be what the code says."""
    NCHECK_inc()
    try:
        ok = bool(_re_real.search(pat, (ROOT / relfile).read_text(), _re_real.S))
    except OSError:
        ok = False
    if not ok:
        FAILS.append(f"{label}: '{pat}' not found in {relfile}")
    return ok


def _cp_upper_zero(n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound for 0 successes in n trials: 1-(1-conf)^(1/n)."""
    return 1 - (1 - conf) ** (1 / n)


def _a_list(txt) -> list:
    return [int(x) for x in _re_real.findall(r"A(\d)", str(txt))]


def task2_section():
    """Every number in sections/03_task2.tex, paragraph- or sentence-scoped against
    results/task2_*.json, ops_measurements.json (copilot_latency) and brief_citations.json.
    Derived values (counts of attacks that worked, 42 -> 10, bounds, ratios) are computed here."""
    d = load("task2_redteam.json")
    sm = load("task2_summary.json")
    pr = load("task2_retrieval_probe.json")
    zb = load("task2_zero_rate_bounds.json")
    ops = load("ops_measurements.json")["copilot_latency"]
    bc = load("brief_citations.json")
    kb = sm["kb"]
    runs = d["runs_per_attack"]
    per_list = sm["per_attack"]
    per = {p["key"]: p for p in per_list}
    keys = [p["key"] for p in per_list]
    bef = {r["key"]: r for r in d["before"]}
    aft = {r["key"]: r for r in d["after"]}
    ben_a, ben_b = sm["benign"]["after"], sm["benign"]["before"]
    ge = sm["guard_evidence"]
    NCHECK_inc()
    expected = ["2A-direct-override", "2A-direct-secret", "2A-direct-prompt-exfil", "2A-kb-override",
                "2A-kb-secret", "2A-kb-misinfo", "2B-roleplay", "2B-base64", "2B-manyshot"]
    if keys != expected or keys != [r["key"] for r in d["before"]] or keys != [r["key"] for r in d["after"]]:
        FAILS.append(f"t2: the stored attack order is not A1..A9 = {expected}")
        return
    A = {k: i for i, k in enumerate(keys, 1)}

    # ---- Target paragraph
    p = _para1("t2 target", "\\lead{Target.} The target is a retrieval-augmented security copilot")
    m = _rx("t2 target retrieval", r"top (\d+) of ([\d{},]+) documents", p)
    if m:
        int_cell_check("t2 target top-k", m.group(1), pr["top_k"])
        int_cell_check("t2 target collection size", m.group(2), kb["total"])
        _file_has("t2 target top-k config", "chatbot/copilot.py", rf"top_k: int = {m.group(1)}\b")
        NCHECK_inc()
        if not (kb["total"] == kb["attack"] + kb["kev"] == pr["collection_count"] == sum(kb["by_source"].values())):
            FAILS.append("t2 target: the stored KB counts do not add up to the collection size")
    m = _rx("t2 target KB", r"\\textbf\{KB:\} (\d+) ATT\\&CK techniques and ([\d{},]+) CISA KEV entries\. These differ "
            r"from the brief's (\d+) and about ([\d{},]+) because the feeds are live", p)
    if m:
        int_cell_check("t2 target ATT&CK count", m.group(1), kb["attack"])
        int_cell_check("t2 target KEV count", m.group(2), kb["kev"])
        s1 = bc["sentences"]["s1"]
        b_att = _re_real.search(r"the (\d+) ATT&CK techniques", s1)
        b_kev = _re_real.search(r"\(([\d,]+) entries\)", s1)
        frag = {c["id"]: c["fragment"] for c in bc["checks"]}
        NCHECK_inc()
        if not (b_att and b_kev and frag.get("attack_techniques", "") in s1 and frag.get("kev_entries", "") in s1):
            FAILS.append("t2 target: results/brief_citations.json does not hold the brief's ATT&CK / KEV counts")
        else:
            int_cell_check("t2 target brief ATT&CK count (stored brief sentence)", m.group(3), int(b_att.group(1)))
            int_cell_check("t2 target brief KEV count (stored brief sentence)", m.group(4), int(b_kev.group(1).replace(",", "")))
            NCHECK_inc()
            if kb["attack"] == int(b_att.group(1)) or kb["kev"] == int(b_kev.group(1).replace(",", "")):
                FAILS.append("t2 target: the stored KB counts equal the brief's, so 'These differ' is wrong")
    m = _rx("t2 target model", r"\(Q(\d)\\_K\\_M, (\d+)-bit compressed weights\)", p)
    if m:
        NCHECK_inc()
        if m.group(1) != m.group(2):
            FAILS.append(f"t2 target: quantisation Q{m.group(1)} but '{m.group(2)}-bit' in the text")
        int_cell_check("t2 target bits", m.group(2), int(m.group(1)))
        _file_has("t2 target answer-model quantisation", "chatbot/llm.py", rf"LLM_MODEL = \"[^\"]*Q{m.group(1)}_K_M")

    # ---- Threat model: channel lists are checked in task2_threat_and_benign()

    # ---- Attacks and success conditions
    p = _para1("t2 attacks", "\\lead{Attacks and success conditions.}")
    m = _rx("t2 attacks runs", r"Each of the (\w+) attacks ran (\d+) times per arm", p)
    if m:
        _word_is("t2 attacks count", m.group(1), len(keys))
        int_cell_check("t2 attacks runs per arm", m.group(2), runs)
        NCHECK_inc()
        if any(r["runs"] != runs for r in d["before"] + d["after"]):
            FAILS.append("t2 attacks: not every attack ran the stated number of times in both arms")
    m = _rx("t2 attacks sentinel", r"A1 override: \\texttt\{PURPLE-UNICORN-(\d+)\} appears\. A2 secret extraction: "
            r"\\texttt\{THUNDER-(\d+)\} appears", p)
    if m:
        def _const(relfile, pat):
            try:
                mm = _re_real.search(pat, (ROOT / relfile).read_text())
            except OSError:
                mm = None
            return int(mm.group(1)) if mm else -1
        int_cell_check("t2 attacks A1 marker digits (chatbot/attacks.py SENTINEL)", m.group(1),
                       _const("chatbot/attacks.py", r'SENTINEL = "PURPLE-UNICORN-(\d+)"'))
        int_cell_check("t2 attacks A2 secret digits (chatbot/prompts.py SECRET_CODE)", m.group(2),
                       _const("chatbot/prompts.py", r'SECRET_CODE = "THUNDER-(\d+)"'))
    m = _rx("t2 attacks many-shot", r"A9 many-shot \((\d+) examples in the prompt\)", p)
    if m:
        try:
            src = (ROOT / "chatbot/attacks.py").read_text()
            blk = src[src.index("JB_MANYSHOT = Attack("):src.index("ATTACKS: list[Attack]")]
            n_ex = len(_re_real.findall(r"\"A: \S", blk))
        except (OSError, ValueError):
            n_ex = -1
        int_cell_check("t2 attacks many-shot examples", m.group(1), n_ex)
        NCHECK_inc()
        if not (A["2B-roleplay"] == 7 and A["2B-manyshot"] == 9 and A["2B-base64"] == 8 and A["2A-kb-misinfo"] == 6):
            FAILS.append("t2 attacks: the A-numbers in the text do not match the stored attack order")

    # ---- Guardrail configuration: the Llama Guard categories that fired
    p = _para1("t2 guardrails", "\\lead{Guardrail configuration.}")
    codes = sorted(int(k[1:]) for k in ge["after_trials_with_each_llama_guard_category"])
    m = _rx("t2 guardrails categories", r"Categories seen: ((?:S\d+(?:, )?)+)\.", p)
    if m:
        got = [x for x in re.finditer(r"S(\d+)", m.group(1))]
        NCHECK_inc()
        if [int(x.group(1)) for x in got] != codes:
            FAILS.append(f"t2 guardrails: the text lists categories {[x.group(1) for x in got]}, the stored records show {codes}")
        for x in got:
            int_cell_check("t2 guardrails category code", x.group(1), int(x.group(1)) if int(x.group(1)) in codes else -1)

    # ---- Which attacks worked before the guardrails
    p = _para1("t2 worked", "\\lead{Which attacks worked before the guardrails.}")
    worked = [A[p_["key"]] for p_ in per_list if p_["before_successes"] > 0]
    stopped = [A[p_["key"]] for p_ in per_list if p_["before_successes"] > 0 and p_["after_successes"] == 0]
    stayed = [A[p_["key"]] for p_ in per_list if p_["before_successes"] > 0 and p_["after_successes"] > 0]
    zero_before = [A[p_["key"]] for p_ in per_list if p_["before_successes"] == 0]
    tot_b = sum(p_["before_successes"] for p_ in per_list)
    tot_n = sum(p_["n_before"] for p_ in per_list)
    m = _rx("t2 worked", r"(\w+) of the (\w+) did \((\d+/\d+) trials\): ([A\d, and]+)\. The guardrails stopped (\w+) of "
            r"them \(([A\d, ]+)\)\. A(\d), the fabricated technique, stayed at (\d+/\d+)\. The other (\w+) \(([A\d, ]+)\) "
            r"were (\d+/\d+) \\emph\{before\}, so their (\d+/\d+) after shows no defense effect\.", p)
    if m:
        _word_is("t2 worked count", m.group(1), len(worked))
        _word_is("t2 worked of nine", m.group(2), len(keys))
        _frac("t2 worked trials", m.group(3), tot_b, tot_n)
        NCHECK_inc()
        if tot_b != sum(sm["totals"][k] for k in ("before_successes",)) or tot_n != sm["totals"]["trials_per_arm"]:
            FAILS.append("t2 worked: the per-attack counts do not add up to the stored totals")
        NCHECK_inc()
        if _a_list(m.group(4)) != worked:
            FAILS.append(f"t2 worked: the text names {_a_list(m.group(4))}, the stored before-arm successes are in A{worked}")
        _word_is("t2 worked stopped count", m.group(5), len(stopped))
        NCHECK_inc()
        if _a_list(m.group(6)) != stopped:
            FAILS.append(f"t2 worked: the text says the guardrails stopped {_a_list(m.group(6))}, the stored results say {stopped}")
        NCHECK_inc()
        if stayed != [int(m.group(7))] or A["2A-kb-misinfo"] != int(m.group(7)):
            FAILS.append(f"t2 worked: the text says A{m.group(7)} stayed, the stored results say A{stayed}")
        p6 = per["2A-kb-misinfo"]
        _frac("t2 worked A6 stayed", m.group(8), p6["after_successes"], p6["n_after"])
        _word_is("t2 worked other count", m.group(9), len(zero_before))
        NCHECK_inc()
        if _a_list(m.group(10)) != zero_before:
            FAILS.append(f"t2 worked: the text names {_a_list(m.group(10))} as 0 before, the stored results say {zero_before}")
        for lab, lit, which in (("before", m.group(11), "n_before"), ("after", m.group(12), "n_after")):
            _frac(f"t2 worked zero cells {lab}", lit, 0, per_list[0][which])
        NCHECK_inc()
        if any(per_list[i - 1]["before_successes"] != 0 or per_list[i - 1]["after_successes"] != 0
               or per_list[i - 1]["n_before"] != per_list[i - 1]["n_after"] for i in zero_before):
            FAILS.append("t2 worked: an attack named as 0 before is not 0/n before and 0/n after")

    # ---- 0/10 cells and the bound
    p = _para1("t2 zero cells", "\\lead{Read the 0/10 cells with care.}")
    cells = zb["attack_cells_with_zero_successes"]
    n0 = cells[0]["n"]
    m = _rx("t2 zero cells", r"\\lead\{Read the (\d+/\d+) cells with care\.\} A (\d+/\d+) cell does not mean zero risk: the true "
            r"success rate could be up to ([\d.]+) \(Clopper-Pearson bound, Key terms\)\. Run-to-run variance was small: "
            r"every cell was (\d+/\d+) or (\d+/\d+) except role-play \((\d+/\d+) before\)\.", p)
    if m:
        _frac("t2 zero cells lead", m.group(1), 0, n0)
        _frac("t2 zero cells body", m.group(2), 0, n0)
        NCHECK_inc()
        if any(c["successes"] != 0 or c["n"] != n0 for c in cells):
            FAILS.append("t2 zero cells: the stored zero-success cells do not share one n")
        cell_check("t2 zero bound (stored)", m.group(3), cells[0]["one_sided_95_upper"])
        cell_check("t2 zero bound (recomputed 1-0.05^(1/n))", m.group(3), _cp_upper_zero(n0))
        NCHECK_inc()
        if "beta.ppf(0.95" not in zb["one_sided_95_upper_formula"]:
            FAILS.append("t2 zero cells: the stored bound is not the one-sided 95% Clopper-Pearson formula")
        pairs = [(k, arm, rec["successes"], rec["runs"]) for arm, dd in (("before", bef), ("after", aft)) for k, rec in dd.items()]
        odd = [(k, arm, s_, n_) for k, arm, s_, n_ in pairs if s_ not in (0, n_)]
        _frac("t2 zero cells all-or-none 0", m.group(4), 0, runs)
        _frac("t2 zero cells all-or-none full", m.group(5), runs, runs)
        NCHECK_inc()
        if len(odd) != 1 or odd[0][:2] != ("2B-roleplay", "before"):
            FAILS.append(f"t2 zero cells: 'every cell was 0 or {runs} except role-play' is contradicted, exceptions: {odd}")
        else:
            _frac("t2 zero cells role-play", m.group(6), odd[0][2], odd[0][3])

    # ---- Each layer stopped different attacks
    p = _para1("t2 layers", "\\lead{Each layer stopped different attacks.}")
    m = _rx("t2 layers", r"but blocked all (\d+) role-play \(S(\d+)\) and many-shot \(S(\d+)\) trials\..*?The output filter "
            r"stopped (\d+/\d+) of the KB secret theft", p)
    if m:
        byatk = ge["by_attack"]
        for key, g in (("2B-roleplay", m.group(2)), ("2B-manyshot", m.group(3))):
            tr = aft[key]["trials"]
            NCHECK_inc()
            if not (len(tr) == int(m.group(1)) and all(any("llama_guard" in x for x in (t.get("block_layers") or [])) for t in tr)):
                FAILS.append(f"t2 layers: not all {m.group(1)} {key} trials were blocked by Llama Guard")
            codes_k = sorted(byatk[key])
            NCHECK_inc()
            if codes_k != [f"S{g}"] or byatk[key][f"S{g}"] != len(tr):
                FAILS.append(f"t2 layers: {key} guard code S{g} does not match the stored records {byatk[key]}")
        int_cell_check("t2 layers trials", m.group(1), runs)
        int_cell_check("t2 layers role-play code", m.group(2), int(next(iter(byatk["2B-roleplay"]))[1:]))
        int_cell_check("t2 layers many-shot code", m.group(3), int(next(iter(byatk["2B-manyshot"]))[1:]))
        n_out = sum(1 for t in aft["2A-kb-secret"]["trials"] if any("secret_code" in x for x in (t.get("block_layers") or [])))
        _frac("t2 layers output filter", m.group(4), n_out, aft["2A-kb-secret"]["runs"])

    # ---- F1/F2
    p = _para1("t2 F1F2", "\\lead{F1/F2: the knowledge base is an instruction channel.}")
    ranks = {}
    for key, pid in (("2A-kb-override", "T9998"), ("2A-kb-secret", "T9997")):
        rs_ = {tr["retrieved_ids"].index(pid) + 1 for tr in bef[key]["trials"]}
        ranks[key] = rs_.pop() if len(rs_) == 1 else -1
    m = _rx("t2 F1F2", r"\(rank~(\d+) and rank~(\d+)\)\. The copilot obeyed it and emitted the marker and the secret, "
            r"(\d+/\d+) each: .*?Guardrails brought both to (\d+/\d+)\. By contrast, the textbook expects role-play to stay open; ours was "
            r"only (\d+/\d+) before, because the (\d+)B model already refuses most jailbreaks\.", p)
    if m:
        int_cell_check("t2 F1F2 planted rank (override)", m.group(1), ranks["2A-kb-override"])
        int_cell_check("t2 F1F2 planted rank (secret)", m.group(2), ranks["2A-kb-secret"])
        p4, p5 = per["2A-kb-override"], per["2A-kb-secret"]
        NCHECK_inc()
        if (p4["before_successes"], p4["n_before"]) != (p5["before_successes"], p5["n_before"]) \
                or (p4["after_successes"], p4["n_after"]) != (p5["after_successes"], p5["n_after"]):
            FAILS.append("t2 F1F2: A4 and A5 do not share one before-count and one after-count, 'each' / 'both' is wrong")
        _frac("t2 F1F2 before", m.group(3), p4["before_successes"], p4["n_before"])
        _frac("t2 F1F2 after", m.group(4), p4["after_successes"], p4["n_after"])
        rp = per["2B-roleplay"]
        _frac("t2 F1F2 role-play before", m.group(5), rp["before_successes"], rp["n_before"])
        _file_has("t2 F1F2 answer-model size", "chatbot/llm.py", rf"LLM_MODEL = \"[^\"]*Llama-3\.1-{m.group(6)}B")
        int_cell_check("t2 F1F2 model size", m.group(6), int(m.group(6)))

    # ---- F3
    p = _para1("t2 F3", "\\lead{F3: guardrails do not reach false content.}")
    m = _rx("t2 F3", r"restated as fact (\d+/\d+) before and (\d+/\d+) after, with no layer firing", p)
    if m:
        p6 = per["2A-kb-misinfo"]
        _frac("t2 F3 before", m.group(1), p6["before_successes"], p6["n_before"])
        _frac("t2 F3 after", m.group(2), p6["after_successes"], p6["n_after"])
        NCHECK_inc()
        if ge["no_block_layer_by_attack"].get("2A-kb-misinfo") != p6["n_after"]:
            FAILS.append("t2 F3: a guard layer fired on the fabricated-technique trials but the text says none did")

    # ---- F7
    p = _para1("t2 F7", "\\lead{F7: retrieval cannot find an exact CVE ID.}")
    a = pr["a_cve_2024_3400"]
    m = _rx("t2 F7 cosine", r"It scores cosine ([\d.]+), while the five returned documents score ([\d.]+) down to ([\d.]+)\. "
            r"The same holds for (\d+/\d+) KEV-ID probes and the (\d+/\d+) benign ID question\.", p)
    if m:
        cell_check("t2 F7 target cosine", m.group(1), a["target_similarity"])
        cell_check("t2 F7 highest returned cosine", m.group(2), max(a["top5_similarities"]))
        cell_check("t2 F7 lowest returned cosine", m.group(3), min(a["top5_similarities"]))
        NCHECK_inc()
        if not (len(a["top5_similarities"]) == pr["top_k"] == 5 and a["in_top5"] is False and a["rank_in_top5"] is None
                and a["target_similarity"] < min(a["top5_similarities"])):
            FAILS.append("t2 F7: the stored probe does not show the exact document outside the five returned, below the fifth")
        b2, b1q = pr["b2_fixed_kev_probe"], pr["b1_benign_questions_naming_ids"]
        _frac("t2 F7 KEV-ID probes", m.group(4), b2["n_exact_doc_in_top5"], b2["n_probes"])
        _frac("t2 F7 benign ID question", m.group(5), b1q["n_exact_doc_in_top5"], b1q["n_naming_an_id"])
        NCHECK_inc()
        if not pr["c_required_action_absent"]["required_action_absent_from_embedded_text"]:
            FAILS.append("t2 F7: the stored text of the document does contain the required action, 'leaves requiredAction out' is wrong")

    # ---- Refusal rate on normal questions, hedges
    p = _para1("t2 refusals", "\\lead{Refusal rate on normal questions.}")
    zb_b = zb["benign_guardrail_refusals"]
    m = _rx("t2 refusals", r"We sent (\w+) normal ATT\\&CK and KEV questions, (\d+) runs each, through the guarded system\. "
            r"\\textbf\{(\d+/\d+) questions were refused \((\d+/\d+) runs\)\}, before and after\. The true refusal rate could "
            r"be up to ([\d.]+) \(same bound\)\.", p)
    if m:
        nq = ben_a["n_questions"]
        _word_is("t2 refusals question count", m.group(1), nq)
        int_cell_check("t2 refusals runs each", m.group(2), ben_a["total_runs"] // nq)
        NCHECK_inc()
        if any(q["runs"] != int(m.group(2)) for q in d["benign"]["after"]["per_question"]):
            FAILS.append("t2 refusals: not every normal question ran the stated number of times")
        for arm, bb in (("after", ben_a), ("before", ben_b)):
            _frac(f"t2 refusals questions ({arm})", m.group(3), bb["questions_with_guardrail_refusal"], bb["n_questions"])
            _frac(f"t2 refusals runs ({arm})", m.group(4), bb["guardrail_refusals"], bb["total_runs"])
        nb = [x for x in zb_b if x["arm"] == "after" or "after" in x["cell"]] or zb_b
        cell_check("t2 refusals bound (stored, after arm)", m.group(5), nb[0]["one_sided_95_upper"])
        cell_check("t2 refusals bound (recomputed 1-0.05^(1/n))", m.group(5), _cp_upper_zero(ben_a["total_runs"]))
        NCHECK_inc()
        if not all(x["successes"] == 0 and x["n"] == ben_a["total_runs"] for x in zb_b):
            FAILS.append("t2 refusals: the stored benign bound cells are not 0/n over the same runs")
        NCHECK_inc()
        if zb["method"] != "exact Clopper-Pearson via scipy.stats.beta":
            FAILS.append("t2 refusals: '(same bound)' but the stored bound method differs")
    p = _para1("t2 hedges", "The only cost we saw is unblocked hedges")
    m = _rx("t2 hedges", r"They were \\textbf\{(\d+/\d+) before the guardrails and (\d+/\d+) after\}, all on the F7 question\.", p)
    if m:
        _frac("t2 hedges before", m.group(1), ben_b["unblocked_hedges"], ben_b["total_runs"])
        _frac("t2 hedges after", m.group(2), ben_a["unblocked_hedges"], ben_a["total_runs"])
        f7q = pr["a_cve_2024_3400"]["question"]
        on_f7 = sum(q["unblocked_hedge"] for q in d["benign"]["after"]["per_question"] if q["question"] == f7q)
        NCHECK_inc()
        if not (on_f7 == ben_a["unblocked_hedges"] and ben_a["questions_with_hedge"] == [f7q]):
            FAILS.append("t2 hedges: the hedges are not all on the CVE-2024-3400 question")

    # ---- Latency
    p = _para1("t2 latency", "\\lead{Cost of the guardrails in time.}")
    m = _rx("t2 latency", r"On the (\w+) KB attacks where the model answered in both arms \((\d+) trials each\), a trial took "
            r"([\d.]+)\\,s guarded against ([\d.]+)\\,s unguarded on average \(ratio ([\d.]+)\)\.", p)
    if m:
        both = [k for k in keys if all(not t["blocked_input"] for t in aft[k]["trials"]) and bef[k].get("seconds") is not None]
        NCHECK_inc()
        if both != ["2A-kb-override", "2A-kb-secret", "2A-kb-misinfo"]:
            FAILS.append(f"t2 latency: attacks where every guarded trial reached the model are {both}")
        tb = sum(bef[k]["seconds"] for k in both)
        ta = sum(aft[k]["seconds"] for k in both)
        nb_, na_ = sum(bef[k]["runs"] for k in both), sum(aft[k]["runs"] for k in both)
        _word_is("t2 latency attack count", m.group(1), len(both))
        NCHECK_inc()
        if nb_ != na_ or any(bef[k]["runs"] != aft[k]["runs"] for k in both):
            FAILS.append("t2 latency: the guarded and unguarded arms do not have the same trials per attack")
        int_cell_check("t2 latency trials per arm", m.group(2), na_)
        cell_check("t2 latency guarded mean", m.group(3), ta / na_)
        cell_check("t2 latency unguarded mean", m.group(4), tb / nb_)
        cell_check("t2 latency ratio", m.group(5), (ta / na_) / (tb / nb_))
        o = ops["b_attack_trials_answered_in_both_arms"]["model_called_in_both_arms_all_trials"]
        NCHECK_inc()
        if not (o["attacks"] == both and abs(o["pooled_mean_seconds_per_trial_after"] - ta / na_) < 1e-9
                and abs(o["pooled_mean_seconds_per_trial_before"] - tb / nb_) < 1e-9
                and abs(o["ratio_after_over_before"] - (ta / na_) / (tb / nb_)) < 1e-9):
            FAILS.append("t2 latency: results/ops_measurements.json copilot_latency disagrees with the recomputation from task2_redteam.json")
        NCHECK_inc()
        if ops["c_guard_share_of_guarded_latency"].get("value") is not None:
            FAILS.append("t2 latency: the stored file now has the guard's own share, 'was not recorded' is wrong")

    # ---- Deploy verdict
    p = _para1("t2 deploy", "\\lead{Would we deploy?} Verdict:")
    m = _rx("t2 deploy", r"Verdict: \\S(\d+) \(advisory use, reviewed ingest\)\. No legitimate question was refused, and F1 and "
            r"F2 are (\d+/\d+) after the guardrails\. The residual F3, (\d+/\d+) after,", p)
    if m:
        no = _sec_no("Executive summary")
        int_cell_check("t2 deploy section pointer", m.group(1), no if no is not None else -1)
        ex7 = [q for q in _paras() if "\\emph{Copilot:} advisory use only, with reviewed ingest" in q]
        NCHECK_inc()
        if len(ex7) != 1:
            FAILS.append("t2 deploy: the executive summary does not carry the copilot verdict the pointer refers to")
        NCHECK_inc()
        if ben_a["guardrail_refusals"] != 0 or ben_b["guardrail_refusals"] != 0:
            FAILS.append("t2 deploy: a legitimate question was refused, 'No legitimate question was refused' is wrong")
        p4, p5, p6 = per["2A-kb-override"], per["2A-kb-secret"], per["2A-kb-misinfo"]
        _frac("t2 deploy F1/F2 after", m.group(2), p4["after_successes"], p4["n_after"])
        NCHECK_inc()
        if (p5["after_successes"], p5["n_after"]) != (p4["after_successes"], p4["n_after"]):
            FAILS.append("t2 deploy: F1 and F2 do not share one after-count")
        _frac("t2 deploy F3 after", m.group(3), p6["after_successes"], p6["n_after"])

    # ---- Scope sentence (OWASP 2025 categories)
    sc = [q for q in _paras() if "\\emph{Scope.} The nine attacks cover" in q]
    NCHECK_inc()
    if len(sc) != 1:
        FAILS.append(f"t2 owasp scope: paragraph matched {len(sc)} times, expected 1")
    else:
        t = str(sc[0])
        names = {"LLM01": "Prompt Injection", "LLM07": "System Prompt Leakage", "LLM09": "Misinformation"}
        m = _rx("t2 owasp scope", r"The (\w+) attacks cover LLM01, LLM07 System Prompt Leakage \(the secret\) and LLM09 "
                r"Misinformation \(the false technique\)\. Not exercised: LLM06 \(no tools\), LLM03 and LLM04 \(models used "
                r"as published\), LLM10 \(no load test\)\.", sc[0])
        if m:
            _word_is("t2 owasp scope attack count", m.group(1), len(keys))
        rolled = set(_re_real.findall(r"(LLM\d\d):2025", ROLL))
        NCHECK_inc()
        if rolled != set(names):
            FAILS.append(f"t2 owasp scope: the roll-up covers {sorted(rolled)}, the scope sentence says {sorted(names)}")
        NCHECK_inc()
        not_ex = set(_re_real.findall(r"LLM\d\d", t.split("Not exercised:")[-1])) if "Not exercised:" in t else set()
        if not_ex != {"LLM06", "LLM03", "LLM04", "LLM10"} or not_ex & rolled:
            FAILS.append(f"t2 owasp scope: 'Not exercised' lists {sorted(not_ex)}, which overlaps the exercised {sorted(rolled)} or is not the stated four")
        NCHECK_inc()
        if any(f"{i} {nm}" not in t for i, nm in names.items() if i != "LLM01") or "LLM05" in t:
            FAILS.append("t2 owasp scope: a 2025 category name is missing or LLM05 appears")
        NCHECK_inc()
        if re.search(r"F7[^.]*(is|are) (an? )?LLM08|F7 \(LLM08\)", t):
            FAILS.append("t2 owasp scope: F7 is stated as LLM08")


def task2_rollup():
    """The generated OWASP roll-up: per-category subtotals and every 'Stopped by' cell are
    recomputed here from the raw trial records (task2_redteam.json), independently of
    scripts/make_task2_evidence.py, and compared row by row."""
    d = load("task2_redteam.json")
    sm = load("task2_summary.json")
    cat_of = {p["key"]: p["owasp_full"] for p in sm["per_attack"]}
    rows, dag = {}, {}
    for ln in _doc_lines("rollup"):
        ln = ln.strip()
        if ln.startswith("\\quad "):
            cells = [c.strip() for c in ln[len("\\quad "):].rstrip("\\ ").split(" & ")]
            marked = cells[0].endswith("$^\\dagger$")
            cells[0] = re.sub(r" \(\d+ examples\)$", "", cells[0].replace("$^\\dagger$", ""))
            rows[cells[0]] = cells
            dag[cells[0]] = marked
    names = {"2A-direct-override": "Direct instruction override",
             "2A-direct-secret": "Direct secret extraction",
             "2A-direct-prompt-exfil": "Direct prompt exfiltration",
             "2A-kb-override": "KB-planted override", "2A-kb-secret": "KB-planted secret theft",
             "2A-kb-misinfo": "Fabricated technique", "2B-roleplay": "Role-play exploit code",
             "2B-base64": "Base64 smuggle", "2B-manyshot": "Many-shot jailbreak"}
    sub = {}
    for rec_b, rec_a in zip(d["before"], d["after"]):
        key = rec_a["key"]
        c = sub.setdefault(cat_of[key], [0, 0, 0])
        c[0] += rec_b["successes"]; c[1] += rec_a["successes"]; c[2] += rec_a["runs"]
        # recompute the layers that fired in the after arm
        fam, none = {}, 0
        for t in rec_a["trials"]:
            ls = t.get("block_layers") or []
            if not ls:
                none += 1
            seen = set()
            for x in ls:
                if x.startswith("input_filter"):
                    seen.add("input filter")
                elif x.startswith("llama_guard"):
                    seen.update("Llama Guard " + c_ for c_ in re.findall(r"S\d+", x))
                elif x == "secret_code":
                    seen.add("output filter")
            for f in seen:
                fam[f] = fam.get(f, 0) + 1
        NCHECK_inc()
        # dagger = "tested against a working attack": derived here from the raw before-arm
        # trial records (an attack that succeeded at least once before the guardrails)
        if dag.get(names[key]) != (rec_b["successes"] > 0 or any(t["success"] for t in rec_b["trials"])):
            FAILS.append(f"rollup {key}: dagger mark is {dag.get(names[key])}, before-arm successes are {rec_b['successes']}")
        cells = rows.get(names[key])
        if cells is None or len(cells) != 3:
            FAILS.append(f"rollup {key}: row '{names[key]}' missing or malformed")
            continue
        stop = _strip_bold(cells[2])
        got = {m.group(1): int(m.group(2))
               for m in re.finditer(r"(input filter|output filter|Llama Guard S\d+) \((\d+)/(\d+)\)", stop)}
        if got != fam:
            FAILS.append(f"rollup {key}: 'Stopped by' says {got}, trial records say {fam}")
        NCHECK_inc()
        if none and not ("no layer fired" in stop or "nothing" in stop and f"({none}/" in stop):
            FAILS.append(f"rollup {key}: {none} trials had no layer but the cell says: {stop}")
        if none and f"({none}/{rec_a['runs']})" not in stop:
            FAILS.append(f"rollup {key}: cell does not state the {none} no-layer trials: {stop}")
    NCHECK_inc()
    if "$^\\dagger$ the attack already succeeded before the guardrails" not in ROLL:
        FAILS.append("rollup: the dagger footnote is missing")
    for cname, (b, a, n) in sub.items():
        NCHECK_inc()
        want = f"{cname.replace('&', chr(92) + '&')}}} --- subtotal {b}/{n} $\\to$ {a}/{n}"
        if want not in ROLL:
            FAILS.append(f"rollup subtotal: expected '{want}' (from raw trials) not in the table")
        sm_c = sm["per_category"][cname]
        if (sm_c["before_successes"], sm_c["after_successes"]) != (b, a):
            FAILS.append(f"rollup subtotal {cname}: summary JSON disagrees with raw trials")
    NCHECK_inc()
    tb = sum(v[0] for v in sub.values()); ta = sum(v[1] for v in sub.values())
    tn = sum(v[2] for v in sub.values())
    if f"{tb}/{tn} $\\to$ {ta}/{tn} successful trials" not in ROLL:
        FAILS.append(f"rollup total: expected {tb}/{tn} -> {ta}/{tn} not in the table")


def _para1(label: str, anchor: str):
    """The single paragraph (squashed Src with line numbers) holding `anchor`; zero or
    several is a FAIL (returns None), never a skip."""
    NCHECK_inc()
    ps = [p for p in _paras() if anchor in p]
    if len(ps) != 1:
        FAILS.append(f"{label}: anchor '{anchor}' matched {len(ps)} paragraphs, expected 1")
        return None
    return ps[0]


def _curve_at(curve, eps):
    return next(r for r in curve if abs(r["eps_std"] - eps) < 1e-9)


def task1_text():
    """Section 'Task 1' (sections/02_task1.tex): every number in the outcome, target,
    threat-model, 1A, and 1B text and tables against results/*.json, scoped to its own
    paragraph or table row, at the rounding the text prints."""
    b1 = load("task1b_evasion_wednesday.json")
    a1 = load("task1a_membership.json")
    days = {x["day"]: x for x in a1["days"]}
    wd, th = days["Wednesday"], days["Thursday"]
    bu = load("task1_attacker_budget.json")
    ex = load("report_extras.json")
    x1 = load("task1_extras.json")["blocks"]
    ops = load("ops_measurements.json")
    df = load("task1c_defense.json")
    ma = load("task1c_membership_after.json")
    raw10 = load("task1b_raw_shift_eps010.json")["raw_radius_eps_0.10_seconds"]
    thu = x1["thursday_evasion"]["variants"]["B_filter_then_sample300"]
    ben = x1["benign_to_attack"]
    grid = {r["eps_std"] for r in b1["curve"]}
    nflows = b1["n_eval_flows"]
    ATK = ("random_search", "fd_fgsm", "pgd")
    w_det = lambda e, a: _curve_at(b1["curve"], e)["attacks"][a]["detection_rate"]
    t_det = lambda e, a: _curve_at(thu["curve"], e)["methods"][a]["detection"]
    wed_mi = wd["arms"][0]["attacks"]["confidence_threshold"]["accuracy"]
    n_bal = wd["eval_index_provenance"]["n_member"] == wd["eval_index_provenance"]["n_nonmember"]
    chance = 1 / 2 if n_bal else -1

    # ---- Outcome paragraph
    p = _para1("t1 outcome", "\\lead{Outcome.}")
    m = _rx("t1 outcome", r"Membership inference failed: ([\d.]+) against a (\d+)\\% guess, below a blind control\. "
            r"Evasion worked on Wednesday's data\. A black-box attacker cut detection to ([\d.]+) at a change "
            r"cap of \$\\vtheta=([\d.]+)\$, and to ([\d.]+) at ([\d.]+)\. .*?A (\d+)-flow Thursday sample agrees", p)
    if m:
        cell_check("t1 outcome MI accuracy", m.group(1), wed_mi)
        cell_check("t1 outcome chance level (%)", m.group(2), chance * 100)
        cell_check("t1 outcome FD-FGSM @eps", m.group(3), w_det(0.25, "fd_fgsm"))
        _eps_lit("t1 outcome eps", m.group(4), grid)
        NCHECK_inc()
        if float(m.group(4)) != 0.25:
            FAILS.append("t1 outcome: the change cap quoted with the 0.350 is not the eps=0.25 cell")
        cell_check("t1 outcome FD-FGSM @0.5", m.group(5), w_det(0.5, "fd_fgsm"))
        _eps_lit("t1 outcome eps2", m.group(6), grid)
        NCHECK_inc()
        if float(m.group(6)) != 0.5:
            FAILS.append("t1 outcome: the second change cap is not the eps=0.5 cell")
        int_cell_check("t1 outcome Thursday sample flows", m.group(7), thu["n_flows"])

    # ---- Target paragraph: hidden layers (source: the Lab 1 training code)
    p = _para1("t1 target", "\\lead{Target.} We attack the \\emph{first version} of the Lab~1 model")
    m = _rx("t1 target hidden units", r"two hidden layers of (\d+) and (\d+) units", p)
    if m:
        try:
            ok = bool(_re_real.search(rf"hidden_layer_sizes=\({m.group(1)}, {m.group(2)}\)",
                                      (ROOT / "lab1/src/task1_supervised.py").read_text()))
        except OSError:
            ok = False
        if not ok:
            FAILS.append(f"t1 target hidden units: ({m.group(1)}, {m.group(2)}) not found in lab1/src/task1_supervised.py")
        else:
            cover(p, [m.group(1), m.group(2)], "config")

    # ---- Data / split / access bullets
    import datetime as _dt
    p = _para1("t1 data bullet", "\\textbf{Data:}")
    ts = ex["split"]["last_train_timestamp"].split()[0]
    d0 = _dt.date.fromisoformat(ts)
    m = _rx("t1 data bullet", r"CICIDS-(\d{4}) network flows from (\w+) (\d+) (\w+) (\d{4}), Benign against Attack, (\d+) features", p)
    if m:
        NCHECK_inc()
        want = (str(d0.year), d0.strftime("%A"), str(d0.day), d0.strftime("%B"), str(d0.year))
        got = tuple(str(m.group(i)) for i in range(1, 6))
        if got != want or b1["day"] != want[1]:
            FAILS.append(f"t1 data bullet: report says {got}, the stored split date/day say {want} / {b1['day']}")
        else:
            cover(p, [m.group(1), m.group(3), m.group(5)], "date")
        int_cell_check("t1 data bullet features", m.group(6), bu["n_features_total"])
    p = _para1("t1 split bullet", "\\textbf{Split:}")
    m = _rx("t1 split bullet", r"(\d+/\d+) by row order: ([\d{},]+) training and ([\d{},]+) test rows", p)
    if m:
        tf = round(ex["split"]["train_fraction"] * 100)
        NCHECK_inc()
        if m.group(1) != f"{tf}/{100 - tf}":
            FAILS.append(f"t1 split bullet: report says {m.group(1)}, the split fraction is {tf}/{100 - tf}")
        else:
            cover(p, [m.group(1)], "config")
        int_cell_check("t1 split train rows", m.group(2), ex["split"]["n_train"])
        int_cell_check("t1 split test rows", m.group(3), ex["split"]["n_test"])
        NCHECK_inc()
        if ex["split"]["n_train"] != round((ex["split"]["n_train"] + ex["split"]["n_test"]) * ex["split"]["train_fraction"]):
            FAILS.append("t1 split bullet: n_train is not the stated fraction of all rows")
    p = _para1("t1 access bullet", "\\textbf{Access:}")
    m = _rx("t1 access bullet", r"the label at the ([\d.]+) threshold", p)
    if m:
        cell_check("t1 access threshold", m.group(1), b1["detect_threshold"])

    # ---- PGD threat-model bullet: FD-FGSM vs PGD on the two-round (shipped, adversarially trained) model
    p = _para1("t1 pgd bullet", "\\textbf{1B PGD-40:}")
    m = _rx("t1 pgd bullet", r"On the two-round model FD-FGSM did better at \$\\vtheta=([\d.]+)\$ \(detection ([\d.]+) against ([\d.]+), see 1C\)", p)
    if m:
        _eps_lit("t1 pgd bullet eps", m.group(1), grid)
        e = float(m.group(1))
        fd2, pg2 = cell(df["curve_after"], e, "fd_fgsm"), cell(df["curve_after"], e, "pgd")
        cell_check("t1 pgd bullet FD-FGSM", m.group(2), fd2)
        cell_check("t1 pgd bullet PGD", m.group(3), pg2)
        NCHECK_inc()
        if not (df["adversarial_training"]["rounds"] == 2 and fd2 < pg2):
            FAILS.append("t1 pgd bullet: JSON is not a two-round model where FD-FGSM beats PGD at that eps")

    # ---- 1A
    p = _para1("t1 1A pool", "disjoint calibration pool")
    m = _rx("t1 1A pool", r"calibration pool \(([\d{},]+) members and non-members per day\)", p)
    if m:
        NCHECK_inc()
        for d in (wd, th):
            if d["eval_index_provenance"]["n_member"] != d["eval_index_provenance"]["n_nonmember"]:
                FAILS.append("t1 1A pool: members and non-members differ in count")
        int_cell_check("t1 1A calibration pool per class", m.group(1), a1["n_calibration_per_class"])
        NCHECK_inc()
        if a1["n_calibration_per_class"] != wd["eval_index_provenance"]["n_member"]:
            FAILS.append("t1 1A pool: calibration pool size differs from the scored members per class")
    p = _para1("t1 1A table lead", "\\lead{Membership-inference accuracy")
    m = _rx("t1 1A table lead", r"accuracy against the (\d+)\\% guess", p)
    if m:
        cell_check("t1 1A table chance level (%)", m.group(1), chance * 100)
    rows = table_rows("t1 1A table", "\\textbf{Day} & \\textbf{Attack}")
    c = row_len("t1 1A two-round row", table_row("t1 1A two-round row", rows, "Wednesday, two-round model &"), 5)
    if c is not None:
        da = ma["defended_target_arm"]
        NCHECK_inc()
        if da["confidence_threshold"]["accuracy"] != da["margin_threshold"]["accuracy"]:
            FAILS.append("t1 1A two-round row: confidence and margin accuracy differ; row label 'confidence / margin' is wrong")
        NCHECK_inc()
        if ma["defended_target_arm"] is None or df["adversarial_training"]["rounds"] != 2:
            FAILS.append("t1 1A two-round row: the stored defended model is not the two-round model")
        cell_check("t1 1A two-round accuracy", c[2], da["confidence_threshold"]["accuracy"])
        ci = da["confidence_threshold"]["ci95"]
        mm = re.fullmatch(r"\[([\d.]+), ([\d.]+)\]", c[3])
        NCHECK_inc()
        if not mm:
            FAILS.append(f"t1 1A two-round CI cell malformed: {c[3]}")
        else:
            cell_check("t1 1A two-round CI low", mm.group(1), ci[0])
            cell_check("t1 1A two-round CI high", mm.group(2), ci[1])
        cell_check("t1 1A two-round blind control", c[4], wd["arms"][1]["attacks"]["confidence_threshold"]["accuracy"])
        NCHECK_inc()
        if not float(c[2]) < float(c[4]):
            FAILS.append("t1 1A two-round row: accuracy is not below the blind control")
    claim("t1 1A why: measured", r"We measured ([\d.]+): above the (\d+)\\% guess but \\emph\{below\} the blind control",
          wed_mi, ("pct", chance), unit="para")
    claim("t1 1A why: attack shares",
          r"attack share ([\d.]+)\\% in Wednesday training, ([\d.]+)\\% in test\)",
          wd["split"]["train_attack_pct"], wd["split"]["test_attack_pct"], unit="para")
    g_w, g_t = wd["generalisation_gap"]["gap"], th["generalisation_gap"]["gap"]
    claim("t1 1A why: gaps and duplicates",
          r"near zero \(\$([+-])(\d\\times10\^\{-\d+\})\$ Wednesday, \$([+-])([\d.]+)\$ Thursday\), and ([\d.]+)\\% of "
          r"Wednesday and ([\d.]+)\\% of Thursday test rows duplicate a training row exactly",
          ("text", "+" if g_w >= 0 else "-"), ("sci", g_w), ("text", "+" if g_t >= 0 else "-"), abs(g_t),
          ("pct", wd["duplicate_control"]["frac_test_exact_duplicates"]),
          ("pct", th["duplicate_control"]["frac_test_exact_duplicates"]), unit="para")

    # ---- 1B: mutable features
    claim("t1 1B mutable split", r"\\textbf\{(\d+) of (\d+) features are mutable\}.*?\\textbf\{(\d+) are not\}",
          bu["n_mutable_features"], bu["n_features_total"], bu["n_immutable_features"], unit="para")
    # 1B: cost line (random search stops at the first success, so its cost is the largest per-eps count)
    srv = lambda a: [_curve_at(b1["curve"], e)["attacks"][a].get("server_measured", {}).get("rows_scored")
                     for e in sorted(grid) if e > 0]
    grads = [_curve_at(b1["curve"], e)["attacks"]["pgd"]["budget"]["gradient_computations"] for e in sorted(grid) if e > 0]
    NCHECK_inc()
    if len(set(srv("fd_fgsm"))) != 1 or len(set(grads)) != 1 or None in srv("random_search"):
        FAILS.append("t1 1B cost line: FD-FGSM queries or PGD gradients differ across eps (the text quotes one number)")
        fd_cost, pgd_cost = -1, -1
    else:
        fd_cost, pgd_cost = srv("fd_fgsm")[0], grads[0]
    claim("t1 1B cost line",
          r"Wednesday uses (\d+) attack flows the model already scores above ([\d.]+)\. .*?Cost per (\d+) flows: "
          r"([\d{},]+) queries for random search, ([\d{},]+) for FD-FGSM, ([\d{},]+) gradients for PGD",
          nflows, b1["confident_threshold"], nflows, max(srv("random_search")), fd_cost, pgd_cost, unit="para")
    NCHECK_inc()
    if b1["eval_set_provenance"]["n"] != nflows:
        FAILS.append("t1 1B: eval set size differs from n_flows")

    # ---- 1B: Thursday paragraph
    key = f"n_attack_test_flows_with_P_gt_{b1['confident_threshold']}"
    tc = x1["thursday_evasion"].get(key)
    NCHECK_inc()
    if tc is None:
        FAILS.append(f"t1 Thursday: '{key}' missing from task1_extras.json")
        tc = {"count": -1, "n": -1}
    f25 = lambda a: t_det(0.25, a)
    claim("t1 Thursday paragraph",
          r"\\lead\{Thursday: a (\d+)-flow sample, not the whole day\.\}.*?We repeated 1B on the Thursday model\. "
          r"Only ([\d{},]+) of ([\d{},]+) Thursday attack test flows score above ([\d.]+)\. We kept those and drew "
          r"(\d+) with seed (\d+), so .*?At \$\\vtheta=([\d.]+)\$ the detector still caught (\d+/\d+) under random "
          r"search, but only (\d+/\d+) under FD-FGSM and (\d+/\d+) under PGD-40",
          thu["n_flows"], tc["count"], x1["thursday_evasion"]["n_attack_test_flows"], b1["confident_threshold"],
          thu["n_flows"], x1["thursday_evasion"]["seed"], ("eps", grid),
          ("frac", f25("random_search")["count"], f25("random_search")["n"]),
          ("frac", f25("fd_fgsm")["count"], f25("fd_fgsm")["n"]),
          ("frac", f25("pgd")["count"], f25("pgd")["n"]), unit="para")
    NCHECK_inc()
    if tc["n"] != x1["thursday_evasion"]["n_attack_test_flows"] or not tc["count"] >= thu["n_flows"]:
        FAILS.append("t1 Thursday: the P>0.9 pool is not larger than the 300-flow draw, or its denominator differs")

    # ---- 1B table (Wednesday and Thursday columns)
    rows = table_rows("t1 1B table", "\\textbf{Thursday sample}")
    for eps in (0.010, 0.025, 0.050, 0.100, 0.150, 0.250, 0.500):
        lab = f"t1 1B eps={eps:.3f}"
        c = row_len(lab, table_row(lab, rows, f"{eps:.3f} &"), 7)
        if c is None:
            continue
        cell_check(f"{lab} eps cell", c[0], eps)
        for k, a in ((1, "random_search"), (2, "fd_fgsm"), (3, "pgd")):
            cell_check(f"{lab} Wednesday {a}", c[k], w_det(eps, a))
            cell_check(f"{lab} Thursday {a}", c[k + 3], t_det(eps, a)["value"])
    NCHECK_inc()
    if sorted(e for e in grid if e > 0) != [0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5] or \
            sorted(r["eps_std"] for r in thu["curve"]) != sorted(grid):
        FAILS.append("t1 1B table: the stored eps grids (Wednesday, Thursday) differ from the seven rows printed")

    # ---- 1B: benign flows pushed toward attack
    fa = lambda e, a: _curve_at(ben["curve"], e)["methods"][a]["false_alarm"]
    up25 = max(fa(e, a)["count"] for e in sorted(grid) if 0 < e <= 0.25 for a in ATK)
    NCHECK_inc()
    if ben["n_flows"] != nflows or any(fa(0.0, a)["count"] != ben["clean_false_alarm_before_attack"]["count"] for a in ATK):
        FAILS.append("t1 benign: flow count differs from 1B, or the eps=0 false alarms differ from the clean count")
    NCHECK_inc()
    if not fa(0.5, "fd_fgsm")["count"] == max(fa(0.5, a)["count"] for a in ATK):
        FAILS.append("t1 benign: FD-FGSM is not the worst method at eps=0.5, but the text names it")
    claim("t1 benign paragraph",
          r"We reversed the attack on (\d+) Wednesday benign flows \((\d+/\d+) false alarms before the attack\)\. "
          r"Up to \$\\vtheta=([\d.]+)\$ at most (\d+/\d+) became false alarms, for any method\. At \$\\vtheta=([\d.]+)\$, "
          r"(\d+/\d+) did \(FD-FGSM\)",
          ben["n_flows"], ("frac", ben["clean_false_alarm_before_attack"]["count"], ben["clean_false_alarm_before_attack"]["n"]),
          ("eps", grid), ("frac", up25, ben["n_flows"]), ("eps", grid),
          ("frac", fa(0.5, "fd_fgsm")["count"], fa(0.5, "fd_fgsm")["n"]), unit="para")
    NCHECK_inc()
    if not [e for e in grid if abs(e - 0.25) < 1e-9]:
        FAILS.append("t1 benign: 0.25 is not an eps point")

    # ---- 1B: what the curves show
    claim("t1 1B curves", r"Random search barely dents the detector \(([\d.]+) at \$\\vtheta=([\d.]+)\$\)",
          w_det(0.5, "random_search"), ("eps", grid), unit="para")
    NCHECK_inc()
    if any(w_det(e, "fd_fgsm") != w_det(e, "pgd") for e in grid):
        FAILS.append("t1 1B curves: 'FD-FGSM matched PGD-40 at every change size' is not true of the stored curve")

    # ---- 1B: realism in traffic units
    fd25 = _curve_at(b1["curve"], 0.25)["attacks"]["fd_fgsm"]
    first = min(e for e in sorted(grid) if e > 0 and w_det(e, "fd_fgsm") < 1)
    claim("t1 realism",
          r"At \$\\vtheta=([\d.]+)\$ successful changes move (\d+) of (\d+) mutable features\. They shift "
          r"\\texttt\{Fwd IAT Tot\} \(total time between forward packets\) by \$\+([\d.]+)\$\\,s and "
          r"\\texttt\{Flow IAT Max\} \(longest gap between two packets\) by \$\+([\d.]+)\$\\,s\. "
          r"At \$\\vtheta=([\d.]+)\$, where evasion first appears, they still move ([\d.]+)\\,s and ([\d.]+)\\,s",
          ("eps", grid), round(fd25["median_features_changed"]), b1["n_mutable_features"],
          fd25["example_raw_shifts"]["Fwd IAT Tot"] / 1e6, fd25["example_raw_shifts"]["Flow IAT Max"] / 1e6,
          ("tol", first, 1e-9), raw10["Fwd IAT Tot"], raw10["Flow IAT Max"], unit="para")

    # ---- what a real attacker would need to know
    bm = bu["black_box_methods"]
    NCHECK_inc()
    if bm["fd_fgsm"]["server_measured_queries_per_flow_min"] != bm["fd_fgsm"]["server_measured_queries_per_flow_max"] or \
            bm["fd_fgsm"]["n_mutable_x2"] != bm["fd_fgsm"]["server_measured_queries_per_flow_max"]:
        FAILS.append("t1 attacker: FD-FGSM queries per flow are not a constant 2 x mutable features")
    lat = ops["detector_latency"]["models"]["deployed_lab1"]["single_flow"]["median_ms"]
    claim("t1 attacker knowledge",
          r"which (\d+) features they control\..*?A budget of (\d+) queries per flow for FD-FGSM and up to (\d+) for "
          r"random search.*?takes a median ([\d.]+)\\,ms",
          bu["n_mutable_features"], bm["fd_fgsm"]["server_measured_queries_per_flow_max"],
          bm["random_search"]["server_measured_queries_per_flow_max"], lat, unit="para")


def task1_phase2():
    """Defended-model 1A row and sentence, realism claims (44 of 52, +6.7 s, +3.5 s) against
    the FD-FGSM cell at eps=0.25, and the attacker-knowledge paragraph against the budget JSON."""
    m = load("task1c_membership_after.json")
    # (the 1A-on-the-two-round-model number is checked in task1c_text())
    NCHECK_inc()
    if not m["indices_match_original_1a"]["provenance_equals_task1a_json"]:
        FAILS.append("1A defended: eval indices differ from the original 1A")


def wording_checks():
    """Sentence-scoped checks for claims added in the revision: the 0/10 rows, the
    monitoring threshold, the file named next to each Task 1 body claim, the deploy verdicts."""
    sm = load("task2_summary.json")
    # (b) the F1/F2 0/10 baseline is checked in recommendations_text() (05 re-test lines)
    # (c, c2: the defended-gap file name and the Fig. 2 caption are checked in task1c_text())
    # (d) the section 3 deploy verdict is checked in task2_section()


def _expand(labels: str) -> list[int] | None:
    """'A1--A3, A7--A9' -> [1, 2, 3, 7, 8, 9]; None if the text is not in that form."""
    out = []
    for part in [x.strip() for x in labels.split(",")]:
        m = re.fullmatch(r"A(\d+)(?:--A(\d+))?", part)
        if not m:
            return None
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        out += list(range(lo, hi + 1))
    return out


def task2_threat_and_benign():
    """(1) Task 2 threat-model paragraph: the appendix numbers it gives for the user and
    KB channels must match each attack's stored `channel`, in appendix (= JSON) order.
    (2) Its claim that 2C re-ran 2A/2B unchanged: same attacks in the same order with the
    same name/category/channel, and identical per-run seeds, in both arms.
    (3) The generated normal-question rows (appendix): each question's refusal and hedge
    counts against the stored benign records, row-scoped by the question text."""
    d = load("task2_redteam.json")
    paras = [p for p in _paras() if "\\lead{Threat model.} 2A and 2B are black-box" in p]
    NCHECK_inc()
    if len(paras) != 1:
        FAILS.append(f"2 threat model: paragraph matched {len(paras)} times, expected 1")
        return
    t = paras[0]
    NCHECK_inc()
    m = re.search(r"(A\d--A\d(?: and A\d--A\d)?) need only the chat window; (A\d--A\d) also need to add one document", t)
    if not m:
        FAILS.append("2 threat model: the user-channel / KB-channel attack lists are not in the expected form")
    else:
        for chan, lit in (("user", m.group(1).replace(" and ", ", ")), ("kb", m.group(2))):
            got = _expand(lit)
            want = [i for i, r in enumerate(d["before"], 1) if r["channel"] == chan]
            NCHECK_inc()
            if got != want:
                FAILS.append(f"2 threat model {chan} channel: report says {lit} -> {got}, JSON says {want}")
            else:
                cover(m.group(1) if chan == "user" else m.group(2), [lit], "channel_labels")
    NCHECK_inc()
    if "the same payloads and seeds are re-run, so only the guardrails differ" not in t:
        FAILS.append("2 threat model: the unchanged re-run claim is missing")
    NCHECK_inc()
    fields = ("key", "name", "owasp", "channel", "runs")
    sig = lambda arm: [tuple(r[f] for f in fields) for r in d[arm]]
    if sig("before") != sig("after"):
        FAILS.append("2 unchanged re-run: the two arms do not run the same attacks")
    NCHECK_inc()
    for b, a in zip(d["before"], d["after"]):
        if [x["seed"] for x in b["trials"]] != [x["seed"] for x in a["trials"]]:
            FAILS.append(f"2 unchanged re-run: {b['key']} seeds differ between arms")
    # (3) normal-question rows in the generated appendix
    pq = d["benign"]["after"]["per_question"]
    lines = [_squash(l) for l in _doc_lines("payloads")]
    hdr = [l for l in lines if "normal questions (refusal test)" in l]
    NCHECK_inc()
    m = re.search(r"The (\d+) normal questions \(refusal test\)\} --- guarded system, "
                  r"(\d+) runs each", hdr[0]) if len(hdr) == 1 else None
    if not m:
        FAILS.append(f"2 normal questions: header matched {len(hdr)} lines or bad form")
    else:
        int_cell_check("2 normal questions count", m.group(1), len(pq))
        int_cell_check("2 normal questions runs each", m.group(2), pq[0]["runs"])
    for j, q in enumerate(pq, 1):
        qt = q["question"].replace("&", "\\&")
        hits = [l for l in lines if l.startswith(f"\\item[Q{j}] ") and qt in l]
        NCHECK_inc()
        if len(hits) != 1:
            FAILS.append(f"2 normal question Q{j}: row matched {len(hits)} lines, expected 1")
            continue
        m = re.search(r"refused (\d+)/(\d+), unblocked hedges (\d+)/(\d+)\.", hits[0])
        if not m:
            FAILS.append(f"2 normal question Q{j}: counts not in the expected form")
            continue
        for lab, got, want in ((f"Q{j} refused", m.group(1), q["guardrail_refusal"]),
                               (f"Q{j} runs", m.group(2), q["runs"]),
                               (f"Q{j} hedges", m.group(3), q["unblocked_hedge"]),
                               (f"Q{j} runs (hedge)", m.group(4), q["runs"])):
            int_cell_check(f"2 normal question {lab}", got, want)


def task2_owasp_filing():
    """The jailbreaks are filed under LLM01 (OWASP 2025: jailbreaking is prompt injection).
    Checked row by row: the category stored on every raw trial record, the LLM01 block of the
    roll-up (header subtotal and exactly its five attack rows), the payload-appendix headings
    and the report sentence. LLM05 must appear nowhere."""
    d = load("task2_redteam.json")
    want_keys = ["2A-direct-override", "2A-kb-override", "2B-roleplay", "2B-base64",
                 "2B-manyshot"]
    names = {"2A-direct-override": "Direct instruction override",
             "2A-kb-override": "KB-planted override", "2B-roleplay": "Role-play exploit code",
             "2B-base64": "Base64 smuggle", "2B-manyshot": "Many-shot jailbreak"}
    b = a = n = 0
    for rb, ra in zip(d["before"], d["after"]):
        for r in (rb, ra):
            NCHECK_inc()
            if r["key"] in want_keys and not r["owasp"].startswith("LLM01"):
                FAILS.append(f"owasp filing: {r['key']} stored as {r['owasp']}, want LLM01")
            if r["key"] not in want_keys and r["owasp"].startswith("LLM01"):
                FAILS.append(f"owasp filing: {r['key']} unexpectedly stored under LLM01")
        if rb["owasp"].startswith("LLM01"):
            b += rb["successes"]; a += ra["successes"]; n += ra["runs"]
    # roll-up: the LLM01 block, from its header to the next addlinespace
    lines = [l.strip() for l in _doc_lines("rollup")]
    hdr = [i for i, l in enumerate(lines) if "LLM01:2025 Prompt Injection" in l]
    NCHECK_inc()
    if len(hdr) != 1:
        FAILS.append(f"owasp filing: LLM01 header appears {len(hdr)} times in the roll-up")
        return
    i = hdr[0]
    NCHECK_inc()
    want = (f"\\multicolumn{{3}}{{@{{}}l}}{{\\textbf{{LLM01:2025 Prompt Injection}} --- "
            f"subtotal {b}/{n} $\\to$ {a}/{n}}} \\\\")
    if lines[i] != want:
        FAILS.append(f"owasp filing: LLM01 header row is '{lines[i]}', want '{want}'")
    block = []
    for l in lines[i + 1:]:
        if not l.startswith("\\quad "):
            break
        block.append(re.sub(r" \(\d+ examples\)$", "", l[len("\\quad "):].split(" & ")[0].replace("$^\\dagger$", "").strip()))
    NCHECK_inc()
    if sorted(block) != sorted(names[k] for k in want_keys):
        FAILS.append(f"owasp filing: LLM01 block rows are {block}")
    NCHECK_inc()
    if "LLM05" in ROLL or "Improper Output" in ROLL:
        FAILS.append("owasp filing: LLM05 present in the roll-up")
    NCHECK_inc()
    if "LLM05" in TEX or "Improper Output" in TEX:
        FAILS.append("owasp filing: LLM05 present in report.tex")
    NCHECK_inc()
    ps = [p for p in _paras() if "It files jailbreaking under LLM01, so A7--A9 sit there." in p]
    if len(ps) != 1:
        FAILS.append("owasp filing: the LLM01 mapping sentence ('It files jailbreaking under LLM01, so A7--A9 sit there.') "
                     f"matched {len(ps)} paragraphs, expected 1")
    else:
        lo, hi = [int(x) for x in re.search(r"A(\d)--A(\d) sit there", ps[0]).groups()]
        filed = [i for i, r in enumerate(d["before"], 1) if r["owasp"].startswith("LLM01") and r["key"].startswith("2B-")]
        if list(range(lo, hi + 1)) != filed:
            FAILS.append(f"owasp filing: the text puts A{lo}--A{hi} under LLM01, the stored 2B attacks filed there are {filed}")
    for key, nm in (("A7", "Role-play jailbreak"), ("A9", "Many-shot jailbreak")):
        NCHECK_inc()
        ls = [l for l in _doc_lines("payloads") if f"\\textbf{{{key}. {nm}" in l]
        if len(ls) != 1 or "LLM01:2025 Prompt Injection;" not in ls[0]:
            FAILS.append(f"owasp filing: payload heading {key} not filed under LLM01")


def task2_excerpts():
    """Every stored-response excerpt printed in the appendix (\\texttt after Before/After) must
    be a verbatim, whitespace-collapsed substring of a stored response of that attack and
    arm. A6 and A7 must be long enough to contain the text that satisfied the success rule."""
    d = load("task2_redteam.json")
    unesc = [("\\allowbreak{}", ""), ("\\textbackslash{}", "\\"), ("\\textless{}", "<"),
             ("\\textgreater{}", ">"), ("\\textbar{}", "|"), ("\\textasciitilde{}", "~"),
             ("\\textasciicircum{}", "^"), ("\\&", "&"), ("\\%", "%"), ("\\$", "$"),
             ("\\#", "#"), ("\\_", "_"), ("\\{", "{"), ("\\}", "}")]

    def plain(t):
        for a, b in unesc:
            t = t.replace(a, b)
        return t

    sq = lambda t: " ".join(t.split())
    keys = [r["key"] for r in d["before"]]
    lines = _doc_lines("payloads")
    heads = [(i, l) for i, l in enumerate(lines) if "\\textbf{A" in l and ". " in l]
    entry_keys = {}
    for n, (i, l) in enumerate(heads):
        m = re.search(r"\\textbf\{A(\d+)(?:/A(\d+))?\.", l)
        entry_keys[i] = [int(x) for x in m.groups() if x]
    for n, (i, l) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        nums = entry_keys[i]
        for ln in lines[i:end]:
            for arm in ("Before", "After"):
                if not ln.startswith(f"\\item[{arm}]"):
                    continue
                ln2 = ln.replace("\\allowbreak{}", "")
                for ex in re.findall(r"\\texttt\{((?:[^{}]|\\[a-z]+\{\}|\\\{|\\\})*)\}", ln2):
                    txt = plain(ex)
                    core = txt.strip(".").strip()
                    NCHECK_inc()
                    pool = [sq(t["response"]) for k in nums
                            for t in d[arm.lower()][k - 1]["trials"]]
                    if not any(core in r for r in pool):
                        FAILS.append(f"excerpt ({arm}, A{nums}) is not verbatim in the stored "
                                     f"responses: {core[:70]}")
    # the long excerpts must contain the evidence for the success rule
    for num, arm, needle in ((6, "Before", "Quantum Keystroke Harvesting"), (6, "After", "T9999"),
                             (7, "Before", "import os")):
        NCHECK_inc()
        i = next((i for i, l in heads if f"\\textbf{{A{num}." in l), None)
        blk = lines[i:i + 12] if i is not None else []
        bl = [x for x in blk if x.startswith(f"\\item[{arm}]")]
        if not bl or needle.replace(" ", "") not in plain(bl[0].replace("\\allowbreak{}", "")).replace(" ", ""):
            FAILS.append(f"A{num} {arm} excerpt does not show '{needle}'")


def fixes_10_12():
    """Sentence-scoped checks for the 0/n bounds, the eps=0.10 raw size and the smoothed
    detection at eps=0.15 (JSONs: task2_zero_rate_bounds, task1b_raw_shift_eps010,
    task1c_smoothing_detection)."""
    global NCHECK
    zb = load("task2_zero_rate_bounds.json")
    att = zb["attack_cells_with_zero_successes"]
    ben = zb["benign_guardrail_refusals"]
    # (Fix 10, the 0/n bounds sentence, is checked in task2_section())
    # (Fix 12, the smoothed-detection bullet, is checked in task1c_text())
    # Fix 14 (noise clause): the shipped-config gap is significant, so it must stay out
    nc = load("noise_check.json")
    NCHECK_inc()
    if "inside the noise" in TEX and nc["mcnemar_exact"]["significant_at_0.05"]:
        FAILS.append("noise clause present although McNemar p="
                     f"{nc['mcnemar_exact']['p_value_two_sided']:.4f} < 0.05")


# ---------------------------------------------------------------------------------------
# Coverage closure: every remaining measured number in the prose, row-by-row or
# sentence-by-sentence, against a results file or a derivation printed here. The coverage
# tool (scripts/number_coverage.py) reports a number that no check reaches as UNBACKED.
# ---------------------------------------------------------------------------------------
def _units(unit):
    if unit == "para":
        return _paras()
    return [_squash(l) for l in _doc_lines("report") if not l.lstrip().startswith("%")]


def claim(label, pat, *vals, unit="line", count=1):
    """Find the `count` report line(s)/paragraph(s) matching `pat` (a regex whose groups are
    the printed numbers) and check each group against its value. Zero or a different
    number of matches is a FAILURE. A value is a number (float at its printed precision, int
    exactly) or a tuple: ("pct", v) fraction printed as a percentage, ("frac", a, b) "a/b",
    ("sci", v) a x 10^e, ("eps", grid) membership of a configured grid, ("tol", v, t)."""
    global NCHECK
    NCHECK += 1
    hits = [(u, re.search(pat, u)) for u in _units(unit)]
    hits = [(u, m) for u, m in hits if m]
    if len(hits) != count:
        FAILS.append(f"{label}: pattern '{pat}' matched {len(hits)} {unit}(s), expected {count}")
        return
    for u, m in hits:
        for gi, v in enumerate(vals, 1):
            g = m.group(gi)
            lab = f"{label} [{gi}]"
            if isinstance(v, tuple):
                kind = v[0]
                if kind == "pct":
                    cell_check(lab, g, v[1] * 100, v[2] if len(v) > 2 else None)
                elif kind == "tol":
                    cell_check(lab, g, v[1], v[2])
                elif kind == "frac":
                    NCHECK_inc()
                    cover(u, [g]) if g == f"{v[1]}/{v[2]}" else FAILS.append(
                        f"{lab}: report says {g}, results say {v[1]}/{v[2]}")
                elif kind == "eps":
                    NCHECK_inc()
                    try:
                        ok = any(abs(float(g) - e) < 1e-9 for e in v[1])
                    except ValueError:
                        ok = False
                    cover(u, [g]) if ok else FAILS.append(
                        f"{lab}: {g} is not in the configured grid {sorted(v[1])}")
                elif kind == "sci":
                    NCHECK_inc()
                    mm = re.fullmatch(r"(\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}", g)
                    if not mm:
                        FAILS.append(f"{lab}: '{g}' is not a x10^e literal")
                        continue
                    mant, ex = float(mm.group(1)), int(mm.group(2))
                    dec = len(mm.group(1).split(".")[1]) if "." in mm.group(1) else 0
                    if abs(round(abs(v[1]) / 10 ** ex, dec) - mant) > 10 ** -(dec + 4):
                        FAILS.append(f"{lab}: report says {g}, results say {v[1]:.3e}")
                    else:
                        cover(u, [g])
                elif kind == "text":   # exact label text verified elsewhere in the gate
                    NCHECK_inc()
                    cover(u, [g]) if g == v[1] else FAILS.append(f"{lab}: {g!r} != {v[1]!r}")
            elif isinstance(v, int) and not isinstance(v, bool):
                int_cell_check(lab, g, v)
            else:
                cell_check(lab, g, v)


def cfg(label, pat, relfile, filepat, unit="line", group=0):
    """A configuration value quoted in the text must still be what the code says: the
    regex `filepat` must match the file `relfile`; only then is the quoted span covered."""
    global NCHECK
    NCHECK += 1
    hits = [(u, re.search(pat, u)) for u in _units(unit)]
    hits = [(u, m) for u, m in hits if m]
    if not hits:
        FAILS.append(f"{label}: pattern '{pat}' not found in the report")
        return
    try:
        ok = bool(_re_real.search(filepat, (ROOT / relfile).read_text(), _re_real.S))
    except OSError:
        ok = False
    if not ok:
        FAILS.append(f"{label}: '{filepat}' not found in {relfile}")
        return
    for u, m in hits:
        cover(u, [m.group(group)], "config")


def _closure_vars():
    b1 = load("task1b_evasion_wednesday.json")
    df = load("task1c_defense.json")
    a1 = {x["day"]: x for x in load("task1a_membership.json")["days"]}
    ma = load("task1c_membership_after.json")
    bu = load("task1_attacker_budget.json")
    t2 = load("task2_redteam.json")
    sm = load("task2_summary.json")
    pr = load("task2_retrieval_probe.json")
    zb = load("task2_zero_rate_bounds.json")
    sdet = load("task1c_smoothing_detection.json")
    ex = load("report_extras.json")
    abl = {c["config"]: c for c in load("task1c_ablation.json")["configs"]
           + load("task1c_ablation_rounds.json")["configs"]}
    rnd = load("task1c_ablation_rounds.json")
    lab1 = next(r for r in json.loads((ROOT / "lab1/outputs/tables/task1_metrics_comparison.json")
                                      .read_text())
                if r["model_name"] == "MLP_NeuralNet" and "Wednesday" in r["dataset_name"])
    grid = {r["eps_std"] for r in b1["curve"]}
    wd = a1["Wednesday"]
    nflows = b1["n_eval_flows"]
    per = {p["key"]: p for p in sm["per_attack"]}
    ben = sm["benign"]["after"]
    cm = lambda c, e, a: cell(abl[c]["curve_after"], e, a)

    cb = pr["b1_benign_questions_naming_ids"]
    clean_b, clean_a = df["clean_accuracy"]["before"], df["clean_accuracy"]["after"]
    ntr, nte = ex["split"]["n_train"], ex["split"]["n_test"]
    sm1 = df["smoothing"]
    cr = rnd["crossing_by_round"]
    from statistics import NormalDist
    return (b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist,)

def closure_part1():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # (Task 1 target paragraph is checked in task1_text())


def closure_part2():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # ---- 1A
    NCHECK_inc()
    n_m, n_n = wd["eval_index_provenance"]["n_member"], wd["eval_index_provenance"]["n_nonmember"]
    if n_m != n_n:
        FAILS.append("1A: member and non-member counts differ, so 50% is not the chance rate")
    NCHECK_inc()
    if not all(set(a1[d]["arms"][0]["attacks"]) >= {"confidence_threshold"} and
               "ci95" in a1[d]["arms"][0]["attacks"]["confidence_threshold"] for d in a1):
        FAILS.append("1A: stored arms carry no ci95")
    claim("1A CI label", r"\\textbf\{(95)\\% CI\}", ("text", "95"))
    cfg("1A CI config", r"\\textbf\{(95)\\% CI\}", "detector/membership.py",
        r'"ci95": \[lo, hi\]')
    for day in ("Wednesday", "Thursday"):
        ci = a1[day]["arms"][0]["attacks"]["confidence_threshold"]["ci95"]
        claim(f"1A {day} CI", rf"^{day}\s+& confidence / margin / loss & [\d.]+ & \[([\d.]+), ([\d.]+)\]",
              ci[0], ci[1])


def closure_part3():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # ---- 1B: checked in task1_text()


def closure_part4():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # ---- 1C: every number in the prose and the two tables is checked in task1c_text(). What
    # stays here: the scan of every change-size label, the stored scale-effect invariants the
    # Cause 1 explanation rests on, and the guards against figures that must not come back.
    # eps labels in table headers and prose: every \vtheta=... / \vtheta{=}... is a grid value
    for i, u in enumerate(_units("line")):
        for m in re.finditer(r"\\vtheta\s*(?:=|\{=\})\s*([\d.]+)", u):
            NCHECK_inc()
            if not any(abs(float(m.group(1)) - e) < 1e-9 for e in grid):
                FAILS.append(f"eps label {m.group(1)} is not in the evaluated grid")
            else:
                cover(u, [m.group(1)], "eps_label")
    se = load("task1c_scale_effect.json")
    sv_i, sv_ii = se["sensitivity_defended_i_own_units"], se["sensitivity_defended_ii_real_box"]
    sdg, srl = se["summary_degenerate"], se["summary_real"]
    dnames = se["degenerate_mutable_features"]
    NCHECK_inc()
    if not (dnames == ex["scalers"]["degenerate_mutable_features"]
            and [se["per_feature"][n]["sb"] for n in dnames] == ex["scalers"]["original_scale_on_them"]
            and [se["per_feature"][n]["sd"] for n in dnames] == ex["scalers"]["defended_scale_on_them"]
            and se["verdict"].startswith("DOES NOT CANCEL")
            and not se["scale_cancels_on_degenerate"] and se["scale_cancels_on_real"] is False
            and abs(sdg["std_move_defended"]["median"] - se["eps_std"] * sdg["sb_over_sd"]["median"]) < 1e-9
            and sdg["sb_over_sd"]["min"] == sdg["sb_over_sd"]["max"]):
        FAILS.append("cause1: scale_effect.json no longer says the real box is a different "
                     "standardized move on the five features (or disagrees with report_extras)")
    NCHECK_inc()
    # direction: each of the five can lower the logit in some box direction, and the lower
    # corner mirrors the upper one (so the attacker is free to pick the sign)
    up, lo = sv_ii["signed_mean_upper_by_feature"], sv_ii["signed_mean_lower_by_feature"]
    if not (all(up[n] < 0 or lo[n] < 0 for n in dnames)
            and all(abs(up[n] + lo[n]) <= 0.01 * abs(up[n]) for n in dnames)):
        FAILS.append("cause1: stored signed sensitivities do not support 'moving down reverses the sign'")
    NCHECK_inc()
    sun = se["sensitivity_undefended"]
    if not (se["undefended_i_equals_ii"] and sun["degenerate_max"] == 0.0 and sun["degenerate_min"] == 0.0
            and all(v == 0.0 for v in sun["degenerate_by_feature"].values())
            and se["undefended_std_move_equals_eps"]):
        FAILS.append("cause1: stored undefended sensitivity is not exactly zero on the five features")
    NCHECK_inc()
    sx = ex["sensitivity"]
    if not (sx["undefended_all_exactly_zero"]
            and all(v == 0.0 for v in sx["undefended_by_feature"].values())
            and list(sx["defended_by_feature"]) == dnames
            and sx["defended_degenerate_min"] == sv_i["degenerate_min"]
            and sx["defended_degenerate_max"] == sv_i["degenerate_max"]):
        FAILS.append("cause1: report_extras sensitivity disagrees with scale_effect (i) or is not zero undefended")
    pg = {r["eps_std"]: r for r in se["pgd_scale_test"]["curve"]}
    NCHECK_inc()
    if not (pg[0.15]["all_mutable_real_box"]["evasion_rate"]
            == {r["eps_std"]: r for r in ex["restricted_pgd"]["curve"]}[0.15]["all_mutable"]["evasion_rate"]):
        FAILS.append("cause1: PGD scale test disagrees with report_extras restricted_pgd at eps=0.15")
    # values that must NOT come back (unreproducible earlier figures)
    for lab, pat in (("0.42 lower end", r"(?<![\d.])0\.42(?![\d])"), ("median 0.50", r"median\s+0\.50"),
                     ("0.42--0.86", r"0\.42\s*--\s*0\.86"), ("mean-logit pair", r"31\.2|14\.7"),
                     ("mean-logit wording", r"mean logit")):
        NCHECK_inc()
        if re.search(pat, TEX):
            FAILS.append(f"report: removed unreproducible figure is back ({lab})")


def closure_part5():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # ---- Task 2 prose: checked in task2_section(); only the configuration cross-checks stay here
    cfg("t2 temperature config", r"temperature (0\.7)", "chatbot/llm.py", r"temperature: float = 0\.7")
    cfg("t2 top_p config", r"top\\_p (0\.9)", "chatbot/llm.py", r"top_p: float = 0\.9")
    cfg("t2 secret config", r"secret \\texttt\{(THUNDER-42)\}", "chatbot/prompts.py", r"THUNDER-42")
    # (BM25 / hybrid names handled by the allowlist; the 0/n bounds are checked in task2_section())
    NCHECK_inc()
    if "radius_p25" not in sm1:
        FAILS.append("smooth: no radius_p25 in the stored certificate")
    claim("smooth p25 label", r"\(p(25) [\d.]+,", ("text", "25"))
    # (the 0/10 baseline in the recommendations is checked in recommendations_text())
    NCHECK_inc()
    if per["2A-kb-override"]["after_successes"] != per["2A-kb-secret"]["after_successes"] or per["2A-kb-override"]["n_after"] != per["2A-kb-secret"]["n_after"]:
        FAILS.append("t2 monitoring: F1 and F2 do not share one after-count")


def closure_part6():
    b1, df, a1, ma, bu, t2, sm, pr, zb, sdet, ex, abl, rnd, lab1, grid, wd, nflows, per, ben, cm, cb, clean_b, clean_a, ntr, nte, sm1, cr, NormalDist, = _closure_vars()
    # ---- labels that name a measured thing: verify the label against the stored record
    # (the round labels of the crossing fractions are checked in task1c_text(), '1c cause 2 round label')
    tid = pr["a_cve_2024_3400"]["target_id"]
    claim("F7 CVE id", r"required action for (CVE-\d+-\d+)\?''.*document \\texttt\{kev-(CVE-\d+-\d+)\}",
          ("text", tid.replace("kev-", "")), ("text", tid.replace("kev-", "")), unit="para")
    cfg("fabricated technique id", r"``(T9999) Quantum Keystroke", "chatbot/attacks.py",
        r'FAKE_TECHNIQUE_ID = "T9999"')
    NCHECK_inc()
    if a1["Wednesday"]["day"] != "Wednesday":
        FAILS.append("task1: stored day is not Wednesday")
    # (the Llama Guard category codes are checked in task2_section())
    # appendix numbers named in the threat-model paragraph and the recommendation tags
    prio = {t: int(r["prio"]) for t, r in _risk_cells().items()}   # parsed from the section-4 table
    for u in _units("line"):
        for mm in re.finditer(r"\\textbf\{\(((?:F\d(?:--F\d)?)(?:, F\d)*), (P\d(?:/P\d)*)\)\}", u):
            NCHECK_inc()
            fs = []
            for part in [x.strip() for x in mm.group(1).split(",")]:
                lo, _, hi = part.partition("--")
                fs += [f"F{i}" for i in range(int(lo[1:]), int((hi or lo)[1:]) + 1)]
            ps = [int(x.replace("P", "")) for x in mm.group(2).split("/")]
            if any(f not in prio for f in fs) or [prio[f] for f in fs] != ps:
                FAILS.append(f"recommendation tag ({mm.group(1)}, {mm.group(2)}): the risk table's "
                             f"priorities are {[prio.get(f) for f in fs]}")
            else:
                cover(u, [mm.group(0)], "priority_tag")
    # the brief's numbers are quoted, so they must be in the brief


def coverage_closure():
    """Runs each closure part on its own so that one broken anchor aborts only its part."""
    for fn in (closure_part1, closure_part2, closure_part3, closure_part4, closure_part5, closure_part6):
        _CURRENT[0] = fn.__name__
        try:
            fn()
        except Exception as e:
            FAILS.append(f"{fn.__name__}: check aborted ({type(e).__name__}: {e}); its remaining checks did not run")


def fix5():
    """Sentence-scoped checks: every number-bearing sentence, scoped to
    its own sentence/paragraph, with every comparison word ('most', 'stronger', 'at least as
    good', 'only', 'more than doubles') computed here from the stored JSON."""
    se = load("task1c_scale_effect.json")
    b1 = load("task1b_evasion_wednesday.json")
    df = load("task1c_defense.json")
    sm = load("task2_summary.json")
    zb = load("task2_zero_rate_bounds.json")
    abl = {c["config"]: c for c in load("task1c_ablation.json")["configs"]
           + load("task1c_ablation_rounds.json")["configs"]}
    grid = {r["eps_std"] for r in b1["curve"]}
    und = lambda e, a: cell(b1["curve"], e, a)
    dfd = lambda e, a: cell(df["curve_after"], e, a)
    sr = scale_ref()["ref"]
    ATT = ("fd_fgsm", "pgd")

    # --- the word 'halved' stays out: no computed ratio supports it
    NCHECK_inc()
    if re.search(r"halved|halves", TEX):
        FAILS.append("report: 'halved' is back; no computed ratio supports it")
    # (2.3: the scale effect at 0.15 vs the collapse at 0.25 is checked in task1c_text())
    # --- PGD-40 is the white-box reference, not an upper bound
    NCHECK_inc()
    if re.search(r"white-box ceiling|an upper bound for a stronger attacker|PGD-40 is an upper bound", TEX):
        FAILS.append("report: PGD-40 is called a ceiling / upper bound")
    # (the two-round FD-FGSM vs PGD-40 sentence is checked in task1_text() and task1c_text())
    NCHECK_inc()
    if not all(abs(und(e, "fd_fgsm") - und(e, "pgd")) < 1e-9 for e in grid):
        FAILS.append("1B text: 'FD-FGSM matches PGD-40 at every eps' (undefended) no longer holds")
    # --- Cause 1 names both measured parts wherever the mechanism is named
    NCHECK_inc()
    sd = se["summary_degenerate"]
    if not (sd["sb_over_sd"]["min"] == sd["sb_over_sd"]["max"] == 8.0):
        FAILS.append("cause1: 'eightfold' / 'eight times' needs sb/sd == 8 on all five features")
    # (the two recommendation/caveat sentences that named both parts are gone; Cause 1 is checked in task1c_text())
    # --- freezing removes both parts; range check on the always-zero features
    cfg("cause1 freeze removes it", r"Freezing the ten features during training removes it",
        "detector/ablate_1c.py", r"train_mutable\[frozen\] = 0\.0", unit="para")
    zv = se["degenerate_training_value"]
    NCHECK_inc()
    if not (list(zv) == se["degenerate_mutable_features"] and all(v == 0.0 for v in zv.values())):
        FAILS.append("cause1: the five features are not all exactly 0 in Lab 1's training data")
    NCHECK_inc()
    c = [q for q in _paras() if "These inputs cost the attacker nothing to move" in q]
    if len(c) != 1 or not ("so the feasibility limits do not constrain them" in c[0]
                           and "the defense created its own attack surface" in c[0]):
        FAILS.append("cause1: the paragraph no longer says the five inputs are free of the feasibility limits")
    # (the range-check recommendation is checked in recommendations_text())
    # --- C6: 'at least as good as undefended everywhere except FD-FGSM at 0.15', over the grid
    def worse(cfgname):
        es = sorted(r["eps_std"] for r in abl[cfgname]["curve_after"])   # the ablation grid (0.05 ... 0.50)
        return {(a, e) for a in ATT for e in es
                if cell(abl[cfgname]["curve_after"], e, a) < und(e, a) - 1e-9}
    ok_cfgs = {c for c in abl if worse(c) <= {("fd_fgsm", 0.15)}}
    NCHECK_inc()
    if not (worse("full_1r_frozen") == {("fd_fgsm", 0.15)} and ok_cfgs == {"full_1r_frozen"}):
        FAILS.append(f"C6: 'the only one at least as good as undefended everywhere except FD-FGSM at 0.15' fails: "
                     f"worse cells {worse('full_1r_frozen')}, configs satisfying it {sorted(ok_cfgs)}")
    NCHECK_inc()
    if not cell(abl["full_1r_frozen"]["curve_after"], 0.25, "pgd") > 2 * und(0.25, "pgd"):
        FAILS.append("C6: 'PGD more than doubles' at 0.25 is false")
    # (the 'at least as good everywhere except FD-FGSM at 0.15' sentence is checked in task1c_text())
    NCHECK_inc()
    if not all(dfd(e, a) < und(e, a) for e in (0.15, 0.25) for a in ("random_search", "fd_fgsm", "pgd")):
        FAILS.append("deploy: 'cut detection under all three attacks at 0.15 and 0.25' is false")
    # --- detector release gate and monitoring use the stronger attack (wording checked in recommendations_text())
    mx = lambda f, e: max(1 - f(e, a) for a in ATT)
    NCHECK_inc()
    if not all(mx(dfd, e) > mx(und, e) for e in (0.15, 0.25)) or not mx(dfd, 0.15) > 1 - dfd(0.15, "pgd"):
        FAILS.append("monitoring: the max-over-attacks gate would not fire on the shipped defense, or equals PGD-40 alone at 0.15")
    # (the Task 2 deploy sentence is checked in task2_section())

REC_NOW_MAX, REC_NEXT_MAX = 4, 6     # the sequence rule stated in the section 5 preamble


def _fcodes(spec: str) -> list:
    """'F1--F3, F5' -> ['F1', 'F2', 'F3', 'F5']."""
    ids = []
    for part in [x.strip() for x in spec.split(",") if x.strip().startswith("F")]:
        lo, _, hi = part.partition("--")
        ids += [f"F{i}" for i in range(int(lo[1:]), int((hi or lo)[1:]) + 1)]
    return ids


def _sec5() -> str:
    a = TEX.index("\\section{Recommendations}")
    return TEX[a:TEX.index("\\section{Evidence quality}", a)]


def recommendation_controls():
    """Section 5 structure: every control has an owner, effort and sequence from the allowed sets;
    every finding whose residual in the risk table is Medium or High has a tagged control; the
    'Now' controls cover every finding ranked first to fourth; the pillars are present; the
    detector-API alert threshold comes from the results JSON by a rule computed here."""
    import math
    OWN, EFF, SEQ = {"ML", "SOC", "SRE", "KB"}, {"S", "M", "L"}, ("Now", "Next", "Later")
    heads = ("\\lead{Knowledge base and pipeline (top recommendation).}", "\\lead{Model.}",
             "\\lead{Monitoring.}", "\\lead{Human-in-the-loop.}", "\\lead{Re-test plan.}")
    sec = _sec5()
    NCHECK_inc()
    for need in heads:
        if need not in sec:
            FAILS.append(f"recommendations: pillar heading '{need}' missing")
    rkc = _risk_cells()
    resid = {t: r["rating"] for t, r in rkc.items()}
    prio = {t: int(r["prio"]) for t, r in rkc.items()}
    NCHECK_inc()
    if sorted(resid) != [f"F{i}" for i in range(1, 9)]:
        FAILS.append(f"recommendations: risk table findings parsed as {sorted(resid)}")
        return
    need_cover = {f for f, l in resid.items() if l in ("Medium", "High")}
    now_must = {f for f, p in prio.items() if p <= 2}
    print("recommendations: residual Medium/High findings:", sorted(need_cover),
          "| priority 1-2 findings:", sorted(now_must))
    items = _re_real.findall(r"\\item (.*?)(?=\n\\item |\n\\end\{itemize\}|\n\\begin\{itemize\})", sec, flags=re.S)
    band_items = [i for i in items if _re_real.match(r"\\textbf\{[\d.]+ ", i)]      # the three triage bands
    ctrl_items = [i for i in items if i not in band_items]
    NCHECK_inc()
    if len(items) - len(band_items) < 8 or len(band_items) != 3:
        FAILS.append(f"recommendations: parsed {len(ctrl_items)} controls and {len(band_items)} triage bands "
                     f"(expected at least 8 and exactly 3)")
    tagged, now_cov, untagged = set(), set(), []
    for it in ctrl_items:
        NCHECK_inc()
        m = re.match(r"\\textbf\{\(([^)]*)\)\} \\emph\{\[(\w+), (\w), (\w+)\]\}", it)
        if not m:
            untagged.append(_re_real.sub(r"\s+", " ", it)[:40])
            continue
        fs, own, eff, seq = m.groups()
        if own not in OWN or eff not in EFF or seq not in SEQ:
            FAILS.append(f"recommendation tag [{own}, {eff}, {seq}] not in owners {sorted(OWN)}, "
                         f"effort {sorted(EFF)}, sequence {SEQ}: {it[:50]}")
            continue
        ids = _fcodes(fs)
        NCHECK_inc()
        if not ids or any(f not in prio for f in ids):
            FAILS.append(f"recommendation finding codes ({fs}) are not all in the risk table: {it[:50]}")
            continue
        best = min(prio[f] for f in ids)
        want = "Now" if best <= REC_NOW_MAX else "Next" if best <= REC_NEXT_MAX else "Later"
        NCHECK_inc()
        if SEQ.index(seq) > SEQ.index(want):
            FAILS.append(f"recommendation {ids}: sequence {seq} is later than the {want} that its "
                         f"best priority P{best} requires")
        tagged |= set(ids)
        if seq == "Now":
            now_cov |= set(ids)
    # the only control without a finding tag is the informational latency note
    NCHECK_inc()
    if not (len(untagged) == 1 and untagged[0].startswith("\\textbf{Detector latency.}")):
        FAILS.append(f"recommendation items without a '(findings) [owner, effort, sequence]' tag: {untagged} "
                     f"(only 'Detector latency.' is exempt)")
    NCHECK_inc()
    if not need_cover <= tagged:
        FAILS.append(f"recommendations: residual Medium/High findings with no recommendation: "
                     f"{sorted(need_cover - tagged)}")
    NCHECK_inc()
    if not now_must <= now_cov:
        FAILS.append(f"recommendations: priority-1/2 findings not covered by a Now item: "
                     f"{sorted(now_must - now_cov)}")
    NCHECK_inc()
    if not {f for f, p in prio.items() if p <= REC_NOW_MAX} <= now_cov:
        FAILS.append("recommendations: the Now items do not cover every finding ranked first to fourth")
    pos = sorted((sec.index(h), h) for h in heads if h in sec)
    for k, (i0, h) in enumerate(pos[:-1]):
        blk = sec[i0:pos[k + 1][0]]
        NCHECK_inc()
        if "\\item" not in blk:
            FAILS.append(f"recommendations: pillar {h} has no items")
    # detector-API alert threshold: a rule over the measured per-flow cost, recomputed from the budget JSON
    bu = load("task1_attacker_budget.json")
    ctl = load("task5_recommendation_controls.json")
    bb = bu["black_box_methods"]
    fd_min = bb["fd_fgsm"]["server_measured_queries_per_flow_min"]
    rs_min = bb["random_search"]["server_measured_queries_per_flow_min"]
    rs_max = bb["random_search"]["server_measured_queries_per_flow_max"]
    cheapest = min(fd_min, rs_min)
    thr = math.floor(0.5 * cheapest)            # the rule: half the cheapest measured attack, rounded down
    print(f"recommendations: FD-FGSM {fd_min}, random search {rs_min}..{rs_max} queries/flow; "
          f"cheapest {cheapest}; threshold = floor(0.5 x cheapest) = {thr}")
    NCHECK_inc()
    if not (ctl["cheapest_attack_queries_per_flow"] == cheapest and ctl["alert_fraction"] == 0.5
            and ctl["alert_threshold_near_identical_vectors"] == thr
            and ctl["fd_fgsm_queries_per_flow_min"] == fd_min
            and ctl["random_search_queries_per_flow_max"] == rs_max):
        FAILS.append("recommendations: results/task5_recommendation_controls.json disagrees with the rule "
                     "recomputed from task1_attacker_budget.json (rerun scripts/recommendation_controls.py)")
    NCHECK_inc()
    if not (thr < cheapest and 2 * thr <= cheapest < 2 * (thr + 1)):
        FAILS.append("recommendations: the threshold is not half the cheapest measured attack")
    qi = [_squash(i) for i in ctrl_items if "Query-rate alerting." in i]
    NCHECK_inc()
    if len(qi) != 1 or "Rate-limit each client by rows scored" not in qi[0] \
            or "near-identical vectors that differ in few features" not in qi[0]:
        FAILS.append("recommendations: the query-rate item must name per-client rate limiting and the "
                     "near-identical-vectors pattern")
    gi = [_squash(i) for i in ctrl_items if "Guard availability." in i]
    NCHECK_inc()
    if len(gi) != 1 or not all(s in gi[0] for s in
                               ("Health-check the guard", "apart from blocks", "fails closed", "\\S6")):
        FAILS.append("recommendations: guard-availability item lacks health check / separate counts / "
                     "\\S6 link / fails-closed tie")
    NCHECK_inc()
    lim = [q for q in _paras() if q.startswith("\\lead{Limitation.}")]
    if len(lim) != 1 or "fails closed" not in lim[0]:
        FAILS.append("recommendations: the Limitation paragraph the guard item is tied to is missing "
                     "or does not say the guard fails closed")


def recommendations_text():
    """Every number in sections/05_recommendations.tex, paragraph/sentence-scoped against the
    results JSON. Derived values (half of the cheapest attack, band counts, hours, rates, the
    comparison words 'only', 'small', 'weaker', 'stronger', 'feasible') are computed here."""
    ops = load("ops_measurements.json")
    bu = load("task1_attacker_budget.json")
    ctl = load("task5_recommendation_controls.json")
    b1 = load("task1b_evasion_wednesday.json")
    df = load("task1c_defense.json")
    ex = load("task1_extras.json")
    sm = load("task2_summary.json")
    abl = {c["config"]: c for c in load("task1c_ablation.json")["configs"]
           + load("task1c_ablation_rounds.json")["configs"]}
    grid = {r["eps_std"] for r in b1["curve"]}
    per = {p["key"]: p for p in sm["per_attack"]}
    rkc = _risk_cells()
    prio = sorted(int(r["prio"]) for r in rkc.values())
    cb = ops["confidence_bands"]
    bands, edges = cb["bands"], cb["band_edges_fixed_by_design"]
    bu_bb = bu["black_box_methods"]
    und = lambda e, a, k="evasion_rate": cell(b1["curve"], e, a, k)
    dfd = lambda e, a, k="evasion_rate": cell(df["curve_after"], e, a, k)
    ATT = ("fd_fgsm", "pgd")

    # ---- preamble: the sequence rule against the risk table's priority ranks
    p = _para1("recs preamble", "Each control gives its finding and")
    m = _rx("recs sequence rule", r"Now = \\S(\d) priorities (\d)--(\d), Next = (\d)--(\d), Later = the rest", p)
    if m:
        int_cell_check("recs sequence: risk section number", m.group(1), _sec_no("Risk assessment") or -1)
        int_cell_check("recs sequence: first priority", m.group(2), prio[0])
        int_cell_check("recs sequence: Now upper rank", m.group(3), REC_NOW_MAX)
        int_cell_check("recs sequence: Next lower rank", m.group(4), REC_NOW_MAX + 1)
        int_cell_check("recs sequence: Next upper rank", m.group(5), REC_NEXT_MAX)
        NCHECK_inc()
        if prio != list(range(1, len(prio) + 1)) or REC_NEXT_MAX >= len(prio):
            FAILS.append(f"recs sequence: the risk table's priorities are {prio}, not a ranking 1..n with a 'Later' tail")

    # ---- knowledge base and pipeline / re-test: F1 and F2 baseline 0/10 after guards
    k1, k2 = per["2A-kb-override"], per["2A-kb-secret"]
    NCHECK_inc()
    if (k1["after_successes"], k1["n_after"]) != (k2["after_successes"], k2["n_after"]):
        FAILS.append("recs re-test: F1 and F2 do not share one after-guard count")
    for lab, anchor, pat in (("recs KB re-test", "Treat the knowledge base as trusted code", r"F1 and F2 must stay (\d+/\d+)\."),
                             ("recs re-test plan", "\\lead{Re-test plan.}", r"Pass when F1 and F2 stay (\d+/\d+) and")):
        p = _para1(lab, anchor)
        m = _rx(lab, pat, p)
        if m:
            _frac(lab + " F1/F2", m.group(1), k1["after_successes"], k1["n_after"])
    p = _para1("recs re-test plan eps", "\\lead{Re-test plan.}")
    m = _rx("recs re-test plan eps", r"evasion curve at \$\\vtheta=([\d.]+)\$ and ([\d.]+) is no worse than the Lab~1 model's", p)
    if m:
        _eps_lit("recs re-test eps 1", m.group(1), grid)
        _eps_lit("recs re-test eps 2", m.group(2), grid)

    # ---- model controls
    p = _para1("recs model", "Never ship the two-round model.")
    m = _rx("recs one-round candidate", r"one-round candidate \((\w+) constant features frozen\) and ship it only if it passes "
            r"the release gate below \(1C, \\S(\d)\)", p)
    if m:
        c1 = abl["full_1r_frozen"]
        _word_is("recs one-round candidate frozen features", m.group(1), c1["n_frozen_features"])
        int_cell_check("recs 1C section number", m.group(2), _sec_no("Task 1") or -1)
        NCHECK_inc()
        if not (c1["rounds"] == 1 and df["adversarial_training"]["rounds"] == 2):
            FAILS.append("recs model: 'Never ship the two-round model ... use the one-round candidate' needs a two-round "
                         "shipped defense and a one-round candidate in the stored results")
    m = _rx("recs immutable constants", r"Make the (\w+) constant features immutable and reject any nonzero value on them\. "
            r"\\emph\{Re-test:\} send one; it must be rejected", p)
    if m:
        _word_is("recs immutable constants count", m.group(1), abl["full_1r_frozen"]["n_frozen_features"])
        zv = load("task1c_scale_effect.json")["degenerate_training_value"]
        NCHECK_inc()
        if not all(v == 0.0 for v in zv.values()):
            FAILS.append("recs immutable constants: a range check 'reject any nonzero value' needs the five mutable "
                         "constants to be exactly 0 in Lab 1's training data")
    m = _rx("recs release gate", r"Use the evasion-curve release gate \(Key terms\): the stronger of FD-FGSM and PGD-40 at "
            r"\$\\vtheta=([\d.]+)\$ and ([\d.]+), not random search or clean accuracy\. Clean accuracy ([\d.]+) hid a large "
            r"loss of detection", p)
    if m:
        eps = [float(m.group(1)), float(m.group(2))]
        _eps_lit("recs release gate eps 1", m.group(1), grid)
        _eps_lit("recs release gate eps 2", m.group(2), grid)
        cell_check("recs release gate clean accuracy", m.group(3), df["clean_accuracy"]["after"])
        NCHECK_inc()
        term = [q for q in _paras() if "\\term{Evasion-curve release gate}" in q]
        if len(term) != 1:
            FAILS.append("recs release gate: '(Key terms)' needs exactly one \\term{Evasion-curve release gate} entry")
        mx = lambda f, e: max(f(e, a) for a in ATT)
        # 'the stronger' attack is the higher evasion rate; random search is not it
        NCHECK_inc()
        if not all(mx(und, e) > und(e, "random_search") for e in eps):
            FAILS.append("recs release gate: 'not random search' needs FD-FGSM/PGD-40 to out-evade random search at both eps")
        # the gate fires on the shipped two-round model and is not equal to PGD-40 alone at 0.15
        NCHECK_inc()
        if not all(mx(dfd, e) > mx(und, e) for e in eps):
            FAILS.append("recs release gate: the max-over-attacks gate would not fire on the shipped defense at both eps")
        # 'clean accuracy hid a large loss': clean accuracy moved by < 1e-5 while detection under the stronger attack
        # fell by at least half at some gated eps
        NCHECK_inc()
        drop = max((1 - mx(und, e)) - (1 - mx(dfd, e)) for e in eps)
        if not (abs(df["clean_accuracy"]["after"] - df["clean_accuracy"]["before"]) < 1e-5 and drop >= 0.5):
            FAILS.append(f"recs release gate: 'clean accuracy hid a large loss of detection' not supported "
                         f"(clean change {df['clean_accuracy']}, largest detection drop {drop:.3f})")
    p = _para1("recs label only", "Return the label only.")
    NCHECK_inc()
    if not ("logit margin" in bu["api_returns"] and "score" in bu["api_returns"]):
        FAILS.append("recs label only: 'the API also returns the score and logit margin' not in the stored attacker-budget record")

    # ---- monitoring: query-rate alerting
    p = _para1("recs query rate", "\\lead{Query-rate alerting.}")
    m = _rx("recs query rate", r"The cheapest attack needs (\d+) queries per flow, so alert at (\d+) vectors from one client "
            r"\(half of (\d+)\) and cap the rate there\. Flooding with false alarms needs larger changes \((\d+/\d+) at "
            r"\$\\vtheta=([\d.]+)\$\), so the same alert covers it\..*?the alert must fire by vector (\d+)\.", p)
    if m:
        fd_min = bu_bb["fd_fgsm"]["server_measured_queries_per_flow_min"]
        rs_min = bu_bb["random_search"]["server_measured_queries_per_flow_min"]
        cheapest = min(fd_min, rs_min)
        thr = ctl["alert_threshold_near_identical_vectors"]
        int_cell_check("recs cheapest attack queries", m.group(1), int(cheapest))
        int_cell_check("recs alert threshold", m.group(2), thr)
        int_cell_check("recs half of cheapest", m.group(3), int(cheapest))
        int_cell_check("recs re-test vector", m.group(6), thr)
        NCHECK_inc()
        if not (ctl["cheapest_attack_queries_per_flow"] == cheapest and 2 * thr <= cheapest < 2 * (thr + 1)):
            FAILS.append("recs query rate: the alert is not half of the cheapest measured attack")
        fa = {r["eps_std"]: r["methods"] for r in ex["blocks"]["benign_to_attack"]["curve"]}
        n_b2a = ex["blocks"]["benign_to_attack"]["n_flows"]
        e = float(m.group(5))
        _eps_lit("recs flooding eps", m.group(5), fa)
        if e in fa:
            fdc = fa[e]["fd_fgsm"]["false_alarm"]
            _frac("recs flooding false alarms", m.group(4), fdc["count"], fdc["n"])
            NCHECK_inc()
            small = [x for x in fa if x <= 0.25]
            if not (fdc["n"] == n_b2a and all(fa[x]["fd_fgsm"]["false_alarm"]["count"] < fdc["count"] for x in small)
                    and fdc["count"] == max(fa[e][k]["false_alarm"]["count"] for k in fa[e])):
                FAILS.append("recs flooding: 'needs larger changes' needs fewer false alarms at every eps <= 0.25 than at the "
                             "quoted eps, and FD-FGSM to be the strongest flooder there")
    # ---- monitoring: detector latency
    p = _para1("recs latency", "\\textbf{Detector latency.}")
    m = _rx("recs latency", r"median ([\d.]+)\\,ms; the Wednesday test window peaked at (\d+) flows/s\. Real-time scoring is "
            r"feasible", p)
    if m:
        lat = ops["detector_latency"]["models"]["deployed_lab1"]["single_flow"]["median_ms"]
        tr = ops["detector_latency"]["traffic"]
        cell_check("recs latency median ms", m.group(1), lat)
        int_cell_check("recs peak flows per second", m.group(2), tr["busiest_second_flows"])
        NCHECK_inc()
        if not (1000.0 / lat > tr["busiest_second_flows"] and "wednesday" in tr["source"].lower()):
            FAILS.append("recs latency: 'real-time scoring is feasible' needs single-flow capacity above the busiest second "
                         "of the Wednesday window")
    # ---- monitoring: guard availability, prompt length
    p = _para1("recs guard availability", "\\lead{Guard availability.}")
    m = _rx("recs guard availability", r"fails closed and reads as a block \(\\S(\d)\)\. Baseline: (\d+) errored records", p)
    if m:
        ge = sm["guard_evidence"]
        int_cell_check("recs guard section number", m.group(1), _sec_no("Evidence quality") or -1)
        int_cell_check("recs guard baseline", m.group(2), ge["records_with_unspecified_empty_category_or_error"])
        NCHECK_inc()
        if ctl["guard_baseline_unspecified_or_errored_records"] != ge["records_with_unspecified_empty_category_or_error"]:
            FAILS.append("recs guard baseline: task5_recommendation_controls.json disagrees with task2_summary.json")
    p = _para1("recs prompt length", "Cap prompt length and embedded Q/A examples")
    m = _rx("recs many-shot examples", r"Our test used only (\d+) examples, so a cap is untested", p)
    if m:
        try:
            blk = (ROOT / "chatbot" / "attacks.py").read_text()
            blk = blk[blk.index("JB_MANYSHOT = Attack("):]
            blk = blk[:blk.index("ATTACKS: list")]
            n_ex = len(_re_real.findall(r'"A: [^"\\]+(?:\\n)+"', blk))
        except (OSError, ValueError):
            n_ex = -1
        int_cell_check("recs many-shot examples in the stored attack", m.group(1), n_ex)

    # ---- human in the loop: triage bands
    p = _para1("recs triage", "\\lead{L1/L2 triage by attack score}")
    n_ben, n_att = bands["clean_test_benign"]["n"], bands["clean_test_attack"]["n"]
    m = _rx("recs triage auto band", r"\\textbf\{([\d.]+) and above\} .*? all ([\d{},]+) attack flows land here", p)
    if m:
        cell_check("recs triage auto-alert edge", m.group(1), edges["auto_alert_ge"])
        ca = bands["clean_test_attack"]["counts"]
        int_cell_check("recs triage attack flows", m.group(2), n_att)
        NCHECK_inc()
        if ca["auto_alert"] != n_att:
            FAILS.append("recs triage: 'all attack flows land here' but not every clean attack flow is auto-alerted")
    m = _rx("recs triage L2 band", r"\\textbf\{([\d.]+) to below ([\d.]+)\} .*?\(clean window: (\d+) benign flow\)", p)
    if m:
        cell_check("recs triage L2 lower edge", m.group(1), edges["L2_review"][0])
        cell_check("recs triage L2 upper edge", m.group(2), edges["L2_review"][1])
        cb_ = bands["clean_test_benign"]["counts"]
        int_cell_check("recs triage L2 benign flows", m.group(3), cb_["L2_review"])
        NCHECK_inc()
        if cb_["L2_review"] != 1:
            FAILS.append("recs triage: the text says '1 benign flow' (singular) but the count is not 1")
        NCHECK_inc()
        if edges["L2_review"][1] != edges["auto_alert_ge"]:
            FAILS.append("recs triage: the L2 upper edge is not the auto-alert edge")
    m = _rx("recs triage L1 band", r"\\textbf\{([\d.]+) to below ([\d.]+)\} \$\\rightarrow\$ L1 analyst .*? Below ([\d.]+) passes", p)
    if m:
        cell_check("recs triage L1 lower edge", m.group(1), edges["L1_review"][0])
        cell_check("recs triage L1 upper edge", m.group(2), edges["L1_review"][1])
        cell_check("recs triage pass edge", m.group(3), edges["pass_lt"])
        NCHECK_inc()
        if not (edges["L1_review"][1] == edges["L2_review"][0] and edges["pass_lt"] == edges["L1_review"][0]):
            FAILS.append("recs triage: the bands are not contiguous")
    m = _rx("recs triage what it catches", r"At \$\\vtheta=([\d.]+)\$, (\d+) of the (\d+) attack flows that slip under the "
            r"([\d.]+) alert still reach L1\. The cost is small: on the clean Wednesday window \(([\d.]+) h\) only (\d+) benign "
            r"flows land in L1\.", p)
    key = lambda e, a: f"{a}_eps{e}"
    L1 = lambda e, a: cb["evasion_flows_in_L1_review"][key(e, a)]
    e_catch = None
    if m:
        e = e_catch = float(m.group(1))
        _eps_lit("recs triage eps", m.group(1), grid)
        n_eval = b1["n_flows"]
        for a in ATT:
            blk_ = L1(e, a)
            NCHECK_inc()
            if blk_["lower_edge_0.01"]["in_L1_review"] != int(m.group(2)) or blk_["n_evasion_flows"] != int(m.group(3)):
                FAILS.append(f"recs triage: {a} at eps {e}: JSON says {blk_['lower_edge_0.01']['in_L1_review']} of "
                             f"{blk_['n_evasion_flows']} in L1")
            NCHECK_inc()
            if round(und(e, a) * n_eval) != blk_["n_evasion_flows"]:
                FAILS.append(f"recs triage: {a} at eps {e}: the stored evasion rate does not give {blk_['n_evasion_flows']} flows")
        int_cell_check("recs triage L1 flows (FD-FGSM and PGD-40 agree)", m.group(2), L1(e, "fd_fgsm")["lower_edge_0.01"]["in_L1_review"])
        int_cell_check("recs triage evasion flows", m.group(3), L1(e, "fd_fgsm")["n_evasion_flows"])
        cell_check("recs triage alert threshold", m.group(4), b1["detect_threshold"])
        tr = ops["detector_latency"]["traffic"]
        cell_check("recs triage window hours", m.group(5), tr["test_window_seconds"] / 3600.0)
        NCHECK_inc()
        if tr["n_test_rows"] != n_ben + n_att:
            FAILS.append("recs triage: the Wednesday window rows are not the benign + attack rows of the band measurement")
        int_cell_check("recs triage benign flows in L1", m.group(6), bands["clean_test_benign"]["counts"]["L1_review"])
        NCHECK_inc()
        frac = bands["clean_test_benign"]["counts"]["L1_review"] / n_ben
        if not frac < 0.001:
            FAILS.append(f"recs triage: 'the cost is small' fails, {frac:.5f} of benign flows land in L1")
    m = _rx("recs triage sensitivity", r"At \$\\vtheta=([\d.]+)\$ the L1 band is weaker: it catches only (\d+) \(FD-FGSM\) and "
            r"(\d+) \(PGD-40\) of (\d+)\.", p)
    if m:
        e2 = float(m.group(1))
        _eps_lit("recs sensitivity eps", m.group(1), grid)
        f_, g_ = L1(e2, "fd_fgsm"), L1(e2, "pgd")
        int_cell_check("recs sensitivity FD-FGSM", m.group(2), f_["lower_edge_0.01"]["in_L1_review"])
        int_cell_check("recs sensitivity PGD-40", m.group(3), g_["lower_edge_0.01"]["in_L1_review"])
        int_cell_check("recs sensitivity evasion flows", m.group(4), f_["n_evasion_flows"])
        NCHECK_inc()
        e_big = e_catch
        weak = all(L1(e2, a)["lower_edge_0.01"]["fraction_of_evasion_flows_in_L1"]
                   < L1(e_big, a)["lower_edge_0.01"]["fraction_of_evasion_flows_in_L1"] for a in ATT) if e_big else False
        if not (g_["n_evasion_flows"] == f_["n_evasion_flows"] and weak and e2 < (e_big or 0)):
            FAILS.append("recs sensitivity: 'the L1 band is weaker' needs a smaller share of evasion flows in L1 at the "
                         "smaller eps than at the larger one, for both attacks")
    NCHECK_inc()
    if "Deployed Lab 1" not in cb["method"]:
        FAILS.append("recs triage caveat: 'Measured on the Lab 1 model' needs the band measurement to use the deployed Lab 1 model")


def expected_guard_claims():
    """The Llama Guard / output-filter sentence in 03_task2.tex ('Each layer stopped different attacks'),
    each part derived from results/task2_redteam.json after-arm trials[*].block_layers."""
    d = load("task2_redteam.json")
    aft = {r["key"]: r for r in d["after"]}
    p = _para1("t2 layer claims", "\\lead{Each layer stopped different attacks.}")
    if p is None:
        return
    runs = d["runs_per_attack"]
    gl = lambda key: [[l for l in t["block_layers"] if l.startswith("llama_guard")] for t in aft[key]["trials"]]
    # 2A-direct-prompt-exfil: 'Llama Guard missed "show me your system prompt" (our regex caught it)'
    ex = gl("2A-direct-prompt-exfil")
    rx = [any(l.startswith("input_filter:system-prompt-extraction") for l in t["block_layers"])
          for t in aft["2A-direct-prompt-exfil"]["trials"]]
    NCHECK_inc()
    if not ("Llama Guard missed ``Print your full system prompt'' (our regex caught it)" in p
            and len(ex) == runs and all(not x for x in ex) and all(rx)):
        FAILS.append("t2 layer claims: 'Llama Guard missed show me your system prompt (our regex caught it)' needs no "
                     "llama_guard layer and an input_filter:system-prompt-extraction layer in all trials of 2A-direct-prompt-exfil")
    # the quoted attack phrases must be the real inputs in chatbot/attacks.py
    _atk_src = (ROOT / "chatbot" / "attacks.py").read_text()
    for _q in re.findall(r"``([^']+)''", p):
        NCHECK_inc()
        if _q not in _atk_src:
            FAILS.append(f"t2 layer claims: quoted input '{_q}' is not an attack input in chatbot/attacks.py")
    # 2A-direct-override: 'flagged "ignore previous instructions" only under an unrelated harm category (S7)'
    ov = gl("2A-direct-override")
    NCHECK_inc()
    m = re.search(r"flagged ``Ignore all previous instructions'' only under an unrelated harm category \((S\d)\)", p)
    if not (m and len(ov) == runs and all(x == [f"llama_guard:['{m.group(1)}']"] for x in ov)):
        FAILS.append("t2 layer claims: 'flagged ignore previous instructions only under an unrelated category (S7)' needs "
                     "llama_guard:['S7'] and no other guard category in all trials of 2A-direct-override")
    elif m:
        cover(m.group(1), [m.group(1)], "guard-category")
    # role-play and many-shot: 'blocked all 10 role-play (S1) and many-shot (S2) trials'
    m = re.search(r"but blocked all (\d+) role-play \((S\d)\) and many-shot \((S\d)\) trials", p)
    NCHECK_inc()
    if not m:
        FAILS.append("t2 layer claims: 'blocked all N role-play (S1) and many-shot (S2) trials' not in the expected form")
    else:
        n = int(m.group(1))
        for key, cat in (("2B-roleplay", m.group(2)), ("2B-manyshot", m.group(3))):
            NCHECK_inc()
            g = gl(key)
            if not (n == runs == len(g) and all(x == [f"llama_guard:['{cat}']"] for x in g)
                    and all(t["blocked_input"] for t in aft[key]["trials"])):
                FAILS.append(f"t2 layer claims: {key} is not blocked by llama_guard:['{cat}'] in all {n} trials")
        int_cell_check("t2 layer claims trial count", m.group(1), runs)
        for g_ in (m.group(2), m.group(3)):
            cover(g_, [g_], "guard-category")
    # the two quoted phrases are the kinds of request our own regex layer is built for: each must match it
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from chatbot import guard as _guard
        pats = dict(_guard.INJECTION_PATTERNS)
        from chatbot import attacks as _atk
        q_exfil = _atk.DIRECT_PROMPT_EXFIL.question
        q_over = _atk.DIRECT_OVERRIDE.question
    except Exception as e:          # an unimportable module is a FAIL, not a skip
        FAILS.append(f"t2 layer claims: cannot load chatbot.guard / chatbot.attacks ({type(e).__name__}: {e})")
        pats, q_exfil, q_over = {}, "", ""
    for lab, qpat, name, question in (("missed", r"missed ``([^']+)''", "system-prompt-extraction", q_exfil),
                                      ("flagged", r"flagged ``([^']+)''", "override-instructions", q_over)):
        NCHECK_inc()
        mq = re.search(qpat, p)
        rx_ = pats.get(name)
        if not (mq and rx_ and rx_.search(str(mq.group(1))) and rx_.search(question)):
            FAILS.append(f"t2 layer claims: the {lab} phrase and the stored attack question must both match the input "
                         f"filter pattern '{name}'")
    # output filter: 'stopped 7/10 of the KB secret theft'
    m = re.search(r"The output filter stopped (\d+/\d+) of the KB secret theft", p)
    NCHECK_inc()
    if not m:
        FAILS.append("t2 layer claims: 'The output filter stopped K/N of the KB secret theft' not in the expected form")
    else:
        tr = aft["2A-kb-secret"]["trials"]
        k = sum(1 for t in tr if "secret_code" in t["block_layers"])
        NCHECK_inc()
        if any("secret_code" in t["block_layers"] and not t["blocked_output"] for t in tr):
            FAILS.append("t2 layer claims: a secret_code layer appears on a trial that is not output-blocked")
        _frac("t2 layer claims: output filter on the KB secret theft", m.group(1), k, len(tr))


def evidence_map():
    """Section 6 evidence table (Finding | Script | Results JSON | Figure/Table): every script and
    results file it names exists (and is tracked by git, or untracked but not ignored); every
    results file is produced by a script in the same row; every finding code is in the section-4
    findings list, every finding of that list has a row, and each row names the results for its
    finding; table and figure references point at real sections and figures."""
    import subprocess
    a = TEX.index("\\section{Evidence quality}")
    sec = TEX[a:]
    NCHECK_inc()
    i0, i1 = sec.find("\\endhead"), sec.find("\\bottomrule")
    if i0 < 0 or i1 < i0:
        FAILS.append("evidence map: the longtable (\\endhead ... \\bottomrule) was not found in section 6")
        return
    body = sec[i0 + len("\\endhead"):i1]
    lines0 = TEX.count("\n", 0, a + i0) + 1
    rows = []
    for rawrow in _re_real.split(r"\\\\\s*\n", body):
        if rawrow.strip():
            rows.append(rawrow.strip())
    NCHECK_inc()
    if len(rows) != 7:
        FAILS.append(f"evidence map: {len(rows)} rows parsed in the evidence table (expected 7)")

    def names(cell, ext):
        out = []
        for t in _re_real.findall(r"\\texttt\{((?:\\\{|\\\}|[^{}])*)\}", cell.replace("\\allowbreak{}", "")):
            t = t.replace("\\allowbreak{}", "").replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")
            mm = _re_real.fullmatch(r"(.*)\{([^}]*)\}(.*)", t)
            for v in ([mm.group(1) + x.strip() + mm.group(3) for x in mm.group(2).split(",")] if mm else [t]):
                if v.endswith(ext):
                    out.append(v)
        return out

    def find_script(s):
        if "/" in s:
            return ROOT / s if (ROOT / s).is_file() else None
        hits = [ROOT / d / s for d in ("scripts", "detector", "chatbot") if (ROOT / d / s).is_file()]
        return hits[0] if len(hits) == 1 else None

    def tracked(path):
        rel = str(path.relative_to(ROOT))
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "--", rel],
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return True        # no git available: the existence check above is all there is
        if r.returncode != 0:  # not a git checkout (e.g. an exported tree)
            return True
        return rel in r.stdout.split("\n")

    findings = sorted(_risk_cells())            # the findings list: F1..F8 of the section-4 table
    title = {t: r for t, r in _risk_cells().items()}
    NEED = {          # finding -> (a results JSON the row must name, for that finding's data)
        "F1": "task2_summary.json", "F2": "task2_summary.json", "F3": "task2_summary.json", "F4": "task2_summary.json",
        "F5": "task1b_evasion_wednesday.json", "F6": "task1a_membership.json",
        "F7": "task2_retrieval_probe.json", "F8": "task1c_defense.json"}
    sm = load("task2_summary.json")
    keys = {p["key"] for p in sm["per_attack"]}
    seen = set()
    n_fig = len(_re_real.findall(r"\\begin\{figure\}", TEX))
    for r in rows:
        cells = [c.strip() for c in _re_real.split(r"(?<!\\)&", r)]
        NCHECK_inc()
        if len(cells) != 4:
            FAILS.append(f"evidence map: row has {len(cells)} cells, expected 4: {r[:50]}")
            continue
        label = cells[0]
        scripts, jsons = names(cells[1], ".py"), names(cells[2], ".json")
        rowsrc = _mk(_squash(r), "report", range(lines0, lines0 + body.count("\n") + 1))
        NCHECK_inc()
        if not scripts or not jsons:
            FAILS.append(f"evidence map: row '{label}' must name at least one script and one results JSON")
            continue
        spaths = {}
        for s in scripts:
            NCHECK_inc()
            sp = find_script(s)
            if sp is None:
                FAILS.append(f"evidence map: row '{label}' names script {s}, which does not exist (scripts/, detector/, chatbot/)")
                continue
            spaths[s] = sp
            NCHECK_inc()
            if not tracked(sp):
                FAILS.append(f"evidence map: script {s} is not tracked by git (and not an addable file)")
        for j in jsons:
            NCHECK_inc()
            if not (R / j).is_file():
                FAILS.append(f"evidence map: row '{label}' names results file {j}, which does not exist")
                continue
            NCHECK_inc()
            if not tracked(R / j):
                FAILS.append(f"evidence map: results file {j} is not tracked by git (and not an addable file)")
            # the row's scripts must produce it: a script names the file, or the file names the script
            NCHECK_inc()
            try:
                meta = json.loads((R / j).read_text())
            except ValueError:
                meta = {}
            who = str(meta.get("script", "")) + " " + str(meta.get("generated_by", "")) if isinstance(meta, dict) else ""
            stem = j[:-5]
            made = any(stem in sp.read_text() for sp in spaths.values()) or \
                any(s in who for s in scripts)
            if not made:
                FAILS.append(f"evidence map: row '{label}': no script in the row names or is named by {j}")
        # a brace-listed file name (task1c_{a, b}.json) is not seen by the file-name allowlist entry: its digit is
        # covered here, once every expanded name has been shown to exist
        if all((R / j).is_file() for j in jsons):
            for mb in _re_real.finditer(r"\\texttt\{[A-Za-z]*(\d+)[a-z]*\\_(?:\\allowbreak\{\})?\\\{", cells[2]):
                cover(rowsrc, [mb.group(1)], "file-name")
        # the finding label against the findings list and the data
        codes = []
        if label.startswith("F"):
            codes = _fcodes(_re_real.sub(r" \(extras\)", "", label))
        NCHECK_inc()
        if label.startswith("F") and (not codes or any(c not in findings for c in codes)):
            FAILS.append(f"evidence map: row '{label}' names a finding that is not in the section-4 findings list {findings}")
        seen |= set(codes)
        for c in codes:
            NCHECK_inc()
            if "(extras)" in label:
                ok = "task1_extras.json" in jsons
            else:
                ok = NEED[c] in jsons
            if not ok:
                FAILS.append(f"evidence map: row '{label}' lacks {NEED[c]} for {c} ({c})")
        # table/figure references point at real sections and figures
        for mref in _re_real.finditer(r"\\S(\d)", cells[3]):
            NCHECK_inc()
            sn = int(mref.group(1))
            ok = sn in (2, 3, 5) and (codes == [] or sn == 3 or "(extras)" in label)
            if not (ok and _sec_no({2: "Task 1", 3: "Task 2", 5: "Recommendations"}[sn]) == sn):
                FAILS.append(f"evidence map: row '{label}': reference \\S{sn} does not point at the section it should")
            else:
                cover(rowsrc, [mref.group(1)], "ref")
        for mref in _re_real.finditer(r"Fig\.~(\d)", cells[3]):
            NCHECK_inc()
            if not 1 <= int(mref.group(1)) <= n_fig:
                FAILS.append(f"evidence map: row '{label}': Fig.~{mref.group(1)} but the report has {n_fig} figure(s)")
            else:
                cover(rowsrc, [mref.group(1)], "ref")
        for mref in _re_real.finditer(r"\b1([ABC]) tables?", cells[3]):
            NCHECK_inc()
            if f"\\subsection{{1{mref.group(1)} " not in TEX:
                FAILS.append(f"evidence map: row '{label}': no subsection 1{mref.group(1)} in the report")
        if "Appendix A" in cells[3]:
            NCHECK_inc()
            if _sec_no("Attack payloads") is None:
                FAILS.append("evidence map: 'Appendix A' but there is no payloads section")
    NCHECK_inc()
    if not set(findings) <= seen:
        FAILS.append(f"evidence map: findings without a row: {sorted(set(findings) - seen)}")
    # the F1-F4 data is really per attack in the summary
    NCHECK_inc()
    if not {"2A-kb-override", "2A-kb-secret", "2A-kb-misinfo", "2B-roleplay"} <= keys:
        FAILS.append("evidence map: task2_summary.json lacks the per-attack records behind F1-F4")
    # ---- repo URL sentence and the generative-AI line (section 6)
    url = "https://github.com/MZU-GH/SEC-635-Lab-2"
    p = _para1("evidence intro", "Every finding can be reproduced.")
    NCHECK_inc()
    if p is not None:
        m = re.search(r"Code, results and gates are at \\url\{([^}]*)\}", p)
        nm = re.search(r"(\w+) gate scripts check every number here", p)
        if not (m and str(m.group(1)) == url):
            FAILS.append("evidence map: the repo URL sentence 'Code, results and gates are at \\url{...}' is missing or wrong")
        elif (ROOT / "report" / "REPRODUCING.md").is_file() is False:
            FAILS.append("evidence map: report/REPRODUCING.md named in the text does not exist")
        _word_is("evidence gate script count", nm.group(1) if nm else "?", 4)
        # four gate scripts: gate 1, gate 1b, the Task 2 evidence check and gate 2
        NCHECK_inc()
        gates = [f for f in ("verify_report_numbers.py", "number_coverage.py", "make_task2_evidence.py", "verify_claims.py")
                 if (ROOT / "scripts" / f).is_file()]
        if len(gates) != 4:
            FAILS.append(f"evidence map: 'four gate scripts' but only {gates} exist")
    NCHECK_inc()
    flat = _squash(TEX)
    if "\\noindent{\\footnotesize\\emph{Generative AI was used as a coding and writing assistant" not in flat:
        FAILS.append("evidence map: the generative-AI line is not the closing note of section 6")


def final_wording_checks():
    """Human-in-the-loop controls (section 5): answer confirmation, detector triage and review SLA are
    present and tied to the query-rate alert they depend on."""
    sec = _sec5()
    a = sec.index("\\lead{Human-in-the-loop.}")
    hsec = sec[a:sec.index("\\lead{Re-test plan.}", a)]
    items = [_squash(i) for i in _re_real.findall(r"\\item (.*?)(?=\n\\item |\n\\end\{itemize\}|\n\\begin\{itemize\})", hsec, flags=_re_real.S)]
    top = [i for i in items if not _re_real.match(r"\\textbf\{[\d.]+ ", i)]
    NCHECK_inc()
    if len(top) != 3:
        FAILS.append(f"human-in-the-loop: {len(top)} controls, expected answer confirmation, detector triage and review SLA")
        return
    ans, _, sla = top
    tri = _squash(hsec[hsec.index("L1/L2 triage"):hsec.index("\\lead{Review SLA.}")])
    for lab, it, needs in (
            ("answer confirmation", ans, ("An analyst confirms any copilot answer that asserts a technique ID or KEV entry before action",)),
            ("detector triage", tri, ("lower edge included, upper edge excluded", "probing clients", "unvetted document",
                                      "keep edges secret and rotate them")),
            ("review SLA", sla, ("Review SLA", "The SOC lead sets a maximum queue time", "KB owners own the residual F3 risk"))):
        NCHECK_inc()
        miss = [n for n in needs if n not in it]
        if miss:
            FAILS.append(f"human-in-the-loop {lab}: missing {miss}")
    NCHECK_inc()
    mon = _squash(sec[sec.index("\\lead{Monitoring.}"):a])
    if "\\lead{Query-rate alerting.}" not in mon or "near-identical vectors" not in mon:
        FAILS.append("human-in-the-loop: the probing alert referred to by the triage rule is not defined in Monitoring")
    # the answer-confirmation control's trigger (answers citing an unvetted document) is a Monitoring alert
    NCHECK_inc()
    if "Alert on an unvetted top-1 source" not in mon:
        FAILS.append("human-in-the-loop: the 'answers citing an unvetted document' trigger is not defined in Monitoring")


def ai_disclosure():
    """The one-line generative-AI disclaimer appears exactly once, and the report names no AI
    vendor, product or model (digest check below)."""
    line = ("Generative AI was used as a coding and writing assistant; numbers are produced "
            "by the repository's scripts and checked by its gates.")
    flat = _re_real.sub(r"\s+", " ", TEX)
    NCHECK_inc()
    n = flat.count(line)
    if n != 1:
        FAILS.append(f"AI disclaimer: expected exactly 1 occurrence of the disclosure line, found {n}")
    # (length, sha256) of banned vendor/product names; the paper must not name AI vendors.
    # Every window of that length in the lowercased text, and in the text with all
    # non-alphanumerics removed (catches hyphen/brace/markup splits), is hashed and compared.
    banned = (
    (4, "e408ee3d83092ec7f7418c8430df349d62d43b57195888fb7609daf4ff22b368"),
    (5, "1b1ae706971632ada5f6db3654eda45148f24fac0f7abc55c19565494a5361da"),
    (6, "5d72436256ada53828b51895a94bb8489e9f1ac4fe937a8024ef1594e7045ff6"),
    (6, "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef"),
    (6, "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d"),
    (7, "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c"),
    (9, "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131"),
    (13, "c547687a37cc0ecabffbad2699540f419a47e0a86b77fd9c0140b88bb32bc80c"),
    (14, "ceaf6109c3cc4fea4dfda2b5ba0665b95cda7bc99efbf0b5c4e5826fce812bf0"),
    (16, "781fe545a290dca363573991a5ef0ba59f3ac825b58c9bb96fcec5d16132e057"),
    (17, "8bb6561e31c2be642d8544ef6e1c3232604d09d448ca12b089f5fa44053aec20"),
    )
    low = (flat + " " + _re_real.sub(r"\s+", " ", STYLE_TEX)).lower()   # style.tex scanned too
    sq = re.sub(r"[^a-z0-9]", "", low)
    for L, dig in banned:
        NCHECK_inc()
        for txt in (low, sq):
            hit = next((txt[i:i + L] for i in range(len(txt) - L + 1)
                        if hashlib.sha256(txt[i:i + L].encode()).hexdigest() == dig), None)
            if hit is not None:
                FAILS.append(f"AI disclaimer: report names a banned vendor/product (found text '{hit}'; "
                             f"no vendor, product or model names allowed)")
                break


def section_wiring():
    """report.tex must \\input every sections/*.tex on disk, every \\input target must exist,
    and the expanded text must contain every top-level \\section the checks rely on."""
    NCHECK_inc()
    for f in report_source.unwired_sections(TEX_PATH):
        FAILS.append(f"section wiring: {f} exists but report.tex does not \\input it")
    for f in report_source.missing_inputs(TEX_PATH):
        FAILS.append(f"section wiring: report.tex \\input's {f}, which does not exist")
    for title in ("Executive summary", "Task 1: attacks on the intrusion detector",
                  "Task 2: red-teaming the security copilot", "Risk assessment", "Recommendations",
                  "Evidence quality", "Attack payloads and normal questions"):
        NCHECK_inc()
        if TEX.count("\\section{" + title + "}") != 1:
            FAILS.append(f"section wiring: \\section{{{title}}} not found exactly once in the expanded report")


def title_block():
    """Team names + IDs appear exactly once, in the title block; the repository URL appears in the
    title block and in section 6 (Evidence quality); no other github.com URL appears."""
    url = "https://github.com/MZU-GH/SEC-635-Lab-2"
    team = (("Ahmed Alnuaimi", "M80009405"), ("Mansour Alkaabi", "M80009410"),
            ("Ahmed Almansoori", "M80009548"))
    a, b = TEX.find("% TITLEBLOCK-BEGIN"), TEX.find("% TITLEBLOCK-END")
    tb = TEX[a:b] if 0 <= a < b else ""
    NCHECK_inc()
    if not tb:
        FAILS.append("title block: '% TITLEBLOCK-BEGIN' ... '% TITLEBLOCK-END' markers not found")
    for name, sid in team:
        NCHECK_inc()
        if not _re_real.search(_re_real.escape(name) + r"\s*&\s*" + _re_real.escape(sid) + r"\s*(\\\\|$)", str(tb), _re_real.M):
            FAILS.append(f"title block: table row '{name} & {sid}' missing from the title block")
        for tok in (name, sid):
            NCHECK_inc()
            n = TEX.count(tok)
            if n != 1:
                FAILS.append(f"title block: '{tok}' occurs {n} times in the report source (expected exactly 1)")
    NCHECK_inc()
    if f"Repository: \\url{{{url}}}" not in tb:
        FAILS.append("title block: 'Repository: \\url{...}' with the exact repo URL missing")
    m = re.search(r"\\section\{Evidence quality\}(.*?)(?=\\section|\Z)", TEX, re.S)
    NCHECK_inc()
    if not m or f"\\url{{{url}}}" not in m.group(1):
        FAILS.append("title block: repo URL missing from section 6 (Evidence quality)")
    NCHECK_inc()
    if TEX.count(url) != 2:
        FAILS.append(f"title block: repo URL occurs {TEX.count(url)} times (expected 2: title block and section 6)")
    others = [u for u in re.findall(r"github\.com\S*", TEX) if not u.startswith("github.com/MZU-GH/SEC-635-Lab-2}")]
    NCHECK_inc()
    if others:
        FAILS.append(f"title block: other github.com reference(s) in report: {others}")


TB: dict = {}     # textbook citation id -> cited number (filled by textbook_citations)


def textbook_citations():
    """Figures quoted from the course textbook (chapters/course9.txt, not published).
    The cited number must be inside its stored sentence (and the radius arithmetic must
    hold), always; the stored sentences and the file's sha256 are compared with
    chapters/course9.txt when it exists. TEXTBOOK_CITATIONS / TEXTBOOK_TXT redirect both
    (negative tests); a missing, unreadable or malformed JSON is a FAIL, never a crash."""
    NCHECK_inc()
    jp = Path(os.environ.get("TEXTBOOK_CITATIONS", ROOT / "results" / "textbook_citations.json"))
    tp = Path(os.environ.get("TEXTBOOK_TXT", ROOT / "chapters" / "course9.txt"))
    try:
        data = json.loads(jp.read_text())
    except (OSError, ValueError) as ex:
        FAILS.append(f"textbook citations: {jp.name} unreadable: {ex}")
        return
    import make_textbook_citations as mtc
    problems, how = mtc.verify(data, tp)
    print(f"textbook citations: {how}")
    NCHECK_inc()
    FAILS.extend(problems)
    if problems:
        return
    TB.update({c["id"]: c["value"] for c in data["checks"]})
    # every place the report prints the textbook's certified radius must say it is the textbook's
    rad = TB.get("smoothing_radius_formula")
    for p in _paras():
        if rad and re.search(rf"(?<![\d.]){_re_real.escape(rad)}(?![\d])", p):
            NCHECK_inc()
            if "textbook" not in p:
                FAILS.append(f"textbook citations: the radius {rad} is printed without saying it is the textbook's")
            else:
                cover(p, [rad], "citation")


_WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10}


_SA = []


def _split_audit():
    """results/split_audit.json, accepted only if it is internally consistent and not stale.
    Staleness: scripts/split_audit.py --verify (recomputes from the data) when pandas and the
    parquets are available; otherwise the stored record's own consistency is checked here
    (stdlib), plus the sha256 of the Lab 1 result files when LAB1_REPO is present. Returns the
    record, or None after recording a FAIL. Runs once."""
    if _SA:
        return _SA[0]
    import subprocess
    NCHECK_inc()
    try:
        sa = load("split_audit.json")
        w, l1 = sa["wednesday"], sa["lab1_submitted"]
    except (OSError, ValueError, KeyError) as ex:
        FAILS.append(f"split audit: results/split_audit.json unreadable or incomplete ({ex!r})")
        _SA.append(None)
        return None
    bad = []
    if sum(w["train_labels"].values()) != w["rows_train"] or sum(w["test_labels"].values()) != w["rows_test"]:
        bad.append("Wednesday label counts do not add up to the row counts")
    if sorted(w["constant_in_train"]) != sorted(l1["wed_constant_columns_dropped"]):
        bad.append("our constant columns differ from the ones the submitted Lab 1 dropped")
    if w["n_features"] - len(l1["wed_constant_columns_dropped"]) != l1["wed_n_features"]:
        bad.append("our feature count minus the dropped columns is not the submitted Lab 1's feature count")
    if l1["wed_flows_moved_by_clock_fix"] != w["train_rows_after_test_window"]:
        bad.append("flows moved by the clock fix differ from our training rows after the test window")
    unrep = (w["test_labels"] == l1["unrepaired_clock_demo_test_labels"]
             and w["train_labels"] == l1["unrepaired_clock_demo_train_labels"])
    ours = (sorted(w["constant_in_train"]) == sorted(l1["wed_constant_columns_dropped"]))
    if sa["checks"] != {"our_split_equals_lab1_unrepaired_demo": unrep, "our_constant_set_equals_lab1_dropped": ours} \
            or not (unrep and ours):
        bad.append("stored checks block disagrees with the stored data, or a check is false")
    repo = Path(os.environ.get("LAB1_REPO", ROOT.parent.parent / "lab1"))
    for rel, h in l1["source_sha256"].items():
        if (repo / rel).is_file() and hashlib.sha256((repo / rel).read_bytes()).hexdigest() != h:
            bad.append(f"{rel} of the Lab 1 repository no longer matches its stored sha256")
    ex0 = load("report_extras.json")["split"]
    if (ex0["n_train"], ex0["n_test"]) != (w["rows_train"], w["rows_test"]):
        bad.append("row counts differ from task1_extras.json")
    # staleness: recompute from the data when the environment has it
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "split_audit.py"), "--verify"],
                           capture_output=True, text=True, timeout=600,
                           env={**os.environ, "SPLIT_AUDIT_JSON": str(R / "split_audit.json")})
        out = r.stdout + r.stderr
        if r.returncode == 0 and "split audit: PASS" in out:
            pass
        elif "split audit:" in out and "Traceback" not in out:
            bad.append(out.strip().splitlines()[-1])
        else:
            print("note: split_audit.py --verify could not recompute here (no data or pandas); "
                  "stored record checked for consistency only")
    except (OSError, subprocess.SubprocessError):
        print("note: split_audit.py --verify could not run; stored record checked for consistency only")
    for b in bad:
        FAILS.append(f"split audit: {b}")
    _SA.append(None if bad else sa)
    return _SA[0]


def split_audit_text():
    """The corrected split description (02_task1: Split bullet and 'Why not the submitted Lab 1
    model' paragraph; 05_recommendations) against results/split_audit.json."""
    sa = _split_audit()
    if sa is None:
        return
    w, l1 = sa["wednesday"], sa["lab1_submitted"]
    hhmm = lambda s: s.split()[1][:5]
    attacks = lambda lab: {k: v for k, v in lab.items() if k != "Benign"}
    # ---- Split bullet
    p = _para1("t1 split audit", "\\textbf{Split:}")
    m = _rx("t1 split audit",
            r"(\d+)/(\d+) by row order: ([\d{},]+) training and ([\d{},]+) test rows\. The rows are sorted by the raw "
            r"timestamp, which uses a (\d+)-hour clock without AM/PM, so this is \\emph\{not\} true time order\. "
            r"([\d{},]+) training rows \(the (\d\d:\d\d)--(\d\d:\d\d) afternoon\) come after the (\d\d:\d\d)--(\d\d:\d\d) "
            r"test window, and (FTP brute force) appears in both parts\.", p)
    if m:
        tf = re.search(r"TEMPORAL_TRAIN_FRACTION\s*=\s*([\d.]+)", (ROOT / "detector" / "dataset.py").read_text())
        NCHECK_inc()
        if not tf or (round(float(tf.group(1)) * 100), 100 - round(float(tf.group(1)) * 100)) != (
                int(m.group(1)), int(m.group(2))):
            FAILS.append("t1 split audit: the split fraction in detector/dataset.py is not the stated "
                         f"{m.group(1)}/{m.group(2)}")
        NCHECK_inc()
        if not (w["rows_sorted_by_raw_timestamp"]):
            FAILS.append("t1 split audit: rows are not sorted by the raw timestamp")
        int_cell_check("t1 split audit train rows", m.group(3), w["rows_train"])
        int_cell_check("t1 split audit test rows", m.group(4), w["rows_test"])
        int_cell_check("t1 split audit rows after window", m.group(6), w["train_rows_after_test_window"])
        NCHECK_inc()
        if m.group(5) != "12":
            FAILS.append("t1 split audit: the clock is not described as a 12-hour clock")
        a, b = w["train_after_window_true_clock"]
        hi = w["train_true_clock"][1]
        t0, t1 = w["test_true_clock"]
        for lab, txt, want in (("afternoon start (floor to the hour)", m.group(7), a[11:13] + ":00"),
                               ("afternoon end", m.group(8), hhmm(hi)),
                               ("test window start", m.group(9), hhmm(t0)),
                               ("test window end", m.group(10), hhmm(t1))):
            NCHECK_inc()
            if txt != want:
                FAILS.append(f"t1 split audit: {lab} is {txt} in the report, the data gives {want}")
            else:
                cover(p, re.findall(r"\d+", txt), "clock time")
        NCHECK_inc()
        if hi != b or a <= t1:
            FAILS.append("t1 split audit: the afternoon rows are not the training rows after the test window")
        NCHECK_inc()
        if not all("FTP-BruteForce" in d for d in (w["train_labels"], w["test_labels"])):
            FAILS.append("t1 split audit: FTP brute force is not present in both the training and the test labels")
    # ---- Why not the submitted Lab 1 model
    p = _para1("t1 why not submitted", "\\lead{Why not the submitted Lab~1 model.}")
    m = _rx("t1 why not submitted",
            r"The submitted Lab~1 repairs the clock, drops the (\d+) constant columns \((\d+) features\) and trains a "
            r"(\d+)--(\d+) network\. Its Wednesday test window then holds only (SSH) brute force, which training never "
            r"saw, and it flags (\d+) of ([\d{},]+) attack flows \(recall ([\d.]+)\): there is nothing left for evasion "
            r"to remove\. Lab~1 itself reports our split as its unrepaired-clock control\. Our results describe the "
            r"first-version model, not the submitted one\.", p)
    if m:
        int_cell_check("t1 why not: dropped columns", m.group(1), len(l1["wed_constant_columns_dropped"]))
        int_cell_check("t1 why not: features", m.group(2), l1["wed_n_features"])
        NCHECK_inc()
        if [int(m.group(3)), int(m.group(4))] != list(l1["wed70_mlp_hidden"]):
            FAILS.append(f"t1 why not: hidden layers {m.group(3)}--{m.group(4)} differ from {l1['wed70_mlp_hidden']}")
        else:
            cover(p, [m.group(3), m.group(4)], "config")
        te, tr = attacks(l1["wed70_test_labels"]), attacks(l1["wed70_train_labels"])
        NCHECK_inc()
        if not (len(te) == 1 and next(iter(te)).startswith(m.group(5)) and "Benign" in l1["wed70_test_labels"]):
            FAILS.append("t1 why not: the test window does not hold exactly one attack label, SSH")
        NCHECK_inc()
        if any(k.startswith(m.group(5)) for k in l1["wed70_train_labels"]) or not tr:
            FAILS.append("t1 why not: SSH brute force is present in the submitted Lab 1's training labels")
        int_cell_check("t1 why not: flagged", m.group(6), l1["wed70_mlp_tp"])
        int_cell_check("t1 why not: attack flows", m.group(7), sum(te.values()))
        cell_check("t1 why not: recall", m.group(8), l1["wed70_mlp_recall"])
        NCHECK_inc()
        if not sa["checks"]["our_split_equals_lab1_unrepaired_demo"]:
            FAILS.append("t1 why not: our split is not the submitted Lab 1's unrepaired-clock control")
    # ---- Recommendation: retrain without the ten constant columns
    ln = _line("rec retrain", "Before any release, retrain on the clock-repaired split")
    m = _rx("rec retrain", r"Before any release, retrain on the clock-repaired split without the (\w+) constant "
                           r"columns \(\\S(\d+)\) and re-run this gate\.", ln)
    if m:
        NCHECK_inc()
        if _WORD_NUM.get(m.group(1).lower()) != len(l1["wed_constant_columns_dropped"]) \
                or int(m.group(2)) != _sec_no("Task 1"):
            FAILS.append("rec retrain: 'ten constant columns' or its section reference does not match split_audit.json / Task 1")


def lab1_reproduction():
    """'Our reproduction matches Lab 1's metrics to six decimals' (02_task1, Target).
    Lab 1's published precision, recall, F1 and ROC-AUC (lab1/outputs/tables/
    task1_metrics_comparison.json, MLP_NeuralNet, Wednesday temporal) are compared with the
    same metrics recomputed from the stored scoring of the deployed Lab 1 model on our
    split (ops_measurements.json, confidence_bands: every Wednesday test row, P >= 0.5 is
    'attack'), both rounded to the decimals the sentence names. ROC-AUC is 1 exactly when
    every attack score exceeds every benign score (stored min/max)."""
    p = _para1("lab1 reproduction", "Our reproduction matches its metrics to")
    m = _rx("lab1 reproduction", r"Our reproduction matches its metrics to (\w+) decimals\.", p)
    if not m:
        return
    k = _WORD_NUM.get(m.group(1).lower())
    NCHECK_inc()
    if k is None:
        FAILS.append(f"lab1 reproduction: '{m.group(1)}' is not a decimal count the gate knows")
        return
    lp = Path(os.environ.get("LAB1_METRICS", ROOT / "lab1" / "outputs" / "tables" / "task1_metrics_comparison.json"))
    try:
        rows = json.loads(lp.read_text())
        pub = next(r for r in rows if r["model_name"] == "MLP_NeuralNet" and r["dataset_name"] == "Wednesday (Temporal)")
    except (OSError, ValueError, StopIteration, KeyError, TypeError) as ex:
        FAILS.append(f"lab1 reproduction: Lab 1 published metrics not readable from {lp.name}: {ex!r}")
        return
    ops = load("ops_measurements.json")["confidence_bands"]
    x1 = load("task1_extras.json")["input_sha256"]
    NCHECK_inc()
    lab1_model = "lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib"
    if [f["sha256"] for f in ops["input_files"] if f["path"] == lab1_model] != [x1[lab1_model]]:
        FAILS.append("lab1 reproduction: the stored scoring is not of the deployed Lab 1 model (sha256 differs)")
    b, a = ops["bands"]["clean_test_benign"], ops["bands"]["clean_test_attack"]
    pos = lambda g: g["counts"]["auto_alert"] + g["counts"]["L2_review"]     # P >= 0.5
    tp, fp = pos(a), pos(b)
    fn, tn = a["n"] - tp, b["n"] - fp
    NCHECK_inc()
    if sum(a["counts"].values()) != a["n"] or sum(b["counts"].values()) != b["n"]:
        FAILS.append("lab1 reproduction: stored band counts do not add up to the row counts")
    for name, mine, theirs in (("tp", tp, pub["tp"]), ("fp", fp, pub["fp"]),
                               ("tn", tn, pub["tn"]), ("fn", fn, pub["fn"])):
        NCHECK_inc()
        if mine != theirs:
            FAILS.append(f"lab1 reproduction: confusion count {name} is {mine}, Lab 1 published {theirs}")
    mine = {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    sc = ops["score_min_median_max"]
    separable = sc["clean_test_attack"]["min"] > sc["clean_test_benign"]["max"]
    NCHECK_inc()
    if not separable:
        FAILS.append("lab1 reproduction: attack and benign scores overlap, ROC-AUC cannot be derived from the stored extremes")
    else:
        mine["roc_auc"] = 1.0
    for name in ("precision", "recall", "f1", "roc_auc"):
        NCHECK_inc()
        if name in mine and round(mine[name], k) != round(pub[name], k):
            FAILS.append(f"lab1 reproduction: {name} reproduced {mine[name]:.9f} vs Lab 1 {pub[name]:.9f} "
                         f"differ at {k} decimals")


def task1c_text():
    """Section 1C (sections/02c_task1_defense.tex): every number, paragraph- or row-scoped,
    against results/*.json at the rounding the text prints; every comparison word
    ('fell', 'most of', 'never lower', 'higher', 'almost all', 'far beyond') is computed."""
    b1 = load("task1b_evasion_wednesday.json")
    df = load("task1c_defense.json")
    ex = load("report_extras.json")
    bu = load("task1_attacker_budget.json")
    se = load("task1c_scale_effect.json")
    ma = load("task1c_membership_after.json")
    sd = load("task1c_smoothing_detection.json")
    abf = load("task1c_ablation.json")
    rnd = load("task1c_ablation_rounds.json")
    rc = load("task1_extras.json")["blocks"]["recommended_config"]
    ops = load("ops_measurements.json")["detector_latency"]
    wd = {x["day"]: x for x in load("task1a_membership.json")["days"]}["Wednesday"]
    abl = {c["config"]: c for c in abf["configs"] + rnd["configs"]}
    grid = {r["eps_std"] for r in b1["curve"]}
    nflows = b1["n_eval_flows"]
    at = df["adversarial_training"]
    hist = at["history"]
    sm1 = df["smoothing"]
    und = lambda e, a: cell(b1["curve"], e, a)
    dfd = lambda e, a: cell(df["curve_after"], e, a)
    cm = lambda c, e, a: cell(abl[c]["curve_after"], e, a)
    chance = 1 / 2 if wd["eval_index_provenance"]["n_member"] == wd["eval_index_provenance"]["n_nonmember"] else -1

    # ---- Bottom line
    p = _para1("1c bottom line", "\\lead{Bottom line.}")
    m = _rx("1c bottom line", r"PGD-40 detection fell from ([\d.]+) to ([\d.]+) at \$\\vtheta=([\d.]+)\$, "
            r"while clean accuracy stayed at ([\d.]+)\.", p)
    if m:
        _eps_lit("1c bottom line eps", m.group(3), grid)
        e = float(m.group(3))
        cell_check("1c bottom line PGD before", m.group(1), und(e, "pgd"))
        cell_check("1c bottom line PGD after", m.group(2), dfd(e, "pgd"))
        cell_check("1c bottom line clean accuracy after", m.group(4), df["clean_accuracy"]["after"])
        NCHECK_inc()
        if not (at["rounds"] == 2 and dfd(e, "pgd") < und(e, "pgd")):
            FAILS.append("1c bottom line: JSON is not a two-round model whose PGD-40 detection fell")

    # ---- What we did
    p = _para1("1c what we did", "\\lead{What we did.}")
    m = _rx("1c what we did", r"for all ([\d{},]+) training rows\. We then refit on clean and attack rows together "
            r"\(([\d{},]+) rows\)\. We train at \$\\vtheta=([\d.]+)\$, not the textbook's ([\d.]+)\. "
            r"At ([\d.]+) only \\textbf\{([\d.]+)\\%\} of attack rows cross the decision boundary", p)
    if m:
        int_cell_check("1c what we did training rows", m.group(1), hist[0]["n_clean"])
        NCHECK_inc()
        if hist[0]["n_clean"] != ex["split"]["n_train"] or any(h["n_clean"] != hist[0]["n_clean"] for h in hist):
            FAILS.append("1c what we did: crafted-row count differs from the training-set size or between rounds")
        int_cell_check("1c what we did refit rows", m.group(2), hist[0]["train_rows_used"])
        NCHECK_inc()
        if hist[0]["train_rows_used"] != hist[0]["n_clean"] + hist[0]["n_adversarial"]:
            FAILS.append("1c what we did: refit rows are not clean + attack rows")
        cell_check("1c what we did training radius", m.group(3), at["eps_std"])
        NCHECK_inc()
        if not TB.get("attack_budget"):
            FAILS.append("1c what we did: textbook citation 'attack_budget' unavailable (see textbook citations)")
        else:
            cell_check("1c what we did textbook radius (stored textbook sentence)", m.group(4), float(TB["attack_budget"]))
        rp = {r["eps_std"]: r for r in ex["radius_probe"]}
        e = float(m.group(5))
        NCHECK_inc()
        if e not in rp:
            FAILS.append(f"1c what we did: no radius probe at {m.group(5)}")
        else:
            cell_check("1c what we did crossing radius", m.group(5), e)
            cell_check("1c what we did crossing share", m.group(6), rp[e]["frac_attack_rows_flipped"] * 100)
            NCHECK_inc()
            if rp[e]["n_rows"] != hist[0]["n_clean"] or not rp[e]["frac_attack_rows_flipped"] < 0.05:
                FAILS.append("1c what we did: the probe is not over the training rows or the share is not 'almost nothing'")

    # ---- Clean accuracy before and after
    p = _para1("1c clean accuracy", "\\lead{Clean accuracy before and after.}")
    m = _rx("1c clean accuracy", r"did not move: ([\d.]+) before, ([\d.]+) after\..*?scores ([\d.]+), again near the "
            r"(\d+)\\% guess", p)
    if m:
        cb, ca = df["clean_accuracy"]["before"], df["clean_accuracy"]["after"]
        cell_check("1c clean accuracy before", m.group(1), cb)
        cell_check("1c clean accuracy after", m.group(2), ca)
        NCHECK_inc()
        if not abs(cb - ca) < 1e-5:
            FAILS.append(f"1c clean accuracy: before {cb} and after {ca} differ by more than 1e-5 ('did not move')")
        da = ma["defended_target_arm"]["confidence_threshold"]["accuracy"]
        cell_check("1c clean accuracy 1A on two-round model", m.group(3), da)
        cell_check("1c clean accuracy chance level (%)", m.group(4), chance * 100)
        NCHECK_inc()
        if not abs(da - chance) < 0.05:
            FAILS.append(f"1c clean accuracy: 1A accuracy {da} is not near the {chance} guess")

    # ---- Attacks re-run: random noise, same flows; 'fell under every gradient attack'
    p = _para1("1c rerun", "\\lead{Attacks re-run on the hardened model.}")
    m = _rx("1c rerun", r"random noise also lost ground \(([\d.]+) to ([\d.]+) at \$\\vtheta=([\d.]+)\$\)\. "
            r"The table uses the same (\d+) flows", p)
    if m:
        _eps_lit("1c rerun eps", m.group(3), grid)
        e = float(m.group(3))
        cell_check("1c rerun random before", m.group(1), und(e, "random_search"))
        cell_check("1c rerun random after", m.group(2), dfd(e, "random_search"))
        int_cell_check("1c rerun flows", m.group(4), nflows)
        NCHECK_inc()
        if not dfd(e, "random_search") < und(e, "random_search"):
            FAILS.append("1c rerun: random-search detection did not fall")
    NCHECK_inc()
    notfell = [(a, e) for a in ("fd_fgsm", "pgd") for e in (0.05, 0.10, 0.15, 0.25, 0.50)   # the table's eps
               if dfd(e, a) > und(e, a) + 1e-9 or (und(e, a) > 0 and dfd(e, a) >= und(e, a) - 1e-9)]
    if notfell:
        FAILS.append(f"1c rerun: 'detection fell under every gradient attack' fails at {notfell}")

    # ---- Why the defense backfired (F8): the scaler
    p = _para1("1c scaler", "\\lead{Why the defense backfired (F8).}")
    m = _rx("1c scaler", r"\(median factor ([\d.]+)\)\. But (\d+) of the (\d+) features never varied in Lab~1's training "
            r"data, so they had no spread and Lab~1 divided them by ([\d.]+)\. \(The submitted Lab~1 drops these (\w+) "
            r"columns, which removes this cause\.\) Five of those ten are counters .*? "
            r"a tiny spread of ([\d.]+)\. The same allowed change therefore looks eight times larger", p)
    if m:
        sc = ex["scalers"]
        cell_check("1c scaler median factor", m.group(1), sc["median_ratio_all_77"])
        int_cell_check("1c scaler constant features", m.group(2), len(abf["degenerate_features"]))
        int_cell_check("1c scaler feature count", m.group(3), bu["n_features_total"])
        NCHECK_inc()
        if sc["n_features"] != bu["n_features_total"]:
            FAILS.append("1c scaler: report_extras and the attacker budget disagree on the feature count")
        cell_check("1c scaler Lab 1 divisor", m.group(4), sc["original_scale_on_them"][0])
        cell_check("1c scaler refit spread", m.group(6), sc["defended_scale_on_them"][0])
        sa = _split_audit()
        NCHECK_inc()
        if sa is None or _WORD_NUM.get(m.group(5).lower()) != len(sa["lab1_submitted"]["wed_constant_columns_dropped"]) \
                or not sa["checks"]["our_constant_set_equals_lab1_dropped"] \
                or len(sa["lab1_submitted"]["wed_constant_columns_dropped"]) != len(abf["degenerate_features"]):
            FAILS.append("1c scaler: 'the submitted Lab 1 drops these ten columns' does not match results/split_audit.json "
                         "(dropped-column count, or our constant set differs from the dropped set)")
        NCHECK_inc()
        names = sc["degenerate_mutable_features"]
        if not (len(set(sc["original_scale_on_them"])) == 1 and len(set(sc["defended_scale_on_them"])) == 1
                and len(names) == 5
                and all("\\texttt{" + n.replace("_", "\\_") + "}" in p for n in names)
                and abs(sc["original_scale_on_them"][0] / sc["defended_scale_on_them"][0] - 8) < 1e-9
                and all(v == 0.0 for v in se["degenerate_training_value"].values())):
            FAILS.append("1c scaler: the five named counters, 'constant in training', or 'eight times' do not match the JSON")

    # ---- Ablation table
    rows = table_rows("1c ablation table", "\\textbf{Config}")
    eps_cols = (0.10, 0.15, 0.25, 0.50)
    spec = (("Lab~1 model", None), ("two-round model (2 rounds)", "full_2r"), ("1 round, not frozen", "full_1r"),
            ("2 rounds, 100{,}000 rows", "sub100k_2r"), ("2 rounds, frozen", "full_2r_frozen"),
            ("one-round candidate (1 round, frozen)", "full_1r_frozen"))
    for prefix, key in spec:
        lab = f"1c ablation {key or 'undefended'}"
        c = row_len(lab, table_row(lab, rows, prefix), 10)
        if c is None:
            continue
        for i, e in enumerate(eps_cols):
            for j, a in enumerate(("fd_fgsm", "pgd")):
                cell_check(f"{lab} eps={e} {a}", c[1 + 2 * i + j],
                           und(e, a) if key is None else cm(key, e, a))
        cell_check(f"{lab} clean accuracy", c[9], df["clean_accuracy"]["before"] if key is None
                   else abl[key]["clean_accuracy"])
        if key is None:
            continue
        r = abl[key]
        mm = _rx(f"{lab} label", r"(\d+) rounds?", c[0])
        if mm:
            int_cell_check(f"{lab} rounds", mm.group(1), r["rounds"])
        if key == "sub100k_2r":
            mm = _rx(f"{lab} rows", r"([\d{},]+) rows", c[0])
            if mm:
                int_cell_check(f"{lab} subsample rows", mm.group(1), r["n_rows"])
        NCHECK_inc()
        if ("frozen" in c[0] and "not frozen" not in c[0]) != (r["n_frozen_features"] == len(abf["degenerate_features"])) \
                or r["n_frozen_features"] not in (0, len(abf["degenerate_features"])):
            FAILS.append(f"{lab}: row label's frozen/not frozen disagrees with n_frozen_features {r['n_frozen_features']}")

    # ---- Cause 1
    p = _para1("1c cause 1", "\\lead{Cause 1, measured:")
    m = _rx("1c cause 1 head", r"restores ([\d.]+)/([\d.]+) \(FD-FGSM/PGD-40\) at \$\\vtheta=([\d.]+)\$, where the "
            r"unfrozen two-round model gave ([\d.]+)/([\d.]+)\.", p)
    if m:
        _eps_lit("1c cause 1 eps", m.group(3), grid)
        e = float(m.group(3))
        cell_check("1c cause 1 frozen FD", m.group(1), cm("full_2r_frozen", e, "fd_fgsm"))
        cell_check("1c cause 1 frozen PGD", m.group(2), cm("full_2r_frozen", e, "pgd"))
        cell_check("1c cause 1 unfrozen FD", m.group(4), cm("full_2r", e, "fd_fgsm"))
        cell_check("1c cause 1 unfrozen PGD", m.group(5), cm("full_2r", e, "pgd"))
        NCHECK_inc()
        if not (cm("full_2r", e, "fd_fgsm") < 0.5 < cm("full_2r_frozen", e, "fd_fgsm")
                and cm("full_2r", e, "pgd") < cm("full_2r_frozen", e, "pgd")):
            FAILS.append("1c cause 1: freezing does not restore detection at that eps in the JSON")
    sdg = se["summary_degenerate"]
    m = _rx("1c cause 1 magnified", r"the two-round model divides these five features by ([\d.]+)\. A change the "
            r"attacker sees as ([\d.]+) looks like ([\d.]+) to the model, eight times larger", p)
    if m:
        cell_check("1c cause 1 refit divisor", m.group(1), ex["scalers"]["defended_scale_on_them"][0])
        cell_check("1c cause 1 attacker's move", m.group(2), se["eps_std"])
        cell_check("1c cause 1 model's move", m.group(3), sdg["std_move_defended"]["median"], tol=0.005)
        NCHECK_inc()
        if not (sdg["sb_over_sd"]["min"] == sdg["sb_over_sd"]["max"] == 8.0
                and abs(se["eps_std"] * 8 - sdg["std_move_defended"]["median"]) < 1e-9):
            FAILS.append("1c cause 1: 'eight times larger' is not sb/sd == 8 on all five features")
    NCHECK_inc()
    sv = se["sensitivity_defended_i_own_units"]
    if not (all(v == 0.0 for v in se["sensitivity_undefended"]["degenerate_by_feature"].values())
            and sv["degenerate_min"] > 0.1):
        FAILS.append("1c cause 1: 'Lab 1 ignores the five features / the refit gives them live weights' not supported by the sensitivities")
    pg = {r["eps_std"]: r for r in se["pgd_scale_test"]["curve"]}
    ev = {r["eps_std"]: r for r in se["evasion_reference"]}
    m = _rx("1c cause 1 effect", r"At \$\\vtheta=([\d.]+)\$, PGD evades the two-round model ([\d.]+) of the time\. If the "
            r"five features get a limit in their own units, it falls to ([\d.]+); the Lab~1 model is evaded ([\d.]+) of "
            r"the time\. So most of the excess comes from the magnified inputs\.", p)
    if m:
        _eps_lit("1c cause 1 effect eps", m.group(1), grid)
        e = float(m.group(1))
        NCHECK_inc()
        if e not in pg or e not in ev:
            FAILS.append(f"1c cause 1 effect: no scale test at {m.group(1)}")
        else:
            real, own = pg[e]["all_mutable_real_box"]["evasion_rate"], pg[e]["all_mutable_five_own_units"]["evasion_rate"]
            cell_check("1c cause 1 effect real box", m.group(2), real)
            cell_check("1c cause 1 effect own units", m.group(3), own)
            cell_check("1c cause 1 effect undefended", m.group(4), ev[e]["undefended_pgd_evasion"])
            NCHECK_inc()
            share = (real - own) / (real - ev[e]["undefended_pgd_evasion"])
            if not (abs(real - (1 - dfd(e, "pgd"))) < 1e-9 and abs(ev[e]["undefended_pgd_evasion"] - (1 - und(e, "pgd"))) < 1e-9
                    and share > 0.5 and abs(share - ev[e]["share_of_excess_removed_by_own_units"]) < 1e-9):
                FAILS.append("1c cause 1 effect: scale test disagrees with the curves, or 'most of the excess' (> half) is false")
    NCHECK_inc()
    if not (abl["full_2r_frozen"]["n_frozen_features"] == len(abf["degenerate_features"])
            and all(v == 0.0 for v in se["degenerate_training_value"].values())):
        FAILS.append("1c cause 1: 'freezing the ten features removes it' needs a ten-feature frozen config over constant features")

    # ---- Cause 2
    p = _para1("1c cause 2", "\\lead{Cause 2, open: the second round.}")
    m = _rx("1c cause 2", r"At \$\\vtheta=([\d.]+)\$, one round at full data holds ([\d.]+) \(FD-FGSM\), but two rounds "
            r"give ([\d.]+) \(Fig\.~\\ref\{fig:ablation\}\)\. Two rounds on a ([\d{},]+)-row subsample still give "
            r"([\d.]+), and freezing does not prevent the collapse, so it is not the scaler\. Too few boundary-crossing "
            r"rows is ruled out: the attack generators cross \\textbf\{([\d.]+)\\%\} \(round (\d)\) and "
            r"\\textbf\{([\d.]+)\\%\} \(round (\d)\) of training rows at ([\d.]+)\. One unproven candidate: a full-data "
            r"round takes about (\d+)\$\\times\$ the gradient steps of the ([\d{},]+)-row round\.", p)
    if m:
        _eps_lit("1c cause 2 eps", m.group(1), grid)
        e = float(m.group(1))
        cell_check("1c cause 2 one round", m.group(2), cm("full_1r", e, "fd_fgsm"))
        cell_check("1c cause 2 two rounds", m.group(3), cm("full_2r", e, "fd_fgsm"))
        int_cell_check("1c cause 2 subsample rows", m.group(4), abl["sub100k_2r"]["n_rows"])
        cell_check("1c cause 2 subsample", m.group(5), cm("sub100k_2r", e, "fd_fgsm"))
        cr = rnd["crossing_by_round"]
        cell_check("1c cause 2 crossing round 0", m.group(6), cr[0]["frac_crossed_by_generator"] * 100)
        int_cell_check("1c cause 2 round label 0", m.group(7), cr[0]["round"])
        cell_check("1c cause 2 crossing round 1", m.group(8), cr[1]["frac_crossed_by_generator"] * 100)
        int_cell_check("1c cause 2 round label 1", m.group(9), cr[1]["round"])
        cell_check("1c cause 2 crossing eps", m.group(10), at["eps_std"])
        NCHECK_inc()
        if not (cr[0]["n_rows"] == cr[1]["n_rows"] == hist[0]["n_clean"]):
            FAILS.append("1c cause 2: crossing fractions are not over the training rows")
        int_cell_check("1c cause 2 step ratio", m.group(11), round(cr[0]["n_rows"] / abl["sub100k_2r"]["n_rows"]))
        int_cell_check("1c cause 2 step ratio base rows", m.group(12), abl["sub100k_2r"]["n_rows"])
        NCHECK_inc()
        if not (cm("full_1r", e, "fd_fgsm") > 0.5 > cm("full_2r", e, "fd_fgsm")
                and cm("sub100k_2r", e, "fd_fgsm") > 0.5 and cm("full_2r_frozen", e, "fd_fgsm") < 0.5
                and cr[1]["frac_crossed_by_generator"] > cr[0]["frac_crossed_by_generator"]):
            FAILS.append("1c cause 2: the open-item pattern (1 round holds, 2 rounds collapse, subsample holds, frozen "
                         "still collapses, round 1 crosses more than round 0) no longer holds in the JSON")
    NCHECK_inc()
    if re.search(r"(?i)\bproven that|\bcaused by|the cause is", p):
        FAILS.append("1c cause 2: the paragraph asserts a mechanism")
    NCHECK_inc()
    if not ("We claim no cause" in p and "We cannot explain" in p and "One unproven candidate" in p):
        FAILS.append("1c cause 2: lost its open-item wording")

    # ---- Figure 2 caption
    p = _para1("1c fig 2", "\\caption{Ablation, FD-FGSM")
    m = _rx("1c fig 2", r"same (\d+) flows\. Left: freezing the ten constant features removes the collapse at "
            r"\$\\vtheta=([\d.]+)\$ \(Cause 1\)\. Right: at ([\d.]+) the two-round full-data model sits at ([\d.]+), "
            r"while one round and the ([\d{},]+)-row refit hold ([\d.]+) \(Cause 2\)\.", p)
    if m:
        int_cell_check("1c fig 2 flows", m.group(1), nflows)
        _eps_lit("1c fig 2 left eps", m.group(2), grid)
        e1, e2 = float(m.group(2)), float(m.group(3))
        _eps_lit("1c fig 2 right eps", m.group(3), grid)
        cell_check("1c fig 2 two-round", m.group(4), cm("full_2r", e2, "fd_fgsm"))
        int_cell_check("1c fig 2 subsample rows", m.group(5), abl["sub100k_2r"]["n_rows"])
        cell_check("1c fig 2 one round and subsample", m.group(6), cm("full_1r", e2, "fd_fgsm"))
        NCHECK_inc()
        if not (_agree(m.group(6), cm("sub100k_2r", e2, "fd_fgsm"))
                and cm("full_2r", e1, "fd_fgsm") < 0.5 < cm("full_2r_frozen", e1, "fd_fgsm")):
            FAILS.append("1c fig 2: left/right panel claims disagree with the ablation curves")

    # ---- Randomized smoothing
    p = _para1("1c smoothing", "\\lead{Randomized smoothing and the certified radius.}")
    m = _rx("1c smoothing setup", r"noise \$\\sigma=([\d.]+)\$ standardized, \$n=([\d{},]+)\$ samples, a statistical "
            r"lower bound on the vote share \(Clopper-Pearson\), and (\d+) test flows on the two-round model", p)
    if m:
        cell_check("1c smoothing sigma", m.group(1), sm1["sigma_std"])
        int_cell_check("1c smoothing n", m.group(2), sm1["n_samples"])
        int_cell_check("1c smoothing test flows", m.group(3), sm1["n_certified"])
        NCHECK_inc()
        if not (at["rounds"] == 2 and sd["smoothing"]["n_samples"] == sm1["n_samples"]
                and sd["smoothing"]["sigma_std"] == sm1["sigma_std"]):
            FAILS.append("1c smoothing: not the two-round model's certificate, or the detection run used other settings")
    m = _rx("1c smoothing radius", r"median certified radius is \\textbf\{R = ([\d.]+)\} standardized units, with (\d+) "
            r"abstentions and a median vote share of ([\d.]+); the spread is small \(p25 ([\d.]+), mean ([\d.]+)\)", p)
    if m:
        cell_check("1c smoothing median radius", m.group(1), sm1["radius_median"])
        int_cell_check("1c smoothing abstentions", m.group(2), sm1["n_abstain"])
        cell_check("1c smoothing vote share", m.group(3), sm1["median_vote_share"])
        cell_check("1c smoothing p25", m.group(4), sm1["radius_p25"])
        cell_check("1c smoothing mean", m.group(5), sm1["radius_mean"])
        NCHECK_inc()
        if not (sm1["radius_mean"] - sm1["radius_p25"] < 0.01 + 0 and sm1["radius_p75"] - sm1["radius_p25"] < 0.1):
            FAILS.append("1c smoothing: 'the spread is small' contradicted by p25/p75/mean")
    from statistics import NormalDist
    lb = (sm1["alpha"] / 2) ** (1 / sm1["n_samples"])
    z = NormalDist().inv_cdf(lb)
    m = _rx("1c smoothing ceiling", r"Even if all ([\d{},]+) noisy copies agree, we can only be sure of ([\d.]+)\\% "
            r"agreement\. That caps the radius at \$([\d.]+)\\times([\d.]+)=([\d.]+)\$", p)
    if m:
        int_cell_check("1c smoothing ceiling copies", m.group(1), sm1["n_samples"])
        cell_check("1c smoothing guaranteed share (%)", m.group(2), lb * 100)
        cell_check("1c smoothing ceiling sigma", m.group(3), sm1["sigma_std"])
        cell_check("1c smoothing ceiling z", m.group(4), z)
        cell_check("1c smoothing ceiling radius", m.group(5), sm1["sigma_std"] * z)
        NCHECK_inc()
        if not (abs(sm1["sigma_std"] * z - sm1["radius_median"]) < 1e-9 and sm1["radius_p75"] == sm1["radius_median"]):
            FAILS.append("1c smoothing ceiling: the median is not the n-capped radius sigma x z in the JSON")
    m = _rx("1c smoothing extent", r"the attack at \$\\vtheta=([\d.]+)\$ reaches a Euclidean length of \\textbf\{([\d.]+)\}, "
            r"far beyond ([\d.]+)\.", p)
    if m:
        l2 = se["l2_extent"]
        cell_check("1c smoothing extent eps", m.group(1), l2["eps_std"])
        cell_check("1c smoothing extent length", m.group(2), l2["l2_defended_units"])
        cell_check("1c smoothing extent radius", m.group(3), sm1["radius_median"])
        NCHECK_inc()
        if not (l2["certified_radius_median"] == sm1["radius_median"] and l2["l2_defended_units"] > 2 * sm1["radius_median"]):
            FAILS.append("1c smoothing extent: the attack's length is not far beyond the certified radius in the JSON")
    cv = next((x for x in sd["curve"] if abs(x["eps_std"] - 0.15) < 1e-9), None)
    m = _rx("1c smoothing detection", r"At \$\\vtheta=([\d.]+)\$, PGD is detected on ([\d.]+) of flows, and the adaptive "
            r"attack \(PGD averaged over (\d+) noise draws per step\) on ([\d.]+)\. The unsmoothed two-round model "
            r"scores ([\d.]+) against PGD\. FD-FGSM is detected on ([\d.]+) \((\d+/\d+)\) at \$\\vtheta=([\d.]+)\$, "
            r"as unsmoothed", p)
    if m and cv is None:
        FAILS.append("1c smoothing detection: no eps=0.15 point in task1c_smoothing_detection.json")
    elif m:
        _eps_lit("1c smoothing detection eps", m.group(1), grid)
        _eps_lit("1c smoothing detection FD eps", m.group(8), grid)
        e = float(m.group(1))
        cv = next((x for x in sd["curve"] if abs(x["eps_std"] - e) < 1e-9), None)
        if cv is None:
            FAILS.append(f"1c smoothing detection: no point at {m.group(1)}")
        else:
            pgx, eot, fdx = cv["sources"]["defended_base"]["pgd"], cv["sources"]["eot_pgd"]["pgd_vs_smoothed"], \
                cv["sources"]["defended_base"]["fd_fgsm"]
            cell_check("1c smoothing detection PGD", m.group(2), pgx["detection_rate"])
            int_cell_check("1c smoothing detection EOT draws", m.group(3), sd["eot"]["draws_per_step"])
            cell_check("1c smoothing detection EOT", m.group(4), eot["detection_rate"])
            cell_check("1c smoothing detection unsmoothed PGD", m.group(5), pgx["unsmoothed_defended_detection_rate"])
            cell_check("1c smoothing detection FD", m.group(6), fdx["detection_rate"])
            _frac("1c smoothing detection FD count", m.group(7), fdx["n_detected"], fdx["n_flows"])
            NCHECK_inc()
            if not (abs(pgx["unsmoothed_defended_detection_rate"] - dfd(e, "pgd")) < 1e-9
                    and all(v["detection_rate"] <= v["unsmoothed_defended_detection_rate"] for v in (pgx, eot, fdx))
                    and fdx["detection_rate"] == fdx["unsmoothed_defended_detection_rate"]):
                FAILS.append("1c smoothing detection: 'did not raise detection' / 'as unsmoothed' / unsmoothed = table cell not supported")

    # ---- What the defense costs
    p = _para1("1c costs", "\\lead{What the defense costs.}")
    m = _rx("1c costs", r"two CPU refits of ([\d.]+)\\,s and ([\d.]+)\\,s\. Scoring one flow takes a median ([\d.]+)\\,ms, "
            r"the same as the Lab~1 model, so retraining adds no serving cost\. Smoothing's ([\d{},]+) noisy copies took "
            r"a median ([\d.]+)\\,ms as one batch", p)
    if m:
        NCHECK_inc()
        if len(hist) != 2:
            FAILS.append("1c costs: 'two refits' but the history has a different number of rounds")
        cell_check("1c costs refit 1", m.group(1), hist[0]["seconds"])
        cell_check("1c costs refit 2", m.group(2), hist[1]["seconds"])
        lat = ops["models"]
        adv, lab = lat["adv_trained_2round_1c"], lat["deployed_lab1"]
        cell_check("1c costs scoring one flow", m.group(3), adv["single_flow"]["median_ms"])
        NCHECK_inc()
        if round(adv["single_flow"]["median_ms"], 2) != round(lab["single_flow"]["median_ms"], 2):
            FAILS.append("1c costs: 'the same as the Lab 1 model' - the two single-flow medians differ at 2 decimals")
        int_cell_check("1c costs smoothing copies", m.group(4), sm1["n_samples"])
        NCHECK_inc()
        if adv["batch_1000"]["batch_size"] != sm1["n_samples"] or lab["batch_1000"]["batch_size"] != sm1["n_samples"]:
            FAILS.append("1c costs: the stored batch is not the smoothing batch size")
        # the quoted batch time is the deployed Lab 1 model's; the two-round model's is within 0.01 ms of it
        cell_check("1c costs smoothing batch (deployed Lab 1 model)", m.group(5),
                   lab["batch_1000"]["median_batch_seconds"] * 1000)
        NCHECK_inc()
        if abs(adv["batch_1000"]["median_batch_seconds"] - lab["batch_1000"]["median_batch_seconds"]) * 1000 > 0.01:
            FAILS.append("1c costs: the two-round model's batch time is not within 0.01 ms of the quoted one")

    # ---- The one-round candidate
    c1, e0 = "full_1r_frozen", 0.15
    p = _para1("1c candidate", "\\lead{The one-round candidate.}")
    NCHECK_inc()
    if not (abl[c1]["rounds"] == 1 and abl[c1]["n_frozen_features"] == len(abf["degenerate_features"])
            and rc["adversarial_training"]["rounds"] == 1 and rc["n_frozen_features"] == len(abf["degenerate_features"])
            and rc["reproduction"]["all_equal"]):
        FAILS.append("1c candidate: the stored candidate is not one round with the ten constant features frozen")
    NCHECK_inc()
    worse = {(a, e) for a in ("fd_fgsm", "pgd") for e in sorted(r["eps_std"] for r in abl[c1]["curve_after"])
             if cm(c1, e, a) < und(e, a) - 1e-9}
    if worse != {("fd_fgsm", e0)}:
        FAILS.append(f"1c candidate: 'never lower than the Lab 1 model, except FD-FGSM at 0.15' fails: lower at {sorted(worse)}")
    NCHECK_inc()
    # 'removes almost all of F8's extra risk': share of the two-round model's extra evasion removed
    shares = []
    for a in ("fd_fgsm", "pgd"):
        ex2 = (1 - cm("full_2r", e0, a)) - (1 - und(e0, a))
        exc = (1 - cm(c1, e0, a)) - (1 - und(e0, a))
        shares.append(1 - max(exc, 0) / ex2 if ex2 > 0 else None)
    if None in shares or min(shares) < 0.9:
        FAILS.append(f"1c candidate: 'removes almost all of F8's extra risk' - share removed {shares} (needs >= 0.9)")
    NCHECK_inc()
    if not 0 < und(e0, "fd_fgsm") - cm(c1, e0, "fd_fgsm") < 0.05:
        FAILS.append("1c candidate: 'FD-FGSM at 0.15 is slightly lower' - the gap is not in (0, 0.05)")
    m = _rx("1c candidate detection", r"except FD-FGSM at \$\\vtheta=([\d.]+)\$ \((\d+/\d+) flows detected against "
            r"(\d+/\d+)\)\. It is higher at ([\d.]+) \(([\d.]+)/([\d.]+) against ([\d.]+), FD-FGSM/PGD-40\) and, "
            r"for PGD-40, at ([\d.]+) \(([\d.]+) against ([\d.]+); FD-FGSM ties at ([\d.]+)\)\. "
            r"Clean accuracy is ([\d.]+)\.", p)
    if m:
        _eps_lit("1c candidate eps", m.group(1), grid)
        e = float(m.group(1))
        for lit, v, lab in ((m.group(2), cm(c1, e, "fd_fgsm"), "candidate"), (m.group(3), und(e, "fd_fgsm"), "Lab 1")):
            k = round(v * nflows)
            NCHECK_inc()
            if abs(v * nflows - k) > 1e-6:
                FAILS.append(f"1c candidate {lab}: rate {v} is not a whole number of {nflows} flows")
            _frac(f"1c candidate {lab} flows", lit, k, nflows)
        # 'higher at 0.10': candidate FD-FGSM / PGD-40 against the Lab 1 model (source: the 'Before' table, b1 curve)
        e1, e2 = float(m.group(4)), float(m.group(8))
        _eps_lit("1c candidate eps", m.group(4), grid)
        _eps_lit("1c candidate eps", m.group(8), grid)
        cell_check(f"1c candidate FD at {e1}", m.group(5), cm(c1, e1, "fd_fgsm"))
        cell_check(f"1c candidate PGD at {e1}", m.group(6), cm(c1, e1, "pgd"))
        cell_check(f"1c Lab 1 FD at {e1}", m.group(7), und(e1, "fd_fgsm"))
        cell_check(f"1c Lab 1 PGD at {e1}", m.group(7), und(e1, "pgd"))
        # 'higher at 0.25 for PGD-40, FD-FGSM ties'
        cell_check(f"1c candidate PGD at {e2}", m.group(9), cm(c1, e2, "pgd"))
        cell_check(f"1c Lab 1 PGD at {e2}", m.group(10), und(e2, "pgd"))
        cell_check(f"1c candidate FD tie at {e2}", m.group(11), cm(c1, e2, "fd_fgsm"))
        cell_check(f"1c Lab 1 FD tie at {e2}", m.group(11), und(e2, "fd_fgsm"))
        r3 = lambda v: round(v, 3)
        NCHECK_inc()
        if not all(r3(cm(c1, e1, a)) > r3(und(e1, a)) for a in ("fd_fgsm", "pgd")):
            FAILS.append(f"1c candidate: 'higher' (strictly, at 3 decimals) fails at {e1} for FD-FGSM or PGD-40")
        NCHECK_inc()
        if not r3(cm(c1, e2, "pgd")) > r3(und(e2, "pgd")):
            FAILS.append(f"1c candidate: 'higher' (strictly, at 3 decimals) fails for PGD-40 at {e2}")
        NCHECK_inc()
        if r3(cm(c1, e2, "fd_fgsm")) != r3(und(e2, "fd_fgsm")):
            FAILS.append(f"1c candidate: 'FD-FGSM ties' (equal at 3 decimals) fails at {e2}")
        cell_check("1c candidate clean accuracy", m.group(12), abl[c1]["clean_accuracy"])
        NCHECK_inc()
        if abl[c1]["clean_accuracy"] != rc["clean_accuracy"]:
            FAILS.append("1c candidate: ablation and recommended-config clean accuracy differ")
    mem = rc["membership"]
    m = _rx("1c candidate membership", r"Membership-inference accuracy:\} ([\d.]+) \((\d+)\\% CI \[([\d.]+), ([\d.]+)\]\) "
            r"against the blind control ([\d.]+), so no leak", p)
    if m:
        arm = mem["recommended_target_arm"]["confidence_threshold"]
        cell_check("1c candidate membership accuracy", m.group(1), arm["accuracy"])
        NCHECK_inc()
        if "ci95" in arm and m.group(2) == "95":
            cover(m.group(2), [m.group(2)], "ci level")
        else:
            FAILS.append("1c candidate membership: the interval is not the stored 95% interval (ci95)")
        cell_check("1c candidate membership CI low", m.group(3), arm["ci95"][0])
        cell_check("1c candidate membership CI high", m.group(4), arm["ci95"][1])
        ctrl = mem["references"]["blind_control"]["confidence_threshold"]["accuracy"]
        cell_check("1c candidate membership blind control", m.group(5), ctrl)
        NCHECK_inc()
        if not (arm["ci95"][1] < ctrl and ctrl == wd["arms"][1]["attacks"]["confidence_threshold"]["accuracy"]
                and mem["index_provenance_equals_task1a_json"]):
            FAILS.append("1c candidate membership: not below the blind control, or not the 1A evaluation indices")
    m = _rx("1c candidate radius", r"Certified radius:\} median ([\d.]+), the same as the two-round model, with "
            r"(\d+/\d+) abstentions", p)
    if m:
        cr2 = rc["smoothing"]["certify_200_test_rows"]
        cell_check("1c candidate radius", m.group(1), cr2["radius_median"])
        _frac("1c candidate abstentions", m.group(2), cr2["n_abstain"], cr2["n_certified"])
        NCHECK_inc()
        if round(cr2["radius_median"], 3) != round(sm1["radius_median"], 3):
            FAILS.append("1c candidate radius: not the same as the two-round model's median radius")
    m = _rx("1c candidate smoothing", r"Detection under smoothing:\} about the same as unsmoothed \(FD-FGSM at "
            r"\$\\vtheta=([\d.]+)\$: (\d+/\d+) smoothed, (\d+/\d+) unsmoothed\)", p)
    if m:
        _eps_lit("1c candidate smoothing eps", m.group(1), grid)
        e = float(m.group(1))
        for mask, cells in (("frozen", rc["smoothing"]["detection_frozen_mask"]["cells"]),
                            ("unfrozen", rc["smoothing"]["detection_unfrozen_mask"]["cells_eps_0.15_0.25"])):
            pt = next((c for c in cells if abs(c["eps_std"] - e) < 1e-9), None)
            NCHECK_inc()
            if pt is None:
                FAILS.append(f"1c candidate smoothing: no {mask}-mask point at {m.group(1)}")
                continue
            f = pt["sources"]["defended_base"]["fd_fgsm"]
            if (f["detected"]["fraction"], f["unsmoothed_detected"]["fraction"]) != (m.group(2), m.group(3)):
                FAILS.append(f"1c candidate smoothing ({mask} mask): JSON says {f['detected']['fraction']} smoothed, "
                             f"{f['unsmoothed_detected']['fraction']} unsmoothed")
            else:
                NCHECK_inc()
                if not abs(f["detected"]["value"] - f["unsmoothed_detected"]["value"]) < 0.05:
                    FAILS.append(f"1c candidate smoothing ({mask} mask): not 'about the same' (within 5 points)")
        _frac("1c candidate smoothing = ablation count", m.group(3), round(cm(c1, e, "fd_fgsm") * nflows), nflows)
        pt = next((c for c in rc["smoothing"]["detection_frozen_mask"]["cells"] if abs(c["eps_std"] - e) < 1e-9), None)
        if pt is not None:
            f = pt["sources"]["defended_base"]["fd_fgsm"]
            _frac("1c candidate smoothing smoothed count", m.group(2), f["detected"]["count"], f["detected"]["n"])


def main() -> int:
    fns = (section_wiring, task1a, task1b, task1_text, textbook_citations, lab1_reproduction, split_audit_text, task1c_defense, task1c_text, task2, cross_file,
           exec_summary, task2_text, task2_section, task2_rollup, task1_phase2, risk_table, risk_rule_checks,
           wording_checks, task2_excerpts, task2_threat_and_benign, task2_owasp_filing,
           fixes_10_12, fix5, recommendation_controls, recommendations_text, expected_guard_claims, evidence_map, final_wording_checks, ai_disclosure, title_block, coverage_closure)
    if os.environ.get("GATE_SKIP_CLOSURE"):      # coverage baseline only (number_coverage --baseline)
        fns = fns[:-1]
    for fn in fns:
        _CURRENT[0] = fn.__name__
        try:
            fn()
        except Exception as e:      # a check that cannot run is a FAILURE, never a crash or skip
            FAILS.append(f"{fn.__name__}: check aborted ({type(e).__name__}: {e}); "
                         f"its remaining checks did not run")
        if fn is section_wiring and FAILS:     # structure broken: later checks would only crash
            print(f"{NCHECK} checks run, {len(FAILS)} failures")
            for f in FAILS:
                print("  FAIL:", f)
            print("\nreport (report.tex + sections/) does NOT match results/*.json")
            return 1
    if "--dump-coverage" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--dump-coverage") + 1])
        out.write_text(json.dumps({"n_checks": NCHECK, "n_fail": len(FAILS),
                                   "entries": REGISTRY}, indent=0))
    print(f"{NCHECK} checks run, {len(FAILS)} failures")
    for f in FAILS:
        print("  FAIL:", f)
    if FAILS:
        print("\nreport (report.tex + sections/) does NOT match results/*.json")
        return 1
    print("report.tex matches results/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
