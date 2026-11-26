"""Shared reader for the report source: report/report.tex plus its section files.

report.tex holds the preamble and title block and pulls the body in with
    \\input{sections/NN_name}
(one per line). Every gate must see the EXPANDED text, so they all call this module
instead of reading report.tex directly.

Design
  * Only `\\input{sections/...}` lines are expanded (recursively). `\\input{style}` and
    `\\input{generated/...}` stay as written: the gates read the generated files separately
    and check that the \\input line is present.
  * The input line is replaced, newline included, by the file's exact bytes, so the
    expanded text is the old single-file report byte for byte.
  * origins() gives, for every line of the expanded text, (file, line) -- "file" is relative
    to report/ -- so coverage reports can say sections/02_task1.tex:45.
  * unwired_sections() lists sections/*.tex on disk that report.tex does not \\input, and
    missing_inputs() lists \\input targets that do not exist; the gate fails on either.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = ROOT / "report" / "report.tex"
_INPUT = re.compile(r"^\s*\\input\{(sections/[^}]+)\}\s*$")


def _strip_comment(line: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", line)


def _section_path(base: Path, name: str) -> Path:
    return base / (name if name.endswith(".tex") else name + ".tex")


def _walk(path: Path, base: Path, rel: str, text: list, origin: list, seen: tuple) -> None:
    if path in seen:
        raise RuntimeError(f"circular \\input of {rel}")
    raw = path.read_text()
    for n, line in enumerate(raw.splitlines(keepends=True), 1):
        m = _INPUT.match(_strip_comment(line.rstrip("\n")))
        if m:
            sec = _section_path(base, m.group(1))
            if not sec.exists():
                raise FileNotFoundError(f"{rel}:{n}: \\input{{{m.group(1)}}} -> {sec} does not exist")
            _walk(sec, base, str(Path(m.group(1)).with_suffix(".tex")), text, origin, seen + (path,))
        else:
            text.append(line)
            origin.append((rel, n))


def _expand(tex: Path):
    tex = Path(tex)
    text: list = []
    origin: list = []
    _walk(tex, tex.parent, tex.name, text, origin, ())
    return "".join(text), origin


def read_expanded(tex=None) -> str:
    """Full report text with every \\input{sections/..} replaced by the file's content."""
    return _expand(tex or DEFAULT_TEX)[0]


def origins(tex=None) -> list:
    """[(file, lineno)] for each line of read_expanded(tex), 0-based index = line - 1."""
    return _expand(tex or DEFAULT_TEX)[1]


def locate(origin: list, line: int) -> str:
    """'sections/02_task1.tex:45' for a 1-based line of the expanded text."""
    if 1 <= line <= len(origin):
        f, n = origin[line - 1]
        return f"{f}:{n}"
    return f"report.tex(expanded):{line}"


def wired_sections(tex=None) -> list:
    tex = Path(tex or DEFAULT_TEX)
    out = []
    for line in tex.read_text().splitlines():
        m = _INPUT.match(_strip_comment(line))
        if m:
            out.append(str(Path(m.group(1)).with_suffix(".tex")))
    return out


def unwired_sections(tex=None) -> list:
    """section files present on disk but not \\input from report.tex"""
    tex = Path(tex or DEFAULT_TEX)
    wired = set(wired_sections(tex))
    return sorted(str(p.relative_to(tex.parent)) for p in (tex.parent / "sections").glob("*.tex")
                  if str(p.relative_to(tex.parent)) not in wired)


def missing_inputs(tex=None) -> list:
    tex = Path(tex or DEFAULT_TEX)
    return [s for s in wired_sections(tex) if not (tex.parent / s).exists()]


if __name__ == "__main__":
    import sys
    sys.stdout.write(read_expanded(sys.argv[1] if len(sys.argv) > 1 else None))
