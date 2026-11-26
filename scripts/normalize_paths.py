"""Rewrite machine-specific absolute paths in stored provenance strings to repo-relative form.

Only the string fields listed in TARGETS are touched; no gate reads them and no number is
changed. Idempotent. Run from anywhere:

    python3 scripts/normalize_paths.py            # rewrite in place
    python3 scripts/normalize_paths.py --check    # report only, exit 1 if anything would change
"""
from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent

# (results file, key path) of each pure-provenance string field.
TARGETS = [
    ("results/task2_summary.json", ("input",)),
    ("results/task1c_ablation.json", ("configs", None, "adversarial_training", "loaded_from_cache")),
]


def to_relative(value: str) -> str:
    p = PurePosixPath(value)
    if not p.is_absolute():
        return value
    try:
        return str(p.relative_to(ROOT.as_posix()))
    except ValueError:
        pass
    if "models" in p.parts:               # a cached model path from another machine
        return str(PurePosixPath(*p.parts[p.parts.index("models"):]))
    return p.name


def walk(node, keys, changes, trail=""):
    if not keys:
        return
    k, rest = keys[0], keys[1:]
    items = (list(enumerate(node)) if isinstance(node, list) else list(node.items())) if k is None \
        else ([(k, node[k])] if isinstance(node, dict) and k in node else [])
    for key, child in items:
        if rest:
            walk(child, rest, changes, f"{trail}/{key}")
        elif isinstance(child, str):
            new = to_relative(child)
            if new != child:
                node[key] = new
                changes.append((f"{trail}/{key}", child, new))


def main() -> int:
    check = "--check" in sys.argv
    dirty = 0
    for rel, keys in TARGETS:
        path = ROOT / rel
        text = path.read_text()
        data = json.loads(text)
        changes: list = []
        walk(data, keys, changes)
        for where, old, new in changes:
            print(f"{rel}{where}: {old} -> {new}")
        if changes:
            dirty += 1
            if not check:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False)
                            + ("\n" if text.endswith("\n") else ""))
    return 1 if (check and dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
