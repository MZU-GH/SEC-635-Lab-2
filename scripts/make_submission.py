"""Build the submission package (stdlib only).

    python3 scripts/make_submission.py              # source = git archive of HEAD (after committing)
    python3 scripts/make_submission.py --worktree   # source = the current working tree (tracked
                                                    # plus untracked-not-ignored files), for a
                                                    # check before committing
    python3 scripts/make_submission.py --out DIR    # write somewhere other than ../submission

The source tarball is made with `git archive`, so the `export-ignore` rules of
.gitattributes apply: course materials and internal notes are not in it, and it has the
same content as the published repository. The script lists the tarball and stops if an
excluded path is present. It writes, outside the repository, to ../submission/:
    sec635-lab2-submission.tar.gz     source tree, prefix sec635-lab2/
    SEC635-Lab2-Report.pdf            copy of report/report.pdf
    sec635-lab2-data-20261124.tar.gz  copy of the data snapshot (original name, sha256 checked)
    SHA256SUMS                        sha256 of the three files (computed with hashlib)
    README.txt                        the files, the repository and the data Release
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = "sec635-lab2-submission.tar.gz"
PDF = "SEC635-Lab2-Report.pdf"
DATA = "sec635-lab2-data-20261124.tar.gz"
DATA_SRC = ROOT.parent / DATA   # the data snapshot sits next to the repository directory
DATA_SHA256 = "cab06d805fd04e49aed0f5f4845b750d291bc0332e66584338d701a087d4817f"
REPO_URL = "https://github.com/MZU-GH/SEC-635-Lab-2"
DATA_URL = (f"{REPO_URL}/releases/download/data-snapshot-20261124/{DATA}")

README = f"""SEC-635 Lab 2 submission

Files:
  {PDF}   the report
  {ARCHIVE}   source code (same content as the public repository)
  {DATA}   data snapshot (not in the source archive)
  SHA256SUMS   checksums of the three files above

Public repository: {REPO_URL}
Data snapshot, also published as a GitHub Release asset (tag data-snapshot-20261124):
  {DATA_URL}
  sha256 {DATA_SHA256}

Quick start:
  1. tar xzf {ARCHIVE} && tar xzf {DATA} -C sec635-lab2   (source first, then the data into it)
  2. cd sec635-lab2 and run the gates as in report/REPRODUCING.md (Step 0 and section 12)
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded_patterns() -> list[str]:
    """Paths .gitattributes marks export-ignore, as repository-relative strings."""
    out = []
    for line in (ROOT / ".gitattributes").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and "export-ignore" in parts[1:] and not line.startswith("#"):
            out.append(parts[0].lstrip("/"))
    return out


def worktree_tree() -> str:
    """Tree object of the working tree (tracked + untracked, honouring .gitignore), built
    with a throw-away index; the real index, HEAD and refs are not touched."""
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / "index"))
        subprocess.run(["git", "add", "-A"], cwd=ROOT, env=env, check=True)
        r = subprocess.run(["git", "write-tree"], cwd=ROOT, env=env, check=True,
                           capture_output=True, text=True)
    return r.stdout.strip()


def check_no_excluded(tarball: Path) -> None:
    pats = excluded_patterns()
    with tarfile.open(tarball) as t:
        names = [n.split("/", 1)[1] if "/" in n else "" for n in t.getnames()]
    bad = [n for n in names for p in pats
           if n == p.rstrip("/") or n.startswith(p.rstrip("/") + "/")]
    if bad:
        raise SystemExit(f"excluded paths are in {tarball.name}: {bad[:5]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", action="store_true",
                    help="archive the working tree instead of HEAD (use before committing)")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "submission")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "archive", "--format=tar.gz", "--prefix=sec635-lab2/", "-o", str(out / ARCHIVE)]
    if a.worktree:
        cmd += ["--worktree-attributes", worktree_tree()]
    else:
        cmd += ["HEAD"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    check_no_excluded(out / ARCHIVE)
    if sha256(DATA_SRC) != DATA_SHA256:
        raise SystemExit(f"{DATA_SRC} does not match the published sha256 {DATA_SHA256}")
    shutil.copyfile(ROOT / "report" / "report.pdf", out / PDF)
    shutil.copyfile(DATA_SRC, out / DATA)
    lines = [f"{sha256(out / name)}  {name}" for name in (ARCHIVE, PDF, DATA)]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    (out / "README.txt").write_text(README)
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
