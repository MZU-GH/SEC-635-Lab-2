# SEC635 — Lab 2: Adversarial AI and LLM Attack–Defense

Coursework repository for SEC-635 (AI for Cybersecurity), Lab 2, drawn from
Chapter 11 of the course text. The assignment brief and the course chapters are course
material and are not part of this repository.

Two targets are built from course material, attacked through their outputs, then defended:

1. **Intrusion detector** — the Lab 1 neural-network classifier trained on a day of
   CICIDS-2018, served behind a small HTTP API. Membership inference (1A), feature-space
   evasion (1B), adversarial-training hardening (1C).
2. **Security chatbot** — a RAG copilot over the 706 live ATT&CK techniques plus the CISA KEV
   list. Prompt injection (2A), jailbreak (2B), Llama Guard 3 + input filter + hardened
   prompt (2C).

## Layout

| Path | Contents |
| --- | --- |
| `lab1/` | First-version Lab 1 detector (the attack target): preprocessing, training, saved model. Its 70/30 split follows the raw 12-hour timestamp, so it is not true time order; the "honest temporal" wording in `lab1/src` comes from Lab 1 and predates this finding (`scripts/split_audit.py`, report §2). |
| `detector/` | HTTP API around the detector, plus the Lab 2 attacks against it. |
| `chatbot/` | RAG copilot, knowledge-base ingest, and the Lab 2 attacks against it. |
| `results/` | Committed numbers, payload/response transcripts, and figures cited in the report. |
| `report/` | Red-team assessment report (max 8 pages). `report.tex` holds the title block and pulls in `sections/*.tex` (the body) and `style.tex` (layout); `scripts/report_source.py` expands them so every gate reads the full text. `report/generated/` is written by `scripts/make_task2_evidence.py`. |
| `scripts/` | Gates (`verify_report_numbers.py`, `number_coverage.py`, `verify_claims.py`), the report-source reader (`report_source.py`), result scripts (`ops_measurements.py`, `detector_extras.py`, `report_extras.py`, `scale_effect.py`, `noise_check.py` and others, listed in `report/REPRODUCING.md`), citation recorders (`make_textbook_citations.py`, `make_brief_citations.py`), the split audit (`split_audit.py` writes `results/split_audit.json`: the Lab 2 split is not in true time order, and the submitted Lab 1's clock-repaired split gives a test window it never learned), Task 2 evidence and retrieval probe, attacker budget, `make_submission.py`, and `scripts/audit/` comparison tools. |

## Quick start

The repository holds code, results and the report. The 217 MB data snapshot (cleaned
CICIDS-2018 parquets for the two Lab 1 days, the Chroma knowledge base built from the MITRE
ATT&CK enterprise bundle and the CISA KEV catalog, the public `all-MiniLM-L6-v2` embedding
model, and the cached Task 1C model) is a GitHub Release asset, tag `data-snapshot-20261124`:

- URL: https://github.com/MZU-GH/SEC-635-Lab-2/releases/download/data-snapshot-20261124/sec635-lab2-data-20261124.tar.gz
- sha256: `cab06d805fd04e49aed0f5f4845b750d291bc0332e66584338d701a087d4817f`

```bash
git clone https://github.com/MZU-GH/SEC-635-Lab-2.git sec635-lab2 && cd sec635-lab2
curl -fL -o sec635-lab2-data-20261124.tar.gz \
  https://github.com/MZU-GH/SEC-635-Lab-2/releases/download/data-snapshot-20261124/sec635-lab2-data-20261124.tar.gz
echo "cab06d805fd04e49aed0f5f4845b750d291bc0332e66584338d701a087d4817f  sec635-lab2-data-20261124.tar.gz" | sha256sum -c - && tar xzf sec635-lab2-data-20261124.tar.gz
```

(`sha256sum -c` must print `OK` before the archive is extracted; on macOS use `shasum -a 256 -c -`.)
Rebuilding the data from the live feeds is possible (`report/REPRODUCING.md`, sections 1, 9
and 10) but gives different KEV counts, because the CISA feed grows over time; the report
numbers that quote them then no longer match. The snapshot is the reference.

## Reproducing

See `report/REPRODUCING.md` for the ordered commands. Setup is a stdlib venv plus pip
(`python3 -m venv .venv`, Python 3.12), then `pip install -r requirements.lock` for the exact
installed versions; set `TMPDIR` to a disk with space, since the torch wheel pulls several GB
of CUDA libraries (`TMPDIR=~/pip-tmp .venv/bin/pip install -r requirements.lock`). Tasks 1A and 1B attack the detector over HTTP:

```bash
.venv/bin/uvicorn detector.api:app --host 127.0.0.1 --port 8055    # leave running
.venv/bin/python -u detector/membership.py --url-wednesday http://127.0.0.1:8055 ...
.venv/bin/python -u detector/evasion.py --via-http --base-url http://127.0.0.1:8055
python3 scripts/make_task2_evidence.py --check-report              # Task 2 appendix is current
```

Task 2 needs llama.cpp servers for Llama-3.1-8B-Instruct and Llama-Guard-3-1B on a local GPU
server (set `LLM_URL` and `GUARD_URL`, read in `chatbot/llm.py`); re-scoring from stored responses does not. Raw CSVs and the embedding model
are not committed; the fetch step is scripted. Lab 1's shipped weights *are* committed
under `lab1/outputs/models/` — they are the attack target, not a build artifact.

## Verification

Three gates; each exits non-zero on any mismatch. Run these before trusting a number.

```bash
.venv/bin/python scripts/verify_report_numbers.py   # gate 1, ~2 s
.venv/bin/python scripts/number_coverage.py         # gate 1b, ~3 s
.venv/bin/python scripts/verify_claims.py           # gate 2, ~2 min
```

- **Gate 1** (`verify_report_numbers.py`) parses the numbers printed in the report source
  (`report/report.tex` expanded with `report/sections/*.tex` by `scripts/report_source.py`) and
  checks each against `results/*.json` at the precision the report prints. It needs no model.
  It currently prints `1434 checks run, 0 failures`.
- **Gate 1b** (`number_coverage.py`) lists every numeric token in the report and its generated
  appendix, and fails while any is neither verified by gate 1 nor allowlisted as a
  non-measurement (`scripts/number_allowlist.json`). It must print `number coverage: 0 UNBACKED`.
- **Gate 2** (`verify_claims.py`) recomputes the derived claims from the data and the model
  weights, which catches a results JSON that is itself wrong. It loads torch and the data
  split, so run it where those are available.

`scripts/make_task2_evidence.py --check-report` additionally confirms the generated Task 2
appendix files match the stored red-team data.

The course chapters (`chapters/`) and the assignment brief (`docs/labs/`) are course material
and are not in the repository. The citations the report takes from them are stored as
sentences in `results/textbook_citations.json` and `results/brief_citations.json`; the gates
check the numbers against those sentences always, and check the sentences against the
chapter / brief only when those files are present. `models/task1c_recommended.joblib` is
git-ignored and not in the data snapshot; `scripts/detector_extras.py recommended` retrains it
deterministically (the stored run records identical file and weight hashes across two runs).

The rule for this repo: a number is verified when a script prints it, not when a person
or a model agrees with it. `report/REPRODUCING.md` ("Known limits and design choices") lists what the report leaves open.

## Status

Complete. All six tasks (1A/1B/1C, 2A/2B/2C) done, report within the 8-page limit, all
verification gates passing. Headline results: membership inference is
a negative result (blind control scores higher); one finite-difference gradient evades the
detector on 65% of flows at ε=0.25 and on all of them at ε=0.50; and the textbook defense — adversarial training — made the
detector *less* robust while leaving clean accuracy at 1.0000. On the copilot, a planted
knowledge-base document is a fully effective instruction channel, and fabricated
intelligence is restated as fact with no guard firing.

(For comparison, a 200-try random search still leaves detection at 0.743 at ε=0.50; see
the 1B table in the report.)
