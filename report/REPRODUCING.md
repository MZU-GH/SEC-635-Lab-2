# Reproducing every number in the report

All commands run from the repository root with `.venv/bin/python`. Python 3.12 is the
verified version (analysis host: 4 CPUs, 7.7 GiB RAM, no GPU). Wall-clock times quoted below
for Tasks 1A to 1C were measured on an 8-core VM; later steps carry no timing. The analysis
host is small, so run one heavy job (a model-loading script or an API server plus its client)
at a time.

## Step 0: data snapshot

The source tree excludes data by design (`data/raw/`, `data/processed/`, `data/chroma/`,
`models/` and `results/raw/` are gitignored). The data snapshot is a GitHub Release asset of
https://github.com/MZU-GH/SEC-635-Lab-2 (tag `data-snapshot-20261124`; the submission folder
carries the same file):

- URL: https://github.com/MZU-GH/SEC-635-Lab-2/releases/download/data-snapshot-20261124/sec635-lab2-data-20261124.tar.gz
- sha256: `cab06d805fd04e49aed0f5f4845b750d291bc0332e66584338d701a087d4817f`

Download, verify and extract it into the repository root (a clone, or the extracted source
archive):

```bash
curl -fL -o sec635-lab2-data-20261124.tar.gz \
  https://github.com/MZU-GH/SEC-635-Lab-2/releases/download/data-snapshot-20261124/sec635-lab2-data-20261124.tar.gz
echo "cab06d805fd04e49aed0f5f4845b750d291bc0332e66584338d701a087d4817f  sec635-lab2-data-20261124.tar.gz" | sha256sum -c - && tar xzf sec635-lab2-data-20261124.tar.gz
# python alternative to sha256sum:
python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sec635-lab2-data-20261124.tar.gz
```

From the submission folder instead: `tar xzf sec635-lab2-submission.tar.gz` (creates
`sec635-lab2/`), then `tar xzf sec635-lab2-data-20261124.tar.gz -C sec635-lab2`.

The data come from public sources: CICIDS-2018 (CIC / University of New Brunswick; the two
cleaned Lab 1 days), the MITRE ATT&CK enterprise bundle (`mitre/cti` on GitHub) and the CISA
Known Exploited Vulnerabilities (KEV) catalog (`known_exploited_vulnerabilities.json`,
fetched 2026-11-24). The embedding model is the public `sentence-transformers/all-MiniLM-L6-v2`.

It contains (from `tar tzf`):

- `data/processed/`: the two cleaned Lab 1 parquets, `cleaned_wednesday_14-02-2018.parquet`
  and `cleaned_thursday_01-03-2018.parquet`;
- `data/chroma/`: the Task 2 Chroma knowledge base (`chroma.sqlite3` plus one collection
  directory);
- `models/all-MiniLM-L6-v2/`: the staged 384-dim embedding model (config, tokenizer,
  `model.safetensors`);
- `models/task1c_defended.joblib`: the cached Task 1C defended model.

Gates 1, 1b and 2 need it (gate 1 reads `data/chroma/chroma.sqlite3`; gate 2 loads the
parquets and models). Without `data/chroma`, gate 1 fails with
"data/chroma not found: extract sec635-lab2-data-20261124.tar.gz into the repo root
(REPRODUCING.md step 0)". After the install below, check the snapshot with the gates:

```bash
.venv/bin/python scripts/verify_report_numbers.py   # gate 1
.venv/bin/python scripts/number_coverage.py         # gate 1b
.venv/bin/python -u scripts/verify_claims.py        # gate 2 (needs torch)
```

Rebuilding from the public sources instead is possible (sections 1, 9 and 10), but the CISA
KEV feed grows over time, so the KEV counts and the newest-entry date will differ from the
snapshot, and the report numbers that quote them will no longer match.

The course chapters (`chapters/`) and the assignment brief (a .docx) are course material
and are not in the repository (`.gitattributes` marks `chapters/`, `docs/` and `uploads/`
export-ignore, so neither the published tree nor the submission archive carries them).
Gate 1 checks the textbook figures the report cites (the 0.10 attack budget and the 0.616
smoothing radius) against the sentences stored in `results/textbook_citations.json`, and
against `chapters/course9.txt` and its sha256 only when that file is present;
`python3 scripts/make_textbook_citations.py --verify` runs the check alone and works without
`chapters/`, and `python3 scripts/make_textbook_citations.py` regenerates the JSON from the
chapter. The same holds for the brief: Gate 1b
checks the three brief citations in the report (the ATT&CK technique count, the KEV entry
count and the activity number the brief names) against the sentences stored in
`results/brief_citations.json`. When the brief is present at
`docs/labs/SEC635Lab2IntegratedLabII.docx` the gate checks against it instead and also
confirms the JSON still matches its sha256 and text; regenerate the JSON with
`python3 scripts/make_brief_citations.py`. The gate fails if the report cites a number that
differs from the stored sentence.

## Install the Python environment (do this right after Step 0)

There is no `uv` requirement. A stdlib venv plus pip works:

```bash
export https_proxy=http://bot.lab:3128 http_proxy=http://bot.lab:3128   # PyPI only
export NO_PROXY=.lab,10.0.1.0/24,localhost,127.0.0.1
python3 -m venv .venv
mkdir -p ~/pip-tmp                              # any disk with several GB free
TMPDIR=~/pip-tmp .venv/bin/pip install -r requirements.lock
```

`requirements.lock` is the output of `.venv/bin/pip freeze` and pins every installed package
to its exact version (the direct dependencies are scikit-learn, torch, pandas, pyarrow,
scipy, matplotlib, fastapi, uvicorn, httpx, requests, joblib, sentence-transformers and
chromadb). To regenerate it after changing the environment, run
`.venv/bin/pip freeze > requirements.lock`.

Two details matter. The PyPI `torch` wheel pulls the CUDA libraries (several GB), and pip
unpacks them under `$TMPDIR`; on a small `/tmp` tmpfs the install fails, so point `TMPDIR`
at a disk with space (15 GB free on `/home` was enough). `scikit-learn` is pinned to 1.9.1
because the Lab 1 joblib files were saved with that version.

PDF build: `tectonic report.tex` or `latexmk -pdf report.tex`, run inside `report/`. The
report is split: `report/report.tex` holds the title block and `\input{sections/NN_name}`
lines, `report/sections/*.tex` the body, and `report/style.tex` the layout. The build
expands them itself; the gates expand them with `scripts/report_source.py`, which also fails
on a section file that `report.tex` does not `\input` and on an `\input` target that is
missing.

## 1. Data

```bash
# CICIDS-2018 raw CSVs go in data/raw/ (not committed, not present in a fresh clone).

# Task 2 knowledge-base sources (ATT&CK via squid, KEV via webget because cisa.gov is
# not on the squid allowlist). Prints the technique/KEV counts the report quotes.
bash scripts/fetch_kb_data.sh
```

Lab 1's cleaned parquets live in `data/processed/`. `detector/dataset.py` rebuilds the
exact temporal split from them; `lab1/src/task1_supervised.py` regenerates the Lab 1
models and figures if you want to rebuild from scratch.

## 2. Verify the target is really Lab 1's model

```bash
.venv/bin/python detector/torch_model.py
# max_abs_prob_diff: ~5.8e-08, passes: True, identical accuracy on 5k held-out rows
```

Lab 1's own reported metrics are reproduced by scoring `lab1/outputs/models/
task1_wednesday_mlp_neuralnet_temporal.joblib` against `detector/dataset.py`'s split:
precision 0.999992, recall 1.000000, F1 0.999996, ROC-AUC 1.000000.

## 3. Task 1 attack surface: the detector API

Tasks 1A and 1B attack the detector through HTTP only. Start the API (it loads just the
feature names and the model, not the full training split) and leave it running:

```bash
.venv/bin/uvicorn detector.api:app --host 127.0.0.1 --port 8055
```

`SEC635_MODEL=<path to a joblib pipeline>` serves a different model (default: the Lab 1
Wednesday MLP). `GET /stats` returns the server-side request and row counters, which the
attacks use to report query budgets measured at the server rather than by the client.

## 4. Task 1A - membership inference

Two APIs: one serving the Wednesday model (section 3) and one serving the Thursday model.
The port for the second is arbitrary.

```bash
SEC635_MODEL=lab1/outputs/models/task1_thursday_mlp_neuralnet_temporal.joblib \
    .venv/bin/uvicorn detector.api:app --host 127.0.0.1 --port 8057 &
.venv/bin/python -u detector/membership.py \
    --url-wednesday http://127.0.0.1:8055 --url-thursday http://127.0.0.1:8057
# -> results/task1a_membership.json (use --out to write elsewhere)
```

Without the two URLs the target arms are scored in-process. The HTTP client sends rows in
chunks of 40000 (`HttpModel`); that chunk size is required for a bit-exact AUC, because
the batch size changes float summation order. The output JSON has a `transport` key.

## 5. Task 1B - evasion

```bash
.venv/bin/python -u detector/evasion.py --via-http --base-url http://127.0.0.1:8055
# -> results/task1b_evasion_wednesday.json (use --out to write elsewhere)
```

Random search and FD-FGSM go over HTTP; PGD stays in-process because it is white-box.
`--limit-flows N` and `--eps E [E ...]` shrink the run for timing or smoke tests only.
Without `--via-http` everything runs in-process. The stored JSON records `transport` and a
`superseded_cells` list (the random-search eps 0.25 / 0.5 cells whose earlier values were
not reproducible; the current code is deterministic and HTTP equals in-process).

## 6. Task 1C - defense (~25 min) and ablation (~40 min)

```bash
.venv/bin/python detector/defend.py                 # full: all 731k rows, 2 rounds
.venv/bin/python detector/defend.py --smoke         # 40k random rows, quick check
.venv/bin/python detector/defend.py --skip-smoothing
.venv/bin/python detector/defend.py --out /tmp/curve_after.json   # evaluation only
```

`defend.py` caches its refit at `models/task1c_defended.joblib` so re-running the
evaluation does not retrain (~6 min per round). Delete that file to force a retrain.
With `--out`, it re-evaluates `curve_after` from the cached model, writes only that
payload to the given path, and never retrains or touches `results/`.

Task 1A must also be re-run against the defended model. Serve it on its own port, then
run `membership_after.py` (it reuses the original 1A member / non-member positions):

```bash
SEC635_MODEL=models/task1c_defended.joblib \
    .venv/bin/uvicorn detector.api:app --host 127.0.0.1 --port 8056 &
.venv/bin/python -u detector/membership_after.py --base-url http://127.0.0.1:8056
# -> results/task1c_membership_after.json (use --out to write elsewhere)
```

The two ablations that produced the 1C findings are separate scripts, because each one
answers a question the other cannot:

```bash
.venv/bin/python detector/ablate_1c.py           # ~34 min -> results/task1c_ablation.json
.venv/bin/python detector/ablate_1c_rounds.py    # ~20 min -> results/task1c_ablation_rounds.json
```

`ablate_1c.py` runs four configs (full 2 rounds, full 1 round, 100k 2 rounds, full 2
rounds with the constant-in-train features frozen) on the same 300 flows as Task 1B, and
checkpoints to JSON after each config so an interrupted run keeps its results.
`ablate_1c_rounds.py` runs the fifth config (full data, 1 round, constant features
frozen: the last row of the report's ablation table) and measures the per-round crossing
fraction used in the Cause 2 paragraph.

## 7. Attacker budget, recommendation thresholds and audit checks

```bash
.venv/bin/python scripts/attacker_budget.py   # -> results/task1_attacker_budget.json
.venv/bin/python scripts/recommendation_controls.py   # -> results/task5_recommendation_controls.json (the section 5 alert threshold; stdlib, reads the budget and Task 2 summary JSONs)
```

`scripts/audit/` holds the comparison tools used to check the HTTP re-runs against the
stored results. Each takes file paths as positional arguments (see the header comment of
each file):

| script | usage |
| --- | --- |
| `compare_1a.py` | `compare_1a.py NEW.json SERVER_ROWS_BASELINE_WED [OLD.json]`: cell-by-cell 1A comparison; accuracy, CI, AUC and counts must be exactly equal |
| `compare_1b.py` | `compare_1b.py NEW.json SERVER_ROWS_BASELINE [OLD.json]`: 1B cells, plus server-counted vs client-counted rows |
| `compare_1c.py` | `compare_1c.py NEW_EVAL.json [STORED_task1c_defense.json]`: compare a `defend.py --out` file with the stored `curve_after` |
| `diag_random.py` | `diag_random.py HTTP_RESULT.json [STORED.json]`: in-process random search twice vs stored vs HTTP (shows the code is deterministic) |
| `diag_scores.py` | `diag_scores.py BASE_URL`: single-row vs batch, in-process vs HTTP score differences (needs the Wednesday-model API) |
| `install_1b.py` | `install_1b.py NEW_HTTP.json OLD_STORED.json OUT.json`: merges a reproduced 1B curve into the stored file and records `superseded_cells` |

These scripts have no `--help`; running one without arguments prints a usage error.

## 8. LLM serving on a local GPU server

Task 2 needs two llama.cpp servers on a GPU machine reachable from the lab network (the
machine running the repo has no GPU and cannot host an 8B model). Each exposes an
HTTP chat-completions API (`/v1/chat/completions`) and runs as a systemd unit:

| unit | port | model |
| --- | --- | --- |
| `llm8b.service` | 11434 | Meta-Llama-3.1-8B-Instruct Q4_K_M |
| `guard1b.service` | 11435 | Llama-Guard-3-1B Q4_K_M |

The endpoint base URLs are read from the environment by `chatbot/llm.py`; the defaults are
`http://127.0.0.1:11434` (LLM) and `http://127.0.0.1:11435` (guard), and the
`/v1/chat/completions` path is appended by the code. Set both before any Task 2 script that calls a model. To re-run Task 2 against a GPU server:

```bash
export LLM_URL=http://<gpu-server>:11434
export GUARD_URL=http://<gpu-server>:11435
.venv/bin/python chatbot/run_redteam.py --runs 10
```

Model files are GGUF, in a models directory on that server.
The weights come from Hugging Face; the LFS CDN is not on the proxy allowlist, so
`lemonade pull` / `huggingface-cli` fail with CURL 56. Fetch the files with `webget` on a
machine that can reach the CDN and copy them over. Verified sha256:

```
cbc0dba2676b77ff3a69b0ed359f1b99710d7083dffcab8018187cb7090bd777  Llama-Guard-3-1B.Q4_K_M.gguf
7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Re-scoring from the stored responses (section 12, `scripts/rescore_task2.py`) does not need these servers.

## 9. Embedding model

`models/all-MiniLM-L6-v2/` (384-dim) is also staged by `webget` for the same CDN reason;
`sentence-transformers` cannot fetch it itself. Loaded with `local_files_only=True`.

## 10. Task 2: knowledge base, red-team suite, evidence

```bash
.venv/bin/python chatbot/kb.py     # ~3 min -> data/chroma/, 2498 docs (706 ATT&CK + 1792 KEV)
.venv/bin/python chatbot/run_redteam.py --runs 10   # ~25 min -> results/task2_redteam.json
.venv/bin/python chatbot/run_redteam.py --only 2A-kb-override --runs 3   # single attack
```

The red-team run needs the servers of section 8. Everything after it uses stored data
only:

```bash
python3 scripts/remap_owasp.py                  # only if attacks.py categories change: rewrites the stored owasp field only
python3 scripts/make_task2_evidence.py     # stdlib only; writes report/generated/payloads.tex,
                                           # report/generated/owasp_rollup.tex, results/task2_summary.json
python3 scripts/make_task2_evidence.py --check-report   # read-only check, see below
.venv/bin/python scripts/probe_retrieval.py             # F7 probe -> results/task2_retrieval_probe.json
```

`make_task2_evidence.py` flags: `--redteam FILE` (default `results/task2_redteam.json`),
`--out-dir DIR` (default `report/generated`), `--summary FILE`, `--report-tex FILE` (the
report source is read through `scripts/report_source.py`),
`--payload-limit N` (default 0: every payload is printed in full), `--excerpt-len N`
(default 60 characters of each response excerpt), and `--check-report`. `payloads.tex`
also lists the five normal questions with their per-question refusal and hedge counts. With
`--check-report` nothing is written: it verifies that `report/report.tex` contains
`\input{generated/owasp_rollup}` and `\input{generated/payloads}` and that both generated
files on disk equal a fresh in-memory regeneration; exit code 1 on any failure.

`probe_retrieval.py` has no command-line flags and runs on start. It makes no LLM calls,
copies the Chroma collection to a temp dir so `data/chroma` is untouched, and loads the
embedding model, so run it alone.

## 11. Figures and report

```bash
.venv/bin/python results/make_figures.py    # -> results/figures/*.png
# Fig. 1 (task1b_evasion.png) has two panels: 1B (task1b_evasion_wednesday.json) and the
# same attacks after adversarial training (curve_after in task1c_defense.json).
cd report && tectonic report.tex            # or: latexmk -pdf report.tex
```

## 12. Verification gates (run these before trusting any number)

Three scripts, each exits non-zero on a mismatch, so they are gates rather than reports.

```bash
.venv/bin/python scripts/verify_report_numbers.py   # gate 1, ~2 s, no model loading
.venv/bin/python scripts/number_coverage.py         # gate 1b, ~3 s: every number in the report is covered
.venv/bin/python -u scripts/verify_claims.py        # gate 2, ~2 min, loads the split + models
.venv/bin/python scripts/make_task2_evidence.py --check-report   # generated appendix is current
```

`verify_report_numbers.py` (currently prints `1434 checks run, 0 failures`) parses each figure out of the report source (`report/report.tex` expanded with `report/sections/*.tex` by `scripts/report_source.py`)
and compares it to `results/*.json` at the precision the report prints. It catches a
report that drifted from its data. It also checks that every finding entry of report section 6 names an existing script
(`scripts/`, `detector/`, `chatbot/`) and an existing `results/` JSON.

Environment overrides for negative tests: `GATE_TEX`, `GATE_ROLLUP`, `GATE_RESULTS` and
`GATE_PAYLOADS` point the first gate at a copy of `report.tex`, of the two generated files,
or of `results/`. Corrupt the copy, never the real file, and the gate must fail.

### Gate 1b: number coverage audit

```bash
.venv/bin/python scripts/number_coverage.py          # ~3 s, stdlib only; exit 0 = no UNBACKED number
```

Gate 1 checks the numbers someone thought to check; gate 1b proves nothing was missed (it
currently prints `number coverage: 0 UNBACKED`). It
extracts every numeric token (integers, decimals, percentages, `a\times10^{-b}`, `a/b`,
interval ends, `{,}` thousands) from the expanded report source and the `\input`'ed generated files, runs
gate 1 with `--dump-coverage` (each helper that verifies a number records the source lines
it matched) and classifies each token as COVERED (a gate check verified that literal on that
line; one check covers one occurrence), ALLOWLISTED (a non-measurement matched by a tightly
anchored regex in `scripts/number_allowlist.json`: layout, refs, labels, names, IDs; each
entry has a written reason and some carry a verifier such as "the file exists" or "the
brief contains it") or UNBACKED. Tokens in `payloads.tex` / `owasp_rollup.tex` count as
covered only while `scripts/make_task2_evidence.py --check-report` confirms the files equal
a regeneration from the stored data. Configuration values (hidden-layer sizes, sampling
temperature, top-k, `max_iter`, the eps grid, rounds, sigma, n) are checked against the code
or the results JSON instead of being allowlisted. `scripts/report_extras.py` (needs torch:
run it on the GPU server) stores the numbers no other results file held (`results/report_extras.json`).
`GATE_TEX`, `GATE_ROLLUP`, `GATE_PAYLOADS`, `GATE_RESULTS` and `NUMBER_ALLOWLIST` redirect it
for negative tests; `--baseline` shows what the original literal checks alone covered.

`verify_claims.py` (gate 2; prints its claim count and `0 failures`; needs torch, so run it on the GPU server) recomputes the derived claims from the raw matrices, the
feature classifier and the two model pipelines: the 52/25 mutable split, the 10
constant-in-train features (5 of them mutable), the standardized move the defended
model sees under the real attack box (measured through its own scaler; not
$\varepsilon$ on the five constant features), that the undefended model has exactly zero logit sensitivity to those features
while the defended one has live weights, that a raw-unit smoothing sigma is ~1e-3
standardized, the shared eval-set sha256, and Lab~1's four metrics to 6 dp. It catches a
JSON that is itself wrong, which the first gate cannot.

`scripts/rescore_task2.py` re-derives the Task 2 success counts from the stored response
text using the current `chatbot/attacks.py` criteria, so a scorer fix does not need the
GPU. It is how the sentinel-refusal false positive was corrected without re-running the
model.

The gates exist because reading a number is not checking it: two read-throughs of this
repo each reported a "mismatch" that did not exist, and the real errors (a role-play count of
3 instead of 2, a scorer that scored a refusal as a leak) were only found by computing.

## 13. Additional result scripts (2026-11-25)

```bash
.venv/bin/python scripts/report_extras.py         # results/report_extras.json (torch + full split: GPU server only)
.venv/bin/python scripts/scale_effect.py          # results/task1c_scale_effect.json (torch + full split: GPU server only)
.venv/bin/python scripts/scale_effect.py --augment # adds l2_extent, evasion_reference, degenerate_training_value to that JSON from stored JSONs + the Lab 1 scaler (seconds, any machine; run after the line above)
.venv/bin/python scripts/zero_rate_bounds.py      # results/task2_zero_rate_bounds.json (stdlib + scipy)
.venv/bin/python scripts/raw_shift_eps.py         # results/task1b_raw_shift_eps010.json (loads the Lab 1 model)
.venv/bin/python detector/smoothing_eval.py       # results/task1c_smoothing_detection.json (loads models; slow)
.venv/bin/python scripts/noise_check.py --recompute-perflow   # retrains the 1-round frozen defense; writes results/noise_check_perflow.json
.venv/bin/python scripts/noise_check.py           # exact McNemar + Wilson/Clopper-Pearson; writes results/noise_check.json
```

`noise_check.py` shows that the FD-FGSM eps=0.15 gap between the undefended model and the
recommended defense is not noise (exact McNemar), which is why the report does not call it that.

Results added with the split report (each script's header states its inputs):

```bash
.venv/bin/python -u scripts/ops_measurements.py   # results/ops_measurements.json: detector latency, confidence bands, copilot latency (GPU machine: loads the models and the full Wednesday parquet; timings vary between runs, the bands and traffic rate do not)
.venv/bin/python -u scripts/detector_extras.py thursday    # one block per process; each writes a part file
.venv/bin/python -u scripts/detector_extras.py benign
.venv/bin/python -u scripts/detector_extras.py recommended   # retrains models/task1c_recommended.joblib
.venv/bin/python -u scripts/detector_extras.py assemble      # -> results/task1_extras.json
python3 scripts/make_textbook_citations.py         # -> results/textbook_citations.json (needs chapters/course9.txt)
python3 scripts/make_brief_citations.py            # -> results/brief_citations.json (needs the brief .docx)
.venv/bin/python scripts/split_audit.py            # -> results/split_audit.json (reads the two Lab 1 parquets; --verify recomputes and compares; the submitted Lab 1 repo is read via LAB1_REPO, default ../../lab1, and is optional for --verify)
.venv/bin/python scripts/make_submission.py        # source archive + data copy + checksums (see its header)
```

`split_audit.py` shows that the Lab 2 detector split sorts by the raw 12-hour timestamp (afternoon flows carry 01:00-05:59), so its test window is the late morning and many training rows are later than it, and that the submitted Lab 1's clock-repaired split has a test window holding only SSH brute force; gate 1 runs `--verify` and fails if the stored file is stale.

`detector_extras.py` (model-loading, run it on the GPU machine) writes
`results/task1_extras.json`: the Thursday-model evasion curves, the benign-to-attack
direction, and the one-round recommended defense (membership re-run and randomized
smoothing). Its `harness_check` block re-runs the Wednesday 1B flow through the same driver
and compares every cell with the stored 1B file; `assemble` merges two runs and records the
input hashes. `models/task1c_recommended.joblib` is git-ignored (`*.joblib`; only
`models/task1c_defended.joblib` is force-added and shipped in the snapshot), so a fresh
clone does not have it. The `recommended` block retrains it with the same code as
`detector/ablate_1c_rounds.py` and checks it against `results/task1c_ablation_rounds.json`;
the stored `determinism` entry for it records equal file and weight sha256 across two runs.
`make_task2_evidence.py` and `results/make_figures.py` were also updated for the new report
and figures.

## 14. Known limits and design choices

These are stated in the report; they are listed here so a reproduction does not mistake them
for bugs.

- The brief's ATT&CK technique count is older than the live feed (the live count excludes
  revoked and deprecated techniques); the report states both counts. The KEV catalog grows daily, hence the dated snapshot.
- The brief names Ollama; Task 2 serves the same two models (Llama-3.1-8B-Instruct,
  Llama-Guard-3-1B) through llama.cpp on a GPU server because the analysis host has no GPU.
  Model identity is unchanged, only the server differs.
- Task 1C: adversarial training made the detector less robust to the finite-difference
  attack at the smallest budgets for two independent reasons. The first (live weights on
  features that are constant in training, plus the rescaled attack box) is measured; the
  second is not explained, and the report says so. Do not add a mechanism for it.
- The Task 1C scale effect does not cancel on the five constant-in-train features under the
  real attack box (`scripts/scale_effect.py`).
- Task 2 finding F3 (a fabricated ATT&CK technique restated as fact) is not mitigated by the
  guardrails, and F7 (dense retrieval missing an exact CVE ID) has two defects; both are
  reported as open.
- `results/task2_summary.json` records the feed fetch date as provenance written when the
  snapshot was built; the raw feeds themselves are not in the snapshot.

## What is not committed

`data/raw/`, `data/processed/`, `data/chroma/`, `models/` and `results/raw/` are gitignored
(hundreds of MB, all re-derivable from the steps above). Committed results are the JSON
summaries and the PNG figures the report cites.
