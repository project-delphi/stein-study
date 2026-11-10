# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Stein/JEPA study workspace

An unattended, multi-phase study runs here, driven by `runner/stein.sh`. The spec is
`PROMPT.md` (a copy of `runner/stein_paper_prompt.md`, refreshed on every start, so edit the
original). Its section "How this run works" binds every session, including after the
context is compacted. In short:

- Do only the phase the runner named, and keep `gates/GATE_<n>.md` current from the start.
- End a phase by writing one word to `gates/P<n>.status`: `complete`, `compute` (hand
  `jobs/P<n>.sh` to the runner), or `blocked`.
- Nobody can answer questions until morning: decide, and record the decision in the gate file.
- Python is `.venv/bin/python` only. A foreground command may run up to 45 minutes; never
  start a background job and poll it.
- Every number in any output comes from a script in this directory. Never fabricate
  findings, numbers, or citations.
- Stay inside this directory, don't modify `runner/` or the top-level `README.md` (the
  runner and its guide), and don't run git commands that write: the runner commits. The
  study's own reproduction guide is `REPRODUCE.md`.

## State of the study

All phases P0–P4 are `complete` (`gates/P*.status`); `FINAL.md` summarises the outcome. Work in an
interactive session is therefore post-hoc edits to the paper, blog, or explainer, not a new phase.
Keep reported numbers traceable to `results/` via the generating scripts.

## Study 2 (follow-up: theory and its test)

`runner/stein2.sh` runs a second study through the same runner, with the spec
`study2/PROMPT.md` (a copy of `runner/stein2_prompt.md`). When the runner's message is that spec,
every rule above holds with these paths: gates are `study2/gates/GATE_<n>.md` and
`study2/gates/P<n>.status`, jobs are `study2/jobs/P<n>.sh`, outputs go to `theory/`, `results2/`,
`paper2/`, `REPRODUCE2.md` and `FINAL2.md`. Study 1's `gates/`, `results/`, `paper/`, `blog/`,
`FINAL.md` and `REPRODUCE.md` are then a frozen record: read them, never write them. New config keys
must stay out of `cfg_hash` at their default value, so the cached `runs/` remain valid.

## Commands

All from the repo root. Full pipeline, timings and seeds are in `REPRODUCE.md`.

```bash
.venv/bin/python -m pytest -q tests                              # all tests
.venv/bin/python -m pytest -q tests/test_losses.py -k <name>     # single test
.venv/bin/python -m stein.validate                               # validation gates -> results/validation.json
.venv/bin/python -m stein.train configs/base.yaml key=val ...    # one run (tier/arm/lam/seed/... see DEFAULTS in train.py)
.venv/bin/python -m stein.sweep <grid> --workers 6 --out results/<x>.parquet   # grid in stein/grids.py: tune | heldout | heldout_h5 | lam_sens
.venv/bin/python -W ignore -m stein.analyze                      # CIs + contrasts -> results/
.venv/bin/python -W ignore -m stein.report                       # figures, tables, numbers -> paper/, blog/
quarto render                                                    # paper/paper.pdf (typst) + blog/index.html
```

Cheapest end-to-end check: the H5 reproduction in `REPRODUCE.md` §5 (~15 min).

## Architecture

- **Run cache.** `stein/train.py` hashes the resolved config (`cfg_hash`, seed included) and writes
  `runs/<hash>/result.json`; finished runs are skipped, so every sweep is interruptible and idempotent.
  `STEIN_RUNS=<dir>` redirects the cache. The device is not part of the hash. Changing any default in
  `DEFAULTS` or the grids changes hashes and silently invalidates cached runs.
- **Grids** (`stein/grids.py`) define every experiment (tuning on seeds 100–101, held-out on seeds 1–5, λ
  sensitivity); `stein/sweep.py` runs them across workers. Tuned levels flow
  `tune` → `select_tuning.py` → `results/tuning_selected.json` → held-out grids.
- **Tiers** (`tier=` in config): `shapes` (3DShapes JEPA, core), `cifar`, `temporal` (V-JEPA-style clips from
  3DShapes), `vae`. Arms are the regularisers in `stein/losses.py`; SIGReg comes from the pinned external
  `lejepa` package.
- **Results → documents.** `analyze.py` → `results/*.parquet`/`analysis.md`; `report.py` writes
  `paper/_generated/` (numbers.yml, tab_*.md), `paper/figures/`, `blog/data/`, `blog/figures/`. The `.qmd`
  files read these and **execute no Python**; never hand-type a number into a `.qmd`.
- **Literature.** `notes/<citekey>.md` are verified notes written by the `lit-reviewer` sub-agent
  (`.claude/agents/`); `notes/_work/build_bib.py` regenerates `paper/refs.bib` from them.
- **Not in git:** `data/`, `.venv/`, `logs/`, `state/`, `.quarto/`, `*.h5/*.npy`. `runs/` and `results/` are
  the cache and outputs; check `.gitignore` before assuming a file is tracked.
