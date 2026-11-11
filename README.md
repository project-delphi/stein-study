# Stein/JEPA study

An unattended study run with `claude -p`. It tests Stein-discrepancy regularisers for
JEPA, V-JEPA-style temporal prediction and, optionally, VAEs, on real image data
(3DShapes, CIFAR-10, Moving MNIST). The output is a paper and a draft blog post. A
follow-up study, run the same way, builds a linear theory of what the first study found and
tests it; it has its own paper.
`runner/` holds the runner; everything else in this folder is the study's workspace.
Billing goes only to an Anthropic API key, never your claude.ai plan, capped at $90 (the
follow-up: $60).

## Papers

The two studies are kept apart, each with its own paper, results and reproduction guide. The second
starts from findings the first left unexplained. The code (`stein/`), the run cache (`runs/`) and the
literature notes (`notes/`) are shared. Study 2's code changes leave study 1's run hashes and results
unchanged: a study-1 run redone under the new code gives the same metrics (`results2/validation2.json`).

### Study 1: Stein regularisers for JEPAs

- **[Study paper](paper/paper.pdf)** (14 pages): *Dimension, not noise: what limits nuisance leakage in
  regularised JEPAs.* Source: `paper/paper.qmd`. Start here for the results.
- **[Explainer](explainer/stein-jepa-aside.pdf)** (short note): why a Stein penalty can keep a JEPA from
  collapsing, and how it differs from a VAE's KL term. Source: `explainer/stein-jepa-aside.qmd`.
- [Blog post](blog/index.html), draft; [`FINAL.md`](FINAL.md), the outcome; and
  [`REPRODUCE.md`](REPRODUCE.md), the exact commands to reproduce every number.

### Study 2: a linear theory of JEPA regularisers, and its test

- **[Follow-up paper](paper2/paper.pdf)** (12 pages): *Why distributional pressure buys nuisance: a linear
  theory of JEPA regularisers and its test.* Study 1 found that leakage rose with λ, that noise on the
  predictor input raised it, and that a matched prior did not improve identifiability. This paper derives a
  linear theory of these findings and tests the predictions it registered before training. Source:
  `paper2/paper.qmd`.
- [`theory/theory.md`](theory/theory.md), the statements and proofs, and
  [`theory/predictions.md`](theory/predictions.md), the registered predictions.
- [`FINAL2.md`](FINAL2.md), the outcome of each prediction; and [`REPRODUCE2.md`](REPRODUCE2.md), the exact
  commands to reproduce every number.

## What to do

1. **Once: save the API key.** In [console.anthropic.com](https://console.anthropic.com),
   create a key in the organisation that holds the $100 credit, and copy it. Then:

   ```bash
   mkdir -p ~/.config/stein
   pbpaste > ~/.config/stein/anthropic_api_key && chmod 600 ~/.config/stein/anthropic_api_key
   ```

2. **Check everything:**

   ```bash
   cd ~/Code/stein-study/runner
   ./stein.sh preflight
   ```

   It should end with `preflight passed`. The venv and data are already in place, so it
   takes seconds and one test call that costs well under a cent.

3. **Optional daytime trial:** `STOP_AFTER=0 ./stein.sh start` runs only the literature
   review (about $12), then pauses. Read `gates/GATE_0.md` before going further.

4. **Tonight:** plug in, leave the lid open and Wi-Fi on, then run `./stein.sh start`. It
   returns at once, and the terminal can be closed.

5. **While it runs**, from `runner/`:

   | Command | What it does |
   |---|---|
   | `./stein.sh watch` | follows the live log; Ctrl-C stops watching, not the study |
   | `./stein.sh status` | shows each phase, the spend, and the latest activity |
   | `./stein.sh stop` | stops the study; `./stein.sh start` resumes it |

6. **In the morning**, run `./stein.sh status`. Then read `gates/GATE_0.md` onwards, and
   `FINAL.md` once P4 has finished. It will probably still be running: P2 alone hands
   up to 6 hours of compute to the machine.

If the Mac restarts (macOS updates can), run `./stein.sh start` again. It resumes where it
stopped.

The follow-up study runs the same way with `./stein2.sh` in place of `./stein.sh`; its
gates are in `study2/gates/`. See the "Study 2" section of `runner/README.md`.

## What is here

| Path | What it holds |
|---|---|
| `runner/` | the runner, the study spec (`stein_paper_prompt.md`) and the literature sub-agent. `runner/README.md` covers how it behaves unattended, money, and settings |
| `PROMPT.md` | the spec the agent reads: a copy of `runner/stein_paper_prompt.md`, refreshed on every start, so edit the original |
| `CLAUDE.md` | the standing rules every agent session loads |
| `gates/` | one gate file per phase (done, decisions, problems, next), and `P<n>.status` |
| `logs/progress.log` | the readable log; the runner's own lines start with `==>` |
| `notes/`, `stein/`, `results/`, `jobs/` | literature notes, code, results, and compute jobs, as the phases write them |
| `paper/`, `blog/`, `REPRODUCE.md` | the outputs, from P3 |
| `study2/` | study 2's spec (`PROMPT.md`), gates, compute job and P4 review reports |
| `theory/`, `results2/`, `paper2/`, `REPRODUCE2.md`, `FINAL2.md` | study 2's theory and checks, results, paper and guides |
| `data/`, `.venv/`, `state/`, `logs/` | data, the Python environment and runner state; not in git |

`git log` has one commit per phase, plus one before each compute job.
