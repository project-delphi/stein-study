# Stein/JEPA study

An unattended study run with `claude -p`. It tests Stein-discrepancy regularisers for
JEPA, V-JEPA-style temporal prediction and, optionally, VAEs, on real image data
(3DShapes, CIFAR-10, Moving MNIST). The output is a 15-page paper and a draft blog post.
`runner/` holds the runner; everything else in this folder is the study's workspace.

## Papers

- **[Study paper](paper/paper.pdf)** (14 pages): *Dimension, not noise: what limits nuisance leakage in
  regularised JEPAs.* Source: `paper/paper.qmd`. Start here for the results.
- **[Explainer](explainer/stein-jepa-aside.pdf)** (short note): why a Stein penalty can keep a JEPA from
  collapsing, and how it differs from a VAE's KL term. Source: `explainer/stein-jepa-aside.qmd`.
- [Blog post](blog/index.html), draft; and [`REPRODUCE.md`](REPRODUCE.md), the exact commands to reproduce
  every number.
Billing goes only to an Anthropic API key, never your claude.ai plan, capped at $90.

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
| `data/`, `.venv/`, `state/`, `logs/` | data, the Python environment and runner state; not in git |

`git log` has one commit per phase, plus one before each compute job.
