# Stein/JEPA study: the overnight runner

`stein.sh` runs the study in `stein_paper_prompt.md` from start to finish with `claude -p`,
unattended and billed to an Anthropic API key. It runs P0 literature review → P1 code and
validation on real data (3DShapes, CIFAR-10, Moving MNIST) → P2 sweep → P3 paper and blog
draft → P4 independent review.

The runner lives in `runner/` inside the study's workspace, `~/Code/stein-study`, a git
repo of its own. The agent works in the workspace around it and leaves `runner/` alone.
None of it is in the blog repo: that repo's hooks, `AGENTS.md` and Quarto render would act
on the agent's files. The blog post moves into the blog once the study is done.

| File | What it is |
|---|---|
| `stein.sh` | the runner |
| `stein_paper_prompt.md` | the study spec; copied to `../PROMPT.md` on every `start`, so edit it here |
| `lit-reviewer.md` | the Sonnet sub-agent P0 uses, one call per paper |

## Once, by day

1. In [console.anthropic.com](https://console.anthropic.com), create an API key in the
   organisation that holds the $100 credit, and copy it. Then:

   ```bash
   mkdir -p ~/.config/stein
   pbpaste > ~/.config/stein/anthropic_api_key && chmod 600 ~/.config/stein/anthropic_api_key
   ```

2. `cd ~/Code/stein-study/runner && ./stein.sh preflight`. It checks the workspace, the
   Python 3.12 venv (torch, the scientific stack, `lejepa` from GitHub at a pinned commit)
   and about 450 MB of data, all already in place. Then it checks the key with a free API call
   and makes one Haiku test call through the study's own flags (well under a cent). Rerun
   it until it ends with `preflight passed`.

Optional dry run: `STOP_AFTER=0 ./stein.sh start` does only the literature review
(about $12) and pauses. Read `~/Code/stein-study/gates/GATE_0.md`, and if it looks right,
start the rest at night.

## Tonight

Plugged in, lid open, online. From `~/Code/stein-study/runner`:

```bash
./stein.sh start     # returns at once; closing the terminal is fine
./stein.sh watch     # live progress; Ctrl-C stops watching, not the study
./stein.sh status    # phases, spend, latest activity
./stein.sh stop      # stop; `./stein.sh start` resumes where it stopped
```

`start` also resumes after a crash or a reboot (macOS updates can restart the machine
overnight). Finished phases are skipped, and an interrupted one picks up from its gate file.

Expect it to run past morning: P2 alone hands up to 6 hours of compute to the machine.

## In the morning

Everything is in `~/Code/stein-study`:

- `logs/progress.log` has one line per agent step. The runner's own lines start with `==>`:
  phase starts and ends, spend, compute progress, retries.
- `gates/GATE_<n>.md` says what each phase did, what it decided on its own, and what went
  wrong. Read these first.
- `FINAL.md` appears once P4 finishes; `results/`, `paper/` and `blog/` hold the outputs.
- `git log` has one commit per phase, plus one before each compute job.

## Money

- **Only the API key pays.** The runner refuses to start without it and checks it first.
  The first line of every run says how that run is billed, and any run not billed to the key,
  for example to your claude.ai Max plan, is killed at once and the study stops.
- **Caps.** Phase plans are $12 / $24 / $18 / $16 / $10, and unspent money rolls forward.
  The whole study is capped at $90 (`TOTAL_CAP`), which leaves room for the last call to
  overshoot. A phase that runs out of money gets one short wrap-up run to finish its gate
  file honestly, and the study moves on.
- Spend comes from the CLI's own cost report. A run killed before reporting is booked at an
  estimate. A bad key or an empty balance stops the study with a notification.

## How it behaves unattended

- Each phase starts a fresh session and ends by writing `complete`, `compute` or `blocked`
  to `gates/P<n>.status`. Gates are approved automatically, so the spec makes P1's
  validation gates binding: an arm that fails them is dropped, not swept.
- The agent never waits for an answer. It decides, and records the decision in the gate file.
- Long compute is handed to the machine. The agent writes `jobs/P<n>.sh`, and the runner runs
  it within the phase's allowance (P1: 1 h, P2: 6 h), which costs no tokens, then resumes the
  agent with the result.
- If the network drops, it waits for it to return. Failed runs retry with backoff
  (2 → 30 min), and 8 failures in a row stop the study. `caffeinate` keeps the Mac awake
  while it runs.

## Settings

All are environment variables, set on the `start` command line:

| Variable | Default | |
|---|---|---|
| `TOTAL_CAP` | `90` | hard cap on the study's spend, USD |
| `STOP_AFTER` | `4` | pause once this phase is done |
| `COMPUTE_HOURS_P2` | `6` | wall clock for P2's compute jobs (`COMPUTE_HOURS_P1`: 1) |
| `STEIN_KEY_FILE` | `~/.config/stein/anthropic_api_key` | where the key lives |
| `STEIN_STYLE` | `~/Code/github.com/ml-blog/STYLE.md` | the blog's style guide, for P3's blog draft |
| `STEIN_WORKDIR` | the parent of `runner/` | the workspace |
| `MODEL`, `EFFORT` | `opus`, `high` | the main agent |
| `ALLOW_BATTERY` | unset | `1` starts even when not plugged in |

To redo a phase, delete `gates/P<n>.status` in the workspace, plus `gates/GATE_<n>.md`
for a clean slate, and run `./stein.sh start`.
