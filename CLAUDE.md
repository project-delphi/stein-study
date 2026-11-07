# Stein/JEPA study workspace

An unattended, multi-phase study runs here, driven by `runner/stein.sh`. The spec is
`PROMPT.md`. Its section "How this run works" binds every session, including after the
context is compacted. In short:

- Do only the phase the runner named, and keep `gates/GATE_<n>.md` current from the start.
- End a phase by writing one word to `gates/P<n>.status`: `complete`, `compute` (hand
  `jobs/P<n>.sh` to the runner), or `blocked`.
- Nobody can answer questions until morning: decide, and record the decision in the gate file.
- Python is `.venv/bin/python` only. A foreground command may run up to 45 minutes; never
  start a background job and poll it.
- Every number in any output comes from a script in this directory. Never fabricate
  findings, numbers, or citations.
- Stay inside this directory, don't modify `runner/` (the runner itself), and don't run
  git commands that write: the runner commits.
