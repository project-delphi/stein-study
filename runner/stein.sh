#!/usr/bin/env bash
# Unattended runner for the Stein/JEPA study: phases P0-P4 back to back with `claude -p`,
# billed to an Anthropic API key, with long compute jobs handed to the machine.
#
#   ./stein.sh preflight   check everything; build the workspace, venv and data (safe to rerun)
#   ./stein.sh start       start or resume the study in the background, then return
#   ./stein.sh watch       follow the progress log (Ctrl-C stops watching, not the study)
#   ./stein.sh status      phases, spend, and the latest activity
#   ./stein.sh stop        stop the study; `start` resumes it
#
# It lives in runner/ inside the study's own workspace, which is its parent directory.
# Settings are environment variables; README.md lists them.
#
# Written for macOS's bash 3.2. There is deliberately no `set -e`: one failed command must
# not end an overnight run, so every step checks its own result.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
W="${STEIN_WORKDIR:-$(dirname "$HERE")}"
STYLE_SRC="${STEIN_STYLE:-$HOME/Code/github.com/ml-blog/STYLE.md}"   # the blog's style guide
KEY_FILE="${STEIN_KEY_FILE:-$HOME/.config/stein/anthropic_api_key}"
PROMPT_SRC="${STEIN_PROMPT:-$HERE/stein_paper_prompt.md}"
TOTAL_CAP="${TOTAL_CAP:-90}"            # hard cap on the whole study, USD
STOP_AFTER="${STOP_AFTER:-4}"           # pause once this phase is done
MODEL="${MODEL:-opus}"
EFFORT="${EFFORT-high}"                 # empty: leave the flag off
FALLBACK_MODEL="${FALLBACK_MODEL:-sonnet}"
PLANNED="${PHASE_BUDGETS:-12 24 18 16 10}"   # USD per phase; unspent money rolls forward
TURNS="200 400 300 300 200"             # turn cap per run; the budget is the real limit
WRAP_BUDGET=1.5                         # one wrap-up run per phase once its money is gone
MAX_FAILS=8                             # failed runs in a row before the study stops
BACKOFF_UNIT="${BACKOFF_UNIT:-60}"      # seconds per backoff step (tests shorten it)
SKIP_SETUP="${SKIP_SETUP:-0}"           # 1: no venv and no data (runner self-test only)
TOOLS="Read,Write,Edit,Bash,Glob,Grep,WebSearch,WebFetch,Task,Agent,ToolSearch,TodoWrite,TaskCreate,TaskGet,TaskList,TaskUpdate,TaskStop"

LOGS="$W/logs"; RUNS="$LOGS/runs"; ST="$W/state"
PROGRESS="$LOGS/progress.log"; LEDGER="$ST/ledger.tsv"; PIDFILE="$ST/runner.pid"
# Start a command as the leader of its own process group, so stopping it also stops
# everything it spawned (Bash tools, worker pools). macOS has no setsid(1).
SETSID=(perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV or die "exec $ARGV[0]: $!\n"' --)
CLAUDE_PID=""; FOLLOW_PID=""; COMPUTE_PID=""; API_KEY=""; CLASS=""; CLASS_TEXT=""

# ---------------------------------------------------------------- small helpers
say()  { printf '%s\n' "$*"; }
ok()   { printf 'ok    %s\n' "$*"; }
warn() { printf 'warn  %s\n' "$*"; }
die()  { printf 'error %s\n' "$*" >&2; exit 1; }
log()  { mkdir -p "$LOGS"; printf '%s ==> %s\n' "$(date '+%H:%M:%S')" "$*" >> "$PROGRESS"; }
notify() { osascript -e "display notification \"$1\" with title \"Stein study\"" >/dev/null 2>&1 || true; }
nap()  { sleep "$1" & wait $! 2>/dev/null; }          # a sleep a signal can interrupt
fadd() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { printf "%.4f", a + b }'; }
fsub() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { printf "%.4f", a - b }'; }
fmin() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { printf "%.4f", (a < b ? a : b) }'; }
ge()   { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { exit !(a + 0 >= b + 0) }'; }
usd()  { awk -v a="${1:-0}" 'BEGIN { printf "$%.2f", a }'; }
hrs()  { awk -v s="${1:-0}" 'BEGIN { printf "%.1f", s / 3600 }'; }
nth()  { echo "$2" | awk -v i="$1" '{ print $(i + 1) }'; }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
killgroup() { alive "${1:-}" || return 0; kill -TERM -"$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null; }
compute_hours() { case "$1" in 1) echo "${COMPUTE_HOURS_P1:-1}";; 2) echo "${COMPUTE_HOURS_P2:-6}";; *) echo 0;; esac; }
phase_status() { [ -f "$W/gates/P$1.status" ] && tr -d '[:space:]' < "$W/gates/P$1.status" | tr '[:upper:]' '[:lower:]'; }
on_power() { pmset -g batt 2>/dev/null | grep -q "AC Power"; }

load_key() {   # the key file wins over the environment, so a stray exported key is not used
  if [ -f "$KEY_FILE" ]; then
    chmod 600 "$KEY_FILE" 2>/dev/null
    API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
  else
    API_KEY="${ANTHROPIC_API_KEY:-}"
  fi
  [ -n "$API_KEY" ]
}

check_key() {  # a free call; the key goes to curl on stdin, never on a command line
  local code
  code=$(printf 'header = "x-api-key: %s"\nheader = "anthropic-version: 2023-06-01"\n' "$API_KEY" |
         curl -s -m 20 -o /dev/null -w '%{http_code}' -K - https://api.anthropic.com/v1/models)
  [ "$code" = 200 ]
}

# ---------------------------------------------------------------- money
# One ledger row per finished run: time, phase, kind, session, usd, raw log.
#   R  the session total the CLI reported. It is cumulative across --resume (checked on
#      2.1.319: a resumed run reports the earlier runs' cost plus its own), so a session
#      counts once, at its largest R. --max-budget-usd, by contrast, caps each run alone.
#   E  an estimate for a run that died before reporting. Its session is never resumed,
#      so no later R can count it again.
spent() {
  [ -f "$LEDGER" ] || { echo 0; return; }
  awk -F'\t' -v ph="${1:-all}" '
    ph == "all" || $2 == ph {
      if ($3 == "R") { if ($5 + 0 > cum[$4] + 0) cum[$4] = $5 + 0 } else est += $5
    }
    END { s = est; for (k in cum) s += cum[k]; printf "%.4f", s }' "$LEDGER"
}

allowance() {  # what phase $1 may still spend: its plan plus whatever earlier phases left
  local n="$1" i=0 plan=0 used=0
  while [ "$i" -le "$n" ]; do
    plan=$(fadd "$plan" "$(nth "$i" "$PLANNED")"); used=$(fadd "$used" "$(spent "$i")"); i=$((i + 1))
  done
  fmin "$(fsub "$plan" "$used")" "$(fsub "$TOTAL_CAP" "$(spent)")"
}

finalize_run() {   # finalize_run phase session raw: book a run's cost, exactly once
  local n="$1" sid="$2" raw="$3" cum main subs est
  [ -f "$raw" ] || return 0
  grep -qF "$raw" "$LEDGER" 2>/dev/null && return 0
  cum=$(jq -rR 'fromjson? | select(.type == "result") | .total_cost_usd // empty' "$raw" 2>/dev/null | tail -1)
  if [ -n "$cum" ]; then
    printf '%s\t%s\tR\t%s\t%s\t%s\n' "$(date +%s)" "$n" "$sid" "$cum" "$raw" >> "$LEDGER"
    echo 1 > "$ST/P$n.resumable"
  else
    main=$(jq -rR 'fromjson? | select(.type == "assistant" and .parent_tool_use_id == null) | .message.id' "$raw" 2>/dev/null | sort -u | wc -l | tr -d ' ')
    subs=$(jq -rR 'fromjson? | select(.type == "assistant" and .parent_tool_use_id != null) | .message.id' "$raw" 2>/dev/null | sort -u | wc -l | tr -d ' ')
    est=$(awk -v m="$main" -v s="$subs" 'BEGIN { printf "%.4f", m * 0.20 + s * 0.03 }')
    printf '%s\t%s\tE\t%s\t%s\t%s\n' "$(date +%s)" "$n" "$sid" "$est" "$raw" >> "$LEDGER"
    echo 0 > "$ST/P$n.resumable"
    log "P$n: the run ended without reporting its cost; booked an estimate of $(usd "$est") for its $main API calls"
  fi
}

# ---------------------------------------------------------------- workspace
setup_workspace() {
  mkdir -p "$W" "$RUNS" "$ST" "$W/gates" "$W/notes" "$W/results" "$W/jobs" "$W/paper" \
           "$W/blog" "$W/ref" "$W/data/raw/mnist" "$W/.claude/agents" || die "cannot create $W"
  [ -d "$W/.git" ] || git -C "$W" init -q -b stein-study || die "git init failed in $W"
  cp "$PROMPT_SRC" "$W/PROMPT.md" || die "cannot copy $PROMPT_SRC"
  cp "$HERE/lit-reviewer.md" "$W/.claude/agents/lit-reviewer.md" || die "cannot copy lit-reviewer.md"
  [ -f "$STYLE_SRC" ] && cp "$STYLE_SRC" "$W/ref/STYLE.md"
  [ -f "$W/.gitignore" ] || cat > "$W/.gitignore" <<'EOF'
data/
.venv/
logs/
state/
checkpoints/
__pycache__/
.pytest_cache/
.quarto/
*.pt
*.pth
*.ckpt
*.h5
*.npy
*.npz
.DS_Store
EOF
  cat > "$W/CLAUDE.md" <<'EOF'
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
EOF
  # Turns the CLI's stream-json into one readable line per event for the progress log.
  cat > "$ST/activity.jq" <<'EOF'
def clip($n): tostring | gsub("\\s+"; " ") | if length > $n then .[0:$n] + "…" else . end;
def t: now | strflocaltime("%H:%M:%S");
def brief: (if type == "object" then (.command // .description // .file_path // .path // .pattern // .query // .url // .) else . end) | clip(150);
try (
  fromjson
  | if .type == "system" and .subtype == "init" then
      "\(t) \($p)  ▶ session \(.session_id[0:8]) started (\(.model), billed via \(.apiKeySource))"
    elif .type == "system" and .subtype == "compact_boundary" then
      "\(t) \($p)  ⋯ context compacted"
    elif .type == "assistant" then
      (if .parent_tool_use_id then "    ↳ " else "  " end) as $ind
      | .message.content[]?
      | if .type == "text" and $ind == "  " then "\(t) \($p)\($ind)💬 \(.text | clip(300))"
        elif .type == "tool_use" then "\(t) \($p)\($ind)🔧 \(.name)  \(.input | brief)"
        else empty end
    elif .type == "result" then
      "\(t) \($p)  🏁 run ended: \(.subtype); session total $\((.total_cost_usd // 0) * 100 | round / 100); \(.num_turns) turns"
    else empty end
) catch empty
EOF
}

setup_venv() {
  local py="$W/.venv/bin/python" sha
  [ -x "$py" ] || uv venv -q --python 3.12 "$W/.venv" || die "could not create $W/.venv"
  if ! "$py" -c 'import torch, numpy, scipy, pandas, pyarrow, sklearn, h5py, yaml, pytest, joblib, matplotlib' 2>/dev/null; then
    say "installing the Python stack into $W/.venv (a few minutes, once)..."
    uv pip install -q --python "$py" torch numpy scipy pandas pyarrow scikit-learn h5py \
      pyyaml pytest joblib matplotlib tqdm || die "package install failed"
  fi
  if ! "$py" -c 'import lejepa' 2>/dev/null; then
    sha=$(git ls-remote https://github.com/rbalestr-lab/lejepa HEAD 2>/dev/null | cut -f1)
    if [ -n "$sha" ] && uv pip install -q --python "$py" "lejepa @ git+https://github.com/rbalestr-lab/lejepa@$sha" \
       && "$py" -c 'import lejepa' 2>/dev/null; then
      echo "$sha" > "$ST/lejepa.sha"
    else
      warn "could not install lejepa from GitHub; P1 must implement SIGReg itself and say so"
    fi
  fi
  "$py" -c 'import platform, torch; print(f"Python {platform.python_version()}, torch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}.")' \
    > "$ST/env.txt" 2>/dev/null || die "the venv's torch does not import"
  [ -f "$ST/lejepa.sha" ] && echo "lejepa from github.com/rbalestr-lab/lejepa at commit $(cut -c1-12 "$ST/lejepa.sha")." >> "$ST/env.txt"
  ok "venv: $(tr '\n' ' ' < "$ST/env.txt")"
}

fetch() {   # fetch url dest bytes: resumable download, checked by size
  local url="$1" dest="$2" bytes="$3" have
  have=$(stat -f %z "$dest" 2>/dev/null || echo 0)
  [ "$have" -ge "$bytes" ] && return 0
  have=$(stat -f %z "$dest.part" 2>/dev/null || echo 0)
  if [ "$have" -lt "$bytes" ]; then
    say "downloading $(basename "$dest") ($(awk -v b="$bytes" 'BEGIN { printf "%.0f MB", b / 1e6 }'))..."
    curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors -C - -o "$dest.part" "$url" || die "download failed: $url"
    have=$(stat -f %z "$dest.part" 2>/dev/null || echo 0)
    [ "$have" -ge "$bytes" ] || die "$(basename "$dest") is short ($have of $bytes bytes); rerun preflight to resume"
  fi
  mv "$dest.part" "$dest"
}

fetch_data() {
  local r="$W/data/raw" m="https://ossci-datasets.s3.amazonaws.com/mnist"
  fetch https://storage.googleapis.com/3d-shapes/3dshapes.h5 "$r/3dshapes.h5" 267573662
  fetch https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "$r/cifar-10-python.tar.gz" 170498071
  fetch "$m/train-images-idx3-ubyte.gz" "$r/mnist/train-images-idx3-ubyte.gz" 9912422
  fetch "$m/train-labels-idx1-ubyte.gz" "$r/mnist/train-labels-idx1-ubyte.gz" 28881
  fetch "$m/t10k-images-idx3-ubyte.gz" "$r/mnist/t10k-images-idx3-ubyte.gz" 1648877
  fetch "$m/t10k-labels-idx1-ubyte.gz" "$r/mnist/t10k-labels-idx1-ubyte.gz" 4542
  ok "data: $(du -sh "$r" | cut -f1 | tr -d ' ') in $r"
}

smoke_test() {   # one real headless call with the study's own flags, on Haiku (well under a cent)
  local out="$ST/smoke.jsonl" err="$ST/smoke.err" cmd src agents res cum
  if [ -x "$W/.venv/bin/python" ]; then
    cmd=".venv/bin/python -c 'import torch; print(\"torch\", torch.__version__)'"
  else
    cmd="echo smoke-ok"
  fi
  echo "Run this one bash command, then reply with only its output: $cmd" |
    ( cd "$W" && ANTHROPIC_API_KEY="$API_KEY" DISABLE_AUTOUPDATER=1 claude -p --model haiku \
        --max-budget-usd 0.05 --max-turns 3 --output-format stream-json --verbose \
        --permission-mode dontAsk --permission-prompts none --strict-mcp-config \
        --disallowedTools Skill --allowedTools "$TOOLS" ) > "$out" 2> "$err"
  src=$(jq -rR 'fromjson? | select(.type == "system" and .subtype == "init") | .apiKeySource' "$out" 2>/dev/null | head -1)
  agents=$(jq -rR 'fromjson? | select(.type == "system" and .subtype == "init") | .agents | join(" ")' "$out" 2>/dev/null | head -1)
  res=$(jq -rR 'fromjson? | select(.type == "result") | .result // ""' "$out" 2>/dev/null | tail -1)
  cum=$(jq -rR 'fromjson? | select(.type == "result") | .total_cost_usd // 0' "$out" 2>/dev/null | tail -1)
  printf '%s\tpre\tR\tsmoke-%s\t%s\tsmoke-%s\n' "$(date +%s)" "$$" "${cum:-0}" "$$" >> "$LEDGER"
  [ "$src" = ANTHROPIC_API_KEY ] || die "the test call authenticated as '${src:-unknown}', not with the API key ($err)"
  case " $agents " in *" lit-reviewer "*) ;; *) die "the lit-reviewer sub-agent did not load";; esac
  case "$res" in *torch*|*smoke-ok*) ;; *) die "the test call did not run its command (got: $(printf '%s' "$res" | head -c 200)); see $err";; esac
  ok "headless test call: billed to the API key, ran a command, loaded lit-reviewer ($(usd "$cum"))"
}

# ---------------------------------------------------------------- messages to the agent
earlier() {
  local n="$1" i=0 out="" s
  while [ "$i" -lt "$n" ]; do
    s=$(phase_status "$i"); out="$out${out:+, }P$i ${s:-not run}"; i=$((i + 1))
  done
  echo "$out"
}

run_context() {
  local n="$1" a="$2" chip cores perf mem disk data
  chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown CPU")
  cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "?")
  perf=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
  mem=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  disk=$(df -g "$W" 2>/dev/null | awk 'NR == 2 { print $4 }')
  data=$(cd "$W" && find data/raw -type f ! -name '*.part' 2>/dev/null | sort | while read -r f; do
           printf '%s (%s), ' "$f" "$(du -h "$f" | cut -f1 | tr -d ' ')"; done | sed 's/, $//')
  cat <<EOF
## Run context (written by the runner, $(date '+%Y-%m-%d %H:%M %Z'))
- Machine: $chip, $cores cores ($perf performance), $mem GB RAM, ${disk:-?} GB free disk.
- Environment: $( [ -f "$ST/env.txt" ] && tr '\n' ' ' < "$ST/env.txt" || echo "not built")
- Data: ${data:-none}
- Budget: this phase may spend $(usd "$a") more, across all its runs. That is a hard cap:
  a run is cut off when it reaches it. The study has spent $(usd "$(spent)") of its
  $(usd "$TOTAL_CAP") cap.
- Compute allowance for handed-off jobs in this phase: $(compute_hours "$n") h of wall clock
  (0 means no handoff: keep every command under 45 minutes).
- Bash: the default command timeout is 10 minutes; pass a longer one (up to 2700000 ms)
  when a command needs it.
EOF
}

fresh_message() {   # fresh_message phase allowance news-file: the first message of a session
  local n="$1" a="$2" news="$3"
  cat "$W/PROMPT.md"
  printf '\n\n---\n\n# This run: P%s\n\n' "$n"
  if [ "$n" = 0 ]; then
    echo "This is the first session of the study. Do P0 only."
  else
    echo "This is a fresh session for P$n. Earlier phases: $(earlier "$n"). Their results are on disk: start with gates/GATE_0.md to gates/GATE_$((n - 1)).md, and open other files only when you need them. Do P$n only."
  fi
  if [ -f "$W/gates/GATE_$n.md" ]; then
    printf '\nAn earlier session already worked on P%s and was cut off. Read gates/GATE_%s.md and check what is on disk before doing anything else, and do not redo finished work.\n' "$n" "$n"
  fi
  if [ -s "$news" ]; then printf '\n## Since the last session\n\n'; cat "$news"; fi
  echo
  run_context "$n" "$a"
}

# ---------------------------------------------------------------- running claude
start_follower() {   # start_follower raw label: tail the run into the progress log
  ( exec "${SETSID[@]}" bash -c 'tail -n +1 -F "$1" 2>/dev/null | jq --unbuffered -rR --arg p "$2" -f "$3"' \
      _ "$1" "$2" "$ST/activity.jq" >> "$PROGRESS" 2>/dev/null ) &
  FOLLOW_PID=$!
}

check_auth_source() {   # the run must bill the API key, never the claude.ai login
  local raw="$1" pid="$2" i=0 src=""
  while [ "$i" -lt 120 ]; do
    src=$(jq -rR 'fromjson? | select(.type == "system" and .subtype == "init") | .apiKeySource' "$raw" 2>/dev/null | head -1)
    [ -n "$src" ] && break
    alive "$pid" || return 0            # ended before starting; classify() sorts it out
    nap 0.5; i=$((i + 1))
  done
  [ "$src" = ANTHROPIC_API_KEY ] && return 0
  killgroup "$pid"
  if [ -z "$src" ]; then log "the run never reported how it was billed; stopped it"; return 2; fi
  log "FATAL: a run was billed via '$src' instead of the API key; stopped it at once"
  return 1
}

run_claude() {   # run_claude phase budget message-file: sets RUN_RAW, RUN_ERR, AUTH_BAD
  local n="$1" budget="$2" msg="$3" k sid mode raw err turns fb="" auth=0
  k=$(( $(ls "$RUNS"/P"$n"-*.jsonl 2>/dev/null | wc -l) + 1 ))
  raw="$RUNS/P$n-$k.jsonl"; err="$RUNS/P$n-$k.err"
  if [ "$(cat "$ST/P$n.resumable" 2>/dev/null)" = 1 ] && [ -s "$ST/P$n.session" ]; then
    sid=$(cat "$ST/P$n.session"); mode=--resume
  else
    sid=$(uuidgen | tr '[:upper:]' '[:lower:]'); echo "$sid" > "$ST/P$n.session"; mode=--session-id
  fi
  echo 0 > "$ST/P$n.resumable"          # until this run reports its cost
  turns=$(nth "$n" "$TURNS")
  [ "$FALLBACK_MODEL" != "$MODEL" ] && fb="--fallback-model $FALLBACK_MODEL"
  log "P$n: run $k ($( [ "$mode" = --resume ] && echo "resuming" || echo "new") session ${sid:0:8}, may spend $(usd "$budget"))"
  : > "$raw"
  # The prompt goes in on stdin: --allowedTools takes a variable number of values and
  # would swallow a prompt placed after it.
  ( cd "$W" && exec "${SETSID[@]}" claude -p "$mode" "$sid" --model "$MODEL" ${EFFORT:+--effort "$EFFORT"} $fb \
      --max-budget-usd "$budget" --max-turns "$turns" --output-format stream-json --verbose \
      --permission-mode dontAsk --permission-prompts none --strict-mcp-config \
      --disallowedTools Skill --allowedTools "$TOOLS" < "$msg" > "$raw" 2> "$err" ) &
  CLAUDE_PID=$!
  printf '%s %s %s %s\n' "$n" "$sid" "$raw" "$CLAUDE_PID" > "$ST/inflight"
  start_follower "$raw" "P$n"
  check_auth_source "$raw" "$CLAUDE_PID"; auth=$?
  wait "$CLAUDE_PID" 2>/dev/null
  CLAUDE_PID=""
  nap 2; killgroup "$FOLLOW_PID"; FOLLOW_PID=""
  finalize_run "$n" "$sid" "$raw"; rm -f "$ST/inflight"
  RUN_RAW="$raw"; RUN_ERR="$err"; AUTH_BAD="$auth"
}

classify() {   # classify raw err: sets CLASS (ok|turns|budget|fatal|noresume|error), CLASS_TEXT
  local raw="$1" err="$2" sub iserr
  sub=$(jq -rR 'fromjson? | select(.type == "result") | .subtype' "$raw" 2>/dev/null | tail -1)
  iserr=$(jq -rR 'fromjson? | select(.type == "result") | .is_error' "$raw" 2>/dev/null | tail -1)
  CLASS_TEXT=$( { jq -rR 'fromjson? | select(.type == "result") | "\(.result // "") \((.errors // []) | join(" "))"' "$raw" 2>/dev/null | tail -1
                  grep -v 'claude.ai connectors are disabled' "$err" 2>/dev/null | tail -c 2000; } | tr '\n' ' ' | cut -c1-400)
  # Match error text only on runs that failed: a healthy run's final message may mention
  # anything, including the phrases below.
  if [ "$sub" = success ] && [ "$iserr" != true ]; then CLASS=ok
  elif [ "$sub" = error_max_budget_usd ]; then CLASS=budget
  elif [ "$sub" = error_max_turns ]; then CLASS=turns
  elif printf '%s' "$CLASS_TEXT" | grep -Eiq 'credit balance is too low|invalid api key|invalid x-api-key|authentication_error|permission_error|organization (has been )?disabled'; then
    CLASS=fatal
  elif printf '%s' "$CLASS_TEXT" | grep -Eiq 'no conversation found'; then CLASS=noresume
  else CLASS=error; fi
}

wait_for_network() {
  local said=0
  until curl -s -m 15 -o /dev/null https://api.anthropic.com/; do
    [ "$said" = 1 ] || { log "no network; waiting for it to come back"; said=1; }
    check_stop; nap 60
  done
  [ "$said" = 1 ] && log "network is back"
  return 0
}

# ---------------------------------------------------------------- compute handoff
run_compute() {   # run_compute phase news-file: run jobs/P<n>.sh within the allowance
  local n="$1" news="$2" k used limit clog start now elapsed rc timed_out=0 lastlog last
  k=$(( $(cat "$ST/P$n.jobs" 2>/dev/null || echo 0) + 1 ))
  used=$(cat "$ST/P$n.compute_s" 2>/dev/null || echo 0)
  limit=$(awk -v h="$(compute_hours "$n")" -v u="$used" 'BEGIN { printf "%d", h * 3600 - u }')
  if [ ! -f "$W/jobs/P$n.sh" ]; then
    echo "You wrote \`compute\` to gates/P$n.status, but jobs/P$n.sh does not exist, so nothing ran. Write it and hand off again, or carry on without a handoff." > "$news"
    : > "$W/gates/P$n.status"; return
  fi
  if [ "$k" -gt 2 ] || [ "$limit" -lt 300 ]; then
    echo "Compute handoff refused: P$n has $( [ "$k" -gt 2 ] && echo "already used both of its handoffs" || echo "less than 5 minutes of compute allowance left"). Nothing ran. Work with the results on disk, record the shortfall in gates/GATE_$n.md, and carry on with P$n." > "$news"
    : > "$W/gates/P$n.status"; return
  fi
  commit "P$n: before compute job $k"
  clog="$LOGS/compute-P$n-$k.log"
  log "P$n: compute job $k started (allowance $(hrs "$limit") h); its log is logs/compute-P$n-$k.log"
  ( cd "$W" && exec "${SETSID[@]}" env -u ANTHROPIC_API_KEY bash "jobs/P$n.sh" ) > "$clog" 2>&1 &
  COMPUTE_PID=$!; echo "$COMPUTE_PID" > "$ST/compute.pid"
  start=$(date +%s); lastlog=$start
  while alive "$COMPUTE_PID"; do
    nap 30
    now=$(date +%s); elapsed=$((now - start))
    echo $((used + elapsed)) > "$ST/P$n.compute_s"
    if [ "$elapsed" -ge "$limit" ]; then
      log "P$n: compute job $k reached the allowance; stopping it"
      killgroup "$COMPUTE_PID"; nap 60; kill -KILL -"$COMPUTE_PID" 2>/dev/null; timed_out=1; break
    fi
    if [ $((now - lastlog)) -ge 600 ]; then
      last=$(grep -a '^PROGRESS' "$clog" 2>/dev/null | tail -1 | cut -c1-160)
      log "P$n: job $k, $(hrs "$elapsed") h in: ${last:-no PROGRESS line yet}"; lastlog=$now
    fi
  done
  wait "$COMPUTE_PID" 2>/dev/null; rc=$?
  COMPUTE_PID=""; rm -f "$ST/compute.pid"
  elapsed=$(( $(date +%s) - start )); echo $((used + elapsed)) > "$ST/P$n.compute_s"
  echo "$k" > "$ST/P$n.jobs"        # counted only once finished: a stopped runner reruns it
  log "P$n: compute job $k ended (exit $rc after $(hrs "$elapsed") h$( [ "$timed_out" = 1 ] && echo ", stopped at the allowance"))"
  { echo "The runner ran compute job $k (\`bash jobs/P$n.sh\`): exit code $rc after $(hrs "$elapsed") h$( [ "$timed_out" = 1 ] && echo ", stopped because it reached this phase's compute allowance; whatever it finished is on disk")."
    echo "Compute left for this phase: $(hrs $((limit - elapsed))) h, and $((2 - k)) more handoff(s)."
    echo "Full log: logs/compute-P$n-$k.log. Its last 40 lines:"
    echo '```'; tail -n 40 "$clog" | cut -c1-300; echo '```'
    echo "Continue P$n."; } > "$news"
  : > "$W/gates/P$n.status"
}

commit() {   # commit message: snapshot the workspace; never fails the run
  ( cd "$W" || exit 0
    git add -A >/dev/null 2>&1
    git diff --cached --name-only -z | while IFS= read -r -d '' f; do     # nothing over 50 MB
      if [ -f "$f" ] && [ "$(stat -f %z "$f")" -gt 52428800 ]; then
        git reset -q -- "$f"; echo "/$f" >> .git/info/exclude
      fi
    done
    git commit -q -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 ) || true
}

# ---------------------------------------------------------------- the phase loop
run_phase() {   # returns 0 to go on, 2 when blocked, 3 when the study must stop
  local n="$1" st a fails=0 nudges=0 wrapped=0 force_wrap=0 before wait_min
  local news="$ST/news.txt" msg="$ST/message.txt"
  st=$(phase_status "$n")
  case "$st" in
    complete*|incomplete*) log "P$n: already $st; skipping"; return 0;;
    blocked*) log "P$n: blocked (see gates/GATE_$n.md); the study cannot go on"; return 2;;
  esac
  log "P$n: starting; it may spend $(usd "$(allowance "$n")") (study so far $(usd "$(spent)") of $(usd "$TOTAL_CAP"))"
  [ -f "$news" ] || : > "$news"
  while :; do
    check_stop
    st=$(phase_status "$n")
    case "$st" in
      complete*) commit "P$n complete"; log "P$n: COMPLETE (phase spend $(usd "$(spent "$n")"))"; : > "$news"; return 0;;
      blocked*)  commit "P$n blocked"; log "P$n: BLOCKED; read gates/GATE_$n.md"; return 2;;
      compute*)  run_compute "$n" "$news"; continue;;
    esac
    if ge "$(spent)" "$(fsub "$TOTAL_CAP" 0.5)"; then
      log "the study's budget is spent ($(usd "$(spent)") of $(usd "$TOTAL_CAP"))"; return 3
    fi
    a=$(allowance "$n")
    if [ "$force_wrap" = 1 ] || ! ge "$a" 1; then
      if [ "$wrapped" = 0 ] && ge "$(fsub "$TOTAL_CAP" "$(spent)")" "$WRAP_BUDGET"; then
        wrapped=1; a=$WRAP_BUDGET
        log "P$n: phase budget used up; one wrap-up run (at most $(usd "$a"))"
        echo "This phase's budget is spent. Start no new work. Make gates/GATE_$n.md an honest final record (done, not done, what the next phase needs to know), then write \`complete\` to gates/P$n.status if the next phase can build on what exists, or \`blocked\` if it cannot, and end the run." >> "$news"
      else
        echo incomplete > "$W/gates/P$n.status"; commit "P$n incomplete"
        log "P$n: INCOMPLETE (out of budget); moving on"; : > "$news"; return 0
      fi
    fi
    if [ "$(cat "$ST/P$n.resumable" 2>/dev/null)" = 1 ] && [ -s "$news" ]; then
      cp "$news" "$msg"
    else
      fresh_message "$n" "$a" "$news" > "$msg"
    fi
    : > "$news"
    wait_for_network
    before=$(spent)
    run_claude "$n" "$a" "$msg"
    if [ "$AUTH_BAD" = 1 ]; then return 3; fi
    classify "$RUN_RAW" "$RUN_ERR"
    ge "$(fsub "$(spent)" "$before")" 0.05 && fails=0
    case "$CLASS" in
      ok)
        if [ -z "$(phase_status "$n")" ]; then
          nudges=$((nudges + 1)); [ "$nudges" -gt 3 ] && force_wrap=1
          echo "You ended the run, but gates/P$n.status is empty. If P$n is finished, finalise gates/GATE_$n.md and write the status. Otherwise carry on with P$n." > "$news"
        fi;;
      turns)  echo "The run reached its turn cap. Carry on with P$n where you left off." > "$news";;
      budget) : ;;                      # the next pass sees the allowance is gone
      fatal)  log "FATAL: $CLASS_TEXT"; return 3;;
      noresume)
        echo 0 > "$ST/P$n.resumable"
        echo "The previous session could not be resumed, so this is a fresh one." > "$news";;
      error)
        fails=$((fails + 1))
        if [ "$fails" -ge "$MAX_FAILS" ]; then
          log "P$n: $fails failed runs in a row; stopping. Last error: ${CLASS_TEXT:-none reported}"; return 3
        fi
        wait_min=$(( 2 ** fails )); [ "$wait_min" -gt 30 ] && wait_min=30
        log "P$n: run failed (${CLASS_TEXT:-no error text; see $RUN_ERR}); retrying in $wait_min min"
        nap $((wait_min * BACKOFF_UNIT))
        echo "The previous run stopped on an error (${CLASS_TEXT:-no error text}). Carry on with P$n where you left off." > "$news";;
    esac
  done
}

check_stop() { if [ -f "$ST/STOP" ]; then log "stop requested"; cleanup; exit 0; fi; }

cleanup() {
  local n sid raw pid
  killgroup "$CLAUDE_PID"; killgroup "$COMPUTE_PID"; killgroup "$FOLLOW_PID"
  if [ -f "$ST/inflight" ]; then
    read -r n sid raw pid < "$ST/inflight"; nap 2
    finalize_run "$n" "$sid" "$raw"; rm -f "$ST/inflight"
  fi
  rm -f "$PIDFILE"
}

recover() {   # tidy up after a runner that died without cleaning up (reboot, kill -9)
  local n sid raw pid
  if [ -f "$ST/inflight" ]; then
    read -r n sid raw pid < "$ST/inflight"
    if alive "$pid" && ps -p "$pid" -o command= 2>/dev/null | grep -q claude; then killgroup "$pid"; nap 3; fi
    finalize_run "$n" "$sid" "$raw"; rm -f "$ST/inflight"
  fi
  if [ -f "$ST/compute.pid" ]; then
    pid=$(cat "$ST/compute.pid")
    if alive "$pid" && ps -p "$pid" -o command= 2>/dev/null | grep -q 'jobs/P'; then
      killgroup "$pid"; log "stopped an orphaned compute job (pid $pid); it will rerun"
    fi
    rm -f "$ST/compute.pid"
  fi
}

main_loop() {
  local n=0 rc=0
  setup_workspace
  load_key || { log "FATAL: no API key in $KEY_FILE"; exit 1; }
  # Inherited by every claude run (and so never on a command line). The two timeouts let
  # the agent run a pilot in the foreground for up to 45 minutes.
  export ANTHROPIC_API_KEY="$API_KEY" DISABLE_AUTOUPDATER=1 \
         BASH_DEFAULT_TIMEOUT_MS=600000 BASH_MAX_TIMEOUT_MS=2700000
  echo $$ > "$PIDFILE"
  trap 'log "runner stopped by a signal"; cleanup; exit 130' TERM INT HUP
  caffeinate -i -s -w $$ >/dev/null 2>&1 &       # no sleep while this process lives
  recover
  log "runner started $(date '+%a %d %b %Y') (pid $$); study cap $(usd "$TOTAL_CAP"), spent so far $(usd "$(spent)")"
  while [ "$n" -le 4 ]; do
    run_phase "$n"; rc=$?
    [ "$rc" -ne 0 ] && break
    if [ "$n" -ge "$STOP_AFTER" ] && [ "$n" -lt 4 ]; then
      log "pausing after P$n as STOP_AFTER=$STOP_AFTER asks; ./stein.sh start carries on"
      notify "P$n done; paused as asked"; cleanup; exit 0
    fi
    n=$((n + 1))
  done
  case "$rc" in
    0) log "ALL PHASES DONE. Total spend $(usd "$(spent)"). Start with $W/FINAL.md"; notify "Study finished ($(usd "$(spent)"))";;
    2) notify "Study blocked at P$n; see the progress log";;
    *) log "the study stopped at P$n"; notify "Study stopped at P$n; see the progress log";;
  esac
  cleanup
}

# ---------------------------------------------------------------- commands
cmd_preflight() {
  local t avail
  say "Preflight for the Stein study (workspace: $W)"
  for t in claude jq git curl uv perl caffeinate uuidgen pmset; do
    command -v "$t" >/dev/null 2>&1 || die "missing tool: $t"
  done
  ok "tools: claude $(claude --version 2>/dev/null | awk '{ print $1 }'), $(jq --version), uv $(uv --version | awk '{ print $2 }')"
  if command -v quarto >/dev/null 2>&1; then ok "quarto $(quarto --version) (P3 renders the paper and blog)"
  else warn "quarto not found; P3 needs it to render the paper"; fi
  if on_power; then ok "on mains power"; else warn "on battery: plug in before starting"; fi
  setup_workspace
  avail=$(df -g "$W" | awk 'NR == 2 { print $4 }')
  [ "${avail:-0}" -ge 15 ] || die "only ${avail:-?} GB free; the study needs about 15 GB"
  ok "workspace $W ($avail GB free)"
  if [ "$SKIP_SETUP" != 1 ]; then setup_venv; fetch_data; fi
  load_key || die "no API key: put it in $KEY_FILE (README.md shows how), then rerun"
  check_key || die "api.anthropic.com rejected the key in $KEY_FILE"
  ok "API key accepted; every run bills that key's Console credits, never the claude.ai plan"
  smoke_test
  ok "preflight passed; study spend so far $(usd "$(spent)") of the $(usd "$TOTAL_CAP") cap"
}

cmd_start() {
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null)
  alive "$pid" && die "already running (pid $pid); ./stein.sh status shows where it is"
  cmd_preflight
  if ! on_power && [ "${ALLOW_BATTERY:-0}" != 1 ]; then
    die "not on mains power. Plug in, or set ALLOW_BATTERY=1"
  fi
  rm -f "$ST/STOP"
  ( exec "${SETSID[@]}" nohup bash "$SELF" _loop < /dev/null >> "$LOGS/runner.log" 2>&1 ) &
  sleep 3
  pid=$(cat "$PIDFILE" 2>/dev/null)
  alive "$pid" || die "the runner did not start; see $LOGS/runner.log"
  say ""
  say "Started (pid $pid). You can close this terminal; keep the Mac plugged in, awake and online."
  say "  follow:  ./stein.sh watch     summary: ./stein.sh status     stop: ./stein.sh stop"
  say "  log:     $PROGRESS"
}

cmd_status() {
  local pid n st
  [ -d "$ST" ] || die "no workspace at $W yet; run ./stein.sh preflight"
  pid=$(cat "$PIDFILE" 2>/dev/null)
  say "Workspace: $W"
  if alive "$pid"; then say "Runner:    running (pid $pid)"; else say "Runner:    not running"; fi
  printf '\n  %-6s %-12s %9s %9s\n' phase status spent plan
  for n in 0 1 2 3 4; do
    st=$(phase_status "$n")
    if [ -z "$st" ]; then
      if ls "$RUNS"/P"$n"-*.jsonl >/dev/null 2>&1; then st="in progress"; else st="not started"; fi
    fi
    printf '  %-6s %-12s %9s %9s\n' "P$n" "$st" "$(usd "$(spent "$n")")" "$(usd "$(nth "$n" "$PLANNED")")"
  done
  printf '\n  spent %s of the %s cap\n' "$(usd "$(spent)")" "$(usd "$TOTAL_CAP")"
  if [ -f "$ST/compute.pid" ] && alive "$(cat "$ST/compute.pid")"; then
    printf '  compute job running: %s\n' "$(grep -ah '^PROGRESS' "$LOGS"/compute-*.log 2>/dev/null | tail -1)"
  fi
  printf '\nLatest activity (%s):\n' "$PROGRESS"
  tail -n 12 "$PROGRESS" 2>/dev/null | cut -c1-200
}

cmd_stop() {
  local pid i=0
  pid=$(cat "$PIDFILE" 2>/dev/null)
  alive "$pid" || { say "The runner is not running."; return 0; }
  touch "$ST/STOP"; kill -TERM "$pid" 2>/dev/null
  while alive "$pid" && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
  alive "$pid" && kill -KILL "$pid" 2>/dev/null
  say "Stopped. ./stein.sh start resumes where it left off."
}

case "${1:-}" in
  preflight) cmd_preflight;;
  start)     cmd_start;;
  status)    cmd_status;;
  watch)     [ -f "$PROGRESS" ] || die "no progress log yet at $PROGRESS"; tail -n 40 -F "$PROGRESS";;
  stop)      cmd_stop;;
  _loop)     main_loop;;
  *)         sed -n '2,10p' "$SELF"; exit 1;;
esac
