#!/usr/bin/env bash
# Dev cockpit: one tmux session with rebuild watcher, interactive driver,
# test watcher, and a scratch pane.
#
# Layout (2x2):
#   +------------------------+------------------------+
#   | cargo xtask watch      | dev-session.py         |
#   | (rebuild on change)    | (interactive driver,   |
#   |                        |  owns the daemon)      |
#   +------------------------+------------------------+
#   | cargo xtask test-watch | scratch shell          |
#   | (rerun tests)          | (tail / git / gh / etc)|
#   +------------------------+------------------------+
#
# The driver pane (top-right) spawns its own daemon as a child, so we
# intentionally do NOT start a standalone daemon in any pane. The
# bottom-right pane is left as a free shell rather than auto-launching the
# TUI (which would need its own daemon process).
#
# Re-running attaches to the existing session instead of erroring.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

SESSION="rs-dev"
DAEMON_BIN="$REPO_ROOT/target/debug/rocket-surgeon"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------------

command -v tmux >/dev/null || die "tmux not found in PATH — brew install tmux"
command -v cargo >/dev/null || die "cargo not found in PATH"

if [[ ! -x "$DAEMON_BIN" ]]; then
    warn "daemon binary missing at $DAEMON_BIN"
    warn "the 'cargo xtask watch' pane will build it on first run,"
    warn "but if you've never bootstrapped, run: scripts/bootstrap.sh"
fi

# ----------------------------------------------------------------------------
# Attach-if-exists
# ----------------------------------------------------------------------------

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "session '$SESSION' already exists, attaching..."
    exec tmux attach-session -t "$SESSION"
fi

# ----------------------------------------------------------------------------
# Build the 2x2 layout
# ----------------------------------------------------------------------------
#
# Start with a single pane (top-left), then:
#   1. split horizontally  -> top-left | top-right
#   2. split top-left vertically  -> top-left / bottom-left  (left column)
#   3. split top-right vertically -> top-right / bottom-right (right column)
#
# Pane indices after these splits (default tmux numbering, base-index 0,
# created in order: TL exists, then TR from -h split, then BL from -v split
# of TL, then BR from -v split of TR):
#   0 = top-left      (cargo xtask watch)
#   1 = top-right     (dev-session.py)
#   2 = bottom-left   (cargo xtask test-watch)
#   3 = bottom-right  (scratch)
#
# We then resize the top row to 60% height and the left column to 50% width.

# Pane 0: top-left — cargo xtask watch
tmux new-session -d -s "$SESSION" -n cockpit -c "$REPO_ROOT" \
    "cargo xtask watch; exec ${SHELL:-/bin/bash}"

# Split horizontally: creates pane 1 to the right (top-right)
tmux split-window -h -t "${SESSION}:cockpit.0" -c "$REPO_ROOT" \
    "PYTHONPATH=python python scripts/dev-session.py; exec ${SHELL:-/bin/bash}"

# Split pane 0 (top-left) vertically -> new pane is bottom-left
tmux split-window -v -t "${SESSION}:cockpit.0" -c "$REPO_ROOT" \
    "cargo xtask test-watch; exec ${SHELL:-/bin/bash}"

# Split pane 1 (top-right) vertically -> bottom-right becomes pane 3.
tmux split-window -v -t "${SESSION}:cockpit.1" -c "$REPO_ROOT" \
    "printf '\\033[1;36m# scratch pane — for TUI, logs, git, whatever\\033[0m\\n'; exec ${SHELL:-/bin/bash}"

# ----------------------------------------------------------------------------
# Resize: top row 60% height, bottom row 40%.
# ----------------------------------------------------------------------------
#
# tmux resize-pane -y sets the pane height in lines. We approximate 60/40
# by computing from the window height. Fall back gracefully if tput fails.

WIN_HEIGHT="$(tmux display-message -p -t "${SESSION}:cockpit" '#{window_height}' 2>/dev/null || echo 40)"
TOP_HEIGHT=$(( WIN_HEIGHT * 6 / 10 ))
if (( TOP_HEIGHT > 0 )); then
    tmux resize-pane -t "${SESSION}:cockpit.0" -y "$TOP_HEIGHT" 2>/dev/null || true
    tmux resize-pane -t "${SESSION}:cockpit.1" -y "$TOP_HEIGHT" 2>/dev/null || true
fi

# Equalize column widths to 50/50.
WIN_WIDTH="$(tmux display-message -p -t "${SESSION}:cockpit" '#{window_width}' 2>/dev/null || echo 200)"
LEFT_WIDTH=$(( WIN_WIDTH / 2 ))
if (( LEFT_WIDTH > 0 )); then
    tmux resize-pane -t "${SESSION}:cockpit.0" -x "$LEFT_WIDTH" 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# Pane titles (visible if pane-border-status is enabled in user's tmux conf).
# ----------------------------------------------------------------------------

tmux set-option -t "$SESSION" pane-border-status top 2>/dev/null || true
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} ' 2>/dev/null || true

tmux select-pane -t "${SESSION}:cockpit.0" -T "watch (rebuild)"
tmux select-pane -t "${SESSION}:cockpit.1" -T "driver (dev-session.py)"
tmux select-pane -t "${SESSION}:cockpit.2" -T "test-watch"
tmux select-pane -t "${SESSION}:cockpit.3" -T "scratch"

# ----------------------------------------------------------------------------
# Focus the driver pane and attach.
# ----------------------------------------------------------------------------

tmux select-pane -t "${SESSION}:cockpit.1"

log "started rs-dev session, attaching..."
exec tmux attach-session -t "$SESSION"
