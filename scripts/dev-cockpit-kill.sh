#!/usr/bin/env bash
# Kill the rs-dev tmux cockpit session. No-op if it doesn't exist.
set -euo pipefail
tmux kill-session -t rs-dev 2>/dev/null || true
