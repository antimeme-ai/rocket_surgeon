#!/usr/bin/env bash
# Idempotent project bootstrap.
#
# - Reads .python-version for the interpreter pin.
# - Uses `uv` to install Python, create .venv, install dev deps.
# - Builds the PyO3 extension into the venv via maturin.
# - Builds the Rust workspace.
# - Smoke-checks that the daemon binary exists and the Python extension imports.
#
# Re-running is safe: every step short-circuits when the desired state is
# already in place.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY_VERSION_FILE="$REPO_ROOT/.python-version"
VENV_DIR="$REPO_ROOT/.venv"
TARGET_DIR="$REPO_ROOT/target/debug"
DAEMON_BIN="$TARGET_DIR/rocket-surgeon"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------------

[[ -f "$PY_VERSION_FILE" ]] || die "missing $PY_VERSION_FILE"
PY_VERSION="$(tr -d '[:space:]' < "$PY_VERSION_FILE")"
[[ -n "$PY_VERSION" ]] || die ".python-version is empty"

command -v cargo >/dev/null || die "cargo not found in PATH"
command -v uv >/dev/null || die "uv not found — install from https://docs.astral.sh/uv/"
command -v lefthook >/dev/null || die "lefthook not found — install with 'brew install lefthook' or see https://lefthook.dev/installation/"

if ! command -v cargo-watch >/dev/null; then
    log "Installing cargo-watch (needed by 'cargo xtask watch' / 'test-watch')"
    cargo install cargo-watch
else
    log "cargo-watch already installed"
fi

log "Python pin: $PY_VERSION"
log "Repo root:  $REPO_ROOT"

# ----------------------------------------------------------------------------
# 1. Ensure the pinned Python is installed via uv.
# ----------------------------------------------------------------------------

log "Ensuring Python $PY_VERSION via uv (idempotent)"
uv python install "$PY_VERSION"

# ----------------------------------------------------------------------------
# 2. Create / refresh the venv at .venv using the pinned interpreter.
# ----------------------------------------------------------------------------

if [[ -d "$VENV_DIR" ]]; then
    EXISTING_VER="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    if [[ "$EXISTING_VER" != "$PY_VERSION" ]]; then
        warn "Existing .venv uses Python $EXISTING_VER, recreating for $PY_VERSION"
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR with Python $PY_VERSION"
    uv venv --python "$PY_VERSION" "$VENV_DIR"
else
    log "Venv already exists at $VENV_DIR (Python $PY_VERSION)"
fi

# ----------------------------------------------------------------------------
# 3. Install dev dependencies into the venv.
# ----------------------------------------------------------------------------

log "Installing project + dev dependencies"
VIRTUAL_ENV="$VENV_DIR" uv pip install -e ".[dev]"

# ----------------------------------------------------------------------------
# 4. Build the PyO3 extension into the venv via maturin.
# ----------------------------------------------------------------------------

log "Building PyO3 extension (maturin develop)"
VIRTUAL_ENV="$VENV_DIR" "$VENV_DIR/bin/maturin" develop

# ----------------------------------------------------------------------------
# 5. Build the Rust workspace.
# ----------------------------------------------------------------------------
#
# PyO3's build script invokes whatever `python3` it finds on PATH. Without
# this, cargo build picks up the system interpreter (which may be newer than
# PyO3 supports) instead of the pinned venv interpreter. Pin PYO3_PYTHON and
# put the venv on PATH so every PyO3 crate in the workspace agrees.
#
# PyO3 feature unification: rocket-surgeon-python uses `extension-module`
# (suppresses libpython linking) while rocket-surgeon-worker uses
# `auto-initialize` (requires libpython linking). Cargo unifies features
# across a single workspace build, so the worker fails to link. Same
# workaround as xtask::test — build them in two passes.

export PATH="$VENV_DIR/bin:$PATH"
export PYO3_PYTHON="$VENV_DIR/bin/python"
export VIRTUAL_ENV="$VENV_DIR"

# rocket-surgeon-python is a cdylib whose `pyo3/extension-module` feature
# is injected by maturin (see pyproject.toml [tool.maturin].features). A
# plain `cargo build` doesn't know about that, so the cdylib fails to link
# libpython. Maturin already built it in step 4 — exclude it here.
log "Building Rust workspace (excluding PyO3 crates)"
cargo build --workspace \
    --exclude rocket-surgeon-worker \
    --exclude rocket-surgeon-python

log "Building rocket-surgeon-worker (separate pass for PyO3 feature isolation)"
cargo build -p rocket-surgeon-worker

# ----------------------------------------------------------------------------
# 6. Smoke checks.
# ----------------------------------------------------------------------------

log "Smoke check: daemon binary"
[[ -x "$DAEMON_BIN" ]] || die "daemon binary missing at $DAEMON_BIN"

log "Smoke check: Python extension imports"
VIRTUAL_ENV="$VENV_DIR" "$VENV_DIR/bin/python" -c 'import rocket_surgeon._rs; print(f"rocket_surgeon._rs OK from {rocket_surgeon._rs.__file__}")'

# ----------------------------------------------------------------------------
# 7. Install lefthook git hooks (idempotent — lefthook handles its own state).
# ----------------------------------------------------------------------------

if [[ -d "$REPO_ROOT/.git" ]]; then
    log "Installing lefthook git hooks"
    lefthook install
fi

# ----------------------------------------------------------------------------
# Done.
# ----------------------------------------------------------------------------

cat <<EOF

$(printf '\033[1;32m==>\033[0m') Bootstrap complete.

Activate the venv:
    source .venv/bin/activate

Run Python unit tests:
    pytest python/tests/ -v

Run the end-to-end lifecycle test:
    python tests/test_e2e_lifecycle.py

Run the full CI checks:
    cargo xtask ci
EOF
