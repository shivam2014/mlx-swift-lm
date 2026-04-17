#!/bin/bash
# ============================================================
# update_mlxserver_fork.sh — Build MLXServer from this fork
# Fork: https://github.com/shivam2014/mlx-swift-lm
# Branch: ek/tom-eric-moe-tuning
#
# Usage:
#   ./update_mlxserver_fork.sh          # pull latest + build
#   ./update_mlxserver_fork.sh --fresh  # wipe .build and rebuild
# ============================================================

set +e

# --- CONFIG ---
REPO_DIR="$HOME/mlx-env/mlx-swift-lm"
BRANCH="ek/tom-eric-moe-tuning"
BINARY="$REPO_DIR/.build/release/MLXServer"
CORES=$(sysctl -n hw.ncpu)

# --- COLORS ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

cd "$REPO_DIR" || fail "Cannot change directory to $REPO_DIR"

# --- Ensure correct branch ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    warn "On branch '$CURRENT_BRANCH', switching to '$BRANCH'"
    git checkout "$BRANCH" || fail "Checkout of $BRANCH failed"
fi

log "Branch: $BRANCH @ $(git rev-parse --short HEAD)"
log "Commit: $(git log -1 --format='%ci') — $(git log -1 --format='%s')"
echo ""

# --- Pull latest from YOUR fork ---
log "Pulling latest from your fork (fork remote)..."
git pull fork "$BRANCH" 2>&1
if [ $? -ne 0 ]; then
    warn "git pull had issues — continuing with current state"
fi
ok "Pull done. Now @ $(git rev-parse --short HEAD)"
echo ""

# --- Fresh build option ---
if [ "$1" = "--fresh" ]; then
    warn "Fresh build requested — wiping .build directory"
    rm -rf "$REPO_DIR/.build"
fi

# --- Resolve Swift packages ---
log "Resolving Swift package dependencies..."
swift package resolve 2>&1 | tail -5
if [ $? -ne 0 ]; then
    fail "swift package resolve failed"
fi
ok "Packages resolved"
echo ""

# --- Compile Metal shaders ---
# Required: MLXServer crashes on launch with "Failed to load the default metallib"
# if the Metal shader library is not compiled first.
log "Compiling Metal shaders (make metal)..."
make -C "$REPO_DIR" metal 2>&1 | tail -5
if [ $? -ne 0 ]; then
    fail "make metal failed — Metal shaders not compiled"
fi
ok "Metal shaders compiled"
echo ""

# --- Build MLXServer ---
log "Building MLXServer in release mode..."
swift build -c release --product MLXServer 2>&1 | grep -E "error:|warning: [A-Z]|Build complete|Compiling|Linking|[Ee]rror"

if [ ! -f "$BINARY" ]; then
    fail "Build failed — binary not found at $BINARY"
fi

ok "Build complete: $BINARY"
echo ""

# --- Summary ---
log "Build info"
echo ""
echo "  Binary:  $BINARY"
echo "  Size:    $(du -h "$BINARY" | cut -f1)"
echo "  Branch:  $BRANCH"
echo "  Commit:  $(git rev-parse --short HEAD) — $(git log -1 --format='%ci')"
echo ""

ok "MLXServer ready. Launch via scripts/serve_fork.py:"
echo ""
echo "  ./scripts/serve_fork.py"
echo ""
