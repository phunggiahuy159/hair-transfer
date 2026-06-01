#!/usr/bin/env bash
#
# Launch the Hair Transfer demo with BOTH methods available.
#
# Starts the FLUX.2 klein worker (in .venv-flux) in the background, waits for it
# to become healthy, then launches the Gradio app (in .venv). The Gradio method
# selector then exposes Stable-Hair (Method 1) and FLUX.2 klein (Method 2).
#
# If you only want Stable-Hair, just run:  .venv/bin/python gradio_demo_full.py
#
# Usage:
#   ./start_demo.sh                 # start FLUX worker + Gradio app
#   ./start_demo.sh --no-flux       # Gradio app only (Stable-Hair)

set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

VENV_DIR="${REPO_DIR}/.venv"
FLUX_VENV_DIR="${REPO_DIR}/.venv-flux"

# Read the worker host/port from the config so this stays in sync.
read_cfg() {
  python3 - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/flux_klein.yaml"))
print(cfg["server"][sys.argv[1]])
PY
}

WITH_FLUX=1
[ "${1:-}" = "--no-flux" ] && WITH_FLUX=0

FLUX_PID=""
cleanup() { [ -n "$FLUX_PID" ] && kill "$FLUX_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$WITH_FLUX" -eq 1 ]; then
  if [ ! -d "$FLUX_VENV_DIR" ]; then
    echo "WARNING: .venv-flux not found. Run ./setup.sh --flux first, or use --no-flux." >&2
  else
    HOST="$(read_cfg host)"
    PORT="$(read_cfg port)"
    echo "==> Starting FLUX.2 klein worker on http://${HOST}:${PORT} (loading 9B model — may take a while)"
    "$FLUX_VENV_DIR/bin/python" flux/flux_server.py &
    FLUX_PID=$!

    echo "==> Waiting for the FLUX worker to become healthy ..."
    for _ in $(seq 1 120); do
      if curl -sf "http://${HOST}:${PORT}/health" | grep -q '"ready": *true'; then
        echo "==> FLUX worker ready."
        break
      fi
      # Bail early if the worker process died.
      kill -0 "$FLUX_PID" 2>/dev/null || { echo "ERROR: FLUX worker exited during startup." >&2; exit 1; }
      sleep 5
    done
  fi
fi

echo "==> Starting Gradio app on http://0.0.0.0:8986"
"$VENV_DIR/bin/python" gradio_demo_full.py
