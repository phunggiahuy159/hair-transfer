#!/usr/bin/env bash
#
# Launch the Hair Transfer demo and (optionally) the FLUX.1-Kontext worker.
#
# The Gradio app (in .venv) exposes all three methods in its selector. Stable-Hair
# (Method 1) runs in-process; FLUX Kontext (Method 2) and SAM3 + Inpaint (Method 3)
# are both served by ONE worker (flux/kontext_server.py in .venv-flux) that loads
# FLUX.1-Kontext-dev once and shares it between them.
#
# Usage:
#   ./start_demo.sh            # FLUX.1-Kontext worker (8987) + Gradio app
#   ./start_demo.sh --no-flux  # Gradio app only (Stable-Hair)

set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

VENV_DIR="${REPO_DIR}/.venv"
FLUX_VENV_DIR="${REPO_DIR}/.venv-flux"
CFG="configs/kontext.yaml"

WITH_FLUX=1
[ "${1:-}" = "--no-flux" ] && WITH_FLUX=0

# Read host/port from the worker's config so this stays in sync.
read_cfg() {
  python3 - "$CFG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["server"][sys.argv[2]])
PY
}

WORKER_PID=""
cleanup() { [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$WITH_FLUX" -eq 1 ]; then
  if [ ! -d "$FLUX_VENV_DIR" ]; then
    echo "WARNING: .venv-flux not found. Run ./setup.sh --flux first, or use --no-flux." >&2
  else
    HOST="$(read_cfg host)"
    PORT="$(read_cfg port)"
    echo "==> Starting FLUX.1-Kontext worker on http://${HOST}:${PORT} (loading model — may take a while)"
    "$FLUX_VENV_DIR/bin/python" flux/kontext_server.py &
    WORKER_PID=$!

    echo "==> Waiting for the Kontext worker to become healthy ..."
    for _ in $(seq 1 120); do
      if curl -sf "http://${HOST}:${PORT}/health" | grep -q '"ready": *true'; then
        echo "==> Kontext worker ready."
        break
      fi
      # Bail early if the worker process died.
      kill -0 "$WORKER_PID" 2>/dev/null || { echo "ERROR: Kontext worker exited during startup." >&2; exit 1; }
      sleep 5
    done
  fi
fi

echo "==> Starting Gradio app on http://0.0.0.0:8986"
"$VENV_DIR/bin/python" gradio_demo_full.py
