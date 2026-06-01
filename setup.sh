#!/usr/bin/env bash
#
# Stable-Hair setup script
# ------------------------
# Creates a Python virtual environment, installs the (CUDA 11.8) dependency
# stack, and downloads the pretrained Stage-1 / Stage-2 weights.
#
# Usage:
#   ./setup.sh                 # full setup (env + weights) for Stable-Hair (Method 1)
#   ./setup.sh --skip-weights  # only build the environment
#   ./setup.sh --skip-env      # only download the weights (env must exist)
#   ./setup.sh --flux          # ALSO set up FLUX.2 klein (Method 2): .venv-flux + weights
#
# FLUX.2 [klein] 9B runs in a SEPARATE virtual environment (.venv-flux) with a modern
# diffusers, because it is incompatible with the vendored diffusers 0.23.1 Stable-Hair
# uses. It is opt-in via --flux (the model download is large, ~tens of GB).
#
# Tested on: Ubuntu, Python 3.10, NVIDIA driver >= 520 (RTX A6000).
# The wheels are CUDA 11.8 builds; any reasonably recent NVIDIA driver works.

set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
VENV_DIR="${REPO_DIR}/.venv"
FLUX_VENV_DIR="${REPO_DIR}/.venv-flux"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Google Drive folder published by the authors (contains stage1/ and stage2/).
DRIVE_URL="https://drive.google.com/drive/folders/1E-8Udfw8S8IorCWhBgS4FajIbqlrWRbQ"

# FLUX.2 klein model repo (kept in sync with configs/flux_klein.yaml).
FLUX_MODEL_REPO="black-forest-labs/FLUX.2-klein-9B"

SKIP_ENV=0
SKIP_WEIGHTS=0
WITH_FLUX=0
for arg in "$@"; do
  case "$arg" in
    --skip-env)     SKIP_ENV=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --flux)         WITH_FLUX=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
log "Checking prerequisites"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' not found. Install Python 3.10+ first." >&2
  exit 1
fi
"$PYTHON_BIN" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found — a CUDA GPU is required to run the models."
fi

# ---------------------------------------------------------------------------
# 2. Virtual environment + dependencies
# ---------------------------------------------------------------------------
if [ "$SKIP_ENV" -eq 0 ]; then
  log "Installing system dependencies"
  # python venv module (Debian splits it out), a C/C++ toolchain + CMake for the
  # dlib wheel, and the shared libs OpenCV needs at runtime (libGL / libglib).
  PYVER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y \
      "python${PYVER}-venv" python3-pip \
      build-essential cmake \
      libgl1 libglib2.0-0 ffmpeg \
      || echo "WARNING: some apt packages failed to install; continuing."
  else
    echo "WARNING: apt-get not found — ensure venv, a C++ toolchain, CMake and"
    echo "         libGL/libglib are installed for your distro."
  fi

  log "Creating virtual environment at .venv"
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Upgrading pip / setuptools / wheel"
  python -m pip install --upgrade pip setuptools wheel

  log "Installing requirements (this pulls the CUDA 11.8 torch stack — a few GB)"
  # requirements.txt carries the pytorch cu118 --extra-index-url on its first line.
  python -m pip install -r requirements.txt

  log "Verifying the install"
  python - <<'PY'
import torch, torchvision, diffusers, transformers, gradio
print("torch        :", torch.__version__)
print("torchvision  :", torchvision.__version__)
print("diffusers    :", diffusers.__version__, "(vendored copy in ./diffusers is used at runtime)")
print("transformers :", transformers.__version__)
print("gradio       :", gradio.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
PY
else
  log "Skipping environment setup (--skip-env)"
  # shellcheck disable=SC1091
  [ -d "$VENV_DIR" ] && source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------------------
# 3. Pretrained weights
# ---------------------------------------------------------------------------
if [ "$SKIP_WEIGHTS" -eq 0 ]; then
  log "Downloading pretrained weights from Google Drive"
  mkdir -p models
  # Ensure gdown is available even if --skip-env was used.
  python -m pip install --quiet "gdown>=5.2.0" || true

  if [ -f models/stage1/pytorch_model.bin ] && \
     [ -f models/stage2/pytorch_model.bin ] && \
     [ -f models/stage2/pytorch_model_1.bin ] && \
     [ -f models/stage2/pytorch_model_2.bin ]; then
    echo "Weights already present in ./models — skipping download."
  else
    TMP_DL="$(mktemp -d)"
    # --folder mirrors the remote stage1/ and stage2/ directories.
    if python -m gdown --folder "$DRIVE_URL" -O "$TMP_DL" --remaining-ok; then
      # Find the stage1 / stage2 directories wherever gdown placed them.
      for stage in stage1 stage2; do
        src="$(find "$TMP_DL" -type d -name "$stage" | head -n1)"
        if [ -n "$src" ]; then
          mkdir -p "models/$stage"
          cp -rn "$src"/. "models/$stage"/
        fi
      done
      rm -rf "$TMP_DL"
    else
      rm -rf "$TMP_DL"
      cat >&2 <<EOF

WARNING: automatic download failed (Google Drive sometimes rate-limits large
folders). Download manually instead:

  pip install gdown
  gdown --folder $DRIVE_URL

Then arrange the files so the layout is:

  models/stage1/pytorch_model.bin
  models/stage2/pytorch_model.bin      # Hair Extractor / encoder
  models/stage2/pytorch_model_1.bin    # Adapter
  models/stage2/pytorch_model_2.bin    # Latent IdentityNet (controlnet)
EOF
    fi
  fi

  log "Weights status"
  for f in stage1/pytorch_model.bin stage2/pytorch_model.bin \
           stage2/pytorch_model_1.bin stage2/pytorch_model_2.bin; do
    if [ -f "models/$f" ]; then
      printf '  [ok]      models/%s\n' "$f"
    else
      printf '  [MISSING] models/%s\n' "$f"
    fi
  done
else
  log "Skipping weight download (--skip-weights)"
fi

# ---------------------------------------------------------------------------
# 4. FLUX.2 klein (Method 2) — separate venv + weights (opt-in via --flux)
# ---------------------------------------------------------------------------
if [ "$WITH_FLUX" -eq 1 ]; then
  if [ "$SKIP_ENV" -eq 0 ]; then
    log "Creating FLUX virtual environment at .venv-flux"
    rm -rf "$FLUX_VENV_DIR"
    "$PYTHON_BIN" -m venv "$FLUX_VENV_DIR"
    # shellcheck disable=SC1091
    source "$FLUX_VENV_DIR/bin/activate"

    log "Upgrading pip / setuptools / wheel (flux env)"
    python -m pip install --upgrade pip setuptools wheel

    log "Installing FLUX requirements (modern diffusers + CUDA 12 torch — several GB)"
    python -m pip install -r flux/requirements-flux.txt

    log "Verifying the FLUX install"
    python - <<'PY'
import torch, diffusers
from diffusers import Flux2KleinPipeline  # noqa: F401
print("torch     :", torch.__version__)
print("diffusers :", diffusers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU       :", torch.cuda.get_device_name(0))
PY
    deactivate || true
  else
    log "Skipping FLUX environment setup (--skip-env)"
  fi

  if [ "$SKIP_WEIGHTS" -eq 0 ]; then
    log "Pre-downloading FLUX.2 klein weights ($FLUX_MODEL_REPO) into the HF cache"
    [ -d "$FLUX_VENV_DIR" ] && source "$FLUX_VENV_DIR/bin/activate"
    if ! huggingface-cli download "$FLUX_MODEL_REPO" >/dev/null; then
      cat >&2 <<EOF

WARNING: FLUX.2 klein download failed. The model may be gated — accept the license
at https://huggingface.co/$FLUX_MODEL_REPO and log in, then retry:

  source .venv-flux/bin/activate
  huggingface-cli login
  huggingface-cli download $FLUX_MODEL_REPO

(Or just let it download on first run of flux/flux_server.py.)
EOF
    fi
    deactivate || true
  else
    log "Skipping FLUX weight download (--skip-weights)"
  fi
else
  log "Skipping FLUX.2 klein setup (pass --flux to enable Method 2)"
fi

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
log "Setup complete"
cat <<EOF

Activate the environment and run inference:

  source .venv/bin/activate
  python infer_full.py                 # writes ./output/0.jpg
  python gradio_demo_full.py           # web UI on http://0.0.0.0:8986

The base Stable Diffusion 1.5 weights download automatically from Hugging Face
on first run (repo: stable-diffusion-v1-5/stable-diffusion-v1-5).

For the FLUX.2 klein method (Method 2), start its worker in the other venv first
(or use ./start_demo.sh which launches both):

  .venv-flux/bin/python flux/flux_server.py   # serves on http://127.0.0.1:8987
EOF
