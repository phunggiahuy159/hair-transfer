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
#   ./setup.sh --flux          # ALSO set up FLUX.1-Kontext (Method 2): .venv-flux + weights
#   ./setup.sh --sam-inpaint   # ALSO set up SAM3 for Method 3 (segment-then-inpaint); needs --flux
#
# FLUX.1-Kontext runs in a SEPARATE virtual environment (.venv-flux) with a modern
# diffusers, because it is incompatible with the vendored diffusers 0.23.1 Stable-Hair
# uses. It is opt-in via --flux (the model download is large, ~tens of GB).
#
# Methods 2 (prompt-driven edit) and 3 (SAM3 hair mask + reference inpaint) share ONE
# FLUX.1-Kontext-dev model — --flux installs it; --sam-inpaint only adds SAM3 (for Method 3).
# Both facebook/sam3 and black-forest-labs/FLUX.1-Kontext-dev are GATED on Hugging Face:
# accept their licenses and run `huggingface-cli login` first.
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

# FLUX model repos (kept in sync with configs/kontext.yaml). Methods 2 & 3 share Kontext.
KONTEXT_REPO="black-forest-labs/FLUX.1-Kontext-dev"
SAM3_REPO="facebook/sam3"

SKIP_ENV=0
SKIP_WEIGHTS=0
WITH_FLUX=0
WITH_SAM_INPAINT=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --skip-env)     SKIP_ENV=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --flux)         WITH_FLUX=1 ;;
    --sam-inpaint)  WITH_SAM_INPAINT=1 ;;
    --force)        FORCE=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# An existing venv is reused unless --force is given. This makes adding an
# environment (e.g. ./setup.sh --flux after a Method-1-only install) additive
# instead of wiping and rebuilding the venv you already have — which is unsafe
# if something is running out of it.
venv_healthy() { [ -x "$1/bin/python" ] && "$1/bin/python" -c "$2" >/dev/null 2>&1; }

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# Use sudo only when present and we're not already root (root containers often
# have neither sudo installed nor a need for it).
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

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
    $SUDO apt-get update -qq
    $SUDO apt-get install -y \
      "python${PYVER}-venv" python3-pip \
      build-essential cmake \
      libgl1 libglib2.0-0 ffmpeg \
      || echo "WARNING: some apt packages failed to install; continuing."
  else
    echo "WARNING: apt-get not found — ensure venv, a C++ toolchain, CMake and"
    echo "         libGL/libglib are installed for your distro."
  fi

  if [ "$FORCE" -eq 0 ] && venv_healthy "$VENV_DIR" "import torch, diffusers, peft, gradio"; then
    log "Reusing existing .venv (pass --force to rebuild)"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  else
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
  fi

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
# 4. FLUX.1-Kontext (Methods 2 & 3) — separate venv + weights (opt-in via --flux)
# ---------------------------------------------------------------------------
if [ "$WITH_FLUX" -eq 1 ]; then
  if [ "$SKIP_ENV" -eq 0 ]; then
    if [ "$FORCE" -eq 0 ] && venv_healthy "$FLUX_VENV_DIR" "from diffusers import FluxKontextPipeline, FluxKontextInpaintPipeline"; then
      log "Reusing existing .venv-flux (pass --force to rebuild)"
      # shellcheck disable=SC1091
      source "$FLUX_VENV_DIR/bin/activate"
    else
      log "Creating FLUX virtual environment at .venv-flux"
      rm -rf "$FLUX_VENV_DIR"
      "$PYTHON_BIN" -m venv "$FLUX_VENV_DIR"
      # shellcheck disable=SC1091
      source "$FLUX_VENV_DIR/bin/activate"

      log "Upgrading pip / setuptools / wheel (flux env)"
      python -m pip install --upgrade pip setuptools wheel

      log "Installing FLUX requirements (modern diffusers + CUDA 12 torch — several GB)"
      python -m pip install -r flux/requirements-flux.txt
    fi

    log "Verifying the FLUX install"
    python - <<'PY'
import torch, diffusers
from diffusers import FluxKontextPipeline, FluxKontextInpaintPipeline  # noqa: F401
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
    log "Pre-downloading FLUX.1-Kontext weights ($KONTEXT_REPO) into the HF cache"
    [ -d "$FLUX_VENV_DIR" ] && source "$FLUX_VENV_DIR/bin/activate"
    if ! huggingface-cli download "$KONTEXT_REPO" >/dev/null; then
      cat >&2 <<EOF

WARNING: FLUX.1-Kontext download failed. The model is gated — accept the license
at https://huggingface.co/$KONTEXT_REPO and log in, then retry:

  source .venv-flux/bin/activate
  huggingface-cli login
  huggingface-cli download $KONTEXT_REPO

(Or just let it download on first run of flux/kontext_server.py.)
EOF
    fi
    deactivate || true
  else
    log "Skipping FLUX weight download (--skip-weights)"
  fi
else
  log "Skipping FLUX.1-Kontext setup (pass --flux to enable Methods 2 & 3)"
fi

# ---------------------------------------------------------------------------
# 4b. SAM3 for Method 3 — adds segmentation on top of .venv-flux (opt-in via --sam-inpaint)
# ---------------------------------------------------------------------------
if [ "$WITH_SAM_INPAINT" -eq 1 ]; then
  if [ ! -d "$FLUX_VENV_DIR" ]; then
    echo "ERROR: --sam-inpaint reuses .venv-flux, which does not exist. Run with --flux first." >&2
    exit 1
  fi

  if [ "$SKIP_ENV" -eq 0 ]; then
    log "Installing SAM3 requirements into .venv-flux"
    # shellcheck disable=SC1091
    source "$FLUX_VENV_DIR/bin/activate"
    python -m pip install -r flux/requirements-sam-inpaint.txt

    log "Verifying the SAM3 install"
    python - <<'PY'
import transformers
from transformers import Sam3Model  # noqa: F401
print("transformers :", transformers.__version__)
PY
    deactivate || true
  else
    log "Skipping SAM3 environment setup (--skip-env)"
  fi

  if [ "$SKIP_WEIGHTS" -eq 0 ]; then
    log "Pre-downloading SAM3 weights ($SAM3_REPO)"
    [ -d "$FLUX_VENV_DIR" ] && source "$FLUX_VENV_DIR/bin/activate"
    if ! huggingface-cli download "$SAM3_REPO" >/dev/null; then
      cat >&2 <<EOF

WARNING: download of $SAM3_REPO failed. This model is gated — accept the license at
https://huggingface.co/$SAM3_REPO and log in, then retry:

  source .venv-flux/bin/activate
  huggingface-cli login
  huggingface-cli download $SAM3_REPO

(Or just let it download on first inpaint request to flux/kontext_server.py.)
EOF
    fi
    deactivate || true
  else
    log "Skipping SAM3 weight download (--skip-weights)"
  fi
else
  log "Skipping SAM3 setup (pass --sam-inpaint to enable Method 3)"
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

For the Kontext methods (Method 2 prompt-edit + Method 3 SAM3 inpaint), start the
single shared worker in the other venv (or use ./start_demo.sh which launches it):

  .venv-flux/bin/python flux/kontext_server.py       # http://127.0.0.1:8987

It loads FLUX.1-Kontext-dev once and serves both methods (SAM3 loads lazily for Method 3).
EOF
