# Hair Transfer — setup status & next steps (handoff)

Handoff note for the next session. Captures what works, what's been fixed, and the
exact remaining steps to finish verifying Methods 2 & 3.

_Environment used:_ RunPod container, **NVIDIA A40 (46 GB)**, driver **570.195.03
(CUDA 12.8)**, Python 3.11, running as **root** (no `sudo`). Repo: `/workspace/hair-transfer`.

## Status

| Method | State |
|---|---|
| **1 · Stable-Hair** (`.venv`, SD1.5) | ✅ **Working & verified.** `python infer_full.py` produced a correct 4-panel `output/0.jpg` (source · bald · reference · transferred). |
| **2 · FLUX Kontext** (`.venv-flux`) | ⏳ Env built, weights downloaded. **Worker not yet verified** — blocked on the torch/driver fix below. |
| **3 · SAM3 + Kontext** (`.venv-flux`) | ⏳ Same as Method 2; SAM3 weights present, loads lazily on first `/inpaint_transfer`. |

## What was fixed (committed)

1. **`diffusers/models/` was missing from the repo** — the `.gitignore` rule `models/`
   (meant for the downloaded weights) also matched the vendored `diffusers/models/`
   subpackage, so it was never committed. Method 1 crashed with
   `ModuleNotFoundError: No module named 'diffusers.models'`. Restored the subpackage
   and anchored the ignore rule to `/models/`. Also ignore `.venv-flux/`.
2. **`setup.sh` used `sudo`** — root containers have none. Now uses `sudo` only when present.
3. **`setup.sh` `--flux` wiped the main `.venv`** — venv builds are now idempotent
   (reuse a healthy venv unless `--force`), so adding the FLUX env is additive.
4. **`huggingface-cli` removed in `huggingface_hub` 1.x** — `setup.sh` now resolves
   `hf` (falls back to `huggingface-cli`). The FLUX env installs hub 1.18.0.
5. **torch/driver mismatch** — `flux/requirements-flux.txt` pinned to `torch==2.6.0+cu124`
   (was `torch>=2.5.0` + only `--extra-index-url`, which let pip grab the default PyPI
   torch that bundles **CUDA 13** and fails on this CUDA-12.8 driver:
   `"The NVIDIA driver on your system is too old (found version 12080)"`).

## Already done (state on disk, NOT in git — they're large/cached)

- `.venv` built (cu118 torch 2.2.2) and **Method 1 weights** in `./models/` (gdown).
- `.venv-flux` built. **Gated weights downloaded** into the HF cache (~66 GB total):
  `black-forest-labs/FLUX.1-Kontext-dev` and `facebook/sam3`.
- HF token saved at `~/.cache/huggingface/token` (account has accepted both licenses).
- **In progress at handoff:** reinstalling the correct torch into `.venv-flux`:
  `.venv-flux/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0`
  (this is what `requirements-flux.txt` now pins; verify it finished).

## Next steps — verify Methods 2 & 3

```bash
cd /workspace/hair-transfer

# 0. Confirm the cu124 torch reinstall finished and CUDA is visible:
.venv-flux/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   expect: 2.6.0+cu124 True      (NOT 2.x+cu130, NOT False)

# 1. Start the shared FLUX.1-Kontext worker (loads the model once; ~1-2 min):
.venv-flux/bin/python flux/kontext_server.py > /tmp/kontext_server.log 2>&1 &
#    wait for ready:
curl -s http://127.0.0.1:8987/health        # -> {"ready": true, ...}

# 2. Method 2 — prompt-driven edit:
curl -s -X POST http://127.0.0.1:8987/transfer \
  -F image1=@test_imgs/ID/0.jpg -F image2=@test_imgs/Ref/0.jpg \
  | python -c "import sys,json,base64;open('/tmp/m2.png','wb').write(base64.b64decode(json.load(sys.stdin)['result']));print('wrote /tmp/m2.png')"

# 3. Method 3 — SAM3 hair mask + reference inpaint (SAM3 loads lazily on first call):
curl -s -X POST http://127.0.0.1:8987/inpaint_transfer \
  -F image1=@test_imgs/ID/0.jpg -F image2=@test_imgs/Ref/0.jpg -F mask_prompt=hair \
  | python -c "import sys,json,base64;d=json.load(sys.stdin);open('/tmp/m3.png','wb').write(base64.b64decode(d['result']));open('/tmp/m3_mask.png','wb').write(base64.b64decode(d['mask']));print('wrote /tmp/m3.png + mask')"

# 4. Eyeball /tmp/m2.png and /tmp/m3.png (+ mask). Then the whole-stack launcher:
./start_demo.sh        # starts worker + Gradio UI on :8986 (all three methods)
```

If the worker reports CUDA false or "driver too old", the torch reinstall (step 0) did
not take — re-run it, or `./setup.sh --flux --force` now that requirements-flux.txt is fixed.

VRAM note: A40 (46 GB) fits Method 1 (~8 GB) and FLUX bf16 (~24 GB+) fine. For smaller
cards set `low_vram: true` in `configs/kontext.yaml` (CPU offload).
