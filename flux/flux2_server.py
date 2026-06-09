"""FLUX.2-dev (4-bit) multi-reference hair-transfer worker.

Replaces the FLUX.1-Kontext "/transfer" method. Runs in the .venv-flux environment and
MUST be launched as `python flux/flux2_server.py` so that `import diffusers` resolves to
the modern pip package, not the vendored diffusers 0.23.1 at the repo root.

FLUX.2-dev supports NATIVE multi-reference editing, so the source (ID) image and the hair
reference are passed together as a list (image=[source, reference]) with a text instruction
— no side-by-side canvas, no cropping, and the source's framing/aspect ratio is preserved.

  POST /transfer   image1 (ID/source) + image2 (hair reference) -> edited image1
  GET  /health     readiness

The model is the diffusers 4-bit (bitsandbytes NF4) build of FLUX.2-dev — A100/Ampere has
no FP8 tensor cores, so 4-bit (not FP8) is the build that both fits 40GB and is accelerated.
It is loaded once with CPU offload (peak ~30GB on a single 40GB GPU).
"""
import base64
import io
import os

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("FLUX2_CONFIG", os.path.join(REPO_ROOT, "configs", "flux2.yaml"))


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="FLUX.2-dev multi-reference hair-transfer worker")

PIPE = None  # Flux2Pipeline, loaded once at startup.


@app.on_event("startup")
def _startup():
    global PIPE
    from diffusers import Flux2Pipeline

    repo = CONFIG["flux2_repo"]
    print(f"[flux2] loading {repo} (4-bit) ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)

    # Turbo distillation LoRA (DMD2): runs the dev model in ~8 steps instead of ~28 at
    # comparable quality. Stacked on the 4-bit base via PEFT (not fused — fusing into a
    # bnb-4bit base isn't supported, but unfused inference works fine).
    lora_repo = CONFIG.get("turbo_lora_repo")
    if lora_repo:
        print(f"[flux2] loading Turbo LoRA {lora_repo} ...", flush=True)
        pipe.load_lora_weights(lora_repo, weight_name=CONFIG.get("turbo_lora_file"))

    if CONFIG.get("low_vram", True):
        # Keep only the active module on the GPU; needed so the 32B model + LoRA +
        # activations fit a single 40GB card (peak ~36GB with the Turbo LoRA).
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)
    PIPE = pipe
    print("[flux2] model ready.", flush=True)


@app.get("/health")
def health():
    return {"ready": PIPE is not None, "model": CONFIG["flux2_repo"]}


def _load_image(upload_bytes):
    return Image.open(io.BytesIO(upload_bytes)).convert("RGB")


def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_generator(seed):
    if int(seed) >= 0:
        return torch.Generator(device=DEVICE).manual_seed(int(seed))
    return None


@app.post("/transfer")
async def transfer(
    image1: UploadFile = File(...),   # source / ID image (the person to keep)
    image2: UploadFile = File(...),   # hair reference image
    prompt: str = Form(None),
    steps: int = Form(None),
    guidance_scale: float = Form(None),
    seed: int = Form(-1),
):
    if PIPE is None:
        return JSONResponse({"error": "model still loading"}, status_code=503)

    prompt = prompt or CONFIG["edit_prompt"]
    steps = int(steps or CONFIG["num_inference_steps"])
    guidance_scale = float(guidance_scale if guidance_scale is not None else CONFIG["guidance_scale"])

    # The Turbo LoRA needs its specific noise schedule; when configured, the sigmas drive
    # the step count (ignore any client step override to keep the distilled schedule intact).
    extra = {}
    sigmas = CONFIG.get("turbo_sigmas")
    if sigmas:
        extra["sigmas"] = [float(s) for s in sigmas]
        steps = len(sigmas)

    source = _load_image(await image1.read())
    reference = _load_image(await image2.read())

    # Multi-reference edit: image[0] = source (kept), image[1] = hair reference.
    # FLUX.2 derives the output size from the first image (preserves its aspect, ~1MP).
    out = PIPE(
        image=[source, reference],
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_make_generator(seed),
        **extra,
    ).images[0]

    return JSONResponse({"result": _png_b64(out)})


if __name__ == "__main__":
    host = os.environ.get("FLUX2_HOST", CONFIG["server"]["host"])
    port = int(os.environ.get("FLUX2_PORT", CONFIG["server"]["port"]))
    uvicorn.run(app, host=host, port=port)
