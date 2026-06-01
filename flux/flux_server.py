"""FLUX.2 [klein] 9B hair-transfer worker (Method 2).

Runs in the dedicated `.venv-flux` environment with a modern diffusers (>= 0.38).
It MUST be launched as `python flux/flux_server.py` (i.e. with this file's directory,
`flux/`, at the head of sys.path) so that `import diffusers` resolves to the modern
pip-installed package and NOT the vendored `diffusers 0.23.1` at the repo root.

The model is loaded once at startup; the Gradio demo (running in the other venv)
reaches this worker over local HTTP.

Endpoints:
  GET  /health    -> {"ready": bool, "model": str}
  POST /transfer  -> multipart(image1, image2, prompt, steps, guidance_scale, seed, size)
                     returns the edited image as PNG bytes.
"""
import io
import os

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

# Repo root is the parent of this file's directory (flux/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get(
    "FLUX_CONFIG", os.path.join(REPO_ROOT, "configs", "flux_klein.yaml")
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

app = FastAPI(title="FLUX.2 klein hair-transfer worker")

# Populated by startup; None until the model finishes loading.
PIPE = None


def _build_pipeline():
    """Load Flux2KleinPipeline (or the KV variant) per the config."""
    repo = CONFIG["model_repo"]
    low_vram = bool(CONFIG.get("low_vram", False))

    # The KV variant ships a different pipeline class and uses few-step inference.
    if "kv" in repo.lower():
        from diffusers import Flux2KleinKVPipeline as PipelineCls
    else:
        from diffusers import Flux2KleinPipeline as PipelineCls

    pipe = PipelineCls.from_pretrained(repo, torch_dtype=torch.bfloat16)
    if low_vram:
        # Streams modules between CPU and GPU; slower but fits smaller cards.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


@app.on_event("startup")
def _startup():
    global PIPE
    print(f"[flux] loading {CONFIG['model_repo']} (low_vram={CONFIG.get('low_vram', False)}) ...")
    PIPE = _build_pipeline()
    print("[flux] model ready.")


@app.get("/health")
def health():
    return {"ready": PIPE is not None, "model": CONFIG["model_repo"]}


@app.post("/transfer")
async def transfer(
    image1: UploadFile = File(...),   # the person / ID image
    image2: UploadFile = File(...),   # the hair reference image
    prompt: str = Form(None),
    steps: int = Form(None),
    guidance_scale: float = Form(None),
    seed: int = Form(-1),
    size: int = Form(None),
):
    if PIPE is None:
        return JSONResponse({"error": "model still loading"}, status_code=503)

    prompt = prompt or CONFIG["default_prompt"]
    steps = int(steps or CONFIG["num_inference_steps"])
    guidance_scale = float(guidance_scale if guidance_scale is not None else CONFIG["guidance_scale"])
    size = int(size or CONFIG["size"])

    id_img = Image.open(io.BytesIO(await image1.read())).convert("RGB").resize((size, size))
    ref_img = Image.open(io.BytesIO(await image2.read())).convert("RGB").resize((size, size))

    generator = None
    if int(seed) >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    call_kwargs = dict(
        prompt=prompt,
        image=[id_img, ref_img],
        num_inference_steps=steps,
        generator=generator,
    )
    # The KV pipeline does not take a guidance_scale argument.
    if "kv" not in CONFIG["model_repo"].lower():
        call_kwargs["guidance_scale"] = guidance_scale

    result = PIPE(**call_kwargs).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    host = os.environ.get("FLUX_HOST", CONFIG["server"]["host"])
    port = int(os.environ.get("FLUX_PORT", CONFIG["server"]["port"]))
    uvicorn.run(app, host=host, port=port)
