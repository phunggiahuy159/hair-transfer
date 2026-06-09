"""Merged FLUX.1-Kontext worker for Methods 2 & 3.

Runs in the .venv-flux environment and MUST be launched as
`python flux/kontext_server.py` so that `import diffusers` / `import transformers`
resolve to the modern pip packages, not the vendored diffusers 0.23.1 at the repo root.

It loads FLUX.1-Kontext-dev ONCE and serves two methods that share the same weights
(via FluxKontextInpaintPipeline.from_pipe), so only one big model is ever loaded:

  POST /transfer          Method 2 "FLUX Kontext": prompt-driven edit. The source (ID) and
                          reference images are concatenated side by side, Kontext edits the
                          canvas, and the (left) ID half is cropped out and returned.
  POST /inpaint_transfer  Method 3 "SAM3 + Inpaint": SAM3 segments the hair on the source,
                          then Kontext repaints only that region from the reference image.
  GET  /health            readiness + which extras are available.

SAM3 (needed only for Method 3) is loaded lazily, so Method 2 works with just Kontext.
Both facebook/sam3 and black-forest-labs/FLUX.1-Kontext-dev are gated — accept their
licenses on Hugging Face and run `huggingface-cli login` before first use.
"""
import base64
import io
import os

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

# Repo root is the parent of this file's directory (flux/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get(
    "KONTEXT_CONFIG", os.path.join(REPO_ROOT, "configs", "kontext.yaml")
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="FLUX.1-Kontext hair-transfer worker (Methods 2 & 3)")

# Loaded once at startup; the two pipelines share the same underlying modules.
EDIT_PIPE = None      # FluxKontextPipeline           (Method 2)
INPAINT_PIPE = None   # FluxKontextInpaintPipeline     (Method 3)
# SAM3 is loaded lazily on first inpaint request.
SAM_MODEL = None
SAM_PROCESSOR = None
SAM_ERROR = None


@app.on_event("startup")
def _startup():
    global EDIT_PIPE, INPAINT_PIPE
    from diffusers import FluxKontextPipeline, FluxKontextInpaintPipeline

    repo = CONFIG["kontext_repo"]
    print(f"[kontext] loading {repo} ...")
    edit = FluxKontextPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    if CONFIG.get("low_vram", False):
        # On memory-constrained GPUs (e.g. a single 40GB card), enable_model_cpu_offload
        # keeps only the active module on the GPU (peak ~25GB instead of ~38GB resident).
        # IMPORTANT: sharing components via from_pipe() severs the offload eviction chain
        # — modules load onto the GPU during a run but are never offloaded back, so memory
        # accumulates to the full model size and OOMs. Under low_vram we therefore load an
        # independent inpaint pipeline (weights are cached on disk; CPU RAM holds both) and
        # offload each separately so both methods keep a working offload chain.
        edit.enable_model_cpu_offload()
        inpaint = FluxKontextInpaintPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        inpaint.enable_model_cpu_offload()
        print("[kontext] model ready (edit + inpaint, CPU offload).")
    else:
        edit.to(DEVICE)
        # Reuse the same transformer / VAE / text encoders — no extra weights loaded.
        inpaint = FluxKontextInpaintPipeline.from_pipe(edit)
        print("[kontext] model ready (edit + inpaint share weights).")

    EDIT_PIPE = edit
    INPAINT_PIPE = inpaint


def _get_sam():
    """Lazily load SAM3 (Method 3 only). Returns (model, processor) or raises."""
    global SAM_MODEL, SAM_PROCESSOR, SAM_ERROR
    if SAM_MODEL is not None:
        return SAM_MODEL, SAM_PROCESSOR
    if SAM_ERROR is not None:
        raise RuntimeError(SAM_ERROR)
    try:
        from transformers import Sam3Model, Sam3Processor
        print(f"[kontext] loading SAM3 ({CONFIG['sam3_repo']}) ...")
        SAM_MODEL = Sam3Model.from_pretrained(CONFIG["sam3_repo"]).to(DEVICE)
        SAM_PROCESSOR = Sam3Processor.from_pretrained(CONFIG["sam3_repo"])
        print("[kontext] SAM3 ready.")
        return SAM_MODEL, SAM_PROCESSOR
    except Exception as e:  # noqa: BLE001
        SAM_ERROR = (
            f"SAM3 is unavailable ({e}). Install it with "
            "`./setup.sh --sam-inpaint` (transformers bump + facebook/sam3 weights)."
        )
        raise RuntimeError(SAM_ERROR)


@app.get("/health")
def health():
    return {
        "ready": EDIT_PIPE is not None,
        "kontext": CONFIG["kontext_repo"],
        "sam3_loaded": SAM_MODEL is not None,
        "sam3_error": SAM_ERROR,
    }


def _read_image(upload_bytes, size):
    # Square resize — used by Method 3 (inpaint works on a single image + its mask).
    return Image.open(io.BytesIO(upload_bytes)).convert("RGB").resize((size, size))


def _load_image(upload_bytes):
    return Image.open(io.BytesIO(upload_bytes)).convert("RGB")


def _square(img, size):
    """Center-crop to a `size`x`size` square WITHOUT distorting aspect ratio
    (ImageOps.fit crops the long side instead of squishing). Keeping both panels
    square gives a 2:1 side-by-side canvas, which is close to FLUX.1-Kontext's
    trained ~1MP buckets — wide canvases (e.g. 2.5:1 from raw 16:9 + portrait
    inputs) fall outside its range and the edit barely applies."""
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS)


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
    image1: UploadFile = File(...),   # source / ID image (becomes the LEFT half)
    image2: UploadFile = File(...),   # reference hair image (RIGHT half)
    prompt: str = Form(None),
    steps: int = Form(None),
    guidance_scale: float = Form(None),
    seed: int = Form(-1),
    size: int = Form(None),
):
    if EDIT_PIPE is None:
        return JSONResponse({"error": "model still loading"}, status_code=503)

    prompt = prompt or CONFIG["edit_prompt"]
    steps = int(steps or CONFIG["num_inference_steps"])
    guidance_scale = float(guidance_scale if guidance_scale is not None else CONFIG["guidance_scale"])
    size = int(size or CONFIG["size"])

    source = _load_image(await image1.read())
    reference = _load_image(await image2.read())

    # Concatenate side by side: ID on the LEFT, reference on the RIGHT. Each panel is a
    # center-cropped square (no aspect distortion), giving a clean 2:1 canvas and an exact
    # 50% split so the left (source) half can be cropped back reliably.
    src = _square(source, size)
    ref = _square(reference, size)
    canvas = Image.new("RGB", (2 * size, size))
    canvas.paste(src, (0, 0))
    canvas.paste(ref, (size, 0))

    out = EDIT_PIPE(
        image=canvas,
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_make_generator(seed),
    ).images[0]

    # Kontext returns a preferred-bucket size; normalize back to the canvas, then crop the
    # LEFT (source) square.
    out = out.resize((2 * size, size))
    result = out.crop((0, 0, size, size))
    return JSONResponse({"result": _png_b64(result)})


@app.post("/inpaint_transfer")
async def inpaint_transfer(
    image1: UploadFile = File(...),   # source / ID image
    image2: UploadFile = File(...),   # reference hair image
    prompt: str = Form(None),
    mask_prompt: str = Form(None),
    strength: float = Form(None),
    steps: int = Form(None),
    guidance_scale: float = Form(None),
    seed: int = Form(-1),
    size: int = Form(None),
    mask_blur: int = Form(None),
    mask_dilate: int = Form(None),
):
    if INPAINT_PIPE is None:
        return JSONResponse({"error": "model still loading"}, status_code=503)

    try:
        sam_model, sam_processor = _get_sam()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    prompt = prompt or CONFIG["inpaint_prompt"]
    mask_prompt = mask_prompt or CONFIG["mask_prompt"]
    strength = float(strength if strength is not None else CONFIG["strength"])
    steps = int(steps or CONFIG["num_inference_steps"])
    guidance_scale = float(guidance_scale if guidance_scale is not None else CONFIG["guidance_scale"])
    size = int(size or CONFIG["size"])
    mask_blur = int(mask_blur if mask_blur is not None else CONFIG["mask_blur"])
    mask_dilate = int(mask_dilate if mask_dilate is not None else CONFIG["mask_dilate"])

    source = _read_image(await image1.read(), size)
    reference = _read_image(await image2.read(), size)

    # 1. Segment the requested region with SAM3.
    inputs = sam_processor(images=source, text=mask_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    res = sam_processor.post_process_instance_segmentation(
        outputs,
        threshold=float(CONFIG.get("score_threshold", 0.5)),
        mask_threshold=float(CONFIG.get("mask_threshold", 0.5)),
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = res["masks"].cpu().numpy()  # (N, H, W)
    if masks.shape[0] == 0:
        return JSONResponse(
            {"error": f"SAM3 found no '{mask_prompt}' in the source image. Try a different mask prompt."},
            status_code=422,
        )
    mask_u8 = (np.any(masks > 0, axis=0) * 255).astype(np.uint8)

    # 2. Refine the mask: dilate (cover hair edges) then resize to the working size.
    if mask_dilate and mask_dilate > 0:
        import cv2
        k = int(mask_dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    mask_pil = Image.fromarray(mask_u8).convert("L").resize((size, size), Image.NEAREST)
    blurred = INPAINT_PIPE.mask_processor.blur(mask_pil, blur_factor=mask_blur) if mask_blur > 0 else mask_pil

    # 3. Reference-guided inpaint of the masked region only.
    result = INPAINT_PIPE(
        prompt=prompt,
        image=source,
        mask_image=blurred,
        image_reference=reference,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_make_generator(seed),
    ).images[0]

    return JSONResponse({"result": _png_b64(result), "mask": _png_b64(mask_pil.convert("RGB"))})


if __name__ == "__main__":
    host = os.environ.get("KONTEXT_HOST", CONFIG["server"]["host"])
    port = int(os.environ.get("KONTEXT_PORT", CONFIG["server"]["port"]))
    uvicorn.run(app, host=host, port=port)
