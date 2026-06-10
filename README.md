# Hair Transfer

A virtual hair try-on toolkit that transfers a hairstyle from a **reference image**
onto a **person's photo** — with **two** interchangeable methods you can compare from a
single Gradio app.

<a href='https://arxiv.org/pdf/2407.14078'><img src='https://img.shields.io/badge/Stable--Hair-Report-red'></a>
<a href='https://huggingface.co/black-forest-labs/FLUX.2-dev'><img src='https://img.shields.io/badge/FLUX.2-dev-blue'></a>
<a href='https://huggingface.co/fal/FLUX.2-dev-Turbo'><img src='https://img.shields.io/badge/fal-Turbo%20LoRA-purple'></a>

<img src='assets/teaser_.jpg'>

## Why two methods?
Hair transfer can be approached very differently depending on the tradeoff you want
between **identity preservation**, **fidelity to the reference hairstyle**, and
**hardware**. This project bundles a classic trained ControlNet pipeline and a modern
multi-reference diffusion editor so you can pick — or compare — the right tool for a
given image.

| Method | Model | How it works | Best for | VRAM | Env |
|---|---|---|---|---|---|
| **1 · Stable-Hair** | SD 1.5 (trained) | Bald-converter ControlNet → Hair Extractor + Latent IdentityNet | Aligned 512×512 faces; no big downloads | ~8 GB | `.venv` |
| **2 · FLUX.2** | FLUX.2-dev (4-bit) + Turbo LoRA | ID + reference fed together as multi-reference; the model restyles the person's hair | In-the-wild photos; preserves framing & identity | ~36 GB | `.venv-flux` |

Both are selectable from one web UI ([`gradio_demo_full.py`](gradio_demo_full.py)).

## Methods in detail

### Method 1 — Stable-Hair (two-stage SD 1.5)
A diffusion-based framework that robustly transfers real-world hairstyles onto a face.
It is a two-stage pipeline:

1. **Bald Converter** — a Latent ControlNet trained alongside Stable Diffusion removes
   hair from the source face, producing a bald image.
2. **Hair Transfer** — a *Hair Extractor* encodes the reference hairstyle while a
   *Latent IdentityNet* preserves identity and background, transferring the target
   hairstyle with high fidelity onto the bald image.

It is tuned via sliders in the UI and works best on cropped, aligned 512×512 faces. Based
on [Stable-Hair](https://github.com/Xiaojiu-z/Stable-Hair) (see [Credits](#credits)).

<img src='assets/method.jpg'>

### Method 2 — FLUX.2 (multi-reference edit)
[FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) supports **native
multi-reference editing**, which is a much better fit for hair transfer than a
side-by-side trick: the ID image and the hair reference are passed **together** as a list
(`image=[source, reference]`) with a text instruction, and the model gives the person the
reference's hairstyle while preserving their identity **and the original framing/aspect**
— no canvas concatenation, no cropping.

The 32B model runs as the 4-bit [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit)
build (A100/Ampere has no FP8 tensor cores, so 4-bit — not FP8 — is the build that fits a
40 GB card and runs accelerated) with the [`fal/FLUX.2-dev-Turbo`](https://huggingface.co/fal/FLUX.2-dev-Turbo)
distillation LoRA so it generates in **8 steps** (≈80 s/image) instead of ~28.

> Served by a single worker ([`flux/flux2_server.py`](flux/flux2_server.py)) that loads the
> base model + Turbo LoRA **once** with CPU offload and exposes `POST /transfer`.
> **Prompt tip:** describing the *target* hair explicitly (e.g. "long thick voluminous
> curly hair") gives stronger control than a generic instruction — the UI exposes the prompt.

## Architecture
Method 1 imports a **vendored `diffusers 0.23.1`** that the model code depends on; Method 2
needs a **modern `diffusers` (≥ 0.38)** with a bf16 torch, bitsandbytes and peft. A single
Python process can't hold both, so they live in separate virtual environments and
communicate over local HTTP:

```
.venv      ── gradio_demo_full.py ── Method 1 (Stable-Hair, in-process)
                   │  HTTP
                   └─▶ flux/flux2_server.py (:8987) ── Method 2 (FLUX.2)
.venv-flux ── the FLUX.2 worker (FLUX.2-dev 4-bit + Turbo LoRA, loaded once)
```

If the worker isn't running, the FLUX.2 method shows a "start the worker" message;
Method 1 always works in-process.

## Requirements
- **OS:** Linux (tested on Ubuntu 22.04)
- **Python:** 3.10 (Method 1's cu118 wheels need 3.10/3.11; Method 2 is happy on 3.10–3.12)
- **GPU:** NVIDIA GPU + recent driver. Method 1 runs on ≥ 8 GB. Method 2 is a 32B model —
  even in 4-bit with CPU offload it peaks **~36 GB** (tested on a 40 GB A100). For smaller
  cards, the lighter 9B [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  (4-step, ~21 GB) is a drop-in alternative with a small change to the worker.
- **Disk:** ~6–15 GB for Method 1; the FLUX.2 4-bit weights + Turbo LoRA add **~35 GB**.
- **Gated models:** accept the **FLUX.2 [dev]** license on Hugging Face and `hf auth login`
  before first run (older `huggingface_hub`: `huggingface-cli login`).

## Installation

### Method 1 — Stable-Hair
[`setup.sh`](setup.sh) builds `.venv`, installs the (CUDA 11.8) stack, and downloads the
Stable-Hair weights:

```bash
git clone <this-repo> hair-transfer
cd hair-transfer
./setup.sh                  # .venv + Stable-Hair weights
# ./setup.sh --skip-weights # build the environment only
# ./setup.sh --skip-env     # download the weights only
```

The base **Stable Diffusion 1.5** checkpoint
(`stable-diffusion-v1-5/stable-diffusion-v1-5`) is fetched from Hugging Face automatically
on first run.

<details>
<summary>Manual setup (Method 1)</summary>

```bash
# System libraries (Debian/Ubuntu): venv, a C++ toolchain + CMake for dlib,
# and the shared libs OpenCV needs at runtime.
sudo apt-get install -y python3.10-venv build-essential cmake libgl1 libglib2.0-0 ffmpeg

python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the cu118 --extra-index-url is baked into the file
```
</details>

The Stable-Hair weights live in `models/` (downloaded by `setup.sh`, or from
[Google Drive](https://drive.google.com/drive/folders/1E-8Udfw8S8IorCWhBgS4FajIbqlrWRbQ?usp=drive_link)):

```
models/
├── stage1/pytorch_model.bin     # Bald Converter (ControlNet)
└── stage2/
    ├── pytorch_model.bin         # Hair Extractor / reference encoder
    ├── pytorch_model_1.bin       # Adapter
    └── pytorch_model_2.bin       # Latent IdentityNet (ControlNet)
```

### Method 2 — FLUX.2
The FLUX.2 stack is intentionally newer than Method 1's and lives in `.venv-flux`:

```bash
# 1. Build the FLUX.2 environment (modern diffusers + torch cu126 + bitsandbytes + peft):
python3.10 -m venv .venv-flux
.venv-flux/bin/pip install -r flux/requirements-flux.txt

# 2. Authenticate and pre-download the 4-bit model + Turbo LoRA (gated — accept the
#    FLUX.2 [dev] license on Hugging Face first):
.venv-flux/bin/hf auth login
.venv-flux/bin/hf download diffusers/FLUX.2-dev-bnb-4bit
.venv-flux/bin/hf download fal/FLUX.2-dev-Turbo
```

(The weights also download automatically on first run of the worker.) Defaults — model
repo, Turbo LoRA, steps, sigmas, guidance and the edit prompt — live in
[`configs/flux2.yaml`](configs/flux2.yaml). To run the plain dev model instead of the
Turbo LoRA, comment out `turbo_lora_repo` there and raise `num_inference_steps`.

## Usage

### Web demo (both methods)
An interactive Gradio UI with a method selector:

```bash
# Method 1 only (Stable-Hair):
.venv/bin/python gradio_demo_full.py     # http://0.0.0.0:8986

# With FLUX.2 — the launcher starts the worker, then the app:
./start_demo.sh                          # FLUX.2 worker (:8987) + Gradio app
```

Run the processes by hand instead of `start_demo.sh`:
```bash
# Terminal 1 — the FLUX.2 worker:
.venv-flux/bin/python flux/flux2_server.py     # http://127.0.0.1:8987
# Terminal 2 — Gradio app:
.venv/bin/python gradio_demo_full.py           # http://0.0.0.0:8986
```

In the UI, upload an **ID image** and a **reference hair** image, pick a method, and **Run**:
- **Stable-Hair** — tune the sliders; outputs the bald intermediate + the transfer.
- **FLUX.2** — optionally edit the prompt (describe the target hair for stronger control).

### Command-line inference (Method 1)
Edit the source/reference images and parameters in
[`configs/hair_transfer.yaml`](configs/hair_transfer.yaml), then:
```bash
.venv/bin/python infer_full.py     # writes source · bald · reference · transferred to ./output/
```

You can also hit the FLUX.2 worker directly:
```bash
curl -s -X POST http://127.0.0.1:8987/transfer \
  -F image1=@test_imgs/example2/source.jpg \
  -F image2=@test_imgs/example2/ref.jpg \
  -o resp.json   # {"result": "<base64 png>"}
```

### Training (Method 1)
The two stages are trained separately. Adjust the data paths and the accelerate config
([`default_config.yaml`](default_config.yaml)), then:
```bash
bash train_stage1.sh   # Bald Converter
bash train_stage2.sh   # Hair Extractor + Latent IdentityNet
```

## Project structure
```
configs/             inference configs (hair_transfer.yaml, flux2.yaml)
diffusers/           vendored, lightly-modified diffusers 0.23.1 (used by Method 1)
ref_encoder/         Hair Extractor, Latent ControlNet, adapters, attention
utils/               StableHair pipelines (transfer + bald conversion)
flux/                Method 2 worker + requirements (run in .venv-flux)
  flux2_server.py              FLUX.2-dev (4-bit) + Turbo LoRA worker (POST /transfer)
  requirements-flux.txt        FLUX.2 stack (diffusers, torch cu126, bitsandbytes, peft)
test_imgs/           sample ID / reference images (incl. example2/)
infer_full.py        command-line inference (Method 1)
gradio_demo_full.py  web demo (selector: Stable-Hair / FLUX.2)
train_stage{1,2}.py  Stable-Hair training scripts
setup.sh             installer for Method 1 (.venv + Stable-Hair weights)
start_demo.sh        launches the FLUX.2 worker + the Gradio app together
```

> Note: `flux/kontext_server.py` and `configs/kontext.yaml` from the earlier
> FLUX.1-Kontext-based Method 2 are retained for reference but are no longer wired into
> the demo.

## Notes & limitations
- **Method 1** depends on its first stage — if the bald converter struggles, transfer
  quality drops. The released model was trained on a small, FFHQ-aligned dataset
  (≈6k images for stage 1, ≈20k for stage 2), so it works best on cropped, aligned 512×512
  faces.
- **Method 2** is a 32B model: even at 8 steps with 4-bit + CPU offload it takes ~80 s per
  image and peaks ~36 GB. Quality scales with the prompt — describe the target hairstyle.
- The vendored `diffusers 0.23.1` (Method 1) and the modern FLUX.2 stack (Method 2) are kept
  in separate environments on purpose — don't mix them.

## Credits
- **Stable-Hair** (Method 1) — *Stable-Hair: Real-World Hair Transfer via Diffusion Model*
  by Yuxuan Zhang, Qing Zhang, Yiren Song, Jichao Zhang, Hao Tang, Jiaming Liu.
  [Project page](https://xiaojiu-z.github.io/Stable-Hair.github.io/) ·
  [Paper](https://arxiv.org/pdf/2407.14078) ·
  [Original repo](https://github.com/Xiaojiu-z/Stable-Hair). Method 1 here is a maintained
  fork updated to run on a modern machine (dead base-model mirror fixed, dependency/diffusers
  pins corrected, one-command `setup.sh`).
- **FLUX.2** (Method 2) — [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-dev).
- **FLUX.2 Turbo LoRA** (Method 2) — [fal](https://huggingface.co/fal/FLUX.2-dev-Turbo).

## Citation
```bibtex
@misc{zhang2024stablehairrealworldhairtransfer,
      title={Stable-Hair: Real-World Hair Transfer via Diffusion Model},
      author={Yuxuan Zhang and Qing Zhang and Yiren Song and Jiaming Liu},
      year={2024},
      eprint={2407.14078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14078},
}
```

## License
See [LICENSE](LICENSE).
