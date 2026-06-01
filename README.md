# Hair Transfer

A virtual hair try-on toolkit that transfers a hairstyle from a **reference image**
onto a **person's face** — with **three** interchangeable methods you can compare side
by side from a single Gradio app.

<a href='https://arxiv.org/pdf/2407.14078'><img src='https://img.shields.io/badge/Stable--Hair-Report-red'></a>
<a href='https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev'><img src='https://img.shields.io/badge/FLUX.1-Kontext-blue'></a>
<a href='https://huggingface.co/facebook/sam3'><img src='https://img.shields.io/badge/Meta-SAM3-orange'></a>

<img src='assets/teaser_.jpg'>

## Why three methods?
Hair transfer can be approached very differently depending on the tradeoff you want
between **identity preservation**, **fidelity to the reference hairstyle**, and
**hardware**. This project bundles a classic trained ControlNet pipeline and two modern
diffusion-editing approaches so you can pick — or compare — the right tool for a given image.

| Method | Model | How it works | Best for | VRAM | Env |
|---|---|---|---|---|---|
| **1 · Stable-Hair** | SD 1.5 (trained) | Bald-converter ControlNet → Hair Extractor + Latent IdentityNet | Aligned 512×512 faces; no gated downloads | ~8 GB | `.venv` |
| **2 · FLUX Kontext** | FLUX.1-Kontext-dev | ID + reference edited together by a prompt, result cropped back | Quick, prompt-steerable transfers | ~24 GB | `.venv-flux` |
| **3 · SAM3 + Kontext** | SAM3 + FLUX.1-Kontext-dev | SAM3 masks the hair, Kontext repaints only that region from the reference | Surgical edits that keep face/background intact | ~24 GB | `.venv-flux` |

All three are selectable from one web UI ([`gradio_demo_full.py`](gradio_demo_full.py)).

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

### Method 2 — FLUX Kontext (prompt-driven edit)
The ID image and the hair-reference are placed side by side and
[`FluxKontextPipeline`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) edits
the canvas to give the ID the reference's hairstyle; the result is cropped back to the
person. No bald conversion, no ControlNet — just two images and a text instruction.

### Method 3 — SAM3 + FLUX.1-Kontext (segment-then-inpaint)
The most surgical option. [SAM3](https://huggingface.co/facebook/sam3) open-vocabulary
segmentation finds the hair on the source image (text prompt, e.g. `"hair"`), then
[`FluxKontextInpaintPipeline`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
repaints **only** that masked region using the reference image's hairstyle, leaving the
rest of the face and background untouched. The UI shows the SAM3 mask alongside the result.

> **One model for Methods 2 & 3.** Both use **FLUX.1-Kontext-dev**, served by a single
> worker ([`flux/kontext_server.py`](flux/kontext_server.py)) that loads the model **once**
> and shares it between the edit and inpaint pipelines (`from_pipe`). SAM3 (Method 3 only)
> is loaded lazily, so Method 2 works with just Kontext.

## Architecture
Method 1 imports a **vendored `diffusers 0.23.1`** that the model code depends on; the FLUX
methods need a **modern `diffusers` (≥ 0.38)** and a bf16 torch. A single Python process
can't hold both, so they live in separate virtual environments and communicate over local
HTTP:

```
.venv      ── gradio_demo_full.py ── Method 1 (Stable-Hair, in-process)
                   │  HTTP
                   └─▶ flux/kontext_server.py (:8987) ── Methods 2 & 3
.venv-flux ── the Kontext worker (FLUX.1-Kontext-dev, loaded once; SAM3 lazy)
```

If the worker isn't running, the Kontext methods show a "start the worker" message;
Method 1 always works in-process.

## Requirements
- **OS:** Linux (tested on Ubuntu 22.04)
- **Python:** 3.10+
- **GPU:** NVIDIA GPU + driver ≥ 520. Method 1 runs on ≥ 8 GB; the FLUX methods want
  ~24 GB bf16 (set `low_vram: true` in [`configs/kontext.yaml`](configs/kontext.yaml) for
  CPU offload on smaller cards).
- **Disk:** ~15 GB for Method 1 (weights + base SD1.5 + env); the FLUX.1-Kontext and SAM3
  weights add tens of GB more.
- **Gated models:** `black-forest-labs/FLUX.1-Kontext-dev` and `facebook/sam3` require
  accepting their license on Hugging Face and `huggingface-cli login`.

## Installation
[`setup.sh`](setup.sh) builds the environment(s), installs dependencies, and downloads
weights.

```bash
git clone <this-repo> hair-transfer
cd hair-transfer

./setup.sh                         # Method 1: .venv + Stable-Hair weights
# ./setup.sh --skip-weights        # build the environment only
# ./setup.sh --skip-env            # download the weights only
# ./setup.sh --flux                # + Methods 2 & 3: .venv-flux + FLUX.1-Kontext
# ./setup.sh --flux --sam-inpaint  # + SAM3 (enables Method 3's segmentation)
```

The base **Stable Diffusion 1.5** checkpoint
(`stable-diffusion-v1-5/stable-diffusion-v1-5`) is fetched from Hugging Face automatically
on first run of Method 1.

<details>
<summary>Manual setup (Method 1)</summary>

```bash
# System libraries (Debian/Ubuntu): venv, a C++ toolchain + CMake for dlib,
# and the shared libs OpenCV needs at runtime.
sudo apt-get install -y python3.10-venv build-essential cmake libgl1 libglib2.0-0 ffmpeg

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the cu118 --extra-index-url is baked into the file
```
</details>

### Stable-Hair weights
Download from [Google Drive](https://drive.google.com/drive/folders/1E-8Udfw8S8IorCWhBgS4FajIbqlrWRbQ?usp=drive_link)
(or just run `./setup.sh`) and arrange them as:

```
models/
├── stage1/
│   └── pytorch_model.bin        # Bald Converter (ControlNet)
└── stage2/
    ├── pytorch_model.bin        # Hair Extractor / reference encoder
    ├── pytorch_model_1.bin      # Adapter
    └── pytorch_model_2.bin      # Latent IdentityNet (ControlNet)
```

## Usage

### Web demo (all three methods)
An interactive Gradio UI with a method selector:

```bash
# Method 1 only (Stable-Hair):
python gradio_demo_full.py            # http://0.0.0.0:8986

# With the FLUX methods — the launcher starts the worker, then the app:
./start_demo.sh                       # FLUX.1-Kontext worker (:8987) + Gradio app
```

Run the processes by hand instead of `start_demo.sh`:
```bash
# Terminal 1 — the Kontext worker (serves Methods 2 & 3):
.venv-flux/bin/python flux/kontext_server.py       # http://127.0.0.1:8987
# Terminal 2 — Gradio app:
.venv/bin/python gradio_demo_full.py               # http://0.0.0.0:8986
```

In the UI, upload an **ID image** and a **reference hair** image, pick a method, and **Run**:
- **Stable-Hair** — tune the sliders; outputs the bald intermediate + the transfer.
- **FLUX Kontext** — optionally edit the prompt.
- **SAM3 + Inpaint** — set the **mask prompt** (what to segment, default `hair`) and
  optionally the edit prompt; the SAM3 mask appears in the first output, the result in the
  second.

Defaults for the FLUX methods live in [`configs/kontext.yaml`](configs/kontext.yaml).

### Command-line inference (Method 1)
Edit the source/reference images and parameters in
[`configs/hair_transfer.yaml`](configs/hair_transfer.yaml), then:
```bash
python infer_full.py     # writes source · bald · reference · transferred to ./output/
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
configs/             inference configs (hair_transfer.yaml, kontext.yaml)
diffusers/           vendored, lightly-modified diffusers 0.23.1 (used by Method 1)
ref_encoder/         Hair Extractor, Latent ControlNet, adapters, attention
utils/               StableHair pipelines (transfer + bald conversion)
flux/                Methods 2 & 3 worker + requirements (run in .venv-flux)
  kontext_server.py            shared FLUX.1-Kontext worker (/transfer + /inpaint_transfer)
  requirements-flux.txt        FLUX.1-Kontext stack
  requirements-sam-inpaint.txt adds SAM3 (Method 3)
test_imgs/           sample ID / reference images
infer_full.py        command-line inference (Method 1)
gradio_demo_full.py  web demo (selector: Stable-Hair / FLUX Kontext / SAM3 + Inpaint)
train_stage{1,2}.py  Stable-Hair training scripts
setup.sh             installer (--flux for Methods 2 & 3, --sam-inpaint adds SAM3)
start_demo.sh        launches the Kontext worker + the Gradio app together
```

## Notes & limitations
- **Method 1** depends on its first stage — if the bald converter struggles, transfer
  quality drops. The released model was trained on a small, FFHQ-aligned dataset
  (≈6k images for stage 1, ≈20k for stage 2), so it works best on cropped, aligned 512×512
  faces.
- **Method 2** relies on FLUX.1-Kontext preserving the side-by-side layout so the result
  can be cropped back to the person; misframed outputs are the place to revisit the crop.
- The vendored `diffusers 0.23.1` (Method 1) and the modern stack (Methods 2 & 3) are kept
  in separate environments on purpose — don't mix them.

## Credits
- **Stable-Hair** (Method 1) — *Stable-Hair: Real-World Hair Transfer via Diffusion Model*
  by Yuxuan Zhang, Qing Zhang, Yiren Song, Jichao Zhang, Hao Tang, Jiaming Liu.
  [Project page](https://xiaojiu-z.github.io/Stable-Hair.github.io/) ·
  [Paper](https://arxiv.org/pdf/2407.14078) ·
  [Original repo](https://github.com/Xiaojiu-z/Stable-Hair). Method 1 here is a maintained
  fork updated to run on a modern machine (dead base-model mirror fixed, dependency/diffusers
  pins corrected, one-command `setup.sh`).
- **FLUX.1-Kontext** (Methods 2 & 3) — [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
- **SAM 3** (Method 3) — [Meta AI](https://huggingface.co/facebook/sam3).

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
</content>
