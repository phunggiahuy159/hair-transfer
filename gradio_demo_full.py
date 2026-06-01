import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import io
import cv2
import requests
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline


class StableHair:
    def __init__(self, config="./configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float32) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load vae controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)
        controlnet_conditioning_scale = float(controlnet_conditioning_scale)

        # load imgs
        H, W, C = source_image.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return sample, source_image, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image


# ---------------------------------------------------------------------------
# Method 1: Stable-Hair (the original two-stage SD1.5 pipeline).
# The model is loaded lazily so the app still launches when only the FLUX
# method is used (and vice-versa).
# ---------------------------------------------------------------------------
_stable_hair_model = None


def get_stable_hair():
    global _stable_hair_model
    if _stable_hair_model is None:
        _stable_hair_model = StableHair(config="./configs/hair_transfer.yaml", weight_dtype=torch.float32)
    return _stable_hair_model


def model_call(id_image, ref_hair, converter_scale, scale, guidance_scale, controlnet_conditioning_scale):
    model = get_stable_hair()
    id_image = Image.fromarray(id_image.astype('uint8'), 'RGB')
    ref_hair = Image.fromarray(ref_hair.astype('uint8'), 'RGB')
    id_image = id_image.resize((512, 512))
    ref_hair = ref_hair.resize((512, 512))
    id_image_bald = model.get_bald(id_image, converter_scale)

    id_image_bald = np.array(id_image_bald)
    ref_hair = np.array(ref_hair)

    image, source_image, reference_image = model.Hair_Transfer(source_image=id_image_bald,
                                                               reference_image=ref_hair,
                                                               random_seed=-1,
                                                               step=30,
                                                               guidance_scale=guidance_scale,
                                                               scale=scale,
                                                               controlnet_conditioning_scale=controlnet_conditioning_scale
                                                               )

    image = Image.fromarray((image * 255.).astype(np.uint8))
    return id_image_bald, image


# ---------------------------------------------------------------------------
# Method 2: FLUX.2 [klein] 9B — prompt-driven editing.
# This runs in a separate venv (.venv-flux) served by flux/flux_server.py; we
# reach it over local HTTP. Just send the two images + a text prompt.
# ---------------------------------------------------------------------------
FLUX_CONFIG = OmegaConf.load("./configs/flux_klein.yaml")
FLUX_URL = f"http://{FLUX_CONFIG.server.host}:{FLUX_CONFIG.server.port}"


def flux_transfer(id_image, ref_hair, prompt):
    id_pil = Image.fromarray(id_image.astype('uint8'), 'RGB')
    ref_pil = Image.fromarray(ref_hair.astype('uint8'), 'RGB')
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    id_pil.save(buf1, format="PNG")
    ref_pil.save(buf2, format="PNG")
    buf1.seek(0)
    buf2.seek(0)

    files = {
        "image1": ("id.png", buf1, "image/png"),
        "image2": ("ref.png", buf2, "image/png"),
    }
    data = {"prompt": prompt or FLUX_CONFIG.default_prompt}
    try:
        resp = requests.post(f"{FLUX_URL}/transfer", files=files, data=data, timeout=600)
    except requests.exceptions.RequestException:
        raise gr.Error(
            f"Could not reach the FLUX.2 klein worker at {FLUX_URL}. "
            "Start it first:  .venv-flux python flux/flux_server.py  (see README)."
        )
    if resp.status_code != 200:
        raise gr.Error(f"FLUX worker error ({resp.status_code}): {resp.text[:200]}")
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Dispatch + UI
# ---------------------------------------------------------------------------
def run(method, id_image, ref_hair, prompt,
        converter_scale, scale, guidance_scale, controlnet_conditioning_scale):
    if id_image is None or ref_hair is None:
        raise gr.Error("Please provide both an ID image and a reference hair image.")
    if method == "Stable-Hair":
        bald, result = model_call(id_image, ref_hair, converter_scale, scale,
                                  guidance_scale, controlnet_conditioning_scale)
        return bald, result
    # FLUX.2 klein
    result = flux_transfer(id_image, ref_hair, prompt)
    return None, result


def _on_method_change(method):
    is_sh = method == "Stable-Hair"
    # Stable-Hair shows the sliders + bald output; FLUX shows the prompt box.
    return (
        gr.update(visible=not is_sh),  # prompt
        gr.update(visible=is_sh),      # stable-hair sliders group
        gr.update(visible=is_sh),      # bald output
    )


with gr.Blocks(title="Hair Transfer Demo") as iface:
    gr.Markdown(
        "# Hair Transfer Demo\n"
        "Two methods: **Stable-Hair** (two-stage SD1.5 pipeline) and "
        "**FLUX.2 klein** (prompt-driven, just give it the two images + a text instruction).\n\n"
        "Aligned 512×512 faces work best for Stable-Hair. The FLUX method needs its "
        "worker running (see README)."
    )
    method = gr.Radio(["Stable-Hair", "FLUX.2 klein"], value="Stable-Hair", label="Method")

    with gr.Row():
        id_image = gr.Image(label="ID image (the person)")
        ref_hair = gr.Image(label="Reference hair")

    prompt = gr.Textbox(
        label="Prompt (FLUX.2 klein)",
        value=FLUX_CONFIG.default_prompt,
        lines=2,
        visible=False,
    )

    with gr.Group(visible=True) as sh_controls:
        converter_scale = gr.Slider(minimum=0.5, maximum=1.5, value=1, label="Converter Scale")
        scale = gr.Slider(minimum=0.0, maximum=3, value=1.0, label="Hair Encoder Scale")
        guidance_scale = gr.Slider(minimum=1.1, maximum=3.0, value=1.5, label="CFG")
        controlnet_conditioning_scale = gr.Slider(minimum=0.1, maximum=2.0, value=1, label="Latent IdentityNet Scale")

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        output_bald = gr.Image(type="pil", label="Bald_Result")
        output_transfer = gr.Image(type="pil", label="Transfer Result")

    method.change(_on_method_change, inputs=method, outputs=[prompt, sh_controls, output_bald])
    run_btn.click(
        run,
        inputs=[method, id_image, ref_hair, prompt,
                converter_scale, scale, guidance_scale, controlnet_conditioning_scale],
        outputs=[output_bald, output_transfer],
    )

# Launch the Gradio interface
iface.queue().launch(server_name='0.0.0.0', server_port=8986)
