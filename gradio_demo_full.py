import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import io
import base64
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
# Methods 2 & 3 both run on the merged FLUX.1-Kontext worker (flux/kontext_server.py
# in .venv-flux), which loads the model once and serves both over local HTTP.
#   Method 2 "FLUX Kontext" -> /transfer         (ID + reference concatenated, edited, cropped)
#   Method 3 "SAM3 + Inpaint" -> /inpaint_transfer (SAM3 hair mask + reference inpaint)
# ---------------------------------------------------------------------------
KONTEXT_CONFIG = OmegaConf.load("./configs/kontext.yaml")
KONTEXT_URL = f"http://{KONTEXT_CONFIG.server.host}:{KONTEXT_CONFIG.server.port}"

_START_HINT = "Start it first:  .venv-flux python flux/kontext_server.py  (see README)."


def _b64_to_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _two_image_files(id_image, ref_hair):
    id_pil = Image.fromarray(id_image.astype('uint8'), 'RGB')
    ref_pil = Image.fromarray(ref_hair.astype('uint8'), 'RGB')
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    id_pil.save(buf1, format="PNG")
    ref_pil.save(buf2, format="PNG")
    buf1.seek(0)
    buf2.seek(0)
    return {
        "image1": ("source.png", buf1, "image/png"),
        "image2": ("ref.png", buf2, "image/png"),
    }


def _post_kontext(endpoint, files, data):
    try:
        resp = requests.post(f"{KONTEXT_URL}/{endpoint}", files=files, data=data, timeout=600)
    except requests.exceptions.RequestException:
        raise gr.Error(f"Could not reach the FLUX.1-Kontext worker at {KONTEXT_URL}. {_START_HINT}")
    if resp.status_code != 200:
        # The worker returns a helpful JSON {"error": ...} (e.g. 422 = no mask found).
        try:
            msg = resp.json().get("error", resp.text[:200])
        except Exception:
            msg = resp.text[:200]
        raise gr.Error(f"Kontext worker error ({resp.status_code}): {msg}")
    return resp.json()


def kontext_transfer(id_image, ref_hair, prompt):
    files = _two_image_files(id_image, ref_hair)
    data = {"prompt": prompt or KONTEXT_CONFIG.edit_prompt}
    payload = _post_kontext("transfer", files, data)
    return _b64_to_image(payload["result"])


def sam_inpaint_transfer(id_image, ref_hair, prompt, mask_prompt):
    files = _two_image_files(id_image, ref_hair)
    data = {
        "prompt": prompt or KONTEXT_CONFIG.inpaint_prompt,
        "mask_prompt": mask_prompt or KONTEXT_CONFIG.mask_prompt,
    }
    payload = _post_kontext("inpaint_transfer", files, data)
    return _b64_to_image(payload["mask"]), _b64_to_image(payload["result"])


# ---------------------------------------------------------------------------
# Dispatch + UI
# ---------------------------------------------------------------------------
def run(method, id_image, ref_hair, prompt, mask_prompt,
        converter_scale, scale, guidance_scale, controlnet_conditioning_scale):
    if id_image is None or ref_hair is None:
        raise gr.Error("Please provide both an ID image and a reference hair image.")
    if method == "Stable-Hair":
        bald, result = model_call(id_image, ref_hair, converter_scale, scale,
                                  guidance_scale, controlnet_conditioning_scale)
        return bald, result
    if method == "FLUX Kontext":
        result = kontext_transfer(id_image, ref_hair, prompt)
        return None, result
    # SAM3 + Inpaint: first output is the hair mask, second is the inpainted result.
    mask, result = sam_inpaint_transfer(id_image, ref_hair, prompt, mask_prompt)
    return mask, result


def _on_method_change(method):
    is_sh = method == "Stable-Hair"
    is_inpaint = method == "SAM3 + Inpaint"
    # Prompt box is used by both Kontext methods; sliders only by Stable-Hair;
    # mask-prompt only by SAM3+Inpaint. The first output shows a bald face for
    # Stable-Hair and the segmentation mask for SAM3+Inpaint.
    first_output_label = "Hair mask (SAM3)" if is_inpaint else "Bald_Result"
    # The prompt default differs between the two Kontext methods.
    prompt_value = KONTEXT_CONFIG.inpaint_prompt if is_inpaint else KONTEXT_CONFIG.edit_prompt
    return (
        gr.update(visible=not is_sh, value=prompt_value),  # prompt
        gr.update(visible=is_inpaint),                     # mask_prompt
        gr.update(visible=is_sh),                          # stable-hair sliders group
        gr.update(visible=is_sh or is_inpaint, label=first_output_label),  # first output
    )


with gr.Blocks(title="Hair Transfer Demo") as iface:
    gr.Markdown(
        "# Hair Transfer Demo\n"
        "Three methods:\n"
        "- **Stable-Hair** — two-stage SD1.5 pipeline (tuned via sliders).\n"
        "- **FLUX Kontext** — prompt-driven; the ID + reference are edited together and the "
        "result cropped back to the person.\n"
        "- **SAM3 + Inpaint** — segment the hair with SAM3, then repaint just that region "
        "with the reference hairstyle.\n\n"
        "Both Kontext methods share one FLUX.1-Kontext worker (see README). "
        "Aligned 512×512 faces work best for Stable-Hair."
    )
    method = gr.Radio(
        ["Stable-Hair", "FLUX Kontext", "SAM3 + Inpaint"],
        value="Stable-Hair",
        label="Method",
    )

    with gr.Row():
        id_image = gr.Image(label="ID image (the person)")
        ref_hair = gr.Image(label="Reference hair")

    prompt = gr.Textbox(
        label="Prompt (Kontext methods)",
        value=KONTEXT_CONFIG.edit_prompt,
        lines=2,
        visible=False,
    )
    mask_prompt = gr.Textbox(
        label="Mask prompt (SAM3 — what to segment & repaint)",
        value=KONTEXT_CONFIG.mask_prompt,
        lines=1,
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

    method.change(
        _on_method_change,
        inputs=method,
        outputs=[prompt, mask_prompt, sh_controls, output_bald],
    )
    run_btn.click(
        run,
        inputs=[method, id_image, ref_hair, prompt, mask_prompt,
                converter_scale, scale, guidance_scale, controlnet_conditioning_scale],
        outputs=[output_bald, output_transfer],
    )

# Launch the Gradio interface
iface.queue().launch(server_name='0.0.0.0', server_port=8986)
