from types import MethodType

import os
import gradio as gr
import torch
import cv2
from annotator.util import resize_image
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import snapshot_download
from PIL import Image
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

contour_detector = SOFT_HEDdetector()
lineart_detector = LineartDetector()

base_model_path = "runwayml/stable-diffusion-v1-5"
transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
styleshot_model_path = "Gaojunyao/StyleShot"
styleshot_lineart_model_path = "Gaojunyao/StyleShot_lineart"

if not os.path.isdir(base_model_path):
    base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
    print(f"Downloaded model to {base_model_path}")
if not os.path.isdir(transformer_block_path):
    transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
    print(f"Downloaded model to {transformer_block_path}")
if not os.path.isdir(styleshot_model_path):
    styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
    print(f"Downloaded model to {styleshot_model_path}")
if not os.path.isdir(styleshot_lineart_model_path):
    styleshot_lineart_model_path = snapshot_download(styleshot_lineart_model_path, local_dir=styleshot_lineart_model_path)
    print(f"Downloaded model to {styleshot_lineart_model_path}")

    
# weights for ip-adapter and our content-fusion encoder
contour_ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
contour_style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")
contour_transformer_block_path = transformer_block_path
contour_unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
contour_content_fusion_encoder = ControlNetModel.from_unet(contour_unet)

contour_pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=contour_content_fusion_encoder)
contour_styleshot = StyleShot(device, contour_pipe, contour_ip_ckpt, contour_style_aware_encoder_path, contour_transformer_block_path)

lineart_ip_ckpt = os.path.join(styleshot_lineart_model_path, "pretrained_weight/ip.bin")
lineart_style_aware_encoder_path = os.path.join(styleshot_lineart_model_path, "pretrained_weight/style_aware_encoder.bin")
lineart_transformer_block_path = transformer_block_path
lineart_unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
lineart_content_fusion_encoder = ControlNetModel.from_unet(lineart_unet)

lineart_pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=lineart_content_fusion_encoder)
lineart_styleshot = StyleShot(device, lineart_pipe, lineart_ip_ckpt, lineart_style_aware_encoder_path, lineart_transformer_block_path)


def process(style_image, content_image, prompt, num_samples, image_resolution, condition_scale, style_scale,ddim_steps, guidance_scale, seed, a_prompt, n_prompt, btn1, Contour_Threshold=200):
    weight_dtype = torch.float32

    style_shots = []
    btns = []
    contour_content_images = []
    contour_results = []
    lineart_content_images = []
    lineart_results = []

    type1 = 'Contour'
    type2 = 'Lineart'

    if btn1 == type1 or content_image is None:
        style_shots = [contour_styleshot]
        btns = [type1]
    elif btn1 == type2:
        style_shots = [lineart_styleshot]
        btns = [type2]
    elif btn1 == "Both":
        style_shots = [contour_styleshot, lineart_styleshot]
        btns = [type1, type2]

    ori_style_image = style_image.copy()

    
    if content_image is not None:
        ori_content_image = content_image.copy()
    else:
        ori_content_image = None

    for styleshot, btn in zip(style_shots, btns):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prompts = [prompt+" "+a_prompt]

        style_image = Image.fromarray(ori_style_image)

        if ori_content_image is not None:
            if btn == type1:
                content_image = resize_image(ori_content_image, image_resolution)
                content_image = contour_detector(content_image, threshold=Contour_Threshold)
            elif btn == type2:
                content_image = resize_image(ori_content_image, image_resolution)
                content_image = lineart_detector(content_image, coarse=False)

            content_image = Image.fromarray(content_image)
        else:
            content_image = cv2.resize(ori_style_image, (image_resolution, image_resolution))
            content_image = Image.fromarray(content_image)
            condition_scale = 0.0

        g_images = styleshot.generate(style_image=style_image, 
                           prompt=[[prompt]], 
                           negative_prompt=n_prompt,
                           scale=style_scale, 
                           num_samples = num_samples,
                           seed = seed, 
                           num_inference_steps=ddim_steps, 
                           guidance_scale=guidance_scale,
                           content_image=content_image,
                           controlnet_conditioning_scale= float(condition_scale))
        
        if btn == type1:
            contour_content_images = [content_image]
            contour_results = g_images[0]
        elif btn == type2:
            lineart_content_images = [content_image]
            lineart_results = g_images[0]
    if ori_content_image is None:
        contour_content_images = []
        lineart_results = []
        lineart_content_images = []    

    return [contour_results, contour_content_images, lineart_results, lineart_content_images]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Styleshot Demo")
    with gr.Row():
        with gr.Column():
            style_image = gr.Image(sources=['upload'], type="numpy", label='Style Image')
        with gr.Column():
            with gr.Blocks():
                with gr.Column():
                    content_image = gr.Image(sources=['upload'], type="numpy", label='Content Image (optional)')
                    btn1 = gr.Radio(
                        choices=["Contour", "Lineart", "Both"],
                        interactive=True,
                        label="Preprocessor",
                        value="Both",
                    )
                    gr.Markdown("We recommend using 'Contour' for sparse control and 'Lineart' for detailed control. If you choose 'Both', we will provide results for two types of control. If you choose 'Contour', you can adjust the 'Contour Threshold' under the 'Advanced options' for the level of detail in control. ")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
    with gr.Row():
        run_button = gr.Button(value="Run")
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                condition_scale = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)

                Contour_Threshold = gr.Slider(label="Contour Threshold", minimum=0, maximum=255, value=200, step=1)

                style_scale = gr.Slider(label="Style Strength", minimum=0, maximum=2, value=1.0, step=0.01)
                
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647,  value=42, step=1)

                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        
    with gr.Row():
        gr.Markdown("### Results for Contour")
    with gr.Row():
        with gr.Blocks():
            with gr.Row():
                with gr.Column(scale = 1):
                    contour_gallery = gr.Gallery(label='Contour Output', show_label=True, elem_id="gallery", columns=[1], rows=[1], height='auto')
                with gr.Column(scale = 4):
                    image_gallery = gr.Gallery(label='Result for Contour', show_label=True, elem_id="gallery", columns=[4], rows=[1], height='auto')
    with gr.Row():
        gr.Markdown("### Results for Lineart")
    with gr.Row():
        with gr.Blocks():
            with gr.Row():
                with gr.Column(scale = 1):
                    line_gallery = gr.Gallery(label='Lineart Output', show_label=True, elem_id="gallery", columns=[1], rows=[1], height='auto')
                with gr.Column(scale = 4):
                    line_image_gallery = gr.Gallery(label='Result for Lineart', show_label=True, elem_id="gallery", columns=[4], rows=[1], height='auto')
        
    ips = [style_image, content_image, prompt, num_samples, image_resolution, condition_scale, style_scale, ddim_steps, guidance_scale, seed, a_prompt, n_prompt, btn1, Contour_Threshold]
    run_button.click(fn=process, inputs=ips, outputs=[image_gallery, contour_gallery, line_image_gallery, line_gallery])


block.launch(server_name='0.0.0.0')