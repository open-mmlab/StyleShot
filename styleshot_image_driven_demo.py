import os
from types import MethodType

import torch
import cv2
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from huggingface_hub import snapshot_download
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
import argparse

def main(args):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    device = "cuda"

    if args.preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "Gaojunyao/StyleShot_lineart"
    elif args.preprocessor == "Contour":
        detector = SOFT_HEDdetector()
        styleshot_model_path = "Gaojunyao/StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    if not os.path.isdir(styleshot_model_path):
        styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
        print(f"Downloaded model to {styleshot_model_path}")

    # weights for ip-adapter and our content-fusion encoder
    if not os.path.isdir(base_model_path):
        base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
        print(f"Downloaded model to {base_model_path}")
    if not os.path.isdir(transformer_block_path):
        transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
        print(f"Downloaded model to {transformer_block_path}")

    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_fusion_encoder = ControlNetModel.from_unet(unet)
    
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=content_fusion_encoder)
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)

    style_image = Image.open(args.style)
    # processing content image
    content_image = cv2.imread(args.content)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content_image = detector(content_image)
    content_image = Image.fromarray(content_image)
    
    generation = styleshot.generate(style_image=style_image, prompt=[[args.prompt]], content_image=content_image)
    
    generation[0][0].save(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style.png")
    parser.add_argument("--content", type=str, default="content.png")
    parser.add_argument("--preprocessor", type=str, default="Contour", choices=["Contour", "Lineart"])
    parser.add_argument("--prompt", type=str, default="text prompt")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()
    main(args)