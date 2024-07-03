from types import MethodType

import torch
import cv2
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
import argparse

def main(args):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    
    # weights for ip-adapter and our content-retention encoder
    ip_ckpt = "./pretrained_weight/ip.bin"
    style_aware_encoder_path = "./pretrained_weight/style_aware_encoder.bin"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_retention_encoder = ControlNetModel.from_unet(unet)
    
    device = "cuda"
    
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=content_retention_encoder)
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    if args.preprocessor == "Lineart":
        detector = LineartDetector()
    elif args.preprocessor == "Contour":
        detector = SOFT_HEDdetector()
    else:
        raise ValueError("Invalid preprocessor")

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

