from types import MethodType

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from ip_adapter import StyleShot
import argparse

def main(args):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    ip_ckpt = "./pretrained_weight/ip.bin"
    style_aware_encoder_path = "./pretrained_weight/style_aware_encoder.bin"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
    
    device = "cuda"
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet)
    
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    
    image = Image.open(args.style)
    
    condition_image = Image.open(args.condition)
    
    generation = styleshot.generate(style_image=image, prompt=[[args.prompt]], image=condition_image)
    
    generation[0][0].save(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style.png")
    parser.add_argument("--condition", type=str, default="condition.png")
    parser.add_argument("--prompt", type=str, default="text prompt")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()
    main(args)

