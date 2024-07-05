import os
from types import MethodType

import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from huggingface_hub import snapshot_download
from ip_adapter import StyleShot
import argparse

def main(args):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    styleshot_model_path = "Gaojunyao/StyleShot"
    adapter_model_path = "TencentARC/t2iadapter_depth_sd15v2"
    adapter = T2IAdapter.from_pretrained(adapter_model_path, torch_dtype=torch.float16)
    
    device = "cuda"

    if not os.path.isdir(base_model_path):
        base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
        print(f"Downloaded model to {base_model_path}")
    if not os.path.isdir(transformer_block_path):
        transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
        print(f"Downloaded model to {transformer_block_path}")
    if not os.path.isdir(styleshot_model_path):
        styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
        print(f"Downloaded model to {styleshot_model_path}")
    if not os.path.isdir(adapter_model_path):
        adapter_model_path = snapshot_download(adapter_model_path, local_dir=adapter_model_path)
        print(f"Downloaded model to {adapter_model_path}")

    adapter = T2IAdapter.from_pretrained(adapter_model_path, torch_dtype=torch.float16)
    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    
    pipe = StableDiffusionAdapterPipeline.from_pretrained(base_model_path, adapter=adapter)
    
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    
    image = Image.open(args.style)
    
    condition_image = Image.open(args.condition).convert("RGB")
    
    generation = styleshot.generate(style_image=image, prompt=[[args.prompt]], image=[condition_image])
    
    generation[0][0].save(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style.png")
    parser.add_argument("--condition", type=str, default="condition.png")
    parser.add_argument("--prompt", type=str, default="text prompt")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()
    main(args)


