import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import cv2
import numpy as np
import io

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import List
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.style_encoder import Style_Aware_Encoder
from ip_adapter.tools import pre_processing
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, patch_size=32, size=512, image_root_path="", image_json_file=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.patch_size = patch_size
        self.none_loop = 0
        self.image_root_path = image_root_path

        with open(image_json_file, 'r', encoding='utf-8') as f:
            self.data = f.readlines()  # lines of dict: {"souce": "1.png", "prompt": "A dog"} \n {} \n ...

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # centercrop for style condition
        self.crop = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
            ]
        )
        self.content = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
            ]
        )

    def load_data(self, item):
        try:
            text = item["content_prompt"]
        except Exception as e:
            print(e)
            text = ""
        image_file = item["image_file"]
        content_image_file = item["content_image_file"]
        try:
            raw_image = Image.open(os.path.join(self.image_root_path, image_file))
            content_image = Image.open(os.path.join(self.image_root_path, content_image_file))
        except Exception as e:
            print(e)
            raw_image = None
            content_image = None
        return raw_image, content_image, text, image_file

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        raw_image, content_image, text, image_file = self.load_data(item)
        # read image

        while raw_image is None or content_image is None or text is None or np.array(raw_image).shape[-1] < 3:
            # corner cases
            if 0 <= idx < len(self.data) - 1:
                idx += 1
            elif idx == len(self.data) - 1:
                idx = 0
            item = json.loads(self.data[idx])
            raw_image, content_image, text, image_file = self.load_data(item)
            self.none_loop += 1
            if self.none_loop > 10000:
                break

        raw_image = self.crop(raw_image)
        # patching
        high_style_patch, middle_style_patch, low_style_patch = pre_processing(raw_image.convert("RGB"), self.transform)
        # shuffling
        high_style_patch, middle_style_patch, low_style_patch = (high_style_patch[torch.randperm(high_style_patch.shape[0])],
                                                                 middle_style_patch[torch.randperm(middle_style_patch.shape[0])],
                                                                 low_style_patch[torch.randperm(low_style_patch.shape[0])])
        image = self.transform(raw_image.convert("RGB"))
        content_input = self.content(content_image.convert("RGB"))
        # drop
        rand_num = random.random()
        drop_style_embed, drop_content = 0, 0
        if rand_num < 0.05:
            drop_style_embed = 1
            text = ""
            drop_content = 1
        elif rand_num < 0.25:
            drop_style_embed = 1
            text = text
            drop_content = 0
        elif rand_num < 0.45:
            drop_style_embed = 0
            text = text
            drop_content = 1
        elif rand_num < 1.0:
            text = text
            drop_style_embed = 0
            drop_content = 0

        # content input
        if drop_content:
            content_input = torch.zeros_like(content_input)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return {
            "image_file": image_file,
            "image": image,
            "text": text,
            "raw_image": raw_image,
            "text_input_ids": text_input_ids,
            "drop_style_embed": drop_style_embed,
            "high_style_patch": high_style_patch,
            "middle_style_patch": middle_style_patch,
            "low_style_patch": low_style_patch,
            "content_input": content_input,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    image_files = [example["image_file"] for example in data]
    raw_images = [example["raw_image"].convert("RGB").resize((512, 512)) for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    drop_style_embeds = [example["drop_style_embed"] for example in data]
    high_style_patches = torch.cat([example["high_style_patch"] for example in data])
    middle_style_patches = torch.cat([example["middle_style_patch"] for example in data])
    low_style_patches = torch.cat([example["low_style_patch"] for example in data])
    content_inputs = torch.stack([example["content_input"] for example in data])
    return {
        "image_files": image_files,
        "images": images,
        "texts": texts,
        "raw_images": raw_images,
        "text_input_ids": text_input_ids,
        "drop_style_embeds": drop_style_embeds,
        "high_style_patches": high_style_patches,
        "middle_style_patches": middle_style_patches,
        "low_style_patches": low_style_patches,
        "content_inputs": content_inputs,
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, style_image_proj_modules, content_fusion_encoder, adapter_modules, ckpt_path=None):
        super().__init__()
        self.num_tokens = 6
        self.unet = unet
        self.controlnet = content_fusion_encoder
        self.style_image_proj_modules = style_image_proj_modules
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, style_image_embeds, content_inputs):
        style_ip_tokens = []
        # style tokens
        for idx, style_image_embed in enumerate([style_image_embeds[:,0,:], style_image_embeds[:,1,:], style_image_embeds[:,2,:]]):
            style_ip_tokens.append(self.style_image_proj_modules[idx](style_image_embed))
        style_ip_tokens = torch.cat(style_ip_tokens, dim=1)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=style_ip_tokens,
            controlnet_cond=content_inputs,
            return_dict=False,
        )

        encoder_hidden_states = torch.cat([encoder_hidden_states, style_ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample,
                               ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_style_ip_proj_sum = torch.sum(
            torch.stack([torch.sum(p) for p in self.style_image_proj_modules.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        sd = torch.load(ckpt_path, map_location="cpu")
        style_image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("style_image_proj_modules"):
                style_image_proj_sd[k.replace("style_image_proj_modules.", "")] = sd[k]
            elif k.startswith("adapter_modules"):
                ip_sd[k.replace("adapter_modules.", "")] = sd[k]
        # Load state dict for image_proj_model and adapter_modules
        self.style_image_proj_modules.load_state_dict(style_image_proj_sd, strict=True)
        self.adapter_modules.load_state_dict(ip_sd, strict=True)

        # Calculate new checksums
        new_style_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.style_image_proj_modules.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_style_ip_proj_sum != new_style_ip_proj_sum, "Weights of style_image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--pretrained_style_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained style-aware encoder. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--image_json_file",
        type=str,
        default="",
        help="Training data",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="",
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer, style-aware encoder and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    style_aware_encoder = Style_Aware_Encoder(image_encoder)
    style_aware_encoder.load_state_dict(torch.load(args.pretrained_style_encoder_path))

    content_fusion_encoder = ControlNetModel.from_unet(unet)

    content_fusion_encoder.requires_grad_(True)
    style_aware_encoder.requires_grad_(False)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ip-adapter
    style_image_proj_models = torch.nn.ModuleList([
        ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=style_aware_encoder.projection_dim,
            clip_extra_context_tokens=2,
        ),
        ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=style_aware_encoder.projection_dim,
            clip_extra_context_tokens=2,
        ),
        ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=style_aware_encoder.projection_dim,
            clip_extra_context_tokens=2,
        )
    ])
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=6)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = IPAdapter(unet, style_image_proj_models, content_fusion_encoder, adapter_modules, args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    style_aware_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(ip_adapter.controlnet.parameters(),)
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(tokenizer=tokenizer, size=args.resolution, image_root_path=args.image_root_path, image_json_file=args.image_json_file)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with (accelerator.accumulate(ip_adapter)):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with torch.no_grad():
                    style_embeds = style_aware_encoder((batch["high_style_patches"].to(accelerator.device, dtype=weight_dtype),
                                                        batch["middle_style_patches"].to(accelerator.device, dtype=weight_dtype),
                                                        batch["low_style_patches"].to(accelerator.device, dtype=weight_dtype)),
                                                       batch["images"].shape[0])

                for idx, drop_style_embed in enumerate(batch["drop_style_embeds"]):
                    if drop_style_embed == 1:
                        style_embeds[idx] = style_embeds[idx]*0.0
                    else:
                        continue
                content_inputs = batch["content_inputs"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, style_embeds, content_inputs)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
