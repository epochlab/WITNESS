#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
# from PIL import Image
from tqdm.auto import tqdm
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging

class LatentDiffusion():
    def __init__(self, model: str, device: str):
        self.model = model
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(self.model, subfolder="vae", torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.model, subfolder="unet", torch_dtype=torch.float16).to(self.device)
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    def generate(self,
                 config: dataclass, 
                 seed: int,
                 method: str, 
                 encoded: torch.Tensor = None
                 ) -> torch.Tensor:

        generator = torch.manual_seed(seed) # Seed generator to create the inital latent noise

        text_input = self.tokenizer(config.prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].half()

        max_length = text_input.input_ids.shape[-1]
        
        uncond_input = self.tokenizer([config.negative_prompt] * config.batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0].half()

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(config.infer_steps)

        # Latents
        if method == 'txt2img':
            start_step = 0
            latents = torch.randn((config.batch_size, self.unet.config.in_channels, config.h // 8, config.w // 8), generator=generator).to(self.device).half()
            latents = latents * self.scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

        elif method == 'img2img':
            # Prep latents (noising appropriately for start_step)
            start_step = config.sampling_step
            _ = self.scheduler.sigmas[start_step]
            noise = torch.randn_like(encoded)
            latents = self.scheduler.add_noise(encoded, noise, timesteps=torch.tensor([self.scheduler.timesteps[start_step]])).to(self.device).half()

        # Loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            if i >= start_step: # << This is the only modification to the loop we do
                latent_model_input = torch.cat([latents] * 2) # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # Scale the latents (preconditioning):

                # Predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.cfg_scale * (noise_pred_text - noise_pred_uncond)

                # latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample # Predicted x0:

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def encode(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).permute(2, 0, 1)
        x = (x.unsqueeze(0) * 2 - 1).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            latent = self.vae.encode(x) # Note scaling
        return 0.18215 * latent.latent_dist.sample()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (1 / 0.18215) * latents
        with torch.no_grad(): image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach().cpu().permute(0, 2, 3, 1)[0]