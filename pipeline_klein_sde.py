# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import PIL
import torch
from torch import Tensor
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# ==============================================================================
# Interpolant Schedulers
# ==============================================================================

@dataclass
class SchedulerOutput:
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class CondOTScheduler:
    """Linear/OT interpolant: x_t = (1-t)*x_0 + t*noise"""
    
    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=1 - t,
            sigma_t=t,
            d_alpha_t=-torch.ones_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 1 / (snr + 1.0)


class VPScheduler:
    """Variance Preserving interpolant"""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        
        T = 0.5 * t**2 * (B - b) + t * b
        dT = t * (B - b) + b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = (-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b)
        return t


# ==============================================================================
# VP-SDE Scheduler
# ==============================================================================

class FlowMatchVPSDEScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_interpolant = CondOTScheduler()
        self.vp_interpolant = VPScheduler()

    def _get_score_vp(self, velocity, sample, r):
        """Convert velocity to score in VP interpolant (matching SD3 implementation)"""
        vp = self.vp_interpolant(r)
        alpha_t = vp.alpha_t
        sigma_t = vp.sigma_t
        d_alpha_t = vp.d_alpha_t
        d_sigma_t = vp.d_sigma_t
        
        # Note the negative sign at the front (matching SD3 code)
        score = -(alpha_t * velocity - d_alpha_t * sample) / (sigma_t * (d_alpha_t * sigma_t - alpha_t * d_sigma_t))
        return score
    
    def _ot2vp(self, model_fn, r, sample, image_latents):
        """
        Transform velocity from OT to VP interpolant (matching SD3's ot2vp function)
        """
        b = 0.1
        B = 20.0
        
        T = 0.5 * r**2 * (B - b) + r * b
        dT = r * (B - b) + b
        
        alpha_r = torch.exp(-0.5 * T)
        sigma_r = torch.sqrt(1 - torch.exp(-T))
        d_alpha_r = -0.5 * dT * torch.exp(-0.5 * T)
        d_sigma_r = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))
        
        snr = alpha_r / sigma_r
        t_r = 1 / (1 + snr)
        
        alpha_t_r = 1 - t_r
        sigma_t_r = t_r
        d_alpha_t_r = -1
        d_sigma_t_r = 1
        
        s_r = sigma_r / sigma_t_r
        dt_r = (
            sigma_t_r * sigma_t_r
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t_r * d_alpha_t_r - alpha_t_r * d_sigma_t_r))
        )
        ds_r = (sigma_t_r * d_sigma_r - sigma_r * d_sigma_t_r * dt_r) / (sigma_t_r * sigma_t_r)
        
        # Call model with rescaled input and remapped timestep
        x_t_ot = sample / s_r
        v_t_r = model_fn(x_t_ot, t_r, image_latents)

        #tweedie estimate
        x0_hat = x_t_ot - t_r * v_t_r
        
        # Transform velocity
        v_r = ds_r * sample / s_r + dt_r * s_r * v_t_r

        return v_r, x0_hat, sigma_r
    
    def _decode_x0_for_guidance(self, pipe, x0_packed, latent_ids):
        # unpack tokens → spatial
        x0 = pipe._unpack_latents_with_ids(x0_packed, latent_ids)
        # reverse BN normalization
        bn_mean = pipe.vae.bn.running_mean.view(1,-1,1,1).to(x0.device, x0.dtype)
        bn_std  = torch.sqrt(pipe.vae.bn.running_var.view(1,-1,1,1) + pipe.vae.config.batch_norm_eps).to(x0.device, x0.dtype)
        x0 = x0 * bn_std + bn_mean
        # unpatchify
        x0 = pipe._unpatchify_latents(x0)
        # VAE decode (needs grad → use decode without no_grad)
        x0 = pipe.vae.decode(x0, return_dict=False)[0]  # (B, 3, H, W), in [-1, 1]
        return x0
    
    @torch.autocast(device_type="cuda")
    def step_sde(
        self, 
        model_fn: Callable, 
        timestep: torch.Tensor, 
        sample: torch.Tensor, 
        dt: torch.Tensor, 
        diffusion_norm: float = 8.0, 
        use_ode: bool = False,
        image_latents: torch.Tensor = None,
        verifier = None,
        guidance_weight: float = None,
        pipe = None,
        latent_ids = None
    ):
        """
        SDE step with VP interpolant conversion (matching SD3 implementation).
        
        Args:
            model_fn: Callable that takes (hidden_states, timestep) and returns velocity.
            timestep: Current timestep (in [0, 1000] range)
            sample: Current latent sample
            dt: Time step size (negative, since we go from t=1 to t=0)
            diffusion_norm: Scale for diffusion coefficient (default 8.0 matching SD3)
            use_ode: If True, skip noise injection
        """
        if self.step_index is None:
            self._init_step_index(timestep)
        
        r = (timestep / 1000).to(sample.dtype)      

        # Get transformed velocity and decode with grads
        if guidance_weight is not None:
            sample = sample.detach().requires_grad_(True)
            v_r, x0_hat, sigma_r = self._ot2vp(model_fn, r, sample, image_latents)
            x0_hat = self._decode_x0_for_guidance(pipe,x0_hat,latent_ids)
            loss = verifier.compute_loss(x0_hat)
            grad = torch.autograd.grad(loss, sample)[0]
            v_r += guidance_weight*sigma_r*grad
        else:
            v_r, _, _ = self._ot2vp(model_fn, r, sample, image_latents)
        
        # Diffusion coefficient: w = norm * r^2 (matching SD3)
        w = diffusion_norm * r**2
        
        # Score
        s = self._get_score_vp(v_r, sample, r)
        
        # Drift: v + 0.5 * w * s (matching SD3's get_drift)
        drift = (v_r + 0.5 * w * s) * dt
        
        # Diffusion
        if use_ode:
            prev_sample = sample + drift
        else:
            eps = torch.randn_like(sample)
            diffusion = torch.sqrt(w) * eps * torch.sqrt(torch.abs(dt))
            prev_sample = sample + drift + diffusion
        
        self._step_index += 1
        return (prev_sample,)


# ==============================================================================
# Pipeline
# ==============================================================================

class Flux2KleinVPSDEPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    r"""
    Flux2 Klein pipeline with VP-SDE sampling for text-to-image generation.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2Transformer2DModel,
        is_distilled: bool = False,
        verifier = None
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.register_to_config(is_distilled=is_distilled)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        # Replace scheduler with SDE-capable version
        if not isinstance(scheduler, FlowMatchVPSDEScheduler):
            self.scheduler = FlowMatchVPSDEScheduler.from_config(scheduler.config)

        self.verifier = verifier

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _prepare_image_ids(image_latents: List[torch.Tensor], scale: int = 10):
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)
        return latents, latent_ids

    def prepare_image_latents(
        self,
        images: List[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(image_latent)

        image_latent_ids = self._prepare_image_ids(image_latents)

        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)
            packed = packed.squeeze(0)
            packed_latents.append(packed)

        image_latents = torch.cat(packed_latents, dim=0)
        image_latents = image_latents.unsqueeze(0)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale is not None and guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self.config.is_distilled

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
        diffusion_norm: float = 8.0,
    ):
        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. Process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps - linearly spaced for stable SDE sampling
        # Note: we go from 1000 to near 0, but exclude 0 to avoid division by zero in score
        timesteps = torch.linspace(1000, 0, num_inference_steps + 1, device=device)[:-1]
        self.scheduler.timesteps = timesteps.to(dtype=prompt_embeds.dtype)
        self.scheduler._step_index = None
        self.scheduler._begin_index = None
        self._num_timesteps = len(self.scheduler.timesteps)
        
        # Compute dt once (constant step size, negative since going from 1 to 0)
        dt = -((timesteps[0] - timesteps[1]) / 1000).to(dtype=prompt_embeds.dtype)

        # 7. Denoising loop
        self.scheduler.set_begin_index(0)
        
        # Prepare image ids for model call
        latent_image_ids = latent_ids
        if image_latents is not None:
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, r in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = r

                # Define model function for scheduler
                def model_fn(hidden_states, t_remapped, image_latents):
                    latent_input = hidden_states.to(self.transformer.dtype)
                    if image_latents is not None:
                        latent_input = torch.cat([latent_input, image_latents], dim=1)
                    
                    # t_remapped is in [0, 1], model expects [0, 1]
                    timestep_input = t_remapped.expand(latent_input.shape[0])
                    
                    with self.transformer.cache_context("cond"):
                        pred = self.transformer(
                            hidden_states=latent_input,
                            timestep=timestep_input,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    pred = pred[:, :latents.size(1)]
                    
                    if self.do_classifier_free_guidance:
                        with self.transformer.cache_context("uncond"):
                            neg_pred = self.transformer(
                                hidden_states=latent_input,
                                timestep=timestep_input,
                                guidance=None,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self._attention_kwargs,
                                return_dict=False,
                            )[0]
                        neg_pred = neg_pred[:, :latents.size(1)]
                        pred = neg_pred + guidance_scale * (pred - neg_pred)
                    
                    return pred

                # SDE step (last step is deterministic)
                latents_dtype = latents.dtype
                use_ode = (i == len(self.scheduler.timesteps) - 1)
                latents = self.scheduler.step_sde(model_fn, r, latents, dt=dt, diffusion_norm=diffusion_norm, use_ode=use_ode, image_latents=image_latents)[0]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, r, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        self._current_timestep = None

        # 8. Decode latents
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
    
    def guided_inference(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
        diffusion_norm: float = 8.0,
        guidance_weight: float = 1.0
    ):
        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. Process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        self.verifier.set_reference(condition_images[0])

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps - linearly spaced for stable SDE sampling
        # Note: we go from 1000 to near 0, but exclude 0 to avoid division by zero in score
        timesteps = torch.linspace(1000, 0, num_inference_steps + 1, device=device)[:-1]
        self.scheduler.timesteps = timesteps.to(dtype=prompt_embeds.dtype)
        self.scheduler._step_index = None
        self.scheduler._begin_index = None
        self._num_timesteps = len(self.scheduler.timesteps)
        
        # Compute dt once (constant step size, negative since going from 1 to 0)
        dt = -((timesteps[0] - timesteps[1]) / 1000).to(dtype=prompt_embeds.dtype)

        # 7. Denoising loop
        self.scheduler.set_begin_index(0)
        
        # Prepare image ids for model call
        latent_image_ids = latent_ids
        if image_latents is not None:
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, r in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = r

                # Define model function for scheduler
                def model_fn(hidden_states, t_remapped, image_latents):
                    latent_input = hidden_states.to(self.transformer.dtype)
                    if image_latents is not None:
                        latent_input = torch.cat([latent_input, image_latents], dim=1)
                    
                    # t_remapped is in [0, 1], model expects [0, 1]
                    timestep_input = t_remapped.expand(latent_input.shape[0])
                    
                    with self.transformer.cache_context("cond"):
                        pred = self.transformer(
                            hidden_states=latent_input,
                            timestep=timestep_input,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    pred = pred[:, :latents.size(1)]
                    
                    if self.do_classifier_free_guidance:
                        with self.transformer.cache_context("uncond"):
                            neg_pred = self.transformer(
                                hidden_states=latent_input,
                                timestep=timestep_input,
                                guidance=None,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self._attention_kwargs,
                                return_dict=False,
                            )[0]
                        neg_pred = neg_pred[:, :latents.size(1)]
                        pred = neg_pred + guidance_scale * (pred - neg_pred)
                    
                    return pred

                # SDE step (last step is deterministic)
                latents_dtype = latents.dtype
                use_ode = (i == len(self.scheduler.timesteps) - 1)
                latents = self.scheduler.step_sde(
                    model_fn=model_fn,
                    timestep=r, 
                    sample=latents, 
                    dt=dt, 
                    diffusion_norm=diffusion_norm, 
                    use_ode=use_ode, 
                    image_latents=image_latents,
                    verifier=self.verifier,
                    guidance_weight=guidance_weight,
                    pipe=self,
                    latent_ids=latent_ids
                )[0]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, r, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        self._current_timestep = None

        # 8. Decode latents
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.detach(), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)