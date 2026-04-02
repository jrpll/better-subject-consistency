import sys
sys.path.insert(0, "dinov3")

import torch
import torch.nn.functional as F
from flux2_klein_w_grads_pipeline import Flux2KleinPipeline
import gc
from PIL import Image
from dinov3.data.transforms import make_classification_eval_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import CosineSimilarity
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from tqdm import tqdm

def tensor_transform(x, resize_size=256, crop_size=224):
    """
    Transform for tensor input from VAE output.
    Expects input in [-1, 1] range, (B, C, H, W) format.
    """
    # 1. Convert from [-1, 1] to [0, 1]
    x = x / 2 + 0.5

    # 2. Resize smallest side to resize_size using BICUBIC
    _, _, h, w = x.shape
    if h < w:
        new_h = resize_size
        new_w = int(w * resize_size / h)
    else:
        new_w = resize_size
        new_h = int(h * resize_size / w)
    x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)

    # 3. Center crop to crop_size
    _, _, h, w = x.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    x = x[:, :, top:top+crop_size, left:left+crop_size]

    # 4. ImageNet normalization
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def main():

    dinov3_vits16plus = torch.hub.load(
        "dinov3",
        'dinov3_vits16plus',
        source='local',
        weights="./dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    ).cuda().eval()

    ref_img = Image.open("/home_pers/jripoll/vlm-editor/sanitycheck.png").convert("RGB")

    # Standard DINOv3 transform for PIL images
    pil_transform = make_classification_eval_transform()
    
    ref_img_transformed = pil_transform(ref_img)

    with torch.no_grad():
        features_ref_img = dinov3_vits16plus(ref_img_transformed.unsqueeze(0).cuda().float())

    sim = CosineSimilarity()

    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", 
        torch_dtype=dtype, 
        transformer=None, 
        vae=None
    ).to("cuda")

    prompt = "a person's face"
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt)
    del pipe
    flush()

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=dtype,
        text_encoder=None
    ).to("cuda")

    # Add LoRA adapters to transformer
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.enable_gradient_checkpointing()

    # Setup optimizer for LoRA parameters
    lora_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    base_lr = 1e-3
    optimizer = bnb.optim.Adam8bit(lora_params, lr=base_lr)

    pipe.set_progress_bar_config(disable=True)

    # Training config
    num_optimizer_steps = 100
    seed = 42
    warmup_steps = 20
    max_grad_norm = 1.0  # Gradient clipping

    # Loss tracking
    loss_history = []
    ema_loss = None
    ema_alpha = 0.1  # Exponential moving average smoothing factor

    # Training loop
    pbar = tqdm(range(num_optimizer_steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad()

        # Use fixed seed for reproducible latents
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate image with gradients
        image_tensor = pipe.__call_with_grad__(
            prompt_embeds=prompt_embeds,
            generator=generator,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4
        )

        tensor_img_transformed = tensor_transform(image_tensor)
        features_tensor_img = dinov3_vits16plus(tensor_img_transformed.float())

        loss = 1 - sim(features_tensor_img, features_ref_img)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)

        # Learning rate warmup
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.step()

        # Update EMA loss
        current_loss = loss.item()
        loss_history.append(current_loss)
        if ema_loss is None:
            ema_loss = current_loss
        else:
            ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

        # Compute trend (compare current EMA to EMA from 10 steps ago)
        trend = ""
        if len(loss_history) > 10:
            old_ema = sum(loss_history[-20:-10]) / min(10, len(loss_history[-20:-10]))
            if ema_loss < old_ema - 0.005:
                trend = "↓"
            elif ema_loss > old_ema + 0.005:
                trend = "↑"
            else:
                trend = "→"

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{current_loss:.4f}", ema=f"{ema_loss:.4f}", trend=trend, lr=f"{current_lr:.1e}")

    print("\nTraining completed!")
    print(f"Initial loss: {loss_history[0]:.6f} -> Final loss: {loss_history[-1]:.6f}")

    # Final validation generation
    print("\nGenerating final validation image...")
    final_image = pipe(
        prompt_embeds=prompt_embeds,
        generator=generator,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4
    ).images[0]
    final_image.save("final_trained_output.png")
    print("Saved final image to final_trained_output.png")

if __name__ == "__main__":
    main()
