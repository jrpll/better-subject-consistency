import sys
sys.path.insert(0, "dinov3")

import torch
import torch.nn.functional as F
from flux2_klein_w_grads_pipeline import Flux2KleinPipeline
import gc
from PIL import Image
from dinov3.data.transforms import make_classification_eval_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import CosineSimilarity

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

    # Enable gradient checkpointing to reduce memory usage
    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.enable_gradient_checkpointing()

    # Use gradient-enabled generation (returns [-1, 1] range, (B, C, H, W) format)
    image_tensor = pipe.__call_with_grad__(
        prompt_embeds=prompt_embeds,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4
    )

    # Convert to PIL: denormalize [-1,1] -> [0,1] -> [0,255], permute, numpy, uint8
    image_pil = Image.fromarray(
        ((image_tensor[0] / 2 + 0.5).clamp(0, 1) * 255)
        .permute(1, 2, 0).detach().cpu().float().numpy().round().astype("uint8")
    )
    image_pil.save("output_with_grad.png")
    print("Saved image to output_with_grad.png")

    dinov3_vits16plus = torch.hub.load(
        "dinov3",
        'dinov3_vits16plus',
        source='local',
        weights="./dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    ).cuda().eval()

    sanity_check_image = Image.open("/home_pers/jripoll/vlm-editor/sanitycheck.png").convert("RGB")

    # Standard DINOv3 transform for PIL images
    pil_transform = make_classification_eval_transform()
    
    pil_img_transformed = pil_transform(image_pil)
    sanity_check_img_transformed = pil_transform(sanity_check_image)
    tensor_img_transformed = tensor_transform(image_tensor)

    with torch.no_grad():
        features_tensor = dinov3_vits16plus(tensor_img_transformed.float())
        features_pil = dinov3_vits16plus(pil_img_transformed.unsqueeze(0).cuda().float())
        features_sanity_check = dinov3_vits16plus(sanity_check_img_transformed.unsqueeze(0).cuda().float())

    sim = CosineSimilarity()

    print("gt sim",sim(features_tensor,features_pil))
    print("bad sim",sim(features_tensor,features_sanity_check))

if __name__ == "__main__":
    main()
