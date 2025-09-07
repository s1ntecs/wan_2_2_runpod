#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

from huggingface_hub import hf_hub_download

from styles import STYLE_URLS_UNIQUE


model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def download_model():
    if not os.path.exists("./hf_cache"):
        os.makedirs("./hf_cache")
    print("Downloading model...")
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        cache_dir="./hf_cache"
    )

# def download_wan():
#     image_encoder = CLIPVisionModel.from_pretrained(
#     model_id, subfolder="image_encoder", torch_dtype=torch.float32
#     )
#     vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
#     pipe = WanImageToVideoPipeline.from_pretrained(
#         model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
#     )


def download_loras():
    downloaded_paths = []
    for key, lora_url in STYLE_URLS_UNIQUE.items():
        try:
            parts = lora_url.split("/")
            repo_id = "/".join(parts[3:5])
            filename = parts[-1]
            print(f"Downloading LoRA: key={key}, repo_id={repo_id}, filename={filename}")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./loras"
            )
            downloaded_paths.append(local_path)
        except Exception as e:
            print(f"Error downloading LoRA '{key}':", str(e))
            # либо продолжаем (continue), либо прерываем полностью, в зависимости от задачи
            raise RuntimeError(f"Failed to download LoRA '{key}': {str(e)}")
    return downloaded_paths


if __name__ == "__main__":
    # download_loras()
    # download_wan() 
    download_model()
