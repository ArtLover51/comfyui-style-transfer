import os
import requests
from huggingface_hub import hf_hub_download

MODELS = {
    "checkpoints/sdxl_base.safetensors": "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors",
    "ipadapter/ip_adapter_sdxl_plus.bin": "h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    "controlnet/depth.safetensors": "diffusers/controlnet-depth-sdxl-1.0/diffusers_xl_depth_mid.safetensors",
    "controlnet/canny.safetensors": "diffusers/controlnet-canny-sdxl-1.0/diffusers_xl_canny_mid.safetensors",
    "upscale/4x-UltraSharp.pth": "https://huggingface.co/uwg/upscaler/resolve/main/4x-UltraSharp.pth"
}

def download_models():
    for path, model_id in MODELS.items():
        os.makedirs(os.path.dirname(f"models/{path}"), exist_ok=True)
        
        if not os.path.exists(f"models/{path}"):
            print(f"Downloading {path}...")
            if "huggingface.co" in model_id:
                response = requests.get(model_id)
                with open(f"models/{path}", "wb") as f:
                    f.write(response.content)
            else:
                hf_hub_download(
                    repo_id=model_id.split('/')[0],
                    filename=model_id.split('/')[-1],
                    local_dir=f"models/{os.path.dirname(path)}"
                )

if __name__ == "__main__":
    download_models()
