
import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

# def civitai_download(subdir: str, filename: str, url: str) -> str:
#     return (
#         f"comfy --skip-prompt model download --url '{url}'"
#         f" --relative-path 'models/{subdir}'"
#         f" --filename '{filename}'"
#         f" --set-civitai-api-token $7b595cfb2e600158e6d1ff5befe9c3f2"
#     )

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Model download tasks (will be done at runtime)
model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "sasha-000004.safetensors", "alex-000008.safetensors", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

extra_cmds = [
    f"wget https://civitai.com/models/1459553/sasha-flux-1 -P {MODELS_DIR}/loras",
#     civitai_download("upscale_models", "4xUltrasharp_4xUltrasharpV10.pt", "https://civitai.com/api/download/models/125843?type=Model&format=PickleTensor")
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=30,
    timeout=1800,
    gpu="A100",
    volumes={DATA_ROOT: vol},
    # secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("civitai-token")]
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        
        # Ensure DATA_ROOT exists
        os.makedirs(DATA_ROOT, exist_ok=True)
        
        # Copy ComfyUI from default location to volume
        if os.path.exists(DEFAULT_COMFY_DIR):
            print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
            os.makedirs(DATA_BASE, exist_ok=True)
    else:
        print("Volume already initialized, using existing ComfyUI installation")
    
    # Ensure all required directories exist
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Set environment variables for secrets
#     hf_token = os.getenv("HF_API_TOKEN")
#     civ_token = os.getenv("CIVITAI_API_TOKEN")
    
#     if hf_token:
#         os.environ["HF_API_TOKEN"] = hf_token
#         print("HF token configured")
    
#     if civ_token:
#         os.environ["CIVITAI_API_TOKEN"] = civ_token
#         print("Civitai token configured")

    # Download models at runtime (only if missing)
    print("Checking and downloading missing models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn} to {target}...")
            try:
                hf_download(sub, fn, repo, subf)
                print(f"Successfully downloaded {fn}")
            except Exception as e:
                print(f"Error downloading {fn}: {e}")
        else:
            print(f"Model {fn} already exists, skipping download")

    # Run extra download commands
    print("Running additional downloads...")
    for cmd in extra_cmds:
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Command completed successfully")
            else:
                print(f"Command failed with return code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"Error running command {cmd}: {e}")

    # Set COMFY_DIR environment variable to volume location
    os.environ["COMFY_DIR"] = DATA_BASE
    
    # Launch ComfyUI from volume location
    print(f"Starting ComfyUI from {DATA_BASE}...")
    
    # Change to ComfyUI directory and launch
    os.chdir(DATA_BASE)
    
    # Start ComfyUI server with correct syntax
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"]
    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
