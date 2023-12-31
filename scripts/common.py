import os
from typing import Union
from pathlib import Path
import modal
from modal import Stub, Image, Volume, Secret

APP_NAME = "huggingface-train-fsdp"

N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_MEM = int(os.environ.get("GPU_MEM", 80))
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)

image = (
    Image.micromamba(python_version="3.10")
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "numpy>=1.24.4",
        "bitsandbytes",
        "bitsandbytes-cuda117==0.26.0.post2",
        "transformers==4.36.2",
        "accelerate @ git+https://github.com/huggingface/accelerate.git",
        "hf-transfer~=0.1",
        "torch==2.0.1",
        "sentencepiece==0.1.97",
        "huggingface_hub>=0.19.3",
        "einops==0.6.1",
        "datasets",
        "trl",
        "peft",
        "xformers==0.0.22",
        "wandb"
    ).pip_install("flash-attn")
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

stub = Stub(APP_NAME, image=image, secrets=[Secret.from_name("huggingface-secret")])

# Volumes for pre-trained models and training runs.
pretrained_volume = Volume.persisted("pretrained-volume")
finetune_volume = Volume.persisted("finetune-volume")
run_volume = Volume.persisted("run-volume")
VOLUME_CONFIG: dict[Union[str, os.PathLike], Volume] = {
    Path("/pretrained"): pretrained_volume,
    Path("/finetune"): finetune_volume,
    #"/run": run_volume,
}