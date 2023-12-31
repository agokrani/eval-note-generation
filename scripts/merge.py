import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import secrets
import modal
from common import (
    stub,
    VOLUME_CONFIG,
    image,
    finetune_volume, 
    GPU_CONFIG
)
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()

@stub.function(gpu=GPU_CONFIG, timeout=3600, volumes=VOLUME_CONFIG)
def main(args): 
    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    # print(model.state_dict())
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()
    if args.tokenizer_path is not None: 
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")

    finetune_volume.commit()

if __name__ == "__main__":
    args = get_args()
    with stub.run():
        main.remote(args)