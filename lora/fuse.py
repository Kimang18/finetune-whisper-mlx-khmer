# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import utils as lora_utils
from mlx.utils import tree_flatten, tree_unflatten

from model import LoRALinear

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="lora_fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path to the trained adapter weights (npz or safetensors).",
    )

    args = parser.parse_args()

    dtype = mx.float16 if args.get("fp16", True) else mx.float32
    model, tokenizer, config = lora_utils.load(args.model, dtype=dtype)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = int(len([m for m in adapters if "query.lora_a" in m[0]])/2) #NOTE: divide by 2 cuz encoder and decoder were count together

    # Freeze all layers other than LORA linears
    model.freeze()
    lora_utils.linear_to_lora(model, lora_layers, verbose=False)

    model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.fuse())
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    model.update_modules(tree_unflatten(fused_linears))

    weights = dict(tree_flatten(model.parameters()))
    lora_utils.save_model(args.save_path, weights, tokenizer, config)
