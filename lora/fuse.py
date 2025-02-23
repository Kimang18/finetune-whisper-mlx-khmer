# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import utils as lora_utils
from mlx.utils import tree_flatten, tree_unflatten

from model import LoRALinear

def upload_to_hub(path: str, name: str, torch_name_or_path: str):
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"Kimang18/{name}"
    text = f"""
---
library_name: mlx
---

# {name}
This model was converted to MLX format from [`{torch_name_or_path}`]().

## Use with mlx
```bash
pip install mlx-whisper
```

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "FILE_NAME",
    path_or_hf_repo={repo_id},
)
```
"""
    card = ModelCard(text)
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
    )

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
    parser.add_argument(
        "--fp16",
        type=bool,
        default=True,
        help="Load data type float 16",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=-1,
        help="Number of layers to fine-tune",
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--original-model",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    dtype = mx.float16 if args.fp16 else mx.float32
    model, tokenizer, config = lora_utils.load(args.model, dtype=dtype)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = int(len([m for m in adapters if "query.lora_a" in m[0]])/3) #NOTE: divide by 2 cuz encoder and decoder were count together

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
    if args.upload_name:
        upload_to_hub(args.save_path, args.upload_name, args.original_model)
