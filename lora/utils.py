import glob
import json
import logging
from pathlib import Path
import math

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten, tree_unflatten

from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.load_models import load_model

import mlx_whisper.whisper as whisper

from model import LoRALinear

import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

class ModelHodler:
    model = None
    model_path = None

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        if cls.model is None or model_path != cls.model_path:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
        return cls.model, Path(cls.model_path)


def linear_to_lora_layers(
    model: nn.Module,
    num_layers: int,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        rank, scale, and optional layer keys.
    """
    rank, alpha, dropout = 64, 16, 0.1 # Stage 1
    # rank, alpha, dropout = 32, 64, 0.1
    if num_layers > len(model.blocks):
        raise ValueError(
            f"Requested {num_layers} LoRA layers "
            f"but the model only has {len(model.blocks)} layers."
        )

    def to_lora(layer):
        return LoRALinear.from_base(
            layer,
            r=rank,
            alpha=alpha,
            dropout=dropout,
        )
    # keys = set(["query", "key", "value", "out", "mlp1", "mlp2"]) # State 1
    keys = set(["query", "key", "value", "out"])
    for b in model.blocks[-max(num_layers, 0) :]:
        lora_layers = [(k, to_lora(l)) for k, l in b.attn.named_modules() if k in keys]
        if lora_layers:
            b.attn.update_modules(tree_unflatten(lora_layers))
        if b.cross_attn:
            lora_layers = [(k, to_lora(l)) for k, l in b.cross_attn.named_modules() if k in keys]
            if lora_layers:
                b.cross_attn.update_modules(tree_unflatten(lora_layers))

        # lora_modules = [(k, to_lora(l)) for k, l in b.named_modules() if k in keys]
        # if lora_modules:
        #     b.update_modules(tree_unflatten(lora_modules))

    # lora_modules = [(k, to_lora(l)) for k, l in model.named_modules() if k in keys]
    # if lora_modules:
    #     model.update_modules(tree_unflatten(lora_modules))

def linear_to_lora(model, num_layers, verbose=True):
    # print(f"Applying LoRA parameters to AudioEncoder...")
    log.info(f"Applying LoRA parameters to AudioEncoder...")
    linear_to_lora_layers(model.encoder, num_layers)
    if verbose:
        # print("Done applying Encoder LoRA Linear layers")
        log.info("Done applying Encoder LoRA Linear layers")
        enc_tot_params = (
            sum(v.size for _, v in tree_flatten(model.encoder.parameters())) / 10**6
        )
        # print(f"Encoder: Total parameters {enc_tot_params:.3f}M")
        log.info(f"Encoder: Total parameters {enc_tot_params:.3f}M")
        enc_tra_params = (
            sum(v.size for _, v in tree_flatten(
                model.encoder.trainable_parameters()))
            / 10**6
        )
        log.info(f"Encoder: Trainable parameters {enc_tra_params:.3f}M")

    log.info(f"Applying LoRA parameters to TextDecoder...")
    linear_to_lora_layers(model.decoder, num_layers)
    if verbose:
        log.info("Done applying Decoder LoRA Linear layers")
        dec_tot_params = (
            sum(v.size for _, v in tree_flatten(model.decoder.parameters())) / 10**6
        )
        log.info(f"Decoder: Total parameters {dec_tot_params:.3f}M")
        dec_tra_params = (
            sum(v.size for _, v in tree_flatten(
                model.decoder.trainable_parameters()))
            / 10**6
        )
        log.info(f"Decoder: Trainable parameters {dec_tra_params:.3f}M")
        log.info("Finished adding LoRA params! :)")


def load(path_or_hf_repo: str, dtype=mx.float16, tokenizer_config={}):
    model, model_path = ModelHodler.get_model(path_or_hf_repo, dtype=dtype)

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    tokenizer = get_tokenizer(
        model.is_multilingual,
        **tokenizer_config,
        num_languages=model.num_languages,
        task="transcribe"
    )
    return model, tokenizer, config


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def make_shards_0(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        estimated_size = v.size * v.dtype.size
        if shard_size + estimated_size > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += estimated_size
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "weights-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "weights.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i+1, shards_count)
        mx.save_safetensors(
            str(save_dir / shard_name), shard, metadata={"format": "mlx"}
        )
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    # todo: whisper-lora: investigate if `tokenizer.save_pretrained(save_dir)` is necessary?
    # tokenizer.save_pretrained(save_dir)
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )

def save_model_0(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shards = make_shards_0(weights)
    for i, shard in enumerate(shards):
        # TODO use HF file name scheme for simplicity
        mx.savez(str(save_dir / f"weights.npz"), **shard)
        mx.save_safetensors(
            str(save_dir / f"weights.{i:02d}.safetensors"), shard)
    # todo: whisper-lora: investigate if `tokenizer.save_pretrained(save_dir)` is necessary?
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)
