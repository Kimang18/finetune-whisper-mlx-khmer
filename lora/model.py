# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn

class LoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 32,
        dropout: float = 0.05,
        alpha: int = 64,
    ):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            alpha=alpha,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = linear.scales.dtype
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 32,
        dropout: float = 0.05,
        alpha: int = 64,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = float(alpha)/math.sqrt(r)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)
