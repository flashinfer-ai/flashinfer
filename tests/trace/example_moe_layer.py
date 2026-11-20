# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Generate all four Frost MoELayer traces using metadata-only tensors."""

from pathlib import Path

import torch

from flashinfer import fused_moe as moe


def example_inputs(dtype):
    config_cls, weight_format, activation_format = {
        "bf16": (moe.CudnnFrostBf16Config, moe.QuantFormat.BF16, moe.QuantFormat.BF16),
        "mxfp8": (
            moe.CudnnFrostMxfp8Config,
            moe.QuantFormat.MXFP8,
            moe.QuantFormat.MXFP8,
        ),
        "nvfp4": (
            moe.CudnnFrostNvfp4Config,
            moe.QuantFormat.NVFP4,
            moe.QuantFormat.NVFP4,
        ),
        "mxfp8_mxfp4": (
            moe.CudnnFrostMxfp8Mxfp4Config,
            moe.QuantFormat.MXFP4,
            moe.QuantFormat.MXFP8,
        ),
    }[dtype]
    config = moe.MoEConfig(
        routing=moe.RoutingConfig(num_experts=64, top_k=6),
        quant=moe.QuantConfig(weight=weight_format, activation=activation_format),
        experts=moe.ExpertConfig(intermediate_size=1408),
        backend=moe.BackendOptions((config_cls(),)),
    )
    # fi_trace only needs config and pack metadata; no CUDA construction/JIT.
    layer = moe.MoELayer.__new__(moe.MoELayer)
    layer.config = config
    activation_dtype = (
        torch.bfloat16
        if dtype == "bf16"
        else (torch.uint8 if dtype == "nvfp4" else torch.float8_e4m3fn)
    )
    packed_weights = dtype in ("nvfp4", "mxfp8_mxfp4")
    weight_dtype = torch.uint8 if packed_weights else activation_dtype
    pack = 2 if packed_weights else 1
    x = torch.empty(
        9, 2048 // (2 if dtype == "nvfp4" else 1), device="meta", dtype=activation_dtype
    )
    sf = (
        None
        if dtype == "bf16"
        else torch.empty(
            9,
            2048 // (16 if dtype == "nvfp4" else 32),
            device="meta",
            dtype=torch.uint8,
        )
    )
    act = moe.MoEActivationPack(
        x,
        sf,
        torch.empty(9, 6, device="meta", dtype=torch.int32),
        torch.empty(9, 6, device="meta", dtype=torch.float32),
    )
    view = {
        "fc1_expert_weights": torch.empty(
            64, 2816, 2048 // pack, device="meta", dtype=weight_dtype
        ),
        "fc2_expert_weights": torch.empty(
            64, 2048, 1408 // pack, device="meta", dtype=weight_dtype
        ),
    }
    if dtype != "bf16":
        block = 16 if dtype == "nvfp4" else 32
        suffix = "weight_block_scale" if dtype == "nvfp4" else "expert_scales"
        view[f"fc1_{suffix}"] = torch.empty(
            64, 2816, 2048 // block, device="meta", dtype=torch.uint8
        )
        view[f"fc2_{suffix}"] = torch.empty(
            64, 2048, 1408 // block, device="meta", dtype=torch.uint8
        )
    if dtype == "nvfp4":
        for name in ("fc1_dequant_scale", "fc2_dequant_scale"):
            view[name] = torch.empty(64, device="meta", dtype=torch.float32)
        for name in ("fc1_act_global_scale", "fc2_act_global_scale"):
            view[name] = torch.empty(1, device="meta", dtype=torch.float32)
    return layer, act, moe.MoEWeightPack({"cudnn_frost_" + dtype: view})


def generate(save_dir):
    for dtype in ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"):
        layer, act, weights = example_inputs(dtype)
        moe.MoELayer.__call__.fi_trace(
            self=layer,
            act_pack=act,
            weight_pack=weights,
            save_dir=save_dir,
            name=f"moe_layer_cudnn_frost_{dtype}",
        )


if __name__ == "__main__":
    generate(Path(__file__).parent / "fi_trace_out")
