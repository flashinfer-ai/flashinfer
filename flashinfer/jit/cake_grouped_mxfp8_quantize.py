"""JIT loader for generated Blackwell grouped MXFP8 quantization kernels."""

import functools
from pathlib import Path
from typing import Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .utils import write_if_different

CakeGroupedMXFP8Target = Literal["sm100a", "sm103a"]
CakeGroupedMXFP8Input = Literal["bfloat16", "float16"]

_PLACEHOLDER_MARKER = "FLASHINFER_CAKE_GROUPED_MXFP8_DEVICE_PLACEHOLDER"
_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_TARGET_MINOR = {"sm100a": 0, "sm103a": 3}
_INPUT_METADATA = {
    "bfloat16": (
        "cake_grouped_mxfp8_quantize_bf16_device.cu",
        "kernel_cake_grouped_mxfp8_quantize_row2d_bf16",
        "dl_bfloat16",
    ),
    "float16": (
        "cake_grouped_mxfp8_quantize_f16_device.cu",
        "kernel_cake_grouped_mxfp8_quantize_row2d_f16",
        "dl_float16",
    ),
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_grouped_mxfp8_quantize"
    if installed.exists():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "cake_grouped_mxfp8_quantize"
    )
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "Cake grouped MXFP8 sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def cake_grouped_mxfp8_target(device: torch.device) -> CakeGroupedMXFP8Target:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        return "sm100a"
    if (major, minor) == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "the Cake grouped MXFP8 backend requires exact compute capability "
        f"10.0 or 10.3, got {major}.{minor}"
    )


def _input_name(dtype: torch.dtype) -> CakeGroupedMXFP8Input:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError(f"unsupported Cake grouped MXFP8 input dtype: {dtype}")


def _device_source(input_name: CakeGroupedMXFP8Input) -> Path:
    return _get_csrc_dir() / _INPUT_METADATA[input_name][0]


def is_cake_grouped_mxfp8_quantize_available(
    dtype: torch.dtype, device: torch.device
) -> bool:
    """Return whether a generated body is installed for this dtype/device."""

    try:
        cake_grouped_mxfp8_target(device)
        source = _device_source(_input_name(dtype))
        return source.is_file() and _PLACEHOLDER_MARKER not in source.read_text()
    except (FileNotFoundError, RuntimeError, TypeError, OSError):
        return False


def _binding_source(
    input_name: CakeGroupedMXFP8Input,
) -> str:
    body, symbol, dl_dtype = _INPUT_METADATA[input_name]
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_GROUPED_MXFP8_BODY_FILE "{body}"
#define CAKE_GROUPED_MXFP8_KERNEL {symbol}
#define CAKE_GROUPED_MXFP8_INPUT_DLTYPE {dl_dtype}
#include "cake_grouped_mxfp8_quantize_binding.cuh"
"""


@functools.cache
def gen_cake_grouped_mxfp8_quantize_module(
    input_name: CakeGroupedMXFP8Input,
    target: CakeGroupedMXFP8Target,
) -> JitSpec:
    if input_name not in _INPUT_METADATA:
        raise ValueError(f"unsupported Cake grouped MXFP8 input: {input_name}")
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake grouped MXFP8 target: {target}")

    csrc_dir = _get_csrc_dir()
    body = _device_source(input_name)
    if not body.is_file():
        raise FileNotFoundError(f"generated Cake grouped MXFP8 body not found: {body}")
    if _PLACEHOLDER_MARKER in body.read_text():
        raise RuntimeError(
            "the Cake grouped MXFP8 device body is a placeholder; install an "
            f"exported {input_name} profile before selecting backend='cake'"
        )
    for required in (
        "cake_grouped_mxfp8_quantize_binding.cuh",
        "cake_grouped_mxfp8_quantize_launch.cuh",
    ):
        if not (csrc_dir / required).is_file():
            raise FileNotFoundError(
                f"Cake grouped MXFP8 host source not found: {csrc_dir / required}"
            )

    uri = f"cake_grouped_mxfp8_quantize_{input_name}_{target}"
    binding = (
        jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_grouped_mxfp8_quantize_binding.cu"
    )
    write_if_different(binding, _binding_source(input_name))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            f"-DFLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR={_TARGET_MINOR[target]}",
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(
        "Generated Cake grouped MXFP8 %s %s JIT spec: %s",
        input_name,
        target,
        spec.name,
    )
    return spec


@functools.cache
def load_cake_grouped_mxfp8_quantize_module(
    input_name: CakeGroupedMXFP8Input,
    target: CakeGroupedMXFP8Target,
):
    module = gen_cake_grouped_mxfp8_quantize_module(input_name, target).build_and_load()
    logger.info("Loaded Cake grouped MXFP8 %s %s module", input_name, target)
    return module


def get_cake_grouped_mxfp8_quantize_module(dtype: torch.dtype, device: torch.device):
    return load_cake_grouped_mxfp8_quantize_module(
        _input_name(dtype), cake_grouped_mxfp8_target(device)
    )


__all__ = [
    "CakeGroupedMXFP8Input",
    "CakeGroupedMXFP8Target",
    "cake_grouped_mxfp8_target",
    "gen_cake_grouped_mxfp8_quantize_module",
    "get_cake_grouped_mxfp8_quantize_module",
    "is_cake_grouped_mxfp8_quantize_available",
    "load_cake_grouped_mxfp8_quantize_module",
]
