"""JIT loader for the generated Blackwell grouped MXFP8 quantization kernels."""

import functools
from pathlib import Path
from typing import Any, Literal

import torch
import tvm_ffi

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

CakeGroupedMXFP8Target = Literal["sm100a", "sm103a"]
CakeGroupedMXFP8Input = Literal["bfloat16", "float16"]

# Generated programs of the grouped MXFP8 quantizer (one per input dtype; the
# BF16 and FP16 programs differ in their widening sequence).  ``MODULES`` holds
# one record per program: the kernel and launcher translation units under
# ``csrc/cake_grouped_mxfp8_quantize``, the architectures the source compiles
# for, compile flags, FFI entry, argument plan and kernel symbol.  ``KERNELS``
# maps the input dtype to its program.  Both literals are populated by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_grouped_mxfp8_quantize_8d0f000cf214bc5e1375": {
        "sources": [
            "cake_grouped_mxfp8_quantize_8d0f000cf214bc5e1375_kernel.cu",
            "cake_grouped_mxfp8_quantize_8d0f000cf214bc5e1375_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "mask"],
            ["buffer", "quantized"],
            ["buffer", "scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "PADDED_K"],
            ["parameter", "PM_TILES"],
            ["parameter", "PK_TILES"],
            ["parameter", "BLOCKS_PER_ROW"],
            ["parameter", "TOTAL_TASKS"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "kernel": "kernel_cake_grouped_mxfp8_quantize_8d0f000cf214bc5e1375",
    },
    "cake_grouped_mxfp8_quantize_e5d70aeb234fb379ed05": {
        "sources": [
            "cake_grouped_mxfp8_quantize_e5d70aeb234fb379ed05_kernel.cu",
            "cake_grouped_mxfp8_quantize_e5d70aeb234fb379ed05_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "mask"],
            ["buffer", "quantized"],
            ["buffer", "scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "PADDED_K"],
            ["parameter", "PM_TILES"],
            ["parameter", "PK_TILES"],
            ["parameter", "BLOCKS_PER_ROW"],
            ["parameter", "TOTAL_TASKS"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "kernel": "kernel_cake_grouped_mxfp8_quantize_e5d70aeb234fb379ed05",
    },
}
KERNELS: dict[str, str] = {
    "bfloat16": "cake_grouped_mxfp8_quantize_8d0f000cf214bc5e1375",
    "float16": "cake_grouped_mxfp8_quantize_e5d70aeb234fb379ed05",
}

_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_INPUT_NAMES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
_QUANT_BLOCK = 32
_SCALE_TILE = 128
_THREADS = 128


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
        f"Cake grouped MXFP8 sources were not found. Checked:\n  - {installed}\n  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        f"FlashInfer headers were not found. Checked:\n  - {jit_env.FLASHINFER_INCLUDE_DIR}\n  - {checkout}"
    )


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(
            f"the Cake grouped MXFP8 backend requires a CUDA device, got {device}"
        )
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _target_for_device(device_index: int) -> CakeGroupedMXFP8Target:
    major, minor = torch.cuda.get_device_capability(device_index)
    if (major, minor) == (10, 0):
        return "sm100a"
    if (major, minor) == (10, 3):
        return "sm103a"
    raise RuntimeError(
        f"the Cake grouped MXFP8 backend requires exact compute capability 10.0 or 10.3, got {major}.{minor}"
    )


def cake_grouped_mxfp8_target(device: torch.device) -> CakeGroupedMXFP8Target:
    """The exact compile target of ``device`` (cached per device index)."""
    return _target_for_device(_device_index(device))


def _input_name(dtype: torch.dtype) -> CakeGroupedMXFP8Input:
    name = _INPUT_NAMES.get(dtype)
    if name is None:
        raise TypeError(f"unsupported Cake grouped MXFP8 input dtype: {dtype}")
    return name


@functools.cache
def _available(input_name: str, device_index: int) -> bool:
    if input_name not in KERNELS:
        return False
    try:
        _target_for_device(device_index)
    except RuntimeError:
        return False
    return True


def is_cake_grouped_mxfp8_quantize_available(
    dtype: torch.dtype, device: torch.device
) -> bool:
    """Whether a generated program serves ``dtype`` on ``device`` (no I/O; cached)."""
    name = _INPUT_NAMES.get(dtype)
    if name is None or device.type != "cuda":
        return False
    return _available(name, _device_index(device))


@functools.cache
def gen_cake_grouped_mxfp8_quantize_module(
    input_name: CakeGroupedMXFP8Input,
    target: CakeGroupedMXFP8Target,
) -> JitSpec:
    if input_name not in KERNELS:
        raise ValueError(f"unsupported Cake grouped MXFP8 input: {input_name}")
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake grouped MXFP8 target: {target}")
    record = MODULES[KERNELS[input_name]]
    if _TARGET_ARCH[target] not in record["arches"]:
        raise ValueError(
            f"the {input_name} program is not built for {target} ({record['arches']})"
        )
    csrc_dir = _get_csrc_dir()
    # gen_jit_spec's default use_fast_math (True) is left unchanged here,
    # matching the recipe the previously shipped binding used. The scale
    # divide (absmax / 448) in the generated source stays the bare `/`
    # operator rather than an explicit round-to-nearest intrinsic, so it
    # compiles to the same approximate-reciprocal sequence as the kernel
    # already shipped under this flag; a provably exact divide here is a
    # numerics change outside this delivery's scope and is tracked as a
    # maintainer decision item on the tracking issue.
    spec = gen_jit_spec(
        name=f"cake_grouped_mxfp8_quantize_{input_name}_{target}",
        sources=[csrc_dir / relative for relative in record["sources"]],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(
        "Generated Cake grouped MXFP8 %s %s JIT spec: %s", input_name, target, spec.name
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


@functools.cache
def _entry(input_name: str, target: str):
    module = load_cake_grouped_mxfp8_quantize_module(input_name, target)
    record = MODULES[KERNELS[input_name]]
    return getattr(module, record["ffi_entry"]), tuple(
        (kind, argument) for kind, argument in record["arg_plan"]
    )


def cake_grouped_mxfp8_quantize_launch(
    a: torch.Tensor,
    mask: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Quantize ``a`` ``[B, M, K]`` into the caller-owned ``quantized``
    ``[B, M, padded_K]`` (E4M3) and ``scales`` ``[B, padded_M, padded_K / 32]``
    (UE8M0 bytes in the 128x4 grouped-GEMM layout) on the current stream.

    No allocation, no synchronization; the generated launcher rejects
    non-dense views, so callers pass contiguous tensors.  Empty problems
    (``B == 0`` or ``M == 0``) write nothing.
    """
    b, m, k = a.shape
    if b == 0 or m == 0:
        return
    padded_k = quantized.shape[2]
    padded_m = scales.shape[1]
    blocks_per_row = padded_k // _QUANT_BLOCK
    pm_tiles = padded_m // _SCALE_TILE
    pk_tiles = padded_k // _SCALE_TILE
    entry, arg_plan = _entry(_input_name(a.dtype), cake_grouped_mxfp8_target(a.device))
    # The programs read two-byte elements through one pointer type; the FP16
    # program widens with cvt.f32.f16, so its storage is passed as-is (a view).
    values = dict(
        x=a if a.dtype == torch.bfloat16 else a.view(torch.bfloat16),
        mask=mask,
        quantized=quantized,
        scales=scales.view(b, pm_tiles, pk_tiles, 32, 4, 4),
        M=m,
        K=k,
        PADDED_K=padded_k,
        PM_TILES=pm_tiles,
        PK_TILES=pk_tiles,
        BLOCKS_PER_ROW=blocks_per_row,
        TOTAL_TASKS=b * m * blocks_per_row,
        grid_x=(blocks_per_row + _THREADS - 1) // _THREADS,
        grid_y=m,
        grid_z=b,
    )
    arguments = [values[argument] for _kind, argument in arg_plan]
    with tvm_ffi.use_torch_stream():
        entry(*arguments)


__all__ = [
    "CakeGroupedMXFP8Input",
    "CakeGroupedMXFP8Target",
    "cake_grouped_mxfp8_quantize_launch",
    "cake_grouped_mxfp8_target",
    "gen_cake_grouped_mxfp8_quantize_module",
    "get_cake_grouped_mxfp8_quantize_module",
    "is_cake_grouped_mxfp8_quantize_available",
    "load_cake_grouped_mxfp8_quantize_module",
]
