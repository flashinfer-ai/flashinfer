"""
JIT registration for the VibeCUDA Kimi-K3 AttnRes backends.

One exact-architecture JIT module per supported target: each module compiles
the per-architecture kernel payload
(``vibecuda_kimi_k3_attn_res_kernels_{sm100,sm103}.cu``) together with the
shared TVM-FFI binding (``vibecuda_kimi_k3_attn_res_binding.cu``).  The
payloads are exact-architecture programs (SM100 tcgen05/TMEM persistent
routes), so no other architecture is registered: selecting the backend on an
unsupported target raises ``NotImplementedError`` from
:func:`arch_for_device`.
"""

import functools
import hashlib
from pathlib import Path
from typing import TypedDict

import torch

from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ...jit import env as jit_env

_CSRC_ROOT = Path(__file__).resolve().parent / "csrc" / "vibecuda_kimi_k3_attn_res"


class _ModuleSpec(TypedDict):
    payload: str
    flags: list[str]


_MODULES: dict[str, _ModuleSpec] = {
    "sm_100a": {
        "payload": "vibecuda_kimi_k3_attn_res_kernels_sm100.cu",
        "flags": sm100a_nvcc_flags,
    },
    "sm_103a": {
        "payload": "vibecuda_kimi_k3_attn_res_kernels_sm103.cu",
        "flags": sm103a_nvcc_flags,
    },
}

_BINDING = "vibecuda_kimi_k3_attn_res_binding.cu"


def registered_arches() -> tuple[str, ...]:
    """Every architecture with a registered VibeCUDA AttnRes module."""
    return tuple(sorted(_MODULES))


def arch_for_device(device: torch.device) -> str:
    """The registered architecture of ``device`` (loud on unsupported)."""
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise NotImplementedError(
        "the VibeCUDA Kimi-K3 AttnRes backend only supports SM100 (GB100/GB200, "
        f"CC 10.0) and SM103 (B300, CC 10.3); device {device} reports compute "
        f"capability {major}.{minor}"
    )


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_vibecuda_kimi_k3_attn_res_module(arch: str):
    record = _MODULES[arch]
    sources = [_CSRC_ROOT / record["payload"], _CSRC_ROOT / _BINDING]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    name = (
        "vibecuda_kimi_k3_attn_res_"
        + arch.replace("_", "")
        + "_"
        + digest.hexdigest()[:16]
    )
    return gen_jit_spec(
        name=name,
        sources=sources,
        extra_cuda_cflags=[*record["flags"], "-O3"],
        extra_include_paths=[_CSRC_ROOT, *_header_dirs()],
        # The payloads ship their own exact rounding discipline (packed
        # f32x2 round-to-nearest asm); --use_fast_math must stay off.
        use_fast_math=False,
    )


@functools.cache
def load_vibecuda_kimi_k3_attn_res_module(arch: str):
    return gen_vibecuda_kimi_k3_attn_res_module(arch).build_and_load()
