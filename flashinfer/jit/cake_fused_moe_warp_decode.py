"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from pathlib import Path
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

CakeWarpDecodeTarget = Literal["sm103a"]

_MODULE_URI = "cake_fused_moe_warp_decode_sm103a"
_GENERATED_SOURCE = "cake_adaptive_warp_decode_kernels.cu"
_BINDING_SOURCE = "cake_warp_decode_binding.cu"
_GENERATED_MANIFEST = "cake_warp_decode_generated_manifest.cuh"
_CONTRACT_HEADER = "cake_warp_decode_contract.cuh"


def _get_cake_fused_moe_warp_decode_csrc_dir() -> Path:
    """Locate Cake warp-decode sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "warp_decode"
    if installed.exists():
        return installed

    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "fused_moe" / "warp_decode"
    )
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "Cake warp-decode CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

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


def get_cake_fused_moe_warp_decode_uri(
    target: CakeWarpDecodeTarget = "sm103a",
) -> str:
    """Return the exact-SM103a Cake warp-decode JIT module key."""

    if target != "sm103a":
        raise ValueError(f"unsupported Cake warp-decode target: {target}")
    return _MODULE_URI


@functools.cache
def gen_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
) -> JitSpec:
    """Generate the exact-SM103a Cake warp-decode JIT module."""

    uri = get_cake_fused_moe_warp_decode_uri(target)
    csrc_dir = _get_cake_fused_moe_warp_decode_csrc_dir()
    generated_dir = csrc_dir / "generated"
    required_files = (
        generated_dir / _GENERATED_SOURCE,
        csrc_dir / _BINDING_SOURCE,
        generated_dir / _GENERATED_MANIFEST,
        csrc_dir / _CONTRACT_HEADER,
    )
    for source in required_files:
        if not source.is_file():
            raise FileNotFoundError(f"Cake warp-decode source not found: {source}")

    # gen_jit_spec supplies the common optimization flags, including
    # -use_fast_math. The flags below add exactly one SM103a code-generation
    # target and the block-scaled FP4 feature defines.
    spec = gen_jit_spec(
        name=uri,
        sources=[
            generated_dir / _GENERATED_SOURCE,
            csrc_dir / _BINDING_SOURCE,
        ],
        extra_cuda_cflags=[*sm103a_nvcc_flags],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            csrc_dir,
            generated_dir,
            csrc_dir.parents[1],
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated Cake warp-decode {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def _build_and_load_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
) -> Any:
    module = gen_cake_fused_moe_warp_decode_module(target).build_and_load()
    logger.info(f"Loaded Cake warp-decode {target} module")
    return module


def _get_compute_capability(device: Any = None) -> tuple[int, int]:
    # Keep the heavyweight runtime dependency out of module import. This JIT
    # module is also imported by source-only packaging and AOT tooling.
    import torch  # noqa: PLC0415

    from ..utils import get_compute_capability  # noqa: PLC0415

    resolved_device = torch.device("cuda") if device is None else torch.device(device)
    return get_compute_capability(resolved_device)


def _check_exact_sm103a(device: Any = None) -> None:
    major, minor = _get_compute_capability(device)
    if (major, minor) != (10, 3):
        raise RuntimeError(
            "Cake warp decode requires exact compute capability 10.3, "
            f"got {major}.{minor}"
        )


def load_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
    *,
    device: Any = None,
) -> Any:
    """Build or load the module after checking the requested CUDA device."""

    get_cake_fused_moe_warp_decode_uri(target)
    _check_exact_sm103a(device)
    return _build_and_load_cake_fused_moe_warp_decode_module(target)


def get_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
    *,
    device: Any = None,
) -> Any:
    """Return the module exporting size, prepare, launch, and receipt release."""

    return load_cake_fused_moe_warp_decode_module(target, device=device)


__all__ = [
    "CakeWarpDecodeTarget",
    "gen_cake_fused_moe_warp_decode_module",
    "get_cake_fused_moe_warp_decode_module",
    "get_cake_fused_moe_warp_decode_uri",
    "load_cake_fused_moe_warp_decode_module",
]
