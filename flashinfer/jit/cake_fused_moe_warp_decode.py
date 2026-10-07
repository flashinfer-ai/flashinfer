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
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

CakeWarpDecodeTarget = Literal["sm100a", "sm103a"]

_TARGET_FLAGS: dict[CakeWarpDecodeTarget, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_MINOR: dict[CakeWarpDecodeTarget, int] = {"sm100a": 0, "sm103a": 3}
_MODULE_URI: dict[CakeWarpDecodeTarget, str] = {
    "sm100a": "cake_fused_moe_warp_decode_sm100a",
    "sm103a": "cake_fused_moe_warp_decode_sm103a",
}
_BINDING_SOURCE = "cake_warp_decode_binding.cu"
_GENERATED_MANIFEST = "cake_warp_decode_generated_manifest.cuh"
_CONTRACT_HEADER = "cake_warp_decode_contract.cuh"

# Device translation units per exact target. ``generated/common`` holds the
# kernels whose source is the same for SM100a and SM103a; the per-target
# directories hold the kernels that differ. The generated manifest header
# declares every symbol listed here for the matching target.
_COMMON_SOURCES: tuple[str, ...] = (
    "cake_warp_decode_0686b950e5dc33f7f54f_kernel.cu",
    "cake_warp_decode_2d89b2ccfa19074dd9fb_kernel.cu",
    "cake_warp_decode_345efbf7077827bc328e_kernel.cu",
    "cake_warp_decode_3aa297e8ff45f1333ec9_kernel.cu",
    "cake_warp_decode_510ad9e9035da24674b6_kernel.cu",
    "cake_warp_decode_514c77f65f72ffd8d0a5_kernel.cu",
    "cake_warp_decode_5312a73885ff3541a6b6_kernel.cu",
    "cake_warp_decode_5c23234a34054124f4ce_kernel.cu",
    "cake_warp_decode_61f9c98fb7c8656ae154_kernel.cu",
    "cake_warp_decode_7d6f94378c1107da3804_kernel.cu",
    "cake_warp_decode_8388b2049117830020f2_kernel.cu",
    "cake_warp_decode_8acc0c77f5b6d2888496_kernel.cu",
    "cake_warp_decode_958ca99134107b938d76_kernel.cu",
    "cake_warp_decode_9ace9f7e5e0c273ae879_kernel.cu",
    "cake_warp_decode_a11aa740b9780f90e7ee_kernel.cu",
    "cake_warp_decode_a3292d8194689829d62c_kernel.cu",
    "cake_warp_decode_a57e715fd2682ae22fe1_kernel.cu",
    "cake_warp_decode_b0e329ce4c715a895df4_kernel.cu",
    "cake_warp_decode_de0e5a68a49864417599_kernel.cu",
)
_SM100A_SOURCES: tuple[str, ...] = (
    "cake_warp_decode_045dba6ecaaa5f878a96_kernel.cu",
    "cake_warp_decode_0aeb22c5851fb0a20bfb_kernel.cu",
    "cake_warp_decode_2355e31771baf48971ec_kernel.cu",
    "cake_warp_decode_2dd7cc6dccb82323247b_kernel.cu",
    "cake_warp_decode_56ddbbdf57cdb02ccd29_kernel.cu",
    "cake_warp_decode_6aee17fe7c1c9a9fbc4f_kernel.cu",
    "cake_warp_decode_6b5216f21144ca9c3cec_kernel.cu",
    "cake_warp_decode_7f06ba61ca2fcfe503d1_kernel.cu",
    "cake_warp_decode_a4ab70bb923f293f6c03_kernel.cu",
    "cake_warp_decode_b451f69ab3b063de0072_kernel.cu",
    "cake_warp_decode_c6e324e2d43c0e98cca8_kernel.cu",
    "cake_warp_decode_ebcaf3ae004497a9132f_kernel.cu",
    "cake_warp_decode_f4eb1d54caa22ccfeb83_kernel.cu",
    "cake_warp_decode_f5eac8f9332dd5595fce_kernel.cu",
    "cake_warp_decode_fc64115f77142914c771_kernel.cu",
)
_SM103A_SOURCES: tuple[str, ...] = (
    "cake_warp_decode_16387cc0de2abce13ade_kernel.cu",
    "cake_warp_decode_31a3880ae8e28c314b6e_kernel.cu",
    "cake_warp_decode_372605d6f63db4b97d71_kernel.cu",
    "cake_warp_decode_3754ea7477cc2b195d70_kernel.cu",
    "cake_warp_decode_5304d683b9d7578df0b1_kernel.cu",
    "cake_warp_decode_55b2995635e0b6aa2b52_kernel.cu",
    "cake_warp_decode_57d9c60cbe65cd74f52d_kernel.cu",
    "cake_warp_decode_6a78714a458f2b53e363_kernel.cu",
    "cake_warp_decode_707580ee046c8424e63f_kernel.cu",
    "cake_warp_decode_ae864980eb4b1aa0e8c1_kernel.cu",
    "cake_warp_decode_af89090fc3cdb3814f96_kernel.cu",
    "cake_warp_decode_b3500b8b821383793dfd_kernel.cu",
    "cake_warp_decode_c5afe5ba09998bcee409_kernel.cu",
    "cake_warp_decode_ca90fcc9f37486b6d2af_kernel.cu",
    "cake_warp_decode_e2e6dc8e54d3325b2362_kernel.cu",
)
_TARGET_SOURCES: dict[CakeWarpDecodeTarget, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "sm100a": (("common", _COMMON_SOURCES), ("sm_100a", _SM100A_SOURCES)),
    "sm103a": (("common", _COMMON_SOURCES), ("sm_103a", _SM103A_SOURCES)),
}
# Every device TU compiles with fast math except the SiTU static FC1 kernel.
_NO_FAST_MATH_SOURCES: frozenset[str] = frozenset(
    ["cake_warp_decode_958ca99134107b938d76_kernel.cu"]
)


def _get_cake_fused_moe_warp_decode_csrc_dir() -> Path:
    """Locate Cake warp-decode sources in installed and source checkouts."""

    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "fused_moe" / "warp_decode"
    )
    if checkout.exists():
        return checkout

    installed = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "warp_decode"
    if installed.exists():
        return installed

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


def _device_sources(csrc_dir: Path, target: CakeWarpDecodeTarget) -> list[Path]:
    """Resolve the exact-target device translation units."""

    if target not in _TARGET_SOURCES:
        raise ValueError(f"unsupported Cake warp-decode target: {target}")
    sources = [
        csrc_dir / "generated" / subdir / name
        for subdir, names in _TARGET_SOURCES[target]
        for name in names
    ]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cake warp-decode {target} device sources not found: {missing}"
        )
    return sources


def get_cake_fused_moe_warp_decode_uri(
    target: CakeWarpDecodeTarget = "sm103a",
) -> str:
    """Return the exact-architecture Cake warp-decode JIT module key."""

    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake warp-decode target: {target}")
    return _MODULE_URI[target]


@functools.cache
def gen_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
) -> JitSpec:
    """Generate one exact-architecture Cake warp-decode JIT module."""

    uri = get_cake_fused_moe_warp_decode_uri(target)
    csrc_dir = _get_cake_fused_moe_warp_decode_csrc_dir()
    generated_dir = csrc_dir / "generated"
    device_sources = _device_sources(csrc_dir, target)
    for source in (
        csrc_dir / _BINDING_SOURCE,
        generated_dir / _GENERATED_MANIFEST,
        csrc_dir / _CONTRACT_HEADER,
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Cake warp-decode source not found: {source}")

    target_flags = [
        *_TARGET_FLAGS[target],
        f"-DFLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR={_TARGET_MINOR[target]}",
    ]
    spec = gen_jit_spec(
        name=uri,
        sources=[*device_sources, csrc_dir / _BINDING_SOURCE],
        extra_cuda_cflags=target_flags,
        extra_cuda_cflags_by_source={
            source: [
                *target_flags,
                *([] if source.name in _NO_FAST_MATH_SOURCES else ["--use_fast_math"]),
            ]
            for source in device_sources
        },
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


def _check_exact_target(target: CakeWarpDecodeTarget, device: Any = None) -> None:
    major, minor = _get_compute_capability(device)
    expected_minor = _TARGET_MINOR[target]
    if (major, minor) != (10, expected_minor):
        raise RuntimeError(
            f"Cake warp decode target {target} requires exact compute capability "
            f"10.{expected_minor}, "
            f"got {major}.{minor}"
        )


def load_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
    *,
    device: Any = None,
) -> Any:
    """Build or load the module after checking the requested CUDA device."""

    get_cake_fused_moe_warp_decode_uri(target)
    _check_exact_target(target, device)
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
