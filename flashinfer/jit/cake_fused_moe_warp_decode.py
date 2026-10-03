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
    "cake_warp_decode_024bed5eb8061821a022_kernel.cu",
    "cake_warp_decode_2943d5a4f443be1b5408_kernel.cu",
    "cake_warp_decode_670effeafbee07071ec6_kernel.cu",
    "cake_warp_decode_6b2146aa6e4e8f2e2161_kernel.cu",
    "cake_warp_decode_857d1ba629d5d1813f19_kernel.cu",
    "cake_warp_decode_87d1b6a9bbc9ece09ad6_kernel.cu",
    "cake_warp_decode_8851303d75b4e1cec033_kernel.cu",
    "cake_warp_decode_8d34c1c2891be9b66f15_kernel.cu",
    "cake_warp_decode_9021eadecd3078bf13c5_kernel.cu",
    "cake_warp_decode_ac22f4a6ae1ebaae1276_kernel.cu",
    "cake_warp_decode_b5ee48ba618c8145d075_kernel.cu",
    "cake_warp_decode_b8935d27ab092becf9a0_kernel.cu",
    "cake_warp_decode_b9ce2c4ba5690e69450c_kernel.cu",
    "cake_warp_decode_bbea84bc0aa6f631c01e_kernel.cu",
    "cake_warp_decode_de1fffa0c9722d6f3dc2_kernel.cu",
    "cake_warp_decode_e7c996a7418120fdc59d_kernel.cu",
    "cake_warp_decode_fc102671dcafa54593ec_kernel.cu",
)
_SM100A_SOURCES: tuple[str, ...] = (
    "cake_warp_decode_1919fdc835c6d5747044_kernel.cu",
    "cake_warp_decode_36c3fc6de7aff6664eb4_kernel.cu",
    "cake_warp_decode_571467f2fe1a078edd15_kernel.cu",
    "cake_warp_decode_8aa1d75a331e184994b1_kernel.cu",
    "cake_warp_decode_8aec1074daa9fa51c03c_kernel.cu",
    "cake_warp_decode_913a821ce8dee11dafcf_kernel.cu",
    "cake_warp_decode_9bba0f8393c3f5c41338_kernel.cu",
    "cake_warp_decode_aacea66676dc5e3ed74d_kernel.cu",
    "cake_warp_decode_ab11eefabf140deeaf0c_kernel.cu",
    "cake_warp_decode_b1f32bc0ea0d0dbbf453_kernel.cu",
    "cake_warp_decode_c2c3b32fdd0cd7ae0c4c_kernel.cu",
    "cake_warp_decode_d17899c336800a8599a6_kernel.cu",
    "cake_warp_decode_df32a9c78cd8ea22ac78_kernel.cu",
    "cake_warp_decode_e465613750770e29988f_kernel.cu",
    "cake_warp_decode_fc0aed4e58408740ce2a_kernel.cu",
)
_SM103A_SOURCES: tuple[str, ...] = (
    "cake_warp_decode_0d07af7cfe5697b5ecdc_kernel.cu",
    "cake_warp_decode_269d5aebbb5aa995796a_kernel.cu",
    "cake_warp_decode_36ef13a3551d497679cd_kernel.cu",
    "cake_warp_decode_3b1c1adc59f3837a48a4_kernel.cu",
    "cake_warp_decode_3f5bc27d007af5687d63_kernel.cu",
    "cake_warp_decode_49b2dacb8c21fd7ca1c6_kernel.cu",
    "cake_warp_decode_64b75a49bc729f82a995_kernel.cu",
    "cake_warp_decode_65d8dc9a2b51bca5f578_kernel.cu",
    "cake_warp_decode_7173b39130de7a59f9c6_kernel.cu",
    "cake_warp_decode_7e715939a26489a27fcb_kernel.cu",
    "cake_warp_decode_7fc08d4a160ade893bda_kernel.cu",
    "cake_warp_decode_8bce1085cbf8aaa7c7f6_kernel.cu",
    "cake_warp_decode_9805c54bf6db2ee12595_kernel.cu",
    "cake_warp_decode_b0f548cc0bc03def0160_kernel.cu",
    "cake_warp_decode_b47db4977f3026b27967_kernel.cu",
    "cake_warp_decode_b84333fbc5c6282202d1_kernel.cu",
    "cake_warp_decode_e2796e299356440aa3e4_kernel.cu",
)
_TARGET_SOURCES: dict[CakeWarpDecodeTarget, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "sm100a": (("common", _COMMON_SOURCES), ("sm_100a", _SM100A_SOURCES)),
    "sm103a": (("common", _COMMON_SOURCES), ("sm_103a", _SM103A_SOURCES)),
}
# Every device TU compiles with fast math except the SiTU static FC1 kernel.
_NO_FAST_MATH_SOURCES: frozenset[str] = frozenset(
    ["cake_warp_decode_bbea84bc0aa6f631c01e_kernel.cu"]
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
