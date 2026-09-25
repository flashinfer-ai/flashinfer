"""JIT loader for the CAKE-generated Kimi-K3 MLA FP8 paged-attention kernels (SM100 / SM103).

The generated CUDA lives in ``csrc/cake_kimi_k3_mla/<arch>/`` (one ``*_kernel.cu`` +
``*_binding.cu`` pair per physical module).  ``MODULES`` and ``ROUTES`` are written by the
Cake generated-program exporter (``exports/kimi_k3_mla_fp8_paged_attention/export.py``):
``MODULES`` holds one record per physical module (sources, compile flags, FFI entry and
argument plan) and ``ROUTES`` maps ``"<kind>__<arch>"`` (``main_rt16`` .. ``main_rt96``,
``reduce_w1`` / ``reduce_w2`` / ``reduce_w4``, ``reduce_cta``) to its module name.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
PACKAGE_DIR = "cake_kimi_k3_mla"

# Filled by the exporter's ``integrate`` step (see module docstring).
MODULES: dict[str, dict[str, Any]] = {}
ROUTES: dict[str, dict[str, Any]] = {}


def route_key(kind: str, arch: str) -> str:
    return f"{kind}__{arch}"


def get_cake_kimi_k3_mla_route(kind: str, *, arch: str) -> dict[str, Any]:
    """Return the physical module record selected for ``kind`` on ``arch``."""
    try:
        route = ROUTES[route_key(kind, arch)]
    except KeyError as exc:
        raise ValueError(
            f"CAKE Kimi-K3 MLA has no generated route {kind!r} for {arch}"
        ) from exc
    module = MODULES[route["module"]]
    if module["arch"] != arch:
        raise ValueError(
            f"CAKE Kimi-K3 MLA route {kind!r} resolved to a {module['arch']} module on {arch}"
        )
    return dict(module, name=route["module"])


def _get_csrc_dir(arch: str) -> Path:
    if arch not in _ARCH_NVCC_FLAGS:
        raise ValueError(f"unsupported CAKE Kimi-K3 MLA architecture: {arch}")
    installed = jit_env.FLASHINFER_CSRC_DIR / PACKAGE_DIR / arch
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / PACKAGE_DIR / arch
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE Kimi-K3 MLA CUDA sources were not found. Checked:\n"
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


@functools.cache
def gen_cake_kimi_k3_mla_module(name: str) -> JitSpec:
    """JIT spec of one physical generated module (device + binding translation units)."""
    try:
        contract = MODULES[name]
    except KeyError as exc:
        raise ValueError(f"CAKE Kimi-K3 MLA has no generated module {name!r}") from exc
    arch = contract["arch"]
    csrc_dir = _get_csrc_dir(arch)
    sources = [csrc_dir / Path(src).name for src in contract["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE Kimi-K3 MLA generated sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    spec = gen_jit_spec(
        name=f"cake_kimi_k3_mla_{name}_{arch.replace('_', '')}",
        sources=sources,
        extra_cuda_cflags=[
            *_ARCH_NVCC_FLAGS[arch],
            *contract["compile_flags"],
            *contract.get("host_linkage_flags", ()),
        ],
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent.parent, _get_include_dir()],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated CAKE Kimi-K3 MLA {name} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_kimi_k3_mla_module(name: str):
    loaded = gen_cake_kimi_k3_mla_module(name).build_and_load()
    logger.info(f"Loaded CAKE Kimi-K3 MLA {name} module")
    return loaded


__all__ = [
    "MODULES",
    "ROUTES",
    "gen_cake_kimi_k3_mla_module",
    "get_cake_kimi_k3_mla_module",
    "get_cake_kimi_k3_mla_route",
    "route_key",
]
