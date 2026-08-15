"""JIT loader for the versioned standalone Cake FMHA product."""

from __future__ import annotations

import functools
import hashlib
import json
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

CakeFmhaTarget = Literal["sm100a", "sm103a"]

CAKE_FMHA_MANIFEST_SHA256 = (
    "3b6d775c7f375bc31bf572b353fcea8e31f017cefc5f9d4fc4c75d60fcd36acf"
)
CAKE_FMHA_FLASHINFER_MATRIX_REVISION = (
    "5b8da12050f80a5b5cb2bab9e87d9635a8872e5b"
)

_TARGET_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "Cake FMHA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.cache
def get_cake_fmha_manifest() -> dict[str, Any]:
    """Load and fully authenticate the checked-in Cake source package."""

    csrc_dir = _get_csrc_dir()
    manifest_path = csrc_dir / "manifest.json"
    digest_path = csrc_dir / "manifest.sha256"
    actual_digest = _sha256(manifest_path)
    recorded_digest = digest_path.read_text().split()[0]
    if actual_digest != CAKE_FMHA_MANIFEST_SHA256 or recorded_digest != actual_digest:
        raise RuntimeError(
            "Cake FMHA manifest digest mismatch: "
            f"expected {CAKE_FMHA_MANIFEST_SHA256}, got {actual_digest}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("product") != "cake_fmha":
        raise RuntimeError("Cake FMHA package has an invalid product identifier")
    if manifest.get("flashinfer_matrix_revision") != CAKE_FMHA_FLASHINFER_MATRIX_REVISION:
        raise RuntimeError("Cake FMHA package has an unexpected FlashInfer matrix revision")
    if manifest.get("publication", {}).get("promotion_ready") is not True:
        raise RuntimeError("Cake FMHA package is not marked promotion-ready")
    capability = manifest.get("capability", {})
    if not capability.get("complete") or capability.get("cake_coverage_ratio") != 1.0:
        raise RuntimeError("Cake FMHA package does not cover its pinned FlashInfer matrix")

    for relative_path, metadata in manifest.get("artifacts", {}).items():
        artifact = csrc_dir / relative_path
        if not artifact.is_file():
            raise RuntimeError(f"Cake FMHA artifact is missing: {relative_path}")
        if artifact.stat().st_size != metadata["bytes"]:
            raise RuntimeError(f"Cake FMHA artifact size mismatch: {relative_path}")
        if _sha256(artifact) != metadata["sha256"]:
            raise RuntimeError(f"Cake FMHA artifact digest mismatch: {relative_path}")
    return manifest


def get_cake_fmha_compat_uri(target: CakeFmhaTarget) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    return f"cake_fmha_compat_v1_{target}_{CAKE_FMHA_MANIFEST_SHA256[:16]}"


@functools.cache
def gen_cake_fmha_compat_module(target: CakeFmhaTarget) -> JitSpec:
    """Build the complete-domain route from the authenticated source package."""

    manifest = get_cake_fmha_manifest()
    csrc_dir = _get_csrc_dir()
    component = manifest["components"]["compat_v1"]
    arch = {"sm100a": "sm_100a", "sm103a": "sm_103a"}[target]
    source_family = component["source_family"]
    if len(source_family) != 1 or source_family[0]["selector"] != {}:
        raise RuntimeError("Cake FMHA compatibility component has an invalid source family")
    sources = [
        csrc_dir / source_family[0]["sources"][arch],
        csrc_dir / component["binding_source"],
        csrc_dir / "cake_fmha_jit_binding.cu",
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA JIT source not found: {source}")

    spec = gen_jit_spec(
        name=get_cake_fmha_compat_uri(target),
        sources=sources,
        extra_cuda_cflags=[*_TARGET_FLAGS[target], "-use_fast_math"],
        extra_include_paths=[csrc_dir, jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_compat_module(target: CakeFmhaTarget):
    module = gen_cake_fmha_compat_module(target).build_and_load()
    logger.info("Loaded Cake FMHA module: %s", module)
    return module


__all__ = [
    "CAKE_FMHA_FLASHINFER_MATRIX_REVISION",
    "CAKE_FMHA_MANIFEST_SHA256",
    "CakeFmhaTarget",
    "gen_cake_fmha_compat_module",
    "get_cake_fmha_compat_uri",
    "get_cake_fmha_manifest",
    "load_cake_fmha_compat_module",
]
