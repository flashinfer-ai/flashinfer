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
    "cb91d5405db24f79fa60a75e56e08e592a3ec65c09e6a2bae91296de4341d5c8"
)
CAKE_FMHA_FLASHINFER_MATRIX_REVISION = "5b8da12050f80a5b5cb2bab9e87d9635a8872e5b"
CAKE_FMHA_FLASHINFER_BINDINGS_SHA256 = (
    "0ba265ee60e50548481fd1d2e2891496e84c98858ec1c132b452298dabcf8249"
)

_FLASHINFER_BINDINGS = (
    "cake_fmha_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_v1_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_v4_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_fp8_jit_binding.cu",
)

_TARGET_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def get_cake_fmha_csrc_dir() -> Path:
    """Resolve the one checked-in source root shared by base and add-ons."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        f"Cake FMHA sources were not found. Checked:\n  - {installed}\n  - {checkout}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flashinfer_bindings_sha256(csrc_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in _FLASHINFER_BINDINGS:
        binding = csrc_dir / relative_path
        if not binding.is_file():
            raise RuntimeError(
                f"Cake FMHA FlashInfer binding is missing: {relative_path}"
            )
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(binding.read_bytes())
    return digest.hexdigest()


@functools.cache
def get_cake_fmha_manifest() -> dict[str, Any]:
    """Load and fully authenticate the checked-in Cake source package."""

    csrc_dir = get_cake_fmha_csrc_dir()
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
    if (
        manifest.get("flashinfer_matrix_revision")
        != CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    ):
        raise RuntimeError(
            "Cake FMHA package has an unexpected FlashInfer matrix revision"
        )
    if manifest.get("publication", {}).get("promotion_ready") is not True:
        raise RuntimeError("Cake FMHA package is not marked promotion-ready")
    capability = manifest.get("capability", {})
    if not capability.get("complete") or capability.get("cake_coverage_ratio") != 1.0:
        raise RuntimeError(
            "Cake FMHA package does not cover its pinned FlashInfer matrix"
        )
    dcp_addon = manifest.get("add_ons", {}).get("cake_fmha_dcp_spec", {})
    if dcp_addon.get("installed") is not True or not isinstance(
        dcp_addon.get("manifest"), dict
    ):
        raise RuntimeError("Cake FMHA package is missing its authenticated DCP add-on")

    for relative_path, metadata in manifest.get("artifacts", {}).items():
        artifact = csrc_dir / relative_path
        if not artifact.is_file():
            raise RuntimeError(f"Cake FMHA artifact is missing: {relative_path}")
        if artifact.stat().st_size != metadata["bytes"]:
            raise RuntimeError(f"Cake FMHA artifact size mismatch: {relative_path}")
        if _sha256(artifact) != metadata["sha256"]:
            raise RuntimeError(f"Cake FMHA artifact digest mismatch: {relative_path}")
    actual_bindings_digest = _flashinfer_bindings_sha256(csrc_dir)
    if actual_bindings_digest != CAKE_FMHA_FLASHINFER_BINDINGS_SHA256:
        raise RuntimeError(
            "Cake FMHA FlashInfer binding digest mismatch: "
            f"expected {CAKE_FMHA_FLASHINFER_BINDINGS_SHA256}, "
            f"got {actual_bindings_digest}"
        )
    return manifest


def get_cake_fmha_compat_uri(target: CakeFmhaTarget) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    return (
        f"cake_fmha_compat_v1_{target}_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_compat_module(target: CakeFmhaTarget) -> JitSpec:
    """Build the complete-domain route from the authenticated source package."""

    manifest = get_cake_fmha_manifest()
    csrc_dir = get_cake_fmha_csrc_dir()
    component = manifest["components"]["compat_v1"]
    arch = {"sm100a": "sm_100a", "sm103a": "sm_103a"}[target]
    source_family = component["source_family"]
    if len(source_family) != 1 or source_family[0]["selector"] != {}:
        raise RuntimeError(
            "Cake FMHA compatibility component has an invalid source family"
        )
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
    "CAKE_FMHA_FLASHINFER_BINDINGS_SHA256",
    "CAKE_FMHA_FLASHINFER_MATRIX_REVISION",
    "CAKE_FMHA_MANIFEST_SHA256",
    "CakeFmhaTarget",
    "gen_cake_fmha_compat_module",
    "get_cake_fmha_compat_uri",
    "get_cake_fmha_csrc_dir",
    "get_cake_fmha_manifest",
    "load_cake_fmha_compat_module",
]
