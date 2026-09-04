"""JIT loader for the versioned standalone Cake FMHA product."""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
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
CakeFmhaContextExactProfile = Literal["q511", "q257"]

CAKE_FMHA_MANIFEST_SHA256 = (
    "54755a8b6e4a47f762b3e7a6cd9bb695055ef560f787f680a8f450b48386c6a9"
)
CAKE_FMHA_FLASHINFER_MATRIX_REVISION = "5b8da12050f80a5b5cb2bab9e87d9635a8872e5b"
CAKE_FMHA_FLASHINFER_BINDINGS_SHA256 = (
    "7a7f54f5d62c4e55e9666f0e3dd22f841de8db4f5eb002d148c0c1ef35a93052"
)

_FLASHINFER_BINDINGS = (
    "cake_fmha_jit_binding.cu",
    "jit/cake_fmha_context_bf16_jit_binding.cu",
    "jit/cake_fmha_context_fp8_jit_binding.cu",
    "jit/cake_fmha_context_hd256_jit_binding.cu",
    "jit/cake_fmha_decode_native_bf16_jit_binding.cu",
    "jit/cake_fmha_decode_native_fp16_hd512_jit_binding.cu",
    "jit/cake_fmha_decode_native_fp16_nhd_jit_binding.cu",
    "jit/cake_fmha_decode_quant_bf16q_jit_binding.cu",
    "jit/cake_fmha_decode_quant_fp8_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_v1_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_v4_jit_binding.cu",
    "jit/cake_fmha_dcp_spec_bf16_fp8_jit_binding.cu",
)

_TARGET_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_MANIFEST_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_DECODE_NATIVE_BF16_JIT_BINDING = "jit/cake_fmha_decode_native_bf16_jit_binding.cu"
_DECODE_NATIVE_FP16_HD512_JIT_BINDING = (
    "jit/cake_fmha_decode_native_fp16_hd512_jit_binding.cu"
)
_DECODE_NATIVE_FP16_NHD_JIT_BINDING = (
    "jit/cake_fmha_decode_native_fp16_nhd_jit_binding.cu"
)
_DECODE_QUANT_BF16Q_JIT_BINDING = "jit/cake_fmha_decode_quant_bf16q_jit_binding.cu"
_DECODE_QUANT_FP8_JIT_BINDING = "jit/cake_fmha_decode_quant_fp8_jit_binding.cu"
_CONTEXT_BF16_JIT_BINDING = "jit/cake_fmha_context_bf16_jit_binding.cu"
_CONTEXT_FP8_JIT_BINDING = "jit/cake_fmha_context_fp8_jit_binding.cu"
_CONTEXT_HD256_JIT_BINDING = "jit/cake_fmha_context_hd256_jit_binding.cu"


def get_cake_fmha_csrc_dir() -> Path:
    """Resolve the one checked-in source root shared by base and add-ons."""

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha"
    if checkout.exists():
        return checkout

    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha"
    if installed.exists():
        return installed

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


def _validate_decode_native_specialization(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool | None,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> dict[str, int]:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if batch_size <= 0 or q_len <= 0:
        raise ValueError("batch_size and q_len must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        raise ValueError("decode-native requires a head-group ratio in [1, 8]")
    selector = {
        "HAS_WINDOW": int(has_window),
        "RETAIN_KV_L2": int(retain_kv_l2),
        "USE_SCALE_PTR": int(use_scale_ptr),
    }
    if has_sink is not None:
        selector["HAS_SINK"] = int(has_sink)
    return selector


def _validate_decode_quant_bf16q_specialization(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> dict[str, int]:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if q_len != 1:
        raise ValueError("decode-quant BF16Q requires q_len=1")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        raise ValueError("decode-quant BF16Q requires a head-group ratio in [1, 8]")
    if page_size not in (16, 32, 64):
        raise ValueError("decode-quant BF16Q requires page_size 16, 32, or 64")
    return {"PAGE_SIZE": page_size}


def _validate_decode_quant_fp8_specialization(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    *,
    full_blocks: bool,
) -> dict[str, int]:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if q_len != 1:
        raise ValueError("decode-quant FP8 requires q_len=1")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads != 8 * num_kv_heads:
        raise ValueError("decode-quant FP8 requires a head-group ratio of 8")
    if page_size not in (16, 32, 64):
        raise ValueError("decode-quant FP8 requires page_size 16, 32, or 64")
    return {"FULL_BLOCKS": int(full_blocks), "PAGE_SIZE": page_size}


def _validate_decode_quant_nvfp4_specialization(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> dict[str, int]:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if q_len != 1:
        raise ValueError("decode-quant NVFP4 requires q_len=1")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        raise ValueError("decode-quant NVFP4 requires a head-group ratio in [1, 8]")
    if page_size not in (16, 32):
        raise ValueError("decode-quant NVFP4 requires page_size 16 or 32")
    return {"PAGE_SIZE": page_size}


def _get_component_member(
    component_name: str,
    selector: Mapping[str, int],
    *,
    required: bool,
) -> dict[str, Any] | None:
    component = get_cake_fmha_manifest()["components"][component_name]
    normalized_selector = dict(sorted(selector.items()))
    matches = [
        member
        for member in component["source_family"]
        if member.get("selector") == normalized_selector
    ]
    if len(matches) > 1 or (required and len(matches) != 1):
        raise RuntimeError(
            "Cake FMHA component selector is not unique: "
            f"{component_name} {normalized_selector!r}"
        )
    return matches[0] if matches else None


@functools.cache
def _resolve_decode_native_bf16_selector(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> dict[str, int] | None:
    semantic_selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    exact_selector = {
        **semantic_selector,
        "BATCH_SIZE": batch_size,
        "Q_LEN": q_len,
        "NUM_Q_HEADS": num_q_heads,
        "NUM_KV_HEADS": num_kv_heads,
    }
    if (
        _get_component_member("decode_native_bf16", exact_selector, required=False)
        is not None
    ):
        return dict(sorted(exact_selector.items()))
    if (
        _get_component_member("decode_native_bf16", semantic_selector, required=False)
        is not None
    ):
        return dict(sorted(semantic_selector.items()))
    return None


def _is_cake_fmha_decode_native_bf16_available(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> bool:
    """Return whether the authenticated manifest contains this BF16 route."""

    return (
        _resolve_decode_native_bf16_selector(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            has_sink=has_sink,
            has_window=has_window,
            use_scale_ptr=use_scale_ptr,
            retain_kv_l2=retain_kv_l2,
        )
        is not None
    )


def _get_component_launch_sources(
    component_name: str,
    target: CakeFmhaTarget,
    selector: Mapping[str, int],
) -> tuple[Path, Path]:
    component = get_cake_fmha_manifest()["components"][component_name]
    member = _get_component_member(component_name, selector, required=True)
    assert member is not None
    csrc_dir = get_cake_fmha_csrc_dir()
    body = csrc_dir / member["sources"][_TARGET_MANIFEST_ARCH[target]]
    launch_override = member.get("launch_override") or {}
    launch_binding = csrc_dir / launch_override.get(
        "binding_source", component["binding_source"]
    )
    for source in (body, launch_binding):
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA JIT source not found: {source}")
    return body, launch_binding


def _get_component_sources(
    component_name: str,
    target: CakeFmhaTarget,
    selector: Mapping[str, int],
    jit_binding: str,
) -> tuple[Path, Path, Path]:
    body, launch_binding = _get_component_launch_sources(
        component_name, target, selector
    )
    api_binding = get_cake_fmha_csrc_dir() / jit_binding
    if not api_binding.is_file():
        raise FileNotFoundError(f"Cake FMHA JIT source not found: {api_binding}")
    return body, launch_binding, api_binding


def _validate_context_specialization(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    exact_profile: CakeFmhaContextExactProfile | None = None,
) -> dict[str, int]:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if num_m_blocks <= 0:
        raise ValueError("num_m_blocks must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if pack_g <= 0 or num_q_heads % pack_g:
        raise ValueError("pack_g must be positive and divide num_q_heads")
    if pack_g not in (1, num_q_heads // num_kv_heads):
        raise ValueError("pack_g must be 1 or the complete GQA group")
    if page_size not in (16, 32, 64, 128, 256, 512, 1024):
        raise ValueError("Cake context requires a supported page size")
    if l2_swizzle not in (1, 8):
        raise ValueError("l2_swizzle must be 1 or 8")
    if enable_sink and return_lse:
        raise ValueError("the pinned context contract excludes sink plus LSE")
    selector = {
        "ENABLE_SINK": int(enable_sink),
        "IS_CAUSAL": int(is_causal),
        "RETURN_LSE": int(return_lse),
    }
    if exact_profile is None:
        return selector
    if exact_profile == "q511":
        expected = (11, 10, 2, 5, 32, 1)
        exact_selector = {
            "HEADS_PER_GROUP": 5,
            "L2_SWIZZLE": 1,
            "NUM_M_BLOCKS": 11,
            "NUM_Q_HEADS": 10,
            "PACK_G": 5,
            "PAGE_SIZE": 32,
            "SINGLE_MASK_LOOP": 1,
            "TOK_PER_STAGE": 25,
        }
    else:
        expected = (6, 10, 2, 5, 1024, 8)
        exact_selector = {
            "HEADS_PER_GROUP": 5,
            "L2_SWIZZLE": 8,
            "NUM_M_BLOCKS": 6,
            "NUM_Q_HEADS": 10,
            "PACK_G": 5,
            "PAGE_SIZE": 1024,
            "SINGLE_MASK_LOOP": 1,
            "TOK_PER_STAGE": 25,
        }
    actual = (
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
    )
    if actual != expected or selector != {
        "ENABLE_SINK": 0,
        "IS_CAUSAL": 1,
        "RETURN_LSE": 0,
    }:
        raise ValueError(
            f"context BF16 exact profile {exact_profile} does not match its fixed selector"
        )
    return {**selector, **exact_selector}


def get_cake_fmha_context_bf16_uri(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    exact_profile: CakeFmhaContextExactProfile | None = None,
) -> str:
    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
        exact_profile=exact_profile,
    )
    return (
        f"cake_fmha_context_bf16_{target}"
        f"_m{num_m_blocks}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_pack{pack_g}_page{page_size}_l2{l2_swizzle}"
        f"_causal{selector['IS_CAUSAL']}_lse{selector['RETURN_LSE']}"
        f"_sink{selector['ENABLE_SINK']}"
        f"_exact{exact_profile or 'generic'}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_context_bf16_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    exact_profile: CakeFmhaContextExactProfile | None = None,
) -> JitSpec:
    """Build one authenticated context BF16 specialization."""

    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
        exact_profile=exact_profile,
    )
    sources = _get_component_sources(
        "context_bf16", target, selector, _CONTEXT_BF16_JIT_BINDING
    )
    heads_per_group = num_q_heads // num_kv_heads
    tok_per_stage = 128 // pack_g
    spec = gen_jit_spec(
        name=get_cake_fmha_context_bf16_uri(
            target,
            num_m_blocks,
            num_q_heads,
            num_kv_heads,
            pack_g,
            page_size,
            l2_swizzle,
            is_causal=is_causal,
            return_lse=return_lse,
            enable_sink=enable_sink,
            exact_profile=exact_profile,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DNUM_M_BLOCKS={num_m_blocks}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DHEADS_PER_GROUP={heads_per_group}",
            f"-DPACK_G={pack_g}",
            f"-DTOK_PER_STAGE={tok_per_stage}",
            f"-DL2_SWIZZLE={l2_swizzle}",
            f"-DPAGE_SIZE={page_size}",
            f"-DCAKE_FMHA_CONTEXT_IS_CAUSAL={selector['IS_CAUSAL']}",
            f"-DCAKE_FMHA_CONTEXT_RETURN_LSE={selector['RETURN_LSE']}",
            f"-DCAKE_FMHA_CONTEXT_ENABLE_SINK={selector['ENABLE_SINK']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA context BF16 JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_context_bf16_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    exact_profile: CakeFmhaContextExactProfile | None = None,
):
    module = gen_cake_fmha_context_bf16_module(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
        exact_profile=exact_profile,
    ).build_and_load()
    logger.info("Loaded Cake FMHA context BF16 module: %s", module)
    return module


def get_cake_fmha_context_fp8_uri(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
) -> str:
    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
    )
    return (
        f"cake_fmha_context_fp8_{target}"
        f"_m{num_m_blocks}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_pack{pack_g}_page{page_size}_l2{l2_swizzle}"
        f"_causal{selector['IS_CAUSAL']}_lse{selector['RETURN_LSE']}"
        f"_sink{selector['ENABLE_SINK']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_context_fp8_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
) -> JitSpec:
    """Build one authenticated context FP8 specialization."""

    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
    )
    sources = _get_component_sources(
        "context_fp8", target, selector, _CONTEXT_FP8_JIT_BINDING
    )
    heads_per_group = num_q_heads // num_kv_heads
    tok_per_stage = 128 // pack_g
    spec = gen_jit_spec(
        name=get_cake_fmha_context_fp8_uri(
            target,
            num_m_blocks,
            num_q_heads,
            num_kv_heads,
            pack_g,
            page_size,
            l2_swizzle,
            is_causal=is_causal,
            return_lse=return_lse,
            enable_sink=enable_sink,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DNUM_M_BLOCKS={num_m_blocks}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DHEADS_PER_GROUP={heads_per_group}",
            f"-DPACK_G={pack_g}",
            f"-DTOK_PER_STAGE={tok_per_stage}",
            f"-DL2_SWIZZLE={l2_swizzle}",
            f"-DPAGE_SIZE={page_size}",
            f"-DCAKE_FMHA_CONTEXT_IS_CAUSAL={selector['IS_CAUSAL']}",
            f"-DCAKE_FMHA_CONTEXT_RETURN_LSE={selector['RETURN_LSE']}",
            f"-DCAKE_FMHA_CONTEXT_ENABLE_SINK={selector['ENABLE_SINK']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA context FP8 JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_context_fp8_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
    *,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
):
    module = gen_cake_fmha_context_fp8_module(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=is_causal,
        return_lse=return_lse,
        enable_sink=enable_sink,
    ).build_and_load()
    logger.info("Loaded Cake FMHA context FP8 module: %s", module)
    return module


def get_cake_fmha_context_nvfp4_uri(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
) -> str:
    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    if page_size != 16:
        raise ValueError("Cake NVFP4 context requires page_size=16")
    return (
        f"cake_fmha_context_nvfp4_{target}"
        f"_m{num_m_blocks}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_pack{pack_g}_page{page_size}_l2{l2_swizzle}"
        f"_causal{selector['IS_CAUSAL']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_context_nvfp4_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
) -> JitSpec:
    """Build the authenticated fused NVFP4 context kernel."""

    selector = _validate_context_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    if page_size != 16:
        raise ValueError("Cake NVFP4 context requires page_size=16")
    sources = _get_component_sources(
        "context_nvfp4",
        target,
        {**selector, "STATIC_ONE_TILE": 1},
        _CONTEXT_FP8_JIT_BINDING,
    )
    heads_per_group = num_q_heads // num_kv_heads
    tok_per_stage = 128 // pack_g
    spec = gen_jit_spec(
        name=get_cake_fmha_context_nvfp4_uri(
            target,
            num_m_blocks,
            num_q_heads,
            num_kv_heads,
            pack_g,
            page_size,
            l2_swizzle,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DNUM_M_BLOCKS={num_m_blocks}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DHEADS_PER_GROUP={heads_per_group}",
            f"-DPACK_G={pack_g}",
            f"-DTOK_PER_STAGE={tok_per_stage}",
            f"-DL2_SWIZZLE={l2_swizzle}",
            f"-DPAGE_SIZE={page_size}",
            "-DCAKE_FMHA_CONTEXT_IS_CAUSAL=1",
            "-DCAKE_FMHA_CONTEXT_RETURN_LSE=0",
            "-DCAKE_FMHA_CONTEXT_ENABLE_SINK=0",
            "-DCAKE_FMHA_CONTEXT_NVFP4=1",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA context NVFP4 JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_context_nvfp4_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
    page_size: int,
    l2_swizzle: int,
):
    module = gen_cake_fmha_context_nvfp4_module(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        pack_g,
        page_size,
        l2_swizzle,
    ).build_and_load()
    logger.info("Loaded Cake FMHA context NVFP4 module: %s", module)
    return module


def _validate_context_hd256_specialization(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> int:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    if num_m_blocks <= 0:
        raise ValueError("num_m_blocks must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    heads_per_group = num_q_heads // num_kv_heads
    if heads_per_group <= 0:
        raise ValueError("heads_per_group must be positive")
    if page_size not in (16, 32, 64, 128, 256, 512, 1024):
        raise ValueError("HD256 context requires a supported paged-KV page size")
    return heads_per_group


def _get_cake_fmha_context_hd256_uri(
    kind: Literal["fp16", "fp8"],
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> str:
    heads_per_group = _validate_context_hd256_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    return (
        f"cake_fmha_context_{kind}_hd256_{target}"
        f"_m{num_m_blocks}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_g{heads_per_group}_page{page_size}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


def get_cake_fmha_context_fp16_hd256_uri(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> str:
    return _get_cake_fmha_context_hd256_uri(
        "fp16",
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )


def get_cake_fmha_context_fp8_hd256_uri(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> str:
    return _get_cake_fmha_context_hd256_uri(
        "fp8",
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )


def _gen_cake_fmha_context_hd256_module(
    kind: Literal["fp16", "fp8"],
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> JitSpec:
    heads_per_group = _validate_context_hd256_specialization(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    if kind == "fp16":
        component = "context_fp16_hd256"
        selector = {"IS_CAUSAL": 0}
        is_fp8 = 0
    else:
        component = "context_fp8_hd256"
        selector = {"IS_CAUSAL": 1, "OUTPUT_BF16": 1}
        is_fp8 = 1
    main_sources = _get_component_launch_sources(component, target, selector)
    csrc_dir = get_cake_fmha_csrc_dir()
    support_source = csrc_dir / "cuda/context_hd256_support/cake_fmha_hd256_support.cu"
    api_binding = csrc_dir / _CONTEXT_HD256_JIT_BINDING
    for source in (support_source, api_binding):
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA JIT source not found: {source}")
    spec = gen_jit_spec(
        name=_get_cake_fmha_context_hd256_uri(
            kind,
            target,
            num_m_blocks,
            num_q_heads,
            num_kv_heads,
            page_size,
        ),
        sources=[*main_sources, support_source, api_binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DNUM_M_BLOCKS={num_m_blocks}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DHEADS_PER_GROUP={heads_per_group}",
            f"-DCAKE_FMHA_HD256_FP8={is_fp8}",
            f"-DCAKE_FMHA_SOURCE_PAGE_SIZE={page_size}",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir / "include",
            jit_env.FLASHINFER_CSRC_DIR,
        ],
    )
    logger.info("Generated Cake FMHA context %s HD256 JIT spec: %s", kind, spec.name)
    return spec


@functools.cache
def gen_cake_fmha_context_fp16_hd256_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> JitSpec:
    return _gen_cake_fmha_context_hd256_module(
        "fp16",
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )


@functools.cache
def gen_cake_fmha_context_fp8_hd256_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> JitSpec:
    return _gen_cake_fmha_context_hd256_module(
        "fp8",
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    )


@functools.cache
def load_cake_fmha_context_fp16_hd256_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
):
    module = gen_cake_fmha_context_fp16_hd256_module(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    ).build_and_load()
    logger.info("Loaded Cake FMHA context FP16 HD256 module: %s", module)
    return module


@functools.cache
def load_cake_fmha_context_fp8_hd256_module(
    target: CakeFmhaTarget,
    num_m_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
):
    module = gen_cake_fmha_context_fp8_hd256_module(
        target,
        num_m_blocks,
        num_q_heads,
        num_kv_heads,
        page_size,
    ).build_and_load()
    logger.info("Loaded Cake FMHA context FP8 HD256 module: %s", module)
    return module


def get_cake_fmha_decode_native_bf16_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> str:
    selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    return (
        f"cake_fmha_decode_native_bf16_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_sink{selector['HAS_SINK']}_window{selector['HAS_WINDOW']}"
        f"_scale{selector['USE_SCALE_PTR']}_retain{selector['RETAIN_KV_L2']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_native_bf16_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> JitSpec:
    """Build one authenticated decode-native BF16 specialization."""

    selector = _resolve_decode_native_bf16_selector(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    if selector is None:
        raise RuntimeError(
            "Cake FMHA decode-native BF16 specialization is absent from the manifest"
        )
    sources = _get_component_sources(
        "decode_native_bf16",
        target,
        selector,
        _DECODE_NATIVE_BF16_JIT_BINDING,
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_native_bf16_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            has_sink=has_sink,
            has_window=has_window,
            use_scale_ptr=use_scale_ptr,
            retain_kv_l2=retain_kv_l2,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_HAS_SINK={selector['HAS_SINK']}",
            f"-DCAKE_FMHA_HAS_WINDOW={selector['HAS_WINDOW']}",
            f"-DCAKE_FMHA_USE_SCALE_PTR={selector['USE_SCALE_PTR']}",
            f"-DCAKE_FMHA_RETAIN_KV_L2={selector['RETAIN_KV_L2']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA decode-native BF16 JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_native_bf16_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
):
    module = gen_cake_fmha_decode_native_bf16_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    ).build_and_load()
    logger.info("Loaded Cake FMHA decode-native BF16 module: %s", module)
    return module


def get_cake_fmha_decode_native_fp16_nhd_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> str:
    selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    return (
        f"cake_fmha_decode_native_fp16_nhd_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_sink{selector['HAS_SINK']}_window{selector['HAS_WINDOW']}"
        f"_scale{selector['USE_SCALE_PTR']}_retain{selector['RETAIN_KV_L2']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_native_fp16_nhd_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> JitSpec:
    """Build one authenticated decode-native FP16 NHD specialization."""

    selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    sources = _get_component_sources(
        "decode_native_fp16_nhd",
        target,
        selector,
        _DECODE_NATIVE_FP16_NHD_JIT_BINDING,
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_native_fp16_nhd_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            has_sink=has_sink,
            has_window=has_window,
            use_scale_ptr=use_scale_ptr,
            retain_kv_l2=retain_kv_l2,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_HAS_SINK={selector['HAS_SINK']}",
            f"-DCAKE_FMHA_HAS_WINDOW={selector['HAS_WINDOW']}",
            f"-DCAKE_FMHA_USE_SCALE_PTR={selector['USE_SCALE_PTR']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA decode-native FP16 NHD JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_native_fp16_nhd_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_sink: bool,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
):
    module = gen_cake_fmha_decode_native_fp16_nhd_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=has_sink,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    ).build_and_load()
    logger.info("Loaded Cake FMHA decode-native FP16 NHD module: %s", module)
    return module


def get_cake_fmha_decode_native_fp16_hd512_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> str:
    selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=None,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    return (
        f"cake_fmha_decode_native_fp16_hd512_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_window{selector['HAS_WINDOW']}_scale{selector['USE_SCALE_PTR']}"
        f"_retain{selector['RETAIN_KV_L2']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_native_fp16_hd512_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
) -> JitSpec:
    """Build one authenticated decode-native FP16 head-dim-512 specialization."""

    selector = _validate_decode_native_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_sink=None,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    )
    sources = _get_component_sources(
        "decode_native_fp16_hd512",
        target,
        selector,
        _DECODE_NATIVE_FP16_HD512_JIT_BINDING,
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_native_fp16_hd512_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            has_window=has_window,
            use_scale_ptr=use_scale_ptr,
            retain_kv_l2=retain_kv_l2,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_HAS_WINDOW={selector['HAS_WINDOW']}",
            f"-DCAKE_FMHA_USE_SCALE_PTR={selector['USE_SCALE_PTR']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info(
        "Generated Cake FMHA decode-native FP16 head-dim-512 JIT spec: %s",
        spec.name,
    )
    return spec


@functools.cache
def load_cake_fmha_decode_native_fp16_hd512_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    has_window: bool,
    use_scale_ptr: bool,
    retain_kv_l2: bool,
):
    module = gen_cake_fmha_decode_native_fp16_hd512_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        has_window=has_window,
        use_scale_ptr=use_scale_ptr,
        retain_kv_l2=retain_kv_l2,
    ).build_and_load()
    logger.info("Loaded Cake FMHA decode-native FP16 head-dim-512 module: %s", module)
    return module


def get_cake_fmha_decode_quant_bf16q_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> str:
    selector = _validate_decode_quant_bf16q_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    return (
        f"cake_fmha_decode_quant_bf16q_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_page{selector['PAGE_SIZE']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_quant_bf16q_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> JitSpec:
    """Build one authenticated BF16-query/FP8-KV decode specialization."""

    selector = _validate_decode_quant_bf16q_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    sources = _get_component_sources(
        "decode_quant_bf16q",
        target,
        selector,
        _DECODE_QUANT_BF16Q_JIT_BINDING,
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_quant_bf16q_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            page_size,
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_PAGE_SIZE={selector['PAGE_SIZE']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA BF16Q decode JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_quant_bf16q_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
):
    module = gen_cake_fmha_decode_quant_bf16q_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    ).build_and_load()
    logger.info("Loaded Cake FMHA BF16Q decode module: %s", module)
    return module


def get_cake_fmha_decode_quant_fp8_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    *,
    full_blocks: bool,
) -> str:
    selector = _validate_decode_quant_fp8_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
        full_blocks=full_blocks,
    )
    return (
        f"cake_fmha_decode_quant_fp8_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_page{selector['PAGE_SIZE']}_full{selector['FULL_BLOCKS']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_quant_fp8_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    *,
    full_blocks: bool,
) -> JitSpec:
    """Build the authenticated FP8 decode plus split-KV reducer chain."""

    selector = _validate_decode_quant_fp8_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
        full_blocks=full_blocks,
    )
    main_sources = _get_component_launch_sources("decode_quant_fp8", target, selector)
    reduce_sources = _get_component_launch_sources(
        "decode_quant_fp8_reduce", target, {}
    )
    api_binding = get_cake_fmha_csrc_dir() / _DECODE_QUANT_FP8_JIT_BINDING
    if not api_binding.is_file():
        raise FileNotFoundError(f"Cake FMHA JIT source not found: {api_binding}")
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_quant_fp8_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            page_size,
            full_blocks=full_blocks,
        ),
        sources=[*main_sources, *reduce_sources, api_binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_PAGE_SIZE={selector['PAGE_SIZE']}",
            f"-DCAKE_FMHA_FULL_BLOCKS={selector['FULL_BLOCKS']}",
            "-DCAKE_FMHA_NVFP4=0",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA FP8 decode/reduce JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_quant_fp8_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    *,
    full_blocks: bool,
):
    module = gen_cake_fmha_decode_quant_fp8_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
        full_blocks=full_blocks,
    ).build_and_load()
    logger.info("Loaded Cake FMHA FP8 decode/reduce module: %s", module)
    return module


def get_cake_fmha_decode_quant_nvfp4_uri(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> str:
    selector = _validate_decode_quant_nvfp4_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    return (
        f"cake_fmha_decode_quant_nvfp4_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_page{selector['PAGE_SIZE']}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_cake_fmha_decode_quant_nvfp4_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> JitSpec:
    """Build the authenticated portable NVFP4 decode/reducer chain."""

    selector = _validate_decode_quant_nvfp4_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    )
    main_sources = _get_component_launch_sources("decode_quant_nvfp4", target, selector)
    reduce_sources = _get_component_launch_sources(
        "decode_quant_fp8_reduce", target, {}
    )
    api_binding = get_cake_fmha_csrc_dir() / _DECODE_QUANT_FP8_JIT_BINDING
    if not api_binding.is_file():
        raise FileNotFoundError(f"Cake FMHA JIT source not found: {api_binding}")
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_quant_nvfp4_uri(
            target,
            batch_size,
            q_len,
            num_q_heads,
            num_kv_heads,
            page_size,
        ),
        sources=[*main_sources, *reduce_sources, api_binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DBATCH_SIZE={batch_size}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCAKE_FMHA_PAGE_SIZE={selector['PAGE_SIZE']}",
            "-DCAKE_FMHA_FULL_BLOCKS=0",
            "-DCAKE_FMHA_NVFP4=1",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA NVFP4 decode/reduce JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_quant_nvfp4_module(
    target: CakeFmhaTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
):
    module = gen_cake_fmha_decode_quant_nvfp4_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        page_size,
    ).build_and_load()
    logger.info("Loaded Cake FMHA NVFP4 decode/reduce module: %s", module)
    return module


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
    "gen_cake_fmha_context_bf16_module",
    "gen_cake_fmha_context_fp8_module",
    "gen_cake_fmha_context_nvfp4_module",
    "gen_cake_fmha_compat_module",
    "gen_cake_fmha_decode_native_bf16_module",
    "gen_cake_fmha_decode_native_fp16_hd512_module",
    "gen_cake_fmha_decode_native_fp16_nhd_module",
    "get_cake_fmha_context_bf16_uri",
    "get_cake_fmha_context_fp8_uri",
    "get_cake_fmha_context_nvfp4_uri",
    "get_cake_fmha_compat_uri",
    "get_cake_fmha_csrc_dir",
    "get_cake_fmha_decode_native_bf16_uri",
    "get_cake_fmha_decode_native_fp16_hd512_uri",
    "get_cake_fmha_decode_native_fp16_nhd_uri",
    "get_cake_fmha_manifest",
    "load_cake_fmha_context_bf16_module",
    "load_cake_fmha_context_fp8_module",
    "load_cake_fmha_context_nvfp4_module",
    "load_cake_fmha_compat_module",
    "load_cake_fmha_decode_native_bf16_module",
    "load_cake_fmha_decode_native_fp16_hd512_module",
    "load_cake_fmha_decode_native_fp16_nhd_module",
]
