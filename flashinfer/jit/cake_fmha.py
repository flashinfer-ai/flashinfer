"""JIT loader for the standalone Cake FMHA product."""

from __future__ import annotations

import functools
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

CAKE_FMHA_FLASHINFER_MATRIX_REVISION = "5b8da12050f80a5b5cb2bab9e87d9635a8872e5b"
# Build tag carried by every Cake FMHA JIT module name.  It is the tag of the
# last pinned source package, kept so the JIT cache and AOT module names do not
# change; nothing is hashed at runtime (ninja depfiles track source edits).
CAKE_FMHA_JIT_TAG = "34d36b82be62_48d627ad25ca"

_TARGET_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_MANIFEST_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_DECODE_NATIVE_BF16_JIT_BINDING = "jit/cake_fmha_decode_native_bf16_jit_binding.cu"
_DECODE_BALANCED_JIT_BINDING = "jit/cake_fmha_decode_balanced_jit_binding.cu"
CAKE_FMHA_BALANCED_DTYPES = ("bf16", "fp16")
CAKE_FMHA_BALANCED_MTP_ROWS = (32, 64)
_DECODE_BALANCED_FP8_JIT_BINDING = "jit/cake_fmha_decode_balanced_fp8_jit_binding.cu"
_DECODE_BALANCED_HD64_JIT_BINDING = "jit/cake_fmha_decode_balanced_hd64_jit_binding.cu"
_DECODE_BALANCED_HD256_JIT_BINDING = (
    "jit/cake_fmha_decode_balanced_hd256_jit_binding.cu"
)
# FP8-KV balanced decode: the query / output dtype axis of one adapter
# (``CAKE_FMHA_BALANCED_Q_DTYPE`` = index into this tuple).
CAKE_FMHA_BALANCED_FP8_Q_DTYPES = ("fp8", "bf16q", "fp16q")
# BF16 head_dim-256 balanced decode: one program per page size.
CAKE_FMHA_BALANCED_HD256_PAGE_SIZES = (16, 32, 64)
_DECODE_NATIVE_BF16_HD256_SMALLM_JIT_BINDING = (
    "jit/cake_fmha_decode_native_bf16_hd256_smallm_jit_binding.cu"
)
CAKE_FMHA_SMALLM_ROWS = (32, 64)
CAKE_FMHA_SMALLM_PAGE_SIZES = (16, 32, 64)
CAKE_FMHA_SMALLM_MAX_NUM_SPLIT = 256
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


@functools.cache
def get_cake_fmha_manifest() -> dict[str, Any]:
    """Load the checked-in core registry (components, routes, capability)."""

    registry = json.loads((get_cake_fmha_csrc_dir() / "registry.json").read_text())
    if registry.get("product") != "cake_fmha":
        raise RuntimeError("Cake FMHA registry has an invalid product identifier")
    if (
        registry.get("flashinfer_matrix_revision")
        != CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    ):
        raise RuntimeError(
            "Cake FMHA registry has an unexpected FlashInfer matrix revision"
        )
    capability = registry.get("capability", {})
    if not capability.get("complete") or capability.get("cake_coverage_ratio") != 1.0:
        raise RuntimeError(
            "Cake FMHA registry does not cover its pinned FlashInfer matrix"
        )
    return registry


def get_cake_fmha_compat_uri(target: CakeFmhaTarget) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    return f"cake_fmha_compat_v1_{target}_{CAKE_FMHA_JIT_TAG}"


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
    if page_size not in (16, 32):
        raise ValueError("decode-quant BF16Q requires page_size 16 or 32")
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
    if page_size not in (16, 32):
        raise ValueError("decode-quant FP8 requires page_size 16 or 32")
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
    """Return whether the registry contains this BF16 route."""

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
    arch = _TARGET_MANIFEST_ARCH[target]
    body = csrc_dir / member["sources"][arch]
    launch_override = member.get("launch_override") or {}
    if "by_arch" in launch_override:
        launch_override = {**launch_override, **launch_override["by_arch"][arch]}
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
            "Cake FMHA decode-native BF16 specialization is absent from the registry"
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


def cake_fmha_balanced_n_rows(q_len: int) -> int:
    """Packed-row tile of the balanced decode kernel serving ``q_len``.

    ``q_len == 1`` uses the eight-row kernel (0 = no packed tile); ``3..4``
    pack into the 32-row MTP tile and ``5..8`` into the 64-row tile.
    """

    if q_len == 1:
        return 0
    if 3 <= q_len <= 4:
        return 32
    if 5 <= q_len <= 8:
        return 64
    raise ValueError(
        f"balanced decode serves q_len 1 or 3..8 (packed MTP tiles), got {q_len}"
    )


def cake_fmha_balanced_component_name(q_len: int, dtype: str = "bf16") -> str:
    """Manifest component of the balanced decode kernel serving ``q_len`` in ``dtype``."""

    if dtype not in CAKE_FMHA_BALANCED_DTYPES:
        raise ValueError(
            f"balanced decode serves the dtypes {CAKE_FMHA_BALANCED_DTYPES}, got {dtype!r}"
        )
    n_rows = cake_fmha_balanced_n_rows(q_len)
    if n_rows == 0:
        return f"decode_balanced_{dtype}"
    return f"decode_balanced_{dtype}_mtp_n{n_rows}"


def get_cake_fmha_decode_balanced_uri(
    target: CakeFmhaTarget, q_len: int, dtype: str = "bf16"
) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    component = cake_fmha_balanced_component_name(q_len, dtype)
    return f"cake_fmha_{component}_{target}_q{q_len}_{CAKE_FMHA_JIT_TAG}"


def get_cake_fmha_decode_balanced_bf16_uri(target: CakeFmhaTarget, q_len: int) -> str:
    return get_cake_fmha_decode_balanced_uri(target, q_len, "bf16")


def get_cake_fmha_decode_balanced_fp16_uri(target: CakeFmhaTarget, q_len: int) -> str:
    return get_cake_fmha_decode_balanced_uri(target, q_len, "fp16")


@functools.cache
def gen_cake_fmha_decode_balanced_module(
    target: CakeFmhaTarget, q_len: int, dtype: str = "bf16"
) -> JitSpec:
    """Build the on-device load-balanced decode module for one ``(dtype, q_len)``.

    Batch, heads and KV lengths are runtime kernel arguments, so one module
    serves every shape of a ``q_len``; the dtype selects the generated program
    (the same two Cake ForGen kernels rendered with BF16 or FP16 Q/K/V/O) and only the
    packed-row tile (32/64 rows for q_len 3..8) selects a different manifest
    component.  One adapter serves both dtypes (``CAKE_FMHA_BALANCED_FP16``).
    """

    component = cake_fmha_balanced_component_name(q_len, dtype)
    manifest_component = get_cake_fmha_manifest()["components"][component]
    sources = _get_component_sources(
        component, target, {}, _DECODE_BALANCED_JIT_BINDING
    )
    n_rows = cake_fmha_balanced_n_rows(q_len)
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_balanced_uri(target, q_len, dtype),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DQ_LEN={q_len}",
            f"-DCAKE_FMHA_BALANCED_N_ROWS={n_rows}",
            f"-DCAKE_FMHA_BALANCED_FP16={int(dtype == 'fp16')}",
            f"-DCAKE_FMHA_BALANCED_LAUNCH={manifest_component['launch_binding']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info(
        "Generated Cake FMHA balanced %s decode JIT spec: %s", dtype.upper(), spec.name
    )
    return spec


def gen_cake_fmha_decode_balanced_bf16_module(
    target: CakeFmhaTarget, q_len: int
) -> JitSpec:
    return gen_cake_fmha_decode_balanced_module(target, q_len, "bf16")


def gen_cake_fmha_decode_balanced_fp16_module(
    target: CakeFmhaTarget, q_len: int
) -> JitSpec:
    return gen_cake_fmha_decode_balanced_module(target, q_len, "fp16")


@functools.cache
def load_cake_fmha_decode_balanced_module(
    target: CakeFmhaTarget, q_len: int, dtype: str = "bf16"
):
    module = gen_cake_fmha_decode_balanced_module(target, q_len, dtype).build_and_load()
    logger.info("Loaded Cake FMHA balanced %s decode module: %s", dtype.upper(), module)
    return module


@functools.cache
def load_cake_fmha_decode_balanced_bf16_module(target: CakeFmhaTarget, q_len: int):
    return load_cake_fmha_decode_balanced_module(target, q_len, "bf16")


@functools.cache
def load_cake_fmha_decode_balanced_fp16_module(target: CakeFmhaTarget, q_len: int):
    return load_cake_fmha_decode_balanced_module(target, q_len, "fp16")


def cake_fmha_balanced_fp8_component_name(q_dtype: str) -> str:
    """Manifest component of the FP8-KV balanced decode kernel for ``q_dtype``.

    ``"fp8"`` is the all-E4M3 instance (E4M3 query through a u8 TMA map, E4M3
    output); ``"bf16q"`` / ``"fp16q"`` read a BF16 / FP16 query in place over
    the E4M3 cache and write the output in the query dtype.
    """

    if q_dtype not in CAKE_FMHA_BALANCED_FP8_Q_DTYPES:
        raise ValueError(
            "the FP8-KV balanced decode serves the query dtypes "
            f"{CAKE_FMHA_BALANCED_FP8_Q_DTYPES}, got {q_dtype!r}"
        )
    return f"decode_balanced_{q_dtype}"


def get_cake_fmha_decode_balanced_fp8_uri(target: CakeFmhaTarget, q_dtype: str) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    component = cake_fmha_balanced_fp8_component_name(q_dtype)
    return f"cake_fmha_{component}_{target}_{CAKE_FMHA_JIT_TAG}"


@functools.cache
def gen_cake_fmha_decode_balanced_fp8_module(
    target: CakeFmhaTarget, q_dtype: str
) -> JitSpec:
    """Build the on-device load-balanced FP8-KV decode module for one query dtype.

    One adapter serves the three generated programs (``CAKE_FMHA_BALANCED_Q_DTYPE``
    0 = E4M3 query, 1 = BF16, 2 = FP16 over the E4M3 cache); batch, heads and KV
    lengths are runtime kernel arguments, so one module per query dtype serves
    every shape (``q_len == 1``; the scales are host scalars).
    """

    component = cake_fmha_balanced_fp8_component_name(q_dtype)
    manifest_component = get_cake_fmha_manifest()["components"][component]
    sources = _get_component_sources(
        component, target, {}, _DECODE_BALANCED_FP8_JIT_BINDING
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_balanced_fp8_uri(target, q_dtype),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DCAKE_FMHA_BALANCED_Q_DTYPE={CAKE_FMHA_BALANCED_FP8_Q_DTYPES.index(q_dtype)}",
            f"-DCAKE_FMHA_BALANCED_LAUNCH={manifest_component['launch_binding']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info(
        "Generated Cake FMHA balanced %s decode JIT spec: %s", q_dtype, spec.name
    )
    return spec


@functools.cache
def load_cake_fmha_decode_balanced_fp8_module(target: CakeFmhaTarget, q_dtype: str):
    module = gen_cake_fmha_decode_balanced_fp8_module(target, q_dtype).build_and_load()
    logger.info("Loaded Cake FMHA balanced %s decode module: %s", q_dtype, module)
    return module


# Structural instances of the BF16 head_dim-64 balanced decode body: the kernel of
# record processes eight softmax columns and serves 1..8 query heads per KV head;
# the ``_g16`` instance processes sixteen and serves 9..16 (Cake round 5, unit 37).
CAKE_FMHA_BALANCED_HD64_MAX_GROUPS = (8, 16)


def cake_fmha_balanced_hd64_component_name(max_group: int = 8) -> str:
    """Manifest component of the BF16 head_dim-64 balanced decode kernel for ``max_group``."""

    if max_group not in CAKE_FMHA_BALANCED_HD64_MAX_GROUPS:
        raise ValueError(
            "the BF16 head_dim-64 balanced decode serves the head-group bounds "
            f"{CAKE_FMHA_BALANCED_HD64_MAX_GROUPS}, got {max_group!r}"
        )
    return (
        "decode_balanced_bf16_hd64"
        if max_group == 8
        else f"decode_balanced_bf16_hd64_g{max_group}"
    )


def cake_fmha_balanced_hd64_max_group(num_qo_heads: int, num_kv_heads: int) -> int:
    """Head-group bound of the instance serving ``num_qo_heads // num_kv_heads`` heads per KV head."""

    if (
        num_kv_heads <= 0
        or num_qo_heads % num_kv_heads
        or not 1 <= num_qo_heads // num_kv_heads <= 16
    ):
        raise ValueError(
            "the BF16 head_dim-64 balanced decode serves 1..16 query heads per KV head"
        )
    return 8 if num_qo_heads // num_kv_heads <= 8 else 16


def get_cake_fmha_decode_balanced_hd64_uri(
    target: CakeFmhaTarget, max_group: int = 8
) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    component = cake_fmha_balanced_hd64_component_name(max_group)
    return f"cake_fmha_{component}_{target}_{CAKE_FMHA_JIT_TAG}"


@functools.cache
def gen_cake_fmha_decode_balanced_hd64_module(
    target: CakeFmhaTarget, max_group: int = 8
) -> JitSpec:
    """Build the on-device load-balanced BF16 head_dim-64 decode module.

    One body (Q16Kv128 ForGen, page 16, ``q_len == 1``) in two structural
    instances: ``max_group`` 8 (eight softmax columns, 1..8 query heads per KV
    head -- the kernel of record) and 16 (sixteen columns, 9..16); batch, heads
    and KV lengths are runtime kernel arguments.
    """

    component = cake_fmha_balanced_hd64_component_name(max_group)
    manifest_component = get_cake_fmha_manifest()["components"][component]
    sources = _get_component_sources(
        component, target, {}, _DECODE_BALANCED_HD64_JIT_BINDING
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_balanced_hd64_uri(target, max_group),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DCAKE_FMHA_BALANCED_LAUNCH={manifest_component['launch_binding']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA balanced BF16 hd64 decode JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_balanced_hd64_module(
    target: CakeFmhaTarget, max_group: int = 8
):
    module = gen_cake_fmha_decode_balanced_hd64_module(
        target, max_group
    ).build_and_load()
    logger.info(
        "Loaded Cake FMHA balanced BF16 hd64 decode module (max_group %d): %s",
        max_group,
        module,
    )
    return module


def cake_fmha_balanced_hd256_component_name(page_size: int) -> str:
    """Manifest component of the BF16 head_dim-256 balanced decode kernel for ``page_size``."""

    if page_size not in CAKE_FMHA_BALANCED_HD256_PAGE_SIZES:
        raise ValueError(
            "the BF16 head_dim-256 balanced decode serves the page sizes "
            f"{CAKE_FMHA_BALANCED_HD256_PAGE_SIZES}, got {page_size!r}"
        )
    return f"decode_balanced_bf16_hd256_p{page_size}"


def get_cake_fmha_decode_balanced_hd256_uri(
    target: CakeFmhaTarget, page_size: int
) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    component = cake_fmha_balanced_hd256_component_name(page_size)
    return f"cake_fmha_{component}_{target}_{CAKE_FMHA_JIT_TAG}"


@functools.cache
def gen_cake_fmha_decode_balanced_hd256_module(
    target: CakeFmhaTarget, page_size: int
) -> JitSpec:
    """Build the on-device load-balanced BF16 head_dim-256 decode module for one page size.

    The page size is a structural instance of the kernel (16 / 32 / 64 tokens per
    page, one exported program each); batch, heads, KV lengths and the uniform
    query length (1..8, per-row tiles) are runtime kernel arguments.
    """

    component = cake_fmha_balanced_hd256_component_name(page_size)
    manifest_component = get_cake_fmha_manifest()["components"][component]
    sources = _get_component_sources(
        component, target, {}, _DECODE_BALANCED_HD256_JIT_BINDING
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_balanced_hd256_uri(target, page_size),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DCAKE_FMHA_BALANCED_PAGE_SIZE={page_size}",
            f"-DCAKE_FMHA_BALANCED_LAUNCH={manifest_component['launch_binding']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info(
        "Generated Cake FMHA balanced BF16 hd256 page-%d decode JIT spec: %s",
        page_size,
        spec.name,
    )
    return spec


@functools.cache
def load_cake_fmha_decode_balanced_hd256_module(target: CakeFmhaTarget, page_size: int):
    module = gen_cake_fmha_decode_balanced_hd256_module(
        target, page_size
    ).build_and_load()
    logger.info(
        "Loaded Cake FMHA balanced BF16 hd256 page-%d decode module: %s",
        page_size,
        module,
    )
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
    )


@functools.cache
def cake_fmha_smallm_component_name(n_rows: int, page_size: int) -> str:
    """Manifest component of one small-M hd256 structural instance."""

    if n_rows not in CAKE_FMHA_SMALLM_ROWS:
        raise ValueError(f"small-M hd256 decode packs 32 or 64 rows, got {n_rows}")
    if page_size not in CAKE_FMHA_SMALLM_PAGE_SIZES:
        raise ValueError(
            f"small-M hd256 decode supports page sizes 16/32/64, got {page_size}"
        )
    return f"decode_native_bf16_hd256_smallm_n{n_rows}_p{page_size}"


def _validate_smallm_specialization(
    target: CakeFmhaTarget,
    n_rows: int,
    page_size: int,
    q_len: int,
    group: int,
    num_split: int,
) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake FMHA target: {target}")
    component = cake_fmha_smallm_component_name(n_rows, page_size)
    if q_len <= 0 or group <= 0 or not (n_rows // 2 < q_len * group <= n_rows):
        raise ValueError(
            "small-M hd256 decode packs q_len * group rows into the smallest 32/64-row tile"
        )
    if n_rows % group:
        raise ValueError(
            "small-M hd256 decode requires the GQA group to divide the tile rows"
        )
    if not 1 <= num_split <= CAKE_FMHA_SMALLM_MAX_NUM_SPLIT:
        raise ValueError("small-M hd256 decode NUM_SPLIT must be in [1, 256]")
    return component


def get_cake_fmha_decode_native_bf16_hd256_smallm_uri(
    target: CakeFmhaTarget,
    n_rows: int,
    page_size: int,
    q_len: int,
    group: int,
    num_split: int,
) -> str:
    component = _validate_smallm_specialization(
        target, n_rows, page_size, q_len, group, num_split
    )
    return f"cake_fmha_{component}_{target}_q{q_len}_g{group}_s{num_split}"


def gen_cake_fmha_decode_native_bf16_hd256_smallm_module(
    target: CakeFmhaTarget,
    n_rows: int,
    page_size: int,
    q_len: int,
    group: int,
    num_split: int,
) -> JitSpec:
    """Build one small-M BF16 head-dim-256 speculative decode specialization."""

    component = _validate_smallm_specialization(
        target, n_rows, page_size, q_len, group, num_split
    )
    manifest_component = get_cake_fmha_manifest()["components"][component]
    sources = _get_component_sources(
        component,
        target,
        {},
        _DECODE_NATIVE_BF16_HD256_SMALLM_JIT_BINDING,
    )
    spec = gen_jit_spec(
        name=get_cake_fmha_decode_native_bf16_hd256_smallm_uri(
            target, n_rows, page_size, q_len, group, num_split
        ),
        sources=list(sources),
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            "-use_fast_math",
            f"-DQ_LEN={q_len}",
            f"-DGROUP={group}",
            f"-DQ_BOX_ROWS={n_rows // group}",
            f"-DNUM_SPLIT={num_split}",
            f"-DCAKE_FMHA_SMALLM_N_ROWS={n_rows}",
            f"-DCAKE_FMHA_SMALLM_PAGE_SIZE={page_size}",
            f"-DCAKE_FMHA_SMALLM_LAUNCH={manifest_component['launch_binding']}",
        ],
        extra_include_paths=[get_cake_fmha_csrc_dir(), jit_env.FLASHINFER_CSRC_DIR],
    )
    logger.info("Generated Cake FMHA small-M hd256 decode JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fmha_decode_native_bf16_hd256_smallm_module(
    target: CakeFmhaTarget,
    n_rows: int,
    page_size: int,
    q_len: int,
    group: int,
    num_split: int,
):
    return gen_cake_fmha_decode_native_bf16_hd256_smallm_module(
        target, n_rows, page_size, q_len, group, num_split
    ).build_and_load()


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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
        f"_{CAKE_FMHA_JIT_TAG}"
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
    "CAKE_FMHA_BALANCED_DTYPES",
    "CAKE_FMHA_BALANCED_FP8_Q_DTYPES",
    "CAKE_FMHA_BALANCED_HD256_PAGE_SIZES",
    "CAKE_FMHA_FLASHINFER_MATRIX_REVISION",
    "CAKE_FMHA_JIT_TAG",
    "CakeFmhaTarget",
    "gen_cake_fmha_context_bf16_module",
    "gen_cake_fmha_context_fp8_module",
    "gen_cake_fmha_context_nvfp4_module",
    "gen_cake_fmha_compat_module",
    "gen_cake_fmha_decode_balanced_bf16_module",
    "gen_cake_fmha_decode_balanced_fp16_module",
    "gen_cake_fmha_decode_balanced_fp8_module",
    "gen_cake_fmha_decode_balanced_hd256_module",
    "gen_cake_fmha_decode_balanced_hd64_module",
    "cake_fmha_balanced_hd64_component_name",
    "cake_fmha_balanced_hd64_max_group",
    "CAKE_FMHA_BALANCED_HD64_MAX_GROUPS",
    "gen_cake_fmha_decode_balanced_module",
    "gen_cake_fmha_decode_native_bf16_module",
    "gen_cake_fmha_decode_native_fp16_hd512_module",
    "gen_cake_fmha_decode_native_fp16_nhd_module",
    "get_cake_fmha_context_bf16_uri",
    "get_cake_fmha_context_fp8_uri",
    "get_cake_fmha_context_nvfp4_uri",
    "get_cake_fmha_compat_uri",
    "get_cake_fmha_csrc_dir",
    "get_cake_fmha_decode_balanced_bf16_uri",
    "get_cake_fmha_decode_balanced_fp16_uri",
    "get_cake_fmha_decode_balanced_fp8_uri",
    "get_cake_fmha_decode_balanced_hd256_uri",
    "get_cake_fmha_decode_balanced_hd64_uri",
    "get_cake_fmha_decode_balanced_uri",
    "get_cake_fmha_decode_native_bf16_uri",
    "get_cake_fmha_decode_native_fp16_hd512_uri",
    "get_cake_fmha_decode_native_fp16_nhd_uri",
    "get_cake_fmha_manifest",
    "load_cake_fmha_context_bf16_module",
    "load_cake_fmha_context_fp8_module",
    "load_cake_fmha_context_nvfp4_module",
    "load_cake_fmha_compat_module",
    "load_cake_fmha_decode_balanced_bf16_module",
    "load_cake_fmha_decode_balanced_fp16_module",
    "load_cake_fmha_decode_balanced_fp8_module",
    "load_cake_fmha_decode_balanced_hd256_module",
    "load_cake_fmha_decode_balanced_hd64_module",
    "load_cake_fmha_decode_balanced_module",
    "load_cake_fmha_decode_native_bf16_module",
    "load_cake_fmha_decode_native_fp16_hd512_module",
    "load_cake_fmha_decode_native_fp16_nhd_module",
]
