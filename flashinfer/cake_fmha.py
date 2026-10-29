"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  DCP
metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
import functools
import heapq
import warnings
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any, Literal

import torch
import tvm_ffi

from .jit.cake_fmha import (
    CakeFmhaTarget,
    get_cake_fmha_manifest,
    _is_cake_fmha_decode_native_bf16_available,
    load_cake_fmha_context_bf16_module,
    load_cake_fmha_context_fp8_module,
    load_cake_fmha_context_nvfp4_module,
    load_cake_fmha_context_fp16_hd256_module,
    load_cake_fmha_context_fp8_hd256_module,
    load_cake_fmha_compat_module,
    load_cake_fmha_decode_native_bf16_hd256_smallm_module,
    load_cake_fmha_decode_balanced_bf16_module,
    load_cake_fmha_decode_balanced_fp16_module,
    load_cake_fmha_decode_balanced_fp8_module,
    load_cake_fmha_decode_balanced_hd256_module,
    load_cake_fmha_decode_balanced_hd64_module,
    cake_fmha_balanced_hd64_max_group,
    load_cake_fmha_decode_native_bf16_module,
    load_cake_fmha_decode_native_fp16_hd512_module,
    load_cake_fmha_decode_native_fp16_nhd_module,
    load_cake_fmha_decode_quant_bf16q_module,
    load_cake_fmha_decode_quant_fp8_module,
    load_cake_fmha_decode_quant_nvfp4_module,
)
from .cake_fmha_request_ordered import (
    CakeFmhaRequestOrderedDecodePlan,
    plan_cake_fmha_request_ordered_paged_decode,
)
from .cake_fmha_request_ordered import (
    _fallback_cake_fmha_request_ordered_plan as _fallback_cake_fmha_request_ordered_plan,
)
from .cake_fmha_request_ordered import (
    _run_cake_fmha_request_ordered_paged_decode as _run_cake_fmha_request_ordered_paged_decode,
)
from .utils import get_compute_capability, get_device_sm_count, log2e


@dataclass(frozen=True)
class CakeFmhaDecodeRoute:
    """One exact manifest-backed optimized decode specialization."""

    target: CakeFmhaTarget
    batch_size: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    has_sink: bool
    has_window: bool
    use_scale_ptr: bool
    retain_kv_l2: bool
    component: Literal[
        "decode_aligned_bf16",
        "decode_aligned_fp16",
        "decode_balanced_bf16",
        "decode_balanced_fp16",
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd256",
        "decode_native_bf16",
        "decode_native_bf16_hd256_smallm",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8_reduce",
        "decode_quant_fp8",
        "decode_quant_nvfp4",
    ] = "decode_native_bf16"
    page_size: int = 16
    # Small-M hd256 speculative decode: one resident wave of Split-KV CTAs.
    num_split: int = 0
    # The caller passed an ``lse`` output.  The exact-shape decode-native BF16
    # members never write it, so such a route is served by the generic member.
    return_lse: bool = False
    # Aligned hd128 q1 decode: the identifier of the generated program
    # (``flashinfer.jit.cake_fmha_aligned.ALIGNED_C<cluster>_<DTYPE>``) whose
    # cluster size ``aligned_cluster_size`` selected; ``None`` on every other route.
    aligned_program: str | None = None
    # Generated decode program: ``(family, identifier)`` of the in-tree program
    # (``flashinfer.jit.cake_fmha_<family>.<identifier>``) the route launches,
    # selected from host scalars by ``_DECODE_PROGRAM_ROUTES``; ``None`` keeps
    # the route's JIT member.  A ``return_lse`` route keeps the program of the
    # same launch without LSE but runs the JIT member (the BF16 programs write
    # no caller LSE): see :func:`_decode_route_runs_program`.
    program: tuple[str, str] | None = None
    # The bmm1 (base 2) and bmm2 scales are FP32 device scalars the program
    # reads in the kernel (no host read).
    device_scales: bool = False


CakeFmhaContextComponent = Literal[
    "context_bf16",
    "context_fp16_hd256",
    "context_fp8",
    "context_fp8_hd256",
    "context_nvfp4",
]


@dataclass(frozen=True)
class CakeFmhaContextRoute:
    """One exact manifest-backed optimized context specialization."""

    target: CakeFmhaTarget
    component: CakeFmhaContextComponent
    num_m_blocks: int
    num_q_heads: int
    num_kv_heads: int
    pack_g: int
    page_size: int
    l2_swizzle: int
    is_causal: bool
    return_lse: bool
    enable_sink: bool
    exact_profile: Literal["q511", "q257"] | None = None
    # Both scales are device tensors (FP8 / NVFP4): the program reads them
    # through device pointers instead of host scalars.
    device_scales: bool = False
    # The generated program serving the route (an identifier of
    # ``_CONTEXT_PROGRAMS``), or ``None`` for the component's JIT module.
    program: str | None = None


# These are the exact optimized routes in the pinned 57,280-cell matrix.  Keep
# the component sequences explicit: hd256 needs its staging/scatter support,
# and split-KV NVFP4 decode may also need the shared reduction component.
_PRODUCT_ROUTE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "ctx_bf16_hnd_hd128_hgpack_03df_v2": ("context_bf16",),
    "ctx_fp16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp16_hd256",
    ),
    "ctx_fp8_bf16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp8_hd256",
    ),
    "ctx_fp8_hnd_hd128_hgpack_48b5_v1": ("context_fp8",),
    "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1": ("context_nvfp4",),
    "decode_balanced_bf16_v1": (
        "decode_balanced_bf16",
        "decode_balanced_bf16_mtp_n32",
        "decode_balanced_bf16_mtp_n64",
    ),
    "decode_balanced_fp16_v1": (
        "decode_balanced_fp16",
        "decode_balanced_fp16_mtp_n32",
        "decode_balanced_fp16_mtp_n64",
    ),
    "decode_balanced_fp8_v1": ("decode_balanced_fp8",),
    "decode_balanced_bf16q_v1": ("decode_balanced_bf16q",),
    "decode_balanced_fp16q_v1": ("decode_balanced_fp16q",),
    "decode_balanced_bf16_hd64_v1": (
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd64_g16",
    ),
    "decode_balanced_bf16_hd256_v1": (
        "decode_balanced_bf16_hd256_p16",
        "decode_balanced_bf16_hd256_p32",
        "decode_balanced_bf16_hd256_p64",
    ),
    "decode_native_bf16_v1_bece": ("decode_native_bf16",),
    "decode_native_bf16_hd256_smallm_v1": (
        "decode_native_bf16_hd256_smallm_n32_p16",
        "decode_native_bf16_hd256_smallm_n32_p32",
        "decode_native_bf16_hd256_smallm_n32_p64",
        "decode_native_bf16_hd256_smallm_n64_p16",
        "decode_native_bf16_hd256_smallm_n64_p32",
        "decode_native_bf16_hd256_smallm_n64_p64",
    ),
    "decode_native_fp16_hd512_v1_66b1": ("decode_native_fp16_hd512",),
    "decode_native_fp16_nhd_v1_f32d": ("decode_native_fp16_nhd",),
    "decode_quantized_bf16q_9d8b_v1": ("decode_quant_bf16q",),
    "decode_quantized_fp8_8e5b_v1": (
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
    ),
    "decode_quantized_nvfp4_8e5b_v1": (
        "decode_quant_nvfp4",
        "decode_quant_fp8_reduce",
    ),
}

# These components have an authenticated FlashInfer TVM-FFI adapter in the
# checked-in package.  A route remains fail-closed until every component in its
# declared chain is present here and covered by the adapter digest.
_AUTHENTICATED_JIT_COMPONENTS = frozenset(
    {
        "compat_v1",
        "context_bf16",
        "context_fp16_hd256",
        "context_fp8",
        "context_fp8_hd256",
        "context_hd256_support",
        "context_nvfp4",
        "decode_balanced_bf16",
        "decode_balanced_bf16_mtp_n32",
        "decode_balanced_bf16_mtp_n64",
        "decode_balanced_fp16",
        "decode_balanced_fp16_mtp_n32",
        "decode_balanced_fp16_mtp_n64",
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd64_g16",
        "decode_balanced_bf16_hd256_p16",
        "decode_balanced_bf16_hd256_p32",
        "decode_balanced_bf16_hd256_p64",
        "decode_native_bf16",
        "decode_native_bf16_hd256_smallm_n32_p16",
        "decode_native_bf16_hd256_smallm_n32_p32",
        "decode_native_bf16_hd256_smallm_n32_p64",
        "decode_native_bf16_hd256_smallm_n64_p16",
        "decode_native_bf16_hd256_smallm_n64_p32",
        "decode_native_bf16_hd256_smallm_n64_p64",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
        "decode_quant_nvfp4",
    }
)

# The aligned hd128 q1 decode programs are not registry components: they ship
# as the generated in-tree unit ``csrc/cake_fmha_aligned/`` whose loader
# ``flashinfer.jit.cake_fmha_aligned`` publishes one ``CakeProgram`` constant
# per cluster size and dtype, authenticated by the program's content digest.
# A route's component chain is its one program identifier, and the route is
# optimized exactly when that identifier is one of these.  The table pins the
# delivered record of each identifier: ``(cluster_size, dtype, program name)``;
# a loader whose record differs fails closed at load.
_ALIGNED_PROGRAMS: dict[str, tuple[int, str, str]] = {
    "ALIGNED_C8_BF16": (8, "bf16", "cake_fmha_aligned_a9cc814b948505a2cc1b"),
    "ALIGNED_C4_BF16": (4, "bf16", "cake_fmha_aligned_2bd88c52bb7a16a1814f"),
    "ALIGNED_C2_BF16": (2, "bf16", "cake_fmha_aligned_1052b6ae93d516e78ed6"),
    "ALIGNED_C1_BF16": (1, "bf16", "cake_fmha_aligned_6f88390dacdd946afa74"),
    "ALIGNED_C8_FP16": (8, "fp16", "cake_fmha_aligned_8d67475a5eaad8c4abc2"),
    "ALIGNED_C4_FP16": (4, "fp16", "cake_fmha_aligned_b100f226ebd9ce6884cc"),
    "ALIGNED_C2_FP16": (2, "fp16", "cake_fmha_aligned_6b02ec3e27916ce5eee5"),
    "ALIGNED_C1_FP16": (1, "fp16", "cake_fmha_aligned_300f2001dc3a9f642eeb"),
}
_ALIGNED_COMPONENTS = ("decode_aligned_bf16", "decode_aligned_fp16")
# ``CakeFmhaTarget`` -> the exact ``CakeProgram`` target of the same device.
_ALIGNED_PROGRAM_TARGET: dict[str, str] = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
# The binding's host parameters, in the kernel's order (then the grid).
_ALIGNED_HOST_PARAMS = (
    "Qt",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "uniform_seq_len",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "batch_size",
)


# The generated paged context programs: the units ``csrc/cake_fmha_context_bf16/``,
# ``csrc/cake_fmha_context_fp8/`` and ``csrc/cake_fmha_context_nvfp4/`` whose
# loaders ``flashinfer.jit.cake_fmha_context_{bf16,fp8,nvfp4}`` publish one
# ``CakeProgram`` constant per program.  Every program is causal, without LSE
# or sink.  Per target, a route key ``(component, scale form, num_q_heads,
# num_kv_heads, pack_g, num_m_blocks, page_size, exact class)`` names the
# program serving it.  The scale form is ``"host"`` (host-scalar scales; the
# BF16 output scale is 1) or ``"device"`` (both FP8 / NVFP4 scales are device
# tensors).  The key holds neither the batch size nor the L2 section: a program
# serves any batch (when the work units do not fill the last L2 section, the
# program walks that section in another order, which changes the tile order
# only, not a tile's arithmetic).  The exact class names a one-tile BF16 launch
# class (``_context_bf16_exact_profile``); ``None`` serves any batch.  A key
# without a program keeps the component's JIT module.
_CONTEXT_PROGRAM_ROUTES: dict[
    str, dict[tuple[str, str, int, int, int, int, int, str | None], str]
] = {
    "sm100a": {
        ("context_bf16", "host", 64, 8, 8, 256, 16, None): "Q8192",
        ("context_bf16", "host", 32, 4, 8, 2, 16, None): "B128",
        ("context_bf16", "host", 10, 2, 5, 11, 32, "q511"): "Q511_SM100A",
        ("context_bf16", "host", 10, 2, 5, 6, 1024, "q257"): "Q257_SM100A",
        ("context_fp8", "host", 32, 4, 8, 1, 64, None): "B256",
        ("context_fp8", "device", 32, 4, 8, 1, 64, None): "B256_SCALE_PTR",
        ("context_nvfp4", "host", 32, 4, 8, 16, 16, None): "Q511_SM100A",
        ("context_nvfp4", "device", 32, 4, 8, 16, 16, None): "Q511_SCALE_PTR_SM100A",
    },
    "sm103a": {
        ("context_bf16", "host", 64, 8, 8, 256, 16, None): "Q8192",
        ("context_bf16", "host", 32, 4, 8, 2, 16, None): "B128",
        ("context_bf16", "host", 10, 2, 5, 11, 32, "q511"): "Q511_SM103A",
        ("context_bf16", "host", 10, 2, 5, 6, 1024, "q257"): "Q257_SM103A",
        ("context_fp8", "host", 32, 4, 8, 1, 64, None): "B256_KV4",
        ("context_fp8", "device", 32, 4, 8, 1, 64, None): "B256_SCALE_PTR_KV4",
        ("context_nvfp4", "host", 32, 4, 8, 16, 16, None): "Q511_SM103A",
        ("context_nvfp4", "device", 32, 4, 8, 16, 16, None): "Q511_SCALE_PTR_SM103A",
    },
}
# ``(component, identifier)`` -> the delivered record the host pins: the
# program name, the m-block count of its shape class (``None``: the launch
# passes it) and whether it launches one CTA per tile (otherwise at most one
# persistent CTA per SM).  A loader whose record differs fails closed at load.
_CONTEXT_PROGRAMS: dict[tuple[str, str], tuple[str, int | None, bool]] = {
    ("context_bf16", "Q8192"): (
        "cake_fmha_context_bf16_eb5b942877557703876d",
        256,
        False,
    ),
    ("context_bf16", "B128"): ("cake_fmha_context_bf16_31373706c9fcb1a2b1db", 2, False),
    ("context_bf16", "Q511_SM100A"): (
        "cake_fmha_context_bf16_fe2dfd2eb24d75a1bccc",
        11,
        True,
    ),
    ("context_bf16", "Q511_SM103A"): (
        "cake_fmha_context_bf16_8bd7d679916961178c03",
        11,
        True,
    ),
    ("context_bf16", "Q257_SM100A"): (
        "cake_fmha_context_bf16_d400e81a0185922f0097",
        6,
        True,
    ),
    ("context_bf16", "Q257_SM103A"): (
        "cake_fmha_context_bf16_5fa928327be121e0c987",
        6,
        True,
    ),
    ("context_fp8", "B256"): ("cake_fmha_context_fp8_f57014e4e03dcbce72d8", 1, False),
    ("context_fp8", "B256_SCALE_PTR"): (
        "cake_fmha_context_fp8_45b03632970269bac548",
        1,
        False,
    ),
    ("context_fp8", "B256_KV4"): (
        "cake_fmha_context_fp8_64a9d449f206e9476fd4",
        1,
        False,
    ),
    ("context_fp8", "B256_SCALE_PTR_KV4"): (
        "cake_fmha_context_fp8_7bc99d2dda53eeaad98b",
        1,
        False,
    ),
    ("context_nvfp4", "Q511_SM100A"): (
        "cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SM103A"): (
        "cake_fmha_context_nvfp4_c916619510183749b162",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SCALE_PTR_SM100A"): (
        "cake_fmha_context_nvfp4_ab413a0149f11e7a8463",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SCALE_PTR_SM103A"): (
        "cake_fmha_context_nvfp4_1b4750660638d644a971",
        16,
        False,
    ),
    # The FP8 runtime-size programs: the m-block count is a launch argument.
    ("context_fp8", "RUNTIME_PACK8"): (
        "cake_fmha_context_fp8_3fd021cb557d23a90ca8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4"): (
        "cake_fmha_context_fp8_be3f7f385a3af5fe43b8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2"): (
        "cake_fmha_context_fp8_caade70bb69e18537405",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1"): (
        "cake_fmha_context_fp8_ce1b2c2352f741d468e8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PAGE64"): (
        "cake_fmha_context_fp8_1930bf723069ae9f5446",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PAGE64"): (
        "cake_fmha_context_fp8_91a83484d11d796aaf30",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PAGE64"): (
        "cake_fmha_context_fp8_d427125833943a3251b5",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64"): (
        "cake_fmha_context_fp8_34939d99183a1a6e0baa",
        None,
        False,
    ),
    # The FP8 split-KV programs: the runtime-size arguments, then the split plan.
    ("context_fp8", "RUNTIME_SK_PACK8"): (
        "cake_fmha_context_fp8_ee373474e6d16691cab7",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK4"): (
        "cake_fmha_context_fp8_074b1da06353ee6d95c6",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK2"): (
        "cake_fmha_context_fp8_2208354a624a713e3455",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK1"): (
        "cake_fmha_context_fp8_a93f4040f2fe5b5a85bb",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK8_PAGE64"): (
        "cake_fmha_context_fp8_617e3a6b50380cf6e0a5",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK4_PAGE64"): (
        "cake_fmha_context_fp8_1ea3c441ffa379d0978f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK2_PAGE64"): (
        "cake_fmha_context_fp8_c76354901ca9c86f4fec",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK1_PAGE64"): (
        "cake_fmha_context_fp8_9697d1fcd36a1f4a023e",
        None,
        False,
    ),
}
# The FP8 runtime-size context programs: every length, head count, m-block
# count and page size is a launch argument, so one program serves every causal
# FP8 call without LSE or sink whose GQA group its pack (the query heads one
# 128-row tile stage holds) divides; pack 1 serves every group.  Per target,
# ``(pack_g, page_size)`` names the program: a page-64 program folds the page
# arithmetic into constants, ``None`` serves every page size 16 .. 1024.  The
# host consults them only for a call no exact route serves, and only where
# they are not slower than the component's JIT module
# (:func:`_context_fp8_runtime_serves`).
_CONTEXT_FP8_RUNTIME_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (8, 64): "RUNTIME_PACK8_PAGE64",
        (4, 64): "RUNTIME_PACK4_PAGE64",
        (2, 64): "RUNTIME_PACK2_PAGE64",
        (1, 64): "RUNTIME_PACK1_PAGE64",
        (8, None): "RUNTIME_PACK8",
        (4, None): "RUNTIME_PACK4",
        (2, None): "RUNTIME_PACK2",
        (1, None): "RUNTIME_PACK1",
    },
    "sm103a": {
        (8, 64): "RUNTIME_PACK8_PAGE64",
        (4, 64): "RUNTIME_PACK4_PAGE64",
        (2, 64): "RUNTIME_PACK2_PAGE64",
        (1, 64): "RUNTIME_PACK1_PAGE64",
        (8, None): "RUNTIME_PACK8",
        (4, None): "RUNTIME_PACK4",
        (2, None): "RUNTIME_PACK2",
        (1, None): "RUNTIME_PACK1",
    },
}
# The packs the runtime programs ship besides 1: a pack serves every GQA group
# it divides.
_CONTEXT_FP8_RUNTIME_PACKS = (8, 4, 2)
# The FP8 split-KV context programs: the runtime-size schedule and arguments,
# with the key range of the first ``split_tiles`` tiles split over work units
# of ``split_blocks`` KV blocks (at most ``num_splits`` per tile), whose FP32
# partials the tile's last unit merges.  Per target, ``(pack_g, page_size)``
# names the program as for the runtime-size programs.  The host routes an FP8
# call to one only where its split plan
# (:func:`_context_fp8_split_plan`) predicts a gain over the whole-tile
# program; other calls keep their route.
_CONTEXT_FP8_SPLIT_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    target: {
        key: "RUNTIME_SK" + name.removeprefix("RUNTIME") for key, name in routes.items()
    }
    for target, routes in _CONTEXT_FP8_RUNTIME_ROUTES.items()
}
_CONTEXT_FP8_SPLIT_PROGRAMS = frozenset(
    name for routes in _CONTEXT_FP8_SPLIT_ROUTES.values() for name in routes.values()
)
# Units of one split tile at most (the kernel's MAX_SPLITS), and the FP32
# partial bytes of one split unit: O of both 128-row stages, then the row max
# and row sum (16 bytes per row and stage).
_CONTEXT_FP8_SPLIT_MAX = 16
_CONTEXT_FP8_SPLIT_O_BYTES = 2 * 128 * 128 * 4
_CONTEXT_FP8_SPLIT_ML_BYTES = 2 * 128 * 4 * 4
# The binding's host parameters per component, in the kernel's order (then the
# grid).  The quantized programs take the device-scale parameters in every
# build; a host-scalar program never reads them.
_CONTEXT_COMMON_PARAMS = (
    "LSE_ptr",
    "sinks",
    "page_table_k",
    "page_table_v",
    "seq_lens_q",
    "seq_lens_kv",
    "cu_seq_lens_q",
    "softmax_scale_log2",
)
_CONTEXT_HOST_PARAMS: dict[str, tuple[str, ...]] = {
    "context_bf16": (
        "Q",
        "K",
        "V",
        "O_ptr",
        *_CONTEXT_COMMON_PARAMS,
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
    ),
    "context_fp8": (
        "Q",
        "K",
        "V",
        "O_ptr",
        *_CONTEXT_COMMON_PARAMS,
        "output_scale",
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "bmm1_is_log2",
    ),
    "context_nvfp4": (
        "Q",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
        "O_ptr",
        *_CONTEXT_COMMON_PARAMS,
        "output_scale",
        "total_bh",
        "page_row_stride",
        "num_ctas",
        "dynamic_counter",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "bmm1_is_log2",
    ),
}
# A runtime-size FP8 program takes the sizes after the FP8 parameters: the
# scale source (0: the host scalars, 1: the device scalars), the m-block count,
# the whole 8-unit L2 sections of the tile order (``total_bh // 8``), the work
# units per request, the query heads, the GQA group and the log2 page size.
_CONTEXT_FP8_RUNTIME_PARAMS = (
    *_CONTEXT_HOST_PARAMS["context_fp8"],
    "scale_source",
    "num_m_blocks",
    "full_sections",
    "num_bh_units",
    "num_q_heads",
    "heads_per_group",
    "page_shift",
)


# A split-KV FP8 program takes the runtime-size parameters, then the split
# plan, the FP32 partials and the per-tile arrival counters.
_CONTEXT_FP8_SPLIT_PARAMS = (
    *_CONTEXT_FP8_RUNTIME_PARAMS,
    "split_tiles",
    "split_blocks",
    "num_splits",
    "partial_o",
    "partial_ml",
    "tile_counters",
)


def _context_program_params(
    component: str, num_m_blocks: int | None, program: str | None = None
) -> tuple[str, ...]:
    """The host parameters of a pinned program's binding, in the kernel's order
    (then the grid)."""

    if program in _CONTEXT_FP8_SPLIT_PROGRAMS:
        return _CONTEXT_FP8_SPLIT_PARAMS
    if num_m_blocks is None:
        return _CONTEXT_FP8_RUNTIME_PARAMS
    return _CONTEXT_HOST_PARAMS[component]


def _context_program_facts(program_name: str) -> tuple[str, int | None, bool]:
    for (component, _), (name, num_m_blocks, one_tile) in _CONTEXT_PROGRAMS.items():
        if name == program_name:
            return component, num_m_blocks, one_tile
    raise RuntimeError(f"{program_name} is not a Cake FMHA context program")


def cake_fmha_context_launch_grid(
    program,
    total_bh: int,
    sm_count: int,
    num_m_blocks: int | None = None,
    split_tiles: int = 0,
    num_splits: int = 1,
) -> tuple[int, int, int]:
    """The launch grid of one generated context program: a one-tile program
    launches one CTA per tile (``num_m_blocks x total_bh`` tiles, ``total_bh``
    the work units ``batch_size x num_q_heads / pack_g``), a persistent program
    ``min(sm_count, tiles)`` CTAs, a split-KV program ``min(sm_count, tiles +
    (num_splits - 1) x split_tiles)`` (its work units).  ``program`` is the
    generated loader's ``CakeProgram`` record; ``num_m_blocks`` is the launch's
    m-block count, required for a runtime-size program and else, when given,
    its pinned one."""

    _, pinned, one_tile = _context_program_facts(program.name)
    if pinned is None:
        if num_m_blocks is None:
            raise RuntimeError(f"{program.name} takes the m-block count per launch")
    elif num_m_blocks is None:
        num_m_blocks = pinned
    elif int(num_m_blocks) != pinned:
        raise RuntimeError(f"{program.name} runs {pinned} m-blocks, not {num_m_blocks}")
    tiles = int(num_m_blocks) * int(total_bh)
    units = tiles + (int(num_splits) - 1) * int(split_tiles)
    return (tiles if one_tile else min(int(sm_count), units)), 1, 1


def _aligned_program_id(cluster_size: int, dtype: str) -> str:
    return f"ALIGNED_C{cluster_size}_{dtype.upper()}"


def _aligned_route_program(
    route: CakeFmhaDecodeRoute,
) -> tuple[int, str, str] | None:
    """``(cluster_size, dtype, program name)`` of an aligned route's program, or
    ``None`` when the route's program identifier is not one of the module's or
    not of its dtype."""

    if route.component not in _ALIGNED_COMPONENTS or route.aligned_program is None:
        return None
    program = _ALIGNED_PROGRAMS.get(route.aligned_program)
    if program is None or route.component != f"decode_aligned_{program[1]}":
        return None
    return program


def cake_fmha_aligned_launch_grid(
    program, batch_size: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """The launch grid of one aligned hd128 q1 decode: ``(batch_size *
    num_kv_heads, 1, cluster_size)``, one cluster of the program's CTAs along z
    per (request, KV head) tile.  ``program`` is the generated loader's
    ``CakeProgram`` record; its ``cluster`` is ``(1, 1, cluster_size)``."""

    return int(batch_size) * int(num_kv_heads), 1, int(program.cluster[2])


def _route_components(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute,
) -> tuple[str, ...]:
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component in _ALIGNED_COMPONENTS
    ):
        return (route.aligned_program,) if route.aligned_program is not None else ()
    if isinstance(route, CakeFmhaDecodeRoute) and _decode_route_runs_program(route):
        assert route.program is not None
        return (":".join(route.program),)
    if isinstance(route, CakeFmhaDecodeRoute):
        route_name = {
            "decode_balanced_bf16": "decode_balanced_bf16_v1",
            "decode_balanced_fp16": "decode_balanced_fp16_v1",
            "decode_balanced_fp8": "decode_balanced_fp8_v1",
            "decode_balanced_bf16q": "decode_balanced_bf16q_v1",
            "decode_balanced_fp16q": "decode_balanced_fp16q_v1",
            "decode_balanced_bf16_hd64": "decode_balanced_bf16_hd64_v1",
            "decode_balanced_bf16_hd256": "decode_balanced_bf16_hd256_v1",
            "decode_native_bf16": "decode_native_bf16_v1_bece",
            "decode_native_bf16_hd256_smallm": "decode_native_bf16_hd256_smallm_v1",
            "decode_native_fp16_hd512": "decode_native_fp16_hd512_v1_66b1",
            "decode_native_fp16_nhd": "decode_native_fp16_nhd_v1_f32d",
            "decode_quant_bf16q": "decode_quantized_bf16q_9d8b_v1",
            "decode_quant_fp8": "decode_quantized_fp8_8e5b_v1",
            "decode_quant_nvfp4": "decode_quantized_nvfp4_8e5b_v1",
        }[route.component]
    else:
        route_name = {
            "context_bf16": "ctx_bf16_hnd_hd128_hgpack_03df_v2",
            "context_fp16_hd256": "ctx_fp16_nhd_hd256_stage16_v1",
            "context_fp8": "ctx_fp8_hnd_hd128_hgpack_48b5_v1",
            "context_fp8_hd256": "ctx_fp8_bf16_nhd_hd256_stage16_v1",
            "context_nvfp4": "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1",
        }[route.component]
    return _PRODUCT_ROUTE_COMPONENTS[route_name]


def cake_fmha_route_is_optimized(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute | None,
) -> bool:
    """Return whether ``route`` has a fully authenticated runnable adapter."""

    if route is None:
        return False
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component in _ALIGNED_COMPONENTS
    ):
        return _aligned_route_program(route) is not None
    if isinstance(route, CakeFmhaDecodeRoute) and _decode_route_runs_program(route):
        # A generated decode program needs no JIT member of the route.
        return _decode_program_record(route) is not None
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == "decode_native_bf16"
        and not _is_cake_fmha_decode_native_bf16_available(
            route.target,
            route.batch_size,
            route.q_len,
            route.num_q_heads,
            route.num_kv_heads,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
            return_lse=route.return_lse,
        )
    ):
        return False
    return all(
        component in _AUTHENTICATED_JIT_COMPONENTS
        for component in _route_components(route)
    )


def _tma_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether the paged-KV view can be represented by our TMA maps."""

    if tensor.stride(3) != 1:
        return False
    element_size = tensor.element_size()
    return all(
        stride > 0 and stride * element_size % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether packed HND NVFP4 KV is exactly TMA-encodable."""

    return tensor.stride(3) == 1 and all(
        stride > 0 and stride % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_scale_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether HND E4M3 block scales are exactly TMA-encodable."""

    return (
        tensor.stride(3) == 1
        and tensor.stride(2) == 8
        and tensor.stride(1) > 0
        and tensor.stride(0) > 0
        and tensor.stride(1) % 16 == 0
        and tensor.stride(0) % 16 == 0
    )


_SMALLM_MIN_SUPPORTED_SM_COUNT = 148  # B200; used only for non-CUDA capability replays


# CUDA device index -> its SM count (fixed for the process; a launch asks
# more than once).
_SM_COUNTS: dict[int, int] = {}


def _smallm_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return _SMALLM_MIN_SUPPORTED_SM_COUNT
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNTS.get(index)
    if count is None:
        count = _SM_COUNTS[index] = int(
            torch.cuda.get_device_properties(index).multi_processor_count
        )
    return count


# On-device load-balanced split-KV decode (Cake routes ``decode_balanced_bf16_v1``
# and ``decode_balanced_fp16_v1``).  The band is host metadata only: the caller's
# KV length bound (``max_seq_len``) and the number of (request, KV head) work
# tiles, per dtype and query-length class.  BF16: for ``q_len == 1`` the static
# grid-stride route keeps every tile whole below the band; the packed MTP tiles
# (``q_len`` 3..8) take the balanced kernel at every shape (1.6-11.8x faster
# than the static route on every measured shape, B200 + GB300).  FP16 (HND,
# page 16, GQA-8) has no other optimized Cake route -- the previous contract was
# the compat fallback -- so its band admits every shape.  The round-2 families
# (E4M3 cache with E4M3 / BF16 / FP16 query, BF16 head_dim 64, BF16 head_dim
# 256 outside the small-M route) likewise admit every shape: their incumbents
# were the quantized full-block routes, the compat kernel, or nothing.  Values
# mirror the Cake dispatcher's per-arch ``ROUTE_BAND`` and were set from paired
# same-input cold-L2 CUPTI measurements on B200 / GB300.
CAKE_FMHA_BALANCED_ROUTE_BAND: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
    "sm100a": {
        "bf16": {"q1": (32768, 9), "mtp": (1, 1)},
        "fp16": {"q1": (1, 1), "mtp": (1, 1)},
        "fp8": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16q": {"q1": (1, 1), "mtp": (1, 1)},
        "fp16q": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd64": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd256": {"q1": (1, 1), "mtp": (1, 1)},
    },  # (min KV length bound, min work tiles)
    "sm103a": {
        "bf16": {"q1": (32768, 9), "mtp": (1, 1)},
        "fp16": {"q1": (1, 1), "mtp": (1, 1)},
        "fp8": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16q": {"q1": (1, 1), "mtp": (1, 1)},
        "fp16q": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd64": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd256": {"q1": (1, 1), "mtp": (1, 1)},
    },
}
CAKE_FMHA_BALANCED_MAX_REQUESTS = 1024
CAKE_FMHA_BALANCED_MTP_Q_LENS = (3, 4, 5, 6, 7, 8)
# head_dim 256: every query row of a request is its own work tile (no packed
# MTP tile), uniform q_len 1..8.
CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN = 8
CAKE_FMHA_BALANCED_HD64_MAX_GROUP = 16
_BALANCED_MAX_BALANCE_FACTOR = 8
_BALANCED_ROW_PARTIAL_O_PER_SLOT = 8 * 128
_BALANCED_ROW_STATS_PER_SLOT = 16
_BALANCED_MTP_PARTIAL_O_PER_SLOT = 64 * 128
_BALANCED_MTP_STATS_PER_SLOT = 128
_BALANCED_HD64_PARTIAL_O_PER_SLOT = 16 * 64  # 16-row tile x head_dim 64
_BALANCED_HD64_STATS_PER_SLOT = 32  # max[16] then sum[16]
_BALANCED_HD256_PARTIAL_O_PER_SLOT = 8 * 256  # 8-row tile x head_dim 256
_BALANCED_HD256_STATS_PER_SLOT = 16  # max[8] then sum[8]


def _balanced_workspace_bounds(sm_count: int) -> tuple[int, int]:
    """``(max_split_items, max_split_tiles)`` of the persistent balanced grid."""

    return (
        2 * _BALANCED_MAX_BALANCE_FACTOR * sm_count,
        _BALANCED_MAX_BALANCE_FACTOR * sm_count,
    )


def _balanced_slot_words(q_len: int, head_dim: int) -> tuple[int, int]:
    """``(partial O floats, statistics floats)`` per split item of one balanced kernel family."""

    if head_dim == 64:
        return _BALANCED_HD64_PARTIAL_O_PER_SLOT, _BALANCED_HD64_STATS_PER_SLOT
    if head_dim == 256:
        return _BALANCED_HD256_PARTIAL_O_PER_SLOT, _BALANCED_HD256_STATS_PER_SLOT
    if head_dim != 128:
        raise ValueError(
            f"balanced decode serves head_dim 64, 128 or 256, got {head_dim}"
        )
    if q_len == 1:
        return _BALANCED_ROW_PARTIAL_O_PER_SLOT, _BALANCED_ROW_STATS_PER_SLOT
    return _BALANCED_MTP_PARTIAL_O_PER_SLOT, _BALANCED_MTP_STATS_PER_SLOT


def _balanced_counters_per_tile(q_len: int, head_dim: int) -> int:
    """Counter words per split tile: four for the packed head_dim-128 MTP tile, else one."""

    return 4 if (head_dim == 128 and q_len != 1) else 1


def _balanced_q_len_supported(q_len: int, head_dim: int) -> bool:
    """Query lengths a balanced kernel family serves (host metadata only)."""

    if head_dim == 128:
        return q_len == 1 or q_len in CAKE_FMHA_BALANCED_MTP_Q_LENS
    if head_dim == 256:
        return 1 <= q_len <= CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN
    return q_len == 1


def cake_fmha_balanced_counter_bytes(
    sm_count: int, q_len: int = 8, head_dim: int = 128
) -> int:
    """Zero-initialized counter bytes the balanced route needs.

    Tile counters (one word per split tile for the row kernels; four for the
    packed head_dim-128 MTP kernel: arrivals, the two reduce-queue words and the
    two-chunk published flag) are followed by the four 16-byte-aligned queue
    counters.  The kernel resets every counter it touched before it exits, so a
    buffer zeroed once at allocation can be reused across launches (the
    trtllm-gen ``multi_ctas_kv_counter_buffer`` contract).  The default
    arguments size the largest family.
    """

    _, max_split_tiles = _balanced_workspace_bounds(sm_count)
    tile_bytes = max_split_tiles * _balanced_counters_per_tile(q_len, head_dim) * 4
    return (tile_bytes + 15) // 16 * 16 + 4 * 4


def cake_fmha_balanced_workspace_bytes(
    sm_count: int, q_len: int, head_dim: int = 128
) -> int:
    """Partial-output/statistics bytes carved from ``workspace_buffer``.

    Per split item one FP32 partial tile (8 x 128 for the head_dim-128 row
    kernels and the E4M3-cache kernels, 64 x 128 for the packed MTP kernel,
    16 x 64 for head_dim 64, 8 x 256 for head_dim 256) and one statistics slot
    (row maxima then row sums), plus the reserved slot that receives the
    device planner's plan facts (chunk pairs, total items).
    """

    max_split_items, _ = _balanced_workspace_bounds(sm_count)
    partial_o, stats = _balanced_slot_words(q_len, head_dim)
    partial_o_bytes = max_split_items * partial_o * 4
    stats_offset = (partial_o_bytes + 255) // 256 * 256
    return stats_offset + (max_split_items + 1) * stats * 4


def _balanced_route_supported(
    target: CakeFmhaTarget,
    *,
    dtype: str,
    batch_size: int,
    q_len: int,
    num_kv_heads: int,
    max_seq_len: int,
    sm_count: int,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    head_dim: int = 128,
) -> bool:
    """Host-only admission of the on-device load-balanced decode routes.

    ``dtype`` names the band family (``bf16`` / ``fp16`` for head_dim 128,
    ``fp8`` / ``bf16q`` / ``fp16q`` for the E4M3 cache, ``bf16_hd64`` and
    ``bf16_hd256``); ``head_dim`` selects the workspace geometry.
    """

    if not _balanced_q_len_supported(q_len, head_dim):
        return False
    if batch_size > CAKE_FMHA_BALANCED_MAX_REQUESTS:
        return False
    arch_band = CAKE_FMHA_BALANCED_ROUTE_BAND.get(target, {}).get(dtype)
    if arch_band is None:
        return False
    min_kv_len, min_work_tiles = arch_band["q1" if q_len == 1 else "mtp"]
    if max_seq_len < min_kv_len or batch_size * num_kv_heads < min_work_tiles:
        return False
    if not workspace_buffer.is_contiguous():
        return False
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if workspace_bytes < cake_fmha_balanced_workspace_bytes(sm_count, q_len, head_dim):
        return False
    if counter_buffer is not None:
        counter_bytes = counter_buffer.numel() * counter_buffer.element_size()
        if counter_bytes < cake_fmha_balanced_counter_bytes(sm_count, q_len, head_dim):
            return False
    return True


# ---------------------------------------------------------------------------
# Aligned hd128 q1 decode selector: a Python mirror of the source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_tgv_bf16`` of the ``attention_decode_bf16_paged_static_g``
# family, source revision ``ea3a718663acd2818814c2444aa00537f6b604d6``:
# ``SEL_BAND_*``, ``_c1_balanced`` .. ``_c8_balanced``, ``aligned_cluster_size``).  The
# constants are copied verbatim (names and values) and the function keeps the
# source's signature and semantics; change the source first, then re-copy.
# ``tests/attention/test_cake_fmha_aligned.py`` holds the source selector
# test's expectations.  In this module ``None`` keeps the module's present
# route (``decode_native_bf16`` below the balanced band for BF16,
# ``decode_balanced_fp16`` for FP16); in the source it is the persistent
# balanced program.
# ---------------------------------------------------------------------------
TILE_Q = 8
QKV_DTYPES = ("bf16", "fp16")
# The bands of aligned_cluster_size, on tiles = batch_size * num_kv_heads and max_kv = max_seq_len_kv.
SEL_BAND_C8_TILES_MAX = 8
SEL_BAND_C4_TILES_MAX = 32
SEL_BAND_C2_TILES_MAX = 64
SEL_BAND_MID_TILES_MAX = 128
SEL_BAND_WIDE_TILES_MIN = 512
SEL_BAND_LONG_KV_MIN = 32768
SEL_BAND_LONG_KV_MAX = 65536
# Carve-outs: the shapes of a band on which the persistent balanced program
# measured faster than the band's aligned program on both targets (the rule: no
# regression at any probed point of the carve-out, >= 2 % faster at two thirds
# of them).  ``None`` keeps the aligned program on the whole band.  Lengths are
# ``max_kv``.
# c1: the balanced program for SEL_BAND_C1_BALANCED_KV_MIN <= max_kv <= SEL_BAND_C1_BALANCED_KV_MAX.
SEL_BAND_C1_BALANCED_KV_MAX: int | None = 2048
SEL_BAND_C1_BALANCED_KV_MIN: int | None = 512
# c2: the balanced program for lo <= max_kv <= hi; the wide form (t, lo, hi) adds tiles >= t with lo <= max_kv <= hi.
SEL_BAND_C2_BALANCED_KV: tuple[int, int] | None = (6144, 8192)
SEL_BAND_C2_BALANCED_WIDE: tuple[int, int, int] | None = (56, 4096, 6143)
# c4: the balanced program for max_kv >= the bound (the band's 32768 upper bound stays).
SEL_BAND_C4_BALANCED_KV_MIN: int | None = 16384
# c8: the balanced program for max_kv >= the bound.
SEL_BAND_C8_BALANCED_KV_MIN: int | None = 65536


def _c1_balanced(max_kv: int) -> bool:
    hi, lo = SEL_BAND_C1_BALANCED_KV_MAX, SEL_BAND_C1_BALANCED_KV_MIN
    return hi is not None and (lo or 0) <= max_kv <= hi


def _c2_balanced(tiles: int, max_kv: int) -> bool:
    win, wide = SEL_BAND_C2_BALANCED_KV, SEL_BAND_C2_BALANCED_WIDE
    if win is not None and win[0] <= max_kv <= win[1]:
        return True
    return wide is not None and tiles >= wide[0] and wide[1] <= max_kv <= wide[2]


def _c4_balanced(max_kv: int) -> bool:
    lo = SEL_BAND_C4_BALANCED_KV_MIN
    return lo is not None and max_kv >= lo


def _c8_balanced(max_kv: int) -> bool:
    lo = SEL_BAND_C8_BALANCED_KV_MIN
    return lo is not None and max_kv >= lo


def aligned_cluster_size(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len_kv: int,
    *,
    q_len: int,
    num_q_heads: int,
    dtype: str,
) -> int | None:
    """The cluster size (8, 4, 2 or 1) of the aligned program for one launch, or
    ``None`` for the persistent balanced program (this module's present route),
    from host metadata only.

    A shape that is not the aligned programs' (``q_len != 1``, Q heads not exactly
    ``TILE_Q`` per KV head, or a dtype not in ``QKV_DTYPES``) takes the persistent
    program.  Otherwise the bands on ``tiles = batch_size * num_kv_heads`` and
    ``max_kv = max_seq_len_kv`` select:

    * 8: ``tiles <= 8`` below ``SEL_BAND_C8_BALANCED_KV_MIN``;
    * 4: ``8 < tiles <= 32`` with ``max_kv <= 32768`` below
      ``SEL_BAND_C4_BALANCED_KV_MIN``, or ``tiles > 64`` with
      ``32768 <= max_kv <= 65536`` except ``tiles >= 512`` with ``max_kv > 32768``;
    * 2: ``32 < tiles <= 64`` with ``max_kv <= 32768`` outside the c2 carve-outs
      (``SEL_BAND_C2_BALANCED_KV``, ``SEL_BAND_C2_BALANCED_WIDE``), or
      ``tiles >= 512`` with ``32768 < max_kv <= 65536``;
    * ``None``: ``8 < tiles <= 64`` with ``max_kv > 32768``, ``tiles > 64`` with
      ``max_kv > 65536``, ``64 < tiles <= 128`` with ``max_kv < 32768``, and the
      carve-outs of the c8 / c4 / c2 / c1 bands where the balanced program
      measured faster;
    * 1: ``tiles > 128`` with ``max_kv < 32768`` outside
      ``SEL_BAND_C1_BALANCED_KV_MIN .. SEL_BAND_C1_BALANCED_KV_MAX``.

    The bands are complete and disjoint: exactly one holds, or the function
    raises.  Selection is a performance choice only: every program handles any
    length up to ``max_pages_per_seq * PAGE_SIZE``.
    """
    if not (
        q_len == 1 and num_q_heads == TILE_Q * num_kv_heads and dtype in QKV_DTYPES
    ):
        return None
    tiles = batch_size * num_kv_heads
    max_kv = max_seq_len_kv
    sel_c8 = tiles <= SEL_BAND_C8_TILES_MAX and not _c8_balanced(max_kv)
    sel_c4 = (
        SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C4_TILES_MAX
        and max_kv <= SEL_BAND_LONG_KV_MIN
        and not _c4_balanced(max_kv)
    ) or (
        tiles > SEL_BAND_C2_TILES_MAX
        and SEL_BAND_LONG_KV_MIN <= max_kv <= SEL_BAND_LONG_KV_MAX
        and not (tiles >= SEL_BAND_WIDE_TILES_MIN and max_kv > SEL_BAND_LONG_KV_MIN)
    )
    sel_c2 = (
        SEL_BAND_C4_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
        and max_kv <= SEL_BAND_LONG_KV_MIN
        and not _c2_balanced(tiles, max_kv)
    ) or (
        tiles >= SEL_BAND_WIDE_TILES_MIN
        and SEL_BAND_LONG_KV_MIN < max_kv <= SEL_BAND_LONG_KV_MAX
    )
    sel_p = (
        (tiles > SEL_BAND_C2_TILES_MAX and max_kv > SEL_BAND_LONG_KV_MAX)
        or (
            SEL_BAND_C2_TILES_MAX < tiles <= SEL_BAND_MID_TILES_MAX
            and max_kv < SEL_BAND_LONG_KV_MIN
        )
        or (
            SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
            and max_kv > SEL_BAND_LONG_KV_MIN
        )
        or (tiles <= SEL_BAND_C8_TILES_MAX and _c8_balanced(max_kv))
        or (
            SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C4_TILES_MAX
            and max_kv <= SEL_BAND_LONG_KV_MIN
            and _c4_balanced(max_kv)
        )
        or (
            SEL_BAND_C4_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
            and max_kv <= SEL_BAND_LONG_KV_MIN
            and _c2_balanced(tiles, max_kv)
        )
        or (
            tiles > SEL_BAND_MID_TILES_MAX
            and max_kv < SEL_BAND_LONG_KV_MIN
            and _c1_balanced(max_kv)
        )
    )
    sel_c1 = (
        tiles > SEL_BAND_MID_TILES_MAX
        and max_kv < SEL_BAND_LONG_KV_MIN
        and not _c1_balanced(max_kv)
    )
    hits = [
        size
        for size, hit in (
            (8, sel_c8),
            (4, sel_c4),
            (2, sel_c2),
            (None, sel_p),
            (1, sel_c1),
        )
        if hit
    ]
    if len(hits) != 1:
        raise ValueError(
            f"the selector bands must locate exactly one program for tiles={tiles} max_kv={max_kv}, got {hits}"
        )
    return hits[0]


def _smallm_tile_rows(packed_rows: int) -> int | None:
    """Instance tile (32 or 64 rows) holding ``q_len * group`` packed rows.

    Rows in (16, 32] use the 32-row instance and rows in (32, 64] the 64-row
    instance; rows above the packed count are tile padding the kernel never
    stores.  Fewer than 17 rows are left to other routes.
    """

    if packed_rows <= 16 or packed_rows > 64:
        return None
    return 32 if packed_rows <= 32 else 64


def _smallm_tile_rows_for(q_len: int, group: int) -> int | None:
    """Tile rows for ``q_len * group`` packed rows when ``group`` divides the tile."""

    n_rows = _smallm_tile_rows(q_len * group)
    if n_rows is None or group <= 0 or n_rows % group:
        return None
    return n_rows


def _smallm_num_split(
    *, tiles: int, sm_count: int, max_seq_len: int, n_rows: int
) -> int:
    """One-wave Split-KV factor mirroring the Cake route (0 when not resident)."""

    if tiles <= 0 or tiles > sm_count:
        return 0
    blocks = max(1, (max_seq_len + 127) // 128)
    return max(1, min(sm_count // tiles, max(blocks, n_rows), 256))


def _smallm_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    *,
    tiles: int,
    num_split: int,
    n_rows: int,
    lse: torch.Tensor | None,
) -> bool:
    """Mirror the small-M binding's partial/counter/LSE workspace layout."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False

    def align(value: int) -> int:
        return (value + 255) // 256 * 256

    slots = tiles * num_split
    cursor = align(slots * n_rows * 256 * 2)
    cursor = align(cursor + slots * n_rows * 4)
    cursor = align(cursor + tiles * 2 * 4)
    if lse is None:
        cursor += query.shape[0] * query.shape[1] * 4
    return workspace_buffer.numel() * workspace_buffer.element_size() >= cursor


def _decode_native_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    max_seq_len: int,
    pages_per_block: int,
    page_table_rows: int,
    lse: torch.Tensor | None,
) -> bool:
    """Mirror one native binding's resolved metadata/LSE workspace layout."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    even_kv_blocks = max(4, even_kv_blocks)
    required_pages = even_kv_blocks * pages_per_block
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(required_pages, source_pages)
    if pages_per_block == 4:
        padded_pages = (padded_pages + 3) // 4 * 4
    needs_page_padding = source_pages != padded_pages

    cursor = (batch_size * 4 + 15) // 16 * 16
    if needs_page_padding:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    if lse is None:
        cursor += query.shape[0] * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Mirror the BF16Q adapter's padding, scale, and partial workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128 * 2
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    cursor += batch_size * query.shape[1] * 128 * 4
    cursor += 2 * batch_size * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_scale_supported(
    scale: float | torch.Tensor, query: torch.Tensor
) -> bool:
    return not isinstance(scale, torch.Tensor) or (
        scale.device == query.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and scale.is_contiguous()
    )


def _pinned_noop_skip_softmax_supported(scale_factor: float | None) -> bool:
    """Accept the pinned matrix's numerically inert skip-softmax probe."""

    return scale_factor in (None, 0.0, 1e-30)


def _decode_quant_fp8_fixed_split(
    batch_size: int, num_kv_heads: int, max_seq_len: int, sm_count: int
) -> bool:
    """Whether the FP8 decode's host split plan is the fixed-plan member's 8 x 16.

    Mirrors ``ResolveSplitPlanHost`` of the JIT binding: host scalars only.
    """

    even_bucket = max(2, (max_seq_len + 127) // 128 + ((max_seq_len + 127) // 128) % 2)
    total_tiles = batch_size * num_kv_heads
    if total_tiles >= sm_count or even_bucket < 8:
        return False
    target = max(1, sm_count // total_tiles)
    blocks_per_split = (even_bucket + target - 1) // target
    blocks_per_split = max(4, blocks_per_split + blocks_per_split % 2)
    while blocks_per_split < even_bucket and even_bucket % blocks_per_split == 2:
        blocks_per_split += 2
    num_splits = (even_bucket + blocks_per_split - 1) // blocks_per_split
    return (num_splits, blocks_per_split) == (8, 16)


def _decode_quant_fp8_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    page_size: int,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Check a conservative upper bound for runtime split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    if not query.is_contiguous():
        cursor += batch_size * query.shape[1] * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Conservatively bound NVFP4 GQA padding and split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    even_kv_blocks = max(even_kv_blocks, 2)
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _manifest_optimized_route_accounting() -> tuple[int, int]:
    """Return registered/total optimized counts recorded by the manifest.

    This is a route-name drift check, not selector parity.  Exact selector
    coverage requires replaying the independent pinned capability corpus.
    """

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    registered = sum(route_counts.get(name, 0) for name in _PRODUCT_ROUTE_COMPONENTS)
    return registered, total


def _manifest_authenticated_route_accounting() -> tuple[int, int]:
    """Return runnable/total optimized counts recorded by the manifest."""

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    authenticated = sum(
        route_counts.get(route_name, 0)
        for route_name, components in _PRODUCT_ROUTE_COMPONENTS.items()
        if all(component in _AUTHENTICATED_JIT_COMPONENTS for component in components)
    )
    return authenticated, total


# CUDA device index -> its Cake FMHA target (the capability and the toolkit
# version are fixed for the process).
_CAKE_FMHA_TARGETS: dict[int, CakeFmhaTarget] = {}


def _cake_fmha_target(device: torch.device) -> CakeFmhaTarget:
    """Resolve the exact Blackwell cubin target without cross-arch fallback,
    once per CUDA device (a launch asks several times)."""

    index = device.index if device.type == "cuda" else None
    if index is not None:
        target = _CAKE_FMHA_TARGETS.get(index)
        if target is not None:
            return target
    target = _cake_fmha_target_of(get_compute_capability(device))
    if index is not None:
        _CAKE_FMHA_TARGETS[index] = target
    return target


@functools.lru_cache(maxsize=None)
def _cake_fmha_target_of(capability: tuple[int, int]) -> CakeFmhaTarget:
    """The target of a compute capability (the CUDA version is fixed per
    process, so the result is cached; a refusal raises every time)."""

    from .jit.cpp_ext import is_cuda_version_at_least

    if capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError("Cake FMHA on B200 requires CUDA 12.8 or newer")
        return "sm100a"
    if capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError("Cake FMHA on B300 requires CUDA 12.9 or newer")
        return "sm103a"
    raise RuntimeError(
        "Cake FMHA requires compute capability 10.0 (B200/GB200) or "
        f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
    )


def get_cake_fmha_module(device: torch.device):
    """Load the authenticated complete-domain compatibility module."""

    target = _cake_fmha_target(device)
    return load_cake_fmha_compat_module(target)


# ---------------------------------------------------------------------------
# Generated decode programs.
#
# The decode programs exported into the in-tree units
# ``csrc/cake_fmha_decode_<family>/`` (loaders
# ``flashinfer.jit.cake_fmha_decode_<family>``, one ``CakeProgram`` constant
# per program).  A route names its program as ``(family, identifier)`` in
# ``CakeFmhaDecodeRoute.program``; the selector looks it up from host scalars
# only (the launch shape, the scale form, ``max_seq_len`` and the SM count),
# never from device tensors, so a hit launches without a host synchronization
# and can be captured in a CUDA graph.  A launch outside the tables keeps the
# module's JIT route.

# FlashInfer target -> ``(family, *route key)`` -> program identifier.  A
# family's key is its generator's route key without the target (the export's
# parity lane holds this table to the generator's).
_DECODE_PROGRAM_ROUTES: dict[str, dict[tuple[Any, ...], str]] = {
    "sm100a": {},
    "sm103a": {},
}

# ``(family, identifier)`` -> ``(program name, launch)``: the delivered record
# of each program; a loader whose constant has another name fails closed at
# load.  ``launch`` names the grid (:func:`cake_fmha_decode_launch_grid`).
_DECODE_PROGRAMS: dict[tuple[str, str], tuple[str, str]] = {}

# ``(family, identifier)`` of a decode program -> ``(family, identifier)`` of
# the split-KV reducer launched right after it.
_DECODE_PROGRAM_REDUCERS: dict[tuple[str, str], tuple[str, str]] = {}

# family -> the host parameters of its programs' bindings, in call order,
# without the grid (every program of a family shares one kernel ABI, which a
# launch in ``_DECODE_LAUNCH_EXTRA_PARAMS`` extends).
_DECODE_PROGRAM_HOST_PARAMS: dict[str, tuple[str, ...]] = {}

# launch -> ``(the family parameter they follow, the parameters)`` a program
# of that launch adds to its family's: a stream-K program's split counters
# and split scratch follow the sinks.
_DECODE_LAUNCH_EXTRA_PARAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "streamk": ("sinks_ptr", ("split_counter", "split_ws")),
}


def _decode_program_host_params(family: str, launch: str) -> tuple[str, ...]:
    """The host parameters of a ``launch`` program of ``family``, in call
    order, without the grid."""

    params = _DECODE_PROGRAM_HOST_PARAMS[family]
    extra = _DECODE_LAUNCH_EXTRA_PARAMS.get(launch)
    if extra is None:
        return params
    after, names = extra
    cut = params.index(after) + 1
    return (*params[:cut], *names, *params[cut:])


# ``CakeFmhaTarget`` -> the exact ``CakeProgram`` target of the same device.
_DECODE_PROGRAM_TARGET: dict[str, str] = {"sm100a": "sm_100a", "sm103a": "sm_103a"}


def _decode_route_runs_program(route: CakeFmhaDecodeRoute) -> bool:
    """Whether ``route`` launches its generated program: a route naming one
    without LSE.  A ``return_lse`` route names the program of the same launch
    without LSE and runs its JIT member, which writes the caller's LSE."""

    return route.program is not None and not route.return_lse


def _decode_program_record(route: CakeFmhaDecodeRoute) -> tuple[str, str] | None:
    """``(program name, launch)`` the module pins for the route's program, or
    ``None`` when the route names no program or one this module does not
    route on its target."""

    if route.program is None:
        return None
    family, identifier = route.program
    routes = _DECODE_PROGRAM_ROUTES.get(route.target, {})
    if not any(
        key[0] == family and served == identifier for key, served in routes.items()
    ):
        return None
    return _DECODE_PROGRAMS.get((family, identifier))


def _decode_program_lookup(
    device: torch.device, family: str, key: tuple[Any, ...]
) -> tuple[str, str] | None:
    """``(family, identifier)`` of the program serving ``key`` on the device's
    target, or ``None``."""

    routes = _DECODE_PROGRAM_ROUTES[_cake_fmha_target(device)]
    identifier = routes.get((family, *key))
    if identifier is None or (family, identifier) not in _DECODE_PROGRAMS:
        return None
    return family, identifier


def _decode_program_scale_tensor(scale: Any, query: torch.Tensor) -> bool:
    """A device scale a program reads in the kernel: one contiguous FP32
    element on the query's device, 4-byte aligned."""

    return (
        isinstance(scale, torch.Tensor)
        and scale.device == query.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and scale.is_contiguous()
        and scale.data_ptr() % 4 == 0
    )


def _decode_program_scale_form(
    bmm1_scale: Any, bmm2_scale: Any, query: torch.Tensor
) -> str | None:
    """``"device"`` when both scales are device scalars a program reads,
    ``"host"`` when both are host numbers, ``None`` for a mixed or
    unsupported pair (it keeps the module's JIT route)."""

    device_bmm1 = _decode_program_scale_tensor(bmm1_scale, query)
    if device_bmm1 and _decode_program_scale_tensor(bmm2_scale, query):
        return "device"
    if isinstance(bmm1_scale, (int, float)) and isinstance(bmm2_scale, (int, float)):
        return "host"
    return None


def _decode_program_quant_plan(
    batch_size: int, num_kv_heads: int, max_seq_len: int, sm_count: int
) -> tuple[int, int]:
    """``(num_splits, blocks_per_split)`` the quantized decode plans from host
    scalars (``ResolveSplitPlanHost`` of the JIT binding)."""

    blocks = (max_seq_len + 127) // 128
    even = max(2, blocks + blocks % 2)
    tiles = batch_size * num_kv_heads
    if tiles >= sm_count or even < 8:
        return 1, even
    target = max(1, sm_count // tiles)
    per_split = max(4, (even + target - 1) // target)
    per_split += per_split % 2
    while per_split < even and even % per_split == 2:
        per_split += 2
    return (even + per_split - 1) // per_split, per_split


def _decode_program_plan_label(plan: tuple[int, int]) -> str:
    """The split plan label of a route key: ``split1`` for one split,
    ``split<N>x<blocks>`` otherwise."""

    num_splits, per_split = plan
    return "split1" if num_splits == 1 else f"split{num_splits}x{per_split}"


def cake_fmha_decode_launch_grid(
    launch: str,
    program,
    *,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_splits: int,
    sm_count: int,
) -> tuple[int, int, int]:
    """The launch grid of one generated decode program.

    ``launch`` is the program's pinned launch (``_DECODE_PROGRAMS``):

    * ``persistent``: ``(min(sm_count, tiles), 1, 1)`` CTAs striding over the
      ``batch * q_len * num_kv_heads`` tiles;
    * ``cluster``: one cluster of ``program.cluster[0]`` CTAs per resident
      tile, ``(cluster * min(sm_count, tiles), 1, 1)``;
    * ``bhq``: one CTA per (query, KV head, request), ``(q_len, num_kv_heads,
      batch)``;
    * ``quant``: one CTA per (request, KV head, split), ``(batch *
      num_kv_heads * num_splits, 1, 1)``;
    * ``quant128``: the same tiles over at most 128 persistent CTAs;
    * ``reduce``: one warp per (request, query head), ``(batch * num_q_heads,
      1, 1)``;
    * ``split``: one resident wave of ``batch * num_kv_heads * num_splits``
      CTAs (the clusters, if any, along x);
    * ``streamk``: one persistent CTA per SM in whole groups of ``q_len *
      num_kv_heads`` CTAs (:func:`_decode_streamk_groups`); a launch the
      kernel does not admit (``batch < 2 * groups``) raises.
    """

    batch_size, q_len = int(batch_size), int(q_len)
    num_kv_heads, num_splits = int(num_kv_heads), int(num_splits)
    tiles = batch_size * q_len * num_kv_heads
    if launch == "persistent":
        return min(int(sm_count), tiles), 1, 1
    if launch == "cluster":
        if int(sm_count) < tiles:
            raise RuntimeError(
                f"a cluster decode program gives each of its {tiles} tiles its own "
                f"cluster; {int(sm_count)} SMs cannot"
            )
        return int(program.cluster[0]) * tiles, 1, 1
    if launch == "bhq":
        return q_len, num_kv_heads, batch_size
    if launch == "quant":
        return batch_size * num_kv_heads * num_splits, 1, 1
    if launch == "quant128":
        return min(batch_size * num_kv_heads * num_splits, 128), 1, 1
    if launch == "reduce":
        return batch_size * int(num_q_heads), 1, 1
    if launch == "split":
        return batch_size * num_kv_heads * num_splits, 1, 1
    if launch == "streamk":
        groups = _decode_streamk_groups(batch_size, q_len, num_kv_heads, sm_count)
        if groups == 0:
            raise RuntimeError(
                "a stream-K decode program admits batch_size >= 2 x its CTA groups "
                f"of {q_len * num_kv_heads} CTAs; {batch_size} requests on "
                f"{int(sm_count)} SMs"
            )
        return groups * q_len * num_kv_heads, 1, 1
    raise ValueError(f"unknown Cake FMHA decode program launch {launch!r}")


# FP32 words of one part of a stream-K split tile: 8 x 128 partial O, 8 row
# maxima and 8 row sums, padded to 128-byte lines.
_DECODE_STREAMK_PART_WORDS = 1056


def _decode_streamk_groups(
    batch_size: int, q_len: int, num_kv_heads: int, sm_count: int
) -> int:
    """The CTA groups of a stream-K launch, or 0 when the kernel does not
    admit it.  A group is ``q_len * num_kv_heads`` CTAs (one per tile of a
    request) and the grid holds as many whole groups as the SMs; the kernel
    cuts the requests' KV block pairs evenly over the groups, so a request
    has at most two parts only with ``batch >= 2 * groups``."""

    unit = int(q_len) * int(num_kv_heads)
    if unit <= 0:
        return 0
    groups = min(int(sm_count), int(batch_size) * unit) // unit
    return groups if groups >= 1 and int(batch_size) >= 2 * groups else 0


def _decode_streamk_layout(
    batch_size: int, rows: int, grid: int
) -> tuple[int, int, int]:
    """``(split scratch offset, counter offset, end)`` of a stream-K launch's
    workspace, after the BF16 programs' LSE scratch: two split-tile parts per
    CTA, then one u32 counter per CTA (used when the caller's counter buffer
    cannot hold them)."""

    def align(value: int) -> int:
        return (value + 255) // 256 * 256

    lse_offset = (int(batch_size) * 4 + 15) // 16 * 16
    split_ws = align(lse_offset + int(rows) * 4)
    counters = align(split_ws + int(grid) * 2 * _DECODE_STREAMK_PART_WORDS * 4)
    return split_ws, counters, counters + int(grid) * 4


@dataclass(frozen=True)
class CakeFmhaDecodeProgramModule:
    """The loaded generated program of one decode program route: the route,
    the family and identifier, the pinned launch, the ``CakeProgram`` record,
    its exact target and the binding's ``run`` entry; ``reducer`` is the
    ``(family, identifier, launch, program, entry)`` of the split-KV reducer
    launched right after it (``None`` when it launches none)."""

    route: CakeFmhaDecodeRoute
    family: str
    identifier: str
    launch: str
    program: Any
    target: str
    entry: Any
    reducer: tuple[str, str, str, Any, Any] | None = None
    # The family's launch arguments derived from the caller's workspace (see
    # _decode_smallm_program_values): built at the first launch on a
    # workspace, kept until a launch on another one; not part of the module.
    launch_args: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


def _decode_program_constant(family: str, identifier: str, target: str):
    """The checked ``CakeProgram`` constant ``identifier`` of the family's
    generated loader on ``target``: the name this module pins, the target
    shipped and the binding's host parameters the ones this module passes."""

    import importlib

    from .jit.cake_program import GRID_ARGS, CakeProgram

    loader_name = f"cake_fmha_{family}"
    loader = importlib.import_module(f".jit.{loader_name}", __package__)
    program = getattr(loader, identifier, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.{loader_name} publishes no CakeProgram {identifier!r}"
        )
    pinned = _DECODE_PROGRAMS.get((family, identifier))
    if pinned is None or program.name != pinned[0]:
        raise RuntimeError(
            f"flashinfer.jit.{loader_name}.{identifier} is {program.name}; "
            f"this module pins {None if pinned is None else pinned[0]}"
        )
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are "
            f"{list(program.targets)}"
        )
    launch = pinned[1]
    expected = (*_decode_program_host_params(family, launch), *GRID_ARGS)
    if tuple(program.args) != expected:
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {list(expected)}"
        )
    clustered = tuple(program.cluster) != (1, 1, 1)
    if launch != "split" and clustered != (launch == "cluster"):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; its "
            f"launch is {launch!r}"
        )
    return program


# route -> its loaded program (a launch loads its route's program once).
_DECODE_PROGRAM_MODULES: dict[CakeFmhaDecodeRoute, CakeFmhaDecodeProgramModule] = {}


def _load_cake_fmha_decode_program(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaDecodeProgramModule:
    """Build and load the generated program of ``route`` (and its reducer) on
    the route's target, once per route.  The loaders are imported here,
    lazily, so route selection never needs them."""

    module = _DECODE_PROGRAM_MODULES.get(route)
    if module is not None:
        return module
    record = _decode_program_record(route)
    if record is None:
        raise RuntimeError(
            f"Cake FMHA decode route names no program: {route.program!r}"
        )
    family, identifier = route.program
    target = _DECODE_PROGRAM_TARGET[route.target]
    program = _decode_program_constant(family, identifier, target)
    reducer = None
    reducer_key = _DECODE_PROGRAM_REDUCERS.get((family, identifier))
    if reducer_key is not None:
        reducer_program = _decode_program_constant(*reducer_key, target)
        reducer = (
            *reducer_key,
            _DECODE_PROGRAMS[reducer_key][1],
            reducer_program,
            reducer_program.entry(target),
        )
    module = CakeFmhaDecodeProgramModule(
        route=route,
        family=family,
        identifier=identifier,
        launch=record[1],
        program=program,
        target=target,
        entry=program.entry(target),
        reducer=reducer,
    )
    _DECODE_PROGRAM_MODULES[route] = module
    return module


def _decode_program_grid(
    module: CakeFmhaDecodeProgramModule,
    program,
    launch: str,
    *,
    sm_count: int,
    num_splits: int = 1,
) -> dict[str, int]:
    """The grid arguments of one launch of ``module``'s route."""

    route = module.route
    grid = cake_fmha_decode_launch_grid(
        launch,
        program,
        batch_size=route.batch_size,
        q_len=route.q_len,
        num_q_heads=route.num_q_heads,
        num_kv_heads=route.num_kv_heads,
        num_splits=num_splits,
        sm_count=sm_count,
    )
    return {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}


def _decode_program_bytes(workspace_buffer: torch.Tensor) -> torch.Tensor:
    """The workspace's flat byte view (the buffer itself when it is one)."""

    if workspace_buffer.dtype == torch.uint8 and workspace_buffer.dim() == 1:
        return workspace_buffer
    return workspace_buffer.view(-1).view(torch.uint8)


def _decode_program_bytes_of(buffer: torch.Tensor) -> int:
    """The byte size of a contiguous workspace or counter buffer."""

    return buffer.numel() * buffer.element_size()


def _decode_program_workspace(
    flat: torch.Tensor, offset: int, nbytes: int, dtype: torch.dtype
) -> torch.Tensor:
    """A ``dtype`` view of ``nbytes`` bytes at ``offset`` of the workspace's
    flat byte view ``flat`` (:func:`_decode_program_bytes`, taken once per
    launch)."""

    if offset + nbytes > flat.numel():
        raise RuntimeError(
            f"Cake FMHA decode program workspace holds {flat.numel()} bytes; "
            f"the launch needs {offset + nbytes}"
        )
    return flat[offset : offset + nbytes].view(dtype)


# family -> the builder of its launches (see
# :func:`cake_fmha_decode_program_launches`).
_DECODE_PROGRAM_VALUES: dict[str, Any] = {}


def cake_fmha_decode_program_launches(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    lse: torch.Tensor | None,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    sm_count: int,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """The launches of one program route, in stream order: ``(program, entry,
    values)`` per launch, ``values`` naming every host parameter of the
    binding and the grid.  Nothing here reads a device tensor: the values are
    views of the caller's tensors and the workspace, and host scalars."""

    route = module.route
    if (
        query.shape[0] != route.batch_size * route.q_len
        or query.shape[1] != route.num_q_heads
        or key_cache.shape[1] != route.num_kv_heads
        or block_tables.shape[0] != route.batch_size
    ):
        raise RuntimeError(
            "Cake FMHA decode program launch does not match its selected route"
        )
    builder = _DECODE_PROGRAM_VALUES[module.family]
    return builder(
        module,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        out=out,
        workspace_buffer=workspace_buffer,
        counter_buffer=counter_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=int(max_seq_len),
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        lse=lse,
        key_block_scales=key_block_scales,
        value_block_scales=value_block_scales,
        sm_count=int(sm_count),
    )


def _run_cake_fmha_decode_program(
    module: CakeFmhaDecodeProgramModule, **kwargs: Any
) -> None:
    """Launch one decode program route on the current torch stream (see
    :func:`cake_fmha_decode_program_launches`)."""

    query = kwargs["query"]
    launches = cake_fmha_decode_program_launches(module, **kwargs)
    if _ffi_call_takes_the_torch_stream():
        # each launch's first tensor argument is a torch tensor
        for program, entry, values in launches:
            entry(*[values[name] for name in program.args])
        return
    with _torch_stream_context(query.device):
        for program, entry, values in launches:
            entry(*[values[name] for name in program.args])


@functools.cache
def _decode_program_toolkit_at_least(version: str) -> bool:
    """Whether the CUDA toolkit that builds the programs is ``version`` or newer
    (asked once per process)."""

    from .jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


# --- Quantized decode programs (E4M3 query and output) ----------------------
#
# ``cake_fmha_decode_nvfp4``: NVFP4 K / V pages.  A program serves both scale
# forms: host scales travel as kernel scalars (null scale pointers), device
# scales as FP32 pointers (bmm1 in log2 units).  The key is ``(batch,
# num_q_heads, num_kv_heads, page_size, split plan)``.

_DECODE_QUANT_PARAMS = (
    "O",
    "page_table",
    "seq_lens_kv",
    "bmm1_scale_ptr",
    "bmm2_scale_ptr",
    "partial_O",
    "partial_max",
    "partial_sum",
    "pt_batch_stride",
    "pt_v_offset",
    "bmm1_is_log2",
    "num_splits",
    "blocks_per_split",
    "bmm1_scalar",
    "bmm2_scalar",
)
_DECODE_PROGRAM_HOST_PARAMS["decode_nvfp4"] = (
    "Qt",
    "Kp",
    "Vp",
    "Ksf",
    "Vsf",
    *_DECODE_QUANT_PARAMS,
)
_DECODE_PROGRAM_ROUTES["sm100a"].update(
    {
        ("decode_nvfp4", 128, 32, 4, 16, "split1"): "DEC_NVFP4_B128",
    }
)
_DECODE_PROGRAM_ROUTES["sm103a"].update(
    {
        ("decode_nvfp4", 128, 32, 4, 16, "split1"): "DEC_NVFP4_B128",
    }
)
_DECODE_PROGRAMS.update(
    {
        ("decode_nvfp4", "DEC_NVFP4_B128"): (
            "cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4",
            "quant128",
        ),
    }
)


def _decode_quant_program_workspace_bytes(
    *, host_scales: bool, batch_size: int, num_q_heads: int, num_splits: int
) -> int:
    """The workspace of a quantized decode program: the host scalar slots
    (reserved; the scalars travel as kernel arguments), then the FP32 split
    partials ``O`` / max / sum, 16-byte aligned (the binding's layout without
    the padded query and page table, which a hit never needs)."""

    rows = batch_size * num_q_heads * num_splits
    scalar_slots = 16 if host_scales else 0  # two 4-byte slots, aligned to 16
    return scalar_slots + rows * 128 * 4 + 2 * rows * 4


def _decode_quant_program_select(
    device: torch.device,
    family: str,
    *,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    uses_shared_paged_kv_idx: bool,
    enable_pdl: bool | None = None,
) -> tuple[tuple[str, str], bool, int] | None:
    """``(program, device_scales, num_splits)`` of a quantized decode launch
    the family's programs serve, or ``None``.

    A hit takes a contiguous query of eight query heads per KV head, a
    contiguous int32 shared page table at least as wide as the kernel reads
    (the evened 128-token block count of ``max_seq_len``, at least two, in
    pages), int32 lengths, one scale form and a 16-byte aligned workspace
    holding the split partials.  The plan comes from host scalars only.
    ``enable_pdl`` is part of the key of a family whose multi-split programs
    hand their partials to a reducer launched with programmatic dependent
    launch (``None`` for a family without such a key).
    """

    num_q_heads = int(query.shape[1])
    if not (
        uses_shared_paged_kv_idx
        and block_tables.ndim == 2
        and block_tables.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and query.is_contiguous()
        and num_q_heads == 8 * num_kv_heads
        and page_size in (16, 32)
    ):
        return None
    form = _decode_program_scale_form(bmm1_scale, bmm2_scale, query)
    if form is None:
        return None
    blocks = (max_seq_len + 127) // 128
    even = max(2, blocks + blocks % 2)
    if int(block_tables.shape[-1]) < even * (128 // page_size):
        return None
    sm_count = _smallm_sm_count(device)
    plan = _decode_program_quant_plan(batch_size, num_kv_heads, max_seq_len, sm_count)
    required = _decode_quant_program_workspace_bytes(
        host_scales=form == "host",
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_splits=plan[0],
    )
    if (
        not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 16
        or workspace_buffer.numel() * workspace_buffer.element_size() < required
    ):
        return None
    key: tuple[Any, ...] = (
        batch_size,
        num_q_heads,
        num_kv_heads,
        page_size,
        _decode_program_plan_label(plan),
    )
    if enable_pdl is not None:
        key += (plan[0] > 1 and enable_pdl is True,)
    program = _decode_program_lookup(device, family, key)
    if program is None:
        return None
    return program, form == "device", plan[0]


def _decode_quant_program_values(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    sm_count: int,
    **unused: Any,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """The launches of a quantized decode program: the decode program, then
    its split-KV reducer if it has one.  The kernels read the page table in
    place (row stride its width) and the per-request lengths on the device;
    the split partials live in the workspace."""

    route = module.route
    num_splits, blocks_per_split = _decode_program_quant_plan(
        route.batch_size, route.num_kv_heads, max_seq_len, sm_count
    )
    if num_splits != route.num_split:
        raise RuntimeError(
            f"Cake FMHA decode program {module.identifier} was selected for "
            f"{route.num_split} splits; the launch plans {num_splits}"
        )
    device_scales = route.device_scales
    rows = route.batch_size * route.num_q_heads * num_splits
    # FP32 partial O (128 per row), then the row maxima, then the row sums.
    partials = _decode_program_workspace(
        _decode_program_bytes(workspace_buffer),
        0 if device_scales else 16,
        rows * 130 * 4,
        torch.float32,
    )
    partial_o = partials[: rows * 128]
    partial_max = partials[rows * 128 : rows * 129]
    partial_sum = partials[rows * 129 :]
    if device_scales:
        if not isinstance(bmm1_scale, torch.Tensor) or not isinstance(
            bmm2_scale, torch.Tensor
        ):
            raise RuntimeError(
                f"Cake FMHA decode program {module.identifier} was selected for "
                "device scales; the launch passes host scales"
            )
        bmm1_ptr, bmm2_ptr = bmm1_scale.reshape(1), bmm2_scale.reshape(1)
        bmm1_scalar = bmm2_scalar = 1.0
    else:
        bmm1_ptr = bmm2_ptr = None
        bmm1_scalar, bmm2_scalar = float(bmm1_scale), float(bmm2_scale)
    out_bytes = out.view(torch.uint8)
    values: dict[str, Any] = {
        "Qt": query.view(torch.uint8).view(-1, 128),
        "O": out_bytes,
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "bmm1_scale_ptr": bmm1_ptr,
        "bmm2_scale_ptr": bmm2_ptr,
        "partial_O": partial_o,
        "partial_max": partial_max,
        "partial_sum": partial_sum,
        "pt_batch_stride": int(block_tables.shape[-1]),
        "pt_v_offset": 0,
        "bmm1_is_log2": 1 if device_scales else 0,
        "num_splits": num_splits,
        "blocks_per_split": blocks_per_split,
        "bmm1_scalar": bmm1_scalar,
        "bmm2_scalar": bmm2_scalar,
    }
    if module.family == "decode_nvfp4":
        values["Kp"] = key_cache
        values["Vp"] = value_cache
        values["Ksf"] = key_block_scales.view(torch.uint8)
        values["Vsf"] = value_block_scales.view(torch.uint8)
    else:
        values["K"] = key_cache.view(torch.uint8)
        values["V"] = value_cache.view(torch.uint8)
    grid = _decode_program_grid(
        module,
        module.program,
        module.launch,
        sm_count=sm_count,
        num_splits=num_splits,
    )
    values.update(grid)
    launches = [(module.program, module.entry, values)]
    if module.reducer is not None:
        _, _, launch, program, entry = module.reducer
        reduce_values: dict[str, Any] = {
            "partial_O": partial_o,
            "partial_max": partial_max,
            "partial_sum": partial_sum,
            "O": out_bytes,
            "bmm2_scale_ptr": bmm2_ptr,
            "num_split": num_splits,
            "bmm2_scalar": bmm2_scalar,
        }
        reduce_values.update(
            _decode_program_grid(
                module, program, launch, sm_count=sm_count, num_splits=num_splits
            )
        )
        launches.append((program, entry, reduce_values))
    return launches


_DECODE_PROGRAM_VALUES["decode_nvfp4"] = _decode_quant_program_values


# ``cake_fmha_decode_fp8``: E4M3 K / V pages (one stacked or two pools).  The
# key is ``(batch, num_q_heads, num_kv_heads, page_size, split plan,
# enable_pdl)``: a one-split plan runs no reducer (``enable_pdl`` False in its
# key); a program built for the fixed plan of eight splits of sixteen blocks
# serves a call that asks for ``enable_pdl``, and the reducer of
# ``cake_fmha_decode_fp8_pdl`` follows it with programmatic dependent launch.

_DECODE_PROGRAM_HOST_PARAMS["decode_fp8"] = ("Qt", "K", "V", *_DECODE_QUANT_PARAMS)
_DECODE_PROGRAM_HOST_PARAMS["decode_fp8_pdl"] = (
    "partial_O",
    "partial_max",
    "partial_sum",
    "O",
    "bmm2_scale_ptr",
    "num_split",
    "bmm2_scalar",
)
_DECODE_PROGRAM_ROUTES["sm100a"].update(
    {
        ("decode_fp8", 256, 32, 4, 16, "split1", False): "B256_SM100A",
        ("decode_fp8", 4, 32, 4, 32, "split8x16", True): "B4_S8X16_SM100A",
        ("decode_fp8_pdl", 8): "REDUCE_S8",
    }
)
_DECODE_PROGRAM_ROUTES["sm103a"].update(
    {
        ("decode_fp8", 256, 32, 4, 16, "split1", False): "B256_SM103A",
        ("decode_fp8", 4, 32, 4, 32, "split8x16", True): "B4_S8X16_SM103A",
        ("decode_fp8_pdl", 8): "REDUCE_S8",
    }
)
_DECODE_PROGRAMS.update(
    {
        ("decode_fp8", "B256_SM100A"): (
            "cake_fmha_decode_fp8_768aa719bcc0c8dc6de5",
            "quant",
        ),
        ("decode_fp8", "B256_SM103A"): (
            "cake_fmha_decode_fp8_2bf8c20db84aa7b45230",
            "quant",
        ),
        ("decode_fp8", "B4_S8X16_SM100A"): (
            "cake_fmha_decode_fp8_56920e5612bc123d66c9",
            "quant",
        ),
        ("decode_fp8", "B4_S8X16_SM103A"): (
            "cake_fmha_decode_fp8_4a15e1e5d592f3ec8833",
            "quant",
        ),
        ("decode_fp8_pdl", "REDUCE_S8"): (
            "cake_fmha_decode_fp8_pdl_8cb30ca58dc96a96e10c",
            "reduce",
        ),
    }
)
_DECODE_PROGRAM_REDUCERS.update(
    {
        ("decode_fp8", "B4_S8X16_SM100A"): ("decode_fp8_pdl", "REDUCE_S8"),
        ("decode_fp8", "B4_S8X16_SM103A"): ("decode_fp8_pdl", "REDUCE_S8"),
    }
)
_DECODE_PROGRAM_VALUES["decode_fp8"] = _decode_quant_program_values


# --- Small-M speculative decode programs (BF16, head dim 256) ----------------
#
# ``cake_fmha_decode_smallm``: the (query token, query head) pairs of one KV
# head packed into a 32- or 64-row tile, the KV split over one resident wave
# of CTAs whose split count is a build constant.  The key is ``(n_rows,
# page_size, q_len, group, num_split, cluster)``: the module's split rule
# (``_smallm_num_split``), except that one tile whose KV is exactly nine
# 128-token blocks takes the nine-CTA cluster program.

_DECODE_PROGRAM_HOST_PARAMS["decode_smallm"] = (
    "Q",
    "K",
    "V",
    "partial_O_ptr",
    "partial_LSE_ptr",
    "O_ptr",
    "LSE_ptr",
    "counters",
    "page_table",
    "seq_lens",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
)
_DECODE_PROGRAM_ROUTES["sm100a"].update(
    {
        ("decode_smallm", 64, 64, 8, 8, 148, 0): "N64_P64_Q8_S148",
        ("decode_smallm", 64, 64, 7, 8, 148, 0): "N64_P64_Q7_S148",
        ("decode_smallm", 64, 16, 8, 8, 148, 0): "N64_P16_Q8_S148",
        ("decode_smallm", 32, 64, 4, 8, 148, 0): "N32_P64_Q4_S148",
        ("decode_smallm", 32, 64, 3, 8, 148, 0): "N32_P64_Q3_S148",
        ("decode_smallm", 32, 64, 4, 8, 74, 0): "N32_P64_Q4_S74",
        ("decode_smallm", 64, 64, 8, 8, 65, 0): "N64_P64_Q8_S65",
        ("decode_smallm", 64, 64, 7, 8, 65, 0): "N64_P64_Q7_S65",
        ("decode_smallm", 64, 64, 8, 8, 9, 9): "N64_P64_Q8_C9",
    }
)
_DECODE_PROGRAM_ROUTES["sm103a"].update(
    {
        ("decode_smallm", 64, 64, 8, 8, 152, 0): "N64_P64_Q8_S152",
        ("decode_smallm", 64, 64, 7, 8, 152, 0): "N64_P64_Q7_S152",
        ("decode_smallm", 64, 16, 8, 8, 152, 0): "N64_P16_Q8_S152",
        ("decode_smallm", 32, 64, 4, 8, 152, 0): "N32_P64_Q4_S152",
        ("decode_smallm", 32, 64, 3, 8, 152, 0): "N32_P64_Q3_S152",
        ("decode_smallm", 32, 64, 4, 8, 76, 0): "N32_P64_Q4_S76",
        ("decode_smallm", 64, 64, 8, 8, 65, 0): "N64_P64_Q8_S65",
        ("decode_smallm", 64, 64, 7, 8, 65, 0): "N64_P64_Q7_S65",
        ("decode_smallm", 64, 64, 8, 8, 9, 9): "N64_P64_Q8_C9",
    }
)
_DECODE_PROGRAMS.update(
    {
        ("decode_smallm", "N64_P64_Q8_S148"): (
            "cake_fmha_decode_smallm_21ea26612a009f241410",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q7_S148"): (
            "cake_fmha_decode_smallm_87d9cf4f8515047dcbaa",
            "split",
        ),
        ("decode_smallm", "N64_P16_Q8_S148"): (
            "cake_fmha_decode_smallm_a99e225068e33cf99451",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q4_S148"): (
            "cake_fmha_decode_smallm_87d5d62578d02874cada",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q3_S148"): (
            "cake_fmha_decode_smallm_0327fa5913427e139db7",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q4_S74"): (
            "cake_fmha_decode_smallm_d9854bf8ac213ba04ea9",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q8_S152"): (
            "cake_fmha_decode_smallm_60474a8cc8bc55e56484",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q7_S152"): (
            "cake_fmha_decode_smallm_5ea2416d7e4599b34e4b",
            "split",
        ),
        ("decode_smallm", "N64_P16_Q8_S152"): (
            "cake_fmha_decode_smallm_4286f312b61072e4b222",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q4_S152"): (
            "cake_fmha_decode_smallm_ba23b5c11e506e04ac2a",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q3_S152"): (
            "cake_fmha_decode_smallm_3cbe2ae98d241be2267e",
            "split",
        ),
        ("decode_smallm", "N32_P64_Q4_S76"): (
            "cake_fmha_decode_smallm_2e16f0e25463fa51d7d6",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q8_S65"): (
            "cake_fmha_decode_smallm_ab6579d6278a8d29b0a3",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q7_S65"): (
            "cake_fmha_decode_smallm_cdeb96c96f466fa3f07d",
            "split",
        ),
        ("decode_smallm", "N64_P64_Q8_C9"): (
            "cake_fmha_decode_smallm_f8167380e9cf9cf665b5",
            "split",
        ),
    }
)


def _decode_smallm_program_select(
    device: torch.device,
    *,
    n_rows: int,
    page_size: int,
    q_len: int,
    group: int,
    tiles: int,
    max_seq_len: int,
    num_split: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[tuple[str, str], int] | None:
    """``(program, num_split)`` of a small-M launch the generated programs
    serve, or ``None``: the module's split rule, except one 64-row tile of
    eight query tokens over 64-token pages whose KV is exactly nine 128-token
    blocks, which the nine-CTA cluster program merges in distributed shared
    memory."""

    if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return None
    cluster = 0
    # The cluster program's precondition: one tile (one request, one KV head)
    # whose KV fits its nine 128-token blocks.
    if (n_rows, page_size, q_len, group, tiles) == (64, 64, 8, 8, 1) and (
        8 * 128 < max_seq_len <= 9 * 128
    ):
        num_split = cluster = 9
    key = (n_rows, page_size, q_len, group, num_split, cluster)
    program = _decode_program_lookup(device, "decode_smallm", key)
    return None if program is None else (program, num_split)


def _decode_smallm_program_values(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float,
    lse: torch.Tensor | None,
    sm_count: int,
    **unused: Any,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """The one launch of a small-M program: the workspace layout of the
    module's JIT route (BF16 partial ``O`` and FP32 partial LSE slots, two
    merge tickets per tile, the LSE when the caller passes none).  The tickets
    live in the caller's zeroed counter buffer when it is large enough (the
    kernel's last CTA of a tile resets them), else in the workspace, zeroed
    on the stream first.

    Those workspace arguments and the grid depend only on the buffers'
    addresses and sizes, so the module keeps them from its last launch of
    each LSE form (on a CUDA device as FFI tensors, converted once) and a
    launch on the same buffers reads their addresses only; the caller's other
    tensors pass as given.  A cluster program merges one tile whose KV fits its splits'
    128-token blocks and refuses any other launch."""

    route = module.route
    if module.program.cluster[0] > 1 and (
        route.batch_size * route.num_kv_heads != 1
        or max_seq_len > route.num_split * 128
    ):
        raise RuntimeError(
            f"Cake FMHA decode program {module.identifier} merges one tile of at "
            f"most {route.num_split * 128} KV tokens; the launch has "
            f"{route.batch_size * route.num_kv_heads} tiles and max_seq_len "
            f"{max_seq_len}"
        )
    counter_key = None
    if counter_buffer is not None:
        counter_key = (
            counter_buffer.data_ptr(),
            counter_buffer.numel() * counter_buffer.element_size(),
            counter_buffer.is_contiguous(),
        )
    key = (
        workspace_buffer.data_ptr(),
        workspace_buffer.numel() * workspace_buffer.element_size(),
        counter_key,
        sm_count,
    )
    slot = "smallm" if lse is None else "smallm_lse"
    cached = module.launch_args.get(slot)
    if cached is None or cached[0] != key:
        cached = (
            key,
            _decode_smallm_launch_args(
                module,
                query=query,
                workspace_buffer=workspace_buffer,
                counter_buffer=counter_buffer,
                own_lse=lse is None,
                sm_count=sm_count,
            ),
        )
        module.launch_args[slot] = cached
    args, zero = cached[1]
    if zero is not None:
        zero.zero_()
    values: dict[str, Any] = {
        "Q": query,
        "K": key_cache,
        "V": value_cache,
        "O_ptr": out,
        "page_table": block_tables,
        "seq_lens": seq_lens,
        "max_pages_per_seq": int(block_tables.shape[-1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        **args,
    }
    if lse is not None:
        values["LSE_ptr"] = lse
    return [(module.program, module.entry, values)]


def _decode_smallm_launch_args(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    own_lse: bool,
    sm_count: int,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """The workspace arguments and the grid of a small-M launch, and the
    workspace tickets to zero before each launch (``None`` when the tickets
    live in the caller's counter buffer)."""

    route = module.route
    num_split = route.num_split
    n_rows = _smallm_tile_rows_for(route.q_len, route.num_q_heads // route.num_kv_heads)
    tiles = route.batch_size * route.num_kv_heads
    slots = tiles * num_split

    def align(value: int) -> int:
        return (value + 255) // 256 * 256

    partial_lse_offset = align(slots * n_rows * 256 * 2)
    counters_offset = align(partial_lse_offset + slots * n_rows * 4)
    lse_offset = align(counters_offset + tiles * 2 * 4)
    flat = _decode_program_bytes(workspace_buffer)
    args: dict[str, Any] = {
        "partial_O_ptr": _decode_program_workspace(
            flat, 0, slots * n_rows * 256 * 2, torch.bfloat16
        ),
        "partial_LSE_ptr": _decode_program_workspace(
            flat, partial_lse_offset, slots * n_rows * 4, torch.float32
        ),
    }
    counter_bytes = tiles * 2 * 4
    zero = None
    if (
        counter_buffer is not None
        and counter_buffer.device == query.device
        and counter_buffer.is_contiguous()
        and counter_buffer.numel() * counter_buffer.element_size() >= counter_bytes
        and counter_buffer.data_ptr() % 4 == 0
    ):
        counter_flat = counter_buffer.view(-1).view(torch.uint8)
        args["counters"] = counter_flat[:counter_bytes].view(torch.uint32)
    else:
        args["counters"] = zero = _decode_program_workspace(
            flat, counters_offset, counter_bytes, torch.uint32
        )
    if own_lse:
        rows = int(query.shape[0]) * int(query.shape[1])
        args["LSE_ptr"] = _decode_program_workspace(
            flat, lse_offset, rows * 4, torch.float32
        )
    if query.device.type == "cuda":
        args = {name: tvm_ffi.from_dlpack(view) for name, view in args.items()}
    args.update(
        _decode_program_grid(
            module,
            module.program,
            module.launch,
            sm_count=sm_count,
            num_splits=num_split,
        )
    )
    return args, zero


_DECODE_PROGRAM_VALUES["decode_smallm"] = _decode_smallm_program_values


# --- BF16 decode programs (head dim 128, one query token) --------------------
#
# ``cake_fmha_decode_bf16``: 16-token HND pages, eight query heads per KV
# head, a stacked or strided cache.  A host-scale program takes the base-2
# softmax scale as a kernel scalar (bmm2 1); a device-scale twin reads the
# log2-domain bmm1 and the bmm2 FP32 scalars in the kernel.  The key is
# ``(scale form, batch, q_len, num_q_heads, num_kv_heads, page_size, sink,
# retain_kv_l2)``.  The 128-request programs are stream-K: one CTA per SM,
# each group of ``q_len * num_kv_heads`` CTAs walking a contiguous range of
# the requests' KV block pairs, a request cut between two groups merged
# through the split counters and scratch.  The families are tried in order,
# so a family of per-target programs can take a key ahead of this one.
_DECODE_BF16_FAMILIES = ("decode_bf16",)

_DECODE_PROGRAM_HOST_PARAMS["decode_bf16"] = (
    "Qt",
    "K",
    "V",
    "O_ptr",
    "LSE_ptr",
    "page_table",
    "causal_seqlens_kv_global",
    "scale_log2_ptr",
    "bmm2_ptr",
    "sinks_ptr",
    "max_pages_per_seq",
    "max_local_seq_len",
    "softmax_scale_log2",
    "window_left",
    "num_q_heads",
    "num_kv_heads",
    "batch_size",
)
_DECODE_PROGRAM_ROUTES["sm100a"].update(
    {
        ("decode_bf16", "host", 128, 1, 32, 4, 16, False, False): "DEC_B128",
        ("decode_bf16", "device", 128, 1, 32, 4, 16, False, False): "DEC_B128_DS",
        ("decode_bf16", "host", 4, 1, 32, 4, 16, False, True): "DEC_B4C4",
        ("decode_bf16", "device", 4, 1, 32, 4, 16, False, True): "DEC_B4C4_DS",
        ("decode_bf16", "host", 256, 1, 32, 4, 16, True, False): "DEC_SINK",
        ("decode_bf16", "device", 256, 1, 32, 4, 16, True, False): "DEC_SINK_DS",
    }
)
_DECODE_PROGRAM_ROUTES["sm103a"].update(
    {
        ("decode_bf16", "host", 128, 1, 32, 4, 16, False, False): "DEC_B128",
        ("decode_bf16", "device", 128, 1, 32, 4, 16, False, False): "DEC_B128_DS",
        ("decode_bf16", "host", 4, 1, 32, 4, 16, False, True): "DEC_B4C4",
        ("decode_bf16", "device", 4, 1, 32, 4, 16, False, True): "DEC_B4C4_DS",
        ("decode_bf16", "host", 256, 1, 32, 4, 16, True, False): "DEC_SINK",
        ("decode_bf16", "device", 256, 1, 32, 4, 16, True, False): "DEC_SINK_DS",
    }
)
_DECODE_PROGRAMS.update(
    {
        ("decode_bf16", "DEC_B128"): (
            "cake_fmha_decode_bf16_4a8ccd779ab91c5d2cb5",
            "streamk",
        ),
        ("decode_bf16", "DEC_B128_DS"): (
            "cake_fmha_decode_bf16_4771b35b3b41de52fc15",
            "streamk",
        ),
        ("decode_bf16", "DEC_B4C4"): (
            "cake_fmha_decode_bf16_da0b91411447542f0766",
            "cluster",
        ),
        ("decode_bf16", "DEC_B4C4_DS"): (
            "cake_fmha_decode_bf16_41a50922430aef5ca651",
            "cluster",
        ),
        ("decode_bf16", "DEC_SINK"): (
            "cake_fmha_decode_bf16_eb613495c94cae21226f",
            "bhq",
        ),
        ("decode_bf16", "DEC_SINK_DS"): (
            "cake_fmha_decode_bf16_394b0caa8e0362fe9988",
            "bhq",
        ),
    }
)


def _decode_exact_sink_member_cannot_finish(
    route: CakeFmhaDecodeRoute, max_seq_len: int, block_tables: torch.Tensor
) -> bool:
    """Whether ``route`` runs the exact B256 sink member on a launch it cannot
    finish.  That member serves the route's shape without LSE outside the
    retained-KV form; its page producer always publishes the page groups of
    32 KV blocks and reads 256 page-table entries per row, so it finishes
    only when ``max_seq_len`` evens to 30 or 32 blocks (3585..4096) and the
    table holds 256 pages per row.  Every other member keeps its launches."""

    shape = (route.batch_size, route.q_len, route.num_q_heads, route.num_kv_heads)
    return (
        route.has_sink
        and shape == (256, 1, 32, 4)
        and not route.return_lse
        and not route.retain_kv_l2
        and not route.use_scale_ptr
        and not route.has_window
        and not (3585 <= max_seq_len <= 4096 and int(block_tables.shape[-1]) >= 256)
    )


def _decode_bf16_program_select(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    q_len: int,
    max_seq_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    o_scale: float | None,
    sinks: torch.Tensor | None,
    lse: torch.Tensor | None,
    kv_layout: str,
    uses_shared_paged_kv_idx: bool,
    no_block_scales: bool,
) -> tuple[tuple[str, str], bool] | None:
    """``(program, device_scales)`` of a BF16 decode launch the generated
    programs serve, or ``None``.

    A hit takes a contiguous query, 16-token HND pages of head dim 128, a
    contiguous int32 shared page table at least as wide as the kernels read
    (eight pages per evened 128-token block, at least four blocks), int32
    lengths (the kernels derive each request's causal prefix from them), no
    LSE, window, block scales or output scale, one scale form (host: bmm2 1)
    and a workspace holding the kernels' LSE scratch.  A stream-K program
    also needs a batch of at least two requests per CTA group and a
    workspace holding its split scratch and counters
    (:func:`_decode_streamk_layout`).
    """

    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if not (
        query.is_contiguous()
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 128)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and block_tables.ndim == 2
        and block_tables.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and lse is None
        and window_left < 0
        and no_block_scales
        and (o_scale is None or float(o_scale) == 1.0)
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
    ):
        return None
    form = _decode_program_scale_form(bmm1_scale, bmm2_scale, query)
    if form is None or (form == "host" and float(bmm2_scale) != 1.0):
        return None
    blocks = (max_seq_len + 127) // 128
    even = max(4, blocks + blocks % 2)
    if int(block_tables.shape[-1]) < 8 * even:
        return None
    if not _decode_native_workspace_supported(
        workspace_buffer,
        query,
        block_tables,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pages_per_block=8,
        page_table_rows=1,
        lse=None,
    ):
        return None
    key = (
        form,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        16,
        sinks is not None,
        max(1, blocks) <= 9,
    )
    for family in _DECODE_BF16_FAMILIES:
        program = _decode_program_lookup(device, family, key)
        if program is not None:
            launch = _DECODE_PROGRAMS[program][1]
            # A cluster program gives each tile its own cluster and arms its
            # reduction barrier once: a device with fewer SMs than tiles (a
            # partition) would hand a cluster a second tile and hang.
            if (
                launch == "cluster"
                and _smallm_sm_count(device) < batch_size * q_len * num_kv_heads
            ):
                return None
            # A stream-K program cuts each request into at most two parts on
            # adjacent CTA groups, which needs two requests per group; its
            # split scratch follows the LSE scratch in the workspace.
            if launch == "streamk":
                groups = _decode_streamk_groups(
                    batch_size, q_len, num_kv_heads, _smallm_sm_count(device)
                )
                grid = groups * q_len * num_kv_heads
                rows = int(query.shape[0]) * num_q_heads
                end = _decode_streamk_layout(batch_size, rows, grid)[2]
                if groups == 0 or _decode_program_bytes_of(workspace_buffer) < end:
                    return None
            return program, form == "device"
    return None


def _decode_bf16_program_values(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    sm_count: int,
    **unused: Any,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """The one launch of a BF16 decode program.  The kernels read the page
    table in place (row stride its width), each request's length (its causal
    prefix is the length less the query length) and write the per-row LSE
    to scratch in the workspace (the JIT route's offset).  A pointer the
    program does not read (a host-scale program's scales, an absent sink)
    points at the lengths, as the JIT route's do.  A stream-K program also
    takes its split scratch and split counters
    (:func:`_decode_streamk_launch_args`)."""

    route = module.route
    rows = int(query.shape[0]) * int(query.shape[1])
    lse_offset = (route.batch_size * 4 + 15) // 16 * 16
    unread = seq_lens.view(torch.float32)
    if module.route.device_scales:
        if not isinstance(bmm1_scale, torch.Tensor) or not isinstance(
            bmm2_scale, torch.Tensor
        ):
            raise RuntimeError(
                f"Cake FMHA decode program {module.identifier} was selected for "
                "device scales; the launch passes host scales"
            )
        scale_ptr, bmm2_ptr = bmm1_scale.reshape(1), bmm2_scale.reshape(1)
        softmax_scale_log2 = 0.0
    else:
        scale_ptr = bmm2_ptr = unread
        softmax_scale_log2 = float(bmm1_scale) * log2e
    values: dict[str, Any] = {
        "Qt": query,
        "K": key_cache,
        "V": value_cache,
        "O_ptr": out,
        "LSE_ptr": _decode_program_workspace(
            _decode_program_bytes(workspace_buffer),
            lse_offset,
            rows * 4,
            torch.float32,
        ),
        "page_table": block_tables,
        "causal_seqlens_kv_global": seq_lens,
        "scale_log2_ptr": scale_ptr,
        "bmm2_ptr": bmm2_ptr,
        "sinks_ptr": unread if sinks is None else sinks,
        "max_pages_per_seq": int(block_tables.shape[-1]),
        "max_local_seq_len": max_seq_len,
        "softmax_scale_log2": softmax_scale_log2,
        "window_left": -1,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "batch_size": route.batch_size,
    }
    if module.launch == "streamk":
        values.update(
            _decode_streamk_launch_args(
                module,
                query=query,
                workspace_buffer=workspace_buffer,
                counter_buffer=counter_buffer,
                sm_count=sm_count,
            )
        )
        return [(module.program, module.entry, values)]
    grid = _decode_program_grid(
        module, module.program, module.launch, sm_count=sm_count
    )
    values.update(grid)
    return [(module.program, module.entry, values)]


def _decode_streamk_launch_args(
    module: CakeFmhaDecodeProgramModule,
    *,
    query: torch.Tensor,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    sm_count: int,
) -> dict[str, Any]:
    """The split scratch, the split counters and the grid of a stream-K
    launch.  The scratch holds two split-tile parts per CTA after the LSE
    scratch.  The counters (one u32 per CTA) live in the caller's zeroed
    counter buffer when it can hold them (the kernel's second arrival at a
    split tile resets its counter, so they are zero after every launch),
    else in the workspace, zeroed on the stream first.

    These depend only on the buffers' addresses and sizes and the SM count,
    so the module keeps them from its last launch (on a CUDA device as FFI
    tensors, converted once)."""

    counter_key = None
    if counter_buffer is not None:
        counter_key = (
            counter_buffer.data_ptr(),
            _decode_program_bytes_of(counter_buffer),
            counter_buffer.is_contiguous(),
        )
    key = (
        workspace_buffer.data_ptr(),
        _decode_program_bytes_of(workspace_buffer),
        counter_key,
        sm_count,
    )
    cached = module.launch_args.get("streamk")
    if cached is None or cached[0] != key:
        route = module.route
        grid = _decode_program_grid(
            module, module.program, module.launch, sm_count=sm_count
        )
        ctas = grid["grid_x"]
        rows = int(query.shape[0]) * int(query.shape[1])
        split_ws_offset, counters_offset, _ = _decode_streamk_layout(
            route.batch_size, rows, ctas
        )
        flat = _decode_program_bytes(workspace_buffer)
        args: dict[str, Any] = {
            "split_ws": _decode_program_workspace(
                flat,
                split_ws_offset,
                ctas * 2 * _DECODE_STREAMK_PART_WORDS * 4,
                torch.float32,
            ),
        }
        zero = None
        if (
            counter_buffer is not None
            and counter_buffer.device == query.device
            and counter_buffer.is_contiguous()
            and _decode_program_bytes_of(counter_buffer) >= ctas * 4
            and counter_buffer.data_ptr() % 4 == 0
        ):
            counter_flat = counter_buffer.view(-1).view(torch.uint8)
            args["split_counter"] = counter_flat[: ctas * 4].view(torch.uint32)
        else:
            args["split_counter"] = zero = _decode_program_workspace(
                flat, counters_offset, ctas * 4, torch.uint32
            )
        if query.device.type == "cuda":
            args = {name: tvm_ffi.from_dlpack(view) for name, view in args.items()}
        args.update(grid)
        cached = (key, (args, zero))
        module.launch_args["streamk"] = cached
    args, zero = cached[1]
    if zero is not None:
        zero.zero_()
    return args


_DECODE_PROGRAM_VALUES["decode_bf16"] = _decode_bf16_program_values


def select_cake_fmha_decode_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    q_len: int | None,
    max_seq_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    o_scale: float | None,
    sinks: torch.Tensor | None,
    kv_layout: str,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor | None,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    enable_block_sparse_attention: bool,
    lse: torch.Tensor | None = None,
    multi_ctas_kv_counter_buffer: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> CakeFmhaDecodeRoute | None:
    """Select an exact product decode route without broadening its contract.

    ``enable_pdl`` is the caller's launch mode; a generated program whose split-KV
    reducer waits on the decode grid serves only a call that asks for it.

    A launch with ``lse`` takes the program route of the same launch without
    it, with ``return_lse`` set, when its JIT member is the route the module
    selects for the LSE launch: the route then runs that JIT member (the BF16
    programs write no caller LSE).
    """

    if q_len is None or q_len <= 0 or batch_size <= 0 or max_seq_len <= 0:
        return None
    if cum_seq_lens_q is not None or enable_block_sparse_attention:
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if any(
        tensor.device != query.device
        for tensor in (
            key_cache,
            value_cache,
            out,
            workspace_buffer,
            block_tables,
            seq_lens,
        )
    ):
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    if query.shape[0] != batch_size * q_len:
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if (
        query.stride(0) <= 0
        or query.stride(1) <= 0
        or query.stride(0) != num_q_heads * query.stride(1)
    ):
        return None
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    # The balanced head_dim-64 route serves 1..16 query heads per KV head (its
    # 16-row query tile); every other decode route serves 1..8.
    max_group = CAKE_FMHA_BALANCED_HD64_MAX_GROUP if int(query.shape[2]) == 64 else 8
    if not 1 <= num_q_heads // num_kv_heads <= max_group:
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if not block_tables.is_contiguous():
        return None
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            return None
    elif block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
        return None
    if (
        seq_lens.ndim != 1
        or seq_lens.shape[0] != batch_size
        or not seq_lens.is_contiguous()
    ):
        return None
    if seq_lens.dtype not in (torch.int32, torch.uint32):
        return None
    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.device != query.device
        or sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if lse is not None and (
        lse.device != query.device
        or lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
        or lse.stride() != (num_q_heads, 1)
    ):
        return None

    page_size = int(key_cache.shape[2])
    local_blocks = max(1, (max_seq_len + 127) // 128)

    def route(
        component,
        *,
        selected_page_size: int = page_size,
        aligned_program: str | None = None,
    ):
        candidate = CakeFmhaDecodeRoute(
            target=_cake_fmha_target(device),
            batch_size=batch_size,
            q_len=q_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            has_sink=sinks is not None,
            has_window=window_left >= 0,
            use_scale_ptr=isinstance(bmm1_scale, torch.Tensor),
            retain_kv_l2=local_blocks <= 9,
            component=component,
            page_size=selected_page_size,
            return_lse=lse is not None,
            aligned_program=aligned_program,
        )
        if component == "decode_native_bf16" and not cake_fmha_route_is_optimized(
            candidate
        ):
            return None
        return candidate

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None

    def aligned_route(dtype: str):
        """The aligned hd128 q1 route of ``dtype`` (``"bf16"`` / ``"fp16"``), or
        ``None`` when the launch is outside the programs' contract or
        ``aligned_cluster_size`` keeps the module's present route.

        The programs read the query as ``[batch, num_q_heads, 128]`` rows and
        K / V as separate contiguous ``[pages, num_kv_heads, 16, 128]`` pools
        (their TMA maps derive every stride from the shape), the int32 page table
        as ``[batch, max_pages_per_seq]`` and the int32 lengths on device; they
        write ``out`` in the query dtype and take no workspace, counters, LSE,
        sinks, window, device scales or output scale.
        """

        if not (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and q_len == 1
            and num_q_heads == 8 * num_kv_heads
            and block_tables.dtype == torch.int32
            and seq_lens.dtype == torch.int32
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return None
        cluster_size = aligned_cluster_size(
            batch_size,
            num_kv_heads,
            max_seq_len,
            q_len=q_len,
            num_q_heads=num_q_heads,
            dtype=dtype,
        )
        if cluster_size is None:
            return None
        return route(
            f"decode_aligned_{dtype}",
            selected_page_size=16,
            aligned_program=_aligned_program_id(cluster_size, dtype),
        )

    if dtypes == (torch.bfloat16,) * 4:
        group = num_q_heads // num_kv_heads
        smallm_rows = _smallm_tile_rows_for(q_len, group)
        smallm_tiles = batch_size * num_kv_heads
        smallm_sm_count = _smallm_sm_count(device)
        smallm_num_split = _smallm_num_split(
            tiles=smallm_tiles,
            sm_count=smallm_sm_count,
            max_seq_len=max_seq_len,
            n_rows=smallm_rows or 64,
        )
        if (
            query.is_contiguous()
            and query.shape[2] == 64
            and key_cache.shape[2:] == (16, 64)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and q_len == 1
            and 1 <= group <= CAKE_FMHA_BALANCED_HD64_MAX_GROUP
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16_hd64",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
                head_dim=64,
            )
        ):
            # head_dim 64 (Q16Kv128 ForGen body, 1..16 query heads per KV head)
            # on the on-device balanced scheduler; no other product route
            # serves head_dim 64.
            return route("decode_balanced_bf16_hd64", selected_page_size=16)
        if (
            query.is_contiguous()
            and query.shape[2] == 256
            and smallm_rows is not None
            and page_size in (16, 32, 64)
            and key_cache.shape[3] == 256
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and sinks is None
            and window_left < 0
            and not isinstance(bmm1_scale, torch.Tensor)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and smallm_num_split >= 1
            and _smallm_workspace_supported(
                workspace_buffer,
                query,
                tiles=smallm_tiles,
                num_split=smallm_num_split,
                n_rows=smallm_rows,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            smallm_program = _decode_smallm_program_select(
                device,
                n_rows=smallm_rows,
                page_size=page_size,
                q_len=q_len,
                group=group,
                tiles=smallm_tiles,
                max_seq_len=max_seq_len,
                num_split=smallm_num_split,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )
            smallm_key, smallm_split = smallm_program or (None, smallm_num_split)
            candidate = CakeFmhaDecodeRoute(
                target=_cake_fmha_target(device),
                batch_size=batch_size,
                q_len=q_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                has_sink=False,
                has_window=False,
                use_scale_ptr=False,
                retain_kv_l2=local_blocks <= 9,
                component="decode_native_bf16_hd256_smallm",
                page_size=page_size,
                num_split=smallm_split,
                program=smallm_key,
            )
            if smallm_key is not None:
                return candidate
            return candidate if cake_fmha_route_is_optimized(candidate) else None
        if (
            query.is_contiguous()
            and query.shape[2] == 256
            and key_cache.shape[3] == 256
            and page_size in (16, 32, 64)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and 1 <= q_len <= CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN
            and 1 <= group <= 8
            and max_seq_len >= q_len
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16_hd256",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
                head_dim=256,
            )
        ):
            # head_dim 256 outside the small-M route's packed tiles: every
            # query row is its own tile (uniform q_len 1..8) on the on-device
            # balanced scheduler, one program per page size (16 / 32 / 64).
            return route("decode_balanced_bf16_hd256")
        # Aligned hd128 q1 decode: one 8/4/2/1-CTA cluster per (request, KV
        # head) tile, selected by ``aligned_cluster_size`` ahead of the present
        # hd128 routes; ``None`` keeps them.
        aligned = aligned_route("bf16")
        if aligned is not None:
            return aligned
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and group == 8
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
            )
        ):
            # Long-context / ragged decode: the on-device balanced scheduler
            # splits long requests into chunks and merges them on device.
            return route("decode_balanced_bf16", selected_page_size=16)
        # Generated BF16 programs (host or device scales, sinks), ahead of the
        # native JIT members; a launch they do not serve keeps those.  An LSE
        # launch is selected as the same launch without it and decided at the
        # native member below.
        bf16_program = _decode_bf16_program_select(
            device,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            batch_size=batch_size,
            q_len=q_len,
            max_seq_len=max_seq_len,
            window_left=window_left,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            o_scale=o_scale,
            sinks=sinks,
            lse=None,
            kv_layout=kv_layout,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            no_block_scales=no_block_scales,
        )
        program_route = None
        if bf16_program is not None:
            program_route = CakeFmhaDecodeRoute(
                target=_cake_fmha_target(device),
                batch_size=batch_size,
                q_len=q_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                has_sink=sinks is not None,
                has_window=False,
                use_scale_ptr=bf16_program[1],
                retain_kv_l2=local_blocks <= 9,
                component="decode_native_bf16",
                page_size=16,
                program=bf16_program[0],
                device_scales=bf16_program[1],
            )
            if lse is None:
                return program_route
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=8,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            native = route("decode_native_bf16", selected_page_size=16)
            if native is not None and _decode_exact_sink_member_cannot_finish(
                native, max_seq_len, block_tables
            ):
                # A launch the exact sink member cannot finish takes the
                # compat route (a program it fits took its program above).
                return None
            # The LSE launch keeps the program route of the launch without it
            # when that route's JIT member is this one; the route runs the
            # member (see _decode_route_runs_program).
            if program_route is not None and native == dataclass_replace(
                program_route, program=None, device_scales=False, return_lse=True
            ):
                return dataclass_replace(program_route, return_lse=True)
            return native
        return None

    if dtypes == (torch.float16,) * 4:
        # Aligned hd128 q1 decode in FP16: the same programs traced with FP16
        # Q/K/V/O, ahead of the balanced FP16 route; ``None`` keeps it.
        aligned = aligned_route("fp16")
        if aligned is not None:
            return aligned
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and num_q_heads // num_kv_heads == 8
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="fp16",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=_smallm_sm_count(device),
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
            )
        ):
            # Long-context / ragged FP16 decode: the same on-device balanced
            # scheduler and physical schedule, traced with FP16 Q/K/V/O.
            return route("decode_balanced_fp16", selected_page_size=16)
        if (
            query.shape[2] == 128
            and key_cache.shape[2:] == (32, 128)
            and kv_layout == "NHD"
            and not uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=4,
                page_table_rows=2,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_nhd", selected_page_size=32)
        if (
            query.shape[2] == 512
            and query.is_contiguous()
            and key_cache.shape[2:] == (64, 512)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=2,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and sinks is None
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_hd512", selected_page_size=64)
        return None

    quant_extensions_absent = (
        sinks is None
        and lse is None
        and window_left == -1
        and skip_softmax_threshold_scale_factor in (None, 0.0)
    )
    # E4M3 cache on the on-device balanced scheduler (E4M3, BF16 or FP16 query
    # with the output in the query dtype; HND, page 16, GQA-8, q_len 1).  Host
    # scales only: ``bmm1_scale = q_scale * k_scale / sqrt(128)`` becomes the
    # kernel's log2 softmax scale and ``bmm2_scale = v_scale / o_scale`` its
    # output scale; device-tensor scales stay on the quantized routes / compat
    # (a per-launch host read would break graph-replay safety).
    balanced_fp8_kv_component = {
        (torch.float8_e4m3fn,) * 4: "decode_balanced_fp8",
        (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        ): "decode_balanced_bf16q",
        (
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float16,
        ): "decode_balanced_fp16q",
    }.get(dtypes)
    if (
        balanced_fp8_kv_component is not None
        and q_len == 1
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 128)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and num_q_heads // num_kv_heads == 8
        and no_block_scales
        and quant_extensions_absent
        and (o_scale is None or float(o_scale) == 1.0)
        and not isinstance(bmm1_scale, torch.Tensor)
        and not isinstance(bmm2_scale, torch.Tensor)
        and float(bmm1_scale) > 0.0
        and float(bmm2_scale) > 0.0
        and query.is_contiguous()
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and _balanced_route_supported(
            _cake_fmha_target(device),
            dtype=balanced_fp8_kv_component.removeprefix("decode_balanced_"),
            batch_size=batch_size,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            sm_count=_smallm_sm_count(device),
            workspace_buffer=workspace_buffer,
            counter_buffer=multi_ctas_kv_counter_buffer,
        )
    ):
        return route(balanced_fp8_kv_component, selected_page_size=16)

    if (
        dtypes == (torch.float8_e4m3fn,) * 4
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and num_q_heads // num_kv_heads == 8
        and no_block_scales
        and quant_extensions_absent
        and max_seq_len >= 512
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_fp8_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            page_size=page_size,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        # num_split=8 selects the fixed 8 x 16 split-plan member (decode -> reducer PDL).
        fixed = _decode_quant_fp8_fixed_split(
            batch_size, num_kv_heads, max_seq_len, _smallm_sm_count(device)
        )
        candidate = dataclass_replace(
            route("decode_quant_fp8"), num_split=8 if fixed else 0
        )
        quant_program = _decode_quant_program_select(
            device,
            "decode_fp8",
            query=query,
            block_tables=block_tables,
            seq_lens=seq_lens,
            workspace_buffer=workspace_buffer,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            enable_pdl=enable_pdl is True,
        )
        if quant_program is not None:
            return dataclass_replace(
                candidate,
                program=quant_program[0],
                device_scales=quant_program[1],
                num_split=quant_program[2],
            )
        return candidate

    if (
        dtypes
        == (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout in ("HND", "NHD")
        and num_q_heads % num_kv_heads == 0
        and 1 <= num_q_heads // num_kv_heads < 8
        and no_block_scales
        and quant_extensions_absent
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_bf16q")

    if (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and q_len == 1
        and query.shape[2] == 128
        and key_cache.shape[3] == 64
        and page_size in (16, 32)
        and kv_layout == "HND"
        and quant_extensions_absent
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.ndim == 4
        and value_block_scales.shape == key_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.stride(3) == 1
        and value_block_scales.stride(3) == 1
        and key_block_scales.device == query.device
        and value_block_scales.device == query.device
        and num_q_heads // num_kv_heads <= 8
        and _tma_nvfp4_paged_kv_strides_supported(key_cache)
        and _tma_nvfp4_paged_kv_strides_supported(value_cache)
        and _tma_nvfp4_scale_strides_supported(key_block_scales)
        and _tma_nvfp4_scale_strides_supported(value_block_scales)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and key_block_scales.data_ptr() % 16 == 0
        and value_block_scales.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_nvfp4_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        candidate = route("decode_quant_nvfp4")
        quant_program = None
        if _decode_program_toolkit_at_least("13.4"):
            # The generated program converts the E2M1 codes with the CUDA 13.4
            # toolchain's native conversion; older toolkits keep the JIT member.
            quant_program = _decode_quant_program_select(
                device,
                "decode_nvfp4",
                query=query,
                block_tables=block_tables,
                seq_lens=seq_lens,
                workspace_buffer=workspace_buffer,
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            )
        if quant_program is not None:
            return dataclass_replace(
                candidate,
                program=quant_program[0],
                device_scales=quant_program[1],
                num_split=quant_program[2],
            )
        return candidate
    return None


@dataclass(frozen=True)
class CakeFmhaAlignedDecodeModule:
    """One loaded aligned hd128 q1 decode program: the route it serves, the
    ``CakeProgram`` record, its exact target, the cluster size the launch grid
    spans along z and the binding's ``run`` entry."""

    route: CakeFmhaDecodeRoute
    program: Any
    target: str
    cluster_size: int
    entry: Any


def _load_cake_fmha_aligned_decode_module(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaAlignedDecodeModule:
    """Build and load the aligned program of ``route`` on the route's target.

    The loader ``flashinfer.jit.cake_fmha_aligned`` is generated by the Cake
    export beside ``csrc/cake_fmha_aligned/``; it is imported here, lazily, so
    route selection never needs it.  The record's launch facts and argument
    names are checked against the host launch once, at load.
    """

    from .jit import cake_fmha_aligned
    from .jit.cake_program import GRID_ARGS, CakeProgram

    program_facts = _aligned_route_program(route)
    if program_facts is None:
        raise RuntimeError(
            f"Cake FMHA aligned decode route names no program: {route.aligned_program!r}"
        )
    cluster_size, _, program_name = program_facts
    program = getattr(cake_fmha_aligned, route.aligned_program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned publishes no CakeProgram {route.aligned_program!r}"
        )
    if program.name != program_name:
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned.{route.aligned_program} is "
            f"{program.name}; this module pins {program_name}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    if tuple(program.args) != (*_ALIGNED_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_ALIGNED_HOST_PARAMS, *GRID_ARGS]}"
        )
    if tuple(program.cluster) != (1, 1, cluster_size):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; the route "
            f"expects (1, 1, {cluster_size})"
        )
    return CakeFmhaAlignedDecodeModule(
        route=route,
        program=program,
        target=target,
        cluster_size=cluster_size,
        entry=program.entry(target),
    )


def _run_cake_fmha_aligned_decode(
    module: CakeFmhaAlignedDecodeModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Launch one aligned hd128 q1 decode on the current torch stream.

    The binding's host parameters, in the kernel's order: the query rows
    (``Qt``, TMA source, ``[batch x num_kv_heads x 8, 128]``), the K and V pools
    (TMA sources), the output, the page table, the device lengths,
    ``uniform_seq_len`` (0: every CTA reads its request's length from
    ``seq_lens_kv``, so a captured launch follows new lengths),
    ``max_pages_per_seq`` (the page table's row width; the kernel clamps its
    page-id reads to the row), the log2-domain softmax scale, the head counts
    and the batch size; then the grid of :func:`cake_fmha_aligned_launch_grid`.
    """

    route = module.route
    if (
        tuple(query.shape) != (route.batch_size, route.num_q_heads, 128)
        or key_cache.shape[1] != route.num_kv_heads
        or tuple(block_tables.shape[:1]) != (route.batch_size,)
    ):
        raise RuntimeError(
            "Cake FMHA aligned decode launch does not match its selected route"
        )
    grid = cake_fmha_aligned_launch_grid(
        module.program, route.batch_size, route.num_kv_heads
    )
    values = {
        "Qt": query.view(-1, 128),
        "K": key_cache,
        "V": value_cache,
        "O": out,
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "uniform_seq_len": 0,
        "max_pages_per_seq": int(block_tables.shape[1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "batch_size": route.batch_size,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    arguments = [values[name] for name in module.program.args]
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.entry(*arguments)


def _resolve_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
) -> tuple[Any, bool]:
    """Resolve a decode module and whether its optimized ABI remains active.

    An aligned hd128 q1 route resolves to a :class:`CakeFmhaAlignedDecodeModule`
    (a generated ``CakeProgram`` binding launched by
    :func:`_run_cake_fmha_aligned_decode`), a generated decode program route to a
    :class:`CakeFmhaDecodeProgramModule` (launched by
    :func:`_run_cake_fmha_decode_program`), every other optimized route to a
    module exposing ``cake_paged_attention_decode``.
    """

    if route is not None:
        # A route's program was checked when it loaded: later launches reuse it.
        program_module = _DECODE_PROGRAM_MODULES.get(route)
        if program_module is not None and _decode_route_runs_program(route):
            if route.target != _cake_fmha_target(device):
                raise RuntimeError(
                    "Cake FMHA decode route target does not match the device"
                )
            return program_module, True
    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device)), False
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA decode route target does not match the device")
    if route.component in _ALIGNED_COMPONENTS:
        return _load_cake_fmha_aligned_decode_module(route), True
    if _decode_route_runs_program(route):
        return _load_cake_fmha_decode_program(route), True
    loader = {
        "decode_balanced_bf16": load_cake_fmha_decode_balanced_bf16_module,
        "decode_balanced_fp16": load_cake_fmha_decode_balanced_fp16_module,
        "decode_balanced_fp8": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_bf16q": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_fp16q": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_bf16_hd64": load_cake_fmha_decode_balanced_hd64_module,
        "decode_balanced_bf16_hd256": load_cake_fmha_decode_balanced_hd256_module,
        "decode_native_bf16": load_cake_fmha_decode_native_bf16_module,
        "decode_native_bf16_hd256_smallm": load_cake_fmha_decode_native_bf16_hd256_smallm_module,
        "decode_native_fp16_hd512": load_cake_fmha_decode_native_fp16_hd512_module,
        "decode_native_fp16_nhd": load_cake_fmha_decode_native_fp16_nhd_module,
        "decode_quant_bf16q": load_cake_fmha_decode_quant_bf16q_module,
        "decode_quant_fp8": load_cake_fmha_decode_quant_fp8_module,
        "decode_quant_nvfp4": load_cake_fmha_decode_quant_nvfp4_module,
    }.get(route.component)
    if loader is None:
        raise RuntimeError(
            f"Cake FMHA decode route has no authenticated loader: {route.component}"
        )
    common_args = (
        route.target,
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
    )
    if route.component in ("decode_balanced_bf16", "decode_balanced_fp16"):
        return loader(route.target, route.q_len), True
    if route.component in (
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
    ):
        return loader(
            route.target, route.component.removeprefix("decode_balanced_")
        ), True
    if route.component == "decode_balanced_bf16_hd64":
        # Structural instance by head group: eight softmax columns for 1..8
        # query heads per KV head (every product row), sixteen for 9..16.
        return loader(
            route.target,
            cake_fmha_balanced_hd64_max_group(route.num_q_heads, route.num_kv_heads),
        ), True
    if route.component == "decode_balanced_bf16_hd256":
        return loader(route.target, route.page_size), True
    if route.component == "decode_native_bf16_hd256_smallm":
        group = route.num_q_heads // route.num_kv_heads
        return (
            loader(
                route.target,
                _smallm_tile_rows_for(route.q_len, group),
                route.page_size,
                route.q_len,
                group,
                route.num_split,
            ),
            True,
        )
    if route.component == "decode_native_fp16_hd512":
        return (
            loader(
                *common_args,
                has_window=route.has_window,
                use_scale_ptr=route.use_scale_ptr,
                retain_kv_l2=route.retain_kv_l2,
            ),
            True,
        )
    if route.component == "decode_quant_bf16q":
        return loader(*common_args, route.page_size), True
    if route.component == "decode_quant_fp8":
        # The host does not read seq_lens: the kernel masks every tail at run time.
        return (
            loader(
                *common_args,
                route.page_size,
                full_blocks=False,
                fixed_split=route.num_split == 8,
            ),
            True,
        )
    if route.component == "decode_quant_nvfp4":
        try:
            return loader(*common_args, route.page_size), True
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA portable NVFP4 loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target), False
    if route.component == "decode_native_bf16":
        return (
            loader(
                *common_args,
                has_sink=route.has_sink,
                has_window=route.has_window,
                use_scale_ptr=route.use_scale_ptr,
                retain_kv_l2=route.retain_kv_l2,
                return_lse=route.return_lse,
            ),
            True,
        )
    return (
        loader(
            *common_args,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
        ),
        True,
    )


def get_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
):
    """Load an optimized decode module, or the authenticated portable fallback."""

    module, _ = _resolve_cake_fmha_decode_module(device, route)
    return module


def _context_tile_mma_work(q_len: int, kv_len: int, tokens_per_tile: int) -> int:
    """Mirror the standalone route's bottom-right causal tile-work model."""

    total = 0
    full_n_blocks = (kv_len + 127) // 128
    shift = kv_len - q_len
    for m_block in range((q_len + tokens_per_tile - 1) // tokens_per_tile):
        max_n = (m_block + 1) * tokens_per_tile + shift
        total += (max_n + 127) // 128 if max_n < kv_len else full_n_blocks
    return total


@functools.lru_cache(maxsize=1024)
def _context_pack_g(
    max_q_len: int, max_kv_len: int, num_q_heads: int, num_kv_heads: int
) -> int:
    """Choose the canonical packed-GQA axis from host-visible maxima (a pure
    function of its integers, cached: the tile-work sums walk every query
    block, which is tens of microseconds at long query lengths)."""

    group = num_q_heads // num_kv_heads
    if group <= 1 or group > 128:
        return 1
    unpacked = num_q_heads * _context_tile_mma_work(max_q_len, max_kv_len, 256)
    packed = num_kv_heads * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // group)
    )
    return group if packed < unpacked else 1


def _context_fp8_runtime_pack(pack_g: int) -> int:
    """The runtime-size FP8 program's pack for the JIT pack ``pack_g``
    (:func:`_context_pack_g`): the largest shipped pack dividing it, else 1."""

    for pack in _CONTEXT_FP8_RUNTIME_PACKS:
        if pack_g % pack == 0:
            return pack
    return 1


@functools.lru_cache(maxsize=1024)
def _context_fp8_runtime_serves(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
) -> bool:
    """Whether a runtime-size FP8 program serves a causal call without LSE or
    sink that no exact route serves (``False`` keeps the JIT module, whose
    pack is ``pack_g``; a pure function of its arguments, cached like
    :func:`_context_pack_g`)."""

    pack = _context_fp8_runtime_pack(pack_g)
    # A shipped pack below the JIT pack holds fewer query heads per tile: it
    # may run more tile work than the JIT pack.
    if pack != pack_g and (num_q_heads // pack) * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // pack)
    ) > (num_q_heads // pack_g) * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // pack_g)
    ):
        return False
    # sm100a: calls this long run back to back at the board power limit, where
    # the runtime program's SM clock drops below the JIT module's: many
    # (request, KV head) streams, or many query rows at long query lengths.
    return not (
        target == "sm100a"
        and (
            batch_size * num_kv_heads >= 2048
            or (max_q_len >= 128 and batch_size * num_q_heads * max_q_len >= 1 << 20)
        )
    )


# The split planner's cost model per target, in microseconds: the launch's
# fixed cost, one KV block of a unit, the fixed cost of a unit, the partial
# store of a split unit, the merge of a tile of k units (MERGE0 + MERGE1 x k)
# by its last unit, and each extra work unit's share of the memory system
# (the per-CTA schedule does not see it).  Fitted to timed split plans of
# B200 (sm100a) and of GB300 (sm103a).
_CONTEXT_FP8_SPLIT_COSTS: dict[str, tuple[float, ...]] = {
    "sm100a": (5.294, 1.529, 3.327, 3.425, 1.383, 1.661, 0.014),
    "sm103a": (5.703, 1.247, 2.046, 0.256, 3.705, 1.648, 0.023),
}
# The query rows of a tile scale its unit and its partial store and merge (the
# program loads, stores and merges the rows of the stages its queries reach):
# a tile of ``r`` of 256 query rows (``r = min(max_q_len x pack, 256)`` in a
# launch of one m-block, else 256) pays ``S0 + (1 - S0) x r / 256`` of a full
# tile's partial store and merge and ``U0 + (1 - U0) x r / 256`` of its unit.
# The split choice compares packs under these costs
# (:func:`_context_fp8_split_choice`); a pack's own plan keeps the costs
# above.  ``(S0, U0)`` per target, fitted with the costs above fixed to timed
# split launches of every shipped pack (B200, GB300).
_CONTEXT_FP8_SPLIT_ROWS: dict[str, tuple[float, float]] = {
    "sm100a": (0.640, 0.302),
    "sm103a": (0.591, 0.124),
}
# A split must be predicted faster than the whole-tile schedule by both
# margins (2 % and 0.5 us: the smallest that timed plans of B200 and GB300
# support; at 1 % some predicted gains measure as losses); a launch of more
# tiles than this many waves keeps whole tiles (its last partial wave is a
# small share of it).
_CONTEXT_FP8_SPLIT_REL_MARGIN = 0.02
_CONTEXT_FP8_SPLIT_ABS_MARGIN = 0.5
_CONTEXT_FP8_SPLIT_MAX_WAVES = 4


def _context_fp8_tile_blocks(
    batch_size: int, max_q_len: int, max_kv_len: int, num_q_heads: int, pack: int
) -> list[int]:
    """KV blocks of every tile of a runtime-size launch, in the program's tile
    order (sections of 8 work units x every m-block, the largest m-block
    first; bottom-right causal), every request at the planned maxima."""

    tok = 128 // pack
    m_blocks = max((max_q_len + 2 * tok - 1) // (2 * tok), 1)
    total_bh = batch_size * (num_q_heads // pack)
    full_n = max((max_kv_len + 127) // 128, 1)
    shift = max_kv_len - max_q_len
    per_m = []
    for m_block in range(m_blocks):
        top = (m_block * 2 + 2) * tok + shift
        per_m.append(max((top + 127) // 128 if top < max_kv_len else full_n, 1))
    full_sections = total_bh // 8
    blocks = []
    for tile in range(m_blocks * total_bh):
        section, m = divmod(tile >> 3, m_blocks)
        if section >= full_sections:
            m = (tile - section * 8 * m_blocks) % m_blocks
        blocks.append(per_m[m_blocks - 1 - m])
    return blocks


@functools.lru_cache(maxsize=256)
def _context_fp8_split_costs(
    target: str, max_q_len: int, pack: int
) -> tuple[float, ...] | None:
    """The split cost model of a launch of ``pack`` at ``max_q_len`` query
    tokens: :data:`_CONTEXT_FP8_SPLIT_COSTS` with the unit, the partial store
    and the merge scaled by the query rows of its tiles
    (:data:`_CONTEXT_FP8_SPLIT_ROWS`), or ``None`` for a target without one."""

    costs = _CONTEXT_FP8_SPLIT_COSTS.get(target)
    rows = 256
    if max_q_len <= 2 * (128 // pack):
        rows = min(max_q_len * pack, 256)
    if costs is None or rows == 256:
        return costs
    s0, u0 = _CONTEXT_FP8_SPLIT_ROWS.get(target, (1.0, 1.0))
    store_share = s0 + (1.0 - s0) * rows / 256
    unit_share = u0 + (1.0 - u0) * rows / 256
    fixed, block, unit, store, merge0, merge1, extra = costs
    return (
        fixed,
        block,
        unit * unit_share,
        store * store_share,
        merge0 * store_share,
        merge1 * store_share,
        extra,
    )


def _context_fp8_split_makespan(
    blocks: list[int],
    sm_count: int,
    split_tiles: int,
    split_blocks: int,
    num_splits: int,
    costs: tuple[float, ...],
) -> float:
    """The predicted time of a plan: the persistent CTAs draw the work units in
    ticket order (the split units split-major, then the other tiles whole), a
    unit costs ``unit + blocks x block`` (plus the partial store when its tile
    has more than one unit), the tile's last finisher merges, and the launch
    adds its fixed cost and the extra units' share (``costs``: a launch's
    :func:`_context_fp8_split_costs`)."""

    fixed, block, unit, store, merge0, merge1, extra = costs
    tiles = len(blocks)
    split_tiles = min(split_tiles, tiles) if num_splits > 1 else 0
    ctas = min(sm_count, tiles + (num_splits - 1) * split_tiles)
    free = [0.0] * ctas
    ready = [(0.0, cta) for cta in range(ctas)]
    finished: dict[int, list[tuple[float, int]]] = {}

    def draw(cost: float) -> tuple[float, int]:
        # the CTA free first draws the next ticket (entries a merge made stale
        # are skipped)
        while True:
            start, cta = heapq.heappop(ready)
            if start == free[cta]:
                break
        free[cta] = start + cost
        heapq.heappush(ready, (free[cta], cta))
        return free[cta], cta

    for split in range(num_splits if split_tiles else 0):
        for tile in range(split_tiles):
            n = blocks[tile]
            count = min((n + split_blocks - 1) // split_blocks, num_splits)
            if split >= count:
                continue
            nb = split_blocks if split + 1 < count else n - split * split_blocks
            end, cta = draw(unit + nb * block + (store if count > 1 else 0.0))
            finished.setdefault(tile, []).append((end, cta))
            if count > 1 and split + 1 == count:
                merger = max(finished[tile])[1]
                free[merger] += merge0 + merge1 * count
                heapq.heappush(ready, (free[merger], merger))
    for tile in range(split_tiles, tiles):
        draw(unit + blocks[tile] * block)
    return fixed + max(free) + extra * (num_splits - 1) * split_tiles


@functools.lru_cache(maxsize=1024)
def _context_fp8_split_search(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack: int,
    sm_count: int,
    rows: bool,
) -> tuple[tuple[int, int, int] | None, float, float] | None:
    """The split of a launch of ``pack`` the cost model predicts fastest
    (:data:`_CONTEXT_FP8_SPLIT_COSTS`, scaled by the query rows of its tiles
    where ``rows``: :func:`_context_fp8_split_costs`), its predicted time
    and the whole-tile schedule's, or ``None`` without a cost model or for a
    launch of more than :data:`_CONTEXT_FP8_SPLIT_MAX_WAVES` waves of tiles (a
    pure function of its integers, cached).  The candidates split the tiles
    the idle CTAs of the last wave can take, ``num_splits`` from 2 up,
    ``split_blocks`` the even split of the longest tile; at most ``sm_count -
    1`` tiles split, so the per-tile counters fit in the counter buffer after
    its first word."""

    costs = (
        _context_fp8_split_costs(target, max_q_len, pack)
        if rows
        else _CONTEXT_FP8_SPLIT_COSTS.get(target)
    )
    blocks = _context_fp8_tile_blocks(
        batch_size, max_q_len, max_kv_len, num_q_heads, pack
    )
    tiles = len(blocks)
    if costs is None or not tiles or tiles > _CONTEXT_FP8_SPLIT_MAX_WAVES * sm_count:
        return None
    whole = _context_fp8_split_makespan(blocks, sm_count, 0, 1, 1, costs)
    tail = tiles % sm_count or sm_count
    idle = sm_count - tail
    split_counts = {tiles if tiles <= sm_count else tail, tail}
    split_counts.update(idle // d for d in (1, 2, 3, 4))
    longest = max(blocks)
    best, best_time = None, whole
    for split_tiles in sorted(split_counts):
        split_tiles = min(split_tiles, tiles, sm_count - 1)
        if split_tiles <= 0:
            continue
        for num_splits in range(2, _CONTEXT_FP8_SPLIT_MAX + 1):
            split_blocks = (longest + num_splits - 1) // num_splits
            if (longest + split_blocks - 1) // split_blocks < num_splits:
                continue
            time = _context_fp8_split_makespan(
                blocks, sm_count, split_tiles, split_blocks, num_splits, costs
            )
            if time < best_time:
                best, best_time = (split_tiles, split_blocks, num_splits), time
    return best, best_time, whole


def _context_fp8_split_gain(time: float, faster: float) -> bool:
    """Whether ``faster`` beats ``time`` by both split margins."""

    return (
        faster <= time * (1 - _CONTEXT_FP8_SPLIT_REL_MARGIN)
        and time - faster >= _CONTEXT_FP8_SPLIT_ABS_MARGIN
    )


@functools.lru_cache(maxsize=1024)
def _context_fp8_split_plan(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack: int,
    sm_count: int,
) -> tuple[int, int, int] | None:
    """``(split_tiles, split_blocks, num_splits)`` of the split-KV launch of
    ``pack`` the cost model (:data:`_CONTEXT_FP8_SPLIT_COSTS`) predicts
    fastest (:func:`_context_fp8_split_search`), or ``None`` when no split
    beats the whole-tile schedule by the margins (a pure function of its
    integers, cached)."""

    found = _context_fp8_split_search(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count, False
    )
    if (
        found is None
        or found[0] is None
        or not _context_fp8_split_gain(found[2], found[1])
    ):
        return None
    return found[0]


@functools.lru_cache(maxsize=1024)
def _context_fp8_split_choice(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack: int,
    sm_count: int,
) -> tuple[int, tuple[int, int, int]] | None:
    """The pack and the plan of the split-KV launch of a call whose runtime
    pack is ``pack``, or ``None`` when it does not split (a pure function of
    its integers, cached).

    The plan is ``pack``'s (:func:`_context_fp8_split_plan`), except for a
    launch of one m-block, whose tiles a smaller shipped pack holds in one
    query stage each (fewer query rows per tile: the cost model's units,
    partial stores and merges cost less, :func:`_context_fp8_split_costs`):

    - a launch of two query stages whose plan splits every tile runs the same
      split (``split_blocks`` and ``num_splits``) over the tiles of a smaller
      pack, where predicted faster by the margins: every query row meets the
      same KV units, merged in the same order, so the output is the same;
    - then the launch takes the best split of a smaller pack (its own plan)
      where predicted faster than that by the margins: its rows meet other KV
      units, so the output differs by the rounding of the FP32 merge only."""

    plan = _context_fp8_split_plan(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count
    )
    choice = None if plan is None else (pack, plan)
    tokens = 128 // pack
    costs = _context_fp8_split_costs(target, max_q_len, pack)
    blocks = _context_fp8_tile_blocks(
        batch_size, max_q_len, max_kv_len, num_q_heads, pack
    )
    if (
        costs is None
        or max_q_len > 2 * tokens
        or len(blocks) > _CONTEXT_FP8_SPLIT_MAX_WAVES * sm_count
    ):
        return choice
    smaller_packs = [
        smaller
        for smaller in (*_CONTEXT_FP8_RUNTIME_PACKS, 1)
        if smaller < pack and pack % smaller == 0
    ]
    time = _context_fp8_split_makespan(blocks, sm_count, *(plan or (0, 1, 1)), costs)
    if plan is not None and max_q_len > tokens and plan[0] == len(blocks):
        repack: tuple[int, tuple[int, int, int]] | None = None
        repack_time = time
        for smaller in smaller_packs:
            tiles = _context_fp8_tile_blocks(
                batch_size, max_q_len, max_kv_len, num_q_heads, smaller
            )
            tile_costs = _context_fp8_split_costs(target, max_q_len, smaller)
            if len(tiles) > sm_count - 1 or tile_costs is None:
                continue
            candidate = (len(tiles), plan[1], plan[2])
            candidate_time = _context_fp8_split_makespan(
                tiles, sm_count, *candidate, tile_costs
            )
            if candidate_time < repack_time:
                repack, repack_time = (smaller, candidate), candidate_time
        if repack is not None and _context_fp8_split_gain(time, repack_time):
            choice, time = repack, repack_time
    challenger: tuple[int, tuple[int, int, int]] | None = None
    challenger_time = time
    for smaller in smaller_packs:
        found = _context_fp8_split_search(
            target,
            batch_size,
            max_q_len,
            max_kv_len,
            num_q_heads,
            smaller,
            sm_count,
            True,
        )
        if found is None:
            continue
        own_plan, own_time, _ = found
        if own_plan is not None and own_time < challenger_time:
            challenger, challenger_time = (smaller, own_plan), own_time
    if challenger is not None and _context_fp8_split_gain(time, challenger_time):
        return challenger
    return choice


def _context_fp8_launch_split_plan(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack: int,
    sm_count: int,
) -> tuple[int, int, int] | None:
    """The split plan of a launch of the split-KV program of ``pack``: the
    plan :func:`_context_fp8_split_choice` gives the call where it chose this
    pack, else ``pack``'s own plan (the route kept ``pack`` where the
    workspace could not hold the chosen plan's partials)."""

    runtime_pack = _context_fp8_runtime_pack(
        _context_pack_g(max_q_len, max_kv_len, num_q_heads, num_kv_heads)
    )
    choice = _context_fp8_split_choice(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, runtime_pack, sm_count
    )
    if choice is not None and choice[0] == pack:
        return choice[1]
    return _context_fp8_split_plan(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count
    )


def _context_split_sm_count(device: torch.device) -> int | None:
    """The SM count a launch's split plan assumes, or ``None`` off a CUDA
    device (no split)."""

    return get_device_sm_count(device) if device.type == "cuda" else None


def _context_fp8_split_workspace_bytes(split_tiles: int, num_splits: int) -> int:
    """Workspace bytes of a split plan: the partial O of every split unit, then
    their row maxima and sums."""

    return (
        split_tiles
        * num_splits
        * (_CONTEXT_FP8_SPLIT_O_BYTES + _CONTEXT_FP8_SPLIT_ML_BYTES)
    )


def _context_fp8_split_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    query: torch.Tensor,
    plan: tuple[int, int, int],
) -> bool:
    """Whether the caller's workspace holds the plan's FP32 partials."""

    return (
        workspace_buffer is not None
        and workspace_buffer.is_contiguous()
        and workspace_buffer.device == query.device
        and workspace_buffer.data_ptr() % 16 == 0
        and workspace_buffer.numel() * workspace_buffer.element_size()
        >= _context_fp8_split_workspace_bytes(plan[0], plan[2])
    )


def _context_hd256_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    query: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    fp8: bool,
) -> bool:
    """Mirror the HD256 binding's packed Q/K/V/O and metadata arena."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != query.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False

    padded_q = ((max_q_len + 127) // 128) * 128
    max_micro_pages = (max_kv_len + 15) // 16
    total_q_rows = batch_size * num_q_heads * padded_q
    total_micro_pages = batch_size * num_kv_heads * max_micro_pages
    input_bytes = 256 if fp8 else 512

    cursor = total_q_rows * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_q_rows * 512
    cursor = (cursor + 15) // 16 * 16
    cursor += 3 * batch_size * num_q_heads * 4
    cursor += total_micro_pages * 4
    return workspace_buffer.numel() * workspace_buffer.element_size() >= cursor


def _context_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    key_cache: torch.Tensor,
    *,
    batch_size: int,
    num_q_heads: int,
    pack_g: int,
) -> bool:
    """Bound the fused kernel's expanded-metadata workspace exactly."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != key_cache.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False
    total_bh = batch_size * (num_q_heads // pack_g)
    seq_kv_offset = ((total_bh * 4 + 15) // 16) * 16
    required = ((seq_kv_offset + 2 * total_bh * 4 + 15) // 16) * 16
    return workspace_buffer.numel() * workspace_buffer.element_size() >= required


# How the host selects the one-tile BF16 launch classes.  ``False`` (the
# default): from host-visible facts alone; the programs read every request's KV
# length and query offsets at run time, so any request lengths up to the
# planned maxima are served, during CUDA graph capture too.  ``True``: as
# before the generated programs, the host also reads ``seq_lens`` and
# ``cum_seq_lens_q`` (a device synchronization, skipped during capture) and
# selects the class only when every request has the planned maximum lengths;
# other batches keep the generic JIT route.
_CONTEXT_EXACT_PROFILE_NEEDS_UNIFORM_LENGTHS = False


def _context_bf16_exact_profile(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    kv_layout: str,
) -> Literal["q511", "q257"] | None:
    """Resolve the two one-tile mask-loop programs.

    ``query`` holds ``batch_size x max_q_len`` rows, which makes every request
    ``max_q_len`` tokens long; any KV length up to ``max_kv_len`` is served.
    Unless ``_CONTEXT_EXACT_PROFILE_NEEDS_UNIFORM_LENGTHS``, no tensor value is
    read.
    """

    common = (
        batch_size == 4
        and num_q_heads == 10
        and num_kv_heads == 2
        and is_causal
        and not return_lse
        and not enable_sink
        and kv_layout == "HND"
        and seq_lens.device == query.device
        and cum_seq_lens_q.device == query.device
    )
    if not common:
        return None
    profile: Literal["q511", "q257"]
    if (
        max_q_len == 511
        and max_kv_len == 2047
        and page_size == 32
        and uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 511
    ):
        profile = "q511"
    elif (
        max_q_len == 257
        and max_kv_len == 1024
        and page_size == 1024
        and not uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 257
    ):
        profile = "q257"
    else:
        return None
    if not _CONTEXT_EXACT_PROFILE_NEEDS_UNIFORM_LENGTHS:
        return profile
    if query.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return None
    try:
        kv_lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
        q_indptr = [int(value) for value in cum_seq_lens_q.detach().cpu().tolist()]
    except RuntimeError:
        return None
    if kv_lengths != [max_kv_len] * batch_size:
        return None
    if q_indptr != [index * max_q_len for index in range(batch_size + 1)]:
        return None
    return profile


def select_cake_fmha_context_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    is_causal: bool,
    lse: torch.Tensor | None,
    kv_layout: str = "HND",
    workspace_buffer: torch.Tensor | None = None,
) -> CakeFmhaContextRoute | None:
    """Select an exact product context route without broadening its contract."""

    if batch_size <= 0 or max_q_len <= 0 or max_kv_len <= 0 or window_left != -1:
        return None
    if not _decode_quant_scale_supported(
        bmm1_scale, query
    ) or not _decode_quant_scale_supported(bmm2_scale, query):
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    page_size = int(key_cache.shape[2])
    if page_size not in (16, 32, 64, 128, 256, 512, 1024):
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size:
        return None
    if seq_lens.dtype != torch.int32:
        return None
    if not seq_lens.is_contiguous():
        return None
    for indptr in (cum_seq_lens_q, cum_seq_lens_kv):
        if (
            indptr.ndim != 1
            or indptr.shape[0] != batch_size + 1
            or indptr.dtype != torch.int32
            or not indptr.is_contiguous()
        ):
            return None
    if sinks is not None and (
        sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if sinks is not None and lse is not None:
        return None
    if lse is not None and (
        lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
    ):
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if uses_shared_paged_kv_idx:
        if (
            block_tables.ndim != 2
            or block_tables.shape[0] != batch_size
            or block_tables.stride(1) != 1
        ):
            return None
    elif (
        block_tables.ndim != 3
        or block_tables.shape[:2] != (batch_size, 2)
        or block_tables.stride(2) != 1
    ):
        return None

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None
    host_scalar_scales = not isinstance(bmm1_scale, torch.Tensor) and not isinstance(
        bmm2_scale, torch.Tensor
    )
    component: CakeFmhaContextComponent | None = None
    if (
        dtypes == (torch.bfloat16,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
        and host_scalar_scales
        and float(bmm2_scale) == 1.0
    ):
        component = "context_bf16"
    elif (
        dtypes == (torch.float8_e4m3fn,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
    ):
        component = "context_fp8"
    elif (
        dtypes == (torch.float16,) * 4
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and host_scalar_scales
        and not is_causal
        and sinks is None
        and lse is None
        and float(bmm2_scale) == 1.0
    ):
        component = "context_fp16_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and is_causal
        and sinks is None
        and lse is None
    ):
        component = "context_fp8_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 64)
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and is_causal
        and sinks is None
        and lse is None
        and skip_softmax_threshold_scale_factor in (None, 0.0)
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.shape == value_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.is_contiguous()
        and value_block_scales.is_contiguous()
    ):
        component = "context_nvfp4"
    if component is None:
        return None

    exact_profile = None
    if component == "context_bf16":
        exact_profile = _context_bf16_exact_profile(
            query,
            seq_lens,
            cum_seq_lens_q,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            is_causal=is_causal,
            return_lse=lse is not None,
            enable_sink=sinks is not None,
            kv_layout=kv_layout,
        )
    if exact_profile == "q511":
        pack_g = 5
        num_m_blocks = 11
        l2_swizzle = 1
    elif exact_profile == "q257":
        pack_g = 5
        num_m_blocks = 6
        l2_swizzle = 8
    elif component in ("context_fp16_hd256", "context_fp8_hd256"):
        pack_g = 1
        num_m_blocks = (max_q_len + 127) // 128
        l2_swizzle = 1
    else:
        pack_g = _context_pack_g(max_q_len, max_kv_len, num_q_heads, num_kv_heads)
        tok_per_stage = 128 // pack_g
        num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (2 * tok_per_stage)
        total_bh = batch_size * (num_q_heads // pack_g)
        l2_swizzle = 8 if total_bh % 8 == 0 else 1
    if component == "context_nvfp4" and not _context_nvfp4_workspace_supported(
        workspace_buffer,
        key_cache,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        pack_g=pack_g,
    ):
        return None
    if component in ("context_fp16_hd256", "context_fp8_hd256") and not (
        _context_hd256_workspace_supported(
            workspace_buffer,
            query,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            fp8=component == "context_fp8_hd256",
        )
    ):
        return None
    target = _cake_fmha_target(device)
    device_scales = component in ("context_fp8", "context_nvfp4") and (
        isinstance(bmm1_scale, torch.Tensor) and isinstance(bmm2_scale, torch.Tensor)
    )
    # The generated program serving the route on ``target`` (its key is the
    # route's component, scale form, head counts, pack, m-blocks, page size
    # and exact class), or ``None`` to keep the component's JIT module; the
    # programs are causal and write no LSE and take no sink.
    program = None
    if is_causal and lse is None and sinks is None:
        program = _CONTEXT_PROGRAM_ROUTES.get(target, {}).get(
            (
                component,
                "device" if device_scales else "host",
                num_q_heads,
                num_kv_heads,
                pack_g,
                num_m_blocks,
                page_size,
                exact_profile,
            )
        )
        # An FP8 call no exact route serves: the runtime-size program of its
        # pack and page size, where it is not slower than the JIT module.
        runtime = _CONTEXT_FP8_RUNTIME_ROUTES.get(target)
        if (
            program is None
            and component == "context_fp8"
            and runtime
            and _context_fp8_runtime_serves(
                target,
                batch_size,
                max_q_len,
                max_kv_len,
                num_q_heads,
                num_kv_heads,
                pack_g,
            )
        ):
            pack = _context_fp8_runtime_pack(pack_g)
            program = runtime.get((pack, page_size)) or runtime.get((pack, None))
            if program is not None:
                pack_g = pack
                tok_per_stage = 128 // pack
                num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (
                    2 * tok_per_stage
                )
                total_bh = batch_size * (num_q_heads // pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
        # An FP8 call an exact or runtime-size program serves, whose tiles
        # leave SMs idle: the split-KV program of the pack the split choice
        # names (:func:`_context_fp8_split_choice`) and the page size, where the
        # split plan predicts a gain and the workspace holds its partials; the
        # runtime pack's own plan, where it has one, when the workspace cannot
        # hold the chosen pack's partials; else the call keeps its program.
        split = _CONTEXT_FP8_SPLIT_ROUTES.get(target)
        sm_count = _context_split_sm_count(device)
        if program is not None and component == "context_fp8" and split and sm_count:
            pack = _context_fp8_runtime_pack(pack_g)
            choice = _context_fp8_split_choice(
                target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count
            )
            candidates = [] if choice is None else [choice]
            if choice is not None and choice[0] != pack:
                own = _context_fp8_split_plan(
                    target,
                    batch_size,
                    max_q_len,
                    max_kv_len,
                    num_q_heads,
                    pack,
                    sm_count,
                )
                if own is not None:
                    candidates.append((pack, own))
            for pack, plan in candidates:
                split_program = split.get((pack, page_size)) or split.get((pack, None))
                if split_program is None or not _context_fp8_split_workspace_supported(
                    workspace_buffer, query, plan
                ):
                    continue
                program = split_program
                pack_g = pack
                tok_per_stage = 128 // pack
                num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (
                    2 * tok_per_stage
                )
                total_bh = batch_size * (num_q_heads // pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
                break
    return CakeFmhaContextRoute(
        target=target,
        component=component,
        num_m_blocks=num_m_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        pack_g=pack_g,
        page_size=page_size,
        l2_swizzle=l2_swizzle,
        is_causal=is_causal,
        return_lse=lse is not None,
        enable_sink=sinks is not None,
        exact_profile=exact_profile,
        device_scales=device_scales,
        program=program,
    )


class CakeFmhaContextProgramModule:
    """One loaded generated context program behind the context FFI ABI
    (``cake_paged_attention_context``, the arguments ``trtllm_batch_context_
    with_kv_cache`` passes): the route it serves, the ``CakeProgram`` record,
    its exact target and the binding's ``run`` entry."""

    def __init__(self, route: CakeFmhaContextRoute, program: Any, target: str):
        self.route = route
        self.program = program
        self.target = target
        self.entry = program.entry(target)
        self.units_per_batch = route.num_q_heads // route.pack_g
        self.quantized = route.component != "context_bf16"
        _, self.num_m_blocks, self.one_tile = _context_program_facts(program.name)
        # A runtime-size program serves every head count, page size and query
        # length of its pack (the route's): the launch reads them off its
        # tensors and lengths, and the scale form off its scale arguments.
        self.runtime = self.num_m_blocks is None
        self.pack_g = route.pack_g
        self.m_block_tokens = 2 * (128 // route.pack_g)
        # A split-KV program also takes the split plan of each launch
        # (:func:`_context_fp8_launch_split_plan`), its partials in the caller's
        # workspace and its tile counters after the counter buffer's first word.
        self.split = route.program in _CONTEXT_FP8_SPLIT_PROGRAMS

    def cake_paged_attention_context(
        self,
        out: torch.Tensor,
        out_scale_factor: torch.Tensor | None,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        multi_ctas_kv_counter_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float | torch.Tensor,
        bmm2_scale: float | torch.Tensor,
        o_sf_scale: float,
        o_sf_vec_size: int,
        o_sf_start_index: int,
        batch_size: int,
        window_left: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        sm_count: int,
        enable_pdl: bool | None,
        workspace_size: int,
        attention_sinks: torch.Tensor | None,
        key_block_scales: torch.Tensor | None,
        value_block_scales: torch.Tensor | None,
        skip_softmax_threshold_scale_factor: float | None,
        uses_shared_paged_kv_idx: bool,
        use_fp16_softmax: bool,
        uses_spcompress: bool,
        is_causal: bool,
        lse: torch.Tensor | None,
        lse_stride_tokens: int,
        lse_stride_heads: int,
    ) -> None:
        """Launch the program on the current torch stream.

        The kernel reads every request's KV length (``seq_lens``) and query
        offsets (``cum_seq_lens_q``) itself; ``total_bh`` is the work-unit
        count ``batch_size x num_q_heads / pack_g``.  A device-scale program
        reads the raw bmm1 scale and the bmm2 scale through device
        pointers; a host-scalar program takes them as scalars and never reads
        the pointers.  The LSE, sink and Q-length pointers are unread (the
        programs write no LSE, take no sink and derive the query lengths from
        ``cum_seq_lens_q``): they point at a small per-device buffer.  A
        runtime-size program also takes the scale source, the m-block count of
        ``max_q_len``, the head counts, the GQA group and the log2 page size of
        this launch.
        """

        route = self.route
        if lse is not None or attention_sinks is not None:
            # route selection keeps such calls on the JIT module: the unread
            # pointers below never stand in for a caller's LSE or sinks
            raise RuntimeError(
                "a Cake FMHA context program writes no LSE and takes no sinks"
            )
        if self.runtime:
            num_q_heads = query.shape[1]
            units_per_batch = num_q_heads // self.pack_g
            num_m_blocks = (int(max_q_len) + self.m_block_tokens - 1) // (
                self.m_block_tokens
            )
            # the scale form of this call (one program serves both)
            device_scales = isinstance(bmm1_scale, torch.Tensor)
        elif query.shape[1] != route.num_q_heads or key_cache.shape[1] != (
            route.num_kv_heads
        ):
            raise RuntimeError(
                "Cake FMHA context launch does not match its selected route"
            )
        else:
            units_per_batch = self.units_per_batch
            num_m_blocks = self.num_m_blocks
            device_scales = route.device_scales
        total_bh = int(batch_size) * units_per_batch
        tiles = num_m_blocks * total_bh
        num_ctas = tiles if self.one_tile else min(int(sm_count), tiles)
        device = query.device
        counter = multi_ctas_kv_counter_buffer
        if counter.dim() != 1:
            counter = counter.view(-1)
        if counter.dtype != torch.uint32:
            counter = counter.view(torch.uint32)
        if self.split:
            # The plan of this launch; one that the workspace or the counter
            # buffer cannot hold runs as one unit per tile (the whole-tile
            # schedule, which reads neither).
            plan = _context_fp8_launch_split_plan(
                route.target,
                int(batch_size),
                int(max_q_len),
                int(max_kv_len),
                num_q_heads,
                int(key_cache.shape[1]),
                self.pack_g,
                int(sm_count),
            )
            if (
                plan is None
                or counter.numel() < 1 + plan[0]
                or not _context_fp8_split_workspace_supported(
                    workspace_buffer, query, plan
                )
            ):
                plan = (1, 1, 1)
            split_tiles, split_blocks, num_splits = plan
            num_ctas = min(int(sm_count), tiles + (num_splits - 1) * split_tiles)
            if num_splits > 1:
                partials = workspace_buffer.view(-1).view(torch.uint8)
                o_bytes = split_tiles * num_splits * _CONTEXT_FP8_SPLIT_O_BYTES
                ml_bytes = split_tiles * num_splits * _CONTEXT_FP8_SPLIT_ML_BYTES
                partial_o = partials[:o_bytes].view(torch.float32)
                partial_ml = partials[o_bytes : o_bytes + ml_bytes].view(torch.float32)
                tile_counters = counter[1 : 1 + split_tiles]
            else:
                partial_o = partial_ml = _unread_pointers(device)[0]
                tile_counters = counter[:1]
        unread_f32, unread_i32 = _unread_pointers(device)
        # The kernel reads page ``p`` of request ``b`` at ``page_table[b x
        # page_row_stride + p]``: pass 1-D views over the caller's own table
        # storage (no copy) that span every row, and its true row stride.
        tables = (
            block_tables
            if block_tables.dtype == torch.int32
            else block_tables.view(torch.int32)
        )
        strides = tables.stride()
        if strides[-1] != 1:
            raise RuntimeError(
                "Cake FMHA context page tables need a unit stride along the pages"
            )
        row_stride = strides[0]
        if uses_shared_paged_kv_idx and tables.is_contiguous():
            # the binding reads only the table's address: a contiguous table
            # is its own span
            page_table_k = page_table_v = tables
        else:
            shape = tables.shape
            span = (shape[0] - 1) * row_stride + shape[-1]
            offset = tables.storage_offset()
            page_table_k = tables.as_strided((span,), (1,), offset)
            page_table_v = (
                page_table_k
                if uses_shared_paged_kv_idx
                else tables.as_strided((span,), (1,), offset + strides[1])
            )
        if device_scales:
            # bmm1 arrives as the caller's device scale (not in log2 form):
            # with bmm1_is_log2 = 0 the program forms the FP32 product
            # bmm1 * log2e itself, the value the host multiply gave.
            if not isinstance(bmm1_scale, torch.Tensor) or not isinstance(
                bmm2_scale, torch.Tensor
            ):
                raise RuntimeError(
                    "a device-scale Cake FMHA context program takes tensor scales"
                )
            softmax_scale_log2, output_scale, bmm1_is_log2 = 1.0, 1.0, 0
            bmm1_ptr = bmm1_scale if bmm1_scale.dim() == 1 else bmm1_scale.reshape(1)
            bmm2_ptr = bmm2_scale if bmm2_scale.dim() == 1 else bmm2_scale.reshape(1)
        else:
            softmax_scale_log2 = float(bmm1_scale) * log2e
            output_scale, bmm1_is_log2 = float(bmm2_scale), 0
            bmm1_ptr = bmm2_ptr = unread_f32
        # The arguments in the binding's order: ``_CONTEXT_HOST_PARAMS`` of the
        # component (LSE_ptr, sinks and seq_lens_q unread; a runtime-size
        # program's sizes after them), then the grid; the order is checked
        # against the record at load.
        if not self.quantized:
            arguments: tuple[Any, ...] = (
                query,
                key_cache,
                value_cache,
                out,
                unread_f32,
                unread_f32,
                page_table_k,
                page_table_v,
                unread_i32,
                seq_lens,
                cum_seq_lens_q,
                softmax_scale_log2,
                total_bh,
                row_stride,
                num_ctas,
                counter,
                num_ctas,
                1,
                1,
            )
        elif route.component == "context_nvfp4":
            arguments = (
                query.view(torch.uint8),
                key_cache,
                value_cache,
                key_block_scales.view(torch.uint8),
                value_block_scales.view(torch.uint8),
                out.view(torch.uint8),
                unread_f32,
                unread_f32,
                page_table_k,
                page_table_v,
                unread_i32,
                seq_lens,
                cum_seq_lens_q,
                softmax_scale_log2,
                output_scale,
                total_bh,
                row_stride,
                num_ctas,
                counter,
                bmm1_ptr,
                bmm2_ptr,
                bmm1_is_log2,
                num_ctas,
                1,
                1,
            )
        elif self.runtime:
            arguments = (
                query.view(torch.uint8),
                key_cache.view(torch.uint8),
                value_cache.view(torch.uint8),
                out.view(torch.uint8),
                unread_f32,
                unread_f32,
                page_table_k,
                page_table_v,
                unread_i32,
                seq_lens,
                cum_seq_lens_q,
                softmax_scale_log2,
                output_scale,
                total_bh,
                row_stride,
                num_ctas,
                counter,
                bmm1_ptr,
                bmm2_ptr,
                bmm1_is_log2,
                1 if device_scales else 0,
                num_m_blocks,
                total_bh // 8,
                units_per_batch,
                num_q_heads,
                num_q_heads // key_cache.shape[1],
                key_cache.shape[2].bit_length() - 1,
            )
            if self.split:
                arguments = (
                    *arguments,
                    split_tiles,
                    split_blocks,
                    num_splits,
                    partial_o,
                    partial_ml,
                    tile_counters,
                )
            arguments = (*arguments, num_ctas, 1, 1)
        else:
            arguments = (
                query.view(torch.uint8),
                key_cache.view(torch.uint8),
                value_cache.view(torch.uint8),
                out.view(torch.uint8),
                unread_f32,
                unread_f32,
                page_table_k,
                page_table_v,
                unread_i32,
                seq_lens,
                cum_seq_lens_q,
                softmax_scale_log2,
                output_scale,
                total_bh,
                row_stride,
                num_ctas,
                counter,
                bmm1_ptr,
                bmm2_ptr,
                bmm1_is_log2,
                num_ctas,
                1,
                1,
            )
        if _ffi_call_takes_the_torch_stream():
            self.entry(*arguments)
        else:
            with _torch_stream_context(device):
                self.entry(*arguments)


@functools.cache
def _ffi_call_takes_the_torch_stream() -> bool:
    """Whether a tvm-ffi call runs on torch's current stream by itself.

    With the DLPack exchange API on torch tensors (``__dlpack_c_exchange_api__``,
    native or installed by ``import tvm_ffi``), the call sets the FFI stream of
    its first CUDA tensor argument's device to torch's current stream there for
    the call's duration, as for FlashInfer's JIT modules; otherwise the launch
    sets it itself (:func:`_torch_stream_context`).
    """

    return getattr(torch.Tensor, "__dlpack_c_exchange_api__", None) is not None


_FFI_DEVICES: dict[int, Any] = {}
_UNREAD_POINTERS: dict[torch.device, tuple[Any, Any, torch.Tensor]] = {}


def _unread_pointers(device: torch.device) -> tuple[Any, Any]:
    """A 16-byte buffer on ``device`` as float32 and int32 FFI tensors, for
    the binding arguments a context program never reads.

    The buffer is allocated and converted to FFI tensors once per device: an
    argument that already is an FFI tensor skips the per-call DLPack
    conversion a torch tensor argument costs (up to five per launch).
    """

    pointers = _UNREAD_POINTERS.get(device)
    if pointers is None:
        buffer = torch.empty(4, dtype=torch.float32, device=device)
        pointers = _UNREAD_POINTERS[device] = (
            tvm_ffi.from_dlpack(buffer),
            tvm_ffi.from_dlpack(buffer.view(torch.int32)),
            buffer,
        )
    return pointers[0], pointers[1]


def _torch_stream_context(device: torch.device):
    """The FFI stream context on the current torch stream of ``device``.

    The binding selects its device from its tensor arguments and launches on
    the FFI stream of that device; this sets that stream to torch's current
    one (a CUDA graph's capture stream during capture).  The raw stream handle
    and a cached FFI device cost a fraction of ``torch.cuda.device`` plus
    ``tvm_ffi.use_torch_stream()``, which builds a ``torch.cuda.Stream`` and
    parses a device string on every call.
    """

    raw_stream_of = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if raw_stream_of is None:
        return _DeviceTorchStream(index)
    ffi_device = _FFI_DEVICES.get(index)
    if ffi_device is None:
        ffi_device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return tvm_ffi.use_raw_stream(ffi_device, raw_stream_of(index))


class _DeviceTorchStream:
    """``torch.cuda.device(index)`` and ``tvm_ffi.use_torch_stream()``."""

    def __init__(self, index: int):
        self.device = torch.cuda.device(index)
        self.stream = tvm_ffi.use_torch_stream()

    def __enter__(self) -> None:
        self.device.__enter__()
        self.stream.__enter__()

    def __exit__(self, *args: Any) -> None:
        self.stream.__exit__(*args)
        self.device.__exit__(*args)


# The NVFP4 context programs dequantize their E2M1 pages with the PTX ISA 9.4
# packed multiply ``mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4`` (CUDA 13.4 or
# newer); an older nvcc / ptxas rejects the instruction, so on such a toolkit
# a route to one of them fails closed to compat_v1 instead of a build error.
CONTEXT_NVFP4_PROGRAMS_MIN_CUDA = "13.4"


def context_nvfp4_programs_unavailable() -> str | None:
    """Why this CUDA toolkit cannot build the NVFP4 context programs, or ``None``."""

    from .jit.cpp_ext import get_cuda_version, is_cuda_version_at_least

    if is_cuda_version_at_least(CONTEXT_NVFP4_PROGRAMS_MIN_CUDA):
        return None
    return (
        "the generated NVFP4 context programs spell the E2M1 x E4M3 page "
        "dequantization as the PTX ISA 9.4 packed multiply "
        "(mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4), which needs CUDA "
        f"{CONTEXT_NVFP4_PROGRAMS_MIN_CUDA} or newer; the CUDA toolkit found is "
        f"{get_cuda_version()}"
    )


_CONTEXT_PROGRAM_MODULES: dict[tuple[str, str, str], CakeFmhaContextProgramModule] = {}


def _load_cake_fmha_context_program(
    route: CakeFmhaContextRoute,
) -> CakeFmhaContextProgramModule:
    """Build and load the generated program of ``route`` on the route's target.

    The loaders ``flashinfer.jit.cake_fmha_context_{bf16,fp8,nvfp4}`` are
    generated by the Cake export beside ``csrc/cake_fmha_context_*/``; they are
    imported here, lazily, so route selection never needs them.  The record's
    name, target and argument names are checked against the host once, at
    load.
    """

    key = (route.target, route.component, route.program)
    module = _CONTEXT_PROGRAM_MODULES.get(key)
    if module is not None:
        return module
    import importlib

    from .jit.cake_program import GRID_ARGS, CakeProgram

    pinned = _CONTEXT_PROGRAMS.get((route.component, route.program))
    if pinned is None:
        raise RuntimeError(
            f"Cake FMHA {route.component} names no generated program {route.program!r}"
        )
    if route.component == "context_nvfp4":
        unavailable = context_nvfp4_programs_unavailable()
        if unavailable is not None:
            raise RuntimeError(f"Cake FMHA cannot build {pinned[0]}: {unavailable}")
    loader_name = f"cake_fmha_{route.component}"
    loader = importlib.import_module(f".jit.{loader_name}", __package__)
    program = getattr(loader, route.program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.{loader_name} publishes no CakeProgram {route.program!r}"
        )
    if program.name != pinned[0]:
        raise RuntimeError(
            f"flashinfer.jit.{loader_name}.{route.program} is {program.name}; "
            f"this module pins {pinned[0]}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    params = _context_program_params(route.component, pinned[1], route.program)
    if tuple(program.args) != (*params, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*params, *GRID_ARGS]}"
        )
    module = CakeFmhaContextProgramModule(route, program, target)
    _CONTEXT_PROGRAM_MODULES[key] = module
    return module


def get_cake_fmha_context_module(
    device: torch.device, route: CakeFmhaContextRoute | None
):
    """Load an optimized context module, or the authenticated portable fallback."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA context route target does not match the device")
    if route.program is not None:
        if route.component == "context_nvfp4":
            unavailable = context_nvfp4_programs_unavailable()
            if unavailable is not None:
                warnings.warn(
                    f"Cake FMHA NVFP4 context program {route.program} failed closed "
                    f"to compat_v1: {unavailable}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return load_cake_fmha_compat_module(route.target)
        return _load_cake_fmha_context_program(route)
    if route.exact_profile is not None:
        raise RuntimeError(
            f"Cake FMHA context exact profile {route.exact_profile} has no program "
            f"on {route.target}"
        )
    if route.component == "context_fp16_hd256":
        return load_cake_fmha_context_fp16_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_fp8_hd256":
        return load_cake_fmha_context_fp8_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_nvfp4":
        try:
            return load_cake_fmha_context_nvfp4_module(
                route.target,
                route.num_m_blocks,
                route.num_q_heads,
                route.num_kv_heads,
                route.pack_g,
                route.page_size,
                route.l2_swizzle,
            )
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA NVFP4 context loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target)
    common_args = (
        route.target,
        route.num_m_blocks,
        route.num_q_heads,
        route.num_kv_heads,
        route.pack_g,
        route.page_size,
        route.l2_swizzle,
    )
    common_kwargs = {
        "is_causal": route.is_causal,
        "return_lse": route.return_lse,
        "enable_sink": route.enable_sink,
    }
    if route.component == "context_bf16":
        return load_cake_fmha_context_bf16_module(*common_args, **common_kwargs)
    return load_cake_fmha_context_fp8_module(*common_args, **common_kwargs)


def cake_fmha_manifest() -> dict[str, Any]:
    """Return a copy of the authenticated product/capability manifest."""

    return copy.deepcopy(get_cake_fmha_manifest())


def cake_batch_decode_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-decode ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_decode_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's default backend selection.
    """

    from .decode import trtllm_batch_decode_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_decode_with_kv_cache requires backend='cake'")
    return trtllm_batch_decode_with_kv_cache(*args, backend="cake", **kwargs)


def cake_batch_context_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-context ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_context_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's conventional default.
    """

    from .prefill import trtllm_batch_context_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_context_with_kv_cache requires backend='cake'")
    return trtllm_batch_context_with_kv_cache(*args, backend="cake", **kwargs)


__all__ = [
    "CakeFmhaContextRoute",
    "CakeFmhaDecodeRoute",
    "CakeFmhaRequestOrderedDecodePlan",
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "cake_fmha_route_is_optimized",
    "get_cake_fmha_context_module",
    "get_cake_fmha_decode_module",
    "get_cake_fmha_module",
    "plan_cake_fmha_request_ordered_paged_decode",
    "select_cake_fmha_context_route",
    "select_cake_fmha_decode_route",
]
