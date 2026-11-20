"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  DCP
metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
import functools
import heapq
import os
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
    load_cake_fmha_stream_context_module,
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
        "decode_aligned_bf16_hd64",
        "decode_aligned_bf16_hd256",
        "decode_aligned_bf16_mtp_q32",
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
    # cluster size ``aligned_cluster_size`` selected; aligned hd64 q1 decode: the
    # one identifier ``flashinfer.jit.cake_fmha_aligned_hd64.ALIGNED_HD64_BF16``
    # that ``aligned_hd64_selected`` chose; aligned hd256 q1 decode: the
    # identifier (``flashinfer.jit.cake_fmha_aligned_hd256.ALIGNED_HD256_BF16`` /
    # ``ALIGNED_HD256_CGA2_BF16`` / ``ALIGNED_HD256_CGA4_BF16`` /
    # ``ALIGNED_HD256_CGA8_BF16``) of the program
    # ``aligned_hd256_program`` named; aligned speculative decode (1 to 4 query
    # tokens per request): the identifier
    # ``flashinfer.jit.cake_fmha_aligned_mtp_q32.ALIGNED_MTP_Q32_CGA2_BF16`` of the
    # program ``spec_q32_cga_program`` named; ``None`` on every other route.
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


# The aligned hd64 q1 decode program is likewise a generated in-tree unit,
# ``csrc/cake_fmha_aligned_hd64/`` with the loader
# ``flashinfer.jit.cake_fmha_aligned_hd64``: one BF16 program shipped for sm_100a
# and sm_103a, one persistent CTA per (request, KV head) whose 8-row query tile
# holds the KV head's eight query heads.  Its route's component chain is the one
# program identifier below; the table pins the delivered record's name and a
# loader whose record differs fails closed at load.
_ALIGNED_HD64_COMPONENT = "decode_aligned_bf16_hd64"
_ALIGNED_HD64_PROGRAM_ID = "ALIGNED_HD64_BF16"
_ALIGNED_HD64_PROGRAMS: dict[str, str] = {
    _ALIGNED_HD64_PROGRAM_ID: "cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26",
}
# The binding's host parameters, in the kernel's order (then the grid).
_ALIGNED_HD64_HOST_PARAMS = (
    "Q",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "heads_per_kv",
    "batch_size",
    "skip_corr_threshold",
)
# The kernel's correction-skip threshold: 0 disables the skip (the value every
# validation of the program used).
_ALIGNED_HD64_SKIP_CORR_THRESHOLD = 0.0


# The aligned hd256 q1 decode programs are likewise a generated in-tree unit,
# ``csrc/cake_fmha_aligned_hd256/`` with the loader
# ``flashinfer.jit.cake_fmha_aligned_hd256``: four BF16 programs shipped for
# sm_100a and sm_103a whose 8-row query tile holds a KV head's eight query
# heads.  The persistent program runs one CTA per (request, KV head) tile and
# feeds its persistent loop through cluster launch control; the cluster
# programs split each tile's KV sequence over a cluster of 2, 4 or 8 CTAs and
# merge the partials through distributed shared memory.
# ``aligned_hd256_program`` names a launch's program as ``(kind, CTAs per
# tile)``; the route's component chain is that program's identifier below.  The
# table pins each delivered record's name and a loader whose record differs
# fails closed at load.
_ALIGNED_HD256_COMPONENT = "decode_aligned_bf16_hd256"
# ``aligned_hd256_program`` verdict -> the program identifier.
_ALIGNED_HD256_PROGRAM_IDS: dict[tuple[str, int], str] = {
    ("persistent", 1): "ALIGNED_HD256_BF16",
    ("cga", 2): "ALIGNED_HD256_CGA2_BF16",
    ("cga", 4): "ALIGNED_HD256_CGA4_BF16",
    ("cga", 8): "ALIGNED_HD256_CGA8_BF16",
}
# Program identifier -> (CTAs per (request, KV head) tile, program name): the
# CTAs are the launch grid's x extent and the record's cluster ``(CTAs, 1, 1)``.
_ALIGNED_HD256_PROGRAMS: dict[str, tuple[int, str]] = {
    "ALIGNED_HD256_BF16": (1, "cake_fmha_aligned_hd256_50326788c79371d913ba"),
    "ALIGNED_HD256_CGA2_BF16": (2, "cake_fmha_aligned_hd256_3d8340c8d395be9c7587"),
    "ALIGNED_HD256_CGA4_BF16": (4, "cake_fmha_aligned_hd256_7ba007d1a2873ce28e12"),
    "ALIGNED_HD256_CGA8_BF16": (8, "cake_fmha_aligned_hd256_dc665a26118a54d4ccb7"),
}
# The bindings' host parameters, in the kernels' order (then the grid); the
# four programs share them.
_ALIGNED_HD256_HOST_PARAMS = (
    "Q",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "heads_per_kv",
    "batch_size",
    "skip_corr_threshold",
)
# The kernels' correction-skip threshold: 0 disables the skip (the value every
# validation of the programs used).
_ALIGNED_HD256_SKIP_CORR_THRESHOLD = 0.0


# The aligned speculative-decode (MTP verify) program is likewise a generated
# in-tree unit, ``csrc/cake_fmha_aligned_mtp_q32/`` with the loader
# ``flashinfer.jit.cake_fmha_aligned_mtp_q32``: one BF16 program shipped for
# sm_100a and sm_103a whose 32-row query tile holds the 1 to 4 query tokens of a
# KV head's eight query heads (row ``token x 8 + head``), so a request's KV head
# is read once for every one of its tokens.  The program splits each (request,
# KV head) tile's KV sequence over a cluster of 2 CTAs and merges the partials
# through distributed shared memory, under the causal speculative-decode mask
# (token ``j`` of a request of ``L`` tokens attends the first
# ``L - (q_len - 1 - j)``).  ``spec_q32_cga_program`` names a launch's program as
# ``("cga", CTAs per tile)``; the route's component chain is that program's
# identifier below.  The table pins the delivered record's name and a loader
# whose record differs fails closed at load.
_ALIGNED_MTP_Q32_COMPONENT = "decode_aligned_bf16_mtp_q32"
# ``spec_q32_cga_program`` verdict -> the program identifier.
_ALIGNED_MTP_Q32_PROGRAM_IDS: dict[tuple[str, int], str] = {
    ("cga", 2): "ALIGNED_MTP_Q32_CGA2_BF16",
}
# Program identifier -> (CTAs per (request, KV head) tile, program name): the
# CTAs are the launch grid's x extent and the record's cluster ``(CTAs, 1, 1)``.
_ALIGNED_MTP_Q32_PROGRAMS: dict[str, tuple[int, str]] = {
    "ALIGNED_MTP_Q32_CGA2_BF16": (2, "cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503"),
}
# The binding's host parameters, in the kernel's order (then the grid).
_ALIGNED_MTP_Q32_HOST_PARAMS = (
    "Q",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "heads_per_kv",
    "batch_size",
    "seq_len_q",
    "skip_corr_threshold",
)
# The kernel's correction-skip threshold: 0 disables the skip (the value every
# validation of the program used).
_ALIGNED_MTP_Q32_SKIP_CORR_THRESHOLD = 0.0


# The generated paged context programs: the units ``csrc/cake_fmha_context_bf16/``,
# ``csrc/cake_fmha_context_fp8/`` and ``csrc/cake_fmha_context_nvfp4/`` whose
# loaders ``flashinfer.jit.cake_fmha_context_{bf16,fp8,nvfp4}`` publish one
# ``CakeProgram`` constant per program.  Every program of these tables is
# causal, without LSE or sink (the FP8 programs of the other forms have their
# own table, ``_CONTEXT_FP8_RUNTIME_FORM_ROUTES``).  Per target, a route key ``(component, scale form, num_q_heads,
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
        ("context_fp8", "host", 32, 4, 8, 1, 64, None): "B256_PX",
        ("context_fp8", "device", 32, 4, 8, 1, 64, None): "B256_SCALE_PTR_PX",
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
        "cake_fmha_context_bf16_4846f01db726c36dfe88",
        256,
        False,
    ),
    ("context_bf16", "B128"): ("cake_fmha_context_bf16_3a71b4d96d0cdfb0513e", 2, False),
    ("context_bf16", "Q511_SM100A"): (
        "cake_fmha_context_bf16_4621c524ebfd2b2bb926",
        11,
        True,
    ),
    ("context_bf16", "Q511_SM103A"): (
        "cake_fmha_context_bf16_ea4e4fc120a1a50b6101",
        11,
        True,
    ),
    ("context_bf16", "Q257_SM100A"): (
        "cake_fmha_context_bf16_0dacafa48890bec55adf",
        6,
        True,
    ),
    ("context_bf16", "Q257_SM103A"): (
        "cake_fmha_context_bf16_a7a7514ecb102296c143",
        6,
        True,
    ),
    ("context_bf16", "RUNTIME_PACK1"): (
        "cake_fmha_context_bf16_0e97d38f12331ab6bf15",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2"): (
        "cake_fmha_context_bf16_2fd6b5d88ab61e1b00cd",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4"): (
        "cake_fmha_context_bf16_a43572c3d68d52af005e",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5"): (
        "cake_fmha_context_bf16_b9fcf2cc30ac830a5612",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6"): (
        "cake_fmha_context_bf16_332d799e56da647e1489",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7"): (
        "cake_fmha_context_bf16_99afb64fff4548ad0af8",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8"): (
        "cake_fmha_context_bf16_f7d40147af812ca23526",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12"): (
        "cake_fmha_context_bf16_de6d5f8c9e69623b5baa",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16"): (
        "cake_fmha_context_bf16_7c076f3c4a8dd7a6ab97",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_SINK"): (
        "cake_fmha_context_bf16_482e2382bb911034ae07",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_SINK"): (
        "cake_fmha_context_bf16_d7f958d0e6f32932b91f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_SINK"): (
        "cake_fmha_context_bf16_927f70849dcb9b7d659b",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_SINK"): (
        "cake_fmha_context_bf16_58802d18c1f0ff9d2f7d",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_SINK"): (
        "cake_fmha_context_bf16_6b8dc5e5a06de7b8ddf4",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_SINK"): (
        "cake_fmha_context_bf16_19aaac13890ddbaad3ca",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_SINK"): (
        "cake_fmha_context_bf16_2b9bc5919f2729955ba7",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_SINK"): (
        "cake_fmha_context_bf16_db6a1c7757541d45cd22",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16_SINK"): (
        "cake_fmha_context_bf16_add546aaeef4cc5f7f39",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_LSE"): (
        "cake_fmha_context_bf16_ddb09abf5f450f3b2228",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_LSE"): (
        "cake_fmha_context_bf16_3c3bd6700072f0fd45c2",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_LSE"): (
        "cake_fmha_context_bf16_0d435a49bb7c76d4b562",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_LSE"): (
        "cake_fmha_context_bf16_add5b0599a64c23d3fc2",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_LSE"): (
        "cake_fmha_context_bf16_22ad59fa8cff2bfc1647",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_LSE"): (
        "cake_fmha_context_bf16_b488ff4071059d04dc31",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_LSE"): (
        "cake_fmha_context_bf16_cabb262616c2401896b6",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_LSE"): (
        "cake_fmha_context_bf16_15636ff97b7b1de20471",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16_LSE"): (
        "cake_fmha_context_bf16_5164ef76a00d85d4bdd2",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_33dc9c46c95d308f026f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_c8ee4df85b4478b5cc7f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_bd379ba1e19a94494056",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_01d2dcb2e9c3bcc2d0e6",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_a8e4f9470fb696be4246",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_e124cfe4d3609e987f03",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_097f936e2ae96a771e97",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_SINK_SEQ_TAIL"): (
        "cake_fmha_context_bf16_1b8e45773f26db7aaa3c",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_f830493c5d628354ce1e",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_f4e29d546225e9728953",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_0356530db188eb569c53",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_2877961b650239d3821e",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_5e6775d4538ca8de016d",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_9fb2ae574a87a3cca205",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_cf498c5791a9c1294ddc",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_LSE_SEQ_TAIL"): (
        "cake_fmha_context_bf16_ef6955e1a812ecf793c1",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_NONCAUSAL"): (
        "cake_fmha_context_bf16_b85606b52eadf25563c1",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_NONCAUSAL"): (
        "cake_fmha_context_bf16_a2582fa57aa043587f87",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_NONCAUSAL"): (
        "cake_fmha_context_bf16_fda0975ed77ec050db27",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_NONCAUSAL"): (
        "cake_fmha_context_bf16_b37a0a81098560353b6a",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_NONCAUSAL"): (
        "cake_fmha_context_bf16_6375841ae49a636c540c",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_NONCAUSAL"): (
        "cake_fmha_context_bf16_dcbe0498011dc9d8b042",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_NONCAUSAL"): (
        "cake_fmha_context_bf16_622790b157a41f99256e",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_NONCAUSAL"): (
        "cake_fmha_context_bf16_71fb277d88964bff260f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16_NONCAUSAL"): (
        "cake_fmha_context_bf16_907e68c861f4e2898b4c",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_62e0f39c39a59eebb3e6",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_79d0333089a0d28374d4",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_af38e50e15184bc44799",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_126a07604bcc1bd2a126",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_676842ac8d64cf652446",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_2b81c131ad0c9a86a93d",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_6ae60b0708827ddfa82f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_078e5b8c33befc9bc85a",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16_NONCAUSAL_SINK"): (
        "cake_fmha_context_bf16_5607821f28f71bfa4cc1",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK1_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_86d6f1082dedbeccfcdd",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK2_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_6e8ec4e524ae9d6b3e5d",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK4_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_995dea83caaca8f8b5b9",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK5_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_f5a6fb25be32f892b13f",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK6_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_bc5e84d155d847ca7767",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK7_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_abe24aa28fe5f3fa80b0",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK8_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_1f1d7f69b7f798da463e",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK12_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_8417d51c012a393dfb78",
        None,
        False,
    ),
    ("context_bf16", "RUNTIME_PACK16_NONCAUSAL_LSE"): (
        "cake_fmha_context_bf16_8190b799b7e19b7b2103",
        None,
        False,
    ),
    ("context_fp8", "B256"): ("cake_fmha_context_fp8_d910d4336baa2166efb0", 1, False),
    ("context_fp8", "B256_SCALE_PTR"): (
        "cake_fmha_context_fp8_7e4d0a44e40dc1f1247c",
        1,
        False,
    ),
    ("context_fp8", "B256_PX"): (
        "cake_fmha_context_fp8_4782f3647c66d51d7afd",
        1,
        False,
    ),
    ("context_fp8", "B256_SCALE_PTR_PX"): (
        "cake_fmha_context_fp8_a2998bcf7a6a1882f77e",
        1,
        False,
    ),
    # sm103a, launches of at most 256 keys (_CONTEXT_FP8_SHORT_KV_ROUTES): the
    # 4-stage ring with each stage's P tile in its own S columns.
    ("context_fp8", "B256_KV4"): (
        "cake_fmha_context_fp8_6371b7f525b7e1adee0c",
        1,
        False,
    ),
    ("context_fp8", "B256_SCALE_PTR_KV4"): (
        "cake_fmha_context_fp8_87833354af3317307113",
        1,
        False,
    ),
    ("context_nvfp4", "Q511_SM100A"): (
        "cake_fmha_context_nvfp4_41e0f7b224fcc4acfc54",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SM103A"): (
        "cake_fmha_context_nvfp4_e45bc90b69f0fc31215c",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SCALE_PTR_SM100A"): (
        "cake_fmha_context_nvfp4_ab413a0149f11e7a8463",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SCALE_PTR_SM103A"): (
        "cake_fmha_context_nvfp4_81f166c4d715b824c4ed",
        16,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK1_SM100A"): (
        "cake_fmha_context_nvfp4_86d0c41d97852b032e2b",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK1_SM103A"): (
        "cake_fmha_context_nvfp4_33d3b7c3d901d4a2979d",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK2_SM100A"): (
        "cake_fmha_context_nvfp4_85759514a21c2f7af98e",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK2_SM103A"): (
        "cake_fmha_context_nvfp4_7e30df993f080875ccac",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK4_SM100A"): (
        "cake_fmha_context_nvfp4_72dab1767cf5d92e3932",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK4_SM103A"): (
        "cake_fmha_context_nvfp4_8014de3355e94a76a28e",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK5_SM100A"): (
        "cake_fmha_context_nvfp4_3dc30e785b083d6ab935",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK5_SM103A"): (
        "cake_fmha_context_nvfp4_0d3744017807e7c34100",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK6_SM100A"): (
        "cake_fmha_context_nvfp4_e214794d1c7e780bbb9b",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK6_SM103A"): (
        "cake_fmha_context_nvfp4_611ee6aa818250537852",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK7_SM100A"): (
        "cake_fmha_context_nvfp4_8238132fab08fc8a1519",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK7_SM103A"): (
        "cake_fmha_context_nvfp4_f0e620cd381756f2a2c2",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK8_SM100A"): (
        "cake_fmha_context_nvfp4_9c67cb54b6dcaeda2935",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK8_SM103A"): (
        "cake_fmha_context_nvfp4_a5c97d11fcc3c83f65b5",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK12_SM100A"): (
        "cake_fmha_context_nvfp4_069ac3e5e2f6bba89935",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK12_SM103A"): (
        "cake_fmha_context_nvfp4_fa5a7007388a7a382e58",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK16_SM100A"): (
        "cake_fmha_context_nvfp4_db6cae7b4015feede42b",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK16_SM103A"): (
        "cake_fmha_context_nvfp4_de2ab2021b2f5e79afe4",
        None,
        False,
    ),
    # sm103a, launches of at most the keys of _CONTEXT_NVFP4_SHORT_KV_ROUTES:
    # the V dequant on the correction warps.
    ("context_nvfp4", "Q511_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_ea63d1372358797fdf6f",
        16,
        False,
    ),
    ("context_nvfp4", "Q511_SCALE_PTR_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_e673bfa791f7c71aeb61",
        16,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK16_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_a012c93ca3cba6e2a56e",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK12_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_5541b6e1896cbe81f9e5",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK8_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_2a9272cea2714be55f6d",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK7_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_e86e3160f4a150409eb7",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK6_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_85634e40302d7c512ac5",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK5_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_a43177abb2d2639c34ba",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK4_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_a2ccb67e2966ae3b7edc",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK2_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_a07b0866dce3af2b4840",
        None,
        False,
    ),
    ("context_nvfp4", "RUNTIME_PACK1_SHORT_KV_SM103A"): (
        "cake_fmha_context_nvfp4_6f93414ba399d23b5916",
        None,
        False,
    ),
    # The FP8 runtime-size programs: the m-block count is a launch argument.
    ("context_fp8", "RUNTIME_PACK8"): (
        "cake_fmha_context_fp8_27799c2f9548e951013e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4"): (
        "cake_fmha_context_fp8_e193fed2eff47620b919",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2"): (
        "cake_fmha_context_fp8_f873d11e318ec3392f47",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1"): (
        "cake_fmha_context_fp8_ab6de9d118422d291f84",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PAGE64"): (
        "cake_fmha_context_fp8_541cc008db776c108277",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PAGE64"): (
        "cake_fmha_context_fp8_bf040e8cb8577f37d708",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PAGE64"): (
        "cake_fmha_context_fp8_e5d2537ba897c1e551a3",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64"): (
        "cake_fmha_context_fp8_8c22dd8f00a06a0700ed",
        None,
        False,
    ),
    # The sm103a runtime-size programs: the same schedules with a 4-stage
    # K/V ring and each stage's P tile in the other stage's S columns.
    ("context_fp8", "RUNTIME_PACK8_PX"): (
        "cake_fmha_context_fp8_974dea25a1bcbc06775c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PX"): (
        "cake_fmha_context_fp8_479df0d88cefb094c3e7",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PX"): (
        "cake_fmha_context_fp8_b8d0dbb525f2696d3e03",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PX"): (
        "cake_fmha_context_fp8_933bdbad91945bcf2d21",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PAGE64_PX"): (
        "cake_fmha_context_fp8_1222348d842644950386",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PAGE64_PX"): (
        "cake_fmha_context_fp8_29cdc45592b5623137c6",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PAGE64_PX"): (
        "cake_fmha_context_fp8_733558922a0144e96b0f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64_PX"): (
        "cake_fmha_context_fp8_290a4ce6ad5aa95368d1",
        None,
        False,
    ),
    # The one-stage FP8 runtime-size programs: the same schedules with one
    # 128-row query stage per tile (an m-block of 128 / pack_g query tokens).
    ("context_fp8", "RUNTIME_PACK8_Q1"): (
        "cake_fmha_context_fp8_98519e1561f69cc77fda",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_Q1"): (
        "cake_fmha_context_fp8_69371659c5de83d1c6f3",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_Q1"): (
        "cake_fmha_context_fp8_d00439888f9ecc52cf8e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_Q1"): (
        "cake_fmha_context_fp8_23ccecb73487c64c88e4",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PAGE64_Q1"): (
        "cake_fmha_context_fp8_66d9c3d0a157c6288c8f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PAGE64_Q1"): (
        "cake_fmha_context_fp8_09891a177f5196571b27",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PAGE64_Q1"): (
        "cake_fmha_context_fp8_b82165c563e4cb32b133",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64_Q1"): (
        "cake_fmha_context_fp8_888c8c0de1e167c1212c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PX_Q1"): (
        "cake_fmha_context_fp8_b193daaab958d4babaac",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PX_Q1"): (
        "cake_fmha_context_fp8_ec3760f97265869bb4d8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PX_Q1"): (
        "cake_fmha_context_fp8_55b399c09e7b9e963664",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PX_Q1"): (
        "cake_fmha_context_fp8_e3820dae3c8125209907",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_PAGE64_PX_Q1"): (
        "cake_fmha_context_fp8_33898a252fbaa5722dae",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_PAGE64_PX_Q1"): (
        "cake_fmha_context_fp8_b167e399fff8871743fc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_PAGE64_PX_Q1"): (
        "cake_fmha_context_fp8_6934061567e1564d7aa0",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64_PX_Q1"): (
        "cake_fmha_context_fp8_b42343b75e9525f7d215",
        None,
        False,
    ),
    # The sm100a one-stage page-64 programs of packs 2 and 1 without the
    # short-KV chain (_CONTEXT_FP8_MULTI_WAVE_ROUTES).
    ("context_fp8", "RUNTIME_PACK2_PAGE64_Q1_MW"): (
        "cake_fmha_context_fp8_dd2e6e339330c40630fa",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_PAGE64_Q1_MW"): (
        "cake_fmha_context_fp8_d3093a512d060fdfad20",
        None,
        False,
    ),
    # The sm100a pack-8 program for every page size that stores stage 0 as soon
    # as its last PV is in (_CONTEXT_FP8_ONE_WAVE_ROUTES).
    ("context_fp8", "RUNTIME_PACK8_OW"): (
        "cake_fmha_context_fp8_c42e0614ab8103e2499a",
        None,
        False,
    ),
    # The sm100a one-stage programs for every page size with the short-KV chain
    # (the entry L2 warm-up of the static first tile and the partial block's
    # single masking pass), for one-stage launches of one wave
    # (_CONTEXT_FP8_ONE_WAVE_ROUTES).
    ("context_fp8", "RUNTIME_PACK8_Q1_OW"): (
        "cake_fmha_context_fp8_024e5f82f5e94b38d2ff",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_Q1_OW"): (
        "cake_fmha_context_fp8_7421f1ee37279a2bea13",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_Q1_OW"): (
        "cake_fmha_context_fp8_e22e25d0e0baa16f1cb7",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_Q1_OW"): (
        "cake_fmha_context_fp8_3e5440ae1066e6902e1d",
        None,
        False,
    ),
    # The FP8 runtime-size programs of packs 7 and 5 (the calls of that JIT
    # pack): every page size and page 64, then the sm103a 4-stage ring builds
    # with each stage's P tile in the other stage's S columns.
    ("context_fp8", "RUNTIME_PACK7"): (
        "cake_fmha_context_fp8_deed18c419e94dcae492",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5"): (
        "cake_fmha_context_fp8_6aa58bfdfad6e2b3b070",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_PAGE64"): (
        "cake_fmha_context_fp8_1d475a87368eff01cd07",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_PAGE64"): (
        "cake_fmha_context_fp8_6697efee895077ff75db",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_PX"): (
        "cake_fmha_context_fp8_c1e7b4a95bde95adfb5b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_PX"): (
        "cake_fmha_context_fp8_82494a452a599bcce06b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_PAGE64_PX"): (
        "cake_fmha_context_fp8_12f3911feb73818fea63",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_PAGE64_PX"): (
        "cake_fmha_context_fp8_783cc42f6c36478cc6d1",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6"): (
        "cake_fmha_context_fp8_0fc78080232eae07fc8b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12"): (
        "cake_fmha_context_fp8_ce478d84f15b7c0ee7f6",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16"): (
        "cake_fmha_context_fp8_2d54b7d14e534a107e70",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_PAGE64"): (
        "cake_fmha_context_fp8_4315154de6b6492fddba",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_PAGE64"): (
        "cake_fmha_context_fp8_2598d3427a6230e965ca",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_PAGE64"): (
        "cake_fmha_context_fp8_5bca98ec850d2a56d4fa",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_PX"): (
        "cake_fmha_context_fp8_d05673294d892173f58b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_PX"): (
        "cake_fmha_context_fp8_611fce7f1f4f9ddaf157",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_PX"): (
        "cake_fmha_context_fp8_45a79a6e2dc0278bc02f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_PAGE64_PX"): (
        "cake_fmha_context_fp8_3f6317f5fefc0c435974",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_PAGE64_PX"): (
        "cake_fmha_context_fp8_fb8adba84de1e52fbabe",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_PAGE64_PX"): (
        "cake_fmha_context_fp8_0018f22cd0b10e1e5a2a",
        None,
        False,
    ),
    # The FP8 runtime-size programs of the other forms (_CONTEXT_FP8_FORMS:
    # non-causal, LSE, attention sink, non-causal LSE, non-causal sink) of
    # packs 1, 2, 4, 8, 7, 5, 6, 12 and 16, every page size, both targets.
    ("context_fp8", "RUNTIME_PACK1_NC"): (
        "cake_fmha_context_fp8_3c94aa56b26bc9692ddb",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC"): (
        "cake_fmha_context_fp8_c51d1037e40d5f686012",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC"): (
        "cake_fmha_context_fp8_a82e724d7beb4543ed1b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC"): (
        "cake_fmha_context_fp8_23cb749a9b56174b06dd",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC"): (
        "cake_fmha_context_fp8_39e77cfad484d671bdf1",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC"): (
        "cake_fmha_context_fp8_b3f2a547db0024e7513c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC"): (
        "cake_fmha_context_fp8_18c01ba2bfdf7cfc1213",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC"): (
        "cake_fmha_context_fp8_bbd1c3ad0faffa4e4032",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC"): (
        "cake_fmha_context_fp8_5e8e9f8a2ac80ec689dc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_LSE"): (
        "cake_fmha_context_fp8_68188eb65e57813cd400",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_LSE"): (
        "cake_fmha_context_fp8_2649fc5e2adde4764d44",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_LSE"): (
        "cake_fmha_context_fp8_4ad5fa7c3a77046179a0",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_LSE"): (
        "cake_fmha_context_fp8_d20c132ce7ac8015f3aa",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_LSE"): (
        "cake_fmha_context_fp8_57a2591a9a94e34f9a3f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_LSE"): (
        "cake_fmha_context_fp8_07270c5949daaef8b88e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_LSE"): (
        "cake_fmha_context_fp8_d888d7d5891a43b77019",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_LSE"): (
        "cake_fmha_context_fp8_6ccee1242099503af08e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_LSE"): (
        "cake_fmha_context_fp8_6936e327f65dc978c02b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_SINK"): (
        "cake_fmha_context_fp8_8d70f2cc8a0c2b0d85a8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_SINK"): (
        "cake_fmha_context_fp8_0afb1ba0634298237a87",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_SINK"): (
        "cake_fmha_context_fp8_888cb4ba77655cc02f4f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_SINK"): (
        "cake_fmha_context_fp8_51b98d91b3e326c15a2b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_SINK"): (
        "cake_fmha_context_fp8_b6a31d0d586eeb8c3467",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_SINK"): (
        "cake_fmha_context_fp8_00086e1974145b14d190",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_SINK"): (
        "cake_fmha_context_fp8_08d3eec257ffff36dfb9",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_SINK"): (
        "cake_fmha_context_fp8_b7359427ad11f8d60682",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_SINK"): (
        "cake_fmha_context_fp8_fde5af9c61501f5e206d",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_LSE"): (
        "cake_fmha_context_fp8_09a88e2efd0f3faa4ba1",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_LSE"): (
        "cake_fmha_context_fp8_2f3a9e2ef72f854708c5",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_LSE"): (
        "cake_fmha_context_fp8_78bde5f2e6bc5116f8b9",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_LSE"): (
        "cake_fmha_context_fp8_eb6d79120a92bd55db14",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC_LSE"): (
        "cake_fmha_context_fp8_401f00f170131bb034b2",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC_LSE"): (
        "cake_fmha_context_fp8_2ddf5d3394e9bfd463f8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC_LSE"): (
        "cake_fmha_context_fp8_30f9d749610756b11ead",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC_LSE"): (
        "cake_fmha_context_fp8_480ac91cd4870a828748",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC_LSE"): (
        "cake_fmha_context_fp8_98f676c27683dfddb23e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_SINK"): (
        "cake_fmha_context_fp8_f16c6db9ade62789b231",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_SINK"): (
        "cake_fmha_context_fp8_448286e8962b3662fcbc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_SINK"): (
        "cake_fmha_context_fp8_c97a06bf21cccad88264",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_SINK"): (
        "cake_fmha_context_fp8_b3c47455115a96c31bdf",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC_SINK"): (
        "cake_fmha_context_fp8_0ccbe98304525493ba7b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC_SINK"): (
        "cake_fmha_context_fp8_fb85e9569bb460503e8f",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC_SINK"): (
        "cake_fmha_context_fp8_9edf17ea3e8ae77bf128",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC_SINK"): (
        "cake_fmha_context_fp8_257a42a1c41329734128",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC_SINK"): (
        "cake_fmha_context_fp8_4d7e46490a68fabfe45e",
        None,
        False,
    ),
    # The sm103a builds of the same programs with the 4-stage K/V ring and
    # each stage's P tile in the other stage's S columns.
    ("context_fp8", "RUNTIME_PACK1_NC_PX"): (
        "cake_fmha_context_fp8_3f05abdd0dfb416998fd",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_PX"): (
        "cake_fmha_context_fp8_43829028305fe1296e79",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_PX"): (
        "cake_fmha_context_fp8_03f683a3fe844acad53a",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_PX"): (
        "cake_fmha_context_fp8_0c8c4520324b57450d68",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC_PX"): (
        "cake_fmha_context_fp8_949b803cd8ef3e4ee015",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC_PX"): (
        "cake_fmha_context_fp8_11a387bdb94e731b4a81",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC_PX"): (
        "cake_fmha_context_fp8_a6eaaef29b42f9602206",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC_PX"): (
        "cake_fmha_context_fp8_05047f777e8265009b0d",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC_PX"): (
        "cake_fmha_context_fp8_e386241abbbca476ce27",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_LSE_PX"): (
        "cake_fmha_context_fp8_f907e36ece76575a53d1",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_LSE_PX"): (
        "cake_fmha_context_fp8_1aba28a73173872fd3fb",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_LSE_PX"): (
        "cake_fmha_context_fp8_31badec3271fcbce787d",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_LSE_PX"): (
        "cake_fmha_context_fp8_6b73d5dc4ae4c91c8a91",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_LSE_PX"): (
        "cake_fmha_context_fp8_4fc03e35f070593d5604",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_LSE_PX"): (
        "cake_fmha_context_fp8_93267992d47af41601f2",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_LSE_PX"): (
        "cake_fmha_context_fp8_85a868e1aeda0e3ad931",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_LSE_PX"): (
        "cake_fmha_context_fp8_f60026397cd113a89c4a",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_LSE_PX"): (
        "cake_fmha_context_fp8_153db10be76831301707",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_SINK_PX"): (
        "cake_fmha_context_fp8_06551cbf9595d0fec9b2",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_SINK_PX"): (
        "cake_fmha_context_fp8_2eb08cfd64f065c2923d",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_SINK_PX"): (
        "cake_fmha_context_fp8_d9f12a7798a4ef48f3fc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_SINK_PX"): (
        "cake_fmha_context_fp8_4218bd247b760dcb8ebd",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_SINK_PX"): (
        "cake_fmha_context_fp8_5135dab25ffdbce365c9",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_SINK_PX"): (
        "cake_fmha_context_fp8_3b4d3e7797632fd340d8",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_SINK_PX"): (
        "cake_fmha_context_fp8_c828c717314b2c30ecdb",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_SINK_PX"): (
        "cake_fmha_context_fp8_2448bba5c99b727f7bba",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_SINK_PX"): (
        "cake_fmha_context_fp8_64f3ba3f829afe2a9c81",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_LSE_PX"): (
        "cake_fmha_context_fp8_87118ffc6e2fd677b465",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_LSE_PX"): (
        "cake_fmha_context_fp8_ae7ec9d64660cc3b1e3c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_LSE_PX"): (
        "cake_fmha_context_fp8_4615c4178898a152657b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_LSE_PX"): (
        "cake_fmha_context_fp8_f579d90a5bdd681b22f9",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC_LSE_PX"): (
        "cake_fmha_context_fp8_8629ad04ca06ccd1ecd6",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC_LSE_PX"): (
        "cake_fmha_context_fp8_f3a595de408fa3d2ad51",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC_LSE_PX"): (
        "cake_fmha_context_fp8_960bf6763c0cfbeae654",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC_LSE_PX"): (
        "cake_fmha_context_fp8_46865ca87d422fdec925",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC_LSE_PX"): (
        "cake_fmha_context_fp8_79a7946155bc3993b289",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_SINK_PX"): (
        "cake_fmha_context_fp8_f5fa3bfd3b00fd44fabc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_SINK_PX"): (
        "cake_fmha_context_fp8_ebf8d7837e7123a08abe",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_SINK_PX"): (
        "cake_fmha_context_fp8_f35f2ffe6db82fbabebf",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_SINK_PX"): (
        "cake_fmha_context_fp8_adb2255214f458a9ec66",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK7_NC_SINK_PX"): (
        "cake_fmha_context_fp8_909439ede445c07be428",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK5_NC_SINK_PX"): (
        "cake_fmha_context_fp8_ffd4f84506eb37625918",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK6_NC_SINK_PX"): (
        "cake_fmha_context_fp8_1c6b259c9c16a55a3ace",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK12_NC_SINK_PX"): (
        "cake_fmha_context_fp8_0a4c6a377884f71460b7",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK16_NC_SINK_PX"): (
        "cake_fmha_context_fp8_48b599fe731a4ce66d7b",
        None,
        False,
    ),
    # Their one-stage builds of packs 1, 2, 4 and 8, for both targets
    # (_CONTEXT_FP8_FORM_ONE_STAGE_ROUTES).
    ("context_fp8", "RUNTIME_PACK1_NC_PX_Q1"): (
        "cake_fmha_context_fp8_8e82b164f633739b5629",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_PX_Q1"): (
        "cake_fmha_context_fp8_e93a40c1207c92f87a57",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_PX_Q1"): (
        "cake_fmha_context_fp8_3a283b3a3d8e2458600a",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_PX_Q1"): (
        "cake_fmha_context_fp8_0d8e81c1e5cc25d494be",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_6cb0aa1bc299570e3bb6",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_a5fe9901f54f0f82983e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_a20192c91fa6463266af",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_67e4a476e798c6ef5eab",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_42a34f611e416b95f3c4",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_70a1f8d6422b88b2a5d1",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_a8f20df7b8e78c71187b",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_340f1b13ef6db66a4819",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_f39420f79b571a0d7e68",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_94e7c483b24b790cc3ee",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_aaafe5a028f797974fcd",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_LSE_PX_Q1"): (
        "cake_fmha_context_fp8_51fbdbb46101ec85a10c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK1_NC_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_9d57fae5c177b84f333c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK2_NC_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_56ed41488640572f208c",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK4_NC_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_9633967d1993341e9e9e",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_PACK8_NC_SINK_PX_Q1"): (
        "cake_fmha_context_fp8_cdfe342fcf2ea3cd54e1",
        None,
        False,
    ),
    # The FP8 split-KV programs: the runtime-size arguments, then the split plan.
    ("context_fp8", "RUNTIME_SK_PACK8"): (
        "cake_fmha_context_fp8_2b603161717df80f5fad",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK4"): (
        "cake_fmha_context_fp8_65810993ed3b69e0ec02",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK2"): (
        "cake_fmha_context_fp8_af67d1eeceafba84dcfa",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK1"): (
        "cake_fmha_context_fp8_1a415e1054807d030032",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK8_PAGE64"): (
        "cake_fmha_context_fp8_795ed70e65c7e13a5f23",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK4_PAGE64"): (
        "cake_fmha_context_fp8_77d3ca81fef2a7f7b7cc",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK2_PAGE64"): (
        "cake_fmha_context_fp8_7fa02b4209c5bbac7b59",
        None,
        False,
    ),
    ("context_fp8", "RUNTIME_SK_PACK1_PAGE64"): (
        "cake_fmha_context_fp8_f96322ad196e7e677281",
        None,
        False,
    ),
    # The sm100a pack-1 page-64 split-KV program for the plans of two units per
    # split tile (_CONTEXT_FP8_TWO_SPLIT_ROUTES).
    ("context_fp8", "RUNTIME_SK_PACK1_PAGE64_S2"): (
        "cake_fmha_context_fp8_2f52275266d147049a48",
        None,
        False,
    ),
}
# The FP8 runtime-size context programs: every length, head count, m-block
# count and page size is a launch argument, so one program serves every causal
# FP8 call without LSE or sink whose GQA group its pack (the query heads one
# 128-row tile stage holds) divides; pack 1 serves every group, and a pack of
# :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS` the calls of its own JIT pack that
# no dividing pack serves.  Per target,
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
        (7, 64): "RUNTIME_PACK7_PAGE64",
        (5, 64): "RUNTIME_PACK5_PAGE64",
        (7, None): "RUNTIME_PACK7",
        (5, None): "RUNTIME_PACK5",
        (6, 64): "RUNTIME_PACK6_PAGE64",
        (12, 64): "RUNTIME_PACK12_PAGE64",
        (16, 64): "RUNTIME_PACK16_PAGE64",
        (6, None): "RUNTIME_PACK6",
        (12, None): "RUNTIME_PACK12",
        (16, None): "RUNTIME_PACK16",
    },
    "sm103a": {
        (8, 64): "RUNTIME_PACK8_PAGE64_PX",
        (4, 64): "RUNTIME_PACK4_PAGE64_PX",
        (2, 64): "RUNTIME_PACK2_PAGE64_PX",
        (1, 64): "RUNTIME_PACK1_PAGE64_PX",
        (8, None): "RUNTIME_PACK8_PX",
        (4, None): "RUNTIME_PACK4_PX",
        (2, None): "RUNTIME_PACK2_PX",
        (1, None): "RUNTIME_PACK1_PX",
        (7, 64): "RUNTIME_PACK7_PAGE64_PX",
        (5, 64): "RUNTIME_PACK5_PAGE64_PX",
        (7, None): "RUNTIME_PACK7_PX",
        (5, None): "RUNTIME_PACK5_PX",
        (6, 64): "RUNTIME_PACK6_PAGE64_PX",
        (12, 64): "RUNTIME_PACK12_PAGE64_PX",
        (16, 64): "RUNTIME_PACK16_PAGE64_PX",
        (6, None): "RUNTIME_PACK6_PX",
        (12, None): "RUNTIME_PACK12_PX",
        (16, None): "RUNTIME_PACK16_PX",
    },
}
# The runtime-size BF16 programs (``RUNTIME_PACK<g>``): the persistent
# programs of the head-group-packed kernel with every size a launch argument.
# Their tile work is that of the JIT module of the same pack, so their output
# is the JIT module's bit for bit; a program serves only the calls whose JIT
# pack (:func:`_context_pack_g`) is its own.  Per target, ``(pack_g,
# page_size)`` names the program (``None``: every page size 16 .. 1024).  The
# host consults them for a causal call without LSE or sink that no exact route
# serves, except the calls of :data:`_CONTEXT_BF16_RUNTIME_POW2_Q`; the tables
# below serve the other flag forms (:data:`_CONTEXT_RUNTIME_FLAG_ROUTES`).
_CONTEXT_BF16_RUNTIME_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16",
        (12, None): "RUNTIME_PACK12",
        (8, None): "RUNTIME_PACK8",
        (7, None): "RUNTIME_PACK7",
        (6, None): "RUNTIME_PACK6",
        (5, None): "RUNTIME_PACK5",
        (4, None): "RUNTIME_PACK4",
        (2, None): "RUNTIME_PACK2",
        (1, None): "RUNTIME_PACK1",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16",
        (12, None): "RUNTIME_PACK12",
        (8, None): "RUNTIME_PACK8",
        (7, None): "RUNTIME_PACK7",
        (6, None): "RUNTIME_PACK6",
        (5, None): "RUNTIME_PACK5",
        (4, None): "RUNTIME_PACK4",
        (2, None): "RUNTIME_PACK2",
        (1, None): "RUNTIME_PACK1",
    },
}
# The flag forms of the runtime-size BF16 programs: the same packs and page
# sizes, built with the call's attention sinks, LSE output or non-causal mask;
# each computes the tiles of the JIT module of its form, bit for bit.
# causal, attention sinks
_CONTEXT_BF16_RUNTIME_SINK_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_SINK",
        (12, None): "RUNTIME_PACK12_SINK",
        (8, None): "RUNTIME_PACK8_SINK",
        (7, None): "RUNTIME_PACK7_SINK",
        (6, None): "RUNTIME_PACK6_SINK",
        (5, None): "RUNTIME_PACK5_SINK",
        (4, None): "RUNTIME_PACK4_SINK",
        (2, None): "RUNTIME_PACK2_SINK",
        (1, None): "RUNTIME_PACK1_SINK",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_SINK",
        (12, None): "RUNTIME_PACK12_SINK",
        (8, None): "RUNTIME_PACK8_SINK",
        (7, None): "RUNTIME_PACK7_SINK",
        (6, None): "RUNTIME_PACK6_SINK",
        (5, None): "RUNTIME_PACK5_SINK",
        (4, None): "RUNTIME_PACK4_SINK",
        (2, None): "RUNTIME_PACK2_SINK",
        (1, None): "RUNTIME_PACK1_SINK",
    },
}
# causal, LSE output
_CONTEXT_BF16_RUNTIME_LSE_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_LSE",
        (12, None): "RUNTIME_PACK12_LSE",
        (8, None): "RUNTIME_PACK8_LSE",
        (7, None): "RUNTIME_PACK7_LSE",
        (6, None): "RUNTIME_PACK6_LSE",
        (5, None): "RUNTIME_PACK5_LSE",
        (4, None): "RUNTIME_PACK4_LSE",
        (2, None): "RUNTIME_PACK2_LSE",
        (1, None): "RUNTIME_PACK1_LSE",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_LSE",
        (12, None): "RUNTIME_PACK12_LSE",
        (8, None): "RUNTIME_PACK8_LSE",
        (7, None): "RUNTIME_PACK7_LSE",
        (6, None): "RUNTIME_PACK6_LSE",
        (5, None): "RUNTIME_PACK5_LSE",
        (4, None): "RUNTIME_PACK4_LSE",
        (2, None): "RUNTIME_PACK2_LSE",
        (1, None): "RUNTIME_PACK1_LSE",
    },
}
# The causal sink and LSE forms again, built without tail-order code: their
# tail units always run one after another and ``full_sections`` is the plain
# section count.  A launch of those forms whose tail order is the plain one
# (:func:`_context_bf16_tail_interleave` false) runs the program of the same
# pack and page size here; an interleaved launch runs the program of its form's
# table.  The tiles and so the output bits are the same.  sm100a (B200) ships
# them for the packs whose plain-order launches of at most
# :data:`_CONTEXT_BF16_SEQ_TAIL_MAX_KV_LEN` keys ran faster than the form's
# program: sinks at packs 1, 5, 6, 8 and 12, LSE at packs 1, 8 and 12 (at 2048
# and more keys they ran no faster); its other packs run the form's program.
# sm103a (GB300) ships them for every pack but 16, whose launches of at most
# its bound were not measured; pack 16 runs the form's program.
# causal, attention sinks, no tail-order code
_CONTEXT_BF16_RUNTIME_SINK_SEQ_TAIL_ROUTES: dict[
    str, dict[tuple[int, int | None], str]
] = {
    "sm100a": {
        (12, None): "RUNTIME_PACK12_SINK_SEQ_TAIL",
        (8, None): "RUNTIME_PACK8_SINK_SEQ_TAIL",
        (6, None): "RUNTIME_PACK6_SINK_SEQ_TAIL",
        (5, None): "RUNTIME_PACK5_SINK_SEQ_TAIL",
        (1, None): "RUNTIME_PACK1_SINK_SEQ_TAIL",
    },
    "sm103a": {
        (12, None): "RUNTIME_PACK12_SINK_SEQ_TAIL",
        (8, None): "RUNTIME_PACK8_SINK_SEQ_TAIL",
        (7, None): "RUNTIME_PACK7_SINK_SEQ_TAIL",
        (6, None): "RUNTIME_PACK6_SINK_SEQ_TAIL",
        (5, None): "RUNTIME_PACK5_SINK_SEQ_TAIL",
        (4, None): "RUNTIME_PACK4_SINK_SEQ_TAIL",
        (2, None): "RUNTIME_PACK2_SINK_SEQ_TAIL",
        (1, None): "RUNTIME_PACK1_SINK_SEQ_TAIL",
    },
}
# causal, LSE output, no tail-order code
_CONTEXT_BF16_RUNTIME_LSE_SEQ_TAIL_ROUTES: dict[
    str, dict[tuple[int, int | None], str]
] = {
    "sm100a": {
        (12, None): "RUNTIME_PACK12_LSE_SEQ_TAIL",
        (8, None): "RUNTIME_PACK8_LSE_SEQ_TAIL",
        (1, None): "RUNTIME_PACK1_LSE_SEQ_TAIL",
    },
    "sm103a": {
        (12, None): "RUNTIME_PACK12_LSE_SEQ_TAIL",
        (8, None): "RUNTIME_PACK8_LSE_SEQ_TAIL",
        (7, None): "RUNTIME_PACK7_LSE_SEQ_TAIL",
        (6, None): "RUNTIME_PACK6_LSE_SEQ_TAIL",
        (5, None): "RUNTIME_PACK5_LSE_SEQ_TAIL",
        (4, None): "RUNTIME_PACK4_LSE_SEQ_TAIL",
        (2, None): "RUNTIME_PACK2_LSE_SEQ_TAIL",
        (1, None): "RUNTIME_PACK1_LSE_SEQ_TAIL",
    },
}
# not causal
_CONTEXT_BF16_RUNTIME_NONCAUSAL_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL",
        (12, None): "RUNTIME_PACK12_NONCAUSAL",
        (8, None): "RUNTIME_PACK8_NONCAUSAL",
        (7, None): "RUNTIME_PACK7_NONCAUSAL",
        (6, None): "RUNTIME_PACK6_NONCAUSAL",
        (5, None): "RUNTIME_PACK5_NONCAUSAL",
        (4, None): "RUNTIME_PACK4_NONCAUSAL",
        (2, None): "RUNTIME_PACK2_NONCAUSAL",
        (1, None): "RUNTIME_PACK1_NONCAUSAL",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL",
        (12, None): "RUNTIME_PACK12_NONCAUSAL",
        (8, None): "RUNTIME_PACK8_NONCAUSAL",
        (7, None): "RUNTIME_PACK7_NONCAUSAL",
        (6, None): "RUNTIME_PACK6_NONCAUSAL",
        (5, None): "RUNTIME_PACK5_NONCAUSAL",
        (4, None): "RUNTIME_PACK4_NONCAUSAL",
        (2, None): "RUNTIME_PACK2_NONCAUSAL",
        (1, None): "RUNTIME_PACK1_NONCAUSAL",
    },
}
# not causal, attention sinks
_CONTEXT_BF16_RUNTIME_NONCAUSAL_SINK_ROUTES: dict[
    str, dict[tuple[int, int | None], str]
] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL_SINK",
        (12, None): "RUNTIME_PACK12_NONCAUSAL_SINK",
        (8, None): "RUNTIME_PACK8_NONCAUSAL_SINK",
        (7, None): "RUNTIME_PACK7_NONCAUSAL_SINK",
        (6, None): "RUNTIME_PACK6_NONCAUSAL_SINK",
        (5, None): "RUNTIME_PACK5_NONCAUSAL_SINK",
        (4, None): "RUNTIME_PACK4_NONCAUSAL_SINK",
        (2, None): "RUNTIME_PACK2_NONCAUSAL_SINK",
        (1, None): "RUNTIME_PACK1_NONCAUSAL_SINK",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL_SINK",
        (12, None): "RUNTIME_PACK12_NONCAUSAL_SINK",
        (8, None): "RUNTIME_PACK8_NONCAUSAL_SINK",
        (7, None): "RUNTIME_PACK7_NONCAUSAL_SINK",
        (6, None): "RUNTIME_PACK6_NONCAUSAL_SINK",
        (5, None): "RUNTIME_PACK5_NONCAUSAL_SINK",
        (4, None): "RUNTIME_PACK4_NONCAUSAL_SINK",
        (2, None): "RUNTIME_PACK2_NONCAUSAL_SINK",
        (1, None): "RUNTIME_PACK1_NONCAUSAL_SINK",
    },
}
# not causal, LSE output
_CONTEXT_BF16_RUNTIME_NONCAUSAL_LSE_ROUTES: dict[
    str, dict[tuple[int, int | None], str]
] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL_LSE",
        (12, None): "RUNTIME_PACK12_NONCAUSAL_LSE",
        (8, None): "RUNTIME_PACK8_NONCAUSAL_LSE",
        (7, None): "RUNTIME_PACK7_NONCAUSAL_LSE",
        (6, None): "RUNTIME_PACK6_NONCAUSAL_LSE",
        (5, None): "RUNTIME_PACK5_NONCAUSAL_LSE",
        (4, None): "RUNTIME_PACK4_NONCAUSAL_LSE",
        (2, None): "RUNTIME_PACK2_NONCAUSAL_LSE",
        (1, None): "RUNTIME_PACK1_NONCAUSAL_LSE",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_NONCAUSAL_LSE",
        (12, None): "RUNTIME_PACK12_NONCAUSAL_LSE",
        (8, None): "RUNTIME_PACK8_NONCAUSAL_LSE",
        (7, None): "RUNTIME_PACK7_NONCAUSAL_LSE",
        (6, None): "RUNTIME_PACK6_NONCAUSAL_LSE",
        (5, None): "RUNTIME_PACK5_NONCAUSAL_LSE",
        (4, None): "RUNTIME_PACK4_NONCAUSAL_LSE",
        (2, None): "RUNTIME_PACK2_NONCAUSAL_LSE",
        (1, None): "RUNTIME_PACK1_NONCAUSAL_LSE",
    },
}
# Per component, the runtime-size route table of each flag form
# ``(is_causal, return_lse, enable_sink)`` its programs serve: a call that no
# exact route serves takes the program of its pack and page size from the
# table of its own form, and a form the component lists no table for keeps
# the JIT module.  (Sinks with LSE is no route.)
_CONTEXT_RUNTIME_FLAG_ROUTES: dict[
    str,
    dict[tuple[bool, bool, bool], dict[str, dict[tuple[int, int | None], str]]],
] = {
    "context_bf16": {
        (True, False, False): _CONTEXT_BF16_RUNTIME_ROUTES,
        (True, False, True): _CONTEXT_BF16_RUNTIME_SINK_ROUTES,
        (True, True, False): _CONTEXT_BF16_RUNTIME_LSE_ROUTES,
        (False, False, False): _CONTEXT_BF16_RUNTIME_NONCAUSAL_ROUTES,
        (False, False, True): _CONTEXT_BF16_RUNTIME_NONCAUSAL_SINK_ROUTES,
        (False, True, False): _CONTEXT_BF16_RUNTIME_NONCAUSAL_LSE_ROUTES,
    },
}
# The BF16 flag forms ``(is_causal, return_lse, enable_sink)`` whose runtime-size
# programs also ship without tail-order code, and those programs' tables.
_CONTEXT_BF16_RUNTIME_SEQ_TAIL_ROUTES: dict[
    tuple[bool, bool, bool], dict[str, dict[tuple[int, int | None], str]]
] = {
    (True, False, True): _CONTEXT_BF16_RUNTIME_SINK_SEQ_TAIL_ROUTES,
    (True, True, False): _CONTEXT_BF16_RUNTIME_LSE_SEQ_TAIL_ROUTES,
}
# Per target, a runtime-size BF16 program of those forms -> the program of the
# same pack and page size without tail-order code, which its plain-order
# launches run.
_CONTEXT_BF16_SEQ_TAIL_PROGRAMS: dict[str, dict[str, str]] = {
    target: {
        identifier: seq_tail[target][key]
        for form, seq_tail in _CONTEXT_BF16_RUNTIME_SEQ_TAIL_ROUTES.items()
        for key, identifier in _CONTEXT_RUNTIME_FLAG_ROUTES["context_bf16"][form][
            target
        ].items()
        if key in seq_tail[target]
    }
    for target in _CONTEXT_BF16_RUNTIME_SINK_SEQ_TAIL_ROUTES
}
# Per target, the most cache keys (``max_kv_len``) of a plain-order launch that
# runs the program without tail-order code (``None``: no bound); a launch of
# more keys runs its form's program.  sm103a (GB300): at 6144 and more keys
# the pack-1 sink program ran slower than the form's program, at 4096 keys no
# program did.
_CONTEXT_BF16_SEQ_TAIL_MAX_KV_LEN: dict[str, int | None] = {
    "sm100a": 1024,
    "sm103a": 4096,
}
# Per target, ``(queries, keys)``: a causal BF16 call without LSE or sink at a
# JIT pack above 1 whose ``max_q_len`` is a power of two of at least
# ``queries``, over at least ``keys`` cache keys, keeps the JIT module.  The
# JIT module, compiled for the call's sizes, runs a power-of-two query length
# faster than its neighbours; there the runtime-size program ran slower than
# it beyond the A/A band (B200, 10 layouts, batch x queries x keys at 32 query
# and 4 KV heads: 1 x 8192 x 14336 .. 24576, 2 and 4 x 8192 x 16384, 1 x 16384
# x 16384; also packs 2, 4 and 16 at 8192 queries and pack 16 at 4096 and
# 16384), while 8064, 8320, 12288 and 24576 queries ran faster on the program.
_CONTEXT_BF16_RUNTIME_POW2_Q: dict[str, tuple[int, int]] = {"sm100a": (4096, 14336)}

# The runtime-size NVFP4 programs (``RUNTIME_PACK<g>_SM100A`` /
# ``RUNTIME_PACK<g>_SM103A``): the persistent programs of the fused NVFP4
# kernel with every size and the scale source a launch argument, one per
# target with that target's schedule options.  Their tile work is that of the
# JIT module of the same pack, so their output is the JIT module's bit for
# bit; a program serves only the calls whose JIT pack
# (:func:`_context_pack_g`) is its own.  Per target, ``(pack_g, page_size)``
# names the program (``None``: the page size of every NVFP4 route, 16).  The
# host consults them only for a call that no exact route serves.
_CONTEXT_NVFP4_RUNTIME_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    "sm100a": {
        (16, None): "RUNTIME_PACK16_SM100A",
        (12, None): "RUNTIME_PACK12_SM100A",
        (8, None): "RUNTIME_PACK8_SM100A",
        (7, None): "RUNTIME_PACK7_SM100A",
        (6, None): "RUNTIME_PACK6_SM100A",
        (5, None): "RUNTIME_PACK5_SM100A",
        (4, None): "RUNTIME_PACK4_SM100A",
        (2, None): "RUNTIME_PACK2_SM100A",
        (1, None): "RUNTIME_PACK1_SM100A",
    },
    "sm103a": {
        (16, None): "RUNTIME_PACK16_SM103A",
        (12, None): "RUNTIME_PACK12_SM103A",
        (8, None): "RUNTIME_PACK8_SM103A",
        (7, None): "RUNTIME_PACK7_SM103A",
        (6, None): "RUNTIME_PACK6_SM103A",
        (5, None): "RUNTIME_PACK5_SM103A",
        (4, None): "RUNTIME_PACK4_SM103A",
        (2, None): "RUNTIME_PACK2_SM103A",
        (1, None): "RUNTIME_PACK1_SM103A",
    },
}
# The sm103a short-KV NVFP4 programs: per target, routed program -> (the
# program a call of at most ``max_kv_len`` keys takes instead, those keys).
# The routed sm103a programs dequantize V on the page-offset and scheduler
# warps; over few KV blocks per tile that hand-off ran slower than the
# correction warps' own V dequant, which the short-KV programs keep (same
# tile work, pack and arguments, so the same output bits).
_CONTEXT_NVFP4_SHORT_KV_ROUTES: dict[str, dict[str, tuple[str, int]]] = {
    "sm103a": {
        "Q511_SM103A": ("Q511_SHORT_KV_SM103A", 1023),
        "Q511_SCALE_PTR_SM103A": ("Q511_SCALE_PTR_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK16_SM103A": ("RUNTIME_PACK16_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK12_SM103A": ("RUNTIME_PACK12_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK8_SM103A": ("RUNTIME_PACK8_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK7_SM103A": ("RUNTIME_PACK7_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK6_SM103A": ("RUNTIME_PACK6_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK5_SM103A": ("RUNTIME_PACK5_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK4_SM103A": ("RUNTIME_PACK4_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK2_SM103A": ("RUNTIME_PACK2_SHORT_KV_SM103A", 1023),
        "RUNTIME_PACK1_SM103A": ("RUNTIME_PACK1_SHORT_KV_SM103A", 1023),
    },
}
# Per target and mask (causal or not), per JIT pack: the calls the runtime-size
# BF16 programs of the other flag forms (sinks, LSE, not causal) serve,
# ``(keys, query x keys, work)``: a call over more than ``keys`` cache keys
# (``max_kv_len``), of a larger ``max_q_len * max_kv_len``, or of a larger
# ``batch_size * num_q_heads * max_q_len * max_kv_len`` keeps the JIT module of
# its form (``None``: no bound).  Past them the programs ran in or below the A/A
# band of the JIT module, or slower (B200: causal packs 8 and 16 at 4 .. 16 x
# 512 .. 2048 x 8192 keys, packs 2, 4 and 8 at 16384 keys, pack 1 measured up
# to 16384 keys; not causal pack 1 at 8 x 2048 x 8192 x 32 heads, 16 x 512 x
# 16384 keys, 8192 x 8192 and 16384 x 4096).  Causal packs 4 and 16 stop at
# 5120 keys.  sm103a (GB300, 10 layouts each): causal packs 2 .. 16 ran faster
# than the JIT module at 8192 keys and pack 8 not at 12288; causal pack 1 at
# 14336 keys but not at 16384; not causal pack 1 up to 16384 keys, 8192 x 8192
# and 16 x 32 heads x 4096 x 4096; not causal pack 8 up to 16384 x 12288 and
# 2 x 32 heads x 8192 x 16384 but not at 16384 x 16384; pack 16 up to
# 16384 x 12288 and 64 heads x 4096 x 16384 but not at 64 heads x 8192 x 16384.
_CONTEXT_BF16_RUNTIME_FLAG_BOUNDS: dict[
    str, dict[bool, dict[int, tuple[int | None, int | None, int | None]]]
] = {
    "sm100a": {
        True: {
            1: (16384, None, None),
            2: (6144, None, None),
            4: (5120, None, None),
            5: (6144, None, None),
            6: (6144, None, None),
            7: (6144, None, None),
            8: (6144, None, None),
            12: (6144, None, None),
            16: (5120, None, None),
        },
        False: {1: (8192, 4096 * 4096, 1 << 31)},
    },
    "sm103a": {
        True: {
            1: (14336, None, None),
            2: (8192, None, None),
            4: (8192, None, None),
            5: (8192, None, None),
            6: (8192, None, None),
            7: (8192, None, None),
            8: (8192, None, None),
            12: (8192, None, None),
            16: (8192, None, None),
        },
        False: {
            1: (16384, 8192 * 8192, 1 << 33),
            8: (None, 16384 * 12288, 1 << 33),
            16: (None, 16384 * 12288, 1 << 32),
        },
    },
}
# Per target and mask, per JIT pack: further calls of the other flag forms the
# runtime-size BF16 programs serve past :data:`_CONTEXT_BF16_RUNTIME_FLAG_BOUNDS`,
# as boxes ``(batch, min queries, queries, keys, query x keys, work)``: a call
# of at most ``batch`` requests, a ``max_q_len`` from ``min queries`` to
# ``queries``, at most ``keys`` cache keys, at most that ``max_q_len *
# max_kv_len`` and at most ``work`` of ``batch_size * num_q_heads * max_q_len *
# max_kv_len`` takes its form's program (``None``: no bound).  Within each box
# the programs ran faster than the JIT module beyond the A/A band in every
# layout timed (10 layouts; 2 for the B200 pack-8 rows of 1, 64, 1000 and over
# 8192 queries).  B200, causal: one request up to 8192 keys at pack 8 (1 ..
# 32768 queries x 6145 .. 8192 keys at 8 .. 64 query heads) and pack 16 (1 x
# 128 .. 32768 x 8192 keys at 16 .. 64 heads); packs 4 and 16 up to 6144 keys
# and 3 x 2^30 work (1 .. 16 requests x 512 .. 8192 queries; not at 3 x 2^31,
# 16 x 2048 x 6144 keys at 32 heads and 64 x 256 at 64); packs 2 .. 12 up to
# 512 queries, 8192 keys and 2^31 work (1 .. 256 requests x 32 .. 512 queries
# x 8192 keys).
# GB300, causal pack 1: 1 .. 128 requests x 32 .. 16384 queries x 16384 keys
# at 8 .. 32 heads; not causal pack 16: 1 .. 16 requests x 512 .. 16384
# queries x 4096 .. 16384 keys at 64 and 128 query heads.  Sinks and LSE alike.
_CONTEXT_BF16_FLAG_BOX = tuple[
    int | None, int | None, int | None, int | None, int | None, int | None
]
_CONTEXT_BF16_FLAG_BOX_ONE_REQUEST: _CONTEXT_BF16_FLAG_BOX = (
    1, None, None, 8192, None, 1 << 32
)  # fmt: skip
_CONTEXT_BF16_FLAG_BOX_6144: _CONTEXT_BF16_FLAG_BOX = (
    None, None, None, 6144, None, 3 << 30
)  # fmt: skip
_CONTEXT_BF16_FLAG_BOX_SHORT_Q: _CONTEXT_BF16_FLAG_BOX = (
    None, None, 512, 8192, None, 1 << 31
)  # fmt: skip
_CONTEXT_BF16_RUNTIME_FLAG_BOXES: dict[
    str, dict[bool, dict[int, tuple[_CONTEXT_BF16_FLAG_BOX, ...]]]
] = {
    "sm100a": {
        True: {
            2: (_CONTEXT_BF16_FLAG_BOX_SHORT_Q,),
            4: (_CONTEXT_BF16_FLAG_BOX_6144, _CONTEXT_BF16_FLAG_BOX_SHORT_Q),
            5: (_CONTEXT_BF16_FLAG_BOX_SHORT_Q,),
            6: (_CONTEXT_BF16_FLAG_BOX_SHORT_Q,),
            7: (_CONTEXT_BF16_FLAG_BOX_SHORT_Q,),
            8: (_CONTEXT_BF16_FLAG_BOX_ONE_REQUEST, _CONTEXT_BF16_FLAG_BOX_SHORT_Q),
            12: (_CONTEXT_BF16_FLAG_BOX_SHORT_Q,),
            16: (_CONTEXT_BF16_FLAG_BOX_ONE_REQUEST, _CONTEXT_BF16_FLAG_BOX_6144),
        },
    },
    "sm103a": {
        True: {1: ((None, None, None, 16384, None, 1 << 32),)},
        False: {16: ((None, None, None, None, 1 << 26, 1 << 33),)},
    },
}
# Short sm103a launches keep each stage's P tile in its own S columns: the
# program an exact, runtime-size or (every page size) one-stage route names
# maps to the same schedule and arguments with the default P placement, for
# launches of at most the keys the entry names.  With one 128-key block per
# tile the swapped schedule has no next QK product to issue while P is
# written; the exact programs keep the default placement up to two blocks per
# tile, where their launches of many tiles run slower swapped.
_CONTEXT_FP8_SHORT_KV_ROUTES: dict[str, dict[str, tuple[str, int]]] = {
    "sm103a": {
        "B256_PX": ("B256_KV4", 256),
        "B256_SCALE_PTR_PX": ("B256_SCALE_PTR_KV4", 256),
        "RUNTIME_PACK8_PX": ("RUNTIME_PACK8", 128),
        "RUNTIME_PACK4_PX": ("RUNTIME_PACK4", 128),
        "RUNTIME_PACK2_PX": ("RUNTIME_PACK2", 128),
        "RUNTIME_PACK1_PX": ("RUNTIME_PACK1", 128),
        "RUNTIME_PACK8_PAGE64_PX": ("RUNTIME_PACK8_PAGE64", 128),
        "RUNTIME_PACK4_PAGE64_PX": ("RUNTIME_PACK4_PAGE64", 128),
        "RUNTIME_PACK2_PAGE64_PX": ("RUNTIME_PACK2_PAGE64", 128),
        "RUNTIME_PACK1_PAGE64_PX": ("RUNTIME_PACK1_PAGE64", 128),
        "RUNTIME_PACK8_PX_Q1": ("RUNTIME_PACK8_Q1", 128),
        "RUNTIME_PACK4_PX_Q1": ("RUNTIME_PACK4_Q1", 128),
        "RUNTIME_PACK2_PX_Q1": ("RUNTIME_PACK2_Q1", 128),
        "RUNTIME_PACK1_PX_Q1": ("RUNTIME_PACK1_Q1", 128),
        "RUNTIME_PACK7_PX": ("RUNTIME_PACK7", 128),
        "RUNTIME_PACK5_PX": ("RUNTIME_PACK5", 128),
        "RUNTIME_PACK7_PAGE64_PX": ("RUNTIME_PACK7_PAGE64", 128),
        "RUNTIME_PACK5_PAGE64_PX": ("RUNTIME_PACK5_PAGE64", 128),
        "RUNTIME_PACK6_PX": ("RUNTIME_PACK6", 128),
        "RUNTIME_PACK12_PX": ("RUNTIME_PACK12", 128),
        "RUNTIME_PACK16_PX": ("RUNTIME_PACK16", 128),
        "RUNTIME_PACK6_PAGE64_PX": ("RUNTIME_PACK6_PAGE64", 128),
        "RUNTIME_PACK12_PAGE64_PX": ("RUNTIME_PACK12_PAGE64", 128),
        "RUNTIME_PACK16_PAGE64_PX": ("RUNTIME_PACK16_PAGE64", 128),
    },
}
# The packs the runtime programs ship besides 1: a pack serves every GQA group
# it divides.
_CONTEXT_FP8_RUNTIME_PACKS = (8, 4, 2)
# The packs whose runtime-size programs serve the calls whose JIT pack
# (:func:`_context_pack_g`) is that pack itself where the dividing pack of
# :data:`_CONTEXT_FP8_RUNTIME_PACKS` would run more tile work than the JIT
# module (:func:`_context_fp8_runtime_route_pack`): two query stages of
# ``128 // pack`` tokens per tile (125, 126, 126, 120 or 128 of the 128 rows),
# the JIT module's tiles, without split-KV or one-stage programs.
_CONTEXT_FP8_RUNTIME_OWN_PACKS = (7, 5, 6, 12, 16)
# The forms of an FP8 call besides the causal one without LSE or sink, as
# ``(is_causal, return_lse, enable_sink)`` -> form: the runtime-size programs
# of each form serve the calls of that form (``_CONTEXT_FP8_RUNTIME_FORM_ROUTES``).
_CONTEXT_FP8_FORMS: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "NC",
    (True, True, False): "LSE",
    (True, False, True): "SINK",
    (False, True, False): "NC_LSE",
    (False, False, True): "NC_SINK",
}
# Per target, ``(form, pack_g, page_size)`` -> the runtime-size program of an
# FP8 call of that form (``None``: every page size): the program of its pack
# (:func:`_context_fp8_runtime_form_pack`) with the 4-stage K/V ring and each
# stage's P tile in the other stage's S columns, on both targets (the same
# arithmetic; its launches of at most 128 keys take the program with the
# 3-stage ring and P in its own S columns, ``_CONTEXT_FP8_SHORT_KV_ROUTES``).
# Timed B200 (ten seeded layouts): packs 1, 2, 4, 5 and 8, 1 .. 256 requests
# of 32 .. 2048 query tokens over 700 .. 8192 keys, ran 1 .. 6 % faster than
# the 3-stage build with P in its own S columns.  It writes the LSE (log2
# units, one float per query row and head) or reads the sinks of the call.
_CONTEXT_FP8_RUNTIME_FORM_ROUTES: dict[str, dict[tuple[str, int, int | None], str]] = {
    target: {
        (form, pack, None): f"RUNTIME_PACK{pack}_{form}_PX"
        for form in _CONTEXT_FP8_FORMS.values()
        for pack in (
            1,
            *_CONTEXT_FP8_RUNTIME_PACKS[::-1],
            *_CONTEXT_FP8_RUNTIME_OWN_PACKS,
        )
    }
    for target in ("sm100a", "sm103a")
}
_CONTEXT_FP8_SHORT_KV_ROUTES = {
    target: {
        **_CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {}),
        **{program: (program.removesuffix("_PX"), 128) for program in routes.values()},
    }
    for target, routes in _CONTEXT_FP8_RUNTIME_FORM_ROUTES.items()
}
# Per target, ``(form, pack_g, page_size)`` -> the one-stage program of an FP8
# call of that form (one 128-row query stage per tile, an m-block of ``128 /
# pack_g`` query tokens): the call takes it in place of its form program
# (:func:`_context_fp8_form_one_stage_pack`) where every tile holds at most one
# stage of query tokens (``max_q_len <= 128 / pack_g``: the two-stage tile's
# second stage holds no row, so the tile's rows, key blocks and arithmetic are
# the same; on sm100a also over at most 128 keys) or where its one-stage tiles
# fit one wave (at most the SMs: the same rows and key blocks per row in half
# as many rows per tile, all tiles at once), over more keys than the form
# program's short-KV entry (128).  Both targets run the build with the 4-stage
# K/V ring and each stage's P tile in the other stage's S columns (timed B200:
# 4 .. 32 requests of 1 .. 64 query tokens over 64 .. 2048 keys, heads 32/4,
# 32/8, 16/8 and 32/32, ran 5 .. 35 % faster than the two-stage form program
# with P in its own S columns; timed B200 and GB300, ten seeded layouts: 1 ..
# 8 requests of 64 .. 2048 query tokens in 2 .. 16 m-blocks over 1024 .. 4096
# keys, at most the SMs one-stage tiles of packs 1, 2, 4 and 8, ran 20 .. 33 %
# faster than the two-stage form program, and more tiles than the SMs 9 ..
# 39 % slower).
_CONTEXT_FP8_FORM_ONE_STAGE_ROUTES: dict[
    str, dict[tuple[str, int, int | None], str]
] = {
    target: {
        (form, pack, None): f"RUNTIME_PACK{pack}_{form}_PX_Q1"
        for form in _CONTEXT_FP8_FORMS.values()
        for pack in (1, *_CONTEXT_FP8_RUNTIME_PACKS[::-1])
    }
    for target in ("sm100a", "sm103a")
}
# Targets whose form calls of at most one stage of query tokens per tile take
# the one-stage program also over at most the keys of the form program's
# short-KV entry (timed B200: 64 and 128 keys).
_CONTEXT_FP8_FORM_ONE_STAGE_SHORT_KV = frozenset({"sm100a"})
# Per target, the most tiles of a short-KV call (one 128-key block per tile,
# served by a short-KV one-stage program) at which it takes the smallest pack
# whose tiles number at most that many, in place of the one-stage choice: such
# a tile's time is mostly fixed cost that falls with its query rows, and more
# tiles than this ran slower (timed GB300: 1 .. 16 requests of 16 .. 128 query
# tokens over 64 .. 128 keys, heads 32/4, 32/8 and 64/8).
_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES: dict[str, int] = {"sm103a": 64}
# Per target, the one-stage programs a short-KV call of that pack (the smallest
# of at most :data:`_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES` tiles) runs as they
# are: the page-64 programs whose short-KV chain (the entry L2 warm-up of the
# static first tile and the partial block's single masking pass) serves a tile
# of one KV block (timed GB300: 1 and 2 requests of 32 query tokens over 128
# keys, heads 32/8 and 32/16, ran faster than the two-stage program).
_CONTEXT_FP8_SHORT_KV_ONE_STAGE_PROGRAMS: dict[str, frozenset[str]] = {
    "sm103a": frozenset(
        f"RUNTIME_PACK{pack}_PAGE64_PX_Q1" for pack in (*_CONTEXT_FP8_RUNTIME_PACKS, 1)
    ),
}
# Per target, the query tokens at most which a short-KV call of 64-token
# pages tries pack 2 before pack 1 when its pack-2 tiles number at least
# _CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TILES: its page-64 tiles of two query
# heads ran faster than those of one (timed GB300: 1 .. 8 requests of 1 .. 32
# query tokens over 128 keys, heads 8/1 .. 64/8, with 32 or more pack-2
# tiles); with fewer tiles pack 1 spreads the call over more SMs and some of
# those calls ran slower with pack 2.
_CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TOKENS: dict[str, int] = {"sm103a": 32}
_CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TILES: dict[str, int] = {"sm103a": 32}
# The FP8 split-KV context programs: the runtime-size schedule and arguments,
# with the key range of the first ``split_tiles`` tiles split over work units
# of ``split_blocks`` KV blocks (at most ``num_splits`` per tile), whose FP32
# partials the tile's last unit merges.  Per target, ``(pack_g, page_size)``
# names the program as for the runtime-size programs (the sm103a split-KV
# programs are those of the sm100a runtime-size builds).  The host routes an FP8
# call to one only where its split plan
# (:func:`_context_fp8_split_plan`) predicts a gain over the whole-tile
# program; other calls keep their route.
_CONTEXT_FP8_SPLIT_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    target: {
        key: "RUNTIME_SK" + name.removeprefix("RUNTIME").removesuffix("_PX")
        for key, name in routes.items()
        if key[0] not in _CONTEXT_FP8_RUNTIME_OWN_PACKS
    }
    for target, routes in _CONTEXT_FP8_RUNTIME_ROUTES.items()
}
# Per target, the split-KV program a launch whose plan has two units per split
# tile at most takes instead: the same schedule, arguments and launch whose
# merging unit takes its own term from its TMEM O and only the other unit's
# partial from global memory, with the loops over the two units unrolled (the
# same arithmetic and order of every sum).  Timed on B200, heads 32/32, page
# 64: 16 requests of 192 .. 256 query tokens over 2048 .. 4096 keys ran
# 0.9 .. 1.4 % faster, 2 requests of 32 over 2048 keys 4.6 % faster.
_CONTEXT_FP8_TWO_SPLIT_ROUTES: dict[str, dict[str, str]] = {
    "sm100a": {"RUNTIME_SK_PACK1_PAGE64": "RUNTIME_SK_PACK1_PAGE64_S2"},
}
_CONTEXT_FP8_SPLIT_PROGRAMS = frozenset(
    name for routes in _CONTEXT_FP8_SPLIT_ROUTES.values() for name in routes.values()
) | frozenset(
    name
    for routes in _CONTEXT_FP8_TWO_SPLIT_ROUTES.values()
    for name in routes.values()
)
# The FP8 one-stage context programs: the runtime-size schedule, arguments and
# P placement of each target with one 128-row query stage per tile (an m-block
# of ``128 / pack_g`` query tokens) instead of two: the same arithmetic per
# query row, without the second stage's softmax, products and epilogue.  Per
# target, ``(pack_g, page_size)`` names the program as for the runtime-size
# programs.  The host routes an FP8 runtime-size call to one only where every
# tile of it holds at most one stage of query tokens and its plan predicts a
# gain (:func:`_context_fp8_one_stage_choice`); other calls keep their route.
_CONTEXT_FP8_ONE_STAGE_ROUTES: dict[str, dict[tuple[int, int | None], str]] = {
    target: {
        key: name + "_Q1"
        for key, name in routes.items()
        if key[0] not in _CONTEXT_FP8_RUNTIME_OWN_PACKS
    }
    for target, routes in _CONTEXT_FP8_RUNTIME_ROUTES.items()
}
# Per target, the page-64 one-stage programs whose short-KV chain (the entry
# L2 warm-up of the static first tile and the partial block's single masking
# pass) a launch of more tiles over few KV blocks leaves: the program a launch
# of more than the tiles the entry names over at most its keys takes instead,
# the same schedule, arithmetic and arguments without the chain.  The chain
# gains in launches of few tiles and over longer caches, and ran slower in the
# launches of packs 1 and 2 of more tiles over few KV blocks (timed B200: 32
# query tokens, heads 32/16 and 32/32, 128 .. 4096 tiles over 128 .. 512 keys).
_CONTEXT_FP8_MULTI_WAVE_ROUTES: dict[str, dict[str, tuple[str, int, int]]] = {
    "sm100a": {
        "RUNTIME_PACK2_PAGE64_Q1": ("RUNTIME_PACK2_PAGE64_Q1_MW", 64, 384),
        "RUNTIME_PACK1_PAGE64_Q1": ("RUNTIME_PACK1_PAGE64_Q1_MW", 64, 384),
    },
}
# Per target, the runtime-size programs whose launches of one tile per CTA (at
# most as many tiles as the SMs) run the entry's program instead: the same
# schedule, arithmetic and arguments with stage 0 stored as soon as its last PV
# is in.  A persistent CTA's next tile overlaps that store with its first block
# (timed B200: in those launches the early store ran 2.5 % slower); a launch of
# one tile per CTA has no next tile (timed B200: 1 .. 2 requests of 200 .. 400
# query tokens over 1500 keys, heads 32/4, ran 0.5 .. 0.9 us faster).  The
# one-stage programs for every page size run the same build with the short-KV
# chain, as the page-64 ones do (timed B200: 1 .. 16 requests of 1 .. 33
# query tokens over 16 .. 129 keys, heads 8/1 .. 32/32, 1 .. 64 tiles, ran
# 0.35 .. 0.6 us faster).
_CONTEXT_FP8_ONE_WAVE_ROUTES: dict[str, dict[str, str]] = {
    "sm100a": {
        "RUNTIME_PACK8": "RUNTIME_PACK8_OW",
        "RUNTIME_PACK8_Q1": "RUNTIME_PACK8_Q1_OW",
        "RUNTIME_PACK4_Q1": "RUNTIME_PACK4_Q1_OW",
        "RUNTIME_PACK2_Q1": "RUNTIME_PACK2_Q1_OW",
        "RUNTIME_PACK1_Q1": "RUNTIME_PACK1_Q1_OW",
    },
}
# Per target, the most tiles of a one-wave launch of the one-stage programs the
# entries name that takes its one-wave program (fewer than the SMs): as for the
# page-64 programs (_CONTEXT_FP8_MULTI_WAVE_ROUTES), the chain ran slower in
# launches of packs 1 and 2 of more tiles (timed B200: 4 requests of 17 query
# tokens over 128 keys, heads 32/32, 128 tiles of pack 1, ran 0.08 us slower).
_CONTEXT_FP8_ONE_WAVE_TILES: dict[str, dict[str, int]] = {
    "sm100a": {"RUNTIME_PACK2_Q1": 64, "RUNTIME_PACK1_Q1": 64},
}
# Per target, (fewest tiles, most keys, fewest query tokens) of the launches
# whose two-way split the one-wave program outruns: a launch whose runtime
# program has a one-wave entry, whose tiles at that program's pack number at
# least the first and at most the SMs, over at most the keys and more than the
# query tokens, keeps its whole tiles where the split choice names a plan of
# two splits.  The cost model prices those splits below the whole tiles; timed
# B200 (1 .. 4 requests of 160 .. 200 query tokens over 1500 .. 1537 keys,
# heads 32/4, 64/8 and 16/2, 40 .. 56 tiles): the one-wave program ran
# 1.7 .. 3.8 % faster than the two-way split; over 1700 keys and more the split
# ran faster, and the three-way splits as fast or faster.
_CONTEXT_FP8_ONE_WAVE_SPLIT_BOUNDS: dict[str, tuple[int, int, int]] = {
    "sm100a": (40, 1664, 128),
}
_CONTEXT_FP8_ONE_STAGE_PROGRAMS = frozenset(
    name
    for routes in _CONTEXT_FP8_ONE_STAGE_ROUTES.values()
    for name in routes.values()
) | frozenset(
    name
    for routes in _CONTEXT_FP8_MULTI_WAVE_ROUTES.values()
    for name, _, _ in routes.values()
)
_CONTEXT_FP8_ONE_STAGE_PROGRAMS |= frozenset(
    name
    for routes in _CONTEXT_FP8_FORM_ONE_STAGE_ROUTES.values()
    for name in routes.values()
)
_CONTEXT_FP8_ONE_STAGE_PROGRAMS |= frozenset(
    name
    for routes in _CONTEXT_FP8_ONE_WAVE_ROUTES.values()
    for program, name in routes.items()
    if program in _CONTEXT_FP8_ONE_STAGE_PROGRAMS
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


# A runtime-size BF16 program takes the sizes after the BF16 parameters: the
# m-block count, the whole 8-unit L2 sections of the tile order (bit 31: the
# tail order, :data:`_CONTEXT_BF16_TAIL_INTERLEAVE`), the work units per
# request, the query heads, the GQA group and the log2 page size.
_CONTEXT_BF16_RUNTIME_PARAMS = (
    *_CONTEXT_HOST_PARAMS["context_bf16"],
    "m_blocks",
    "full_sections",
    "bh_units",
    "q_heads",
    "group",
    "page_shift",
)

# The tail order of a causal runtime-size BF16 launch, with or without sinks
# or an LSE output.  Its work units past the last whole 8-unit section run one
# unit's m-blocks after another, or, with this bit of ``full_sections`` set,
# interleaved like a section: every tail unit's m-block, the largest first.
# The order moves no tile's work, so the output is the same bit for bit; the
# launch takes the order of the shorter modelled makespan
# (:func:`_context_bf16_tail_interleave`).
_CONTEXT_BF16_TAIL_INTERLEAVE = 1 << 31
# The tail-order model: a tile costs its KV blocks plus this many; the
# interleaved order must be shorter by this fraction; a launch of more waves
# than this keeps the plain order.
_CONTEXT_BF16_TAIL_TILE_COST = 2
_CONTEXT_BF16_TAIL_MARGIN = 0.005
_CONTEXT_BF16_TAIL_MAX_WAVES = 32


def _context_bf16_tail_makespan(costs: list[int], num_ctas: int) -> int:
    """The finish time of a persistent launch over tiles of ``costs`` in tile
    order: CTA ``i`` runs tile ``i`` first and claims its next tile from the
    counter when it starts one."""

    first = min(num_ctas, len(costs))
    ends = costs[:first]
    claims = [(0, cta) for cta in range(first)]
    for cost in costs[first:]:
        _, cta = heapq.heappop(claims)
        start = ends[cta]
        ends[cta] = start + cost
        heapq.heappush(claims, (start, cta))
    return max(ends)


@functools.lru_cache(maxsize=1024)
def _context_bf16_tail_interleave(
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    units_per_batch: int,
    m_block_tokens: int,
    num_ctas: int,
) -> bool:
    """Whether a causal runtime-size BF16 launch runs its tail units
    interleaved: two or more of them, and the modelled makespan of that order
    shorter than the plain one's by :data:`_CONTEXT_BF16_TAIL_MARGIN`.  Every
    request at the planned maxima; a tile's KV blocks are those of its
    bottom-right causal bound (a bound below zero wraps to every block, as the
    kernel's unsigned bound does)."""

    total_bh = batch_size * units_per_batch
    full_sections, tail_units = divmod(total_bh, 8)
    m_blocks = max((max_q_len + m_block_tokens - 1) // m_block_tokens, 1)
    if (
        tail_units < 2
        or num_ctas <= 0
        or m_blocks * total_bh > _CONTEXT_BF16_TAIL_MAX_WAVES * num_ctas
    ):
        return False
    full_n = max((max_kv_len + 127) // 128, 1)
    shift = max_kv_len - max_q_len
    # tile costs per m-block, the largest (last) m-block first
    cost = []
    for m_block in reversed(range(m_blocks)):
        top = (m_block + 1) * m_block_tokens + shift
        blocks = full_n if top < 0 or top >= max_kv_len else (top + 127) // 128
        cost.append(_CONTEXT_BF16_TAIL_TILE_COST + max(blocks, 1))
    head = [
        cost[(tile >> 3) % m_blocks] for tile in range(8 * m_blocks * full_sections)
    ]
    tail = range(tail_units * m_blocks)
    plain = head + [cost[r % m_blocks] for r in tail]
    interleaved = head + [cost[r // tail_units] for r in tail]
    return _context_bf16_tail_makespan(interleaved, num_ctas) < (
        1.0 - _CONTEXT_BF16_TAIL_MARGIN
    ) * _context_bf16_tail_makespan(plain, num_ctas)


# A runtime-size NVFP4 program takes the sizes after the NVFP4 parameters: the
# scale source (0: the host scalars, 1: the device scalars), the m-block count,
# the whole 8-unit L2 sections of the tile order, the work units per request,
# the query heads and the GQA group.
_CONTEXT_NVFP4_RUNTIME_PARAMS = (
    *_CONTEXT_HOST_PARAMS["context_nvfp4"],
    "scale_source",
    "m_blocks",
    "full_sections",
    "bh_units",
    "q_heads",
    "heads_group",
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
        if component == "context_bf16":
            return _CONTEXT_BF16_RUNTIME_PARAMS
        if component == "context_nvfp4":
            return _CONTEXT_NVFP4_RUNTIME_PARAMS
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


def _aligned_hd64_route_program(route: CakeFmhaDecodeRoute) -> str | None:
    """The pinned program name of an aligned hd64 route, or ``None`` when the
    route's program identifier is not the module's."""

    if route.component != _ALIGNED_HD64_COMPONENT or route.aligned_program is None:
        return None
    return _ALIGNED_HD64_PROGRAMS.get(route.aligned_program)


def cake_fmha_aligned_hd64_launch_grid(
    program, batch_size: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """The launch grid of one aligned hd64 q1 decode: ``(1, num_kv_heads,
    batch_size)``, one CTA per (request, KV head) tile (``blockIdx.y`` the KV
    head, ``blockIdx.z`` the request); cluster launch control hands cancelled
    CTAs' coordinates to the persistent loop as further tiles.  ``program`` is
    the generated loader's ``CakeProgram`` record (its ``cluster`` is
    ``(1, 1, 1)``) and does not enter the grid."""

    return 1, int(num_kv_heads), int(batch_size)


def _aligned_hd256_route_program(
    route: CakeFmhaDecodeRoute,
) -> tuple[int, str] | None:
    """``(CTAs per tile, pinned program name)`` of an aligned hd256 route, or
    ``None`` when the route's program identifier is not one of the module's."""

    if route.component != _ALIGNED_HD256_COMPONENT or route.aligned_program is None:
        return None
    return _ALIGNED_HD256_PROGRAMS.get(route.aligned_program)


def cake_fmha_aligned_hd256_launch_grid(
    program, batch_size: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """The launch grid of one aligned hd256 q1 decode: ``(CTAs, num_kv_heads,
    batch_size)``, one CTA (the persistent program) or one cluster of 2 / 4 / 8 CTAs
    (``blockIdx.x`` the cluster rank) per (request, KV head) tile, ``blockIdx.y``
    the KV head and ``blockIdx.z`` the request.  ``program`` is the generated
    loader's ``CakeProgram`` record; its ``cluster`` is ``(CTAs, 1, 1)``.  The
    persistent program's cluster launch control hands cancelled CTAs'
    coordinates to its persistent loop as further tiles."""

    return int(program.cluster[0]), int(num_kv_heads), int(batch_size)


def _aligned_mtp_q32_route_program(
    route: CakeFmhaDecodeRoute,
) -> tuple[int, str] | None:
    """``(CTAs per tile, pinned program name)`` of an aligned speculative-decode
    route, or ``None`` when the route's program identifier is not the module's."""

    if route.component != _ALIGNED_MTP_Q32_COMPONENT or route.aligned_program is None:
        return None
    return _ALIGNED_MTP_Q32_PROGRAMS.get(route.aligned_program)


def cake_fmha_aligned_mtp_q32_launch_grid(
    program, batch_size: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """The launch grid of one aligned speculative decode: ``(CTAs, num_kv_heads,
    batch_size)``, one cluster of the program's CTAs per (request, KV head) tile
    (``blockIdx.x`` the cluster rank, ``blockIdx.y`` the KV head, ``blockIdx.z``
    the request); the tile's 32-row query tile holds every query token of the
    request.  ``program`` is the generated loader's ``CakeProgram`` record; its
    ``cluster`` is ``(CTAs, 1, 1)``."""

    return int(program.cluster[0]), int(num_kv_heads), int(batch_size)


def _route_components(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute,
) -> tuple[str, ...]:
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == _ALIGNED_MTP_Q32_COMPONENT
    ):
        return (route.aligned_program,) if route.aligned_program is not None else ()
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == _ALIGNED_HD256_COMPONENT
    ):
        return (route.aligned_program,) if route.aligned_program is not None else ()
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == _ALIGNED_HD64_COMPONENT
    ):
        return (route.aligned_program,) if route.aligned_program is not None else ()
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
        and route.component == _ALIGNED_MTP_Q32_COMPONENT
    ):
        return _aligned_mtp_q32_route_program(route) is not None
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == _ALIGNED_HD256_COMPONENT
    ):
        return _aligned_hd256_route_program(route) is not None
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == _ALIGNED_HD64_COMPONENT
    ):
        return _aligned_hd64_route_program(route) is not None
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


# ``stream_context_sm_count(stream, device)`` of the stream-context library
# (``csrc/cake_fmha_stream_context.cpp``) once loaded: the SM count of the CUDA
# context a raw stream handle runs on (a green context's partition), ``-1`` when
# the driver cannot tell; ``False`` when the library cannot be loaded (then a
# launch is planned on the device's SM count).  One FFI call per launch: the same
# driver queries through the Python bindings cost several microseconds of host
# time per launch.
_STREAM_CONTEXT_SM_COUNT: Any = None


def _stream_context_query() -> Any:
    global _STREAM_CONTEXT_SM_COUNT
    if _STREAM_CONTEXT_SM_COUNT is None:
        try:
            module = load_cake_fmha_stream_context_module()
            _STREAM_CONTEXT_SM_COUNT = module.stream_context_sm_count
        except Exception as error:  # noqa: BLE001
            warnings.warn(
                "cake_fmha: the stream-context library did not load, so small-M "
                "decode splits are planned on the device's SM count (a launch on "
                f"an SM partition may not complete): {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            _STREAM_CONTEXT_SM_COUNT = False
    return _STREAM_CONTEXT_SM_COUNT


# Whether a small-M launch plans its split count on the SMs of its stream's
# CUDA context (a green context's partition) instead of the device's.  On by
# default: FlashInfer cannot observe every partition a launch may run on (a
# torch green context created, or its stream taken, before FlashInfer was
# imported; torch 2.13's ``GreenContext.create`` returns the C++ base class, so
# no Python wrapper sees its ``Stream()``; a green context of the driver API or
# another library), and a split count planned on more SMs than the stream holds
# never completes.  ``FLASHINFER_CAKE_FMHA_SM_PARTITIONS=0`` or
# ``enable_cake_fmha_sm_partitions(False)`` plans on the device's SM count
# without reading the stream, for a process that never launches on a partition;
# FlashInfer's green-context helpers (``flashinfer.green_ctx``) turn it back on.
_SMALLM_SM_PARTITIONS = os.environ.get("FLASHINFER_CAKE_FMHA_SM_PARTITIONS") != "0"


def enable_cake_fmha_sm_partitions(enabled: bool = True) -> None:
    """Plan small-M decode splits on the SMs of the launch stream's CUDA
    context (the default), or with ``enabled=False`` on the device's SM count
    without reading the stream.  Turn it off only in a process that never
    launches the small-M route on an SM partition: a split count planned on
    more SMs than the stream's context holds never completes.  Creating
    green-context streams with ``flashinfer.green_ctx`` turns it back on."""

    global _SMALLM_SM_PARTITIONS
    _SMALLM_SM_PARTITIONS = bool(enabled)


def _smallm_launch_sm_count(
    device: torch.device, device_sm_count: int | None = None
) -> int:
    """The SMs a small-M launch on ``device``'s current torch stream can hold
    at once: the device's (:func:`_smallm_sm_count`, or ``device_sm_count``
    when the caller has it), bounded by the SMs of the stream's CUDA context (a
    green context's partition).  Every split of a small-M tile waits for all of
    them, so a split count planned on more SMs than the stream's context holds
    never completes.

    Every launch reads its stream's context: the legacy and per-thread default
    streams run on the thread's current context, any other stream on its own
    (``cuStreamGetCtx``; where that does not answer, before CUDA 13 for a stream
    of ``cuGreenCtxStreamCreate``, its green context).  The device's primary
    context is retained once and known by its handle, so a launch on one of its
    streams makes one driver call; any other context's count is cached per
    context ID (``cuCtxGetId``), which the driver never reuses, unlike a
    destroyed context's or stream's handle.  The stream is not keyed by
    ``cuStreamGetId``: on a stream being captured into a CUDA graph that query
    returns ``CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`` and invalidates the
    capture (driver 580, in global and relaxed capture modes alike), while the
    context queries are legal during a capture."""

    count = _smallm_sm_count(device) if device_sm_count is None else device_sm_count
    if device.type != "cuda":
        return count
    query = _STREAM_CONTEXT_SM_COUNT
    if query is None:
        query = _stream_context_query()
    if not query:
        return count
    index = device.index if device.index is not None else torch.cuda.current_device()
    partition = query(torch._C._cuda_getCurrentRawStream(index), index)
    return count if partition < 0 or partition >= count else partition


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


# ---------------------------------------------------------------------------
# Aligned hd64 q1 decode selector: a Python mirror of the source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_forgen_bf16_hd64_grouped`` of the ``attention_decode_bf16_paged``
# family, source revision ``8edb03f0787cf47aa6febe578d99f2840096fa8f``:
# ``SEL_HD64_*``, ``aligned_hd64_selected``).  The constants are copied verbatim
# (names and values) and the function keeps the source's signature and
# semantics; change the source first, then re-copy.
# ``tests/attention/test_cake_fmha_aligned_hd64.py`` holds the source selector
# test's expectations.  ``True`` selects the aligned hd64 program
# (``decode_aligned_bf16_hd64``); ``False`` keeps the module's present route
# (``decode_balanced_bf16_hd64``).  The bands are read per program target
# (``arch``: ``sm_100a`` / ``sm_103a``) on ``tiles = batch_size * num_kv_heads``
# and ``max_kv = max_seq_len_kv``: the pinned points first, then the band
# rectangles (inclusive bounds at probed values); every other shape, and every
# shape outside the bands, keeps the present route.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Aligned hd64 q1 grouped decode selector: a Python mirror of the source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_forgen_bf16_hd64_grouped`` of the ``attention_decode_bf16_paged``
# family: ``SEL_HD64_*``, ``aligned_hd64_selected``).  The
# constants are copied verbatim (names and values) and the function keeps the
# source's signature and semantics; change the source first, then re-copy.
# ``tests/attention/test_cake_fmha_aligned_hd64.py`` holds the source selector
# test's expectations.  In this module ``True`` routes ``decode_aligned_bf16_hd64``
# and ``False`` keeps the present route ``decode_balanced_bf16_hd64``; ``arch`` is
# the source's target spelling (``_ALIGNED_PROGRAM_TARGET`` maps the device target
# at the call site).
# ---------------------------------------------------------------------------
SEL_HD64_ARCHS = ("sm_100a", "sm_103a")
# Query heads per KV head of the measured form (one 8-row Q tile holds the KV head's
# query heads); other groups keep the present route.
SEL_HD64_GROUP = 8
# Pinned points per target: ((tiles, max_kv), program) with "X" = this program and "P" =
# the present route; evaluated before the bands.
SEL_HD64_PINS = {
    "sm_100a": (
        ((16, 237569), "P"),  # agentx_b16_hkv1
        ((32, 237569), "P"),  # agentx_b32_hkv1
        ((64, 237569), "P"),  # agentx_b64_hkv1
        ((128, 237569), "P"),  # agentx_b128_hkv1
        ((192, 237569), "P"),  # agentx_b192_hkv1
        ((256, 237569), "P"),  # agentx_b256_hkv1
        ((256, 130588), "P"),  # random_128_128k_b256_hkv1
        ((512, 65369), "P"),  # random_128_65k_b64_hkv8
        ((32, 65536), "P"),  # uniform_b4_hkv8_s65536
        ((128, 65536), "X"),  # uniform_b16_hkv8_s65536
        ((512, 32768), "X"),  # uniform_b64_hkv8_s32768
        ((1024, 32768), "P"),  # uniform_b128_hkv8_s32768
        ((1024, 4096), "X"),  # uniform_b128_hkv8_s4096
    ),
    "sm_103a": (
        ((16, 237569), "P"),  # agentx_b16_hkv1
        ((32, 237569), "P"),  # agentx_b32_hkv1
        ((64, 237569), "P"),  # agentx_b64_hkv1
        ((128, 237569), "P"),  # agentx_b128_hkv1
        ((192, 237569), "P"),  # agentx_b192_hkv1
        ((256, 237569), "P"),  # agentx_b256_hkv1
        ((256, 130588), "P"),  # random_128_128k_b256_hkv1
        ((512, 65369), "P"),  # random_128_65k_b64_hkv8
        ((32, 65536), "P"),  # uniform_b4_hkv8_s65536
        ((128, 65536), "X"),  # uniform_b16_hkv8_s65536
        ((512, 32768), "P"),  # uniform_b64_hkv8_s32768
        ((1024, 32768), "X"),  # uniform_b128_hkv8_s32768
        ((1024, 4096), "X"),  # uniform_b128_hkv8_s4096
    ),
}
# Band rectangles per target: (tiles_lo, tiles_hi, kv_lo, kv_hi), inclusive bounds at
# probed values; a launch is in a band iff tiles_lo <= tiles <= tiles_hi and
# kv_lo <= max_kv <= kv_hi; outside every band the present route serves.  An empty
# tuple = pins only on that target.
SEL_HD64_BANDS = {
    "sm_100a": (
        (64, 64, 2048, 4096),
        (128, 128, 8192, 131072),
        (384, 384, 8192, 65536),
        (384, 512, 8192, 32768),
        (384, 768, 8192, 16384),
        (384, 2048, 8192, 8192),
        (768, 768, 4096, 16384),
        (768, 2048, 4096, 8192),
        (1536, 1536, 2048, 65536),
        (1536, 2048, 2048, 16384),
    ),
    "sm_103a": (
        (64, 64, 2048, 2048),
        (128, 128, 8192, 131072),
        (128, 2048, 8192, 8192),
        (384, 384, 8192, 65536),
        (384, 512, 8192, 16384),
        (768, 2048, 4096, 8192),
        (1024, 2048, 4096, 65536),
        (1536, 2048, 2048, 65536),
    ),
}


def aligned_hd64_selected(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len_kv: int,
    *,
    q_len: int,
    num_q_heads: int,
    arch: str,
) -> bool:
    """Whether one launch takes this program (``True``) or the host's present route
    (``False``), from host metadata only.

    ``False`` unless ``q_len == 1``, exactly ``SEL_HD64_GROUP`` query heads per KV head and
    ``arch`` in ``SEL_HD64_ARCHS``.  Otherwise the target's pinned points decide first, then
    its band rectangles on ``tiles = batch_size * num_kv_heads`` and ``max_seq_len_kv``:
    exactly one verdict.  Selection is a performance choice only: the program handles any
    length up to ``max_pages_per_seq * PAGE_SIZE``.
    """
    if q_len != 1 or num_q_heads != SEL_HD64_GROUP * num_kv_heads:
        return False
    if arch not in SEL_HD64_ARCHS:
        return False
    tiles = batch_size * num_kv_heads
    for (pin_tiles, pin_kv), program in SEL_HD64_PINS[arch]:
        if pin_tiles == tiles and pin_kv == max_seq_len_kv:
            return program == "X"
    return any(
        tiles_lo <= tiles <= tiles_hi and kv_lo <= max_seq_len_kv <= kv_hi
        for tiles_lo, tiles_hi, kv_lo, kv_hi in SEL_HD64_BANDS[arch]
    )


# ---------------------------------------------------------------------------
# Aligned hd256 q1 grouped decode selector: a Python mirror of the source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_forgen_bf16_hd256_grouped`` of the ``attention_decode_bf16_paged``
# family: ``SEL_HD256_*``, ``aligned_hd256_program``).  The
# constants are copied verbatim (names and values) and the function keeps the
# source's signature and semantics; change the source first, then re-copy.
# ``None`` keeps the present route ``decode_balanced_bf16_hd256``; ``("persistent", 1)``
# selects the aligned persistent hd256 program and ``("cga", N)`` the aligned cluster
# program of cluster size N.  ``arch`` is the source's target spelling and ``sm_count``
# the device's multiprocessor count.
#
# Copied at source revision ``38d7a8878e025385ab9f995b1b0ae5d722685dc2``;
# ``tests/attention/test_cake_fmha_aligned_hd256.py`` holds the source selector
# test's expectations.
# ---------------------------------------------------------------------------
SEL_HD256_ARCHS = ("sm_100a", "sm_103a")
# Query heads per KV head of the measured form (one 8-row Q tile holds the KV head's
# query heads); other groups keep the present route.
SEL_HD256_GROUP = 8
# FlashInfer's trtllm-gen hd256 decode splits a sequence's KV over at most
# ceil(max_kv / SEL_HD256_KV_PER_CTA_SPLIT) CTAs (twice its kernels' 256-token KV step).
SEL_HD256_KV_PER_CTA_SPLIT = 512
# Pinned points per target: ((tiles, max_kv), program) with None = the present route,
# ("persistent", 1) = this program and ("cga", N) = the cluster-of-N program; evaluated
# before the bands.
SEL_HD256_PINS = {
    "sm_100a": (  # derived at 148 SMs
        ((16, 237569), None),  # agentx_b16_hkv1
        ((32, 237569), None),  # agentx_b32_hkv1
        ((64, 237569), None),  # agentx_b64_hkv1
        ((128, 237569), None),  # agentx_b128_hkv1
        ((192, 237569), None),  # agentx_b192_hkv1
        ((256, 237569), None),  # agentx_b256_hkv1
        ((256, 130588), None),  # random_128_128k_b256_hkv1
        ((512, 65369), None),  # random_128_65k_b64_hkv8
        ((32, 65536), ("cga", 4)),  # uniform_b4_hkv8_s65536
        ((128, 65536), ("persistent", 1)),  # uniform_b16_hkv8_s65536
        ((512, 32768), ("persistent", 1)),  # uniform_b64_hkv8_s32768
        ((1024, 32768), ("persistent", 1)),  # uniform_b128_hkv8_s32768
        ((1024, 4096), ("persistent", 1)),  # uniform_b128_hkv8_s4096
        ((8, 4096), ("cga", 8)),  # uniform_b1_hkv8_s4096
    ),
    "sm_103a": (  # derived at 152 SMs
        ((16, 237569), None),  # agentx_b16_hkv1
        ((32, 237569), None),  # agentx_b32_hkv1
        ((64, 237569), None),  # agentx_b64_hkv1
        ((128, 237569), None),  # agentx_b128_hkv1
        ((192, 237569), None),  # agentx_b192_hkv1
        ((256, 237569), None),  # agentx_b256_hkv1
        ((256, 130588), None),  # random_128_128k_b256_hkv1
        ((512, 65369), None),  # random_128_65k_b64_hkv8
        ((32, 65536), ("cga", 4)),  # uniform_b4_hkv8_s65536
        ((128, 65536), ("persistent", 1)),  # uniform_b16_hkv8_s65536
        ((512, 32768), None),  # uniform_b64_hkv8_s32768
        ((1024, 32768), ("persistent", 1)),  # uniform_b128_hkv8_s32768
        ((1024, 4096), ("persistent", 1)),  # uniform_b128_hkv8_s4096
        ((8, 4096), ("cga", 8)),  # uniform_b1_hkv8_s4096
    ),
}
# Band rectangles per target: ((tiles_lo, tiles_hi, kv_lo, kv_hi), program), inclusive
# bounds at probed values; a launch is in a band iff tiles_lo <= tiles <= tiles_hi and
# kv_lo <= max_kv <= kv_hi; outside every band the present route serves.  An empty
# tuple = pins only on that target.
SEL_HD256_BANDS = {
    "sm_100a": (
        ((128, 256, 131072, 131072), ("persistent", 1)),
        ((128, 384, 2048, 65536), ("persistent", 1)),
        ((128, 2048, 2048, 8192), ("persistent", 1)),
        ((1024, 2048, 2048, 65536), ("persistent", 1)),
        ((64, 64, 2048, 65536), ("cga", 2)),
    ),
    "sm_103a": (
        ((128, 256, 131072, 131072), ("persistent", 1)),
        ((128, 384, 2048, 65536), ("persistent", 1)),
        ((128, 512, 2048, 8192), ("persistent", 1)),
        ((128, 2048, 2048, 2048), ("persistent", 1)),
        ((1024, 2048, 2048, 65536), ("persistent", 1)),
        ((64, 64, 2048, 65536), ("cga", 2)),
    ),
}


def aligned_hd256_program(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len_kv: int,
    *,
    q_len: int,
    num_q_heads: int,
    arch: str,
    sm_count: int,
) -> "tuple[str, int] | None":
    """The aligned hd256 program of one launch, or ``None`` for the host's present route,
    from host metadata only.

    ``None`` unless ``q_len == 1``, exactly ``SEL_HD256_GROUP`` query heads per KV head and
    ``arch`` in ``SEL_HD256_ARCHS``.  Otherwise the target's pinned points decide first, then
    its band rectangles on ``tiles = batch_size * num_kv_heads`` and ``max_seq_len_kv``.  A
    program ``("persistent", 1)`` or ``("cga", N)`` is returned only when FlashInfer's
    trtllm-gen split at ``sm_count`` SMs picks the same kernel class: ``N == min(ceil(max_kv /
    SEL_HD256_KV_PER_CTA_SPLIT), max(1, sm_count // tiles))``.  Selection is a performance
    choice only: each program handles any length up to ``max_pages_per_seq * PAGE_SIZE``.
    """
    if q_len != 1 or num_q_heads != SEL_HD256_GROUP * num_kv_heads:
        return None
    if arch not in SEL_HD256_ARCHS:
        return None
    tiles = batch_size * num_kv_heads
    if tiles <= 0 or max_seq_len_kv <= 0 or sm_count <= 0:
        return None
    program = None
    for (pin_tiles, pin_kv), pinned in SEL_HD256_PINS[arch]:
        if pin_tiles == tiles and pin_kv == max_seq_len_kv:
            program = pinned
            break
    else:
        for (tiles_lo, tiles_hi, kv_lo, kv_hi), band in SEL_HD256_BANDS[arch]:
            if tiles_lo <= tiles <= tiles_hi and kv_lo <= max_seq_len_kv <= kv_hi:
                program = band
                break
    if program is None:
        return None
    kv_ctas = -(-max_seq_len_kv // SEL_HD256_KV_PER_CTA_SPLIT)
    ctas_kv = min(kv_ctas, max(1, sm_count // tiles))
    return program if ctas_kv == program[1] else None


# ---------------------------------------------------------------------------
# Aligned speculative-decode grouped decode selector: a Python mirror of the
# source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_forgen_bf16_q32_grouped_cga`` of the
# ``attention_decode_bf16_paged_spec`` family: ``SEL_Q32CGA_*``,
# ``spec_q32_cga_program``).  The constants and comments are copied verbatim
# (names and values; the formatter re-wraps long code lines) and the function
# keeps the source's signature and semantics; change the source first, then
# re-copy.  ``None`` keeps the present route ``decode_balanced_bf16``;
# ``("cga", N)`` selects the aligned speculative-decode program of cluster size
# N.  ``arch`` is the source's target spelling and ``sm_count`` the device's
# multiprocessor count.
#
# Copied at source revision ``1e504c05e3b59b960735759fc085c5edcc606da1``;
# ``tests/attention/test_cake_fmha_aligned_mtp_q32.py`` holds the source selector
# test's expectations.
# ---------------------------------------------------------------------------
SEL_Q32CGA_ARCHS = ("sm_100a", "sm_103a")
# Query heads per KV head of the port (its kernel_requires); other groups keep the present route.
SEL_Q32CGA_GROUP = 8
# Query tokens per request the port's 32-row Q tile holds (its kernel_requires: 1 <= q_len <= 4).
SEL_Q32CGA_MAX_Q_LEN = 4
# FlashInfer's trtllm-gen split puts a sequence's KV on at most ceil(max_kv / SEL_Q32CGA_KV_PER_CTA_SPLIT) CTAs
# (twice the class's 256-token KV step).
SEL_Q32CGA_KV_PER_CTA_SPLIT = 512
# FlashInfer takes the CGA reduction only for a split of 2 to SEL_Q32CGA_MAX_CGA_CTAS CTAs (the largest cluster).
SEL_Q32CGA_MAX_CGA_CTAS = 16
# Pinned points per target: ((batch_size, num_kv_heads, max_seq_len_kv, q_len), program), program ("cga", N) = the
# cluster-of-N config; an empty tuple = the present route everywhere on that target.
SEL_Q32CGA_PINS = {
    "sm_100a": (  # derived at 148 SMs
        ((8, 8, 4096, 3), ("cga", 2)),  # uniform_b8_hkv8_s4096_mtp3
    ),
    "sm_103a": (),  # derived at 152 SMs: the present route serves the mtp3 row
}


def spec_q32_cga_program(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len_kv: int,
    *,
    q_len: int,
    num_q_heads: int,
    arch: str,
    sm_count: int,
) -> "tuple[str, int] | None":
    """The cluster config of this program for one launch, or ``None`` for the host's present route, from host
    metadata only.

    ``None`` unless ``1 <= q_len <= SEL_Q32CGA_MAX_Q_LEN``, exactly ``SEL_Q32CGA_GROUP`` query heads per KV head and
    ``arch`` in ``SEL_Q32CGA_ARCHS``.  Otherwise the target's pinned points on ``(batch_size, num_kv_heads,
    max_seq_len_kv, q_len)`` decide.  A program ``("cga", N)`` is returned only when FlashInfer's trtllm-gen split at
    ``sm_count`` SMs picks the CGA class at the same cluster size: ``N == min(ceil(max_kv /
    SEL_Q32CGA_KV_PER_CTA_SPLIT), max(1, sm_count // tiles))`` with ``tiles = batch_size * num_kv_heads`` (one Q tile
    per (request, KV head) at ``q_len <= 4``, so the split's base CTAs are ``tiles``) and ``1 < N <=
    SEL_Q32CGA_MAX_CGA_CTAS``.  Selection is a performance choice only: each config handles any lengths up to
    ``max_pages_per_seq * PAGE_SIZE``.
    """
    if not 1 <= q_len <= SEL_Q32CGA_MAX_Q_LEN:
        return None
    if num_q_heads != SEL_Q32CGA_GROUP * num_kv_heads:
        return None
    if arch not in SEL_Q32CGA_ARCHS:
        return None
    tiles = batch_size * num_kv_heads
    if batch_size <= 0 or num_kv_heads <= 0 or max_seq_len_kv <= 0 or sm_count <= 0:
        return None
    program = None
    point = (batch_size, num_kv_heads, max_seq_len_kv, q_len)
    for pin, pinned in SEL_Q32CGA_PINS[arch]:
        if pin == point:
            program = pinned
            break
    if program is None:
        return None
    kv_ctas = -(-max_seq_len_kv // SEL_Q32CGA_KV_PER_CTA_SPLIT)
    ctas_kv = min(kv_ctas, max(1, sm_count // tiles))
    if not 1 < ctas_kv <= SEL_Q32CGA_MAX_CGA_CTAS:
        return None
    return program if ctas_kv == program[1] else None


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
    if pages_per_block == 8:
        # The BF16 binding pads a table its 16-byte page-id loads cannot read
        # in place too (a width of a multiple of four pages, 16-byte base).
        padded_pages = max(required_pages, (source_pages + 3) // 4 * 4)
        needs_page_padding = (
            source_pages < required_pages
            or source_pages % 4 != 0
            or block_tables.data_ptr() % 16 != 0
        )

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
    route on its target (a length-aware twin runs where its stream-K
    program is routed)."""

    if route.program is None:
        return None
    routes = _DECODE_PROGRAM_ROUTES.get(route.target, {})
    routed = {(key[0], served) for key, served in routes.items()}
    routed |= {
        twin
        for program, twin in _DECODE_PROGRAM_LENGTH_AWARE.items()
        if program in routed
    }
    if route.program not in routed:
        return None
    return _DECODE_PROGRAMS.get(route.program)


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
    whole_device: bool = True,
) -> tuple[tuple[str, str], int] | None:
    """``(program, num_split)`` of a small-M launch the generated programs
    serve, or ``None``: the module's split rule, except one 64-row tile of
    eight query tokens over 64-token pages whose KV is exactly nine 128-token
    blocks, which the nine-CTA cluster program merges in distributed shared
    memory.  The cluster program needs ``whole_device``: the launch's stream
    holds every SM of the device.  A non-portable nine-CTA cluster needs nine
    SMs of one GPC, which an SM partition (a green context) need not hold, and
    its launch then fails; the binding does not expose the kernel to an
    occupancy query, so a partition takes the split rule."""

    if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return None
    cluster = 0
    # The cluster program's precondition: one tile (one request, one KV head)
    # whose KV fits its nine 128-token blocks, launched where its nine-CTA
    # cluster is resident (``whole_device``).
    if (
        whole_device
        and (n_rows, page_size, q_len, group, tiles) == (64, 64, 8, 8, 1)
        and 8 * 128 < max_seq_len <= 9 * 128
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
        ("decode_bf16", "host", 128, 1, 32, 4, 16, False, True): "DEC_B128",
        ("decode_bf16", "device", 128, 1, 32, 4, 16, False, True): "DEC_B128_DS",
        ("decode_bf16", "host", 256, 1, 32, 4, 16, True, True): "DEC_SINK",
        ("decode_bf16", "device", 256, 1, 32, 4, 16, True, True): "DEC_SINK_DS",
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
        ("decode_bf16", "host", 128, 1, 32, 4, 16, False, True): "DEC_B128",
        ("decode_bf16", "device", 128, 1, 32, 4, 16, False, True): "DEC_B128_DS",
        ("decode_bf16", "host", 256, 1, 32, 4, 16, True, True): "DEC_SINK",
        ("decode_bf16", "device", 256, 1, 32, 4, 16, True, True): "DEC_SINK_DS",
    }
)
_DECODE_PROGRAMS.update(
    {
        ("decode_bf16", "DEC_B128"): (
            "cake_fmha_decode_bf16_bc385ebf14604aab347e",
            "streamk",
        ),
        ("decode_bf16", "DEC_B128_DS"): (
            "cake_fmha_decode_bf16_0e44456c4d63adb24711",
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
        ("decode_bf16", "DEC_B128_LA"): (
            "cake_fmha_decode_bf16_c64d254f01952ec24d95",
            "streamk",
        ),
        ("decode_bf16", "DEC_B128_LA_DS"): (
            "cake_fmha_decode_bf16_fdf3768a052c6ca9cd83",
            "streamk",
        ),
    }
)
# The length-aware twins of the stream-K programs: the same launch with each
# request's units sized by its own length instead of ``max_seq_len``, so a
# batch whose lengths differ streams only its requests' own KV blocks.  A
# call runs the twin only when it asks for the length-aware schedule
# (``length_aware_schedule``); its default route keeps the stream-K program.
_DECODE_PROGRAM_LENGTH_AWARE: dict[tuple[str, str], tuple[str, str]] = {
    ("decode_bf16", "DEC_B128"): ("decode_bf16", "DEC_B128_LA"),
    ("decode_bf16", "DEC_B128_DS"): ("decode_bf16", "DEC_B128_LA_DS"),
}


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


def _decode_bf16_program_padded_table(
    workspace_bytes: int, batch_size: int, max_seq_len: int
) -> tuple[int, int]:
    """``(byte offset, pages per row)`` of the page table a stream-K or BHQ
    BF16 decode program reads when the caller's table does not fit its
    kernels: eight pages per evened 128-token block, at least four blocks, in
    the last 256-byte-aligned bytes of the workspace (a negative offset: the
    workspace cannot hold it).  The columns past the caller's width hold page
    0, as the JIT binding pads them."""

    blocks = (max_seq_len + 127) // 128
    pages = 8 * max(4, blocks + blocks % 2)
    return (workspace_bytes - batch_size * pages * 4) // 256 * 256, pages


def _decode_bf16_program_table_fits(
    launch: str, block_tables: torch.Tensor, pages: int
) -> bool:
    """Whether a program of ``launch`` reads the caller's page table in place:
    at least ``pages`` wide, and for a stream-K program (16-byte page-id
    loads) a width of a multiple of four pages on a 16-byte-aligned base."""

    width = int(block_tables.shape[-1])
    if width < pages:
        return False
    return launch != "streamk" or (width % 4 == 0 and block_tables.data_ptr() % 16 == 0)


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
    pad_table: bool = False,
    length_aware_schedule: bool = False,
) -> tuple[tuple[str, str], bool] | None:
    """``(program, device_scales)`` of a BF16 decode launch the generated
    programs serve, or ``None``.  With ``length_aware_schedule`` a stream-K
    hit runs its length-aware twin (:data:`_DECODE_PROGRAM_LENGTH_AWARE`).

    A hit takes a contiguous query, 16-token HND pages of head dim 128, a
    contiguous int32 shared page table the kernels read in place (eight
    pages per evened 128-token block, at least four blocks; a stream-K
    program's 16-byte loads also need a width of a multiple of four pages,
    :func:`_decode_bf16_program_table_fits`), int32 lengths (the kernels
    derive each request's causal prefix from them), no LSE, window, block
    scales or output scale, one scale form (host: bmm2 1) and a workspace
    holding the kernels' LSE scratch.  With ``pad_table`` a stream-K or BHQ
    program also takes a table that does not fit, read as a padded copy at
    the end of the workspace past its scratch
    (:func:`_decode_bf16_program_padded_table`); the copy is a separate
    launch, so the route asks for it only when no JIT member serves the
    launch.  A stream-K program also needs a batch of at least two requests
    per CTA group and a workspace holding its split scratch and counters
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
            # The end of the launch's own workspace scratch (LSE, then a
            # stream-K program's split scratch and counters).
            end = (batch_size * 4 + 15) // 16 * 16
            end += int(query.shape[0]) * num_q_heads * 4
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
                if length_aware_schedule:
                    program = _DECODE_PROGRAM_LENGTH_AWARE.get(program, program)
            if not _decode_bf16_program_table_fits(launch, block_tables, 8 * even) and (
                not pad_table
                or launch == "cluster"
                or _decode_bf16_program_padded_table(
                    _decode_program_bytes_of(workspace_buffer), batch_size, max_seq_len
                )[0]
                < end
            ):
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
    table in place (row stride its width), or a table that does not fit them
    as a copy padded with page 0 at the end of the workspace
    (:func:`_decode_bf16_program_padded_table`), each request's length (its causal
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
    flat = _decode_program_bytes(workspace_buffer)
    page_table = block_tables
    offset, pages = _decode_bf16_program_padded_table(
        flat.numel(), route.batch_size, max_seq_len
    )
    if not _decode_bf16_program_table_fits(module.launch, block_tables, pages):
        width = min(int(block_tables.shape[-1]), pages)
        page_table = _decode_program_workspace(
            flat, offset, route.batch_size * pages * 4, torch.int32
        ).view(route.batch_size, pages)
        page_table[:, :width].copy_(block_tables[:, :width])
        page_table[:, width:].zero_()
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
            flat,
            lse_offset,
            rows * 4,
            torch.float32,
        ),
        "page_table": page_table,
        "causal_seqlens_kv_global": seq_lens,
        "scale_log2_ptr": scale_ptr,
        "bmm2_ptr": bmm2_ptr,
        "sinks_ptr": unread if sinks is None else sinks,
        "max_pages_per_seq": int(page_table.shape[-1]),
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
    length_aware_schedule: bool = False,
) -> CakeFmhaDecodeRoute | None:
    """Select an exact product decode route without broadening its contract.

    ``enable_pdl`` is the caller's launch mode; a generated program whose split-KV
    reducer waits on the decode grid serves only a call that asks for it.

    ``length_aware_schedule`` is the caller's opt-in to the length-aware twin of
    a stream-K BF16 program (:data:`_DECODE_PROGRAM_LENGTH_AWARE`); every other
    route ignores it.

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
            # head_dim 64, eight query heads per KV head, one query token: the
            # aligned grouped program (one persistent CTA per (request, KV head)
            # tile) where the source selector's bands for this target read it
            # faster than the balanced route below.  It reads the K / V pools
            # and the query in place (contiguous; its TMA maps derive every
            # stride from the shapes) with an int32 page table and lengths.
            if (
                group == 8
                and key_cache.is_contiguous()
                and value_cache.is_contiguous()
                and block_tables.dtype == torch.int32
                and seq_lens.dtype == torch.int32
                and aligned_hd64_selected(
                    batch_size,
                    num_kv_heads,
                    max_seq_len,
                    q_len=q_len,
                    num_q_heads=num_q_heads,
                    arch=_ALIGNED_PROGRAM_TARGET[_cake_fmha_target(device)],
                )
            ):
                return route(
                    _ALIGNED_HD64_COMPONENT,
                    selected_page_size=16,
                    aligned_program=_ALIGNED_HD64_PROGRAM_ID,
                )
            # head_dim 64 (Q16Kv128 ForGen body, 1..16 query heads per KV head)
            # on the on-device balanced scheduler; the aligned hd64 program
            # above and this route are the two product routes of head_dim 64.
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
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            # The small-M split count is planned on the SMs the launch's
            # stream holds (a green context's partition); the other routes
            # keep the device's SM count.  Only a launch this route can take
            # reads its stream, and more tiles than the device's SMs take no
            # split on any partition, so such a launch reads none either.
            smallm_launch_sm_count = (
                _smallm_launch_sm_count(device, smallm_sm_count)
                if _SMALLM_SM_PARTITIONS and 0 < smallm_tiles <= smallm_sm_count
                else smallm_sm_count
            )
            smallm_num_split = _smallm_num_split(
                tiles=smallm_tiles,
                sm_count=smallm_launch_sm_count,
                max_seq_len=max_seq_len,
                n_rows=smallm_rows,
            )
            if smallm_num_split >= 1 and _smallm_workspace_supported(
                workspace_buffer,
                query,
                tiles=smallm_tiles,
                num_split=smallm_num_split,
                n_rows=smallm_rows,
                lse=lse,
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
                    whole_device=smallm_launch_sm_count >= smallm_sm_count,
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
            # head_dim 256, eight query heads per KV head, one query token,
            # 16-token pages: the aligned grouped programs where the source
            # selector's bands for this target read one faster than the
            # balanced route below and the trtllm-gen split at this device's
            # SM count picks that program's kernel class (the persistent
            # program, or the cluster program of 2 / 4 / 8 CTAs).  They read the
            # K / V pools and the query in place (contiguous; their TMA maps
            # derive every stride from the shapes) with an int32 page table
            # and lengths.
            if (
                q_len == 1
                and group == 8
                and page_size == 16
                and key_cache.is_contiguous()
                and value_cache.is_contiguous()
                and block_tables.dtype == torch.int32
                and seq_lens.dtype == torch.int32
            ):
                aligned_hd256 = aligned_hd256_program(
                    batch_size,
                    num_kv_heads,
                    max_seq_len,
                    q_len=q_len,
                    num_q_heads=num_q_heads,
                    arch=_ALIGNED_PROGRAM_TARGET[_cake_fmha_target(device)],
                    sm_count=smallm_sm_count,
                )
                if aligned_hd256 in _ALIGNED_HD256_PROGRAM_IDS:
                    return route(
                        _ALIGNED_HD256_COMPONENT,
                        selected_page_size=16,
                        aligned_program=_ALIGNED_HD256_PROGRAM_IDS[aligned_hd256],
                    )
            # head_dim 256 outside the small-M route's packed tiles and the
            # aligned programs above: every query row is its own tile (uniform
            # q_len 1..8) on the on-device balanced scheduler, one program per
            # page size (16 / 32 / 64).
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
            # Eight query heads per KV head and 1 to 4 query tokens per request
            # (``q_len x 8`` rows: one 32-row query tile per (request, KV
            # head)): the aligned speculative-decode program where the source
            # selector pins it faster than the balanced route below and the
            # trtllm-gen split at this device's SM count picks its cluster size.
            # It reads the K / V pools and the query in place (contiguous; its
            # TMA maps derive every stride from the shapes) with an int32 page
            # table and lengths.
            if (
                key_cache.is_contiguous()
                and value_cache.is_contiguous()
                and block_tables.dtype == torch.int32
                and seq_lens.dtype == torch.int32
            ):
                aligned_mtp = spec_q32_cga_program(
                    batch_size,
                    num_kv_heads,
                    max_seq_len,
                    q_len=q_len,
                    num_q_heads=num_q_heads,
                    arch=_ALIGNED_PROGRAM_TARGET[_cake_fmha_target(device)],
                    sm_count=smallm_sm_count,
                )
                if aligned_mtp in _ALIGNED_MTP_Q32_PROGRAM_IDS:
                    return route(
                        _ALIGNED_MTP_Q32_COMPONENT,
                        selected_page_size=16,
                        aligned_program=_ALIGNED_MTP_Q32_PROGRAM_IDS[aligned_mtp],
                    )
            # Long-context / ragged decode: the on-device balanced scheduler
            # splits long requests into chunks and merges them on device.
            return route("decode_balanced_bf16", selected_page_size=16)
        # Generated BF16 programs (host or device scales, sinks), ahead of the
        # native JIT members; a launch they do not serve keeps those.  An LSE
        # launch is selected as the same launch without it and decided at the
        # native member below.
        program_kwargs = dict(
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
            length_aware_schedule=length_aware_schedule,
        )
        bf16_program = _decode_bf16_program_select(device, **program_kwargs)
        native = None
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
                native = None
        if bf16_program is None and lse is None and native is None:
            # No JIT member serves the launch: a program whose kernels read
            # more of the page table than the caller's holds takes it with a
            # padded copy of the table (one more launch, still far from the
            # compat route).
            bf16_program = _decode_bf16_program_select(
                device, **program_kwargs, pad_table=True
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
        if native is None:
            return None
        # The LSE launch keeps the program route of the launch without it
        # when that route's JIT member is this one; the route runs the
        # member (see _decode_route_runs_program).
        if program_route is not None and native == dataclass_replace(
            program_route, program=None, device_scales=False, return_lse=True
        ):
            return dataclass_replace(program_route, return_lse=True)
        return native

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


@dataclass(frozen=True)
class CakeFmhaAlignedHd64DecodeModule:
    """One loaded aligned hd64 q1 decode program: the route it serves, the
    ``CakeProgram`` record, its exact target and the binding's ``run`` entry.
    Distinct from :class:`CakeFmhaAlignedDecodeModule`: that class's launcher
    is the hd128 programs' (query rows of 128, ``uniform_seq_len``, a cluster
    along z); this one launches the hd64 grouped program."""

    route: CakeFmhaDecodeRoute
    program: Any
    target: str
    entry: Any


def _load_cake_fmha_aligned_hd64_decode_module(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaAlignedHd64DecodeModule:
    """Build and load the aligned hd64 program of ``route`` on the route's target.

    The loader ``flashinfer.jit.cake_fmha_aligned_hd64`` is generated by the Cake
    export beside ``csrc/cake_fmha_aligned_hd64/``; it is imported here, lazily,
    so route selection never needs it.  The record's launch facts and argument
    names are checked against the host launch once, at load.
    """

    from .jit import cake_fmha_aligned_hd64
    from .jit.cake_program import GRID_ARGS, CakeProgram

    program_name = _aligned_hd64_route_program(route)
    if program_name is None:
        raise RuntimeError(
            f"Cake FMHA aligned hd64 decode route names no program: {route.aligned_program!r}"
        )
    program = getattr(cake_fmha_aligned_hd64, route.aligned_program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_hd64 publishes no CakeProgram {route.aligned_program!r}"
        )
    if program.name != program_name:
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_hd64.{route.aligned_program} is "
            f"{program.name}; this module pins {program_name}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    if tuple(program.args) != (*_ALIGNED_HD64_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_ALIGNED_HD64_HOST_PARAMS, *GRID_ARGS]}"
        )
    if tuple(program.cluster) != (1, 1, 1):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; the route "
            "expects (1, 1, 1)"
        )
    return CakeFmhaAlignedHd64DecodeModule(
        route=route,
        program=program,
        target=target,
        entry=program.entry(target),
    )


def _run_cake_fmha_aligned_hd64_decode(
    module: CakeFmhaAlignedHd64DecodeModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Launch one aligned hd64 q1 decode on the current torch stream.

    The binding's host parameters, in the kernel's order: the query rows (``Q``,
    TMA source, ``[batch x num_q_heads, 64]``), the K and V pools (TMA sources;
    the contiguous ``[pages, num_kv_heads, 16, 64]`` HND pools read as
    ``[pages x num_kv_heads, 16, 64]`` rows), the output rows, the page table,
    the device lengths, ``max_pages_per_seq`` (the page table's row width), the
    log2-domain softmax scale, the head counts, ``heads_per_kv`` (the query heads
    per KV head: a plain int the binding turns into the kernel's fast divisor),
    the batch size and the correction-skip threshold (0: disabled); then the grid
    of :func:`cake_fmha_aligned_hd64_launch_grid`.
    """

    route = module.route
    if (
        tuple(query.shape) != (route.batch_size, route.num_q_heads, 64)
        or key_cache.shape[1] != route.num_kv_heads
        or tuple(block_tables.shape[:1]) != (route.batch_size,)
    ):
        raise RuntimeError(
            "Cake FMHA aligned hd64 decode launch does not match its selected route"
        )
    grid = cake_fmha_aligned_hd64_launch_grid(
        module.program, route.batch_size, route.num_kv_heads
    )
    values = {
        "Q": query.view(-1, 64),
        "K": key_cache,
        "V": value_cache,
        "O": out.view(-1, 64),
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "max_pages_per_seq": int(block_tables.shape[1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "heads_per_kv": route.num_q_heads // route.num_kv_heads,
        "batch_size": route.batch_size,
        "skip_corr_threshold": _ALIGNED_HD64_SKIP_CORR_THRESHOLD,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    arguments = [values[name] for name in module.program.args]
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.entry(*arguments)


@dataclass(frozen=True)
class CakeFmhaAlignedHd256DecodeModule:
    """One loaded aligned hd256 q1 decode program: the route it serves, the
    ``CakeProgram`` record, its exact target and the binding's ``run`` entry.
    Distinct from :class:`CakeFmhaAlignedDecodeModule` and
    :class:`CakeFmhaAlignedHd64DecodeModule`: their launchers are the hd128
    programs' (query rows of 128, ``uniform_seq_len``, a cluster along z) and
    the hd64 program's; this one launches the hd256 grouped programs."""

    route: CakeFmhaDecodeRoute
    program: Any
    target: str
    entry: Any


def _load_cake_fmha_aligned_hd256_decode_module(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaAlignedHd256DecodeModule:
    """Build and load the aligned hd256 program of ``route`` on the route's target.

    The loader ``flashinfer.jit.cake_fmha_aligned_hd256`` is generated by the Cake
    export beside ``csrc/cake_fmha_aligned_hd256/``; it is imported here, lazily,
    so route selection never needs it.  The record's launch facts and argument
    names are checked against the host launch once, at load.
    """

    from .jit import cake_fmha_aligned_hd256
    from .jit.cake_program import GRID_ARGS, CakeProgram

    pinned = _aligned_hd256_route_program(route)
    if pinned is None:
        raise RuntimeError(
            f"Cake FMHA aligned hd256 decode route names no program: {route.aligned_program!r}"
        )
    ctas, program_name = pinned
    program = getattr(cake_fmha_aligned_hd256, route.aligned_program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_hd256 publishes no CakeProgram {route.aligned_program!r}"
        )
    if program.name != program_name:
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_hd256.{route.aligned_program} is "
            f"{program.name}; this module pins {program_name}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    if tuple(program.args) != (*_ALIGNED_HD256_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_ALIGNED_HD256_HOST_PARAMS, *GRID_ARGS]}"
        )
    if tuple(program.cluster) != (ctas, 1, 1):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; the route "
            f"expects {(ctas, 1, 1)}"
        )
    return CakeFmhaAlignedHd256DecodeModule(
        route=route,
        program=program,
        target=target,
        entry=program.entry(target),
    )


def _run_cake_fmha_aligned_hd256_decode(
    module: CakeFmhaAlignedHd256DecodeModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Launch one aligned hd256 q1 decode on the current torch stream.

    The binding's host parameters, in the kernel's order: the query rows (``Q``,
    TMA source, ``[batch x num_q_heads, 256]``), the K and V pools (TMA sources;
    the contiguous ``[pages, num_kv_heads, 16, 256]`` HND pools read as
    ``[pages x num_kv_heads, 16, 256]`` rows), the output rows, the page table,
    the device lengths, ``max_pages_per_seq`` (the page table's row width), the
    log2-domain softmax scale, the head counts, ``heads_per_kv`` (the query heads
    per KV head: a plain int the binding turns into the kernel's fast divisor),
    the batch size and the correction-skip threshold (0: disabled); then the grid
    of :func:`cake_fmha_aligned_hd256_launch_grid`.
    """

    route = module.route
    if (
        tuple(query.shape) != (route.batch_size, route.num_q_heads, 256)
        or key_cache.shape[1] != route.num_kv_heads
        or tuple(block_tables.shape[:1]) != (route.batch_size,)
    ):
        raise RuntimeError(
            "Cake FMHA aligned hd256 decode launch does not match its selected route"
        )
    grid = cake_fmha_aligned_hd256_launch_grid(
        module.program, route.batch_size, route.num_kv_heads
    )
    values = {
        "Q": query.view(-1, 256),
        "K": key_cache,
        "V": value_cache,
        "O": out.view(-1, 256),
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "max_pages_per_seq": int(block_tables.shape[1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "heads_per_kv": route.num_q_heads // route.num_kv_heads,
        "batch_size": route.batch_size,
        "skip_corr_threshold": _ALIGNED_HD256_SKIP_CORR_THRESHOLD,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    arguments = [values[name] for name in module.program.args]
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.entry(*arguments)


@dataclass(frozen=True)
class CakeFmhaAlignedMtpQ32DecodeModule:
    """One loaded aligned speculative-decode program: the route it serves, the
    ``CakeProgram`` record, its exact target and the binding's ``run`` entry.
    Distinct from the aligned single-token module classes (hd128
    :class:`CakeFmhaAlignedDecodeModule`, hd64
    :class:`CakeFmhaAlignedHd64DecodeModule`, hd256
    :class:`CakeFmhaAlignedHd256DecodeModule`): this one launches the program
    whose query tile holds every query token of a request."""

    route: CakeFmhaDecodeRoute
    program: Any
    target: str
    entry: Any


def _load_cake_fmha_aligned_mtp_q32_decode_module(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaAlignedMtpQ32DecodeModule:
    """Build and load the aligned speculative-decode program of ``route`` on the
    route's target.

    The loader ``flashinfer.jit.cake_fmha_aligned_mtp_q32`` is generated by the
    Cake export beside ``csrc/cake_fmha_aligned_mtp_q32/``; it is imported here,
    lazily, so route selection never needs it.  The record's launch facts and
    argument names are checked against the host launch once, at load.
    """

    from .jit import cake_fmha_aligned_mtp_q32
    from .jit.cake_program import GRID_ARGS, CakeProgram

    pinned = _aligned_mtp_q32_route_program(route)
    if pinned is None:
        raise RuntimeError(
            f"Cake FMHA aligned MTP decode route names no program: {route.aligned_program!r}"
        )
    ctas, program_name = pinned
    program = getattr(cake_fmha_aligned_mtp_q32, route.aligned_program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_mtp_q32 publishes no CakeProgram {route.aligned_program!r}"
        )
    if program.name != program_name:
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned_mtp_q32.{route.aligned_program} is "
            f"{program.name}; this module pins {program_name}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    if tuple(program.args) != (*_ALIGNED_MTP_Q32_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_ALIGNED_MTP_Q32_HOST_PARAMS, *GRID_ARGS]}"
        )
    if tuple(program.cluster) != (ctas, 1, 1):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; the route "
            f"expects {(ctas, 1, 1)}"
        )
    return CakeFmhaAlignedMtpQ32DecodeModule(
        route=route,
        program=program,
        target=target,
        entry=program.entry(target),
    )


def _run_cake_fmha_aligned_mtp_q32_decode(
    module: CakeFmhaAlignedMtpQ32DecodeModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Launch one aligned speculative decode on the current torch stream.

    The binding's host parameters, in the kernel's order: the query token rows
    (``Q``, TMA source, ``[batch x q_len, num_q_heads, 128]``), the K and V pools
    (TMA sources; the contiguous ``[pages, num_kv_heads, 16, 128]`` HND pools
    viewed as ``[pages x num_kv_heads, 16, 128]``), the output rows (``[batch x
    q_len x num_q_heads, 128]``), the page table, the device lengths,
    ``max_pages_per_seq`` (the page table's row width), the log2-domain softmax
    scale, the head counts, ``heads_per_kv`` (the query heads per KV head: a
    plain int the binding turns into the kernel's fast divisor), the batch size,
    ``seq_len_q`` (the uniform query tokens per request) and the correction-skip
    threshold (0: disabled); then the grid of
    :func:`cake_fmha_aligned_mtp_q32_launch_grid`.
    """

    route = module.route
    if (
        tuple(query.shape) != (route.batch_size * route.q_len, route.num_q_heads, 128)
        or key_cache.shape[1] != route.num_kv_heads
        or tuple(block_tables.shape[:1]) != (route.batch_size,)
    ):
        raise RuntimeError(
            "Cake FMHA aligned MTP decode launch does not match its selected route"
        )
    grid = cake_fmha_aligned_mtp_q32_launch_grid(
        module.program, route.batch_size, route.num_kv_heads
    )
    values = {
        "Q": query,
        "K": key_cache.view(-1, 16, 128),
        "V": value_cache.view(-1, 16, 128),
        "O": out.view(-1, 128),
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "max_pages_per_seq": int(block_tables.shape[1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "heads_per_kv": route.num_q_heads // route.num_kv_heads,
        "batch_size": route.batch_size,
        "seq_len_q": route.q_len,
        "skip_corr_threshold": _ALIGNED_MTP_Q32_SKIP_CORR_THRESHOLD,
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
    :func:`_run_cake_fmha_aligned_decode`), the aligned hd64 q1 route to a
    :class:`CakeFmhaAlignedHd64DecodeModule` (launched by
    :func:`_run_cake_fmha_aligned_hd64_decode`), an aligned hd256 q1 route to
    a :class:`CakeFmhaAlignedHd256DecodeModule` (launched by
    :func:`_run_cake_fmha_aligned_hd256_decode`), an aligned speculative-decode
    route to a :class:`CakeFmhaAlignedMtpQ32DecodeModule` (launched by
    :func:`_run_cake_fmha_aligned_mtp_q32_decode`), a generated decode program
    route to a :class:`CakeFmhaDecodeProgramModule` (launched by
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
    if route.component == _ALIGNED_MTP_Q32_COMPONENT:
        return _load_cake_fmha_aligned_mtp_q32_decode_module(route), True
    if route.component == _ALIGNED_HD256_COMPONENT:
        return _load_cake_fmha_aligned_hd256_decode_module(route), True
    if route.component == _ALIGNED_HD64_COMPONENT:
        return _load_cake_fmha_aligned_hd64_decode_module(route), True
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


def _context_fp8_runtime_route_pack(
    max_q_len: int, max_kv_len: int, num_q_heads: int, pack_g: int
) -> int | None:
    """The pack of the runtime-size FP8 program a call of JIT pack ``pack_g``
    takes: the dividing pack (:func:`_context_fp8_runtime_pack`) unless it
    runs more tile work than the JIT pack, else ``pack_g`` itself where it is
    a pack of :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS`, else ``None`` (the JIT
    module)."""

    pack = _context_fp8_runtime_pack(pack_g)
    # A shipped pack below the JIT pack holds fewer query heads per tile: it
    # may run more tile work than the JIT pack.
    if pack == pack_g or (num_q_heads // pack) * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // pack)
    ) <= (num_q_heads // pack_g) * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // pack_g)
    ):
        return pack
    return pack_g if pack_g in _CONTEXT_FP8_RUNTIME_OWN_PACKS else None


# Per target, the keys from which the sm100a rule of many query rows at long
# query lengths (:func:`_context_fp8_runtime_serves`) no longer keeps the JIT
# module: over these caches the runtime-size program ran 1.4 .. 3.3 % faster
# back to back at the power limit and was not slower cold (1.0 .. 4.2 % faster
# in the median) (timed B200: packs 1, 2, 4 and 8, 128 .. 2048 query tokens,
# 2^20 and 2^21 query rows, 16 .. 256 requests, pages 16 and 64, 1024 ..
# 16384 keys, on two B200 boards).
_CONTEXT_FP8_RUNTIME_ROWS_KV: dict[str, int] = {"sm100a": 1024}
# Per target, ``(keys, streams)``: a call of a pack of
# :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS` over a cache of more keys with more
# (request, KV head) streams keeps the JIT module.  There the cache reads set
# the call's time: the runtime-size program saved 1 .. 3 % in the median, but
# in some memory layouts its slowest round fell inside the A/A band (timed
# B200, ten seeded layouts of four rounds: 16384 keys, 256 .. 1024 streams);
# at 8192 keys (512 and 1024 streams), and at 16384 keys with 64 streams, it
# ran faster than the A/A band in every layout.
_CONTEXT_FP8_RUNTIME_OWN_LONG_KV: dict[str, tuple[int, int]] = {"sm100a": (8192, 64)}
# Per target, ``(keys, streams)``: an FP8 call of a form of
# :data:`_CONTEXT_FP8_FORMS` over a cache of more keys with more (request, KV
# head) streams keeps the JIT module, at every pack.  There the cache reads set
# the call's time: the runtime-size program saved 0.5 .. 3 % in the median,
# but in some memory layouts its slowest round fell inside the A/A band (timed
# B200 and GB300, ten seeded layouts of four rounds: 16384 keys, 1024 streams,
# one (B200) and 128 query tokens); with 256 streams over 16384 keys, and at
# 8192 keys with 1024 streams, it ran faster than the A/A band in every layout.
_CONTEXT_FP8_RUNTIME_FORM_LONG_KV: dict[str, tuple[int, int]] = {
    "sm100a": (8192, 256),
    "sm103a": (8192, 256),
}


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

    pack = _context_fp8_runtime_route_pack(max_q_len, max_kv_len, num_q_heads, pack_g)
    if pack is None:
        return False
    return _context_fp8_runtime_pack_serves(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, num_kv_heads, pack
    )


def _context_fp8_runtime_pack_serves(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack: int,
) -> bool:
    """The rules of :func:`_context_fp8_runtime_serves` for a call whose
    runtime-size program runs at ``pack``."""

    long_kv = _CONTEXT_FP8_RUNTIME_OWN_LONG_KV.get(target)
    if (
        long_kv is not None
        and pack in _CONTEXT_FP8_RUNTIME_OWN_PACKS
        and max_kv_len > long_kv[0]
        and batch_size * num_kv_heads > long_kv[1]
    ):
        return False
    # sm100a: calls this long run back to back at the board power limit, where
    # the runtime program's SM clock drops below the JIT module's: many
    # (request, KV head) streams, or many query rows at long query lengths
    # over short caches (from :data:`_CONTEXT_FP8_RUNTIME_ROWS_KV` keys the
    # runtime program ran faster).  The programs of
    # :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS` keep the rule over every cache:
    # over 1024 keys they ran up to 1.2 % slower than the JIT module at 1024
    # and 2048 query tokens (timed B200).
    return not (
        target == "sm100a"
        and (
            batch_size * num_kv_heads >= 2048
            or (
                max_q_len >= 128
                and batch_size * num_q_heads * max_q_len >= 1 << 20
                and (
                    max_kv_len < _CONTEXT_FP8_RUNTIME_ROWS_KV.get(target, 1 << 62)
                    or pack in _CONTEXT_FP8_RUNTIME_OWN_PACKS
                )
            )
        )
    )


@functools.lru_cache(maxsize=1024)
def _context_fp8_runtime_form_serves(
    target: str,
    form: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
) -> bool:
    """Whether the runtime-size program of ``form`` (``_CONTEXT_FP8_FORMS``)
    serves an FP8 call of that form that no exact route serves (``False``
    keeps the JIT module, whose pack is ``pack_g``): at the pack of
    :func:`_context_fp8_runtime_form_pack`, under the rules of the causal call
    without LSE or sink (:func:`_context_fp8_runtime_pack_serves`) and of
    :data:`_CONTEXT_FP8_RUNTIME_FORM_LONG_KV`.  A pure
    function of its arguments, cached."""

    pack = _context_fp8_runtime_form_pack(
        target, form, max_q_len, max_kv_len, num_q_heads, pack_g
    )
    if pack is None:
        return False
    long_kv = _CONTEXT_FP8_RUNTIME_FORM_LONG_KV.get(target)
    if (
        long_kv is not None
        and max_kv_len > long_kv[0]
        and batch_size * num_kv_heads > long_kv[1]
    ):
        return False
    return _context_fp8_runtime_pack_serves(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, num_kv_heads, pack
    )


def _context_fp8_runtime_form_pack(
    target: str,
    form: str,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack_g: int,
) -> int | None:
    """The pack of the runtime-size program of ``form`` for a call of JIT pack
    ``pack_g``, or ``None`` (the JIT module).  A non-causal form takes the
    pack the causal call without LSE or sink would take
    (:func:`_context_fp8_runtime_route_pack`): each of its rows reads every
    key block of its request.  A causal form takes the JIT pack itself: the
    tokens of a tile decide which key blocks its P.V reads, and a row whose
    keys end before a block still multiplies that block's V by 0; where the
    slots past a request's last key hold NaN, only the JIT's tiles give the
    JIT's bytes."""

    if form in ("NC", "NC_LSE", "NC_SINK"):
        return _context_fp8_runtime_route_pack(
            max_q_len, max_kv_len, num_q_heads, pack_g
        )
    forms = _CONTEXT_FP8_RUNTIME_FORM_ROUTES.get(target) or {}
    return pack_g if (form, pack_g, None) in forms else None


def _context_fp8_form_one_stage_pack(
    target: str,
    form: str,
    batch_size: int,
    max_q_len: int,
    num_q_heads: int,
    pack: int,
    sm_count: int | None,
) -> int | None:
    """The pack of the one-stage program (``_CONTEXT_FP8_FORM_ONE_STAGE_ROUTES``)
    an FP8 call of ``form`` whose form program runs at ``pack`` takes instead,
    or ``None`` (its form program): ``pack`` itself where its tiles hold at
    most one stage of query tokens or where its one-stage tiles (``batch_size
    x num_q_heads / pack x ceil(max_q_len / (128 / pack))``) fit one wave of
    ``sm_count``.  A pack without a one-stage build keeps its form program:
    another pack's tiles would put other rows in each warp of the correction,
    whose rescale vote (any row's scale below 1) the JIT's tiles decide."""

    one_stage = _CONTEXT_FP8_FORM_ONE_STAGE_ROUTES.get(target) or {}
    if (form, pack, None) in one_stage:
        tokens = 128 // pack
        if max_q_len <= tokens:
            return pack
        tiles = batch_size * (num_q_heads // pack) * -(-max_q_len // tokens)
        return pack if sm_count and tiles <= sm_count else None
    return None


# Per target, the query-head tiles (``batch_size x num_q_heads`` at pack 1) from
# which a pack-1 call whose query tokens fit one stage runs the one-stage
# program of pack 1 where the runtime-size program would serve it: the
# one-stage choice (:func:`_context_fp8_one_stage_choice`) admits at most
# :data:`_CONTEXT_FP8_SPLIT_MAX_WAVES` waves of such tiles, and past them the
# one-stage program ran 1.2 .. 1.4x faster than the runtime-size program
# (timed B200: heads 16/16, 32/32 and 64/64, 4 .. 128 query tokens, 640 ..
# 1536 tiles over 2048 .. 16384 keys, pages 16 and 64, cold L2 and back to
# back).
_CONTEXT_FP8_RUNTIME_ONE_STAGE_TILES: dict[str, int] = {"sm100a": 640}


def _context_fp8_runtime_one_stage(
    target: str,
    batch_size: int,
    max_q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
) -> bool:
    """Whether a pack-1 call the runtime-size program serves runs the one-stage
    program of pack 1 instead (:data:`_CONTEXT_FP8_RUNTIME_ONE_STAGE_TILES`)."""

    tiles = _CONTEXT_FP8_RUNTIME_ONE_STAGE_TILES.get(target)
    return (
        tiles is not None
        and pack_g == 1
        and num_q_heads == num_kv_heads
        and max_q_len <= 128
        and batch_size * num_q_heads >= tiles
    )


# Targets whose power-limited launches (the calls :func:`_context_fp8_runtime_serves`
# keeps on the JIT module for the board power limit) run the one-stage program
# of the JIT pack where one query stage holds every query token: the JIT
# module's tiles then hold an empty second stage, whose products and softmax
# run at full cost; the one-stage program runs the same arithmetic per query
# row without them.
_CONTEXT_FP8_POWER_ONE_STAGE_TARGETS = frozenset({"sm100a"})


def _context_fp8_power_one_stage(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    pack_g: int,
) -> bool:
    """Whether a causal FP8 call without LSE or sink that no exact route
    serves, kept on the JIT module by the power limit, runs the one-stage
    program of the JIT pack ``pack_g`` (a shipped pack with one-stage programs
    whose 128-row stage holds every query token) instead.  Every query row meets the same KV blocks in the same order as
    in the JIT module's tile of two stages."""

    return (
        target in _CONTEXT_FP8_POWER_ONE_STAGE_TARGETS
        and _context_fp8_runtime_pack(pack_g) == pack_g
        and max_q_len <= 128 // pack_g
        and (
            not _context_fp8_runtime_serves(
                target,
                batch_size,
                max_q_len,
                max_kv_len,
                num_q_heads,
                num_kv_heads,
                pack_g,
            )
            or _context_fp8_runtime_one_stage(
                target, batch_size, max_q_len, num_q_heads, num_kv_heads, pack_g
            )
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
# The KV block of a whole-tile launch of more than 256 keys whose tiles hold
# all 256 query rows costs this share of a split unit's: on sm103a those
# tiles run the swapped P placement (the next QK product issued while P is
# written), the split units the default one.  Set from timed whole-tile and
# split launches of full tiles (GB300): the least-squares share (0.83) also
# moves long-KV launches whose split units fill a short last wave to whole
# tiles that time slower; at this share no timed launch does.  Tiles of
# fewer query rows keep the split units' cost.
_CONTEXT_FP8_SPLIT_WHOLE_BLOCK: dict[str, float] = {"sm103a": 0.880}
# A split must be predicted faster than the whole-tile schedule by both
# margins (2 % and 0.5 us: the smallest that timed plans of B200 and GB300
# support; at 1 % some predicted gains measure as losses); a launch of more
# tiles than this many waves keeps whole tiles (its last partial wave is a
# small share of it).
_CONTEXT_FP8_SPLIT_REL_MARGIN = 0.02
_CONTEXT_FP8_SPLIT_ABS_MARGIN = 0.5
_CONTEXT_FP8_SPLIT_MAX_WAVES = 4
# Targets whose launches of several m-blocks also take a smaller pack's own
# split plan where that pack holds every query token in one m-block and its
# tiles fit one wave (:func:`_context_fp8_split_choice`): on B200 (sm100a)
# the split units of fewer query rows store and merge less (timed: 128 query
# tokens over 1500 keys, heads 32/4, run pack 1's split faster than pack
# 8's).  A tile of fewer query rows still runs its empty stage at full cost:
# with more than one wave of tiles the smaller pack timed no faster on the
# whole, and some of those launches timed slower.
_CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS = frozenset({"sm100a"})
# The one-stage programs' cost model per target, in the split planner's units
# (the launch's fixed cost is the split model's): the fixed cost of a tile of
# one 128-row query stage and one KV block of it.  Fitted to timed one-stage
# and two-stage launches of 128 query tokens over 768 .. 6016 keys, heads
# 32/4, one wave (B200, GB300), scaled to the split model's two-stage costs
# of the same launches.  The tile's cost does not scale with its query rows.
_CONTEXT_FP8_ONE_STAGE_COSTS: dict[str, tuple[float, float]] = {
    "sm100a": (1.31, 1.213),
    "sm103a": (1.10, 0.852),
}
# The one-stage tiles of one launch at the packs it may take rank by their
# query rows too: a tile of ``r`` query rows costs ``U0 + (1 - U0) x max(r,
# 64) / 128`` of the one-stage unit (fewer rows per tile store and reduce less;
# below 64 rows, more tiles of one wave timed no faster).  This ranks the packs
# only: the gain over the launch's current schedule takes the unit above.
# ``U0`` per target, fitted to timed one-stage launches of 32 query tokens
# at packs 4, 2 and 1, heads 32/4 and 32/8, one or two requests over 128 ..
# 1024 keys (B200, GB300).
_CONTEXT_FP8_ONE_STAGE_ROWS: dict[str, float] = {
    "sm100a": 0.51,
    "sm103a": 0.71,
}
_CONTEXT_FP8_ONE_STAGE_MIN_ROWS = 64
# Per target, the page sizes whose one-stage programs' gain over the current
# launch also takes the unit scaled by the tile's query rows (as the ranking
# does, and as the split model scales the current launch's unit): the page-64
# one-stage programs with the short-KV chain, whose tiles of fewer query rows
# timed faster than the unscaled unit predicts (GB300).
_CONTEXT_FP8_ONE_STAGE_ROWS_GAIN: dict[str, frozenset[int]] = {
    "sm103a": frozenset({64}),
}
# Per target, the most keys of a launch of one KV block per tile that takes
# the one-stage pack ranked fastest wherever the one-stage costs predict it
# faster than the launch's schedule, without the split margins: with one
# 128-key block per tile the one-stage tile drops the empty second stage's
# products, softmax and epilogue, which the costs (fitted over 768 .. 6016
# keys) undercount (timed B200: 32 query tokens, heads 32/32, page 64, 1 .. 4
# requests ran 0.8 .. 1.1 us faster one-stage, predicted 0.3 us).
_CONTEXT_FP8_ONE_BLOCK_ONE_STAGE_KEYS: dict[str, int] = {"sm100a": 128}
# Targets whose runtime-size launches of whole tiles in one wave (at most as
# many tiles as the SMs) that the one-stage costs leave on their launch take
# the one-stage program of their own pack wherever its tiles (one or more
# m-blocks of one query stage per head group) also fit one wave: the launch
# takes as long as its longest tile, and a one-stage tile runs one query
# stage over the KV blocks its rows meet, half the rows of a two-stage tile.
_CONTEXT_FP8_ONE_WAVE_ONE_STAGE_TARGETS = frozenset({"sm100a"})


def _context_fp8_tile_blocks(
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack: int,
    stages: int = 2,
) -> list[int]:
    """KV blocks of every tile of a runtime-size launch, in the program's tile
    order (sections of 8 work units x every m-block, the largest m-block
    first; bottom-right causal), every request at the planned maxima; an
    m-block holds ``stages`` query stages (1 for a one-stage program)."""

    tok = 128 // pack
    m_blocks = max((max_q_len + stages * tok - 1) // (stages * tok), 1)
    total_bh = batch_size * (num_q_heads // pack)
    full_n = max((max_kv_len + 127) // 128, 1)
    shift = max_kv_len - max_q_len
    per_m = []
    for m_block in range(m_blocks):
        top = (m_block + 1) * stages * tok + shift
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


def _context_fp8_whole_costs(
    target: str,
    max_q_len: int,
    max_kv_len: int,
    pack: int,
    costs: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    """``costs`` for the whole-tile schedule of a launch of ``pack``: the KV
    block scaled by :data:`_CONTEXT_FP8_SPLIT_WHOLE_BLOCK` for launches of
    more than 256 keys whose tiles hold 256 query rows (several m-blocks, or
    ``max_q_len x pack`` of at least 256)."""

    share = _CONTEXT_FP8_SPLIT_WHOLE_BLOCK.get(target)
    full = max_q_len > 2 * (128 // pack) or max_q_len * pack >= 256
    if costs is None or share is None or max_kv_len <= 256 or not full:
        return costs
    return (costs[0], costs[1] * share, *costs[2:])


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
    whole = _context_fp8_split_makespan(
        blocks,
        sm_count,
        0,
        1,
        1,
        _context_fp8_whole_costs(target, max_q_len, max_kv_len, pack, costs),
    )
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

    The plan is ``pack``'s (:func:`_context_fp8_split_plan`), except where a
    smaller shipped pack holds the launch in one m-block (fewer query rows per
    tile: the cost model's units, partial stores and merges cost less,
    :func:`_context_fp8_split_costs`), for a launch of one m-block or, on the
    targets of :data:`_CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS`, of several where
    the smaller pack's tiles fit one wave:

    - a launch of one m-block of two query stages whose plan splits every
      tile runs the same split (``split_blocks`` and ``num_splits``) over the
      tiles of a smaller pack, where predicted faster by the margins: every
      query row meets the same KV units, merged in the same order, so the
      output is the same;
    - then the launch takes the best split of a smaller pack of one m-block
      (its own plan) where predicted faster than that by the margins: its rows
      meet other KV units, so the output differs by the rounding of the FP32
      merge only."""

    plan = _context_fp8_split_plan(
        target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count
    )
    choice = None if plan is None else (pack, plan)
    tokens = 128 // pack
    costs = _context_fp8_split_costs(target, max_q_len, pack)
    blocks = _context_fp8_tile_blocks(
        batch_size, max_q_len, max_kv_len, num_q_heads, pack
    )
    one_m_block = max_q_len <= 2 * tokens
    if (
        costs is None
        or not (one_m_block or target in _CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS)
        or len(blocks) > _CONTEXT_FP8_SPLIT_MAX_WAVES * sm_count
    ):
        return choice
    smaller_packs = [
        smaller
        for smaller in (*_CONTEXT_FP8_RUNTIME_PACKS, 1)
        if smaller < pack
        and pack % smaller == 0
        and max_q_len <= 2 * (128 // smaller)
        and (one_m_block or batch_size * (num_q_heads // smaller) <= sm_count)
    ]
    time = _context_fp8_split_makespan(
        blocks,
        sm_count,
        *(plan or (0, 1, 1)),
        costs
        if plan
        else _context_fp8_whole_costs(target, max_q_len, max_kv_len, pack, costs),
    )
    if (
        plan is not None
        and one_m_block
        and max_q_len > tokens
        and plan[0] == len(blocks)
    ):
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


@functools.lru_cache(maxsize=1024)
def _context_fp8_one_stage_choice(
    target: str,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    pack: int,
    sm_count: int,
    current: tuple[int, tuple[int, int, int] | None],
    page_size: int | None = None,
    runtime: bool = False,
) -> int | None:
    """The pack of the one-stage launch of a call whose runtime pack is
    ``pack`` and whose launch is ``current`` (its pack and split plan, ``None``
    for whole tiles), or ``None`` where it keeps that launch (a pure function
    of its arguments, cached).  For a page size the target names in
    :data:`_CONTEXT_FP8_ONE_STAGE_ROWS_GAIN`, the gain takes the scaled unit.
    A ``runtime`` launch (of a runtime-size program at ``pack``) of whole
    tiles in one wave on a target of
    :data:`_CONTEXT_FP8_ONE_WAVE_ONE_STAGE_TARGETS` that the costs below leave
    on its launch takes ``pack`` where its one-stage tiles fit one wave.

    The candidates are ``pack`` and the smaller shipped packs dividing it whose
    tiles hold every query token in one stage (``max_q_len <= 128 / pack``),
    in at most :data:`_CONTEXT_FP8_SPLIT_MAX_WAVES` waves of tiles (one wave
    for a smaller pack); the fastest under the one-stage cost model
    (:data:`_CONTEXT_FP8_ONE_STAGE_COSTS`, its unit scaled by the tile's query
    rows, :data:`_CONTEXT_FP8_ONE_STAGE_ROWS`) is taken where predicted faster
    than ``current`` under the split cost model by the split margins (with
    the unscaled unit), or at all for a launch of at most the keys of
    :data:`_CONTEXT_FP8_ONE_BLOCK_ONE_STAGE_KEYS` (one KV block per tile).  Every
    query row meets the same KV blocks in the same order as in a whole tile
    of two stages (the blocks a tile of two stages adds above a row's keys
    are fully masked for it)."""

    one_stage = _CONTEXT_FP8_ONE_STAGE_COSTS.get(target)
    split_costs = _CONTEXT_FP8_SPLIT_COSTS.get(target)
    current_pack, plan = current
    costs = _context_fp8_split_costs(target, max_q_len, current_pack)
    blocks = _context_fp8_tile_blocks(
        batch_size, max_q_len, max_kv_len, num_q_heads, current_pack
    )
    if one_stage is None or split_costs is None or costs is None or not blocks:
        return None
    split_tiles, split_blocks, num_splits = plan or (0, 1, 1)
    time = _context_fp8_split_makespan(
        blocks,
        sm_count,
        split_tiles,
        split_blocks,
        num_splits,
        costs
        if plan
        else _context_fp8_whole_costs(
            target, max_q_len, max_kv_len, current_pack, costs
        ),
    )
    unit, block = one_stage
    one_stage_costs = (split_costs[0], block, unit, 0.0, 0.0, 0.0, 0.0)
    rows_floor = _CONTEXT_FP8_ONE_STAGE_ROWS.get(target, 1.0)
    best: int | None = None
    best_time = time
    best_rank = time
    for candidate in (*_CONTEXT_FP8_RUNTIME_PACKS, 1):
        if candidate > pack or pack % candidate or max_q_len > 128 // candidate:
            continue
        tiles = _context_fp8_tile_blocks(
            batch_size, max_q_len, max_kv_len, num_q_heads, candidate, 1
        )
        waves = _CONTEXT_FP8_SPLIT_MAX_WAVES if candidate == pack else 1
        if len(tiles) > waves * sm_count:
            continue
        candidate_time = _context_fp8_split_makespan(
            tiles, sm_count, 0, 1, 1, one_stage_costs
        )
        rows = max(max_q_len * candidate, _CONTEXT_FP8_ONE_STAGE_MIN_ROWS)
        share = rows_floor + (1.0 - rows_floor) * min(rows, 128) / 128
        rank_costs = (split_costs[0], block, unit * share, 0.0, 0.0, 0.0, 0.0)
        candidate_rank = _context_fp8_split_makespan(
            tiles, sm_count, 0, 1, 1, rank_costs
        )
        if page_size in _CONTEXT_FP8_ONE_STAGE_ROWS_GAIN.get(target, ()):
            candidate_time = candidate_rank
        if candidate_rank < best_rank:
            best, best_time, best_rank = candidate, candidate_time, candidate_rank
    one_block = max_kv_len <= _CONTEXT_FP8_ONE_BLOCK_ONE_STAGE_KEYS.get(target, 0)
    if best is not None and (
        _context_fp8_split_gain(time, best_time) or (one_block and best_time < time)
    ):
        return best
    if (
        runtime
        and plan is None
        and current_pack == pack
        and target in _CONTEXT_FP8_ONE_WAVE_ONE_STAGE_TARGETS
        and len(blocks) <= sm_count
        and len(
            _context_fp8_tile_blocks(
                batch_size, max_q_len, max_kv_len, num_q_heads, pack, 1
            )
        )
        <= sm_count
    ):
        return pack
    return None


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
    # exact programs are causal and write no LSE and take no sink.
    program = None
    # the pack of the runtime-size program a call no exact route serves takes
    runtime_pack: int | None = None
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
        # The pack of an FP8 call an exact program serves: its one-stage
        # choice takes the packs this one divides (below).
        exact_fp8_pack = (
            pack_g if program is not None and component == "context_fp8" else None
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
            and not (
                _CONTEXT_FP8_ONE_STAGE_ROUTES.get(target)
                and _context_fp8_runtime_one_stage(
                    target,
                    batch_size,
                    max_q_len,
                    num_q_heads,
                    num_kv_heads,
                    pack_g,
                )
            )
        ):
            pack = _context_fp8_runtime_route_pack(
                max_q_len, max_kv_len, num_q_heads, pack_g
            )
            program = runtime.get((pack, page_size)) or runtime.get((pack, None))
            if program is not None:
                runtime_pack = pack
                pack_g = pack
                tok_per_stage = 128 // pack
                num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (
                    2 * tok_per_stage
                )
                total_bh = batch_size * (num_q_heads // pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
        # An NVFP4 call no exact route serves: the runtime-size program of its
        # JIT pack (the same tile work as the JIT module), where the CUDA
        # toolkit builds the NVFP4 programs; else the JIT module, which needs
        # no PTX ISA 9.4 instruction.
        runtime_nvfp4 = _CONTEXT_NVFP4_RUNTIME_ROUTES.get(target)
        if (
            program is None
            and component == "context_nvfp4"
            and runtime_nvfp4
            and context_nvfp4_programs_unavailable() is None
        ):
            program = runtime_nvfp4.get((pack_g, page_size)) or runtime_nvfp4.get(
                (pack_g, None)
            )
        # An NVFP4 call of at most the keys of its program's short-KV entry,
        # where the target names one: the entry's program (the same pack,
        # m-blocks and arguments).
        short_nvfp4 = _CONTEXT_NVFP4_SHORT_KV_ROUTES.get(target, {}).get(program)
        if (
            component == "context_nvfp4"
            and short_nvfp4
            and max_kv_len <= short_nvfp4[1]
        ):
            program = short_nvfp4[0]
        # An FP8 call of at most the keys of its route's short-KV entry, where
        # the target names one: the entry's program (the same schedule, pack,
        # m-blocks and arguments).
        short = _CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {}).get(program)
        shortened = bool(
            component == "context_fp8" and short and max_kv_len <= short[1]
        )
        if shortened:
            program = short[0]
        # An FP8 call an exact or runtime-size program serves, whose tiles
        # leave SMs idle: the split-KV program of the pack the split choice
        # names (:func:`_context_fp8_split_choice`) and the page size, where the
        # split plan predicts a gain and the workspace holds its partials; the
        # runtime pack's own plan, where it has one, when the workspace cannot
        # hold the chosen pack's partials; else the call keeps its program (a
        # call of a pack of :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS` always).
        split = _CONTEXT_FP8_SPLIT_ROUTES.get(target)
        sm_count = _context_split_sm_count(device)
        split_plan: tuple[int, int, int] | None = None
        if (
            program is not None
            and component == "context_fp8"
            and split
            and sm_count
            and runtime_pack not in _CONTEXT_FP8_RUNTIME_OWN_PACKS
        ):
            pack = _context_fp8_runtime_pack(pack_g)
            choice = _context_fp8_split_choice(
                target, batch_size, max_q_len, max_kv_len, num_q_heads, pack, sm_count
            )
            bounds = _CONTEXT_FP8_ONE_WAVE_SPLIT_BOUNDS.get(target)
            if (
                choice is not None
                and bounds is not None
                and choice[1][2] == 2
                and program in _CONTEXT_FP8_ONE_WAVE_ROUTES.get(target, {})
                and bounds[0] <= total_bh * num_m_blocks <= sm_count
                and max_kv_len <= bounds[1]
                and max_q_len > bounds[2]
            ):
                choice = None
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
                split_plan = plan
                pack_g = pack
                tok_per_stage = 128 // pack
                num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (
                    2 * tok_per_stage
                )
                total_bh = batch_size * (num_q_heads // pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
                break
        # An FP8 call the runtime-size route serves, or an exact program (its
        # short-KV entry included), whose tiles hold at most one query stage at
        # a shipped pack: the one-stage program of the pack the one-stage
        # choice names (:func:`_context_fp8_one_stage_choice`) and the page
        # size, where it predicts a gain over the launch chosen above.  A
        # short-KV runtime-size call takes the smallest such pack whose tiles
        # number at most :data:`_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES`, where
        # one exists (pack 2 first for a page-64 call of at most
        # :data:`_CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TOKENS` query tokens and at
        # least :data:`_CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TILES` pack-2 tiles).
        # Of at most the keys of the program's short-KV entry, it takes the
        # entry's program; where the program has none, it runs the program
        # itself if :data:`_CONTEXT_FP8_SHORT_KV_ONE_STAGE_PROGRAMS` names it
        # and the pack is that smallest pack, and keeps its launch otherwise.
        # Else the call keeps its launch, as a call of a pack of
        # :data:`_CONTEXT_FP8_RUNTIME_OWN_PACKS` does.  An exact program's tiles
        # hold two query stages of its pack: at a smaller pack, one stage holds
        # every query token of a tile.
        one_stage = _CONTEXT_FP8_ONE_STAGE_ROUTES.get(target)
        stage_pack = runtime_pack
        if exact_fp8_pack is not None:
            stage_pack = _context_fp8_runtime_pack(exact_fp8_pack)
        short_runtime = shortened and runtime_pack is not None
        if (
            stage_pack is not None
            and stage_pack not in _CONTEXT_FP8_RUNTIME_OWN_PACKS
            and one_stage
            and sm_count
        ):
            one_stage_pack = _context_fp8_one_stage_choice(
                target,
                batch_size,
                max_q_len,
                max_kv_len,
                num_q_heads,
                stage_pack,
                sm_count,
                (pack_g, split_plan),
                page_size,
                runtime_pack is not None,
            )
            short_tiles = _CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES.get(target)
            short_pack = None
            if short_runtime and runtime_pack is not None and short_tiles is not None:
                order = (1, *reversed(_CONTEXT_FP8_RUNTIME_PACKS))
                if (
                    page_size == 64
                    and max_q_len
                    <= _CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TOKENS.get(target, 0)
                    and batch_size * (num_q_heads // 2)
                    >= _CONTEXT_FP8_SHORT_KV_PAGE64_PACK2_TILES.get(target, 0)
                ):
                    order = (2, 1, *reversed(_CONTEXT_FP8_RUNTIME_PACKS[:-1]))
                short_pack = next(
                    (
                        candidate
                        for candidate in order
                        if candidate <= runtime_pack
                        and runtime_pack % candidate == 0
                        and max_q_len <= 128 // candidate
                        and batch_size * (num_q_heads // candidate) <= short_tiles
                    ),
                    None,
                )
                if short_pack is not None:
                    one_stage_pack = short_pack
            one_stage_program = (
                None
                if one_stage_pack is None
                else one_stage.get((one_stage_pack, page_size))
                or one_stage.get((one_stage_pack, None))
            )
            if short_runtime and one_stage_program is not None:
                short = _CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {}).get(
                    one_stage_program
                )
                if short and max_kv_len <= short[1]:
                    one_stage_program = short[0]
                elif not (
                    short_pack is not None
                    and one_stage_program
                    in _CONTEXT_FP8_SHORT_KV_ONE_STAGE_PROGRAMS.get(target, ())
                ):
                    one_stage_program = None
            if one_stage_pack is not None and one_stage_program is not None:
                program = one_stage_program
                pack_g = one_stage_pack
                tok_per_stage = 128 // one_stage_pack
                num_m_blocks = (max_q_len + tok_per_stage - 1) // tok_per_stage
                total_bh = batch_size * (num_q_heads // one_stage_pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
        # An FP8 call the power limit keeps on the JIT module whose query
        # tokens fit one stage of the JIT pack: the one-stage program of that
        # pack and the page size (:func:`_context_fp8_power_one_stage`).
        if (
            program is None
            and component == "context_fp8"
            and one_stage
            and _context_fp8_power_one_stage(
                target,
                batch_size,
                max_q_len,
                max_kv_len,
                num_q_heads,
                num_kv_heads,
                pack_g,
            )
        ):
            program = one_stage.get((pack_g, page_size)) or one_stage.get(
                (pack_g, None)
            )
            if program is not None:
                tok_per_stage = 128 // pack_g
                num_m_blocks = (max_q_len + tok_per_stage - 1) // tok_per_stage
                total_bh = batch_size * (num_q_heads // pack_g)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
        # A split-KV launch whose plan has two units per split tile at most, of
        # a program the target names a two-split entry for: the entry's program
        # (:data:`_CONTEXT_FP8_TWO_SPLIT_ROUTES`).
        two_split = _CONTEXT_FP8_TWO_SPLIT_ROUTES.get(target, {}).get(program)
        if (
            component == "context_fp8"
            and two_split is not None
            and split_plan is not None
            and split_plan[2] <= 2
        ):
            program = two_split
        # A launch of a one-stage program the target names a multi-wave entry
        # for, of more tiles than the entry's over at most its keys: the
        # entry's program (:data:`_CONTEXT_FP8_MULTI_WAVE_ROUTES`).
        multi_wave = _CONTEXT_FP8_MULTI_WAVE_ROUTES.get(target, {}).get(program)
        if (
            component == "context_fp8"
            and multi_wave is not None
            and batch_size * (num_q_heads // pack_g) * num_m_blocks > multi_wave[1]
            and max_kv_len <= multi_wave[2]
        ):
            program = multi_wave[0]
        # A launch of a runtime-size program the target names a one-wave entry
        # for, of at most one tile per SM: the entry's program
        # (:data:`_CONTEXT_FP8_ONE_WAVE_ROUTES`).
        one_wave = _CONTEXT_FP8_ONE_WAVE_ROUTES.get(target, {}).get(program)
        if (
            component == "context_fp8"
            and one_wave is not None
            and sm_count
            and batch_size * (num_q_heads // pack_g) * num_m_blocks
            <= min(
                sm_count,
                _CONTEXT_FP8_ONE_WAVE_TILES.get(target, {}).get(program, sm_count),
            )
        ):
            program = one_wave
    elif component == "context_fp8":
        # An FP8 call of another form (non-causal, LSE or sink) that no exact
        # route serves: the runtime-size program of its form at its pack
        # (:func:`_context_fp8_runtime_form_pack`), where it serves that call
        # (:func:`_context_fp8_runtime_form_serves`); else the JIT module.
        form = _CONTEXT_FP8_FORMS.get((is_causal, lse is not None, sinks is not None))
        forms = _CONTEXT_FP8_RUNTIME_FORM_ROUTES.get(target)
        if (
            form is not None
            and forms
            and _context_fp8_runtime_form_serves(
                target,
                form,
                batch_size,
                max_q_len,
                max_kv_len,
                num_q_heads,
                num_kv_heads,
                pack_g,
            )
        ):
            pack = _context_fp8_runtime_form_pack(
                target, form, max_q_len, max_kv_len, num_q_heads, pack_g
            )
            program = forms.get((form, pack, page_size)) or forms.get(
                (form, pack, None)
            )
            # A call of at most the keys of the program's short-KV entry: the
            # entry's program (the same schedule, pack, m-blocks and
            # arguments), but an sm100a call whose tiles hold at most one
            # stage of query tokens, which takes its one-stage program.
            short = _CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {}).get(program)
            form_one_stage = _CONTEXT_FP8_FORM_ONE_STAGE_ROUTES.get(target, {})
            stages = 2
            if (
                short
                and max_kv_len <= short[1]
                and not (
                    target in _CONTEXT_FP8_FORM_ONE_STAGE_SHORT_KV
                    and max_q_len <= 128 // pack
                    and (form, pack, None) in form_one_stage
                )
            ):
                program = short[0]
            elif program is not None:
                # A call whose tiles hold at most one stage of query tokens, or
                # whose one-stage tiles fit one wave: the one-stage program of
                # its form at the pack of the one-stage choice.
                one_pack = _context_fp8_form_one_stage_pack(
                    target,
                    form,
                    batch_size,
                    max_q_len,
                    num_q_heads,
                    pack,
                    _context_split_sm_count(device),
                )
                if one_pack is not None:
                    pack = one_pack
                    program, stages = form_one_stage[(form, pack, None)], 1
            if program is not None:
                pack_g = pack
                tok_per_stage = 128 // pack
                num_m_blocks = (max_q_len + stages * tok_per_stage - 1) // (
                    stages * tok_per_stage
                )
                total_bh = batch_size * (num_q_heads // pack)
                l2_swizzle = 8 if total_bh % 8 == 0 else 1
    # A call no exact route serves, of a flag form its component's runtime-size
    # programs serve (:data:`_CONTEXT_RUNTIME_FLAG_ROUTES`): the program of its
    # JIT pack and page size in that form's table (the same tile work as the
    # JIT module of the form): the causal form without LSE or sink except the
    # power-of-two query lengths of :data:`_CONTEXT_BF16_RUNTIME_POW2_Q`, the
    # other forms within the bounds of :data:`_CONTEXT_BF16_RUNTIME_FLAG_BOUNDS`
    # or a box of :data:`_CONTEXT_BF16_RUNTIME_FLAG_BOXES`.
    flag_form = (is_causal, lse is not None, sinks is not None)
    runtime_form = (
        _CONTEXT_RUNTIME_FLAG_ROUTES.get(component, {}).get(flag_form, {}).get(target)
    )
    plain = flag_form == (True, False, False)
    if runtime_form and plain and component == "context_bf16":
        # the causal form's own table as the module holds it at this call
        runtime_form = _CONTEXT_BF16_RUNTIME_ROUTES.get(target)
    pow2_q = _CONTEXT_BF16_RUNTIME_POW2_Q.get(target) if plain else None
    jit_pow2 = (
        pow2_q is not None
        and pack_g > 1
        and max_q_len >= pow2_q[0]
        and max_q_len & (max_q_len - 1) == 0
        and max_kv_len >= pow2_q[1]
    )
    bound_keys, bound_area, bound_work = (
        _CONTEXT_BF16_RUNTIME_FLAG_BOUNDS.get(target, {})
        .get(is_causal, {})
        .get(pack_g, (None, None, None))
        if component == "context_bf16" and not plain
        else (None, None, None)
    )
    work = batch_size * num_q_heads * max_q_len * max_kv_len
    boxes = (
        _CONTEXT_BF16_RUNTIME_FLAG_BOXES.get(target, {})
        .get(is_causal, {})
        .get(pack_g, ())
        if component == "context_bf16" and not plain
        else ()
    )
    if (
        program is None
        and runtime_form
        and not jit_pow2
        and (
            (
                (bound_keys is None or max_kv_len <= bound_keys)
                and (bound_area is None or max_q_len * max_kv_len <= bound_area)
                and (bound_work is None or work <= bound_work)
            )
            or any(
                (box_batch is None or batch_size <= box_batch)
                and (box_min_q is None or max_q_len >= box_min_q)
                and (box_q is None or max_q_len <= box_q)
                and (box_keys is None or max_kv_len <= box_keys)
                and (box_area is None or max_q_len * max_kv_len <= box_area)
                and (box_work is None or work <= box_work)
                for box_batch, box_min_q, box_q, box_keys, box_area, box_work in boxes
            )
        )
    ):
        program = runtime_form.get((pack_g, page_size)) or runtime_form.get(
            (pack_g, None)
        )
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
    its exact target and the binding's ``run`` entry; for a runtime-size BF16
    program of :data:`_CONTEXT_BF16_SEQ_TAIL_PROGRAMS`, also the entry of its
    program without tail-order code, which the launches of the plain tail
    order of at most ``seq_tail_max_kv_len`` keys (``None``: any) run."""

    def __init__(
        self,
        route: CakeFmhaContextRoute,
        program: Any,
        target: str,
        seq_tail: Any = None,
        seq_tail_max_kv_len: int | None = None,
    ):
        self.route = route
        self.program = program
        self.target = target
        self.entry = program.entry(target)
        self.seq_tail_entry = None if seq_tail is None else seq_tail.entry(target)
        self.seq_tail_max_kv_len = seq_tail_max_kv_len
        self.units_per_batch = route.num_q_heads // route.pack_g
        self.quantized = route.component != "context_bf16"
        _, self.num_m_blocks, self.one_tile = _context_program_facts(program.name)
        # A runtime-size program serves every head count, page size and query
        # length of its pack (the route's): the launch reads them off its
        # tensors and lengths, and the scale form off its scale arguments.
        self.runtime = self.num_m_blocks is None
        self.pack_g = route.pack_g
        # a one-stage program's m-block holds one query stage, the others two
        stages = 1 if route.program in _CONTEXT_FP8_ONE_STAGE_PROGRAMS else 2
        self.m_block_tokens = stages * (128 // route.pack_g)
        # A split-KV program also takes the split plan of each launch
        # (:func:`_context_fp8_launch_split_plan`), its partials in the caller's
        # workspace and its tile counters after the counter buffer's first word.
        self.split = route.program in _CONTEXT_FP8_SPLIT_PROGRAMS
        # a two-split program merges two units per split tile at most
        self.two_split = route.program in {
            name
            for routes in _CONTEXT_FP8_TWO_SPLIT_ROUTES.values()
            for name in routes.values()
        }

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
        the pointers.  The Q-length pointer is unread (the programs derive the
        query lengths from ``cum_seq_lens_q``), and so are the LSE and sink
        pointers but in a program of an LSE or sink form, which takes the
        caller's (a contiguous ``[tokens, num_q_heads]`` LSE): an unread
        pointer points at a small per-device buffer.  A
        runtime-size program also takes the scale source, the m-block count of
        ``max_q_len``, the head counts, the GQA group and the log2 page size of
        this launch.
        """

        route = self.route
        form = (route.return_lse, route.enable_sink, route.is_causal)
        given = (lse is not None, attention_sinks is not None, bool(is_causal))
        if given[:2] != form[:2] or (form != (False, False, True) and given != form):
            # route selection keeps such calls on the JIT module or on the
            # program of their form: the unread pointers below never stand in
            # for a caller's LSE or sinks, and a program of a flag form takes
            # the calls of its causal mask only
            raise RuntimeError(
                "a Cake FMHA context program serves the causal mask, LSE and "
                "sinks of its route only"
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
                or (self.two_split and plan[2] > 2)
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
        # A program of an LSE or sink form (runtime-size) writes the LSE of
        # every query row and head at ``token x num_q_heads + head`` and reads
        # one sink per query head.
        lse_out = sinks_in = unread_f32
        if lse is not None:
            # the caller's strides must give every token and head that offset;
            # a stride along a dimension of size 1 (one token, or one head)
            # moves no element, as in the JIT module
            tokens, heads = query.shape[0], query.shape[1]
            if (tokens > 1 and lse_stride_tokens != heads) or (
                heads > 1 and lse_stride_heads != 1
            ):
                raise RuntimeError(
                    "a Cake FMHA context program writes a contiguous "
                    "[tokens, heads] LSE"
                )
            lse_out = lse
        if attention_sinks is not None:
            sinks_in = attention_sinks
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
        entry = self.entry
        # The arguments in the binding's order: ``_CONTEXT_HOST_PARAMS`` of the
        # component (LSE_ptr and sinks unread unless the program's flag form
        # takes them, seq_lens_q unread; a runtime-size program's sizes after
        # them), then the grid; the order is checked against the record at load.
        if not self.quantized:
            arguments: tuple[Any, ...] = (
                query,
                key_cache,
                value_cache,
                out,
                lse_out,
                sinks_in,
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
            )
            if self.runtime:
                interleave = is_causal and _context_bf16_tail_interleave(
                    int(batch_size),
                    int(max_q_len),
                    int(max_kv_len),
                    units_per_batch,
                    self.m_block_tokens,
                    num_ctas,
                )
                if (
                    not interleave
                    and self.seq_tail_entry is not None
                    and (
                        self.seq_tail_max_kv_len is None
                        or max_kv_len <= self.seq_tail_max_kv_len
                    )
                ):
                    # the plain tail order: the program without tail-order code
                    entry = self.seq_tail_entry
                arguments = (
                    *arguments,
                    num_m_blocks,
                    total_bh // 8
                    | (_CONTEXT_BF16_TAIL_INTERLEAVE if interleave else 0),
                    units_per_batch,
                    num_q_heads,
                    num_q_heads // key_cache.shape[1],
                    key_cache.shape[2].bit_length() - 1,
                )
            arguments = (*arguments, num_ctas, 1, 1)
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
            )
            if self.runtime:
                arguments = (
                    *arguments,
                    1 if device_scales else 0,
                    num_m_blocks,
                    total_bh // 8,
                    units_per_batch,
                    num_q_heads,
                    num_q_heads // key_cache.shape[1],
                )
            arguments = (*arguments, num_ctas, 1, 1)
        elif self.runtime:
            arguments = (
                query.view(torch.uint8),
                key_cache.view(torch.uint8),
                value_cache.view(torch.uint8),
                out.view(torch.uint8),
                lse_out,
                sinks_in,
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
            entry(*arguments)
        else:
            with _torch_stream_context(device):
                entry(*arguments)


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


def _context_program_record(
    loader: Any,
    component: str,
    identifier: str,
    pinned: tuple[str, int | None, bool],
    target: str,
) -> Any:
    """The generated loader's ``CakeProgram`` of ``identifier``, held to the
    host's pinned record ``pinned``: its name, a ``target`` build and its
    binding's arguments."""

    from .jit.cake_program import GRID_ARGS, CakeProgram

    loader_name = f"cake_fmha_{component}"
    program = getattr(loader, identifier, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.{loader_name} publishes no CakeProgram {identifier!r}"
        )
    if program.name != pinned[0]:
        raise RuntimeError(
            f"flashinfer.jit.{loader_name}.{identifier} is {program.name}; "
            f"this module pins {pinned[0]}"
        )
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    params = _context_program_params(component, pinned[1], identifier)
    if tuple(program.args) != (*params, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*params, *GRID_ARGS]}"
        )
    return program


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
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    program = _context_program_record(
        loader, route.component, route.program, pinned, target
    )
    seq_tail = None
    seq_tail_programs = _CONTEXT_BF16_SEQ_TAIL_PROGRAMS.get(route.target, {})
    if route.component == "context_bf16" and route.program in seq_tail_programs:
        identifier = seq_tail_programs[route.program]
        seq_tail = _context_program_record(
            loader,
            route.component,
            identifier,
            _CONTEXT_PROGRAMS[(route.component, identifier)],
            target,
        )
    module = CakeFmhaContextProgramModule(
        route,
        program,
        target,
        seq_tail,
        _CONTEXT_BF16_SEQ_TAIL_MAX_KV_LEN.get(route.target),
    )
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
