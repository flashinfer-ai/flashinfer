# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context compile specs and compiled-kernel lookup.

Builds go through ``compile_cache.build_and_load_attention_kernel``; see its
module docstring for how concurrent builds share the cache.
"""

import functools
import torch
from dataclasses import dataclass, field
from ...common.hardware import get_max_active_clusters
from ...common.scales import fake_scale
from ...compile_cache import build_and_load_attention_kernel
from .geometry import (
    ContextPlanGeometry,
    ContextScheduler,
    PagedContextPlanGeometry,
    default_exp2_fma_pairs,
    default_exp2_packed_f16x2,
    default_fp8_psmem_early_token,
    default_kv_ring_stages,
    default_umma_fp8_k64,
    default_uses_ldtm_stat,
    make_context_kernel,
    make_context_value_fake,
    resolve_context_scheduler,
    resolve_paged_context_scheduler,
    two_cta_umma_geometry_eligible,
)
from .validation import (
    CONTEXT_KV_TILE_N,
    CONTEXT_TILE_SIZE_Q,
    context_dtype_key,
    fixed_kv_rows,
    validate_query_work_tile_span,
    variable_window_q_tile_size,
)
from ..sage.layout import (
    SAGE_ATOM_TOKENS,
    SageKSection,
    SageQSection,
    SageScaleLayout,
    SageVSection,
)
from ..sage.plan import SagePreparedScales, build_sage_prepared_scales
from ..vc_attention import VCAttentionConfig, vc_scale_shapes
from typing import Optional


COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"


@dataclass(frozen=True)
class ContextCompileSpec:
    """Batch-independent identity for one contiguous context compile."""

    device_index: int
    max_q_len: int | None = field(compare=False, hash=False)
    max_seq_len_k: int | None = field(compare=False, hash=False)
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype_key: str
    pv_dtype_key: str
    out_dtype_key: str
    mask_type: str
    window_left: int
    packed_query: bool
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    scheduler: ContextScheduler
    head_dim_vo: int | None = None
    two_cta_umma: bool = False
    vc: Optional[VCAttentionConfig] = None
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    runtime_lengths: bool = False
    shape_signature: tuple[int, ...] = ()
    scale_types: tuple[bool, bool] = (False, False)
    # A Sage plan's prepared scale layout; the kernel reads the scales.
    sage_scale_layout: SageScaleLayout | None = None


@dataclass(frozen=True)
class PagedContextCompileSpec:
    """Batch-independent identity for one paged context compile."""

    device_index: int
    max_q_len: int | None = field(compare=False, hash=False)
    max_kv_len: int | None = field(compare=False, hash=False)
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype_key: str
    pv_dtype_key: str
    out_dtype_key: str
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    paged_v_tail_is_zero: bool
    packed_dense_k_mask: bool
    scheduler: ContextScheduler
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    runtime_lengths: bool = False
    shape_signature: tuple[int, ...] = ()
    scale_types: tuple[bool, bool] = (False, False)


def build_context_sage_state(geometry: ContextPlanGeometry) -> SagePreparedScales:
    """Allocate a plan's prepared scale buffer and compile its prepare launch."""

    return build_sage_prepared_scales(
        context_sage_layout(geometry),
        geometry.sage,
        batch_size=geometry.batch_size,
        seq_len_q=geometry.max_q_len,
        seq_len_kv=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        device=geometry.device,
        device_index=geometry.device_index,
    )


@functools.cache
def context_compile_spec(
    geometry: ContextPlanGeometry,
    *,
    lse_mode: str = "none",
    use_attention_sinks: bool = False,
    scale_types: tuple[bool, bool] = (False, False),
) -> ContextCompileSpec:
    """Keep resource/precision choices static and ordinary lengths runtime."""
    fixed_seqlen = geometry.sage is not None or geometry.vc is not None
    runtime_lengths = not fixed_seqlen
    shape_signature = (
        () if runtime_lengths else (geometry.max_q_len, geometry.max_seq_len_k)
    )
    return ContextCompileSpec(
        device_index=geometry.device_index,
        max_q_len=geometry.max_q_len if fixed_seqlen else None,
        max_seq_len_k=geometry.max_seq_len_k if fixed_seqlen else None,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        head_dim_vo=geometry.head_dim_vo,
        qk_dtype_key=context_dtype_key(geometry.qk_dtype),
        pv_dtype_key=context_dtype_key(geometry.pv_dtype),
        out_dtype_key=context_dtype_key(geometry.out_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        packed_query=geometry.packed_query,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=geometry.uniform_packed_lengths and fixed_seqlen,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
        packed_dense_k_mask=(
            geometry.packed_dense_k_mask
            if fixed_seqlen
            else geometry.mask_type != "causal"
        ),
        shape_signature=shape_signature,
        runtime_lengths=runtime_lengths,
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
        scale_types=scale_types,
        scheduler=resolve_context_scheduler(geometry),
        two_cta_umma=two_cta_umma_geometry_eligible(geometry),
        vc=geometry.vc,
        sage_scale_layout=(
            context_sage_layout(geometry) if geometry.sage is not None else None
        ),
    )


def context_sage_layout(geometry: ContextPlanGeometry) -> SageScaleLayout:
    """Return the prepared scale layout of a contiguous context Sage plan.

    The Q rows hold one Q tile row per token of one Q head (``_SageQSection``
    without interleaved heads): the query-paired D128 tile.
    """

    sage = geometry.sage
    return SageScaleLayout(
        k_tokens=SageKSection(
            length=geometry.max_seq_len_k,
            block_size=sage.k_block_size,
            source_block_size=sage.k_block_size,
            tile_atoms=CONTEXT_KV_TILE_N // SAGE_ATOM_TOKENS,
        ),
        q_rows=SageQSection(
            seq_len_q=geometry.max_q_len,
            q_block_size=sage.q_block_size,
            heads_q_per_kv=geometry.num_qo_heads // geometry.num_kv_heads,
            rows_per_tile=CONTEXT_TILE_SIZE_Q,
            heads_interleaved=False,
        ),
        v_rows=SageVSection(headdim=geometry.head_dim, has_mean=sage.v_mean),
    )


@functools.cache
def get_compiled_context(
    compile_spec: ContextCompileSpec,
):
    """Compile one batch-dynamic context specialization and cache by topology."""

    device_index = compile_spec.device_index
    # Only Sage and VC plans keep exact lengths in the key; they compile static shapes.
    fixed_seqlen = compile_spec.max_q_len is not None
    max_q_len = compile_spec.max_q_len or 0
    max_seq_len_k = compile_spec.max_seq_len_k or 0
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    head_dim_vo = compile_spec.head_dim_vo or head_dim
    qk_dtype_key = compile_spec.qk_dtype_key
    pv_dtype_key = compile_spec.pv_dtype_key
    out_dtype_key = compile_spec.out_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    packed_query = compile_spec.packed_query
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    causal_single_kv_tile = compile_spec.causal_single_kv_tile
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler
    two_cta_umma = compile_spec.two_cta_umma
    vc_attention = compile_spec.vc is not None
    sage_scale_layout = compile_spec.sage_scale_layout
    vc_qk8 = vc_attention and qk_dtype_key in ("float8_e4m3fn", "int8")

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "int8": cutlass.Int8,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
    }
    input_qk_dtype = dtype_map[qk_dtype_key]
    # MXFP8 Q/K names the E8M0 scale dtype; the Q and K tensors hold E4M3.
    qk_data_dtype = (
        cutlass.Float8E4M3FN if qk_dtype_key == "float8_e8m0fnu" else input_qk_dtype
    )
    input_pv_dtype = dtype_map[pv_dtype_key]
    out_dtype = dtype_map[out_dtype_key]
    has_variable_window = mask_type == "variable_window"
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)
    fmha = make_context_kernel(
        device_index=device_index,
        input_qk_dtype=input_qk_dtype,
        input_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        head_dim_vo=head_dim_vo,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler=scheduler,
        two_cta_umma=two_cta_umma,
        uses_ldtm_stat=default_uses_ldtm_stat(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        fp8_psmem_early_token=default_fp8_psmem_early_token(device_index),
        kv_ring_stages=default_kv_ring_stages(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        umma_fp8_k64=default_umma_fp8_k64(device_index, input_qk_dtype.width),
        vc_attention=("means" if compile_spec.vc.demean else "outliers")
        if vc_attention
        else None,
        sage_scale_layout=sage_scale_layout,
        # The QK8 K-scale table is sized from the K/V rows of the fixed plan.
        max_kv_len=fixed_kv_rows(max_seq_len_k, compile_spec.vc) if vc_qk8 else None,
    )
    if vc_attention and torch.cuda.get_device_capability(device_index) == (10, 7):
        raise ValueError(
            "vc_config is not supported on SM107, where the dense fp8 kernel runs "
            "the softmax exp2 natively"
        )
    fmha.cfg.lse_mode = compile_spec.lse_mode
    fmha.cfg.use_attention_sinks = compile_spec.use_attention_sinks
    fmha.cfg.has_varlen = packed_query
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if vc_qk8:
        # The QK8 scale tables are addressed with the fixed plan lengths, the K/V
        # rows including the repair tiles.
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = fixed_kv_rows(max_seq_len_k, compile_spec.vc)
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = max_seq_len_k
    fmha.cfg.has_q_offset = has_q_offset
    fmha.cfg.exp2_fma_pairs = default_exp2_fma_pairs(device_index, fmha.cfg)
    fmha.cfg.exp2_packed_f16x2 = default_exp2_packed_f16x2(device_index, fmha.cfg)
    if fmha.cfg.kv_tile_n != CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask
    if mask_type != "causal" and not packed_query:
        fmha.cfg.fixed_dense_k_tail = (
            max_seq_len_k % fmha.cfg.kv_tile_n if fixed_seqlen else fmha.cfg.kv_tile_n
        )

    # Compile both bounded and general masks once; choose from runtime K.
    fmha.runtime_dense_tail = (
        not fixed_seqlen and not packed_query and mask_type == "dense"
    )

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        attention_sinks: cute.Tensor | None,
        softmax_scale: cutlass.Float32 | cute.Tensor,
        output_scale: cutlass.Float32 | cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        q_scales: cute.Tensor | None,
        k_scales: cute.Tensor | None,
        variable_window_token_starts: cute.Tensor,
        variable_window_token_ends: cute.Tensor,
        variable_window_cta_starts: cute.Tensor,
        vc_mu: cute.Tensor | None,
        vc_q_scale: cute.Tensor | None,
        vc_k_scale: cute.Tensor | None,
        sage_scales: cute.Tensor | None,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_packed_query: cutlass.Constexpr[bool],
        max_q_len: cutlass.Int32,
        runtime_max_seq_len_k: cutlass.Int32,
        static_vc_repair_tiles: cutlass.Constexpr[int],
    ) -> None:
        """Adapt torch TVM-FFI tensors to the FmhaTs host entry point."""

        if cutlass.const_expr(static_packed_query):
            fmha(
                q,
                k,
                v,
                out,
                softmax_scale,
                output_scale,
                static_max_active_clusters,
                stream,
                cum_seqlen_q=qo_indptr,
                cum_seqlen_k=kv_indptr,
                max_seqlen_q=max_q_len,
                max_seqlen_k=runtime_max_seq_len_k,
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
                vc_mu=vc_mu,
                vc_q_scale=vc_q_scale,
                vc_k_scale=vc_k_scale,
                vc_repair_tiles=static_vc_repair_tiles,
                lse_output=lse,
                attention_sinks=attention_sinks,
            )
        else:
            fmha(
                q,
                k,
                v,
                out,
                softmax_scale,
                output_scale,
                static_max_active_clusters,
                stream,
                q_scales=q_scales,
                k_scales=k_scales,
                variable_window_token_starts=variable_window_token_starts,
                variable_window_token_ends=variable_window_token_ends,
                variable_window_cta_starts=variable_window_cta_starts,
                vc_mu=vc_mu,
                vc_q_scale=vc_q_scale,
                vc_k_scale=vc_k_scale,
                vc_repair_tiles=static_vc_repair_tiles,
                lse_output=lse,
                attention_sinks=attention_sinks,
                sage_scales=sage_scales,
            )

    # Describe row-major symbolic extents; assumed_align is in bytes.
    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q_shape: tuple[object, ...]
    kv_shape: tuple[object, ...]
    out_shape: tuple[object, ...]
    if packed_query:
        runtime_total_q = cute.sym_int()
        runtime_total_k = cute.sym_int()
        runtime_num_q_offsets = cute.sym_int()
        runtime_num_k_offsets = cute.sym_int()
        q_shape = (runtime_total_q, num_qo_heads, head_dim)
        kv_shape = (runtime_total_k, num_kv_heads, head_dim)
        out_shape = (runtime_total_q, num_qo_heads, head_dim_vo)
        qo_indptr_shape = (runtime_num_q_offsets,)
        kv_indptr_shape = (runtime_num_k_offsets,)
    else:
        batch_size = cute.sym_int()
        physical_sq = cute.sym_int() if compile_spec.runtime_lengths else max_q_len
        physical_sk = (
            cute.sym_int()
            if compile_spec.runtime_lengths
            else fixed_kv_rows(max_seq_len_k, compile_spec.vc)
        )
        q_shape = (batch_size, physical_sq, num_qo_heads, head_dim)
        kv_shape = (batch_size, physical_sk, num_kv_heads, head_dim)
        out_shape = (*q_shape[:-1], head_dim_vo)
        qo_indptr_shape = (1,)
        kv_indptr_shape = (1,)
    q_fake = fake_compact(qk_data_dtype, q_shape, 16)
    k_fake = fake_compact(qk_data_dtype, kv_shape, 16)
    v_fake = make_context_value_fake(input_pv_dtype, (*kv_shape[:-1], head_dim_vo))
    out_fake = fake_compact(out_dtype, out_shape, 16)
    lse_fake = (
        fake_compact(cutlass.Float32, out_fake.shape[:-1], 4)
        if compile_spec.lse_mode != "none"
        else None
    )
    scale_fake = fake_scale(compile_spec.scale_types[0])
    output_scale_fake = fake_scale(compile_spec.scale_types[1])
    if vc_attention:
        output_scale_fake = fake_compact(
            cutlass.Float32, (batch_size, num_qo_heads, head_dim_vo), 64
        )
    qo_indptr_fake = fake_compact(cutlass.Int32, qo_indptr_shape, 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, kv_indptr_shape, 4)
    if has_variable_window:
        if packed_query:
            raise RuntimeError("variable-window context requires fixed tensors")
        variable_window_shape = (batch_size * q_shape[1],)
        variable_window_tile_size_q = variable_window_q_tile_size(head_dim)
        variable_window_cta_shape = (
            (cute.sym_int(),)
            if not fixed_seqlen
            else (batch_size * cute.ceil_div(max_q_len, variable_window_tile_size_q),)
        )
    else:
        variable_window_shape = (1,)
        variable_window_cta_shape = (1,)
    variable_window_starts_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_ends_fake = fake_compact(cutlass.Int32, variable_window_shape, 4)
    variable_window_cta_starts_fake = fake_compact(
        cutlass.Int32, variable_window_cta_shape, 4
    )
    vc_mu_fake = None
    vc_q_scale_fake = None
    vc_k_scale_fake = None
    if vc_attention:
        if packed_query:
            raise RuntimeError("VC-Attention context requires fixed tensors")
        # The plan fixes every VC operand shape, so the kernel addresses the
        # tile means and the Q and K scale tables with compile-time extents.
        vc_shapes = vc_scale_shapes(
            compile_spec.vc,
            batch_size=batch_size,
            seq_len_kv=max_seq_len_k,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_vo or head_dim,
            seq_len_q=max_q_len,
            num_qo_heads=num_qo_heads,
            qk_quant=vc_qk8,
        )
        if compile_spec.vc.demean:
            vc_mu_fake = fake_compact(cutlass.BFloat16, vc_shapes["tile_means"], 16)
        if vc_qk8:
            # The kernel reads the Q and K scale tables as one row of blocks per head.
            vc_q_scale_fake = fake_compact(
                cutlass.Float32,
                (num_qo_heads, batch_size * vc_shapes["q_scale"][2]),
                16,
            )
            vc_k_scale_fake = fake_compact(
                cutlass.Float32,
                (num_kv_heads, batch_size * vc_shapes["k_scale"][2]),
                16,
            )
    sinks_fake = (
        fake_compact(cutlass.Float32, (num_qo_heads,), 4)
        if compile_spec.use_attention_sinks
        else None
    )
    # The prepared Sage scales span ``batch * Hkv`` sequences, so their length
    # stays a runtime extent like the batch size; a plan without Sage has no
    # scales.
    sage_scales_fake = (
        fake_compact(cutlass.Float32, (cute.sym_int(),), 16)
        if sage_scale_layout is not None
        else None
    )
    # MXFP8 Q/K scale atoms, one 512-byte 128x4 atom per tile and head. The
    # tile counts follow the Q and K/V extents, symbolic under runtime lengths.
    q_scales_fake = k_scales_fake = None
    if qk_dtype_key == "float8_e8m0fnu":
        q_scales_fake = fake_compact(
            cutlass.Uint8,
            (
                batch_size,
                num_qo_heads,
                cute.sym_int()
                if compile_spec.runtime_lengths
                else cute.ceil_div(max_q_len, 128),
                512,
            ),
            16,
        )
        k_scales_fake = fake_compact(
            cutlass.Uint8,
            (
                batch_size,
                num_kv_heads,
                cute.sym_int()
                if compile_spec.runtime_lengths
                else cute.ceil_div(max_seq_len_k, 128),
                512,
            ),
            16,
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    with torch.cuda.device(device_index):
        compiled = build_and_load_attention_kernel(
            "context",
            "main",
            compile_spec,
            COMPILE_OPTIONS,
            lambda: cute.compile[cute.FrontendNext](
                tensor_adapter,
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                lse_fake,
                sinks_fake,
                scale_fake,
                output_scale_fake,
                qo_indptr_fake,
                kv_indptr_fake,
                q_scales_fake,
                k_scales_fake,
                variable_window_starts_fake,
                variable_window_ends_fake,
                variable_window_cta_starts_fake,
                vc_mu_fake,
                vc_q_scale_fake,
                vc_k_scale_fake,
                sage_scales_fake,
                stream_fake,
                max_active_clusters,
                packed_query,
                cutlass.Int32(0),
                cutlass.Int32(0),
                compile_spec.vc.repair_tiles if vc_attention else 0,
                options=COMPILE_OPTIONS,
            ),
            extra=(max_active_clusters,),
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("causal_single_kv_tile", causal_single_kv_tile),
        ("packed_dense_k_mask", packed_dense_k_mask),
        ("vc_attention", vc_attention),
    )
    return compiled, policy


@functools.cache
def get_compiled_paged_context(
    compile_spec: PagedContextCompileSpec,
):
    """Compile one batch-dynamic packed-Q, paged-K/V specialization."""

    device_index = compile_spec.device_index
    max_q_len = compile_spec.max_q_len or 0
    max_kv_len = compile_spec.max_kv_len or 0
    page_size = compile_spec.page_size
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    qk_dtype_key = compile_spec.qk_dtype_key
    pv_dtype_key = compile_spec.pv_dtype_key
    out_dtype_key = compile_spec.out_dtype_key
    mask_type = compile_spec.mask_type
    window_left = compile_spec.window_left
    head_paired = compile_spec.head_paired
    uniform_packed_lengths = compile_spec.uniform_packed_lengths
    has_q_offset = compile_spec.has_q_offset
    paged_v_tail_is_zero = compile_spec.paged_v_tail_is_zero
    packed_dense_k_mask = compile_spec.packed_dense_k_mask
    scheduler = compile_spec.scheduler

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_qk_dtype = dtype_map[qk_dtype_key]
    input_pv_dtype = dtype_map[pv_dtype_key]
    out_dtype = dtype_map[out_dtype_key]
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)
    fmha = make_context_kernel(
        device_index=device_index,
        input_qk_dtype=input_qk_dtype,
        input_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=False,
        scheduler=scheduler,
        uses_ldtm_stat=default_uses_ldtm_stat(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        fp8_psmem_early_token=default_fp8_psmem_early_token(device_index),
        kv_ring_stages=default_kv_ring_stages(
            device_index, input_qk_dtype.width, input_pv_dtype.width
        ),
        umma_fp8_k64=default_umma_fp8_k64(device_index, input_qk_dtype.width),
        page_size=page_size,
        max_kv_len=compile_spec.max_kv_len,
    )
    fmha.cfg.lse_mode = compile_spec.lse_mode
    fmha.cfg.use_attention_sinks = compile_spec.use_attention_sinks
    fmha.cfg.has_varlen = True
    fmha.cfg.has_uniform_varlen = uniform_packed_lengths
    if uniform_packed_lengths:
        fmha.cfg.uniform_seq_len_q = max_q_len
        fmha.cfg.uniform_seq_len_k = max_kv_len
    fmha.cfg.has_q_offset = has_q_offset
    fmha.cfg.exp2_fma_pairs = default_exp2_fma_pairs(device_index, fmha.cfg)
    fmha.cfg.exp2_packed_f16x2 = default_exp2_packed_f16x2(device_index, fmha.cfg)
    fmha.cfg.paged_v_tail_is_zero = paged_v_tail_is_zero
    if fmha.cfg.kv_tile_n != CONTEXT_KV_TILE_N:
        raise RuntimeError(
            "context packed-K specialization assumes kv_tile_n="
            f"{CONTEXT_KV_TILE_N}, got {fmha.cfg.kv_tile_n}"
        )
    validate_query_work_tile_span(fmha.cfg)
    fmha.cfg.packed_dense_k_mask = packed_dense_k_mask

    # Bind packed Q, native pages and live token bounds to the context launcher.
    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        attention_sinks: cute.Tensor | None,
        softmax_scale: cutlass.Float32 | cute.Tensor,
        output_scale: cutlass.Float32 | cute.Tensor,
        qo_indptr: cute.Tensor,
        block_tables: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        max_q_len: cutlass.Int32,
        runtime_max_kv_len: cutlass.Int32,
    ) -> None:
        fmha(
            q,
            k_cache,
            v_cache,
            out,
            softmax_scale,
            output_scale,
            static_max_active_clusters,
            stream,
            cum_seqlen_q=qo_indptr,
            max_seqlen_q=max_q_len,
            max_seqlen_k=runtime_max_kv_len,
            block_tables=block_tables,
            seq_lens_kv=seq_lens_kv,
            lse_output=lse,
            attention_sinks=attention_sinks,
        )

    # Describe compact metadata or Q/O storage with byte-aligned pointers.
    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    runtime_total_q = cute.sym_int()
    runtime_num_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    q_fake = fake_compact(input_qk_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    kv_shape = (
        runtime_num_pages,
        num_kv_heads,
        page_size,
        head_dim,
    )

    # Keep page/head/token strides dynamic while each head dimension is contiguous.
    def fake_paged_cache(dtype):
        return cute.runtime.make_fake_tensor(
            dtype,
            kv_shape,
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=16,
        )

    k_fake = fake_paged_cache(input_qk_dtype)
    v_fake = fake_paged_cache(input_pv_dtype)
    out_fake = fake_compact(out_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    lse_fake = (
        fake_compact(cutlass.Float32, out_fake.shape[:-1], 4)
        if compile_spec.lse_mode != "none"
        else None
    )
    scale_fake = fake_scale(compile_spec.scale_types[0])
    output_scale_fake = fake_scale(compile_spec.scale_types[1])
    qo_indptr_fake = fake_compact(cutlass.Int32, (runtime_num_q_offsets,), 4)
    runtime_page_columns = cute.sym_int()
    runtime_block_table_row_stride = cute.sym_int()
    block_tables_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch_size, runtime_page_columns),
        stride=(runtime_block_table_row_stride, 1),
        assumed_align=4,
    )
    seq_lens_fake = fake_compact(cutlass.Int32, (batch_size,), 4)
    sinks_fake = (
        fake_compact(cutlass.Float32, (num_qo_heads,), 4)
        if compile_spec.use_attention_sinks
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    with torch.cuda.device(device_index):
        compiled = build_and_load_attention_kernel(
            "context",
            "paged",
            compile_spec,
            COMPILE_OPTIONS,
            lambda: cute.compile[cute.FrontendNext](
                tensor_adapter,
                q_fake,
                k_fake,
                v_fake,
                out_fake,
                lse_fake,
                sinks_fake,
                scale_fake,
                output_scale_fake,
                qo_indptr_fake,
                block_tables_fake,
                seq_lens_fake,
                stream_fake,
                max_active_clusters,
                cutlass.Int32(0),
                cutlass.Int32(0),
                options=COMPILE_OPTIONS,
            ),
            extra=(max_active_clusters,),
        )
    policy = (
        ("scheduler", scheduler),
        ("pairing", "head" if head_paired else "query"),
        ("kv_layout", "paged_hnd"),
        ("page_size", page_size),
        ("uniform_packed_lengths", uniform_packed_lengths),
        ("has_q_offset", has_q_offset),
        ("paged_v_tail_is_zero", paged_v_tail_is_zero),
        ("causal_single_kv_tile", False),
        ("packed_dense_k_mask", packed_dense_k_mask),
    )
    return compiled, policy


@functools.cache
def paged_context_compile_spec(
    geometry: PagedContextPlanGeometry,
    *,
    lse_mode: str = "none",
    use_attention_sinks: bool = False,
    scale_types: tuple[bool, bool] = (False, False),
) -> PagedContextCompileSpec:
    """Build the paged compile key, including separate QK and PV dtype keys."""
    return PagedContextCompileSpec(
        device_index=geometry.device_index,
        max_q_len=None,
        max_kv_len=None,
        page_size=geometry.page_size,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        qk_dtype_key=context_dtype_key(geometry.qk_dtype),
        pv_dtype_key=context_dtype_key(geometry.pv_dtype),
        out_dtype_key=context_dtype_key(geometry.out_dtype),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        uniform_packed_lengths=False,
        has_q_offset=geometry.has_q_offset,
        paged_v_tail_is_zero=geometry.paged_v_tail_is_zero,
        packed_dense_k_mask=(geometry.mask_type != "causal"),
        shape_signature=(),
        runtime_lengths=True,
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
        scale_types=scale_types,
        scheduler=resolve_paged_context_scheduler(geometry),
    )
