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

"""Host-side launch plumbing for the 1CTA D576 sparse (DSA) MLA decode kernel.

The D576 kernel is the ``throughput_latency_1cta`` q64 keeps-MMA-AB FP8 schedule
run with one "batch" per query token: ``cache_seqs[t]`` is the token's active
top-k width and ``page_offsets[:, t]`` its row indices into a flat row pool
with ``page_size == 1``.  K/V tiles are fetched four rows per Gather4 TMA
through the kernel's row maps of the ``[N, 576]`` E4M3 pool: latent 128 B /
SWIZZLE_128B slices and the 64 B / SWIZZLE_64B RoPE columns.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import torch

from .. import sparse_policy
from ..helpers.math import canonicalize_softmax_rescale_threshold_log2

HEAD_DIM = 576
LATENT_DIM = 512
ROPE_DIM = 64
V_DIM = 512
_SUPPORTED_HEADS = (64, 128)
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_PROFILE = "h64_keeps_mma_ab"


@dataclasses.dataclass(frozen=True)
class _SparseMlaD576Spec:
    """Compile identity of one D576 sparse specialization (persistent-cache key)."""

    num_heads: int
    sparse_mla_top_k: int
    softmax_rescale_threshold_log2: float
    split_kv: int
    head_dim_split: int
    max_active_clusters: int


@functools.lru_cache(maxsize=None)
def compile_sparse_mla_d576_kernel(
    device_index: int,
    *,
    num_heads: int,
    sparse_mla_top_k: int,
    softmax_rescale_threshold_log2: float = 0.0,
    split_kv: int = 1,
    head_dim_split: int = 1,
):
    """Compile (or load from the PrimTS disk cache) one D576 sparse 1CTA specialization.

    Kept in process by ``lru_cache`` on the argument list and on disk by
    :func:`build_and_load_attention_kernel` (same cache as the dense MLA path),
    so a new process does not recompile a specialization it has built before.

    ``softmax_rescale_threshold_log2`` is the compile-time skip-correction
    threshold of the 1CTA kernel (0 = exact rescale; see ``MlaConfig``).

    ``split_kv > 1`` splits every token's top-k list into that many contiguous
    tile ranges, one CTA each; the CTAs write BF16 partial O / LSE to the
    caller's workspace (:func:`sparse_mla_d576_workspace_bytes`) and the kernel's
    standalone reducer (launched with PDL) merges them into ``out`` / ``lse``.
    ``head_dim_split > 1`` (split launches only) also splits the 512 V columns
    over that many CTAs: each one still runs the whole QK / softmax for its
    tile range but gathers, multiplies and writes only its V slice.
    """

    import cutlass
    import cutlass.cute as cute

    from ...._compile_cache import build_and_load_attention_kernel
    from ...._hardware import get_max_active_clusters
    from .config import MlaProfile
    from .kernel import ThroughputLatencyMlaDecodeTs

    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)

    # The q64 keeps-MMA-AB schedule tiles 64 flat query rows per CTA; H=128
    # runs as two physical "query positions" of 64 heads over the same token.
    phys_heads = 64
    phys_sq = num_heads // phys_heads
    profile = _PROFILE
    if head_dim_split > 1 and split_kv == 1:
        raise NotImplementedError("D576 sparse V head-dim split requires split-KV")
    if split_kv > 1:
        # Explicit profile: the generic split profile derives the V head-dim
        # split from the compile-time batch (1 here, one batch per token at
        # run time); the caller picks it from the run-time token count instead.
        hdim = "" if head_dim_split == 1 else f"_hdim{V_DIM // head_dim_split}"
        profile = MlaProfile(
            name=f"h64_keeps_mma_ab_splitkv{split_kv}{hdim}_gmem",
            num_ctas_per_seq_kv=split_kv,
            num_ctas_per_head_dim=head_dim_split,
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=1,
            use_cluster_reduction=0,
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
        )
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=phys_heads,
        seq_len_q=phys_sq,
        # The 1CTA profiles start at one 128-key tile; a smaller top-k still
        # runs one tile, and each token's active width comes from top_k_lens.
        # (the plan rounds the capacity up to whole tiles.)
        seq_len_k=max(sparse_mla_top_k, 128),
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
        page_size=1,
        max_active_clusters=max_active_clusters,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=profile,
        reduction_mode=None if split_kv == 1 else "gmem_separate",
        logical_num_heads=num_heads,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=None,
        explicit_persistent=None,
        mask_type="dense",
        softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
    )
    cfg = kernel._make_config()
    if (
        int(cfg.num_ctas_per_seq_kv) != split_kv
        or int(cfg.num_ctas_per_head_dim) != head_dim_split
        or int(cfg.use_cluster_reduction) != 0
    ):
        raise NotImplementedError(
            f"D576 sparse 1CTA MLA expects {split_kv} KV CTAs per token, "
            f"{head_dim_split} V head-dim CTAs and GMEM reduction; got {cfg.num_ctas_per_seq_kv}, "
            f"{cfg.num_ctas_per_head_dim}, cluster={cfg.use_cluster_reduction}"
        )

    total_q = cute.sym_int()
    pool_rows = cute.sym_int()
    table_columns = cute.sym_int()
    table_row_stride = cute.sym_int64(divisibility=1)
    e4m3 = cutlass.Float8E4M3FN
    rope_dim = ROPE_DIM
    head_dim = HEAD_DIM
    q_stride = (head_dim, 1, phys_heads * head_dim, num_heads * head_dim)
    q_latent_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (phys_heads, LATENT_DIM, phys_sq, total_q),
        stride=q_stride,
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (phys_heads, rope_dim, phys_sq, total_q),
        stride=q_stride,
        assumed_align=16,
    )
    c_latent_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (1, LATENT_DIM, pool_rows),
        stride=(head_dim, 1, head_dim),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        e4m3, (1, rope_dim, pool_rows), stride=(head_dim, 1, head_dim), assumed_align=16
    )
    page_offsets_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (table_columns, total_q),
        stride=(1, table_row_stride),
        assumed_align=4,
    )
    out_stride = (V_DIM, 1, phys_heads * V_DIM, num_heads * V_DIM)
    out_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (phys_heads, V_DIM, phys_sq, total_q),
        stride=out_stride,
        assumed_align=16,
    )
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (phys_heads, phys_sq, total_q),
        stride=(1, phys_heads, num_heads),
        assumed_align=16,
    )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (total_q,), stride_order=(0,), assumed_align=4
    )
    workspace_fake = None
    if split_kv > 1:
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8, (cute.sym_int(),), stride_order=(0,), assumed_align=32
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_kernel():
        return cute.compile[cute.FrontendNext](
            kernel,
            q_latent_fake,
            q_rope_fake,
            c_latent_fake,
            c_rope_fake,
            page_offsets_fake,
            out_fake,
            lse_fake,
            workspace_fake,
            cutlass.Int32(split_kv),
            cache_seqs_fake,
            None,  # cu_seqlens_q
            None,  # block_split_kvs
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream_fake,
            options=_COMPILE_OPTIONS,
        )

    spec = _SparseMlaD576Spec(
        num_heads=num_heads,
        sparse_mla_top_k=sparse_mla_top_k,
        softmax_rescale_threshold_log2=float(softmax_rescale_threshold_log2),
        split_kv=split_kv,
        head_dim_split=head_dim_split,
        max_active_clusters=max_active_clusters,
    )
    with torch.cuda.device(device_index):
        return build_and_load_attention_kernel(
            "mla_decode", "sparse_mla_d576", spec, _COMPILE_OPTIONS, compile_kernel
        )


def sparse_mla_d576_workspace_bytes(total_q: int, num_heads: int, split_kv: int) -> int:
    """GMEM bytes of the split-KV partials: BF16 O ``[T, SQ, 64, S, 512]`` + FP32 LSE."""
    if split_kv <= 1:
        return 0
    rows = total_q * num_heads * split_kv
    return rows * (V_DIM * 2 + 4)


def launch_sparse_mla_d576_kernel(
    compiled,
    *,
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    top_k_lens: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    workspace: torch.Tensor | None = None,
    split_kv: int = 1,
) -> None:
    """Bind ``[T, H, 576]`` packed inputs to the one-token-per-batch 1CTA ABI and launch."""

    total_q, num_heads, head_dim = query.shape
    phys_heads = 64
    phys_sq = num_heads // phys_heads
    q4 = query.view(total_q, phys_sq, phys_heads, head_dim)
    q_latent = q4[..., :LATENT_DIM].permute(2, 3, 1, 0)
    q_rope = q4[..., LATENT_DIM:].permute(2, 3, 1, 0)
    pool3 = kv_pool.view(-1, 1, head_dim)
    c_latent = pool3[..., :LATENT_DIM].permute(1, 2, 0)
    c_rope = pool3[..., LATENT_DIM:].permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)  # [Kmax, T], stride (1, Kmax)
    out_kernel = out.view(total_q, phys_sq, phys_heads, V_DIM).permute(2, 3, 1, 0)
    lse_kernel = lse.view(total_q, phys_sq, phys_heads).permute(2, 1, 0)
    # TVM-FFI takes the stream from the device of the first CUDA argument
    # (q_latent); keep the current device on it for any host-side CUDA call.
    with torch.cuda.device(query.device):
        compiled(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_offsets,
            out_kernel,
            lse_kernel,
            workspace,
            split_kv,
            top_k_lens,
            None,
            None,
            float(bmm1_scale),
            float(bmm2_scale),
            # output_quant_args: a kernel reloaded from the disk cache takes
            # every parameter positionally, including this defaulted one.
            None,
        )


# Plan-time helpers of the sparse MLA wrapper's D576 mode.


def check_plan(
    *,
    num_heads: int,
    max_topk: int,
    max_extra_topk: int,
    q_data_type: torch.dtype,
    has_sinks: bool,
    output_quant_block_size: Optional[int],
) -> None:
    """Reject wrapper options the D576 schedule does not implement."""
    if q_data_type != torch.float8_e4m3fn:
        raise ValueError("D576 sparse MLA (qk_rope_head_dim=64) requires E4M3 Q / KV")
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(
            "D576 sparse MLA supports 64 or 128 query heads (64-row tiles), "
            f"got {num_heads}"
        )
    if max_extra_topk:
        raise ValueError("D576 sparse MLA attends to one KV pool (max_extra_topk=0)")
    if has_sinks:
        raise ValueError("D576 sparse MLA does not support attention sinks")
    if output_quant_block_size is not None:
        raise ValueError("D576 sparse MLA does not support output quantization")


def resolve_split(device_index: int, max_rows: int, num_heads: int, max_topk: int):
    """``(split_kv, head_dim_split)`` for a plan of ``max_rows`` query tokens.

    Fixed at plan time, so every run of the plan (and a CUDA graph captured
    from it) uses one compiled kernel.
    """
    return sparse_policy.select_sparse_mla_d576_split(
        max_rows * (num_heads // 64),
        max_topk,
        torch.cuda.get_device_properties(device_index).multi_processor_count,
    )


def compile_plan(
    device_index: int,
    *,
    num_heads: int,
    max_topk: int,
    softmax_rescale_threshold_log2: Optional[float],
    split_kv: int,
    head_dim_split: int,
):
    """Compile (or load from the PrimTS disk cache) the kernel of one plan."""
    threshold = canonicalize_softmax_rescale_threshold_log2(
        0.0
        if softmax_rescale_threshold_log2 is None
        else softmax_rescale_threshold_log2,
        qkv_dtype="e4m3",
    )
    return compile_sparse_mla_d576_kernel(
        device_index,
        num_heads=num_heads,
        # The kernel uses the capacity only through its 128-key tile count:
        # capacities in one tile bucket share a kernel.
        sparse_mla_top_k=-(-max_topk // 128) * 128,
        softmax_rescale_threshold_log2=threshold,
        split_kv=split_kv,
        head_dim_split=head_dim_split,
    )


def kv_rows(cache: torch.Tensor, *, device, kv_layout: str, name: str) -> torch.Tensor:
    """Flatten an E4M3 ``[N, 576]`` / ``[pages, page_size, 576]`` / one-KV-head
    4-D cache to the ``[rows, 576]`` view the Gather4 maps address."""
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if cache.ndim == 4:
        axis = 1 if kv_layout == "HND" else 2
        if cache.shape[axis] != 1:
            raise ValueError(f"{name} requires one KV head")
        cache = cache.squeeze(axis)
    if cache.ndim not in (2, 3) or cache.shape[-1] != HEAD_DIM or cache.numel() == 0:
        raise ValueError(
            f"{name} must have shape [N,{HEAD_DIM}] or [pages,page_size,{HEAD_DIM}]"
        )
    if cache.device != device or cache.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name} must be an E4M3 tensor on the planned device")
    if not cache.is_contiguous() or cache.data_ptr() % 16:
        raise ValueError(
            f"{name} must be contiguous and start at a 16-byte-aligned address"
        )
    rows = cache.view(-1, HEAD_DIM)
    if rows.shape[0] >= 0x7FFFFFFF:
        raise ValueError(f"{name} exceeds the int32 row range")
    return rows
