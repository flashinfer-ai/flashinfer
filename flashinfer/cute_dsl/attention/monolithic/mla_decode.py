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

"""
CuTe DSL MLA Decode Kernel Integration
=======================================

Wraps NVIDIA's CuTe DSL MLA decode kernels (FP16/BF16/FP8) for Blackwell SM100
and exposes them via a PyTorch API compatible with FlashInfer's MLA backend.
"""

import functools
from typing import Callable, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.utils import device_support_pdl

from .mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16
from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from .mla_helpers import MAX_SPLITS, ceil_div, compute_q_tile_layout
from flashinfer.cute_dsl.utils import (
    _as_cute_dsl_workspace_i8,
    get_max_active_clusters,
    get_num_sm,
    torch_to_cutlass_dtype,
)


_CUDA_GRID_Y_MAX = 65_535
_REDUCER_D_TILE_CANDIDATES = (1, 2, 4)
_STATIC_REDUCER_MAX_SPLITS = 32


def _validate_nonpersistent_grid_y(
    batch_size: int, num_q_tiles: int, is_persistent: bool
) -> None:
    """Reject nonpersistent launch grids whose Y dimension CUDA cannot encode."""
    grid_y = batch_size * num_q_tiles
    if not is_persistent and grid_y > _CUDA_GRID_Y_MAX:
        raise ValueError(
            "nonpersistent CuTeDSL MLA grid.y would be "
            f"{grid_y}, exceeding the CUDA limit {_CUDA_GRID_Y_MAX} "
            f"(batch_size={batch_size}, num_q_tiles={num_q_tiles})"
        )


def _get_reducer_d_tiles(
    batch_size: int,
    seq_len_q: int,
    num_heads: int,
    num_sms: int,
    effective_split_kv: int,
) -> int:
    """Choose output-side reducer parallelism for an underfilled row grid.

    A reducer CTA normally owns one D512 row.  When the real row grid does not
    cover one resident wave, compare the conservative wave count for one, two,
    or four equal D bands and keep the smallest topology with the shortest
    per-band critical path.  Once rows already cover every SM, retain one CTA
    per row and avoid duplicated LSE work.
    """
    reducer_rows = batch_size * seq_len_q * num_heads
    if (
        reducer_rows <= 0
        or num_sms <= 0
        or reducer_rows >= num_sms
        or effective_split_kv <= 1
    ):
        return 1

    best_tiles = 1
    best_waves = ceil_div(reducer_rows, num_sms)
    for d_tiles in _REDUCER_D_TILE_CANDIDATES[1:]:
        if d_tiles > effective_split_kv:
            continue
        waves = ceil_div(reducer_rows * d_tiles, num_sms)
        if waves * best_tiles < best_waves * d_tiles:
            best_tiles = d_tiles
            best_waves = waves
    return best_tiles


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: Optional[int] = None,
    occupancy_q_tiles: Optional[int] = None,
) -> Tuple[int, int]:
    """Return the nonempty split count and its workspace requirement.

    The occupancy heuristic can request more splits than the kernel's uniform
    contiguous partitioning actually uses.  When ``max_seq_len`` is known,
    normalize the candidate to the number of nonempty fixed-size K chunks so
    the grid and reducer do not carry an empty final split.

    ``occupancy_q_tiles`` may tighten the occupancy estimate for a compact
    variable-Q launch. Workspace sizing always retains the full rectangular
    ``B * ceil(max_q_len * H / 128)`` capacity.
    """
    # Flatten (query_token, head) into one contiguous row space.  A cooperative
    # 2-CTA MMA tile owns 128 consecutive rows and may cross token boundaries;
    # only the final query tile of each request can be padded.
    mma_qk_tile_m = 128
    mma_qk_tile_n = 128
    _, num_q_tiles, _ = compute_q_tile_layout(H, q_len, mma_qk_tile_m)
    rectangular_q_tiles = B * num_q_tiles
    occupancy_q_tiles = (
        rectangular_q_tiles if occupancy_q_tiles is None else occupancy_q_tiles
    )
    if occupancy_q_tiles <= 0 or occupancy_q_tiles > rectangular_q_tiles:
        raise ValueError(
            "occupancy_q_tiles must be in "
            f"[1, {rectangular_q_tiles}], got {occupancy_q_tiles}"
        )
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
        1, occupancy_q_tiles, max_active_blocks
    )
    if max_seq_len is not None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        k_tile_total = ceil_div(max_seq_len, mma_qk_tile_n)
        k_tiles_per_split = ceil_div(k_tile_total, split_kv)
        split_kv = ceil_div(k_tile_total, k_tiles_per_split)
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        mma_qk_tile_m, num_q_tiles, kv_lora_rank, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


@functools.cache
def _check_can_implement(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    enable_dcp: bool = False,
    cp_world: int = 1,
) -> None:
    """Check if the kernel supports the given configuration (cached)."""
    if not isinstance(enable_dcp, bool):
        raise TypeError(f"enable_dcp must be a bool, got {type(enable_dcp).__name__}")
    if not isinstance(cp_world, int) or isinstance(cp_world, bool) or cp_world <= 0:
        raise ValueError(f"cp_world must be a positive integer, got {cp_world!r}")
    if not enable_dcp and cp_world != 1:
        raise ValueError(
            f"cp_world={cp_world} requires enable_dcp=True; disabled DCP uses cp_world=1"
        )

    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_in_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cutlass_in_dtype,
        cutlass_out_dtype,
        cutlass.Float32,
        cutlass.Float32,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"cute_dsl_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"in_dtype={torch_dtype}, out_dtype={torch_out_dtype})"
        )


@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    seq_len_q: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_q: bool,
    is_var_split_kv: bool,
    reducer_d_tiles: int = 1,
    reducer_max_splits: int = MAX_SPLITS,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    enable_pdl: bool = False,
    enable_dcp: bool = False,
    cp_world: int = 1,
) -> Callable:
    """Compile and cache an MLA decode kernel.

    Returns a callable that accepts (q_latent, q_rope, c_latent, c_rope,
    page_table, o, lse, workspace, split_kv_scalar, cache_seqs,
    cum_seq_lens_q, causal_seqlens_kv_global, cp_rank_scalar,
    block_split_kvs, softmax_scale_scalar, output_scale_scalar).

    All scalar arguments must be pre-wrapped as Int32/Float32.
    """
    # Tile sizes for Blackwell mma.
    # (128, 128) for QK and (128, 256) for PV.
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)
    # 2 CTAs along M (num_heads)
    cluster_shape_mnk = (2, 1, 1)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)

    kernel_obj = KernelClass(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=mma_qk_tiler_mn,
        mma_pv_tiler_mn=mma_pv_tiler_mn,
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        enable_pdl=enable_pdl,
        is_var_q=is_var_q,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        reducer_d_tiles=reducer_d_tiles,
        reducer_max_splits=reducer_max_splits,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
    )

    # All dimensions as sym_int — this matches the original kernel's use of
    # mark_compact_shape_dynamic, which makes ALL shapes dynamic CuTe Integers.
    # Static Python ints would cause cute.assume() to fail with AttributeError
    # inside initialize_workspace() since it expects DSL Integer types.
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int(divisibility=16)
    sym_rope = cute.sym_int(divisibility=16)
    sym_batch = cute.sym_int()  # query/output batch dimension
    sym_kv_batch = cute.sym_int()  # KV cache batch dim (flat pool, =1 in paged mode)
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    # q_latent, q_rope, c_latent, c_rope are slices of contiguous tensors on
    # the last dim (e.g. query[..., :kv_lora_rank]), so they are NOT contiguous:
    #   stride[-2] = D_qk (original full last dim), not the sliced shape.
    # Use make_fake_tensor with fully dynamic strides so the compiled kernel
    # reads actual strides from the runtime tensor.  Last-dim stride is always 1.

    if is_var_q:
        # Compact variable-Q tensors: [total_q, num_heads, dim].
        sym_total_q = cute.sym_int()
        q_latent_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype,
            (sym_total_q, sym_heads, sym_latent),
            stride=(cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
        q_rope_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype,
            (sym_total_q, sym_heads, sym_rope),
            stride=(cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
    else:
        # Fixed Q: [batch_size, seq_len_q, num_heads, dim].
        sym_seq_q = cute.sym_int()
        q_latent_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype,
            (sym_batch, sym_seq_q, sym_heads, sym_latent),
            stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
        q_rope_fake = cute.runtime.make_fake_tensor(
            cutlass_dtype,
            (sym_batch, sym_seq_q, sym_heads, sym_rope),
            stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
    # c_latent: [kv_batch, seq_len_k, latent_dim] — non-contiguous slice
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=num_pages at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_rope: [kv_batch, seq_len_k, rope_dim] — non-contiguous slice
    c_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # page_table: [batch_size, page_count] — contiguous
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=16,
    )
    if is_var_q:
        o_fake = cute.runtime.make_fake_compact_tensor(
            cutlass_out_dtype,
            (sym_total_q, sym_heads, sym_latent),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        lse_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (sym_total_q, sym_heads),
            stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        o_fake = cute.runtime.make_fake_compact_tensor(
            cutlass_out_dtype,
            (sym_batch, sym_seq_q, sym_heads, sym_latent),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        lse_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (sym_batch, sym_seq_q, sym_heads),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
    if is_workspace_size_zero:
        workspace_fake = None
    else:
        # workspace: 1-D int8 buffer. 32-byte alignment because workspace stores
        # fp32 partial sums internally, requiring stricter alignment than tensors.
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=16,
    )
    if is_var_q:
        cum_seq_lens_q_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(),),
            assumed_align=16,
        )
    else:
        cum_seq_lens_q_fake = None

    # DCP's causal boundary is global while cache_seqs remains the physical
    # rank-local length used for paging and split-K traversal. Keep the tensor
    # absent from the disabled specialization so no load or pointer argument
    # reaches its generated device code.
    if enable_dcp:
        causal_seqlens_kv_global_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=4,
        )
    else:
        causal_seqlens_kv_global_fake = None
    # block_split_kvs: [batch_size] — int32 (only needed for is_var_split_kv=True)
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=16,
        )
    else:
        block_split_kvs_fake = None

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        Int32(1),  # split_kv placeholder
        cache_seqs_fake,
        cum_seq_lens_q_fake,
        causal_seqlens_kv_global_fake,
        Int32(0),  # cp_rank placeholder (runtime-uniform, not a JIT key)
        block_split_kvs_fake,
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )

    return compiled_kernel


# TODO: query[..., :kv_lora_rank], do we need to remove such kind of slice and move the logic to call routine in the kernel file.
def cute_dsl_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    is_var_seq: bool = True,
    enable_pdl: Optional[bool] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """CuTe DSL MLA decode kernel for Blackwell SM100.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor where ``D_qk = kv_lora_rank + qk_rope_head_dim``.
        Without ``cum_seq_lens_q``, its shape is ``[B, q_len, H, D_qk]``.
        With ``cum_seq_lens_q``, query tokens are compacted across requests
        and its shape is ``[total_q, H, D_qk]``.
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer (int8 or uint8). Required size depends on batch size
        and split_kv (auto-computed from B, q_len, and number of SMs):

        - Formula: ``B * 128 * q_tiles * split_kv * (kv_lora_rank + 1) * 4``
          bytes, where ``q_tiles = ceil(q_len * H / 128)``
          (0 when ``split_kv == 1``, typically once ``B * q_tiles >= num_SMs / 2``)
        - Typical max: ~18 MB on a 148-SM GPU (e.g. B=4..8, H=128, D=512)
        - Safe default: 128 MB covers all realistic configurations
    kv_lora_rank : int
        Latent dimension (e.g. 512).
    qk_rope_head_dim : int
        RoPE dimension (e.g. 64).
    block_tables : torch.Tensor
        [B, max_pages] — page table indices.
    seq_lens : torch.Tensor
        [B] — per-request KV sequence lengths.
    max_seq_len : int
        Maximum sequence length across the batch.
    softmax_scale : float
        Scale factor for QK^T before softmax.
    output_scale : float
        Scale factor applied to the output.
    out : Optional[torch.Tensor]
        Contiguous pre-allocated output tensor. Its shape is
        ``[B, q_len, H, kv_lora_rank]`` for fixed Q or
        ``[total_q, H, kv_lora_rank]`` for compact variable Q.
    out_dtype : Optional[torch.dtype]
        Output data type. If None, defaults to torch.bfloat16 (matching trtllm-gen backend).
        Supported values: torch.bfloat16, torch.float8_e4m3fn (FP8 input only),
        torch.float16, torch.bfloat16 (FP16/BF16 input).
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    enable_pdl : Optional[bool], default=None
        Whether to enable Programmatic Dependent Launch (PDL).
        If None, auto-detects based on device capability.
    lse : Optional[torch.Tensor]
        Pre-allocated Log-Sum-Exp buffer.  Accepted shapes (dtype must be
        ``torch.float32``):

        * ``[B, q_len, H]`` (native kernel layout, no reshape), or
        * ``[B * q_len, H]`` (matches ``trtllm-gen`` shape; the wrapper
          reshapes via ``.view`` to the native layout), or
        * ``[total_q, H]`` for compact variable Q.

        Caller-provided buffers must be contiguous.

        If ``return_lse`` is True and this is None, a buffer of the native
        fixed or compact shape is allocated internally.
    return_lse : bool
        Whether to return LSE values.  When True, the function returns
        ``(out, lse)`` (the ``lse`` tensor returned is in whatever shape
        the caller supplied).
    cum_seq_lens_q : Optional[torch.Tensor]
        Device-resident int32 cumulative query lengths with shape
        ``[B + 1]``. When provided, selects compact variable-Q input and
        output layouts. Empty requests are allowed.
    max_q_len : Optional[int]
        Static launch capacity for variable Q. Supplying it avoids a host
        synchronization and makes the call CUDA-graph-capturable. It must be
        at least the largest request query length; the compiled kernel is
        specialized to this capacity.
    enable_dcp : bool
        Enable static cyclic decode context-parallel masking. DCP returns a
        rank-local attention state and therefore requires ``return_lse=True``.
    cp_world : int
        Compile-time context-parallel world size. The local cache owns global
        token positions ``cp_world * local_k + cp_rank``.
    cp_rank : int
        Runtime-uniform context-parallel rank.
    causal_seqlens_kv_global : Optional[torch.Tensor]
        Contiguous CUDA int32 tensor ``[B]`` containing the global exclusive
        causal bound for the newest query token. Required when DCP is enabled.
    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Fixed-Q output has shape ``[B, q_len, H, kv_lora_rank]``. Compact
        variable-Q output has shape ``[total_q, H, kv_lora_rank]``. When
        ``return_lse=True``, returns ``(output, lse)``.
    """
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    assert query.dtype in supported_dtypes, (
        f"cute_dsl_mla_decode only supports {supported_dtypes}, got {query.dtype}"
    )
    assert kv_cache.dtype == query.dtype, (
        f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    )
    is_var_q = cum_seq_lens_q is not None
    if is_var_q:
        if query.ndim != 3:
            raise ValueError("query must have shape [total_q, num_heads, head_dim_qk]")
        total_q, H, D_qk = query.shape
        B = cum_seq_lens_q.numel() - 1
        if max_q_len is None:
            q_lens = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
            max_q_len = int(q_lens.max().item())
        if max_q_len <= 0:
            raise ValueError("max_q_len must be greater than 0")
        q_len = max_q_len
        cum_seq_lens_q = cum_seq_lens_q.contiguous()
    else:
        if query.ndim != 4:
            raise ValueError(
                "query must have shape "
                "[batch_size, q_len_per_request, num_heads, head_dim_qk]"
            )
        B, q_len, H, D_qk = query.shape
        total_q = B * q_len
    assert D_qk == kv_lora_rank + qk_rope_head_dim

    if not isinstance(enable_dcp, bool):
        raise TypeError(f"enable_dcp must be a bool, got {type(enable_dcp).__name__}")
    if not isinstance(cp_world, int) or isinstance(cp_world, bool) or cp_world <= 0:
        raise ValueError(f"cp_world must be a positive integer, got {cp_world!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool):
        raise TypeError(f"cp_rank must be an integer, got {type(cp_rank).__name__}")

    if enable_dcp:
        if is_var_q:
            raise ValueError("DCP does not support cum_seq_lens_q / max_q_len")
        if not 0 <= cp_rank < cp_world:
            raise ValueError(
                f"cp_rank must satisfy 0 <= cp_rank < cp_world, got "
                f"cp_rank={cp_rank}, cp_world={cp_world}"
            )
        if not return_lse:
            raise ValueError(
                "enable_dcp=True requires return_lse=True so rank-local states "
                "can be merged with their LSE values"
            )
        if causal_seqlens_kv_global is None:
            raise ValueError(
                "causal_seqlens_kv_global is required when enable_dcp=True"
            )
        if not isinstance(causal_seqlens_kv_global, torch.Tensor):
            raise TypeError(
                "causal_seqlens_kv_global must be a torch.Tensor, got "
                f"{type(causal_seqlens_kv_global).__name__}"
            )
        if causal_seqlens_kv_global.dtype != torch.int32:
            raise ValueError(
                "causal_seqlens_kv_global must have dtype torch.int32, got "
                f"{causal_seqlens_kv_global.dtype}"
            )
        if not causal_seqlens_kv_global.is_cuda:
            raise ValueError("causal_seqlens_kv_global must be a CUDA tensor")
        if causal_seqlens_kv_global.device != query.device:
            raise ValueError(
                "causal_seqlens_kv_global must be on the query device "
                f"{query.device}, got {causal_seqlens_kv_global.device}"
            )
        if tuple(causal_seqlens_kv_global.shape) != (B,):
            raise ValueError(
                f"causal_seqlens_kv_global must have shape ({B},), got "
                f"{tuple(causal_seqlens_kv_global.shape)}"
            )
        if not causal_seqlens_kv_global.is_contiguous():
            raise ValueError("causal_seqlens_kv_global must be contiguous")
    else:
        nondefault = []
        if cp_world != 1:
            nondefault.append(f"cp_world={cp_world}")
        if cp_rank != 0:
            nondefault.append(f"cp_rank={cp_rank}")
        if causal_seqlens_kv_global is not None:
            nondefault.append("causal_seqlens_kv_global")
        if nondefault:
            raise ValueError(
                "DCP arguments require enable_dcp=True; got " + ", ".join(nondefault)
            )

    # Each request's Q descriptor flattens (query_token, head) into one affine
    # row mode. Adjacent tokens must therefore follow all H head rows.
    # Fixed-Q keeps batch as a separate descriptor mode, while compact
    # variable-Q uses one global token mode.
    token_stride_ok = query.stride(-3) == H * query.stride(-2)
    if query.stride(-1) != 1 or not token_stride_ok:
        query = query.contiguous()

    # O/LSE are compiled with compact layouts. In particular, a token-gapped
    # view cannot be flattened across an M128 tile boundary and would be
    # silently written through the gap, so reject noncompact buffers early.
    if out is not None and not out.is_contiguous():
        raise ValueError(f"out must be contiguous, got strides {out.stride()}")
    if lse is not None and not lse.is_contiguous():
        raise ValueError(f"lse must be contiguous, got strides {lse.stride()}")

    q_dtype = query.dtype
    # Resolve output dtype: for FP8 input, default to bfloat16 (matching trtllm-gen backend);
    # for FP16/BF16 input, default to same as input. Allow override via out_dtype or out tensor.
    if out is not None:
        o_dtype = out.dtype
    elif out_dtype is not None:
        o_dtype = out_dtype
    elif q_dtype == torch.float8_e4m3fn:
        o_dtype = torch.bfloat16
    else:
        o_dtype = q_dtype

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components. The kernel reinterprets
    # fixed Q with an explicit batch mode and compact Q with one global
    # token/head-row mode.
    q_latent_k = query[..., :kv_lora_rank]
    q_rope_k = query[..., kv_lora_rank:]

    # KV cache slices — keep contiguous [num_pages, page_size, D].
    # The kernel reinterprets to [page_size, D, num_pages] internally.
    c_latent_k = kv_cache[:, :, :kv_lora_rank]
    c_rope_k = kv_cache[:, :, kv_lora_rank:]

    # Page table: [B, max_pages]: passed directly, kernel reinterprets.
    page_table_k = block_tables

    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

    _, num_q_tiles, _ = compute_q_tile_layout(H, q_len)

    # Compact storage exposes total_q without reading the device indptr. Since
    # H <= 128, adding one token can add at most one M128 tile; therefore
    # min(total_q, rectangular_tiles) is a safe upper bound on active tiles.
    # Use that tighter bound only for occupancy. The helper still sizes the
    # workspace for every rectangular scheduler slot.
    occupancy_q_tiles = (
        max(1, min(total_q, B * num_q_tiles)) if is_var_q else B * num_q_tiles
    )

    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B,
        q_len,
        H,
        kv_lora_rank,
        max_active_blocks,
        max_seq_len,
        occupancy_q_tiles=occupancy_q_tiles,
    )

    is_persistent = not is_var_seq
    _validate_nonpersistent_grid_y(B, num_q_tiles, is_persistent)

    # Prepare workspace: the CUTE signature uses int8, while public FlashInfer
    # workspace buffers are byte storage and may be uint8.
    workspace_buffer = _as_cute_dsl_workspace_i8(workspace_buffer)
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
            f"need {workspace_size} bytes"
        )
    is_workspace_size_zero = workspace_size == 0
    reducer_d_tiles = (
        1
        if is_workspace_size_zero
        else _get_reducer_d_tiles(
            # The reducer grid remains rectangular for variable Q, including
            # inactive request/token slots. Base output slicing on that
            # physical grid so sparse batches do not multiply empty CTAs.
            B,
            q_len,
            H,
            max_active_blocks,
            split_kv,
        )
    )
    if is_workspace_size_zero:
        workspace_bytes = None
    else:
        workspace_bytes = workspace_buffer[:workspace_size]
    output_shape = (
        (total_q, H, kv_lora_rank) if is_var_q else (B, q_len, H, kv_lora_rank)
    )
    if out is not None:
        o_k = out
    else:
        o_k = torch.empty(output_shape, dtype=o_dtype, device=query.device)

    # Fixed Q accepts both public LSE shapes. Compact variable Q has one
    # natural layout, [total_q, H].
    if lse is not None:
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must be torch.float32, got {lse.dtype}")
        if is_var_q:
            if lse.shape != (total_q, H):
                raise ValueError(
                    f"lse must have shape ({total_q}, {H}) for variable Q; "
                    f"got {tuple(lse.shape)}"
                )
            lse_k = lse
        else:
            if lse.shape == (B, q_len, H):
                lse_k = lse
            elif lse.shape == (B * q_len, H):
                lse_k = lse.view(B, q_len, H)
            else:
                raise ValueError(
                    f"lse must have shape (B, q_len, H)=({B}, {q_len}, {H}) "
                    f"or (B*q_len, H)=({B * q_len}, {H}); "
                    f"got {tuple(lse.shape)}"
                )
    else:
        lse_shape = (total_q, H) if is_var_q else (B, q_len, H)
        lse_k = torch.empty(lse_shape, dtype=torch.float32, device=query.device)

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    is_var_split_kv = False
    block_split_kvs = None
    skip_correction_threshold = 0.0

    # Validate configuration (cached, negligible overhead after first call)
    _check_can_implement(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
    )

    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl

    # Get compiled kernel (cached after first compile)
    # Note: when is_workspace_size_zero is True, workspace_bytes is None and it will launch one kernel without workspace.
    # Otherwise, workspace_bytes is not None and it will launch two kernels.
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_heads=H,
        seq_len_q=q_len,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_q=is_var_q,
        is_var_split_kv=is_var_split_kv,
        reducer_d_tiles=reducer_d_tiles,
        reducer_max_splits=_STATIC_REDUCER_MAX_SPLITS,
        skip_correction_threshold=skip_correction_threshold,
        is_workspace_size_zero=is_workspace_size_zero,
        enable_pdl=enable_pdl,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
    )

    # Call the kernel
    compiled_kernel(
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        lse_k,
        workspace_bytes,
        Int32(split_kv),
        cache_seqs,
        cum_seq_lens_q,
        causal_seqlens_kv_global,
        Int32(cp_rank),
        block_split_kvs,
        Float32(softmax_scale),
        Float32(output_scale),
    )

    if return_lse:
        # Return the lse tensor in the shape the caller supplied. For fixed Q,
        # a caller's 2D buffer is viewed as 3D internally but shares storage.
        return o_k, (lse if lse is not None else lse_k)

    return o_k
