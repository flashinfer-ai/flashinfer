"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""GVR Top-K kernels for Blackwell (sm_100+).

Exposes the CuTe DSL GVR Top-K kernel family as a FlashInfer API.
Three public entry points:

* :func:`gvr_topk_decode`        — single-CTA or cluster-mode decode
* :func:`gvr_topk_sort_prepare`  — LJF sort for seqlen-sorted dispatch
* :func:`gvr_topk_lb_prepare`    — load-balance classifier (once per step)
* :func:`gvr_topk_lb_decode`     — hybrid long/short dispatch main kernel

Hardware requirement: NVIDIA Blackwell (sm_100 / B200).  Calling these
functions on pre-Blackwell hardware raises ``RuntimeError``.
"""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from .api_logging import flashinfer_api
from .cute_dsl.utils import (
    get_num_sm,
    is_cute_dsl_available,
    torch_to_cutlass_dtype,
)
from .utils import get_compute_capability

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------


def can_use_gvr_topk(device: torch.device) -> bool:
    """Return True iff the device is Blackwell (sm_100+) and CuTe DSL is available."""
    if not is_cute_dsl_available():
        return False
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        return False
    cap = get_compute_capability(device)
    return cap[0] >= 10


# ---------------------------------------------------------------------------
# Internal: CuTe-DSL compile cache
# ---------------------------------------------------------------------------

if is_cute_dsl_available():
    from .cute_dsl.top_k import GvrTopKKernel, GvrTopKLBKernel, GvrTopKLBPrepareKernel


@functools.cache
def _compile_gvr(
    cute_dtype,
    top_k: int,
    next_n: int,
    enable_unroll_4: bool,
    enable_phase3_unroll: bool,
    use_constant_hint: bool,
    min_blocks_per_mp: int,
    use_256bit_load: bool,
    num_threads_per_block: int,
    enable_warp_parallel_reduce: bool,
    compress_ratio: int,
    return_output_values: bool,
    cluster_size: int,
    seqlen_sorted: bool,
):
    """JIT-compile the GVR kernel for a specific knob combination.

    ``functools.cache`` keys on all args so repeated calls in the same
    process reuse the compiled kernel without an explicit module-level dict.
    """
    n_rows = cute.sym_int()
    n_cols = cute.sym_int()
    n_batch = cute.sym_int()
    in_align = 32 if use_256bit_load else 16
    input_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        (n_rows, n_cols),
        stride_order=(1, 0),
        assumed_align=in_align,
    )
    pre_idx_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch,),
        stride_order=(0,),
    )
    out_values_fake = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype,
            (n_rows, top_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if return_output_values
        else None
    )
    out_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    order_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        if seqlen_sorted
        else None
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = GvrTopKKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads_per_block,
        enable_unroll_4=enable_unroll_4,
        enable_phase3_unroll=enable_phase3_unroll,
        use_constant_hint=use_constant_hint,
        min_blocks_per_mp=min_blocks_per_mp,
        use_256bit_load=use_256bit_load,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        seqlen_sorted=seqlen_sorted,
    )
    return cute.compile(
        kernel,
        input_fake,
        pre_idx_fake,
        seq_lens_fake,
        out_values_fake,
        out_indices_fake,
        order_row_fake,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Public API: gvr_topk_decode
# ---------------------------------------------------------------------------


@flashinfer_api
def gvr_topk_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    enable_unroll_4: Optional[bool] = None,
    enable_phase3_unroll: Optional[bool] = None,
    use_constant_hint: bool = False,
    min_blocks_per_mp: Optional[int] = None,
    use_256bit_load: Optional[bool] = None,
    num_threads_per_block: Optional[int] = None,
    enable_warp_parallel_reduce: Optional[bool] = None,
    compress_ratio: int = 1,
    max_seq_len: Optional[int] = None,
    return_output_values: bool = False,
    cluster_size: int = 1,
    seqlen_sorted: bool = False,
    order_row: Optional[torch.Tensor] = None,
) -> tuple:
    r"""GVR (Guess-Verify-Refine) Top-K decode kernel for Blackwell GPUs.

    Selects the top-``top_k`` elements per row from ``logits`` using a
    histogram-based threshold search.  Designed for the decode phase where a
    compact ``pre_idx`` scan index is available from the prior step.

    This function requires an NVIDIA Blackwell GPU (sm_100 / B200) and the
    ``nvidia-cutlass-dsl`` package.

    Parameters
    ----------
    logits : torch.Tensor
        2-D float tensor of shape ``(num_rows, max_seq_len)`` in
        compressed-token-index space.  Supported dtypes: ``float32``,
        ``bfloat16``, ``float16``.
    pre_idx : torch.Tensor
        2-D ``int32`` tensor of shape ``(num_rows // next_n, top_k)``.
        ``pre_idx[..., 0]`` must contain the argmax index (indexer invariant).
    seq_lens : torch.Tensor
        1-D ``int32`` tensor of shape ``(num_rows // next_n,)`` holding the
        uncompressed token counts per request.  The kernel divides by
        ``compress_ratio`` internally.
    top_k : int
        Number of top elements to select per row.  Must be in ``{512, 1024, 2048}``.
    next_n : int, optional
        Temporal stride — used for V3.2 ``preIdxOffset = (row % next_n) + 1``.
        Default ``1``.
    out_values : torch.Tensor, optional
        Pre-allocated output values tensor of shape ``(num_rows, top_k)``
        and same dtype as ``logits``.  Allocated automatically when ``None``.
    out_indices : torch.Tensor, optional
        Pre-allocated output indices tensor of shape ``(num_rows, top_k)``
        with dtype ``int32``.  Allocated automatically when ``None``.
    num_sms : int, optional
        Number of SMs on the device.  Auto-detected when ``None``.
    enable_unroll_4 : bool, optional
        Unroll Phase-1 inner loop 4×.  Auto-selected when ``None``.
    enable_phase3_unroll : bool, optional
        Unroll Phase-3 candidate collect loop.  Auto-selected when ``None``.
    use_constant_hint : bool, optional
        Use ``__constant__`` memory for histogram hints.  Default ``False``.
    min_blocks_per_mp : int, optional
        ``__launch_bounds__`` min blocks per SM.  Auto-selected when ``None``.
    use_256bit_load : bool, optional
        Use 256-bit wide loads (requires ``float32`` + long rows).
        Auto-selected when ``None``.
    num_threads_per_block : int, optional
        CTA size (512 or 1024).  Auto-selected when ``None``.
    enable_warp_parallel_reduce : bool, optional
        Enable warp-parallel Phase-2 histogram reduction.
        Auto-selected when ``None``.
    compress_ratio : int, optional
        KV-indexer compression factor (1 for DSv3.2, 4 for DSv4).
        Default ``1``.
    max_seq_len : int, optional
        Graph-safe hint for the peak ``logits.shape[1]`` at CUDA Graph
        replay.  Used to drive heuristic without reading tensor metadata.
        Default ``None`` (uses ``logits.shape[1]`` at call time).
    return_output_values : bool, optional
        If ``True``, write top-k values into ``out_values`` and return them.
        If ``False``, only indices are returned (``out_values`` slot is
        ``None``).  Default ``False``.
    cluster_size : int, optional
        Number of CTAs per thread-block cluster for long rows.  Default ``1``
        (no clustering).  Set to 2 or 4 for very long rows (> 64 K).
    seqlen_sorted : bool, optional
        When ``True``, use LJF dispatch order from ``order_row``.
        Default ``False``.
    order_row : torch.Tensor, optional
        Required when ``seqlen_sorted=True``.  ``int32[batch_size]``
        produced by :func:`gvr_topk_sort_prepare`.

    Returns
    -------
    out_values : torch.Tensor or None
        Shape ``(num_rows, top_k)``, same dtype as ``logits``.
        ``None`` when ``return_output_values=False``.
    out_indices : torch.Tensor
        Shape ``(num_rows, top_k)``, dtype ``int32``.

    Raises
    ------
    RuntimeError
        If the device is not Blackwell (sm_100+) or CuTe DSL is unavailable.

    Examples
    --------
    >>> import torch, flashinfer
    >>> B, N, K = 4, 8192, 1024
    >>> logits   = torch.randn(B, N, dtype=torch.bfloat16, device="cuda")
    >>> pre_idx  = torch.topk(logits, K, dim=-1).indices.int()
    >>> seq_lens = torch.full((B,), N, dtype=torch.int32, device="cuda")
    >>> _, indices = flashinfer.gvr_topk_decode(logits, pre_idx, seq_lens, K)
    >>> indices.shape
    torch.Size([4, 1024])
    """
    if not can_use_gvr_topk(logits.device):
        raise RuntimeError(
            "gvr_topk_decode requires a Blackwell GPU (sm_100+) and the "
            "nvidia-cutlass-dsl package.  Use flashinfer.top_k() for other hardware."
        )
    assert logits.is_cuda, "logits must be on CUDA"
    assert logits.dim() == 2, f"logits must be 2D, got shape {logits.shape}"
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    if seqlen_sorted:
        assert (
            order_row is not None
            and order_row.dtype == torch.int32
            and order_row.is_cuda
            and order_row.shape == seq_lens.shape
        ), (
            "seqlen_sorted=True requires order_row: int32[batch_size] on CUDA "
            f"(expected shape {tuple(seq_lens.shape)}, got "
            f"{tuple(order_row.shape) if order_row is not None else None})"
        )

    cute_dtype = torch_to_cutlass_dtype(logits.dtype)

    if num_sms is None:
        num_sms = get_num_sm(logits.device)

    num_rows = logits.shape[0]
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)

    # Resolve None defaults via the same heuristic as the TRT-LLM production runner.
    if enable_unroll_4 is None:
        enable_unroll_4 = True
    if enable_phase3_unroll is None:
        enable_phase3_unroll = True

    N_cols = logits.shape[1]
    N_dec = max_seq_len if max_seq_len is not None else N_cols
    if num_threads_per_block is None:
        if max_seq_len is not None and logits.dtype != torch.float32:
            n_thresh_t = 131072
        else:
            n_thresh_t = 65536
        num_threads_per_block = 1024 if (num_rows <= num_sms and N_dec >= n_thresh_t) else 512
    if use_256bit_load is None:
        use_256bit_load = logits.dtype == torch.float32 and N_dec >= 16384
    if enable_warp_parallel_reduce is None:
        enable_warp_parallel_reduce = num_threads_per_block == 1024

    if min_blocks_per_mp is None:
        vec_bits_host = 256 if use_256bit_load else 128
        vec_w_host = vec_bits_host // (32 if logits.dtype == torch.float32 else 16)
        n_vec_iters = max(1, N_dec // (num_threads_per_block * vec_w_host))
        is_fp32 = logits.dtype == torch.float32
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and N_dec <= 32768:
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

    compiled = _compile_gvr(
        cute_dtype,
        top_k,
        next_n,
        enable_unroll_4,
        enable_phase3_unroll,
        use_constant_hint,
        min_blocks_per_mp,
        use_256bit_load,
        num_threads_per_block,
        enable_warp_parallel_reduce,
        compress_ratio,
        return_output_values,
        cluster_size,
        seqlen_sorted,
    )
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values if return_output_values else None,
        out_indices,
        order_row if seqlen_sorted else None,
    )
    return (out_values if return_output_values else None), out_indices


# ---------------------------------------------------------------------------
# Public API: gvr_topk_sort_prepare
# ---------------------------------------------------------------------------


@flashinfer_api
def gvr_topk_sort_prepare(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build the LJF dispatch order for :func:`gvr_topk_decode`.

    Parameters
    ----------
    seq_lens : torch.Tensor
        1-D ``int32`` tensor of shape ``(batch_size,)`` on CUDA.

    Returns
    -------
    order_row : torch.Tensor
        ``int32[batch_size]`` where entry ``i`` is the original-batch index
        of the i-th longest request (longest-first ordering).  Pass this to
        :func:`gvr_topk_decode` with ``seqlen_sorted=True``.
    """
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    return torch.argsort(seq_lens, descending=True, stable=False).to(torch.int32)


# ---------------------------------------------------------------------------
# Internal: LB compile caches
# ---------------------------------------------------------------------------


@functools.cache
def _compile_lb_prepare(
    num_threads: int,
    batch_size: int,
    long_threshold: int,
    compress_ratio: int,
):
    prep = GvrTopKLBPrepareKernel(
        long_threshold=long_threshold,
        compress_ratio=compress_ratio,
        num_threads=num_threads,
    )
    fake_seq = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch_size,), stride_order=(0,)
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_threads,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        prep,
        fake_seq,
        fake_order,
        fake_ctr,
        cutlass.Int32(0),
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_lb(
    cute_dtype,
    top_k: int,
    next_n: int,
    num_rows: int,
    N: int,
    compress_ratio: int,
    max_batch_size: int,
    num_threads: int,
    cluster_size: int,
    return_output_values: bool,
):
    kernel = GvrTopKLBKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        max_batch_size=max_batch_size,
    )
    n_groups = num_rows // next_n
    fake_logits = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (num_rows, N), stride_order=(1, 0), assumed_align=16
    )
    fake_pre_idx = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_groups, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_seq = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_groups,), stride_order=(0,))
    fake_out_v = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        if return_output_values
        else None
    )
    fake_out_i = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_batch_size,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        fake_logits,
        fake_pre_idx,
        fake_seq,
        fake_out_v,
        fake_out_i,
        fake_order,
        fake_ctr,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Public API: gvr_topk_lb_prepare / gvr_topk_lb_decode
# ---------------------------------------------------------------------------


@flashinfer_api
def gvr_topk_lb_prepare(
    seq_lens: torch.Tensor,
    max_batch_size: int = 1024,
    long_threshold: int = 64 * 1024,
    compress_ratio: int = 1,
    order_row: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> tuple:
    """Run the load-balance classify kernel (once per decode step).

    Partitions requests into *long* (multi-CTA cluster) and *short*
    (single-CTA) buckets based on ``seq_lens`` vs ``long_threshold``.
    The returned ``(order_row, counters)`` are reused by every per-layer
    :func:`gvr_topk_lb_decode` call within the same decode step.

    Parameters
    ----------
    seq_lens : torch.Tensor
        1-D ``int32`` tensor of shape ``(batch_size,)`` on CUDA.
    max_batch_size : int, optional
        Block size for the prepare kernel and length of ``order_row``.
        Must be a power of 2 in ``[64, 1024]``.  Default ``1024``.
    long_threshold : int, optional
        Threshold (in scan-length = ``seq_lens / compress_ratio`` space)
        above which a request is classified as *long*.  Default ``65536``.
    compress_ratio : int, optional
        KV-indexer compression factor.  Default ``1``.
    order_row : torch.Tensor, optional
        Pre-allocated ``int32[max_batch_size]`` buffer to receive the
        dispatch order.  Allocated automatically when ``None``.
    counters : torch.Tensor, optional
        Pre-allocated ``int32[2]`` buffer for ``(n_long, n_short)``.
        Allocated automatically when ``None``.

    Returns
    -------
    order_row : torch.Tensor
        ``int32[max_batch_size]`` — first ``n_long`` entries are long
        request indices, next ``n_short`` are short indices.
    counters : torch.Tensor
        ``int32[2]`` — ``(n_long, n_short)``.
    """
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    if not (64 <= max_batch_size <= 1024) or (max_batch_size & (max_batch_size - 1)) != 0:
        raise ValueError(
            f"max_batch_size must be a power of 2 in [64, 1024]; got {max_batch_size}"
        )
    batch_size = seq_lens.shape[0]
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= max_batch_size ({max_batch_size})"
        )
    if order_row is None:
        order_row = torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device=seq_lens.device
        )
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=seq_lens.device)

    compiled = _compile_lb_prepare(max_batch_size, batch_size, long_threshold, compress_ratio)
    compiled(seq_lens, order_row, counters, cutlass.Int32(batch_size))
    return order_row, counters


@flashinfer_api
def gvr_topk_lb_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    order_row: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    cluster_size: int = 4,
    max_batch_size: int = 1024,
    num_threads: int = 512,
    return_output_values: bool = False,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> tuple:
    """Run the load-balance hybrid (multi-CTA cluster + single-CTA) main kernel.

    ``order_row`` and ``counters`` **must** be populated by a prior call to
    :func:`gvr_topk_lb_prepare`.  They are invariant across per-layer Top-K
    calls within the same decode step so callers run prepare once and reuse.

    Parameters
    ----------
    logits : torch.Tensor
        2-D float tensor ``(num_rows, max_seq_len)``.
    pre_idx : torch.Tensor
        2-D ``int32`` tensor ``(num_rows // next_n, top_k)``.
    seq_lens : torch.Tensor
        1-D ``int32`` tensor ``(num_rows // next_n,)``.
    order_row : torch.Tensor
        From :func:`gvr_topk_lb_prepare`.
    counters : torch.Tensor
        From :func:`gvr_topk_lb_prepare`.
    top_k : int
        Number of top elements per row.
    next_n : int, optional
        Temporal stride.  Default ``1``.
    compress_ratio : int, optional
        KV-indexer compression factor.  Default ``1``.
    cluster_size : int, optional
        Cluster size for long rows.  Default ``4``.
    max_batch_size : int, optional
        Must match ``max_batch_size`` passed to :func:`gvr_topk_lb_prepare`.
        Default ``1024``.
    num_threads : int, optional
        CTA size for single-CTA short rows.  Default ``512``.
    return_output_values : bool, optional
        Return top-k values alongside indices.  Default ``False``.
    out_values : torch.Tensor, optional
        Pre-allocated output values.  Allocated when ``None`` and
        ``return_output_values=True``.
    out_indices : torch.Tensor, optional
        Pre-allocated output indices.  Allocated when ``None``.

    Returns
    -------
    out_values : torch.Tensor or None
    out_indices : torch.Tensor
    """
    assert logits.is_cuda and logits.dim() == 2
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    cute_dtype = torch_to_cutlass_dtype(logits.dtype)

    num_rows = logits.shape[0]
    N = logits.shape[1]
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if not return_output_values:
        out_values = None

    compiled = _compile_lb(
        cute_dtype,
        top_k,
        next_n,
        num_rows,
        N,
        compress_ratio,
        max_batch_size,
        num_threads,
        cluster_size,
        return_output_values,
    )
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values,
        out_indices,
        order_row,
        counters,
    )
    return out_values, out_indices
