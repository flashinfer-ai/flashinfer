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
"""Top-K decode: GVR (Blackwell) and radix-masked (all SM) backends.

Public API
----------
:func:`top_k_decode` — selects top-K per row of decode-step logits.

Backend choices
---------------
``"gvr"``   — GVR (Guess-Verify-Refine) load-balance kernel.
              Requires Blackwell (sm_100+), nvidia-cutlass-dsl, and a
              ``pre_idx`` hint from the previous decode step.
``"radix"`` — masked-radix fallback; masks logits to ``seq_lens`` then
              calls the existing FlashInfer radix top-K.  Runs on any GPU.
``"auto"``  — picks GVR when all its requirements are met, falls back to
              radix otherwise.
"""

import functools
from typing import Literal, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch

from .api_logging import flashinfer_api
from .cute_dsl.top_k.config import GvrTopKLBConfig
from .cute_dsl.utils import (
    is_cute_dsl_available,
    torch_to_cutlass_dtype,
)
from .topk import get_topk_module
from .utils import (
    _get_cache_buf,
    backend_requirement,
    get_compute_capability,
    supported_compute_capability,
)

# ---------------------------------------------------------------------------
# Supported compute-capability sets (major * 10 + minor)
# ---------------------------------------------------------------------------

# All SM tiers FlashInfer ships kernels for.
_ALL_CCS = [75, 80, 86, 89, 90, 100, 103, 110, 120, 121]

# GVR requires Blackwell sm_100a or newer.
_BLACKWELL_PLUS_CCS = [100, 103, 110, 120, 121]

# ---------------------------------------------------------------------------
# Backend requirement checkers
# ---------------------------------------------------------------------------


@supported_compute_capability(_ALL_CCS)
def _radix_top_k_decode_check(
    logits, seq_lens, top_k, pre_idx=None, compress_ratio=1,
    next_n=1, return_values=False, out_indices=None, out_values=None,
    backend="auto", load_balance=True, num_long_rows=None,
):  # noqa: extra kwargs mirror the public signature; unused by the check
    """Radix masked-fallback: runs on all supported SM tiers."""
    return True


@supported_compute_capability(_BLACKWELL_PLUS_CCS)
def _gvr_top_k_decode_check(
    logits, seq_lens, top_k, pre_idx=None, compress_ratio=1,
    next_n=1, return_values=False, out_indices=None, out_values=None,
    backend="auto", load_balance=True, num_long_rows=None,
):  # noqa: extra kwargs mirror the public signature; unused by the check
    """GVR LB: requires Blackwell hardware, CuTe DSL, and a pre_idx hint."""
    return is_cute_dsl_available() and pre_idx is not None


def _top_k_decode_heuristic(suitable_backends, **kwargs):
    """Prefer GVR over radix when both are available."""
    return [b for b in ("gvr", "radix") if b in suitable_backends]


# ---------------------------------------------------------------------------
# Internal: compiled-kernel cache
# ---------------------------------------------------------------------------

if is_cute_dsl_available():
    from .cute_dsl.top_k import GvrTopKKernel, GvrTopKLBKernel, GvrTopKLBPrepareKernel


@functools.cache
def _compile_lb_prepare(num_threads: int, long_threshold: int, compress_ratio: int):
    # batch_size (the seq_lens length) is DYNAMIC: it is passed as a runtime scalar
    # (cutlass.Int32(batch_size)) and only bounds the classifier's per-thread guard
    # (tidx < batch_size); it is never a const_expr / SMEM size / static unroll, and
    # the grid is fixed (1,1,1). So seq_lens is compiled with a symbolic length and
    # one kernel serves every batch size. num_threads stays static (it sizes the
    # order_row buffer and the block-prefix-sum SMEM); counters is constant (2,).
    prep = GvrTopKLBPrepareKernel(
        long_threshold=long_threshold,
        compress_ratio=compress_ratio,
        num_threads=num_threads,
    )
    sym_batch = cute.sym_int()
    return cute.compile(
        prep,
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_batch,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_threads,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,)),
        cutlass.Int32(0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_lb(
    cute_dtype, top_k, next_n, compress_ratio,
    max_batch_size, num_threads, cluster_size, return_output_values,
):
    # num_rows and N are DYNAMIC (symbolic) — see _compile_gvr. The LB kernel's
    # grid is fixed at max_batch_size * next_n * cluster_size (surplus clusters
    # early-exit via counters), so num_rows never touches the launch config, and
    # N is a per-row runtime bound. Only max_batch_size stays static (it sizes the
    # grid and the order_row buffer). One compiled kernel serves every batch size
    # and sequence length for a given (dtype, top_k, next_n, compress_ratio,
    # max_batch_size, num_threads, cluster_size, return_output_values).
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
    sym_groups = cute.sym_int()  # request count (= num_rows // next_n)
    sym_n = cute.sym_int()  # per-row logits width
    sym_rows = sym_groups * next_n
    return cute.compile(
        kernel,
        cute.runtime.make_fake_compact_tensor(cute_dtype, (sym_rows, sym_n), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_groups, top_k), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_groups,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cute_dtype, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16) if return_output_values else None,
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (max_batch_size,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,)),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_gvr(
    cute_dtype, top_k, next_n, compress_ratio,
    num_threads, return_output_values,
):
    # num_rows and N (per-row width) are DYNAMIC dimensions: the kernel reads
    # them from the tensor shapes at runtime (num_rows = input.shape[0] drives the
    # grid in __call__; N is derived per-row from seq_lens) and never uses them in
    # a const_expr / SMEM sizing / static unroll. So they are compiled symbolically
    # via cute.sym_int() — one compiled kernel serves every batch size and
    # sequence length, matching the FlashInfer CuTe-DSL convention (cf.
    # rmsnorm_fp4quant._get_compiled_kernel). Only true specializations (dtype,
    # top_k, next_n, compress_ratio, num_threads, return_output_values) key the
    # cache. sym_groups is the request axis; num_rows = sym_groups * next_n.
    kernel = GvrTopKKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=1,
    )
    sym_groups = cute.sym_int()  # request count (= num_rows // next_n)
    sym_n = cute.sym_int()  # per-row logits width
    sym_rows = sym_groups * next_n
    return cute.compile(
        kernel,
        cute.runtime.make_fake_compact_tensor(cute_dtype, (sym_rows, sym_n), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_groups, top_k), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_groups,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cute_dtype, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16) if return_output_values else None,
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16),
        None,  # order_row unused when seqlen_sorted=False
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Internal: GVR backend implementation
# ---------------------------------------------------------------------------


def _lb_max_batch_size(batch_size: int) -> int:
    for cap in (64, 128, 256, 512, 1024):
        if batch_size <= cap:
            return cap
    raise ValueError(f"batch_size {batch_size} exceeds maximum supported 1024")


def _run_gvr(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run GVR LB prepare + decode in one call (mirrors topk_clusters_exact)."""
    cute_dtype = torch_to_cutlass_dtype(logits.dtype)
    num_rows = logits.shape[0]
    batch_size = seq_lens.shape[0]
    max_batch_size = _lb_max_batch_size(batch_size)
    lb_cfg = GvrTopKLBConfig(max_batch_size=max_batch_size)

    order_row = torch.empty(max_batch_size, dtype=torch.int32, device=logits.device)
    counters = torch.zeros(2, dtype=torch.int32, device=logits.device)
    _compile_lb_prepare(max_batch_size, lb_cfg.long_threshold, compress_ratio)(
        seq_lens, order_row, counters, cutlass.Int32(batch_size)
    )

    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)

    _compile_lb(
        cute_dtype, top_k, next_n, compress_ratio,
        max_batch_size, lb_cfg.num_threads, lb_cfg.cluster_size, return_output_values,
    )(
        logits, pre_idx, seq_lens,
        out_values if return_output_values else None,
        out_indices, order_row, counters,
    )
    return (out_values if return_output_values else None), out_indices


def _run_gvr_no_lb(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run GVR without load-balancing: one CTA per row, no prepare kernel."""
    cute_dtype = torch_to_cutlass_dtype(logits.dtype)
    num_rows = logits.shape[0]

    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)

    lb_cfg = GvrTopKLBConfig()
    _compile_gvr(
        cute_dtype, top_k, next_n, compress_ratio,
        lb_cfg.num_threads, return_output_values,
    )(
        logits, pre_idx, seq_lens,
        out_values if return_output_values else None,
        out_indices, None,
    )
    return (out_values if return_output_values else None), out_indices


def _lb_decision_from_counts(n_long: int, num_rows: int) -> bool:
    """Pure-host LB decision: is the two-kernel LB path worth its overhead?

    Load-balancing splits *long* rows (scan-length > ``long_threshold``) across a
    CTA cluster to shorten the single-wave tail, and packs short rows one-per-CTA.
    It pays off only when the batch is a *mix* of long and non-long rows:

      * No long rows                -> LB adds a prepare kernel + cluster sync for
                                       no tail to cut  => non-LB is faster.
      * All (or a majority of) rows  -> every row already saturates an SM; there
        are long                      are no short rows to pack and the cluster
                                       split just adds DSMEM-sync overhead
                                       => non-LB is faster.
      * A minority (<= half) of rows -> the long rows form a tail that LB cuts
        are long                       ~cluster_size-fold while short rows finish
                                       cheaply  => LB is faster.

    LB is selected iff ``0 < n_long <= num_rows // 2``. Boundary tuned on B200: LB
    wins up to ~50% long-fraction, loses by ~62% (see ``benchmarks/bench_gvr_lb.py``).

    This function takes plain Python ints, so it does no device I/O and is
    **CUDA-graph safe** — the branch resolves at trace time. Callers who know their
    batch's long-row count (they usually track context lengths on the host anyway)
    should pass it via ``top_k_decode(..., num_long_rows=...)``.
    """
    return 0 < n_long <= num_rows // 2


def _count_long_rows(
    seq_lens: torch.Tensor, compress_ratio: int, long_threshold: int
) -> int:
    """Count rows whose scan length exceeds ``long_threshold`` (device reduction).

    Compared in scan-length space (``seq_lens / compress_ratio``), matching the GVR
    prepare kernel. Threshold-scaling avoids the elementwise divide when
    ``compress_ratio == 1`` (the common DSv3.2 case).

    Convenience helper for callers who want to derive the ``num_long_rows`` hint
    for ``top_k_decode(load_balance="auto", ...)`` from a device ``seq_lens``
    tensor. The ``.item()`` read forces a device->host sync (~a few us) and is
    **NOT CUDA-graph safe**, so ``top_k_decode`` never calls it internally — under
    graph capture, compute ``num_long_rows`` on the host from data you already
    track, or use an explicit ``load_balance=True/False``.
    """
    threshold = long_threshold if compress_ratio == 1 else long_threshold * compress_ratio
    return int((seq_lens > threshold).sum().item())


# ---------------------------------------------------------------------------
# Internal: radix backend implementation
# ---------------------------------------------------------------------------


def _run_radix(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Masked-radix fallback: mask logits to seq_lens, then call the radix kernel directly."""
    num_rows, N = logits.shape
    if next_n > 1:
        row_seq_lens = seq_lens.repeat_interleave(next_n)
        row_offsets = torch.arange(next_n, device=logits.device, dtype=torch.int32).repeat(
            seq_lens.shape[0]
        )
        row_seq_lens = (row_seq_lens - next_n + row_offsets + 1) // compress_ratio
    else:
        row_seq_lens = (seq_lens // compress_ratio).clamp(max=N)

    col_idx = torch.arange(N, device=logits.device).unsqueeze(0)
    masked_logits = logits.masked_fill(col_idx >= row_seq_lens.unsqueeze(1), float("-inf"))

    row_states_buffer = _get_cache_buf(
        f"radix_topk_row_states_{logits.device}", 1024 * 1024, logits.device, zero_init=True
    )
    if out_values is None:
        out_values = torch.empty(num_rows, top_k, dtype=logits.dtype, device=logits.device)
    out_i_int32 = get_topk_module().radix_topk(
        masked_logits, top_k,
        False,  # sorted
        False,  # deterministic
        0,      # tie_break = TopKTieBreak.NONE
        row_states_buffer,
        out_values,
        False,  # dsa_graph_safe
    )

    out_i = out_i_int32.to(torch.int32)
    if out_indices is not None:
        out_indices.copy_(out_i)
        out_i = out_indices

    return (out_values if return_output_values else None), out_i


# ---------------------------------------------------------------------------
# Public API: top_k_decode
# ---------------------------------------------------------------------------


@backend_requirement(
    {
        "radix": _radix_top_k_decode_check,
        "gvr": _gvr_top_k_decode_check,
    },
    heuristic_func=_top_k_decode_heuristic,
)
@flashinfer_api
def top_k_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    pre_idx: Optional[torch.Tensor] = None,
    compress_ratio: int = 1,
    next_n: int = 1,
    return_values: bool = False,
    out_indices: Optional[torch.Tensor] = None,
    out_values: Optional[torch.Tensor] = None,
    backend: Literal["radix", "gvr", "auto"] = "auto",
    load_balance: Union[bool, Literal["auto"]] = True,
    num_long_rows: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Top-K selection over batched decode-step logits.

    Selects the top-``top_k`` elements from each row of ``logits``,
    respecting per-request KV-cache lengths given by ``seq_lens``.

    Backend selection
    -----------------
    ``backend="auto"`` (default) chooses GVR when available (Blackwell +
    ``pre_idx`` supplied), otherwise falls back to radix.  Force a specific
    backend with ``backend="gvr"`` or ``backend="radix"``.

    Parameters
    ----------
    logits : torch.Tensor
        2-D float tensor of shape ``(num_rows, max_seq_len)``.
        Supported dtypes: ``float32``, ``bfloat16``, ``float16``.
        For the ``"gvr"`` backend the row width ``max_seq_len`` must be a
        multiple of ``16 // itemsize`` (8 for fp16/bf16, 4 for fp32) so each
        row is 16-byte aligned for GVR's 128-bit vectorized loads; a
        ``ValueError`` is raised otherwise.  The ``"radix"`` backend has no
        such constraint.
    seq_lens : torch.Tensor
        1-D ``int32`` tensor of shape ``(num_rows // next_n,)`` with the
        effective KV-cache length per request.  Logits at or beyond
        ``seq_lens[i]`` are excluded from the search.
    top_k : int
        Number of top elements per row.  GVR backend supports
        ``{512, 1024, 2048}``; radix backend has no restriction.
    pre_idx : torch.Tensor, optional
        ``int32[num_rows // next_n, top_k]`` — top-K KV-cache indices
        selected by **this same layer** at the **previous token's decode
        step**.  GVR exploits the strong correlation between a layer's
        attention pattern at step ``t`` and step ``t+1``; the kernel
        internally applies a ``+1`` offset (DSv3.2) so the previous step's
        indices land correctly in the current step's grown KV-cache space.
        ``pre_idx[:, 0]`` must be the argmax index.
        Required by the ``"gvr"`` backend; ignored by ``"radix"``.
    compress_ratio : int, optional
        KV-index compression factor (``1`` for DSv3.2, ``4`` for DSv4).
        Default ``1``.
    next_n : int, optional
        Speculative-decode temporal stride.  Default ``1``.
    return_values : bool, optional
        When ``True`` also return the selected logit values.
        Default ``False``.
    out_indices : torch.Tensor, optional
        Pre-allocated ``int32[num_rows, top_k]`` output buffer.
    out_values : torch.Tensor, optional
        Pre-allocated values buffer (same dtype as ``logits``).
        Only used when ``return_values=True``.
    backend : {"radix", "gvr", "auto"}, optional
        Backend to use.  Default ``"auto"``.

        ``"gvr"``   — GVR LB kernel (Blackwell sm_100+ only; requires
                      ``pre_idx``).
        ``"radix"`` — Masked radix top-K (all GPUs, no ``pre_idx`` needed).
        ``"auto"``  — GVR when requirements are met, else radix.
    load_balance : bool or ``"auto"``, optional
        Selects the GVR kernel path (ignored by the radix backend).  Default
        ``True``.

        ``True`` (default) — two-kernel LB path (``GvrTopKLBPrepareKernel`` +
                     ``GvrTopKLBKernel``): a prepare kernel classifies requests
                     into long/short buckets, then the main kernel splits each
                     long row across a CTA cluster and packs short rows.  Best
                     for the ragged decode batches GVR targets.
        ``False``  — single-kernel path (``GvrTopKKernel``): one CTA per row,
                     no prepare step.  Faster when the batch has no length
                     variance (all rows short, or all long).
        ``"auto"`` — pick per call with the heuristic: use LB only when the batch
                     is a *mix* of long and non-long rows (long rows form a
                     minority tail worth splitting; see :func:`_lb_decision_from_counts`).
                     **Requires** ``num_long_rows`` — the decision is made purely
                     on the host from that count, with no device read.  Omitting
                     it raises ``ValueError`` (deriving the count from ``seq_lens``
                     would need a device->host sync that is not CUDA-graph safe).

        All three settings are CUDA-graph safe (no host branch on device data).
    num_long_rows : int, optional
        Host-side hint: the number of rows whose scan length
        (``seq_lens / compress_ratio``) exceeds the GVR ``long_threshold`` (64K).
        **Required** when ``load_balance="auto"`` (ignored otherwise).  The LB
        decision is a pure-host branch on this count, keeping ``"auto"`` CUDA-graph
        safe and sync-free.  Callers typically already track context lengths on the
        host; :func:`_count_long_rows` can derive it from ``seq_lens`` in eager mode
        (at the cost of a device sync) if needed.

    Returns
    -------
    indices : torch.Tensor
        ``int32[num_rows, top_k]``.
    (indices, values) : Tuple[torch.Tensor, torch.Tensor]
        When ``return_values=True``.

    Raises
    ------
    BackendSupportedError
        If the requested backend is not supported on the current device or
        the required inputs (e.g. ``pre_idx``) are missing.

    Examples
    --------
    >>> import torch, flashinfer
    >>> torch.manual_seed(42)
    >>> B, N_max, top_k = 32, 8192, 1024
    >>>
    >>> # Step t: no prior indices; use radix to get the first top-K.
    >>> # Each request has a different KV-cache length in [top_k+1, N_max-1].
    >>> logits = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
    >>> seq_lens_t = torch.randint(top_k + 1, N_max, (B,), dtype=torch.int32, device="cuda")
    >>> indices_t = flashinfer.top_k_decode(logits, seq_lens_t, top_k, backend="radix")
    >>> # Reference check: every selected value must be >= the K-th largest.
    >>> for i in range(B):
    ...     s = seq_lens_t[i].item()
    ...     kth = torch.topk(logits[i, :s].float(), top_k).values[-1]
    ...     assert (logits[i, :s].float()[indices_t[i].long()] < kth - 1e-5).sum() == 0
    >>>
    >>> # Step t+1: one new token appended per request; seq_lens grows by 1.
    >>> logits_t1 = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
    >>> seq_lens_t1 = seq_lens_t + 1
    >>> # Pass indices_t as pre_idx; GVR uses it to warm-start the threshold search.
    >>> indices_t1 = flashinfer.top_k_decode(logits_t1, seq_lens_t1, top_k, pre_idx=indices_t)
    >>> for i in range(B):
    ...     s = seq_lens_t1[i].item()
    ...     kth = torch.topk(logits_t1[i, :s].float(), top_k).values[-1]
    ...     assert (logits_t1[i, :s].float()[indices_t1[i].long()] < kth - 1e-5).sum() == 0

    See Also
    --------
    flashinfer.top_k : General-purpose radix/clusters top-K (uniform lengths).
    """
    assert logits.is_cuda and logits.dim() == 2, "logits must be a 2-D CUDA tensor"
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    if backend == "auto":
        backend = top_k_decode.suitable_auto_backends[0]

    if backend == "gvr":
        # GVR scans each row with 128-bit vectorized loads (vec_align = 16 bytes),
        # so every row must start at a 16-byte boundary. Row r begins at byte
        # r * N * itemsize, hence N * itemsize must be a multiple of 16 -- i.e. N
        # must be a multiple of 16 // itemsize (8 for fp16/bf16, 4 for fp32).
        # Otherwise the kernel issues a misaligned LDG and the launch faults with a
        # cryptic CUDA "misaligned address" error, so validate it up front.
        N = logits.shape[1]
        elem_align = 16 // logits.element_size()
        if N % elem_align != 0:
            raise ValueError(
                f"GVR backend requires the logits row width (N={N}) to be a "
                f"multiple of {elem_align} for {logits.dtype} (128-bit aligned "
                f"vectorized loads). Pad the logits buffer's last dimension up to a "
                f"multiple of {elem_align}, or use backend='radix' (no alignment "
                f"constraint)."
            )
        if load_balance == "auto":
            # "auto" requires the host-side long-row count. Deriving it from
            # seq_lens would need a device->host sync (.item()), which is not
            # CUDA-graph safe and adds latency, so it is disallowed: callers must
            # supply num_long_rows (pure-host, graph-safe) or pick a path
            # explicitly with load_balance=True/False.
            if num_long_rows is None:
                raise ValueError(
                    "load_balance='auto' requires num_long_rows (the count of rows "
                    "whose scan length exceeds the GVR long_threshold). Computing it "
                    "from seq_lens needs a device->host sync that is not CUDA-graph "
                    "safe. Pass num_long_rows, or set load_balance=True/False."
                )
            use_lb = _lb_decision_from_counts(int(num_long_rows), seq_lens.shape[0])
        else:
            use_lb = bool(load_balance)
        if use_lb:
            out_v, out_i = _run_gvr(
                logits, pre_idx, seq_lens, top_k, next_n, compress_ratio,
                return_values, out_indices, out_values,
            )
        else:
            out_v, out_i = _run_gvr_no_lb(
                logits, pre_idx, seq_lens, top_k, next_n, compress_ratio,
                return_values, out_indices, out_values,
            )
    else:  # "radix"
        out_v, out_i = _run_radix(
            logits, seq_lens, top_k, next_n, compress_ratio,
            return_values, out_indices, out_values,
        )

    if return_values:
        return out_i, out_v
    return out_i
