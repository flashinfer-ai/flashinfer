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
    next_n=1, return_values=False, out_indices=None, out_values=None, backend="auto",
):
    """Radix masked-fallback: runs on all supported SM tiers."""
    return True


@supported_compute_capability(_BLACKWELL_PLUS_CCS)
def _gvr_top_k_decode_check(
    logits, seq_lens, top_k, pre_idx=None, compress_ratio=1,
    next_n=1, return_values=False, out_indices=None, out_values=None, backend="auto",
):
    """GVR LB: requires Blackwell hardware, CuTe DSL, and a pre_idx hint."""
    return is_cute_dsl_available() and pre_idx is not None


def _top_k_decode_heuristic(suitable_backends, **kwargs):
    """Prefer GVR over radix when both are available."""
    return [b for b in ("gvr", "radix") if b in suitable_backends]


# ---------------------------------------------------------------------------
# Internal: compiled-kernel cache
# ---------------------------------------------------------------------------

if is_cute_dsl_available():
    from .cute_dsl.top_k import GvrTopKLBKernel, GvrTopKLBPrepareKernel


@functools.cache
def _compile_lb_prepare(
    num_threads: int, batch_size: int, long_threshold: int, compress_ratio: int
):
    prep = GvrTopKLBPrepareKernel(
        long_threshold=long_threshold,
        compress_ratio=compress_ratio,
        num_threads=num_threads,
    )
    return cute.compile(
        prep,
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (batch_size,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_threads,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,)),
        cutlass.Int32(0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_lb(
    cute_dtype, top_k, next_n, num_rows, N, compress_ratio,
    max_batch_size, num_threads, cluster_size, return_output_values,
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
    return cute.compile(
        kernel,
        cute.runtime.make_fake_compact_tensor(cute_dtype, (num_rows, N), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_groups, top_k), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_groups,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cute_dtype, (num_rows, top_k), stride_order=(1, 0), assumed_align=16) if return_output_values else None,
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (max_batch_size,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,)),
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
    N = logits.shape[1]
    batch_size = seq_lens.shape[0]
    max_batch_size = _lb_max_batch_size(batch_size)
    lb_cfg = GvrTopKLBConfig(max_batch_size=max_batch_size)

    order_row = torch.empty(max_batch_size, dtype=torch.int32, device=logits.device)
    counters = torch.zeros(2, dtype=torch.int32, device=logits.device)
    _compile_lb_prepare(max_batch_size, batch_size, lb_cfg.long_threshold, compress_ratio)(
        seq_lens, order_row, counters, cutlass.Int32(batch_size)
    )

    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)

    _compile_lb(
        cute_dtype, top_k, next_n, num_rows, N, compress_ratio,
        max_batch_size, lb_cfg.num_threads, lb_cfg.cluster_size, return_output_values,
    )(
        logits, pre_idx, seq_lens,
        out_values if return_output_values else None,
        out_indices, order_row, counters,
    )
    return (out_values if return_output_values else None), out_indices


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
    >>> B, N, K = 32, 8192, 1024
    >>> logits  = torch.randn(B, N, dtype=torch.bfloat16, device="cuda")
    >>> pre_idx = torch.topk(logits, K, dim=-1).indices.int()
    >>> seq_lens = torch.full((B,), N, dtype=torch.int32, device="cuda")
    >>> # auto-selects GVR on Blackwell, radix otherwise
    >>> indices = flashinfer.top_k_decode(logits, seq_lens, K, pre_idx=pre_idx)
    >>> # explicit radix backend (any GPU, no pre_idx needed)
    >>> indices = flashinfer.top_k_decode(logits, seq_lens, K, backend="radix")

    See Also
    --------
    flashinfer.top_k : General-purpose radix/clusters top-K (uniform lengths).
    """
    assert logits.is_cuda and logits.dim() == 2, "logits must be a 2-D CUDA tensor"
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    if backend == "auto":
        backend = top_k_decode.suitable_auto_backends[0]

    if backend == "gvr":
        out_v, out_i = _run_gvr(
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
