# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 paged index selection. BF16 inputs and int32 block identifiers."""

import functools
import json
import os
from importlib import resources
from typing import Any
import torch
from ..autotuner import AutoTuner
from ..api_logging import flashinfer_api
from ..trace.templates.msa import msa_index_decode_trace
from ..jit.msa_index_decode import gen_msa_index_decode_module

from ..jit.core import logger as _LOG

_MODULES: dict[int, Any] = {}
_ERRORS: dict[int, str] = {}
_CHAIN_READY: set[tuple] = set()
_COUNTS = {
    "dispatch_count": 0,
    "fallback_count": 0,
    "capture_unready": 0,
    "direct_dispatch_count": 0,
    "chain_dispatch_count": 0,
}


@functools.cache
def _allowlist():
    """Load the packaged geometry limits once per process."""
    return json.loads(
        resources.files("flashinfer.msa_ops")
        .joinpath("msa_index_decode_workloads.json")
        .read_text()
    )


def _msa_index_decode_stats():
    """Return dispatch counts and per-device preparation diagnostics."""
    return dict(
        _COUNTS,
        compiled_variants=len(_MODULES),
        distinct_kernels_for_allowlist=4,
        precompiled=bool(_MODULES) and bool(_CHAIN_READY),
        prepared_chain_signatures=len(_CHAIN_READY),
        precompile_errors=dict(_ERRORS),
    )


def _supports_geometry(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk=16,
    init_blocks=0,
    local_blocks=1,
    num_kv_heads=1,
    decode_query_len=1,
    max_decode_query_len=1,
    out=None,
    score_out=None,
):
    """Check specialization limits without reading device tensor contents."""
    b = idx_q.shape[0]
    return (
        os.environ.get("FLASHINFER_SPECIALIZED_KERNEL_DISABLE") != "1"
        and not AutoTuner.get().is_tuning_mode
        and score_out is None
        and b in _allowlist()["batch"]
        and idx_q.shape == (b, 1, 128)
        and idx_q.is_cuda
        and idx_q.dtype == torch.bfloat16
        and index_kv_cache.ndim == 3
        and index_kv_cache.shape[1:] == (128, 128)
        and index_kv_cache.dtype == torch.bfloat16
        and block_table.shape == (b, 128)
        and block_table.dtype == torch.int32
        and seq_lens.shape == (b,)
        and seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
        and block_table.stride(1) == 1
        and idx_q.stride(-1) == index_kv_cache.stride(-1) == 1
        and all(
            t.device == idx_q.device for t in (index_kv_cache, block_table, seq_lens)
        )
        and (
            topk,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
        )
        == (16, 0, 1, 1, 1, 1)
        and 0 < max_seq_len <= _allowlist()["max_seq_len_bound"]
        and (
            out is None
            or (
                out.ndim == 3
                and out.shape[0] == 1
                and out.shape[1] >= b
                and out.shape[2] == 16
                and out.stride(2) == 1
                and out.dtype == torch.int32
                and out.device == idx_q.device
            )
        )
    )


def msa_index_decode_supported(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk=16,
    init_blocks=0,
    local_blocks=1,
    num_kv_heads=1,
    decode_query_len=1,
    max_decode_query_len=1,
    out=None,
    score_out=None,
):
    """Whether the specialized indexer accepts this call on its current device.

    Consumers keep their own stock implementation when this returns False.
    During capture, a device without a prepared module returns False without
    querying device properties or compiling.
    """
    if not _supports_geometry(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out,
        score_out,
    ):
        return False
    device = idx_q.device.index
    if device in _MODULES:
        return True
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing() or device in _ERRORS:
            return False
        return torch.cuda.get_device_capability(device) == (9, 0)


def msa_index_decode_warmup(*args, **kwargs):
    """Prepare the current layouts before graph capture and return their result.

    Call this on the same input and output layouts as subsequent graph calls.
    Preparation covers the conservative 16384 sequence bound used by FULL
    graphs, even when the eager call uses a shorter host bound.
    """
    idx_q = args[0] if args else kwargs["idx_q"]
    with torch.cuda.device(idx_q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MiniMax indexer warmup must run before CUDA capture")
        return msa_index_decode(*args, **kwargs)


@flashinfer_api(trace=msa_index_decode_trace)
def msa_index_decode(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk=16,
    init_blocks=0,
    local_blocks=1,
    num_kv_heads=1,
    decode_query_len=1,
    max_decode_query_len=1,
    out=None,
    score_out=None,
):
    """Return [heads, queries, topk] int32 logical block IDs with trailing -1.

    Hopper accelerates single-token decode for batch 8/16, one head of width
    128, 128-token pages and a 128-column block table, through a host maximum
    sequence-length bound of 16384. Runtime short rows skip scoring and sorting
    even under a full CUDA graph; longer rows preserve the stock algorithms.
    Other signatures use
    the composable Triton implementation. Input lengths stay on the device;
    max_seq_len must be a valid upper bound on those lengths. The output
    buffer can contain additional query rows and keeps its stable address.
    FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 selects the stock implementation.

    Parameters
    ----------
    idx_q : torch.Tensor
        BF16 index queries of shape [total_q, num_kv_heads, head_dim] on CUDA.
    index_kv_cache : torch.Tensor
        BF16 paged index keys of shape [num_pages, 128, head_dim].
    block_table : torch.Tensor
        Int32 physical page IDs of shape [num_requests, table_width].
    seq_lens : torch.Tensor
        Int32 live context lengths of shape [num_requests] on the same device.
    max_seq_len : int
        Positive host upper bound on every live context length. The block table
        must cover this bound; lengths remain on the device during graph replay.
    topk : int, optional
        Maximum number of selected logical blocks per query. Default is 16.
    init_blocks : int, optional
        Number of initial blocks forced into the selection. Default is 0.
    local_blocks : int, optional
        Number of recent blocks forced into the selection. Default is 1.
    num_kv_heads : int, optional
        Number of index/KV heads, equal to the query head count. Default is 1.
    decode_query_len : int, optional
        Query tokens per request; total_q = num_requests * decode_query_len.
        Default is 1.
    max_decode_query_len : int, optional
        Compile-time bound on query tokens per request. Default is 1.
    out : torch.Tensor, optional
        Caller-owned int32 output of shape [num_kv_heads, >=total_q, topk].
        Only the first total_q rows are used; their address stays stable.
    score_out : torch.Tensor, optional
        Caller-owned float32 score buffer of shape
        [num_kv_heads, total_q, >=ceil(max_seq_len / 128)]. Supplying it
        selects the stock path, which writes scores using the buffer's strides.

    Returns
    -------
    torch.Tensor
        Int32 logical block IDs of shape [num_kv_heads, total_q, topk],
        with a valid selected prefix followed by -1 padding. Selection
        order within the prefix is unspecified. When out is supplied,
        returns its used view.

    Notes
    -----
    All tensors must be on the query's CUDA device. Call
    msa_index_decode_warmup on the actual layouts before graph
    capture to prepare the specialized kernels.
    """
    b = idx_q.shape[0]
    capturing = False
    if idx_q.is_cuda:
        with torch.cuda.device(idx_q.device):
            capturing = torch.cuda.is_current_stream_capturing()
    supported = _supports_geometry(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out,
        score_out,
    )
    device = idx_q.device.index
    if supported and device not in _MODULES:
        if capturing:
            _COUNTS["capture_unready"] += 1
            supported = False
        elif device in _ERRORS:
            supported = False
        else:
            try:
                with torch.cuda.device(device):
                    if torch.cuda.get_device_capability(device) != (9, 0):
                        supported = False
                    else:
                        _MODULES[device] = (
                            gen_msa_index_decode_module().build_and_load()
                        )
            except (ImportError, OSError, RuntimeError) as exc:
                _ERRORS[device] = str(exc)
                _LOG.warning("MiniMax M3 indexer compilation failed: %s", str(exc))
                supported = False
    from ._index_decode_triton import msa_index_decode as stock

    if supported:
        output = (
            out[:, :b, :]
            if out is not None
            else torch.empty((1, b, 16), dtype=torch.int32, device=idx_q.device)
        )
        signature = (
            device,
            b,
            idx_q.stride(),
            index_kv_cache.stride(),
            block_table.stride(),
            output.stride(),
            tuple(
                t.data_ptr() % 16
                for t in (idx_q, index_kv_cache, block_table, seq_lens, output)
            ),
        )

        def chain(bound, early):
            """Launch the stock or short-row chain on the query device."""
            with torch.cuda.device(device):
                return stock(
                    idx_q,
                    index_kv_cache,
                    block_table,
                    seq_lens,
                    bound,
                    topk,
                    init_blocks,
                    local_blocks,
                    num_kv_heads,
                    decode_query_len,
                    max_decode_query_len,
                    out=output,
                    early_select=early,
                )

        if not capturing:
            # The runner warms with a short bound then captures with max_model_len.
            # Prepare that conservative bound now, on the actual stream/layout.
            for bound in {16384, max_seq_len} if max_seq_len > 2048 else {16384}:
                key = signature + (bound,)
                if key not in _CHAIN_READY:
                    chain(bound, False)  # stock autotuning and cold-cache fallback
                    chain(bound, True)  # compile the device-gated variant
                    _CHAIN_READY.add(key)
        if max_seq_len <= 2048:
            _MODULES[device].msa_index_decode(
                idx_q, index_kv_cache, block_table, seq_lens, output, max_seq_len
            )
            _COUNTS["direct_dispatch_count"] += 1
        elif signature + (max_seq_len,) in _CHAIN_READY:
            chain(max_seq_len, True)
            _COUNTS["chain_dispatch_count"] += 1
        else:
            _COUNTS["capture_unready"] += 1
            _COUNTS["fallback_count"] += 1
            return chain(max_seq_len, False)
        _COUNTS["dispatch_count"] += 1
        if _COUNTS["dispatch_count"] == 1 and not capturing:
            _LOG.info("MiniMax M3 FlashInfer index kernel dispatched")
        return output
    _COUNTS["fallback_count"] += 1
    with torch.cuda.device(idx_q.device):
        return stock(
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
            topk,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
            out=out,
            score_out=score_out,
        )
