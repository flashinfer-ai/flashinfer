# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused bf16 -> quant + routing-repack staging (``src.inputs_process``).

One ``DataPreprocess`` launch replaces the multi-kernel torch staging path
(fp32 upcast + ~a dozen elementwise/reduce launches + per-tensor copies) that
`stage_mega_moe_inputs` used per forward: the kernel reads the bf16 row once,
quantizes it in smem (NVFP4 per-16 E4M3 scales or MXFP8 per-32 E8M0), and
repacks routing to the int64/fp32 layout the mega kernels consume.

Caching mirrors the mega frontends: one ``cute.compile`` per
``(topk, hidden, quant_type)`` per process, plus a launch-args cache keyed on
data pointers + token count + stream, so the steady state is a single cached
kernel launch (CUDA-graph capturable; compile is guarded by
:func:`.comm.ensure_not_capturing` and must happen during warmup).

NVFP4 uses the offline (calibrated constant) scale mode; the online amax mode
exists in the kernel but is not wired here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

from .comm import ensure_not_capturing

_QUANT_TYPES = ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2")


@dataclasses.dataclass
class _CompiledStager:
    dp: Any
    compiled: Any = None
    launch_key: Optional[tuple] = None
    launch_args: Optional[tuple] = None
    launch_kwargs: Optional[dict] = None


# Explicit dict rather than functools.cache: a miss must run the
# ensure_not_capturing guard (a hit must not), and the entry is a mutable
# launch-args cache, not a pure function value.
_STAGERS: dict[tuple, _CompiledStager] = {}

# topk_idx_out.data_ptr() -> num_tokens staged by the last fused call into
# that buffer (tail-fill memoization; see the fill logic below).
_LAST_STAGED_N: dict[int, int] = {}

# Buffers that have ever been staged inside a CUDA graph capture: replays
# bypass host code, so the memo is permanently untrustworthy for them.
_GRAPH_CAPTURED_BUFFERS: set[int] = set()


def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def note_staged_tokens(topk_idx_out: torch.Tensor, num_tokens: int) -> None:
    """Record a non-fused staging into ``topk_idx_out`` (tail-fill memo).

    The torch fallback paths stage live rows and re-mask the full tail; they
    must update the memo so a later fused call cannot skip a fill over rows
    the fallback left live. Unknown buffers default to "assume fully live"
    (prev_n = capacity), which is always safe.
    """
    _LAST_STAGED_N[topk_idx_out.data_ptr()] = num_tokens


def staged_tokens(topk_idx_out: torch.Tensor) -> Optional[int]:
    """Live-token count of the last staging into this buffer, if known."""
    return _LAST_STAGED_N.get(topk_idx_out.data_ptr())


def forget_staged_tokens(topk_idx_out: torch.Tensor) -> None:
    """Evict a buffer from the staging memos at workspace teardown.

    The symmetric heap (and the caching allocator) reuse freed addresses, so
    a later workspace can land on this pointer and would otherwise inherit a
    stale live-count or graph-captured mark from the destroyed buffer.
    """
    ptr = topk_idx_out.data_ptr()
    _LAST_STAGED_N.pop(ptr, None)
    _GRAPH_CAPTURED_BUFFERS.discard(ptr)


def fused_quant_stage_supported(
    hidden_states: torch.Tensor, quant_type: str = "nvfp4"
) -> bool:
    """True when the fused kernel can take this activation tensor.

    The DSL launcher requires 16-byte-aligned data pointers; torch allocations
    satisfy this, but sliced/offset views may not — callers fall back to their
    torch staging path in that case.

    The fused kernel also stages the SF plane at its exact unpadded width, so
    ``hidden // sf_vec`` must be a multiple of 4 (the buffer's round-up-to-4
    SF padding must be zero): hidden % 64 for nvfp4 (sf_vec 16), hidden % 128
    for mxfp8 (sf_vec 32). 128-misaligned mxfp8 shapes (gpt-oss: 2880) fall
    back to torch staging rather than failing.
    """
    sf_align = 64 if quant_type == "nvfp4" else 128
    return hidden_states.data_ptr() % 16 == 0 and hidden_states.shape[1] % sf_align == 0


def fused_quant_stage(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_out: torch.Tensor,
    x_sf_out: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    quant_type: str,
    norm_const: Optional[float] = None,
) -> None:
    """Quantize + stage one batch into the mega symm-buffer views.

    ``hidden_states`` is the live ``(n, hidden)`` bf16 batch; the ``*_out``
    tensors are the full-capacity buffer views (``x``/``x_sf``/``topk_idx``/
    ``topk_weights`` of a mega symm buffer, or plain CUDA tensors with the
    same dtypes). Rows ``[:n]`` are staged and the ``topk_idx`` capacity tail
    is re-masked to ``-1``, matching the torch staging path's contract.

    ``norm_const`` is the NVFP4 offline per-tensor scale (required for
    ``quant_type="nvfp4"``, rejected otherwise).
    """
    if quant_type not in _QUANT_TYPES:
        raise ValueError(
            f"quant_type must be one of {_QUANT_TYPES}, got {quant_type!r}"
        )
    is_nvfp4 = quant_type == "nvfp4"
    if is_nvfp4 and norm_const is None:
        raise ValueError("nvfp4 staging requires an offline norm_const")
    if not is_nvfp4 and norm_const is not None:
        raise ValueError(
            f"{quant_type} staging is self-scaled; norm_const must be None"
        )

    num_tokens, hidden = hidden_states.shape
    capacity = x_out.shape[0]
    if num_tokens == 0:
        return
    sf_vec = 16 if is_nvfp4 else 32
    # hidden // sf_vec must be a multiple of 4 so the buffer's round-up-to-4
    # SF padding is zero and the full-width view is the exact block count —
    # hidden % 64 (nvfp4) / % 128 (mxfp8). Callers gate on
    # fused_quant_stage_supported() and fall back to torch staging otherwise.
    if hidden % (4 * sf_vec) != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {4 * sf_vec} for the fused "
            f"{quant_type} stage (got {hidden}); use the torch staging path."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")
    topk = topk_ids.shape[1]
    n_blocks = hidden // sf_vec
    if x_sf_out.shape[1] != n_blocks:
        raise ValueError(
            f"x_sf trailing dim ({x_sf_out.shape[1]}) must be {n_blocks} "
            f"for hidden={hidden}, {quant_type}."
        )

    key = (topk, hidden, quant_type)
    stager = _STAGERS.get(key)
    if stager is None:
        ensure_not_capturing("fused staging construction")
        from src.inputs_process import DataPreprocess

        stager = _CompiledStager(
            dp=DataPreprocess(topk=topk, hidden=hidden, quant_type=quant_type)
        )
        _STAGERS[key] = stager

    import cuda.bindings.driver as cuda_driver

    stream = torch.cuda.current_stream().cuda_stream
    launch_key = (
        hidden_states.data_ptr(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        x_out.data_ptr(),
        x_sf_out.data_ptr(),
        topk_idx_out.data_ptr(),
        topk_weights_out.data_ptr(),
        num_tokens,
        norm_const,
        stream,
    )
    if stager.compiled is None or stager.launch_key != launch_key:
        import cutlass

        args = (
            _to_cute(hidden_states, 16),
            _to_cute(topk_ids, 4),
            _to_cute(topk_weights, 4),
            None,  # token_padding_info: only live rows are launched
            _to_cute(x_out[:num_tokens], 16),
            _to_cute(x_sf_out[:num_tokens], 4),
            _to_cute(topk_idx_out[:num_tokens], 4),
            _to_cute(topk_weights_out[:num_tokens], 4),
            cuda_driver.CUstream(stream),
        )
        kwargs = {"offline_norm_const": cutlass.Float32(norm_const)} if is_nvfp4 else {}
        if stager.compiled is None:
            ensure_not_capturing("fused staging cute.compile")
            import cutlass.cute as cute

            stager.compiled = cute.compile(stager.dp, *args, **kwargs)
        stager.launch_args = args
        stager.launch_kwargs = kwargs
        stager.launch_key = launch_key

    stager.compiled(*stager.launch_args, **stager.launch_kwargs)

    # Parity with the torch stager: rows beyond this batch must stay masked
    # (-1) so they cannot dispatch as live tokens. The buffer starts fully
    # masked and staging only overwrites [:n], so eagerly a fill is needed
    # ONLY for the [n, prev_n) rows a previous LARGER batch left routed —
    # memoized per buffer so same-or-growing batches skip the launch (all
    # layers share one buffer in an engine, so at most the first layer of a
    # shrinking step pays it).
    #
    # Under CUDA graphs the memo is USELESS AND DANGEROUS: graphs replay
    # device ops without host logic, and engines capture one graph per batch
    # size — a small-size graph replayed after a larger one must mask
    # everything the larger graph staged, and an EAGER call after any replay
    # cannot trust a host count the replays never updated. So every captured
    # graph bakes the full conservative fill, and a buffer that has ever been
    # captured permanently loses the memo optimization (eager fills fully).
    ptr = topk_idx_out.data_ptr()
    if torch.cuda.is_current_stream_capturing():
        _GRAPH_CAPTURED_BUFFERS.add(ptr)
    if ptr in _GRAPH_CAPTURED_BUFFERS:
        if num_tokens < capacity:
            topk_idx_out[num_tokens:capacity].fill_(-1)
    else:
        prev_n = _LAST_STAGED_N.get(ptr, capacity)
        if num_tokens < prev_n:
            topk_idx_out[num_tokens:prev_n].fill_(-1)
    # The memo always records the ACTUAL live count of this staging: it also
    # backs staged_tokens() / compute(output=None) view slicing, which must
    # see the real n even for captured buffers (each captured graph's view is
    # fixed at its own capture-time batch size).
    _LAST_STAGED_N[ptr] = num_tokens
