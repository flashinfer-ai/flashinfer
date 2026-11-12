"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MSA (MiniMax-style sparse attention) stage-1 block-score top-k indexer,
SM107 (Rubin) backend.

This is the variable-length, single-index-head form of the MSA indexing pass:
one BF16 index head scores 128-token key blocks per query token (causal block
MAX of q . k) and selects the token's own block plus the top-15 strictly-prior
blocks, with a deterministic tie-break (lower block id wins). It differs from
:func:`flashinfer.msa_ops.msa_topk_select` in that scoring and selection are
fused over a ragged batch (per-request key extents and prefix offsets) rather
than consuming a precomputed dense score plane.
"""

import functools
import logging
import os

import torch

from ..api_logging import flashinfer_experimental_api
from ..utils import get_compute_capability

logger = logging.getLogger(__name__)

_BLOCK = 128
_D = 128
_TOPK = 16

_FEATURE = "MSA block-score top-k indexer (BF16, block 128, top-16 on SM107)"

#: User-selectable backends. "specialized" is the fused SM107 kernel, the only
#: implementation of this op; "auto" routes to it when the contract gate
#: passes. There is no composable fallback implementation, so a call the
#: specialized backend cannot serve raises ``NotImplementedError`` (with the
#: reason recorded in the dispatch stats) rather than falling back.
BACKENDS = ("auto", "specialized")

_KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

_stats: dict = {
    "dispatches": 0,
    "backend_dispatches": {"specialized": 0},
    "fallback_reasons": {},
    "module_loaded": False,
    "module_error": None,
}


def _record_fallback(reason: str) -> None:
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def msa_blockscore_indexer_stats() -> dict:
    """Diagnostics: per-backend dispatch counters, recorded reasons calls could
    not be served (``fallback_reasons``) and JIT module state."""
    return {
        **_stats,
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "backends": BACKENDS,
    }


@functools.cache
def _get_module():
    """Build/load the SM107 module once; never inside a CUDA-graph capture."""
    from ..jit.msa_blockscore_indexer import gen_msa_blockscore_indexer_sm107_module

    return gen_msa_blockscore_indexer_sm107_module().build_and_load()


def is_msa_blockscore_indexer_supported(device: torch.device) -> bool:
    """Whether :func:`msa_blockscore_topk_indexer` runs on ``device``."""
    return device.type == "cuda" and get_compute_capability(torch.device(device)) == (
        10,
        7,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_blockscore_topk_indexer(
    qi: torch.Tensor,
    ki: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_base: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    block_size: int = 128,
    topk: int = 16,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Fused MSA stage-1 block-score top-k indexer (SM107 / Rubin).

    For every query token, scores each of its request's causally-visible
    128-token key blocks with the block MAX of ``q . k`` under one BF16 index
    head (``D=128``), then selects the token's **own** block (always included)
    plus the ``topk - 1`` highest-scoring blocks **strictly before** it.
    Ties between prior blocks break deterministically: the lower block id
    wins. Selected block ids are returned ascending, with ``-1`` padding when
    fewer than ``topk`` blocks exist.

    Keys are stored per request in contiguous 128-token blocks: request ``r``
    owns blocks ``kv_base[r] .. kv_base[r] + ceil(seq_lens[r] / 128) - 1`` of
    ``ki``, holding its ``seq_lens[r]`` kv tokens (``prefix_lens[r]`` context
    tokens followed by the new chunk) with an undefined padded tail in the
    last block. Query token ``t`` of request ``r`` (``cu_seqlens[r] <= t <
    cu_seqlens[r+1]``) sits at position ``prefix_lens[r] + (t -
    cu_seqlens[r])`` of its request, and block ids in the output are
    request-local (0 is the request's first block).

    The call path performs no device synchronization and allocates nothing
    once the module is built, so it is CUDA-graph capturable; build the module
    (one eager warm-up call) before capture.

    Parameters
    ----------
    qi : torch.Tensor
        ``[TQ, 1, 128]`` bfloat16 query index vectors, requests concatenated
        in ``cu_seqlens`` order.
    ki : torch.Tensor
        ``[TKPAD, 1, 128]`` bfloat16 key index vectors.
        ``TKPAD = sum_r ceil(seq_lens[r] / 128) * 128`` (every request's keys
        padded up to a whole number of 128-token blocks).
    cu_seqlens : torch.Tensor
        ``[B + 1]`` int32 exclusive prefix sum of per-request query counts;
        ``cu_seqlens[0] == 0`` and ``cu_seqlens[B] == TQ``.
    kv_base : torch.Tensor
        ``[B]`` int32 first key *block* index of each request in ``ki``.
    seq_lens : torch.Tensor
        ``[B]`` int32 total kv tokens per request (prefix + new chunk).
    prefix_lens : torch.Tensor
        ``[B]`` int32 context (prefix) token count per request.
    output : torch.Tensor, optional
        Pre-allocated ``[TQ, topk]`` int32 output. Allocated internally if
        not provided.
    block_size : int
        Key block size in tokens. Must be 128.
    topk : int
        Number of blocks selected per query token. Must be 16.
    backend : str
        One of ``BACKENDS`` (``"auto"`` or ``"specialized"``).
        ``"specialized"`` is the fused SM107 kernel, the only implementation
        of this op; ``"auto"`` routes to it when the contract gate passes.
        An explicitly selected backend is never swapped for another one.

    Returns
    -------
    torch.Tensor
        ``[TQ, topk]`` int32 ascending request-local block ids, ``-1``
        tail-padded.

    Notes
    -----
    Device-resident values (``cu_seqlens``, ``kv_base``, ``seq_lens``,
    ``prefix_lens``) are not validated on the host - reading them would force
    a sync, which is illegal under CUDA-graph capture. The caller must supply
    metadata consistent with the layout above; inconsistent metadata reads
    ``ki`` out of bounds.

    Setting ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` disables the
    specialized backend. Because this op has no composable fallback
    implementation, a disabled or unserved call raises
    ``NotImplementedError`` (the reason is recorded in
    :func:`msa_blockscore_indexer_stats` under ``fallback_reasons``) rather
    than silently running a slower path.
    """
    # GPU-free guards first: backend selection, kill switch, then the device
    # gate; tensor contract guards follow below.
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if os.environ.get(_KILL_SWITCH_ENV, "0") == "1":
        _record_fallback("kill_switch")
        raise NotImplementedError(
            f"{_KILL_SWITCH_ENV}=1 disables the specialized backend, and "
            "msa_blockscore_topk_indexer has no composable fallback "
            "implementation; unset the variable or route this op elsewhere"
        )
    if not is_msa_blockscore_indexer_supported(qi.device):
        _record_fallback("unsupported_device")
        raise NotImplementedError(
            "msa_blockscore_topk_indexer is implemented for compute capability "
            f"10.7 (SM107) only (backend {backend!r} resolves to the "
            f"'specialized' kernel, the only implementation); got device "
            f"{qi.device}"
        )
    if block_size != _BLOCK:
        raise ValueError(f"block_size must be {_BLOCK}, got {block_size}")
    if topk != _TOPK:
        raise ValueError(f"topk must be {_TOPK}, got {topk}")

    for name, t in (("qi", qi), ("ki", ki)):
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16, got {t.dtype}")
        if t.ndim != 3 or t.shape[1] != 1 or t.shape[2] != _D:
            raise ValueError(
                f"{name} must have shape [*, 1, {_D}], got {tuple(t.shape)}"
            )
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    tq = qi.shape[0]
    tkpad = ki.shape[0]
    if tq < 1:
        raise ValueError(f"qi must hold at least one query token, got TQ={tq}")
    if tkpad % _BLOCK != 0 or tkpad < _BLOCK:
        raise ValueError(
            f"ki rows (TKPAD={tkpad}) must be a positive multiple of {_BLOCK}: "
            "every request's keys are padded to whole 128-token blocks"
        )

    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be 1D int32")
    batch = cu_seqlens.numel() - 1
    if batch < 1:
        raise ValueError("cu_seqlens must have at least 2 entries")
    for name, t in (
        ("kv_base", kv_base),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if t.dtype != torch.int32 or t.ndim != 1:
            raise ValueError(f"{name} must be 1D int32")
        if t.numel() != batch:
            raise ValueError(
                f"{name} must have B={batch} entries (cu_seqlens has B+1), "
                f"got {t.numel()}"
            )
    for name, t in (
        ("ki", ki),
        ("cu_seqlens", cu_seqlens),
        ("kv_base", kv_base),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if t.device != qi.device:
            raise ValueError(f"{name} must be on qi's device ({qi.device})")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if output is None:
        output = torch.empty((tq, _TOPK), dtype=torch.int32, device=qi.device)
    else:
        if output.shape != (tq, _TOPK):
            raise ValueError(
                f"output shape must be ({tq}, {_TOPK}), got {tuple(output.shape)}"
            )
        if output.dtype != torch.int32:
            raise ValueError(f"output must be int32, got {output.dtype}")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
        if output.device != qi.device:
            raise ValueError(f"output must be on qi's device ({qi.device})")

    try:
        module = _get_module()
        _stats["module_loaded"] = True
    except Exception as exc:
        # No next backend exists for this op: log + record, then propagate.
        error = f"{type(exc).__name__}: {exc}"[:500]
        _stats["module_error"] = error
        _record_fallback("jit_build_failed")
        logger.warning(
            "flashinfer.msa_ops.msa_blockscore_topk_indexer: JIT build of the "
            "specialized kernel failed and no other backend implements this "
            "op: %s",
            error,
        )
        raise

    module.run(qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens, output)
    _stats["dispatches"] += 1
    _stats["backend_dispatches"]["specialized"] += 1
    return output
