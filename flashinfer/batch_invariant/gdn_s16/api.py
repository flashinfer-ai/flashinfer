# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the GDN class ``gdn.bf16.s16``: what :func:`flashinfer.chunk_gated_delta_rule` runs under
``batch_invariant=True``.

The call keeps ``chunk_gated_delta_rule``'s packed argument form (``[T, heads, 128]`` rows, ``cu_seqlens``) with the
fused gate: ``g`` and ``beta`` are the raw bf16 gate inputs and the kernels compute the decay and the sigmoid from them
(``use_gate_in_kernel=True``, ``beta_is_logit=True``, fp32 ``A_log`` / ``dt_bias``). The state is the bf16 pool
``initial_state`` ``[S, HV, 128, 128]``, addressed by ``state_indices`` (``arange`` when omitted) and updated in place;
a slot holds the sequence's state at its last 16-token boundary, the open chunk lives next to the pool (:mod:`.pool`).
Call shapes:

* decode: one token per sequence, no ``prefix_lens``. The token is appended to its slot's open chunk and its row
  returned; the 16th row of a chunk folds the chunk into the state;
* prefill: more tokens than sequences, no ``prefix_lens``. Every sequence runs from its slot's state as a chunk
  boundary (the slot's open rows are dropped) and leaves its tail open in the slot;
* with ``prefix_lens`` (the tokens of each sequence consumed before the call): a sequence at a multiple of 16 starts
  from its slot's state as a chunk boundary; any other sequence continues its slot's open chunk, whose row count must
  equal ``prefix_lens[i] % 16``. This is the form for multi-token decode steps and for prefill pieces cut anywhere.

Every call is one launch that reads nothing on the host. A call can be captured in a CUDA graph when its metadata is
already on the device (int32 ``state_indices``, ``cu_seqlens`` and ``prefix_lens``; a one-token-per-slot call needs no
offsets) and the pool was used once before the capture: the first call on a pool creates its side state and refuses
to do so inside a capture. A sequence whose ``prefix_lens`` is not where its slot's open chunk ends, whose slot index
is outside the pool or whose offsets leave the packed rows gets NaN rows and its slot is left as it was. A new
sequence loaded into a slot must start with a prefill call or carry ``prefix_lens``. A call with one token per sequence
and no ``prefix_lens`` always continues what the slots hold, whatever the other arguments: a batch of new one-token
sequences carries ``prefix_lens`` of zeros. Slot indices must be unique within a call; sequences are nonempty.
``output_final_state=True`` returns the pool, whose slots hold the states at the last chunk boundaries.

Every unsupported argument raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
``gate_domain`` and ``max_seqlen`` are not read (the gate is always the fused one).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ..support import compute_capability, unsupported
from .pool import get_pool
from .spec import CLASS, GDN_ARCHS, HEAD_DIM

_IO = torch.bfloat16


def _check_inputs(q, k, v, g, beta, A_log, dt_bias) -> Tuple[int, int, int]:
    """``(T, HQ, HV)`` after the dtype and shape checks of the packed inputs and the gate parameters."""
    for t, name in ((q, "q"), (k, "k"), (v, "v")):
        if t.ndim != 3 or t.shape[-1] != HEAD_DIM:
            raise unsupported(
                CLASS, f"{name} of shape {tuple(t.shape)}", f"[T, heads, {HEAD_DIM}]"
            )
        if t.dtype != _IO:
            raise unsupported(CLASS, f"{name} dtype {t.dtype}", "bf16 q / k / v")
        if not t.is_contiguous():
            raise unsupported(CLASS, f"a non-contiguous {name}", "contiguous inputs")
        if t.data_ptr() % 16:
            raise unsupported(
                CLASS, f"a {name} that starts off a 16-byte boundary", "aligned rows"
            )
    T, HQ = int(q.shape[0]), int(q.shape[1])
    HV = int(v.shape[1])
    if k.shape != q.shape or v.shape[0] != T:
        raise unsupported(
            CLASS, "q / k / v of different token or head counts", "k.shape == q.shape"
        )
    if T == 0:
        raise unsupported(CLASS, "an empty call", "at least one token")
    if HQ == 0 or HV % HQ:
        raise unsupported(
            CLASS,
            f"{HQ} q heads with {HV} v heads",
            "a v-head count that is a multiple of the q-head count",
        )
    for t, name in ((g, "g"), (beta, "beta")):
        if t is None:
            raise unsupported(
                CLASS, f"a call without {name}", "raw bf16 gate inputs g and beta"
            )
        if t.dtype != _IO or tuple(t.shape) != (T, HV):
            raise unsupported(
                CLASS,
                f"{name} of shape {tuple(t.shape)} {t.dtype}",
                f"bf16 [{T}, {HV}] (the raw gate input; the kernels compute the gate)",
            )
        if not t.is_contiguous():
            raise unsupported(CLASS, f"a non-contiguous {name}", "contiguous inputs")
    for t, name in ((A_log, "A_log"), (dt_bias, "dt_bias")):
        if t is None:
            raise unsupported(
                CLASS, f"a call without {name}", f"fp32 [{HV}] A_log and dt_bias"
            )
        if t.dtype != torch.float32 or tuple(t.shape) != (HV,) or not t.is_contiguous():
            raise unsupported(
                CLASS,
                f"{name} of shape {tuple(t.shape)} {t.dtype}",
                f"a contiguous fp32 [{HV}]",
            )
    return T, HQ, HV


def _check_pool(pool: Optional[torch.Tensor], HV: int, device: torch.device) -> None:
    if pool is None:
        raise unsupported(
            CLASS,
            "a call without initial_state",
            "a bf16 state pool the caller keeps (the slot carries the sequence between calls)",
        )
    if pool.dtype != _IO:
        raise unsupported(
            CLASS,
            f"a {pool.dtype} state",
            "a bf16 state pool (the class rounds the state to bf16 at every 16-token boundary)",
        )
    if pool.ndim != 4 or tuple(pool.shape[1:]) != (HV, HEAD_DIM, HEAD_DIM):
        raise unsupported(
            CLASS,
            f"a state of shape {tuple(pool.shape)}",
            f"[slots, {HV}, {HEAD_DIM}, {HEAD_DIM}]",
        )
    if not pool.is_contiguous() or pool.data_ptr() % 16:
        raise unsupported(
            CLASS,
            "a non-contiguous state pool or one off a 16-byte boundary",
            "a contiguous, aligned pool",
        )
    if pool.device != device:
        raise unsupported(
            CLASS, "a state on another device than the inputs", "one device"
        )


def _slot_ids(
    indices: Optional[torch.Tensor], N: int, device: torch.device, S: int, gp
) -> torch.Tensor:
    if indices is None:
        if N > S:
            raise unsupported(
                CLASS,
                f"{N} sequences on a pool of {S} slots",
                "state_indices or a pool of at least N slots",
            )
        return gp.arange(N)
    if indices.dtype != torch.int32 or indices.ndim != 1 or indices.numel() != N:
        raise unsupported(
            CLASS,
            f"state_indices of shape {tuple(indices.shape)} {indices.dtype}",
            f"int32 [{N}] (one slot per sequence)",
        )
    if indices.device != device:
        raise unsupported(
            CLASS, "state_indices on another device", "the inputs' device"
        )
    return indices.contiguous()


def _host_offsets(cu_seqlens_cpu, N: int, T: int):
    cu = [
        int(x)
        for x in (
            cu_seqlens_cpu.tolist()
            if isinstance(cu_seqlens_cpu, torch.Tensor)
            else cu_seqlens_cpu
        )
    ]
    if (
        len(cu) != N + 1
        or cu[0] != 0
        or cu[-1] != T
        or any(b <= a for a, b in zip(cu, cu[1:], strict=False))
    ):
        raise unsupported(
            CLASS,
            "cu_seqlens_cpu that are not increasing from 0 to the token count",
            "nonempty sequences tiling the tokens",
        )
    return cu


def chunk_gated_delta_rule_batch_invariant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    *,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_cpu: Optional[Union[Sequence[int], torch.Tensor]],
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    prefix_lens: Optional[torch.Tensor],
    use_gate_in_kernel: bool,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    beta_is_logit: bool,
    backend: str,
    use_cp,
    refused: Dict[str, object],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if backend != "auto":
        raise unsupported(
            CLASS,
            f'backend="{backend}" with batch_invariant=True',
            'backend="auto" (the class is the kernel)',
        )
    if use_cp != "auto":
        raise unsupported(
            CLASS,
            f"use_cp={use_cp!r}",
            'use_cp="auto" (the class has no context parallelism)',
        )
    for name, value in refused.items():
        if value is None or value is False or (isinstance(value, int) and value == 0):
            continue
        raise unsupported(CLASS, f"{name}={value!r}", "the class's own inputs")
    if not (use_gate_in_kernel and beta_is_logit and use_qk_l2norm_in_kernel):
        raise unsupported(
            CLASS,
            "precomputed gates or use_qk_l2norm_in_kernel=False",
            "use_gate_in_kernel=True, beta_is_logit=True and use_qk_l2norm_in_kernel=True "
            "(the gate and the Q / K l2-norm are part of the class's term)",
        )
    T, HQ, HV = _check_inputs(q, k, v, g, beta, A_log, dt_bias)
    dev = q.device
    if dev.type != "cuda" or compute_capability(dev) not in GDN_ARCHS:
        raise unsupported(
            CLASS, f"the device {dev}", "a CUDA device of SM100, SM103 or SM107"
        )
    for t in (k, v, g, beta, A_log, dt_bias):
        if t.device != dev:
            raise unsupported(CLASS, "inputs on several devices", "one device")
    _check_pool(initial_state, HV, dev)
    pool = initial_state
    if output_state is not None and output_state is not pool:
        raise unsupported(
            CLASS,
            "an output_state other than the pool",
            "the pool updated in place (output_state omitted or initial_state itself)",
        )
    if output is not None:
        if output.shape != v.shape or output.dtype != _IO or not output.is_contiguous():
            raise unsupported(
                CLASS,
                "an output of another shape, dtype or layout than v",
                f"a contiguous bf16 [T, HV, {HEAD_DIM}] output",
            )
        if output.device != dev:
            raise unsupported(
                CLASS, "an output on another device", "the inputs' device"
            )
    if cu_seqlens is not None and (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2):
        raise unsupported(CLASS, "a malformed cu_seqlens", "a 1-D [N + 1] tensor")
    if cu_seqlens is not None:
        N = int(cu_seqlens.numel()) - 1
    elif cu_seqlens_cpu is not None:
        N = len(cu_seqlens_cpu) - 1
    elif state_indices is not None:
        # no offsets: one token per slot when the counts agree (a decode step), else one sequence
        N = int(state_indices.numel()) if int(state_indices.numel()) == T else 1
    else:
        N = 1
    if N < 1 or N > T:
        raise unsupported(CLASS, f"{N} sequences over {T} tokens", "nonempty sequences")
    gp = get_pool(pool, HQ)
    if gp.HQ != HQ:
        raise unsupported(
            CLASS,
            f"{HQ} q heads on a pool first used with {gp.HQ}",
            "one head geometry per pool",
        )
    slots = _slot_ids(state_indices, N, dev, int(pool.shape[0]), gp)
    if prefix_lens is not None and (
        prefix_lens.ndim != 1
        or prefix_lens.numel() != N
        or prefix_lens.dtype not in (torch.int32, torch.int64)
    ):
        raise unsupported(
            CLASS, "a malformed prefix_lens", f"an int32 or int64 [{N}] tensor"
        )
    scale_f = scale if scale is not None and scale != 0.0 else 1.0 / math.sqrt(HEAD_DIM)
    out = output if output is not None else torch.empty_like(v)
    rows = (q, k, v, g, beta)
    offs = None
    if cu_seqlens_cpu is not None:
        offs = _host_offsets(cu_seqlens_cpu, N, T)

    if T == N and prefix_lens is None:
        gp.decode(pool, slots, rows, A_log, dt_bias, scale_f, out)
        return (out, pool) if output_final_state else out
    if cu_seqlens is not None:
        cu = cu_seqlens
        if cu.dtype != torch.int32 or cu.device != dev or not cu.is_contiguous():
            cu = cu.to(device=dev, dtype=torch.int32).contiguous()
    elif offs is not None:
        cu = torch.tensor(offs, dtype=torch.int32, device=dev)
    else:
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    prefix = None
    if prefix_lens is not None:
        prefix = prefix_lens.to(device=dev, dtype=torch.int32).contiguous()
    # T - N + 1 bounds the longest sequence when no host offsets are given
    if offs is not None:
        max_tokens = max(b - a for a, b in zip(offs, offs[1:], strict=False))
    else:
        max_tokens = T - N + 1
    gp.prefill(pool, slots, cu, prefix, rows, A_log, dt_bias, scale_f, out, max_tokens)
    return (out, pool) if output_final_state else out
