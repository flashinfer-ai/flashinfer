# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the KDA class ``kda.bf16.s16``: what :func:`flashinfer.recurrent_kda` runs under
``batch_invariant=True``.

The call keeps ``recurrent_kda``'s argument forms. The state is the bf16 pool ``initial_state`` ``[S, HV, 128, 128]``,
addressed by ``ssm_state_indices`` (``arange`` when omitted) and updated in place; a slot holds the sequence's state at
its last 16-token boundary, the rows since it live next to the pool (:mod:`.pool`). Call shapes:

* prefill: ``cu_seqlens`` with more tokens than sequences, or ``[B, T > 1]``: one launch of the prefill kernel. Every
  sequence starts from its slot's state as a chunk boundary (open rows are dropped), so a chunked prefill through
  this form cuts at multiples of 16 tokens from the sequence start, except for its last piece;
* decode: ``[B, 1]``, or ``cu_seqlens`` with as many tokens as sequences (the offsets are then not read: token ``i``
  belongs to sequence ``i``): one launch of the decode kernel adds row ``p`` to the slot's open chunk, and the 16th row
  folds the chunk into the boundary state;
* :func:`recurrent_kda_class_s_extend` appends ``[N, T]`` tokens to the slots' open chunks (a chunked prefill cut at
  any token, a multi-token decode step): one launch of the prefill kernel. ``recurrent_kda``'s arguments have no
  spelling for it, since its ``[B, T > 1]`` form starts a prefill.

"One launch" is for contiguous activations; a call without ``ssm_state_indices`` uses slot ids the class keeps per
batch size (made by one extra launch the first time, or inside a graph capture).

The open rows belong to the state the class last wrote. A decode or extend call compares the first eight values of
each head's state with a stamp of what the class wrote and starts a new chunk when they differ, which catches a new
request loaded into the slot. It cannot see a write that leaves those values unchanged: a slot reset to the same
zeros after a request shorter than 16 tokens, a snapshot of the boundary state restored while rows were open, a
state copied from a slot at the same boundary. After any such write the caller prefills the slot with more than one
token or calls :func:`recurrent_kda_class_s_reset`. Slot indices must be unique: a prefill, an extend and a reset
read them back and refuse repeats and indices outside the pool (so with caller-given slots, or ``cu_seqlens``, they
cannot be captured in a CUDA graph); a decode step reads nothing back, and skips a row whose slot is outside the
pool without writing its output.

Unsupported arguments raise :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from ..support import compute_capability, unsupported
from .pool import ClassSPool, get_pool
from .spec import CLASS, HEAD_DIM, KDA_ARCHS, MIN_LOWER_BOUND

_IO = torch.bfloat16


def _check_device(device: torch.device) -> None:
    if device.type != "cuda" or compute_capability(device) not in KDA_ARCHS:
        raise unsupported(
            CLASS,
            f"the device {device}",
            "a CUDA device of compute capability "
            + " / ".join(f"{a}.{b}" for a, b in KDA_ARCHS),
        )


def _check_params(A_log, dt_bias, lower_bound, HV: int) -> None:
    if A_log is None or dt_bias is None:
        raise unsupported(
            CLASS,
            "a call without A_log / dt_bias",
            "the fused gate (A_log and dt_bias given)",
        )
    if A_log.dtype != torch.float32 or tuple(A_log.shape) != (HV,):
        raise unsupported(
            CLASS, f"A_log {tuple(A_log.shape)} {A_log.dtype}", f"fp32 [{HV}]"
        )
    if dt_bias.dtype != torch.float32 or tuple(dt_bias.shape) not in (
        (HV, HEAD_DIM),
        (HV * HEAD_DIM,),
    ):
        raise unsupported(
            CLASS,
            f"dt_bias {tuple(dt_bias.shape)} {dt_bias.dtype}",
            f"fp32 [{HV}, {HEAD_DIM}]",
        )
    if not (A_log.is_contiguous() and dt_bias.is_contiguous()):
        raise unsupported(
            CLASS, "a non-contiguous A_log or dt_bias", "contiguous parameters"
        )
    if (
        lower_bound is None
        or not isinstance(lower_bound, (int, float))
        or not math.isfinite(lower_bound)
        or not MIN_LOWER_BOUND <= lower_bound < 0.0
    ):
        raise unsupported(
            CLASS,
            f"lower_bound={lower_bound!r}",
            f"a gate lower bound in [{MIN_LOWER_BOUND}, 0) (the chunk's decayed operands must fit bf16)",
        )


def _check_inputs(q, k, v, g, beta) -> int:
    """The head count after the dtype and shape checks of the five activations (``[B, T, H, 128]`` / ``[B, T, H]``)."""
    for t, name in ((q, "q"), (k, "k"), (v, "v"), (g, "g")):
        if t.ndim != 4 or t.shape[-1] != HEAD_DIM:
            raise unsupported(
                CLASS, f"{name} of shape {tuple(t.shape)}", f"[B, T, heads, {HEAD_DIM}]"
            )
        if t.dtype != _IO:
            raise unsupported(CLASS, f"{name} dtype {t.dtype}", "bf16 activations")
    if beta.ndim != 3 or beta.dtype != _IO:
        raise unsupported(
            CLASS, f"beta of shape {tuple(beta.shape)} {beta.dtype}", "bf16 [B, T, HV]"
        )
    B, T, H = q.shape[:3]
    HV = v.shape[2]
    if k.shape != q.shape:
        raise unsupported(CLASS, "k of a different shape than q", "k.shape == q.shape")
    if v.shape[:2] != (B, T) or g.shape != v.shape or beta.shape != (B, T, HV):
        raise unsupported(
            CLASS,
            "v / g / beta shapes that do not match q",
            "[B, T, HV, 128] and [B, T, HV]",
        )
    if HV != H:
        raise unsupported(CLASS, f"GQA with H={H}, HV={HV}", "HV == H")
    for t in (k, v, g, beta):
        if t.device != q.device:
            raise unsupported(CLASS, "inputs on several devices", "one device")
    return HV


def _check_pool(pool: torch.Tensor, HV: int, device: torch.device) -> None:
    if pool.dtype != _IO:
        raise unsupported(
            CLASS,
            f"a {pool.dtype} state",
            "a bf16 state pool (the class's boundary state is bf16)",
        )
    if pool.ndim != 4 or tuple(pool.shape[1:]) != (HV, HEAD_DIM, HEAD_DIM):
        raise unsupported(
            CLASS,
            f"a state of shape {tuple(pool.shape)}",
            f"[slots, {HV}, {HEAD_DIM}, {HEAD_DIM}]",
        )
    if not pool.is_contiguous():
        raise unsupported(CLASS, "a non-contiguous state pool", "a contiguous pool")
    if pool.device != device:
        raise unsupported(
            CLASS, "a state on another device than the inputs", "one device"
        )


def _slot_ids(indices: Optional[torch.Tensor], N: int, sp: ClassSPool) -> torch.Tensor:
    if indices is None:
        if sp.slots < N:
            raise unsupported(
                CLASS,
                f"{N} sequences on a pool of {sp.slots} slots",
                "ssm_state_indices or a pool of at least N slots",
            )
        return sp.default_slots(N)
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int32
        or indices.ndim != 1
        or indices.numel() != N
    ):
        what = (
            f"of shape {tuple(indices.shape)} {indices.dtype}"
            if isinstance(indices, torch.Tensor)
            else repr(indices)
        )
        raise unsupported(
            CLASS,
            f"ssm_state_indices {what}",
            f"int32 [{N}] (one slot per sequence; the [N, 1 + S] form belongs to speculative decode)",
        )
    if indices.device != sp.device:
        raise unsupported(
            CLASS, "ssm_state_indices on another device", "the inputs' device"
        )
    return indices.contiguous()


def _no_capture(what: str) -> None:
    """Refuse a call that reads device values back to the host while a CUDA graph is being captured."""
    if torch.cuda.is_current_stream_capturing():
        raise unsupported(
            CLASS,
            f"{what} inside a CUDA graph capture",
            "a decode step, or a [B, T] call on the default slots",
        )


def _check_host_slots(slot_ids: torch.Tensor, S: int) -> None:
    """Refuse slot ids that repeat or leave the pool (one device read)."""
    _no_capture("caller-given slots in a prefill, extend or reset call")
    slot_list = slot_ids.tolist()
    if len(set(slot_list)) != len(slot_list) or any(s < 0 or s >= S for s in slot_list):
        raise unsupported(
            CLASS,
            "ssm_state_indices that repeat a slot or leave the pool",
            f"unique slots in [0, {S})",
        )


def _check_output(output: Optional[torch.Tensor], v: torch.Tensor, inputs) -> None:
    if output is None:
        return
    if (
        output.shape != v.shape
        or output.dtype != _IO
        or not output.is_contiguous()
        or output.device != v.device
    ):
        raise unsupported(
            CLASS,
            "an output of another shape, dtype, layout or device than v",
            "a contiguous bf16 [B, T, HV, 128] output on the inputs' device",
        )
    storage = output.untyped_storage()._cdata
    if any(t.untyped_storage()._cdata == storage for t in inputs):
        raise unsupported(
            CLASS, "an output that shares memory with an input", "a separate output"
        )


def _pack(q, k, v, g, beta, HV: int) -> Tuple[torch.Tensor, ...]:
    return (
        q.reshape(-1, HV, HEAD_DIM).contiguous(),
        k.reshape(-1, HV, HEAD_DIM).contiguous(),
        v.reshape(-1, HV, HEAD_DIM).contiguous(),
        g.reshape(-1, HV, HEAD_DIM).contiguous(),
        beta.reshape(-1, HV).contiguous(),
    )


def recurrent_kda_class_s(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    beta_is_logit: bool,
    backend: str,
    refused: Dict[str, object],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if backend != "auto":
        raise unsupported(
            CLASS,
            f'backend="{backend}" with batch_invariant=True',
            'backend="auto" (the class is the kernel)',
        )
    for name, value in refused.items():
        if value is None or value is False or (isinstance(value, int) and value == 0):
            continue
        limit = {
            "num_spec_tokens": "single-token decode (speculative decode is not in the class yet)",
            "num_accepted_tokens": "single-token decode (speculative decode is not in the class yet)",
        }.get(name, "the class's own inputs")
        raise unsupported(CLASS, f"{name}={value!r}", limit)
    if not (use_qk_l2norm_in_kernel and use_gate_in_kernel and beta_is_logit):
        raise unsupported(
            CLASS,
            "use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False or beta_is_logit=False",
            "the fused contract (in-kernel Q / K l2-norm, gate and beta sigmoid)",
        )
    HV = _check_inputs(q, k, v, g, beta)
    _check_params(A_log, dt_bias, lower_bound, HV)
    dev = q.device
    _check_device(dev)
    if A_log.device != dev or dt_bias.device != dev:
        raise unsupported(CLASS, "inputs on several devices", "one device")
    scale_f = HEAD_DIM**-0.5 if scale is None else float(scale)

    B, T = q.shape[:2]
    if B == 0 or T == 0:
        raise unsupported(CLASS, "an empty call", "at least one token")
    if cu_seqlens is not None:
        if B != 1:
            raise unsupported(
                CLASS,
                f"cu_seqlens with a batch of {B}",
                "[1, total_tokens, ...] inputs",
            )
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.device != dev
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise unsupported(
                CLASS, "a malformed cu_seqlens", "a 1-D integer [N + 1] CUDA tensor"
            )
        N = int(cu_seqlens.numel()) - 1
        decode = T == N
    else:
        N = B
        decode = T == 1

    if initial_state is None:
        raise unsupported(
            CLASS,
            "a call without initial_state",
            "a bf16 state pool the caller keeps (the slot carries the sequence between calls)",
        )
    _check_pool(initial_state, HV, dev)
    S = int(initial_state.shape[0])
    _check_output(output, v, (q, k, v, g, beta, initial_state))
    sp = get_pool(initial_state)
    slot_ids = _slot_ids(ssm_state_indices, N, sp)
    out = output if output is not None else torch.empty_like(v)
    rows = out.view(-1, HV, HEAD_DIM)
    packed = _pack(q, k, v, g, beta, HV)
    dt = dt_bias.reshape(HV, HEAD_DIM)

    if decode:
        q_t, k_t, v_t, g_t, beta_t = packed
        sp.step(
            initial_state,
            slot_ids,
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            A_log,
            dt,
            lower_bound,
            scale_f,
            rows,
        )
        return out, (initial_state if output_final_state else None)

    if cu_seqlens is not None:
        _no_capture("a prefill with cu_seqlens")
        cu: List[int] = [int(x) for x in cu_seqlens.tolist()]
        lens = [cu[i + 1] - cu[i] for i in range(N)]
        if cu[0] != 0 or cu[-1] != T or any(n <= 0 for n in lens):
            raise unsupported(
                CLASS,
                "cu_seqlens that are not increasing from 0 to the token count",
                "nonempty sequences tiling the tokens",
            )
        cu_dev = cu_seqlens.to(torch.int32).contiguous()
    else:
        cu_dev = sp.uniform_cu(N, T)
    if ssm_state_indices is not None:
        _check_host_slots(slot_ids, S)
    sp.prefill(
        initial_state,
        slot_ids,
        packed,
        cu_dev,
        rows,
        A_log,
        dt,
        lower_bound,
        scale_f,
        fresh=True,
    )
    return out, (initial_state if output_final_state else None)


def recurrent_kda_class_s_reset(
    initial_state: torch.Tensor, ssm_state_indices: torch.Tensor
) -> None:
    """Take the pool states of the listed slots (int32 ``[n]``) as chunk boundaries and drop their open rows: for a
    caller that wrote a slot's state in a way the stamp cannot see (see the module docstring)."""
    if not isinstance(initial_state, torch.Tensor) or initial_state.ndim != 4:
        raise unsupported(CLASS, "a state that is not a pool", "[slots, HV, 128, 128]")
    _check_device(initial_state.device)
    _check_pool(initial_state, int(initial_state.shape[1]), initial_state.device)
    sp = get_pool(initial_state)
    if ssm_state_indices is None:
        raise unsupported(CLASS, "a reset without slots", "int32 [n] slot indices")
    slot_ids = _slot_ids(ssm_state_indices, ssm_state_indices.numel(), sp)
    _check_host_slots(slot_ids, sp.slots)
    sp.reset(slot_ids)


def recurrent_kda_class_s_extend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    lower_bound: float,
    scale: Optional[float] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``[N, T, HV, 128]`` tokens (``beta`` ``[N, T, HV]``) appended to each slot's open chunk, where
    ``recurrent_kda``'s ``[B, T > 1]`` form would start a prefill from the slot's state: the next piece of a chunked
    prefill cut at any token, or a multi-token decode step. Returns the bf16 ``[N, T, HV, 128]`` rows; row ``t`` of a
    sequence at position ``p`` is row ``p + t`` of its prefill. One launch of the prefill kernel, which reads the
    open-row counts on the device and folds the chunks it completes. Caller-given slots are read back and checked, so
    only a call on the default slots can be captured in a CUDA graph (after one eager call of the same shape)."""
    HV = _check_inputs(q, k, v, g, beta)
    _check_params(A_log, dt_bias, lower_bound, HV)
    dev = q.device
    _check_device(dev)
    if A_log.device != dev or dt_bias.device != dev:
        raise unsupported(CLASS, "inputs on several devices", "one device")
    _check_pool(initial_state, HV, dev)
    N, T = q.shape[:2]
    if N == 0 or T == 0:
        raise unsupported(CLASS, "an empty call", "at least one token per sequence")
    S = int(initial_state.shape[0])
    _check_output(output, v, (q, k, v, g, beta, initial_state))
    sp = get_pool(initial_state)
    slot_ids = _slot_ids(ssm_state_indices, N, sp)
    if ssm_state_indices is not None:
        _check_host_slots(slot_ids, S)
    out = output if output is not None else torch.empty_like(v)
    sp.prefill(
        initial_state,
        slot_ids,
        _pack(q, k, v, g, beta, HV),
        sp.uniform_cu(N, T),
        out.view(-1, HV, HEAD_DIM),
        A_log,
        dt_bias.reshape(HV, HEAD_DIM),
        lower_bound,
        HEAD_DIM**-0.5 if scale is None else float(scale),
        fresh=False,
    )
    return out
