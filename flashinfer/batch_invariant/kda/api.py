# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the KDA class: what :func:`flashinfer.recurrent_kda` runs under ``batch_invariant=True``.

The call keeps ``recurrent_kda``'s argument forms. The state is the fp32 pool ``initial_state`` ``[S, HV, 128, 128]``
(the pool the serving frameworks keep for KDA), addressed by ``ssm_state_indices`` (``arange`` when omitted) and
updated in place; a slot holds the sequence's state at its last 16-token boundary, the rows since it live next to the
pool (:mod:`.pool`). Call shapes:

* prefill: ``cu_seqlens`` with more tokens than sequences, or ``[B, T > 1]``; every sequence runs from its slot's state
  as a chunk boundary (a chunked prefill therefore cuts at multiples of 16 tokens from the sequence start, except for
  its last piece): its whole chunks through the chunked prefill kernel, which writes the state at the last boundary
  back, and its tail through the step kernel, which leaves the tail's rows cached for the decode steps that follow;
* decode: ``[B, 1]``, or ``cu_seqlens`` with one token per sequence (the packed decode of the frameworks): one launch
  of the step kernel computes row ``p`` of the sequence from the boundary state and the open chunk's cached rows, and
  the 16th row of a chunk commits the next boundary state;
* a multi-token decode step (several tokens appended to an open chunk) has no spelling in ``recurrent_kda``'s
  arguments yet; :func:`recurrent_kda_batch_invariant_extend` is its entry (one launch of the step kernel for fewer
  than 32 tokens per sequence). The call shape decides for every sequence
  of a call: in a prefill call a sequence with a single token also starts from its slot's state as a boundary;
* speculative decode (``num_spec_tokens``) is not in the class yet and raises.

A prefill into a slot discards whatever open rows the slot had (its state is taken as a chunk boundary), which is why
a chunked prefill cuts at multiples of 16. A slot whose state the caller replaced since the class last wrote it (a new
request loaded into the slot) is recognised by a stamp of its first values and starts a new chunk; what the stamp
cannot see is a state replaced by one with the same first eight values per head while the slot has open rows (a slot
reset after a request shorter than 16 tokens, which never moved its boundary state, or a saved boundary state loaded
back): such a slot's next request must be prefilled with more than one token, since a one-token call is a decode step.
Every slot index must be unique and in range: a prefill checks it, a decode step (which reads nothing back to the
host) does not: its kernel returns NaN rows for a sequence whose slot is outside the pool and writes nothing else for
it, two sequences that name one slot in a step share its state (undefined rows for both), and its ``cu_seqlens`` are taken to be ``arange(N + 1)`` without being read. :func:`recurrent_kda_batch_invariant_reset` is the explicit reset of a recycled slot.

Without ``initial_state`` the call is stateless (the form a trainer's forward pass takes): every sequence is whole and
runs from the zero state through the pinned prefill kernel alone, also a sequence of one token, so no call shape is a
decode step. The rows are the ones the pool calls return for the sequence (prefill, chunked prefill, decode); no state
is read, written, kept or returned, and the rows carry no ``grad_fn``. ``ssm_state_indices`` and
``output_final_state=True`` are refused with it. ``checkpoint_every_n_tokens=16`` makes it also write the state before
every 16-token chunk of every sequence into ``state_checkpoints`` (bf16 or fp32), the series a backward reads.

Every unsupported argument raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from ..support import check_device, unsupported
from .pool import ExactPool, get_pool
from .spec import (
    CHUNK,
    CLASS,
    HEAD_DIM,
    KDA_ARCHS,
    ROUTE,
    append_split,
    heads_supported,
)

_IO = torch.bfloat16
#: The step binding's message for a pointer the kernel cannot load from (``csrc/kda_batch_invariant.cu``).
_MISALIGNED = "does not start at such a boundary"


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
    if dt_bias.dtype != torch.float32 or dt_bias.numel() != HV * HEAD_DIM:
        raise unsupported(
            CLASS,
            f"dt_bias {tuple(dt_bias.shape)} {dt_bias.dtype}",
            f"fp32 [{HV}, {HEAD_DIM}]",
        )
    if (
        lower_bound is None
        or not isinstance(lower_bound, (int, float))
        or not math.isfinite(lower_bound)
        or not lower_bound < 0.0
    ):
        raise unsupported(
            CLASS,
            f"lower_bound={lower_bound!r}",
            "a finite negative gate lower bound (the softplus gate is another kernel)",
        )


def _check_inputs(q, k, v, g, beta) -> Tuple[int, int]:
    """``(H, HV)`` after the dtype and shape checks of the five activations (``[B, T, heads, 128]`` / ``[B, T, HV]``)."""
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
    if not heads_supported(HV):
        raise unsupported(
            CLASS,
            f"{HV} heads",
            "a multiple of 8 heads, or 1 to 10 or 12 heads (what the pinned prefill route carries)",
        )
    return H, HV


def _check_pool(pool: torch.Tensor, HV: int, device: torch.device) -> None:
    if pool.dtype != torch.float32:
        raise unsupported(
            CLASS,
            f"a {pool.dtype} state",
            "an fp32 state pool (the class's boundary state is the kernel's fp32 carry)",
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


def _slot_ids(
    indices: Optional[torch.Tensor], N: int, device: torch.device, S: int
) -> torch.Tensor:
    if indices is None:
        if N > S:
            raise unsupported(
                CLASS,
                f"{N} sequences on a pool of {S} slots",
                "ssm_state_indices or a pool of at least N slots",
            )
        return torch.arange(N, dtype=torch.int32, device=device)
    if indices.dtype != torch.int32 or indices.ndim != 1 or indices.numel() != N:
        raise unsupported(
            CLASS,
            f"ssm_state_indices of shape {tuple(indices.shape)} {indices.dtype}",
            f"int32 [{N}] (one slot per sequence; the [N, 1 + S] form belongs to speculative decode)",
        )
    if indices.device != device:
        raise unsupported(
            CLASS, "ssm_state_indices on another device", "the inputs' device"
        )
    return indices.contiguous()


def _host_slots(slot_ids: torch.Tensor, S: int) -> List[int]:
    """The slot ids on the host, refused unless unique and in range (the decode step does not read them back)."""
    slot_list = slot_ids.tolist()
    if len(set(slot_list)) != len(slot_list) or any(s < 0 or s >= S for s in slot_list):
        raise unsupported(
            CLASS,
            "ssm_state_indices that repeat a slot or leave the pool",
            f"unique slots in [0, {S})",
        )
    return slot_list


def _pack(q, k, v, g, beta, H: int, HV: int) -> Tuple[torch.Tensor, ...]:
    """The token rows packed ``[tokens, heads, 128]`` / ``[tokens, HV]``, contiguous and 16-byte aligned: both kernels
    load them 16 bytes at a time (a beta view at an odd token offset is not aligned off a multiple of 8 heads)."""
    rows = (
        q.reshape(-1, H, HEAD_DIM),
        k.reshape(-1, H, HEAD_DIM),
        v.reshape(-1, HV, HEAD_DIM),
        g.reshape(-1, HV, HEAD_DIM),
        beta.reshape(-1, HV),
    )
    rows = tuple(t.contiguous() for t in rows)
    return tuple(t if t.data_ptr() % 16 == 0 else t.clone() for t in rows)


def _check_output(output: torch.Tensor, v: torch.Tensor) -> None:
    if output.shape != v.shape or output.dtype != _IO or not output.is_contiguous():
        raise unsupported(
            CLASS,
            "an output of another shape, dtype or layout than v",
            "a contiguous bf16 [B, T, HV, 128] output",
        )
    if output.device != v.device:
        raise unsupported(CLASS, "an output on another device", "the inputs' device")


MAX_SEQUENCES = 65535

_FIXED_LAYOUT: Dict[tuple, tuple] = {}


def _fixed_layout(B, T, dev):
    """Offsets, order and checkpoint starts of a ``[B, T]`` call, complete on the device before any stream sees them."""
    key = (B, T, str(dev))
    got = _FIXED_LAYOUT.get(key)
    if got is None:
        per = -(-T // CHUNK)
        got = _FIXED_LAYOUT[key] = (
            torch.arange(0, (B + 1) * T, T, dtype=torch.int64).to(dev),
            torch.arange(B, dtype=torch.int32).to(dev),
            torch.arange(0, (B + 1) * per, per, dtype=torch.int64).to(dev),
        )
    return got


def _refuse_shared(buffer, others, what):
    from ...kda_prefill import _storage_ranges_overlap

    for t in others:
        if t is not None and _storage_ranges_overlap(buffer, t):
            raise unsupported(CLASS, what, "a buffer of its own")


def _refuse_inference(buffer, name):
    if buffer.is_inference() and not torch.is_inference_mode_enabled():
        raise unsupported(
            CLASS,
            f"an inference tensor as {name} outside inference mode",
            "a plain buffer, or the call inside torch.inference_mode()",
        )


def _checkpoints(state_checkpoints, checkpoint_cu_starts, lens, v, others, starts_dev):
    """The caller's buffers for the state before every 16-token chunk of every sequence, checked against the lengths."""
    HV, dev = v.shape[2], v.device
    counts = [-(-n // CHUNK) for n in lens]
    total = sum(counts)
    ck = state_checkpoints
    if ck is None:
        ck = torch.empty(total, HV, HEAD_DIM, HEAD_DIM, dtype=_IO, device=dev)
    elif (
        ck.device != dev
        or ck.dtype not in (_IO, torch.float32)
        or ck.ndim != 4
        or tuple(ck.shape[1:]) != (HV, HEAD_DIM, HEAD_DIM)
        or ck.shape[0] < total
        or not ck.is_contiguous()
        or ck.data_ptr() % 16
        or ck.requires_grad
    ):
        raise unsupported(
            CLASS,
            "state_checkpoints of another shape, dtype, layout, alignment or device, with too few rows, "
            "or requiring grad",
            f"a contiguous bf16 or fp32 [rows, {HV}, 128, 128] buffer with one row per started "
            f"16-token chunk of every sequence ({total} here)",
        )
    else:
        _refuse_inference(ck, "state_checkpoints")
        _refuse_shared(
            ck,
            (*others, checkpoint_cu_starts),
            "state_checkpoints that share memory with an input or the output",
        )
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    cs = checkpoint_cu_starts
    if cs is None:
        cs = starts_dev
        if cs is None:
            cs = torch.tensor(starts, dtype=torch.int64, device=dev)
    elif (
        cs.device != dev
        or cs.dtype != torch.int64
        or cs.ndim != 1
        or not cs.is_contiguous()
        or cs.tolist() != starts
    ):
        raise unsupported(
            CLASS,
            "checkpoint_cu_starts that are not the running count of 16-token chunks",
            "a contiguous int64 [N + 1] CUDA tensor of cumulative ceil(length / 16), or None",
        )
    return ck, cs


def _stateless(
    rows,
    A_log,
    dt_bias,
    lower_bound,
    scale_f,
    cu_seqlens,
    output,
    pool_args,
    checkpoints=(None, None, 0),
):
    """The call without a pool: whole sequences from the zero state through the pinned prefill kernel, which then reads
    and stores no state; rows only, written into ``output`` when given. A packed call is preceded by the kernel's own
    launch that orders the sequences on the device, as in a pool prefill."""
    from ... import kda_prefill_cute

    q, k, v, g, beta = (t.detach() for t in rows)
    dev = v.device
    shape = v.shape
    B, T, HV = shape[:3]
    state_checkpoints, checkpoint_cu_starts, every = checkpoints
    for name, value in pool_args.items():
        if value is not None and not (isinstance(value, bool) and not value):
            raise unsupported(
                CLASS,
                f"{name} without initial_state",
                "an fp32 state pool the caller keeps, or none of the state arguments "
                "(whole sequences from the zero state, rows only)",
            )
    count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if count > MAX_SEQUENCES:
        raise unsupported(
            CLASS,
            f"{count} sequences in one call without initial_state",
            f"at most {MAX_SEQUENCES}",
        )
    inputs = (q, k, v, g, beta, A_log, dt_bias, cu_seqlens)
    if output is not None:
        _check_output(output, v)
        if output.requires_grad:
            raise unsupported(CLASS, "an output that requires grad", "a plain buffer")
        _refuse_inference(output, "output")
        _refuse_shared(
            output,
            (*inputs, checkpoint_cu_starts),
            "an output that shares memory with an input",
        )
    with torch.cuda.device(dev):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        raise unsupported(
            CLASS,
            "a call without initial_state inside a CUDA graph capture",
            "an eager call",
        )
    order = starts_dev = None
    if cu_seqlens is not None:
        offs = cu_seqlens.tolist()
        if (
            offs[0] != 0
            or offs[-1] != T
            or any(a >= b for a, b in zip(offs, offs[1:], strict=False))
        ):
            raise unsupported(
                CLASS,
                "cu_seqlens that are not increasing from 0 to the token count",
                "nonempty sequences tiling the tokens",
            )
        lens = [b - a for a, b in zip(offs, offs[1:], strict=False)]
        cu = cu_seqlens.to(torch.int64).contiguous()
    else:
        lens = None
        cu, order, starts_dev = _fixed_layout(B, T, dev)
    ck = cs = None
    if every:
        if every != CHUNK or not isinstance(every, int):
            raise unsupported(
                CLASS,
                f"checkpoint_every_n_tokens={every!r}",
                f"{CHUNK}, the class's chunk",
            )
        ck, cs = _checkpoints(
            state_checkpoints,
            checkpoint_cu_starts,
            [T] * B if lens is None else lens,
            v,
            (*inputs, output),
            starts_dev,
        )
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise unsupported(
            CLASS,
            "state_checkpoints or checkpoint_cu_starts without checkpoint_every_n_tokens",
            f"checkpoint_every_n_tokens={CHUNK} with them",
        )
    q, k, v, g, beta = (
        t.view(1, B * T, *t.shape[1:]) for t in _pack(q, k, v, g, beta, HV, HV)
    )
    aligned = output is not None and output.data_ptr() % 16 == 0
    out = output if aligned else torch.empty(shape, dtype=_IO, device=dev)
    kernel = kda_prefill_cute._get_compiled_cute_dsl_kda(
        mode=ROUTE,
        lower_bound=float(lower_bound),
        state_dtype=torch.float32 if ck is None else ck.dtype,
        has_state_in=False,
        has_state_out=False,
        has_state_ckpt=ck is not None,
        has_state_indices=False,
        has_checkpoint_state_indices=False,
    )
    ckpt_kwargs = {}
    if ck is not None:
        ckpt_kwargs = dict(
            state_ckpt=ck,
            checkpoint_cu_starts=cs,
            checkpoint_state_indices=None,
            ckpt_interval=CHUNK,
        )
    kernel(
        q,
        k,
        v,
        g,
        A_log.detach().contiguous(),
        dt_bias.detach().reshape(HV, HEAD_DIM).contiguous(),
        beta,
        cu,
        None,
        out.view(1, B * T, HV, HEAD_DIM),
        None,
        None,
        torch.cuda.current_stream(dev).cuda_stream,
        scale_f,
        seq_order=order,
        **ckpt_kwargs,
    )
    # the kernel wrote the caller's buffers behind autograd's back
    if output is not None and out is not output:
        output.copy_(out)
    elif output is not None and not output.is_inference():
        torch.autograd.graph.increment_version(output)
    if state_checkpoints is not None and not ck.is_inference():
        torch.autograd.graph.increment_version(ck)
    return (out if output is None else output), ck


def recurrent_kda_batch_invariant(
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
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, ...]:
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
    H, HV = _check_inputs(q, k, v, g, beta)
    _check_params(A_log, dt_bias, lower_bound, HV)
    check_device(q.device, CLASS, KDA_ARCHS)
    dev = q.device
    for t in (k, v, g, beta, A_log, dt_bias):
        if t.device != dev:
            raise unsupported(CLASS, "inputs on several devices", "one device")
    scale_f = HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_f):
        raise unsupported(CLASS, f"scale={scale!r}", "a finite scale")

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
        out, ck = _stateless(
            (q, k, v, g, beta),
            A_log,
            dt_bias,
            lower_bound,
            scale_f,
            cu_seqlens,
            output,
            dict(
                ssm_state_indices=ssm_state_indices,
                output_final_state=output_final_state,
            ),
            (state_checkpoints, checkpoint_cu_starts, checkpoint_every_n_tokens),
        )
        return (out, None) if ck is None else (out, None, ck)
    for name, value in (
        ("state_checkpoints", state_checkpoints),
        ("checkpoint_cu_starts", checkpoint_cu_starts),
        ("checkpoint_every_n_tokens", checkpoint_every_n_tokens or None),
    ):
        if value is not None:
            raise unsupported(
                CLASS,
                f"{name} with a state pool",
                "the call without initial_state (whole sequences from the zero state)",
            )
    _check_pool(initial_state, HV, dev)
    slot_ids = _slot_ids(ssm_state_indices, N, dev, int(initial_state.shape[0]))
    if output is not None:
        _check_output(output, v)
        lo, hi = output.data_ptr(), initial_state.data_ptr()
        if lo < hi + initial_state.numel() * 4 and hi < lo + output.numel() * 2:
            raise unsupported(
                CLASS,
                "an output that shares memory with the state pool",
                "an output buffer of its own",
            )
    ep = get_pool(initial_state)

    if decode:
        out = output
        if out is None:
            out = torch.empty_like(v, memory_format=torch.contiguous_format)
        rows = out.view(N, HV, HEAD_DIM)
        try:
            ep.step(
                initial_state,
                slot_ids,
                q.reshape(N, H, HEAD_DIM).contiguous(),
                k.reshape(N, H, HEAD_DIM).contiguous(),
                v.reshape(N, HV, HEAD_DIM).contiguous(),
                g.reshape(N, HV, HEAD_DIM).contiguous(),
                beta.reshape(N, HV).contiguous(),
                A_log,
                dt_bias,
                lower_bound,
                scale_f,
                rows,
            )
        except RuntimeError as e:
            if _MISALIGNED not in str(e):
                raise
            raise unsupported(
                CLASS,
                "a decode input that does not start at a 16-byte boundary (4 bytes for v)",
                "aligned q / k / v rows (the step loads them 16 and 4 bytes at a time, without a copy)",
            ) from e
        return out, (initial_state if output_final_state else None)

    if cu_seqlens is not None:
        cu = [int(x) for x in cu_seqlens.tolist()]
        lens = [cu[i + 1] - cu[i] for i in range(N)]
        if cu[0] != 0 or cu[-1] != T or any(n <= 0 for n in lens):
            raise unsupported(
                CLASS,
                "cu_seqlens that are not increasing from 0 to the token count",
                "nonempty sequences tiling the tokens",
            )
    else:
        lens = [T] * N
    packed = _pack(q, k, v, g, beta, H, HV)
    rows = _append(
        ep,
        initial_state,
        slot_ids,
        None,
        lens,
        packed,
        A_log,
        dt_bias,
        lower_bound,
        scale,
    )
    if output is not None:
        output.view(-1, HV, HEAD_DIM).copy_(rows)
        out = output
    else:
        out = rows.view(v.shape)
    return out, (initial_state if output_final_state else None)


def _pinned(
    packed,
    cu: torch.Tensor,
    init: torch.Tensor,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    indices: Optional[torch.Tensor] = None,
):
    """The class's term: the chunked prefill on its pinned route, from the fp32 states ``init`` (updated in place; with
    ``indices`` they are the pool slots of the sequences)."""
    from ... import kda_prefill_cute
    from ...kda import recurrent_kda

    q, k, v, g, beta = packed
    dt_bias = dt_bias.reshape(-1, HEAD_DIM)
    with kda_prefill_cute.force_route(ROUTE):
        out, final = recurrent_kda(
            q[None],
            k[None],
            v[None],
            g[None],
            beta[None],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            beta_is_logit=True,
            backend="cute-dsl",
        )[:2]
    if final is None or final.dtype != torch.float32:
        raise RuntimeError("the pinned KDA prefill did not return an fp32 final state")
    return out[0], final


def _append(
    ep: ExactPool,
    pool: torch.Tensor,
    slot_ids: torch.Tensor,
    counts: Optional[List[int]],
    lens: List[int],
    packed,
    A_log,
    dt_bias,
    lower_bound,
    scale,
) -> torch.Tensor:
    """``lens[i]`` packed tokens appended to slot ``slot_ids[i]``, which holds ``counts[i]`` open rows (``None``: a
    prefill, every slot's state is taken as a chunk boundary and its open rows are dropped): the rows up to
    the next boundary in one launch of the step kernel, the whole chunks from it in one pinned call that writes the
    boundary state, the tail in one more launch of the step kernel (:func:`.spec.append_split`). Both kernels give
    the full call's rows by the prefill kernel's causality, the hypothesis the class rests on."""
    dev = pool.device
    slot_list = _host_slots(slot_ids, int(pool.shape[0]))
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    fresh = counts is None
    splits = [
        append_split(0 if fresh else c, n)
        for c, n in zip(counts or lens, lens, strict=True)
    ]
    if fresh:
        # a sequence shorter than a chunk starts with the step kernel: its slot becomes a boundary here
        short = [slot_list[i] for i, (_, w, _) in enumerate(splits) if not w]
        ep.begin(pool, torch.tensor(short, dtype=torch.int32, device=dev))
    scale_f = HEAD_DIM**-0.5 if scale is None else float(scale)
    rows = None
    if any(head or tail for head, _, tail in splits):
        rows = torch.empty_like(packed[2])

    def spans(pieces: List[Tuple[int, int]]) -> None:
        """Rows ``[start, start + n)`` of sequence ``i`` through the step kernel, written into ``rows``: one launch."""
        if any(n for _, n in pieces):
            ep.span(
                pool,
                slot_ids,
                packed,
                rows,
                A_log,
                dt_bias,
                lower_bound,
                scale_f,
                starts=torch.tensor(
                    [s for s, _ in pieces], dtype=torch.int32, device=dev
                ),
                lens=torch.tensor(
                    [n for _, n in pieces], dtype=torch.int32, device=dev
                ),
            )

    spans([(offs[i], head) for i, (head, _, _) in enumerate(splits)])
    whole = [i for i, (_, w, _) in enumerate(splits) if w]
    if whole and all(head == 0 and tail == 0 for head, _, tail in splits):
        # whole chunks from boundaries only: the caller's rows and the pool's slots as they are
        rows, _ = _pinned(
            packed,
            torch.tensor(offs, dtype=torch.int64, device=dev),
            pool,
            A_log,
            dt_bias,
            lower_bound,
            scale,
            indices=slot_ids,
        )
        ep.begin(pool, slot_ids)
    elif whole and _gathers_tokens(splits, int(pool.shape[1])):
        # the whole chunks gathered into one packed input, run on the pool's slots in place
        pieces = [
            (offs[i] + splits[i][0], offs[i] + splits[i][0] + splits[i][1])
            for i in whole
        ]
        src = torch.tensor(
            [r for a, b in pieces for r in range(a, b)], dtype=torch.int64, device=dev
        )
        cu = [0]
        for a, b in pieces:
            cu.append(cu[-1] + b - a)
        idx32 = torch.tensor(
            [slot_list[i] for i in whole], dtype=torch.int32, device=dev
        )
        out, _ = _pinned(
            tuple(x.index_select(0, src) for x in packed),
            torch.tensor(cu, dtype=torch.int64, device=dev),
            pool,
            A_log,
            dt_bias,
            lower_bound,
            scale,
            indices=idx32,
        )
        rows.index_copy_(0, src, out)
        ep.begin(pool, idx32)
    elif whole:
        # the caller's rows as they are: head and tail rows are zero-state segments whose rows are replaced
        cuts, real = [0], []
        for head, w, tail in splits:
            for n, is_whole in ((head, False), (w, True), (tail, False)):
                if n:
                    if is_whole:
                        real.append(len(cuts) - 1)
                    cuts.append(cuts[-1] + n)
        seg = torch.tensor(real, dtype=torch.int64, device=dev)
        idx = slot_ids.long().index_select(
            0, torch.tensor(whole, dtype=torch.int64, device=dev)
        )
        init = torch.zeros(
            (len(cuts) - 1, *pool.shape[1:]), dtype=pool.dtype, device=dev
        )
        init.index_copy_(0, seg, pool.index_select(0, idx))
        full, final = _pinned(
            packed,
            torch.tensor(cuts, dtype=torch.int64, device=dev),
            init,
            A_log,
            dt_bias,
            lower_bound,
            scale,
        )
        heads = [
            r
            for i, (head, _, _) in enumerate(splits)
            for r in range(offs[i], offs[i] + head)
        ]
        if heads:
            hsrc = torch.tensor(heads, dtype=torch.int64, device=dev)
            full.index_copy_(0, hsrc, rows.index_select(0, hsrc))
        rows = full
        pool.index_copy_(0, idx, final.index_select(0, seg))
        ep.begin(pool, idx.to(torch.int32))
    spans([(offs[i] + head + w, tail) for i, (head, w, tail) in enumerate(splits)])
    if rows is None:
        raise RuntimeError("an append call without rows")
    return rows


def _gathers_tokens(splits, HV: int) -> bool:
    """Whether a call with head or tail rows stages its whole chunks by gathering their tokens (true) or by running
    the caller's rows with one fp32 state per segment (false): whichever moves fewer bytes, a function of the shape."""
    token_bytes = sum(w for _, w, _ in splits) * HV * (4 * HEAD_DIM + 1) * 2
    segments = sum(bool(n) for sp in splits for n in sp)
    return token_bytes <= 3 * segments * HV * HEAD_DIM * HEAD_DIM * 4


def recurrent_kda_batch_invariant_reset(
    initial_state: torch.Tensor, ssm_state_indices: Optional[torch.Tensor] = None
) -> None:
    """Take the pool states of the listed slots as chunk boundaries and drop their open rows: the explicit form of
    what the stamp detects, for a caller that recycles a slot (required after a request shorter than 16 tokens).
    ``None`` lists every slot of the pool."""
    check_device(initial_state.device, CLASS, KDA_ARCHS)
    heads = int(initial_state.shape[1]) if initial_state.ndim == 4 else 0
    _check_pool(initial_state, heads, initial_state.device)
    S = int(initial_state.shape[0])
    n = S if ssm_state_indices is None else ssm_state_indices.numel()
    slot_ids = _slot_ids(ssm_state_indices, n, initial_state.device, S)
    _host_slots(slot_ids, S)
    get_pool(initial_state).begin(initial_state, slot_ids)


def recurrent_kda_batch_invariant_extend(
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
) -> torch.Tensor:
    """A multi-token decode step of the class: ``[N, T, HV, 128]`` (``beta`` ``[N, T, HV]``) appended to each slot's
    open chunk, where ``recurrent_kda``'s ``[B, T > 1]`` form would start a prefill from the slot's state. Returns the
    bf16 ``[N, T, HV, 128]`` rows; row ``t`` of a sequence at position ``p`` is row ``p + t`` of its prefill. Fewer
    than 32 rows per sequence are one launch of the step kernel, which reads the open-row counts on the device and
    commits the chunks it completes; longer calls read the counts on the host and run their whole chunks through the
    prefill kernel. Caller-given slots are read back and checked, so only a call on the default slots can be
    captured in a CUDA graph; as in a decode step, the heads of a slot are taken to hold one count."""
    H, HV = _check_inputs(q, k, v, g, beta)
    _check_params(A_log, dt_bias, lower_bound, HV)
    check_device(q.device, CLASS, KDA_ARCHS)
    for t in (k, v, g, beta, A_log, dt_bias):
        if t.device != q.device:
            raise unsupported(CLASS, "inputs on several devices", "one device")
    _check_pool(initial_state, HV, q.device)
    N, T = q.shape[:2]
    if N == 0 or T == 0:
        raise unsupported(CLASS, "an empty call", "at least one token per sequence")
    S = int(initial_state.shape[0])
    slot_ids = _slot_ids(ssm_state_indices, N, q.device, S)
    # caller-given slots are read back and checked; the default arange needs no check, so that call captures
    slot_list = (
        list(range(N)) if ssm_state_indices is None else _host_slots(slot_ids, S)
    )
    ep = get_pool(initial_state)
    packed = _pack(q, k, v, g, beta, H, HV)
    if T < 2 * CHUNK:
        # fewer rows than two chunks: the step kernel alone, one launch, the open-row count read on the device
        rows = torch.empty_like(packed[2])
        ep.span(
            initial_state,
            slot_ids,
            packed,
            rows,
            A_log,
            dt_bias,
            lower_bound,
            HEAD_DIM**-0.5 if scale is None else float(scale),
            stride=T,
        )
        return rows.view(v.shape)
    rows = _append(
        ep,
        initial_state,
        slot_ids,
        ep.counts(initial_state, slot_list),
        [T] * N,
        packed,
        A_log,
        dt_bias,
        lower_bound,
        scale,
    )
    return rows.view(v.shape)
