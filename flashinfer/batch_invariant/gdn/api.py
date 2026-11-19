# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the GDN class: what :func:`flashinfer.chunk_gated_delta_rule` runs under ``batch_invariant=True``.

The call keeps ``chunk_gated_delta_rule``'s packed argument form (``[T, heads, 128]`` rows, ``cu_seqlens``). The state
is the fp32 pool ``initial_state`` ``[S, HV, 128, 128]``, addressed by ``state_indices`` (``arange`` when omitted) and
updated in place; a slot holds the sequence's state at its last 64-token boundary, the rows since it live next to the
pool (:mod:`.pool`). Call shapes:

* decode: one token per sequence. The token is appended to its slot's open chunk and its row returned; the 64th row of
  a chunk commits the next boundary state. Nothing is read back to the host;
* prefill: more tokens than sequences, no ``prefix_lens``. Every sequence runs from its slot's state as a chunk
  boundary (the slot's open rows are dropped), so a chunked prefill cuts at multiples of 64 tokens from the sequence
  start, except for its last piece;
* with ``prefix_lens`` (the tokens of each sequence consumed before the call): a sequence at a multiple of 64 starts
  from its slot's state as a chunk boundary; any other sequence continues its slot's open chunk, whose row count must
  equal ``prefix_lens[i] % 64``. This is the form for multi-token decode steps and for prefill pieces cut anywhere.

A call whose sequences bring fewer than 64 tokens is one launch that reads nothing on the host (pass
``cu_seqlens_cpu``, or fewer than 64 tokens in all, so that the offsets are known without a read); it checks
``prefix_lens`` on the device, and a sequence whose slot holds another open-row count gets NaN rows while its slot is
left as it was. A prefill that leaves an open tail returns its rows and keeps the tail's raw rows next to the pool; the
slot's next call builds its cached rows first (a captured decode step cannot, and returns NaN rows for such a slot: run
the first step after a prefill eagerly). A new sequence loaded into a slot must start with a prefill call or carry
``prefix_lens``; a one-token
call without it always continues what the slot holds. As for the operator's other state-pool routes, every slot index
must be unique and in range, and ``cu_seqlens_cpu`` must agree with ``cu_seqlens``; a decode call's ``cu_seqlens`` are
taken to be ``arange(N + 1)``. The calls that bring whole chunks read the slots on the host and check them there
(not inside a CUDA graph capture, where a whole-chunk prefill trusts them). A call of fewer than 64 tokens without
``cu_seqlens_cpu`` does not check a device ``cu_seqlens``: rows that no sequence covers are left unwritten.
``output`` must not partly overlap an input (not checked). ``output_final_state=True`` returns the pool, whose slots
hold the states at the last chunk boundaries.

Without ``initial_state`` the call is stateless: every sequence is whole and runs from the zero state in one pinned
launch, the rows are the ones the pool calls return for it, and no state is read, written, kept or returned (the form a
trainer's forward pass takes). The rows carry no ``grad_fn``. The other state arguments are refused with it. With
``train_states`` the same launch also stores the state every 64-token chunk starts from in that caller-owned buffer
(:mod:`.train`), in the row layout cuDNN's GDN backward takes as ``state_checkpoints``. Given both
``cu_seqlens`` and ``cu_seqlens_cpu``, the stateless call compares their lengths and runs from the host offsets (copied
to the device once per distinct offsets), so the two cannot disagree in it. Its ``output`` and ``train_states`` may
share no memory with an input, the offsets, each other or a state pool, and ``output``'s autograd version is bumped.

Every unsupported argument raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

import math
import operator
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..support import check_device, unsupported
from .pool import capturing, get_pool, refuse_shared, zero_state_prefill
from .spec import CHUNK, CLASS, GDN_ARCHS, HEAD_DIM, uses_step_kernel

_IO = torch.bfloat16


def _check_inputs(q, k, v, g, beta) -> Tuple[int, int, int]:
    """``(T, HQ, HV)`` after the dtype and shape checks of the five packed inputs."""
    for t, name in ((q, "q"), (k, "k"), (v, "v")):
        if t.ndim != 3 or t.shape[-1] != HEAD_DIM:
            raise unsupported(
                CLASS, f"{name} of shape {tuple(t.shape)}", f"[T, heads, {HEAD_DIM}]"
            )
        if t.dtype != _IO:
            raise unsupported(CLASS, f"{name} dtype {t.dtype}", "bf16 q / k / v")
        if not t.is_contiguous():
            raise unsupported(CLASS, f"a non-contiguous {name}", "contiguous inputs")
    T, HQ = int(q.shape[0]), int(q.shape[1])
    HV = int(v.shape[1])
    if k.shape != q.shape or v.shape[0] != T:
        raise unsupported(
            CLASS, "q / k / v of different token or head counts", "k.shape == q.shape"
        )
    if T == 0:
        raise unsupported(CLASS, "an empty call", "at least one token")
    if HQ == 0 or HV == 0 or HV % HQ:
        raise unsupported(
            CLASS,
            f"{HQ} q heads with {HV} v heads",
            "a v-head count that is a multiple of the q-head count",
        )
    for t, name in ((g, "g"), (beta, "beta")):
        if t is None:
            raise unsupported(
                CLASS, f"a call without {name}", "fp32 log gates g and fp32 beta"
            )
        if t.dtype != torch.float32 or tuple(t.shape) != (T, HV):
            raise unsupported(
                CLASS,
                f"{name} of shape {tuple(t.shape)} {t.dtype}",
                f"fp32 [{T}, {HV}]",
            )
        if not t.is_contiguous():
            raise unsupported(CLASS, f"a non-contiguous {name}", "contiguous inputs")
    return T, HQ, HV


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
                "state_indices or a pool of at least N slots",
            )
        return torch.arange(N, dtype=torch.int32, device=device)
    if indices.dtype != torch.int32 or indices.ndim != 1 or indices.numel() != N:
        raise unsupported(
            CLASS,
            f"state_indices of shape {tuple(indices.shape)} {indices.dtype}",
            f"int32 [{N}] (one slot per sequence; a decode batch passes cu_seqlens = arange(N + 1))",
        )
    if indices.device != device:
        raise unsupported(
            CLASS, "state_indices on another device", "the inputs' device"
        )
    return indices.contiguous()


def _host_offsets(
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_cpu: Optional[Union[Sequence[int], torch.Tensor]],
    T: int,
    integers_only: bool = False,
) -> List[int]:
    if cu_seqlens is None and cu_seqlens_cpu is None:
        return [0, T]
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if integers_only:
        return _checked_offsets(_integers(src), T)
    cu = [int(x) for x in (src.tolist() if isinstance(src, torch.Tensor) else src)]
    return _checked_offsets(cu, T)


def _integers(src: Union[Sequence[int], torch.Tensor]) -> List[int]:
    cu = None
    if isinstance(src, torch.Tensor):
        whole = src.ndim == 1 and not (src.is_floating_point() or src.is_complex())
        if whole and src.dtype != torch.bool:
            cu = src.tolist()
    else:
        try:
            values = list(src)
            if not any(isinstance(x, bool) for x in values):
                cu = [operator.index(x) for x in values]
        except TypeError:
            pass
    if cu is None:
        raise unsupported(
            CLASS, "offsets that are not 1-D integers", "1-D integer offsets"
        )
    return cu


def _checked_offsets(cu: List[int], T: int) -> List[int]:
    if (
        len(cu) < 2
        or cu[0] != 0
        or cu[-1] != T
        or any(b <= a for a, b in zip(cu, cu[1:], strict=False))
    ):
        raise unsupported(
            CLASS,
            "cu_seqlens that are not increasing from 0 to the token count",
            "nonempty sequences tiling the tokens",
        )
    return cu


def _check_output(output: Optional[torch.Tensor], v: torch.Tensor) -> None:
    if output is None:
        return
    if output.shape != v.shape or output.dtype != _IO or not output.is_contiguous():
        raise unsupported(
            CLASS,
            "an output of another shape, dtype or layout than v",
            f"a contiguous bf16 [T, HV, {HEAD_DIM}] output",
        )
    if output.device != v.device:
        raise unsupported(CLASS, "an output on another device", "the inputs' device")
    if output.requires_grad:
        raise unsupported(CLASS, "an output that requires grad", "a plain buffer")


def _stateless(
    rows, scale, output, cu_seqlens, cu_seqlens_cpu, pool_args, train_states=None
):
    """The call without a pool: whole sequences from the zero state, rows only, nothing kept."""
    v = rows[2]
    for name, value in pool_args.items():
        if value is not None and value is not False:
            raise unsupported(
                CLASS,
                f"{name} without initial_state",
                "an fp32 state pool the caller keeps, or none of the state arguments "
                "(whole sequences from the zero state, rows only)",
            )
    _check_output(output, v)
    if capturing(v.device):
        raise unsupported(
            CLASS,
            "a call without initial_state inside a CUDA graph capture",
            "an eager call",
        )
    if cu_seqlens is not None and (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
    ):
        raise unsupported(CLASS, "a malformed cu_seqlens", "a 1-D [N + 1] tensor")
    if (
        output is not None
        and output.is_inference()
        and not torch.is_inference_mode_enabled()
    ):
        raise unsupported(
            CLASS,
            "an output allocated under inference_mode in a call outside it",
            "a buffer allocated in the mode of the call",
        )
    offs = _host_offsets(cu_seqlens, cu_seqlens_cpu, int(v.shape[0]), True)
    if cu_seqlens is not None and cu_seqlens.numel() != len(offs):
        raise unsupported(
            CLASS,
            "cu_seqlens and cu_seqlens_cpu of different lengths",
            "the same offsets on the device and on the host",
        )
    cu = cu_seqlens
    if cu_seqlens_cpu is not None or not (
        cu is not None
        and cu.dtype == torch.int32
        and cu.device == v.device
        and cu.is_contiguous()
    ):
        cu = None
    named = [("cu_seqlens", cu_seqlens), ("cu_seqlens_cpu", cu_seqlens_cpu)]
    named += zip(("q", "k", "v", "g", "beta"), rows, strict=True)
    if output is not None:
        refuse_shared("an output", output, named)
    scale_f = scale if scale is not None and scale != 0.0 else 1.0 / math.sqrt(HEAD_DIM)
    out = output if output is not None else torch.empty_like(v)
    plain = tuple(x.detach() for x in rows)
    if train_states is None:
        zero_state_prefill(plain, cu, offs, scale_f, out)
    else:
        from .train import check_train_states, zero_state_prefill_with_states

        named.append(("output", output))
        check_train_states(train_states, int(v.shape[1]), v.device, offs, named)
        zero_state_prefill_with_states(plain, offs, train_states, scale_f, out)
    if output is not None and not output.is_inference():
        torch.autograd.graph.increment_version(output)
    return out


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
    train_states: Optional[torch.Tensor],
    gate_domain: str,
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
            'use_cp="auto" (the class runs without context parallelism)',
        )
    for name, value in refused.items():
        if value is None or value is False or (isinstance(value, int) and value == 0):
            continue
        raise unsupported(CLASS, f"{name}={value!r}", "the class's own inputs")
    if gate_domain != "log" or not use_qk_l2norm_in_kernel:
        raise unsupported(
            CLASS,
            'gate_domain="linear" or use_qk_l2norm_in_kernel=False',
            'log-domain gates (gate_domain="log") and the in-kernel Q / K l2-norm',
        )
    T, HQ, HV = _check_inputs(q, k, v, g, beta)
    dev = q.device
    check_device(dev, CLASS, GDN_ARCHS)
    if int((torch.version.cuda or "0").split(".")[0]) < 13:
        raise unsupported(
            CLASS,
            f"a torch build for CUDA {torch.version.cuda}",
            "CUDA 13 (the chunked prefill kernel the class is)",
        )
    for t in (k, v, g, beta):
        if t.device != dev:
            raise unsupported(CLASS, "inputs on several devices", "one device")
    if initial_state is None:
        return _stateless(
            (q, k, v, g, beta),
            scale,
            output,
            cu_seqlens,
            cu_seqlens_cpu,
            dict(
                output_state=output_state,
                state_indices=state_indices,
                prefix_lens=prefix_lens,
                output_final_state=output_final_state,
            ),
            train_states,
        )
    if train_states is not None:
        raise unsupported(
            CLASS,
            "train_states with initial_state",
            "the stateless call (no initial_state), whose state series it stores, or a pool call",
        )
    _check_pool(initial_state, HV, dev)
    pool = initial_state
    if output_state is not None and output_state is not pool:
        raise unsupported(
            CLASS,
            "an output_state other than the pool",
            "the pool updated in place (output_state omitted or initial_state itself)",
        )
    _check_output(output, v)
    if cu_seqlens is not None and (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2):
        raise unsupported(CLASS, "a malformed cu_seqlens", "a 1-D [N + 1] tensor")
    if cu_seqlens is not None:
        N = int(cu_seqlens.numel()) - 1
    elif cu_seqlens_cpu is not None:
        N = len(cu_seqlens_cpu) - 1
    else:
        N = 1
    slots = _slot_ids(state_indices, N, dev, int(pool.shape[0]))
    if prefix_lens is not None and (
        prefix_lens.ndim != 1
        or prefix_lens.numel() != N
        or prefix_lens.dtype not in (torch.int32, torch.int64)
    ):
        raise unsupported(
            CLASS, "a malformed prefix_lens", f"an int32 or int64 [{N}] tensor"
        )
    scale_f = scale if scale is not None and scale != 0.0 else 1.0 / math.sqrt(HEAD_DIM)
    gp = get_pool(pool, HQ)
    if gp.HQ != HQ:
        raise unsupported(
            CLASS,
            f"{HQ} q heads on a pool first used with {gp.HQ}",
            "one head geometry per pool",
        )
    if gp.slots < N:
        raise unsupported(
            CLASS,
            f"{N} sequences on a pool of {gp.slots} slots",
            "one distinct slot per sequence",
        )
    out = output if output is not None else torch.empty_like(v)

    rows = (q, k, v, g, beta)
    if T == N and prefix_lens is None:
        if cu_seqlens_cpu is not None:
            _checked_offsets([int(x) for x in cu_seqlens_cpu], T)
        gp.step(pool, slots, *rows, scale_f, out)
        return (out, pool) if output_final_state else out
    cu = cu_seqlens
    if cu is not None and (cu.dtype != torch.int32 or cu.device != dev):
        cu = None
    # fewer than 64 tokens in the call, or per sequence by the host offsets: the step kernel alone, nothing read back
    offs = None
    if cu_seqlens_cpu is not None or T >= CHUNK or cu_seqlens is None:
        offs = _host_offsets(cu_seqlens, cu_seqlens_cpu, T)
    if offs is None or uses_step_kernel(
        max(b - a for a, b in zip(offs, offs[1:], strict=False))
    ):
        if cu is None:
            cu = (
                torch.tensor(offs, dtype=torch.int32, device=dev)
                if offs is not None
                else cu_seqlens.to(device=dev, dtype=torch.int32)
            )
        if prefix_lens is None:
            prefix = gp.zero_prefix[:N]
        else:
            prefix = prefix_lens.to(device=dev, dtype=torch.int32).contiguous()
        gp.rebuild(pool)
        gp.span(pool, slots, cu[:-1], cu[1:], prefix, rows, scale_f, out)
        return (out, pool) if output_final_state else out
    whole = prefix_lens is None and not any(
        (b - a) % CHUNK for a, b in zip(offs, offs[1:], strict=False)
    )
    slot_list = None
    if not (whole and capturing(dev)):
        slot_list = slots.tolist()
        if (
            len(set(slot_list)) != N
            or min(slot_list) < 0
            or max(slot_list) >= int(pool.shape[0])
        ):
            raise unsupported(
                CLASS,
                "state_indices that repeat a slot or leave the pool",
                "one distinct in-range slot per sequence",
            )
    if whole:
        gp.extend_whole_chunks(pool, slots, slot_list, offs, cu, rows, scale_f, out)
        return (out, pool) if output_final_state else out
    if prefix_lens is None:
        counts = [0] * N
    else:
        prefix = prefix_lens.tolist()
        if any(p < 0 for p in prefix):
            raise unsupported(CLASS, "a negative prefix_lens", "token counts")
        counts = [int(p) % CHUNK for p in prefix]
    gp.extend(pool, slots, slot_list, counts, offs, cu, rows, scale_f, out)
    if output_final_state:
        return out, pool
    return out
