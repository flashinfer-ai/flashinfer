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

A new sequence loaded into a slot must therefore start with a prefill call or carry ``prefix_lens``; a one-token call
without ``prefix_lens`` always continues what the slot holds. As for the operator's other state-pool routes, every slot
index must be unique and in range, and ``cu_seqlens_cpu`` must agree with ``cu_seqlens``; a decode call's ``cu_seqlens``
are taken to be ``arange(N + 1)``. None of this is checked on the device (a decode step and a prefill of whole chunks
read nothing back); the calls that read the slots and offsets on the host check them there. ``output_final_state=True`` returns the pool, whose slots hold the
states at the last chunk boundaries.

Every unsupported argument raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..support import check_device, unsupported
from .pool import get_pool
from .spec import CHUNK, CLASS, GDN_ARCHS, HEAD_DIM

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
    if HQ == 0 or HV % HQ:
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


def _check_pool(pool: Optional[torch.Tensor], HV: int, device: torch.device) -> None:
    if pool is None:
        raise unsupported(
            CLASS,
            "a call without initial_state",
            "an fp32 state pool the caller keeps (the slot carries the sequence between calls)",
        )
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
            f"int32 [{N}] (one slot per sequence)",
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
) -> List[int]:
    if cu_seqlens is None and cu_seqlens_cpu is None:
        return [0, T]
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    cu = [int(x) for x in (src.tolist() if isinstance(src, torch.Tensor) else src)]
    return _checked_offsets(cu, T)


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
    out = output if output is not None else torch.empty_like(v)

    if T == N and prefix_lens is None:
        if cu_seqlens_cpu is not None:
            _checked_offsets([int(x) for x in cu_seqlens_cpu], T)
        gp.step(pool, slots, q, k, v, g, beta, scale_f, out)
    else:
        offs = _host_offsets(cu_seqlens, cu_seqlens_cpu, T)
        cu = cu_seqlens
        if cu is not None and (cu.dtype != torch.int32 or cu.device != dev):
            cu = None
        rows = (q, k, v, g, beta)
        if prefix_lens is None and gp.extend_whole_chunks(
            pool, slots, offs, cu, rows, scale_f, out
        ):
            return (out, pool) if output_final_state else out
        slot_list = slots.tolist()
        if len(set(slot_list)) != N or min(slot_list) < 0 or max(slot_list) >= gp.slots:
            raise unsupported(
                CLASS,
                "state_indices that repeat a slot or leave the pool",
                "one distinct in-range slot per sequence",
            )
        if prefix_lens is None:
            counts = [0] * N
        else:
            prefix = prefix_lens.tolist()
            if any(p < 0 for p in prefix):
                raise unsupported(CLASS, "a negative prefix_lens", "token counts")
            counts = [int(p) % CHUNK for p in prefix]
        cont = [i for i, c in enumerate(counts) if c]
        if cont:
            held = gp.counts([slot_list[i] for i in cont])
            for i, h in zip(cont, held, strict=True):
                if h != counts[i]:
                    raise unsupported(
                        CLASS,
                        f"prefix_lens[{i}] at chunk row {counts[i]} on a slot holding {h} open rows",
                        "a sequence continued where its slot's open chunk ends (resume is exact at multiples of 64)",
                    )
        gp.extend(pool, slots, slot_list, counts, offs, cu, rows, scale_f, out)
    if output_final_state:
        return out, pool
    return out
