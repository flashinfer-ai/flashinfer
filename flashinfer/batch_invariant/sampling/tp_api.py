# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entries of the tensor-parallel protocol of ``lse.row`` (``README.md``, Sampling: the tensor-parallel
protocol).

A rank holds the slot-aligned shard ``shard_logits = logits[..., lo:hi]`` of a vocabulary of ``N`` columns (``lo`` and
``hi`` multiples of 8; the last shard may end at ``N``). The protocol, on every rank::

    p1 = log_probs_tp_partials(x_r, lo, hi, N, T)                  # (M, P_r) fp32: the max tree's in-shard sub-trees
    m  = log_probs_tp_merge(all_gather(p1), bounds, N, kind="max")  # the row maximum: the TP = 1 kernel's bits
    p2 = log_probs_tp_partials(x_r, lo, hi, N, T, m=m)             # the sum tree's sub-trees over ex2((x - m) c)
    s  = log_probs_tp_merge(all_gather(p2), bounds, N, kind="sum")
    logp_r, lse = log_probs_tp(x_r, ids, lo, hi, N, m, s, T)       # logp_r: the token's log-probability where its id
    logp = all_reduce_sum(logp_r)                                   #   lies in [lo, hi), +0 elsewhere: the SUM is exact

:func:`lse_row_tp_stats` runs the two rounds given the trainer's all-gather (``all_gather`` takes the rank's ``(M, Pmax)``
partials -- every rank's padded to ``Pmax = max_r P_r`` with zeros, :func:`max_partials_width` -- and returns ``(M, R,
Pmax)`` in rank order). The merged ``(m, s)`` and every sharded output equal the corresponding bits of the full-row class
(``api.py``) at TP = 1: the partials are the class's own sub-trees and the merge rebuilds its trees, so the outputs carry
the class's fingerprint (``fingerprint("log_probs", N=, dtype=, temperature=)``); the wire format is a layout, not a
class. :func:`tp_simulate` runs every rank of a tiling on one device from a full logits tensor (the tests). Every
unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from ..support import check_device, unsupported
from .api import CLASS, LSE_ROW_ARCHS, _DTYPE_STR, _dummy, _pdl, _rows, _temperature
from .schedule import V, device_sm_count, max_vec_elems
from .shards import ShardLayout, check_bounds, partials_width
from .tp_schedule import TPConfig, tp_config

_tables: dict = {}
Bounds = Sequence[Tuple[int, int]]


def _slot_table(lay: ShardLayout, device) -> torch.Tensor:
    key = ("slots", lay, str(device))
    if key not in _tables:
        flat = [s for lane in lay.slot_table() for lv in lane for s in lv]
        _tables[key] = torch.tensor(flat, dtype=torch.int32, device=device)
    return _tables[key]


def _base_table(layouts: Sequence[ShardLayout], device) -> torch.Tensor:
    key = ("bases", tuple(layouts), str(device))
    if key not in _tables:
        flat = [b for lay in layouts for b in lay.base[:32]]
        _tables[key] = torch.tensor(flat, dtype=torch.int32, device=device)
    return _tables[key]


def _layout(lo: int, hi: int, N: int) -> ShardLayout:
    """The shard's layout, or the class's refusal: shards are slot aligned (multiples of 8; the last may end at ``N``)."""
    lo, hi, N = int(lo), int(hi), int(N)
    if not (0 < N < (1 << 31)):
        raise unsupported(CLASS, f"a vocabulary of {N}", "0 < vocabulary < 2^31")
    if not (0 <= lo <= hi <= N):
        raise ValueError(f"shard [{lo}, {hi}) is not inside the vocabulary [0, {N})")
    if lo % V or (hi % V and hi != N):
        raise unsupported(
            CLASS,
            f"the shard [{lo}, {hi}) of {N}",
            "shards on multiples of 8 columns (the last may end at the vocabulary)",
        )
    return ShardLayout(lo, hi, N)


def _bounds(bounds: Bounds, N: int) -> Tuple[Tuple[int, int], ...]:
    """The ranks' shards as a contiguous slot-aligned tiling of ``[0, N)``, or the class's refusal."""
    b = tuple((int(lo), int(hi)) for lo, hi in bounds)
    try:
        check_bounds(b, int(N))
    except ValueError as e:
        raise unsupported(
            CLASS,
            f"the shards {b} of {N}",
            f"a contiguous tiling of [0, N) on multiples of 8 columns ({e})",
        ) from None
    return b


def _shard_rows(
    shard_logits: torch.Tensor, lay: ShardLayout
) -> Tuple[torch.Tensor, int, int]:
    """``(rows view, M, vector width)``: the shard as ``(M, hi - lo)`` with a unit column stride; the vector width is the
    largest that keeps every row's loads aligned (the data pointer and the row stride: a view into the full row of a
    simulation, or the trainers' last shard of a vocabulary that is not a multiple of 8, may force a narrow one)."""
    check_device(shard_logits.device, CLASS, LSE_ROW_ARCHS)
    if shard_logits.dtype not in _DTYPE_STR:
        raise unsupported(
            CLASS, f"logits dtype {shard_logits.dtype}", "bf16, f16 or fp32 logits"
        )
    if shard_logits.dim() < 1:
        raise unsupported(
            CLASS,
            f"a {shard_logits.dim()}-D input",
            "shard logits of shape (..., hi - lo)",
        )
    cols = lay.hi - lay.lo
    if shard_logits.shape[-1] != cols:
        raise ValueError(
            f"shard_logits has {shard_logits.shape[-1]} columns; the shard [{lay.lo}, {lay.hi}) has {cols}"
        )
    if cols == 0:
        # an empty shard (a rank without vocabulary): one dummy column so the kernels have a tensor; no predicate admits it
        M = shard_logits.shape[:-1].numel()
        return (
            torch.zeros(M, 1, dtype=shard_logits.dtype, device=shard_logits.device),
            M,
            1,
        )
    try:
        x2 = _rows(shard_logits, cols, "shard_logits")
    except ValueError:
        x2 = shard_logits.reshape(
            -1, cols
        )  # a copy: the input could not be viewed as rows
    M = x2.shape[0]
    eb = shard_logits.element_size()
    vec = max_vec_elems(cols, eb)
    while vec > 1 and (x2.data_ptr() % (vec * eb) or (M > 1 and x2.stride(0) % vec)):
        vec //= 2
    return x2, M, vec


def max_partials_width(bounds: Bounds, N: int) -> int:
    """``Pmax``: the width every rank pads its partials to before the all-gather (``log_probs_tp_partials(width=Pmax)``)."""
    return max(partials_width(lo, hi, int(N)) for lo, hi in _bounds(bounds, N))


def log_probs_tp_partials_batch_invariant(
    shard_logits: torch.Tensor,
    shard_start: int,
    shard_end: int,
    vocab_size: int,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    m: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    width: Optional[int] = None,
    config: Optional[TPConfig] = None,
    sm_count: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.log_probs_tp_partials``: a rank's partials of the class ``lse.row`` over its shard.

    Round 1 (``m is None``): the max tree's maximal complete sub-trees inside the shard, ``(..., P)`` fp32 with
    ``P = partials_width(shard_start, shard_end, vocab_size)`` (or ``width >= P``, the tail zero). Round 2 (``m`` the merged
    row maximum, fp32 of the leading shape): the sum tree's sub-trees over ``ex2((x - m) c)``. The temperature enters
    as in the row class (``c``). Every member of the configuration space writes the identical tensor: the wire format
    is the shard's (``shards.ShardLayout``: the dyadic pieces of each lane's in-shard block range, in block order)."""
    from .tp_kernel import get_partials_kernel

    lay = _layout(shard_start, shard_end, vocab_size)
    x2, M, vec = _shard_rows(shard_logits, lay)
    dev = shard_logits.device
    lead = tuple(shard_logits.shape[:-1])
    W = lay.P if width is None else int(width)
    if W < lay.P:
        raise ValueError(f"width {W} < the shard's partial count {lay.P}")
    if out is None:
        out = torch.zeros(M, W, dtype=torch.float32, device=dev)
    elif out.shape != (M, W) or out.dtype != torch.float32 or out.stride(-1) != 1:
        raise ValueError(
            f"out must be a ({M}, {W}) fp32 tensor with a unit column stride"
        )
    if M == 0 or lay.P == 0:
        return out.view(*lead, W)
    temp_t, c = _temperature(temperature, M, dev)
    if config is None:
        config = tp_config(
            lay,
            M,
            shard_logits.element_size(),
            sm_count or device_sm_count(dev),
            vec_max=vec,
        )
    else:
        config.validate(lay)
    if config.vec_elems > vec:
        raise ValueError("config vector width exceeds the shard's alignment")
    kind = "max" if m is None else "sum"
    if m is None:
        m_t = _dummy(dev, torch.float32)
    else:
        m_t = m.reshape(-1).to(device=dev, dtype=torch.float32).contiguous()
        if m_t.shape != (M,):
            raise ValueError(
                f"m must hold one value per row ({M}); got {tuple(m.shape)}"
            )
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    kernel = get_partials_kernel(
        _DTYPE_STR[shard_logits.dtype],
        lay,
        config,
        kind,
        temp_t is not None,
        _pdl(enable_pdl, dev),
    )
    kernel(x2, out, m_t, temp_arg, _slot_table(lay, dev), float(c), M)
    return out.view(*lead, W)


def log_probs_tp_merge_batch_invariant(
    partials: torch.Tensor,
    bounds: Bounds,
    vocab_size: int,
    *,
    kind: str,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.log_probs_tp_merge``: the row statistic from every rank's partials.

    ``partials`` is ``(..., R, P)`` (rank ``r``'s ``(..., P_r)`` in ``[..., r, :P_r]``, ``P >= max_r P_r``) in the order of
    ``bounds`` (the contiguous slot-aligned tiling of ``[0, vocab_size)``); ``kind`` is ``"max"`` (round 1: the row maximum)
    or ``"sum"`` (round 2: the row sum). fp32 of the leading shape, identical on every rank, and bit for bit the row
    class's statistic."""
    from .tp_kernel import get_merge_kernel

    b = _bounds(bounds, vocab_size)
    if kind not in ("max", "sum"):
        raise ValueError(f"kind must be 'max' or 'sum'; got {kind!r}")
    check_device(partials.device, CLASS, LSE_ROW_ARCHS)
    if (
        partials.dim() < 2
        or partials.shape[-2] != len(b)
        or partials.dtype != torch.float32
    ):
        raise ValueError(
            f"partials must be an (..., {len(b)}, P) fp32 tensor; got {tuple(partials.shape)} {partials.dtype}"
        )
    lead = partials.shape[:-2]
    R, P = int(partials.shape[-2]), int(partials.shape[-1])
    M = int(lead.numel())
    layouts = [ShardLayout(lo, hi, int(vocab_size)) for lo, hi in b]
    if max(lay.P for lay in layouts) > P:
        raise ValueError(
            f"P = {P} < the widest shard's partial count {max(lay.P for lay in layouts)}"
        )
    p2 = partials.reshape(M, R, P).contiguous().view(M, R * P)
    dev = partials.device
    if out is None:
        out = torch.empty(M, dtype=torch.float32, device=dev)
    if M == 0:
        return out.view(lead)
    kernel = get_merge_kernel(b, int(vocab_size), kind, _pdl(enable_pdl, dev))
    kernel(p2, _base_table(layouts, dev), out.reshape(-1), P, M)
    return out.view(lead)


def _stats(m: torch.Tensor, s: torch.Tensor, M: int, dev):
    mt = m.reshape(-1).to(device=dev, dtype=torch.float32).contiguous()
    st = s.reshape(-1).to(device=dev, dtype=torch.float32).contiguous()
    if mt.shape != (M,) or st.shape != (M,):
        raise ValueError(f"m and s must hold one value per row ({M})")
    return mt, st


def log_probs_tp_batch_invariant(
    shard_logits: torch.Tensor,
    token_ids: torch.Tensor,
    shard_start: int,
    shard_end: int,
    vocab_size: int,
    m: torch.Tensor,
    s: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    return_lse: bool = False,
    enable_pdl: Optional[bool] = None,
):
    """``flashinfer.sampling.log_probs_tp``: the token's log-probability on a shard from the merged statistics.

    fp32 of the leading shape: the class's ``logp = ((x_t - m) c - lg2(s)) ln2`` for the rows whose id lies in
    ``[shard_start, shard_end)``, exactly ``+0`` for the other ids in ``[0, vocab_size)`` (the ranks' SUM is then exact:
    one non-zero term per row; ``-0.0`` is unreachable for a log-probability of this class) and NaN for an id outside
    ``[0, vocab_size)``; with ``return_lse`` also the row's ``lse = (m c + lg2(s)) ln2``, identical on every rank."""
    from .tp_kernel import get_outputs_kernel

    lay = _layout(shard_start, shard_end, vocab_size)
    x2, M, vec = _shard_rows(shard_logits, lay)
    dev = shard_logits.device
    lead = tuple(shard_logits.shape[:-1])
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise unsupported(
            CLASS, f"token_ids of dtype {token_ids.dtype}", "int32 or int64 ids"
        )
    if tuple(token_ids.shape) != lead:
        raise ValueError(
            f"token_ids must have one id per row (shape {lead}); got {tuple(token_ids.shape)}"
        )
    lp = torch.empty(lead, dtype=torch.float32, device=dev)
    lse = torch.empty(lead, dtype=torch.float32, device=dev)
    if M == 0:
        return (lp, lse) if return_lse else lp
    temp_t, c = _temperature(temperature, M, dev)
    mt, st = _stats(m, s, M, dev)
    tok = token_ids.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    kernel = get_outputs_kernel(
        _DTYPE_STR[shard_logits.dtype],
        lay,
        "logp",
        temp_t is not None,
        vec,
        _pdl(enable_pdl, dev),
    )
    y = _dummy(dev, torch.float32).view(1, 8)
    kernel(
        x2,
        y,
        mt,
        st,
        tok,
        lp.reshape(-1),
        lse.reshape(-1),
        temp_arg,
        float(c),
        M,
        int(vocab_size),
    )
    return (lp, lse) if return_lse else lp


def _row_output(mode: str, shard_logits, lo, hi, N, m, s, temperature, out, enable_pdl):
    from .tp_kernel import get_outputs_kernel

    lay = _layout(lo, hi, N)
    x2, M, vec = _shard_rows(shard_logits, lay)
    dev = shard_logits.device
    out_dtype = shard_logits.dtype if mode == "probs" else torch.float32
    if out is None:
        out = torch.empty(shard_logits.shape, dtype=out_dtype, device=dev)
    if out.shape != shard_logits.shape or out.dtype != out_dtype:
        raise ValueError(
            "out must have the shard's shape"
            + (" and the input dtype" if mode == "probs" else " and dtype fp32")
        )
    if M == 0 or lay.lo == lay.hi:
        return out
    y2 = _rows(out, lay.hi - lay.lo, "out")
    vec_out = min(vec, max(1, 16 // out.element_size()))
    while vec > 1 and (
        y2.data_ptr() % (vec_out * out.element_size())
        or (M > 1 and y2.stride(0) % vec_out)
    ):
        vec //= 2
        vec_out = min(vec, max(1, 16 // out.element_size()))
    temp_t, c = _temperature(temperature, M, dev)
    mt, st = _stats(m, s, M, dev)
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    kernel = get_outputs_kernel(
        _DTYPE_STR[shard_logits.dtype],
        lay,
        mode,
        temp_t is not None,
        vec,
        _pdl(enable_pdl, dev),
    )
    kernel(
        x2,
        y2,
        mt,
        st,
        _dummy(dev, torch.int64),
        _dummy(dev, torch.float32),
        _dummy(dev, torch.float32),
        temp_arg,
        float(c),
        M,
        int(N),
    )
    return out


def log_softmax_tp_batch_invariant(
    shard_logits: torch.Tensor,
    shard_start: int,
    shard_end: int,
    vocab_size: int,
    m: torch.Tensor,
    s: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.log_softmax_tp``: the shard's columns of the natural log-softmax row (fp32), bit for bit the
    full row's ``log_softmax(...)[..., shard_start:shard_end]``."""
    return _row_output(
        "logprobs",
        shard_logits,
        shard_start,
        shard_end,
        vocab_size,
        m,
        s,
        temperature,
        out,
        enable_pdl,
    )


def softmax_tp_batch_invariant(
    shard_logits: torch.Tensor,
    shard_start: int,
    shard_end: int,
    vocab_size: int,
    m: torch.Tensor,
    s: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.softmax_tp``: the shard's columns of the softmax row (input dtype), bit for bit the full row's
    ``softmax(..., batch_invariant=True)[..., shard_start:shard_end]``."""
    return _row_output(
        "probs",
        shard_logits,
        shard_start,
        shard_end,
        vocab_size,
        m,
        s,
        temperature,
        out,
        enable_pdl,
    )


def lse_row_tp_stats(
    shard_logits: torch.Tensor,
    shard_start: int,
    shard_end: int,
    vocab_size: int,
    bounds: Bounds,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    config: Optional[TPConfig] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two rounds on this rank: ``(m, s)`` fp32 of the leading shape. ``all_gather`` maps this rank's ``(..., Pmax)``
    partials to the ``(..., R, Pmax)`` stack of every rank's in rank order (the trainer's collective; the default serves a
    single rank)."""
    b = _bounds(bounds, vocab_size)
    if all_gather is None:
        if len(b) != 1:
            raise ValueError("all_gather is required for more than one rank")
        all_gather = lambda p: p.unsqueeze(-2)  # noqa: E731
    Pmax = max_partials_width(b, vocab_size)
    p1 = log_probs_tp_partials_batch_invariant(
        shard_logits,
        shard_start,
        shard_end,
        vocab_size,
        temperature,
        width=Pmax,
        config=config,
        enable_pdl=enable_pdl,
    )
    m = log_probs_tp_merge_batch_invariant(
        all_gather(p1), b, vocab_size, kind="max", enable_pdl=enable_pdl
    )
    p2 = log_probs_tp_partials_batch_invariant(
        shard_logits,
        shard_start,
        shard_end,
        vocab_size,
        temperature,
        m=m,
        width=Pmax,
        config=config,
        enable_pdl=enable_pdl,
    )
    s = log_probs_tp_merge_batch_invariant(
        all_gather(p2), b, vocab_size, kind="sum", enable_pdl=enable_pdl
    )
    return m, s


def tp_simulate(
    logits: torch.Tensor,
    bounds: Bounds,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    token_ids: Optional[torch.Tensor] = None,
    *,
    config: Optional[TPConfig] = None,
    outputs: Sequence[str] = ("logp", "logprobs", "probs"),
    enable_pdl: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """Every rank of the tiling on one device, from the full ``(M, N)`` logits: the ranks' partials are stacked in place of
    the all-gather, the token log-probabilities summed in place of the all-reduce. Returns ``m``, ``s``, the padded
    partials of both rounds (``(M, R, Pmax)``) and the requested outputs assembled from the shards (``logp`` summed,
    ``lse`` from rank 0, ``logprobs`` / ``probs`` concatenated)."""
    N = int(logits.shape[-1])
    b = _bounds(bounds, N)
    x2 = logits.reshape(-1, N)
    Pmax = max_partials_width(b, N)
    shards_ = [x2[:, lo:hi] for lo, hi in b]
    p1 = torch.stack(
        [
            log_probs_tp_partials_batch_invariant(
                xs,
                lo,
                hi,
                N,
                temperature,
                width=Pmax,
                config=config,
                enable_pdl=enable_pdl,
            )
            for xs, (lo, hi) in zip(shards_, b, strict=True)
        ],
        dim=1,
    )
    m = log_probs_tp_merge_batch_invariant(p1, b, N, kind="max", enable_pdl=enable_pdl)
    p2 = torch.stack(
        [
            log_probs_tp_partials_batch_invariant(
                xs,
                lo,
                hi,
                N,
                temperature,
                m=m,
                width=Pmax,
                config=config,
                enable_pdl=enable_pdl,
            )
            for xs, (lo, hi) in zip(shards_, b, strict=True)
        ],
        dim=1,
    )
    s = log_probs_tp_merge_batch_invariant(p2, b, N, kind="sum", enable_pdl=enable_pdl)
    res = {"m": m, "s": s, "partials_max": p1, "partials_sum": p2}
    if "logp" in outputs and token_ids is not None:
        ids = token_ids.reshape(-1)
        lp_sum, lse = None, None
        for xs, (lo, hi) in zip(shards_, b, strict=True):
            lp, lse_r = log_probs_tp_batch_invariant(
                xs,
                ids,
                lo,
                hi,
                N,
                m,
                s,
                temperature,
                return_lse=True,
                enable_pdl=enable_pdl,
            )
            lp_sum = (
                lp if lp_sum is None else lp_sum + lp
            )  # the all-reduce SUM: every other term is +0
            lse = lse_r if lse is None else lse
        res["logp"], res["lse"] = lp_sum, lse
    if "logprobs" in outputs:
        res["logprobs"] = torch.cat(
            [
                log_softmax_tp_batch_invariant(
                    xs, lo, hi, N, m, s, temperature, enable_pdl=enable_pdl
                )
                for xs, (lo, hi) in zip(shards_, b, strict=True)
            ],
            dim=-1,
        )
    if "probs" in outputs:
        res["probs"] = torch.cat(
            [
                softmax_tp_batch_invariant(
                    xs, lo, hi, N, m, s, temperature, enable_pdl=enable_pdl
                )
                for xs, (lo, hi) in zip(shards_, b, strict=True)
            ],
            dim=-1,
        )
    return res
