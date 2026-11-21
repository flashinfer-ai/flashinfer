# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the truncated draw: the support class's kept set, then the draw class over it.

:func:`top_k_sampling_from_probs_batch_invariant`, :func:`top_p_sampling_from_probs_batch_invariant`,
:func:`min_p_sampling_from_probs_batch_invariant`, :func:`top_k_top_p_sampling_from_probs_batch_invariant` and
:func:`top_k_top_p_sampling_from_logits_batch_invariant` are what the FlashInfer entries of those names run under
``batch_invariant=True``: one token per output row from the kept set of ``support.row.v1`` (top-k: the k-th largest
weight, ties kept; top-p: the smallest upward-closed set whose declared fixed-point mass reaches ``p``; min-p:
``w >= mul(w_max, min_p)``; ``filter_apply_order`` the two orders of the support class), drawn by ``sample.icdf.v1``'s
inverse CDF over the canonical sum tree of the kept weights (the dropped ones ``+0``) at the fp32 uniform of the
request-keyed stream (``stream.py``: the row's ``(seed, offset)``, the draw index the Philox counter). The RNG is the
classes' own, as for ``sampling_from_probs``: one key per output row, the slot in the batch enters nothing, and the
uniforms are not those of the operator's own samplers. No class of its own: the kept set is the support class's integer
procedure and the draw is the draw class's reading of the masked tree (``classes.py`` joins the two fingerprints).

:func:`draw_support` is the generic entry the tests drive (``config=`` to force a member of the enumerated space,
``sm_count=`` to ask the dispatcher for another part's pick, ``draw=`` the block index, ``return_logprob`` the drawn
token's log-probability under the full distribution -- ``lse.row``'s ``log_probs`` at the token, what a trainer re-scores
-- and ``return_kept_logprob`` the one renormalised over the kept set, the probability it was drawn with). Every unsupported
input raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import functools

import torch

from ..support import check_no_overlap, out_buffer, scratch, unsupported
from .api import (
    _DTYPE_STR,
    _aligned_rows,
    _dummy,
    _pdl,
    _rows,
    _temperature,
    check_input,
)
from .draw_api import _resolve_keys, _row_keys
from .schedule import BLOCK, SMEM_BUDGET, LSEConfig, device_sm_count, max_vec_elems
from .support_schedule import support_config
from .support_spec import ORDERS, Filters

#: The registry's name of the composition (``classes.py``): the draw class over the support class's kept set.
CLASS = "sample.icdf.v1+support.row.v1"
#: The key reservation of the operator's own entries per output row (``get_seed_and_offset``'s increment: 32 for the
#: rejection samplers, one for min-p) -- the generator-only call shape reserves the pair as the operator's path does.
INCREMENT_PER_ROW = {"top_k": 32, "top_p": 32, "min_p": 1, "top_k_top_p": 32}
#: The compute capabilities the support class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
SUPPORT_ARCHS = ((10, 0), (10, 3), (10, 7))


def _check_input(x: torch.Tensor, name: str) -> int:
    return check_input(x, name, CLASS, SUPPORT_ARCHS)


#: (device, dtype, value, rows) -> [the tensor, the event of its fill (None once that fill is known complete)]
_CONST_ROWS: Dict[tuple, list] = {}
_CONST_ROWS_MAX = 256


def _const_rows(device, dtype: torch.dtype, v, M: int) -> torch.Tensor:
    """The per-row tensor of a Python-scalar filter value, cached per (device, dtype, value, rows): the kernel only reads
    it, so one fill serves every call with that scalar (a fill kernel and an allocation per scalar filter per call before
    2026-11-18). The fill runs on the creating call's stream and the entry carries its event: a call on another stream
    waits on that event, device side, until the fill is known complete, after which the entry is used as it stands (the
    review of #9695 read the tensor from a second stream before its fill had run). Inside a CUDA-graph capture on the
    tensor's device the cache is not consulted: the call fills its own tensor into the graph, as every call did before (an
    event recorded outside a capture cannot be waited on inside it). The cache is bounded (past the bound a call fills a
    fresh tensor)."""
    dev = torch.device(device)
    if dev.index is not None and dev.index != torch.cuda.current_device():
        with torch.cuda.device(dev):
            capturing = torch.cuda.is_current_stream_capturing()
    else:
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        return torch.full((M,), v, dtype=dtype, device=device)
    key = (device, dtype, v, M)
    entry = _CONST_ROWS.get(key)
    if entry is None:
        t = torch.full((M,), v, dtype=dtype, device=device)
        if len(_CONST_ROWS) < _CONST_ROWS_MAX:
            fill_done = torch.cuda.Event()
            fill_done.record(torch.cuda.current_stream(dev))
            _CONST_ROWS[key] = [t, fill_done]
        return t
    t, fill_done = entry
    if fill_done is not None:
        if fill_done.query():
            entry[1] = None  # the fill is complete: no ordering needed from here on
        else:
            torch.cuda.current_stream(dev).wait_event(fill_done)
    return t


@functools.lru_cache(maxsize=None)
def _filters_for(use_k: bool, use_p: bool, use_m: bool, order: str) -> Filters:
    """The filter-set descriptor of a call (which filters, the order): the values are the class's placeholders (the kernel reads
    the per-row tensors), so one object per set -- its construction rounds the placeholders through the exact binary32 model."""
    return Filters(
        top_k=1 if use_k else None,
        top_p=1.0 if use_p else None,
        min_p=0.0 if use_m else None,
        order=order,
    )


def _row_param(
    value,
    M: int,
    lead: Tuple[int, ...],
    device,
    dtype: torch.dtype,
    name: str,
    lo: float = 0.0,
    hi: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], bool]:
    """``None`` -> ``(None, False)``; a scalar or a tensor of one value, of shape ``(M,)`` or of the input's leading
    shape -> ``((M,) tensor of dtype, True)``. The parameters are exact in their dtype (``top_k`` int32, ``top_p`` /
    ``min_p`` the binary32 values the class compares with); an integer ``top_k`` tensor of another width is clamped to
    ``[0, hi]`` (``hi = N``) in its own dtype before the int32 cast, so a value past the int32 range never wraps
    (``k <= 0`` keeps nothing and ``k >= N`` everything: the boundaries are preserved)."""
    if value is None:
        return None, False
    if isinstance(value, torch.Tensor):
        if (
            value.dtype == dtype
            and value.device == device
            and value.dim() == 1
            and value.shape[0] == M
            and value.is_contiguous()
        ):
            return value, True  # already the kernel's form: no copy, no view
        t = value.to(device=device)
        if (
            dtype == torch.int32
            and t.dtype != torch.int32
            and not t.dtype.is_floating_point
            and hi is not None
        ):
            t = t.clamp(0, hi)
        t = t.to(dtype=dtype).reshape(-1)
        if t.numel() == 1 and M != 1:
            t = t.expand(M)
        if t.shape != (M,):
            raise ValueError(
                f"{name} must be a scalar or a tensor of one value, of shape ({M},) or of the input's leading shape "
                f"{lead}; got {tuple(value.shape)}"
            )
        return t.contiguous(), True
    v: Union[int, float]
    if dtype == torch.int32:
        v = int(value)
        if not (-(1 << 31) <= v < (1 << 31)):
            raise ValueError(f"{name}={value!r} out of the int32 range")
    else:
        v = float(value)
        if not (v >= lo) or v != v or v == float("inf"):
            raise ValueError(f"{name}={value!r} must be a finite number >= {lo}")
    return _const_rows(device, dtype, v, M), True


def _source_row_param(value, indices, idx, Msrc: int, name: str):
    """The operator's rule for a per-row filter tensor under an index gather (#5340): one entry per SOURCE row (a row
    of ``probs`` / ``logits``), read at ``indices[i]`` for output ``i``, so outputs that map to the same row share it.
    Gathered here once, so every entry reads output ``i``'s threshold at ``indices[i]`` (an index outside the source
    rows reads row 0: that output is invalid anyway); a tensor of one value broadcasts; without ``indices`` the tensor
    is per row as given."""
    if indices is None or not isinstance(value, torch.Tensor) or value.numel() == 1:
        return value
    t = value.reshape(-1).to(idx.device)
    if t.numel() != Msrc:
        raise ValueError(
            f"{name}: with indices a per-row tensor has one entry per source row ({Msrc}, the operator's rule); got "
            f"{t.numel()}"
        )
    if Msrc == 0:
        return torch.zeros(idx.numel(), dtype=t.dtype, device=t.device)
    return t[idx.clamp(0, Msrc - 1)]


def draw_support(
    x: torch.Tensor,
    seed,
    offset,
    *,
    from_logits: bool = False,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    top_k: Optional[Union[int, torch.Tensor]] = None,
    top_p: Optional[Union[float, torch.Tensor]] = None,
    min_p: Optional[Union[float, torch.Tensor]] = None,
    order: str = "top_k_first",
    indices: Optional[torch.Tensor] = None,
    draw: int = 0,
    return_logprob: bool = False,
    return_kept_logprob: bool = False,
    enable_pdl: bool = False,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
    early_finish: str = "mass",
    count_list: Optional[bool] = None,
    return_valid: bool = True,
    return_mass: bool = True,
    out: Optional[torch.Tensor] = None,
    logprob_out: Optional[torch.Tensor] = None,
    kept_logprob_out: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """One launch of the composition: ``{"samples"`` (int32, ``-1`` for an invalid row), ``"valid"`` (bool, with
    ``return_valid``), ``"mass"`` (with ``return_mass``)
    (fp32: the kept weights' canonical tree), with ``return_logprob`` (logits only) ``"logp"`` (fp32: the drawn token's
    natural log-probability under the FULL distribution at the given temperature, ``mul(add(z, neg(lg2(S))), LN2)`` with
    ``S`` the row's sum tree -- ``lse.row``'s ``log_probs`` at the token, bit for bit; never renormalised over the kept set;
    NaN where invalid) and with ``return_kept_logprob`` (logits only) ``"kept_logp"`` (fp32: the one renormalised over the
    kept set, ``mul(add(z, neg(lg2(mass))), LN2)``, the probability the token was drawn with; NaN where invalid)``}``, each of
    shape ``indices.shape`` or ``x.shape[:-1]``. ``top_k`` / ``top_p`` / ``min_p`` (at least one) are scalars or per-row
    tensors, ``order`` the support class's (``"top_k_first"``: top-p on the top-k kept mass; ``"joint"``: both on the
    row). ``seed`` / ``offset`` are the rows' keys (int64 / uint64 tensors of one value per output row; a single value for
    one row), ``draw`` the block of the stream the uniform is read from. A row whose kept mass is not a positive finite
    number (nothing kept: ``k <= 0``, no positive weight; a non-finite row) is invalid, and so is an output whose index
    lies outside the source rows. Bitwise identical for every ``config`` of the support kernel's space (one row per CTA,
    the staged share within the budget), every batch composition and every ``sm_count``; ``early_finish`` (``"mass"``,
    ``"all"``, ``""``) and ``count_list`` (``None``: the kernel's rule) are the selection's placements, pinned against each
    other by the suite. ``out`` (int32), ``logprob_out`` and ``kept_logprob_out`` (fp32; each
    implies its ``return_*``) are caller-owned buffers of the output shape, written in place and returned as given."""
    from .draw_support_kernel import get_compiled_kernel

    name = "logits" if from_logits else "probs"
    N = _check_input(x, name)
    if order not in ORDERS:
        raise ValueError(f"order={order!r}: one of {ORDERS}")
    if top_k is None and top_p is None and min_p is None:
        raise ValueError(
            "at least one of top_k / top_p / min_p (the plain draw is sampling_from_probs / sampling_from_logits)"
        )
    if logprob_out is not None:
        return_logprob = True
    if kept_logprob_out is not None:
        return_kept_logprob = True
    if (return_logprob or return_kept_logprob) and not from_logits:
        raise ValueError("the drawn token's log-probability is defined for logits")
    if early_finish not in ("", "mass", "all"):
        raise ValueError(
            f"early_finish must be '', 'mass' or 'all', got {early_finish!r}"
        )
    try:
        x2 = _rows(x, N, name)
    except ValueError:
        x2 = x.reshape(-1, N).contiguous()  # a strided view: copied first, a placement
    Msrc = x2.shape[0]
    dev = x.device
    if indices is not None:
        if indices.dtype not in (torch.int32, torch.int64):
            raise unsupported(
                CLASS, f"indices of dtype {indices.dtype}", "int32 or int64 indices"
            )
        lead: Tuple[int, ...] = tuple(indices.shape)
        idx = indices.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
        M = idx.numel()
    else:
        lead = tuple(x.shape[:-1])
        idx = _dummy(dev, torch.int64)
        M = Msrc
    seeds, offs = _row_keys(seed, offset, M, dev)
    top_k_in, top_p_in, min_p_in = (
        v if isinstance(v, torch.Tensor) else None for v in (top_k, top_p, min_p)
    )
    # a per-row filter tensor is one entry per SOURCE row when indices is given (the operator's rule, #5340)
    top_k = _source_row_param(top_k, indices, idx, Msrc, "top_k")
    top_p = _source_row_param(top_p, indices, idx, Msrc, "top_p")
    min_p = _source_row_param(min_p, indices, idx, Msrc, "min_p")
    k_t, use_k = _row_param(top_k, M, lead, dev, torch.int32, "top_k", hi=N)
    p_t, use_p = _row_param(top_p, M, lead, dev, torch.float32, "top_p")
    m_t, use_m = _row_param(min_p, M, lead, dev, torch.float32, "min_p")
    filters = _filters_for(use_k, use_p, use_m, order)
    temp_t, c = _temperature(temperature, M, lead, dev) if from_logits else (None, 0.0)
    elem_bytes = x.element_size()
    if config is None:
        try:
            config = support_config(N, M, elem_bytes, sm_count or device_sm_count(dev))
        except (
            ValueError
        ) as e:  # no member of the space stages this row within the budget
            raise unsupported(
                CLASS, f"a vocabulary of {N} with {elem_bytes}-byte elements", str(e)
            ) from None
    else:
        config.validate(N, elem_bytes)
        if config.rows_per_cta != 1:
            raise ValueError("the truncated draw needs a config with one row per CTA")
        if config.blocks_per_cta(N) * BLOCK * elem_bytes > SMEM_BUDGET:
            raise ValueError(
                "the CTA's share of the row exceeds the staging budget: use a config with more CTAs per row"
            )
        if config.vec_elems > max_vec_elems(N, elem_bytes):
            raise ValueError("config vector width exceeds the row alignment")
    x2 = _aligned_rows(x2, config.vec_elems * elem_bytes)
    ob = out_buffer(out, "out", lead, torch.int32, x, 4)
    tok = (
        ob.view(-1) if ob is not None else torch.empty(M, dtype=torch.int32, device=dev)
    )
    # the kernel's validity word is never read (validity is the token's sign: an invalid row draws -1) and the kept mass
    # only when asked for: the two go to a per-device scratch otherwise, and the call allocates nothing it does not return
    # (a Python-scalar filter is broadcast into a per-row tensor each call; per-row tensors avoid even that)
    valid = scratch(dev, torch.int32, M)
    mass = (
        torch.empty(M, dtype=torch.float32, device=dev)
        if return_mass
        else scratch(dev, torch.float32, M)
    )
    lb = kb = None
    lp = _dummy(dev, torch.float32)
    if return_logprob:
        lb = out_buffer(logprob_out, "logprob_out", lead, torch.float32, x, 4)
        lp = (
            lb.view(-1)
            if lb is not None
            else torch.empty(M, dtype=torch.float32, device=dev)
        )
    lpk = _dummy(dev, torch.float32)
    if return_kept_logprob:
        kb = out_buffer(kept_logprob_out, "kept_logprob_out", lead, torch.float32, x, 4)
        lpk = (
            kb.view(-1)
            if kb is not None
            else torch.empty(M, dtype=torch.float32, device=dev)
        )
    check_no_overlap(
        [("out", ob), ("logprob_out", lb), ("kept_logprob_out", kb)],
        [
            ("x", x),
            ("seed", seeds),
            ("offset", offs),
            ("indices", indices),
            ("temperature", temp_t),
            (
                "temperature (as given)",
                temperature if isinstance(temperature, torch.Tensor) else None,
            ),
            ("top_k", k_t if use_k else None),
            ("top_k (as given)", top_k_in),
            ("top_p", p_t if use_p else None),
            ("top_p (as given)", top_p_in),
            ("min_p", m_t if use_m else None),
            ("min_p (as given)", min_p_in),
        ],
    )
    if M > 0 and (indices is None or Msrc > 0):
        kernel = get_compiled_kernel(
            _DTYPE_STR[x.dtype],
            N,
            config,
            from_logits,
            filters,
            temp_t is not None,
            indices is not None,
            return_logprob,
            bool(enable_pdl),
            early_finish,
            return_kept_logprob,
            count_list=count_list,
        )
        kernel(
            x2,
            seeds,
            offs,
            idx,
            _dummy(dev, torch.int64),  # token ids: the score form's
            k_t if use_k else _dummy(dev, torch.int32),
            p_t if use_p else _dummy(dev, torch.float32),
            m_t if use_m else _dummy(dev, torch.float32),
            temp_t if temp_t is not None else _dummy(dev, torch.float32),
            tok,
            valid,
            lp,
            lpk,
            mass,
            float(c),
            int(draw),
            M,
            Msrc,
            N,
            int(bool(enable_pdl)),
        )
    elif M > 0:
        # a gather from an empty source: every index lies outside the source rows, nothing is read or kept
        tok.fill_(-1)
        if return_mass:
            mass.zero_()
        if return_logprob:
            lp.fill_(float("nan"))
        if return_kept_logprob:
            lpk.fill_(float("nan"))
    res = {"samples": ob if ob is not None else tok.view(lead)}
    if return_mass:
        res["mass"] = mass.view(lead)
    if return_valid:  # a comparison kernel: formed only when asked for
        res["valid"] = (tok != -1).view(lead)
    if return_logprob:
        res["logp"] = lb if lb is not None else lp.view(lead)
    if return_kept_logprob:
        res["kept_logp"] = kb if kb is not None else lpk.view(lead)
    return res


def kept_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    *,
    top_k: Optional[Union[int, torch.Tensor]] = None,
    top_p: Optional[Union[float, torch.Tensor]] = None,
    min_p: Optional[Union[float, torch.Tensor]] = None,
    order: str = "top_k_first",
    return_logprob: bool = False,
    enable_pdl: bool = False,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
    early_finish: str = "mass",
    out: Optional[torch.Tensor] = None,
    logprob_out: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """``log_probs(logits, token_ids, temperature, top_k=, top_p=, min_p=, filter_apply_order=)``: the score form of the
    composition. One launch computes, per row, ``support.row.v1``'s kept set and mass over ``lse.row``'s weights (the
    sampler's selection, verbatim) and returns ``{"kept_logp"`` (fp32, ``logits.shape[:-1]``: the log-probability of
    ``token_ids[r]`` under the kept-set distribution, ``mul(add(z_t, neg(lg2(mass))), LN2)`` -- bit for bit the
    ``kept_logp`` the truncated samplers return for the token they drew; ``-inf`` for a token outside the kept set, NaN for
    an id outside ``[0, N)`` or a row whose kept mass is not a positive finite number) ``[, "logp"]`` (with
    ``return_logprob``: the full distribution's, ``log_probs`` at the token)``}``. One id per row; ``top_k`` / ``top_p`` /
    ``min_p`` (at least one) and ``order`` as :func:`draw_support` takes them. ``out`` / ``logprob_out`` are caller-owned
    fp32 buffers of the row shape, written in place and returned as given."""
    from .draw_support_kernel import get_compiled_kernel

    N = _check_input(logits, "logits")
    if order not in ORDERS:
        raise ValueError(f"order={order!r}: one of {ORDERS}")
    if top_k is None and top_p is None and min_p is None:
        raise ValueError(
            "at least one of top_k / top_p / min_p (the plain log-probability is log_probs without them)"
        )
    if logprob_out is not None:
        return_logprob = True
    try:
        x2 = _rows(logits, N, "logits")
    except ValueError:
        x2 = logits.reshape(
            -1, N
        ).contiguous()  # a strided view: copied first, a placement
    M = x2.shape[0]
    dev = logits.device
    lead: Tuple[int, ...] = tuple(logits.shape[:-1])
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise unsupported(
            CLASS, f"token_ids of dtype {token_ids.dtype}", "int32 or int64 ids"
        )
    if tuple(token_ids.shape) != lead:
        raise ValueError(
            f"one token id per row under a filter (shape {lead}); got {tuple(token_ids.shape)}"
        )
    ids = token_ids.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
    k_t, use_k = _row_param(top_k, M, lead, dev, torch.int32, "top_k", hi=N)
    p_t, use_p = _row_param(top_p, M, lead, dev, torch.float32, "top_p")
    m_t, use_m = _row_param(min_p, M, lead, dev, torch.float32, "min_p")
    filters = _filters_for(use_k, use_p, use_m, order)
    temp_t, c = _temperature(temperature, M, lead, dev)
    elem_bytes = logits.element_size()
    if config is None:
        try:
            config = support_config(N, M, elem_bytes, sm_count or device_sm_count(dev))
        except (
            ValueError
        ) as e:  # no member of the space stages this row within the budget
            raise unsupported(
                CLASS, f"a vocabulary of {N} with {elem_bytes}-byte elements", str(e)
            ) from None
    else:
        config.validate(N, elem_bytes)
        if config.rows_per_cta != 1:
            raise ValueError("the support's score needs a config with one row per CTA")
        if config.blocks_per_cta(N) * BLOCK * elem_bytes > SMEM_BUDGET:
            raise ValueError(
                "the CTA's share of the row exceeds the staging budget: use a config with more CTAs per row"
            )
        if config.vec_elems > max_vec_elems(N, elem_bytes):
            raise ValueError("config vector width exceeds the row alignment")
    x2 = _aligned_rows(x2, config.vec_elems * elem_bytes)
    kb = out_buffer(out, "out", lead, torch.float32, logits, 4)
    lpk = (
        kb.view(-1)
        if kb is not None
        else torch.empty(M, dtype=torch.float32, device=dev)
    )
    lb = None
    lp = _dummy(dev, torch.float32)
    if return_logprob:
        lb = out_buffer(logprob_out, "logprob_out", lead, torch.float32, logits, 4)
        lp = (
            lb.view(-1)
            if lb is not None
            else torch.empty(M, dtype=torch.float32, device=dev)
        )
    # the membership word and the kept mass are not returned: the per-device scratch (dead writes)
    kept = scratch(dev, torch.int32, M)
    mass = scratch(dev, torch.float32, M)
    check_no_overlap(
        [("out", kb), ("logprob_out", lb)],
        [
            ("logits", logits),
            ("token_ids", token_ids),
            ("temperature", temp_t),
            (
                "temperature (as given)",
                temperature if isinstance(temperature, torch.Tensor) else None,
            ),
            ("top_k", k_t if use_k else None),
            ("top_k (as given)", top_k if isinstance(top_k, torch.Tensor) else None),
            ("top_p", p_t if use_p else None),
            ("top_p (as given)", top_p if isinstance(top_p, torch.Tensor) else None),
            ("min_p", m_t if use_m else None),
            ("min_p (as given)", min_p if isinstance(min_p, torch.Tensor) else None),
        ],
    )
    if M > 0:
        kernel = get_compiled_kernel(
            _DTYPE_STR[logits.dtype],
            N,
            config,
            True,
            filters,
            temp_t is not None,
            False,
            return_logprob,
            bool(enable_pdl),
            early_finish,
            True,
            True,
        )
        kernel(
            x2,
            _dummy(dev, torch.int64),  # seed: no stream in the score form
            _dummy(dev, torch.int64),  # offset
            _dummy(dev, torch.int64),  # indices
            ids,
            k_t if use_k else _dummy(dev, torch.int32),
            p_t if use_p else _dummy(dev, torch.float32),
            m_t if use_m else _dummy(dev, torch.float32),
            temp_t if temp_t is not None else _dummy(dev, torch.float32),
            _dummy(
                dev, torch.int32
            ),  # tokens: nothing is drawn (the store is compiled out)
            kept,
            lp,
            lpk,
            mass,
            float(c),
            0,
            M,
            M,
            N,
            int(bool(enable_pdl)),
        )
    res = {"kept_logp": kb if kb is not None else lpk.view(lead)}
    if return_logprob:
        res["logp"] = lb if lb is not None else lp.view(lead)
    return res


def _entry(
    x: torch.Tensor,
    from_logits: bool,
    filters: dict,
    per_row: int,
    indices,
    generator,
    seed,
    offset,
    return_valid: bool,
    enable_pdl,
    config,
    sm_count,
    temperature=None,
    return_logprob: bool = False,
    return_kept_logprob: bool = False,
    out: Optional[torch.Tensor] = None,
    logprob_out: Optional[torch.Tensor] = None,
    kept_logprob_out: Optional[torch.Tensor] = None,
):
    """The public entries' common body: the operator's key resolution and validation, the draw, the output dtype rule
    (the indices' dtype with ``indices``), the outputs in the flags' order: ``samples[, logp][, kept_logp][, valid]``.
    ``out`` takes the samples: an int32 buffer is the kernel's own output; with int64 ``indices`` an int64 buffer is
    filled by a cast from the kernel's int32 tokens (the operator's dtype contract). The two log-probability buffers
    imply their ``return_*`` flags."""
    name = "logits" if from_logits else "probs"
    N = _check_input(x, name)
    if indices is not None:
        if indices.dtype not in (torch.int32, torch.int64):
            raise unsupported(
                CLASS, f"indices of dtype {indices.dtype}", "int32 or int64 indices"
            )
        M = int(indices.numel())
    else:
        M = int(x.numel() // N) if N else 0
    seeds, offs = _resolve_keys(seed, offset, generator, M * per_row, M, x.device)
    out64 = None
    if (
        out is not None
        and indices is not None
        and indices.dtype == torch.int64
        and out.dtype == torch.int64
    ):
        # the operator's contract gives int64 samples with int64 indices: the kernel's int32 tokens are cast into the
        # caller's int64 buffer (an int32 buffer is written directly)
        out64, out = (
            out_buffer(out, "out", tuple(indices.shape), torch.int64, x, 8),
            None,
        )
        check_no_overlap(
            [
                ("out", out64),
                ("logprob_out", logprob_out),
                ("kept_logprob_out", kept_logprob_out),
            ],
            [
                ("x", x),
                ("seed", seeds),
                ("offset", offs),
                ("indices", indices),
                (
                    "temperature",
                    temperature if isinstance(temperature, torch.Tensor) else None,
                ),
                *(
                    (name, filters.get(name))
                    for name in ("top_k", "top_p", "min_p")
                    if isinstance(filters.get(name), torch.Tensor)
                ),
            ],
        )
    if logprob_out is not None:
        return_logprob = True
    if kept_logprob_out is not None:
        return_kept_logprob = True
    res = draw_support(
        x,
        seeds,
        offs,
        from_logits=from_logits,
        temperature=temperature,
        indices=indices,
        return_logprob=return_logprob,
        return_kept_logprob=return_kept_logprob,
        enable_pdl=_pdl(enable_pdl, x.device),
        config=config,
        sm_count=sm_count,
        return_valid=return_valid,
        return_mass=False,
        out=out,
        logprob_out=logprob_out,
        kept_logprob_out=kept_logprob_out,
        **filters,
    )
    samples = res["samples"]
    if out64 is not None:
        samples = out64.copy_(samples)
    elif out is None and indices is not None and indices.dtype != samples.dtype:
        samples = samples.to(
            indices.dtype
        )  # the operator's contract: the output takes the indices' dtype
    outs = [samples]
    if return_logprob:
        outs.append(res["logp"])
    if return_kept_logprob:
        outs.append(res["kept_logp"])
    if return_valid:
        outs.append(res["valid"])
    return outs[0] if len(outs) == 1 else tuple(outs)


def top_k_sampling_from_probs_batch_invariant(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    indices: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.top_k_sampling_from_probs(probs, top_k, indices, generator=, seed=, offset=, return_valid=,
    batch_invariant=True)``: one token per output row from the top-k kept set of ``support.row.v1`` (the k-th largest
    weight is the cut, ties at it are kept; ``k <= 0`` keeps nothing, ``k >= N`` everything), drawn by ``sample.icdf.v1``
    over the kept weights at the fp32 uniform of the request stream keyed by the row's ``(seed, offset)`` -- one key per
    output row (a scalar, a length-one tensor or the pair reserved from the ``generator`` serves a single row; with more
    rows such a shared key raises ``ValueError``). ``probs``: unnormalised weights ``>= 0``, bf16 / f16 / fp32, any
    vocabulary; ``top_k`` an int or an int32 / int64 tensor of one value or one per row; ``indices[i]`` selects the source
    row of output ``i``. Returns int32 tokens (the indices' dtype with ``indices``; ``-1`` for a row whose kept weights
    do not sum to a positive finite number -- nothing kept, no positive weight, a non-finite row -- and for an output
    whose index lies outside the source rows), and with ``return_valid`` the bool mask of the valid rows.
    ``enable_pdl=None`` follows the device; ``config`` / ``sm_count`` force a configuration / another part's dispatcher
    pick (members of the enumerated space: the bits never depend on them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the classes (dtype,
    device, architecture); it never falls back."""
    return _entry(
        probs,
        False,
        {"top_k": top_k},
        INCREMENT_PER_ROW["top_k"],
        indices,
        generator,
        seed,
        offset,
        return_valid,
        enable_pdl,
        config,
        sm_count,
        out=out,
    )


def top_p_sampling_from_probs_batch_invariant(
    probs: torch.Tensor,
    top_p: Union[float, torch.Tensor],
    indices: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.top_p_sampling_from_probs(probs, top_p, ..., batch_invariant=True)``: the draw of
    :func:`top_k_sampling_from_probs_batch_invariant` from the top-p kept set of ``support.row.v1`` -- the smallest
    upward-closed set of weights whose declared fixed-point mass (``trunc(mul(p_j, div(2^44, S)))``, an exact integer sum)
    reaches ``trunc(mul(p, 2^44))``, ties at the cut kept, every positive weight when ``p >= 1`` or when the positive mass
    does not reach the target, zero weights never; ``top_p`` a float or an fp32 tensor of one value or one per row."""
    return _entry(
        probs,
        False,
        {"top_p": top_p},
        INCREMENT_PER_ROW["top_p"],
        indices,
        generator,
        seed,
        offset,
        return_valid,
        enable_pdl,
        config,
        sm_count,
        out=out,
    )


def min_p_sampling_from_probs_batch_invariant(
    probs: torch.Tensor,
    min_p: Union[float, torch.Tensor],
    indices: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.min_p_sampling_from_probs(probs, min_p, ..., batch_invariant=True)``: the draw of
    :func:`top_k_sampling_from_probs_batch_invariant` from the min-p kept set of ``support.row.v1`` -- the weights
    ``w_j >= mul(w_max, min_p)`` with ``w_max`` the row's ``max`` tree; ``min_p`` a float or an fp32 tensor of one value
    or one per row (``0`` keeps every positive weight)."""
    return _entry(
        probs,
        False,
        {"min_p": min_p},
        INCREMENT_PER_ROW["min_p"],
        indices,
        generator,
        seed,
        offset,
        return_valid,
        enable_pdl,
        config,
        sm_count,
        out=out,
    )


def top_k_top_p_sampling_from_probs_batch_invariant(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices, filter_apply_order, ...,
    batch_invariant=True)``: the draw of :func:`top_k_sampling_from_probs_batch_invariant` from the kept set under both
    filters -- ``"top_k_first"``: the top-k cut, then the top-p cut with the kept mass as ``S`` (the masked tree);
    ``"joint"``: both cuts on the whole row, intersected (the support class's two orders; the order is a dimension of the
    fingerprint). ``top_k_max`` of the operator's entry is ignored (one kernel, no fast path)."""
    if filter_apply_order not in ORDERS:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")
    return _entry(
        probs,
        False,
        {"top_k": top_k, "top_p": top_p, "order": filter_apply_order},
        INCREMENT_PER_ROW["top_k_top_p"],
        indices,
        generator,
        seed,
        offset,
        return_valid,
        enable_pdl,
        config,
        sm_count,
        out=out,
    )


def top_k_top_p_sampling_from_logits_batch_invariant(
    logits: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    *,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    return_logprob: bool = False,
    return_kept_logprob: bool = False,
    out: Optional[torch.Tensor] = None,
    logprob_out: Optional[torch.Tensor] = None,
    kept_logprob_out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p, indices, filter_apply_order, ...,
    batch_invariant=True, temperature=, return_logprob=, return_kept_logprob=)``:
    :func:`top_k_top_p_sampling_from_probs_batch_invariant` over the ``lse.row`` weights ``e_j = ex2.approx.ftz((x_j - m)
    c)`` of the row, ``c = log2(e) / temperature`` (``None``, a positive scalar or one per row, as :func:`softmax` takes
    it), without materialising the probabilities; the row's ``S`` is ``lse.row``'s ``s``, so the kept set and the draw are
    those of the row's softmax at that temperature. A row of ``-inf`` logits (or with a NaN) is invalid and draws ``-1``.

    ``return_logprob`` adds the drawn token's natural log-probability under the full distribution at the given
    temperature -- ``lse.row``'s ``log_probs(logits, samples, temperature)`` bit for bit, never renormalised over the kept
    set (what a trainer re-scores); ``return_kept_logprob`` adds the one renormalised over the kept set (the probability
    the token was drawn with); both fp32 of the samples' shape, NaN for an invalid row. Returns the samples alone, or the
    tuple ``(samples[, logp][, kept_logp])``. ``out`` (int32 of the output shape, or int64 with int64 ``indices``),
    ``logprob_out`` and ``kept_logprob_out`` (fp32; each implies its ``return_*``) are caller-owned buffers, written in
    place and returned."""
    if filter_apply_order not in ORDERS:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")
    return _entry(
        logits,
        True,
        {"top_k": top_k, "top_p": top_p, "order": filter_apply_order},
        INCREMENT_PER_ROW["top_k_top_p"],
        indices,
        generator,
        seed,
        offset,
        False,
        enable_pdl,
        config,
        sm_count,
        temperature=temperature,
        return_logprob=return_logprob,
        return_kept_logprob=return_kept_logprob,
        out=out,
        logprob_out=logprob_out,
        kept_logprob_out=kept_logprob_out,
    )
