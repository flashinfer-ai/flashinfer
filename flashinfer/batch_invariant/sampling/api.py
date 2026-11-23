# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the ``lse.row`` class.

:func:`softmax_batch_invariant` is what :func:`flashinfer.sampling.softmax` runs under ``batch_invariant=True``:
``softmax(logits / temperature)`` along the last dimension in the input dtype (or in fp32 with ``out_dtype``), with
FlashInfer's argument forms (temperature ``None``, a scalar or a per-row tensor of f16 / bf16 / f32 / f64 of shape
``logits.shape[:-1]`` or ``(rows,)``; a scalar is validated on the host, a tensor's values are not validated on the
device -- a check would need a synchronisation and would break CUDA-graph capture -- so they must be positive and
finite: a ``0`` yields a NaN row).
:func:`log_softmax_batch_invariant` is :func:`flashinfer.sampling.log_softmax`: the fp32 log-softmax row, or with
``out_dtype`` the input's 16-bit dtype the row converted once to it (the 16-bit output member, its own fingerprint).
:func:`log_probs_batch_invariant` is :func:`flashinfer.sampling.log_probs`: the natural log-probability of one or ``K``
token ids per row in fp32, and with ``return_lse`` the row's log-sum-exp (the same launch). Every unsupported input
raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.

:func:`lse_row` is the generic entry the tests drive (``mode=`` for every output of the class, ``config=`` to force a
member of the enumerated space, ``sm_count=`` to ask the dispatcher for another part's pick).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch

from ..support import check_device, check_no_overlap, out_buffer, unsupported
from .schedule import (
    VOCAB_COVERED,
    LSEConfig,
    default_config,
    device_sm_count,
    max_vec_elems,
)
from .spec import (
    LOG_SOFTMAX,
    LOG_SOFTMAX_CVT,
    LOGP,
    LOGP_LSE,
    SOFTMAX,
    SOFTMAX_F32,
    Mode,
    scale_value,
)

CLASS = "lse.row"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
LSE_ROW_ARCHS = ((10, 0), (10, 3), (10, 7))
MAX_VOCAB = 1 << 31
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_TEMPERATURE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_dummies: dict = {}


def _dummy(device, dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _dummies:
        _dummies[key] = torch.zeros(8, dtype=dtype, device=device)
    return _dummies[key]


_dummies2: Dict[tuple, torch.Tensor] = {}


def _dummy2(device, dtype) -> torch.Tensor:
    """The ``(1, 8)`` view of the dummy, cached (a view per call otherwise)."""
    key = (device, dtype)
    if key not in _dummies2:
        _dummies2[key] = _dummy(device, dtype).view(1, 8)
    return _dummies2[key]


def _rows(t: torch.Tensor, N: int, name: str) -> torch.Tensor:
    try:
        t2 = t.view(-1, N)
    except RuntimeError:
        raise ValueError(
            f"{name} of shape {tuple(t.shape)} cannot be viewed as (rows, {N}) without a copy"
        ) from None
    if t2.stride(-1) != 1:
        raise ValueError(f"{name} must have a unit stride along the vocabulary")
    return t2


def _temperature(
    temperature, M: int, lead: Tuple[int, ...], device
) -> Tuple[Optional[torch.Tensor], float]:
    """``(per-row f32 tensor | None, host scale c)``. A scalar is rounded to binary32 and ``c = div(LOG2E, T)`` is formed
    on the host with the device's ``div.rn``; a tensor of shape ``(M,)`` or ``lead`` (the logits' leading dimensions) is
    widened / rounded (RNE) to f32 and the kernel forms ``c`` per row. The tensor's values are not validated (a device
    check would need a synchronisation and break CUDA-graph capture): they must be positive and finite; a ``0`` gives
    ``c = inf`` and a NaN row, a negative or non-finite value an undefined row."""
    if temperature is None:
        return None, scale_value(None)
    if isinstance(temperature, torch.Tensor):
        if temperature.dtype not in _TEMPERATURE_DTYPES:
            raise unsupported(
                CLASS,
                f"a temperature tensor of dtype {temperature.dtype}",
                "f16 / bf16 / f32 / f64 temperatures",
            )
        if (
            temperature.dtype == torch.float32
            and temperature.device == device
            and temperature.dim() == 1
            and temperature.shape[0] == M
            and temperature.is_contiguous()
        ):
            return temperature, 0.0  # already the kernel's form: no copy, no view
        t = temperature.to(device=device, dtype=torch.float32)
        if t.dim() == 0 or t.numel() == 1:
            t = t.reshape(1).expand(M)
        elif tuple(t.shape) == tuple(lead):
            t = t.reshape(M)
        if t.shape != (M,):
            raise ValueError(
                f"temperature must have one value per row: shape ({M},) or {tuple(lead)}; got shape {tuple(temperature.shape)}"
            )
        return t.contiguous(), 0.0
    T = float(temperature)
    if not (T > 0.0) or not math.isfinite(T):
        raise ValueError(
            f"temperature must be a positive finite number; got {temperature!r}"
        )
    return None, scale_value(T)


def check_input(
    x: torch.Tensor,
    name: str,
    cls: str,
    archs,
    vocab_hint: str = (
        f"every vocabulary up to {VOCAB_COVERED} has a configuration; beyond it the row's alignment decides"
    ),
) -> int:
    """The device, dtype, rank and vocabulary checks shared by the mode's sampling entries (the row class, the draw,
    the support): ``BatchInvariantUnsupportedError`` naming ``cls`` on a miss; returns the vocabulary ``N``."""
    check_device(x.device, cls, archs)
    if x.dtype not in _DTYPE_STR:
        raise unsupported(cls, f"{name} dtype {x.dtype}", f"bf16, f16 or fp32 {name}")
    if x.dim() < 1:
        raise unsupported(
            cls, f"a {x.dim()}-D input", f"{name} of shape (..., vocabulary)"
        )
    N = int(x.shape[-1])
    if not (0 < N < MAX_VOCAB):
        raise unsupported(
            cls, f"a vocabulary of {N}", f"0 < vocabulary < 2^31 ({vocab_hint})"
        )
    return N


def _check_logits(logits: torch.Tensor) -> int:
    return check_input(logits, "logits", CLASS, LSE_ROW_ARCHS)


def _aligned_rows(x2: torch.Tensor, align: int) -> torch.Tensor:
    """The row view the kernel's vector loads accept: a base aligned to ``align`` bytes and a row stride that is a
    multiple of it (a view into a wider tensor or an under-aligned storage offset is copied: a placement, not numerics)."""
    M = x2.shape[0]
    if x2.data_ptr() % align or (M > 1 and (x2.stride(0) * x2.element_size()) % align):
        return x2.clone(memory_format=torch.contiguous_format)
    return x2


def lse_row(
    mode: Mode,
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    token_ids: Optional[torch.Tensor] = None,
    *,
    enable_pdl: bool = False,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    logp_out: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """One launch of the class for the outputs ``mode`` names: ``"probs"`` (the softmax row in the input dtype, or in
    fp32 for ``probs_f32``), ``"logprobs"`` (the log-softmax row, fp32, or in the input dtype for ``logprobs_cvt``),
    ``"lse"`` (fp32 per row) and ``"logp"`` (fp32:
    the log-probability of ``token_ids``, one id per row of shape ``logits.shape[:-1]`` or ``K`` ids per row of shape
    ``logits.shape[:-1] + (K,)``, NaN for an id outside ``[0, N)``). Bitwise identical for every ``config`` in
    :func:`schedule.all_valid_configs`, every batch composition and every ``sm_count``. ``out`` (the row output),
    ``lse_out`` and ``logp_out`` are caller-owned buffers written in place and returned as given (``support.out_buffer``:
    the output dtype and shape, contiguous, the row output 16-byte aligned); without them the outputs are allocated."""
    from .kernel import get_compiled_kernel, k_slots_for

    N = _check_logits(logits)
    lead = logits.shape[:-1]
    try:
        x2 = _rows(logits, N, "logits")
    except ValueError:
        # not viewable as rows with a unit stride along the vocabulary (a strided column view, a non-contiguous
        # leading layout): copied first, a placement
        x2 = logits.reshape(-1, N).contiguous()
    M = x2.shape[0]
    temp_t, c = _temperature(temperature, M, tuple(lead), logits.device)
    elem_bytes = logits.element_size()
    if config is None:
        try:
            config = default_config(
                N,
                M,
                elem_bytes,
                sm_count or device_sm_count(logits.device),
                row_output=mode.row_output,
            )
        except (
            ValueError
        ) as e:  # an empty configuration space: a long odd row beyond VOCAB_COVERED
            raise unsupported(
                CLASS, str(e), f"vocabularies up to {VOCAB_COVERED}"
            ) from None
    else:
        config.validate(N, elem_bytes)
    if config.vec_elems > max_vec_elems(N, elem_bytes):
        raise ValueError("config vector width exceeds the row alignment")
    x2 = _aligned_rows(x2, config.vec_elems * elem_bytes)
    dev = logits.device
    res: Dict[str, torch.Tensor] = {}
    yb = lb = pb = None  # the caller-owned buffers, when given
    y = _dummy2(dev, torch.float32)
    if mode.row_output:
        row_dtype = (
            logits.dtype
            if ((mode.probs and not mode.probs_f32) or mode.logprobs_cvt)
            else torch.float32
        )
        yb = out_buffer(out, "out", tuple(logits.shape), row_dtype, logits, 16)
        y = (
            yb.view(M, N)
            if yb is not None
            else torch.empty((M, N), dtype=row_dtype, device=dev)
        )
        res["probs" if mode.probs else "logprobs"] = (
            yb if yb is not None else y.view(*lead, N)
        )
    elif out is not None:
        raise ValueError(
            "out is the row output's buffer: this mode writes no row output"
        )
    lse_t = _dummy(dev, torch.float32)
    if mode.lse:
        lb = out_buffer(lse_out, "lse_out", tuple(lead), torch.float32, logits, 4)
        lse_t = (
            lb.view(-1)
            if lb is not None
            else torch.empty(M, dtype=torch.float32, device=dev)
        )
        res["lse"] = lb if lb is not None else lse_t.view(lead)
    elif lse_out is not None:
        raise ValueError("lse_out needs return_lse=True")
    logp_t = _dummy2(dev, torch.float32)
    tok = _dummy2(dev, torch.int64)
    K = 1
    if mode.logp:
        if token_ids is None:
            raise ValueError("token_ids are required for the log-probability output")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise unsupported(
                CLASS, f"token_ids of dtype {token_ids.dtype}", "int32 or int64 ids"
            )
        lead_t = tuple(lead)
        if tuple(token_ids.shape) == lead_t:
            multi = False
        elif (
            token_ids.dim() == len(lead_t) + 1
            and tuple(token_ids.shape[:-1]) == lead_t
            and token_ids.shape[-1] > 0
        ):
            multi, K = True, int(token_ids.shape[-1])
        else:
            raise ValueError(
                f"token_ids must have one id per row (shape {lead_t}) or K >= 1 ids per row (shape {lead_t + ('K',)}); "
                f"got {tuple(token_ids.shape)}"
            )
        tok = token_ids.to(device=dev, dtype=torch.int64).reshape(M, K).contiguous()
        pb = out_buffer(
            logp_out,
            "logp_out",
            tuple(lead) + ((K,) if multi else ()),
            torch.float32,
            logits,
            4,
        )
        logp_t = (
            pb.view(M, K)
            if pb is not None
            else torch.empty((M, K), dtype=torch.float32, device=dev)
        )
        res["logp"] = (
            pb
            if pb is not None
            else (logp_t.view(*lead, K) if multi else logp_t.view(lead))
        )
    elif logp_out is not None:
        raise ValueError(
            "logp_out is the token log-probabilities' buffer: this mode writes none"
        )
    # an output may not overlap another output or an input, except the row output aliasing the logits exactly (in
    # place: every thread writes the elements it read, after reading them). Not with a token gather in the same launch:
    # the token's logit may lie in a peer CTA's share, already overwritten by its row output (the public entries never
    # combine a row output with the gather; the generic entry refuses the alias then)
    check_no_overlap(
        [("out", yb), ("lse_out", lb), ("logp_out", pb)],
        [("logits", logits), ("temperature", temp_t), ("token_ids", token_ids)],
        in_place=() if mode.logp else (("out", "logits"),),
    )
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    if M == 0:  # no rows: nothing to launch; the outputs keep their documented shapes
        return res
    kernel = get_compiled_kernel(
        _DTYPE_STR[logits.dtype],
        N,
        config,
        mode,
        temp_t is not None,
        bool(enable_pdl),
        k_slots_for(K),
    )
    kernel(
        x2,
        y,
        lse_t,
        tok,
        logp_t,
        temp_arg,
        float(c),
        M,
        N,
        K,
        int(bool(enable_pdl)),
    )
    return res


def _pdl(enable_pdl: Optional[bool], device) -> bool:
    if enable_pdl is None:
        from ...utils import device_support_pdl

        return bool(device_support_pdl(device))
    return bool(enable_pdl)


def softmax_batch_invariant(
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.softmax(logits, temperature, enable_pdl, batch_invariant=True, out_dtype=, out=)``: the
    class ``lse.row``. ``out`` is a caller-owned buffer of the output dtype and ``logits.shape`` (contiguous, 16-byte
    aligned), written in place and returned.

    ``softmax(logits / temperature)`` along the last dimension, in the input dtype: ``p_j = cvt(ex2((x_j - m) c) / s)``
    with the canonical row trees for ``m`` and ``s``. ``out_dtype=torch.float32`` returns the probabilities before the
    output conversion (for a 16-bit input these are the fp32 input's class bit for bit, since the leaves are the exact
    widenings: ``softmax(x, out_dtype=torch.float32) == softmax(x.float())``). ``enable_pdl=None`` follows the device
    (PDL on SM >= 9.0). ``config`` / ``sm_count`` force a configuration / another part's dispatcher pick (members of the
    enumerated space: the bits never depend on them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    device, architecture); it never falls back."""
    _check_logits(logits)
    if out_dtype is None or out_dtype == logits.dtype:
        mode = SOFTMAX
    elif out_dtype == torch.float32:
        mode = SOFTMAX_F32
    else:
        raise unsupported(
            CLASS,
            f"out_dtype {out_dtype}",
            "the input dtype (out_dtype=None) or torch.float32",
        )
    return lse_row(
        mode,
        logits,
        temperature,
        enable_pdl=_pdl(enable_pdl, logits.device),
        config=config,
        sm_count=sm_count,
        out=out,
    )["probs"]


def log_softmax_batch_invariant(
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.log_softmax(logits, temperature, enable_pdl, out_dtype=, out=)``: the class ``lse.row``.
    ``out`` is a caller-owned buffer of the output dtype and ``logits.shape`` (contiguous, 16-byte aligned), written in
    place and returned.

    The natural log-softmax row of ``logits / temperature`` in fp32, shape ``logits.shape``: ``logp_j = ((x_j - m) c -
    lg2(s)) ln2`` from the same two trees as the softmax; :func:`log_probs_batch_invariant` is this row gathered at the
    token ids. ``out_dtype`` the input's 16-bit dtype returns the row converted once (``cvt.rn``) to it -- the 16-bit
    output member (``spec.LOG_SOFTMAX_CVT``, its own fingerprint: ``fingerprint("log_softmax", ..., out_dtype=)``), the
    trainers' ``aten::_log_softmax(bf16) -> bf16`` shape, bit for bit torch's RNE conversion of the fp32 row; ``None`` /
    ``torch.float32`` the fp32 row.

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    device, architecture, an ``out_dtype`` that is neither the input dtype nor fp32); it never falls back."""
    _check_logits(logits)
    if out_dtype is None or out_dtype == torch.float32:
        mode = LOG_SOFTMAX
    elif out_dtype == logits.dtype:
        mode = LOG_SOFTMAX_CVT
    else:
        raise unsupported(
            CLASS,
            f"out_dtype {out_dtype} for {logits.dtype} logits",
            "the input dtype (the 16-bit output member) or torch.float32 (out_dtype=None)",
        )
    return lse_row(
        mode,
        logits,
        temperature,
        enable_pdl=_pdl(enable_pdl, logits.device),
        config=config,
        sm_count=sm_count,
        out=out,
    )["logprobs"]


def log_probs_batch_invariant(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
    *,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
    top_k: Optional[Union[int, torch.Tensor]] = None,
    top_p: Optional[Union[float, torch.Tensor]] = None,
    min_p: Optional[Union[float, torch.Tensor]] = None,
    filter_apply_order: str = "top_k_first",
):
    """``flashinfer.sampling.log_probs(logits, token_ids, temperature, return_lse=..., out=, lse_out=)``: the class
    ``lse.row``. ``out`` (fp32, the ids' shape) and ``lse_out`` (fp32, ``logits.shape[:-1]``) are caller-owned buffers,
    written in place and returned.

    With ``top_k`` / ``top_p`` / ``min_p`` (and ``filter_apply_order``, as the truncated samplers take them) the value is
    the token's log-probability under the sampler's kept-set distribution instead: ``support.row.v1``'s kept set and mass at
    the row, then ``mul(add(z_t, neg(lg2(mass))), LN2)`` -- bit for bit the ``kept_logp`` the truncated samplers return for
    the token they drew (the engines' processed log-probability); ``-inf`` for a token outside the kept set, NaN for an id
    outside ``[0, N)`` or a row whose kept mass is not a positive finite number (``draw_support_api.kept_log_probs``). One
    id per row; ``return_lse`` is the full row's statistic and is refused with a filter.

    The natural log-probability of ``token_ids[r]`` under ``softmax(logits[r] / temperature)``, fp32 of shape
    ``logits.shape[:-1]`` for one id per row or ``logits.shape[:-1] + (K,)`` for ``K`` ids per row:
    ``logp = ((x_t - m) c - lg2(s)) ln2`` from the same two trees as the softmax (NaN for an id outside ``[0, N)``). With
    ``return_lse`` also the row's natural log-sum-exp ``lse = (m c + lg2(s)) ln2`` (one launch).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class; it
    never falls back."""
    if top_k is not None or top_p is not None or min_p is not None:
        if return_lse or lse_out is not None:
            raise ValueError(
                "return_lse is the full row's statistic: call log_probs without the filters for it"
            )
        from .draw_support_api import kept_log_probs

        return kept_log_probs(
            logits,
            token_ids,
            temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            order=filter_apply_order,
            enable_pdl=_pdl(enable_pdl, logits.device),
            config=config,
            sm_count=sm_count,
            out=out,
        )["kept_logp"]
    _check_logits(logits)
    res = lse_row(
        LOGP_LSE if return_lse else LOGP,
        logits,
        temperature,
        token_ids,
        enable_pdl=_pdl(enable_pdl, logits.device),
        config=config,
        sm_count=sm_count,
        logp_out=out,
        lse_out=lse_out,
    )
    return (res["logp"], res["lse"]) if return_lse else res["logp"]
