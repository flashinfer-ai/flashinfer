# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the ``lse.row`` class.

:func:`softmax_batch_invariant` is what :func:`flashinfer.sampling.softmax` runs under ``batch_invariant=True``:
``softmax(logits / temperature)`` along the last dimension in the input dtype (or in fp32 with ``out_dtype``), with
FlashInfer's argument forms (temperature ``None``, a scalar or a per-row tensor of f16 / bf16 / f32 / f64 of shape
``logits.shape[:-1]`` or ``(rows,)``; a scalar is validated on the host, a tensor's values are not validated on the
device -- a check would need a synchronisation and would break CUDA-graph capture -- so they must be positive and
finite: a ``0`` yields a NaN row).
:func:`log_softmax_batch_invariant` is :func:`flashinfer.sampling.log_softmax`: the fp32 log-softmax row.
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

from ..support import check_device, unsupported
from .schedule import (
    VOCAB_COVERED,
    LSEConfig,
    default_config,
    device_sm_count,
    max_vec_elems,
)
from .spec import LOG_SOFTMAX, LOGP, LOGP_LSE, SOFTMAX, SOFTMAX_F32, Mode, scale_value

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


def _check_logits(logits: torch.Tensor) -> int:
    check_device(logits.device, CLASS, LSE_ROW_ARCHS)
    if logits.dtype not in _DTYPE_STR:
        raise unsupported(
            CLASS, f"logits dtype {logits.dtype}", "bf16, f16 or fp32 logits"
        )
    if logits.dim() < 1:
        raise unsupported(
            CLASS, f"a {logits.dim()}-D input", "logits of shape (..., vocabulary)"
        )
    N = int(logits.shape[-1])
    if not (0 < N < MAX_VOCAB):
        raise unsupported(
            CLASS,
            f"a vocabulary of {N}",
            f"0 < vocabulary < 2^31 (every vocabulary up to {VOCAB_COVERED} has a configuration; beyond it the row's alignment decides)",
        )
    return N


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
) -> Dict[str, torch.Tensor]:
    """One launch of the class for the outputs ``mode`` names: ``"probs"`` (the softmax row in the input dtype, or in
    fp32 for ``probs_f32``), ``"logprobs"`` (the log-softmax row, fp32), ``"lse"`` (fp32 per row) and ``"logp"`` (fp32:
    the log-probability of ``token_ids``, one id per row of shape ``logits.shape[:-1]`` or ``K`` ids per row of shape
    ``logits.shape[:-1] + (K,)``, NaN for an id outside ``[0, N)``). Bitwise identical for every ``config`` in
    :func:`schedule.all_valid_configs`, every batch composition and every ``sm_count``."""
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
    out: Dict[str, torch.Tensor] = {}
    y = _dummy(dev, torch.float32).view(1, 8)
    if mode.row_output:
        y = torch.empty(
            (M, N),
            dtype=logits.dtype
            if (mode.probs and not mode.probs_f32)
            else torch.float32,
            device=dev,
        )
        out["probs" if mode.probs else "logprobs"] = y.view(*lead, N)
    lse_t = _dummy(dev, torch.float32)
    if mode.lse:
        lse_t = torch.empty(M, dtype=torch.float32, device=dev)
        out["lse"] = lse_t.view(lead)
    logp_t = _dummy(dev, torch.float32).view(1, 8)
    tok = _dummy(dev, torch.int64).view(1, 8)
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
        logp_t = torch.empty((M, K), dtype=torch.float32, device=dev)
        out["logp"] = logp_t.view(*lead, K) if multi else logp_t.view(lead)
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    if M == 0:  # no rows: nothing to launch; the outputs keep their documented shapes
        return out
    kernel = get_compiled_kernel(
        _DTYPE_STR[logits.dtype],
        N,
        config,
        mode,
        temp_t is not None,
        bool(enable_pdl),
        k_slots_for(K),
    )
    kernel(x2, y, lse_t, tok, logp_t, temp_arg, float(c), M, N, K)
    return out


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
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.softmax(logits, temperature, enable_pdl, batch_invariant=True, out_dtype=)``: the class
    ``lse.row``.

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
    )["probs"]


def log_softmax_batch_invariant(
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
    *,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.log_softmax(logits, temperature, enable_pdl)``: the class ``lse.row``.

    The natural log-softmax row of ``logits / temperature`` in fp32, shape ``logits.shape``: ``logp_j = ((x_j - m) c -
    lg2(s)) ln2`` from the same two trees as the softmax; :func:`log_probs_batch_invariant` is this row gathered at the
    token ids.

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class; it
    never falls back."""
    _check_logits(logits)
    return lse_row(
        LOG_SOFTMAX,
        logits,
        temperature,
        enable_pdl=_pdl(enable_pdl, logits.device),
        config=config,
        sm_count=sm_count,
    )["logprobs"]


def log_probs_batch_invariant(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
    *,
    return_lse: bool = False,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.log_probs(logits, token_ids, temperature, return_lse=...)``: the class ``lse.row``.

    The natural log-probability of ``token_ids[r]`` under ``softmax(logits[r] / temperature)``, fp32 of shape
    ``logits.shape[:-1]`` for one id per row or ``logits.shape[:-1] + (K,)`` for ``K`` ids per row:
    ``logp = ((x_t - m) c - lg2(s)) ln2`` from the same two trees as the softmax (NaN for an id outside ``[0, N)``). With
    ``return_lse`` also the row's natural log-sum-exp ``lse = (m c + lg2(s)) ln2`` (one launch).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class; it
    never falls back."""
    _check_logits(logits)
    res = lse_row(
        LOGP_LSE if return_lse else LOGP,
        logits,
        temperature,
        token_ids,
        enable_pdl=_pdl(enable_pdl, logits.device),
        config=config,
        sm_count=sm_count,
    )
    return (res["logp"], res["lse"]) if return_lse else res["logp"]
