# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the draw class ``sample.icdf.v1``.

:func:`sampling_from_probs_batch_invariant` is what :func:`flashinfer.sampling.sampling_from_probs` runs under
``batch_invariant=True``, :func:`sampling_from_logits_batch_invariant` what :func:`flashinfer.sampling.sampling_from_logits`
runs: one token per output row by the inverse CDF over the canonical sum tree of the row's weights (the probabilities as
given, or the ``lse.row`` weights ``e_j = ex2((x_j - m) c)`` of the logits), from the fp32 uniform of the request-keyed
stream (``stream.py``). The RNG is the class's own: a row's key is ``(seed, offset)`` -- the per-row tensors as given, one
key per output row; a scalar, a length-one tensor or the pair FlashInfer reserves from the ``generator`` exactly as the
operator's own path does serves a single row, and a call with more rows and such a shared key raises ``ValueError``
(every row would draw at the same uniform: identical rows identical tokens) -- and the draw index is the Philox counter;
the row's slot in the batch enters nothing, so the token is a function of (row, seed, offset) alone. The uniforms are not those of the operator's own samplers for any slot (those seed
``curand_init(seed, slot, offset)``).

:func:`draw` is the generic entry the tests drive (``config=`` to force a member of the enumerated space, ``sm_count=``
to ask the dispatcher for another part's pick, ``draw=`` the block index, ``return_logprob`` the drawn token's
log-probability from the same statistics). Every unsupported input raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from ..support import check_device, unsupported
from .api import (
    _DTYPE_STR,
    MAX_VOCAB,
    _aligned_rows,
    _dummy,
    _pdl,
    _rows,
    _temperature,
)
from .draw_schedule import SHARE_MAX, draw_config
from .schedule import (
    LOADS_PER_THREAD_MAX,
    VOCAB_COVERED,
    LSEConfig,
    device_sm_count,
    max_vec_elems,
)
from .stream import MASK64

CLASS = "sample.icdf.v1"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
DRAW_ARCHS = ((10, 0), (10, 3), (10, 7))
_KEY_DTYPES = (torch.int64, torch.uint64)


def _check_input(x: torch.Tensor, name: str) -> int:
    """The device, dtype, rank and vocabulary checks of an input; returns ``N``."""
    check_device(x.device, CLASS, DRAW_ARCHS)
    if x.dtype not in _DTYPE_STR:
        raise unsupported(CLASS, f"{name} dtype {x.dtype}", f"bf16, f16 or fp32 {name}")
    if x.dim() < 1:
        raise unsupported(
            CLASS, f"a {x.dim()}-D input", f"{name} of shape (..., vocabulary)"
        )
    N = int(x.shape[-1])
    if not (0 < N < MAX_VOCAB):
        raise unsupported(
            CLASS,
            f"a vocabulary of {N}",
            f"0 < vocabulary < 2^31 (every vocabulary up to {VOCAB_COVERED} has a configuration; beyond it only where the row's alignment admits a member within {SHARE_MAX} elements per thread)",
        )
    return N


def _shared_key_error(M: int) -> ValueError:
    """The refusal of a key shared by several output rows (the operator takes ``seed`` / ``offset`` both as tensors or both
    as scalars, so the per-row form is two tensors, at least one with a value per output row)."""
    return ValueError(
        f"a batch-invariant draw of {M} rows needs one key per output row: pass seed and offset both as int64 / uint64 "
        f"tensors, at least one of them with {M} values (a request's own (seed, offset) pair); a key shared by every row "
        "(scalars, length-one tensors or a bare generator) would give every row the same uniform, so identical rows would "
        "draw identical tokens"
    )


def _signed64(v: int) -> int:
    """A Python int reduced to its signed 64-bit value (uint64 keys by their bits)."""
    v = int(v) & MASK64
    return v - (1 << 64) if v >= 1 << 63 else v


def _row_keys(seed, offset, M: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The rows' ``(seed, offset)`` keys as two int64 ``(M,)`` tensors: a tensor of ``M`` values as given (int64, or
    uint64 by its bits); a scalar or a length-one tensor broadcasts to the rows -- for a single output row only: with
    ``M > 1`` a key shared by every row is refused (``ValueError``), since every row would then draw at the same uniform
    and identical rows would draw identical tokens (the class has no slot to separate them by)."""

    def one(v, name):
        if isinstance(v, torch.Tensor):
            if v.dtype not in _KEY_DTYPES:
                raise ValueError(f"{name} tensor must be int64/uint64")
            t = v.to(device=device)
            if t.dtype == torch.uint64:
                t = t.view(torch.int64)
            t = t.reshape(-1)
            shared = t.numel() == 1
            if shared and M != 1:
                t = t.expand(M)
            if t.shape != (M,):
                raise ValueError(
                    f"{name} tensor length must be 1 or {M} (one value per output row); got {tuple(v.shape)}"
                )
            return t.contiguous(), shared
        return torch.full((M,), _signed64(v), dtype=torch.int64, device=device), True

    s, s_shared = one(seed, "seed")
    o, o_shared = one(offset, "offset")
    if M > 1 and s_shared and o_shared:
        raise _shared_key_error(M)
    return s, o


def _resolve_keys(seed, offset, generator, increment: int, M: int, device):
    """FlashInfer's own key resolution: without both ``seed`` and ``offset`` the pair is reserved from the generator
    (``get_seed_and_offset``, the same increment as the operator's path), then both are validated as the operator
    validates them (both tensors or both scalars; int64 / uint64, 1-D, one value or one per output row)."""
    from ...sampling import _validate_and_convert_seed_offset, get_seed_and_offset

    if seed is None or offset is None:
        if (
            M > 1
        ):  # the generator's pair would key every row alike: refused before the generator's state is advanced
            raise _shared_key_error(M)
        seed, offset = get_seed_and_offset(increment, generator, device)
    _validate_and_convert_seed_offset(seed, offset, device, M)
    return _row_keys(seed, offset, M, device)


def draw(
    x: torch.Tensor,
    seed,
    offset,
    *,
    from_logits: bool = False,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    indices: Optional[torch.Tensor] = None,
    draw: int = 0,
    return_logprob: bool = False,
    enable_pdl: bool = False,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """One launch of the class: ``{"samples"`` (int32, ``-1`` for an invalid row), ``"valid"`` (bool), and with
    ``return_logprob`` (logits only) ``"logp"`` (fp32: the drawn token's natural log-probability from the same ``m`` and
    ``S``, NaN where invalid)``}``, each of shape ``indices.shape`` or ``x.shape[:-1]``. ``seed`` / ``offset`` are the
    rows' keys (int64 / uint64 tensors of one value per output row; a single value, as a Python int or a length-one
    tensor, for one row), ``draw`` the block of the stream the uniform is read from. An index outside the source rows
    makes its output row invalid (token ``-1``). Bitwise identical for every ``config`` in
    :func:`schedule.all_valid_configs` (at one row per CTA with ``indices``), every batch composition and every
    ``sm_count``."""
    from .draw_kernel import get_compiled_kernel

    name = "logits" if from_logits else "probs"
    if return_logprob and not from_logits:
        raise ValueError("the drawn token's log-probability is defined for logits")
    N = _check_input(x, name)
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
    temp_t, c = _temperature(temperature, M, lead, dev) if from_logits else (None, 0.0)
    elem_bytes = x.element_size()
    if config is None:
        try:
            config = draw_config(N, M, elem_bytes, sm_count or device_sm_count(dev))
        except (
            ValueError
        ) as e:  # no member within the share bound: a row beyond VOCAB_COVERED
            raise unsupported(
                CLASS,
                str(e),
                f"vocabularies up to {VOCAB_COVERED} (beyond it only where a member keeps the share within {SHARE_MAX} elements per thread)",
            ) from None
    else:
        config.validate(N, elem_bytes)
        loads = config.elems_per_thread(N) // config.vec_elems
        if config.stage != "regs" and loads > LOADS_PER_THREAD_MAX:
            raise ValueError(
                f"config {config.describe()} issues {loads} loads per thread per pass; the enumerated space bounds a "
                f"re-read or staged member at {LOADS_PER_THREAD_MAX} (such a kernel compiles for minutes)"
            )
        loads = config.elems_per_thread(N) // config.vec_elems
        if config.stage != "regs" and loads > LOADS_PER_THREAD_MAX:
            raise ValueError(
                f"config {config.describe()} issues {loads} loads per thread per pass; the enumerated space bounds a "
                f"re-read or staged member at {LOADS_PER_THREAD_MAX} (such a kernel compiles for minutes)"
            )
    if indices is not None and config.rows_per_cta != 1:
        raise ValueError("an index gather needs a config with one row per CTA")
    if config.vec_elems > max_vec_elems(N, elem_bytes):
        raise ValueError("config vector width exceeds the row alignment")
    x2 = _aligned_rows(x2, config.vec_elems * elem_bytes)
    tok = torch.empty(M, dtype=torch.int32, device=dev)
    valid = torch.empty(M, dtype=torch.int32, device=dev)
    lp = (
        torch.empty(M, dtype=torch.float32, device=dev)
        if return_logprob
        else _dummy(dev, torch.float32)
    )
    temp_arg = temp_t if temp_t is not None else _dummy(dev, torch.float32)
    if M > 0 and (indices is None or Msrc > 0):
        kernel = get_compiled_kernel(
            _DTYPE_STR[x.dtype],
            N,
            config,
            from_logits,
            temp_t is not None,
            indices is not None,
            return_logprob,
            bool(enable_pdl),
        )
        kernel(
            x2,
            seeds,
            offs,
            idx,
            tok,
            valid,
            lp,
            temp_arg,
            float(c),
            int(draw),
            M,
            Msrc,
            N,
        )
    elif (
        M > 0
    ):  # a gather from an empty source: every index lies outside the source rows, nothing is read
        tok.fill_(-1)
        valid.zero_()
        if return_logprob:
            lp.fill_(float("nan"))
    out = {"samples": tok.view(lead), "valid": valid.to(torch.bool).view(lead)}
    if return_logprob:
        out["logp"] = lp.view(lead)
    return out


def _entry(
    x: torch.Tensor,
    from_logits: bool,
    indices,
    generator,
    seed,
    offset,
    return_valid: bool,
    enable_pdl,
    config,
    sm_count,
):
    """The public entries' common body: the operator's key resolution and validation, the draw, the output
    dtype rule (the indices' dtype with ``indices``)."""
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
    increment = M * N if from_logits else M  # the operator's own reservation per entry
    seeds, offs = _resolve_keys(seed, offset, generator, increment, M, x.device)
    res = draw(
        x,
        seeds,
        offs,
        from_logits=from_logits,
        indices=indices,
        enable_pdl=_pdl(enable_pdl, x.device),
        config=config,
        sm_count=sm_count,
    )
    samples = res["samples"]
    if indices is not None and indices.dtype != samples.dtype:
        samples = samples.to(
            indices.dtype
        )  # the operator's contract: the output takes the indices' dtype
    return (samples, res["valid"]) if return_valid else samples


def sampling_from_probs_batch_invariant(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
    *,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.sampling.sampling_from_probs(probs, indices, generator=, seed=, offset=, return_valid=,
    batch_invariant=True)``: the class ``sample.icdf.v1`` over the given weights.

    One token per output row of ``probs`` (unnormalised weights ``>= 0``; bf16 / f16 / fp32, any vocabulary) by the
    inverse CDF over the canonical sum tree of the row, with the fp32 uniform of the request stream keyed by the row's
    ``(seed, offset)`` at block 0 -- one key per output row (a scalar, a length-one tensor or the pair reserved from the
    ``generator`` serves a single row; with more rows such a shared key raises ``ValueError``). ``indices[i]`` selects the
    source row of output ``i``. Returns int32 tokens (the indices' dtype with ``indices``; ``-1`` for a row whose weights
    do not sum to a positive finite number, and for an output whose index lies outside the source rows), and with
    ``return_valid`` the bool mask of the valid rows. ``enable_pdl=None`` follows the device. ``config`` / ``sm_count``
    force a configuration / another part's dispatcher pick (members of the enumerated space: the bits never depend on
    them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    device, architecture); it never falls back."""
    return _entry(
        probs,
        False,
        indices,
        generator,
        seed,
        offset,
        return_valid,
        enable_pdl,
        config,
        sm_count,
    )


def sampling_from_logits_batch_invariant(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    *,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """``flashinfer.sampling.sampling_from_logits(logits, indices, generator=, seed=, offset=, batch_invariant=True)``:
    the class ``sample.icdf.v1`` over the ``lse.row`` weights ``e_j = ex2.approx.ftz((x_j - m) log2(e))`` of the row (no
    temperature: the entry has none), without materialising the probabilities. The same draw as
    :func:`sampling_from_probs_batch_invariant`; a row of ``-inf`` logits (or with a NaN) is invalid and draws ``-1``.

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class; it never
    falls back."""
    return _entry(
        logits,
        True,
        indices,
        generator,
        seed,
        offset,
        False,
        enable_pdl,
        config,
        sm_count,
    )
