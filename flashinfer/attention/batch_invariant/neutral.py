# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The wrapper arguments the batch-invariant attention class does not consume, and the values of each that ask for nothing.

Serving engines pass a fixed argument list on every ``plan()`` / ``run()`` call, with neutral values where a feature is off
(unit scales for an unquantized model, the KV lengths next to the page table that already states them, the split knobs of the
regular kernels). The class takes none of these as an operand, so each has one rule here: a value that cannot change what the
caller gets is accepted and has no effect on the output bits; any other value raises the class's typed error, as before. The
checks are host-side and device-free unless a caller hands in a device tensor.

==========================================  =====================================================================================
argument                                    accepted (no effect on the bits); everything else raises
==========================================  =====================================================================================
``q_scale`` / ``k_scale`` / ``v_scale`` /   ``None`` or exactly ``1.0``, as a host number or a one-element CPU tensor (a device
``o_scale``                                 tensor would cost a readback per ``run()``: refused)
``fixed_split_size``                        ``None`` / ``-1`` (no request) and any positive size: the request is a KV split that
                                            does not follow the batch, which the class's 512-key chunks on the absolute KV axis
                                            already are; the size itself is not used
``disable_split_kv``                        ``False``, and ``True``: every unit keeps its chunks in one CTA (the fused placement
                                            of the same class, so the same bits) and the partials workspace is not used
``seq_lens``                                the KV lengths the page table (or ``kv_indptr`` of the ragged wrapper) states
``seq_lens_q``                              the query lengths ``qo_indptr`` states
``block_tables``                            a dense ``[batch, max_pages]`` table whose rows list the pages of the CSR triple
==========================================  =====================================================================================
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ...batch_invariant.support import unsupported

SCALE_NAMES = ("q_scale", "k_scale", "v_scale", "o_scale")
_UNIT_SCALES = "unscaled bf16 operands (q_scale / k_scale / v_scale / o_scale of None or exactly 1.0, as host numbers)"


def check_unit_scales(
    class_name: str, scales: Sequence, free: Sequence[str] = ()
) -> None:
    """``scales`` in the order of :data:`SCALE_NAMES`: each must be ``None`` or exactly one (compared with ``1.0`` exactly,
    no tolerance: engines pass the exact value). A unit scale multiplies nothing, so it is the unscaled class; any other
    value is a term the class does not have. ``free`` names the scales a class takes as operands (the per-tensor
    ``k_scale`` / ``v_scale`` of a low-bit KV class): those are not checked here. The query and the output scale are
    unit-only for every class."""
    for name, value in zip(SCALE_NAMES, scales, strict=False):
        if value is None or name in free:
            continue
        shown = value
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise unsupported(
                    class_name,
                    f"{name} as a tensor on {value.device} (reading it would stall every run())",
                    _UNIT_SCALES,
                )
            if value.numel() != 1:
                raise unsupported(
                    class_name,
                    f"{name} as a tensor of {value.numel()} elements",
                    _UNIT_SCALES,
                )
            shown = value.item()
        try:
            unit = float(shown) == 1.0
        except (TypeError, ValueError):
            unit = False
        if not unit:
            raise unsupported(class_name, f"{name}={shown!r}", _UNIT_SCALES)


def check_split_request(class_name: str, fixed_split_size, disable_split_kv) -> bool:
    """The split knobs of the regular kernels. Returns whether the plan keeps every unit fused (``disable_split_kv``).

    The rule for a determinism knob of the regular kernels (these two, and any later one of the kind, such as a fixed query
    tile): it asks for a placement that does not follow the batch. Every placement of the class already has that property,
    so a value that asks for it is accepted, and the value itself selects nothing unless the class has a placement that
    honours it literally (the fused units below).

    ``fixed_split_size`` asks the regular kernels for a KV split that does not depend on the batch. The class has that property
    by construction (its chunk grid is a class constant on the absolute KV axis), so ``None`` / ``-1`` and every positive size
    are satisfied and the size is not used. ``disable_split_kv=True`` is honoured with the fused placement, which is the
    same class: the split kind buys parallelism, never bits."""
    if fixed_split_size is not None:
        ok = isinstance(fixed_split_size, int) and not isinstance(
            fixed_split_size, bool
        )
        if not ok or (fixed_split_size != -1 and fixed_split_size <= 0):
            raise unsupported(
                class_name,
                f"fixed_split_size={fixed_split_size!r}",
                "None or -1 (no request) or a positive size (accepted and not used: the class's chunk length is a class "
                "constant)",
            )
    if not isinstance(disable_split_kv, bool):
        raise unsupported(
            class_name, f"disable_split_kv={disable_split_kv!r}", "True or False"
        )
    return disable_split_kv


def check_lengths(
    class_name: str,
    name: str,
    given,
    planned: Sequence[int],
    source: str,
    alternative: Optional[Sequence[int]] = None,
) -> None:
    """``given`` (``seq_lens`` / ``seq_lens_q``) restates lengths the plan already has from ``source``: it must equal
    ``planned`` entry for entry (``alternative[i]`` is also taken where a second convention exists for request ``i``). A host
    tensor is compared on the host; a device tensor is read back once, like the device CSR arrays of the same plan."""
    limit = f"{name} equal to the lengths of {source} (the class plans from those), or no {name}"
    if not isinstance(given, torch.Tensor):
        given = torch.as_tensor(given)
    if given.is_floating_point() or given.is_complex() or given.dtype == torch.bool:
        raise unsupported(class_name, f"{name} of dtype {given.dtype}", limit)
    host = given.detach().flatten().to("cpu", non_blocking=False).to(torch.int64)
    if host.numel() != len(planned):
        raise unsupported(
            class_name,
            f"{name} of {host.numel()} entries for {len(planned)} requests",
            limit,
        )
    want = torch.tensor(list(planned), dtype=torch.int64)
    bad = host != want
    if alternative is not None:
        bad &= host != torch.tensor(list(alternative), dtype=torch.int64)
    if bool(bad.any()):
        i = int(torch.nonzero(bad)[0])
        raise unsupported(
            class_name,
            f"{name}[{i}] = {int(host[i])} where {source} gives {int(want[i])}",
            limit,
        )


_BLOCK_TABLE = (
    "block_tables whose row i lists the pages kv_indices[kv_indptr[i] : kv_indptr[i + 1]] of request i (the class reads the "
    "CSR triple), or no block_tables"
)


def check_block_table_shape(
    class_name: str, block_tables, kv_spans: torch.Tensor
) -> None:
    """The device-free part of the ``block_tables`` rule: an integer ``[requests, max_pages]`` tensor wide enough for the
    longest request of the CSR triple (``kv_spans``: its pages per request, on the host)."""
    if not isinstance(block_tables, torch.Tensor):
        raise unsupported(
            class_name,
            f"block_tables of type {type(block_tables).__name__}",
            _BLOCK_TABLE,
        )
    if (
        block_tables.is_floating_point()
        or block_tables.is_complex()
        or block_tables.dtype == torch.bool
    ):
        raise unsupported(
            class_name, f"block_tables of dtype {block_tables.dtype}", _BLOCK_TABLE
        )
    batch = int(kv_spans.numel())
    widest = int(kv_spans.max()) if batch else 0
    if (
        block_tables.dim() != 2
        or block_tables.shape[0] != batch
        or block_tables.shape[1] < widest
    ):
        raise unsupported(
            class_name,
            f"block_tables of shape {tuple(block_tables.shape)} for {batch} requests of up to {widest} pages",
            _BLOCK_TABLE,
        )


def block_table_mismatches(
    block_tables: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_spans: torch.Tensor,
) -> torch.Tensor:
    """The number of page entries of the CSR triple that ``block_tables`` states differently, as a 0-dim int64 tensor on the
    device the comparison ran on (a CUDA device when either input is there, so the caller folds it into a readback it already
    makes). The shape was checked by :func:`check_block_table_shape`; the offsets are non-decreasing."""
    start = int(kv_indptr_host[0])
    stop = int(kv_indptr_host[-1])
    rows = torch.repeat_interleave(torch.arange(kv_spans.numel()), kv_spans)
    cols = torch.arange(start, stop) - kv_indptr_host[:-1][rows]
    where = block_tables.device
    got = block_tables[rows.to(where), cols.to(where)].to(torch.int64)
    want = kv_indices[start:stop].to(torch.int64)
    target = want.device if want.is_cuda else got.device
    return (got.to(target) != want.to(target)).sum()


def block_table_mismatch_error(class_name: str, count: int):
    return unsupported(
        class_name,
        f"block_tables that differ from kv_indices at {count} page entries",
        _BLOCK_TABLE,
    )
