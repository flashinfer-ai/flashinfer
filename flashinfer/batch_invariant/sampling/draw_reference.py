# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU model of ``sample.icdf.v1`` (``draw_spec.py``): per row the weights, the canonical sum ``S``, the stream's
uniform and the threshold ``t = mul(u, S)``, the node table of the sum tree (torch f32 adds, one rounding each), the
inclusive prefixes in tree order and the first positive-weight crossing. Fully on the CPU for the probability input; the logits input
takes its weights from ``lse.row``'s chain (``reference.py``: the device ``ex2`` tables).

Outside every kernel cache closure (no kernel module imports it)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch

from ..fp_interface import exact as fp32
from . import stream
from .draw_spec import LANES, V, index_of_address
from .reference import _scales, ex2_device, row_max, row_sum
from .schedule import BLOCK, padded_blocks


def slot_folds(w: torch.Tensor, N: int):
    """``(fold_incl (Kp*256,), present (Kp*256,) bool, totals (32, Kp))`` of one row of f32 weights: the running slot
    fold through each element (absent elements carry the previous value), and the slot totals indexed ``[lane, block]``."""
    Kp = padded_blocks(N)
    L = Kp * BLOCK
    full = torch.zeros(L, dtype=torch.float32)
    full[:N] = w
    present = torch.zeros(L, dtype=torch.bool)
    present[:N] = True
    t = full.view(Kp, LANES, V)
    acc = torch.zeros(Kp, LANES, dtype=torch.float32)
    folds = torch.empty(Kp, LANES, V, dtype=torch.float32)
    for e in range(V):
        acc = torch.add(acc, t[:, :, e])
        folds[:, :, e] = acc
    return folds.reshape(L), present, acc.t().contiguous()  # totals [lane, block]


def node_tables(totals: torch.Tensor):
    """The sum tree's nodes: ``block_nodes[level] (32, Kp >> level)`` per lane, ``lane_nodes[level] (32 >> level,)``."""
    block_nodes = [totals]
    while block_nodes[-1].shape[1] > 1:
        cur = block_nodes[-1]
        block_nodes.append(torch.add(cur[:, 0::2], cur[:, 1::2]))
    lane_nodes = [block_nodes[-1][:, 0].contiguous()]
    while lane_nodes[-1].shape[0] > 1:
        cur = lane_nodes[-1]
        lane_nodes.append(torch.add(cur[0::2], cur[1::2]))
    return block_nodes, lane_nodes


def outer_prefix(block_nodes, lane_nodes, Kp: int) -> torch.Tensor:
    """``outer[lane, block]``: the add chain over the set bits of the slot's address, ascending, of the left-sibling
    nodes."""
    kb = Kp.bit_length() - 1
    lanes = torch.arange(LANES).view(LANES, 1)
    blocks = torch.arange(Kp).view(1, Kp)
    acc = torch.zeros(LANES, Kp, dtype=torch.float32)
    for b in range(kb):
        bit = ((blocks >> b) & 1).bool().expand(LANES, Kp)
        grp = (blocks >> b) - 1  # the group just below, valid where the bit is set
        node = block_nodes[b][
            lanes.expand(LANES, Kp), grp.clamp(min=0).expand(LANES, Kp)
        ]
        acc = torch.add(acc, torch.where(bit, node, torch.zeros_like(node)))
    for k in range(5):
        bit = ((lanes >> k) & 1).bool().expand(LANES, Kp)
        grp = (lanes >> k) - 1
        node = lane_nodes[k][grp.clamp(min=0)].expand(LANES, Kp)
        acc = torch.add(acc, torch.where(bit, node, torch.zeros_like(node)))
    return acc


def draw_row(w: torch.Tensor, N: int, u: float) -> Tuple[int, bool, float, float]:
    """One row: ``(token, valid, S, t)`` from f32 weights ``w`` (N,) and the uniform ``u``."""
    Kp = padded_blocks(N)
    S = row_sum(w, N)
    if not (S > 0.0) or math.isnan(S) or math.isinf(S):
        return -1, False, S, float("nan")
    t = fp32.mul(u, S)
    folds, present, totals = slot_folds(w, N)
    block_nodes, lane_nodes = node_tables(totals)
    outer = outer_prefix(block_nodes, lane_nodes, Kp)  # [lane, block]
    # the inclusive prefix of every element, laid out in tree order (lane, block, elem)
    f_tree = folds.view(Kp, LANES, V).permute(1, 0, 2).reshape(-1)
    p_tree = present.view(Kp, LANES, V).permute(1, 0, 2).reshape(-1)
    w_tree = torch.zeros(Kp * BLOCK, dtype=torch.float32)
    w_tree[:N] = w
    w_tree = w_tree.view(Kp, LANES, V).permute(1, 0, 2).reshape(-1)
    incl = torch.add(outer.reshape(-1, 1).expand(LANES * Kp, V).reshape(-1), f_tree)
    cross = (
        (incl >= t) & p_tree & (w_tree > 0)
    )  # the positivity is part of the rule (draw_spec.SELECT_RULE)
    idx = torch.nonzero(cross).reshape(-1)
    if idx.numel():
        a = int(idx[0])
    else:
        nz = torch.nonzero((w_tree > 0) & p_tree).reshape(-1)
        if not nz.numel():
            return -1, False, S, t
        a = int(nz[-1])
    return index_of_address(a, Kp), True, S, t


def draw_reference(
    x: torch.Tensor,
    seeds: Sequence[int],
    offsets: Sequence[int],
    from_logits: bool = False,
    temperature=None,
    draw: int = 0,
    rows: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """The exact model for ``rows`` of ``x`` (probabilities, or logits with ``from_logits``; ``(M, N)``): ``{"samples"
    int64, "valid" bool, "u" f32, "S" f32, "t" f32}`` (``-1`` / False for an invalid row). ``seeds`` / ``offsets`` are
    the rows' keys (Python ints, as signed or unsigned 64-bit values); ``rows`` the source row of each output (one outside
    ``[0, M)`` is an invalid output row: token ``-1``, as the kernel's gather reports it)."""
    x = x.detach().cpu()
    M, N = x.shape
    rows = list(range(M)) if rows is None else list(rows)
    xf = x.to(torch.float32)
    cs = (
        _scales(temperature, len(rows)) if from_logits else None
    )  # per output row, as the kernel reads it
    out: Dict[str, list] = {k: [] for k in ("samples", "valid", "u", "S", "t")}
    for i, r in enumerate(
        rows
    ):  # output i reads source row r with output i's key (and temperature)
        if not 0 <= r < M:
            for k, v in (
                ("samples", -1),
                ("valid", False),
                ("u", float("nan")),
                ("S", float("nan")),
                ("t", float("nan")),
            ):
                out[k].append(v)
            continue
        if from_logits:
            assert cs is not None
            m = row_max(xf[r])
            z = torch.mul(
                torch.add(xf[r], torch.tensor(fp32.neg(m), dtype=torch.float32)),
                torch.tensor(cs[i], dtype=torch.float32),
            )
            w = ex2_device(z)
        else:
            w = xf[r]
        u = stream.uniforms(int(seeds[i]), int(offsets[i]), draw)[0]
        tok, ok, S, t = draw_row(w, N, u)
        out["samples"].append(tok)
        out["valid"].append(ok)
        out["u"].append(u)
        out["S"].append(S)
        out["t"].append(t)
    return {
        "samples": torch.tensor(out["samples"], dtype=torch.int64),
        "valid": torch.tensor(out["valid"], dtype=torch.bool),
        "u": torch.tensor(out["u"], dtype=torch.float64).to(torch.float32),
        "S": torch.tensor(out["S"], dtype=torch.float64).to(torch.float32),
        "t": torch.tensor(out["t"], dtype=torch.float64).to(torch.float32),
    }
