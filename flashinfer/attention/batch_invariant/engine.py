# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-shaped entry points for the RL-ready paged attention (classes ``attn.bf16.d64`` / ``d128`` / ``d256``:
one class per head dim, ``spec.HEAD_DIMS``).

* ``attention_paged(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, ...)``: the
  single-call form (prefill or decode: a request with ``q_len == 1`` is a decode step; positions are
  bottom-right causal, ``window_left`` as FlashInfer's);
* ``BatchPrefillWithPagedKVCacheWrapper`` / ``BatchDecodeWithPagedKVCacheWrapper``: FlashInfer's
  plan / run call shapes on top of it.

Q is ``(total_q, Hq, D)`` bf16, K/V caches ``(num_pages, page, Hkv, D)`` (NHD) or ``(num_pages, Hkv, page, D)``
(HND) bf16, metadata int32 on the device; the output is ``(total_q, Hq, D)`` bf16. The planner runs on
host copies of the indptr arrays (as FlashInfer's ``plan`` does); ``run`` launches the kernel and, for
split units, the fixup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .plan import DESC_FIELDS, GROUP_FIELDS, build_plan, ws_row
from .spec import HEAD_DIMS, scale_log2


PAGE_SIZES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
)  # every power of two up to 128 (page size is a placement / layout input, never a class parameter)
MAX_GROUP = 128  # GQA group g = Hq / Hkv: 1 .. 128 (a Q tile holds (128 // g) g rows)


def check_supported(D: int, Hq: int, Hkv: int, page: int) -> None:
    """The (D, g, page) combinations the FMHA body implements; anything else raises here (never a silent fallback that could
    change bits: the class term is the same for every supported page size and group, both are placement inputs)."""
    if D not in HEAD_DIMS:
        raise ValueError(f"head_dim {D} is not a class (supported: {HEAD_DIMS})")
    if Hkv <= 0 or Hq % Hkv != 0 or not 1 <= Hq // Hkv <= MAX_GROUP:
        raise ValueError(
            f"num_qo_heads / num_kv_heads = {Hq} / {Hkv}: the GQA group must be an integer in 1..{MAX_GROUP}"
        )
    if page not in PAGE_SIZES:
        raise ValueError(f"page_size {page}: supported page sizes are {PAGE_SIZES}")


def check_sm_scale(sm_scale: float) -> float:
    """``scale_log2(sm_scale)`` for a positive, finite ``sm_scale`` whose log2 scale is a binary32 number; ``ValueError`` otherwise
    (the kernel forms every exp2 argument with this constant)."""
    s = float(sm_scale)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(
            f"sm_scale {sm_scale}: the class takes a positive finite scale"
        )
    try:
        c = scale_log2(s)
    except OverflowError:
        raise ValueError(
            f"sm_scale {sm_scale}: sm_scale * log2(e) is outside the binary32 range"
        ) from None
    if c == 0.0:
        raise ValueError(
            f"sm_scale {sm_scale}: sm_scale * log2(e) rounds to zero in binary32"
        )
    return c


def valid_kv_slots(D: int = 128) -> tuple:
    """The ring depths of the FMHA body at head dim ``D``: ``2 .. 512 / D`` (``blocks.operands.max_kv_slots``: 128 KB of K/V in
    flight at every D), i.e. 2..8 at D = 64, 2..4 at D = 128, {2} at D = 256."""
    from .blocks.operands import max_kv_slots

    return tuple(range(2, max_kv_slots(D) + 1))


@dataclass(frozen=True)
class AttnConfig:
    """A configuration of the kernel: every member of the space computes the class (R3)."""

    kv_slots: int = 4  # smem ring depth (K or V tiles): 2..4
    split_policy: str = "auto"  # "auto" | "split" | "fused": the planner's unit kind
    grid: int = 0  # persistent CTAs (0 = min(units, SMs)); a launch parameter (I10)
    softmax_wgs: int = (
        1  # softmax warpgroups (1 or 2): all work on every tile, splitting its keys
    )
    merge: str = "auto"  # where a split group's chunk fold runs: "auto" (in-kernel for groups of <= 32 rows, the fixup kernel
    # otherwise), "fixup" (always the kernel), "inkernel" (always the last arriver): plan.py, same bits
    small: str = "swap"  # the placement of a batch whose units all have <= SWAPS_MAX_ROWS (8) live rows (decode at g <= 8): "swap" = the
    # Swaps-AB kernel (``swaps``: keys on the MMA M dimension, the rows on N = 8 / 16 / 32, two
    # softmax warpgroups on alternate tiles; the default: measured faster than "rep" at every D), "rep" = the row body's
    # replicated-row placement;
    # mixed batches take the row body either way (same bits)
    q2: bool = False  # with softmax_wgs = 2: the paired placement -- two Q tiles per unit, warpgroup k owns Q tile k (whole rows), one
    # K / V stream for both; D = 64 (blocks/operands.py: S_A | S_B | O_c,A | O_m,A | O_c,B | O_m,B in TMEM)

    def describe(self) -> str:
        return (
            f"s{self.kv_slots}_w{self.softmax_wgs}_{self.split_policy}"
            + (f"_g{self.grid}" if self.grid else "")
            + ({"auto": "", "fixup": "_fx", "inkernel": "_ik"}[self.merge])
            + ("_sw" if self.small == "swap" else "")
            + ("_q2" if self.q2 else "")
        )

    def well_formed(self, D: int = 128) -> bool:
        """The body's static knob constraints at head dim ``D`` (mirror of ``BIK.Attn.ConfigOKD``: ``2 <= kv_slots``,
        ``kv_slots * D <= 512``, ``small <= 1``)."""
        return (
            self.kv_slots in valid_kv_slots(D)
            and self.split_policy in ("auto", "split", "fused")
            and self.softmax_wgs in (1, 2)
            and self.grid >= 0
            and self.merge in ("auto", "fixup", "inkernel")
            and self.small in ("rep", "swap")
            and (not self.q2 or (self.softmax_wgs == 2 and D == 64))
        )

    def lean(self) -> str:
        """The record ``BIK.Attn.Config`` (kvSlots, policy, softmaxWgs, merge, small) of ``lean/BIKernels/Attention.lean``."""
        return (
            f"⟨{self.kv_slots}, {dict(auto=0, split=1, fused=2)[self.split_policy]}, {self.softmax_wgs}, "
            f"{dict(auto=0, fixup=1, inkernel=2)[self.merge]}, {dict(rep=0, swap=1)[self.small]}⟩"
        )


def all_valid_configs(D: int = 128) -> list:
    """The enumerated space at head dim ``D`` (R3 is claimed over it): ring ``valid_kv_slots(D)`` x unit kind x softmax warpgroups x
    merge placement x small-unit placement = 252 / 108 / 36 points at D = 64 / 128 / 256."""
    assert D in HEAD_DIMS, D
    return [
        AttnConfig(s, p, 0, w, merge=m, small=sm)
        for sm in ("rep", "swap")
        for m in ("auto", "fixup", "inkernel")
        for w in (1, 2)
        for s in valid_kv_slots(D)
        for p in ("auto", "split", "fused")
    ]


SWAPS_MAX_ROWS = 8  # units of at most this many live rows take the Swaps-AB kernel under small = "swap": the 8-row
# (n_pad = 8) kernel beats the row body; the 16-row kernel (24/2 heads, g = 12) measured slower than
# the row body at every D (its per-row chains and register spills), so 9-32-row units keep the row
# body's replicated placement (the 16 / 32-row kernels stay built and IR-pinned as retired placements)
SWAPS_DIMS = (64, 128, 256)  # head dims the Swaps-AB kernel implements


Q2_MIN_UNITS_PER_SM = 4  # the paired placement halves the unit count: below this many units per SM the lost parallelism
# outweighs the overlap of the two softmax warpgroups, so smaller batches take the one-warpgroup body (a placement rule: same bits)


def pair_units(desc: list) -> list:
    """The paired placement's unit table: rows ``2 u`` (Q tile A) and ``2 u + 1`` (Q tile B) of pair ``u``. Two units pair when they share the
    request, the KV head, the chunk range and the kind (``F_MODE``): consecutive Q tiles of one (request, head, chunk range) in token order; an
    odd one out pairs with an empty B (no rows, empty tile range, no group). Pairs are ordered longest first (LPT, as ``build_plan``). A
    placement: every unit's descriptor is unchanged, only its CTA and warpgroup move."""
    from .plan import (
        F_CHUNK_HI,
        F_CHUNK_LO,
        F_GROUP,
        F_KVHEAD,
        F_MODE,
        F_QTOK0,
        F_REQ,
        F_ROWS,
        F_TILE_HI,
        F_TILE_LO,
    )

    buckets: Dict[tuple, List[list]] = {}
    for d in desc:
        buckets.setdefault(
            (d[F_REQ], d[F_KVHEAD], d[F_CHUNK_LO], d[F_CHUNK_HI], d[F_MODE]), []
        ).append(d)
    pairs = []
    for key in buckets:
        units = sorted(buckets[key], key=lambda d: d[F_QTOK0])
        for i in range(0, len(units), 2):
            a = units[i]
            if i + 1 < len(units):
                b = units[i + 1]
            else:
                b = list(a)
                b[F_ROWS] = 0
                b[F_TILE_LO] = b[F_TILE_HI] = a[F_TILE_LO]
                b[F_GROUP] = -1
            pairs.append((a, b))
    pairs.sort(
        key=lambda ab: (
            -max(
                ab[0][F_TILE_HI] - ab[0][F_TILE_LO], ab[1][F_TILE_HI] - ab[1][F_TILE_LO]
            )
        )
    )
    return [row for ab in pairs for row in ab]


def swaps_n_pad(plan) -> int:
    """The Swaps-AB kernel's B-operand width for a plan whose units all have <= SWAPS_MAX_ROWS live rows (0 when some unit is wider: the row body)."""
    from .swaps.operands import n_pad_for

    rows = max((d[2] for d in plan.desc), default=0)
    return n_pad_for(rows) if 1 <= rows <= SWAPS_MAX_ROWS else 0


DEFAULT_RING = {
    64: 4,
    128: 4,
    256: 2,
}  # ring depth per head dim: D = 64 and 128 keep 4 (the D = 64 ring-8 candidate of the
# page-16 search was a noise-selected maximum: a same-GPU interleaved A/B found no difference beyond
# the noise), D = 256 has only 2


def default_config(D: int = 128):
    """The default placement of the class at head dim ``D``: the ring depth of ``DEFAULT_RING`` (4 / 4 / 2 at D = 64 / 128 / 256), one
    softmax warpgroup, the planner's unit kind and fold placement (``plan.build_plan``: per head dim).."""
    return AttnConfig(DEFAULT_RING[D], "auto", 0, 1)


def _kv_view(
    cache: torch.Tensor, kv_layout: str, Hkv: int, page: int, D: int
) -> torch.Tensor:
    """(D, page, Hkv, num_pages) view of an NHD / HND pool (runtime strides; no copy)."""
    if kv_layout == "NHD":
        return cache.permute(3, 1, 2, 0)
    return cache.permute(3, 2, 1, 0)


def _kv_view_ragged(kv: torch.Tensor, kv_layout: str, Hkv: int, D: int) -> torch.Tensor:
    """The packed K / V of every request -- ``(total_kv, Hkv, D)`` (NHD) or ``(Hkv, total_kv, D)`` (HND) -- as the kernel's
    ``(D, page, Hkv, num_pages)`` pool with ONE page of ``total_kv`` tokens (no copy; the size-1 page dimension gets a 16-byte
    multiple stride, which the kernel's operand contract declares)."""
    if kv_layout == "NHD":
        total = kv.shape[0]
        base = kv.permute(2, 0, 1)
    else:
        total = kv.shape[1]
        base = kv.permute(2, 1, 0)
    return base.as_strided(
        (D, total, Hkv, 1),
        (base.stride(0), base.stride(1), base.stride(2), max(8, total * Hkv * D)),
        base.storage_offset(),
    )


class WorkspaceTooSmall(RuntimeError):
    """The caller's workspace cannot hold the split-KV partials of this batch; ``needed`` is the byte count."""

    def __init__(self, needed: int, given: int):
        super().__init__(
            f"workspace of {given} bytes; the split partials of this batch need {needed}"
        )
        self.needed = needed


class _Planned:
    def __init__(
        self,
        plan,
        desc: torch.Tensor,
        groups: torch.Tensor,
        ws: torch.Tensor,
        scratch: Optional[torch.Tensor] = None,
        fixup_groups: Optional[torch.Tensor] = None,
        fold_counters: Optional[torch.Tensor] = None,
    ):
        self.plan, self.desc, self.groups, self.ws = plan, desc, groups, ws
        self.scratch = scratch  # the per-CTA O_m scratch (D = 256)
        self.ragged = False  # the packed-token KV layout (set by plan_attention)
        # what the descriptor table was built for; run_attention refuses anything else (the table holds one unit per KV head
        # and per Q tile, the configuration decides pairing and the unit body)
        self.heads: Optional[tuple] = None  # (Hq, Hkv)
        self.total_q: Optional[int] = None
        self.window_left: Optional[int] = None
        self.config: Optional[AttnConfig] = None
        self.page: Optional[int] = None
        self.fixup_groups = (
            fixup_groups if fixup_groups is not None else groups
        )  # the groups the fixup kernel folds (the FMHA body folds the others in-kernel)
        self.n_fixup_groups = len(getattr(plan, "fixup_groups", plan.groups))
        self.fold_counters = fold_counters  # the FMHA body's per-group arrival counters (in-kernel fold); zero between launches


def plan_attention(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    Hq: int,
    Hkv: int,
    D: int,
    page: int,
    window_left: int = -1,
    config: Optional[AttnConfig] = None,
    workspace: Optional[torch.Tensor] = None,
    device=None,
    sm_count: Optional[int] = None,
    ragged: bool = False,
) -> _Planned:
    """``sm_count`` (default: the device's) only steers the ``auto`` unit-kind choice (I10: a launch
    parameter, never a quantity that reaches the bits). ``ragged``: the K / V are packed along the token dimension (``kv_indptr``
    holds token offsets, ``kv_last_page_len`` is ignored): planned as page size 1, run through the ragged producer."""
    if ragged:
        page = 1
        kv_last_page_len = torch.ones(max(0, kv_indptr.numel() - 1), dtype=torch.int32)
    check_supported(D, Hq, Hkv, page)
    config = config or default_config(D=D)
    assert config.well_formed(D), (config, D)
    device = device or qo_indptr.device
    sm = (
        sm_count
        if sm_count is not None
        else (
            torch.cuda.get_device_properties(device).multi_processor_count
            if torch.device(device).type == "cuda"
            else 204
        )
    )
    plan = build_plan(
        qo_indptr.cpu().tolist(),
        kv_indptr.cpu().tolist(),
        kv_last_page_len.cpu().tolist(),
        Hq,
        Hkv,
        D,
        page,
        window_left,
        config.split_policy,
        sm,
        merge=config.merge,
        small_swap=config.small == "swap" and D in SWAPS_DIMS,
        swap_max_rows=SWAPS_MAX_ROWS,
    )  # Swaps batches at D = 256 take the
    # D = 128 rules (TMEM merges)
    desc = torch.tensor(
        plan.desc if plan.desc else [[0] * DESC_FIELDS],
        dtype=torch.int32,
        device=device,
    )
    if (
        config.q2
        and len(plan.desc) >= Q2_MIN_UNITS_PER_SM * sm
        and not (config.small == "swap" and D in SWAPS_DIMS and swaps_n_pad(plan))
    ):
        paired = pair_units(plan.desc)
        plan.num_launch = len(paired) // 2
        desc = torch.tensor(paired, dtype=torch.int32, device=device)
    groups = torch.tensor(
        plan.groups if plan.groups else [[0] * GROUP_FIELDS],
        dtype=torch.int32,
        device=device,
    )
    fx = getattr(plan, "fixup_groups", plan.groups)
    fixup_groups = torch.tensor(
        fx if fx else [[0] * GROUP_FIELDS], dtype=torch.int32, device=device
    )
    need = max(1, plan.ws_rows) * ws_row(D)
    if (
        workspace is None or plan.ws_rows == 0
    ):  # no split group: no partials; a one-row placeholder
        ws = torch.empty(need, dtype=torch.float32, device=device)
    else:
        if workspace.numel() * workspace.element_size() < need * 4:
            raise WorkspaceTooSmall(
                need * 4, workspace.numel() * workspace.element_size()
            )
        ws = workspace.view(torch.uint8).reshape(-1)[: need * 4].view(torch.float32)
    # the per-CTA O_m scratch of fused multi-chunk units when O_m is not in TMEM (D = 256: blocks/operands.om_scratch_blocks); one 128 x D fp32
    # block per CTA in the row-interleaved layout of operands.OM_VEC, touched once per chunk end, L2-resident (26 MB at 204 CTAs x D = 256)
    from .blocks.operands import om_scratch_blocks

    multi = any(d[6] == 0 and d[5] - d[4] > 1 for d in plan.desc)
    scr = 1
    if om_scratch_blocks(D) and multi:
        n_cta = max(1, min(plan.num_units, max(sm, config.grid)))
        scr = n_cta * om_scratch_blocks(D) * 128 * D
    # the in-kernel fold's arrival counters: one per group, zero at every launch (the last arriver resets its group's counter)
    planned = _Planned(
        plan,
        desc,
        groups,
        ws,
        torch.empty(scr, dtype=torch.float32, device=device),
        fixup_groups,
        torch.zeros(max(1, len(plan.groups)), dtype=torch.int32, device=device),
    )
    planned.ragged = ragged
    planned.heads = (int(Hq), int(Hkv))
    planned.total_q = int(qo_indptr[-1]) if qo_indptr.numel() else 0
    planned.window_left = int(window_left)
    planned.config = config
    planned.page = int(page)
    return planned


def workspace_bytes(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: Optional[torch.Tensor],
    Hq: int,
    Hkv: int,
    D: int,
    page: int,
    window_left: int = -1,
    config: Optional[AttnConfig] = None,
    device=None,
    ragged: bool = False,
) -> int:
    """The float workspace bytes ``plan_attention`` needs for this batch's split-KV partials (0 when the plan has no split
    group); the planner runs on the host, nothing is allocated."""
    if ragged:
        page = 1
        kv_last_page_len = torch.ones(max(0, kv_indptr.numel() - 1), dtype=torch.int32)
    assert kv_last_page_len is not None
    check_supported(D, Hq, Hkv, page)
    config = config or default_config(D=D)
    device = device or qo_indptr.device
    sm = (
        torch.cuda.get_device_properties(device).multi_processor_count
        if torch.device(device).type == "cuda"
        else 204
    )
    plan = build_plan(
        qo_indptr.cpu().tolist(),
        kv_indptr.cpu().tolist(),
        kv_last_page_len.cpu().tolist(),
        Hq,
        Hkv,
        D,
        page,
        window_left,
        config.split_policy,
        sm,
        merge=config.merge,
        small_swap=config.small == "swap" and D in SWAPS_DIMS,
        swap_max_rows=SWAPS_MAX_ROWS,
    )
    return plan.ws_rows * ws_row(D) * 4


def _output_overlaps_kv(out: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Conservative byte-span overlap for a contiguous output and nonnegative-strided caches.

    Disjoint slices of one allocation are safe. Strided cache spans include any holes,
    so interleaved views can still be refused even when their live elements are disjoint.
    """
    if not out.numel():
        return False
    out_start = out.data_ptr()
    out_end = out_start + out.numel() * out.element_size()
    for cache in (k, v):
        if not cache.numel():
            continue
        start = cache.data_ptr()
        if start >= out_end:
            continue
        extent = (
            cache.numel()
            if cache.is_contiguous()
            else 1
            + sum((n - 1) * s for n, s in zip(cache.shape, cache.stride(), strict=True))
        )
        if out_start < start + extent * cache.element_size():
            return True
    return False


def run_attention(
    planned: _Planned,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    window_left: int = -1,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    config: Optional[AttnConfig] = None,
    ragged: bool = False,
) -> torch.Tensor:
    total_q, Hq, D = q.shape
    if config is None:
        config = planned.config or default_config(D=D)
    elif planned.config is not None and config != planned.config:
        raise ValueError(
            f"run_attention with {config.describe()} over a plan built for {planned.config.describe()}: the descriptor table is "
            "specific to the configuration (pairing, unit body); plan with the same configuration"
        )
    assert bool(getattr(planned, "ragged", False)) == bool(ragged), (
        "plan_attention and run_attention must agree on ragged"
    )
    if planned.window_left is not None and int(window_left) != planned.window_left:
        raise ValueError(
            f"window_left {window_left} at run_attention over a plan built for window_left {planned.window_left}"
        )
    if ragged:
        # Packed K / V: (total_kv, Hkv, D) under NHD, (Hkv, total_kv, D) under HND.
        Hkv = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[0]
        for cache in (k_cache, v_cache):
            assert (
                cache.shape[1:] == (Hkv, D)
                if kv_layout == "NHD"
                else cache.shape[0] == Hkv and cache.shape[2] == D
            ), cache.shape
    else:
        page = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[2]
        if planned.page is not None and int(page) != planned.page:
            raise ValueError(
                f"a KV cache of page size {page} at run_attention over a plan built for page size {planned.page}"
            )
        Hkv = k_cache.shape[2] if kv_layout == "NHD" else k_cache.shape[1]
        for cache in (k_cache, v_cache):
            assert cache.shape[1:] == (
                (page, Hkv, D) if kv_layout == "NHD" else (Hkv, page, D)
            ), cache.shape
    if planned.total_q is not None and int(total_q) != planned.total_q:
        raise ValueError(
            f"{total_q} query rows at run_attention over a plan built for {planned.total_q} (the units write the planned rows)"
        )
    if planned.heads is not None and (int(Hq), int(Hkv)) != planned.heads:
        raise ValueError(
            f"{Hq} query heads over {Hkv} KV heads at run_attention over a plan built for {planned.heads[0]} over "
            f"{planned.heads[1]} (one unit per KV head: the table does not cover other head counts)"
        )
    assert (
        q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and v_cache.dtype == torch.bfloat16
    )
    assert q.is_contiguous(), "Q must be a packed (total_q, Hq, D) tensor"
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError(
            "the K / V caches must have a unit stride along the head dim (the kernels are compiled for it)"
        )
    for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
        if t.data_ptr() % 16:
            raise ValueError(
                f"{name} must be 16-byte aligned (the TMA descriptors' requirement); its base address is not"
            )
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        if any(st % 8 for st in t.stride()[:-1]):
            raise ValueError(
                f"{name} outer strides must be multiples of 8 elements (16 bytes: the TMA descriptors' requirement); got {tuple(t.stride())}"
            )
    supplied_out = out is not None
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
    if out.data_ptr() % 16:
        raise ValueError(
            "out must be 16-byte aligned (the TMA descriptors' requirement)"
        )
    if supplied_out and _output_overlaps_kv(out, k_cache, v_cache):
        raise ValueError(
            "out must not alias the K / V cache: CTAs store output rows while others still read pages"
        )
    c = check_sm_scale(sm_scale)
    return _run_attention_validated(
        planned,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        c,
        window_left,
        kv_layout,
        out,
        config,
        ragged,
    )


def _run_attention_validated(
    planned: _Planned,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    c: float,
    window_left: int,
    kv_layout: str,
    out: torch.Tensor,
    config: AttnConfig,
    ragged: bool,
) -> torch.Tensor:
    """Launch after the public engine or wrapper has checked inputs; ``c`` is the binary32 log2 scale."""
    from .blocks.compile import get_compiled_attention
    from .fixup import MAX_GRID_Y, get_fixup
    from .spec import TILE_KEYS

    total_q, Hq, D = q.shape
    if ragged:
        page = TILE_KEYS
        Hkv = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[0]
        kv_k = _kv_view_ragged(k_cache, kv_layout, Hkv, D)
        kv_v = _kv_view_ragged(v_cache, kv_layout, Hkv, D)
    else:
        page = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[2]
        Hkv = k_cache.shape[2] if kv_layout == "NHD" else k_cache.shape[1]
        kv_k = _kv_view(k_cache, kv_layout, Hkv, page, D)
        kv_v = _kv_view(v_cache, kv_layout, Hkv, page, D)
    g = Hq // Hkv
    plan = planned.plan
    if total_q == 0 or plan.num_units == 0:
        return out  # nothing to compute: no TMA descriptor over an empty query extent
    assert plan.g == g
    q_view = q.permute(2, 1, 0)  # (D, Hq, total_q)
    o_view = out.view(total_q * Hq, D)
    # the TVM-FFI object takes torch tensors (views with their strides) and Python scalars
    sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = config.grid if config.grid else min(plan.num_units, sm)
    grid = max(1, min(grid, plan.num_units))
    assert planned.fold_counters is not None and planned.scratch is not None, (
        "plan_attention must be called with the same (FMHA) config"
    )
    n_pad = swaps_n_pad(plan) if (config.small == "swap" and D in SWAPS_DIMS) else 0
    if n_pad:  # every unit has <= 32 live rows: the Swaps-AB kernel (same descriptors, same bits)
        from .swaps.compile import get_compiled_swaps

        from .swaps.operands import arch_smem_cap

        # the ring depth follows the execution device's shared memory: part of the kernel's identity, in-process and on disk
        kernel = get_compiled_swaps(D, g, page, n_pad, ragged, arch_smem_cap(q.device))
        kernel(
            q_view,
            kv_k,
            kv_v,
            o_view,
            planned.ws,
            planned.groups,
            planned.fold_counters,
            planned.desc,
            kv_indptr,
            kv_indices,
            float(c),
            float(-c),
            int(window_left),
            int(plan.num_units),
            int(grid),
        )
    else:
        n_launch = getattr(plan, "num_launch", None) if config.q2 else None
        if (
            config.q2 and n_launch is None
        ):  # too few units to pair: the one-warpgroup body (same bits)
            kernel = get_compiled_attention(
                D, g, page, config.kv_slots, 1, False, ragged
            )
        else:
            kernel = get_compiled_attention(
                D, g, page, config.kv_slots, config.softmax_wgs, config.q2, ragged
            )
        if n_launch is not None:  # the paired placement walks pairs of units
            grid = max(
                1, min(config.grid if config.grid else min(n_launch, sm), n_launch)
            )
        if (
            planned.scratch.numel() > 1
        ):  # O_m in the per-CTA global scratch: om_scratch_blocks(D) blocks per CTA
            from .blocks.operands import om_scratch_blocks

            grid = max(
                1,
                min(grid, planned.scratch.numel() // (om_scratch_blocks(D) * 128 * D)),
            )
        kernel(
            q_view,
            kv_k,
            kv_v,
            o_view,
            planned.ws,
            planned.scratch,
            planned.groups,
            planned.fold_counters,
            planned.desc,
            kv_indptr,
            kv_indices,
            float(c),
            float(-c),
            int(window_left),
            int(n_launch if n_launch is not None else plan.num_units),
            int(grid),
        )
    fx = getattr(plan, "fixup_groups", plan.groups)
    if fx:
        max_rows = max(gr[1] for gr in fx)
        get_fixup(D, g)(
            planned.ws,
            o_view,
            planned.fixup_groups,
            float(c),
            float(-c),
            int(window_left),
            int(Hq),
            int(len(fx)),
            int(min(len(fx), MAX_GRID_Y)),
            int(max_rows),
        )
    return out


def attention_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    config: Optional[AttnConfig] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-call paged attention (prefill and decode alike; causal, bottom-right aligned)."""
    total_q, Hq, D = q.shape
    page = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[2]
    Hkv = k_cache.shape[2] if kv_layout == "NHD" else k_cache.shape[1]
    sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
    planned = plan_attention(
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        Hq,
        Hkv,
        D,
        page,
        window_left,
        config,
        workspace,
        q.device,
    )
    return run_attention(
        planned,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        sm_scale,
        window_left,
        kv_layout,
        out,
        config,
    )
