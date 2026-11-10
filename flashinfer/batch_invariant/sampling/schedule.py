# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``lse.row`` config space, ownership maps and placements (the row engine's shape on vocabulary rows).

Pure integer logic shared by the CuTe kernel (trace time), the symbolic tracer and the Lean model. Canonical constants:
``V = 8`` elements per slot, 32 lanes, 256 elements per block, ``Kp = pow2ceil(blocks)``; the flat index of an element
is ``(block * 32 + lane) * 8 + elem`` and the tree is elem (fold) -> block (pairwise) -> lane (pairwise), so the tree
depends on the vocabulary size only -- never on the batch, the config or the device.

Configs: ``threads_per_row`` 32 .. 1024 (one or more warps per row; ``W`` warps own aligned block ranges), rows per
CTA, the vector width, whether the row's fragments stay in registers between the passes (``regs``), are re-read from
global memory (``reread``, L2-hot) or are staged once in shared memory (``smem``), and the cluster size (CTAs per row).
Every member computes the canonical dataflow; ``default_config`` only picks.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Optional

V = 8
LANES = 32
BLOCK = V * LANES  # 256 elements
THREADS = (32, 64, 128, 256, 512, 1024)


def pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def pow2floor(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def num_slots(N: int) -> int:
    return -(-N // V)


def num_blocks(N: int) -> int:
    return -(-num_slots(N) // LANES)


def padded_blocks(N: int) -> int:
    """``Kp``: the block count rounded up to a power of two (absent blocks are identity leaves of the pairwise tree)."""
    return pow2ceil(num_blocks(N))


# The bounds of the enumerated configuration space; each is an empirical choice of what the space contains.
KEPT_BYTES_MAX = 512  # the register form is enumerated while a thread keeps at most this many input bytes.
REREAD_MIN_ELEMS = 64  # the re-read form is enumerated where a thread holds at least this many elements.
CLUSTERS = (2, 4, 8, 16)  # the multi-CTA members (one row per cluster).
SMEM_BUDGET = (
    200 * 1024
)  # bytes of dynamic shared memory a staged share may take (the exchange buffers come on top).
SMEM_MIN_ELEMS = 64  # the staged form is enumerated where a thread's share is at least this many elements.
VOCAB_COVERED = (
    1 << 21
)  # every vocabulary up to this has a member for every dtype and alignment (an odd fp32 row of
#   2^21 + 1 elements has none: its tree pads to 16384 blocks, a thread of the widest cluster would keep 1 KB of fragments).
LOADS_PER_THREAD_MAX = 1024  # the re-read and staged forms unroll a thread's loads per pass: a member issues at most this
#   many (device-free compile of T32_R1_V1_RR: 256 loads 13 s, 1024 loads 39 s, 2048 loads 2 min, 4096 loads 20+ min, 8192 a
#   compiler crash). The register form is bounded by KEPT_BYTES_MAX, the staged form by SMEM_BUDGET as well.


@dataclass(frozen=True)
class LSEConfig:
    """One kernel configuration. All valid configs compute the canonical dataflow."""

    threads_per_row: int  # 32 .. 1024 per CTA: W = T / 32 warps per row (per CTA)
    rows_per_cta: int = 1
    vec_elems: int = V  # elements per vector load: 1, 2, 4 or 8 (16 bytes at most)
    stage: str = "regs"  # "regs": fragments kept between the passes; "reread": re-read from global (L2-hot); "smem":
    #   the CTA's share of the row staged once in shared memory with cp.async, every pass reading it back
    cluster: int = 1  # CTAs per row (1, 2, 4, 8, 16): CTA c of the cluster owns blocks [c Kp / C, (c + 1) Kp / C) and
    #   the per-lane partials cross the cluster through distributed shared memory (the engine's `cta` factor of the
    #   block digit); the tree is the same, the row is read once per pass

    @property
    def keep_regs(self) -> bool:
        return self.stage == "regs"

    @property
    def warps_per_row(self) -> int:  # W
        return self.threads_per_row // LANES

    @property
    def num_threads(self) -> int:
        return self.threads_per_row * self.rows_per_cta

    def blocks_per_warp(self, N: int) -> int:
        return padded_blocks(N) // (self.warps_per_row * self.cluster)

    def blocks_per_cta(self, N: int) -> int:
        return padded_blocks(N) // self.cluster

    def elems_per_thread(self, N: int) -> int:
        return V * self.blocks_per_warp(N)

    def validate(self, N: int, elem_bytes: Optional[int] = None) -> None:
        """Raise ``ValueError`` unless the config is a well-formed member for vocabulary ``N``; with ``elem_bytes`` the
        staged form is checked against the element size (16-byte vectors on 16-byte rows within the budget) instead of
        the element-size-agnostic lower bound."""
        if self.threads_per_row not in THREADS:
            raise ValueError(f"threads_per_row={self.threads_per_row}")
        if self.vec_elems not in (1, 2, 4, 8):
            raise ValueError(f"vec_elems={self.vec_elems}")
        if self.rows_per_cta < 1 or self.num_threads > 1024:
            raise ValueError(
                f"rows_per_cta={self.rows_per_cta} (num_threads={self.num_threads})"
            )
        if self.cluster not in (1,) + CLUSTERS:
            raise ValueError(f"cluster={self.cluster}")
        if padded_blocks(N) % (self.warps_per_row * self.cluster):
            raise ValueError(
                f"warps_per_row x cluster = {self.warps_per_row} x {self.cluster} does not divide Kp={padded_blocks(N)}"
            )
        if self.stage not in ("regs", "reread", "smem"):
            raise ValueError(f"stage={self.stage}")
        if self.stage == "smem":
            if self.rows_per_cta != 1:
                raise ValueError("smem staging holds one row per CTA")
            if elem_bytes is not None:
                if not self.smem_stage_ok(N, elem_bytes):
                    raise ValueError(
                        f"smem staging needs 16-byte vectors ({self.vec_elems} x {elem_bytes} bytes) on 16-byte rows "
                        f"and a tile within {SMEM_BUDGET} bytes (this one: {self.smem_tile_bytes(N, elem_bytes)})"
                    )
            elif (
                self.smem_tile_bytes(N, 2 if self.vec_elems == 8 else 4) > SMEM_BUDGET
                and self.smem_tile_bytes(N, 1) > SMEM_BUDGET
            ):
                raise ValueError("smem tile over budget")

    def is_valid(self, N: int) -> bool:
        try:
            self.validate(N)
            return True
        except ValueError:
            return False

    def owned_blocks(self, N: int, w: int, c: int = 0) -> range:
        """Blocks owned by warp ``w`` of CTA ``c`` of a row: an aligned, contiguous range."""
        n = self.blocks_per_warp(N)
        base = c * self.blocks_per_cta(N)
        return range(base + w * n, base + (w + 1) * n)

    def smem_tile_bytes(self, N: int, elem_bytes: int) -> int:
        """The staged share of the row (``Kp / C`` blocks of 256 elements) in bytes."""
        return self.rows_per_cta * self.blocks_per_cta(N) * BLOCK * elem_bytes

    def smem_stage_ok(self, N: int, elem_bytes: int) -> bool:
        """cp.async staging needs 16-byte vectors on 16-byte aligned rows and a tile under the budget."""
        return (
            self.vec_elems * elem_bytes == 16
            and (N * elem_bytes) % 16 == 0
            and self.rows_per_cta == 1
            and self.smem_tile_bytes(N, elem_bytes) <= SMEM_BUDGET
        )

    def describe(self) -> str:
        return (
            f"T{self.threads_per_row}_R{self.rows_per_cta}_V{self.vec_elems}"
            + {"regs": "", "reread": "_RR", "smem": "_SM"}[self.stage]
            + (f"_C{self.cluster}" if self.cluster > 1 else "")
        )

    def placement(self, N: int):
        """This config as a :class:`flashinfer.batch_invariant.canon.Placement` of the row reductions: CTA ``c`` of the
        cluster owns blocks ``[c Kp / C, (c + 1) Kp / C)`` and warp ``w`` of it ``[w Kp / (W C), (w + 1) Kp / (W C))``
        within (block = reg x warp x cta), lane = lane, elem = reg."""
        from ..canon import Placement

        self.validate(N)
        return Placement.make(
            {
                "elem": [(V, "reg")],
                "block": [
                    (self.blocks_per_warp(N), "reg"),
                    (self.warps_per_row, "warp"),
                    (self.cluster, "cta"),
                ],
                "lane": [(LANES, "lane")],
            },
            rows_per_cta=self.rows_per_cta,
            vec_elems=self.vec_elems,
            stage=self.stage,
        )


def max_vec_elems(N: int, elem_bytes: int) -> int:
    """Largest power-of-two vector (<= 8 elements, <= 16 bytes) that keeps row loads aligned."""
    v = 8
    while v > 1 and ((N % v) or (v * elem_bytes > 16)):
        v //= 2
    return v


def all_valid_configs(
    N: int, elem_bytes: int = 2, max_threads: int = 1024
) -> List[LSEConfig]:
    """Every config the test-suite checks for config invariance: the single-CTA members at 1 / 2 / 4 rows per CTA
    (register and re-read staging), the cluster members at one row per cluster, the staged members."""
    out = []
    vmax = max_vec_elems(N, elem_bytes)
    vecs = sorted({vmax, max(1, vmax // 2)})
    for T in THREADS:
        for rows in (1, 2, 4):
            for vec in vecs:
                cfg = LSEConfig(T, rows, vec)
                if cfg.num_threads > max_threads or not cfg.is_valid(N):
                    continue
                e = cfg.elems_per_thread(N)
                if e * elem_bytes <= KEPT_BYTES_MAX:
                    out.append(cfg)
                if e >= REREAD_MIN_ELEMS and e // vec <= LOADS_PER_THREAD_MAX:
                    out.append(LSEConfig(T, rows, vec, stage="reread"))
        for C in CLUSTERS:
            cfg = LSEConfig(T, 1, vmax, cluster=C)
            if (
                cfg.is_valid(N)
                and cfg.elems_per_thread(N) * elem_bytes <= KEPT_BYTES_MAX
            ):
                out.append(cfg)
        for C in (1,) + CLUSTERS:
            cfg = LSEConfig(T, 1, vmax, "smem", C)
            if (
                cfg.is_valid(N)
                and cfg.smem_stage_ok(N, elem_bytes)
                and cfg.elems_per_thread(N) >= SMEM_MIN_ELEMS
                and cfg.elems_per_thread(N) // vmax <= LOADS_PER_THREAD_MAX
            ):
                out.append(cfg)
    return out


def parse_config(desc: str) -> LSEConfig:
    """Inverse of :meth:`LSEConfig.describe` (``T256_R1_V8[_RR|_SM][_C<n>]``)."""
    import re

    m = re.fullmatch(r"T(\d+)_R(\d+)_V(\d+)(_RR|_SM)?(_C(\d+))?", desc)
    if not m:
        raise ValueError(desc)
    stage = {None: "regs", "_RR": "reread", "_SM": "smem"}[m.group(4)]
    return LSEConfig(
        int(m.group(1)), int(m.group(2)), int(m.group(3)), stage, int(m.group(6) or 1)
    )


# ---------------------------------------------------------------------------------------------------- the dispatcher
# The rule's constants are empirical (chosen on SM100 / SM107 parts); one sentence states what each one decides.
BYTES_PER_THREAD = 256  # latency regime: about this many input bytes per thread of a CTA's share of the row.
THREADS_PER_CTA_TARGET = (
    128  # single-CTA rows: rows per CTA = max(1, this / T), at most four.
)
REGS_BUDGET_BYTES = 131072  # fragments stay in registers while kept bytes x threads per CTA stay within this (about half the register file).
CLUSTER_ROWS = 512  # fewer rows than SMs: C = min(MAX_CLUSTER, pow2floor(CLUSTER_ROWS / M)) CTAs per row.
MAX_CLUSTER = 16  # the largest cluster the rule picks.
CLUSTER_MIN_SHARE = 4096  # a CTA's share of a row is at least this many bytes (shorter rows stay single-CTA).
ROW_OUT_SHARE = (
    128 * 1024
)  # throughput regime, three-pass members (a row output): one CTA per this many bytes of the row.
ROW_SCALAR_SHARE = (
    64 * 1024
)  # throughput regime, two-pass members (a row scalar): one CTA per this many bytes of the row.
TP_BYTES_PER_THREAD = (
    512  # throughput regime: about this many input bytes per thread of a CTA's share.
)
TP_MAX_CLUSTER = 8  # throughput regime: at most this many CTAs per row.
TP_PAD_MAX = 0.25  # throughput regime: no cluster while the tree's padding (Kp - blocks) / Kp exceeds this (a cluster's idle
#   CTAs wait at its barriers holding SMs); the three-pass members then take the single-CTA pick below, the two-pass members
TP_PAD_THREADS = (
    512  # ... a single CTA of this many threads re-reading the row from L2 ...
)
TP_PAD_VEC = 4  # ... with vector loads of at most this many elements (the wider vectors lose at that share).


def _clamp_T(T: int, Kp: int) -> int:
    T = min(1024, max(32, pow2ceil(T)))
    while T > 32 and Kp % (T // LANES):
        T //= 2
    return T


def default_config(
    N: int, M: int, elem_bytes: int, sm_count: int, row_output: bool = True
) -> LSEConfig:
    """A closed-form pick (no table). Fewer rows than SMs: a cluster of ``min(16, pow2floor(CLUSTER_ROWS / M))`` CTAs
    per row, ~256 input bytes per thread of each CTA's share. Otherwise one CTA per ``ROW_OUT_SHARE`` (a row output) or
    ``ROW_SCALAR_SHARE`` (a row scalar) bytes of the row, in clusters of up to 8, at ~512 input bytes per thread; or,
    for short rows, one CTA per row with up to four rows per CTA of ~128 threads. Fragments stay in registers while
    they take at most half the register file per thread and are re-read from L2 beyond. Every pick is a member of
    :func:`all_valid_configs`, so it never changes bits."""
    return _rule(int(N), int(M), int(elem_bytes), int(sm_count), bool(row_output))


@functools.lru_cache(maxsize=65536)
def _rule(
    N: int, M: int, elem_bytes: int, sm: int, row_output: bool = True
) -> LSEConfig:
    vec = max_vec_elems(N, elem_bytes)
    Kp = padded_blocks(N)
    row_bytes = N * elem_bytes
    space = all_valid_configs(N, elem_bytes)
    if not space:
        raise ValueError(
            f"no configuration of the class covers a vocabulary of {N} with {elem_bytes}-byte elements (every vocabulary up "
            f"to {VOCAB_COVERED} is covered; beyond it the row's alignment decides)"
        )

    def staged(T, rows, C):
        cfg0 = LSEConfig(T, rows, vec, cluster=C)
        kept = cfg0.elems_per_thread(N) * elem_bytes
        return LSEConfig(
            T,
            rows,
            vec,
            "regs" if kept * cfg0.num_threads <= REGS_BUDGET_BYTES else "reread",
            C,
        )

    def first_member(T, C):
        if LSEConfig(T, 1, vec, cluster=C).is_valid(N):
            for cand in (
                staged(T, 1, C),
                LSEConfig(T, 1, vec, "regs", C),
                LSEConfig(T, 1, vec, "reread", C),
            ):
                if cand in space:
                    return cand
        return None

    if sm > M:
        C = min(
            MAX_CLUSTER,
            max(1, pow2floor(CLUSTER_ROWS // max(M, 1))),
            max(1, pow2floor(row_bytes // CLUSTER_MIN_SHARE)),
        )
        while C > 1:
            T = _clamp_T(-(-(row_bytes // C) // BYTES_PER_THREAD), Kp // C)
            cand = first_member(T, C)
            if cand is not None:
                return cand
            C //= 2
    else:
        share = ROW_OUT_SHARE if row_output else ROW_SCALAR_SHARE
        C = min(TP_MAX_CLUSTER, max(1, pow2ceil(-(-row_bytes // share))))
        if (Kp - num_blocks(N)) > TP_PAD_MAX * Kp:
            if not row_output:
                cand = LSEConfig(TP_PAD_THREADS, 1, min(vec, TP_PAD_VEC), "reread", 1)
                if cand in space:
                    return cand
            T1 = _clamp_T(-(-row_bytes // TP_BYTES_PER_THREAD), Kp)
            T1 = max(T1, 128) if Kp >= 4 else T1
            single = first_member(T1, 1)
            if single is not None:
                return single
            # no single-CTA member within LOADS_PER_THREAD_MAX (a long odd row): the cluster path below
        while C >= 1:
            T = _clamp_T(-(-(row_bytes // C) // TP_BYTES_PER_THREAD), Kp // C)
            T = max(T, 128) if Kp // C >= 4 else T
            cand = first_member(T, C)
            if cand is not None:
                return cand
            C //= 2
    T = _clamp_T(-(-row_bytes // BYTES_PER_THREAD), Kp)
    rows = 1 if sm > M else max(1, min(4, THREADS_PER_CTA_TARGET // T))
    cfg = staged(T, rows, 1)
    if cfg in space:
        return cfg
    for alt in (
        LSEConfig(T, rows, vec, "regs"),
        LSEConfig(T, rows, vec, "reread"),
        LSEConfig(T, 1, vec, "regs"),
        LSEConfig(T, 1, vec, "reread"),
    ):
        if alt in space:
            return alt
    # no regime candidate is a member (a long odd row whose single-CTA re-read members exceed the loads bound):
    # the member with the fewest loads per thread, then the fewest CTAs per row, then the most threads
    return min(
        space,
        key=lambda c: (
            c.elems_per_thread(N) // c.vec_elems,
            c.cluster,
            -c.threads_per_row,
        ),
    )


_sm_count_cache: dict = {}


def device_sm_count(device) -> int:
    import torch

    key = torch.device(device).index
    if key not in _sm_count_cache:
        _sm_count_cache[key] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return _sm_count_cache[key]


_ = Optional  # typing re-export for the API module
