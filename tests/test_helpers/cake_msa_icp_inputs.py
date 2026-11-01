"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CPU oracle, input generators and golden vectors for the Cake MSA indexer
context-parallel (ICP) path (FlashInfer issue #7896).

The contract is the one of vLLM-project/MSA#46 (``f380d0e8``) as consumed by
vLLM#61929 / vLLM#63671 (``a76900d6``): FP8 E4M3 index Q ``[T, H, 128]``
replicated on every rank, index K sharded by context, ranking on logical
128-token blocks (B128), Top16 per ``(token, head)``.

Layouts (``P`` page size, ``W`` world size, ``R = P / W`` local rows, ``r``
rank, ``c`` local column = page index in the block table)::

    layout              global row of local (c, row)   gid of local column c   s, o
    P128 TP2/TP4        c*128 + row*W + r              c                       1, 0
    P256 TP2            c*256 + r*128 + row            2c + r                  2, r
    P256 TP4            c*256 + r*64 + row             2c + r//2               2, r//2

``s = index_scale``, ``o = index_offset = r*s//W``;
``LocalPrefixCount(g, s, o) = 0 if g <= o else (g - 1 - o)//s + 1`` is the
number of local columns whose gid is below ``g``.

Score semantics (all scorers): raw fp32 ``max_j q.k`` over the local key rows
``j`` of the column that are visible to the query (``global_row <= q_pos`` and
``global_row < kv_len``); NaN-ignoring max starting from ``-inf`` (a column
with no visible row is ``-inf``); ``-0.0`` is canonicalized to ``+0.0``.  The
oracle accumulates in float64; on the integer and narrow-exponent fixtures of
this module every fp32 summation order is exact, so kernels are compared with
``atol = 0``.

Canonical order (every selector, every merge)::

    canon(x) = -inf if isnan(x) else (+0.0 if x == 0 else x)
    ordk(x)  = 32-bit order key of canon(x) (larger is better)
    forced   -> ORD_PINF = ordk(+inf)  (equal to a real +inf; the lower id wins)
    order    = key descending, then global id ascending

Select (``select_prefill`` carrier rows and ``select_publish`` rows, one row
per ``(t, h)``)::

    ng = min(max(nvalid, 0), 8192, K*s);  L = LocalPrefixCount(ng, s, o)
    candidates = local columns c < L;  gid(c) = c*s + o
    forced(c)  = gid < force_begin  or  gid >= max(ng - force_end, 0)
    pick the first 16 in canonical order, emit ascending by gid:
    words[0:16]  = RAW f32 bits of scores[c] (raw, even for NaN / -0 / forced)
    words[16:32] = gid; padding = (0xFF800000, -1)
    word[32]     = 1 (prefill carrier marker / legacy ready flag)

Publish (decode, ``recv_bases[h // hpr]`` of the owner rank): row index
``((slot*W + rank)*hpr + h % hpr)*capacity + t``; legacy rows are 33 x i32 at
word offset ``row*33`` (publisher waits for ``ready == 0``, writes 32 words,
release-stores ``ready = 1``); tagged rows are 32 x u64 ``(gen << 32) |
payload`` at u64 offset ``row*32`` of the region that starts
``legacy_words*4`` bytes after the base, with ``gen = generations[slot, h, t] +
1`` modulo 2**32 skipping 0.

Merge (owner, ``[T, hpr, 16]`` int32)::

    n = min(nvalid, 8192); a candidate is valid iff 0 <= id < n
    NaN -> -inf; duplicate ids keep the max; forced (id < init or
    id >= max(0, n - local)) -> +inf; canonical order; first 16 written
    ascending by id, -1 padding; prefill rows with n == 0 -> [0, -1, ...];
    decode rows with n <= 0 -> all -1.  Legacy: wait ready == 1 on all W
    sources then clear them; tagged: wait both words == expected + 1 (0
    skipped) then store expected.

Containment: when every rank selects with the canonical order and the same
``ng``, ``merge(exchange(select_r)) == oracle_global`` bit for bit (checked by
:func:`self_test`).

Intentional deviations from the MSA#46 / vLLM reference (DESIGN.md section
2.3); the Cake kernels follow this oracle, not the reference, and the E2E gate
attributes cross-arm id differences to these classes:

* D1  MSA ``select_publish`` breaks exact cutoff ties toward the HIGHER local
  id when ``L <= 128`` (``MakeFinalCandidateKey`` sorts by staging position,
  ``sparse_topk_select.cuh:645-666,1166-1177``) and in atomic order when
  ``L > 128``.  Here every selector uses the lower id (as MSA
  ``select_prefill`` and the vLLM merge do).
* D2  MSA fills the final threshold bin in atomic order when more than 128
  candidates share the cutoff (``cuh:413-425``), prefill included: cutoff ids
  are not deterministic there.  Here: lower id.
* D3  MSA histogram bins ``+0`` and ``-0`` apart when ``L > 128``
  (``cuh:156-187``).  Here ``-0 == +0`` everywhere.
* D4  MSA orders ``+NaN`` above ``+inf`` and ``-NaN`` below ``-inf`` (raw bit
  key, ``cuh:669-680``); the vLLM merge maps NaN to ``-inf``
  (``index_decode_score_sharded.py:632``).  Here NaN ranks as ``-inf`` in every
  selector; the raw NaN bits are still published.
* D5  MSA forces with ``FLT_MAX`` in selection (``cuh:297,815,841``), so a real
  ``+inf`` score outranks a forced block locally; the vLLM merge forces with
  ``+inf``.  Here forced keys equal ``ordk(+inf)`` (ties: lower id).
  Unreachable from FP8 Q.K (finite), reachable through the select API.
* D6  MSA clamps the select valid count to ``K*s`` only
  (``sparse_topk_select.cu:211,250``); the merge clamps to 8192
  (``index_decode_score_sharded.py:625-630``).  Here selectors clamp to
  ``min(nvalid, 8192, K*s)`` (merge-consistent).  Rows with ``nvalid > 8192``
  and ``K*s > 8192`` (P256 decode, prefill ``K`` up to 12287) select
  differently from MSA.
* D8/D9 Upstream tests compare score multisets only, so D1-D4 are untested
  there; FP8 products have at most 8 significant bits, so exact block ties are
  common on real data.
* Scores: MSA/vLLM scorers do not canonicalize ``-0.0`` explicitly (the vLLM
  decode scorer cannot produce it, ``+0``-initialized accumulators); FI
  scorers canonicalize, so a published score word may differ from MSA's only in
  the sign of a zero.  Merged ids cannot differ.
* Decode plane coverage follows the vLLM decode scorer exactly: columns
  ``c < min(K, 4*ceil(ceil(seq_len/P)/4))`` are written (columns at or beyond
  ``ceil(seq_len/P)`` hold ``-inf``); other columns are untouched.
* Prefill coverage (``initialize=False``): every in-segment row gets the
  columns ``< min(K, C_t)`` written with ``C_t = LocalPrefixCount(ceil(U_t/128),
  s, o)`` (compact / P128) or ``ceil(U_t/128)`` gid columns (P256 non-compact,
  off-parity columns ``-inf``), ``U_t = min(kv_len, q_pos + 1)``.  MSA writes the
  same columns for P128 and pre-fills ``-inf`` for P256 non-compact and P256 TP4
  compact; FI does not pre-fill when ``initialize=False``.  Graph-padding rows
  (in no segment) are neither written nor read.

Golden vectors: :func:`build_golden` returns the content of
``tests/test_helpers/cake_msa_icp_golden.json`` (every array is
``{"dtype", "shape", "zlib_b64"}``: little-endian raw bytes, zlib-compressed,
base64; ``f32`` arrays carry raw IEEE bits so NaN payloads and ``-0`` survive).
Inputs are produced by a self-contained SplitMix64 generator, so the content is
reproducible on any platform; compare content with :func:`golden_equal`, not
file bytes (zlib builds differ).

Command line (run on a CPU compute allocation, never on a login node)::

    python -m tests.test_helpers.cake_msa_icp_inputs self-test
    python -m tests.test_helpers.cake_msa_icp_inputs write-golden [PATH]
    python -m tests.test_helpers.cake_msa_icp_inputs check-golden [PATH]
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

# ----------------------------------------------------------------------------- constants

HEAD_DIM = 128
NUM_HEADS = 4  # MiniMax-M3 index heads; the v1 FI scope (DESIGN OQ-22)
TOPK = 16
BLOCK = 128  # logical ranking block (B128)
MAX_BLOCKS = 8192  # merge clamp (vLLM MAX_K_TILES); also the decode column cap
MAX_SELECT_COLUMNS = 12288  # select requires 0 < K < 12288 (MSA cu:182,233)
ROW_WORDS = 2 * TOPK + 1  # legacy / carrier row: 16 score bits, 16 ids, ready / marker
TAGGED_WORDS = 2 * TOPK  # tagged row: 32 x u64 (gen << 32 | payload)
NUM_SLOT_SETS = 3
NEG_INF_BITS = 0xFF800000
POS_INF_BITS = 0x7F800000
FLT_MAX_BITS = 0x7F7FFFFF
NEG_ZERO_BITS = 0x80000000
QNAN_BITS = 0x7FC00000
PAD_ID = -1
PAD_PAGE = -2147480000  # block-table tail poison: any read is a bug
E4M3_NAN = 0x7F
E4M3_MAX = 448.0
DEFAULT_SPIN_LIMIT = 1 << 26
PLAN_WORKSPACE_BYTES = 32 << 20
SUPPORTED_TP = (2, 4)
SUPPORTED_PAGE_SIZES = (128, 256)
DECODE_PAGES_PER_STAGE = 4  # vLLM decode scorer writes columns in groups of 4
GOLDEN_FILENAME = "cake_msa_icp_golden.json"
GOLDEN_VERSION = 1
_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF

# ----------------------------------------------------------------------------- bit helpers


def f32_bits(x: float) -> int:
    """IEEE-754 binary32 bits of ``x`` (rounded to fp32)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & _MASK32))[0]


def is_nan_bits(bits: int) -> bool:
    bits &= _MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def canon_bits(bits: int) -> int:
    """NaN -> -inf, -0 -> +0, else unchanged."""
    bits &= _MASK32
    if is_nan_bits(bits):
        return NEG_INF_BITS
    if bits == NEG_ZERO_BITS:
        return 0
    return bits


def ordk_bits(bits: int) -> int:
    """Canonical 32-bit order key of a raw f32 bit pattern (larger is better)."""
    b = canon_bits(bits)
    return (~b & _MASK32) if (b >> 31) else (b | 0x80000000)


def ordk(x: float) -> int:
    return ordk_bits(f32_bits(x))


ORD_PINF = ordk_bits(POS_INF_BITS)  # 0xFF800000: forced key, equal to a real +inf
ORD_NINF = ordk_bits(NEG_INF_BITS)  # 0x007FFFFF: lowest key of a real candidate
ABSENT = 0  # below ORD_NINF; never selected


def outranks(ka: int, ia: int, kb: int, ib: int) -> bool:
    """Canonical order: key descending, then id ascending."""
    return ka > kb or (ka == kb and ia < ib)


def to_i32(u: int) -> int:
    u &= _MASK32
    return u - (1 << 32) if u >> 31 else u


def to_u32(i: int) -> int:
    return i & _MASK32


def to_i64(u: int) -> int:
    u &= _MASK64
    return u - (1 << 64) if u >> 63 else u


def next_generation(g: int) -> int:
    """Publish / merge generation step: ``g + 1`` modulo 2**32, skipping 0."""
    g = (g + 1) & _MASK32
    return 1 if g == 0 else g


def local_prefix_count(g: int, s: int, o: int) -> int:
    """Number of local columns ``c`` with ``c*s + o < g`` (MSA ``LocalPrefixCount``)."""
    return 0 if g <= o else (g - 1 - o) // s + 1


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def bits_tensor(x: torch.Tensor) -> torch.Tensor:
    """Raw f32 bits of a float32 tensor as int64 in ``[0, 2**32)``."""
    return x.to(torch.float32).contiguous().view(torch.int32).to(torch.int64) & _MASK32


def _as_i32_bits(bits: torch.Tensor) -> torch.Tensor:
    """int32 view of a bits tensor given as int32 or int64 ``[0, 2**32)``."""
    if bits.dtype == torch.int32:
        return bits
    b = bits.to(torch.int64) & _MASK32
    return torch.where(b >= (1 << 31), b - (1 << 32), b).to(torch.int32)


def ordk_tensor(bits: torch.Tensor) -> torch.Tensor:
    """Vectorized :func:`ordk_bits`: int32 / int64 bits -> int64 keys."""
    b = bits.to(torch.int64) & _MASK32
    nan = ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0)
    b = torch.where(nan, torch.full_like(b, NEG_INF_BITS), b)
    b = torch.where(b == NEG_ZERO_BITS, torch.zeros_like(b), b)
    neg = (b >> 31) == 1
    return torch.where(neg, (~b) & _MASK32, b | 0x80000000)


# ----------------------------------------------------------------------------- layouts


@dataclass(frozen=True)
class IcpLayout:
    """Index-K sharding of one (page size, world size) pair."""

    page_size: int
    tp: int

    def __post_init__(self) -> None:
        if self.page_size not in SUPPORTED_PAGE_SIZES or self.tp not in SUPPORTED_TP:
            raise ValueError(f"unsupported layout P{self.page_size} TP{self.tp}")

    @property
    def name(self) -> str:
        return f"p{self.page_size}_tp{self.tp}"

    @property
    def rows(self) -> int:
        """Local key rows per page (``R = P / W``)."""
        return self.page_size // self.tp

    @property
    def interleaved(self) -> bool:
        return self.page_size == 128

    @property
    def index_scale(self) -> int:
        return 2 if self.page_size == 256 else 1

    def index_offset(self, rank: int) -> int:
        return rank * self.index_scale // self.tp

    def global_row(self, column: int, row: int, rank: int) -> int:
        """Global token position of local key row ``row`` of local page ``column``."""
        if self.interleaved:
            return column * 128 + row * self.tp + rank
        return column * 256 + rank * self.rows + row

    def global_rows(self, rank: int, columns: int) -> torch.Tensor:
        """int64 ``[columns, R]`` global positions of every local key row."""
        c = torch.arange(columns, dtype=torch.int64)[:, None]
        j = torch.arange(self.rows, dtype=torch.int64)[None, :]
        if self.interleaved:
            return c * 128 + j * self.tp + rank
        return c * 256 + rank * self.rows + j

    def page_rows(self, rank: int) -> torch.Tensor:
        """In-page offsets (0..P-1) of the rank's local rows, in local-row order."""
        return self.global_rows(rank, 1)[0]

    def gid(self, column: int, rank: int) -> int:
        """Global B128 id of local (compact) column ``column``."""
        return column * self.index_scale + self.index_offset(rank)

    def holders(self, block: int) -> tuple[int, ...]:
        """Ranks holding key rows of global block ``block``."""
        if self.interleaved:
            return tuple(range(self.tp))
        if self.tp == 2:
            return (block % 2,)
        return (2 * (block % 2), 2 * (block % 2) + 1)

    def slot_owner_local(self, offset: int) -> tuple[int, int]:
        """vLLM cache-write mapping (``index_prefill.py:38-45``): in-page offset ->
        ``(owner rank, local row)``."""
        if self.page_size == 256:
            return offset // self.rows, offset % self.rows
        return offset % self.tp, offset // self.tp

    def shard(self, cache: torch.Tensor, rank: int) -> torch.Tensor:
        """Rank-local shard ``[pages, R, 128]`` of full pages ``[pages, P, 128]``."""
        return cache[:, self.page_rows(rank)].contiguous()


LAYOUTS = tuple(IcpLayout(p, w) for p in SUPPORTED_PAGE_SIZES for w in SUPPORTED_TP)


def layout_by_name(name: str) -> IcpLayout:
    for lay in LAYOUTS:
        if lay.name == name:
            return lay
    raise KeyError(name)


# ----------------------------------------------------------------------------- FP8 helpers

_E4M3_TABLE: torch.Tensor | None = None


def e4m3_table() -> torch.Tensor:
    """float64 value of every E4M3FN byte (NaN for 0x7F / 0xFF)."""
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        _E4M3_TABLE = (
            torch.arange(256, dtype=torch.int32)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float64)
        )
    return _E4M3_TABLE


def e4m3_to_f64(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise TypeError(f"expected uint8 / float8_e4m3fn bytes, got {t.dtype}")
    return e4m3_table()[t.cpu().to(torch.int64)]


def f64_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Exact E4M3FN encoding (raises if a value is not representable)."""
    v = values.to(torch.float32)
    b = v.to(torch.float8_e4m3fn).view(torch.uint8)
    back = e4m3_to_f64(b)
    if not torch.equal(back, v.to(torch.float64)):
        raise ValueError("value not exactly representable in E4M3FN")
    return b


def as_fp8(t: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn view of uint8 bytes."""
    return t.view(torch.float8_e4m3fn) if t.dtype == torch.uint8 else t


# ----------------------------------------------------------------------------- deterministic RNG


class SplitMix64:
    """Platform-independent generator for reproducible golden inputs."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return (self.next_u64() * n) >> 64

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in ``[lo, hi]``."""
        return lo + self.below(hi - lo + 1)

    def ints(self, count: int, lo: int, hi: int) -> list[int]:
        return [self.randint(lo, hi) for _ in range(count)]

    def permutation(self, n: int) -> list[int]:
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            p[i], p[j] = p[j], p[i]
        return p


def _narrow_value(rng: SplitMix64) -> float:
    """Nonzero E4M3 value with exponent in [-2, 1]: products are multiples of
    2**-10 and |q.k| < 2**11, so every fp32 summation order is exact."""
    sign = -1.0 if rng.below(2) else 1.0
    return sign * (1.0 + rng.below(8) / 8.0) * (2.0 ** rng.randint(-2, 1))


def fp8_values(rng: SplitMix64, count: int, kind: str) -> list[float]:
    """``count`` exactly representable E4M3 values of fixture ``kind``.

    ``int4``: integers in [-4, 4]; ``int2``: integers in [-2, 2] (MSA's
    ``randint(-2, 3)``); ``narrow``: see :func:`_narrow_value`.  All three keep
    every fp32 summation of a 128-term dot product exact.
    """
    if kind == "int4":
        return [float(v) for v in rng.ints(count, -4, 4)]
    if kind == "int2":
        return [float(v) for v in rng.ints(count, -2, 2)]
    if kind == "narrow":
        return [_narrow_value(rng) for _ in range(count)]
    raise ValueError(f"unknown fp8 kind {kind!r}")


def fp8_tensor(rng: SplitMix64, shape: Sequence[int], kind: str) -> torch.Tensor:
    n = math.prod(shape)
    return f64_to_e4m3(
        torch.tensor(fp8_values(rng, n, kind), dtype=torch.float64)
    ).reshape(tuple(shape))


# ----------------------------------------------------------------------------- scorer oracle


def _nan_to_neg_inf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x)


def block_max(dots: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """NaN-ignoring max over the last dim of the visible entries, ``-inf`` when
    none is visible, ``-0 -> +0``; float64 in, float32 out."""
    masked = torch.where(
        visible, _nan_to_neg_inf(dots), torch.full_like(dots, float("-inf"))
    )
    m = masked.amax(dim=-1) + 0.0  # -0.0 + 0.0 == +0.0
    return m.to(torch.float32)


def local_column_scores(
    q: torch.Tensor,
    pages: torch.Tensor,
    layout: IcpLayout,
    rank: int,
    q_pos: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    """Per-rank partial block maxima.

    ``q``: E4M3 bytes ``[N, H, 128]``; ``pages``: E4M3 bytes ``[n, P, 128]``
    (full logical pages ``0..n-1`` of one request); ``q_pos`` / ``kv_len``:
    int ``[N]``.  Returns float32 ``[N, H, n]``: column ``c`` = max over the
    rank's local rows of page ``c`` visible to the query (``-inf`` if none).
    """
    n = pages.shape[0]
    if n == 0:
        return torch.empty((q.shape[0], q.shape[1], 0), dtype=torch.float32)
    keys = e4m3_to_f64(layout.shard(pages, rank))  # [n, R, 128]
    qq = e4m3_to_f64(q)  # [N, H, 128]
    dots = torch.einsum("thd,crd->thcr", qq, keys)  # [N, H, n, R]
    g = layout.global_rows(rank, n)  # [n, R]
    qp = torch.as_tensor(q_pos, dtype=torch.int64)[:, None, None, None]
    kl = torch.as_tensor(kv_len, dtype=torch.int64)[:, None, None, None]
    visible = (g[None, None] <= qp) & (g[None, None] < kl)
    return block_max(dots, visible.expand_as(dots))


def global_block_scores(
    q: torch.Tensor,
    pages: torch.Tensor,
    q_pos: torch.Tensor,
    kv_len: torch.Tensor,
    nblocks: int,
) -> torch.Tensor:
    """Unsharded B128 maxima ``[N, H, nblocks]`` from full logical pages
    (independent of the layout formulas)."""
    if nblocks == 0:
        return torch.empty((q.shape[0], q.shape[1], 0), dtype=torch.float32)
    rows = e4m3_to_f64(pages).reshape(-1, HEAD_DIM)  # [n*P, 128] in token order
    rows = rows[: nblocks * BLOCK]
    if rows.shape[0] < nblocks * BLOCK:
        raise ValueError("pages do not cover the requested blocks")
    dots = torch.einsum("thd,kd->thk", e4m3_to_f64(q), rows).reshape(
        q.shape[0], q.shape[1], nblocks, BLOCK
    )
    g = torch.arange(nblocks * BLOCK, dtype=torch.int64).reshape(nblocks, BLOCK)
    qp = torch.as_tensor(q_pos, dtype=torch.int64)[:, None, None, None]
    kl = torch.as_tensor(kv_len, dtype=torch.int64)[:, None, None, None]
    visible = (g[None, None] <= qp) & (g[None, None] < kl)
    return block_max(dots, visible.expand_as(dots))


# ----------------------------------------------------------------------------- select oracle


def oracle_select_row(
    row_bits: Sequence[int],
    nvalid: int,
    force_begin: int,
    force_end: int,
    index_scale: int = 1,
    index_offset: int = 0,
) -> list[int]:
    """Literal transcription of DESIGN section 5 ``oracle_select_row``.

    ``row_bits``: the K raw f32 bit patterns of one ``(t, h)`` row.  Returns
    the 33 words of the carrier / publish row as signed int32 values (word 32
    = 1).
    """
    k = len(row_bits)
    s, o = index_scale, index_offset
    ng = min(max(nvalid, 0), MAX_BLOCKS, k * s)
    L = local_prefix_count(ng, s, o)
    fe0 = max(ng - force_end, 0)

    def forced(c: int) -> bool:
        return (c * s + o) < force_begin or (c * s + o) >= fe0

    def key(c: int) -> int:
        return ORD_PINF if forced(c) else ordk_bits(row_bits[c])

    sel = sorted(sorted(range(L), key=lambda c: (-key(c), c))[:TOPK])
    scores = [row_bits[c] & _MASK32 for c in sel] + [NEG_INF_BITS] * (TOPK - len(sel))
    ids = [c * s + o for c in sel] + [PAD_ID] * (TOPK - len(sel))
    return [to_i32(w) for w in scores] + ids + [1]


def select_rows(
    score_bits: torch.Tensor,
    nvalid: torch.Tensor,
    force_begin: int,
    force_end: int,
    index_scale: int = 1,
    index_offset: int = 0,
    *,
    chunk_rows: int = 512,
) -> torch.Tensor:
    """Vectorized select: ``score_bits`` int32/int64 ``[N, K]`` raw f32 bits,
    ``nvalid`` int ``[N]`` (the row's global B128 count) -> int32 ``[N, 33]``."""
    if score_bits.dim() != 2:
        raise ValueError("score_bits must be [N, K]")
    n_rows, k = score_bits.shape
    s, o = index_scale, index_offset
    out = torch.empty((n_rows, ROW_WORDS), dtype=torch.int32)
    nv = torch.as_tensor(nvalid, dtype=torch.int64).reshape(-1)
    col = torch.arange(k, dtype=torch.int64)
    gid = col * s + o
    for lo in range(0, n_rows, chunk_rows):
        hi = min(n_rows, lo + chunk_rows)
        bits = score_bits[lo:hi].to(torch.int64) & _MASK32
        ng = nv[lo:hi].clamp(min=0).clamp(max=MAX_BLOCKS).clamp(max=k * s)
        L = torch.where(ng <= o, torch.zeros_like(ng), (ng - 1 - o) // s + 1)
        fe0 = (ng - force_end).clamp(min=0)
        valid = col[None, :] < L[:, None]
        forced = (gid[None, :] < force_begin) | (gid[None, :] >= fe0[:, None])
        key = torch.where(forced, torch.full_like(bits, ORD_PINF), ordk_tensor(bits))
        comp = (key << 31) | (0x7FFFFFFF - col[None, :])
        comp = torch.where(valid, comp, torch.full_like(comp, -1))
        top = min(TOPK, k)
        vals, idx = torch.topk(comp, top, dim=1)
        keep = vals >= 0
        c_sel = torch.where(keep, idx, torch.full_like(idx, 1 << 40))
        c_sorted, _ = torch.sort(c_sel, dim=1)
        have = c_sorted < (1 << 40)
        c_safe = torch.where(have, c_sorted, torch.zeros_like(c_sorted))
        sc = torch.gather(bits, 1, c_safe)
        sc = torch.where(have, sc, torch.full_like(sc, NEG_INF_BITS))
        ids = torch.where(have, c_safe * s + o, torch.full_like(c_safe, PAD_ID))
        if top < TOPK:
            pad = TOPK - top
            sc = torch.cat(
                [sc, torch.full((hi - lo, pad), NEG_INF_BITS, dtype=torch.int64)], 1
            )
            ids = torch.cat(
                [ids, torch.full((hi - lo, pad), PAD_ID, dtype=torch.int64)], 1
            )
        out[lo:hi, :TOPK] = _as_i32_bits(sc)
        out[lo:hi, TOPK : 2 * TOPK] = ids.to(torch.int32)
        out[lo:hi, 2 * TOPK] = 1
    return out


# ----------------------------------------------------------------------------- merge oracle


def oracle_merge_row(
    cands: Iterable[tuple[int, int]],
    nvalid: int,
    init_blocks: int,
    local_blocks: int,
    prefill: bool,
) -> list[int]:
    """Literal transcription of DESIGN section 5 ``oracle_merge_row``.

    ``cands``: ``(score_bits, id)`` pairs of all W sources.  Returns 16 ids.
    """
    n = min(nvalid, MAX_BLOCKS)
    best: dict[int, int] = {}
    for sb, i in cands:
        if not (0 <= i < n):
            continue
        kk = (
            ORD_PINF
            if (i < init_blocks or i >= max(0, n - local_blocks))
            else ordk_bits(sb)
        )
        best[i] = max(best.get(i, 0), kk)
    out = sorted(sorted(best, key=lambda i: (-best[i], i))[:TOPK]) + [PAD_ID] * TOPK
    out = out[:TOPK]
    if prefill and n == 0:
        out[0] = 0
    return out


def oracle_global(
    g_bits: Sequence[int],
    nvalid: int,
    init_blocks: int,
    local_blocks: int,
    prefill: bool,
) -> list[int]:
    """Unsharded reference: merge over every block ``b < len(g_bits)`` with its
    global score (DESIGN section 5 ``oracle_global``)."""
    return oracle_merge_row(
        [(g_bits[b], b) for b in range(len(g_bits))],
        nvalid,
        init_blocks,
        local_blocks,
        prefill,
    )


def merge_rows(
    cand_bits: torch.Tensor,
    cand_ids: torch.Tensor,
    nvalid: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    prefill: bool,
    *,
    chunk_rows: int = 2048,
) -> torch.Tensor:
    """Vectorized merge: ``cand_bits`` / ``cand_ids`` ``[N, M]`` (M = W*16),
    ``nvalid`` ``[N]`` -> int32 ``[N, 16]``."""
    n_rows, m = cand_ids.shape
    out = torch.empty((n_rows, TOPK), dtype=torch.int32)
    nv_all = torch.as_tensor(nvalid, dtype=torch.int64).reshape(-1)
    pos = torch.arange(m, dtype=torch.int64)
    for lo in range(0, n_rows, chunk_rows):
        hi = min(n_rows, lo + chunk_rows)
        ids = cand_ids[lo:hi].to(torch.int64)
        bits = cand_bits[lo:hi].to(torch.int64) & _MASK32
        n = nv_all[lo:hi].clamp(max=MAX_BLOCKS)[:, None]
        valid = (ids >= 0) & (ids < n)
        forced = (ids < init_blocks) | (ids >= (n - local_blocks).clamp(min=0))
        key = torch.where(forced, torch.full_like(bits, ORD_PINF), ordk_tensor(bits))
        comp = (key << 31) | (0x7FFFFFFF - ids.clamp(min=0, max=0x7FFFFFFF))
        comp = torch.where(valid, comp, torch.full_like(comp, -1))
        same = (
            (ids[:, :, None] == ids[:, None, :]) & valid[:, :, None] & valid[:, None, :]
        )
        beats = (comp[:, None, :] > comp[:, :, None]) | (
            (comp[:, None, :] == comp[:, :, None])
            & (pos[None, None, :] < pos[None, :, None])
        )
        dominated = (same & beats).any(dim=2)
        comp = torch.where(dominated, torch.full_like(comp, -1), comp)
        vals, _ = torch.topk(comp, min(TOPK, m), dim=1)
        sel = torch.where(
            vals >= 0, 0x7FFFFFFF - (vals & 0x7FFFFFFF), torch.full_like(vals, 1 << 40)
        )
        sel, _ = torch.sort(sel, dim=1)
        res = torch.where(sel < (1 << 40), sel, torch.full_like(sel, PAD_ID))
        if res.shape[1] < TOPK:
            res = torch.cat(
                [
                    res,
                    torch.full(
                        (hi - lo, TOPK - res.shape[1]), PAD_ID, dtype=torch.int64
                    ),
                ],
                1,
            )
        if prefill:
            zero = n[:, 0] == 0
            res[:, 0] = torch.where(zero, torch.zeros_like(res[:, 0]), res[:, 0])
        out[lo:hi] = res.to(torch.int32)
    return out


def merge_carrier_rows(
    received: torch.Tensor,
    nvalid: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    prefill: bool,
) -> torch.Tensor:
    """Merge received rows ``[W, hpr, T, >=32]`` (legacy / carrier words) -> ``[T, hpr, 16]``."""
    w, hpr, t, _ = received.shape
    rows = received[..., : 2 * TOPK].permute(2, 1, 0, 3)  # [T, hpr, W, 32]
    bits = rows[..., :TOPK].reshape(t * hpr, w * TOPK)
    ids = rows[..., TOPK : 2 * TOPK].reshape(t * hpr, w * TOPK)
    nv = torch.as_tensor(nvalid, dtype=torch.int64).reshape(-1).repeat_interleave(hpr)
    return merge_rows(bits, ids, nv, init_blocks, local_blocks, prefill).reshape(
        t, hpr, TOPK
    )


# ----------------------------------------------------------------------------- reference transcriptions (cross-checks)


def vllm_merge_transcription(
    cands: Sequence[tuple[int, int]],
    world: int,
    nvalid: int,
    init: int,
    local: int,
    prefill: bool,
) -> list[int]:
    """Literal Python transcription of vLLM ``_index_k_merge_candidates_kernel``
    (``index_decode_score_sharded.py:534-695`` at ``a76900d6``), one row; used
    only to cross-check :func:`oracle_merge_row`."""
    ids = [i for _, i in cands]
    scores = [bits_f32(b) for b, _ in cands]
    m = len(cands)
    nv = min(nvalid, MAX_BLOCKS)
    valid = [(i >= 0) and (i < nv) for i in ids]
    scores = [s if s == s else float("-inf") for s in scores]
    if world != 2:
        new_scores, unique = [], []
        for a in range(m):
            eq = [b for b in range(m) if ids[a] == ids[b] and valid[a] and valid[b]]
            new_scores.append(max((scores[b] for b in eq), default=float("-inf")))
            first = min(eq, default=world * TOPK)
            unique.append(valid[a] and a == first)
        scores = new_scores
    forced = [(i < init) or (i >= max(0, nv - local)) for i in ids]
    scores = [float("inf") if f else s for s, f in zip(scores, forced, strict=True)]
    ordered = []
    for s in scores:
        b = f32_bits(0.0 if s == 0 else s)
        ordered.append((~b & _MASK32) if (b & 0x80000000) else (b ^ 0x80000000))
    if world == 2:
        unique = []
        new_ordered = []
        for a in range(m):
            peers = [b for b in range(m) if ids[b] == ids[a]]
            new_ordered.append(max(ordered[b] for b in peers))
            unique.append(valid[a] and a == min(peers))
        ordered = new_ordered
    keys = [
        ((ordered[a] << 32) | ((0xFFFFFFFF - (ids[a] & _MASK32)) & _MASK32))
        if unique[a]
        else 0
        for a in range(m)
    ]
    keys.sort(reverse=True)
    sel_keys = keys[:TOPK]
    selected = [
        to_i32(0xFFFFFFFF - (k & _MASK32)) if k != 0 else 0x7FFFFFFF for k in sel_keys
    ]
    selected.sort()
    out = [-1 if v == 0x7FFFFFFF else v for v in selected]
    if prefill and nv == 0:
        out[0] = 0
    return out


def vllm_decode_plane_transcription(
    case: "DecodeCase", tp: int, rank: int
) -> torch.Tensor:
    """Transcription of the vLLM CuTe decode scorer
    (``index_decode_score_sharded.py:120-412``): 4-page stages ``super_id <
    ceil(num_blocks / 4)``, the last page reloaded past ``num_blocks``,
    ``k_pos = tile*P + rank + row*TP`` (P128) or ``tile*P + rank*ROWS + row``
    (P256), mask ``q_pos >= k_pos``, ``fmax`` from ``-inf``.  The shard rows
    follow the vLLM cache writer (``index_prefill.py:38-45``).  float32
    ``[T, H, columns]``, NaN where nothing is written.  Cross-check only."""
    page, pages_per_stage = case.page_size, DECODE_PAGES_PER_STAGE
    rows = page // tp
    dql = case.decode_query_len
    out = torch.full(
        (case.tokens, case.num_heads, case.columns), float("nan"), dtype=torch.float32
    )
    for b in range(case.batch):
        seqlen = case.seq_lens[b]
        num_blocks = _cdiv(seqlen, page)
        q = e4m3_to_f64(case.q[b * dql : (b + 1) * dql])
        for super_id in range(_cdiv(num_blocks, pages_per_stage)):
            for p in range(pages_per_stage):
                tile = super_id * pages_per_stage + p
                full = e4m3_to_f64(
                    case.cache[case.block_table[b][min(tile, num_blocks - 1)]]
                )
                if page == 128:
                    shard = full[rank::tp]
                    k_pos = torch.tensor(
                        [tile * page + rank + i * tp for i in range(rows)]
                    )
                else:
                    shard = full[rank * rows : (rank + 1) * rows]
                    k_pos = torch.tensor(
                        [tile * page + rank * rows + i for i in range(rows)]
                    )
                dots = torch.einsum("jhd,rd->jhr", q, shard) + 0.0
                for j in range(dql):
                    q_pos = seqlen - dql + j
                    vals = torch.where(
                        k_pos[None, :] <= q_pos,
                        dots[j],
                        torch.full_like(dots[j], float("-inf")),
                    )
                    vals = torch.where(
                        torch.isnan(vals), torch.full_like(vals, float("-inf")), vals
                    )
                    out[b * dql + j, :, tile] = vals.amax(-1).to(torch.float32)
    return out


def msa_prefill_scores_transcription(
    case: "PrefillCase", window: "PrefillWindow", tp: int, rank: int, columns: int
) -> torch.Tensor:
    """Transcription of the MSA smoke-test reference
    (``tests/smoke/test_index_score_sharded.py:200-262``) for one window:
    non-compact gid columns, ``-inf`` where the rank owns no row of the
    block, causal mask ``key_pos > q_pos``.  Cross-check only."""
    page = case.page_size
    rows = page // tp
    local_rows = (
        torch.arange(rank, page, tp)
        if page == 128
        else torch.arange(rank * rows, (rank + 1) * rows)
    )
    out = torch.full(
        (window.total_q, case.num_heads, columns), float("-inf"), dtype=torch.float32
    )
    for b in range(window.batch):
        if window.q_lens[b] == 0:
            continue
        n = _cdiv(window.kv_lens[b], page)
        table = torch.tensor(case.block_table[b][:n], dtype=torch.int64)
        shard = e4m3_to_f64(case.cache[table][:, local_rows]).reshape(-1, HEAD_DIM)
        key_pos = (torch.arange(n)[:, None] * page + local_rows[None, :]).flatten()
        first = window.start + window.local_begin[b]
        q = e4m3_to_f64(case.q[first : first + window.q_lens[b]])
        dots = torch.einsum("qhd,kd->qhk", q, shard)
        causal = torch.arange(window.q_lens[b]) + window.qo_offsets[b]
        dots = dots.masked_fill(
            key_pos[None, None, :] > causal[:, None, None], float("-inf")
        )
        for block in range(min(columns, _cdiv(window.kv_lens[b], BLOCK))):
            owned = key_pos // BLOCK == block
            if bool(owned.any()):
                lb = window.local_begin[b]
                out[lb : lb + window.q_lens[b], :, block] = (
                    dots[..., owned].amax(-1).to(torch.float32)
                )
    return out


def msa_select_model_row(
    row_bits: Sequence[int],
    nvalid: int,
    force_begin: int,
    force_end: int,
    index_scale: int = 1,
    index_offset: int = 0,
    *,
    publish: bool,
) -> tuple[list[int], bool]:
    """Attribution model of MSA ``sparse_topk_select_{prefill,publish}`` (not a
    bit-exact emulation).  Returns ``(gids, determinate)``.

    Encodes D1 (publish ties -> higher local id when ``L <= 128``), D4 (raw
    NaN order), D5 (forced = ``FLT_MAX``) and D6 (clamp to ``K*s`` only).
    ``determinate`` is False when MSA's choice is not determined: ``L > 128``
    with an exact key tie straddling the cutoff (D1/D2), or with both zero
    signs at the cutoff (D3).
    """
    k = len(row_bits)
    s, o = index_scale, index_offset
    ng = min(max(nvalid, 0), k * s)
    L = local_prefix_count(ng, s, o)
    if L <= TOPK:
        return [c * s + o for c in range(L)] + [PAD_ID] * (TOPK - L), True
    fe0 = ng - force_end if force_end <= ng else 0

    def msa_key(bits: int) -> int:
        b = bits & _MASK32
        if (b & 0x7FFFFFFF) == 0:
            b = 0
        return (~b & _MASK32) if (b & 0x80000000) else (b ^ 0x80000000)

    def key(c: int) -> int:
        g = c * s + o
        return (
            msa_key(FLT_MAX_BITS)
            if (g < force_begin or g >= fe0)
            else msa_key(row_bits[c])
        )

    keys = [key(c) for c in range(L)]
    if publish and L <= 128:
        order = sorted(range(L), key=lambda c: (-keys[c], -c))
    else:
        order = sorted(range(L), key=lambda c: (-keys[c], c))
    sel = order[:TOPK]
    determinate = True
    if L > 128:
        cut = keys[sel[-1]]
        tied = [c for c in range(L) if keys[c] == cut]
        inside = [c for c in sel if keys[c] == cut]
        if len(tied) > len(inside):
            determinate = False
        if cut == msa_key(0) and any(
            (row_bits[c] & _MASK32) == NEG_ZERO_BITS for c in tied
        ):
            determinate = False
    gids = sorted(c * s + o for c in sel)
    return gids, determinate


# ----------------------------------------------------------------------------- publish / comm protocol


def comm_layout(world: int, heads_per_rank: int, capacity: int) -> dict[str, Any]:
    """Caller-owned comm storage of one rank (vLLM ``IndexKShardComm``,
    ``indexer_msa.py:1003-1068``): one int32 buffer with the legacy rows
    ``[3, W, hpr, cap, 33]`` i32 at byte 0 and the tagged rows
    ``[3, W, hpr, cap, 32]`` i64 at ``tagged_offset_bytes``."""
    rows = NUM_SLOT_SETS * world * heads_per_rank * capacity
    legacy_words = rows * ROW_WORDS
    tagged_offset_bytes = legacy_words * 4
    if tagged_offset_bytes % 8:
        raise ValueError("tagged region would not be 8-byte aligned")
    return {
        "rows": rows,
        "legacy_words": legacy_words,
        "storage_words": legacy_words + rows * 2 * TAGGED_WORDS,
        "legacy_offset_bytes": 0,
        "tagged_offset_bytes": tagged_offset_bytes,
        "legacy_shape": (NUM_SLOT_SETS, world, heads_per_rank, capacity, ROW_WORDS),
        "tagged_shape": (NUM_SLOT_SETS, world, heads_per_rank, capacity, TAGGED_WORDS),
        "generations_shape": (NUM_SLOT_SETS, world * heads_per_rank, capacity),
        "expected_shape": (NUM_SLOT_SETS, heads_per_rank, capacity),
    }


def publish_row_index(
    slot: int,
    world: int,
    rank: int,
    heads_per_rank: int,
    head: int,
    capacity: int,
    token: int,
) -> int:
    """Row of ``(token, head)`` from ``rank`` in the owner's slot set ``slot``."""
    return (
        (slot * world + rank) * heads_per_rank + head % heads_per_rank
    ) * capacity + token


def publish_owner(head: int, heads_per_rank: int) -> int:
    return head // heads_per_rank


def generation_index(
    slot: int, world: int, heads_per_rank: int, head: int, capacity: int, token: int
) -> int:
    """Flat index of the publisher counter ``generations[slot, head, token]``."""
    return (slot * world * heads_per_rank + head) * capacity + token


def tagged_words(generation: int, words32: Sequence[int]) -> list[int]:
    """32 u64 tagged words (as unsigned ints) of one publish row."""
    return [((generation & _MASK32) << 32) | (w & _MASK32) for w in words32[: 2 * TOPK]]


def slot_set(ordinal: int, num_layers: int) -> int:
    """vLLM slot rotation (``indexer_msa.py:1082-1089``)."""
    n = num_layers
    if n % 2 == 0:
        return ordinal % 2
    if n % 3 == 0:
        return ordinal % 3
    return 2 if ordinal == n - 1 else ordinal % 2


class ProtocolTimeout(RuntimeError):
    """The kernel would spin until ``spin_limit`` (``errors += 1``, trap)."""


class CommEmulator:
    """W-rank emulation of the decode publish / merge protocol on host memory.

    Storage, generations and expected counters start zeroed (invariant I-INIT)
    or at ``generation_start`` on every rank (wrap tests).  ``publish`` and
    ``merge`` raise :class:`ProtocolTimeout` where the kernel would spin to its
    limit.
    """

    def __init__(
        self, world: int, heads_per_rank: int, capacity: int, generation_start: int = 0
    ):
        self.world, self.hpr, self.capacity = world, heads_per_rank, capacity
        self.layout = comm_layout(world, heads_per_rank, capacity)
        self.storage = [
            torch.zeros(self.layout["storage_words"], dtype=torch.int32)
            for _ in range(world)
        ]
        g = to_i32(generation_start)
        self.generations = [
            torch.full(self.layout["generations_shape"], g, dtype=torch.int32)
            for _ in range(world)
        ]
        self.expected = [
            torch.full(self.layout["expected_shape"], g, dtype=torch.int32)
            for _ in range(world)
        ]

    def legacy(self, rank: int) -> torch.Tensor:
        lw = self.layout["legacy_words"]
        return self.storage[rank][:lw].view(self.layout["legacy_shape"])

    def tagged(self, rank: int) -> torch.Tensor:
        lw = self.layout["legacy_words"]
        return (
            self.storage[rank][lw:].view(torch.int64).view(self.layout["tagged_shape"])
        )

    def publish(
        self, rank: int, slot: int, rows: torch.Tensor, *, tagged: bool
    ) -> None:
        """Publish select rows ``[T, H, >=32]`` (int32 words) of ``rank``."""
        t_count, heads = rows.shape[0], rows.shape[1]
        if (
            heads != self.world * self.hpr
            or t_count > self.capacity
            or not 0 <= slot < NUM_SLOT_SETS
        ):
            raise ValueError("bad publish geometry")
        for t in range(t_count):
            for h in range(heads):
                owner = publish_owner(h, self.hpr)
                hl = h % self.hpr
                words = [int(v) for v in rows[t, h, : 2 * TOPK]]
                if tagged:
                    gen = self.generations[rank][slot, h, t]
                    g = next_generation(int(gen) & _MASK32)
                    tw = tagged_words(g, words)
                    self.tagged(owner)[slot, rank, hl, t] = torch.tensor(
                        [to_i64(v) for v in tw], dtype=torch.int64
                    )
                    self.generations[rank][slot, h, t] = to_i32(g)
                else:
                    row = self.legacy(owner)[slot, rank, hl, t]
                    if int(row[2 * TOPK]) != 0:
                        raise ProtocolTimeout(
                            f"legacy publish: ready != 0 (owner {owner}, slot {slot}, t {t})"
                        )
                    row[: 2 * TOPK] = torch.tensor(words, dtype=torch.int32)
                    row[2 * TOPK] = 1

    def merge(
        self,
        owner: int,
        slot: int,
        nvalid: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        *,
        tagged: bool,
    ) -> torch.Tensor:
        """Owner merge of slot set ``slot`` for ``T = len(nvalid)`` tokens -> ``[T, hpr, 16]``."""
        nv = torch.as_tensor(nvalid, dtype=torch.int64).reshape(-1)
        t_count = nv.shape[0]
        recv = torch.empty((self.world, self.hpr, t_count, 2 * TOPK), dtype=torch.int64)
        gens = {}
        for hl in range(self.hpr):
            for t in range(t_count):
                if tagged:
                    g = next_generation(
                        int(self.expected[owner][slot, hl, t]) & _MASK32
                    )
                    words = self.tagged(owner)[slot, :, hl, t]  # [W, 32] i64
                    tags = (words >> 32) & _MASK32
                    if not bool((tags == g).all()):
                        raise ProtocolTimeout(
                            f"tagged merge: tag != {g:#x} (owner {owner}, slot {slot}, t {t})"
                        )
                    recv[:, hl, t] = words & _MASK32
                    gens[(hl, t)] = g
                else:
                    rows = self.legacy(owner)[slot, :, hl, t]  # [W, 33]
                    if not bool((rows[:, 2 * TOPK] == 1).all()):
                        raise ProtocolTimeout(
                            f"legacy merge: ready != 1 (owner {owner}, slot {slot}, t {t})"
                        )
                    recv[:, hl, t] = rows[:, : 2 * TOPK].to(torch.int64) & _MASK32
        out = merge_carrier_rows(
            _as_i32_bits(recv).view(recv.shape), nv, init_blocks, local_blocks, False
        )
        for hl in range(self.hpr):
            for t in range(t_count):
                if tagged:
                    self.expected[owner][slot, hl, t] = to_i32(gens[(hl, t)])
                else:
                    self.legacy(owner)[slot, :, hl, t, 2 * TOPK] = 0
        return out

    def storage_sha256(self, rank: int) -> str:
        return hashlib.sha256(self.storage[rank].numpy().tobytes()).hexdigest()


# ----------------------------------------------------------------------------- decode problems


@dataclass
class DecodeCase:
    """One decode step (all requests) in full-page form; layouts derive shards.

    ``q``: E4M3 bytes ``[T, H, 128]`` with ``T = batch * decode_query_len``;
    ``cache``: E4M3 bytes ``[num_physical_pages, P, 128]`` (token order within a
    page); ``block_table``: ``[batch][max_pages]`` physical page ids,
    :data:`PAD_PAGE` beyond ``ceil(seq_len / P)``; ``nvalid``: global B128
    count per token (``q_pos // 128 + 1``, 0 for padding).
    """

    label: str
    page_size: int
    decode_query_len: int
    seq_lens: list[int]
    block_table: list[list[int]]
    columns: int
    q: torch.Tensor
    cache: torch.Tensor
    nvalid: list[int]
    init_blocks: int
    local_blocks: int
    notes: str = ""

    @property
    def batch(self) -> int:
        return len(self.seq_lens)

    @property
    def tokens(self) -> int:
        return self.batch * self.decode_query_len

    @property
    def num_heads(self) -> int:
        return int(self.q.shape[1])

    def request_of(self, t: int) -> int:
        return t // self.decode_query_len

    def q_pos(self, t: int) -> int:
        b = self.request_of(t)
        return self.seq_lens[b] - self.decode_query_len + t % self.decode_query_len

    def pages_of(self, b: int) -> int:
        return _cdiv(self.seq_lens[b], self.page_size)

    def request_pages(self, b: int) -> torch.Tensor:
        n = self.pages_of(b)
        ids = self.block_table[b][:n]
        if any(p < 0 or p >= self.cache.shape[0] for p in ids):
            raise AssertionError("oracle would read a poisoned block-table entry")
        return self.cache[torch.tensor(ids, dtype=torch.int64)] if n else self.cache[:0]

    def written_columns(self, t: int) -> int:
        """Columns of token ``t`` the decode plane scorer writes (vLLM parity)."""
        n = self.pages_of(self.request_of(t))
        return min(
            self.columns, DECODE_PAGES_PER_STAGE * _cdiv(n, DECODE_PAGES_PER_STAGE)
        )


def default_nvalid(q_pos: int) -> int:
    """vLLM ``num_valid_pages`` of a token (``index_prefill.py:33-37``)."""
    return q_pos // BLOCK + 1 if q_pos >= 0 else 0


def _build_pages(
    rng: SplitMix64,
    page_size: int,
    pages_needed: list[int],
    *,
    k_kind: str,
    page_order: str,
    distinct_pages: int | None,
    seq_lens: list[int],
) -> tuple[torch.Tensor, list[list[int]], int]:
    """Physical page pool + per-request page lists.

    ``page_order``: ``identity`` / ``shuffled`` (distinct pages) or
    ``repeat`` (``distinct_pages`` physical pages reused cyclically, which
    produces exact cross-block ties).  ``k_kind``: an :func:`fp8_values`
    kind, ``all_equal`` (one row everywhere), ``palette`` (rows drawn from 3
    ``int4`` rows) or ``sparse8`` (8 random ``int4`` entries per row, the rest
    0; compact golden payloads, many exact ties and exact zeros).  Two extra physical pages hold the
    poison spike rows (wrong-page reads show up as huge scores).  Rows past
    ``seq_len`` in a request's last page hold the spike row too (causal tail
    poison).
    """
    total = sum(pages_needed)
    if page_order == "repeat":
        nphys = max(1, distinct_pages or 1)
    else:
        nphys = total
    pool = nphys + 2
    if k_kind == "all_equal":
        row = fp8_values(rng, HEAD_DIM, "int4")
        vals = torch.tensor(row, dtype=torch.float64).repeat(pool * page_size)
    elif k_kind == "palette":
        palette = [fp8_values(rng, HEAD_DIM, "int4") for _ in range(3)]
        vals = torch.tensor(
            [v for _ in range(pool * page_size) for v in palette[rng.below(3)]],
            dtype=torch.float64,
        )
    elif k_kind == "sparse8":
        vals = torch.zeros(pool * page_size * HEAD_DIM, dtype=torch.float64)
        for row in range(pool * page_size):
            cols: set[int] = set()
            while len(cols) < 8:
                cols.add(rng.below(HEAD_DIM))
            for c in sorted(cols):
                vals[row * HEAD_DIM + c] = float(rng.randint(-4, 4))
    else:
        vals = torch.tensor(
            fp8_values(rng, pool * page_size * HEAD_DIM, k_kind), dtype=torch.float64
        )
    cache = vals.reshape(pool, page_size, HEAD_DIM).clone()
    spike = torch.zeros(HEAD_DIM, dtype=torch.float64)
    spike[0] = E4M3_MAX
    if page_order == "shuffled":
        phys = rng.permutation(nphys)
    else:
        phys = list(range(nphys))
    tables: list[list[int]] = []
    cursor = 0
    tails: list[tuple[int, int]] = []
    for b, n in enumerate(pages_needed):
        if page_order == "repeat":
            ids = [phys[(cursor + i) % nphys] for i in range(n)]
        else:
            ids = phys[cursor : cursor + n]
        cursor += n
        tables.append(ids)
        if n and seq_lens[b] % page_size:
            tails.append((ids[-1], seq_lens[b] % page_size))
    for page, start in tails:
        if page_order != "repeat":
            cache[page, start:] = spike
    cache[nphys:] = spike  # unreferenced pages
    return f64_to_e4m3(cache), tables, nphys


def _decode_q(rng: SplitMix64, tokens: int, heads: int, q_kind: str) -> torch.Tensor:
    """Q fixture: column 0 is +4 and the other columns are ``int2``, so a
    poison spike row (``k[0] = 448``) scores 1792 while any real ``int4`` key
    row scores at most 16 + 127 * 8 = 1032: a missing causal mask or a
    wrong-page read always wins its block.  ``sparse8`` keeps column 0 and 7
    random ``int2`` entries (compact golden payloads); ``narrow`` has no spike
    guarantee."""
    if q_kind == "narrow":
        return fp8_tensor(rng, (tokens, heads, HEAD_DIM), "narrow")
    if q_kind == "sparse8":  # column 0 = 4 plus 7 random int2 entries
        vals = torch.zeros((tokens * heads, HEAD_DIM), dtype=torch.float64)
        for row in range(tokens * heads):
            cols: set[int] = set()
            while len(cols) < 7:
                cols.add(1 + rng.below(HEAD_DIM - 1))
            for c in sorted(cols):
                vals[row, c] = float(rng.randint(-2, 2))
        vals[:, 0] = 4.0
        return f64_to_e4m3(vals.reshape(tokens, heads, HEAD_DIM))
    vals = torch.tensor(
        fp8_values(rng, tokens * heads * HEAD_DIM, "int2"), dtype=torch.float64
    )
    vals = vals.reshape(tokens, heads, HEAD_DIM)
    vals[..., 0] = 4.0
    return f64_to_e4m3(vals)


def make_decode_case(
    label: str,
    *,
    page_size: int,
    seq_lens: Sequence[int],
    decode_query_len: int = 1,
    columns: int | None = None,
    seed: int = 7896,
    k_kind: str = "int4",
    q_kind: str = "int2",
    page_order: str = "shuffled",
    distinct_pages: int | None = None,
    init_blocks: int = 0,
    local_blocks: int = 1,
    nvalid: Sequence[int] | None = None,
    num_heads: int = NUM_HEADS,
    notes: str = "",
) -> DecodeCase:
    """Decode problem generator (``seq_len`` 0 = graph-padding request).

    ``seq_lens`` count visible tokens including the ``decode_query_len`` new
    rows; ``columns`` defaults to ``4 * ceil(max_pages / 4)``.
    """
    rng = SplitMix64(seed)
    seq = [int(s) for s in seq_lens]
    pages_needed = [_cdiv(s, page_size) for s in seq]
    max_pages = max(pages_needed + [1])
    if columns is None:
        columns = DECODE_PAGES_PER_STAGE * _cdiv(max_pages, DECODE_PAGES_PER_STAGE)
    if (
        columns % DECODE_PAGES_PER_STAGE
        or not 0 < columns <= MAX_BLOCKS
        or columns < max_pages
    ):
        raise ValueError(f"{label}: bad columns {columns}")
    cache, tables, _ = _build_pages(
        rng,
        page_size,
        pages_needed,
        k_kind=k_kind,
        page_order=page_order,
        distinct_pages=distinct_pages,
        seq_lens=seq,
    )
    table = [row + [PAD_PAGE] * (max_pages - len(row)) for row in tables]
    tokens = len(seq) * decode_query_len
    q = _decode_q(rng, tokens, num_heads, q_kind)
    case = DecodeCase(
        label=label,
        page_size=page_size,
        decode_query_len=decode_query_len,
        seq_lens=seq,
        block_table=table,
        columns=columns,
        q=q,
        cache=cache,
        nvalid=[],
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        notes=notes,
    )
    case.nvalid = (
        [int(v) for v in nvalid]
        if nvalid is not None
        else [
            default_nvalid(case.q_pos(t)) if seq[case.request_of(t)] > 0 else 0
            for t in range(tokens)
        ]
    )
    if len(case.nvalid) != tokens:
        raise ValueError(f"{label}: nvalid needs {tokens} entries")
    return case


def decode_local_plane(
    case: DecodeCase, layout: IcpLayout, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected decode score plane of ``rank``: float32 ``[T, H, columns]`` and
    int32 ``written[T]``; columns ``>= written[t]`` are untouched (NaN here)."""
    if layout.page_size != case.page_size:
        raise ValueError("layout page size mismatch")
    t_count, heads = case.tokens, case.num_heads
    plane = torch.full(
        (t_count, heads, case.columns), float("nan"), dtype=torch.float32
    )
    written = torch.zeros(t_count, dtype=torch.int32)
    for b in range(case.batch):
        toks = list(range(b * case.decode_query_len, (b + 1) * case.decode_query_len))
        n = case.pages_of(b)
        w = case.written_columns(toks[0])
        for t in toks:
            written[t] = w
        if w == 0:
            continue
        q_pos = torch.tensor([case.q_pos(t) for t in toks], dtype=torch.int64)
        kv = torch.full_like(q_pos, case.seq_lens[b])
        sc = local_column_scores(
            case.q[toks], case.request_pages(b), layout, rank, q_pos, kv
        )
        plane[toks, :, :w] = float("-inf")
        plane[toks, :, :n] = sc
    return plane, written


def decode_local_rows(
    case: DecodeCase, layout: IcpLayout, rank: int, plane: torch.Tensor | None = None
) -> torch.Tensor:
    """Expected local select rows of ``rank``: int32 ``[T, H, 33]`` (the
    ``select_publish`` / fused ``topk16_publish`` payload; word 32 = 1)."""
    if plane is None:
        plane, _ = decode_local_plane(case, layout, rank)
    t_count, heads, cols = plane.shape
    bits = bits_tensor(plane)  # unwritten (NaN) columns lie past every row's L
    nv = torch.tensor(case.nvalid, dtype=torch.int64).repeat_interleave(heads)
    rows = select_rows(
        bits.reshape(t_count * heads, cols),
        nv,
        case.init_blocks,
        case.local_blocks,
        layout.index_scale,
        layout.index_offset(rank),
    )
    return rows.reshape(t_count, heads, ROW_WORDS)


def exchange_decode_rows(
    rows_by_rank: Sequence[torch.Tensor], layout: IcpLayout, owner: int
) -> torch.Tensor:
    """Owner receive view ``[W, hpr, T, 33]`` of the published rows."""
    hpr = rows_by_rank[0].shape[1] // layout.tp
    return torch.stack(
        [r[:, owner * hpr : (owner + 1) * hpr].permute(1, 0, 2) for r in rows_by_rank]
    )


def decode_owner_output(
    case: DecodeCase,
    layout: IcpLayout,
    rows_by_rank: Sequence[torch.Tensor],
    owner: int,
) -> torch.Tensor:
    recv = exchange_decode_rows(rows_by_rank, layout, owner)
    return merge_carrier_rows(
        recv, torch.tensor(case.nvalid), case.init_blocks, case.local_blocks, False
    )


def decode_global_output(case: DecodeCase) -> torch.Tensor:
    """Unsharded reference ``[T, H, 16]`` (independent of the layout)."""
    t_count, heads = case.tokens, case.num_heads
    out = torch.full((t_count, heads, TOPK), PAD_ID, dtype=torch.int32)
    for b in range(case.batch):
        toks = list(range(b * case.decode_query_len, (b + 1) * case.decode_query_len))
        nb = _cdiv(case.seq_lens[b], BLOCK)
        q_pos = torch.tensor([case.q_pos(t) for t in toks], dtype=torch.int64)
        kv = torch.full_like(q_pos, case.seq_lens[b])
        g = global_block_scores(case.q[toks], case.request_pages(b), q_pos, kv, nb)
        for i, t in enumerate(toks):
            for h in range(heads):
                gb = [f32_bits(float(v)) for v in g[i, h]]
                out[t, h] = torch.tensor(
                    oracle_global(
                        gb, case.nvalid[t], case.init_blocks, case.local_blocks, False
                    ),
                    dtype=torch.int32,
                )
    return out


# ----------------------------------------------------------------------------- prefill problems


@dataclass
class PrefillWindow:
    """One bounded Q chunk ``[start, end)`` of the flattened prefill tokens with
    its clipped segments (vLLM ``prepare_prefill_window`` semantics)."""

    start: int
    end: int
    local_begin: list[int]
    q_lens: list[int]
    kv_lens: list[int]
    qo_offsets: list[int]

    @property
    def total_q(self) -> int:
        return self.end - self.start

    @property
    def batch(self) -> int:
        return len(self.q_lens)


@dataclass
class PrefillCase:
    """Prefill / mixed step: requests ``b`` with ``q_lens[b]`` new tokens after
    ``prefixes[b]`` cached tokens, then ``padding_tokens`` graph-padding rows
    (``nvalid = 0``, in no segment).  ``columns`` is the score pitch ``K``
    (compact for P256)."""

    label: str
    page_size: int
    q_lens: list[int]
    prefixes: list[int]
    padding_tokens: int
    chunk: int
    block_table: list[list[int]]
    columns: int
    q: torch.Tensor
    cache: torch.Tensor
    init_blocks: int
    local_blocks: int
    notes: str = ""

    @property
    def batch(self) -> int:
        return len(self.q_lens)

    @property
    def total_q(self) -> int:
        return sum(self.q_lens) + self.padding_tokens

    @property
    def num_heads(self) -> int:
        return int(self.q.shape[1])

    @property
    def query_start(self) -> list[int]:
        out = [0]
        for n in self.q_lens:
            out.append(out[-1] + n)
        return out

    @property
    def seq_lens(self) -> list[int]:
        return [p + q for p, q in zip(self.prefixes, self.q_lens, strict=True)]

    def positions(self) -> list[int]:
        pos: list[int] = []
        for p, n in zip(self.prefixes, self.q_lens, strict=True):
            pos.extend(range(p, p + n))
        return pos + [-1] * self.padding_tokens

    def nvalid(self) -> list[int]:
        return [default_nvalid(p) for p in self.positions()]

    def windows(self) -> list[PrefillWindow]:
        qs = self.query_start
        out = []
        for start in range(0, self.total_q, self.chunk):
            end = min(start + self.chunk, self.total_q)
            lb, ql, kl, qo = [], [], [], []
            for b in range(self.batch):
                begin, finish = qs[b], qs[b + 1]
                local_begin = min(max(begin - start, 0), end - start)
                local_end = min(max(finish - start, 0), end - start)
                q_len = local_end - local_begin
                prefix = self.prefixes[b] + min(max(start - begin, 0), finish - begin)
                lb.append(local_begin)
                ql.append(q_len)
                kl.append(prefix + q_len)
                qo.append(prefix)
            out.append(PrefillWindow(start, end, lb, ql, kl, qo))
        return out

    def request_pages(self, b: int) -> torch.Tensor:
        n = _cdiv(self.seq_lens[b], self.page_size)
        ids = self.block_table[b][:n]
        if any(p < 0 or p >= self.cache.shape[0] for p in ids):
            raise AssertionError("oracle would read a poisoned block-table entry")
        return self.cache[torch.tensor(ids, dtype=torch.int64)] if n else self.cache[:0]


def prefill_columns(max_seq_len: int, page_size: int, compact: bool = True) -> int:
    """vLLM score pitch: ``ceil(L/128)`` (halved, rounded up, for P256 compact),
    rounded up to a multiple of 16 (``indexer_msa.py:430-435``)."""
    blocks = _cdiv(max_seq_len, BLOCK)
    if page_size == 256 and compact:
        blocks = _cdiv(blocks, 2)
    return 16 * _cdiv(max(blocks, 1), 16)


def make_prefill_case(
    label: str,
    *,
    page_size: int,
    q_lens: Sequence[int],
    prefixes: Sequence[int],
    padding_tokens: int = 0,
    chunk: int = 8192,
    columns: int | None = None,
    seed: int = 7896,
    k_kind: str = "int4",
    q_kind: str = "int2",
    page_order: str = "shuffled",
    distinct_pages: int | None = None,
    init_blocks: int = 0,
    local_blocks: int = 1,
    num_heads: int = NUM_HEADS,
    notes: str = "",
) -> PrefillCase:
    rng = SplitMix64(seed)
    seq = [int(p) + int(q) for p, q in zip(prefixes, q_lens, strict=True)]
    pages_needed = [_cdiv(s, page_size) for s in seq]
    max_pages = max(pages_needed + [1])
    cache, tables, _ = _build_pages(
        rng,
        page_size,
        pages_needed,
        k_kind=k_kind,
        page_order=page_order,
        distinct_pages=distinct_pages,
        seq_lens=seq,
    )
    table = [row + [PAD_PAGE] * (max_pages - len(row)) for row in tables]
    total_q = sum(int(v) for v in q_lens) + padding_tokens
    q = _decode_q(rng, total_q, num_heads, q_kind)
    if columns is None:
        columns = prefill_columns(max(seq + [1]), page_size, True)
    if not 0 < columns < MAX_SELECT_COLUMNS:
        raise ValueError(f"{label}: bad columns {columns}")
    return PrefillCase(
        label=label,
        page_size=page_size,
        q_lens=[int(v) for v in q_lens],
        prefixes=[int(v) for v in prefixes],
        padding_tokens=int(padding_tokens),
        chunk=int(chunk),
        block_table=table,
        columns=int(columns),
        q=q,
        cache=cache,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        notes=notes,
    )


def prefill_window_scores(
    case: PrefillCase,
    window: PrefillWindow,
    layout: IcpLayout,
    rank: int,
    *,
    compact: bool = True,
    columns: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected ``max_score`` of one window on ``rank``.

    Returns float32 ``[T_w, H, K]`` as written by ``initialize=True`` (every
    column not written by the scorer is ``-inf``) and int32 ``coverage[T_w]``:
    with ``initialize=False`` exactly the columns ``< coverage[t]`` must hold
    these values (padding rows: 0, untouched).  ``compact`` matters for P256
    only (non-compact columns are gids, off-parity columns ``-inf``).
    """
    k = case.columns if columns is None else columns
    heads = case.num_heads
    out = torch.full((window.total_q, heads, k), float("-inf"), dtype=torch.float32)
    cov = torch.zeros(window.total_q, dtype=torch.int32)
    s, o = layout.index_scale, layout.index_offset(rank)
    for b in range(window.batch):
        if window.q_lens[b] == 0:
            continue
        rows = list(
            range(window.local_begin[b], window.local_begin[b] + window.q_lens[b])
        )
        q_pos = torch.tensor(
            [window.qo_offsets[b] + i for i in range(window.q_lens[b])],
            dtype=torch.int64,
        )
        kv = torch.full_like(q_pos, window.kv_lens[b])
        n_pages = _cdiv(window.kv_lens[b], case.page_size)
        pages = case.request_pages(b)[:n_pages]
        sc = local_column_scores(
            case.q[[window.start + r for r in rows]], pages, layout, rank, q_pos, kv
        )
        for i, r in enumerate(rows):
            u = min(window.kv_lens[b], int(q_pos[i]) + 1)
            nb = _cdiv(u, BLOCK)
            if layout.page_size == 256 and not compact:
                c_t = nb
                for g in range(min(k, c_t)):
                    if g % 2 == o:
                        out[r, :, g] = sc[i, :, g // 2]
            else:
                c_t = local_prefix_count(nb, s, o)
                m = min(k, c_t)
                out[r, :, :m] = sc[i, :, :m]
            cov[r] = min(k, c_t)
    return out, cov


def prefill_local_carrier(
    case: PrefillCase, layout: IcpLayout, rank: int
) -> torch.Tensor:
    """Expected carrier of ``rank`` after every window: int32 ``[H, T, 33]``."""
    heads, t_total = case.num_heads, case.total_q
    nvalid = torch.tensor(case.nvalid(), dtype=torch.int64)
    carrier = torch.empty((heads, t_total, ROW_WORDS), dtype=torch.int32)
    for w in case.windows():
        sc, _ = prefill_window_scores(case, w, layout, rank)
        bits = bits_tensor(sc).reshape(w.total_q * heads, case.columns)
        nv = nvalid[w.start : w.end].repeat_interleave(heads)
        rows = select_rows(
            bits,
            nv,
            case.init_blocks,
            case.local_blocks,
            layout.index_scale,
            layout.index_offset(rank),
        ).reshape(w.total_q, heads, ROW_WORDS)
        carrier[:, w.start : w.end] = rows.permute(1, 0, 2)
    return carrier


def prefill_exchange(
    carriers_by_rank: Sequence[torch.Tensor], layout: IcpLayout, owner: int
) -> torch.Tensor:
    """``all_to_all_single`` on ``send.view(W, hpr, T, 33)``: owner's received ``[W, hpr, T, 33]``."""
    heads = carriers_by_rank[0].shape[0]
    hpr = heads // layout.tp
    return torch.stack([c[owner * hpr : (owner + 1) * hpr] for c in carriers_by_rank])


def prefill_owner_output(
    case: PrefillCase,
    layout: IcpLayout,
    carriers_by_rank: Sequence[torch.Tensor],
    owner: int,
) -> torch.Tensor:
    recv = prefill_exchange(carriers_by_rank, layout, owner)
    return merge_carrier_rows(
        recv, torch.tensor(case.nvalid()), case.init_blocks, case.local_blocks, True
    )


def prefill_global_output(case: PrefillCase) -> torch.Tensor:
    heads, t_total = case.num_heads, case.total_q
    out = torch.full((t_total, heads, TOPK), PAD_ID, dtype=torch.int32)
    pos = case.positions()
    nv = case.nvalid()
    qs = case.query_start
    for t in range(t_total):
        if pos[t] < 0:
            out[t] = torch.tensor(
                oracle_global([], nv[t], case.init_blocks, case.local_blocks, True)
            )
    for b in range(case.batch):
        if case.q_lens[b] == 0:
            continue
        toks = list(range(qs[b], qs[b + 1]))
        q_pos = torch.tensor([pos[t] for t in toks], dtype=torch.int64)
        kv = torch.full_like(q_pos, case.seq_lens[b])
        nb = _cdiv(case.seq_lens[b], BLOCK)
        g = global_block_scores(case.q[toks], case.request_pages(b), q_pos, kv, nb)
        for i, t in enumerate(toks):
            for h in range(heads):
                gb = [f32_bits(float(v)) for v in g[i, h, : _cdiv(pos[t] + 1, BLOCK)]]
                out[t, h] = torch.tensor(
                    oracle_global(gb, nv[t], case.init_blocks, case.local_blocks, True),
                    dtype=torch.int32,
                )
    return out


def plan_storage(num_qo_heads: int, total_q: int, batch: int) -> dict[str, int]:
    """MSA device-plan storage formulas (``api.py:1588-1637``); the int64
    ``packed_work_range`` holds exactly one word per SM of the device."""
    return {
        "work_info_numel_min": (_cdiv(total_q, 128) + batch - 1) * num_qo_heads,
        "workspace_bytes_min": PLAN_WORKSPACE_BYTES,
    }


def plan_tiles(window: PrefillWindow, tile_tokens: int = 32) -> list[dict[str, int]]:
    """Semantic content of the F1 plan: every 32-token Q tile of every segment
    with ``C_end`` = global B128 columns its last row needs.  A valid plan
    covers every ``(tile, column < C_end)`` exactly once (layout-local column
    ends follow from :func:`local_prefix_count`)."""
    tiles = []
    for b in range(window.batch):
        for j in range(_cdiv(window.q_lens[b], tile_tokens)):
            first = j * tile_tokens
            last = min(window.q_lens[b], first + tile_tokens) - 1
            u_last = min(window.kv_lens[b], window.qo_offsets[b] + last + 1)
            tiles.append(
                {
                    "segment": b,
                    "tile": j,
                    "row_begin": window.local_begin[b] + first,
                    "row_end": window.local_begin[b] + last + 1,
                    "c_end": _cdiv(u_last, BLOCK),
                }
            )
    return tiles


def check_plan_coverage(
    window: PrefillWindow, units: Iterable[tuple[int, int, int]], tile_tokens: int = 32
) -> None:
    """Raise unless ``units`` (``(segment, tile, column)`` triples from a
    decoded plan) cover every required unit of :func:`plan_tiles` exactly once."""
    need = {
        (t["segment"], t["tile"], c)
        for t in plan_tiles(window, tile_tokens)
        for c in range(t["c_end"])
    }
    seen: dict[tuple[int, int, int], int] = {}
    for u in units:
        seen[u] = seen.get(u, 0) + 1
    dup = [u for u, n in seen.items() if n > 1]
    missing = need - set(seen)
    extra = set(seen) - need
    if dup or missing or extra:
        raise AssertionError(
            f"plan coverage: {len(missing)} missing, {len(extra)} extra, {len(dup)} duplicated"
        )


# ----------------------------------------------------------------------------- accumulation bounds (E2E gate (b)/(c))

# B_arm = c_arm * max_j sum_i |q_i k_ji| over the block's visible key rows.
# Model (the PTX ISA does not document tensor-core accumulation rounding):
# each MMA instruction adds k exact products to the fp32 accumulator, aligning
# all k + 1 addends to the largest exponent with at least 24 significant bits
# and truncating; the normalized result is truncated once more.  One
# instruction then errs by < (k + 2) * 2**-23 * sum|addends|, and n / k
# instructions over a 128-term dot product by < (n + 2n/k) * 2**-23 * M.  Two
# extra units cover second-order terms.  E4M3 products are exact in fp32
# (and in the f16 MMA inputs vLLM's decode scorer uses).  PROVISIONAL until
# :func:`accumulation_adversarial_cases` passes on each arm on hardware
# (a narrower internal accumulator, as reported for Hopper FP8, fails it).
C_ARM = {
    "tcgen05_f8f6f4_k32": (128 + 2 * 128 // 32 + 2)
    * 2.0**-23,  # FI sm_100a/sm_103a; MSA prefill (CUTLASS)
    "tcgen05_f8f6f4_k64": (128 + 2 * 128 // 64 + 2)
    * 2.0**-23,  # FI sm_107a (inst_k = 64)
    "mma_sync_f16_k16": (128 + 2 * 128 // 16 + 2)
    * 2.0**-23,  # vLLM decode scorer (FP8 -> f16, m16n8k16)
}
C_ARM_STATUS = "provisional: analytic model, hardware adversarial check pending"


def score_error_bound(
    q: torch.Tensor, keys: torch.Tensor, visible: torch.Tensor, c_arm: float
) -> torch.Tensor:
    """``B_arm`` per (query row, block): ``q`` f64 ``[N, 128]``, ``keys`` f64
    ``[N, J, 128]`` (the block's rows per query), ``visible`` bool ``[N, J]``."""
    mag = torch.einsum("nd,njd->nj", q.abs(), keys.abs())
    mag = torch.where(visible, mag, torch.zeros_like(mag))
    return c_arm * mag.amax(dim=-1)


def accumulation_adversarial_cases(
    seed: int = 7896,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """E4M3 ``(q [N, 128], k [N, 128])`` dot products that expose truncating or
    narrow accumulators: one large product (448 * 448) at K position 0, 31, 32,
    63, 64 or 127 plus 127 small same-sign products at scales 2**-12 .. 2**6
    (each below the fp32 truncation unit of the large one for the smaller
    scales), and alternating-sign cancellation rows.  Compare a kernel's fp32
    result ``s`` with the float64 value ``s64``: ``|s - s64| <= c_arm * M``,
    ``M = sum |q_i k_i|``."""
    rng = SplitMix64(seed)
    qs, ks, labels = [], [], []
    small_vals = [2.0**e for e in range(-9, 4)]  # E4M3-representable magnitudes
    for big_pos in (0, 31, 32, 63, 64, 127):
        for qv in small_vals:
            for kv in (2.0**-3, 1.0, 2.0**3):
                q = torch.full((HEAD_DIM,), qv, dtype=torch.float64)
                k = torch.full((HEAD_DIM,), kv, dtype=torch.float64)
                q[big_pos], k[big_pos] = E4M3_MAX, E4M3_MAX
                qs.append(q)
                ks.append(k)
                labels.append(f"big@{big_pos} small={qv * kv:g}")
    for i in range(16):
        q = (
            torch.tensor(fp8_values(rng, HEAD_DIM, "narrow"), dtype=torch.float64)
            * 64.0
        )
        k = torch.tensor(
            [(-1.0) ** j for j in range(HEAD_DIM)], dtype=torch.float64
        ) * q.abs().clamp(max=E4M3_MAX)
        q = q.clamp(-E4M3_MAX, E4M3_MAX)
        qs.append(q)
        ks.append(k.clamp(-E4M3_MAX, E4M3_MAX))
        labels.append(f"cancel#{i}")
    q8 = f64_to_e4m3(torch.stack(qs))
    k8 = f64_to_e4m3(torch.stack(ks))
    return q8, k8, labels


def check_accumulation_bound(
    observed: torch.Tensor, q8: torch.Tensor, k8: torch.Tensor, c_arm: float
) -> dict[str, Any]:
    """Max ``|s - s64| / (c_arm * M)`` over rows; the bound holds iff < = 1."""
    q, k = e4m3_to_f64(q8), e4m3_to_f64(k8)
    s64 = (q * k).sum(-1)
    m = (q * k).abs().sum(-1)
    err = (observed.to(torch.float64) - s64).abs()
    ratio = torch.where(m > 0, err / (c_arm * m), torch.zeros_like(err))
    return {
        "max_ratio": float(ratio.max()),
        "argmax": int(ratio.argmax()),
        "holds": bool((ratio <= 1).all()),
    }


# ----------------------------------------------------------------------------- golden encoding

_DT = {
    "u8": (torch.uint8, 1),
    "i32": (torch.int32, 4),
    "u32": (torch.int32, 4),
    "i64": (torch.int64, 8),
    "f32": (torch.float32, 4),
}


def encode_array(t: torch.Tensor, dtype: str) -> dict[str, Any]:
    """``{"dtype", "shape", "zlib_b64"}``; ``f32`` keeps raw bits, ``u32`` takes
    int64 values in ``[0, 2**32)`` or int32 bit patterns."""
    t = t.detach().cpu().contiguous()
    if dtype == "f32":
        raw = (
            t.to(torch.float32).view(torch.int32)
            if t.dtype == torch.float32
            else _as_i32_bits(t)
        )
    elif dtype in ("i32", "u32"):
        raw = (
            _as_i32_bits(t)
            if t.dtype == torch.int64 and dtype == "u32"
            else t.to(torch.int32)
        )
    elif dtype == "u8":
        raw = (
            t.view(torch.uint8)
            if t.dtype in (torch.uint8, torch.float8_e4m3fn)
            else t.to(torch.uint8)
        )
    elif dtype == "i64":
        raw = t.to(torch.int64)
    else:
        raise ValueError(dtype)
    data = raw.contiguous().numpy().tobytes()
    if sys.byteorder != "little":
        raise RuntimeError("golden encoding assumes a little-endian host")
    return {
        "dtype": dtype,
        "shape": list(t.shape),
        "zlib_b64": base64.b64encode(zlib.compress(data, 9)).decode("ascii"),
    }


def decode_array(obj: dict[str, Any]) -> torch.Tensor:
    """Inverse of :func:`encode_array`: ``f32`` -> float32 (raw bits kept),
    ``u32`` -> int64 in ``[0, 2**32)``, others their natural dtype."""
    tdt, _ = _DT[obj["dtype"]]
    raw = bytearray(zlib.decompress(base64.b64decode(obj["zlib_b64"])))
    t = torch.frombuffer(raw, dtype=tdt).clone() if raw else torch.empty(0, dtype=tdt)
    t = t.reshape(obj["shape"])
    if obj["dtype"] == "u32":
        return t.to(torch.int64) & _MASK32
    return t


def _is_array(obj: Any) -> bool:
    return isinstance(obj, dict) and set(obj) == {"dtype", "shape", "zlib_b64"}


def decode_tree(obj: Any) -> Any:
    """Decode every array of a loaded golden tree."""
    if _is_array(obj):
        return decode_array(obj)
    if isinstance(obj, dict):
        return {k: decode_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_tree(v) for v in obj]
    return obj


def golden_path() -> Path:
    return Path(__file__).with_name(GOLDEN_FILENAME)


def load_golden(
    path: str | Path | None = None, *, decode: bool = True
) -> dict[str, Any]:
    with open(path or golden_path(), encoding="utf-8") as f:
        obj = json.load(f)
    return decode_tree(obj) if decode else obj


def golden_equal(a: Any, b: Any, where: str = "$") -> list[str]:
    """Content comparison of two undecoded golden trees (arrays by decoded
    bytes); returns the differing paths."""
    diffs: list[str] = []
    if _is_array(a) and _is_array(b):
        da, db = decode_array(a), decode_array(b)
        if a["dtype"] != b["dtype"] or da.shape != db.shape:
            return [where]
        if a["dtype"] == "f32":
            da, db = da.view(torch.int32), db.view(torch.int32)
        if not torch.equal(da, db):
            diffs.append(where)
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{where}.{k}")
            else:
                diffs.extend(golden_equal(a[k], b[k], f"{where}.{k}"))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [where]
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diffs.extend(golden_equal(x, y, f"{where}[{i}]"))
        return diffs
    return [] if a == b else [where]


# ----------------------------------------------------------------------------- golden content

_SELECT_MODES = (
    "unique",
    "all_equal",
    "cluster",
    "low_bits",
    "cross_bin",
    "zero",
    "nan",
    "inf",
    "flt_max",
    "neg_inf",
    "int_ties",
)


def select_row_values(rng: SplitMix64, mode: str, count: int) -> list[int]:
    """Raw f32 bits of ``count`` scores of one select row in ``mode`` (MSA
    smoke-test modes plus NaN / inf / FLT_MAX / integer-tie modes)."""
    if mode == "unique":
        vals = [f32_bits(float(i) - count / 2.0) for i in range(count)]
        perm = rng.permutation(count)
        return [vals[p] for p in perm]
    if mode == "all_equal":
        return [f32_bits(0.5)] * count
    if mode == "cluster":
        return [0x3F070000 + i for i in rng.permutation(count)]
    if mode == "low_bits":
        return [0x3F070000 + (i % 4) for i in rng.permutation(count)]
    if mode == "cross_bin":
        out = [f32_bits(0.51)] * count
        for i in range(max(0, count - 7), count):
            out[i] = f32_bits(0.70)
        return out
    if mode == "zero":
        return [NEG_ZERO_BITS if rng.below(2) else 0 for _ in range(count)]
    if mode == "nan":
        choices = [QNAN_BITS, 0xFFC00000, 0x7F800001, 0xFFFFFFFF]
        return [
            choices[rng.below(4)]
            if rng.below(3) == 0
            else f32_bits(float(rng.randint(-8, 8)))
            for _ in range(count)
        ]
    if mode == "inf":
        return [
            POS_INF_BITS
            if rng.below(5) == 0
            else (
                NEG_INF_BITS
                if rng.below(4) == 0
                else f32_bits(float(rng.randint(-8, 8)))
            )
            for _ in range(count)
        ]
    if mode == "flt_max":
        return [
            FLT_MAX_BITS if rng.below(4) == 0 else f32_bits(float(rng.randint(-50, 50)))
            for _ in range(count)
        ]
    if mode == "neg_inf":
        return [NEG_INF_BITS] * count
    if mode == "int_ties":
        return [f32_bits(float(rng.randint(-2, 2))) for _ in range(count)]
    raise ValueError(mode)


def _select_group(
    name: str,
    rng: SplitMix64,
    *,
    k: int,
    s: int,
    o: int,
    fb: int,
    fe: int,
    rows: list[tuple[str, int]],
    roles: Sequence[str],
) -> dict[str, Any]:
    """One select vector group: ``rows`` = ``(mode, nvalid)``; values cover the
    whole row (NaN-poisoned beyond the row's L to prove nothing past L is read)."""
    bits_rows = []
    nv = []
    labels = []
    for mode, nvalid in rows:
        ng = min(max(nvalid, 0), MAX_BLOCKS, k * s)
        L = local_prefix_count(ng, s, o)
        vals = select_row_values(rng, mode, L) if L else []
        tail = [0x7FC0DEAD] * (k - L)  # NaN poison past the valid prefix
        bits_rows.append(vals + tail)
        nv.append(nvalid)
        labels.append(f"{mode}/nvalid={nvalid}")
    bits = torch.tensor(bits_rows, dtype=torch.int64)
    nvt = torch.tensor(nv, dtype=torch.int64)
    expected = select_rows(bits, nvt, fb, fe, s, o)
    for i in range(len(rows)):  # literal transcription agrees on every golden row
        lit = oracle_select_row(bits_rows[i], nv[i], fb, fe, s, o)
        if lit != expected[i].tolist():
            raise AssertionError(f"select oracles disagree on {name} row {i}")
    return {
        "name": name,
        "roles": list(roles),
        "K": k,
        "index_scale": s,
        "index_offset": o,
        "force_begin": fb,
        "force_end": fe,
        "row_labels": labels,
        "scores": encode_array(bits, "f32"),
        "num_valid_pages": encode_array(nvt.to(torch.int32), "i32"),
        "expected_rows": encode_array(expected, "i32"),
    }


def _golden_select(rng: SplitMix64) -> list[dict[str, Any]]:
    groups = []
    nv_small = [0, -3, 1, 2, 15, 16, 17, 20, 33, 39, 40, 41, 100]
    forced_sets = [(0, 0), (0, 1), (1, 1), (2, 3), (3, 4), (8, 8), (16, 0), (0, 16)]
    roles = ("F4_select_prefill", "F5_select_publish", "F6_topk16_publish")
    for fb, fe in forced_sets:
        rows = []
        for i, mode in enumerate(_SELECT_MODES):
            rows.append((mode, 40))
            rows.append((mode, nv_small[(i * 3 + fb + fe) % len(nv_small)]))
        rows += [("unique", v) for v in nv_small]
        groups.append(
            _select_group(
                f"k40_s1_fb{fb}_fe{fe}",
                rng,
                k=40,
                s=1,
                o=0,
                fb=fb,
                fe=fe,
                rows=rows,
                roles=roles,
            )
        )
    for o in (0, 1):
        rows = [
            (m, v)
            for m in ("unique", "all_equal", "int_ties", "nan", "zero")
            for v in (0, 1, 2, 3, 31, 32, 33, 34, 79, 80, 81)
        ]
        groups.append(
            _select_group(
                f"k40_s2_o{o}_fb2_fe3",
                rng,
                k=40,
                s=2,
                o=o,
                fb=2,
                fe=3,
                rows=rows,
                roles=roles,
            )
        )
    # medium rows: > 128 candidates (D1-D3 territory), L boundaries around 128 / 1280 / 1536
    nv_med = [127, 128, 129, 1279, 1280, 1344, 1535, 1536, 1537]
    for fb, fe, kmed in ((0, 1, 1537), (2, 3, 300)):
        rows = [
            (m, kmed)
            for m in (
                "unique",
                "all_equal",
                "cluster",
                "low_bits",
                "cross_bin",
                "zero",
                "int_ties",
                "nan",
            )
        ]
        rows += [("int_ties", v) for v in nv_med if v <= kmed or kmed == 1537]
        groups.append(
            _select_group(
                f"k{kmed}_s1_fb{fb}_fe{fe}",
                rng,
                k=kmed,
                s=1,
                o=0,
                fb=fb,
                fe=fe,
                rows=rows,
                roles=roles,
            )
        )
    # D6: nvalid > 8192 with K*s > 8192 (clamped to 8192, merge-consistent)
    for o in (0, 1):
        rows = [("int_ties", v) for v in (8191, 8192, 8193, 8400)]
        groups.append(
            _select_group(
                f"k4200_s2_o{o}_d6",
                rng,
                k=4200,
                s=2,
                o=o,
                fb=0,
                fe=1,
                rows=rows,
                roles=roles,
            )
        )
    rows = [("int_ties", v) for v in (8192, 8199, 12287)]
    groups.append(
        _select_group(
            "k8200_s1_d6", rng, k=8200, s=1, o=0, fb=1, fe=1, rows=rows, roles=roles
        )
    )
    rows = [("cluster", 12287), ("int_ties", 12287)]
    groups.append(
        _select_group(
            "k12287_s1_max", rng, k=12287, s=1, o=0, fb=0, fe=1, rows=rows, roles=roles
        )
    )
    return groups


def _merge_case_rows(
    rng: SplitMix64, world: int
) -> list[tuple[str, list[tuple[int, int]], int]]:
    """Synthetic merge rows ``(label, cands[W*16], nvalid)``."""
    m = world * TOPK
    rows: list[tuple[str, list[tuple[int, int]], int]] = []

    def pad(c: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return (c + [(NEG_INF_BITS, PAD_ID)] * m)[:m]

    def src_rows(lists: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
        out = []
        for lst in lists:
            lst = sorted(lst, key=lambda x: x[1])
            out += (lst + [(NEG_INF_BITS, PAD_ID)] * TOPK)[:TOPK]
        return pad(out)

    nb = 60
    for trial in range(
        6
    ):  # random rank lists over a shared id space (duplicates with different scores)
        lists = []
        for _ in range(world):
            ids = sorted(rng.permutation(nb)[:TOPK])
            lists.append([(f32_bits(float(rng.randint(-6, 6))), i) for i in ids])
        rows.append((f"random_dup#{trial}", src_rows(lists), nb))
    same = sorted(rng.permutation(nb)[:TOPK])
    lists = [[(f32_bits(float(r - 1)), i) for i in same] for r in range(world)]
    rows.append(("same_ids_dup_max", src_rows(lists), nb))
    lists = [
        [(f32_bits(1.0), i) for i in range(r * 4, r * 4 + TOPK)] for r in range(world)
    ]
    rows.append(("exact_ties_lower_id", src_rows(lists), nb))
    lists = [
        [(NEG_ZERO_BITS if (i + r) % 2 else 0, i) for i in range(r, r + TOPK)]
        for r in range(world)
    ]
    rows.append(("pm_zero_ties", src_rows(lists), nb))
    lists = [
        [
            (
                QNAN_BITS
                if i % 3 == 0
                else (0xFFC00001 if i % 3 == 1 else f32_bits(-1.0)),
                i,
            )
            for i in range(r * 16, r * 16 + 16)
        ]
        for r in range(world)
    ]
    rows.append(("nan_is_neg_inf", src_rows(lists), 200))
    lists = [[(POS_INF_BITS, i) for i in range(10, 10 + TOPK)] for _ in range(world)]
    rows.append(("real_inf_vs_forced", src_rows(lists), 40))
    lists = [
        [(f32_bits(float(i)), i) for i in range(8180 + r, 8180 + r + TOPK)]
        for r in range(world)
    ]
    rows.append(("clamp_8192", src_rows(lists), 9000))
    rows.append(("clamp_8192_exact", src_rows(lists), 8192))
    lists = [[(f32_bits(float(i)), i) for i in range(0, 5)] for _ in range(world)]
    rows.append(("few_valid", src_rows(lists), 5))
    rows.append(("padding_nvalid0", src_rows(lists), 0))
    rows.append(("padding_nvalid_neg", src_rows(lists), -2))
    lists = [
        [(f32_bits(3.0), -5), (f32_bits(9.0), 70), (f32_bits(2.0), 3)]
        for _ in range(world)
    ]
    rows.append(("invalid_ids", src_rows(lists), 50))
    rows.append(("all_pads", pad([]), 30))
    # duplicate ids inside one source (not produced by a correct selector, but legal merge input)
    dup = [(f32_bits(float(rng.randint(0, 9))), 7) for _ in range(TOPK)]
    rows.append(
        (
            "dup_within_source",
            pad(dup + src_rows([[(f32_bits(1.0), i) for i in range(16)]])[: m - TOPK]),
            40,
        )
    )
    return rows


def _golden_merge(rng: SplitMix64) -> list[dict[str, Any]]:
    """Per W: candidate rows ``[N, W, 32]`` (words 0..15 score bits, 16..31
    ids of each source row) and the expected ``[N, 16]`` output for every
    ``(init, local, prefill)`` setting."""
    groups = []
    for world in SUPPORTED_TP:
        rows = _merge_case_rows(rng, world)
        bits = torch.tensor([[c[0] for c in r[1]] for r in rows], dtype=torch.int64)
        ids = torch.tensor([[c[1] for c in r[1]] for r in rows], dtype=torch.int64)
        nv = torch.tensor([r[2] for r in rows], dtype=torch.int64)
        settings = []
        for init, local in ((0, 1), (1, 1), (2, 3), (0, 0), (8, 8)):
            for prefill in (False, True):
                exp = merge_rows(bits, ids, nv, init, local, prefill)
                for i, r in enumerate(rows):
                    lit = oracle_merge_row(r[1], r[2], init, local, prefill)
                    vt = vllm_merge_transcription(
                        r[1], world, r[2], init, local, prefill
                    )
                    if lit != exp[i].tolist() or vt != lit:
                        raise AssertionError(
                            f"merge oracles disagree on W{world} {r[0]}"
                        )
                settings.append(
                    {
                        "init_blocks": init,
                        "local_blocks": local,
                        "prefill": prefill,
                        "expected_output": encode_array(exp, "i32"),
                    }
                )
        cands = torch.stack([bits, ids], dim=-1).reshape(len(rows), world, TOPK, 2)
        groups.append(
            {
                "name": f"w{world}",
                "roles": ["F7_candidate_merge"],
                "world_size": world,
                "row_labels": [r[0] for r in rows],
                "candidates": encode_array(
                    torch.cat(
                        [_as_i32_bits(cands[..., 0]), cands[..., 1].to(torch.int32)],
                        dim=-1,
                    ),
                    "i32",
                ),
                "num_valid_pages": encode_array(nv.to(torch.int32), "i32"),
                "settings": settings,
            }
        )
    return groups


def _golden_addressing() -> dict[str, Any]:
    """Publish destination table (F5/F6): one row per (W, slot, rank, head, token)."""
    columns = [
        "world_size",
        "heads_per_rank",
        "capacity",
        "slot",
        "rank",
        "head",
        "token",
        "owner",
        "row",
        "legacy_word_offset",
        "legacy_ready_word",
        "tagged_byte_offset",
        "generation_index",
        "storage_words",
    ]
    rows = []
    for world in SUPPORTED_TP:
        hpr = NUM_HEADS // world
        cap = 7
        lay = comm_layout(world, hpr, cap)
        for slot in range(NUM_SLOT_SETS):
            for rank in range(world):
                for h in range(NUM_HEADS):
                    for t in (0, cap - 1):
                        row = publish_row_index(slot, world, rank, hpr, h, cap, t)
                        rows.append(
                            [
                                world,
                                hpr,
                                cap,
                                slot,
                                rank,
                                h,
                                t,
                                publish_owner(h, hpr),
                                row,
                                row * ROW_WORDS,
                                row * ROW_WORDS + 2 * TOPK,
                                lay["tagged_offset_bytes"] + row * TAGGED_WORDS * 8,
                                generation_index(slot, world, hpr, h, cap, t),
                                lay["storage_words"],
                            ]
                        )
    return {
        "roles": ["F5_select_publish", "F6_topk16_publish"],
        "columns": columns,
        "rows": rows,
    }


def _golden_protocol(rng: SplitMix64) -> list[dict[str, Any]]:
    """Multi-layer, multi-step decode publish/merge sequences (slot rotation,
    legacy / tagged mixes, generation wrap 0xFFFFFFFE -> 0xFFFFFFFF -> 1)."""
    configs = [
        # (world, layers, modes per step, page_size, capacity, tokens, gen_start)
        (2, 1, ("legacy", "legacy", "legacy"), 128, 4, 3, 0),
        (2, 2, ("tagged", "tagged", "tagged"), 128, 4, 3, 0xFFFFFFFE),
        (2, 3, ("tagged", "legacy", "tagged"), 256, 4, 3, 0xFFFFFFFE),
        (2, 5, ("tagged", "tagged", "legacy"), 128, 4, 2, 0),
        (2, 5, ("legacy", "legacy", "legacy"), 256, 4, 2, 0),
        (4, 2, ("tagged", "legacy", "tagged"), 256, 3, 2, 0xFFFFFFFE),
        (4, 3, ("legacy", "tagged", "tagged"), 128, 3, 2, 0),
    ]
    k = 18
    out = []
    for world, layers, modes, page, cap, tokens, gstart in configs:
        lay = IcpLayout(page, world)
        hpr = NUM_HEADS // world
        em = CommEmulator(world, hpr, cap, gstart)
        steps = []
        for step, mode in enumerate(modes):
            tagged = mode == "tagged" and layers > 1
            for layer in range(layers):
                slot = slot_set(layer, layers)
                nv = [rng.randint(0, 2 * k * lay.index_scale) for _ in range(tokens)]
                if step == 1:
                    nv[0] = (
                        0  # graph padding row: published (16 pads) and merged (all -1)
                    )
                scores, rows = [], []
                for rank in range(world):
                    sc = torch.tensor(
                        [
                            [
                                [f32_bits(float(rng.randint(-1, 1))) for _ in range(k)]
                                for _ in range(NUM_HEADS)
                            ]
                            for _ in range(tokens)
                        ],
                        dtype=torch.int64,
                    )
                    r = select_rows(
                        sc.reshape(tokens * NUM_HEADS, k),
                        torch.tensor(nv).repeat_interleave(NUM_HEADS),
                        0,
                        1,
                        lay.index_scale,
                        lay.index_offset(rank),
                    ).reshape(tokens, NUM_HEADS, ROW_WORDS)
                    scores.append(sc)
                    rows.append(r)
                    em.publish(rank, slot, r, tagged=tagged)
                outs = [
                    em.merge(owner, slot, torch.tensor(nv), 0, 1, tagged=tagged)
                    for owner in range(world)
                ]
                steps.append(
                    {
                        "step": step,
                        "layer": layer,
                        "slot": slot,
                        "tagged": tagged,
                        "num_valid_pages": encode_array(
                            torch.tensor(nv, dtype=torch.int32), "i32"
                        ),
                        "scores": [encode_array(sc, "f32") for sc in scores],
                        "expected_output": [encode_array(o, "i32") for o in outs],
                    }
                )
        out.append(
            {
                "name": f"w{world}_l{layers}_{'-'.join(modes)}_p{page}_g{gstart:x}",
                "roles": [
                    "F5_select_publish",
                    "F6_topk16_publish",
                    "F7_candidate_merge",
                ],
                "world_size": world,
                "heads_per_rank": hpr,
                "capacity": cap,
                "tokens": tokens,
                "num_layers": layers,
                "page_size": page,
                "K": k,
                "force_begin": 0,
                "force_end": 1,
                "init_blocks": 0,
                "local_blocks": 1,
                "generation_start": gstart,
                "comm_layout": {
                    kk: (list(v) if isinstance(v, tuple) else v)
                    for kk, v in em.layout.items()
                },
                "steps": steps,
                "final_storage_sha256": [em.storage_sha256(r) for r in range(world)],
                "final_publish_generations": [
                    encode_array(g.to(torch.int64) & _MASK32, "u32")
                    for g in em.generations
                ],
                "final_expected_generations": [
                    encode_array(g.to(torch.int64) & _MASK32, "u32")
                    for g in em.expected
                ],
            }
        )
    return out


def golden_decode_cases() -> list[DecodeCase]:
    """Decode chain fixtures (small; both page sizes)."""
    cases = []
    for page in SUPPORTED_PAGE_SIZES:
        cases.append(
            make_decode_case(
                f"dec_p{page}_mixed_q1",
                page_size=page,
                seq_lens=[1, 40, 100, 128, 129, 0, 200, 641],
                decode_query_len=1,
                columns=None,
                k_kind="sparse8",
                seed=78960 + page,
                q_kind="sparse8",
                init_blocks=0,
                local_blocks=1,
                notes="causal tails at block / page edges, empty shards (P256 TP4 seq<=64, TP2 seq<=128), graph padding (seq 0), extra untouched columns",
            )
        )
        cases.append(
            make_decode_case(
                f"dec_p{page}_mtp_q4",
                page_size=page,
                seq_lens=[3, 2300],
                decode_query_len=4,
                columns=24 if page == 128 else 12,
                seed=78970 + page,
                k_kind="sparse8",
                q_kind="sparse8",
                page_order="repeat",
                distinct_pages=3 if page == 128 else 2,
                init_blocks=1,
                local_blocks=1,
                notes="EAGLE3 dql 4, q_pos < 0 rows (nvalid 0), L > 16 with repeated pages (exact cross-block ties)",
            )
        )
        cases.append(
            make_decode_case(
                f"dec_p{page}_ties_palette",
                page_size=page,
                seq_lens=[2200, 300],
                decode_query_len=1,
                seed=78980 + page,
                k_kind="palette",
                page_order="repeat",
                distinct_pages=4 if page == 128 else 2,
                init_blocks=2,
                local_blocks=3,
                notes="3-row palette keys: massive exact ties at the cutoff -> lower id",
            )
        )
        cases.append(
            make_decode_case(
                f"dec_p{page}_all_equal",
                page_size=page,
                seq_lens=[2100, 1],
                decode_query_len=2,
                seed=78990 + page,
                k_kind="all_equal",
                page_order="repeat",
                distinct_pages=2,
                init_blocks=0,
                local_blocks=1,
                q_kind="narrow",
                notes="every block scores the same; narrow-exponent (non-integer) Q",
            )
        )
    return cases


def golden_prefill_cases() -> list[PrefillCase]:
    cases = []
    for page in SUPPORTED_PAGE_SIZES:
        cases.append(
            make_prefill_case(
                f"pre_p{page}_msa_segments",
                page_size=page,
                q_lens=[0, 65, 0, 3],
                prefixes=[127, 194, 0, 127],
                padding_tokens=3,
                chunk=32,
                k_kind="sparse8",
                seed=79000 + page,
                q_kind="sparse8",
                init_blocks=0,
                local_blocks=1,
                notes="MSA smoke segments (empty segments, prefix 127/194), chunk 32 -> 3 windows, 3 graph-padding rows",
            )
        )
        cases.append(
            make_prefill_case(
                f"pre_p{page}_two_req",
                page_size=page,
                q_lens=[33, 1],
                prefixes=[0, 128],
                padding_tokens=0,
                chunk=16,
                k_kind="sparse8",
                seed=79010 + page,
                q_kind="sparse8",
                init_blocks=1,
                local_blocks=1,
                notes="prefix 0 causal tails inside the first block; chunk 16 splits request 0 over 3 windows",
            )
        )
        cases.append(
            make_prefill_case(
                f"pre_p{page}_long_ties",
                page_size=page,
                q_lens=[8, 3],
                prefixes=[2200, 0],
                padding_tokens=1,
                chunk=8,
                seed=79020 + page,
                k_kind="palette",
                page_order="repeat",
                distinct_pages=4 if page == 128 else 2,
                init_blocks=2,
                local_blocks=3,
                notes="L > 16 after a long prefix, palette ties, carrier across 2 windows",
            )
        )
    return cases


def _decode_case_json(case: DecodeCase) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": case.label,
        "roles": [
            "F3_decode_score",
            "F5_select_publish",
            "F6_topk16_publish",
            "F7_candidate_merge",
        ],
        "notes": case.notes,
        "page_size": case.page_size,
        "decode_query_len": case.decode_query_len,
        "seq_lens": encode_array(torch.tensor(case.seq_lens, dtype=torch.int32), "i32"),
        "block_table": encode_array(
            torch.tensor(case.block_table, dtype=torch.int32), "i32"
        ),
        "columns": case.columns,
        "index_q": encode_array(case.q, "u8"),
        "cache_pages": encode_array(case.cache, "u8"),
        "num_valid_pages": encode_array(
            torch.tensor(case.nvalid, dtype=torch.int32), "i32"
        ),
        "init_blocks": case.init_blocks,
        "local_blocks": case.local_blocks,
        "per_tp": {},
    }
    glob = decode_global_output(case)
    entry["expected_global_output"] = encode_array(glob, "i32")
    for tp in SUPPORTED_TP:
        lay = IcpLayout(case.page_size, tp)
        planes, written, rows = [], None, []
        for rank in range(tp):
            plane, written = decode_local_plane(case, lay, rank)
            planes.append(plane)
            rows.append(decode_local_rows(case, lay, rank, plane))
        outs = [decode_owner_output(case, lay, rows, owner) for owner in range(tp)]
        hpr = case.num_heads // tp
        merged = torch.cat(outs, dim=1)
        if not torch.equal(merged, glob):
            raise AssertionError(f"containment failed for {case.label} {lay.name}")
        assert written is not None
        entry["per_tp"][str(tp)] = {
            "layout": lay.name,
            "index_scale": lay.index_scale,
            "index_offset": [lay.index_offset(r) for r in range(tp)],
            "heads_per_rank": hpr,
            "written_columns": encode_array(written, "i32"),
            "expected_scores": [encode_array(p, "f32") for p in planes],
            "expected_rows": [encode_array(r, "i32") for r in rows],
            "expected_output": [encode_array(o, "i32") for o in outs],
        }
    return entry


def _prefill_case_json(case: PrefillCase) -> dict[str, Any]:
    windows = case.windows()
    entry: dict[str, Any] = {
        "name": case.label,
        "roles": [
            "F1_prefill_plan",
            "F2_prefill_score",
            "F4_select_prefill",
            "F7_candidate_merge",
        ],
        "notes": case.notes,
        "page_size": case.page_size,
        "q_lens": encode_array(torch.tensor(case.q_lens, dtype=torch.int32), "i32"),
        "prefixes": encode_array(torch.tensor(case.prefixes, dtype=torch.int32), "i32"),
        "padding_tokens": case.padding_tokens,
        "chunk": case.chunk,
        "block_table": encode_array(
            torch.tensor(case.block_table, dtype=torch.int32), "i32"
        ),
        "columns": case.columns,
        "index_q": encode_array(case.q, "u8"),
        "cache_pages": encode_array(case.cache, "u8"),
        "positions": encode_array(
            torch.tensor(case.positions(), dtype=torch.int32), "i32"
        ),
        "num_valid_pages": encode_array(
            torch.tensor(case.nvalid(), dtype=torch.int32), "i32"
        ),
        "init_blocks": case.init_blocks,
        "local_blocks": case.local_blocks,
        "windows": [
            {
                "start": w.start,
                "end": w.end,
                "local_begin": w.local_begin,
                "q_lens": w.q_lens,
                "kv_lens": w.kv_lens,
                "qo_offsets": w.qo_offsets,
                "plan_storage": plan_storage(case.num_heads, w.total_q, w.batch),
                "plan_tiles32": plan_tiles(w),
            }
            for w in windows
        ],
        "per_tp": {},
    }
    glob = prefill_global_output(case)
    entry["expected_global_output"] = encode_array(glob, "i32")
    noncompact_cols = prefill_columns(max(case.seq_lens + [1]), case.page_size, False)
    for tp in SUPPORTED_TP:
        lay = IcpLayout(case.page_size, tp)
        carriers, scores, cover, nc_scores = [], [], [], []
        for rank in range(tp):
            carriers.append(prefill_local_carrier(case, lay, rank))
            per_w, per_c, per_nc = [], [], []
            for w in windows:
                sc, cov = prefill_window_scores(case, w, lay, rank)
                per_w.append(encode_array(sc, "f32"))
                per_c.append(encode_array(cov, "i32"))
                if case.page_size == 256:
                    nc, ncov = prefill_window_scores(
                        case, w, lay, rank, compact=False, columns=noncompact_cols
                    )
                    per_nc.append(
                        {
                            "scores": encode_array(nc, "f32"),
                            "coverage": encode_array(ncov, "i32"),
                        }
                    )
            scores.append(per_w)
            cover.append(per_c)
            nc_scores.append(per_nc)
        outs = [prefill_owner_output(case, lay, carriers, owner) for owner in range(tp)]
        merged = torch.cat(outs, dim=1)
        if not torch.equal(merged, glob):
            raise AssertionError(f"containment failed for {case.label} {lay.name}")
        per = {
            "layout": lay.name,
            "index_scale": lay.index_scale,
            "index_offset": [lay.index_offset(r) for r in range(tp)],
            "heads_per_rank": case.num_heads // tp,
            "expected_scores_initialize_true": scores,
            "coverage_initialize_false": cover,
            "expected_carrier": [encode_array(c, "i32") for c in carriers],
            "expected_output": [encode_array(o, "i32") for o in outs],
        }
        if case.page_size == 256:
            per["noncompact_columns"] = noncompact_cols
            per["expected_scores_noncompact"] = nc_scores
        entry["per_tp"][str(tp)] = per
    return entry


def build_golden() -> dict[str, Any]:
    """Content of ``cake_msa_icp_golden.json`` (deterministic)."""
    rng = SplitMix64(0x7896_0001)
    return {
        "meta": {
            "version": GOLDEN_VERSION,
            "generator": "tests/test_helpers/cake_msa_icp_inputs.py build_golden()",
            "contract": "vLLM-project/MSA#46 f380d0e8 + vLLM#63671 a76900d6 with the FI deviations of the helper docstring (D1-D6)",
            "encoding": "arrays are {dtype, shape, zlib_b64}: little-endian raw bytes, zlib, base64; f32 = raw IEEE bits; u32 decodes to int64; u8 E4M3 bytes",
            "constants": {
                "head_dim": HEAD_DIM,
                "num_heads": NUM_HEADS,
                "topk": TOPK,
                "block": BLOCK,
                "max_blocks": MAX_BLOCKS,
                "max_select_columns_exclusive": MAX_SELECT_COLUMNS,
                "row_words": ROW_WORDS,
                "tagged_words": TAGGED_WORDS,
                "num_slot_sets": NUM_SLOT_SETS,
                "neg_inf_bits": NEG_INF_BITS,
                "ord_pinf": ORD_PINF,
                "ord_ninf": ORD_NINF,
                "pad_page": PAD_PAGE,
            },
            "roles": {
                "F1_prefill_plan": "prefill[*].windows[*].plan_tiles32 / plan_storage (coverage units)",
                "F2_prefill_score": "prefill[*].per_tp[tp].expected_scores_initialize_true / coverage_initialize_false (+ noncompact for P256)",
                "F3_decode_score": "decode[*].per_tp[tp].expected_scores / written_columns (topk16 mode: expected_rows)",
                "F4_select_prefill": "select[*] (word 32 = 1) and prefill[*].per_tp[tp].expected_carrier",
                "F5_select_publish": "select[*] words 0..31, addressing[*], protocol[*]",
                "F6_topk16_publish": "decode[*].per_tp[tp].expected_rows, protocol[*]",
                "F7_candidate_merge": "merge[*].settings[*] (candidates [N, W, 32]), decode/prefill expected_output, protocol[*]",
            },
            "c_arm": {"values": C_ARM, "status": C_ARM_STATUS},
        },
        "select": _golden_select(rng),
        "merge": _golden_merge(rng),
        "addressing": _golden_addressing(),
        "protocol": _golden_protocol(rng),
        "decode": [_decode_case_json(c) for c in golden_decode_cases()],
        "prefill": [_prefill_case_json(c) for c in golden_prefill_cases()],
    }


def dump_golden(obj: dict[str, Any]) -> str:
    # compact: the file must stay below the 500 KB check-added-large-files limit
    return json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"


def write_golden(path: str | Path | None = None) -> Path:
    p = Path(path) if path else golden_path()
    p.write_text(dump_golden(build_golden()), encoding="utf-8")
    return p


def check_golden(path: str | Path | None = None) -> list[str]:
    """Regenerate and compare content with the file; returns differing paths."""
    fresh = json.loads(dump_golden(build_golden()))
    with open(path or golden_path(), encoding="utf-8") as f:
        stored = json.load(f)
    return golden_equal(fresh, stored)


# ----------------------------------------------------------------------------- self test


def _rand_bits_row(rng: SplitMix64, k: int) -> list[int]:
    mode = _SELECT_MODES[rng.below(len(_SELECT_MODES))]
    return select_row_values(rng, mode, k)


def self_test(verbose: bool = True) -> dict[str, int]:
    """Internal consistency checks (CPU, no GPU)."""
    counts: dict[str, int] = {}

    def ok(name: str, n: int = 1) -> None:
        counts[name] = counts.get(name, 0) + n

    rng = SplitMix64(0x5E1F)
    # 1. keys
    assert ORD_PINF == 0xFF800000 and ORD_NINF == 0x007FFFFF
    assert ordk(0.0) == ordk(-0.0) == 0x80000000
    assert ordk_bits(QNAN_BITS) == ordk_bits(0xFFC00000) == ORD_NINF
    assert ordk(1.0) > ordk(0.5) > ordk(0.0) > ordk(-0.5) > ordk(float("-inf"))
    assert ordk_bits(FLT_MAX_BITS) < ORD_PINF
    assert (
        next_generation(0xFFFFFFFE) == 0xFFFFFFFF
        and next_generation(0xFFFFFFFF) == 1
        and next_generation(0) == 1
    )
    probe = torch.tensor(
        [
            0,
            NEG_ZERO_BITS,
            QNAN_BITS,
            NEG_INF_BITS,
            POS_INF_BITS,
            0x3F800000,
            0xBF800000,
        ],
        dtype=torch.int64,
    )
    assert ordk_tensor(probe).tolist() == [ordk_bits(int(b)) for b in probe]
    ok("keys")
    # 2. layouts: formula vs vLLM slot mapping vs holders
    for lay in LAYOUTS:
        for off in range(lay.page_size):
            owner, local = lay.slot_owner_local(off)
            assert lay.global_row(0, local, owner) == off
            assert lay.gid(0, owner) * BLOCK <= off < (lay.gid(0, owner) + 1) * BLOCK
            assert owner in lay.holders(off // BLOCK)
        for r in range(lay.tp):
            assert lay.index_offset(r) == r * lay.index_scale // lay.tp
            rows = lay.page_rows(r)
            assert sorted(rows.tolist()) == [
                o for o in range(lay.page_size) if lay.slot_owner_local(o)[0] == r
            ]
        ok("layouts")
    for g in range(0, 40):
        for s, o in ((1, 0), (2, 0), (2, 1)):
            assert local_prefix_count(g, s, o) == len(
                [c for c in range(64) if c * s + o < g]
            )
    ok("local_prefix_count")
    # 3. select: literal vs vectorized
    for trial in range(400):
        k = [1, 5, 16, 17, 40, 129, 300][trial % 7]
        s, o = ((1, 0), (2, 0), (2, 1))[trial % 3]
        fb = rng.randint(0, 8)
        fe = rng.randint(0, 16 - fb)
        rows = [_rand_bits_row(rng, k) for _ in range(4)]
        nv = [rng.randint(-2, k * s + 3) for _ in range(4)]
        if trial % 50 == 0:
            nv[0] = 9000
        vec = select_rows(
            torch.tensor(rows, dtype=torch.int64), torch.tensor(nv), fb, fe, s, o
        )
        for i in range(4):
            assert oracle_select_row(rows[i], nv[i], fb, fe, s, o) == vec[i].tolist(), (
                trial,
                i,
            )
        ok("select_literal_vs_vector", 4)
    # 4. MSA smoke expectations reproduced
    for off in (0, 1):
        row = [f32_bits(float(i)) for i in range(32)]
        out = oracle_select_row(row, 61, 2, 3, 2, off)
        gids = list(range(off, 61, 2))
        forced = {i for i in gids if i < 2 or i >= 58}
        ordinary = [i for i in gids if i not in forced]
        selected = forced | set(ordinary[-(16 - len(forced)) :])
        assert out[16:32] == sorted(selected)
        assert out[:16] == [row[(i - off) // 2] for i in sorted(selected)]
        ok("msa_compact_ids_restored")
    for fe in (0, 2):
        out = oracle_select_row([f32_bits(1.0)] * 64, 64, 0, fe)
        assert out[16:32] == list(range(16 - fe)) + list(range(64 - fe, 64))
        ok("msa_stable_cutoff_ties")
    # 5. MSA attribution model: equal to the oracle wherever no deviation applies
    for trial in range(300):
        k = [40, 129, 300][trial % 3]
        vals = [f32_bits(float(v)) for v in rng.permutation(k)]  # unique, finite, no -0
        nv = rng.randint(0, k)
        fb = rng.randint(0, 4)
        fe = rng.randint(0, 4)
        ref = oracle_select_row(vals, nv, fb, fe)[16:32]
        for publish in (False, True):
            got, det = msa_select_model_row(vals, nv, fb, fe, publish=publish)
            assert det and got == ref
        ok("msa_model_no_deviation", 2)
    ties = [f32_bits(1.0)] * 64
    got, det = msa_select_model_row(ties, 64, 0, 0, publish=True)
    assert det and got == list(
        range(48, 64)
    )  # D1: higher local ids win in decode publish
    assert oracle_select_row(ties, 64, 0, 0)[16:32] == list(range(16))
    ok("msa_model_d1")
    # 6. merge: literal vs vectorized vs vLLM transcription
    for world in SUPPORTED_TP:
        for trial in range(150):
            m = world * TOPK
            cands = []
            for _ in range(m):
                i = rng.randint(-2, 70)
                cands.append((_rand_bits_row(rng, 1)[0], i))
            nv = (
                [rng.randint(-1, 72), 0, 9000][trial % 3]
                if trial % 10 == 0
                else rng.randint(1, 72)
            )
            init, local = rng.randint(0, 3), rng.randint(0, 3)
            prefill = bool(trial % 2)
            lit = oracle_merge_row(cands, nv, init, local, prefill)
            vec = merge_rows(
                torch.tensor([[c[0] for c in cands]]),
                torch.tensor([[c[1] for c in cands]]),
                torch.tensor([nv]),
                init,
                local,
                prefill,
            )[0].tolist()
            vt = vllm_merge_transcription(cands, world, nv, init, local, prefill)
            assert lit == vec == vt, (world, trial, lit, vec, vt)
            ok("merge_literal_vector_vllm")
    # 7. containment theorem on synthetic partial planes (ties, NaN, -0, forced)
    for lay in LAYOUTS:
        for trial in range(120):
            nb = rng.randint(1, 70)
            nvalid = nb if trial % 7 else rng.randint(0, nb)
            fb, fe = [(0, 1), (1, 1), (2, 3), (0, 0)][trial % 4]
            mode = ["int_ties", "all_equal", "zero", "unique"][trial % 4]
            gvals = select_row_values(rng, mode, nb)
            gvals = [b if not is_nan_bits(b) else 0 for b in gvals]
            k_cols = _cdiv(nb, lay.index_scale) + rng.randint(0, 3)
            parts = []
            # each holder gets a partial score <= the global score; at least one holder equals it
            for r in range(lay.tp):
                o = lay.index_offset(r)
                row = []
                for c in range(k_cols):
                    b = c * lay.index_scale + o
                    if b < nb and r in lay.holders(b):
                        hs = lay.holders(b)
                        if r == hs[b % len(hs)] or rng.below(3) == 0:
                            row.append(gvals[b])
                        else:
                            row.append(
                                f32_bits(bits_f32(gvals[b]) - float(rng.randint(1, 3)))
                                if not math.isinf(bits_f32(gvals[b]))
                                else NEG_INF_BITS
                            )
                    else:
                        row.append(NEG_INF_BITS if b < nb else 0x7FC0BEEF)
                parts.append(row)
            sel = [
                oracle_select_row(
                    parts[r], nvalid, fb, fe, lay.index_scale, lay.index_offset(r)
                )
                for r in range(lay.tp)
            ]
            cands = [
                (sb, i)
                for r in range(lay.tp)
                for sb, i in zip(sel[r][:16], sel[r][16:32], strict=True)
            ]
            for prefill in (False, True):
                assert oracle_merge_row(
                    cands, nvalid, fb, fe, prefill
                ) == oracle_global(gvals, nvalid, fb, fe, prefill)
            ok("containment_synthetic")
    # 8. chain fixtures: decode / prefill (golden + extra generated) against
    #    independent transcriptions of the vLLM decode scorer and the MSA
    #    prefill smoke-test reference, plus containment vs the unsharded oracle
    extra_dec = [
        make_decode_case(
            "st_dec_p128",
            page_size=128,
            seq_lens=[1500, 129, 0, 2601],
            decode_query_len=4,
            seed=11,
            init_blocks=1,
            local_blocks=1,
        ),
        make_decode_case(
            "st_dec_p256",
            page_size=256,
            seq_lens=[1500, 63, 0, 2601],
            decode_query_len=1,
            seed=12,
        ),
        make_decode_case(
            "st_dec_narrow",
            page_size=128,
            seq_lens=[2400],
            decode_query_len=2,
            seed=13,
            q_kind="narrow",
            k_kind="narrow",
        ),
    ]
    for case in golden_decode_cases() + extra_dec:
        glob = decode_global_output(case)
        for tp in SUPPORTED_TP:
            lay = IcpLayout(case.page_size, tp)
            planes = [decode_local_plane(case, lay, r) for r in range(tp)]
            for r in range(tp):
                ref = vllm_decode_plane_transcription(case, tp, r)
                got = planes[r][0]
                same = (ref.view(torch.int32) == got.view(torch.int32)) | (
                    torch.isnan(ref) & torch.isnan(got)
                )
                assert bool(same.all()), (case.label, lay.name, r)
            rows = [decode_local_rows(case, lay, r, planes[r][0]) for r in range(tp)]
            merged = torch.cat(
                [decode_owner_output(case, lay, rows, o) for o in range(tp)], dim=1
            )
            assert torch.equal(merged, glob), (case.label, lay.name)
            ok("decode_chain")
    extra_pre = [
        make_prefill_case(
            "st_pre_p128",
            page_size=128,
            q_lens=[300, 0, 77],
            prefixes=[1000, 5, 2000],
            padding_tokens=5,
            chunk=128,
            seed=21,
        ),
        make_prefill_case(
            "st_pre_p256",
            page_size=256,
            q_lens=[257, 1],
            prefixes=[0, 128],
            chunk=256,
            seed=22,
            init_blocks=1,
            local_blocks=1,
        ),
    ]
    for case in golden_prefill_cases() + extra_pre:
        glob = prefill_global_output(case)
        nc_cols = prefill_columns(max(case.seq_lens), case.page_size, False)
        for tp in SUPPORTED_TP:
            lay = IcpLayout(case.page_size, tp)
            for r in range(tp):
                for w in case.windows():
                    ref = msa_prefill_scores_transcription(case, w, tp, r, nc_cols)
                    nc, nc_cov = prefill_window_scores(
                        case, w, lay, r, compact=False, columns=nc_cols
                    )
                    cmp_cols = torch.arange(nc_cols)[None, :] < nc_cov[:, None].to(
                        torch.int64
                    )  # [T_w, K]
                    m = cmp_cols[:, None, :].expand_as(nc)
                    assert torch.equal(nc[m], ref[m]), (
                        case.label,
                        lay.name,
                        r,
                        w.start,
                    )
                    if case.page_size == 128:
                        c, c_cov = prefill_window_scores(case, w, lay, r)
                        assert torch.equal(c_cov, nc_cov)
                        assert torch.equal(
                            c[..., : min(case.columns, nc_cols)],
                            nc[..., : min(case.columns, nc_cols)],
                        )
                    else:  # compact column p == non-compact gid column 2p + o
                        c, c_cov = prefill_window_scores(case, w, lay, r)
                        o = lay.index_offset(r)
                        for t in range(w.total_q):
                            for p in range(int(c_cov[t])):
                                assert torch.equal(c[t, :, p], nc[t, :, 2 * p + o]), (
                                    case.label,
                                    lay.name,
                                    r,
                                    t,
                                    p,
                                )
            carriers = [prefill_local_carrier(case, lay, r) for r in range(tp)]
            merged = torch.cat(
                [prefill_owner_output(case, lay, carriers, o) for o in range(tp)], dim=1
            )
            assert torch.equal(merged, glob), (case.label, lay.name)
            ok("prefill_chain")
    # 9. protocol: rotation distances, wrap, timeouts
    for n in range(1, 8):
        sl = [slot_set(i, n) for i in range(n)]
        seq = sl * 3
        for i in range(1, len(seq)):
            assert seq[i] != seq[i - 1] or n == 1
    em = CommEmulator(2, 2, 2)
    rows = torch.zeros((2, 4, ROW_WORDS), dtype=torch.int32)
    em.publish(0, 0, rows, tagged=False)
    try:
        em.publish(0, 0, rows, tagged=False)
        raise AssertionError("second legacy publish must time out")
    except ProtocolTimeout:
        pass
    try:
        em.merge(0, 0, torch.tensor([1, 1]), 0, 1, tagged=False)
        raise AssertionError("merge with a missing source must time out")
    except ProtocolTimeout:
        pass
    em2 = CommEmulator(2, 2, 2, 0xFFFFFFFF)
    em2.publish(0, 1, rows, tagged=True)
    em2.publish(1, 1, rows, tagged=True)
    em2.merge(0, 1, torch.tensor([3, 0]), 0, 1, tagged=True)
    assert int(em2.generations[0][1, 0, 0]) == 1 and int(em2.expected[0][1, 0, 0]) == 1
    ok("protocol")
    # 10. golden round trip + size
    g = build_golden()
    text = dump_golden(g)
    assert len(text.encode()) < 500 * 1024, len(
        text
    )  # pre-commit check-added-large-files
    assert not golden_equal(json.loads(text), g)
    ok("golden_build")
    # 11. adversarial accumulation fixtures are valid E4M3 and exact in float64
    q8, k8, labels = accumulation_adversarial_cases()
    assert q8.shape == k8.shape and len(labels) == q8.shape[0]
    exact = check_accumulation_bound(
        (e4m3_to_f64(q8) * e4m3_to_f64(k8)).sum(-1), q8, k8, C_ARM["tcgen05_f8f6f4_k32"]
    )
    assert exact["holds"] and exact["max_ratio"] == 0.0
    ok("accumulation_fixtures")
    if verbose:
        for k_, v in counts.items():
            print(f"self-test {k_}: {v} passed")
        print(f"golden bytes: {len(text.encode())}")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("self-test", "write-golden", "check-golden"):
        print(__doc__.split("Command line")[1] if __doc__ else "")
        return 2
    cmd = args[0]
    path = args[1] if len(args) > 1 else None
    if cmd == "self-test":
        self_test()
        return 0
    if cmd == "write-golden":
        p = write_golden(path)
        data = p.read_bytes()
        print(f"wrote {p} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}")
        return 0
    diffs = check_golden(path)
    if diffs:
        print("golden content differs at:", *diffs[:20], sep="\n  ")
        return 1
    print("golden content matches the generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
