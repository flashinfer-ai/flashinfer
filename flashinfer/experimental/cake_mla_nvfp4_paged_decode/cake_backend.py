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
"""

# Cake dense MLA decode over an NVFP4 paged latent cache (SM100 / SM103).
#
# Host side of the generated Cake programs for absorbed multi-head latent
# attention decode (Kimi-K3 / DeepSeek-V3 geometry: 512 latent + 64 rope
# channels per key, 512-wide value) over the NVFP4 paged cache layout of
# flashinfer-ai/flashinfer#4676:
#
# * ``ckv_cache``    uint8          [num_pages, page_size, 256]  packed E2M1 (even element low nibble)
# * ``ckv_sf_cache`` float8_e4m3fn  [num_pages, page_size, 32]   one block scale per 16 latent channels
# * ``kpe_cache``    float8_e4m3fn  [num_pages, page_size, 64]   rope channels at the cache scale ``kpe_scale``
#
# with a decoded key ``e2m1 * sf * ckv_scale | fp8 * kpe_scale``.  The query is
# NVFP4 too (``quantize_mla_nvfp4_query``): per (token, head) row 256 packed
# E2M1 bytes, 32 E4M3 block scales, 64 FP8 rope channels and one FP32 row
# scale ``q_scale`` chosen so the rope product lands in the latent logit unit;
# the kernel's logits are ``sm_scale * ckv_scale * q_scale * (QN . KN + QR . KR)``.
#
# The attention kernel is a swapped-AB tcgen05 schedule: one CTA per (request,
# row tile, KV split) computes S^T[128 tokens, RT rows] with two block-scaled
# ``kind::mxf4nvf4`` MMAs (the cache block scales are the A scale factors, the
# query block scales the B scale factors) plus one ``kind::f8f6f4`` rope MMA,
# converts V on chip to E4M3 and accumulates O^T on ``kind::f8f6f4``.  Row
# tiles of 16, 32 and 48 packed (token, head) rows are separate programs; a
# split-KV merge kernel (warp-per-row reducers for ``num_split <= 32``, a CTA
# reducer otherwise) folds the partials and emits the natural-log LSE.  The
# plan (row tile, split count, grids, reducer) depends only on the batch
# shape, the longest KV and the device's SM count; it is computed once per
# distinct shape (:func:`plan_mla_nvfp4_paged_decode`) and reused.  Nothing
# here reads a CUDA tensor's contents.

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import ARG_PLANS, FFI_ENTRY, load_program, select_program

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE  # 576 BF16 query channels before quantization
V_DIM = LATENT
SF_VEC = 16  # latent channels per E4M3 block scale
CKV_BYTES = LATENT // 2  # 256 packed E2M1 bytes per token / query row
SF_BYTES = LATENT // SF_VEC  # 32 block scales per token / query row
E2M1_MAX = 6.0
E4M3_MAX = 448.0
TILE_TOK = 128  # tokens per KV tile (tcgen05 M of the QK MMA)
BOX_TOK = 32  # tokens per TMA box: page sizes are powers of two >= 32
MAX_SPLITS = 256
MIN_TILES_PER_SPLIT = 2
ROW_TILES = (16, 32, 48)  # packed (token, head) rows per CTA
# Two-CTA wide route (requests with more than WIDE_MIN_ROWS packed rows): 128 rows per cluster of two CTAs.
WIDE_MIN_ROWS = 48
WIDE_BLOCK_M = 128
WIDE_CLUSTER = 2  # CTAs per cluster
WIDE_LSE_BIAS = 6.0  # the wide kernel's P = 2^6 exp2(s - m) (uniform V shift; no lazy-ratchet slack in the bias)
REDUCE_WARPS = 8  # warp reducer CTA: 256 threads
REDUCE_CTA_MIN_SPLITS = 33  # above the warp reducer's lane-per-split range
REDUCE_DIM_CHUNKS = 4  # CTA reducer: 128 latent dims per CTA
QUANTIZE_ROWS_PER_CTA = 4  # query quantizer: one warp per (token, head) row
# The softmax writes P = exp2(s - R + LSE_BIAS) with a per-column ratchet
# reference R; the merge removes the bias when it forms the LSE.  Baked into
# the attention programs (log2(448) - 3 - 2).
LSE_BIAS = math.log2(E4M3_MAX) - 3.0 - 2.0
LOG2E = math.log2(math.e)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
WORKSPACE_ALIGNMENT = 16

# Keyword names the stages are bound with (the registry's argument plans refer
# to these names; ``_bind_stage`` orders them by the generated argument plan).
MAIN_KWARGS = (
    "tmap_qn",
    "tmap_qs",
    "tmap_qr",
    "tmap_k",
    "tmap_ks",
    "tmap_kr",
    "q_scale",
    "partial_O",
    "partial_max",
    "partial_sum",
    "lse",
    "seq_lens",
    "kv_len_global",
    "cum_seq_lens_q",
    "page_table",
    "softmax_scale_log2",
    "bmm2_scale",
    "num_heads",
    "num_split",
    "max_pages_per_seq",
    "page_shift",
    "cp_world",
    "cp_rank",
    "has_lse",
    "grid",
)
REDUCE_KWARGS = (
    "partial_O",
    "partial_max",
    "partial_sum",
    "O",
    "lse",
    "cum_seq_lens_q",
    "batch",
    "num_heads",
    "num_split",
    "bmm2_scale",
    "lse_bias",
    "has_lse",
    "grid",
)
QUANTIZE_KWARGS = (
    "q_bf16",
    "q_nope",
    "q_sf",
    "q_rope",
    "q_scale",
    "rows",
    "c_nope",
    "c_rope",
    "kpe_scale",
    "ckv_scale",
    "grid",
)


# ---------------------------------------------------------------------------
# Planning (pure functions of host scalars)
# ---------------------------------------------------------------------------


def rt_for_rows(rows_per_request: int) -> tuple[int, int]:
    """``(row tile, row tiles per request)``: the fewest 48-row tiles, then the
    smallest row tile that still covers the request's rows."""
    m_tiles = max(1, (rows_per_request + ROW_TILES[-1] - 1) // ROW_TILES[-1])
    for rt in ROW_TILES:
        if rt * m_tiles >= rows_per_request:
            return rt, m_tiles
    return ROW_TILES[-1], m_tiles


def use_wide_route(rows_per_request: int) -> bool:
    """Requests with more packed rows than the largest row tile take the two-CTA wide route."""
    return rows_per_request > WIDE_MIN_ROWS


def wide_tiles(rows_per_request: int) -> int:
    return (rows_per_request + WIDE_BLOCK_M - 1) // WIDE_BLOCK_M


def plan_num_split(items: int, max_seq_len: int, sm_count: int) -> int:
    """One CTA per SM per wave: fill the machine with (work item, split) pairs."""
    target = max(1, sm_count // max(1, items))
    max_by_len = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK // MIN_TILES_PER_SPLIT)
    return max(1, min(target, MAX_SPLITS, max_by_len))


def reduce_warps_per_row(rows: int) -> int:
    if rows <= 256:
        return 4
    if rows <= 1024:
        return 2
    return 1


@dataclass(frozen=True)
class DecodePlan:
    """Launch plan of one batch shape: row tile, split plan, grids and reducer."""

    rt: int  # rows per CTA (16 / 32 / 48) or per cluster (128, wide route)
    m_tiles: int
    num_split: int
    rows_max: int
    grid_main: tuple[
        int, int, int
    ]  # wide route: 2 * num_split CTAs along x (one cluster per split)
    reduce_kind: str  # "reduce_w4" / "reduce_w2" / "reduce_w1" / "reduce_cta"; unused when num_split == 1
    reduce_warps: int  # 0 for the CTA reducer
    grid_reduce: tuple[int, int, int]
    wide: bool = False

    @property
    def main_kind(self) -> str:
        return "main_wide" if self.wide else f"main_rt{self.rt}"

    @property
    def lse_bias(self) -> float:
        return WIDE_LSE_BIAS if self.wide else LSE_BIAS


@functools.lru_cache(maxsize=1024)
def plan_mla_nvfp4_paged_decode(
    *,
    batch: int,
    max_q_len: int,
    num_heads: int,
    max_seq_len: int,
    sm_count: int,
    num_split: Optional[int] = None,
    rows: Optional[int] = None,
) -> DecodePlan:
    """The plan of a batch shape; cached, so a decode loop plans each shape once.

    ``rows`` is the number of packed (token, head) rows the query actually holds (the split partials and the
    merge grid are sized by it); it defaults to the dense ``batch * max_q_len * num_heads``.
    """
    if batch <= 0 or max_q_len <= 0 or num_heads <= 0:
        raise ValueError("batch, max_q_len and num_heads must be positive")
    rows_max = batch * max_q_len * num_heads if rows is None else int(rows)
    if not 0 < rows_max <= batch * max_q_len * num_heads:
        raise ValueError(
            f"rows must be in [1, batch * max_q_len * num_heads = {batch * max_q_len * num_heads}], got {rows_max}"
        )
    wide = use_wide_route(max_q_len * num_heads)
    if wide:
        rt, m_tiles = WIDE_BLOCK_M, wide_tiles(max_q_len * num_heads)
        ctas_per_item = WIDE_CLUSTER
    else:
        rt, m_tiles = rt_for_rows(max_q_len * num_heads)
        ctas_per_item = 1
    items = batch * m_tiles
    splits = (
        int(num_split)
        if num_split
        else plan_num_split(items, int(max_seq_len), sm_count // ctas_per_item)
    )
    if not 1 <= splits <= MAX_SPLITS:
        raise ValueError(f"num_split must be in [1, {MAX_SPLITS}], got {splits}")
    if splits >= REDUCE_CTA_MIN_SPLITS:
        reduce_kind, reduce_warps = "reduce_cta", 0
        grid_reduce = (rows_max, REDUCE_DIM_CHUNKS, 1)
    else:
        reduce_warps = reduce_warps_per_row(rows_max)
        reduce_kind = f"reduce_w{reduce_warps}"
        rows_per_cta = REDUCE_WARPS // reduce_warps
        grid_reduce = ((rows_max + rows_per_cta - 1) // rows_per_cta, 1, 1)
    return DecodePlan(
        rt=rt,
        m_tiles=m_tiles,
        num_split=splits,
        rows_max=rows_max,
        grid_main=(ctas_per_item * splits, m_tiles, batch),
        reduce_kind=reduce_kind,
        reduce_warps=reduce_warps,
        grid_reduce=grid_reduce,
        wide=wide,
    )


# ---------------------------------------------------------------------------
# Workspace, device facts and bindings
# ---------------------------------------------------------------------------


def _align(n: int) -> int:
    return (n + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT


def workspace_bytes(rows_max: int, num_split: int) -> int:
    """Bytes of ``workspace_buffer`` a plan needs: BF16 partial O (split plans), FP32
    partial max / sum and the one-word LSE placeholder of calls without LSE."""
    partial_o = rows_max * num_split * V_DIM * 2 if num_split > 1 else 0
    stats = rows_max * num_split * 4
    return _align(partial_o) + 2 * _align(stats) + _align(4)


def max_workspace_bytes(rows_max: int) -> int:
    """Upper bound of :func:`workspace_bytes` over every split count."""
    return workspace_bytes(rows_max, MAX_SPLITS)


def _carve_workspace(workspace: torch.Tensor, rows_max: int, num_split: int):
    if (
        workspace.device.type != "cuda"
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace_buffer must be a contiguous uint8 CUDA tensor")
    raw = workspace.reshape(-1)
    need = workspace_bytes(rows_max, num_split)
    if raw.numel() < need:
        raise ValueError(
            f"workspace_buffer needs at least {need} bytes for this Cake NVFP4 MLA decode plan, got {raw.numel()}"
        )
    off = 0
    partial_o = None
    if num_split > 1:
        n = rows_max * num_split * V_DIM * 2
        partial_o = (
            raw[off : off + n].view(torch.bfloat16).view(rows_max, num_split, V_DIM)
        )
        off += _align(n)
    n = rows_max * num_split * 4
    partial_max = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    off += _align(n)
    partial_sum = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    off += _align(n)
    lse_placeholder = raw[off : off + 4].view(torch.float32)
    return partial_o, partial_max, partial_sum, lse_placeholder


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(device_index: int) -> tuple[str, int]:
    """``(arch, sm_count)`` of a device, read once per process."""
    props = torch.cuda.get_device_properties(device_index)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get((props.major, props.minor))
    if arch is None:
        raise ValueError(
            "Cake NVFP4 MLA decode requires compute capability 10.0 or 10.3 "
            f"(got {props.major}.{props.minor})"
        )
    return arch, int(props.multi_processor_count)


_DENSE_Q_INDPTR: dict[tuple[int, int, int], torch.Tensor] = {}


def _dense_q_indptr(batch: int, q_len: int, device: torch.device) -> torch.Tensor:
    """``[0, q_len, 2 q_len, ...]`` for a dense ``[B, q_len, H, .]`` query, kept per shape and device."""
    key = (batch, q_len, _device_index(device))
    cached = _DENSE_Q_INDPTR.get(key)
    if cached is None:
        cached = torch.arange(
            0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device
        )
        # A tensor allocated during CUDA-Graph capture belongs to the graph's private pool: it
        # serves the captured launch but must not be reused by later eager calls.
        if not torch.cuda.is_current_stream_capturing():
            _DENSE_Q_INDPTR[key] = cached
    return cached


def _bind_stage(
    role: str, program: str, arch: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``role`` and load ``program``."""
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in ARG_PLANS[role]:
        if kind == "grid":
            arguments.append(int(grid[name]))
        elif name in kwargs:
            value = kwargs[name]
            if isinstance(value, torch.Tensor):
                # Exported through DLPack once: the launch hands the FFI a ready
                # ``tvm_ffi.Tensor`` (same storage) instead of converting per call.
                value = tvm_ffi.from_dlpack(value)
            arguments.append(value)
        else:
            raise KeyError(
                f"generated program {program!r} ({role}) expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_program(program, arch)
    return getattr(module, FFI_ENTRY), tuple(arguments)


def _check_cache(name: str, t: torch.Tensor, width: int, dtype) -> None:
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if t.ndim != 3 or int(t.shape[-1]) != width:
        raise ValueError(
            f"{name} must be [num_pages, page_size, {width}], got shape {tuple(t.shape)}"
        )
    if t.stride(-1) != 1 or t.stride(-2) % 16 or t.stride(-3) % 16 or t.data_ptr() % 16:
        raise ValueError(
            f"{name}: the last dim must be contiguous and the token / page strides and base 16-byte aligned"
        )


# ---------------------------------------------------------------------------
# Query quantizer
# ---------------------------------------------------------------------------


def query_scale_constants(ckv_scale: float, kpe_scale: float) -> tuple[float, float]:
    """``(c_nope, c_rope)`` of the quantizer in float32 arithmetic: ``1 / (6 * 448)`` and
    ``kpe_scale * (1 / (448 * ckv_scale))``."""
    f32 = torch.float32
    c_nope = torch.tensor(1.0 / (E2M1_MAX * E4M3_MAX), dtype=f32)
    c_rope = torch.tensor(kpe_scale, dtype=f32) * (
        torch.tensor(1.0, dtype=f32) / torch.tensor(E4M3_MAX * ckv_scale, dtype=f32)
    )
    return float(c_nope), float(c_rope)


def mla_nvfp4_query_buffers(
    lead: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Fresh ``(q_nope, q_sf, q_rope, q_scale)`` buffers for query rows of leading shape ``lead``."""
    return (
        torch.empty((*lead, CKV_BYTES), dtype=torch.uint8, device=device),
        torch.empty((*lead, SF_BYTES), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((*lead, ROPE), dtype=torch.float8_e4m3fn, device=device),
        torch.empty(lead, dtype=torch.float32, device=device),
    )


class CakeMlaNvfp4QueryQuantize:
    """Prepared BF16 -> NVFP4 query quantizer: ``launch`` allocates nothing.

    ``query`` is a contiguous BF16 tensor ``[.., 576]`` (512 latent + 64 rope channels per
    (token, head) row after the up-projection / absorption); ``out`` holds the four query
    operands of :class:`CakeMlaNvfp4PagedDecode` (``mla_nvfp4_query_buffers`` allocates them).
    Per row ``q_scale = max(amax(nope) / (6 * 448), amax(rope) * kpe_scale / (448 * ckv_scale))``
    (1 for an all-zero row), per 16-channel block ``sf = e4m3(amax_block / (6 q_scale))`` and
    ``codes = e2m1(x / (sf q_scale))``, rope ``e4m3(x * kpe_scale / (q_scale * ckv_scale))``.
    """

    def __init__(
        self,
        *,
        query: torch.Tensor,
        ckv_scale: float,
        kpe_scale: float,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        if query.device.type != "cuda":
            raise ValueError("query must be a CUDA tensor")
        if (
            query.dtype != torch.bfloat16
            or query.shape[-1] != QK_DIM
            or not query.is_contiguous()
        ):
            raise ValueError(
                f"query must be a contiguous bfloat16 tensor with last dim {QK_DIM}"
            )
        if not (ckv_scale > 0.0 and kpe_scale > 0.0):
            raise ValueError("ckv_scale and kpe_scale must be positive")
        q_nope, q_sf, q_rope, q_scale = out
        rows = query.numel() // QK_DIM
        lead = tuple(query.shape[:-1])
        for name, t, shape, dtype in (
            ("q_nope", q_nope, (*lead, CKV_BYTES), torch.uint8),
            ("q_sf", q_sf, (*lead, SF_BYTES), torch.float8_e4m3fn),
            ("q_rope", q_rope, (*lead, ROPE), torch.float8_e4m3fn),
            ("q_scale", q_scale, lead, torch.float32),
        ):
            if (
                tuple(t.shape) != shape
                or t.dtype != dtype
                or not t.is_contiguous()
                or t.device != query.device
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} tensor of shape {shape} on the query's device"
                )
        self.arch, _ = _device_facts(_device_index(query.device))
        self.rows = int(rows)
        self.out = out
        self.program = select_program("quantize", self.arch)
        c_nope, c_rope = query_scale_constants(float(ckv_scale), float(kpe_scale))
        kwargs = dict(
            q_bf16=query.reshape(-1, QK_DIM),
            q_nope=q_nope.view(torch.uint32).reshape(-1, CKV_BYTES // 4),
            q_sf=q_sf.view(torch.uint8).reshape(-1, SF_BYTES),
            q_rope=q_rope.view(torch.uint32).reshape(-1, ROPE // 4),
            q_scale=q_scale.reshape(-1),
            rows=self.rows,
            c_nope=c_nope,
            c_rope=c_rope,
            kpe_scale=float(kpe_scale),
            ckv_scale=float(ckv_scale),
            grid=(
                (self.rows + QUANTIZE_ROWS_PER_CTA - 1) // QUANTIZE_ROWS_PER_CTA,
                1,
                1,
            ),
        )
        assert tuple(kwargs) == QUANTIZE_KWARGS
        self._entry, self._arguments = _bind_stage(
            "quantize", self.program, self.arch, kwargs
        )

    def launch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize on the current stream; no allocation, no host synchronization."""
        with tvm_ffi.use_torch_stream():
            self._entry(*self._arguments)
        return self.out


def quantize_mla_nvfp4_query(
    query: torch.Tensor,
    ckv_scale: float,
    kpe_scale: float,
    *,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot query quantization: ``(q_nope, q_sf, q_rope, q_scale)`` of a BF16 query ``[.., 576]``."""
    if out is None:
        out = mla_nvfp4_query_buffers(tuple(query.shape[:-1]), query.device)
    return CakeMlaNvfp4QueryQuantize(
        query=query, ckv_scale=ckv_scale, kpe_scale=kpe_scale, out=out
    ).launch()


# ---------------------------------------------------------------------------
# Prepared decode launcher
# ---------------------------------------------------------------------------


class CakeMlaNvfp4PagedDecode:
    """Prepared launcher: validation and binding at construction, no allocation at ``launch``.

    Query operands are ``[B, q_len, H, .]`` or ``[total_q, H, .]`` (with ``cum_seq_lens_q`` /
    ``max_q_len``): ``q_nope`` uint8 ``[.., 256]``, ``q_sf`` float8_e4m3fn ``[.., 32]``,
    ``q_rope`` float8_e4m3fn ``[.., 64]``, ``q_scale`` float32 ``[..]`` (contiguous).  Cache:
    ``ckv_cache`` uint8 ``[pages, page_size, 256]``, ``ckv_sf_cache`` float8_e4m3fn
    ``[pages, page_size, 32]``, ``kpe_cache`` float8_e4m3fn ``[pages, page_size, 64]`` (strided
    views of one allocation are fine: token / page strides and the base must be 16-byte
    multiples).  ``page_size`` is a power of two >= 32.  ``out`` BF16 ``q.shape[:-1] + (512,)``;
    ``lse`` optional FP32 ``q.shape[:-1]`` (natural log).  ``seq_lens`` int32 ``[B]`` local KV
    lengths, ``block_tables`` int32 ``[B, max_pages_per_seq]``.  Variable-length / MTP queries
    pack the tokens of request ``b`` at rows ``cum_seq_lens_q[b] .. cum_seq_lens_q[b + 1]``,
    bottom-right causal against the request's last keys.  DCP: ``cp_world`` / ``cp_rank`` with
    ``kv_len_global`` int32 ``[B]`` (rank ``r`` holds the global positions ``cp_world * k + r``).
    ``workspace_buffer`` is a uint8 CUDA buffer of at least ``workspace_bytes(rows, num_split)``
    bytes (``max_workspace_bytes(rows)`` covers every plan of a row count).
    """

    def __init__(
        self,
        *,
        q_nope: torch.Tensor,
        q_sf: torch.Tensor,
        q_rope: torch.Tensor,
        q_scale: torch.Tensor,
        ckv_cache: torch.Tensor,
        ckv_sf_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sm_scale: float,
        ckv_scale: float,
        o_scale: float = 1.0,
        lse: Optional[torch.Tensor] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        kv_len_global: Optional[torch.Tensor] = None,
        num_split: Optional[int] = None,
    ):
        device = q_nope.device
        if device.type != "cuda":
            raise ValueError("query operands must be CUDA tensors")
        u8, e4m3 = torch.uint8, torch.float8_e4m3fn
        if (
            q_nope.dtype != u8
            or q_sf.dtype != e4m3
            or q_rope.dtype != e4m3
            or q_scale.dtype != torch.float32
        ):
            raise ValueError(
                "q_nope must be uint8, q_sf / q_rope float8_e4m3fn, q_scale float32"
            )
        if (
            q_nope.shape[-1] != CKV_BYTES
            or q_sf.shape[-1] != SF_BYTES
            or q_rope.shape[-1] != ROPE
        ):
            raise ValueError(
                f"query operand widths must be {CKV_BYTES} / {SF_BYTES} / {ROPE}"
            )
        _check_cache("ckv_cache", ckv_cache, CKV_BYTES, u8)
        _check_cache("ckv_sf_cache", ckv_sf_cache, SF_BYTES, e4m3)
        _check_cache("kpe_cache", kpe_cache, ROPE, e4m3)
        if (
            ckv_sf_cache.shape[:2] != ckv_cache.shape[:2]
            or kpe_cache.shape[:2] != ckv_cache.shape[:2]
        ):
            raise ValueError("cache tensors must share [num_pages, page_size]")
        page_size = int(ckv_cache.shape[1])
        if page_size < BOX_TOK or page_size & (page_size - 1):
            raise ValueError(
                f"page_size must be a power of two >= {BOX_TOK}, got {page_size}"
            )
        if q_nope.ndim == 4:
            batch, q_len, num_heads, _ = q_nope.shape
            if cum_seq_lens_q is None:
                cum_seq_lens_q = _dense_q_indptr(int(batch), int(q_len), device)
            max_q_len = int(q_len)
        elif q_nope.ndim == 3:
            if cum_seq_lens_q is None or max_q_len is None:
                raise ValueError(
                    "a packed [total_q, H, .] query needs cum_seq_lens_q and max_q_len"
                )
            batch = int(cum_seq_lens_q.shape[0]) - 1
            num_heads = int(q_nope.shape[1])
        else:
            raise ValueError("query operands must be 3D or 4D")
        lead = tuple(q_nope.shape[:-1])
        for name, t, shape in (
            ("q_sf", q_sf, (*lead, SF_BYTES)),
            ("q_rope", q_rope, (*lead, ROPE)),
            ("q_scale", q_scale, lead),
            ("out", out, (*lead, V_DIM)),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(t.shape)}"
                )
        for name, t in (
            ("q_nope", q_nope),
            ("q_sf", q_sf),
            ("q_rope", q_rope),
            ("q_scale", q_scale),
            ("out", out),
            ("block_tables", block_tables),
        ):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be bfloat16")
        for name, t, numel in (
            ("seq_lens", seq_lens, batch),
            ("cum_seq_lens_q", cum_seq_lens_q, batch + 1),
        ):
            if (
                t.dtype != torch.int32
                or not t.is_contiguous()
                or int(t.numel()) != numel
            ):
                raise ValueError(
                    f"{name} must be a contiguous int32 tensor with {numel} entries"
                )
        if (
            block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or int(block_tables.shape[0]) != batch
        ):
            raise ValueError(
                "block_tables must be an int32 [batch, max_pages_per_seq] tensor"
            )
        if lse is not None and (
            lse.dtype != torch.float32
            or tuple(lse.shape) != lead
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be a contiguous float32 tensor of shape query.shape[:-1]"
            )
        if (
            type(cp_world) is not int
            or type(cp_rank) is not int
            or cp_world < 1
            or not 0 <= cp_rank < cp_world
        ):
            raise ValueError("need int cp_world >= 1 and 0 <= cp_rank < cp_world")
        if cp_world > 1:
            if kv_len_global is None:
                raise ValueError("DCP (cp_world > 1) needs kv_len_global")
            if (
                kv_len_global.dtype != torch.int32
                or not kv_len_global.is_contiguous()
                or int(kv_len_global.numel()) != batch
            ):
                raise ValueError(
                    "kv_len_global must be a contiguous int32 tensor with batch entries"
                )
        elif kv_len_global is not None:
            raise ValueError("kv_len_global requires cp_world > 1")
        if not (sm_scale > 0.0 and ckv_scale > 0.0):
            raise ValueError("sm_scale and ckv_scale must be positive")
        tensors = [
            q_nope,
            q_sf,
            q_rope,
            q_scale,
            ckv_cache,
            ckv_sf_cache,
            kpe_cache,
            block_tables,
            seq_lens,
            out,
            workspace_buffer,
        ]
        tensors += [t for t in (cum_seq_lens_q, lse, kv_len_global) if t is not None]
        if not all(t.device == device for t in tensors):
            raise ValueError("Expected all tensors on one CUDA device")
        if max_seq_len is None:
            max_seq_len = int(block_tables.shape[-1]) * page_size
        self.arch, sm_count = _device_facts(_device_index(device))
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.max_q_len = int(max_q_len)
        self.page_size = page_size
        rows = q_nope.numel() // CKV_BYTES
        if rows > self.batch * self.max_q_len * self.num_heads:
            raise ValueError(
                f"query holds {rows} rows, more than batch * max_q_len * num_heads = "
                f"{self.batch * self.max_q_len * self.num_heads}"
            )
        self.plan = plan_mla_nvfp4_paged_decode(
            batch=self.batch,
            max_q_len=self.max_q_len,
            num_heads=self.num_heads,
            max_seq_len=int(max_seq_len),
            sm_count=sm_count,
            num_split=None if num_split is None else int(num_split),
            rows=rows,
        )
        plan = self.plan
        self.rt = plan.rt
        self.num_split = plan.num_split
        self.rows_max = plan.rows_max
        self.max_pages_per_seq = int(block_tables.shape[-1])
        self.softmax_scale_log2 = float(sm_scale) * float(ckv_scale) * LOG2E
        self.bmm2_scale = float(ckv_scale) * float(o_scale)
        self.out = out
        self.lse = lse
        self.o_rows = out.view(-1, V_DIM)
        self.partial_O, self.partial_max, self.partial_sum, lse_placeholder = (
            _carve_workspace(workspace_buffer, plan.rows_max, plan.num_split)
        )
        lse_rows = lse.view(-1) if lse is not None else lse_placeholder
        has_lse = int(lse is not None)
        self.main_program = select_program(plan.main_kind, self.arch)
        self.reduce_program = (
            select_program(plan.reduce_kind, self.arch) if plan.num_split > 1 else None
        )
        main_kwargs = dict(
            tmap_qn=q_nope.reshape(-1, CKV_BYTES),
            tmap_qs=q_sf.view(u8).reshape(-1, SF_BYTES),
            tmap_qr=q_rope.view(u8).reshape(-1, ROPE),
            tmap_k=ckv_cache,
            tmap_ks=ckv_sf_cache.view(u8),
            tmap_kr=kpe_cache.view(u8),
            q_scale=q_scale.reshape(-1),
            partial_O=self.o_rows if plan.num_split == 1 else self.partial_O,
            partial_max=self.partial_max,
            partial_sum=self.partial_sum,
            lse=lse_rows,
            seq_lens=seq_lens,
            kv_len_global=kv_len_global if kv_len_global is not None else seq_lens,
            cum_seq_lens_q=cum_seq_lens_q,
            page_table=block_tables.reshape(-1),
            softmax_scale_log2=self.softmax_scale_log2,
            bmm2_scale=self.bmm2_scale,
            num_heads=self.num_heads,
            num_split=plan.num_split,
            max_pages_per_seq=self.max_pages_per_seq,
            page_shift=page_size.bit_length() - 1,
            cp_world=int(cp_world),
            cp_rank=int(cp_rank),
            has_lse=has_lse,
            grid=plan.grid_main,
        )
        assert tuple(main_kwargs) == MAIN_KWARGS
        self._main_entry, self._main_arguments = _bind_stage(
            "main", self.main_program, self.arch, main_kwargs
        )
        self._reduce_entry: Optional[Callable[..., Any]] = None
        self._reduce_arguments: tuple = ()
        if self.reduce_program is not None:
            reduce_kwargs = dict(
                partial_O=self.partial_O,
                partial_max=self.partial_max,
                partial_sum=self.partial_sum,
                O=self.o_rows,
                lse=lse_rows,
                cum_seq_lens_q=cum_seq_lens_q,
                batch=self.batch,
                num_heads=self.num_heads,
                num_split=plan.num_split,
                bmm2_scale=self.bmm2_scale,
                lse_bias=plan.lse_bias,
                has_lse=has_lse,
                grid=plan.grid_reduce,
            )
            assert tuple(reduce_kwargs) == REDUCE_KWARGS
            self._reduce_entry, self._reduce_arguments = _bind_stage(
                "reduce", self.reduce_program, self.arch, reduce_kwargs
            )
        self.route_metadata = dict(
            backend="cake",
            arch=self.arch,
            rt=plan.rt,
            m_tiles=plan.m_tiles,
            num_split=plan.num_split,
            reducer=plan.reduce_kind if plan.num_split > 1 else None,
            main_program=self.main_program,
            reduce_program=self.reduce_program,
            grid_main=plan.grid_main,
            grid_reduce=plan.grid_reduce if plan.num_split > 1 else None,
        )

    def launch(self) -> torch.Tensor:
        """Enqueue the attention (and the split merge) on the current stream; no allocation."""
        # The attention kernel is the first launch of the call (ordinary launch); it signals
        # griddepcontrol.launch_dependents when a CTA is done and the merge, launched with the
        # programmatic-dependent-launch attribute, waits on griddepcontrol.wait before its
        # first partial read.
        with tvm_ffi.use_torch_stream():
            self._main_entry(*self._main_arguments)
            if self._reduce_entry is not None:
                self._reduce_entry(*self._reduce_arguments)
        return self.out

    __call__ = launch


def cake_mla_nvfp4_paged_decode(
    q_nope: torch.Tensor,
    q_sf: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    ckv_scale: float,
    o_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: Optional[torch.Tensor] = None,
    backend: str = "cake",
):
    """One-shot dense NVFP4 MLA decode (prepare + one launch).

    Allocates ``out`` (and ``lse`` with ``return_lse=True``) when not given; the plan is
    cached per batch shape and the device facts per device.  Returns ``out`` or
    ``(out, lse)``.  Use :class:`CakeMlaNvfp4PagedDecode` for a launch-only runner.
    """
    if backend != "cake":
        raise ValueError("Cake NVFP4 MLA decode supports backend='cake'")
    lead = tuple(q_nope.shape[:-1])
    if out is None:
        out = torch.empty((*lead, V_DIM), dtype=torch.bfloat16, device=q_nope.device)
    if return_lse and lse is None:
        lse = torch.empty(lead, dtype=torch.float32, device=q_nope.device)
    runner = CakeMlaNvfp4PagedDecode(
        q_nope=q_nope,
        q_sf=q_sf,
        q_rope=q_rope,
        q_scale=q_scale,
        ckv_cache=ckv_cache,
        ckv_sf_cache=ckv_sf_cache,
        kpe_cache=kpe_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        sm_scale=sm_scale,
        ckv_scale=ckv_scale,
        o_scale=o_scale,
        lse=lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
    )
    runner.launch()
    return (out, lse) if return_lse else out


__all__ = [
    "CKV_BYTES",
    "CakeMlaNvfp4PagedDecode",
    "CakeMlaNvfp4QueryQuantize",
    "DecodePlan",
    "LSE_BIAS",
    "MAIN_KWARGS",
    "MAX_SPLITS",
    "QK_DIM",
    "QUANTIZE_KWARGS",
    "REDUCE_KWARGS",
    "ROPE",
    "ROW_TILES",
    "SF_BYTES",
    "WIDE_BLOCK_M",
    "WIDE_CLUSTER",
    "WIDE_LSE_BIAS",
    "WIDE_MIN_ROWS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "V_DIM",
    "cake_mla_nvfp4_paged_decode",
    "max_workspace_bytes",
    "mla_nvfp4_query_buffers",
    "plan_mla_nvfp4_paged_decode",
    "plan_num_split",
    "quantize_mla_nvfp4_query",
    "query_scale_constants",
    "reduce_warps_per_row",
    "rt_for_rows",
    "use_wide_route",
    "wide_tiles",
    "workspace_bytes",
]
