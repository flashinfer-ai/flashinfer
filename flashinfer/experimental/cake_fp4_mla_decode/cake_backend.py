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

Host side of the Cake MLA decode over TensorRT-LLM's ``fp4_mla`` KV cache.

:func:`plan_fp4_mla_decode` fixes one decode bucket (batch, query tokens per
request, heads, page-table width, optionally a bound on the KV lengths) on one
device, chooses the KV split count, the decode build and the reducer build,
and allocates the split partials once; the returned
:class:`Fp4MlaDecodeRunner` reads TensorRT-LLM's own tensors in place on every
call (no copy of the cache, the query or its scales) and writes the
caller-owned BF16 output.  A call allocates no CUDA memory and reads no
device value on the host, so one runner serves every CUDA-graph replay of
its bucket while ``kv_lens`` changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tvm_ffi

from .cake_jit import (
    ARCHES,
    LATENT,
    MODULES,
    TILE_ROWS,
    device_arch,
    launch_grid,
    load,
    record_for,
)
from .cake_jit import sm_count as _sm_count

PAGE_TOKENS = 128
TOKEN_BYTES = 320  # 640 E2M1 channels: NoPE 512, RoPE base 64, RoPE residual 64
KV_PAGE_BYTES = PAGE_TOKENS * TOKEN_BYTES  # 40,960
SF_PAGE_BYTES = PAGE_TOKENS * 40  # 5,120: 40 E4M3 scales per token
V_SCALE_PAGE_BYTES = 4096  # the V-scale pool: 512 latent dims x 8 blocks of 16 tokens
Q_SF_BLOCK_BYTES = 128 * 40  # scales of 128 query rows
MAX_HEADS = 128
MAX_SPLITS = 16
# The split rule's cost model, in pages of one CTA pair's time: ``s`` splits
# cost ``ceil(items s / pairs) (ceil(width / s) + ITEM_PAGES)`` plus, for
# ``s > 1``, ``rows MERGE_ROW_PAGES`` for the reducer; the plan takes the
# fewest splits within ``SPLIT_SLACK`` of the least cost, at least two pages
# per split.  One exception: when that count takes several waves but every
# item fits one wave at some count, the plan takes the cheapest one-wave count
# if, priced with ``ITEM_PAGES_ONE_WAVE`` per item, it is within
# ``SPLIT_SLACK_ONE_WAVE`` of the least cost so priced (waves that keep the
# whole device busy run each page slower than one partly filled wave).
ITEM_PAGES = 6.2
MERGE_ROW_PAGES = 0.0002
SPLIT_SLACK = 0.03
ITEM_PAGES_ONE_WAVE = 9.0
SPLIT_SLACK_ONE_WAVE = 0.01
MIN_PAGES_PER_SPLIT = 2
# The decode dithers the P scales of near-flat key blocks in rows that see at
# least DITHER_KEYS keys; a bucket none of whose rows can see that many runs
# the decode build without that branch (``long_rows=False``), which gives the
# same result on such rows.
DITHER_KEYS = 32768
# Warps per CTA of the split reducer (256 threads).
MERGE_WARPS = 8

_U8 = (torch.uint8, torch.float8_e4m3fn)


def num_split_for(
    *,
    pairs: int,
    batch: int,
    m_tiles: int,
    rows: int,
    page_table_width: int,
    max_splits: int = MAX_SPLITS,
) -> int:
    """KV splits per row tile for ``pairs`` two-CTA clusters (``sm_count //
    2``), ``batch * m_tiles`` work items, ``rows`` output rows and a page
    table ``page_table_width`` pages wide (the module's cost model, with its
    one-wave exception)."""
    items = batch * m_tiles

    def cost(splits: int, item_pages: float = ITEM_PAGES) -> float:
        waves = -(-items * splits // max(1, pairs))
        merge = rows * MERGE_ROW_PAGES if splits > 1 else 0.0
        return waves * (-(-page_table_width // splits) + item_pages) + merge

    counts = range(
        1, max(1, min(max_splits, page_table_width // MIN_PAGES_PER_SPLIT)) + 1
    )
    best = min(cost(splits) for splits in counts)
    chosen = next(s for s in counts if cost(s) <= best * (1.0 + SPLIT_SLACK))
    one_wave = [s for s in counts if items * s <= pairs]
    if not one_wave or chosen in one_wave:
        return chosen
    best_one = min(cost(s, ITEM_PAGES_ONE_WAVE) for s in counts)
    cheapest = min(one_wave, key=lambda s: cost(s, ITEM_PAGES_ONE_WAVE))
    if cost(cheapest, ITEM_PAGES_ONE_WAVE) <= best_one * (1.0 + SPLIT_SLACK_ONE_WAVE):
        return cheapest
    return chosen


def merge_warps_per_row(rows: int) -> int:
    """The split reducer's warps per output row: 4 up to 256 rows, 2 up to
    1,024, else 1 (more warps per row for few rows, so the merge still spans
    the device)."""
    return 4 if rows <= 256 else 2 if rows <= 1024 else 1


def long_rows_for(*, page_table_width: int, max_kv: int | None = None) -> bool:
    """Whether a row of the bucket can see ``DITHER_KEYS`` keys: its keys are at
    most ``page_table_width * PAGE_TOKENS``, and at most ``max_kv`` when the
    caller bounds every ``kv_lens`` value of the bucket."""
    bound = page_table_width * PAGE_TOKENS
    if max_kv is not None:
        bound = min(bound, max_kv)
    return bound >= DITHER_KEYS


@dataclass(frozen=True)
class Fp4MlaDecodePlan:
    """The static launch of one bucket."""

    batch: int
    q_len: int
    num_heads: int
    page_table_width: int
    sm_count: int
    num_split: int
    long_rows: bool = True

    @property
    def rows_per_request(self) -> int:
        return self.q_len * self.num_heads

    @property
    def rows(self) -> int:
        return self.batch * self.rows_per_request

    @property
    def m_tiles(self) -> int:
        return -(-self.rows_per_request // TILE_ROWS)

    @property
    def merge_warps_per_row(self) -> int:
        return merge_warps_per_row(self.rows)


def _plan(batch, q_len, num_heads, page_table_width, sm_count, max_splits, max_kv):
    for name, value in (
        ("batch", batch),
        ("q_len", q_len),
        ("num_heads", num_heads),
        ("page_table_width", page_table_width),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if not num_heads <= MAX_HEADS:
        raise ValueError(
            f"num_heads {num_heads}: the decode serves 1..{MAX_HEADS} heads"
        )
    if not 1 <= max_splits <= MAX_SPLITS:
        raise ValueError(f"max_splits {max_splits} outside 1..{MAX_SPLITS}")
    if max_kv is not None and max_kv < 0:
        raise ValueError(f"max_kv {max_kv}: a bound on kv_lens, >= 0")
    m_tiles = -(-(q_len * num_heads) // TILE_ROWS)
    splits = num_split_for(
        pairs=sm_count // 2,
        batch=batch,
        m_tiles=m_tiles,
        rows=batch * q_len * num_heads,
        page_table_width=page_table_width,
        max_splits=max_splits,
    )
    return Fp4MlaDecodePlan(
        batch,
        q_len,
        num_heads,
        page_table_width,
        sm_count,
        splits,
        long_rows_for(page_table_width=page_table_width, max_kv=max_kv),
    )


def _page_view(pool, page_bytes, name):
    """``pool`` as ``[pages, page_bytes]`` bytes in place: the first dimension
    is the page id, the rest one contiguous page; returns (view, page stride in
    bytes).  The checks read the shape and strides only (no tensor op)."""
    if pool.dtype not in _U8:
        raise TypeError(f"{name}: dtype {pool.dtype}, expected uint8 bytes")
    shape, strides = pool.shape, pool.stride()
    size = 1
    for extent in shape[1:]:
        size *= extent
    if len(shape) < 2 or size != page_bytes:
        raise ValueError(
            f"{name}: shape {tuple(shape)} is not [pages, ...] of "
            f"{page_bytes}-byte pages"
        )
    expected = 1
    for axis in range(len(shape) - 1, 0, -1):
        if shape[axis] != 1 and strides[axis] != expected:
            raise ValueError(f"{name}: a page is not contiguous ({tuple(strides)})")
        expected *= shape[axis]
    stride = strides[0]
    if stride < page_bytes:
        raise ValueError(f"{name}: page stride {stride} B overlaps the pages")
    return pool.view(torch.uint8), stride


def _arg_slots(record, static, grid):
    """The ``run`` arguments of ``record`` (its ``arg_plan``, as
    ``cake_jit.launch_args`` orders them) with every value fixed by the plan
    filled in, and ``(position, logical name)`` of the values each call
    passes."""
    logical = record["logical"]
    args, per_call = [], []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid["xyz".index(key[-1])]))
            continue
        if key not in logical:
            raise ValueError(f"kernel parameter {key!r} has no logical name")
        name = logical[key]
        if name not in static and name not in _PER_CALL:
            raise ValueError(f"the launch needs logical argument {name!r}")
        if name in static:
            args.append(static[name])
        else:
            per_call.append((len(args), name))
            args.append(None)
    return args, tuple(per_call)


# The logical arguments each call passes (the rest are the plan's).
_PER_CALL = frozenset(
    (
        "kv_k",
        "kv_v",
        "kv_sf",
        "v_pool",
        "q",
        "q_sf",
        "page_table",
        "kv_lens",
        "q_gs",
        "kv_gs",
        "out",
        "sm_scale",
    )
)


def _raw_stream(index):
    """The handle of torch's current stream of device ``index`` (a CUDA
    graph's capture stream during capture)."""
    raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw is not None:
        return raw(index)
    return torch.cuda.current_stream(index).cuda_stream


class Fp4MlaDecodeRunner:
    """The decode of one bucket (:func:`plan_fp4_mla_decode`).

    ``runner(...)`` decodes one step: TensorRT-LLM's per-layer ``fp4_mla``
    cache views, the step's prequantized query, the page table and the KV
    lengths in, the caller-owned BF16 ``out`` written.  The split partials
    belong to the runner, so two runners never share them.
    """

    def __init__(self, plan: Fp4MlaDecodePlan, device: torch.device):
        self.plan = plan
        self.device = device
        self.device_index = device.index
        self.arch = device_arch(device.index)
        self.decode = record_for("decode", self.arch, long_rows=plan.long_rows)
        self.merge = record_for(
            "merge", self.arch, warps_per_row=plan.merge_warps_per_row
        )
        p = plan
        s = p.num_split
        # One split writes ``out`` directly and leaves the partials alone;
        # they keep the plan's shape either way, as the kernel's tests launch it.
        rows = p.rows
        self.partial_O = torch.empty(
            (rows, s, LATENT), dtype=torch.bfloat16, device=device
        )
        self.partial_max = torch.empty((rows, s), dtype=torch.float32, device=device)
        self.partial_sum = torch.empty((rows, s), dtype=torch.float32, device=device)
        self.cum_seq_lens_q = (
            torch.arange(p.batch + 1, dtype=torch.int32, device=device) * p.q_len
        )
        self.decode_grid = launch_grid(
            MODULES[self.decode],
            batch=p.batch,
            rows_per_request=p.rows_per_request,
            num_split=s,
        )
        self.merge_grid = launch_grid(
            MODULES[self.merge],
            batch=p.batch,
            rows_per_request=p.rows_per_request,
            num_split=s,
        )
        # What every call launches with but the per-call tensors: the plan's
        # scalars, the partials and the grids, in each entry's argument order;
        # the runner's own tensors as FFI tensors once (no per-call conversion).
        static = {
            "partial_O": tvm_ffi.from_dlpack(self.partial_O),
            "partial_max": tvm_ffi.from_dlpack(self.partial_max),
            "partial_sum": tvm_ffi.from_dlpack(self.partial_sum),
            "num_heads": p.num_heads,
            "q_len": p.q_len,
            "num_split": s,
            "page_stride": p.page_table_width,
            "page_offset": 0,
            "batch": p.batch,
            "cum_seq_lens_q": tvm_ffi.from_dlpack(self.cum_seq_lens_q),
            "bmm2_scale": 1.0,
            "merge_m_tiles": 1,
            "n_full_items": 0,
            "merge_tile_rows": TILE_ROWS,
        }
        self._decode_args = _arg_slots(MODULES[self.decode], static, self.decode_grid)
        self._merge_args = _arg_slots(MODULES[self.merge], static, self.merge_grid)
        self._ffi_device = tvm_ffi.device(f"cuda:{self.device_index}")
        self._entries = None

    def _entry_points(self):
        """The ``run`` entries of the decode and the reducer (loaded, or
        JIT-built, on the first call)."""
        if self._entries is None:
            self._entries = tuple(
                getattr(load(name, self.arch), MODULES[name]["ffi_entry"])
                for name in (self.decode, self.merge)
            )
        return self._entries

    def __call__(
        self,
        kv_cache: torch.Tensor,
        kv_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        q_fp4: torch.Tensor,
        q_scale: torch.Tensor,
        page_table: torch.Tensor,
        kv_lens: torch.Tensor,
        q_global_scale: torch.Tensor,
        kv_global_scale: torch.Tensor,
        out: torch.Tensor,
        *,
        sm_scale: float,
    ) -> torch.Tensor:
        """Decode into ``out`` and return it.

        ``kv_cache``: the layer's KV pages, uint8 ``[pages, ...]`` of 40,960
        contiguous bytes per page (TensorRT-LLM's per-layer view; any page
        stride that is a multiple of 16 bytes).  ``kv_scale_cache``: the
        layer's K scales, E4M3 ``[pages, ...]`` of 5,120 contiguous bytes per
        page at a page stride of 5,120 bytes.  ``v_scale_cache``: the layer's
        V scales, TensorRT-LLM's MLA V-scale pool (``v_pool``: per latent dim,
        the K scale of the first token of each 16-token block), E4M3 ``[pages,
        ...]`` of 4,096 contiguous bytes per page at a page stride of 4,096
        bytes, addressed by the page ids of ``kv_cache``; the decode reads its
        V scales from this pool (it does not derive them from
        ``kv_scale_cache``), and refuses any other layout.  ``q_fp4``: uint8
        ``[rows_cap, 320]`` contiguous, rows ``(b q_len + t) H + h``;
        ``q_scale``: its swizzled E4M3 scales, contiguous, at least
        ``ceil(rows / 128) * 5120`` bytes.  ``page_table``: int32 ``[B, W]``
        contiguous (``W`` the plan's width); ``kv_lens``: int32 ``[>= B]``
        contiguous, each request's KV length including this step's tokens
        (at most the plan's ``max_kv`` when it was given one: the decode build
        without long rows does not dither longer rows).
        ``q_global_scale`` / ``kv_global_scale``: FP32 one-element device
        tensors.  ``out``: BF16, ``B q_len H x 512`` contiguous elements.
        """
        p = self.plan
        kv, kv_page_stride = _page_view(kv_cache, KV_PAGE_BYTES, "kv_cache")
        if kv_page_stride % 16:
            raise ValueError(
                f"kv_cache: page stride {kv_page_stride} B is not a multiple of 16"
            )
        sf, sf_page_stride = _page_view(kv_scale_cache, SF_PAGE_BYTES, "kv_scale_cache")
        if sf_page_stride != SF_PAGE_BYTES:
            raise ValueError(
                f"kv_scale_cache: page stride {sf_page_stride} B, expected {SF_PAGE_BYTES}"
            )
        vsf, vsf_page_stride = _page_view(
            v_scale_cache, V_SCALE_PAGE_BYTES, "v_scale_cache"
        )
        if vsf_page_stride != V_SCALE_PAGE_BYTES:
            raise ValueError(
                f"v_scale_cache: page stride {vsf_page_stride} B; the decode reads "
                f"TensorRT-LLM's MLA V-scale pool as {V_SCALE_PAGE_BYTES}-byte pages "
                f"at a {V_SCALE_PAGE_BYTES}-byte stride, addressed by kv_cache's "
                "page ids, and takes no other layout"
            )
        pages = kv.shape[0]
        if sf.shape[0] < pages:
            raise ValueError(
                f"kv_scale_cache holds {sf.shape[0]} pages, fewer than kv_cache's {pages}"
            )
        if vsf.shape[0] < pages:
            raise ValueError(
                f"v_scale_cache holds {vsf.shape[0]} pages, fewer than kv_cache's "
                f"{pages}: the decode reads the V-scale page of every kv_cache page id"
            )
        if q_fp4.dtype not in _U8 or q_fp4.dim() != 2 or q_fp4.shape[1] != TOKEN_BYTES:
            raise ValueError(
                f"q_fp4: uint8 [rows, {TOKEN_BYTES}], got {q_fp4.dtype} {tuple(q_fp4.shape)}"
            )
        if not q_fp4.is_contiguous() or q_fp4.shape[0] < p.rows:
            raise ValueError(f"q_fp4: contiguous with at least {p.rows} rows")
        need = -(-p.rows // 128) * Q_SF_BLOCK_BYTES
        if (
            q_scale.dtype not in _U8
            or not q_scale.is_contiguous()
            or q_scale.numel() < need
        ):
            raise ValueError(f"q_scale: contiguous E4M3 bytes, at least {need}")
        if (
            page_table.dtype != torch.int32
            or tuple(page_table.shape) != (p.batch, p.page_table_width)
            or not page_table.is_contiguous()
        ):
            raise ValueError(
                f"page_table: contiguous int32 [{p.batch}, {p.page_table_width}], got "
                f"{page_table.dtype} {tuple(page_table.shape)}"
            )
        if (
            kv_lens.dtype != torch.int32
            or kv_lens.dim() != 1
            or kv_lens.shape[0] < p.batch
            or not kv_lens.is_contiguous()
        ):
            raise ValueError(f"kv_lens: contiguous int32 [>= {p.batch}]")
        for name, scale in (
            ("q_global_scale", q_global_scale),
            ("kv_global_scale", kv_global_scale),
        ):
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(f"{name}: an FP32 one-element device tensor")
        if (
            out.dtype != torch.bfloat16
            or out.numel() != p.rows * LATENT
            or not out.is_contiguous()
        ):
            raise ValueError(f"out: contiguous BF16 with {p.rows} x {LATENT} elements")
        index = self.device_index
        for t in (
            kv,
            sf,
            vsf,
            q_fp4,
            q_scale,
            page_table,
            kv_lens,
            q_global_scale,
            kv_global_scale,
            out,
        ):
            if t.get_device() != index:
                raise ValueError(f"every tensor must be on {self.device}")
        call = {
            "kv_k": torch.as_strided(
                kv, (pages, PAGE_TOKENS, TOKEN_BYTES), (kv_page_stride, TOKEN_BYTES, 1)
            ),
            "kv_sf": sf.view(-1).view(torch.uint32),
            "v_pool": vsf.view(-1).view(torch.uint32),
            "q": q_fp4.view(torch.uint8)[: p.rows].view(torch.uint32),
            "q_sf": q_scale.view(torch.uint8).view(-1)[:need].view(torch.uint32),
            "page_table": page_table,
            "kv_lens": kv_lens[: p.batch],
            "q_gs": q_global_scale.view(1),
            "kv_gs": kv_global_scale.view(1),
            "out": out.view(p.rows, LATENT),
            "sm_scale": float(sm_scale),
        }
        call["kv_v"] = call["kv_k"]
        decode, merge = self._entry_points()
        decode_args, decode_slots = self._decode_args
        decode_args = list(decode_args)
        for position, name in decode_slots:
            decode_args[position] = call[name]
        if p.num_split > 1:
            merge_args, merge_slots = self._merge_args
            merge_args = list(merge_args)
            for position, name in merge_slots:
                merge_args[position] = call[name]
        # One stream lookup per call: both launches go to torch's current
        # stream of the runner's device.
        with tvm_ffi.use_raw_stream(self._ffi_device, _raw_stream(index)):
            decode(*decode_args)
            if p.num_split > 1:
                merge(*merge_args)
        return out


def plan_fp4_mla_decode(
    *,
    batch: int,
    q_len: int,
    num_heads: int,
    page_table_width: int,
    device: torch.device | str | int | None = None,
    max_splits: int = MAX_SPLITS,
    max_kv: int | None = None,
    backend: str = "cake",
) -> Fp4MlaDecodeRunner:
    """The runner of one decode bucket on ``device``: ``batch`` requests of
    ``q_len`` query tokens and ``num_heads`` heads (at most 128) over a page
    table ``page_table_width`` pages wide.  The split count depends on the
    bucket and the device's SM count only, never on ``kv_lens``.  ``max_kv``,
    when given, bounds every ``kv_lens`` value of every call of the runner
    (for example the deployment's maximum sequence length); with the page
    table's width it picks the decode build, never the split count."""
    if backend != "cake":
        raise ValueError(
            f"the fp4_mla cache decode is served by backend='cake', got {backend!r}"
        )
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if dev.type != "cuda":
        raise ValueError(f"the decode runs on a CUDA device, got {dev}")
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    dev = torch.device("cuda", index)
    if device_arch(index) not in ARCHES:
        raise RuntimeError(f"no Cake fp4_mla decode program for {dev}")
    plan = _plan(
        batch,
        q_len,
        num_heads,
        page_table_width,
        _sm_count(index),
        max_splits,
        max_kv,
    )
    return Fp4MlaDecodeRunner(plan, dev)


__all__ = [
    "Fp4MlaDecodePlan",
    "Fp4MlaDecodeRunner",
    "long_rows_for",
    "merge_warps_per_row",
    "num_split_for",
    "plan_fp4_mla_decode",
]
