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
device, chooses the KV split count, the decode build (with or without long
rows, per item or persistent) and the reducer build, and allocates the split
partials once; the returned
:class:`Fp4MlaDecodeRunner` reads TensorRT-LLM's own tensors in place on every
call (no copy of the cache, the query or its scales) and writes the
caller-owned BF16 output.  A call allocates no CUDA memory and reads no
device value on the host, so one runner serves every CUDA-graph replay of
its bucket while ``kv_lens`` changes.  The runner keeps the FFI views it
derives from the caller's tensors, keyed by each tensor's address, dtype,
shape, strides and device, so a call with the tensors of an earlier call
repeats none of the layout checks and conversions.
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
# The persistent decode build runs at most one cluster per SM pair, each
# looping over the work items (the same bits as one cluster per item); it
# overlaps an item's loads with the previous item's last pages and epilogue.
# sm_100a takes it for every bucket; on sm_103a, whose persistent loop runs
# each page some 6 % slower, only a bucket without long rows whose items
# take several waves and have at most PERSISTENT_PAGES_SM103 pages each.
PERSISTENT_PAGES_SM103 = 32
# Views of the caller's tensors a runner keeps (their FFI tensors).
VIEW_CACHE_ENTRIES = 256

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


def persistent_for(
    *, arch: str | None, items: int, pairs: int, pages_per_item: int, long_rows: bool
) -> bool:
    """Whether the bucket takes the persistent decode build: every bucket on
    ``sm_100a``; on ``sm_103a`` a bucket without long rows whose ``items``
    (splits included) exceed the device's ``pairs`` and whose items have at
    most ``PERSISTENT_PAGES_SM103`` pages; none on another architecture.  The
    choice changes neither the split count nor the arithmetic."""
    if arch == "sm_100a":
        return True
    if arch == "sm_103a":
        return (
            not long_rows and items > pairs and pages_per_item <= PERSISTENT_PAGES_SM103
        )
    return False


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
    persistent: bool = False

    @property
    def pairs(self) -> int:
        return self.sm_count // 2

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


def _plan(
    batch, q_len, num_heads, page_table_width, sm_count, max_splits, max_kv, arch
):
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
    long_rows = long_rows_for(page_table_width=page_table_width, max_kv=max_kv)
    persistent = persistent_for(
        arch=arch,
        items=splits * m_tiles * batch,
        pairs=sm_count // 2,
        pages_per_item=-(-page_table_width // splits),
        long_rows=long_rows,
    )
    return Fp4MlaDecodePlan(
        batch,
        q_len,
        num_heads,
        page_table_width,
        sm_count,
        splits,
        long_rows,
        persistent,
    )


def _page_view(pool, page_bytes, name):
    """``pool`` as ``[pages, page_bytes]`` bytes in place: the first dimension
    is the page id, the rest one contiguous page; returns (page count, page
    stride in bytes).  The checks read the shape and strides only."""
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
    return shape[0], stride


def _arg_slots(record, static, grid):
    """The ``run`` arguments of ``record`` (its ``arg_plan``, as
    ``cake_jit.launch_args`` orders them) with every value fixed by the plan
    filled in, and ``(position, index into _PER_CALL)`` of the values each
    call passes."""
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
            per_call.append((len(args), _PER_CALL.index(name)))
            args.append(None)
    return tuple(args), tuple(per_call)


# The logical arguments each call passes (the rest are the plan's), in the
# order of the values ``Fp4MlaDecodeRunner.__call__`` gathers.
_PER_CALL = (
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
        self.decode = record_for(
            "decode",
            self.arch,
            long_rows=plan.long_rows,
            persistent=plan.persistent,
        )
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
            pairs=p.pairs,
        )
        self.merge_grid = launch_grid(
            MODULES[self.merge],
            batch=p.batch,
            rows_per_request=p.rows_per_request,
            num_split=s,
            pairs=p.pairs,
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
        # (role, address, dtype, shape, strides, device) -> what the role's
        # checks derived from that tensor (its FFI view and page geometry).
        self._views: dict[tuple, object] = {}

    def _entry_points(self):
        """The ``run`` entries of the decode and the reducer (loaded, or
        JIT-built, on the first call)."""
        if self._entries is None:
            self._entries = tuple(
                getattr(load(name, self.arch), MODULES[name]["ffi_entry"])
                for name in (self.decode, self.merge)
            )
        return self._entries

    def _view(self, role, tensor):
        """What ``role``'s checks derive from ``tensor``, kept per tensor
        geometry: a later call with a tensor of the same address, dtype,
        shape, strides and device reuses it."""
        key = (
            role,
            tensor.data_ptr(),
            tensor.dtype,
            tensor.shape,
            tensor.stride(),
            tensor.get_device(),
        )
        found = self._views.get(key)
        if found is None:
            found = _DERIVE[role](self, tensor)
            if len(self._views) >= VIEW_CACHE_ENTRIES:
                del self._views[next(iter(self._views))]
            self._views[key] = found
        return found

    def _device_check(self, tensor):
        if tensor.get_device() != self.device_index:
            raise ValueError(f"every tensor must be on {self.device}")

    def _kv(self, kv_cache):
        pages, stride = _page_view(kv_cache, KV_PAGE_BYTES, "kv_cache")
        if stride % 16:
            raise ValueError(
                f"kv_cache: page stride {stride} B is not a multiple of 16"
            )
        self._device_check(kv_cache)
        kv = torch.as_strided(
            kv_cache.view(torch.uint8),
            (pages, PAGE_TOKENS, TOKEN_BYTES),
            (stride, TOKEN_BYTES, 1),
        )
        return pages, tvm_ffi.from_dlpack(kv)

    def _sf(self, kv_scale_cache):
        pages, stride = _page_view(kv_scale_cache, SF_PAGE_BYTES, "kv_scale_cache")
        if stride != SF_PAGE_BYTES:
            raise ValueError(
                f"kv_scale_cache: page stride {stride} B, expected {SF_PAGE_BYTES}"
            )
        self._device_check(kv_scale_cache)
        words = kv_scale_cache.view(torch.uint8).view(-1).view(torch.uint32)
        return pages, tvm_ffi.from_dlpack(words)

    def _vsf(self, v_scale_cache):
        pages, stride = _page_view(v_scale_cache, V_SCALE_PAGE_BYTES, "v_scale_cache")
        if stride != V_SCALE_PAGE_BYTES:
            raise ValueError(
                f"v_scale_cache: page stride {stride} B; the decode reads "
                f"TensorRT-LLM's MLA V-scale pool as {V_SCALE_PAGE_BYTES}-byte pages "
                f"at a {V_SCALE_PAGE_BYTES}-byte stride, addressed by kv_cache's "
                "page ids, and takes no other layout"
            )
        self._device_check(v_scale_cache)
        words = v_scale_cache.view(torch.uint8).view(-1).view(torch.uint32)
        return pages, tvm_ffi.from_dlpack(words)

    def _q(self, q_fp4):
        p = self.plan
        if q_fp4.dtype not in _U8 or q_fp4.dim() != 2 or q_fp4.shape[1] != TOKEN_BYTES:
            raise ValueError(
                f"q_fp4: uint8 [rows, {TOKEN_BYTES}], got {q_fp4.dtype} {tuple(q_fp4.shape)}"
            )
        if not q_fp4.is_contiguous() or q_fp4.shape[0] < p.rows:
            raise ValueError(f"q_fp4: contiguous with at least {p.rows} rows")
        self._device_check(q_fp4)
        words = q_fp4.view(torch.uint8)[: p.rows].view(torch.uint32)
        return tvm_ffi.from_dlpack(words)

    def _q_sf(self, q_scale):
        need = -(-self.plan.rows // 128) * Q_SF_BLOCK_BYTES
        if (
            q_scale.dtype not in _U8
            or not q_scale.is_contiguous()
            or q_scale.numel() < need
        ):
            raise ValueError(f"q_scale: contiguous E4M3 bytes, at least {need}")
        self._device_check(q_scale)
        words = q_scale.view(torch.uint8).view(-1)[:need].view(torch.uint32)
        return tvm_ffi.from_dlpack(words)

    def _page_table(self, page_table):
        p = self.plan
        if (
            page_table.dtype != torch.int32
            or tuple(page_table.shape) != (p.batch, p.page_table_width)
            or not page_table.is_contiguous()
        ):
            raise ValueError(
                f"page_table: contiguous int32 [{p.batch}, {p.page_table_width}], got "
                f"{page_table.dtype} {tuple(page_table.shape)}"
            )
        self._device_check(page_table)
        return tvm_ffi.from_dlpack(page_table)

    def _kv_lens(self, kv_lens):
        p = self.plan
        if (
            kv_lens.dtype != torch.int32
            or kv_lens.dim() != 1
            or kv_lens.shape[0] < p.batch
            or not kv_lens.is_contiguous()
        ):
            raise ValueError(f"kv_lens: contiguous int32 [>= {p.batch}]")
        self._device_check(kv_lens)
        return tvm_ffi.from_dlpack(kv_lens[: p.batch])

    def _scale(self, name, scale):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(f"{name}: an FP32 one-element device tensor")
        self._device_check(scale)
        return tvm_ffi.from_dlpack(scale.view(1))

    def _out(self, out):
        p = self.plan
        if (
            out.dtype != torch.bfloat16
            or out.numel() != p.rows * LATENT
            or not out.is_contiguous()
        ):
            raise ValueError(f"out: contiguous BF16 with {p.rows} x {LATENT} elements")
        self._device_check(out)
        return tvm_ffi.from_dlpack(out.view(p.rows, LATENT))

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
        view = self._view
        pages, kv = view("kv_cache", kv_cache)
        sf_pages, sf = view("kv_scale_cache", kv_scale_cache)
        if sf_pages < pages:
            raise ValueError(
                f"kv_scale_cache holds {sf_pages} pages, fewer than kv_cache's {pages}"
            )
        vsf_pages, vsf = view("v_scale_cache", v_scale_cache)
        if vsf_pages < pages:
            raise ValueError(
                f"v_scale_cache holds {vsf_pages} pages, fewer than kv_cache's "
                f"{pages}: the decode reads the V-scale page of every kv_cache page id"
            )
        values = (
            kv,
            kv,
            sf,
            vsf,
            view("q_fp4", q_fp4),
            view("q_scale", q_scale),
            view("page_table", page_table),
            view("kv_lens", kv_lens),
            view("q_global_scale", q_global_scale),
            view("kv_global_scale", kv_global_scale),
            view("out", out),
            float(sm_scale),
        )
        decode, merge = self._entry_points()
        decode_args, decode_slots = self._decode_args
        decode_args = list(decode_args)
        for position, index in decode_slots:
            decode_args[position] = values[index]
        # One stream lookup per call: both launches go to torch's current
        # stream of the runner's device.
        with tvm_ffi.use_raw_stream(self._ffi_device, _raw_stream(self.device_index)):
            decode(*decode_args)
            if self.plan.num_split > 1:
                merge_args, merge_slots = self._merge_args
                merge_args = list(merge_args)
                for position, index in merge_slots:
                    merge_args[position] = values[index]
                merge(*merge_args)
        return out


# What each role's checks derive from the caller's tensor (Fp4MlaDecodeRunner._view).
_DERIVE = {
    "kv_cache": Fp4MlaDecodeRunner._kv,
    "kv_scale_cache": Fp4MlaDecodeRunner._sf,
    "v_scale_cache": Fp4MlaDecodeRunner._vsf,
    "q_fp4": Fp4MlaDecodeRunner._q,
    "q_scale": Fp4MlaDecodeRunner._q_sf,
    "page_table": Fp4MlaDecodeRunner._page_table,
    "kv_lens": Fp4MlaDecodeRunner._kv_lens,
    "q_global_scale": lambda runner, t: runner._scale("q_global_scale", t),
    "kv_global_scale": lambda runner, t: runner._scale("kv_global_scale", t),
    "out": Fp4MlaDecodeRunner._out,
}


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
    bucket and the device's SM count only, never on ``kv_lens``; the decode
    build on the bucket and the device's architecture.  ``max_kv``,
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
    arch = device_arch(index)
    if arch not in ARCHES:
        raise RuntimeError(f"no Cake fp4_mla decode program for {dev}")
    plan = _plan(
        batch,
        q_len,
        num_heads,
        page_table_width,
        _sm_count(index),
        max_splits,
        max_kv,
        arch,
    )
    return Fp4MlaDecodeRunner(plan, dev)


__all__ = [
    "Fp4MlaDecodePlan",
    "Fp4MlaDecodeRunner",
    "long_rows_for",
    "merge_warps_per_row",
    "num_split_for",
    "persistent_for",
    "plan_fp4_mla_decode",
]
