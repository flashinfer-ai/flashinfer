"""CAKE backend for Kimi-K3 MLA attention over an FP8 (E4M3) paged latent cache (SM100 / SM103).

Generated from the Cake schedule ``loom/examples/weave/kimi_k3_mla_fp8_paged_attention.py``
(Linear CAKE-645).  One prepared launcher covers low-head paged decode (``q_len = 1``),
packed variable-Q / MTP (``cum_seq_lens_q``) and incremental prefill on the paged FP8
cache: BF16 output into a caller-owned buffer, FP8 query and KV (latent 512 + rope 64),
page size 64, bottom-right causal mask, current stream, CUDA-Graph replayable
(``launch`` allocates nothing; the split-KV partials live in ``workspace_buffer``).

Host planning mirrors the Cake module exactly (``swapped_rt``, ``plan_num_split``,
``reduce_warps_per_row``); the exporter's ``prepare`` step checks that both arms built the
same plan.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch

from ..utils import get_compute_capability

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
V_DIM = LATENT
PAGE_SIZE = 64
TILE_TOK = 128  # tokens per KV tile of the swapped-AB schedule (two pages)
MAX_SPLITS = 256
MIN_TILES_PER_SPLIT = 2
REDUCE_WARPS = 8  # warp reducer CTA: 256 threads
REDUCE_WARP_MAX_SPLITS = 32  # lane s owns split s
REDUCE_DIM_CHUNKS = 4  # CTA reducer: 128 latent dims per CTA
ROW_TILES = (16, 32, 48, 64, 96)
# Two-CTA wide route (Cake ``kimi_k3_mla_wide``): 128 packed rows per cluster of two CTAs, K tokens
# split across the pair.  Taken for requests with more than WIDE_MIN_ROWS packed rows whose longest
# KV is at least WIDE_MIN_KV tokens (the lazy-E4M3 probability reference of that schedule is out of
# the contract tolerance on shorter KV prefill, which stays on the row tiles).
WIDE_MIN_ROWS = 64
WIDE_MIN_KV = 16384
WIDE_TILE_Q = 128  # packed rows per two-CTA cluster
WIDE_CLUSTER = 2  # CTAs per cluster: one SM pair per work item
WIDE_MIN_TILES_PER_SPLIT = 2


def _target_arch(device: torch.device) -> str:
    major, minor = get_compute_capability(device)
    return f"sm_{major}{minor}a"


def swapped_rt(rows_per_request: int) -> int:
    """Row tile for a request: the smallest template holding all rows, else 96-row tiles."""
    for rt in ROW_TILES:
        if rows_per_request <= rt:
            return rt
    return 96


def plan_num_split(items: int, max_seq_len: int, sm_count: int) -> int:
    """One CTA per SM per wave over (work item, split) pairs without a ragged tail."""
    target = max(1, sm_count // max(1, items))
    max_by_len = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK // MIN_TILES_PER_SPLIT)
    return max(1, min(target, MAX_SPLITS, max_by_len))


def use_wide_route(rows_per_request: int, max_seq_len: int) -> bool:
    """Whether a request shape runs the two-CTA wide route (else a swapped-AB row tile)."""
    return rows_per_request > WIDE_MIN_ROWS and int(max_seq_len) >= WIDE_MIN_KV


def plan_num_split_wide(
    clusters: int,
    max_seq_len: int,
    sm_count: int,
    min_tiles_per_split: int = WIDE_MIN_TILES_PER_SPLIT,
    max_splits: int = MAX_SPLITS,
) -> int:
    """KV splits per cluster of the wide route (mirrors Cake ``plan_num_split_wide``).

    Work items (clusters x splits) run one per SM pair; the cost of ``s`` splits in 128-token
    tile periods is ``ceil(items / pairs) * ceil(tiles / s)`` plus ~0.05 tile periods per item for
    the split merge.  A split is taken only when that model predicts at least 15 %.
    """
    pairs = max(1, sm_count // WIDE_CLUSTER)
    tiles = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK)
    max_s = max(1, min(max_splits, tiles // max(1, min_tiles_per_split)))
    best_s, best_cost, cost_one = 1, None, None
    for s in range(1, max_s + 1):
        items = clusters * s
        cost = -(-items // pairs) * (-(-tiles // s)) + 0.05 * items
        if s == 1:
            cost_one = cost
        if best_cost is None or cost < best_cost:
            best_s, best_cost = s, cost
    assert cost_one is not None and best_cost is not None
    if best_s > 1 and cost_one / best_cost < 1.15:
        return 1
    return best_s


def reduce_warps_per_row(rows: int) -> int:
    if rows <= 256:
        return 4
    if rows <= 1024:
        return 2
    return 1


def _align16(n: int) -> int:
    return (n + 15) & ~15


def workspace_bytes(rows_max: int, num_split: int) -> int:
    """Bytes of ``workspace_buffer`` needed: BF16 partial O + FP32 partial max / sum."""
    partial_o = rows_max * num_split * V_DIM * 2 if num_split > 1 else 0
    stats = rows_max * num_split * 4
    return _align16(partial_o) + 2 * _align16(stats)


def _carve_workspace(workspace: torch.Tensor, rows_max: int, num_split: int):
    if workspace.device.type != "cuda" or not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be a contiguous CUDA tensor")
    raw = workspace.view(torch.uint8).reshape(-1)
    need = workspace_bytes(rows_max, num_split)
    if raw.numel() < need:
        raise ValueError(
            f"workspace_buffer needs at least {need} bytes for this CAKE Kimi-K3 MLA plan, "
            f"got {raw.numel()}"
        )
    off = 0
    partial_o = None
    if num_split > 1:
        n = rows_max * num_split * V_DIM * 2
        partial_o = (
            raw[off : off + n].view(torch.bfloat16).view(rows_max, num_split, V_DIM)
        )
        off += _align16(n)
    n = rows_max * num_split * 4
    partial_max = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    off += _align16(n)
    partial_sum = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    return partial_o, partial_max, partial_sum


def _bound_args(
    contract: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
):
    """Order ``values`` by the generated argument plan (grid dims appended by kind)."""
    grid_args = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    args = []
    for kind, name in contract["arg_plan"]:
        if kind == "grid":
            args.append(int(grid_args[name]))
        elif kind == "workspace":
            raise ValueError(
                f"CAKE Kimi-K3 MLA module {contract['name']} has an unresolved workspace "
                f"argument {name!r} (the package binds TMA descriptors by value)"
            )
        else:
            try:
                args.append(values[name])
            except KeyError as exc:
                raise ValueError(
                    f"CAKE Kimi-K3 MLA module {contract['name']} needs argument {name!r}"
                ) from exc
    return args


class KimiK3MlaFp8PagedAttention:
    """Prepared launcher (no allocation at ``launch``; all planning at construction).

    Route: requests with more than ``WIDE_MIN_ROWS`` packed (token, head) rows and a longest KV of
    at least ``WIDE_MIN_KV`` tokens run the two-CTA wide schedule (``main_wide``: 128 rows per
    cluster); every other shape runs the swapped-AB row tile ``main_rt{16,32,48,64,96}``.  Both
    routes share the split-KV merge kernels and this workspace layout.

    Args mirror ``trtllm_batch_decode_with_kv_cache_mla``: ``query`` FP8 ``[B, q_len, H, 576]``
    or ``[total_q, H, 576]`` with ``cum_seq_lens_q``; ``kv_cache`` FP8 ``[pages, 64, 576]`` or
    ``[pages, 1, 64, 576]``; ``block_tables`` int32 ``[B, width]``; ``seq_lens`` int32 ``[B]``;
    ``out`` BF16 ``query.shape[:-1] + (512,)``; ``workspace_buffer`` a CUDA byte buffer of at
    least ``workspace_bytes(rows_max, num_split)`` bytes.
    """

    def __init__(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float = 1.0,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_split: Optional[int] = None,
    ):
        from ..jit.cake_kimi_k3_mla import get_cake_kimi_k3_mla_route

        device = query.device
        if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
            raise ValueError("query and kv_cache must be float8_e4m3fn")
        if query.shape[-1] != QK_DIM or kv_cache.shape[-1] != QK_DIM:
            raise ValueError(f"query / kv_cache last dim must be {QK_DIM}")
        if kv_cache.shape[-2] != PAGE_SIZE:
            raise ValueError(f"page_size must be {PAGE_SIZE}")
        if query.ndim == 4:
            batch, q_len, num_heads, _ = query.shape
            if cum_seq_lens_q is None:
                cum_seq_lens_q = torch.arange(
                    0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device
                )
            max_q_len = int(q_len)
        elif query.ndim == 3:
            if cum_seq_lens_q is None or max_q_len is None:
                raise ValueError("ragged query needs cum_seq_lens_q and max_q_len")
            batch = int(cum_seq_lens_q.shape[0]) - 1
            _, num_heads, _ = query.shape
        else:
            raise ValueError("query must be 3D or 4D")
        for name, t in (
            ("seq_lens", seq_lens),
            ("block_tables", block_tables),
            ("cum_seq_lens_q", cum_seq_lens_q),
        ):
            if t.dtype != torch.int32:
                raise ValueError(f"{name} must be int32")
        if (
            out.dtype != torch.bfloat16
            or out.shape[-1] != V_DIM
            or out.numel() != query.numel() // QK_DIM * V_DIM
        ):
            raise ValueError("out must be BF16 with shape query.shape[:-1] + (512,)")
        if not (
            query.is_contiguous()
            and kv_cache.is_contiguous()
            and out.is_contiguous()
            and block_tables.is_contiguous()
        ):
            raise ValueError("query, kv_cache, block_tables and out must be contiguous")
        if max_seq_len is None:
            max_seq_len = int(block_tables.shape[-1]) * PAGE_SIZE
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.arch = _target_arch(device)
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.max_q_len = int(max_q_len)
        self.rows_max = self.batch * self.max_q_len * self.num_heads
        rows_per_request = self.max_q_len * self.num_heads
        self.wide = use_wide_route(rows_per_request, int(max_seq_len))
        if self.wide:
            self.rt: Optional[int] = None
            self.m_tiles = (rows_per_request + WIDE_TILE_Q - 1) // WIDE_TILE_Q
            self.num_split = (
                int(num_split)
                if num_split
                else plan_num_split_wide(
                    self.batch * self.m_tiles, int(max_seq_len), sm_count
                )
            )
        else:
            self.rt = swapped_rt(rows_per_request)
            self.m_tiles = (rows_per_request + self.rt - 1) // self.rt
            self.num_split = (
                int(num_split)
                if num_split
                else plan_num_split(self.batch * self.m_tiles, int(max_seq_len), sm_count)
            )
        self.max_pages_per_seq = int(block_tables.shape[-1])
        self.softmax_scale_log2 = float(bmm1_scale) * math.log2(math.e)
        self.bmm2_scale = float(bmm2_scale)
        self.q_rows = query.view(torch.uint8).reshape(-1, QK_DIM)
        self.kv_rows = kv_cache.view(torch.uint8).reshape(-1, QK_DIM)
        self.o_rows = out.view(-1, V_DIM)
        self.seq_lens = seq_lens
        self.cum_seq_lens_q = cum_seq_lens_q
        self.block_tables = block_tables.reshape(-1)
        self.partial_O, self.partial_max, self.partial_sum = _carve_workspace(
            workspace_buffer, self.rows_max, self.num_split
        )
        # The wide route launches one cluster of two CTAs per (split, row tile, request) item.
        self.grid_main = (
            (WIDE_CLUSTER if self.wide else 1) * self.num_split,
            self.m_tiles,
            self.batch,
        )
        if self.num_split <= REDUCE_WARP_MAX_SPLITS:
            self.reduce_warps = reduce_warps_per_row(self.rows_max)
            rows_per_cta = REDUCE_WARPS // self.reduce_warps
            self.grid_reduce = (
                (self.rows_max + rows_per_cta - 1) // rows_per_cta,
                1,
                1,
            )
            reduce_kind = f"reduce_w{self.reduce_warps}"
        else:
            self.reduce_warps = 0
            self.grid_reduce = (self.rows_max, REDUCE_DIM_CHUNKS, 1)
            reduce_kind = "reduce_cta"
        main_kind = "main_wide" if self.wide else f"main_rt{self.rt}"
        self._main = get_cake_kimi_k3_mla_route(main_kind, arch=self.arch)
        self._reduce = get_cake_kimi_k3_mla_route(reduce_kind, arch=self.arch)
        self.route_metadata = dict(
            backend="cake",
            arch=self.arch,
            route="wide" if self.wide else "swapped",
            rt=self.rt,
            reducer=reduce_kind,
            main_module=self._main["name"],
            reduce_module=self._reduce["name"],
        )
        self.plan = dict(
            rt=self.rt,
            m_tiles=self.m_tiles,
            num_split=self.num_split,
            rows_max=self.rows_max,
            grid_main=tuple(self.grid_main),
            grid_reduce=tuple(self.grid_reduce),
            reduce_warps=self.reduce_warps,
            softmax_scale_log2=self.softmax_scale_log2,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        write_target = self.o_rows if self.num_split == 1 else self.partial_O
        self._main_args = _bound_args(
            self._main,
            dict(
                tmap_q=self.q_rows,
                tmap_qr=self.q_rows,
                tmap_k=self.kv_rows,
                tmap_kr=self.kv_rows,
                tmap_v=self.kv_rows,
                partial_O=write_target,
                partial_max=self.partial_max,
                partial_sum=self.partial_sum,
                seq_lens=self.seq_lens,
                cum_seq_lens_q=self.cum_seq_lens_q,
                page_table=self.block_tables,
                softmax_scale_log2=self.softmax_scale_log2,
                bmm2_scale=self.bmm2_scale,
                num_heads=self.num_heads,
                num_split=self.num_split,
                max_pages_per_seq=self.max_pages_per_seq,
            ),
            self.grid_main,
        )
        self._reduce_args = None
        if self.num_split > 1:
            self._reduce_args = _bound_args(
                self._reduce,
                dict(
                    partial_O=self.partial_O,
                    partial_max=self.partial_max,
                    partial_sum=self.partial_sum,
                    O=self.o_rows,
                    cum_seq_lens_q=self.cum_seq_lens_q,
                    batch=self.batch,
                    num_heads=self.num_heads,
                    num_split=self.num_split,
                    bmm2_scale=self.bmm2_scale,
                ),
                self.grid_reduce,
            )
        self._main_fn = None
        self._reduce_fn = None

    def _load(self) -> Callable[..., Any]:
        from ..jit.cake_kimi_k3_mla import get_cake_kimi_k3_mla_module

        main_fn = getattr(
            get_cake_kimi_k3_mla_module(self._main["name"]), self._main["ffi_entry"]
        )
        self._main_fn = main_fn
        if self._reduce_args is not None:
            self._reduce_fn = getattr(
                get_cake_kimi_k3_mla_module(self._reduce["name"]),
                self._reduce["ffi_entry"],
            )
        return main_fn

    def launch(self) -> None:
        """Enqueue the attention (and the split merge) on the current stream; no allocation."""
        main_fn = self._main_fn if self._main_fn is not None else self._load()
        main_fn(*self._main_args)
        if self._reduce_fn is not None and self._reduce_args is not None:
            self._reduce_fn(*self._reduce_args)


def run_cake_kimi_k3_mla_fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """One-shot entry used by ``trtllm_batch_decode_with_kv_cache_mla(backend="cake")``."""
    runner = KimiK3MlaFp8PagedAttention(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
    )
    runner.launch()
    return out


__all__ = [
    "KimiK3MlaFp8PagedAttention",
    "run_cake_kimi_k3_mla_fp8_paged_attention",
    "workspace_bytes",
    "swapped_rt",
    "use_wide_route",
    "plan_num_split",
    "plan_num_split_wide",
    "reduce_warps_per_row",
]
