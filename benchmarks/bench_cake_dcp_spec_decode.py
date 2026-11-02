"""Benchmark FlashInfer's DCP speculative-decode routes on Blackwell.

Two arms per row, both served by ``flashinfer.cake_dcp.run_dcp_spec_decode``
on the same rank-local paged cache:

* ``static``   -- the incumbent FlashInfer programs (BF16/page-16 v1/v4
                  split-K, the E4M3/page-64 split-1..4 family and the
                  E4M3/page-64 head_dim-256 discrete split family).
* ``balanced`` -- the on-device load-balanced programs: a persistent grid
                  draining a queue of 256-token chunk pairs, split tiles
                  reduced by the finishing CTA, packed 32/64-row instances.

Rows are the round-3 fixed rows of the three profiles (BF16/page-16 D128,
E4M3/page-64 D128, E4M3/page-64 D256 GQA-16) plus the optional band probes
(``--bands``). Timing is CUPTI with a cold L2 through
``flashinfer.testing.bench_gpu_time_with_cupti``; ``--graph`` adds a
CUDA-graph replay column. Each row prints the band decision (the route
``route="auto"`` takes on this device), the two medians, their ratio (>1:
balanced faster) and the max-abs difference between the two arms.

Cold L2 means cold for both arms: the incumbent programs load K/V through
TMA with an ``evict_last`` hint, and a plain fill of twice the L2 (the
timer's flush) does not displace such lines, so on rows whose rank-local
K/V fits in L2 the static arm would otherwise re-read the previous
iteration's lines (about 3.5 us on a 15 us row on GB300).  After every
timed launch the bench synchronizes and calls
``cuCtxResetPersistingL2Cache`` so the next flush evicts those lines; the
CUPTI span still covers only the kernel.  ``--keep-persisting-l2``
reproduces the timer's default behaviour.

Example::

    python benchmarks/bench_cake_dcp_spec_decode.py --family fp8 --graph \\
        --json /tmp/dcp_fp8.json
    python benchmarks/bench_cake_dcp_spec_decode.py --family all --bands
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
import time

import torch

from flashinfer.cake_dcp import (
    dcp_balanced_band,
    dcp_balanced_n_rows,
    get_dcp_spec_balanced_counter_bytes,
    get_dcp_spec_balanced_workspace_bytes,
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
    run_dcp_spec_decode,
)
from flashinfer.jit.cake_fmha import CAKE_FMHA_JIT_TAG
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.utils import get_compute_capability

# The FlashInfer bench's ragged prefix pattern (flashinfer-ai/flashinfer#4832).
AGENTX = [
    8193, 57345, 73729, 81921, 98305, 106497, 114689, 131073,
    139265, 147457, 163841, 180225, 196609, 212993, 229377, 237569,
]  # fmt: skip

_KIND_HEADS = {"bf16_p16": (64, 8), "fp8_p64": (64, 8), "fp8_p64_d256": (16, 1)}
_KIND_HEAD_DIM = {"bf16_p16": 128, "fp8_p64": 128, "fp8_p64_d256": 256}
_KIND_PAGE_SIZE = {"bf16_p16": 16, "fp8_p64": 64, "fp8_p64_d256": 64}
_KIND_KV_DTYPE = {
    "bf16_p16": torch.bfloat16,
    "fp8_p64": torch.float8_e4m3fn,
    "fp8_p64_d256": torch.float8_e4m3fn,
}
_FAMILY_KIND = {"bf16": "bf16_p16", "fp8": "fp8_p64", "d256": "fp8_p64_d256"}
# Static split-K scratch is sized for the widest split the profile may take.
_STATIC_MAX_SPLIT = {"bf16_p16": 16, "fp8_p64": 4, "fp8_p64_d256": 16}
# E4M3 caches: the source is N(0, 0.2^2); a 4-sigma amax proxy sets the scale,
# so bmm1 = sm_scale * k_scale and bmm2 = v_scale like a real quantized cache.
_FP8_SOURCE_SIGMA = 0.2
_FP8_SCALE = 4.0 * _FP8_SOURCE_SIGMA / float(torch.finfo(torch.float8_e4m3fn).max)
_PAGE_CHUNK = 4096


def _agentx(batch: int) -> list[int]:
    return [AGENTX[i % len(AGENTX)] for i in range(batch)]


def _random_lengths(batch: int, lo: int, hi: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (batch,), generator=gen).tolist()


def _local_len(prefix: int, q_len: int, cp_world: int, cp_rank: int) -> int:
    """Rank-local keys visible to the last speculative row: |arange(rank, prefix + q_len, W)|."""

    last = prefix + q_len - 1 - cp_rank
    return 0 if last < 0 else last // cp_world + 1


# row -> (kind, global prefixes, q_len, cp_world, cp_rank)
ROWS: dict[str, tuple[str, list[int], int, int, int]] = {}


def _add(
    name: str, kind: str, prefixes, q_len: int, cp_world: int, cp_rank: int
) -> None:
    if isinstance(prefixes, int):
        raise TypeError("prefixes must be a list")
    assert name not in ROWS, name
    ROWS[name] = (kind, [int(p) for p in prefixes], q_len, cp_world, cp_rank)


# --- BF16 / page 16 / D128 (unit 2) ---------------------------------------
for _batch in (1, 8, 64):
    _add(f"perf_b{_batch}_s4096_q4_w4_r0", "bf16_p16", [4096] * _batch, 4, 4, 0)
    _add(f"perf_b{_batch}_s16384_q8_w4_r0", "bf16_p16", [16384] * _batch, 8, 4, 0)
_add("perf_b8_s16383_q8_w4_r3_tail", "bf16_p16", [16383] * 8, 8, 4, 3)
_add("dcp_bf16_agentx_b16_q4_cp4_r0", "bf16_p16", _agentx(16), 4, 4, 0)
_add("dcp_bf16_agentx_b16_q8_cp4_r3", "bf16_p16", _agentx(16), 8, 4, 3)
_add(
    "dcp_bf16_random_128_65k_b64_q4_cp4_r0",
    "bf16_p16",
    _random_lengths(64, 128, 65536, 1),
    4,
    4,
    0,
)
_add(
    "dcp_bf16_random_128_32k_b128_q4_cp4_r0",
    "bf16_p16",
    _random_lengths(128, 128, 32768, 3),
    4,
    4,
    0,
)

# --- E4M3 / page 64 / D128 (unit 3) ----------------------------------------
for _batch in (1, 8, 32, 64, 128, 192, 256):
    _add(f"prod_b{_batch}_s8192_q4_cp4_graph", "fp8_p64", [8192] * _batch, 4, 4, 0)
for _batch in (8, 64, 256):
    for _prefix in (4096, 16384):
        for _q_len in (4, 8):
            _add(
                f"prod_b{_batch}_s{_prefix}_q{_q_len}_cp4",
                "fp8_p64",
                [_prefix] * _batch,
                _q_len,
                4,
                0,
            )
for _prefix in (8191, 8192, 8193, 8194):
    _add(f"prod_b64_s{_prefix}_q4_cp4_residue", "fp8_p64", [_prefix] * 64, 4, 4, 0)
for _cp_world in (2, 8):
    _add(f"prod_b64_s8192_q4_cp{_cp_world}", "fp8_p64", [8192] * 64, 4, _cp_world, 0)
for _batch in (1, 8, 64, 256):
    _add(f"cp1_peer_b{_batch}_s8192_q4", "fp8_p64", [8192] * _batch, 4, 1, 0)
_add("stretch_b384_s8192_q4_cp4", "fp8_p64", [8192] * 384, 4, 4, 0)
_add("dcp_fp8_agentx_b16_q4_cp4_r0", "fp8_p64", _agentx(16), 4, 4, 0)
_add("dcp_fp8_agentx_b16_q8_cp4_r3", "fp8_p64", _agentx(16), 8, 4, 3)
_add(
    "dcp_fp8_random_128_65k_b64_q4_cp4_r0",
    "fp8_p64",
    _random_lengths(64, 128, 65536, 1),
    4,
    4,
    0,
)
_add(
    "dcp_fp8_random_128_32k_b128_q4_cp4_r0",
    "fp8_p64",
    _random_lengths(128, 128, 32768, 3),
    4,
    4,
    0,
)

# --- E4M3 / page 64 / D256 GQA-16 (unit 4): ctx32768 = prefix + q_len ------
for _batch in (1, 8, 16, 32, 64, 128, 192, 256):
    _add(
        f"prod_d256_b{_batch}_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        [32768 - 4] * _batch,
        4,
        4,
        0,
    )
for _q_len in (1, 2, 3, 5, 6, 7, 8):
    _add(
        f"prod_d256_b128_ctx32768_q{_q_len}_cp4_graph",
        "fp8_p64_d256",
        [32768 - _q_len] * 128,
        _q_len,
        4,
        0,
    )
_add("dcp_d256_agentx_b16_q4_cp4_r0", "fp8_p64_d256", _agentx(16), 4, 4, 0)
_add("dcp_d256_agentx_b64_q4_cp4_r0", "fp8_p64_d256", _agentx(64), 4, 4, 0)
_add(
    "dcp_d256_random_128_65k_b128_q4_cp4_r3",
    "fp8_p64_d256",
    _random_lengths(128, 128, 65536, 5),
    4,
    4,
    3,
)

FIXED_ROWS = tuple(ROWS)

# --- band probes (opt-in): the 19 + 3 BF16 and the 27 E4M3 crossover rows ---
_BAND_GEOMETRY = [
    (1, 8, 4096, 4, 0), (2, 4, 4096, 4, 0), (4, 4, 4096, 4, 0), (1, 4, 8192, 4, 0),
    (1, 8, 8192, 4, 0), (2, 8, 16384, 4, 0), (1, 4, 32768, 4, 0), (1, 4, 65536, 4, 0),
    (8, 4, 512, 4, 0), (8, 4, 1024, 4, 0), (64, 4, 1024, 4, 0), (8, 3, 4096, 4, 0),
    (8, 6, 4096, 4, 0), (8, 4, 4096, 8, 5), (8, 4, 4096, 2, 1), (8, 2, 4096, 4, 0),
    (8, 1, 4096, 4, 0), (256, 4, 4096, 4, 0), (128, 8, 16384, 4, 0),
    (1, 4, 6144, 4, 0), (1, 4, 7168, 4, 0), (2, 8, 6144, 4, 0),
]  # fmt: skip
for _batch, _q_len, _prefix, _cp_world, _cp_rank in _BAND_GEOMETRY:
    _add(
        f"bandbf16_b{_batch}_s{_prefix}_q{_q_len}_w{_cp_world}_r{_cp_rank}",
        "bf16_p16",
        [_prefix] * _batch,
        _q_len,
        _cp_world,
        _cp_rank,
    )
for _batch, _q_len, _prefix, _cp_world, _cp_rank in _BAND_GEOMETRY + [
    (1, 4, 16384, 4, 0), (1, 4, 24576, 4, 0), (1, 4, 8192, 1, 0), (1, 4, 4096, 1, 0), (8, 4, 8192, 1, 0),
]:  # fmt: skip
    _add(
        f"bandfp8_b{_batch}_s{_prefix}_q{_q_len}_w{_cp_world}_r{_cp_rank}",
        "fp8_p64",
        [_prefix] * _batch,
        _q_len,
        _cp_world,
        _cp_rank,
    )

BAND_ROWS = tuple(name for name in ROWS if name not in FIXED_ROWS)


def _device_arch(device: torch.device) -> str:
    major, minor = get_compute_capability(device)
    return {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(
        (major, minor), f"sm{major}{minor}f"
    )


class _Problem:
    """One rank-local DCP problem with scratch provisioned for both routes."""

    def __init__(
        self,
        kind: str,
        prefixes: list[int],
        q_len: int,
        cp_world: int,
        cp_rank: int,
        *,
        device,
        seed: int,
    ):
        self.kind, self.q_len, self.cp_world, self.cp_rank = (
            kind,
            q_len,
            cp_world,
            cp_rank,
        )
        self.prefixes = prefixes
        self.device = device
        self.num_q_heads, self.num_kv_heads = _KIND_HEADS[kind]
        self.head_dim, self.page_size = _KIND_HEAD_DIM[kind], _KIND_PAGE_SIZE[kind]
        kv_dtype = _KIND_KV_DTYPE[kind]
        batch = len(prefixes)
        gen = torch.Generator(device=device).manual_seed(seed)

        self.local_lens = [_local_len(p, q_len, cp_world, cp_rank) for p in prefixes]
        self.max_local = max(self.local_lens)
        data_pages = [max(1, math.ceil(n / self.page_size)) for n in self.local_lens]
        total_pages = sum(data_pages)
        dummy_page = total_pages
        blocks = max(1, math.ceil(self.max_local / 128))
        blocks += blocks % 2
        max_pages_per_seq = blocks * 128 // self.page_size

        shape = (total_pages + 1, self.num_kv_heads, self.page_size, self.head_dim)
        self.k_cache = torch.empty(shape, dtype=kv_dtype, device=device)
        self.v_cache = torch.empty(shape, dtype=kv_dtype, device=device)
        for cache in (self.k_cache, self.v_cache):
            for start in range(0, shape[0], _PAGE_CHUNK):
                count = min(_PAGE_CHUNK, shape[0] - start)
                block = (
                    torch.randn(
                        (count, *shape[1:]),
                        dtype=torch.float32,
                        device=device,
                        generator=gen,
                    )
                    * _FP8_SOURCE_SIGMA
                )
                if kv_dtype == torch.float8_e4m3fn:
                    fp8_max = float(torch.finfo(kv_dtype).max)
                    block = (block / _FP8_SCALE).clamp_(-fp8_max, fp8_max)
                cache[start : start + count] = block.to(kv_dtype)
        self.block_tables = torch.full(
            (batch, max_pages_per_seq), dummy_page, dtype=torch.int32, device=device
        )
        order = torch.randperm(total_pages, device=device, generator=gen).to(
            torch.int32
        )
        next_page = 0
        for batch_idx, page_count in enumerate(data_pages):
            self.block_tables[batch_idx, :page_count] = order[
                next_page : next_page + page_count
            ]
            next_page += page_count
        self.seq_lens = torch.tensor(self.local_lens, dtype=torch.int32, device=device)
        self.prefix_tensor = torch.tensor(prefixes, dtype=torch.int32, device=device)

        sm_scale = self.head_dim**-0.5
        if kv_dtype == torch.float8_e4m3fn:
            self.bmm1_scale, self.bmm2_scale = sm_scale * _FP8_SCALE, _FP8_SCALE
        else:
            self.bmm1_scale, self.bmm2_scale = sm_scale, 1.0
        self.query = (
            torch.randn(
                (batch * q_len, self.num_q_heads, self.head_dim),
                dtype=torch.float32,
                device=device,
                generator=gen,
            )
            * 0.2
        ).to(torch.bfloat16)
        self.out = torch.empty_like(self.query)
        self.lse = torch.empty(
            (batch * q_len, self.num_q_heads), dtype=torch.float32, device=device
        )

        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.empty(
            max(
                get_dcp_spec_balanced_workspace_bytes(sm_count, self.head_dim),
                get_dcp_spec_workspace_size_bytes(
                    batch,
                    q_len,
                    self.num_q_heads,
                    _STATIC_MAX_SPLIT[kind],
                    head_dim=self.head_dim,
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.counter = torch.zeros(
            max(
                get_dcp_spec_balanced_counter_bytes(sm_count),
                get_dcp_spec_counter_bytes(batch, q_len, self.num_kv_heads),
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.sm_count = sm_count

    def bytes_moved(self) -> int:
        kv = (
            2
            * sum(self.local_lens)
            * self.num_kv_heads
            * self.head_dim
            * self.k_cache.element_size()
        )
        q = self.query.numel() * self.query.element_size()
        return kv + 2 * q  # query read plus output write

    def band(self):
        return dcp_balanced_band(
            self.kind,
            batch_size=len(self.prefixes),
            q_len=self.q_len,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_local_seq_len=self.max_local,
            cp_world=self.cp_world,
            sm_count=self.sm_count,
            arch=_device_arch(self.device),
        )

    def run(self, route: str) -> None:
        run_dcp_spec_decode(
            query=self.query,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            workspace_buffer=self.workspace,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            causal_seqlens_kv_global=self.prefix_tensor,
            max_local_seq_len=self.max_local,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            cp_world=self.cp_world,
            cp_rank=self.cp_rank,
            q_len_per_req=self.q_len,
            out=self.out,
            lse=self.lse,
            completion_buffer=self.counter,
            route=route,
        )


def _reset_persisting_l2() -> None:
    """Demote every L2 line marked ``evict_last`` to normal priority.

    Run after a launch has completed: the timer's next flush (a plain fill of
    twice the L2) then evicts the K/V lines the incumbent programs installed
    with their ``evict_last`` TMA hint, which the fill alone leaves resident.
    """

    from cuda.bindings import driver as cuda

    (err,) = cuda.cuCtxResetPersistingL2Cache()
    if int(err) != 0:
        raise RuntimeError(f"cuCtxResetPersistingL2Cache -> {err}")


_KEEP_PERSISTING_L2 = False


def _median_ms(fn, use_cuda_graph: bool = False) -> float:
    if _KEEP_PERSISTING_L2:
        times = bench_gpu_time_with_cupti(
            fn, cold_l2_cache=True, use_cuda_graph=use_cuda_graph
        )
    else:
        runner = fn
        if use_cuda_graph:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            runner = graph.replay

        def launch_then_demote() -> None:
            # The timer brackets the GPU activity of this call and synchronizes
            # after it; the demotion is a host call outside the kernel span.
            runner()
            torch.cuda.synchronize()
            _reset_persisting_l2()

        times = bench_gpu_time_with_cupti(
            launch_then_demote, cold_l2_cache=True, use_cuda_graph=False
        )
    times = sorted(times)
    return float(times[len(times) // 2])


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if bool((torch.isfinite(a) != torch.isfinite(b)).any()):
        return float("inf")
    if not bool(finite.any()):
        return 0.0
    return float((a[finite] - b[finite]).abs().max().item())


def _bench_row(name: str, *, device, graph: bool, seed: int) -> dict:
    kind, prefixes, q_len, cp_world, cp_rank = ROWS[name]
    problem = _Problem(
        kind, prefixes, q_len, cp_world, cp_rank, device=device, seed=seed
    )
    band = problem.band()
    result = {
        "row": name,
        "kind": kind,
        "batch": len(prefixes),
        "q_len": q_len,
        "cp_world": cp_world,
        "cp_rank": cp_rank,
        "max_local_seq_len": problem.max_local,
        "local_tokens": sum(problem.local_lens),
        "n_rows": dcp_balanced_n_rows(kind, q_len),
        "auto_route": band.route,
        "reason": band.reason,
        "num_split": band.num_split,
        "waves": band.waves,
        "blocks_per_cta": band.blocks_per_cta,
        "items": band.items,
        "bytes": problem.bytes_moved(),
    }
    outputs = {}
    for route in ("static", "balanced"):

        def launch(route: str = route) -> None:
            problem.run(route)

        try:
            launch()  # JIT load + the outputs the arms are compared on
            torch.cuda.synchronize(device)
            outputs[route] = (problem.out.clone(), problem.lse.clone())
            result[f"{route}_ms"] = _median_ms(launch)
            if graph:
                result[f"{route}_graph_ms"] = _median_ms(launch, use_cuda_graph=True)
            result[f"{route}_gbps"] = result["bytes"] / result[f"{route}_ms"] / 1.0e6
        except Exception as error:  # noqa: BLE001 - report the arm, keep the sweep going
            torch.cuda.synchronize(device)
            result[f"{route}_error"] = f"{type(error).__name__}: {error}"
        if int(torch.count_nonzero(problem.counter).item()) != 0:
            result[f"{route}_error"] = (
                result.get(f"{route}_error", "") + " counters not self-reset"
            )
    if "static_ms" in result and "balanced_ms" in result:
        result["ratio"] = result["static_ms"] / result["balanced_ms"]
        if graph:
            result["graph_ratio"] = (
                result["static_graph_ms"] / result["balanced_graph_ms"]
            )
        result["max_abs_diff_out"] = _max_abs_diff(
            outputs["static"][0], outputs["balanced"][0]
        )
        result["max_abs_diff_lse"] = _max_abs_diff(
            outputs["static"][1], outputs["balanced"][1]
        )
    torch.cuda.empty_cache()
    return result


def _format(value, width: int, digits: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rjust(width)
    return str(value).rjust(width)


def _print_row(result: dict, graph: bool) -> None:
    cells = [
        result["row"].ljust(44),
        result["auto_route"].ljust(8),
        result["reason"].ljust(14),
        _format(result.get("static_ms"), 10),
        _format(result.get("balanced_ms"), 11),
        _format(result.get("ratio"), 7, 3),
    ]
    if graph:
        cells += [
            _format(result.get("static_graph_ms"), 10),
            _format(result.get("balanced_graph_ms"), 11),
            _format(result.get("graph_ratio"), 7, 3),
        ]
    cells += [
        _format(result.get("balanced_gbps"), 8, 0),
        _format(result.get("max_abs_diff_out"), 9, 5),
        _format(result.get("max_abs_diff_lse"), 9, 5),
    ]
    line = " ".join(cells)
    errors = [
        f"{route}: {result[f'{route}_error']}"
        for route in ("static", "balanced")
        if f"{route}_error" in result
    ]
    if errors:
        line += "  !! " + " | ".join(errors)
    print(line, flush=True)


def _print_header(graph: bool) -> None:
    cells = [
        "row".ljust(44),
        "auto".ljust(8),
        "reason".ljust(14),
        "static_ms".rjust(10),
        "balanced_ms".rjust(11),
        "ratio".rjust(7),
    ]
    if graph:
        cells += ["st_graph".rjust(10), "bal_graph".rjust(11), "g_ratio".rjust(7)]
    cells += ["bal_GB/s".rjust(8), "dO_max".rjust(9), "dLSE_max".rjust(9)]
    print(" ".join(cells))
    print("-" * len(" ".join(cells)))


def _select_rows(args) -> list[str]:
    kinds = (
        set(_FAMILY_KIND.values())
        if args.family == "all"
        else {_FAMILY_KIND[args.family]}
    )
    names = list(FIXED_ROWS) + (list(BAND_ROWS) if args.bands else [])
    if args.rows:
        patterns = [pattern for pattern in args.rows.split(",") if pattern]
        names = [
            name
            for name in ROWS
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        ]
        missing = [
            pattern
            for pattern in patterns
            if not any(fnmatch.fnmatchcase(name, pattern) for name in ROWS)
        ]
        if missing:
            raise SystemExit(f"unknown rows: {missing}")
    return [name for name in names if ROWS[name][0] in kinds]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--family", choices=("bf16", "fp8", "d256", "all"), default="all"
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="comma-separated row names or fnmatch patterns (overrides --family filtering by name)",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="also run the BF16 and E4M3 band probe rows",
    )
    parser.add_argument(
        "--graph", action="store_true", help="add CUDA-graph replay medians"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the selected rows and exit"
    )
    parser.add_argument("--seed", type=int, default=685)
    parser.add_argument(
        "--keep-persisting-l2",
        action="store_true",
        help="do not demote evict_last L2 lines between iterations (the timer's "
        "default flush; the static arm then re-reads resident K/V on small rows)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    names = _select_rows(args)
    if args.list:
        for name in names:
            kind, prefixes, q_len, cp_world, cp_rank = ROWS[name]
            print(
                f"{name}: {kind} batch={len(prefixes)} q_len={q_len} cp_world={cp_world} cp_rank={cp_rank} max_prefix={max(prefixes)}"
            )
        return 0
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    header = {
        "device": properties.name,
        "compute_capability": ".".join(map(str, get_compute_capability(device))),
        "arch": _device_arch(device),
        "sm_count": properties.multi_processor_count,
        "jit_tag": CAKE_FMHA_JIT_TAG,
        "torch": torch.__version__,
        "timing": "CUPTI, cold L2"
        + (
            " (evict_last lines kept)"
            if args.keep_persisting_l2
            else " (evict_last lines demoted)"
        )
        + ", median"
        + (" (+ CUDA-graph replay)" if args.graph else ""),
        "rows": len(names),
    }
    print(json.dumps(header))
    global _KEEP_PERSISTING_L2
    _KEEP_PERSISTING_L2 = bool(args.keep_persisting_l2)
    _print_header(args.graph)
    results = []
    started = time.time()
    for name in names:
        result = _bench_row(name, device=device, graph=args.graph, seed=args.seed)
        results.append(result)
        _print_row(result, args.graph)
    print(f"# {len(results)} rows in {time.time() - started:.0f} s")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"header": header, "results": results}, handle, indent=1)
        print(f"# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
