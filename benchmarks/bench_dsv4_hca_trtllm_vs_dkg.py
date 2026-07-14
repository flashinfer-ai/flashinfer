#!/usr/bin/env python3
"""Compare TRTLLM-GEN and CuTe DSL implementations of DSV4 HCA.

The benchmark lowers the same logical HCA page tables to the two backend ABIs:

* TRTLLM-GEN receives one physical token-row index per sparse slot.
* CuTe DSL receives physical page IDs for the window and compressed pools.

Both backends therefore read the same FP8 cache rows in the same logical order.
The benchmark reports raw backend latency and public FlashInfer API latency with
both hot and flushed L2 caches. JIT compilation and the first launch are always
excluded from timing.

For continuity with the source project, CSV rows labeled ``dkg`` refer to the
DKG-derived CuTe DSL HCA kernel integrated into this FlashInfer checkout. They
do not invoke the original DKG checkout; original-versus-integrated parity is
measured by the separate port-parity benchmark.

For causal Q > 1, the current HCA ABI requires one compacted 128-slot window
pool row per query token. This benchmark materializes that row-private pool
before timing; its construction cost is intentionally excluded from both
backends' kernel measurements.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path
import site
import statistics
import sys
from typing import Callable, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HEAD_DIM = 512
WINDOW_CAPACITY = 128
HCA_COMPRESS_RATIO = 128


@dataclasses.dataclass
class HcaCase:
    batch_size: int
    raw_seq_len: int
    num_heads: int = 128
    q_len: int = 1
    window_page_size: int = 32
    compressed_page_size: int = 128
    page_layout: str = "shuffled"
    trt_index_layout: str = "native"
    dkg_persistent: bool = False

    @property
    def compressed_len(self) -> int:
        return self.raw_seq_len // HCA_COMPRESS_RATIO

    @property
    def compressed_pages_per_row(self) -> int:
        return math.ceil(self.compressed_len / self.compressed_page_size)

    @property
    def compressed_capacity(self) -> int:
        return self.compressed_pages_per_row * self.compressed_page_size

    @property
    def active_hca_len(self) -> int:
        return WINDOW_CAPACITY + self.compressed_len

    @property
    def hca_capacity(self) -> int:
        return WINDOW_CAPACITY + self.compressed_capacity

    @property
    def trt_index_capacity(self) -> int:
        if self.trt_index_layout == "page-capacity":
            return self.hca_capacity
        return math.ceil(self.active_hca_len / 4) * 4


@dataclasses.dataclass
class HcaInputs:
    case: HcaCase
    query: torch.Tensor
    window_cache: torch.Tensor
    compressed_cache: torch.Tensor
    window_tables: torch.Tensor
    compressed_tables: torch.Tensor
    sparse_indices: torch.Tensor
    seq_lens: torch.Tensor
    hca_seq_lens: torch.Tensor
    sparse_topk_lens: torch.Tensor
    window_valid_lens: torch.Tensor
    workspace: torch.Tensor
    out_trtllm: torch.Tensor
    out_dkg: torch.Tensor

    @property
    def window_cache_hnd(self) -> torch.Tensor:
        return self.window_cache.unsqueeze(1)

    @property
    def compressed_cache_hnd(self) -> torch.Tensor:
        return self.compressed_cache.unsqueeze(1)


@dataclasses.dataclass
class BackendRunners:
    raw: Callable[[], None]
    public: Callable[[], None]
    split_kv: int | None


def _configure_source_checkout() -> None:
    """Point JIT source paths at this checkout instead of wheel package data."""
    from flashinfer.jit import env as jit_env

    root = REPO_ROOT
    jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = root / "include"

    # Prefer the checkout's submodules when they are present. Source archives
    # often omit them, so fall back to the matching headers shipped in an
    # installed FlashInfer package while still compiling this checkout's csrc.
    cutlass = root / "3rdparty" / "cutlass"
    cccl = root / "3rdparty" / "cccl"
    spdlog = root / "3rdparty" / "spdlog" / "include"
    if cutlass.exists() and cccl.exists() and spdlog.exists():
        jit_env.CUTLASS_INCLUDE_DIRS = [
            cutlass / "include",
            cutlass / "tools" / "util" / "include",
        ]
        jit_env.SPDLOG_INCLUDE_DIR = spdlog
        jit_env.CCCL_INCLUDE_DIRS = [
            cccl / "cub",
            cccl / "libcudacxx" / "include",
            cccl / "thrust",
        ]
        return

    package_data = next(
        (
            Path(site_root) / "flashinfer" / "data"
            for site_root in site.getsitepackages()
            if (
                Path(site_root) / "flashinfer" / "data" / "cutlass" / "include"
            ).is_dir()
        ),
        None,
    )
    if package_data is None:
        raise RuntimeError(
            "FlashInfer third-party headers are unavailable: initialize 3rdparty "
            "submodules or install a FlashInfer wheel in this Python environment"
        )
    jit_env.FLASHINFER_DATA = package_data
    jit_env.CUTLASS_INCLUDE_DIRS = [
        package_data / "cutlass" / "include",
        package_data / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        package_data / "cccl" / "cub",
        package_data / "cccl" / "libcudacxx" / "include",
        package_data / "cccl" / "thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = package_data / "spdlog" / "include"


def _expand_page_table_to_token_rows(
    page_table: torch.Tensor, page_size: int
) -> torch.Tensor:
    offsets = torch.arange(page_size, dtype=torch.int32, device=page_table.device)
    return (page_table.unsqueeze(-1) * page_size + offsets).flatten(1)


def make_inputs(case: HcaCase, workspace_size: int) -> HcaInputs:
    if case.raw_seq_len < HCA_COMPRESS_RATIO:
        raise ValueError("raw_seq_len must contain at least one compressed HCA slot")
    if case.raw_seq_len % HCA_COMPRESS_RATIO != 0:
        raise ValueError("raw_seq_len must be divisible by the HCA compression ratio")
    if case.q_len <= 0 or case.q_len > case.raw_seq_len:
        raise ValueError("q_len must be in [1, raw_seq_len]")
    if WINDOW_CAPACITY % case.window_page_size != 0:
        raise ValueError("window_page_size must divide the 128-slot window")

    device = torch.device("cuda")
    rows = case.batch_size * case.q_len
    window_pages_per_row = WINDOW_CAPACITY // case.window_page_size
    compressed_pages_per_row = case.compressed_pages_per_row

    # Fill FP8 tensors in place to avoid long-context FP32 staging allocations.
    # Token-varying values in dimension zero make the cross-backend comparison
    # sensitive to selecting the wrong physical rows or KV pool.
    query = torch.full(
        (case.batch_size, case.q_len, case.num_heads, HEAD_DIM),
        0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    window_cache = torch.full(
        (
            rows * window_pages_per_row,
            case.window_page_size,
            HEAD_DIM,
        ),
        0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    compressed_cache = torch.full(
        (
            case.batch_size * compressed_pages_per_row,
            case.compressed_page_size,
            HEAD_DIM,
        ),
        0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    window_cache.view(-1, HEAD_DIM)[:, 0].copy_(
        torch.linspace(0.25, 1.0, WINDOW_CAPACITY, device=device).repeat(rows)
    )
    compressed_cache.view(-1, HEAD_DIM)[:, 0].copy_(
        torch.linspace(-1.0, -0.25, case.compressed_capacity, device=device).repeat(
            case.batch_size
        )
    )
    if case.page_layout == "shuffled":
        window_tables = torch.randperm(
            rows * window_pages_per_row, dtype=torch.int32, device=device
        ).view(rows, window_pages_per_row)
        compressed_tables_per_batch = torch.randperm(
            case.batch_size * compressed_pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(case.batch_size, compressed_pages_per_row)
    elif case.page_layout == "contiguous":
        window_tables = torch.arange(
            rows * window_pages_per_row, dtype=torch.int32, device=device
        ).view(rows, window_pages_per_row)
        compressed_tables_per_batch = torch.arange(
            case.batch_size * compressed_pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(case.batch_size, compressed_pages_per_row)
    else:
        raise ValueError(f"Unsupported page layout: {case.page_layout}")
    compressed_tables = compressed_tables_per_batch.repeat_interleave(
        case.q_len, dim=0
    ).contiguous()

    window_indices = _expand_page_table_to_token_rows(
        window_tables, case.window_page_size
    )
    compressed_indices = _expand_page_table_to_token_rows(
        compressed_tables, case.compressed_page_size
    )
    trt_compressed_capacity = case.trt_index_capacity - WINDOW_CAPACITY
    sparse_indices = torch.cat(
        (window_indices, compressed_indices[:, :trt_compressed_capacity]), dim=1
    ).contiguous()

    seq_lens = torch.full(
        (case.batch_size,), case.raw_seq_len, dtype=torch.int32, device=device
    )
    hca_seq_lens = torch.full(
        (case.batch_size,), case.active_hca_len, dtype=torch.int32, device=device
    )
    query_positions = torch.arange(case.q_len, dtype=torch.int32, device=device).repeat(
        case.batch_size
    )
    visible_raw_lens = case.raw_seq_len - case.q_len + query_positions + 1
    sparse_topk_lens = (
        WINDOW_CAPACITY + visible_raw_lens // HCA_COMPRESS_RATIO
    ).contiguous()
    window_valid_lens = visible_raw_lens.clamp(max=WINDOW_CAPACITY).contiguous()
    window_columns = torch.arange(WINDOW_CAPACITY, device=device)
    sparse_indices[:, :WINDOW_CAPACITY].masked_fill_(
        window_columns.unsqueeze(0) >= window_valid_lens.unsqueeze(1), -1
    )
    compressed_columns = torch.arange(trt_compressed_capacity, device=device)
    sparse_indices[:, WINDOW_CAPACITY:].masked_fill_(
        compressed_columns.unsqueeze(0)
        >= (sparse_topk_lens - WINDOW_CAPACITY).unsqueeze(1),
        0,
    )
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    output_shape = (case.batch_size, case.q_len, case.num_heads, HEAD_DIM)
    out_trtllm = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
    out_dkg = torch.empty_like(out_trtllm)
    return HcaInputs(
        case=case,
        query=query,
        window_cache=window_cache,
        compressed_cache=compressed_cache,
        window_tables=window_tables,
        compressed_tables=compressed_tables,
        sparse_indices=sparse_indices,
        seq_lens=seq_lens,
        hca_seq_lens=hca_seq_lens,
        sparse_topk_lens=sparse_topk_lens,
        window_valid_lens=window_valid_lens,
        workspace=workspace,
        out_trtllm=out_trtllm,
        out_dkg=out_dkg,
    )


def make_dkg_runners(inputs: HcaInputs) -> BackendRunners:
    from cutlass import Float32, Int32
    from flashinfer.cute_dsl.attention.wrappers import batch_hca
    from flashinfer.cute_dsl.utils import get_max_active_clusters
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    case = inputs.case
    if not case.dkg_persistent:
        batch_hca._check_nonpersistent_grid(case.batch_size, case.q_len)
    max_active_blocks = get_max_active_clusters(2) * 2
    split_kv, required_workspace = batch_hca._get_split_kv_and_workspace_size(
        case.batch_size,
        case.q_len,
        case.num_heads,
        case.hca_capacity,
        max_active_blocks,
    )
    if required_workspace > inputs.workspace.numel():
        raise RuntimeError(
            f"DKG workspace requires {required_workspace} bytes, "
            f"but only {inputs.workspace.numel()} bytes were allocated"
        )
    workspace = (
        None
        if required_workspace == 0
        else inputs.workspace[:required_workspace].view(torch.int8)
    )
    compiled = batch_hca._compile_hca_kernel(
        case.compressed_page_size,
        case.window_page_size,
        case.q_len,
        True,
        case.dkg_persistent,
        required_workspace == 0,
        torch.cuda.get_device_capability(),
    )
    lse = torch.empty(
        (case.batch_size, case.q_len, case.num_heads),
        dtype=torch.float32,
        device=inputs.query.device,
    )
    sink_unscaled = torch.full(
        (128,), -float("inf"), dtype=torch.float32, device=inputs.query.device
    )
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    def run_raw() -> None:
        compiled(
            inputs.query,
            inputs.window_cache,
            inputs.compressed_cache,
            inputs.window_tables,
            inputs.compressed_tables,
            inputs.out_dkg,
            lse,
            workspace,
            Int32(split_kv),
            inputs.hca_seq_lens,
            None,
            inputs.sparse_topk_lens,
            inputs.window_valid_lens,
            Float32(softmax_scale),
            Float32(1.0),
            sink_unscaled,
        )

    def run_public() -> None:
        trtllm_batch_decode_sparse_mla_dsv4(
            query=inputs.query,
            swa_kv_cache=inputs.window_cache_hnd,
            workspace_buffer=inputs.workspace,
            sparse_indices=None,
            compressed_kv_cache=inputs.compressed_cache_hnd,
            sparse_topk_lens=inputs.sparse_topk_lens,
            out=inputs.out_dkg,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            sinks=None,
            kv_layout="HND",
            swa_topk_lens=inputs.window_valid_lens,
            backend="cute-dsl",
            hca_swa_block_tables=inputs.window_tables,
            hca_compressed_block_tables=inputs.compressed_tables,
            hca_seq_lens=inputs.hca_seq_lens,
            hca_use_persistent=case.dkg_persistent,
        )

    # Force compilation and all lazy runtime initialization before timing.
    run_raw()
    torch.cuda.synchronize()
    return BackendRunners(raw=run_raw, public=run_public, split_kv=split_kv)


def make_trtllm_runners(inputs: HcaInputs) -> BackendRunners:
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
    from flashinfer.mla._core import (
        _get_trtllm_gen_multi_ctas_kv_counter_buffer,
        get_trtllm_gen_fmha_module,
    )
    from flashinfer.utils import get_device_sm_count

    case = inputs.case
    module = get_trtllm_gen_fmha_module()
    run_op = module.trtllm_paged_attention_decode_sparse_mla_dsv4
    sm_count = get_device_sm_count(inputs.query.device)
    counters = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
        case.batch_size, case.num_heads, sm_count, inputs.query.device
    )
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    query_flat = inputs.query.flatten(0, 1)

    def run_raw() -> None:
        run_op(
            inputs.out_trtllm,
            query_flat,
            inputs.compressed_cache_hnd,
            inputs.window_cache_hnd,
            inputs.workspace,
            counters,
            inputs.sparse_indices,
            inputs.seq_lens,
            inputs.sparse_topk_lens,
            softmax_scale,
            1.0,
            case.batch_size,
            case.q_len,
            sm_count,
            False,
            inputs.workspace.numel(),
            None,
            None,
        )

    def run_public() -> None:
        trtllm_batch_decode_sparse_mla_dsv4(
            query=inputs.query,
            swa_kv_cache=inputs.window_cache_hnd,
            workspace_buffer=inputs.workspace,
            sparse_indices=inputs.sparse_indices,
            compressed_kv_cache=inputs.compressed_cache_hnd,
            sparse_topk_lens=inputs.sparse_topk_lens,
            seq_lens=inputs.seq_lens,
            out=inputs.out_trtllm,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            sinks=None,
            kv_layout="HND",
            enable_pdl=False,
            backend="trtllm-gen",
        )

    # Force JIT launcher compilation, cubin loading, and lazy buffers before timing.
    run_raw()
    torch.cuda.synchronize()
    return BackendRunners(raw=run_raw, public=run_public, split_kv=None)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_times(times_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(value) for value in times_ms)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "latency_us": statistics.median(values) * 1e3,
        "p10_us": _percentile(values, 0.10) * 1e3,
        "p90_us": _percentile(values, 0.90) * 1e3,
        "cv_percent": 0.0 if mean == 0.0 else std / mean * 100.0,
    }


def benchmark_runner(
    runner: Callable[[], None],
    warmup: int,
    iterations: int,
    cold_l2: bool,
    timer: str,
) -> dict[str, float]:
    from flashinfer.testing import (
        bench_gpu_time_with_cuda_event,
        bench_gpu_time_with_cudagraph,
    )

    if timer == "graph":
        if cold_l2:
            raise ValueError("The graph timer currently supports hot L2 only")
        times = bench_gpu_time_with_cudagraph(
            runner,
            dry_run_iters=warmup,
            repeat_iters=iterations,
            num_iters_within_graph=10,
            cold_l2_cache=False,
        )
    else:
        times = bench_gpu_time_with_cuda_event(
            runner,
            dry_run_iters=warmup,
            repeat_iters=iterations,
            cold_l2_cache=cold_l2,
        )
    return summarize_times(times)


def add_derived_metrics(row: dict[str, object], case: HcaCase) -> None:
    latency_s = float(row["latency_us"]) * 1e-6
    visible_raw_lens = range(case.raw_seq_len - case.q_len + 1, case.raw_seq_len + 1)
    logical_active_slots_per_batch = sum(
        min(visible, WINDOW_CAPACITY) + visible // HCA_COMPRESS_RATIO
        for visible in visible_raw_lens
    )
    encoded_active_slots_per_batch = sum(
        WINDOW_CAPACITY + visible // HCA_COMPRESS_RATIO
        for visible in range(case.raw_seq_len - case.q_len + 1, case.raw_seq_len + 1)
    )
    logical_active_slots = case.batch_size * logical_active_slots_per_batch
    logical_flops = 4.0 * case.num_heads * HEAD_DIM * logical_active_slots
    row["logical_active_slots"] = logical_active_slots
    row["encoded_active_slots"] = case.batch_size * encoded_active_slots_per_batch
    row["query_tokens_per_s"] = case.batch_size * case.q_len / latency_s
    row["logical_tflops"] = logical_flops / latency_s / 1e12
    row["trtllm_metadata_bytes"] = (
        case.batch_size * case.q_len * case.trt_index_capacity * 4
    )
    row["dkg_metadata_bytes"] = (
        case.batch_size
        * case.q_len
        * (WINDOW_CAPACITY // case.window_page_size + case.compressed_pages_per_row)
        * 4
    )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--raw-seq-lens", default="8192,32768,131072")
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument(
        "--page-layout",
        choices=("contiguous", "shuffled"),
        default="shuffled",
    )
    parser.add_argument(
        "--trt-index-layout",
        choices=("native", "page-capacity"),
        default="native",
        help="Use TRT's minimal multiple-of-4 topK or DKG's padded page capacity.",
    )
    parser.add_argument(
        "--dkg-scheduler",
        choices=("persistent", "nonpersistent"),
        default="nonpersistent",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument(
        "--timing-levels",
        choices=("raw", "public", "both"),
        default="both",
    )
    parser.add_argument(
        "--cache-modes",
        choices=("hot", "cold", "both"),
        default="both",
    )
    parser.add_argument(
        "--timer",
        choices=("event", "graph"),
        default="event",
        help="Event includes dispatch gaps; graph amortizes launch overhead.",
    )
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    raw_seq_lens = parse_int_list(args.raw_seq_lens)
    if not batch_sizes or not raw_seq_lens:
        parser.error("--batch-sizes and --raw-seq-lens must not be empty")
    if any(value <= 0 for value in (*batch_sizes, *raw_seq_lens)):
        parser.error("batch sizes and raw sequence lengths must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("This comparison requires SM100 or SM103")
    if args.num_heads != 128:
        parser.error("DeepSeek V4 HCA requires --num-heads 128")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations must be positive")

    _configure_source_checkout()
    import flashinfer

    flashinfer_path = Path(flashinfer.__file__).resolve()
    if not flashinfer_path.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"Expected FlashInfer from {REPO_ROOT}, imported {flashinfer_path}"
        )
    torch.manual_seed(2026)
    rows: list[dict[str, object]] = []
    timing_levels = (
        ("raw", "public") if args.timing_levels == "both" else (args.timing_levels,)
    )
    cache_modes = ("hot", "cold") if args.cache_modes == "both" else (args.cache_modes,)
    if args.timer == "graph" and cache_modes != ("hot",):
        parser.error("--timer graph requires --cache-modes hot")

    print(
        f"GPU={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"L2={torch.cuda.get_device_properties(0).L2_cache_size / (1 << 20):.1f} MiB "
        f"flashinfer={flashinfer_path}"
    )
    case_index = 0
    for raw_seq_len in raw_seq_lens:
        for batch_size in batch_sizes:
            case = HcaCase(
                batch_size=batch_size,
                raw_seq_len=raw_seq_len,
                num_heads=args.num_heads,
                q_len=args.q_len,
                page_layout=args.page_layout,
                trt_index_layout=args.trt_index_layout,
                dkg_persistent=args.dkg_scheduler == "persistent",
            )
            print(
                f"Preparing B={batch_size} Q={case.q_len} raw_seq={raw_seq_len} "
                f"active_hca={case.active_hca_len} capacity={case.hca_capacity}",
                flush=True,
            )
            inputs = make_inputs(case, args.workspace_mb << 20)
            dkg = make_dkg_runners(inputs)
            trtllm = make_trtllm_runners(inputs)
            dkg.raw()
            trtllm.raw()
            torch.cuda.synchronize()
            raw_max_abs_diff = float(
                (inputs.out_dkg.float() - inputs.out_trtllm.float()).abs().max().item()
            )
            dkg.public()
            trtllm.public()
            torch.cuda.synchronize()
            public_max_abs_diff = float(
                (inputs.out_dkg.float() - inputs.out_trtllm.float()).abs().max().item()
            )
            max_abs_diff = max(raw_max_abs_diff, public_max_abs_diff)
            if not torch.isfinite(inputs.out_dkg).all().item():
                raise RuntimeError("DKG produced non-finite output")
            if not torch.isfinite(inputs.out_trtllm).all().item():
                raise RuntimeError("TRTLLM-GEN produced non-finite output")
            if max_abs_diff > 0.13:
                raise RuntimeError(
                    f"Backend outputs differ by {max_abs_diff}, exceeding atol=0.13"
                )

            backends = (("dkg", dkg), ("trtllm", trtllm))
            # Alternate order between adjacent cases to reduce clock/thermal bias.
            if case_index % 2:
                backends = tuple(reversed(backends))
            case_index += 1
            for timing_level in timing_levels:
                for cache_mode in cache_modes:
                    for backend_name, backend in backends:
                        runner = getattr(backend, timing_level)
                        stats = benchmark_runner(
                            runner,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            cold_l2=cache_mode == "cold",
                            timer=args.timer,
                        )
                        row: dict[str, object] = {
                            "backend": backend_name,
                            "timing_level": timing_level,
                            "cache_mode": cache_mode,
                            "timer": args.timer,
                            "batch_size": batch_size,
                            "q_len": case.q_len,
                            "num_heads": case.num_heads,
                            "raw_seq_len": raw_seq_len,
                            "compressed_len": case.compressed_len,
                            "active_hca_len": case.active_hca_len,
                            "hca_capacity": case.hca_capacity,
                            "trt_index_capacity": case.trt_index_capacity,
                            "page_layout": case.page_layout,
                            "trt_index_layout": case.trt_index_layout,
                            "dkg_scheduler": args.dkg_scheduler,
                            "split_kv": backend.split_kv,
                            "raw_max_abs_diff": raw_max_abs_diff,
                            "public_max_abs_diff": public_max_abs_diff,
                            "max_abs_diff": max_abs_diff,
                            **stats,
                        }
                        add_derived_metrics(row, case)
                        rows.append(row)
                        print(
                            f"  {backend_name:7s} {timing_level:6s} {cache_mode:4s} "
                            f"median={stats['latency_us']:.2f} us "
                            f"p10={stats['p10_us']:.2f} p90={stats['p90_us']:.2f}",
                            flush=True,
                        )

            del dkg, trtllm, inputs
            torch.cuda.empty_cache()

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
