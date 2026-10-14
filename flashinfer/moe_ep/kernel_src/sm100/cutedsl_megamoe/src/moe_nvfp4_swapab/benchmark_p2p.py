#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Single-file two-GPU raw UBLKCP P2P bandwidth benchmark.

This script intentionally does not import any local project modules.  It uses
PyTorch for allocation/timing, cuda-python for peer access, and CuTeDSL for the
small inline-PTX kernel.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import Int32, Int64, T, Uint8, dsl_user_op


Mode = Literal["pull", "push"]

DEFAULT_BUFFER_MIB = 300
DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 5
NUM_WARPS = 4
THREADS_PER_CTA = NUM_WARPS * 32
TMA_CACHE_HINT_EVICT_NORMAL = 0x1000000000000000
TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def ublkcp_pull_raw(
    dst_smem, src_gmem_addr: Int64, mbar_smem, num_bytes: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem_addr.ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ublkcp_push_raw(
    dst_gmem_addr: Int64, src_smem, num_bytes: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            dst_gmem_addr.ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            Int64(TMA_CACHE_HINT_EVICT_NORMAL).ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
        "[$0], [$1], $2, $3;",
        "l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_u32(smem_ptr, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def read_clock64(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def ublkcp_bench_kernel(
    local_sink: cute.Tensor,
    clock_stats: cute.Tensor,
    remote_base_ptr: cute.Pointer,
    base_offset: Int32,
    iters: Int32,
    mode: cutlass.Constexpr[str],
    pull_clock_stats: cutlass.Constexpr[bool],
    push_clock_stats: cutlass.Constexpr[bool],
    x_ctas: cutlass.Constexpr[int],
    y_copy_per_iter: cutlass.Constexpr[int],
    z_bytes_per_inst: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(tidx // Int32(32))
    lane_idx = cute.arch.lane_idx()

    smem = cutlass.utils.SmemAllocator()
    pull_mbar = smem.allocate_array(Int64, NUM_WARPS)
    payload = smem.allocate_array(Uint8, NUM_WARPS * y_copy_per_iter * z_bytes_per_inst)

    if cutlass.const_expr(mode == "pull"):
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar + warp_idx, 1)
        cute.arch.sync_warp()

    remote_base_addr = remote_base_ptr.toint()
    clock_min_cycles = Int64(0x7FFFFFFFFFFFFFFF)
    clock_max_cycles = Int64(0)
    phase = Int32(0)
    iter_idx = Int32(0)
    while iter_idx < iters:
        iter_t0 = Int64(0)
        if cutlass.const_expr(
            (mode == "push" and push_clock_stats)
            or (mode == "pull" and pull_clock_stats)
        ):
            if lane_idx == Int32(0):
                iter_t0 = read_clock64()

        for u in cutlass.range_constexpr(0, y_copy_per_iter, 1):
            copy_linear = (
                (Int64(iter_idx) * Int64(x_ctas) + Int64(bidx)) * Int64(NUM_WARPS)
                + Int64(warp_idx)
            ) * Int64(y_copy_per_iter) + Int64(u)
            remote_addr = (
                remote_base_addr
                + Int64(base_offset)
                + copy_linear * Int64(z_bytes_per_inst)
            )
            smem_off = (warp_idx * Int32(y_copy_per_iter) + Int32(u)) * Int32(
                z_bytes_per_inst
            )
            smem_ptr = payload + smem_off
            with cute.arch.elect_one():
                if cutlass.const_expr(mode == "pull"):
                    ublkcp_pull_raw(
                        smem_ptr,
                        remote_addr,
                        pull_mbar + warp_idx,
                        Int32(z_bytes_per_inst),
                    )
                else:
                    ublkcp_push_raw(remote_addr, smem_ptr, Int32(z_bytes_per_inst))

        if cutlass.const_expr(mode == "pull"):
            if lane_idx == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    pull_mbar + warp_idx,
                    Int32(y_copy_per_iter * z_bytes_per_inst),
                )
                cute.arch.mbarrier_wait(pull_mbar + warp_idx, phase)
                if cutlass.const_expr(pull_clock_stats):
                    pull_delta = read_clock64() - iter_t0
                    clock_min_cycles = cutlass.min(clock_min_cycles, pull_delta)
                    clock_max_cycles = cutlass.max(clock_max_cycles, pull_delta)
                sample = ld_shared_u32(
                    payload + warp_idx * Int32(y_copy_per_iter * z_bytes_per_inst)
                )
                sink_idx = (iter_idx * Int32(x_ctas) + Int32(bidx)) * Int32(
                    NUM_WARPS
                ) + warp_idx
                local_sink[sink_idx] = sample
        else:
            if lane_idx == Int32(0):
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                if cutlass.const_expr(push_clock_stats):
                    push_delta = read_clock64() - iter_t0
                    clock_min_cycles = cutlass.min(clock_min_cycles, push_delta)
                    clock_max_cycles = cutlass.max(clock_max_cycles, push_delta)
        cute.arch.sync_warp()

        if cutlass.const_expr(mode == "pull"):
            phase = phase ^ Int32(1)
        iter_idx = iter_idx + Int32(1)

    if cutlass.const_expr(
        (mode == "push" and push_clock_stats) or (mode == "pull" and pull_clock_stats)
    ):
        if lane_idx == Int32(0):
            stats_base = (Int32(bidx) * Int32(NUM_WARPS) + warp_idx) * Int32(2)
            clock_stats[stats_base] = clock_min_cycles
            clock_stats[stats_base + Int32(1)] = clock_max_cycles

    if cutlass.const_expr(mode == "push"):
        if tidx == Int32(0):
            cute.arch.fence_acq_rel_sys()


@cute.jit
def launch_ublkcp_bench(
    local_sink: cute.Tensor,
    clock_stats: cute.Tensor,
    remote_base_ptr: cute.Pointer,
    base_offset: Int32,
    iters: Int32,
    mode: cutlass.Constexpr[str],
    pull_clock_stats: cutlass.Constexpr[bool],
    push_clock_stats: cutlass.Constexpr[bool],
    x_ctas: cutlass.Constexpr[int],
    y_copy_per_iter: cutlass.Constexpr[int],
    z_bytes_per_inst: cutlass.Constexpr[int],
):
    ublkcp_bench_kernel(
        local_sink,
        clock_stats,
        remote_base_ptr,
        base_offset,
        iters,
        mode,
        pull_clock_stats,
        push_clock_stats,
        x_ctas,
        y_copy_per_iter,
        z_bytes_per_inst,
    ).launch(grid=[x_ctas, 1, 1], block=[THREADS_PER_CTA, 1, 1])


@dataclass(frozen=True)
class BenchConfig:
    mode: Mode
    x_ctas: int
    y_copy_per_iter: int
    z_bytes_per_inst: int
    w_comm_mbytes: int
    pull_clock_stats: bool
    push_clock_stats: bool


@dataclass
class BenchResult:
    mode: Mode
    x_ctas: int
    y_copy_per_iter: int
    z_bytes_per_inst: int
    w_comm_mbytes: int
    iters: int
    requested_bytes: int
    actual_bytes: int
    ms_min: float
    ms_median: float
    ms_mean: float
    gbps_min_time: float
    gbps_median_time: float
    request_128b_count: int
    request_128b_ns_min_time: float
    request_128b_ns_median_time: float
    shared_bytes_per_cta: int
    buffer_mib: int
    skipped: str = ""


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"empty integer list: {text!r}")
    return values


def default_x_ctas(sm_count: int) -> str:
    candidates = [1, 2, 4, 8, 16, 32, 64, 96, 128, sm_count]
    values = []
    for value in candidates:
        if value <= sm_count and value not in values:
            values.append(value)
    return ",".join(str(v) for v in values)


def mib_to_bytes(value: int) -> int:
    return value * 1024 * 1024


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def import_cudart():
    try:
        from cuda.bindings import runtime as cudart  # type: ignore

        return cudart
    except Exception:
        from cuda import cudart  # type: ignore

        return cudart


def check_cuda_error(cudart, result, what: str):
    err = result[0] if isinstance(result, tuple) else result
    success = cudart.cudaError_t.cudaSuccess
    if err != success:
        raise RuntimeError(f"{what} failed: {err}")
    return result[1:] if isinstance(result, tuple) else ()


def enable_peer_access_0_to_1() -> None:
    cudart = import_cudart()
    check_cuda_error(cudart, cudart.cudaSetDevice(0), "cudaSetDevice(0)")
    can_access = check_cuda_error(
        cudart,
        cudart.cudaDeviceCanAccessPeer(0, 1),
        "cudaDeviceCanAccessPeer(0, 1)",
    )[0]
    if int(can_access) == 0:
        raise RuntimeError("device 0 cannot access peer device 1")

    result = cudart.cudaDeviceEnablePeerAccess(1, 0)
    err = result[0] if isinstance(result, tuple) else result
    already = getattr(cudart.cudaError_t, "cudaErrorPeerAccessAlreadyEnabled", None)
    if err != cudart.cudaError_t.cudaSuccess and err != already:
        raise RuntimeError(f"cudaDeviceEnablePeerAccess(1) failed: {err}")


def pointer_owner_device(ptr: int) -> int | None:
    cudart = import_cudart()
    result = cudart.cudaPointerGetAttributes(ptr)
    try:
        attrs = check_cuda_error(cudart, result, "cudaPointerGetAttributes")[0]
    except Exception:
        return None
    return getattr(attrs, "device", None)


def smem_capacity_bytes() -> int:
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    try:
        return int(cutlass.utils.get_smem_capacity_in_bytes(arch))
    except Exception:
        if major >= 10:
            return 227 * 1024
        return 99 * 1024


def validate_environment() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"this benchmark requires exactly 2 CUDA devices, got {torch.cuda.device_count()}"
        )
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 10:
        raise RuntimeError("UBLKCP benchmark expects a Blackwell-class GPU (sm_100+)")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    enable_peer_access_0_to_1()
    return sm_count


def make_arg_parser(sm_count: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x_ctas", default=default_x_ctas(sm_count))
    parser.add_argument("--y_copy_per_iter", default="1,2,4,8")
    parser.add_argument("--z_bytes_per_inst", default="128,256,512,1024,2048")
    parser.add_argument("--w_comm_mbytes", default="256")
    parser.add_argument("--mode", choices=("pull", "push", "all"), default="all")
    parser.add_argument(
        "--pull_clock_stats",
        action="store_true",
        help=(
            "For pull, record per-warp min/max cycles from issuing the first "
            "UBLKCP.S.G in an iter to mbarrier_wait completion."
        ),
    )
    parser.add_argument(
        "--push_clock_stats",
        action="store_true",
        help=(
            "For push, record per-warp min/max cycles from issuing the first "
            "UBLKCP.G.S in an iter to cp_async_bulk_wait_group(0)."
        ),
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    return parser


def iter_configs(args: argparse.Namespace) -> Iterable[BenchConfig]:
    modes: list[Mode] = ["pull", "push"] if args.mode == "all" else [args.mode]
    for mode in modes:
        for x in parse_int_list(args.x_ctas):
            for y in parse_int_list(args.y_copy_per_iter):
                for z in parse_int_list(args.z_bytes_per_inst):
                    for w in parse_int_list(args.w_comm_mbytes):
                        yield BenchConfig(
                            mode,
                            x,
                            y,
                            z,
                            w,
                            args.pull_clock_stats,
                            args.push_clock_stats,
                        )


def compile_config(
    config: BenchConfig, local_sink, clock_stats, remote_base_ptr, iters: int
):
    @cute.jit
    def launcher(
        local_sink: cute.Tensor,
        clock_stats: cute.Tensor,
        remote_base_ptr: cute.Pointer,
        base_offset: Int32,
    ):
        launch_ublkcp_bench(
            local_sink,
            clock_stats,
            remote_base_ptr,
            base_offset,
            iters,
            config.mode,
            config.pull_clock_stats,
            config.push_clock_stats,
            config.x_ctas,
            config.y_copy_per_iter,
            config.z_bytes_per_inst,
        )

    return cute.compile(
        launcher,
        local_sink,
        clock_stats,
        remote_base_ptr,
        0,
    )


def run_one_config(
    config: BenchConfig,
    sm_count: int,
    smem_capacity: int,
    remote_buf: torch.Tensor,
) -> BenchResult:
    if config.x_ctas <= 0 or config.x_ctas > sm_count:
        return skipped_result(config, f"x_ctas must be in [1, {sm_count}]")
    if config.y_copy_per_iter <= 0:
        return skipped_result(config, "y_copy_per_iter must be positive")
    if config.z_bytes_per_inst <= 0 or config.z_bytes_per_inst % 128 != 0:
        return skipped_result(
            config, "z_bytes_per_inst must be a positive multiple of 128"
        )
    if config.w_comm_mbytes <= 0:
        return skipped_result(config, "w_comm_mbytes must be positive")

    bytes_per_iter = (
        config.x_ctas * NUM_WARPS * config.y_copy_per_iter * config.z_bytes_per_inst
    )
    requested_bytes = mib_to_bytes(config.w_comm_mbytes)
    iters = max(1, math.ceil(requested_bytes / bytes_per_iter))
    actual_bytes = iters * bytes_per_iter
    shared_bytes = NUM_WARPS * config.y_copy_per_iter * config.z_bytes_per_inst
    shared_bytes_with_mbar = round_up(NUM_WARPS * 8, 16) + shared_bytes
    if shared_bytes_with_mbar > smem_capacity:
        return skipped_result(
            config,
            f"shared memory required {shared_bytes_with_mbar} B > capacity {smem_capacity} B",
            iters=iters,
            requested_bytes=requested_bytes,
            actual_bytes=actual_bytes,
            shared_bytes=shared_bytes_with_mbar,
        )

    buffer_bytes = max(mib_to_bytes(DEFAULT_BUFFER_MIB), actual_bytes)
    buffer_bytes = round_up(buffer_bytes, 128)
    buffer_mib = math.ceil(buffer_bytes / (1024 * 1024))

    torch.cuda.set_device(0)
    local_sink_t = torch.empty(
        (config.x_ctas * NUM_WARPS * iters,), dtype=torch.int32, device="cuda"
    )
    local_sink = from_dlpack(local_sink_t).mark_layout_dynamic()
    clock_stats_t = torch.empty(
        (config.x_ctas * NUM_WARPS * 2,), dtype=torch.int64, device="cuda"
    )
    clock_stats = from_dlpack(clock_stats_t).mark_layout_dynamic()

    if remote_buf.numel() < buffer_bytes:
        raise RuntimeError(
            f"remote buffer has {remote_buf.numel()} bytes, need {buffer_bytes}; "
            "internal allocation bug"
        )
    remote_ptr = int(remote_buf.data_ptr())
    if remote_ptr % 128 != 0:
        raise RuntimeError(
            f"remote buffer pointer is not 128B aligned: 0x{remote_ptr:x}"
        )
    owner = pointer_owner_device(remote_ptr)
    if owner is not None and int(owner) != 1:
        raise RuntimeError(f"remote pointer owner device is {owner}, expected 1")
    remote_base_ptr = make_ptr(
        Uint8, remote_ptr, cute.AddressSpace.gmem, assumed_align=128
    )

    print(
        f"compile mode={config.mode} x={config.x_ctas} y={config.y_copy_per_iter} "
        f"z={config.z_bytes_per_inst} w={config.w_comm_mbytes}MiB iters={iters}"
    )
    compiled = compile_config(config, local_sink, clock_stats, remote_base_ptr, iters)

    def run_once(offset: int) -> float:
        local_sink_t.zero_()
        if (config.mode == "push" and config.push_clock_stats) or (
            config.mode == "pull" and config.pull_clock_stats
        ):
            clock_stats_t.zero_()
        torch.cuda.synchronize(0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(
            local_sink,
            clock_stats,
            remote_base_ptr,
            offset,
        )
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    max_offset = max(0, buffer_bytes - actual_bytes)
    offset_step = round_up(actual_bytes, 128)
    for i in range(DEFAULT_WARMUPS):
        offset = 0 if max_offset == 0 else (i * offset_step) % (max_offset + 1)
        offset -= offset % 128
        run_once(offset)

    timings = []
    for i in range(DEFAULT_REPEATS):
        raw_offset = (DEFAULT_WARMUPS + i) * offset_step
        offset = 0 if max_offset == 0 else raw_offset % (max_offset + 1)
        offset -= offset % 128
        timings.append(run_once(offset))

    ms_min = min(timings)
    ms_median = statistics.median(timings)
    ms_mean = statistics.mean(timings)
    if config.mode == "push" and config.push_clock_stats:
        print_clock_stats(clock_stats_t, "push_smem_reuse_wait_cycles")
    if config.mode == "pull" and config.pull_clock_stats:
        print_clock_stats(clock_stats_t, "pull_data_arrival_wait_cycles")
    return BenchResult(
        mode=config.mode,
        x_ctas=config.x_ctas,
        y_copy_per_iter=config.y_copy_per_iter,
        z_bytes_per_inst=config.z_bytes_per_inst,
        w_comm_mbytes=config.w_comm_mbytes,
        iters=iters,
        requested_bytes=requested_bytes,
        actual_bytes=actual_bytes,
        ms_min=ms_min,
        ms_median=ms_median,
        ms_mean=ms_mean,
        gbps_min_time=actual_bytes / (ms_min * 1.0e6),
        gbps_median_time=actual_bytes / (ms_median * 1.0e6),
        request_128b_count=actual_bytes // 128,
        request_128b_ns_min_time=ms_min * 1.0e6 / (actual_bytes // 128),
        request_128b_ns_median_time=ms_median * 1.0e6 / (actual_bytes // 128),
        shared_bytes_per_cta=shared_bytes_with_mbar,
        buffer_mib=buffer_mib,
    )


def skipped_result(
    config: BenchConfig,
    reason: str,
    *,
    iters: int = 0,
    requested_bytes: int = 0,
    actual_bytes: int = 0,
    shared_bytes: int = 0,
) -> BenchResult:
    return BenchResult(
        mode=config.mode,
        x_ctas=config.x_ctas,
        y_copy_per_iter=config.y_copy_per_iter,
        z_bytes_per_inst=config.z_bytes_per_inst,
        w_comm_mbytes=config.w_comm_mbytes,
        iters=iters,
        requested_bytes=requested_bytes,
        actual_bytes=actual_bytes,
        ms_min=float("nan"),
        ms_median=float("nan"),
        ms_mean=float("nan"),
        gbps_min_time=float("nan"),
        gbps_median_time=float("nan"),
        request_128b_count=actual_bytes // 128 if actual_bytes else 0,
        request_128b_ns_min_time=float("nan"),
        request_128b_ns_median_time=float("nan"),
        shared_bytes_per_cta=shared_bytes,
        buffer_mib=0,
        skipped=reason,
    )


def print_result(result: BenchResult) -> None:
    if result.skipped:
        print(
            f"SKIP mode={result.mode} x={result.x_ctas} y={result.y_copy_per_iter} "
            f"z={result.z_bytes_per_inst} w={result.w_comm_mbytes}MiB: {result.skipped}"
        )
        return
    print(
        f"{result.mode:4s} x={result.x_ctas:3d} y={result.y_copy_per_iter:2d} "
        f"z={result.z_bytes_per_inst:5d} w={result.w_comm_mbytes:5d}MiB "
        f"iters={result.iters:6d} min={result.ms_min:8.3f} ms "
        f"median={result.ms_median:8.3f} ms "
        f"BW(min-time)={result.gbps_min_time:8.2f} GB/s "
        f"request_128B={result.request_128b_count} "
        f"request_128B_ns(min/median)="
        f"{result.request_128b_ns_min_time:.3f}/{result.request_128b_ns_median_time:.3f} "
        f"smem={result.shared_bytes_per_cta} B"
    )


def _format_stat_entry(flat_idx: int, cycles: int) -> str:
    cta_warp, slot = divmod(flat_idx, 2)
    cta, warp = divmod(cta_warp, NUM_WARPS)
    name = "min" if slot == 0 else "max"
    return f"(cta={cta}, warp={warp}, {name}={cycles})"


def print_clock_stats(clock_stats_t: torch.Tensor, label: str, topk: int = 10) -> None:
    stats_cpu = clock_stats_t.detach().cpu().reshape(-1)
    stats_3d = clock_stats_t.detach().cpu().reshape(-1, NUM_WARPS, 2)
    mins = stats_3d[:, :, 0].reshape(-1)
    maxs = stats_3d[:, :, 1].reshape(-1)

    min_order = torch.argsort(mins)[:topk].tolist()
    max_order = torch.argsort(maxs, descending=True)[:topk].tolist()

    print(f"  {label}_min_top10:")
    for idx in min_order:
        flat_idx = idx * 2
        print(f"    {_format_stat_entry(flat_idx, int(stats_cpu[flat_idx]))}")

    print(f"  {label}_max_top10:")
    for idx in max_order:
        flat_idx = idx * 2 + 1
        print(f"    {_format_stat_entry(flat_idx, int(stats_cpu[flat_idx]))}")


def write_csv(path: str | Path, results: list[BenchResult]) -> None:
    fieldnames = list(BenchResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def allocate_remote_buffer(configs: list[BenchConfig], sm_count: int) -> torch.Tensor:
    max_actual = 0
    for config in configs:
        if config.x_ctas <= 0 or config.x_ctas > sm_count:
            continue
        if config.y_copy_per_iter <= 0 or config.z_bytes_per_inst <= 0:
            continue
        bytes_per_iter = (
            config.x_ctas * NUM_WARPS * config.y_copy_per_iter * config.z_bytes_per_inst
        )
        requested = mib_to_bytes(config.w_comm_mbytes)
        actual = max(1, math.ceil(requested / bytes_per_iter)) * bytes_per_iter
        max_actual = max(max_actual, actual)
    buffer_bytes = round_up(max(mib_to_bytes(DEFAULT_BUFFER_MIB), max_actual), 128)
    torch.cuda.set_device(1)
    remote = torch.empty((buffer_bytes,), dtype=torch.uint8, device="cuda")
    remote.fill_(0x5A)
    torch.cuda.synchronize(1)
    return remote


def main() -> int:
    sm_count = validate_environment()
    parser = make_arg_parser(sm_count)
    args = parser.parse_args()

    configs = list(iter_configs(args))
    remote_buf = allocate_remote_buffer(configs, sm_count)
    smem_capacity = smem_capacity_bytes()

    print(
        f"device0={torch.cuda.get_device_name(0)!r} device1={torch.cuda.get_device_name(1)!r} "
        f"sm_count={sm_count} smem_capacity={smem_capacity} B "
        f"remote_buffer={remote_buf.numel() / (1024 * 1024):.1f} MiB"
    )

    results = []
    for config in configs:
        result = run_one_config(config, sm_count, smem_capacity, remote_buf)
        results.append(result)
        print_result(result)

    if args.csv:
        write_csv(args.csv, results)
        print(f"wrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
