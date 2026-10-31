"""Benchmark segmented custom-mask packing and attention planning.

Run on each revision with separate JIT caches. Kernel timings use CUDA graphs;
API/plan timings include allocation and synchronization using wall-clock time.
Neither measurement represents a model forward pass or serving throughput.
"""

import argparse
import json
import random
import statistics
import time

import torch

import flashinfer
from flashinfer.quantization.packbits import get_quantization_module
from flashinfer.testing import bench_gpu_time
from flashinfer.testing.utils import calculate_rotation_count


def wall_time_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(30):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        measurements.append((time.perf_counter() - start) * 1e6)
    return statistics.median(measurements)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--include-plan", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    order = random.Random(args.order_seed)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, **vars(args)
            )
        ),
        flush=True,
    )
    cases = [
        (batch, size, bitorder)
        for batch in [1, 4, 16, 128, 1024]
        for size in [17, 2048, 2049, 65536, 1048576, 16777216]
        if batch * size <= 64 * 1024 * 1024
        for bitorder in ["big", "little"]
    ]
    order.shuffle(cases)
    for case_index, (batch, size, bitorder) in enumerate(cases):
        if args.cold_l2 and batch * size < 1024 * 1024:
            continue
        x = torch.randint(0, 2, (batch * size,), device="cuda", dtype=torch.bool)
        indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * size
        output, output_indptr = flashinfer.quantization.segment_packbits(
            x, indptr, bitorder
        )
        module = get_quantization_module()

        def kernel(data):
            module.segment_packbits(data, indptr, output_indptr, bitorder, output)

        graph_iters = 10
        if args.cold_l2:
            rotations = calculate_rotation_count([x])
            graph_iters = ((graph_iters + rotations - 1) // rotations) * rotations
        measurements = bench_gpu_time(
            kernel,
            input_args=(x,),
            enable_cupti=False,
            use_cuda_graph=True,
            cold_l2_cache=args.cold_l2,
            num_iters_within_graph=graph_iters,
            dry_run_iters=5,
            repeat_iters=30,
        )
        result = dict(
            case_index=case_index,
            batch=batch,
            size=size,
            bitorder=bitorder,
            kernel_us=statistics.median(measurements) * 1000,
        )
        if not args.cold_l2:
            result["api_us"] = wall_time_us(
                lambda: flashinfer.quantization.segment_packbits(x, indptr, bitorder)
            )
        print(json.dumps(result), flush=True)

    if args.include_plan:
        cases = [(1, 128, 32768), (1, 4096, 4096), (4, 128, 32768), (16, 128, 8192)]
        order.shuffle(cases)
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        for batch, qo_len, kv_len in cases:
            q_indptr = torch.arange(batch + 1, dtype=torch.int32) * qo_len
            kv_indptr = torch.arange(batch + 1, dtype=torch.int32) * kv_len
            mask = torch.randint(
                0, 2, (batch * qo_len * kv_len,), device="cuda", dtype=torch.bool
            )
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, backend="fa2"
            )

            def plan():
                wrapper.plan(
                    q_indptr,
                    kv_indptr,
                    8,
                    2,
                    128,
                    custom_mask=mask,
                    q_data_type=torch.float16,
                )

            print(
                json.dumps(
                    dict(
                        stage="plan",
                        batch=batch,
                        qo_len=qo_len,
                        kv_len=kv_len,
                        plan_us=wall_time_us(plan),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
