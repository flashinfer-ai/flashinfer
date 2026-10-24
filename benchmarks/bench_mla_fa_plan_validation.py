"""Compare single/double FA3 workload validation using host plan() wall time.

Both arms use this checkout. The double arm restores the former second check
through the private backend flag; alternating arm order controls host drift.
GPU synchronization is outside the timed region. Kernels are not timed here.
"""

import argparse
import json
import os
import statistics
import time

import torch

import flashinfer
from flashinfer.mla._batch_mla._backends import _fa_common as fa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--cpu-core", type=int)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        parser.error("iterations and rounds must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    core = args.cpu_core
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        if core is None:
            core = min(allowed)
        if core not in allowed:
            parser.error("CPU core is outside the process affinity")
        os.sched_setaffinity(0, {core})
    elif core is not None:
        parser.error("CPU affinity is unsupported on this platform")
    original_plan = fa._BatchMLAGeneratedFaMechanics._plan_generated_fa
    original_validate = fa._validate_fa_plan_workload
    results = []
    try:
        for rows, live in ((1, 1), (64, 64), (1024, 1024), (8192, 64), (8192, 1024)):
            q = torch.zeros(rows, dtype=torch.int32)
            q[:live] = 1
            lengths = q * 2048
            zero = torch.zeros(1, dtype=torch.int32)
            qo = torch.cat((zero, q.cumsum(0, dtype=torch.int32))).pin_memory()
            kv = torch.cat((zero, lengths.cumsum(0, dtype=torch.int32))).pin_memory()
            lengths = lengths.pin_memory()
            indices = torch.zeros(live * 2048, dtype=torch.int32, device="cuda")
            wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
                backend="fa3",
                use_cuda_graph=True,
                qo_indptr=torch.empty_like(qo, device="cuda"),
                kv_indptr=torch.empty_like(kv, device="cuda"),
                kv_indices=indices,
                kv_len_arr=torch.empty_like(lengths, device="cuda"),
            )
            metadata = flashinfer.mla.MLAPlanMetadata.csr(qo, kv, indices, lengths)
            samples = {"double": [], "single": []}
            calls = [0]

            def counted(*values):
                calls[0] += 1
                return original_validate(*values)

            fa._validate_fa_plan_workload = counted
            for round_index in range(args.rounds):
                order = (
                    ("double", "single")
                    if round_index % 2 == 0
                    else ("single", "double")
                )
                for arm in order:

                    def selected(self, **kwargs):
                        kwargs["_workload_validated"] = arm == "single"
                        return original_plan(self, **kwargs)

                    fa._BatchMLAGeneratedFaMechanics._plan_generated_fa = selected
                    for iteration in range(args.iterations + 5):
                        torch.cuda.synchronize()
                        calls[0] = 0
                        start = time.perf_counter()
                        wrapper.plan(
                            metadata=metadata,
                            num_heads=64,
                            head_dim_ckv=512,
                            head_dim_kpe=0,
                            page_size=1,
                            causal=False,
                            sm_scale=512**-0.5,
                            q_data_type=torch.bfloat16,
                            kv_data_type=torch.bfloat16,
                        )
                        elapsed = (time.perf_counter() - start) * 1000
                        assert calls[0] == (1 if arm == "single" else 2)
                        if iteration >= 5:
                            samples[arm].append(elapsed)
            results.append(
                {
                    "rows": rows,
                    "live": live,
                    "median_ms": {
                        arm: statistics.median(values)
                        for arm, values in samples.items()
                    },
                }
            )
    finally:
        fa._BatchMLAGeneratedFaMechanics._plan_generated_fa = original_plan
        fa._validate_fa_plan_workload = original_validate
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "source": flashinfer.__file__,
                "cpu_core": core,
                "samples_per_arm": args.iterations * args.rounds,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
