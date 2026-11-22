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

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import flashinfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    results = []
    for index_dtype in (torch.int32, torch.int64):
        for rows in (591, 4728, 23640):
            q_ptr = torch.arange(rows + 1, dtype=index_dtype) * 128
            kv_ptr = torch.arange(rows + 1, dtype=index_dtype) * 7560
            workspace = torch.empty(
                128 * 1024 * 1024,
                device="cuda",
                dtype=torch.uint8,
            )
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                backend="fa2",
            )

            def plan(attention_wrapper):
                attention_wrapper.plan(
                    q_ptr,
                    kv_ptr,
                    1,
                    1,
                    128,
                    q_data_type=torch.bfloat16,
                )

            for _ in range(5):
                plan(wrapper)
            assert not bool(wrapper._plan_info[-1])
            samples = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                plan(wrapper)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            entry = {
                "index_dtype": str(index_dtype),
                "requests": rows,
                "total_q_tokens": rows * 128,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
            print(entry, flush=True)
            results.append(entry)
            del wrapper, workspace
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "flashinfer": flashinfer.__version__,
                "timing": "synchronized host wall time; excludes JIT and workspace creation",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
