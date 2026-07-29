# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUPTI benchmark for the six frozen recurrent-KDA prefill contract shapes.

The reported public end-to-end time includes the required same-stream final
state copy-back into ``initial_state``. Input allocation, packed metadata,
sequence ordering, output/state allocation, and JIT/cache warmup are outside
the measured region.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from flashinfer.kda_decode import (
    RecurrentKDAPrefillWorkspace,
    recurrent_kda,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability


@dataclass(frozen=True)
class Case:
    name: str
    num_heads: int
    seq_lens: tuple[int, ...]
    packed: bool


CASES = tuple(
    Case(
        name=f"h{num_heads}_{name}",
        num_heads=num_heads,
        seq_lens=seq_lens,
        packed=packed,
    )
    for num_heads in (96, 64)
    for name, seq_lens, packed in (
        ("fixed8192", (8192,), False),
        ("mixed", (1300, 547, 2048, 963, 271, 3063), True),
        ("uniform", (1024,) * 8, True),
    )
)


def _make_case(
    case: Case,
) -> tuple[
    Callable[[], tuple[torch.Tensor, Optional[torch.Tensor]]],
    dict,
]:
    total_tokens = sum(case.seq_lens)
    shape = (1, total_tokens, case.num_heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (1, total_tokens, case.num_heads),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = 0.1 * torch.randn(case.num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn(
        (case.num_heads, 128), dtype=torch.float32, device="cuda"
    )
    state = torch.zeros(
        (len(case.seq_lens), case.num_heads, 128, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(q)
    workspace = RecurrentKDAPrefillWorkspace(q.device)

    cu_seqlens = None
    seq_order = None
    if case.packed:
        offsets = [0]
        for seq_len in case.seq_lens:
            offsets.append(offsets[-1] + seq_len)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
        seq_order = torch.tensor(
            sorted(
                range(len(case.seq_lens)),
                key=case.seq_lens.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device="cuda",
        )

    def run():
        return recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=state,
            output=output,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            cu_seqlens=cu_seqlens,
            beta_is_logit=True,
            seq_order=seq_order,
            prefill_workspace=workspace,
        )

    metadata = {
        "name": case.name,
        "num_heads": case.num_heads,
        "seq_lens": list(case.seq_lens),
        "total_tokens": total_tokens,
        "layout": "packed" if case.packed else "fixed",
        "variant": "m64" if case.name == "h64_fixed8192" else "m128",
    }
    return run, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optionally write the result list as JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        raise RuntimeError("frozen recurrent-KDA prefill requires B200 (cc 10.0)")

    results = []
    for case in CASES:
        run, result = _make_case(case)
        run()
        torch.cuda.synchronize()
        measurements = bench_gpu_time(
            run,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=args.warmup,
            repeat_iters=args.iters,
        )
        result["median_ms"] = float(np.median(measurements))
        result["median_us"] = result["median_ms"] * 1000.0
        result["timing_scope"] = "public_end_to_end_with_state_copy_back"
        results.append(result)
        print(
            f"{result['name']:<18} {result['variant']:<4} "
            f"{result['median_us']:10.3f} us"
        )

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
