# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GDP decode / MTP verification: ``cake_gated_delta_product_mtp`` with and without
the row-split route, under CUDA graphs.

For every (num_householder, T, heads, batch) the script captures the public call
twice, once with ``FLASHINFER_DISABLE_GDP_DECODE_ROWSPLIT=1`` (every call on the
Cake programs, as before the route) and once without (the route decides), and
reports the kernel time per call and whether the outputs and the state pools are
bitwise equal.  Inputs: K = 128, V = 64, BF16 activations, FP32 ``A_log`` /
``dt_bias``, an FP32 pool of ``B * T + 8`` slots, per-token state scatter (the
MTP-verify call).

Timing: CUDA events around replays of one graph holding one call per input copy;
``--cold`` (default) rotates through enough input copies
(``calculate_rotation_count``: five times the L2 size) that each call reads its
state from HBM, ``--hot`` reuses one input set (the state stays L2-resident when
it fits).  The median of the replays, in microseconds per call.

Example::

    python benchmarks/bench_gdp_decode_rowsplit.py --num-householder 3 \\
        --tokens 4 6 8 --heads 10 40 --batch 1 4 16 64
"""

import argparse
import os
import statistics

import torch

from flashinfer import gdp_decode_rowsplit as rs
from flashinfer.cake_gdp import cake_gated_delta_product_mtp
from flashinfer.testing.utils import calculate_rotation_count

K = 128
V = 64


def make_inputs(n: int, T: int, H: int, B: int, seed: int = 0) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device="cuda")

    pool = B * T + 8
    slots = torch.randperm(pool, generator=gen, device="cuda")[: B * T]
    slots = slots.to(torch.int32).view(B, T).contiguous()
    return dict(
        q=randn(B, T, H, K).bfloat16(),
        k=randn(B, T, n, H, K).bfloat16(),
        v=randn(B, T, n, H, V).bfloat16(),
        initial_state=randn(pool, H, V, K).mul_(0.05),
        initial_state_indices=slots[:, -1].contiguous(),
        A_log=torch.rand(H, generator=gen, device="cuda").mul_(2.7),
        a=randn(B, T, H).mul_(2.0).bfloat16(),
        dt_bias=randn(H).mul_(0.5),
        b=randn(B, T, n, H).bfloat16(),
        ssm_state_indices=slots,
        output=torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="cuda"),
    )


def _copy(inputs: dict) -> dict:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def graph_time_us(inputs: dict, cold: bool, max_copies: int = 256) -> float:
    """Kernel time per public call: one graph holds one call per input copy."""

    tensors = [value for value in inputs.values() if isinstance(value, torch.Tensor)]
    copies = calculate_rotation_count(tensors) if cold else 1
    copies = min(copies, max_copies)
    sets = [inputs] + [_copy(inputs) for _ in range(copies - 1)]
    calls = max(copies, 20)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            cake_gated_delta_product_mtp(**sets[i % copies])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            cake_gated_delta_product_mtp(**sets[i % copies])
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    return statistics.median(times)


def run_case(n: int, T: int, H: int, B: int, cold: bool = True) -> dict:
    inputs = make_inputs(n, T, H, B)
    results = {}
    for name, disabled in (("cake", "1"), ("routed", "0")):
        os.environ[rs.DISABLE_ENV] = disabled
        case = _copy(inputs)
        plan = rs.route(
            **{
                key: case[key]
                for key in (
                    "q",
                    "k",
                    "v",
                    "initial_state",
                    "initial_state_indices",
                    "A_log",
                    "a",
                    "dt_bias",
                    "b",
                    "output",
                    "ssm_state_indices",
                )
            }
        )
        cake_gated_delta_product_mtp(**case)
        torch.cuda.synchronize()
        results[name] = dict(
            plan=plan,
            out=case["output"].clone(),
            pool=case["initial_state"].clone(),
            us=graph_time_us(_copy(inputs), cold),
        )
    os.environ.pop(rs.DISABLE_ENV, None)
    return dict(
        n=n,
        T=T,
        H=H,
        B=B,
        plan=results["routed"]["plan"],
        us_cake=round(results["cake"]["us"], 2),
        us_routed=round(results["routed"]["us"], 2),
        bitwise=bool(
            torch.equal(results["cake"]["out"], results["routed"]["out"])
            and torch.equal(results["cake"]["pool"], results["routed"]["pool"])
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-householder", type=int, nargs="+", default=[3])
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--heads", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64])
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--cold", dest="cold", action="store_true", default=True)
    timing.add_argument("--hot", dest="cold", action="store_false")
    args = parser.parse_args()
    print(
        f"{torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).multi_processor_count} SMs, "
        f"{'cold' if args.cold else 'hot'} L2"
    )
    print(
        f"{'n':>2} {'T':>3} {'H':>4} {'B':>4} {'cake us':>9} {'routed us':>10} {'change':>8}  bitwise  plan"
    )
    with torch.inference_mode():
        for n in args.num_householder:
            for T in args.tokens:
                for H in args.heads:
                    for B in args.batch:
                        r = run_case(n, T, H, B, args.cold)
                        change = 100.0 * (r["us_routed"] / r["us_cake"] - 1.0)
                        print(
                            f"{n:>2} {T:>3} {H:>4} {B:>4} {r['us_cake']:>9.2f} {r['us_routed']:>10.2f} "
                            f"{change:>+7.1f}%  {str(r['bitwise']):>7}  {r['plan']}"
                        )


if __name__ == "__main__":
    main()
