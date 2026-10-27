#!/usr/bin/env python3
"""CUDA graph replay of SM120 Mamba SSD against the Triton combined scan.

Both implementations read the same fixed initial state and write separate final
states. Measurements include time-step processing, chunk-state propagation, and
all token outputs. Projections, causal convolution, and normalization are excluded.
"""

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "mamba"))

import torch
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import mamba_ssd
from flashinfer.testing import bench_gpu_time
from triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[129, 512, 4096])
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument(
        "--autotune", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        parser.error("requires an assigned SM120 or SM121 device")
    print(torch.cuda.get_device_name(), flush=True)
    print(
        f"heads={args.heads} groups={args.groups} head_dim={args.head_dim} dstate={args.dstate} state=BF16",
        flush=True,
    )
    for tokens in args.tokens:
        torch.manual_seed(13)
        h, g, d, n = args.heads, args.groups, args.head_dim, args.dstate

        def rand(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, device="cuda", dtype=dtype)

        x, B, C = rand(tokens, h, d), rand(tokens, g, n), rand(tokens, g, n)
        dt, A = rand(tokens, h, dtype=torch.float32), -torch.rand(h, device="cuda") - 1
        D, bias = rand(h, dtype=torch.float32), rand(h, dtype=torch.float32) - 4
        state = rand(3, h, d, n)

        def integer(values, dtype=torch.int32):
            return torch.tensor(values, device="cuda", dtype=dtype)

        cu, src, dst = (
            integer([0, tokens]),
            integer([1], torch.int64),
            integer([2], torch.int64),
        )
        fresh, ns, nt = integer([False], torch.bool), integer([1]), integer([tokens])
        operands = x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt
        out, reference = torch.empty_like(x), torch.empty_like(x)
        kwargs = dict(
            x=x[None],
            dt=dt[None],
            A=A,
            B=B[None],
            C=C[None],
            D=D,
            dt_bias=bias,
            initial_states=state[1:2],
            chunk_size=128,
            dt_softplus=True,
            out=reference[None],
            cu_seqlens=cu,
            seq_idx=torch.zeros((1, tokens), device="cuda", dtype=torch.int32),
            chunk_indices=torch.arange(
                (tokens + 127) // 128, device="cuda", dtype=torch.int32
            ),
            chunk_offsets=torch.zeros(
                (tokens + 127) // 128, device="cuda", dtype=torch.int32
            ),
        )
        expected_state = _mamba_chunk_scan_combined_fwd(**kwargs)[-1]
        plan = mamba_ssd.plan(
            mamba_ssd.query_from_call(*operands, out=out), device=out.device
        )
        workspace = torch.empty(
            plan.memory_requirements().scratch_nbytes, device="cuda", dtype=torch.uint8
        )
        seed = x.clone()
        with PreparationSession(
            device=out.device, autotune=args.autotune, compile_workers=1
        ) as session:

            def call(prepared):
                return PreparedCall(
                    run=lambda: prepared.run(*operands, out=out, workspace=workspace),
                    output=out,
                    produce=lambda: x.copy_(seed),
                    owners=(seed,),
                )

            session.prepare(
                (plan.request(name="ssd", prepare_call=call, benchmark_call=call),)
            )
            session.freeze()

            def run():
                return mamba_ssd.run(*operands, out=out, plan=plan, workspace=workspace)

            run()
            error = (
                (out.float() - reference.float()).norm() / reference.float().norm()
            ).item()
            state_error = (
                (state[2:3].float() - expected_state.float()).norm()
                / expected_state.float().norm()
            ).item()
            if max(error, state_error) >= 0.008:
                raise AssertionError(
                    f"SSD mismatch: output={error}, state={state_error}"
                )
            print(
                f"tokens={tokens} config={plan.selection.config} output_rel_l2={error:.6g} state_rel_l2={state_error:.6g}",
                flush=True,
            )
            for name, fn in (
                ("triton", lambda: _mamba_chunk_scan_combined_fwd(**kwargs)),
                ("b12x", run),
            ):
                times = bench_gpu_time(
                    fn,
                    enable_cupti=False,
                    use_cuda_graph=True,
                    cold_l2_cache=False,
                    dry_run_time_ms=50,
                    repeat_time_ms=200,
                )
                print(
                    f"tokens={tokens} implementation={name} median_us={statistics.median(times) * 1000:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
