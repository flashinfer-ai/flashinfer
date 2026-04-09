#!/usr/bin/env python3
"""
Benchmark: state dump every step vs. dump every Nth step + replay.

Simulates a full generation loop of output_seq_len tokens with MTP speculation.
Compares the cost of dumping state at every MTP step (baseline) against dumping
every state_dump_steps and replaying after acceptance.

See .plans/state_replate_study.md for design details.

Usage:
    docker exec -w /home/scratch.ishovkun_gpu/code/flashinfer-dev flashinfer-cu130-dev-ishovkun \
        python benchmarks/bench_state_replay.py

    # Custom params:
    docker exec -w /home/scratch.ishovkun_gpu/code/flashinfer-dev flashinfer-cu130-dev-ishovkun \
        python benchmarks/bench_state_replay.py \
            --mtp 8 --state-dump-steps 1 2 4 8 -a 2 \
            --batch 32 --osl 16384

    # Explicit dump schedule (1-indexed MTP steps to dump):
    docker exec -w /home/scratch.ishovkun_gpu/code/flashinfer-dev flashinfer-cu130-dev-ishovkun \
        python benchmarks/bench_state_replay.py \
            --dump-schedule 1 3 5 8
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "mamba"))
from utils import create_test_inputs

from flashinfer.mamba import selective_state_update


def make_dump_schedule(mtp, state_dump_steps):
    """Return list of MTP steps (1-indexed) that get dumped.

    Step 1 is always dumped (non-speculative token).
    After that, dump every state_dump_steps steps.

    Example: mtp=8, state_dump_steps=4 → [1, 4, 8]
    Example: mtp=8, state_dump_steps=1 → [1, 2, 3, 4, 5, 6, 7, 8]
    """
    if state_dump_steps == 1:
        return list(range(1, mtp + 1))
    dumps = [1]
    step = state_dump_steps
    while step <= mtp:
        dumps.append(step)
        step += state_dump_steps
    return dumps


def compute_replay(n_accepted, dump_schedule, step_to_slot):
    """Given n_accepted, find last dumped state and compute replay steps.

    Returns (source_slot, replay_steps).
    """
    last_dump_step = None
    for step in reversed(dump_schedule):
        if step <= n_accepted:
            last_dump_step = step
            break
    assert last_dump_step is not None, f"No dump at or before n_accepted={n_accepted}"
    source_slot = step_to_slot[last_dump_step]
    replay_steps = n_accepted - last_dump_step
    return source_slot, replay_steps


def run_scenario(
    mtp,
    state_dump_steps,
    n_accepted,
    output_seq_len,
    batch_size,
    nheads,
    dim,
    dstate,
    ngroups,
    state_dtype,
    num_warmup=2,
    num_repeats=3,
    dump_schedule=None,
):
    """Run a full generation loop and return median elapsed time in ms."""

    if dump_schedule is None:
        dump_schedule = make_dump_schedule(mtp, state_dump_steps)
    num_dumps_per_iter = len(dump_schedule)

    num_iterations = (output_seq_len + n_accepted - 1) // n_accepted
    cache_holds_iterations = 10
    num_dump_slots = num_dumps_per_iter * cache_holds_iterations
    total_slots = batch_size + num_dump_slots

    # Create inputs using the test utility
    inputs = create_test_inputs(
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dstate,
        ngroups=ngroups,
        input_dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        state_dtype=state_dtype,
        matrixA_dtype=torch.float32,
        generate_z=False,
        cache_steps=mtp,
        device="cuda",
        seed=0,
    )

    # Allocate single big state tensor
    state = torch.randn(
        total_slots, nheads, dim, dstate, dtype=state_dtype, device="cuda"
    )

    # Shared tensors (same data every iteration — we care about timing, not values)
    x = inputs["x"]  # (batch, mtp, nheads, dim)
    dt = inputs["dt"]  # (batch, mtp, nheads, dim)
    B = inputs["B"]  # (batch, mtp, ngroups, dstate)
    C = inputs["C"]  # (batch, mtp, ngroups, dstate)
    A = inputs["A"]  # (nheads, dim, dstate)
    D = inputs["D"]  # (nheads, dim)
    dt_bias = inputs["dt_bias"]  # (nheads, dim)
    out = torch.empty_like(x)

    # For replay: pre-allocate tensors outside the timed loop.

    # state_batch_indices: which slot to read initial state from
    state_batch_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    # For replay: read from the dump slot, write back to main slot
    main_slots = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    pad_slot_id = -1

    # Pre-allocate dst_state_batch_indices (rewritten each iteration)
    dst_indices = torch.full(
        (batch_size, mtp), pad_slot_id, dtype=torch.int32, device="cuda"
    )

    # The replay_steps is the same every iteration (n_accepted and dump_schedule are fixed)
    _, fixed_replay_steps = compute_replay(
        n_accepted,
        dump_schedule,
        {step: 999 for step in dump_schedule},  # dummy slots, we only need replay_steps
    )

    # Pre-allocate replay tensors (if replay is needed)
    # Use MTP path (cache_steps=fixed_replay_steps) for replay so dst_state_batch_indices works.
    # For replay_steps=1, the kernel with ntokens_mtp=1 has different stride expectations,
    # so we use cache_steps=2 and pad with an extra dummy step that writes to pad_slot_id.
    if fixed_replay_steps > 0:
        replay_cache_steps = max(fixed_replay_steps, 2)
        replay_read_indices = torch.empty(batch_size, dtype=torch.int32, device="cuda")
        replay_dst = torch.full(
            (batch_size, replay_cache_steps),
            pad_slot_id,
            dtype=torch.int32,
            device="cuda",
        )
        # Write main slot at the last real replay step (0-indexed = fixed_replay_steps - 1)
        replay_dst[:, fixed_replay_steps - 1] = main_slots
        # Don't call .contiguous() — dt has special strided layout (stride(2)=1, stride(3)=0)
        # that the kernel requires. Slicing preserves strides.
        replay_x = x[:, :replay_cache_steps]
        replay_dt = dt[:, :replay_cache_steps]
        replay_B = B[:, :replay_cache_steps]
        replay_C = C[:, :replay_cache_steps]
        replay_out = torch.empty_like(replay_x)

    def run_loop():
        next_slot = batch_size  # start allocating dump slots after main slots
        tokens_generated = 0

        for _iteration in range(num_iterations):
            if tokens_generated >= output_seq_len:
                break

            # 1. Build dst_state_batch_indices for this iteration (update in-place)
            dst_indices.fill_(pad_slot_id)
            step_to_slot = {}
            for step in dump_schedule:
                slot = next_slot
                step_to_slot[step] = slot
                next_slot = batch_size + (next_slot - batch_size + 1) % (
                    total_slots - batch_size
                )
                dst_indices[:, step - 1] = slot

            # 2. Launch MTP kernel
            selective_state_update(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_batch_indices,
                dst_state_batch_indices=dst_indices,
                pad_slot_id=pad_slot_id,
                out=out,
                cache_steps=mtp,
                algorithm="simple",
            )

            # 3. Simulate acceptance: find source slot for the last dump ≤ n_accepted
            source_slot, _ = compute_replay(n_accepted, dump_schedule, step_to_slot)

            if fixed_replay_steps > 0:
                # Replay: read from dump slot, write final state to main slot
                replay_read_indices.fill_(source_slot)
                selective_state_update(
                    state=state,
                    x=replay_x,
                    dt=replay_dt,
                    A=A,
                    B=replay_B,
                    C=replay_C,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=replay_read_indices,
                    dst_state_batch_indices=replay_dst,
                    pad_slot_id=pad_slot_id,
                    out=replay_out,
                    cache_steps=replay_cache_steps,
                )
            else:
                # No replay needed — copy dump slot → main slot
                state[main_slots] = state[source_slot].clone()

            tokens_generated += n_accepted

    # Warmup: JIT compile + GPU warmup
    print(f"    Warming up (dump_schedule={dump_schedule})...")
    for _ in range(num_warmup):
        run_loop()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for r in range(num_repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        run_loop()
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed)
        print(f"    Run {r + 1}/{num_repeats}: {elapsed:.2f} ms")

    times.sort()
    median_time = times[len(times) // 2]
    return median_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark state dump vs. dump+replay strategies"
    )
    parser.add_argument("--mtp", type=int, default=8, help="MTP steps (default: 8)")
    parser.add_argument(
        "--state-dump-steps",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="State dump intervals to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--dump-schedule",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dump schedule (1-indexed steps), overrides --state-dump-steps. E.g. --dump-schedule 1 3 5 8",
    )
    parser.add_argument(
        "-a",
        "--accepted",
        type=int,
        default=2,
        help="Tokens accepted per iteration (default: 2)",
    )
    parser.add_argument(
        "--osl", type=int, default=16384, help="Output sequence length (default: 16384)"
    )
    parser.add_argument(
        "-b-", "--batch", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--dstate", type=int, default=128, help="State dimension (default: 128)"
    )
    parser.add_argument(
        "--nheads", type=int, default=64, help="Number of heads (default: 64)"
    )
    parser.add_argument(
        "--dim", type=int, default=64, help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--ngroups", type=int, default=8, help="Number of groups (default: 8)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "f32"],
        help="State dtype (default: bf16)",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=3,
        help="Number of timed repeats (default: 3)",
    )
    args = parser.parse_args()

    state_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    num_iterations = (args.osl + args.accepted - 1) // args.accepted

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("STATE DUMP vs REPLAY BENCHMARK")
    print("=" * 70)
    print(f"GPU: {gpu_name}")
    print(f"mtp={args.mtp}, n_accepted={args.accepted}, output_seq_len={args.osl}")
    print(
        f"batch_size={args.batch}, nheads={args.nheads}, dim={args.dim}, dstate={args.dstate}"
    )
    print(f"state_dtype={args.dtype}, ngroups={args.ngroups}")
    print(f"num_iterations={num_iterations}")
    if args.dump_schedule:
        print(f"explicit dump_schedule: {args.dump_schedule}")
    else:
        print(f"state_dump_steps to test: {args.state_dump_steps}")
    print("=" * 70)

    # Build list of (label, dump_schedule) pairs to benchmark
    scenarios = []
    if args.dump_schedule:
        # Always include baseline (dump every step) for comparison
        baseline_schedule = make_dump_schedule(args.mtp, 1)
        scenarios.append(("baseline (sds=1)", baseline_schedule))
        print(
            f"\n  baseline: dump_schedule={baseline_schedule}, n_accepted={args.accepted}, replay_steps=0"
        )

        schedule = sorted(args.dump_schedule)
        _, replay_steps = compute_replay(
            args.accepted,
            schedule,
            {step: i for i, step in enumerate(schedule)},
        )
        label = f"custom {schedule}"
        scenarios.append((label, schedule))
        print(f"\n  {label}: n_accepted={args.accepted}, replay_steps={replay_steps}")
    else:
        for sds in args.state_dump_steps:
            schedule = make_dump_schedule(args.mtp, sds)
            _, replay_steps = compute_replay(
                args.accepted,
                schedule,
                {step: i for i, step in enumerate(schedule)},
            )
            label = f"sds={sds}"
            scenarios.append((label, schedule))
            print(
                f"\n  state_dump_steps={sds}: dump_schedule={schedule}, n_accepted={args.accepted}, replay_steps={replay_steps}"
            )

    results = []
    for label, schedule in scenarios:
        median_ms = run_scenario(
            mtp=args.mtp,
            state_dump_steps=None,
            n_accepted=args.accepted,
            output_seq_len=args.osl,
            batch_size=args.batch,
            nheads=args.nheads,
            dim=args.dim,
            dstate=args.dstate,
            ngroups=args.ngroups,
            state_dtype=state_dtype,
            num_repeats=args.num_repeats,
            dump_schedule=schedule,
        )
        results.append((label, schedule, median_ms))

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"RESULTS (median time) — mtp={args.mtp}, n_accepted={args.accepted}")
    print("=" * 70)
    print(
        f"{'scenario':>20} {'dump_schedule':>25} {'time (ms)':>12} {'ms/token':>12} {'speedup':>10}"
    )
    print("-" * 82)

    first_ms = results[0][2]
    for label, schedule, ms in results:
        ms_per_token = ms / args.osl
        speedup = first_ms / ms if first_ms and ms > 0 else float("nan")
        print(
            f"{label:>20} {str(schedule):>25} {ms:>12.2f} {ms_per_token:>12.4f} {speedup:>10.3f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
