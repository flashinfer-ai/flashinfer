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

"""PCIe IPC all-reduce latency, against NCCL.

Intended for intra-node PCIe machines without NVLink, which is what the kernel
is tuned for. NCCL is the baseline because it is what the caller falls back to
when a shape is not in the tuning table.

One run covers ONE world size. The world size comes from the launcher and the
collective uses ``dist.group.WORLD`` throughout -- there is no subgroup logic --
so a 2/4/8-rank comparison needs three launches:

    sudo nvidia-smi -lgc 2520,2520   # nothing below pins clocks; boost drift
                                     # swamps the 2-3% differences being measured

    for n in 2 4 8; do
        torchrun --standalone --nproc_per_node=$n \\
            benchmarks/comm/bench_pcie_ipc_all_reduce.py --json bench_tp$n.json
    done

Hidden size follows the world size unless overridden, so no other flag is needed.

To measure what a protocol fix costs, rather than how it compares to NCCL:

    for n in 2 4 8; do
        timeout 3000 torchrun --standalone --nproc_per_node=$n \\
            benchmarks/comm/bench_pcie_ipc_all_reduce.py \\
            --protocol-ab auto --json ab_tp$n.json
    done

The external timeout is not boilerplate. At 8 ranks the historically faithful
baseline for the staged kernels IS the cross-island protocol that was removed
for being broken, so rank skew can leave it spinning with nothing to time it
out from inside.

'auto' compares each shape against the protocol it actually used to run, which
at 8 ranks differs between the pack and staged kernels. Naming a switch
explicitly is still allowed, and rows it does not historically fit are marked
SYNTHETIC.

To measure the launch configuration rather than read it from the table:

    torchrun --standalone --nproc_per_node=8 \\
        benchmarks/comm/bench_pcie_ipc_all_reduce.py --tune --json bench_tuned.json

Rows whose configuration tuning changed are annotated with what the table would
have chosen. The result is persisted, so a later run without --tune reuses it.

Options:
    --hidden N        Hidden size (default: 6144 at 8 ranks, 4096 at 4, 2048 at 2)
    --batches a,b,c   Batch sizes to sweep
    --dtype           bfloat16 (default) or float16
    --json FILE       Write results to JSON
    --tune            Measure the launch configuration instead of reading it
    --tune-cache FILE Where --tune persists its result
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm import pcie_ipc_ar
from flashinfer.comm.pcie_ipc_policy import IpcVariant
from flashinfer.jit.comm import gen_pcie_ipc_comm_debug_module
from flashinfer.testing.utils import bench_gpu_time

_DEFAULT_HIDDEN = {2: 2048, 4: 4096, 8: 6144}
_DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

# Pin the iteration counts. Left to auto-tune, each rank derives its own count
# from its own timings, and a collective where the ranks disagree on how many
# times to call it deadlocks.
_BENCH_KWARGS = dict(
    use_cuda_graph=True,
    # Each replay carries a fixed cost -- the launch plus the event pair timing
    # it -- that the reported number amortises over the iterations in the graph,
    # so anything compared against these numbers must use the same count.
    num_iters_within_graph=20,
    dry_run_iters=5,
    repeat_iters=20,
    # The timed callables close over their tensors rather than taking them as
    # arguments, which is all the cold-L2 helper inspects.
    cold_l2_cache=False,
)


def _median_us(samples) -> float:
    # bench_gpu_time reports milliseconds.
    return float(torch.tensor(samples).median()) * 1e3


def _group_median_us(samples, device, group) -> float:
    """Median over iterations of the group maximum at each iteration.

    Order matters here. Taking each rank's median and then the max across ranks
    computes ``max_rank median_iter``, which is not what a collective costs: if
    rank 0 is slow on odd iterations and rank 4 on even ones, every rank's own
    median is low while every actual iteration had a straggler. The tail is
    ``median_iter max_rank``, so the max has to be taken per sample, before the
    median.

    Requires every rank to have the same samples in the same order, which is why
    the iteration counts are pinned rather than auto-tuned; asserted below
    because silently reducing across mismatched vectors would produce a number
    that looks fine.
    """
    local = list(samples)
    # One collective for both bounds: max of n, and max of -n (i.e. -min).
    bounds = torch.tensor([len(local), -len(local)], dtype=torch.int64, device=device)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=group)
    if int(bounds[0].item()) != -int(bounds[1].item()):
        raise RuntimeError(
            "ranks produced different sample counts "
            f"({-int(bounds[1].item())}..{int(bounds[0].item())}); "
            "per-iteration aggregation needs them aligned"
        )
    t = torch.as_tensor(local, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    # bench_gpu_time reports milliseconds.
    return float(t.median()) * 1e3


def _group_all(flag: bool, device, group) -> bool:
    """True only when every rank says True.

    Correctness cannot be judged locally here. The cross-island race this
    harness can rebuild produced errors confined to *one island*: rank 0 came
    out clean while ranks 4-7 had millions of wrong elements. A rank-local check
    would have published a cost for a baseline that was wrong.
    """
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--batches", type=str, default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--json", type=str, default=None)
    p.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Measure the launch configuration instead of reading it from the "
            "table, then report against NCCL. Collective and slow (a few "
            "seconds per batch bucket); pin clocks first."
        ),
    )
    p.add_argument(
        "--tune-cache",
        type=str,
        default=None,
        help="Where --tune persists its result. Defaults to the workspace dir.",
    )
    p.add_argument(
        "--protocol-ab",
        choices=[
            "auto",
            "per-block-epoch",
            "no-block-epoch",
            "no-barrier-entry-sync",
        ],
        default=None,
        help=(
            "Instead of comparing to NCCL, measure what one protocol mechanism "
            "costs: rebuild the kernels with it disabled and time both in the "
            "same process, A-B-A per shape, reporting the A/A spread as a noise "
            "floor. 'auto' compares each shape against the mechanism it "
            "actually used to run, which at 8 ranks differs between the pack "
            "and staged kernels. Every switch produces an incorrect build -- run "
            "under an external timeout, since one of them can spin. See the "
            "module docstring."
        ),
    )
    return p.parse_args()


def _historical_switch(world_size, config):
    """Which A/B switch actually rebuilds what this shape used to run.

    The kernels do not share one history. TP2 and TP4 always double-buffered by
    per-block parity, so ``per-block-epoch`` restores their past. The two
    topology-staged TP8 kernels never had an epoch at all -- their scratch was
    pinned to half 0 and reused across calls, which is what ``no-block-epoch``
    rebuilds. The TP8 pack kernel did have per-block parity, like TP2/TP4, and
    so does ``FLAT_STAGED``: it is the same generic template TP4 runs.

    Running the other switch is not wrong, it just measures a protocol that
    never shipped, so rows are labelled rather than refused.
    """
    tp8_topo_staged = world_size == 8 and config.variant in (
        IpcVariant.STAGED,
        IpcVariant.STAGED_RING,
    )
    return "no-block-epoch" if tp8_topo_staged else "per-block-epoch"


# The two switches that rebuild an *epoch* protocol. Which of them is faithful
# depends on the kernel; every other switch rebuilds a state that no kernel ever
# had a different version of, so it is historical everywhere.
_EPOCH_SWITCHES = ("per-block-epoch", "no-block-epoch")


def _is_historical(world_size, config, switch):
    if switch not in _EPOCH_SWITCHES:
        return True
    return switch == _historical_switch(world_size, config)


def _protocol_ab_plan(world_size, shape_config, mode):
    """Which batches each A/B leg should actually run.

    Returns ``{switch: [batch, ...]}``. In ``auto`` this partitions the shapes by
    the protocol they used to run, and that partition is the executable plan, not
    just a labelling rule: a leg that runs a shape it does not own executes a
    protocol that shape never had, and one of those -- the fixed-half build on a
    kernel that always double-buffered -- wedges its sentinel loop outright. A
    row discarded at reporting time has already run by then.

    With an explicit switch the caller has asked for exactly that comparison, so
    every tuned shape runs and the mismatched ones are reported as SYNTHETIC.
    """
    if mode != "auto":
        return {mode: sorted(shape_config)} if shape_config else {}
    plan = {}
    for batch, config in shape_config.items():
        plan.setdefault(_historical_switch(world_size, config), []).append(batch)
    # Unsafe legs last. 'no-block-epoch' rebuilds a protocol that can spin
    # forever, and a hang there would otherwise take down the safe leg's results
    # with it -- results that were already complete and cost nothing to keep.
    order = sorted(plan, key=lambda sw: (sw == "no-block-epoch", sw))
    return {switch: sorted(plan[switch]) for switch in order}


def _run_ab_legs(plan, run_sweep, on_switch_done=None):
    """Drive A-B-A for every switch in `plan`, over that switch's shapes only.

    Split out from the timing so it can be tested without a GPU. The property
    that matters is not which switch a shape is *labelled* with but which leg
    actually executes it: a leg running a shape it does not own runs a protocol
    that shape never had, and one of those wedges outright. That mistake was
    made once already, with a correct labelling function sitting right next to
    it, so the wiring is now a seam a test can hold.

    ``run_sweep(broken_switch, batches) -> {batch: result}``; ``broken_switch``
    is None for the shipping build.
    """
    legs = {}
    for switch, leg_batches in plan.items():
        legs[switch] = (
            run_sweep(switch, leg_batches),
            run_sweep(None, leg_batches),
            run_sweep(switch, leg_batches),
        )
        if on_switch_done is not None:
            on_switch_done(switch, legs[switch])
    return legs


def _time_one_batch(workspace, batch, hidden, dtype, device, group):
    """Time one shape, after proving this build still computes it correctly.

    The correctness check is not decoration. Two of the three A/B baselines are
    protocols known to be broken, and a broken protocol that has landed on the
    wrong half but not yet wedged its spin loop still produces a perfectly
    timeable kernel. Without this, such a run reports a number that means
    nothing.
    """
    ref_src = torch.randint(1, 16, (batch, hidden), dtype=torch.int32, device=device)
    inp = ref_src.to(dtype)
    ref = inp.clone()
    dist.all_reduce(ref, group=group)

    config = workspace.launch_config(inp)
    if config is None:
        return None
    dst = torch.empty_like(inp)
    torch.cuda.synchronize()
    workspace.rebind_stream()

    workspace.all_reduce(inp, out=dst, config=config)
    torch.cuda.synchronize()
    correct = _group_all(bool(torch.equal(dst, ref)), device, group)
    if not correct:
        # Nothing to learn from timing a build that is already wrong, and this
        # is exactly where the unsafe baselines fail -- running the graph
        # hundreds more times mostly buys a chance to spin forever instead of
        # reporting. Safe to branch: `correct` is group-wide, so every rank
        # leaves here together.
        return None, None, config, False, False

    # The check above bound the workspace to the current stream, and
    # bench_gpu_time warms up on a side stream before capturing. Sequential, not
    # concurrent -- the synchronize above is the ordering the escape hatch asks
    # for -- so release the binding again.
    workspace.rebind_stream()
    samples = bench_gpu_time(
        lambda: workspace.all_reduce(inp, out=dst, config=config), **_BENCH_KWARGS
    )
    us = _group_median_us(samples, device, group)
    rank_us = _median_us(samples)

    # And once more afterwards. The pre-check only says this build was right
    # once; the timed run then replays the same call hundreds of times inside a
    # graph, which is the repetition a broken protocol needs to show itself. It
    # still cannot falsify a race -- the payload is constant, so an overwrite may
    # store a bit-identical value -- but it catches anything that has actually
    # gone wrong by the end, and that costs one comparison.
    torch.cuda.synchronize()
    workspace.rebind_stream()
    workspace.all_reduce(inp, out=dst, config=config)
    torch.cuda.synchronize()
    post_correct = _group_all(bool(torch.equal(dst, ref)), device, group)
    return us, rank_us, config, correct, post_correct


def _run_protocol_ab(args, group, world_size, hidden, batches, dtype, device, rank):
    """Time the shipping protocol against one with a fix compiled out.

    The workspace resolves its module through a module-global, so swapping that
    is enough to put an instrumented build under the same wrapper the shipping
    path uses. A-B-A: the two A runs bracket B, and their spread is the noise
    floor any claimed delta has to clear.

    **One workspace per batch, per leg.** A shared workspace would let one shape
    contaminate the next, and specifically so on the side that matters: the
    per-block baseline *is* the protocol whose parity desynchronises for good
    once the grid changes, so every batch after the first grid change would be
    timed against state left by its predecessors rather than against a clean
    single-shape run. It usually does not hang -- the default batch list happens
    to avoid the shapes that wedge immediately -- and "did not hang" is not the
    same as "measured the thing we named".

    ``auto`` runs one sweep per switch and keeps, for each shape, only the sweep
    whose switch matches that shape's actual history. At 8 ranks that is the
    only way to get a complete historically faithful table, because the pack and
    staged kernels came from different protocols.
    """
    # (no_block_epoch, per_block_epoch, no_barrier_entry_sync)
    _SWITCH = {
        "no-block-epoch": (1, 0, 0),
        "per-block-epoch": (0, 1, 0),
        "no-barrier-entry-sync": (0, 0, 1),
    }

    # Work out which shape maps to which kernel before building anything, so
    # auto mode can run only the switches some shape actually needs. Running a
    # switch nothing needs is not merely wasted time: 'no-block-epoch' at 2 or 4
    # ranks removes the double buffer from a kernel that has always had one, and
    # that wedges the sentinel loop outright.
    probe = comm.PcieIpcAllReduceWorkspace(
        group=group, max_numel=hidden * max(batches), dtype=dtype
    )
    try:
        shape_config = {}
        for batch in batches:
            cfg = probe.launch_config(
                torch.empty(batch, hidden, dtype=dtype, device=device)
            )
            if cfg is not None:
                shape_config[batch] = cfg
    finally:
        probe.destroy()

    plan = _protocol_ab_plan(world_size, shape_config, args.protocol_ab)
    if not plan:
        if rank == 0:
            print("no tuned shape in this batch list; nothing to compare")
        return

    def measure(broken_switch, leg_batches):
        """One sweep over `leg_batches`, with `broken_switch` compiled out or not."""
        pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()
        original = pcie_ipc_ar.gen_pcie_ipc_comm_module
        flags = _SWITCH[broken_switch] if broken_switch else (0, 0, 0)

        def _gen():
            return gen_pcie_ipc_comm_debug_module(0, 0, *flags)

        pcie_ipc_ar.gen_pcie_ipc_comm_module = _gen
        try:
            out = {}
            for batch in leg_batches:
                ws = comm.PcieIpcAllReduceWorkspace(
                    group=group, max_numel=hidden * batch, dtype=dtype
                )
                try:
                    got = _time_one_batch(ws, batch, hidden, dtype, device, group)
                finally:
                    ws.destroy()
                if got is not None:
                    out[batch] = got
            return out
        finally:
            pcie_ipc_ar.gen_pcie_ipc_comm_module = original
            pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()

    # A-B-A per switch, over that switch's own shapes only. B runs the same
    # subset, so all three legs of a comparison execute an identical workload.
    rows: List[Dict[str, object]] = []

    def _flush():
        """Write what is finished so far, atomically.

        Granularity is one complete A-B-A group, not one leg: without both A
        legs there is no A/A spread, so a half-finished switch has nothing worth
        keeping. What this does buy is that a switch which spins cannot take an
        earlier switch's finished rows down with it -- the unsafe one is ordered
        last for the same reason. rename() makes a reader see either the old
        file or a complete new one.
        """
        if rank != 0 or not args.json:
            return
        tmp = args.json + ".partial"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, args.json)

    def _emit(switch, leg):
        """Report one finished leg, then persist."""
        for batch in plan[switch]:
            if any(batch not in part for part in leg):
                continue
            a1, b, a2 = (part[batch] for part in leg)
            with_fix, with_fix_rank0, config, fixed_ok, fixed_post = b
            baseline_ok = a1[3] and a2[3] and a1[4] and a2[4]
            # A leg that failed its pre-check returns no timing at all.
            timed = baseline_ok and fixed_ok and fixed_post
            without = (a1[0] + a2[0]) / 2 if timed else float("nan")
            drift = abs(a1[0] - a2[0]) / without * 100 if timed else float("nan")
            historical = _historical_switch(world_size, config)
            is_historical = _is_historical(world_size, config, switch)
            ok = timed
            row = {
                "batch": batch,
                "hidden": hidden,
                "world_size": world_size,
                "switch": switch,
                "unsafe_baseline": switch
                in ("no-block-epoch", "no-barrier-entry-sync"),
                "historical_switch": historical
                if switch in _EPOCH_SWITCHES
                else switch,
                "synthetic_baseline": not is_historical,
                "blocks": config.blocks,
                "threads": config.threads,
                "variant": config.variant.name,
                "with_fix_us": with_fix,
                "with_fix_rank0_us": with_fix_rank0,
                "timed": timed,
                "latency_agg": "median_of_per_iteration_group_max",
                "with_fix_correct": fixed_ok and fixed_post,
                "baseline_correct": baseline_ok,
                "correctness_is_group_wide": True,
                "aa_drift_pct": drift,
            }
            # A cost against a baseline that computed the wrong answer is not a
            # cost, so do not publish one.
            if ok:
                row["without_fix_us"] = without
                row["cost_pct"] = (with_fix - without) / without * 100
            rows.append(row)
            if rank == 0:
                if not (fixed_ok and fixed_post):
                    note = "WRONG RESULT"
                elif not baseline_ok:
                    note = "baseline wrong"
                else:
                    note = "historical" if is_historical else "SYNTHETIC"
                    if switch in ("no-block-epoch", "no-barrier-entry-sync"):
                        note += " UNSAFE"
                    if not is_historical:
                        note += f" (history: {historical})"
                if ok:
                    cost = f"{(with_fix - without) / without * 100:>+7.1f}%"
                    print(
                        f"{batch:>7} {config.blocks:>7} {without:>12.2f} "
                        f"{with_fix:>10.2f} {cost} {drift:>6.1f}%  {note}"
                    )
                else:
                    print(
                        f"{batch:>7} {config.blocks:>7} {'--':>12} {'--':>10} "
                        f"{'--':>8} {'--':>6}  {note} (not timed)"
                    )
        _flush()

    if rank == 0:
        label = (
            "auto (per-shape history: "
            + ", ".join(f"{sw} x{len(b)}" for sw, b in plan.items())
            + ")"
            if args.protocol_ab == "auto"
            else args.protocol_ab
        )
        print(f"world_size={world_size} hidden={hidden} A/B on {label}")
        print(
            f"{'batch':>7} {'blocks':>7} {'without(us)':>12} {'with(us)':>10} "
            f"{'cost':>8} {'A/A':>7}  baseline"
        )
        print(
            "  (latency: median of the per-iteration group max; "
            "correctness: group-wide)"
        )

    _run_ab_legs(plan, measure, on_switch_done=_emit)

    if rank == 0:
        print(
            "  historical = the baseline is what this shape actually used to run; "
            "SYNTHETIC = a protocol it never ran; UNSAFE = the baseline itself is "
            "known-broken, so its timings describe this workload only."
        )
        if args.json:
            print(f"wrote {args.json} ({len(rows)} rows, flushed after each leg)")


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    hidden = args.hidden if args.hidden is not None else _DEFAULT_HIDDEN.get(world_size)
    if hidden is None:
        raise ValueError(f"--hidden is required at {world_size} ranks")
    batches = (
        [int(b) for b in args.batches.split(",")]
        if args.batches
        else list(_DEFAULT_BATCHES)
    )
    dtype = getattr(torch, args.dtype)

    if args.protocol_ab is not None:
        _run_protocol_ab(args, group, world_size, hidden, batches, dtype, device, rank)
        dist.destroy_process_group()
        return

    workspace = comm.PcieIpcAllReduceWorkspace(
        group=group, max_numel=hidden * max(batches), dtype=dtype
    )
    table_configs = {}
    if args.tune:
        # Record what the table would have chosen before overwriting it, so the
        # report shows what tuning actually changed rather than just the result.
        for batch in batches:
            probe = torch.empty(batch, hidden, dtype=dtype, device=device)
            table_configs[batch] = workspace.launch_config(probe)
        if rank == 0:
            print(f"tuning {len(batches)} batch buckets at hidden {hidden} ...")
        torch.cuda.synchronize()
        workspace.rebind_stream()
        workspace.tune([hidden], dtype=dtype, cache=args.tune_cache)
    if rank == 0:
        print(f"world_size={world_size} hidden={hidden} dtype={args.dtype}")
        print(f"profile={workspace.profile} ({workspace.profile_reason})")
        print(f"{'batch':>7} {'ours(us)':>10} {'nccl(us)':>10} {'speedup':>9}  config")

    rows: List[Dict[str, object]] = []
    for batch in batches:
        inp = torch.randn(batch, hidden, dtype=dtype, device=device)
        # The tuned answer, so the reported configuration is the one the timed
        # call below actually runs.
        config = workspace.tuned_launch_config(inp)
        if config is None:
            if rank == 0:
                print(
                    f"{batch:>7} {'-':>10} {'-':>10} {'-':>9}  untuned, would fall back"
                )
            continue
        out = torch.empty_like(inp)

        # bench_gpu_time warms up on a side stream before capturing a graph.
        # That is sequential, not concurrent, so tell the workspace it may move
        # its binding -- after making sure the previous stream is drained.
        torch.cuda.synchronize()
        workspace.rebind_stream()
        # A collective finishes when its slowest rank does, and which rank that
        # is can change from iteration to iteration -- so the max is taken per
        # sample and the median after, not the other way round. The local
        # medians stay in the JSON so an older table can still be lined up
        # against a newer one.
        ours_samples = bench_gpu_time(
            lambda: workspace.all_reduce(inp, out=out), **_BENCH_KWARGS
        )
        nccl_inp = inp.clone()
        nccl_samples = bench_gpu_time(
            lambda: dist.all_reduce(nccl_inp, group=group), **_BENCH_KWARGS
        )
        ours_us = _group_median_us(ours_samples, device, group)
        nccl_us = _group_median_us(nccl_samples, device, group)
        ours_rank_us = _median_us(ours_samples)
        nccl_rank_us = _median_us(nccl_samples)
        rows.append(
            {
                "batch": batch,
                "hidden": hidden,
                "world_size": world_size,
                "dtype": args.dtype,
                "ours_us": ours_us,
                "nccl_us": nccl_us,
                "ours_rank0_us": ours_rank_us,
                "nccl_rank0_us": nccl_rank_us,
                "latency_agg": "median_of_per_iteration_group_max",
                "speedup": nccl_us / ours_us,
                "blocks": config.blocks,
                "threads": config.threads,
                "variant": config.variant.name,
            }
        )
        if rank == 0:
            print(
                f"{batch:>7} {ours_us:>10.2f} {nccl_us:>10.2f} "
                f"{nccl_us / ours_us:>8.2f}x  blocks={config.blocks} "
                f"threads={config.threads} variant={config.variant.name}"
                + (
                    ""
                    if not args.tune or table_configs.get(batch) == config
                    else f"  (table: {table_configs[batch]})"
                )
            )

    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")

    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
