# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Eight-B300 developer benchmark for the public GLM-5 TP8 operators.

Run from a source checkout on a node with eight B300 GPUs::

    torchrun --standalone --nproc-per-node=8 \
        benchmarks/experimental/bench_glm5_tp8_decode.py --output results.json

    torchrun --standalone --nproc-per-node=8 \
        benchmarks/experimental/bench_glm5_tp8_decode.py \
        --past-lens 4096 32768 65536 --capacity 65544 --output long_context.json

    torchrun --standalone --nproc-per-node=8 \
        benchmarks/experimental/bench_glm5_tp8_decode.py --operators attention \
        --past-lens 524288 1048576 --capacity 1052672 --output million_context.json

This developer script imports deterministic input generators from
``tests/experimental/glm5_tp8_decode/data.py`` in the same checkout. It constructs
weights, workspaces and plans through the public API; it does not import the test
references or run correctness checks. Run the operator tests separately before
interpreting these measurements. Synthetic inputs do not establish model
throughput, and the script does not compare against other implementations.

Each timed invocation replays a graph containing one complete collective call.
Compilation, packing, preparation, input copies and graph capture are excluded.
All ranks use identical fixed iteration counts. CUPTI is requested explicitly;
the library's CUDA-event fallback measures the same single-call graph. Default
cache condition is repeated use of the same buffers without an L2 flush; use
``--cold-l2`` to flush L2 before each measured replay.

The completion-boundary statistic is the median of the maximum rank duration
for each corresponding iteration, not the maximum of rank medians. It includes
kernel-internal peer waits, but does not subtract timestamps from different GPU
clocks or include Python/Gloo barrier time. The JSON output keeps every per-rank
sample. The script does not lock or change GPU clocks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import faulthandler
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import traceback


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
import torch.distributed as dist

from flashinfer import glm5_tp8_decode as api
from flashinfer.experimental.glm5_tp8_decode.geometry import MAX_SEQ_LEN
from flashinfer.testing import bench_gpu_time
from tests.experimental.glm5_tp8_decode import data


class Collective:
    def __init__(self, rank, device):
        self.rank, self.device = rank, device

    def gather(self, value):
        values = [None] * 8
        dist.all_gather_object(values, value)
        return values

    def checked(self, function):
        error, result = None, None
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
        errors = self.gather(error)
        if any(errors):
            raise RuntimeError(
                "\n".join(
                    f"rank {rank}:\n{error}"
                    for rank, error in enumerate(errors)
                    if error
                )
            )
        return result

    def broadcast(self, value, src=0):
        values = [value]
        dist.broadcast_object_list(values, src=src)
        return values[0]

    def sync(self):
        torch.cuda.synchronize(self.device)
        dist.barrier()


def _cupti_status():
    try:
        from cupti import cupti  # noqa: F401

        installed = version("cupti-python")
        if int(installed.split(".")[0]) < 13:
            return {
                "timer": "cuda_event_fallback",
                "cupti_version": installed,
                "reason": "cupti-python >= 13 is required",
            }
        return {"timer": "cupti", "cupti_version": installed, "reason": None}
    except Exception as exc:
        return {
            "timer": "cuda_event_fallback",
            "cupti_version": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def make_moe_layer(control, seed):
    """Synthetic MoE checkpoint layer: rank-local experts, replicated router/norm."""
    layer = control.checked(
        lambda: data.make_moe_layer(control.rank, seed, control.device)
    )
    replicated = control.checked(
        lambda: {name: layer[name].cpu() for name in data.MOE_REPLICATED}
        if control.rank == 0
        else None
    )
    replicated = control.broadcast(replicated)
    layer.update({name: value.to(control.device) for name, value in replicated.items()})
    return layer


def _moe_weights(control, seed):
    layer = make_moe_layer(control, seed)
    return control.checked(
        lambda: api.pack_glm5_tp8_moe_weights(layer, rank=control.rank)
    )


def _attention_weights(control, seed):
    # Every rank builds the same CPU checkpoint layer, as in the tests; packing
    # copies only the rank's slices to its GPU.
    layer = data.make_attention_layer(seed)
    return control.checked(
        lambda: api.pack_glm5_tp8_attention_weights(
            layer, rank=control.rank, device=control.device
        )
    )


@dataclass
class PreparedCase:
    """A prepared public plan with its bound input and output tensors."""

    x: torch.Tensor
    out: torch.Tensor
    plan: object
    selection: torch.Tensor | None = None


def prepare_moe_case(control, workspace, packed, *, seed, q_len):
    x = control.checked(lambda: data.make_input(q_len, seed, control.device))
    plan = api.prepare_glm5_tp8_moe(x, packed, workspace=workspace)
    return PreparedCase(x=x, out=plan.out, plan=plan)


def prepare_attention_case(
    control, workspace, packed, caches, rope, position, *, seed, q_len, mode, past_len
):
    x = control.checked(lambda: data.make_attention_input(q_len, seed, control.device))
    selection = control.checked(
        lambda: data.make_selection(q_len, past_len, seed + 2).to(control.device)
        if control.rank == 0 and mode == "shared"
        else None
    )
    control.checked(lambda: position.fill_(past_len))
    plan = api.prepare_glm5_tp8_sparse_attention(
        x,
        packed,
        caches=caches,
        rope=rope,
        cur_pos=position,
        workspace=workspace,
        indexer=mode,
        selection=selection,
    )
    return PreparedCase(x=x, out=plan.out, plan=plan, selection=selection)


def _memory(workspace, device):
    usage = workspace.memory_usage()
    return {
        **usage,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def _measure(control, workspace, plan, args, *, family, q_len, mode, past_len):
    faulthandler.dump_traceback_later(args.timeout_seconds, repeat=False, exit=False)
    control.sync()
    # Warm up all ranks with the same count before capture. Workspace creation
    # already compiled/loaded the modules and initialized state before the first
    # exchange.
    stream = torch.cuda.Stream(device=control.device)
    stream.wait_stream(torch.cuda.current_stream(control.device))
    with torch.cuda.stream(stream):
        for _ in range(args.warmup_iters):
            plan()
    control.sync()
    generation = int(plan._diagnostics["generation"]())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan()
    control.sync()
    control.checked(lambda: _check_capture_generation(plan, generation))
    torch.cuda.reset_peak_memory_stats(control.device)
    # The timing helper performs all-gathers internally. Select this rank from
    # each gathered sample so its final default max-reduction does not erase the
    # per-rank series. Fixed counts make the estimate reduction immaterial.
    times = bench_gpu_time(
        graph.replay,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        cold_l2_cache=args.cold_l2,
        sleep_after_run=False,
        aggregate_op=lambda rank_values: rank_values[control.rank],
    )
    control.sync()
    local = {
        "rank": control.rank,
        "cuda_device": control.device.index,
        "samples_us": [float(value) * 1000 for value in times],
        "memory": _memory(workspace, control.device),
        "generation": int(plan._diagnostics["generation"]()),
    }
    ranks = control.gather(local)
    if any(len(rank["samples_us"]) != args.repeat_iters for rank in ranks):
        raise RuntimeError("timer returned an unexpected number of samples")
    if any(
        not math.isfinite(value) or value <= 0
        for rank in ranks
        for value in rank["samples_us"]
    ):
        raise RuntimeError("timer returned a nonpositive or nonfinite duration")
    if len({rank["generation"] for rank in ranks}) != 1:
        raise RuntimeError("collective launch generations differ between ranks")
    boundary = [
        max(sample)
        for sample in zip(*(rank["samples_us"] for rank in ranks), strict=True)
    ]
    for rank in ranks:
        rank["median_us"] = statistics.median(rank["samples_us"])
    result = {
        "family": family,
        "q_len": q_len,
        "indexer": mode,
        "past_len": past_len,
        "capacity": args.capacity,
        "cache": "cold_l2" if args.cold_l2 else "reused_buffers",
        "rank_results": ranks,
        "max_rank_completion_median_us": statistics.median(boundary),
        "max_rank_completion_samples_us": boundary,
    }
    if control.rank == 0:
        medians = ",".join(f"{rank['median_us']:.3f}" for rank in ranks)
        workspace_bytes = [
            rank["memory"]["symmetric_bytes"] + rank["memory"]["local_tensor_bytes"]
            for rank in ranks
        ]
        print(
            f"{family} q={q_len} mode={mode or '-'} past={past_len} "
            f"rank_median_us=[{medians}] "
            f"max_rank_completion_median_us={result['max_rank_completion_median_us']:.3f} "
            f"workspace_bytes={workspace_bytes}",
            flush=True,
        )
    # Drop the executable graph after its last replay completes, before the
    # caller releases plans or destroys the symmetric allocation.
    graph.reset()
    del graph
    control.sync()
    return result


def _check_capture_generation(plan, generation):
    if int(plan._diagnostics["generation"]()) != generation:
        raise RuntimeError("graph capture unexpectedly executed the operator")


def _benchmark_moe(control, args):
    packed = _moe_weights(control, args.seed)
    workspace = api.create_glm5_tp8_decode_workspace(
        rank=control.rank,
        device=control.device,
        group=dist.group.WORLD,
        operators=("moe",),
        max_seq_len=args.capacity,
    )
    results = []
    try:
        for q_len in (1, 4):
            case = prepare_moe_case(
                control, workspace, packed, seed=args.seed + 17, q_len=q_len
            )
            results.append(
                _measure(
                    control,
                    workspace,
                    case.plan,
                    args,
                    family="moe",
                    q_len=q_len,
                    mode=None,
                    past_len=None,
                )
            )
            del case
    finally:
        workspace.destroy()
    return results


def _benchmark_attention(control, args):
    packed = _attention_weights(control, args.seed)
    caches = control.checked(
        lambda: data.make_state(
            control.rank, args.capacity + 8, args.seed, control.device
        )
    )
    rope = control.checked(lambda: data.make_rope_table(args.capacity, control.device))
    position = control.checked(
        lambda: torch.zeros(1, dtype=torch.int32, device=control.device)
    )
    workspace = api.create_glm5_tp8_decode_workspace(
        rank=control.rank,
        device=control.device,
        group=dist.group.WORLD,
        operators=("attention",),
        max_seq_len=args.capacity,
    )
    results = []
    try:
        for past_len in args.past_lens:
            for q_len, mode in ((4, "full"), (4, "shared"), (1, "shared")):
                case = prepare_attention_case(
                    control,
                    workspace,
                    packed,
                    caches,
                    rope,
                    position,
                    seed=args.seed + 17,
                    q_len=q_len,
                    mode=mode,
                    past_len=past_len,
                )
                results.append(
                    _measure(
                        control,
                        workspace,
                        case.plan,
                        args,
                        family="attention",
                        q_len=q_len,
                        mode=mode,
                        past_len=past_len,
                    )
                )
                del case
    finally:
        workspace.destroy()
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=("moe", "attention"),
        default=["moe", "attention"],
    )
    parser.add_argument("--past-lens", nargs="+", type=int, default=[4096])
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=531017)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument(
        "--identity-devices",
        action="store_true",
        help="use CUDA ordinal = logical rank (default: (rank + 3) %% 8)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.capacity = (
        max(4104, *(past + 8 for past in args.past_lens))
        if args.capacity is None
        else args.capacity
    )
    if not 1 <= args.capacity <= MAX_SEQ_LEN or any(
        past < 0 or past + 4 > args.capacity for past in args.past_lens
    ):
        parser.error(
            f"capacity must be in [1,{MAX_SEQ_LEN}] and each past length in [0,capacity-4]"
        )
    if (
        min(
            args.warmup_iters,
            args.dry_run_iters,
            args.repeat_iters,
            args.timeout_seconds,
        )
        < 1
    ):
        parser.error("iteration counts and timeout must be positive")
    return args


def main():
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=8")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout_seconds))
    rank = dist.get_rank()
    device = torch.device("cuda", rank if args.identity_devices else (rank + 3) % 8)
    control = Collective(rank, device)
    faulthandler.dump_traceback_later(args.timeout_seconds, repeat=False, exit=False)
    try:
        configurations = control.gather(vars(args))
        if any(config != configurations[0] for config in configurations):
            raise RuntimeError(
                "all eight ranks must receive identical benchmark arguments"
            )

        def admit():
            if torch.cuda.device_count() != 8:
                raise RuntimeError(
                    "benchmark requires eight visible B300 GPUs in each process"
                )
            torch.cuda.set_device(device)
            prop = torch.cuda.get_device_properties(device)
            if (prop.major, prop.minor, prop.multi_processor_count) != (10, 3, 148):
                raise RuntimeError("benchmark requires B300 / SM103 with 148 SMs")
            return {
                "rank": rank,
                "cuda_device": device.index,
                "name": prop.name,
                "uuid": str(prop.uuid),
                "host": socket.gethostname(),
            }

        devices = control.gather(control.checked(admit))
        if (
            len({item["uuid"] for item in devices}) != 8
            or len({item["host"] for item in devices}) != 1
        ):
            raise RuntimeError("benchmark requires eight distinct GPUs on one node")
        timers = control.gather(_cupti_status())
        if len({item["timer"] for item in timers}) != 1:
            raise RuntimeError(
                "CUPTI availability differs across ranks; fix the environment before timing"
            )
        if rank == 0:
            mapping = (
                "logical_rank" if args.identity_devices else "logical_rank+3(mod8)"
            )
            print(
                f"timer={timers[0]['timer']} cupti_version={timers[0]['cupti_version']} "
                f"fallback_reason={timers[0]['reason']} mapping={mapping}",
                flush=True,
            )
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "arguments": {
                **vars(args),
                "output": str(args.output) if args.output else None,
            },
            "devices": devices,
            "timer_status_by_rank": timers,
            "results": [],
        }
        for family in dict.fromkeys(args.operators):
            faulthandler.dump_traceback_later(
                args.timeout_seconds, repeat=False, exit=False
            )
            benchmark = _benchmark_moe if family == "moe" else _benchmark_attention
            report["results"].extend(benchmark(control, args))
            control.sync()
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0 and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"results_json={args.output.resolve()}", flush=True)
        control.sync()
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
