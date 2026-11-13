# SPDX-License-Identifier: Apache-2.0
"""Runnable module example; benchmarks include assignment for prepared plans."""

import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.testing import bench_gpu_time
from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets


def checked_relative_error(output, reference):
    """Return relative L2 error only when both tensors are finite and error < 3%."""
    error = (
        output.float() - reference.float()
    ).norm() / reference.float().norm().clamp_min(1e-8)
    error = error.item()
    if not (
        torch.isfinite(output).all()
        and torch.isfinite(reference).all()
        and error < 0.03
    ):
        raise RuntimeError(
            "Output check requires finite tensors and relative error below 0.03"
        )
    return error


def compare_planners(args):
    """Static metadata, shared operands; host planning stays outside Graphs."""
    from flashinfer.experimental.xqa_decode_buckets.jit import get_module

    records = []
    scenarios = (
        ("balanced_b32_adapted_from_tencent_b64", [4096] * 32),
        ("skew_b8", [65536] + [4096] * 7),
        ("skew_b32", [131072] + [4096] * 31),
        ("short_direct_control", [512]),
    )
    stream = torch.cuda.current_stream()

    def wall_samples(call):
        """Measure completed wall microseconds, synchronizing each warmed call."""
        for _ in range(5):
            call()
            stream.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter_ns()
            call()
            stream.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        return samples

    def graph_samples(call):
        """Measure Graph GPU microseconds with CUPTI and cold-L2 flushing both off."""
        return [
            value * 1000
            for value in bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
                cold_l2_cache=False,
                dry_run_iters=5,
                repeat_iters=20,
            )
        ]

    for label, lengths in scenarios:
        torch.manual_seed(args.seed)
        batch, width = (
            len(lengths),
            (max(lengths) + args.page_size - 1) // args.page_size,
        )
        cpu_lengths = torch.tensor(lengths, dtype=torch.int32)
        lens = cpu_lengths.to(args.device)
        table = torch.zeros((batch, width), dtype=torch.int32, device=args.device)
        pages = 0
        for row, length in enumerate(lengths):
            count = (length + args.page_size - 1) // args.page_size
            table[row, :count] = torch.arange(pages, pages + count, device=args.device)
            pages += count
        plans = {
            planner: create_xqa_decode_bucket_plan(
                lens,
                page_table_width=width,
                page_size=args.page_size,
                num_kv_heads=args.num_kv_heads,
                head_group_ratio=args.head_group_ratio,
                q_len_per_req=args.q_len_per_req,
                query_tile=args.query_tile,
                mode=args.mode,
                planner=planner,
                host_seq_lens=cpu_lengths if planner == "host" else None,
            )
            for planner in ("gpu", "host")
        }
        bank = []
        for _ in range(args.layers):
            q = (
                torch.randn(
                    (batch, args.num_kv_heads * args.head_group_ratio, 256)
                    if args.q_len_per_req == 1
                    else (
                        batch,
                        args.q_len_per_req,
                        args.num_kv_heads * args.head_group_ratio,
                        256,
                    ),
                    dtype=torch.bfloat16,
                    device=args.device,
                )
                * 0.25
            )
            kv = [
                torch.randint(
                    0,
                    256,
                    (pages, args.page_size, args.num_kv_heads, 128),
                    dtype=torch.uint8,
                    device=args.device,
                )
                for _ in range(2)
            ]
            sf = [
                torch.ones(
                    (pages, args.page_size, args.num_kv_heads, 16),
                    dtype=torch.float8_e4m3fn,
                    device=args.device,
                )
                for _ in range(2)
            ]
            outputs = {p: torch.empty_like(q) for p in plans}
            workspaces = {p: plans[p].create_workspace() for p in plans}
            bank.append((q, kv, sf, outputs, workspaces))

        def compute(planner):
            """Consume one planner's descriptors across layers without refreshing them."""
            for q, kv, sf, outputs, workspaces in bank:
                xqa_decode_with_buckets(
                    q,
                    *kv,
                    *sf,
                    table,
                    plans[planner],
                    workspaces[planner],
                    bmm1_scale=0.7 / 16,
                    bmm2_scale=1.3,
                    out=outputs[planner],
                )

        compute_graphs = {}
        for planner in plans:
            compute(planner)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute(planner)
            compute_graphs[planner] = graph
        full_gpu = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_gpu):
            plans["gpu"].prepare()
            compute("gpu")

        def external(planner):
            """Refresh outside capture, then replay the compute-only consumer Graph."""
            plans[planner].prepare()
            compute_graphs[planner].replay()

        samples = {}
        # Bracket the host arm with the GPU arm on the same data and placement.
        for name, call in (
            ("gpu_captured_completed_wall_before_us", full_gpu.replay),
            ("host_prepare_h2d_graph_completed_wall_us", lambda: external("host")),
            ("gpu_captured_completed_wall_after_us", full_gpu.replay),
            ("gpu_external_prepare_graph_completed_wall_us", lambda: external("gpu")),
        ):
            samples[name] = wall_samples(call)
        for planner in compute_graphs:
            samples[planner + "_compute_only_graph_gpu_us"] = graph_samples(
                lambda planner=planner: compute(planner)
            )

        def gpu_window():
            """Include one GPU assignment followed by all layer consumers in timing."""
            plans["gpu"].prepare()
            compute("gpu")

        samples["gpu_assign_compute_graph_gpu_us"] = graph_samples(gpu_window)
        host = plans["host"]
        if not host.direct:
            module, _ = get_module(
                args.mode == "persistent",
                lens.device,
                page_size=args.page_size,
                head_group_ratio=args.head_group_ratio,
                multi_query=args.q_len_per_req > 1,
                query_tile=args.query_tile,
            )
            staging = torch.empty_like(host._host_descriptors, pin_memory=True)
            fill = lambda: module.plan_host(
                cpu_lengths,
                staging,
                host.ctas,
                width,
                args.mode == "aligned",
                args.q_len_per_req,
            )
            for _ in range(5):
                fill()
            fill_samples = []
            for _ in range(20):
                start = time.perf_counter_ns()
                for _ in range(20):
                    fill()
                fill_samples.append((time.perf_counter_ns() - start) / 20000)
            samples["host_cpp_ffi_fill_cpu_us"] = fill_samples
            samples["descriptor_h2d_completed_wall_us"] = wall_samples(
                lambda: host._device_descriptors.copy_(staging, non_blocking=True)
            )
        external("host")
        full_gpu.replay()
        stream.synchronize()
        errors = []
        for _, _, _, outputs, workspaces in bank:
            errors.append(checked_relative_error(outputs["host"], outputs["gpu"]))
            if any(workspace.semaphores.any() for workspace in workspaces.values()):
                raise RuntimeError("Counter reset check failed")
        record = dict(
            scenario=label,
            num_kv_heads=args.num_kv_heads,
            head_group_ratio=args.head_group_ratio,
            q_len_per_req=args.q_len_per_req,
            groups=host.num_groups,
            page_size=args.page_size,
            lengths=lengths,
            layers=args.layers,
            mode=args.mode,
            seed=args.seed,
            direct=host.direct,
            ctas=host.ctas,
            resources=host.resources,
            descriptor_bytes=0 if host.direct else host._host_descriptors.numel() * 4,
            cpu_metadata="caller-provided exact CPU lengths; unchanged during timing",
            length_upload="excluded equally: static metadata already resident for both arms",
            graph_scope="compute-only except gpu_assign_compute graph; host prepare is external",
            samples=samples,
            medians={key: statistics.median(v) for key, v in samples.items()},
            host_vs_gpu_relative_rms=errors,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.output_json:
            target = Path(args.output_json)
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.write_text(
                json.dumps(dict(complete=False, records=records), indent=2)
            )
            temporary.replace(target)
    if args.output_json:
        target = Path(args.output_json)
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(json.dumps(dict(complete=True, records=records), indent=2))
        temporary.replace(target)


def suite_cases(suite="tencent22"):
    """Tencent length distributions, with the documented bounded adaptation."""
    patterns = {
        "uniform512": [512] * 32,
        "uniform4096": [4096] * 32,
        "mixed": [128] * 16 + [4096] * 16,
        "extreme": [64] * 15 + [16384],
        "one64k7": [65536] + [4096] * 7,
        "one64k15": [65536] + [4096] * 15,
        "one64k31": [65536] + [4096] * 31,
        "one128k31": [131072] + [4096] * 31,
        "two32k30": [32768] * 2 + [4096] * 30,
    }
    configs = [
        (32, 1, 6, 1, tuple(patterns)),
        (16, 2, 4, 2, ("uniform512", "uniform4096", "extreme", "one64k7")),
        (32, 4, 6, 3, ("uniform512", "uniform4096", "extreme", "one64k7")),
        (64, 8, 8, 5, ("uniform512", "uniform4096", "extreme", "one64k7")),
        (32, 4, 4, 4, ("one64k7",)),
    ]
    cases = [
        dict(
            case_id=f"p{page}_h{heads}_r{ratio}_t{tokens}_{pattern}",
            pattern=pattern,
            lengths=patterns[pattern],
            page_size=page,
            num_kv_heads=heads,
            head_group_ratio=ratio,
            q_len_per_req=tokens,
        )
        for page, heads, ratio, tokens, names in configs
        for pattern in names
    ]
    if suite in ("query_tiles4", "query_modes4"):
        return [
            case
            for case in cases
            if (case["num_kv_heads"], case["q_len_per_req"]) == (4, 3)
            and case["pattern"] in ("uniform512", "uniform4096", "one64k7")
            or (case["num_kv_heads"], case["q_len_per_req"], case["pattern"])
            == (8, 5, "one64k7")
        ]
    return cases


def atomic_json(target, value):
    """Replace the report after its complete JSON payload reaches a sibling file."""
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def native_launch_geometry(lengths, table_width, page_size, heads, ratio, tokens, sms):
    """Source-derived generic SM120 grid; this is not a profiler observation."""
    splits = min(
        max(1, sms // (len(lengths) * heads)), (table_width * page_size + 255) // 256
    )
    rows = 32 if tokens > 1 else 16
    row_blocks = (tokens * ratio + rows - 1) // rows
    groups = heads * row_blocks
    return dict(
        source="csrc/xqa/mha.cu:launchMHAFlashInfer; csrc/xqa/defines.h",
        interpretation="source-derived generic SM120 launch, not an observed profiler launch",
        split_formula="min(max(1,floor(SMs/(B*KVheads))),ceil(table_capacity/256))",
        sm_count=sms,
        kv_heads=heads,
        query_rows_per_block=rows,
        row_blocks_per_head=row_blocks,
        splits_per_request=splits,
        grid=[splits, groups, len(lengths)],
        grid_ctas=splits * groups * len(lengths),
        nonempty_tile_ctas=sum(min(splits, (n + 255) // 256) for n in lengths) * groups,
        requests_with_inter_cta_merge_per_group=len(lengths) if splits > 1 else 0,
        tile_iteration="strided by splits within each request",
    )


def descriptor_statistics(header, tasks, requests, spans, groups):
    """Summarize actual meaningful descriptors, after an untimed device copy."""
    counts = [end - begin for begin, end in spans]
    partials = [count for _, count in requests]
    assert len(tasks) == sum(counts) == sum(partials) == header[1]
    assert sum(count > 0 for count in counts) == header[2]
    assert max(partials, default=0) == header[3]
    tile_counts = [
        sum(tasks[index][2] - tasks[index][1] for index in range(begin, end))
        for begin, end in spans
    ]
    return dict(
        tile_budget=header[0],
        tasks_per_group=header[1],
        active_owners_per_group=header[2],
        idle_owners_per_group=len(spans) - header[2],
        owners_with_multiple_tasks_per_group=sum(count > 1 for count in counts),
        owner_task_count_histogram=dict(sorted(Counter(counts).items())),
        owner_tile_count_histogram=dict(sorted(Counter(tile_counts).items())),
        request_partial_count_histogram=dict(sorted(Counter(partials).items())),
        requests_with_inter_cta_merge_per_group=sum(count > 1 for count in partials),
        max_request_partials=header[3],
        grid_ctas=len(spans) * groups,
        active_ctas=header[2] * groups,
        groups=groups,
    )


def describe_prepared_plan(plan):
    """Cold diagnostics only; descriptor readback is excluded from all windows."""
    if plan.direct:
        return dict(
            source="direct plan has empty descriptors",
            grid_ctas=plan.ctas * plan.num_groups,
        )
    header = plan.header.cpu().tolist()
    tasks = plan.tasks[: header[1]].cpu().tolist()
    requests, spans = plan.requests.cpu().tolist(), plan.spans.cpu().tolist()
    return dict(
        source="actual prepared device descriptors copied outside timing",
        header=header,
        tasks=tasks,
        requests=requests,
        spans=spans,
        statistics=descriptor_statistics(
            header, tasks, requests, spans, plan.num_groups
        ),
    )


def run_suite(args):
    """Finite single-attention suite; completed matching cases can be resumed."""
    target = Path(args.output_json)
    configuration = dict(
        suite=args.suite,
        layers=1,
        mode="persistent",
        modes=["persistent", "aligned"]
        if args.suite == "query_modes4"
        else ["persistent"],
        planner="gpu",
        device=str(args.device),
        seed=args.seed,
        warmups=5,
        samples=20,
        cuda_graph=True,
        cupti=False,
        cold_l2_cache=False,
        cases=suite_cases(args.suite),
        query_tiles=[32]
        if args.suite == "query_modes4"
        else [16, 32]
        if args.suite == "query_tiles4"
        else [16],
    )
    state = dict(configuration=configuration, complete=False, records=[], failures=[])
    if target.exists():
        if not args.resume:
            raise FileExistsError("Use --resume to retain completed suite cases")
        state = json.loads(target.read_text())
        if state["configuration"] != configuration:
            raise ValueError("Existing suite configuration differs; use a new output")
    completed = {record["case_id"] for record in state["records"]}
    atomic_json(target, state)
    for case in configuration["cases"]:
        if case["case_id"] in completed:
            print(dict(case_id=case["case_id"], state="resumed_complete"), flush=True)
            continue
        print(dict(case_id=case["case_id"], state="starting"), flush=True)
        case_args = copy.copy(args)
        for key in ("page_size", "num_kv_heads", "head_group_ratio", "q_len_per_req"):
            setattr(case_args, key, case[key])
        case_args.compare_query_tiles = args.suite == "query_tiles4"
        case_args.compare_modes = args.suite == "query_modes4"
        if case_args.compare_modes:
            case_args.query_tile = 32
        try:
            result = run_case(
                case_args, case["lengths"], max(case["lengths"]), suite=True
            )
        except Exception as error:
            state["failures"].append(dict(case_id=case["case_id"], error=repr(error)))
            atomic_json(target, state)
            raise
        state["records"].append(dict(case_id=case["case_id"], **result))
        atomic_json(target, state)
        print(dict(case_id=case["case_id"], state="complete"), flush=True)
    state["complete"] = True
    atomic_json(target, state)


def main(argv=None):
    """Validate comparison modes and route CLI requests to finite example windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="131072,4096,4096,4096")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-group-ratio", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument("--q-len-per-req", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--query-tile", type=int, choices=(16, 32), default=16)
    parser.add_argument("--compare-query-tiles", action="store_true")
    parser.add_argument("--compare-modes", action="store_true")
    parser.add_argument("--compare-compile", action="store_true")
    parser.add_argument(
        "--capacity-length", type=int, help="Reserved table capacity in tokens"
    )
    parser.add_argument(
        "--mode", choices=("persistent", "aligned"), default="persistent"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--compare-planners", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--suite", choices=("tencent22", "query_tiles4", "query_modes4")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Time planner and compute windows separately",
    )
    args = parser.parse_args(argv)
    if args.num_kv_heads < 1:
        parser.error("--num-kv-heads must be positive")
    if args.suite:
        if (
            args.layers != 1
            or args.mode != "persistent"
            or not args.output_json
            or args.compare_planners
            or args.diagnose
            or args.capacity_length is not None
            or args.query_tile != 16
            or args.compare_query_tiles
            or args.compare_modes
            or args.compare_compile
        ):
            parser.error("Suite requires --layers 1, persistent mode and --output-json")
        run_suite(args)
        return
    if args.resume:
        parser.error("--resume requires --suite")
    if args.compare_compile and (
        args.compare_query_tiles
        or args.compare_modes
        or args.compare_planners
        or args.diagnose
    ):
        parser.error("--compare-compile uses the existing single-plan layer window")
    if args.compare_modes and (
        args.query_tile != 32
        or args.mode != "persistent"
        or args.compare_query_tiles
        or args.compare_planners
        or args.diagnose
    ):
        parser.error("--compare-modes requires M32/persistent as the first arm")
    if (args.query_tile == 32 or args.compare_query_tiles) and args.q_len_per_req == 1:
        parser.error("M32 requires --q-len-per-req greater than 1")
    if args.compare_query_tiles and (
        args.query_tile != 16 or args.compare_planners or args.diagnose
    ):
        parser.error("--compare-query-tiles uses M16/M32 with the GPU planner")
    if args.compare_planners:
        if args.layers < 1:
            parser.error("Use at least one layer")
        device = torch.device(args.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        args.device = device
        torch.cuda.set_device(device)
        compare_planners(args)
        return
    lengths = [int(n) for n in args.lengths.split(",")]
    if not 1 <= len(lengths) <= 32 or min(lengths) < 1 or args.layers < 1:
        parser.error("Use 1-32 positive lengths and at least one layer")
    if min(lengths) < args.q_len_per_req:
        parser.error("Active KV lengths must include the complete query suffix")
    capacity = max(lengths) if args.capacity_length is None else args.capacity_length
    if capacity < max(lengths):
        parser.error("--capacity-length must cover every live length")
    result = run_case(args, lengths, capacity)
    if args.output_json:
        atomic_json(Path(args.output_json), dict(complete=True, result=result))
    return result


def run_case(args, lengths, capacity, *, suite=False):
    """Compare bucket and native attention on shared operands and fixed capacity.

    Check outputs/counters before and after any requested timing. Optional
    tile, scheduling-mode and compile comparisons retain their own workspaces;
    returned records describe these attention windows rather than model speed.
    """
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    counts = [(n + args.page_size - 1) // args.page_size for n in lengths]
    width = (capacity + args.page_size - 1) // args.page_size
    table = torch.zeros((len(lengths), width), dtype=torch.int32, device=device)
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(offset, offset + count, device=device)
        offset += count
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)

    def make_plan(query_tile, mode=args.mode):
        """Keep lengths and reserved table capacity identical across plan arms."""
        return create_xqa_decode_bucket_plan(
            lens,
            page_table_width=width,
            page_size=args.page_size,
            mode=mode,
            num_kv_heads=args.num_kv_heads,
            head_group_ratio=args.head_group_ratio,
            q_len_per_req=args.q_len_per_req,
            query_tile=query_tile,
        )

    plan = make_plan(args.query_tile)
    primary_key = args.mode if args.compare_modes else args.query_tile
    comparison = args.compare_query_tiles or args.compare_modes
    plans = {primary_key: plan}
    if args.compare_query_tiles:
        plans[32] = make_plan(32)
    elif args.compare_modes:
        # The diagnostic arms share the same cold constant mask tensor.
        plans["aligned"] = replace(
            make_plan(32, "aligned"), causal_mask=plan.causal_mask
        )
    bank = []
    for _ in range(args.layers):
        q = (
            torch.randn(
                (len(lengths), args.num_kv_heads * args.head_group_ratio, 256)
                if args.q_len_per_req == 1
                else (
                    len(lengths),
                    args.q_len_per_req,
                    args.num_kv_heads * args.head_group_ratio,
                    256,
                ),
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.25
        )
        k, v = [
            torch.randint(
                0,
                256,
                (offset, args.page_size, args.num_kv_heads, 128),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(2)
        ]
        ks, vs = [
            torch.ones(
                (offset, args.page_size, args.num_kv_heads, 16),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            for _ in range(2)
        ]
        out, native_out = torch.empty_like(q), torch.empty_like(q)
        workspace = plan.create_workspace()
        native_scratch = torch.zeros(40 * 2**20, dtype=torch.uint8, device=device)
        bank.append((q, k, v, ks, vs, out, native_out, workspace, native_scratch))

    banks = {primary_key: bank}
    for key, selected_plan in plans.items():
        if key != primary_key:
            banks[key] = [
                (
                    q,
                    k,
                    v,
                    ks,
                    vs,
                    torch.empty_like(out),
                    native_out,
                    selected_plan.create_workspace(),
                    scratch,
                )
                for q, k, v, ks, vs, out, native_out, _, scratch in bank
            ]

    def compute(selected_plan=plan, selected_bank=bank):
        """Consume the selected prepared plan using one workspace per layer."""
        for q, k, v, ks, vs, out, _, workspace, _ in selected_bank:
            xqa_decode_with_buckets(
                q,
                k,
                v,
                ks,
                vs,
                table,
                selected_plan,
                workspace,
                bmm1_scale=0.7 / 16,
                bmm2_scale=1.3,
                out=out,
            )

    def candidate(selected_plan=plan, selected_bank=bank):
        """Refresh the shared plan once, then execute every compatible layer."""
        selected_plan.prepare()
        compute(selected_plan, selected_bank)

    def native():
        """Run native XQA with matching operands, scales and causal suffix mask."""
        for q, k, v, ks, vs, _, native_out, _, scratch in bank:
            xqa_batch_decode_with_kv_cache(
                q.view(-1, args.num_kv_heads * args.head_group_ratio, 256),
                (k, v),
                scratch,
                table,
                lens,
                max(lengths),
                bmm1_scale=0.7 / 16,
                bmm2_scale=1.3,
                out=native_out.view(-1, args.num_kv_heads * args.head_group_ratio, 256),
                kv_cache_sf=(ks, vs),
                q_len_per_req=args.q_len_per_req,
                mask=plan.causal_mask if args.q_len_per_req > 1 else None,
            )

    native()
    for key, selected_plan in plans.items():
        candidate(selected_plan, banks[key])
    torch.cuda.synchronize()

    def check_outputs(selected_bank=bank):
        """Require finite agreement with native output and zeroed merge counters."""
        errors = []
        for _, _, _, _, _, out, native_out, workspace, _ in selected_bank:
            errors.append(checked_relative_error(out, native_out))
            if workspace.semaphores.any():
                raise RuntimeError("Module output/counter check failed")
        return errors

    errors_before = check_outputs()
    configuration = dict(
        device=str(device),
        lengths=lengths,
        layers=args.layers,
        mode=args.mode,
        seed=args.seed,
        resources=plan.resources,
        capacity_length=width * args.page_size,
        page_size=args.page_size,
        num_kv_heads=args.num_kv_heads,
        head_group_ratio=args.head_group_ratio,
        q_len_per_req=args.q_len_per_req,
        query_tile=args.query_tile,
        groups=plan.num_groups,
        direct=plan.direct,
        assignment_per_window=0 if plan.direct else 1,
    )
    print(configuration, flush=True)
    result = dict(
        configuration=configuration, arms=[], relative_rms_before=errors_before
    )
    configuration_field = (
        "mode_configurations" if args.compare_modes else "tile_configurations"
    )
    if comparison:
        result[configuration_field] = {
            str(key): dict(
                query_tile=selected_plan.query_tile,
                mode=selected_plan.mode,
                resources=selected_plan.resources,
                groups=selected_plan.num_groups,
                ctas=selected_plan.ctas,
                scratch_bytes=selected_plan.scratch_bytes,
                direct=selected_plan.direct,
                assignment_per_window=0 if selected_plan.direct else 1,
                relative_rms_before=check_outputs(banks[key]),
            )
            for key, selected_plan in plans.items()
        }
    if args.compare_modes:
        result["native_launch_derived"] = native_launch_geometry(
            lengths,
            width,
            args.page_size,
            args.num_kv_heads,
            args.head_group_ratio,
            args.q_len_per_req,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        for key, selected_plan in plans.items():
            result[configuration_field][str(key)]["topology"] = describe_prepared_plan(
                selected_plan
            )
    if args.compare_compile:
        explanation = torch._dynamo.explain(candidate)()
        targets = [
            str(node.target)
            for g in explanation.graphs
            for node in g.graph.nodes
            if node.op == "call_function"
        ]
        counts = dict(
            plan=sum("experimental_xqa_bucket_plan" in name for name in targets),
            compute=sum("experimental_xqa_bucket_compute" in name for name in targets),
        )
        assert explanation.graph_count == 1 and explanation.graph_break_count == 0
        assert counts == dict(plan=int(not plan.direct), compute=args.layers)
        # explain() resets Dynamo caches; warm the real Inductor call afterward.
        compiled_candidate = torch.compile(candidate, fullgraph=True)
        compiled_candidate()
        result.update(
            compiled_backend="default Inductor fullgraph",
            captured_ops=counts,
            compiled_fx_graph=explanation.graphs[0].code,
            assignments_per_window=dict(
                eager_per_layer=0 if plan.direct else args.layers,
                eager_once=0 if plan.direct else 1,
                compiled_once=0 if plan.direct else 1,
            ),
        )
        print(dict(compiled_public_window=counts), flush=True)

        def per_layer_prepare():
            """Control window that repeats the same assignment before each layer."""
            for operands in bank:
                plan.prepare()
                compute(plan, [operands])

    if args.diagnose:
        if plan.direct:
            # Explicitly derived outside timing; not a profiler launch record.
            live_lengths = plan.lengths.cpu().tolist()
            per_request = plan.ctas // len(live_lengths)
            request_bounds = []
            for request, live in enumerate(live_lengths):
                tiles = (live + 255) // 256
                parts = min(tiles, per_request) if len(live_lengths) == 1 else tiles
                base = request * per_request
                request_bounds.append(
                    dict(
                        request=request,
                        live_length=live,
                        partial_begin_group0=base,
                        partial_count=parts,
                        group_stride=parts if len(live_lengths) == 1 else plan.ctas,
                        total_slots=(parts if len(live_lengths) == 1 else plan.ctas)
                        * plan.num_groups,
                        intervals=[
                            [i * tiles // parts, (i + 1) * tiles // parts]
                            for i in range(parts)
                        ],
                    )
                )
            topology = dict(
                source="derived from live device length outside timed region",
                representation="direct_empty_descriptors",
                request_bounds_derived=request_bounds,
                grid=[per_request, plan.num_groups, len(live_lengths)],
                ctas_per_group=plan.ctas,
                grid_ctas=plan.ctas * plan.num_groups,
                nonempty_tile_ctas_derived=sum(
                    r["partial_count"] for r in request_bounds
                )
                * plan.num_groups,
            )
            print(
                dict(
                    arm="assign_only",
                    gpu_launches=0,
                    reason="direct prepare is a no-op",
                ),
                flush=True,
            )
        else:
            header = plan.header.cpu().tolist()
            topology = dict(
                source="prepared device descriptors",
                header=header,
                tasks=plan.tasks[: header[1]].cpu().tolist(),
                requests=plan.requests.cpu().tolist(),
                grid_ctas=plan.ctas * plan.num_groups,
                ctas_per_group=plan.ctas,
                active_ctas=header[2] * plan.num_groups,
            )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        print(
            dict(
                plan_topology=topology,
                native_launch_derived=native_launch_geometry(
                    lengths,
                    width,
                    args.page_size,
                    args.num_kv_heads,
                    args.head_group_ratio,
                    args.q_len_per_req,
                    sms,
                ),
                compute_math="native QK/PV; transposed compute is not implemented",
            ),
            flush=True,
        )
    if args.benchmark or args.diagnose or suite or comparison or args.compare_compile:
        # Warm both and bracket the candidate with native; report absolute samples.
        windows = (
            ("native_before", native),
            ("candidate", candidate),
            ("native_after", native),
        )
        direct_aliases = {}
        if args.compare_compile:
            windows = (
                ("native_before", native),
                ("eager_per_layer_prepare", per_layer_prepare),
                ("eager_once_prepare", candidate),
                ("compiled_once_prepare", compiled_candidate),
                ("native_after", native),
            )
        elif comparison:
            windows = [("native_before", native)]
            for key, selected_plan in plans.items():
                selected_bank = banks[key]
                prefix = str(key) if args.compare_modes else f"m{key}"
                compute_label, full_label = (
                    f"{prefix}_compute_only",
                    f"{prefix}_assign_and_compute",
                )
                windows.append(
                    (
                        compute_label,
                        lambda p=selected_plan, b=selected_bank: compute(p, b),
                    )
                )
                if selected_plan.direct:
                    direct_aliases[compute_label] = full_label
                else:
                    windows.append(
                        (
                            full_label,
                            lambda p=selected_plan, b=selected_bank: candidate(p, b),
                        )
                    )
            windows.append(("native_after", native))
        elif args.diagnose:
            assignment_windows = () if plan.direct else (("assign_only", plan.prepare),)
            windows = (
                ("native_before", native),
                *assignment_windows,
                ("compute_only", compute),
                ("assign_and_compute", candidate),
                ("native_after", native),
            )
        elif suite:
            windows = (
                ("native_before", native),
                ("compute_only", compute),
                *(() if plan.direct else (("assign_and_compute", candidate),)),
                ("native_after", native),
            )
        for label, call in windows:
            times = bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
                cold_l2_cache=False,
                dry_run_iters=5,
                repeat_iters=20,
            )
            record = dict(
                arm=label,
                median_window_us=1000 * statistics.median(times),
                samples_window_us=[1000 * t for t in times],
            )
            result["arms"].append(record)
            print(record, flush=True)
            alias = direct_aliases.get(label)
            if alias or suite and plan.direct and label == "compute_only":
                identical = dict(
                    record, arm=alias or "assign_and_compute", same_window_as=label
                )
                result["arms"].append(identical)
                print(identical, flush=True)
        if args.diagnose:
            print(
                dict(diagnostic_relative_rms=check_outputs(), counters_reset=True),
                flush=True,
            )
    result.update(relative_rms_after=check_outputs(), counters_reset=True)
    if comparison:
        for key in plans:
            result[configuration_field][str(key)].update(
                relative_rms_after=check_outputs(banks[key]), counters_reset=True
            )
    if suite:
        print(
            dict(relative_rms_after=result["relative_rms_after"], counters_reset=True),
            flush=True,
        )
    return result


if __name__ == "__main__":
    main()
