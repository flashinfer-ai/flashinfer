# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare public KDA/GDN prefill calls, including their host work.

Examples (run on an otherwise idle allocated GPU):
  python benchmarks/bench_cudnn_linear_attention.py --family kda --lengths 8192 --heads 16 --output kda.json
  python benchmarks/bench_cudnn_linear_attention.py --family kda --lengths 8192 --heads 16 --backends native-auto,auto,cudnn --output kda-auto.json
  python benchmarks/bench_cudnn_linear_attention.py --family gdn --lengths 1024,3072 --heads 4 --value-heads 8 --state-dtype fp32 --backends auto,cake_gdn,cake_gdn_cp,cudnn --output gdn.json
  python benchmarks/bench_cudnn_linear_attention.py --family gdn --gdn-raw-gates --lengths 8192 --heads 4 --value-heads 8 --state-dtype fp32 --backends native-auto,auto,cudnn --output gdn-raw.json

CPU active, wall enqueue, synchronized completion and CUDA-event graph replay
are separate experiments, not additive components. Host clocks exclude state
reset/synchronization. Each captured invocation includes the same state reset;
the replay span includes reset, copies, kernels and gaps, not just kernel time.
Caller-owned outputs/state and retained capture workspaces are used throughout.

KDA uses additive 1e-6 normalization on every backend. Zero/tiny rows check
this against an independent FP32 serial reference. GDN uses pre-normalized
Q/K by default; --gdn-normalize measures additive normalization, including
any provider-owned conversion.
These are adapter measurements, not serving TTFT or full-model accuracy.
The native-auto arm disables only the cuDNN preference outside each
timed block, keeping the public callable and original native selection intact.
--gdn-raw-gates passes raw BF16 gates and FP32 A_log/bias to every provider.
Its native baseline includes the public API's Torch gate materialization; it
does not reproduce a caller using a separate fused gate producer such as vLLM.
--seq-order passes the B1 identity scheduling hint to auto/native arms, as
upstream Kimi does. Explicit cuDNN currently rejects that optional hint, so its
arm omits it; the per-arm result records whether it was supplied.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("kda", "gdn"), default="kda")
    parser.add_argument(
        "--lengths", default="8192", help="Comma-separated packed lengths"
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int)
    parser.add_argument("--state-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--layout", choices=("dense", "glm-qk"), default="dense")
    parser.add_argument("--gdn-normalize", action="store_true")
    parser.add_argument("--gdn-raw-gates", action="store_true")
    parser.add_argument(
        "--seq-order",
        action="store_true",
        help="Supply the B1 identity scheduling hint to KDA auto/native arms",
    )
    parser.add_argument("--backends", default="auto,cudnn")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--cpu", type=int, help="Optional CPU affinity; no implicit first-core pin"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(",")]
    args.backends = args.backends.split(",")
    args.value_heads = args.value_heads or args.heads
    if (
        min(
            *args.lengths,
            args.heads,
            args.value_heads,
            args.iters,
            args.repeats,
            args.unroll,
        )
        <= 0
    ):
        parser.error("shapes and iteration counts must be positive")
    if args.backends[0] not in ("auto", "native-auto"):
        parser.error("the first backend must be auto or native-auto")
    if args.gdn_raw_gates and args.family != "gdn":
        parser.error("--gdn-raw-gates applies only to GDN")
    if args.seq_order and (args.family != "kda" or len(args.lengths) != 1):
        parser.error("--seq-order currently exercises B1 KDA only")
    if args.family == "kda" and args.value_heads != args.heads:
        parser.error("this KDA comparison currently uses equal heads")
    if args.family == "kda" and args.unroll != 1:
        parser.error("KDA's explicit workspace is captured once; use --unroll 1")
    if args.family == "kda" and (
        sum(args.lengths) <= len(args.lengths) or sum(args.lengths) < 3
    ):
        parser.error(
            "KDA requires an ordinary multi-token prefill with at least three tokens"
        )
    if args.family == "gdn" and args.layout != "dense":
        parser.error("glm-qk is a KDA layout")
    if args.value_heads % args.heads or args.value_heads < args.heads:
        parser.error("value heads must be a multiple of query heads")
    return args


def make_data(args, seed):
    torch.manual_seed(seed)
    total, hq, hv, dim = sum(args.lengths), args.heads, args.value_heads, 128
    dtype = torch.bfloat16
    q = torch.randn(total, hq, dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(total, hv, dim, device="cuda", dtype=dtype)
    if args.family == "kda" or args.gdn_normalize:
        for tensor in (q, k):
            tensor[0::127] = 0
            tensor[1::127] *= 1e-5
            tensor[2::127] *= 1e-4
    if args.family == "kda":
        if args.layout == "glm-qk":
            carrier = torch.empty(total, 3 * hq * dim, device="cuda", dtype=dtype)
            for source, view in zip(
                (q, k), carrier.split(hq * dim, -1)[:2], strict=True
            ):
                view.copy_(source.flatten(1))
            q, k = (
                view.view(total, hq, dim) for view in carrier.split(hq * dim, -1)[:2]
            )
        q, k, v = (tensor.unsqueeze(0) for tensor in (q, k, v))
        gate = torch.randn(1, total, hv, dim, device="cuda", dtype=dtype) * 0.1
        beta = torch.randn(1, total, hv, device="cuda", dtype=dtype)
    else:
        if not args.gdn_normalize:
            q, k = (
                torch.nn.functional.normalize(tensor.float(), dim=-1).to(dtype)
                for tensor in (q, k)
            )
        gate = 0.8 + 0.15 * torch.rand(total, hv, device="cuda")
        beta = 0.2 + 0.6 * torch.rand(total, hv, device="cuda")
    offsets = [0]
    for length in args.lengths:
        offsets.append(offsets[-1] + length)
    state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    result = dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        cu=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        seed=0.1
        * torch.randn(
            len(args.lengths), hv, dim, dim, device="cuda", dtype=state_dtype
        ),
        A_log=torch.zeros(hv, device="cuda"),
        dt_bias=torch.zeros(hv, dim, device="cuda"),
    )
    if args.family == "gdn" and args.gdn_raw_gates:
        result.update(
            g=-1 + 0.5 * torch.randn(total, hv, device="cuda", dtype=dtype),
            beta=torch.randn(total, hv, device="cuda", dtype=dtype),
            A_log=0.3 * torch.randn(hv, device="cuda"),
            dt_bias=0.2 * torch.randn(hv, device="cuda"),
        )
    if args.seq_order:
        result["seq_order"] = torch.zeros(1, dtype=torch.int32, device="cuda")
    return result


def serial_reference(args, data, norm):
    q, k, v = (
        data[name][0] if args.family == "kda" else data[name]
        for name in ("q", "k", "v")
    )
    q, k, v = q.float(), k.float(), v.float()
    if args.family == "kda" or args.gdn_normalize:

        def normalize(tensor):
            square = tensor.square().sum(-1, keepdim=True)
            return tensor * (square + 1e-6).rsqrt()

        q, k = normalize(q), normalize(k)
    if args.family == "kda":
        alpha = (
            -5.0
            * torch.sigmoid(
                data["A_log"].exp()[None, :, None]
                * (data["g"][0].float() + data["dt_bias"])
            )
        ).exp()
        beta = data["beta"][0].float().sigmoid()
    elif args.gdn_raw_gates:
        alpha = torch.exp(
            -data["A_log"].float().exp()
            * torch.nn.functional.softplus(data["g"].float() + data["dt_bias"].float())
        )
        beta = data["beta"].float().sigmoid().to(data["beta"].dtype).float()
    else:
        alpha, beta = data["g"], data["beta"]
    state = data["seed"].float().clone()
    output = torch.empty(
        sum(args.lengths), args.value_heads, 128, device=q.device, dtype=torch.float32
    )
    q_heads = torch.arange(args.value_heads, device=q.device) // (
        args.value_heads // args.heads
    )
    start = 0
    for sequence, length in enumerate(args.lengths):
        current = state[sequence]
        for token in range(start, start + length):
            key = k[token, q_heads]
            current = current * alpha[token].reshape(args.value_heads, 1, -1)
            residual = v[token] - torch.einsum("hvk,hk->hv", current, key)
            current = (
                current
                + (beta[token, :, None] * residual)[:, :, None] * key[:, None, :]
            )
            output[token] = (
                torch.einsum("hvk,hk->hv", current, q[token, q_heads]) / 128**0.5
            )
        state[sequence] = current
        start += length
    return (output.unsqueeze(0) if args.family == "kda" else output), state


def errors(args, actual, expected):
    def relative(got, want):
        if not torch.isfinite(got).all() or not torch.isfinite(want).all():
            return float("inf")
        return ((got.float() - want).norm() / want.norm().clamp_min(1e-12)).item()

    result = {
        name: relative(got, want)
        for name, got, want in zip(("output", "state"), actual, expected, strict=True)
    }
    if args.family == "kda" or args.gdn_normalize:
        got = actual[0][0] if args.family == "kda" else actual[0]
        want = expected[0][0] if args.family == "kda" else expected[0]
        for offset in (1, 2):
            result[f"tiny_rows_{offset}"] = relative(
                got[offset::127], want[offset::127]
            )
        result["zero_rows"] = float(torch.count_nonzero(got[0::127]).item())
    return result


def assert_result(args, actual, expected):
    measured = errors(args, actual, expected)
    if max(measured.values()) > args.tolerance:
        raise AssertionError(f"independent-reference mismatch: {measured}")
    return measured


@contextmanager
def dispatch_policy(args, backend):
    """Choose the native baseline outside measured calls, without unwrapping APIs."""
    if backend != "native-auto":
        yield
        return
    module = importlib.import_module(
        "flashinfer.kda" if args.family == "kda" else "flashinfer.gdn_prefill"
    )
    name = f"_prefer_cudnn_{args.family}_prefill"
    previous = getattr(module, name)
    setattr(module, name, lambda *args: False)
    try:
        yield
    finally:
        setattr(module, name, previous)


def make_runner(
    args, data, backend, norm=None, *, capture=False, capture_uses_cudnn=False
):
    state, output = (
        torch.empty_like(data["seed"]),
        torch.empty(data["v"].shape, device="cuda", dtype=data["v"].dtype),
    )
    final = state if args.family == "kda" else torch.empty_like(state)
    if args.family == "kda":
        from flashinfer.kda import recurrent_kda
        from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace

        workspace = (
            RecurrentKDAPrefillWorkspace("cuda")
            if capture and backend != "cudnn" and not capture_uses_cudnn
            else None
        )
        kwargs = dict(
            A_log=data["A_log"],
            dt_bias=data["dt_bias"],
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            lower_bound=-5.0,
            beta_is_logit=True,
            initial_state=state,
            output=output,
            output_final_state=True,
            cu_seqlens=data["cu"],
            backend="auto" if backend == "native-auto" else backend,
            prefill_workspace=workspace,
        )
        if "seq_order" in data and backend != "cudnn":
            kwargs["seq_order"] = data["seq_order"]

        def run():
            return recurrent_kda(
                *(data[name] for name in ("q", "k", "v", "g", "beta")), **kwargs
            )
    else:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        kwargs = dict(
            initial_state=state,
            output=output,
            output_state=final,
            output_final_state=True,
            cu_seqlens=data["cu"],
            use_qk_l2norm_in_kernel=args.gdn_normalize,
            max_seqlen=max(args.lengths),
            backend="cake_gdn" if backend == "cake_gdn_cp" else backend,
            use_cp=True if backend == "cake_gdn_cp" else "auto",
        )
        if backend == "native-auto":
            kwargs["backend"] = "auto"
        if args.gdn_raw_gates:
            kwargs.update(
                use_gate_in_kernel=True,
                A_log=data["A_log"],
                dt_bias=data["dt_bias"],
                beta_is_logit=True,
            )

        def run():
            return chunk_gated_delta_rule(
                *(data[name] for name in ("q", "k", "v", "g", "beta")), **kwargs
            )

    return dict(run=run, state=state, output=output, final=final, data=data)


def observe_route(run):
    """Collect real Python launch/build paths on an untimed call only."""
    routes = set()
    plans = []
    adapter = importlib.import_module("flashinfer.cudnn.linear_attention")
    original_run = adapter._run_la_graph
    cudnn_executions = 0

    def record_execution(*args, **kwargs):
        nonlocal cudnn_executions
        result = original_run(*args, **kwargs)
        # A build decline may enter the adapter before auto falls back. Only
        # a successful execution proves cuDNN ran, independent of FE internals.
        cudnn_executions += 1
        return result

    def trace(frame, event, arg):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        if module.startswith(
            ("flashinfer.kda", "flashinfer.gdn", "flashinfer.cudnn")
        ) and (
            name.startswith(("_run", "run_", "chunk_", "cudnn_")) or "launch" in name
        ):
            routes.add(f"{module}.{name}")
        if module == "cudnn.linear_attention.frost.engine" and name == "execute":
            compiled = frame.f_locals["self"].compiled
            plans.append(
                {
                    key: getattr(compiled, key, None)
                    for key in ("plan_name", "chain", "dv_split", "prep")
                }
            )

    previous = sys.getprofile()
    adapter._run_la_graph = record_execution
    try:
        sys.setprofile(trace)
        result = run()
    finally:
        sys.setprofile(previous)
        adapter._run_la_graph = original_run
    return result, dict(
        python_calls=sorted(routes),
        frontend_plans=plans,
        cudnn_executions=cudnn_executions,
    )


def check_route(backend, route, eager_route=None):
    """Reject missing route proof and eager/capture provider changes."""
    uses_cudnn = bool(route["cudnn_executions"])
    if route["frontend_plans"] and not uses_cudnn:
        raise AssertionError("frontend execution bypassed the cuDNN route observer")
    if backend == "cudnn" and not uses_cudnn:
        raise AssertionError("cuDNN route detection observed no successful execution")
    if backend not in ("auto", "cudnn") and uses_cudnn:
        raise AssertionError(f"{backend} unexpectedly executed cuDNN")
    if eager_route is not None and uses_cudnn != bool(eager_route["cudnn_executions"]):
        raise AssertionError("eager and captured calls selected different providers")


def check_ownership(args, arm):
    if args.family == "gdn":
        torch.testing.assert_close(arm["state"], arm["data"]["seed"], rtol=0, atol=0)


def prepare_arm(args, data, backend, references, norm):
    arm = make_runner(args, data, backend, norm)
    arm["state"].copy_(data["seed"])
    actual, route = observe_route(arm["run"])
    check_route(backend, route)
    torch.cuda.synchronize()
    choices = {key: errors(args, actual, value) for key, value in references.items()}
    matching = [
        key for key, value in choices.items() if max(value.values()) <= args.tolerance
    ]
    if len(matching) != 1:
        raise AssertionError(
            f"normalization must match exactly one reference: {choices}"
        )
    actual_norm = matching[0]
    if backend in ("auto", "cudnn") and norm is not None and actual_norm != norm:
        raise AssertionError(
            f"{backend} did not preserve the baseline normalization: {actual_norm} != {norm}"
        )
    check_ownership(args, arm)
    arm.update(
        norm=actual_norm,
        route=route,
        checks={"initial": choices[actual_norm]},
        samples={
            key: []
            for key in (
                "cpu_active_us",
                "host_enqueue_us",
                "completed_call_us",
                "replay_with_reset_us",
            )
        },
    )
    for _ in range(5):
        arm["state"].copy_(data["seed"])
        arm["run"]()
    torch.cuda.synchronize()
    # Supplying a native workspace intentionally retains the native provider.
    # Follow the provider that actually executed eagerly so graph comparisons
    # do not silently measure a different engine after auto gains cuDNN.
    captured = make_runner(
        args,
        data,
        backend,
        actual_norm,
        capture=True,
        capture_uses_cudnn=bool(route["cudnn_executions"]),
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        captured["state"].copy_(data["seed"])
        result, capture_route = observe_route(captured["run"])
        check_route(backend, capture_route, eager_route=route)
        capture_stream.synchronize()
        assert_result(args, result, references[actual_norm])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(args.unroll):
                captured["state"].copy_(data["seed"])
                captured["run"]()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    torch.cuda.synchronize()
    arm.update(
        captured=captured,
        graph=graph,
        capture_route=capture_route,
        capture_stream=capture_stream,
    )
    arm["checks"]["capture"] = assert_result(
        args, (captured["output"], captured["final"]), references[actual_norm]
    )
    return arm


def measure(args, arm, reference):
    run, state, seed = arm["run"], arm["state"], arm["data"]["seed"]
    batch = {key: [] for key in arm["samples"]}
    for _ in range(args.iters):
        state.copy_(seed)
        torch.cuda.synchronize()
        cpu0, wall0 = time.thread_time_ns(), time.perf_counter_ns()
        run()
        wall1, cpu1 = time.perf_counter_ns(), time.thread_time_ns()
        batch["cpu_active_us"].append((cpu1 - cpu0) / 1e3)
        batch["host_enqueue_us"].append((wall1 - wall0) / 1e3)
        torch.cuda.synchronize()
        state.copy_(seed)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        run()
        torch.cuda.synchronize()
        batch["completed_call_us"].append((time.perf_counter_ns() - start) / 1e3)
    arm["checks"]["last_timed_host"] = assert_result(
        args, (arm["output"], arm["final"]), reference
    )
    check_ownership(args, arm)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(args.iters):
        start.record()
        arm["graph"].replay()
        end.record()
        end.synchronize()
        batch["replay_with_reset_us"].append(
            start.elapsed_time(end) * 1000 / args.unroll
        )
    captured = arm["captured"]
    arm["checks"]["last_timed_replay"] = assert_result(
        args, (captured["output"], captured["final"]), reference
    )
    check_ownership(args, captured)
    for key, samples in batch.items():
        arm["samples"][key].append(samples)


def source_info(module):
    path = Path(module.__file__).resolve()
    revision = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        revision=revision,
    )


def main():
    args = arguments()
    import flashinfer
    import cudnn
    from flashinfer.cudnn import linear_attention

    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    data = make_data(args, 41)
    norms = (
        ("additive-1e-6",)
        if args.family == "kda" or args.gdn_normalize
        else ("pre-normalized",)
    )
    references = {norm: serial_reference(args, data, norm) for norm in norms}
    arms, records = {}, {}
    default_norm = None
    for backend in args.backends:
        with dispatch_policy(args, backend):
            arm = prepare_arm(args, data, backend, references, default_norm)
        if default_norm is None:
            default_norm = arm["norm"]
        arms[backend] = arm
    enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(args.repeats):
            order = (
                args.backends[repeat % len(args.backends) :]
                + args.backends[: repeat % len(args.backends)]
            )
            for backend in order:
                arm = arms[backend]
                with dispatch_policy(args, backend):
                    measure(args, arm, references[arm["norm"]])
    finally:
        if enabled:
            gc.enable()
    # Refresh actual captured inputs in place and separately test fresh eager
    # addresses. Keep both datasets alive to exclude allocator-address reuse.
    changed_args = argparse.Namespace(**vars(args))
    changed_args.lengths = list(reversed(args.lengths))
    changed = make_data(changed_args, 43)
    changed_references = {
        norm: serial_reference(changed_args, changed, norm) for norm in norms
    }
    for backend, arm in arms.items():
        with dispatch_policy(args, backend):
            fresh = make_runner(args, changed, backend, arm["norm"])
            fresh["state"].copy_(changed["seed"])
            actual = fresh["run"]()
        torch.cuda.synchronize()
        arm["checks"]["fresh_eager_addresses"] = assert_result(
            args, actual, changed_references[arm["norm"]]
        )
    for name, tensor in data.items():
        tensor.copy_(changed[name])
    for backend, arm in arms.items():
        arm["graph"].replay()
        torch.cuda.synchronize()
        cap = arm["captured"]
        arm["checks"]["changed_replay"] = assert_result(
            args, (cap["output"], cap["final"]), changed_references[arm["norm"]]
        )
        records[backend] = dict(
            normalization=arm["norm"],
            seq_order_supplied="seq_order" in data and backend != "cudnn",
            comparable_to_stock_auto=arm["norm"] == default_norm,
            route=arm["route"],
            capture_route=arm["capture_route"],
            checks=arm["checks"],
            measurements={
                key: dict(
                    median_us=statistics.median(
                        statistics.median(batch) for batch in batches
                    ),
                    repeat_medians_us=[statistics.median(batch) for batch in batches],
                    samples_us=batches,
                )
                for key, batches in arm["samples"].items()
            },
        )
    output = dict(
        args={**vars(args), "output": str(args.output)},
        results=records,
        versions=dict(
            flashinfer=flashinfer.__version__,
            frontend=cudnn.__version__,
            backend=cudnn.backend_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
        ),
        gpu=dict(
            name=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        ),
        affinity=sorted(os.sched_getaffinity(0)),
        sources={
            "flashinfer": source_info(flashinfer),
            "frontend": source_info(cudnn),
            "adapter": source_info(linear_attention),
            "public_family": source_info(
                sys.modules[
                    "flashinfer.kda"
                    if args.family == "kda"
                    else "flashinfer.gdn_prefill"
                ]
            ),
            "benchmark": source_info(sys.modules[__name__]),
        },
        input_layouts={
            name: dict(
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                dtype=str(tensor.dtype),
            )
            for name, tensor in data.items()
        },
        changed_lengths=changed_args.lengths,
        scope=__doc__,
        l2_policy="warm repeated input buffers; no cache flush",
        replay_reset_bytes=data["seed"].numel() * data["seed"].element_size(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {backend: item["measurements"] for backend, item in records.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
