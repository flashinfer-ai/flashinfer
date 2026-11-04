"""Legacy NVFP4 GIN qualification through the formal FI layer.

Launch on two independent NVLink domains, two ranks per domain. A single
NVLink domain is intentionally rejected by the runtime's rail/LSA preflight.
This file executes correctness checks only and reports no timings.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.config import (
    Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


def _bootstrap():
    topology = tuple(
        int(os.environ.get(name, "1"))
        for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE")
    )
    if topology != (4, 2, 2):
        pytest.skip("requires N2R2 torchrun: two worker groups, two local ranks each")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    return BootstrapConfig(
        world_size=4,
        rank=dist.get_rank(),
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
    )


def _layer(bootstrap, *, retain_weights=False, quantize_input=True, **changes):
    generator = torch.Generator(device="cuda").manual_seed(101 + bootstrap.rank)
    config = dict(intermediate_size=1536, top_k=4, internode=True, ranks_per_node=2)
    config.update(changes)
    # Small finite weights keep the independent Torch reference well scaled.
    weights = MoEWeightPack(
        w13=torch.randn(
            (12, 3072, 7168), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 0.01,
        w2=torch.randn(
            (12, 7168, 1536), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 0.01,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=48, max_tokens_per_rank=32, token_hidden_size=7168
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**config),
            quantize_input=quantize_input,
            preprocess_weights=True,
        ),
    )
    if retain_weights:
        layer._gin_oracle_weights = weights
    return layer


def _inputs(rank, iteration, case):
    generator = torch.Generator(device="cuda").manual_seed(301 + rank + iteration * 4)
    n = 32 if case == "balanced" else (32, 7, 32, 0 if rank == 1 else 13)[iteration % 4]
    x = torch.randn((n, 7168), dtype=torch.bfloat16, device="cuda", generator=generator)
    if case == "balanced":
        scores = torch.rand((n, 48), device="cuda", generator=generator)
        weights, ids = scores.topk(4, dim=-1)
        return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
    node = rank // 2
    routes = (
        [rank * 12 + i for i in range(4)],
        [(1 - node) * 24 + i for i in range(4)],
        [0, 1, 24, 25],
    )[iteration % 3]
    ids = torch.tensor(routes, dtype=torch.int64, device="cuda").repeat(n, 1)
    scores = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cuda"
    ).repeat(n, 1)
    if iteration == 8:
        ids[::3, 1] = -1
        scores[::3, 1] = float("nan")
    return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=scores)


@pytest.mark.parametrize("case", ["balanced", "matrix"])
def test_nvfp4_gin_public_layer_relaunch(case):
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        for iteration in range(12):
            inputs = _inputs(bootstrap.rank, iteration, case)
            output = layer.forward(inputs)
            torch.cuda.synchronize()
            assert output.shape == inputs.hidden_states.shape
            assert torch.isfinite(output).all()
            dist.barrier()
            print(
                f"[rank {bootstrap.rank}] GIN_PUBLIC_RELAUNCH {iteration} finite PASS",
                flush=True,
            )
    except BaseException:
        failed = True
        raise
    finally:
        # Preserve the first failure rather than calling CUDA again after a
        # fatal launch error; the launcher exits the whole failed job.
        if not failed:
            layer.destroy()


@pytest.mark.parametrize(
    "changes",
    [
        {"knobs": "auto"},
        {"enable_fc1_activation_per_token_scale": True},
        {"swiglu_alpha": 1.702, "swiglu_beta": 1.0},
        {"ranks_per_node": 4},
        {"fc1_alpha": 2.0},
    ],
)
def test_nvfp4_gin_public_config_rejection(changes):
    bootstrap = _bootstrap()
    with pytest.raises((RuntimeError, ValueError), match="GIN|gin|internode"):
        _layer(bootstrap, **changes)


def test_nvfp4_gin_public_rank_local_errors():
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        layer.forward(_inputs(bootstrap.rank, 0, "balanced"))
        torch.cuda.synchronize()
        for kind in ("route", "dtype", "capacity", "output", "epilogue_scale"):
            inputs = _inputs(bootstrap.rank, 1, "balanced")
            if bootstrap.rank == 1:
                if kind == "route":
                    inputs.topk_ids[0, 0] = 48
                elif kind == "dtype":
                    inputs.hidden_states = inputs.hidden_states.float()
                elif kind == "capacity":
                    inputs.hidden_states = inputs.hidden_states.repeat(2, 1)
                    inputs.topk_ids = inputs.topk_ids.repeat(2, 1)
                    inputs.topk_weights = inputs.topk_weights.repeat(2, 1)
                elif kind == "epilogue_scale":
                    inputs.fc1_alpha = torch.ones(
                        12, device="cuda", dtype=torch.float32
                    )
                else:
                    inputs.output = torch.empty(
                        (32, 1), device="cuda", dtype=torch.bfloat16
                    )
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.forward(inputs)
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.stage_inputs(inputs)
            dist.barrier()
        result = layer.forward(_inputs(bootstrap.rank, 2, "balanced"))
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_rank_local_errors_control_headers(monkeypatch):
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
        collective_gin_preflight,
    )

    object_reports = []
    original = dist.all_gather_object

    def detailed_report(*args, **kwargs):
        object_reports.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(dist, "all_gather_object", detailed_report)
    collective_gin_preflight(dist, phase="ready", signature=(32,), local_error=None)
    assert not object_reports
    for kind in ("phase", "configuration", "error", "subgroup"):
        different = bootstrap.rank == 1
        before = len(object_reports)
        with pytest.raises(GinPreflightError, match="GIN"):
            collective_gin_preflight(
                dist,
                phase="different-boundary"
                if different and kind == "phase"
                else "ready",
                signature=(64 if different and kind == "configuration" else 32,),
                local_error="rank-local rejection"
                if different and kind == "error"
                else None,
                requested_group=object() if different and kind == "subgroup" else None,
            )
        assert len(object_reports) == before + 1
        # Rejection must leave the common control sequence usable by all ranks.
        collective_gin_preflight(
            dist, phase="recovered", signature=(32,), local_error=None
        )
        assert len(object_reports) == before + 1


def test_nvfp4_gin_public_sessions_and_recreation():
    bootstrap = _bootstrap()
    layers = [_layer(bootstrap), _layer(bootstrap)]
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        expected = []
        for layer in layers:
            expected.append(layer.forward(inputs).clone())
            torch.cuda.synchronize()
        identities = [
            layer._workspace._frontend._gin_session.identity for layer in layers
        ]
        assert identities[0] != identities[1]
        # Every rank enters the public layer, but one uses a different live
        # communicator. Reject collectively before launching either session.
        with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
            layers[bootstrap.rank == 1].forward(inputs)
        for layer, reference in zip(layers, expected, strict=True):
            actual = layer.forward(inputs)
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        retired = layers.pop()
        retired.destroy()
        retired.destroy()
        layers.append(_layer(bootstrap))
        result = layers[-1].forward(inputs)
        torch.cuda.synchronize()
        assert torch.equal(result.view(torch.int16), expected[-1].view(torch.int16))
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            for layer in reversed(layers):
                layer.destroy()


def test_nvfp4_gin_public_rank_local_stream_rebinding():
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        expected = layer.forward(inputs).clone()
        torch.cuda.synchronize()
        mega = layer._workspace._frontend._mega
        compiled = mega.compiled
        streams = []

        def tracked(**kwargs):
            streams.append(int(kwargs["stream"]))
            return compiled(**kwargs)

        mega.compiled = tracked
        layer.forward(inputs)  # Collectively prepare the instrumented target.
        torch.cuda.synchronize()
        default = torch.cuda.current_stream()
        alternate = torch.cuda.Stream()
        for iteration in range(3):
            target = (
                alternate if bootstrap.rank == 1 and iteration % 2 == 0 else default
            )
            target.wait_stream(default)
            with torch.cuda.stream(target):
                actual = layer.forward(inputs)
                assert streams[-1] == target.cuda_stream
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        mega.compiled = compiled
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_rank_local_device_binding():
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        layer.forward(_inputs(bootstrap.rank, 0, "balanced"))
        torch.cuda.synchronize()
        bound_device = torch.cuda.current_device()
        target = (bound_device + 1) % 2 if bootstrap.rank == 1 else bound_device
        with torch.cuda.device(target):
            inputs = _inputs(bootstrap.rank, 1, "balanced")
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.forward(inputs)
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.stage_inputs(inputs)
        actual = layer.forward(_inputs(bootstrap.rank, 2, "balanced"))
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


@pytest.mark.parametrize("quantize_input", [True, False], ids=["bf16", "packed"])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
def test_nvfp4_gin_public_graph_replay(quantize_input, partial, monkeypatch, tmp_path):
    """Capture the public staging/kernel/reduce path and adjacent GPU work."""
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_mega_launch_thunk,
        nvfp4_quantize_per_block_16,
    )

    layer = _layer(bootstrap, quantize_input=quantize_input)
    failed = False
    graphs, outputs = [], []
    n = (0, 7, 32, 13)[bootstrap.rank] if partial else 32

    def batch(iteration):
        t = _inputs(bootstrap.rank, iteration, "balanced")
        # Keep the graph's shape bucket fixed while changing payloads/routes.
        if iteration % 3 != 0:
            node = bootstrap.rank // 2
            expert = bootstrap.rank * 12 if iteration % 3 == 1 else (1 - node) * 24
            t.topk_ids.copy_(
                torch.arange(expert, expert + 4, device="cuda").repeat(32, 1)
            )
        if iteration % 4 == 3:
            t.topk_ids[::3, 1] = -1
            t.topk_weights[::3, 1] = float("nan")
        if not quantize_input:
            t.hidden_states, t.scales = nvfp4_quantize_per_block_16(
                t.hidden_states.float(), 1.0
            )
            t.scales = t.scales[:n].contiguous()
        t.hidden_states = t.hidden_states[:n].contiguous()
        t.topk_ids = t.topk_ids[:n].contiguous()
        t.topk_weights = t.topk_weights[:n].contiguous()
        return t

    try:
        static = batch(0)
        layer.warmup(static)
        torch.cuda.synchronize()
        mega = layer._workspace._frontend._mega
        compiled, streams = mega.compiled, []

        def tracked(**kwargs):
            streams.append(int(kwargs["stream"]))
            return compiled(**kwargs)

        monkeypatch.setattr(mega, "compiled", tracked)
        layer.forward(static)  # Prepare the instrumented callable outside capture.
        torch.cuda.synchronize()
        for mode in ("forward", "split", "prepared"):
            prepared = None
            if mode == "prepared":
                weights = layer._get_transformed_weights()
                prepared = nvfp4_mega_launch_thunk(
                    weights[0], weights[1], layer._workspace
                )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                # A caller owns capture and can include surrounding kernels.
                t = MoEEpTensors(
                    hidden_states=static.hidden_states.clone(),
                    scales=static.scales,
                    topk_ids=static.topk_ids,
                    topk_weights=static.topk_weights,
                )
                if prepared is not None:
                    layer.stage_inputs(t)
                    prepared()
                    y = layer._workspace.output_activation[:n]
                elif mode == "split":
                    layer.stage_inputs(t)
                    y = layer.compute_staged(output=None)
                else:
                    y = layer.forward(t)
                output = y.float() * 2 + 1
            assert streams[-1] == stream.cuda_stream
            graphs.append(graph)
            outputs.append(output)
        dist.barrier()

        def no_host_agreement(*args, **kwargs):
            raise AssertionError("graph replay must not execute Python GIN agreement")

        def poison_workspace():
            # Eager reference calls have already staged current inputs. Erase
            # them so every graph must restore its own recorded staging.
            workspace = layer._workspace
            workspace.x.view(torch.uint8).zero_()
            workspace.x_sf.view(torch.uint8).zero_()
            workspace.topk_idx.fill_(-1)
            workspace.topk_weights.fill_(float("nan"))
            workspace.output_activation.fill_(float("nan"))

        for iteration in range(1, 13):
            current = batch(iteration)
            expected = layer.forward(current).float() * 2 + 1
            static.hidden_states.copy_(current.hidden_states)
            static.topk_ids.copy_(current.topk_ids)
            static.topk_weights.copy_(current.topk_weights)
            if static.scales is not None:
                static.scales.copy_(current.scales)
            torch.cuda.synchronize()
            for graph, actual in zip(graphs, outputs, strict=True):
                # An empty graph, stale workspace result, or missing staging
                # cannot pass by reusing the just-computed eager result.
                poison_workspace()
                actual.fill_(float("nan"))
                calls = len(streams)
                with monkeypatch.context() as context:
                    context.setattr(
                        layer._workspace._frontend, "_gin_agree", no_host_agreement
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                assert len(streams) == calls
                assert torch.equal(actual, expected), (bootstrap.rank, iteration)
            dist.barrier()

        snapshots = [torch.empty_like(expected) for _ in range(12)]
        torch.cuda.synchronize()
        calls = len(streams)
        with monkeypatch.context() as context:
            context.setattr(layer._workspace._frontend, "_gin_agree", no_host_agreement)
            for iteration, snapshot in enumerate(snapshots):
                index = iteration % len(graphs)
                poison_workspace()
                outputs[index].fill_(float("nan"))
                graphs[index].replay()
                snapshot.copy_(outputs[index])
            # No host sync, control collective or allocation between replays.
            torch.cuda.synchronize()
        assert len(streams) == calls
        assert all(torch.equal(snapshot, expected) for snapshot in snapshots)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            graphs[0].replay()
            torch.cuda.synchronize()
        trace = tmp_path / "gin-graph.json"
        prof.export_chrome_trace(str(trace))
        names = [
            event["name"]
            for event in json.loads(trace.read_text())["traceEvents"]
            if event.get("cat") == "kernel"
        ]
        assert sum("fc1fc2_kernel_impl" in name for name in names) == 1, names
        assert sum("kernel_cutlass__reduce_" in name for name in names) == 1, names
        assert torch.equal(outputs[0], expected)
        print(
            f"[rank {bootstrap.rank}] GIN_GRAPH_PASS packed={not quantize_input} "
            f"tokens={n} graphs=3 replays=48 queued=12 mega=1 reduce=1",
            flush=True,
        )
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            graphs.clear()
            outputs.clear()
            layer.destroy()


@pytest.mark.parametrize("method", ["forward", "stage_inputs"])
def test_nvfp4_gin_public_mixed_capture_rejection(method):
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
    )

    layer = _layer(bootstrap)
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        expected = layer.forward(inputs).clone()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        capture = (
            torch.cuda.graph(graph, stream=stream)
            if bootstrap.rank == 1
            else nullcontext()
        )
        with pytest.raises(GinPreflightError, match="configuration differs"), capture:
            getattr(layer, method)(inputs)
        actual = layer.forward(inputs)
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        del graph
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_graph_requires_warmup():
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
    )

    layer = _layer(bootstrap)
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        # Only one rank is cold: all peers must reject before recording work.
        layer.warmup(inputs)
        torch.cuda.synchronize()
        if bootstrap.rank == 1:
            layer._kernel._thunk_states.clear()
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(GinPreflightError, match="warmup"),
            torch.cuda.graph(graph),
        ):
            layer.forward(inputs)
        # Rewarm all ranks in the same preparation phase after the rejection.
        layer._kernel._thunk_states.clear()
        output = layer.forward(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        del graph
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_graph_metadata_error_recovery():
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
    )

    layer = _layer(bootstrap)
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        layer.warmup(inputs)
        torch.cuda.synchronize()
        bad = _inputs(bootstrap.rank, 1, "balanced")
        if bootstrap.rank == 1:
            bad.hidden_states = bad.hidden_states.float()
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(GinPreflightError, match="dtype"),
            torch.cuda.graph(graph),
        ):
            layer.forward(bad)
        output = layer.forward(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        del graph
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_prestaged_inputs():
    bootstrap = _bootstrap()
    layers = [_layer(bootstrap), _layer(bootstrap, quantize_input=False)]
    failed = False
    try:
        original = _inputs(bootstrap.rank, 1, "matrix")
        expected = layers[0].forward(original).clone()
        torch.cuda.synchronize()
        workspace = layers[0]._workspace
        n = original.hidden_states.shape[0]
        inputs = MoEEpTensors(
            hidden_states=workspace.x[:n].clone(),
            scales=workspace.x_sf[:n].clone(),
            topk_ids=original.topk_ids,
            topk_weights=original.topk_weights,
        )
        actual = layers[1].forward(inputs)
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            for layer in reversed(layers):
                layer.destroy()


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
@pytest.mark.parametrize("failure", ["overflow", "mismatch", "unsupported"])
def test_nvfp4_gin_compile_bucket_collective_rejection(capturing, failure):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
    )

    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    inputs = _inputs(bootstrap.rank, 0, "balanced")
    layer.warmup(inputs)
    if failure == "overflow" and bootstrap.rank != 0:
        inputs = MoEEpTensors(
            hidden_states=inputs.hidden_states[:7],
            topk_ids=inputs.topk_ids[:7],
            topk_weights=inputs.topk_weights[:7],
        )
    expected = layer.forward(inputs).clone()
    torch.cuda.synchronize()
    bucket = (
        16
        if failure == "overflow"
        else (32 if failure == "unsupported" or bootstrap.rank != 0 else None)
    )
    graph = torch.cuda.CUDAGraph()
    context = torch.cuda.graph(graph) if capturing else nullcontext()
    match = {
        "overflow": "live token count",
        "mismatch": "configuration differs",
        "unsupported": "does not support compile token buckets",
    }[failure]
    with pytest.raises(GinPreflightError, match=match), context:
        layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
    layer.stage_inputs(inputs)
    actual = layer.compute_staged(output=torch.empty_like(expected))
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    del graph
    layer.destroy()


def _gin_factory_args(bootstrap):
    return dict(
        num_total_experts=48,
        num_max_tokens=32,
        num_topk=4,
        hidden=7168,
        intermediate=3072,
        rank=bootstrap.rank,
        world_size=4,
        internode=True,
        ranks_per_node=2,
    )


@pytest.mark.parametrize("all_capture", [False, True], ids=["one-rank", "all-ranks"])
@pytest.mark.parametrize("entry", ["workspace", "session"])
def test_nvfp4_gin_capture_initialization_collective_rejection(
    all_capture, entry, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import gin, nvfp4

    bootstrap = _bootstrap()
    created = []
    original = gin.GinSession.__init__

    def track(self, **kwargs):
        created.append(self)
        original(self, **kwargs)

    graph = torch.cuda.CUDAGraph()
    context = (
        torch.cuda.graph(graph) if all_capture or bootstrap.rank == 1 else nullcontext()
    )
    with monkeypatch.context() as patch:
        patch.setattr(gin.GinSession, "__init__", track)
        with pytest.raises(gin.GinPreflightError, match="capture"), context:
            if entry == "workspace":
                nvfp4.get_symm_buffer_for_mega_moe(**_gin_factory_args(bootstrap))
            else:
                gin.create_gin_session(
                    rank=bootstrap.rank,
                    world_size=4,
                    ranks_per_node=2,
                    device=torch.cuda.current_device(),
                    layout=gin.build_gin_arena_layout(
                        transport_bytes=128,
                        num_tokens=1,
                        hidden=64,
                        num_topk=4,
                        shared_workspace_bytes=128,
                    ),
                    signal_count=1,
                    signature=("capture-init",),
                )
    assert created == []
    del graph
    workspace = nvfp4.get_symm_buffer_for_mega_moe(**_gin_factory_args(bootstrap))
    workspace.destroy()


@pytest.mark.parametrize("allocation", ["view", "routes", "output", "epilogue"])
def test_nvfp4_gin_workspace_allocation_failure_cleanup(allocation, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import gin, nvfp4

    bootstrap = _bootstrap()
    sessions = []
    create = gin.create_gin_session

    def track(**kwargs):
        session = create(**kwargs)
        sessions.append(session)
        return session

    def inject(original, predicate):
        def wrapped(*args, **kwargs):
            if bootstrap.rank == 1 and predicate(*args, **kwargs):
                raise torch.OutOfMemoryError("injected local allocation failure")
            return original(*args, **kwargs)

        return wrapped

    with monkeypatch.context() as patch:
        patch.setattr(gin, "create_gin_session", track)
        if allocation == "view":
            patch.setattr(
                gin.GinSession,
                "view",
                inject(
                    gin.GinSession.view, lambda self, name, **kw: name == "activation"
                ),
            )
        elif allocation == "routes":
            patch.setattr(
                torch,
                "full",
                inject(torch.full, lambda shape, *a, **kw: shape == (32, 4)),
            )
        elif allocation == "output":
            patch.setattr(
                torch, "empty", inject(torch.empty, lambda *a, **kw: a == ((32, 7168),))
            )
        else:
            patch.setattr(
                nvfp4,
                "_resolve_per_expert_epilogue",
                inject(
                    nvfp4._resolve_per_expert_epilogue,
                    lambda name, *a, **kw: name == "fc1_norm_const",
                ),
            )
        with pytest.raises(
            gin.GinPreflightError, match="injected local allocation failure"
        ):
            nvfp4.get_symm_buffer_for_mega_moe(**_gin_factory_args(bootstrap))
    assert len(sessions) == 1
    session = sessions[0]
    assert session._closed and session.root is None and session.views == {}
    assert session.comm is None and session.dev_comm is None and session.window is None
    # Every rank can create another registered session after coordinated cleanup.
    workspace = nvfp4.get_symm_buffer_for_mega_moe(**_gin_factory_args(bootstrap))
    workspace.destroy()
