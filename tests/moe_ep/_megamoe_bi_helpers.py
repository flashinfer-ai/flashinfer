# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Shared device-test checks, independent of MegaMoE's arithmetic helpers."""


def fail_stop_distributed(error, world):
    """Let torchrun terminate peers before failed collective cleanup can hang."""
    import os
    import sys
    import traceback

    import pytest

    if (
        world <= 1
        or int(os.environ.get("WORLD_SIZE", "1")) <= 1
        or isinstance(error, (pytest.skip.Exception, pytest.xfail.Exception))
    ):
        return
    try:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(1)


def all_gather_rows(value):
    """Gather packed storage bytes; NCCL need not support the FP4/FP8 dtype."""
    import torch
    import torch.distributed as dist

    raw = value.contiguous().view(torch.uint8)
    peers = [torch.empty_like(raw) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, raw)
    return torch.cat(peers, dim=0).view(value.dtype)


def assert_collectively(condition, device, context):
    """Fail every rank before the next fused collective can hang."""
    import torch
    import torch.distributed as dist

    ok = torch.tensor(int(bool(condition)), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert ok.item(), context


def assert_bits(actual, expected, context):
    import torch

    assert_collectively(
        torch.equal(
            actual.contiguous().view(torch.int16),
            expected.contiguous().view(torch.int16),
        ),
        actual.device,
        context,
    )


def assert_reference(actual, expected, tolerance, context):
    """Accuracy is separate from bitwise invariance and publication checks."""
    import torch

    error = (actual.float() - expected.float()).norm()
    scale = expected.float().norm().clamp_min(1e-6)
    assert_collectively(
        torch.isfinite(actual).all() and error / scale < tolerance,
        actual.device,
        f"{context}: relative L2={float(error / scale)}",
    )


def remote_routes(ids, rank, world, local_experts):
    """Exercise every source-to-remote destination, rather than one ring edge."""
    import torch

    rows = torch.arange(ids.shape[0], device=ids.device)
    destination = (rank + 1 + rows.remainder(world - 1)).remainder(world)
    result = ids.clone()
    result[:, -1] = destination * local_experts + (rows // (world - 1)).remainder(
        local_experts
    )
    return result


def boundary_batches(config, weights, x, ids, scores, layer_factory, plans):
    """Small batches for long K loops and both ends of the top-k domain."""
    from dataclasses import replace

    import torch
    from flashinfer.moe_ep import MoEEpTensors

    hidden = x.shape[1]
    with layer_factory(config, weights, 0, 1, 33, hidden) as layer:
        anchor = layer.forward(MoEEpTensors(x[:33], ids[:33], scores[:33])).clone()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
    for knobs in plans:
        with layer_factory(
            replace(config, knobs=knobs), weights, 0, 1, 17, hidden
        ) as layer:
            for n in (0, 1, 17):
                order = torch.arange(n, device=x.device)
                for selected in (order, order.flip(0), order.remainder(5)):
                    actual = layer.forward(
                        MoEEpTensors(x[selected], ids[selected], scores[selected])
                    )
                    assert_bits(actual, anchor[selected], "long-K/top-k batch and plan")
            order = torch.arange(17, device=x.device)
            output = torch.empty_like(anchor[order])
            inputs = MoEEpTensors(
                x[order].clone(),
                ids[order].clone(),
                scores[order].clone(),
                output=output,
            )
            layer.forward(inputs)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    layer.forward(inputs)
                for selected in (order.flip(0), order.remainder(5), order):
                    inputs.hidden_states.copy_(x[selected])
                    inputs.topk_ids.copy_(ids[selected])
                    inputs.topk_weights.copy_(scores[selected])
                    graph.replay()
                    assert_bits(output, anchor[selected], "long-K/top-k graph")
            finally:
                graph.reset()


def autotune_disk_reload(
    *,
    make_config,
    weights,
    x,
    ids,
    scores,
    layer_factory,
    provider,
    directory,
    monkeypatch,
):
    """Profile a public layer, discard winners, then require a real disk hit."""
    import importlib
    import json

    import torch
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import (
        MeasurementPolicy,
        ReplayCoverage,
        autotune_v2_reload,
    )
    from flashinfer.autotuner import AutoTuner
    from flashinfer.moe_ep import MoEEpTensors

    hidden = x.shape[1]
    native_path, managed_root = directory / "native.json", directory / "managed"
    inputs = MoEEpTensors(x[:17], ids[:17], scores[:17])
    tuner = AutoTuner.get()

    def tried():
        return dict(tuner.stats.tuned_op_total_configs)

    def receipt_is(layer, receipt):
        workspace = layer._workspace
        frontend = getattr(workspace, "_frontend", workspace)
        assert layer.batch_invariant_fingerprint == receipt
        assert frontend.batch_invariant_fingerprint == receipt

    autotune_v2_reload()
    try:
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        with layer_factory(make_config({}), weights, 0, 1, 257, hidden) as layer:
            anchor = layer.forward(inputs).clone()
            receipt = layer.batch_invariant_fingerprint
            assert receipt and torch.isfinite(anchor).all()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(native_path))
        policy = MeasurementPolicy(execution_mode="eager", cold_l2=False)
        before = tried()
        with (
            layer_factory(make_config("auto"), weights, 0, 1, 257, hidden) as layer,
            autotune_v2(
                cache_root=managed_root,
                cache_scope="context",
                measurement_policy=policy,
            ),
        ):
            layer.warmup(inputs)
            assert_bits(layer.forward(inputs), anchor, "fresh public autotune")
            receipt_is(layer, receipt)
            winner = layer.get_tuned_knobs()
        after = tried()
        assert native_path.is_file(), "autotune did not publish its native disk cache"
        entries = json.loads(native_path.read_text())["entries"]
        assert any(
            record.get(
                "batch_invariant_fingerprint",
                record.get("problem", {}).get("batch_invariant_fingerprint"),
            )
            == receipt
            for record in entries
        )
        autotune_v2_reload()
        before_replay = tried()
        if provider == "legacy":
            with layer_factory(make_config(None), weights, 0, 1, 257, hidden) as layer:
                cache = importlib.import_module(layer._kernel._knob_cache_module)
                selected, source = cache.resolve_knobs(
                    **layer._kernel._tuning_key(layer._fleet_params)
                )
                assert source == "cache" and selected == winner
                assert_bits(layer.forward(inputs), anchor, "native disk-cache reload")
                receipt_is(layer, receipt)
                assert layer._kernel._workspace_knobs(layer._fleet_params) == winner
        else:
            assert any(value > before.get(op, 0) for op, value in after.items()), (
                "fresh context did not profile"
            )
            assert list(managed_root.rglob("*.json")), (
                "managed disk cache was not written"
            )
            monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
            report = ReplayCoverage()
            with (
                layer_factory(make_config("auto"), weights, 0, 1, 257, hidden) as layer,
                autotune_v2(
                    mode="replay",
                    cache_root=managed_root,
                    cache_scope="context",
                    measurement_policy=policy,
                    replay_report=report,
                    strict_replay=True,
                ),
            ):
                layer.warmup(inputs)
                assert_bits(layer.forward(inputs), anchor, "strict managed disk replay")
                receipt_is(layer, receipt)
                assert layer.get_tuned_knobs() == winner
            assert report.hits > 0 and report.misses == 0
        assert tried() == before_replay, "disk-cache replay unexpectedly profiled"
    finally:
        autotune_v2_reload()


def distributed_reference(
    *,
    config,
    weights,
    packed,
    scales,
    ids,
    scores,
    reference,
    layer_factory,
    hidden,
    rank,
    world,
    tolerance,
):
    """All-gather oracle, actual remote sensitivity, masking and graph replay.

    ``reference`` uses canonical packed weights gathered by the caller and
    independent Torch arithmetic. It never calls a production FC kernel or
    shares the device activation/quantization chain.
    """
    import json

    import torch
    from flashinfer.moe_ep import MoEEpTensors

    count = ids.shape[0]
    global_packed = all_gather_rows(packed)
    global_scales = None if scales is None else all_gather_rows(scales)
    global_ids = all_gather_rows(ids)
    global_scores = all_gather_rows(scores)
    expected = reference(global_packed, global_scales, global_ids, global_scores)
    expected = expected[rank * count : (rank + 1) * count]
    with layer_factory(
        config, weights, rank, world, count + 1, hidden, quantize=False
    ) as layer:
        output = torch.empty(
            (count, hidden), device=packed.device, dtype=torch.bfloat16
        )
        inputs = MoEEpTensors(
            packed, ids.clone(), scores.clone(), scales=scales, output=output
        )
        baseline = layer.forward(inputs).clone()
        assert_reference(
            baseline, expected, tolerance, "distributed independent oracle"
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                layer.forward(inputs)
            for peer in range(world):
                if peer == rank:
                    continue
                drop = ids.div(weights.w13.shape[0], rounding_mode="floor") == peer
                assert_collectively(
                    drop.any(), packed.device, "missing source-to-peer traffic"
                )
                masked_ids = ids.masked_fill(drop, -1)
                masked_scores = scores.masked_fill(drop, 0)
                masked_expected = reference(packed, scales, masked_ids, masked_scores)
                assert_collectively(
                    (
                        expected.view(torch.int16) != masked_expected.view(torch.int16)
                    ).any(),
                    packed.device,
                    "remote route has no independently observable contribution",
                )
                # Prime with valid nonzero remote outputs, mask them with NaN
                # scores, then restore the same graph and buffer addresses.
                for masked in (False, True, False):
                    inputs.topk_ids.copy_(masked_ids if masked else ids)
                    inputs.topk_weights.copy_(
                        scores.masked_fill(drop, float("nan")) if masked else scores
                    )
                    graph.replay()
                    if masked:
                        assert_reference(
                            output, masked_expected, tolerance, "masked remote oracle"
                        )
                        assert_collectively(
                            (
                                output.view(torch.int16) != baseline.view(torch.int16)
                            ).any(),
                            packed.device,
                            "removing remote traffic did not change actual output bits",
                        )
                    else:
                        assert_bits(output, baseline, "remote valid/masked/valid graph")
        finally:
            graph.reset()
        print(
            json.dumps(
                dict(
                    event="megamoe_bi_distributed_oracle_passed",
                    rank=rank,
                    world_size=world,
                    hidden=hidden,
                    intermediate=config.intermediate_size,
                    receipt=layer.batch_invariant_fingerprint,
                    remote_peers=[peer for peer in range(world) if peer != rank],
                )
            ),
            flush=True,
        )
