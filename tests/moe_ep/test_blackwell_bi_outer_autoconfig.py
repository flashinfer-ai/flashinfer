# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell/Rubin complete-forward BI selection, capture and disk replay."""

from contextlib import contextmanager
from dataclasses import replace
import os

import pytest

from ._megamoe_bi_helpers import assert_bits, fail_stop_distributed
from .test_nvfp4_megamoe_batch_invariant import _layer, _problem

pytestmark = pytest.mark.gpu
KINDS = ("bf16", "nvfp4", "w8a16")


def _require():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("outer BI requires SM100/SM103/SM107")


def _configs(kind):
    from flashinfer import moe_ep

    if kind == "bf16":
        first = moe_ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=512,
            top_k=3,
            batch_invariant=True,
            knobs={},
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        alternate = {
            "load_balance_mode": "atomic_counter",
            "token_back_mode": "reuse_dispatch_warps",
        }
    elif kind == "w8a16":
        first = moe_ep.Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=512,
            top_k=3,
            batch_invariant=True,
            knobs={},
            kind="bf16_mxfp8_e4m3",
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        alternate = {
            "load_balance_mode": "atomic_counter",
            "token_back_mode": "reuse_dispatch_warps",
        }
    else:
        from .test_nvfp4_megamoe_batch_invariant import _config

        first = replace(
            _config("standard", True, 512, 512, knobs={}),
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        alternate = {
            "schedule_policy": ("grouped", None),
            "work_id_mode": "atomic_counter",
            "token_back_mode": "reuse_dispatch_warps",
        }
    return first, replace(first, knobs=alternate)


@contextmanager
def _outer(configs, weights, rank, world):
    import torch
    from flashinfer import moe_ep

    # The selector consumes [up,gate]; direct Mega tests use [gate,up].
    intermediate = weights.w2.shape[-1]
    selector_weights = moe_ep.MoEWeightPack(
        w13=torch.cat(
            (weights.w13[:, intermediate:], weights.w13[:, :intermediate]), dim=1
        ),
        w2=weights.w2,
    )
    layer = moe_ep.MoEEpLayer(
        bootstrap=moe_ep.BootstrapConfig(
            rank=rank, world_size=world, auto_bootstrap=world == 1
        ),
        fleet_params=moe_ep.FleetParams(
            num_experts=4 * world, max_tokens_per_rank=129, token_hidden_size=512
        ),
        weights=selector_weights,
        backend=moe_ep.AutoConfig(
            candidates=tuple(moe_ep.MegaConfig(config) for config in configs)
        ),
    )
    try:
        yield layer
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        layer.destroy()


def _run(kind, rank, world, monkeypatch, directory, *, force_inner=False):
    import torch
    import torch.distributed as dist
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import (
        MeasurementPolicy,
        ReplayCoverage,
        autotune_v2_reload,
    )
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.core.selection import Collective
    from ._megamoe_bi_helpers import remote_routes

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    configs = _configs(kind)
    x, scores, ids, weights = _problem(rank, world, 512, 512)
    if world > 1:
        ids = remote_routes(ids, rank, world, 4)
    with _layer(configs[0], weights, rank, world, 129, 512) as layer:
        anchor = layer.forward(MoEEpTensors(x[:129], ids[:129], scores[:129])).clone()
        receipt = layer.batch_invariant_fingerprint
    if force_inner:
        from flashinfer.autotuner import AutoTuner
        from flashinfer.moe_ep.core.selection import NAMESPACE
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
            tuning_protocol,
        )

        enumerate_candidates = tuning_protocol.sm100_standard_candidates
        monkeypatch.setattr(
            tuning_protocol,
            "sm100_standard_candidates",
            lambda *args, **kwargs: enumerate_candidates(*args, **kwargs)[:2],
        )
        configs = (replace(configs[0], knobs="auto"), configs[1])
        tuner = AutoTuner.get()
        choose = tuner.choose_with_protocol

        def force_outer(custom_op, protocol, **kwargs):
            # Exercise forced complete-forward preparation through the public
            # layer. The native child protocol and disk replay keep their policy.
            if custom_op == NAMESPACE and tuner.is_tuning_mode:
                kwargs["force_tune"] = True
            return choose(custom_op, protocol, **kwargs)

        monkeypatch.setattr(tuner, "choose_with_protocol", force_outer)
    args = MoEEpTensors(
        x[:17].clone(),
        ids[:17].clone(),
        scores[:17].clone(),
        output=torch.empty_like(x[:17]),
        runtime_max_tokens_per_rank=17,
    )
    options = dict(
        cache_root=directory,
        cache_scope="context",
        measurement_policy=MeasurementPolicy(execution_mode="eager", cold_l2=False),
    )
    autotune_v2_reload()
    original_measure = Collective.measure
    measurements = []

    def measured(collective, *values, **kwargs):
        result = original_measure(collective, *values, **kwargs)
        measurements.append(result)
        return result

    monkeypatch.setattr(Collective, "measure", measured)
    try:
        with _outer(configs, weights, rank, world) as layer:
            with autotune_v2(**options):
                layer.warmup(args)
                assert_bits(layer(args), anchor[:17], "outer fresh profiling")
                winner = layer.get_selected_plan()
                if force_inner:
                    records = AutoTuner.get().get_protocol_records(
                        tuning_protocol.NAMESPACE
                    )
                    assert records, (
                        "forced outer selection did not record an inner search"
                    )
                    assert all(
                        record.get("timing_boundary") == "complete_forward"
                        for record in records.values()
                    )
            assert measurements, "outer selection did not profile complete forwards"
            assert winner["candidate"] in (0, 1)
            assert isinstance(winner["selected"]["knobs"], dict)
            assert ("kernel_variant" in winner["selected"]) == (kind == "nvfp4")
            plan = layer._default._plan
            assert plan.layer.batch_invariant_fingerprint == receipt
            if world > 1:
                winners = [None] * world
                dist.all_gather_object(winners, winner)
                assert all(item == winner for item in winners)
            # Lookup and capture must retain the tuning context's policy identity.
            with autotune_v2(mode="replay", **options):
                layer.warmup(args)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        layer(args)
                    for selected in (
                        torch.arange(17, device="cuda").flip(0),
                        torch.arange(17, device="cuda").remainder(5),
                    ):
                        args.hidden_states.copy_(x[selected])
                        args.topk_ids.copy_(ids[selected])
                        args.topk_weights.copy_(scores[selected])
                        graph.replay()
                        assert_bits(args.output, anchor[selected], "outer winner graph")
                        assert layer.get_selected_plan() == winner
                finally:
                    graph.reset()
                for n in (0, 1, 17):
                    live = max(0, n - 1) if world > 1 and rank == 0 else n
                    inputs = MoEEpTensors(
                        x[:live],
                        ids[:live],
                        scores[:live],
                        output=torch.empty_like(x[:live]),
                        runtime_max_tokens_per_rank=17,
                    )
                    assert_bits(
                        layer(inputs), anchor[:live], "outer winner uneven serving rows"
                    )
                    assert layer.get_selected_plan() == winner
        assert list(directory.rglob("*.json")), "outer persistent cache was not written"
        autotune_v2_reload()
        measurements.clear()
        replay_args = MoEEpTensors(
            x[:17].clone(),
            ids[:17].clone(),
            scores[:17].clone(),
            output=torch.empty_like(x[:17]),
            runtime_max_tokens_per_rank=17,
        )
        coverage = ReplayCoverage()
        with (
            _outer(configs, weights, rank, world) as layer,
            autotune_v2(
                mode="replay",
                strict_replay=world == 1,
                replay_report=coverage,
                **options,
            ),
        ):
            layer.warmup(replay_args)
            assert_bits(layer(replay_args), anchor[:17], "outer managed disk replay")
            assert layer.get_selected_plan() == winner
        assert not measurements, "strict disk replay unexpectedly profiled"
        assert coverage.hits > 0 and coverage.misses == 0
    finally:
        autotune_v2_reload()


@pytest.mark.parametrize("kind", KINDS)
def test_blackwell_batch_invariant_outer_autoconfig(kind, monkeypatch, tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank outer selection")
    _require()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run(kind, 0, 1, monkeypatch, tmp_path)


def test_blackwell_batch_invariant_outer_forced_inner_search_blackwell_only(
    monkeypatch, tmp_path
):
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank forced outer selection")
    _require()
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("this forced-inner probe uses Blackwell's native tuning protocol")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run("nvfp4", 0, 1, monkeypatch, tmp_path, force_inner=True)


def test_blackwell_batch_invariant_outer_caller_tensor_updates_blackwell_only(
    monkeypatch,
):
    import torch
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2_reload
    from flashinfer.moe_ep import FinalizeMode, MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank caller-owned tensor regression")
    _require()
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("Rubin block-scaled kernels do not expose persistent output")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    alpha = torch.ones(4, dtype=torch.float32, device="cuda")
    first, second = _configs("nvfp4")
    first = replace(
        first,
        finalize_mode=FinalizeMode.PERSISTENT,
        num_valid_tokens_tensor=count,
        fc2_alpha=alpha,
    )
    configs = (first, replace(first, knobs=second.knobs))
    x, scores, ids, weights = _problem(0, 1, 512, 512)
    args = MoEEpTensors(x[:17], ids[:17], scores[:17], output=torch.empty_like(x[:17]))
    direct_args = replace(args, output=torch.empty_like(args.output))
    graph = torch.cuda.CUDAGraph()
    autotune_v2_reload()
    try:
        with (
            _layer(first, weights, 0, 1, 129, 512) as direct,
            _outer(configs, weights, 0, 1) as layer,
            autotune_v2(
                persistent_cache=False,
                cache_scope="context",
                measurement_policy=MeasurementPolicy(
                    execution_mode="eager", cold_l2=False
                ),
            ),
        ):
            try:
                # Construction must borrow the original zero-valued live count.
                # Updating it before warmup must affect the selected plan too.
                count.fill_(17)
                expected = direct.forward(direct_args).clone()
                assert torch.count_nonzero(expected) > 0
                args.output.fill_(-13)
                layer.warmup(args)
                assert_bits(
                    layer(args), expected, "outer caller count after construction"
                )
                with torch.cuda.graph(graph):
                    layer(args)
                for live, scale in ((17, 0), (7, 1), (0, 1), (17, 1)):
                    count.fill_(live)
                    alpha.fill_(scale)
                    direct_args.output.fill_(-13)
                    expected = direct.forward(direct_args).clone()
                    args.output.fill_(-13)
                    graph.replay()
                    assert_bits(
                        args.output, expected, "outer caller tensors graph replay"
                    )
            finally:
                graph.reset()
    finally:
        autotune_v2_reload()


@contextmanager
def _distributed(kind):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if world < 2:
        pytest.skip("requires torchrun")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    moe_ep.ensure_moe_ep_cuda_device(bootstrap)
    _require()
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(_configs(kind)[0]).runtime_requirements(bootstrap)
    )
    try:
        yield rank, world
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", KINDS)
def test_blackwell_batch_invariant_outer_autoconfig_expert_parallel(
    kind, monkeypatch, tmp_path
):
    import torch.distributed as dist

    with _distributed(kind) as (rank, world):
        paths = [str(tmp_path) if rank == 0 else None]
        dist.broadcast_object_list(paths)
        _run(kind, rank, world, monkeypatch, type(tmp_path)(paths[0]))


@pytest.mark.gpu_2
def test_blackwell_batch_invariant_outer_autoconfig_expert_parallel_empty_tuning(
    monkeypatch,
):
    import torch
    import torch.distributed as dist
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import MeasurementPolicy
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.core.selection import Collective
    from ._megamoe_bi_helpers import remote_routes

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    configs = _configs("bf16")
    with _distributed("bf16") as (rank, world):
        x, scores, ids, weights = _problem(rank, world, 512, 512)
        ids = remote_routes(ids, rank, world, 4)
        with _layer(configs[0], weights, rank, world, 129, 512) as direct:
            anchor = direct.forward(
                MoEEpTensors(x[:129], ids[:129], scores[:129])
            ).clone()
            receipt = direct.batch_invariant_fingerprint
        n = 0 if rank == 0 else 17
        args = MoEEpTensors(
            x[:n],
            ids[:n],
            scores[:n],
            output=torch.empty_like(x[:n]),
            runtime_max_tokens_per_rank=17,
        )
        measurements = []
        measure = Collective.measure

        def measured(collective, *values, **kwargs):
            result = measure(collective, *values, **kwargs)
            measurements.append(result)
            return result

        monkeypatch.setattr(Collective, "measure", measured)
        with (
            _outer(configs, weights, rank, world) as layer,
            autotune_v2(
                persistent_cache=False,
                cache_scope="context",
                measurement_policy=MeasurementPolicy(
                    execution_mode="eager", cold_l2=False
                ),
            ),
        ):
            workspace = layer.create_workspace(args)
            layer.warmup(args, workspace=workspace)
            assert_bits(
                layer(args, workspace=workspace), anchor[:n], "outer empty-rank tuning"
            )
            assert len(measurements) == len(configs), (
                "empty-rank tuning skipped candidates"
            )
            winner = layer.get_selected_plan(workspace=workspace)
            winners = [None] * world
            dist.all_gather_object(winners, winner)
            assert all(item == winner for item in winners)
            assert workspace._plan.layer.batch_invariant_fingerprint == receipt
