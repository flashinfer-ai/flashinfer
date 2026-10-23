"""Rubin automatic selection, persisted reuse and graph replay over real EP."""

import contextlib
import dataclasses
import os

import pytest
import torch
import torch.distributed as dist

from flashinfer import autotune, autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpTensors,
    PrequantizedMoEWeights,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import sm107_candidates
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    autotune as native,
    knob_cache,
    tuning_protocol,
)
from tests.moe_ep.sm107_test_utils import (
    assert_reference,
    canonical_reference,
    quantize,
    weight_pack,
)
from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem


@pytest.fixture(scope="module")
def rubin_bootstrap(request):
    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if not torch.cuda.is_available() or world < 2:
        pytest.skip("requires torchrun on at least two Rubin GPUs")
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability()[0:2] != (10, 7):
        pytest.skip("requires native SM107")
    cfg = problem._megakernel_config("nvfp4")
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(cfg).runtime_requirements(bootstrap)
    )
    # Layers and candidate workspaces share one runtime throughout the matrix.
    yield dataclasses.replace(bootstrap, auto_bootstrap=False)
    if not request.session.testsfailed:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "kind,variant,ikr,geometry,policy_name",
    [
        (kind, variant, ikr, geometry, "native")
        for kind, variant, ikr in [
            ("nvfp4", "inference", False),
            ("mxfp8_e4m3", "inference", False),
            ("mxfp8_e5m2", "inference", False),
            ("mxfp4_mxfp8", "inference", False),
            ("nvfp4", "genphase", False),
            ("nvfp4", "inference", True),
        ]
        for geometry in [(512, 256, 8, 96, 128), (2048, 512, 16, 192, 256)]
    ]
    + [
        (kind, "genphase", False, (512, 256, 8, 96, 128), "native")
        for kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
    ]
    + [
        ("nvfp4", "inference", False, (512, 256, 8, 96, 128), policy)
        for policy in (
            "auto-cold",
            "eager-hot",
            "eager-cold",
            "graph-hot",
            "graph-cold",
            "graph-cupti",
        )
    ],
)
def test_rubin_auto_persistence_and_capture(
    kind, variant, ikr, geometry, policy_name, monkeypatch, tmp_path, rubin_bootstrap
):
    policy = {
        "native": None,
        "auto-cold": MeasurementPolicy(cold_l2=True),
        "eager-hot": MeasurementPolicy(execution_mode="eager", cold_l2=False),
        "eager-cold": MeasurementPolicy(execution_mode="eager", cold_l2=True),
        "graph-hot": MeasurementPolicy(execution_mode="cuda_graph", cold_l2=False),
        "graph-cold": MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True),
        "graph-cupti": MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti"),
    }[policy_name]
    hidden, intermediate, experts, tokens, capacity = geometry
    for name, value in zip(
        ("HIDDEN", "INTERMEDIATE", "NUM_EXPERTS", "NUM_TOKENS", "MAX_TOKENS"),
        geometry,
        strict=True,
    ):
        monkeypatch.setattr(problem, name, value)
    bootstrap = rubin_bootstrap
    world, rank = bootstrap.world_size, bootstrap.rank
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    cfg = problem._megakernel_config(
        kind, knobs="auto", kernel_variant=variant, in_kernel_fc2_reduce=ikr
    )
    layers = []
    try:
        w13, w2, x, ids, scores = problem._global_problem(world)
        pack, _, _ = weight_pack(w13, w2, kind)
        count = experts // world
        local = slice(rank * count, (rank + 1) * count)
        local_pack = PrequantizedMoEWeights(
            *(
                getattr(pack, name)[local]
                for name in ("w13", "w2", "w13_scale", "w2_scale")
            )
        )
        fleet = FleetParams(
            num_experts=experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        )
        layer = MoEEpMegaLayer(bootstrap, fleet, local_pack, MegaConfig(megakernel=cfg))
        layers.append(layer)
        live = tokens - rank
        tensors = MoEEpTensors(
            x[rank, :live].clone(), ids[rank, :live], scores[rank, :live]
        )
        candidates = sm107_candidates(kind, kernel_variant=variant)
        if ikr:
            candidates = [{**c, "reduce_topk_in_kernel": True} for c in candidates]
        monkeypatch.setattr(
            native, "sm107_candidates", lambda *args, **kwargs: candidates
        )
        from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

        monkeypatch.setattr(
            package, "sm107_candidates", lambda *args, **kwargs: candidates
        )
        calls = []
        sweep = native._sweep_sm107_block_scaled_mega_moe

        def counted_sweep(*args, **kwargs):
            calls.append(True)
            return sweep(*args, **kwargs)

        monkeypatch.setattr(native, "_sweep_sm107_block_scaled_mega_moe", counted_sweep)
        tuner = AutoTuner.get()
        tuner._managed_cache = None
        tuner._managed_stores.clear()
        tuner.clear_cache()
        with autotune(False):
            layer.forward(tensors)
        assert layer._workspace._tuning_state.winner is None and not calls
        workspace = layer._workspace
        buffers = {
            n: getattr(workspace, n).data_ptr()
            for n in ("x", "x_sf", "topk_idx", "topk_weights")
        }
        old_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_graph):
            old_output = layer.forward(tensors)
        cache_root = str(tmp_path / "managed")

        @contextlib.contextmanager
        def tuning(mode="tune"):
            replays = 3 if policy_name in ("graph-hot", "graph-cold") else None
            with (
                autotune_v2(
                    mode=mode, cache_root=cache_root, measurement_policy=policy
                ),
                autotune(cuda_graph_profile_replays=replays),
            ):
                yield

        with tuning():
            layer.warmup(tensors)
        assert len(calls) == 1 and layer._workspace._tuning_state.winner is not None
        assert all(getattr(workspace, n).data_ptr() == p for n, p in buffers.items())
        assert layer._workspace is not workspace
        assert workspace in layer._retired_tuning_workspaces
        assert all(
            getattr(layer._workspace, n).data_ptr() != p for n, p in buffers.items()
        )
        workspace = layer._workspace
        assert layer.get_tuned_knobs() == workspace._tuning_state.winner
        assert workspace.config.reduce_topk_in_kernel == ikr

        def reference():
            xq, xsf = quantize(tensors.hidden_states, kind)
            return canonical_reference(
                xq, xsf, tensors.topk_ids, tensors.topk_weights, pack, kind
            )

        assert_reference(layer.forward(tensors), reference(), kind)
        old_graph.replay()
        torch.cuda.synchronize()
        assert_reference(old_output, reference(), kind)
        if rank == 0:
            assert list((tmp_path / "managed").glob("v2/*/entries/*.json"))
        tuner.clear_cache()
        tuner._managed_cache = None
        tuner._managed_stores.clear()
        with tuning("replay"):
            pass
        dist.barrier()
        native_path = tmp_path / "selected.json"
        monkeypatch.setenv(
            "FLASHINFER_MOE_EP_KNOB_CACHE", str(native_path) if rank == 0 else "off"
        )
        if rank == 0:
            record = next(
                iter(tuner.get_protocol_records(tuning_protocol.NAMESPACE).values())
            )
            alternative = next(
                c for c in candidates if c != layer._workspace._tuning_state.winner
            )
            knob_cache.write_record(
                {**record, "knobs": knob_cache._knobs_to_json(alternative)}
            )
        # Only rank zero has either cache. The FlashInfer record takes precedence.
        cached = MoEEpMegaLayer(
            bootstrap,
            fleet,
            local_pack,
            MegaConfig(megakernel=cfg, transformed_weights=layer._transformed),
        )
        layers.append(cached)
        with autotune(False):
            cached.warmup(tensors)
        assert len(calls) == 1 and cached._workspace._tuning_state.winner is not None
        assert (
            cached._workspace._tuning_state.winner
            == layer._workspace._tuning_state.winner
        )
        if rank == 0:
            records = knob_cache._load_entries(str(native_path))
            assert len(records) == 1
            assert (
                knob_cache._knobs_from_json(records[0]["knobs"])
                == cached._workspace._tuning_state.winner
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = cached.forward(tensors)
        tensors.hidden_states.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(output, reference(), kind)
        old_graph.replay()
        torch.cuda.synchronize()
        assert_reference(old_output, reference(), kind)
        from flashinfer.autotune_cache import autotune_v2_reload

        captured_workspace = cached._workspace
        autotune_v2_reload()
        with tuning():
            cached.warmup(tensors)
        assert cached._workspace is not captured_workspace
        assert captured_workspace in cached._retired_tuning_workspaces
        assert len(calls) == 1
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(output, reference(), kind)
        handle = cached.create_workspace(capacity)
        with tuning():
            cached.warmup(tensors, workspace=handle)
        assert len(calls) == 1
        assert handle._backend_workspace is not cached._workspace
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(output, reference(), kind)
        # A fixed layer can consume the selected knobs without another search.
        fixed = MoEEpMegaLayer(
            bootstrap,
            fleet,
            local_pack,
            MegaConfig(
                megakernel=dataclasses.replace(cfg, knobs=cached.get_tuned_knobs()),
                transformed_weights=layer._transformed,
            ),
        )
        layers.append(fixed)
        fixed_handle = fixed.create_workspace(capacity, knobs=cached.get_tuned_knobs())
        assert_reference(
            fixed.forward(tensors, workspace=fixed_handle), reference(), kind
        )
        with monkeypatch.context() as patch:
            patch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
            with autotune_v2(
                mode="replay",
                cache_root=f"{cache_root}-empty",
                measurement_policy=policy,
            ):
                layer.warmup(tensors)
            assert layer._workspace._tuning_state.winner is None
            with pytest.raises(
                RuntimeError, match="no completed automatic tuning result"
            ):
                layer.get_tuned_knobs()
            for name, value in layer._workspace._tuning_state.initial_knobs.items():
                assert getattr(layer._workspace.config, name) == value
            assert_reference(layer.forward(tensors), reference(), kind)
        with tuning("replay"):
            layer.warmup(tensors)
        assert (
            layer._workspace._tuning_state.winner
            == cached._workspace._tuning_state.winner
        )
        assert len(calls) == 1
        with autotune_v2(
            cache_root=cache_root,
            measurement_policy=policy,
            skip_ops={tuning_protocol.NAMESPACE},
        ):
            layer.warmup(tensors)
            assert layer._workspace._tuning_state.winner is None
            with pytest.raises(
                RuntimeError, match="no completed automatic tuning result"
            ):
                layer.get_tuned_knobs()
            for name, value in layer._workspace._tuning_state.initial_knobs.items():
                assert getattr(layer._workspace.config, name) == value
        assert len(calls) == 1
        old_graph.replay()
        torch.cuda.synchronize()
        assert_reference(old_output, reference(), kind)
        graph.reset()
        old_graph.reset()
    except BaseException:
        # A rank may still be inside a collective kernel. Let torchrun stop peers.
        layers.clear()
        raise
    finally:
        tuner = AutoTuner.get()
        tuner._managed_cache = None
        tuner._managed_stores.clear()
        tuner.clear_cache()
        for layer in reversed(layers):
            layer.destroy()
