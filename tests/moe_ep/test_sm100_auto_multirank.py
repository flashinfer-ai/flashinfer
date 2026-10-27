"""Blackwell automatic tuning, persisted winners and captured workspace lifetime."""

import dataclasses
import json
import os

import pytest
import torch
import torch.distributed as dist

from flashinfer import autotune_v2, moe_ep
from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2_reload
from flashinfer.autotuner import AutoTuner
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg
from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
    autotune as native,
    knob_cache,
    tuning_protocol,
)


def config(kind, variant, **options):
    common = dict(
        intermediate_size=256,
        top_k=3,
        knobs="auto",
        finalize_mode=moe_ep.FinalizeMode.DEFAULT,
        **options,
    )
    if kind == "nvfp4":
        return moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **common
        )
    return moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        kind=kind,
        kernel_variant=variant,
        kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl_next"
        if variant == "inference"
        else "sm100_mxfp8_mxfp8_bf16_cutedsl",
        **common,
    )


@pytest.fixture(scope="module")
def blackwell_bootstrap(request):
    if not torch.cuda.is_available():
        pytest.skip("requires Blackwell")
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    if world == 1:
        os.environ["MEGA_NO_DIST"] = "1"
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap,
        create_mega_kernel(config("nvfp4", "inference")).runtime_requirements(
            bootstrap
        ),
    )
    yield dataclasses.replace(bootstrap, auto_bootstrap=False)
    if not request.session.testsfailed:
        moe_ep.finalize_moe_ep_runtime(runtime)


@pytest.mark.arch_blackwell
@pytest.mark.gpu_2
def test_phase_interleave_fc1_readiness(blackwell_bootstrap, monkeypatch):
    import cutlass.cute as cute
    from cutlass import Boolean, Int32
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.kernel_src.schedulers import (
        fc12_scheduler as scheduler,
    )
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.kernel_src.schedulers.fc12_mapping import (
        create_phase_interleaved_fc12_mapping_state,
    )

    spin_wait = scheduler.spin_wait

    @cute.jit
    def split_ready(pointer, condition):
        # Model a stale peek on one CTA after the watermark is reached.
        spin_wait(pointer, condition)
        return cute.arch.block_idx()[0] != 0

    monkeypatch.setattr(scheduler, "spin_peek", split_ready)

    @dataclasses.dataclass(frozen=True)
    class Probe:
        counter: cute.Pointer | None = None
        minimum_global_fc1_claims = 8

        @cute.jit
        def get_atomic_counter_pointer(self, index):
            return self.counter + index

        @cute.kernel
        def kernel(
            self, counter: cute.Tensor, tokens: cute.Tensor, output: cute.Tensor
        ):
            worker = Probe(counter.iterator)
            mapping = create_phase_interleaved_fc12_mapping_state(
                expert_count=1,
                intermediate_gateup_size=256,
                hidden_size=256,
                mapping_cta_tile_shape_mnk=(128, 256, 128),
                mapping_cluster_shape_mn=(1, 1),
                token_padding_block=128,
                sf_padding_block=128,
                is_swap_ab=True,
                expert_token_sizes=tokens,
                expert_token_prefix_sum=None,
            )
            exhausted = (
                scheduler.PhaseInterleavedFc12Scheduler._wait_for_fc1_prologue_claims(
                    worker,
                    Boolean(False),
                    worker,
                    mapping,
                )
            )
            if cute.arch.thread_idx()[0] == 0:
                output[cute.arch.block_idx()[0]] = Int32(exhausted)

        @cute.jit
        def __call__(
            self, counter: cute.Tensor, tokens: cute.Tensor, output: cute.Tensor
        ):
            self.kernel(counter, tokens, output).launch(
                grid=(2, 1, 1), block=(32, 1, 1), cluster=(2, 1, 1)
            )

    # The runtime stream has one tile, below the static watermark. Neither
    # readiness path may mark FC1 exhausted before a uniform work claim.
    counter = torch.tensor([8], device="cuda", dtype=torch.int32)
    tokens = torch.tensor([1], device="cuda", dtype=torch.int32)
    output = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    args = [from_dlpack(t) for t in (counter, tokens, output)]
    cute.compile(Probe(), *args)(*args)
    torch.cuda.synchronize()
    assert output.tolist() == [0, 0]


@pytest.mark.arch_blackwell
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "kind,variant,policy_name",
    [
        ("nvfp4", "inference", "native"),
        ("mxfp8_e4m3", "inference", "native"),
        ("mxfp8_e5m2", "inference", "native"),
        ("mxfp8_e4m3", "genphase", "native"),
        ("nvfp4", "inference", "eager-cold"),
        ("nvfp4", "inference", "auto-cold"),
        ("mxfp8_e4m3", "genphase", "graph-cold"),
        ("nvfp4", "inference", "cupti"),
    ],
)
def test_auto_persistence_and_capture(
    kind, variant, policy_name, blackwell_bootstrap, monkeypatch, tmp_path
):
    bootstrap = blackwell_bootstrap
    rank, world = bootstrap.rank, bootstrap.world_size
    capacity, live, hidden = 128, 65 - rank, 512
    policy = {
        "native": None,
        "auto-cold": MeasurementPolicy(cold_l2=True),
        "eager-cold": MeasurementPolicy(execution_mode="eager", cold_l2=True),
        "graph-cold": MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True),
        "cupti": MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti"),
    }[policy_name]
    generator = torch.Generator(device="cuda").manual_seed(317 + rank)
    weights = moe_ep.MoEWeightPack(
        (
            torch.randn(4, 512, hidden, device="cuda", generator=generator)
            / hidden**0.5
        ).bfloat16(),
        (
            torch.randn(4, hidden, 256, device="cuda", generator=generator) / 256**0.5
        ).bfloat16(),
    )
    x = torch.randn(live, hidden, device="cuda", generator=generator).bfloat16()
    ids = (
        torch.rand(live, 4 * world, device="cuda", generator=generator)
        .topk(3, dim=-1)
        .indices.int()
    )
    ids[0] = -1
    ids[1, 1] = 4 * world
    scores = torch.rand(live, 3, device="cuda", generator=generator) + 0.5
    scores[(ids < 0) | (ids >= 4 * world)] = float("nan")
    tensors = moe_ep.MoEEpTensors(x, ids, scores)
    cfg = config(kind, variant)
    fleet = moe_ep.FleetParams(
        num_experts=4 * world, max_tokens_per_rank=capacity, token_hidden_size=hidden
    )
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    calls = []
    sweep = native._sweep_sm100_mega_moe

    def counted(*args, **kwargs):
        result = sweep(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(native, "_sweep_sm100_mega_moe", counted)
    layers = []
    graphs = []
    root = str(tmp_path / "managed")

    def context(mode="tune", **kwargs):
        return autotune_v2(
            mode=mode, cache_root=root, measurement_policy=policy, **kwargs
        )

    def make_layer(options=cfg, transformed=None):
        layer = MoEEpMegaLayer(
            bootstrap,
            fleet,
            weights,
            moe_ep.MegaConfig(megakernel=options, transformed_weights=transformed),
        )
        layers.append(layer)
        return layer

    def check(layer, output, workspace=None):
        ws = layer._workspace if workspace is None else workspace
        indices, expected = pkg.sampled_reference(ws, *layer._transformed, live)
        error = torch.tensor(pkg.output_error(output, indices, expected), device="cuda")
        if world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error.item() < 0.02

    try:
        layer = make_layer()
        with context("replay"):
            layer.warmup(tensors)
        assert not calls and layer._workspace._tuning_state.winner is None
        old_workspace = layer._workspace
        pointers = [
            getattr(old_workspace, n).data_ptr() for n in ("x", "x_sf", "topk_idx")
        ]
        old_graph = torch.cuda.CUDAGraph()
        graphs.append(old_graph)
        with torch.cuda.graph(old_graph):
            old_output = layer.forward(tensors)
        with context():
            layer.warmup(tensors)
        assert len(calls) == 1 and calls[0][2] > 0
        winner = layer.get_tuned_knobs()
        assert layer._workspace is not old_workspace
        assert old_workspace in layer._retired_tuning_workspaces
        assert pointers == [
            getattr(old_workspace, n).data_ptr() for n in ("x", "x_sf", "topk_idx")
        ]
        check(layer, layer.forward(tensors))
        old_graph.replay()
        torch.cuda.synchronize()
        check(layer, old_output, old_workspace)
        if rank == 0:
            assert list((tmp_path / "managed").glob("v2/*/entries/*.json"))
        tuner.clear_cache()
        tuner._managed_cache = None
        tuner._managed_stores.clear()

        native_path = tmp_path / "selected.json"
        monkeypatch.setenv(
            "FLASHINFER_MOE_EP_KNOB_CACHE", str(native_path) if rank == 0 else "off"
        )
        cached = make_layer(transformed=layer._transformed)
        with context("replay"):
            cached.warmup(tensors)
        assert cached.get_tuned_knobs() == winner and len(calls) == 1
        if rank == 0:
            record = json.loads(native_path.read_text())["entries"][0]
            assert knob_cache._decode(record, cached._workspace.config) == winner
            assert (
                record in tuner.get_protocol_records(tuning_protocol.NAMESPACE).values()
            )
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        with torch.cuda.graph(graph):
            captured = cached.forward(tensors)
        tensors.hidden_states.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        check(cached, captured)
        previous = cached._workspace
        autotune_v2_reload()
        with context():
            cached.warmup(tensors)
        assert cached._workspace is not previous and len(calls) == 1
        graph.replay()
        torch.cuda.synchronize()
        check(cached, captured, previous)
        fixed = make_layer(dataclasses.replace(cfg, knobs=winner), layer._transformed)
        handle = fixed.create_workspace(capacity, knobs=winner)
        fixed_output = fixed.forward(tensors, workspace=handle)
        check(fixed, fixed_output, handle._backend_workspace)
        with pytest.warns(UserWarning, match="live_tokens is ignored"):
            auto_handle = cached.create_workspace(capacity, live_tokens=live)
        with context():
            cached.warmup(tensors, workspace=auto_handle)
        assert cached.get_tuned_knobs(workspace=auto_handle) == winner
        assert len(calls) == 1
        with context(skip_ops={tuning_protocol.NAMESPACE}):
            cached.warmup(tensors)
            assert cached._workspace._tuning_state.winner is None
        check(cached, cached.forward(tensors))
        for g in graphs:
            g.reset()
    except BaseException:
        # Avoid collective frees while a peer may still be running a failed kernel.
        layers.clear()
        raise
    finally:
        for layer in reversed(layers):
            layer.destroy()
        tuner.clear_cache()
        tuner._managed_cache = None
        tuner._managed_stores.clear()


@pytest.mark.arch_blackwell
@pytest.mark.gpu_4
@pytest.mark.parametrize("kind,early", [("nvfp4", False), ("mxfp8_e4m3", True)])
def test_auto_unfinalized(kind, early, monkeypatch):
    from tests.moe_ep.test_sm100_standard_features import (
        test_standard_features as check_features,
    )

    with autotune_v2(persistent_cache=False):
        check_features(
            kind,
            "situ_unbounded",
            "NONE",
            early,
            "bf16",
            False,
            monkeypatch,
            autotune=True,
        )
