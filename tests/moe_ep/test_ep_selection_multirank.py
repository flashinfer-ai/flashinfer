"""Complete EP search and replay using Mega and a fixed split transport."""

import dataclasses

import pytest
import torch
import torch.distributed as dist

from flashinfer import autotune_v2, moe_ep
from flashinfer.autotuner import AutoTuner
from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.fused_moe.api import (
    BackendOptions,
    CuteDslConfig,
    QuantConfig,
    QuantFormat,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import MoELayer
from flashinfer.moe_ep.core.selection import BACKEND_NAMESPACE, NAMESPACE, Collective
from tests.moe_ep.test_fused_moe_weights import _nvfp4_moe_config
from tests.moe_ep.sm107_test_utils import (
    canonical_reference,
    quantize,
    weight_pack,
    assert_reference,
)
from tests.moe_ep.test_sm107_auto_multirank import rubin_bootstrap  # noqa: F401
from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem


@pytest.fixture(autouse=True)
def quiet_native_progress(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import autotune

    monkeypatch.setattr(autotune, "print", lambda *args, **kwargs: None, raising=False)


def inputs_and_config(bootstrap, *, forced=None, capacity=64):
    rank, world = bootstrap.rank, bootstrap.world_size
    w13, w2, x, ids, scores = problem._global_problem(world)
    local = problem.NUM_EXPERTS // world
    weights = moe_ep.MoEWeightPack(
        w13[rank * local : (rank + 1) * local], w2[rank * local : (rank + 1) * local]
    )
    cfg = dataclasses.replace(
        _nvfp4_moe_config(
            num_experts=problem.NUM_EXPERTS,
            local_num_experts=local,
            offset=rank * local,
            intermediate=problem.INTERMEDIATE,
        ),
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
    )
    candidates = (
        moe_ep.MegaConfig(
            problem._megakernel_config(
                "mxfp4_mxfp8",
                knobs="auto",
                kernel_variant="auto",
                apply_routing_weights_before_fc2=False,
            )
        ),
        moe_ep.SplitConfig(
            comm=moe_ep.NVLinkOneSidedConfig(sync_protocol="fence"),
            kernel=moe_ep.FusedMoeKernelConfig(cfg),
        ),
    )
    fleet = moe_ep.FleetParams(
        problem.NUM_EXPERTS, capacity, problem.HIDDEN, layout=moe_ep.EpLayout.RANK_MAJOR
    )
    live = 16 - rank
    tensors = moe_ep.MoEEpTensors(
        x[rank, :live].clone(),
        ids[rank, :live].clone(),
        scores[rank, :live].clone(),
        runtime_max_tokens_per_rank=16,
    )
    return (
        fleet,
        weights,
        moe_ep.AutoConfig(candidates, candidate=forced),
        tensors,
        weight_pack(
            torch.cat(
                (w13[:, problem.INTERMEDIATE :], w13[:, : problem.INTERMEDIATE]), dim=1
            ),
            w2,
            "mxfp4_mxfp8",
        )[0],
    )


def reference(t, weights):
    xq, xsf = quantize(t.hidden_states, "mxfp4_mxfp8")
    return canonical_reference(
        xq, xsf, t.topk_ids, t.topk_weights, weights, "mxfp4_mxfp8", early=False
    )


def assert_output(output, t, weights):
    error = None
    try:
        assert_reference(output, reference(t, weights), "mxfp4_mxfp8")
    except AssertionError as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), errors


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "forced,policy",
    [
        (None, None),
        (0, None),
        (1, None),
        (None, "graph"),
        (None, "capture_failure"),
        (None, "cupti"),
        (1, "nccl"),
        (None, "transports"),
    ],
)
def test_search_replay_and_captured_lifetime(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
    forced,
    policy,
):
    bootstrap = rubin_bootstrap
    if policy == "nccl" and not all(
        Collective(bootstrap).gather(moe_ep.have_nccl_ep())
    ):
        pytest.skip("requires NCCL-EP on every rank")
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    if policy in ("nccl", "transports"):
        monkeypatch.setattr(problem, "HIDDEN", 4096)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setattr(
        MoELayer, "_select_winner", lambda *a, **k: pytest.fail("compute-only tuning")
    )
    fleet, weights, config, t, global_weights = inputs_and_config(
        bootstrap, forced=forced
    )
    if policy in ("nccl", "transports"):
        split = dataclasses.replace(config.candidates[1], comm=moe_ep.NcclEpConfig())
        candidates = (
            (config.candidates[0], split)
            if policy == "nccl"
            else (*config.candidates, split)
        )
        config = dataclasses.replace(config, candidates=candidates)
    if policy == "capture_failure":
        from flashinfer.moe_ep.modes.selection import SplitPlan

        forward = SplitPlan.forward

        def reject_capture(plan, tensors):
            output = forward(plan, tensors)
            # Reject on one rank after recording the complete collective call.
            if torch.cuda.is_current_stream_capturing() and bootstrap.rank == 0:
                raise RuntimeError("split capture rejected")
            return output

        monkeypatch.setattr(SplitPlan, "forward", reject_capture)
    options = dict(
        cache_root=str(tmp_path / "managed"),
        measurement_policy=(
            MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True)
            if policy in ("graph", "capture_failure")
            else MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti")
            if policy == "cupti"
            else None
        ),
    )
    measure = Collective.measure
    measured = []

    def checked_measure(c, fn, device, settings, **kwargs):
        score = measure(c, fn, device, settings, **kwargs)
        assert_output(fn(), t, global_weights)
        measured.append(score)
        return score

    monkeypatch.setattr(Collective, "measure", checked_measure)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    graphs, layers = [], [layer]
    try:
        with autotune_v2(mode="replay", **options):
            layer.warmup(t)
            initial = layer._default._plan
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                before = layer(t)
            graphs.append(graph)
        with autotune_v2(**options):
            layer.warmup(t)
            assert_output(layer(t), t, global_weights)
        chosen = layer.get_selected_plan()
        if policy == "capture_failure":
            assert chosen["candidate"] == 0
            assert "No split tactic completed" in layer.rejected_candidates[1]
        if policy == "cupti":
            assert not tuner._cupti_disabled
        if forced is not None:
            assert chosen["candidate"] == forced
        gathered = [None] * bootstrap.world_size
        dist.all_gather_object(gathered, chosen)
        assert all(v == chosen for v in gathered)
        assert initial in layer._default._retained
        graph.replay()
        torch.cuda.synchronize()
        assert_output(before, t, global_weights)
        eager = layer(t).clone()
        # Once settled, eager execution must not enter selector collectives.
        with monkeypatch.context() as patch:
            patch.setattr(
                layer._collective,
                "gather",
                lambda *a: pytest.fail("steady selector collective"),
            )
            torch.testing.assert_close(layer(t), eager)
        t.topk_ids.copy_((t.topk_ids + 1) % problem.NUM_EXPERTS)
        t.hidden_states.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        assert_output(before, t, global_weights)
        assert_output(layer(t), t, global_weights)
        measured.clear()
        with autotune_v2(mode="replay", **options):
            replay = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
            layers.append(replay)
            replay.warmup(t)
            assert replay.get_selected_plan() == chosen
            torch.testing.assert_close(replay(t), layer(t))
        assert not measured
        if bootstrap.rank == 0:
            assert tuner.get_protocol_records(NAMESPACE)
            if forced != 0 and policy != "capture_failure":
                assert tuner.get_protocol_records(BACKEND_NAMESPACE)
        # A capacity change uses independent storage and a different record.
        with autotune_v2(mode="replay", **options):
            handle = layer.create_workspace(t, max_tokens_per_rank=128)
            assert handle.fleet.max_tokens_per_rank == 128
            assert_output(layer(t, workspace=handle), t, global_weights)
            handle.destroy()
        shorter = dataclasses.replace(
            t,
            hidden_states=t.hidden_states[:-1],
            topk_ids=t.topk_ids[:-1],
            topk_weights=t.topk_weights[:-1],
        )
        assert_output(layer(shorter), shorter, global_weights)
        # Rank-local invalidation is reconciled during the next collective warmup.
        with autotune_v2(mode="replay", **options):
            if bootstrap.rank == 0:
                tuner.clear_cache()
            layer.warmup(t)
            assert_output(layer(t), t, global_weights)
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        for current in reversed(layers):
            current.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
def test_capture_rejection_blocks_backend_and_outer_replay(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from flashinfer.autotune_cache import autotune_v2_reload
    from flashinfer.moe_ep.modes.selection import SplitPlan

    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    fleet, weights, config, t, global_weights = inputs_and_config(bootstrap)
    options = dict(
        cache_root=str(tmp_path / f"rejected-rank{bootstrap.rank}"),
        measurement_policy=MeasurementPolicy(execution_mode="cuda_graph"),
    )
    layers, graphs = [], []

    def create(cfg):
        layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=cfg)
        layers.append(layer)
        layer.warmup(t)
        return layer

    try:
        # Seed a backend winner and a forced split outer winner.
        forced = dataclasses.replace(config, candidate=1)
        with autotune_v2(**options):
            seed = create(forced)
            assert seed.get_selected_plan()["candidate"] == 1
        forward = SplitPlan.forward

        def reject_capture(plan, tensors):
            output = forward(plan, tensors)
            if torch.cuda.is_current_stream_capturing() and bootstrap.rank == 0:
                raise RuntimeError("split capture rejected")
            return output

        monkeypatch.setattr(SplitPlan, "forward", reject_capture)
        with autotune_v2(**options):
            layer = create(config)
            assert layer.get_selected_plan()["candidate"] == 0
            assert "No split tactic completed" in layer.rejected_candidates[1]
        dist.barrier()
        autotune_v2_reload()
        measure = Collective.measure
        with autotune_v2(mode="replay", **options):
            monkeypatch.setattr(
                Collective, "measure", lambda *a, **k: pytest.fail("replay measured")
            )
            # The old outer winner must not bypass the recorded rejection.
            with pytest.raises(RuntimeError, match="No EP candidate completed"):
                create(forced)
            # A different outer key must also skip the rejected backend winner.
            split = config.candidates[1]
            moe = split.kernel.moe_config
            split = dataclasses.replace(
                split,
                kernel=dataclasses.replace(
                    split.kernel,
                    moe_config=dataclasses.replace(
                        moe, backend=BackendOptions(candidates=tuple(moe.backend)[::-1])
                    ),
                ),
            )
            reordered = dataclasses.replace(
                config, candidates=(split, config.candidates[0])
            )
            replay = create(reordered)
            assert replay.get_selected_plan()["candidate"] == 1
            assert_output(replay(t), t, global_weights)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                output = replay(t)
            graph.replay()
            torch.cuda.synchronize()
            assert_output(output, t, global_weights)
        # Successful retries must replace each rank's persisted rejection.
        monkeypatch.setattr(SplitPlan, "forward", forward)
        monkeypatch.setattr(Collective, "measure", measure)
        forced_reordered = dataclasses.replace(reordered, candidate=0)
        with autotune_v2(**options):
            recovered = create(forced_reordered)
            assert recovered.get_selected_plan()["candidate"] == 0
        dist.barrier()
        autotune_v2_reload()
        with autotune_v2(mode="replay", **options):
            monkeypatch.setattr(
                Collective, "measure", lambda *a, **k: pytest.fail("replay measured")
            )
            records = AutoTuner.get().get_protocol_records(BACKEND_NAMESPACE)
            assert records and all(
                not r.get("capture_rejected") for r in records.values()
            )
            recovered = create(forced_reordered)
            assert recovered.get_selected_plan()["candidate"] == 0
            assert_output(recovered(t), t, global_weights)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                output = recovered(t)
            graph.replay()
            torch.cuda.synchronize()
            assert_output(output, t, global_weights)
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
def test_ineligible_split_and_rank_agreement(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE as NATIVE_NAMESPACE,
    )

    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    fleet, weights, config, t, global_weights = inputs_and_config(bootstrap)
    split = config.candidates[1]
    moe = split.kernel.moe_config
    small = dataclasses.replace(
        split,
        kernel=dataclasses.replace(
            split.kernel,
            moe_config=dataclasses.replace(
                moe, execution=dataclasses.replace(moe.execution, tune_max_num_tokens=1)
            ),
        ),
    )
    options = dict(
        cache_root=str(tmp_path / "eligibility"), skip_ops={NATIVE_NAMESPACE}
    )
    layer = moe_ep.MoEEpLayer(
        bootstrap,
        fleet,
        weights,
        backend=moe_ep.AutoConfig((config.candidates[0], small)),
    )
    try:
        with autotune_v2(**options):
            layer.warmup(t)
            assert layer.get_selected_plan()["candidate"] == 0
            assert_output(layer(t), t, global_weights)
        with (
            autotune_v2(mode="tune" if bootstrap.rank == 0 else "replay", **options),
            pytest.raises(ValueError, match="disagree on tuning controls"),
        ):
            layer.warmup(t)
    finally:
        layer.destroy()
    forced = moe_ep.MoEEpLayer(
        bootstrap, fleet, weights, backend=moe_ep.AutoConfig((small,), candidate=0)
    )
    try:
        with (
            autotune_v2(**options),
            pytest.raises(RuntimeError, match="No EP candidate"),
        ):
            forced.warmup(t)
    finally:
        forced.destroy()
    different = dataclasses.replace(
        split, comm=dataclasses.replace(split.comm, timeout_sec=300 + bootstrap.rank)
    )
    with pytest.raises(ValueError, match="disagree on candidate configurations"):
        moe_ep.MoEEpLayer(
            bootstrap, fleet, weights, backend=moe_ep.AutoConfig((different,))
        )


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("forced", [0, 1])
def test_empty_rank_output_copy_and_multiple_captures(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
    forced,
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE as NATIVE_NAMESPACE,
    )

    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    fleet, weights, config, t, global_weights = inputs_and_config(
        bootstrap, forced=forced
    )
    if bootstrap.rank == bootstrap.world_size - 1:
        t = dataclasses.replace(
            t,
            hidden_states=t.hidden_states[:0],
            topk_ids=t.topk_ids[:0],
            topk_weights=t.topk_weights[:0],
        )
    t.output = torch.empty(
        t.num_tokens, problem.HIDDEN * 2, device="cuda", dtype=torch.bfloat16
    )[:, ::2]
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    graphs = []
    try:
        with autotune_v2(
            cache_root=str(tmp_path / "first"), skip_ops={NATIVE_NAMESPACE}
        ):
            layer.warmup(t)
            assert_output(layer(t), t, global_weights)
        for epoch in range(3):
            with autotune_v2(
                mode="replay", cache_root=str(tmp_path / f"epoch-{epoch}")
            ):
                layer.warmup(t)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    assert layer(t).data_ptr() == t.output.data_ptr()
                graphs.append(graph)
        assert len(layer._default._retained) == 2
        t.hidden_states.mul_(0.5)
        t.topk_ids.copy_((t.topk_ids + 1) % problem.NUM_EXPERTS)
        for graph in graphs:
            graph.replay()
            torch.cuda.synchronize()
            assert_output(t.output, t, global_weights)
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("available", [False, True])
def test_transport_constraints_are_checked_before_preparation(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
    available,
):
    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    if not available:
        monkeypatch.setattr(problem, "HIDDEN", 4096)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setattr(moe_ep, "have_nccl_ep", lambda: available)
    fleet, weights, config, t, global_weights = inputs_and_config(bootstrap)
    split = dataclasses.replace(config.candidates[1], comm=moe_ep.NcclEpConfig())
    config = dataclasses.replace(config, candidates=(config.candidates[0], split))
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    try:
        reason = "low-latency does not support" if available else "not available"
        assert reason in str(layer.rejected_candidates[1])
        with autotune_v2(mode="replay", cache_root=str(tmp_path / "missing")):
            layer.warmup(t)
            assert layer.get_selected_plan()["candidate"] == 0
            assert_output(layer(t), t, global_weights)
    finally:
        layer.destroy()
    with pytest.raises(ValueError, match="No eligible EP candidate"):
        moe_ep.MoEEpLayer(
            bootstrap,
            fleet,
            weights,
            backend=dataclasses.replace(config, candidate=1),
        )


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "invalid", ["routing_dtype", "input_stride", "zero_runtime_max"]
)
def test_rank_local_invalid_inputs_fail_before_dispatch(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    invalid,
):
    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    fleet, weights, config, t, _ = inputs_and_config(bootstrap)
    if bootstrap.rank == bootstrap.world_size - 1:
        if invalid == "routing_dtype":
            t = dataclasses.replace(t, topk_ids=t.topk_ids.float())
        elif invalid == "input_stride":
            hidden = torch.empty(
                t.num_tokens, problem.HIDDEN * 2, device="cuda", dtype=torch.bfloat16
            )[:, ::2]
            hidden.copy_(t.hidden_states)
            t = dataclasses.replace(t, hidden_states=hidden)
        else:
            t = dataclasses.replace(
                t,
                hidden_states=t.hidden_states[:0],
                topk_ids=t.topk_ids[:0],
                topk_weights=t.topk_weights[:0],
                runtime_max_tokens_per_rank=0,
            )
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    try:
        with (
            autotune_v2(persistent_cache=False),
            pytest.raises(RuntimeError, match="EP candidate preparation failed"),
        ):
            layer.warmup(t)
        assert not layer._workspaces
    finally:
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("stage", ["graph_state", "quantize", "compute"])
def test_rank_local_split_preparation_failure(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    stage,
):
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.backend import (
        FusedMoeSplitKernelBackend,
    )
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    fleet, weights, config, t, _ = inputs_and_config(bootstrap, forced=1)
    targets = {
        "graph_state": [(MoEEpSplitLayer, "create_graph_state")],
        "quantize": [
            (FusedMoeSplitKernelBackend, "pack_dispatch_payload"),
            (FusedMoeSplitKernelBackend, "quantize_for_dispatch"),
        ],
        "compute": [(FusedMoeSplitKernelBackend, "compute")],
    }[stage]

    def inject_failure(owner, name):
        original = getattr(owner, name)

        def fail_one_rank(*args, **kwargs):
            if stage == "graph_state":
                result = original(*args, **kwargs)
            if bootstrap.rank == bootstrap.world_size - 1:
                raise RuntimeError(f"injected {stage} failure")
            return result if stage == "graph_state" else original(*args, **kwargs)

        monkeypatch.setattr(owner, name, fail_one_rank)

    for owner, name in targets:
        inject_failure(owner, name)
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    try:
        with (
            autotune_v2(persistent_cache=False),
            pytest.raises(RuntimeError, match=f"injected {stage} failure"),
        ):
            layer.warmup(t)
        assert not layer._workspaces
    finally:
        torch.cuda.synchronize()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "transport,runtime_bound",
    [("mega", 16), ("nvlink", 16), ("nccl", 16), ("nccl", None), ("nixl", 16)],
)
def test_asymmetric_live_rows_reuse_bound_and_keep_captured_plans(
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
    transport,
    runtime_bound,
):
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.backend import (
        FusedMoeSplitKernelBackend,
    )

    bootstrap = rubin_bootstrap
    monkeypatch.setattr(problem, "INTERMEDIATE", 384)
    monkeypatch.setattr(problem, "HIDDEN", 4096)
    if transport in ("nccl", "nixl"):
        available = moe_ep.have_nccl_ep if transport == "nccl" else moe_ep.have_nixl_ep
        if not available():
            pytest.skip(f"requires {transport}-EP")
    fleet, weights, config, t, global_weights = inputs_and_config(bootstrap)
    candidate = config.candidates[transport != "mega"]
    if transport == "mega":
        candidate = dataclasses.replace(
            candidate,
            megakernel=dataclasses.replace(
                candidate.megakernel, knobs=None, kernel_variant="inference"
            ),
        )
    enumerate_tactics = FusedMoeSplitKernelBackend.tuning_candidates

    def one_tactic_per_backend(backend, context):
        # This regression checks changing rows and storage lifetime. The
        # complete search is covered by test_search_replay_and_captured_lifetime.
        return [
            {**choice, "tactics": choice["tactics"][:1]}
            for choice in enumerate_tactics(backend, context)
        ]

    monkeypatch.setattr(
        FusedMoeSplitKernelBackend, "tuning_candidates", one_tactic_per_backend
    )
    if transport in ("nccl", "nixl"):
        candidate = dataclasses.replace(
            candidate,
            comm=moe_ep.NcclEpConfig() if transport == "nccl" else moe_ep.NvepConfig(),
        )
    if transport == "nixl":
        fleet = dataclasses.replace(fleet, layout=moe_ep.EpLayout.EXPERT_MAJOR)
        cfg = candidate.kernel.moe_config
        candidate = dataclasses.replace(
            candidate,
            kernel=dataclasses.replace(
                candidate.kernel,
                moe_config=dataclasses.replace(
                    cfg,
                    execution=dataclasses.replace(
                        cfg.execution,
                        tune_max_num_tokens=fleet.max_tokens_per_rank
                        * bootstrap.world_size
                        * problem.TOP_K,
                    ),
                ),
            ),
        )
    bound = runtime_bound or fleet.max_tokens_per_rank
    # Start at the bound so a later partial batch must switch from direct
    # execution to staging without preparing a new transport handle.
    repeats = (bound + t.num_tokens - 1) // t.num_tokens
    t = dataclasses.replace(
        t,
        hidden_states=t.hidden_states.repeat(repeats, 1)[:bound].contiguous(),
        topk_ids=t.topk_ids.repeat(repeats, 1)[:bound].contiguous(),
        topk_weights=t.topk_weights.repeat(repeats, 1)[:bound].contiguous(),
        runtime_max_tokens_per_rank=runtime_bound,
    )
    if transport == "nixl":
        t.topk_ids = t.topk_ids.int()
    output = torch.empty(
        bound, problem.HIDDEN * 2, device="cuda", dtype=torch.bfloat16
    )[:, ::2]
    t.output = output
    config = moe_ep.AutoConfig((candidate,))
    options = dict(cache_root=str(tmp_path / "rows"))
    AutoTuner.get().clear_cache()
    layer = moe_ep.MoEEpLayer(bootstrap, fleet, weights, backend=config)
    graphs = []
    try:
        with autotune_v2(**options):
            layer.warmup(t)
        layer.warmup(t)
        original = layer._default._plan
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_output = layer(t)
        graphs.append(graph)
        for rows in (1, 0, bound):
            live = rows if bootstrap.rank == bootstrap.world_size - 1 else bound
            short = dataclasses.replace(
                t,
                hidden_states=t.hidden_states[:live],
                topk_ids=t.topk_ids[:live],
                topk_weights=t.topk_weights[:live],
                output=output[:live],
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    layer._collective,
                    "gather",
                    lambda *a: pytest.fail("serving collective"),
                )
                result = layer(short)
            assert result.shape == short.hidden_states.shape
            assert_output(result, short, global_weights)
            short_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(short_graph):
                captured = layer(short)
            graphs.append(short_graph)
            captured.fill_(float("nan"))
            short_graph.replay()
            torch.cuda.synchronize()
            assert_output(captured, short, global_weights)
        # Only one rank changes its tuning workload. Every rank must reselect,
        # using a distinct record and fresh storage for the captured plan.
        live = 1 if bootstrap.rank == bootstrap.world_size - 1 else bound
        short = dataclasses.replace(
            t,
            hidden_states=t.hidden_states[:live],
            topk_ids=t.topk_ids[:live],
            topk_weights=t.topk_weights[:live],
            output=output[:live],
        )
        with autotune_v2(**options):
            layer.warmup(short)
            assert_output(layer(short), short, global_weights)
        assert original in layer._default._retained
        if bootstrap.rank == 0:
            assert len(AutoTuner.get().get_protocol_records(NAMESPACE)) == 2
        graph.replay()
        torch.cuda.synchronize()
        assert_output(full_output, t, global_weights)
        invalid = dataclasses.replace(
            t,
            topk_ids=(
                t.topk_ids.float()
                if bootstrap.rank == bootstrap.world_size - 1
                else t.topk_ids
            ),
        )
        with pytest.raises(RuntimeError, match="profile changed"):
            layer.warmup(invalid)
        assert_output(layer(t), t, global_weights)
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        layer.destroy()
