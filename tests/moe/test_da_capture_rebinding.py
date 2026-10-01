"""Numerical coverage for prepared MoE bindings relocated at graph capture."""

import pytest
import torch

from benchmarks.bench_moe_da import _canonical_inputs, _realization
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    TrtllmBf16Config,
    trtllm_bf16_routed_moe,
    trtllm_moe_acquire_da_graph_leases,
    trtllm_moe_da_diagnostics,
    trtllm_moe_release_da_resources,
)
from flashinfer.fused_moe import da_runtime
from flashinfer.fused_moe.da_tuner import RoutingRealizationFactory
from flashinfer.fused_moe.tactic_search import FactorizedSearch
from flashinfer.tllm_enums import RoutingMethodType, WeightLayout
from tests.moe.da_acceptance_utils import compact_shape, require_sm100


@pytest.mark.parametrize("routing_mode", ("packed", "unpacked"))
def test_da_capture_rebinds_live_tensors(monkeypatch, routing_mode):
    """Relocated bindings survive capture, changed-input replay, release, and recapture."""
    require_sm100()
    monkeypatch.setattr(da_runtime, "DA_MOE_REGISTRY", da_runtime.DaMoeRegistry())
    tuner = AutoTuner(warmup=0, repeat=1)
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "uniform,ddist:4")
    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTION_SAMPLES", "1")
    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
    monkeypatch.setenv("FLASHINFER_DA_FACTORIZED_AUTOTUNE", "1")

    # Pin two legal complete bodies; timing noise must not turn this into a singleton test.
    selection_index = 0

    def select_tile(self, space, measure):
        nonlocal selection_index
        assert len(space.tiles) >= 2
        tactic = space.anchor(space.tiles[selection_index % 2])
        selection_index += 1
        return tactic

    monkeypatch.setattr(FactorizedSearch, "search", select_tile)
    monkeypatch.setattr(tuner, "get_factorized_search_result", lambda *args: None)
    shape = compact_shape(num_tokens=32)
    hidden, w1, w2, ids, weights = _canonical_inputs(shape)
    view = TrtllmBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=shape.local_num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        device=hidden.device,
    )
    packed = torch.empty_like(ids)
    output = torch.empty_like(hidden)
    factory = RoutingRealizationFactory()

    def stage(distribution):
        live_ids, live_weights = _realization(factory, shape, distribution)
        hidden.normal_(std=0.1)
        ids.copy_(live_ids)
        weights.copy_(live_weights)
        packed.copy_(
            ids.bitwise_left_shift(16).bitwise_or(
                weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
            )
        )

    def invoke(destination):
        kwargs = dict(
            hidden_states=hidden,
            gemm1_weights=view["gemm1_weights"],
            gemm2_weights=view["gemm2_weights"],
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=shape.local_expert_offset,
            local_num_experts=shape.local_num_experts,
            routing_method_type=RoutingMethodType.Renormalize.value,
            weight_layout=WeightLayout.BlockMajorK,
            output=destination,
            tune_max_num_tokens=shape.num_tokens,
        )
        return trtllm_bf16_routed_moe(
            topk_ids=packed if routing_mode == "packed" else (ids, weights), **kwargs
        )

    stage("uniform")
    with autotune(True, tuning_buckets=(shape.num_tokens,)):
        invoke(output)
    invoke(output)
    (diagnostic,) = trtllm_moe_da_diagnostics()
    assert diagnostic["policy"] == "da_switch", diagnostic
    warm = (hidden, ids, weights, packed, output)
    warm[-1].fill_(float("nan"))
    # Keep every old allocation alive so a recycled address cannot mask missing rebinding.
    retained = [warm]
    selected_bodies = set()
    # Small random weights produce sub-milliscale outputs; a large absolute tolerance hides zeros.
    tolerance = dict(rtol=3e-2, atol=1e-6)
    try:
        for _ in range(2):
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            leases = ()
            try:
                # No eager warmup of these relocated buffers: admission must use the layer/ABI key.
                with torch.cuda.graph(graph):
                    hidden, ids, weights, packed, output = tuple(
                        torch.empty_like(value) for value in warm
                    )
                    invoke(output)
                retained.append((hidden, ids, weights, packed, output))
                assert all(
                    new.data_ptr() != old.data_ptr()
                    for new, old in zip(retained[-1], warm, strict=True)
                )
                leases = trtllm_moe_acquire_da_graph_leases(graph)
                assert len(leases) == 1
                (diagnostic,) = trtllm_moe_da_diagnostics()
                assert diagnostic["topology"]["conditional_node_count"] == 1
                assert diagnostic["prepared_workspace_lane_count"] == 1
                reference = torch.empty_like(output)
                previous = None
                for distribution in ("uniform", "ddist:4", "uniform"):
                    stage(distribution)
                    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
                    invoke(reference)
                    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, reference, **tolerance)
                    assert not torch.allclose(
                        torch.zeros_like(reference), reference, **tolerance
                    )
                    assert torch.isnan(warm[-1]).all()
                    if previous is not None:
                        assert not torch.allclose(previous, reference, **tolerance)
                    previous = output.clone()
                    (diagnostic,) = trtllm_moe_da_diagnostics()
                    selected_bodies.add(diagnostic["selected_body"])
            finally:
                torch.cuda.synchronize()
                graph.reset()
                for lease in leases:
                    lease.release()
        assert selected_bodies == {0, 1}
    finally:
        trtllm_moe_release_da_resources()
