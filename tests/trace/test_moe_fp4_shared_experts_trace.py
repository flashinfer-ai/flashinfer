"""FP4 shared-expert trace dispatch and geometry coverage."""

import json
from pathlib import Path

import pytest

from flashinfer.trace.templates.moe import (
    trtllm_fp4_block_scale_moe_ds_routing_trace as ROUTED,
    trtllm_fp4_block_scale_moe_ds_shared_experts_trace as SHARED,
    trtllm_fp4_block_scale_moe_trace_dispatch as DISPATCH,
)

SHARED_JSON = (
    Path(__file__).parent
    / "fi_trace_out"
    / "moe_fp4_block_scale_ds_shared_experts_s1_e33_topk8_h256_i128_act3_ng8_kg4.json"
)


def _axes(**overrides):
    axes = dict(
        seq_len=8,
        num_experts=32,
        top_k=8,
        hidden_size=1024,
        intermediate_size=512,
        activation_type=3,
        n_group=8,
        topk_group=4,
    )
    axes.update(overrides)
    return axes


def test_fp4_dispatch_selects_shared_sibling():
    assert DISPATCH(routing_method_type=2, num_fused_shared_experts=1) is SHARED
    assert DISPATCH(routing_method_type=2, num_fused_shared_experts=0) is ROUTED
    assert SHARED in DISPATCH.templates


def test_fp4_shared_name_records_s_and_preserves_routed_name():
    routed = ROUTED.definition_name(_axes(num_local_experts=33))
    shared = SHARED.definition_name(
        _axes(num_weight_rows=33, num_fused_shared_experts=1)
    )
    assert routed != shared
    assert "_s1_" in shared
    assert "shared_experts" in shared


def test_fp4_shared_init_rejects_inconsistent_geometry():
    from flashinfer.trace.templates.moe import (
        _moe_fp4_block_scale_ds_shared_experts_init as init,
    )

    with pytest.raises(ValueError, match="inconsistent shared-expert definition"):
        init(
            seq_len=8,
            num_weight_rows=33,
            num_fused_shared_experts=2,
            num_experts=32,
            hidden_size=1024,
            intermediate_size=512,
            n_group=8,
            topk_group=4,
            top_k=8,
        )


def test_fp4_shared_artifact_renders_standalone_dependencies():
    definition = json.loads(SHARED_JSON.read_text())
    init_namespace = {}
    reference_namespace = {}
    exec(definition["init"], init_namespace)  # noqa: S102
    exec(definition["reference"], reference_namespace)  # noqa: S102
    assert "_moe_fp4_block_scale_ds_init" in init_namespace
    assert "_fp4_moe_run_experts" in reference_namespace
