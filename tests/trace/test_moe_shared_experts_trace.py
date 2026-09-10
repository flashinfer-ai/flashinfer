"""Regression coverage for shared-expert trace dispatch, naming, standalone
artifacts, and end-to-end emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashinfer.trace.templates.moe import (
    trtllm_fp4_block_scale_moe_ds_routing_trace as FP4_ROUTED,
    trtllm_fp4_block_scale_moe_ds_shared_experts_trace as FP4_SHARED,
    trtllm_fp4_block_scale_moe_trace_dispatch as FP4_DISPATCH,
    trtllm_fp8_block_scale_moe_ds_routing_trace as ROUTED,
    trtllm_fp8_block_scale_moe_ds_shared_experts_trace as SHARED,
    trtllm_fp8_block_scale_moe_trace_dispatch as DISPATCH,
)

FI_TRACE_OUT = Path(__file__).parent / "fi_trace_out"
SHARED_JSON = (
    FI_TRACE_OUT
    / "moe_fp8_block_scale_ds_shared_experts_s1_e33_topk8_ng8_kg4_h7168_i2048.json"
)
FP4_SHARED_JSON = (
    FI_TRACE_OUT
    / "moe_fp4_block_scale_ds_shared_experts_s1_e33_topk8_h256_i128_act3_ng8_kg4.json"
)

_DEEPSEEK_V3 = 2


def _axes(**overrides):
    base = dict(
        seq_len=128,
        num_experts=256,
        top_k=8,
        n_group=8,
        topk_group=4,
        num_local_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Dispatch selection
# ---------------------------------------------------------------------------


def test_dispatch_selects_sibling_when_shared_experts():
    template = DISPATCH(routing_method_type=_DEEPSEEK_V3, num_fused_shared_experts=1)
    assert template is SHARED


def test_dispatch_selects_routed_without_shared_experts():
    template = DISPATCH(routing_method_type=_DEEPSEEK_V3, num_fused_shared_experts=0)
    assert template is ROUTED


def test_dispatch_selects_routed_when_argument_absent():
    assert DISPATCH(routing_method_type=_DEEPSEEK_V3) is ROUTED


def test_only_deepseek_v3_can_select_the_sibling():
    """No other routing kernel emits the appended shared slots."""
    template = DISPATCH(routing_method_type=0, num_fused_shared_experts=1)
    assert template is not SHARED


def test_sibling_template_is_registered_for_consistency_checks():
    """_attach_fi_trace walks .templates; an unlisted template is never validated."""
    assert SHARED in DISPATCH.templates
    assert ROUTED in DISPATCH.templates


# ---------------------------------------------------------------------------
# 2. Definition-name separation
# ---------------------------------------------------------------------------


def test_shared_and_routed_names_differ_at_equal_physical_rows():
    """(E=32, S=0) and (E=31, S=1) both occupy 32 expert-major rows.

    The expert component of the routed name is read from those rows, so without
    a distinct name_prefix and an S component these two would collide on one
    key -- despite differing in effective top-k, and therefore in tactic.
    """
    routed = ROUTED.definition_name(_axes(num_local_experts=32, num_experts=32))
    shared = SHARED.definition_name(
        _axes(num_experts=31, num_weight_rows=32, num_fused_shared_experts=1)
    )
    assert routed != shared


def test_shared_name_encodes_s():
    num_shared = 1
    name = SHARED.definition_name(
        _axes(num_weight_rows=256 + num_shared, num_fused_shared_experts=num_shared)
    )
    assert f"_s{num_shared}_" in name


def test_routed_name_is_unchanged_by_this_feature():
    """The pre-existing definition name must stay byte-identical.

    Solutions and saved benchmarks are keyed by this string; renaming it would
    silently orphan them.
    """
    assert (
        ROUTED.definition_name(_axes())
        == "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e256_h7168_i2048"
    )


# ---------------------------------------------------------------------------
# 3. Committed artifact is runnable standalone
# ---------------------------------------------------------------------------


def _exec_in_fresh_namespace(source: str) -> dict:
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 — exercising the emitted source is the point
    return namespace


@pytest.mark.skipif(
    not SHARED_JSON.exists(), reason="shared-expert trace not generated"
)
def test_committed_shared_expert_init_runs_standalone():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(SHARED_JSON.read_text())["init"]
    namespace = _exec_in_fresh_namespace(source)
    init = namespace["_moe_fp8_block_scale_ds_shared_experts_init"]

    # Calling is the point: a missing dependency defines fine and only fails here.
    out = init(
        seq_len=8,
        num_weight_rows=33,
        num_fused_shared_experts=1,
        num_experts=32,
        hidden_size=1024,
        intermediate_size=512,
        n_group=8,
        topk_group=4,
        top_k=8,
    )
    # Replay must rebuild E+S rows while reporting the routed-only count; a
    # defaulted routed count would silently produce 257 rows here.
    assert out["gemm1_weights"].shape[0] == 33
    assert out["local_num_experts"] == 32
    assert out["num_fused_shared_experts"] == 1


@pytest.mark.skipif(
    not SHARED_JSON.exists(), reason="shared-expert trace not generated"
)
def test_committed_shared_expert_reference_runs_standalone():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(SHARED_JSON.read_text())["reference"]
    namespace = _exec_in_fresh_namespace(source)
    reference = namespace["_trtllm_fp8_block_scale_moe_ds_routing_reference"]

    T, H, I, E, S = 4, 256, 128, 8, 1
    dev = "cuda"
    out = reference(
        torch.randn(T, E, device=dev),
        torch.zeros(E, device=dev, dtype=torch.bfloat16),
        torch.zeros(T, H, device=dev, dtype=torch.float8_e4m3fn),
        torch.ones(H // 128, T, device=dev),
        torch.zeros(E + S, 2 * I, H, device=dev, dtype=torch.float8_e4m3fn),
        torch.ones(E + S, (2 * I) // 128, H // 128, device=dev),
        torch.zeros(E + S, H, I, device=dev, dtype=torch.float8_e4m3fn),
        torch.ones(E + S, H // 128, I // 128, device=dev),
        2,
        2,
        1,
        0,
        2.5,
        num_fused_shared_experts=S,
    )
    assert out.shape == (T, H)


# ---------------------------------------------------------------------------
# 4. End-to-end: fi_trace itself must emit the shared-expert definition
# ---------------------------------------------------------------------------


def _fi_trace_shared(num_shared, *, num_experts=32, hidden=1024, intermediate=512):
    """Exercise dispatch through public ``fi_trace``, not only direct calls."""
    import torch

    import flashinfer

    rows = num_experts + num_shared
    dev = "cuda"
    bs = 128
    return flashinfer.fused_moe.trtllm_fp8_block_scale_moe.fi_trace(
        routing_logits=torch.randn(8, num_experts, dtype=torch.float32, device=dev),
        routing_bias=torch.zeros(num_experts, dtype=torch.bfloat16, device=dev),
        hidden_states=torch.zeros(8, hidden, dtype=torch.float8_e4m3fn, device=dev),
        hidden_states_scale=torch.ones(
            hidden // bs, 8, dtype=torch.float32, device=dev
        ),
        gemm1_weights=torch.zeros(
            rows, 2 * intermediate, hidden, dtype=torch.float8_e4m3fn, device=dev
        ),
        gemm1_weights_scale=torch.ones(
            rows,
            (2 * intermediate) // bs,
            hidden // bs,
            dtype=torch.float32,
            device=dev,
        ),
        gemm2_weights=torch.zeros(
            rows, hidden, intermediate, dtype=torch.float8_e4m3fn, device=dev
        ),
        gemm2_weights_scale=torch.ones(
            rows, hidden // bs, intermediate // bs, dtype=torch.float32, device=dev
        ),
        num_experts=num_experts,
        top_k=8,
        n_group=8,
        topk_group=4,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=2.5,
        routing_method_type=_DEEPSEEK_V3,
        num_fused_shared_experts=num_shared,
    )


def test_fi_trace_emits_shared_expert_definition():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    num_shared = 2
    defn = _fi_trace_shared(num_shared)

    assert defn["name"].startswith("moe_fp8_block_scale_ds_shared_experts")
    assert f"_s{num_shared}_" in defn["name"]

    axes = defn["axes"]
    # The routed count must survive as num_experts; the physical rows carry S.
    assert axes["num_experts"]["value"] == 32
    assert axes["num_fused_shared_experts"]["value"] == num_shared
    assert axes["num_weight_rows"]["value"] == 32 + num_shared
    # The routed template's axis must not reappear and re-introduce the
    # mislabeling this template exists to avoid.
    assert "num_local_experts" not in axes
    assert "num_fused_shared_experts" in defn["inputs"]


def test_fi_trace_emits_routed_definition_when_no_shared_experts():
    """The S=0 path must be untouched by the sibling template."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    defn = _fi_trace_shared(0)

    assert defn["name"].startswith("moe_fp8_block_scale_ds_routing")
    assert "shared_experts" not in defn["name"]
    assert "num_fused_shared_experts" not in defn["axes"]
    assert defn["axes"]["num_local_experts"]["value"] == 32


def test_shared_expert_init_rejects_inconsistent_geometry():
    """Reject replay when ``num_weight_rows - S`` disagrees with routed E."""
    from flashinfer.trace.templates.moe import (
        _moe_fp8_block_scale_ds_shared_experts_init as init,
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


# ---------------------------------------------------------------------------
# 5. FP4 sibling trace
# ---------------------------------------------------------------------------


def _fp4_axes(**overrides):
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
    assert FP4_DISPATCH(routing_method_type=2, num_fused_shared_experts=1) is FP4_SHARED
    assert FP4_DISPATCH(routing_method_type=2, num_fused_shared_experts=0) is FP4_ROUTED
    assert FP4_SHARED in FP4_DISPATCH.templates
    assert "num_fused_shared_experts" not in FP4_ROUTED.inputs


def test_fp4_shared_name_records_s_and_preserves_routed_name():
    routed = FP4_ROUTED.definition_name(_fp4_axes(num_local_experts=33))
    shared = FP4_SHARED.definition_name(
        _fp4_axes(num_weight_rows=33, num_fused_shared_experts=1)
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


@pytest.mark.skipif(
    not FP4_SHARED_JSON.exists(), reason="FP4 shared-expert trace not generated"
)
def test_fp4_committed_shared_expert_init_runs_standalone():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(FP4_SHARED_JSON.read_text())["init"]
    namespace = _exec_in_fresh_namespace(source)
    init = namespace["_moe_fp4_block_scale_ds_shared_experts_init"]
    out = init(
        seq_len=2,
        num_weight_rows=9,
        num_fused_shared_experts=1,
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        n_group=2,
        topk_group=1,
        top_k=2,
    )
    assert out["gemm1_weights"].shape[0] == 9
    assert out["local_num_experts"] == 8
    assert out["num_fused_shared_experts"] == 1


@pytest.mark.skipif(
    not FP4_SHARED_JSON.exists(), reason="FP4 shared-expert trace not generated"
)
def test_fp4_committed_shared_expert_reference_runs_standalone():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(FP4_SHARED_JSON.read_text())["reference"]
    namespace = _exec_in_fresh_namespace(source)
    reference = namespace["_trtllm_fp4_block_scale_moe_ds_routing_reference"]

    T, H, I, E, S = 2, 256, 128, 8, 1
    output = reference(
        torch.randn(T, E, device="cuda"),
        torch.zeros(E, dtype=torch.bfloat16, device="cuda"),
        torch.zeros(T, H // 2, dtype=torch.uint8, device="cuda"),
        torch.ones(T, H // 16, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(E + S, 2 * I, H // 2, dtype=torch.uint8, device="cuda"),
        torch.ones(E + S, 2 * I, H // 16, dtype=torch.float8_e4m3fn, device="cuda"),
        None,
        torch.zeros(E + S, H, I // 2, dtype=torch.uint8, device="cuda"),
        torch.ones(E + S, H, I // 16, dtype=torch.float8_e4m3fn, device="cuda"),
        None,
        2,
        2,
        1,
        0,
        2.5,
        num_fused_shared_experts=S,
    )
    assert output.shape == (T, H)


def test_fi_trace_emits_fp4_shared_expert_definition():
    import torch

    import flashinfer

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    T, H, I, E, S = 2, 256, 128, 8, 1
    rows = E + S
    ones = torch.ones(rows, dtype=torch.float32, device="cuda")
    definition = flashinfer.fused_moe.trtllm_fp4_block_scale_moe.fi_trace(
        routing_logits=torch.randn(T, E, dtype=torch.bfloat16, device="cuda"),
        routing_bias=torch.zeros(E, dtype=torch.bfloat16, device="cuda"),
        hidden_states=torch.zeros(T, H // 2, dtype=torch.uint8, device="cuda"),
        hidden_states_scale=torch.ones(
            T, H // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        gemm1_weights=torch.zeros(
            rows, 2 * I, H // 2, dtype=torch.uint8, device="cuda"
        ),
        gemm1_weights_scale=torch.ones(
            rows, 2 * I, H // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        gemm1_bias=None,
        gemm1_alpha=ones,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.zeros(rows, H, I // 2, dtype=torch.uint8, device="cuda"),
        gemm2_weights_scale=torch.ones(
            rows, H, I // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        gemm2_bias=None,
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        num_experts=E,
        top_k=2,
        n_group=2,
        topk_group=1,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=2.5,
        routing_method_type=2,
        num_fused_shared_experts=S,
    )

    assert definition["name"].startswith("moe_fp4_block_scale_ds_shared_experts")
    assert definition["axes"]["num_experts"]["value"] == E
    assert definition["axes"]["num_weight_rows"]["value"] == rows
    assert definition["axes"]["num_fused_shared_experts"]["value"] == S
    assert "num_local_experts" not in definition["axes"]
