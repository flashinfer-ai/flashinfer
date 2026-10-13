from __future__ import annotations

# TODO(port): this whole suite exercised the retired pre-facade b12x.integration.tp_moe
# host layer (quant-mode normalization, workspace planning, and
# prepare_b12x_fp6_moe_weights), which is not part of the current API.
# The test bodies are kept as reference behind pytest.skip gates; rewrite them
# against b12x.moe.fused_moe plan_weights/prepare_weights/plan/bind/run
# (quant_mode="w6a8_mx", source_format="mxfp6_e2m3") once the w6a8_mx run path
# lands.

import pytest

from b12x._lib.utils import mxfp6_packed_k_bytes


def _tp_moe():
    from b12x.integration import tp_moe  # retired pre-facade API (module TODO)

    return tp_moe


def test_normalize_quant_mode_w6a6() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    assert tp_moe._normalize_quant_mode("W6A6") == "w6a6"


def test_normalize_quant_mode_rejects_unknown() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    with pytest.raises(ValueError, match="unsupported quant_mode"):
        tp_moe._normalize_quant_mode("fp6")


def test_fp6_source_format_default_is_mixed() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    assert tp_moe._normalize_fp6_source_format("mxfp6_default") == "mxfp6_default"
    assert tp_moe._normalize_fp6_source_format("mxfp6_mixed") == "mxfp6_mixed"


def test_mxfp6_cutlass_dtypes_default_mixed() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    import cutlass

    tp_moe = _tp_moe()
    act, wt, sf = tp_moe._mxfp6_cutlass_dtypes("mxfp6_default")
    assert act is cutlass.Float6E3M2FN
    assert wt is cutlass.Float6E2M3FN
    assert sf is cutlass.Float8E8M0FNU


def test_mxfp6_cutlass_dtypes_uniform_e3m2() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    import cutlass

    tp_moe = _tp_moe()
    act, wt, sf = tp_moe._mxfp6_cutlass_dtypes("mxfp6_e3m2")
    assert act is cutlass.Float6E3M2FN
    assert wt is cutlass.Float6E3M2FN
    assert sf is cutlass.Float8E8M0FNU


def test_packed_moe_cols_w6a6() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    k = 128
    assert tp_moe._packed_moe_cols(k, quant_mode="w6a6") == mxfp6_packed_k_bytes(k)
    assert tp_moe._packed_moe_cols(k, quant_mode="nvfp4") == k // 2


def test_static_workspace_packed_input_shape_w6a6() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    plan = tp_moe._plan_core_workspace(
        implementation="static",
        quant_mode="w6a6",
        state_E=4,
        weight_E=4,
        k=128,
        n=128,
        num_topk=8,
        device=__import__("torch").device("cpu"),
        dtype=__import__("torch").bfloat16,
        routed_rows=32,
        max_rows=128,
    )
    packed_spec = next(s for s in plan.tensor_specs if s.name == "packed_input")
    assert packed_spec.shape == (4, 128, mxfp6_packed_k_bytes(128))
    scale_spec = next(s for s in plan.tensor_specs if s.name == "packed_input_scale")
    assert scale_spec.shape[2] == tp_moe._cols_pad_sf(128, quant_mode="w6a6")


def test_prepare_b12x_fp6_moe_weights_no_alphas() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    tp_moe = _tp_moe()
    prepared = tp_moe.prepare_b12x_fp6_moe_weights(
        w1_global_scale=__import__("torch").ones(4),
        w2_global_scale=__import__("torch").ones(4),
        prepare_runtime_alphas=False,
    )
    assert prepared.source_format == "mxfp6_default"
    assert prepared.w1_runtime_alphas is None
