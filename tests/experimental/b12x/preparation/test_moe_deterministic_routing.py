"""MoE candidate routing agrees with native compilation and reduction contracts."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe import _impl, _preparation
from b12x.moe.fused_moe._tuning import MoeDecodeConfig, MoeDecodeQuery, TUNING
from b12x.preparation import DeviceIdentity


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
CAPACITIES = (1, 2, 4, 16, 64, 128, 256, 512)


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    def reject_cuda():
        pytest.fail("metadata routing validation must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)
    monkeypatch.delenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", raising=False)
    monkeypatch.delenv(_impl._DYNAMIC_WORK_SOURCE_ENV, raising=False)


def _qwen_query(tokens, deterministic, mode="nvfp4"):
    weights = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a4" if mode == "nvfp4" else "auto",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=moe.MoEGeometry(
            num_experts=512, hidden_size=2560, intermediate_size=640
        ),
    )
    native = replace(weights._impl, nvfp4_inline_scales=True)
    query = MoeDecodeQuery(
        quant_mode=mode,
        quant_modes=tuple(sorted(native.quant_modes)),
        source_format=native.source_format,
        activation=native.activation,
        io_dtype=native.io_dtype,
        num_experts=512,
        hidden_size=2560,
        intermediate_size=640,
        top_k=10,
        num_tokens=tokens,
        routed_rows=tokens * 10,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=deterministic,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        w13_layout=native.w13_layout,
        weight_layouts=tuple(sorted(item.value for item in native.weight_layouts)),
        w4a16_weight_layout=native.w4a16_weight_layout,
        w4a16_scale_format=native.w4a16_scale_format,
        w4a16_block_size_m=None,
        fast_math=False,
        numerical_recipe=None,
        controls=_preparation._control_snapshot(),
        shared_input_scales=True,
        nvfp4_inline_scales=True,
    )
    return query, native


@pytest.mark.parametrize("mode", ("nvfp4", "multi", "nvfp4_auto"))
@pytest.mark.parametrize("deterministic", (True, False))
@pytest.mark.parametrize("tokens", CAPACITIES)
def test_qwen_candidates_lower_through_native_route_compiler(tokens, deterministic, mode):
    query, weights = _qwen_query(tokens, deterministic, mode)
    candidates = TUNING.eligible_plan(query, DEVICE).candidates
    dynamic = [config for _, config in candidates if config.backend == "dynamic"]
    assert dynamic
    direct = [config for config in dynamic if config.dynamic_route_mode == "direct"]
    external = [config for config in dynamic if config.route_planner == "triton"]
    assert bool(direct) == (not deterministic and tokens <= 2)
    assert bool(external) == (not deterministic and tokens <= 16)
    for config in dynamic:
        caps = _preparation._lower_caps(query, config, weights, torch.device("cpu"))
        scratch = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
        plan = scratch.launch_plan
        arguments = _preparation._dynamic_program_arguments(plan, caps)
        payloads = _preparation._dynamic_route_plan_payloads(plan, caps)
        assert plan.deterministic_output == deterministic
        assert arguments["direct_routing"] == (config.dynamic_route_mode == "direct")
        assert arguments["external_route_plan"] == (config.route_planner == "triton")
        assert bool(payloads) == (config.route_planner == "triton")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("route", ("direct", "triton"))
@pytest.mark.parametrize("mode", ("nvfp4", "multi", "nvfp4_auto"))
def test_explicit_deterministic_route_overrides_reject_before_planning(route, mode):
    query, _ = _qwen_query(1, True, mode)
    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="triton" if route == "triton" else "internal",
        max_active_clusters=None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped" if route == "triton" else "direct",
    )
    with pytest.raises(ValueError, match="deterministic"):
        TUNING.configure(query, device=DEVICE, override=config)
    assert TUNING.configure(
        replace(query, deterministic_output=False), device=DEVICE, override=config
    ).pinned == config


@pytest.mark.parametrize("requested", (None, False, True))
@pytest.mark.parametrize("environment", ("0", "1"))
def test_native_direct_plan_resolves_environment_before_route_geometry(monkeypatch, requested, environment):
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", environment)
    _, weights = _qwen_query(1, requested)
    config = MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=16, dynamic_route_mode="direct",
    )
    arguments = dict(
        num_tokens=1, num_topk=10, device=torch.device("cuda", 0),
        weight_plan=weights, quant_mode="nvfp4", deterministic_output=requested,
        decode_config=config,
    )
    deterministic = environment == "1" if requested is None else requested
    if deterministic:
        with pytest.raises(RuntimeError, match="direct routing is unsupported"):
            _impl.plan_tp_moe_execution(**arguments)
    else:
        plan = _impl.plan_tp_moe_execution(**arguments)
        assert not plan.deterministic_output
        assert plan.decode_config.dynamic_route_mode == "direct"


@pytest.mark.parametrize("deterministic", (True, False))
def test_dynamic_heuristic_honors_reduction_contract(deterministic):
    query, _ = _qwen_query(1, deterministic)
    assert _impl._heuristic_dynamic_route_mode(query, DEVICE, planned_tile_m=16) == (
        "grouped" if deterministic else "direct"
    )


@pytest.mark.parametrize("tokens", CAPACITIES)
def test_qwen_deterministic_defaults_keep_existing_kernel_routes(tokens):
    query, _ = _qwen_query(tokens, True)
    config = TUNING.configure(query, device=DEVICE).default
    if tokens <= 4:
        assert config.backend == "micro" and config.dynamic_route_mode is None
    else:
        assert config.backend == "dynamic" and config.dynamic_route_mode == "grouped"
