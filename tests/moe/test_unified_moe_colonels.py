from dataclasses import asdict, replace

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    ClampedReLU2,
    ColonelsBf16Config,
    ColonelsFp8BlockConfig,
    ColonelsTactic,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    SwiGLU,
    get_colonels_tactics,
)
from flashinfer.fused_moe.api import (
    GELU,
    BackendOptions,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    SiLU,
    SiTU,
    SwiGLUStep,
)
from flashinfer.fused_moe.backends.colonels.catalog import (
    get_fused_moe_configs,
    kernel_config_permutations,
)
from flashinfer.fused_moe.backends.colonels.prepare import interleave_gate_up


def _arch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if arch not in (90, 103):
        pytest.skip("Colonels requires SM90 or SM103")
    return arch


def _reference(x, w1, w2, ids, weights, activation, *, offset=0, shared=0, factor=1.0):
    result = torch.zeros_like(x, dtype=torch.float32)
    local = w1.shape[0] - shared
    for e in range(w1.shape[0]):
        z = x.float() @ w1[e].float().T
        if isinstance(activation, SwiGLU):
            up, gate = z.chunk(2, dim=-1)
            hidden = up * torch.nn.functional.silu(gate)
        else:
            hidden = torch.relu(z)
            if isinstance(activation, ClampedReLU2):
                hidden = activation.limit * torch.tanh(hidden / activation.limit)
            hidden = hidden.square()
        y = hidden @ w2[e].float().T
        route = (
            ((ids == e + offset) * weights).sum(dim=1) * factor if e < local else 1.0
        )
        result += y * (route[:, None] if isinstance(route, torch.Tensor) else route)
    return result.to(torch.bfloat16)


def _case(
    tactic,
    activation=None,
    *,
    m=9,
    local=5,
    offset=0,
    shared=0,
    i=256,
    tune_stages=True,
):
    _arch()
    activation = SwiGLU() if activation is None else activation
    torch.manual_seed(731)
    h, k = 1024, 2
    global_e = local + offset + (3 if offset else 0)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16) * 0.08
    w1 = (
        torch.randn(
            local + shared,
            i * (2 if activation.is_gated else 1),
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    # Deliberately asymmetric halves detect a gate/up swap.
    if activation.is_gated:
        w1[:, :i] *= 2
    elif isinstance(activation, ClampedReLU2):
        w1 *= 4  # Exercise values where the soft clamp changes the result materially.
    w2 = torch.randn(local + shared, h, i, device="cuda", dtype=torch.bfloat16) * 0.05
    ids = torch.topk(torch.rand(m, global_e, device="cuda"), k, dim=-1).indices.int()
    weights = torch.softmax(torch.randn(m, k, device="cuda"), dim=-1) * 1.7
    tactics = (
        (
            tuple(
                replace(tactic, stages=s)
                for s in range(1, 8 if tactic.block_m == 256 else 6)
            )
        )
        if tune_stages and tactic.variant.startswith("bf16_sm100")
        else (tactic,)
    )
    allowed = set(get_colonels_tactics(_arch()))
    tactics = tuple(t for t in tactics if t in allowed)
    assert tactics
    backend = ColonelsBf16Config(tactics=tactics)
    config = MoEConfig(
        quant=QuantConfig(),
        routing=RoutingConfig(num_experts=global_e, top_k=k, routed_scaling_factor=1.7),
        experts=ExpertConfig(
            i,
            local_num_experts=local,
            local_expert_offset=offset,
            num_fused_shared_experts=shared,
        ),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
    )
    pack = MoEWeightPack()
    pack.prepare_for(
        "colonels_bf16",
        backend.prepare_weights(
            w1,
            w2,
            num_local_experts=local + shared,
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        ),
    )
    act = MoEActivationPack(x, None, ids, weights)
    layer = MoELayer(config)
    reference = _reference(
        x, w1, w2, ids, weights, activation, offset=offset, shared=shared, factor=1.0
    )
    return layer, act, pack, reference, (w1, w2)


def test_weight_order_and_repr():
    w = torch.cat(
        (torch.full((2, 128, 128), 3.0), torch.full((2, 128, 128), 7.0)), dim=1
    )
    prepared = interleave_gate_up(w, 8)
    assert torch.all(prepared[:, :8] == 7)
    assert torch.all(prepared[:, 8:16] == 3)
    import flashinfer.fused_moe.api as api

    for value in (
        ClampedReLU2(0.7),
        ColonelsBf16Config(tactics=(ColonelsTactic(),)),
        ColonelsFp8BlockConfig(),
    ):
        assert eval(repr(value), vars(api)) == value


@pytest.mark.parametrize(
    "variant",
    [
        "bf16_sm100",
        "bf16_sm100_persistent",
        "bf16_sm100_tmem",
        "bf16_sm100_tmem_persistent",
        "bf16_sm100_gmem",
        "bf16_sm100_gmem_persistent",
        "bf16_gemv",
        "bf16",
        "bf16_persistent",
    ],
)
@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2(), ClampedReLU2(0.7)])
def test_bf16_families(variant, activation):
    arch = _arch()
    if variant != "bf16_gemv" and ("sm100" in variant) != (arch == 103):
        pytest.skip("Other GPU architecture")
    tactic = ColonelsTactic(
        variant=variant,
        block_n=0 if "gmem" in variant else 128 if variant == "bf16_gemv" else 64,
        warp_n=8 if arch == 90 and variant != "bf16_gemv" else 4,
        stages=4 if variant == "bf16_gemv" else 1,
    )
    layer, act, pack, reference, _ = _case(tactic, activation)
    actual = layer(act, pack)
    torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.002)
    actual.fill_(float("nan"))
    torch.testing.assert_close(layer(act, pack), reference, rtol=0.05, atol=0.002)


@pytest.mark.parametrize("shared", [0, 1, 2])
def test_expert_parallel_shared_and_graph(shared):
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    layer, act, pack, reference, source = _case(
        ColonelsTactic(), offset=2, shared=shared
    )
    torch.testing.assert_close(layer(act, pack), reference, rtol=0.05, atol=0.002)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer(act, pack)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(act, pack)
    torch.cuda.current_stream().wait_stream(stream)
    for shift in (0, 1, 3):
        act.topk_ids.copy_((act.topk_ids + shift) % layer.config.routing.num_experts)
        graph.replay()
        expected = _reference(
            act.hidden_states_q,
            *source,
            act.topk_ids,
            act.topk_weights,
            SwiGLU(),
            offset=2,
            shared=shared,
            factor=1.0,
        )
        torch.testing.assert_close(output, expected, rtol=0.05, atol=0.002)


@pytest.mark.parametrize("m", [16, 17])
def test_autotuning(m):
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    layer, act, pack, reference, _ = _case(ColonelsTactic(), m=m, i=16)
    layer = MoELayer(
        replace(
            layer.config, backend=BackendOptions(candidates=(ColonelsBf16Config(),))
        )
    )
    runner = layer.runners[0]
    assert runner.tactics == get_colonels_tactics(_arch())
    valid = runner.get_valid_tactics(runner.pack_inputs(act, pack), None)
    if m > 16:
        # Only GEMV supports this narrow shape, so the cutoff leaves no tactics.
        assert not valid
        return
    gemv = ColonelsTactic(variant="bf16_gemv", block_n=16, warp_n=1, z_split=2)
    assert tuple(asdict(gemv).values()) in valid
    with autotune(True, tuning_buckets=(act.num_tokens,)):
        actual = layer(act, pack)
    torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.002)
    assert layer.winner_backend == "colonels_bf16"


def test_catalogue_has_all_families():
    for arch in (90, 103):
        tactics = get_colonels_tactics(arch)
        expected = len(get_fused_moe_configs(arch, "bf16", SwiGLU())) + len(
            kernel_config_permutations("bf16_gemv")
        )
        assert len(tactics) == len(set(tactics)) == expected
        assert len({t.variant for t in tactics}) == (3 if arch == 90 else 7)
    fp8_tactics = get_colonels_tactics(90, fp8=True)
    assert (
        len(fp8_tactics)
        == len(set(fp8_tactics))
        == len(get_fused_moe_configs(90, "fp8", SwiGLU()))
    )
    assert {t.variant for t in fp8_tactics} == {
        "w8a8",
        "w8a8_persistent",
    }


@pytest.mark.parametrize("arch", [90, 103])
@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2(), ClampedReLU2(0.7)])
def test_catalogue_shared_activations(arch, activation):
    configs = get_fused_moe_configs(arch, "bf16", activation)
    assert configs
    assert all(config["gated"] == activation.is_gated for config in configs)


@pytest.mark.parametrize(
    "activation",
    [
        GELU(),
        GeGLU(),
        GeGLUTanh(),
        Identity(),
        ReLU(),
        SiLU(),
        SiTU(),
        SwiGLUStep(),
        SwiGLU(alpha=2.0),
        SwiGLU(beta=1.0),
        SwiGLU(limit=7.0),
    ],
)
def test_unsupported_shared_activations(activation):
    with pytest.raises(NotImplementedError, match="Colonels"):
        get_fused_moe_configs(90, "bf16", activation)
    rows = 128 * (2 if activation.is_gated else 1)
    with pytest.raises(NotImplementedError, match="Colonels"):
        ColonelsBf16Config.prepare_weights(
            torch.empty(1, rows, 128, dtype=torch.bfloat16),
            torch.empty(1, 128, 128, dtype=torch.bfloat16),
            num_local_experts=1,
            hidden_size=128,
            intermediate_size=128,
            activation=activation,
        )


@pytest.mark.parametrize("m", [0, 1, 129, 1100])
def test_token_counts_and_masked_routes(m):
    arch = _arch()
    tactic = (
        ColonelsTactic()
        if arch == 103
        else ColonelsTactic(variant="bf16", warp_n=8, stages=3)
    )
    layer, act, pack, _, source = _case(tactic, m=m)
    if m:
        act.topk_ids[::2, 0] = -1
        act.topk_ids[::3, 1] = 99
    expected = _reference(
        act.hidden_states_q,
        *source,
        act.topk_ids,
        act.topk_weights,
        SwiGLU(),
        factor=1.0,
    )
    torch.testing.assert_close(layer(act, pack), expected, rtol=0.05, atol=0.002)


def test_arbitrary_expert_map_and_multiple_packs():
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    layer, act, pack, _, source = _case(ColonelsTactic())
    mapping = torch.tensor([4, 1, -1, 0, 3], device="cuda", dtype=torch.int32)
    mapped = MoEWeightPack()
    mapped.prepare_for(
        "colonels_bf16",
        ColonelsBf16Config.prepare_weights(
            *source,
            num_local_experts=5,
            hidden_size=1024,
            intermediate_size=256,
            expert_map=mapping,
        ),
    )
    for selected in (mapped, pack, mapped):
        ids = mapping[act.topk_ids.long()] if selected is mapped else act.topk_ids
        expected = _reference(
            act.hidden_states_q, *source, ids, act.topk_weights, SwiGLU(), factor=1.0
        )
        with autotune(True):
            actual = layer(act, selected)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.002)


@pytest.mark.parametrize("persistent", [False, True])
def test_fp8_block(persistent):
    if _arch() != 90:
        pytest.skip("FP8 requires Hopper")
    tactic = ColonelsTactic(
        variant="w8a8_persistent" if persistent else "w8a8", warp_n=8, stages=3
    )
    _, act, _, _, source = _case(ColonelsTactic(variant="bf16", warp_n=8, stages=3))
    backend = ColonelsFp8BlockConfig(tactics=(tactic,))
    weights = backend.prepare_weights(
        *source, num_local_experts=5, hidden_size=1024, intermediate_size=256
    )
    x, scale = backend.prepare_activations(act.hidden_states_q)
    config = MoEConfig(
        RoutingConfig(5, 2, routed_scaling_factor=1.7),
        QuantConfig(QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
        ExpertConfig(256),
        backend=BackendOptions((backend,)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("colonels_fp8_block", weights)
    actual = MoELayer(config)(
        MoEActivationPack(x, scale, act.topk_ids, act.topk_weights), pack
    )
    expected = _reference(
        act.hidden_states_q,
        *source,
        act.topk_ids,
        act.topk_weights,
        SwiGLU(),
        factor=1.0,
    )
    torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.01)


def test_fp8_preparation():
    torch.manual_seed(20)
    w1 = torch.randn(2, 512, 256, dtype=torch.bfloat16)
    w2 = torch.randn(2, 256, 256, dtype=torch.bfloat16)
    from flashinfer.fused_moe.backends.colonels.prepare import quantize_weights

    q1, s1 = quantize_weights(w1)
    q2, s2 = quantize_weights(w2)
    kwargs = dict(num_local_experts=2, hidden_size=256, intermediate_size=256)
    from_bf16 = ColonelsFp8BlockConfig.prepare_weights(w1, w2, **kwargs)
    from_fp8 = ColonelsFp8BlockConfig.prepare_weights(
        q1, q2, w1_scale=s1, w2_scale=s2, **kwargs
    )
    for name in ("w1", "w2", "w1_scale", "w2_scale"):
        torch.testing.assert_close(
            from_bf16[name].float(), from_fp8[name].float(), rtol=0, atol=0
        )
    x = torch.randn(7, 256, dtype=torch.bfloat16)
    q, scale = ColonelsFp8BlockConfig.prepare_activations(x)
    torch.testing.assert_close(
        q.float() * scale.T.repeat_interleave(128, dim=1),
        x.float(),
        rtol=0.065,
        atol=0.01,
    )
    q, scale = ColonelsFp8BlockConfig.prepare_activations(x[:0])
    assert q.shape == (0, 256) and scale.shape == (2, 0)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), 1e-80, 1e80])
def test_invalid_clamp(limit):
    with pytest.raises(ValueError):
        ClampedReLU2(limit)


def test_direct_launch_buffers():
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    from flashinfer.fused_moe.backends.colonels.runtime import launch

    tactic = ColonelsTactic()
    layer, act, pack, reference, _ = _case(tactic)
    layer(act, pack)
    runner = layer.runners[0]
    ids, weights = runner._routes(act.topk_ids, act.topk_weights, runner._empty)
    routing = runner._workspace(*act.hidden_states_q.shape).sort(ids, weights, 5, 64)
    view = pack.get_view("colonels_bf16")
    output = torch.empty_like(act.hidden_states_q)
    assert launch(
        tactic, 0, act.hidden_states_q, view["w1"], view["w2"], routing, weights, output
    )
    torch.testing.assert_close(output, reference, rtol=0.05, atol=0.002)
    with pytest.raises(ValueError, match="alias"):
        launch(
            tactic,
            0,
            act.hidden_states_q,
            view["w1"],
            view["w2"],
            routing,
            weights,
            act.hidden_states_q,
        )
    flags = torch.zeros(100, dtype=torch.int32, device="cuda")
    with pytest.raises(Exception, match="Completion flags"):
        launch(
            tactic,
            0,
            act.hidden_states_q,
            view["w1"],
            view["w2"],
            routing,
            weights,
            output,
            experts_done=flags,
        )


@pytest.mark.parametrize(
    "updates",
    [
        dict(red_store=True),
        dict(variant="bf16_sm100_tmem", stages=2),
        dict(
            variant="bf16_sm100_gmem",
            block_n=0,
            clusters=2,
            weight_eviction=True,
            epilogue_warps=8,
        ),
        dict(
            variant="bf16_sm100_gmem_persistent",
            block_m=256,
            block_n=0,
            clusters=4,
            cluster_fallback=True,
            producer_warps=8,
        ),
    ],
)
def test_kernel_options(updates):
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    tactic = replace(ColonelsTactic(), **updates)
    layer, act, pack, reference, _ = _case(
        tactic, ReLU2(), i=512, tune_stages="stages" not in updates
    )
    torch.testing.assert_close(layer(act, pack), reference, rtol=0.05, atol=0.002)


def test_precomputed_shared_slots_and_unpacked_weights():
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    from flashinfer.fused_moe import RoutingInputMode

    layer, act, pack, _, source = _case(ColonelsTactic(), shared=2)
    global_e = layer.config.routing.num_experts
    shared_ids = torch.arange(global_e, global_e + 2, device="cuda", dtype=torch.int32)
    ids = torch.cat((act.topk_ids, shared_ids.expand(act.num_tokens, -1)), dim=1)
    weights = torch.cat(
        (act.topk_weights, torch.ones_like(ids[:, :2], dtype=torch.float32)), dim=1
    )
    packed = MoEActivationPack(
        act.hidden_states_q,
        None,
        ids,
        weights.bfloat16(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    expected = _reference(
        act.hidden_states_q,
        *source,
        ids[:, :2],
        weights[:, :2].bfloat16().float(),
        SwiGLU(),
        shared=2,
    )
    with autotune(True):
        actual = layer(packed, pack)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.002)


@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2(), ClampedReLU2()])
def test_gemv_narrow_intermediate(activation):
    _arch()
    tactic = ColonelsTactic(variant="bf16_gemv", block_n=16, warp_n=1, z_split=2)
    layer, act, pack, reference, _ = _case(tactic, activation, i=16, m=1)
    torch.testing.assert_close(layer(act, pack), reference, rtol=0.05, atol=0.002)


def test_hopper_completion_flags():
    if _arch() != 90:
        pytest.skip("Completion protocol requires Hopper")
    from flashinfer.fused_moe.backends.colonels.runtime import launch

    tactic = ColonelsTactic(variant="bf16_persistent", block_n=32, warp_n=8, stages=2)
    layer, act, pack, reference, _ = _case(tactic)
    layer(act, pack)
    runner = layer.runners[0]
    routing = runner._workspace(*act.hidden_states_q.shape).sort(
        act.topk_ids, act.topk_weights, 5, tactic.block_m
    )
    view = pack.get_view("colonels_bf16")
    splits = view["w1"].shape[1] // (tactic.block_n * tactic.warp_n)
    flags = torch.zeros(
        act.topk_ids.numel() * splits + 1, dtype=torch.int32, device="cuda"
    )
    output = torch.empty_like(act.hidden_states_q)
    assert launch(
        tactic,
        0,
        act.hidden_states_q,
        view["w1"],
        view["w2"],
        routing,
        act.topk_weights,
        output,
        experts_done=flags,
    )
    torch.testing.assert_close(output, reference, rtol=0.05, atol=0.002)
    assert (flags == 1).all()


@pytest.mark.parametrize("block_m", [8, 24, 48, 80])
def test_non_power_of_two_sort_padding(block_m):
    _arch()
    from flashinfer.fused_moe.backends.colonels.runtime import Workspace

    ids = torch.tensor(
        [[0, 1], [1, -1], [0, 2], [2, 1]], device="cuda", dtype=torch.int32
    )
    weights = torch.ones_like(ids, dtype=torch.float32)
    workspace = Workspace(4, 128, 128, 3, 2, (block_m,), ids.device, 1, False)
    sorted_ids, experts, padded = workspace.sort(ids, weights, 3, block_m)
    padded = padded.item()
    assert padded == 3 * block_m
    seen = []
    for tile in range(padded // block_m):
        expert = experts[tile].item()
        routes = sorted_ids[tile * block_m : (tile + 1) * block_m]
        live = routes[routes < ids.numel()].long()
        assert (ids.flatten()[live] == expert).all()
        seen.extend(live.tolist())
    assert sorted(seen) == [0, 1, 2, 4, 5, 6, 7]


def test_explicit_gemv_larger_batch():
    _arch()
    tactic = ColonelsTactic(variant="bf16_gemv", block_n=128, warp_n=4, z_split=4)
    layer, act, pack, expected, _ = _case(tactic, m=17)
    torch.testing.assert_close(layer(act, pack), expected, rtol=0.05, atol=0.002)


@pytest.mark.parametrize("shared", [0, 2])
def test_more_routes_than_local_experts(shared):
    if _arch() != 103:
        pytest.skip("Blackwell tactic")
    layer, act, pack, expected, _ = _case(
        ColonelsTactic(), local=1, offset=2, shared=shared
    )
    torch.testing.assert_close(layer(act, pack), expected, rtol=0.05, atol=0.002)
