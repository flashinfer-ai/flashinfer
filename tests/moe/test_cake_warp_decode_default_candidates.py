"""The Cake warp-decode entry at the end of MoELayer's default backend list.

``MoELayer`` times every eligible candidate on the first call for each token
bucket and keeps a later one only when it is strictly faster. The Cake entry is
last, so it can replace an earlier backend only by measuring faster. These
tests check that the entry is built last on exact SM100 / SM103 and absent when
switched off or unsupported, that packs it cannot serve fall back to the other
backends, and that every selection and a forced Cake launch match the public
TRT-LLM NVFP4 reference on the same physical tensors. CPU tests use the fake
runner from ``test_cake_warp_decode`` to check the default entry's eligibility
limits during CUDA Graph capture and for cached stream workspaces.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.autotuner import autotuner as autotuner_module
from flashinfer.fused_moe import (
    BackendOptions,
    CakeStepFunConfig,
    CakeWarpDecodeConfig,
    CakeWarpDecodeRunner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SwiGLU,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.api import _DEFAULT_BACKEND
from flashinfer.fused_moe import runners as _runners_module
from flashinfer.fused_moe.layer import NoEligibleBackend
from tests.moe.cake_warp_decode_reference import (
    QUANT,
    TARGETS,
    Fixture,
    Geometry,
    assert_close,
    prepare_fixture,
    reference,
)
from tests.moe.test_cake_warp_decode import (
    _activation_pack as _spec_activation_pack,
    _config as _cpu_config,
    _runner as _cpu_runner,
    _weight_pack as _cpu_weight_pack,
)

_SWITCH_ENV = "FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES"


def _cpu_activation_pack(
    *,
    num_tokens: int = 7,
    top_k: int = 4,
    weights_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    """CPU activation pack with real routing weights."""
    topk_weights = torch.rand((num_tokens, top_k), dtype=torch.float32)
    return _spec_activation_pack(
        num_tokens=num_tokens,
        top_k=top_k,
        weights_dtype=weights_dtype,
        topk_weights=topk_weights.to(weights_dtype),
        **kwargs,
    )


_EXPERIMENTAL_ENV = "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS"
# Default SwiGLU, so one prepared dictionary serves both the Cake and the
# TRT-LLM FP4 runner.
_GEOMETRY = Geometry("e60_i1536_k4", 2048, 1536, 60, 4, (1, 8, 32))
_SEED = 20261117
# Above the 32-token Cake limit and within the layer ceiling.
_LONG_TOKENS = 40
_TUNE_MAX_NUM_TOKENS = 64
_TRTLLM_KEY = "trtllm_fp4_routed"


def _cuda_arch() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("MoELayer default-candidate tests require a CUDA device")
    return torch.cuda.get_device_capability(torch.device("cuda"))


def _require_cake_arch() -> None:
    capability = _cuda_arch()
    if capability not in TARGETS:
        pytest.skip(
            "Cake warp decode requires exact SM100 or SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.delenv(_SWITCH_ENV, raising=False)
    monkeypatch.delenv(_EXPERIMENTAL_ENV, raising=False)


@functools.lru_cache(maxsize=1)
def _fixture() -> Fixture:
    return prepare_fixture(_GEOMETRY, _SEED)


def _config(
    backend: BackendOptions | None = None,
    *,
    enable_pdl: bool | None = None,
    intermediate_size: int = _GEOMETRY.intermediate_size,
) -> MoEConfig:
    kwargs = {} if backend is None else {"backend": backend}
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=_GEOMETRY.num_experts,
            top_k=_GEOMETRY.top_k,
            method=RoutingMethodType.TopK,
        ),
        quant=QUANT,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU(),
        execution=ExecutionConfig(
            enable_pdl=enable_pdl, tune_max_num_tokens=_TUNE_MAX_NUM_TOKENS
        ),
        **kwargs,
    )


def _layer(config: MoEConfig) -> MoELayer:
    layer = MoELayer(config, device=torch.device("cuda"))
    layer.tuner.clear_cache()
    layer.tuner.reset_statistics()
    return layer


def _weights(*keys: str) -> MoEWeightPack:
    weights = MoEWeightPack()
    for key in keys:
        weights.prepare_for(key, dict(_fixture().weight_view))
    return weights


def _activations(
    num_tokens: int, *, weights_dtype: torch.dtype = torch.bfloat16
) -> MoEActivationPack:
    fixture = _fixture()
    return MoEActivationPack(
        hidden_states_q=fixture.hidden_states_q[:num_tokens],
        hidden_states_scale=fixture.hidden_states_scale[:num_tokens],
        topk_ids=fixture.topk_ids[:num_tokens],
        topk_weights=fixture.topk_weights[:num_tokens].to(weights_dtype),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


def _long_activations() -> MoEActivationPack:
    """A pack above the Cake token limit, quantized like the fixture."""
    fixture = _fixture()
    torch.manual_seed(_SEED + 1)
    hidden = torch.randn(
        _LONG_TOKENS, _GEOMETRY.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(hidden, quant=QUANT)
    repeat = _LONG_TOKENS - fixture.topk_ids.shape[0]
    return MoEActivationPack(
        hidden_states_q=hidden_q,
        hidden_states_scale=hidden_scale,
        topk_ids=torch.cat([fixture.topk_ids, fixture.topk_ids[:repeat]]).contiguous(),
        topk_weights=torch.cat(
            [fixture.topk_weights, fixture.topk_weights[:repeat]]
        ).contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


def _cake_indices(layer: MoELayer) -> list[int]:
    return [
        index
        for index, runner in enumerate(layer.runners)
        if isinstance(runner, CakeWarpDecodeRunner)
    ]


def test_default_list_ends_with_one_stable_cake_candidate():
    assert _DEFAULT_BACKEND.candidates[-1] == CakeWarpDecodeConfig(
        default_candidate=True
    )
    assert [
        candidate
        for candidate in _DEFAULT_BACKEND
        if isinstance(candidate, CakeWarpDecodeConfig)
    ] == [CakeWarpDecodeConfig(default_candidate=True)]
    assert not any(isinstance(c, CakeStepFunConfig) for c in _DEFAULT_BACKEND)
    assert (
        MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QUANT,
            experts=ExpertConfig(intermediate_size=512),
        ).backend
        == _DEFAULT_BACKEND
    )
    explicit = CakeWarpDecodeConfig(backend="cake")
    assert not explicit.default_candidate
    assert explicit != CakeWarpDecodeConfig(default_candidate=True)
    assert repr(explicit) == "CakeWarpDecodeConfig(backend='cake')"
    assert (
        repr(CakeWarpDecodeConfig(default_candidate=True))
        == "CakeWarpDecodeConfig(backend='cake', default_candidate=True)"
    )


def _cpu_default_runner() -> CakeWarpDecodeRunner:
    config = replace(
        _cpu_config(),
        backend=BackendOptions((CakeWarpDecodeConfig(default_candidate=True),)),
    )
    runner, _ = _cpu_runner(config)
    return runner


def test_explicit_cake_reports_unservable_inputs_as_before():
    explicit, _ = _cpu_runner()
    default = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    long_pack = _cpu_activation_pack(num_tokens=_LONG_TOKENS)
    fp32_pack = _cpu_activation_pack(weights_dtype=torch.float32)

    assert explicit.accepts(long_pack, weights)
    assert explicit.accepts(fp32_pack, weights)
    with pytest.raises(ValueError, match="1 <= num_tokens <= 32"):
        explicit.pack_inputs(long_pack, weights)
    with pytest.raises(TypeError, match="topk_weights"):
        explicit.pack_inputs(fp32_pack, weights)
    assert not default.accepts(long_pack, weights)
    assert not default.accepts(fp32_pack, weights)


def test_default_entry_needs_eager_warmup_for_capture(monkeypatch):
    runner = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    topk_ids = torch.zeros((7, 4), dtype=torch.int32)
    act = _cpu_activation_pack(topk_ids=topk_ids)
    capturing = False
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: capturing)

    assert runner.accepts(act, weights)
    capturing = True
    assert not runner.accepts(act, weights), "no workspace for this token count"

    capturing = False
    runner.pack_inputs(act, weights)
    capturing = True
    assert runner.accepts(act, weights)
    runner.pack_inputs(act, weights)

    # Capture needs only the prepared workspace of this token count: a router
    # output allocated during capture, or one updated in place, is accepted.
    fresh = _cpu_activation_pack(topk_ids=topk_ids.clone())
    assert runner.accepts(fresh, weights)
    runner.pack_inputs(fresh, weights)
    topk_ids[0, 0] = 1
    assert runner.accepts(act, weights)
    runner.pack_inputs(act, weights)
    other_tokens = _cpu_activation_pack(
        num_tokens=8, topk_ids=torch.zeros((8, 4), dtype=torch.int32)
    )
    assert not runner.accepts(other_tokens, weights)


def test_default_entry_never_syncs_on_fresh_topk_ids(monkeypatch):
    runner = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()

    def no_item(self):
        raise AssertionError("the default entry read a tensor value on the host")

    monkeypatch.setattr(torch.Tensor, "item", no_item)
    # Keep every tensor alive, so no tensor identity is reused.
    live = []
    for step in range(129):
        topk_ids = torch.full((7, 4), step % 60, dtype=torch.int32)
        live.append(topk_ids)
        act = _cpu_activation_pack(topk_ids=topk_ids)
        assert runner.accepts(act, weights), f"not eligible at fresh tensor {step}"
        inputs = runner.pack_inputs(act, weights)
        torch.testing.assert_close(inputs[4], topk_ids, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_id", [-1, 60])
def test_cake_entries_pass_out_of_range_expert_ids_to_the_kernels(invalid_id):
    explicit, _ = _cpu_runner()
    default = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    topk_ids = torch.zeros((7, 4), dtype=torch.int32)
    topk_ids[0, 0] = invalid_id
    act = _cpu_activation_pack(topk_ids=topk_ids)

    # The kernels drop the slot on the device, as the TRT-LLM routing kernels
    # do, so neither entry checks or rewrites the routing on the host.
    for runner in (explicit, default):
        assert runner.accepts(act, weights)
        inputs = runner.pack_inputs(act, weights)
        assert inputs[4] is act.topk_ids
        assert inputs[5] is act.topk_weights
    torch.testing.assert_close(act.topk_ids, topk_ids, rtol=0, atol=0)


def test_switch_rejects_only_the_default_runner(monkeypatch):
    explicit, _ = _cpu_runner()
    default = _cpu_default_runner()
    default._check_support()

    monkeypatch.setenv(_SWITCH_ENV, "1")
    with pytest.raises(NotImplementedError, match=_SWITCH_ENV):
        default._check_support()
    explicit._check_support()


@pytest.mark.parametrize("tuning", [False, True])
def test_default_entry_leaves_group_tuning(monkeypatch, tuning):
    explicit, _ = _cpu_runner()
    default = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    act = _cpu_activation_pack(topk_ids=torch.zeros((7, 4), dtype=torch.int32))
    monkeypatch.setattr(autotuner_module.AutoTuner.get(), "is_tuning_mode", tuning)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())

    # Outside tuning a process group checks nothing.
    assert default.accepts(act, weights) is not tuning
    assert explicit.accepts(act, weights)


def test_default_entry_leaves_before_stream_workspace_limit(monkeypatch):
    runner = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    act = _cpu_activation_pack(topk_ids=torch.zeros((7, 4), dtype=torch.int32))
    limit = runner._MAX_STREAM_WORKSPACES
    stream = SimpleNamespace(cuda_stream=0)
    monkeypatch.setattr(runner, "_current_stream", lambda: stream)

    # Each winner selection times the candidates from a new warmup stream.
    for stream_id in range(1, limit + 1):
        stream = SimpleNamespace(cuda_stream=stream_id)
        if not runner.accepts(act, weights):
            break
        runner.pack_inputs(act, weights)
    assert len(runner._workspace_cache) == limit - 1
    # The last accepted selection could add one more stream workspace.
    stream = SimpleNamespace(cuda_stream=limit + 1)
    runner.pack_inputs(act, weights)
    assert len(runner._workspace_cache) == limit


def test_default_entry_keeps_streams_with_a_workspace_at_the_limit(monkeypatch):
    runner = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    act = _cpu_activation_pack(topk_ids=torch.zeros((7, 4), dtype=torch.int32))
    limit = runner._MAX_STREAM_WORKSPACES
    stream = SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(runner, "_current_stream", lambda: stream)

    for stream_id in range(1, limit):
        stream = SimpleNamespace(cuda_stream=stream_id)
        runner.pack_inputs(act, weights)
    assert len(runner._workspace_cache) == limit - 1
    # A new stream would need a new workspace: the entry leaves.
    stream = SimpleNamespace(cuda_stream=limit)
    assert not runner.accepts(act, weights)
    # A stream that already holds this geometry's workspace stays eligible.
    for stream_id in (1, limit - 1):
        stream = SimpleNamespace(cuda_stream=stream_id)
        assert runner.accepts(act, weights)
        runner.pack_inputs(act, weights)
    assert len(runner._workspace_cache) == limit - 1
    # At the limit a winner selection's warmup stream would not fit: even a
    # stream with a workspace leaves.
    stream = SimpleNamespace(cuda_stream=limit)
    runner.pack_inputs(act, weights)
    assert len(runner._workspace_cache) == limit
    for stream_id in (1, limit):
        stream = SimpleNamespace(cuda_stream=stream_id)
        assert not runner.accepts(act, weights)


def test_default_entry_leaves_after_a_failed_module_load(monkeypatch):
    runner = _cpu_default_runner()
    weights, _ = _cpu_weight_pack()
    act = _cpu_activation_pack(topk_ids=torch.zeros((7, 4), dtype=torch.int32))
    runner._module = None
    monkeypatch.setattr(_runners_module, "_DEFAULT_CAKE_LOAD_ERRORS", {})
    loads = []

    def failing_load():
        loads.append(1)
        raise RuntimeError("no compiler")

    monkeypatch.setattr(runner, "_load_module", failing_load)
    assert not runner.accepts(act, weights)
    assert not runner.accepts(act, weights)
    assert len(loads) == 1, "a failed load is not retried"
    assert isinstance(runner._load_error, RuntimeError)
    # Another layer's default entry on the same architecture does not rebuild.
    other = _cpu_default_runner()
    other._module = None
    monkeypatch.setattr(other, "_load_module", failing_load)
    assert not other.accepts(act, weights)
    assert len(loads) == 1


def test_default_layer_falls_back_when_the_cake_load_fails(monkeypatch):
    _require_cake_arch()
    monkeypatch.setattr(_runners_module, "_DEFAULT_CAKE_LOAD_ERRORS", {})
    layer = _layer(_config())
    cake = layer.runners[-1]
    assert isinstance(cake, CakeWarpDecodeRunner) and cake._module is None

    def failing_load():
        raise RuntimeError("no compiler")

    monkeypatch.setattr(cake, "_load_module", failing_load)
    weights = _weights(_TRTLLM_KEY, "cake")
    activations = _activations(8)
    assert len(layer.runners) - 1 not in layer._eligible_runners(activations, weights)
    assert isinstance(cake._load_error, RuntimeError)
    output = layer(activations, weights)
    assert layer.winner_backend == _TRTLLM_KEY
    assert_close(output, reference(_fixture(), 8))


def test_default_layer_builds_cake_runner_last():
    _require_cake_arch()
    layer = _layer(_config())

    cake_indices = _cake_indices(layer)
    assert cake_indices == [len(layer.runners) - 1]
    assert len(layer.runners) >= 2, "no backend precedes the Cake candidate"
    assert layer._tuning_runner_ids[-1] == len(_DEFAULT_BACKEND) - 1
    # enable_pdl=None resolved from the device; the module loads on first use.
    assert layer.runners[-1]._module is None


def test_switch_removes_only_the_default_cake_candidate(monkeypatch):
    _require_cake_arch()
    monkeypatch.setenv(_SWITCH_ENV, "1")

    assert _cake_indices(_layer(_config())) == []
    explicit = _layer(
        _config(BackendOptions((CakeWarpDecodeConfig(),)), enable_pdl=True)
    )
    assert _cake_indices(explicit) == [0]


def test_stable_cake_candidate_needs_no_experimental_opt_in():
    _require_cake_arch()
    # The autouse fixture leaves the experimental opt-in unset.
    layer = _layer(_config())
    assert _cake_indices(layer) == [len(layer.runners) - 1]


def test_default_cake_candidate_keeps_unsupported_configs_out():
    _require_cake_arch()
    assert _cake_indices(_layer(_config(enable_pdl=False))) == []
    # (I=512, E=60, top_k=4) is not a Cake geometry.
    assert _cake_indices(_layer(_config(intermediate_size=512))) == []


def test_explicit_cake_still_requires_enable_pdl_true():
    _require_cake_arch()
    with pytest.raises(NoEligibleBackend, match="enable_pdl=True"):
        MoELayer(
            _config(BackendOptions((CakeWarpDecodeConfig(),)), enable_pdl=None),
            device=torch.device("cuda"),
        )


def test_unsupported_arch_has_no_cake_candidate():
    capability = _cuda_arch()
    if capability in TARGETS:
        pytest.skip("Cake warp decode is supported on this architecture")
    assert not CakeWarpDecodeConfig.supported(capability[0] * 10 + capability[1])
    try:
        layer = MoELayer(_config(), device=torch.device("cuda"))
    except NoEligibleBackend:
        return
    assert _cake_indices(layer) == []


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_untuned_call_without_cake_view_selects_as_before(num_tokens):
    _require_cake_arch()
    layer = _layer(_config())
    weights = _weights(_TRTLLM_KEY)
    activations = _activations(num_tokens)

    eligible = layer._eligible_runners(activations, weights)
    assert eligible == (0,)
    assert layer.runners[0].backend_key == _TRTLLM_KEY
    output = layer(activations, weights)
    assert layer.winner_backend == _TRTLLM_KEY
    assert_close(output, reference(_fixture(), num_tokens))


def test_untuned_call_with_switch_off_selects_as_before(monkeypatch):
    _require_cake_arch()
    monkeypatch.setenv(_SWITCH_ENV, "1")
    layer = _layer(_config())
    weights = _weights(_TRTLLM_KEY, "cake")
    activations = _activations(8)

    assert layer._eligible_runners(activations, weights) == (0,)
    output = layer(activations, weights)
    assert layer.winner_backend == _TRTLLM_KEY
    assert_close(output, reference(_fixture(), 8))


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
def test_inputs_cake_cannot_serve_fall_back(weights_dtype):
    _require_cake_arch()
    layer = _layer(_config())
    cake_index = len(layer.runners) - 1
    weights = _weights(_TRTLLM_KEY, "cake")
    trtllm_only = _layer(_config(BackendOptions((TrtllmFp4Config(),))))

    if weights_dtype == torch.bfloat16:
        activations = _long_activations()
    else:
        # Cake consumes BF16 routing weights only.
        activations = _activations(8, weights_dtype=weights_dtype)
    assert cake_index not in layer._eligible_runners(activations, weights)
    output = layer(activations, weights)
    assert layer.winner_backend == _TRTLLM_KEY
    assert_close(output, trtllm_only(activations, weights))


@pytest.mark.parametrize("tuned", [False, True], ids=["untuned", "autotune"])
@pytest.mark.parametrize("num_tokens", _GEOMETRY.token_counts)
def test_selection_with_cake_matches_reference(num_tokens, tuned):
    _require_cake_arch()
    layer = _layer(_config())
    cake = layer.runners[-1]
    weights = _weights(_TRTLLM_KEY, "cake")
    activations = _activations(num_tokens)
    assert layer._eligible_runners(activations, weights) == (
        0,
        len(layer.runners) - 1,
    )

    cake_calls = 0
    cake_forward = cake.forward

    def counted_forward(*args, **kwargs):
        nonlocal cake_calls
        cake_calls += 1
        return cake_forward(*args, **kwargs)

    cake.forward = counted_forward
    with flashinfer.autotune(tuned):
        first = layer(activations, weights).clone()
    second = layer(activations, weights).clone()
    torch.cuda.synchronize()

    assert cake_calls > 0, "the Cake candidate was not measured"
    assert layer.winner_backend in (_TRTLLM_KEY, "cake")
    expected = reference(_fixture(), num_tokens)
    assert_close(first, expected)
    assert_close(second, expected)


@pytest.mark.parametrize("num_tokens", _GEOMETRY.token_counts)
def test_forced_cake_tactic_matches_reference(num_tokens):
    _require_cake_arch()
    layer = _layer(_config())
    cake = layer.runners[-1]
    weights = _weights(_TRTLLM_KEY, "cake")
    activations = _activations(num_tokens)
    expected = reference(_fixture(), num_tokens)

    inputs = cake.pack_inputs(activations, weights)
    launch_kwargs = cake.launch_kwargs_for(inputs)
    with flashinfer.autotune(True):
        selected, tactic = layer.tuner.choose_one(
            custom_op="moe_cake",
            runners=[cake],
            tuning_config=cake.tuning_config_for(inputs),
            inputs=inputs,
            **launch_kwargs,
        )
    assert selected is cake
    assert tactic == -1
    direct = cake.forward(inputs, tactic=tactic, **launch_kwargs).clone()
    through_layer = layer._run_tactic(
        activations, weights, (layer._tuning_runner_ids[-1], tactic)
    ).clone()
    torch.cuda.synchronize()

    assert_close(direct, expected)
    assert_close(through_layer, expected)


@pytest.mark.parametrize("num_tokens", [8, 32])
def test_default_entry_drops_out_of_range_routes_like_trtllm(num_tokens):
    """SGLang pads CUDA Graph decode batches with topk_ids = -1."""
    _require_cake_arch()
    layer = _layer(_config())
    weights = _weights(_TRTLLM_KEY, "cake")
    activations = _activations(num_tokens)
    topk_ids = activations.topk_ids.clone()
    topk_ids[-2:] = -1
    topk_ids[0, -1] = _GEOMETRY.num_experts
    caller_ids = topk_ids.clone()
    padded = MoEActivationPack(
        hidden_states_q=activations.hidden_states_q,
        hidden_states_scale=activations.hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=activations.topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    trtllm_only = _layer(_config(BackendOptions((TrtllmFp4Config(),))))

    assert len(layer.runners) - 1 in layer._eligible_runners(padded, weights)
    through_cake = layer._run_tactic(
        padded, weights, (layer._tuning_runner_ids[-1], -1)
    ).clone()
    expected = trtllm_only(padded, weights).clone()
    torch.cuda.synchronize()

    assert torch.equal(through_cake[-2:], torch.zeros_like(through_cake[-2:]))
    assert_close(through_cake, expected)
    # The caller's routing tensor is not modified.
    assert torch.equal(topk_ids, caller_ids)
