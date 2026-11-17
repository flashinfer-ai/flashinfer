"""Specialized MoE kernel ``bf16_decode_sm103_glm52`` of the unified MoE API.

One served configuration (GLM-5.2 MTP-layer decode: 256 experts, DeepSeekV3
top-8 with one group from fp32 logits and an fp32 bias, hidden 6144,
intermediate 256, SwiGLU, BF16, 1/2/6/10 tokens) is a kernel of the default
``specialized`` MoELayer backend on SM103 GPUs with 150..152 SMs. It reads the
``trtllm_bf16_routed`` weight view; the layer skips it for every other
configuration and device and leaves every other call on ``trtllm_bf16_routed``.

Regression anchors: the registration and both guards (any host, meta tensors),
and on SM103 the kernel through ``MoELayer`` against an fp32 reference of the
BF16 data path and against ``trtllm_bf16_routed``
(``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``) for every token count the kernels
specialize, degenerate inputs, the lazy build and CUDA-graph capture and replay.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashinfer.utils as fi_utils
from flashinfer import RoutingMethodType
from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SpecializedMoEConfig,
    SwiGLU,
    TrtllmBf16Config,
)
from flashinfer.fused_moe.backends import specialized
from flashinfer.fused_moe.backends.specialized.kernels import (
    bf16_decode_sm103_glm52 as kernel_package,
)
from flashinfer.fused_moe.backends.specialized.kernels.bf16_decode_sm103_glm52 import (
    kernel as bf16_decode,
)
from flashinfer.fused_moe.backends.specialized.runner import SpecializedMoERunner
from flashinfer.utils import get_compute_capability

_NAME = "bf16_decode_sm103_glm52"
_KEY = "specialized"
_SIBLING = "trtllm_bf16_routed"
_E = bf16_decode.NUM_EXPERTS
_TOP_K = bf16_decode.TOP_K
_H = bf16_decode.HIDDEN_SIZE
_I = bf16_decode.INTERMEDIATE_SIZE
_QUANT = QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)


def _here() -> bool:
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda")
    lo, hi = bf16_decode.SM_COUNT_RANGE
    return (
        get_compute_capability(dev) == (10, 3)
        and lo <= torch.cuda.get_device_properties(dev).multi_processor_count <= hi
    )


requires_kernel = pytest.mark.skipif(
    not _here(), reason=f"{_NAME} runs on SM103 GPUs with 150..152 SMs"
)


@pytest.fixture
def kernel_state(monkeypatch):
    monkeypatch.delenv(bf16_decode.KILL_SWITCH_ENV, raising=False)
    bf16_decode._reset_for_tests()
    yield bf16_decode
    bf16_decode._reset_for_tests()


def _config(**overrides) -> MoEConfig:
    """The served configuration (default backend list), with field overrides."""
    routing = dict(
        num_experts=_E,
        top_k=_TOP_K,
        method=RoutingMethodType.DeepSeekV3,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
    )
    experts = dict(intermediate_size=_I)
    for key in list(overrides):
        if key in routing or key == "method":
            routing[key] = overrides.pop(key)
        elif key in ("intermediate_size", "local_expert_offset", "local_num_experts"):
            experts[key] = overrides.pop(key)
    overrides.setdefault("quant", _QUANT)
    overrides.setdefault("execution", ExecutionConfig(tune_max_num_tokens=16))
    return MoEConfig(
        routing=RoutingConfig(**routing),
        experts=ExpertConfig(**experts),
        **overrides,
    )


def _sibling_config(**overrides) -> MoEConfig:
    return _config(
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)), **overrides
    )


# ---------------------------------------------------------------------------
# Registration and guards (any host)
# ---------------------------------------------------------------------------


def test_registration():
    kernel = specialized.registered_kernels()[_NAME]
    assert kernel is kernel_package.KERNEL is bf16_decode.Bf16DecodeSm103Glm52Kernel
    assert kernel.weight_view == _SIBLING
    assert kernel.quant_variants == ((QuantFormat.BF16, QuantFormat.BF16),)
    assert kernel.routing_modes == (RoutingInputMode.FromLogits,)
    assert kernel.activation_classes == (SwiGLU,)
    assert kernel.compute_capabilities == (103,)
    assert SpecializedMoEConfig.supported(103)
    assert (QuantFormat.BF16, QuantFormat.BF16) in (
        SpecializedMoERunner.supported_quant_variants
    )
    assert _SIBLING in SpecializedMoERunner.weight_view_aliases
    # the token-count allowlist ships as package data
    bf16_decode._load_allowlist.cache_clear()
    allowlist = bf16_decode._load_allowlist()
    assert allowlist is not None and allowlist["num_tokens"] == {1, 2, 6, 10}


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="trtllm_bf16_routed runs on SM100/SM103",
)
def test_layer_candidates():
    """The default layer of the served configuration builds the specialized
    backend with this kernel only on SM103 with 150..152 SMs."""
    layer = MoELayer(_config())
    keys = [r.backend_key for r in layer.runners]
    assert _SIBLING in keys
    if _here():
        assert _KEY in keys
        (runner,) = [r for r in layer.runners if r.backend_key == _KEY]
        assert [k.name for k in runner.kernels] == [_NAME]
    else:
        assert _KEY not in keys
    # An explicit trtllm_bf16_routed list never includes it.
    explicit = MoELayer(_sibling_config())
    assert [r.backend_key for r in explicit.runners] == [_SIBLING]


_CONFIG_CASES = {
    "served": ({}, None),
    "no_routed_scale": ({"routed_scaling_factor": None}, None),
    "num_experts": ({"num_experts": 128}, "serves only"),
    "top_k": ({"top_k": 6}, "serves only"),
    "intermediate": ({"intermediate_size": 512}, "serves only"),
    "expert_parallel": ({"local_num_experts": 128}, "expert parallelism"),
    "expert_offset": (
        {"local_expert_offset": 128, "local_num_experts": 128},
        "expert parallelism",
    ),
    "renormalize": (
        {
            "method": RoutingMethodType.Renormalize,
            "n_group": None,
            "topk_group": None,
            "routed_scaling_factor": None,
        },
        "DeepSeekV3",
    ),
    "routing_groups": ({"n_group": 8, "topk_group": 4}, "one expert group"),
    "zero_routed_scale": ({"routed_scaling_factor": 0.0}, "routed_scaling_factor"),
    "swiglu_limit": ({"activation": SwiGLU(limit=7.0)}, "default SwiGLU"),
    "relu2": ({"activation": ReLU2()}, "does not support"),
    "no_finalize": ({"finalize": MoEFinalizeConfig(do_finalize=False)}, "do_finalize"),
    "sm100": ({"_arch": (10, 0)}, "exact SM103"),
    "sm_count_148": ({"_sms": 148}, "150..152 SMs"),
}


@pytest.mark.parametrize("case", list(_CONFIG_CASES), ids=list(_CONFIG_CASES))
def test_config_guard(monkeypatch, case):
    """``check_support`` binds the kernel for exactly the served configuration (so
    MoELayer skips the backend for every other one). The device is pinned to an
    SM103 with 152 SMs here so the rules run on any host."""
    overrides, message = _CONFIG_CASES[case]
    overrides = dict(overrides)
    arch = overrides.pop("_arch", (10, 3))
    sms = overrides.pop("_sms", 152)
    monkeypatch.setattr(fi_utils, "get_compute_capability", lambda device: arch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )
    runner = SpecializedMoERunner(_config(**overrides), torch.device("cuda", 0))
    if message is None:
        runner.check_support()
        assert [k.name for k in runner.kernels] == [_NAME]
        return
    with pytest.raises(NotImplementedError, match=message):
        runner.check_support()


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _meta_call(num_tokens=1):
    """A served call (activation pack + weight view) on meta tensors."""
    pack = dict(
        hidden_states_q=_meta((num_tokens, _H), torch.bfloat16),
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=_meta((num_tokens, _E), torch.float32),
        routing_bias=_meta((_E,), torch.float32),
    )
    view = {
        "gemm1_weights": _meta(bf16_decode.GEMM1_SHAPE, torch.bfloat16),
        "gemm2_weights": _meta(bf16_decode.GEMM2_SHAPE, torch.bfloat16),
    }
    return pack, view


# (pack overrides, weight-view overrides, expected reject reason). Meta tensors
# pass every check but the device check, the last one.
_CALL_CASES = {
    "served_1": ({}, {}, "device"),
    "served_2": ({"num_tokens": 2}, {}, "device"),
    "served_6": ({"num_tokens": 6}, {}, "device"),
    "served_10": ({"num_tokens": 10}, {}, "device"),
    "tokens_3": ({"num_tokens": 3}, {}, "num_tokens"),
    "tokens_8": ({"num_tokens": 8}, {}, "num_tokens"),
    "tokens_16": ({"num_tokens": 16}, {}, "num_tokens"),
    "no_bias": ({"routing_bias": None}, {}, "input_layout"),
    "bf16_bias": (
        {"routing_bias": _meta((_E,), torch.bfloat16)},
        {},
        "input_layout",
    ),
    "bf16_logits": (
        {"routing_logits": _meta((1, _E), torch.bfloat16)},
        {},
        "input_layout",
    ),
    "strided_logits": (
        {
            "num_tokens": 2,
            "routing_logits": _meta((2, 2 * _E), torch.float32)[:, :_E],
        },
        {},
        "input_layout",
    ),
    "fp16_hidden": (
        {"hidden_states_q": _meta((1, _H), torch.float16)},
        {},
        "input_layout",
    ),
    "hidden_4096": (
        {"hidden_states_q": _meta((1, 4096), torch.bfloat16)},
        {},
        "input_layout",
    ),
    "swiglu_params_in_view": (
        {},
        {"gemm1_clamp_limit": _meta((_E,), torch.float32)},
        "weight_view_extras",
    ),
    "weights_not_blocked": (
        {},
        {"gemm1_weights": _meta((_E, 2 * _I, _H), torch.bfloat16)},
        "weight_layout",
    ),
    "relu2_weights": (
        {},
        {"gemm1_weights": _meta((_E, _H // 64, _I, 64), torch.bfloat16)},
        "weight_layout",
    ),
}


@pytest.mark.parametrize("case", list(_CALL_CASES), ids=list(_CALL_CASES))
def test_call_guard(kernel_state, case):
    pack_overrides, view_overrides, reason = _CALL_CASES[case]
    pack_overrides = dict(pack_overrides)
    pack, view = _meta_call(pack_overrides.pop("num_tokens", 1))
    pack.update(pack_overrides)
    view.update(view_overrides)
    seen: set = set()
    device = torch.device("cuda", 0)
    assert not kernel_state.admit(MoEActivationPack(**pack), view, device, seen)
    assert kernel_state.stats()["rejects"] == {reason: 1}
    assert not seen


def test_kill_switch_stats_and_errors(kernel_state, monkeypatch):
    spec = kernel_state
    pack, view = _meta_call()
    pack = MoEActivationPack(**pack)
    device = torch.device("cuda", 0)
    # read at call time
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
    assert not spec.admit(pack, view, device, set())
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "0")
    assert not spec.admit(pack, view, device, set())
    st = spec.stats()
    assert st["rejects"] == {"kill_switch": 1, "device": 1}
    assert st["calls"] == 2 and st["dispatch_count"] == 0
    assert specialized.stats()[_NAME]["rejects"] == st["rejects"]
    json.dumps(st)

    # a guard failure rejects the call instead of raising
    def broken():
        raise RuntimeError("allowlist read failed")

    monkeypatch.setattr(spec, "_load_allowlist", broken)
    assert not spec.admit(pack, view, device, set())
    assert spec.stats()["rejects"]["guard_error"] == 1

    # direct use outside the contract raises instead of launching
    bad = MoEActivationPack(
        **{**_meta_call()[0], "routing_logits": _meta((1, _E), torch.bfloat16)}
    )
    with pytest.raises(ValueError, match="input_layout"):
        spec.check_contract(bad, view)


def test_stats_side_effect_free(kernel_state, monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("stats() touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda)
    monkeypatch.setattr(torch.cuda, "current_device", no_cuda)
    kernel_state._load_allowlist.cache_clear()
    st = kernel_state.stats()
    assert kernel_state._load_allowlist.cache_info().currsize == 0
    assert st["allowlist"] is None and st["precompiled"] == []


def test_aot_registration(monkeypatch):
    """flashinfer.aot precompiles the kernels' JIT module for exact SM103 only."""
    from flashinfer.fused_moe.backends.specialized.registry import (
        gen_specialized_moe_aot_modules,
    )
    from flashinfer.jit import bf16_moe_decode_sm103 as jit_module

    spec = SimpleNamespace(name="bf16_moe_decode_sm103")
    monkeypatch.setattr(jit_module, "gen_bf16_moe_decode_sm103_module", lambda: spec)
    assert spec in gen_specialized_moe_aot_modules({"sm103a_exact": True})
    assert spec not in gen_specialized_moe_aot_modules(
        {"sm100a_exact": True, "sm103": True}
    )


# ---------------------------------------------------------------------------
# The served path on SM103 (150..152 SMs)
# ---------------------------------------------------------------------------

_WEIGHTS: dict = {}


def _weights():
    """(canonical w13 [up, gate], w2) and the weight pack registering
    ``TrtllmBf16Config.prepare_weights`` of them under "trtllm_bf16_routed" only."""
    if "pack" not in _WEIGHTS:
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev).manual_seed(0)
        w13 = (torch.randn(_E, 2 * _I, _H, device=dev, generator=gen) * 0.125).to(
            torch.bfloat16
        )
        w2 = (torch.randn(_E, _H, _I, device=dev, generator=gen) * 0.125).to(
            torch.bfloat16
        )
        view = TrtllmBf16Config.prepare_weights(
            w13,
            w2,
            num_local_experts=_E,
            hidden_size=_H,
            intermediate_size=_I,
            activation=SwiGLU(),
            device=dev,
        )
        assert tuple(view["gemm1_weights"].shape) == bf16_decode.GEMM1_SHAPE
        assert tuple(view["gemm2_weights"].shape) == bf16_decode.GEMM2_SHAPE
        pack = MoEWeightPack()
        pack.prepare_for(_SIBLING, view)
        _WEIGHTS["canonical"] = (w13, w2)
        _WEIGHTS["pack"] = pack
    return _WEIGHTS["canonical"], _WEIGHTS["pack"]


def _inputs(num_tokens, kind="serving", seed=1):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    hidden = torch.randn(num_tokens, _H, device=dev, generator=gen).to(torch.bfloat16)
    bias = (torch.randn(_E, device=dev, generator=gen) * 0.01).float()
    logits = torch.randn(num_tokens, _E, device=dev, generator=gen).float()
    if kind == "equal_logits":  # a server's dummy warm-up routing, real activations
        logits.zero_()
        bias.zero_()
    elif kind == "zeros":  # a server's dummy warm-up batch
        logits.zero_()
        hidden.zero_()
    elif kind == "bias_reorders":  # the bias decides the top-8, the weights do not
        bias = (torch.randn(_E, device=dev, generator=gen) * 4.0).float()
    elif kind == "skew":  # every token on the same eight experts
        logits[:, 100:108] += 20.0
    return logits, bias, hidden


def _pack(logits, bias, hidden):
    return MoEActivationPack(
        hidden_states_q=hidden,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


@torch.no_grad()
def _reference(logits, bias, hidden, scale=2.5):
    """fp32 reference of the BF16 data path from the canonical weights, and the
    gap between each token's 8th and 9th routing keys (sigmoid(logit) + bf16(bias),
    lowest expert first on ties; routing weights, SwiGLU and GEMM2 outputs rounded
    to bf16)."""
    (w13, w2), _ = _weights()
    s = torch.sigmoid(logits.float())
    key = s + bias.to(torch.bfloat16).float()
    ordered = torch.sort(-key, dim=-1, stable=True)
    idx = ordered.indices[:, :_TOP_K]
    gap = ordered.values[:, _TOP_K] - ordered.values[:, _TOP_K - 1]
    w = s.gather(-1, idx)
    w = w * (1.0 if scale is None else scale) / (w.sum(-1, keepdim=True) + 1e-20)
    w = w.to(torch.bfloat16).float()
    x = hidden.float()
    out = torch.zeros(x.shape[0], _H, dtype=torch.float32, device=x.device)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for t in range(x.shape[0]):
            for k in range(_TOP_K):
                e = int(idx[t, k])
                g = x[t : t + 1] @ w13[e].float().t()
                a = (F.silu(g[:, _I:]) * g[:, :_I]).to(torch.bfloat16)
                o = (a.float() @ w2[e].float().t()).to(torch.bfloat16).float()
                out[t] += w[t, k] * o[0]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out.to(torch.bfloat16), gap


def _assert_close(actual, expected):
    """The same bf16 data path up to fp32 summation order: almost every element
    within one bf16 ulp."""
    a, e = actual.float().reshape(-1), expected.float().reshape(-1)
    assert bool(torch.isfinite(a).all())
    if not bool(e.any()):
        assert not bool(a.any())
        return
    cos = float(torch.dot(a, e) / (a.norm() * e.norm()))
    rel = float((a - e).norm() / e.norm())
    _, ex = torch.frexp(e.abs().clamp_min(float(e.abs().max()) / 2**10))
    ulp = torch.ldexp(torch.ones_like(e), (ex - 8).to(torch.int32))
    within = float(((a - e).abs() <= ulp).float().mean())
    assert cos >= 0.9999 and rel <= 0.003 and within >= 0.9, (cos, rel, within)


def _runner(layer):
    (runner,) = [r for r in layer.runners if r.backend_key == _KEY]
    return runner


def _run_kernel(runner, pack, weight_pack, output=None):
    """One call on the specialized runner alone (the numerics under test)."""
    assert runner.accepts(pack, weight_pack)
    packed = runner.pack_inputs(pack, weight_pack)
    if output is not None:
        packed[-1] = output
    return runner.forward(packed)


@requires_kernel
@pytest.mark.parametrize("num_tokens", [1, 2, 6, 10])
def test_layer_matches_reference(kernel_state, monkeypatch, num_tokens):
    spec = kernel_state
    _, weight_pack = _weights()
    inputs = _inputs(num_tokens)
    pack = _pack(*inputs)
    layer = MoELayer(_config())

    # The specialized runner on its own (the numerics under test) ...
    out = _run_kernel(_runner(layer), pack, weight_pack).clone()
    assert spec.stats()["dispatch_count"] == 1
    # ... and through the layer, which times it against trtllm_bf16_routed once.
    layered = layer(pack, weight_pack).clone()
    winner = layer.winner_backend
    if num_tokens != 2:  # >= 1.3x faster here: the selection is not noise
        assert winner == _KEY
    # The kill switch leaves the call to trtllm_bf16_routed.
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
    dispatched = spec.stats()["dispatch_count"]
    stock = layer(pack, weight_pack).clone()
    assert layer.winner_backend == _SIBLING
    assert spec.stats()["dispatch_count"] == dispatched
    torch.cuda.synchronize()

    reference, gap = _reference(*inputs)
    _assert_close(out, reference)
    # against trtllm-gen where the top-8 is not a near tie (see
    # test_degenerate_inputs)
    clear = gap.abs() >= 1e-3
    assert int(clear.sum()) >= (num_tokens + 1) // 2
    _assert_close(out[clear], stock[clear])
    if winner == _KEY:
        _assert_close(layered, reference)
    else:
        _assert_close(layered[clear], reference[clear])


@requires_kernel
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16])
def test_matches_trtllm_bf16_routed(kernel_state, monkeypatch, num_tokens):
    """The kernels serve 1..16 tokens; widen the allowlist to cover every
    specialization (one, two, six, ten tokens) and the generic path."""
    monkeypatch.setattr(
        kernel_state,
        "_load_allowlist",
        lambda: {"num_tokens": frozenset({num_tokens})},
    )
    _, weight_pack = _weights()
    inputs = _inputs(num_tokens)
    pack = _pack(*inputs)
    expected = MoELayer(_sibling_config())(pack, weight_pack).clone()
    reference, gap = _reference(*inputs)
    actual = _run_kernel(_runner(MoELayer(_config())), pack, weight_pack)
    assert kernel_state.stats()["dispatched_num_tokens"] == {num_tokens: 1}
    _assert_close(actual, reference)
    clear = gap.abs() >= 1e-3  # see test_degenerate_inputs
    _assert_close(actual[clear], expected[clear])


@requires_kernel
@pytest.mark.parametrize("kind", ["equal_logits", "zeros", "bias_reorders", "skew"])
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("scale", [2.5, None])
def test_degenerate_inputs(kernel_state, monkeypatch, kind, num_tokens, scale):
    monkeypatch.setattr(
        kernel_state,
        "_load_allowlist",
        lambda: {"num_tokens": frozenset({num_tokens})},
    )
    _, weight_pack = _weights()
    inputs = _inputs(num_tokens, kind)
    pack = _pack(*inputs)
    expected = MoELayer(_sibling_config(routed_scaling_factor=scale))(
        pack, weight_pack
    ).clone()
    reference, gap = _reference(*inputs, scale=scale)
    out = torch.full(
        (num_tokens, _H), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    runner = _runner(MoELayer(_config(routed_scaling_factor=scale)))
    actual = _run_kernel(runner, pack, weight_pack, output=out)
    assert kernel_state.stats()["dispatch_count"] == 1
    assert actual.data_ptr() == out.data_ptr()
    # every token against the exact fp32 routing reference
    _assert_close(actual, reference)
    # against trtllm-gen where the top-8 is not a near tie: its routing picked a
    # different 8th expert than the exact fp32 key order for a 16-token call with
    # an 8th/9th key gap of 3.5e-3 (bias ~ N(0, 16)); exact ties (equal logits)
    # resolve to the lowest expert index on both paths
    clear = (gap.abs() >= 1e-2) | (gap == 0)
    if bool(reference.float().any()):  # all-zero activations: zero whatever the routing
        assert int(clear.sum()) >= num_tokens // 2
    _assert_close(actual[clear], expected[clear])


@requires_kernel
def test_lazy_build_and_range(kernel_state):
    spec = kernel_state
    _, weight_pack = _weights()
    layer = MoELayer(_config())
    # outside the allowlist: trtllm_bf16_routed only, nothing built
    layer(_pack(*_inputs(3)), weight_pack)
    assert layer.winner_backend == _SIBLING
    st = spec.stats()
    assert st["rejects"] == {"num_tokens": 1} and st["precompiled"] == []
    # the first admitted eager call loads the kernels for the device
    pack = _pack(*_inputs(1))
    layer(pack, weight_pack)
    st = spec.stats()
    assert torch.cuda.current_device() in st["precompiled"]
    assert 1 in st["dispatched_num_tokens"]
    # a cached winner: one launch per later call
    assert layer.winner_backend == _KEY
    dispatched = st["dispatch_count"]
    layer(pack, weight_pack)
    assert spec.stats()["dispatch_count"] == dispatched + 1


@requires_kernel
@pytest.mark.parametrize("case", ["selected_eagerly", "unbuilt", "unselected"])
def test_cuda_graph(kernel_state, monkeypatch, case):
    """A captured call runs the kernel only when it is loaded and the layer
    already selected its winner for that call eagerly; otherwise it is captured
    on trtllm_bf16_routed (never a build or a timing inside capture)."""
    spec = kernel_state
    _, weight_pack = _weights()
    num_tokens = 6
    logits, bias, hidden = _inputs(num_tokens, seed=2)
    pack = _pack(logits, bias, hidden)
    layer = MoELayer(_config())
    if case == "selected_eagerly":
        layer(pack, weight_pack)
        assert layer.winner_backend == _KEY
    else:
        # warm trtllm_bf16_routed for this call eagerly (its first launch must not
        # happen in capture) with the specialized backend switched off
        monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
        layer(pack, weight_pack)
        monkeypatch.delenv(spec.KILL_SWITCH_ENV)
        if case == "unselected":  # loaded by another call size, then capture 6
            layer(_pack(*_inputs(1)), weight_pack)
            assert spec.is_ready(hidden.device)
    torch.cuda.synchronize()

    captured_before = spec.stats()["capture_dispatch_count"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = layer(pack, weight_pack)
    st = spec.stats()
    specialized_capture = case == "selected_eagerly"
    assert st["capture_dispatch_count"] - captured_before == int(specialized_capture), (
        st
    )
    if not specialized_capture:
        reason = "capture_unready" if case == "unbuilt" else "capture_unselected"
        assert st["rejects"].get(reason, 0) == 1, st

    new = _inputs(num_tokens, seed=3)
    for dst, src in zip((logits, bias, hidden), new, strict=True):
        dst.copy_(src)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    reference, _ = _reference(*new)
    _assert_close(out.clone(), reference)


@requires_kernel
def test_cuda_graph_pool_replay(kernel_state, monkeypatch):
    """Selected eagerly, captured into one graph pool next to a size the kernel
    does not admit, replayed on changed inputs (outputs pre-filled with NaN)
    against trtllm_bf16_routed."""
    spec = kernel_state
    monkeypatch.setattr(
        spec, "_load_allowlist", lambda: {"num_tokens": frozenset({1, 2})}
    )
    _, weight_pack = _weights()
    layer = MoELayer(_config())
    stock_layer = MoELayer(_sibling_config())
    sizes = [2, 1, 3]  # 3 stays on trtllm_bf16_routed, captured between admitted sizes
    static = {t: _inputs(t, seed=10 + t) for t in sizes}
    winners = {}
    for t in sizes:  # eager warm-up, as a server does before capture
        layer(_pack(*static[t]), weight_pack)
        winners[t] = layer.winner_backend
    torch.cuda.synchronize()
    assert winners[1] == _KEY and winners[3] == _SIBLING
    # MoELayer times its candidates in CUDA graphs during the eager warm-up;
    # count only the captures below
    captured_before = spec.stats()["capture_dispatch_count"]
    pool = torch.cuda.graph_pool_handle()
    graphs, outs = {}, {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for t in sizes:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                outs[t] = layer(_pack(*static[t]), weight_pack)
            graphs[t] = g
    torch.cuda.synchronize()
    # the captured calls run the eager winners
    captured = spec.stats()["capture_dispatch_count"] - captured_before
    assert captured == sum(winners[t] == _KEY for t in sizes)
    for rnd, kind in enumerate(["serving", "zeros", "skew", "bias_reorders"]):
        for t in sizes:
            for dst, src in zip(
                static[t], _inputs(t, kind, seed=100 + rnd), strict=True
            ):
                dst.copy_(src)
            outs[t].fill_(float("nan"))
            graphs[t].replay()
            torch.cuda.synchronize()
            expected = stock_layer(_pack(*static[t]), weight_pack)
            _assert_close(outs[t], expected)
