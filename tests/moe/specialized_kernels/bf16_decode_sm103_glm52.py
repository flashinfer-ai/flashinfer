"""Spec of the specialized MoE kernel ``bf16_decode_sm103_glm52``.

One served configuration (GLM-5.2 MTP-layer decode: 256 experts, DeepSeekV3
top-8 with one group from fp32 logits and an fp32 bias, hidden 6144,
intermediate 256, SwiGLU, BF16, 1/2/6/10 tokens) on SM103 GPUs with 150..152
SMs. It reads the ``trtllm_bf16_routed`` weight view; the layer skips it for
every other configuration and device and leaves every other call on
``trtllm_bf16_routed``. The shared tests are in
``tests/moe/test_unified_moe_specialized.py``; the checks below are this
kernel's own: every token count the kernels specialize against
``trtllm_bf16_routed``, degenerate inputs, routing ties broken by expert ID,
concurrent streams and a CUDA-graph pool.
"""

from __future__ import annotations

from types import SimpleNamespace

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
    SwiGLU,
    TrtllmBf16Config,
)
from flashinfer.fused_moe.backends.specialized.kernels import (
    bf16_decode_sm103_glm52 as kernel_package,
)
from flashinfer.fused_moe.backends.specialized.kernels.bf16_decode_sm103_glm52 import (
    kernel as bf16_decode,
)
from flashinfer.utils import get_compute_capability

from . import Check, KernelSpec

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


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _meta_call(num_tokens=None):
    """A served call (activation pack + weight view) on meta tensors."""
    num_tokens = 1 if num_tokens is None else num_tokens
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


def _widen_allowlist(monkeypatch, *num_tokens):
    """The kernels serve 1..16 tokens; admit exactly ``num_tokens``."""
    monkeypatch.setattr(
        bf16_decode,
        "_load_allowlist",
        lambda: {"num_tokens": frozenset(num_tokens)},
    )


# ---------------------------------------------------------------------------
# This kernel's own checks
# ---------------------------------------------------------------------------


def _check_matches_trtllm_bf16_routed(spec, ctx):
    """Every specialization (one, two, six, ten tokens) and the generic path,
    with the allowlist widened to the count under test."""
    num_tokens = ctx.num_tokens
    _widen_allowlist(ctx.monkeypatch, num_tokens)
    _, weight_pack = _weights()
    inputs = _inputs(num_tokens)
    pack = _pack(*inputs)
    expected = MoELayer(_sibling_config())(pack, weight_pack).clone()
    reference, gap = _reference(*inputs)
    actual = _run_kernel(_runner(MoELayer(_config())), pack, weight_pack)
    assert bf16_decode.stats()["dispatched_num_tokens"] == {num_tokens: 1}
    _assert_close(actual, reference)
    clear = gap.abs() >= 1e-3  # see the degenerate-inputs check
    _assert_close(actual[clear], expected[clear])


def _check_degenerate_inputs(spec, ctx):
    num_tokens, kind, scale = ctx.num_tokens, ctx.kind, ctx.scale
    _widen_allowlist(ctx.monkeypatch, num_tokens)
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
    assert bf16_decode.stats()["dispatch_count"] == 1
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


def _check_routing_ties(spec, ctx):
    """Tied routing keys go to the lowest expert IDs, the rule of the reference
    (stable sort) and of trtllm-gen (its packed top-k key ranks equal scores by
    ascending index): equal logits pick experts 0..7; experts 4..11 and 128..131
    tied at the top pick 4..11. The six-token kernel holds two expert ranges per
    lane, so it must not resolve ties by lane (nor its early top-1)."""
    num_tokens, ties = ctx.num_tokens, ctx.ties
    _widen_allowlist(ctx.monkeypatch, num_tokens)
    _, weight_pack = _weights()
    logits, bias, hidden = _inputs(num_tokens, seed=7)  # real activations
    bias.zero_()
    if ties == "all_equal":
        logits.zero_()
        experts = list(range(8))
    else:
        logits.fill_(-1.0)
        logits[:, 4:12] = 1.0
        logits[:, 128:132] = 1.0
        experts = list(range(4, 12))
    key = torch.sigmoid(logits) + bias.to(torch.bfloat16).float()
    picked = torch.sort(-key, dim=-1, stable=True).indices[:, :_TOP_K]
    assert picked.tolist() == [experts] * num_tokens
    pack = _pack(logits, bias, hidden)
    reference, _ = _reference(logits, bias, hidden)
    out = _run_kernel(_runner(MoELayer(_config())), pack, weight_pack).clone()
    assert bf16_decode.stats()["dispatched_num_tokens"] == {num_tokens: 1}
    _assert_close(out, reference)
    # trtllm-gen resolves the same ties the same way
    stock = MoELayer(_sibling_config())(pack, weight_pack)
    _assert_close(stock, reference)


def _check_concurrent_streams(spec, ctx):
    """Calls on different streams share the kernels' device-global state (GEMM1
    accumulators, counters, launch parity); launches serialize on it in the
    kernels, so each call produces its own result: eager calls on two streams,
    then a CUDA graph replayed on one stream next to eager calls on another.
    Every output is pre-filled with NaN and checked, and the dispatch counters
    show that every eager call and every captured launch ran the kernels (a
    launch that failed would run trtllm-gen, with the same numerics). Stream 1
    routes every token to the same eight experts; with ``eager_tokens`` its
    calls have another token count than stream 0 and the graph, e.g. six-token
    graphs (kernel B's bulk-reduction path) next to ten-token calls on kernel
    B's wide returning-atomic path."""
    num_tokens = ctx.num_tokens
    eager_tokens = getattr(ctx, "eager_tokens", num_tokens)
    _, weight_pack = _weights()
    runner = _runner(MoELayer(_config()))
    inputs = [
        _inputs(num_tokens, "serving", seed=300),
        _inputs(eager_tokens, "skew", seed=301),
    ]
    packs = [_pack(*x) for x in inputs]
    references = [_reference(*x)[0] for x in inputs]
    rounds = 16

    def nan_outputs(tokens):
        return [
            torch.full((tokens, _H), float("nan"), dtype=torch.bfloat16, device="cuda")
            for _ in range(rounds)
        ]

    def counters():
        stats = bf16_decode.stats()
        assert not stats["run_errors"], stats["run_errors"]
        return (
            stats["dispatch_count"],
            stats["capture_dispatch_count"],
            stats["dispatched_num_tokens"],
        )

    start = counters()
    streams = [torch.cuda.Stream() for _ in packs]
    outs = [nan_outputs(t) for t in (num_tokens, eager_tokens)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for r in range(rounds):
        for k, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                _run_kernel(runner, packs[k], weight_pack, output=outs[k][r])
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    eager = counters()
    assert eager[0] - start[0] == 2 * rounds and eager[1] == start[1]
    for k, reference in enumerate(references):
        for out in outs[k]:
            _assert_close(out, reference)

    # A graph of four launches, replayed on stream 0 while stream 1 runs eagerly.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_outs = [_run_kernel(runner, packs[0], weight_pack) for _ in range(4)]
    torch.cuda.synchronize()
    built = counters()
    assert built[0] - eager[0] == 4 and built[1] - eager[1] == 4
    snapshots = [nan_outputs(num_tokens) for _ in graph_outs]
    outs = nan_outputs(eager_tokens)
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for r in range(rounds):
        with torch.cuda.stream(streams[0]):
            for out in graph_outs:
                out.fill_(float("nan"))
            graph.replay()
            for snapshot, out in zip(snapshots, graph_outs, strict=True):
                snapshot[r].copy_(out)
        with torch.cuda.stream(streams[1]):
            _run_kernel(runner, packs[1], weight_pack, output=outs[r])
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # replays run no Python: only the eager calls are new dispatches
    done = counters()
    assert done[0] - built[0] == rounds and done[1] == built[1]
    expected = dict(start[2])
    for tokens, calls in ((num_tokens, rounds + 4), (eager_tokens, 2 * rounds)):
        expected[tokens] = expected.get(tokens, 0) + calls
    assert done[2] == dict(sorted(expected.items()))
    for snapshot in snapshots:
        for out in snapshot:
            _assert_close(out, references[0])
    for out in outs:
        _assert_close(out, references[1])


def _check_cuda_graph_pool_replay(spec, ctx):
    """Selected eagerly, captured into one graph pool next to a size the kernel
    does not admit, replayed on changed inputs (outputs pre-filled with NaN)
    against trtllm_bf16_routed."""
    _widen_allowlist(ctx.monkeypatch, 1, 2)
    _, weight_pack = _weights()
    layer = MoELayer(_config())
    # The eager selection must not depend on measured speed: the reference
    # backend leaves the admitted sizes (1 and 2) to the kernel on this layer.
    (sibling,) = [r for r in layer.runners if r.backend_key == _SIBLING]
    sibling_accepts = sibling.accepts
    ctx.monkeypatch.setattr(
        sibling,
        "accepts",
        lambda act, weights: act.num_tokens not in (1, 2)
        and sibling_accepts(act, weights),
    )
    stock_layer = MoELayer(_sibling_config())
    sizes = [2, 1, 3]  # 3 stays on trtllm_bf16_routed, captured between admitted sizes
    static = {t: _inputs(t, seed=10 + t) for t in sizes}
    winners = {}
    for t in sizes:  # eager warm-up, as a server does before capture
        layer(_pack(*static[t]), weight_pack)
        winners[t] = layer.winner_backend
    torch.cuda.synchronize()
    assert winners == {2: _KEY, 1: _KEY, 3: _SIBLING}
    # count only the captures below
    captured_before = bf16_decode.stats()["capture_dispatch_count"]
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
    captured = bf16_decode.stats()["capture_dispatch_count"] - captured_before
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


def _params(**axes):
    """Cartesian product of named axes as (id, kwargs) pairs."""
    items = [("", {})]
    for key, values in axes.items():
        items = [
            ((f"{i}-" if i else "") + f"{key}{v}", {**kw, key: v})
            for i, kw in items
            for v in values
        ]
    return tuple(items)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


class _Bf16DecodeSpec(KernelSpec):
    def available(self):
        return _here()

    def config(self, **overrides):
        return _config(**overrides)

    def pin_device(self, monkeypatch, overrides):
        arch = overrides.pop("_arch", (10, 3))
        sms = overrides.pop("_sms", 152)
        monkeypatch.setattr(fi_utils, "get_compute_capability", lambda device: arch)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(multi_processor_count=sms),
        )
        return overrides

    def meta_call(self, num_tokens=None):
        return _meta_call(num_tokens)

    def contract_violation(self):
        return {"routing_logits": _meta((1, _E), torch.bfloat16)}

    def check_allowlist(self, allowlist, payload):
        assert allowlist["num_tokens"] == {1, 2, 6, 10}
        assert sorted(payload["num_tokens"]) == [1, 2, 6, 10]
        assert tuple(payload["compute_capability"]) == bf16_decode.COMPUTE_CAPABILITY

    def check_aot(self, monkeypatch):
        """flashinfer.aot precompiles the kernels' JIT module for exact SM103 only."""
        from flashinfer.fused_moe.backends.specialized.registry import (
            gen_specialized_moe_aot_modules,
        )
        from flashinfer.jit import bf16_moe_decode_sm103 as jit_module

        sentinel = SimpleNamespace(name="bf16_moe_decode_sm103")
        monkeypatch.setattr(
            jit_module, "gen_bf16_moe_decode_sm103_module", lambda: sentinel
        )
        assert sentinel in gen_specialized_moe_aot_modules({"sm103a_exact": True})
        assert sentinel not in gen_specialized_moe_aot_modules(
            {"sm100a_exact": True, "sm103": True}
        )

    def weights(self):
        return _weights()

    def inputs(self, num_tokens, kind="serving", seed=1):
        return _inputs(num_tokens, kind, seed)

    def pack(self, *inputs):
        return _pack(*inputs)

    def check_numerics(self, inputs, out, stock, layered, on_kernel):
        num_tokens = out.shape[0]
        reference, gap = _reference(*inputs)
        _assert_close(out, reference)
        # against trtllm-gen where the top-8 is not a near tie (see the
        # degenerate-inputs check)
        clear = gap.abs() >= 1e-3
        assert int(clear.sum()) >= (num_tokens + 1) // 2
        _assert_close(out[clear], stock[clear])
        if on_kernel:
            _assert_close(layered, reference)
        else:
            _assert_close(layered[clear], reference[clear])

    def check_replay(self, inputs, replayed, eager, on_kernel):
        reference, gap = _reference(*inputs)
        if on_kernel:
            _assert_close(replayed, reference)
        else:  # trtllm-gen against the exact key order
            clear = gap.abs() >= 1e-3
            _assert_close(replayed[clear], reference[clear])
        _assert_close(replayed, eager)


SPEC = _Bf16DecodeSpec(
    name=_NAME,
    kernel_module=bf16_decode,
    package=kernel_package,
    kernel_class=bf16_decode.Bf16DecodeSm103Glm52Kernel,
    sibling=_SIBLING,
    sibling_config=TrtllmBf16Config(),
    quant_variants=((QuantFormat.BF16, QuantFormat.BF16),),
    routing_modes=(RoutingInputMode.FromLogits,),
    activation_classes=(SwiGLU,),
    compute_capabilities=(103,),
    skip_reason=f"{_NAME} runs on SM103 GPUs with 150..152 SMs",
    config_cases=_CONFIG_CASES,
    call_cases=_CALL_CASES,
    meta_tokens=1,
    meta_rejected_tokens=3,
    numerics_cases=tuple((str(t), {"num_tokens": t}) for t in (1, 2, 6, 10)),
    lazy_rejected_tokens=3,
    lazy_admitted_tokens=1,
    graph_tokens=6,
    perf_cases=(("1", 1),),
    min_speedup=1.3,
    checks=(
        Check(
            "matches_trtllm_bf16_routed",
            _check_matches_trtllm_bf16_routed,
            _params(num_tokens=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)),
        ),
        Check(
            "degenerate_inputs",
            _check_degenerate_inputs,
            _params(
                kind=("equal_logits", "zeros", "bias_reorders", "skew"),
                num_tokens=(1, 16),
                scale=(2.5, None),
            ),
        ),
        Check(
            "routing_ties",
            _check_routing_ties,
            _params(ties=("all_equal", "cross_half"), num_tokens=(1, 2, 6, 10, 16)),
        ),
        Check(
            "concurrent_streams",
            _check_concurrent_streams,
            _params(num_tokens=(1, 2, 6, 10))
            + (("num_tokens6-eager_tokens10", {"num_tokens": 6, "eager_tokens": 10}),),
        ),
        Check("cuda_graph_pool_replay", _check_cuda_graph_pool_replay),
    ),
)
