"""Specialized MoE kernel ``fp8_block_prefill_sm103_qwen35`` of the unified MoE API.

One served configuration (Qwen3.5-397B-A17B-FP8 prefill: 512 experts, top-10
RenormalizeNaive from bf16 logits, hidden 4096, intermediate 1024, DeepSeek-FP8
SwiGLU, 1792..32768 tokens) is a kernel of the default ``specialized`` MoELayer
backend on SM103 with the CuTe DSL. It reads the ``trtllm_fp8_block`` weight
view; the layer skips it for every other configuration and arch and leaves
every other call on ``trtllm_fp8_block``.

Regression anchors: the registration and both guards (any host, meta tensors),
the expert-count permutation (any SM80+ GPU with the CuTe DSL) and on SM103 the
kernel through ``MoELayer`` against an fp32 reference of the FP8 data path and
against ``trtllm_fp8_block`` (``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``),
including the lazy build and CUDA-graph capture and replay.
"""

from __future__ import annotations

import ast
import json
import sys
from importlib import resources

import pytest
import torch
import torch.nn.functional as F

import flashinfer.utils as fi_utils
from flashinfer import RoutingMethodType
from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SpecializedMoEConfig,
    SwiGLU,
    TrtllmFp8BlockConfig,
)
from flashinfer.fused_moe.backends import specialized
from flashinfer.fused_moe.backends.specialized.kernels import (
    fp8_block_prefill_sm103_qwen35 as kernel_package,
)
from flashinfer.fused_moe.backends.specialized.kernels.fp8_block_prefill_sm103_qwen35 import (  # noqa: E501
    kernel as fp8_prefill,
)
from flashinfer.fused_moe.backends.specialized.runner import SpecializedMoERunner
from flashinfer.utils import get_compute_capability

_NAME = "fp8_block_prefill_sm103_qwen35"
_KEY = "specialized"
_SIBLING = "trtllm_fp8_block"
_E = fp8_prefill.NUM_EXPERTS
_TOP_K = fp8_prefill.TOP_K
_H = fp8_prefill.HIDDEN_SIZE
_I = fp8_prefill.INTERMEDIATE_SIZE
_BLOCK = fp8_prefill.BLOCK
_DEEPSEEK = QuantConfig(
    weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
)


def _here() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(torch.device("cuda")) == (10, 3)
        and is_cute_dsl_available()
    )


requires_kernel = pytest.mark.skipif(
    not _here(), reason=f"{_NAME} runs on SM103 with the CuTe DSL"
)


@pytest.fixture
def kernel_state(monkeypatch):
    monkeypatch.delenv(fp8_prefill.KILL_SWITCH_ENV, raising=False)
    fp8_prefill._reset_for_tests()
    yield fp8_prefill
    fp8_prefill._reset_for_tests()


def _config(**overrides) -> MoEConfig:
    """The served configuration (default backend list), with field overrides."""
    routing = dict(
        num_experts=_E, top_k=_TOP_K, method=RoutingMethodType.RenormalizeNaive
    )
    experts = dict(intermediate_size=_I)
    for key in list(overrides):
        if key in (
            "num_experts",
            "top_k",
            "method",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
        ):
            routing[key] = overrides.pop(key)
        elif key in ("intermediate_size", "local_expert_offset", "local_num_experts"):
            experts[key] = overrides.pop(key)
    overrides.setdefault("quant", _DEEPSEEK)
    overrides.setdefault("execution", ExecutionConfig(tune_max_num_tokens=32768))
    return MoEConfig(
        routing=RoutingConfig(**routing),
        experts=ExpertConfig(**experts),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Registration and guards (any host)
# ---------------------------------------------------------------------------


def test_registration():
    kernel = specialized.registered_kernels()[_NAME]
    assert kernel is kernel_package.KERNEL
    assert kernel is fp8_prefill.Fp8BlockPrefillSm103Qwen35Kernel
    assert kernel.weight_view == _SIBLING
    assert kernel.quant_variants == (
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
    )
    assert kernel.routing_modes == (RoutingInputMode.FromLogits,)
    assert kernel.activation_classes == (SwiGLU,)
    assert kernel.compute_capabilities == (103,)
    assert SpecializedMoEConfig.supported(103)
    assert _DEEPSEEK.pair in SpecializedMoERunner.supported_quant_variants
    assert _SIBLING in SpecializedMoERunner.weight_view_aliases
    assert fp8_prefill._KERNEL_MODULE == (
        "flashinfer.fused_moe.cute_dsl.blackwell.fp8_block_scale_moe_prefill"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoELayer needs CUDA")
def test_layer_candidates():
    """The default layer of the served configuration builds the specialized
    backend with this kernel only on SM103 with the CuTe DSL; elsewhere it is
    inert."""
    layer = MoELayer(_config())
    keys = [runner.backend_key for runner in layer.runners]
    if _here():
        assert keys == [_SIBLING, _KEY]
        assert [k.name for k in layer.runners[1].kernels] == [_NAME]
    else:
        assert keys == [_SIBLING]
    # An explicit trtllm_fp8_block list never includes it.
    explicit = MoELayer(
        _config(backend=BackendOptions(candidates=(TrtllmFp8BlockConfig(),)))
    )
    assert [runner.backend_key for runner in explicit.runners] == [_SIBLING]


_CONFIG_CASES = {
    "served": ({}, None),
    "num_experts": ({"num_experts": 256, "local_num_experts": 256}, "serves only"),
    "top_k": ({"top_k": 8}, "serves only"),
    "intermediate": ({"intermediate_size": 512}, "serves only"),
    "expert_parallel": ({"local_num_experts": 128}, "expert parallelism"),
    "expert_offset": (
        {"local_expert_offset": 128, "local_num_experts": 384},
        "expert parallelism",
    ),
    "renormalize": ({"method": RoutingMethodType.Renormalize}, "RenormalizeNaive"),
    "default_routing": ({"method": RoutingMethodType.Default}, "RenormalizeNaive"),
    "routing_groups": ({"n_group": 8, "topk_group": 4}, "grouped routing"),
    "routed_scaling": ({"routed_scaling_factor": 2.5}, "routed_scaling_factor"),
    "mxfp8": (
        {"quant": QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)},
        "does not support",
    ),
    "swiglu_limit": ({"activation": SwiGLU(limit=7.0)}, "default SwiGLU"),
    "geglu": ({"activation": GeGLU()}, "does not support"),
    "no_finalize": (
        {"finalize": MoEFinalizeConfig(do_finalize=False)},
        "do_finalize",
    ),
    "sm100": ({"_arch": 100}, "exact SM103"),
    "no_cute_dsl": ({"_cute_dsl": False}, "CuTe DSL"),
}


@pytest.mark.parametrize("case", list(_CONFIG_CASES), ids=list(_CONFIG_CASES))
def test_config_guard(monkeypatch, case):
    """``check_support`` binds the kernel for exactly the served configuration (so
    MoELayer skips the backend for every other one); the arch is pinned to SM103
    here so the geometry rules run on any host."""
    overrides, message = _CONFIG_CASES[case]
    overrides = dict(overrides)
    arch = overrides.pop("_arch", 103)
    cute_dsl = overrides.pop("_cute_dsl", True)
    monkeypatch.setattr(fp8_prefill, "cute_dsl_available", lambda: cute_dsl)
    monkeypatch.setattr(fi_utils, "get_compute_capability", lambda d: divmod(arch, 10))
    runner = SpecializedMoERunner(_config(**overrides), torch.device("cuda", 0))
    if message is None:
        runner.check_support()
        assert [k.name for k in runner.kernels] == [_NAME]
        return
    with pytest.raises(NotImplementedError, match=message):
        runner.check_support()


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _meta_call(num_tokens=4096):
    """A served call (activation pack + weight view) on meta tensors."""
    fp8 = torch.float8_e4m3fn
    pack = dict(
        hidden_states_q=_meta((num_tokens, _H), fp8),
        hidden_states_scale=_meta((_H // _BLOCK, num_tokens), torch.float32),
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=_meta((num_tokens, _E), torch.bfloat16),
    )
    view = {
        "gemm1_weights": _meta((_E, 2 * _I, _H), fp8),
        "gemm1_weights_scale": _meta(
            (_E, 2 * _I // _BLOCK, _H // _BLOCK), torch.float32
        ),
        "gemm2_weights": _meta((_E, _H, _I), fp8),
        "gemm2_weights_scale": _meta((_E, _H // _BLOCK, _I // _BLOCK), torch.float32),
    }
    return pack, view


# (pack overrides, weight-view overrides, expected reject reason). Meta tensors
# pass every check but the device check, the last one.
_CALL_CASES = {
    "served": ({}, {}, "device"),
    "tokens_1792": ({"num_tokens": 1792}, {}, "device"),
    "tokens_32768": ({"num_tokens": 32768}, {}, "device"),
    "tokens_1791": ({"num_tokens": 1791}, {}, "num_tokens"),
    "tokens_128": ({"num_tokens": 128}, {}, "num_tokens"),
    "tokens_32769": ({"num_tokens": 32769}, {}, "num_tokens"),
    "routing_bias": (
        {"routing_bias": _meta((_E,), torch.bfloat16)},
        {},
        "routing_options",
    ),
    "fp32_logits": (
        {"routing_logits": _meta((4096, _E), torch.float32)},
        {},
        "input_layout",
    ),
    "bf16_hidden": (
        {"hidden_states_q": _meta((4096, _H), torch.bfloat16)},
        {},
        "input_layout",
    ),
    "hidden_2048": (
        {
            "hidden_states_q": _meta((4096, 2048), torch.float8_e4m3fn),
            "hidden_states_scale": _meta((16, 4096), torch.float32),
        },
        {},
        "input_layout",
    ),
    # a prefix slice keeps the [32, T] shape but not the row stride T
    "scale_prefix_slice": (
        {"hidden_states_scale": _meta((_H // _BLOCK, 8192), torch.float32)[:, :4096]},
        {},
        "input_layout",
    ),
    "scale_token_major": (
        {"hidden_states_scale": _meta((4096, _H // _BLOCK), torch.float32).t()},
        {},
        "input_layout",
    ),
    "swiglu_params_in_view": (
        {},
        {"gemm1_clamp_limit": _meta((_E,), torch.float32)},
        "weight_view_extras",
    ),
    "weights_major_mn": (
        {},
        {"gemm2_weights": _meta((_E, _I, _H), torch.float8_e4m3fn).transpose(1, 2)},
        "weight_layout",
    ),
    "weights_padded": (
        {},
        {"gemm2_weights": _meta((_E, _H, _I + 128), torch.float8_e4m3fn)},
        "weight_layout",
    ),
    "mxfp8_scales": (
        {},
        {"gemm1_weights_scale": _meta((_E, 2 * _I, _H // 32), torch.uint8)},
        "weight_layout",
    ),
}


@pytest.mark.parametrize("case", list(_CALL_CASES), ids=list(_CALL_CASES))
def test_call_guard(kernel_state, case):
    pack_overrides, view_overrides, reason = _CALL_CASES[case]
    pack_overrides = dict(pack_overrides)
    pack, view = _meta_call(pack_overrides.pop("num_tokens", 4096))
    pack.update(pack_overrides)
    view.update(view_overrides)
    seen: set = set()
    device = torch.device("cuda", 0)
    assert not kernel_state.admit(MoEActivationPack(**pack), view, device, seen)
    assert kernel_state.stats()["rejects"] == {reason: 1}
    assert not seen


def test_kill_switch_and_errors(kernel_state, monkeypatch):
    spec = kernel_state
    pack, view = _meta_call()
    pack = MoEActivationPack(**pack)
    device = torch.device("cuda", 0)
    # read at call time
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
    assert not spec.admit(pack, view, device, set())
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "0")
    assert not spec.admit(pack, view, device, set())
    assert spec.stats()["rejects"] == {"kill_switch": 1, "device": 1}

    # a guard failure rejects the call instead of raising
    def broken():
        raise RuntimeError("allowlist read failed")

    monkeypatch.setattr(spec, "_load_allowlist", broken)
    assert not spec.admit(pack, view, device, set())
    assert spec.stats()["rejects"]["guard_error"] == 1

    # direct use outside the contract raises instead of launching
    bad = MoEActivationPack(
        **{**_meta_call()[0], "routing_logits": _meta((4096, _E), torch.float32)}
    )
    with pytest.raises(ValueError, match="input_layout"):
        spec.check_contract(bad, view)


def test_stats(kernel_state, monkeypatch):
    spec = kernel_state
    # the token-count allowlist ships as package data
    payload = json.loads(
        resources.files(kernel_package.__name__).joinpath("workloads.json").read_text()
    )
    spec._load_allowlist.cache_clear()
    allowlist = spec._load_allowlist()
    assert allowlist is not None
    assert allowlist["min"] == payload["num_tokens"]["min"]
    assert allowlist["max"] == payload["num_tokens"]["max"] <= spec.MAX_TOKENS
    assert tuple(payload["compute_capability"]) == spec.COMPUTE_CAPABILITY

    device = torch.device("cuda", 0)
    for num_tokens in (1024, 4096):
        pack, view = _meta_call(num_tokens)
        spec.admit(MoEActivationPack(**pack), view, device, set())
    st = kernel_package.stats()
    assert st == specialized.stats()[_NAME]
    assert st["calls"] == 2 and st["dispatch_count"] == 0
    assert st["rejects"] == {"num_tokens": 1, "device": 1}
    json.dumps(st)

    # side-effect free: no kernel-module import, build, CUDA call or file read
    monkeypatch.delitem(sys.modules, spec._KERNEL_MODULE, raising=False)

    def no_cuda(*args, **kwargs):
        raise AssertionError("stats() touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda)
    monkeypatch.setattr(torch.cuda, "current_device", no_cuda)
    spec._load_allowlist.cache_clear()
    st = kernel_package.stats()
    assert spec._KERNEL_MODULE not in sys.modules
    assert spec._load_allowlist.cache_info().currsize == 0
    assert st["kernel_module_loaded"] is False and st["allowlist"] is None


def test_no_aot_modules():
    """The CuTe DSL programs compile at run time; nothing goes to flashinfer.aot."""
    kernel = specialized.registered_kernels()[_NAME]
    assert kernel.aot_jit_specs({"sm103": True, "sm103a_exact": True}) == []


def test_scan_clamp():
    # shfl.sync.up clamps at the lower segment bound, so its packed clamp operand
    # must be 0. CuTe DSL < 4.7 defaults shuffle_sync_up to 31, which makes every
    # lane read its own value: the scan of the per-expert counts then places rows
    # far past the workspace. Every call passes the clamp explicitly.
    source = (
        resources.files("flashinfer.fused_moe.cute_dsl.blackwell")
        .joinpath("fp8_block_scale_moe_prefill.py")
        .read_text()
    )
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        == "shuffle_sync_up"
    ]
    assert len(calls) >= 2
    for call in calls:
        clamp = {kw.arg: kw.value for kw in call.keywords}.get("mask_and_clamp")
        assert isinstance(clamp, ast.Constant) and clamp.value == 0, ast.unparse(call)


_PERM_PROGRAMS: dict = {}


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 0)
        and is_cute_dsl_available()
    ),
    reason="the permutation programs need a CUDA GPU (SM80+) with the CuTe DSL",
)
@pytest.mark.parametrize(
    "num_tokens, kind",
    [(1792, "serving"), (8704, "serving"), (32768, "serving"), (8704, "warmup")],
)
def test_permutation(num_tokens, kind):
    """Rank / scan / rows / place of the expert permutation against a torch
    reference; they use no Blackwell feature, so they run on the current GPU."""
    import cuda.bindings.driver as cuda

    from flashinfer.fused_moe.cute_dsl.blackwell import (
        fp8_block_scale_moe_prefill as kernels,
    )

    index = torch.cuda.current_device()
    if index not in _PERM_PROGRAMS:
        _PERM_PROGRAMS[index] = kernels._compile_perm()
    rank_p, scan_p, rows_p, place_p = _PERM_PROGRAMS[index]
    dev = torch.device("cuda", index)
    if kind == "warmup":  # equal logits: every token on experts 0..9 (one scan lane)
        ids = torch.arange(_TOP_K, dtype=torch.int32).repeat(num_tokens, 1)
    else:  # serving-like skew over every expert, so every scan lane holds tokens
        gen = torch.Generator().manual_seed(0)
        popularity = 1.0 / (1.0 + torch.randperm(_E, generator=gen).float() / 8.0)
        ids = torch.multinomial(
            popularity.expand(num_tokens, _E), _TOP_K, generator=gen
        ).to(torch.int32)
    flat = ids.to(dev).view(-1)
    tk, m = flat.numel(), kernels.padded_rows(num_tokens)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def i32(n):
        return torch.empty(n, dtype=torch.int32, device=dev)

    rank, row_of = i32(tk), i32(tk)
    pad_off, pad_end, shift_e = i32(_E), i32(_E), i32(_E)
    gidx, src_tok, rstate, gidx_enc = i32(m), i32(m), i32(m), i32(m)
    cnt = torch.zeros(_E, dtype=torch.int32, device=dev)
    rank_p(flat, rank, cnt, stream)
    scan_p(cnt, pad_off, pad_end, shift_e, stream)
    rows_p(pad_end, pad_off, cnt, gidx, src_tok, rstate, shift_e, gidx_enc, stream)
    place_p(flat, rank, pad_off, row_of, src_tok, stream)
    torch.cuda.synchronize()

    counts = torch.bincount(flat.long(), minlength=_E)
    assert torch.equal(cnt.long(), counts)
    padded = (counts + _BLOCK - 1) // _BLOCK * _BLOCK
    ref_end = torch.cumsum(padded, 0)
    ref_off = ref_end - padded
    assert torch.equal(pad_off.long(), ref_off)
    assert torch.equal(pad_end.long(), ref_end)
    assert torch.equal(shift_e.long(), torch.cumsum(counts, 0) - counts - ref_off)
    # every assignment lands on its own real row of its expert, inside the workspace
    rows = row_of.long()
    assert int(rows.min()) >= 0 and int(rows.max()) < m
    assert rows.unique().numel() == tk
    assert bool((rstate[rows] == 1).all()) and int((rstate == 1).sum()) == tk
    assert torch.equal(gidx[rows], flat)
    assert torch.equal(src_tok[rows].long(), torch.arange(tk, device=dev) // _TOP_K)
    assert bool((gidx[int(ref_end[-1]) :] == -1).all())
    # compact (unpadded) rows of the T <= 8192 programs: a permutation of 0..tk-1
    enc = gidx_enc[rows].long()
    assert torch.equal(enc & 1023, flat.long())
    assert torch.equal((rows - (enc >> 10)).sort().values, torch.arange(tk, device=dev))


# ---------------------------------------------------------------------------
# The served path on SM103 with the CuTe DSL
# ---------------------------------------------------------------------------


def _quant_rows(x):
    """[n, C] fp32 -> (FP8 E4M3 [n, C], scales [n, C/128]), scale max(amax/448, FLT_MIN)."""
    n, c = x.shape
    xb = x.reshape(n, c // _BLOCK, _BLOCK)
    s = torch.clamp(
        xb.abs().amax(dim=-1, keepdim=True) / 448.0, min=torch.finfo(torch.float32).tiny
    )
    return (xb / s).to(torch.float8_e4m3fn).reshape(n, c), s.reshape(n, c // _BLOCK)


def _block_gemm(aq, sa, wq, sw):
    """sum over 128-wide K blocks of sa[:, kb] * sw[n // 128, kb] * (aq_kb . wq_kb),
    exact FP8 products accumulated in fp32."""
    n, k = aq.shape
    kb = k // _BLOCK
    w = wq.float().reshape(wq.shape[0], kb, _BLOCK).permute(1, 2, 0)
    a = aq.float().reshape(n, kb, _BLOCK).transpose(0, 1)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        p = torch.bmm(a, w)  # [kb, n, N]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    p.mul_(sa.float().t().unsqueeze(-1))
    p.mul_(sw.float().repeat_interleave(_BLOCK, dim=0).t().unsqueeze(1))
    return p.sum(dim=0)


@torch.no_grad()
def _reference(logits, hs, hs_scale, w1, s1, w2, s2):
    """fp32 reference of the DeepSeek-FP8 data path: GEMM1 and SwiGLU outputs
    re-quantized to FP8 per (token, 128 columns)."""
    p = torch.softmax(logits.float(), dim=-1)
    val, idx = torch.topk(p, _TOP_K, dim=-1)
    wts = val / val.sum(dim=-1, keepdim=True)
    xs = hs_scale.float().t().contiguous()
    out = torch.zeros(hs.shape[0], _H, dtype=torch.float32, device=hs.device)
    for e in torch.unique(idx).tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        g = _block_gemm(hs[tok], xs[tok], w1[e], s1[e])
        gq, gs = _quant_rows(g)
        g = gq.float().reshape(-1, 2 * _I // _BLOCK, _BLOCK) * gs[..., None]
        g = g.reshape(-1, 2 * _I)
        aq, sa = _quant_rows(F.silu(g[:, _I:]) * g[:, :_I])
        o = _block_gemm(aq, sa, w2[e], s2[e])
        out.index_add_(0, tok, o * wts[tok, slot][:, None])
    return out.to(torch.bfloat16)


def _cos(ref, out):
    r, u = ref.float().reshape(-1), out.float().reshape(-1)
    assert torch.isfinite(u).all()
    cos = float(torch.dot(r, u) / (r.norm() * u.norm() + 1e-30))
    return cos, float((u - r).norm() / (r.norm() + 1e-30))


@pytest.fixture(scope="module")
def weights():
    """(w1, s1, w2, s2) of the served geometry in the trtllm_fp8_block layout, and
    the weight pack registering them under "trtllm_fp8_block" only."""
    if not _here():
        pytest.skip("requires SM103 with the CuTe DSL")
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    w1 = torch.empty(_E, 2 * _I, _H, dtype=torch.float8_e4m3fn, device=dev)
    w2 = torch.empty(_E, _H, _I, dtype=torch.float8_e4m3fn, device=dev)
    for e in range(_E):
        w1[e] = torch.randn(2 * _I, _H, device=dev, generator=gen).to(w1.dtype)
        w2[e] = torch.randn(_H, _I, device=dev, generator=gen).to(w2.dtype)
    s1 = 0.01 + 0.01 * torch.rand(
        _E, 2 * _I // _BLOCK, _H // _BLOCK, device=dev, generator=gen
    )
    s2 = 0.01 + 0.01 * torch.rand(
        _E, _H // _BLOCK, _I // _BLOCK, device=dev, generator=gen
    )
    pack = MoEWeightPack()
    pack.prepare_for(
        _SIBLING,
        {
            "gemm1_weights": w1,
            "gemm1_weights_scale": s1,
            "gemm2_weights": w2,
            "gemm2_weights_scale": s2,
        },
    )
    return (w1, s1, w2, s2), pack


def _inputs(num_tokens, seed=1, skew=False):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    logits = torch.randn(num_tokens, _E, device=dev, generator=gen)
    if skew:  # half of the tokens prefer the same ten experts
        logits[: num_tokens // 2, :_TOP_K] += 8.0
    hs = torch.randn(num_tokens, _H, device=dev, generator=gen).to(torch.float8_e4m3fn)
    hs_scale = 0.5 + torch.rand(_H // _BLOCK, num_tokens, device=dev, generator=gen)
    return logits.to(torch.bfloat16), hs, hs_scale


def _pack(logits, hs, hs_scale):
    return MoEActivationPack(
        hidden_states_q=hs,
        hidden_states_scale=hs_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
    )


def _runner(layer):
    (runner,) = [r for r in layer.runners if r.backend_key == _KEY]
    return runner


@requires_kernel
@pytest.mark.parametrize(
    "num_tokens, skew",
    [(1800, False), (4096, True), (8192, False), (9216, False)],
    ids=["1800", "4096_skew", "8192_compact", "9216_padded"],
)
def test_matches_reference(kernel_state, weights, monkeypatch, num_tokens, skew):
    spec = kernel_state
    tensors, weight_pack = weights
    pack = _pack(*_inputs(num_tokens, skew=skew))
    layer = MoELayer(_config())
    assert [r.backend_key for r in layer.runners] == [_SIBLING, _KEY]

    # The specialized runner on its own (the numerics under test) ...
    runner = _runner(layer)
    assert runner.accepts(pack, weight_pack)
    out = runner.forward(runner.pack_inputs(pack, weight_pack)).clone()
    assert spec.stats()["dispatched_num_tokens"] == {num_tokens: 1}
    # ... and through the layer, which times it against trtllm_fp8_block once.
    layered = layer(pack, weight_pack).clone()
    assert num_tokens in spec.stats()["dispatched_num_tokens"]
    if num_tokens >= 4096:  # >= 1.4x faster here: the selection is not noise
        assert layer.winner_backend == _KEY
    if layer.winner_backend == _KEY:
        # every output row depends only on its own routed rows: bitwise repeatable
        assert torch.equal(layered, out)

    # The kill switch leaves the call to trtllm_fp8_block.
    monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
    dispatched = spec.stats()["dispatch_count"]
    stock = layer(pack, weight_pack).clone()
    assert layer.winner_backend == _SIBLING
    assert spec.stats()["dispatch_count"] == dispatched
    torch.cuda.synchronize()

    ref = _reference(
        pack.routing_logits, pack.hidden_states_q, pack.hidden_states_scale, *tensors
    )
    # FP8-data-path gate: another accumulation order lands at rel-L2 0.002-0.004
    # here; dropping an FP8 re-quantization or rounding weights to bf16, >= 0.016.
    for name, o in (("specialized", out), (_SIBLING, stock)):
        cos, rel = _cos(ref, o)
        assert cos >= 0.9995 and rel <= 0.012, (name, cos, rel)
    cos, _ = _cos(stock, out)
    assert cos >= 0.999, cos


@requires_kernel
def test_lazy_build_and_range(kernel_state, weights):
    spec = kernel_state
    _, weight_pack = weights
    layer = MoELayer(_config())
    # outside the allowlist: trtllm_fp8_block only, nothing built
    layer(_pack(*_inputs(1024)), weight_pack)
    assert layer.winner_backend == _SIBLING
    st = spec.stats()
    assert st["rejects"] == {"num_tokens": 1} and st["precompiled"] == []
    # the first admitted eager call builds the programs for the device
    pack = _pack(*_inputs(2048))
    layer(pack, weight_pack)
    st = spec.stats()
    assert torch.cuda.current_device() in st["precompiled"]
    assert st["compiled_variants"] >= st["distinct_kernels_for_allowlist"]
    assert 2048 in st["dispatched_num_tokens"]
    # a cached winner: one launch per later call
    if layer.winner_backend == _KEY:
        dispatched = st["dispatch_count"]
        layer(pack, weight_pack)
        assert spec.stats()["dispatch_count"] == dispatched + 1


@requires_kernel
@pytest.mark.parametrize("case", ["selected_eagerly", "unbuilt", "unselected"])
def test_cuda_graph(kernel_state, weights, monkeypatch, case):
    """A captured call runs the kernel only when its programs exist and the layer
    already selected its winner for that call eagerly; otherwise it is captured
    on trtllm_fp8_block (never a build or a timing inside capture)."""
    spec = kernel_state
    _, weight_pack = weights
    num_tokens = 4608
    logits, hs, hs_scale = _inputs(num_tokens, seed=2)
    pack = _pack(logits, hs, hs_scale)
    layer = MoELayer(_config())
    if case == "selected_eagerly":
        layer(pack, weight_pack)
        assert layer.winner_backend == _KEY
    else:
        # warm trtllm_fp8_block for this call eagerly (its first launch must not
        # happen in capture) with the specialized backend switched off
        monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
        layer(pack, weight_pack)
        monkeypatch.delenv(spec.KILL_SWITCH_ENV)
        if case == "unselected":  # built by another call size, then capture 4608
            layer(_pack(*_inputs(2048)), weight_pack)
            assert spec.is_ready(hs.device)
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
    for dst, src in zip((logits, hs, hs_scale), new, strict=True):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    if not specialized_capture:
        monkeypatch.setenv(spec.KILL_SWITCH_ENV, "1")
    eager = layer(_pack(*new), weight_pack)
    torch.cuda.synchronize()
    if specialized_capture:
        assert layer.winner_backend == _KEY
        assert torch.equal(replayed, eager)
    else:
        cos, _ = _cos(eager, replayed)
        assert cos >= 0.9999, cos


@requires_kernel
@pytest.mark.parametrize("num_tokens", [1800, 9216])
def test_compact_and_padded_agree(kernel_state, weights, num_tokens):
    """The build launches both program sets (compact rows for T <= 8192, padded
    rows above); each is exact for any token count."""
    from flashinfer.fused_moe.cute_dsl.blackwell import (
        fp8_block_scale_moe_prefill as kernels,
    )

    tensors, weight_pack = weights
    logits, hs, hs_scale = _inputs(num_tokens, seed=5)
    layer = MoELayer(_config())
    layer(_pack(logits, hs, hs_scale), weight_pack)  # builds
    assert kernel_state.is_ready(hs.device)
    outs = []
    for compact in (True, False):
        out = torch.empty(num_tokens, _H, dtype=torch.bfloat16, device="cuda")
        kernels.run(logits, hs, hs_scale, *tensors, out, compact=compact)
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
