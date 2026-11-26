"""Spec of the specialized MoE kernel ``fp8_block_prefill_sm103_qwen35``.

One served configuration (Qwen3.5-397B-A17B-FP8 prefill: 512 experts, top-10
RenormalizeNaive from bf16 logits, hidden 4096, intermediate 1024, DeepSeek-FP8
SwiGLU, 1792..32768 tokens) on SM103 with the CuTe DSL. It reads the
``trtllm_fp8_block`` weight view; the layer skips it for every other
configuration and arch and leaves every other call on ``trtllm_fp8_block``.
The shared tests are in ``tests/moe/test_unified_moe_specialized.py``; the
checks below are this kernel's own: the explicit shuffle clamp of the CuTe DSL
scan and the expert-count permutation (any SM80+ GPU with the CuTe DSL).
"""

from __future__ import annotations

import ast
from importlib import resources

import torch
import torch.nn.functional as F

import flashinfer.utils as fi_utils
from flashinfer import RoutingMethodType
from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.fused_moe import (
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
    SwiGLU,
    TrtllmFp8BlockConfig,
)
from flashinfer.fused_moe.backends.specialized.kernels import (
    fp8_block_prefill_sm103_qwen35 as kernel_package,
)
from flashinfer.fused_moe.backends.specialized.kernels.fp8_block_prefill_sm103_qwen35 import (  # noqa: E501
    kernel as fp8_prefill,
)
from flashinfer.utils import get_compute_capability

from . import Check, KernelSpec

_NAME = "fp8_block_prefill_sm103_qwen35"
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


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _meta_call(num_tokens=None):
    """A served call (activation pack + weight view) on meta tensors."""
    num_tokens = 4096 if num_tokens is None else num_tokens
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


# ---------------------------------------------------------------------------
# The served path on SM103 with the CuTe DSL
# ---------------------------------------------------------------------------

_WEIGHTS: dict = {}


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


def _weights():
    """(w1, s1, w2, s2) of the served geometry in the trtllm_fp8_block layout, and
    the weight pack registering them under "trtllm_fp8_block" only (built once)."""
    if "pack" in _WEIGHTS:
        return _WEIGHTS["tensors"], _WEIGHTS["pack"]
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
    _WEIGHTS["tensors"], _WEIGHTS["pack"] = (w1, s1, w2, s2), pack
    return _WEIGHTS["tensors"], _WEIGHTS["pack"]


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


# ---------------------------------------------------------------------------
# This kernel's own checks
# ---------------------------------------------------------------------------


_PERM_PROGRAMS: dict = {}


def _check_scan_clamp(spec, ctx):
    # shfl.sync.up clamps at the lower segment bound, so its packed clamp operand
    # must be 0. CuTe DSL < 4.7 defaults shuffle_sync_up to 31, which makes every
    # lane read its own value: the scan of the per-expert counts then places rows
    # far past the workspace. Every call passes the clamp explicitly.
    # Resolve the file from the already-imported flashinfer.fused_moe package:
    # resources.files() on the blackwell package would import it, and its
    # __init__ imports the CuTe DSL (cutlass), which this host check must not need.
    source = (
        resources.files("flashinfer.fused_moe")
        .joinpath("cute_dsl")
        .joinpath("blackwell")
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


def _check_permutation(spec, ctx):
    """Rank / scan / rows / place of the expert permutation against a torch
    reference; they use no Blackwell feature, so they run on the current GPU."""
    num_tokens, kind = ctx.num_tokens, ctx.kind
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


def _compact_padded(num_tokens):
    """The build launches both program sets (compact rows for T <= 8192, padded
    rows above); each is exact for any token count."""
    from flashinfer.fused_moe.cute_dsl.blackwell import (
        fp8_block_scale_moe_prefill as kernels,
    )

    tensors, weight_pack = _weights()
    logits, hs, hs_scale = _inputs(num_tokens, seed=5)
    layer = MoELayer(_config())
    layer(_pack(logits, hs, hs_scale), weight_pack)  # builds
    assert fp8_prefill.is_ready(hs.device)
    outs = []
    for compact in (True, False):
        out = torch.empty(num_tokens, _H, dtype=torch.bfloat16, device="cuda")
        kernels.run(logits, hs, hs_scale, *tensors, out, compact=compact)
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


def _perm_skip():
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 0)
        and is_cute_dsl_available()
    ):
        return "the permutation programs need a CUDA GPU (SM80+) with the CuTe DSL"
    return None


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


class _Fp8PrefillSpec(KernelSpec):
    def available(self):
        return _here()

    def config(self, **overrides):
        return _config(**overrides)

    def pin_device(self, monkeypatch, overrides):
        arch = overrides.pop("_arch", 103)
        cute_dsl = overrides.pop("_cute_dsl", True)
        monkeypatch.setattr(fp8_prefill, "cute_dsl_available", lambda: cute_dsl)
        monkeypatch.setattr(
            fi_utils, "get_compute_capability", lambda d: divmod(arch, 10)
        )
        return overrides

    def meta_call(self, num_tokens=None):
        return _meta_call(num_tokens)

    def contract_violation(self):
        return {"routing_logits": _meta((4096, _E), torch.float32)}

    def check_allowlist(self, allowlist, payload):
        assert allowlist["min"] == payload["num_tokens"]["min"]
        assert (
            allowlist["max"] == payload["num_tokens"]["max"] <= fp8_prefill.MAX_TOKENS
        )
        assert tuple(payload["compute_capability"]) == fp8_prefill.COMPUTE_CAPABILITY

    def check_aot(self, monkeypatch):
        """The CuTe DSL programs compile at run time; nothing goes to flashinfer.aot."""
        kernel = kernel_package.KERNEL
        assert kernel.aot_jit_specs({"sm103": True, "sm103a_exact": True}) == []

    def weights(self):
        return _weights()

    def inputs(self, num_tokens, kind="serving", seed=1):
        return _inputs(num_tokens, seed=seed, skew=kind == "skew")

    def pack(self, *inputs):
        return _pack(*inputs)

    def check_numerics(self, inputs, out, stock, layered, on_kernel):
        if on_kernel:
            # every output row depends only on its own routed rows: bitwise repeatable
            assert torch.equal(layered, out)
        tensors, _ = _weights()
        ref = _reference(*inputs, *tensors)
        # FP8-data-path gate: another accumulation order lands at rel-L2 0.002-0.004
        # here; dropping an FP8 re-quantization or rounding weights to bf16, >= 0.016.
        for name, o in (("specialized", out), (_SIBLING, stock), ("layer", layered)):
            cos, rel = _cos(ref, o)
            assert cos >= 0.9995 and rel <= 0.012, (name, cos, rel)
        cos, _ = _cos(stock, out)
        assert cos >= 0.999, cos

    def check_replay(self, inputs, replayed, eager, on_kernel):
        if on_kernel:
            assert torch.equal(replayed, eager)
        else:
            cos, _ = _cos(eager, replayed)
            assert cos >= 0.9999, cos

    def check_build_stats(self, stats):
        assert stats["compiled_variants"] >= stats["distinct_kernels_for_allowlist"]

    def check_compact_padded(self, num_tokens):
        _compact_padded(num_tokens)


SPEC = _Fp8PrefillSpec(
    name=_NAME,
    kernel_module=fp8_prefill,
    package=kernel_package,
    kernel_class=fp8_prefill.Fp8BlockPrefillSm103Qwen35Kernel,
    sibling=_SIBLING,
    sibling_config=TrtllmFp8BlockConfig(),
    quant_variants=((QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),),
    routing_modes=(RoutingInputMode.FromLogits,),
    activation_classes=(SwiGLU,),
    compute_capabilities=(103,),
    skip_reason=f"{_NAME} runs on SM103 with the CuTe DSL",
    kernel_module_path="flashinfer.fused_moe.cute_dsl.blackwell.fp8_block_scale_moe_prefill",
    config_cases=_CONFIG_CASES,
    call_cases=_CALL_CASES,
    meta_tokens=4096,
    meta_rejected_tokens=1024,
    numerics_cases=(
        ("1800", {"num_tokens": 1800}),
        ("4096_skew", {"num_tokens": 4096, "kind": "skew"}),
        ("8192_compact", {"num_tokens": 8192}),
        ("9216_padded", {"num_tokens": 9216}),
    ),
    lazy_rejected_tokens=1024,
    lazy_admitted_tokens=2048,
    graph_tokens=4608,
    compact_padded_tokens=(1800, 9216),
    perf_cases=(("4096", 4096), ("8192_compact", 8192), ("9216_padded", 9216)),
    min_speedup=1.2,
    checks=(
        Check("scan_clamp", _check_scan_clamp, needs_kernel=False),
        Check(
            "permutation",
            _check_permutation,
            (
                ("1792-serving", {"num_tokens": 1792, "kind": "serving"}),
                ("8704-serving", {"num_tokens": 8704, "kind": "serving"}),
                ("32768-serving", {"num_tokens": 32768, "kind": "serving"}),
                ("8704-warmup", {"num_tokens": 8704, "kind": "warmup"}),
            ),
            needs_kernel=False,
            skip=_perm_skip,
        ),
    ),
)
