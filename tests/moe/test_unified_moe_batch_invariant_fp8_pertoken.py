# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-token fp8 batch-invariant MoE backend of the unified MoE layer (``BatchInvariantFp8PerTokenConfig``): the classes
``moe.e4m3.<act>.pertoken`` -- e4m3 experts with one fp32 scale per output channel, e4m3 activations with one dynamic fp32
scale per token (the intermediate requantized per row) -- one per typed activation of the layer and its scalars.

The class has exact CPU models of everything but the tensor-core step (the routing, the per-token quantization -- the row
amax, the multiplier ``div(448, nzor(amax, 448))``, the e4m3 conversion, the scale ``div(1, m)`` -- the activation chain
under the (row x channel) scales, GEMM2's scaled chain, the slot-order combine), so the tests are bit comparisons: the
quantizer kernel against the CPU chain in both placements (the rows themselves, the scatter to the tokens' dense rows with
dropped slots), the layer's output against the class composed from its own pieces (the routing on the CPU, the grouped
GEMM1 as the raw fp32 fold, the CPU chains, the dense batch-invariant fp8 GEMM per expert for GEMM2, the CPU combine) for
every activation, both activation inputs (bf16 rows quantized by the class, the caller's per-token quantization), run-to-run
determinism including CUDA-graph replay, batch / permutation / sub-batch invariance, agreement of every tactic the autotuner
may pick and of the swapped records with the grouped ones, the three routing input modes, the error contract, and a
plausibility observation against FlashInfer's torch reference of the per-channel / per-token fp8 MoE. The device-free tests
pin the registration, the fingerprint dimensions, the class's terms and the dispatcher's membership in the record spaces.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    BatchInvariantFp8PerTokenConfig,
    ExecutionConfig,
    ExpertConfig,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    PowLU,
    QuantConfig,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND
from flashinfer.fused_moe.batch_invariant.runner import BatchInvariantMoeRunner
from flashinfer.fused_moe.batch_invariant.runner_fp8 import (
    MOE_FP8_ARCHS,
    BatchInvariantFp8PerTensorRunner,
)
from flashinfer.fused_moe.batch_invariant.runner_fp8_pertoken import (
    BatchInvariantFp8PerTokenRunner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_FP8_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)
_SWIGLU = SwiGLU()
ACTIVATIONS = [
    SwiGLU(),
    SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
    SwiGLUStep(),
    SwiGLUStep(limit=3.0),
    GeGLU(),
    GeGLUTanh(),
    SiTU(),
    SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=5.0),
    PowLU(),
    PowLU(m=1.5, limit=4.0),
    ReLU2(),
    Identity(),
    GELU(),
    ReLU(),
    SiLU(),
]
ACTIVATION_IDS = [
    "swiglu",
    "swiglu_oai",
    "swiglu_step",
    "swiglu_step_limit3",
    "geglu",
    "geglu_tanh",
    "situ",
    "situ_clamped",
    "powlu",
    "powlu_limit4",
    "relu2",
    "identity",
    "gelu",
    "relu",
    "silu",
]
KEY = BatchInvariantFp8PerTokenRunner.backend_key


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    return t.view(
        {
            torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float32: torch.int32,
            torch.float8_e4m3fn: torch.uint8,
        }[t.dtype]
    )


def _problem(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    seed: int,
    gated: bool = True,
    adversarial: bool = False,
):
    """bf16 activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated); the adversarial
    form spreads the rows' and the experts' magnitudes over the format's range (per-row exponents: the per-token scales differ per row,
    the per-channel scales per weight row)."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        x = (
            torch.randn(T, H, generator=g)
            * torch.exp2(torch.randint(-8, 6, (T, 1), generator=g).float())
        ).to(torch.bfloat16)
        x[min(3, T - 1)] = 0.0  # a zero row: multiplier 1
        logits = (torch.randn(T, E, generator=g) * 2).float()
        logits[0, : min(k, E)] = 1.0
        span = torch.exp2(torch.randint(-3, 3, (E, 1, 1), generator=g).float())
        rowspan = torch.exp2(
            torch.randint(-4, 5, (E, 2 * I if gated else I, 1), generator=g).float()
        )
    else:
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
        logits = (torch.randn(T, E, generator=g) * 2).float()
        span = rowspan = 1.0
    n1 = 2 * I if gated else I
    w1 = (torch.randn(E, n1, H, generator=g) * 0.05 * span * rowspan).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05 * span).to(torch.bfloat16)
    return x.cuda(), logits.cuda(), w1.cuda(), w2.cuda()


def _config(
    k: int,
    E: int,
    I: int,
    act=_SWIGLU,
    method=RoutingMethodType.Renormalize,
    tune_max: int = 4096,
):
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=QuantConfig(
            weight=QuantFormat.FP8PerChannel, activation=QuantFormat.FP8PerToken
        ),
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp8PerTokenConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8PerTokenConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(KEY, view)
    return pack


def _logits_pack(
    x: torch.Tensor, logits: torch.Tensor, scale=None, bias=None
) -> MoEActivationPack:
    return MoEActivationPack(
        x,
        scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    packed: bool = False,
    scale=None,
) -> MoEActivationPack:
    if packed:
        return MoEActivationPack(
            x,
            scale,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        x,
        scale,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The config is a known backend of the layer, mapped to its runner, absent from the default candidates, and declares the
    per-channel / per-token fp8 quantization pair, the three routing modes and every activation class of the layer."""
    assert BatchInvariantFp8PerTokenConfig in ALL_BACKEND_CONFIGS
    assert (
        _BACKEND_RUNNERS[BatchInvariantFp8PerTokenConfig]
        is BatchInvariantFp8PerTokenRunner
    )
    assert not any(
        isinstance(c, BatchInvariantFp8PerTokenConfig)
        for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantFp8PerTokenConfig.supported(a)
    ] == [100, 103, 107]
    assert issubclass(BatchInvariantFp8PerTokenRunner, BatchInvariantFp8PerTensorRunner)
    assert (
        KEY
        == "batch_invariant_fp8_per_token"
        != BatchInvariantFp8PerTensorRunner.backend_key
    )
    assert BatchInvariantFp8PerTokenRunner.supported_quant_variants == (
        (QuantFormat.FP8PerChannel, QuantFormat.FP8PerToken),
    )
    assert QuantFormat.FP8PerChannel.value == 9 and QuantFormat.FP8PerToken.value == 10
    assert set(BatchInvariantFp8PerTokenRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert set(BatchInvariantFp8PerTokenRunner.supported_activation_classes) == set(
        BatchInvariantMoeRunner.supported_activation_classes
    )
    assert (
        repr(BatchInvariantFp8PerTokenConfig()) == "BatchInvariantFp8PerTokenConfig()"
    )
    assert (
        "fused_moe_fp8_pertoken" in bi.CLASSES
        and bi.class_name("fused_moe_fp8_pertoken") == "moe.e4m3.pertoken"
    )


def test_fingerprint_dimensions_terms_and_pins():
    """``fingerprint("fused_moe_fp8_pertoken", ...)`` takes the geometry, the routing method and the activation; it differs from the
    per-tensor class's at the same dimensions; every listed activation is its own class; the terms are the class's: the multiplier
    with the select, the row-norm family's conversion on it, the gated chain under two scales (the equal-scale case is the W4A4 chain
    form), GEMM2 under the (row x channel) scale; the chain record names the class in the kernel's cache key."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_fp8 as S8
    from flashinfer.fused_moe.batch_invariant.blocks.fp8_epilogue import Fp8Chain
    from flashinfer.fused_moe.batch_invariant.mterm import TermAlgebra, sexpr

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_fp8_pertoken", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert fp != bi.fingerprint("fused_moe_fp8", **dims) and fp != bi.fingerprint(
        "fused_moe", **dims
    )
    assert (
        bi.fingerprint(
            "fused_moe_fp8_pertoken", method="Renormalize", act="swiglu", **dims
        )
        == fp
    )
    assert bi.fingerprint("fused_moe_fp8_pertoken", act=SwiGLU(), **dims) == fp
    assert (
        bi.fingerprint(
            "fused_moe_fp8_pertoken", method=RoutingMethodType.Sigmoid, **dims
        )
        != fp
    )
    g84 = bi.fingerprint(
        "fused_moe_fp8_pertoken",
        method=RoutingMethodType.DeepSeekV3,
        n_group=8,
        topk_group=4,
        **dims,
    )  # DeepSeekV3's groups are parameters of its routing term: the group shape is a dimension
    assert g84 != bi.fingerprint(
        "fused_moe_fp8_pertoken", method=RoutingMethodType.DeepSeekV3, **dims
    )
    assert g84 == bi.fingerprint(
        "fused_moe_fp8_pertoken", method="sigmoid_renorm_bias_scaled_g8x4", **dims
    )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_fp8_pertoken",
            method=RoutingMethodType.TopK,
            n_group=8,
            topk_group=4,
            **dims,
        )
    fps = {bi.fingerprint("fused_moe_fp8_pertoken", act=a, **dims) for a in ACTIVATIONS}
    assert len(fps) == len(ACTIVATIONS) and fp in fps
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        assert bi.fingerprint(
            "fused_moe_fp8_pertoken", act=cfg, **dims
        ) == S8.spec_fingerprint_fp8_pertoken(
            2048, 768, 8, 128, S.SOFTMAX_RENORM, a.act, a.params, a.gated
        )
    alg = TermAlgebra()
    v, m, amax = alg.leaf("x", 0), alg.sym("m"), alg.sym("amax")
    assert (
        sexpr(S8.pertoken_quant_elem(alg, v, m))
        == sexpr(S8.x_quant_e4m3_elem(alg, v, m))
        == "(cvt e4m3 (mul (sym m) (x 0)))"
    )
    assert (
        sexpr(S8.pertoken_mult(alg, amax))
        == "(div (const 448.0) (nzor (sym amax) (const 448.0)))"
    )
    assert sexpr(S8.pertoken_scale(alg, m)) == sexpr(S8.inv_scale(alg, m))
    red = S8.pertoken_row_reduction(24, lambda fp_, i: fp_.leaf("x", i))
    sx = sexpr(red.canonical(alg))
    assert (
        sx.count("(fabs (x ") == 24 and "(x 24)" not in sx and sx.count("(fmaxc ") == 23
    )
    g, u, sg, su = (
        alg.leaf("g", 0, 0),
        alg.leaf("u", 0, 0),
        alg.sym("sg"),
        alg.sym("su"),
    )
    assert sexpr(S8.glu_scaled2_elem(alg, g, u, sg, su)) != sexpr(
        S8.glu_scaled2_elem(alg, g, u, sg, sg)
    )
    assert sexpr(S8.glu_scaled2_elem(alg, g, u, sg, sg)) == sexpr(
        S.glu_elem(alg, alg.mul(g, sg), alg.mul(u, sg))
    )
    s = S8.Fp8PerTokenMoESpec(64, 32, 2, 8)
    assert (
        s.name == "moe.e4m3.swiglu.pertoken"
        and S8.Fp8PerTokenMoESpec(64, 32, 2, 8, act="relu2", gated=False).name
        == "moe.e4m3.relu2.pertoken"
    )
    sx1 = sexpr(s.gemm1())
    assert (
        "(mul (sw1 0 1) (sym s_x))" in sx1
        and "(mul (sw1 0 0) (sym s_x))" in sx1
        and sx1.startswith("(cvt bf16 ")
    )
    assert sexpr(s.gemm2()) == "(cvt bf16 (mul (acc2 0 0) (mul (pts2 0) (sw2 0 0))))"
    assert sexpr(s.x_quant()).startswith(
        "(cvt e4m3 (mul (div (const 448.0) (nzor (fmaxc "
    )
    # the stable pins of the class
    assert bi.fingerprint("fused_moe_fp8_pertoken", **dims) == "moe-625e1da780d8e5f9"
    assert (
        bi.fingerprint("fused_moe_fp8_pertoken", act=ReLU2(), **dims)
        == "moe-43d82902de132ea4"
    )
    # the chains
    assert (
        Fp8Chain("glu", "silu", "bf16", (), True, True).describe()
        == "glu_silu_pertoken_bf16"
    )
    assert (
        Fp8Chain("plain", "silu", "bf16", (), True, True).describe()
        == "plain_silu_pertoken_bf16"
    )
    with pytest.raises(AssertionError):
        Fp8Chain("glu", "silu", "e4m3", (), True, True)
    with pytest.raises(AssertionError):
        Fp8Chain("glu", "silu", "f32", (), True, True)
    with pytest.raises(AssertionError):
        Fp8Chain("plain", "silu", "bf16", (), False, True)


def test_vectorised_and_exact_chains_agree_on_adversarial_inputs():
    """The vectorised CPU model of the quantizer and the chains equals the exact scalar model on adversarial values (the
    format's range, zero rows, saturating magnitudes, signed zeros), element by element."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant import spec_fp8 as S8

    g = torch.Generator().manual_seed(7)
    x = torch.randn(9, 40, generator=g) * torch.exp2(
        torch.randint(-20, 12, (9, 1), generator=g).float()
    )
    x[2] = 0.0
    x[4, 5] = 1e30
    x[5, 0] = -0.0
    x = x.to(torch.bfloat16)
    q, s = R8.pertoken_quant_cpu(x)
    for r in range(x.shape[0]):
        row = [float(v) for v in x[r].float()]
        amax = 0.0
        for v in row:
            amax = R.ExactF32.fmaxc(amax, R.ExactF32.fabs(v))
        m = S8.pertoken_mult(R.ExactF32, amax)
        assert float(s[r]) == S8.pertoken_scale(R.ExactF32, m), r
        for j, v in enumerate(row):
            assert float(q[r, j]) == S8.pertoken_quant_elem(R.ExactF32, v, m), (r, j)
    acc = torch.randn(9, 16, generator=g) * torch.exp2(
        torch.randint(-6, 6, (9, 1), generator=g).float()
    )
    acc2 = torch.randn(9, 16, generator=g)
    pts = torch.exp2(torch.randint(-8, 8, (9,), generator=g).float())
    sw = torch.exp2(torch.randint(-8, 8, (16,), generator=g).float())
    for act, params in (("silu", ()), ("gelu_tanh", ()), ("swiglu_step", (3.0,))):
        h = R8.pertoken_gemm1_cpu(acc, acc2, pts, sw, sw.flip(0), act, params)
        for r in range(9):
            for j in range(16):
                sg = S8.pertoken_elem_scale(R.ExactF32, float(pts[r]), float(sw[j]))
                su = S8.pertoken_elem_scale(
                    R.ExactF32, float(pts[r]), float(sw.flip(0)[j])
                )
                assert float(h[r, j]) == S8.glu_scaled2_elem(
                    R.ExactF32,
                    float(acc[r, j]),
                    float(acc2[r, j]),
                    sg,
                    su,
                    act,
                    "bf16",
                    params,
                ), (act, r, j)
    y = R8.pertoken_gemm2_cpu(acc, pts, sw)
    for r in range(9):
        for j in range(16):
            assert float(y[r, j]) == S8.scaled_out_elem(
                R.ExactF32,
                float(acc[r, j]),
                S8.pertoken_elem_scale(R.ExactF32, float(pts[r]), float(sw[j])),
                "bf16",
            )


@pytest.mark.parametrize(
    "family,sm_count", [("sm_107", 212), ("sm_100", 148), ("sm_103", 152)]
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for the per-token chains lies in the enumerated record spaces of its GEMM (dense or swapped) at the
    nominal parts; the per-token chains are accepted by both bodies' rules."""
    from flashinfer.fused_moe.batch_invariant.api_fp8 import GEMM2_CHAIN_PERTOKEN
    from flashinfer.fused_moe.batch_invariant.blocks.fp8_epilogue import Fp8Chain
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import (
        all_valid_configs_fp8_both,
        default_configs_fp8,
        swapped_knobs_fp8,
    )

    for H, I, E, k in [
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (512, 256, 64, 6),
    ]:
        ch1 = Fp8Chain("glu", "silu", "bf16", (), True, True)
        ch1u = Fp8Chain("act", "relu2", "bf16", (), True, True)
        s1, s1u, s2 = (
            set(all_valid_configs_fp8_both(2 * I, H, E, ch1)),
            set(all_valid_configs_fp8_both(I, H, E, ch1u)),
            set(all_valid_configs_fp8_both(H, I, E, GEMM2_CHAIN_PERTOKEN)),
        )
        assert s1 and s1u and s2 and set(swapped_knobs_fp8()) <= s2
        for T in (1, 7, 64, 300, 4096):
            c1, c2 = default_configs_fp8(
                T, k, H, I, E, sm_count, family, True, nominal=True, chain1=ch1
            )
            assert c1 in s1 and c2 in s2, (
                family,
                T,
                H,
                I,
                E,
                k,
                c1.describe(),
                c2.describe(),
            )
            c1u, _ = default_configs_fp8(
                T, k, H, I, E, sm_count, family, False, nominal=True, chain1=ch1u
            )
            assert c1u in s1u


# ----------------------------------------------------------------------------- device


def _reference_route(logits: torch.Tensor, k: int, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _reference_ffn(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    view: dict,
    act,
    x_scale=None,
):
    """The class composed on the device from its own pieces: the per-token quantization by the CPU chain (or the caller's ``(x,
    x_scale)``), the class's permutation (torch indexing), GEMM1 as the grouped fp8 kernel with the unscaled fp32 identity epilogue
    (the same fold), the CPU chain of the activation under the (token x channel) scales, the CPU requantization per row, GEMM2 as the
    dense batch-invariant fp8 GEMM per expert (unit scales, fp32 out), the CPU GEMM2 chain, the CPU slot-order combine. Bit for bit
    what the fused op must produce."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant.api import permute_plan
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3PerChannel,
        grouped_gemm_fp8,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.fp8_epilogue import Fp8Chain
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import all_valid_configs_fp8
    from flashinfer.gemm.batch_invariant.api import mm_fp8_batch_invariant
    from flashinfer.gemm.batch_invariant.schedule import chunks_for

    W = MoEWeightsE4M3PerChannel.from_view(view)
    T, H = x.shape
    E, k = W.num_experts, ids.shape[1]
    I = W.intermediate_size
    assert chunks_for(H, I, "e4m3") == 1
    if x_scale is None:
        q_vals, s_x = R8.pertoken_quant_cpu(x.cpu())
        xq = R8.e4m3_bytes(q_vals).to(x.device)
    else:
        xq, s_x = x, x_scale.cpu()
    counts, dest, src = permute_plan(ids, E)
    n = T * k
    tok = (src // k).long()
    x_perm = xq[tok].contiguous()
    pts1 = s_x[tok.cpu()]
    raw = Fp8Chain("plain", "silu", "f32", (), False)
    one = torch.ones(1, dtype=torch.float32, device=x.device)
    acc = grouped_gemm_fp8(
        x_perm,
        W.w1q,
        torch.ones(E, device=x.device),
        one,
        one,
        counts,
        all_valid_configs_fp8(W.w1q.shape[1], H, E, raw)[0],
        raw,
    )
    off = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cpu().long().cumsum(0)])
    h = torch.zeros(n, I, dtype=torch.bfloat16)
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            h[lo:hi] = R8.pertoken_chain_cpu(
                acc[lo:hi].cpu(),
                pts1[lo:hi],
                W.s_w1[e].cpu(),
                act.act,
                act.params,
                act.gated,
            )
    aq_vals, pts2 = R8.pertoken_quant_cpu(h)
    a = R8.e4m3_bytes(aq_vals).to(x.device)
    acc2 = torch.empty(n, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            acc2[lo:hi] = mm_fp8_batch_invariant(
                a[lo:hi].contiguous(), W.w2q[e].t(), one, one, out_dtype=torch.float32
            )
    y_perm = torch.zeros(n, H, dtype=torch.bfloat16)
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            y_perm[lo:hi] = R8.pertoken_gemm2_cpu(
                acc2[lo:hi].cpu(), pts2[lo:hi], W.s_w2[e].cpu()
            )
    y = y_perm[dest.view(-1).long().cpu()]
    ident = torch.arange(n, dtype=torch.int32).view(T, k)
    return R.combine_cpu(y, ident, weights.cpu()), dict(
        x_perm=x_perm, pts1=pts1, h=h, a=a, pts2=pts2, counts=counts, dest=dest
    )


@needs_device
@pytest.mark.parametrize("K", [64, 768, 2048, 7168])
def test_quantizer_kernel_equals_the_cpu_chain(K):
    """The class's quantizer == the CPU chain bit for bit (codes and scales) on adversarial rows incl. a zero row and e4m3-saturating
    magnitudes, at the rows themselves and scattered to the tokens' dense rows with dropped slots (nothing stored, the other rows
    untouched); the torch model of the chain gives the same bits."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        quantize_e4m3_pertoken,
        quantize_e4m3_pertoken_kernel,
    )

    M = 37
    g = torch.Generator().manual_seed(K)
    x = (
        (
            torch.randn(M, K, generator=g)
            * torch.exp2(torch.randint(-20, 12, (M, 1), generator=g).float())
        )
        .to(torch.bfloat16)
        .cuda()
    )
    x[3] = 0.0
    x[5] = torch.where(torch.arange(K, device="cuda") % 3 == 0, 1e30, 2.0).to(
        torch.bfloat16
    )
    q_ref, s_ref = R8.pertoken_quant_cpu(x.cpu())
    q = torch.full((M, K), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    s = torch.full((M,), float("nan"), device="cuda")
    quantize_e4m3_pertoken_kernel(x, q, s)
    assert torch.equal(bits(q.cpu()), bits(R8.e4m3_bytes(q_ref))) and torch.equal(
        bits(s.cpu()), bits(s_ref)
    )
    qt, st = quantize_e4m3_pertoken(x)
    assert torch.equal(bits(qt), bits(q)) and torch.equal(bits(st.cpu()), bits(s_ref))
    k = 3
    dest = torch.randperm(M * k, generator=g).to(torch.int32)
    dest[::5] = -1
    R_rows = M * k + 16
    q2 = torch.full((R_rows, K), 0xFF, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    s2 = torch.full((R_rows,), -1.0, device="cuda")
    quantize_e4m3_pertoken_kernel(x, q2, s2, dest.cuda(), k)
    exp_q = torch.full((R_rows, K), 0xFF, dtype=torch.uint8)
    exp_s = torch.full((R_rows,), -1.0)
    qb = bits(R8.e4m3_bytes(q_ref))
    for i in range(M * k):
        if dest[i] >= 0:
            exp_q[dest[i]] = qb[i // k]
            exp_s[dest[i]] = s_ref[i // k]
    assert torch.equal(bits(q2.cpu()), exp_q) and torch.equal(
        bits(s2.cpu()), bits(exp_s)
    )


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits, bf16 activations quantized by the class) == the class's own routing on the CPU, the per-token
    quantization, the activation's chain under the (token x channel) scales on the raw grouped accumulators, the requantization,
    the dense batch-invariant fp8 GEMM per expert and the CPU combine: the oracle ladder, bit for bit, for every activation; the
    op's intermediates equal the CPU pieces; the caller's per-token quantization (``prepare_activations``) gives the same bits."""
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3PerChannel,
        fused_moe_fp8_pertoken,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    W = _weights(w1, w2, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(x, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    view = W.get_view(KEY)
    ref, pieces = _reference_ffn(x, ids.cuda(), weights.cuda(), view, chain)
    assert torch.equal(bits(out.cpu()), bits(ref))
    _, im = fused_moe_fp8_pertoken(
        x,
        ids.cuda(),
        weights.cuda(),
        MoEWeightsE4M3PerChannel.from_view(view),
        return_intermediates=True,
    )
    assert torch.equal(
        bits(im["x_perm"].cpu()), bits(pieces["x_perm"].cpu())
    ) and torch.equal(bits(im["pts1"].cpu()), bits(pieces["pts1"]))
    assert torch.equal(bits(im["h"].cpu()), bits(pieces["h"])) and torch.equal(
        bits(im["a"].cpu()), bits(pieces["a"].cpu())
    )
    assert torch.equal(bits(im["pts2"].cpu()), bits(pieces["pts2"]))
    xq, sx = BatchInvariantFp8PerTokenConfig.prepare_activations(x)
    assert torch.equal(
        bits(layer(_logits_pack(xq, logits, scale=sx), W).cpu()), bits(ref)
    )


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their inputs;
    repeated runs and a CUDA-graph replay reproduce them."""
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    out_l = layer(_logits_pack(x, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack(x, ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    w16 = weights.to(torch.bfloat16)
    out_u16 = layer(_routed_pack(x, ids.cuda(), w16.cuda()), W)
    out_p = layer(_routed_pack(x, ids.cuda(), w16.cuda(), packed=True), W)
    assert torch.equal(bits(out_u16), bits(out_p))
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack(x, logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack(x, logits)
    with torch.cuda.stream(stream):
        layer(pack, W)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_g = layer(pack, W)
    torch.cuda.synchronize()
    for _ in range(2):
        out_g.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out_g), bits(out_l))


@needs_device
def test_grouped_deepseekv3_from_logits_matches_the_pre_routed_path():
    """DeepSeekV3's group-limited selection from logits (the groups are parameters of the routing term, derived by the
    per-tensor runner's shared function): the layer's bits equal the pre-routed path on the exact CPU model's ids and
    weights, and repeat; expert groups with another method are refused from logits and never consulted with pre-routed
    ids (the bits of the ungrouped layer on the same ids and weights)."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=7, adversarial=True)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(8)) * 0.1)
        .float()
        .cuda()
    )
    W = _weights(w1, w2)

    def grouped_layer(method, n_group, topk_group, **kw):
        return MoELayer(
            MoEConfig(
                routing=RoutingConfig(
                    num_experts=E,
                    top_k=k,
                    method=method,
                    n_group=n_group,
                    topk_group=topk_group,
                    **kw,
                ),
                quant=QuantConfig(
                    weight=QuantFormat.FP8PerChannel,
                    activation=QuantFormat.FP8PerToken,
                ),
                experts=ExpertConfig(intermediate_size=I),
                activation=_SWIGLU,
                backend=BackendOptions((BatchInvariantFp8PerTokenConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )

    for groups in ((8, 4), (4, 1)):
        rm = S.METHOD_OF_FLASHINFER[RoutingMethodType.DeepSeekV3].with_groups(*groups)
        layer = grouped_layer(
            RoutingMethodType.DeepSeekV3, *groups, routed_scaling_factor=1.75
        )
        out = layer(_logits_pack(x, logits, bias=bias), W)
        ex, w = R.route_cpu(logits.cpu(), k, rm, bias.cpu(), 1.75)
        assert torch.equal(
            bits(out), bits(layer(_routed_pack(x, ex.cuda(), w.cuda()), W))
        ), groups
        assert torch.equal(
            bits(layer(_logits_pack(x, logits, bias=bias), W)), bits(out)
        ), groups
    grouped = grouped_layer(RoutingMethodType.Renormalize, 2, 1)
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        grouped(_logits_pack(x, logits), W)
    assert "n_group" in str(excinfo.value)
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    plain = _layer(k, E, I, tune_max=T)
    assert torch.equal(
        bits(grouped(_routed_pack(x, ids.cuda(), wts.cuda()), W)),
        bits(plain(_routed_pack(x, ids.cuda(), wts.cuda()), W)),
    )


@needs_device
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens create."""
    T, k, H, I, E = 500, 8, 512, 256, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    full = bits(layer(_logits_pack(x, logits), W))
    for n in (1, 2, 7, 64, 200, 333):
        assert torch.equal(bits(layer(_logits_pack(x[:n], logits[:n]), W)), full[:n]), n
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(4)).cuda()
    assert torch.equal(
        bits(layer(_logits_pack(x[perm].contiguous(), logits[perm].contiguous()), W)),
        full[perm],
    )
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    base = bits(layer(_routed_pack(x, ids, weights), W))
    g = torch.Generator().manual_seed(5)
    x_extra = (torch.randn(556, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    w_extra = torch.rand(556, k, generator=g).cuda()
    for t in (0, T // 2, T - 1):
        hot = ids[t : t + 1].expand(556, k).contiguous()
        out = layer(
            _routed_pack(
                torch.cat([x[t : t + 1], x_extra]),
                torch.cat([ids[t : t + 1], hot]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("hot group", t)
        mine = set(ids[t].tolist())
        others = (
            torch.tensor(
                [e for e in range(E) if e not in mine][:k],
                dtype=torch.int32,
                device="cuda",
            )
            .view(1, k)
            .expand(556, k)
            .contiguous()
        )
        out = layer(
            _routed_pack(
                torch.cat([x[t : t + 1], x_extra]),
                torch.cat([ids[t : t + 1], others]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("unrelated hot experts", t)


@needs_device
def test_every_tactic_computes_the_class_and_swapped_records_equal_the_grouped():
    """Every tactic the autotuner may pick gives the same bits, tuning keeps them; at a decode-sized batch the tactics include the
    swapped records and every record of the swapped space gives the bits of the class's CPU composition."""
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3PerChannel,
        fused_moe_fp8_pertoken,
    )
    from flashinfer.fused_moe.batch_invariant.runner import (
        _configs_from_tactic,
        _tactic,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import swapped_knobs_fp8
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=6, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantFp8PerTokenRunner)
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert 1 <= len(tactics) <= 9 and len(set(tactics)) == len(tactics)
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        ), t
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    AutoTuner.get().clear_cache()
    xd, ld, _, _ = _problem(3, k, H, I, E, seed=8, adversarial=True)
    packd = _logits_pack(xd, ld)
    ids, weights = _reference_route(ld, k, RoutingMethodType.Renormalize)
    view = W.get_view(KEY)
    refd, _ = _reference_ffn(
        xd, ids.cuda(), weights.cuda(), view, activation_of(_SWIGLU)
    )
    assert torch.equal(bits(layer(packd, W).cpu()), bits(refd))
    tactics_d = runner.get_valid_tactics(runner.pack_inputs(packd, W), None)
    assert any(_configs_from_tactic(t)[0].swap_ab for t in tactics_d), tactics_d
    for t in tactics_d:
        c1, c2 = _configs_from_tactic(t)
        assert _tactic(c1, c2) == t
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(packd, W), tactic=t).cpu()),
            bits(refd),
        ), t
    Wv = MoEWeightsE4M3PerChannel.from_view(view)
    for rec in swapped_knobs_fp8():
        out = fused_moe_fp8_pertoken(
            xd, ids.cuda(), weights.cuda(), Wv, config1=rec, config2=rec
        )
        assert torch.equal(bits(out.cpu()), bits(refd)), rec.describe()


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: e4m3 activations without the per-token scale tensor, bf16 activations with one, a scale of the wrong
    shape, weights prepared for another activation; a zero-token call returns an empty output."""
    k, H, I, E = 6, 512, 256, 64
    errors = (
        bi.BatchInvariantUnsupportedError,
        RuntimeError,
        NotImplementedError,
        ValueError,
        TypeError,
    )
    x, logits, w1, w2 = _problem(16, k, H, I, E, seed=12)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    xq, sx = BatchInvariantFp8PerTokenConfig.prepare_activations(x)
    with pytest.raises(errors):
        layer(_logits_pack(xq, logits), W)
    with pytest.raises(errors):
        layer(_logits_pack(x, logits, scale=sx), W)
    with pytest.raises(errors):
        layer(_logits_pack(xq, logits, scale=sx[:8]), W)
    with pytest.raises(errors):
        layer(_logits_pack(x, logits), _weights(w1, w2, GeGLU()))
    out0 = layer(_logits_pack(x[:0], logits[:0]), W)
    assert tuple(out0.shape) == (0, H)


def _flashinfer_per_channel_reference(
    x, w1, w2, ids, weights, act, activation_fn=None
) -> torch.Tensor:
    """FlashInfer's torch reference of the per-token / per-channel fp8 MoE: the recipe of ``tests/moe/trtllm_gen_fused_moe_utils.py``
    (``quant_fp8_per_channel`` for the experts: ``448 / amax`` per output channel, the activations per token the same way; the
    dequantized fp32 operands; fp32 GEMM1; the typed activation in fp32 by the layer's own formulas; the intermediate requantized per
    token; bf16 expert outputs; the fp32 routed sum). Independent of the class (its exact model is :mod:`spec_fp8`; the bitwise tests
    above are the promise). ``activation_fn`` replaces the activation (the negative control of the tolerance)."""
    from tests.moe.utils import compute_reference_activation, powlu_ref

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weights = weights.to(torch.bfloat16).float()

    def qdq_rows(w):
        amax = w.float().abs().amax(dim=-1).clamp(min=1e-12)
        s = fp8_max / amax
        return (w.float() * s.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(
            torch.float8_e4m3fn
        ).float() / s.unsqueeze(-1)

    x_deq, w1_deq, w2_deq = qdq_rows(x), qdq_rows(w1), qdq_rows(w2)
    I = w2.shape[2]
    if activation_fn is None:

        def activation_fn(gemm1):
            if isinstance(act, PowLU):
                up, gate = gemm1.split(I, dim=-1)
                gg = powlu_ref(gate, act.m)
                if act.limit is not None:
                    gg = gg.clamp_max(act.limit)
                    up = up.clamp(-act.limit, act.limit)
                return gg * up
            return compute_reference_activation(gemm1, act, I, out_dtype=torch.float32)

    out = torch.zeros_like(x_deq)
    for e in range(w1.shape[0]):
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        inter = activation_fn(x_deq[tok] @ w1_deq[e].t())
        y = (qdq_rows(inter) @ w2_deq[e].t()).to(torch.bfloat16).float()
        out[tok] += weights[tok, slot, None] * y
    return out


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_plausible_against_flashinfers_per_channel_torch_reference(act):
    """An observation on moderate inputs under FlashInfer's own discipline for the fp8 pairs (``check_accuracy`` at atol 0.05 / rtol
    0.3 on 99 % of the outputs against the torch reference above, and the negative control that the tolerance rejects a wrong
    activation formula). Not a promise of the class (the exact model is; tested bit for bit above), and never a comparison with
    another backend's output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    T, k, H, I, E = 128, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11, gated=activation_of(act).gated)
    out = _layer(k, E, I, act)(_logits_pack(x, logits), _weights(w1, w2, act)).float()
    vals, ids = torch.topk(logits, k, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    ref = _flashinfer_per_channel_reference(x, w1, w2, ids, weights, act)
    check_accuracy(out, ref, atol=0.05, rtol=0.3, percent=0.99)
    wrong_fn = (
        (lambda g1: g1[:, :I])
        if activation_of(act).gated
        else ((lambda g1: g1) if isinstance(act, ReLU) else torch.relu)
    )
    wrong = _flashinfer_per_channel_reference(
        x, w1, w2, ids, weights, act, activation_fn=wrong_fn
    )
    if not isinstance(act, ReLU):
        with pytest.raises(Exception, match="Mismatch percentage"):
            check_accuracy(out, wrong, atol=0.05, rtol=0.3, percent=0.99)
