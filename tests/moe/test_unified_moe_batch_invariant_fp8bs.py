# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeekFp8 x DeepSeekFp8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(DeepSeekFp8, DeepSeekFp8)``, the backend grouping the fp8 formats
with block scales under one key as TRTLLM-Gen's ``TrtllmFp8BlockConfig`` does): the DeepSeek-style classes
``moe.e4m3bs.<act>``, one per typed activation of the layer and its scalars, on the surface of the
TRTLLM-Gen block-fp8 backend under ``QuantConfig(DeepSeekFp8, DeepSeekFp8)`` (e4m3 activations with fp32 block scales per
1x128 block, the weights with fp32 block scales per 128x128 block, every routing input mode).

The class has an exact CPU model piece by piece (the routing, both quantizations by the ``fp8bs.quant`` class, the dense
block-scale class's fold per expert -- the device's block partials promoted in exact fp32 arithmetic --, the activation
chain, GEMM2's conversion, the slot-order combine), so the tests are bit comparisons: the layer's output against the class
composed from its own pieces for several activations, the fused and the standalone requantization giving one output,
run-to-run determinism including CUDA-graph replay, batch / permutation / sub-batch invariance, agreement of every tactic the
autotuner may pick (both requantization spellings), the three routing input modes giving the same bits, the largest expert
count, alternating inputs under programmatic dependent launch, the error contract, and a plausibility observation against
FlashInfer's torch reference of the DeepSeek block-fp8 MoE. The device-free tests pin the registration, the fingerprint
dimensions, the quantizer's vectorised and exact models, the weight view (FlashInfer's DeepSeek view taken as given) and the
dispatcher's membership in the enumerated space at the nominal parts.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    BatchInvariantFp8BlockConfig,
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
    TrtllmFp8BlockConfig,
)
from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND
from flashinfer.fused_moe.batch_invariant.runner import BatchInvariantMoeRunner
from flashinfer.fused_moe.batch_invariant.runner_fp8_block import (
    BatchInvariantFp8BlockRunner,
)
from flashinfer.fused_moe.batch_invariant.api_fp8bs import MOE_FP8BS_ARCHS
from flashinfer.fused_moe.batch_invariant.runner_fp8bs import (
    BatchInvariantDeepSeekFp8Runner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_FP8BS_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
QUANT = QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8)
KEY = BatchInvariantFp8BlockRunner.backend_key
#: The activations the suites cover (the bf16 backend's list): every ``ActivationConfig`` class of the layer with its defaults,
#: and the parameterised ones once more with scalars that move every branch.
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
BLOCK = 128


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    )


def _same_bits_or_both_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal bits on every lane, except that a NaN lane matches a NaN lane of any payload (a NaN's payload is the part's
    canonical, not a value of the class)."""
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        bits(a)[~torch.isnan(a)], bits(b)[~torch.isnan(b)]
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
    """bf16 activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated).
    Adversarial: wide per-row and per-block exponent spreads (the block scales span the fp32 range the recipe uses),
    zero blocks (the clamped scale) and a saturating block."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        x = torch.randn(T, H, generator=g) * torch.exp2(
            torch.randint(-8, 6, (T, 1), generator=g).float()
        )
        x = x * torch.exp2(
            torch.randint(-6, 6, (1, H // BLOCK), generator=g).float()
        ).repeat_interleave(BLOCK, 1)
        x[:, 5::97] = 0.0
        if T > 2:
            x[2, :BLOCK] = 0.0
        logits = (torch.randn(T, E, generator=g) * 2).float()
        logits[0, : min(k, E)] = 1.0
        span = torch.exp2(torch.randint(-3, 3, (E, 1, 1), generator=g).float())
    else:
        x = torch.randn(T, H, generator=g) * 0.5
        logits = (torch.randn(T, E, generator=g) * 2).float()
        span = 1.0
    x = x.to(torch.bfloat16)
    n1 = 2 * I if gated else I
    w1 = (torch.randn(E, n1, H, generator=g) * 0.05 * span).to(torch.bfloat16)
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
        quant=QUANT,
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(BatchInvariantFp8BlockRunner.backend_key, view)
    return pack


def _xq(x: torch.Tensor):
    xq, sx = BatchInvariantFp8BlockConfig.prepare_activations(x, quant=QUANT)
    assert (
        xq.dtype == torch.float8_e4m3fn
        and sx.dtype == torch.float32
        and tuple(sx.shape) == (x.shape[1] // BLOCK, x.shape[0])
    )
    return xq, sx


def _logits_pack(xq, sx, logits: torch.Tensor, bias=None) -> MoEActivationPack:
    return MoEActivationPack(
        xq,
        sx,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(
    xq, sx, ids: torch.Tensor, weights: torch.Tensor, packed: bool = False
) -> MoEActivationPack:
    if packed:
        return MoEActivationPack(
            xq,
            sx,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        xq,
        sx,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The backend's registration, modes, activation classes and default-candidate rule are the MXFP8 suite's; here the pair's
    own facts: the backend serves both block-scaled fp8 pairs and constructs this pair's runner for the DeepSeek pair, which
    admits that pair alone; the kernel-class registry names the class."""
    assert BatchInvariantFp8BlockConfig in ALL_BACKEND_CONFIGS
    assert (
        _BACKEND_RUNNERS[BatchInvariantFp8BlockConfig] is BatchInvariantFp8BlockRunner
    )
    assert not any(
        isinstance(c, BatchInvariantFp8BlockConfig) for c in _DEFAULT_BACKEND.candidates
    )
    assert issubclass(BatchInvariantDeepSeekFp8Runner, BatchInvariantFp8BlockRunner)
    assert BatchInvariantFp8BlockRunner.supported_quant_variants == (
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
    )
    assert BatchInvariantDeepSeekFp8Runner.supported_quant_variants == (QUANT.pair,)
    assert (
        BatchInvariantFp8BlockRunner.implementation(QUANT.pair)
        is BatchInvariantDeepSeekFp8Runner
    )
    assert BatchInvariantFp8BlockRunner.supports_quant(QUANT)
    assert not BatchInvariantFp8BlockRunner.supports_quant(QuantConfig())
    assert (
        set(BatchInvariantDeepSeekFp8Runner.supported_activation_classes)
        == set(BatchInvariantMoeRunner.supported_activation_classes)
        == {
            SwiGLU,
            SwiGLUStep,
            GeGLU,
            GeGLUTanh,
            SiTU,
            PowLU,
            ReLU2,
            Identity,
            GELU,
            ReLU,
            SiLU,
        }
    )
    assert (
        "fused_moe_fp8bs" in bi.CLASSES
        and bi.class_name("fused_moe_fp8bs") == "moe.e4m3bs"
    )


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_fp8bs", ...)`` takes the geometry, the routing method and the activation; it differs from the bf16
    class's at the same dimensions; every listed activation is its own class; the routing method is a dimension; the block
    scales' values are not; the quantizer class's fingerprint has no dimension."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_fp8bs as SB

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_fp8bs", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert fp != bi.fingerprint("fused_moe", **dims)
    assert (
        bi.fingerprint("fused_moe_fp8bs", method="Renormalize", act="swiglu", **dims)
        == fp
    )
    assert bi.fingerprint("fused_moe_fp8bs", act=SwiGLU(), **dims) == fp
    assert (
        bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.Sigmoid, **dims)
        != fp
    )
    assert (
        bi.fingerprint("fused_moe_fp8bs", H=2048, I=768, top_k=8, num_experts=64) != fp
    )
    assert (
        bi.fingerprint("fused_moe_fp8bs", H=2048, I=640, top_k=8, num_experts=128) != fp
    )
    fps = {bi.fingerprint("fused_moe_fp8bs", act=a, **dims) for a in ACTIVATIONS}
    assert len(fps) == len(ACTIVATIONS) and fp in fps
    assert bi.fingerprint("fused_moe_fp8bs", act="relu2", **dims) == bi.fingerprint(
        "fused_moe_fp8bs", act=ReLU2(), **dims
    )
    # the routing terms the bf16 model serves are classes of this backend too (Llama4 = the top-1 TopKSigmoid term, its own
    # class at top_k = 1; another top_k is refused by the runners' rule); a method the classes do not implement is refused
    dims1 = dict(dims, top_k=1)
    assert bi.fingerprint(
        "fused_moe_fp8bs", method=RoutingMethodType.Llama4, **dims1
    ) != bi.fingerprint("fused_moe_fp8bs", **dims1)
    with pytest.raises(ValueError, match="Llama4 routing takes top_k = 1"):
        bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.Llama4, **dims)
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.MiniMax2, **dims)
    with pytest.raises(TypeError):
        bi.fingerprint("fused_moe_fp8bs", block=128, **dims)
    # the group-limited selection is a parameter of the DeepSeekV3 term: a dimension of the class, refused with another method
    ds = bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.DeepSeekV3, **dims)
    g84 = bi.fingerprint(
        "fused_moe_fp8bs",
        method=RoutingMethodType.DeepSeekV3,
        n_group=8,
        topk_group=4,
        **dims,
    )
    assert g84 != ds and g84 == bi.fingerprint(
        "fused_moe_fp8bs", method="sigmoid_renorm_bias_scaled_g8x4", **dims
    )
    assert g84 != bi.fingerprint(
        "fused_moe_fp8bs",
        method=RoutingMethodType.DeepSeekV3,
        n_group=4,
        topk_group=2,
        **dims,
    )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_fp8bs",
            method=RoutingMethodType.Renormalize,
            n_group=8,
            topk_group=4,
            **dims,
        )
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        assert bi.fingerprint(
            "fused_moe_fp8bs", act=cfg, **dims
        ) == SB.spec_fingerprint_fp8bs(
            2048, 768, 8, 128, S.SOFTMAX_RENORM, a.act, a.params, a.gated
        )
    s = SB.Fp8BsMoESpec(256, 128, 2, 8)
    assert (
        s.name == "moe.e4m3bs.swiglu"
        and SB.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False).name
        == "moe.e4m3bs.relu2"
    )
    from flashinfer.fused_moe.batch_invariant import mterm

    assert mterm.sexpr(s.gemm1()) == mterm.sexpr(
        S.MoESpec(256, 128, 2, 8, alg=s.alg).gemm1()
    )  # the bf16 chain, no scale symbol
    assert mterm.sexpr(s.gemm2()) == "(cvt bf16 (acc2 0 0))"
    q = SB.quant_fingerprint()
    assert q.startswith("fp8bsq-") and len(q) == 23 and q == SB.quant_fingerprint()


def _adversarial_rows(M: int, K: int, seed: int) -> torch.Tensor:
    """bf16 rows for the quantizer: wide exponent spreads inside one block (the block scale saturates the small values to zero),
    blocks at the e4m3 maximum, signed zeros, subnormals, a zero block, special values."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g) * torch.exp2(
        torch.randint(-20, 20, (M, K), generator=g).float()
    )
    x[:, ::37] *= 2.0**30
    x[:, 5::64] = 0.0
    x[:, 6::64] = -0.0
    x[:, 7::64] = torch.rand(M, len(range(7, K, 64)), generator=g) * 2.0**-130
    if M > 1:
        x[1, :BLOCK] = 0.0
    x = x.to(torch.bfloat16)
    x[0, :16] = torch.tensor(
        [
            0.0,
            -0.0,
            448.0,
            -448.0,
            500.0,
            -1e30,
            2.0**-10,
            2.0**-12,
            1e-38,
            0.17,
            0.1875,
            0.203125,
            0.21875,
            0.1953125,
            1.0,
            2.0,
        ]
    ).to(torch.bfloat16)
    return x


def test_quantizer_vectorised_model_equals_exact_model_and_weight_reference():
    """The class's vectorised CPU reference (``reference_fp8bs``) is the exact scalar model block for block on adversarial rows
    (both dtypes, both scale layouts); the weight reference is the row chain per 128x128 block; FlashInfer's DeepSeek recipe
    gives the same scales and codes within an e4m3 rounding (it divides, the class multiplies by the reciprocal)."""
    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_activations

    for dtype in (torch.bfloat16, torch.float16):
        x = _adversarial_rows(6, 512, 3).to(dtype)
        if dtype == torch.float16:
            x = x.float().clamp(-60000.0, 60000.0).to(dtype)
        codes_k, s_k = RB.quantize_rows_reference(x, "K")
        codes_mn, s_mn = RB.quantize_rows_reference(x, "MN")
        assert (
            torch.equal(codes_k, codes_mn)
            and torch.equal(s_k, s_mn.t())
            and s_k.shape == (6, 4)
        )
        for r in range(6):
            for b in range(4):
                s_e, _, codes_e = RB.exact_block(
                    x[r, b * BLOCK : (b + 1) * BLOCK].float().tolist()
                )
                assert s_e == s_k[r, b].item(), (dtype, r, b)
                assert codes_e == codes_k[r, b * BLOCK : (b + 1) * BLOCK].tolist(), (
                    dtype,
                    r,
                    b,
                )
        assert (
            s_k[1, 0].item() == fp32.to_f32(1e-12) and (codes_k[1, :BLOCK] == 0).all()
        )  # the zero block
        assert codes_k.abs().max().item() <= 448.0
        by, _ = RB.quantize_rows_bytes(x, "K")
        assert torch.equal(by, RB.e4m3_bytes(codes_k)) and by.dtype == torch.uint8
    x = (torch.randn(64, 512, generator=torch.Generator().manual_seed(7)) * 0.5).to(
        torch.bfloat16
    )
    q_fi, s_fi = _deepseek_fp8_quantize_activations(x)
    codes, s = RB.quantize_rows_reference(x, "MN")
    assert torch.equal(bits(s), bits(s_fi.cpu())), (
        "the same block scales as FlashInfer's recipe"
    )
    off = (codes != q_fi.cpu().float()).float().mean().item()
    assert off < 0.01, (
        off
    )  # a few elements per block round differently under the division
    w = _adversarial_rows(256, 384, 9)
    codes, s = RB.quantize_weight_reference(w)
    assert codes.shape == (256, 384) and s.shape == (2, 3)
    wf = w.float()
    for nb in range(2):
        for kb in range(3):
            blk = wf[nb * 128 : (nb + 1) * 128, kb * 128 : (kb + 1) * 128]
            s_e = fp32.fmaxc(
                fp32.div(blk.abs().max().item(), 448.0), fp32.to_f32(1e-12)
            )
            inv_e = fp32.div(1.0, s_e)
            assert s_e == s[nb, kb].item(), (nb, kb)
            for i, j in [(0, 0), (127, 127), (5, 37), (64, 1)]:
                assert codes[nb * 128 + i, kb * 128 + j].item() == fp32.to_fp8(
                    fp32.mul(blk[i, j].item(), inv_e), "e4m3"
                ), (nb, kb, i, j)


@needs_device
def test_weight_view_and_the_trtllm_view_taken_as_given():
    """``prepare_weights`` quantizes per expert per 128x128 block by the class's reference into the pack on the device,
    interleaves the gated weight's quantized rows and keeps the canonical rows' scales; the ungated weight is taken as given;
    the acceptance (H, I multiples of 128) raises the mode's error; ``from_trtllm_view`` consumes FlashInfer's DeepSeek weight
    view as given (its CPU tensors moved one expert at a time onto ``device``)."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        check_shape_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    g = torch.Generator().manual_seed(3)
    E, I, H = 3, 128, 256
    w1 = (
        torch.randn(E, 2 * I, H, generator=g)
        * torch.exp2(torch.randint(-6, 7, (E, 2 * I, H), generator=g).float())
    ).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    w1[1, :128, :128] = 0.0
    dev = torch.device("cuda")
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1, w2, quant=QUANT, activation=SwiGLU(), device=dev
    )
    W = MoEWeightsE4M3BS.from_view(view)
    q1 = torch.stack(
        [RB.e4m3_bytes(RB.quantize_weight_reference(w1[e])[0]) for e in range(E)]
    )
    s1 = torch.stack([RB.quantize_weight_reference(w1[e])[1] for e in range(E)])
    assert torch.equal(
        W.w1.view(torch.uint8).cpu(), interleave_fc1(q1)
    ) and torch.equal(bits(W.w1_sf.cpu()), bits(s1))
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w2[e])
        assert torch.equal(
            W.w2[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(W.w2_sf[e].cpu()), bits(rs)), e
    assert (
        W.gated
        and W.act == "silu"
        and W.params == ()
        and (W.num_experts, W.hidden_size, W.intermediate_size) == (E, H, I)
    )
    Wu = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1[:, :I].contiguous(), w2, quant=QUANT, activation=ReLU2(), device=dev
        )
    )
    assert not Wu.gated and Wu.act == "relu2" and Wu.w1.shape == (E, I, H)
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w1[e, :I])
        assert torch.equal(
            Wu.w1[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(Wu.w1_sf[e].cpu()), bits(rs)), e
    a = activation_of(SwiGLU(alpha=1.702, beta=1.0, limit=7.0))
    Wp = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=QUANT,
            activation=SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
            device=dev,
        )
    )
    assert (Wp.act, Wp.params) == (a.act, a.params)
    # FlashInfer's DeepSeek weight view: the canonical rows' block quantization taken as given, interleaved here
    fi_view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        activation=SwiGLU(),
        device=torch.device("cpu"),
    )
    Wg = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.from_trtllm_view(
            fi_view, quant=QUANT, activation=SwiGLU(), device=dev
        )
    )
    assert torch.equal(
        Wg.w1.view(torch.uint8).cpu(),
        interleave_fc1(fi_view["gemm1_weights"].view(torch.uint8)),
    )
    assert torch.equal(
        bits(Wg.w1_sf.cpu()), bits(fi_view["gemm1_weights_scale"].float())
    ) and torch.equal(
        Wg.w2.view(torch.uint8).cpu(), fi_view["gemm2_weights"].view(torch.uint8)
    )
    assert torch.equal(bits(W.w1_sf.cpu()), bits(Wg.w1_sf.cpu())), (
        "the same block scales as FlashInfer's recipe"
    )
    for bad in [
        (w1[:, :, :192].contiguous(), w2[:, :192].contiguous()),
        (w1[:, :192].contiguous(), w2[:, :, :96].contiguous()),
    ]:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            MoEWeightsE4M3BS.prepare(*bad, device=dev)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        check_shape_fp8bs(256, 128, 1025)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        check_shape_fp8bs(256, 128, 8, 9)
    check_shape_fp8bs(7168, 2048, 1024, 16)


def test_swapped_body_builds_for_the_block_scale_chains_and_the_kernels_trigger_at_the_shared_sites():
    """The block-scale fp8 chains run the backend's swapped body (:mod:`blocks.swap`) on the promoted operand block at both
    block sizes; the gather form (GEMM1's token rows and their scales through the plan's map, its own producer) takes the bf16
    / f32 chains and refuses the fused requantization (whose code stores read the box form's permuted positions); the
    block-scaled k-block is not this format's. The trigger placement is the shared module constant (:mod:`pdl`: the
    persistent bodies' ``TRIGGER``, the one-wave kernels' ``TRIGGER_SMALL``, the dependents launching at completion): the
    standalone quantizer and the row + scale gather trigger at the one-wave site right after their wait, the plan + quantizer
    launch waits and never triggers (the plan's rule), the promoted producer loops trigger at the shared late site, and no
    kernel of the class carries a trigger knob of its own."""
    import inspect

    from flashinfer.fused_moe.batch_invariant import pdl
    from flashinfer.fused_moe.batch_invariant.blocks import epilogue_fp8bs, fp8bs_small
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import (
        GEMM2_CHAIN,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.swap import SwappedGroupedGemm
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import swapped_knobs_fp8bs

    cfg = swapped_knobs_fp8bs()[0]
    SwappedGroupedGemm(cfg, GEMM2_CHAIN)
    SwappedGroupedGemm(cfg, GEMM2_CHAIN.at(32))
    assert SwappedGroupedGemm(cfg, GEMM2_CHAIN, gather=True).gather
    assert SwappedGroupedGemm(cfg, gemm1_chain("silu"), gather=True).gather
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(cfg, gemm1_chain("silu", fused=True), gather=True)
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(cfg, GEMM2_CHAIN, k_block=256)
    assert pdl.TRIGGER == "none" and pdl.TRIGGER_SMALL == "none"
    assert not hasattr(pdl, "BOX_FORM_TRIGGERS") and not hasattr(
        fp8bs_small, "TRIGGERS"
    )
    quant = inspect.getsource(fp8bs_small.BSQuantKernel.kernel)
    assert quant.index("griddepcontrol_wait()") < quant.index("pdl.trigger_one_wave()")
    gather = inspect.getsource(fp8bs_small.GatherRowsScalesKernel.kernel)
    assert gather.index("griddepcontrol_wait()") < gather.index(
        "pdl.trigger_one_wave()"
    )
    plan = inspect.getsource(fp8bs_small.PlanQuantKernel.kernel)
    assert "griddepcontrol_wait()" in plan and "trigger" not in plan
    assert "launch_dependents" not in inspect.getsource(fp8bs_small)
    body = inspect.getsource(epilogue_fp8bs)
    assert "pdl.trigger_late()" in body and "launch_dependents" not in body
    assert "trigger" not in inspect.signature(fp8bs_small.quant_kernel_name).parameters
    assert (
        "trigger"
        not in inspect.signature(fp8bs_small.BSQuantKernel.__init__).parameters
    )


def test_fp8bs_classes_run_on_the_unified_bodies():
    """The block-scale fp8 classes have no body of their own: the backend's grouped and swapped bodies (:mod:`blocks.kernel`,
    :mod:`blocks.swap`) take the chain through the promoted operand block (the dense fp8 fold reset per scale block, the
    partials promoted by two warpgroups from an fp32 scale ring; the block and the weight interleave read off the chain;
    whole warpgroups of threads for the register reconfiguration), the record spaces are the class's own (the dense
    block-scale records at the promotion tiles; 64- / 128-row weight tiles x 8- to 64-token tiles swapped), the compiled
    objects carry the format, the block and the interleave in their names, and one JIT module (``batch_invariant_moe``) holds
    the kernels, the quantizer and the scale gathers, with no class-specific body in its source closure."""
    from flashinfer.batch_invariant import jit
    from flashinfer.fused_moe.batch_invariant.blocks import compile as C
    from flashinfer.fused_moe.batch_invariant.blocks import fp8bs_small
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import (
        GEMM2_CHAIN,
        Fp8BsChain,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.kernel import GroupedBlockGemm
    from flashinfer.fused_moe.batch_invariant.blocks.operands import PromotedOperands
    from flashinfer.fused_moe.batch_invariant.blocks.swap import (
        SwappedGroupedGemm,
        is_swapped_record,
        swapped_knobs,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        can_implement_fp8bs,
        can_implement_grouped_fp8bs,
        can_implement_swapped_fp8bs,
        swapped_knobs_fp8bs,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs

    fused, plain, plain32 = (
        gemm1_chain("silu", fused=True),
        GEMM2_CHAIN,
        GEMM2_CHAIN.at(32),
    )
    oracle = Fp8BsChain("plain", "silu", "f32", interleaved=True)
    assert (
        fused.quant and not plain.quant and fused.operands == () and not fused.needs_aux
    )
    assert (fused.block, fused.il, plain32.block, oracle.il, GEMM2_CHAIN.il) == (
        128,
        True,
        32,
        True,
        False,
    )
    assert fused.twin == gemm1_chain("silu") and plain.twin is plain
    assert fused.grouped_loop() is None and fused.swapped_loop() is None
    with pytest.raises(AssertionError):
        gemm1_chain("silu", fused=True).at(
            32
        )  # the fused requantization is the 128-wide class's
    with pytest.raises(AssertionError):
        Fp8BsChain(
            "glu", "silu", "bf16", interleaved=False
        )  # the gated chain pairs the interleaved weight
    dense = Knobs((128, 256), (1, 1), False, True, "m", 1)
    for ch in (fused, plain, plain32, oracle):
        body = GroupedBlockGemm(dense, ch)
        ops = body.operands
        assert ops == PromotedOperands(ch.block, ch.il) and ops.promoted
        assert not ops.block_scaled and ops.dense_a and ops.in_fmt == "e4m3_bs"
        assert (ops.spb, body.epi_warpgroups, body.threads) == (128 // ch.block, 2, 384)
    # the record spaces are the class's own
    assert [(c.tile_m, c.tile_n) for c in swapped_knobs("e4m3_bs")] == [
        (wt, tn) for wt in (64, 128) for tn in (8, 16, 32, 64)
    ]
    assert swapped_knobs_fp8bs() == swapped_knobs("e4m3_bs")
    sw = swapped_knobs_fp8bs()[0]
    assert is_swapped_record(sw, "e4m3_bs") and not is_swapped_record(sw, "bf16")
    assert SwappedGroupedGemm(sw, plain).operands == PromotedOperands(128, False)
    assert can_implement_grouped_fp8bs(dense, 1024, 2048, 8, fused)
    assert not can_implement_grouped_fp8bs(dense, 1024, 2048, 8, fused, block=32)
    assert not can_implement_swapped_fp8bs(sw, 1024, 2048, 8, fused)
    assert can_implement_swapped_fp8bs(sw, 1024, 2048, 8, plain, block=32)
    assert not can_implement_fp8bs(
        dense, 1024 + 64, 2048, 8, plain
    )  # whole weight blocks
    assert not can_implement_fp8bs(
        dense, 1024, 2048 + 64, 8, plain
    )  # whole smem stages of K
    assert not can_implement_fp8bs(
        Knobs((128, 32), (1, 1), False, True, "m", 1), 1024, 2048, 8, plain
    )  # the promotion tiles
    # the object names: the format, the chain with its block and interleave, the record
    assert C.kernel_name(dense, fused).startswith("grouped_e4m3_bs_glu_silu_e4m3bs_")
    assert C.kernel_name(dense, plain32).startswith(
        "grouped_e4m3_bs_plain_silu_bf16_b32_"
    )
    assert C.kernel_name(dense, oracle).startswith("grouped_e4m3_bs_plain_silu_f32_il_")
    assert C.swapped_kernel_name(sw, plain).startswith(
        "swapped_e4m3_bs_plain_silu_bf16_"
    )
    # one module, no class-specific body in its closure
    assert C.MODULE == fp8bs_small.MODULE == "batch_invariant_moe"
    assert "batch_invariant_moe_fp8bs" not in jit.MODULE_ROOTS
    srcs = jit.relative_sources("batch_invariant_moe")
    gone = ("fp8bs_kernel.py", "fp8bs_swap_kernel.py", "fp8bs_compile.py")
    assert not any(os.path.basename(src) in gone for src in srcs), srcs
    for src in (
        "fused_moe/batch_invariant/blocks/epilogue_fp8bs.py",
        "fused_moe/batch_invariant/blocks/fp8bs_small.py",
        "fused_moe/batch_invariant/blocks/operands.py",
    ):
        assert src in srcs, srcs


@pytest.mark.parametrize(
    "family,sm_count",
    [("sm_107", 212), ("sm_107", 204), ("sm_100", 148), ("sm_103", 152)],
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for every bench geometry and token count lies in the enumerated grouped block-scale space of its
    GEMM (the fused requantization's chain in its whole-block subspace), for the gated and the ungated classes, at the nominal
    parts; the ranked list is short, distinct and inside the space; the requantization rule is a function of the rows per expert."""
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule import rows_per_expert
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        FUSED_QUANT_MIN_ROWS,
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        default_configs_fp8bs,
        fused_quant_default,
        gemm1_chain,
        grouped_knobs_fp8bs,
        rank_grouped_fp8bs,
        swapped_knobs_fp8bs,
    )

    assert len(grouped_knobs_fp8bs()) == 54
    # the swapped decode space: 64 / 128 weight rows x 8 / 16 / 32 / 64 tokens, 1-CTA; the bf16 / f32 chains only; disjoint
    sw = swapped_knobs_fp8bs()
    assert len(sw) == 8 and all(
        k.swap_ab and k.tile_m in (64, 128) and k.tile_n in (8, 16, 32, 64) for k in sw
    )
    assert not set(sw) & set(grouped_knobs_fp8bs())
    geoms = [
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (5120, 8192, 16, 1),
        (512, 256, 64, 6),
        (1024, 512, 64, 8),
    ]
    for H, I, E, k in geoms:
        s1 = set(all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu")))
        s1q = set(all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu", fused=True)))
        s1u = set(all_valid_configs_fp8bs(I, H, E, gemm1_chain("relu2", (), False)))
        s1uq = set(
            all_valid_configs_fp8bs(I, H, E, gemm1_chain("relu2", (), False, True))
        )
        s2 = set(all_valid_configs_fp8bs(H, I, E, GEMM2_CHAIN))
        assert (
            len(s1) == len(s2) == 54
            and len(s1q) == 18
            and len(s1uq) == 36
            and s1q <= s1
            and s1uq <= s1u
        )
        w1 = set(all_valid_configs_fp8bs_swapped(2 * I, H, E, gemm1_chain("silu")))
        w1u = set(
            all_valid_configs_fp8bs_swapped(I, H, E, gemm1_chain("relu2", (), False))
        )
        w2 = set(all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN))
        assert len(w1) == len(w1u) == len(w2) == 8
        assert not all_valid_configs_fp8bs_swapped(
            2 * I, H, E, gemm1_chain("silu", fused=True)
        )
        for T in (1, 7, 64, 300, 800, 4096, 16384):
            for fused in (False, True):
                c1, c2 = default_configs_fp8bs(
                    T,
                    k,
                    H,
                    I,
                    E,
                    sm_count,
                    family,
                    "silu",
                    True,
                    (),
                    fused,
                    nominal=True,
                )
                assert c1 in (s1q if fused else s1 | w1) and c2 in s2 | w2, (
                    family,
                    T,
                    H,
                    I,
                    E,
                    k,
                    fused,
                    c1.describe(),
                    c2.describe(),
                )
                assert not (
                    fused and c1.swap_ab
                )  # the fused requantization is the padded body's spelling
                c1u, _ = default_configs_fp8bs(
                    T,
                    k,
                    H,
                    I,
                    E,
                    sm_count,
                    family,
                    "relu2",
                    False,
                    (),
                    fused,
                    nominal=True,
                )
                assert c1u in (s1uq if fused else s1u | w1u), (
                    family,
                    T,
                    H,
                    I,
                    E,
                    k,
                    fused,
                    c1u.describe(),
                )
            ranked = rank_grouped_fp8bs(
                rows_per_expert(T, k, E),
                H,
                I,
                E,
                GEMM2_CHAIN,
                sm_count,
                family,
                3,
                nominal=True,
            )
            assert (
                1 <= len(ranked) <= 3
                and len(set(ranked)) == len(ranked)
                and all(c in s2 | w2 for c in ranked)
            )
            assert not fused_quant_default(T, k, E) and fused_quant_default(
                T, k, E, gated=False
            ) == (rows_per_expert(T, k, E) >= FUSED_QUANT_MIN_ROWS)
    # the acceptance: whole scale blocks of every operand
    assert (
        all_valid_configs_fp8bs(2 * 760, 2048, 128, gemm1_chain("silu")) == []
    )  # I % 128
    assert all_valid_configs_fp8bs(2048, 760, 128, GEMM2_CHAIN) == []  # K % 128
    assert all_valid_configs_fp8bs(2048 + 64, 768, 128, GEMM2_CHAIN) == []  # N % 128


def test_unsupported_device_raises_before_any_kernel():
    """Constructing the backend's runner for the DeepSeek pair gives this pair's runner; on a CPU device it fails its support
    check with the mode's error naming the class. The pair's runner admits its own pair only, and ``from_trtllm_view`` takes
    the DeepSeek pair only (TRTLLM-Gen's MXFP8 view is shuffled)."""
    runner = BatchInvariantFp8BlockRunner(_config(8, 64, 256), torch.device("cpu"))
    assert type(runner) is BatchInvariantDeepSeekFp8Runner
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        runner._check_support()
    assert "moe.e4m3bs" in str(excinfo.value) and "cpu" in str(excinfo.value)
    mxfp8 = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=64, top_k=8),
        quant=mxfp8,
        experts=ExpertConfig(intermediate_size=256),
        backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
    )
    with pytest.raises(NotImplementedError):
        BatchInvariantDeepSeekFp8Runner(cfg, torch.device("cpu"))._check_support()
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="DeepSeekFp8"):
        BatchInvariantFp8BlockConfig.from_trtllm_view({}, quant=mxfp8)


# ----------------------------------------------------------------------------- GPU


def _reference_route(logits: torch.Tensor, k: int, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _exact_acc(
    a: torch.Tensor,
    sa_mn: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts,
    canon_rows,
    sel,
) -> torch.Tensor:
    """The dense block-scale class, exactly, on every selected expert's rows of the dense packing: the device block partials
    (the per-tensor fp8 class at K = 128 with an fp32 output: ``tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py``)
    promoted in exact fp32 arithmetic with the row's and the weight block's scales. ``canon_rows``: the canonical weight row
    of every output column (the gated weight's interleave), or None."""
    from tests.gemm.test_batch_invariant_gemm_fp8_blockscale import (
        block_partials,
        exact_promotion,
    )

    Tk = a.shape[0]
    N = w.shape[1]
    acc = torch.zeros(Tk, N, dtype=torch.float32)
    sa = sa_mn.t().cpu()
    off = 0
    for e, c in enumerate(counts):
        if c and e in sel:
            rows = a[off : off + c].contiguous()
            if (
                canon_rows is not None
            ):  # the exact fold on the canonical rows, its columns permuted to the interleaved order
                w_canon = torch.empty_like(w[e])
                w_canon[canon_rows] = w[e]
                acc[off : off + c] = exact_promotion(
                    block_partials(rows, w_canon), sa[off : off + c], w_sf[e]
                )[:, canon_rows]
            else:
                acc[off : off + c] = exact_promotion(
                    block_partials(rows, w[e]), sa[off : off + c], w_sf[e]
                )
        off += c
    return acc


def _selected(counts, limit: int = 32):
    active = [e for e, c in enumerate(counts) if c]
    if len(active) <= limit:
        return set(active)
    step = -(-len(active) // (limit - 4))
    return set(active[::step]) | {
        active[0],
        active[-1],
        max(active, key=lambda e: counts[e]),
        min(active, key=lambda e: counts[e]),
    }


def _ladder(xq, sx, ids, weights, view, act, out=None):
    """The class composed from its own pieces, bit for bit: the dense packing, the exact block-scale fold per (selected) expert,
    the CPU chain, the quantizer's reference on the device's bf16 intermediate, the exact fold of GEMM2, the CPU conversion and
    combine; the op's intermediates in both requantization spellings against it (a NaN lane by its mask: the payload is the part's
    canonical). Returns the reference output rows checked."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    W = MoEWeightsE4M3BS.from_view(view)
    T, H = xq.shape
    k = ids.shape[1]
    I = W.intermediate_size
    N1 = W.w1.shape[1]
    ref_out, im = fused_moe_fp8bs(
        xq, sx, ids, weights, W, return_intermediates=True, fuse_quant=False
    )
    counts = im["counts"].cpu().tolist()
    assert (
        counts
        == torch.bincount(
            ids.cpu().reshape(-1).long(), minlength=W.num_experts
        ).tolist()
    )
    Tk = T * k
    dest = im["dest"].cpu().reshape(-1).long()
    tok = torch.arange(Tk) // k
    assert torch.equal(
        im["x_perm"].view(torch.uint8).cpu()[dest], xq.view(torch.uint8).cpu()[tok]
    )
    assert torch.equal(bits(im["sx_perm"].cpu()[:, dest]), bits(sx.cpu()[:, tok]))
    assert torch.equal(im["src"].cpu().long()[dest], torch.arange(Tk))
    sel = _selected(counts)
    mask = torch.zeros(Tk, dtype=torch.bool)
    off = 0
    for e, c in enumerate(counts):
        if e in sel:
            mask[off : off + c] = True
        off += c
    acc1 = _exact_acc(
        im["x_perm"],
        im["sx_perm"],
        W.w1,
        W.w1_sf,
        counts,
        RB.orig_rows(N1) if W.gated else None,
        sel,
    )
    a_ref = RB.chain_cpu(acc1, act)
    a = im["a"].cpu()
    assert a.shape == (Tk, I) and _same_bits_or_both_nan(a[mask], a_ref[mask]), (
        "gemm1",
        (a[mask] != a_ref[mask]).sum().item(),
    )
    aq_ref, sa_ref = RB.quantize_rows_bytes(a, "MN")
    assert torch.equal(im["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(im["sa"].cpu()), bits(sa_ref)
    ), "the requantization"
    acc2 = _exact_acc(im["aq"], im["sa"], W.w2, W.w2_sf, counts, None, sel)
    y_ref = R.TorchF32.cvt_out(acc2, "bf16")
    y = im["y"].cpu()
    assert _same_bits_or_both_nan(y[mask], y_ref[mask]), (
        "gemm2",
        (y[mask] != y_ref[mask]).sum().item(),
    )
    assert _same_bits_or_both_nan(
        ref_out.cpu(), R.combine_cpu(im["y"], im["dest"], weights)
    )
    outq, imq = fused_moe_fp8bs(
        xq, sx, ids, weights, W, return_intermediates=True, fuse_quant=True
    )
    assert imq["a"] is None and imq["fused"]
    assert torch.equal(imq["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(imq["sa"].cpu()), bits(sa_ref)
    ), "the fused requantization"
    assert torch.equal(bits(imq["y"].cpu()), bits(y)) and torch.equal(
        bits(outq), bits(ref_out)
    )
    if out is not None:
        assert torch.equal(bits(out), bits(ref_out))
    return ref_out


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU and the oracle ladder of the class (the exact block-scale
    fold per expert, the CPU chain, the quantizer's reference, GEMM2, the combine) bit for bit, for every activation; the fused
    and the standalone requantization give the same bytes and the same output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    xq, sx = _xq(x)
    W = _weights(w1, w2, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(xq, sx, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    _ladder(
        xq,
        sx,
        ids.cuda(),
        weights.cuda(),
        W.get_view(BatchInvariantFp8BlockRunner.backend_key),
        chain,
        out,
    )


@needs_device
def test_bf16_activations_quantize_in_the_plan_launch():
    """bf16 activations handed to the op or the layer are quantized by the class's own chain in one launch with the dense
    permutation plan: the launch's codes, scales and plan equal the two launches' bit for bit (and above the plan's multi-CTA
    threshold the two launches run); the op from bf16 rows equals the op from the class's pair; the layer takes the raw-bf16
    pack and gives the pair's bits; bf16 rows with a scale are refused; the routed form keeps its route + plan launch."""
    from flashinfer.fused_moe.batch_invariant.api import (
        PLAN_MULTI_MIN_N,
        permute_plan,
        route,
    )
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fp8_block_quantize,
        fp8_block_quantize_plan,
        fused_moe_fp8bs,
    )

    for T, k, H, I, E, seed in ((7, 4, 512, 256, 32, 41), (40, 8, 1024, 512, 64, 43)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed, adversarial=True)
        ids, wts = route(logits, k)
        q0, s0 = fp8_block_quantize(x, "MN")
        c0, d0, r0 = permute_plan(ids, E)
        q1, s1, c1, d1, r1 = fp8_block_quantize_plan(x, ids, E)
        assert torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)) and torch.equal(
            bits(s0), bits(s1)
        )
        assert torch.equal(c0, c1) and torch.equal(d0, d1) and torch.equal(r0, r1)
        view = BatchInvariantFp8BlockConfig.prepare_weights(
            w1, w2, quant=QUANT, activation=SwiGLU()
        )
        W = MoEWeightsE4M3BS.from_view(view)
        ref = fused_moe_fp8bs(q0, s0, ids, wts, W)
        assert torch.equal(bits(fused_moe_fp8bs(x, None, ids, wts, W)), bits(ref))
        # a given plan (the routed form's): the standalone quantizer, the same bits
        assert torch.equal(
            bits(fused_moe_fp8bs(x, None, ids, wts, W, plan=(c0, d0, r0))), bits(ref)
        )
        with pytest.raises(ValueError):
            fused_moe_fp8bs(x, s0, ids, wts, W)
        # the layer: the raw-bf16 pack against the class's pair
        Wp = _weights(w1, w2)
        layer = _layer(k, E, I)
        out_pair = layer(_routed_pack(q0, s0, ids, wts, packed=True), Wp)
        out_bf16 = layer(_routed_pack(x, None, ids, wts, packed=True), Wp)
        assert torch.equal(bits(out_pair), bits(out_bf16))
        with pytest.raises((bi.BatchInvariantUnsupportedError, NotImplementedError)):
            # the runner refuses the pair's scale beside bf16 rows (the layer refuses the pack's shape first)
            layer(_routed_pack(x, s0, ids, wts, packed=True), Wp)
    # above the plan's multi-CTA threshold: the two launches, the same outputs
    T, k, H, I, E = PLAN_MULTI_MIN_N // 8 + 8, 8, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=47)
    ids, _ = route(logits, k)
    q0, s0 = fp8_block_quantize(x, "MN")
    c0, d0, r0 = permute_plan(ids, E)
    q1, s1, c1, d1, r1 = fp8_block_quantize_plan(x, ids, E)
    assert torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)) and torch.equal(
        bits(s0), bits(s1)
    )
    assert torch.equal(c0, c1) and torch.equal(d0, d1) and torch.equal(r0, r1)


@needs_device
def test_input_quantization_is_the_class_on_the_device():
    """``prepare_activations`` runs the ``fp8bs.quant`` kernel: codes and block scales == the class's reference bit for bit on
    adversarial rows, in both output layouts of the standalone quantizer and every warps-per-row placement; R1 / R2 at op
    level; the layer takes the pair in the DeepSeek layout alone (the token-major layout is refused naming it), and
    FlashInfer's own DeepSeek pack as given."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        fp8_block_quantize,
        quant_warps_per_row,
    )

    for M, K in [(1, 128), (37, 256), (300, 2048), (1000, 512), (129, 384)]:
        x = _adversarial_rows(M, K, M + K).cuda()
        rq, rs = RB.quantize_rows_bytes(x, "MN")
        for lay in ("MN", "K"):
            for wpr in (1, 2, 4, 8):
                if wpr > max(1, (K // BLOCK + 1) // 2):
                    continue
                q, s = fp8_block_quantize(x, lay, wpr=wpr)
                s_mn = s if lay == "MN" else s.t()
                assert torch.equal(q.view(torch.uint8).cpu(), rq) and torch.equal(
                    bits(s_mn.cpu().contiguous()), bits(rs)
                ), (M, K, lay, wpr)
        assert 1 <= quant_warps_per_row(M, K, 212) <= 8
        q, s = BatchInvariantFp8BlockConfig.prepare_activations(x, quant=QUANT)
        assert torch.equal(q.view(torch.uint8).cpu(), rq) and torch.equal(
            bits(s.cpu()), bits(rs)
        )
    x = _adversarial_rows(200, 1024, 3).cuda()
    q, s = fp8_block_quantize(x)
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(1)).cuda()
    qp, sp = fp8_block_quantize(x[perm].contiguous())
    assert torch.equal(qp.view(torch.uint8), q.view(torch.uint8)[perm]) and torch.equal(
        bits(sp), bits(s[:, perm])
    )
    for n in (1, 7, 64):
        qs, ss = fp8_block_quantize(x[:n].contiguous())
        assert torch.equal(
            qs.view(torch.uint8), q.view(torch.uint8)[:n]
        ) and torch.equal(bits(ss), bits(s[:, :n]))
    T, k, H, I, E = 100, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=23, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    ref = bits(layer(_logits_pack(xq, sx, logits), W))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="hidden_states_scale"):
        layer(_logits_pack(xq, sx.t().contiguous(), logits), W)
    assert torch.equal(ref, bits(layer(_logits_pack(xq, sx, logits), W)))
    # FlashInfer's own DeepSeek pair (its torch recipe divides by the scale): the class starts at the bytes and the scales it is handed
    q_fi, s_fi = TrtllmFp8BlockConfig.prepare_activations(x, quant=QUANT)
    out_fi = layer(_logits_pack(q_fi, s_fi, logits), W)
    assert out_fi.shape == (T, H) and torch.isfinite(out_fi.float()).all()


@needs_device
def test_gemm1_gather_form_equals_the_box_form_at_decode():
    """GEMM1's swapped records read the token rows and their scales through the plan's map (the gather form, no routed gather
    launch): the same bits as the box form over the permuted copies, gated and ungated, at a few rows per expert; the fused op
    takes the gather form by default for a swapped GEMM1 record without the fused requantization and refuses it elsewhere."""
    from flashinfer.fused_moe.batch_invariant.api import GATHER_TOKEN_TILE_MAX, route
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        gemm1_chain,
    )

    for gated in (True, False):
        T, k, H, I, E = 7, 4, 512, 256, 32
        x, logits, w1, w2 = _problem(
            T, k, H, I, E, seed=31, gated=gated, adversarial=True
        )
        xq, sx = _xq(x)
        view = BatchInvariantFp8BlockConfig.prepare_weights(
            w1, w2, quant=QUANT, activation=SwiGLU() if gated else ReLU2()
        )
        W = MoEWeightsE4M3BS.from_view(view)
        ids, wts = route(logits, k)
        box = fused_moe_fp8bs(xq, sx, ids, wts, W, gather=False)
        gat = fused_moe_fp8bs(xq, sx, ids, wts, W, gather=True)
        dflt, im = fused_moe_fp8bs(xq, sx, ids, wts, W, return_intermediates=True)
        assert torch.equal(bits(box), bits(gat)) and torch.equal(bits(box), bits(dflt))
        assert (
            im["config1"].swap_ab and not im["fused"] and im["gather"]
        )  # the decode pick: a narrow swapped record, the gather form the default
        # a wide swapped token tile takes the box form by default (the bf16 op's rule); its gather form gives the same bits
        wide = next(
            c
            for c in all_valid_configs_fp8bs_swapped(
                W.w1.shape[1], H, E, gemm1_chain(W.act, W.params, gated)
            )
            if c.tile_n > GATHER_TOKEN_TILE_MAX
        )
        out_w, im_w = fused_moe_fp8bs(
            xq, sx, ids, wts, W, config1=wide, return_intermediates=True
        )
        assert not im_w["gather"] and torch.equal(bits(out_w), bits(box))
        assert torch.equal(
            bits(fused_moe_fp8bs(xq, sx, ids, wts, W, config1=wide, gather=True)),
            bits(box),
        )
        # a grouped record has no gather form; nor has the fused requantization
        grouped = next(
            c
            for c in all_valid_configs_fp8bs(
                W.w1.shape[1], H, E, gemm1_chain(W.act, W.params, gated)
            )
            if not c.swap_ab
        )
        with pytest.raises(ValueError):
            fused_moe_fp8bs(xq, sx, ids, wts, W, config1=grouped, gather=True)
        if gated:
            with pytest.raises(ValueError):
                fused_moe_fp8bs(xq, sx, ids, wts, W, fuse_quant=True, gather=True)


@needs_device
def test_the_largest_expert_count_and_slot_count_route_and_run():
    """1024 experts and 16 slots: the class's routing from the logits equals the CPU model and the layer equals the class
    composition."""
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    for T, k, H, I, E in ((96, 8, 256, 128, 1024), (40, 16, 256, 128, 64)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
        xq, sx = _xq(x)
        ids, wts = route(logits, k)
        ref_ids, ref_w = _reference_route(logits, k, RoutingMethodType.Renormalize)
        assert torch.equal(ids.cpu(), ref_ids) and torch.equal(
            bits(wts.cpu()), bits(ref_w)
        )
        W = _weights(w1, w2)
        out = _layer(k, E, I)(_logits_pack(xq, sx, logits), W)
        _ladder(
            xq,
            sx,
            ids,
            wts,
            W.get_view(BatchInvariantFp8BlockRunner.backend_key),
            activation_of(_SWIGLU),
            out,
        )


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their inputs;
    repeated runs and a CUDA-graph replay reproduce them."""
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    out_l = layer(_logits_pack(xq, sx, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack(xq, sx, ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    w16 = weights.to(torch.bfloat16)
    out_u16 = layer(_routed_pack(xq, sx, ids.cuda(), w16.cuda()), W)
    out_p = layer(_routed_pack(xq, sx, ids.cuda(), w16.cuda(), packed=True), W)
    assert torch.equal(bits(out_u16), bits(out_p))
    assert torch.equal(
        bits(out_u16),
        bits(layer(_routed_pack(xq, sx, ids.cuda(), w16.float().cuda()), W)),
    )
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack(xq, sx, logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack(xq, sx, logits)
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
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens create
    (the requantization spelling the rows-per-expert rule picks changes with the load; the bits do not)."""
    T, k, H, I, E = 500, 8, 512, 256, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    full = bits(layer(_logits_pack(xq, sx, logits), W))
    for n in (1, 2, 7, 64, 200, 333):
        assert torch.equal(
            bits(layer(_logits_pack(xq[:n], sx[:, :n].contiguous(), logits[:n]), W)),
            full[:n],
        ), n
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(4)).cuda()
    assert torch.equal(
        bits(
            layer(
                _logits_pack(
                    xq[perm].contiguous(),
                    sx[:, perm].contiguous(),
                    logits[perm].contiguous(),
                ),
                W,
            )
        ),
        full[perm],
    )
    for n in (1, 2, 7, 64, 200):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(n))[:n].cuda()
        assert torch.equal(
            bits(
                layer(
                    _logits_pack(
                        xq[idx].contiguous(),
                        sx[:, idx].contiguous(),
                        logits[idx].contiguous(),
                    ),
                    W,
                )
            ),
            full[idx],
        ), n
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    base = bits(layer(_routed_pack(xq, sx, ids, weights), W))
    g = torch.Generator().manual_seed(5)
    xq_extra, sx_extra = _xq(
        (torch.randn(556, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    )
    w_extra = torch.rand(556, k, generator=g).cuda()
    for t in (0, T // 2, T - 1):
        hot = ids[t : t + 1].expand(556, k).contiguous()
        out = layer(
            _routed_pack(
                torch.cat([xq[t : t + 1], xq_extra]),
                torch.cat([sx[:, t : t + 1], sx_extra], 1).contiguous(),
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
                torch.cat([xq[t : t + 1], xq_extra]),
                torch.cat([sx[:, t : t + 1], sx_extra], 1).contiguous(),
                torch.cat([ids[t : t + 1], others]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("unrelated hot experts", t)


@needs_device
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits():
    """Every tactic the autotuner may pick (the dispatcher's ranked record pairs in both requantization spellings) gives the same
    bits; every record of the fast grouped space in both spellings too; tuning the layer profiles them and the output before,
    during and after tuning is one output."""
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        gemm1_chain,
    )

    T, k, H, I, E = 300, 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=6, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantDeepSeekFp8Runner)
    pack = _logits_pack(xq, sx, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert (
        2 <= len(tactics) <= 18
        and len(set(tactics)) == len(tactics)
        and {t[-1] for t in tactics} == {0, 1}
        and all(len(t) == 15 for t in tactics)
    )
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        ), t
    failed_before = (
        len(AutoTuner.get().stats.failed_tactics)
        if hasattr(AutoTuner.get(), "stats")
        else 0
    )
    layer.reset_winner()
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    if hasattr(AutoTuner.get(), "stats"):
        assert len(AutoTuner.get().stats.failed_tactics) == failed_before
    # the fast cluster subset of the whole grouped space, both GEMMs, both spellings
    Wd = MoEWeightsE4M3BS.from_view(W.get_view(runner.backend_key))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    clusters = ((1, 1), (1, 2), (2, 1))
    s1 = all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu"), clusters)
    s1q = all_valid_configs_fp8bs(
        2 * I, H, E, gemm1_chain("silu", fused=True), clusters
    )
    s2 = all_valid_configs_fp8bs(H, I, E, GEMM2_CHAIN, clusters)
    assert (
        len(s1) == len(s2) == 24 and len(s1q) == 8
    )  # 1x1 and 1x2: 6 records each; 2x1: the 1-CTA and the 2-CTA records
    bad = [
        c.describe()
        for c in s1
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=False
                )
            ),
            ref,
        )
    ]
    bad += [
        "fused:" + c.describe()
        for c in s1q
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=True
                )
            ),
            ref,
        )
    ]
    bad += [
        "gemm2:" + c.describe()
        for c in s2
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=s1[0], config2=c, fuse_quant=False
                )
            ),
            ref,
        )
    ]
    # the swapped decode records (the dispatcher's at a few rows per expert; every one a member of the class): both GEMMs
    sw1 = all_valid_configs_fp8bs_swapped(2 * I, H, E, gemm1_chain("silu"))
    sw2 = all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN)
    assert len(sw1) == len(sw2) == 8
    bad += [
        "swapped:" + c.describe()
        for c in sw1
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=False
                )
            ),
            ref,
        )
    ]
    bad += [
        "swapped gemm2:" + c.describe()
        for c in sw2
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=s1[0], config2=c, fuse_quant=False
                )
            ),
            ref,
        )
    ]
    assert not bad, bad[:8]


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_swapped_decode_body_equals_the_padded_body(act):
    """At a few rows per active expert the dispatcher picks the swapped decode body for both GEMMs (the structural rule); the layer's
    output equals the padded body's bit for bit, every swapped record of both GEMMs agrees, the first tactic is the swapped pair
    and the autotuner keeps the bits."""
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs_swapped,
        default_configs_fp8bs,
        gemm1_chain,
    )

    T, k, H, I, E = 8, 4, 1024, 512, 64
    gated = not isinstance(act, ReLU2)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    if not gated:
        w1 = w1[:, :I].contiguous()
    xq, sx = _xq(x)
    layer = _layer(k, E, I, act=act)
    W = _weights(w1, w2, act=act)
    runner = layer.runners[0]
    pack = _logits_pack(xq, sx, logits)
    got = bits(layer(pack, W))  # the launch binds the hidden size the dispatcher reads
    c1, c2, fused = runner._default(T)
    assert c1.swap_ab and c2.swap_ab and not fused, (
        c1.describe(),
        c2.describe(),
        fused,
    )
    Wd = MoEWeightsE4M3BS.from_view(W.get_view(runner.backend_key))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    sm, fam = runner._part if runner._part is not None else (None, None)
    name = "silu" if gated else "relu2"
    p1, p2 = default_configs_fp8bs(
        T, k, H, I, E, sm, fam, name, gated, (), False, swap=False
    )
    assert not p1.swap_ab and not p2.swap_ab
    ref = bits(
        fused_moe_fp8bs(xq, sx, ids, wts, Wd, config1=p1, config2=p2, fuse_quant=False)
    )
    assert torch.equal(got, ref), (c1.describe(), c2.describe())
    n1 = 2 * I if gated else I
    ch1 = gemm1_chain(name, (), gated)
    for c in all_valid_configs_fp8bs_swapped(n1, H, E, ch1):
        out = fused_moe_fp8bs(
            xq, sx, ids, wts, Wd, config1=c, config2=p2, fuse_quant=False
        )
        assert torch.equal(bits(out), ref), ("gemm1", c.describe())
    for c in all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN):
        out = fused_moe_fp8bs(
            xq, sx, ids, wts, Wd, config1=p1, config2=c, fuse_quant=False
        )
        assert torch.equal(bits(out), ref), ("gemm2", c.describe())
    tactics = runner.get_valid_tactics(runner.pack_inputs(pack, W), None)
    # both records swapped (field 6 of each 7-field record), the standalone spelling last
    assert tactics[0][6] == 1 and tactics[0][13] == 1 and tactics[0][14] == 0, tactics[
        0
    ]
    layer.reset_winner()
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)


@needs_device
def test_routing_methods_from_logits_match_the_cpu_model():
    """Every routing method the class implements, from logits, selects and weights as the exact CPU model; the layer's bits follow."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=7, adversarial=True)
    xq, sx = _xq(x)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(8)) * 0.1)
        .float()
        .cuda()
    )
    W = _weights(w1, w2)
    for method, groups in (
        (RoutingMethodType.Default, None),
        (RoutingMethodType.Renormalize, None),
        (RoutingMethodType.Sigmoid, None),
        (RoutingMethodType.SigmoidRenorm, None),
        (RoutingMethodType.DeepSeekV3, None),
        (RoutingMethodType.DeepSeekV3, (8, 4)),
        (RoutingMethodType.DeepSeekV3, (4, 1)),
    ):
        rm = S.METHOD_OF_FLASHINFER[method]
        kwargs = {"routed_scaling_factor": 1.75} if rm.scaled else {}
        if groups is not None:
            # the group-limited selection: a term of the DeepSeekV3 method (the shared routing_term builds the grouped method)
            rm = rm.with_groups(*groups)
            kwargs["n_group"], kwargs["topk_group"] = groups
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=method, **kwargs),
                quant=QUANT,
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )
        b = bias if rm.bias else None
        out = layer(_logits_pack(xq, sx, logits, b), W)
        ex, w = R.route_cpu(
            logits.cpu(),
            k,
            rm,
            b.cpu() if b is not None else None,
            1.75 if rm.scaled else None,
        )
        ref = layer(_routed_pack(xq, sx, ex.cuda(), w.cuda()), W)
        assert torch.equal(bits(out), bits(ref)), (method, groups)
        assert torch.equal(bits(out), bits(layer(_logits_pack(xq, sx, logits, b), W)))


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the classes raises the mode's error at build or at the first call (top_k /
    num_experts outside the bounds, fused shared experts, an expert-parallel shard, expert groups with a method other than
    DeepSeekV3 from logits, a group shape outside the rule, no finalize, an intermediate size that is not a multiple of 128, a hidden size that is not, bf16 activations with a scale tensor or f32 activations handed to the fp8 layer, no
    activation scale tensor, a scale tensor of the wrong shape, weights prepared for another activation, an unimplemented routing
    method from logits); a zero-token call returns an empty output."""
    k, H, I, E = 6, 512, 256, 64
    errors = (bi.BatchInvariantUnsupportedError, RuntimeError, NotImplementedError)
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=320), "intermediate_size"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.e4m3bs" in str(excinfo.value)
    routing = RoutingConfig(num_experts=E, top_k=k)
    experts = ExpertConfig(intermediate_size=I)
    for kw, what in (
        (
            dict(
                routing=RoutingConfig(
                    num_experts=E, top_k=k, method=RoutingMethodType.DeepSeekV3
                ),
                experts=ExpertConfig(intermediate_size=I, num_fused_shared_experts=1),
            ),
            "fused shared experts",
        ),
        (dict(finalize=MoEFinalizeConfig(do_finalize=False)), "do_finalize"),
    ):
        with pytest.raises(errors) as excinfo:
            MoELayer(
                MoEConfig(
                    routing=kw.get("routing", routing),
                    quant=QUANT,
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.e4m3bs" in str(excinfo.value), what
    # the group shape rule of the DeepSeekV3 term is refused at the from-logits call, not at construction (the shared runner's
    # rule: the pre-routed modes never consult the term, so a pre-routed layer builds under any RoutingConfig)
    for n_group, topk_group, kk in ((3, 1, k), (64, 4, k), (8, 9, k), (8, 1, 9)):
        grouped_layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(
                    num_experts=E,
                    top_k=kk,
                    method=RoutingMethodType.DeepSeekV3,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=1.0,
                ),
                quant=QUANT,
                experts=experts,
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
            ),
            device=torch.device("cuda"),
        )
        x_g, logits_g, w1_g, w2_g = _problem(8, kk, H, I, E, seed=9)
        xq_g, sx_g = _xq(x_g)
        with pytest.raises(errors) as excinfo:
            grouped_layer(
                _logits_pack(xq_g, sx_g, logits_g, torch.zeros(E, device="cuda")),
                _weights(w1_g, w2_g),
            )
        assert "n_group" in str(excinfo.value) or "top_k" in str(excinfo.value), (
            n_group,
            topk_group,
            kk,
        )
    # an expert-parallel shard (the layer's local expert range): the fp8-block backend declares no expert parallelism, so the
    # layer's candidate filter and the runner's support check refuse the shard by name (the bf16 backend's base runner, which
    # computes shards, is not this runner's base)
    assert BatchInvariantDeepSeekFp8Runner.supports_expert_parallelism is False
    for shard in (
        dict(local_expert_offset=E // 2, local_num_experts=E // 2),
        dict(local_expert_offset=0, local_num_experts=E // 2),
    ):
        sharded = MoEConfig(
            routing=routing,
            quant=QUANT,
            experts=ExpertConfig(intermediate_size=I, **shard),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        )
        with pytest.raises(RuntimeError) as excinfo:
            MoELayer(sharded, device=torch.device("cuda"))
        assert "expert parallelism" in str(excinfo.value), shard
        # the class's own refusal fires first and names the shard (the shared assertion stands behind it)
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="expert-parallel shards"
        ):
            BatchInvariantDeepSeekFp8Runner(
                sharded, torch.device("cuda")
            )._check_support()
    # the bf16 and the per-tensor fp8 quantization pairs never reach this runner
    for quant in (
        QuantConfig(),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
    ):
        with pytest.raises(errors):
            MoELayer(
                MoEConfig(
                    routing=routing,
                    quant=quant,
                    experts=experts,
                    backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                ),
                device=torch.device("cuda"),
            )
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=9)
    xq, sx = _xq(x)
    W = _weights(w1, w2)
    with pytest.raises(errors):
        layer(
            _logits_pack(x, sx, logits), W
        )  # bf16 activations with a scale tensor: the raw-bf16 pack carries no scale (the class quantizes the rows itself)
    with pytest.raises(errors):
        layer(
            _logits_pack(x.float(), None, logits), W
        )  # f32 activations: neither the pair nor the raw-bf16 pack
    with pytest.raises(errors):
        layer(_logits_pack(xq, None, logits), W)  # no block scales
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx[:1].contiguous(), logits), W
        )  # a scale tensor of the wrong shape
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2())
        )  # the ungated weights with the gated class
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx, logits), _weights(w1, w2, GeGLU())
        )  # another activation's chain
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        _layer(k, E, I)(
            _logits_pack(xq[:, :384].contiguous(), sx[:3].contiguous(), logits), W
        )  # H = 384 does not match the weights
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=QUANT,
            experts=experts,
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    # Llama4 routing (TopK with sigmoid weights) is a routing term the bf16 model serves: the layer runs from logits, deterministically
    out_l = llama(_logits_pack(xq, sx, logits), W)
    assert out_l.shape == (8, H) and torch.equal(
        bits(out_l), bits(llama(_logits_pack(xq, sx, logits), W))
    )
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(
        _routed_pack(xq, sx, ids, torch.ones(8, 1, device="cuda")), W
    ).shape == (8, H)
    # expert groups are a parameter of the DeepSeekV3 term: with another method they are refused from logits only (with
    # pre-routed ids the group structure never enters the computation: the ungrouped layer's bits on the same ids and weights)
    grouped = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.Renormalize,
                n_group=2,
                topk_group=1,
            ),
            quant=QUANT,
            experts=experts,
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        grouped(_logits_pack(xq, sx, logits), W)
    assert "n_group" in str(excinfo.value)
    g_ids, g_wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(grouped(_routed_pack(xq, sx, g_ids.cuda(), g_wts.cuda()), W)),
        bits(layer(_routed_pack(xq, sx, g_ids.cuda(), g_wts.cuda()), W)),
    )
    out = layer(_logits_pack(xq[:0], sx[:, :0].contiguous(), logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


@needs_device
def test_alternating_inputs_under_programmatic_dependent_launch():
    """Two problems run alternately through the op, back to back without host syncs (the kernels' PDL chains overlap and every
    reused intermediate holds the other call's bytes), many times, in both requantization spellings; every output equals the
    problem's isolated result bit for bit."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    T, k, H, I, E = 64, 8, 256, 128, 256
    probs = []
    for seed, fused in ((3, False), (11, True)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed, adversarial=True)
        W = MoEWeightsE4M3BS.prepare(w1, w2)
        xq, sx = _xq(x)
        ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
        ref = bits(fused_moe_fp8bs(xq, sx, ids, wts, W, fuse_quant=fused))
        torch.cuda.synchronize()
        probs.append((xq, sx, ids, wts, W, fused, ref))
    outs = []
    for _ in range(20):
        for xq, sx, ids, wts, W, fused, _r in probs:
            outs.append(fused_moe_fp8bs(xq, sx, ids, wts, W, fuse_quant=fused))
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(bits(out), probs[i % 2][6]), i


def _flashinfer_block_fp8_reference(
    x, w1, w2, ids, weights, act, activation_fn=None
) -> torch.Tensor:
    """FlashInfer's torch reference of the DeepSeek block-fp8 MoE: the form of ``tests/moe/test_unified_moe_fp8.py``'s
    ``_block_fp8_reference`` (the reference its TRTLLM-Gen backend is checked against): the activations quantized per 1x128
    block and the weights per 128x128 block by FlashInfer's own recipe (``prepare.py``: ``(amax / 448).clamp(1e-12)``, the
    division, the clamp) and dequantized to fp32; fp32 GEMM1; the typed activation in fp32 (``tests.moe.utils``'s formulas, the
    layer's own); the intermediate requantized through bf16 the same way; fp32 GEMM2; the routing weights narrowed to bf16;
    the fp32 routed sum. Independent of the class (its exact model is the promise, tested bit for bit above).
    ``activation_fn`` replaces the activation (the negative control of the tolerance)."""
    from flashinfer.fused_moe.prepare import (
        _deepseek_fp8_quantize_activations,
        _deepseek_fp8_quantize_weights,
    )
    from tests.moe.utils import compute_reference_activation, powlu_ref

    def deq_act(t):
        q, s = _deepseek_fp8_quantize_activations(t.to(torch.bfloat16))
        return q.float() * s.transpose(0, 1).repeat_interleave(BLOCK, dim=1)

    def deq_w(w):
        q, s = _deepseek_fp8_quantize_weights(w)
        return q.float() * s.repeat_interleave(BLOCK, dim=1).repeat_interleave(
            BLOCK, dim=2
        )

    weights = weights.to(torch.bfloat16).float()
    x_deq = deq_act(x)
    w1_deq, w2_deq = deq_w(w1), deq_w(w2)
    I = w2.shape[2]
    if activation_fn is None:

        def activation_fn(gemm1):
            if isinstance(act, PowLU):
                up, gate = gemm1.split(I, dim=-1)
                g = powlu_ref(gate, act.m)
                if act.limit is not None:
                    g = g.clamp_max(act.limit)
                    up = up.clamp(-act.limit, act.limit)
                return g * up
            return compute_reference_activation(gemm1, act, I, out_dtype=torch.float32)

    out = torch.zeros_like(x_deq)
    for e in range(w1.shape[0]):
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        inter = activation_fn(x_deq[tok] @ w1_deq[e].t())
        y = deq_act(inter) @ w2_deq[e].t()
        out[tok] += weights[tok, slot, None] * y
    return out


def _wrong_activation(act, I: int):
    """A genuinely different formula over the same weights and scales (FlashInfer's negative control: the tolerance must reject
    it): the up branch alone for a gated activation, plain ReLU for an ungated one (the identity for ReLU itself)."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    if activation_of(act).gated:
        return lambda gemm1: gemm1[:, :I]
    if isinstance(act, ReLU):
        return lambda gemm1: gemm1
    return torch.relu


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_plausible_against_flashinfers_block_fp8_torch_reference(act):
    """An observation on moderate inputs under FlashInfer's own discipline for this quantization pair
    (``tests/moe/test_unified_moe_fp8.py``: ``check_accuracy`` at atol 0.05 / rtol 0.3 on 99 % of the outputs against the torch
    reference above, and the negative control that the tolerance rejects a wrong activation formula). Not a promise of the
    class (the reference and the class round at different points; the exact model is the promise, tested bit for bit above),
    and never a comparison with another backend's output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    T, k, H, I, E = 128, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11, gated=activation_of(act).gated)
    xq, sx = _xq(x)
    out = _layer(k, E, I, act)(
        _logits_pack(xq, sx, logits), _weights(w1, w2, act)
    ).float()
    vals, ids = torch.topk(
        logits, k, dim=-1
    )  # the layer's Renormalize routing: top-k, softmax over the selected
    weights = torch.softmax(vals, dim=-1)
    ref = _flashinfer_block_fp8_reference(x, w1, w2, ids, weights, act)
    check_accuracy(out, ref, atol=0.05, rtol=0.3, percent=0.99)
    wrong = _flashinfer_block_fp8_reference(
        x, w1, w2, ids, weights, act, activation_fn=_wrong_activation(act, I)
    )
    with pytest.raises(Exception, match="Mismatch percentage"):
        check_accuracy(out, wrong, atol=0.05, rtol=0.3, percent=0.99)


# ----------------------------------------------------------------------------- the 32-wide classes (deepseek_fp8_block_size = 32)

BLOCK32 = 32


def _weights32(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
        deepseek_fp8_block_size=BLOCK32,
    )
    pack.prepare_for(BatchInvariantFp8BlockRunner.backend_key, view)
    return pack


def _xq32(x: torch.Tensor):
    xq, sx = BatchInvariantFp8BlockConfig.prepare_activations(
        x, quant=QUANT, deepseek_fp8_block_size=BLOCK32
    )
    assert tuple(sx.shape) == (x.shape[1] // BLOCK32, x.shape[0])
    return xq, sx


def _block_partials32(a: torch.Tensor, w: torch.Tensor, block: int) -> torch.Tensor:
    """``P[kb, m, n]`` of the per-tensor fp8 class's fold of each ``block``-wide K slice from +0 (the dense kernel's zero-filled tail
    steps are identities: at 32 the slice is one instruction step), through ``mm_fp8_batch_invariant`` at the pinned record."""
    from tests.gemm.test_batch_invariant_gemm_fp8_blockscale import PARTIAL_RECORD

    from flashinfer.gemm.batch_invariant.api import mm_fp8_batch_invariant

    M, K = a.shape
    N = w.shape[0]
    N16 = -(-N // 16) * 16
    if N16 != N:
        w = torch.cat([w, torch.zeros(N16 - N, K, dtype=w.dtype, device=w.device)], 0)
    one = torch.ones(1, dtype=torch.float32, device=a.device)
    P = torch.empty(K // block, M, N16, dtype=torch.float32, device=a.device)
    for kb in range(K // block):
        mm_fp8_batch_invariant(
            a[:, kb * block : (kb + 1) * block].contiguous(),
            w[:, kb * block : (kb + 1) * block].contiguous().t(),
            one,
            one,
            out=P[kb],
            out_dtype=torch.float32,
            config=PARTIAL_RECORD,
            chunks=1,
        )
    return P[:, :, :N] if N16 != N else P


def _exact_promotion32(P, sa, sb, block: int):
    """The class's fp32 fold over the block partials with ``block``-column weight scale blocks (the 128-wide helper with the block as
    a parameter)."""
    from flashinfer.batch_invariant.fp_interface import exact

    P, sa, sb = P.cpu(), sa.cpu().float(), sb.cpu().float()
    KB, M, N = P.shape
    col_blk = torch.arange(N) // block
    acc = torch.zeros(M, N, dtype=torch.float32)
    tiny = 2 * 2.0**-126
    for kb in range(KB):
        scale = (sa[:, kb].double()[:, None] * sb[col_blk, kb].double()[None, :]).to(
            torch.float32
        )
        t = scale.double() * P[kb].double()
        acc64 = acc.double()
        s = t + acc64
        bb = s - t
        err = (t - (s - bb)) + (acc64 - bb)
        out = s.to(torch.float32)
        midpoint = (s.view(torch.int64) & ((1 << 29) - 1)) == (1 << 28)
        risky = (err != 0) & torch.isfinite(s) & (midpoint | (s.abs() < tiny))
        for m, n in risky.nonzero(as_tuple=False).tolist():
            out[m, n] = exact.fma(
                float(scale[m, n]), float(P[kb, m, n]), float(acc[m, n])
            )
        acc = out
    return acc


def _exact_acc32(a, sa_mn, w, w_sf, counts, canon_rows, sel, block: int):
    """The 32-wide core, exactly, on every selected expert's rows of the dense packing."""
    Tk = a.shape[0]
    N = w.shape[1]
    acc = torch.zeros(Tk, N, dtype=torch.float32)
    sa = sa_mn.t().cpu()
    off = 0
    for e, c in enumerate(counts):
        if c and e in sel:
            rows = a[off : off + c].contiguous()
            if canon_rows is not None:
                w_canon = torch.empty_like(w[e])
                w_canon[canon_rows] = w[e]
                acc[off : off + c] = _exact_promotion32(
                    _block_partials32(rows, w_canon, block),
                    sa[off : off + c],
                    w_sf[e],
                    block,
                )[:, canon_rows]
            else:
                acc[off : off + c] = _exact_promotion32(
                    _block_partials32(rows, w[e], block),
                    sa[off : off + c],
                    w_sf[e],
                    block,
                )
        off += c
    return acc


def test_block32_is_its_own_class_with_the_same_chains_and_spaces():
    """``deepseek_fp8_block_size`` is a dimension of the class: the fingerprint, the class name (``moe.e4m3bs32.<act>``), the GEMM
    core (``gemm.e4m3.kblock32``: one step per block, a different term from the 128-wide core at the same K) and the quantizer's
    instance differ at 32 while the 128-wide class's values are unchanged; the chains are the same; the record spaces and the
    acceptance read the block; the dispatcher's picks at 32 are members; the registry refuses another block."""
    from flashinfer.fused_moe.batch_invariant import mterm
    from flashinfer.fused_moe.batch_invariant import spec_fp8bs as SB
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import (
        GEMM2_CHAIN,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        can_implement_fp8bs,
        default_configs_fp8bs,
    )
    from flashinfer.gemm.batch_invariant.spec import GemmSpec

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp, fp32 = (
        bi.fingerprint("fused_moe_fp8bs", **dims),
        bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=32, **dims),
    )
    assert fp != fp32 and fp32.startswith("moe-")
    assert bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=128, **dims) == fp
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=64, **dims)
    assert (
        SB.quant_fingerprint(32) != SB.quant_fingerprint() == SB.quant_fingerprint(128)
    )
    assert SB.gemm_core_fingerprint(256, "f32", 32).startswith(
        "gemm-e4m3-kblock32-"
    ) and SB.gemm_core_fingerprint(256, "f32", 128).startswith("gemm-e4m3-kblock128-")
    s32 = SB.Fp8BsMoESpec(256, 128, 2, 8, block=32)
    assert (
        s32.name == "moe.e4m3bs32.swiglu"
        and SB.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False, block=32).name
        == "moe.e4m3bs32.relu2"
    )
    s128 = SB.Fp8BsMoESpec(256, 128, 2, 8, alg=s32.alg)
    assert mterm.sexpr(s32.gemm1()) == mterm.sexpr(s128.gemm1()) and mterm.sexpr(
        s32.gemm2()
    ) == mterm.sexpr(s128.gemm2())
    from flashinfer.batch_invariant.fp_interface.term import TermAlgebra, sexpr

    alg = TermAlgebra()
    core32 = sexpr(SB.reduction_kblock(32, 0, 0, 32).canonical(alg))
    # one instruction step per block, promoted once; the 128-wide core at K = 128 is four steps promoted once
    assert (
        core32
        == "(fma (mma32 (const 0.0) (ab (a 0 0) (b 0 0))) (mul (sfa_blk 0 0) (sfb_blk 0 0)) (const 0.0))"
    )
    core128 = sexpr(SB.reduction_kblock(128, 0, 0, 128).canonical(alg))
    assert (
        core128.startswith("(fma (mma32 (mma32 (mma32 (mma32 (const 0.0)")
        and core128.count("(mul (sfa_blk") == 1
    )
    assert (
        sexpr(SB.reduction_kblock(128, 0, 0, 32).canonical(alg)).count("(mul (sfa_blk")
        == 4
    )
    assert "(sfb_blk 1 0)" in sexpr(SB.reduction_kblock(32, 3, 33, 32).canonical(alg))
    assert sexpr(SB.reduction_kblock(128, 0, 0, 32).canonical(alg)) != sexpr(
        GemmSpec(128, "f32", in_fmt="e4m3_bs").acc(0, 0)
    )
    # the spaces and the acceptance at 32: the same records, whole 32-row blocks (the halves of the interleaved weight each), no
    # fused requantization
    assert (
        len(all_valid_configs_fp8bs(1280, 2560, 513, gemm1_chain("silu"), block=32))
        == 54
    )
    assert (
        len(all_valid_configs_fp8bs_swapped(1280, 2560, 513, gemm1_chain("silu"), 32))
        == 8
    )
    assert (
        all_valid_configs_fp8bs(
            1280, 2560, 513, gemm1_chain("silu", fused=True), block=32
        )
        == []
    )
    assert (
        len(
            all_valid_configs_fp8bs(1280 + 64, 2560, 513, gemm1_chain("silu"), block=32)
        )
        == 54
    )
    assert (
        all_valid_configs_fp8bs(1280 + 32, 2560, 513, gemm1_chain("silu"), block=32)
        == []
    )
    assert (
        all_valid_configs_fp8bs(1280, 2560 + 64, 513, gemm1_chain("silu"), block=32)
        == []
    )
    # the dispatcher's spelling rule: the fused epilogue for the ungated chain from 128 rows per expert at 128, never at 32 and
    # never for the gated chain (its block spans the two promotion warpgroups)
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import fused_quant_default

    assert fused_quant_default(4096, 8, 128, gated=False) and not fused_quant_default(
        4096, 8, 128, 32, gated=False
    )
    assert not fused_quant_default(4096, 8, 128) and not fused_quant_default(
        4096, 8, 128, 32
    )
    assert not fused_quant_default(64, 8, 128, gated=False) and not fused_quant_default(
        64, 8, 128, 32, gated=False
    )
    for T in (1, 7, 64, 300, 4096):
        for family, sm in (("sm_107", 212), ("sm_100", 148)):
            for act, gated in (("silu", True), ("relu2", False)):
                c1, c2 = default_configs_fp8bs(
                    T,
                    8,
                    2048,
                    768,
                    128,
                    sm,
                    family,
                    act,
                    gated,
                    (),
                    False,
                    nominal=True,
                    block=32,
                )
                assert can_implement_fp8bs(
                    c1,
                    2 * 768 if gated else 768,
                    2048,
                    128,
                    gemm1_chain(act, (), gated),
                    None,
                    32,
                ) and can_implement_fp8bs(c2, 2048, 768, 128, GEMM2_CHAIN, None, 32)
                if T <= 7:
                    assert c1.swap_ab and c2.swap_ab


def test_block32_weight_view_and_quantizer_reference():
    """``prepare_weights(..., deepseek_fp8_block_size=32)`` quantizes per 32x32 block by the class's reference at 32 into the
    pack on the device, writes the block into the view (read back by the runner: the key is required and must agree with the
    scales' shape), and the activation preparation at 32 is the quantizer's reference; the MXFP8 pair refuses the parameter;
    the quantizer's reference at 32 is the exact model block for block (device-free)."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import MoEWeightsE4M3BS

    g = torch.Generator().manual_seed(5)
    E, I, H = 3, 128, 256
    w1 = (
        torch.randn(E, 2 * I, H, generator=g)
        * torch.exp2(torch.randint(-6, 7, (E, 2 * I, H), generator=g).float())
    ).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    w1[1, 32:64, :32] = 0.0
    with pytest.raises(ValueError):
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
            activation=SwiGLU(),
            deepseek_fp8_block_size=32,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        MoEWeightsE4M3BS.prepare(w1, w2, block=64)
    # the quantizer at 32 on the CPU reference: the vectorised model is the exact model block for block
    x = _adversarial_rows(4, 128, seed=11)
    codes, s = RB.quantize_rows_reference(x, "K", 32)
    assert tuple(s.shape) == (4, 4)
    for r in range(4):
        for b in range(4):
            s_e, _inv, codes_e = RB.exact_block(
                x[r, b * 32 : (b + 1) * 32].float().tolist()
            )
            assert (
                codes_e == codes[r, b * 32 : (b + 1) * 32].tolist()
                and s_e == s[r, b].item()
            ), (r, b)
    if not _supported:
        return
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        activation=SwiGLU(),
        deepseek_fp8_block_size=32,
        device=torch.device("cuda"),
    )
    assert view["deepseek_fp8_block_size"] == 32
    W = MoEWeightsE4M3BS.from_view(view)
    assert W.block == 32 and W.in_fmt == "e4m3_bs32"
    assert tuple(W.w1_sf.shape) == (E, 2 * I // 32, H // 32) and tuple(
        W.w2_sf.shape
    ) == (
        E,
        H // 32,
        I // 32,
    )
    q1 = torch.stack(
        [RB.e4m3_bytes(RB.quantize_weight_reference(w1[e], 32)[0]) for e in range(E)]
    )
    s1 = torch.stack([RB.quantize_weight_reference(w1[e], 32)[1] for e in range(E)])
    assert torch.equal(
        W.w1.view(torch.uint8).cpu(), interleave_fc1(q1)
    ) and torch.equal(bits(W.w1_sf.cpu()), bits(s1))
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w2[e], 32)
        assert torch.equal(
            W.w2[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(W.w2_sf[e].cpu()), bits(rs)), e
    # the block is read off the scales' shape (a caller's quantization) and the view's key must agree with it; a view without
    # the key is refused (the reader takes no default)
    assert MoEWeightsE4M3BS.block_of(W.w1, W.w1_sf) == 32
    view2 = dict(view)
    del view2["deepseek_fp8_block_size"]
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="deepseek_fp8_block_size"
    ):
        MoEWeightsE4M3BS.from_view(view2)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="agree"):
        MoEWeightsE4M3BS.from_view(dict(view, deepseek_fp8_block_size=128))


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_block32_layer_matches_the_class_composition(act):
    """The 32-wide class end to end: the layer (from logits) == the class composed from its own pieces -- the quantizer at 32 on the
    device, the exact 32-wide core per (selected) expert (one instruction step per block, the exact fp32 promotion with the 32-row
    weight blocks), the CPU chain, the quantizer's reference on the device's bf16 intermediate at 32, GEMM2, the combine; every
    record of both spaces gives the same bits; the activation scales of the wrong block are refused."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    clusters = ((1, 1), (1, 2), (2, 1))
    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=5, gated=chain.gated, adversarial=True
    )
    xq, sx = _xq32(x)
    rq, rs = RB.quantize_rows_bytes(x, "MN", 32)
    assert torch.equal(xq.view(torch.uint8).cpu(), rq) and torch.equal(
        bits(sx.cpu()), bits(rs)
    )
    Wp = _weights32(w1, w2, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(xq, sx, logits), Wp)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    W = MoEWeightsE4M3BS.from_view(
        Wp.get_view(BatchInvariantFp8BlockRunner.backend_key)
    )
    assert W.block == 32
    ref_out, im = fused_moe_fp8bs(xq, sx, ids, weights, W, return_intermediates=True)
    assert not im["fused"] and im["a"] is not None
    assert torch.equal(bits(out), bits(ref_out))
    counts = im["counts"].cpu().tolist()
    Tk = T * k
    assert tuple(im["sx_perm"].shape) == (H // 32, Tk) and tuple(im["sa"].shape) == (
        I // 32,
        Tk,
    )
    sel = _selected(counts, 16)
    mask = torch.zeros(Tk, dtype=torch.bool)
    off = 0
    for e, c in enumerate(counts):
        if e in sel:
            mask[off : off + c] = True
        off += c
    N1 = W.w1.shape[1]
    acc1 = _exact_acc32(
        im["x_perm"],
        im["sx_perm"],
        W.w1,
        W.w1_sf,
        counts,
        RB.orig_rows(N1) if W.gated else None,
        sel,
        32,
    )
    a_ref = RB.chain_cpu(acc1, chain)
    a = im["a"].cpu()
    assert torch.equal(bits(a[mask]), bits(a_ref[mask])), (
        "gemm1",
        (a[mask] != a_ref[mask]).sum().item(),
    )
    aq_ref, sa_ref = RB.quantize_rows_bytes(a, "MN", 32)
    assert torch.equal(im["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(im["sa"].cpu()), bits(sa_ref)
    )
    acc2 = _exact_acc32(im["aq"], im["sa"], W.w2, W.w2_sf, counts, None, sel, 32)
    y = im["y"].cpu()
    assert torch.equal(bits(y[mask]), bits(R.TorchF32.cvt_out(acc2, "bf16")[mask])), (
        "gemm2"
    )
    assert _same_bits_or_both_nan(
        ref_out.cpu(), R.combine_cpu(im["y"], im["dest"], weights)
    )
    # every record of both spaces at the class (the fast clusters) gives the same bits
    ch1 = W.chain1(False)
    space1 = all_valid_configs_fp8bs(
        N1, H, E, ch1, clusters, block=32
    ) + all_valid_configs_fp8bs_swapped(N1, H, E, ch1, 32)
    space2 = all_valid_configs_fp8bs(
        H, I, E, block=32, clusters=clusters
    ) + all_valid_configs_fp8bs_swapped(H, I, E, block=32)
    assert len(space1) >= 18 + 8 and len(space2) >= 18 + 8
    ref = bits(ref_out.cpu())
    for c in space1:
        assert torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, weights, W, config1=c, config2=space2[0]
                ).cpu()
            ),
            ref,
        ), c.describe()
    for c in space2:
        assert torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, weights, W, config1=space1[0], config2=c
                ).cpu()
            ),
            ref,
        ), c.describe()
    # the fused requantization is not a spelling of this class; the 128-wide activation scales are refused with these weights
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        fused_moe_fp8bs(xq, sx, ids, weights, W, fuse_quant=True)
    xq128, sx128 = _xq(x)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(_logits_pack(xq128, sx128, logits), Wp)


@needs_device
def test_block32_tactics_and_graph_replay_keep_the_bits():
    """Every tactic the autotuner may pick at the 32-wide class computes the class; the autotuned layer and a CUDA-graph replay give
    the same bits; the 128-wide and the 32-wide layers are different classes on the same weights' values (their outputs differ)."""
    from flashinfer.autotuner import autotune

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    xq, sx = _xq32(x)
    W = _weights32(w1, w2)
    layer = _layer(k, E, I)
    pack = _logits_pack(xq, sx, logits)
    ref = bits(layer(pack, W))
    runner = layer.runners[0]
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics and all(len(t) == 15 and t[14] == 0 for t in tactics)
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        )
    # at 128 rows per expert (the 128-wide class's fused-epilogue regime) the 32-wide class still takes the standalone pass
    x_big, logits_big, _, _ = _problem(1024, k, H, I, E, seed=10)
    xq_big, sx_big = _xq32(x_big)
    pack_big = _logits_pack(xq_big, sx_big, logits_big)
    ref_big = bits(layer(pack_big, W))
    tactics_big = runner.get_valid_tactics(runner.pack_inputs(pack_big, W), None)
    assert tactics_big and all(t[14] == 0 for t in tactics_big)
    assert torch.equal(
        bits(runner.forward(runner.pack_inputs(pack_big, W), tactic=tactics_big[0])),
        ref_big,
    )
    layer.reset_winner()
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out_g = layer(pack, W)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out_g), ref)
    xq128, sx128 = _xq(x)
    out128 = _layer(k, E, I)(_logits_pack(xq128, sx128, logits), _weights(w1, w2))
    assert not torch.equal(bits(out128), ref)


# ----------------------------------------------------------------------------- the contract's rules (device-free where possible)


def _runner(
    quant=QUANT, k=2, E=8, I=128, method=RoutingMethodType.Renormalize, **experts
):
    return BatchInvariantFp8BlockRunner(
        MoEConfig(
            routing=RoutingConfig(num_experts=E, top_k=k, method=method),
            quant=quant,
            experts=ExpertConfig(intermediate_size=I, **experts),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        torch.device("cuda", 0),
    )


def test_runner_support_state_is_initialised_at_construction():
    """The routing term's state (``_method``, ``_grouped_other``, ``_logits_refusal``) and the pair's own (``_block``,
    ``_fused``) exist from construction, so no reader needs a default; a refusal deferred to the from-logits call is kept as
    the message's arguments (no exception object) and raised afresh there; the layer's protocol still orders the support
    check before the build."""
    import inspect

    from flashinfer.fused_moe.batch_invariant.runner import BatchInvariantMoeRunnerBase

    r = _runner()
    assert type(r) is BatchInvariantDeepSeekFp8Runner
    assert r._method is None and r._grouped_other is False
    assert r._logits_refusal is None and r._scale is None and r._part is None
    assert r._block == BLOCK and r._fused is False
    assert {"_method", "_grouped_other", "_logits_refusal", "_block", "_fused"} <= set(
        vars(r)
    )
    for cls in (
        BatchInvariantMoeRunnerBase,
        BatchInvariantFp8BlockRunner,
        BatchInvariantDeepSeekFp8Runner,
    ):
        assert "getattr(self, " not in inspect.getsource(cls), cls.__name__
    deferred = _runner(k=2, E=16, method=RoutingMethodType.Llama4)
    deferred._routing_term(deferred.config.routing, deferred.CLASS)
    assert deferred._method is None and isinstance(deferred._logits_refusal, tuple)
    assert "Llama4" in str(deferred._logits_refusal)
    for _ in range(2):  # raised afresh each time, from the stored arguments
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="Llama4"):
            raise bi.BatchInvariantUnsupportedError(*deferred._logits_refusal)
    with pytest.raises(RuntimeError):
        r.build()
    if _supported:
        r.check_support()
        assert r._method is not None and r._logits_refusal is None
        r.build()
        assert r._part is not None


def test_fingerprint_refuses_the_shapes_the_layer_refuses():
    """``fingerprint("fused_moe_fp8bs", ...)`` applies the class's complete shape rule: ``H`` and ``I`` whole scale blocks,
    the bounds on ``E`` and ``top_k``, a block outside the two classes, the runners' Llama4 top-1 rule and the group
    dimensions of the DeepSeekV3 term; the model shapes keep their receipts (the block a dimension, the scales' values none,
    no architecture dimension: the fp8 instruction folds the same elements on every part)."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import check_shape_fp8bs

    op = "fused_moe_fp8bs"
    base = dict(H=4096, I=1536, top_k=8, num_experts=128)
    for bad, match in (
        (dict(H=4160), "multiple of 128"),
        (dict(I=1600), "multiple of 128"),
        (dict(num_experts=1025), "num_experts"),
        (dict(top_k=17), "top_k"),
        (dict(top_k=0), "top_k"),
        (dict(deepseek_fp8_block_size=64), "deepseek_fp8_block_size"),
        (dict(method="Llama4", top_k=2), "Llama4 routing takes top_k = 1"),
    ):
        with pytest.raises(ValueError, match=match):
            bi.fingerprint(op, **dict(base, **bad))
    for H, I, E, k in ((4160, 1536, 128, 8), (4096, 1536, 1025, 1), (4096, 1536, 8, 9)):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            check_shape_fp8bs(H, I, E, k)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        check_shape_fp8bs(4096, 1536, 128, 8, block=64)
    assert bi.fingerprint(op, **base) == "moe-bf8fe148b87e294b"
    assert bi.fingerprint(op, act=ReLU2(), **base) == "moe-46b16d9ca71db7d2"
    assert bi.fingerprint(op, act=PowLU(), **base) == "moe-288f2c3c4deb96f0"
    assert bi.fingerprint(op, method="Default", **base) == "moe-32331319f4f39b8f"
    dsv3 = dict(H=7168, I=2048, top_k=8, num_experts=256)
    assert bi.fingerprint(op, **dsv3) == "moe-dbdaa5ec287b8769"
    assert (
        bi.fingerprint(op, deepseek_fp8_block_size=32, **base) == "moe-c99964edb2481aae"
    )
    assert (
        bi.fingerprint(op, act=ReLU2(), deepseek_fp8_block_size=32, **base)
        == "moe-ea6fa2f7a3209262"
    )
    assert (
        bi.fingerprint(op, deepseek_fp8_block_size=32, **dsv3) == "moe-ec90445f4ea47fdd"
    )
    grouped = bi.fingerprint(op, method="DeepSeekV3", n_group=8, topk_group=4, **base)
    assert grouped != bi.fingerprint(op, method="DeepSeekV3", **base)
    with pytest.raises(ValueError, match="DeepSeekV3"):
        bi.fingerprint(op, method="Renormalize", n_group=8, topk_group=4, **base)
    one = dict(base, top_k=1)
    assert bi.fingerprint(op, method="Llama4", **one) != bi.fingerprint(op, **one)


def test_pack_layout_and_tactic_fields_are_pinned():
    """The packed inputs are ``[xq, sx, ids, wts, w1, w1_sf, w2, w2_sf, out]`` followed by the plan: the head counts the base
    runner slices by, the weight tensors in the order the view reader rebuilds them (the block read off the scales); the
    tactic is the base runner's seven scalars per record plus the requantization spelling, round-trips every record kind,
    and a tuple of another length is refused."""
    from flashinfer.fused_moe.batch_invariant import runner_fp8bs as RF
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import MoEWeightsE4M3BS
    from flashinfer.fused_moe.batch_invariant.runner import _TACTIC_FIELDS
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        grouped_knobs_fp8bs,
        swapped_knobs_fp8bs,
    )

    r = _runner(k=8, E=128, I=768)
    assert (r.N_ACT_TENSORS, r.N_WEIGHT_TENSORS) == (2, 4)
    positions = (
        RF._X,
        RF._SX,
        RF._IDS,
        RF._W,
        RF._W1,
        RF._W1SF,
        RF._W2,
        RF._W2SF,
        RF._OUT,
    )
    assert positions == tuple(range(9))
    assert r.N_ACT_TENSORS + 2 + r.N_WEIGHT_TENSORS + 1 == 9
    W = MoEWeightsE4M3BS(
        torch.zeros(1, 256, 128, dtype=torch.float8_e4m3fn),
        torch.ones(1, 2, 1),
        torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
        torch.ones(1, 1, 1),
        block=BLOCK,
    )
    ts = r._weight_tensors(W)
    assert all(t is u for t, u in zip(ts, (W.w1, W.w1_sf, W.w2, W.w2_sf), strict=True))
    a = r._activation()
    W2 = r._weights_from_tensors(a, ts)
    assert (W2.w1, W2.w1_sf, W2.w2, W2.w2_sf) == (W.w1, W.w1_sf, W.w2, W.w2_sf)
    assert (W2.gated, W2.act, W2.params, W2.block) == (a.gated, a.act, a.params, BLOCK)
    assert r._block == BLOCK
    W32 = MoEWeightsE4M3BS(
        W.w1, torch.ones(1, 8, 4), W.w2, torch.ones(1, 4, 4), block=32
    )
    assert r._weights_from_tensors(a, r._weight_tensors(W32)).block == 32
    assert r._block == 32
    dense, box = grouped_knobs_fp8bs()[0], swapped_knobs_fp8bs()[0]
    for c1, c2 in ((dense, dense), (box, dense), (box, box)):
        for fused in (False, True):
            t = RF._tactic_bs(c1, c2, fused)
            assert len(t) == 2 * _TACTIC_FIELDS + 1 == 15
            assert all(isinstance(v, (int, str)) for v in t)
            assert RF._configs_from_tactic_bs(t) == (c1, c2, fused)
    with pytest.raises(ValueError, match="tactic"):
        RF._configs_from_tactic_bs(t[:-1])
    with pytest.raises(ValueError, match="tactic"):
        RF._configs_from_tactic_bs(t + (0,))
    assert r.TACTIC_TAG.startswith("moe.e4m3bs")


def test_pack_contract_flags_and_shards_are_refused_at_support():
    """The configuration's rules the pack contract states, refused at the support check before any device is consulted:
    ``QuantConfig(swizzled_scale_factors=True)`` names a layout the class does not take (the fp32 block scales have one
    layout), ``per_token_scale=True`` a scale the pack has none of (the layer's shared rule), an expert-parallel shard is
    refused by name in the class's terms, fused shared experts by the layer's rule; the runner declares neither."""
    assert BatchInvariantDeepSeekFp8Runner.supports_per_token_scale is False
    assert BatchInvariantDeepSeekFp8Runner.supports_expert_parallelism is False
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="swizzled_scale_factors"
    ) as e:
        _runner(
            dataclasses.replace(QUANT, swizzled_scale_factors=True)
        )._check_support()
    assert "moe.e4m3bs" in str(e.value) and "DeepSeek" in str(e.value)
    _runner(dataclasses.replace(QUANT, swizzled_scale_factors=False))
    with pytest.raises(
        (NotImplementedError, bi.BatchInvariantUnsupportedError),
        match="per_token_scale",
    ):
        _runner(dataclasses.replace(QUANT, per_token_scale=True))._check_support()
    with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
        _runner(local_expert_offset=4, local_num_experts=4)._check_support()
    assert "moe.e4m3bs" in str(e.value) and "expert-parallel" in str(e.value)
    with pytest.raises(NotImplementedError, match="shared experts"):
        _runner(
            method=RoutingMethodType.DeepSeekV3, num_fused_shared_experts=1
        )._check_support()


def test_prepare_weights_rejects_dimensions_that_disagree_with_the_weights():
    """A supplied ``num_local_experts`` / ``hidden_size`` / ``intermediate_size`` must be the weights' own, for the gated
    geometry (``I`` = half of w1's rows) and the ungated one (``I`` = w1's rows): the mode's error before any view is
    built, naming the dimension; weights on the CPU without a destination device are refused (the pack lives on a CUDA
    device); the consistent call and the call without dimensions give the view (on a device), naming the pair and the
    block."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        VIEW_PAIR,
        MoEWeightsE4M3BS,
    )

    E, I, H = 4, 128, 256
    w1 = torch.randn(E, 2 * I, H).to(torch.bfloat16)
    w2 = torch.randn(E, H, I).to(torch.bfloat16)
    w1u = torch.randn(E, I, H).to(torch.bfloat16)
    for w1_, act, kw in (
        (w1, _SWIGLU, dict(num_local_experts=E + 1)),
        (w1, _SWIGLU, dict(hidden_size=H // 2)),
        (w1, _SWIGLU, dict(intermediate_size=2 * I)),
        (w1u, ReLU2(), dict(intermediate_size=I // 2)),
    ):
        with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
            BatchInvariantFp8BlockConfig.prepare_weights(
                w1_, w2, quant=QUANT, activation=act, **kw
            )
        assert next(iter(kw)) in str(excinfo.value) and "moe.e4m3bs" in str(
            excinfo.value
        )
    with pytest.raises(TypeError):
        BatchInvariantFp8BlockConfig.prepare_weights(w1[0], w2, quant=QUANT)
    with pytest.raises(TypeError):
        MoEWeightsE4M3BS.prepare(w1.to(torch.int32), w2)
    with pytest.raises(ValueError, match="does not match"):
        MoEWeightsE4M3BS.prepare(w1, w2[:, :, : I // 2])
    for call in (
        lambda: MoEWeightsE4M3BS.prepare(w1, w2),
        lambda: BatchInvariantFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=QUANT,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size=I,
        ),
    ):
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="device"
        ) as excinfo:
            call()
        assert "moe.e4m3bs" in str(excinfo.value)
    if not _supported:
        return
    w1, w2, w1u = w1.cuda(), w2.cuda(), w1u.cuda()
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1, w2, quant=QUANT, num_local_experts=E, hidden_size=H, intermediate_size=I
    )
    assert view["w1"].shape == (E, 2 * I, H) and view["gated"]
    assert view["quant"] == VIEW_PAIR and view["deepseek_fp8_block_size"] == BLOCK
    assert torch.equal(
        BatchInvariantFp8BlockConfig.prepare_weights(w1, w2, quant=QUANT)["w2"].view(
            torch.uint8
        ),
        view["w2"].view(torch.uint8),
    )
    view_u = BatchInvariantFp8BlockConfig.prepare_weights(
        w1u, w2, quant=QUANT, intermediate_size=I, activation=ReLU2()
    )
    assert not view_u["gated"] and view_u["w1"].shape == (E, I, H)


def test_trtllm_view_scales_are_copied_and_validated():
    """``from_trtllm_view`` takes FlashInfer's DeepSeek view's codes as given and its block scales as copies: a scale that is
    not finite, positive and normal (NaN, zero, negative, infinite, subnormal) is refused naming the expert before any device
    is consulted, a scale tensor of another shape too; the pack lives on a CUDA device (the view's, or ``device=``); a later
    write to the caller's scales does not reach the pack, and scales on another device than the codes are moved to the
    pack's device (a placement: the same bits)."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import MoEWeightsE4M3BS

    E, I, H = 3, 128, 256
    g = torch.Generator().manual_seed(11)
    w1 = (torch.randn(E, 2 * I, H, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    fi_view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        activation=SwiGLU(),
        device=torch.device("cpu"),
    )
    s1 = fi_view["gemm1_weights_scale"].float()
    s2 = fi_view["gemm2_weights_scale"].float()
    for value, expert, which in (
        (float("nan"), 1, "gemm1"),
        (0.0, 2, "gemm1"),
        (-1.0, 0, "gemm2"),
        (float("inf"), 1, "gemm2"),
        (1e-39, 2, "gemm1"),
    ):
        s = (s1 if which == "gemm1" else s2).clone()
        s[expert].view(-1)[1] = value
        bad = dict(fi_view, **{f"{which}_weights_scale": s})
        with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
            BatchInvariantFp8BlockConfig.from_trtllm_view(bad, quant=QUANT)
        assert f"expert {expert}" in str(e.value) and "finite" in str(e.value), (
            value,
            which,
        )
    with pytest.raises(ValueError, match="block scales"):
        BatchInvariantFp8BlockConfig.from_trtllm_view(
            dict(fi_view, gemm1_weights_scale=s1[:, :, :1].contiguous()), quant=QUANT
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="device"):
        BatchInvariantFp8BlockConfig.from_trtllm_view(fi_view, quant=QUANT)
    if not _supported:
        return
    W = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.from_trtllm_view(
            fi_view, quant=QUANT, device=torch.device("cuda")
        )
    )
    assert all(t.is_cuda for t in (W.w1, W.w1_sf, W.w2, W.w2_sf))
    assert torch.equal(bits(W.w1_sf.cpu()), bits(s1)) and torch.equal(
        bits(W.w2_sf.cpu()), bits(s2)
    )
    cuda_view = {
        k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in fi_view.items()
    }
    W2 = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.from_trtllm_view(cuda_view, quant=QUANT)
    )
    assert W2.w1_sf.data_ptr() != cuda_view["gemm1_weights_scale"].data_ptr()
    assert W2.w2_sf.data_ptr() != cuda_view["gemm2_weights_scale"].data_ptr()
    cuda_view["gemm1_weights_scale"].fill_(7.0)
    cuda_view["gemm2_weights_scale"].fill_(7.0)
    assert torch.equal(bits(W2.w1_sf.cpu()), bits(s1)) and torch.equal(
        bits(W2.w2_sf.cpu()), bits(s2)
    )
    assert torch.equal(W2.w1.view(torch.uint8), W.w1.view(torch.uint8))
    other = (
        torch.device(
            "cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count()
        )
        if torch.cuda.device_count() > 1
        else torch.device("cpu")
    )
    mixed = dict(
        cuda_view, gemm1_weights_scale=s1.to(other), gemm2_weights_scale=s2.to(other)
    )
    W3 = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.from_trtllm_view(mixed, quant=QUANT)
    )
    assert W3.w1_sf.device == W2.w1_sf.device
    assert torch.equal(bits(W3.w1_sf), bits(W2.w1_sf)) and torch.equal(
        bits(W3.w2_sf), bits(W2.w2_sf)
    )


def test_grouped_gemm_capacity_checks_are_host_errors():
    """Every buffer the kernels write or the plan's map they read is checked on the host before any record or kernel is
    consulted (device-free: CPU tensors): an output of another row count or dtype, a scale tensor of another shape, dtype
    or layout, weight scales that do not tile the weight, the fused requantization's scale output, the gather form's slot
    map shorter than the live rows, and the gather form with a grouped record or the fused requantization; the fused entry
    refuses an output of another shape and a scale tensor of the token-major layout."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        TAIL_ROWS,
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
        grouped_gemm_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import (
        GEMM2_CHAIN,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        can_implement_grouped_fp8bs,
        grouped_knobs_fp8bs,
        swapped_knobs_fp8bs,
    )

    E, N, K, Tk = 4, 256, 256, 16
    rows = Tk + TAIL_ROWS
    a = torch.zeros(rows, K, dtype=torch.float8_e4m3fn)
    sfa = torch.ones(K // BLOCK, rows)
    w = torch.zeros(E, N, K, dtype=torch.float8_e4m3fn)
    w_sf = torch.ones(E, N // BLOCK, K // BLOCK)
    cnt = torch.zeros(E, dtype=torch.int32)
    cfg = next(
        c
        for c in grouped_knobs_fp8bs()
        if can_implement_grouped_fp8bs(c, N, K, E, GEMM2_CHAIN)
    )
    good = dict(out=torch.empty(rows, N, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="out must be"):
        grouped_gemm_fp8bs(
            a, sfa, w, w_sf, cnt, cfg, GEMM2_CHAIN,
            out=torch.empty(rows - 1, N, dtype=torch.bfloat16),
        )  # fmt: skip
    with pytest.raises(ValueError, match="out must be"):
        grouped_gemm_fp8bs(
            a, sfa, w, w_sf, cnt, cfg, GEMM2_CHAIN,
            out=torch.empty(rows, N, dtype=torch.float32),
        )  # fmt: skip
    for bad in (sfa[:, :-1], sfa.t().contiguous(), sfa.to(torch.bfloat16)):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="sfa"):
            grouped_gemm_fp8bs(a, bad, w, w_sf, cnt, cfg, GEMM2_CHAIN, **good)
    with pytest.raises(ValueError, match="w_sf"):
        grouped_gemm_fp8bs(a, sfa, w, w_sf[:, :, :1], cnt, cfg, GEMM2_CHAIN, **good)
    gq = gemm1_chain("silu", fused=True)
    cfgq = next(
        c for c in grouped_knobs_fp8bs() if can_implement_grouped_fp8bs(c, N, K, E, gq)
    )
    with pytest.raises(ValueError, match="sfo"):
        grouped_gemm_fp8bs(
            a, sfa, w, w_sf, cnt, cfgq, gq, sfo=torch.empty(N // 2 // BLOCK, rows - 1)
        )
    with pytest.raises(ValueError, match="out must be"):
        grouped_gemm_fp8bs(
            a, sfa, w, w_sf, cnt, cfgq, gq,
            out=torch.empty(rows, N // 2, dtype=torch.bfloat16),
        )  # fmt: skip
    box = swapped_knobs_fp8bs()[0]
    x = torch.zeros(4, K, dtype=torch.float8_e4m3fn)
    sx = torch.ones(K // BLOCK, 4)
    g1 = gemm1_chain("silu")
    src = torch.zeros(Tk, dtype=torch.int32)
    out1 = torch.empty(rows, N // 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="src"):
        grouped_gemm_fp8bs(
            x, sx, w, w_sf, cnt, box, g1, out=out1, src=src[:-1], k_div=4, rows_out=rows
        )
    with pytest.raises(ValueError, match="gather form"):
        grouped_gemm_fp8bs(
            x, sx, w, w_sf, cnt, cfg, g1, out=out1, src=src, k_div=4, rows_out=rows
        )
    with pytest.raises(ValueError, match="gather form"):
        grouped_gemm_fp8bs(
            x, sx, w, w_sf, cnt, box, gq, out=out1, src=src, k_div=4, rows_out=rows
        )
    W = MoEWeightsE4M3BS(
        w, w_sf,
        torch.zeros(E, K, N // 2, dtype=torch.float8_e4m3fn),
        torch.ones(E, K // BLOCK, N // 2 // BLOCK),
    )  # fmt: skip
    T = 8
    xq = torch.zeros(T, K, dtype=torch.float8_e4m3fn)
    ids = torch.zeros(T, 2, dtype=torch.int32)
    wts = torch.ones(T, 2)
    with pytest.raises(ValueError, match="out must be"):
        fused_moe_fp8bs(
            xq, torch.ones(K // BLOCK, T), ids, wts, W,
            out=torch.empty(T - 1, K, dtype=torch.bfloat16),
        )  # fmt: skip
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="hidden_states_scale"):
        fused_moe_fp8bs(xq, torch.ones(T, K // BLOCK), ids, wts, W)


# ----------------------------------------------------------------------------- GPU: the contract's rules


def _misaligned(t: torch.Tensor, off: int) -> torch.Tensor:
    """``t``'s bytes at a storage offset of ``off`` bytes (contiguous, the same shape and dtype)."""
    es, n = t.element_size(), t.numel()
    big = torch.empty(n * es + 16, dtype=torch.uint8, device="cuda")
    v = big[off : off + n * es].view(t.dtype).view(t.shape)
    v.copy_(t)
    assert v.data_ptr() % 16 == off and v.is_contiguous()
    return v


def _poison_allocator(byte: int):
    """Fill the caching allocator's cached blocks (the large pool and 256 small-pool segments) with one byte pattern, so the
    ``torch.empty`` intermediates of the next op call start with it (0x7F / 0xFF: e4m3 and bf16 NaN patterns; 0x00: zeros)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]
    blocks = [
        torch.empty(
            int(min(free // 2, 8 << 30)), dtype=torch.uint8, device="cuda"
        ).fill_(byte)
    ]
    blocks += [
        torch.empty(1 << 19, dtype=torch.uint8, device="cuda").fill_(byte)
        for _ in range(256)
    ]
    torch.cuda.synchronize()
    del blocks


def _family() -> str:
    from flashinfer.gemm.batch_invariant.api import part_of

    return part_of(torch.device("cuda"))[1]


def _fills(fn) -> int:
    """The fill launches and host copies among the CUDA kernels of ``fn()`` (the profiler's view)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    return sum(1 for n in names if "Fill" in n or "fill" in n or "Memcpy" in n)


def _capture(layer, act, W):
    """The layer's forward captured in a CUDA graph on a side stream after one warm call there: ``(graph, out)``."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        layer(act, W)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = layer(act, W)
    torch.cuda.synchronize()
    return g, out


def _form_of(runner, T: int) -> str:
    """The form the dispatcher picks at ``T`` tokens (the op's rule on the first picks): GEMM1's gather form of a narrow
    swapped record, the box form of a wide one, the padded body with the standalone requantization or with the fused one."""
    from flashinfer.fused_moe.batch_invariant.api import GATHER_TOKEN_TILE_MAX

    c1, _, fused = runner._default(T)
    if c1.swap_ab:
        return "gather" if not fused and c1.tile_n <= GATHER_TOKEN_TILE_MAX else "box"
    return "fused" if fused else "padded"


@needs_device
def test_prepare_weights_quantizes_one_expert_at_a_time_into_the_pack():
    """The pack is preallocated and filled one expert at a time: the bits of the per-expert quantizer (every expert's blocks
    quantized alone, its gated rows interleaved, stacked), and a peak memory beyond the inputs bounded by the pack plus a
    few experts' temporaries, below one full-size bf16 copy of the experts -- for contiguous weights and for a strided view
    of them, with the same bits; and through ``prepare_weights`` for CPU-resident bf16 weights moved to the device, for fp32
    weights on the device and for CPU fp32 weights (each expert's slice is moved and cast right before its quantization,
    never a full-size copy), with the bits of the same-device bf16 path and the same bound."""
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        quantize_weight_blocks,
    )

    E, I, H = 32, 512, 1024
    _, _, w1, w2 = _problem(4, 2, H, I, E, seed=81)
    q1, s1, q2, s2 = [], [], [], []
    for e in range(E):
        c1, f1 = quantize_weight_blocks(w1[e])
        c2, f2 = quantize_weight_blocks(w2[e])
        q1.append(interleave_fc1(c1.unsqueeze(0))[0].view(torch.uint8))
        s1.append(f1)
        q2.append(c2.view(torch.uint8))
        s2.append(f2)
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    W = MoEWeightsE4M3BS.prepare(w1, w2)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert torch.equal(u8(W.w1), torch.stack(q1)) and torch.equal(
        bits(W.w1_sf), bits(torch.stack(s1))
    )
    assert torch.equal(u8(W.w2), torch.stack(q2)) and torch.equal(
        bits(W.w2_sf), bits(torch.stack(s2))
    )
    assert W.w1.dtype == W.w2.dtype == torch.float8_e4m3fn
    pack = sum(t.numel() * t.element_size() for t in (W.w1, W.w1_sf, W.w2, W.w2_sf))
    expert = (w1[0].numel() + w2[0].numel()) * w1.element_size()
    full = (w1.numel() + w2.numel()) * w1.element_size()
    assert pack + 8 * expert < full  # the bound separates the paths
    assert peak <= pack + 8 * expert, (peak, pack, expert)
    same = lambda a, b: all(  # noqa: E731
        torch.equal(u8(t), u8(u))
        for t, u in ((a.w1, b.w1), (a.w1_sf, b.w1_sf), (a.w2, b.w2), (a.w2_sf, b.w2_sf))
    )
    w1s = w1.transpose(1, 2).contiguous().transpose(1, 2)
    w2s = w2.transpose(1, 2).contiguous().transpose(1, 2)
    assert not w1s.is_contiguous() and torch.equal(w1s, w1) and torch.equal(w2s, w2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    Ws = MoEWeightsE4M3BS.prepare(w1s, w2s)
    torch.cuda.synchronize()
    peak_s = torch.cuda.max_memory_allocated() - before
    assert same(Ws, W)
    assert peak_s <= pack + 8 * expert, (peak_s, pack, expert)
    assert 2 * full > pack + 8 * expert
    for name, a, b in (
        ("cpu bf16", w1.cpu(), w2.cpu()),
        ("fp32", w1.float(), w2.float()),
        ("cpu fp32", w1.cpu().float(), w2.cpu().float()),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        view = BatchInvariantFp8BlockConfig.prepare_weights(
            a, b, quant=QUANT, device=torch.device("cuda")
        )
        torch.cuda.synchronize()
        peak_d = torch.cuda.max_memory_allocated() - before
        Wd = MoEWeightsE4M3BS.from_view(view)
        assert all(t.is_cuda for t in (Wd.w1, Wd.w1_sf, Wd.w2, Wd.w2_sf)), name
        assert same(Wd, W), name
        assert peak_d <= pack + 8 * expert, (name, peak_d, pack, expert)


@needs_device
def test_weights_with_a_nan_element_quantize_to_nan_codes_in_their_block():
    """The class's own quantization of the experts: a NaN weight gives the NaN code in its place and its block's scale the
    chain's floor (the block amax takes the finite operand), every other block the clean bits -- the NaN propagates, nothing
    is rewritten; an all-zero expert quantizes to zero codes at the floor scale (finite, positive, normal); an infinite
    weight makes its block's scale infinite and is refused naming the expert."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import MoEWeightsE4M3BS
    from flashinfer.fused_moe.batch_invariant.spec_fp8bs import SCALE_EPS

    E, I, H = 2, 256, 256
    _, _, w1, w2 = _problem(4, 2, H, I, E, seed=5, gated=False)
    clean = MoEWeightsE4M3BS.prepare(w1, w2, gated=False)
    w1n = w1.clone()
    w1n[1, 130, 7] = float("nan")  # block row 1, block column 0 of expert 1
    W = MoEWeightsE4M3BS.prepare(w1n, w2, gated=False)
    assert torch.isfinite(W.w1_sf).all() and bool((W.w1_sf > 0).all())
    assert W.w1_sf[1, 1, 0].item() == SCALE_EPS
    other = torch.ones_like(W.w1_sf, dtype=torch.bool)
    other[1, 1, 0] = False
    assert torch.equal(bits(W.w1_sf)[other], bits(clean.w1_sf)[other])
    codes, ref = W.w1.view(torch.uint8), clean.w1.view(torch.uint8)
    assert codes[1, 130, 7].item() == 0x7F  # the e4m3 NaN byte
    block = torch.zeros_like(codes, dtype=torch.bool)
    block[1, 128:256, 0:128] = True
    assert torch.equal(codes[~block], ref[~block])
    inside = codes[block].view(128, 128)
    inside[2, 7] = 0
    assert not torch.isnan(inside.view(torch.float8_e4m3fn).float()).any()
    assert torch.equal(W.w2.view(torch.uint8), clean.w2.view(torch.uint8))
    w1z = w1.clone()
    w1z[0] = 0.0
    Wz = MoEWeightsE4M3BS.prepare(w1z, w2, gated=False)
    assert (
        bool((Wz.w1_sf[0] == SCALE_EPS).all()) and not Wz.w1[0].view(torch.uint8).any()
    )
    assert torch.equal(Wz.w1[1].view(torch.uint8), ref[1])
    w1i = w1.clone()
    w1i[1, 5, 9] = float("inf")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="expert 1") as e:
        MoEWeightsE4M3BS.prepare(w1i, w2, gated=False)
    assert "finite" in str(e.value)


@needs_device
def test_weight_view_names_its_pair_and_a_view_without_it_is_declined():
    """``prepare_weights`` writes the pair into the view (``quant``) and the block; the runner's candidate filter requires the
    pair -- a view without the key or of the MXFP8 pair is declined, never read as this pair's -- the runner driven directly
    refuses it with the mode's error naming the key, the view reader takes no default for the pair or the block."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        VIEW_PAIR,
        MoEWeightsE4M3BS,
    )

    T, k, H, I, E = 8, 2, 256, 128, 4
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=73)
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1, w2, quant=QUANT, activation=_SWIGLU
    )
    assert view["quant"] == VIEW_PAIR == "deepseek_fp8"
    assert MoEWeightsE4M3BS.prepare(w1, w2).view()["quant"] == VIEW_PAIR
    layer = _layer(k, E, I)
    runner = layer.runners[0]
    xq, sx = _xq(x)
    act = _logits_pack(xq, sx, logits)
    pack = MoEWeightPack({})
    pack.prepare_for(KEY, view)
    assert runner.accepts(act, pack)
    ref = bits(layer(act, pack))
    for bad in (
        {kk: v for kk, v in view.items() if kk != "quant"},
        dict(view, quant="mxfp8"),
    ):
        other = MoEWeightPack({})
        other.prepare_for(KEY, bad)
        assert not runner.accepts(act, other)
        with pytest.raises(NotImplementedError):
            layer(act, other)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="quant"):
            runner.pack_inputs(act, other)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="quant"):
            MoEWeightsE4M3BS.from_view(other.get_view(KEY))
    no_block = MoEWeightPack({})
    no_block.prepare_for(
        KEY, {kk: v for kk, v in view.items() if kk != "deepseek_fp8_block_size"}
    )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="deepseek_fp8_block_size"
    ):
        runner.pack_inputs(act, no_block)
    assert torch.equal(bits(layer(act, pack)), ref)


@needs_device
def test_misaligned_inputs_are_refused_with_the_alignment_named():
    """Quantized rows or scales whose storage offset is not a multiple of 16 bytes (the kernels' vector loads and TMA boxes)
    are refused by the layer and by the op with the mode's error naming the alignment, before any kernel; so are the raw
    bf16 rows; the aligned pack runs."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    T, k, H, I, E = 40, 4, 256, 128, 8
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5)
    layer, W_pack = _layer(k, E, I), _weights(w1, w2)
    W = MoEWeightsE4M3BS.from_view(W_pack.get_view(KEY))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    xq, sx = _xq(x)
    ref = bits(fused_moe_fp8bs(xq, sx, ids, wts, W))
    for q, s in ((_misaligned(xq, 8), sx), (xq, _misaligned(sx, 4))):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="16 bytes"):
            layer(_routed_pack(q, s, ids, wts), W_pack)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="16 bytes"):
            fused_moe_fp8bs(q, s, ids, wts, W)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="16 bytes"):
        layer(_routed_pack(_misaligned(x, 8), None, ids, wts), W_pack)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="16 bytes"):
        fused_moe_fp8bs(_misaligned(x, 8), None, ids, wts, W)
    assert torch.equal(bits(layer(_routed_pack(xq, sx, ids, wts), W_pack)), ref)
    assert torch.equal(bits(layer(_routed_pack(x, None, ids, wts), W_pack)), ref)


@needs_device
def test_scale_layout_is_taken_as_given_and_the_empty_pack_runs():
    """The pack's scales are the DeepSeek layout ``[H / 128, T]``, taken as given: the token-major ``[T, H / 128]`` tensor,
    rows of another count, a 3-D tensor, another dtype and a flat tensor are refused by the layer and the op with the mode's
    error naming the layout, before any kernel; a strided tensor of the layout is copied once (a placement: the bits). The
    empty pack (``T = 0``) of any empty scale shape, and the empty raw-bf16 pack, give the empty ``[0, H]`` output through
    the layer (from logits and pre-routed) and the op."""
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    T, k, H, I, E = 48, 2, 512, 128, 8
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=734)
    layer, W_pack = _layer(k, E, I), _weights(w1, w2)
    W = MoEWeightsE4M3BS.from_view(W_pack.get_view(KEY))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    xq, sx = _xq(x)
    KB = H // BLOCK
    assert tuple(sx.shape) == (KB, T) and T != KB
    ref = bits(fused_moe_fp8bs(xq, sx, ids, wts, W))
    for bad in (
        sx.t().contiguous(),
        sx[:-1].contiguous(),
        sx.reshape(KB, 2, T // 2),
        sx.to(torch.bfloat16),
        torch.ones(KB * T, device="cuda"),
    ):
        with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
            layer(_routed_pack(xq, bad, ids, wts), W_pack)
        assert "hidden_states_scale" in str(e.value)
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            fused_moe_fp8bs(xq, bad, ids, wts, W)
    strided = sx.t().contiguous().t()
    assert not strided.is_contiguous() and tuple(strided.shape) == (KB, T)
    assert torch.equal(bits(layer(_routed_pack(xq, strided, ids, wts), W_pack)), ref)
    assert torch.equal(bits(fused_moe_fp8bs(xq, strided, ids, wts, W)), ref)
    assert torch.equal(bits(layer(_routed_pack(xq, sx, ids, wts), W_pack)), ref)
    x0 = torch.empty(0, H, dtype=torch.float8_e4m3fn, device="cuda")
    for s0 in (
        torch.empty(KB, 0, device="cuda"),
        torch.empty(0, device="cuda"),
        torch.empty(0, 5, device="cuda"),
    ):
        out = fused_moe_fp8bs(x0, s0, ids[:0], wts[:0], W)
        assert out.shape == (0, H) and out.dtype == torch.bfloat16
        assert layer(_routed_pack(x0, s0, ids[:0], wts[:0]), W_pack).shape == (0, H)
        assert layer(_logits_pack(x0, s0, logits[:0]), W_pack).shape == (0, H)
    xb0 = torch.empty(0, H, dtype=torch.bfloat16, device="cuda")
    assert fused_moe_fp8bs(xb0, None, ids[:0], wts[:0], W).shape == (0, H)
    assert layer(_routed_pack(xb0, None, ids[:0], wts[:0]), W_pack).shape == (0, H)
    assert layer(_logits_pack(xb0, None, logits[:0]), W_pack).shape == (0, H)


@needs_device
def test_tactic_replays_on_every_pack_form_independent_of_the_ranking():
    """A tactic (two records and the requantization spelling) is valid on every pack form of the pair -- the e4m3 pair and
    the raw bf16 rows, from logits and pre-routed -- so a winner the layer keys by token bucket replays whatever pack the
    bucket next brings: pairs built from the record spaces (a narrow swapped record whose GEMM1 takes the gather form, a
    wide one taking the box form, grouped records with the standalone and the fused requantization, and the mixed pairs)
    give the eager bits on every form, independently of which records the part's ranking places."""
    from flashinfer.fused_moe.batch_invariant import runner_fp8bs as RF
    from flashinfer.fused_moe.batch_invariant.api import GATHER_TOKEN_TILE_MAX
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_fp8bs import (
        GEMM2_CHAIN,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        can_implement_grouped_fp8bs,
        can_implement_swapped_fp8bs,
        grouped_knobs_fp8bs,
        swapped_knobs_fp8bs,
    )

    T, k, H, I, E = 16, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=503, adversarial=True)
    layer, W = _layer(k, E, I, tune_max=16), _weights(w1, w2)
    runner = layer.runners[0]
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    xq, sx = _xq(x)
    packs = {
        "logits": _logits_pack(xq, sx, logits),
        "routed": _routed_pack(xq, sx, ids.cuda(), wts.cuda()),
        "bf16 logits": _logits_pack(x, None, logits),
        "bf16 routed": _routed_pack(x, None, ids.cuda(), wts.cuda()),
    }
    ref = bits(layer(packs["logits"], W))
    for name, p in packs.items():
        assert torch.equal(bits(layer(p, W)), ref), name
    g1, g1q = gemm1_chain("silu"), gemm1_chain("silu", fused=True)
    dense = next(
        c
        for c in grouped_knobs_fp8bs()
        if can_implement_grouped_fp8bs(c, 2 * I, H, E, g1)
        and can_implement_grouped_fp8bs(c, H, I, E, GEMM2_CHAIN)
    )
    dense_q = next(
        c
        for c in grouped_knobs_fp8bs()
        if can_implement_grouped_fp8bs(c, 2 * I, H, E, g1q)
    )
    swapped = [
        c
        for c in swapped_knobs_fp8bs()
        if can_implement_swapped_fp8bs(c, 2 * I, H, E, g1)
        and can_implement_swapped_fp8bs(c, H, I, E, GEMM2_CHAIN)
    ]
    narrow = next(c for c in swapped if c.tile_n <= GATHER_TOKEN_TILE_MAX)
    wide = next(c for c in swapped if c.tile_n > GATHER_TOKEN_TILE_MAX)
    pairs = (
        (narrow, narrow, False),
        (wide, narrow, False),
        (dense, dense, False),
        (dense_q, dense, True),
        (narrow, dense, False),
        (dense, wide, False),
    )
    for c1, c2, fused in pairs:
        t = RF._tactic_bs(c1, c2, fused)
        assert RF._configs_from_tactic_bs(t) == (c1, c2, fused)
        for name, p in packs.items():
            out = runner.forward(runner.pack_inputs(p, W), tactic=t)
            assert torch.equal(bits(out), ref), (
                name,
                c1.describe(),
                c2.describe(),
                fused,
            )


@needs_device
def test_routing_constants_are_read_from_the_cache_under_graph_capture():
    """The route kernels' absent bias and unit scale are cached device constants (``api._const``): a call under CUDA graph
    capture reads the cached tensor when it exists (no fill node in the graph), a default first met inside a capture is a
    fill of that capture and is never cached, and a captured from-logits decode call of the layer replays the eager call's
    bits with no fill launch among its kernels (the class builds no constant of its own)."""
    from flashinfer.fused_moe.batch_invariant import api

    dev = torch.device("cuda", torch.cuda.current_device())
    k, H, I, E = 8, 512, 256, 64
    x, logits, w1, w2 = _problem(4, k, H, I, E, seed=31, adversarial=True)
    xq, sx = _xq(x)
    W, layer = _weights(w1, w2), _layer(k, E, I, tune_max=4)
    pack = _logits_pack(xq, sx, logits)
    eager = bits(layer(pack, W))
    b0, s0 = api._const(dev, E, 0.0), api._const(dev, 1, 1.0)
    assert api._const(dev, E, 0.0) is b0 and api._const(dev, 1, 1.0) is s0
    api._CONST.pop((dev.index, 3 * E, 0.0), None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        layer(pack, W)
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert api._const(dev, E, 0.0) is b0 and api._const(dev, 1, 1.0) is s0
            inside = api._const(dev, 3 * E, 0.0)
            out = layer(pack, W)
    torch.cuda.synchronize()
    assert api._const(dev, 3 * E, 0.0) is not inside
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), eager)
    assert _fills(graph.replay) == 1  # the one constant first met inside the capture
    assert torch.equal(bits(out), eager)


@needs_device
@pytest.mark.parametrize("method", ["Renormalize", "DeepSeekV3"])
def test_runner_first_used_under_graph_capture_gives_the_eager_bits(
    method, monkeypatch
):
    """A layer whose first forward runs under CUDA-graph capture with the shared constant cache empty (the route kernels'
    absent bias and unit scale, ``api._const``: isolated from the eager reference's warm cache, so every constant of that
    capture is a device fill of its own, as is the runner's routed scale), captured twice: the cache stays empty after
    both captures, the second graph replays before the first, an eager call then fills the cache once per device and both
    graphs replay again; a layer constructed inside a capture (its runner built there, the routed scale a per-call fill)
    and forwarded in it, its scale unset and the cache empty after the capture, before any replay; and the warm-cache
    control. Every output equals the eager bits of a layer that never saw a capture, under the softmax-renorm method (the
    cached constants in play) and under DeepSeekV3 with a routed scaling factor and a bias (the scale constant in play)."""
    from flashinfer.fused_moe.batch_invariant import api

    T, k, H, I, E = 48, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=91)
    xq, sx = _xq(x)
    rm = RoutingMethodType[method]
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(92)) * 0.1)
        .float()
        .cuda()
        if rm is RoutingMethodType.DeepSeekV3
        else None
    )
    kwargs = dict(routed_scaling_factor=1.75) if bias is not None else {}

    def make():
        return MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=rm, **kwargs),
                quant=QUANT,
                experts=ExpertConfig(intermediate_size=I),
                activation=_SWIGLU,
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )

    W = _weights(w1, w2)
    act = _logits_pack(xq, sx, logits, bias)
    ref = bits(make()(act, W))
    torch.cuda.synchronize()
    nan = float("nan")
    s = torch.cuda.Stream()
    warm = api._CONST
    scaled = bias is not None
    assert scaled or warm

    def captured(layer):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            out = layer(act, W)
        torch.cuda.synchronize()
        return g, out

    monkeypatch.setattr(api, "_CONST", {})
    fresh = make()
    graphs = [captured(fresh) for _ in range(2)]
    assert not api._CONST
    for g, out in reversed(graphs):
        out.fill_(nan)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
    assert torch.equal(bits(fresh(act, W)), ref)
    assert bool(api._CONST) == (not scaled)
    for g, out in graphs:
        out.fill_(nan)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
    monkeypatch.setattr(api, "_CONST", {})
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        inside = make()
        out = inside(act, W)
    torch.cuda.synchronize()
    assert inside.runners[0]._scale is None and not api._CONST
    out.fill_(nan)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), ref)
    assert inside.runners[0]._scale is None and not api._CONST
    assert torch.equal(bits(inside(act, W)), ref)
    assert (inside.runners[0]._scale is not None) == scaled
    assert bool(api._CONST) == (not scaled)
    monkeypatch.setattr(api, "_CONST", warm)
    warm_layer = make()
    assert torch.equal(bits(warm_layer(act, W)), ref)
    g, out = captured(warm_layer)
    out.fill_(nan)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), ref)
    assert torch.equal(bits(warm_layer(act, W)), ref)


@needs_device
def test_graphs_captured_at_two_token_counts_and_two_routing_modes_replay_interleaved():
    """One layer, four CUDA graphs: captured at a decode and a mid token count, each from logits and from pre-routed
    (unpacked) inputs, replayed interleaved twice with eager calls of every case between the rounds and once more back to
    back. Every replay gives the eager bits of its own inputs (the graphs share the layer's compiled bodies, winners and
    constant cache without interference), and the eager calls after the replays give them too."""
    k, H, I, E = 4, 256, 128, 16
    layer = _layer(k, E, I)
    cases, W = {}, None
    for name, T, seed in (("decode", 4, 21), ("mid", 200, 22)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed)
        if W is None:
            W = _weights(w1, w2)
        xq, sx = _xq(x)
        ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
        cases[(name, "logits")] = _logits_pack(xq, sx, logits)
        cases[(name, "routed")] = _routed_pack(xq, sx, ids.cuda(), wts.cuda())
    ref = {key: bits(layer(act, W)) for key, act in cases.items()}
    assert torch.equal(ref[("decode", "logits")], ref[("decode", "routed")])
    assert torch.equal(ref[("mid", "logits")], ref[("mid", "routed")])
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    graphs = {}
    for key, act in cases.items():
        with torch.cuda.stream(s):
            layer(act, W)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = layer(act, W)
        torch.cuda.synchronize()
        graphs[key] = (g, out)
    order = [
        ("decode", "logits"),
        ("mid", "routed"),
        ("mid", "logits"),
        ("decode", "routed"),
    ]
    nan = float("nan")
    for _ in range(2):
        for key in order:
            g, out = graphs[key]
            out.fill_(nan)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(bits(out), ref[key]), key
        for key in reversed(order):
            assert torch.equal(bits(layer(cases[key], W)), ref[key]), key
    for key in order:
        g, out = graphs[key]
        out.fill_(nan)
        g.replay()
    torch.cuda.synchronize()
    for key in order:
        assert torch.equal(bits(graphs[key][1]), ref[key]), key


@needs_device
def test_layer_forward_captured_in_every_dispatcher_form():
    """The whole layer forward under CUDA-graph capture, from logits and pre-routed, on the e4m3 pair and on the raw bf16
    rows (the plan + quantizer launch pre-routed, the standalone quantizer with the route + plan launch from logits), in
    every form the dispatcher picks: GEMM1's gather form of a narrow swapped record at a decode token count, the padded body
    with the standalone requantization at a prefill count (the gated chain) and with the fused requantization (the ungated
    chain). Each captures (no host sync, no host-side read), replays the eager bits, two replays agree, and the replay's
    fill launches and host copies are the eager call's (the constants host-built outside the capture: no fill of a
    constant, no host copy inside the graph)."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    seen = {}
    for name, T, k, H, I, E, act in (
        ("gather", 4, 4, 256, 128, 16, _SWIGLU),
        ("padded", 512, 4, 256, 128, 16, _SWIGLU),
        ("fused", 512, 4, 256, 128, 16, ReLU2()),
    ):
        a = activation_of(act)
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=50 + T, gated=a.gated)
        xq, sx = _xq(x)
        layer, W = _layer(k, E, I, act), _weights(w1, w2, act)
        ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
        packs = (
            ("logits", _logits_pack(xq, sx, logits)),
            ("routed", _routed_pack(xq, sx, ids.cuda(), wts.cuda())),
            ("bf16 logits", _logits_pack(x, None, logits)),
            ("bf16 routed", _routed_pack(x, None, ids.cuda(), wts.cuda())),
        )
        ref = bits(layer(packs[0][1], W))
        form = _form_of(layer.runners[0], T)
        if _family() == "sm_107":
            assert form == name, (name, form, layer.runners[0]._default(T))
        seen[name] = form
        for mode, pack in packs:
            assert torch.equal(bits(layer(pack, W)), ref), (name, mode)
            eager_fills = _fills(lambda: layer(pack, W))
            g, out = _capture(layer, pack, W)
            for _ in range(2):
                out.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(bits(out), ref), (name, mode)
            assert _fills(g.replay) == eager_fills, (name, mode, eager_fills)
            assert torch.equal(bits(layer(pack, W)), ref), (name, mode)
    assert len(seen) == 3


@needs_device
def test_tuned_tactic_captured_in_a_graph_replays_while_another_pack_form_runs_eagerly():
    """A layer autotuned at a decode token count (``reset_winner`` first), its forward captured in a CUDA graph with the
    tuned tactic: the same layer then serves the raw bf16 rows at that count (the other pack form, the tuned tactic reused)
    and a prefill count whose first picks run the other body eagerly (autotuned there too), and the graph replays unchanged
    (the tuned records, the eager bits) before, between and after those calls."""
    from flashinfer.autotuner import autotune

    k, H, I, E = 4, 256, 128, 16
    layer = _layer(k, E, I, tune_max=1024)
    x, logits, w1, w2 = _problem(4, k, H, I, E, seed=61)
    W = _weights(w1, w2)
    xq, sx = _xq(x)
    act = _logits_pack(xq, sx, logits)
    layer.reset_winner()
    with autotune(True):
        layer(act, W)
    ref = bits(layer(act, W))
    g, out = _capture(layer, act, W)
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), ref)
    raw = _logits_pack(x, None, logits)
    assert torch.equal(bits(layer(raw, W)), ref)
    x2, logits2, _, _ = _problem(512, k, H, I, E, seed=62)
    xq2, sx2 = _xq(x2)
    other = _logits_pack(xq2, sx2, logits2)
    if _family() == "sm_107":
        assert _form_of(layer.runners[0], 4) != _form_of(layer.runners[0], 512)
    ref_other = bits(layer(other, W))
    with autotune(True):
        assert torch.equal(bits(layer(other, W)), ref_other)
    for _ in range(2):
        out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
        assert torch.equal(bits(layer(other, W)), ref_other)
        assert torch.equal(bits(layer(raw, W)), ref)
    assert torch.equal(bits(layer(act, W)), ref)


@needs_device
def test_poisoned_tail_rows_and_intermediates_do_not_reach_the_bits(monkeypatch):
    """The packed row and scale buffers carry tail rows past the live rows (the TMA boxes stay in bounds; the tail is never
    stored) and every intermediate starts from the caching allocator: with the allocator's blocks pre-filled with e4m3 /
    bf16 NaN patterns and every ``torch.empty`` of the run filled with the pattern as well, the layer at a mid token count
    on both pack forms and the op's forms at a decode count -- the padded body with the standalone and the fused
    requantization, the swapped records in the box and the gather form, the raw bf16 rows through the plan + quantizer
    launch -- are finite and equal a clean run's bits, each other's on their tokens and the class's composition."""
    from flashinfer.fused_moe.batch_invariant import api_fp8bs as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        default_configs_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.gemm.batch_invariant.api import part_of

    k, H, I, E, T, Td = 4, 256, 128, 16, 64, 3
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=70, adversarial=True)
    xq, sx = _xq(x)
    layer, W = _layer(k, E, I), _weights(w1, w2)
    Wq = F.MoEWeightsE4M3BS.from_view(W.get_view(KEY))
    ids, wts = route(logits, k)
    sm, fam = part_of(xq.device)
    s1, s2 = default_configs_fp8bs(Td, k, H, I, E, sm, fam, swap=True)
    p1, p2 = default_configs_fp8bs(T, k, H, I, E, sm, fam, swap=False)
    q1, _ = default_configs_fp8bs(T, k, H, I, E, sm, fam, fused=True, swap=False)
    act = _logits_pack(xq, sx, logits)
    sxd = sx[:, :Td].contiguous()
    runs = {
        "layer": lambda: layer(act, W),
        "layer bf16": lambda: layer(_logits_pack(x, None, logits), W),
        "padded standalone": lambda: F.fused_moe_fp8bs(
            xq, sx, ids, wts, Wq, config1=p1, config2=p2, fuse_quant=False
        ),
        "padded fused": lambda: F.fused_moe_fp8bs(
            xq, sx, ids, wts, Wq, config1=q1, config2=p2, fuse_quant=True
        ),
        "swapped box": lambda: F.fused_moe_fp8bs(
            xq[:Td], sxd, ids[:Td], wts[:Td], Wq, config1=s1, config2=s2, gather=False
        ),
        "swapped gather": lambda: F.fused_moe_fp8bs(
            xq[:Td], sxd, ids[:Td], wts[:Td], Wq, config1=s1, config2=s2, gather=True
        ),
        "bf16 rows": lambda: F.fused_moe_fp8bs(
            x[:Td], None, ids[:Td], wts[:Td], Wq, config1=s1, config2=s2
        ),
    }
    for run in runs.values():
        run()  # the compile and the winner, outside the poisoned runs
    _poison_allocator(0x00)
    ref = {name: run() for name, run in runs.items()}
    for name, out in ref.items():
        assert torch.isfinite(out.float()).all(), name
    for name in ("layer bf16", "padded standalone", "padded fused"):
        assert torch.equal(bits(ref[name]), bits(ref["layer"])), name
    for name in ("swapped box", "swapped gather", "bf16 rows"):
        assert torch.equal(bits(ref[name]), bits(ref["layer"])[:Td]), name
    ids_r, wts_r = _reference_route(logits, k, RoutingMethodType.Renormalize)
    _ladder(
        xq, sx, ids_r.cuda(), wts_r.cuda(), W.get_view(KEY), activation_of(_SWIGLU),
        ref["layer"],
    )  # fmt: skip
    empty = torch.empty
    byte = 0x00

    def poisoned_empty(*args, **kwargs):
        t = empty(*args, **kwargs)
        if t.is_cuda and t.numel():
            t.flatten().view(torch.uint8).fill_(byte)
        return t

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    for byte in (0xFF, 0x7F):
        _poison_allocator(byte)
        for name, run in runs.items():
            out = run()
            assert torch.isfinite(out.float()).all(), (name, hex(byte))
            assert torch.equal(bits(out), bits(ref[name])), (name, hex(byte))


@needs_device
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``: a framework's dropped slot) is dropped by the
    block-scale op as by the bf16 one: the plan does not count it (``dest -1``), the gathers and the two GEMMs never touch
    it, and the combine folds the remaining slots in slot order with the term ``+0`` in its place, so the output equals the
    class on the same rows with those slots at a valid id and weight ``+0`` (the composition at the smaller count); a token
    whose slots are all dropped gives a zero row; the other tokens keep their bits; the raw bf16 rows agree (the plan +
    quantizer launch below ``PLAN_MULTI_MIN_N`` slots, the two launches above it); both record kinds agree. With every slot
    dropped the GEMMs launch without a tile and the output is all zero rows, on the dense and the swapped records."""
    from flashinfer.fused_moe.batch_invariant import api_fp8bs as F
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N, permute_plan
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        default_configs_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.gemm.batch_invariant.api import part_of

    k, H, I, E = 8, 256, 128, 64
    bad = (-1, E, 1 << 30)
    for T in (96, PLAN_MULTI_MIN_N // k + 3):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=21, adversarial=True)
        xq, sx = _xq(x)
        W, layer = _weights(w1, w2), _layer(k, E, I)
        ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
        ids_bad = ids.clone()
        rows = torch.randperm(T, generator=torch.Generator().manual_seed(T))[: T // 4]
        for j, t in enumerate(rows.tolist()):
            ids_bad[t, j % k] = bad[j % 3]
            ids_bad[t, (j + 3) % k] = bad[(j + 1) % 3]
        ids_bad[rows[0]] = torch.tensor(
            [bad[i % 3] for i in range(k)], dtype=torch.int32
        )
        dropped = (ids_bad < 0) | (ids_bad >= E)
        n_valid = int((~dropped).sum())
        counts, dest, src, *_ = permute_plan(ids_bad.cuda(), E)
        assert int(counts.sum()) == n_valid
        assert torch.equal(dest.cpu().view(-1) < 0, dropped.view(-1))
        assert src[src >= 0].unique().numel() == n_valid
        out = layer(_routed_pack(xq, sx, ids_bad.cuda(), wts.cuda()), W)
        ids_ref = torch.where(dropped, torch.zeros_like(ids), ids).cuda()
        wts_ref = torch.where(dropped, torch.zeros_like(wts), wts).cuda()
        if T == 96:
            ref = _ladder(
                xq, sx, ids_ref, wts_ref, W.get_view(KEY), activation_of(_SWIGLU)
            )
        else:
            ref = layer(_routed_pack(xq, sx, ids_ref, wts_ref), W)
        assert torch.equal(bits(out), bits(ref)), T
        assert torch.all(out[rows[0]] == 0)
        assert torch.equal(
            bits(layer(_routed_pack(x, None, ids_bad.cuda(), wts.cuda()), W)), bits(ref)
        )
        Wq = F.MoEWeightsE4M3BS.from_view(W.get_view(KEY))
        sm, fam = part_of(xq.device)
        for swap in (False, True):
            c1, c2 = default_configs_fp8bs(T, k, H, I, E, sm, fam, swap=swap)
            out_api = F.fused_moe_fp8bs(
                xq, sx, ids_bad.cuda(), wts.cuda(), Wq, config1=c1, config2=c2
            )
            assert torch.equal(bits(out_api), bits(out)), (T, swap)
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack(xq, sx, ids.cuda(), wts.cuda()), W)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))
    T, k, E = 37, 4, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=44)
    xq, sx = _xq(x)
    Wq = F.MoEWeightsE4M3BS.from_view(_weights(w1, w2).get_view(KEY))
    ids_none = torch.full((T, k), -1, dtype=torch.int32, device="cuda")
    wts_any = torch.rand(T, k, device="cuda")
    out, im = F.fused_moe_fp8bs(
        xq, sx, ids_none, wts_any, Wq, return_intermediates=True
    )
    torch.cuda.synchronize()
    assert torch.all(out == 0) and int(im["counts"].sum()) == 0
    assert torch.all(im["dest"] == -1)
    sm, fam = part_of(xq.device)
    for swap in (False, True):
        c1, c2 = default_configs_fp8bs(T, k, H, I, E, sm, fam, swap=swap)
        assert c1.swap_ab == swap and c2.swap_ab == swap
        out2 = F.fused_moe_fp8bs(xq, sx, ids_none, wts_any, Wq, config1=c1, config2=c2)
        torch.cuda.synchronize()
        assert torch.all(out2 == 0), swap


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_nan_activations_stay_in_their_rows(act):
    """An e4m3 NaN in a token's quantized row gives that token the composition's bits -- the NaN mask and the finite values
    (the requantization's block amax follows the non-NaN max rule the quantizer class states, so a NaN accumulator row leaves
    the chain as the codes the composition writes) -- and leaves every other token's bits unchanged, the tokens sharing its
    experts included: a row's fold, its block amax and its requantization never read another row; the raw bf16 rows with a
    NaN element agree with the quantized pair's rows."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    a = activation_of(act)
    T, k, H, I, E = 40, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=77, gated=a.gated, adversarial=True
    )
    xq, sx = _xq(x)
    xq = xq.clone()
    W, layer = _weights(w1, w2, act), _layer(k, E, I, act)
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    clean = layer(_routed_pack(xq, sx, ids, wts), W)
    rows = (3, 17, 39)
    xb = xq.view(torch.uint8)
    for r in rows:
        xb[r, (r * 7) % H] = 0x7F  # the e4m3 NaN byte
    out = layer(_routed_pack(xq, sx, ids, wts), W)
    ref = _ladder(xq, sx, ids, wts, W.get_view(KEY), a)
    nan_out, nan_ref = torch.isnan(out), torch.isnan(ref)
    assert torch.equal(nan_out, nan_ref)
    assert torch.equal(bits(out)[~nan_out], bits(ref)[~nan_out])
    for r in rows:  # the NaN moved its own row (the composition says where)
        assert not torch.equal(bits(out[r]), bits(clean[r])), r
    others = torch.tensor([t not in rows for t in range(T)], device="cuda")
    assert torch.equal(bits(out[others]), bits(clean[others]))
    assert not nan_out[others].any()
    xn = x.clone()
    for r in rows:
        xn[r, (r * 7) % H] = float("nan")
    out_b = layer(_routed_pack(xn, None, ids, wts), W)
    assert torch.equal(torch.isnan(out_b), nan_out)
    assert torch.equal(bits(out_b[others]), bits(clean[others]))


@needs_device
def test_forward_memoises_the_pick_and_the_launch_handles():
    """A forward after the first at a shape ranks nothing (the pick with its requantization spelling is memoised per token
    count, hidden size and block) and fetches no compiled object (the cached launch handles of the block-scale bodies); the
    public op on the same records and spelling gives the runner's bits."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    T, k, H, I, E = 5, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=77)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    xq, sx = _xq(x)
    pack = _logits_pack(xq, sx, logits)
    ref = bits(layer(pack, W))
    api.clear_kernel_cache()
    runner._picks.clear()
    calls = []
    orig = runner._ranked

    def counted(*args, **kw):
        calls.append(1)
        return orig(*args, **kw)

    runner._ranked = counted  # type: ignore[method-assign]
    try:
        assert torch.equal(bits(layer(pack, W)), ref)
        handles = dict(api._HANDLES)
        assert len(calls) == 1 and len(handles) >= 2
        assert torch.equal(bits(layer(pack, W)), ref)
        assert len(calls) == 1 and handles == api._HANDLES
    finally:
        del runner._ranked
    ids, wts = api.route(logits, k, runner._method)
    c1, c2 = runner._pick(T)
    w = MoEWeightsE4M3BS.from_view(W.get_view(runner.backend_key))
    out = fused_moe_fp8bs(
        xq, sx, ids, wts, w, config1=c1, config2=c2, fuse_quant=runner._fused
    )
    assert torch.equal(bits(out), ref)
