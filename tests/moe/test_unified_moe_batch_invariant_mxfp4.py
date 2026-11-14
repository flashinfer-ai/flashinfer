# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 x MXFP4 pair of the batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config``
with ``QuantConfig(weight=MXFP4, activation=MXFP4)``): the classes ``moe.mxfp4.<act>`` (MXFP4 experts and MXFP4
activations, one UE8M0 scale per 32 elements on both operands, no global or per-token scale), one per typed activation.

As the NVFP4 pair's suite, the tests are bit comparisons against the class composed from its own pieces: the routing on the
CPU, the ``mxfp4.quant`` class's reference for the two block quantizations, the dense batch-invariant ``mm_fp4`` (the MX
class, the instruction class the part runs) per expert with the bf16 CPU chains on the raw accumulators, the CPU combine.
Determinism including CUDA-graph replay, the two packings and the swapped decode body, the three routing input modes and
the pre-quantized pack forms, batch / permutation / sub-batch invariance, every tactic, the shape rules and the error
contract (the pair has no per-token scale and no static-intermediate class). Device-free: the registration of the pair,
the fingerprint's dimensions, the dispatcher's membership in the record spaces.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    BackendOptions,
    BatchInvariantFp4Config,
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
)
from flashinfer.fused_moe.batch_invariant.runner_fp4 import (
    PAIRS,
    BatchInvariantFp4Runner,
)
from flashinfer.fused_moe.batch_invariant.runner_mxfp4 import (
    MOE_MXFP4_ARCHS,
    BatchInvariantMoeMxfp4Runner,
)
from flashinfer.tllm_enums import RoutingMethodType
from tests.moe.test_unified_moe_batch_invariant import ACTIVATION_IDS, ACTIVATIONS
from tests.moe.test_unified_moe_batch_invariant_nvfp4 import (
    _family,
    _logits_pack,
    _poison_allocator,
    _problem,
    _reference_route,
    _routed_pack,
    bits,
)

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_MXFP4_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
_MX = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP4)
_NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
KEY = BatchInvariantFp4Runner.backend_key
MX_BLOCK = 32


def _config(
    k: int,
    E: int,
    I: int,
    act=_SWIGLU,
    method=RoutingMethodType.Renormalize,
    tune_max: int = 4096,
    quant=_MX,
    backend=None,
):
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=quant,
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((backend or BatchInvariantFp4Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(
    w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU, quant=_MX
) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp4Config.prepare_weights(
        w1,
        w2,
        quant=quant,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(KEY, view)
    return pack


def _weights_of(pack: MoEWeightPack):
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import MoEWeightsMXFP4

    return MoEWeightsMXFP4.from_view(pack.get_view(KEY))


# ----------------------------------------------------------------------------- device-free


def test_backend_serves_the_pair_and_never_a_default_candidate():
    """The fp4 backend serves the MXFP4 x MXFP4 pair through the MX implementation runner (the backend's key, the three routing
    modes and every activation class of the layer); the pair carries no per-token scale; the backend is not a default candidate."""
    from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND

    assert BatchInvariantFp4Config in ALL_BACKEND_CONFIGS
    assert not any(
        isinstance(c, BatchInvariantFp4Config) for c in _DEFAULT_BACKEND.candidates
    )
    assert PAIRS[_MX.pair] is BatchInvariantMoeMxfp4Runner
    assert BatchInvariantMoeMxfp4Runner.backend_key == KEY
    assert _MX.pair in BatchInvariantFp4Runner.supported_quant_variants
    assert BatchInvariantFp4Runner.supports_quant(_MX)
    assert not BatchInvariantMoeMxfp4Runner.supports_per_token_scale
    assert BatchInvariantFp4Runner.supports_per_token_scale  # the NVFP4 pair's
    assert set(
        BatchInvariantFp4Runner.supported_activation_classes_by_quant[_MX.pair]
    ) == {type(a) for a in ACTIVATIONS}
    assert set(BatchInvariantMoeMxfp4Runner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    # the mixed pair is still refused and named as pending
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    mixed = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        BatchInvariantFp4Config.prepare_activations(torch.zeros(2, 64), quant=mixed)
    assert "MXFP4 x MXFP4" in str(excinfo.value) and "NVFP4 x NVFP4" in str(
        excinfo.value
    )
    x, sf = BatchInvariantFp4Config.prepare_activations(torch.zeros(2, 64), quant=_MX)
    assert x.dtype == torch.bfloat16 and sf is None


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_mxfp4", ...)``: the geometry, the routing method, the activation and the architecture (the
    instruction class of the expert GEMMs is the part's); the classes differ from the NVFP4 and the bf16 ones, between
    activations, methods and parts; the spec's chain is the bf16 class's and GEMM2 is the plain conversion."""
    from flashinfer.fused_moe.batch_invariant import mterm, spec_mxfp4
    from flashinfer.fused_moe.batch_invariant import spec as S

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_mxfp4", arch="sm_107", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert bi.fingerprint("fused_moe_mxfp4", arch="sm_107", act=SwiGLU(), **dims) == fp
    assert bi.fingerprint("fused_moe_mxfp4", arch="sm_100", **dims) != fp
    assert bi.fingerprint("fused_moe_mxfp4", arch="sm_103", **dims) == bi.fingerprint(
        "fused_moe_mxfp4", arch="sm_100", **dims
    )
    assert bi.fingerprint("fused_moe_mxfp4", arch=(10, 7), **dims) == fp
    assert bi.fingerprint("fused_moe_mxfp4", arch="sm_107", act=ReLU2(), **dims) != fp
    assert (
        bi.fingerprint(
            "fused_moe_mxfp4", arch="sm_107", method=RoutingMethodType.Default, **dims
        )
        != fp
    )
    assert fp != bi.fingerprint("fused_moe_nvfp4", arch="sm_107", **dims)
    assert fp != bi.fingerprint("fused_moe", **dims)
    fps = {
        bi.fingerprint("fused_moe_mxfp4", arch="sm_107", act=a, **dims)
        for a in ACTIVATIONS
    }
    assert len(fps) == len(ACTIVATIONS)
    # the geometry is a dimension through the GEMM classes and the routing, not through the quantizer (row-length free)
    assert (
        bi.fingerprint(
            "fused_moe_mxfp4", arch="sm_107", H=4096, I=768, top_k=8, num_experts=128
        )
        != fp
    )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_mxfp4", arch="sm_107", method=RoutingMethodType.MiniMax2, **dims
        )
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_mxfp4", arch="sm_107", act="tanh", **dims)
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_mxfp4", arch="sm_90", **dims)
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_mxfp4",
            arch="sm_107",
            H=2048 + 16,
            I=768,
            top_k=8,
            num_experts=128,
        )
    assert bi.class_name("fused_moe_mxfp4") == "moe.mxfp4"
    # the spec: GEMM1 is the bf16 class's chain, GEMM2 the conversion; the instruction class is a dimension
    s = spec_mxfp4.MoEMxfp4Spec(64, 32, 2, 8)
    b = S.MoESpec(64, 32, 2, 8)
    assert mterm.sexpr(s.gemm1()) == mterm.sexpr(b.gemm1())
    assert mterm.sexpr(s.gemm2()) == "(cvt bf16 (acc2 0 0))"
    assert spec_mxfp4.spec_fingerprint_mxfp4(
        64, 32, 2, 8, mx_class=("mxfp4", "mxfp4")
    ) != spec_mxfp4.spec_fingerprint_mxfp4(
        64, 32, 2, 8, mx_class=("mxfp4_k128", "mxfp4_k128")
    )


def test_shape_rules_accept_the_model_grid_and_raise_elsewhere():
    import itertools

    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import check_shape_mxfp4

    Es, ks = (8, 16, 32, 64, 128, 256, 384, 512, 1024), (1, 2, 4, 6, 8, 10, 16)
    Is, Hs = (
        (256, 512, 640, 768, 1024, 1408, 1536, 2048, 2816, 4096, 14336),
        (2048, 2560, 3072, 4096, 5120, 7168),
    )
    for I, H, E, k in itertools.product(Is, Hs, Es, ks):
        if k > E:
            continue
        check_shape_mxfp4(H, I, E, k)
        check_shape_mxfp4(H, I, E, k, gated=False)
        check_shape_mxfp4(None, I, E, k)
    for args, why in (
        ((2048 + 16, 768, 128), "multiples of 32"),
        ((2048, 768 + 16, 128), "multiples of 32"),
        ((2048, 768, 0), "num_experts"),
        ((2048, 768, 1025), "num_experts"),
        ((2048, 768, 8, 9), "top_k"),
        ((2048, 768, 128, 17), "top_k"),
        ((4096, 64, 8), "unsplit"),
    ):
        with pytest.raises(BatchInvariantUnsupportedError, match=why):
            check_shape_mxfp4(*args)
    with pytest.raises(BatchInvariantUnsupportedError, match="multiples of 32"):
        check_shape_mxfp4(None, 768 + 16, 128)


@pytest.mark.parametrize("family,sm_count", [("sm_107", 212), ("sm_100", 148)])
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's picks for both GEMMs are members of the grouped or the swapped record space, accepted by the body for
    the shape; at decode the swapped records lead; every record of the ranked lists is distinct."""
    from flashinfer.fused_moe.batch_invariant.blocks.compile import (
        can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import (
        MXFP4_GEMM2_CHAIN,
        Mxfp4Chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4 import (
        default_configs_mxfp4,
        rank_grouped_mxfp4,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        grouped_knobs_blockscaled,
        rows_per_expert,
        swapped_knobs_blockscaled,
        token_tile_default,
    )
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx

    grouped, swapped = (
        set(grouped_knobs_blockscaled()),
        set(swapped_knobs_blockscaled()),
    )
    in1, in2 = expert_classes_mx(family)
    assert in1 == in2 == ("mxfp4_k128" if family == "sm_107" else "mxfp4")
    c1 = Mxfp4Chain("glu", "silu", "bf16", ())
    for H, I, E, k in ((2048, 768, 128, 8), (4096, 14336, 8, 2), (7168, 2048, 256, 8)):
        for T in (1, 4, 16, 64, 256, 1024, 4096, 16384):
            a, b = default_configs_mxfp4(
                T, k, H, I, E, (in1, in2), c1, MXFP4_GEMM2_CHAIN, sm_count, family
            )
            for cfg, N, K, ch in ((a, 2 * I, H, c1), (b, H, I, MXFP4_GEMM2_CHAIN)):
                assert cfg in grouped or cfg in swapped, (T, cfg.describe())
                ok = (
                    can_implement_swapped_blockscaled
                    if cfg.swap_ab
                    else can_implement_grouped_blockscaled
                )
                assert ok(cfg, N, K, E, ch, in1), (T, cfg.describe())
            if token_tile_default(rows_per_expert(T, k, E)) is not None:
                assert a.swap_ab and b.swap_ab, (T, a.describe(), b.describe())
            r = rank_grouped_mxfp4(
                rows_per_expert(T, k, E), 2 * I, H, E, c1, in1, sm_count, family
            )
            assert r and len(set(r)) == len(r) and r[0] == a


def test_unsupported_device_and_configurations_raise_before_any_kernel():
    """The pair implementation and the fp4 backend's runner over it refuse a CPU device alike, in the class's terms; the
    per-token flag (the pair has none) and the static-intermediate config (the NVFP4 pair's class) are refused at support."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    for cls in (BatchInvariantMoeMxfp4Runner, BatchInvariantFp4Runner):
        runner = cls(_config(8, 64, 256), torch.device("cpu"))
        with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
            runner._check_support()
        assert "moe.mxfp4" in str(excinfo.value) and "cpu" in str(excinfo.value)
    runner = BatchInvariantFp4Runner(
        _config(
            8,
            64,
            256,
            quant=QuantConfig(
                weight=QuantFormat.MXFP4,
                activation=QuantFormat.MXFP4,
                per_token_scale=True,
            ),
        ),
        torch.device("cpu"),
    )
    with pytest.raises(NotImplementedError, match="per_token_scale=True"):
        runner._check_support()
    runner = BatchInvariantFp4Runner(
        _config(8, 64, 256, backend=BatchInvariantFp4Config(intermediate="static")),
        torch.device("cpu"),
    )
    with pytest.raises(BatchInvariantUnsupportedError, match="static"):
        runner._check_support()
    w1 = torch.zeros(2, 64, 64, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 64, 32, dtype=torch.bfloat16)
    with pytest.raises(BatchInvariantUnsupportedError, match="static"):
        BatchInvariantFp4Config.prepare_weights(
            w1, w2, quant=_MX, intermediate="static"
        )
    with pytest.raises(BatchInvariantUnsupportedError):
        BatchInvariantFp4Config.prepare_weights(
            w1, w2, quant=_MX, intermediate_scales=torch.ones(2)
        )


# ----------------------------------------------------------------------------- GPU


def _dense_mx(a4, sf, w, wsf, in_fmt):
    """The dense batch-invariant MXFP4 GEMM (``mm_fp4``'s MX class, the instruction class pinned) with the fp32 output: the
    raw accumulators of the class."""
    from flashinfer.gemm.batch_invariant.api import mm_fp4_batch_invariant

    return mm_fp4_batch_invariant(
        a4,
        w.t(),
        sf,
        wsf,
        None,
        torch.float32,
        block_size=MX_BLOCK,
        use_nvfp4=False,
        in_fmt=in_fmt,
    )


def _quantized(x: torch.Tensor, check: bool = True):
    """The class's MXFP4 quantization of the rows on the device, checked against the quantizer class's reference and the
    public ``mxfp4_quantize(..., batch_invariant=True)`` entry."""
    from flashinfer.batch_invariant.quantization.mx_api import (
        mxfp4_quantize_batch_invariant,
    )
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )
    from flashinfer.batch_invariant.quantization.mx_spec import DEFAULT_RULE, MXFP4
    from flashinfer.tllm_enums import SfLayout
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import quantize_rows_mx

    x4, sf = quantize_rows_mx(x)
    if check:
        q_ref, sf_ref, _ = quantize_mx_reference(x, MXFP4, DEFAULT_RULE, "128x4")
        assert torch.equal(x4.cpu(), q_ref) and torch.equal(
            sf.cpu(), sf_ref.reshape(-1)
        )
        q_pub, sf_pub = mxfp4_quantize_batch_invariant(x, SfLayout.layout_128x4)
        assert torch.equal(x4, q_pub) and torch.equal(sf, sf_pub.reshape(-1))
    return x4, sf


def _reference_ffn(x_q, ids, weights, W, act, in_fmts):
    """The class composed from its own pieces, independently of the fused op: the (given) quantized rows, per expert the
    dense batch-invariant ``mm_fp4`` (the MX class) on the expert's rows with the bf16 CPU GEMM1 chain of the activation, the
    quantizer class's reference on the bf16 intermediate rows, ``mm_fp4`` per expert for GEMM2 with the CPU conversion, the
    CPU slot-order combine. Bit for bit what the fused op must produce."""
    from flashinfer.batch_invariant.quantization.layout import (
        swizzle_sf_128x4,
        unswizzle_sf_128x4,
    )
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )
    from flashinfer.batch_invariant.quantization.mx_spec import DEFAULT_RULE, MXFP4
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_mxfp4 as RQ
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP

    x4, x_sf = x_q
    T, Hh = x4.shape
    H, I, E, k = 2 * Hh, W.intermediate_size, W.num_experts, ids.shape[1]
    dev = x4.device
    sf_rows = unswizzle_sf_128x4(x_sf.reshape(-1).cpu(), T, H, MX_BLOCK)
    ids_c = ids.cpu()
    y = torch.zeros(T, k, H, dtype=torch.bfloat16)
    for e in range(E):
        tok, slot = (ids_c == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        rows = tok.to(dev)
        acc1 = _dense_mx(
            x4[rows].contiguous(),
            swizzle_sf_128x4(sf_rows[tok], tok.numel(), H, MX_BLOCK).to(dev),
            W.w1[e],
            W.w1_sf[e],
            in_fmts[0],
        ).cpu()
        if act.gated:
            u = acc1.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
            g = acc1.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
            a = RQ.mx_gemm1_cpu(g, u, act.act, act.params)
        else:
            a = RQ.mx_act_cpu(acc1, act.act, act.params)
        a4, a_sf, _ = quantize_mx_reference(a, MXFP4, DEFAULT_RULE, "128x4")
        acc2 = _dense_mx(
            a4.to(dev), a_sf.reshape(-1).to(dev), W.w2[e], W.w2_sf[e], in_fmts[1]
        ).cpu()
        y[tok, slot] = RQ.mx_gemm2_cpu(acc2)
    ident = torch.arange(T * k, dtype=torch.int32).view(T, k)
    return R.combine_cpu(y.view(T * k, H), ident, weights.cpu())


@needs_device
def test_prepared_weights_are_the_quantizer_class():
    """``prepare_weights`` quantizes every expert with the ``mxfp4.quant`` class (the quantizer's reference bytes, the gate / up
    rows interleaved first); the view names the pair and round-trips."""
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )
    from flashinfer.batch_invariant.quantization.mx_spec import DEFAULT_RULE, MXFP4
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1

    _, _, w1, w2 = _problem(4, 2, 256, 128, 4, seed=9)
    pack = _weights(w1, w2)
    view = pack.get_view(KEY)
    assert view["quant"] == "mxfp4" and "gemm1_alpha" not in view
    W = _weights_of(pack)
    w1i = interleave_fc1(w1)
    for e in range(4):
        q1, sf1, _ = quantize_mx_reference(w1i[e].cpu(), MXFP4, DEFAULT_RULE, "128x4")
        q2, sf2, _ = quantize_mx_reference(w2[e].cpu(), MXFP4, DEFAULT_RULE, "128x4")
        assert torch.equal(W.w1[e].cpu(), q1) and torch.equal(W.w1_sf[e].cpu(), sf1)
        assert torch.equal(W.w2[e].cpu(), q2) and torch.equal(W.w2_sf[e].cpu(), sf2)
    assert W.classes(_family()) == (
        ("mxfp4_k128", "mxfp4_k128") if _family() == "sm_107" else ("mxfp4", "mxfp4")
    )


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits, bf16 activations) == the class's routing on the CPU, the class's block quantization (== the
    quantizer's reference), the dense batch-invariant ``mm_fp4`` (MX class) per expert with the bf16 CPU chain of the
    activation, the quantizer's reference on the intermediate, ``mm_fp4`` + the CPU conversion, the CPU combine: bit for bit,
    for every activation of the layer, at the instruction class the part runs."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx

    T, k, H, I, E = 120, 4, 512, 256, 16
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=1, gated=chain.gated)
    layer = _layer(k, E, I, act)
    W_pack = _weights(w1, w2, act)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ref = _reference_ffn(
        _quantized(x),
        ids.cuda(),
        weights,
        _weights_of(W_pack),
        chain,
        expert_classes_mx(_family()),
    )
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
def test_swapped_body_equals_the_dense_class_per_expert_at_every_token_tile():
    """The swapped decode body at the MX instruction class: at every token tile (8 / 16 / 32) its fp32 accumulators == the
    dense batch-invariant ``mm_fp4`` on the group's rows per expert; the gated, the ungated and the plain chains == the bf16
    CPU chains of those accumulators. Uneven loads, one-row groups, empty groups, E = 256, N not a multiple of 128; rows past
    the live rows are never stored; a swapped record needs the dense packing."""
    from flashinfer.batch_invariant.quantization.layout import (
        swizzle_sf_128x4,
        unswizzle_sf_128x4,
    )
    from flashinfer.fused_moe.batch_invariant import reference_mxfp4 as RQ
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import (
        DENSE_TAIL,
        grouped_gemm_mxfp4,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.compile import (
        can_implement_swapped_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import (
        GROUP,
        MXFP4_GEMM2_CHAIN,
        Mxfp4Chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        swapped_knobs_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx

    f32_chain = Mxfp4Chain("plain", "silu", "f32", ())
    relu2_chain = Mxfp4Chain("act", "relu2", "bf16", ())
    in1, _ = expert_classes_mx(_family())
    for counts, I, K, kk in (
        ([5, 0, 70, 1, 33, 64, 0, 3], 128, 512, 2),
        ([1] * 16, 96, 1024, 1),
        ([0] * 200 + [40] + [0] * 55, 160, 256, 3),
        ([3, 7, 11, 13] * 16, 512, 128, 1),
        ([1], 128, 2048, 1),
    ):
        E, Tk, N = len(counts), sum(counts), 2 * I
        T = -(-Tk // kk)
        x, _, w1, w2 = _problem(T, kk, K, I, E, seed=E + 3)
        W = _weights_of(_weights(w1, w2))
        x4, sf = _quantized(x)
        g = torch.Generator().manual_seed(E + 7)
        tok = torch.randperm(T * kk, generator=g)[:Tk] // kk
        off, offp = [0], [0]
        for c in counts:
            off.append(off[-1] + c)
            offp.append(offp[-1] + -(-c // 128) * 128)
        Tk_pad = offp[-1]
        pos_pad = torch.cat(
            [torch.arange(offp[e], offp[e] + c) for e, c in enumerate(counts) if c]
        )
        a_dense = torch.zeros(Tk + DENSE_TAIL, K // 2, dtype=torch.uint8, device="cuda")
        a_dense[:Tk] = x4[tok.cuda()]
        sf_rows = unswizzle_sf_128x4(sf.cpu(), T, K, MX_BLOCK)
        sf_pad_rows = torch.zeros(
            Tk_pad + DENSE_TAIL, sf_rows.shape[1], dtype=sf_rows.dtype
        )
        sf_pad_rows[pos_pad] = sf_rows[tok]
        sf_pad = swizzle_sf_128x4(sf_pad_rows, Tk_pad + DENSE_TAIL, K, MX_BLOCK).cuda()
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        ref32 = torch.zeros(Tk, N, dtype=torch.float32)
        for e, c in enumerate(counts):
            if c:
                rows = tok[off[e] : off[e + 1]]
                ref32[off[e] : off[e + 1]] = _dense_mx(
                    x4[rows.cuda()].contiguous(),
                    swizzle_sf_128x4(sf_rows[rows], rows.numel(), K, MX_BLOCK).cuda(),
                    W.w1[e],
                    W.w1_sf[e],
                    in1,
                ).cpu()
        u = ref32.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
        gt = ref32.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
        refs = {
            "glu": (W.chain1, RQ.mx_gemm1_cpu(gt, u, "silu", ())),
            "relu2": (relu2_chain, RQ.mx_act_cpu(ref32, "relu2", ())),
            "plain": (MXFP4_GEMM2_CHAIN, RQ.mx_gemm2_cpu(ref32)),
        }
        common = (a_dense, sf_pad, W.w1, W.w1_sf, cnt)
        for cfg in swapped_knobs_blockscaled():
            assert can_implement_swapped_blockscaled(cfg, N, K, E, f32_chain, in1), (
                cfg.describe(),
                N,
                K,
                E,
            )
            out32 = torch.full(
                (Tk + DENSE_TAIL, N), float("nan"), dtype=torch.float32, device="cuda"
            )
            grouped_gemm_mxfp4(
                *common,
                cfg,
                f32_chain,
                in1,
                out=out32,
                dense_a=True,
                rows_sf=Tk_pad + DENSE_TAIL,
            )
            assert torch.equal(bits(out32[:Tk].cpu()), bits(ref32)), (
                counts[:4],
                cfg.describe(),
                in1,
            )
            assert torch.isnan(out32[Tk:]).all()
            for name, (chain, ref) in refs.items():
                got = grouped_gemm_mxfp4(
                    *common, cfg, chain, in1, dense_a=True, rows_sf=Tk_pad + DENSE_TAIL
                )
                assert torch.equal(bits(got[:Tk].cpu()), bits(ref)), (
                    counts[:4],
                    cfg.describe(),
                    name,
                )
            with pytest.raises(ValueError):
                grouped_gemm_mxfp4(*common, cfg, f32_chain, in1)


@needs_device
def test_both_packings_and_the_prequantized_pack_forms_give_the_bits():
    """The dense and the padded packing are placements (the same bits), the swapped and the padded body too; the pack forms
    agree: bf16 rows (quantized by the class), the quantized pack with the 128x4 swizzled scales (uint8 or e8m0 view) and
    with the linear ``[T, H/32]`` scales. A per-token scale on the pack is refused by the layer's shared contract, the opt-in
    flag by the pair."""
    from flashinfer.batch_invariant.quantization.layout import unswizzle_sf_128x4
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import fused_moe_mxfp4

    T, k, H, I, E = 257, 6, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2)
    layer = _layer(k, E, I)
    W_pack = _weights(w1, w2)
    W = _weights_of(W_pack)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    out_d = fused_moe_mxfp4(x, ids, weights, W, path="dense")
    out_p = fused_moe_mxfp4(x, ids, weights, W, path="padded")
    assert torch.equal(bits(out_d), bits(out_p)) and torch.equal(bits(out_d), bits(out))
    out_s = fused_moe_mxfp4(x, ids, weights, W, path="dense", swap=True)
    out_n = fused_moe_mxfp4(x, ids, weights, W, path="dense", swap=False)
    assert torch.equal(bits(out_s), bits(out)) and torch.equal(bits(out_n), bits(out))
    with pytest.raises(ValueError):
        fused_moe_mxfp4(x, ids, weights, W, path="padded", swap=True)
    x4, sf = _quantized(x)
    mode = RoutingInputMode.UnpackedPrecomputed
    for scale in (sf, sf.view(torch.float8_e8m0fnu)):
        out_q = layer(
            MoEActivationPack(
                x4, scale, topk_ids=ids, topk_weights=weights, routing_input_mode=mode
            ),
            W_pack,
        )
        assert torch.equal(bits(out_q), bits(out))
    sf_lin = unswizzle_sf_128x4(sf.cpu(), T, H, MX_BLOCK).cuda()
    assert tuple(sf_lin.shape) == (T, H // MX_BLOCK)
    out_l = layer(
        MoEActivationPack(
            x4, sf_lin, topk_ids=ids, topk_weights=weights, routing_input_mode=mode
        ),
        W_pack,
    )
    assert torch.equal(bits(out_l), bits(out))
    assert torch.equal(bits(fused_moe_mxfp4((x4, sf_lin), ids, weights, W)), bits(out))
    with pytest.raises(ValueError, match="per_token_scale"):
        layer(
            MoEActivationPack(
                x4,
                sf,
                topk_ids=ids,
                topk_weights=weights,
                routing_input_mode=mode,
                per_token_scale=torch.ones(T, device="cuda"),
            ),
            W_pack,
        )
    with pytest.raises((NotImplementedError, RuntimeError, ValueError)):
        _layer(
            k,
            E,
            I,
            quant=QuantConfig(
                weight=QuantFormat.MXFP4,
                activation=QuantFormat.MXFP4,
                per_token_scale=True,
            ),
        )


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3)
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
def test_layer_fused_routing_at_small_t_matches_prerouted_routing():
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import api_mxfp4 as A
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 5, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3)
    layer, W_pack = _layer(k, E, I), _weights(w1, w2)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
    assert torch.equal(bits(out), bits(layer(_routed_pack(x, ids, wts), W_pack)))
    W = _weights_of(W_pack)
    ids3, wts3, *plan = A.route_plan_dual(logits, k, S.SOFTMAX_RENORM)
    out3 = A.fused_moe_mxfp4(x, ids3, wts3, W, plan=tuple(plan))
    assert torch.equal(bits(out3), bits(out))
    assert torch.equal(bits(A.fused_moe_mxfp4(x, ids, wts, W)), bits(out))
    with pytest.raises(ValueError):
        A.fused_moe_mxfp4(x, ids3, wts3, W, path="padded", plan=tuple(plan))


@needs_device
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens
    create: prefixes, permutations, sub-batches, one token alone, and a hot expert group around a probe token."""
    T, k, H, I, E = 500, 8, 512, 256, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=4)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    full = bits(layer(_logits_pack(x, logits), W))
    for n in (1, 2, 7, 64, 200, 333):
        assert torch.equal(bits(layer(_logits_pack(x[:n], logits[:n]), W)), full[:n]), n
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(5)).cuda()
    assert torch.equal(
        bits(layer(_logits_pack(x[perm].contiguous(), logits[perm].contiguous()), W)),
        full[perm],
    )
    for n in (1, 7, 64, 200):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(n))[:n].cuda()
        assert torch.equal(
            bits(layer(_logits_pack(x[idx].contiguous(), logits[idx].contiguous()), W)),
            full[idx],
        ), n
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    base = bits(layer(_routed_pack(x, ids, weights), W))
    g = torch.Generator().manual_seed(6)
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


@needs_device
@pytest.mark.parametrize("T", [300, 8], ids=["mid", "decode"])
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits(T):
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.batch_invariant.runner_nvfp4 import (
        _configs_from_tactic_nvfp4,
        _tactic_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        rows_per_expert,
        token_tile_default,
    )

    k, H, I, E = 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=7)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantFp4Runner)
    assert isinstance(runner.impl, BatchInvariantMoeMxfp4Runner)
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import TOP_K_FP4

    # GEMM1's list carries the structural rule's gather twin beside its box record (one more than the ranking's head)
    assert 1 <= len(tactics) <= TOP_K_FP4 * (TOP_K_FP4 + 1) and len(
        set(tactics)
    ) == len(tactics)
    r1, r2 = runner.impl._ranked(T, H)
    assert tactics == [_tactic_nvfp4(c1, c2) for c1 in r1 for c2 in r2]
    c1, c2 = _configs_from_tactic_nvfp4(tactics[0])
    assert (c1, c2) == (r1[0], r2[0])
    # this pair's tactic space: the box head plus the structural rule's gather twin of the narrow swapped tile
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import GATHER_TILE_N

    box = [c for c in r1 if not c.gather_a]
    assert r1[: len(box)] == box and [c.describe() for c in r1 if c.gather_a] == [
        c.describe() + "_ga" for c in box if c.swap_ab and c.tile_n == GATHER_TILE_N
    ]
    if token_tile_default(rows_per_expert(T, k, E)) is not None:
        assert c1.swap_ab and c2.swap_ab, (T, c1.describe(), c2.describe())
    for t in tactics:
        out = runner.forward(runner.pack_inputs(pack, W), tactic=t)
        assert torch.equal(bits(out), ref), t
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)


@needs_device
def test_swapped_gather_record_gives_the_box_forms_bits_on_every_pack_form():
    """The MXFP4 pair's swapped gather records (:func:`...api_nvfp4.fused_moe_nvfp4`'s form on this class): the box form's bits for
    the bf16 input, the pre-quantized pair with the 128x4 and with the linear scale layout, every token tile; the padded
    packing and GEMM2 refuse the record."""
    from flashinfer.batch_invariant.quantization.layout import unswizzle_sf_128x4
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import fused_moe_mxfp4
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4 import (
        default_configs_mxfp4,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        GATHER_TILE_N,
        gather_twins,
        swapped_knobs_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import MX_BLOCK

    twins = gather_twins(swapped_knobs_blockscaled())
    H, I, E = 512, 256, 32
    for T, k in ((5, 8), (300, 6)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3 + T)
        W = _weights_of(_weights(w1, w2))
        ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        ref = bits(fused_moe_mxfp4(x, ids, weights, W, gather=False))
        in1, in2 = W.classes(_family())
        _, c2 = default_configs_mxfp4(
            T, k, H, I, E, (in1, in2), W.chain1, W.chain2, 212, _family()
        )
        x4, sf = _quantized(x)[:2]
        sf_lin = (
            unswizzle_sf_128x4(sf.cpu(), T, H, MX_BLOCK)
            .cuda()
            .view(torch.uint8)
            .reshape(T, H // MX_BLOCK)
        )
        for rec in twins:
            out, im = fused_moe_mxfp4(
                x, ids, weights, W, config1=rec, config2=c2, return_intermediates=True
            )
            assert im["config1"].gather_a and im["x_perm4"] is None
            assert torch.equal(bits(out), ref), rec.describe()
            assert torch.equal(
                bits(
                    fused_moe_mxfp4((x4, sf), ids, weights, W, config1=rec, config2=c2)
                ),
                ref,
            )
            assert torch.equal(
                bits(
                    fused_moe_mxfp4(
                        (x4, sf_lin), ids, weights, W, config1=rec, config2=c2
                    )
                ),
                ref,
            )
        if T <= 16:
            out_g, im = fused_moe_mxfp4(
                x, ids, weights, W, gather=True, return_intermediates=True
            )
            assert im["config1"].gather_a and torch.equal(bits(out_g), ref)
            # the default: the rule's form (this pair's scatter launch folded into the narrow swapped tile)
            for xin in (x, (x4, sf), (x4, sf_lin)):
                out_d, im = fused_moe_mxfp4(
                    xin, ids, weights, W, return_intermediates=True
                )
                c = im["config1"]
                assert (
                    c.gather_a == (c.swap_ab and c.tile_n == GATHER_TILE_N)
                    and (im["x_perm4"] is None) == c.gather_a
                )
                assert torch.equal(bits(out_d), ref)
    with pytest.raises(ValueError):
        fused_moe_mxfp4(x, ids, weights, W, path="padded", gather=True)
    with pytest.raises(ValueError):
        fused_moe_mxfp4(x, ids, weights, W, config1=c2, config2=twins[0])


@needs_device
@pytest.mark.parametrize("body", ["padded", "dense", "swapped"])
def test_pdl_chain_alternating_inputs_give_each_call_its_own_bits(body):
    import os

    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import fused_moe_mxfp4

    T, k, H, I, E = 16, 8, 2048, 768, 128
    kw = {"path": "padded"} if body == "padded" else {"swap": body == "swapped"}
    W = None
    calls, refs = [], []
    for seed in (5, 6):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed)
        if W is None:
            W = _weights_of(_weights(w1, w2))
        ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        calls.append((x, ids, weights))
        refs.append(bits(fused_moe_mxfp4(x, ids, weights, W, **kw).cpu()))
    assert not torch.equal(refs[0], refs[1])
    for i in range(int(os.environ.get("FLASHINFER_BI_STRESS_ITERS", "20"))):
        for (x, ids, weights), ref in zip(calls, refs, strict=True):
            assert torch.equal(
                bits(fused_moe_mxfp4(x, ids, weights, W, **kw).cpu()), ref
            ), (i, body)


@needs_device
@pytest.mark.parametrize("body", ["padded", "dense", "swapped"])
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_poisoned_intermediates_do_not_reach_the_bits(body, act):
    """Unspecified bytes and timing never reach the output: with the caching allocator's blocks filled with NaN patterns before
    every call and a bandwidth-bound kernel on a side stream, each body gives the bits of a clean run, which is the class
    composed from its pieces (a shape whose GEMM2 K is below the K = 128 class's k-tile: the zero-filled tail)."""
    import os

    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import fused_moe_mxfp4
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx

    T, k, H, I, E = 64, 8, 256, 64, 256
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5, gated=chain.gated)
    W = _weights_of(_weights(w1, w2, act))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    kw = (
        {"path": "padded"}
        if body == "padded"
        else {"path": "dense", "swap": body == "swapped"}
    )
    ref = bits(fused_moe_mxfp4(x, ids, weights, W, **kw).cpu())
    side = torch.cuda.Stream()
    junk = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(int(os.environ.get("FLASHINFER_BI_STRESS_ITERS", "20"))):
        for byte in (0x7F, 0xFF, 0x00):
            _poison_allocator(byte)
            with torch.cuda.stream(side):
                for _ in range(3):
                    junk.add_(1)
            out = fused_moe_mxfp4(x, ids, weights, W, **kw)
            assert torch.equal(bits(out.cpu()), ref), (i, hex(byte))
    torch.cuda.synchronize()
    comp = _reference_ffn(
        _quantized(x), ids, weights, W, chain, expert_classes_mx(_family())
    )
    assert torch.equal(ref, bits(comp.cpu()))


@needs_device
def test_instruction_classes_on_the_part():
    """The op runs the part's MX instruction class (the sm_107 K = 128 step there, the K = 64 step elsewhere); the K = 64 class
    is a class of every part, and the op at a forced class equals the composition at that class (another class: other bits in
    general, both canonical)."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import fused_moe_mxfp4
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx

    T, k, H, I, E = 70, 4, 512, 256, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=8)
    W = _weights_of(_weights(w1, w2))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    fam = _family()
    out, inter = fused_moe_mxfp4(x, ids, weights, W, return_intermediates=True)
    assert inter["in_fmts"] == expert_classes_mx(fam)
    chain = activation_of(_SWIGLU)
    for fmts in {expert_classes_mx(fam), ("mxfp4", "mxfp4")}:
        got = fused_moe_mxfp4(x, ids, weights, W, in_fmts=fmts)
        ref = _reference_ffn(_quantized(x, check=False), ids, weights, W, chain, fmts)
        assert torch.equal(bits(got.cpu()), bits(ref)), fmts


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract of the pair: the bounds and alignment rules, fused shared experts, EP shards, no finalize, the wrong
    pair's weights, scales with bf16 rows, packed rows without scales, a per-token scale, the static config; a zero-token call
    returns an empty output."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    k, H, I, E = 6, 512, 256, 64
    errors = (
        BatchInvariantUnsupportedError,
        RuntimeError,
        NotImplementedError,
        ValueError,
    )
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=264), "multiples of 32"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.mxfp4" in str(excinfo.value), what
    routing = RoutingConfig(num_experts=E, top_k=k)
    experts = ExpertConfig(intermediate_size=I)
    for kw, what in (
        (
            dict(experts=ExpertConfig(intermediate_size=I, num_fused_shared_experts=1)),
            "shared",
        ),
        (
            dict(experts=ExpertConfig(intermediate_size=I, local_num_experts=E // 2)),
            "expert-parallel",
        ),
        (
            dict(experts=ExpertConfig(intermediate_size=I, local_expert_offset=E // 2)),
            "expert-parallel",
        ),
        (dict(finalize=MoEFinalizeConfig(do_finalize=False)), "do_finalize"),
        (dict(quant=QuantConfig()), "BF16"),
        (dict(backend=BatchInvariantFp4Config(intermediate="static")), "static"),
    ):
        with pytest.raises(errors) as excinfo:
            MoELayer(
                MoEConfig(
                    routing=kw.get("routing", routing),
                    quant=kw.get("quant", _MX),
                    experts=kw.get("experts", experts),
                    backend=BackendOptions(
                        (kw.get("backend", BatchInvariantFp4Config()),)
                    ),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.mxfp4" in str(excinfo.value), what
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=12)
    W = _weights(w1, w2)
    with pytest.raises(errors):
        layer(_logits_pack(x.to(torch.float16), logits), W)
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(_logits_pack(x, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2()))
    with pytest.raises(BatchInvariantUnsupportedError, match="another pair"):
        layer(
            _logits_pack(x, logits), _weights(w1, w2, quant=_NVFP4)
        )  # the NVFP4 pair's weights
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(
            _logits_pack(
                x,
                logits,
                sf=torch.zeros(8, H // MX_BLOCK, dtype=torch.uint8, device="cuda"),
            ),
            W,
        )
    x4, sf = _quantized(x, check=False)
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(_logits_pack(x4, logits), W)  # packed rows without their scales
    with pytest.raises(ValueError, match="per_token_scale"):
        layer(_logits_pack(x4, logits, sf=sf, pts=torch.ones(8, device="cuda")), W)
    out = layer(_logits_pack(x[:0], logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16
    # the NVFP4 layer refuses this pair's weights likewise
    nv = _layer(k, E, I, quant=_NVFP4)
    with pytest.raises(BatchInvariantUnsupportedError, match="another pair"):
        nv(_logits_pack(x, logits), W)


# ----------------------------------------------------------------------------- dropped slots


@needs_device
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``) is a dropped slot for this pair as for the others:
    the plan marks it (``dest`` / ``dest_pad`` -1, ``pos_pad`` -1 past the live rows), the scatter of the codes and the
    scale re-tiling write nothing for it (nothing outside the live rows or the buffers: sentinel guards), and the combine
    folds ``+0`` in its place: the output equals the composition with those slots at a valid id and weight 0, a token whose
    slots are all dropped gives a zero row, the other tokens keep their bits, the combine oracle over the op's intermediates
    gives the bits; both plan forms, both packings; every slot dropped on the dense and the swapped records."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N, permute_plan
    from flashinfer.fused_moe.batch_invariant.api_mxfp4 import (
        _placeholder,
        fused_moe_mxfp4,
        mx_sf_cols,
        mx_sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        DENSE_TAIL,
        _part,
        padded_rows_bound,
        permute_plan_dual,
    )
    from flashinfer.fused_moe.batch_invariant.permute import warps_per_row
    from flashinfer.fused_moe.batch_invariant.permute_nvfp4 import (
        get_compiled_scatter_fp4,
        get_compiled_scatter_sf,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4 import (
        default_configs_mxfp4,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4 import expert_classes_mx
    from tests.moe.test_unified_moe_batch_invariant_nvfp4 import (
        _guarded,
        _guards_untouched,
    )

    k, H, I, E = 8, 256, 128, 64
    bad = (-1, E, 1 << 30)
    chain = activation_of(_SWIGLU)
    sm_count, family = _part(torch.device("cuda"))
    for T in (96, PLAN_MULTI_MIN_N // k + 3):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=21)
        W_pack, layer = _weights(w1, w2), _layer(k, E, I)
        W = _weights_of(W_pack)
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
        counts, dest, src = permute_plan(ids_bad.cuda(), E)
        assert int(counts.sum()) == n_valid and torch.equal(
            dest.cpu().view(-1) < 0, dropped.view(-1)
        )
        _, _, _, dest_pad, pos_pad = permute_plan_dual(ids_bad.cuda(), E)
        assert torch.equal(dest_pad.cpu().view(-1) < 0, dropped.view(-1))
        assert torch.all(pos_pad[n_valid:] == -1) and torch.all(pos_pad[:n_valid] >= 0)
        x4, sf = _quantized(x)
        Tk, Tk_pad = T * k, padded_rows_bound(T, k, E)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        unit = _placeholder(x.device)
        x_perm4, big4 = _guarded(Tk_a * (H // 2))
        x_perm_sf, bigsf = _guarded(mx_sf_numel(Tk_sf, H))
        x_perm4 = x_perm4.view(Tk_a, H // 2)
        get_compiled_scatter_fp4(
            k, warps_per_row(Tk, H // 32, sm_count), "128x4", False, True, MX_BLOCK
        )(
            x4,
            sf,
            unit,
            dest.reshape(-1),
            dest_pad.reshape(-1),
            x_perm4,
            x_perm_sf,
            unit,
            Tk,
            H // 2,
            mx_sf_cols(H),
        )
        torch.cuda.synchronize()
        assert _guards_untouched(big4) and _guards_untouched(bigsf)
        assert bool((x_perm4[n_valid:] == 0xAB).all())
        assert torch.equal(x_perm4[:n_valid], x4[(src[:n_valid] // k).long()])
        a_sf_q = torch.randint(
            0, 255, (mx_sf_numel(Tk, I),), dtype=torch.uint8, device="cuda"
        )
        a_sf_s, big_s = _guarded(mx_sf_numel(Tk_sf, I))
        get_compiled_scatter_sf(1, True)(
            a_sf_q, unit, pos_pad, a_sf_s, unit, Tk, mx_sf_cols(I)
        )
        torch.cuda.synchronize()
        assert _guards_untouched(big_s)
        # the op
        out = layer(_routed_pack(x, ids_bad.cuda(), wts.cuda()), W_pack)
        ids_ref = torch.where(dropped, torch.zeros_like(ids), ids)
        wts_ref = torch.where(dropped, torch.zeros_like(wts), wts)
        ref = _reference_ffn(
            _quantized(x, check=False),
            ids_ref.cuda(),
            wts_ref,
            W,
            chain,
            expert_classes_mx(family),
        )
        assert torch.equal(bits(out.cpu()), bits(ref)), T
        assert torch.all(out[rows[0]] == 0)
        for path in ("dense", "padded"):
            out_api, im = fused_moe_mxfp4(
                x, ids_bad.cuda(), wts.cuda(), W, path=path, return_intermediates=True
            )
            assert torch.equal(bits(out_api), bits(out)), path
            assert torch.equal(
                bits(out_api.cpu()), bits(R.combine_cpu(im["y"], im["dest"], wts))
            ), path
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack(x, ids.cuda(), wts.cuda()), W_pack)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))
    T, k, E = 37, 4, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=44)
    W = _weights_of(_weights(w1, w2))
    ids_none = torch.full((T, k), -1, dtype=torch.int32, device="cuda")
    wts_any = torch.rand(T, k, device="cuda")
    out, im = fused_moe_mxfp4(x, ids_none, wts_any, W, return_intermediates=True)
    torch.cuda.synchronize()
    assert (
        torch.all(out == 0)
        and int(im["counts"].sum()) == 0
        and torch.all(im["dest"] == -1)
    )
    for swap in (False, True):
        c1, c2 = default_configs_mxfp4(
            T,
            k,
            H,
            I,
            E,
            expert_classes_mx(family),
            W.chain1,
            W.chain2,
            sm_count,
            family,
            swap=swap,
        )
        assert c1.swap_ab == swap and c2.swap_ab == swap
        assert torch.all(
            fused_moe_mxfp4(x, ids_none, wts_any, W, config1=c1, config2=c2) == 0
        ), swap
    assert torch.all(fused_moe_mxfp4(x, ids_none, wts_any, W, path="padded") == 0)
