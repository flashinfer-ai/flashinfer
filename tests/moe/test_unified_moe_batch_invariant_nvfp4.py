# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 x NVFP4 pair of the batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config``
with ``QuantConfig(weight=NVFP4, activation=NVFP4)``): the W4A4 classes
``moe.nvfp4.<act>`` (NVFP4 experts with one per-tensor global scale per expert and GEMM, per-token NVFP4 activations), one
per typed activation of the layer.

The class has exact CPU models of everything but the tensor-core steps (the routing, the W4A4 chains on the fp32
accumulators, the slot-order combine) and the quantizer class's own reference for the two per-token quantizations, so the
tests are bit comparisons: the layer's output against the class composed from its own pieces (routing on the CPU, the dense
batch-invariant ``mm_fp4`` per expert with the instruction class the part runs, the CPU chains, the quantizer's reference on
the materialised intermediate, the CPU combine) for every activation, run-to-run determinism including CUDA-graph replay,
the two packings of the expanded rows, the three routing input modes and the pre-quantized pack forms, batch / permutation /
sub-batch / hot-expert invariance, agreement of every tactic the autotuner may pick, the shape rules and the error contract
(an unsupported configuration raises before any kernel runs), and a plausibility observation against the fp64 evaluation of
the same quantized pipeline and FlashInfer's torch reference of the unquantized model. The device-free tests pin the
registration, the fingerprint's dimensions (the architecture among them), the agreement of the vectorised and the exact CPU
chains on adversarial accumulators and scales, and the dispatcher's membership in the grouped block-scaled space.
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
from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND
from flashinfer.fused_moe.batch_invariant.runner_fp4 import BatchInvariantFp4Runner
from flashinfer.fused_moe.batch_invariant.runner_nvfp4 import (
    MOE_NVFP4_ARCHS,
    BatchInvariantMoeNvfp4Runner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_NVFP4_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
_NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
# a pre-quantized pack with the class's per-token scales needs the layer's opt-in (the shared pack contract)
_NVFP4_PTS = QuantConfig(
    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
)
KEY = BatchInvariantFp4Runner.backend_key
# the activations the suites cover (every typed activation of the layer, the parameterised ones once more with scalars that
# move every branch): the bf16 backend's list, each its own class of this backend too
from tests.moe.test_unified_moe_batch_invariant import (  # noqa: E402
    ACTIVATION_IDS,
    ACTIVATIONS,
)


def bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    return t.contiguous().view(torch.int32)


def _family() -> str:
    from flashinfer.gemm.batch_invariant.api import part_of

    return part_of(torch.device("cuda"))[1]


def _problem(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    seed: int,
    gated: bool = True,
    adversarial: bool = True,
):
    """Activations, logits and canonical bf16 weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated): spread
    exponents and per-expert magnitude spans, so the per-token and per-expert scales and the fp4 codes span the formats."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        x = (
            torch.randn(T, H, generator=g)
            * torch.exp2(torch.randint(-8, 6, (T, 1), generator=g).float())
        ).to(torch.bfloat16)
        logits = (torch.randn(T, E, generator=g) * 2).float()
        logits[0, : min(k, E)] = 1.0
        span = torch.exp2(torch.randint(-3, 3, (E, 1, 1), generator=g).float())
    else:
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
        logits = (torch.randn(T, E, generator=g) * 2).float()
        span = 1.0
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
    quant=_NVFP4,
):
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=quant,
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp4Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(KEY, view)
    return pack


def _logits_pack(x, logits, bias=None, sf=None, pts=None) -> MoEActivationPack:
    return MoEActivationPack(
        x,
        sf,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
        per_token_scale=pts,
    )


def _routed_pack(x, ids, weights, packed: bool = False) -> MoEActivationPack:
    if packed:
        return MoEActivationPack(
            x,
            None,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        x,
        None,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The fp4 config is a known backend of the layer, mapped to the fp4 runner, absent from the default candidates; the
    runner serves the NVFP4 x NVFP4 pair through the NVFP4 implementation runner (the two share the backend's key) beside the
    MXFP4 x MXFP4 pair, declares the three routing modes and every activation class of the layer for the pair."""
    assert BatchInvariantFp4Config in ALL_BACKEND_CONFIGS
    assert _BACKEND_RUNNERS[BatchInvariantFp4Config] is BatchInvariantFp4Runner
    assert not any(
        isinstance(c, BatchInvariantFp4Config) for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantFp4Config.supported(a)
    ] == [100, 103, 107]
    assert KEY == "batch_invariant_fp4"
    assert BatchInvariantMoeNvfp4Runner.backend_key == KEY
    assert BatchInvariantFp4Runner.supported_quant_variants == (
        (QuantFormat.NVFP4, QuantFormat.NVFP4),
        (QuantFormat.MXFP4, QuantFormat.MXFP4),
    )  # the pairs served: this suite's and the MXFP4 x MXFP4 pair's (its own suite)
    assert BatchInvariantFp4Runner.supports_quant(_NVFP4)
    assert BatchInvariantFp4Runner.supports_per_token_scale
    assert set(BatchInvariantFp4Runner.supported_activation_classes) == {
        type(a) for a in ACTIVATIONS
    }
    assert set(
        BatchInvariantFp4Runner.supported_activation_classes_by_quant[_NVFP4.pair]
    ) == {type(a) for a in ACTIVATIONS}
    assert set(BatchInvariantFp4Runner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert repr(BatchInvariantFp4Config()) == "BatchInvariantFp4Config()"


def test_unserved_quant_pair_is_refused_and_names_the_served_pairs():
    """A quantization pair the fp4 backend does not serve: the layer's candidate filter skips the runner
    (``supports_quant``), and ``prepare_weights`` / ``prepare_activations`` / the runner's ``check_support`` raise the
    mode's error naming the served pairs and the pending one."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    mixed = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    assert not BatchInvariantFp4Runner.supports_quant(mixed)
    assert not BatchInvariantFp4Runner.supports_quant(QuantConfig())
    w1 = torch.zeros(2, 64, 64, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 64, 32, dtype=torch.bfloat16)
    for call in (
        lambda: BatchInvariantFp4Config.prepare_weights(w1, w2, quant=mixed),
        lambda: BatchInvariantFp4Config.prepare_activations(w1[0], quant=mixed),
        lambda: BatchInvariantFp4Runner(
            MoEConfig(
                routing=RoutingConfig(num_experts=2, top_k=1),
                quant=mixed,
                experts=ExpertConfig(intermediate_size=32),
                backend=BackendOptions((BatchInvariantFp4Config(),)),
            ),
            torch.device("cuda", 0),
        ).check_support(),
    ):
        with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
            call()
        msg = str(excinfo.value)
        assert "weight=MXFP4, activation=MXFP8" in msg
        assert "NVFP4 x NVFP4" in msg and "mixed MoE PR" in msg and KEY in msg
    # the default pair is NVFP4 x NVFP4: the bf16 rows as given, no scale
    x, sf = BatchInvariantFp4Config.prepare_activations(w1[0].float())
    assert x.dtype == torch.bfloat16 and sf is None


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_nvfp4", ...)`` takes the geometry, the routing method, the activation and the architecture
    (the instruction class per expert weight shape); the classes differ from the bf16 ones, between activations, between
    the methods and between the parts; an unimplemented method and an unknown chain are refused."""
    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_nvfp4", arch="sm_107", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert (
        bi.fingerprint(
            "fused_moe_nvfp4", arch="sm_107", method="Renormalize", act="swiglu", **dims
        )
        == fp
    )
    assert bi.fingerprint("fused_moe_nvfp4", arch="sm_107", act=SwiGLU(), **dims) == fp
    assert (
        bi.fingerprint("fused_moe_nvfp4", arch="sm_100", **dims) != fp
    )  # the K = 64 instruction class: another class
    assert bi.fingerprint("fused_moe_nvfp4", arch="sm_103", **dims) == bi.fingerprint(
        "fused_moe_nvfp4", arch="sm_100", **dims
    )
    assert bi.fingerprint("fused_moe_nvfp4", arch=(10, 7), **dims) == fp
    assert bi.fingerprint("fused_moe_nvfp4", arch="sm_107", act=ReLU2(), **dims) != fp
    assert bi.fingerprint(
        "fused_moe_nvfp4", arch="sm_107", act="relu2", **dims
    ) == bi.fingerprint("fused_moe_nvfp4", arch="sm_107", act=ReLU2(), **dims)
    assert (
        bi.fingerprint(
            "fused_moe_nvfp4", arch="sm_107", method=RoutingMethodType.Default, **dims
        )
        != fp
    )
    assert bi.fingerprint("fused_moe_nvfp4", arch="sm_107", **dims) != bi.fingerprint(
        "fused_moe", **dims
    )
    fps = {
        bi.fingerprint("fused_moe_nvfp4", arch="sm_107", act=a, **dims)
        for a in ACTIVATIONS
    }
    assert len(fps) == len(ACTIVATIONS)
    # the routing methods of the bf16 backend's surface (TopK, Llama4's TopKSigmoid, the grouped DeepSeek-V3 selection) are
    # the class's as well: a method is part of the identifier; a method the classes do not implement is refused
    assert (
        bi.fingerprint(
            "fused_moe_nvfp4", arch="sm_107", method=RoutingMethodType.Llama4, **dims
        )
        != fp
    )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_nvfp4", arch="sm_107", method=RoutingMethodType.MiniMax2, **dims
        )
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_nvfp4", arch="sm_107", act="tanh", **dims)
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_nvfp4", arch="sm_90", **dims)
    assert bi.class_name("fused_moe_nvfp4") == "moe.nvfp4"


def test_shape_rules_accept_the_model_grid_and_raise_elsewhere():
    """The complete acceptance rule of the class: the real-model geometries pass; shapes outside it raise the mode's error at
    weight preparation and at the op's entry, naming the limit; the class never routes a shape elsewhere."""
    import itertools

    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import check_shape_nvfp4

    Es, ks = (8, 16, 32, 64, 128, 256, 384, 512, 1024), (1, 2, 4, 6, 8, 10, 16)
    Is, Hs = (
        (256, 512, 640, 768, 1024, 1408, 1536, 2048, 2816, 4096, 14336),
        (2048, 2560, 3072, 4096, 5120, 7168),
    )
    n = 0
    for I, H, E, k in itertools.product(Is, Hs, Es, ks):
        if k > E:
            continue
        check_shape_nvfp4(H, I, E, k)
        check_shape_nvfp4(H, I, E, k, gated=False)
        check_shape_nvfp4(
            None, I, E, k
        )  # the configuration-time rule (no hidden size yet)
        n += 1
    assert n == len(Is) * len(Hs) * sum(1 for E in Es for k in ks if k <= E)
    for args, why in (
        ((2048 + 16, 768, 128), "multiples of 32"),
        ((2048, 768 + 16, 128), "multiples of 32"),
        ((2048, 768, 0), "num_experts"),
        ((2048, 768, 1025), "num_experts"),
        ((2048, 768, 8, 9), "top_k"),
        ((2048, 768, 128, 17), "top_k"),
        ((2048, 768, 128, 0), "top_k"),
        ((4096, 64, 8), "unsplit"),
    ):
        with pytest.raises(BatchInvariantUnsupportedError, match=why):
            check_shape_nvfp4(*args)
    check_shape_nvfp4(4096, 128, 8)
    with pytest.raises(BatchInvariantUnsupportedError, match="unsplit"):
        check_shape_nvfp4(
            4096, 128, 8, gated=False
        )  # the ungated GEMM1 weight (I, H) is the narrower one
    check_shape_nvfp4(
        None, 128, 8, gated=False
    )  # the chunk rule waits for the hidden size
    with pytest.raises(BatchInvariantUnsupportedError, match="multiples of 32"):
        check_shape_nvfp4(None, 768 + 16, 128)


@needs_device
@pytest.mark.parametrize(
    "H, I, E, k",
    [
        (2048, 768, 128, 8),
        (4096, 1536, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (2048, 1024, 128, 6),
    ],
    ids=["qwen3_30b", "qwen3_235b", "mixtral_8x7b", "deepseek_v3", "nemotron3_30b"],
)
def test_layer_constructs_for_the_real_model_geometries(H, I, E, k):
    """The layer accepts the backend for the real-model geometries at construction (the configuration fixes I, E and k; the
    hidden size enters at the first call) and the first call runs: the construction-time rule must not stand in for the
    hidden size."""
    act = ReLU2() if (H, I) == (2048, 1024) else _SWIGLU
    layer = _layer(k, E, I, act=act)
    assert isinstance(layer.runners[0], BatchInvariantFp4Runner)
    assert isinstance(layer.runners[0].impl, BatchInvariantMoeNvfp4Runner)
    if 3 * E * I * H * 2 > 2**33:
        return  # the construction is the point; the largest experts exceed a per-test allocation
    x, logits, w1, w2 = _problem(3, k, H, I, E, seed=H + I, gated=act is _SWIGLU)
    out = layer(_logits_pack(x, logits), _weights(w1, w2, act))
    out = out if isinstance(out, torch.Tensor) else out[0]
    assert (
        out.shape == (3, H)
        and out.dtype == torch.bfloat16
        and torch.isfinite(out.float()).all()
    )


def _adversarial(n: int, seed: int) -> torch.Tensor:
    f32 = torch.float32
    special = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-149,
            -(2.0**-149),
            2.0**-127,
            1e-30,
            -1e-30,
            1e-3,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            8.0,
            87.0,
            -87.0,
            88.0,
            -88.0,
            89.0,
            1e4,
            -1e4,
            1e30,
            -1e30,
            3.0e38,
            -3.0e38,
        ],
        dtype=f32,
    )
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randn(n, generator=g) * torch.exp2(
        torch.randint(-12, 12, (n,), generator=g).float()
    )
    return torch.cat([special, rnd.to(f32)])


@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_vectorised_and_exact_chains_agree_on_adversarial_accumulators(act):
    """The vectorised W4A4 chains (the GPU suites' oracle) and the exact scalar model give the same bf16 bits on adversarial
    accumulators, alphas and per-token scales for every activation; the row scale enters once, before the chain."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_nvfp4 as RQ
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_nvfp4 as Q

    a = S.activation_of(act)
    g = _adversarial(200, 40)
    u = torch.roll(_adversarial(200, 41), 5)
    n = g.numel()
    gen = torch.Generator().manual_seed(42)
    alpha = torch.exp2((torch.rand(n, generator=gen) * 2 - 1) * 10).float()
    pts = torch.exp2((torch.rand(n, generator=gen) * 2 - 1) * 10).float()
    if a.gated:
        vec = RQ.w4a4_gemm1_cpu(
            g.view(-1, 1), u.view(-1, 1), alpha, pts, a.act, a.params
        ).view(-1)
        exact = [
            Q.glu_scaled_elem(
                R.ExactF32,
                gi,
                ui,
                Q.w4a4_row_scale(R.ExactF32, al, p),
                a.act,
                "bf16",
                a.params,
            )
            for gi, ui, al, p in zip(
                g.tolist(), u.tolist(), alpha.tolist(), pts.tolist(), strict=True
            )
        ]
    else:
        vec = RQ.w4a4_act_cpu(g.view(-1, 1), alpha, pts, a.act, a.params).view(-1)
        exact = [
            Q.act_scaled_elem(
                R.ExactF32,
                gi,
                Q.w4a4_row_scale(R.ExactF32, al, p),
                a.act,
                "bf16",
                a.params,
            )
            for gi, al, p in zip(g.tolist(), alpha.tolist(), pts.tolist(), strict=True)
        ]
    ex = torch.tensor(exact, dtype=torch.float32).to(torch.bfloat16)
    bad = (bits(vec) != bits(ex)).nonzero().flatten()[:8].tolist()
    assert not bad, [
        (i, g[i].item(), alpha[i].item(), pts[i].item(), vec[i].item(), ex[i].item())
        for i in bad
    ]
    vec2 = RQ.w4a4_gemm2_cpu(g.view(-1, 1), alpha, pts).view(-1)
    ex2 = torch.tensor(
        [
            Q.scaled_out_elem(R.ExactF32, gi, Q.w4a4_row_scale(R.ExactF32, al, p))
            for gi, al, p in zip(g.tolist(), alpha.tolist(), pts.tolist(), strict=True)
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    assert torch.equal(bits(vec2), bits(ex2))


@pytest.mark.parametrize(
    "family,sm_count",
    [("sm_107", 212), ("sm_107", 204), ("sm_100", 148), ("sm_103", 152)],
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for the bench geometries and token counts lies in the grouped block-scaled space of its GEMM at
    the part's instruction class, for the gated and the ungated chains, at the nominal parts; the space has the dense
    nvfp4 records at swizzle 1 and no other."""
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        GEMM2_CHAIN,
        MoEWeightsNVFP4,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        active_experts,
        all_valid_configs_nvfp4,
        all_valid_configs_nvfp4_swapped,
        default_configs_nvfp4,
        grouped_knobs_blockscaled,
        padded_rows,
        rank_grouped_nvfp4,
        rows_per_expert,
        swapped_knobs_blockscaled,
        token_tile_default,
    )
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    space = grouped_knobs_blockscaled()
    assert len(space) == 72 and all(
        k.swizzle_size == 1 and k.cluster_k == 1 and not k.swap_ab and not k.b_reuse
        for k in space
    )
    assert {k.tile_n for k in space} == {64, 128, 192, 256} and {
        k.tile_m for k in space
    } == {128, 256}
    # the swapped decode space: one 1-CTA record per narrow token tile, disjoint from the grouped space
    swap_space = swapped_knobs_blockscaled()
    assert [k.tile_n for k in swap_space] == [8, 16, 32] and all(
        k.swap_ab
        and k.tile_m == 128
        and k.cluster_shape_mn == (1, 1)
        and not k.use_2cta_instrs
        and k.well_formed("nvfp4")
        and k.well_formed("nvfp4_k128")
        for k in swap_space
    )
    assert not set(swap_space) & set(space)
    assert [token_tile_default(r) for r in (1, 6, 7, 12, 13, 24, 25, 32, 128)] == [
        8,
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    for H, I, E, k in (
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (2560, 640, 512, 10),
        (2048, 1024, 128, 6),
        (512, 256, 64, 6),
    ):
        for gated in (True, False):
            in1, in2 = expert_classes(H, I, family, gated)
            assert (in1 == "nvfp4_k128") == (family == "sm_107") and in1 == in2
            chain1 = MoEWeightsNVFP4(
                *([None] * 6), H, I, gated, "silu" if gated else "relu2"
            ).chain1
            n1 = 2 * I if gated else I
            s1 = set(all_valid_configs_nvfp4(n1, H, E, chain1, in1))
            s2 = set(all_valid_configs_nvfp4(H, I, E, GEMM2_CHAIN, in2))
            assert len(s1) == 72 and len(s2) == 72
            w1 = set(all_valid_configs_nvfp4_swapped(n1, H, E, chain1, in1))
            w2 = set(all_valid_configs_nvfp4_swapped(H, I, E, GEMM2_CHAIN, in2))
            assert len(w1) == 3 and len(w2) == 3
            for T in (1, 7, 64, 300, 800, 4096, 16384):
                c1, c2 = default_configs_nvfp4(
                    T,
                    k,
                    H,
                    I,
                    E,
                    (in1, in2),
                    chain1,
                    GEMM2_CHAIN,
                    sm_count,
                    family,
                    nominal=True,
                )
                assert c1 in s1 | w1 and c2 in s2 | w2, (
                    family,
                    T,
                    H,
                    I,
                    E,
                    k,
                    gated,
                    c1.describe(),
                    c2.describe(),
                )
                # the ranking decides the body (both record spaces ranked together): the pick is the ranked list's head; the
                # swapped body forced is a swapped record, forbidden a grouped one
                rows = rows_per_expert(T, k, E)
                a1, a2 = default_configs_nvfp4(
                    T,
                    k,
                    H,
                    I,
                    E,
                    (in1, in2),
                    chain1,
                    GEMM2_CHAIN,
                    sm_count,
                    family,
                    nominal=True,
                    swap=True,
                )
                assert a1 in w1 and a2 in w2 and a1.swap_ab and a2.swap_ab
                d1, d2 = default_configs_nvfp4(
                    T,
                    k,
                    H,
                    I,
                    E,
                    (in1, in2),
                    chain1,
                    GEMM2_CHAIN,
                    sm_count,
                    family,
                    nominal=True,
                    swap=False,
                )
                assert d1 in s1 and d2 in s2 and not d1.swap_ab and not d2.swap_ab
                ranked = rank_grouped_nvfp4(
                    rows,
                    H,
                    I,
                    E,
                    GEMM2_CHAIN,
                    in2,
                    sm_count,
                    family,
                    3,
                    nominal=True,
                    n_active=active_experts(T, k, E),
                )
                assert (
                    1 <= len(ranked) <= 3
                    and len(set(ranked)) == len(ranked)
                    and all(c in s2 | w2 for c in ranked)
                    and ranked[0] == c2
                )
                # one row per active expert ranks the swapped body first on both GEMMs; at a group of many padded atoms no
                # swapped record ranks first (the regret of the ranking between is measured, not asserted)
                if rows == 1:
                    assert c1.swap_ab and c2.swap_ab, (
                        family,
                        T,
                        H,
                        I,
                        E,
                        k,
                        c1.describe(),
                    )
                if rows > 128:
                    assert not c1.swap_ab and not c2.swap_ab, (
                        family,
                        T,
                        H,
                        I,
                        E,
                        k,
                        c1.describe(),
                        c2.describe(),
                    )
                # the tile-fit guard: at a group of one padded atom the picks and the ranked list are 128-row tiles (a
                # 256-row 2-CTA tile would compute a second atom of padding); at taller groups every tile is a candidate
                if padded_rows(rows_per_expert(T, k, E)) == 128:
                    assert c1.tile_m == 128 and c2.tile_m == 128, (
                        family,
                        T,
                        H,
                        I,
                        E,
                        k,
                        c1.describe(),
                        c2.describe(),
                    )
                    assert all(c.tile_m == 128 for c in ranked)


def test_unsupported_device_raises_before_any_kernel():
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    # the pair implementation and the fp4 backend's runner over it refuse alike (the backend forwards the pair's check)
    for cls in (BatchInvariantMoeNvfp4Runner, BatchInvariantFp4Runner):
        runner = cls(_config(8, 64, 256), torch.device("cpu"))
        with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
            runner._check_support()
        assert "moe.nvfp4" in str(excinfo.value) and "cpu" in str(excinfo.value)


# ----------------------------------------------------------------------------- GPU


def _reference_route(logits: torch.Tensor, k: int, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _dense_fp4(a4, sf, w, wsf, in_fmt):
    """The dense batch-invariant NVFP4 GEMM (``mm_fp4``'s class, the instruction class pinned) with a unit alpha and the fp32
    output: the raw accumulators of the class."""
    from flashinfer.gemm.batch_invariant.api import mm_fp4_batch_invariant

    return mm_fp4_batch_invariant(
        a4,
        w.t(),
        sf,
        wsf,
        torch.ones(1, device=a4.device),
        torch.float32,
        in_fmt=in_fmt,
    )


def _reference_ffn(x_q, ids, weights, W, act, in_fmts, gs_inv=None):
    """The class composed from its own pieces, independently of the fused op: the (given) per-token quantized rows, per
    expert the dense batch-invariant ``mm_fp4`` on the expert's rows (the instruction class pinned, a unit alpha, fp32), the
    CPU W4A4 GEMM1 chain of the activation, the quantizer class's exact reference on the bf16 intermediate rows, ``mm_fp4``
    per expert for GEMM2 with the CPU GEMM2 chain, the CPU slot-order combine. Bit for bit what the fused op must produce."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        quantize_reference,
        swizzle_sf_128x4,
        unswizzle_sf_128x4,
    )
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_nvfp4 as RQ
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import TE_GS_INV
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP

    gs_inv = TE_GS_INV if gs_inv is None else gs_inv
    x4, x_sf, x_pts = x_q
    T, Hh = x4.shape
    H, I, E, k = 2 * Hh, W.intermediate_size, W.num_experts, ids.shape[1]
    dev = x4.device
    sf_rows = unswizzle_sf_128x4(x_sf.reshape(-1).cpu(), T, H)
    ids_c = ids.cpu()
    y = torch.zeros(T, k, H, dtype=torch.bfloat16)
    for e in range(E):
        tok, slot = (ids_c == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        rows = tok.to(dev)
        acc1 = _dense_fp4(
            x4[rows].contiguous(),
            swizzle_sf_128x4(sf_rows[tok], tok.numel(), H).to(dev),
            W.w1[e],
            W.w1_sf[e],
            in_fmts[0],
        ).cpu()
        al1 = torch.full((tok.numel(),), W.w1_alpha[e].item())
        pts1 = x_pts.cpu()[tok]
        if act.gated:
            u = acc1.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
            g = acc1.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
            a = RQ.w4a4_gemm1_cpu(g, u, al1, pts1, act.act, act.params)
        else:
            a = RQ.w4a4_act_cpu(acc1, al1, pts1, act.act, act.params)
        a4, a_sf, a_pts = quantize_reference(a, gs_inv, per_token=True)
        acc2 = _dense_fp4(
            a4.to(dev), a_sf.reshape(-1).to(dev), W.w2[e], W.w2_sf[e], in_fmts[1]
        ).cpu()
        y[tok, slot] = RQ.w4a4_gemm2_cpu(
            acc2, torch.full((tok.numel(),), W.w2_alpha[e].item()), a_pts
        )
    ident = torch.arange(T * k, dtype=torch.int32).view(T, k)
    return R.combine_cpu(y.view(T * k, H), ident, weights.cpu())


def _quantized(x: torch.Tensor, check: bool = True):
    """The class's per-token quantization of the activations on the device, checked against the quantizer's exact reference."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        quantize_reference,
    )
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        TE_GS_INV,
        quantize_activations,
    )

    x4, sf, pts = quantize_activations(x)
    if check:
        q_ref, sf_ref, pts_ref = quantize_reference(x, TE_GS_INV, per_token=True)
        assert (
            torch.equal(x4.cpu(), q_ref)
            and torch.equal(sf.cpu(), sf_ref.reshape(-1))
            and torch.equal(bits(pts.cpu()), bits(pts_ref))
        )
    return x4, sf, pts


def _weights_of(pack: MoEWeightPack):
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import MoEWeightsNVFP4

    return MoEWeightsNVFP4.from_view(pack.get_view(KEY))


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits, bf16 activations) == the class's routing on the CPU, the class's per-token quantization (==
    the quantizer's reference), the dense batch-invariant ``mm_fp4`` per expert with the CPU W4A4 chain of the activation,
    the quantizer's reference on the intermediate, ``mm_fp4`` + the CPU GEMM2 chain, the CPU combine: bit for bit, for
    every activation of the layer, at the instruction class the part runs."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    T, k, H, I, E = 120, 4, 512, 256, 16
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=1, gated=chain.gated)
    layer = _layer(k, E, I, act)
    W_pack = _weights(w1, w2, act)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    W = _weights_of(W_pack)
    ref = _reference_ffn(
        _quantized(x),
        ids.cuda(),
        weights,
        W,
        chain,
        expert_classes(H, I, _family(), chain.gated),
    )
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
def test_swapped_body_equals_the_dense_class_per_expert_at_every_token_tile():
    """The swapped decode body (weights on the MMA's M side, the dense packing's token rows as a narrow N tile, their scale
    factors gathered from the padded positions): at every token tile (8 / 16 / 32) its fp32 accumulators == the dense
    batch-invariant ``mm_fp4`` on the group's rows per expert, the GEMM1 chains (the gated chain, the ungated smoke chain,
    the per-column token scale) and the GEMM2 chain == the CPU W4A4 chains of those accumulators. Groups spanning several
    token tiles (uneven loads), one-row groups (T = 1), empty groups, E = 256, N not a multiple of 128; rows past the live
    rows are never stored."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        swizzle_sf_128x4,
        unswizzle_sf_128x4,
    )
    from flashinfer.fused_moe.batch_invariant import reference_nvfp4 as RQ
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        DENSE_TAIL,
        GEMM2_CHAIN,
        grouped_gemm_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.compile import (
        can_implement_swapped_blockscaled as can_implement_swapped_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP, W4A4Chain
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        swapped_knobs_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    f32_chain = W4A4Chain("plain", "silu", "f32", (), "none")
    relu2_chain = W4A4Chain("act", "relu2", "bf16", (), "w4a4")
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
        in1, _ = expert_classes(K, I, _family())
        x4, sf, pts = _quantized(x)
        g = torch.Generator().manual_seed(E + 7)
        src = torch.randperm(T * kk, generator=g)[:Tk]
        tok = src // kk
        off = [0]
        offp = [0]
        for c in counts:
            off.append(off[-1] + c)
            offp.append(offp[-1] + -(-c // 128) * 128)
        Tk_pad = offp[-1]
        pos_pad = torch.cat(
            [torch.arange(offp[e], offp[e] + c) for e, c in enumerate(counts) if c]
        )
        a_dense = torch.zeros(Tk + DENSE_TAIL, K // 2, dtype=torch.uint8, device="cuda")
        a_dense[:Tk] = x4[tok.cuda()]
        sf_rows = unswizzle_sf_128x4(sf.cpu(), T, K)
        sf_pad_rows = torch.zeros(
            Tk_pad + DENSE_TAIL, sf_rows.shape[1], dtype=sf_rows.dtype
        )
        sf_pad_rows[pos_pad] = sf_rows[tok]
        sf_pad = swizzle_sf_128x4(sf_pad_rows, Tk_pad + DENSE_TAIL, K).cuda()
        pts_pad = torch.zeros(Tk_pad + DENSE_TAIL, dtype=torch.float32)
        pts_pad[pos_pad] = pts.cpu()[tok]
        pts_pad = pts_pad.cuda()
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        ref32 = torch.zeros(Tk, N, dtype=torch.float32)
        alpha_rows = torch.zeros(Tk)
        for e, c in enumerate(counts):
            if c:
                rows = tok[off[e] : off[e + 1]]
                ref32[off[e] : off[e + 1]] = _dense_fp4(
                    x4[rows.cuda()].contiguous(),
                    swizzle_sf_128x4(sf_rows[rows], rows.numel(), K).cuda(),
                    W.w1[e],
                    W.w1_sf[e],
                    in1,
                ).cpu()
                alpha_rows[off[e] : off[e + 1]] = W.w1_alpha[e].item()
        pts_dense = pts.cpu()[tok]
        u = ref32.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
        gt = ref32.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
        ref_glu = RQ.w4a4_gemm1_cpu(gt, u, alpha_rows, pts_dense, "silu", ())
        ref_act = RQ.w4a4_act_cpu(ref32, alpha_rows, pts_dense, "relu2", ())
        ref_plain = RQ.w4a4_gemm2_cpu(ref32, alpha_rows, pts_dense)
        for cfg in swapped_knobs_blockscaled():
            assert can_implement_swapped_nvfp4(cfg, N, K, E, f32_chain, in1), (
                cfg.describe(),
                N,
                K,
                E,
            )
            out32 = torch.full(
                (Tk + DENSE_TAIL, N), float("nan"), dtype=torch.float32, device="cuda"
            )
            grouped_gemm_nvfp4(
                a_dense,
                sf_pad,
                pts_pad,
                W.w1,
                W.w1_sf,
                W.w1_alpha,
                cnt,
                cfg,
                f32_chain,
                in1,
                out=out32,
                dense_a=True,
            )
            assert torch.equal(bits(out32[:Tk].cpu()), bits(ref32)), (
                counts[:4],
                cfg.describe(),
                in1,
            )
            assert torch.isnan(
                out32[Tk:]
            ).all()  # rows past the live rows are never stored
            got = grouped_gemm_nvfp4(
                a_dense,
                sf_pad,
                pts_pad,
                W.w1,
                W.w1_sf,
                W.w1_alpha,
                cnt,
                cfg,
                W.chain1,
                in1,
                dense_a=True,
            )
            assert torch.equal(bits(got[:Tk].cpu()), bits(ref_glu)), (
                counts[:4],
                cfg.describe(),
                "glu",
            )
            got = grouped_gemm_nvfp4(
                a_dense,
                sf_pad,
                pts_pad,
                W.w1,
                W.w1_sf,
                W.w1_alpha,
                cnt,
                cfg,
                relu2_chain,
                in1,
                dense_a=True,
            )
            assert torch.equal(bits(got[:Tk].cpu()), bits(ref_act)), (
                counts[:4],
                cfg.describe(),
                "relu2",
            )
            got = grouped_gemm_nvfp4(
                a_dense,
                sf_pad,
                pts_pad,
                W.w1,
                W.w1_sf,
                W.w1_alpha,
                cnt,
                cfg,
                GEMM2_CHAIN,
                in1,
                dense_a=True,
            )
            assert torch.equal(bits(got[:Tk].cpu()), bits(ref_plain)), (
                counts[:4],
                cfg.describe(),
                "plain",
            )
            # a swapped record needs the dense packing
            with pytest.raises(ValueError):
                grouped_gemm_nvfp4(
                    a_dense,
                    sf_pad,
                    pts_pad,
                    W.w1,
                    W.w1_sf,
                    W.w1_alpha,
                    cnt,
                    cfg,
                    f32_chain,
                    in1,
                )


@needs_device
def test_both_packings_and_the_prequantized_pack_forms_give_the_bits():
    """The dense and the padded packing of the expanded rows are placements (the same bits); the pack forms agree: bf16 rows
    (quantized by the class), the per-token quantized pack with the 128x4 swizzled scales and with the linear scales, and the
    per-tensor pack of the TRTLLM backends (FlashInfer's quantizer at a unit global scale, no per-token scale), which the class takes
    at unit per-token scales and which equals the class composed on those bytes."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import fused_moe_nvfp4
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    T, k, H, I, E = 257, 6, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2)
    layer = _layer(k, E, I)
    W_pack = _weights(w1, w2)
    W = _weights_of(W_pack)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    # the two packings of the op, and the two bodies of the dense packing (the swapped decode body on every GEMM, the padded body)
    out_d = fused_moe_nvfp4(x, ids, weights, W, path="dense")
    out_p = fused_moe_nvfp4(x, ids, weights, W, path="padded")
    assert torch.equal(bits(out_d), bits(out_p)) and torch.equal(bits(out_d), bits(out))
    out_s = fused_moe_nvfp4(x, ids, weights, W, path="dense", swap=True)
    out_n = fused_moe_nvfp4(x, ids, weights, W, path="dense", swap=False)
    assert torch.equal(bits(out_s), bits(out)) and torch.equal(bits(out_n), bits(out))
    with pytest.raises(ValueError):
        fused_moe_nvfp4(x, ids, weights, W, path="padded", swap=True)
    # the pre-quantized packs: the class's own quantization in both scale layouts (a pack with per-token scales on a layer
    # configured for them; the layer's shared contract refuses the scale otherwise)
    x4, sf, pts = _quantized(x)
    with pytest.raises(ValueError, match="per_token_scale"):
        layer(
            MoEActivationPack(
                x4,
                sf,
                topk_ids=ids,
                topk_weights=weights,
                routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
                per_token_scale=pts,
            ),
            W_pack,
        )
    layer_pts = _layer(k, E, I, quant=_NVFP4_PTS)
    out_q = layer_pts(
        MoEActivationPack(
            x4,
            sf,
            topk_ids=ids,
            topk_weights=weights,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            per_token_scale=pts,
        ),
        W_pack,
    )
    assert torch.equal(bits(out_q), bits(out))
    sf_lin = unswizzle_sf_128x4(sf.cpu(), T, H).cuda().view(torch.float8_e4m3fn)
    out_l = layer_pts(
        MoEActivationPack(
            x4,
            sf_lin,
            topk_ids=ids,
            topk_weights=weights,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            per_token_scale=pts,
        ),
        W_pack,
    )
    assert torch.equal(bits(out_l), bits(out))
    # the per-tensor pack of the TRTLLM backends (unit global scale, linear scales, no per-token scale): the class at unit pts
    # the per-tensor pack of the TRTLLM backends (what their ``prepare_activations`` produces: FlashInfer's quantizer at a unit
    # global scale, the linear scale layout), built here from the quantizer itself
    q_t, sf_t = fp4_quantize(
        x,
        global_scale=torch.ones(1, device=x.device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    sf_t = sf_t.view(torch.float8_e4m3fn).reshape(T, -1)
    out_t = layer(
        MoEActivationPack(
            q_t,
            sf_t,
            topk_ids=ids,
            topk_weights=weights,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        ),
        W_pack,
    )
    from flashinfer.batch_invariant.quantization.nvfp4_reference import swizzle_sf_128x4

    sf_t_sw = swizzle_sf_128x4(sf_t.view(torch.uint8).cpu(), T, H).cuda()
    ref_t = _reference_ffn(
        (q_t.view(torch.uint8), sf_t_sw, torch.ones(T, device="cuda")),
        ids,
        weights,
        W,
        activation_of(_SWIGLU),
        expert_classes(H, I, _family()),
    )
    assert torch.equal(bits(out_t.cpu()), bits(ref_t))
    assert not torch.equal(
        bits(out_t), bits(out)
    )  # another quantization of the input: another function of x (an observation)


@needs_device
def test_swapped_gather_record_gives_the_box_forms_bits_on_every_pack_form():
    """The swapped records' gather form (the body reads the token rows and their scale factors through the plan's map: no scatter
    launch, no permuted tensor) is a placement: the box form's bits for the bf16 input (quantized over the T rows), the
    pre-quantized triple with the 128x4 and with the linear scale layout, the per-tensor pack (unit per-token scales), the static
    class, every swapped token tile, at decode and beyond it, twice. The record is the swapped body's: a grouped record with the
    flag is refused, so is the padded packing and GEMM2; the ranking's three forms (excluded / forced / ranked) and the tactic
    round trip carry it."""
    import dataclasses

    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        MoEWeightsNVFP4,
        fused_moe_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.compile import (
        can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        GATHER_TILE_N,
        default_configs_nvfp4,
        gather_form_rule,
        gather_twins,
        rank_grouped_nvfp4,
        swapped_knobs_blockscaled,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs

    twins = gather_twins(swapped_knobs_blockscaled())
    assert [t.tile_n for t in twins] == [8, 16, 32] and all(
        t.gather_a
        and t.swap_ab
        and dataclasses.replace(t, gather_a=False) in swapped_knobs_blockscaled()
        for t in twins
    )
    # the structural rule: the narrow swapped tile's gather form when the box form would scatter in a launch of its own
    sw = swapped_knobs_blockscaled()
    assert [gather_form_rule(c, True).gather_a for c in sw] == [
        c.tile_n == GATHER_TILE_N for c in sw
    ]
    assert not any(gather_form_rule(c, False).gather_a for c in sw)
    grouped = Knobs((128, 256), (1, 1), False, True, "n", 1)
    assert gather_form_rule(grouped, True) == grouped
    H, I, E = 512, 256, 32
    for T, k in ((3, 4), (16, 6), (257, 6)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2 + T)
        W = _weights_of(_weights(w1, w2))
        ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        ref = bits(fused_moe_nvfp4(x, ids, weights, W, gather=False))
        in1, in2 = W.classes(_family())
        _, c2 = default_configs_nvfp4(
            T, k, H, I, E, (in1, in2), W.chain1, W.chain2, 212, _family()
        )
        n1 = 2 * I
        x4, sf, pts = _quantized(x)
        sf_lin = (
            unswizzle_sf_128x4(sf.cpu(), T, H)
            .cuda()
            .view(torch.uint8)
            .reshape(T, H // 16)
        )
        ref_u = bits(fused_moe_nvfp4((x4, sf_lin, None), ids, weights, W, gather=False))
        for rec in twins:
            assert can_implement_swapped_blockscaled(rec, n1, H, E, W.chain1, in1)
            out, im = fused_moe_nvfp4(
                x, ids, weights, W, config1=rec, config2=c2, return_intermediates=True
            )
            assert (
                im["config1"].gather_a and im["x_perm4"] is None
            )  # no permuted tensor
            assert torch.equal(bits(out), ref), rec.describe()
            assert torch.equal(
                bits(
                    fused_moe_nvfp4(
                        (x4, sf, pts), ids, weights, W, config1=rec, config2=c2
                    )
                ),
                ref,
            )
            assert torch.equal(
                bits(
                    fused_moe_nvfp4(
                        (x4, sf_lin, pts), ids, weights, W, config1=rec, config2=c2
                    )
                ),
                ref,
            )
            assert torch.equal(
                bits(
                    fused_moe_nvfp4(
                        (x4, sf_lin, None), ids, weights, W, config1=rec, config2=c2
                    )
                ),
                ref_u,
            )
        assert torch.equal(
            bits(fused_moe_nvfp4(x, ids, weights, W, config1=twins[0], config2=c2)), ref
        )  # twice
        Ws = MoEWeightsNVFP4.prepare(w1, w2, intermediate="static")
        ref_s = bits(fused_moe_nvfp4(x, ids, weights, Ws, gather=False))
        assert torch.equal(
            bits(fused_moe_nvfp4(x, ids, weights, Ws, config1=twins[1], config2=c2)),
            ref_s,
        )
        if T <= 16:
            out_g, im = fused_moe_nvfp4(
                x, ids, weights, W, gather=True, return_intermediates=True
            )
            assert (
                im["config1"].gather_a
                and im["config1"].swap_ab
                and torch.equal(bits(out_g), ref)
            )
            # the default: the rule's form -- the pre-quantized input's narrow swapped tile gathers, the bf16 input's quantizer scatters
            out_d, im = fused_moe_nvfp4(
                (x4, sf, pts), ids, weights, W, return_intermediates=True
            )
            c = im["config1"]
            assert (
                c.gather_a == (c.swap_ab and c.tile_n == GATHER_TILE_N)
                and (im["x_perm4"] is None) == c.gather_a
            )
            assert torch.equal(bits(out_d), ref)
            _, im = fused_moe_nvfp4(x, ids, weights, W, return_intermediates=True)
            assert not im["config1"].gather_a and im["x_perm4"] is not None
    # the ranking's forms and the refusals
    rows = 1
    r_g = rank_grouped_nvfp4(
        rows, n1, H, E, W.chain1, in1, 212, _family(), 6, n_active=8, gather=True
    )
    r_b = rank_grouped_nvfp4(
        rows, n1, H, E, W.chain1, in1, 212, _family(), 6, n_active=8
    )
    r_n = rank_grouped_nvfp4(
        rows, n1, H, E, W.chain1, in1, 212, _family(), 6, n_active=8, gather=None
    )
    assert r_g[0].gather_a and all(c.gather_a or not c.swap_ab for c in r_g)
    assert not any(c.gather_a for c in r_b)
    assert any(c.gather_a for c in r_n) and any(
        c.swap_ab and not c.gather_a for c in r_n
    )
    grouped_ga = Knobs((128, 256), (1, 1), False, True, "n", 1, gather_a=True)
    assert not can_implement_grouped_blockscaled(grouped_ga, n1, H, E, W.chain1, in1)
    assert not can_implement_swapped_blockscaled(grouped_ga, n1, H, E, W.chain1, in1)
    with pytest.raises(ValueError):
        fused_moe_nvfp4(x, ids, weights, W, path="padded", gather=True)
    with pytest.raises(ValueError):
        fused_moe_nvfp4(x, ids, weights, W, config1=c2, config2=twins[0])
    with pytest.raises(ValueError):
        fused_moe_nvfp4(x, ids, weights, W, config1=grouped_ga, config2=c2)


@needs_device
@pytest.mark.parametrize("body", ["padded", "swapped"])
def test_pdl_chain_alternating_inputs_give_each_call_its_own_bits(body):
    """The op's kernels run under programmatic dependent launch, each warp waiting for the predecessor grid before its first read
    of the predecessor's output (the weight loads start before the wait). Two inputs alternate over many calls, so every
    intermediate buffer the caching allocator hands back holds the OTHER call's bytes: a read that did not wait would take them.
    Every call gives the bits of its own clean reference, for the padded body and the swapped decode body."""
    import os

    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import fused_moe_nvfp4

    T, k, H, I, E = 16, 8, 2048, 768, 128
    W = None
    calls, refs = [], []
    for seed in (5, 6):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed)
        if W is None:
            W = _weights_of(_weights(w1, w2))
        ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        calls.append((x, ids, weights))
        refs.append(
            bits(fused_moe_nvfp4(x, ids, weights, W, swap=body == "swapped").cpu())
        )
    assert not torch.equal(refs[0], refs[1])
    for i in range(int(os.environ.get("FLASHINFER_BI_STRESS_ITERS", "20"))):
        for (x, ids, weights), ref in zip(calls, refs, strict=True):
            out = fused_moe_nvfp4(x, ids, weights, W, swap=body == "swapped")
            assert torch.equal(bits(out.cpu()), ref), (i, body)


@needs_device
def test_packed_f32_weights_are_rounded_by_the_combine():
    """The packed pre-routed form takes f32 weights that are not bf16 values and rounds them in the combine (the packed word's
    convention, no host pass): the output equals the same pack given the caller-rounded weights, and the unpacked form given
    the bf16 weights, bit for bit."""
    T, k, H, I, E = 133, 4, 512, 256, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    raw = (
        weights.float() * (1.0 + 2.0**-12)
    ).cuda()  # not bf16 values: the rounding is real work
    assert not torch.equal(raw.to(torch.bfloat16).float(), raw)
    out_raw = layer(_routed_pack(x, ids.cuda(), raw, packed=True), W)
    out_rounded = layer(
        _routed_pack(x, ids.cuda(), raw.to(torch.bfloat16), packed=True), W
    )
    out_unpacked = layer(_routed_pack(x, ids.cuda(), raw.to(torch.bfloat16)), W)
    assert torch.equal(bits(out_raw), bits(out_rounded))
    assert torch.equal(bits(out_raw), bits(out_unpacked))


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
@pytest.mark.parametrize(
    "T,k,E",
    [(1, 1, 8), (3, 8, 64), (16, 16, 128), (16, 8, 1024), (7, 4, 1000), (0, 4, 16)],
)
@pytest.mark.parametrize("method", ["softmax_renorm", "sigmoid_bias_renorm_scaled"])
def test_route_plan_dual_equals_route_and_the_dual_plan(T, k, E, method):
    """``api_nvfp4.route_plan_dual`` (the routing and the dense packing's dual permutation plan in one launch for small token
    counts) gives the route kernel's bits and the dual plan's integers: experts, weights, counts, dest, src, dest_pad and
    pos_pad equal those of ``route`` followed by ``permute_plan_dual``, up to the fused kernel's slot bound and expert count."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import api_nvfp4 as A
    from flashinfer.fused_moe.batch_invariant import spec as S

    meth = (
        S.SOFTMAX_RENORM if method == "softmax_renorm" else S.SIGMOID_BIAS_RENORM_SCALED
    )
    g = torch.Generator().manual_seed(T * 7 + E)
    logits = (torch.randn(max(T, 1), E, generator=g) * 2).float()[:T].cuda()
    bias = (torch.randn(E, generator=g) * 0.1).float().cuda() if meth.bias else None
    scale = 2.5 if meth.scaled else None
    ex, w = api.route(logits, k, meth, bias, scale)
    counts, dest, src, dest_pad, pos_pad = A.permute_plan_dual(ex, E)
    ex2, w2, counts2, dest2, src2, dest_pad2, pos_pad2 = A.route_plan_dual(
        logits, k, meth, bias, scale
    )
    torch.cuda.synchronize()
    assert torch.equal(ex, ex2) and torch.equal(bits(w), bits(w2))
    assert torch.equal(counts, counts2) and torch.equal(dest, dest2)
    assert torch.equal(src, src2)
    assert torch.equal(dest_pad, dest_pad2) and torch.equal(pos_pad, pos_pad2)


@needs_device
def test_layer_fused_routing_at_small_t_matches_prerouted_routing():
    """From logits at a small token count the runner routes and plans in one launch; the output equals the pre-routed modes'
    (the same ids and weights from ``route``) bit for bit, and the op given the fused plan equals the op planning itself."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import api_nvfp4 as A
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 5, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3)
    layer, W_pack = _layer(k, E, I), _weights(w1, w2)
    out = layer(_logits_pack(x, logits), W_pack)
    ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
    out2 = layer(_routed_pack(x, ids, wts), W_pack)
    assert torch.equal(bits(out), bits(out2))
    W = _weights_of(W_pack)
    ids3, wts3, *plan = A.route_plan_dual(logits, k, S.SOFTMAX_RENORM)
    out3 = A.fused_moe_nvfp4(x, ids3, wts3, W, plan=tuple(plan))
    out4 = A.fused_moe_nvfp4(x, ids, wts, W)
    assert torch.equal(bits(out3), bits(out)) and torch.equal(bits(out4), bits(out))
    with pytest.raises(ValueError):
        A.fused_moe_nvfp4(x, ids3, wts3, W, path="padded", plan=tuple(plan))


@needs_device
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens
    create: prefixes, permutations, sub-batches, one token alone, and a hot expert group around a probe token (groups spanning
    several 128-row scale-factor atoms)."""
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
@pytest.mark.parametrize("T", [300, 8], ids=["mid", "decode"])
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits(T):
    """Every autotuner tactic (the dispatcher's ranked records of the two GEMMs, both record spaces ranked together) computes the
    class; the list is the ranking's product in rank order, the tactic round-trips its record (the swapped-record flag included:
    at decode the ranking leads with the swapped body's records), and autotuning keeps the bits."""
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
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import TOP_K_FP4

    # GEMM1's list carries the structural rule's gather twin beside its box record (one more than the ranking's head)
    assert 1 <= len(tactics) <= TOP_K_FP4 * (TOP_K_FP4 + 1) and len(
        set(tactics)
    ) == len(tactics)
    import dataclasses

    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import GATHER_TILE_N

    r1, r2 = runner.impl._ranked(T, H)
    assert tactics == [_tactic_nvfp4(c1, c2) for c1 in r1 for c2 in r2]
    c1, c2 = _configs_from_tactic_nvfp4(tactics[0])
    assert _tactic_nvfp4(c1, c2) == tactics[0] and (c1, c2) == (r1[0], r2[0])
    # a pre-quantized input's tactic space: the box head plus the structural rule's gather twin of the narrow swapped tile
    r1g, _ = runner.impl._ranked(T, H, True)
    assert r1g[: len(r1)] == r1 and not any(c.gather_a for c in r1)
    assert [c for c in r1g if c.gather_a] == [
        dataclasses.replace(c, gather_a=True)
        for c in r1
        if c.swap_ab and c.tile_n == GATHER_TILE_N
    ]
    g1 = dataclasses.replace(r1[0], gather_a=True)
    assert _configs_from_tactic_nvfp4(_tactic_nvfp4(g1, c2)) == (g1, c2)
    assert len(tactics[0]) == 16
    if token_tile_default(rows_per_expert(T, k, E)) is not None:
        assert c1.swap_ab and c2.swap_ab, (T, c1.describe(), c2.describe())
    for t in tactics:
        out = runner.forward(runner.pack_inputs(pack, W), tactic=t)
        assert torch.equal(bits(out), ref), t
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)


@needs_device
def test_row_mapped_intermediate_quantization_equals_the_quantizer_and_the_retile():
    """The dense path's intermediate quantization: the class-local row-mapped launch (the quantizer's kernel with the scale
    factors and row scales stored at the padded positions) gives the quantizer's codes, scale bytes and row scales exactly as
    the quantizer plus the re-tiling move does, for row counts across the kernel's buckets and padded maps with gaps; the op
    gives the same bits either way."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.batch_invariant.quantization.nvfp4_schedule import (
        padded_sf_cols,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant import api_nvfp4 as A
    from flashinfer.fused_moe.batch_invariant.permute_nvfp4 import (
        get_compiled_scatter_sf,
    )

    g = torch.Generator().manual_seed(77)
    for M, K in ((1, 256), (5, 768), (130, 1024), (257, 1536), (1000, 2048)):
        x = (
            (
                torch.randn(M, K, generator=g)
                * torch.exp2(torch.randint(-8, 8, (M, 1), generator=g).float())
            )
            .to(torch.bfloat16)
            .cuda()
        )
        # a padded map: groups of random sizes, each starting at a 128-row atom (the dense path's pos_pad)
        sizes, pos = [], []
        left = M
        while left:
            c = int(torch.randint(1, min(left, 300) + 1, (1,), generator=g))
            sizes.append(c)
            left -= c
        off = 0
        for c in sizes:
            pos += list(range(off, off + c))
            off += -(-c // 128) * 128
        M_pad = off + 256
        row_map = torch.tensor(pos, dtype=torch.int32, device="cuda")
        x4a = torch.empty(M, K // 2, dtype=torch.uint8, device="cuda")
        sfa = torch.empty(sf_numel(M_pad, K), dtype=torch.uint8, device="cuda")
        ptsa = torch.empty(M_pad, dtype=torch.float32, device="cuda")
        A.quantize_rows_mapped(x, A.TE_GS_INV, row_map, (x4a, sfa, ptsa))
        x4b, sfb_dense, ptsb_dense = A.quantize_activations(x, A.TE_GS_INV)
        sfb = torch.empty_like(sfa)
        ptsb = torch.empty_like(ptsa)
        get_compiled_scatter_sf(1)(
            sfb_dense, ptsb_dense, row_map, sfb, ptsb, M, padded_sf_cols(K)
        )
        assert torch.equal(x4a, x4b), (M, K)
        assert torch.equal(bits(ptsa[row_map.long()]), bits(ptsb[row_map.long()])), (
            M,
            K,
        )
        # the scale bytes of the mapped rows: every scale of a mapped row equals the re-tiled one
        ua = unswizzle_sf_128x4(sfa, M_pad, K)[row_map.long()]
        ub = unswizzle_sf_128x4(sfb, M_pad, K)[row_map.long()]
        assert torch.equal(ua, ub), (M, K)
    # the op: both spellings give the bits
    T, k, H, I, E = 133, 4, 512, 256, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=78)
    W = _weights_of(_weights(w1, w2))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    outs = {}
    saved = A.INTERMEDIATE_RETILE
    try:
        for spelling in A.RETILES:
            A.INTERMEDIATE_RETILE = spelling
            outs[spelling] = A.fused_moe_nvfp4(x, ids, weights, W).cpu()
    finally:
        A.INTERMEDIATE_RETILE = saved
    assert torch.equal(bits(outs["mapped"]), bits(outs["scatter"]))


@needs_device
def test_fused_input_quantize_scatter_equals_the_quantizer_and_the_scatter():
    """The dense path's input: the class-local launch that quantizes a bf16 token once and stores its codes at the k dense
    rows and its scales at the k padded positions gives the quantizer's bytes exactly as the quantizer plus the fp4 scatter
    does, over row counts across the kernel's buckets and every k of the contract; the op gives the same bits either way."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.batch_invariant.quantization.nvfp4_schedule import (
        padded_sf_cols,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant import api_nvfp4 as A
    from flashinfer.fused_moe.batch_invariant.permute_nvfp4 import (
        get_compiled_scatter_fp4,
    )

    g = torch.Generator().manual_seed(79)
    for T, K, k, E in (
        (1, 256, 8, 16),
        (7, 768, 1, 4),
        (130, 1024, 6, 32),
        (257, 1536, 16, 64),
        (600, 2048, 2, 8),
    ):
        x = (
            (
                torch.randn(T, K, generator=g)
                * torch.exp2(torch.randint(-8, 8, (T, 1), generator=g).float())
            )
            .to(torch.bfloat16)
            .cuda()
        )
        ids = (
            torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
            .to(torch.int32)
            .cuda()
        )
        counts, dest, src, dest_pad, pos_pad = A.permute_plan_dual(ids, E)
        Tk = T * k
        Tk_a, Tk_sf = Tk + A.DENSE_TAIL, A.padded_rows_bound(T, k, E) + A.DENSE_TAIL
        outs = []
        for spelling in ("fused", "scatter"):
            x_perm4 = torch.zeros(Tk_a, K // 2, dtype=torch.uint8, device="cuda")
            x_perm_sf = torch.zeros(
                sf_numel(Tk_sf, K), dtype=torch.uint8, device="cuda"
            )
            x_perm_pts = torch.zeros(Tk_sf, dtype=torch.float32, device="cuda")
            if spelling == "fused":
                A.quantize_scatter_rows(
                    x,
                    A.TE_GS_INV,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    k,
                    (x_perm4, x_perm_sf, x_perm_pts),
                )
            else:
                x4, x_sf, x_pts = A.quantize_activations(x, A.TE_GS_INV)
                get_compiled_scatter_fp4(k)(
                    x4,
                    x_sf,
                    x_pts,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    x_perm4,
                    x_perm_sf,
                    x_perm_pts,
                    Tk,
                    K // 2,
                    padded_sf_cols(K),
                )
            outs.append(
                (
                    x_perm4[:Tk].clone(),
                    unswizzle_sf_128x4(x_perm_sf, Tk_sf, K)[
                        dest_pad.reshape(-1).long()
                    ].clone(),
                    x_perm_pts[dest_pad.reshape(-1).long()].clone(),
                )
            )
        (c0, s0, p0), (c1, s1, p1) = outs
        assert torch.equal(c0, c1), ("codes", T, K, k)
        assert torch.equal(s0, s1), ("scales", T, K, k)
        assert torch.equal(bits(p0), bits(p1)), ("pts", T, K, k)
        # the scatter's multi-warp placements (warps_per_row): the same bytes
        dp = dest_pad.reshape(-1).long()
        for wpr in (2, 4, 8):
            x_perm4 = torch.zeros(Tk_a, K // 2, dtype=torch.uint8, device="cuda")
            x_perm_sf = torch.zeros(
                sf_numel(Tk_sf, K), dtype=torch.uint8, device="cuda"
            )
            x_perm_pts = torch.zeros(Tk_sf, dtype=torch.float32, device="cuda")
            get_compiled_scatter_fp4(k, wpr)(
                x4,
                x_sf,
                x_pts,
                dest.reshape(-1),
                dest_pad.reshape(-1),
                x_perm4,
                x_perm_sf,
                x_perm_pts,
                Tk,
                K // 2,
                padded_sf_cols(K),
            )
            assert torch.equal(x_perm4[:Tk], c1), (wpr, T, K, k)
            assert torch.equal(unswizzle_sf_128x4(x_perm_sf, Tk_sf, K)[dp], s1), (
                wpr,
                T,
                K,
                k,
            )
            assert torch.equal(bits(x_perm_pts[dp]), bits(p1)), (wpr, T, K, k)
    T, k, H, I, E = 133, 4, 512, 256, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=80)
    W = _weights_of(_weights(w1, w2))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    res = {}
    saved = A.INPUT_SCATTER
    try:
        for spelling in A.INPUT_SCATTERS:
            A.INPUT_SCATTER = spelling
            res[spelling] = A.fused_moe_nvfp4(x, ids, weights, W).cpu()
    finally:
        A.INPUT_SCATTER = saved
    assert torch.equal(bits(res["fused"]), bits(res["scatter"]))


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


@needs_device
@pytest.mark.parametrize("body", ["padded", "dense", "swapped"])
@pytest.mark.parametrize("act", [_SWIGLU, ReLU2()], ids=["swiglu", "relu2"])
def test_poisoned_intermediates_do_not_reach_the_bits(body, act):
    """Unspecified bytes and timing never reach the output: the op's intermediates come from ``torch.empty`` (the dense
    packing's tail rows, GEMM1's output, GEMM2's output, the output itself); with the caching allocator's blocks filled with
    NaN patterns before every call and a bandwidth-bound kernel running on a side stream of the same device during the call
    (the contention that exposed a producer race: a K tail zero-filled by generic stores that the cp.async arrive did not
    order before the MMA), the padded body, the dense-packing body and the swapped decode body each give the bits of a clean
    run, ``FLASHINFER_BI_STRESS_ITERS`` times (default 20), at a shape whose GEMM2 K is below the K = 128 class's k-tile (the
    zero-filled tail); the clean run is the class composed from its pieces."""
    import os

    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import fused_moe_nvfp4
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

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
    ref = bits(fused_moe_nvfp4(x, ids, weights, W, **kw).cpu())
    side = torch.cuda.Stream()
    junk = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(int(os.environ.get("FLASHINFER_BI_STRESS_ITERS", "20"))):
        for byte in (0x7F, 0xFF, 0x00):
            _poison_allocator(byte)
            with torch.cuda.stream(side):
                for _ in range(3):
                    junk.add_(1)
            out = fused_moe_nvfp4(x, ids, weights, W, **kw)
            assert torch.equal(bits(out.cpu()), ref), (i, hex(byte))
    torch.cuda.synchronize()
    comp = _reference_ffn(
        _quantized(x),
        ids,
        weights,
        W,
        chain,
        expert_classes(H, I, _family(), chain.gated),
    )
    assert torch.equal(ref, bits(comp.cpu()))


@needs_device
def test_instruction_classes_on_the_part():
    """The part's default instruction class is the dense rule's (the sm_107 K = 128 step on SM107, the K = 64 step elsewhere)
    and the op follows it; on SM107 the K = 64 class stays reachable by name, computes a different fp32 fold of the same
    operands, and its op equals its own composition."""
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        F32_CHAIN,
        fused_moe_nvfp4,
        grouped_gemm_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    T, k, H, I, E = 96, 2, 1024, 512, 8
    # per-element exponent spreads inside every row and weight: the scaled products differ in magnitude, so the fp32 folds of
    # the two steps are not exact and separate (operands of one scale give exactly representable sums in either order)
    g = torch.Generator().manual_seed(8)
    x = (
        (
            torch.randn(T, H, generator=g)
            * torch.exp2(torch.randint(-10, 10, (T, H), generator=g).float())
        )
        .to(torch.bfloat16)
        .cuda()
    )
    logits = (torch.randn(T, E, generator=g) * 2).float().cuda()
    w1 = (
        (
            torch.randn(E, 2 * I, H, generator=g)
            * 0.05
            * torch.exp2(torch.randint(-6, 7, (E, 2 * I, H), generator=g).float())
        )
        .to(torch.bfloat16)
        .cuda()
    )
    w2 = (
        (
            torch.randn(E, H, I, generator=g)
            * 0.05
            * torch.exp2(torch.randint(-6, 7, (E, H, I), generator=g).float())
        )
        .to(torch.bfloat16)
        .cuda()
    )
    W = _weights_of(_weights(w1, w2))
    family = _family()
    fmts = expert_classes(H, I, family)
    assert fmts == (
        ("nvfp4_k128", "nvfp4_k128") if family == "sm_107" else ("nvfp4", "nvfp4")
    )
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    out, im = fused_moe_nvfp4(x, ids, weights, W, return_intermediates=True)
    assert im["in_fmts"] == fmts
    if family != "sm_107":
        return
    out64 = fused_moe_nvfp4(x, ids, weights, W, in_fmts=("nvfp4", "nvfp4"))
    ref64 = _reference_ffn(
        _quantized(x, check=False),
        ids,
        weights,
        W,
        activation_of(_SWIGLU),
        ("nvfp4", "nvfp4"),
    )
    assert torch.equal(bits(out64.cpu()), bits(ref64))
    accs = [
        grouped_gemm_nvfp4(
            im["x_perm4"],
            im["x_perm_sf"],
            im["x_perm_pts"],
            W.w1,
            W.w1_sf,
            W.w1_alpha,
            im["counts"],
            im["config1"],
            F32_CHAIN,
            f,
            dense_a=True,
        ).cpu()
        for f in ("nvfp4_k128", "nvfp4")
    ]
    assert not torch.equal(bits(accs[0]), bits(accs[1]))  # two folds: two classes


@needs_device
def test_routing_methods_from_logits_match_the_cpu_model():
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(10)) * 0.1)
        .float()
        .cuda()
    )
    for method in (
        RoutingMethodType.Default,
        RoutingMethodType.SigmoidRenorm,
        RoutingMethodType.DeepSeekV3,
    ):
        rm = S.METHOD_OF_FLASHINFER[method]
        kwargs = {"routed_scaling_factor": 1.75} if rm.scaled else {}
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=method, **kwargs),
                quant=_NVFP4,
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantFp4Config(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )
        W = _weights(w1, w2)
        b = bias if rm.bias else None
        out = layer(_logits_pack(x, logits, b), W)
        ex, w = R.route_cpu(
            logits.cpu(),
            k,
            rm,
            b.cpu() if b is not None else None,
            1.75 if rm.scaled else None,
        )
        assert torch.equal(
            bits(out), bits(layer(_routed_pack(x, ex.cuda(), w.cuda()), W))
        ), method


@needs_device
def test_plausibility_against_the_fp64_pipeline_and_flashinfers_reference():
    """An observation, not a promise: the class's output is within the fp32-vs-fp64 accumulation difference of the fp64
    evaluation of the same quantized pipeline (the dequantized operands, the spec's rounding points), and passes FlashInfer's
    own MoE accuracy check (``tests.moe.utils.check_accuracy``) against FlashInfer's torch reference
    (``compute_reference_moe_fp4``: the fp32 MoE on the quantized-and-dequantized operands, the intermediate requantized per
    token by FlashInfer's quantizer reference) -- the same reference and tolerance discipline FlashInfer's fp4 MoE tests use.
    Nothing here compares against another backend's kernels."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        dequantize,
        quantize_reference,
    )
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import TE_GS_INV
    from tests.moe.utils import (
        check_accuracy,
        compute_reference_moe_fp4,
        quant_dequant_fp4_reference,
    )

    T, k, H, I, E = 64, 4, 512, 256, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11, adversarial=False)
    layer = _layer(k, E, I)
    W_pack = _weights(w1, w2)
    out = layer(_logits_pack(x, logits), W_pack).float().cpu()
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    W = _weights_of(W_pack)
    # the fp64 pipeline on the dequantized operands
    x4, xsf, xpts = quantize_reference(x, TE_GS_INV, per_token=True)
    xd = dequantize(x4, xsf, 1.0, T, H, xpts)
    ref64 = torch.zeros(T, H, dtype=torch.float64)
    for e in range(E):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w1d = (
            dequantize(W.w1[e].cpu(), W.w1_sf[e].cpu(), 1.0, 2 * I, H)
            * W.w1_alpha[e].item()
        )
        w2d = (
            dequantize(W.w2[e].cpu(), W.w2_sf[e].cpu(), 1.0, H, I)
            * W.w2_alpha[e].item()
        )
        acc = (xd[tok] @ w1d.t()).float()
        u = acc.view(-1, I // 8, 2, 8)[:, :, 0, :].reshape(-1, I)
        g = acc.view(-1, I // 8, 2, 8)[:, :, 1, :].reshape(-1, I)
        a = S.glu_elem(R.TorchF32, g, u, "silu", "bf16")
        a4, asf, apts = quantize_reference(a, TE_GS_INV, per_token=True)
        ad = dequantize(a4, asf, 1.0, tok.numel(), I, apts)
        y = (ad @ w2d.t()).float().to(torch.bfloat16).double()
        ref64.index_add_(0, tok, weights[tok, slot].double().view(-1, 1) * y)
    err = ((out.double() - ref64).abs().max() / ref64.abs().max()).item()
    assert err < 2e-2, err
    # FlashInfer's torch reference on the quantized-and-dequantized operands: the experts through FlashInfer's per-tensor
    # quant-dequant at this backend's global scales (1 / alpha), the activations through its per-token quant-dequant, the
    # intermediate requantized per token inside the reference; checked with FlashInfer's accuracy rule
    gs1 = (1.0 / W.w1_alpha.float()).cuda()
    gs2 = (1.0 / W.w2_alpha.float()).cuda()
    w1_deq = torch.stack(
        [quant_dequant_fp4_reference(w1[e], gs1[e : e + 1]) for e in range(E)]
    )
    w2_deq = torch.stack(
        [quant_dequant_fp4_reference(w2[e], gs2[e : e + 1]) for e in range(E)]
    )
    x_deq = xd.float().cuda()
    ref_fi = compute_reference_moe_fp4(
        x_deq,
        w1_deq,
        w2_deq,
        ids.cuda(),
        weights.cuda(),
        T,
        E,
        k,
        H,
        I,
        fc2_input_scale=torch.tensor([TE_GS_INV], device="cuda"),
        use_per_token_activation=True,
    )
    ok, pct, atol = check_accuracy(out.cuda(), ref_fi)
    assert ok, (pct, atol)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the class raises the mode's error at build or at the first call (top_k /
    num_experts outside the bounds, an intermediate size that is not a multiple of 32, fused shared experts, an expert-parallel
    shard, no finalize, the wrong quantization pair, weights of the other chain, scales with bf16 rows, packed rows without
    scales, group-limited routing from logits, an unimplemented routing method from logits); a zero-token call returns an
    empty output."""
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
        assert what in str(excinfo.value) or "moe.nvfp4" in str(excinfo.value), what
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
            "shared",
        ),
        (
            dict(
                experts=ExpertConfig(intermediate_size=I, local_num_experts=E // 2),
            ),
            "expert-parallel",
        ),
        (
            dict(
                experts=ExpertConfig(intermediate_size=I, local_expert_offset=E // 2),
            ),
            "expert-parallel",
        ),
        (
            dict(
                experts=ExpertConfig(
                    intermediate_size=I,
                    local_expert_offset=E // 2,
                    local_num_experts=E // 2,
                )
            ),
            "parallel",
        ),
        (dict(finalize=MoEFinalizeConfig(do_finalize=False)), "do_finalize"),
        (dict(quant=QuantConfig()), "BF16"),
    ):
        with pytest.raises(errors) as excinfo:
            MoELayer(
                MoEConfig(
                    routing=kw.get("routing", routing),
                    quant=kw.get("quant", _NVFP4),
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantFp4Config(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.nvfp4" in str(excinfo.value), what
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=12)
    W = _weights(w1, w2)
    with pytest.raises(errors):
        layer(
            _logits_pack(x.to(torch.float16), logits), W
        )  # fp16 rows: no runner accepts the pack
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(
            _logits_pack(x, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2())
        )  # the ungated chain's weights
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(
            _logits_pack(
                x, logits, sf=torch.zeros(8, H // 16, dtype=torch.uint8, device="cuda")
            ),
            W,
        )  # scales with bf16 rows
    x4, sf, pts = _quantized(x, check=False)
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(_logits_pack(x4, logits), W)  # packed rows without their scales
    with pytest.raises(BatchInvariantUnsupportedError):
        BatchInvariantFp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size=I,
            quant=QuantConfig(),
        )
    # the grouped selection (DeepSeek-V3's n_group / topk_group) is a routing term the bf16 model serves: the layer runs from
    # logits, deterministically; with pre-routed ids the group structure never enters the computation, and the output equals
    # the ungrouped configuration's on the same ids and weights
    grouped = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.DeepSeekV3,
                n_group=2,
                topk_group=1,
            ),
            quant=_NVFP4,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp4Config(),)),
        ),
        device=torch.device("cuda"),
    )
    g_bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(12)) * 0.1)
        .float()
        .cuda()
    )  # the DeepSeek-V3 method selects on sigmoid(l) + bias: the bias is part of its term
    out_g = grouped(_logits_pack(x, logits, bias=g_bias), W)
    assert out_g.shape == (x.shape[0], H) and torch.equal(
        bits(out_g), bits(grouped(_logits_pack(x, logits, bias=g_bias), W))
    )
    g_ids, g_wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(grouped(_routed_pack(x, g_ids.cuda(), g_wts.cuda()), W)),
        bits(layer(_routed_pack(x, g_ids.cuda(), g_wts.cuda()), W)),
    )
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=_NVFP4,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp4Config(),)),
        ),
        device=torch.device("cuda"),
    )
    # Llama4 routing (TopK with sigmoid weights) is a routing term the bf16 model serves: the layer runs from logits,
    # deterministically
    out_l = llama(_logits_pack(x, logits), W)
    assert out_l.shape == (x.shape[0], H) and torch.equal(
        bits(out_l), bits(llama(_logits_pack(x, logits), W))
    )
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(_routed_pack(x, ids, torch.ones(8, 1, device="cuda")), W).shape == (
        8,
        H,
    )
    out = layer(_logits_pack(x[:0], logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


# ----------------------------------------------------------------------------- the pre-quantized pack of the TRTLLM backends


@needs_device
def test_linear_scale_unit_pack_equals_swizzled_pack_with_ones():
    """The per-tensor activation pack of the TRTLLM backends (fp4 codes, the e4m3 block scales row-major ``[T, H/16]``, no
    per-token scale) through the op == the same bytes given as the class's 128x4 buffer with unit per-token scales: the fp4
    scatter's two placements (the scale atom assembled from the linear row vs copied from the swizzled buffer; the unit
    per-token scale written vs read) move the same bytes to the same positions."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import fused_moe_nvfp4

    T, k, H, I, E = 100, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=24)
    W = _weights_of(_weights(w1, w2))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    x4, sf, _ = _quantized(x)
    sf_lin = unswizzle_sf_128x4(sf, T, H).cuda().contiguous()
    assert sf_lin.shape == (T, H // 16)
    ref = fused_moe_nvfp4((x4, sf, torch.ones(T, device="cuda")), ids, wts, W)
    assert torch.equal(
        bits(fused_moe_nvfp4((x4, sf_lin, None), ids, wts, W)), bits(ref)
    )
    # the scatter's placements: the box form (the default form of a pre-quantized input may gather instead)
    out, im = fused_moe_nvfp4(
        (x4, sf_lin, None), ids, wts, W, return_intermediates=True, gather=False
    )
    assert torch.equal(bits(out), bits(ref))
    pos = im["dest_pad"].reshape(-1).long()
    assert torch.equal(im["x_perm_pts"][pos].cpu(), torch.ones(T * k))
    assert torch.equal(
        unswizzle_sf_128x4(im["x_perm_sf"], im["Tk_pad"], H).cpu()[pos.cpu()],
        sf_lin.cpu()[torch.arange(T * k) // k],
    )


@needs_device
def test_every_record_computes_the_class_at_long_k_on_every_instruction_class_of_the_part():
    """At a K long enough for every CTA to cycle its whole operand ring (16 k-tiles per output tile), every swapped record and
    the deepest-ring padded records compute the class on every instruction class the part runs (on SM107 the K = 64 class
    by name as well): the op equals the per-expert dense oracle. A ring placed past the sm_100 instructions' shared-memory
    address window (the plan's ``UMMA_SMEM_ADDR_MAX``) would read other bytes here."""
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import fused_moe_nvfp4
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        grouped_knobs_blockscaled,
        swapped_knobs_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    T, k, H, I, E = 16, 2, 4096, 2048, 8
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=77)
    W = _weights_of(_weights(w1, w2))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    chain = activation_of(_SWIGLU)
    classes = {expert_classes(H, I, _family())[0]}
    if _family() == "sm_107":
        classes.add("nvfp4")
    deep = [
        c
        for c in grouped_knobs_blockscaled()
        if c.raster_along == "m"
        and c.cluster_shape_mn == (1, 1)
        and c.tile_n in (64, 256)
    ]
    for fmt in sorted(classes):
        fmts = (fmt, fmt)
        ref = bits(
            _reference_ffn(_quantized(x, check=False), ids, wts, W, chain, fmts).cpu()
        )
        for c in swapped_knobs_blockscaled() + deep:
            out = fused_moe_nvfp4(x, ids, wts, W, in_fmts=fmts, config1=c, config2=c)
            assert torch.equal(bits(out.cpu()), ref), (fmt, c.describe())


# ----------------------------------------------------------------------------- the static-intermediate class


def test_static_class_fingerprints_pin():
    """The static class's fingerprint (``moe.nvfp4.<act>.static``): its own components, pinned, different from the per-token
    class's at every activation; the config field is validated."""
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import (
        INTERMEDIATES,
        MoENvfp4Spec,
        spec_fingerprint_nvfp4,
    )

    assert INTERMEDIATES == ("per_token", "static")
    assert (
        spec_fingerprint_nvfp4(2048, 768, 8, 128, intermediate="static")
        == "moe-a07bb9a65670a187"
    )
    assert (
        spec_fingerprint_nvfp4(
            2048, 1024, 6, 128, act="relu2", gated=False, intermediate="static"
        )
        == "moe-96f1a997c9250826"
    )
    for act, gated in (("silu", True), ("gelu_tanh", True), ("relu2", False)):
        a, b = (
            spec_fingerprint_nvfp4(
                2048, 768, 8, 128, act=act, gated=gated, intermediate=i
            )
            for i in INTERMEDIATES
        )
        assert a != b
    spec = MoENvfp4Spec(512, 256, 4, 16, act="silu", intermediate="static")
    assert spec.name.endswith(".static") and spec.gs2(0) is not None
    assert BatchInvariantFp4Config(intermediate="static").intermediate == "static"
    with pytest.raises(ValueError):
        BatchInvariantFp4Config(intermediate="rowwise")


def _static_ladder(T, k, H, I, E, act, seed, gs_pow2=True):
    """The static class's ladder at one problem: GEMM1's codes and scale bytes == the exact CPU model of the static chain on the
    f32 oracle accumulators (``w4a4_static_gemm1_cpu``), the other body (swapped / padded) gives the same codes and scale bytes,
    GEMM2's chain == the CPU model with ``inv2[e]``, the combine == CPU, the padded packing == the dense packing, two runs equal,
    and the class differs from the per-token class (another quantization of the intermediate)."""
    from flashinfer.batch_invariant.quantization.nvfp4_reference import (
        unswizzle_sf_128x4,
    )
    from flashinfer.batch_invariant.quantization.nvfp4_schedule import sf_numel
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_nvfp4 as RN
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        DENSE_TAIL,
        F32_CHAIN,
        MoEWeightsNVFP4,
        fused_moe_nvfp4,
        grouped_gemm_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import W4A4Chain
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        rank_grouped_nvfp4,
        rows_per_expert,
        swapped_knobs_blockscaled,
    )
    from flashinfer.fused_moe.batch_invariant.spec import ACTS
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

    gated = act in ACTS
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed, gated=gated)
    g = torch.Generator(device="cuda").manual_seed(seed)
    gs2 = (
        torch.exp2(torch.randint(-2, 3, (E,), device="cuda", generator=g).float())
        if gs_pow2
        else torch.rand(E, device="cuda", generator=g) * 3 + 0.5
    )
    W = MoEWeightsNVFP4.prepare(
        w1, w2, gated=gated, act=act, intermediate="static", intermediate_scales=gs2
    )
    assert W.intermediate == "static" and torch.equal(W.gs2, gs2)
    in1, in2 = expert_classes(H, I, _family(), gated)
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    out, im = fused_moe_nvfp4(x, ids, wts, W, return_intermediates=True)
    assert torch.equal(bits(out), bits(fused_moe_nvfp4(x, ids, wts, W)))
    assert im["a"] is None and im["path"] == "dense"
    Tk, Tk_pad = T * k, im["Tk_pad"]
    counts = im["counts"].cpu().tolist()
    off = [0]
    for c in counts:
        off.append(off[-1] + c)
    pos_pad = im["pos_pad"].cpu().long()
    # the f32 oracle accumulators of GEMM1 (a padded-body record on the dense packing)
    oracle1 = (
        im["config1"]
        if not im["config1"].swap_ab
        else rank_grouped_nvfp4(
            rows_per_expert(T, k, E), w1.shape[1], H, E, F32_CHAIN, in1, swap=False
        )[0]
    )
    acc1 = grouped_gemm_nvfp4(
        im["x_perm4"],
        im["x_perm_sf"],
        im["x_perm_pts"],
        W.w1,
        W.w1_sf,
        W.w1_alpha,
        im["counts"],
        oracle1,
        F32_CHAIN,
        in1,
        dense_a=True,
    ).cpu()
    alpha1, gsr, alpha2, inv2r = (torch.zeros(Tk) for _ in range(4))
    for e in range(E):
        alpha1[off[e] : off[e + 1]] = W.w1_alpha.cpu()[e]
        gsr[off[e] : off[e + 1]] = W.gs2.cpu()[e]
        alpha2[off[e] : off[e + 1]] = W.w2_alpha.cpu()[e]
        inv2r[off[e] : off[e + 1]] = W.inv2.cpu()[e]
    pts_dense = im["x_perm_pts"].cpu()[pos_pad]
    ref_codes, ref_sf = RN.w4a4_static_gemm1_cpu(
        acc1, alpha1, pts_dense, gsr, act, (), gated
    )
    assert torch.equal(im["a4"].cpu(), ref_codes), "GEMM1 codes"
    assert torch.equal(
        unswizzle_sf_128x4(im["a_sf_pad"].cpu(), Tk_pad, I)[pos_pad], ref_sf
    ), "GEMM1 scale bytes"
    # the other body gives the same codes and scale bytes (the swapped body's warp-level block vs the padded body's in-register
    # block): a swapped record needs the token tile's rows past the live rows of its A operand
    chain1 = W4A4Chain("glu" if gated else "act", act, "nvfp4", (), "w4a4")
    if im["config1"].swap_ab:
        other = rank_grouped_nvfp4(
            rows_per_expert(T, k, E), w1.shape[1], H, E, chain1, in1, swap=False
        )[0]
        a_in = im["x_perm4"]
    else:
        # the swapped body reads a token box past the live rows and writes the same row count: the op's tail on both buffers
        other = swapped_knobs_blockscaled()[0]
        a_in = torch.cat(
            [
                im["x_perm4"],
                torch.zeros(DENSE_TAIL, H // 2, dtype=torch.uint8, device="cuda"),
            ]
        )
    codes_b = torch.full(
        (a_in.shape[0], I // 2), 0xAA, dtype=torch.uint8, device="cuda"
    )
    sf_b = torch.full((sf_numel(Tk_pad, I),), 0xAA, dtype=torch.uint8, device="cuda")
    grouped_gemm_nvfp4(
        a_in,
        im["x_perm_sf"],
        im["x_perm_pts"],
        W.w1,
        W.w1_sf,
        W.w1_alpha,
        im["counts"],
        other,
        chain1,
        in1,
        out=codes_b,
        dense_a=True,
        out_sf=sf_b,
        aux=W.gs2,
    )
    assert torch.equal(codes_b[:Tk].cpu(), ref_codes) and torch.equal(
        unswizzle_sf_128x4(sf_b.cpu(), Tk_pad, I)[pos_pad], ref_sf
    ), ("the other body", other.describe())
    # GEMM2 == CPU on the codes and scale bytes the epilogue wrote; the combine == CPU
    oracle2 = (
        im["config2"]
        if not im["config2"].swap_ab
        else rank_grouped_nvfp4(
            rows_per_expert(T, k, E), H, I, E, F32_CHAIN, in2, swap=False
        )[0]
    )
    acc2 = grouped_gemm_nvfp4(
        im["a4"],
        im["a_sf_pad"],
        im["a_pts_pad"],
        W.w2,
        W.w2_sf,
        W.w2_alpha,
        im["counts"],
        oracle2,
        F32_CHAIN,
        in2,
        dense_a=True,
    ).cpu()
    assert torch.equal(
        bits(im["y"].cpu()), bits(RN.w4a4_static_gemm2_cpu(acc2, alpha2, inv2r))
    ), "GEMM2 chain"
    assert torch.equal(bits(out.cpu()), bits(R.combine_cpu(im["y"], im["dest"], wts)))
    assert torch.equal(
        bits(fused_moe_nvfp4(x, ids, wts, W, path="padded")), bits(out)
    ), "the padded packing"
    Wp = MoEWeightsNVFP4.prepare(w1, w2, gated=gated, act=act)
    assert not torch.equal(bits(fused_moe_nvfp4(x, ids, wts, Wp)), bits(out))
    return out


@needs_device
@pytest.mark.parametrize(
    "T,k,H,I,E,act,seed",
    [
        (3, 4, 512, 256, 16, "silu", 31),
        (37, 2, 256, 128, 8, "silu", 32),
        (300, 6, 512, 256, 64, "gelu_tanh", 33),
        (64, 8, 256, 64, 256, "silu", 34),
        (200, 6, 2048, 1024, 128, "relu2", 35),
        (1000, 8, 2048, 768, 128, "silu", 36),
    ],
)
def test_static_class_ladder(T, k, H, I, E, act, seed):
    _static_ladder(T, k, H, I, E, act, seed)


@needs_device
def test_static_class_takes_any_positive_scale_and_unit_default():
    """The static scales are per-expert inputs (any positive value); the default is one, the open backends' unified-layer
    value."""
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        MoEWeightsNVFP4,
        fused_moe_nvfp4,
    )

    _static_ladder(37, 4, 512, 256, 16, "silu", 41, gs_pow2=False)
    T, k, H, I, E = 40, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=42)
    W = MoEWeightsNVFP4.prepare(w1, w2, intermediate="static")
    W1 = MoEWeightsNVFP4.prepare(
        w1, w2, intermediate="static", intermediate_scales=torch.ones(E, device="cuda")
    )
    assert torch.equal(W.gs2, torch.ones(E, device="cuda")) and torch.equal(
        W.inv2, torch.ones(E, device="cuda")
    )
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    assert torch.equal(
        bits(fused_moe_nvfp4(x, ids, wts, W)), bits(fused_moe_nvfp4(x, ids, wts, W1))
    )


@needs_device
def test_static_class_through_the_layer_batch_invariance_graph_replay_and_pack_check():
    """The static class selected through the layer (``BatchInvariantFp4Config(intermediate="static")`` with weights prepared
    for it): the rows of a sub-batch equal the rows of the full batch, a permutation permutes, graph replay reproduces eager,
    the autotuner's tactics keep the bits, the view carries the class and its scales, and a pack prepared for the other class
    is refused."""
    from flashinfer.autotuner import autotune

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=43)
    gs2 = torch.exp2(
        torch.randint(-2, 3, (E,), generator=torch.Generator().manual_seed(43)).float()
    ).cuda()
    cfg = MoEConfig(
        routing=RoutingConfig(
            num_experts=E, top_k=k, method=RoutingMethodType.Renormalize
        ),
        quant=_NVFP4,
        experts=ExpertConfig(intermediate_size=I),
        activation=_SWIGLU,
        backend=BackendOptions((BatchInvariantFp4Config(intermediate="static"),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=4096),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )
    layer = MoELayer(cfg, device=torch.device("cuda"))
    view = BatchInvariantFp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        activation=_SWIGLU,
        intermediate="static",
        intermediate_scales=gs2,
    )
    assert view["intermediate"] == "static" and torch.equal(
        view["intermediate_scale"], gs2
    )
    pack = MoEWeightPack({})
    pack.prepare_for(KEY, view)
    full = layer(_logits_pack(x, logits), pack)
    assert torch.equal(bits(full), bits(layer(_logits_pack(x, logits), pack)))
    sub = layer(_logits_pack(x[:5].contiguous(), logits[:5].contiguous()), pack)
    assert torch.equal(bits(sub), bits(full[:5]))
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(44)).cuda()
    out_p = layer(_logits_pack(x[perm].contiguous(), logits[perm].contiguous()), pack)
    assert torch.equal(bits(out_p), bits(full[perm]))
    with autotune(True):
        assert torch.equal(bits(layer(_logits_pack(x, logits), pack)), bits(full))
    assert torch.equal(bits(layer(_logits_pack(x, logits), pack)), bits(full))
    s = torch.cuda.Stream()
    act = _logits_pack(x, logits)
    with torch.cuda.stream(s):
        layer(act, pack)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = layer(act, pack)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(full))
    # the per-token class's output differs (another quantization of the intermediate), and its pack is refused by this layer
    per_token = _weights(w1, w2)
    assert not torch.equal(
        bits(_layer(k, E, I)(_logits_pack(x, logits), per_token)), bits(full)
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        layer(_logits_pack(x, logits), per_token)
    assert "intermediate" in str(excinfo.value)


# ----------------------------------------------------------------------------- dropped slots


def _guarded(n: int, dtype=torch.uint8, guard: int = 1 << 20, byte: int = 0xAB):
    """A tensor of ``n`` elements inside a sentinel-filled allocation with ``guard`` bytes before and after it, and the whole
    allocation (to check that no byte outside the live rows, and nothing outside the buffer, was written)."""
    es = torch.tensor([], dtype=dtype).element_size()
    big = torch.empty(guard + n * es + guard, dtype=torch.uint8, device="cuda").fill_(
        byte
    )
    return big[guard : guard + n * es].view(dtype), big


def _guards_untouched(
    big: torch.Tensor, guard: int = 1 << 20, byte: int = 0xAB
) -> bool:
    return bool((big[:guard] == byte).all()) and bool((big[-guard:] == byte).all())


@needs_device
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``: a framework's dropped slot) is dropped by the op as
    by the bf16 one: the plan does not count it (``dest -1``, ``dest_pad -1``, ``pos_pad -1`` past the live rows), the scatter
    and the two fused quantizers write nothing for it (nothing outside the live rows, nothing outside the buffers: sentinel
    guards around every destination), the GEMMs never touch it, and the combine folds the remaining slots in slot order with
    ``+0`` in its place, so the output equals the class's composition with those slots at a valid id and weight ``+0``; a token
    whose slots are all dropped gives a zero row; the other tokens keep their bits; the combine oracle over the op's own
    intermediates gives the kernel's bits. Both plan forms, both packings; with every slot dropped the GEMMs launch without a
    tile on the dense and on the swapped records."""
    from flashinfer.batch_invariant.quantization.nvfp4_schedule import (
        padded_sf_cols,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N, permute_plan
    from flashinfer.fused_moe.batch_invariant.api_nvfp4 import (
        DENSE_TAIL,
        TE_GS_INV,
        _gs_inv,
        _part,
        fused_moe_nvfp4,
        padded_rows_bound,
        permute_plan_dual,
        quantize_activations,
        quantize_rows_mapped,
        quantize_scatter_rows,
    )
    from flashinfer.fused_moe.batch_invariant.permute import warps_per_row
    from flashinfer.fused_moe.batch_invariant.permute_nvfp4 import (
        get_compiled_scatter_fp4,
        get_compiled_scatter_sf,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_nvfp4 import (
        default_configs_nvfp4,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.fused_moe.batch_invariant.spec_nvfp4 import expert_classes

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
        assert int(counts.sum()) == n_valid
        assert torch.equal(dest.cpu().view(-1) < 0, dropped.view(-1))
        assert torch.all(src[n_valid:] == -1) and torch.all(src[:n_valid] >= 0)
        counts_d, dest_d, src_d, dest_pad, pos_pad = permute_plan_dual(
            ids_bad.cuda(), E
        )
        assert torch.equal(counts_d, counts) and torch.equal(dest_d, dest)
        assert torch.equal(dest_pad.cpu().view(-1) < 0, dropped.view(-1))
        assert torch.all(pos_pad[n_valid:] == -1) and torch.all(pos_pad[:n_valid] >= 0)
        # the scatter and the fused quantizers on sentinel-guarded destinations: nothing outside the live rows / positions
        x4, sf, pts = _quantized(x)
        Tk, Tk_pad = T * k, padded_rows_bound(T, k, E)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        pcols_h, pcols_i = padded_sf_cols(H), padded_sf_cols(I)
        live_pad = torch.zeros(Tk_sf, dtype=torch.bool, device="cuda")
        live_pad[pos_pad[:n_valid].long()] = True
        wpr = warps_per_row(Tk, H // 32, sm_count)
        for fused in (False, True):
            x_perm4, big4 = _guarded(Tk_a * (H // 2))
            x_perm_sf, bigsf = _guarded(sf_numel(Tk_sf, H))
            x_perm_pts, bigpts = _guarded(Tk_sf, torch.float32)
            x_perm4 = x_perm4.view(Tk_a, H // 2)
            if fused:
                quantize_scatter_rows(
                    x,
                    _gs_inv(TE_GS_INV, x.device),
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    k,
                    (x_perm4, x_perm_sf, x_perm_pts),
                )
            else:
                get_compiled_scatter_fp4(k, wpr, "128x4", False)(
                    x4,
                    sf,
                    pts,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    x_perm4,
                    x_perm_sf,
                    x_perm_pts,
                    Tk,
                    H // 2,
                    pcols_h,
                )
            torch.cuda.synchronize()
            assert (
                _guards_untouched(big4)
                and _guards_untouched(bigsf)
                and _guards_untouched(bigpts)
            ), fused
            assert bool((x_perm4[n_valid:] == 0xAB).all()), (
                fused
            )  # the dropped slots' rows and the tail: never written
            tok = (src[:n_valid] // k).long()
            assert torch.equal(x_perm4[:n_valid], x4[tok]), fused
            assert bool(
                (x_perm_pts.view(torch.uint8).view(Tk_sf, 4)[~live_pad] == 0xAB).all()
            ), fused
            assert torch.equal(x_perm_pts[pos_pad[:n_valid].long()], pts[tok]), fused
        # the intermediate's two spellings (the row-mapped quantizer; the quantizer + the scale re-tiling)
        a = (
            (torch.randn(Tk, I, generator=torch.Generator().manual_seed(T + 1)) * 0.5)
            .to(torch.bfloat16)
            .cuda()
        )
        gsi = _gs_inv(TE_GS_INV, a.device)
        a4_m, big_a4 = _guarded(Tk * (I // 2))
        a_sf_m, big_asf = _guarded(sf_numel(Tk_sf, I))
        a_pts_m, big_apts = _guarded(Tk_sf, torch.float32)
        quantize_rows_mapped(a, gsi, pos_pad, (a4_m.view(Tk, I // 2), a_sf_m, a_pts_m))
        torch.cuda.synchronize()
        assert (
            _guards_untouched(big_a4)
            and _guards_untouched(big_asf)
            and _guards_untouched(big_apts)
        )
        assert bool((a_pts_m.view(torch.uint8).view(Tk_sf, 4)[~live_pad] == 0xAB).all())
        a4_q, a_sf_q, a_pts_q = quantize_activations(a, gsi)
        assert torch.equal(
            a4_m.view(Tk, I // 2), a4_q
        )  # the codes stay in place for every row, mapped or not
        a_sf_s, big_s = _guarded(sf_numel(Tk_sf, I))
        a_pts_s, big_p = _guarded(Tk_sf, torch.float32)
        get_compiled_scatter_sf(1)(
            a_sf_q, a_pts_q, pos_pad, a_sf_s, a_pts_s, Tk, pcols_i
        )
        torch.cuda.synchronize()
        assert _guards_untouched(big_s) and _guards_untouched(big_p)
        assert torch.equal(a_pts_s[live_pad], a_pts_m[live_pad]) and bool(
            (a_pts_s.view(torch.uint8).view(Tk_sf, 4)[~live_pad] == 0xAB).all()
        )
        assert torch.equal(
            a_sf_s, a_sf_m
        )  # the same bytes at the padded positions, the same sentinel elsewhere
        # the op: the dropped slots are +0 in the combine; the composition with weight 0 at a valid id gives the bits
        out = layer(_routed_pack(x, ids_bad.cuda(), wts.cuda()), W_pack)
        ids_ref = torch.where(dropped, torch.zeros_like(ids), ids)
        wts_ref = torch.where(dropped, torch.zeros_like(wts), wts)
        ref = _reference_ffn(
            _quantized(x),
            ids_ref.cuda(),
            wts_ref,
            W,
            chain,
            expert_classes(H, I, family),
        )
        assert torch.equal(bits(out.cpu()), bits(ref)), T
        assert torch.all(out[rows[0]] == 0)
        for path in ("dense", "padded"):
            out_api, im = fused_moe_nvfp4(
                x, ids_bad.cuda(), wts.cuda(), W, path=path, return_intermediates=True
            )
            assert torch.equal(bits(out_api), bits(out)), path
            assert torch.equal(
                bits(out_api.cpu()), bits(R.combine_cpu(im["y"], im["dest"], wts))
            ), path
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack(x, ids.cuda(), wts.cuda()), W_pack)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))
    # every slot dropped: zero counts, the grouped GEMMs launch without a tile, the zero row of every token
    T, k, E = 37, 4, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=44)
    W = _weights_of(_weights(w1, w2))
    ids_none = torch.full((T, k), -1, dtype=torch.int32, device="cuda")
    wts_any = torch.rand(T, k, device="cuda")
    out, im = fused_moe_nvfp4(x, ids_none, wts_any, W, return_intermediates=True)
    torch.cuda.synchronize()
    assert (
        torch.all(out == 0)
        and int(im["counts"].sum()) == 0
        and torch.all(im["dest"] == -1)
    )
    for swap in (False, True):
        c1, c2 = default_configs_nvfp4(
            T,
            k,
            H,
            I,
            E,
            expert_classes(H, I, family),
            W.chain1,
            W.chain2,
            sm_count,
            family,
            swap=swap,
        )
        assert c1.swap_ab == swap and c2.swap_ab == swap
        out2 = fused_moe_nvfp4(x, ids_none, wts_any, W, config1=c1, config2=c2)
        torch.cuda.synchronize()
        assert torch.all(out2 == 0), swap
    assert torch.all(fused_moe_nvfp4(x, ids_none, wts_any, W, path="padded") == 0)
