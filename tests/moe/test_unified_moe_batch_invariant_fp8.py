# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor batch-invariant MoE backend of the unified MoE layer (``BatchInvariantFp8PerTensorConfig``): the W8A8
classes ``moe.e4m3.<act>``, one per typed activation of the layer and its scalars, on the surface of the TRTLLM-Gen fp8
per-tensor backend (e4m3 activations quantized with the static ``hidden_states_scale_global``, the weights prepared with
the two calibrated multipliers, every routing input mode).

The class has exact CPU models of everything but the tensor-core step (the routing, the static input quantization, the
fused e4m3 quantization of the activation chain on the alpha-scaled accumulators, GEMM2's scaled chain, the slot-order
combine), so the tests are bit comparisons: the layer's output against the class composed from its own pieces (the
routing on the CPU, the grouped GEMM1 as the raw fp32 fold, the CPU chain with the exact requantization, the dense
batch-invariant fp8 GEMM per expert for GEMM2, the CPU combine) for several activations, run-to-run determinism including
CUDA-graph replay, batch / permutation / sub-batch invariance, agreement of every tactic the autotuner may pick, the three
routing input modes giving the same bits, the error contract, and a plausibility observation against FlashInfer's torch
reference of the fp8 per-tensor MoE. The device-free tests pin the registration, the fingerprint dimensions, the
agreement of the vectorised and the exact CPU chains on adversarial accumulators and scales, the input quantization, the
weight preparation's multipliers and the dispatcher's membership in the enumerated space at the nominal parts.
"""

from __future__ import annotations

import os

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    BatchInvariantFp8PerTensorConfig,
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
from flashinfer.fused_moe.batch_invariant.runner import (
    BatchInvariantMoeRunner,
    BatchInvariantMoeRunnerBase,
)
from flashinfer.fused_moe.batch_invariant.runner_fp8 import (
    MOE_FP8_ARCHS,
    BatchInvariantFp8PerTensorRunner,
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
#: The activations the suites cover (the bf16 backend's list): every ``ActivationConfig`` class of the layer with its defaults,
#: and the parameterised ones once more with scalars that move every branch; the clamps bite in dequantized units.
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
M_INTER = 24.0  # the intermediate multiplier: large enough that the adversarial problems saturate some e4m3 outputs


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()]
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
    """bf16 activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated), and
    the static input multiplier ``448 / amax(x)``."""
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
    amax = x.float().abs().amax().clamp_min(1e-30)
    m_x = (448.0 / amax).reshape(1)
    return x.cuda(), logits.cuda(), w1.cuda(), w2.cuda(), m_x.cuda()


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
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    m_x: torch.Tensor,
    act=_SWIGLU,
    m_inter: float = M_INTER,
) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=m_x,
        intermediate_scale_global=m_inter,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(BatchInvariantFp8PerTensorRunner.backend_key, view)
    return pack


def _xq(x: torch.Tensor, m_x: torch.Tensor) -> torch.Tensor:
    xq, scale = BatchInvariantFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=m_x
    )
    assert scale is None and xq.dtype == torch.float8_e4m3fn
    return xq


def _logits_pack(
    xq: torch.Tensor, logits: torch.Tensor, bias=None
) -> MoEActivationPack:
    return MoEActivationPack(
        xq,
        None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(
    xq: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, packed: bool = False
) -> MoEActivationPack:
    if packed:
        return MoEActivationPack(
            xq,
            None,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        xq,
        None,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The config is a known backend of the layer, mapped to its runner, absent from the default candidates, and declares the
    fp8 per-tensor quantization pair, the three routing modes and every activation class of the layer (the bf16 runner's)."""
    assert BatchInvariantFp8PerTensorConfig in ALL_BACKEND_CONFIGS
    assert (
        _BACKEND_RUNNERS[BatchInvariantFp8PerTensorConfig]
        is BatchInvariantFp8PerTensorRunner
    )
    assert not any(
        isinstance(c, BatchInvariantFp8PerTensorConfig)
        for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantFp8PerTensorConfig.supported(a)
    ] == [100, 103, 107]
    # both class runners are the shared base's (the pack / plan / launch flow with the format hooks)
    assert issubclass(BatchInvariantFp8PerTensorRunner, BatchInvariantMoeRunnerBase)
    assert issubclass(BatchInvariantMoeRunner, BatchInvariantMoeRunnerBase)
    assert not issubclass(BatchInvariantFp8PerTensorRunner, BatchInvariantMoeRunner)
    assert (
        BatchInvariantFp8PerTensorRunner.backend_key
        == "batch_invariant_fp8_per_tensor"
        != BatchInvariantMoeRunner.backend_key
    )
    assert BatchInvariantFp8PerTensorRunner.supported_quant_variants == (
        (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),
    )
    assert set(BatchInvariantFp8PerTensorRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert (
        set(BatchInvariantFp8PerTensorRunner.supported_activation_classes)
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
        repr(BatchInvariantFp8PerTensorConfig()) == "BatchInvariantFp8PerTensorConfig()"
    )
    assert (
        "fused_moe_fp8" in bi.CLASSES and bi.class_name("fused_moe_fp8") == "moe.e4m3"
    )


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_fp8", ...)`` takes the geometry, the routing method (with DeepSeekV3's group shape) and the
    activation; it differs from the bf16 class's at the same dimensions; every listed activation is its own class; the
    routing method is a dimension; the scales' values are not."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_fp8 as S8

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_fp8", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert fp != bi.fingerprint("fused_moe", **dims)
    assert (
        bi.fingerprint("fused_moe_fp8", method="Renormalize", act="swiglu", **dims)
        == fp
    )
    assert bi.fingerprint("fused_moe_fp8", act=SwiGLU(), **dims) == fp
    assert (
        bi.fingerprint("fused_moe_fp8", method=RoutingMethodType.Sigmoid, **dims) != fp
    )
    assert bi.fingerprint("fused_moe_fp8", H=2048, I=768, top_k=8, num_experts=64) != fp
    fps = {bi.fingerprint("fused_moe_fp8", act=a, **dims) for a in ACTIVATIONS}
    assert len(fps) == len(ACTIVATIONS) and fp in fps
    assert bi.fingerprint("fused_moe_fp8", act="relu2", **dims) == bi.fingerprint(
        "fused_moe_fp8", act=ReLU2(), **dims
    )
    assert (
        bi.fingerprint("fused_moe_fp8", method=RoutingMethodType.Llama4, **dims) != fp
    )  # the Llama4 (top-k sigmoid) routing term is a class dimension of its own
    g84 = bi.fingerprint(
        "fused_moe_fp8",
        method=RoutingMethodType.DeepSeekV3,
        n_group=8,
        topk_group=4,
        **dims,
    )  # DeepSeekV3's groups are parameters of its routing term: the group shape is a dimension
    assert g84 != bi.fingerprint(
        "fused_moe_fp8", method=RoutingMethodType.DeepSeekV3, **dims
    )
    assert g84 == bi.fingerprint(
        "fused_moe_fp8", method="sigmoid_renorm_bias_scaled_g8x4", **dims
    )
    assert g84 != bi.fingerprint(
        "fused_moe_fp8",
        method=RoutingMethodType.DeepSeekV3,
        n_group=4,
        topk_group=2,
        **dims,
    )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_fp8",
            method=RoutingMethodType.TopK,
            n_group=8,
            topk_group=4,
            **dims,
        )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe_fp8", method=RoutingMethodType.MiniMax2, **dims
        )  # a method without a term in the mode
    with pytest.raises(TypeError):
        bi.fingerprint("fused_moe_fp8", m_x=1.0, **dims)
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        assert bi.fingerprint(
            "fused_moe_fp8", act=cfg, **dims
        ) == S8.spec_fingerprint_fp8(
            2048, 768, 8, 128, S.SOFTMAX_RENORM, a.act, a.params, a.gated
        )
    s = S8.Fp8MoESpec(64, 32, 2, 8)
    assert (
        s.name == "moe.e4m3.swiglu"
        and S8.Fp8MoESpec(64, 32, 2, 8, act="relu2", gated=False).name
        == "moe.e4m3.relu2"
    )


def test_expert_and_slot_bounds_at_every_entry_point():
    """``1 <= num_experts <= MAX_E`` and ``1 <= top_k <= min(num_experts, MAX_TOP_K)`` (the plan's and the grouped GEMMs'
    tables, the route kernels' selection, the gather and combine kernels' fold) at the fingerprint and at every entry point of
    the class that takes them, before any kernel: ``fused_moe_fp8``, ``grouped_gemm_fp8``, ``MoEWeightsE4M3.prepare`` /
    ``BatchInvariantFp8PerTensorConfig.prepare_weights``; the bounds themselves are taken."""
    from flashinfer.fused_moe.batch_invariant import api_fp8
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.blocks.scheduler import MAX_E
    from flashinfer.fused_moe.batch_invariant.schedule import grouped_knobs

    assert S.MAX_TOP_K == 16
    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    for kw in (
        dict(top_k=17),
        dict(top_k=0),
        dict(top_k=9, num_experts=8),
        dict(num_experts=MAX_E + 1),
        dict(num_experts=0),
    ):
        with pytest.raises(ValueError):
            bi.fingerprint("fused_moe_fp8", **{**dims, **kw})
    assert bi.fingerprint(
        "fused_moe_fp8", H=2048, I=768, top_k=16, num_experts=MAX_E
    ).startswith("moe-")
    e4m3, bf16 = torch.float8_e4m3fn, torch.bfloat16
    E = MAX_E + 1
    w1, w2 = torch.zeros(E, 32, 16, dtype=bf16), torch.zeros(E, 16, 16, dtype=bf16)
    errors = (ValueError, bi.BatchInvariantUnsupportedError)
    with pytest.raises(errors):
        api_fp8.MoEWeightsE4M3.prepare(w1, w2, 1.0, 1.0)
    with pytest.raises(errors):
        BatchInvariantFp8PerTensorConfig.prepare_weights(
            w1, w2, hidden_states_scale_global=1.0, intermediate_scale_global=1.0
        )
    xq = torch.zeros(2, 16, dtype=e4m3)
    one = torch.ones(1)
    with pytest.raises(ValueError):
        api_fp8.grouped_gemm_fp8(
            xq,
            torch.zeros(E, 16, 16, dtype=e4m3),
            torch.ones(E),
            one,
            one,
            torch.zeros(E, dtype=torch.int32),
            grouped_knobs()[0],
        )
    w4 = api_fp8.MoEWeightsE4M3.prepare(
        torch.zeros(4, 32, 16, dtype=bf16), torch.zeros(4, 16, 16, dtype=bf16), 1.0, 1.0
    )
    assert w4.num_experts == 4
    for kk in (17, 0):
        with pytest.raises(ValueError):
            api_fp8.fused_moe_fp8(
                xq, torch.zeros(2, kk, dtype=torch.int32), torch.ones(2, kk), w4
            )


def test_ranking_is_memoised_per_token_count():
    """``rank_grouped_fp8`` (the shared dispatcher's enumeration, acceptance and pricing of both fp8 record spaces at the
    chain's format) is memoised on its scalar arguments (the chain a frozen dataclass) in the one memo of ``schedule``: a
    repeated call is a cache hit returning an equal list, and ``default_configs_fp8`` (the ``fused_moe_fp8`` path without
    records) hits twice per call; the lists are copies."""
    from flashinfer.fused_moe.batch_invariant.schedule import _rank_grouped
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import (
        default_configs_fp8,
        gemm1_chain,
        rank_grouped_fp8,
    )

    _rank_grouped.cache_clear()
    args = (7, 1536, 2048, 128, gemm1_chain("silu"), 212, "sm_107", 3)
    a = rank_grouped_fp8(*args, n_active=56)
    misses = _rank_grouped.cache_info().misses
    b = rank_grouped_fp8(*args, n_active=56)
    assert b == a and b is not a and len(a) == 3
    assert _rank_grouped.cache_info().hits == 1
    assert _rank_grouped.cache_info().misses == misses
    a.clear()
    assert rank_grouped_fp8(*args, n_active=56) == b
    c = default_configs_fp8(1, 8, 2048, 768, 128, 212, "sm_107", True)
    hits = _rank_grouped.cache_info().hits
    assert default_configs_fp8(1, 8, 2048, 768, 128, 212, "sm_107", True) == c
    assert _rank_grouped.cache_info().hits == hits + 2


def test_swapped_box_form_compiles_under_the_chain_trigger_rule_only(monkeypatch):
    """The swapped body at an fp8 chain reads the plan's group counts under the chain's trigger rule
    (``pdl.BOX_FORM_TRIGGERS``: the dependents launch after the predecessor's own wait, the body's own wait before the tables
    is the margin) and its builder asserts the shipped placement is one of them, as it does for the bf16 chains; the gather
    form belongs to the bf16 classes."""
    from flashinfer.fused_moe.batch_invariant import pdl
    from flashinfer.fused_moe.batch_invariant.blocks.swap import SwappedGroupedGemm
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import (
        gemm1_chain,
        swapped_knobs_fp8,
    )

    cfg = swapped_knobs_fp8()[0]
    assert pdl.TRIGGER in pdl.BOX_FORM_TRIGGERS
    SwappedGroupedGemm(cfg, gemm1_chain("silu"))
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(cfg, gemm1_chain("silu"), gather=True)
    monkeypatch.setattr(pdl, "TRIGGER", "early")
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(cfg, gemm1_chain("silu"))


def test_one_body_per_form_serves_both_formats():
    """The grouped and the swapped bodies are one source each, parametrised by the chain's operand format and epilogue block:
    the fp8 chains declare their format and scale operands, the bf16 chains none; the swapped record space follows the
    format's instruction (``kind::f8f6f4`` takes 16 or 32 tokens, ``kind::f16`` 8 as well); the compiled objects keep their
    names and one JIT module (``batch_invariant_moe``) holds both formats' kernels, with no format-specific body in its
    source closure; the bodies take a chain of either format and the gather form stays the bf16 classes'."""
    from flashinfer.batch_invariant import jit
    from flashinfer.fused_moe.batch_invariant.blocks import compile as C
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import (
        GEMM2_CHAIN,
        Fp8Chain,
        GroupedChain,
        GroupedEpilogueBlock,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.kernel import (
        K_BLOCKS_OF,
        DenseOperands,
        GroupedBlockGemm,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.swap import (
        SwappedGroupedGemm,
        is_swapped_record,
        swapped_knobs,
    )
    from flashinfer.fused_moe.batch_invariant.schedule import rank_grouped
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import gemm1_chain
    from flashinfer.gemm.batch_invariant.generator import Knobs

    bf16, fp8 = GroupedChain.legacy("swiglu"), gemm1_chain("silu")
    for ch in (bf16, GroupedChain("plain", "silu", "f32"), fp8, GEMM2_CHAIN):
        assert isinstance(ch, GroupedEpilogueBlock)
    assert (bf16.in_fmt, bf16.scale_operands, bf16.scaled, bf16.paired_cvt) == (
        "bf16",
        False,
        False,
        False,
    )
    assert (fp8.in_fmt, fp8.scale_operands, fp8.scaled, fp8.paired_cvt, fp8.pair) == (
        "e4m3",
        True,
        True,
        True,
        True,
    )
    assert (GEMM2_CHAIN.scaled, GEMM2_CHAIN.paired_cvt) == (True, False)
    raw = Fp8Chain("plain", "silu", "f32", (), False)
    assert raw.scale_operands and not raw.scaled
    # the record spaces of the two formats
    assert [(c.tile_m, c.tile_n) for c in swapped_knobs("bf16")] == [
        (128, 8),
        (128, 16),
        (128, 32),
        (64, 16),
        (64, 32),
    ]
    assert [(c.tile_m, c.tile_n) for c in swapped_knobs("e4m3")] == [
        (128, 16),
        (128, 32),
        (64, 16),
        (64, 32),
    ]
    sw8 = swapped_knobs("bf16")[0]
    assert is_swapped_record(sw8, "bf16") and not is_swapped_record(sw8, "e4m3")
    assert not C.can_implement_swapped(sw8, 256, 512, 8, fp8)
    assert C.can_implement_swapped(sw8, 256, 512, 8, "plain")
    # the bodies take a chain of either format; the operand block follows the chain
    dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
    for ch in (bf16, fp8):
        body = GroupedBlockGemm(dense, ch)
        assert body.in_fmt == ch.in_fmt and body.operands == DenseOperands(ch.in_fmt)
        sw = swapped_knobs(ch.in_fmt)[0]
        assert SwappedGroupedGemm(sw, ch).operands.in_fmt == ch.in_fmt
    assert K_BLOCKS_OF["swapped"]["bf16"] == K_BLOCKS_OF["swapped"]["e4m3"]
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(swapped_knobs("e4m3")[0], fp8, gather=True)
    # the object names carry the format; one JIT module, no format-specific body in its closure
    assert C.kernel_name(dense, bf16) == "grouped_bf16_swiglu_bf16_" + dense.describe()
    assert C.kernel_name(dense, fp8).startswith("grouped_e4m3_glu_silu_w8a8_e4m3_")
    assert C.swapped_kernel_name(swapped_knobs("e4m3")[0], GEMM2_CHAIN).startswith(
        "swapped_e4m3_plain_silu_w8a8_bf16_"
    )
    assert C.MODULE == "batch_invariant_moe" and "batch_invariant_moe_fp8" not in jit.MODULE_ROOTS
    srcs = jit.relative_sources("batch_invariant_moe")
    assert not any("fp8_" in os.path.basename(s) or "swap_bf16" in s for s in srcs), srcs
    for body in ("compile", "kernel", "swap", "epilogue"):
        assert f"fused_moe/batch_invariant/blocks/{body}.py" in srcs
    # the shared ranking at the chain's format: the fp8 chain's list is a subset of its own space
    ranked = rank_grouped(1, 1536, 2048, 128, fp8, 212, "sm_107", 3, False, 8)
    assert ranked and all(C.can_implement(c, 1536, 2048, 128, fp8) for c in ranked)
    with pytest.raises(AssertionError):
        rank_grouped(1, 1536, 2048, 128, fp8, 212, "sm_107", 3, False, 8, in_fmt="bf16")


def _adversarial_accumulators(limits, n_random: int, seed: int) -> torch.Tensor:
    f32 = torch.float32
    inf = torch.tensor(float("inf"), dtype=f32)
    edges = []
    for lim in limits:
        t = torch.tensor(float(lim), dtype=f32)
        up, down = torch.nextafter(t, inf), torch.nextafter(t, -inf)
        edges += [t, -t, up, down, -up, -down]
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
            -1e-3,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            8.0,
            -8.0,
            87.0,
            -87.0,
            88.0,
            -88.0,
            448.0,
            -448.0,
            464.0,
            -464.0,
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
    rnd = torch.randn(n_random, generator=g) * torch.exp2(
        torch.randint(-12, 12, (n_random,), generator=g).float()
    )
    parts = [special, rnd.to(f32)]
    if edges:
        parts.insert(1, torch.stack(edges))
    return torch.cat(parts)


@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_vectorised_and_exact_chains_agree_on_adversarial_accumulators_and_scales(act):
    """The vectorised GEMM1 chain (``reference_fp8.w8a8_gemm1_cpu`` / ``w8a8_gemm1_act_cpu`` through ``TorchF32``) and the exact one
    (``spec_fp8.glu_quant_e4m3_elem`` / ``act_quant_e4m3_elem`` through ``ExactF32``) give the same e4m3 values on adversarial fp32
    accumulators (zeros of both signs, subnormals, the exp clamp's edges, the e4m3 maximum and beyond, the configuration's
    clamp limits and their neighbours, spread magnitudes) with spread per-row weight scales and intermediate multipliers;
    the saturating conversion is exercised. The GEMM2 chain likewise."""
    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_fp8 as S8

    a = S.activation_of(act)
    limits = [p for p in a.params if p is not None]
    g = _adversarial_accumulators(limits, 160, seed=20)
    u = torch.roll(_adversarial_accumulators(limits, 160, seed=21), 5)
    n = g.numel()
    gen = torch.Generator().manual_seed(22)
    s_w = torch.exp2((torch.rand(n, generator=gen) * 2 - 1) * 8).float()
    s_x, m_inter = fp32.to_f32(0.0073), fp32.to_f32(37.0)
    alpha = [fp32.mul(s_x, float(v)) for v in s_w.tolist()]
    if a.gated:
        vec = R8.w8a8_gemm1_cpu(
            g.view(-1, 1), u.view(-1, 1), s_x, s_w, m_inter, a.act, a.params
        ).view(-1)
        exact = [
            S8.glu_quant_e4m3_elem(R.ExactF32, gi, ui, al, m_inter, a.act, a.params)
            for gi, ui, al in zip(g.tolist(), u.tolist(), alpha, strict=True)
        ]
    else:
        vec = R8.w8a8_gemm1_act_cpu(
            g.view(-1, 1), s_x, s_w, m_inter, a.act, a.params
        ).view(-1)
        exact = [
            S8.act_quant_e4m3_elem(R.ExactF32, gi, al, m_inter, a.act, a.params)
            for gi, al in zip(g.tolist(), alpha, strict=True)
        ]
    ex = torch.tensor(exact, dtype=torch.float32)
    bad = (bits(vec) != bits(ex)).nonzero().flatten()[:8].tolist()
    assert not bad, [
        (i, g[i].item(), u[i].item(), vec[i].item(), ex[i].item()) for i in bad
    ]
    assert (vec.abs() <= 448).all()
    if (
        not limits
    ):  # a clamped chain's activation stays below the e4m3 maximum at this multiplier
        assert (vec.abs() == 448).any(), "the saturating conversion is exercised"
    assert torch.equal(bits(R8.e4m3_bytes(vec).float()), bits(vec)), (
        "the e4m3 values convert to bytes exactly"
    )
    # GEMM2: the scaled accumulator to bf16
    vec2 = R8.w8a8_gemm2_cpu(g.view(-1, 1), m_inter, s_w).view(-1)
    ex2 = torch.tensor(
        [
            S8.scaled_out_elem(R.ExactF32, gi, fp32.mul(m_inter, float(v)), "bf16")
            for gi, v in zip(g.tolist(), s_w.tolist(), strict=True)
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    assert torch.equal(bits(vec2), bits(ex2))


def test_input_quantization_and_dequantization_scales_are_the_declared_chains():
    """``prepare_activations`` is ``cvt.rn.satfinite.e4m3(mul(x, m_x))`` bit for bit on adversarial values (saturation, subnormals,
    signed zeros, ties); the dequantization scales are the correctly rounded reciprocals of the multipliers; the weight
    multipliers are FlashInfer's ``448 / amax`` and the quantized experts are the chain's bytes."""
    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3,
        weight_multipliers,
    )
    from flashinfer.fused_moe.prepare import _quantize_fp8_per_expert

    x = _adversarial_accumulators([448.0, 1.0], 2000, seed=5).to(torch.bfloat16)
    x = x[: x.numel() // 16 * 16].view(-1, 16)
    for m in (0.007, 1.0, 3.0, 2.0**20):
        q, scale = BatchInvariantFp8PerTensorConfig.prepare_activations(
            x, hidden_states_scale_global=m
        )
        assert scale is None and q.dtype == torch.float8_e4m3fn and q.shape == x.shape
        ex = torch.tensor(
            [
                fp32.to_fp8(fp32.mul(float(v), fp32.to_f32(m)), "e4m3")
                for v in x.float().view(-1).tolist()
            ],
            dtype=torch.float32,
        )
        assert torch.equal(bits(q.float().view(-1)), bits(ex)), m
        inv = R8.dequant_scale(torch.tensor([m], dtype=torch.float32))
        assert inv.item() == fp32.div(1.0, fp32.to_f32(m))
    w1 = (torch.randn(4, 64, 32, generator=torch.Generator().manual_seed(3)) * 0.1).to(
        torch.bfloat16
    )
    w2 = (torch.randn(4, 32, 32, generator=torch.Generator().manual_seed(4)) * 0.1).to(
        torch.bfloat16
    )
    W = MoEWeightsE4M3.prepare(w1, w2, 2.0, 16.0)
    q_fi, m_fi = _quantize_fp8_per_expert(w2)
    assert torch.equal(bits(W.m_w2), bits(m_fi)) and torch.equal(
        bits(weight_multipliers(w2)), bits(m_fi)
    )
    assert torch.equal(bits(W.s_w2), bits(R8.dequant_scale(m_fi)))
    ex = torch.tensor(
        [
            fp32.to_fp8(fp32.mul(float(v), float(mm)), "e4m3")
            for e in range(4)
            for v in w2[e].float().view(-1).tolist()
            for mm in (m_fi[e].item(),)
        ],
        dtype=torch.float32,
    )
    assert torch.equal(bits(W.w2q.float().view(-1)), bits(ex))
    # observation: FlashInfer's torch cast of the same scaled values gives the same bytes (its conversion rounds to nearest even)
    assert torch.equal(W.w2q.view(torch.uint8), q_fi.view(torch.uint8))
    assert (
        W.gated
        and W.intermediate_size == 32
        and W.w1q.shape == (4, 64, 32)
        and W.m_inter.item() == 16.0
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        MoEWeightsE4M3.prepare(
            w1[:, :, :24].contiguous(), w2[:, :24].contiguous(), 2.0, 16.0
        )  # H = 24


@pytest.mark.parametrize(
    "family,sm_count",
    [("sm_107", 212), ("sm_107", 204), ("sm_100", 148), ("sm_103", 152)],
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for every bench geometry and token count lies in the enumerated grouped fp8 space of its GEMM, for
    the gated and the ungated classes, at the nominal parts; the ranked list is short, distinct and inside the space."""
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule import rows_per_expert
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import (
        all_valid_configs_fp8,
        all_valid_configs_fp8_both,
        all_valid_configs_swapped_fp8,
        default_configs_fp8,
        gemm1_chain,
        rank_grouped_fp8,
        swapped_knobs_fp8,
    )

    sw = swapped_knobs_fp8()
    assert [(c.tile_m, c.tile_n) for c in sw] == [
        (128, 16),
        (128, 32),
        (64, 16),
        (64, 32),
    ] and all(c.swap_ab and c.cluster_shape_mn == (1, 1) for c in sw)
    geoms = [
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (5120, 8192, 16, 1),
        (512, 256, 64, 6),
        (1024, 512, 64, 8),
    ]
    for H, I, E, k in geoms:
        s1 = set(all_valid_configs_fp8_both(2 * I, H, E, gemm1_chain("silu")))
        s1u = set(all_valid_configs_fp8_both(I, H, E, gemm1_chain("relu2", (), False)))
        s2 = set(all_valid_configs_fp8_both(H, I, E, GEMM2_CHAIN))
        assert s1 and s1u and s2
        # the two record spaces are disjoint (the swapped flag) and the swapped one has every record at these shapes
        assert not (set(all_valid_configs_fp8(H, I, E, GEMM2_CHAIN)) & set(sw))
        assert set(all_valid_configs_swapped_fp8(H, I, E, GEMM2_CHAIN)) == set(sw)
        for T in (1, 7, 64, 300, 800, 4096, 16384):
            c1, c2 = default_configs_fp8(
                T, k, H, I, E, sm_count, family, True, nominal=True
            )
            a1, a2 = default_configs_fp8(
                T, k, H, I, E, sm_count, family, True, nominal=True, swap=True
            )
            assert a1.swap_ab and a2.swap_ab and a1 in s1 and a2 in s2
            n1, n2 = default_configs_fp8(
                T, k, H, I, E, sm_count, family, True, nominal=True, swap=False
            )
            assert not n1.swap_ab and not n2.swap_ab and n1 in s1 and n2 in s2
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
                T, k, H, I, E, sm_count, family, False, nominal=True
            )
            assert c1u in s1u, (family, T, H, I, E, k, c1u.describe())
            ranked = rank_grouped_fp8(
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
                and all(c in s2 for c in ranked)
            )
    # the acceptance: 16-byte e4m3 rows of every operand
    assert (
        all_valid_configs_fp8(2 * 760, 2048, 128, gemm1_chain("silu")) == []
    )  # I % 16
    assert all_valid_configs_fp8(2048, 760, 128, GEMM2_CHAIN) == []  # K % 16


def test_unsupported_device_raises_before_any_kernel():
    """The runner on a CPU device fails its support check with the mode's error naming the class."""
    runner = BatchInvariantFp8PerTensorRunner(_config(8, 64, 256), torch.device("cpu"))
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        runner._check_support()
    assert "moe.e4m3" in str(excinfo.value) and "cpu" in str(excinfo.value)


# ----------------------------------------------------------------------------- GPU


def _reference_route(logits: torch.Tensor, k: int, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _reference_ffn(
    xq: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, view: dict, act
) -> torch.Tensor:
    """The class composed on the device from its own pieces: the class's permutation (the rows gathered by torch indexing),
    GEMM1 as the grouped fp8 kernel with the unscaled fp32 identity epilogue (the same fold; any record of the space gives its
    bits), the CPU chain of the activation with the exact e4m3 requantization on those accumulators and the view's scales,
    GEMM2 as the dense batch-invariant fp8 GEMM per expert (``mm_fp8_batch_invariant`` with unit scales, fp32 out: the raw
    fold at this weight shape, asserted unsplit), the CPU GEMM2 chain, the CPU slot-order combine. Bit for bit what the
    fused op must produce."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8 as R8
    from flashinfer.fused_moe.batch_invariant.api import permute_plan
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3,
        grouped_gemm_fp8,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import Fp8Chain
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import all_valid_configs_fp8
    from flashinfer.gemm.batch_invariant.api import mm_fp8_batch_invariant
    from flashinfer.gemm.batch_invariant.schedule import chunks_for

    W = MoEWeightsE4M3.from_view(view)
    T, H = xq.shape
    E, k = W.num_experts, ids.shape[1]
    I = W.intermediate_size
    assert chunks_for(H, I, "e4m3") == 1
    counts, dest, src = permute_plan(ids, E)
    n = T * k
    x_perm = xq[(src // k).long()].contiguous()
    raw = Fp8Chain("plain", "silu", "f32", (), False)
    acc = grouped_gemm_fp8(
        x_perm,
        W.w1q,
        W.s_w1,
        W.s_x,
        W.m_inter,
        counts,
        all_valid_configs_fp8(W.w1q.shape[1], H, E, raw)[0],
        raw,
    )
    off = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cpu().long().cumsum(0)])
    sw1 = torch.repeat_interleave(W.s_w1.cpu(), counts.cpu().long())
    sw2 = torch.repeat_interleave(W.s_w2.cpu(), counts.cpu().long())
    if act.gated:
        u = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
        g = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
        a_vals = R8.w8a8_gemm1_cpu(
            g.cpu(), u.cpu(), W.s_x.cpu(), sw1, W.m_inter.cpu(), act.act, act.params
        )
    else:
        a_vals = R8.w8a8_gemm1_act_cpu(
            acc.cpu(), W.s_x.cpu(), sw1, W.m_inter.cpu(), act.act, act.params
        )
    a = R8.e4m3_bytes(a_vals).to(xq.device)
    acc2 = torch.empty(n, H, dtype=torch.float32, device=xq.device)
    one = torch.ones(1, dtype=torch.float32, device=xq.device)
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            acc2[lo:hi] = mm_fp8_batch_invariant(
                a[lo:hi].contiguous(), W.w2q[e].t(), one, one, out_dtype=torch.float32
            )
    y_perm = R8.w8a8_gemm2_cpu(acc2.cpu(), W.s_inter.cpu(), sw2)
    y = y_perm[dest.view(-1).long().cpu()]
    ident = torch.arange(n, dtype=torch.int32).view(T, k)
    return R.combine_cpu(y, ident, weights.cpu()), dict(
        a=a, counts=counts, dest=dest, acc=acc
    )


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU, the fused e4m3 chain of the activation on the raw grouped
    accumulators, the dense batch-invariant fp8 GEMM per expert and the CPU combine: the oracle ladder of the class, bit for
    bit, for every activation; the fused op's intermediate e4m3 rows equal the CPU chain's bytes."""
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3,
        fused_moe_fp8,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2, m_x = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    xq = _xq(x, m_x)
    W = _weights(w1, w2, m_x, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(xq, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    view = W.get_view(BatchInvariantFp8PerTensorRunner.backend_key)
    ref, pieces = _reference_ffn(xq, ids.cuda(), weights.cuda(), view, chain)
    assert torch.equal(bits(out.cpu()), bits(ref))
    _, im = fused_moe_fp8(
        xq,
        ids.cuda(),
        weights.cuda(),
        MoEWeightsE4M3.from_view(view),
        return_intermediates=True,
    )
    assert torch.equal(im["a"].view(torch.uint8), pieces["a"].view(torch.uint8))
    if not [
        p for p in chain.params if p is not None
    ]:  # an unclamped chain: the adversarial problem saturates some intermediates
        assert (im["a"].float().abs() == 448).any()


def test_weight_multipliers_stay_finite_for_tiny_experts():
    """An expert whose largest magnitude is below ``448 / FLT_MAX`` would take an infinite ``448 / amax``; the multiplier is
    clamped to the largest finite one, so the quantized bytes stay finite (an all-zero expert keeps the unit multiplier)."""
    from flashinfer.fused_moe.batch_invariant.api_fp8 import weight_multipliers

    w = torch.zeros(3, 16, 32, dtype=torch.bfloat16)
    w[0, 0, 0] = 2.0**-120
    w[1, 3, 5] = 0.5
    m = weight_multipliers(w)
    assert torch.isfinite(m).all() and m[0] == torch.finfo(torch.float32).max
    assert m[1] == 448.0 / 0.5 and m[2] == 1.0
    q = (w.float() * m.view(-1, 1, 1)).to(torch.float8_e4m3fn).float()
    assert torch.isfinite(q).all()


@needs_device
def test_the_largest_expert_count_routes_and_runs():
    """1024 experts: the class's routing from the logits equals the CPU model and the layer equals the class composition."""
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 96, 8, 256, 128, 1024
    x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=9, adversarial=True)
    xq = _xq(x, m_x)
    ids, wts = route(logits, k)
    ref_ids, ref_w = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(ids.cpu(), ref_ids) and torch.equal(bits(wts.cpu()), bits(ref_w))
    W = _weights(w1, w2, m_x)
    out = _layer(k, E, I)(_logits_pack(xq, logits), W)
    ref, _ = _reference_ffn(
        xq,
        ids,
        wts,
        W.get_view(BatchInvariantFp8PerTensorRunner.backend_key),
        activation_of(_SWIGLU),
    )
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their inputs;
    repeated runs and a CUDA-graph replay reproduce them."""
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=2, adversarial=True)
    xq = _xq(x, m_x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2, m_x)
    out_l = layer(_logits_pack(xq, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack(xq, ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    w16 = weights.to(torch.bfloat16)
    out_u16 = layer(_routed_pack(xq, ids.cuda(), w16.cuda()), W)
    out_p = layer(_routed_pack(xq, ids.cuda(), w16.cuda(), packed=True), W)
    assert torch.equal(bits(out_u16), bits(out_p))
    assert torch.equal(
        bits(out_u16), bits(layer(_routed_pack(xq, ids.cuda(), w16.float().cuda()), W))
    )
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack(xq, logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack(xq, logits)
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
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens create."""
    T, k, H, I, E = 500, 8, 512, 256, 64
    x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=3, adversarial=True)
    xq = _xq(x, m_x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2, m_x)
    full = bits(layer(_logits_pack(xq, logits), W))
    for n in (1, 2, 7, 64, 200, 333):
        assert torch.equal(
            bits(layer(_logits_pack(xq[:n], logits[:n]), W)), full[:n]
        ), n
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(4)).cuda()
    assert torch.equal(
        bits(layer(_logits_pack(xq[perm].contiguous(), logits[perm].contiguous()), W)),
        full[perm],
    )
    for n in (1, 2, 7, 64, 200):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(n))[:n].cuda()
        assert torch.equal(
            bits(
                layer(_logits_pack(xq[idx].contiguous(), logits[idx].contiguous()), W)
            ),
            full[idx],
        ), n
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    base = bits(layer(_routed_pack(xq, ids, weights), W))
    g = torch.Generator().manual_seed(5)
    x_extra = _xq(
        (torch.randn(556, H, generator=g) * 0.5).to(torch.bfloat16).cuda(), m_x
    )
    w_extra = torch.rand(556, k, generator=g).cuda()
    for t in (0, T // 2, T - 1):
        hot = ids[t : t + 1].expand(556, k).contiguous()
        out = layer(
            _routed_pack(
                torch.cat([xq[t : t + 1], x_extra]),
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
                torch.cat([xq[t : t + 1], x_extra]),
                torch.cat([ids[t : t + 1], others]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("unrelated hot experts", t)


@needs_device
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits():
    """Every tactic the autotuner may pick (the dispatcher's ranked fp8 record pairs) gives the same bits; tuning the layer
    profiles them and the output before, during and after tuning is one output."""
    from flashinfer.autotuner import AutoTuner, autotune

    T, k, H, I, E = 300, 6, 1024, 512, 64
    x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=6, adversarial=True)
    xq = _xq(x, m_x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2, m_x)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantFp8PerTensorRunner)
    pack = _logits_pack(xq, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert 1 <= len(tactics) <= 9 and len(set(tactics)) == len(tactics)
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        ), t
    failed_before = (
        len(AutoTuner.get().stats.failed_tactics)
        if hasattr(AutoTuner.get(), "stats")
        else 0
    )
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    if hasattr(AutoTuner.get(), "stats"):
        assert len(AutoTuner.get().stats.failed_tactics) == failed_before
    # a decode-sized batch: the ranked tactics include the swapped records (the swapped body), every tactic gives the bits of
    # the class's CPU composition, and a tactic round-trips its swapped flag
    from flashinfer.fused_moe.batch_invariant.runner import (
        _configs_from_tactic,
        _tactic,
    )

    xd, ld, _, _, _ = _problem(3, k, H, I, E, seed=8, adversarial=True)
    xqd = _xq(xd, m_x)
    packd = _logits_pack(xqd, ld)
    ids, weights = _reference_route(ld, k, RoutingMethodType.Renormalize)
    view = W.get_view(BatchInvariantFp8PerTensorRunner.backend_key)
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    refd, _ = _reference_ffn(
        xqd, ids.cuda(), weights.cuda(), view, activation_of(_SWIGLU)
    )
    assert torch.equal(bits(layer(packd, W).cpu()), bits(refd))
    inputs = runner.pack_inputs(packd, W)
    tactics_d = runner.get_valid_tactics(inputs, None)
    assert any(_configs_from_tactic(t)[0].swap_ab for t in tactics_d), tactics_d
    for t in tactics_d:
        c1, c2 = _configs_from_tactic(t)
        assert _tactic(c1, c2) == t
        out = runner.forward(runner.pack_inputs(packd, W), tactic=t)
        assert torch.equal(bits(out.cpu()), bits(refd)), t


@needs_device
def test_swapped_records_equal_the_grouped_records():
    """The fp8 swapped body (the weights on the MMA's M side -- 128 or 64 rows per CTA -- and the group's e4m3 rows a TMA box
    of 16 / 32 tokens from the dense rows): at every swapped record its rows are a grouped record's rows bit for bit -- the raw
    fp32 fold, GEMM2's scaled bf16 chain, every gated and the ungated fused e4m3 chain -- with groups of 1 .. 70 rows (several
    token tiles), empty groups, E = 256, a weight width that is not a multiple of the weight tile, a partial last k-tile, and
    the dense buffer with and without the token tile's pad rows past the live ones (the last tiles' boxes then read the pad, or
    are zero-filled past the buffer: never stored)."""
    from flashinfer.fused_moe.batch_invariant.api_fp8 import grouped_gemm_fp8
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import (
        GEMM2_CHAIN,
        Fp8Chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8 import (
        gemm1_chain,
        swapped_knobs_fp8,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs

    g = torch.Generator().manual_seed(11)

    def rnd_e4m3(*shape, span=2.0):
        return (
            ((torch.rand(*shape, generator=g) * 2 - 1) * span)
            .to(torch.float8_e4m3fn)
            .cuda()
        )

    def bytes_of(t):
        return bits(
            t.view(torch.uint8).cpu() if t.dtype == torch.float8_e4m3fn else t.cpu()
        )

    dense = Knobs((64, 64), (1, 1), False, True, "m", 1)
    chains = [
        Fp8Chain("plain", "silu", "f32", (), False),
        GEMM2_CHAIN,
        gemm1_chain("silu"),
        gemm1_chain("gelu_tanh"),
        gemm1_chain("relu2", (), False),
    ]
    for counts, N, K in [
        ([5, 0, 70, 1, 33, 64, 0, 3], 256, 512),
        ([1] * 16, 192, 1024),
        ([0] * 200 + [40] + [0] * 55, 320, 320),
        ([3, 7, 11, 13] * 16, 1024, 128),
    ]:
        E, Tk = len(counts), sum(counts)
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        w = rnd_e4m3(E, N, K)
        s_w = (torch.rand(E, generator=g) * 0.01 + 0.001).float().cuda()
        s_a = torch.tensor([0.02], device="cuda")
        inv = torch.tensor([40.0], device="cuda")
        for pad in (0, 32):
            a_buf = rnd_e4m3(Tk + pad, K)
            refs = {
                ch.describe(): bytes_of(
                    grouped_gemm_fp8(a_buf, w, s_w, s_a, inv, cnt, dense, ch, Tk=Tk)
                )
                for ch in chains
            }
            for cfg in swapped_knobs_fp8():
                for ch in chains:
                    got = grouped_gemm_fp8(a_buf, w, s_w, s_a, inv, cnt, cfg, ch, Tk=Tk)
                    assert torch.equal(bytes_of(got), refs[ch.describe()]), (
                        cfg.describe(),
                        ch.describe(),
                        counts[:4],
                        N,
                        K,
                        pad,
                    )


@needs_device
def test_routing_methods_from_logits_match_the_cpu_model():
    """Every routing method the class implements, from logits, selects and weights as the exact CPU model (the strict order
    on exact binary32 scores); the layer's bits follow and repeat. DeepSeekV3's group-limited selection is a term of its
    method with the groups as parameters: the grouped layer from logits gives the bits of the pre-routed path on the
    model's ids and weights."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=7, adversarial=True)
    xq = _xq(x, m_x)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(8)) * 0.1)
        .float()
        .cuda()
    )
    W = _weights(w1, w2, m_x)
    for method, groups in (
        (RoutingMethodType.Default, None),
        (RoutingMethodType.Renormalize, None),
        (RoutingMethodType.Sigmoid, None),
        (RoutingMethodType.SigmoidRenorm, None),
        (RoutingMethodType.DeepSeekV3, None),
        (RoutingMethodType.TopK, None),
        (RoutingMethodType.TopKSigmoid, None),
        (RoutingMethodType.Llama4, None),
        (RoutingMethodType.DeepSeekV3, (8, 4)),
        (RoutingMethodType.DeepSeekV3, (4, 1)),
    ):
        rm = S.METHOD_OF_FLASHINFER[method]
        kwargs = {"routed_scaling_factor": 1.75} if rm.scaled else {}
        if groups is not None:
            rm = rm.with_groups(*groups)
            kwargs["n_group"], kwargs["topk_group"] = groups
        k_m = 1 if method is RoutingMethodType.Llama4 else k
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(
                    num_experts=E, top_k=k_m, method=method, **kwargs
                ),
                quant=QuantConfig(
                    weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
                ),
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )
        b = bias if rm.bias else None
        out = layer(_logits_pack(xq, logits, b), W)
        ex, w = R.route_cpu(
            logits.cpu(),
            k_m,
            rm,
            b.cpu() if b is not None else None,
            1.75 if rm.scaled else None,
        )
        ref = layer(_routed_pack(xq, ex.cuda(), w.cuda()), W)
        assert torch.equal(bits(out), bits(ref)), (method, groups)
        assert torch.equal(bits(layer(_logits_pack(xq, logits, b), W)), bits(out)), (
            method,
            groups,
        )


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the classes raises the mode's error at build or at the first call (top_k /
    num_experts outside the bounds, fused shared experts, an expert-parallel shard, group-limited routing, no finalize, an
    intermediate size that is not a multiple of 16, bf16 activations handed to the fp8 layer, an activation scale tensor,
    weights prepared for another activation, an unimplemented routing method from logits); a zero-token call returns an
    empty output."""
    k, H, I, E = 6, 512, 256, 64
    errors = (bi.BatchInvariantUnsupportedError, RuntimeError, NotImplementedError)
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=264), "intermediate_size"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.e4m3" in str(excinfo.value)
    routing = RoutingConfig(num_experts=E, top_k=k)
    experts = ExpertConfig(intermediate_size=I)
    quant = QuantConfig(
        weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
    )
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
    ):
        with pytest.raises(errors) as excinfo:
            MoELayer(
                MoEConfig(
                    routing=kw.get("routing", routing),
                    quant=quant,
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.e4m3" in str(excinfo.value), what
    # the bf16 quantization pair never reaches this runner
    with pytest.raises(errors):
        MoELayer(
            MoEConfig(
                routing=routing,
                quant=QuantConfig(),
                experts=experts,
                backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
            ),
            device=torch.device("cuda"),
        )
    layer = _layer(k, E, I)
    x, logits, w1, w2, m_x = _problem(8, k, H, I, E, seed=9)
    xq = _xq(x, m_x)
    W = _weights(w1, w2, m_x)
    with pytest.raises(errors):
        layer(
            _logits_pack(x, logits), W
        )  # bf16 activations: no runner accepts the pack
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            MoEActivationPack(
                xq,
                m_x,
                routing_input_mode=RoutingInputMode.FromLogits,
                routing_logits=logits,
            ),
            W,
        )  # a scale tensor
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, logits), _weights(w1[:, :I].contiguous(), w2, m_x, ReLU2())
        )  # the ungated weights with the gated class
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, logits), _weights(w1, w2, m_x, GeGLU())
        )  # another activation's chain
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=quant,
            experts=experts,
            backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    # the Llama4 (top-k sigmoid) routing term is the shared backend's: the layer routes from the logits and the bits repeat
    out_l = llama(_logits_pack(xq, logits), W)
    assert out_l.shape == (xq.shape[0], H) and out_l.dtype == torch.bfloat16
    assert torch.equal(
        out_l.view(torch.int16), llama(_logits_pack(xq, logits), W).view(torch.int16)
    )
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(_routed_pack(xq, ids, torch.ones(8, 1, device="cuda")), W).shape == (
        8,
        H,
    )
    # expert groups with a method other than DeepSeekV3 have no meaning in the module: the layer builds, the logits mode
    # refuses them, pre-routed ids never consult them (the bits of the ungrouped layer on the same ids and weights); the
    # DeepSeekV3 group shape rule is refused at the from-logits call too (the pre-routed modes never consult the term)
    grouped = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.Renormalize,
                n_group=2,
                topk_group=1,
            ),
            quant=quant,
            experts=experts,
            backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        grouped(_logits_pack(xq, logits), W)
    assert "n_group" in str(excinfo.value)
    g_ids, g_wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(grouped(_routed_pack(xq, g_ids.cuda(), g_wts.cuda()), W)),
        bits(layer(_routed_pack(xq, g_ids.cuda(), g_wts.cuda()), W)),
    )
    for n_group, topk_group, kk in (
        (3, 1, k),
        (64, 4, k),
        (8, 9, k),
        (8, 1, 9),
        (32, 4, 1),
    ):
        shaped = MoELayer(
            MoEConfig(
                routing=RoutingConfig(
                    num_experts=E,
                    top_k=kk,
                    method=RoutingMethodType.DeepSeekV3,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=1.0,
                ),
                quant=quant,
                experts=experts,
                backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
            ),
            device=torch.device("cuda"),
        )
        with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
            shaped(_logits_pack(xq, logits), W)
        assert "n_group" in str(excinfo.value) or "top_k" in str(excinfo.value), (
            n_group,
            topk_group,
            kk,
        )
    out = layer(_logits_pack(xq[:0], logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


@needs_device
def test_pre_routed_layers_build_under_any_routing_term():
    """A routing term the from-logits mode has no class for (``DeepSeekV3`` with more groups than the kernels hold,
    ``Llama4`` with ``top_k > 1``) does not refuse the layer at construction: the pre-routed modes never consult it, so the
    layer runs pre-routed (unpacked and packed) with the bits of the plain fp8 layer on the same ids and weights, and refuses
    from logits with the term's limit (the bf16 runner's rule, inherited)."""
    k, H, I, E = 6, 512, 256, 64
    x, logits, w1, w2, m_x = _problem(16, k, H, I, E, seed=13, adversarial=True)
    xq = _xq(x, m_x)
    W = _weights(w1, w2, m_x)
    quant = QuantConfig(
        weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
    )
    for routing, what, kk in (
        (
            RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.DeepSeekV3,
                n_group=32,
                topk_group=4,
                routed_scaling_factor=1.0,
            ),
            "n_group",
            k,
        ),
        (
            RoutingConfig(num_experts=E, top_k=2, method=RoutingMethodType.Llama4),
            "Llama4",
            2,
        ),
    ):
        layer = MoELayer(
            MoEConfig(
                routing=routing,
                quant=quant,
                experts=ExpertConfig(intermediate_size=I),
                backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=16),
            ),
            device=torch.device("cuda"),
        )
        plain = _layer(kk, E, I, tune_max=16)
        ids, weights = _reference_route(logits, kk, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        for packed in (False, True):
            assert torch.equal(
                bits(layer(_routed_pack(xq, ids, weights, packed=packed), W)),
                bits(plain(_routed_pack(xq, ids, weights, packed=packed), W)),
            ), (what, packed)
        with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
            layer(_logits_pack(xq, logits), W)
        assert what in str(excinfo.value), what


@needs_device
def test_packed_weights_round_in_the_combine_kernel():
    """The packed pre-routed mode's f32 weights enter the fold as their bf16 rounding (the packed word's value), rounded by
    the combine kernel: the bits equal the unpacked mode on the host-rounded weights, and differ from the unpacked mode on
    the unrounded f32 weights where the rounding moves a weight."""
    k, H, I, E = 4, 512, 256, 32
    x, logits, w1, w2, m_x = _problem(24, k, H, I, E, seed=21, adversarial=True)
    xq = _xq(x, m_x)
    W = _weights(w1, w2, m_x)
    layer = _layer(k, E, I)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda().float()
    packed = layer(_routed_pack(xq, ids, weights, packed=True), W)
    rounded = layer(_routed_pack(xq, ids, weights.to(torch.bfloat16).float()), W)
    assert torch.equal(bits(packed), bits(rounded))
    assert torch.equal(
        bits(layer(_routed_pack(xq, ids, weights.to(torch.bfloat16), packed=True), W)),
        bits(packed),
    )
    assert not torch.equal(weights, weights.to(torch.bfloat16).float())


@needs_device
def test_nan_logits_follow_the_shared_route_kernels_rule():
    """The from-logits path of this backend is the shared route kernel's, its NaN rule included: a NaN logit is never
    selected (the row maximum skips it, its softmax term is ``+0``, its selection score NaN), a row with fewer than ``top_k``
    finite logits takes the lowest untaken experts with NaN weights and its output is NaN, a row with exactly ``top_k`` finite
    logits selects them in the strict order, and the rows without a NaN keep the class's bits (the pre-routed path on the exact
    CPU model's ids and weights) -- in the fused route + plan form and the two-launch form."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    k, H, I, E = 8, 512, 256, 64
    keep = {
        1: 3,
        5: 0,
        9: 1,
        7: k,
    }  # token -> its finite logits (7: a boundary row, k of them)
    for T in (12, 40):
        x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=5, adversarial=True)
        logits = logits.clone()
        finite = {}
        for t, m in keep.items():
            cols = torch.randperm(E, generator=torch.Generator().manual_seed(t))[:m]
            finite[t] = cols.tolist()
            mask = torch.ones(E, dtype=torch.bool)
            mask[cols] = False
            logits[t, mask.cuda()] = float("nan")
        xq, W, layer = _xq(x, m_x), _weights(w1, w2, m_x), _layer(k, E, I)
        out = layer(_logits_pack(xq, logits), W)
        assert torch.equal(bits(layer(_logits_pack(xq, logits), W)), bits(out))
        logits_c = logits.cpu()
        for t, cols in finite.items():
            sel, ws = R.route_exact(logits_c[t].tolist(), k, S.SOFTMAX_RENORM)
            order = sorted(cols, key=lambda e: (-float(logits_c[t, e]), e))
            assert sel[: len(cols)] == order and all(0 <= e < E for e in sel), t
            if len(cols) < k:
                assert torch.isnan(out[t]).all(), t
            else:
                assert not torch.isnan(out[t]).any(), t
        # the rows without a NaN: the pre-routed path on the vectorised model's ids and weights gives their bits
        clean = torch.tensor([t not in keep for t in range(T)])
        idx = clean.nonzero().flatten()
        ref_ids, ref_w = _reference_route(
            logits_c[idx], k, RoutingMethodType.Renormalize
        )
        xq_clean = xq.view(torch.uint8)[idx.cuda()].view(torch.float8_e4m3fn)
        ref = layer(_routed_pack(xq_clean, ref_ids.cuda(), ref_w.cuda()), W)
        assert torch.equal(bits(out[idx.cuda()]), bits(ref))
        # the boundary row (exactly top_k finite logits): the exact model's selection and weights, pre-routed, give its bits
        sel, ws = R.route_exact(logits_c[7].tolist(), k, S.SOFTMAX_RENORM)
        xq7 = xq.view(torch.uint8)[7:8].view(torch.float8_e4m3fn)
        ref7 = layer(
            _routed_pack(
                xq7,
                torch.tensor([sel], dtype=torch.int32, device="cuda"),
                torch.tensor([ws], dtype=torch.float32, device="cuda"),
            ),
            W,
        )
        assert torch.equal(bits(out[7:8]), bits(ref7))


@needs_device
def test_routing_constants_are_read_from_the_cache_under_graph_capture():
    """The route kernels' absent bias and unit scale are cached device constants (``api._const``): a call under CUDA graph
    capture reads the cached tensor when it exists (no fill node in the graph), a default first met inside a capture is a
    fill of that capture and is never cached, and a captured from-logits decode call of the fp8 layer replays the eager
    call's bits with no fill launch among its kernels."""
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.fused_moe.batch_invariant import api

    dev = torch.device("cuda", torch.cuda.current_device())
    k, H, I, E = 8, 512, 256, 64
    x, logits, w1, w2, m_x = _problem(4, k, H, I, E, seed=31, adversarial=True)
    xq, W, layer = _xq(x, m_x), _weights(w1, w2, m_x), _layer(k, E, I, tune_max=4)
    pack = _logits_pack(xq, logits)
    eager = bits(layer(pack, W))  # the constants of this expert count are now cached
    b0, s0 = api._const(dev, E, 0.0), api._const(dev, 1, 1.0)
    assert api._const(dev, E, 0.0) is b0 and api._const(dev, 1, 1.0) is s0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        layer(pack, W)
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert api._const(dev, E, 0.0) is b0 and api._const(dev, 1, 1.0) is s0
            inside = api._const(
                dev, 3 * E, 0.0
            )  # first met inside the capture: a fill of this capture alone
            out = layer(pack, W)
    torch.cuda.synchronize()
    assert api._const(dev, 3 * E, 0.0) is not inside
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), eager)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    fills = [n for n in names if "Fill" in n or "fill" in n]
    assert len(fills) == 1, (
        names
    )  # the one constant first met inside the capture, none for the cached bias and scale
    assert torch.equal(bits(out), eager)


@needs_device
def test_alternating_inputs_under_programmatic_dependent_launch():
    """Two W8A8 problems run alternately through the op, back to back without host syncs (the kernels' PDL chains overlap
    and every reused intermediate holds the other call's bytes), many times; every output equals the problem's isolated
    result bit for bit."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api_fp8 import (
        MoEWeightsE4M3,
        fused_moe_fp8,
    )

    T, k, H, I, E = 64, 8, 256, 64, 256
    probs = []
    for seed in (3, 11):
        x, logits, w1, w2, m_x = _problem(T, k, H, I, E, seed=seed, adversarial=True)
        W = MoEWeightsE4M3.prepare(w1, w2, m_x, M_INTER)
        xq = _xq(x, m_x)
        ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
        ref = bits(fused_moe_fp8(xq, ids, wts, W))
        torch.cuda.synchronize()
        probs.append((xq, ids, wts, W, ref))
    outs = []
    for _ in range(20):
        for xq, ids, wts, W, _r in probs:
            outs.append(fused_moe_fp8(xq, ids, wts, W))
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(bits(out), probs[i % 2][4]), i


def _flashinfer_fp8_per_tensor_reference(
    x, w1, w2, ids, weights, m_x, m_inter, act, activation_fn=None
) -> torch.Tensor:
    """FlashInfer's torch reference of the fp8 per-tensor MoE: the form of ``tests/moe/test_unified_moe_fp8.py``'s
    ``_per_tensor_fp8_reference`` (the reference its TRTLLM-Gen backend is checked against), computed here from the canonical
    bf16 weights. The activations and every expert quantized to e4m3 by the static multipliers (``448 / amax`` per expert) and
    dequantized to fp32; fp32 GEMM1; the typed activation in fp32 (``tests.moe.utils.compute_reference_activation``'s formulas,
    the layer's own, so every activation is covered; FlashInfer's form spells out SwiGLU / SwiGLUStep / GeGLU / ReLU2); the
    intermediate requantized through the static multiplier; bf16 expert outputs; the routing weights rounded to bf16; the fp32
    routed sum. Independent of the class (its exact model is :mod:`flashinfer.fused_moe.batch_invariant.spec_fp8`; the bitwise
    tests above are the promise). ``activation_fn`` replaces the activation (the negative control of the tolerance)."""
    from tests.moe.utils import compute_reference_activation, powlu_ref

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weights = weights.to(torch.bfloat16).float()
    x_deq = (x.float() * m_x).clamp(-fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    ).float() / m_x

    def quant_dequant(w):
        amax = w.float().abs().amax(dim=(-1, -2))
        s = torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))
        q = (w.float() * s[:, None, None]).clamp(-fp8_max, fp8_max)
        return q.to(torch.float8_e4m3fn).float() / s[:, None, None]

    w1_deq, w2_deq = quant_dequant(w1), quant_dequant(w2)
    I = w2.shape[2]
    if activation_fn is None:

        def activation_fn(gemm1):
            if isinstance(
                act, PowLU
            ):  # FlashInfer's PowLU form (``tests/moe/test_unified_moe_cutlass.py``): the
                up, gate = gemm1.split(
                    I, dim=-1
                )  # adaptive power on the gate, the limit clamps both branches
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
        inter_deq = (inter * m_inter).clamp(-fp8_max, fp8_max).to(
            torch.float8_e4m3fn
        ).float() / m_inter
        y = (inter_deq @ w2_deq[e].t()).to(torch.bfloat16).float()
        out[tok] += weights[tok, slot, None] * y
    return out


def _wrong_activation(act, I: int):
    """A genuinely different formula over the same weights and scales (FlashInfer's negative control: the tolerance must
    reject it): the up branch alone for a gated activation, plain ReLU for an ungated one (the identity for ReLU itself)."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    if activation_of(act).gated:
        return lambda gemm1: gemm1[:, :I]
    if isinstance(act, ReLU):
        return lambda gemm1: gemm1
    return torch.relu


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_plausible_against_flashinfers_fp8_torch_reference(act):
    """An observation on moderate inputs under FlashInfer's own discipline for this quantization pair
    (``tests/moe/test_unified_moe_fp8.py``: ``check_accuracy`` at atol 0.05 / rtol 0.3 on 99 % of the outputs against the torch
    reference above, and the negative control that the tolerance rejects a wrong activation formula). Not a promise of the
    class (the reference and the class round at different points; the exact model is the promise, tested bit for bit above),
    and never a comparison with another backend's output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    T, k, H, I, E = 128, 4, 512, 256, 32
    m_inter = 8.0
    x, logits, w1, w2, m_x = _problem(
        T, k, H, I, E, seed=11, gated=activation_of(act).gated
    )
    out = _layer(k, E, I, act)(
        _logits_pack(_xq(x, m_x), logits), _weights(w1, w2, m_x, act, m_inter=m_inter)
    ).float()
    vals, ids = torch.topk(
        logits, k, dim=-1
    )  # the layer's Renormalize routing: top-k, softmax over the selected
    weights = torch.softmax(vals, dim=-1)
    ref = _flashinfer_fp8_per_tensor_reference(
        x, w1, w2, ids, weights, m_x, m_inter, act
    )
    check_accuracy(out, ref, atol=0.05, rtol=0.3, percent=0.99)
    wrong = _flashinfer_fp8_per_tensor_reference(
        x,
        w1,
        w2,
        ids,
        weights,
        m_x,
        m_inter,
        act,
        activation_fn=_wrong_activation(act, I),
    )
    with pytest.raises(Exception, match="Mismatch percentage"):
        check_accuracy(out, wrong, atol=0.05, rtol=0.3, percent=0.99)
