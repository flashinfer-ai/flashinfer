# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(MXFP8, MXFP8)``, the backend grouping the fp8 formats with block
scales under one key as TRTLLM-Gen's ``TrtllmFp8BlockConfig`` does): the classes ``moe.mxfp8.<act>``, one per typed activation of the layer (the bf16 classes' chains on the dequantised accumulators), the
expert weights and the activations quantized by the batch-invariant MXFP8 quantizer, the two GEMMs as grouped placements of
the dense batch-invariant MXFP8 GEMM class of the part (SM107: the K = 64 instruction class; SM100 / SM103: the K = 32 one).

The class has exact CPU models of everything but the block-scaled step (the routing, the chain on the fp32 accumulators,
the quantizer, the slot-order combine), so the tests are bit comparisons: the layer's output against the class's own
composition (the routing on the CPU, the padded permutation, the dense batch-invariant ``mm_mxfp8`` per expert, the CPU
chain, the quantizer's CPU reference on the intermediate, the dense ``mm_mxfp8`` again, the CPU combine) for every
activation, run-to-run determinism including CUDA-graph replay, batch / permutation / sub-batch / hot-expert invariance,
agreement of every tactic the autotuner may pick, the three routing input modes giving the same bits, the error contract
(an unsupported configuration raises before any kernel runs), zero tokens, the largest expert count, and the plausibility
of the op against FlashInfer's dequantized MXFP8 MoE reference (the one its TRTLLM-Gen MXFP8 tests use) under its accuracy
discipline; no test compares against another backend's kernels. The device-free tests pin the registration, the
fingerprint dimensions (the instruction class of the part among them), the specification terms (no scale symbol: the
bf16 chains, the plain GEMM2 conversion), the shape rules and the dispatcher's membership in the enumerated space.
"""

from __future__ import annotations

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
)
from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND
from flashinfer.fused_moe.batch_invariant.runner_fp8_block import (
    BatchInvariantFp8BlockRunner,
)
from flashinfer.fused_moe.batch_invariant.runner_mxfp8 import (
    MOE_MXFP8_ARCHS,
    BatchInvariantMxfp8Runner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_MXFP8_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
_QUANT = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
KEY = BatchInvariantFp8BlockRunner.backend_key
BLOCK = 32

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


def bits(t: torch.Tensor) -> torch.Tensor:
    return (
        t.contiguous().view(torch.int16)
        if t.dtype == torch.bfloat16
        else t.contiguous().view(torch.int32)
    )


def _problem(T, k, H, I, E, seed, gated=True, adversarial=False):
    """Activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated)."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        # spread exponents inside the 32-element scale blocks (the block scale saturates the small values), near-cancelling
        # pairs and ties in the logits: what moves bits when an order moves
        x = (
            torch.randn(T, H, generator=g)
            * torch.exp2(torch.randint(-8, 6, (T, H), generator=g).float())
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


def _config(k, E, I, act=_SWIGLU, method=RoutingMethodType.Renormalize, tune_max=4096):
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=_QUANT,
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1, w2, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=_QUANT,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(KEY, view)
    return pack


def _quant(x: torch.Tensor):
    """The canonical MXFP8 activation pack by the backend's quantizer."""
    return BatchInvariantFp8BlockConfig.prepare_activations(x, quant=_QUANT)


def _logits_pack(x, logits, bias=None) -> MoEActivationPack:
    xq, xsf = x if isinstance(x, tuple) else _quant(x)
    return MoEActivationPack(
        xq,
        xsf,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(x, ids, weights, packed=False) -> MoEActivationPack:
    xq, xsf = x if isinstance(x, tuple) else _quant(x)
    if packed:
        return MoEActivationPack(
            xq,
            xsf,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        xq,
        xsf,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The config is a known backend of the layer, mapped to the fp8-block runner, absent from the default candidates; the
    runner declares the three routing modes, every activation class of the layer and the pairs it serves (MXFP8 x MXFP8
    here, run by the pair's runner); a pair it does not serve is not admitted."""
    assert BatchInvariantFp8BlockConfig in ALL_BACKEND_CONFIGS
    assert (
        _BACKEND_RUNNERS[BatchInvariantFp8BlockConfig] is BatchInvariantFp8BlockRunner
    )
    assert not any(
        isinstance(c, BatchInvariantFp8BlockConfig) for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantFp8BlockConfig.supported(a)
    ] == [100, 103, 107]
    assert set(BatchInvariantFp8BlockRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert (
        BatchInvariantFp8BlockRunner.supported_quant_variants
        == BatchInvariantMxfp8Runner.supported_quant_variants
        == ((QuantFormat.MXFP8, QuantFormat.MXFP8),)
    )
    assert issubclass(BatchInvariantMxfp8Runner, BatchInvariantFp8BlockRunner)
    assert (
        BatchInvariantFp8BlockRunner.implementation(_QUANT.pair)
        is BatchInvariantMxfp8Runner
    )
    assert set(BatchInvariantFp8BlockRunner.supported_activation_classes) == {
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
    assert KEY == "batch_invariant_fp8_block"
    assert repr(BatchInvariantFp8BlockConfig()) == "BatchInvariantFp8BlockConfig()"
    deepseek = QuantConfig(
        weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
    )
    assert not BatchInvariantFp8BlockRunner.supports_quant(deepseek)
    assert BatchInvariantFp8BlockRunner.implementation(deepseek.pair) is None


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_mxfp8", ...)`` takes the geometry, the routing method, the activation and the architecture (the
    instruction class of the dense MXFP8 GEMM: SM107 folds 64 elements per step, SM100 / SM103 32); the classes are distinct
    from the bf16 backend's and from each other; production fingerprints pinned."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.spec_mxfp8 import (
        MoEMxfp8Spec,
        spec_fingerprint_mxfp8,
    )

    fp = lambda **kw: bi.fingerprint(
        "fused_moe_mxfp8", H=2048, I=768, top_k=8, num_experts=128, **kw
    )  # noqa: E731
    assert bi.class_name("fused_moe_mxfp8") == "moe.mxfp8"
    s107, s100 = fp(arch="sm_107"), fp(arch="sm_100")
    assert s107 == "moe-09e3cd790ad7fc25" and s100 == "moe-f3eab1a331649294"
    assert (
        fp(arch="sm_103") == s100
        and fp(arch=(10, 7)) == s107
        and fp(arch=(10, 0)) == s100
    )
    assert fp(arch="sm_107", act="relu2") == "moe-9022e7b7e0bdaa66"
    assert fp(arch="sm_100", act="relu2") == "moe-844e0cdbf63de703"
    assert (
        fp(arch="sm_107", act=SwiGLU()) == s107
        and fp(arch="sm_107", act="silu") == s107
    )
    distinct = {
        s107,
        s100,
        fp(arch="sm_107", act="relu2"),
        fp(arch="sm_107", act=GeGLUTanh()),
        fp(arch="sm_107", act=SwiGLU(alpha=1.702, beta=1.0, limit=7.0)),
        fp(arch="sm_107", act=SiLU()),
        fp(arch="sm_107", method="Default"),
        fp(arch="sm_107", method=RoutingMethodType.DeepSeekV3),
        bi.fingerprint(
            "fused_moe_mxfp8", H=2048, I=768, top_k=4, num_experts=128, arch="sm_107"
        ),
        bi.fingerprint(
            "fused_moe_mxfp8", H=4096, I=768, top_k=8, num_experts=128, arch="sm_107"
        ),
        bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=128),
    }
    assert len(distinct) == 11
    assert s107 == spec_fingerprint_mxfp8(
        2048, 768, 8, 128, S.SOFTMAX_RENORM, "silu", (), True, "mxfp8_k64"
    )
    # the routing terms the bf16 model serves are classes of this backend too (Llama4 = the top-1 TopKSigmoid term, its own
    # class); a method the classes do not implement is refused
    with pytest.raises(ValueError):
        fp(arch="sm_107", method=RoutingMethodType.MiniMax2)
    assert fp(arch="sm_107", method=RoutingMethodType.TopKSigmoid) != s107
    assert bi.fingerprint(
        "fused_moe_mxfp8",
        method=RoutingMethodType.Llama4,
        H=2048,
        I=768,
        top_k=1,
        num_experts=128,
        arch="sm_107",
    ) == bi.fingerprint(
        "fused_moe_mxfp8",
        method=RoutingMethodType.TopKSigmoid,
        H=2048,
        I=768,
        top_k=1,
        num_experts=128,
        arch="sm_107",
    )
    with pytest.raises(ValueError):
        fp(arch="sm_107", act="tanh")
    with pytest.raises(ValueError):
        fp(arch="sm_90")
    # the terms: GEMM1 is the bf16 class's chain (no alpha, per-token or static scale symbol), GEMM2 the plain conversion
    from flashinfer.fused_moe.batch_invariant import mterm as term

    q = MoEMxfp8Spec(64, 32, 2, 8)
    b = S.MoESpec(64, 32, 2, 8)
    assert term.sexpr(q.gemm1()) == term.sexpr(b.gemm1()) and term.sexpr(
        q.combine()
    ) == term.sexpr(b.combine())
    assert term.sexpr(q.gemm2()) == "(cvt bf16 (acc2 0 0))"
    for act in ACTIVATIONS:
        a = S.activation_of(act)
        qa = MoEMxfp8Spec(64, 32, 2, 8, act=a.act, params=a.params, gated=a.gated)
        ba = S.MoESpec(64, 32, 2, 8, act=a.act, params=a.params, gated=a.gated)
        g1 = term.sexpr(qa.gemm1())
        assert (
            g1 == term.sexpr(ba.gemm1())
            and "alpha" not in g1
            and "pts" not in g1
            and "sym" not in g1
        )
        assert qa.name == f"moe.mxfp8.{a.act}"
    with pytest.raises(AssertionError):
        MoEMxfp8Spec(64, 32, 2, 8, mx_class="nvfp4")


def test_shape_rules_and_padded_bound():
    """The class's acceptance rule (H, I multiples of 32; 1 <= E <= 1024; 1 <= top_k <= min(E, 16); unsplit expert weights) raises
    the mode's error elsewhere; the padded row bound covers every routing without reading the device."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        GROUP_ALIGN,
        check_shape_mxfp8,
        padded_rows_bound,
        sf_numel,
    )

    for H, I, E, k in (
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (32, 32, 1, 1),
        (2560, 640, 1024, 16),
    ):
        check_shape_mxfp8(H, I, E, k)
    for args in (
        (2048 + 16, 768, 128),
        (2048, 768 + 8, 128),
        (0, 768, 128),
        (2048, 768, 0),
        (2048, 768, 1025),
        (2048, 768, 8, 9),
        (2048, 768, 128, 17),
        (2048, 768, 128, 0),
        (4096, 32, 8),
    ):
        with pytest.raises(BatchInvariantUnsupportedError, match="moe.mxfp8"):
            check_shape_mxfp8(*args)
    assert (
        GROUP_ALIGN == 128
        and sf_numel(128, 2048) == 128 * 64
        and sf_numel(1, 32) == 512
    )
    for T, k, E in (
        (1, 1, 1),
        (1, 8, 128),
        (37, 2, 8),
        (300, 6, 64),
        (4096, 8, 1024),
        (16384, 2, 8),
    ):
        n = T * k
        bound = padded_rows_bound(T, k, E)
        assert bound % GROUP_ALIGN == 0 and bound >= n
        # the worst case: as many active experts as possible, each padding 127 rows
        worst = min(E, n) * (GROUP_ALIGN - 1) + n
        assert bound >= worst and bound < worst + GROUP_ALIGN
    assert padded_rows_bound(0, 8, 128) == 0


def test_swapped_k_block_is_a_shape_rule():
    """The swapped body's k-block (K elements per smem stage) is a function of the class and K alone: two atoms of K when
    they divide K, else the one-atom default (`swap_k_block`), spelled into the kernel name with the default as 0
    (`swap_k_block_arg`: one object per k-block, the shared compile wrapper's ``_kb`` suffix), accepted by the body iff it
    divides K; the grouped body and the dense class keep the default (`Knobs.k_block` does not exist in this tree: the
    record space is untouched)."""
    from flashinfer.fused_moe.batch_invariant.blocks.compile import swapped_kernel_name
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        SWAP_K_BLOCKS,
        can_implement_swapped_mxfp8,
        swap_k_block,
        swap_k_block_arg,
        swapped_knobs_mx,
    )

    assert SWAP_K_BLOCKS == (128, 256)
    for fmt in ("mxfp8", "mxfp8_k64"):
        assert [
            swap_k_block(fmt, K) for K in (128, 256, 384, 512, 640, 768, 2048, 7168)
        ] == [128, 256, 128, 256, 128, 256, 256, 256]
        assert swap_k_block_arg(fmt, 640) == 0 and swap_k_block_arg(fmt, 2048) == 256
        assert (
            swap_k_block_arg(fmt, 2048, 128) == 0
            and swap_k_block_arg(fmt, 2048, 256) == 256
        )
        cfg = swapped_knobs_mx()[0]
        f32 = GroupedChain("plain", "silu", "f32")
        mx = MxChain.of(f32, fmt)
        assert swapped_kernel_name(cfg, mx, k_block=256).endswith(
            "_kb256"
        ) and not swapped_kernel_name(cfg, mx).endswith("_kb256")
        assert can_implement_swapped_mxfp8(cfg, 256, 2048, 8, f32, fmt, 256)
        assert not can_implement_swapped_mxfp8(cfg, 256, 640, 8, f32, fmt, 256), (
            "a k-block that does not divide K is refused"
        )
        assert not can_implement_swapped_mxfp8(cfg, 256, 2048, 8, f32, fmt, 512), (
            "four atoms are not admitted"
        )


def test_swapped_box_form_compiles_under_the_chain_trigger_rule_only(monkeypatch):
    """The MX chains run the backend's swapped body (:mod:`blocks.swap`), which reads the plan's group counts under the
    chain's trigger rule (``pdl.BOX_FORM_TRIGGERS``: the dependents launch after the predecessor's own wait, the body's own
    wait before the tables is the margin) and whose builder asserts the shipped placement is one of them; the gather form
    belongs to the bf16 and the 4-bit block-scaled classes; a k-block is whole instruction steps of the class."""
    from flashinfer.fused_moe.batch_invariant import pdl
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.blocks.swap import SwappedGroupedGemm
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        GEMM2_CHAIN,
        swapped_knobs_mx,
    )

    cfg = swapped_knobs_mx()[0]
    assert pdl.TRIGGER in pdl.BOX_FORM_TRIGGERS
    for fmt in ("mxfp8", "mxfp8_k64"):
        ch = MxChain("glu", "silu", "mxfp8", (), fmt)
        SwappedGroupedGemm(cfg, ch)
        assert SwappedGroupedGemm(cfg, ch, k_block=256).k_block == 256
        with pytest.raises(AssertionError):
            SwappedGroupedGemm(cfg, ch, k_block=48)
        with pytest.raises(AssertionError):
            SwappedGroupedGemm(cfg, ch, gather=True)
    monkeypatch.setattr(pdl, "TRIGGER", "early")
    with pytest.raises(AssertionError):
        SwappedGroupedGemm(cfg, GEMM2_CHAIN)


def test_mx_classes_run_on_the_unified_bodies():
    """The MXFP8 classes have no body of their own: the backend's grouped and swapped bodies (:mod:`blocks.kernel`,
    :mod:`blocks.swap`) take the MX chain through the block-scaled operand block (e4m3 elements, UE8M0 scales per 32
    elements, four epilogue warpgroups, the 128-row scale atom as the group alignment, both packings) and the fused
    requantization through the epilogue block's hooks (its own loops, its exchange buffer for the gated chain only); the
    compiled objects carry the instruction class, the packing and the k-block in their names; the shared acceptance runs at
    the chain's class; one JIT module (``batch_invariant_moe``) holds the kernels and the MX scatter, with no MX-specific
    body in its source closure."""
    from flashinfer.batch_invariant import jit
    from flashinfer.fused_moe.batch_invariant.blocks import compile as C
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import (
        MxChain,
        grouped_epilogue_quant_loop,
        swap_xch_bytes,
        swapped_epilogue_quant_loop,
        xch_bytes,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.kernel import GroupedBlockGemm
    from flashinfer.fused_moe.batch_invariant.blocks.operands import (
        BlockScaledOperands,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.swap import (
        SwappedGroupedGemm,
        is_swapped_record,
        swapped_knobs,
    )
    from flashinfer.fused_moe.batch_invariant.mx_permute import MODULE as SCATTER_MODULE
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        GEMM2_CHAIN,
        swapped_knobs_mx,
    )
    from flashinfer.gemm.batch_invariant.blocks.operands import IN_DT
    from flashinfer.gemm.batch_invariant.generator import Knobs

    dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
    quant, plain = MxChain("glu", "silu", "mxfp8"), GEMM2_CHAIN
    assert quant.quant and not plain.quant
    assert quant.operands == () and not quant.needs_aux and not quant.per_row_scale
    assert quant.grouped_loop() is grouped_epilogue_quant_loop
    assert quant.swapped_loop() is swapped_epilogue_quant_loop
    assert plain.grouped_loop() is None and plain.swapped_loop() is None
    # the exchange buffer: the gated fused requantization's alone (an ungated block is one thread's / one warp's)
    assert xch_bytes(quant, 4) > 0 and swap_xch_bytes(quant, 16) > 0
    assert (
        xch_bytes(plain, 4) == 0 and xch_bytes(MxChain("act", "relu2", "mxfp8"), 4) == 0
    )
    for fmt in ("mxfp8", "mxfp8_k64"):
        for packing in ("padded", "dense"):
            body = GroupedBlockGemm(dense, quant.at(fmt), packing=packing)
            ops = body.operands
            assert ops == BlockScaledOperands(fmt, packing) and ops.block_scaled
            assert (ops.sf_vec, ops.elem_bits, ops.group_align) == (32, 8, 128)
            assert ops.dtype is IN_DT[fmt] and body.epi_warpgroups == 4
            assert ops.dense_a == (packing == "dense")
        assert swapped_knobs(fmt) == swapped_knobs_mx()
        sw = swapped_knobs_mx()[0]
        assert is_swapped_record(sw, fmt)
        s = SwappedGroupedGemm(sw, plain.at(fmt), k_block=256)
        assert s.operands.block_scaled and s.k_block == 256
        # the scale-factor warp follows the epilogue warpgroups: the shared epilogue's count at the token tile for the
        # plain chains, the one warpgroup the fused requantization's own loop is written for
        assert s.sf_warp == len(s.epilogue_warps) == 4 * s.epi_wgs
        q = SwappedGroupedGemm(sw, quant.at(fmt))
        assert q.epi_wgs == 1 and q.sf_warp == 4
        assert C.can_implement_grouped_blockscaled(dense, 256, 2048, 8, quant, fmt)
        assert C.can_implement_swapped_blockscaled(sw, 256, 2048, 8, quant, fmt)
        assert not C.can_implement_grouped_blockscaled(dense, 256, 2064, 8, quant, fmt)
    # a bf16 chain names no instruction class: the shared wrapper takes the MX chain (the MX dispatcher converts)
    assert not C.can_implement_grouped_blockscaled(
        dense, 256, 2048, 8, GroupedChain("plain", "silu", "bf16"), "mxfp8"
    )
    assert C.kernel_name(dense, quant.at("mxfp8_k64"), packing="dense").startswith(
        "grouped_mxfp8k64_densea_swiglu_mxfp8_"
    )
    assert C.kernel_name(dense, quant, packing="padded").startswith(
        "grouped_mxfp8_swiglu_mxfp8_"
    )
    name = C.swapped_kernel_name(
        swapped_knobs_mx()[1], plain.at("mxfp8_k64"), k_block=256
    )
    # the plain chain on the shared swapped epilogue: the k-block and the epilogue warpgroups of the 16-token tile
    assert name.startswith("swapped_mxfp8k64_plain_bf16_") and name.endswith("_kb256_e2")
    name = C.swapped_kernel_name(
        swapped_knobs_mx()[1], quant.at("mxfp8_k64"), k_block=256
    )
    # the fused requantization's own loop: one warpgroup, no epilogue suffix
    assert name.startswith("swapped_mxfp8k64_swiglu_mxfp8_") and name.endswith("_kb256")
    assert C.MODULE == SCATTER_MODULE == "batch_invariant_moe"
    assert "batch_invariant_moe_mxfp8" not in jit.MODULE_ROOTS
    srcs = jit.relative_sources("batch_invariant_moe")
    gone = ("blockscaled_kernel.py", "swap_mxfp8.py", "compile_mxfp8.py")
    assert not any(os.path.basename(src) in gone for src in srcs), srcs
    for src in (
        "fused_moe/batch_invariant/mx_permute.py",
        "fused_moe/batch_invariant/blocks/epilogue_mx.py",
        "fused_moe/batch_invariant/blocks/operands.py",
    ):
        assert src in srcs, srcs


@pytest.mark.parametrize(
    "family,sm_count", [("sm_100", 148), ("sm_103", 148), ("sm_107", 204)]
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The grouped MX space: 72 dense block-scaled records at swizzle 1 (one list for both instruction classes, every record
    well-formed for both) and the swapped record space (the three narrow token tiles of the swapped decode body, disjoint
    from the grouped space); the dispatcher's picks for both classes, every chain of the layer and a grid of token counts
    are members of the union the guarantee is claimed over; at a few rows per active expert the ranking picks the swapped
    records, at a prefill's rows the grouped ones; ``swap`` forces either body."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        GEMM2_CHAIN,
        all_valid_configs_mx,
        all_valid_configs_mx_swapped,
        default_configs_mx,
        grouped_knobs_mx,
        packing_for,
        rank_grouped_mx,
        swapped_knobs_mx,
    )
    from flashinfer.fused_moe.batch_invariant.schedule import rows_per_expert
    from flashinfer.gemm.batch_invariant.schedule import mx_class_for

    ks = grouped_knobs_mx("mxfp8")
    assert len(ks) == 72 and ks == grouped_knobs_mx("mxfp8_k64")
    sw = swapped_knobs_mx()
    assert [c.tile_n for c in sw] == [8, 16, 32] and all(
        c.swap_ab
        and c.tile_m == 128
        and c.cluster_shape_mn == (1, 1)
        and not c.use_2cta_instrs
        and c.well_formed("mxfp8")
        and c.well_formed("mxfp8_k64")
        for c in sw
    )
    assert not set(sw) & set(ks)
    for k in ks:
        assert (
            k.well_formed("mxfp8")
            and k.well_formed("mxfp8_k64")
            and k.swizzle_size == 1
        )
        assert (
            k.cluster_k == 1
            and not k.swap_ab
            and not k.b_reuse
            and not k.cta_chunks
            and k.tile_n in (64, 128, 192, 256)
        )
    fmt = mx_class_for("mxfp8", family)
    assert fmt == ("mxfp8_k64" if family == "sm_107" else "mxfp8")
    chains = [
        GroupedChain(
            S.activation_of(a).chain,
            S.activation_of(a).act,
            "bf16",
            S.activation_of(a).params,
        )
        for a in ACTIVATIONS
    ]
    for H, I, E, k in (
        (512, 256, 64, 6),
        (2048, 768, 128, 8),
        (4096, 1536, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (2048, 1024, 128, 6),
        (256, 128, 1024, 16),
    ):
        for chain in chains:
            n1 = 2 * I if chain.pair else I
            space1 = set(all_valid_configs_mx(n1, H, E, chain, fmt))
            space2 = set(all_valid_configs_mx(H, I, E, GEMM2_CHAIN, fmt))
            assert len(space1) == 72 and len(space2) == 72
            sw1 = set(all_valid_configs_mx_swapped(n1, H, E, chain, fmt))
            sw2 = set(all_valid_configs_mx_swapped(H, I, E, GEMM2_CHAIN, fmt))
            assert sw1 == set(sw) and sw2 == set(sw)
            for T in (1, 2, 7, 64, 300, 1024, 4096, 16384):
                c1, c2 = default_configs_mx(
                    T, k, H, I, E, chain, fmt, sm_count, family, nominal=True
                )
                assert c1 in space1 | sw1 and c2 in space2 | sw2, (
                    family,
                    T,
                    H,
                    I,
                    E,
                    chain.describe(),
                    c1.describe(),
                    c2.describe(),
                )
                rows = rows_per_expert(T, k, E)
                if rows <= 2:
                    # a few rows per active expert: the swapped decode body (the weights read once per expert) on both GEMMs
                    assert c1.swap_ab and c2.swap_ab, (
                        family,
                        T,
                        H,
                        I,
                        E,
                        c1.describe(),
                        c2.describe(),
                    )
                    assert packing_for(c1, c2) == "dense"
                if rows >= 512:
                    assert (
                        not c1.swap_ab
                        and not c2.swap_ab
                        and packing_for(c1, c2) == "padded"
                    ), (family, T, H, I, E, c1.describe(), c2.describe())
                if not (c1.swap_ab or c2.swap_ab):
                    # the padded body's picks at a decode-sized group: the 1-CTA tiles in an N-cluster (the dense dispatcher's guard)
                    if rows <= 64:
                        for c in (c1, c2):
                            assert (
                                not c.use_2cta_instrs
                                and c.cluster_shape_mn[0] == 1
                                and c.tile_n >= 64
                            ), (family, T, H, I, E, c.describe())
                s1, s2 = default_configs_mx(
                    T, k, H, I, E, chain, fmt, sm_count, family, nominal=True, swap=True
                )
                assert s1.swap_ab and s2.swap_ab and s1 in sw1 and s2 in sw2
                d1, d2 = default_configs_mx(
                    T,
                    k,
                    H,
                    I,
                    E,
                    chain,
                    fmt,
                    sm_count,
                    family,
                    nominal=True,
                    swap=False,
                )
                assert (
                    not d1.swap_ab and not d2.swap_ab and d1 in space1 and d2 in space2
                )
                ranked = rank_grouped_mx(
                    max(1, T * k // E),
                    n1,
                    H,
                    E,
                    chain,
                    fmt,
                    sm_count,
                    family,
                    3,
                    True,
                    min(E, T * k),
                )
                assert (
                    1 <= len(ranked) <= 3
                    and len(set(ranked)) == len(ranked)
                    and set(ranked) <= space1 | sw1
                )


def test_unsupported_device_and_quant_raise_before_any_kernel():
    """Constructing the backend's runner gives the pair's runner; on a CPU device it fails its support check with the mode's
    error naming the class. A pair the backend does not serve is refused with the pairs served -- at construction and at
    ``prepare_weights`` / ``prepare_activations``, before any kernel or quantization runs (the layer never admits it:
    ``supports_quant`` is false); the pair's runner itself admits its own pair only."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    runner = BatchInvariantFp8BlockRunner(_config(8, 64, 256), torch.device("cpu"))
    assert type(runner) is BatchInvariantMxfp8Runner
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        runner._check_support()
    assert "moe.mxfp8" in str(excinfo.value) and "cpu" in str(excinfo.value)
    for quant in (
        QuantConfig(),
        QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
    ):
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=8),
            quant=quant,
            experts=ExpertConfig(intermediate_size=256),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        )
        assert not BatchInvariantFp8BlockRunner.supports_quant(quant)
        with pytest.raises(BatchInvariantUnsupportedError, match="MXFP8") as excinfo:
            BatchInvariantFp8BlockRunner(cfg, torch.device("cpu"))
        msg = str(excinfo.value)
        assert KEY in msg and "DeepSeekFp8" in msg and quant.weight.name in msg
        with pytest.raises(BatchInvariantUnsupportedError, match="MXFP8"):
            BatchInvariantFp8BlockConfig.prepare_activations(
                torch.zeros(4, 64, dtype=torch.bfloat16), quant=quant
            )
        with pytest.raises(BatchInvariantUnsupportedError, match="MXFP8"):
            BatchInvariantFp8BlockConfig.prepare_weights(
                torch.zeros(2, 64, 64, dtype=torch.bfloat16),
                torch.zeros(2, 64, 32, dtype=torch.bfloat16),
                quant=quant,
            )
        with pytest.raises(NotImplementedError):
            BatchInvariantMxfp8Runner(cfg, torch.device("cpu"))._check_support()


# ----------------------------------------------------------------------------- GPU


def _reference_route(logits, k, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _unswizzle(sf_flat, rows, K):
    from flashinfer.batch_invariant.quantization.layout import unswizzle_sf

    return unswizzle_sf(sf_flat.reshape(-1), rows, K, BLOCK, "128x4")


def _swizzle(sf_rows, K):
    from flashinfer.batch_invariant.quantization.layout import swizzle_sf

    return swizzle_sf(sf_rows, sf_rows.shape[0], K, BLOCK, "128x4")


def _quant_ref(x: torch.Tensor):
    """The quantizer class's CPU reference on bf16 rows: ``(codes uint8 [M, K], sf [M, K / 32] uint8)``."""
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )

    pk, _, sfr = quantize_mx_reference(x.detach().cpu())
    return pk, sfr


def _reference_ffn(xq, xsf, ids, weights, w1, w2, act):
    """The class composed from its own pieces on the quantized activation pack: the padded permutation of the class, the
    scatter by indexing (the padding rows unspecified), GEMM1 as the dense batch-invariant MXFP8 GEMM (``mm_mxfp8_batch_invariant``, what ``mm_mxfp8(..., batch_invariant=True)``
    runs; the fp32 output the oracle needs, the instruction class pinned: the same fold) per expert, the CPU chain ``act`` on those accumulators, the quantizer's CPU reference on the
    bf16 intermediate, GEMM2 as the dense ``mm_mxfp8`` (bf16 out) per expert, the CPU slot-order combine. Bit for bit what the
    fused op must produce."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        MoEWeightsMXFP8,
        mx_class_of,
        permute_plan_padded,
    )
    from flashinfer.gemm.batch_invariant.api import mm_mxfp8_batch_invariant
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP
    from flashinfer.gemm.batch_invariant.schedule import chunks_for

    T, H = xq.shape
    E, k = w1.shape[0], ids.shape[1]
    I = w2.shape[2]
    fmt = mx_class_of(xq.device)
    assert chunks_for(w1.shape[1], H, fmt) == 1 and chunks_for(H, I, fmt) == 1
    W = MoEWeightsMXFP8.prepare(w1, w2, gated=act.gated, act=act.act, params=act.params)
    # the weights are the quantizer class per expert (the interleave before the quantization)
    w1i = interleave_fc1(w1) if act.gated else w1
    for e in (0, E - 1):
        pk, sfr = _quant_ref(w1i[e])
        assert torch.equal(W.w1[e].view(torch.uint8).cpu(), pk) and torch.equal(
            _unswizzle(W.w1_sf[e], w1i.shape[1], H).cpu(), sfr
        )
        pk, sfr = _quant_ref(w2[e])
        assert torch.equal(W.w2[e].view(torch.uint8).cpu(), pk) and torch.equal(
            _unswizzle(W.w2_sf[e], H, I).cpu(), sfr
        )
    counts, dest, src = permute_plan_padded(ids, E)
    counts_l = counts.tolist()
    n = T * k
    srcl = src.cpu().long()
    live = srcl >= 0
    assert int(live.sum()) == n and torch.equal(
        src[dest.view(-1).long()].cpu(), torch.arange(n, dtype=torch.int32)
    )
    tok = srcl.clamp_min(0) // k
    x_perm = torch.where(
        live.view(-1, 1).cuda(),
        xq.view(torch.uint8)[tok.cuda()],
        torch.zeros(1, dtype=torch.uint8, device="cuda"),
    )
    sf_perm = torch.where(
        live.view(-1, 1).cuda(),
        xsf.view(torch.uint8).reshape(T, -1)[tok.cuda()],
        torch.zeros(1, dtype=torch.uint8, device="cuda"),
    )
    off = [0]
    for c in counts_l:
        off.append(off[-1] + -(-c // 128) * 128)
    a = torch.zeros(src.shape[0], I, dtype=torch.bfloat16)
    for e, c in enumerate(counts_l):
        if not c:
            continue
        lo, hi = off[e], off[e] + c
        acc = mm_mxfp8_batch_invariant(
            x_perm[lo:hi].view(torch.float8_e4m3fn),
            W.w1[e].t(),
            _swizzle(sf_perm[lo:hi], H),
            W.w1_sf[e],
            out_dtype=torch.float32,
            in_fmt=fmt,
        ).cpu()
        if act.gated:
            u = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
            g = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
            a[lo:hi] = R.glu_cpu_act(g, u, act.act, act.params)
        else:
            a[lo:hi] = R.act_cpu(acc, act.act, act.params)
    y = torch.zeros(src.shape[0], H, dtype=torch.bfloat16)
    for e, c in enumerate(counts_l):
        if not c:
            continue
        lo, hi = off[e], off[e] + c
        pk, sfr = _quant_ref(
            a[lo:hi]
        )  # the quantizer class on the materialised bf16 rows (the declared second pass)
        y[lo:hi] = mm_mxfp8_batch_invariant(
            pk.view(torch.float8_e4m3fn).cuda(),
            W.w2[e].t(),
            _swizzle(sfr.cuda(), I),
            W.w2_sf[e],
            out_dtype=torch.bfloat16,
            in_fmt=fmt,
        ).cpu()
    return R.combine_cpu(y, dest.cpu(), weights.cpu())


def _fi_reference(xq, xsf, ids, weights, w1, w2, activation):
    """FlashInfer's dequantized MXFP8 MoE reference, as ``tests/moe/test_unified_moe_fp8.py::_block_fp8_reference`` computes it
    for the TRTLLM-Gen MXFP8 tests: the activation pack dequantized (``mxfp8_dequantize_host``), each expert's canonical weights
    quantized and dequantized through ``mxfp8_quantize(is_sf_swizzled_layout=False)``, fp32 GEMMs, the activation on the
    canonical (up, gate) halves, the intermediate requantized the same way, the routing weights narrowed to bf16, the slots
    accumulated per expert in fp32. SwiGLU / GeGLU / ReLU2 are that reference's own cases; the layer's other typed activations
    take their documented semantics (``spec.activation_of`` evaluated in fp32 by ``reference.glu_cpu_act`` / ``act_cpu``)."""
    import torch.nn.functional as F

    from flashinfer.fused_moe import GeGLU, ReLU2, SwiGLU
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from flashinfer.quantization.fp8_quantization import (
        mxfp8_dequantize_host,
        mxfp8_quantize,
    )

    def deq(q, sf):
        return mxfp8_dequantize_host(
            q.detach().cpu().view(torch.uint8),
            sf.detach().cpu().view(torch.uint8).reshape(-1),
            False,
        ).float()

    def qdq(w):
        q, sf = mxfp8_quantize(w, is_sf_swizzled_layout=False)
        return deq(q, sf.view(torch.uint8).reshape(w.shape[0], w.shape[1] // BLOCK))

    T, H = xq.shape
    I = w2.shape[2]
    x = deq(xq, xsf.view(torch.uint8).reshape(T, -1))
    spec = activation_of(activation)
    wts = weights.to(torch.bfloat16).float().cpu()
    ids = ids.cpu()
    out = torch.zeros(T, H, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        tok, slot = torch.where(ids == e)
        fc1 = x[tok] @ qdq(w1[e]).t()
        if isinstance(activation, ReLU2):
            act = F.relu(fc1) ** 2
        elif isinstance(activation, SwiGLU) and activation == SwiGLU():
            up, gate = fc1[:, :I], fc1[:, I:]
            act = F.silu(gate) * up
        elif isinstance(activation, GeGLU):
            up, gate = fc1[:, :I], fc1[:, I:]
            act = F.gelu(gate) * up
        elif spec.gated:
            up, gate = fc1[:, :I], fc1[:, I:]
            act = R.glu_cpu_act(gate, up, spec.act, spec.params).float()
        else:
            act = R.act_cpu(fc1, spec.act, spec.params).float()
        inter = qdq(act.to(torch.bfloat16).to(xq.device))
        out[tok] += wts[tok, slot, None] * (inter @ qdq(w2[e]).t())
    return out


def _assert_plausible(actual, expected):
    """FlashInfer's accuracy discipline for the MXFP8 MoE backends (``tests/moe/test_unified_moe_fp8.py::_assert_fp8_close``)."""
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    check_accuracy(
        expected.float(), actual.float().cpu(), atol=0.05, rtol=0.3, percent=0.99
    )


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU, the dense batch-invariant MXFP8 GEMM per expert with the
    CPU chain of the activation, the quantizer's reference on the intermediate, and the combine: the oracle ladder of the
    class, bit for bit, for every activation of the layer; and the op is plausible against FlashInfer's dequantized MXFP8 MoE
    reference under FlashInfer's own accuracy discipline for this quant pair."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack((xq, xsf), logits), _weights(w1, w2, act))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ref = _reference_ffn(xq, xsf, ids.cuda(), weights.cuda(), w1, w2, chain)
    assert torch.equal(bits(out.cpu()), bits(ref))
    _assert_plausible(out, _fi_reference(xq, xsf, ids, weights, w1, w2, act))


@needs_device
def test_activation_pack_is_the_quantizer_class_and_other_producers_packs_are_taken_as_given():
    """``prepare_activations`` is the batch-invariant MXFP8 quantizer (codes and token-major UE8M0 bytes == the class's CPU
    reference, == ``mxfp8_quantize(..., batch_invariant=True)``); the pack of another producer (FlashInfer's own MXFP8
    quantizer, the one the other MXFP8 backends' activation preparation calls) is taken as given: the layer's bits on it are
    the class's composition from those bytes."""
    import flashinfer
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 200, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11, adversarial=True)
    xq, xsf = _quant(x)
    pk, sfr = _quant_ref(x)
    assert (
        xq.dtype == torch.float8_e4m3fn
        and xsf.dtype == torch.uint8
        and tuple(xsf.shape) == (T, H // BLOCK)
    )
    assert torch.equal(xq.view(torch.uint8).cpu(), pk) and torch.equal(xsf.cpu(), sfr)
    q2, s2 = flashinfer.mxfp8_quantize(
        x, is_sf_swizzled_layout=False, batch_invariant=True
    )
    assert torch.equal(q2.view(torch.uint8), xq.view(torch.uint8)) and torch.equal(
        s2.view(torch.uint8).reshape(T, -1), xsf
    )
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    # another producer's pack: FlashInfer's own MXFP8 quantizer (the one the other MXFP8 backends' activation preparation calls)
    tq, tsf = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
    tsf8 = tsf.view(torch.uint8).reshape(T, -1)
    out = layer(_routed_pack((tq, tsf8), ids.cuda(), weights.cuda()), W)
    ref = _reference_ffn(
        tq, tsf8, ids.cuda(), weights.cuda(), w1, w2, activation_of(_SWIGLU)
    )
    assert torch.equal(bits(out.cpu()), bits(ref))
    # the UE8M0-typed scale tensor is the same pack
    out2 = layer(
        _routed_pack((tq, tsf8.view(torch.float8_e8m0fnu)), ids.cuda(), weights.cuda()),
        W,
    )
    assert torch.equal(bits(out2), bits(out))


@needs_device
def test_instruction_class_is_the_parts_and_the_grouped_kernel_is_the_dense_class():
    """The runner's instruction class is the part's (`mx_class_for`); the grouped kernel equals the dense batch-invariant
    ``mm_mxfp8`` per expert on the raw fp32 accumulators for several records (unaligned group starts, empty groups); where the
    part has both instructions (SM107) the two classes are distinct on the accumulators and both reachable by name."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        MoEWeightsMXFP8,
        grouped_gemm_mxfp8,
        mx_class_of,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.gemm.batch_invariant.generator import Knobs
    from flashinfer.gemm.batch_invariant.heuristic import family_of
    from flashinfer.gemm.batch_invariant.schedule import mx_class_for
    import flashinfer

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability(dev)
    fmt = mx_class_of(dev)
    assert (
        fmt
        == mx_class_for("mxfp8", family_of(cc))
        == ("mxfp8_k64" if cc == (10, 7) else "mxfp8")
    )
    layer = _layer(6, 64, 256)
    layer._ensure_built() if hasattr(layer, "_ensure_built") else None
    counts = [5, 0, 130, 1, 257, 64, 0, 3]
    E, N, K = len(counts), 256, 512
    Tk = sum(counts)
    x = (
        (
            torch.randn(Tk, K, generator=torch.Generator().manual_seed(3))
            * torch.exp2(
                torch.randint(
                    -6, 6, (Tk, K), generator=torch.Generator().manual_seed(4)
                ).float()
            )
        )
        .to(torch.bfloat16)
        .cuda()
    )
    xq, xsf = _quant(x)
    off = [0]
    for c in counts:
        off.append(off[-1] + -(-c // 128) * 128)
    Tk_pad = off[-1]
    a8 = torch.zeros(Tk_pad, K, dtype=torch.uint8, device="cuda")
    sfr = torch.zeros(Tk_pad, K // BLOCK, dtype=torch.uint8, device="cuda")
    pos = 0
    for e, c in enumerate(counts):
        a8[off[e] : off[e] + c] = xq.view(torch.uint8)[pos : pos + c]
        sfr[off[e] : off[e] + c] = xsf[pos : pos + c]
        pos += c
    a_sf = _swizzle(sfr, K)
    live = torch.zeros(Tk_pad, dtype=torch.bool)
    for e, c in enumerate(counts):
        live[off[e] : off[e] + c] = True
    g = torch.Generator().manual_seed(5)
    W = MoEWeightsMXFP8.prepare(
        (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        interleaved=True,
    )
    assert W.w1_sf.shape == (E, sf_numel(N, K))
    cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
    f32 = GroupedChain("plain", "silu", "f32")

    def dense(fmt_):
        ref = torch.zeros(Tk_pad, N, dtype=torch.float32, device="cuda")
        for e, c in enumerate(counts):
            if c:
                lo, hi = off[e], off[e] + c
                ref[lo:hi] = (
                    flashinfer.gemm.batch_invariant.api.mm_mxfp8_batch_invariant(
                        a8[lo:hi].view(torch.float8_e4m3fn),
                        W.w1[e].t(),
                        _swizzle(sfr[lo:hi], K),
                        W.w1_sf[e],
                        out_dtype=torch.float32,
                        in_fmt=fmt_,
                    )
                )
        return ref.cpu()

    fmts = ("mxfp8_k64", "mxfp8") if cc == (10, 7) else ("mxfp8",)
    accs = {}
    for f in fmts:
        ref = dense(f)
        for cfg in (
            Knobs((128, 128), (1, 1), False, True, "m", 1),
            Knobs((128, 64), (1, 2), False, True, "n", 1),
            Knobs((256, 128), (2, 1), True, True, "m", 1),
            Knobs((128, 192), (1, 1), False, True, "m", 1),
        ):
            got = grouped_gemm_mxfp8(
                a8.view(torch.float8_e4m3fn), a_sf, W.w1, W.w1_sf, cnt, cfg, f32, f
            ).cpu()
            assert torch.equal(bits(got[live]), bits(ref[live])), (f, cfg.describe())
        accs[f] = ref[live]
    if len(fmts) == 2:
        assert not torch.equal(bits(accs["mxfp8_k64"]), bits(accs["mxfp8"])), (
            "the two instruction classes agree on every accumulator: not two classes"
        )


@needs_device
def test_the_largest_expert_count_routes_and_runs():
    """1024 experts (the routing kernel's widest bucket, the group tables' size): the layer equals the class composition."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 96, 8, 256, 128, 1024
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    xq, xsf = _quant(x)
    layer = _layer(k, E, I)
    out = layer(_logits_pack((xq, xsf), logits), _weights(w1, w2))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ref = _reference_ffn(
        xq, xsf, ids.cuda(), weights.cuda(), w1, w2, activation_of(_SWIGLU)
    )
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
@pytest.mark.parametrize("T", [1, 5, 16, 17])
def test_layer_from_logits_at_small_token_counts_equals_the_prerouted_path(T):
    """Up to 16 tokens the runner routes and plans in one launch (``route_plan`` with the padded alignment); the layer's bits
    from logits equal the pre-routed path's on the routing's own ids and weights, below and above the threshold, and repeat."""
    k, H, I, E = 8, 512, 256, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11 + T, adversarial=True)
    xq, xsf = _quant(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    out_l = layer(_logits_pack((xq, xsf), logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack((xq, xsf), ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    assert torch.equal(bits(layer(_logits_pack((xq, xsf), logits), W)), bits(out_l))


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their
    inputs (the packed mode's bf16-rounded weights are its inputs); repeated runs and a CUDA-graph replay reproduce them."""
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2, adversarial=True)
    xq, xsf = _quant(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    out_l = layer(_logits_pack((xq, xsf), logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack((xq, xsf), ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    w16 = weights.to(torch.bfloat16)
    out_u16 = layer(_routed_pack((xq, xsf), ids.cuda(), w16.cuda()), W)
    out_p = layer(_routed_pack((xq, xsf), ids.cuda(), w16.cuda(), packed=True), W)
    assert torch.equal(bits(out_u16), bits(out_p))
    assert torch.equal(
        bits(out_u16),
        bits(layer(_routed_pack((xq, xsf), ids.cuda(), w16.float().cuda()), W)),
    )
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack((xq, xsf), logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack((xq, xsf), logits)
    with torch.cuda.stream(stream):
        layer(pack, W)  # the compile happens outside the capture
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
def test_batch_permutation_subbatch_and_hot_expert_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens
    create: prefixes, permutations, sub-batches, one token alone, a hot expert group around a probe token (its group spanning
    several 128-row atoms), an unrelated hot group (the probe's groups holding one row each)."""
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
    for n in (1, 2, 7, 64, 200):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(n))[:n].cuda()
        assert torch.equal(
            bits(layer(_logits_pack(x[idx].contiguous(), logits[idx].contiguous()), W)),
            full[idx],
        ), n
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
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits():
    """Every tactic the autotuner may pick (the dispatcher's ranked record pairs) gives the same bits; tuning the layer
    profiles them and the output before, during and after tuning is one output."""
    from flashinfer.autotuner import AutoTuner, autotune

    T, k, H, I, E = 300, 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=6, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantMxfp8Runner)
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert 1 <= len(tactics) <= 9 and len(set(tactics)) == len(tactics)
    for t in tactics:
        out = runner.forward(runner.pack_inputs(pack, W), tactic=t)
        assert torch.equal(bits(out), ref), t
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


@needs_device
def test_routing_methods_from_logits_match_the_cpu_model():
    """Every routing method the class implements, from logits, selects and weights as the exact CPU model, and the layer's
    bits follow (the pre-routed path on the CPU model's ids and weights)."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=7, adversarial=True)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(8)) * 0.1)
        .float()
        .cuda()
    )
    for method in (
        RoutingMethodType.Default,
        RoutingMethodType.Renormalize,
        RoutingMethodType.Sigmoid,
        RoutingMethodType.SigmoidRenorm,
        RoutingMethodType.DeepSeekV3,
    ):
        rm = S.METHOD_OF_FLASHINFER[method]
        kwargs = {"routed_scaling_factor": 1.75} if rm.scaled else {}
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=method, **kwargs),
                quant=_QUANT,
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
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
        ref = layer(_routed_pack(x, ex.cuda(), w.cuda()), W)
        assert torch.equal(bits(out), bits(ref)), method


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
@pytest.mark.parametrize(
    "requant,path,swap",
    [
        ("fused", "padded", False),
        ("standalone", "padded", False),
        ("fused", "dense", False),
        ("fused", "dense", True),
    ],
    ids=["fused", "standalone", "dense", "swapped"],
)
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "silu"])
def test_poisoned_intermediates_do_not_reach_the_bits(requant, path, swap, gated):
    """Unspecified bytes and timing never reach the output: the op's intermediates (``x_perm`` and its scales with their
    padding rows or tail rows, GEMM1's output or its fused codes and scales, GEMM2's output, the output itself) come from
    ``torch.empty``; with the caching allocator's blocks filled with NaN patterns before every call and a bandwidth-bound
    kernel running on a side stream of the same device during the call, both requantization spellings, both packings and
    both bodies (the padded body and the swapped decode body on the dense packing) give the bits of a clean run, each
    other's bits and the layer's, ``RLK_STRESS_ITERS`` times (default 20) at a shape whose GEMM2 K is one k-tile."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp8 as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import default_configs_mx

    T, k, H, I, E = 64, 8, 256, 128, 256
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=T + 3 * E, gated=gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = F.MoEWeightsMXFP8.prepare(w1, w2, gated=gated, act="silu" if gated else "relu2")
    ids, weights = route(logits, k)
    sm_count, family = F._part(xq.device)
    c1, c2 = default_configs_mx(
        T, k, H, I, E, W.chain1, F.mx_class_of(xq.device), sm_count, family, swap=swap
    )
    assert c1.swap_ab == c2.swap_ab == swap
    run = lambda: F.fused_moe_mxfp8(  # noqa: E731
        (xq, xsf), ids, weights, W, requant=requant, path=path, config1=c1, config2=c2
    )
    ref = bits(
        F.fused_moe_mxfp8((xq, xsf), ids, weights, W)
    )  # the dispatcher's picks and packing
    assert torch.equal(bits(run()), ref)
    side = torch.cuda.Stream()
    junk = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(int(os.environ.get("RLK_STRESS_ITERS", "20"))):
        for byte in (0x7F, 0xFF, 0x00):
            _poison_allocator(byte)
            with torch.cuda.stream(side):
                for _ in range(3):
                    junk.add_(1)
            out = run()
            assert torch.equal(bits(out), ref), (i, hex(byte))
    torch.cuda.synchronize()


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_fused_requantization_equals_the_standalone_spelling(act):
    """GEMM1's quantizing epilogue (the ``mxfp8.quant`` class applied to the bf16-rounded chain in place, codes and 128x4 block
    scales out) is a placement of the standalone pass: the codes, the scales, GEMM2's output and the op are bit-equal on the
    live rows for every typed activation; the op's default is the fused spelling."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp8 as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=3, gated=chain.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = F.MoEWeightsMXFP8.prepare(
        w1, w2, gated=chain.gated, act=chain.act, params=chain.params
    )
    ids, weights = route(logits, k)
    out_s, im_s = F.fused_moe_mxfp8(
        (xq, xsf), ids, weights, W, return_intermediates=True, requant="standalone"
    )
    out_f, im_f = F.fused_moe_mxfp8(
        (xq, xsf),
        ids,
        weights,
        W,
        return_intermediates=True,
        requant="fused",
        path="padded",
    )
    live = (im_s["src"] >= 0).cpu()
    Tk_pad = im_s["Tk_pad"]
    assert im_f["a"] is None and im_s["a"] is not None
    assert torch.equal(
        im_f["aq"].view(torch.uint8).cpu()[live],
        im_s["aq"].view(torch.uint8).cpu()[live],
    )
    assert torch.equal(
        _unswizzle(im_f["a_sf"], Tk_pad, I).cpu()[live],
        _unswizzle(im_s["a_sf"], Tk_pad, I).cpu()[live],
    )
    assert torch.equal(bits(im_f["y"].cpu())[live], bits(im_s["y"].cpu())[live])
    assert torch.equal(bits(out_f), bits(out_s))
    assert torch.equal(bits(F.fused_moe_mxfp8((xq, xsf), ids, weights, W)), bits(out_f))


@needs_device
def test_every_record_computes_the_class_at_long_k_on_every_instruction_class_of_the_part():
    """Every record of the grouped MXFP8 GEMM (the padded body's records at the deepest operand rings and the default, the
    swapped body's three token tiles) at a K of 16 and of 8 k-tiles per tile -- every stage of the deepest operand ring
    cycled -- computes the dense class per expert on this part, on every instruction class the part has: the guard for the
    operand rings' placement in the CTA's shared memory (``UMMA_SMEM_ADDR_MAX``: the M-side operand ring placed above it
    reads wrong data on the sm_100 instruction plans) -- the test that would catch the wrap: a ring stage above the bound
    is cycled by these K extents on every record."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        DENSE_TAIL,
        MoEWeightsMXFP8,
        grouped_gemm_mxfp8,
        mx_class_of,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        can_implement_grouped_mxfp8,
        can_implement_swapped_mxfp8,
        swapped_knobs_mx,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs
    import flashinfer

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability(dev)
    fmts = ("mxfp8_k64", "mxfp8") if cc == (10, 7) else (mx_class_of(dev),)
    f32 = GroupedChain("plain", "silu", "f32")
    padded = (
        Knobs((128, 128), (1, 1), False, True, "m", 1),
        Knobs((128, 64), (1, 2), False, True, "n", 1),
        Knobs((256, 64), (2, 1), True, True, "n", 1),
        Knobs((256, 128), (2, 1), True, True, "m", 1),
        Knobs((128, 256), (1, 1), False, True, "m", 1),
    )
    counts = [40, 1, 0, 130, 7, 64, 0, 3]
    E, Tk, N = len(counts), sum(counts), 256
    for K in (2048, 1024):
        g = torch.Generator().manual_seed(K)
        x = (
            torch.randn(Tk, K, generator=g)
            * torch.exp2(torch.randint(-6, 6, (Tk, K), generator=g).float())
        ).to(torch.bfloat16)
        xq, xsf = _quant(x.cuda())
        off, offp = [0], [0]
        for c in counts:
            off.append(off[-1] + c)
            offp.append(offp[-1] + -(-c // 128) * 128)
        Tk_pad = offp[-1]
        pos_pad = torch.cat(
            [torch.arange(offp[e], offp[e] + c) for e, c in enumerate(counts) if c]
        )
        a8 = torch.zeros(Tk_pad, K, dtype=torch.uint8, device="cuda")
        a8[pos_pad.cuda()] = xq.view(torch.uint8)
        sfr = torch.zeros(Tk_pad, K // BLOCK, dtype=torch.uint8, device="cuda")
        sfr[pos_pad.cuda()] = xsf
        a_sf = _swizzle(sfr, K)
        a_dense = torch.zeros(Tk + DENSE_TAIL, K, dtype=torch.uint8, device="cuda")
        a_dense[:Tk] = xq.view(torch.uint8)
        sf_pad_rows = torch.zeros(
            Tk_pad + DENSE_TAIL, K // BLOCK, dtype=torch.uint8, device="cuda"
        )
        sf_pad_rows[pos_pad.cuda()] = xsf
        sf_pad = _swizzle(sf_pad_rows, K)
        W = MoEWeightsMXFP8.prepare(
            (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            interleaved=True,
        )
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        for fmt in fmts:
            ref = torch.zeros(Tk, N, dtype=torch.float32, device="cuda")
            for e, c in enumerate(counts):
                if c:
                    lo, hi = off[e], off[e] + c
                    ref[lo:hi] = (
                        flashinfer.gemm.batch_invariant.api.mm_mxfp8_batch_invariant(
                            xq[lo:hi],
                            W.w1[e].t(),
                            _swizzle(xsf[lo:hi], K),
                            W.w1_sf[e],
                            out_dtype=torch.float32,
                            in_fmt=fmt,
                        )
                    )
            ref = ref.cpu()
            for cfg in padded:
                assert can_implement_grouped_mxfp8(cfg, N, K, E, f32, fmt), (
                    cfg.describe(),
                    fmt,
                )
                got = grouped_gemm_mxfp8(
                    a8.view(torch.float8_e4m3fn),
                    a_sf,
                    W.w1,
                    W.w1_sf,
                    cnt,
                    cfg,
                    f32,
                    fmt,
                ).cpu()[pos_pad]
                assert torch.equal(bits(got), bits(ref)), (
                    K,
                    cfg.describe(),
                    fmt,
                    (got != ref).sum().item(),
                )
            for cfg in swapped_knobs_mx():
                assert can_implement_swapped_mxfp8(cfg, N, K, E, f32, fmt), (
                    cfg.describe(),
                    fmt,
                )
                out32 = torch.full(
                    (Tk + DENSE_TAIL, N),
                    float("nan"),
                    dtype=torch.float32,
                    device="cuda",
                )
                grouped_gemm_mxfp8(
                    a_dense.view(torch.float8_e4m3fn),
                    sf_pad,
                    W.w1,
                    W.w1_sf,
                    cnt,
                    cfg,
                    f32,
                    fmt,
                    out=out32,
                    dense_a=True,
                    sf_rows=Tk_pad + DENSE_TAIL,
                )
                got = out32[:Tk].cpu()
                assert torch.equal(bits(got), bits(ref)), (
                    K,
                    cfg.describe(),
                    fmt,
                    (got != ref).sum().item(),
                )


@needs_device
def test_swapped_body_equals_the_dense_class_per_expert_at_every_token_tile():
    """The swapped decode body (the weights on the MMA's M side, the group's tokens an 8 / 16 / 32-row token tile of the dense
    packing, their scale factors gathered from the padded positions): its fp32 accumulators equal the dense batch-invariant
    ``mm_mxfp8`` per expert for both instruction classes where the part has both, the bf16 chains equal the padded body's,
    and the fused requantization in the transposed orientation (the block amax gathered across lanes, the gated chain's
    block across two warps) equals the quantizer class's CPU reference on the bf16-rounded chain: codes on the dense rows,
    scale bytes at the padded positions. Groups spanning several token tiles, one-row groups, empty groups, a K of one
    k-tile, N not a multiple of 128."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        DENSE_TAIL,
        MoEWeightsMXFP8,
        grouped_gemm_mxfp8,
        mx_class_of,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        SWAP_K_BLOCKS,
        can_implement_grouped_mxfp8,
        can_implement_swapped_mxfp8,
        swapped_knobs_mx,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    import flashinfer

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability(dev)
    fmts = ("mxfp8_k64", "mxfp8") if cc == (10, 7) else (mx_class_of(dev),)
    f32 = GroupedChain("plain", "silu", "f32")
    plain = GroupedChain("plain", "silu", "bf16")
    for counts, N, K in [
        ([5, 0, 70, 1, 33, 64, 0, 3], 256, 512),
        ([1] * 16, 192, 1024),
        ([0] * 200 + [40] + [0] * 55, 320, 256),
        ([3, 7, 11, 13] * 16, 1024, 128),
        ([1], 256, 2048),
    ]:
        E, Tk = len(counts), sum(counts)
        g = torch.Generator().manual_seed(E + 3)
        x = (
            torch.randn(Tk, K, generator=g)
            * torch.exp2(torch.randint(-6, 6, (Tk, K), generator=g).float())
        ).to(torch.bfloat16)
        xq, xsf = _quant(x.cuda())
        off = [0]
        offp = [0]
        for c in counts:
            off.append(off[-1] + c)
            offp.append(offp[-1] + -(-c // 128) * 128)
        Tk_pad = offp[-1]
        pos_pad = torch.cat(
            [torch.arange(offp[e], offp[e] + c) for e, c in enumerate(counts) if c]
        )
        a_dense = torch.zeros(Tk + DENSE_TAIL, K, dtype=torch.uint8, device="cuda")
        a_dense[:Tk] = xq.view(torch.uint8)
        sf_pad_rows = torch.zeros(
            Tk_pad + DENSE_TAIL, K // BLOCK, dtype=torch.uint8, device="cuda"
        )
        sf_pad_rows[pos_pad.cuda()] = xsf
        sf_pad = _swizzle(sf_pad_rows, K)
        sf_rows = Tk_pad + DENSE_TAIL
        W = MoEWeightsMXFP8.prepare(
            (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            interleaved=True,
        )
        Wu = MoEWeightsMXFP8.prepare(
            (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            (torch.randn(E, K, N, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            gated=False,
            act="relu2",
        )
        assert W.w1_sf.shape == (E, sf_numel(N, K))
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        for fmt in fmts:
            ref32 = torch.zeros(Tk, N, dtype=torch.float32, device="cuda")
            ref32u = torch.zeros(Tk, N, dtype=torch.float32, device="cuda")
            for e, c in enumerate(counts):
                if c:
                    lo, hi = off[e], off[e] + c
                    for ref, w in ((ref32, W), (ref32u, Wu)):
                        ref[lo:hi] = (
                            flashinfer.gemm.batch_invariant.api.mm_mxfp8_batch_invariant(
                                xq[lo:hi],
                                w.w1[e].t(),
                                _swizzle(xsf[lo:hi], K),
                                w.w1_sf[e],
                                out_dtype=torch.float32,
                                in_fmt=fmt,
                            )
                        )
            gg = lambda w, cfg, chain: grouped_gemm_mxfp8(  # noqa: E731
                a_dense.view(torch.float8_e4m3fn),
                sf_pad,
                w.w1,
                w.w1_sf,
                cnt,
                cfg,
                chain,
                fmt,
                dense_a=True,
                sf_rows=sf_rows,
            )
            # every swapped record at every k-block the body admits at this K (the two-atom stage is the rule's pick where it
            # divides K: `swap_k_block`; each is a placement of the same fold)
            for cfg, kb in [
                (c, kb)
                for c in swapped_knobs_mx()
                for kb in SWAP_K_BLOCKS
                if K % kb == 0
            ]:
                assert can_implement_swapped_mxfp8(
                    cfg, N, K, E, f32, fmt, 0 if kb == SWAP_K_BLOCKS[0] else kb
                ) and not can_implement_grouped_mxfp8(cfg, N, K, E, f32, fmt)
                out32 = torch.full(
                    (Tk + DENSE_TAIL, N),
                    float("nan"),
                    dtype=torch.float32,
                    device="cuda",
                )
                grouped_gemm_mxfp8(
                    a_dense.view(torch.float8_e4m3fn),
                    sf_pad,
                    W.w1,
                    W.w1_sf,
                    cnt,
                    cfg,
                    f32,
                    fmt,
                    out=out32,
                    dense_a=True,
                    sf_rows=sf_rows,
                    k_block=kb,
                )
                assert torch.equal(bits(out32[:Tk].cpu()), bits(ref32.cpu())), (
                    counts[:4],
                    cfg.describe(),
                    fmt,
                    kb,
                )
                assert torch.isnan(out32[Tk:]).all(), (
                    "rows past the live rows are never stored"
                )
                assert torch.equal(
                    bits(gg(W, cfg, plain)[:Tk].cpu()),
                    bits(ref32.cpu().to(torch.bfloat16)),
                )
                for act in (
                    SwiGLU(),
                    GeGLUTanh(),
                    SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
                ):
                    a = activation_of(act)
                    chain = GroupedChain("glu", a.act, "bf16", a.params)
                    ref_a = _reference_chain(ref32.cpu(), a)
                    assert torch.equal(
                        bits(gg(W, cfg, chain)[:Tk].cpu()), bits(ref_a)
                    ), (
                        counts[:4],
                        cfg.describe(),
                        a.act,
                    )
                    codes, sf = gg(W, cfg, MxChain("glu", a.act, "mxfp8", a.params))
                    pk, sfr = _quant_ref(ref_a)
                    assert torch.equal(codes.view(torch.uint8).cpu()[:Tk], pk), (
                        counts[:4],
                        cfg.describe(),
                        a.act,
                        "fused codes",
                    )
                    assert torch.equal(
                        _unswizzle(sf, sf_rows, N // 2).cpu()[pos_pad], sfr
                    ), (counts[:4], cfg.describe(), a.act, "fused scales")
                ref_u = _reference_chain(ref32u.cpu(), activation_of(ReLU2()))
                assert torch.equal(
                    bits(gg(Wu, cfg, GroupedChain("act", "relu2", "bf16"))[:Tk].cpu()),
                    bits(ref_u),
                )
                codes, sf = gg(Wu, cfg, MxChain("act", "relu2", "mxfp8"))
                pk, sfr = _quant_ref(ref_u)
                assert torch.equal(codes.view(torch.uint8).cpu()[:Tk], pk)
                assert torch.equal(_unswizzle(sf, sf_rows, N).cpu()[pos_pad], sfr)


def _reference_chain(acc: torch.Tensor, a) -> torch.Tensor:
    """The CPU spec of GEMM1's chain on the fp32 accumulator rows (the gated chains pair the interleaved columns)."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.reference import TorchF32

    acc = acc.float()
    if not a.gated:
        return S.act_elem(TorchF32, acc, a.act, "bf16", a.params)
    rows, N = acc.shape
    v = acc.view(rows, N // 16, 2, 8)
    u, g = v[:, :, 0, :].reshape(rows, N // 2), v[:, :, 1, :].reshape(rows, N // 2)
    return S.glu_elem(TorchF32, g, u, a.act, "bf16", a.params)


@needs_device
@pytest.mark.parametrize(
    "act", [SwiGLU(), GeGLUTanh(), ReLU2()], ids=["swiglu", "geglu_tanh", "relu2"]
)
@pytest.mark.parametrize(
    "T,k,H,I,E",
    [
        (1, 8, 512, 256, 128),
        (16, 6, 2048, 1024, 128),
        (64, 8, 256, 128, 256),
        (300, 6, 512, 256, 64),
    ],
)
def test_swapped_path_equals_the_padded_path(T, k, H, I, E, act):
    """The op on the dense packing with the swapped decode body on both GEMMs equals the op on the padded packing with the
    padded body bit for bit; the fused codes and scales and GEMM2's rows agree per expanded row; a mixed pair (a swapped
    GEMM1, a padded GEMM2 on the dense packing) agrees too; the dispatcher's own picks give the same bits whatever the
    packing it chose."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp8 as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import default_configs_mx
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    a = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=T + E + len(a.act), gated=a.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = F.MoEWeightsMXFP8.prepare(w1, w2, gated=a.gated, act=a.act, params=a.params)
    ids, weights = route(logits, k)
    sm_count, family = F._part(xq.device)
    fmt = F.mx_class_of(xq.device)
    p1, p2 = default_configs_mx(
        T, k, H, I, E, W.chain1, fmt, sm_count, family, swap=False
    )
    s1, s2 = default_configs_mx(
        T, k, H, I, E, W.chain1, fmt, sm_count, family, swap=True
    )
    ref, im_p = F.fused_moe_mxfp8(
        (xq, xsf), ids, weights, W, config1=p1, config2=p2, return_intermediates=True
    )
    got, im_s = F.fused_moe_mxfp8(
        (xq, xsf), ids, weights, W, config1=s1, config2=s2, return_intermediates=True
    )
    assert (
        im_p["path"] == "padded"
        and im_s["path"] == "dense"
        and im_s["swap"] == (True, True)
    )
    assert torch.equal(bits(got), bits(ref))
    dd, dp = im_s["dest"].reshape(-1).long(), im_s["dest_pad"].reshape(-1).long()
    assert torch.equal(
        im_s["aq"].view(torch.uint8)[dd], im_p["aq"].view(torch.uint8)[dp]
    )
    assert torch.equal(
        _unswizzle(im_s["a_sf"], im_s["Tk_sf"], I)[dp],
        _unswizzle(im_p["a_sf"], im_p["Tk_pad"], I)[dp],
    )
    assert torch.equal(bits(im_s["y"][dd].cpu()), bits(im_p["y"][dp].cpu()))
    mixed = F.fused_moe_mxfp8((xq, xsf), ids, weights, W, config1=s1, config2=p2)
    assert torch.equal(bits(mixed), bits(ref))
    dense_padded = F.fused_moe_mxfp8(
        (xq, xsf), ids, weights, W, config1=p1, config2=p2, path="dense"
    )
    assert torch.equal(bits(dense_padded), bits(ref))
    auto, im_a = F.fused_moe_mxfp8(
        (xq, xsf), ids, weights, W, return_intermediates=True
    )
    assert torch.equal(bits(auto), bits(ref)) and im_a["path"] in ("padded", "dense")
    with pytest.raises(ValueError, match="dense packing"):
        F.fused_moe_mxfp8(
            (xq, xsf), ids, weights, W, config1=s1, config2=s2, path="padded"
        )


@needs_device
def test_decode_tactics_include_the_swapped_records_and_agree():
    """At a decode-sized batch the ranked tactics include the swapped records (the swapped body on the dense packing, the
    layer's routing-and-plan launch in its dual form) and every tactic gives one output; the tactic round-trips its swapped
    flag; from logits and pre-routed agree."""
    from flashinfer.fused_moe.batch_invariant.runner import _configs_from_tactic

    T, k, H, I, E = 4, 8, 512, 256, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=41, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    assert len(inputs) == 14, "the dual plan: counts, dest, src, dest_pad, pos_pad"
    tactics = runner.get_valid_tactics(inputs, None)
    assert 1 <= len(tactics) <= 9 and len(set(tactics)) == len(tactics)
    assert all(
        _configs_from_tactic(t)[0].swap_ab or _configs_from_tactic(t)[1].swap_ab
        for t in tactics[:1]
    )
    for t in tactics:
        out = runner.forward(runner.pack_inputs(pack, W), tactic=t)
        assert torch.equal(bits(out), ref), t
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(layer(_routed_pack((_quant(x)), ids.cuda(), weights.cuda()), W)), ref
    )


@needs_device
def test_grouped_routing_config_takes_prerouted_ids_and_refuses_logits():
    """A group-limited routing configuration (``n_group`` / ``topk_group`` > 1) constructs; with a method other than
    DeepSeekV3 it is refused from logits (the groups are a term of the DeepSeekV3 method alone); with pre-routed ids the group
    structure is irrelevant and the output is the ungrouped configuration's bit for bit on the same ids and weights; the
    grouped DeepSeekV3 layer runs from logits deterministically, and its output is the pre-routed path's on the ids and
    weights of the bf16 model's grouped routing term (the routing kernel is the shared one; the GEMMs never see the groups)."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import METHOD_OF_FLASHINFER

    T, k, H, I, E = 64, 8, 512, 256, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5)
    xq, xsf = _quant(x)
    W = _weights(w1, w2)
    grouped = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.Renormalize,
                n_group=8,
                topk_group=4,
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
            execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=4096),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        grouped(_logits_pack((xq, xsf), logits), W)
    assert "n_group" in str(excinfo.value)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out = grouped(_routed_pack((xq, xsf), ids.cuda(), weights.cuda()), W)
    ref = _layer(k, E, I)(_routed_pack((xq, xsf), ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out), bits(ref))
    dsv3 = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.DeepSeekV3,
                n_group=8,
                topk_group=4,
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
            execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=4096),
        ),
        device=torch.device("cuda"),
    )
    bias = torch.randn(E, device="cuda")
    out_g = dsv3(_logits_pack((xq, xsf), logits, bias), W)
    assert out_g.shape == (T, H) and torch.equal(
        bits(out_g), bits(dsv3(_logits_pack((xq, xsf), logits, bias), W))
    )
    term = METHOD_OF_FLASHINFER[RoutingMethodType.DeepSeekV3].with_groups(8, 4)
    ids_g, weights_g = route(logits, k, term, bias, 1.0)
    assert torch.equal(
        bits(out_g), bits(dsv3(_routed_pack((xq, xsf), ids_g, weights_g), W))
    )


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the classes raises the mode's error at build or at the first call (top_k /
    num_experts outside the bounds, an intermediate size that is not a multiple of 32, fused shared experts, an expert-parallel
    shard, group-limited routing, no finalize, a bf16 activation without scales, a swizzled scale buffer, weights of the other
    gating, an unimplemented routing method from logits); a zero-token call returns an empty output."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    k, H, I, E = 6, 512, 256, 64
    errors = (
        BatchInvariantUnsupportedError,
        RuntimeError,
        ValueError,
        NotImplementedError,
    )
    oai = _layer(k, E, I, SwiGLU(alpha=1.702, beta=1.0, limit=7.0))
    assert isinstance(oai.runners[0], BatchInvariantMxfp8Runner)
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=200), "intermediate_size"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.mxfp8" in str(excinfo.value)
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
                    quant=_QUANT,
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.mxfp8" in str(excinfo.value), what
    # an expert-parallel shard (the layer's local expert range): the fp8-block family declares no expert parallelism, so the
    # layer's candidate filter and the runner's support check refuse the shard by name (the bf16 backend's base runner, which
    # computes shards, is not this runner's base)
    assert BatchInvariantMxfp8Runner.supports_expert_parallelism is False
    for shard in (
        dict(local_expert_offset=E // 2, local_num_experts=E // 2),
        dict(local_expert_offset=0, local_num_experts=E // 2),
    ):
        sharded = MoEConfig(
            routing=routing,
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I, **shard),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        )
        with pytest.raises(RuntimeError) as excinfo:
            MoELayer(sharded, device=torch.device("cuda"))
        assert "expert parallelism" in str(excinfo.value), shard
        # the class's own refusal fires first and names the shard (the shared assertion stands behind it)
        with pytest.raises(
            BatchInvariantUnsupportedError, match="expert-parallel shards"
        ):
            BatchInvariantMxfp8Runner(sharded, torch.device("cuda"))._check_support()
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=9)
    W = _weights(w1, w2)
    xq, xsf = _quant(x)
    # a bf16 activation without scales: no runner accepts the pack, nothing is computed
    with pytest.raises(errors):
        layer(
            MoEActivationPack(
                x,
                None,
                routing_input_mode=RoutingInputMode.FromLogits,
                routing_logits=logits,
            ),
            W,
        )
    # a swizzled scale buffer is not the activation pack
    import flashinfer

    _, sw = flashinfer.mxfp8_quantize(
        x, is_sf_swizzled_layout=True, batch_invariant=True
    )
    with pytest.raises(errors):
        layer(
            MoEActivationPack(
                xq,
                sw.view(torch.uint8),
                routing_input_mode=RoutingInputMode.FromLogits,
                routing_logits=logits,
            ),
            W,
        )
    # the ungated weights with the gated class: refused at the call, not computed
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(
            _logits_pack((xq, xsf), logits),
            _weights(w1[:, :I].contiguous(), w2, ReLU2()),
        )
    # MiniMax2 from logits is refused (another term); the same layer takes pre-routed inputs; Llama4 is the top-1 term
    minimax = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.MiniMax2
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(BatchInvariantUnsupportedError):
        minimax(_logits_pack((xq, xsf), logits), W)
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert minimax(
        _routed_pack((xq, xsf), ids, torch.ones(8, 1, device="cuda")), W
    ).shape == (8, H)
    # Llama4 routing (the top-1 of the logits with its sigmoid as the weight) is a routing term the bf16 model serves: the layer
    # runs from logits, deterministically; at any other top_k the class refuses it (TopKSigmoid is that term)
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    out_l = llama(_logits_pack((xq, xsf), logits), W)
    assert out_l.shape == (8, H) and torch.equal(
        bits(out_l), bits(llama(_logits_pack((xq, xsf), logits), W))
    )
    # a routing term the from-logits mode has no class for (Llama4 at top_k > 1) is refused at the from-logits call, not at
    # construction (the shared runner's rule: a layer that only receives pre-routed ids builds under any RoutingConfig)
    llama2 = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=2, method=RoutingMethodType.Llama4
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(errors) as excinfo:
        llama2(_logits_pack((xq, xsf), logits), W)
    assert "Llama4" in str(excinfo.value)
    ids2 = torch.stack(
        [torch.zeros(8, dtype=torch.int32), torch.ones(8, dtype=torch.int32)], 1
    ).cuda()
    assert llama2(
        _routed_pack((xq, xsf), ids2, torch.full((8, 2), 0.5, device="cuda")), W
    ).shape == (8, H)
    # weights on the CPU cannot be quantized by the class
    with pytest.raises(TypeError):
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1.cpu(),
            w2.cpu(),
            quant=_QUANT,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size=I,
        )
    # zero tokens
    out = layer(_logits_pack((xq[:0], xsf[:0]), logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


# ----------------------------------------------------------------------------- the shared runner's and entries' rules


def test_ranking_is_memoised_per_token_count():
    """The dispatcher's ranking (`schedule_mxfp8.rank_grouped_mx`) is memoised on its arguments: a forward without a tactic pays
    the ranking once per token count; a bf16 chain object is taken at the instruction class (the same key as its MX chain), a
    chain at another class is another key."""
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import (
        _rank_grouped_mx,
        rank_grouped_mx,
    )

    _rank_grouped_mx.cache_clear()
    chain = MxChain("glu", "silu", "bf16", (), "mxfp8_k64")
    args = (1536, 2048, 128, chain, "mxfp8_k64", 212, "sm_107", 3, False, 8)
    a = rank_grouped_mx(1, *args)
    assert a == rank_grouped_mx(1, *args)
    info = _rank_grouped_mx.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    rank_grouped_mx(2, *args)
    assert _rank_grouped_mx.cache_info().misses == 2
    bf16 = GroupedChain("glu", "silu", "bf16")
    assert rank_grouped_mx(1, 1536, 2048, 128, bf16, *args[4:]) == a
    assert _rank_grouped_mx.cache_info().hits == 2
    rank_grouped_mx(1, 1536, 2048, 128, bf16, "mxfp8", *args[5:])
    assert _rank_grouped_mx.cache_info().misses == 3
    assert all(c.swap_ab or c.tile_n in (64, 128, 192, 256) for c in a) and a


def test_grouped_gemm_shape_rules_are_host_errors():
    """The grouped MX GEMM entry checks the chain's shape rules on the host before any record or kernel is consulted: K in whole
    scale blocks, N at the weight-row rule, whole gate / up column pairs for a paired chain, an output width a multiple of the
    output format's row (a whole scale block for the fused requantization), a 16-byte-aligned output base. Device-free: the
    rules raise before the instruction class is looked up on a device."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import (
        MoEWeightsMXFP8,
        grouped_gemm_mxfp8,
        sf_numel,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.gemm.batch_invariant.generator import Knobs

    cfg = Knobs((128, 128), (1, 1), False, True, "m", 1)
    plain = GroupedChain("plain", "silu", "bf16")

    def operands(Tk, N, K, E=4):
        a = torch.zeros(Tk, K, dtype=torch.float8_e4m3fn)
        w = torch.zeros(E, N, K, dtype=torch.float8_e4m3fn)
        Kb = (
            -(-K // 32) * 32
        )  # scale buffers of a valid size: the operand rule is the one under test
        a_sf = torch.zeros(sf_numel(Tk, Kb), dtype=torch.uint8)
        w_sf = torch.zeros(E, sf_numel(-(-N // 8) * 8, Kb), dtype=torch.uint8)
        cnt = torch.full((E,), Tk // E, dtype=torch.int32)
        return a, a_sf, w, w_sf, cnt

    with pytest.raises(ValueError, match="K a multiple of 32"):
        grouped_gemm_mxfp8(*operands(128, 64, 48), cfg, plain, "mxfp8_k64")
    with pytest.raises(ValueError, match="N of 8"):
        grouped_gemm_mxfp8(*operands(128, 20, 64), cfg, plain, "mxfp8_k64")
    W = MoEWeightsMXFP8.__new__(MoEWeightsMXFP8)
    W.gated, W.act, W.params = True, "silu", ()
    with pytest.raises(ValueError, match="whole gate / up column pairs"):
        grouped_gemm_mxfp8(*operands(128, 24, 64), cfg, W.chain1, "mxfp8_k64")
    with pytest.raises(ValueError, match="output width of 16 mxfp8"):
        grouped_gemm_mxfp8(*operands(128, 32, 64), cfg, W.chain1_quant, "mxfp8_k64")
    with pytest.raises(ValueError, match="a multiple of the 128-row scale-factor atom"):
        grouped_gemm_mxfp8(*operands(96, 64, 64), cfg, plain, "mxfp8_k64")
    a, a_sf, w, w_sf, cnt = operands(128, 64, 64)
    buf = torch.empty(128, 64 + 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        grouped_gemm_mxfp8(
            a, a_sf, w, w_sf, cnt, cfg, plain, "mxfp8_k64", out=buf[:, 4:68]
        )
    with pytest.raises(ValueError, match="allocates its two outputs"):
        grouped_gemm_mxfp8(
            a, a_sf, w, w_sf, cnt, cfg, W.chain1_quant, "mxfp8_k64", out=buf[:, :64]
        )


@needs_device
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``: a framework's dropped slot) is dropped by the MXFP8 op
    as by the bf16 one: the plan does not count it (``dest -1``), the scatter and the two GEMMs never touch it, and the combine
    folds the remaining slots in slot order with the term ``+0`` in its place, so the output equals the class's composition on
    the same rows with those slots at a valid id and weight ``+0``; a token whose slots are all dropped gives a zero row; the
    other tokens keep their bits. Both packings (the dense packing of the swapped records, the padded packing) and both plan
    forms (the one-launch plan below ``PLAN_MULTI_MIN_N`` expanded rows and the two-launch form above it). With every slot
    dropped the GEMMs launch without a tile and the output is all zero rows, on the dense and the swapped records."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp8 as F
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N, permute_plan
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp8 import default_configs_mx
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    k, H, I, E = 8, 256, 128, 64
    bad = (-1, E, 1 << 30)
    a = activation_of(_SWIGLU)
    for T in (96, PLAN_MULTI_MIN_N // k + 3):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=21, adversarial=True)
        xq, xsf = _quant(x)
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
        for form in (
            {},
            dict(align=F.GROUP_ALIGN),
            dict(dual=True, pad_align=F.GROUP_ALIGN),
        ):
            counts, dest, src, *_ = permute_plan(ids_bad.cuda(), E, **form)
            assert int(counts.sum()) == n_valid, form
            assert torch.equal(dest.cpu().view(-1) < 0, dropped.view(-1)), form
            assert src[src >= 0].unique().numel() == n_valid, form
        out = layer(_routed_pack((xq, xsf), ids_bad.cuda(), wts.cuda()), W)
        ids_ref = torch.where(dropped, torch.zeros_like(ids), ids)
        wts_ref = torch.where(dropped, torch.zeros_like(wts), wts)
        ref = _reference_ffn(xq, xsf, ids_ref.cuda(), wts_ref.cuda(), w1, w2, a)
        assert torch.equal(bits(out.cpu()), bits(ref)), T
        assert torch.all(out[rows[0]] == 0)
        Wq = F.MoEWeightsMXFP8.from_view(W.get_view(KEY))
        for path in ("padded", "dense"):
            sm, fam = F._part(xq.device)
            c1, c2 = default_configs_mx(
                T,
                k,
                H,
                I,
                E,
                Wq.chain1,
                F.mx_class_of(xq.device),
                sm,
                fam,
                swap=None if path == "dense" else False,
            )
            out_api = F.fused_moe_mxfp8(
                (xq, xsf),
                ids_bad.cuda(),
                wts.cuda(),
                Wq,
                config1=c1,
                config2=c2,
                path=path,
            )
            assert torch.equal(bits(out_api), bits(out)), (T, path)
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack((xq, xsf), ids.cuda(), wts.cuda()), W)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))
    # every slot dropped: zero counts, the GEMMs launch without a tile, the zero row of every token, on both record kinds
    T, k, E = 37, 4, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=44)
    xq, xsf = _quant(x)
    Wq = F.MoEWeightsMXFP8.from_view(_weights(w1, w2).get_view(KEY))
    ids_none = torch.full((T, k), -1, dtype=torch.int32, device="cuda")
    wts_any = torch.rand(T, k, device="cuda")
    out, im = F.fused_moe_mxfp8(
        (xq, xsf), ids_none, wts_any, Wq, return_intermediates=True
    )
    torch.cuda.synchronize()
    assert torch.all(out == 0) and int(im["counts"].sum()) == 0
    assert torch.all(im["dest"] == -1)
    sm, fam = F._part(xq.device)
    for swap in (False, True):
        c1, c2 = default_configs_mx(
            T, k, H, I, E, Wq.chain1, F.mx_class_of(xq.device), sm, fam, swap=swap
        )
        assert c1.swap_ab == swap and c2.swap_ab == swap
        out2 = F.fused_moe_mxfp8(
            (xq, xsf), ids_none, wts_any, Wq, config1=c1, config2=c2
        )
        torch.cuda.synchronize()
        assert torch.all(out2 == 0), swap


@needs_device
@pytest.mark.parametrize(
    "act",
    [SwiGLU(), SwiGLU(alpha=1.702, beta=1.0, limit=7.0), ReLU2()],
    ids=["swiglu", "swiglu_oai", "relu2"],
)
def test_nan_activations_stay_in_their_rows(act):
    """An e4m3 NaN in a token's quantized row gives that token the composition's bits -- the NaN mask and the finite values
    (the requantization's block amax and saturation follow the device's non-NaN max / min rule the quantizer class states, so
    a NaN accumulator row leaves the chain as finite codes; what the kernel writes is what the composition writes) -- and
    leaves every other token's bits unchanged, the tokens sharing its experts included: a row's fold, its block amax and its
    requantization never read another row."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    a = activation_of(act)
    T, k, H, I, E = 40, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=77, gated=a.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    xq = xq.clone()
    W, layer = _weights(w1, w2, act), _layer(k, E, I, act)
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    clean = layer(_routed_pack((xq, xsf), ids, wts), W)
    rows = (3, 17, 39)
    xb = xq.view(torch.uint8)
    for r in rows:
        xb[r, (r * 7) % H] = 0x7F  # the e4m3 NaN byte
    out = layer(_routed_pack((xq, xsf), ids, wts), W)
    ref = _reference_ffn(xq, xsf, ids, wts, w1, w2, a)
    nan_out, nan_ref = torch.isnan(out.cpu()), torch.isnan(ref)
    assert torch.equal(nan_out, nan_ref)
    assert torch.equal(bits(out.cpu())[~nan_out], bits(ref)[~nan_out])
    for r in rows:  # the NaN moved its own row (the composition says where)
        assert not torch.equal(bits(out[r]), bits(clean[r])), r
    others = torch.tensor([t not in rows for t in range(T)], device="cuda")
    assert torch.equal(bits(out[others]), bits(clean[others]))
    assert not nan_out[others.cpu()].any()


@needs_device
def test_pre_routed_layers_build_under_any_routing_term():
    """A routing term the from-logits mode has no class for (``DeepSeekV3`` with more groups than the kernels hold, ``Llama4``
    with ``top_k > 1``) does not refuse the layer at construction: the pre-routed modes never consult it, so the layer runs
    pre-routed (unpacked and packed) with the bits of the plain layer on the same ids and weights, and refuses from logits with
    the term's limit (the shared runner's rule)."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    k, H, I, E = 6, 512, 256, 64
    x, logits, w1, w2 = _problem(16, k, H, I, E, seed=13, adversarial=True)
    xq, xsf = _quant(x)
    W = _weights(w1, w2)
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
                quant=_QUANT,
                experts=ExpertConfig(intermediate_size=I),
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=16),
            ),
            device=torch.device("cuda"),
        )
        plain = _layer(kk, E, I, tune_max=16)
        ids, weights = _reference_route(logits, kk, RoutingMethodType.Renormalize)
        ids, weights = ids.cuda(), weights.cuda()
        for packed in (False, True):
            assert torch.equal(
                bits(layer(_routed_pack((xq, xsf), ids, weights, packed=packed), W)),
                bits(plain(_routed_pack((xq, xsf), ids, weights, packed=packed), W)),
            ), (what, packed)
        with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
            layer(_logits_pack((xq, xsf), logits), W)
        assert what in str(excinfo.value), what


@needs_device
def test_packed_weights_round_in_the_combine_kernel():
    """The packed pre-routed mode's f32 weights enter the fold as their bf16 rounding (the packed word's value), rounded by
    the combine kernel: the bits equal the unpacked mode on the host-rounded weights, and differ from the unpacked mode on
    the unrounded f32 weights where the rounding moves a weight."""
    k, H, I, E = 4, 512, 256, 32
    x, logits, w1, w2 = _problem(24, k, H, I, E, seed=21, adversarial=True)
    xq, xsf = _quant(x)
    W = _weights(w1, w2)
    layer = _layer(k, E, I)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda().float()
    packed = layer(_routed_pack((xq, xsf), ids, weights, packed=True), W)
    rounded = layer(_routed_pack((xq, xsf), ids, weights.to(torch.bfloat16).float()), W)
    assert torch.equal(bits(packed), bits(rounded))
    assert torch.equal(
        bits(
            layer(
                _routed_pack((xq, xsf), ids, weights.to(torch.bfloat16), packed=True), W
            )
        ),
        bits(packed),
    )
    assert not torch.equal(weights, weights.to(torch.bfloat16).float())


@needs_device
def test_routing_constants_are_read_from_the_cache_under_graph_capture():
    """The route kernels' absent bias and unit scale are cached device constants (``api._const``): a call under CUDA graph
    capture reads the cached tensor when it exists (no fill node in the graph), a default first met inside a capture is a
    fill of that capture and is never cached, and a captured from-logits decode call of the MXFP8 layer replays the eager
    call's bits with no fill launch among its kernels."""
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.fused_moe.batch_invariant import api

    dev = torch.device("cuda", torch.cuda.current_device())
    k, H, I, E = 8, 512, 256, 64
    x, logits, w1, w2 = _problem(4, k, H, I, E, seed=31, adversarial=True)
    xq, xsf = _quant(x)
    W, layer = _weights(w1, w2), _layer(k, E, I, tune_max=4)
    pack = _logits_pack((xq, xsf), logits)
    eager = bits(layer(pack, W))  # the constants of this expert count are now cached
    b0, s0 = api._const(dev, E, 0.0), api._const(dev, 1, 1.0)
    assert api._const(dev, E, 0.0) is b0 and api._const(dev, 1, 1.0) is s0
    # the probe constant below must be unmet in this process: another suite's expert count may have cached it
    api._CONST.pop((dev.index, 3 * E, 0.0), None)
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
