"""The batch-invariant mixed MXFP4-weight x MXFP8-activation MoE backend of the unified MoE layer
(``BatchInvariantFp4Config`` with ``QuantConfig(weight=MXFP4, activation=MXFP8)``: the third pair of the fp4 backend, on its
runner and its weight preparation; the fp4 suite runs the backend's per-pair tests for this pair too, this file holds the
pair's own): the classes ``moe.mxfp4w8a.<act>``, one per typed activation of the layer (the bf16
classes' chains on the dequantised accumulators), the expert weights quantized by the batch-invariant MXFP4 quantizer and the
activations by the batch-invariant MXFP8 quantizer, the two GEMMs as grouped placements of the dense batch-invariant mixed
GEMM class of the part (SM107: the K = 64 instruction class; SM100 / SM103: the K = 32 one, whose TMA unpacks the e2m1 weight
and takes ``hidden_size`` / ``intermediate_size`` in multiples of 128). The quant pair TRTLLM-Gen serves as ``TrtllmFp4Config``
with ``QuantConfig(MXFP4, MXFP8)``. The class has exact CPU models of everything but the block-scaled step (the routing, the
chain on the fp32 accumulators, both quantizers, the slot-order combine), so the tests are bit comparisons: the layer's output
against the class's own composition (the routing on the CPU, the padded permutation, the dense batch-invariant
``mm_mxfp8_mxfp4`` per expert, the CPU chain, the MXFP8 quantizer's CPU reference on the intermediate, the dense mixed GEMM
again, the CPU combine) for every activation, run-to-run determinism including CUDA-graph replay, batch / permutation /
sub-batch / hot-expert invariance, agreement of every tactic the autotuner may pick, the three routing input modes giving the
same bits, the error contract (an unsupported configuration raises before any kernel runs), zero tokens, the largest expert
count, and the plausibility of the op against FlashInfer's dequantized MXFP4 x MXFP8 MoE reference (the one its TRTLLM-Gen fp4
tests use for this pair) under its accuracy discipline; no test compares against another backend's kernels. The device-free
tests pin the registration, the fingerprint dimensions (the instruction class of the part among them), the specification
terms (no scale symbol: the bf16 chains, the plain GEMM2 conversion), the shape rules (the 128-granule of the SM100 / SM103
class) and the dispatcher's membership in the enumerated space.
"""

from __future__ import annotations

import os

import pytest
import cutlass
import cutlass.cute as cute
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    BatchInvariantFp4Config,
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
from flashinfer.fused_moe.batch_invariant.api_fp4 import MXFP4W8A, MoEWeightsFp4
from flashinfer.fused_moe.batch_invariant.runner_fp4 import (
    MOE_FP4_ARCHS,
    BatchInvariantFp4Runner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_FP4_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
_QUANT = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
KEY = BatchInvariantFp4Runner.backend_key
BLOCK = 32
#: The record of the dense oracle calls (the dense class is pinned to one record: the oracle is the class, not its dispatcher;
#: the dense dispatcher's swapped decode records at a few rows take a 256-element stage, which the f32-output plan cannot ring)
_DENSE_RECORD = ((128, 128), (1, 1), False, True, "m", 1)

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


def _family() -> str:
    from flashinfer.gemm.batch_invariant.api import part_of

    return part_of(torch.device("cuda"))[1]


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
        backend=BackendOptions((BatchInvariantFp4Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1, w2, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        quant=_QUANT,
        activation=act,
    )
    pack.prepare_for(KEY, view)
    return pack


def _quant(x: torch.Tensor):
    """The canonical MXFP8 activation pack by the backend's quantizer."""
    return BatchInvariantFp4Config.prepare_activations(x, quant=_QUANT)


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
    """The fp4 config is a known backend of the layer, mapped to the fp4 runner, absent from the default candidates; the
    runner serves the MXFP4 x MXFP8 pair as its third pair (the pair descriptor: the mixed activation form, the block
    scales alone, no static class), with the three routing modes and every activation class of the layer; a pair the
    backend does not serve is refused naming the served ones."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.runner_fp4 import PAIRS

    assert BatchInvariantFp4Config in ALL_BACKEND_CONFIGS
    assert _BACKEND_RUNNERS[BatchInvariantFp4Config] is BatchInvariantFp4Runner
    assert not any(
        isinstance(c, BatchInvariantFp4Config) for c in _DEFAULT_BACKEND.candidates
    )
    assert PAIRS[_QUANT.pair] is MXFP4W8A
    assert MXFP4W8A.mixed and MXFP4W8A.act_fmt == "mxfp8" and not MXFP4W8A.scaled
    assert (MXFP4W8A.name, MXFP4W8A.cls, MXFP4W8A.block) == (
        "mxfp4w8a",
        "moe.mxfp4w8a",
        32,
    )
    assert MXFP4W8A.fmts == ("mxfp4w8a", "mxfp4w8a_k64")
    assert BatchInvariantFp4Runner.supported_quant_variants[-1] == _QUANT.pair
    assert BatchInvariantFp4Runner.supports_quant(_QUANT)
    assert set(
        BatchInvariantFp4Runner.supported_activation_classes_by_quant[_QUANT.pair]
    ) == {type(a) for a in ACTIVATIONS}
    assert KEY == "batch_invariant_fp4"
    w4a16 = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16)
    assert not BatchInvariantFp4Runner.supports_quant(w4a16)
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        BatchInvariantFp4Config.prepare_activations(
            torch.zeros(2, 64, dtype=torch.bfloat16), quant=w4a16
        )
    assert "MXFP4 x MXFP8" in str(excinfo.value) and "NVFP4 x NVFP4" in str(
        excinfo.value
    )


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_mxfp4w8a", ...)`` takes the geometry, the routing method, the activation and the architecture (the
    instruction class of the dense mixed GEMM: SM107 folds 64 elements per step, SM100 / SM103 32); the classes are distinct
    from the bf16 backend's and from each other; production fingerprints pinned."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.spec_mxfp4w8a import (
        MoEMxfp4w8aSpec,
        spec_fingerprint_mxfp4w8a,
    )

    fp = lambda **kw: bi.fingerprint(
        "fused_moe_mxfp4w8a", H=2048, I=768, top_k=8, num_experts=128, **kw
    )  # noqa: E731
    assert bi.class_name("fused_moe_mxfp4w8a") == "moe.mxfp4w8a"
    s107, s100 = fp(arch="sm_107"), fp(arch="sm_100")
    assert s107 == "moe-d052e19078f8166f" and s100 == "moe-c65c0fc11ce51102"
    assert (
        fp(arch="sm_103") == s100
        and fp(arch=(10, 7)) == s107
        and fp(arch=(10, 0)) == s100
    )
    assert fp(arch="sm_107", act="relu2") == "moe-98239c5cb16e1740"
    assert fp(arch="sm_100", act="relu2") == "moe-cca0c4d297a5dd48"
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
            "fused_moe_mxfp4w8a", H=2048, I=768, top_k=4, num_experts=128, arch="sm_107"
        ),
        bi.fingerprint(
            "fused_moe_mxfp4w8a", H=4096, I=768, top_k=8, num_experts=128, arch="sm_107"
        ),
        bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=128),
    }
    assert len(distinct) == 11
    assert s107 == spec_fingerprint_mxfp4w8a(
        2048, 768, 8, 128, S.SOFTMAX_RENORM, "silu", (), True, "mxfp4w8a_k64"
    )
    # the routing terms the bf16 model serves are classes of this backend too, under the runners' rules (Llama4 is the top-1
    # of the logits with its sigmoid as the weight: the receipt at top_k = 1 is TopKSigmoid's, another top_k has none); a
    # method the classes do not implement is refused
    with pytest.raises(ValueError, match="Llama4 routing takes top_k = 1"):
        fp(arch="sm_107", method=RoutingMethodType.Llama4)
    one = lambda m: bi.fingerprint(  # noqa: E731
        "fused_moe_mxfp4w8a",
        H=2048,
        I=768,
        top_k=1,
        num_experts=128,
        method=m,
        arch="sm_107",
    )
    assert one(RoutingMethodType.Llama4) == one("TopKSigmoid") not in distinct
    with pytest.raises(ValueError):
        fp(arch="sm_107", method=RoutingMethodType.MiniMax2)
    with pytest.raises(ValueError):
        fp(arch="sm_107", act="tanh")
    with pytest.raises(ValueError):
        fp(arch="sm_90")
    # the terms: GEMM1 is the bf16 class's chain (no alpha, per-token or static scale symbol), GEMM2 the plain conversion
    from flashinfer.fused_moe.batch_invariant import mterm as term

    q = MoEMxfp4w8aSpec(64, 32, 2, 8)
    b = S.MoESpec(64, 32, 2, 8)
    assert term.sexpr(q.gemm1()) == term.sexpr(b.gemm1()) and term.sexpr(
        q.combine()
    ) == term.sexpr(b.combine())
    assert term.sexpr(q.gemm2()) == "(cvt bf16 (acc2 0 0))"
    for act in ACTIVATIONS:
        a = S.activation_of(act)
        qa = MoEMxfp4w8aSpec(64, 32, 2, 8, act=a.act, params=a.params, gated=a.gated)
        ba = S.MoESpec(64, 32, 2, 8, act=a.act, params=a.params, gated=a.gated)
        g1 = term.sexpr(qa.gemm1())
        assert (
            g1 == term.sexpr(ba.gemm1())
            and "alpha" not in g1
            and "pts" not in g1
            and "sym" not in g1
        )
        assert qa.name == f"moe.mxfp4w8a.{a.act}"
    for other in (
        "nvfp4",
        "mxfp8",
        "mxfp8_k64",
    ):  # the MXFP8 classes are not instruction classes of this one
        with pytest.raises(AssertionError):
            MoEMxfp4w8aSpec(64, 32, 2, 8, mx_class=other)


def test_shape_rules_and_padded_bound():
    """The class's acceptance rule (H, I multiples of 32 on the SM107 class and of 128 on the SM100 / SM103 class; 1 <= E <= 1024;
    1 <= top_k <= min(E, 16); unsplit expert weights) raises the mode's error elsewhere; the padded row bound covers every
    routing without reading the device."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.api import padded_rows_bound
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import GROUP_ALIGN, sf_numel

    def check_shape_mxfp4w8a(H, I, E, k=None, fmt="mxfp4w8a_k64"):
        MXFP4W8A.check_shape(H, I, E, k, in_fmt=fmt)

    for H, I, E, k in (
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (32, 32, 1, 1),
        (2560, 640, 1024, 16),
    ):
        check_shape_mxfp4w8a(H, I, E, k, "mxfp4w8a_k64")
        if H % 128 == 0 and I % 128 == 0:
            check_shape_mxfp4w8a(H, I, E, k, "mxfp4w8a")
        else:
            with pytest.raises(BatchInvariantUnsupportedError, match="128"):
                check_shape_mxfp4w8a(H, I, E, k, "mxfp4w8a")
    for H, I in (
        (2048, 736),
        (2080, 768),
        (1056, 736),
    ):  # 32-aligned, not 128-aligned: the SM107 class only
        check_shape_mxfp4w8a(H, I, 128, 8, "mxfp4w8a_k64")
        with pytest.raises(BatchInvariantUnsupportedError, match="128"):
            check_shape_mxfp4w8a(H, I, 128, 8, "mxfp4w8a")
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
        with pytest.raises(BatchInvariantUnsupportedError, match="moe.mxfp4w8a"):
            check_shape_mxfp4w8a(*args)
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


@pytest.mark.parametrize(
    "family,sm_count", [("sm_100", 148), ("sm_103", 148), ("sm_107", 204)]
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The grouped mixed space: 72 dense block-scaled records at swizzle 1 (one list for both instruction classes, every record
    well-formed for both) and the swapped decode space (three narrow token tiles); the dispatcher's picks for both classes,
    every chain of the layer and a grid of token counts are members of the spaces the guarantee is claimed over (the
    fused-requantization chain GEMM1 runs and the bf16 chain accept the same records on either body). The ranking over both spaces
    takes the swapped decode body at one row per active expert on the large models (the small models' one-row cells go to the
    padded body, whose prologue is shorter) and the padded body once an expert fills a 128-row block; forbidden (``swap=False``),
    the padded candidates at a decode-sized group are the 1-CTA tiles in an N-cluster (the dense dispatcher's guard for a space
    without swapped records)."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        GEMM2_CHAIN,
        all_valid_configs_mx,
        all_valid_configs_mx_swapped,
        default_configs_mx,
        grouped_knobs_mx,
        rank_grouped_mx,
    )
    from flashinfer.fused_moe.batch_invariant.schedule import rows_per_expert
    from flashinfer.gemm.batch_invariant.heuristic import DECODE_M
    from flashinfer.gemm.batch_invariant.schedule import mx_class_for

    ks = grouped_knobs_mx("mxfp4w8a")
    assert len(ks) == 72 and ks == grouped_knobs_mx("mxfp4w8a_k64")
    for k in ks:
        assert (
            k.well_formed("mxfp4w8a")
            and k.well_formed("mxfp4w8a_k64")
            and k.swizzle_size == 1
        )
        assert (
            k.cluster_k == 1
            and not k.swap_ab
            and not k.b_reuse
            and not k.cta_chunks
            and k.tile_n in (64, 128, 192, 256)
        )
    fmt = mx_class_for("mxfp4w8a", family)
    assert fmt == ("mxfp4w8a_k64" if family == "sm_107" else "mxfp4w8a")
    chains = [
        GroupedChain(
            S.activation_of(a).chain,
            S.activation_of(a).act,
            "bf16",
            S.activation_of(a).params,
        )
        for a in ACTIVATIONS
    ]
    swapped_at_decode = False  # the swapped GEMM1 record is the pick at some one-row cell of the large shapes
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
            quant_chain = MxChain(chain.kind, chain.act, "mxfp8", chain.params, fmt)
            assert len(space1) == 72 and len(space2) == 72
            assert set(all_valid_configs_mx(n1, H, E, quant_chain, fmt)) == space1
            swap1 = set(all_valid_configs_mx_swapped(n1, H, E, chain, fmt))
            swap2 = set(all_valid_configs_mx_swapped(H, I, E, GEMM2_CHAIN, fmt))
            assert len(swap1) == 3 and len(swap2) == 3 and not (swap1 & space1)
            assert (
                set(all_valid_configs_mx_swapped(n1, H, E, quant_chain, fmt)) == swap1
            )
            for T in (1, 2, 7, 64, 300, 1024, 4096, 16384):
                rows = rows_per_expert(T, k, E)
                c1, c2 = default_configs_mx(
                    T, k, H, I, E, chain, fmt, sm_count, family, nominal=True
                )
                assert c1 in space1 | swap1 and c2 in space2 | swap2, (
                    family,
                    T,
                    H,
                    I,
                    E,
                    chain.describe(),
                    c1.describe(),
                    c2.describe(),
                )
                # the ranking over both spaces: the swapped decode body for GEMM1 at one row per active expert on the large
                # models (on the small ones, and for GEMM2 at the one-row cells generally, the padded body's shorter prologue
                # wins, as measured; the SM100 / SM103 class's swapped records have no two-atom stage, so its one-row cells
                # split by model), the padded body once an expert fills a 128-row block at the model-sized weights; in
                # between the model decides (every record computes the class)
                if rows <= 2 and H >= 4096 and c1.swap_ab:
                    swapped_at_decode = True
                if rows >= 128 and H >= 2048:
                    assert not c1.swap_ab and not c2.swap_ab, (
                        family,
                        T,
                        H,
                        I,
                        E,
                        c1.describe(),
                        c2.describe(),
                    )
                p1, p2 = default_configs_mx(
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
                assert p1 in space1 and p2 in space2
                s1, s2 = default_configs_mx(
                    T, k, H, I, E, chain, fmt, sm_count, family, nominal=True, swap=True
                )
                assert s1 in swap1 and s2 in swap2
                if rows <= DECODE_M:
                    # without the swapped body the candidates are the 1-CTA tiles in an N-cluster (the dense dispatcher's
                    # guard for such a space), never the dense class's swapped records read in the grouped orientation
                    for c in (p1, p2):
                        assert (
                            not c.use_2cta_instrs
                            and c.cluster_shape_mn[0] == 1
                            and c.tile_n >= 64
                        ), (family, T, H, I, E, c.describe())
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
                    and set(ranked) <= space1 | swap1
                )
    assert swapped_at_decode, family


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
    """The MXFP8 quantizer class's CPU reference on bf16 rows: ``(codes uint8 [M, K], sf [M, K / 32] uint8)``."""
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )

    pk, _, sfr = quantize_mx_reference(x.detach().cpu())
    return pk, sfr


def _quant4_ref(w: torch.Tensor):
    """The MXFP4 quantizer class's CPU reference on bf16 rows: ``(packed e2m1 codes uint8 [M, K / 2], sf [M, K / 32] uint8)``."""
    from flashinfer.batch_invariant.quantization import mx_spec
    from flashinfer.batch_invariant.quantization.mx_reference import (
        quantize_mx_reference,
    )

    pk, _, sfr = quantize_mx_reference(w.detach().cpu(), mx_spec.MXFP4)
    return pk, sfr


_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _deq_mxfp4(packed: torch.Tensor, sf_rows: torch.Tensor) -> torch.Tensor:
    """fp32 values of packed e2m1 codes (element ``2j`` in the low nibble) under UE8M0 row-block scales ``[M, K / 32]``."""
    q = packed.detach().cpu().view(torch.uint8).to(torch.int64)
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).reshape(q.shape[0], -1)
    vals = _E2M1_MAGNITUDES[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    scale = torch.exp2(sf_rows.detach().cpu().to(torch.float32) - 127.0)
    return (vals * scale.repeat_interleave(BLOCK, dim=1)).float()


def _reference_ffn(xq, xsf, ids, weights, w1, w2, act):
    """The class composed from its own pieces on the quantized activation pack: the padded permutation of the class, the
    scatter by indexing (the padding rows unspecified), GEMM1 as the dense batch-invariant mixed GEMM
    (``mm_mxfp8_mxfp4_batch_invariant``; the fp32 output the oracle needs, the instruction class pinned: the same fold) per
    expert, the CPU chain ``act`` on those accumulators, the MXFP8 quantizer's CPU reference on the bf16 intermediate, GEMM2 as
    the dense mixed GEMM (bf16 out) per expert, the CPU slot-order combine. Bit for bit what the fused op must produce."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import mx_class_of
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import permute_plan_padded
    from flashinfer.gemm.batch_invariant.api import mm_mxfp8_mxfp4_batch_invariant
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP
    from flashinfer.gemm.batch_invariant.generator import Knobs
    from flashinfer.gemm.batch_invariant.schedule import chunks_for

    T, H = xq.shape
    E, k = w1.shape[0], ids.shape[1]
    I = w2.shape[2]
    fmt = mx_class_of(xq.device)
    assert chunks_for(w1.shape[1], H, fmt) == 1 and chunks_for(H, I, fmt) == 1
    W = MoEWeightsFp4.prepare(
        MXFP4W8A, w1, w2, gated=act.gated, act=act.act, params=act.params
    )
    # the weights are the MXFP4 quantizer class per expert (the interleave before the quantization), packed two codes per byte
    w1i = interleave_fc1(w1) if act.gated else w1
    assert W.w1.dtype == torch.uint8 and tuple(W.w1.shape) == (E, w1i.shape[1], H // 2)
    for e in (0, E - 1):
        pk, sfr = _quant4_ref(w1i[e])
        assert torch.equal(W.w1[e].cpu(), pk) and torch.equal(
            _unswizzle(W.w1_sf[e], w1i.shape[1], H).cpu(), sfr
        )
        pk, sfr = _quant4_ref(w2[e])
        assert torch.equal(W.w2[e].cpu(), pk) and torch.equal(
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
        acc = mm_mxfp8_mxfp4_batch_invariant(
            x_perm[lo:hi].view(torch.float8_e4m3fn),
            W.w1[e].t(),
            _swizzle(sf_perm[lo:hi], H),
            W.w1_sf[e],
            out_dtype=torch.float32,
            in_fmt=fmt,
            config=Knobs(*_DENSE_RECORD),
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
        y[lo:hi] = mm_mxfp8_mxfp4_batch_invariant(
            pk.view(torch.float8_e4m3fn).cuda(),
            W.w2[e].t(),
            _swizzle(sfr.cuda(), I),
            W.w2_sf[e],
            out_dtype=torch.bfloat16,
            in_fmt=fmt,
            config=Knobs(*_DENSE_RECORD),
        ).cpu()
    return R.combine_cpu(y, dest.cpu(), weights.cpu())


def _fi_reference(xq, xsf, ids, weights, w1, w2, activation):
    """FlashInfer's dequantized MXFP4 x MXFP8 MoE reference, as ``tests/moe/trtllm_gen_fused_moe_utils.py::run_moe_reference_fp4``
    computes it for the TRTLLM-Gen fp4 tests at ``FP4_MXFP4_MXFP8``: the activation pack dequantized (``mxfp8_dequantize_host``),
    each expert's canonical weights quantized to MXFP4 by FlashInfer's quantizer (``fp4_quantize`` with unit global scale, 32-wide
    UE8M0 scales) and dequantized, fp32 GEMMs, the activation on the canonical (up, gate) halves, the intermediate requantized to
    MXFP8 through ``mxfp8_quantize``, the routing weights narrowed to bf16, the slots accumulated per expert in fp32. SwiGLU /
    GeGLU / ReLU2 are that reference's own cases; the layer's other typed activations take their documented semantics
    (``spec.activation_of`` evaluated in fp32 by ``reference.glu_cpu_act`` / ``act_cpu``)."""
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

    def qdq4(w):
        from flashinfer.fp4_quantization import fp4_quantize

        one = torch.ones(1, device=w.device, dtype=torch.float32)
        q, sf = fp4_quantize(
            w,
            global_scale=one,
            sf_vec_size=32,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )
        return _deq_mxfp4(
            q, sf.view(torch.uint8).reshape(w.shape[0], w.shape[1] // BLOCK)
        )

    T, H = xq.shape
    I = w2.shape[2]
    x = deq(xq, xsf.view(torch.uint8).reshape(T, -1))
    spec = activation_of(activation)
    wts = weights.to(torch.bfloat16).float().cpu()
    ids = ids.cpu()
    out = torch.zeros(T, H, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        tok, slot = torch.where(ids == e)
        fc1 = x[tok] @ qdq4(w1[e]).t()
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
        out[tok] += wts[tok, slot, None] * (inter @ qdq4(w2[e]).t())
    return out


def _assert_plausible(actual, expected):
    """FlashInfer's accuracy discipline for its fp4 MoE backends (the parameters ``tests/moe/trtllm_gen_fused_moe_utils.py``'s
    fp4 cases pass to ``check_accuracy``)."""
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    check_accuracy(
        expected.float(), actual.float().cpu(), atol=0.1, rtol=0.85, percent=0.925
    )


def test_grouped_records_round_trip_through_the_autotuners_tactic():
    """Every record of both spaces of both mixed classes survives the runner's flat tactic encoding (the autotuner's cache
    stores tactics as JSON): the tiler, the cluster, the 2-CTA flag, the raster, the swapped flag and the stage width are the
    record; the swizzle and the dense-only knobs are the grouped kernels' fixed defaults, so nothing is dropped. The swapped
    decode space: the three narrow token tiles at the dense class's stage width for the instruction class (256 elements on
    the SM107 class, the default on the SM100 / SM103 one), disjoint from the grouped space."""
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        swapped_knobs_mx,
    )
    from flashinfer.fused_moe.batch_invariant.runner_fp4 import (
        _configs_from_tactic_fp4,
        _tactic_fp4,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import grouped_knobs_mx

    for fmt in ("mxfp4w8a_k64", "mxfp4w8a"):
        ks = grouped_knobs_mx(fmt)
        sw = swapped_knobs_mx(fmt)
        assert len(ks) == 72 and len(sw) == 3 and not set(sw) & set(ks)
        for k in ks:
            assert k.k_block == 0 and k.swizzle_size == 1 and not k.swap_ab, (
                k.describe()
            )
        for k in sw:
            assert k.swap_ab and k.tile_m == 128 and k.tile_n in (8, 16, 32)
            assert k.k_block == (256 if fmt == "mxfp4w8a_k64" else 0), k.describe()
            assert k.well_formed(fmt)
        for k in ks + sw:
            assert _configs_from_tactic_fp4(_tactic_fp4(k, sw[0])) == (k, sw[0]), (
                k.describe()
            )
            assert _configs_from_tactic_fp4(_tactic_fp4(ks[0], k)) == (ks[0], k), (
                k.describe()
            )


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU, the dense batch-invariant mixed GEMM per expert with the
    CPU chain of the activation, the quantizer's reference on the intermediate, and the combine: the oracle ladder of the
    class, bit for bit, for every activation of the layer; and the op is plausible against FlashInfer's dequantized MXFP4 x MXFP8 MoE
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
    ``mm_mxfp8_mxfp4`` per expert on the raw fp32 accumulators for several records (unaligned group starts, empty groups);
    where the part has both instructions (SM107) the two classes are distinct on the accumulators and both reachable by name."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import (
        grouped_gemm_mxfp4w8a,
        mx_class_of,
    )
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import sf_numel
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
        == mx_class_for("mxfp4w8a", family_of(cc))
        == ("mxfp4w8a_k64" if cc == (10, 7) else "mxfp4w8a")
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
    W = MoEWeightsFp4.prepare(
        MXFP4W8A,
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
                    flashinfer.gemm.batch_invariant.api.mm_mxfp8_mxfp4_batch_invariant(
                        a8[lo:hi].view(torch.float8_e4m3fn),
                        W.w1[e].t(),
                        _swizzle(sfr[lo:hi], K),
                        W.w1_sf[e],
                        out_dtype=torch.float32,
                        in_fmt=fmt_,
                        config=Knobs(*_DENSE_RECORD),
                    )
                )
        return ref.cpu()

    fmts = ("mxfp4w8a_k64", "mxfp4w8a") if cc == (10, 7) else ("mxfp4w8a",)
    accs = {}
    for f in fmts:
        ref = dense(f)
        for cfg in (
            Knobs((128, 128), (1, 1), False, True, "m", 1),
            Knobs((128, 64), (1, 2), False, True, "n", 1),
            Knobs((256, 128), (2, 1), True, True, "m", 1),
            Knobs((128, 192), (1, 1), False, True, "m", 1),
        ):
            got = grouped_gemm_mxfp4w8a(
                a8.view(torch.float8_e4m3fn), a_sf, W.w1, W.w1_sf, cnt, cfg, f32, f
            ).cpu()
            assert torch.equal(bits(got[live]), bits(ref[live])), (f, cfg.describe())
        accs[f] = ref[live]
    if len(fmts) == 2:
        assert not torch.equal(bits(accs["mxfp4w8a_k64"]), bits(accs["mxfp4w8a"])), (
            "the two instruction classes agree on every accumulator: not two classes"
        )


def test_launch_tile_bound_dominates_the_scheduler_total_over_the_mixed_spaces():
    """The persistent grid's clip on the mixed class's launches: the static bound on a launch's cluster tiles is at least
    the grouped scheduler's total -- the sum over experts of ceil(count / cluster rows), times the N tiles -- for every
    record of both record spaces (the 72 grouped records, the swapped decode records at both stage widths) over random
    count vectors (empty experts, one-row experts, one expert holding everything), and tight at one row per active
    expert."""
    import random

    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        swapped_knobs_mx,
    )
    from flashinfer.fused_moe.batch_invariant.schedule import launch_tile_bound
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import grouped_knobs_mx

    rng = random.Random(5)
    records = (
        grouped_knobs_mx()
        + swapped_knobs_mx("mxfp4w8a_k64")
        + swapped_knobs_mx("mxfp4w8a")
    )
    for E, Tk, N in [
        (8, 2, 4096),
        (128, 8, 1536),
        (128, 128, 1536),
        (256, 2048, 7168),
        (64, 64, 512),
        (1024, 4096, 768),
    ]:
        vectors = [[Tk] + [0] * (E - 1), [1] * min(E, Tk) + [0] * max(0, E - Tk)]
        if Tk >= E:
            vectors[1] = [Tk // E + (1 if i < Tk % E else 0) for i in range(E)]
        for _ in range(6):
            cuts = sorted(rng.randrange(Tk + 1) for _ in range(E - 1))
            vectors.append(
                [b - a for a, b in zip([0] + cuts, cuts + [Tk], strict=True)]
            )
        for cfg in records:
            if cfg.swap_ab:
                cluster_rows, tn, cn = cfg.tile_n, cfg.mma_tiler_mn[0], 1
            else:
                cta_m = cfg.mma_tiler_mn[0] // (2 if cfg.use_2cta_instrs else 1)
                cluster_rows, tn, cn = (
                    cta_m * cfg.cluster_shape_mn[0],
                    cfg.mma_tiler_mn[1],
                    cfg.cluster_shape_mn[1],
                )
            n_tiles = -(-N // (tn * cn))
            bound = launch_tile_bound(cfg, Tk, E, N)
            for counts in vectors:
                total = sum(-(-c // cluster_rows) for c in counts) * n_tiles
                assert total <= bound, (
                    cfg.describe(),
                    E,
                    Tk,
                    N,
                    counts[:8],
                    total,
                    bound,
                )
            if Tk <= E:
                assert bound == Tk * n_tiles, (cfg.describe(), E, Tk, N, bound)


@needs_device
def test_swapped_body_equals_the_dense_class_per_expert():
    """The swapped decode body on the twin plan of the mixed class (the e2m1 weight as the MMA's A -- through the TMA's
    unpacking container on the SM100 class, packed on the SM107 one -- the group's tokens as a narrow N tile from the dense
    packing, their scale factors gathered from the padded positions): at every token tile of its space its fp32 accumulators
    equal the dense batch-invariant ``mm_mxfp8_mxfp4`` per expert on the group's rows, the GEMM1 chain and GEMM2's conversion
    equal the grouped body's, rows past the live rows are never stored, and the fused requantization's codes and block
    scales are the ``mxfp8.quant`` class on the grouped body's bf16 chain per live row (the codes on the dense rows, the
    scale bytes at the padded positions); both instruction classes where the part has both (the SM107 class at its
    256-element stage, the SM100 class at the default)."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import grouped_gemm_mxfp4w8a
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import DENSE_TAIL
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        can_implement_grouped_mxfp4w8a,
        can_implement_swapped_mxfp4w8a,
        swapped_knobs_mx,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability(dev)
    fmts = ("mxfp4w8a_k64", "mxfp4w8a") if cc == (10, 7) else ("mxfp4w8a",)
    f32 = GroupedChain("plain", "silu", "f32")
    swiglu = GroupedChain("glu", "silu", "bf16")
    padded = Knobs(*_DENSE_RECORD)
    for counts, N, K in (
        ([5, 0, 70, 1, 33, 64, 0, 3], 256, 512),
        ([1] * 16, 256, 1024),
        ([1], 256, 2048),
        ([3, 7, 11, 13] * 8, 512, 128),
    ):
        E, Tk = len(counts), sum(counts)
        a8, sfr, live, off, Tk_pad = _padded_operand(counts, K, seed=E + 7)
        a_sf = _swizzle(sfr, K)
        a_dense, sf_dense, pos_pad = _dense_operand(a8, sfr, live, DENSE_TAIL)
        sf_rows = Tk_pad + DENSE_TAIL
        g = torch.Generator().manual_seed(E + 7)
        W = MoEWeightsFp4.prepare(
            MXFP4W8A,
            (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
            interleaved=True,
        )
        cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
        for fmt in fmts:
            ref = _dense_per_expert(a8, sfr, W, counts, off, fmt)[live]
            a_pad = grouped_gemm_mxfp4w8a(
                a8.view(torch.float8_e4m3fn),
                a_sf,
                W.w1,
                W.w1_sf,
                cnt,
                padded,
                swiglu,
                fmt,
            ).cpu()[live]
            gg = lambda cfg, chain, **kw: grouped_gemm_mxfp4w8a(  # noqa: E731
                a_dense.view(torch.float8_e4m3fn),
                sf_dense,
                W.w1,
                W.w1_sf,
                cnt,
                cfg,
                chain,
                fmt,
                dense_a=True,
                sf_rows=sf_rows,
                **kw,
            )
            for cfg in swapped_knobs_mx(fmt):
                assert can_implement_swapped_mxfp4w8a(cfg, N, K, E, f32, fmt)
                assert not can_implement_grouped_mxfp4w8a(cfg, N, K, E, f32, fmt)
                out = torch.full(
                    (Tk + DENSE_TAIL, N),
                    float("nan"),
                    dtype=torch.float32,
                    device="cuda",
                )
                gg(cfg, f32, out=out)
                got = out.cpu()
                assert torch.equal(bits(got[:Tk]), bits(ref)), (
                    fmt,
                    counts[:4],
                    cfg.describe(),
                )
                assert torch.isnan(got[Tk:]).all(), (
                    "rows past the live rows are never stored"
                )
                got1 = gg(cfg, swiglu).cpu()[:Tk]
                assert torch.equal(bits(got1), bits(a_pad)), (
                    fmt,
                    cfg.describe(),
                    "swiglu",
                )
                codes, sf = gg(cfg, MxChain("glu", "silu", "mxfp8", (), fmt))
                pk, sfr_q = _quant_ref(a_pad)
                assert torch.equal(codes.view(torch.uint8).cpu()[:Tk], pk), (
                    fmt,
                    cfg.describe(),
                    "fused codes",
                )
                assert torch.equal(
                    _unswizzle(sf, sf_rows, N // 2).cpu()[pos_pad], sfr_q
                ), (fmt, cfg.describe(), "fused scales")


@needs_device
def _padded_operand(counts, K, seed):
    """A padded e4m3 operand (128-row atoms per expert) of ``counts`` adversarial token rows: ``(a8, sfr, live, off, Tk_pad)``."""
    Tk = sum(counts)
    g = torch.Generator().manual_seed(seed)
    x = (
        (
            torch.randn(Tk, K, generator=g)
            * torch.exp2(torch.randint(-6, 6, (Tk, K), generator=g).float())
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
    live = torch.zeros(Tk_pad, dtype=torch.bool)
    pos = 0
    for e, c in enumerate(counts):
        a8[off[e] : off[e] + c] = xq.view(torch.uint8)[pos : pos + c]
        sfr[off[e] : off[e] + c] = xsf[pos : pos + c]
        live[off[e] : off[e] + c] = True
        pos += c
    return a8, sfr, live, off, Tk_pad


def _dense_operand(a8, sfr, live, tail):
    """The dense packing of a padded operand: ``(a_dense [Tk + tail, K], sf_dense (the scale factors at the padded positions,
    the buffer carrying ``tail`` rows past the padded rows), pos_pad [Tk])`` -- the swapped body's operand."""
    K = a8.shape[1]
    pos_pad = torch.nonzero(live).view(-1)
    Tk = int(pos_pad.numel())
    a_dense = torch.zeros(Tk + tail, K, dtype=torch.uint8, device="cuda")
    a_dense[:Tk] = a8[pos_pad.cuda()]
    sf_rows = torch.zeros(
        a8.shape[0] + tail, K // BLOCK, dtype=torch.uint8, device="cuda"
    )
    sf_rows[: a8.shape[0]] = sfr
    return a_dense, _swizzle(sf_rows, K), pos_pad


def _dense_per_expert(a8, sfr, W, counts, off, fmt):
    """The dense batch-invariant mixed class's fp32 accumulators per expert on its live rows (the padded rows zero)."""
    import flashinfer

    from flashinfer.gemm.batch_invariant.generator import Knobs

    K, N = a8.shape[1], W.w1.shape[1]
    ref = torch.zeros(a8.shape[0], N, dtype=torch.float32, device="cuda")
    for e, c in enumerate(counts):
        if c:
            lo, hi = off[e], off[e] + c
            ref[lo:hi] = (
                flashinfer.gemm.batch_invariant.api.mm_mxfp8_mxfp4_batch_invariant(
                    a8[lo:hi].view(torch.float8_e4m3fn),
                    W.w1[e].t(),
                    _swizzle(sfr[lo:hi], K),
                    W.w1_sf[e],
                    out_dtype=torch.float32,
                    in_fmt=fmt,
                    config=Knobs(*_DENSE_RECORD),
                )
            )
    return ref.cpu()


@needs_device
def test_swapped_body_rings_stay_inside_the_plans_address_bound():
    """The swapped body's shared memory on the SM100 instruction plan (``mxfp4w8a``'s twin, the plan with a bound:
    ``UMMA_SMEM_ADDR_MAX``; the SM107 plan has none): the unified operand block's plan keeps the barriers, the group tables,
    the exchange buffer and the operand rings the instruction reads (at the smem container's width: the unpacked e2m1 weight
    occupies a byte per element) inside that bound on every record of the swapped space, and the 8-token record at the
    default stage width -- the one whose unbounded ring would cross the bound -- computes the class at a K that cycles every
    stage of its ring, bit for bit the dense mixed class's accumulators per expert."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import grouped_gemm_mxfp4w8a
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import DENSE_TAIL
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import (
        MxChain,
        swap_xch_bytes,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.operands import (
        BlockScaledOperands,
        xch_smem_bytes,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import swapped_knobs_mx
    from flashinfer.gemm.batch_invariant.blocks.operands import MBAR_RESERVE_BYTES
    from flashinfer.gemm.batch_invariant.generator import Knobs

    fmt = "mxfp4w8a"
    ops = BlockScaledOperands(fmt, "dense")
    plan_cls = ops.plan_cls_for("swapped")
    assert ops.mixed and plan_cls is not ops.plan_cls
    assert plan_cls.A_SMEM_DT is not None and plan_cls.A_TMA_INTERNAL is not None
    assert (
        plan_cls.UMMA_SMEM_ADDR_MAX is not None
        and BlockScaledOperands("mxfp4w8a_k64", "dense")
        .plan_cls_for("swapped")
        .UMMA_SMEM_ADDR_MAX
        is None
    )
    quant = MxChain("glu", "silu", "mxfp8", (), fmt)
    records = swapped_knobs_mx(fmt) + [
        Knobs((128, tn), (1, 1), False, True, "m", 1, swap_ab=True, k_block=256)
        for tn in (8, 16, 32)
    ]
    seen = []

    @cute.jit
    def probe(x: cute.Tensor):
        for cfg in records:
            xch = quant.smem_layout(1, token_tile=cfg.tile_n)
            own = ops.table_bytes + xch_smem_bytes(xch)
            assert xch_smem_bytes(xch) == swap_xch_bytes(quant, cfg.tile_n)
            st = ops.plan(cfg, lambda st: own, "swapped", plan_cls.UMMA_SMEM_ADDR_MAX)
            a_one = cute.slice_(st.a_smem_layout_staged, (None, None, None, 0))
            b_one = cute.slice_(st.b_smem_layout_staged, (None, None, None, 0))
            ring = (
                cute.size_in_bytes(st.a_smem_dtype, a_one)
                + cute.size_in_bytes(st.b_smem_dtype, b_one)
            ) * st.num_ab_stage
            seen.append(
                (cfg.describe(), ring + own + MBAR_RESERVE_BYTES, st.num_ab_stage)
            )

    cute.compile(probe, cute.runtime.make_fake_tensor(cutlass.Float32, (8,), (1,)))
    assert len(seen) == len(records)
    for name, footprint, stages in seen:
        assert footprint <= plan_cls.UMMA_SMEM_ADDR_MAX and stages >= 2, (
            name,
            footprint,
            stages,
        )
    # the 8-token record at the default stage width over a K that cycles every stage of the ring: the dense class's bits
    counts, N, K = [3, 0, 70, 1, 33], 256, 4096
    E, Tk = len(counts), sum(counts)
    a8, sfr, live, off, Tk_pad = _padded_operand(counts, K, seed=29)
    a_dense, sf_dense, _ = _dense_operand(a8, sfr, live, DENSE_TAIL)
    g = torch.Generator().manual_seed(29)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A,
        (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        interleaved=True,
    )
    ref = _dense_per_expert(a8, sfr, W, counts, off, fmt)[live]
    cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
    got = grouped_gemm_mxfp4w8a(
        a_dense.view(torch.float8_e4m3fn),
        sf_dense,
        W.w1,
        W.w1_sf,
        cnt,
        records[0],
        GroupedChain("plain", "silu", "f32"),
        fmt,
        dense_a=True,
        sf_rows=Tk_pad + DENSE_TAIL,
    )
    assert torch.equal(bits(got.cpu()[:Tk]), bits(ref)), records[0].describe()


@needs_device
def test_every_record_computes_the_class_at_long_k():
    """The long-K guard: every swapped record (on the dense packing) and the deepest grouped records (on the padded one) at a K
    that cycles every stage of their operand rings, on every instruction class of the part, bit-equal to the dense mixed
    class per expert (a ring placed where the instruction reads other bytes would show here first)."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import grouped_gemm_mxfp4w8a
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import DENSE_TAIL
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import swapped_knobs_mx
    from flashinfer.gemm.batch_invariant.generator import Knobs

    cc = torch.cuda.get_device_capability(torch.device("cuda"))
    fmts = ("mxfp4w8a_k64", "mxfp4w8a") if cc == (10, 7) else ("mxfp4w8a",)
    f32 = GroupedChain("plain", "silu", "f32")
    counts, N, K = [3, 0, 70, 1, 33], 512, 8192
    E, Tk = len(counts), sum(counts)
    a8, sfr, live, off, Tk_pad = _padded_operand(counts, K, seed=31)
    a_dense, sf_dense, _ = _dense_operand(a8, sfr, live, DENSE_TAIL)
    g = torch.Generator().manual_seed(31)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A,
        (torch.randn(E, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        (torch.randn(E, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        interleaved=True,
    )
    cnt = torch.tensor(counts, dtype=torch.int32, device="cuda")
    deepest = [
        Knobs((128, 64), (1, 1), False, True, "m", 1),
        Knobs((128, 64), (1, 2), False, True, "n", 1),
        Knobs((128, 128), (1, 1), False, True, "m", 1),
    ]
    for fmt in fmts:
        ref = _dense_per_expert(a8, sfr, W, counts, off, fmt)[live]
        for cfg in deepest:
            got = grouped_gemm_mxfp4w8a(
                a8.view(torch.float8_e4m3fn),
                _swizzle(sfr, K),
                W.w1,
                W.w1_sf,
                cnt,
                cfg,
                f32,
                fmt,
            ).cpu()[live]
            assert torch.equal(bits(got), bits(ref)), (
                fmt,
                cfg.describe(),
                (got != ref).sum().item(),
            )
        for cfg in swapped_knobs_mx(fmt):
            got = grouped_gemm_mxfp4w8a(
                a_dense.view(torch.float8_e4m3fn),
                sf_dense,
                W.w1,
                W.w1_sf,
                cnt,
                cfg,
                f32,
                fmt,
                dense_a=True,
                sf_rows=Tk_pad + DENSE_TAIL,
            ).cpu()[:Tk]
            assert torch.equal(bits(got), bits(ref)), (
                fmt,
                cfg.describe(),
                (got != ref).sum().item(),
            )


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_swapped_op_equals_the_padded_op(act):
    """At a decode-sized group the op with the swapped decode body on both GEMMs (the fused requantization in its transposed
    epilogue, or the standalone pass) equals the op with the padded body (either spelling) bit for bit, the default dispatch
    (the ranking's pick per GEMM: on this small shape the padded body's shorter prologue wins the one-row cells, as measured)
    gives the same bits, and the layer from logits at a small token count gives the same bits as the pre-routed path."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 5, 2, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=9, gated=chain.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A, w1, w2, gated=chain.gated, act=chain.act, params=chain.params
    )
    ids, weights = route(logits, k)
    ref, im_p = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, return_intermediates=True, swap=False
    )
    got, im_s = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, return_intermediates=True, swap=True
    )
    assert (
        im_p["swap"] == (False, False)
        and im_p["requant"] == "fused"
        and im_p["a"] is None
    )
    assert (
        im_s["swap"] == (True, True)
        and im_s["requant"] == "fused"
        and im_s["a"] is None
    )
    assert torch.equal(bits(got), bits(ref))
    for sw in (False, True):
        assert torch.equal(
            bits(
                F.fused_moe_mxfp4w8a(
                    (xq, xsf), ids, weights, W, swap=sw, requant="standalone"
                )
            ),
            bits(ref),
        ), sw
    out0, im0 = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, return_intermediates=True
    )
    assert isinstance(im0["swap"], tuple) and len(im0["swap"]) == 2
    assert torch.equal(bits(out0), bits(ref))
    layer = _layer(k, E, I, act=act)
    assert torch.equal(
        bits(layer(_logits_pack((xq, xsf), logits), _weights(w1, w2, act))), bits(ref)
    )


@needs_device
def test_dense_oracle_at_decode_rows_takes_the_dispatchers_record():
    """The oracle's per-expert dense call with an f32 output at decode row counts, with the dispatcher's pick instead of the
    pinned record: on a tree that carries the mixed GEMM classes and the grouped kernels together, the dense class's swapped
    decode records (on SM107 with their 256-element stage) must build and compute the class, bitwise the pinned record's
    accumulators (the `MainloopBlock` override by name; a positional build put the k-block in PR 1's `inner_steps`)."""
    from flashinfer.gemm.batch_invariant.generator import Knobs
    from flashinfer.gemm.batch_invariant.heuristic import family_of
    from flashinfer.gemm.batch_invariant.schedule import default_config
    import flashinfer

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability(dev)
    N, K = 256, 512
    g = torch.Generator().manual_seed(11)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A,
        (torch.randn(1, N, K, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        (torch.randn(1, K, N // 2, generator=g) * 0.1).to(torch.bfloat16).cuda(),
        interleaved=True,
    )
    x = (
        (
            torch.randn(64, K, generator=g)
            * torch.exp2(torch.randint(-6, 6, (64, K), generator=g).float())
        )
        .to(torch.bfloat16)
        .cuda()
    )
    xq, xsf = _quant(x)
    fmts = ("mxfp4w8a_k64", "mxfp4w8a") if cc == (10, 7) else ("mxfp4w8a",)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    picks = []
    for fmt in fmts:
        for M in (1, 5, 28, 64):
            pick = default_config(M, N, K, 1, "k", sms, family_of(cc), fmt)
            picks.append(pick)
            a = xq[:M].view(torch.float8_e4m3fn)
            sf = _swizzle(xsf[:M].contiguous(), K)
            args = (a, W.w1[0].t(), sf, W.w1_sf[0])
            got = flashinfer.gemm.batch_invariant.api.mm_mxfp8_mxfp4_batch_invariant(
                *args, out_dtype=torch.float32, in_fmt=fmt
            )
            ref = flashinfer.gemm.batch_invariant.api.mm_mxfp8_mxfp4_batch_invariant(
                *args,
                out_dtype=torch.float32,
                in_fmt=fmt,
                config=Knobs(*_DENSE_RECORD),
            )
            assert torch.equal(got, ref), (fmt, M, pick.describe())
    assert any(p.swap_ab for p in picks), [p.describe() for p in picks]


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
@pytest.mark.parametrize("T", [300, 8], ids=["mid", "decode"])
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits(T):
    """Every tactic the autotuner may pick (the dispatcher's ranked record pairs over both record spaces) gives the same
    bits; tuning the layer profiles them and the output before, during and after tuning is one output."""
    from flashinfer.autotuner import AutoTuner, autotune

    k, H, I, E = 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=6, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantFp4Runner) and runner.pair is MXFP4W8A
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
    layer.reset_winner()
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    if hasattr(AutoTuner.get(), "stats"):
        assert len(AutoTuner.get().stats.failed_tactics) == failed_before


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
@pytest.mark.parametrize("requant", ["fused", "standalone"])
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "silu"])
def test_poisoned_intermediates_do_not_reach_the_bits(requant, gated):
    """Unspecified bytes and timing never reach the output: the op's intermediates (``x_perm`` and its scales with their
    padding rows, GEMM1's output or its fused codes and scales, GEMM2's output, the output itself) come from ``torch.empty``;
    with the caching allocator's blocks filled with NaN patterns before every call and a bandwidth-bound kernel running on a
    side stream of the same device during the call, both requantization spellings give the bits of a clean run, each other's
    bits and the layer's, ``RLK_STRESS_ITERS`` times (default 20) at a shape whose GEMM2 K is one k-tile."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F
    from flashinfer.fused_moe.batch_invariant.api import route

    T, k, H, I, E = 64, 8, 256, 128, 256
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=T + 3 * E, gated=gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A, w1, w2, gated=gated, act="silu" if gated else "relu2"
    )
    ids, weights = route(logits, k)
    ref = bits(F.fused_moe_mxfp4w8a((xq, xsf), ids, weights, W, requant=requant))
    other = "standalone" if requant == "fused" else "fused"
    assert torch.equal(
        bits(F.fused_moe_mxfp4w8a((xq, xsf), ids, weights, W, requant=other)), ref
    )
    side = torch.cuda.Stream()
    junk = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(int(os.environ.get("FLASHINFER_BI_STRESS_ITERS", "20"))):
        for byte in (0x7F, 0xFF, 0x00):
            _poison_allocator(byte)
            with torch.cuda.stream(side):
                for _ in range(3):
                    junk.add_(1)
            out = F.fused_moe_mxfp4w8a((xq, xsf), ids, weights, W, requant=requant)
            assert torch.equal(bits(out), ref), (i, hex(byte))
    torch.cuda.synchronize()


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_fused_requantization_equals_the_standalone_spelling(act):
    """GEMM1's quantizing epilogue (the ``mxfp8.quant`` class applied to the bf16-rounded chain in place, codes and 128x4 block
    scales out) is a placement of the standalone pass: the codes, the scales, GEMM2's output and the op are bit-equal on the
    live rows for every typed activation; the op's default (the dispatcher's body and spelling) gives the same bits."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=3, gated=chain.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A, w1, w2, gated=chain.gated, act=chain.act, params=chain.params
    )
    ids, weights = route(logits, k)
    # the padded body's two spellings, then the swapped decode body's
    out_s, im_s = F.fused_moe_mxfp4w8a(
        (xq, xsf),
        ids,
        weights,
        W,
        return_intermediates=True,
        requant="standalone",
        swap=False,
    )
    out_f, im_f = F.fused_moe_mxfp4w8a(
        (xq, xsf),
        ids,
        weights,
        W,
        return_intermediates=True,
        requant="fused",
        swap=False,
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
    assert torch.equal(
        bits(F.fused_moe_mxfp4w8a((xq, xsf), ids, weights, W)), bits(out_f)
    )
    # the swapped decode body (the dense packing) runs the fused spelling: its codes on the dense rows and the op's bits are the
    # grouped body's; the standalone spelling is a padded-path form, so asking for it with the swapped body gives the grouped
    # records (the dispatcher's rule), with the same bits
    out_w, im_w = F.fused_moe_mxfp4w8a(
        (xq, xsf),
        ids,
        weights,
        W,
        return_intermediates=True,
        requant="fused",
        swap=True,
    )
    assert im_w["swap"] == (True, True) and im_w["path"] == "dense"
    dd, dp = im_w["dest"].reshape(-1).long(), im_w["dest_pad"].reshape(-1).long()
    assert torch.equal(
        im_w["aq"].view(torch.uint8)[dd], im_s["aq"].view(torch.uint8)[dp]
    )
    assert torch.equal(bits(out_w), bits(out_s))
    out_n, im_n = F.fused_moe_mxfp4w8a(
        (xq, xsf),
        ids,
        weights,
        W,
        return_intermediates=True,
        requant="standalone",
        swap=True,
    )
    assert im_n["swap"] == (False, False) and im_n["requant"] == "standalone"
    assert torch.equal(bits(out_n), bits(out_s))


@needs_device
def test_grouped_routing_config_takes_prerouted_ids_and_refuses_logits():
    """A group-limited routing configuration (``n_group`` / ``topk_group`` > 1) constructs; with a method other than
    DeepSeekV3 it is refused from logits (the groups are a term of the DeepSeekV3 method alone); with pre-routed ids the group
    structure is irrelevant and the output is the ungrouped configuration's bit for bit on the same ids and weights; the
    grouped DeepSeekV3 layer runs from logits deterministically (the bf16 model's grouped routing term)."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

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
            backend=BackendOptions((BatchInvariantFp4Config(),)),
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
            backend=BackendOptions((BatchInvariantFp4Config(),)),
            execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=4096),
        ),
        device=torch.device("cuda"),
    )
    bias = torch.randn(E, device="cuda")
    out_g = dsv3(_logits_pack((xq, xsf), logits, bias), W)
    assert out_g.shape == (T, H) and torch.equal(
        bits(out_g), bits(dsv3(_logits_pack((xq, xsf), logits, bias), W))
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
    assert isinstance(oai.runners[0], BatchInvariantFp4Runner)
    assert oai.runners[0].pair is MXFP4W8A
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=200), "intermediate_size"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.mxfp4w8a" in str(excinfo.value)
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
                    backend=BackendOptions((BatchInvariantFp4Config(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.mxfp4w8a" in str(excinfo.value), what
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
    # Llama4 routing (the top-1 with its sigmoid as the weight) is a routing term the bf16 model serves: the layer runs from
    # logits, deterministically, and takes pre-routed inputs; a method the classes do not implement is refused from logits
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp4Config(),)),
        ),
        device=torch.device("cuda"),
    )
    out_l = llama(_logits_pack((xq, xsf), logits), W)
    assert out_l.shape == (8, H) and torch.equal(
        bits(out_l), bits(llama(_logits_pack((xq, xsf), logits), W))
    )
    minimax = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=k, method=RoutingMethodType.MiniMax2
            ),
            quant=_QUANT,
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantFp4Config(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(BatchInvariantUnsupportedError):
        minimax(_logits_pack((xq, xsf), logits), W)
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(
        _routed_pack((xq, xsf), ids, torch.ones(8, 1, device="cuda")), W
    ).shape == (8, H)
    # weights on the CPU with no CUDA pack device: refused by name (the quantizer class runs on the device)
    with pytest.raises(BatchInvariantUnsupportedError, match="CUDA device"):
        BatchInvariantFp4Config.prepare_weights(
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


# ----------------------------------------------------------------------------- the unified bodies and the reviewed rules


def test_mixed_classes_run_on_the_unified_bodies():
    """The mixed classes have no body of their own: the backend's grouped and swapped bodies (:mod:`blocks.kernel`,
    :mod:`blocks.swap`) take the MX chain at a mixed instruction class through the block-scaled operand block (an e4m3 token
    operand against a byte-packed e2m1 weight operand, UE8M0 scales per 32 on both, four epilogue warpgroups, the 128-row
    scale atom as the group alignment, both packings; the swapped body on the twin plan with the operand formats
    exchanged) and the fused requantization through the MX epilogue block's hooks; the compiled objects carry the
    instruction class, the packing and the record's k-block in their names; the shared acceptance runs at the chain's
    class, with the sm_100 class's 128-element K granule; one JIT module (``batch_invariant_moe``) holds the kernels and the
    MX scatter, with no mixed-specific body in its source closure."""
    from flashinfer.batch_invariant import jit
    from flashinfer.fused_moe.batch_invariant.blocks import compile as C
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import (
        MxChain,
        grouped_epilogue_quant_loop,
        swapped_epilogue_quant_loop,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.kernel import GroupedBlockGemm
    from flashinfer.fused_moe.batch_invariant.blocks.operands import (
        BlockScaledOperands,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.swap import (
        SwappedGroupedGemm,
        epi_warpgroups,
        is_swapped_record,
    )
    from flashinfer.fused_moe.batch_invariant.mx_permute import MODULE as SCATTER_MODULE
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        GEMM2_CHAIN,
        swapped_knobs_mx,
    )
    from flashinfer.gemm.batch_invariant.blocks.blockscaled_mx_mixed import (
        PLANS,
        SWAPPED_PLANS,
    )
    from flashinfer.gemm.batch_invariant.generator import Knobs

    dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
    for fmt in ("mxfp4w8a", "mxfp4w8a_k64"):
        quant = MxChain("glu", "silu", "mxfp8", (), fmt)
        plain = GEMM2_CHAIN.at(fmt)
        assert quant.quant and not plain.quant and quant.in_fmt == plain.in_fmt == fmt
        assert quant.grouped_loop() is grouped_epilogue_quant_loop
        assert quant.swapped_loop() is swapped_epilogue_quant_loop
        assert quant.dead_lane_select and quant.operands == ()
        for packing in ("padded", "dense"):
            body = GroupedBlockGemm(dense, quant, packing=packing)
            ops = body.operands
            assert ops == BlockScaledOperands(fmt, packing) and ops.block_scaled
            assert ops.mixed and ops.plan_cls is PLANS[fmt]
            assert ops.plan_cls_for("grouped") is PLANS[fmt]
            assert ops.plan_cls_for("swapped") is SWAPPED_PLANS[fmt]
            assert (ops.sf_vec, ops.group_align, body.epi_warpgroups) == (32, 128, 4)
            assert (ops.token_elem_bits, ops.weight_elem_bits) == (8, 4)
            assert ops.dtype is cutlass.Float8E4M3FN
            assert ops.dense_a == (packing == "dense")
        for sw in swapped_knobs_mx(fmt):
            assert is_swapped_record(sw, fmt)
            s = SwappedGroupedGemm(sw, plain)
            assert s.operands.mixed and s.k_block == 0
            # the shared transposed epilogue runs over the record's warpgroups (one per eight-token subtile, at most four);
            # the scale-factor warp, the MMA warp and the TMA warp follow them
            assert (
                s.epi_wgs
                == epi_warpgroups(sw.tile_n, plain)
                == max(1, min(4, sw.tile_n // 8))
            )
            assert (s.sf_warp, s.mma_warp, s.tma_warp) == (
                4 * s.epi_wgs,
                4 * s.epi_wgs + 1,
                4 * s.epi_wgs + 2,
            )
            # the fused requantization's own transposed epilogue splits the subtiles the same way
            sq = SwappedGroupedGemm(sw, quant)
            assert (sq.epi_wgs, sq.sf_warp) == (s.epi_wgs, s.sf_warp)
        gran = 128 if fmt == "mxfp4w8a" else 32
        assert C.blockscaled_k_align(fmt) == gran
        assert C.can_implement_grouped_blockscaled(dense, 256, 2048, 8, quant, fmt)
        assert C.can_implement_swapped_blockscaled(
            swapped_knobs_mx(fmt)[0], 256, 2048, 8, quant, fmt
        )
        # K in whole granules of the class: whole 32-element scale blocks on the sm_107 class, whole 128-element runs of
        # the e2m1 weight on the sm_100 class
        assert not C.can_implement_grouped_blockscaled(
            dense, 256, 2048 + 16, 8, quant, fmt
        )
        for dk in (32, 64, 96):
            assert C.can_implement_grouped_blockscaled(
                dense, 256, 2048 + dk, 8, quant, fmt
            ) == (fmt == "mxfp4w8a_k64"), (fmt, dk)
        assert C.can_implement_grouped_blockscaled(
            dense, 256, 2048 + 128, 8, quant, fmt
        )
    # a bf16 chain names no instruction class: the shared wrapper takes the MX chain (the mixed dispatcher converts)
    assert not C.can_implement_grouped_blockscaled(
        dense, 256, 2048, 8, GroupedChain("plain", "silu", "bf16"), "mxfp4w8a"
    )
    assert C.kernel_name(
        dense, MxChain("glu", "silu", "mxfp8", (), "mxfp4w8a_k64"), packing="padded"
    ).startswith("grouped_mxfp4w8ak64_swiglu_mxfp8_")
    assert C.kernel_name(
        dense, MxChain("glu", "silu", "mxfp8", (), "mxfp4w8a"), packing="dense"
    ).startswith("grouped_mxfp4w8a_densea_swiglu_mxfp8_")
    # the plain chain's 16-token record: the record's k-block, then the two-warpgroup epilogue's suffix
    name = C.swapped_kernel_name(
        swapped_knobs_mx("mxfp4w8a_k64")[1], GEMM2_CHAIN.at("mxfp4w8a_k64")
    )
    assert name.startswith("swapped_mxfp4w8ak64_plain_bf16_") and name.endswith(
        "_swap_kb256_e2"
    )
    # the fused requantization's epilogue over the same warpgroups: the same suffix
    name = C.swapped_kernel_name(
        swapped_knobs_mx("mxfp4w8a_k64")[1],
        MxChain("glu", "silu", "mxfp8", (), "mxfp4w8a_k64"),
    )
    assert name.endswith("_swap_kb256_e2")
    assert C.swapped_kernel_name(
        swapped_knobs_mx("mxfp4w8a_k64")[0],
        MxChain("glu", "silu", "mxfp8", (), "mxfp4w8a_k64"),
    ).endswith("_swap_kb256")
    assert C.MODULE == SCATTER_MODULE == "batch_invariant_moe"
    assert "batch_invariant_moe_mxfp4w8a" not in jit.MODULE_ROOTS
    srcs = jit.relative_sources("batch_invariant_moe")
    gone = ("blockscaled_kernel.py", "swap_mxfp8.py", "compile_mxfp4w8a.py")
    assert not any(os.path.basename(src) in gone for src in srcs), srcs
    for src in (
        "fused_moe/batch_invariant/mx_permute.py",
        "fused_moe/batch_invariant/blocks/epilogue_mx.py",
        "fused_moe/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/blocks/blockscaled_mx_mixed.py",
    ):
        assert src in srcs, srcs


def test_swapped_records_carry_their_k_block_and_trigger_at_the_shared_sites():
    """The swapped decode records of the mixed classes carry their stage width as a field of the record (the sm_107 class
    streams its e2m1 weight at a 256-element stage, the sm_100 class at the format's default), so the record names the
    compiled object and the body stages the record's k-block; the body refuses the one-format classes' gather form; the
    trigger placement is the shared module constant (:mod:`pdl`), with no body-local trigger set."""
    import inspect

    from flashinfer.fused_moe.batch_invariant import pdl
    from flashinfer.fused_moe.batch_invariant.blocks import kernel, swap
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.blocks.swap import SwappedGroupedGemm
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        GEMM2_CHAIN,
        swapped_knobs_mx,
    )
    from flashinfer.gemm.batch_invariant.generator import SWAP_K_BLOCK

    assert SWAP_K_BLOCK.get("mxfp4w8a_k64") == 256 and "mxfp4w8a" not in SWAP_K_BLOCK
    for fmt in ("mxfp4w8a", "mxfp4w8a_k64"):
        ch = MxChain("glu", "silu", "mxfp8", (), fmt)
        for cfg in swapped_knobs_mx(fmt):
            assert cfg.k_block == SWAP_K_BLOCK.get(fmt, 0) and cfg.well_formed(fmt)
            SwappedGroupedGemm(cfg, ch)
        cfg = swapped_knobs_mx(fmt)[0]
        with pytest.raises(AssertionError):
            SwappedGroupedGemm(cfg, ch, k_block=48)
        with pytest.raises(AssertionError):
            SwappedGroupedGemm(cfg, ch, gather=True)
        SwappedGroupedGemm(cfg, GEMM2_CHAIN.at(fmt))
    assert pdl.TRIGGER == "none" and pdl.TRIGGER_SMALL == "none"
    for mod in (swap, kernel):
        src = inspect.getsource(mod)
        assert "pdl.trigger_early()" in src and "pdl.trigger_late()" in src
        assert "BOX_FORM_TRIGGERS" not in src
    assert not hasattr(pdl, "BOX_FORM_TRIGGERS")


def test_ranking_is_memoised_per_token_count():
    """The dispatcher's ranking (`schedule_mxfp4w8a.rank_grouped_mx`) is memoised on its arguments: a forward without a
    tactic pays the ranking once per token count; a bf16 chain object is taken at the instruction class (the same key as its
    MX chain), a chain at the other class is another key."""
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue_mx import MxChain
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        _rank_grouped_mx,
        rank_grouped_mx,
    )

    _rank_grouped_mx.cache_clear()
    chain = MxChain("glu", "silu", "mxfp8", (), "mxfp4w8a_k64")
    args = (1536, 2048, 128, chain, "mxfp4w8a_k64", 212, "sm_107", 3, False, 8)
    a = rank_grouped_mx(1, *args)
    assert a == rank_grouped_mx(1, *args)
    info = _rank_grouped_mx.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    rank_grouped_mx(2, *args)
    assert _rank_grouped_mx.cache_info().misses == 2
    # the same chain spelled as an MX chain at another class, or as the bf16 chain it wraps: the same key
    assert (
        rank_grouped_mx(1, 1536, 2048, 128, MxChain("glu", "silu", "mxfp8"), *args[4:])
        == a
    )
    assert _rank_grouped_mx.cache_info().hits == 2
    bf16 = GroupedChain("glu", "silu", "bf16")
    rank_grouped_mx(1, 1536, 2048, 128, bf16, "mxfp4w8a", *args[5:])
    assert _rank_grouped_mx.cache_info().misses == 3
    assert all(c.swap_ab or c.tile_n in (64, 128, 192, 256) for c in a) and a


def test_grouped_gemm_shape_rules_are_host_errors():
    """The grouped mixed GEMM entry checks the chain's shape rules on the host before any record or kernel is consulted: K
    in whole granules of the class (32 on the SM107 class, 128 on the SM100 / SM103 one), N at the weight-row rule, whole
    gate / up column pairs for a paired chain, an output width a multiple of the output format's row (a whole scale block
    for the fused requantization), a 16-byte-aligned output base, the packed weight's own layout. Device-free: the rules
    raise before the instruction class is looked up on a device."""
    from flashinfer.fused_moe.batch_invariant.api_mxfp4w8a import grouped_gemm_mxfp4w8a
    from flashinfer.fused_moe.batch_invariant.api_mxfp8 import sf_numel
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    from flashinfer.gemm.batch_invariant.generator import Knobs

    cfg = Knobs((128, 128), (1, 1), False, True, "m", 1)
    plain = GroupedChain("plain", "silu", "bf16")

    def operands(Tk, N, K, E=4):
        a = torch.zeros(Tk, K, dtype=torch.float8_e4m3fn)
        w = torch.zeros(E, N, K // 2, dtype=torch.uint8)
        Kb = (
            -(-K // 32) * 32
        )  # scale buffers of a valid size: the operand rule is the one under test
        a_sf = torch.zeros(sf_numel(Tk, Kb), dtype=torch.uint8)
        w_sf = torch.zeros(E, sf_numel(-(-N // 8) * 8, Kb), dtype=torch.uint8)
        cnt = torch.full((E,), Tk // E, dtype=torch.int32)
        return a, a_sf, w, w_sf, cnt

    with pytest.raises(ValueError, match="K a multiple of 32"):
        grouped_gemm_mxfp4w8a(*operands(128, 64, 48), cfg, plain, "mxfp4w8a_k64")
    with pytest.raises(ValueError, match="K a multiple of 128"):
        grouped_gemm_mxfp4w8a(*operands(128, 64, 64), cfg, plain, "mxfp4w8a")
    with pytest.raises(ValueError, match="N of 8"):
        grouped_gemm_mxfp4w8a(*operands(128, 20, 64), cfg, plain, "mxfp4w8a_k64")
    W = MoEWeightsFp4(MXFP4W8A, None, None, None, None, 64, 64, True, "silu", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="whole gate / up column pairs"):
        grouped_gemm_mxfp4w8a(*operands(128, 24, 64), cfg, W.chain1, "mxfp4w8a_k64")
    with pytest.raises(ValueError, match="output width of 16 mxfp8"):
        grouped_gemm_mxfp4w8a(
            *operands(128, 32, 64), cfg, W.chain1_quant, "mxfp4w8a_k64"
        )
    with pytest.raises(ValueError, match="a multiple of the 128-row scale-factor atom"):
        grouped_gemm_mxfp4w8a(*operands(96, 64, 64), cfg, plain, "mxfp4w8a_k64")
    a, a_sf, w, w_sf, cnt = operands(128, 64, 64)
    buf = torch.empty(128, 64 + 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        grouped_gemm_mxfp4w8a(
            a, a_sf, w, w_sf, cnt, cfg, plain, "mxfp4w8a_k64", out=buf[:, 4:68]
        )
    with pytest.raises(ValueError, match="allocates its two outputs"):
        grouped_gemm_mxfp4w8a(
            a, a_sf, w, w_sf, cnt, cfg, W.chain1_quant, "mxfp4w8a_k64", out=buf[:, :64]
        )
    # the dense packing's scale rows must cover the padded positions of the live rows (the capacity check before the launch)
    with pytest.raises(ValueError, match="padded positions"):
        grouped_gemm_mxfp4w8a(
            a, a_sf, w, w_sf, cnt, cfg, W.chain1_quant, "mxfp4w8a_k64",
            dense_a=True, sf_rows=128, n_live=128 * 4,
        )  # fmt: skip
    with pytest.raises(
        TypeError
    ):  # the weight is the packed e2m1 bytes, two codes per byte
        grouped_gemm_mxfp4w8a(
            a, a_sf, w.view(torch.float8_e4m3fn), w_sf, cnt, cfg, plain, "mxfp4w8a_k64"
        )
    with pytest.raises(
        ValueError, match="dense packing"
    ):  # a swapped record reads the dense packing
        from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
            swapped_knobs_mx,
        )

        grouped_gemm_mxfp4w8a(
            a,
            a_sf,
            w,
            w_sf,
            cnt,
            swapped_knobs_mx("mxfp4w8a_k64")[0],
            plain,
            "mxfp4w8a_k64",
        )


@needs_device
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``: a framework's dropped slot) is dropped by the mixed
    op as by the bf16 one: the plan does not count it (``dest -1``), the scatter and the two GEMMs never touch it (the
    scatter writes nothing for a dropped slot: under a poisoned caching allocator the bits are the composition's), and the
    combine folds the remaining slots in slot order with the term ``+0`` in its place, so the output equals the class's
    composition on the same rows with those slots at a valid id and weight ``+0``; a token whose slots are all dropped
    gives a zero row; the other tokens keep their bits. Both packings (the dense packing of the swapped records, the padded
    packing) and both plan forms (the one-launch plan below ``PLAN_MULTI_MIN_N`` expanded rows and the two-launch form
    above it). With every slot dropped the GEMMs launch without a tile and the output is all zero rows, on the grouped and
    the swapped records."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N, permute_plan
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        default_configs_mx,
    )
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
        Wq = MoEWeightsFp4.from_view(W.get_view(KEY))
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
            for byte in (None, 0x7F, 0xFF):
                if byte is not None:
                    _poison_allocator(byte)
                out_api = F.fused_moe_mxfp4w8a(
                    (xq, xsf),
                    ids_bad.cuda(),
                    wts.cuda(),
                    Wq,
                    config1=c1,
                    config2=c2,
                    path=path,
                )
                assert torch.equal(bits(out_api), bits(out)), (T, path, byte)
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack((xq, xsf), ids.cuda(), wts.cuda()), W)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))
    # every slot dropped: zero counts, the GEMMs launch without a tile, the zero row of every token, on both record kinds
    T, k, E = 37, 4, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=44)
    xq, xsf = _quant(x)
    Wq = MoEWeightsFp4.from_view(_weights(w1, w2).get_view(KEY))
    ids_none = torch.full((T, k), -1, dtype=torch.int32, device="cuda")
    wts_any = torch.rand(T, k, device="cuda")
    out, im = F.fused_moe_mxfp4w8a(
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
        out2 = F.fused_moe_mxfp4w8a(
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
    (the requantization's block amax and saturation follow the device's non-NaN max / min rule the quantizer class states,
    so a NaN accumulator row leaves the chain as finite codes; what the kernel writes is what the composition writes) -- and
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
def test_routing_constants_are_read_from_the_cache_under_graph_capture():
    """The route kernels' absent bias and unit scale are cached device constants (``api._const``): a call under CUDA graph
    capture reads the cached tensor when it exists (no fill node in the graph), a default first met inside a capture is a
    fill of that capture and is never cached, and a captured from-logits decode call of the mixed layer replays the eager
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
    grouped body bit for bit; the fused codes and scales and GEMM2's rows agree per expanded row; a mixed pair (a swapped
    GEMM1, a grouped GEMM2 on the dense packing) agrees too; the dispatcher's own picks give the same bits whatever the
    packing it chose; a swapped record asked to run on the padded packing is refused."""
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.schedule_mxfp4w8a import (
        default_configs_mx,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    a = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=T + E + len(a.act), gated=a.gated, adversarial=True
    )
    xq, xsf = _quant(x)
    W = MoEWeightsFp4.prepare(
        MXFP4W8A, w1, w2, gated=a.gated, act=a.act, params=a.params
    )
    ids, weights = route(logits, k)
    sm_count, family = F._part(xq.device)
    fmt = F.mx_class_of(xq.device)
    p1, p2 = default_configs_mx(
        T, k, H, I, E, W.chain1, fmt, sm_count, family, swap=False
    )
    s1, s2 = default_configs_mx(
        T, k, H, I, E, W.chain1, fmt, sm_count, family, swap=True
    )
    ref, im_p = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, config1=p1, config2=p2, return_intermediates=True
    )
    got, im_s = F.fused_moe_mxfp4w8a(
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
    mixed = F.fused_moe_mxfp4w8a((xq, xsf), ids, weights, W, config1=s1, config2=p2)
    assert torch.equal(bits(mixed), bits(ref))
    dense_padded = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, config1=p1, config2=p2, path="dense"
    )
    assert torch.equal(bits(dense_padded), bits(ref))
    auto, im_a = F.fused_moe_mxfp4w8a(
        (xq, xsf), ids, weights, W, return_intermediates=True
    )
    assert torch.equal(bits(auto), bits(ref)) and im_a["path"] in ("padded", "dense")
    with pytest.raises(ValueError, match="dense packing"):
        F.fused_moe_mxfp4w8a(
            (xq, xsf), ids, weights, W, config1=s1, config2=s2, path="padded"
        )


@needs_device
def test_decode_tactics_include_the_swapped_records_and_agree():
    """At a decode-sized batch of a large model's shape the ranked tactics include the swapped records (the swapped body on
    the dense packing, the layer's routing-and-plan launch in its dual form) and every tactic gives one output; the tactic
    round-trips its swapped flag and its k-block; from logits and pre-routed agree."""
    from flashinfer.fused_moe.batch_invariant.runner_fp4 import (
        _configs_from_tactic_fp4,
    )

    T, k, H, I, E = 1, 8, 4096, 1536, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=41, adversarial=True)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert runner.pair is MXFP4W8A
    pack = _logits_pack(x, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert 1 <= len(tactics) <= 9 and len(set(tactics)) == len(tactics)
    firsts = _configs_from_tactic_fp4(tactics[0])
    if firsts[0].swap_ab or firsts[1].swap_ab:
        assert len(inputs) == 14 + 5, (
            "the dual plan: counts, dest, src, dest_pad, pos_pad"
        )
    else:
        assert len(inputs) == 14 + 3, "the padded plan: counts, dest, src"
    assert (
        any(
            _configs_from_tactic_fp4(t)[0].swap_ab
            or _configs_from_tactic_fp4(t)[1].swap_ab
            for t in tactics
        )
        or len(inputs) == 12
    )
    for t in tactics:
        out = runner.forward(runner.pack_inputs(pack, W), tactic=t)
        assert torch.equal(bits(out), ref), t
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(layer(_routed_pack((_quant(x)), ids.cuda(), weights.cuda()), W)), ref
    )


@needs_device
def test_misaligned_pack_is_refused_with_the_alignment_named():
    """The e4m3 rows or the UE8M0 scale bytes of the pack at a storage offset that is not a multiple of 16 bytes are refused
    by the layer and by the op with the mode's error naming the alignment, before any kernel (the shared MXFP8 pack rule
    through the fp4 backend's runner); the aligned pack runs, and TRTLLM-Gen's e4m3-typed scale carrier is the same pack."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant import api_mxfp4w8a as F

    T, k, H, I, E = 40, 4, 256, 128, 8
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5)
    xq, xsf = _quant(x)
    layer, W = _layer(k, E, I), _weights(w1, w2)
    Wq = MoEWeightsFp4.from_view(W.get_view(KEY))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    ref = bits(layer(_routed_pack((xq, xsf), ids, weights), W))

    def misaligned(t: torch.Tensor, off: int) -> torch.Tensor:
        es, n = t.element_size(), t.numel()
        big = torch.empty(n * es + 16, dtype=torch.uint8, device="cuda")
        v = big[off : off + n * es].view(t.dtype).view(t.shape)
        v.copy_(t)
        assert v.data_ptr() % 16 == off and v.is_contiguous()
        return v

    for q, sf in ((misaligned(xq, 2), xsf), (xq, misaligned(xsf, 1))):
        with pytest.raises(BatchInvariantUnsupportedError, match="16 bytes"):
            layer(_routed_pack((q, sf), ids, weights), W)
        with pytest.raises(BatchInvariantUnsupportedError, match="16 bytes"):
            F.fused_moe_mxfp4w8a((q, sf), ids, weights, Wq)
    assert torch.equal(bits(layer(_routed_pack((xq, xsf), ids, weights), W)), ref)
    carrier = xsf.view(
        torch.float8_e4m3fn
    )  # TrtllmFp4Config's typing of the UE8M0 bytes for this pair
    assert torch.equal(bits(layer(_routed_pack((xq, carrier), ids, weights), W)), ref)


@needs_device
def test_tactic_of_the_other_packing_replays_with_the_bits():
    """A cached autotune winner may replay at a token count whose first picks pack the other way (the layer keys its winners
    by token bucket): the runner takes the records' own packing and keeps the plan the records can read (the dual plan feeds
    both bodies) or drops it (the op computes its own: the same integers); the bits are the eager call's either way. The
    tactics are taken from the token counts where the ranking places them, so the test holds on every part."""
    from flashinfer.fused_moe.batch_invariant.runner_fp4 import _configs_from_tactic_fp4

    k, H, I, E = 8, 2048, 768, 128
    layer, W = None, None
    tactics, packs, refs = {}, {}, {}
    for name, T, seed in (("decode", 16, 71), ("mid", 512, 72)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed)
        if layer is None:
            layer, W = _layer(k, E, I), _weights(w1, w2)
        runner = layer.runners[0]
        packs[name] = _logits_pack(_quant(x), logits)
        refs[name] = bits(layer(packs[name], W))
        tactics[name] = runner.get_valid_tactics(
            runner.pack_inputs(packs[name], W), None
        )
    runner = layer.runners[0]
    if _family() == "sm_107":
        assert runner._packing(16, H) != runner._packing(512, H), (
            "the two token counts pack the other way"
        )
    for here, there in (("decode", "mid"), ("mid", "decode")):
        for t in tactics[there]:
            c1, c2 = _configs_from_tactic_fp4(t)
            inputs = runner.pack_inputs(packs[here], W)
            out = runner.forward(inputs, tactic=t)
            assert torch.equal(bits(out), refs[here]), (
                here,
                c1.describe(),
                c2.describe(),
            )
