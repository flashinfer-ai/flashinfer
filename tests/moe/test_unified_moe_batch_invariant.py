# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MoE backend of the unified MoE layer (``BatchInvariantBf16Config``): the bf16 classes
``moe.bf16.<act>``, one per typed activation of the layer and its scalars (the gated chains on the (gate, up) accumulator
pair -- ``SwiGLU`` with any parameters, ``SwiGLUStep``, ``GeGLU``, ``GeGLUTanh``, ``SiTU``, ``PowLU`` -- and the ungated
ones on a single accumulator -- ``ReLU2``, ``Identity``, ``GELU``, ``ReLU``, ``SiLU``; the default ``SwiGLU`` is
``moe.bf16.swiglu``, ``ReLU2`` is ``moe.bf16.relu2``).

The class has exact CPU models of everything but the tensor-core step (the routing, the activation chain on the fp32
accumulators, the slot-order combine), so the tests are bit comparisons: the layer's output against the class's own
composition on the CPU (routing on the CPU, the grouped GEMMs checked against the dense batch-invariant GEMM per expert)
for every activation, run-to-run determinism including CUDA-graph replay, batch / permutation / sub-batch invariance,
agreement of every tactic the autotuner may pick, the three routing input modes giving the same bits, and the error
contract (an unsupported configuration raises before any kernel runs). The device-free tests pin the registration, the
activation table (every ``ActivationConfig`` of the layer maps onto a chain; the fingerprints are distinct classes), the
agreement of the vectorised and the exact CPU chains on adversarial accumulators, the plausibility of every chain against
FlashInfer's torch reference of the activation, the fingerprint dimensions and the dispatcher's membership in the
enumerated space.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    ActivationConfig,
    BackendOptions,
    BatchInvariantBf16Config,
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
    MOE_BF16_ARCHS,
    BatchInvariantMoeRunner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_BF16_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


_SWIGLU = SwiGLU()  # the default (gated) activation of the layer

#: The activations the suites cover: every ``ActivationConfig`` class of the layer with its defaults, and the
#: parameterised ones once more with scalars that move every branch (a clamp that bites, a non-unit alpha / beta, a SiTU
#: without a linear scale, a PowLU with a limit). Each is its own class of the backend.
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
    """Activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated)."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        # spread exponents, near-cancelling pairs and ties in the logits: what moves bits when an order moves
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
):
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantBf16Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )
    return cfg


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(BatchInvariantMoeRunner.backend_key, view)
    return pack


def _logits_pack(x: torch.Tensor, logits: torch.Tensor, bias=None) -> MoEActivationPack:
    return MoEActivationPack(
        x,
        None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(
    x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, packed: bool = False
) -> MoEActivationPack:
    if packed:
        # the layer's packed mode carries ids and fp32 weights; the runner packs them, rounding the weights to bf16
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
    """The config is a known backend of the layer, mapped to its runner, absent from the default candidates (it is
    reached by naming it), and declares the three routing modes and every activation class of the layer."""
    assert BatchInvariantBf16Config in ALL_BACKEND_CONFIGS
    assert _BACKEND_RUNNERS[BatchInvariantBf16Config] is BatchInvariantMoeRunner
    assert not any(
        isinstance(c, BatchInvariantBf16Config) for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantBf16Config.supported(a)
    ] == [100, 103, 107]
    assert set(BatchInvariantMoeRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert set(BatchInvariantMoeRunner.supported_activation_classes) == {
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
    assert repr(BatchInvariantBf16Config()) == "BatchInvariantBf16Config()"


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe", ...)`` takes the geometry, the routing method and the activation; the gated and the
    ungated chain are two classes; an unimplemented routing method and an unknown chain are refused; the routing method
    is a dimension."""
    fp = bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=128)
    assert fp.startswith("moe-") and len(fp) == 20
    assert (
        bi.fingerprint(
            "fused_moe",
            H=2048,
            I=768,
            top_k=8,
            num_experts=128,
            method="Renormalize",
            act="swiglu",
        )
        == fp
    )
    assert (
        bi.fingerprint(
            "fused_moe", H=2048, I=768, top_k=8, num_experts=128, act="relu2"
        )
        != fp
    )
    assert (
        bi.fingerprint(
            "fused_moe",
            H=2048,
            I=768,
            top_k=8,
            num_experts=128,
            method=RoutingMethodType.Default,
        )
        != fp
    )
    assert (
        bi.fingerprint(
            "fused_moe",
            H=2048,
            I=768,
            top_k=8,
            num_experts=128,
            method=RoutingMethodType.SigmoidRenorm,
        )
        != fp
    )
    assert bi.fingerprint(
        "fused_moe", H=2048, I=768, top_k=8, num_experts=128
    ) != bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=64)
    with pytest.raises(ValueError):
        bi.fingerprint(
            "fused_moe",
            H=2048,
            I=768,
            top_k=8,
            num_experts=128,
            method=RoutingMethodType.Llama4,
        )
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=128, act="tanh")
    assert bi.class_name("fused_moe") == "moe.bf16"


def test_activation_of_covers_every_activation_config():
    """``activation_of`` maps every ``ActivationConfig`` subclass of the layer's API (found by introspection, built with
    its defaults) and the listed configurations onto a chain of the class: gated exactly when the config is, a chain
    name of the spec's tables with the declared number of binary32 parameters. The listed configurations are distinct
    classes (distinct fingerprints), and all but the default ``SwiGLU`` differ from the default class's."""
    import inspect

    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.fused_moe import api as A
    from flashinfer.fused_moe.batch_invariant import spec as S

    classes = {
        c
        for _, c in inspect.getmembers(A, inspect.isclass)
        if issubclass(c, ActivationConfig) and c is not ActivationConfig
    }
    assert len(classes) == 11
    assert classes == set(BatchInvariantMoeRunner.supported_activation_classes)
    assert {type(a) for a in ACTIVATIONS} == classes
    assert len(ACTIVATIONS) == len(ACTIVATION_IDS) == 15
    for cfg in [c() for c in classes] + ACTIVATIONS:
        a = S.activation_of(cfg)
        assert isinstance(a, S.ActSpec) and a.gated == cfg.is_gated, cfg
        assert a.act in (S.ACTS if a.gated else S.UNGATED_ACTS), cfg
        assert len(a.params) == len(S.ACT_PARAMS.get(a.act, ())), cfg
        assert all(v is None or v == fp32.to_f32(v) for v in a.params), cfg
        assert a.chain == ("glu" if a.gated else "act")
    assert S.activation_of(SwiGLU()) == S.ActSpec(True, "silu")
    assert S.activation_of(SwiGLU(alpha=1.0)) == S.ActSpec(True, "silu")
    assert S.activation_of(SwiGLU(alpha=1.702, beta=1.0, limit=7.0)) == S.ActSpec(
        True, "swiglu_oai", (fp32.to_f32(1.702), 1.0, 7.0)
    )
    assert S.activation_of(SiTU()) == S.ActSpec(True, "situ", (4.0, 25.0, None))
    assert S.activation_of(PowLU()) == S.ActSpec(True, "powlu", (2.5, None))
    assert S.activation_of(ReLU2()) == S.ActSpec(False, "relu2")
    assert S.activation_of(SiLU()) == S.ActSpec(False, "silu")
    with pytest.raises(ValueError):
        S.activation_of("swiglu")
    H, I, k, E = 2048, 768, 8, 128
    default = S.spec_fingerprint(H, I, k, E)
    fps = {}
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        fps[cfg] = S.spec_fingerprint(
            H, I, k, E, S.SOFTMAX_RENORM, "bf16", a.act, a.params, a.gated
        )
        assert fps[cfg].startswith("moe-") and len(fps[cfg]) == 20
    assert len(set(fps.values())) == len(ACTIVATIONS)
    assert fps[SwiGLU()] == default
    assert all(fp != default for cfg, fp in fps.items() if cfg != SwiGLU())


def test_registry_fingerprint_takes_an_activation_config():
    """``fingerprint("fused_moe", ..., act=<ActivationConfig>)`` is the spec's fingerprint of that config's chain for
    every listed configuration; the string forms name the same classes (``"swiglu"`` / ``"relu2"`` the default gated /
    ungated chains, a chain name with ``params`` and ``gated`` the others); the routing method stays a dimension of
    every class."""
    from flashinfer.fused_moe.batch_invariant import spec as S

    H, I, k, E = 2048, 768, 8, 128
    dims = dict(H=H, I=I, top_k=k, num_experts=E)
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        assert bi.fingerprint("fused_moe", act=cfg, **dims) == S.spec_fingerprint(
            H, I, k, E, S.SOFTMAX_RENORM, "bf16", a.act, a.params, a.gated
        ), cfg
        assert bi.fingerprint(
            "fused_moe", act=cfg, method=RoutingMethodType.Sigmoid, **dims
        ) != bi.fingerprint("fused_moe", act=cfg, **dims), cfg
    default = bi.fingerprint("fused_moe", **dims)
    assert bi.fingerprint("fused_moe", act="swiglu", **dims) == default
    assert bi.fingerprint("fused_moe", act=SwiGLU(), **dims) == default
    assert bi.fingerprint("fused_moe", act="relu2", **dims) == bi.fingerprint(
        "fused_moe", act=ReLU2(), **dims
    )
    assert bi.fingerprint(
        "fused_moe", act="swiglu_oai", params=(1.702, 1.0, 7.0), **dims
    ) == bi.fingerprint(
        "fused_moe", act=SwiGLU(alpha=1.702, beta=1.0, limit=7.0), **dims
    )
    # a name in both tables is the gated chain unless told otherwise
    assert bi.fingerprint("fused_moe", act="gelu", **dims) == bi.fingerprint(
        "fused_moe", act=GeGLU(), **dims
    )
    assert bi.fingerprint(
        "fused_moe", act="gelu", gated=False, **dims
    ) == bi.fingerprint("fused_moe", act=GELU(), **dims)
    assert bi.fingerprint("fused_moe", act=GeGLU(), **dims) != bi.fingerprint(
        "fused_moe", act=GELU(), **dims
    )


def _adversarial_accumulators(
    limits: list[float], n_random: int, seed: int
) -> torch.Tensor:
    """fp32 accumulators that move the chains' branches: zeros of both signs, the smallest subnormals, +-1, small and
    large magnitudes up to the format's edge, the declared exp's clamp edges, the configuration's clamp limits with their
    binary32 neighbours, and randn values with spread exponents."""
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
            -(2.0**-127),
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
            89.0,
            -89.0,
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
def test_vectorised_and_exact_chains_agree_on_adversarial_accumulators(act):
    """The vectorised CPU chain (``glu_cpu_act`` / ``act_cpu`` through ``TorchF32``) and the exact one (``glu_elem`` /
    ``act_elem`` through ``ExactF32``: Fractions, one rounding per op) give the same bf16 bits on adversarial fp32
    accumulators -- zeros of both signs, subnormals, +-1, the exp clamp's edges, the configuration's clamp limits and
    their neighbours, magnitudes up to the format's edge, spread random values -- for every listed configuration. The
    vectorised chain is the GPU suites' oracle; this pins it to the ground truth."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    a = S.activation_of(act)
    limits = [p for p in a.params if p is not None]
    g = _adversarial_accumulators(limits, 160, seed=20)
    u = torch.roll(_adversarial_accumulators(limits, 160, seed=21), 5)
    if a.gated:
        vec = R.glu_cpu_act(g, u, a.act, a.params)
        exact = [
            S.glu_elem(R.ExactF32, gi, ui, a.act, "bf16", a.params)
            for gi, ui in zip(g.tolist(), u.tolist(), strict=True)
        ]
    else:
        vec = R.act_cpu(g, a.act, a.params)
        exact = [
            S.act_elem(R.ExactF32, gi, a.act, "bf16", a.params) for gi in g.tolist()
        ]
    assert vec.dtype == torch.bfloat16 and vec.shape == g.shape
    # the exact values are bf16 values already: the conversion is the identity
    ex = torch.tensor(exact, dtype=torch.float32).to(torch.bfloat16)
    bad = (bits(vec) != bits(ex)).nonzero().flatten()[:8].tolist()
    assert not bad, [
        (i, g[i].item(), u[i].item() if a.gated else None, vec[i].item(), ex[i].item())
        for i in bad
    ]


def _powlu_torch(gate: torch.Tensor, up: torch.Tensor, act: PowLU) -> torch.Tensor:
    """FlashInfer's torch reference has no PowLU branch: the activation's definition in torch,
    ``g > 0 ? g ** (m / (sqrt(g) + 1)) * sigmoid(g) : g * sigmoid(g)``, then the optional limit (the gate branch
    clamped to ``(-inf, limit]``, the linear one to ``[-limit, limit]``)."""
    pos = gate.clamp(min=0)
    powered = pos ** (act.m / (pos.sqrt() + 1))
    res = torch.where(
        gate > 0, powered * torch.sigmoid(gate), gate * torch.sigmoid(gate)
    )
    if act.limit is not None:
        res = res.clamp(max=act.limit)
        up = up.clamp(min=-act.limit, max=act.limit)
    return (res * up).to(torch.bfloat16)


@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_chains_are_plausible_against_the_torch_reference(act):
    """An observation, not a promise of the class: on moderate random accumulators the declared chain lands within a
    loose bound of FlashInfer's torch reference of the activation (``compute_reference_activation`` on the canonical
    ``[up, gate]`` layout; PowLU, which that reference lacks, written here from the activation's definition): max |error|
    at most the larger of 2 bf16 ulps of the reference's magnitude and 1e-2. The class's guarantee is the bitwise
    agreement with its own exact spec (the tests above and the GPU suites); this test only says the spec computes the
    activation it is named after."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S
    from tests.moe.utils import compute_reference_activation

    I, rows = 256, 48
    gen = torch.Generator().manual_seed(30)
    a = S.activation_of(act)
    if a.gated:
        values = torch.randn(rows, 2 * I, generator=gen) * 2  # [up, gate] in fp32
        up, gate = values.split(I, dim=-1)
        ours = R.glu_cpu_act(gate, up, a.act, a.params)
        ref = (
            _powlu_torch(gate, up, act)
            if isinstance(act, PowLU)
            else compute_reference_activation(values, act, I)
        )
    else:
        values = torch.randn(rows, I, generator=gen) * 2
        ours = R.act_cpu(values, a.act, a.params)
        ref = compute_reference_activation(values, act, I)
    assert ours.dtype == ref.dtype == torch.bfloat16 and ours.shape == ref.shape
    assert torch.isfinite(ours.float()).all()
    err = (ours.float() - ref.float()).abs()
    mag = ref.float().abs().clamp_min(2.0**-126)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)  # the bf16 ulp at the reference
    tol = torch.maximum(2 * ulp, torch.full_like(ulp, 1e-2))
    worst = (err / tol).max().item()
    assert worst <= 1.0, (act, worst, err.max().item())


def test_routing_method_map_matches_the_spec():
    """The FlashInfer routing methods the class implements map onto the spec's methods with the same semantics; the
    rest are absent (the runner refuses them for ``FromLogits``)."""
    from flashinfer.fused_moe.batch_invariant import spec as S

    m = S.METHOD_OF_FLASHINFER
    assert m[RoutingMethodType.Default] is S.SOFTMAX
    assert m[RoutingMethodType.Renormalize] is S.SOFTMAX_RENORM
    assert (
        m[RoutingMethodType.RenormalizeNaive] is S.SOFTMAX_RENORM
    )  # softmax -> top-k -> divide by the top-k sum: the same dataflow
    assert m[RoutingMethodType.Sigmoid] is S.SIGMOID
    assert m[RoutingMethodType.SigmoidRenorm] is S.SIGMOID_RENORM
    assert m[RoutingMethodType.DeepSeekV3] is S.SIGMOID_BIAS_RENORM_SCALED
    assert RoutingMethodType.Llama4 not in m and RoutingMethodType.TopK not in m


@pytest.mark.parametrize(
    "family,sm_count",
    [("sm_107", 212), ("sm_107", 204), ("sm_100", 148), ("sm_103", 152)],
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for every bench geometry and token count lies in the enumerated grouped space of its GEMM
    (the space the configuration-agreement guarantee is claimed over), for both classes, at the nominal parts."""
    from flashinfer.fused_moe.batch_invariant.schedule import (
        all_valid_configs,
        default_configs,
        rank_grouped,
        rows_per_expert,
    )

    geoms = [
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (5120, 8192, 16, 1),
        (512, 256, 64, 6),
        (1024, 512, 64, 8),
    ]
    for H, I, E, k in geoms:
        s1, s1u, s2 = (
            set(all_valid_configs(2 * I, H, E, "swiglu")),
            set(all_valid_configs(I, H, E, "relu2")),
            set(all_valid_configs(H, I, E, "plain")),
        )
        assert s1 and s1u and s2
        for T in (1, 7, 64, 300, 800, 4096, 16384):
            c1, c2 = default_configs(
                T, k, H, I, E, sm_count, family, True, nominal=True
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
            c1u, _ = default_configs(
                T, k, H, I, E, sm_count, family, False, nominal=True
            )
            assert c1u in s1u, (family, T, H, I, E, k, c1u.describe())
            ranked = rank_grouped(
                rows_per_expert(T, k, E),
                H,
                I,
                E,
                "plain",
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


def test_unsupported_device_raises_before_any_kernel():
    """The backend's runner on a CPU device fails its support check with the mode's error naming the class (no fallback, no
    kernel); the layer itself admits CUDA devices only, so the check is driven on the runner."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError
    from flashinfer.fused_moe.batch_invariant.runner import BatchInvariantMoeRunner

    runner = BatchInvariantMoeRunner(_config(8, 64, 256), torch.device("cpu"))
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        runner._check_support()
    assert "moe.bf16" in str(excinfo.value) and "cpu" in str(excinfo.value)


# ----------------------------------------------------------------------------- GPU


def _reference_route(
    logits: torch.Tensor, k: int, method
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    ex, w = R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])
    return ex, w


def _reference_ffn(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    act,
) -> torch.Tensor:
    """The class composed on the device from its own pieces: the class's permutation, GEMM1 as the grouped kernel with the
    identity epilogue in fp32 (the same fold; any record of the space gives its bits), the CPU chain ``act`` (an
    ``ActSpec``: the gated chain on the (gate, up) accumulators or the ungated one on the single accumulator) on those
    accumulators, GEMM2 as the dense batch-invariant GEMM per expert (the dense class, whose chunk rule gives one fold at
    this weight shape: asserted), the CPU slot-order combine. Bit for bit what the fused op must produce."""
    import flashinfer
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant.api import (
        grouped_gemm,
        interleave_fc1,
        permute_plan,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.epilogue import GROUP
    from flashinfer.fused_moe.batch_invariant.schedule import all_valid_configs
    from flashinfer.gemm.batch_invariant.schedule import chunks_for

    T, H = x.shape
    E, k = w1.shape[0], ids.shape[1]
    I = w2.shape[2]
    assert chunks_for(H, I) == 1
    w1i = interleave_fc1(w1) if act.gated else w1
    counts, dest, src = permute_plan(ids, E)
    n = T * k
    assert torch.equal(
        src[dest.view(-1).long()].cpu(), torch.arange(n, dtype=torch.int32)
    )
    x_perm = x[(src // k).long()].contiguous()
    acc = grouped_gemm(
        x_perm,
        w1i,
        counts,
        all_valid_configs(w1i.shape[1], H, E, "plain")[0],
        "plain",
        torch.float32,
    )
    if act.gated:
        u = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
        g = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
        a = R.glu_cpu_act(g.cpu(), u.cpu(), act.act, act.params).to(x.device)
    else:
        a = R.act_cpu(acc.cpu(), act.act, act.params).to(x.device)
    y_perm = torch.empty(n, H, dtype=torch.bfloat16, device=x.device)
    off = 0
    for e, c in enumerate(counts.tolist()):
        if c:
            y_perm[off : off + c] = flashinfer.mm_bf16(
                a[off : off + c].contiguous(),
                w2[e].t(),
                backend="auto",
                batch_invariant=True,
            )
        off += c
    y = y_perm[dest.view(-1).long()]  # back to (token, slot) order
    # the slot-order combine of the class on the per-slot expert outputs (rows in slot order, dest = the identity)
    ident = torch.arange(n, dtype=torch.int32).view(T, k)
    return R.combine_cpu(y.cpu(), ident, weights.cpu())


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU and the dense batch-invariant GEMM per expert with the
    CPU chain of the activation and the combine: the oracle ladder of the class, bit for bit, for every activation of the
    layer (gated weights ``[E, 2I, H]``, ungated ``[E, I, H]``)."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(x, logits), _weights(w1, w2, act))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ref = _reference_ffn(x, ids.cuda(), weights.cuda(), w1, w2, chain)
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
def test_the_largest_expert_count_routes_and_runs():
    """1024 experts (the routing kernel's widest bucket, the group tables' size): the class's routing from the logits
    equals the CPU model, and the layer equals the class composition at that expert count."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api import route

    T, k, H, I, E = 96, 8, 256, 128, 1024
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    ids, wts = route(logits, k)
    ref_ids, ref_w = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(ids.cpu(), ref_ids) and torch.equal(bits(wts.cpu()), bits(ref_w))
    layer = _layer(k, E, I)
    out = layer(_logits_pack(x, logits), _weights(w1, w2))
    ref = _reference_ffn(x, ids, wts, w1, w2, S.activation_of(_SWIGLU))
    assert torch.equal(bits(out.cpu()), bits(ref))


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their
    inputs (the packed mode's bf16-rounded weights are its inputs); repeated runs and a CUDA-graph replay reproduce them."""
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
    assert torch.equal(
        bits(out_u16), bits(layer(_routed_pack(x, ids.cuda(), w16.float().cuda()), W))
    )
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack(x, logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack(x, logits)
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
def test_routing_constants_first_made_under_graph_capture_are_not_cached():
    """A routing constant (the absent bias, the scale) first materialised while a CUDA graph is being captured is a
    graph-pool allocation whose fill runs only at replay: it serves that capture alone and is never cached, so an eager
    call after the capture gives the bits of an eager call before it, and the replay gives them too."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, E = 64, 8, 96
    g = torch.Generator().manual_seed(11)
    logits = (torch.randn(T, E, generator=g) * 2).cuda()
    bias = (torch.randn(E, generator=g) * 0.1).cuda()
    calls = (
        # the absent bias: a cached zero vector per expert count
        dict(method=S.METHOD_OF_FLASHINFER[RoutingMethodType.Renormalize]),
        # the scale given as a number: a cached one-element tensor
        dict(
            method=S.METHOD_OF_FLASHINFER[RoutingMethodType.DeepSeekV3],
            bias=bias,
            scaling_factor=2.5,
        ),
    )
    for kw in calls:
        ex0, w0 = api.route(logits, k, **kw)  # eager: the compile, the constants cached
        torch.cuda.synchronize()
        api._CONST.clear()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ex_g, w_g = api.route(logits, k, **kw)
        torch.cuda.synchronize()
        assert not api._CONST  # nothing made under capture is served to later calls
        ex1, w1 = api.route(logits, k, **kw)
        assert torch.equal(ex1, ex0) and torch.equal(bits(w1), bits(w0))
        ex_g.fill_(-1)
        w_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ex_g, ex0) and torch.equal(bits(w_g), bits(w0))


@needs_device
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens
    create: prefixes, permutations, sub-batches, one token alone, and a hot expert group around a probe token."""
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
    # a probe token among 556 others routed to the probe's own experts (one hot group), then among others on unrelated experts
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
    assert isinstance(runner, BatchInvariantMoeRunner)
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
    """Every routing method the class implements, from logits, selects and weights as the exact CPU model (the strict
    order on exact binary32 scores: ties in the logits included), and the layer's bits follow."""
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
        kwargs = {}
        if rm.scaled:
            kwargs["routed_scaling_factor"] = 1.75
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=method, **kwargs),
                quant=QuantConfig(),
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantBf16Config(),)),
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


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the classes raises the mode's error at build or at the first call
    (top_k / num_experts outside the bounds, fused shared experts, an expert-parallel shard, group-limited routing from
    logits, no finalize, an fp16 activation, weights of the other gating, an unimplemented routing method from logits); a
    non-default ``SwiGLU`` is a class of the backend (the ``swiglu_oai`` chain), not an error; a zero-token call returns
    an empty output."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    k, H, I, E = 6, 512, 256, 64
    errors = (BatchInvariantUnsupportedError, RuntimeError, NotImplementedError)
    oai = _layer(k, E, I, SwiGLU(alpha=1.702, beta=1.0, limit=7.0))
    assert isinstance(oai.runners[0], BatchInvariantMoeRunner)
    for cfg_kw, what in ((dict(k=17), "top_k"), (dict(E=1025), "num_experts")):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), I)
        assert what in str(excinfo.value) or "moe.bf16" in str(excinfo.value)
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
                    quant=QuantConfig(),
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantBf16Config(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.bf16" in str(excinfo.value), what
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=9)
    W = _weights(w1, w2)
    # an fp16 activation: no runner accepts the pack, nothing is computed
    with pytest.raises(errors):
        layer(_logits_pack(x.to(torch.float16), logits), W)
    # the ungated weights with the gated class: refused at the call, not computed
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(_logits_pack(x, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2()))
    # group-limited routing is refused from logits only: with pre-routed ids the group structure never enters the
    # computation, and the output equals the ungrouped configuration's on the same ids and weights
    grouped = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E,
                top_k=k,
                method=RoutingMethodType.DeepSeekV3,
                n_group=2,
                topk_group=1,
            ),
            quant=QuantConfig(),
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantBf16Config(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(BatchInvariantUnsupportedError) as excinfo:
        grouped(_logits_pack(x, logits), W)
    assert "n_group" in str(excinfo.value)
    g_ids, g_wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    assert torch.equal(
        bits(grouped(_routed_pack(x, g_ids.cuda(), g_wts.cuda()), W)),
        bits(layer(_routed_pack(x, g_ids.cuda(), g_wts.cuda()), W)),
    )
    # an unimplemented routing method from logits is refused; the same layer takes pre-routed inputs
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=QuantConfig(),
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantBf16Config(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(BatchInvariantUnsupportedError):
        llama(_logits_pack(x, logits), W)
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(_routed_pack(x, ids, torch.ones(8, 1, device="cuda")), W).shape == (
        8,
        H,
    )
    # zero tokens
    out = layer(_logits_pack(x[:0], logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


def _poison_allocator(byte: int):
    """Fill the caching allocator's cached blocks (the large pool and 256 small-pool segments) with one byte pattern, so the
    ``torch.empty`` intermediates of the next call start with it (0x7F / 0xFF: bf16 NaN patterns; 0x00: zeros)."""
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
@pytest.mark.parametrize("act", [_SWIGLU, ReLU2()], ids=["gated", "ungated"])
def test_poisoned_intermediates_do_not_reach_the_bits(act):
    """Unspecified bytes and timing never reach the output: the op's intermediates (the permuted rows, GEMM1's output, GEMM2's
    output, the output itself) come from ``torch.empty``; with the caching allocator's blocks filled with NaN patterns before
    every call and a bandwidth-bound kernel running on a side stream of the same device during the call, the layer gives the
    bits of a clean run, ``RLK_STRESS_ITERS`` times (default 20), at a shape whose partial tiles leave padding rows in every
    GEMM (the rows past a group's end that the epilogue skips or masks)."""
    import os

    T, k, H, I, E = 64, 8, 256, 64, 256
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=7, gated=act.is_gated, adversarial=True
    )
    layer, pack = _layer(k, E, I, act), _weights(w1, w2, act)
    ref = bits(layer(_logits_pack(x, logits), pack).cpu())
    side = torch.cuda.Stream()
    junk = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(int(os.environ.get("RLK_STRESS_ITERS", "20"))):
        for byte in (0x7F, 0xFF, 0x00):
            _poison_allocator(byte)
            with torch.cuda.stream(side):
                for _ in range(3):
                    junk.add_(1)
            out = layer(_logits_pack(x, logits), pack)
            assert torch.equal(bits(out.cpu()), ref), (i, hex(byte))
    torch.cuda.synchronize()


@needs_device
def test_combine_and_gather_warp_placements_keep_the_bits():
    """The combine's warps per token and the gather's warps per row are placements: every count gives the bits of one warp
    per token / row (each output vector's slot fold stays on one lane; the gather moves the same bytes), the gather is its
    definition ``x_perm[p] = x[src[p] // k]`` for bf16 and e4m3 rows, and the rules pick members of those sets at every T."""
    from flashinfer.fused_moe.batch_invariant.combine import (
        get_compiled_combine,
        warps_per_token,
    )
    from flashinfer.fused_moe.batch_invariant.permute import (
        get_compiled_gather,
        warps_per_row,
    )
    from flashinfer.gemm.batch_invariant.api import sm_count_of

    sms = sm_count_of(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(3)
    for T, k, H in (
        (1, 8, 2048),
        (7, 4, 512),
        (300, 8, 2048),
        (1000, 8, 7168),
        (4096, 2, 4096),
    ):
        Tk = T * k
        perm = torch.randperm(Tk, device="cuda", generator=g).to(torch.int32)
        dest = perm.view(T, k)
        y = torch.randn(Tk, H, device="cuda", generator=g).to(torch.bfloat16)
        w = torch.rand(T, k, device="cuda", generator=g)
        outs = []
        for wpt in (1, 2, 4, 8, 16):
            out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
            get_compiled_combine(k, "bf16", "bf16", wpt)(y, dest, w, out, T, H)
            outs.append(out)
        torch.cuda.synchronize()
        assert all(torch.equal(bits(outs[0]), bits(o)) for o in outs[1:]), (T, k, H)
        assert warps_per_token(T, H, sms) in (1, 2, 4, 8, 16)
        src = perm  # expanded row i = src[p] sits at position p: token i // k
        for fmt, x in (
            ("bf16", torch.randn(T, H, device="cuda", generator=g).to(torch.bfloat16)),
            (
                "e4m3",
                torch.randn(T, H, device="cuda", generator=g).to(torch.float8_e4m3fn),
            ),
        ):
            xps = []
            for wpr in (1, 2, 4, 8):
                xp = torch.empty(Tk, H, dtype=x.dtype, device="cuda")
                get_compiled_gather(k, fmt, wpr)(x, src, xp, Tk, H)
                xps.append(xp)
            torch.cuda.synchronize()
            assert torch.equal(bits(xps[0]), bits(x[(src // k).long()])), (fmt, T)
            assert all(torch.equal(bits(xps[0]), bits(xp)) for xp in xps[1:]), (fmt, T)
            assert warps_per_row(Tk, H // (16 // x.element_size()), sms) in (1, 2, 4, 8)


def _padded_plan_host(ex: torch.Tensor, E: int, align: int):
    """The padded permutation plan on the host: ``counts``, ``dest`` (the group offsets the exclusive prefix of the counts
    rounded up to ``align`` rows, the slots of a group in slot order) and ``src`` over the padded bound with ``-1`` in every
    padding slot -- the integers the block-scaled classes' padded plan defines."""
    T, k = ex.shape
    flat = ex.reshape(-1).cpu().long()
    n = flat.numel()
    counts = (
        torch.bincount(flat, minlength=E) if n else torch.zeros(E, dtype=torch.long)
    )
    padded = -(-counts // align) * align
    off = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    sorted_e = flat[order]
    group_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(n) - group_start[sorted_e]
    dest = torch.empty(n, dtype=torch.long)
    dest[order] = off[sorted_e] + rank
    n_pad = -(-(n + (align - 1) * min(E, n)) // align) * align if n else 0
    src = torch.full((n_pad,), -1, dtype=torch.long)
    src[dest] = torch.arange(n)
    return counts.int().cuda(), dest.view(T, k).int().cuda(), src.int().cuda()


@needs_device
@pytest.mark.parametrize("align", [1, 128])
@pytest.mark.parametrize(
    "T,k,E,method",
    [
        (1, 8, 128, "softmax_renorm"),
        (7, 4, 8, "softmax_renorm"),
        (16, 16, 1024, "softmax_renorm"),
        (13, 8, 128, "sigmoid_bias_renorm_scaled"),
        (0, 8, 128, "softmax_renorm"),
    ],
)
def test_route_plan_equals_route_and_plan(T, k, E, method, align):
    """``api.route_plan`` (the routing and the permutation plan in one launch for small token counts) gives the route kernel's
    bits and the plan's integers: experts, weights, counts, dest and src equal those of ``route`` followed by ``permute_plan``
    (``align`` 1) or by the padded plan of the block-scaled classes (``align`` 128: the group offsets at whole 128-row atoms,
    ``src`` over the padded bound with -1 in the padding slots, filled inside the launch)."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    meth = (
        S.SOFTMAX_RENORM if method == "softmax_renorm" else S.SIGMOID_BIAS_RENORM_SCALED
    )
    g = torch.Generator().manual_seed(T * 7 + E)
    logits = (torch.randn(max(T, 1), E, generator=g) * 2).float()[:T].cuda()
    bias = (torch.randn(E, generator=g) * 0.1).float().cuda() if meth.bias else None
    scale = 2.5 if meth.scaled else None
    ex, w = api.route(logits, k, meth, bias, scale)
    counts, dest, src = (
        api.permute_plan(ex, E) if align == 1 else _padded_plan_host(ex, E, align)
    )
    ex2, w2, counts2, dest2, src2 = api.route_plan(
        logits, k, meth, bias, scale, align=align
    )
    torch.cuda.synchronize()
    assert torch.equal(ex, ex2) and torch.equal(bits(w), bits(w2))
    assert (
        torch.equal(counts, counts2)
        and torch.equal(dest, dest2)
        and torch.equal(src, src2)
    )


@needs_device
def test_alternating_inputs_under_programmatic_dependent_launch():
    """Two problems run alternately through the op, back to back without host syncs (the kernels' PDL chains overlap: a
    kernel's prologue runs while its predecessor computes, and the allocator hands every reused intermediate the other
    call's bytes), many times; every output equals the problem's isolated result bit for bit. A missing or misplaced
    ``griddepcontrol.wait`` would read the other call's intermediates."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 8, 256, 64, 256
    probs = []
    for seed in (3, 11):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed, adversarial=True)
        ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
        ref = bits(api.fused_moe(x, ids, wts, (w1, w2)))
        torch.cuda.synchronize()
        probs.append((x, ids, wts, (w1, w2), ref))
    outs = []
    for _ in range(20):
        for x, ids, wts, w, _r in probs:
            outs.append(api.fused_moe(x, ids, wts, w))
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(bits(out), probs[i % 2][4]), i


@needs_device
@pytest.mark.parametrize(
    "T,k,E", [(3, 8, 64), (16, 16, 128), (16, 8, 1024), (0, 4, 16)]
)
def test_route_plan_dual_equals_route_and_the_dual_plan(T, k, E):
    """``api.route_plan(dual=True)``: the dense positions and the padded ones in one launch equal ``route`` followed by the
    block-scaled classes' ``permute_plan_dual`` (experts, weights, counts, dest, src, dest_pad, pos_pad)."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    permute_plan_dual = pytest.importorskip(
        "flashinfer.fused_moe.batch_invariant.api_nvfp4"
    ).permute_plan_dual  # the block-scaled package: a later PR

    g = torch.Generator().manual_seed(T * 3 + E)
    logits = (torch.randn(max(T, 1), E, generator=g) * 2).float()[:T].cuda()
    ex, w = api.route(logits, k, S.SOFTMAX_RENORM)
    counts, dest, src, dest_pad, pos_pad = permute_plan_dual(ex, E)
    ex2, w2, counts2, dest2, src2, dest_pad2, pos_pad2 = api.route_plan(
        logits, k, S.SOFTMAX_RENORM, dual=True
    )
    torch.cuda.synchronize()
    assert torch.equal(ex, ex2) and torch.equal(bits(w), bits(w2))
    assert torch.equal(counts, counts2) and torch.equal(dest, dest2)
    assert torch.equal(src, src2)
    assert torch.equal(dest_pad, dest_pad2) and torch.equal(pos_pad, pos_pad2)


@needs_device
def test_layer_fused_routing_at_small_t_matches_prerouted_routing():
    """From logits at a small token count the runner routes and plans in one launch; the output equals the pre-routed modes'
    (the same ids and weights from ``route``) bit for bit."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 5, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3)[:4]
    layer, W = _layer(k, E, I), _weights(w1, w2)
    out = layer(_logits_pack(x, logits), W)
    ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
    out2 = layer(_routed_pack(x, ids, wts), W)
    assert torch.equal(bits(out), bits(out2))
def test_pre_routed_ids_outside_the_expert_range_are_dropped_slots():
    """A pre-routed expert id outside ``[0, E)`` (``-1``, ``E``, ``2^30``: a framework's dropped slot) is dropped: the plan
    does not count it (``dest -1``, ``src -1`` past the valid rows), the gather and the GEMMs never touch it, and the
    combine folds the remaining slots in slot order with the term ``+0`` in its place, so the output equals the class's on
    the same rows with those slots at a valid id and weight ``+0``; a token whose slots are all dropped gives a zero row;
    the other tokens keep their bits. No host validation, no fault; the one-launch plan below ``PLAN_MULTI_MIN_N``
    expanded rows and the two-launch form (a partial final block) above it."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api import PLAN_MULTI_MIN_N

    k, H, I, E = 8, 256, 128, 64
    bad = (-1, E, 1 << 30)
    for T in (96, PLAN_MULTI_MIN_N // k + 3):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=21, adversarial=True)
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
        counts, dest, src = api.permute_plan(ids_bad.cuda(), E)
        n_valid = int((~dropped).sum())
        assert int(counts.sum()) == n_valid
        assert torch.equal(dest.cpu().view(-1) < 0, dropped.view(-1))
        assert torch.all(src[n_valid:] == -1) and torch.all(src[:n_valid] >= 0)
        valid_pos = dest.view(-1)[~dropped.view(-1).cuda()].long()
        assert torch.equal(
            src[valid_pos].cpu(), torch.nonzero(~dropped.view(-1)).view(-1).int()
        )
        layer, W = _layer(k, E, I), _weights(w1, w2)
        out = layer(_routed_pack(x, ids_bad.cuda(), wts.cuda()), W)
        ids_ref = torch.where(dropped, torch.zeros_like(ids), ids)
        wts_ref = torch.where(dropped, torch.zeros_like(wts), wts)
        ref = _reference_ffn(
            x, ids_ref.cuda(), wts_ref.cuda(), w1, w2, S.activation_of(_SWIGLU)
        )
        assert torch.equal(bits(out.cpu()), bits(ref))
        assert torch.all(out[rows[0]] == 0)
        whole = (~dropped.any(dim=1)).cuda()
        ref_whole = layer(_routed_pack(x, ids.cuda(), wts.cuda()), W)
        assert torch.equal(bits(out[whole]), bits(ref_whole[whole]))


def test_nan_logits_route_to_the_lowest_untaken_experts_with_nan_weights(E):
    """An expert whose selection score is NaN is never selected. A row with fewer than ``top_k`` non-NaN logits (one with
    none at all) gets the lowest expert indices it has not taken in its remaining slots, with NaN weights: every id stays
    in ``[0, E)`` (never the bucket's absent experts at ``E = 48``, never the argmax sentinel at ``E = 64``), the row's
    output is NaN, nothing faults, and the rows without a NaN keep the class's bits, in the fused route + plan form (a
    small token count) and the two-launch form; ``route_plan`` gives ``route`` + ``permute_plan``'s integers on such
    logits. A row with exactly ``top_k`` finite logits selects them in the strict order."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S

    k, H, I = 8, 256, 128
    keep = {
        1: 3,
        5: 0,
        9: 1,
        7: k,
    }  # token -> its finite logits (7: a boundary row, k of them)
    for T in (12, 40):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=5, adversarial=True)
        logits = logits.clone()
        finite = {}
        for t, m in keep.items():
            cols = torch.randperm(E, generator=torch.Generator().manual_seed(t))[:m]
            finite[t] = cols.tolist()
            mask = torch.ones(E, dtype=torch.bool)
            mask[cols] = False
            logits[t, mask.cuda()] = float("nan")
        ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
        ids_c, wts_c, logits_c = ids.cpu(), wts.cpu(), logits.cpu()
        assert torch.all((ids_c >= 0) & (ids_c < E))
        for t, cols in finite.items():
            order = sorted(cols, key=lambda e: (-float(logits_c[t, e]), e))
            m = len(cols)
            assert ids_c[t, :m].tolist() == order, t
            taken, pad = set(order), []
            for _ in range(k - m):
                pad.append(min(set(range(E)) - taken))
                taken.add(pad[-1])
            assert ids_c[t, m:].tolist() == pad, t
            if m < k:
                assert torch.isnan(wts_c[t]).all(), (
                    t
                )  # the renormalisation spreads the NaN over the row
        clean = torch.tensor([t not in keep for t in range(T)])
        ref_ids, ref_w = _reference_route(
            logits_c[clean], k, RoutingMethodType.Renormalize
        )
        assert torch.equal(ids_c[clean], ref_ids) and torch.equal(
            bits(wts_c[clean]), bits(ref_w)
        )
        ex2, w2_, counts2, dest2, src2 = api.route_plan(logits, k, S.SOFTMAX_RENORM)
        counts, dest, src = api.permute_plan(ids, E)
        assert torch.equal(ids, ex2) and torch.equal(bits(wts), bits(w2_))
        assert torch.equal(counts, counts2) and torch.equal(dest, dest2)
        assert torch.equal(src, src2)
        layer, W = _layer(k, E, I), _weights(w1, w2)
        out = layer(_logits_pack(x, logits), W)
        nan_rows = torch.tensor([keep.get(t, k) < k for t in range(T)]).cuda()
        assert torch.isnan(out[nan_rows]).all()
        ref = layer(_routed_pack(x[clean.cuda()], ref_ids.cuda(), ref_w.cuda()), W)
        assert torch.equal(bits(out[clean.cuda()]), bits(ref))


