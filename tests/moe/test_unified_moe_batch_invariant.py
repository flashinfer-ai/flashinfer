# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MoE backend of the unified MoE layer (``BatchInvariantMoeConfig``): the bf16 classes
``moe.bf16.swiglu`` and ``moe.bf16.relu2``.

The class has exact CPU models of everything but the tensor-core step (the routing, the activation chain on the fp32
accumulators, the slot-order combine), so the tests are bit comparisons: the layer's output against the class's own
composition on the CPU (routing on the CPU, the grouped GEMMs checked against the dense batch-invariant GEMM per expert),
run-to-run determinism including CUDA-graph replay, batch / permutation / sub-batch invariance, agreement of every tactic
the autotuner may pick, the three routing input modes giving the same bits, and the error contract (an unsupported
configuration raises before any kernel runs). The device-free tests pin the registration, the fingerprint dimensions and
the dispatcher's membership in the enumerated space.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    BackendOptions,
    BatchInvariantMoeConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
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
        backend=BackendOptions((BatchInvariantMoeConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )
    return cfg


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantMoeConfig.prepare_weights(
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
    reached by naming it), and declares the three routing modes and the two activation classes of this PR."""
    assert BatchInvariantMoeConfig in ALL_BACKEND_CONFIGS
    assert _BACKEND_RUNNERS[BatchInvariantMoeConfig] is BatchInvariantMoeRunner
    assert not any(
        isinstance(c, BatchInvariantMoeConfig) for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantMoeConfig.supported(a)
    ] == [100, 103, 107]
    assert set(BatchInvariantMoeRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert set(BatchInvariantMoeRunner.supported_activation_classes) == {SwiGLU, ReLU2}
    assert repr(BatchInvariantMoeConfig()) == "BatchInvariantMoeConfig()"


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe", ...)`` takes the geometry, the routing method and the activation; the gated and the
    ungated chain are two classes; an unimplemented routing method is refused; the routing method is a dimension."""
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
        bi.fingerprint("fused_moe", H=2048, I=768, top_k=8, num_experts=128, act="gelu")
    assert bi.class_name("fused_moe") == "moe.bf16"


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
    gated: bool,
) -> torch.Tensor:
    """The class composed on the device from its own pieces: the class's permutation, GEMM1 as the grouped kernel with the
    identity epilogue in fp32 (the same fold; any record of the space gives its bits), the CPU chain on those
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
    w1i = interleave_fc1(w1) if gated else w1
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
    if gated:
        u = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
        g = acc.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
        a = R.swiglu_cpu(g.cpu(), u.cpu()).to(x.device)
    else:
        a = R.relu2_cpu(acc.cpu()).to(x.device)
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
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU and the dense batch-invariant GEMM per expert with the
    CPU chain and combine: the oracle ladder of the class, bit for bit."""
    T, k, H, I, E = 300, 6, 512, 256, 64
    gated = isinstance(act, SwiGLU)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=1, gated=gated, adversarial=True)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(x, logits), _weights(w1, w2, act))
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ref = _reference_ffn(x, ids.cuda(), weights.cuda(), w1, w2, gated)
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
                backend=BackendOptions((BatchInvariantMoeConfig(),)),
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
    """The error contract: a configuration outside the class raises the mode's error at build or at the first call; a
    zero-token call returns an empty output; a wrong-shaped weight view is refused."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    k, H, I, E = 6, 512, 256, 64
    for cfg_kw, what in (
        (dict(act=SwiGLU(alpha=1.702)), "SwiGLU parameters"),
        (dict(k=17), "top_k"),
        (dict(E=600), "num_experts"),
    ):
        with pytest.raises((BatchInvariantUnsupportedError, RuntimeError)) as excinfo:
            kk = cfg_kw.get("k", k)
            EE = cfg_kw.get("E", E)
            _layer(kk, EE, I, cfg_kw.get("act", SwiGLU()))
        assert what in str(excinfo.value) or "moe.bf16" in str(excinfo.value)
    with pytest.raises((BatchInvariantUnsupportedError, RuntimeError)):
        MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k),
                quant=QuantConfig(),
                experts=ExpertConfig(intermediate_size=I),
                backend=BackendOptions((BatchInvariantMoeConfig(),)),
                finalize=MoEFinalizeConfig(do_finalize=False),
            ),
            device=torch.device("cuda"),
        )
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=9)
    W = _weights(w1, w2)
    # the ungated weights with the gated class: refused at the call, not computed
    with pytest.raises(BatchInvariantUnsupportedError):
        layer(_logits_pack(x, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2()))
    # an unimplemented routing method from logits is refused; the same layer takes pre-routed inputs
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=QuantConfig(),
            experts=ExpertConfig(intermediate_size=I),
            backend=BackendOptions((BatchInvariantMoeConfig(),)),
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
