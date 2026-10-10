"""CUDA graph capture/replay for the SM90 pull-style mega path (single rank).

Hopper counterpart of ``test_mega_cuda_graph.py`` for
``sm90_fp8_fp8_bf16_pull_cutedsl``, ``sm90_bf16_bf16_bf16_pull_cutedsl`` and
``sm90_bf16_nvfp4_bf16_pull_cutedsl``: after ``MoEEpMegaLayer.warmup()``,
``layer.forward`` captures into a ``torch.cuda.CUDAGraph`` and replays match
eager forwards bit-exactly (the default separate-reduce path is
deterministic), including replays over in-place-mutated inputs with masked
(``-1``) routes (and, for W4A16, mutated runtime alphas), and per-size graphs
interleaved with eager calls.

Process isolation: imports the SM90 kernel tree, which is mutually exclusive
with the SM100 tree per process -- excluded from run_tests.sh's ``unit``
target and run by the ``oracle_sm90`` target.  Directly, on one Hopper GPU::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_sm90_pull_mega_cuda_graph.py -v -m arch_hopper
"""

from __future__ import annotations

import pytest

E4M3_MAX = 448.0
# Static per-tensor calibration, identical on every rank by contract (see
# test_moe_ep_sm90_pull_fp8_mega_multirank.py).
ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)


def _require_sm90_tree():
    import torch

    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")
    try:
        import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel  # noqa: F401
    except RuntimeError as exc:
        pytest.skip(f"SM90 kernel tree unavailable in this process: {exc}")


def _single_rank_layer(
    *,
    compute: str,
    swap_ab: bool,
    weights: str,
    hidden: int = 1024,
    intermediate: int = 512,
):
    """MoEEpMegaLayer on one rank (MEGA_NO_DIST) with bf16 staging.

    ``compute``: ``"per_tensor"`` / ``"blockwise"`` select the FP8 backend's
    scale mode; ``"bf16"`` selects ``sm90_bf16_bf16_bf16_pull_cutedsl`` and
    ``"w4a16"`` ``sm90_bf16_nvfp4_bf16_pull_cutedsl`` (swap-AB only).
    ``weights``: checkpoint format, ``"bf16"``, ``"mxfp8"`` (FP8 only) or
    ``"nvfp4"`` (W4A16 only; per-expert alphas on the config).
    """
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
        Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
        Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
    )

    from ._mxfp8_reference import mxfp8_quantize_ref

    num_experts, topk, max_tokens = 8, 4, 64
    g = torch.Generator(device="cuda").manual_seed(17)
    w13 = torch.randn(
        num_experts, 2 * intermediate, hidden, device="cuda", generator=g
    ) * (hidden**-0.5)
    w2 = torch.randn(num_experts, hidden, intermediate, device="cuda", generator=g) * (
        intermediate**-0.5
    )
    alphas = (None, None)
    if weights == "mxfp8":
        (w13, w13_scale), (w2, w2_scale) = (
            mxfp8_quantize_ref(w13),
            mxfp8_quantize_ref(w2),
        )
        pack = MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale)
    elif weights == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
            quantize_nvfp4,
        )

        (p13, s13, a13), (p2, s2, a2) = quantize_nvfp4(w13), quantize_nvfp4(w2)
        pack = MoEWeightPack(
            w13=p13,
            w2=p2,
            w13_scale=s13.view(torch.float8_e4m3fn),
            w2_scale=s2.view(torch.float8_e4m3fn),
        )
        alphas = (a13, a2)
    else:
        pack = MoEWeightPack(w13=w13.to(torch.bfloat16), w2=w2.to(torch.bfloat16))

    if compute == "w4a16":
        megakernel = Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            swap_ab=swap_ab,
            gate_up_clamp=10.0,
            fc1_alpha=alphas[0],
            fc2_alpha=alphas[1],
        )
    elif compute == "bf16":
        megakernel = Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            swap_ab=swap_ab,
            gate_up_clamp=10.0,
        )
    else:
        megakernel = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            fp8_scale_mode=compute,
            swap_ab=swap_ab,
            gate_up_clamp=10.0,
            fc1_activation_dequant_scale=ACT_SCALE,
            fc2_activation_dequant_scale=ACT_SCALE,
        )
    layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=pack,
        backend=MegaConfig(
            megakernel=megakernel,
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    return layer, dict(hidden=hidden, num_experts=num_experts, topk=topk)


def _random_batch(problem: dict, *, seed: int, num_tokens: int = 32, masked=False):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(num_tokens, problem["num_experts"], device="cuda", generator=g)
    topk_weights, topk_ids = torch.topk(scores, problem["topk"], dim=-1, sorted=False)
    if masked:
        topk_ids[::3, 0] = -1
        topk_ids[1::4, -1] = -1
    return MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int64),
        topk_weights=torch.softmax(topk_weights, dim=-1),
    )


_CONFIGS = [
    ("per_tensor", False, "bf16"),
    ("per_tensor", True, "bf16"),
    ("blockwise", False, "bf16"),
    ("blockwise", True, "bf16"),
    ("blockwise", False, "mxfp8"),
    ("blockwise", True, "mxfp8"),
    ("bf16", False, "bf16"),
    ("bf16", True, "bf16"),
    ("w4a16", True, "bf16"),
    ("w4a16", True, "nvfp4"),
]


@pytest.mark.arch_hopper
@pytest.mark.parametrize("compute,swap_ab,weights", _CONFIGS)
def test_sm90_pull_graph_capture_replay_matches_eager(
    monkeypatch, compute, swap_ab, weights
):
    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer(
        compute=compute, swap_ab=swap_ab, weights=weights
    )
    graph = None
    try:
        layer.warmup()
        t = _random_batch(problem, seed=3)
        y_eager = layer.forward(t).clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = layer.forward(t)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_graph, y_eager)

        # In-place new data, now with masked routes: the replay must track
        # the new routing (stale combine rows of now-masked slots excluded)
        # and match a fresh eager forward.
        t2 = _random_batch(problem, seed=11, masked=True)
        t.hidden_states.copy_(t2.hidden_states)
        t.topk_ids.copy_(t2.topk_ids)
        t.topk_weights.copy_(t2.topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        y_replay = y_graph.clone()
        assert not torch.equal(y_replay, y_eager)
        y_eager2 = layer.forward(t2)
        torch.cuda.synchronize()
        assert torch.equal(y_replay, y_eager2)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_graph_tracks_runtime_alpha(monkeypatch):
    """Runtime alphas are staged inside the graph: in-place updates apply on replay."""
    import dataclasses

    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer(compute="w4a16", swap_ab=True, weights="nvfp4")
    graph = None
    try:
        layer.warmup()
        num_experts = problem["num_experts"]
        alpha1 = torch.full((num_experts,), 0.5, device="cuda")
        alpha2 = torch.linspace(0.25, 2.0, num_experts, device="cuda")
        t = dataclasses.replace(
            _random_batch(problem, seed=21), fc1_alpha=alpha1, fc2_alpha=alpha2
        )
        y_eager = layer.forward(t).clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = layer.forward(t)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_graph, y_eager)

        alpha1.mul_(1.5)
        alpha2.copy_(alpha2.flip(0))
        graph.replay()
        torch.cuda.synchronize()
        y_replay = y_graph.clone()
        assert not torch.equal(y_replay, y_eager)
        y_eager2 = layer.forward(t)
        torch.cuda.synchronize()
        assert torch.equal(y_replay, y_eager2)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_capture_without_warmup_raises(monkeypatch):
    """Lazy workspace alloc inside capture must fail loudly, not corrupt."""
    import torch

    from flashinfer.moe_ep import MoEEpConfigError

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer(
        compute="blockwise", swap_ab=False, weights="bf16"
    )
    try:
        t = _random_batch(problem, seed=5)
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(MoEEpConfigError, match="warmup"),
            torch.cuda.graph(graph),
        ):
            layer.forward(t)
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "compute,weights",
    [
        ("blockwise", "bf16"),
        ("blockwise", "mxfp8"),
        ("bf16", "bf16"),
        ("w4a16", "nvfp4"),
    ],
)
def test_sm90_pull_multi_size_graphs_and_eager_interleave(
    monkeypatch, compute, weights
):
    """One graph per batch size plus eager calls, interleaved on one workspace."""
    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer(compute=compute, swap_ab=True, weights=weights)
    try:
        layer.warmup()
        t64 = _random_batch(problem, seed=51, num_tokens=64)
        t7 = _random_batch(problem, seed=52, num_tokens=7, masked=True)
        y64_ref = layer.forward(t64).clone()
        y7_ref = layer.forward(t7).clone()
        torch.cuda.synchronize()

        g64 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g64):
            y64_g = layer.forward(t64)
        g7 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g7):
            y7_g = layer.forward(t7)

        g64.replay()
        g7.replay()
        torch.cuda.synchronize()
        assert torch.equal(y7_g, y7_ref), "g7 after g64 diverged (stale rows)"
        g7.replay()
        g64.replay()
        torch.cuda.synchronize()
        assert torch.equal(y64_g, y64_ref)

        g64.replay()
        y7_eager = layer.forward(t7)
        torch.cuda.synchronize()
        assert torch.equal(y7_eager, y7_ref), "eager after replay diverged"
    finally:
        layer.destroy()
