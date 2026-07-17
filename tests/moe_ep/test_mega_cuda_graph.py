"""CUDA graph capture/replay for the cutedsl mega paths (single rank).

Contract under test:

1. ``MoEEpMegaLayer.warmup()`` forces every lazy host-side step (workspace
   alloc, ``cute.compile``, one real launch) eagerly.
2. After warmup, ``layer.forward`` is capturable in a ``torch.cuda.CUDAGraph``
   and replays match eager forwards bit-exactly (default non-ikr configs are
   deterministic), including replays over in-place-mutated input buffers.
3. Host-side compile/alloc/free paths raise loudly if they would fire
   mid-capture instead of silently corrupting the graph.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_mega_cuda_graph.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def _require_blackwell():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(f"cutedsl mega kernels need sm_100/sm_103; got sm_{cap[0]}{cap[1]}")


def _single_rank_layer(backend_name: str):
    """MoEEpMegaLayer on one rank (MEGA_NO_DIST) with bf16 staging."""
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Mxfp8CutedslMegaMoeConfig,
        Nvfp4CutedslMegaMoeConfig,
    )

    hidden = 2048
    intermediate = 1024
    num_experts = 4
    topk = 4
    max_tokens = 64

    g = torch.Generator(device="cuda").manual_seed(21)
    w13 = torch.randn(
        num_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )

    if backend_name == "nvfp4":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate, top_k=topk, gate_up_clamp=10.0
        )
    else:
        mk = Mxfp8CutedslMegaMoeConfig(
            intermediate_size=intermediate, top_k=topk, gate_up_clamp=10.0
        )

    layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=mk,
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    return layer, dict(
        hidden=hidden, num_experts=num_experts, topk=topk, max_tokens=max_tokens
    )


def _random_batch(problem: dict, *, seed: int, num_tokens: int = 32):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens,
        problem["hidden"],
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    scores = torch.randn(
        num_tokens,
        problem["num_experts"],
        dtype=torch.float32,
        device="cuda",
        generator=g,
    )
    topk_weights, topk_ids = torch.topk(
        scores, problem["topk"], dim=-1, largest=True, sorted=False
    )
    return MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int64),
        topk_weights=topk_weights.to(torch.float32),
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("backend_name", ["nvfp4", "mxfp8"])
def test_mega_layer_graph_capture_replay_matches_eager(monkeypatch, backend_name):
    import torch

    _require_blackwell()

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer(backend_name)
    try:
        t = _random_batch(problem, seed=3)

        # 1) Warmup contract: all lazy host-side work happens here, eagerly.
        layer.warmup()

        # Eager reference on the real batch (also proves warmup's dummy batch
        # left the layer in a working steady state).
        y_eager = layer.forward(t).clone()
        torch.cuda.synchronize()

        # 2) Capture one forward. The graph's input buffers are t's tensors
        # (stage_inputs reads them); its output is the tensor forward()
        # returned at capture time.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = layer.forward(t)

        # 3) Replay on the captured inputs: must reproduce eager bit-exactly
        # (default configs are deterministic; ikr would need tolerance).
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_graph, y_eager)

        # 4) Replay after mutating the input buffers in place: graph output
        # must track new data and match a fresh eager forward.
        t2 = _random_batch(problem, seed=11)
        t.hidden_states.copy_(t2.hidden_states)
        t.topk_ids.copy_(t2.topk_ids)
        t.topk_weights.copy_(t2.topk_weights)

        graph.replay()
        torch.cuda.synchronize()
        y_replay = y_graph.clone()

        y_eager2 = layer.forward(t)
        torch.cuda.synchronize()
        assert torch.equal(y_replay, y_eager2)
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
def test_mega_layer_capture_without_warmup_raises(monkeypatch):
    """Lazy workspace alloc inside capture must fail loudly, not corrupt."""
    import torch

    _require_blackwell()

    from flashinfer.moe_ep import MoEEpConfigError

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer("nvfp4")
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


@pytest.mark.arch_blackwell
def test_mega_compute_output_none_returns_workspace_view(monkeypatch):
    """compute(output=None) is a zero-copy view, bit-equal to the copy path."""
    import torch

    _require_blackwell()

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer("nvfp4")
    try:
        layer.warmup()
        t = _random_batch(problem, seed=42)
        y_copy = layer.forward(t).clone()
        torch.cuda.synchronize()

        kernel, ws, transformed = layer._kernel, layer._workspace, layer._transformed
        kernel.stage_inputs(t, ws, quantize_input=True)
        view = kernel.compute(ws, transformed, output=None)
        torch.cuda.synchronize()
        assert view.data_ptr() == ws.output_activation.data_ptr()
        assert torch.equal(view, y_copy)
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
def test_mega_layer_multi_size_graphs_and_eager_interleave(monkeypatch):
    """Engine pattern: one graph per batch size + eager calls, interleaved.

    Regression test for the tail-mask memo bug: a small-size graph replayed
    after a larger one (or an eager call after replays) must not leave the
    larger batch's rows live in the shared workspace — stale live rows
    dispatch garbage work and can change outputs.
    """
    import torch

    _require_blackwell()

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    layer, problem = _single_rank_layer("nvfp4")
    try:
        layer.warmup()
        t64 = _random_batch(problem, seed=51, num_tokens=64)
        t32 = _random_batch(problem, seed=52, num_tokens=32)

        y64_ref = layer.forward(t64).clone()
        y32_ref = layer.forward(t32).clone()
        torch.cuda.synchronize()

        g64 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g64):
            y64_g = layer.forward(t64)
        g32 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g32):
            y32_g = layer.forward(t32)

        # Small replay AFTER large replay: the 64-row staging must not leak
        # into the 32-row step.
        g64.replay()
        g32.replay()
        torch.cuda.synchronize()
        assert torch.equal(y32_g, y32_ref), "g32 after g64 diverged (stale rows)"
        g32.replay()
        g64.replay()
        torch.cuda.synchronize()
        assert torch.equal(y64_g, y64_ref)

        # Eager after replays: host-side state must not trust counts the
        # replays never updated.
        g64.replay()
        y32_eager = layer.forward(t32)
        torch.cuda.synchronize()
        assert torch.equal(y32_eager, y32_ref), "eager after replay diverged"
    finally:
        layer.destroy()
