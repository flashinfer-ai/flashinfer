"""2-rank lockstep CUDA-graph capture/replay for the nvfp4 mega layer.

Multirank completion of ``test_mega_cuda_graph.py`` (todo_cuda_graph.md
item 5): the mega kernel has cross-rank device-side barriers, so ranks must
replay together — this test warms up collectively, captures each rank's
``layer.forward`` (capture records without executing, so no cross-rank
dependency during capture), then replays in lockstep and compares against
the eager reference bit-exactly (default non-ikr config is deterministic).

Launched via torchrun (any even world size >= 2; runs at world size 2+):
    torchrun --nproc_per_node=2 -m pytest \\
        tests/moe_ep/test_mega_cuda_graph_multirank.py -v \\
        -m "gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import pytest

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import (
    _launcher_ranks,
    _mega_problem,
    _require_cuda,
)

_REPLAYS = 4


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_nvfp4_mega_two_rank_graph_replay_lockstep():
    pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("needs >=2 ranks")

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        Nvfp4CutedslMegaMoeConfig,
        MoEWeightPack,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)

    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=problem["num_experts"],
            max_tokens_per_rank=problem["max_tokens"],
            token_hidden_size=problem["hidden"],
        ),
        weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        backend=MegaConfig(
            megakernel=Nvfp4CutedslMegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                gate_up_clamp=problem["gate_up_clamp"],
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)
    try:
        t = MoEEpTensors(
            hidden_states=problem["hidden_states"],
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
        )

        # Collective warmup: compile + workspace + one real launch, all ranks.
        mega.warmup()
        dist.barrier()

        y_eager = mega.forward(t).clone()
        torch.cuda.synchronize()
        dist.barrier()

        # Capture records without executing — safe per-rank; barrier keeps
        # ranks aligned so no rank starts REPLAYING while a peer captures.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = mega.forward(t)
        dist.barrier()

        # Lockstep replays: the kernel's cross-rank barriers require every
        # rank to replay the same iteration together.
        for _ in range(_REPLAYS):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        assert torch.equal(y_graph, y_eager), (
            f"rank {rank}: lockstep graph replay diverged from eager"
        )

        # Replay over mutated inputs (fresh values, same buffers).
        g = torch.Generator(device="cuda").manual_seed(1234 + rank)
        t.hidden_states.copy_(
            torch.randn(
                *t.hidden_states.shape,
                dtype=t.hidden_states.dtype,
                device="cuda",
                generator=g,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        y_replay = y_graph.clone()

        y_eager2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.equal(y_replay, y_eager2), (
            f"rank {rank}: replay-after-mutation diverged from eager"
        )
    finally:
        mega.destroy()
        torch.cuda.synchronize()
        dist.barrier()
