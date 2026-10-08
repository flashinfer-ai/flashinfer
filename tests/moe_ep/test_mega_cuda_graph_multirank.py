"""2-rank lockstep CUDA-graph capture/replay for the nvfp4 mega layer.

Multirank completion of ``test_mega_cuda_graph.py``: the mega kernel has cross-rank device-side barriers, so ranks must
replay together — this test warms up collectively, captures each rank's
``layer.forward`` (capture records without executing, so no cross-rank
dependency during capture), then replays in lockstep and compares against
the eager reference bit-exactly (default non-ikr config is deterministic).

Launched via torchrun (any even world size >= 2; runs at world size 2+):
    torchrun --nproc_per_node=2 -m pytest \\
        tests/moe_ep/test_mega_cuda_graph_multirank.py -v \\
        -m "gpu_2 and arch_blackwell"
"""

from __future__ import annotations

import pytest

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import (
    _assert_ikr_close,
    _launcher_ranks,
    _mega_problem,
    _require_cuda,
)

_REPLAYS = 4


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mode,token_back_mode,tuning,in_kernel_fc2_reduce,alpha_source",
    [
        ("w4a4", "epi_warps", "manual", False, "config"),
        ("w4a16", "epi_warps", "manual", False, "config"),
        ("w4a16", None, "auto", False, "config"),
        ("w4a16", "reuse_dispatch_warps", "manual", False, "runtime"),
        ("w4a16", "reuse_dispatch_warps", "manual", True, "runtime"),
    ],
)
def test_nvfp4_mega_two_rank_graph_replay_lockstep(
    mode, token_back_mode, tuning, in_kernel_fc2_reduce, alpha_source
):
    pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")
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
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        MoEWeightPack,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)

    assert mode in ("w4a4", "w4a16"), mode
    assert tuning in ("manual", "auto"), tuning
    assert tuning == "manual" or (mode == "w4a16" and token_back_mode is None)
    alphas = {}
    options = {}
    if mode == "w4a16":
        local_experts = problem["num_experts"] // world_size
        alphas = dict(
            fc1_alpha=torch.linspace(0.71013, 1.23017, local_experts, device="cuda"),
            fc2_alpha=torch.linspace(1.17019, 0.83023, local_experts, device="cuda"),
        )
        options = dict(
            enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            knobs="auto"
            if tuning == "auto"
            else {
                "token_back_mode": token_back_mode,
                "in_kernel_fc2_reduce": in_kernel_fc2_reduce,
            },
            **(alphas if alpha_source == "config" else {}),
        )
    configs = {
        "w4a4": Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        "w4a16": Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    }
    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=problem["num_experts"],
            max_tokens_per_rank=problem["max_tokens"],
            token_hidden_size=problem["hidden"],
        ),
        weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        backend=MegaConfig(
            megakernel=configs[mode](
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                gate_up_clamp=problem["gate_up_clamp"],
                **options,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)
    graph = None
    try:
        t = MoEEpTensors(
            hidden_states=problem["hidden_states"],
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            **(alphas if alpha_source == "runtime" else {}),
        )

        # Collective warmup: compile + workspace + one real launch, all ranks.
        mega.warmup(t)
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
        if in_kernel_fc2_reduce:
            _assert_ikr_close(y_graph, y_eager, topk=problem["topk"])
        else:
            assert torch.equal(y_graph, y_eager), (
                f"rank {rank}: lockstep graph replay diverged from eager"
            )

        if alpha_source == "runtime":
            # Change only scales, then replay before eager can pre-stage them.
            alpha_ptrs = (t.fc1_alpha.data_ptr(), t.fc2_alpha.data_ptr())
            t.fc1_alpha.mul_(0.5)
            t.fc2_alpha.mul_(1.25)
            assert (t.fc1_alpha.data_ptr(), t.fc2_alpha.data_ptr()) == alpha_ptrs
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            y_alpha_replay = y_graph.clone()
            y_alpha_eager = mega.forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            assert not torch.equal(y_alpha_eager, y_eager)
            if in_kernel_fc2_reduce:
                _assert_ikr_close(y_alpha_replay, y_alpha_eager, topk=problem["topk"])
            else:
                assert torch.equal(y_alpha_replay, y_alpha_eager)

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
        if in_kernel_fc2_reduce:
            _assert_ikr_close(y_replay, y_eager2, topk=problem["topk"])
        else:
            assert torch.equal(y_replay, y_eager2), (
                f"rank {rank}: replay-after-mutation diverged from eager"
            )
        if in_kernel_fc2_reduce:
            t.hidden_states.zero_()
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            assert torch.count_nonzero(y_graph) == 0
    finally:
        if graph is not None:
            graph.reset()
        mega.destroy()
        torch.cuda.synchronize()
        dist.barrier()
