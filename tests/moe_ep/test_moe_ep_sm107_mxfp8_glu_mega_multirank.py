"""4-GPU SM107 (Rubin) mxfp8 GLU mega multirank: MoEEpLayer vs torch oracle.

Runs ONLY under its own torchrun invocation (``run_tests.sh mega_sm107``)::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_sm107_mxfp8_glu_mega_multirank.py -v \
        -m "gpu_4 and arch_rubin"

Every rank builds the SAME global expert bank (fixed seed), the layer routes
real cross-rank EP traffic over the NVLink symmetric heap, and each rank's
output is checked against the pure-torch oracle evaluated over the full bank
for that rank's tokens (tolerance bands: the oracle emulates but does not
bit-match the in-kernel FC2-input requantization).
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe")

from flashinfer.moe_ep import (  # noqa: E402
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.core.kernel.registry import (  # noqa: E402
    _MEGA_KERNEL_REGISTRY,
    create_mega_kernel,
)
from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer  # noqa: E402

HIDDEN = 512
INTERMEDIATE = 256
NUM_EXPERTS = 8
TOP_K = 4
NUM_TOKENS = 96
MAX_TOKENS = 128


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


def _launcher_ranks() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world


def _global_problem(world_size: int):
    """Identical on every rank (fixed seed): full expert bank + per-rank tokens."""
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    w13 = (
        torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda",
            dtype=torch.float32, generator=gen,
        )
        * HIDDEN**-0.5
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda",
            dtype=torch.float32, generator=gen,
        )
        * INTERMEDIATE**-0.5
    ).to(torch.bfloat16)
    x = torch.randn(
        world_size, NUM_TOKENS, HIDDEN, device="cuda",
        dtype=torch.float32, generator=gen,
    ).to(torch.bfloat16)
    # Routing that guarantees cross-rank traffic: token 0 of every rank pins
    # one expert per EP rank; the rest is random distinct top-k.
    topk_ids = torch.stack(
        [
            torch.stack(
                [
                    torch.randperm(NUM_EXPERTS, device="cuda", generator=gen)[:TOP_K]
                    for _ in range(NUM_TOKENS)
                ]
            )
            for _ in range(world_size)
        ]
    ).to(torch.int32)
    experts_per_rank = NUM_EXPERTS // world_size
    pinned = torch.arange(world_size, device="cuda") * experts_per_rank
    topk_ids[:, 0, : min(TOP_K, world_size)] = pinned[: min(TOP_K, world_size)].to(
        torch.int32
    )
    topk_weights = torch.softmax(
        torch.randn(
            world_size, NUM_TOKENS, TOP_K, device="cuda",
            dtype=torch.float32, generator=gen,
        ),
        dim=-1,
    )
    return w13, w2, x, topk_ids, topk_weights


def _megakernel_config(**overrides) -> Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    return Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, top_k=TOP_K, **overrides
    )


def _torch_oracle(x_rank, topk_ids_rank, topk_weights_rank, w13, w2, apply_topk_in_fc1):
    """Full-bank oracle for one rank's tokens over the layer's exact quant path."""
    from flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe import (
        compute_megamoe_reference_sm107_glu,
        interleave_gate_up_32,
        quantize_mxfp8_block32,
    )

    x_q, x_sf = quantize_mxfp8_block32(x_rank.to(torch.float32), torch.float8_e4m3fn)
    w13_q, w13_sf = quantize_mxfp8_block32(
        interleave_gate_up_32(
            w13.to(torch.float32), intermediate_size=INTERMEDIATE
        ).contiguous(),
        torch.float8_e4m3fn,
    )
    w2_q, w2_sf = quantize_mxfp8_block32(
        w2.to(torch.float32).contiguous(), torch.float8_e4m3fn
    )
    return compute_megamoe_reference_sm107_glu(
        x_q,
        x_sf,
        topk_ids_rank,
        topk_weights_rank,
        w13_q.permute(0, 2, 1),
        w13_sf,
        w2_q.permute(0, 2, 1),
        w2_sf,
        local_expert_offset=0,
        gate_up_clamp=None,
        apply_topk_in_fc1=apply_topk_in_fc1,
    )


def test_sm107_mega_kernel_is_registered():
    assert "sm107_mxfp8_mxfp8_bf16_cutedsl" in _MEGA_KERNEL_REGISTRY
    backend = create_mega_kernel(_megakernel_config())
    assert backend.kernel_name() == "sm107_mxfp8_mxfp8_bf16_cutedsl"


@pytest.mark.arch_rubin
def test_sm107_preprocess_mega_weights_from_bf16():
    _require_cuda()
    from flashinfer.moe_ep import preprocess_sm107_mxfp8_glu_mega_weights

    w13 = torch.randn(2, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(2, HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    (fc1_w, fc1_sf), (fc2_w, fc2_sf) = preprocess_sm107_mxfp8_glu_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
    )
    assert fc1_w.shape == (2, HIDDEN, 2 * INTERMEDIATE)
    assert fc2_w.shape == (2, INTERMEDIATE, HIDDEN)
    assert fc1_w.dtype == torch.float8_e4m3fn
    assert fc1_sf.dtype == torch.float8_e8m0fnu


@pytest.mark.gpu_4
@pytest.mark.arch_rubin
@pytest.mark.parametrize("in_kernel_fc2_reduce", [False, True])
def test_moe_ep_sm107_glu_mega_multirank_torch_oracle(in_kernel_fc2_reduce):
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("requires torchrun with >= 2 ranks")

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    w13, w2, x, topk_ids, topk_weights = _global_problem(world_size)
    experts_per_rank = NUM_EXPERTS // world_size
    local = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)

    cfg = _megakernel_config(in_kernel_fc2_reduce=in_kernel_fc2_reduce)
    kernel = create_mega_kernel(cfg)
    runtime = bootstrap_moe_ep_runtime(bootstrap, kernel.runtime_requirements(bootstrap))
    try:
        mega = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size, rank=rank, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=NUM_EXPERTS,
                max_tokens_per_rank=MAX_TOKENS,
                token_hidden_size=HIDDEN,
            ),
            weights=MoEWeightPack(w13=w13[local].clone(), w2=w2[local].clone()),
            backend=MegaConfig(megakernel=cfg),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        for iteration in range(2):  # second forward = stale-state regression guard
            y = mega.forward(
                MoEEpTensors(
                    hidden_states=x[rank],
                    topk_ids=topk_ids[rank],
                    topk_weights=topk_weights[rank],
                )
            )
            y_ref = _torch_oracle(
                x[rank], topk_ids[rank], topk_weights[rank], w13, w2,
                apply_topk_in_fc1=cfg.apply_topk_in_fc1,
            )[:NUM_TOKENS]
            yk = y[:NUM_TOKENS].to(torch.float32)
            yr = y_ref.to(torch.float32)
            rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
            print(
                f"[sm107 multirank] rank={rank} ikr={in_kernel_fc2_reduce} "
                f"iter={iteration} rel_l2={rel_l2.item():.5f} "
                f"max|d|={(yk - yr).abs().max().item():.5f}"
            )
            assert rel_l2.item() < 0.02, (
                f"rank {rank} iter {iteration}: rel_l2 {rel_l2.item()} out of band"
            )
        mega.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)
