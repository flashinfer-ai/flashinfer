"""4-GPU SM107 (Rubin) block-scaled mega multirank: MoEEpLayer vs torch oracle.

Runs ONLY under its own torchrun invocation (``run_tests.sh mega_sm107``)::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_sm107_block_scaled_mega_multirank.py -v \
        -m "gpu_4 and arch_rubin"

Every rank builds the SAME global expert bank (fixed seed), the layer routes
real cross-rank EP traffic over the NVLink symmetric heap, and each rank's
output is checked against the pure-torch oracle evaluated over the full bank
for that rank's tokens (tolerance bands: the oracle emulates but does not
bit-match the in-kernel FC2-input requantization).  Covers BOTH sm107
backends (mxfp8_e4m3 and nvfp4).
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe")

from flashinfer.moe_ep import (  # noqa: E402
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
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

# The nvfp4 wire is much coarser (4-bit data, per-16 fp8 scales through TWO
# GEMMs); the mxfp8 band matches the previous GLU-kernel test.
_REL_L2_BAND = {"mxfp8_e4m3": 0.02, "mxfp8_e5m2": 0.02, "nvfp4": 0.06}


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
            NUM_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        * HIDDEN**-0.5
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        * INTERMEDIATE**-0.5
    ).to(torch.bfloat16)
    x = torch.randn(
        world_size,
        NUM_TOKENS,
        HIDDEN,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
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
    pinned = list(range(0, NUM_EXPERTS, experts_per_rank))
    pinned.extend(e for e in range(NUM_EXPERTS) if e not in pinned)
    topk_ids[:, 0] = torch.tensor(pinned[:TOP_K], device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            world_size,
            NUM_TOKENS,
            TOP_K,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ),
        dim=-1,
    )
    return w13, w2, x, topk_ids, topk_weights


def _megakernel_config(quant_kind: str, **overrides):
    if quant_kind == "nvfp4":
        return Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=INTERMEDIATE, top_k=TOP_K, **overrides
        )
    return Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, top_k=TOP_K, kind=quant_kind, **overrides
    )


def _torch_oracle(
    x_rank, topk_ids_rank, topk_weights_rank, w13, w2, quant_kind, apply_topk_at_fc1
):
    """Full-bank oracle for one rank's tokens over the layer's exact quant path."""
    if x_rank.shape[0] == 0:
        return torch.empty((0, HIDDEN), device=x_rank.device, dtype=torch.bfloat16)
    import flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe as pkg

    w13_interleaved = pkg.interleave_gate_up_16(
        w13.to(torch.float32), intermediate_size=INTERMEDIATE
    ).contiguous()
    w2_f32 = w2.to(torch.float32).contiguous()
    if quant_kind == "nvfp4":
        x_q, x_sf = pkg.quantize_nvfp4_block16(x_rank.to(torch.float32))
        w13_q, w13_sf = pkg.quantize_nvfp4_block16(w13_interleaved)
        w2_q, w2_sf = pkg.quantize_nvfp4_block16(w2_f32)
    else:
        x_q, x_sf = pkg.quantize_mxfp8_block32(
            x_rank.to(torch.float32),
            (torch.float8_e4m3fn if quant_kind == "mxfp8_e4m3" else torch.float8_e5m2),
        )
        w13_q, w13_sf = pkg.quantize_mxfp8_block32(
            w13_interleaved,
            (torch.float8_e4m3fn if quant_kind == "mxfp8_e4m3" else torch.float8_e5m2),
        )
        w2_q, w2_sf = pkg.quantize_mxfp8_block32(
            w2_f32,
            (torch.float8_e4m3fn if quant_kind == "mxfp8_e4m3" else torch.float8_e5m2),
        )
    return pkg.compute_megamoe_reference_sm107_block_scaled(
        x_q,
        x_sf,
        topk_ids_rank,
        topk_weights_rank,
        w13_q,
        w13_sf,
        w2_q,
        w2_sf,
        quant_kind=quant_kind,
        local_expert_offset=0,
        gate_up_clamp=None,
        apply_topk_at_fc1=apply_topk_at_fc1,
    )


def test_sm107_mega_kernels_are_registered():
    assert "sm107_mxfp8_mxfp8_bf16_cutedsl" in _MEGA_KERNEL_REGISTRY
    assert "sm107_nvfp4_nvfp4_bf16_cutedsl" in _MEGA_KERNEL_REGISTRY
    for kind in ("mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"):
        backend = create_mega_kernel(_megakernel_config(kind))
        assert backend.kernel_name().startswith("sm107_")


@pytest.mark.arch_rubin
@pytest.mark.parametrize("quant_kind", ["mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"])
def test_sm107_preprocess_mega_weights_from_bf16(quant_kind):
    _require_cuda()
    from flashinfer.moe_ep import (
        preprocess_sm107_mxfp8_mega_weights,
        preprocess_sm107_nvfp4_mega_weights,
    )

    w13 = torch.randn(2, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(2, HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    pack = MoEWeightPack(w13=w13, w2=w2)
    if quant_kind == "nvfp4":
        (fc1_w, fc1_sf), (fc2_w, fc2_sf) = preprocess_sm107_nvfp4_mega_weights(
            pack, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )
        assert fc1_w.shape == (2, HIDDEN // 2, 2 * INTERMEDIATE)
        assert fc2_w.shape == (2, INTERMEDIATE // 2, HIDDEN)
        assert fc1_sf.dtype == torch.float8_e4m3fn
        assert fc2_sf.dtype == torch.float8_e4m3fn
    else:
        (fc1_w, fc1_sf), (fc2_w, fc2_sf) = preprocess_sm107_mxfp8_mega_weights(
            pack, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN, kind=quant_kind
        )
        assert fc1_w.shape == (2, HIDDEN, 2 * INTERMEDIATE)
        assert fc2_w.shape == (2, INTERMEDIATE, HIDDEN)
        assert fc1_w.dtype == (
            torch.float8_e4m3fn if quant_kind == "mxfp8_e4m3" else torch.float8_e5m2
        )
        assert fc2_w.dtype == (
            torch.float8_e4m3fn if quant_kind == "mxfp8_e4m3" else torch.float8_e5m2
        )
        assert fc1_sf.dtype == torch.float8_e8m0fnu
        assert fc2_sf.dtype == torch.float8_e8m0fnu


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "quant_kind, in_kernel_fc2_reduce, apply_topk_in_fc1",
    [
        (kind, ikr, early)
        for kind in ("mxfp8_e4m3", "mxfp8_e5m2", "nvfp4")
        for ikr, early in ((False, True), (False, False), (True, True))
    ],
)
def test_moe_ep_sm107_block_scaled_mega_multirank_torch_oracle(
    quant_kind, in_kernel_fc2_reduce, apply_topk_in_fc1
):
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("requires torchrun with >= 2 ranks")

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    w13, w2, x, topk_ids, topk_weights = _global_problem(world_size)
    experts_per_rank = NUM_EXPERTS // world_size
    local = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)

    cfg = _megakernel_config(
        quant_kind,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        apply_topk_in_fc1=apply_topk_in_fc1,
    )
    kernel = create_mega_kernel(cfg)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
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

        band = _REL_L2_BAND[quant_kind]
        # Changing routes and live counts, one idle rank, all ranks idle, then
        # nonzero reuse. Every rank still launches at every collective step.
        for iteration, live in enumerate(
            (NUM_TOKENS, 0 if rank == 0 else NUM_TOKENS - 3, 0, 1, NUM_TOKENS)
        ):
            current_x = x[rank, :live] * (1 + 0.1 * iteration)
            current_ids = (topk_ids[rank, :live] + iteration) % NUM_EXPERTS
            current_ids[::3, 0] = -1
            current_weights = topk_weights[rank, :live] * (1 + 0.2 * iteration)
            y = mega.forward(
                MoEEpTensors(
                    hidden_states=current_x,
                    topk_ids=current_ids,
                    topk_weights=current_weights,
                )
            )
            y_ref = _torch_oracle(
                current_x,
                current_ids,
                current_weights,
                w13,
                w2,
                quant_kind,
                apply_topk_at_fc1=cfg.apply_topk_in_fc1,
            )[:live]
            yk = y[:live].to(torch.float32)
            yr = y_ref.to(torch.float32)
            rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
            print(
                f"[sm107 multirank] rank={rank} kind={quant_kind} "
                f"ikr={in_kernel_fc2_reduce} iter={iteration} "
                f"rel_l2={rel_l2.item():.5f} "
                f"max|d|={(yk - yr).abs().max().item() if live else 0:.5f}"
            )
            assert rel_l2.item() < band, (
                f"rank {rank} iter {iteration}: rel_l2 {rel_l2.item()} out of "
                f"band {band}"
            )
        mega.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_sm107_pooled_layers_graph_rebinds_weights_and_stream(kind):
    import torch.distributed as dist

    rank, world = _launcher_ranks()
    if world < 2:
        pytest.skip("requires torchrun with at least two ranks")
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    w13, w2, x, ids, scores = _global_problem(world)
    local_experts = NUM_EXPERTS // world
    local = slice(rank * local_experts, (rank + 1) * local_experts)
    cfg = _megakernel_config(kind)
    kernel = create_mega_kernel(cfg)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layers = []
    try:
        banks = [(w13, w2), (-w13, w2 * 0.75)]
        tensors = MoEEpTensors(x[rank].clone(), ids[rank].clone(), scores[rank].clone())
        for a, b in banks:
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    rank=rank, world_size=world, auto_bootstrap=False
                ),
                fleet_params=FleetParams(
                    num_experts=NUM_EXPERTS,
                    max_tokens_per_rank=MAX_TOKENS,
                    token_hidden_size=HIDDEN,
                ),
                weights=MoEWeightPack(w13=a[local].clone(), w2=b[local].clone()),
                backend=MegaConfig(megakernel=cfg),
            )
            layers.append(layer)
            layer.warmup(tensors)
        assert layers[0]._workspace is layers[1]._workspace
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs = [layer.forward(tensors) for layer in layers]
        for iteration in range(3):
            tensors.hidden_states.copy_(x[rank] * (1 + iteration * 0.2))
            tensors.topk_ids.copy_((ids[rank] + iteration) % NUM_EXPERTS)
            tensors.topk_ids[::3, 0] = -1
            tensors.topk_weights.copy_(scores[rank] * (1 + iteration * 0.1))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            for output, (a, b) in zip(outputs, banks, strict=False):
                reference = _torch_oracle(
                    tensors.hidden_states,
                    tensors.topk_ids,
                    tensors.topk_weights,
                    a,
                    b,
                    kind,
                    True,
                )
                error = (
                    output.float() - reference.float()
                ).norm() / reference.float().norm().clamp_min(1e-6)
                assert (
                    torch.isfinite(output).all() and float(error) < _REL_L2_BAND[kind]
                )
        del graph, outputs
    finally:
        for layer in layers:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
