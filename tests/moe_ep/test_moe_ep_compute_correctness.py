"""Multi-GPU functional correctness for LL EXPERT_MAJOR + RANK_MAJOR (bf16).

Unlike the latency benchmark (random weights, no accuracy check) and the
identity-roundtrip test (no compute), this exercises the *real* expert FFN and
checks numerics: each rank runs ``dispatch → compute → combine`` over its own
128 tokens and compares the combined output against a single-process dense MoE
reference computed from the **full** (replicated) expert weights.

Why no cross-rank gather is needed: the EP combined output for a token is the
sum over its top-k experts — wherever they live across ranks — of
``topk_weight · FFN_e(token)``. Since every rank holds the full weight set (same
seed → identical on all ranks), each rank can compute that full reference for its
own tokens locally and compare. Both LL layouts must reproduce it:
  * EXPERT_MAJOR — compute is top_k=1 per dispatched row; combine reweights on recv.
  * RANK_MAJOR   — compute runs at the real top_k over received routing (non-local
    picks masked to 0); combine sums across ranks.

Config (per the request): num_experts=16, top_k=8, 128 tokens/rank, hidden=8192,
intermediate=2048, world=8, bf16. (NVFP4 variants are a planned follow-up — the
reference + harness here are written to extend to it.)

Launch (Pre-Nyx single node, 8× B200; intra-tray NVLink, no MNNVL):
    torchrun --nproc_per_node=8 -m pytest \\
        tests/moe_ep/test_moe_ep_compute_correctness.py -v -s -m "nvep and gpu_8"

Or directly (no pytest):
    torchrun --nproc_per_node=8 tests/moe_ep/test_moe_ep_compute_correctness.py
"""

from __future__ import annotations

import os

import pytest

# --- config under test ------------------------------------------------------
NUM_EXPERTS = 16
TOP_K = 8
TOKENS_PER_RANK = 128
HIDDEN = 8192
INTERMEDIATE = 2048
# bf16 tolerance: weights are initialized at ~1/sqrt(fan_in) so activations stay
# O(1) and the fp32-reference vs bf16-kernel gap is precision-bound, not scale-bound.
RTOL = 3e-2
ATOL = 3e-2


def _block_major_k(w):
    """BlockMajorK shuffle for the trtllm bf16 routed runner (per-expert).

    Matches the authoritative recipe in
    ``tests/moe/test_trtllm_gen_routed_fused_moe.py`` (which validates
    ``trtllm_bf16_routed_moe`` directly): ``shuffle_matrix_a(w, epilogue_tile_m=64)``
    then ``convert_to_block_layout(., block_k=128)``, for BOTH gemm1 and gemm2, with
    NO ``reorder_rows_for_gated_act_gemm`` (the bf16 routed kernel handles SwiGLU
    gating internally). This is a pure layout transform, so the dense reference uses
    the *unshuffled* weights and still matches numerically.

    NOTE: ``epilogue_tile_m`` MUST be 64 here. The latency benchmark
    (``bench_moe_ep.py``) used 128 — harmless for timing (same shape, just a
    different row permutation) but numerically wrong; this functional test is what
    surfaced it.
    """
    import torch

    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe.core import convert_to_block_layout

    epilogue_tile_m = 64
    block_k = 128
    shuffled = []
    for i in range(w.shape[0]):
        s = shuffle_matrix_a(w[i].view(torch.uint8), epilogue_tile_m)
        s = convert_to_block_layout(s, block_k)
        shuffled.append(s)
    return torch.stack(shuffled).view(torch.bfloat16)


def _build_bf16_compute(w1_local, w2_local, *, offset, local_num_experts, max_tokens):
    """Build (MoEConfig, MoEWeightPack) for the bf16 trtllm routed runner from the
    given per-rank local expert weight slices."""
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoEWeightPack,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    wp = MoEWeightPack()
    wp.prepare_for(
        "trtllm_bf16_routed",
        {
            "gemm1_weights": _block_major_k(w1_local),
            "gemm2_weights": _block_major_k(w2_local),
        },
    )
    return cfg, wp


def _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights):
    """Single-process full MoE through the SAME unified ``MoELayer`` kernel.

    All experts local (offset 0, num_local_experts=num_experts), real top_k,
    do_finalize. This is the non-EP equivalent of the EP round-trip: it uses the
    identical compute kernel + weight layout, so comparing EP against it isolates
    dispatch/compute/combine correctness and is immune to any kernel-convention /
    weight-shuffle quirk (those cancel out). Returns [num_tokens, hidden] bf16.
    """
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer

    num_tokens = x.shape[0]
    cfg, wp = _build_bf16_compute(
        w1_full, w2_full, offset=0, local_num_experts=NUM_EXPERTS, max_tokens=num_tokens
    )
    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=x.device),
        selected_experts=topk_ids.to(torch.int32),
        final_scales=topk_weights.to(torch.float32),
    )
    return MoELayer(cfg)(act, wp)


def _torch_dense_reference(x, w1, w2, topk_ids, topk_weights):
    """Textbook dense MoE in fp32 (diagnostic only — convention-sensitive).

    Mirrors ``tests/moe/test_cute_dsl_fused_moe.py::compute_reference_moe_fp4``
    (bf16 path): gemm1 → split [linear | gate] → ``silu(gate) * linear`` → gemm2.
    Used to cross-check the kernel against textbook MoE; NOT the primary assertion
    (the trtllm-gen gate/up convention may differ from this split).
    """
    import torch

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    for t in range(num_tokens):
        ti = xf[t : t + 1]
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            s = float(topk_weights[t, k].item())
            g1 = ti @ w1f[e].T
            linear = g1[:, :INTERMEDIATE]
            gate = g1[:, INTERMEDIATE:]
            act = (gate * torch.sigmoid(gate)) * linear
            g2 = act @ w2f[e].T
            out[t] += s * g2.squeeze(0)
    return out


def _run_one_layout(layout_str):
    """Run dispatch→compute→combine for one LL layout on this rank and compare
    against the dense reference. Returns (rank, max_rel_err) for reporting."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts

    # Full expert weights — IDENTICAL on every rank (constant seed, no broadcast).
    # Scaled to ~1/sqrt(fan_in) so activations stay O(1) (bf16-precision regime).
    gw = torch.Generator(device="cuda").manual_seed(2024)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2_full = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    # Per-rank tokens + routing (distinct seed per rank).
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(TOKENS_PER_RANK, HIDDEN, device="cuda", generator=g).to(
        torch.bfloat16
    )
    # Distinct top_k experts per token via topk over random scores.
    scores = torch.randn(TOKENS_PER_RANK, NUM_EXPERTS, device="cuda", generator=g)
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(TOKENS_PER_RANK, TOP_K, device="cuda", generator=g), dim=-1
    )

    layout = (
        EpLayout.RANK_MAJOR if layout_str == "rank_major" else EpLayout.EXPERT_MAJOR
    )
    # Compute batch: EXPERT_MAJOR pads to local_experts * per_rank * world;
    # RANK_MAJOR processes per_rank * world received tokens.
    if layout is EpLayout.RANK_MAJOR:
        max_tokens = TOKENS_PER_RANK * world_size
    else:
        max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    cfg, wp = _build_bf16_compute(
        w1_full[offset : offset + local_num_experts].contiguous(),
        w2_full[offset : offset + local_num_experts].contiguous(),
        offset=offset,
        local_num_experts=local_num_experts,
        max_tokens=max_tokens,
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        nccl_comm=None,
    )
    layer = MoEEpLayer(
        bootstrap,
        FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=layout,
        ),
        backend="nccl_ep",
        compute_config=cfg,
        weights=wp,
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape, (
        f"{layout_str}: shape {tuple(y.shape)} != {tuple(x.shape)}"
    )

    # Primary reference: the SAME kernel run non-EP (full experts, real top_k).
    y_kernel = _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights)
    # Diagnostic reference: textbook torch dense MoE (convention-sensitive).
    y_torch = _torch_dense_reference(x, w1_full, w2_full, topk_ids, topk_weights)

    yf, kf, tf = y.float(), y_kernel.float(), y_torch.float()

    def _rel(a, b):
        return (a - b).abs().amax().item() / b.abs().amax().clamp_min(1e-6).item()

    ep_vs_kernel = _rel(yf, kf)
    kernel_vs_torch = _rel(kf, tf)
    ep_vs_torch = _rel(yf, tf)
    if rank == 0:
        msg = (
            f"[{layout_str}] rel-err  EP-vs-kernel={ep_vs_kernel:.4f}  "
            f"kernel-vs-torch={kernel_vs_torch:.4f}  EP-vs-torch={ep_vs_torch:.4f}\n"
            f"[{layout_str}] mean|.|  EP={yf.abs().mean():.4f}  "
            f"kernel={kf.abs().mean():.4f}  torch={tf.abs().mean():.4f}  "
            f"shapes EP={tuple(yf.shape)} kernel={tuple(kf.shape)}\n"
            f"[{layout_str}] sample row0[:4]  EP={yf[0, :4].tolist()}  "
            f"kernel={kf[0, :4].tolist()}  torch={tf[0, :4].tolist()}\n"
        )
        print(msg)
        # Capture-proof: also write to the mounted Lustre log dir when present
        # (on-cluster diagnostics); best-effort, never fail the test on its absence.
        import contextlib

        with (
            contextlib.suppress(OSError),
            open(f"/host/logs/relerr_{layout_str}_w{world_size}.txt", "w") as fh,
        ):
            fh.write(msg)

    # Primary correctness: EP must reproduce the non-EP kernel MoE exactly
    # (immune to kernel-convention quirks; tests dispatch/compute/combine).
    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    layer.destroy()
    return rank, ep_vs_kernel, kernel_vs_torch, ep_vs_torch


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        metafunc.parametrize("layout", ["expert_major", "rank_major"])


@pytest.mark.nvep
@pytest.mark.gpu_8
@pytest.mark.arch_blackwell
def test_moe_ep_compute_matches_dense_reference(layout):
    """LL bf16 dispatch→compute→combine equals a dense MoE reference (8 GPU)."""
    import torch
    import torch.distributed as dist

    backend_name = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend_name)
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        pytest.skip(
            f"num_experts={NUM_EXPERTS} not divisible by world_size={world_size}"
        )

    rank, ep_vs_kernel, kernel_vs_torch, ep_vs_torch = _run_one_layout(layout)
    dist.barrier()
    print(
        f"rank {rank}: {layout} EP==kernel OK "
        f"(EP-vs-kernel={ep_vs_kernel:.4f}, kernel-vs-torch={kernel_vs_torch:.4f})"
    )


def _main():
    """Standalone (no-pytest) entry: run both layouts under torchrun."""
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        if rank == 0:
            print(
                f"SKIP: num_experts={NUM_EXPERTS} not divisible by world={world_size}"
            )
        return
    for layout in ("expert_major", "rank_major"):
        r, ep_vs_kernel, kernel_vs_torch, ep_vs_torch = _run_one_layout(layout)
        dist.barrier()
        if r == 0:
            print(
                f"[OK] {layout}: EP==kernel (EP-vs-kernel={ep_vs_kernel:.4f}, "
                f"kernel-vs-torch={kernel_vs_torch:.4f}, EP-vs-torch={ep_vs_torch:.4f})"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
