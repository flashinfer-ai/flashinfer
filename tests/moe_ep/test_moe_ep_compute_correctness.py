"""Multi-GPU functional correctness for LL EXPERT_MAJOR + RANK_MAJOR (bf16).

Exercises dispatch → fused_moe compute → combine and compares against a
single-process dense MoE reference via the same ``MoELayer`` kernel.

Launch (4 GPU):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_compute_correctness.py -v -s -m "nvep and gpu_4"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)

NUM_EXPERTS = 16
TOP_K = 8
TOKENS_PER_RANK = 128
HIDDEN = 8192
INTERMEDIATE = 2048
RTOL = 3e-2
ATOL = 3e-2
# EP-vs-torch-oracle band: looser than EP-vs-kernel because the oracle runs
# dense fp32 GEMMs against the kernel's bf16 accumulation on |y|~O(1).
ORACLE_RTOL = 5e-2
ORACLE_ATOL = 5e-2


def _build_bf16_moe_config(*, offset, local_num_experts, max_tokens):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    return MoEConfig(
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


def _build_fused_moe_weights(w1_local, w2_local):
    # Same weight prep as the EP layer (gated-act reorder + shuffle + BlockMajorK).
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    cfg = _build_bf16_moe_config(
        offset=0, local_num_experts=w1_local.shape[0], max_tokens=1
    )
    return materialize_fused_moe_weights(MoEWeightPack(w13=w1_local, w2=w2_local), cfg)


def _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights):
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer

    num_tokens = x.shape[0]
    cfg = _build_bf16_moe_config(
        offset=0, local_num_experts=NUM_EXPERTS, max_tokens=num_tokens
    )
    wp = _build_fused_moe_weights(w1_full, w2_full)
    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=x.device),
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
    )
    return MoELayer(cfg)(act, wp)


def _torch_dense_reference(x, w1, w2, topk_ids, topk_weights):
    """Textbook dense MoE in fp32 (asserted torch oracle).

    Mirrors the asserted ``tests/moe`` trtllm-gen reference
    (``run_moe_dequant``): gemm1 → split ``[x1 | x2]`` → ``silu(x2) * x1`` →
    gemm2 → topk-weighted sum — the same split used here (``linear`` = first
    half, ``gate`` = second half).  Single-GPU per-dtype oracles live in
    ``test_split_fused_moe_kernel_vs_reference.py``; this one additionally
    pins the full EP path against textbook MoE math.
    """
    import os

    import torch

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

    # Progress bar for the slow per-token python loop; only rank 0 draws it, and
    # it degrades to a plain range() when tqdm isn't installed.
    token_iter = range(num_tokens)
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        try:
            from tqdm import tqdm

            token_iter = tqdm(
                token_iter, desc="torch dense ref", unit="tok", leave=False
            )
        except ImportError:
            pass

    for t in token_iter:
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
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts

    gw = torch.Generator(device="cuda").manual_seed(2024)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2_full = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(TOKENS_PER_RANK, HIDDEN, device="cuda", generator=g).to(
        torch.bfloat16
    )
    scores = torch.randn(TOKENS_PER_RANK, NUM_EXPERTS, device="cuda", generator=g)
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(TOKENS_PER_RANK, TOP_K, device="cuda", generator=g), dim=-1
    )

    layout = (
        EpLayout.RANK_MAJOR if layout_str == "rank_major" else EpLayout.EXPERT_MAJOR
    )
    if layout is EpLayout.RANK_MAJOR:
        max_tokens = TOKENS_PER_RANK * world_size
    else:
        max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    moe_config = _build_bf16_moe_config(
        offset=offset,
        local_num_experts=local_num_experts,
        max_tokens=max_tokens,
    )
    canonical_weights = MoEWeightPack(
        w13=w1_full[offset : offset + local_num_experts].contiguous(),
        w2=w2_full[offset : offset + local_num_experts].contiguous(),
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
        weights=canonical_weights,
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape

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
        print(
            f"[{layout_str}] rel-err  EP-vs-kernel={ep_vs_kernel:.4f}  "
            f"kernel-vs-torch={kernel_vs_torch:.4f}  EP-vs-torch={ep_vs_torch:.4f}\n"
            f"[{layout_str}] mean|.|  EP={yf.abs().mean():.4f}  "
            f"kernel={kf.abs().mean():.4f}  torch={tf.abs().mean():.4f}  "
            f"shapes EP={tuple(yf.shape)} kernel={tuple(kf.shape)}\n"
            f"[{layout_str}] sample row0[:4]  EP={yf[0, :4].tolist()}  "
            f"kernel={kf[0, :4].tolist()}  torch={tf[0, :4].tolist()}"
        )

    # Primary correctness: EP must reproduce the non-EP kernel MoE exactly
    # (immune to kernel-convention quirks; tests dispatch/compute/combine).
    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    # Oracle correctness: EP must also match textbook torch MoE math, so a
    # bug shared by the EP and non-EP kernel paths cannot pass silently.
    torch.testing.assert_close(yf, tf, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
    layer.destroy()
    return rank, ep_vs_kernel, kernel_vs_torch, ep_vs_torch


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        metafunc.parametrize("layout", ["expert_major", "rank_major"])


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_compute_matches_dense_reference(layout):
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
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
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
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
