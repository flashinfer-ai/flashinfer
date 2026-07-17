"""fi_dg determinism probe: repeated forwards under deliberate rank skew.

Discriminates the two live hypotheses for the vLLM e2e fi_dg run-to-run
nondeterminism (``todo_fi_dg_nondeterminism.md``): staged bytes and kernel
args are proven identical, so divergence must enter either through

  (a) cross-rank arrival order — the deep_gemm mega kernel's combine being
      sensitive to which rank's data lands first (skew-dependent reduction
      order), with fi's busier host loop making skew variable in-engine; or
  (b) non-argument state (workspace residuals / allocator addresses).

This test forces gross (~100 ms), per-iteration-rotating rank skew around
otherwise identical forwards and asserts bit-exact outputs. FAILURE here
confirms (a) and reproduces the engine bug at layer level; PASSING under
gross skew exonerates arrival order and points the hunt at (b).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_deep_gemm_skew_determinism.py -v \\
        -m "gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import pytest

from .test_moe_ep_deep_gemm_mega_multirank import (
    _launcher_ranks,
    _mega_problem,
    _require_cuda,
)

_ITERS = 8
_SKEW_CYCLES = 200_000_000  # ~100 ms at ~2 GHz — grossly exceeds engine skew


def _build_layer(rank: int, world_size: int):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
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
        weights=problem["weights"],
        backend=MegaConfig(
            megakernel=DeepGemmMegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                activation_clamp=problem["activation_clamp"],
                fast_math=problem["fast_math"],
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    t = MoEEpTensors(
        hidden_states=problem["hidden_states"],
        topk_ids=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
    )
    return mega, t


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_deep_gemm_mega_repeated_forward_bitexact_under_skew():
    pytest.importorskip("deep_gemm")
    pytest.importorskip("triton")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")

    import torch
    import torch.distributed as dist

    mega, t = _build_layer(rank, world_size)
    try:
        # Control: back-to-back repeats with natural (small) skew.
        y0 = mega.forward(t).clone()
        torch.cuda.synchronize()
        dist.barrier()
        for it in range(2):
            y = mega.forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            assert torch.equal(y, y0), f"control repeat {it} diverged (no skew)"

        # Probe: rotate a ~100 ms GPU stall across ranks before the forward.
        # Arrival-order-sensitive combine => bit flips; order-fixed => exact.
        mismatched = []
        for it in range(_ITERS):
            if rank == it % world_size:
                torch.cuda._sleep(_SKEW_CYCLES)
            y = mega.forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            if not torch.equal(y, y0):
                delta = (y.float() - y0.float()).abs().max().item()
                mismatched.append((it, delta))
        assert not mismatched, (
            f"rank {rank}: {len(mismatched)}/{_ITERS} skewed forwards diverged "
            f"(iter, max|dy|): {mismatched} — deep_gemm mega combine is "
            "arrival-order sensitive; skew, not state, drives the e2e "
            "nondeterminism"
        )
    finally:
        mega.destroy()
        torch.cuda.synchronize()
        dist.barrier()


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_deep_gemm_mega_bitexact_across_input_addresses():
    """Same input BYTES at different base addresses/alignments must match.

    In-engine, the MoE layer's inputs are fresh per-step allocations whose
    addresses can differ run-to-run (host-side allocation races); if the
    staging quant (`per_token_cast_to_fp8`) or the kernel's input path is
    alignment-sensitive in its reduction order, identical bytes at different
    addresses produce different fp8 scales -> the e2e nondeterminism.
    """
    pytest.importorskip("deep_gemm")
    pytest.importorskip("triton")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpTensors

    mega, t = _build_layer(rank, world_size)
    try:
        n, h = t.hidden_states.shape
        y0 = mega.forward(t).clone()
        torch.cuda.synchronize()
        dist.barrier()
        x0 = mega._workspace.x[:n].clone()
        sf0 = mega._workspace.x_sf[:n].clone()

        # Re-issue the same bytes from shifted base addresses. Offsets are in
        # bf16 elements: 8 keeps 16B alignment, 4 = 8B, 64 = 128B.
        mismatches = []
        for off in (8, 4, 64, 128):
            storage = torch.empty(n * h + off, dtype=torch.bfloat16, device="cuda")
            hs = storage[off : off + n * h].view(n, h)
            hs.copy_(t.hidden_states)
            t2 = MoEEpTensors(
                hidden_states=hs,
                topk_ids=t.topk_ids,
                topk_weights=t.topk_weights,
            )
            y = mega.forward(t2)
            torch.cuda.synchronize()
            dist.barrier()
            stage_exact = torch.equal(
                mega._workspace.x[:n].view(torch.uint8), x0.view(torch.uint8)
            ) and torch.equal(
                mega._workspace.x_sf[:n].view(torch.uint8), sf0.view(torch.uint8)
            )
            out_exact = torch.equal(y, y0)
            if not (stage_exact and out_exact):
                delta = (y.float() - y0.float()).abs().max().item()
                mismatches.append((off, stage_exact, out_exact, delta))
        assert not mismatches, (
            f"rank {rank}: address-shifted inputs diverged "
            f"(offset_elems, staged_bytes_equal, output_equal, max|dy|): "
            f"{mismatches} — staging/kernel is address/alignment sensitive"
        )
    finally:
        mega.destroy()
        torch.cuda.synchronize()
        dist.barrier()
