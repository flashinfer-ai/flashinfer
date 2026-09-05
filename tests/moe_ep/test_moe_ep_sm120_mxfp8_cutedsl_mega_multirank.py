"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (sm120_mxfp8_mxfp8_bf16_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_sm120"

Requires Blackwell-consumer (sm_120/sm_121), >=4 GPUs, and CuTeDSL runtime
deps (``nvidia-cutlass-dsl``, ``nvshmem4py``).  Kernels ship in-tree under
``flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel``.  The SM120
tree is process-exclusive with the sm_100/sm_90 trees (shared top-level kernel
module names), so run this file in its own torchrun process (see
run_tests.sh ``run_mega_sm120``).

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

Weights: the kernel consumes MXFP8 expert weights in kernel-ready layout —
K-major permuted fp8 views with gate/up interleaved in groups of 8 (the
swap-AB register interleave) plus atom-swizzled E8M0 scale factors. These
tests pass canonical bf16 :class:`~flashinfer.moe_ep.MoEWeightPack`; the layer
quantizes them at init via ``preprocess_weights=True``.

Torch-oracle anchor: parity alone cannot catch a kernel that is wrong but
self-consistent at ``world_size > 1`` (peer-pull addressing, expert→rank
ownership, cross-rank combine), because both sides run the same CUDA kernel.
``test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank_torch_oracle`` closes that
gap with the sm100 twin's methodology: every rank all-gathers the actual
staged MXFP8 payloads, routing, and plain weight legs, runs the
multi-rank-native SM120 ``compute_megamoe_reference_mxfp8`` (which pins
``gate_up_interleave=8`` and ``apply_topk_in_fc1=True``) on the global
problem, and checks its own rank's slice against the real-EP kernel output.
"""

from __future__ import annotations

import os

import pytest

# This test verifies the mega path only through the swapab_cutedsl_megakernel
# shim public API; it never imports the src/ kernel packages directly, so a
# new src/ drop can't silently break it.
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel")

from .mega_oracle_compare import (  # noqa: E402
    _assert_mega_oracle_term_band_close,
)


# Upstream gap (see the tree's VENDOR.md): the current drop's REDG in-flight
# combine crashes the kernel team's own mega_runner with an illegal memory
# access (verified 2026-08-06, RTX PRO 6000, DSL 4.6.1) and appears in none of
# their test scripts; the FI backend rejects the flag. Drop this skip together
# with the backend guard once a fixed drop lands.
_IKR_BROKEN_UPSTREAM = pytest.mark.skip(
    reason="in_kernel_fc2_reduce is broken in the current SM120 kernel drop "
    "(upstream REDG path crashes; see swapab_cutedsl_megakernel/VENDOR.md)"
)


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _launcher_ranks() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size


def _make_inputs(
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    return (
        hidden_states,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int64),
    )


def _make_bf16_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    return w13, w2


def _mega_problem(
    rank: int, world_size: int, *, num_tokens: int = 64, max_tokens: int = 64
):
    hidden = 2048
    intermediate = 1024
    num_experts = 8
    topk = 4
    # Unlike the sm100 twin, no gate_up_clamp: the current SM120 drop's
    # kernel silently ignores it (dead plumbing — see VENDOR.md), so the
    # backend rejects a set clamp until a fixed drop lands.
    gate_up_clamp = None
    fast_math = True
    kind = "mxfp8_e4m3"

    assert hidden % 32 == 0
    assert intermediate % 32 == 0
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
    )
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        fast_math=fast_math,
        kind=kind,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
    )


def _reference_sm120_mxfp8_mega_moe_staged(
    problem: dict, *, destroy_buffer: bool = True, knobs: dict | None = None
):
    """Reference with bf16 activations staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel import (
        get_symm_buffer_for_sm120_mxfp8_mega_moe,
        sm120_mxfp8_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_sm120_mxfp8_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
        knobs=knobs,
    )
    num_tokens = problem["num_tokens"]
    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        symm_buffer.x,
        symm_buffer.x_sf,
        symm_buffer.topk_idx,
        symm_buffer.topk_weights,
        kind=problem["kind"],
    )

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    sm120_mxfp8_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _reference_sm120_mxfp8_mega_moe_prestaged(
    problem: dict, x_fp8, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied MXFP8 activations + E8M0 block scales."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel import (
        get_symm_buffer_for_sm120_mxfp8_mega_moe,
        sm120_mxfp8_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_sm120_mxfp8_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].view(torch.uint8).copy_(x_fp8.view(torch.uint8))
    symm_buffer.x_sf[:num_tokens].view(torch.uint8).copy_(x_sf.view(torch.uint8))
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    sm120_mxfp8_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _assert_ikr_close(y, y_ref, *, topk):
    """Scale-aware compare for the in-flight (REDG) top-k reduce.

    Mirrors the sm100 twin: the ikr path accumulates the K per-topk bf16
    terms in nondeterministic order vs the reference's fp32 explicit reduce,
    so where large terms nearly cancel the achievable agreement is bounded by
    the bf16 round-off of the largest TERM, not of the final value.  Bound
    per row: K terms x bf16 eps (2^-8) x safety 8.  A missing per-launch
    combine zero (2x accumulation) overshoots this band by ~64x.
    """
    import torch

    a = y.float()
    b = y_ref.float()
    diff = (a - b).abs()
    row_scale = torch.maximum(a.abs(), b.abs()).amax(dim=1, keepdim=True)
    tol = 5e-2 + (topk * 2.0**-8 * 8.0) * row_scale
    worst = (diff - tol).max().item()
    assert worst <= 0.0, (
        f"ikr output outside the bf16 K-term accumulation band "
        f"(worst overshoot {worst:.4f}, max diff {diff.max().item():.4f})"
    )


def _megakernel_config(
    problem: dict,
    knobs: dict | None = None,
    *,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: str = "epi_warps",
):
    from flashinfer.moe_ep import Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig

    return Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        knobs=knobs,
    )


def _run_mega_layer(
    rank,
    world_size,
    *,
    quantize_input: bool,
    num_tokens: int = 64,
    max_tokens: int = 64,
    knobs: dict | None = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: str = "epi_warps",
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = _mega_problem(
        rank, world_size, num_tokens=num_tokens, max_tokens=max_tokens
    )
    kernel = create_mega_kernel(
        _megakernel_config(
            problem,
            knobs=knobs,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            token_back_mode=token_back_mode,
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )

    try:
        if quantize_input:
            t_hidden = problem["hidden_states"]
            t_scales = None
        else:
            from flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel import (
                get_symm_buffer_for_sm120_mxfp8_mega_moe,
            )

            staging_buffer = get_symm_buffer_for_sm120_mxfp8_mega_moe(
                problem["num_experts"],
                problem["max_tokens"],
                problem["topk"],
                problem["hidden"],
                problem["intermediate"],
                rank,
                world_size,
                kind=problem["kind"],
                gate_up_clamp=problem["gate_up_clamp"],
            )
            num_tokens = problem["num_tokens"]
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                staging_buffer.x,
                staging_buffer.x_sf,
                staging_buffer.topk_idx,
                staging_buffer.topk_weights,
                kind=problem["kind"],
            )
            t_hidden = staging_buffer.x[:num_tokens].clone()
            t_scales = staging_buffer.x_sf[:num_tokens].clone()
            staging_buffer.destroy()

        mega = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
            ),
            fleet_params=FleetParams(
                num_experts=problem["num_experts"],
                max_tokens_per_rank=problem["max_tokens"],
                token_hidden_size=problem["hidden"],
            ),
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            backend=MegaConfig(
                # knobs= must reach the LAYER config (not just the throwaway
                # runtime-requirements kernel above) for pinned-knob tests to
                # actually exercise the pinned profile.
                megakernel=_megakernel_config(
                    problem,
                    knobs=knobs,
                    in_kernel_fc2_reduce=in_kernel_fc2_reduce,
                    token_back_mode=token_back_mode,
                ),
                quantize_input=quantize_input,
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        t = MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            scales=t_scales,
        )
        y_layer = mega.forward(t).clone()
        # Repeated forward on the same session: the SM120 kernel does NOT
        # tail-clean its local counters, so the second launch exercises the
        # shim's per-launch zero_local_counter_regions contract (a broken
        # reset deadlocks or corrupts here rather than silently passing).
        y_layer2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()

        if quantize_input:
            y_ref = _reference_sm120_mxfp8_mega_moe_staged(
                problem, destroy_buffer=True, knobs=knobs
            )
        else:
            y_ref = _reference_sm120_mxfp8_mega_moe_prestaged(
                problem, t_hidden, t_scales, destroy_buffer=True
            )
        dist.barrier()

        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        if in_kernel_fc2_reduce:
            # Tolerance verdict vs the explicit-reduce (plain-sum) reference;
            # see _assert_ikr_close.  The repeated forward doubles as the
            # regression guard for the per-launch combine_output.zero_()
            # (accumulate-from-zero contract): without it y_layer2 would be
            # ~2x the reference and fail loudly.
            _assert_ikr_close(y_layer, y_ref, topk=problem["topk"])
            _assert_ikr_close(y_layer2, y_ref, topk=problem["topk"])
        else:
            torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
            torch.testing.assert_close(y_layer2, y_ref, atol=0.0, rtol=0.0)
        mega.destroy()
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_moe_ep_sm120_mxfp8_cutedsl_mega_layer_matches_reference():
    """MoEEpMegaLayer (sm120_mxfp8_mxfp8_bf16_cutedsl) with on-the-fly bf16→MXFP8 staging."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True)
    print(
        f"rank {rank}: sm120_mxfp8_mxfp8_bf16_cutedsl mega layer (staged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_moe_ep_sm120_mxfp8_cutedsl_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer (sm120_mxfp8_mxfp8_bf16_cutedsl) with pre-staged MXFP8 activations."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=False)
    print(
        f"rank {rank}: sm120_mxfp8_mxfp8_bf16_cutedsl mega layer (prestaged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
@_IKR_BROKEN_UPSTREAM
def test_moe_ep_sm120_mxfp8_cutedsl_mega_layer_in_kernel_fc2_reduce():
    """In-flight top-k combine (``in_kernel_fc2_reduce=True``) for SM120 MXFP8.

    Under ikr the kernel's REDG path atomically accumulates into the
    symmetric ``(T, 1, hidden)`` combine buffer, which the shim zeroes before
    every launch (accumulate-from-zero contract; the second forward inside
    ``_run_mega_layer`` would come back ~2x without it).  SM120 ikr requires
    epi-warp token-back, which is the config default.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=True, in_kernel_fc2_reduce=True
    )
    print(
        f"rank {rank}: sm120_mxfp8_mxfp8_bf16_cutedsl mega layer (in_kernel_fc2_reduce) "
        "matches reference within tolerance"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_moe_ep_sm120_mxfp8_cutedsl_mega_layer_large_tokens_dispatch_token_back():
    """Large-token (>=2048) dispatch-warp token-back for SM120 MXFP8.

    Pins the non-default correctness paths in one profile: dispatch-warp
    token-back (``reuse_dispatch_warps``, which stages fc2 output through the
    local workspace), atomic-counter load balancing, and batched release
    flags.  (Multi-CTA clusters stay at the (1,1,1) default — cluster_m > 1
    does not compile on the current drop, see VENDOR.md.)  Exercises the
    ``fc2_output_workspace`` local region and the token-back push path end to
    end, bit-exact vs the direct-shim parity reference.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        num_tokens=2048,
        max_tokens=2048,
        token_back_mode="reuse_dispatch_warps",
        knobs={
            "flag_batch": 4,
            "epi_flag_batch": (2, 4),
            "load_balance_mode": "atomic_counter",
        },
    )
    print(
        f"rank {rank}: sm120_mxfp8_mxfp8_bf16_cutedsl mega layer (large tokens, "
        "dispatch token-back) matches reference"
    )


def _all_gather_stack(t):
    """all_gather a per-rank tensor and stack it on a new leading rank dim.

    FP8 payloads and E8M0 scale planes travel as uint8 bytes (NCCL supports
    neither dtype) and are reinterpreted after the stack.  Under the
    rank-sharing gloo process group (MEGA_SINGLE_GPU_GLOO=1, single-GPU
    sm_12x boxes) the wire additionally stages through the host: gloo has no
    CUDA all_gather and silently corrupts device tensors (the kernel drop's
    own bootstrap monkey-patches the same CPU staging).
    """
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    byte_wire = tc.element_size() == 1 and tc.dtype != torch.uint8
    wire = tc.view(torch.uint8) if byte_wire else tc
    cpu_wire = "nccl" not in str(dist.get_backend()).lower()
    if cpu_wire:
        wire = wire.cpu()
    gathered = [torch.empty_like(wire) for _ in range(world_size)]
    dist.all_gather(gathered, wire)
    stacked = torch.stack(gathered)
    if cpu_wire:
        stacked = stacked.to(tc.device)
    return stacked.view(tc.dtype) if byte_wire else stacked


def _plain_mxfp8_from_bf16(problem: dict):
    """Quantize this rank's bf16 weights into the reference's plain layout.

    SM120 twin of the sm100 helper: the fc1 gate/up rows are interleaved in
    groups of 8 (``SWAP_AB_INTERLEAVE``) via the sm120 backend's
    ``_fc1_weight_from_w13`` so the oracle sees the same weight-row order the
    kernel consumes.  Returns per-rank ``(fc1 (E, hidden, 2I), fc1_sf
    (E, 2I, hidden/32), fc2 (E, I, hidden), fc2_sf (E, hidden, I/32))`` —
    ``_all_gather_stack`` adds the leading rank dim the reference expects.
    """
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.weights import (
        _fc1_weight_from_w13,
        _quantize_mxfp8_weight_k_major,
    )

    intermediate_size = problem["intermediate"]
    kind = problem["kind"]
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    num_experts = pack.w13.shape[0]

    fc1_interleaved = _fc1_weight_from_w13(
        pack.w13, intermediate_size=intermediate_size
    )
    fc1_weights = []
    fc1_plain_sf = []
    fc2_weights = []
    fc2_plain_sf = []
    for expert in range(num_experts):
        # (2I, hidden) K-trailing → quantize → transpose into the reference's
        # (hidden, 2I) fc1 slot (the reference wants K on dim 2 of the
        # gathered (R, E, hidden, 2I) stack).
        fc1_q, fc1_sf = _quantize_mxfp8_weight_k_major(
            fc1_interleaved[expert],
            kind=kind,
        )
        fc1_weights.append(fc1_q.transpose(0, 1))
        fc1_plain_sf.append(fc1_sf)

        # (hidden, I) K-trailing → quantize → (I, hidden) reference fc2 slot.
        fc2_q, fc2_sf = _quantize_mxfp8_weight_k_major(
            pack.w2[expert],
            kind=kind,
        )
        fc2_weights.append(fc2_q.transpose(0, 1))
        fc2_plain_sf.append(fc2_sf)

    return (
        torch.stack(fc1_weights, dim=0),
        torch.stack(fc1_plain_sf, dim=0),
        torch.stack(fc2_weights, dim=0),
        torch.stack(fc2_plain_sf, dim=0),
    )


def _run_mega_torch_oracle(rank, world_size, *, in_kernel_fc2_reduce: bool = False):
    """Real-EP kernel launch vs the drop's torch GLOBAL reference.

    Parity alone cannot catch a kernel that is wrong but self-consistent at
    ``world_size > 1`` (peer-pull addressing, expert→rank ownership,
    cross-rank combine), because both sides run the same CUDA kernel; this
    closes that gap (sm100/sm90 twin methodology).

    Every rank stages its own bf16 shard, runs the fused kernel with real
    cross-rank NVSHMEM traffic, then all-gathers the ACTUAL staged MXFP8
    activations + routing + plain (pre-swizzle) weight legs (no reliance on
    cross-rank RNG determinism) and feeds the global problem to the SM120
    ``compute_megamoe_reference_mxfp8`` — which is multi-rank native and pins
    ``gate_up_interleave=8`` / ``apply_topk_in_fc1=True`` internally.  Each
    rank asserts its own output slice within the term-magnitude band.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm120.swapab_cutedsl_megakernel import (
        compute_megamoe_reference_mxfp8,
        get_symm_buffer_for_sm120_mxfp8_mega_moe,
        sm120_mxfp8_mega_moe,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)
    num_local = problem["num_experts"] // world_size
    # Guarantee cross-rank traffic by construction: token 0 routes one expert
    # per EP rank (contiguous block ownership: rank r owns [r*L, (r+1)*L)).
    forced = (
        torch.arange(min(problem["topk"], world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    problem["topk_ids"][0, : forced.numel()] = forced

    kernel = create_mega_kernel(
        _megakernel_config(problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce)
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )
    try:
        n = problem["num_tokens"]

        symm_buffer = get_symm_buffer_for_sm120_mxfp8_mega_moe(
            problem["num_experts"],
            problem["max_tokens"],
            problem["topk"],
            problem["hidden"],
            problem["intermediate"],
            rank,
            world_size,
            kind=problem["kind"],
            gate_up_clamp=problem["gate_up_clamp"],
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        )
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            kind=problem["kind"],
        )
        # Snapshot exactly what the kernel consumes (this rank's shard).
        x_local = symm_buffer.x[:n].clone()
        x_sf_local = symm_buffer.x_sf[:n].clone()
        idx_local = symm_buffer.topk_idx[:n].clone()
        w_local = symm_buffer.topk_weights[:n].clone()

        transformed_l1, transformed_l2 = preprocess_mega_weights(
            MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            intermediate_size=problem["intermediate"],
            hidden_size=problem["hidden"],
            kind=problem["kind"],
            gate_up_clamp=problem["gate_up_clamp"],
        )

        y_kernel = torch.empty(
            n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        sm120_mxfp8_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=n,
            gate_up_clamp=problem["gate_up_clamp"],
            fast_math=problem["fast_math"],
        )
        torch.cuda.synchronize()
        dist.barrier()
        symm_buffer.destroy()

        # Reassemble the global problem from the operands each rank staged;
        # the reference consumes (num_ranks, ...) stacks directly.  The SM120
        # wrapper pins gate_up_interleave=8 and apply_topk_in_fc1=True itself.
        fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_mxfp8_from_bf16(problem)
        combine_ref = compute_megamoe_reference_mxfp8(
            input_activation=_all_gather_stack(x_local),
            input_activation_sf=_all_gather_stack(x_sf_local),
            input_topk_idx=_all_gather_stack(idx_local),
            input_topk_weights=_all_gather_stack(w_local),
            fc1_weight=_all_gather_stack(fc1_plain),
            fc1_weight_sf=_all_gather_stack(fc1_sf),
            fc2_weight=_all_gather_stack(fc2_plain),
            fc2_weight_sf=_all_gather_stack(fc2_sf),
            ab_dtype=torch.float8_e4m3fn,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        # The topk weight is already folded before the fc1-out round-trip, so
        # the per-topk terms reduce with a plain sum; compare this rank's slice.
        y_ref = combine_ref[rank].to(torch.float32).sum(dim=1)

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        rel_l2 = (yk - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
        print(
            f"[sm120 mxfp8 multirank oracle rank {rank} ikr={in_kernel_fc2_reduce}] "
            f"rel_l2={rel_l2.item():.4g} "
            f"max|d|={(yk - y_ref).abs().max().item():.4g} "
            f"amax(ref)={y_ref.abs().max().item():.4g}"
        )
        # Per-cell bf16 term-magnitude band (shared helper): the bound derives
        # from the oracle's own per-topk terms; the ikr coefficient absorbs
        # the REDG nondeterministic reduce order (same bound as
        # _assert_ikr_close).
        _assert_mega_oracle_term_band_close(
            yk, combine_ref[rank], ikr=in_kernel_fc2_reduce, label=f"rank{rank}"
        )
        assert rel_l2.item() < (0.03 if in_kernel_fc2_reduce else 0.02)
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
@pytest.mark.parametrize(
    "in_kernel_fc2_reduce",
    [False, pytest.param(True, marks=_IKR_BROKEN_UPSTREAM)],
)
def test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank_torch_oracle(in_kernel_fc2_reduce):
    """Real cross-rank EP kernel vs the drop's torch global math (see helper doc)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(
        rank, world_size, in_kernel_fc2_reduce=in_kernel_fc2_reduce
    )
    print(
        f"rank {rank}: sm120_mxfp8_mxfp8_bf16_cutedsl mega kernel (ikr={in_kernel_fc2_reduce}) "
        "matches the multi-rank torch oracle"
    )


@pytest.mark.arch_sm120
def test_sm120_mxfp8_cutedsl_preprocess_mega_weights_from_bf16():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_local_experts = problem["num_experts"] // world_size

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    fc1_weight, fc1_sf = transformed_l1
    fc2_weight, fc2_sf = transformed_l2
    assert fc1_weight.shape == (
        num_local_experts,
        problem["hidden"],
        2 * problem["intermediate"],
    )
    assert fc2_weight.shape == (
        num_local_experts,
        problem["intermediate"],
        problem["hidden"],
    )
    # The SM120 kernel consumes K-major views (K = dim 1 stride-1), not the
    # sm100 tree's contiguous layout.
    assert fc1_weight.stride(1) == 1
    assert fc2_weight.stride(1) == 1
    assert fc1_weight.dtype == torch.float8_e4m3fn
    assert fc2_weight.dtype == torch.float8_e4m3fn
    assert fc1_sf.shape[0] == num_local_experts
    assert fc2_sf.shape[0] == num_local_experts


def test_sm120_mxfp8_cutedsl_mega_kernel_is_registered():
    from flashinfer.moe_ep import Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "sm120_mxfp8_mxfp8_bf16_cutedsl"
