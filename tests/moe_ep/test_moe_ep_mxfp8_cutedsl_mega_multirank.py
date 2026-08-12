"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (sm100_mxfp8_mxfp8_bf16_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and CuTeDSL runtime deps
(``nvidia-cutlass-dsl[cu13]``, ``nvshmem4py-cu13``).  Kernels ship in-tree under
``flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe``.

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

Weights: the CuTeDSL kernel consumes MXFP8 expert weights in kernel-ready
(swizzled E8M0 scale-factor) layout. These tests pass canonical bf16
:class:`~flashinfer.moe_ep.MoEWeightPack`; the layer quantizes them at init via
``preprocess_weights=True``. To supply pre-quantized MXFP8 weights instead, pass
kernel-layout ``w13``/``w2`` plus ``w13_scale``/``w2_scale``.

Torch-oracle anchor: parity alone cannot catch a kernel that is wrong but
self-consistent at ``world_size > 1`` (peer-pull addressing, expert→rank
ownership, cross-rank combine), because both sides run the same CUDA kernel.
``test_moe_ep_mxfp8_cutedsl_mega_multirank_torch_oracle`` closes that gap with
the sm90_fp8_fp8_bf16_pull_cutedsl twin's methodology: every rank all-gathers the actual staged
MXFP8 payloads, routing, and plain weight legs, runs the multi-rank-native
``compute_megamoe_reference_mxfp8`` on the global problem, and checks its own
rank's slice against the real-EP kernel output.
"""

from __future__ import annotations

import os

import pytest

# This test verifies the mega path only through the cutedsl_megamoe shim public
# API (``flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe``); it never imports the
# src/ kernel packages directly, so a new src/ drop can't silently break it.
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")


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
    gate_up_clamp = 10.0
    fast_math = True
    kind = "mxfp8_e4m3"

    assert hidden % 128 == 0
    assert intermediate % 128 == 0
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


def _reference_mxfp8_mega_moe_staged(
    problem: dict, *, destroy_buffer: bool = True, knobs: dict | None = None
):
    """Reference with bf16 activations staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mxfp8_mega_moe,
        mxfp8_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
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
    mxfp8_mega_moe(
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


def _reference_mxfp8_mega_moe_prestaged(
    problem: dict, x_fp8, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied MXFP8 activations + E8M0 block scales."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mxfp8_mega_moe,
        mxfp8_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
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
    mxfp8_mega_moe(
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

    Mirrors the NVFP4 twin: the ikr path accumulates the K per-topk bf16
    terms in nondeterministic order vs the reference's fp32 explicit reduce,
    so where large terms nearly cancel the achievable agreement is bounded by
    the bf16 round-off of the largest TERM, not of the final value.  Bound
    per row: K terms x bf16 eps (2^-8) x safety 8.  A missing per-launch
    output zero (2x accumulation) overshoots this band by ~64x.
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
):
    from flashinfer.moe_ep import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig

    return Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
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
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
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
            problem, knobs=knobs, in_kernel_fc2_reduce=in_kernel_fc2_reduce
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
            from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
                get_symm_buffer_for_mxfp8_mega_moe,
            )

            staging_buffer = get_symm_buffer_for_mxfp8_mega_moe(
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
                    problem, knobs=knobs, in_kernel_fc2_reduce=in_kernel_fc2_reduce
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
        # Repeated forward on the same session: with no per-launch host reset
        # (run() default reset_counters=False) the second launch relies on the
        # kernel's tail cleanup of its workspace counters/flags -- this is the
        # regression guard for that contract.
        y_layer2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()

        if quantize_input:
            y_ref = _reference_mxfp8_mega_moe_staged(
                problem, destroy_buffer=True, knobs=knobs
            )
        else:
            y_ref = _reference_mxfp8_mega_moe_prestaged(
                problem, t_hidden, t_scales, destroy_buffer=True
            )
        dist.barrier()

        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        if in_kernel_fc2_reduce:
            # Tolerance verdict vs the explicit-reduce (plain-sum) reference;
            # see _assert_ikr_close.  The repeated forward doubles as the
            # regression guard for the per-launch output_activation.zero_()
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
@pytest.mark.arch_blackwell
def test_moe_ep_mxfp8_cutedsl_mega_layer_matches_reference():
    """MoEEpMegaLayer (sm100_mxfp8_mxfp8_bf16_cutedsl) with on-the-fly bf16→MXFP8 staging."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True)
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega layer (staged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mxfp8_cutedsl_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer (sm100_mxfp8_mxfp8_bf16_cutedsl) with pre-staged MXFP8 activations."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=False)
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega layer (prestaged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mxfp8_cutedsl_mega_layer_in_kernel_fc2_reduce():
    """In-flight top-k combine (``in_kernel_fc2_reduce=True``) for MXFP8.

    Regression guard for the sym-heap output fix: the MXFP8 symm buffer used
    to allocate ``output_activation`` rank-locally even when the ikr param was
    set, which would crash the cross-rank REDG path.  The output now always
    lives on the symmetric heap and is zeroed before every launch
    (accumulate-from-zero contract; the second forward inside
    ``_run_mega_layer`` would come back ~2x without it).  MXFP8 ikr requires
    epi-warp token-back, which is the measured MXFP8 default profile.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=True, in_kernel_fc2_reduce=True
    )
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega layer (in_kernel_fc2_reduce) "
        "matches reference within tolerance"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mxfp8_cutedsl_mega_layer_large_tokens_matches_reference():
    """Large-token (>=2048) dispatch-warp token-back for MXFP8.

    The MXFP8 default heuristic now uses flag_batch=4 + epi_warps at all sizes
    (measured faster 2026-07-14), so the dispatch-warp combo is pinned here
    explicitly via knobs: this stays the regression guard for whether MXFP8
    large-token dispatch-warp token-back (token_back_by_dispatch=True, which
    has no non_ubulk_fc2_store escape hatch) compiles + runs bit-exact.
    MXFP8's mma_tiler stays kernel-fixed at (256,256).
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
        # The full pre-2026-07-14 LARGE profile (explicit knobs skip the
        # heuristic entirely, so pin every knob the old profile set).
        knobs={
            "cluster_shape_mnk": (2, 1, 1),
            "group_hint": 512,
            "flag_batch": 8,
            "epi_flag_batch": (2, 4),
            "token_back_mode": "reuse_dispatch_warps",
            "load_balance_mode": "atomic_counter",
        },
    )
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega layer (large tokens) matches reference"
    )


def _all_gather_stack(t):
    """all_gather a per-rank tensor and stack it on a new leading rank dim.

    FP8 payloads and E8M0 scale planes travel as uint8 bytes (NCCL supports
    neither dtype) and are reinterpreted after the stack.
    """
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    byte_wire = tc.element_size() == 1 and tc.dtype != torch.uint8
    wire = tc.view(torch.uint8) if byte_wire else tc
    gathered = [torch.empty_like(wire) for _ in range(world_size)]
    dist.all_gather(gathered, wire)
    stacked = torch.stack(gathered)
    return stacked.view(tc.dtype) if byte_wire else stacked


def _run_mega_torch_oracle(rank, world_size, *, in_kernel_fc2_reduce: bool = False):
    """Real-EP kernel launch vs the drop's torch GLOBAL reference.

    Parity alone cannot catch a kernel that is wrong but self-consistent at
    ``world_size > 1`` (peer-pull addressing, expert→rank ownership,
    cross-rank combine), because both sides run the same CUDA kernel; this
    closes that gap (sm90_fp8_fp8_bf16_pull_cutedsl twin methodology).

    Every rank stages its own bf16 shard, runs the fused kernel with real
    cross-rank NVSHMEM traffic, then all-gathers the ACTUAL staged MXFP8
    activations + routing + plain (pre-swizzle) weight legs (no reliance on
    cross-rank RNG determinism) and feeds the global problem to
    ``compute_megamoe_reference_mxfp8`` — which is multi-rank native: it takes
    ``(num_ranks, tokens_per_rank, ...)`` operands and computes
    ``expert(topk_idx[r, t, k])`` across rank boundaries.  Each rank asserts
    its own output slice within the single-GPU oracle's tolerances.

    ``in_kernel_fc2_reduce`` runs the REDG in-flight combine variant (torch
    reference unchanged; the compare widens by the bf16 K-term accumulation
    band, see ``_assert_ikr_close``).
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
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        compute_megamoe_reference_mxfp8,
        get_symm_buffer_for_mxfp8_mega_moe,
        mxfp8_mega_moe,
    )

    from .test_mxfp8_cutedsl_preprocess_vs_reference import (
        _assert_mega_oracle_term_band_close,
        _plain_mxfp8_from_bf16,
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

        symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
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
        try:
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
            mxfp8_mega_moe(
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

            # Reassemble the global problem from the operands each rank staged;
            # the reference consumes (num_ranks, ...) stacks directly.
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
                apply_topk_in_fc1=True,
            )
            # The topk weight is already folded before the fc1-out round-trip, so
            # the per-topk terms reduce with a plain sum; compare this rank's slice.
            y_ref = combine_ref[rank].to(torch.float32).sum(dim=1)

            assert torch.isfinite(y_kernel).all()
            yk = y_kernel.to(torch.float32)
            rel_l2 = (yk - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
            print(
                f"[mxfp8 multirank oracle rank {rank} ikr={in_kernel_fc2_reduce}] "
                f"rel_l2={rel_l2.item():.4g} "
                f"max|d|={(yk - y_ref).abs().max().item():.4g} "
                f"amax(ref)={y_ref.abs().max().item():.4g}"
            )
            # Per-cell bf16 term-magnitude band (see
            # _assert_mega_oracle_term_band_close): the old flat atol=8.0 was
            # "1 bf16 ULP at |term|~2048" calibrated on GB200's rounding and
            # tripped by one cell on B200; the band derives the bound from the
            # oracle's own per-topk terms instead. The ikr coefficient absorbs
            # the REDG nondeterministic reduce order (same bound as
            # _assert_ikr_close).
            _assert_mega_oracle_term_band_close(
                yk, combine_ref[rank], ikr=in_kernel_fc2_reduce, label=f"rank{rank}"
            )
            assert rel_l2.item() < (0.03 if in_kernel_fc2_reduce else 0.02)
            return rank
        finally:
            # A failing rank must still free its symmetric-heap slice;
            # leaking it turns a clean failure into a multi-rank hang.
            symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("in_kernel_fc2_reduce", [False, True])
def test_moe_ep_mxfp8_cutedsl_mega_multirank_torch_oracle(in_kernel_fc2_reduce):
    """Real cross-rank EP kernel vs the drop's torch global math (see helper doc)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(
        rank, world_size, in_kernel_fc2_reduce=in_kernel_fc2_reduce
    )
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega kernel (ikr={in_kernel_fc2_reduce}) "
        "matches the multi-rank torch oracle"
    )



@pytest.mark.arch_blackwell
def test_mxfp8_cutedsl_preprocess_mega_weights_from_bf16():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
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
    assert fc1_weight.dtype == torch.float8_e4m3fn
    assert fc2_weight.dtype == torch.float8_e4m3fn
    assert fc1_sf.shape[0] == num_local_experts
    assert fc2_sf.shape[0] == num_local_experts


def test_mxfp8_cutedsl_mega_kernel_is_registered():
    from flashinfer.moe_ep import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "sm100_mxfp8_mxfp8_bf16_cutedsl"
