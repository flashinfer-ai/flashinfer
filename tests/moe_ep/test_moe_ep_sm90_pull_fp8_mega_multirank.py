"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (sm90_pull_fp8).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm90_pull_fp8_mega_multirank.py -v -m "gpu_4 and arch_hopper"

Requires Hopper (exactly sm_90), >=4 GPUs, and CuTeDSL runtime deps
(``nvidia-cutlass-dsl[cu13]``, ``nvshmem4py-cu13``).  Kernels ship in-tree under
``flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel``.

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

Parity methodology (mirrors the mxfp8_cutedsl twin): each test drives the SAME
fused kernel twice — once through the full ``MoEEpLayer`` EP plumbing and once
directly through the shim API (``hopper_fp8_mega_moe`` on a fresh symm buffer)
with identical staged inputs, quantization, and preprocessed weights — and
asserts bit-exact equality for the deterministic separate-reduce paths.  The
kernel-vs-textbook-math anchor is the single-GPU oracle
(``test_sm90_pull_fp8_kernel_vs_reference.py`` vs the drop's
``compute_megamoe_reference_fp8``); this file anchors the EP layer plumbing on
real cross-rank traffic.  The ikr (REDG) test compares against the
explicit-reduce reference within the bf16 K-term accumulation band instead.

per_tensor contract: the activation dequant scales are STATIC config scalars
identical on every EP rank (the kernel dequantizes peer tokens with the local
copy), so both the layer config and the direct-shim reference use the same
fixed calibration constants below.

Process isolation: the SM90 and SM100 kernel trees share top-level module
names and are mutually exclusive per process — this file is excluded from
run_tests.sh's ``unit`` target and runs in its own torchrun pytest process via
the ``mega_sm90`` target.
"""

from __future__ import annotations

import os

import pytest

# This test verifies the mega path only through the pull_style_cutedsl_megakernel
# shim public API (``flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel``);
# it never imports the src/ kernel packages directly, so a new src/ drop can't
# silently break it.
pytest.importorskip(
    "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
)

E4M3_MAX = 448.0
# Static per-tensor calibration (identical on all ranks by contract): randn
# bf16 activations stay within |x| <= 8 for these sizes, and the 1/sqrt(K)
# weight normalization keeps SwiGLU outputs O(1) (<= 8 with the topk softmax
# weights).  Both scales carry the reference's 0.95 headroom margin.
FC1_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)
FC2_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)


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
    world_size: int,
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
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_ids = topk_ids.to(torch.int64)

    # Guarantee cross-rank traffic by construction (random routing makes it
    # near-certain; this makes it certain): token 0 routes one expert per EP
    # rank — with topk == world_size that is experts {0, L, 2L, 3L} (distinct,
    # so no duplicate-expert rows).
    num_local = num_experts // world_size
    forced = torch.arange(
        min(topk, world_size), device="cuda", dtype=torch.int64
    ) * num_local
    topk_ids[0, : forced.numel()] = forced

    return hidden_states, topk_weights.to(torch.float32), topk_ids


def _make_bf16_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    """O(1)-output weights (1/sqrt(K) normalized, like the single-GPU oracle)."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (hidden**-0.5)
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (intermediate**-0.5)
    return w13, w2


def _mega_problem(
    rank: int,
    world_size: int,
    *,
    fp8_scale_mode: str,
    swap_ab: bool = False,
    num_tokens: int = 64,
    max_tokens: int = 64,
):
    hidden = 2048
    intermediate = 1024
    num_experts = 8
    topk = 4
    gate_up_clamp = 10.0
    fast_math = True
    kind = "fp8_e4m3"

    assert hidden % 128 == 0
    assert intermediate % 128 == 0
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        world_size,
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
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
    )


def _preprocess_weights(problem: dict):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8.weights import (
        preprocess_mega_weights,
    )

    return preprocess_mega_weights(
        MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
        fp8_scale_mode=problem["fp8_scale_mode"],
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )


def _alloc_symm_buffer(problem: dict, rank: int, world_size: int):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_fp8_mega_moe,
    )

    return get_symm_buffer_for_hopper_fp8_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        kind=problem["kind"],
        fp8_scale_mode=problem["fp8_scale_mode"],
        swap_ab=problem["swap_ab"],
        gate_up_clamp=problem["gate_up_clamp"],
    )


def _reference_sm90_fp8_mega_moe_staged(problem: dict, *, destroy_buffer: bool = True):
    """Reference: direct shim launch with bf16 staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_fp8_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = _alloc_symm_buffer(problem, rank, world_size)
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
        fp8_scale_mode=problem["fp8_scale_mode"],
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
    )

    transformed_l1, transformed_l2 = _preprocess_weights(problem)

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    hopper_fp8_mega_moe(
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


def _reference_sm90_fp8_mega_moe_prestaged(
    problem: dict, x_fp8, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied FP8 activations + per-mode scales."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_fp8_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = _alloc_symm_buffer(problem, rank, world_size)
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].view(torch.uint8).copy_(x_fp8.view(torch.uint8))
    symm_buffer.x_sf[:num_tokens].view(torch.uint8).copy_(x_sf.view(torch.uint8))
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    transformed_l1, transformed_l2 = _preprocess_weights(problem)

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    hopper_fp8_mega_moe(
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

    Mirrors the mxfp8 twin: the ikr path accumulates the K per-topk bf16
    terms in nondeterministic order vs the reference's explicit reduce, so
    where large terms nearly cancel the achievable agreement is bounded by
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


def _megakernel_config(problem: dict, *, in_kernel_fc2_reduce: bool = False):
    from flashinfer.moe_ep import Sm90PullFp8MegaMoeConfig

    return Sm90PullFp8MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        kind=problem["kind"],
        fp8_scale_mode=problem["fp8_scale_mode"],
        swap_ab=problem["swap_ab"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )


def _run_mega_layer(
    rank,
    world_size,
    *,
    quantize_input: bool,
    fp8_scale_mode: str,
    swap_ab: bool = False,
    num_tokens: int = 64,
    max_tokens: int = 64,
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
    from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = _mega_problem(
        rank,
        world_size,
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
    )
    kernel = create_mega_kernel(
        _megakernel_config(problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce)
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
            staging_buffer = _alloc_symm_buffer(problem, rank, world_size)
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                staging_buffer.x,
                staging_buffer.x_sf,
                staging_buffer.topk_idx,
                staging_buffer.topk_weights,
                kind=problem["kind"],
                fp8_scale_mode=problem["fp8_scale_mode"],
                fc1_activation_dequant_scale=FC1_ACT_SCALE,
            )
            t_hidden = staging_buffer.x[: problem["num_tokens"]].clone()
            t_scales = staging_buffer.x_sf[: problem["num_tokens"]].clone()
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
                megakernel=_megakernel_config(
                    problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce
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
        # kernel's tail cleanup of its workspace counters/flags AND on the
        # launch-kwargs cache hitting (same buffers/stream) -- this is the
        # regression guard for both contracts.
        y_layer2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()

        if quantize_input:
            y_ref = _reference_sm90_fp8_mega_moe_staged(problem, destroy_buffer=True)
        else:
            y_ref = _reference_sm90_fp8_mega_moe_prestaged(
                problem, t_hidden, t_scales, destroy_buffer=True
            )
        dist.barrier()

        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        if in_kernel_fc2_reduce:
            # Tolerance verdict vs the explicit-reduce reference; see
            # _assert_ikr_close.  The repeated forward doubles as the
            # regression guard for the per-launch output_activation.zero_()
            # (accumulate-from-zero contract): without it y_layer2 would be
            # ~2x the reference and fail loudly.
            _assert_ikr_close(y_layer, y_ref, topk=problem["topk"])
            _assert_ikr_close(y_layer2, y_ref, topk=problem["topk"])
        else:
            # Same kernel, same staged operands, same static scales, and the
            # separate-reduce path is deterministic -> bit-exact parity, like
            # the mxfp8 twin.
            torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
            torch.testing.assert_close(y_layer2, y_ref, atol=0.0, rtol=0.0)
        mega.destroy()
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("fp8_scale_mode", ["per_tensor", "blockwise"])
def test_moe_ep_sm90_pull_fp8_mega_layer_matches_reference(fp8_scale_mode):
    """MoEEpMegaLayer (sm90_pull_fp8) with on-the-fly bf16→FP8 staging."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=True, fp8_scale_mode=fp8_scale_mode
    )
    print(
        f"rank {rank}: sm90_pull_fp8 mega layer ({fp8_scale_mode}, staged inputs) "
        "matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_fp8_mega_layer_swap_ab_matches_reference():
    """Swap-AB geometry ((256, 32, 128) token-N tile) on the per_tensor mode."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        fp8_scale_mode="per_tensor",
        swap_ab=True,
    )
    print(f"rank {rank}: sm90_pull_fp8 mega layer (swap_ab) matches reference")


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_fp8_mega_layer_prestaged_inputs_matches_reference():
    """Pre-staged FP8 activations + blockwise fp32 scales through MoEEpTensors."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=False, fp8_scale_mode="blockwise"
    )
    print(
        f"rank {rank}: sm90_pull_fp8 mega layer (prestaged blockwise inputs) "
        "matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_fp8_mega_layer_in_kernel_fc2_reduce():
    """In-flight top-k combine (``in_kernel_fc2_reduce=True``) for SM90 FP8.

    The symm buffer allocates ``output_activation`` on the symmetric heap
    unconditionally (cross-rank REDG atomic-add target) and the shim zeroes it
    before every launch (accumulate-from-zero contract; the second forward
    inside ``_run_mega_layer`` would come back ~2x without it).
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        fp8_scale_mode="per_tensor",
        in_kernel_fc2_reduce=True,
    )
    print(
        f"rank {rank}: sm90_pull_fp8 mega layer (in_kernel_fc2_reduce) "
        "matches reference within tolerance"
    )


@pytest.mark.arch_hopper
def test_sm90_pull_fp8_preprocess_mega_weights_from_bf16():
    _require_cuda()

    import torch

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size, fp8_scale_mode="per_tensor")
    num_local_experts = problem["num_experts"] // world_size

    transformed_l1, transformed_l2 = _preprocess_weights(problem)

    fc1_weight, fc1_sf, fc1_act_scale, fc1_w_scale = transformed_l1
    fc2_weight, fc2_sf, fc2_act_scale, fc2_w_scale = transformed_l2
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
    # K-major invariant: GEMM K must be the stride-1 axis (dim 1).
    assert fc1_weight.stride(1) == 1
    assert fc2_weight.stride(1) == 1
    assert fc1_weight.dtype == torch.float8_e4m3fn
    assert fc2_weight.dtype == torch.float8_e4m3fn
    assert fc1_sf.shape[0] == num_local_experts
    assert fc2_sf.shape[0] == num_local_experts
    assert fc1_act_scale.shape == (1,) and fc1_act_scale.dtype == torch.float32
    assert fc2_act_scale.shape == (1,) and fc2_act_scale.dtype == torch.float32
    assert fc1_w_scale.shape == (num_local_experts,)
    assert fc2_w_scale.shape == (num_local_experts,)


def test_sm90_pull_fp8_mega_kernel_is_registered():
    from flashinfer.moe_ep import Sm90PullFp8MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Sm90PullFp8MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "sm90_pull_fp8"
