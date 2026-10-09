"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (sm90_bf16_bf16_bf16_pull_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm90_pull_bf16_mega_multirank.py -v -m "gpu_4 and arch_hopper"

Requires Hopper (exactly sm_90), >=4 GPUs, and CuTeDSL runtime deps
(``nvidia-cutlass-dsl[cu13]``, ``nvshmem4py-cu13``).  Kernels ship in-tree under
``flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel``.

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

BF16 twin of ``test_moe_ep_sm90_pull_fp8_mega_multirank.py``.  Parity
methodology: each layer test drives the SAME fused kernel twice — once through
the full ``MoEEpLayer`` EP plumbing and once directly through the shim API
(``hopper_bf16_mega_moe`` on a fresh symm buffer) with identical staged inputs
and preprocessed weights — and asserts bit-exact equality for the
deterministic separate-reduce paths.  The ikr (REDG) test compares against the
explicit-reduce reference within the bf16 K-term accumulation band instead.

Torch-oracle anchor: parity alone cannot catch a kernel that is wrong but
self-consistent at ``world_size > 1`` (peer-pull addressing, expert→rank
ownership, cross-rank combine), because both sides run the same CUDA kernel.
``test_moe_ep_sm90_pull_bf16_mega_multirank_torch_oracle`` closes that gap:
every rank all-gathers the ACTUAL staged bf16 payload, routing, and
preprocessed weights, runs the drop's multi-rank-native pure-torch ground
truth (``compute_megamoe_reference_bf16``) on the global problem, and checks
its own rank's slice against the real-EP kernel output.  The single-GPU oracle
(``test_sm90_pull_bf16_kernel_vs_reference.py``) remains the
``world_size == 1`` anchor.

The activation is bf16 end to end (no quantization, no scale planes), so the
``quantize_input=True`` layer path and the "pre-staged" ``quantize_input=False``
path both take the same bf16 ``hidden_states`` with ``scales=None``.

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
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")


def _require_cuda():
    import torch

    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")


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
    seed: int = 7,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed + rank)
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
    forced = (
        torch.arange(min(topk, world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    if num_tokens > 0:
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
    swap_ab: bool = False,
    num_tokens: int = 64,
    max_tokens: int = 64,
    num_experts: int = 8,
    topk: int = 4,
    hidden: int = 2048,
):
    intermediate = 1024
    gate_up_clamp = 10.0
    fast_math = True

    # BF16 shim shape contract.
    assert hidden % 256 == 0
    assert intermediate % 64 == 0
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
        swap_ab=swap_ab,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
    )


def _preprocess_weights(problem: dict):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )

    return preprocess_mega_weights(
        MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )


def _alloc_symm_buffer(
    problem: dict, rank: int, world_size: int, *, generate_c: bool = False
):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_bf16_mega_moe,
    )

    return get_symm_buffer_for_hopper_bf16_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        swap_ab=problem["swap_ab"],
        gate_up_clamp=problem["gate_up_clamp"],
        generate_c=generate_c,
    )


def _reference_sm90_bf16_mega_moe_staged(problem: dict, *, destroy_buffer: bool = True):
    """Reference: direct shim launch with bf16 staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_bf16_mega_moe,
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
        symm_buffer.topk_idx,
        symm_buffer.topk_weights,
    )

    transformed_l1, transformed_l2 = _preprocess_weights(problem)

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    hopper_bf16_mega_moe(
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


def _reference_sm90_bf16_mega_moe_prestaged(
    problem: dict, x_bf16, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied bf16 activations copied in directly.

    Bypasses ``stage_mega_moe_inputs``: the rows and routing go straight into
    the symm buffer (whose ``topk_idx`` tail starts at the -1 pad mask).
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_bf16_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = _alloc_symm_buffer(problem, rank, world_size)
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].copy_(x_bf16)
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    transformed_l1, transformed_l2 = _preprocess_weights(problem)

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    hopper_bf16_mega_moe(
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

    Mirrors the FP8 twin: the ikr path accumulates the K per-topk bf16 terms
    in nondeterministic order vs the reference's explicit reduce, so where
    large terms nearly cancel the achievable agreement is bounded by the bf16
    round-off of the largest TERM, not of the final value.  Bound per row:
    K terms x bf16 eps (2^-8) x safety 8.  A missing per-launch output zero
    (2x accumulation) overshoots this band by ~64x.
    """
    import torch

    assert y.shape == y_ref.shape, (tuple(y.shape), tuple(y_ref.shape))
    if y.numel() == 0:
        return  # empty rank: nothing to compare
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
    *,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: str | None = None,
    active_dispatch_warps: int = 1,
    fold_producer_warps: bool | None = None,
    mma_tiler_mnk=None,
    pingpong=None,
    cluster_shape_mnk=None,
    tail_split_pairs: bool | None = None,
):
    from flashinfer.moe_ep import Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig

    # fold_producer_warps None -> the config's default (True); the fold tests
    # pin it explicitly.
    fold_kw = (
        {}
        if fold_producer_warps is None
        else {"fold_producer_warps": fold_producer_warps}
    )
    return Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(
        **fold_kw,
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        swap_ab=problem["swap_ab"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
        enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        active_dispatch_warps=active_dispatch_warps,
        mma_tiler_mnk=mma_tiler_mnk,
        pingpong=pingpong,
        cluster_shape_mnk=cluster_shape_mnk,
        tail_split_pairs=tail_split_pairs,
    )


def _run_mega_layer(
    rank,
    world_size,
    *,
    quantize_input: bool,
    swap_ab: bool = False,
    num_tokens: int = 64,
    max_tokens: int = 64,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: str | None = None,
    active_dispatch_warps: int = 1,
    fold_producer_warps: bool | None = None,
    mma_tiler_mnk=None,
    pingpong=None,
    cluster_shape_mnk=None,
    tail_split_pairs: bool | None = None,
    num_experts: int = 8,
    topk: int = 4,
    hidden: int = 2048,
    zero_token_ranks: tuple[int, ...] = (),
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
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    # ``zero_token_ranks`` stage an EMPTY local batch (0 rows) while the other
    # ranks keep ``num_tokens``; the empty rank must still serve its experts
    # and take part in the collective launch.
    if rank in zero_token_ranks:
        num_tokens = 0
    problem = _mega_problem(
        rank,
        world_size,
        swap_ab=swap_ab,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        hidden=hidden,
    )
    config_kwargs = dict(
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        active_dispatch_warps=active_dispatch_warps,
        fold_producer_warps=fold_producer_warps,
        mma_tiler_mnk=mma_tiler_mnk,
        pingpong=pingpong,
        cluster_shape_mnk=cluster_shape_mnk,
        tail_split_pairs=tail_split_pairs,
    )
    kernel = create_mega_kernel(_megakernel_config(problem, **config_kwargs))
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )

    try:
        # bf16 on both paths: quantize_input=False ("pre-staged") accepts the
        # same bf16 rows (nothing to quantize) and must not carry scales.
        t_hidden = problem["hidden_states"]

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
                megakernel=_megakernel_config(problem, **config_kwargs),
                quantize_input=quantize_input,
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        t = MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            scales=None,
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
            y_ref = _reference_sm90_bf16_mega_moe_staged(problem, destroy_buffer=True)
        else:
            y_ref = _reference_sm90_bf16_mega_moe_prestaged(
                problem, t_hidden, destroy_buffer=True
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
            # Same kernel, same staged operands, and the separate-reduce path
            # is deterministic -> bit-exact parity.
            torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
            torch.testing.assert_close(y_layer2, y_ref, atol=0.0, rtol=0.0)
        mega.destroy()
        return rank
    except BaseException:
        # Print the real failure before finalize: a kernel fault poisons the
        # CUDA context and nvshmem finalize then segfaults, which would
        # otherwise swallow this traceback.
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_bf16_mega_layer_matches_reference():
    """MoEEpMegaLayer (sm90_bf16_bf16_bf16_pull_cutedsl), default heuristic geometry."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True)
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer (staged inputs) "
        "matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_bf16_mega_layer_swap_ab_matches_reference():
    """Swap-AB geometry (manual-mode default (256, 32, 64) token-N tile)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True, swap_ab=True)
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer (swap_ab) "
        "matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "token_back_mode", ["reuse_dispatch_warps", "standalone_warps"]
)
def test_moe_ep_sm90_pull_bf16_mega_layer_token_back_matches_reference(
    token_back_mode,
):
    """Push-style fc2 write-back modes match the epi-warps-validated reference.

    ``reuse_dispatch_warps`` is an autotune candidate placement and
    ``standalone_warps`` a tuner knob value, so both need the same bit-level
    gate as the ``epi_warps`` default.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        token_back_mode=token_back_mode,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(token_back={token_back_mode}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("active_dispatch_warps", [2, 4])
def test_moe_ep_sm90_pull_bf16_mega_layer_active_dispatch_warps(active_dispatch_warps):
    """Non-default active pull-warp counts are bit-exact.

    The knob only re-partitions which dispatch warps issue the NVLink pulls
    (the default 1 is covered by every other case in this file), so all
    three settings must reproduce the reference exactly.  The fold layout
    self-gates off when more than one dispatch warp is active.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        active_dispatch_warps=active_dispatch_warps,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(active_dispatch_warps={active_dispatch_warps}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("swap_ab", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_layer_fold_producer_warps(swap_ab):
    """TMA-A/TMA-B/sched folded into the dispatch warpgroup (no producer WG).

    Exercises the merged warp layout with early fc1_done publication on the
    non-swap and swap-AB kernels.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        swap_ab=swap_ab,
        fold_producer_warps=True,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(fold_producer_warps, swap_ab={swap_ab}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("cluster_shape_mnk", [(1, 1, 1), (2, 2, 1)])
def test_moe_ep_sm90_pull_bf16_mega_layer_coop_n256(cluster_shape_mnk):
    """Non-swap cooperative M64N256 (2 epilogue WGs, no ping-pong).

    The BF16 heuristic table never selects N256, so this geometry (two WGMMA
    fragments per tile) is pinned here explicitly.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        swap_ab=False,
        num_tokens=512,
        max_tokens=512,
        mma_tiler_mnk=(64, 256, 64),
        pingpong=False,
        cluster_shape_mnk=cluster_shape_mnk,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(coop M64N256, cga={cluster_shape_mnk}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "case",
    [
        # (tile, pingpong, cga, token_back, num_tokens, tail_split_pairs) --
        # the swap-AB rows of moe_hopper_bf16/heuristic_config.py.
        ((128, 16, 64), False, (2, 1, 1), "epi_warps", 8, False),
        ((128, 16, 64), False, (1, 2, 1), "epi_warps", 16, False),
        ((128, 8, 64), False, (1, 1, 1), "epi_warps", 32, False),
        ((128, 8, 64), False, (1, 2, 1), "epi_warps", 128, False),
        ((128, 32, 64), False, (2, 1, 1), "epi_warps", 256, False),
        ((128, 64, 64), False, (1, 1, 1), "epi_warps", 512, False),
        ((128, 64, 64), True, (1, 2, 1), "epi_warps", 1024, True),
        ((128, 128, 64), True, (1, 2, 1), "epi_warps", 2048, True),
    ],
    ids=[
        "t8_basic_M128N16_cga21",
        "t16_basic_M128N16_cga12",
        "t32_basic_M128N8_cga11",
        "t128_basic_M128N8_cga12",
        "t256_basic_M128N32_cga21",
        "t512_basic_M128N64_cga11",
        "t1024_pp_M128N64_cga12_split",
        "t2048_pp_M128N128_cga12_split",
    ],
)
def test_moe_ep_sm90_pull_bf16_mega_layer_heuristic_rows(case):
    """Bit-exact check of the BF16 heuristic table's geometries.

    The multirank tests otherwise pass explicit geometry (manual mode) or run
    the 64-token bucket only, so every distinct table row is pinned here with
    its exact tile / ping-pong / cluster shape / token-back / tail split.
    """
    tile, pingpong, cga, token_back, num_tokens, tail_split = case
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        swap_ab=True,
        num_tokens=num_tokens,
        max_tokens=max(num_tokens, 64),
        token_back_mode=token_back,
        mma_tiler_mnk=tile,
        pingpong=pingpong,
        cluster_shape_mnk=cga,
        tail_split_pairs=tail_split,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(heuristic row swap tile={tile} pp={pingpong} cga={cga} "
        f"tb={token_back} tokens={num_tokens} split={tail_split}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "case",
    [
        # (swap_ab, tile, pingpong, cga, token_back): the swap-AB ping-pong
        # M128N128 cga(1,2,1) geometry of the 1024+ heuristic rows and the
        # non-swap cooperative M64N256 cga(2,1,1) tail-split geometry, under
        # both token-back placements (the dispatch-driven one gates on the
        # fc2_done count the split tail changes).
        (True, (128, 128, 64), True, (1, 2, 1), "epi_warps"),
        (True, (128, 128, 64), True, (1, 2, 1), "reuse_dispatch_warps"),
        (False, (64, 256, 64), False, (2, 1, 1), "epi_warps"),
        (False, (64, 256, 64), False, (2, 1, 1), "reuse_dispatch_warps"),
    ],
    ids=[
        "pp_M128N128_cga12_epi",
        "pp_M128N128_cga12_reuse",
        "coop_N256_cga21_epi",
        "coop_N256_cga21_reuse",
    ],
)
def test_moe_ep_sm90_pull_bf16_mega_layer_tail_split_pairs(case):
    """Bit-exact check of the tail-split pair tasks against the reference.

    1088 tokens per rank over 8 experts / top-4 give ~544 rows per expert:
    4.25 swap-AB N=128 token tiles and 8.5 non-swap M=64 tiles, so every
    expert ends in an odd CTA-tile count and its tail cluster block runs as
    pair tasks (both CTAs on the single valid token tile, adjacent weights).
    """
    swap_ab, tile, pingpong, cga, token_back = case
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        swap_ab=swap_ab,
        num_tokens=1088,
        max_tokens=1088,
        token_back_mode=token_back,
        mma_tiler_mnk=tile,
        pingpong=pingpong,
        cluster_shape_mnk=cga,
        tail_split_pairs=True,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        f"(tail-split pair tasks swap={swap_ab} tile={tile} "
        f"pp={pingpong} cga={cga} tb={token_back}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "case",
    [
        ((128, 8, 64), True, (2, 1, 1)),
        ((256, 8, 64), False, (2, 1, 1)),
        ((256, 8, 64), False, (1, 1, 1)),
    ],
    ids=["pp_M128N8_cga21", "coop_M256N8_cga21", "coop_M256N8_cga11"],
)
def test_moe_ep_sm90_pull_bf16_mega_layer_swapab_token_tile_8(case):
    """Swap-AB token tile N=8 (wgmma m64n8k16) bit-exact check."""
    tile, pingpong, cga = case
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        swap_ab=True,
        num_tokens=16,
        mma_tiler_mnk=tile,
        pingpong=pingpong,
        cluster_shape_mnk=cga,
    )
    print(
        f"rank {rank}: swap-AB token tile 8 (tile={tile} pp={pingpong} "
        f"cga={cga}) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_bf16_mega_layer_prestaged_inputs_matches_reference():
    """``quantize_input=False``: pre-staged bf16 activations, ``scales=None``."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=False)
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        "(prestaged bf16 inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("in_kernel_fc2_reduce", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_layer_zero_token_rank(in_kernel_fc2_reduce):
    """One rank stages an empty batch while the others route tokens to it.

    Regression guard for the shim's former ``num_tokens == 0`` early return
    (in_kernel_fc2_reduce): the launch is collective, so an empty rank must
    still launch the padded buffer (all ``topk_idx == -1``), serve its
    experts to the peers' pulls and reach every cross-rank barrier.  Skipping
    it hung the non-empty ranks.  The non-empty ranks' outputs must match the
    reference exactly (separate reduce) / within the ikr band.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        zero_token_ranks=(0,),
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer with an "
        f"empty rank 0 (in_kernel_fc2_reduce={in_kernel_fc2_reduce}) matches "
        "reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_bf16_mega_layer_in_kernel_fc2_reduce():
    """In-flight top-k combine (``in_kernel_fc2_reduce=True``) for SM90 BF16.

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
        in_kernel_fc2_reduce=True,
    )
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega layer "
        "(in_kernel_fc2_reduce) matches reference within tolerance"
    )


def _all_gather_stack(t):
    """all_gather a per-rank tensor and stack it on a new leading rank dim."""
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    gathered = [torch.empty_like(tc) for _ in range(world_size)]
    dist.all_gather(gathered, tc)
    return torch.stack(gathered)


def _check_generate_c_output(fc1_c, ref_map, idx_g, rank, num_local_experts):
    """generate_c: compare the kernel's fc1_c pool with the reference gate+up.

    Rows inside an expert segment follow the dispatch arrival order, so each
    expert is compared as a sorted flat array (the drop runner's recipe); the
    128-row segment offsets are rebuilt from the global routing, and the pad
    rows must have stayed zero.
    """
    import torch

    assert fc1_c is not None, "generate_c=True but fc1_c is None"
    expert_start = rank * num_local_experts
    counts = [
        int((idx_g == expert_start + e).sum().item()) for e in range(num_local_experts)
    ]
    offsets = [0]
    for v in counts:
        offsets.append(offsets[-1] + ((v + 127) // 128) * 128)
    assert offsets[-1] <= fc1_c.shape[0], (offsets[-1], fc1_c.shape)
    checked = 0
    for e in range(num_local_experts):
        v = counts[e]
        ref = ref_map.get(expert_start + e)
        if v == 0 or ref is None:
            continue
        rows = fc1_c[offsets[e] : offsets[e] + v]
        assert rows.shape == ref.shape, (e, tuple(rows.shape), tuple(ref.shape))
        kernel_c = rows.float().flatten().sort().values
        ref_c = ref.to(rows.device).float().flatten().sort().values
        torch.testing.assert_close(kernel_c, ref_c, atol=1e-2, rtol=1e-2)
        pad = fc1_c[offsets[e] + v : offsets[e + 1]]
        assert pad.numel() == 0 or pad.abs().max().item() == 0.0, (
            f"expert {e}: non-zero pad rows"
        )
        checked += 1
    # Rows past the last live segment are padding too; on a second launch with
    # fewer tokens they would otherwise hold the previous launch's values.
    tail = fc1_c[offsets[-1] :]
    assert tail.numel() == 0 or tail.abs().max().item() == 0.0, (
        f"{int((tail != 0).sum().item())} non-zero elements past the last "
        "live expert segment"
    )
    assert checked > 0, "no local expert received tokens"
    return checked


def _run_mega_torch_oracle(
    rank,
    world_size,
    *,
    swap_ab=False,
    generate_c=False,
    relaunch_token_counts: tuple[int, ...] = (),
):
    """Real-EP kernel launch vs the drop's pure-torch GLOBAL reference.

    Every rank stages its own bf16 shard, runs the fused kernel with real
    cross-rank NVSHMEM pulls, then all-gathers the ACTUAL staged bf16
    activations + routing + preprocessed K-major weights (no reliance on
    cross-rank RNG determinism) and feeds the global problem to
    ``compute_megamoe_reference_bf16`` — which is multi-rank native: it takes
    ``(num_ranks, tokens_per_rank, ...)`` operands and computes
    ``expert(topk_idx[r, t, k])`` across rank boundaries.  Each rank asserts
    its own output slice within the single-GPU oracle's tolerances.

    ``relaunch_token_counts`` re-stages the SAME session (symm buffer, compiled
    kernel, ``fc1_c`` pool) with fresh routing of that many tokens per rank and
    repeats the full check after each launch -- the regression guard for state
    that must not leak between launches (``fc1_c`` pad rows).
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        compute_megamoe_reference_bf16,
        hopper_bf16_mega_moe,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size, swap_ab=swap_ab)
    kernel = create_mega_kernel(_megakernel_config(problem))
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )
    try:
        hidden = problem["hidden"]

        symm_buffer = _alloc_symm_buffer(
            problem, rank, world_size, generate_c=generate_c
        )
        try:
            transformed_l1, transformed_l2 = _preprocess_weights(problem)
            # Keep the weights K-major across the gather: ship the contiguous
            # transpose and transpose back so the gather does not silently
            # re-stride them to row-major.
            fc1_w_g = _all_gather_stack(transformed_l1.mT).mT  # (R, E_local, H, 2I)
            fc2_w_g = _all_gather_stack(transformed_l2.mT).mT  # (R, E_local, I, H)

            def launch_and_check(hidden_states, topk_weights, topk_ids, tag):
                n = hidden_states.shape[0]
                stage_mega_moe_inputs(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    symm_buffer.x,
                    symm_buffer.topk_idx,
                    symm_buffer.topk_weights,
                )
                # Snapshot exactly what the kernel consumes (this rank's shard).
                x_local = symm_buffer.x[:n].clone()

                y_kernel = torch.empty(n, hidden, dtype=torch.bfloat16, device="cuda")
                hopper_bf16_mega_moe(
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

                # Reassemble the global problem from the operands each rank
                # staged.
                x_g = _all_gather_stack(x_local)  # (R, n, hidden) bf16
                idx_g = _all_gather_stack(topk_ids)  # (R, n, K) int64
                w_g = _all_gather_stack(topk_weights)  # (R, n, K) fp32

                combine_ref = compute_megamoe_reference_bf16(
                    input_activation=x_g,
                    input_topk_idx=idx_g,
                    input_topk_weights=w_g,
                    fc1_weight=fc1_w_g,
                    fc2_weight=fc2_w_g,
                    ref_compute_graph="deepgemm",  # matches apply_topk_in_fc1
                    fc2_output_dtype=torch.bfloat16,
                    gate_up_clamp=problem["gate_up_clamp"],
                    return_fc1_gateup=generate_c,
                )
                # deepgemm graph folds topk weights before the bf16 fc1-out
                # store, so the per-topk terms reduce with a plain sum; compare
                # this rank's slice.
                fc1_gateup_ref = None
                if generate_c:
                    combine_ref, fc1_gateup_ref = combine_ref
                y_ref = combine_ref[rank].to(torch.float32).sum(dim=1)

                assert torch.isfinite(y_kernel).all()
                yk = y_kernel.to(torch.float32)
                rel_l2 = (yk - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
                print(
                    f"[sm90 bf16 multirank oracle rank {rank} swap_ab={swap_ab} "
                    f"{tag}] rel_l2={rel_l2.item():.4g} "
                    f"max|d|={(yk - y_ref).abs().max().item():.4g} "
                    f"amax(ref)={y_ref.abs().max().item():.4g}"
                )
                # Single-GPU oracle tolerances (drop mega_runner:
                # atol=rtol=1e-2), valid because the problem is conditioned to
                # O(1) outputs and kernel + reference share the same gathered
                # bf16 operands.
                torch.testing.assert_close(yk, y_ref, atol=1e-2, rtol=1e-2)
                assert rel_l2.item() < 0.02
                if generate_c:
                    checked = _check_generate_c_output(
                        symm_buffer.fc1_c,
                        fc1_gateup_ref,
                        idx_g,
                        rank,
                        problem["num_experts"] // world_size,
                    )
                    print(
                        f"[sm90 bf16 generate_c rank {rank} swap_ab={swap_ab} "
                        f"{tag}] fc1_c matches the reference gate+up for "
                        f"{checked} local experts"
                    )

            launch_and_check(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                f"launch 0 ({problem['num_tokens']} tok)",
            )
            for i, count in enumerate(relaunch_token_counts, start=1):
                assert 0 < count <= problem["max_tokens"], count
                hs, tw, ti = _make_inputs(
                    rank,
                    world_size,
                    num_tokens=count,
                    hidden=hidden,
                    num_experts=problem["num_experts"],
                    topk=problem["topk"],
                    seed=101 * i,
                )
                launch_and_check(hs, tw, ti, f"launch {i} ({count} tok)")
            return rank
        finally:
            # A failing rank must still free its symmetric-heap slice;
            # leaking it turns a clean failure into a multi-rank hang.
            symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("swap_ab", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_multirank_torch_oracle(swap_ab):
    """Real cross-rank EP kernel vs pure-torch global math (see helper doc)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(rank, world_size, swap_ab=swap_ab)
    print(
        f"rank {rank}: sm90_bf16_bf16_bf16_pull_cutedsl mega kernel "
        f"(swap_ab={swap_ab}) matches the multi-rank torch oracle"
    )


@pytest.mark.arch_hopper
def test_sm90_pull_bf16_preprocess_mega_weights_from_bf16():
    _require_cuda()

    import torch

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_local_experts = problem["num_experts"] // world_size

    fc1_weight, fc2_weight = _preprocess_weights(problem)

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
    assert fc1_weight.dtype == torch.bfloat16
    assert fc2_weight.dtype == torch.bfloat16


def test_sm90_pull_bf16_mega_kernel_is_registered():
    from flashinfer.moe_ep import Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "sm90_bf16_bf16_bf16_pull_cutedsl"


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("swap_ab", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_multirank_generate_c(swap_ab):
    """Training forward (generate_c=True): the raw pre-SwiGLU fc1 gate+up
    pool written by the kernel matches the multi-rank torch reference for
    every local expert, on both layouts, while the combined output still
    matches the oracle."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_mega_torch_oracle(rank, world_size, swap_ab=swap_ab, generate_c=True)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("swap_ab", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_multirank_generate_c_shrinking_routing(swap_ab):
    """generate_c pad-rows-zero contract ACROSS launches on one session.

    Launch 64 tokens/rank, then re-stage the same session with 1 token/rank
    (token 0 is forced onto one expert per rank, so every rank's local expert
    0 drops from ~128 rows to 4 while the rest drop to 0), then 16.  Without
    the per-launch ``fc1_c`` zero the rows that became padding keep the
    previous launch's activations; each launch is checked against the torch
    oracle including the pad rows and the tail past the last live segment.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_mega_torch_oracle(
        rank,
        world_size,
        swap_ab=swap_ab,
        generate_c=True,
        relaunch_token_counts=(1, 16),
    )
