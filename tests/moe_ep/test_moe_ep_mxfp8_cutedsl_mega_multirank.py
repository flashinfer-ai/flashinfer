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

from flashinfer.moe_ep import FinalizeMode

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


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "tile_n,cluster_m,capacity",
    [(128, 4, 129), (64, 4, 129), (64, 2, 129), (64, 2, 1024)],
)
def test_blackwell_genphase_layer_graph_and_weight_formats(
    tile_n, cluster_m, capacity, monkeypatch
):
    """Public layer: both input formats, four weight formats and changing routes."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    _require_cuda()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires native SM100/103")
    rank, world = _launcher_ranks()
    if world not in (1, 2, 4):
        pytest.skip("qualified at EP1/2/4")
    if world == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    hidden, intermediate, topk, experts = 384, 192, 3, 8 * world
    generator = torch.Generator(device="cuda").manual_seed(6447 + rank)
    x = torch.randn(capacity, hidden, generator=generator, device="cuda").bfloat16()
    w13 = (
        torch.randn(8, 2 * intermediate, hidden, generator=generator, device="cuda")
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(8, hidden, intermediate, generator=generator, device="cuda")
        / intermediate**0.5
    ).bfloat16()
    qx, sx = pkg.quantize_genphase_mxfp8(x, torch.float8_e4m3fn)
    q13, s13 = pkg.quantize_genphase_mxfp8(w13, torch.float8_e4m3fn)
    q2, s2 = pkg.quantize_genphase_mxfp8(w2, torch.float8_e4m3fn)
    source = MoEWeightPack(w13, w2)
    packed = MoEWeightPack(q13, q2, s13, s2)
    byte_packed = MoEWeightPack(q13, q2, s13.view(torch.uint8), s2.view(torch.uint8))
    ids = (
        torch.arange(capacity * topk, device="cuda").reshape(capacity, topk) + rank * 7
    ) % experts
    scores = torch.full((capacity, topk), 1 / topk, device="cuda")
    layers = []

    def check(ws, weights, output, count):
        indices, expected = pkg.sampled_reference(ws, *weights, count, sample_size=32)
        error = torch.tensor(pkg.output_error(output, indices, expected), device="cuda")
        if world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error.item() < 0.02
        assert torch.count_nonzero(ws.output_activation[count:]) == 0

    try:
        baseline = None
        workspace = None
        for quantize_input in (True, False):
            for weight_format, weights in (
                ("bf16", source),
                ("prequantized", packed),
                ("byte_scales", byte_packed),
                ("transformed", source),
            ):
                layer = MoEEpLayer(
                    BootstrapConfig(world_size=world, rank=rank),
                    FleetParams(
                        num_experts=experts,
                        max_tokens_per_rank=capacity,
                        token_hidden_size=hidden,
                    ),
                    weights,
                    backend=MegaConfig(
                        megakernel=Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                            intermediate_size=intermediate,
                            top_k=topk,
                            kernel_variant="genphase",
                            mma_tiler_mnk=(256, tile_n, 128),
                            cluster_shape_mn=(cluster_m, 1),
                        ),
                        quantize_input=quantize_input,
                        preprocess_weights=weight_format != "transformed",
                        transformed_weights=layers[0]._transformed
                        if weight_format == "transformed"
                        else None,
                    ),
                )
                layers.append(layer)

                def inputs(count):
                    return MoEEpTensors(
                        hidden_states=x[:count] if quantize_input else qx[:count],
                        scales=None if quantize_input else sx[:count],
                        topk_ids=ids[:count].clone(),
                        topk_weights=scores[:count].clone(),
                    )

                dense = inputs(capacity)
                layer.warmup(dense)
                ws = layer._ensure_workspace()
                if workspace is None:
                    workspace = ws
                assert ws is workspace
                y = layer.forward(dense, return_workspace_view=True).clone()
                check(ws, layer._transformed, y, capacity)
                if baseline is None:
                    baseline = y
                else:
                    assert torch.equal(y, baseline)
                for count in (1, 0, min(65, capacity)):
                    current = inputs(count)
                    if count and world > 1 and rank == world - 1:
                        current.topk_ids.fill_(-1)
                        current.topk_weights.fill_(float("nan"))
                    ws._pre_reduced_activation.view(torch.bfloat16).fill_(float("nan"))
                    y = layer.forward(current, return_workspace_view=True)
                    check(ws, layer._transformed, y, count)

                current = inputs(min(65, capacity))
                layer.warmup(current)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=stream):
                    captured = layer.forward(current, return_workspace_view=True)
                torch.cuda.current_stream().wait_stream(stream)
                for state in ("empty", "holes", "restore"):
                    current.topk_ids.copy_(ids[: current.num_tokens])
                    if state == "empty":
                        current.topk_ids.fill_(-1)
                    elif state == "holes":
                        current.topk_ids[::2, 1:].fill_(-1)
                    # Grow eagerly before replaying the smaller graph, which
                    # must conservatively re-mask its entire capacity tail.
                    layer.forward(dense)
                    ws._pre_reduced_activation.view(torch.bfloat16).fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    check(ws, layer._transformed, captured, current.num_tokens)
                for _ in range(64):
                    graph.replay()
                torch.cuda.synchronize()
                check(ws, layer._transformed, captured, current.num_tokens)
    finally:
        for layer in reversed(layers):
            layer.destroy()


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
        enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
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


def _run_mega_layer_zero_token_ikr_regression(
    rank,
    world_size,
    *,
    num_iters: int = 60,
):
    """Interleave num_tokens=0 and real forward() calls, in_kernel_fc2_reduce=True,
    no barrier between iterations, at an independent per-rank schedule.

    Regression guard for mxfp8_mega_moe()'s num_tokens==0 shortcut, which used to
    return WITHOUT ever calling frontend.run() (i.e. without launching the kernel
    at all) when in_kernel_fc2_reduce was enabled:

        if n == 0 and symm_buffer._frontend.config.in_kernel_fc2_reduce:
            return symm_buffer.output_activation[:0] if y is None else None

    Sm100MegaMoEMxfp8Kernel is a persistent megakernel -- its CTA grid
    (MoEFusedFc12SchedulerParams.get_grid_shape -> (cluster_mn[0], cluster_mn[1],
    max_active_clusters)) is sized from hardware occupancy, never from
    num_tokens, so even a genuinely zero-token launch still runs every CTA and
    the warp-specialized dispatch / token-back / tail-cleanup logic that keeps
    a rank's cross-rank REDG atomic-add combine session in lockstep with its EP
    peers. A rank that takes the old shortcut instead silently skips that
    round's kernel launch, desynchronizing its session state from its peers'
    -- their subsequent launches then wait on a signal that rank never posts.

    This only manifests when DP/EP ranks call forward() independently (no
    cross-rank barrier between rounds, exactly how SGLang's per-rank scheduler
    loop drives it) AND some rank legitimately hits num_tokens==0 (SGLang's own
    idle-batch mechanism for keeping DP ranks in lockstep) while its peers have
    real work -- light/symmetric/all-nonzero testing never exercises it. See
    kernel_src/sm100/cutedsl_megamoe/shim/mxfp8.py::mxfp8_mega_moe.

    Shapes/scale intentionally match the real repro (hidden=2048,
    intermediate=768, num_experts=128, top_k=8, max_tokens_per_rank=16384 --
    the Qwen3-30B-A3B MXFP8 SGLang config that originally hit this), not this
    file's usual small test defaults: the same test at
    hidden=2048/intermediate=1024/num_experts=8/topk=4/max_tokens=64 (this
    file's ``_mega_problem`` default) plus a deterministic modulo-based
    zero/nonzero schedule passes vacuously even without the fix -- neither the
    smaller buffer/expert-count nor a merely-deterministic (as opposed to
    randomly-timed) schedule reproduces the desync on its own; both the scale
    and genuinely independent per-rank timing (via per-rank-seeded
    ``random.Random``, not a fixed formula) were needed to reproduce it.

    CAVEAT: a regression here manifests as a LIVELOCK (100% GPU utilization,
    zero forward progress, no exception, no crash), not a clean test failure --
    a same-process watchdog can't interrupt a rank frozen inside
    torch.cuda.synchronize() on a raw CUDA kernel wait (this isn't an NCCL
    collective op, so dist.init_process_group(timeout=...) doesn't cover it
    either). Rely on the CI job's own wall-clock timeout to catch a real
    regression.

    This exact scenario (matched shapes, matched random schedule, even a
    deliberate timing nudge on real rounds, NUM_ITERS=2000 -- the script
    defaults to 60) was confirmed to
    reliably livelock pre-fix when run as a plain torchrun-launched script
    (tests/moe_ep/../repro_ikr_zero_token_idle.py -- the authoritative
    regression artifact for this bug) but passes vacuously pre-fix when run
    under `torchrun -m pytest` specifically (root cause not fully isolated;
    pytest's execution environment appears to dampen the wall-clock
    divergence between ranks this race depends on). This test is kept as a
    documented, passing correctness check of the exact scenario under the
    project's normal test harness -- not as a guaranteed regression trap.
    """
    import random
    import time

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    hidden = 2048
    intermediate = 768
    num_experts = 128
    topk = 8
    max_tokens = 16384
    real_tokens = 4
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    warmup_hidden_states, warmup_topk_weights, warmup_topk_ids = _make_inputs(
        rank, num_tokens=real_tokens, hidden=hidden, num_experts=num_experts, topk=topk
    )
    megakernel_config = _megakernel_config(
        dict(
            intermediate=intermediate,
            topk=topk,
            kind="mxfp8_e4m3",
            gate_up_clamp=10.0,
            fast_math=True,
        ),
        in_kernel_fc2_reduce=True,
    )

    mega = MoEEpMegaLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(megakernel=megakernel_config, preprocess_weights=True),
    )
    try:
        # Matched-count collective warmup -- every rank calls forward() with
        # real tokens once, together, before the independent-cadence loop.
        mega.forward(
            MoEEpTensors(
                hidden_states=warmup_hidden_states,
                topk_ids=warmup_topk_ids,
                topk_weights=warmup_topk_weights,
            )
        )
        torch.cuda.synchronize()
        dist.barrier()

        # Independently-seeded per-rank RNG, no barrier between iterations:
        # rank 0 always real (mirrors an always-busy rank); other ranks
        # independently coin-flip zero/real every iteration, so each rank's
        # actual wall-clock cadence diverges from its peers' in a way a fixed
        # formula doesn't produce. Seeded for CI reproducibility.
        rnd = random.Random(4242 + rank)
        for it in range(num_iters):
            n = real_tokens if rank == 0 else (0 if rnd.random() < 0.5 else real_tokens)
            g = torch.Generator(device="cuda").manual_seed(1000 * it + rank)
            hidden_states = torch.randn(
                n, hidden, dtype=torch.bfloat16, device="cuda", generator=g
            )
            scores = torch.randn(
                n, num_experts, dtype=torch.float32, device="cuda", generator=g
            )
            topk_weights, topk_ids = torch.topk(
                scores, topk, dim=-1, largest=True, sorted=False
            )
            t = MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids.to(torch.int64),
                topk_weights=topk_weights.to(torch.float32),
            )
            if n > 0:
                # Nudge real per-rank wall-clock divergence: pre-fix, a
                # zero-token round skips the kernel launch entirely and is
                # near-instant, while a real round pays actual GPU cost --
                # under plain torchrun that gap alone is enough to desync
                # ranks within tens of rounds, but empirically not reliably
                # under pytest (unconfirmed why; see the CAVEAT above). This
                # doesn't guarantee detection here, just improves the odds.
                time.sleep(0.003)
            y = mega.forward(t)
            torch.cuda.synchronize()
            assert y.shape == (n, hidden)
            assert y.dtype == torch.bfloat16
            assert torch.isfinite(y).all()

        dist.barrier()
        return rank
    finally:
        mega.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mxfp8_cutedsl_mega_layer_in_kernel_fc2_reduce_zero_token_regression():
    """Zero-token / in_kernel_fc2_reduce livelock regression guard (MXFP8).

    See ``_run_mega_layer_zero_token_ikr_regression`` for the full bug
    writeup. Before the fix, this reliably livelocks within tens of
    iterations; after the fix, all ``num_iters`` complete cleanly regardless
    of each rank's independent zero/nonzero token schedule.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer_zero_token_ikr_regression(rank, world_size)
    print(
        f"rank {rank}: sm100_mxfp8_mxfp8_bf16_cutedsl mega layer survives "
        "interleaved zero-token/real in_kernel_fc2_reduce forward calls"
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
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
            enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
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
                apply_routing_weights_before_fc2=True,
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
    import torch

    from flashinfer.moe_ep import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        finalize_mode=FinalizeMode.PERSISTENT,
        num_valid_tokens_tensor=torch.zeros(1, dtype=torch.int32),
    )
    kernel = create_mega_kernel(config)
    assert kernel.kernel_name() == "sm100_mxfp8_mxfp8_bf16_cutedsl"
    assert config.finalize_mode is FinalizeMode.PERSISTENT
    with pytest.raises(ValueError, match="requires num_valid_tokens_tensor"):
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            finalize_mode=FinalizeMode.PERSISTENT,
        )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_blackwell_genphase_auto_warmup_cache_and_graph(monkeypatch):
    """Real collective sweep, stable I/O, persisted selection and changing routes."""
    import dataclasses
    import tempfile
    from pathlib import Path

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    _require_cuda()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/103")
    rank, world = _launcher_ranks()
    if world not in (1, 2, 4):
        pytest.skip("EP1/2/4")
    if world == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
    # Multi-rank CI runs on one node; every process reads the same cache file.
    path = (
        Path(tempfile.gettempdir())
        / f"genphase-auto-{os.environ.get('MASTER_PORT', os.getpid())}-{world}.json"
    )
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    capacity, tokens, hidden, intermediate, topk = 129, 17, 384, 192, 3
    experts = 8 * world
    fleet = FleetParams(
        num_experts=experts,
        max_tokens_per_rank=capacity,
        token_hidden_size=hidden,
    )
    generator = torch.Generator(device="cuda").manual_seed(717 + rank)
    weights = MoEWeightPack(
        (
            torch.randn(8, 2 * intermediate, hidden, generator=generator, device="cuda")
            / hidden**0.5
        ).bfloat16(),
        (
            torch.randn(8, hidden, intermediate, generator=generator, device="cuda")
            / intermediate**0.5
        ).bfloat16(),
    )
    x = torch.randn(tokens, hidden, generator=generator, device="cuda").bfloat16()
    ids = (
        torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) + 7 * rank
    ) % experts
    inputs = MoEEpTensors(
        hidden_states=x,
        topk_ids=ids.clone(),
        topk_weights=torch.full((tokens, topk), 1 / topk, device="cuda"),
    )
    layers = []
    try:

        def create(knobs):
            """Build a public layer sharing test weights with the requested knobs."""
            layer = MoEEpLayer(
                BootstrapConfig(rank=rank, world_size=world),
                fleet,
                weights,
                backend=MegaConfig(
                    megakernel=Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                        intermediate, topk, kernel_variant="genphase", knobs=knobs
                    )
                ),
            )
            layers.append(layer)
            return layer

        with pytest.warns(UserWarning, match="COLLECTIVE"):
            layer = create("auto")
        if rank == 0:
            path.unlink(missing_ok=True)
        if world > 1:
            dist.barrier()
        ws = layer._ensure_workspace()
        pointers = [
            getattr(ws, n).data_ptr()
            for n in ("x", "x_sf", "topk_idx", "topk_weights", "output_activation")
        ]
        from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
            knob_cache,
        )
        import time

        record_knobs = knob_cache.record_knobs
        tune = pkg.autotune_sm100_genphase_mega_moe

        def delayed_record(*args, **kwargs):
            """Delay rank zero's cache write to expose premature peer returns."""
            # Expose peers returning before a slow shared-cache write completes.
            if world > 1:
                time.sleep(0.25)
            return record_knobs(*args, **kwargs)

        def tune_and_check_publication(*args, **kwargs):
            """Require all ranks to observe the cached winner when tuning returns."""
            winner = tune(*args, **kwargs)
            resolved = knob_cache.resolve_knobs(
                ws.config,
                sm_count=torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count,
                live_tokens=tokens,
            )
            observed = [resolved]
            if world > 1:
                observed = [None] * world
                dist.all_gather_object(observed, resolved)
            assert all(value == (winner, "cache") for value in observed), (
                "cache publication incomplete at autotune return",
                observed,
            )
            return winner

        monkeypatch.setattr(knob_cache, "record_knobs", delayed_record)
        monkeypatch.setattr(
            pkg, "autotune_sm100_genphase_mega_moe", tune_and_check_publication
        )
        layer.warmup(inputs)
        winner = {
            k: getattr(ws.config, k)
            for k in ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        }
        assert pointers == [
            getattr(ws, n).data_ptr()
            for n in ("x", "x_sf", "topk_idx", "topk_weights", "output_activation")
        ]
        selected = layer.forward(inputs, return_workspace_view=True).clone()
        cached = create(None)
        assert cached.live_token_buckets(capacity) == [tokens]
        handle = cached.create_workspace(capacity, live_tokens=tokens)
        cached.warmup(inputs, workspace=handle)
        cached_ws = handle._backend_workspace
        assert all(getattr(cached_ws.config, k) == v for k, v in winner.items())
        assert torch.equal(
            selected,
            cached.forward(inputs, workspace=handle, return_workspace_view=True),
        )
        # Force the other geometry before capture to exercise workspace resizing
        # even when the heuristic happened to win the empirical sweep.
        alternative = dict(
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mn=(4, 1),
            schedule_policy=("grouped", None),
        )
        if ws.config.resolved_mma_tiler_mnk[1] == 128:
            alternative.update(mma_tiler_mnk=(256, 64, 128), cluster_shape_mn=(2, 1))
        ws.retune(alternative)
        assert dataclasses.replace(ws.config, **alternative) == ws.config
        assert pointers == [
            getattr(ws, n).data_ptr()
            for n in ("x", "x_sf", "topk_idx", "topk_weights", "output_activation")
        ]
        assert torch.equal(selected, layer.forward(inputs, return_workspace_view=True))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer.forward(inputs, return_workspace_view=True)
        for state in ("empty", "holes", "restored"):
            inputs.topk_ids.copy_(ids)
            if state == "empty":
                inputs.topk_ids.fill_(-1)
            elif state == "holes":
                inputs.topk_ids[::2, 1:] = -1
            for _ in range(65):
                graph.replay()
            torch.cuda.synchronize()
            indices, expected = pkg.sampled_reference(ws, *layer._transformed, tokens)
            error = torch.tensor(
                pkg.output_error(output, indices, expected), device="cuda"
            )
            if world > 1:
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
            assert error.item() < 0.02
        del graph
    finally:
        for layer in reversed(layers):
            layer.destroy()
        if rank == 0:
            path.unlink(missing_ok=True)
