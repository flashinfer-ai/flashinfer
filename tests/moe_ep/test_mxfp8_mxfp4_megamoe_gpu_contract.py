# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-B200 staging, graph, and gated full-kernel contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch


def _require_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MXFP8 x MXFP4 GPU contracts need CUDA")
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")


def _buffers(capacity: int, hidden: int, topk: int):
    return (
        torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
        torch.empty(
            (capacity, ((hidden // 32 + 3) // 4) * 4),
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        ),
        torch.empty((capacity, topk), dtype=torch.int64, device="cuda"),
        torch.empty((capacity, topk), dtype=torch.float32, device="cuda"),
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33])
def test_mxfp8_mxfp4_megamoe_prequantized_staging_is_byte_exact(tokens: int) -> None:
    _require_b200()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.staging import stage_inputs

    capacity, hidden, topk = 64, 160, 4
    data = torch.randn((tokens, hidden), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(
        1, 254, (tokens, hidden // 32), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
            tokens, topk
        )
        % 8
    )
    scores = torch.randn((tokens, topk), device="cuda", dtype=torch.float32)
    out = _buffers(capacity, hidden, topk)
    staged = stage_inputs(data, scales, ids, scores, *out, quantize_input=False)
    torch.cuda.synchronize()
    assert staged.num_tokens == tokens
    assert torch.equal(staged.data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(staged.scales.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(staged.route_ids, ids.to(torch.int64))
    assert torch.equal(staged.route_scores, scores)
    if tokens < capacity:
        assert bool((out[2][tokens:] == -1).all())


@pytest.mark.arch_blackwell
def test_mxfp8_mxfp4_megamoe_frontend_requires_warmup_before_graph_capture() -> None:
    _require_b200()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.frontend import (
        FrontendConfig,
        KernelLaunchSpec,
        Mxfp8Mxfp4Frontend,
    )

    class Adapter:
        def __init__(self) -> None:
            self.compiles = 0
            self.binds = 0

        def compile(self, config, spec):
            self.compiles += 1
            return object()

        def bind(self, compiled, config, spec):
            self.binds += 1

            def launch():
                spec.output[: spec.num_tokens].copy_(
                    spec.activation[: spec.num_tokens].to(torch.bfloat16)
                )

            return launch

    hidden, intermediate, topk, capacity, tokens = 128, 128, 2, 8, 3
    activation, scales, ids, scores = _buffers(capacity, hidden, topk)
    activation.copy_(torch.randn((capacity, hidden), device="cuda"))
    tactic = (256, 128, 128)
    config = FrontendConfig(hidden, intermediate, topk, 2, capacity, tactic)
    spec = KernelLaunchSpec(
        activation=activation,
        activation_scales=scales,
        fc1_weight=torch.zeros(
            (2, hidden // 2, 2 * intermediate), dtype=torch.uint8, device="cuda"
        ),
        fc1_weight_scales=torch.zeros(
            (2, 1), dtype=torch.float8_e8m0fnu, device="cuda"
        ),
        fc2_weight=torch.zeros(
            (2, intermediate // 2, hidden), dtype=torch.uint8, device="cuda"
        ),
        fc2_weight_scales=torch.zeros(
            (2, 1), dtype=torch.float8_e8m0fnu, device="cuda"
        ),
        route_ids=ids,
        route_scores=scores,
        output=torch.empty((capacity, hidden), dtype=torch.bfloat16, device="cuda"),
        workspace_regions={
            "scratch": torch.zeros(16, dtype=torch.uint8, device="cuda")
        },
        num_tokens=tokens,
        stream=torch.cuda.current_stream().cuda_stream,
        static_tactic=tactic,
    )
    adapter = Adapter()
    frontend = Mxfp8Mxfp4Frontend(config, adapter)
    frontend.warmup(spec)
    assert (adapter.compiles, adapter.binds) == (1, 1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = frontend.run(spec)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, activation[:tokens].to(torch.bfloat16))
    assert (adapter.compiles, adapter.binds) == (1, 1)

    missed = replace(spec, output=torch.empty_like(spec.output))
    with (
        pytest.raises(RuntimeError, match="warmup"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        frontend.run(missed)


def _full_kernel_case(
    tokens,
    hidden,
    intermediate,
    experts,
    topk,
    *,
    knobs=None,
    capacity_tactics=None,
    check_reference=True,
):
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        get_symm_buffer_for_mxfp8_mxfp4_mega_moe,
        mxfp8_mxfp4_mega_moe,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
        compute_reference,
        interleave_gate_up_8,
        quantize_mxfp4,
        quantize_mxfp8,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        to_atom_swizzled,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260922)

    def random(shape, scale):
        return torch.randn(shape, generator=generator, device="cuda") * scale

    def quantize_weights(weights):
        # The exhaustive FP4 oracle materializes 16 distances per value.
        # Bound its temporary memory for large multi-expert weights.
        if weights.numel() > 256 * 1024 * 1024:
            quantized = [quantize_mxfp4(expert) for expert in weights]
            return tuple(torch.stack(parts) for parts in zip(*quantized, strict=True))
        return quantize_mxfp4(weights)

    xq, xs = quantize_mxfp8(random((tokens, hidden), 0.5))
    w13q, w13s = quantize_weights(
        interleave_gate_up_8(random((experts, 2 * intermediate, hidden), 0.125))
    )
    w2q, w2s = quantize_weights(random((experts, hidden, intermediate), 0.125))
    ids = torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) % experts
    # Empty experts, masked routes, nonuniform scores, and partial token tiles.
    if experts > topk:
        ids %= experts - 1
    if tokens:
        ids[-1, -1] = -1
    scores = random((tokens, topk), 0.3)
    expected = None
    if check_reference:
        reference = compute_reference(
            input_q=xq.unsqueeze(0),
            input_scale=xs.unsqueeze(0),
            topk_ids=ids.unsqueeze(0),
            topk_weights=scores.unsqueeze(0),
            w13_packed=w13q,
            w13_scale=w13s,
            w2_packed=w2q,
            w2_scale=w2s,
        )
        expected = reference.output[0]
    workspace = get_symm_buffer_for_mxfp8_mxfp4_mega_moe(
        experts,
        max(tokens, 1),
        topk,
        hidden,
        intermediate,
        0,
        1,
        knobs=knobs,
        capacity_tactics=capacity_tactics,
    )
    workspace.x[:tokens].copy_(xq)
    workspace.x_sf[:tokens].view(torch.uint8).copy_(xs)
    workspace.topk_idx[:tokens].copy_(ids)
    workspace.topk_weights[:tokens].copy_(scores)
    w13 = (w13q.transpose(1, 2), torch.stack([to_atom_swizzled(s) for s in w13s]))
    w2 = (w2q.transpose(1, 2), torch.stack([to_atom_swizzled(s) for s in w2s]))

    def launch(output=None):
        return mxfp8_mxfp4_mega_moe(output, w13, w2, workspace, num_tokens=tokens)

    return workspace, launch, expected


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("epilogue_warps", [4, 8])
@pytest.mark.parametrize("tile_tokens", [64, 128, 192])
def test_fc2_staged_stores_bitwise(epilogue_warps, tile_tokens):
    _require_b200()
    knobs = dict(
        mma_tiler_mnk=(256, tile_tokens, 128),
        epilogue_warps=epilogue_warps,
    )
    workspace, launch, expected = _full_kernel_case(200, 256, 256, 4, 2, knobs=knobs)
    try:
        baseline = launch().clone()
        torch.testing.assert_close(baseline, expected, rtol=0.05, atol=0.015)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = launch()
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "load_balance_mode", ["static", "atomic_counter", "alternating"]
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 128, 128, 1, 1),
        (33, 256, 256, 4, 2),
        (129, 256, 128, 4, 2),
        (0, 128, 128, 2, 1),
    ],
)
def test_mxfp8_mxfp4_megamoe_full_kernel_boundary_shapes(
    shape, load_balance_mode
) -> None:
    _require_b200()
    workspace, launch, expected = _full_kernel_case(
        *shape,
        knobs={
            "load_balance_mode": load_balance_mode,
        },
    )
    try:
        for _ in range(3):
            actual = launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "load_balance_mode", ["static", "atomic_counter", "alternating"]
)
@pytest.mark.parametrize(
    "tokens,tile_tokens,cluster_shape",
    [
        (33, 64, (1, 1, 1)),
        # More tiles than resident clusters exercises repeated atomic claims.
        (4097, 128, (1, 1, 1)),
        (129, 128, (2, 1, 1)),
    ],
)
def test_mxfp8_mxfp4_megamoe_full_kernel_cuda_graph_replay(
    load_balance_mode,
    tokens,
    tile_tokens,
    cluster_shape,
) -> None:
    _require_b200()
    workspace, launch, expected = _full_kernel_case(
        tokens,
        256,
        256,
        4,
        2,
        knobs={
            "load_balance_mode": load_balance_mode,
            "mma_tiler_mnk": (128, tile_tokens, 128),
            "cluster_shape_mnk": cluster_shape,
            "group_hint": 1,
        },
    )
    try:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            baseline = launch().clone()
            for _ in range(2):
                torch.testing.assert_close(launch(), expected, atol=0.015, rtol=0.05)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            actual = launch()
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("epilogue_warps", [4, 8])
@pytest.mark.parametrize(
    "load_balance_mode", ["static", "atomic_counter", "alternating"]
)
@pytest.mark.parametrize(
    "tokens,hidden,mma_m,tile_tokens",
    [
        (0, 128, 128, 64),
        (33, 384, 128, 64),
        (193, 256, 256, 192),
        (4097, 256, 256, 128),
        (17, 4096, 256, 128),
        (17, 6400, 128, 64),
    ],
)
def test_megamoe_in_kernel_reduce_is_bit_exact(
    tokens,
    hidden,
    mma_m,
    tile_tokens,
    epilogue_warps,
    load_balance_mode,
) -> None:
    _require_b200()
    baseline = None
    for in_kernel_reduce in (False, True):
        workspace, launch, expected = _full_kernel_case(
            tokens,
            hidden,
            256,
            4,
            2,
            knobs={
                "in_kernel_reduce": in_kernel_reduce,
                "epilogue_warps": epilogue_warps,
                "load_balance_mode": load_balance_mode,
                "mma_tiler_mnk": (mma_m, tile_tokens, 128),
                "group_hint": 1,
                # A small pinned operand ring as well as the automatically
                # fitted rings of the smaller cases.
                **({"num_stages": 2} if hidden >= 4096 else {}),
            },
        )
        try:
            actual = launch().clone()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            else:
                assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = launch()
            # Replays alternate live, masked, and restored routes without
            # clearing route terms. Masked NaN scores must never contribute.
            ids = workspace.topk_idx.clone()
            scores = workspace.topk_weights.clone()
            for live in (True, False, True, False, True):
                if live:
                    workspace.topk_idx.copy_(ids)
                    workspace.topk_weights.copy_(scores)
                else:
                    workspace.topk_idx.fill_(-1)
                    workspace.topk_weights.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                expected_bits = baseline if live else torch.zeros_like(output)
                assert torch.equal(
                    output.view(torch.uint8), expected_bits.view(torch.uint8)
                )
            del graph
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("topk", [1, 8, 32])
@pytest.mark.parametrize("epilogue_warps", [4, 8])
def test_megamoe_in_kernel_reduce_topk_limits(topk, epilogue_warps) -> None:
    _require_b200()
    baseline = None
    for enabled in (False, True):
        workspace, launch, expected = _full_kernel_case(
            33,
            256,
            256,
            32,
            topk,
            knobs={
                "in_kernel_reduce": enabled,
                "epilogue_warps": epilogue_warps,
            },
        )
        try:
            actual = launch().clone()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            else:
                assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tile_tokens", [64, 128, 192])
@pytest.mark.parametrize("mma_m", [128, 256])
def test_megamoe_eight_warp_epilogue_is_bit_exact(tile_tokens, mma_m) -> None:
    _require_b200()
    baseline = None
    for warps in (4, 8):
        workspace, launch, expected = _full_kernel_case(
            193,
            256,
            256,
            4,
            2,
            knobs={
                "epilogue_warps": warps,
                "mma_tiler_mnk": (mma_m, tile_tokens, 128),
                "cluster_shape_mnk": (2 if mma_m == 256 else 1, 1, 1),
                "load_balance_mode": "atomic_counter",
                "group_hint": 1,
            },
        )
        try:
            actual = launch().clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            else:
                assert torch.equal(actual, baseline)
            for _ in range(3):
                assert torch.equal(launch(), baseline)
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("claim_order", ["expert", "longest_first"])
@pytest.mark.parametrize("feature_ctas", [1, 2])
def test_alternating_device_decode_matches_host(feature_ctas, claim_order):
    """Replay fixed phase-local counter outcomes on both cluster CTAs."""
    _require_b200()
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_scheduler import (
        ALTERNATING_DONE,
        DeviceSchedulerParams,
        PersistentDeviceScheduler,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.scheduler import (
        PersistentFc1Fc2Scheduler,
        SchedulerConfig,
    )

    config = SchedulerConfig(
        expert_token_counts=(0, 65, 0, 17, 0),
        gate_up_size=512,
        hidden_size=256,
        cta_tile_m=64,
        cta_tile_n=128,
        cluster_shape_mn=(1, feature_ctas),
        group_hint=1,
        token_padding_block=64,
        sf_padding_block=128,
        num_persistent_clusters=1,
        fc2_ready_threshold=4,
        claim_order=claim_order,
    )
    model = PersistentFc1Fc2Scheduler(config)
    # Monotone within each phase, with independent cursors across empty
    # experts and distinct data/SF padding. Include gaps as atomic peers do.
    claims = [
        0,
        1,
        -1,
        3,
        -2,
        3 * config.fc1_feature_blocks - 1,
        -(3 * config.fc2_feature_blocks),
        ALTERNATING_DONE,
    ]
    counts = torch.tensor(config.expert_token_counts, device="cuda", dtype=torch.int32)
    indices = torch.tensor(claims, device="cuda", dtype=torch.int32)
    counters = torch.zeros(32, device="cuda", dtype=torch.int32)
    output = torch.empty(
        (feature_ctas, len(claims), 8), device="cuda", dtype=torch.int32
    )

    @cute.kernel
    def kernel(counts, indices, counters, output):
        params = DeviceSchedulerParams(
            expert_token_sizes=counts,
            expert_count=5,
            gate_up_size=512,
            hidden_size=256,
            cta_tile_tokens=64,
            cta_tile_features=128,
            cluster_shape_token_feature=(1, feature_ctas),
            group_hint=1,
            token_padding_block=64,
            sf_padding_block=128,
            fc1_ready_counter_ptr=counters.iterator,
            fc1_done_counter_ptr=counters.iterator,
            fc2_ready_threshold=4,
            load_balance_mode="alternating",
            claim_order=claim_order,
            load_balance_counter_ptr=counters.iterator,
        )
        storage = utils.SmemAllocator().allocate(
            PersistentDeviceScheduler.make_storage_struct(params)
        )
        scheduler = PersistentDeviceScheduler.create(
            params, cute.arch.block_idx(), cute.arch.grid_dim(), storage, 32
        )
        scheduler.internal_init(cutlass.Int32(0), 0)
        bx, _, _ = cute.arch.block_idx()
        for i in cutlass.range(len(claims), unroll=1):
            scheduler._decode_alternating(indices[i])
            scheduler = scheduler
            words = scheduler.current_work.to_rmem()
            if cute.arch.lane_idx() == 0:
                for j in cutlass.range_constexpr(8):
                    output[bx, i, j] = words[j]

    @cute.jit
    def launch(counts, indices, counters, output):
        kernel(counts, indices, counters, output).launch(
            grid=(feature_ctas, 1, 1),
            block=(32, 1, 1),
            cluster=(feature_ctas, 1, 1),
        )

    args = [from_dlpack(t) for t in (counts, indices, counters, output)]
    cute.compile(launch, *args)(*args)
    torch.cuda.synchronize()
    expected = [
        [
            model.record_for_alternating(c, cta_coord_mn=(0, f)).to_words()
            for c in claims
        ]
        for f in range(feature_ctas)
    ]
    assert output.cpu().tolist() == [[list(w) for w in row] for row in expected]


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "shape",
    [
        (193, 256, 4096, 4, 2),
        (4097, 256, 256, 4, 2),
        # This 32-expert geometry with FC1/FC2 widths 50/16 leaves work after
        # warm-up, exercising actual phase alternation on a full B200 grid.
        (4, 4096, 6400, 32, 8),
    ],
)
@pytest.mark.parametrize("epilogue_warps", [4, 8])
def test_alternating_order_is_bit_exact_to_static(shape, epilogue_warps):
    _require_b200()
    baseline = None
    for mode in ("static", "alternating"):
        workspace, launch, expected = _full_kernel_case(
            *shape,
            knobs={
                "load_balance_mode": mode,
                "epilogue_warps": epilogue_warps,
                "group_hint": 1,
            },
        )
        try:
            actual = launch().clone()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            else:
                assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
            for _ in range(3):
                assert torch.equal(
                    launch().view(torch.uint8), baseline.view(torch.uint8)
                )
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("epilogue_warps", [4, 8])
def test_alternating_claims_without_reference_gemm(epilogue_warps):
    """Sanitizer target: compare all output bytes without instrumenting cuBLAS.

    The 31 live expert blocks and 8/4 feature widths leave an alternating tail
    after the warm-up on a full B200. Independent-reference coverage lives in
    test_alternating_order_is_bit_exact_to_static.
    """
    _require_b200()
    baseline = None
    for mode in ("static", "alternating"):
        workspace, launch, _ = _full_kernel_case(
            129,
            1024,
            1024,
            32,
            8,
            knobs={"load_balance_mode": mode, "epilogue_warps": epilogue_warps},
            check_reference=False,
        )
        try:
            actual = launch().clone()
            assert torch.isfinite(actual).all()
            if baseline is None:
                baseline = actual
            for _ in range(3):
                assert torch.equal(
                    launch().view(torch.uint8), baseline.view(torch.uint8)
                )
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ["atomic_counter", "alternating"])
def test_megamoe_apply_knobs_switches_tactics_bit_exact(load_balance_mode) -> None:
    """One workspace sized for two tactics switches A -> B -> A in place."""

    _require_b200()
    shape = (129, 256, 256, 4, 2)
    tactics = [
        {
            "mma_tiler_mnk": (256, 64, 128),
            "epilogue_warps": 4,
            "in_kernel_reduce": False,
            "group_hint": 1,
        },
        {
            "mma_tiler_mnk": (256, 128, 128),
            "epilogue_warps": 8,
            "in_kernel_reduce": True,
            "group_hint": 1,
            "load_balance_mode": load_balance_mode,
        },
    ]
    fresh = []
    for knobs in tactics:
        workspace, launch, expected = _full_kernel_case(*shape, knobs=knobs)
        try:
            fresh.append(launch().clone())
            torch.testing.assert_close(fresh[-1], expected, atol=0.015, rtol=0.05)
        finally:
            workspace.destroy()
    workspace, launch, _ = _full_kernel_case(
        *shape, knobs=tactics[0], capacity_tactics=tactics
    )
    try:
        assert all(workspace.fits(knobs) for knobs in tactics)
        for index in (0, 1, 0, 1):
            workspace.apply_knobs(tactics[index])
            actual = launch()
            assert torch.equal(actual.view(torch.uint8), fresh[index].view(torch.uint8))
    finally:
        workspace.destroy()
    # A tactic whose plan exceeds the allocation is rejected, not clipped: N64
    # needs more task-tile counters than an N128-only workspace holds.
    workspace, _, _ = _full_kernel_case(
        *shape, knobs=tactics[1], capacity_tactics=[tactics[1]]
    )
    try:
        assert not workspace.fits(tactics[0])
        with pytest.raises(ValueError, match="capacity_tactics"):
            workspace.apply_knobs(tactics[0])
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens,experts", [(0, 3), (33, 3), (2048, 32)])
@pytest.mark.parametrize(
    "epilogue_warps,tile_tokens,load_balance_mode",
    [
        (4, 64, "static"),
        (8, 192, "atomic_counter"),
        (8, 192, "alternating"),
    ],
)
def test_startup_fixes_bitwise_graph_relaunch(
    tokens, experts, epilogue_warps, tile_tokens, load_balance_mode
):
    _require_b200()
    knobs = dict(
        epilogue_warps=epilogue_warps,
        mma_tiler_mnk=(256, tile_tokens, 128),
        load_balance_mode=load_balance_mode,
    )
    workspace, launch, expected = _full_kernel_case(
        tokens, 256, 256, experts, 2, knobs=knobs
    )
    try:
        baseline = launch().clone()
        torch.testing.assert_close(baseline, expected, atol=0.015, rtol=0.05)
        workspace.apply_knobs(dict(knobs, cta_local_counts=True))
        assert torch.equal(launch().view(torch.uint8), baseline.view(torch.uint8))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = launch()
        for _ in range(12):
            graph.replay()
        assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("knobs", [None, "auto"])
def test_public_layer_general_autotuner_and_graph(monkeypatch, tmp_path, knobs):
    _require_b200()
    from flashinfer import autotune
    from flashinfer.autotuner import AutoTuner
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
        config,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        create_dummy_mxfp8_mxfp4_inputs,
        mxfp8_mxfp4_mega_moe,
    )

    candidates = [
        {"mma_tiler_mnk": (128, 64, 128), "in_kernel_reduce": False},
        {"mma_tiler_mnk": (128, 128, 128), "in_kernel_reduce": True},
    ]
    monkeypatch.setattr(config, "candidate_tactics", lambda **kwargs: candidates)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    _, fc1, fc2, source = create_dummy_mxfp8_mxfp4_inputs(
        0,
        1,
        4,
        16,
        7,
        2,
        256,
        128,
        capacity_tactics=candidates,
    )
    layer = None
    try:
        expected = mxfp8_mxfp4_mega_moe(None, fc1, fc2, source, num_tokens=7).clone()
        inputs = MoEEpTensors(
            hidden_states=source.x[:7],
            scales=source.x_sf[:7],
            topk_ids=source.topk_idx[:7],
            topk_weights=source.topk_weights[:7],
        )
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            FleetParams(num_experts=4, max_tokens_per_rank=16, token_hidden_size=256),
            MoEWeightPack(w13=fc1[0], w2=fc2[0]),
            MegaConfig(
                megakernel=config.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=128,
                    top_k=2,
                    knobs=knobs,
                ),
                transformed_weights=(fc1, fc2),
                quantize_input=False,
            ),
        )
        # The public API deliberately rejects extra workspaces for knobs="auto".
        # Exercise its default workspace while testing a larger explicit
        # workspace with the standard autotune() workflow.
        profile = layer.create_workspace(32) if knobs is None else None
        with autotune(cache=str(tmp_path / "general.json")):
            layer.warmup(inputs, workspace=profile)
        assert (tmp_path / "general.json").exists()
        assert tuner.stats.tuned_op_successful_configs["mxfp8_mxfp4_mega_moe"] == 1

        def unexpected_prepare(*args, **kwargs):
            raise AssertionError("serving/capture must reuse the selected tactic")

        monkeypatch.setattr(tuner, "_prepare_tactic", unexpected_prepare)
        actual = layer(inputs, workspace=profile)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        # A surrounding tuning context must not re-enter collective selection
        # during capture after this workspace has been warmed eagerly.
        with autotune(), torch.cuda.graph(graph):
            captured = layer(inputs, workspace=profile)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        inputs.topk_ids.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(captured) == 0
    finally:
        if layer is not None:
            layer.destroy()
        source.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mode", ["static", "atomic_counter", "alternating"])
@pytest.mark.parametrize("mma_m", [128, 256])
def test_persistent_grid_bounds_are_bit_exact(mode: str, mma_m: int) -> None:
    """Exercise single-cluster synchronization and occupancy clamping."""
    _require_b200()
    baseline = None
    for clusters in (None, 1, 64, 1000):
        workspace, launch, expected = _full_kernel_case(
            33,
            256,
            256,
            4,
            2,
            knobs={
                "load_balance_mode": mode,
                "persistent_clusters": clusters,
                "mma_tiler_mnk": (mma_m, 64, 128),
            },
        )
        try:
            actual = launch().clone()
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
            if baseline is None:
                baseline = actual
            assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = launch()
            for _ in range(3):
                graph.replay()
                assert torch.equal(result.view(torch.uint8), baseline.view(torch.uint8))
        finally:
            workspace.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mode", ["static", "atomic_counter", "alternating"])
@pytest.mark.parametrize("tokens,experts", [(0, 7), (33, 7), (2048, 33)])
def test_longest_first_is_bit_exact_with_skew_and_graph_replay(mode, tokens, experts):
    _require_b200()
    knobs = dict(
        mma_tiler_mnk=(256, 64, 128),
        load_balance_mode=mode,
        claim_order="expert",
        group_hint=5,
    )
    workspace, launch, _ = _full_kernel_case(
        tokens, 256, 256, experts, 2, knobs=knobs, check_reference=False
    )
    try:
        if tokens:
            # Put the largest expert last in pool order, leave another empty,
            # and retain a masked route. Each token still has distinct experts.
            workspace.topk_idx[:tokens, 0] = experts - 1
            workspace.topk_idx[:tokens, 1] = torch.arange(tokens, device="cuda") % (
                experts - 2
            )
            workspace.topk_idx[tokens - 1, 1] = -1
        expected = launch().clone()
        for order in ("longest_first", "expert"):
            workspace.apply_knobs(dict(knobs, claim_order=order))
            assert torch.equal(launch().view(torch.uint8), expected.view(torch.uint8))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = launch()
            for _ in range(4):
                graph.replay()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    finally:
        workspace.destroy()


@pytest.mark.arch_blackwell
def test_public_layer_owned_view_and_partial_capacity():
    _require_b200()
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        create_dummy_mxfp8_mxfp4_inputs,
        mxfp8_mxfp4_mega_moe,
    )

    _, fc1, fc2, source = create_dummy_mxfp8_mxfp4_inputs(
        0, 1, 4, 16, 7, 2, 256, 128, capacity_tactics=[{}]
    )
    layer = None
    try:
        expected = mxfp8_mxfp4_mega_moe(None, fc1, fc2, source, num_tokens=7).clone()
        inputs = MoEEpTensors(
            hidden_states=source.x[:7],
            scales=source.x_sf[:7],
            topk_ids=source.topk_idx[:7],
            topk_weights=source.topk_weights[:7],
        )
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            FleetParams(num_experts=4, max_tokens_per_rank=16, token_hidden_size=256),
            MoEWeightPack(w13=fc1[0], w2=fc2[0]),
            MegaConfig(
                megakernel=Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=128, top_k=2, knobs={}
                ),
                transformed_weights=(fc1, fc2),
                quantize_input=False,
            ),
        )
        profile = layer.create_workspace(32)
        layer.warmup(inputs, workspace=profile)
        owned = layer(inputs, workspace=profile)
        view = layer(inputs, workspace=profile, return_workspace_view=True)
        assert owned.shape == (7, 256)
        assert owned.data_ptr() != view.data_ptr()
        torch.testing.assert_close(owned, expected, atol=0, rtol=0)
        torch.testing.assert_close(view, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(inputs, workspace=profile)
        inputs.topk_ids.fill_(-1)
        for _ in range(2):
            graph.replay()
            assert torch.count_nonzero(captured) == 0
        # Borrowed storage was reused; the earlier owned result survives.
        assert torch.count_nonzero(view) == 0
        torch.testing.assert_close(owned, expected, atol=0, rtol=0)
    finally:
        if layer is not None:
            layer.destroy()
        source.destroy()
