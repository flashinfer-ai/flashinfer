"""Bitwise conformance for the SM100/SM103/SM107 BF16 MegaMoE modes.

The single-rank tests run with plain pytest. Multi-rank coverage is opt-in:
    torchrun --nproc_per_node=2 -m pytest tests/moe_ep/test_bf16_megamoe_batch_invariant.py -m gpu_2
Use --full to exercise all three token-back carriers and additional capacities.
"""

from __future__ import annotations

from contextlib import contextmanager
import os

import pytest

_TOP_K = 3
pytestmark = pytest.mark.gpu


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("BF16 MegaMoE batch invariance requires SM100, SM103 or SM107")
    if (major, minor) == (10, 7):
        from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported

        if not is_cute_dsl_arch_supported(10, 7, native_only=True):
            pytest.skip("installed CuTe DSL cannot natively target SM107")


def _assert_bits(actual, expected, context):
    import torch
    import torch.distributed as dist

    # Compare representations, including the sign of zero.
    a, b = actual.view(torch.int16), expected.view(torch.int16)
    mismatches = (a != b).sum()
    if dist.is_initialized() and dist.get_world_size() > 1:
        # All ranks fail together before the next fused collective launch.
        dist.all_reduce(mismatches, op=dist.ReduceOp.SUM)
    assert mismatches.item() == 0, (
        f"{context}: {mismatches.item()} output elements changed bits across ranks"
    )


def _problem(rank, world_size, *, hidden=256, intermediate=256):
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    topk = _TOP_K
    generator = torch.Generator(device="cuda").manual_seed(71 + rank)
    x = torch.randn(
        257, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights = torch.rand(257, topk, device="cuda", generator=generator)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()
    weights[19].zero_()
    weights[20, 1] = 0
    # Three terms expose changes in reduction association. Every token reaches
    # the same three experts at world_size=1, crossing the 128/256 expert-row
    # boundaries and leaving one expert empty. With EP, two routes stay local
    # and one is remote.
    experts = (
        torch.tensor(
            [
                4 * rank,
                4 * rank + 1,
                (
                    4 * rank + 2
                    if world_size == 1
                    else 4 * (rank + 1) % (4 * world_size)
                ),
            ],
            device="cuda",
            dtype=torch.int64,
        )
        .expand(257, topk)
        .contiguous()
    )
    w13 = (
        torch.randn(
            4,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            4,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    raw_weights = MoEWeightPack(w13=w13, w2=w2)
    transformed = preprocess_mega_weights(
        raw_weights,
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    return x, weights, experts, transformed, raw_weights


@contextmanager
def _session(problem, rank, world_size, capacity, knobs, weight_before_fc2):
    import torch

    from flashinfer.batch_invariant import fingerprint

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        bf16_mega_moe,
        get_symm_buffer_for_bf16_mega_moe,
    )

    x, weights, experts, transformed, raw_weights = problem
    hidden, intermediate = x.shape[1], raw_weights.w2.shape[-1]
    buffer = get_symm_buffer_for_bf16_mega_moe(
        4 * world_size,
        capacity,
        weights.shape[1],
        hidden,
        intermediate,
        rank,
        world_size,
        batch_invariant=True,
        apply_routing_weights_before_fc2=weight_before_fc2,
        knobs=knobs,
    )
    try:
        assert buffer._frontend.batch_invariant_fingerprint == fingerprint(
            "megamoe_bf16",
            hidden=hidden,
            intermediate=intermediate,
            top_k=weights.shape[1],
            num_experts=4 * world_size,
            apply_routing_weights_before_fc2=weight_before_fc2,
        )

        def stage(indices, *, masked=False):
            selected_weights, selected_experts = weights[indices], experts[indices]
            if masked:
                # Mask an entire token and one slot of a second token after a
                # valid run has populated their combine workspace cells.
                all_masked = indices == 19
                one_masked = indices == 20
                selected_experts[all_masked] = -1
                selected_weights[all_masked] = float("nan")
                selected_experts[one_masked, 1] = -1
                selected_weights[one_masked, 1] = float("nan")
            stage_mega_moe_inputs(
                x[indices],
                selected_weights,
                selected_experts,
                buffer.x,
                buffer.topk_idx,
                buffer.topk_weights,
            )

        def launch(output):
            bf16_mega_moe(output, *transformed, buffer, num_tokens=output.shape[0])

        yield stage, launch
    finally:
        torch.cuda.synchronize()
        buffer.destroy()


def _run_matrix(rank, world_size, weight_before_fc2, full):
    import torch

    problem = _problem(rank, world_size)
    baseline = {"load_balance_mode": "static", "token_back_mode": "epi_warps"}
    # One common anchor covers every token, every capacity and every plan.
    # No plan is allowed to establish its own numerical baseline.
    with _session(problem, rank, world_size, 257, baseline, weight_before_fc2) as (
        stage,
        launch,
    ):
        stage(torch.arange(257, device="cuda"))
        anchor = torch.empty_like(problem[0])
        launch(anchor)
        torch.cuda.synchronize()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        _assert_bits(anchor[19], torch.zeros_like(anchor[19]), "zero routing weights")

    plans = [
        baseline,
        {
            "load_balance_mode": "atomic_counter",
            "token_back_mode": "reuse_dispatch_warps",
        },
    ]
    if full:
        plans = [
            {"load_balance_mode": schedule, "token_back_mode": carrier}
            for schedule in ("static", "atomic_counter")
            for carrier in ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
        ]
    capacities = (1, 32, 128, 129, 256, 257) if full else (129, 257)
    sizes = (0, 1, 31, 32, 127, 128, 129, 257)
    for knobs in plans:
        for capacity in capacities:
            with _session(
                problem, rank, world_size, capacity, knobs, weight_before_fc2
            ) as (stage, launch):
                for n in sizes:
                    if n > capacity:
                        continue
                    # A changed cohort, repeated tokens and a token at another
                    # row must all reproduce the same anchor's representations.
                    indices = (torch.arange(n, device="cuda") * 37) % 257
                    for order in (indices, indices.flip(0), indices.remainder(17)):
                        stage(order)
                        expected = anchor[order]
                        output = torch.empty_like(expected)
                        for _ in range(3):
                            launch(output)
                            _assert_bits(
                                output,
                                expected,
                                f"rank={rank}, {knobs}, capacity={capacity}, T={n}",
                            )
                if capacity >= 4:
                    order = torch.tensor([19, 20, 1, 53], device="cuda")
                    prime = torch.tensor([1, 53, 19, 20], device="cuda")
                    output = torch.empty_like(anchor[order])
                    for masked in (False, True, False):
                        if masked:
                            # Fill the soon-to-be-invalid cells with nonzero
                            # outputs, so an omitted mask cannot pass by luck.
                            stage(prime)
                            launch(output)
                            _assert_bits(output, anchor[prime], "prime stale cells")
                        stage(order, masked=masked)
                        launch(output)
                        _assert_bits(
                            output,
                            anchor[order],
                            f"valid/masked/valid, {knobs}, capacity={capacity}",
                        )
                    stage(order[:0])
                    empty = torch.empty_like(anchor[:0])
                    launch(empty)
                    _assert_bits(empty, anchor[:0], "empty batch after a live batch")
                    stage(order)
                    launch(output)
                    _assert_bits(
                        output, anchor[order], "live batch after an empty batch"
                    )
                    if full and world_size > 1:
                        uneven = torch.arange(0 if rank == 0 else 1, device="cuda")
                        stage(uneven)
                        output = torch.empty_like(anchor[uneven])
                        launch(output)
                        _assert_bits(output, anchor[uneven], "one empty EP rank")
                # Capture after warm-up, then change the staged token values.
                # The same captured plan must reproduce the common anchor too.
                if capacity == 257:
                    order = torch.arange(129, device="cuda")
                    stage(order)
                    output = torch.empty_like(anchor[:129])
                    launch(output)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch(output)
                    for order in (order, order.flip(0), (order * 41) % 257):
                        for masked in (False, True, False):
                            if masked:
                                prime = (order + 73) % 257
                                stage(prime)
                                graph.replay()
                                _assert_bits(
                                    output, anchor[prime], "graph prime stale cells"
                                )
                            stage(order, masked=masked)
                            for _ in range(3):
                                graph.replay()
                                _assert_bits(output, anchor[order], f"graph, {knobs}")
                    del graph


def _run_tile_matrix(
    rank, world_size, weight_before_fc2, full, shape, *, geometries=None
):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.batch_invariant import (
        bf16_batch_invariant_smem_capacity,
        bf16_plan_smem_bytes,
    )
    from flashinfer.moe_ep.batch_invariant.spec import (
        ALLOWED_PLAN_CONFIGS,
        plan_config_compatible,
    )

    hidden, intermediate = shape
    problem = _problem(rank, world_size, hidden=hidden, intermediate=intermediate)
    # Every geometry, batch and scheduling plan shares the original K64 anchor.
    with _session(problem, rank, world_size, 257, {}, weight_before_fc2) as (
        stage,
        launch,
    ):
        stage(torch.arange(257, device="cuda"))
        anchor = torch.empty_like(problem[0])
        launch(anchor)
        torch.cuda.synchronize()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0

    schedules = (
        [
            (s, c)
            for s in ("static", "atomic_counter")
            for c in ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
        ]
        if full
        else [("static", "epi_warps"), ("atomic_counter", "reuse_dispatch_warps")]
    )
    capacities = (32, 129, 257) if full else (257,)
    if geometries is None:
        geometries = ALLOWED_PLAN_CONFIGS
    for geometry in geometries:
        if not plan_config_compatible(hidden, intermediate, geometry):
            continue
        for schedule, carrier in schedules:
            if (
                bf16_plan_smem_bytes(hidden, 4 * world_size, geometry, carrier)
                > bf16_batch_invariant_smem_capacity()
            ):
                continue
            knobs = dict(
                mma_tiler_mnk=geometry.mma_tiler_mnk,
                cluster_shape_mnk=geometry.cluster_shape_mnk,
                use_2cta_instrs=geometry.use_2cta_instrs,
                load_balance_mode=schedule,
                token_back_mode=carrier,
            )
            for capacity in capacities:
                with _session(
                    problem, rank, world_size, capacity, knobs, weight_before_fc2
                ) as (stage, launch):
                    for n in (0, 1, 31, 32, 127, 128, 129, 256, 257):
                        if n > capacity:
                            continue
                        indices = (torch.arange(n, device="cuda") * 37) % 257
                        for order in (indices, indices.flip(0), indices.remainder(17)):
                            stage(order)
                            output = torch.empty_like(anchor[order])
                            for _ in range(2):
                                launch(output)
                                _assert_bits(
                                    output, anchor[order], f"{shape}, {knobs}, T={n}"
                                )
                    if capacity < 4:
                        continue
                    order = torch.tensor([19, 20, 1, 53], device="cuda")
                    prime = torch.tensor([1, 53, 19, 20], device="cuda")
                    output = torch.empty_like(anchor[order])
                    for masked in (False, True, False):
                        stage(prime)
                        launch(output)
                        stage(order, masked=masked)
                        launch(output)
                        _assert_bits(
                            output, anchor[order], f"tile masked routes, {knobs}"
                        )
                    if world_size > 1:
                        for n in (
                            0 if rank == 0 else 1,
                            1 if rank == 0 else min(129, capacity),
                        ):
                            uneven = torch.arange(n, device="cuda")
                            stage(uneven)
                            output = torch.empty_like(anchor[uneven])
                            launch(output)
                            _assert_bits(
                                output, anchor[uneven], f"uneven EP tiles, {knobs}"
                            )
                        output = torch.empty_like(anchor[order])
                        stage(order)
                        launch(output)
                    # Test publication, graph replay and a change of cohort for
                    # each tile, including the single-CTA communication path.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch(output)
                    for masked in (False, True, False):
                        stage(prime)
                        graph.replay()
                        stage(order, masked=masked)
                        graph.replay()
                        _assert_bits(output, anchor[order], f"tile CUDA graph, {knobs}")
                    del graph


@pytest.mark.parametrize("weight_before_fc2", (False, True))
@pytest.mark.parametrize(
    "shape", ((96, 128), (160, 192), (224, 256), (32, 64), (2016, 2048))
)
def test_bf16_batch_invariant_across_tiles(
    shape, weight_before_fc2, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank tile conformance test")
    _require_cuda()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_tile_matrix(0, 1, weight_before_fc2, request.config.getoption("--full"), shape)


@pytest.mark.parametrize("weight_before_fc2", (False, True))
def test_bf16_batch_invariant_across_batches_and_plans(
    weight_before_fc2, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test; run the EP conformance test under torchrun")
    _require_cuda()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_matrix(0, 1, weight_before_fc2, request.config.getoption("--full"))


@pytest.mark.parametrize("weight_before_fc2", (False, True))
@pytest.mark.parametrize(
    "geometry,schedule,carrier",
    (
        pytest.param(
            (128, 128, 64), "atomic_counter", "epi_warps", id="one_cta_atomic_epi"
        ),
        pytest.param(
            (128, 128, 256),
            "atomic_counter",
            "standalone_warps",
            id="one_cta_atomic_standalone",
        ),
        pytest.param(
            (128, 256, 128),
            "atomic_counter",
            "reuse_dispatch_warps",
            id="one_cta_atomic_reuse",
        ),
        pytest.param(
            (128, 128, 64), "static", "epi_warps", id="one_cta_static_control"
        ),
        pytest.param(
            (256, 128, 64),
            "atomic_counter",
            "standalone_warps",
            id="two_cta_atomic_control",
        ),
    ),
)
def test_bf16_batch_invariant_scheduler_sanitizer(
    geometry, schedule, carrier, weight_before_fc2, monkeypatch
):
    """Exercise atomic claim publication under device memory checking.

    Run this target with ``compute-sanitizer --tool memcheck --error-exitcode 1
    python -m pytest``. The former singleton-cluster st.async can pass ordinary
    bitwise checks while memcheck reports an invalid shared-memory write.
    Include first and subsequent claims, all carriers, and both control paths.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank sanitizer regression")
    _require_cuda()
    import torch

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    problem = _problem(0, 1, hidden=224, intermediate=256)
    baseline = {"load_balance_mode": "static", "token_back_mode": "epi_warps"}
    with _session(problem, 0, 1, 257, baseline, weight_before_fc2) as (stage, launch):
        stage(torch.arange(257, device="cuda"))
        anchor = torch.empty_like(problem[0])
        launch(anchor)
        torch.cuda.synchronize()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0

    paired = geometry[0] == 256
    knobs = dict(
        mma_tiler_mnk=geometry,
        cluster_shape_mnk=(2 if paired else 1, 1, 1),
        use_2cta_instrs=paired,
        load_balance_mode=schedule,
        token_back_mode=carrier,
    )
    small = torch.tensor([19, 20, 1, 53], device="cuda")
    with _session(problem, 0, 1, 257, knobs, weight_before_fc2) as (stage, launch):
        # Active CTAs claim another ID after publishing their first tile.
        # Larger cohorts also cross the M128 and M256 token-tile boundaries.
        for order in (
            small,
            torch.arange(129, device="cuda"),
            torch.arange(257, device="cuda"),
        ):
            stage(order)
            output = torch.empty_like(anchor[order])
            for _ in range(2):
                launch(output)
                _assert_bits(output, anchor[order], f"scheduler sanitizer, {knobs}")
        stage(small)
        output = torch.empty_like(anchor[small])
        launch(output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(output)
        prime = small.roll(2)
        for masked in (False, True, False):
            stage(prime)
            graph.replay()
            _assert_bits(output, anchor[prime], "scheduler graph prime")
            stage(small, masked=masked)
            graph.replay()
            _assert_bits(output, anchor[small], "scheduler graph masked routes")
        del graph


def test_bf16_batch_invariant_public_layer_autotune_and_reload(monkeypatch):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank public-layer autotune smoke")
    _require_cuda()
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.batch_invariant import fingerprint

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "0")
    problem = _problem(0, 1)
    x, weights, experts, _transformed, raw_weights = problem
    with _session(problem, 0, 1, 257, {}, True) as (stage, launch):
        stage(torch.arange(257, device="cuda"))
        anchor = torch.empty_like(x)
        launch(anchor)
        torch.cuda.synchronize()

    def tensors(order):
        return MoEEpTensors(
            hidden_states=x[order], topk_weights=weights[order], topk_ids=experts[order]
        )

    def make_layer(config):
        return MoEEpMegaLayer(
            bootstrap=BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=4, max_tokens_per_rank=257, token_hidden_size=256
            ),
            weights=raw_weights,
            backend=MegaConfig(megakernel=config),
        )

    automatic = make_layer(
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=weights.shape[1],
            batch_invariant=True,
            knobs="auto",
        )
    )
    receipt = fingerprint(
        "megamoe_bf16",
        hidden=256,
        intermediate=256,
        top_k=weights.shape[1],
        num_experts=4,
    )
    try:
        assert automatic.batch_invariant_fingerprint == receipt
        automatic.warmup(tensors(torch.arange(31, device="cuda")))
        selected = automatic.get_tuned_config()
        assert selected.batch_invariant is True
        assert selected.enable_in_kernel_fc2_reduce is False
        assert selected.knobs.get("in_kernel_fc2_reduce", False) is False
        for n in (1, 31, 129, 257):
            order = (torch.arange(n, device="cuda") * 37) % 257
            _assert_bits(
                automatic.forward(tensors(order)),
                anchor[order],
                "public autotuned layer",
            )
    finally:
        torch.cuda.synchronize()
        automatic.destroy()
    fixed = make_layer(selected)
    try:
        assert fixed.batch_invariant_fingerprint == receipt
        order = torch.arange(257, device="cuda").flip(0)
        _assert_bits(
            fixed.forward(tensors(order)), anchor[order], "fixed tuned-config reload"
        )
    finally:
        torch.cuda.synchronize()
        fixed.destroy()


@pytest.mark.gpu_2
@pytest.mark.parametrize("weight_before_fc2", (False, True))
def test_bf16_batch_invariant_expert_parallel(weight_before_fc2, monkeypatch, request):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("requires torchrun with at least two SM100/SM103/SM107 GPUs")
    from flashinfer.moe_ep import (
        BootstrapConfig,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    rank = int(os.environ["RANK"])
    bootstrap = BootstrapConfig(rank=rank, world_size=world_size)
    ensure_moe_ep_cuda_device(bootstrap)
    _require_cuda()
    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    backend = create_mega_kernel(
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=_TOP_K,
            batch_invariant=True,
            apply_routing_weights_before_fc2=weight_before_fc2,
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, backend.runtime_requirements(bootstrap)
    )
    try:
        _run_matrix(
            rank, world_size, weight_before_fc2, request.config.getoption("--full")
        )
        full = request.config.getoption("--full")
        if full:
            _run_tile_matrix(rank, world_size, weight_before_fc2, True, (224, 256))
        else:
            from flashinfer.moe_ep.batch_invariant.spec import PlanConfig

            _run_tile_matrix(
                rank,
                world_size,
                weight_before_fc2,
                False,
                (96, 128),
                geometries=(
                    PlanConfig((128, 128, 128), (1, 1, 1), False),
                    PlanConfig((256, 128, 128), (2, 1, 1), True),
                ),
            )
    finally:
        finalize_moe_ep_runtime(runtime)
