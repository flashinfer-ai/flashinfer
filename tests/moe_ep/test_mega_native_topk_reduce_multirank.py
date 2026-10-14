"""Four-rank integration gate for the native MegaMoE terminal reducer.

Run on four exact-SM100a (B200) or SM103a (B300) GPUs from the FlashInfer
repository root::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_mega_native_topk_reduce_multirank.py -v \
        -m "gpu_4 and arch_blackwell"

The non-deferred CuTeDSL terminal reducer is the full-layer reference.  Both
paths consume the same transformed NVFP4 weights, so this test isolates the
native terminal reducer while retaining real four-rank dispatch/combine
traffic and the complete MegaMoE data path.
"""

from __future__ import annotations

import dataclasses

import pytest

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import (
    _identity_epilogue_params,
    _launcher_ranks,
    _make_bf16_weights,
    _make_inputs,
    _megakernel_config,
    _require_cuda,
)


pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell]

_HIDDEN = 4096
# Terminal-reducer eligibility is independent of the expert intermediate
# width; keep the surrounding full-layer reference tractable.
_INTERMEDIATE = 1024
_NUM_EXPERTS = 8
_TOP_K = 6
_GATE_UP_CLAMP = 10.0
_ISSUE_SHAPES = (
    (1, 256),
    (8, 256),
    (64, 256),
    (128, 256),
    (256, 256),
    (4096, 4096),
)
_ATOL = 1e-2
_RTOL = 1e-2


def _problem(rank: int, world_size: int) -> dict:
    num_local_experts = _NUM_EXPERTS // world_size
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=_HIDDEN,
        intermediate=_INTERMEDIATE,
    )
    fc1_alpha, fc2_alpha, fc1_norm_const = _identity_epilogue_params(num_local_experts)
    return {
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "num_experts": _NUM_EXPERTS,
        "topk": _TOP_K,
        "gate_up_clamp": _GATE_UP_CLAMP,
        "fast_math": True,
        "w13": w13,
        "w2": w2,
        "fc1_alpha": fc1_alpha,
        "fc2_alpha": fc2_alpha,
        "fc1_norm_const": fc1_norm_const,
    }


def _batch(rank: int, num_tokens: int):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        num_tokens=num_tokens,
        hidden=_HIDDEN,
        num_experts=_NUM_EXPERTS,
        topk=_TOP_K,
    )
    if num_tokens:
        # Expert ownership is rank-major with two local experts per rank.
        # Force row zero to visit every EP rank, while retaining six distinct
        # routes, so even the one-token shape exercises peer traffic.
        topk_ids[0].copy_(
            torch.tensor([0, 2, 4, 6, 1, 3], dtype=torch.int64, device="cuda")
        )
    return MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def _empty_batch():
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=torch.empty((0, _HIDDEN), dtype=torch.bfloat16, device="cuda"),
        topk_ids=torch.empty((0, _TOP_K), dtype=torch.int64, device="cuda"),
        topk_weights=torch.empty((0, _TOP_K), dtype=torch.float32, device="cuda"),
    )


def _allocate_reference_workspace(
    problem: dict,
    capacity: int,
    rank: int,
    world_size: int,
    *,
    apply_topk_in_fc1: bool = True,
):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
    )

    return get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        capacity,
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        defer_topk_reduce=False,
        apply_topk_in_fc1=apply_topk_in_fc1,
        combine_dtype="bf16",
        fc1_alpha=problem["fc1_alpha"],
        fc2_alpha=problem["fc2_alpha"],
        fc1_norm_const=problem["fc1_norm_const"],
    )


def _reference_forward(t, workspace, transformed_weights, problem: dict):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl import (
        staging,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import nvfp4_mega_moe

    # Allocate the owned output before staging.  Nothing that can allocate or
    # synchronize is inserted between the staged collective round and compute.
    output = torch.empty(
        (t.num_tokens, problem["hidden"]),
        dtype=torch.bfloat16,
        device=t.hidden_states.device,
    )
    staging.stage_mega_moe_inputs(
        t.hidden_states,
        t.topk_weights,
        t.topk_ids,
        workspace.x,
        workspace.x_sf,
        workspace.topk_idx,
        workspace.topk_weights,
    )
    nvfp4_mega_moe(
        output,
        transformed_weights[0],
        transformed_weights[1],
        workspace,
        num_tokens=t.num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    return output


def _public_pointer_snapshot(workspace) -> tuple[int, ...]:
    return tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.x,
            workspace.x_sf,
            workspace.topk_idx,
            workspace.topk_weights,
            workspace.output_activation,
        )
    )


def _deferred_pointer_snapshot(workspace) -> tuple[int, int, tuple[int, ...]]:
    partials, root, descriptor = workspace._frontend.deferred_topk_reduce_workspace()
    return partials.data_ptr(), root.data_ptr(), tuple(descriptor["shape"])


def _assert_close(actual, expected) -> None:
    import torch

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize(
    "backend,apply_topk_in_fc1",
    [
        ("nvfp4", False),
        ("nvfp4", True),
        ("bf16", True),
        ("w4a16", False),
        ("w4a16", True),
    ],
)
def test_external_weighted_reduction_live_rows_and_cuda_graph(
    tmp_path, monkeypatch, backend, apply_topk_in_fc1
):
    """Caller finalization works across backends, empty rounds, and graphs."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size != 4:
        pytest.skip("requires exactly four EP ranks")

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires sm_100a or sm_103a")
    hidden, intermediate, topk, capacity = (
        (5120, 2560, 8, 64) if backend == "nvfp4" else (1024, 512, 2, 32)
    )
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=2,
        hidden=hidden,
        intermediate=intermediate,
    )
    if backend == "nvfp4":
        alpha1, alpha2, norm = _identity_epilogue_params(2)
        problem = dict(
            hidden=hidden,
            intermediate=intermediate,
            num_experts=8,
            topk=topk,
            gate_up_clamp=None,
            fast_math=True,
            fc1_alpha=alpha1,
            fc2_alpha=alpha2,
            fc1_norm_const=norm,
        )
        config = _megakernel_config(
            problem,
            epilogue_via_config=True,
            apply_topk_in_fc1=apply_topk_in_fc1,
            do_finalize=False,
            knobs=None,
        )
    else:
        config_cls = (
            Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig
            if backend == "bf16"
            else Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        config = config_cls(
            intermediate_size=intermediate,
            top_k=topk,
            gate_up_clamp=1.5,
            do_finalize=False,
            **({"apply_topk_in_fc1": apply_topk_in_fc1} if backend == "w4a16" else {}),
        )
    layer = MoEEpMegaLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(8, capacity, hidden),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=config,
            quantize_input=backend == "nvfp4",
        ),
    )
    reference_workspace = None
    reference_layer = None
    graph = None
    try:
        if backend == "nvfp4":
            from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.tuner import (
                tune_one,
            )
            from flashinfer.moe_ep.tune import _parse_args

            cache_path = [str(tmp_path / "knobs.json") if rank == 0 else None]
            dist.broadcast_object_list(cache_path, src=0)
            monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", cache_path[0])
            args = _parse_args(
                [
                    "--hidden",
                    str(hidden),
                    "--intermediate",
                    str(intermediate),
                    "--num-experts",
                    "8",
                    "--topk",
                    str(topk),
                    "--max-tokens",
                    str(capacity),
                    "--live-tokens",
                    "9",
                    "--no-do-finalize",
                    "--apply-topk-in-fc1"
                    if apply_topk_in_fc1
                    else "--no-apply-topk-in-fc1",
                    "--sweep",
                    "schedule",
                    "--base-knobs",
                    '{"flag_batch": 8}',
                    "--max-candidates",
                    "2",
                    "--warmup-iters",
                    "1",
                    "--timed-iters",
                    "1",
                ]
            )
            winner = tune_one(args, rank, world_size, capacity)
            dist.barrier()
            reference_workspace = _allocate_reference_workspace(
                problem,
                capacity,
                rank,
                world_size,
                apply_topk_in_fc1=apply_topk_in_fc1,
            )
            workspace = None
        else:
            reference_layer = MoEEpMegaLayer(
                bootstrap=bootstrap,
                fleet_params=FleetParams(8, capacity, hidden),
                weights=MoEWeightPack(w13=w13, w2=w2),
                backend=MegaConfig(
                    megakernel=dataclasses.replace(config, do_finalize=True),
                    quantize_input=False,
                    transformed_weights=layer._transformed,
                ),
            )
            workspace = layer.create_workspace(capacity)
        pointers = None

        def batch(n):
            x, weights, ids = _make_inputs(
                rank, num_tokens=n, hidden=hidden, num_experts=8, topk=topk
            )
            if backend != "nvfp4":
                # Exercise int32 and strided fused staging for W4A16.
                ids = ids.to(torch.int32)
                x = x.repeat_interleave(2, dim=1)[:, ::2]
            return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)

        def forward(t):
            nonlocal pointers
            rows, weights, indices = layer.forward_unfinalized(t, workspace=workspace)
            assert rows.shape == (t.num_tokens * topk, hidden)
            assert rows.dtype == torch.bfloat16 and rows.is_contiguous()
            assert weights.shape == indices.shape == (t.num_tokens, topk)
            assert weights.dtype == torch.float32 and indices.dtype == torch.int32
            current = tuple(
                value.untyped_storage().data_ptr() for value in (rows, weights, indices)
            )
            if pointers is None:
                pointers = current
            assert current == pointers
            partials = rows[indices.long()]
            return (partials.float() * weights.unsqueeze(-1)).sum(1).bfloat16()

        def reference_forward(t):
            if reference_layer is not None:
                return reference_layer.forward(t)
            return _reference_forward(
                t, reference_workspace, layer._transformed, problem
            )

        def assert_close(actual, expected, t):
            # Share failures before teardown so every rank leaves the collective.
            error = None
            try:
                if backend == "nvfp4":
                    _assert_close(actual, expected)
                else:
                    torch.testing.assert_close(
                        workspace._backend_workspace.combine_output[: t.num_tokens],
                        reference_layer._workspace.combine_output[: t.num_tokens],
                        atol=0,
                        rtol=0,
                    )
                    # The built-in weighted reducer uses FMA; separate multiply/sum
                    # can cross a BF16 rounding boundary even with identical rows.
                    weighted = backend == "w4a16" and not apply_topk_in_fc1
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=2e-3 if weighted else 0,
                        rtol=8e-3 if weighted else 0,
                    )
            except AssertionError as exc:
                error = str(exc)
            errors = [None] * world_size
            dist.all_gather_object(errors, error)
            assert not any(errors), errors

        for n in (9 if backend == "nvfp4" else 7, 0 if rank == 0 else 5, 0, 17):
            t = batch(n)
            actual = forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            expected = reference_forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            assert_close(actual, expected, t)
        if backend == "nvfp4":
            config = layer._workspace._frontend.config
            assert {name: getattr(config, name) for name in winner} == winner
        else:
            assert layer._workspace is None  # All calls used the explicit profile.
            assert workspace._backend_workspace is not reference_layer._workspace

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer.warmup(t, workspace=workspace)
            forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = forward(t)
        for _ in range(3):
            t.hidden_states.mul_(-0.5)
            t.topk_weights.add_(0.1)
            expected = reference_forward(t)
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            assert_close(captured_output, expected, t)
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        if reference_workspace is not None:
            reference_workspace.destroy()
        if reference_layer is not None:
            reference_layer.destroy()
        layer.destroy()


def test_native_reducer_reusable_workspaces_four_rank_end_to_end():
    """Exercise the complete native-workspace contract in one EP session."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size != 4:
        pytest.skip(f"requires exactly four EP ranks, got {world_size}")

    import torch
    import torch.distributed as dist

    from flashinfer.jit.cake_megamoe_topk_reduce import (
        is_cake_megamoe_topk_reduce_module_loaded,
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.core.kernel.workspace_pool import (
        pooled_workspace_refcount,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native terminal reducer requires exact SM100a or SM103a")

    problem = _problem(rank, world_size)
    megakernel = _megakernel_config(problem, epilogue_via_config=True)
    runtime_kernel = create_mega_kernel(megakernel)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, runtime_kernel.runtime_requirements(bootstrap)
    )

    layer = None
    small = large = None
    small_raw = large_raw = None
    reference_small = reference_large = None
    graph_small = graph_large = None
    try:
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
            ),
            fleet_params=FleetParams(
                num_experts=problem["num_experts"],
                max_tokens_per_rank=256,
                token_hidden_size=problem["hidden"],
            ),
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            backend=MegaConfig(
                megakernel=megakernel,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        assert isinstance(layer, MoEEpMegaLayer)
        assert layer._preprocessing_count == 1
        del problem["w13"], problem["w2"]

        # Workspace creation is collective; every rank performs it in the
        # same small-then-large order before any forward or graph capture.
        small = layer.create_workspace(256)
        large = layer.create_workspace(4096)
        small_raw = small._backend_workspace
        large_raw = large._backend_workspace
        assert small_raw is not large_raw
        assert small.max_tokens_per_rank == 256
        assert large.max_tokens_per_rank == 4096
        assert layer._kernel._uses_native_topk_reduce(small._fleet_params)
        assert layer._kernel._uses_native_topk_reduce(large._fleet_params)
        assert small_raw._frontend.config.defer_topk_reduce
        assert large_raw._frontend.config.defer_topk_reduce
        assert pooled_workspace_refcount(small_raw) == 1
        assert pooled_workspace_refcount(large_raw) == 1
        assert layer._preprocessing_count == 1

        small_public_ptrs = _public_pointer_snapshot(small_raw)
        large_public_ptrs = _public_pointer_snapshot(large_raw)

        # These reference sessions differ only in terminal reduction: they
        # retain the vendored non-deferred CuTeDSL reducer and consume the
        # layer's already-transformed weights.
        reference_small = _allocate_reference_workspace(problem, 256, rank, world_size)
        reference_large = _allocate_reference_workspace(problem, 4096, rank, world_size)
        transformed_weights = layer._transformed
        assert transformed_weights is not None
        assert layer._preprocessing_count == 1

        batches = {
            (num_tokens, capacity): _batch(rank, num_tokens)
            for num_tokens, capacity in _ISSUE_SHAPES
        }
        references = {}
        for num_tokens, capacity in _ISSUE_SHAPES:
            handle = small if capacity == 256 else large
            raw = small_raw if capacity == 256 else large_raw
            reference_workspace = (
                reference_small if capacity == 256 else reference_large
            )
            t = batches[(num_tokens, capacity)]

            native_view = layer.forward(
                t,
                workspace=handle,
                return_workspace_view=True,
            )
            assert native_view.data_ptr() == raw.output_activation.data_ptr()
            native = native_view.clone()
            reference = _reference_forward(
                t, reference_workspace, transformed_weights, problem
            )
            torch.cuda.synchronize()
            dist.barrier()
            _assert_close(native, reference)
            references[(num_tokens, capacity)] = reference

        assert is_cake_megamoe_topk_reduce_module_loaded()
        small_deferred_ptrs = _deferred_pointer_snapshot(small_raw)
        large_deferred_ptrs = _deferred_pointer_snapshot(large_raw)
        assert small_deferred_ptrs[2] == (256, _TOP_K, _HIDDEN)
        assert large_deferred_ptrs[2] == (4096, _TOP_K, _HIDDEN)

        # Alternate profile selection after the faithful issue-shape pass.
        # The large handle deliberately runs a short live batch to prove that
        # capacity and live-token count remain separate launch parameters.
        for t, handle, expected in (
            (batches[(64, 256)], small, references[(64, 256)]),
            (batches[(64, 256)], large, references[(64, 256)]),
            (batches[(128, 256)], small, references[(128, 256)]),
        ):
            actual = layer.forward(t, workspace=handle).clone()
            torch.cuda.synchronize()
            dist.barrier()
            _assert_close(actual, expected)

        # A zero-token round must still participate in the upstream EP launch.
        # Follow it immediately by a live round on a non-default stream; the
        # reference comparison detects either an early-return desync or a
        # terminal reducer launched on the wrong stream.
        side_stream = torch.cuda.Stream()
        dist.barrier()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            zero = layer.forward(_empty_batch(), workspace=small)
            after_zero = layer.forward(batches[(8, 256)], workspace=small).clone()
        assert zero.shape == (0, _HIDDEN)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        dist.barrier()
        _assert_close(after_zero, references[(8, 256)])
        assert side_stream.cuda_stream in {
            key[3] for key in layer._kernel._thunk_states
        }

        # Capture one graph per explicit capacity profile.  Prewarm the exact
        # capture streams so no stream-bound launch thunk is created inside
        # capture.  Barriers bracket capture, while every replay is
        # synchronized in lockstep because the fused kernel has cross-rank
        # barriers.
        small_capture_stream = torch.cuda.Stream()
        large_capture_stream = torch.cuda.Stream()
        small_capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(small_capture_stream):
            layer.warmup(batches[(64, 256)], workspace=small)
        large_capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(large_capture_stream):
            layer.warmup(batches[(4096, 4096)], workspace=large)

        graph_small = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph_small, stream=small_capture_stream):
            graph_small_output = layer.forward(
                batches[(64, 256)],
                workspace=small,
                return_workspace_view=True,
            )
        dist.barrier()

        graph_large = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_large, stream=large_capture_stream):
            graph_large_output = layer.forward(
                batches[(4096, 4096)],
                workspace=large,
                return_workspace_view=True,
            )
        dist.barrier()

        assert graph_small_output.data_ptr() == small_raw.output_activation.data_ptr()
        assert graph_large_output.data_ptr() == large_raw.output_activation.data_ptr()
        for graph, output, expected in (
            (graph_small, graph_small_output, references[(64, 256)]),
            (graph_large, graph_large_output, references[(4096, 4096)]),
            (graph_small, graph_small_output, references[(64, 256)]),
        ):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            _assert_close(output, expected)

        # Captured graphs read the original tensor addresses.  Mutating those
        # tensors in place must change replay output without allocating or
        # switching workspaces.
        captured_small = batches[(64, 256)]
        captured_small.hidden_states.neg_()
        captured_small.topk_ids.add_(1).remainder_(problem["num_experts"])
        captured_small.topk_weights.mul_(0.75)
        dist.barrier()
        graph_small.replay()
        torch.cuda.synchronize()
        dist.barrier()
        graph_mutated = graph_small_output.clone()
        mutated_reference = _reference_forward(
            captured_small, reference_small, transformed_weights, problem
        )
        torch.cuda.synchronize()
        dist.barrier()
        _assert_close(graph_mutated, mutated_reference)

        assert _public_pointer_snapshot(small_raw) == small_public_ptrs
        assert _public_pointer_snapshot(large_raw) == large_public_ptrs
        assert _deferred_pointer_snapshot(small_raw) == small_deferred_ptrs
        assert _deferred_pointer_snapshot(large_raw) == large_deferred_ptrs
        assert layer._preprocessing_count == 1

        # Drop graphs before releasing their borrowed workspace addresses.
        graph_small = graph_large = None
        graph_small_output = graph_large_output = None
        torch.cuda.synchronize()
        dist.barrier()

        # Layer destruction releases explicit profiles in deterministic reverse
        # allocation order on every rank.
        reference_large.destroy()
        reference_large = None
        reference_small.destroy()
        reference_small = None
        layer.destroy()
        assert large.is_destroyed
        assert large_raw._destroyed
        assert pooled_workspace_refcount(large_raw) == 0
        assert small.is_destroyed
        assert small_raw._destroyed
        assert pooled_workspace_refcount(small_raw) == 0
        assert not layer._workspaces
        assert layer._destroyed
        large = None
        small = None
        layer = None
        dist.barrier()
    finally:
        # Keep all ranks on the same best-effort teardown path after failures;
        # every destroy operation is idempotent.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if reference_large is not None:
            reference_large.destroy()
        if reference_small is not None:
            reference_small.destroy()
        if large is not None:
            large.destroy()
        if small is not None:
            small.destroy()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
