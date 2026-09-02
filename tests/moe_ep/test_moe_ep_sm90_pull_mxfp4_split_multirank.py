"""Single/2/4/8-rank correctness and replay stress for SM90 MXFP4 split MegaMoE.

Run this file in its own process; the vendored SM90 and SM100 CuTeDSL trees
use colliding top-level module names. Final acceptance uses four H200 GPUs::

    torchrun --standalone --nproc_per_node=4 -m pytest \
      tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_split_multirank.py -v \
      -m arch_hopper

The numerical oracle is test-owned and starts from raw packed E2M1/E8M0
weights. It covers balanced, skewed, remote-heavy, masked, zero-token,
one-token, and max-capacity routing. Every rank checks every rank's output.
"""

from __future__ import annotations

import os

import pytest


pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")


HIDDEN = 128
INTERMEDIATE = 128
LOCAL_EXPERTS = 4
TOP_K = 4
MAX_TOKENS = 64
H200_SPLIT_SMS = (80, 52)
H20_SPLIT_SMS = (48, 30)
REPLAY_ITERS = int(os.environ.get("SM90_MXFP4_SPLIT_REPLAY_ITERS", "256"))


def _launcher_ranks() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _top_k(world_size: int) -> int:
    # Preserve the original 1/2/4-rank semantics. EP8 additionally exercises
    # the customer's top-k=8 routing contract.
    return 8 if world_size == 8 else TOP_K


def _split_sm_counts() -> tuple[int, int]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_sms = int(properties.multi_processor_count)
    if total_sms == 132:
        return H200_SPLIT_SMS
    if total_sms == 78:
        return H20_SPLIT_SMS
    raise AssertionError(
        "this explicit split correctness gate supports Hopper devices with "
        f"132 or 78 SMs, got {properties.name!r} with {total_sms} SMs"
    )


def _split_config(
    *,
    graph_variant: str,
    counter_banks: int,
    top_k: int = TOP_K,
):
    from flashinfer.moe_ep import (
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    )

    k1_sms, k2_sms = _split_sm_counts()
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=top_k,
        execution_mode="split",
        load_balance_mode="static",
        token_back_mode="epi_warps",
        split_k1_mma_tiler_mnk=(256, 64, 128),
        split_k2_mma_tiler_mnk=(128, 64, 128),
        split_k1_cluster_shape_mnk=(1, 1, 1),
        split_k2_cluster_shape_mnk=(1, 1, 1),
        split_k1_num_sched_stages=2,
        split_k2_num_sched_stages=2,
        split_k1_sm_count=k1_sms,
        split_k2_sm_count=k2_sms,
        split_counter_epoch_banks=counter_banks,
        split_graph_variant=graph_variant,
    )


def _fused_config(*, top_k: int = TOP_K):
    from flashinfer.moe_ep import (
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    )

    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=top_k,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(1, 1, 1),
        load_balance_mode="static",
        token_back_mode="epi_warps",
    )


def _make_case(
    rank: int,
    world_size: int,
    case: str,
    tokens: int,
    seed: int,
    *,
    top_k: int = TOP_K,
):
    from tests.moe_ep._sm90_mxfp4_split_e2e import make_hidden
    from tests.moe_ep._sm90_mxfp4_split_reference import make_routing_case

    hidden = make_hidden(rank, tokens, seed=seed)
    ids, weights = make_routing_case(
        case=case,
        rank=rank,
        world_size=world_size,
        num_tokens=tokens,
        top_k=top_k,
        local_experts=LOCAL_EXPERTS,
    )
    return hidden, ids.cuda(), weights.cuda()


def _tensors(item):
    from flashinfer.moe_ep import MoEEpTensors

    hidden, ids, weights = item
    return MoEEpTensors(
        hidden_states=hidden,
        topk_ids=ids,
        topk_weights=weights,
    )


def _new_layer(*, bootstrap, raw, config, world_size: int):
    from flashinfer.moe_ep import FleetParams, MegaConfig, MoEEpLayer

    return MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=world_size * LOCAL_EXPERTS,
            max_tokens_per_rank=MAX_TOKENS,
            token_hidden_size=HIDDEN,
        ),
        weights=raw,
        backend=MegaConfig(
            megakernel=config,
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


def _replace_with_fused(
    *,
    layer,
    bootstrap,
    raw,
    comparisons,
    world_size: int,
    top_k: int,
):
    import torch
    import torch.distributed as dist

    from tests.moe_ep._sm90_mxfp4_split_e2e import (
        all_gather_stack,
        assert_output_matches,
    )

    layer.destroy()
    dist.barrier()
    comparisons = tuple(comparisons)
    fused = None
    try:
        fused = _new_layer(
            bootstrap=bootstrap,
            raw=raw,
            config=_fused_config(top_k=top_k),
            world_size=world_size,
        )

        def check_comparison(comparison, *, output_label: str) -> None:
            _, item, split_output_global, expected = comparison
            fused_output = fused.forward(_tensors(item)).clone()
            torch.cuda.synchronize()
            fused_output_global = all_gather_stack(fused_output)
            assert_output_matches(
                fused_output_global,
                expected,
                label=output_label,
            )
            torch.testing.assert_close(
                fused_output_global.to(torch.float32),
                split_output_global.to(torch.float32),
                atol=2.0e-2,
                rtol=2.0e-2,
            )

        masked = next(
            (
                comparison
                for comparison in comparisons
                if comparison[0].startswith("masked_")
            ),
            None,
        )
        if masked is not None:
            # A fresh zero-initialized combine plane can hide invalid-route
            # contamination. Prove masked routing works from a clean session,
            # then run the original matrix below so all-valid routes overwrite
            # every slot before the same layer transitions back to -1 routes.
            check_comparison(masked, output_label=f"fused_fresh_{masked[0]}")

        for comparison in comparisons:
            check_comparison(
                comparison,
                output_label=f"fused_{comparison[0]}",
            )
        return fused
    except BaseException:
        if fused is not None:
            fused.destroy()
        raise


def _assert_split_session(workspace, *, graph_variant: str, counter_banks: int):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        MegaMoEHopperMxfp4SplitSymmBuffer,
    )

    assert isinstance(workspace, MegaMoEHopperMxfp4SplitSymmBuffer)
    assert not hasattr(workspace, "_frontend")
    session = workspace.session
    assert session.captured and not session.poisoned and not session.destroyed
    assert session.graph_variant == graph_variant
    expected_sm_counts = _split_sm_counts()
    assert session.green_sm_counts == expected_sm_counts
    assert session.max_active_clusters == expected_sm_counts
    assert session.config.counter_epoch_banks == counter_banks
    assert session.fixed_pointer_key is not None
    assert session._pair.k1_kernel.split_role == "k1"
    assert session._pair.k2_kernel.split_role == "k2"
    return session


def _validate_handoff(session, *, rank, ids_global, route_q, route_scale):
    from tests.moe_ep._sm90_mxfp4_split_reference import (
        validate_route_indexed_handoff,
    )

    valid_counts = [
        int((ids_global == rank * LOCAL_EXPERTS + expert).sum().item())
        for expert in range(LOCAL_EXPERTS)
    ]
    result = validate_route_indexed_handoff(
        actual_payload=session.handoff_payload_view(),
        actual_scale=session.handoff_scale_view(),
        actual_metadata=session.handoff_metadata_view(),
        valid_counts=valid_counts,
        route_payload=route_q,
        route_scale=route_scale,
        global_topk_idx=ids_global,
        target_rank=rank,
        local_experts=LOCAL_EXPERTS,
        token_padding_block=session.config.handoff_token_n,
    )
    assert result.valid_rows == sum(valid_counts)


@pytest.mark.arch_hopper
def test_mxfp4_split_route_matrix_handoff_and_steady_replay_stress() -> None:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpMegaLayer,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from tests.moe_ep._sm90_mxfp4_split_e2e import (
        all_gather_stack,
        assert_output_matches,
        global_route_reference,
        output_digest,
    )
    from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_mega_multirank import (
        _gather_raw_weights,
        _make_raw_weights,
    )

    assert torch.cuda.is_available()
    rank, world_size, local_rank = _launcher_ranks()
    assert world_size in (1, 2, 4, 8)
    top_k = _top_k(world_size)
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    config = _split_config(
        graph_variant="steady_k3_reset",
        counter_banks=1,
        top_k=top_k,
    )
    kernel = create_mega_kernel(config)
    assert kernel.kernel_name() == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layer = None
    try:
        raw = _make_raw_weights(rank)
        raw_global = _gather_raw_weights(raw)
        layer = _new_layer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            raw=raw,
            config=config,
            world_size=world_size,
        )
        assert isinstance(layer, MoEEpMegaLayer)
        matrix = (
            ("balanced", 8),
            ("skewed", 8),
            ("remote_heavy", 8),
            ("masked", 8),
            ("edge", 1),
            ("edge", MAX_TOKENS),
            ("edge", 0),
        )
        workspace = None
        pointer_key = None
        stress_item = None
        fused_comparisons = []
        stress_expected = None
        for index, (case, tokens) in enumerate(matrix):
            item = _make_case(
                rank,
                world_size,
                case,
                tokens,
                seed=4100 + index * 17,
                top_k=top_k,
            )
            hidden_global = all_gather_stack(item[0])
            ids_global = all_gather_stack(item[1])
            weights_global = all_gather_stack(item[2])
            expected, route_q, route_scale = global_route_reference(
                hidden_global,
                ids_global,
                weights_global,
                raw_global,
                return_handoff=True,
            )

            actual = layer.forward(_tensors(item)).clone()
            if workspace is None:
                workspace = layer._workspace
                session = _assert_split_session(
                    workspace,
                    graph_variant="steady_k3_reset",
                    counter_banks=1,
                )
                pointer_key = session.fixed_pointer_key
            else:
                assert layer._workspace is workspace
                session = workspace.session
                assert session.fixed_pointer_key == pointer_key
            torch.cuda.synchronize()
            actual_global = all_gather_stack(actual)
            label = f"{case}_t{tokens}"
            assert_output_matches(actual_global, expected, label=label)
            handoff_error = None
            try:
                _validate_handoff(
                    session,
                    rank=rank,
                    ids_global=ids_global,
                    route_q=route_q,
                    route_scale=route_scale,
                )
            except Exception as exc:
                handoff_error = f"rank {rank}: {type(exc).__name__}: {exc}"
            handoff_errors = [None] * world_size
            dist.all_gather_object(handoff_errors, handoff_error)
            failures = [error for error in handoff_errors if error is not None]
            if failures:
                raise AssertionError(
                    f"{label} handoff validation failed on one or more ranks:\n"
                    + "\n".join(failures)
                )
            fused_comparisons.append((label, item, actual_global, expected))
            if index == 0:
                stress_item = item
                stress_expected = actual.clone()

        assert workspace is not None and stress_item is not None
        assert stress_expected is not None and pointer_key is not None
        baseline_digest = output_digest(stress_expected)
        result = None
        for _ in range(REPLAY_ITERS):
            result = layer.forward(_tensors(stress_item))
        assert result is not None
        result = result.clone()
        torch.cuda.synchronize()
        assert output_digest(result) == baseline_digest
        assert workspace.session.fixed_pointer_key == pointer_key
        assert not workspace.session.poisoned
        layer = _replace_with_fused(
            layer=layer,
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            raw=raw,
            comparisons=fused_comparisons,
            world_size=world_size,
            top_k=top_k,
        )
        dist.barrier()
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)


@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "graph_variant,counter_banks",
    [("cold_k0", 1), ("steady_k3_reset", 2)],
)
def test_mxfp4_split_cold_and_dual_bank_replay_lifecycle(
    graph_variant: str,
    counter_banks: int,
) -> None:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from tests.moe_ep._sm90_mxfp4_split_e2e import (
        all_gather_stack,
        assert_output_matches,
        global_route_reference,
        output_digest,
    )
    from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_mega_multirank import (
        _gather_raw_weights,
        _make_raw_weights,
    )

    rank, world_size, local_rank = _launcher_ranks()
    assert world_size in (1, 2, 4, 8)
    top_k = _top_k(world_size)
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    config = _split_config(
        graph_variant=graph_variant,
        counter_banks=counter_banks,
        top_k=top_k,
    )
    kernel = create_mega_kernel(config)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layer = None
    try:
        raw = _make_raw_weights(rank)
        raw_global = _gather_raw_weights(raw)
        layer = _new_layer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            raw=raw,
            config=config,
            world_size=world_size,
        )
        item = _make_case(
            rank,
            world_size,
            "balanced",
            16,
            seed=7301,
            top_k=top_k,
        )
        hidden_global = all_gather_stack(item[0])
        ids_global = all_gather_stack(item[1])
        weights_global = all_gather_stack(item[2])
        expected = global_route_reference(
            hidden_global, ids_global, weights_global, raw_global
        )
        baseline = layer.forward(_tensors(item)).clone()
        baseline_global = all_gather_stack(baseline)
        assert_output_matches(
            baseline_global,
            expected,
            label=f"{graph_variant}_banks{counter_banks}",
        )
        workspace = layer._workspace
        session = _assert_split_session(
            workspace,
            graph_variant=graph_variant,
            counter_banks=counter_banks,
        )
        pointer_key = session.fixed_pointer_key
        torch.cuda.synchronize()
        digest = output_digest(baseline)

        result = None
        for _ in range(REPLAY_ITERS):
            result = layer.forward(_tensors(item))
        assert result is not None
        result = result.clone()
        torch.cuda.synchronize()
        assert output_digest(result) == digest
        assert session.fixed_pointer_key == pointer_key
        assert not session.poisoned
        layer = _replace_with_fused(
            layer=layer,
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            raw=raw,
            comparisons=(
                (
                    f"{graph_variant}_banks{counter_banks}",
                    item,
                    baseline_global,
                    expected,
                ),
            ),
            world_size=world_size,
            top_k=top_k,
        )
        dist.barrier()
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
