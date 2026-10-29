"""Automatic tuning shared by the Blackwell MegaMoE backend adapters."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import torch

    from flashinfer.autotuner import TuningContext
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.block_scaled import (
        BlockScaledSymmBuffer,
        TransformedBlockScaledWeights,
    )
    from .mxfp8_mxfp8_bf16_cutedsl.backend import Mxfp8CutedslMegaKernelBackend
    from .standard_cutedsl import _StandardBlockScaledBackend

    BlackwellBackend: TypeAlias = (
        Mxfp8CutedslMegaKernelBackend | _StandardBlockScaledBackend
    )


def tuning_context() -> TuningContext:
    from flashinfer.autotuner import AutoTuner
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE,
        validate_context,
    )

    context = AutoTuner.get().protocol_context(NAMESPACE)
    validate_context(context)
    return context


def needs_selection(workspace: BlockScaledSymmBuffer, context: TuningContext) -> bool:
    state = workspace._tuning_state
    if state.cache_identity != context.cache_identity:
        return True
    if context.tuning_enabled:
        return state.tuned_identity != tuning_identity(context, workspace)
    return state.winner is None and state.lookup_scope != (
        context.scope,
        context.cache_identity,
    )


def tuning_identity(context: TuningContext, workspace: BlockScaledSymmBuffer) -> tuple:
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.measurement import (
        measurement_settings,
    )

    return context.cache_identity, measurement_settings(context)


def coordinate_workspace_replacement(
    backend: BlackwellBackend, workspace: BlockScaledSymmBuffer
) -> bool:
    """Agree on replacement before any rank allocates new symmetric storage."""
    import torch

    if backend._config.knobs != "auto":
        return False
    context = tuning_context()
    state = workspace._tuning_state
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        if state.cache_identity != context.cache_identity:
            raise RuntimeError(
                "Warm up Blackwell automatic selection after changing the cache "
                "or measurement policy, before CUDA graph capture."
            )
        return False
    replace = (
        state.graph_captured
        and needs_selection(workspace, context)
        and (context.tuning_enabled or state.cache_identity != context.cache_identity)
    )
    if backend.ep_world_size > 1:
        import torch.distributed as dist

        # A rank-local cache invalidation can require replacement on a peer.
        # Even a locally settled workspace must participate in this decision.
        decisions = [(False, False)] * backend.ep_world_size
        dist.all_gather_object(
            decisions, (context.tuning_enabled, replace), group=backend.ep_comm_group
        )
        if any(enabled != context.tuning_enabled for enabled, _ in decisions):
            raise ValueError("EP ranks disagree on whether MegaMoE tuning is enabled")
        replace = any(decision for _, decision in decisions)
    return replace


def get_tuned_knobs(workspace: BlockScaledSymmBuffer | None) -> dict[str, Any]:
    if workspace is None or workspace._tuning_state.winner is None:
        raise RuntimeError(
            "MegaMoE has no completed automatic tuning result for this workspace; "
            "warm it up with knobs='auto' in an autotune_v2() context first"
        )
    return deepcopy(workspace._tuning_state.winner)


def prepare_auto_configuration(
    backend: BlackwellBackend,
    workspace: BlockScaledSymmBuffer,
    weights: tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights],
    output: torch.Tensor | None,
    num_tokens: int,
) -> None:
    """Select and install cached or tuned knobs, or restore the initial fallback.

    Selection is collective across EP ranks and can replace scratch and output
    storage before capture. Captured explicit workspaces retain their storage.
    """
    if backend._config.knobs != "auto":
        return
    context = tuning_context()
    select = needs_selection(workspace, context)
    import torch

    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if backend.ep_world_size > 1 and not capturing:
        import torch.distributed as dist

        # A peer may have invalidated its cache even when this rank is settled.
        decisions = [(False, False)] * backend.ep_world_size
        dist.all_gather_object(
            decisions, (context.tuning_enabled, select), group=backend.ep_comm_group
        )
        if any(enabled != context.tuning_enabled for enabled, _ in decisions):
            raise ValueError("EP ranks disagree on whether MegaMoE tuning is enabled")
        select = any(pending for _, pending in decisions)
    if not select:
        return
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.autotune import (
        autotune_sm100_block_scaled_mega_moe,
    )
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import knob_keys

    cfg = workspace.config
    state = workspace._tuning_state
    if state.initial_knobs is None:
        state.initial_knobs = {name: getattr(cfg, name) for name in knob_keys(cfg)}
    constraints = (
        backend._genphase_constraints() if cfg.kernel_variant == "genphase" else None
    )
    winner = autotune_sm100_block_scaled_mega_moe(
        output,
        weights[0],
        weights[1],
        workspace,
        num_tokens=num_tokens,
        constraints=constraints,
        process_group=backend.ep_comm_group if backend.ep_world_size > 1 else None,
        force_tune=False,
        install_winner=True,
        allow_nondeterministic=backend._config.enable_in_kernel_fc2_reduce,
    )
    if winner is None and state.graph_captured:
        # Captured explicit handles retain their prepared configuration.
        return
    if winner is None and state.winner is not None:
        from .....kernel_src.sm100.next_cutedsl_megamoe.shim.autotune import (
            _prepare_collectively,
        )

        # A miss in another store must not relabel the previous store's winner.
        _prepare_collectively(
            lambda: workspace.install_tuned_knobs(
                state.initial_knobs, weights[0], weights[1]
            ),
            backend.ep_world_size,
            backend.ep_comm_group if backend.ep_world_size > 1 else None,
        )
    state.record_selection(
        winner,
        context.cache_identity,
        (
            tuning_identity(context, workspace)
            if winner is not None and context.tuning_enabled
            else None
        ),
    )
