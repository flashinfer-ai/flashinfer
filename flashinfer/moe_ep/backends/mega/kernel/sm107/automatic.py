"""Automatic tuning shared by the Rubin MegaMoE backend adapters."""

from typing import Any


def tuning_context():
    from flashinfer.autotuner import AutoTuner
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE,
        validate_context,
    )

    context = AutoTuner.get().protocol_context(NAMESPACE)
    validate_context(context)
    return context


def needs_selection(workspace, context):
    if context.tuning_enabled:
        return getattr(workspace, "_autotune_tuned_identity", None) != tuning_identity(
            context
        )
    if getattr(workspace, "_autotune_cache_identity", None) != context.cache_identity:
        return True
    return getattr(workspace, "_autotune_winner", None) is None and getattr(
        workspace, "_autotune_lookup_scope", None
    ) != (context.scope, context.cache_identity)


def tuning_identity(context):
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_measurement import (
        measurement_settings,
    )

    return context.cache_identity, measurement_settings(context)


def needs_replacement(backend, workspace):
    import torch

    if backend._kernel_config.knobs != "auto":
        return False
    context = tuning_context()
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        if (
            getattr(workspace, "_autotune_cache_identity", None)
            != context.cache_identity
        ):
            raise RuntimeError(
                "Warm up Rubin automatic selection after changing the cache "
                "or measurement policy, before CUDA graph capture."
            )
        return False
    if not getattr(workspace, "_autotune_graph_captured", False):
        return False
    replace = needs_selection(workspace, context) and (
        context.tuning_enabled
        or getattr(workspace, "_autotune_cache_identity", None)
        != context.cache_identity
    )
    if backend.ep_world_size > 1:
        import torch.distributed as dist

        decisions = [None] * backend.ep_world_size
        dist.all_gather_object(
            decisions, (context.tuning_enabled, replace), group=backend.ep_comm_group
        )
        if any(enabled != context.tuning_enabled for enabled, _ in decisions):
            raise ValueError("EP ranks disagree on whether MegaMoE tuning is enabled")
        replace = any(decision for _, decision in decisions)
    return replace


def maybe_tune(
    backend: Any, workspace: Any, weights: Any, output: Any, num_tokens: int
) -> None:
    if backend._kernel_config.knobs != "auto":
        return
    context = tuning_context()
    if not needs_selection(workspace, context):
        backend._autotune_winner = getattr(workspace, "_autotune_winner", None)
        return
    from .....kernel_src.sm107.next_cutedsl_megamoe import (
        KNOB_KEYS,
        autotune_sm107_block_scaled_mega_moe,
        sm107_candidates,
    )

    cfg = workspace.config
    if not hasattr(workspace, "_autotune_initial_knobs"):
        workspace._autotune_initial_knobs = {
            name: getattr(cfg, name) for name in KNOB_KEYS
        }
    candidates = sm107_candidates(
        cfg.quant_kind,
        kernel_variant=cfg.kernel_variant,
        allow_in_kernel_fc2_reduce=backend._kernel_config.in_kernel_fc2_reduce
        and cfg.combine_dtype == "bf16",
    )
    winner = autotune_sm107_block_scaled_mega_moe(
        output,
        weights[0],
        weights[1],
        workspace,
        num_tokens=num_tokens,
        candidates=candidates,
        process_group=backend.ep_comm_group if backend.ep_world_size > 1 else None,
        force_tune=False,
        install_winner=True,
        allow_nondeterministic=backend._kernel_config.in_kernel_fc2_reduce,
    )
    if winner is not None:
        workspace._autotune_winner = winner
        if context.tuning_enabled:
            workspace._autotune_tuned_identity = tuning_identity(context)
    elif getattr(workspace, "_autotune_graph_captured", False):
        # Captured explicit handles retain their prepared configuration.
        backend._autotune_winner = getattr(workspace, "_autotune_winner", None)
        return
    elif getattr(workspace, "_autotune_winner", None) is not None:
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            _prepare_collectively,
        )

        # A miss in another store must not relabel the previous store's winner.
        _prepare_collectively(
            lambda: workspace.install_tuned_knobs(
                workspace._autotune_initial_knobs, weights[0], weights[1]
            ),
            backend.ep_world_size,
            backend.ep_comm_group if backend.ep_world_size > 1 else None,
        )
        workspace._autotune_winner = None
        workspace._autotune_tuned_identity = None
    workspace._autotune_cache_identity = context.cache_identity
    backend._autotune_winner = getattr(workspace, "_autotune_winner", None)
