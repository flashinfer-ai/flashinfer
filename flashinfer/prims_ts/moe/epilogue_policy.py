# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Host-side epilogue shortlist for Rubin non-swap-AB NVFP4 MoE tuning.

Policy zero retains both epilogues. Together the eight policies form the
Cartesian product of two FC1 and four fused-FC2 choices. Policy order is part
of the persisted tactic identity; change the MoE cache version when changing it.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MoeEpiloguePolicy:
    fc1_streaming: bool
    fc2_prefix: int | None
    fc2_read_ahead: int = 0


NVFP4_EPILOGUE_POLICIES = tuple(
    MoeEpiloguePolicy(streaming, prefix, ahead)
    for streaming in (False, True)
    for prefix, ahead in ((None, 0), (1, 1), (4, 0), (4, 1))
)


def fused_finalize_requested(runner, kwargs=None) -> bool:
    """One call-mode predicate for enumeration, mapping and execution."""
    import torch

    options = (
        dict(getattr(runner, "_cache_key_static_extras", ()))
        if kwargs is None
        else kwargs
    )
    return bool(
        options.get("use_fused_finalize", getattr(runner, "use_fused_finalize", True))
        and options.get("do_finalize", True)
        # Atomic reduction requires clearing the whole output. Preserve rows
        # without local experts through the separate finalizer when requested.
        and options.get("zero_rows_without_local_expert", True)
        and not getattr(runner, "_is_partitioned", False)
        and not torch.are_deterministic_algorithms_enabled()
    )


def supports_fused_finalize(cfg) -> bool:
    return bool(
        cfg.is_rubin
        and not cfg.is_swap_ab
        and cfg.dense_preload_epilogue
        and cfg.dense_epilogue_warps in (4, 8)
    )


def fused_finalize_overrides(tile_token: int) -> dict:
    """Complete retained profile, including operand queues and CGA."""
    from ..batched_gemm.epilogue_schedule import (
        FINALIZE_FULL_TILE_4W,
        FINALIZE_PREFIX_8W,
    )

    if tile_token == 512:
        return dict(
            FINALIZE_FULL_TILE_4W.overrides(),
            num_stages_a=3,
            num_stages_smem_sfa=3,
            num_stages_b=4,
            num_stages_smem_sfb=4,
            cluster_dim_y=4,
        )
    return dict(
        FINALIZE_PREFIX_8W.overrides(),
        num_stages_b=3,
        num_stages_smem_sfb=3,
    )


def fused_finalize_config(cfg):
    if cfg.fuse_moe_finalize:
        return cfg
    if cfg.epilogue_steps:
        # Explicit plans already own their operand/scratch profile.
        return replace(cfg, fuse_moe_finalize=1)
    return replace(cfg, **fused_finalize_overrides(cfg.tile_token))


def nvfp4_epilogue_overrides(policy_index: int) -> tuple[dict, dict]:
    """Return FC1/FC2 overrides for the Rubin non-swap-AB 512-row profile."""
    from ..batched_gemm.epilogue_schedule import (
        epilogue_steps_for_prefix,
        fc1_streaming_steps,
    )

    if not 0 <= policy_index < len(NVFP4_EPILOGUE_POLICIES):
        raise ValueError(f"Unknown NVFP4 MoE epilogue policy {policy_index}")
    policy = NVFP4_EPILOGUE_POLICIES[policy_index]
    explicit = dict(
        dense_preload_epilogue=1,
        dense_preload_store_prefix_fragments=0,
        epilogue_tmem_load_width=0,
        epilogue_tmem_load_ahead=-1,
        epilogue_smem_staging=0,
        epilogue_tmem_release=0,
        num_stages_c_smem=2,
    )
    fc1 = (
        dict(
            explicit,
            dense_epilogue_warps=8,
            epilogue_regs=144,
            epilogue_steps=fc1_streaming_steps(warps=8, batch_fragments=1),
        )
        if policy.fc1_streaming
        else {}
    )
    # Start with the same queues/CGA as retained fused FC2, not the unfused
    # JSON profile. Otherwise the four-warp C scratch can exceed SMEM capacity.
    fc2 = fused_finalize_overrides(512)
    if policy.fc2_prefix is not None:
        fc2.update(
            explicit,
            epilogue_steps=epilogue_steps_for_prefix(
                policy.fc2_prefix,
                finalize=True,
                read_ahead=policy.fc2_read_ahead,
                warps=4,
            ),
        )
    return fc1, fc2
