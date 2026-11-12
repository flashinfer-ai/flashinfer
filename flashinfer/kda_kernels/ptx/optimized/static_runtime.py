# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Select retained programs through the shared FlashInfer PTX loader."""

from ..static_runtime import StaticPTXModule


def compiled_stream_m64_fixed_final(*, split_major=False, **_):
    """Select the retained split-major M64 final-state program when requested."""
    return StaticPTXModule("fixed_h64") if split_major else None


def compiled_stream_m64_fixed():
    """Indicate that no retained M64 output-only specialization is available."""
    return None


def compiled_stream_m128_fixed_h96_final(**_):
    """Select the retained fixed-H96 M128 final-state program."""
    return StaticPTXModule("fixed_h96")


def compiled_stream_m128_cluster8(**_):
    """Indicate that no retained eight-CTA-cluster specialization is available."""
    return None


def compiled_stream_m128_cluster2(
    *, full_chunks=False, store_final_state=True, handoff=False, num_heads=None, **_
):
    """Select the partial-piece uniform program or retained mixed fallback."""
    if not full_chunks and store_final_state and handoff:
        if num_heads in (64, 96):
            return StaticPTXModule("optimized_varlen_uniform_h64_t60p")
        return StaticPTXModule("varlen_mixed_h64")
    return None


def compiled_stream_m128_dual(*, num_heads=None, **_):
    """Dual-chain tail shim of the fixed split."""
    if num_heads == 64:
        return StaticPTXModule("optimized_varlen_uniform_h64_dual_dd90")
    if num_heads == 96:
        return StaticPTXModule("optimized_varlen_uniform_h64_dual_dd")
    return StaticPTXModule("varlen_uniform_h64_dual")


def compiled_stream_m128(
    *, full_chunks=False, store_final_state=True, handoff=False, num_heads=None, **_
):
    """Select the full-chunk or identity-padded persistent recurrence program."""
    if full_chunks and store_final_state:
        if num_heads in (64, 96):
            return StaticPTXModule("optimized_varlen_uniform_h64_t60")
        return StaticPTXModule("varlen_uniform_h64")
    if not full_chunks and store_final_state:
        if num_heads in (64, 96):
            return StaticPTXModule("optimized_varlen_uniform_h64_t60p")
        return StaticPTXModule("varlen_mixed_h64")
    return None
