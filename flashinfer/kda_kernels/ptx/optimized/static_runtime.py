# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Select retained programs through the shared FlashInfer PTX loader."""

from ..static_runtime import StaticPTXModule


def compiled_stream_m64_fixed_final(*, split_major=False, **_):
    return StaticPTXModule("fixed_h64") if split_major else None


def compiled_stream_m64_fixed():
    return None


def compiled_stream_m128_fixed_h96_final(**_):
    return StaticPTXModule("fixed_h96")


def compiled_stream_m128_cluster8(**_):
    return None


def compiled_stream_m128_cluster2(
    *, full_chunks=False, store_final_state=True, handoff=False, num_heads=None, **_
):
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
    if full_chunks and store_final_state:
        if num_heads in (64, 96):
            return StaticPTXModule("optimized_varlen_uniform_h64_t60")
        return StaticPTXModule("varlen_uniform_h64")
    if not full_chunks and store_final_state:
        if num_heads in (64, 96):
            return StaticPTXModule("optimized_varlen_uniform_h64_t60p")
        return StaticPTXModule("varlen_mixed_h64")
    return None
