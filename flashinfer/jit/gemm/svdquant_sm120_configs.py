"""Compiled tactic configuration names for the SM120 SVDQuant JIT module."""

from typing import Final


# One TU per tactic: 4 CTA tiles x swap_ab {false,true} x {persistent, Stream-K},
# fixed 1x1x1 cluster. Order matches the KernelShapeSm120 enum / tactic ids.
SVDQUANT_SM120_CONFIGS: Final[tuple[str, ...]] = (
    "Tactic128x128x128Config",
    "Tactic128x128x128SwapConfig",
    "Tactic128x128x128SkConfig",
    "Tactic128x128x128SwapSkConfig",
    "Tactic128x128x256Config",
    "Tactic128x128x256SwapConfig",
    "Tactic128x128x256SkConfig",
    "Tactic128x128x256SwapSkConfig",
    "Tactic256x128x128Config",
    "Tactic256x128x128SwapConfig",
    "Tactic256x128x128SkConfig",
    "Tactic256x128x128SwapSkConfig",
    "Tactic128x256x128Config",
    "Tactic128x256x128SwapConfig",
    "Tactic128x256x128SkConfig",
    "Tactic128x256x128SwapSkConfig",
    "Tactic128x64x128SwapConfig",
    "Tactic128x64x128SwapSkConfig",
    "Tactic128x64x256SwapConfig",
    "Tactic128x32x128SwapConfig",
    "Tactic64x128x128Config",
    "Tactic64x128x256Config",
    "Tactic128x64x256SwapSkConfig",
    "Tactic128x64x256SwapStaticConfig",
    "Tactic128x32x128SwapStaticConfig",
    "Tactic64x128x128StaticConfig",
    "Tactic128x32x256SwapConfig",
    "Tactic128x32x256SwapStaticConfig",
    "Tactic64x32x256SwapStaticConfig",
    "Tactic256x128x128SwapStaticConfig",
    "Tactic256x64x128SwapStaticConfig",
)
