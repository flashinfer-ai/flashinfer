"""Placeholder for a future fused MoE split kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedMoeKernelConfig:
    kernel_name: str = "fused_moe"
