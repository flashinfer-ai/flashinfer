"""Configuration for the SM90 native BF16 (Cake-generated GEMMs) push mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper native BF16 backend.

    Precision contract: bf16 activations and weights into both expert GEMMs,
    fp32 accumulation, bf16 intermediate, bf16 dispatch payload and combine
    wire, bf16 output.  Nothing is quantized to FP8.

    ``clamp_limit`` optionally clamps the FC1 gate/up pre-activations to
    ``[-clamp_limit, clamp_limit]`` (fp32, before SwiGLU); ``None`` disables
    the clamp, which is the reference behaviour.
    ``init_timeout_s`` is applied to the EP process group only while the
    workspace is set up (JIT builds, peer handle exchange); the group's
    previous per-backend timeouts are restored afterwards when PyTorch
    exposes them (``backend.options._timeout``), otherwise the init timeout
    stays in effect for later collectives on that group.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_push_cake"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    clamp_limit: float | None = None
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
