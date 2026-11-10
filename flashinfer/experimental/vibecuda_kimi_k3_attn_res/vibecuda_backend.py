"""
VibeCUDA Kimi-K3 AttnRes backend: the original unfused attention-residual
mixing (``_apply_attn_res``) on SM100 / SM103.

This backend implements the unfused contract only:

* ``delta is None`` — the prefix is never mutated,
* ``block_write_idx == -1`` — the bank is never written,
* ``output_norm_weight`` is required — the output RMSNorm is always applied,
* the two RMSNorm epsilons are identical (``eps == output_norm_eps``),
* the per-channel score weight is ``norm_weight`` alone (``qk_weight`` must
  be ``None``; the CAKE fused weight product is not part of the unfused op).

Rows are mixed in bank-then-prefix order with FP32 accumulation and the
exact epsilon placement of the reference. The kernels are VibeCUDA-generated
production payloads for NVIDIA GB200 and NVIDIA B300, integrated into
this JIT package; production route selection (persistent TMEM programs and
lean CTA-per-token programs) happens inside the compiled module.

The two architecture payloads are tuned and retained independently: the
SM100 payload serves CC 10.0 (GB200) and the SM103 payload serves CC 10.3
(B300); neither is a byte-level copy of the other and dispatch keys off the
runtime device capability only. On SM100 the nvb == 6 / T >= 2048 cell is
routed through a dedicated persistent TMEM program whose crossover over the
lean CTA-per-token ladder was verified against the pinned SGLang arm at
repo-matrix scale; per-architecture crossovers are fixed constants inside
the compiled module, not host-side heuristics.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .vibecuda_jit import (
    arch_for_device,
    load_vibecuda_kimi_k3_attn_res_module,
    registered_arches,
)

H = 7168


def vibecuda_supported_device(device: torch.device) -> bool:
    """True when ``device`` has a registered VibeCUDA AttnRes module."""
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def validate_vibecuda_kimi_k3_attn_res_inputs(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: Optional[torch.Tensor],
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> None:
    """Validate the unfused VibeCUDA AttnRes contract (loud on violation)."""
    if delta is not None:
        raise NotImplementedError(
            "the VibeCUDA Kimi-K3 AttnRes backend implements the original "
            "unfused operator only: delta must be None (no residual increment)"
        )
    if block_write_idx != -1:
        raise NotImplementedError(
            "the VibeCUDA Kimi-K3 AttnRes backend implements the original "
            "unfused operator only: block_write_idx must be -1 (no bank "
            "snapshot write)"
        )
    if output_norm_weight is None:
        raise NotImplementedError(
            "the VibeCUDA Kimi-K3 AttnRes backend always applies the output "
            "RMSNorm: output_norm_weight is required"
        )
    if qk_weight is not None:
        raise NotImplementedError(
            "the VibeCUDA Kimi-K3 AttnRes backend takes the per-channel "
            "score weight in norm_weight alone: qk_weight must be None"
        )
    if eps != output_norm_eps:
        raise NotImplementedError(
            "the VibeCUDA Kimi-K3 AttnRes backend uses one epsilon for both "
            f"RMSNorms (eps == output_norm_eps), got eps={eps} and "
            f"output_norm_eps={output_norm_eps}"
        )
    if not (0 <= num_blocks <= 8):
        raise ValueError(f"num_blocks must be in [0, 8], got {num_blocks}")
    for name, tensor in (
        ("prefix", prefix),
        ("out", out),
        ("norm_weight", norm_weight),
        ("output_norm_weight", output_norm_weight),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16, got {tensor.dtype}")
    if not blocks.is_cuda or blocks.dtype != torch.bfloat16:
        raise ValueError("blocks must be a CUDA BF16 tensor")
    if prefix.dim() != 2 or prefix.shape[1] != H:
        raise ValueError(f"prefix must be [M, {H}], got {tuple(prefix.shape)}")
    if out.shape != prefix.shape:
        raise ValueError(
            f"out must match prefix, got {tuple(out.shape)} vs {tuple(prefix.shape)}"
        )
    if blocks.dim() != 3 or blocks.shape[0] != prefix.shape[0] or blocks.shape[2] != H:
        raise ValueError(
            f"blocks must be [M, B, {H}] with the same M as prefix, got "
            f"{tuple(blocks.shape)}"
        )
    if blocks.shape[1] < num_blocks:
        raise ValueError(
            f"blocks has {blocks.shape[1]} rows, fewer than num_blocks={num_blocks}"
        )
    if norm_weight.shape != (H,) or output_norm_weight.shape != (H,):
        raise ValueError(f"norm_weight and output_norm_weight must be [{H}]")
    if blocks.stride(2) != 1 or blocks.stride(1) != H:
        raise ValueError(
            "blocks must be dense in its last two dims (row-padded leading "
            "strides are supported)"
        )
    if blocks.stride(0) < num_blocks * H:
        raise ValueError(
            f"blocks row stride {blocks.stride(0)} cannot hold "
            f"num_blocks={num_blocks} rows of {H}"
        )


@dataclass
class VibeCUDAKimiK3AttnResRunner:
    """Amortized plan of one VibeCUDA AttnRes call shape.

    ``launch`` runs the production route on the current stream without
    allocating and returns ``out``; with no arguments it reuses the tensors
    bound at ``prepare`` time (like the CAKE runner), and explicit tensor
    arguments of the same shapes/strides run the identical route.  It is
    CUDA-graph capturable (capture belongs to the caller, after ``prepare``
    has run the one-time module warmup outside capture).
    """

    module: Any
    num_blocks: int
    eps: float
    enable_pdl: bool
    route_id: int
    prefix: torch.Tensor
    blocks: torch.Tensor
    norm_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    out: torch.Tensor

    def launch(
        self,
        prefix: Optional[torch.Tensor] = None,
        blocks: Optional[torch.Tensor] = None,
        norm_weight: Optional[torch.Tensor] = None,
        output_norm_weight: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prefix = self.prefix if prefix is None else prefix
        blocks = self.blocks if blocks is None else blocks
        norm_weight = self.norm_weight if norm_weight is None else norm_weight
        output_norm_weight = (
            self.output_norm_weight
            if output_norm_weight is None
            else output_norm_weight
        )
        out = self.out if out is None else out
        if prefix is None or blocks is None or out is None:
            raise ValueError(
                "VibeCUDA AttnRes launch needs prefix, blocks and out bound "
                "either by prepare() or by explicit launch() arguments"
            )
        # Bind through local variables only: the queued kernel owns device
        # ordering via the stream, and the plan must not extend the lifetime
        # of the caller's storages beyond this call (peak-memory parity with
        # the reference runner depends on prompt release).
        self.module.run(
            out,
            prefix,
            blocks,
            norm_weight,
            output_norm_weight,
            self.num_blocks,
            blocks.stride(0),
            self.eps,
            1 if self.enable_pdl else 0,
        )
        return out


def prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: Optional[torch.Tensor],
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-6,
    output_norm_eps: float = 1e-6,
    enable_pdl: bool = True,
) -> VibeCUDAKimiK3AttnResRunner:
    """Prepare one unfused VibeCUDA AttnRes call for repeated launches.

    The one-time work (module load, route-table initialization, production
    route validation for ``(num_blocks, M)``) happens here, outside any
    timing or CUDA-graph capture region; :meth:`VibeCUDAKimiK3AttnResRunner.launch`
    is the amortized, allocation-free hot call.
    """
    validate_vibecuda_kimi_k3_attn_res_inputs(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
    )
    arch = arch_for_device(prefix.device)
    module = load_vibecuda_kimi_k3_attn_res_module(arch)
    with torch.cuda.device(prefix.device):
        module.warmup()
        route_id = module.route_id(num_blocks, prefix.shape[0])
    return VibeCUDAKimiK3AttnResRunner(
        module=module,
        num_blocks=num_blocks,
        eps=float(eps),
        enable_pdl=bool(enable_pdl),
        route_id=int(route_id),
        prefix=prefix,
        blocks=blocks,
        norm_weight=norm_weight,
        output_norm_weight=output_norm_weight,
        out=out,
    )


def kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: Optional[torch.Tensor],
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-6,
    output_norm_eps: float = 1e-6,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Run one unfused VibeCUDA AttnRes call (prepare + launch)."""
    runner = prepare_kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
    )
    return runner.launch(prefix, blocks, norm_weight, output_norm_weight, out)


__all__ = [
    "VibeCUDAKimiK3AttnResRunner",
    "kimi_k3_attn_res",
    "prepare_kimi_k3_attn_res",
    "registered_arches",
    "validate_vibecuda_kimi_k3_attn_res_inputs",
    "vibecuda_supported_device",
]
