"""Offline MX-FP6 (W6A6) weight quantization + save/load for fused MoE.

This is the *weight-only* offline step of the W6A6 scheme. Weights are static, so
they are quantized once here into the exact packed MX-FP6 + swizzled UE8M0
block-scale layout that the fused FP6 MoE integration consumes.
Activations are **not** handled here: they are data-dependent and are quantized to
FP6 on the fly inside the kernel (the producer warps), so the MoE matmul still runs
fully in 6-bit. Only a scalar activation global-scale is needed at launch.

Layout produced (matches the passing numeric test ``test_moe_fp6_numeric_vs_reference``):

* ``w1_fp6``        : ``(E, 2*N, 3*K//4)`` uint8 — packed FC1 (gate+up) codes
* ``w1_blockscale`` : swizzled UE8M0 bytes, ``(E, pad128(2*N), pad4(K//32))`` uint8
* ``w2_fp6``        : ``(E, K, 3*N//4)`` uint8 — packed FC2 (down) codes
* ``w2_blockscale`` : swizzled UE8M0 bytes, ``(E, pad128(K), pad4(N//32))`` uint8
* ``*_alphas``      : ``(E,)`` f32 dequant scales (1.0 under unit global scale)
* ``a{1,2}_gscale`` : ``(1,)`` f32 activation global scales (1.0 default)

The packed codes come from the GPU quantizer (``compile_bf16_to_fp6_tma``) — the
kernel's own quantizer, bit-identical to the torch reference — and the block scales
are computed with the same per-32-block UE8M0 + cutlass swizzle the kernel reads.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    _ue8m0_scale_from_block_max,
)
from b12x._lib.utils import mxfp6_packed_k_bytes

_TILE = 128


def _weight_fmt_for_source(source_format: str) -> str:
    """FP6 sub-format for the *weight* operand (mirrors ``_mxfp6_fmt_pair``)."""
    sf = source_format.lower()
    if sf in ("mxfp6_e3m2", "e3m2"):
        return "e3m2"
    if sf in ("mxfp6_e2m3", "e2m3"):
        return "e2m3"
    # mxfp6_default / mxfp6_mixed: E3M2 activations, E2M3 weights.
    return "e2m3"


@dataclass
class FP6MoEWeights:
    """Quantized MX-FP6 MoE weights ready for the fused FP6 MoE path."""

    w1_fp6: torch.Tensor
    w1_blockscale: torch.Tensor
    w1_alphas: torch.Tensor
    w2_fp6: torch.Tensor
    w2_blockscale: torch.Tensor
    w2_alphas: torch.Tensor
    a1_gscale: torch.Tensor
    a2_gscale: torch.Tensor
    num_experts: int
    k: int
    n: int
    weight_fmt: str
    source_format: str
    activation: str

    def to(self, device: torch.device | str) -> "FP6MoEWeights":
        dev = torch.device(device)
        return FP6MoEWeights(
            w1_fp6=self.w1_fp6.to(dev),
            w1_blockscale=self.w1_blockscale.to(dev),
            w1_alphas=self.w1_alphas.to(dev),
            w2_fp6=self.w2_fp6.to(dev),
            w2_blockscale=self.w2_blockscale.to(dev),
            w2_alphas=self.w2_alphas.to(dev),
            a1_gscale=self.a1_gscale.to(dev),
            a2_gscale=self.a2_gscale.to(dev),
            num_experts=self.num_experts,
            k=self.k,
            n=self.n,
            weight_fmt=self.weight_fmt,
            source_format=self.source_format,
            activation=self.activation,
        )


def _swizzled_block_scales(
    values_bf16: torch.Tensor, fmt: str, *, global_scale: float = 1.0
) -> torch.Tensor:
    """Per-32-block UE8M0 scales in the cutlass weight swizzle (flat uint8 view).

    Mirrors the validated ``_swizzled_scales_for_quantized`` test helper: compute
    the UE8M0 exponent byte per 32-element block from ``block_max * global_scale``,
    then apply ``swizzle_block_scale``. The bytes already hold the UE8M0 exponent
    (e.g. 127 == 2^0), so we BITCAST (``.view``) to ``float8_e8m0fnu`` rather than
    re-encoding the integer.
    """
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    groups, rows, cols = values_bf16.shape
    if cols % SF_VEC_SIZE_FP6 != 0:
        raise ValueError(
            f"K/N must be divisible by {SF_VEC_SIZE_FP6}, got {cols}"
        )
    blocks = cols // SF_VEC_SIZE_FP6
    scales = torch.zeros((groups, rows, blocks), dtype=torch.uint8, device=values_bf16.device)
    for g in range(groups):
        x = values_bf16[g].float()
        sliced = x.view(rows, blocks, SF_VEC_SIZE_FP6)
        block_max = sliced.abs().amax(dim=-1)
        scales[g] = _ue8m0_scale_from_block_max(block_max * global_scale, fmt_max)
    return swizzle_block_scale(scales.view(torch.float8_e8m0fnu)).view(torch.uint8)


def _quantize_codes(
    values_bf16: torch.Tensor, fmt: str, *, use_gpu: bool
) -> torch.Tensor:
    """Pack ``(E, M, K)`` bf16 to ``(E, M, 3K//4)`` uint8 MX-FP6 codes (unit gscale)."""
    experts, m, k = values_bf16.shape
    packed_k = mxfp6_packed_k_bytes(k)
    if use_gpu:
        if m % _TILE != 0 or k % _TILE != 0:
            raise ValueError(
                f"GPU quantizer requires M%{_TILE}==0 and K%{_TILE}==0, got M={m} K={k}. "
                "Pass use_gpu=False to use the (slow) torch reference path."
            )
        # Lazy import to avoid an import cycle with the package __init__.
        from b12x.quantization.mxfp6 import (
            allocate_bf16_to_fp6_tma_outputs,
            compile_bf16_to_fp6_tma,
        )

        launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt)
        gs_one = torch.ones(1, dtype=torch.float32, device=values_bf16.device)
        packed = torch.empty(experts, m, packed_k, dtype=torch.uint8, device=values_bf16.device)
        for e in range(experts):
            out = allocate_bf16_to_fp6_tma_outputs(m, k, device=values_bf16.device)
            launch(values_bf16[e].contiguous(), gs_one, out.packed_a_flat, out.scale_flat)
            packed[e] = out.packed_a_storage.view(m, packed_k)
        torch.cuda.synchronize()
        return packed

    from b12x._lib.fp6 import quantize_grouped_mxfp6_torch

    row_counts = torch.full((experts,), m, dtype=torch.int32, device=values_bf16.device)
    gs = torch.ones(experts, dtype=torch.float32, device=values_bf16.device)
    packed_grp, _ = quantize_grouped_mxfp6_torch(
        values_bf16, row_counts, gs, fmt=fmt  # type: ignore[arg-type]
    )
    return packed_grp.permute(2, 0, 1).contiguous()


def quantize_moe_weights_to_fp6(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    source_format: str = "mxfp6_w6a8",
    activation: str = "silu",
    use_gpu: bool = True,
) -> FP6MoEWeights:
    """Quantize BF16 MoE expert weights to MX-FP6 (W6A6 weight operand).

    Args:
        w1_bf16: FC1 weights ``(E, 2*N, K)`` (gate+up stacked along rows, in the
            same row order the fused FP6 MoE kernel expects). For ReLU2
            (non-gated) this is ``(E, N, K)``.
        w2_bf16: FC2 (down) weights ``(E, K, N)``.
        source_format: selects the weight FP6 sub-format and the runtime activation
            format written to ``config.json``. Default ``mxfp6_w6a8``: E2M3 weights /
            E4M3 activations. ``mxfp6_default``: E2M3 / E3M2 (legacy W6A6).
        activation: ``"silu"`` (gated) or ``"relu2"``.
        use_gpu: use the fast GPU quantizer for packed codes (requires CUDA and
            M/K multiples of 128). ``False`` falls back to the slow torch reference.

    Returns:
        :class:`FP6MoEWeights` with everything needed to launch the fused FP6 MoE.
        Uses unit global scales (per-block UE8M0 handles dynamic range), so
        ``alphas`` and ``a*_gscale`` are all 1.0.
    """
    if w1_bf16.ndim != 3 or w2_bf16.ndim != 3:
        raise ValueError("w1_bf16 and w2_bf16 must be rank-3 (E, *, *) tensors")
    experts, w1_rows, k = w1_bf16.shape
    e2, k2, n = w2_bf16.shape
    if e2 != experts:
        raise ValueError(f"expert count mismatch: w1 E={experts}, w2 E={e2}")
    if k2 != k:
        raise ValueError(f"K mismatch: w1 K={k}, w2 K={k2}")
    is_gated = activation == "silu"
    expected_w1_rows = (2 if is_gated else 1) * n
    if w1_rows != expected_w1_rows:
        raise ValueError(
            f"w1 rows {w1_rows} != {'2*N' if is_gated else 'N'}={expected_w1_rows} "
            f"for activation={activation!r} (N inferred from w2 as {n})"
        )

    weight_fmt = _weight_fmt_for_source(source_format)
    w1_bf16 = w1_bf16.to(torch.bfloat16)
    w2_bf16 = w2_bf16.to(torch.bfloat16)

    w1_fp6 = _quantize_codes(w1_bf16, weight_fmt, use_gpu=use_gpu)
    w2_fp6 = _quantize_codes(w2_bf16, weight_fmt, use_gpu=use_gpu)
    w1_bs = _swizzled_block_scales(w1_bf16, weight_fmt)
    w2_bs = _swizzled_block_scales(w2_bf16, weight_fmt)

    device = w1_bf16.device
    ones_e = torch.ones(experts, dtype=torch.float32, device=device)
    ones_1 = torch.ones(1, dtype=torch.float32, device=device)
    return FP6MoEWeights(
        w1_fp6=w1_fp6,
        w1_blockscale=w1_bs,
        w1_alphas=ones_e.clone(),
        w2_fp6=w2_fp6,
        w2_blockscale=w2_bs,
        w2_alphas=ones_e.clone(),
        a1_gscale=ones_1.clone(),
        a2_gscale=ones_1.clone(),
        num_experts=experts,
        k=k,
        n=n,
        weight_fmt=weight_fmt,
        source_format=source_format,
        activation=activation,
    )


def save_fp6_moe_weights(weights: FP6MoEWeights, path: str) -> None:
    """Serialize :class:`FP6MoEWeights` to ``path`` as safetensors.

    Tensors are stored on CPU; the non-tensor dataclass fields travel as
    JSON-encoded safetensors metadata. Pickle (.pt) persistence is
    intentionally not supported (project policy: safetensors only).
    """
    import json

    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    # historical schema id; do not rename (existing artifacts carry it)
    metadata = {"__format__": "b12x_fp6_moe_weights_v1"}
    # fields()+getattr instead of asdict(): asdict deep-copies every field,
    # cloning the (potentially on-GPU) packed tensors before the .cpu() move.
    for f in fields(weights):
        value = getattr(weights, f.name)
        if isinstance(value, torch.Tensor):
            tensors[f.name] = value.detach().cpu().contiguous()
        else:
            metadata[f.name] = json.dumps(value)
    save_file(tensors, path, metadata=metadata)


def load_fp6_moe_weights(
    path: str, *, device: torch.device | str = "cuda"
) -> FP6MoEWeights:
    """Load weights saved by :func:`save_fp6_moe_weights` onto ``device``."""
    import json

    from safetensors.torch import safe_open

    fields: dict[str, object] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            fields[key] = f.get_tensor(key)
    for key, value in metadata.items():
        if key == "__format__":
            continue
        fields[key] = json.loads(value)
    weights = FP6MoEWeights(**fields)
    return weights.to(device)
