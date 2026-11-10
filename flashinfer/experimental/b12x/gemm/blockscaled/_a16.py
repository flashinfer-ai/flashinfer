"""One-shot BF16 activation GEMMs with shared NVFP4/MXFP8 weight storage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import triton

from b12x.preparation import Plan
from . import _quantize as kernels

Mode = Literal["auto", "a16", "quantized"]


@dataclass(frozen=True)
class NVFP4LinearWeight:
    """Shared E2M1 values, swizzled K/16 E4M3 scales, and weight-only scale.

    ``global_scale_kind='multiplier'`` reconstructs weights as
    ``E2M1 * E4M3 * global_scale``. ``'reciprocal'`` divides by that tensor.
    Tensors are borrowed; callers must keep them alive and unmodified during use.
    """

    values: torch.Tensor
    scale_mma: torch.Tensor
    global_scale: torch.Tensor
    global_scale_kind: str
    in_features: int
    out_features: int

    @property
    def padded_in_features(self) -> int:
        return int(self.values.shape[1]) * 2


def _check_tensor(name, tensor, device, dtype=None):
    if not isinstance(tensor, torch.Tensor) or tensor.device != device:
        raise ValueError(f"{name} must be a tensor on {device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def scale_storage(scale: torch.Tensor, n: int, k: int, group: int) -> torch.Tensor:
    """Return a zero-copy flat byte view of the F8_128x4 physical storage."""
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e8m0fnu):
        raise ValueError("block scales must have uint8/E4M3/UE8M0 storage")
    nt, kt = triton.cdiv(n, 128), triton.cdiv(k // group, 4)
    if scale.ndim == 6:
        if tuple(scale.shape) != (32, 4, nt, 4, kt, 1):
            raise ValueError("block scale MMA view has the wrong shape")
        physical = scale.permute(5, 2, 4, 0, 1, 3)
    else:
        physical = scale
    if not physical.is_contiguous() or physical.numel() != nt * kt * 512:
        raise ValueError("block scales must use contiguous F8_128x4 swizzled storage")
    return physical.view(torch.uint8).view(-1)


def pack_nvfp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    global_scale_kind: str = "multiplier",
) -> NVFP4LinearWeight:
    if weight.device.type != "cuda":
        raise ValueError("NVFP4 weights must be on CUDA")
    _check_tensor("weight", weight, weight.device, torch.uint8)
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("NVFP4 weight must have positive shape [N,K/2]")
    n, storage_k = weight.shape
    k = storage_k * 2
    if k % 16:
        raise ValueError("NVFP4 K must be divisible by 16")
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 requires E4M3 block scales")
    storage = scale_storage(scale, n, k, 16)
    _check_tensor("scale", storage, weight.device, torch.uint8)
    _check_tensor("global_scale", global_scale, weight.device, torch.float32)
    if global_scale.numel() != 1 or global_scale_kind not in (
        "multiplier",
        "reciprocal",
    ):
        raise ValueError(
            "global_scale must be scalar, with kind 'multiplier' or 'reciprocal'"
        )
    from b12x._lib.intrinsics import as_grouped_scale_view

    mma = as_grouped_scale_view(storage.view(1, -1), n, k)
    return NVFP4LinearWeight(weight, mma, global_scale, global_scale_kind, k, n)


def _stream_context(stream, device):
    if stream is None:
        return nullcontext()
    if not isinstance(stream, torch.cuda.Stream):
        from b12x._lib.utils import cuda_stream_to_int

        stream = torch.cuda.ExternalStream(cuda_stream_to_int(stream), device=device)
    if stream.device != device:
        raise ValueError("stream must be on the operand device")
    return torch.cuda.stream(stream)


def _overlap(a, b):
    if not a.numel() or not b.numel():
        return False
    return (
        a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
    )


def _validate_output(source, out, n):
    shape = (*source.shape[:-1], n)
    if out is None:
        return torch.empty(shape, device=source.device, dtype=torch.bfloat16)
    _check_tensor("out", out, source.device, torch.bfloat16)
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}")
    return out


def _config(config):
    config = (64, 64, 1) if config is None else tuple(config)
    if (
        len(config) != 3
        or config[0] not in (64, 128)
        or config[1] not in (64, 128)
        or config[2] not in (1, 2, 4, 8)
    ):
        raise ValueError(
            "A16 config must be (N tile 64/128, K tile 64/128, split-K 1/2/4/8)"
        )
    return config


def w4a16(
    source: torch.Tensor,
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    global_scale_kind: str = "multiplier",
    stream: object = None,
    plan: Plan,
) -> torch.Tensor:
    """NVFP4 W4A16 using shared packed weights and swizzled block scales.

    No weight preparation or activation quantization is performed. The global
    scale is weight-only; specify ``'reciprocal'`` for a quantizer multiplier.
    ``workspace`` is contiguous uint8 storage for optional split-K partials.
    """
    if global_scale_kind not in ("multiplier", "reciprocal"):
        raise ValueError("global_scale_kind must be 'multiplier' or 'reciprocal'")
    if block_scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 requires E4M3 block scales")
    from ._ops import linear

    return linear(
        source,
        weight,
        block_scale,
        global_scale,
        out=out,
        workspace=workspace,
        plan=plan,
        global_scale_kind=global_scale_kind,
        required_mode="a16",
        stream=stream,
    )


def w8a16(
    source: torch.Tensor,
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    stream: object = None,
    plan: Plan,
) -> torch.Tensor:
    """MXFP8 W8A16 using E4M3 weights and shared swizzled UE8M0 scales."""
    if block_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("MXFP8 requires UE8M0 block scales")
    from ._ops import linear

    return linear(
        source,
        weight,
        block_scale,
        None,
        out=out,
        workspace=workspace,
        plan=plan,
        global_scale_kind="none",
        required_mode="a16",
        stream=stream,
    )


def _weight_parts(weight):
    from ._linear import MXFP8LinearWeight
    from ._iq2_xs import IQ2XSLinearWeight

    if isinstance(weight, IQ2XSLinearWeight):
        return weight.values, weight.metadata, None, False
    if isinstance(weight, NVFP4LinearWeight):
        return weight.values, weight.scale_mma, weight.global_scale, True
    if isinstance(weight, MXFP8LinearWeight):
        return weight.weight.values, weight.weight.scale_mma, None, False
    raise TypeError("BF16 blockscaled linear requires NVFP4, MXFP8, or IQ2_XS weights")


def _layout(m, n, k, fp4, config=None):
    """Byte offsets into caller-owned scratch; all kernel pointers stay aligned."""
    value_bytes = m * (k // 2 if fp4 else k)
    scale_bytes = triton.cdiv(m, 128) * triton.cdiv(k // (16 if fp4 else 32), 4) * 512
    scale_start = triton.cdiv(value_bytes, 256) * 256
    alpha_start = triton.cdiv(scale_start + scale_bytes, 256) * 256
    partial_start = alpha_start + 256
    split = _config(config)[2]
    partial_bytes = max(
        split * m * n * 4 if split > 1 else 0,
        2 * min(m, 8) * n * 4 if not fp4 else 0,
    )
    return scale_start, alpha_start, partial_start, partial_start + partial_bytes
