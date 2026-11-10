# SPDX-License-Identifier: Apache-2.0
# Native layout and folding derived from sgl-project/sglang at 9cdd6624fe.
"""Native HC storage preparation for caller-owned model parameters."""

import torch

from .api import _contract, _overlap, _target_device, _tensor


def support_reason(*, hidden_size, lowrank, dtype, device, rms_eps=1e-6):
    """Return an unsupported reason, or None; no allocation or JIT loading."""
    try:
        _contract(hidden_size, lowrank, rms_eps)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if dtype not in (torch.bfloat16, torch.float16):
        return "HC sum-state requires BF16/FP16"
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "HC sum-state requires a CUDA device"
    try:
        capability = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, IndexError) as exc:
        return str(exc)
    if capability not in ((10, 0), (10, 3)):
        return "HC sum-state requires SM100 or SM103"
    from .jit import require_sources

    try:
        require_sources()
    except RuntimeError as exc:
        return str(exc)
    return None


def weight_shapes(hidden_size, lowrank):
    _contract(hidden_size, lowrank, 1e-6)
    return (4, (lowrank + 11) // 8 * 8, hidden_size), (hidden_size, 4)


def _pack_native(down, inject, norm):
    h, lowrank = norm.shape[1], down.shape[0]
    packed = down.new_zeros((4, (lowrank + 11) // 8 * 8, h))
    for branch in range(4):
        gamma = 1.0 + norm[branch].float()
        packed[branch, :lowrank].copy_(
            (down[:, branch * h : (branch + 1) * h].float() * gamma).to(down.dtype)
        )
        packed[branch, lowrank : lowrank + 4].copy_(
            (inject[:, branch * h : (branch + 1) * h].float() * gamma).to(down.dtype)
        )
    return packed, norm.T.contiguous()


@torch.no_grad()
def pack_into(down, inject, norm, *, out_down, out_norm):
    """Fold on CPU, then copy directly into existing native output storage.

    All raw inputs must be on CPU, in the destination BF16/FP16 dtype. Native
    outputs may be CPU tensors (serialization) or SM100/103 CUDA tensors.
    Nothing is rebound, no raw checkpoint tensor is staged on the GPU, and no
    second native GPU allocation is created. Existing plans/graphs must be
    discarded before modifying their weights. Call only outside capture.
    """
    for name, tensor in (
        ("down", down),
        ("inject", inject),
        ("norm", norm),
        ("out_down", out_down),
        ("out_norm", out_norm),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor")
    if norm.ndim != 2 or norm.shape[0] != 4:
        raise ValueError("norm must have shape [4,H]")
    if not isinstance(down, torch.Tensor) or down.ndim != 2:
        raise ValueError("down must have shape [L,4*H]")
    h, lowrank = norm.shape[1], down.shape[0]
    down_shape, norm_shape = weight_shapes(h, lowrank)
    dtype, device = out_down.dtype, out_down.device
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("HC native weights require BF16/FP16")
    if device.type == "cuda":
        _target_device(device, dtype)
    elif device.type != "cpu":
        raise ValueError("native output storage must be on CPU or CUDA")
    for tensor, shape, name in (
        (down, (lowrank, 4 * h), "down"),
        (inject, (4, 4 * h), "inject"),
        (norm, (4, h), "norm"),
    ):
        _tensor(tensor.detach(), shape, dtype, torch.device("cpu"), name, 1)
    for tensor, shape, name in (
        (out_down, down_shape, "out_down"),
        (out_norm, norm_shape, "out_norm"),
    ):
        _tensor(
            tensor.detach(),
            shape,
            dtype,
            device,
            name,
            32 if device.type == "cuda" else 1,
        )
    if _overlap(out_down, out_norm):
        raise ValueError("native output tensors must not overlap")
    packed, interleaved_norm = _pack_native(down, inject, norm)
    out_down.copy_(packed)
    out_norm.copy_(interleaved_norm)
