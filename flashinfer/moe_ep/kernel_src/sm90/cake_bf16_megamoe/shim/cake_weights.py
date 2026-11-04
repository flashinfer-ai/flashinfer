"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

import torch

__all__ = [
    "GATE_UP_GROUP",
    "Sm90CakeBf16Weights",
    "Sm90CakeWcxWeights",
    "encode_sm90_cake_wcx",
    "interleave_gate_up",
    "make_sm90_cake_bf16_weights",
    "wcx_fallback_layers",
    "wcx_supported_geometry",
]

# The Cake FC1 kernel consumes gate/up rows in alternating groups of this many
# rows so one 256-column WGMMA tile holds matching gate and up columns and the
# SwiGLU epilogue never crosses tiles.  Must match the generator's
# ``GATE_UP_GROUP`` (Cake ``sm90_bf16_moe_grouped_gemm``).
GATE_UP_GROUP = 32


def interleave_gate_up(
    w13: torch.Tensor, intermediate: int, group: int = GATE_UP_GROUP
) -> torch.Tensor:
    """Reorder ``w13`` rows ``[gate(I) | up(I)]`` into ``group``-row gate/up pairs."""
    num_experts, two_i, hidden = w13.shape
    if two_i != 2 * intermediate:
        raise ValueError(
            f"w13 second dim must be 2 * intermediate = {2 * intermediate}, got {two_i}"
        )
    if intermediate % group != 0:
        raise ValueError(
            f"intermediate_size must be a multiple of {group}, got {intermediate}"
        )
    return (
        w13.view(num_experts, 2, intermediate // group, group, hidden)
        .transpose(1, 2)
        .reshape(num_experts, two_i, hidden)
        .contiguous()
    )


@dataclass(frozen=True)
class Sm90CakeWcxWeights:
    """Lossless compressed (WCX codec version 2) copies of one layer's expert weights.

    Read by the swap-AB decode GEMMs of the early-start pair (runner
    ``wcx_max_tokens`` and ``WCX_WINDOWS``): ``w13c`` is
    ``[E, 2I / 256, H / 64, stage_bytes13]`` uint8 (the encoded gate/up-interleaved
    ``w13``), ``w2c`` is ``[E, H / 256, I / 64, stage_bytes2]`` uint8.  ``slots13`` /
    ``slots2`` are the escape slots per stage of each tensor (at most
    ``WCX2_S_CAP``).  The kernels decode the operands back to the bf16 weights bit
    for bit, so a round computes the same products in the same order as the bf16
    pair.
    """

    w13c: torch.Tensor
    w2c: torch.Tensor
    stage_bytes13: int
    stage_bytes2: int
    slots13: int
    slots2: int

    def __post_init__(self) -> None:
        from .cake_wcx import wcx2_stage_bytes

        for name, tensor, sb, slots in (
            ("w13c", self.w13c, self.stage_bytes13, self.slots13),
            ("w2c", self.w2c, self.stage_bytes2, self.slots2),
        ):
            if tensor.dtype != torch.uint8 or tensor.ndim != 4:
                raise ValueError(f"{name} must be a rank-4 uint8 tensor")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
            if tensor.shape[3] != sb or wcx2_stage_bytes(slots) != sb:
                raise ValueError(
                    f"{name} stage bytes {tensor.shape[3]} do not match {slots} escape slots"
                )

    @property
    def nbytes(self) -> int:
        """Device bytes of both compressed tensors."""
        return int(self.w13c.numel() + self.w2c.numel())


@dataclass(frozen=True)
class Sm90CakeBf16Weights:
    """Kernel-ready BF16 expert weights for ``sm90_bf16_bf16_bf16_push_cake``.

    ``w13``: ``[E, 2I, H]`` bf16 with gate/up rows interleaved in
    :data:`GATE_UP_GROUP`-row groups; ``w2``: ``[E, H, I]`` bf16.  Both are
    contiguous CUDA tensors on the same device.  No scales: the kernels read
    bf16 operands directly and accumulate in fp32.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    num_local_experts: int
    hidden_size: int
    intermediate_size: int
    w13_interleaved: bool = True
    # optional WCX copies for the decode rounds of the runner's ``wcx_max_tokens``
    # (dual format: the bf16 tensors above stay and serve every other round)
    wcx: Sm90CakeWcxWeights | None = None

    def __post_init__(self) -> None:
        if self.w13.dtype != torch.bfloat16 or self.w2.dtype != torch.bfloat16:
            raise ValueError(
                f"Sm90CakeBf16Weights expects bf16 tensors, got w13={self.w13.dtype}, w2={self.w2.dtype}"
            )
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        if tuple(self.w13.shape) != (e, 2 * i, h):
            raise ValueError(
                f"w13 must be ({e}, {2 * i}, {h}), got {tuple(self.w13.shape)}"
            )
        if tuple(self.w2.shape) != (e, h, i):
            raise ValueError(f"w2 must be ({e}, {h}, {i}), got {tuple(self.w2.shape)}")
        if (
            not (self.w13.is_cuda and self.w2.is_cuda)
            or self.w13.device != self.w2.device
        ):
            raise ValueError("w13 and w2 must be CUDA tensors on one device")
        if not (self.w13.is_contiguous() and self.w2.is_contiguous()):
            raise ValueError("w13 and w2 must be contiguous")
        if not self.w13_interleaved:
            raise ValueError("the Cake FC1 kernel requires gate/up-interleaved w13")
        if self.wcx is not None:
            nst = (
                (e, 2 * i // 256, h // 64),
                (e, h // 256, i // 64),
            )
            if (
                tuple(self.wcx.w13c.shape[:3]) != nst[0]
                or tuple(self.wcx.w2c.shape[:3]) != nst[1]
            ):
                raise ValueError(
                    f"wcx stage grids {tuple(self.wcx.w13c.shape[:3])} / {tuple(self.wcx.w2c.shape[:3])} "
                    f"do not match the weights ({nst[0]} / {nst[1]})"
                )
            if self.wcx.w13c.device != self.w13.device:
                raise ValueError("wcx tensors must be on the weights' device")

    @property
    def device(self) -> torch.device:
        return self.w13.device


def make_sm90_cake_bf16_weights(
    w13: torch.Tensor, w2: torch.Tensor
) -> Sm90CakeBf16Weights:
    """Transform canonical ``[E, 2I, H]`` / ``[E, H, I]`` bf16 weights for the kernels.

    The only transformation is the gate/up row interleave of ``w13``; ``w2`` is
    used as-is (a contiguous copy is made only when needed).
    """
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("w13 and w2 must be rank-3 [E, rows, cols] tensors")
    num_experts, two_i, hidden = w13.shape
    if two_i % 2 != 0:
        raise ValueError(f"w13 second dim must be even, got {two_i}")
    intermediate = two_i // 2
    if tuple(w2.shape) != (num_experts, hidden, intermediate):
        raise ValueError(
            f"w2 must be ({num_experts}, {hidden}, {intermediate}) to match w13, got {tuple(w2.shape)}"
        )
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError(
            "w13 and w2 must be bf16 (native BF16 backend; no requantization)"
        )
    return Sm90CakeBf16Weights(
        w13=interleave_gate_up(w13, intermediate),
        w2=w2.contiguous(),
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )


_WCX_FALLBACK = {"layers": 0, "warned": False}


def wcx_fallback_layers() -> int:
    """Layers whose WCX encoding fell back to the bf16 path in this process."""
    return _WCX_FALLBACK["layers"]


def wcx_supported_geometry(hidden_size: int, intermediate_size: int) -> bool:
    """Geometries of the WCX decode units: both GEMMs pair their 256-column
    n-tiles (``2I / 256`` and ``H / 256`` even), FC2 needs >= 4 k-steps."""
    return (
        hidden_size % 512 == 0
        and intermediate_size % 256 == 0
        and intermediate_size >= 256
    )


def encode_sm90_cake_wcx(weights: Sm90CakeBf16Weights) -> Sm90CakeBf16Weights:
    """Attach WCX (codec version 2) copies of ``w13`` / ``w2`` to ``weights``.

    The bf16 tensors are kept (dual format), so the layer's device memory grows
    by the compressed bytes (about 0.77 x the bf16 expert bytes on Gaussian
    weights).  A layer falls back to the bf16 path (returned unchanged, with
    ``wcx`` None) when a stage of either tensor needs more than ``WCX2_S_CAP``
    escape slots or the geometry has no WCX unit; fallbacks are counted
    (:func:`wcx_fallback_layers`) and warned about once per process.
    """
    from .cake_wcx import WCX2_S_CAP, wcx2_encode, wcx2_stage_escape_counts

    reason = None
    slots = []
    if not wcx_supported_geometry(weights.hidden_size, weights.intermediate_size):
        reason = (
            f"geometry H={weights.hidden_size}, I={weights.intermediate_size} has no WCX unit "
            "(needs H % 512 == 0 and I % 256 == 0)"
        )
    else:
        for name, tensor in (("w13", weights.w13), ("w2", weights.w2)):
            s = 0
            for e in range(tensor.shape[0]):
                counts = wcx2_stage_escape_counts(tensor[e])
                if counts.numel():
                    s = max(s, int(counts.max()))
            slots.append(s)
            if s > WCX2_S_CAP:
                reason = f"{name} needs {s} escape slots per stage (> WCX2_S_CAP {WCX2_S_CAP})"
                break
    if reason is not None:
        _WCX_FALLBACK["layers"] += 1
        if not _WCX_FALLBACK["warned"]:
            _WCX_FALLBACK["warned"] = True
            warnings.warn(
                f"sm90_cake_bf16 WCX: a layer keeps the bf16 expert GEMMs: {reason} "
                "(further fallbacks are counted, not warned)",
                RuntimeWarning,
                stacklevel=2,
            )
        return dataclasses.replace(weights, wcx=None)
    w13c, st13 = wcx2_encode(weights.w13, slots=slots[0])
    w2c, st2 = wcx2_encode(weights.w2, slots=slots[1])
    return dataclasses.replace(
        weights,
        wcx=Sm90CakeWcxWeights(
            w13c=w13c,
            w2c=w2c,
            stage_bytes13=st13.stage_bytes,
            stage_bytes2=st2.stage_bytes,
            slots13=st13.slots,
            slots2=st2.slots,
        ),
    )
