"""Canonical NVFP4(K16/E4M3) weights for SM90 MegaMoE.

A decoded weight is G*q*s. The kernel consumes RN_E4M3(q*s/8),
then restores 8G through its per-expert output scale. Packed weights remain
E2M1 throughout GMEM/SMEM; preprocessing changes only their physical layout.

K-permuted layouts (``kperm`` 32 or 64) reorder K inside every K32/K64 chunk
so that each lane's eight values of one A-fragment row share one K16 scale
(kperm32), and one scale load serves a K32 fragment pair (kperm64). The same
permutation is applied to the FC1 activation hidden dimension during staging
and to the W1 gate and up output rows, which puts the FC2 input in the order
of the permuted W2 K. Every K64 block keeps its element set, so per-token and
per-K64 activation scales are unchanged; kperm32 output is byte-identical to
the natural layout, kperm64 changes only FP8 accumulation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, SupportsIndex, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch

NVFP4_GROUP_SIZE = 16
NVFP4_FOLD_M = 64
NVFP4_FOLD_K = 128
NVFP4_EPILOGUE_COMPENSATION = 8.0
NVFP4_GATE_UP_INTERLEAVE = 8
NVFP4_KPERM_CHOICES = (0, 32, 64)
_WeightLeg = Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]


class TransformedMegaWeights(Tuple[_WeightLeg, _WeightLeg]):
    """Two launch-compatible weight legs with their immutable K-layout identity."""

    _nvfp4_kperm: int

    def __new__(
        cls, w13: _WeightLeg, w2: _WeightLeg, *, nvfp4_kperm: int
    ) -> "TransformedMegaWeights":
        result = tuple.__new__(cls, (w13, w2))
        object.__setattr__(result, "_nvfp4_kperm", validate_nvfp4_kperm(nvfp4_kperm))
        return result

    @property
    def nvfp4_kperm(self) -> int:
        return self._nvfp4_kperm

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Transformed NVFP4 layout metadata is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Transformed NVFP4 layout metadata is read-only")

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> Tuple[Any, Tuple[_WeightLeg, _WeightLeg, int]]:
        return _restore_transformed_mega_weights, (self[0], self[1], self.nvfp4_kperm)


def _restore_transformed_mega_weights(
    w13: _WeightLeg, w2: _WeightLeg, nvfp4_kperm: int
) -> TransformedMegaWeights:
    return TransformedMegaWeights(w13, w2, nvfp4_kperm=nvfp4_kperm)


# One natural-order K chunk viewed as four factors (sizes, strides 16/8/4/1),
# and the factor order that iterates it in permuted order: new[p] = old[pi(p)].
#   kperm32: p = 16h + 8a + 4b + j  ->  pi(p) = 16a + 8b + 4h + j
#   kperm64: p = 32i + 16h + 4c + j ->  pi(p) = 16c + 8i + 4h + j
_NVFP4_KPERM_FACTORS = {
    32: ((2, 2, 2, 4), (2, 0, 1, 3)),
    64: ((4, 2, 2, 4), (1, 2, 0, 3)),
}


def validate_nvfp4_kperm(kperm) -> int:
    if type(kperm) is not int or kperm not in NVFP4_KPERM_CHOICES:
        raise ValueError(f"NVFP4 kperm must be 0, 32 or 64; got {kperm!r}")
    return int(kperm)


def nvfp4_kperm_view(tensor: "torch.Tensor", dim: int, kperm: int) -> "torch.Tensor":
    """View ``tensor`` with ``dim`` split into kperm chunks iterated in pi order.

    The result has ``dim`` replaced by ``(length // kperm, *factors)``;
    flattening those dimensions in order yields the permuted sequence.
    """
    if not validate_nvfp4_kperm(kperm):
        raise ValueError("kperm=0 has no permuted view")
    sizes, order = _NVFP4_KPERM_FACTORS[kperm]
    dim %= tensor.ndim
    length = tensor.shape[dim]
    if length % kperm:
        raise ValueError(f"K-permuted dimension {length} is not a multiple of {kperm}")
    split = tensor.reshape(
        *tensor.shape[:dim], length // kperm, *sizes, *tensor.shape[dim + 1 :]
    )
    first = dim + 1
    dims = list(range(split.ndim))
    dims[first : first + 4] = [first + factor for factor in order]
    return split.permute(dims)


def nvfp4_kperm(tensor: "torch.Tensor", dim: int, kperm: int) -> "torch.Tensor":
    """Contiguous copy of ``tensor`` permuted by pi along ``dim`` (identity for 0)."""
    if not validate_nvfp4_kperm(kperm):
        return tensor
    return nvfp4_kperm_view(tensor, dim, kperm).reshape(tensor.shape)


def nvfp4_kperm_index(length: int, kperm: int, device=None) -> "torch.Tensor":
    """Gather index of the permutation: ``permuted = natural[..., index]``."""
    import torch

    return nvfp4_kperm(torch.arange(length, device=device), 0, kperm)


def copy_nvfp4_kperm_hidden(
    dst: "torch.Tensor", src: "torch.Tensor", *, kperm: int
) -> None:
    """Write natural-order rows ``src`` into contiguous ``dst`` in pi order.

    Used by activation staging: one strided copy, no extra pass or buffer.
    pi never splits a run of four consecutive K values (the innermost factor),
    so 1-byte rows move as 4-byte words: on H200 a byte-granular permuted copy
    of 32768x7168 bytes takes 590 us against 116 us for the plain copy, the
    word copy 183 us.
    """
    import torch

    sizes, order = _NVFP4_KPERM_FACTORS[validate_nvfp4_kperm(kperm)]
    if sizes[-1] != 4 or order[-1] != len(order) - 1:
        raise AssertionError("pi must keep runs of four K values")
    if src.element_size() != 1 or src.storage_offset() % 4 or dst.storage_offset() % 4:
        view = nvfp4_kperm_view(src, -1, kperm)
        dst.view(view.shape).copy_(view)
        return
    words = src.view(torch.int32)
    split = words.reshape(*words.shape[:-1], -1, *sizes[:-1])
    first = words.ndim
    dims = list(range(split.ndim))
    dims[first:] = [first + factor for factor in order[:-1]]
    view = split.permute(dims)
    dst.view(torch.int32).view(view.shape).copy_(view)


def _kperm_payload(payload: "torch.Tensor", kperm: int) -> "torch.Tensor":
    """Permute the K order of packed E2M1 bytes (two codes per byte, low first)."""
    import torch

    codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(
        *payload.shape[:-1], payload.shape[-1] * 2
    )
    codes = nvfp4_kperm(codes, -1, kperm)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _kperm_gate_up_rows(
    tensor: "torch.Tensor", *, intermediate_size: int, kperm: int
) -> "torch.Tensor":
    """Apply pi to the gate rows and to the up rows of canonical W1 [E,2I,...]."""
    e, rows, cols = tensor.shape
    halves = tensor.reshape(e, 2, intermediate_size, cols)
    return nvfp4_kperm(halves, 2, kperm).reshape(e, rows, cols)


def _swap_kperm_scale_pairs(scale: "torch.Tensor") -> "torch.Tensor":
    """Swap S(16q+g, 2t+1) with S(16q+g+8, 2t) in physical row order.

    Afterwards the folded u16 that natural-order conversion reads as two K16
    scales of one row holds one K16 scale of rows g and g+8 instead.
    """
    e, rows, groups = scale.shape
    s = scale.reshape(e, rows // 16, 2, 8, groups // 2, 2)
    out = s.clone()
    out[:, :, 0, :, :, 1] = s[:, :, 1, :, :, 0]
    out[:, :, 1, :, :, 0] = s[:, :, 0, :, :, 1]
    return out.reshape(e, rows, groups)


@dataclass(frozen=True)
class Nvfp4WeightPack(PrequantizedMoEWeights):
    """Raw E2M1 bytes, canonical K16 scale planes, and each leg's FP32 G.

    Global scales may be scalar tensors or one value per local expert.
    Plain PrequantizedMoEWeights is also accepted, with G=1 explicitly.
    """

    w13_global_scale: "torch.Tensor"
    w2_global_scale: "torch.Tensor"

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


def fold_nvfp4_scales(scales: "torch.Tensor") -> "torch.Tensor":
    """Physically transpose aligned K16 pairs inside each M64/K128 block."""
    import torch

    if scales.ndim != 3 or scales.dtype != torch.uint8:
        raise ValueError("NVFP4 scales must be uint8 [E,M,K/16]")
    e, rows, groups = scales.shape
    if rows % 64 or groups % 8:
        raise ValueError("NVFP4 scale folding requires M64/K128 alignment")
    return (
        scales.reshape(e, rows // 64, 4, 16, groups // 8, 4, 2)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(e, rows // 64, groups // 8, 32, 16)
    )


def _quantize_leg(weight: "torch.Tensor"):
    """RN-even E2M1 quantization with E4M3 K16 scales and per-expert G."""
    import torch

    e, rows, k = weight.shape
    blocks = weight.float().reshape(e, rows, k // 16, 16)
    amax = blocks.abs().amax(-1)
    global_scale = (amax.flatten(1).amax(1) / (6 * 448)).clamp_min(1e-30)
    scale = (
        (amax / (6 * global_scale[:, None, None])).clamp(0, 448).to(torch.float8_e4m3fn)
    )
    normalized = blocks / (scale.float() * global_scale[:, None, None]).clamp_min(
        1e-30
    ).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    boundaries = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=weight.device
    )
    codes = torch.bucketize(magnitude, boundaries).to(torch.uint8)
    ties = magnitude == boundaries[codes.clamp_max(6).long()]
    codes += (ties & ((codes & 1) != 0)).to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    codes = codes.reshape(e, rows, k)
    payload = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return payload, scale.view(torch.uint8).contiguous(), global_scale


def _raw_leg(payload, scale, global_scale, *, experts, rows, k, name):
    import torch

    if tuple(payload.shape) != (experts, rows, k // 2) or payload.dtype != torch.uint8:
        raise ValueError(f"{name} payload must be uint8 {(experts, rows, k // 2)}")
    if scale.dtype == torch.float8_e4m3fn:
        scale = scale.view(torch.uint8)
    if tuple(scale.shape) != (experts, rows, k // 16) or scale.dtype != torch.uint8:
        raise ValueError(f"{name} scales must be E4M3 bytes {(experts, rows, k // 16)}")
    if not payload.is_cuda or not scale.is_cuda or payload.device != scale.device:
        raise ValueError(f"{name} payload and scale must share a CUDA device")
    if not payload.is_contiguous() or not scale.is_contiguous():
        raise ValueError(f"{name} raw payload and scale must be contiguous")
    if not bool((scale < 127).all()):
        raise ValueError(f"{name} scales must be finite nonnegative E4M3 codes 0..126")
    if global_scale is None:
        global_scale = torch.ones(experts, device=payload.device, dtype=torch.float32)
    if (
        not isinstance(global_scale, torch.Tensor)
        or global_scale.dtype != torch.float32
    ):
        raise ValueError(f"{name} global scale must be a float32 tensor")
    if global_scale.device != payload.device or global_scale.numel() not in (
        1,
        experts,
    ):
        raise ValueError(
            f"{name} global scale must be scalar or one value per expert on the weight device"
        )
    global_scale = global_scale.reshape(-1).expand(experts).contiguous()
    if not bool((torch.isfinite(global_scale) & (global_scale > 0)).all()):
        raise ValueError(f"{name} global scales must be positive and finite")
    return payload, scale, global_scale


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind="fp8_e4m3",
    fp8_scale_mode="mxfp4_hybrid",
    expert_chunk_size: int = 4,
    nvfp4_kperm: int = 64,
) -> TransformedMegaWeights:
    """Raw NVFP4 or BF16 weights to packed SM90 kernel layouts.

    ``nvfp4_kperm`` selects the K layout used by input staging and the kernel.
    The returned two-leg tuple carries this identity for validation before launch.
    """
    import torch
    from flashinfer.fused_moe.prepare import interleave_moe_weights_for_sm90_mixed_gemm
    from ..fp8_mxfp4_bf16_pull_cutedsl.weights import _interleave_gate_up_8

    if kind != "fp8_e4m3" or fp8_scale_mode != "mxfp4_hybrid":
        raise ValueError("NVFP4 requires FP8 E4M3 per-token activation scaling")
    kperm = validate_nvfp4_kperm(nvfp4_kperm)
    if (
        hidden_size <= 0
        or intermediate_size <= 0
        or hidden_size % 128
        or intermediate_size % 128
        or expert_chunk_size <= 0
    ):
        raise ValueError(
            "NVFP4 requires positive M64/K128-aligned shapes and chunk size"
        )
    experts = weights.w13.shape[0]
    if experts < 1 or weights.w2.shape[0] != experts:
        raise ValueError("both NVFP4 legs must have the same positive expert count")
    shapes = ((2 * intermediate_size, hidden_size), (hidden_size, intermediate_size))
    legs = []
    for leg, (rows, k) in enumerate(shapes):
        name = "w13" if leg == 0 else "w2"
        original = getattr(weights, name)
        if isinstance(weights, PrequantizedMoEWeights):
            raw = _raw_leg(
                original,
                getattr(weights, name + "_scale"),
                getattr(weights, name + "_global_scale", None),
                experts=experts,
                rows=rows,
                k=k,
                name=name,
            )
        else:
            if tuple(original.shape) != (experts, rows, k) or original.dtype not in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                raise ValueError(
                    f"{name} canonical weights must have shape {(experts, rows, k)}"
                )
            if not original.is_cuda or not bool(torch.isfinite(original).all()):
                raise ValueError(
                    f"{name} canonical weights must be finite CUDA tensors"
                )
            raw = None
        result = None
        for begin in range(0, experts, expert_chunk_size):
            end = min(begin + expert_chunk_size, experts)
            if raw is None:
                payload, scale, global_scale = _quantize_leg(original[begin:end])
            else:
                payload, scale, global_scale = (x[begin:end] for x in raw)
            if kperm:
                payload = _kperm_payload(payload, kperm)
                if leg == 0:
                    payload, scale = (
                        _kperm_gate_up_rows(
                            t, intermediate_size=intermediate_size, kperm=kperm
                        )
                        for t in (payload, scale)
                    )
            if leg == 0:
                payload = _interleave_gate_up_8(
                    payload, intermediate_size=intermediate_size
                )
                scale = _interleave_gate_up_8(
                    scale, intermediate_size=intermediate_size
                )
            if kperm:
                scale = _swap_kperm_scale_pairs(scale)
            packed = interleave_moe_weights_for_sm90_mixed_gemm(
                payload, "fp4_fp8"
            ).transpose(1, 2)
            folded = fold_nvfp4_scales(scale)
            decode = (global_scale * 8).contiguous()
            if result is None:
                out_payload = torch.empty(
                    (experts, rows, k // 2), device=packed.device, dtype=torch.uint8
                ).transpose(1, 2)
                out_scale = torch.empty(
                    (experts, rows // 64, k // 128, 32, 16),
                    device=packed.device,
                    dtype=torch.uint8,
                )
                out_decode = torch.empty(
                    experts, device=packed.device, dtype=torch.float32
                )
                result = (
                    out_payload,
                    out_scale,
                    torch.ones(1, device=packed.device, dtype=torch.float32),
                    out_decode,
                )
            result[0][begin:end].copy_(packed)
            result[1][begin:end].copy_(folded)
            result[3][begin:end].copy_(decode)
        legs.append(result)
    if legs[0][0].device != legs[1][0].device:
        raise ValueError("both NVFP4 weight legs must share one CUDA device")
    return TransformedMegaWeights(legs[0], legs[1], nvfp4_kperm=kperm)


def validate_transformed_mega_weights(
    transformed_weights,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind="fp8_e4m3",
    fp8_scale_mode="mxfp4_hybrid",
    world_size: int,
    num_experts: int,
    nvfp4_kperm: int = 64,
) -> None:
    """Validate layout identity before inspecting the packed-K/scale tensors."""
    expected_kperm = validate_nvfp4_kperm(nvfp4_kperm)
    if (
        not isinstance(transformed_weights, TransformedMegaWeights)
        or type(getattr(transformed_weights, "nvfp4_kperm", None)) is not int
    ):
        raise ValueError(
            "NVFP4 transformed weights require layout metadata; use "
            "preprocess_sm90_pull_nvfp4_mega_weights"
        )
    if transformed_weights.nvfp4_kperm != expected_kperm:
        raise ValueError(
            f"NVFP4 transformed layout nvfp4_kperm={transformed_weights.nvfp4_kperm} "
            f"does not match configuration nvfp4_kperm={expected_kperm}"
        )
    import torch

    if kind != "fp8_e4m3" or fp8_scale_mode != "mxfp4_hybrid":
        raise ValueError("NVFP4 requires FP8 E4M3 hybrid activation scaling")
    if (
        world_size <= 0
        or num_experts <= 0
        or num_experts % world_size
        or len(transformed_weights) != 2
    ):
        raise ValueError("NVFP4 requires an even expert partition and two weight legs")
    e = num_experts // world_size
    device = None
    for leg, (rows, k) in zip(
        transformed_weights,
        ((2 * intermediate_size, hidden_size), (hidden_size, intermediate_size)),
        strict=True,
    ):
        if len(leg) != 4:
            raise ValueError("each NVFP4 leg must have four launch slots")
        payload, sf, unit, decode = leg
        if (
            payload.dtype != torch.uint8
            or tuple(payload.shape) != (e, k // 2, rows)
            or payload.stride(1) != 1
        ):
            raise ValueError("NVFP4 packed weights require storage-K stride 1")
        if (
            sf.dtype != torch.uint8
            or tuple(sf.shape) != (e, rows // 64, k // 128, 32, 16)
            or not sf.is_contiguous()
        ):
            raise ValueError("NVFP4 scales must use the transposed K16-pair layout")
        if not bool((sf < 127).all()):
            raise ValueError("NVFP4 scales contain signed or nonfinite codes")
        for t, shape in ((unit, (1,)), (decode, (e,))):
            if (
                t.dtype != torch.float32
                or tuple(t.shape) != shape
                or not t.is_contiguous()
            ):
                raise ValueError("NVFP4 launch scales must be contiguous FP32 tensors")
            if not bool((torch.isfinite(t) & (t > 0)).all()):
                raise ValueError("NVFP4 launch scales must be positive and finite")
        if not bool((unit == 1).all()):
            raise ValueError("NVFP4 activation placeholder must be exactly one")
        for t in leg:
            if (
                not t.is_cuda
                or t.device != torch.device("cuda", torch.cuda.current_device())
                or t.data_ptr() % 16
            ):
                raise ValueError(
                    "NVFP4 launch tensors must be aligned on the current CUDA device"
                )
            if device is None:
                device = t.device
            elif t.device != device:
                raise ValueError("NVFP4 launch tensors must share one CUDA device")
