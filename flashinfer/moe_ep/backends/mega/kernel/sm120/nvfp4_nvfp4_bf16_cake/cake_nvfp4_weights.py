# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight preprocessing for the Cake SM120 NVFP4 x NVFP4 (W4A4) MegaMoE backend.

Canonical input (the contract of the sibling ``nvfp4_nvfp4_bf16_cutedsl`` backends):

* :class:`~flashinfer.moe_ep.weights.PrequantizedMoEWeights` with packed E2M1 ``w13``
  ``[local_E, 2 * I, H // 2]`` (gate rows then up rows, low nibble first) and ``w2``
  ``[local_E, H, I // 2]``, and E4M3 block-16 scales ``w13_scale`` ``[local_E, 2 * I, H // 16]``
  / ``w2_scale`` ``[local_E, H, I // 16]`` (``float8_e4m3fn`` or their ``uint8`` bytes);
  weight global scales travel in ``fc1_alpha`` / ``fc2_alpha`` of the config;
* or BF16 :class:`~flashinfer.moe_ep.weights.UnquantizedMoEWeights` (``w13 [local_E, 2 * I, H]``,
  ``w2 [local_E, H, I]``), quantized here per 16 values with ``scale = e4m3(amax / 6)``.

MXFP4 checkpoints (UE8M0 block-32 scales) convert exactly with :func:`mxfp4_scales_to_nvfp4`
when every scale is a power of two inside the E4M3 range (2^-9 .. 2^8).  The in-kernel FP8
shared expert (``shared_expert="fp8_in_kernel"``) takes
:class:`~flashinfer.moe_ep.CakePrequantizedMoEWeightsWithSharedExpert` (routed NVFP4 planes plus
FP8 e4m3 shared weights with per-32 UE8M0 scales).

Kernel layout: the MXFP8 backend's (gate / up interleaved by 8 rows, scales packed four bytes
per ``uint32`` word K-block-major, then the scale-vector permutation); an NVFP4 word holds 64
values, so a K128 block takes two words, and the shared-expert segment keeps its UE8M0 words in
the first ``K // 128`` rows of a ``K // 64``-row segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mxfp4_mxfp8_bf16_cake.cake_weights import (
    FP4_PACKED_DTYPES,
    FP8_DTYPES,
    NOSHARED_SCALE_FILL_WORD,
    SCALE_DTYPES,
    CakePrequantizedMoEWeightsWithSharedExpert,
    TransformedCakeMegaWeights,
    _as_packed_u8,
    _check_shape,
    _noshared_operands,
    interleave_gate_up_rows,
    pack_scales_kblock_major,
    scale_vector_layout,
)

try:  # imported as part of flashinfer
    from ......weights import (
        MoEWeightPack,
        PrequantizedMoEWeights,
        UnquantizedMoEWeights,
    )
except ImportError:  # host-only unit tests import this module standalone
    MoEWeightPack = object  # type: ignore[assignment,misc]
    PrequantizedMoEWeights = UnquantizedMoEWeights = None  # type: ignore[assignment,misc]

NVFP4_BLOCK = 16
E4M3_SCALE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
_E2M1_GRID_GT = (0.25, 1.25, 2.5, 5.0)  # ties to the lower (even) code
_E2M1_GRID_GE = (0.75, 1.75, 3.5)  # ties to the upper (even) code


def e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """fp32 values already divided by their block scale -> e2m1 nibbles (round to nearest even, saturating)."""

    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.int32, device=scaled.device)
    for threshold in _E2M1_GRID_GT:
        code += (magnitude > threshold).to(torch.int32)
    for threshold in _E2M1_GRID_GE:
        code += (magnitude >= threshold).to(torch.int32)
    return code | (torch.signbit(scaled).to(torch.int32) << 3)


def nvfp4_quantize_torch(
    values: torch.Tensor, norm_const: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32/bf16 ``[..., K]`` -> (packed E2M1 ``uint8 [..., K // 2]``, E4M3 scale bytes ``uint8 [..., K // 16]``).

    Per 16 values ``scale = e4m3(min((amax / 6) * norm, 448))`` and codes
    ``e2m1(x * (norm * (1 / scale)))`` with an IEEE round-to-nearest reciprocal, ties to the even
    code (a zero scale gives zero codes).  ``norm_const`` is a scalar or a tensor broadcast over the
    leading dims.

    Why a separate quantizer: this is the Cake kernel's own requantization rule (the
    ``norm * (1 / scale)`` product, not ``norm / scale``), reproduced bit for bit so the tests can
    compare packed codes with the kernel exactly and the staged input matches what an NVFP4
    producer of the kernel would emit.  It is the reference of the fused Triton twin
    (:func:`~.cake_nvfp4_quant.nvfp4_quantize_fused`, used for ``quantize_input``); it is not a
    replacement for the public :func:`flashinfer.nvfp4_quantize`.
    """

    if values.shape[-1] % NVFP4_BLOCK:
        raise ValueError(
            f"the last dim must be a multiple of {NVFP4_BLOCK}, got {tuple(values.shape)}"
        )
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], shape[-1] // NVFP4_BLOCK, NVFP4_BLOCK)
    # a Python scalar stays a scalar (no host->device copy: the call must be CUDA-graph capturable)
    norm = (
        float(norm_const)
        if isinstance(norm_const, (int, float))
        else norm_const.to(blocks.device, torch.float32)
    )
    if not isinstance(norm, float) and norm.dim():
        norm = norm.reshape(*norm.shape, *([1] * (blocks.dim() - 1 - norm.dim())))
    raw = (blocks.abs().amax(dim=-1) * (1.0 / 6.0)) * norm
    scale = raw.clamp(max=448.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    inverse = torch.where(
        scale_f > 0, norm * torch.reciprocal(scale_f), torch.zeros_like(scale_f)
    )
    nibbles = e2m1_codes(blocks * inverse.unsqueeze(-1)).reshape(shape)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed, scale.view(torch.uint8)


def mxfp4_scales_to_nvfp4(sf_ue8m0: torch.Tensor) -> torch.Tensor:
    """UE8M0 block-32 scale bytes ``[..., K // 32]`` -> the equal E4M3 block-16 scales ``uint8 [..., K // 16]``.

    Exact (each block-32 scale serves both of its block-16 halves); raises when a scale lies
    outside the E4M3 range 2^-9 .. 2^8.
    """

    exponent = sf_ue8m0.view(torch.uint8).to(torch.int32) - 127
    if bool(((exponent < -9) | (exponent > 8)).any()):
        raise ValueError(
            "UE8M0 scale outside the E4M3 range 2^-9..2^8: no exact NVFP4 scale"
        )
    e4m3 = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent).to(
        torch.float8_e4m3fn
    )
    return e4m3.view(torch.uint8).repeat_interleave(2, dim=-1)


def mxfp4_scales_to_nvfp4_shifted(
    sf_ue8m0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert exact conversion of UE8M0 block-32 scales ``[E, ..., K // 32]`` whose span exceeds E4M3's.

    E4M3 holds every power of two 2^-9 .. 2^8 (18 octaves).  Each expert ``e`` gets one shift ``s_e``
    (a power of two applied to the whole tensor, ``s_e = 0`` when its scales already fit) so that
    ``stored = true * 2^s_e`` lies in that range; the returned ``alpha[e] = 2^-s_e`` (fp32 ``[E]``) restores
    the true accumulation and is multiplied into ``fc1_alpha`` (for ``w13`` scales) or ``fc2_alpha`` (``w2``).
    Still bit-exact: only exponents move.  Returns (E4M3 scale bytes ``uint8 [E, ..., K // 16]``, alpha).
    Raises when one expert's own span exceeds 18 octaves, or on a NaN / zero scale byte.
    """

    raw = sf_ue8m0.view(torch.uint8)
    if raw.dim() < 2:
        raise ValueError("expected [E, ..., K // 32] scale bytes")
    if bool(((raw == 255) | (raw == 0)).any()):
        raise ValueError("UE8M0 scale byte 0 (2^-127) or 255 (NaN) cannot be converted")
    exponent = raw.to(torch.int32) - 127
    flat = exponent.reshape(exponent.shape[0], -1)
    lo, hi = flat.amin(dim=1), flat.amax(dim=1)
    if bool((hi - lo > 17).any()):
        raise ValueError(
            "an expert's UE8M0 scales span more than 18 octaves: no exact E4M3 representation"
        )
    # lo < -9 raises the smallest scale to 2^-9; hi > 8 lowers the largest to 2^8; else no shift
    shift = torch.where(
        lo < -9, -9 - lo, torch.where(hi > 8, 8 - hi, torch.zeros_like(lo))
    )
    shifted = exponent + shift.view(-1, *([1] * (exponent.dim() - 1)))
    if bool(((shifted < -9) | (shifted > 8)).any()):
        raise ValueError("no shift fits an expert's UE8M0 scales into 2^-9 .. 2^8")
    e4m3 = torch.ldexp(torch.ones_like(shifted, dtype=torch.float32), shifted).to(
        torch.float8_e4m3fn
    )
    alpha = torch.ldexp(torch.ones_like(shift, dtype=torch.float32), -shift)
    return e4m3.view(torch.uint8).repeat_interleave(2, dim=-1), alpha


@dataclass(frozen=True)
class TransformedCakeNvfp4MegaWeights(TransformedCakeMegaWeights):
    """Kernel-layout rank-local NVFP4 weights (scale words hold 64 values)."""


def _shared_segment(
    sf_words: torch.Tensor, words: int, columns: int, device: torch.device
) -> torch.Tensor:
    """The shared expert's UE8M0 words padded to an NVFP4-sized segment of ``words`` rows."""

    segment = torch.full(
        (words, columns), NOSHARED_SCALE_FILL_WORD, dtype=torch.int64, device=device
    )
    segment = segment.to(torch.int32).view(torch.uint32)
    segment[: sf_words.shape[0]].copy_(sf_words)
    return segment


def transform_nvfp4_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    shared_w13: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w13_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
) -> TransformedCakeNvfp4MegaWeights:
    """Canonical packed E2M1 + E4M3 block-16 planes (and the optional FP8 shared expert) -> kernel layout."""

    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    H, I = hidden_size, intermediate_size
    w13 = _as_packed_u8(w13, "w13", FP4_PACKED_DTYPES)
    w2 = _as_packed_u8(w2, "w2", FP4_PACKED_DTYPES)
    w13_scale = _as_packed_u8(w13_scale, "w13_scale", E4M3_SCALE_DTYPES)
    w2_scale = _as_packed_u8(w2_scale, "w2_scale", E4M3_SCALE_DTYPES)
    L = int(w13.shape[0])
    for name, tensor, shape in (
        ("w13", w13, (L, 2 * I, H // 2)),
        ("w2", w2, (L, H, I // 2)),
        ("w13_scale", w13_scale, (L, 2 * I, H // NVFP4_BLOCK)),
        ("w2_scale", w2_scale, (L, H, I // NVFP4_BLOCK)),
    ):
        _check_shape(name, tensor, shape)
    shared = (shared_w13, shared_w2, shared_w13_scale, shared_w2_scale)
    has_shared = shared_w13 is not None
    if any((t is None) != (not has_shared) for t in shared):
        raise ValueError(
            "the shared expert needs all four of shared_w13/w2 and their scales"
        )
    device = w13.device
    # four E4M3 bytes per word = 64 values: the MXFP8 packer works unchanged on the byte planes
    w1_sf = pack_scales_kblock_major(interleave_gate_up_rows(w13_scale))
    w2_sf = pack_scales_kblock_major(w2_scale)
    if has_shared:
        s_w13 = _as_packed_u8(shared_w13, "shared_w13", FP8_DTYPES)
        s_w2 = _as_packed_u8(shared_w2, "shared_w2", FP8_DTYPES)
        s_w13_scale = _as_packed_u8(shared_w13_scale, "shared_w13_scale", SCALE_DTYPES)
        s_w2_scale = _as_packed_u8(shared_w2_scale, "shared_w2_scale", SCALE_DTYPES)
        for name, tensor, sshape in (
            ("shared_w13", s_w13, (2 * I, H)),
            ("shared_w2", s_w2, (H, I)),
            ("shared_w13_scale", s_w13_scale, (2 * I, H // 32)),
            ("shared_w2_scale", s_w2_scale, (H, I // 32)),
        ):
            _check_shape(name, tensor, sshape)
        w1_b8 = interleave_gate_up_rows(s_w13, axis=0)
        w2_b8 = s_w2.contiguous()
        w1_shared = pack_scales_kblock_major(
            interleave_gate_up_rows(s_w13_scale, axis=0).unsqueeze(0)
        )
        w2_shared = pack_scales_kblock_major(s_w2_scale.unsqueeze(0))
    else:
        w1_b8, w2_b8 = _noshared_operands(device, H, I)
        w1_shared = w1_sf.new_empty((0, 2 * I))
        w2_shared = w2_sf.new_empty((0, H))
    w1_seg = _shared_segment(w1_shared, H // 64, 2 * I, device)
    w2_seg = _shared_segment(w2_shared, I // 64, H, device)
    return TransformedCakeNvfp4MegaWeights(
        w1_fp4=interleave_gate_up_rows(w13),
        w1_sfb=scale_vector_layout(torch.cat([w1_sf, w1_seg], dim=0)),
        w2_fp4=w2.contiguous(),
        w2_sfb=scale_vector_layout(torch.cat([w2_sf, w2_seg], dim=0)),
        w1_b8=w1_b8,
        w2_b8=w2_b8,
        hidden_size=H,
        intermediate_size=I,
        local_experts=L,
        has_shared_expert=has_shared,
    )


def preprocess_nvfp4_mega_weights(
    weights: MoEWeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
    shared_expert: str | None = None,
) -> TransformedCakeNvfp4MegaWeights:
    shared: dict[str, torch.Tensor] = {}
    if shared_expert is not None:
        if not isinstance(weights, CakePrequantizedMoEWeightsWithSharedExpert):
            raise NotImplementedError(
                "shared_expert='fp8_in_kernel' requires CakePrequantizedMoEWeightsWithSharedExpert "
                "(NVFP4 routed planes plus FP8 e4m3 shared_w13/shared_w2 with per-32 UE8M0 scales)"
            )
        shared = {
            "shared_w13": weights.shared_w13,
            "shared_w2": weights.shared_w2,
            "shared_w13_scale": weights.shared_w13_scale,
            "shared_w2_scale": weights.shared_w2_scale,
        }
    if UnquantizedMoEWeights is not None and isinstance(weights, UnquantizedMoEWeights):
        if shared:
            raise NotImplementedError(
                "BF16 routed weights with the in-kernel FP8 shared expert are not supported"
            )
        w13, w13_scale = nvfp4_quantize_torch(weights.w13)
        w2, w2_scale = nvfp4_quantize_torch(weights.w2)
    elif PrequantizedMoEWeights is not None and isinstance(
        weights, PrequantizedMoEWeights
    ):
        w13, w2, w13_scale, w2_scale = (
            weights.w13,
            weights.w2,
            weights.w13_scale,
            weights.w2_scale,
        )
    else:
        raise NotImplementedError(
            "sm120_nvfp4_nvfp4_bf16_cake takes PrequantizedMoEWeights (packed E2M1 + E4M3 block-16 scales) "
            "or UnquantizedMoEWeights (BF16)"
        )
    return transform_nvfp4_weights(
        w13,
        w2,
        w13_scale,
        w2_scale,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        **shared,
    )


__all__ = [
    "TransformedCakeNvfp4MegaWeights",
    "e2m1_codes",
    "mxfp4_scales_to_nvfp4",
    "mxfp4_scales_to_nvfp4_shifted",
    "nvfp4_quantize_torch",
    "preprocess_nvfp4_mega_weights",
    "transform_nvfp4_weights",
]
