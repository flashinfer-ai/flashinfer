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

Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, MIT licensed;
see LICENSE.DeepGEMM at the root of this package. Sources:
``deep_gemm/utils/math.py`` and ``deep_gemm/mega/__init__.py`` of the upstream
revision (../VENDOR.md).
"""

# Weight preparation of the experimental MOSCA MegaMoE backend.
#
# The kernel reads the same weights as ``sm100_fp8_fp4_bf16_deepgemm``: packed
# FP4 (E2M1) with per-32 UE8M0 scales, L1 gate/up interleaved in groups of 8
# rows, scale factors packed four K-blocks per int32, MN-major, TMA-aligned and
# transposed for UTCCP. The stable backend reaches these transforms through the
# installed ``deep_gemm`` package; this backend carries torch ports of them so
# that it never imports ``deep_gemm`` (the transforms are pure layout moves, so
# the bytes are identical; the tests compare against ``deep_gemm`` when it is
# installed).

from __future__ import annotations

from typing import Tuple

import torch

from ....moe_ep.weights import MoEWeightPack, PrequantizedMoEWeights

GRAN_K = 32  # per-32 UE8M0 scales (DeepGEMM recipe (1, 1, 32))

TransformedMoscaMegaWeights = Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


# ---------------------------------------------------------------------------------------------
# Quantization (DeepGEMM ``ceil_to_ue8m0`` / ``_quantize_to_fp4_e2m1`` / ``per_token_cast_to_fp4``).
# ---------------------------------------------------------------------------------------------
def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary, round_up in zip(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        (False, True, False, True, False, True, False),
        strict=False,
    ):
        code += (ax >= boundary if round_up else ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(
    x: torch.Tensor, gran_k: int = GRAN_K
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[m, n]`` -> packed FP4 ``int8 [m, n / 2]`` and FP32 UE8M0 scales ``[m, n / gran_k]``."""
    m, n = x.shape
    if n % 2 != 0:
        raise ValueError("FP4 packing needs an even inner dimension")
    padded_n = _align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), sf


def dequantize_fp4(
    packed: torch.Tensor, sf: torch.Tensor, gran_k: int = GRAN_K
) -> torch.Tensor:
    """Inverse of :func:`per_token_cast_to_fp4` (for references and tests)."""
    m, n2 = packed.shape
    n = n2 * 2
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
        dtype=torch.float,
    )
    sign, idx = (unpacked & 0x08) != 0, (unpacked & 0x07).to(torch.long)
    value = values[idx]
    deq = torch.where(sign & (idx != 0), -value, value)
    return deq * sf[:, torch.arange(n, device=packed.device) // gran_k]


# ---------------------------------------------------------------------------------------------
# Scale-factor layout (DeepGEMM ``transform_sf_into_required_layout`` for (FP32, 1, 32) on SM100:
# UE8M0 bytes packed four K-blocks per int32, MN-major, TMA-aligned MN stride).
# ---------------------------------------------------------------------------------------------
def pack_sf_mn_major(sf: torch.Tensor) -> torch.Tensor:
    """FP32 power-of-two scales ``[G, MN, K/32]`` -> int32 ``[G, MN, ceil(K/128)]``, MN-major."""
    if sf.dtype != torch.float32 or sf.dim() != 3:
        raise ValueError("expected FP32 scales of shape [groups, mn, k_blocks]")
    num_groups, mn, k_blocks = sf.shape
    ue8m0 = (sf.contiguous().view(torch.int) >> 23).to(torch.uint8)
    aligned_mn = _align(mn, 4)  # 16-byte TMA alignment of int32 rows
    aligned_k = _align(k_blocks, 4)
    padded = torch.zeros(
        (num_groups, aligned_mn, aligned_k), device=sf.device, dtype=torch.uint8
    )
    padded[:, :mn, :k_blocks] = ue8m0
    packed = (
        padded.view(-1).view(torch.int).view(num_groups, aligned_mn, aligned_k // 4)
    )
    out = torch.empty(
        (num_groups, aligned_k // 4, aligned_mn), device=sf.device, dtype=torch.int
    ).transpose(1, 2)
    out[:, :, :] = packed
    return out[:, :mn, :]


def _as_packed_int_sf(
    sf: torch.Tensor, mn: int, k: int, num_groups: int
) -> torch.Tensor:
    if sf.dtype == torch.uint8:
        sf = (sf.to(torch.int32) << 23).view(torch.float32)
    if sf.dtype == torch.float32:
        if tuple(sf.shape) != (num_groups, mn, _ceil_div(k, GRAN_K)):
            raise ValueError(
                f"weight scales must be [{num_groups}, {mn}, {_ceil_div(k, GRAN_K)}], "
                f"got {tuple(sf.shape)}"
            )
        return pack_sf_mn_major(sf)
    if sf.dtype == torch.int32:
        expected = (num_groups, mn, _ceil_div(k, GRAN_K * 4))
        if (
            tuple(sf.shape) != expected
            or sf.stride(1) != 1
            or sf.stride(2) != _align(mn, 4)
        ):
            raise ValueError(
                f"packed int32 weight scales must be MN-major {expected} with a TMA-aligned stride"
            )
        return sf
    raise TypeError(f"unsupported weight scale dtype {sf.dtype}")


# ---------------------------------------------------------------------------------------------
# MegaMoE weight transform (DeepGEMM ``transform_weights_for_mega_moe``, FP8 x FP4 branch).
# ---------------------------------------------------------------------------------------------
def _interleave_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """``[G, 2I, ...]`` gate|up halves -> interleaved blocks of ``gran`` rows."""
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(
        torch.stack([gate, up], dim=2).reshape(g, n, *rest)
    )


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    if sf.dtype != torch.int or mn % 128 != 0:
        raise ValueError(
            "UTCCP scale transpose needs int32 scales with MN a multiple of 128"
        )
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def _as_fp4_bytes(w: torch.Tensor) -> torch.Tensor:
    if w.dtype in (torch.uint8, torch.int8):
        return w.view(torch.int8).contiguous()
    raise TypeError(
        "pre-quantized MoEWeightPack weights must be packed FP4 bytes (torch.int8 or "
        f"torch.uint8), got {w.dtype}"
    )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMoscaMegaWeights:
    """Canonical weights -> the kernel layout ``((l1, l1_sf), (l2, l2_sf))``.

    ``weights`` holds this rank's experts: ``w13`` ``[E, 2 * intermediate, hidden]``
    (gate rows, then up rows) and ``w2`` ``[E, hidden, intermediate]``, either
    BF16 (quantized here, per-32 UE8M0 FP4) or pre-quantized packed FP4 bytes
    (``[..., K / 2]``) with FP32 / UE8M0 ``[E, N, K / 32]`` scales.
    """
    if isinstance(weights, PrequantizedMoEWeights):
        w13 = _as_fp4_bytes(weights.w13)
        w2 = _as_fp4_bytes(weights.w2)
        num_groups = w13.shape[0]
        w13_sf = _as_packed_int_sf(
            weights.w13_scale, 2 * intermediate_size, hidden_size, num_groups
        )
        w2_sf = _as_packed_int_sf(
            weights.w2_scale, hidden_size, intermediate_size, num_groups
        )
    else:
        num_groups = weights.w13.shape[0]
        w13 = torch.empty(
            (num_groups, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.int8,
            device=weights.w13.device,
        )
        w2 = torch.empty(
            (num_groups, hidden_size, intermediate_size // 2),
            dtype=torch.int8,
            device=weights.w2.device,
        )
        w13_sf_f = torch.empty(
            (num_groups, 2 * intermediate_size, hidden_size // GRAN_K),
            dtype=torch.float32,
            device=weights.w13.device,
        )
        w2_sf_f = torch.empty(
            (num_groups, hidden_size, intermediate_size // GRAN_K),
            dtype=torch.float32,
            device=weights.w2.device,
        )
        for e in range(num_groups):
            w13[e], w13_sf_f[e] = per_token_cast_to_fp4(weights.w13[e])
            w2[e], w2_sf_f[e] = per_token_cast_to_fp4(weights.w2[e])
        w13_sf = pack_sf_mn_major(w13_sf_f)
        w2_sf = pack_sf_mn_major(w2_sf_f)

    if tuple(w13.shape) != (num_groups, 2 * intermediate_size, hidden_size // 2):
        raise ValueError(
            f"w13 must be [E, {2 * intermediate_size}, {hidden_size // 2}] FP4 bytes"
        )
    if tuple(w2.shape) != (num_groups, hidden_size, intermediate_size // 2):
        raise ValueError(
            f"w2 must be [E, {hidden_size}, {intermediate_size // 2}] FP4 bytes"
        )
    l1 = (
        _interleave_gate_up(w13),
        _transpose_sf_for_utccp(_interleave_gate_up(w13_sf)),
    )
    l2 = (w2, _transpose_sf_for_utccp(w2_sf))
    return l1, l2


def validate_transformed_mega_weights(
    transformed: TransformedMoscaMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
) -> None:
    """One-time check of caller-supplied kernel-ready weights (``preprocess_weights=False``)."""
    from ....moe_ep.core.validation.common import MoEEpConfigError

    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError("transformed_weights must be ((l1, l1_sf), (l2, l2_sf))")
    expected = (
        (
            "l1",
            (num_local_experts, 2 * intermediate_size, hidden_size // 2),
            2 * intermediate_size,
            hidden_size,
        ),
        (
            "l2",
            (num_local_experts, hidden_size, intermediate_size // 2),
            hidden_size,
            intermediate_size,
        ),
    )
    for (label, w_shape, mn, k), pair in zip(expected, transformed, strict=False):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be a (weight, scale) pair"
            )
        w, sf = pair
        if (
            not isinstance(w, torch.Tensor)
            or w.dtype != torch.int8
            or tuple(w.shape) != w_shape
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be int8 {w_shape}"
            )
        sf_shape = (num_local_experts, mn, _ceil_div(k, GRAN_K * 4))
        if (
            not isinstance(sf, torch.Tensor)
            or sf.dtype != torch.int32
            or tuple(sf.shape) != sf_shape
            or sf.stride(1) != 1
            or sf.stride(2) != _align(mn, 4)
            # Expert groups back to back: the kernel's tensor map folds them
            # into one dimension (upstream: stride(-3) == stride(-1) * size(-1)).
            or sf.stride(0) != sf.stride(2) * sf.size(2)
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} scale must be MN-major int32 "
                f"{sf_shape} with expert groups packed back to back"
            )


__all__ = [
    "GRAN_K",
    "TransformedMoscaMegaWeights",
    "dequantize_fp4",
    "pack_sf_mn_major",
    "per_token_cast_to_fp4",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
