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

"""Weight preprocessing for the Cake SM120 MXFP4 x MXFP8 MegaMoE backend.

Canonical input (same contract as the sibling ``mxfp4_mxfp8_bf16_cutedsl``
backend): a :class:`~flashinfer.moe_ep.weights.PrequantizedMoEWeights` with

* ``w13``: packed E2M1 ``[local_E, 2 * I, H // 2]`` (gate rows then up rows,
  low nibble first), ``uint8`` / ``int8`` / ``float4_e2m1fn_x2``;
* ``w2``: packed E2M1 ``[local_E, H, I // 2]``;
* ``w13_scale``: UE8M0 exponent bytes ``[local_E, 2 * I, H // 32]``;
* ``w2_scale``: UE8M0 exponent bytes ``[local_E, H, I // 32]``.

The in-kernel FP8 shared expert (``shared_expert="fp8_in_kernel"``) additionally
takes :class:`CakePrequantizedMoEWeightsWithSharedExpert`: FP8 e4m3 ``shared_w13``
``[2 * I, H]`` / ``shared_w2`` ``[H, I]`` with per-32 UE8M0 scales
``[2 * I, H // 32]`` / ``[H, I // 32]``.  Checkpoints that store 128 x 128 block
scales convert exactly with :func:`block128_scales_to_k32` when every block scale
is a power of two.

Kernel layout:

* gate/up rows interleaved in 16-row groups ``[gate 8 rows, up 8 rows]``
  (``gate_up_interleave_index``), applied to the routed and shared ``w13`` and
  to their scales;
* scales packed four K-blocks per ``uint32`` word, K-block-major
  ``[E * (K // 128), N]`` (``pack_scales_kblock_major``), with one extra
  segment ``E`` holding the shared expert's scales (a constant filler for the
  routed-only variant), then permuted into the kernel's scale-vector order
  (``scale_vector_layout``): per 128-column N tile, eight 16-word vectors whose
  word ``nt`` of vector ``g`` is column ``(nt ^ ((g >> 1) * 4)) * 8 + g``.

These helpers are pure PyTorch and importable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:  # imported as part of flashinfer
    from ......weights import MoEWeightPack, PrequantizedMoEWeights
except ImportError:  # host-only unit tests import this module standalone
    MoEWeightPack = object  # type: ignore[assignment,misc]
    PrequantizedMoEWeights = None  # type: ignore[assignment,misc]

FP4_PACKED_DTYPES = (torch.uint8, torch.int8, torch.float4_e2m1fn_x2)
FP8_DTYPES = (torch.uint8, torch.float8_e4m3fn)
SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
SCALE_TILE_N = 128  # MMA_TILE_N: columns per scale tile
SCALE_VECTOR_WORDS = 16  # MMA_N_SUBTILES: words per scale vector
# Routed-only variant: content of the (unused) shared-expert operands, as in the
# generator's own qualification harness.
NOSHARED_FP8_FILL = 0x38  # e4m3 1.0
NOSHARED_SCALE_FILL_WORD = 0x7F7F7F7F  # four UE8M0 2^0 exponents


if PrequantizedMoEWeights is not None:

    @dataclass(frozen=True)
    class CakePrequantizedMoEWeightsWithSharedExpert(PrequantizedMoEWeights):
        """Routed MXFP4 experts plus one FP8 e4m3 shared expert (fp8shared variant)."""

        shared_w13: torch.Tensor  # e4m3 [2I, H], gate rows then up rows
        shared_w2: torch.Tensor  # e4m3 [H, I]
        shared_w13_scale: torch.Tensor  # UE8M0 [2I, H // 32]
        shared_w2_scale: torch.Tensor  # UE8M0 [H, I // 32]

        def __new__(cls, *args: Any, **kwargs: Any):
            # MoEWeightPack.__new__ has the fixed four-tensor signature and
            # would reject the shared_* keywords; this variant constructs
            # directly and its dataclass __init__ checks the field set.
            return object.__new__(cls)

else:  # pragma: no cover - standalone host tests
    CakePrequantizedMoEWeightsWithSharedExpert = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class TransformedCakeMegaWeights:
    """Kernel-layout rank-local weights (TMA sources owned by this object)."""

    w1_fp4: torch.Tensor  # uint8 [local_E, 2I, H//2], gate/up interleaved by 8 rows
    w1_sfb: (
        torch.Tensor
    )  # uint32 scale vectors over (local_E + 1) K-block-major segments
    w2_fp4: torch.Tensor  # uint8 [local_E, H, I//2]
    w2_sfb: torch.Tensor  # uint32 scale vectors over (local_E + 1) segments
    w1_b8: torch.Tensor  # uint8 e4m3 [2I, H] shared expert (interleaved) or filler
    w2_b8: torch.Tensor  # uint8 e4m3 [H, I] shared expert or filler
    hidden_size: int
    intermediate_size: int
    local_experts: int
    has_shared_expert: bool


def gate_up_interleave_index(intermediate: int) -> torch.Tensor:
    """Physical W1 row ``p`` -> row of ``cat([gate (I rows); up (I rows)])``.

    The kernel consumes the ``2I`` gate|up rows in 16-row groups of
    ``[8 gate rows, 8 up rows]``.  Index-selecting the N axis of a stacked
    ``[gate; up]`` tensor with the returned permutation yields the kernel
    layout; ``torch.argsort`` of it restores the stacked layout.
    """

    n = 2 * intermediate
    p = torch.arange(n)
    group, lane = p // 16, p % 16
    logical = group * 8 + (lane % 8)
    return torch.where(lane < 8, logical, logical + intermediate)


def interleave_gate_up_rows(w13: torch.Tensor, *, axis: int = 1) -> torch.Tensor:
    """Return ``w13`` with its ``2I`` axis permuted into the kernel order."""

    if w13.dim() <= axis or w13.shape[axis] % 16 != 0:
        raise ValueError(
            f"w13 axis {axis} must hold 2I rows with 2I % 16 == 0, got {tuple(w13.shape)}"
        )
    index = gate_up_interleave_index(int(w13.shape[axis]) // 2).to(w13.device)
    return w13.index_select(axis, index).contiguous()


def pack_scales_kblock_major(
    sf: torch.Tensor, *, k_block_elems: int = 32
) -> torch.Tensor:
    """Pack UE8M0 exponent bytes ``[E, N, K//32]`` into the kernel's ``uint32 [E * K//128, N]``.

    Word ``w`` of row ``(e, kb)`` holds K-blocks ``4*kb .. 4*kb+3`` of column
    ``n`` in little-endian byte order (block ``4*kb`` in the low byte).
    """

    if sf.dtype not in SCALE_DTYPES:
        raise TypeError(f"scales must be uint8/e8m0 exponent bytes, got {sf.dtype}")
    if sf.dim() != 3 or sf.shape[2] % 4 != 0:
        raise ValueError(
            f"scales must be [E, N, K//32] with K//32 % 4 == 0, got {tuple(sf.shape)}"
        )
    experts, n, k_blocks = (int(v) for v in sf.shape)
    words = k_blocks // 4
    # [E, N, words, 4] -> [E, words, N, 4] -> uint32 [E * words, N]
    packed = (
        sf.view(torch.uint8)
        .reshape(experts, n, words, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return packed.view(torch.uint32).reshape(experts * words, n)


def unpack_scales_kblock_major(packed: torch.Tensor, *, experts: int) -> torch.Tensor:
    """Inverse of :func:`pack_scales_kblock_major` -> uint8 ``[E, N, K//32]``."""

    rows, n = (int(v) for v in packed.shape)
    words = rows // experts
    bytes_ = packed.view(torch.uint8).reshape(experts, words, n, 4)
    return bytes_.permute(0, 2, 1, 3).reshape(experts, n, words * 4).contiguous()


def _scale_vector_index(device: torch.device) -> torch.Tensor:
    group = torch.arange(8, device=device).reshape(8, 1)
    subtile = torch.arange(SCALE_VECTOR_WORDS, device=device).reshape(
        1, SCALE_VECTOR_WORDS
    )
    return subtile ^ ((group >> 1) * 4)


def scale_vector_layout(packed: torch.Tensor) -> torch.Tensor:
    """K-block-major ``uint32 [R, N]`` scales -> the kernel's scale-vector order.

    Bit-preserving: per row and 128-column tile, the ``[16 subtiles x 8]`` word
    block is transposed to ``[8 x 16]`` and each 16-word vector ``g`` is gathered
    with the XOR swizzle ``nt ^ ((g >> 1) * 4)``.  Returns ``uint32 [R * N // 16, 16]``.
    """

    if packed.dtype not in (torch.uint32, torch.int32) or packed.dim() != 2:
        raise ValueError(
            f"packed scales must be 2-D uint32, got {packed.dtype} {tuple(packed.shape)}"
        )
    columns = int(packed.shape[-1])
    if columns % SCALE_TILE_N:
        raise ValueError(
            f"scale columns {columns} must be a multiple of {SCALE_TILE_N}"
        )
    tiles = (
        packed.view(torch.int32)
        .reshape(-1, columns // SCALE_TILE_N, SCALE_VECTOR_WORDS, 8)
        .transpose(-1, -2)
    )
    index = _scale_vector_index(packed.device).reshape(1, 1, 8, SCALE_VECTOR_WORDS)
    gathered = torch.gather(tiles, -1, index.expand(tiles.shape))
    return gathered.contiguous().reshape(-1, SCALE_VECTOR_WORDS).view(torch.uint32)


def scale_vector_layout_inverse(vectors: torch.Tensor, *, columns: int) -> torch.Tensor:
    """Inverse of :func:`scale_vector_layout` -> K-block-major ``uint32 [R, columns]``."""

    tiles = vectors.view(torch.int32).reshape(
        -1, columns // SCALE_TILE_N, 8, SCALE_VECTOR_WORDS
    )
    index = _scale_vector_index(vectors.device).reshape(1, 1, 8, SCALE_VECTOR_WORDS)
    restored = torch.empty_like(tiles)
    restored.scatter_(-1, index.expand(tiles.shape), tiles)
    return (
        restored.transpose(-1, -2).contiguous().reshape(-1, columns).view(torch.uint32)
    )


def block128_scales_to_k32(
    block_exponents: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    """128 x 128 block UE8M0 exponents ``[ceil(N/128), ceil(K/128)]`` -> per-32 ``[N, K // 32]``.

    Exact: the four per-32 groups of a 128-wide K block share the block's exponent.
    """

    if block_exponents.dtype not in SCALE_DTYPES:
        raise TypeError("block scales must be UE8M0 exponent bytes")
    sf = block_exponents.view(torch.uint8).repeat_interleave(128, dim=0)[:n]
    return sf.repeat_interleave(4, dim=1)[:, : k // 32].contiguous()


_NOSHARED_OPERANDS: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _noshared_operands(
    device: torch.device, H: int, I: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read-only filler shared-expert operands of the routed-only variant (one per device)."""

    key = (str(device), H, I)
    if key not in _NOSHARED_OPERANDS:
        _NOSHARED_OPERANDS[key] = (
            torch.full((2 * I, H), NOSHARED_FP8_FILL, dtype=torch.uint8, device=device),
            torch.full((H, I), NOSHARED_FP8_FILL, dtype=torch.uint8, device=device),
        )
    return _NOSHARED_OPERANDS[key]


def _as_packed_u8(t: torch.Tensor, name: str, dtypes: tuple) -> torch.Tensor:
    if t.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {t.dtype}")
    return t.view(torch.uint8)


def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def transform_prequantized_weights(
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
) -> TransformedCakeMegaWeights:
    """Canonical packed E2M1 + UE8M0 planes (and the optional FP8 shared expert) -> kernel layout."""

    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    H, I = hidden_size, intermediate_size
    w13 = _as_packed_u8(w13, "w13", FP4_PACKED_DTYPES)
    w2 = _as_packed_u8(w2, "w2", FP4_PACKED_DTYPES)
    w13_scale = _as_packed_u8(w13_scale, "w13_scale", SCALE_DTYPES)
    w2_scale = _as_packed_u8(w2_scale, "w2_scale", SCALE_DTYPES)
    local_experts = int(w13.shape[0])
    routed_shapes: tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...] = (
        ("w13", w13, (local_experts, 2 * I, H // 2)),
        ("w2", w2, (local_experts, H, I // 2)),
        ("w13_scale", w13_scale, (local_experts, 2 * I, H // 32)),
        ("w2_scale", w2_scale, (local_experts, H, I // 32)),
    )
    for name, tensor, shape in routed_shapes:
        _check_shape(name, tensor, shape)
    shared = (shared_w13, shared_w2, shared_w13_scale, shared_w2_scale)
    has_shared = shared_w13 is not None
    if any((t is None) != (not has_shared) for t in shared):
        raise ValueError(
            "the shared expert needs all four of shared_w13/w2 and their scales"
        )
    device = w13.device
    w1_sf = pack_scales_kblock_major(interleave_gate_up_rows(w13_scale))
    w2_sf = pack_scales_kblock_major(w2_scale)
    if has_shared:
        s_w13 = _as_packed_u8(shared_w13, "shared_w13", FP8_DTYPES)
        s_w2 = _as_packed_u8(shared_w2, "shared_w2", FP8_DTYPES)
        s_w13_scale = _as_packed_u8(shared_w13_scale, "shared_w13_scale", SCALE_DTYPES)
        s_w2_scale = _as_packed_u8(shared_w2_scale, "shared_w2_scale", SCALE_DTYPES)
        shared_shapes: tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...] = (
            ("shared_w13", s_w13, (2 * I, H)),
            ("shared_w2", s_w2, (H, I)),
            ("shared_w13_scale", s_w13_scale, (2 * I, H // 32)),
            ("shared_w2_scale", s_w2_scale, (H, I // 32)),
        )
        for name, tensor, shape in shared_shapes:
            _check_shape(name, tensor, shape)
        w1_b8 = interleave_gate_up_rows(s_w13, axis=0)
        w2_b8 = s_w2.contiguous()
        w1_shared_sf = pack_scales_kblock_major(
            interleave_gate_up_rows(s_w13_scale, axis=0).unsqueeze(0)
        )
        w2_shared_sf = pack_scales_kblock_major(s_w2_scale.unsqueeze(0))
    else:
        w1_b8, w2_b8 = _noshared_operands(device, H, I)
        filler = torch.tensor(NOSHARED_SCALE_FILL_WORD, dtype=torch.int64).to(
            torch.int32
        )
        w1_shared_sf = (
            filler.expand(H // 128, 2 * I).to(device).contiguous().view(torch.uint32)
        )
        w2_shared_sf = (
            filler.expand(I // 128, H).to(device).contiguous().view(torch.uint32)
        )
    w1_sfb = scale_vector_layout(torch.cat([w1_sf, w1_shared_sf], dim=0))
    w2_sfb = scale_vector_layout(torch.cat([w2_sf, w2_shared_sf], dim=0))
    return TransformedCakeMegaWeights(
        w1_fp4=interleave_gate_up_rows(w13),
        w1_sfb=w1_sfb,
        w2_fp4=w2.contiguous(),
        w2_sfb=w2_sfb,
        w1_b8=w1_b8,
        w2_b8=w2_b8,
        hidden_size=H,
        intermediate_size=I,
        local_experts=local_experts,
        has_shared_expert=has_shared,
    )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
    shared_expert: str | None = None,
) -> TransformedCakeMegaWeights:
    if PrequantizedMoEWeights is None or not isinstance(
        weights, PrequantizedMoEWeights
    ):
        raise NotImplementedError(
            "sm120_mxfp4_mxfp8_bf16_cake requires pre-quantized packed E2M1 weights with both "
            "UE8M0 block-32 scale planes (PrequantizedMoEWeights)"
        )
    shared: dict[str, torch.Tensor] = {}
    if shared_expert is not None:
        if not isinstance(weights, CakePrequantizedMoEWeightsWithSharedExpert):
            raise NotImplementedError(
                "shared_expert='fp8_in_kernel' requires CakePrequantizedMoEWeightsWithSharedExpert "
                "(FP8 e4m3 shared_w13/shared_w2 with per-32 UE8M0 scales)"
            )
        shared = {
            "shared_w13": weights.shared_w13,
            "shared_w2": weights.shared_w2,
            "shared_w13_scale": weights.shared_w13_scale,
            "shared_w2_scale": weights.shared_w2_scale,
        }
    return transform_prequantized_weights(
        weights.w13,
        weights.w2,
        weights.w13_scale,
        weights.w2_scale,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        **shared,
    )


def validate_transformed_mega_weights(
    transformed: TransformedCakeMegaWeights,
    *,
    hidden_size: int,
    intermediate_size: int,
    local_experts: int,
    shared_expert: str | None,
) -> None:
    if not isinstance(transformed, TransformedCakeMegaWeights):
        raise TypeError("transformed weights must be TransformedCakeMegaWeights")
    H, I, L = hidden_size, intermediate_size, local_experts
    expected = (
        ("w1_fp4", transformed.w1_fp4, (L, 2 * I, H // 2), torch.uint8),
        (
            "w1_sfb",
            transformed.w1_sfb,
            ((L + 1) * (H // 128) * 2 * I // 16, 16),
            torch.uint32,
        ),
        ("w2_fp4", transformed.w2_fp4, (L, H, I // 2), torch.uint8),
        (
            "w2_sfb",
            transformed.w2_sfb,
            ((L + 1) * (I // 128) * H // 16, 16),
            torch.uint32,
        ),
        ("w1_b8", transformed.w1_b8, (2 * I, H), torch.uint8),
        ("w2_b8", transformed.w2_b8, (H, I), torch.uint8),
    )
    for name, tensor, shape, dtype in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA {dtype} tensor of shape {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    if (shared_expert is not None) != transformed.has_shared_expert:
        raise ValueError(
            "shared_expert='fp8_in_kernel' requires shared-expert FP8 weights in the transformed "
            "pack, and the routed-only variant must not carry them"
        )


__all__ = [
    "CakePrequantizedMoEWeightsWithSharedExpert",
    "TransformedCakeMegaWeights",
    "block128_scales_to_k32",
    "gate_up_interleave_index",
    "interleave_gate_up_rows",
    "pack_scales_kblock_major",
    "preprocess_mega_weights",
    "scale_vector_layout",
    "scale_vector_layout_inverse",
    "transform_prequantized_weights",
    "unpack_scales_kblock_major",
    "validate_transformed_mega_weights",
]
