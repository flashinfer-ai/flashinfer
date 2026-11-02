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

The NVFP4 paged-cache format of ``nvfp4_quantize_append_paged_kv_cache_with_slot_mapping``
in plain torch (the writer's recipe, ``include/flashinfer/page.cuh``
``nvfp4_append_quantize_block``): per 16 dims the E4M3 block scale
``sf = e4m3(amax / (6 * global_scale))`` and the E2M1 codes
``e2m1(value / (sf * global_scale))``, packed two to a byte with the even dim in the
low nibble.  Shared by ``tests/experimental/test_cake_qsa_nvfp4.py`` (whose backend
suite imports nothing of flashinfer but the package) and
``tests/experimental/test_cake_qsa_nvfp4_api.py`` (which compares it with the writer).

The writer runs under nvcc's ``-use_fast_math`` (approximate FP32 division); this port
divides in IEEE FP32, so a value that lands within rounding of a ladder midpoint or
of an E4M3 rounding boundary may take the neighbouring code.  Such cases are isolated;
every other byte is the writer's.
"""

import torch

HEAD_DIM = 256
BLOCK = 16  # dims per E4M3 block scale
PACKED_BYTES = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // BLOCK
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# The writer's ladder: a magnitude at a midpoint takes the even code (``!(mag > 0.25) -> 0``,
# ``< 0.75 -> 1``, ``<= 1.25 -> 2``, ``< 1.75 -> 3``, ``<= 2.5 -> 4``, ``< 3.5 -> 5``, ``<= 5 -> 6``).
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MAX = 448.0


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """The 4-bit E2M1 code (sign in bit 3) of FP32 ``x``: the nearest of ``E2M1_GRID`` in
    magnitude, a midpoint rounding to the even code."""
    bounds = torch.tensor(E2M1_MIDPOINTS, device=x.device)
    magnitude = x.abs().unsqueeze(-1)
    above = (magnitude > bounds).sum(-1)
    at_or_above = (magnitude >= bounds).sum(-1)
    midpoint = above != at_or_above
    even = torch.where(at_or_above % 2 == 0, at_or_above, above)
    code = torch.where(midpoint, even, above)
    return (code | (torch.signbit(x).to(code.dtype) << 3)).to(torch.uint8)


def quantize_rows(
    x: torch.Tensor, global_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x[..., 256]`` (any float dtype) as the writer does with the NVFP4 global
    scale ``global_scale``: the packed codes ``[..., 128]`` (uint8) and the block scale
    bytes ``[..., 16]`` (float8_e4m3fn)."""
    groups = x.float().reshape(*x.shape[:-1], SCALE_BYTES, BLOCK)
    amax = groups.abs().amax(-1, keepdim=True)
    scale = torch.tensor(float(global_scale), dtype=torch.float32, device=x.device)
    sf_value = torch.where(
        (amax > 0) & (scale > 0), amax / (6.0 * scale), torch.zeros_like(amax)
    )
    sf = sf_value.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    sf_rounded = sf.float()
    output_scale = torch.where(
        (amax > 0) & (sf_rounded > 0) & (scale > 0),
        1.0 / (sf_rounded * scale),
        torch.zeros_like(amax),
    )
    codes = e2m1_encode(groups * output_scale)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return (
        packed.reshape(*x.shape[:-1], PACKED_BYTES),
        sf.reshape(*x.shape[:-1], SCALE_BYTES),
    )


def write_slots(
    k: torch.Tensor,
    v: torch.Tensor,
    data: tuple[torch.Tensor, torch.Tensor],
    sf: tuple[torch.Tensor, torch.Tensor],
    k_scale: float,
    v_scale: float,
    kv_layout: str,
) -> None:
    """Quantize ``k`` / ``v`` ``[slots, num_kv_heads, 256]`` into every slot of the cache
    views ``data`` (uint8 ``[blocks, page, heads, 128]`` for NHD, ``[blocks, heads, page, 128]``
    for HND) and ``sf`` (float8_e4m3fn ``[..., 16]``), slot ``s`` at block ``s // page``,
    token ``s % page``: the writer with the identity slot mapping."""
    for side, scale, out, out_sf in zip(
        (k, v), (k_scale, v_scale), data, sf, strict=True
    ):
        packed, scales = quantize_rows(side, scale)  # [slots, heads, *]
        page = out.shape[2] if kv_layout == "HND" else out.shape[1]
        packed = packed.view(-1, page, *packed.shape[1:])  # [blocks, page, heads, 128]
        scales = scales.view(-1, page, *scales.shape[1:])
        if kv_layout == "HND":
            packed, scales = packed.transpose(1, 2), scales.transpose(1, 2)
        out.copy_(packed)
        out_sf.copy_(scales)


__all__ = [
    "BLOCK",
    "E2M1_GRID",
    "E2M1_MIDPOINTS",
    "E4M3_MAX",
    "HEAD_DIM",
    "PACKED_BYTES",
    "SCALE_BYTES",
    "e2m1_encode",
    "quantize_rows",
    "write_slots",
]
