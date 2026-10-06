"""Lookup tables for the ``lut_e4m3`` trellis codebook.

``lut_e4m3`` serves 2-, 3- and 4-bit trellis weights. A fixed integer
permutation maps each 16-bit decode window to a rank in ``[0, 65536)``, and a
monotone, sign-symmetric value law maps the rank to an E4M3 byte. Device
decoders read the 4 KiB value table, which holds one byte for every sixteen
consecutive ranks. The direct tables precompose the permutation with the value
table for each bit width. None of these process-global tables is checkpoint
state.
"""

from __future__ import annotations

import functools

import torch


LUT_E4M3_DIRECT_TABLE_ENTRIES = 3 * (1 << 16)

# Positive-half rank thresholds of the value law. The E4M3 magnitude code
# increments at each listed rank; negative ranks mirror the positive half.
_E4M3_POSITIVE_TRANSITIONS = (
    17,
    51,
    85,
    119,
    153,
    187,
    221,
    255,
    289,
    323,
    357,
    391,
    426,
    460,
    494,
    528,
    579,
    647,
    715,
    783,
    851,
    919,
    987,
    1055,
    1157,
    1293,
    1429,
    1565,
    1701,
    1837,
    1973,
    2108,
    2312,
    2583,
    2854,
    3124,
    3395,
    3665,
    3934,
    4203,
    4606,
    5141,
    5674,
    6205,
    6732,
    7258,
    7779,
    8298,
    9070,
    10085,
    11084,
    12065,
    13026,
    13967,
    14885,
    15781,
    17081,
    18725,
    20265,
    21696,
    23017,
    24229,
    25332,
    26330,
    27637,
    29054,
    30143,
    30957,
    31548,
    31967,
    32255,
    32447,
    32617,
    32717,
    32753,
    32764,
    32767,
)


def lut_e4m3_decode_ranks_torch(ranks: torch.Tensor) -> torch.Tensor:
    """Map ranks to E4M3 bytes through the value law."""

    rank = ranks.to(torch.int64) & 0xFFFF
    negative = rank < 0x8000
    positive_offset = rank & 0x7FFF
    positive_offset = torch.where(negative, positive_offset ^ 0x7FFF, positive_offset)
    transitions = torch.tensor(
        _E4M3_POSITIVE_TRANSITIONS,
        dtype=torch.int64,
        device=positive_offset.device,
    )
    magnitude = torch.bucketize(positive_offset, transitions, right=True)
    sign = negative.to(torch.int64) << 7
    sign = torch.where(magnitude == 0, torch.zeros_like(sign), sign)
    return (magnitude | sign).to(torch.uint8)


@functools.cache
def lut_e4m3_value_table_cpu() -> torch.Tensor:
    """Build the 4 KiB value table.

    Each entry holds the most frequent E4M3 byte among sixteen consecutive
    ranks; ties take the lower byte.
    """

    ranks = torch.arange(1 << 16, dtype=torch.int64)
    codes = lut_e4m3_decode_ranks_torch(ranks).reshape(1 << 12, 16)
    result = torch.empty(1 << 12, dtype=torch.uint8)
    for index, block in enumerate(codes):
        labels, counts = torch.unique(block, return_counts=True)
        result[index] = labels[counts == counts.max()].min()
    return result.contiguous()


_LUT_E4M3_VALUE_TABLE_DEVICE: dict[tuple[str, int | None], torch.Tensor] = {}


def _lut_e4m3_value_table_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    key = (device_type, device_index)
    cached = _LUT_E4M3_VALUE_TABLE_DEVICE.get(key)
    if cached is None:
        device = torch.device(device_type, device_index)
        cached = lut_e4m3_value_table_cpu().to(device=device).contiguous()
        _LUT_E4M3_VALUE_TABLE_DEVICE[key] = cached
    return cached


def lut_e4m3_value_table(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime 4 KiB value table for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _lut_e4m3_value_table_device(resolved.type, index)


def lut_e4m3_value_table_resident(device: torch.device | str) -> torch.Tensor | None:
    """Return the resident device table without materializing a missing entry."""
    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _LUT_E4M3_VALUE_TABLE_DEVICE.get((resolved.type, index))


@functools.cache
def _lut_e4m3_direct_table_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return lut_e4m3_direct_table_cpu().to(device=device).contiguous()


def lut_e4m3_direct_table(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime rate-indexed 192 KiB direct table.

    Rows are the 2-, 3- and 4-bit slices in rate order: byte(window, bits) =
    table[((bits - 2) << 16) | window]. Each byte precomposes the permutation
    with the value table, so lookups are bit-identical to the in-kernel
    decode.
    """

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _lut_e4m3_direct_table_device(resolved.type, index)


def _lut_e4m3_permutation(codewords: torch.Tensor, bits: int) -> torch.Tensor:
    """Map 16-bit decode windows to value-law ranks."""

    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported lut_e4m3 bit width {bits}")
    width = 16 - bits
    history_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    codeword = codewords.to(torch.int64) & 0xFFFF
    history = codeword >> bits
    branch = codeword & branch_mask

    mixed = history ^ (history >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    low = product & history_mask
    branch_key = product >> (32 - bits)

    reversed_branch = torch.zeros_like(branch)
    for index in range(bits):
        reversed_branch |= ((branch >> index) & 1) << (bits - 1 - index)
    high = reversed_branch ^ branch_key
    return (high << width) | low


@functools.cache
def lut_e4m3_direct_table_cpu() -> torch.Tensor:
    """Build independent 2-, 3- and 4-bit codeword tables (192 KiB total)."""

    codewords = torch.arange(1 << 16, dtype=torch.int64)
    values = lut_e4m3_value_table_cpu()
    return torch.cat(
        [
            values[_lut_e4m3_permutation(codewords, bits) >> 4]
            for bits in (2, 3, 4)
        ]
    ).contiguous()


__all__ = [
    "LUT_E4M3_DIRECT_TABLE_ENTRIES",
    "lut_e4m3_decode_ranks_torch",
    "lut_e4m3_direct_table",
    "lut_e4m3_direct_table_cpu",
    "lut_e4m3_value_table",
    "lut_e4m3_value_table_cpu",
    "lut_e4m3_value_table_resident",
]
