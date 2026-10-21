# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Admission for independent and persistent ping-pong NVFP4 kernels."""

from typing import NamedTuple

VERSION = "sm12x_cute_nvfp4_v5"
SMALL = ("independent",)
TMA = ("independent_tma",)


class RawConfig(NamedTuple):
    """Complete code-generation choice for the persistent ping-pong kernel."""

    epi_m: int
    epi_n: int
    swizzle: int
    elected_release: bool
    raster_along_m: bool
    tile_k: int = 128
    internal_swap: bool = False
    extra_mainloop_stage: bool = False
    half_stage_wait: bool = False
    register_redistribution: bool = False


def _raw(*fields, **flags):
    return ("raw", *RawConfig(*fields, **flags))


def raw_config(tactic):
    if tactic[0] != "raw" or len(tactic) != len(RawConfig._fields) + 1:
        raise ValueError("Invalid SM12x raw tactic")
    return RawConfig(*tactic[1:])


RAW_TACTICS = (
    _raw(64, 32, 8, False, True),
    _raw(32, 64, 13, True, True),
)
_WIDE_K256 = _raw(64, 32, 8, False, True, 256, True)
_SWAP_K256 = _raw(32, 64, 13, True, True, 256, False)
_SM121_RAW = {
    (256, 7168, 256): _SWAP_K256,
    (256, 7168, 512): _SWAP_K256,
    (256, 9216, 7168): _raw(
        64, 32, 2, False, True, 256, True, register_redistribution=True
    ),
    (512, 7168, 5120): _raw(
        64, 32, 8, False, True, 256, False, register_redistribution=True
    ),
    (512, 8192, 2048): _raw(
        64, 32, 4, False, True, 256, True, register_redistribution=True
    ),
    (1024, 5120, 17408): _WIDE_K256,
    (1024, 7168, 4608): _WIDE_K256,
    (2048, 5120, 17408): _WIDE_K256,
    (4096, 5120, 17408): _raw(64, 32, 8, False, True, extra_mainloop_stage=True),
    (8192, 5120, 17408): _raw(
        64, 32, 8, False, True, extra_mainloop_stage=True, half_stage_wait=True
    ),
    (8192, 34816, 5120): _SWAP_K256,
}


def check_shape(m, n, k):
    if min(m, n, k) <= 0 or n % 64 or k % 64:
        raise ValueError("SM12x cute-dsl requires M > 0, N % 64 = 0, K % 64 = 0")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("SM12x cute-dsl requires element offsets below 2**31")


def valid_tactics(m, n, k, *, compute_capability=None):
    check_shape(m, n, k)
    choices = (SMALL, TMA) if m > 32 else (SMALL,)
    if m % 128 == 0 and n % 128 == 0 and k % 256 == 0:
        choices += RAW_TACTICS
        preferred = _SM121_RAW.get((m, n, k))
        if compute_capability == (12, 1) and preferred is not None:
            choices += (preferred,)
    return choices


def compatible(m, n, k, tactic, *, compute_capability=None):
    try:
        choices = valid_tactics(m, n, k, compute_capability=compute_capability)
    except ValueError:
        return False
    return tactic is None or tactic == -1 or tactic in choices


def default_tactic(m, n, k, *, compute_capability=None):
    choices = valid_tactics(m, n, k, compute_capability=compute_capability)
    if len(choices) > 2:
        if compute_capability == (12, 1) and (m, n, k) in _SM121_RAW:
            return _SM121_RAW[m, n, k]
        return RAW_TACTICS[int(n >= k)]
    return SMALL if m <= 64 else TMA
