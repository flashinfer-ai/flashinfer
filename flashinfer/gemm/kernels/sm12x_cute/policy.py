# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Admission for independent and persistent ping-pong NVFP4 kernels."""

VERSION = "sm12x_cute_nvfp4_v3"
SMALL = ("independent",)
TMA = ("independent_tma",)
RAW_TACTICS = (
    ("raw", 64, 32, 8, False, True),
    ("raw", 32, 64, 13, True, True),
)
_SM121_RAW = {
    (256, 7168, 256): ("raw", 32, 64, 13, True, True, 256, False),
    (256, 7168, 512): ("raw", 32, 64, 13, True, True, 256, False),
    (256, 9216, 7168): ("raw", 64, 32, 2, False, True, 256, True),
    (512, 7168, 5120): ("raw", 64, 32, 8, False, True, 256, False),
    (512, 8192, 2048): ("raw", 64, 32, 4, False, True, 256, True),
    (1024, 5120, 17408): ("raw", 64, 32, 8, False, True, 256, True),
    (1024, 7168, 4608): ("raw", 64, 32, 8, False, True, 256, True),
    (2048, 5120, 17408): ("raw", 64, 32, 8, False, True, 256, True),
    (8192, 34816, 5120): ("raw", 32, 64, 13, True, True, 256, False),
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
