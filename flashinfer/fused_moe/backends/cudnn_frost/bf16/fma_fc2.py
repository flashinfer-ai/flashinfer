# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; FMA device code is owned by frost_low_latency."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module(
    "flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc2"
)
