# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Resolve Blackwell's native graph timing and explicit v2 policies."""

from flashinfer.moe_ep.core.tuning_measurement import measurement_settings as _settings


def measurement_settings(context, calls_per_sample=8, warmup_iters=8):
    return _settings(
        context,
        native_graph_calls=calls_per_sample,
        native_warmups=warmup_iters,
    )
