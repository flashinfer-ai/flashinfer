# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run this file in a fresh eight-process invocation with CUDA_MODULE_LOADING=LAZY."""

import os

import pytest

from ._moe_case import MoECase
from ._attention_case import AttentionCase

pytestmark = pytest.mark.node_exclusive


def test_first_exchange_is_graph_replay_after_warmup(collective, collective_watchdog):
    if os.environ.get("CUDA_MODULE_LOADING", "").upper() != "LAZY":
        pytest.skip("this test requires a fresh process with CUDA_MODULE_LOADING=LAZY")
    case = MoECase(collective)
    try:
        case.capture()  # no eager MoE exchange has run
        case.run(4, 1, graph=True)
        case.run(1, 7, graph=True)
        case.run(4, 23, graph=True)
    finally:
        case.close()


def test_lazy_first_attention_exchange_is_graph_replay(collective, collective_watchdog):
    if os.environ.get("CUDA_MODULE_LOADING", "").upper() != "LAZY":
        pytest.skip("run this file in fresh workers with CUDA_MODULE_LOADING=LAZY")
    case = AttentionCase(collective)
    try:
        case.capture()
        case.run(4, "full", 2048, 1, graph=True)
        case.run(1, "shared", 4096, 7, graph=True)
        case.run(4, "shared", 31, 23, graph=True)
    finally:
        case.close()
