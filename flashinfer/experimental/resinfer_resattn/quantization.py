# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer JIT binding for the ResInfer grouped Q producer."""

import functools

from ...jit.resinfer import gen_resinfer_grouped_q_module


@functools.cache
def _module():
    return gen_resinfer_grouped_q_module().build_and_load()


def quantize_grouped_q(q, packed, scales):
    _module().quantize(q, packed, scales)
