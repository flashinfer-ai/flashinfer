# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Explicit replay boundary for distributed RDMA MoE AlltoAll.

Dispatch and combine require a live, collectively initialized communicator,
matching calls on every rank, and owner-bound dispatch metadata. A local tensor
does not describe their remote inputs or synchronization state. There is no
single-rank reference or initializer for these APIs, including their planned
reduction mode.

Returning ``None`` uses the standard trace-dispatch convention: ``fi_trace``
returns an empty dict and writes no replay JSON. In particular, these hooks do
not substitute an identity operation or initialize CUDA/NCCL while tracing.
"""


def rdma_moe_dispatch_trace(save_dir=None, name=None, **kwargs):
    """Decline local replay of collective dispatch, including receive hooks."""
    return None


def rdma_moe_combine_trace(save_dir=None, name=None, **kwargs):
    """Decline local replay of combine and its owner-bound dispatch handle."""
    return None


# No local replay variant is currently available for registry enumeration.
rdma_moe_dispatch_trace.templates = ()  # type: ignore[attr-defined]
rdma_moe_combine_trace.templates = ()  # type: ignore[attr-defined]
