# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile/launch entry of the bf16 MoE kernels: the compiled callables of the grouped GEMM, the routing, permutation
and combine kernels, and the host-side grid helpers. The cache module ``batch_invariant_moe`` is rooted at the kernel
files themselves (``flashinfer.batch_invariant.jit.MODULE_ROOTS``), so this module and the API stay outside its key.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from .blocks.compile import (  # noqa: F401
    GRID_PLACEHOLDER,
    can_implement_grouped,
    get_compiled_grouped,
)
from .combine import get_compiled_combine, warps_per_token  # noqa: F401
from .permute import (  # noqa: F401
    PLAN_BLOCK,
    get_compiled_gather,
    get_compiled_histogram,
    get_compiled_plan,
    get_compiled_plan_multi,
    get_compiled_positions,
)
from .route import get_compiled_route  # noqa: F401
from .route_plan import get_compiled_route_plan  # noqa: F401


def _device(device: Optional[torch.device]) -> torch.device:
    return (
        torch.device("cuda", torch.cuda.current_device()) if device is None else device
    )


def sm_count(device: Optional[torch.device] = None) -> int:
    """The SM count of ``device`` (the current device by default)."""
    return sm_count_of(_device(device))


def max_active_clusters(cfg: Knobs, device: Optional[torch.device] = None) -> int:
    """The device's co-resident cluster capacity for the record's cluster size (the GEMM class's occupancy query,
    ``api.persistent_clusters``), ignoring the record's own ``grid_clusters``: the grid of the default launch."""
    return persistent_clusters(
        _device(device), dataclasses.replace(cfg, grid_clusters=0)
    )


def grid_clusters(cfg: Knobs, device: Optional[torch.device] = None) -> int:
    """The grid argument of a grouped launch: the record's own cap when it carries one, else the device's capacity. A
    launch parameter (which cluster computes which tile), never the bits."""
    return cfg.grid_clusters or max_active_clusters(cfg, device)
