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

Backend of ``mm_nvfp4_svdquant(..., backend="cake")``: the Cake NVFP4 SVDQuant
GEMM programs (SM100 / SM103) as one self-contained package (the host API
bodies, the ``cake_jit`` registry and loader glue, and the generated CUDA
sources under ``csrc/``).  The stable, ``@flashinfer_api``-decorated entry
point and the user-facing documentation live in
:mod:`flashinfer.gemm.gemm_svdquant`; import that module.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    BLOCK_M,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MAX_RANK,
    MODULES,
    RANK_GRANULARITY,
    ROUTE_TABLE,
    arch_for_capability,
    launch_grid,
    plan_values,
    select_record,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def is_supported(
    *, m: int, n: int, k: int, rank: int, has_bias: bool, device: torch.device
) -> bool:
    """Whether ``device`` is an exported target and ``(M, N, K, rank, has_bias)``
    has a route on its architecture and SM count (the shape rule of
    :func:`cake_jit.select_record`)."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        index = _device_index(device)
        select_record(
            m=m,
            n=n,
            k=k,
            rank=rank,
            has_bias=has_bias,
            arch=_loader.device_arch(index),
            sm_count=_loader.sm_count(index),
        )
        return True
    except (RuntimeError, ValueError):
        return False


def run(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Launch the routed program for one problem and return ``out``.

    The route is :func:`cake_jit.select_record` over the host scalars and the
    device's architecture and SM count (both read once per device); the
    arguments are :func:`cake_jit.plan_values`
    (scale-factor views, bias carrier, kernel scalars, grid).  ``enable_pdl``
    must be ``None`` or equal to the selected program's launch attribute; a
    program's programmatic dependent launch is a property of its rendered
    binding, not a runtime toggle.
    """
    m, k_packed = (int(value) for value in a.shape)
    n = int(b.shape[0])
    k = 2 * k_packed
    rank = int(d.shape[1])
    has_bias = bias is not None
    device_index = _device_index(a.device)
    name = select_record(
        m=m,
        n=n,
        k=k,
        rank=rank,
        has_bias=has_bias,
        arch=_loader.device_arch(device_index),
        sm_count=_loader.sm_count(device_index),
    )
    record = MODULES[name]
    program_pdl = bool(record["launch"]["use_pdl"])
    if enable_pdl is not None and bool(enable_pdl) is not program_pdl:
        raise ValueError(
            f"the selected Cake program requires enable_pdl={program_pdl}, got {enable_pdl}"
        )
    values = plan_values(
        record,
        a=a,
        b=b,
        a_sf=a_sf,
        b_sf=b_sf,
        alpha=alpha,
        d=d,
        l1=l1,
        bias=bias,
        out=out,
    )
    _loader.run_module(name, values, grid=values["grid"], device_index=device_index)
    return out


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BLOCK_M",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_RANK",
    "MODULES",
    "RANK_GRANULARITY",
    "ROUTE_TABLE",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "arch_for_capability",
    "is_supported",
    "launch_grid",
    "plan_values",
    "run",
    "select_record",
]
