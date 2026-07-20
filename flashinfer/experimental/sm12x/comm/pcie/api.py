# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for comm.pcie (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import is_sm12x
from .pcie_dcp_a2a import (
    PCIeDCPA2A as DcpAllToAll,
)
from .pcie_dcp_a2a import (
    PCIeDCPA2APool as DcpAllToAllPool,
)
from .pcie_dcp_a2a import (
    lse_reduce_scatter_reference,
)
from .pcie_dma import (
    PCIeDmaAllReduce as DmaAllReduce,
)
from .pcie_dma import (
    autotune_crossovers as autotune_dma_crossovers,
)
from .pcie_oneshot import (
    PCIeOneshotAllReduce as OneshotAllReduce,
)
from .pcie_oneshot import (
    PCIeOneshotAllReducePool as OneshotAllReducePool,
)
from .pcie_oneshot import (
    parse_pcie_oneshot_max_size as parse_oneshot_max_size,
)
from .pcie_twoshot import (
    PCIeTwoShotSP as TwoShotReduceScatter,
)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with >= 2 visible CUDA devices.

    Raw CUDA op: unlike the CuTe-DSL ops, this does not require
    nvidia-cutlass-dsl (it needs nvcc at runtime instead; see META.notes).
    """
    import torch

    return is_sm12x(device) and torch.cuda.device_count() >= 2


__all__ = [
    "OneshotAllReduce",
    "OneshotAllReducePool",
    "DmaAllReduce",
    "TwoShotReduceScatter",
    "DcpAllToAll",
    "DcpAllToAllPool",
    "autotune_dma_crossovers",
    "parse_oneshot_max_size",
    "lse_reduce_scatter_reference",
    "is_supported",
]
