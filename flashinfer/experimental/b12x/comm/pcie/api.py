"""Public surface for comm.pcie (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import is_b12x
from .pcie_allreduce import (
    PCIeAllReduce as AllReduce,
)
from .pcie_dcp_a2a import (
    PCIeDCPA2A as DcpAllToAll,
)
from .pcie_dcp_a2a import (
    PCIeDCPA2APool as DcpAllToAllPool,
)
from .pcie_dcp_a2a import (
    kimi_topk16,
    lse_reduce_scatter_reference,
)
from .pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange as DcpTopKOwnerExchange,
    owner_stage_reference,
)
from .pcie_dma import (
    PCIeDmaAllReduce as DmaAllReduce,
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
from .pcie_twoshot_bf16 import PCIeTwoShotBF16
from .pcie_twoshot import (
    PCIeTwoShotSP as TwoShotReduceScatter,
)
from .pcie_vocab_argmax import (
    PCIeVocabParallelArgmax as VocabParallelArgmax,
)
from ._preparation import plan, query_from_runtime
from ._tuning import PcieConfig, PcieQuery
from b12x.preparation import Plan


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with >= 2 visible CUDA devices.

    Device programs are compiled and primed by PreparationSession before use;
    no repo-authored C++/CUDA extension or runtime nvcc build is involved.
    """
    import torch

    return is_b12x(device) and torch.cuda.device_count() >= 2


__all__ = [
    "Plan",
    "PcieConfig",
    "PcieQuery",
    "plan",
    "query_from_runtime",
    "AllReduce",
    "OneshotAllReduce",
    "OneshotAllReducePool",
    "DmaAllReduce",
    "PCIeTwoShotBF16",
    "TwoShotReduceScatter",
    "DcpAllToAll",
    "DcpAllToAllPool",
    "DcpTopKOwnerExchange",
    "VocabParallelArgmax",
    "kimi_topk16",
    "parse_oneshot_max_size",
    "lse_reduce_scatter_reference",
    "owner_stage_reference",
    "is_supported",
]
