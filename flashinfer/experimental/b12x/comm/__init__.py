"""Communication ops for b12x.

- ``pcie``: collectives for consumer PCIe fabrics (no NVLink) — one-shot and
  DMA/CE-ring all-reduce, FP8-transport two-shot reduce-scatter, and the DCP
  attention all-to-all with fused LSE merge.
- ``roce`` (RoCEnante): one-shot RDMA all-reduce and all-gather for tensor
  parallelism across DGX Spark nodes over their ConnectX-7 RoCE ports.
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("pcie", "roce")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
