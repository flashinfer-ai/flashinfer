# flashinfer/ep/__init__.py
#
# Unified Expert Parallelism API for FlashInfer.
# Abstracts over DeepEP and NCCL-EP via a single interface.
# Supports High-Throughput (prefill/training) and Low-Latency (decode) modes.

from flashinfer.ep.types import (
    Backend,
    OutputLayout,
    EpStatus,
    EpError,
    TensorTag,
)
from flashinfer.ep.group import (
    EpGroup,
    create_group,
    get_buffer_size_hint,
    get_low_latency_rdma_size_hint,
    get_dispatch_config,
    get_combine_config,
    KernelConfig,
    DispatchResult,
    CombineResult,
    LayoutInfo,
    EpHandle,
    RoutingCache,
    StreamDep,
)

__all__ = [
    # Enums
    "Backend",
    "OutputLayout",
    "TensorTag",
    # Status
    "EpStatus",
    "EpError",
    # Group lifecycle
    "create_group",
    "EpGroup",
    # Buffer sizing
    "get_buffer_size_hint",
    "get_low_latency_rdma_size_hint",
    "get_dispatch_config",
    "get_combine_config",
    "KernelConfig",
    # Result types
    "DispatchResult",
    "CombineResult",
    "LayoutInfo",
    # Handle types
    "EpHandle",
    "RoutingCache",
    "StreamDep",
]
