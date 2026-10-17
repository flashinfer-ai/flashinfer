# flashinfer/ep/types.py
#
# Core type definitions for the unified EP API.

from enum import Enum
from dataclasses import dataclass, field


class Backend(Enum):
    """Supported EP backends."""
    DEEP_EP = "deepep"
    NCCL_EP = "nccl_ep"
    NIXL_EP = "nixl_ep"


class OutputLayout(Enum):
    """Output tensor layout for dispatch results.

    Issue #1: Normalizes backend-specific tensor layouts.
    - FLAT_2D:        [total_recv_tokens, hidden_dim] — universal default
    - EXPERT_MAJOR_3D: [num_local_experts, max_tok, hidden] — grouped GEMM optimized
    """
    FLAT_2D = "flat_2d"
    EXPERT_MAJOR_3D = "expert_major_3d"


class TensorTag(Enum):
    """Semantic tags for NCCL-EP tensor descriptors. Internal use."""
    TOKENS = 0
    TOPK_IDX = 1
    TOPK_WEIGHTS = 2
    SCALES = 3
    EXPERT_COUNT_DEVICE = 4
    EXPERT_COUNT_HOST = 5


@dataclass
class EpStatus:
    """Status of an EP operation.

    Issue #22: Lightweight on success — error_code == 0, no string alloc.
    Error details (error_msg, failed_ranks) populated only on failure.
    """
    error_code: int = 0
    error_msg: str = ""
    failed_ranks: list = field(default_factory=list)

    def ok(self) -> bool:
        """True if the operation succeeded."""
        return self.error_code == 0

    def raise_if_error(self):
        """Raise RuntimeError if the operation failed."""
        if self.error_code != 0:
            raise RuntimeError(f"EP operation failed: {self.error_msg}")


class EpError(RuntimeError):
    """Raised by EP operations on failure."""
    def __init__(self, status: EpStatus):
        super().__init__(status.error_msg)
        self.status = status
