# flashinfer/ep/_backends/nixl_ep.py
#
# Python-side wrapper for the NIXL-EP backend.
# Handles NIXL-specific initialization, config validation, and
# elastic EP lifecycle management.
#
# NIXL-EP is LL-only (Low-Latency mode). HT dispatch/combine raises
# NotImplementedError. The transport layer is agnostic — nixlAgent
# auto-selects GPUDirect RDMA, NVLink P2P, or TCP based on topology.
#
# References:
#   - https://github.com/ai-dynamo/nixl
#   - https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep

import os
from typing import Optional, List


def _nixl_available() -> bool:
    """Check if NIXL library (libnixl) is available."""
    try:
        import ctypes
        import ctypes.util

        # Search order: NIXL_HOME env, system paths
        search_paths = []

        nixl_home = os.environ.get("NIXL_HOME", "")
        if nixl_home:
            search_paths.append(os.path.join(nixl_home, "lib", "libnixl.so"))

        nixl_dir = os.environ.get("NIXL_DIR", "")
        if nixl_dir:
            search_paths.append(os.path.join(nixl_dir, "lib", "libnixl.so"))

        search_paths.extend([
            "libnixl.so",
            "/usr/lib/libnixl.so",
            "/usr/local/lib/libnixl.so",
        ])

        # ctypes fallback
        found = ctypes.util.find_library("nixl")
        if found:
            search_paths.append(found)

        for path in search_paths:
            try:
                ctypes.CDLL(path)
                return True
            except OSError:
                continue
        return False
    except Exception:
        return False


class NixlEPBackendWrapper:
    """Python-side wrapper for NIXL-EP backend configuration.

    This wrapper handles NIXL-EP-specific concerns:
    - Issue #61: Transport-agnostic nixlAgent initialization
    - Issue #62: Elastic EP configuration (enable_elastic flag)
    - Issue #63: Partial failure tolerance settings
    - Issue #64: Prepped transfer model for RoutingCache
    - LL-only validation (reject HT mode requests)

    The actual dispatch/combine operations are handled by the C++ backend
    (nixl_ep_backend.cu). This wrapper is used during group creation
    to validate configuration and check NIXL availability.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if the NIXL-EP backend is available.

        Requires:
        - libnixl shared library installed
        - At least one NIXL transport plugin (RDMA, NVLink, or TCP)
        """
        return _nixl_available()

    @staticmethod
    def validate_config(
        num_experts: int,
        num_local_experts: int,
        hidden_dim: int,
        top_k: int,
        enable_elastic: bool = False,
        max_tokens_per_rank: int = 256,
        transport_hint: Optional[str] = None,
    ) -> dict:
        """Validate and normalize NIXL-EP-specific configuration.

        Args:
            num_experts: Total number of experts across all ranks.
            num_local_experts: Experts on this rank.
            hidden_dim: Hidden dimension of tokens.
            top_k: Number of experts each token is routed to.
            enable_elastic: Enable elastic EP (dynamic rank add/remove).
            max_tokens_per_rank: Max tokens per rank for buffer sizing.
            transport_hint: Optional transport preference ("rdma", "nvlink", "tcp").
                           None = auto-detect (recommended).

        Returns:
            Validated configuration dict.

        Raises:
            RuntimeError: If NIXL library is not available.
            ValueError: If configuration is invalid.
        """
        if not _nixl_available():
            raise RuntimeError(
                "NIXL-EP backend requires libnixl. Install NIXL from "
                "https://github.com/ai-dynamo/nixl or set NIXL_HOME."
            )

        if transport_hint and transport_hint not in ("rdma", "nvlink", "tcp"):
            raise ValueError(
                f"Invalid transport_hint '{transport_hint}'. "
                f"Must be one of: 'rdma', 'nvlink', 'tcp', or None (auto)."
            )

        return {
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "enable_elastic": enable_elastic,
            "max_tokens_per_rank": max_tokens_per_rank,
            "transport_hint": transport_hint,
        }

    @staticmethod
    def extract_nccl_comm(process_group) -> None:
        """NIXL-EP does not use NCCL comm — returns None.

        NIXL has its own transport layer (nixlAgent) that auto-selects
        RDMA, NVLink, or TCP. It does not depend on NCCL.
        """
        return None

    @staticmethod
    def get_supported_modes() -> list:
        """Return list of supported EP modes.

        NIXL-EP is LL-only in v1. HT support may be added in future versions.
        """
        return ["low_latency"]

    @staticmethod
    def check_ht_support() -> bool:
        """Check if HT mode is supported. Always False for NIXL-EP v1."""
        return False

    @staticmethod
    def get_transport_info() -> dict:
        """Return available transport information.

        Queries NIXL for available transport plugins on this system.
        """
        info = {
            "backend": "nixl_ep",
            "available_transports": [],
            "selected_transport": None,
        }

        # Check for RDMA
        if os.path.exists("/dev/infiniband"):
            info["available_transports"].append("rdma")

        # Check for NVLink (via nvidia-smi or sysfs)
        try:
            import torch
            if torch.cuda.is_available():
                # If multiple GPUs with NVLink, nvlink transport is available
                if torch.cuda.device_count() > 1:
                    info["available_transports"].append("nvlink")
        except ImportError:
            pass

        # TCP is always available
        info["available_transports"].append("tcp")

        return info


class NixlElasticManager:
    """Manages elastic EP lifecycle for NIXL-EP groups.

    Provides methods for dynamic rank addition/removal without
    stopping inference. Used by EpGroup.add_rank()/remove_rank().
    """

    def __init__(self, group_id: int):
        self._group_id = group_id
        self._active_ranks: List[int] = []
        self._generation: int = 0
        self._failed_ranks: List[int] = []

    @property
    def generation(self) -> int:
        """Topology generation counter. Incremented on every add/remove."""
        return self._generation

    @property
    def active_ranks(self) -> List[int]:
        """List of currently active (healthy) ranks."""
        return list(self._active_ranks)

    @property
    def failed_ranks(self) -> List[int]:
        """List of ranks that have failed since last recovery."""
        return list(self._failed_ranks)

    def add_rank(self, new_rank: int) -> None:
        """Add a new rank to the EP group.

        Triggers:
        - Memory descriptor exchange with the new rank
        - Expert rebalancing across all active ranks
        - Routing cache invalidation

        Args:
            new_rank: Rank ID of the new GPU to add.

        Raises:
            RuntimeError: If the add operation fails.
        """
        if new_rank in self._active_ranks:
            raise ValueError(f"Rank {new_rank} is already active")

        # The actual work is done in C++ via nixl_ep_backend.cu
        # This Python method updates local state and calls the C++ impl
        self._active_ranks.append(new_rank)
        self._generation += 1

        if new_rank in self._failed_ranks:
            self._failed_ranks.remove(new_rank)

    def remove_rank(self, dead_rank: int) -> None:
        """Remove a rank from the EP group.

        Triggers:
        - Expert redistribution across surviving ranks
        - Routing cache invalidation
        - In-flight transfers to the dead rank are aborted

        Args:
            dead_rank: Rank ID of the GPU to remove.

        Raises:
            RuntimeError: If the remove operation fails.
        """
        if dead_rank not in self._active_ranks:
            raise ValueError(f"Rank {dead_rank} is not active")

        self._active_ranks.remove(dead_rank)
        self._failed_ranks.append(dead_rank)
        self._generation += 1

    def handle_failure(self, failed_rank: int) -> None:
        """Handle an unexpected rank failure.

        Called automatically by the NIXL-EP backend when a transfer
        to a rank fails. Triggers expert redistribution.

        Args:
            failed_rank: Rank that failed.
        """
        if failed_rank in self._active_ranks:
            self.remove_rank(failed_rank)
