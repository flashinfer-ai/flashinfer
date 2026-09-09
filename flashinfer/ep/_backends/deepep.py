# flashinfer/ep/_backends/deepep.py
#
# Python-side wrapper for the DeepEP backend.
# Handles DeepEP-specific initialization and configuration.

from typing import Optional


class DeepEPBackendWrapper:
    """Python-side wrapper for DeepEP backend configuration.

    This wrapper handles DeepEP-specific concerns:
    - Issue #12: Mode switching between HT and LL
    - Issue #13: Auto-setting num_qps_per_rank = num_local_experts
    - Issue #17: Ping-pong scratch buffer allocation
    - Issue #19: CUDA graph dtype guard

    The actual dispatch/combine operations are handled by the C++ backend
    via the pybind11 layer. This wrapper is used during group creation
    to validate and transform DeepEP-specific configuration.
    """

    @staticmethod
    def validate_config(
        num_experts: int,
        num_local_experts: int,
        hidden_dim: int,
        num_qps_per_rank: int = 0,
        nvl_buffer_bytes: Optional[int] = None,
        rdma_buffer_bytes: Optional[int] = None,
    ) -> dict:
        """Validate and normalize DeepEP-specific configuration.

        Args:
            num_experts: Total number of experts across all ranks.
            num_local_experts: Experts on this rank.
            hidden_dim: Hidden dimension of tokens.
            num_qps_per_rank: QPs per rank (0 = auto = num_local_experts).
            nvl_buffer_bytes: NVLink buffer size (None = auto).
            rdma_buffer_bytes: RDMA buffer size (None = auto).

        Returns:
            Validated configuration dict.

        Raises:
            ValueError: If configuration is invalid.
        """
        if num_qps_per_rank != 0 and num_qps_per_rank != num_local_experts:
            raise ValueError(
                f"DeepEP: num_qps_per_rank ({num_qps_per_rank}) must equal "
                f"num_local_experts ({num_local_experts}) for LL mode. "
                f"Set to 0 for auto."
            )

        return {
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "hidden_dim": hidden_dim,
            "num_qps_per_rank": num_qps_per_rank if num_qps_per_rank > 0 else num_local_experts,
            "nvl_buffer_bytes": nvl_buffer_bytes,
            "rdma_buffer_bytes": rdma_buffer_bytes,
        }

    @staticmethod
    def extract_nccl_comm(process_group) -> None:
        """DeepEP does not use NCCL comm — returns None."""
        return None
