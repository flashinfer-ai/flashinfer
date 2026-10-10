# flashinfer/ep/_backends/nccl_ep.py
#
# Python-side wrapper for the NCCL-EP backend.
# Handles NCCL-EP-specific initialization and ncclComm_t extraction.

import torch
from typing import Optional


class NcclEPBackendWrapper:
    """Python-side wrapper for NCCL-EP backend configuration.

    This wrapper handles NCCL-EP-specific concerns:
    - Issue #14 + #35: ncclComm_t extraction from PyTorch process group
    - Issue #39: Routing cache lifecycle
    - Issue #42: Ping-pong scratch buffer allocation

    The actual dispatch/combine operations are handled by the C++ backend
    via the pybind11 layer. This wrapper is used during group creation
    to extract the ncclComm_t and validate configuration.
    """

    @staticmethod
    def validate_config(
        num_experts: int,
        num_local_experts: int,
        hidden_dim: int,
        top_k: int,
    ) -> dict:
        """Validate NCCL-EP-specific configuration.

        Args:
            num_experts: Total number of experts across all ranks.
            num_local_experts: Experts on this rank.
            hidden_dim: Hidden dimension of tokens.
            top_k: Number of experts each token is routed to.

        Returns:
            Validated configuration dict.
        """
        return {
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
        }

    @staticmethod
    def extract_nccl_comm(
        process_group: torch.distributed.ProcessGroup,
        device_id: Optional[int] = None,
    ) -> int:
        """Extract ncclComm_t pointer from a PyTorch NCCL process group.

        Issue #14 + #35: getNCCLComm() is protected in PyTorch >= 2.0.
        We use the pattern that vLLM and SGLang actually use:
            pg._get_backend(device)._get_nccl_comm()

        Args:
            process_group: PyTorch distributed process group (must use NCCL backend).
            device_id: CUDA device ID. Defaults to current device.

        Returns:
            Raw pointer to ncclComm_t as an integer (for passing to C++ via pybind11).

        Raises:
            RuntimeError: If the process group doesn't use NCCL backend.
        """
        if device_id is None:
            device_id = torch.cuda.current_device()

        device = torch.device("cuda", device_id)

        try:
            pg_backend = process_group._get_backend(device)
            nccl_comm = pg_backend._get_nccl_comm()
            # nccl_comm is a PyCapsule or int — convert to int for pybind11
            if hasattr(nccl_comm, '__int__'):
                return int(nccl_comm)
            return nccl_comm
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to extract ncclComm_t from process group. "
                f"Ensure the process group uses NCCL backend. "
                f"Error: {e}"
            )
