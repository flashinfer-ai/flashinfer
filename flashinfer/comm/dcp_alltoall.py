"""
DCP All-to-All Operations for DCP Attention Reduction

Provides the DCP LL128 FIFO-based all-to-all kernel for context-parallel
attention reduction. Uses SM90+ features (TMA, mbarrier).

Usage protocol::

    # 1. Query workspace size
    ws_bytes = decode_cp_a2a_workspace_size(cp_size)

    # 2. Allocate workspace (MNNVL or plain device memory)
    workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank, mapping=mapping)

    # 3. Initialize workspace (synchronous — includes stream sync)
    decode_cp_a2a_init_workspace(workspace, cp_rank, cp_size)

    # 4. Cross-rank barrier (REQUIRED before first alltoall)
    dist.barrier(group)

    # 5. Run all-to-all
    recv_o, recv_stats = decode_cp_a2a_alltoall(
        partial_o, softmax_stats, workspace, cp_rank, cp_size
    )

.. important::
    All ranks MUST complete ``decode_cp_a2a_init_workspace`` and execute a
    cross-rank barrier before ANY rank calls ``decode_cp_a2a_alltoall``.
    Failure to do so causes a deadlock on MNNVL workspaces.

Tensor specifications:

- ``partial_o``: ``[..., cp_size, D]`` — half or bfloat16,
  ``D * element_size`` must be 16-byte aligned.
- ``softmax_stats``: ``[..., cp_size, S]`` — float32, ``S >= 2`` and even.
  Batch dims must match ``partial_o``.
- ``workspace``: ``[cp_size, ws_elems_per_rank]`` — int64, from
  :func:`decode_cp_a2a_allocate_workspace`.
"""

import functools
import logging
from types import SimpleNamespace
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.comm import gen_dcp_alltoall_module
from ..trace.templates.comm import decode_cp_a2a_alltoall_trace
from ..utils import device_support_pdl, register_custom_op
from .mapping import Mapping
from .mnnvl import MnnvlConfig, MnnvlMemory

logger = logging.getLogger(__name__)


# ─── Cached JIT Loader ───────────────────────────────────────────────────


@functools.cache
def get_dcp_alltoall_module():
    """Build (once) and return the DCP A2A JIT module with custom op wrappers."""
    module = gen_dcp_alltoall_module().build_and_load()

    @register_custom_op(
        "flashinfer::decode_cp_a2a_init_workspace",
        mutates_args=("workspace",),
    )
    def decode_cp_a2a_init_workspace(
        workspace: torch.Tensor,
        cp_rank: int,
        cp_size: int,
    ):
        module.initialize_dcp_workspace(workspace, cp_rank, cp_size)

    @register_custom_op(
        "flashinfer::decode_cp_a2a_alltoall",
        mutates_args=("workspace",),
    )
    def decode_cp_a2a_alltoall(
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
        workspace: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        enable_pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return module.alltoall_dcp_native(
            partial_o, softmax_stats, workspace, cp_rank, cp_size, enable_pdl
        )

    return SimpleNamespace(
        get_workspace_size_per_rank=module.get_dcp_workspace_size_per_rank,
        initialize_workspace=decode_cp_a2a_init_workspace,
        alltoall=decode_cp_a2a_alltoall,
    )


# ─── Public API ───────────────────────────────────────────────────────────


@flashinfer_api
def decode_cp_a2a_workspace_size(cp_size: int) -> int:
    """Return the workspace size **in bytes** per rank for the given CP group size.

    Args:
        cp_size: Context-parallel group size (number of ranks).

    Returns:
        Workspace size in bytes per rank.

    Example::

        >>> decode_cp_a2a_workspace_size(4)
        16778240
    """
    return get_dcp_alltoall_module().get_workspace_size_per_rank(cp_size)


@flashinfer_api
def decode_cp_a2a_allocate_workspace(
    cp_size: int,
    cp_rank: int,
    *,
    mapping: Optional[Mapping] = None,
    mnnvl_config: Optional[MnnvlConfig] = None,
) -> torch.Tensor:
    """Allocate a workspace tensor of shape ``[cp_size, ws_elems_per_rank]``.

    After allocation, call :func:`decode_cp_a2a_init_workspace` followed by a
    cross-rank barrier before the first :func:`decode_cp_a2a_alltoall` call.

    Two allocation modes:

    - **MNNVL** (``mapping`` provided): Cross-rank visible GPU memory via
      FlashInfer's ``MnnvlMemory``. Required for multi-node or when ranks
      cannot see each other's device memory directly.
    - **Plain device memory** (``mapping=None``): Standard ``torch.zeros``
      allocation. Sufficient for single-node with NVLink P2P.

    Args:
        cp_size: Context-parallel group size.
        cp_rank: This rank's position in the CP group.
        mapping: Mapping object for MNNVL allocation. If provided, MNNVL is
            used. The mapping must have ``cp_size`` set correctly. The
            communicator is split using ``mapping.pp_rank``, ``mapping.cp_rank``,
            and ``mapping.tp_rank``.
        mnnvl_config: Configuration for the MNNVL communication backend.
            Required when using MNNVL with ``torch.distributed`` (pass
            ``MnnvlConfig(comm_backend=TorchDistBackend(group))``).

    Returns:
        ``torch.int64`` tensor of shape ``[cp_size, ws_elems_per_rank]``.
    """
    ws_bytes = decode_cp_a2a_workspace_size(cp_size)

    if mapping is not None:
        MnnvlMemory.initialize()
        if mnnvl_config:
            MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)

        mnnvl_mem = MnnvlMemory(mapping, ws_bytes)
        workspace = mnnvl_mem.as_torch_strided_tensor(torch.int64)
        workspace._mnnvl_mem = mnnvl_mem  # prevent GC of MNNVL handle
        logger.info(
            "Rank %d: DCP MNNVL workspace allocated — shape=%s, stride=%s",
            cp_rank,
            list(workspace.shape),
            list(workspace.stride()),
        )
        return workspace

    ws_elems_per_rank = (ws_bytes + 7) // 8
    return torch.zeros(cp_size, ws_elems_per_rank, dtype=torch.int64, device="cuda")


@flashinfer_api
def decode_cp_a2a_init_workspace(
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Initialize the workspace FIFO buffers. Call once before the first alltoall.

    Resets the FIFO buffers in the **local** workspace row
    (``workspace[cp_rank]``). This function is **synchronous**: when it
    returns, the GPU memset is guaranteed to have completed.

    .. important::
        With MNNVL workspaces, **all ranks** must complete
        ``decode_cp_a2a_init_workspace`` and execute a cross-rank barrier
        (e.g. ``dist.barrier(group)``) before **any** rank calls
        :func:`decode_cp_a2a_alltoall`. Without the barrier, a rank may
        start writing to a peer's FIFO before that peer has finished
        initializing → deadlock.

    Args:
        workspace: ``[cp_size, ws_elems_per_rank]`` int64 tensor from
            :func:`decode_cp_a2a_allocate_workspace`.
        cp_rank: This rank's position in the CP group.
        cp_size: Context-parallel group size.
    """
    get_dcp_alltoall_module().initialize_workspace(workspace, cp_rank, cp_size)
    # CRITICAL: The C++ op uses cudaMemsetAsync. Without this sync, a
    # subsequent cross-GPU alltoall can race with the unfinished memset
    # on MNNVL memory, causing a deadlock.
    torch.cuda.current_stream().synchronize()


@flashinfer_api(trace=decode_cp_a2a_alltoall_trace)
def decode_cp_a2a_alltoall(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    enable_pdl: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform the DCP all-to-all exchange.

    Each rank sends its ``partial_o[..., peer, :]`` slice to the
    corresponding peer and receives all peers' contributions into the
    output tensors.

    Args:
        partial_o: ``[..., cp_size, D]`` — half or bfloat16.
            ``D * element_size`` must be 16-byte aligned.
        softmax_stats: ``[..., cp_size, S]`` — float32, ``S >= 2`` and even.
            Batch dimensions must match ``partial_o``.
        workspace: ``[cp_size, ws_elems_per_rank]`` int64 tensor from
            :func:`decode_cp_a2a_allocate_workspace`, already initialized.
        cp_rank: This rank's position in the CP group.
        cp_size: Context-parallel group size.
        enable_pdl: Enable Programmatic Dependent Launch (SM90+).
            Defaults to ``True`` on SM90+ GPUs, ``False`` otherwise.

    Returns:
        ``(partial_o_out, softmax_stats_out)`` with the same shapes and
        dtypes as the inputs. Each output contains the gathered data from
        all peers for this rank.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(partial_o.device)
    return get_dcp_alltoall_module().alltoall(
        partial_o, softmax_stats, workspace, cp_rank, cp_size, enable_pdl
    )


__all__ = [
    "decode_cp_a2a_workspace_size",
    "decode_cp_a2a_allocate_workspace",
    "decode_cp_a2a_init_workspace",
    "decode_cp_a2a_alltoall",
]
