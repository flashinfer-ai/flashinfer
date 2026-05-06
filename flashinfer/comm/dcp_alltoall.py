"""
DCP All-to-All Operations for DCP Attention Reduction

Provides the DCP LL128 FIFO-based all-to-all kernel for context-parallel
attention reduction. Uses SM90+ features (TMA, mbarrier).

The kernel addresses peer FIFOs via ``params.workspace + peer_rank * stride``,
so it requires a single unified VA spanning all CP ranks — i.e. MNNVL
fabric memory (currently provided by GB200-NVL72 systems). Non-MNNVL
allocations cannot satisfy this layout and would deadlock at runtime.

Usage protocol::

    # 1. Query workspace size
    ws_bytes = decode_cp_a2a_workspace_size(cp_size)

    # 2. Allocate MNNVL-backed workspace (Mapping is required and carries cp_size/cp_rank)
    workspace = decode_cp_a2a_allocate_mnnvl_workspace(mapping)

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
    Failure to do so causes a deadlock.

Tensor specifications:

- ``partial_o``: ``[..., cp_size, D]`` — half or bfloat16,
  ``D * element_size`` must be 16-byte aligned.
- ``softmax_stats``: ``[..., cp_size, S]`` — float32, ``S >= 2`` and even.
  Batch dims must match ``partial_o``.
- ``workspace``: ``[cp_size, ws_elems_per_rank]`` — int64, from
  :func:`decode_cp_a2a_allocate_mnnvl_workspace`.
"""

import functools
import logging
from types import SimpleNamespace
from typing import Dict, Optional

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


# Module-level keep-alive for MNNVL workspace handles. The kernel uses raw
# pointers from the strided tensor, but the underlying fabric memory is owned
# by the MnnvlMemory wrapper — when its refcount hits zero, ``__del__`` calls
# ``close_mnnvl_memory`` and unmaps the VA. Without a stable reference outside
# the returned tensor, any caller-side ``view`` / ``slice`` / ``clone`` that
# drops the original tensor would silently free the workspace under the kernel.
_workspace_keepalive: Dict[int, MnnvlMemory] = {}


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
def decode_cp_a2a_allocate_mnnvl_workspace(
    mapping: Mapping,
    *,
    mnnvl_config: Optional[MnnvlConfig] = None,
) -> torch.Tensor:
    """Allocate an MNNVL-backed workspace of shape ``[cp_size, ws_elems_per_rank]``.

    The DCP A2A kernel requires a single unified VA spanning all CP ranks
    (see module docstring), so workspace allocation must go through MNNVL
    fabric memory. This function is the only supported allocator.

    After allocation, call :func:`decode_cp_a2a_init_workspace` followed by a
    cross-rank barrier before the first :func:`decode_cp_a2a_alltoall` call.

    Args:
        mapping: Mapping object for MNNVL allocation. Carries ``cp_size`` and
            ``cp_rank``. The communicator is split using ``mapping.pp_rank``,
            ``mapping.cp_rank``, and ``mapping.tp_rank``.
        mnnvl_config: Configuration for the MNNVL communication backend.
            Required when using MNNVL with ``torch.distributed`` (pass
            ``MnnvlConfig(comm_backend=TorchDistBackend(group))``).

    Returns:
        ``torch.int64`` tensor of shape ``[cp_size, ws_elems_per_rank]``.
    """
    ws_bytes = decode_cp_a2a_workspace_size(mapping.cp_size)

    MnnvlMemory.initialize()
    if mnnvl_config:
        MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)

    mnnvl_mem = MnnvlMemory(mapping, ws_bytes)
    workspace = mnnvl_mem.as_torch_strided_tensor(torch.int64)
    _workspace_keepalive[workspace.data_ptr()] = mnnvl_mem
    logger.info(
        "Rank %d: DCP MNNVL workspace allocated — shape=%s, stride=%s",
        mapping.cp_rank,
        list(workspace.shape),
        list(workspace.stride()),
    )
    return workspace


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
            :func:`decode_cp_a2a_allocate_mnnvl_workspace`.
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
            :func:`decode_cp_a2a_allocate_mnnvl_workspace`, already initialized.
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
    "decode_cp_a2a_allocate_mnnvl_workspace",
    "decode_cp_a2a_init_workspace",
    "decode_cp_a2a_alltoall",
]
