"""CFT logical endpoints over an :class:`NVLinkOneSidedAlltoAll` workspace.

CFT (Compute Fabric Transport) fabric operations address a peer's memory by
logical-endpoint ID and offset instead of a mapped virtual address. A data path
that moves workspace rows with CFT binds one logical endpoint per rank to the
rank's workspace slice; :class:`CftEndpoints` creates, exchanges, queries and
releases them (the ``moe_a2a_cft_*`` ops). The all-to-all itself does
not use CFT, and its configuration does not expose it.

Logical endpoints need compute capability 10.0+, a driver of branch 615 or
newer, CUDA 13.4+ headers at build time, an IMEX daemon and a workspace shared
through FABRIC handles. A process binds one workspace at a time, and the
workspace stays allocated until its endpoints are destroyed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

    from ......comm.abstractions import CommBackend
    from .communication import NVLinkOneSidedAlltoAll

_MIN_DRIVER_BRANCH = 615

# The process's endpoints, from bind until destroy: the native ops hold one
# binding per process.
_bound: Optional["CftEndpoints"] = None


def _driver_branch() -> Optional[int]:
    """Major version of the installed NVIDIA driver, or ``None`` if NVML fails."""
    import pynvml

    try:
        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None
    if isinstance(version, bytes):
        version = version.decode(errors="replace")
    match = re.match(r"\s*(\d+)", str(version))
    return int(match.group(1)) if match else None


def _probe_cft_support(device_index: int) -> Optional[str]:
    import torch

    from ......comm.mnnvl import cuda

    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 10:
        return f"compute capability {major}.{minor} is below 10.0"
    branch = _driver_branch()
    if branch is None:
        return "the NVIDIA driver version could not be queried"
    if branch < _MIN_DRIVER_BRANCH:
        return f"driver branch {branch} is older than {_MIN_DRIVER_BRANCH}"
    try:
        attribute = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED
    except AttributeError:
        return "the installed cuda-python does not expose logical endpoints"
    status, supported = cuda.cuDeviceGetAttribute(attribute, device_index)
    if status != cuda.CUresult.CUDA_SUCCESS:
        return f"querying {attribute.name} failed with {status.name}"
    if not supported:
        return f"the device does not support {attribute.name}"
    return None


def cft_unsupported_reason(device_index: int, fabric_handles: bool) -> Optional[str]:
    """Why this rank cannot bind CFT logical endpoints, or ``None`` if it can.

    ``fabric_handles`` tells whether the workspace is shared through FABRIC
    handles; logical endpoints bind only such memory. Never raises, so that
    every rank reaches a collective agreement on the result.
    """
    if not fabric_handles:
        return "the workspace is not shared through FABRIC handles"
    try:
        return _probe_cft_support(device_index)
    except Exception as exc:  # NVML, CUDA driver or cuda-python errors alike
        return f"the CFT support probe failed: {type(exc).__name__}: {exc}"


class CftEndpoints:
    """The process's CFT logical endpoints on one all-to-all workspace.

    Every method is collective over the EP group: every rank calls it in the
    same order and gets the same result. The endpoints hold a reference to the
    workspace, so destroying the all-to-all first leaves them usable.
    """

    def __init__(self, alltoall: "NVLinkOneSidedAlltoAll") -> None:
        self._alltoall = alltoall
        self._state: Optional[dict] = alltoall._state

    @classmethod
    def bind(
        cls, alltoall: "NVLinkOneSidedAlltoAll", comm: "CommBackend"
    ) -> Optional["CftEndpoints"]:
        """Bind a logical endpoint to every rank's slice of ``alltoall``'s
        workspace, or return ``None`` on every rank if some rank cannot,
        including when its process already holds endpoints."""
        import torch

        from ......comm.mnnvl import all_ranks_agree

        global _bound
        state = alltoall._state
        if state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        if _bound is not None:
            reason = "this process already holds CFT logical endpoints"
        else:
            reason = cft_unsupported_reason(
                torch.cuda.current_device(), state["mnnvl_mem"].fabric_handles
            )
        if not all_ranks_agree(comm, reason is None):
            return None
        endpoints = cls(alltoall)
        if not endpoints._create(comm):
            return None
        state["refcount"] += 1
        _bound = endpoints
        return endpoints

    def _create(self, comm: "CommBackend") -> bool:
        """Create this rank's endpoint and import every peer's; a failure on
        any rank leaves no endpoint behind."""
        import torch

        from ......comm.mnnvl import all_ranks_agree

        a2a, state = self._alltoall, self._state
        module = a2a._module
        handle = torch.empty(
            module.moe_a2a_cft_handle_bytes(), dtype=torch.uint8, device="cpu"
        )
        created = module.moe_a2a_cft_create_endpoint(
            state["workspace"],
            state["mnnvl_mem"].local_mem_handle,
            int(state["metainfo"][a2a._index.WORKSPACE_SIZE_INDEX]),
            a2a.ep_rank,
            a2a.ep_size,
            handle,
        )
        bound = all_ranks_agree(comm, created)
        if bound:
            handles = comm.allgather(handle.numpy().tobytes())
            all_handles = torch.frombuffer(
                bytearray(b"".join(handles)), dtype=torch.uint8
            )
            bound = all_ranks_agree(
                comm,
                module.moe_a2a_cft_import_endpoints(
                    state["workspace"], a2a.ep_rank, all_handles
                ),
            )
        if not bound:
            module.moe_a2a_cft_destroy(state["workspace"], a2a.ep_rank)
            return False
        comm.barrier()
        return True

    def _workspace(self) -> "torch.Tensor":
        if self._state is None:
            raise RuntimeError("the CFT logical endpoints have been destroyed")
        return self._state["workspace"]

    def ids(self) -> "torch.Tensor":
        """CPU ``int64`` logical-endpoint ID of every rank's slice, by rank."""
        import torch

        a2a = self._alltoall
        ids = torch.empty(a2a.ep_size, dtype=torch.int64, device="cpu")
        a2a._module.moe_a2a_cft_endpoint_ids(self._workspace(), a2a.ep_rank, ids)
        return ids

    def release(self) -> None:
        """Destroy the endpoints but keep their IDs reserved, before
        :meth:`NVLinkOneSidedAlltoAll.checkpoint_prepare` releases the memory."""
        a2a = self._alltoall
        a2a._module.moe_a2a_cft_release_endpoints(self._workspace(), a2a.ep_rank)

    def rebind(self, comm: "CommBackend", previous_ids: "torch.Tensor") -> None:
        """Recreate the endpoints under ``previous_ids`` after
        :meth:`NVLinkOneSidedAlltoAll.checkpoint_restore`. Raises if that is
        not possible, since kernels captured with the old IDs would write
        elsewhere; the endpoints are then destroyed."""
        from ......comm.mnnvl import all_ranks_agree

        if self._state is None:
            raise RuntimeError("the CFT logical endpoints have been destroyed")
        if not self._create(comm):
            self._unbind()
            raise RuntimeError("could not recreate the CFT logical endpoints")
        if not all_ranks_agree(comm, bool((self.ids() == previous_ids).all())):
            self.destroy()
            raise RuntimeError("the CFT logical-endpoint IDs changed across the rebind")

    def destroy(self) -> None:
        """Destroy the endpoints, release their IDs and drop their reference to
        the workspace; every rank must have finished its CFT work on the
        workspace. Does nothing once they are destroyed."""
        if self._state is None:
            return
        a2a = self._alltoall
        a2a._module.moe_a2a_cft_destroy(self._state["workspace"], a2a.ep_rank)
        self._unbind()

    def _unbind(self) -> None:
        """Forget the native endpoints, which are gone, and drop the reference
        to the workspace."""
        global _bound
        if _bound is self:
            _bound = None
        state, self._state = self._state, None
        self._alltoall._release_workspace(state)
