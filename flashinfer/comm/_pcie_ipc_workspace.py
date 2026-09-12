"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._pcie_ipc_ag_rs_topology import resolve_pcie_ipc_ag_rs_topology
from .cuda_ipc import create_shared_buffer, free_shared_buffer
from .pcie_ipc_topology import resolve_pcie_ipc_profile


PCIE_IPC_MAX_BLOCKS = 64
_PACK_BYTES = 16
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_WORLD_SIZES = (2, 4, 8)


class _PcieIpcWorkspace:
    """Own one CUDA IPC slab and one protocol handle for an AG or RS channel."""

    def __init__(
        self,
        *,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype,
        max_blocks: int,
        profile: Optional[str],
        collective_name: str,
        module_getter: Callable[[], Any],
        workspace_size_name: str,
        init_name: str,
        dispose_name: str,
        layout_uses_ordered_4plus4: bool = False,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        self._group = group
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._dtype = dtype
        self._max_numel = max_numel
        self._max_blocks = max_blocks
        self._element_size = 0
        self._profile = ""
        self._profile_reason = ""
        self._ordered_4plus4 = False
        self._ordered_4plus4_reason = "not evaluated"
        self._placement_fingerprint = ""

        self._collective_name = collective_name
        self._ipc_ptrs: Optional[List[int]] = None
        self._handle: Optional[int] = None
        self._dispose: Optional[Callable[[int], None]] = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._tuning_state: Optional[Any] = None

        error: Optional[str] = None
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            error = (
                f"world size {self.world_size} unsupported; "
                f"expected one of {_SUPPORTED_WORLD_SIZES}"
            )
        elif dtype not in _SUPPORTED_DTYPES:
            error = f"dtype {dtype} unsupported; expected one of {_SUPPORTED_DTYPES}"
        else:
            self._element_size = torch.empty((), dtype=dtype).element_size()
            pack_elements = _PACK_BYTES // self.element_size
            if type(max_numel) is not int or max_numel <= 0:
                error = f"max_numel must be a positive integer, got {max_numel!r}"
            elif max_numel % pack_elements != 0:
                error = (
                    f"max_numel must be a multiple of {pack_elements} for "
                    f"{dtype}, got {max_numel}"
                )
            elif (
                type(max_blocks) is not int
                or max_blocks <= 0
                or max_blocks > PCIE_IPC_MAX_BLOCKS
            ):
                error = (
                    f"max_blocks must be an integer in [1, {PCIE_IPC_MAX_BLOCKS}], "
                    f"got {max_blocks!r}"
                )

        self._joint_check(
            {
                "error": error,
                "max_numel": max_numel,
                "dtype": str(dtype),
                "max_blocks": max_blocks,
                "profile": profile,
            },
            "validating arguments",
        )

        try:
            decision = resolve_pcie_ipc_profile(group, requested=profile)
            self._profile = decision.profile
            self._profile_reason = decision.reason
            topology = resolve_pcie_ipc_ag_rs_topology(group, self.device)
            self._ordered_4plus4 = topology.ordered_4plus4
            self._ordered_4plus4_reason = topology.reason
            self._placement_fingerprint = topology.placement_fingerprint
            module = module_getter()
            workspace_size = getattr(module, workspace_size_name)
            init = getattr(module, init_name)
            self._dispose = getattr(module, dispose_name)
            layout_args = (self.ordered_4plus4,) if layout_uses_ordered_4plus4 else ()
            nbytes = workspace_size(
                self.world_size,
                self.max_numel,
                self.element_size,
                self.max_blocks,
                *layout_args,
            )
        except Exception as err:  # noqa: BLE001 - re-raised collectively
            self._joint_check(
                {"error": f"{type(err).__name__}: {err}"}, "preparing the workspace"
            )
            raise
        self._joint_check({"error": None}, "preparing the workspace")

        self._ipc_ptrs = create_shared_buffer(nbytes, group=group)
        bind_error: Optional[str] = None
        try:
            self._handle = init(
                self._ipc_ptrs,
                self.rank,
                self.max_numel,
                self.element_size,
                self.max_blocks,
                *layout_args,
            )
            # init() clears this rank's slab. No peer may publish into it until
            # every rank has completed the memset.
            torch.cuda.synchronize(self.device)
        except Exception as err:  # noqa: BLE001 - re-raised collectively
            bind_error = f"{type(err).__name__}: {err}"

        try:
            self._joint_check({"error": bind_error}, "binding the workspace")
        except Exception:
            self.destroy()
            raise

    def _joint_check(self, local: dict, what: str) -> None:
        """Make a construction or tuning decision identical on every rank."""
        gathered: List[Optional[dict]] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.group)
        entries = [entry for entry in gathered if entry is not None]

        failed = {
            rank: entry["error"]
            for rank, entry in enumerate(entries)
            if entry.get("error")
        }
        if failed:
            raise ValueError(f"{self._collective_name} failed while {what}: {failed}")

        mismatched = {
            key: [entry[key] for entry in entries]
            for key in local
            if key != "error" and len({repr(entry[key]) for entry in entries}) > 1
        }
        if mismatched:
            raise ValueError(
                "every rank must use identical collective arguments, "
                f"but these differ: {mismatched}"
            )

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("workspace has been destroyed")
        return self._handle

    @property
    def group(self) -> ProcessGroup:
        return self._group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def max_numel(self) -> int:
        return self._max_numel

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    @property
    def element_size(self) -> int:
        return self._element_size

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def profile_reason(self) -> str:
        return self._profile_reason

    @property
    def ordered_4plus4(self) -> bool:
        return self._ordered_4plus4

    @property
    def ordered_4plus4_reason(self) -> str:
        return self._ordered_4plus4_reason

    @property
    def placement_fingerprint(self) -> str:
        return self._placement_fingerprint

    def _check_stream(self, operation: str) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        current = torch.cuda.current_stream(self.device)
        if self._stream is None:
            self._stream = current
        elif current != self._stream:
            raise RuntimeError(
                f"this workspace is bound to {self._stream}, but {operation} "
                f"was called on {current}; use one workspace per stream"
            )

    def rebind_stream(self) -> None:
        """Allow the next call to bind after the previous stream is ordered."""
        self._stream = None

    def _supports_input(self, inp: torch.Tensor) -> bool:
        if inp.device != self.device:
            raise ValueError(
                f"input is on {inp.device} but the workspace was built on {self.device}"
            )
        supported = (
            self._handle is not None
            and inp.dtype is self.dtype
            and inp.dim() == 2
            and inp.is_contiguous()
            and inp.numel() > 0
        )
        if supported and inp.data_ptr() % _PACK_BYTES != 0:
            raise ValueError("input must be 16-byte aligned")
        return supported

    def _validate_output(self, out: torch.Tensor, expected_shape: tuple) -> None:
        if out.device != self.device:
            raise ValueError(
                f"output is on {out.device} but the workspace was built on {self.device}"
            )
        if out.dtype is not self.dtype:
            raise ValueError(f"output dtype must be {self.dtype}")
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"output shape must be {expected_shape}, got {tuple(out.shape)}"
            )
        if not out.is_contiguous():
            raise ValueError("output must be contiguous")
        if out.data_ptr() % _PACK_BYTES != 0:
            raise ValueError("output must be 16-byte aligned")

    @staticmethod
    def _validate_no_overlap(inp: torch.Tensor, out: torch.Tensor) -> None:
        input_begin = inp.data_ptr()
        input_end = input_begin + inp.numel() * inp.element_size()
        output_begin = out.data_ptr()
        output_end = output_begin + out.numel() * out.element_size()
        if max(input_begin, output_begin) < min(input_end, output_end):
            raise ValueError("input and output must not overlap or alias")

    def destroy(self) -> None:
        """Collectively wait for users, dispose the handle, and free the slab."""
        if (
            self._handle is None
            and self._ipc_ptrs is None
            and self._tuning_state is None
        ):
            return

        torch.cuda.synchronize(self.device)
        if self._tuning_state is not None:
            self._tuning_state.destroy()
            self._tuning_state = None
        if self._handle is not None:
            if self._dispose is None:
                raise RuntimeError("workspace has no dispose operation")
            self._dispose(self._handle)
            self._handle = None
        if self._ipc_ptrs is not None:
            free_shared_buffer(self._ipc_ptrs, group=self.group)
            self._ipc_ptrs = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.destroy()
