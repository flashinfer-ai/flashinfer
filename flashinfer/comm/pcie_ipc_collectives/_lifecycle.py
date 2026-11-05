# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Group coordination and resource retirement shared by PCIe IPC workspaces."""

from typing import Callable, List, Optional, Protocol

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def joint_check(
    group: ProcessGroup,
    world_size: int,
    local: dict,
    what: str,
    *,
    collective_name: str,
    require_identical: bool = True,
) -> List[dict]:
    """Exchange outcomes before any rank takes a collective-dependent branch."""
    gathered: List[Optional[dict]] = [None] * world_size
    dist.all_gather_object(gathered, local, group=group)
    entries = [entry for entry in gathered if entry is not None]
    failed = {
        rank: entry["error"] for rank, entry in enumerate(entries) if entry.get("error")
    }
    if failed:
        raise ValueError(f"{collective_name} failed while {what}: {failed}")

    mismatched = {
        key: [entry[key] for entry in entries]
        for key in local
        if require_identical
        and key != "error"
        and len({repr(entry[key]) for entry in entries}) > 1
    }
    if mismatched:
        raise ValueError(
            "every rank must use identical collective arguments, "
            f"but these differ: {mismatched}"
        )
    return entries


def bind_stream(
    device: torch.device, bound: Optional[torch.cuda.Stream], operation: str
) -> Optional[torch.cuda.Stream]:
    """Require ordered execution of the workspace's mutable protocol state.

    Capture records work on a side stream; the caller must order graph replays
    with all other uses of the workspace, just as it must order eager calls.
    """
    if torch.cuda.is_current_stream_capturing():
        return bound
    current = torch.cuda.current_stream(device)
    if bound is not None and current != bound:
        raise RuntimeError(
            f"this workspace is already bound to {bound}, but {operation} was called "
            f"on {current}; use one workspace per stream"
        )
    return current


class _WorkspaceResources(Protocol):
    _handle: Optional[int]
    _ipc_ptrs: Optional[List[int]]

    @property
    def device(self) -> torch.device: ...

    @property
    def group(self) -> ProcessGroup: ...


def release_workspace(
    workspace: _WorkspaceResources,
    *,
    dispose: Optional[Callable[[int], None]],
    free: Callable[..., None],
) -> None:
    """Wait for GPU users and clear ownership after each successful release.

    Clear each owner field only after its release succeeds. In particular, a
    failed peer unmap must not cause a disposed protocol handle to be retried.
    The IPC helper owns cross-rank barriers and any recovery from partial
    unmapping; retaining the pointer list does not make such recovery safe.
    """
    if workspace._handle is None and workspace._ipc_ptrs is None:
        return
    torch.cuda.synchronize(workspace.device)
    if workspace._handle is not None:
        if dispose is None:
            raise RuntimeError("workspace has no dispose operation")
        dispose(workspace._handle)
        workspace._handle = None
    if workspace._ipc_ptrs is not None:
        free(workspace._ipc_ptrs, group=workspace.group)
        workspace._ipc_ptrs = None
