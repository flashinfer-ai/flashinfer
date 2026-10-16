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

Symmetric-memory exchange workspace of the Cake ICP indexer chain.

One workspace per rank holds the transport pool the exchange-merge program
pushes its top-16 lists into and polls its peers' lists from: two parity
halves of ``tokens_cap * heads_local`` (token, local head) rows times
``world_size`` source regions of ``TOPK`` 64-bit words, allocated through
PyTorch symmetric memory so every peer's half is mapped into this rank's
address space.  The per-peer base pointers form the peer table the generated
host binding passes to the kernel by value; a second, small symmetric
allocation provides the process-group flag words the binding requires
alongside the table.  The workspace also owns the per-row generation words,
the status word and the per-CTA timestamp rows of the merge launch.

Creation and destruction are collectives: every rank of the group must call
them in the same order.  Neither is allowed inside CUDA Graph capture; a
prepared runner launches against a created workspace with no allocation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

NUM_HEADS = 4
TOPK = 16
PEER_TABLE_CAPACITY = 8  # entries of the by-value peer table ABI
FLAG_WORDS = 64  # int32 words of the process-group flag pad (zeroed; the merge does not read them)
POISON_WORD = -1  # int64 all-ones: every transport word reads the poison generation
OUT_POISON = -7  # never a block id nor the -1 tail


def ffi_pointer_table(pointers) -> Any:
    """Per-peer device pointers as the ``Shape`` the generated binding reads into its peer table.

    The binding checks ``len(pointers)`` against the kernel's static world size
    and rejects null entries; pointers at or above ``2**63`` travel as the
    two's-complement ``int64`` the FFI boundary carries.
    """
    import tvm_ffi

    values = []
    for pointer in pointers:
        pointer = int(pointer)
        if pointer <= 0:
            raise ValueError(
                "peer pointer table entries must be non-null device pointers"
            )
        values.append(pointer if pointer < (1 << 63) else pointer - (1 << 64))
    if not 1 <= len(values) <= PEER_TABLE_CAPACITY:
        raise ValueError(
            f"peer pointer table holds 1..{PEER_TABLE_CAPACITY} entries, got {len(values)}"
        )
    return tvm_ffi.Shape(tuple(values))


def workspace_bytes(world_size: int, tokens_cap: int) -> dict[str, int]:
    """Byte budget of one rank's workspace (``pool_bytes`` is the symmetric allocation)."""
    world_size, tokens_cap = int(world_size), int(tokens_cap)
    if world_size not in (2, 4):
        raise ValueError("the ICP indexer chain runs on a world of 2 or 4 ranks")
    if tokens_cap < 1:
        raise ValueError("tokens_cap must be positive")
    heads_local = NUM_HEADS // world_size
    rows_cap = tokens_cap * heads_local
    par_stride = rows_cap * world_size * TOPK
    words = 2 * par_stride
    return dict(
        heads_local=heads_local,
        rows_cap=rows_cap,
        par_stride=par_stride,
        pool_words=words,
        pool_bytes=words * 8,
        flags_bytes=FLAG_WORDS * 4,
        gen_rows_bytes=tokens_cap * NUM_HEADS * 4,
        stamps_bytes=tokens_cap * NUM_HEADS * 3 * 8,
        status_bytes=4,
    )


class MSAIcpIndexerWorkspace:
    """Exchange workspace of one rank (see the module docstring).

    ``mode == "symmetric"`` allocates through PyTorch symmetric memory on
    ``group`` (default: the world group) and is the production form.
    ``mode == "emulate"`` builds the ``world_size`` ranks' pools in one
    process (plain device tensors shared by the emulated workspaces); the
    kernels see device pointers either way.  Emulated workspaces are for
    single-GPU tests of the exchange protocol and must drive the merge with
    the serialised push / poll phases (see ``MSAIcpIndexerRunner.launch_merge``).
    """

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        tokens_cap: int,
        device: torch.device,
        group=None,
        mode: str = "symmetric",
        emulated_pools: Optional[list[torch.Tensor]] = None,
        emulated_flags: Optional[torch.Tensor] = None,
    ):
        sizes = workspace_bytes(world_size, tokens_cap)
        self.world_size, self.rank, self.tokens_cap = (
            int(world_size),
            int(rank),
            int(tokens_cap),
        )
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {rank} is outside the world of {world_size}")
        self.heads_local = sizes["heads_local"]
        self.rows_cap = sizes["rows_cap"]
        self.par_stride = sizes["par_stride"]
        self.words = sizes["pool_words"]
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the ICP indexer workspace lives on a CUDA device")
        self.mode = mode
        self.group = group
        self._handles: tuple = ()
        if mode == "symmetric":
            self._allocate_symmetric()
        elif mode == "emulate":
            if emulated_pools is None or len(emulated_pools) != self.world_size:
                raise ValueError(
                    "emulate mode needs the pool tensors of every emulated rank"
                )
            if emulated_flags is None:
                raise ValueError("emulate mode needs the shared flag pad")
            self.pool_i64 = emulated_pools[self.rank]
            self.peer_pointers = tuple(int(p.data_ptr()) for p in emulated_pools)
            self.flags = emulated_flags
            self.flag_pointers = tuple(
                int(emulated_flags.data_ptr()) for _ in range(self.world_size)
            )
        else:
            raise ValueError("workspace mode must be 'symmetric' or 'emulate'")
        if int(self.pool_i64.numel()) < self.words:
            raise ValueError(
                f"pool holds {int(self.pool_i64.numel())} words, {self.words} needed"
            )
        # The kernel declares the pool as 64-bit unsigned words, [2 parities][par_stride];
        # the storage stays int64 (allocation and the -1 poison fill).
        self.pool = (
            self.pool_i64[: self.words].view(torch.uint64).view(2, self.par_stride)
        )
        self.gen_rows = torch.zeros(
            (self.tokens_cap * NUM_HEADS,), dtype=torch.int32, device=self.device
        )
        self.status = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.stamps_i64 = torch.zeros(
            (self.tokens_cap * NUM_HEADS * 3,), dtype=torch.int64, device=self.device
        )
        self.stamps = self.stamps_i64.view(torch.uint64)
        self.stamp_rows = self.tokens_cap * NUM_HEADS
        self._peer_table = ffi_pointer_table(self.peer_pointers)
        self._flag_table = ffi_pointer_table(self.flag_pointers)
        self.reset()
        if mode == "symmetric":
            # Every peer must observe the poisoned pool and the zeroed flags before
            # any rank launches against them.
            torch.cuda.synchronize(self.device)
            self._barrier()
        self.destroyed = False

    # ------------------------------------------------------------------ allocation

    def _group_name(self) -> str:
        import torch.distributed as dist

        group = self.group if self.group is not None else dist.group.WORLD
        if group is None:
            raise RuntimeError(
                "torch.distributed must be initialised before creating a symmetric workspace"
            )
        return group.group_name

    def _barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self.group)

    def _allocate_symmetric(self) -> None:
        import torch.distributed as dist

        from ...comm.torch_symmetric_memory import _alloc_symm_buffer_bytes

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialised before creating a symmetric workspace"
            )
        group = self.group if self.group is not None else dist.group.WORLD
        if dist.get_world_size(group) != self.world_size:
            raise ValueError(
                f"the process group has {dist.get_world_size(group)} ranks, the workspace was asked for {self.world_size}"
            )
        if dist.get_rank(group) != self.rank:
            raise ValueError(
                f"this process is rank {dist.get_rank(group)} of the group, the workspace was asked for {self.rank}"
            )
        group_name = self._group_name()
        pool_ptrs, pool, pool_handle = _alloc_symm_buffer_bytes(
            self.words * 8, self.world_size, torch.int64, self.device, group_name
        )
        flag_ptrs, flags, flag_handle = _alloc_symm_buffer_bytes(
            FLAG_WORDS * 4, self.world_size, torch.int32, self.device, group_name
        )
        if len(pool_ptrs) != self.world_size or len(flag_ptrs) != self.world_size:
            raise RuntimeError(
                "symmetric memory returned a peer table of the wrong size"
            )
        self.pool_i64 = pool
        self.flags = flags
        self.flags.zero_()
        self.peer_pointers = tuple(int(p) for p in pool_ptrs)
        self.flag_pointers = tuple(int(p) for p in flag_ptrs)
        self._handles = (pool_handle, flag_handle)

    @classmethod
    def emulated(
        cls, world_size: int, tokens_cap: int, device: torch.device
    ) -> list["MSAIcpIndexerWorkspace"]:
        """The ``world_size`` ranks' workspaces of one process sharing plain device pools."""
        sizes = workspace_bytes(world_size, tokens_cap)
        device = torch.device(device)
        pools = [
            torch.empty((sizes["pool_words"],), dtype=torch.int64, device=device)
            for _ in range(int(world_size))
        ]
        flags = torch.zeros((FLAG_WORDS,), dtype=torch.int32, device=device)
        return [
            cls(
                world_size=world_size,
                rank=rank,
                tokens_cap=tokens_cap,
                device=device,
                mode="emulate",
                emulated_pools=pools,
                emulated_flags=flags,
            )
            for rank in range(int(world_size))
        ]

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Re-poison the pool and zero the counters (required after a failed launch before reuse)."""
        self.pool_i64.fill_(POISON_WORD)
        self.gen_rows.zero_()
        self.status.zero_()
        self.stamps_i64.zero_()

    def status_value(self) -> int:
        """Host read of the status word.  Never call it during CUDA Graph capture."""
        return int(self.status.item())

    def merge_bindings(self) -> dict[str, Any]:
        """Kernel arguments of the exchange-merge program that this workspace provides."""
        return dict(
            xpool=self.pool,
            xpool_peers=self._peer_table,
            pg_world=self.world_size,
            pg_rank=self.rank,
            pg_flags=self._flag_table,
            gen_ctr=self.gen_rows,
            stamps=self.stamps,
            x_par_stride=self.par_stride,
        )

    def stamp_rows_of(self, token_count: int) -> torch.Tensor:
        """CPU ``[rows, 3]`` globaltimer stamps (ns) of the last merge launch over ``token_count`` tokens."""
        rows = int(token_count) * NUM_HEADS
        return self.stamps_i64[: rows * 3].view(rows, 3).cpu()

    def owner_rows(self, token_count: int) -> torch.Tensor:
        """CPU indices of the (token, head) rows whose head this rank owns (the rows that poll and write ``out``)."""
        rows = torch.arange(int(token_count) * NUM_HEADS)
        return rows[((rows % NUM_HEADS) // self.heads_local) == self.rank]

    def destroy(self) -> None:
        """Release the workspace.  Collective in ``symmetric`` mode: every rank calls it, after its last launch completed."""
        if getattr(self, "destroyed", False):
            return
        torch.cuda.synchronize(self.device)
        if self.mode == "symmetric":
            self._barrier()
        self._handles = ()
        self.pool = self.pool_i64 = self.flags = None  # type: ignore[assignment]
        self.destroyed = True


def create_workspace(
    *,
    world_size: int,
    rank: int,
    tokens_cap: int,
    device: torch.device,
    group=None,
) -> MSAIcpIndexerWorkspace:
    return MSAIcpIndexerWorkspace(
        world_size=world_size,
        rank=rank,
        tokens_cap=tokens_cap,
        device=device,
        group=group,
        mode="symmetric",
    )


def destroy_workspace(workspace: MSAIcpIndexerWorkspace) -> None:
    if not isinstance(workspace, MSAIcpIndexerWorkspace):
        raise TypeError("destroy_workspace expects an MSAIcpIndexerWorkspace")
    workspace.destroy()


__all__ = [
    "FLAG_WORDS",
    "MSAIcpIndexerWorkspace",
    "NUM_HEADS",
    "OUT_POISON",
    "PEER_TABLE_CAPACITY",
    "POISON_WORD",
    "TOPK",
    "create_workspace",
    "destroy_workspace",
    "ffi_pointer_table",
    "workspace_bytes",
]
