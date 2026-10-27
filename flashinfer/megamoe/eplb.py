"""Redundant expert slots and the replica plan of the SM90 MXFP4 MegaMoE kernel.

Each rank keeps its ``H = num_experts / ranks`` resident experts in local slots
``[0, H)`` and ``S`` redundant expert slots in ``[H, H + S)``; the kernel runs on
``ranks * (H + S)`` physical experts. A replica plan is a set of tables in a
symmetric workspace: which logical expert each redundant slot holds and how
each expert's token blocks split across its copies. The logical routing is
rewritten onto the physical slots before the kernel runs; slots whose expert
changed receive their weights from the resident rank over NVLink, either by a
separate copy kernel or by the MegaMoE kernel itself. Correctness never
depends on the plan: any slot holding an expert's weights computes the same
output.

This module owns the allocations and exposes the tables as tensor views. A
framework either writes the tables itself (:meth:`MegaMoEExpertSlots.set_replica_plan`)
or lets the in-kernel planner write them from the routing of the call
(:meth:`MegaMoEExpertSlots.plan_and_remap`); the kernels consume them either way.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..jit.megamoe import (
    gen_megamoe_sm90_eplb_module,
    gen_megamoe_sm90_layout_module,
)
from ._jit import _get_jit_module
from .heuristics import select_mxfp4_fused_plan
from .weights import MXFP4_TILE_BYTES

# Positions in the array `megamoe_sm90_eplb_workspace_layout` returns.
_WORKSPACE_FIELDS = (
    "num_words",
    "slot_expert",
    "copies",
    "expert_base",
    "local_counter",
    "copy_list",
    "copy_count",
    "copy_done",
    "remap_done",
    "flags",
    "hist",
    "local_hist",
    "plan_ticket",
    "plan_ready",
    "plan_generation",
    "plan_smem_bytes",
)
_MAX_REDUNDANT_SLOTS = 32
_MAX_RANKS = 32
# Dynamic shared memory an SM90 CTA can request.
_PLAN_SMEM_CAPACITY = 227 * 1024
_RESIDENT_BOUND = 0x7FFFFFFF


def _check_geometry(
    num_ranks: int,
    num_experts: int,
    num_redundant_slots: int,
    num_physical_experts: int,
) -> None:
    if not 0 < num_ranks <= _MAX_RANKS:
        raise ValueError(f"num_ranks must be in [1, {_MAX_RANKS}]")
    if num_experts <= 0 or num_experts % num_ranks:
        raise ValueError("num_experts must divide evenly across ranks")
    if not 0 < num_redundant_slots <= _MAX_REDUNDANT_SLOTS:
        raise ValueError(f"num_redundant_slots must be in [1, {_MAX_REDUNDANT_SLOTS}]")
    if num_physical_experts != num_experts + num_ranks * num_redundant_slots:
        raise ValueError(
            "the symmetric buffer must be sized for the physical slots, "
            f"num_experts + ranks * num_redundant_slots = {num_experts + num_ranks * num_redundant_slots}"
        )


class MegaMoEExpertSlots:
    """Redundant expert slots, their weights and the replica plan of one layer.

    Allocation and rendezvous are collective and must precede CUDA graph
    capture. Every rank must install identical ``slot_expert`` and ``copies``
    tables and call :meth:`remap` and the kernel in matching order.

    Parameters
    ----------
    symm_buffer : MegaMoESymmBuffer
        Buffer of the layer, sized for the physical slots:
        ``symm_buffer.num_experts == num_experts + ranks * num_redundant_slots``.
    num_experts : int
        Logical expert count, divisible by the number of ranks.
    num_redundant_slots : int
        Redundant expert slots per rank, 1 to 32.

    Attributes
    ----------
    l1_weights, l2_weights : torch.Tensor
        Packed MXFP4 weights of this rank's ``H + S`` local slots, the format of
        :func:`transform_mxfp4_weights_for_mega_moe`, in the symmetric arena.
    l1_global_scales, l2_global_scales : torch.Tensor
        FP32 ``[H + S]`` global scales of the local slots, in the arena.
    slot_expert : torch.Tensor
        INT32 ``[ranks, S]``: logical expert held by each redundant slot, -1
        when empty. Identical on every rank.
    copies : torch.Tensor
        INT32 ``[num_experts, ranks, 2]``: per expert, (rank, cumulative token
        blocks) in assignment order, resident rank first, terminated by rank -1.
        A token block goes to the first copy whose bound exceeds its index and
        past the last bound stays on the last copy.
    expert_base : torch.Tensor
        INT32 ``[num_experts]``: ordinal base of this rank's tokens per expert,
        the tokens the ranks below it route to the expert; makes block splits
        consistent across ranks. Any value is correct (only the counts per
        physical slot matter), zero gives rank-local splits.
    copy_list, copy_count, copy_done : torch.Tensor
        INT32 ``[S, 2]`` (slot, expert) pairs of this rank's slots whose expert
        changed, their count, and the ``[S, 2]`` readiness counters of the
        in-kernel weight copy (zero for every listed slot before the launch).
    local_counter : torch.Tensor
        INT32 ``[num_experts]`` ordinal counters of the remap, zero between
        launches.
    plan_generation : torch.Tensor
        INT32 ``[1]`` number of plans :meth:`plan_and_remap` has written,
        advanced on the device by every launch.
    """

    l1_weights: torch.Tensor
    l2_weights: torch.Tensor
    l1_global_scales: torch.Tensor
    l2_global_scales: torch.Tensor

    def __init__(self, symm_buffer, num_experts: int, num_redundant_slots: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        b = symm_buffer
        group = b.group
        num_ranks = group.size()
        _check_geometry(num_ranks, num_experts, num_redundant_slots, b.num_experts)
        self.symm_buffer = b
        self.group = group
        self.num_ranks = num_ranks
        self.rank = group.rank()
        self.num_experts = num_experts
        self.num_redundant_slots = num_redundant_slots
        self.num_resident_experts = num_experts // num_ranks
        self.num_slots_per_rank = self.num_resident_experts + num_redundant_slots

        layout = list(
            _get_jit_module(
                gen_megamoe_sm90_layout_module
            ).megamoe_sm90_eplb_workspace_layout(
                num_ranks, num_experts, num_redundant_slots
            )
        )
        words = dict(zip(_WORKSPACE_FIELDS, layout, strict=True))
        if words["plan_smem_bytes"] > _PLAN_SMEM_CAPACITY:
            raise ValueError(
                f"the in-kernel planner needs {words['plan_smem_bytes']} bytes of shared "
                f"memory for {num_ranks} ranks x {num_experts} experts, above the SM90 capacity"
            )
        self.workspace = symm_mem.empty(
            words["num_words"], dtype=torch.int32, device="cuda"
        )
        self._workspace_handle = symm_mem.rendezvous(self.workspace, group=group)
        self.workspace.zero_()
        E, R, S = num_experts, num_ranks, num_redundant_slots
        self.slot_expert = self._words(words["slot_expert"], (R, S))
        self.copies = self._words(words["copies"], (E, R, 2))
        self.expert_base = self._words(words["expert_base"], (E,))
        self.local_counter = self._words(words["local_counter"], (E,))
        self.copy_list = self._words(words["copy_list"], (S, 2))
        self.copy_count = self._words(words["copy_count"], (1,))
        self.copy_done = self._words(words["copy_done"], (S, 2))
        self.plan_generation = self._words(words["plan_generation"], (1,))

        # The weight arena: one plane per kernel operand, every plane holding
        # the H + S local slots contiguously, 256-byte aligned.
        h, ih = b.hidden, b.intermediate_hidden
        planes: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {
            "l1_weights": ((ih * 2 // 128 * (h // 128), MXFP4_TILE_BYTES), torch.uint8),
            "l2_weights": ((h // 128 * (ih // 128), MXFP4_TILE_BYTES), torch.uint8),
            "l1_global_scales": ((), torch.float32),
            "l2_global_scales": ((), torch.float32),
        }
        self._planes: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name, (shape, dtype) in planes.items():
            bytes_per_slot = math.prod(shape) * dtype.itemsize
            offset = (offset + 255) // 256 * 256
            self._planes[name] = (offset, bytes_per_slot)
            offset += bytes_per_slot * self.num_slots_per_rank
        self.arena = symm_mem.empty(offset, dtype=torch.uint8, device="cuda")
        self._arena_handle = symm_mem.rendezvous(self.arena, group=group)
        self.arena.zero_()
        self.l1_weights = self._plane_view("l1_weights", *planes["l1_weights"])
        self.l2_weights = self._plane_view("l2_weights", *planes["l2_weights"])
        self.l1_global_scales = self._plane_view(
            "l1_global_scales", *planes["l1_global_scales"]
        )
        self.l2_global_scales = self._plane_view(
            "l2_global_scales", *planes["l2_global_scales"]
        )
        self.l1_global_scales.fill_(1.0)
        self.l2_global_scales.fill_(1.0)
        self._arena_ptrs = list(self._arena_handle.buffer_ptrs)
        self._workspace_ptrs = list(self._workspace_handle.buffer_ptrs)
        self._module = _get_jit_module(
            gen_megamoe_sm90_eplb_module, num_ranks, num_experts, num_redundant_slots
        )
        self._reset_plan()
        group.barrier()
        torch.cuda.synchronize()

    def _words(self, offset: int, shape: Tuple[int, ...]) -> torch.Tensor:
        return self.workspace[offset : offset + math.prod(shape)].view(*shape)

    def _plane_view(
        self, name: str, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        offset, bytes_per_slot = self._planes[name]
        flat = self.arena[offset : offset + bytes_per_slot * self.num_slots_per_rank]
        return flat.view(dtype).view(self.num_slots_per_rank, *shape)

    def _fill_resident_copies(self) -> None:
        # One copy per expert, its resident rank, with an unbounded block
        # budget: a remap sends everything resident.
        self.copies.fill_(-1)
        self.copies[:, 0, 0] = (
            torch.arange(self.num_experts, device=self.copies.device, dtype=torch.int32)
            // self.num_resident_experts
        )
        self.copies[:, 0, 1] = _RESIDENT_BOUND

    def _reset_plan(self) -> None:
        self.slot_expert.fill_(-1)
        self._fill_resident_copies()
        self.expert_base.zero_()
        self.local_counter.zero_()
        self.copy_count.zero_()

    def set_resident_weights(
        self,
        l1_weights: torch.Tensor,
        l2_weights: torch.Tensor,
        l1_global_scales: Optional[torch.Tensor] = None,
        l2_global_scales: Optional[torch.Tensor] = None,
    ) -> None:
        """Install this rank's ``H`` resident experts into the arena.

        Takes the output of :func:`transform_mxfp4_weights_for_mega_moe` and
        the optional per-expert global scales (1 when omitted). Collective:
        every redundant slot is emptied and the plan reset to resident-only.
        """
        H = self.num_resident_experts
        for tensor, view in (
            (l1_weights, self.l1_weights),
            (l2_weights, self.l2_weights),
        ):
            if (
                tuple(tensor.shape) != (H, *view.shape[1:])
                or tensor.dtype != torch.uint8
            ):
                raise ValueError(
                    f"mxfp4 weights must have shape {(H, *view.shape[1:])} and dtype uint8"
                )
        for tensor in (l1_global_scales, l2_global_scales):
            if tensor is not None and (
                tensor.shape != (H,) or tensor.dtype != torch.float32
            ):
                raise ValueError(
                    f"global scales must have shape {(H,)} and dtype float32"
                )
        self.l1_weights[:H].copy_(l1_weights)
        self.l2_weights[:H].copy_(l2_weights)
        self.l1_global_scales.fill_(1.0)
        self.l2_global_scales.fill_(1.0)
        if l1_global_scales is not None:
            self.l1_global_scales[:H].copy_(l1_global_scales)
        if l2_global_scales is not None:
            self.l2_global_scales[:H].copy_(l2_global_scales)
        self._reset_plan()
        torch.cuda.synchronize()
        self.group.barrier()

    def set_replica_plan(
        self,
        slot_expert: Any,
        copies: Optional[Any] = None,
        expert_base: Optional[Any] = None,
    ) -> int:
        """Install a replica plan; returns the number of this rank's slots that change.

        Parameters
        ----------
        slot_expert : array-like
            ``[ranks, S]`` logical expert per redundant slot, -1 when empty;
            the same table on every rank.
        copies : array-like, optional
            ``[num_experts, ranks, 2]`` copy table (see :attr:`copies`). Every
            non-resident copy rank must hold the expert in ``slot_expert``.
            Omitted: every expert stays on its resident rank.
        expert_base : array-like, optional
            ``[num_experts]`` ordinal base of this rank (see :attr:`expert_base`).
            Omitted: zero.

        The changed slots are listed in ``copy_list``; call :meth:`copy_weights`
        before the kernel, or pass the slots to the kernel as ``expert_slots``
        so it copies them itself. The routing of the next call must be remapped
        under the new tables (:meth:`remap`).
        """
        R, S, E, H = (
            self.num_ranks,
            self.num_redundant_slots,
            self.num_experts,
            self.num_resident_experts,
        )
        table = torch.as_tensor(slot_expert, dtype=torch.int32).reshape(R, S).cpu()
        if int(table.min()) < -1 or int(table.max()) >= E:
            raise ValueError("slot_expert entries must be -1 or a logical expert")
        for r in range(R):
            held = [int(e) for e in table[r] if int(e) >= 0]
            if len(held) != len(set(held)) or any(e // H == r for e in held):
                raise ValueError(
                    "a rank's redundant slots must hold distinct non-resident experts"
                )
        prev = self.slot_expert.cpu()
        changed = [
            (s, int(table[self.rank, s]))
            for s in range(S)
            if 0 <= int(table[self.rank, s]) != int(prev[self.rank, s])
        ]
        self.slot_expert.copy_(table)
        if copies is None:
            self._fill_resident_copies()
        else:
            self.copies.copy_(
                torch.as_tensor(copies, dtype=torch.int32).reshape(E, R, 2)
            )
        if expert_base is None:
            self.expert_base.zero_()
        else:
            self.expert_base.copy_(
                torch.as_tensor(expert_base, dtype=torch.int32).reshape(E)
            )
        self.local_counter.zero_()
        self.copy_list.zero_()
        if changed:
            self.copy_list[: len(changed)].copy_(
                torch.tensor(changed, dtype=torch.int32)
            )
            self.copy_done[[s for s, _ in changed]] = 0
        self.copy_count.fill_(len(changed))
        return len(changed)

    def block_m(self, num_tokens: int) -> int:
        """Token block size the kernel uses for ``num_tokens``, the unit of the block split."""
        b = self.symm_buffer
        return select_mxfp4_fused_plan(
            b.num_sms,
            self.num_ranks,
            b.num_experts,
            b.num_topk,
            b.hidden,
            b.intermediate_hidden,
            b.num_max_tokens_per_rank,
            num_tokens,
        ).block_m

    def remap(self, num_tokens: int, block_m: Optional[int] = None) -> None:
        """Rewrite the staged logical routing to physical slots under the current tables.

        Operates in place on the first ``num_tokens`` rows of
        ``symm_buffer.topk_idx`` on the current stream; capturable in a CUDA graph.
        ``block_m`` defaults to the kernel's block size for ``num_tokens``.
        """
        b = self.symm_buffer
        self._check_tokens(num_tokens)
        self._module.megamoe_sm90_eplb_remap(
            self.workspace,
            b.topk_idx,
            num_tokens,
            self.block_m(num_tokens) if block_m is None else block_m,
            b.num_sms,
        )

    def plan_and_remap(
        self, num_tokens: int, block_m: Optional[int] = None, open_gate_blocks: int = 0
    ) -> None:
        """Plan from the staged logical routing of every rank, then remap it in place.

        One launch on the current stream, capturable in a CUDA graph: the ranks
        exchange their expert histograms through the workspace and each runs
        the same deterministic planner (MegaFlux's load-only block-granular
        greedy), writing :attr:`slot_expert`, :attr:`copies`, :attr:`expert_base`
        and the copy list of the slots whose expert changed. Every rank must call
        it the same number of times. The slots in ``copy_list`` are then filled by
        :meth:`copy_weights` or by the kernel run with the slots as ``expert_slots``.

        Parameters
        ----------
        num_tokens : int
            Staged tokens on this rank, equal on every rank.
        block_m : int, optional
            Token block size of the split; defaults to the kernel's block size
            for ``num_tokens``.
        open_gate_blocks : int
            Smallest allocation, in blocks, that justifies copying weights into
            an empty slot. 0 opens a slot for any allocation.
        """
        b = self.symm_buffer
        self._check_tokens(num_tokens)
        if open_gate_blocks < 0:
            raise ValueError("open_gate_blocks must be non-negative")
        self._module.megamoe_sm90_eplb_plan_and_remap(
            self.workspace,
            self._workspace_ptrs,
            self.rank,
            b.topk_idx,
            num_tokens,
            self.block_m(num_tokens) if block_m is None else block_m,
            open_gate_blocks,
            b.num_sms,
        )

    def copy_weights(self) -> None:
        """Copy the slots in ``copy_list`` from their resident ranks with a separate kernel.

        Runs on the current stream and clears ``copy_count`` afterwards, so a
        later launch with the slots as ``expert_slots`` copies nothing.
        """
        self._module.megamoe_sm90_eplb_copy_weights(
            self.workspace,
            self.arena,
            self._arena_ptrs,
            self.rank,
            self._plane_specs(
                ("l1_weights", "l1_global_scales", "l2_weights", "l2_global_scales")
            ),
        )
        self.copy_count.zero_()

    def _check_tokens(self, num_tokens: int) -> None:
        b = self.symm_buffer
        if b.buffer is None:
            raise ValueError("symmetric buffer has been destroyed")
        if not 0 < num_tokens <= b.num_max_tokens_per_rank:
            raise ValueError("num_tokens must fit the buffer capacity")

    def _plane_specs(self, names: Sequence[str]) -> List[int]:
        return [v for name in names for v in self._planes[name]]

    def _in_kernel_copy_args(self) -> Tuple[Any, ...]:
        return (
            self.workspace,
            self.arena,
            self._arena_ptrs,
            self.num_resident_experts,
            self._plane_specs(("l1_weights", "l1_global_scales")),
            self._plane_specs(("l2_weights", "l2_global_scales")),
        )

    def _check_kernel_operands(
        self, symm_buffer, l1_weights, l2_weights, l1_global_scales, l2_global_scales
    ) -> None:
        if symm_buffer is not self.symm_buffer:
            raise ValueError("expert_slots belong to another symmetric buffer")
        for tensor, view in (
            (l1_weights, self.l1_weights),
            (l2_weights, self.l2_weights),
            (l1_global_scales, self.l1_global_scales),
            (l2_global_scales, self.l2_global_scales),
        ):
            if tensor is not None and tensor.data_ptr() != view.data_ptr():
                raise ValueError(
                    "with expert_slots the weights and global scales must be the slots' arena views"
                )

    def destroy(self) -> None:
        """Release this rank's views and symmetric-memory handles."""
        self._workspace_handle = self._arena_handle = None
        self.workspace = self.arena = None
        self.slot_expert = self.copies = self.expert_base = self.local_counter = None
        self.copy_list = self.copy_count = self.copy_done = self.plan_generation = None
        self.l1_weights = self.l2_weights = self.l1_global_scales = (
            self.l2_global_scales
        ) = None
