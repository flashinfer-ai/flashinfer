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

Host of the Cake TP MoE finalize + RMSNorm kernels behind
``flashinfer.comm.tp_moe_finalize_rmsnorm(..., backend="cake")``: one
:class:`CakeTpMoeFinalize` per tensor-parallel group, initialized collectively
by ``init_tp_moe_finalize_rmsnorm``.

The package ships five program kinds per world size (``cake_jit.MODULES``):
``eo`` and ``onehop`` launch one cluster per token, ``rsag`` one CTA per token,
``token1`` one cluster per token group of the world, and the persistent
``tokown`` program a co-resident grid.  Every program exchanges its rows
through Lamport rings in symmetric memory: a buffer per program kind whose size
is a per-slot byte count times a slot count (``workspace_rule`` of the kind's
records), sentinel-filled once, mapped on every rank and through NVLink
multicast, with per-rank address tables and rank-local records beside it.  The
host provisions them once per group from ``torch.distributed._symmetric_memory``
(the mechanism the CuTe path uses), agrees the co-resident CTA capacity of
every program across the ranks (the smallest), and routes each call by its
token count through the registry's windows (``cake_jit.selected_module``) to a
program whose grid fits that capacity.  A launch enqueues one kernel on the
current stream with no host synchronization, so it captures into CUDA graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import tvm_ffi

from . import cake_jit

logger = logging.getLogger(__name__)

HIDDEN = 4096
TOPK = 8
#: Byte alignment of every region of the symmetric workspace.
ALIGNMENT = 256
#: The persistent program's CTA cap: the slot count its rings are provisioned
#: with and the bound of its grid, the kernel tests' and the routing
#: measurements' value; the agreed co-residency caps it further.
TOKOWN_MAX_CTAS = 160
#: Storage dtype of a rank-local record the kernel binds as a typed pointer.
_VIEW_DTYPES = {
    "gen": torch.int32,
    "part": torch.float32,
    "ctr": torch.uint32,
    "ctl": torch.int32,
}


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


def _signed32(word: int) -> int:
    """The int32 reading of a u32 fill word (``torch.Tensor.fill_`` on an int32 view)."""
    word &= 0xFFFFFFFF
    return word - (1 << 32) if word >= 1 << 31 else word


def _vote(ok: bool, group: Any, device: torch.device) -> bool:
    """Every rank's ``ok`` reduced with AND; every rank calls it (collective, synchronizing)."""
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(int(flag.item()))


def _agreed_minimum(values: list[int], group: Any, device: torch.device) -> list[int]:
    """Per position the smallest value over the ranks (collective)."""
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return [int(value) for value in tensor.tolist()]


def _same_everywhere(value: int, group: Any, device: torch.device) -> bool:
    """Whether every rank passes the same ``value`` (collective)."""
    import torch.distributed as dist

    tensor = torch.tensor([value, -value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    largest, smallest = int(tensor[0].item()), -int(tensor[1].item())
    return largest == smallest == value


def kind_slots(arch: str, route: str) -> dict[str, int]:
    """Per program kind of ``route`` on ``arch`` the slot count its workspace is provisioned with: the largest
    bounded window edge of the kind's records (every call the windows route to the kind fits), the persistent
    program excepted (its slots are its CTA cap)."""
    slots: dict[str, int] = {}
    for name in cake_jit.route_modules(arch, route):
        record = cake_jit.MODULES[name]
        kind = record["program"]
        if kind == "tokown":
            continue
        bounds = [
            int(at_most)
            for _above, at_most in record["select_rule"]["windows"].get(arch, ())
            if at_most is not None
        ]
        if not bounds:
            raise RuntimeError(
                f"Cake TP MoE finalize record {name} ({kind}) has no bounded token window on {arch}; "
                "its workspace has no slot count"
            )
        slots[kind] = max(slots.get(kind, 0), *bounds)
    return slots


@dataclass
class _KindWorkspace:
    """One program kind's share of the group's workspace."""

    kind: str
    slots: int
    rule: dict[str, Any]
    logical: dict[str, str]
    regions: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # symmetric buffer -> (offset, bytes)
    locals: dict[str, torch.Tensor] = field(
        default_factory=dict
    )  # local buffer -> this rank's record
    values: dict[str, Any] = field(
        default_factory=dict
    )  # the launch values the workspace fixes, by logical name


class CakeTpMoeFinalize:
    """The Cake finalize of one tensor-parallel group: the routed programs of its (architecture, world size), the
    agreed capacities, the workspace and the launch.  Construct through :meth:`initialize`."""

    def __init__(
        self,
        *,
        group: Any,
        device: torch.device,
        arch: str,
        world: int,
        rank: int,
        route: str,
        capacity: dict[str, int],
        kinds: dict[str, _KindWorkspace],
        buffer: torch.Tensor,
        handle: Any,
    ) -> None:
        self.group = group
        self.device = device
        self.arch = arch
        self.world = world
        self.rank = rank
        self.route = route
        self.capacity = capacity
        self.kinds = kinds
        self.buffer = buffer
        self.handle = handle
        self.last: tuple[str, tuple[int, int, int]] | None = None
        self._modules = {name: cake_jit.load(name, arch) for name in capacity}

    # ------------------------------------------------------------------ initialization
    @classmethod
    def initialize(
        cls, group: Any, *, device: torch.device | None = None
    ) -> CakeTpMoeFinalize | None:
        """Collectively set the group up; ``None`` (on every rank, with one warning on the rank that found the
        reason) when the Cake kernels cannot serve it: a world size the package does not ship, a device outside
        the exported architectures or without a route of record there, ranks on different architectures, a
        library that does not load, a workspace that does not allocate, or no NVLink multicast mapping."""
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if world not in cake_jit.WORLDS:
            logger.warning(
                "Cake TP MoE finalize ships world sizes %s; the group has %d ranks.",
                cake_jit.WORLDS,
                world,
            )
            return None
        route = cake_jit.route_key(world)
        capability = tuple(torch.cuda.get_device_capability(device))
        arch = cake_jit.arch_for_capability(capability)
        local_ok = arch is not None and cake_jit.route_available(arch, route)
        if not _vote(local_ok, group, device):
            if not local_ok:
                logger.warning(
                    "Cake TP MoE finalize has no programs for capability %s at world size %d (route %s).",
                    capability,
                    world,
                    route,
                )
            return None
        assert arch is not None
        if not _same_everywhere(capability[0] * 10 + capability[1], group, device):
            logger.warning(
                "Cake TP MoE finalize needs every rank on the same architecture; rank %d is %s.",
                rank,
                arch,
            )
            return None
        names = list(cake_jit.route_modules(arch, route))
        capacities: list[int] = []
        error: Exception | None = None
        try:
            for name in names:
                capacities.append(
                    int(cake_jit.cluster_capacity(name, arch, device.index))
                )
        except Exception as exc:  # a library that does not build or load, an occupancy query that fails
            error = exc
        if not _vote(error is None, group, device):
            if error is not None:
                logger.warning(
                    "Cake TP MoE finalize libraries are unavailable: %s", error
                )
            return None
        capacity = dict(
            zip(names, _agreed_minimum(capacities, group, device), strict=True)
        )
        if not _vote(all(value > 0 for value in capacity.values()), group, device):
            logger.warning(
                "Cake TP MoE finalize: a program is not resident on this device: %s",
                capacity,
            )
            return None
        slots = kind_slots(arch, route)
        tokown = [
            name for name in names if cake_jit.MODULES[name]["program"] == "tokown"
        ]
        if tokown:
            slots["tokown"] = min(TOKOWN_MAX_CTAS, *(capacity[name] for name in tokown))
            for name in tokown:
                capacity[name] = slots["tokown"]
        kinds, total = cls._layout(names, slots)
        buffer = None
        try:
            buffer = symm_mem.empty((total,), dtype=torch.uint8, device=device)
            cls._fill(buffer, kinds, device)
        except RuntimeError as exc:
            error = exc
        if not _vote(error is None, group, device):
            if error is not None:
                logger.warning(
                    "Cake TP MoE finalize workspace allocation failed: %s", error
                )
            return None
        assert buffer is not None
        torch.cuda.synchronize(device)
        handle = symm_mem.rendezvous(buffer, group=group)
        multicast = int(getattr(handle, "multicast_ptr", 0) or 0)
        if not _vote(multicast != 0, group, device):
            if multicast == 0:
                logger.warning(
                    "Cake TP MoE finalize requires an NVLink multicast mapping of its workspace."
                )
            return None
        try:
            peers = [int(pointer) for pointer in handle.buffer_ptrs]
        except (AttributeError, TypeError):
            peers = [
                handle.get_buffer(r, (total,), torch.uint8).data_ptr()
                for r in range(world)
            ]
        cls._bind(kinds, peers=peers, multicast=multicast, rank=rank, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        return cls(
            group=group,
            device=device,
            arch=arch,
            world=world,
            rank=rank,
            route=route,
            capacity=capacity,
            kinds=kinds,
            buffer=buffer,
            handle=handle,
        )

    @staticmethod
    def _layout(
        names: list[str], slots: dict[str, int]
    ) -> tuple[dict[str, _KindWorkspace], int]:
        """The workspace layout: per program kind (first record's order) its buffers in declaration order, the
        symmetric ones as aligned regions of one allocation, the local ones as this rank's records."""
        kinds: dict[str, _KindWorkspace] = {}
        offset = 0
        for name in names:
            record = cake_jit.MODULES[name]
            kind = record["program"]
            rule = record["workspace_rule"]
            if kind in kinds:
                if kinds[kind].rule != rule:
                    raise RuntimeError(
                        f"Cake TP MoE finalize records of kind {kind} declare different workspaces"
                    )
                continue
            space = kinds[kind] = _KindWorkspace(
                kind, slots[kind], rule, dict(record["logical"])
            )
            for buffer in rule["buffers"]:
                nbytes = int(buffer["bytes_per_slot"]) * space.slots
                if nbytes % 4:
                    raise RuntimeError(
                        f"Cake TP MoE finalize buffer {kind}.{buffer['name']} is not whole words"
                    )
                if buffer["placement"] == "symmetric":
                    offset = _round_up(offset, ALIGNMENT)
                    space.regions[buffer["name"]] = (offset, nbytes)
                    offset += nbytes
                elif buffer["placement"] == "local":
                    space.locals[buffer["name"]] = torch.empty(
                        0
                    )  # allocated by _fill on the device
                else:
                    raise RuntimeError(
                        f"unknown Cake TP MoE finalize buffer placement {buffer['placement']!r}"
                    )
        return kinds, _round_up(offset, ALIGNMENT)

    @staticmethod
    def _fill(
        buffer: torch.Tensor, kinds: dict[str, _KindWorkspace], device: torch.device
    ) -> None:
        """Sentinel-fill every symmetric region and allocate this rank's local records, as the kernels declare."""
        for space in kinds.values():
            for declared in space.rule["buffers"]:
                name = declared["name"]
                fill = declared["fill"]
                if declared["placement"] == "symmetric":
                    offset, nbytes = space.regions[name]
                    words = buffer[offset : offset + nbytes].view(torch.int32)
                    words.fill_(_signed32(int(fill)) if fill is not None else 0)
                else:
                    view = declared["view"]
                    dtype = _VIEW_DTYPES.get(view, torch.int32)
                    nbytes = int(declared["bytes_per_slot"]) * space.slots
                    local = torch.zeros(
                        nbytes // torch.empty((), dtype=dtype).element_size(),
                        dtype=dtype,
                        device=device,
                    )
                    if fill:
                        local.view(torch.int32).fill_(_signed32(int(fill)))
                    space.locals[name] = local

    @staticmethod
    def _bind(
        kinds: dict[str, _KindWorkspace],
        *,
        peers: list[int],
        multicast: int,
        rank: int,
        device: torch.device,
    ) -> None:
        """The launch values the workspace fixes per kind, by logical name: the rank, the slot count, each local
        record, each multicast address and the device address tables (every rank's copy of a symmetric buffer in
        rank order, this rank's local record)."""
        for space in kinds.values():
            logical = space.logical
            values: dict[str, Any] = {
                logical["rank"]: rank,
                logical[space.rule["slot_parameter"]]: space.slots,
            }
            for declared in space.rule["buffers"]:
                name = declared["name"]
                if declared["view"] is not None:
                    values[logical[declared["view"]]] = space.locals[name]
                if declared["multicast"] is not None:
                    offset, _nbytes = space.regions[name]
                    values[logical[declared["multicast"]]] = multicast + offset
            for table in space.rule["tables"]:
                addresses: list[int] = []
                for entry in table["entries"]:
                    if entry in space.regions:
                        offset, _nbytes = space.regions[entry]
                        addresses.extend(peer + offset for peer in peers)
                    else:
                        addresses.append(space.locals[entry].data_ptr())
                values[logical[table["parameter"]]] = torch.tensor(
                    addresses, dtype=torch.int64, device=device
                )
            space.values = values

    # ------------------------------------------------------------------ the calls
    def route_for(self, n_tok: int) -> tuple[str, tuple[int, int, int]]:
        """The record and grid of a call of ``n_tok`` tokens; ``LookupError`` names the reason when none serves it
        (no window admits the count, or the admitted program's grid is not co-resident at the agreed capacity)."""
        try:
            name = cake_jit.selected_module(self.arch, self.route, n_tok)
        except (NotImplementedError, ValueError) as exc:
            raise LookupError(str(exc)) from None
        record = cake_jit.MODULES[name]
        values = self.kinds[record["program"]].values
        try:
            grid = cake_jit.launch_grid(
                record,
                n_tok,
                device_index=self.device.index,
                values=values,
                capacity=self.capacity[name],
            )
        except (RuntimeError, ValueError) as exc:
            raise LookupError(str(exc)) from None
        return name, grid

    def select(self, n_tok: int) -> tuple[str, tuple[int, int, int]] | None:
        """:meth:`route_for`, or ``None`` when no program serves the call."""
        try:
            return self.route_for(n_tok)
        except LookupError:
            return None

    def run(
        self,
        name: str,
        grid: tuple[int, int, int],
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        eps: float,
        gain_center: float,
        weight_before_norm: bool,
    ) -> None:
        """Launch record ``name`` at ``grid`` on the current stream (no host synchronization; graph-capturable)."""
        record = cake_jit.MODULES[name]
        n_tok = int(expanded_idx_to_permuted_idx.shape[0])
        values = {
            **self.kinds[record["program"]].values,
            "gemm2_permuted": gemm2_permuted,
            "row_stride": int(gemm2_permuted.stride(0)),
            "rows": int(gemm2_permuted.shape[0]),
            "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
            "expert_weights_words": expert_weights.view(torch.uint32),
            "weights_fp32": int(expert_weights.dtype == torch.float32),
            "norm_weight": norm_weight,
            "output": output,
            "n_tok": n_tok,
            "eps": float(eps),
            "gain_center": float(gain_center),
            "weight_before_norm": int(bool(weight_before_norm)),
        }
        args = cake_jit.launch_args(record, values, grid)
        entry = getattr(self._modules[name], record["ffi_entry"])
        with torch.cuda.device(self.device), tvm_ffi.use_torch_stream():
            entry(*args)
        self.last = (name, (int(grid[0]), int(grid[1]), int(grid[2])))
