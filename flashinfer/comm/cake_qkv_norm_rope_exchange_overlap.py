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

Cake backend of :class:`flashinfer.comm.QkvNormRopeExchangeOverlap`: the fused
QKV projection + per-head RMSNorm + RoPE producer that stores every head band
straight into the destination rank's symmetric receive slot (SM90 WGMMA, SM100
/ SM103 / SM107 tcgen05), the one-CTA readiness wait kernel and the two-phase NVLink
barrier -- as a separate kernel, or folded into the producer as a rendezvous
warp (one extra warp per CTA on SM100 / SM103, the otherwise idle TMA warp on
SM90) that runs the rendezvous and releases the stores -- driven from Python
one launcher call per stage.

Launch path (``fold_barrier``; the keyword, else ``FLASHINFER_QKV_EXCHANGE_FOLD_BARRIER``
= ``0`` / ``1`` / ``auto``, else the architecture default
``loader.FOLD_BARRIER_DEFAULT_BY_ARCH``: folded on SM90, SM100, SM103 and SM107;
``True`` fails closed on a route table without folded producers).  Both paths
compute and store identical bits.  Per call (current stream = producer stream):

1. wait for the slot's previous exchange (``exchange_done[slot]``; eager only),
2. folded (two launches): record the fork event, then the producer build of
   this call's rendezvous phase (``producer_fold<phase>``, persistent grid
   ``min(num_tiles, sm_count - 2)``; the rendezvous warp -- one extra warp per
   CTA on SM100 / SM103, the idle TMA warp on SM90 -- of CTA 0 runs the
   cross-rank barrier and publishes ``ready_target`` into this rank's arrive
   flag, every CTA acquires it once before its first store);
   separate barrier (three launches): the barrier kernel of the phase, the
   fork event, then the plain producer.  Either producer publishes
   ``ready_target`` into every destination's readiness pad per chunk,
3. consumer stream: wait the fork event, wait kernel over this call's
   ``W * num_words`` words, record ``exchange_done[slot]``,
4. the current stream waits ``exchange_done[slot]`` so the outputs are
   consumable in stream order.

Producer build (``loader.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``, resolved once
per session by :func:`resolve_producer_variant`: the default of the session's
architecture and world size unless ``FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT``
overrides it -- ``auto``, ``shipped`` (the baseline build) or a delivered build
name; rank-uniform, checked collectively): SM100 groups of 8 ranks run the
``bulk_ow256`` build (wide operand loads, bulk row stores), SM100 groups of 4
ranks and SM103 groups of 8 or 4 ranks the ``ow256`` build (wide operand loads),
SM107 groups of 4 ranks the ``bulk`` build (bulk row stores), groups of 2 ranks
and SM90 the baseline build.  Every build stores identical bits; the wide-load
builds require 32-byte-aligned ``cos_sin`` and norm-weight bases
(:meth:`CakeQkvExchangeSession._validate` refuses a misaligned view and names
the opt-out).  Wherever a variant is the default, the baseline build of the
same tile width is delivered too, so the opt-out always launches a delivered
program; a build that is not delivered for the session's tile width raises and
names the delivered ones.

Group scaling (``plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD``, resolved once per
session by :func:`resolve_group_m_scale`: the default of the architecture and
world size unless ``FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE`` overrides it --
``auto``, ``1``, ``2`` or ``4``; rank-uniform): the ``grouped`` rows with a
hidden size of at least ``plans.GROUP_M_SCALE_MIN_K`` take ``group_m`` times the
scale (the tile order changes, the readiness words and the stored bits do not).

Row-count gates (``plans.SHAPE_GATE_DEFAULT_BY_ARCH`` with the (architecture,
world size) cells of ``plans.SHAPE_GATE_DEFAULT_BY_ARCH_WORLD`` on top, resolved
once per session by :func:`resolve_shape_gates`: the default of the architecture
and world size unless ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` overrides it --
``auto``, ``off`` or ``gate=value,...`` with ``0`` = never and ``all`` = every row
for the small-row word gate; rank-uniform, checked collectively): a call with
``batch * s_local <= words_capacity_small_m`` rows publishes one readiness word
per destination (``num_words = 1``; every call on SM90 with 8 ranks); a call
with ``batch * s_local >= reserved_sms_large_m`` rows runs the persistent
producer grid over every SM (``reserved_sms = 0`` instead of
``plans.RESERVED_SMS``); a call with ``batch * s_local >= per_dest_large_m`` rows
launches the producer once per destination run of its tile table
(``plans.dest_parts``; on the folded path the first part carries the rendezvous
and the later parts are the plain producer, stream-ordered behind it; on the
barrier path every part is the plain producer) instead of one persistent grid;
a 256-column call with ``batch * s_local >= pair_off_large_m`` rows runs the
single-CTA producer instead of the 2-CTA pair tile (tile order unchanged; the
single-CTA ``block_n = 256`` producer must be delivered for the architecture --
no delivered Blackwell route ships it today, so the opt-in refuses at
construction when a 256-column call can reach the gate (``max_tokens >= N``),
naming the gate; SM90 builds no pair tile and refuses the gate).  SM100 / SM103
ship ``4096`` / off / ``16384`` / off; SM90 ships ``4096`` / ``16384`` / off /
off, with the word gate on every row (``all``) for an 8-rank group; the
pair-tile gate ships nowhere.  The first three
gates change the host plan only -- never the tile table or the stored bits; the
pair-tile gate changes the producer program and the tile table (no lockstep
pads) of the rows it selects; no gate changes the stored bits -- and
``describe()`` reports them (``shape_gates``) beside the resolved build
(``producer_variant``) and scale (``group_m_scale``), with every prepared row's
verdict (``prepared_row_plans``, including its ``reserved_sms`` and
``part_grids``).

Eager calls publish a monotonic epoch; a call issued while the current stream
is capturing publishes the sentinel ``0xFFFFFFFF``, clears its slot's words
with one stream memset after the wait kernel, uses capture-local events and
never waits on an eager predecessor (the capturing caller orders its replays).
Nothing allocates, synchronises, builds or runs a collective inside a capture:
the session prepares its symmetric slots, readiness pad, barrier flags, JIT
programs and the tile plan of the sample rows eagerly, and refuses a row count
it has not seen (or a row count above ``max_tokens``) inside a capture.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import tvm_ffi

from ..jit import cake_qkv_norm_rope_exchange_overlap as loader
from . import cake_qkv_exchange_plan as plans

CAPTURE_READY_TARGET = 0xFFFFFFFF
EAGER_READY_TARGET_MAX = 0xFFFFFFFE
# torch's default CUDA symmetric-memory backend provides the peer buffer
# pointers, ``memset32_`` and the rendezvous this backend needs.
SUPPORTED_SYMMETRIC_MEMORY_BACKENDS = ("CUDA",)
# Launch-path override read when the ``fold_barrier`` keyword is ``None``: ``0`` = the separate barrier kernel,
# ``1`` = the producer with the folded rendezvous, ``auto`` / unset = the architecture default of the loader.
FOLD_BARRIER_ENV = "FLASHINFER_QKV_EXCHANGE_FOLD_BARRIER"
_FOLD_BARRIER_ENV_VALUES: dict[str, Optional[bool]] = {
    "": None,
    "auto": None,
    "0": False,
    "1": True,
}


def symmetric_memory_backend(device: torch.device) -> str:
    """Upper-case name of torch's symmetric-memory backend for ``device`` (empty when none)."""

    name = symm_mem.get_backend(device)
    return "" if name is None else str(name).upper()


def supports_symmetric_memory_backend(device: torch.device) -> bool:
    """Whether torch's symmetric-memory backend for ``device`` can serve this backend (unset is selectable)."""

    return symmetric_memory_backend(device) in (
        "",
        *SUPPORTED_SYMMETRIC_MEMORY_BACKENDS,
    )


def fold_barrier_from_env() -> Optional[bool]:
    """The launch-path override of :data:`FOLD_BARRIER_ENV` (``None`` when unset or ``auto``)."""

    raw = os.environ.get(FOLD_BARRIER_ENV, "")
    try:
        return _FOLD_BARRIER_ENV_VALUES[raw.strip().lower()]
    except KeyError:
        raise ValueError(
            f"{FOLD_BARRIER_ENV} must be 0, 1 or auto, got {raw!r}"
        ) from None


# Row-count gate override read once per session: ``auto`` (or unset) = the (architecture, world size) default of
# ``plans.default_shape_gates``; ``off`` / ``0`` = every gate off; ``gate=value[,gate=value]`` with ``gate`` in
# ``plans.SHAPE_GATE_NAMES`` and ``value`` an int >= 0 (``0`` = never), ``auto`` = that gate's default or, for the
# small-row word gate, ``all`` = every row.
SHAPE_GATES_ENV = "FLASHINFER_QKV_EXCHANGE_SHAPE_GATES"


def shape_gates_from_env() -> Optional[dict[str, Optional[int]]]:
    """The gate override of :data:`SHAPE_GATES_ENV`: ``None`` when unset or ``auto``; otherwise one entry per gate
    (``None`` = keep the architecture default of that gate)."""

    raw = os.environ.get(SHAPE_GATES_ENV, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    if raw.lower() in ("off", "0", "none"):
        return {name: plans.SHAPE_GATE_OFF for name in plans.SHAPE_GATE_NAMES}
    override: dict[str, Optional[int]] = {name: None for name in plans.SHAPE_GATE_NAMES}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in plans.SHAPE_GATE_NAMES:
            raise ValueError(
                f"{SHAPE_GATES_ENV} must be auto, off or gate=value pairs over {plans.SHAPE_GATE_NAMES}, got {raw!r}"
            )
        if value.lower() == "auto":
            override[name] = None
        elif value.lower() == "all":
            if name not in plans.SHAPE_GATE_ALL_ROWS_NAMES:
                raise ValueError(
                    f"{SHAPE_GATES_ENV}: {name}=all is not a spelling of this gate ('all' = every row for "
                    f"{plans.SHAPE_GATE_ALL_ROWS_NAMES}; a large-row gate that never fires is {name}=0)"
                )
            override[name] = plans.SHAPE_GATE_ALL_ROWS
        elif value.isdigit():
            if int(value) > plans.SHAPE_GATE_ALL_ROWS:
                raise ValueError(
                    f"{SHAPE_GATES_ENV}: {name}={value} is above SHAPE_GATE_ALL_ROWS "
                    f"({plans.SHAPE_GATE_ALL_ROWS}); every row is spelled {name}=all (the "
                    f"small-row word gate only), a gate that never fires {name}=0"
                )
            override[name] = int(value)
        else:
            raise ValueError(
                f"{SHAPE_GATES_ENV}: {name} must be an int >= 0 or `auto` (and `all` for "
                f"{', '.join(plans.SHAPE_GATE_ALL_ROWS_NAMES)}), got {value!r}"
            )
    return override


def resolve_shape_gates(arch: str, world_size: int) -> tuple[dict[str, int], str]:
    """The row-count gates of a session on ``arch`` with ``world_size`` ranks: the (architecture, world size) default
    (``plans.default_shape_gates``) with :data:`SHAPE_GATES_ENV` applied on top.  Returns ``(gates, source)`` with
    ``source`` ``"default"`` or ``"env"``."""

    gates = plans.default_shape_gates(arch, world_size=world_size)
    override = shape_gates_from_env()
    if override is None:
        return plans.check_shape_gates(gates), "default"
    for name, value in override.items():
        if value is not None:
            gates[name] = int(value)
    checked = plans.check_shape_gates(gates)
    if arch not in plans.PAIR_ARCHES:
        for name in plans.SHAPE_GATE_BN:
            if checked[name]:
                raise ValueError(
                    f"{SHAPE_GATES_ENV}: {name}={checked[name]} is refused on {arch}: the architecture builds no "
                    f"2-CTA pair producer (pair architectures: {plans.PAIR_ARCHES}), so there is no pair tile to "
                    "switch off; drop the gate (0 or auto)"
                )
    return checked, "env"


def resolve_fold_barrier(
    arch: str, requested: Optional[bool] = None
) -> tuple[bool, str]:
    """Resolve the launch path on ``arch``: the explicit ``requested`` value, else :data:`FOLD_BARRIER_ENV`,
    else the architecture default.  Returns ``(fold_barrier, source)`` with ``source`` in ``"keyword"``,
    ``"env"``, ``"default"``.  ``True`` fails closed where the route table delivers no folded producer (no
    delivered architecture; the check guards the table): the message names the architecture default and the
    opt-out, since the barrier path is always delivered."""

    if requested is not None and not isinstance(requested, bool):
        raise ValueError(f"fold_barrier must be a bool or None, got {requested!r}")
    if requested is not None:
        fold, source = requested, "keyword"
    else:
        override = fold_barrier_from_env()
        if override is not None:
            fold, source = override, "env"
        else:
            fold, source = loader.default_fold_barrier(arch), "default"
    if fold and not loader.fold_producers_delivered(arch):
        raise ValueError(
            f"fold_barrier=True ({source}): no producer program with the folded rendezvous is delivered for "
            f"{arch} (its route table carries no fold key; the architecture default is "
            f"fold_barrier={loader.default_fold_barrier(arch)}) -- pass fold_barrier=False to select the "
            "separate barrier kernel, which is delivered on every architecture"
        )
    return fold, source


# Producer build override read once per session: ``auto`` (or unset) = the default of the session's architecture
# and world size (``loader.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``); ``shipped`` = the baseline build; a name in
# ``loader.PRODUCER_VARIANTS`` = that build.
PRODUCER_VARIANT_ENV = "FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT"
PRODUCER_VARIANT_SHIPPED = "shipped"


def producer_variant_from_env() -> Optional[str]:
    """The producer build override of :data:`PRODUCER_VARIANT_ENV`: ``None`` when unset or ``auto``; ``""`` for
    ``shipped``; otherwise a name of ``loader.PRODUCER_VARIANTS``."""

    raw = os.environ.get(PRODUCER_VARIANT_ENV, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    if raw.lower() == PRODUCER_VARIANT_SHIPPED:
        return loader.SHIPPED_VARIANT
    if raw in loader.PRODUCER_VARIANTS and raw != loader.SHIPPED_VARIANT:
        return raw
    names = tuple(name for name in loader.PRODUCER_VARIANTS if name)
    raise ValueError(
        f"{PRODUCER_VARIANT_ENV} must be auto, {PRODUCER_VARIANT_SHIPPED} or one of {names}, got {raw!r}"
    )


def resolve_producer_variant(
    arch: str, world_size: int, block_n: int
) -> tuple[str, str]:
    """The producer build of a session on ``arch`` with ``world_size`` ranks and tile width ``block_n``: the
    (architecture, world size) default unless :data:`PRODUCER_VARIANT_ENV` overrides it.  Returns ``(variant,
    source)`` with ``source`` ``"default"`` or ``"env"``.  Fails closed when the build is delivered for neither pair
    flag of the tile width, naming the environment variable, the ``shipped`` opt-out and the delivered builds."""

    override = producer_variant_from_env()
    if override is None:
        variant, source = loader.default_producer_variant(arch, world_size), "default"
    else:
        variant, source = override, "env"
    delivered = {
        pair: loader.delivered_producer_variants(arch, block_n, pair)
        for pair in (False, True)
    }
    if not any(variant in names for names in delivered.values()):
        raise ValueError(
            f"producer build {variant or PRODUCER_VARIANT_SHIPPED!r} ({source}) is not delivered for {arch} at "
            f"block_n={block_n} (delivered: 1-CTA {delivered[False]}, pair {delivered[True]}; '' is the baseline "
            f"build) -- set {PRODUCER_VARIANT_ENV}={PRODUCER_VARIANT_SHIPPED} or one of the delivered names on "
            "every rank"
        )
    return variant, source


# Group-scaling override read once per session: ``auto`` (or unset) = the default of the session's architecture
# and world size (``plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD``); ``1`` / ``2`` / ``4`` = that scale (``1`` = the
# baseline plan).
GROUP_M_SCALE_ENV = "FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE"


def group_m_scale_from_env() -> Optional[int]:
    """The group-scaling override of :data:`GROUP_M_SCALE_ENV`: ``None`` when unset or ``auto``; otherwise one of
    ``plans.GROUP_M_SCALES``."""

    raw = os.environ.get(GROUP_M_SCALE_ENV, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    if raw.isdigit() and int(raw) in plans.GROUP_M_SCALES:
        return int(raw)
    raise ValueError(
        f"{GROUP_M_SCALE_ENV} must be auto or one of {plans.GROUP_M_SCALES}, got {raw!r}"
    )


def resolve_group_m_scale(arch: str, world_size: int) -> tuple[int, str]:
    """The ``group_m_scale`` of a session on ``arch`` with ``world_size`` ranks: the (architecture, world size)
    default unless :data:`GROUP_M_SCALE_ENV` overrides it.  Returns ``(scale, source)`` with ``source``
    ``"default"`` or ``"env"``."""

    override = group_m_scale_from_env()
    if override is None:
        return plans.default_group_m_scale(arch, world_size), "default"
    return override, "env"


def device_arch(device: torch.device) -> Optional[str]:
    """Exported architecture of ``device`` or ``None`` when the Cake programs do not target it."""

    index = device.index if device.index is not None else torch.cuda.current_device()
    return loader.device_facts(int(index)).arch


def _group_name(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError("the process group does not expose a non-empty group_name")
    return name


def _peer_pointers(
    handle: Any, world_size: int, numel: int, dtype: torch.dtype
) -> list[int]:
    return [
        int(handle.get_buffer(peer, (int(numel),), dtype, 0).data_ptr())
        for peer in range(int(world_size))
    ]


def _rendezvous(
    tensor: torch.Tensor, *, group_name: str, rank: int, world_size: int, what: str
):
    handle = symm_mem.rendezvous(tensor, group=group_name)
    if int(handle.rank) != rank or int(handle.world_size) != world_size:
        raise RuntimeError(
            f"{what}: symmetric-memory topology does not match the process group"
        )
    if int(handle.buffer_size) < tensor.numel() * tensor.element_size():
        raise RuntimeError(
            f"{what}: symmetric-memory rendezvous returned an undersized handle"
        )
    return handle


@dataclass
class _RowPlan:
    policy: plans.Policy
    plan: plans.TilePlan
    tiles: torch.Tensor
    grid: int
    program: str
    # Both rendezvous phases of the folded producer of this tile width (empty where none is delivered).
    fold_programs: tuple[str, ...]
    # The row-count gate verdict of this row and the producer launches it selects: ``(start, stop, grid, tiles)``
    # per launch -- one whole-table entry for a persistent row, one per destination run for a ``per_dest`` row.
    verdict: plans.ShapeGateVerdict
    parts: tuple[tuple[int, int, int, torch.Tensor], ...]
    # SMs every producer launch of this row leaves free (0 when the large-row gate fired, else plans.RESERVED_SMS).
    reserved_sms: int

    @property
    def launches_per_call(self) -> int:
        """Producer launches of one call of this row (the wait kernel and, on the barrier path, the barrier add one each)."""

        return len(self.parts)


def reachable_pairs(
    arch: str,
    bn: int,
    pair_available: bool,
    shape_gates: Mapping[str, int],
    max_tokens: int,
) -> tuple[bool, ...]:
    """The pair flags a session of tile width ``bn`` on ``arch`` can launch for row counts up to ``max_tokens``: the
    policy's pair tile where the pair producer is delivered (``pair_available``) and the geometry pairs, the
    single-CTA producer otherwise -- and BOTH on a 256-column session whose ``pair_off_large_m`` gate can fire below
    ``max_tokens`` (the gate selects the single-CTA producer for its rows).  Off the pair architectures only the
    single-CTA producer exists."""

    if arch not in plans.PAIR_ARCHES:
        return (False,)
    if bn == plans.PAIR_BN:
        if not pair_available:
            return (False,)
        pair_off = int(shape_gates["pair_off_large_m"])
        return (False, True) if pair_off and max_tokens >= pair_off else (True,)
    pairs = {False} if max_tokens < plans.PAIR_MIN_M else set()
    if max_tokens >= plans.PAIR_MIN_M:
        pairs.add(bool(pair_available))
    return tuple(sorted(pairs))


def undelivered_producer_error(
    arch: str,
    bn: int,
    pair: bool,
    variant: str,
    variant_source: str,
    shape_gates: Mapping[str, int],
) -> str:
    """The construction refusal for a reachable producer the route table does not deliver.  When the single-CTA
    256-column producer is reachable only because the pair-tile gate is on, the message names the gate and its
    opt-out (no delivered Blackwell route ships that producer today); otherwise it names the producer build and
    :data:`PRODUCER_VARIANT_ENV`."""

    delivered = loader.delivered_producer_variants(arch, bn, pair)
    pair_off = int(shape_gates["pair_off_large_m"])
    if not pair and bn == plans.PAIR_BN and arch in plans.PAIR_ARCHES and pair_off:
        return (
            f"{SHAPE_GATES_ENV}: pair_off_large_m={pair_off} requires the single-CTA block_n={bn} producer, which "
            f"this delivery does not ship for {arch} (delivered builds for block_n={bn}, pair=False: {delivered}; "
            "the delivered routes carry the 2-CTA pair producer for this tile width) -- drop the gate (0 or auto) on "
            "every rank, or deliver that producer"
        )
    return (
        f"producer build {variant or PRODUCER_VARIANT_SHIPPED!r} ({variant_source}) is not delivered for {arch} at "
        f"block_n={bn}, pair={pair} (delivered: {delivered}; '' is the baseline build) -- set "
        f"{PRODUCER_VARIANT_ENV}={PRODUCER_VARIANT_SHIPPED} or one of the delivered names on every rank"
    )


class CakeQkvExchangeSession:
    """One prepared session on one NCCL process group (collective constructor; see the module docstring)."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        device: torch.device,
        hidden: int,
        num_heads: int,
        head_dim: int,
        layout: str,
        max_tokens: int,
        num_slots: int,
        sample_batch: Optional[int] = None,
        sample_s_local: Optional[int] = None,
        fold_barrier: Optional[bool] = None,
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the fused QKV exchange session is collective and cannot be created inside a capture"
            )
        if str(dist.get_backend(group)).lower() != "nccl":
            raise ValueError("group must use the NCCL backend")
        if device.type != "cuda":
            raise ValueError(f"the Cake backend needs a CUDA device, got {device}")
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.device_index = int(self.device.index)
        arch = device_arch(self.device)
        if arch is None:
            raise ValueError(
                f"device {self.device} (capability {loader.device_facts(self.device_index).capability}) is not an "
                f"architecture the Cake programs target ({sorted(loader.CAPABILITY_ARCH.values())})"
            )
        if not supports_symmetric_memory_backend(self.device):
            raise RuntimeError(
                f"torch symmetric-memory backend {symmetric_memory_backend(self.device)!r} is not supported; the Cake "
                f"backend needs one of {SUPPORTED_SYMMETRIC_MEMORY_BACKENDS}"
            )
        self.arch = arch
        # The group's rank and size first: the producer build and the group scaling are keyed by world size.
        self.group = group
        self.group_name = _group_name(group)
        self.rank = int(dist.get_rank(group=group))
        self.world_size = int(dist.get_world_size(group=group))
        if not 0 <= self.rank < self.world_size:
            raise RuntimeError(
                f"process group reports rank {self.rank} outside [0, {self.world_size})"
            )
        supported = loader.supported_world_sizes(arch)
        if self.world_size not in supported:
            raise ValueError(
                f"world size must be one of {supported} on {arch}, got {self.world_size}"
            )
        self.fold_barrier, self.fold_barrier_source = resolve_fold_barrier(
            arch, fold_barrier
        )
        self.fold_barrier_requested = fold_barrier
        self.shape_gates, self.shape_gates_source = resolve_shape_gates(
            arch, self.world_size
        )
        if layout not in plans.LAYOUTS:
            raise ValueError(f"layout must be one of {plans.LAYOUTS}, got {layout!r}")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if (
            isinstance(num_slots, bool)
            or not isinstance(num_slots, int)
            or num_slots < 2
        ):
            raise ValueError(
                f"num_slots must be an int of at least 2, got {num_slots!r}"
            )
        if int(num_heads) * int(head_dim) != int(hidden):
            raise ValueError(
                f"hidden={hidden} must equal num_heads * head_dim = {num_heads} * {head_dim}"
            )
        # Any row count up to max_tokens shares the band width; the geometry of max_tokens rows sizes the slots.
        capacity = plans.Geometry(
            world_size=self.world_size,
            batch=1,
            s_local=int(max_tokens),
            hidden=int(hidden),
            head_dim=int(head_dim),
        )
        self.hidden, self.num_heads, self.head_dim = (
            int(hidden),
            int(num_heads),
            int(head_dim),
        )
        self.layout = layout
        self.max_tokens = int(max_tokens)
        self.num_slots = int(num_slots)
        self.bn = plans.block_n_for(capacity)
        self.slot_stride = capacity.slot_elems
        self.words_cap = plans.words_capacity_for_rows(self.max_tokens)
        self.sm_count = loader.device_facts(self.device_index).sm_count
        # The producer build and the group scaling of this session (the (architecture, world size) defaults
        # unless their environment variables override them; both rank-uniform, checked collectively below).
        self.producer_variant, self.producer_variant_source = resolve_producer_variant(
            self.arch, self.world_size, self.bn
        )
        self.group_m_scale, self.group_m_scale_source = resolve_group_m_scale(
            self.arch, self.world_size
        )
        if symmetric_memory_backend(self.device) == "":
            symm_mem.set_backend("CUDA")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.group_name)

        # Programs (JIT build is collective-safe only outside the barrier: every rank builds before any launch).
        # The barrier programs are loaded on both paths; of the producers only the launch path's builds are:
        # the plain producer for the barrier path, both rendezvous phases of the folded producer otherwise.
        self.pair_available = {
            bn: loader.has_producer(self.arch, bn, True, self.producer_variant)
            for bn in plans.BLOCK_N_CHOICES
        }
        for pair in self._reachable_pairs():
            if not loader.has_producer(self.arch, self.bn, pair, self.producer_variant):
                raise ValueError(
                    undelivered_producer_error(
                        self.arch,
                        self.bn,
                        pair,
                        self.producer_variant,
                        self.producer_variant_source,
                        self.shape_gates,
                    )
                )
        self._barrier_names = tuple(
            loader.barrier_program(self.arch, phase) for phase in (0, 1)
        )
        self._barrier = tuple(
            loader.load(name, self.arch) for name in self._barrier_names
        )
        self._wait_name = loader.wait_program(self.arch)
        self._wait = loader.load(self._wait_name, self.arch)
        self._producers: dict[str, Any] = {}
        per_dest_reachable = (
            plans.gate_rows(
                self.shape_gates, self.max_tokens, bn=self.bn
            ).producer_launches
            == "per_dest"
        )
        for pair in self._reachable_pairs():
            for name in self._launched_producers(pair, per_dest=per_dest_reachable):
                self._producers[name] = loader.load(name, self.arch)

        # Symmetric memory: receive slots, readiness pad, barrier flags (rendezvous = collective).  The
        # readiness pad holds ``words`` symmetric words the peers publish into, then one rank-local word: the
        # arrive flag of the folded rendezvous (CTA 0's rendezvous warp publishes the call's epoch into it after
        # the cross-rank barrier, every CTA acquires it once before its first store; peers never touch it).
        words = self.num_slots * self.world_size * self.words_cap
        self.arrive_flag_offset = words
        flag_words = loader.barrier_flag_words(self.world_size)
        self._recv = symm_mem.empty(
            self.num_slots * self.slot_stride, dtype=torch.bfloat16, device=self.device
        )
        self._ready = symm_mem.empty(words + 1, dtype=torch.uint32, device=self.device)
        self._flags = symm_mem.empty(flag_words, dtype=torch.uint32, device=self.device)
        self._recv.zero_()
        self._ready.zero_()
        self._flags.zero_()

        def rendezvous(tensor: torch.Tensor, what: str):
            return _rendezvous(
                tensor,
                group_name=self.group_name,
                rank=self.rank,
                world_size=self.world_size,
                what=what,
            )

        self._recv_handle = rendezvous(self._recv, "receive slots")
        self._ready_handle = rendezvous(self._ready, "readiness pad")
        self._flags_handle = rendezvous(self._flags, "barrier flags")
        self.dst_ptrs = torch.tensor(
            _peer_pointers(
                self._recv_handle, self.world_size, self._recv.numel(), torch.bfloat16
            ),
            dtype=torch.int64,
            device=self.device,
        )
        ready_peers = _peer_pointers(
            self._ready_handle, self.world_size, words, torch.uint32
        )
        self.ready_ptrs = torch.tensor(
            ready_peers, dtype=torch.int64, device=self.device
        )
        # The folded producer reads ``world_size + 1`` pointers: the peers' pads, then this rank's arrive flag.
        self.ready_ptrs_fold = torch.tensor(
            [
                *ready_peers,
                int(self._ready.data_ptr()) + 4 * self.arrive_flag_offset,
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.flag_peers = tvm_ffi.Shape(
            _peer_pointers(
                self._flags_handle, self.world_size, flag_words, torch.uint32
            )
        )
        self.counters = torch.zeros(
            self.world_size * self.words_cap, dtype=torch.uint32, device=self.device
        )
        # The captured-call clear needs ``memset32_`` on the dedicated pad; probe it once with a rank-uniform
        # verdict, and check that every rank resolved the same launch path.
        self.memset32_probe = self._rank_uniform_probe(words)
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=group)

        self._consumer = torch.cuda.Stream(device=self.device)
        self._fork = torch.cuda.Event()
        self._exchange_done = [torch.cuda.Event() for _ in range(self.num_slots)]
        self._capture_done: dict[int, torch.cuda.Event] = {}
        self._lock = threading.Lock()
        self._phase = 0
        self._epoch = 0
        self._next_slot = 0
        self._calls = 0
        self._captured_calls = 0
        self._poisoned = False
        self._closed = False
        self._plans: dict[tuple[int, int], _RowPlan] = {}
        self._last: Optional[tuple[int, _RowPlan]] = None
        if sample_batch is not None and sample_s_local is not None:
            self.prepare_rows(int(sample_batch), int(sample_s_local))

    # ------------------------------------------------------------------ preparation

    def _reachable_pairs(self) -> tuple[bool, ...]:
        """Pair flags the policy and the pair-tile gate can select for row counts up to ``max_tokens`` (programs to
        load eagerly): :func:`reachable_pairs` on this session's tile width, pair delivery and gates."""

        return reachable_pairs(
            self.arch,
            self.bn,
            self.pair_available[self.bn],
            self.shape_gates,
            self.max_tokens,
        )

    def _fold_producer_names(
        self, pair: bool, variant: Optional[str] = None
    ) -> tuple[str, ...]:
        """Both rendezvous phases of the folded producer of this tile width and build (the session's build unless
        ``variant`` names another delivered one; empty where it is not delivered)."""

        variant = self.producer_variant if variant is None else variant
        if not loader.has_fold_producer(self.arch, self.bn, pair, variant):
            return ()
        return tuple(
            loader.fold_producer_program(self.arch, self.bn, pair, phase, variant)
            for phase in loader.FOLD_PHASES
        )

    def _launched_producers(
        self, pair: bool, *, per_dest: bool = False
    ) -> tuple[str, ...]:
        """Producer programs the resolved launch path launches for this tile width (the session's build); a
        ``per_dest`` row on the folded path adds the plain producer (its second and later destination parts)."""

        plain = loader.producer_program(self.arch, self.bn, pair, self.producer_variant)
        if not self.fold_barrier:
            return (plain,)
        names = self._fold_producer_names(pair)
        if not names:
            raise RuntimeError(
                f"fold_barrier=True, but {self.arch} delivers no folded producer for block_n={self.bn}, "
                f"pair={pair}, build {self.producer_variant or PRODUCER_VARIANT_SHIPPED!r}; pass fold_barrier=False"
            )
        return (*names, plain) if per_dest else names

    def _rank_uniform_probe(self, words: int) -> dict[str, Any]:
        """Collective self-check before the first launch: ``memset32_`` accepts the dedicated pad on every
        rank (the captured-call clear needs it) and every rank resolved the same launch path, row-count gates,
        producer build and group scaling."""

        local_error: Optional[str] = None
        try:
            torch.ops.symm_mem.memset32_(self._ready, 0, 0, int(words))
        except Exception as error:  # noqa: BLE001 - the verdict is published to every rank below
            local_error = f"{type(error).__name__}: {error}"[:500]
        verdicts: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            verdicts,
            {
                "rank": self.rank,
                "error": local_error,
                "fold_barrier": self.fold_barrier,
                "shape_gates": dict(self.shape_gates),
                "producer_variant": self.producer_variant,
                "group_m_scale": self.group_m_scale,
            },
            group=self.group,
        )
        rejecting = {
            int(v["rank"]): v["error"] for v in verdicts if v.get("error") is not None
        }
        if rejecting:
            raise RuntimeError(
                "torch.ops.symm_mem.memset32_ rejected the dedicated readiness pad on ranks "
                f"{sorted(rejecting)}: {rejecting[sorted(rejecting)[0]]}; the captured-call clear needs it"
            )
        paths = {int(v["rank"]): bool(v["fold_barrier"]) for v in verdicts}
        if len(set(paths.values())) != 1:
            raise RuntimeError(
                f"the launch path differs across ranks (fold_barrier per rank: {paths}); pass the same "
                f"fold_barrier keyword or the same {FOLD_BARRIER_ENV} value on every rank"
            )
        # The gates decide every rank's readiness words and launch count per row; they must agree before any launch.
        gates = {
            int(v["rank"]): tuple(sorted((v.get("shape_gates") or {}).items()))
            for v in verdicts
        }
        if len(set(gates.values())) != 1:
            raise RuntimeError(
                f"the row-count gates differ across ranks (per rank: {gates}); set the same {SHAPE_GATES_ENV} value "
                "on every rank (or none)"
            )
        # The producer build decides the programs every rank launches and the group scaling its tile order; a
        # rank-dependent choice would surface as a hang, so both must agree before any launch.
        variants = {int(v["rank"]): str(v.get("producer_variant")) for v in verdicts}
        if len(set(variants.values())) != 1:
            raise RuntimeError(
                f"the producer build differs across ranks (per rank: {variants}); set the same "
                f"{PRODUCER_VARIANT_ENV} value on every rank (or none)"
            )
        scales = {int(v["rank"]): v.get("group_m_scale") for v in verdicts}
        if len(set(scales.values())) != 1:
            raise RuntimeError(
                f"the group scaling differs across ranks (per rank: {scales}); set the same {GROUP_M_SCALE_ENV} "
                "value on every rank (or none)"
            )
        return {"accepted": True, "words": int(words)}

    def geometry(self, batch: int, s_local: int) -> plans.Geometry:
        return plans.Geometry(
            world_size=self.world_size,
            batch=int(batch),
            s_local=int(s_local),
            hidden=self.hidden,
            head_dim=self.head_dim,
        )

    def prepare_rows(self, batch: int, s_local: int) -> _RowPlan:
        """Build and upload the tile plan of ``batch * s_local`` rows (eager; cached per ``(batch, s_local)``)."""

        key = (int(batch), int(s_local))
        row = self._plans.get(key)
        if row is not None:
            return row
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"the fused QKV exchange session has no tile plan for batch={batch}, s_local={s_local}; a row count "
                "first seen inside a CUDA-graph capture would allocate -- call the session eagerly once with that shape"
            )
        geometry = self.geometry(batch, s_local)
        if geometry.rows > self.max_tokens:
            raise ValueError(
                f"batch * s_local = {geometry.rows} exceeds the session's max_tokens={self.max_tokens}"
            )
        if geometry.slot_elems > self.slot_stride:
            raise ValueError(
                "row geometry exceeds the receive slot capacity of the session"
            )
        # Row-count gates of this call (the session's gates on ``rows = batch * s_local`` and the session's tile
        # width): one readiness word per destination for a small row, one producer launch per destination run for a
        # large row, the single-CTA producer for a large 256-column row under the pair-tile gate.
        verdict = plans.gate_rows(self.shape_gates, int(geometry.rows), bn=self.bn)
        words_capacity = (
            verdict.words_capacity_per_dest
            if verdict.words_capacity_per_dest is not None
            else self.words_cap
        )
        policy, plan = plans.plan_for(
            geometry,
            self.layout,
            self.rank,
            arch=self.arch,
            pair_available=self.pair_available[self.bn],
            words_capacity_per_dest=words_capacity,
            group_m_scale=self.group_m_scale,
            pair=verdict.pair,
        )
        if plan.bn != self.bn or plan.num_words > self.words_cap:
            raise RuntimeError("tile plan disagrees with the session capacity")
        program = loader.producer_program(
            self.arch, plan.bn, plan.pair, self.producer_variant
        )
        per_dest = verdict.producer_launches == "per_dest"
        for name in self._launched_producers(plan.pair, per_dest=per_dest):
            if name not in self._producers:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        f"program {name!r} is not loaded; a JIT build cannot run inside a capture"
                    )
                self._producers[name] = loader.load(name, self.arch)
        # SMs every producer launch of this row leaves free: none when the large-row gate fired, else the
        # session constant (one for the wait kernel, one slack).
        reserved_sms = (
            verdict.reserved_sms
            if verdict.reserved_sms is not None
            else plans.RESERVED_SMS
        )
        tiles = plan.tiles.to(self.device)
        ranges = plans.dest_parts(plan) if per_dest else ((0, plan.num_tiles),)
        parts = tuple(
            (
                start,
                stop,
                plans.producer_grid(
                    stop - start,
                    sm_count=self.sm_count,
                    reserved_sms=reserved_sms,
                    lockstep=plan.lockstep,
                ),
                tiles if (start, stop) == (0, plan.num_tiles) else tiles[start:stop],
            )
            for start, stop in ranges
        )
        row = _RowPlan(
            policy=policy,
            plan=plan,
            tiles=tiles,
            grid=parts[0][2]
            if len(parts) == 1
            else plans.producer_grid(
                plan.num_tiles,
                sm_count=self.sm_count,
                reserved_sms=reserved_sms,
                lockstep=plan.lockstep,
            ),
            program=program,
            fold_programs=self._fold_producer_names(plan.pair),
            verdict=verdict,
            parts=parts,
            reserved_sms=int(reserved_sms),
        )
        self._plans[key] = row
        return row

    def bound_programs(self, batch: int, s_local: int) -> dict[str, str]:
        """Program names of every delivered stage of a ``(batch, s_local)`` call (prepares the row eagerly):
        the barrier programs, the plain producer of the session's build, its folded producers where delivered
        (``producer_fold0`` / ``producer_fold1``), the baseline build's three programs where the session's build
        is a variant (``producer_shipped``, ``producer_shipped_fold0`` / ``producer_shipped_fold1``: the
        ``shipped`` opt-out path of this row) and the wait kernel -- six or nine names.  :meth:`describe` says
        which of them a call launches (``launch_sequence``)."""

        row = self.prepare_rows(batch, s_local)
        bound = {
            "barrier_p0": self._barrier_names[0],
            "barrier_p1": self._barrier_names[1],
            "producer": row.program,
        }
        if row.fold_programs:
            for phase, name in zip(loader.FOLD_PHASES, row.fold_programs, strict=True):
                bound[f"producer_fold{phase}"] = name
        if self.producer_variant != loader.SHIPPED_VARIANT:
            bound["producer_shipped"] = loader.producer_program(
                self.arch, row.plan.bn, row.plan.pair, loader.SHIPPED_VARIANT
            )
            shipped_folds = self._fold_producer_names(
                row.plan.pair, loader.SHIPPED_VARIANT
            )
            if not shipped_folds:
                raise RuntimeError(
                    f"{self.arch} delivers the {self.producer_variant!r} build for block_n={row.plan.bn}, "
                    f"pair={row.plan.pair} without the baseline build's folded producers"
                )
            for phase, name in zip(loader.FOLD_PHASES, shipped_folds, strict=True):
                bound[f"producer_shipped_fold{phase}"] = name
        bound["wait"] = self._wait_name
        return bound

    # ------------------------------------------------------------------ calls

    def _validate(self, x, w_qkv, q_scale, k_scale, cos_sin) -> tuple[int, int]:
        for name, tensor in (
            ("x", x),
            ("w_qkv", w_qkv),
            ("q_norm_weight", q_scale),
            ("k_norm_weight", k_scale),
            ("cos_sin", cos_sin),
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
                raise ValueError(f"{name} must be a tensor on {self.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if x.dtype != torch.bfloat16 or x.dim() != 3 or int(x.shape[2]) != self.hidden:
            raise ValueError(
                f"x must be BF16 [B, S_local, {self.hidden}], got {tuple(x.shape)} {x.dtype}"
            )
        batch, s_local = int(x.shape[0]), int(x.shape[1])
        if batch <= 0 or s_local <= 0:
            raise ValueError("x must have positive batch and sequence dimensions")
        if w_qkv.dtype != torch.bfloat16 or tuple(w_qkv.shape) != (
            3 * self.hidden,
            self.hidden,
        ):
            raise ValueError(
                f"w_qkv must be BF16 [{3 * self.hidden}, {self.hidden}], got {tuple(w_qkv.shape)} {w_qkv.dtype}"
            )
        for name, scale in (("q_norm_weight", q_scale), ("k_norm_weight", k_scale)):
            if scale.dtype != torch.bfloat16 or tuple(scale.shape) != (self.head_dim,):
                raise ValueError(
                    f"{name} must be BF16 [{self.head_dim}], got {tuple(scale.shape)} {scale.dtype}"
                )
        rows = batch * s_local
        if cos_sin.dtype != torch.float32 or tuple(cos_sin.shape) != (
            rows,
            self.head_dim,
        ):
            raise ValueError(
                f"cos_sin must be fp32 [{rows}, {self.head_dim}], got {tuple(cos_sin.shape)} {cos_sin.dtype}"
            )
        if rows > self.max_tokens:
            raise ValueError(
                f"batch * s_local = {rows} exceeds the session's max_tokens={self.max_tokens}"
            )
        # The wide-operand-load builds read ``cos_sin`` and the norm weights through 32-byte-aligned bases (the
        # rows are 512 B by the shape check above); a misaligned view of a larger tensor must be refused here, the
        # kernel does not check it.
        if loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH[self.producer_variant] == 256:
            for name, tensor in (
                ("cos_sin", cos_sin),
                ("q_norm_weight", q_scale),
                ("k_norm_weight", k_scale),
            ):
                if tensor.data_ptr() % 32:
                    raise ValueError(
                        f"{name} must start at a 32-byte-aligned address for the {self.producer_variant!r} producer "
                        f"build (data_ptr % 32 = {tensor.data_ptr() % 32}); pass a contiguous tensor that starts "
                        f"at its allocation, or set {PRODUCER_VARIANT_ENV}={PRODUCER_VARIANT_SHIPPED} on every rank"
                    )
        return batch, s_local

    def __call__(self, x, w_qkv, q_scale, k_scale, cos_sin) -> int:
        """Run one exchange; returns the slot the outputs live in (see :meth:`outputs`)."""

        if self._closed:
            raise RuntimeError("the fused QKV exchange session is closed")
        batch, s_local = self._validate(x, w_qkv, q_scale, k_scale, cos_sin)
        capturing = bool(torch.cuda.is_current_stream_capturing())
        row = self._plans.get((batch, s_local))
        if row is None:
            row = self.prepare_rows(batch, s_local)  # raises inside a capture
        plan = row.plan
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "the fused QKV exchange session is poisoned by a prior failed collective"
                )
            slot = self._next_slot
            if capturing:
                if slot in self._capture_done:
                    raise RuntimeError(
                        f"at most num_slots={self.num_slots} calls of this session can be captured into one graph "
                        "without an eager call in between (the slot's capture-local event would be reused)"
                    )
                ready_target = CAPTURE_READY_TARGET
            else:
                if self._epoch >= EAGER_READY_TARGET_MAX:
                    raise RuntimeError(
                        "the eager readiness epoch exhausted the uint32 range"
                    )
                ready_target = self._epoch + 1
            phase = self._phase
            stream = torch.cuda.current_stream(self.device)
            consumer = self._consumer
            # Readiness words of one call: ``world_size`` contiguous pads of ``plan.num_words`` words
            # (the wait kernel scans ``words_total`` words from ``slot_base``).  Slots are spaced by the
            # session capacity, not by this call's ``num_words``: consecutive calls with different row
            # shapes may be in flight on different slots, and a shape-dependent stride would let a
            # smaller call's slot overlap a larger call's words.
            words_total = self.world_size * plan.num_words
            slot_base = slot * self.world_size * self.words_cap
            pad_base = slot_base + self.rank * plan.num_words
            try:
                if capturing:
                    fork = torch.cuda.Event()
                    done = torch.cuda.Event()
                else:
                    stream.wait_event(self._exchange_done[slot])
                    fork = self._fork
                    done = self._exchange_done[slot]
                a_view = x.view(plan.M, plan.K)

                def producer_args(tiles, num_tiles, ready_ptrs):
                    # One launch over ``tiles`` (the whole table, or one destination run of a ``per_dest`` row);
                    # every part shares the call's slot, pad and epoch.
                    return (
                        a_view,
                        w_qkv,
                        cos_sin,
                        q_scale,
                        k_scale,
                        tiles,
                        self.dst_ptrs,
                        ready_ptrs,
                        self.counters,
                        num_tiles,
                        plan.M,
                        plan.K,
                        s_local,
                        plan.head_stride,
                        plan.batch_stride,
                        slot * self.slot_stride,
                        pad_base,
                        plan.num_words,
                        self.rank,
                        self.world_size,
                        ready_target,
                    )

                with tvm_ffi.use_torch_stream():
                    if self.fold_barrier:
                        # Folded rendezvous: the fork is recorded before the producer so the wait kernel runs
                        # beside it; the FIRST launch's rendezvous warp runs the cross-rank barrier of this call's
                        # phase and releases every CTA's stores through the arrive flag; the later destination
                        # parts of a ``per_dest`` row are the plain producer, stream-ordered behind it.
                        fork.record(stream)
                        start, stop, grid, tiles = row.parts[0]
                        self._producers[row.fold_programs[phase]].run(
                            *producer_args(tiles, stop - start, self.ready_ptrs_fold),
                            self.world_size,
                            self.rank,
                            self.flag_peers,
                            grid,
                            1,
                            1,
                        )
                        if capturing:
                            # The arrive flag holds the sentinel: clear it once every CTA acquired it (stream
                            # order behind the producer) so the next replay waits for a fresh rendezvous.
                            torch.ops.symm_mem.memset32_(
                                self._ready, self.arrive_flag_offset, 0, 1
                            )
                        for start, stop, grid, tiles in row.parts[1:]:
                            self._producers[row.program].run(
                                *producer_args(tiles, stop - start, self.ready_ptrs),
                                grid,
                                1,
                                1,
                            )
                    else:
                        self._barrier[phase].run(
                            self.device_index,
                            self.world_size,
                            self.rank,
                            self.flag_peers,
                            1,
                            1,
                            1,
                        )
                        fork.record(stream)
                        for start, stop, grid, tiles in row.parts:
                            self._producers[row.program].run(
                                *producer_args(tiles, stop - start, self.ready_ptrs),
                                grid,
                                1,
                                1,
                            )
                    consumer.wait_event(fork)
                    with torch.cuda.stream(consumer):
                        self._wait.run(
                            self._ready, words_total, slot_base, ready_target, 1, 1, 1
                        )
                        if capturing:
                            # A replay re-issues the sentinel: clear this call's words once the wait kernel consumed them.
                            torch.ops.symm_mem.memset32_(
                                self._ready, slot_base, 0, words_total
                            )
                        done.record(consumer)
                stream.wait_event(done)
            except Exception:
                # A partially submitted collective cannot be retried with either barrier phase or the slot.
                self._poisoned = True
                raise
            self._phase = 1 - phase
            self._next_slot = (slot + 1) % self.num_slots
            self._calls += 1
            if capturing:
                self._capture_done[slot] = done
                self._captured_calls += 1
            else:
                self._epoch = ready_target
                self._capture_done.clear()
            self._last = (slot, row)
            return slot

    # ------------------------------------------------------------------ outputs and state

    @property
    def recv_local(self) -> torch.Tensor:
        return self._recv

    def outputs(self, slot: int, row: _RowPlan):
        """Views of ``slot`` for the row geometry: ``(q, k, v)`` for ``"bshd"``, one payload for ``"fused"``."""

        plan = row.plan
        base = int(slot) * self.slot_stride
        region = self._recv[base : base + plan.slot_elems]
        if self.layout == "bshd":
            shape = (plan.batch, plan.s_global, plan.h_local, plan.head_dim)
            return tuple(
                region[index * plan.plane : (index + 1) * plan.plane].view(shape)
                for index in range(3)
            )
        return region.view(plan.batch, plan.s_global, plan.h_local, 3 * plan.head_dim)

    def last_outputs(self):
        if self._last is None:
            raise RuntimeError("no call has run on this session yet")
        slot, row = self._last
        return self.outputs(slot, row)

    def poison_receive_slots(self) -> None:
        """Fill this rank's whole receive region with ``0xFF`` bytes (eager only; proves the next call rewrites it)."""

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("poison_receive_slots must not be captured")
        self._recv.view(torch.int16).fill_(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "nvlink",
            "arch": self.arch,
            "world_size": self.world_size,
            "rank": self.rank,
            "layout": self.layout,
            "hidden": self.hidden,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "block_n": self.bn,
            "pair_available": dict(self.pair_available),
            "max_tokens": self.max_tokens,
            "num_slots": self.num_slots,
            "slot_stride_elems": self.slot_stride,
            "words_capacity_per_dest": self.words_cap,
            "reserved_sms": plans.RESERVED_SMS,
            "sm_count": self.sm_count,
            "ready_mode": "dedicated_symmetric_u32",
            "memset32_probe": dict(self.memset32_probe),
            "fold_barrier": self.fold_barrier,
            "fold_barrier_requested": (
                "auto"
                if self.fold_barrier_requested is None
                else self.fold_barrier_requested
            ),
            "fold_barrier_source": self.fold_barrier_source,
            # The launch structure of a row below the per_dest gate; a row at or above it launches one producer
            # part per destination run (``prepared_row_plans[row]["launches_per_call"]``).
            "launches_per_call": 2 if self.fold_barrier else 3,
            "launch_sequence": (
                ["producer_fold<phase>", "wait"]
                if self.fold_barrier
                else ["barrier_p<phase>", "producer", "wait"]
            ),
            "launch_sequence_per_dest": (
                ["producer_fold<phase>", "producer (x parts - 1)", "wait"]
                if self.fold_barrier
                else ["barrier_p<phase>", "producer (x parts)", "wait"]
            ),
            "shape_gates": {
                "values": plans.spell_shape_gates(self.shape_gates),
                "source": self.shape_gates_source,
                "env": SHAPE_GATES_ENV,
                "default": plans.spell_shape_gates(
                    plans.default_shape_gates(self.arch, world_size=self.world_size)
                ),
            },
            "producer_variant": {
                "value": self.producer_variant,
                "source": self.producer_variant_source,
                "env": PRODUCER_VARIANT_ENV,
                "default": loader.default_producer_variant(self.arch, self.world_size),
            },
            "group_m_scale": {
                "value": self.group_m_scale,
                "source": self.group_m_scale_source,
                "env": GROUP_M_SCALE_ENV,
                "default": plans.default_group_m_scale(self.arch, self.world_size),
            },
            "prepared_row_plans": {
                f"{batch}x{s_local}": {
                    **row.verdict.summary(),
                    "num_words": row.plan.num_words,
                    "producer_parts": row.launches_per_call,
                    "launches_per_call": row.launches_per_call
                    + (1 if self.fold_barrier else 2),
                    "part_grids": [grid for _, _, grid, _ in row.parts],
                }
                for (batch, s_local), row in sorted(self._plans.items())
            },
            "programs": {
                "barrier_p0": self._barrier_names[0],
                "barrier_p1": self._barrier_names[1],
                "wait": self._wait_name,
                "producers": sorted(self._producers),
            },
            "prepared_rows": sorted(
                f"{batch}x{s_local}" for batch, s_local in self._plans
            ),
            "calls": self._calls,
            "captured_calls": self._captured_calls,
        }

    def close(self) -> None:
        """Release the session (collective: every rank calls it at the same point, never inside a capture)."""

        if self._closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("close() must not be captured")
        self._closed = True
        torch.cuda.current_stream(self.device).synchronize()
        self._consumer.synchronize()
        dist.barrier(group=self.group)
        self._plans.clear()
        self._last = None
        self._producers.clear()
        self._capture_done.clear()
        self._recv = self._ready = self._flags = None  # type: ignore[assignment]
        self._recv_handle = self._ready_handle = self._flags_handle = None
        self.dst_ptrs = self.ready_ptrs = self.counters = None  # type: ignore[assignment]
        self.ready_ptrs_fold = None  # type: ignore[assignment]


__all__ = [
    "CAPTURE_READY_TARGET",
    "EAGER_READY_TARGET_MAX",
    "FOLD_BARRIER_ENV",
    "GROUP_M_SCALE_ENV",
    "PRODUCER_VARIANT_ENV",
    "PRODUCER_VARIANT_SHIPPED",
    "SHAPE_GATES_ENV",
    "SUPPORTED_SYMMETRIC_MEMORY_BACKENDS",
    "CakeQkvExchangeSession",
    "device_arch",
    "fold_barrier_from_env",
    "group_m_scale_from_env",
    "producer_variant_from_env",
    "resolve_fold_barrier",
    "resolve_group_m_scale",
    "resolve_producer_variant",
    "resolve_shape_gates",
    "shape_gates_from_env",
    "supports_symmetric_memory_backend",
    "symmetric_memory_backend",
]
