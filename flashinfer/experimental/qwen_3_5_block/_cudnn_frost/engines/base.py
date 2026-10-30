# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend (engine) contract for the Python graph: claim -> compile -> execute.

A backend is one of the interchangeable implementations the ranked plan list
dispatches to (Python DSLs, a naive reference, the cuDNN Graph backend, ...). The
lifecycle mirrors a real JIT/DSL engine:

  1. ``check_support(graph)`` -> accept, or decline the whole graph by raising
     ``NotImplementedError`` / ``cudnn.cudnnGraphNotSupportedError``. An engine
     does NOT propose its own plans: which configs to try, in what order, and
     where the backend's own entries belong is one comparison across every
     candidate, which no engine can make from the inside. That lives in
     ``engines/heuristics.py`` and the family hook it dispatches to.
  2. ``build_plan(graph, plan)`` -> a ``CompiledPlan`` — the expensive JIT step,
     run ONCE per (graph, selected plan) at ``graph.build_plans()`` time; the
     compiled artifact lives on the graph, so one engine instance is safely
     reusable across graphs.
  3. ``CompiledPlan.execute(graph, uid_to_data, ctx)`` — hot path.
     ``uid_to_data`` is the caller's variant pack (tensor uid -> device
     buffer, exactly as the classic backend receives it); the
     ``ExecutionContext`` carries the caller's handle / stream / workspace
     explicitly; engines must not hard-code a stream or silently allocate
     hidden workspace. Engines that address buffers by
     port name call ``resolve_node_buffers(graph, uid_to_data)`` (see below).

Simple eager engines only implement ``execute()`` — the default ``build_plan``
wraps it in a trivial ``CompiledPlan``.

Example:
    class MyEngine(BaseEngine):
        name = "my_engine"
        engine_id = PYTHON_ENGINE_ID_BASE + 7  # stable id it owns

        def check_support(self, graph):
            for node in graph.nodes:
                if node.node_type != NodeType.MATMUL:
                    raise NotImplementedError(...)

        def execute(self, graph, uid_to_data, ctx):
            node_buffers = resolve_node_buffers(graph, uid_to_data)
            ...  # write results into caller-provided output buffers
"""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple

from .engine_ids import PYTHON_ENGINE_ID_BASE  # noqa: F401 — re-exported for engine authors

if TYPE_CHECKING:
    from ..pygraph import pygraph


def decline_types():
    """The exception types that mean "this engine does not serve this graph".

    ImportError counts: an engine whose optional dependency is absent cannot
    serve the graph, and since lowering imports are deferred past check_support
    that only becomes visible at build time.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    return (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)


@dataclass(frozen=True)
class PlanConfig:
    """One candidate execution plan: an engine id + its knobs.

    ``engine_id`` lives in the shared flat id space (``engine_ids``); knobs are
    engine-specific tuning (cuDNN knob dict, or a python engine's config). The
    plan's source is derived from the id via ``is_python_engine`` — no separate
    field, so cuDNN and python plans are interchangeable in the ranked list.
    One engine may propose several plans differing only in knobs.

    ``knobs`` here is the engine's NATIVE form (a python engine may keep a typed
    config object). At the public surface — ``get_engine_and_knobs_at_index``,
    ``create_execution_plan``, ``get_plan_name_at_index`` — every plan speaks
    the one shared vocabulary instead: ``{cudnn.knob_type: int}``, backend and
    python alike, converted through ``BaseEngine.knobs_to_public`` /
    ``knobs_from_public``. That pair is what autotuners persist and replay.

    ``cpp_index`` is set only on backend entries: the position this plan holds
    in the lowered graph's own plan list, so building it is one
    ``build_plan_at_index`` instead of a rebuild from (engine_id, knobs).

    ``mode`` is the heuristic mode that produced the entry. Ranking needs it:
    "the backend's mode-A entries ahead of ours, its fallbacks behind" cannot
    be said about a list whose entries do not remember where they came from.
    """

    engine_id: int
    knobs: Any = None
    cpp_index: Any = None
    mode: Any = None


class ExecutionContext(NamedTuple):
    """Runtime context passed to a compiled plan at execute time.

    Everything an engine may need is explicit here — no engine should reach
    into private graph state, hard-code a stream, or allocate hidden workspace.
    ``stream`` is resolved from the handle when available (classic
    ``cudnn.set_stream(handle, ...)`` semantics).

    A ``NamedTuple`` (not a frozen dataclass): both immutable, but built on every
    execute, where the NamedTuple is ~220 ns/execute cheaper.
    """

    handle: Any = None
    stream: Any = None
    workspace: Any = None


class VariantPack:
    """The caller's variant pack, normalized ONCE at the top of execute().

    Whatever the caller passed — a torch tensor, a DeviceView, any
    ``__dlpack__`` / ``__cuda_array_interface__`` producer, or a bare device
    address — is converted here and then dropped. Below this point the cuDNN
    backend and every python engine see the same two index-aligned sequences
    and nothing else, so neither can behave differently on account of what the
    caller happened to hold.

    The operands live in ``native``, a C container holding one ``DLTensor``
    each, read through the producer's ``__dlpack_c_exchange_api__`` vtable and
    handed to kernels through the same one.

    ``uids`` is ASCENDING, matching the backend's own operand order
    (``get_variant_pack_uids_sorted()``), so ``address`` goes straight to
    ``_execute_with_raw_ptrs`` with no copy and no per-operand hash lookup.

    A slot describes the buffer as the GRAPH declares it whenever the caller's
    own geometry disagrees but covers the declared bytes (``graph_described``
    names those slots, bare addresses included). Strided views and buffers smaller
    than their declarations keep their own descriptions. Reordered scale blobs
    retain their physical extents for capacity checks. A described slot carries the
    DECLARED dtype, as does one of the declared extents whose slots are as wide
    (FlashInfer binds packed fp4 and e4m3 scale blocks as uint8). Overrides are
    written into the slot too. An engine reads the IR port for the shape the plan was built for and
    this pack for the shape about to run; frost_gemm takes its M/N/K from here,
    the backend takes only the pointer.

    Allocated per call. Two threads may execute one graph concurrently with
    different buffers, and a shared pack would hand each thread the other's
    pointers — silently, because every pointer in it is individually valid.
    """

    __slots__ = ("uids", "native", "_index_of", "workspace", "workspace_bytes", "_device", "graph_described", "overridden")

    def __init__(self, uids, native, workspace_ptr: int = 0, workspace_bytes: int | None = 0, graph_described=(), overridden=()):
        self.uids = uids
        self.native = native
        self.workspace = workspace_ptr
        # Capacity is optional, never encoded in a byte count: None means an
        # unmeasured raw address; every integer (including zero) is a bound.
        if workspace_bytes is not None and workspace_bytes < 0:
            raise ValueError("workspace capacity must be nonnegative or None for an unknown raw-address capacity")
        self.workspace_bytes = workspace_bytes
        # Slots whose dim/stride were lent by the graph because the caller
        # passed a bare address. Usually empty. An engine that reads extents by
        # axis position needs this: the graph and the caller order a matmul's B
        # differently, and the description does not say which one it is.
        self.graph_described = graph_described
        # Explicit execute overrides change the requested geometry, while a
        # producer view can still be raw storage under the graph declaration.
        self.overridden = overridden
        self._index_of = None  # built on first lookup: the backend never does one
        self._device = None

    @property
    def address(self) -> int:
        """The ``void*[]`` in operand order, for ``_execute_with_raw_ptrs``."""
        return self.native.address

    def all_contiguous(self):
        """``(ok, index)`` over every filled operand, decided from the strides
        the native pack already holds."""
        ok, offender = self.native.all_contiguous()
        return ok, (int(offender) if offender else -1)

    def all_dense_layout(self):
        """``(ok, slot)`` over every filled operand: the innermost size>1 dim
        must be stride-1. Padded or permuted outer strides pass."""
        ok, offender = self.native.all_dense_layout()
        return ok, (int(offender) if offender else -1)

    @property
    def index_of_uid(self):
        if self._index_of is None:
            self._index_of = {u: i for i, u in enumerate(self.uids)}
        return self._index_of

    @property
    def device(self) -> int:
        """The GPU this execute is going to, for the views handed to kernels.

        One per pack, not one per operand: cuDNN's own variant pack carries no
        device at all. Read on demand; the backend path never asks.
        """
        if self._device is None:
            from ..frost.device import current_device

            self._device = current_device()
        return self._device

    def index_of(self, tensor_or_uid) -> int:
        """Index of a tensor's operand. KeyError when it is not caller-filled
        (a virtual intermediate, or a value the graph itself supplies)."""
        uid = tensor_or_uid if isinstance(tensor_or_uid, int) else tensor_or_uid.uid
        return self.index_of_uid[uid]

    def ptr(self, tensor_or_uid) -> int:
        return self.native.pointer(self.index_of(tensor_or_uid))

    def observed_bytes(self, index: int) -> int:
        """Bytes the PRODUCER guarantees addressable for operand ``index`` (-1: unknown, a bare address),
        recorded at normalization in the producer's own element width and untouched by graph
        re-description or overrides. An engine deriving a capacity divides by ITS element size."""
        return self.native.observed_bytes(index)

    def observed_device(self, index: int):
        """The producer's DLPack ``(device_type, device_id)`` for operand ``index``; ``(-1, -1)`` unknown."""
        return self.native.observed_device(index)

    def operands(self, indices):
        """The buffers for ``indices``, in one crossing."""
        return self.native.operands(list(indices), self.device)

    def operand(self, index: int):
        """A DLPack producer over one operand, for a kernel that needs an
        object rather than an address.

        This is the whole reason an engine never sees the caller's buffer: the
        kernel gets ours, built from the pointer and the geometry recorded at
        normalization. It costs more than handing the torch tensor straight
        through (measured +1.6 us per operand, because tvm-ffi reads a torch
        tensor through a C vtable and any python producer through a capsule) —
        which is an argument for making the producer a C type, not for keeping
        the caller's object.
        """
        return self.native.operand(index, self.device)

    def __len__(self) -> int:
        return len(self.uids)


@dataclass(frozen=True)
class PortIndices:
    """Per-node ``{port_name: operand index}``. A port with no caller operand — a
    virtual intermediate — is ABSENT, so ``.get(port) is None`` keeps meaning
    what it meant when these were buffers."""

    inputs: Dict[str, int]
    outputs: Dict[str, int]


def bind_ports(graph: "pygraph", variant_pack: VariantPack) -> Dict[Any, PortIndices]:
    """Join each node's wired ports with the operand layout. Strict: every
    non-virtual port must have an operand."""

    def resolve(node, ports, direction):
        indices = {}
        for port, t in ports.items():
            if t is None:
                continue
            index = variant_pack.index_of_uid.get(t.uid)
            if index is None:
                if t.is_virtual:
                    continue  # engine-internal intermediate
                raise ValueError(f"node {node.name!r}: no buffer for {direction} port {port!r} (tensor {t.name!r})")
            indices[port] = index
        return indices

    return {node: PortIndices(resolve(node, node.inputs, "input"), resolve(node, node.outputs, "output")) for node in graph.nodes}


def _view_over_address(address: int, tensor, node, port: str):
    """A DLPack view over a caller-supplied bare address, shaped by the IR.

    Only reachable when the caller passed an int for this port. Requires the
    graph to declare a dim and a dtype for it — an address carries neither, so
    if the graph does not say, nobody can.
    """
    from ..datatypes import _cudnn_to_frost_dtype_name
    from ..frost import buffers

    dtype = _cudnn_to_frost_dtype_name(tensor.data_type)
    if not tensor.dim or dtype is None:
        raise ValueError(
            f"node {node.name!r}: port {port!r} was given a bare device address, "
            f"but the graph declares no {'dim' if not tensor.dim else 'data_type'} for tensor "
            f"{tensor.name!r} — pass a buffer that carries its own shape and dtype, or declare them"
        )
    return buffers.DeviceView(address, tuple(tensor.dim), dtype, buffers.current_device_id())


@dataclass(frozen=True)
class NodeBuffers:
    """DEPRECATED, kept until every engine takes ``VariantPack``.

    Per-node ``{port_name: caller buffer}`` maps, the result of
    ``resolve_node_buffers``."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


def resolve_node_buffers(graph: "pygraph", uid_to_data: Dict[int, Any]) -> Dict[Any, NodeBuffers]:
    """Join each node's wired ports with the caller's variant pack into
    per-node ``NodeBuffers``. Strict: every non-virtual port must have a
    buffer. Torch tensors are detached here (DLPack and
    ``__cuda_array_interface__`` refuse ``requires_grad`` export; graph-level
    gradients belong to the backward nodes, not to autograd tracing through
    engines)."""

    def resolve(node, ports, direction):
        bufs = {}
        for port, t in ports.items():
            if t is None:
                continue
            b = uid_to_data.get(t.uid)
            if b is None:
                if t.is_virtual:
                    continue  # engine-internal intermediate
                raise ValueError(f"node {node.name!r}: no buffer for {direction} port {port!r} (tensor {t.name!r})")
            if type(b) is int:
                # A bare device address. The backend has always taken one
                # (_pygraph._describe), so a python engine must too, or the
                # same graph.execute() call succeeds or fails depending on
                # which plan the heuristics happened to pick. The geometry is
                # the one the graph declares for this port.
                b = _view_over_address(b, t, node, port)
            bufs[port] = b.detach() if hasattr(b, "detach") else b
        return bufs

    return {node: NodeBuffers(resolve(node, node.inputs, "input"), resolve(node, node.outputs, "output")) for node in graph.nodes}


class CompiledPlan:
    """A compiled (graph, plan) artifact. Subclass for real JIT engines."""

    # Set True once execute() takes VariantPack. Until then execute() is handed the
    # caller's raw {uid: buffer} map, as before. Migration flag; it goes away
    # with the last engine that does not set it.
    takes_variant_pack: bool = False

    def get_workspace_size(self) -> int:
        """Workspace bytes this plan needs at execute time (default 0)."""
        return 0

    def execute(self, graph: "pygraph", variant_pack: "VariantPack", ctx: ExecutionContext) -> None:
        raise NotImplementedError


class _EagerPlan(CompiledPlan):
    """Default CompiledPlan for simple eager engines (delegates to engine.execute)."""

    def __init__(self, engine: "BaseEngine", plan: PlanConfig):
        self.engine = engine
        self.plan = plan

    def get_workspace_size(self) -> int:
        return self.engine.get_workspace_size()

    def execute(self, graph, uid_to_data, ctx: ExecutionContext) -> None:
        self.engine.execute(graph, uid_to_data, ctx)


class BaseEngine(ABC):
    """Abstract base class for python graph execution backends.

    Attributes:
        name: Human-readable identifier.
        engine_id: Stable id in the shared flat engine-id space, in the reserved
            Python region (>= PYTHON_ENGINE_ID_BASE). Subclasses MUST declare it;
            the base default (None) is rejected when the manifest builds it.
        behavior_notes / numerical_notes: what this engine's plans are, in the
            same vocabulary the backend's plans answer in, so
            deselect_behavior_notes(...) and friends mean one thing across the
            ranked list. Declared, not inferred — an engine that says nothing
            is simply never filtered out by a note.
    """

    name: str = "base"
    # Subclasses MUST declare a stable id in the reserved python region; the
    # base intentionally has none so a forgotten override fails at registration
    # instead of silently colliding with another engine.
    engine_id: Any = None
    behavior_notes: tuple = ()
    numerical_notes: tuple = ()

    def __init__(self):
        pass

    # An engine answers for exactly ONE id: the one its manifest slot handed it.
    # (The id_end / owned_id_range block this used to declare was for engines
    # registered at runtime, which no longer exist.)

    def check_support(self, graph: "pygraph") -> None:
        """Raise to decline ``graph``.

        Decline ONLY via ``NotImplementedError`` or
        ``cudnn.cudnnGraphNotSupportedError`` (the classic unsupported-graph
        signal); any other exception is treated as an engine bug and propagates.
        Default: accept everything (subclasses should narrow this).
        """
        _ = graph

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: "ExecutionContext" = None) -> CompiledPlan:
        """Compile ``graph`` for ``plan`` (the expensive step; run once per
        graph/plan at build_plans() time). ``ctx`` carries the build context —
        handle and stream when available — so device-specific AoT compilers get
        their inputs explicitly instead of reading private graph state. Default
        wraps eager ``execute()``."""
        return _EagerPlan(self, plan)

    def get_workspace_size(self) -> int:
        """Workspace bytes for eager engines (compiled plans report their own)."""
        return 0

    def execute(self, graph: "pygraph", uid_to_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        """Eager execution hook (used by the default ``build_plan``).

        ``uid_to_data`` maps tensor UIDs (inputs + outputs) to their device
        data; call ``resolve_node_buffers`` to address buffers by port name.
        Write results directly into the caller-provided output buffers, on
        ``ctx.stream`` when set.
        """
        raise NotImplementedError(f"Engine '{self.name}' must implement execute() or build_plan()")

    # ------------------------------------------------------------------
    # Public knob vocabulary
    # ------------------------------------------------------------------
    # Every plan in the ranked list is described to callers as
    # ``(engine_id, {cudnn.knob_type: int})`` -- the same pair the backend's
    # plans answer in, so an autotuner persists and replays one shape of record
    # for both. An engine keeps whatever native knob form it likes in
    # ``PlanConfig.knobs`` and converts at this boundary. Knob TYPES are the
    # shared ``KnobType_t`` vocabulary (knobs.h): reuse a backend type where the
    # meaning matches, add to the frontend-only band otherwise -- never a
    # private namespace.

    def knobs_to_public(self, knobs: Any) -> Dict[Any, int]:
        """Native ``PlanConfig.knobs`` -> ``{cudnn.knob_type: int}``.

        Default: ``None`` (no tuning axes) is ``{}``; a dict is passed through.
        Engines with a typed knob object override this."""
        if knobs is None:
            return {}
        if isinstance(knobs, dict):
            return dict(knobs)
        raise TypeError(
            f"engine {self.name!r} carries knobs of type {type(knobs).__name__} but does not implement "
            "knobs_to_public(); every engine must speak the shared {cudnn.knob_type: int} vocabulary"
        )

    def knobs_from_public(self, public: Dict[Any, int]) -> Any:
        """``{cudnn.knob_type: int}`` -> native ``PlanConfig.knobs`` (replay path).

        Default: an empty dict means "no preference" (``None``); anything else
        is passed through as a dict. Engines with a typed knob object override."""
        if not public:
            return None
        return dict(public)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, engine_id={self.engine_id})"


def public_knobs_repr(public: Dict[Any, int]) -> str:
    """Stable, greppable rendering of a public knob dict: ``TILE_M=128, SPLIT_KV=2``.

    Keys render by knob-type NAME (``cudnn.knob_type`` members carry one; a bare
    int falls back to its number), sorted by name, so the same plan always
    prints the same way regardless of dict order. Plan names wrap it in
    ``engine[...]``."""
    items = []
    for k, v in public.items():
        name = getattr(k, "name", None) or str(int(k))
        items.append((name, int(v)))
    return ", ".join(f"{n}={v}" for n, v in sorted(items))
