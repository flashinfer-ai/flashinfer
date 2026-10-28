# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python graph representation for cuDNN Frontend.

All graph structure and attributes are kept in Python. Graph construction is
backend-agnostic; a backend is chosen at create_execution_plans() time by the
heuristics, and the backend-specific representation (e.g. the C++ cuDNN graph) is
generated lazily only then.

Execution flow (unification proposal):
    build ops -> create_execution_plans() -> heuristics -> selected backend
    (a python engine of the graph's family, or the cuDNN Graph backend by lazy
    lowering)

Example (pass torch tensors directly):
    >>> graph = pygraph()
    >>> C = graph.matmul(a_tensor, b_tensor)  # auto-creates descriptors
    >>> graph.execute({C: c_tensor})  # routes to a supporting engine, else cuDNN
"""

import atexit
import ctypes
from dataclasses import dataclass
import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

from ._device import ensure_current_context
from ._handle import Handle, to_backend_handle
from .datatypes import _buffer_dtype_to_cudnn, _dlpack_code_bits, _dlpack_lanes, _torch_to_cudnn_data_type
from .engines.base import ExecutionContext, VariantPack
from .engines.engine_ids import is_python_engine
from .graph_types import NodeType, Tensor, byte_size as _byte_size, describing_tensor, storage_geometry, storage_slot_bytes
from .nodes import Node, _row_major_stride

_LOG = logging.getLogger("cudnn.pygraph")

# A handle-less graph used to let the C++ PyGraph cudnnCreate a handle of its own
# at lowering and cudnnDestroy it from the destructor: a GC-timed backend release
# that, inside someone else's stream capture, poisons every later launch (Rule 8,
# #1151). Lend it a process default instead: one per (thread, device) -- cuDNN
# forbids sharing a handle across threads -- never re-streamed (a fresh handle
# runs on stream 0, exactly like the owned one did), destroyed at interpreter exit.
_DEFAULT_HANDLES = threading.local()
_DEFAULT_HANDLE_REGISTRY: List[Handle] = []
_DEFAULT_HANDLE_LOCK = threading.Lock()


def _default_backend_handle() -> int:
    from .frost.device import ambient_device

    device = ambient_device()
    by_device = getattr(_DEFAULT_HANDLES, "by_device", None)
    if by_device is None:
        by_device = _DEFAULT_HANDLES.by_device = {}
    handle = by_device.get(device)
    if handle is None:
        handle = by_device[device] = cudnn.create_handle()
        with _DEFAULT_HANDLE_LOCK:
            _DEFAULT_HANDLE_REGISTRY.append(handle)
    return to_backend_handle(handle)


def _destroy_default_handles() -> None:
    with _DEFAULT_HANDLE_LOCK:
        handles, _DEFAULT_HANDLE_REGISTRY[:] = list(_DEFAULT_HANDLE_REGISTRY), []
    for handle in handles:
        try:
            cudnn.destroy_handle(handle)
        except Exception:  # noqa: BLE001 -- CUDA may already be torn down at exit
            pass


atexit.register(_destroy_default_handles)


_PYTHON_ONLY_BOOL_ATTRS = frozenset({"attn_scale_prefolded"})


def _python_only_attr_is_set(attr: str, value) -> bool:
    """Whether a python-only op attribute is SET (-> python engines only).  A bool attribute
    (``attn_scale_prefolded``) is set when truthy: ``False`` / ``0`` / a false numpy bool are its
    default and leave the node backend-lowerable.  Every other attribute is set when not None --
    including an explicit ``softmax_precision=FLOAT``, which selects the f32 pipeline ON the python
    engines (the backend is not consulted; ``key()`` / ``serialize()`` refuse it, as documented)."""
    if value is None:
        return False
    if attr in _PYTHON_ONLY_BOOL_ATTRS:
        return bool(value)
    return True


def _detached_exception(exc: Exception) -> Exception:
    """Preserve the backend error type/message without retaining traceback frames.

    Clearing only __traceback__ leaves chained exceptions pointing at the graph.
    Never raise the stored copy either: raising attaches a new traceback to it.
    """
    return type(exc)(*exc.args)


def _is_dense(dim, stride) -> bool:
    """Row-major compact."""
    expect = 1
    for extent, step in zip(reversed(tuple(dim)), reversed(tuple(stride))):
        if extent != 1 and step != expect:
            return False
        expect *= extent
    return True


def _observed_span(data) -> Optional[int]:
    """Element span of a caller's buffer as the CALLER describes it (``1 + sum((size-1)*stride)``,
    numel when compact); None for a bare address or a producer without shape/stride."""
    if data is None or type(data) is int:
        return None
    try:
        shape, strides = tuple(data.shape), tuple(data.stride())
    except (AttributeError, TypeError):
        try:
            return int(data.numel())
        except (AttributeError, TypeError):
            return None
    n = 1
    for extent in shape:
        n *= int(extent)
    if n == 0:
        return 0
    return 1 + sum((int(size) - 1) * int(stride) for size, stride in zip(shape, strides))


def _producer_itemsize(data, declared_data_type) -> int:
    """Bytes per element of the caller's buffer as the caller types it; the declaration's slot width when the producer does not say."""
    es = getattr(data, "element_size", None)
    if callable(es):
        try:
            return int(es())
        except TypeError:
            pass
    slot = storage_slot_bytes(declared_data_type)
    return int(slot) if slot else 1


def _in_axis_order_of(shape, stride, reference_stride):
    """``(shape, stride)`` re-expressed in the axis order ``reference_stride`` uses.

    ``override_shapes`` speaks the GRAPH's declaration (a matmul's B is
    ``[batch, K, N]``); the slot holds what the caller's buffer reports (B is
    allocated ``(batch, N, K)``). Same memory, two orders — so an engine
    indexing an extent by position would read the wrong one. Both orders rank
    their axes the same way by stride, which gives the permutation.
    """
    if len(shape) != len(stride) or len(stride) != len(reference_stride):
        return tuple(shape), tuple(stride)
    by_stride = sorted(range(len(stride)), key=lambda i: -stride[i])
    reference = sorted(range(len(reference_stride)), key=lambda i: -reference_stride[i])
    permutation = [0] * len(stride)
    for rank, axis in enumerate(by_stride):
        permutation[reference[rank]] = axis
    return tuple(shape[a] for a in permutation), tuple(stride[a] for a in permutation)


def _python_plan_deserialize_args(args: tuple, kwargs: Dict[str, Any]):
    """``(handle, data, enforce_precompiled)`` when a deserialize call carries a python-engine
    plan blob, else None. Mirrors the two classic overloads: ``(data)`` and
    ``(handle, data, enforce_precompiled=...)``, positional or by the pybind names."""
    from . import _plan_serdes

    rest = list(args)
    handle = kwargs.get("handle_")
    data = kwargs.get("pyobj")
    if data is not None and handle is None and rest:
        handle = rest.pop(0)
    elif data is None and rest and _plan_serdes.is_python_plan(rest[0]):
        data = rest.pop(0)
    elif data is None and len(rest) >= 2 and _plan_serdes.is_python_plan(rest[1]):
        handle, data = rest[0], rest[1]
        rest = rest[2:]
    if data is None or not _plan_serdes.is_python_plan(data):
        return None
    enforce = kwargs.get("enforce_precompiled", rest[0] if rest else False)
    return handle, data, bool(enforce)


def cudnn_graph_not_supported(message: str) -> Exception:
    """The classic unsupported-graph error (built lazily: importing cudnn at
    module scope here would be circular)."""
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    return cudnn.cudnnGraphNotSupportedError(message)


if TYPE_CHECKING:
    from .engines.base import BaseEngine


@dataclass
class GraphContext:
    """Graph-level configuration defaults."""

    io_data_type: Any = None
    intermediate_data_type: Any = None
    compute_data_type: Any = None


class pygraph:
    """Pure Python graph representation.

    All graph structure and attributes are kept in Python. C++ is only
    used for execution via lazy lowering.

    Two constructor arguments are statements about the CALLER, not graph
    properties, and integrations pass them: ``is_override_shape_enabled``
    (the graph will execute with shape / stride overrides; forwarded to the
    backend graph) and the keyword-only ``is_cuda_graph_replay_expected``
    (``execute`` is captured into a CUDA graph and replayed, so the python
    engines' heuristics may lead with the plan that is fastest on the GPU
    alone; never forwarded to the backend, changes no numerics). After
    planning, ``selected_engine`` and ``backend_plan_entries()`` report which
    side serves the graph -- see their docstrings.

    Example:
        >>> graph = pygraph(io_data_type=cudnn.data_type.HALF)
        >>> A = graph.tensor(dim=[8, 64, 128], name="A")
        >>> B = graph.tensor(dim=[8, 128, 256], name="B")
        >>> C = graph.matmul(A, B, name="mm1")
        >>> C.set_output(True)  # outputs are explicit, like the classic API
        >>>
        >>> # Inspect graph
        >>> print(graph.nodes)  # [Node('mm1', MATMUL)]
        >>> print(graph.nodes[0].inputs)  # {"A": ..., "B": ...}
        >>> print(graph.nodes[0].params)  # {"padding": 0.0}
    """

    def __init__(
        self,
        # ---- classic pybind constructor, POSITIONALLY IDENTICAL (existing
        # callers pass name/handle/sm_count/... by position; guarded by
        # test_api_signature_parity) --------------------------------------
        name: str = "test_graph",
        io_data_type: Any = None,
        intermediate_data_type: Any = None,
        compute_data_type: Any = None,
        handle: Any = None,
        sm_count: Any = None,
        sm_version: Any = None,
        kernel_cache: Any = None,
        device_property: Any = None,
        is_dynamic_shape_enabled: bool = False,
        is_override_shape_enabled: bool = False,
        cuda_graph_retention: Any = None,
        *,
        is_cuda_graph_replay_expected: bool = False,
        **kwargs,
    ):
        self._context = GraphContext(
            io_data_type=io_data_type,
            intermediate_data_type=intermediate_data_type or io_data_type,
            compute_data_type=compute_data_type or io_data_type,
        )
        self._handle = handle  # cuDNN handle for the cuDNN lowering path
        # Classic graph-level configuration, forwarded verbatim to the C++
        # graph at lowering (**kwargs covers future binding args).
        self._cpp_graph_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._cpp_graph_kwargs["name"] = name
        for _k, _v in (
            ("sm_count", sm_count),
            ("sm_version", sm_version),
            ("kernel_cache", kernel_cache),
            ("device_property", device_property),
            ("cuda_graph_retention", cuda_graph_retention),
        ):
            if _v is not None:
                self._cpp_graph_kwargs[_k] = _v
        if is_dynamic_shape_enabled:
            self._cpp_graph_kwargs["is_dynamic_shape_enabled"] = True
        if is_override_shape_enabled:
            self._cpp_graph_kwargs["is_override_shape_enabled"] = True
        # The caller's statement about how it will RUN this graph, not a graph
        # property: it captures graph.execute into a CUDA graph and replays it,
        # so per-execute host costs (a plan's second launch, slab carving) are
        # paid once at capture, never per step. Python engines' heuristics may
        # then lead with the plan that is fastest on the GPU alone (the SDPA
        # analyzer records it as facts.cuda_graph_replay). Never forwarded to
        # the backend; changes no numerics and no support.
        self.is_cuda_graph_replay_expected = bool(is_cuda_graph_replay_expected)
        self._nodes: List[Node] = []
        self._tensors: Dict[str, Tensor] = {}
        self._tensor_by_uid: Dict[int, Tensor] = {}
        self._next_uid: int = 1
        self._node_count: Dict[str, int] = {}
        self._lowered_graph: Any = None
        self._is_validated: bool = False
        self._is_built: bool = False
        self._data_bindings: Dict[int, Any] = {}  # uid -> tensor data for auto-bound inputs

        # Backend routing (see engines/heuristics.py). Graph construction is
        # backend-agnostic. At create_execution_plans() the heuristics build a flat
        # ranked plan list (python engines + cuDNN) in one shared engine-id
        # space; each plan is dispatched by its id (is_python_engine -> python
        # registry, else lower to cuDNN). ``_plan_index`` selects the plan to run.
        self._plans: List[Any] = []  # list[PlanConfig], populated by create_execution_plans()
        self._planning_done: bool = False  # create_execution_plans() ran (one-shot)
        self._frozen: bool = False  # whole-surface freeze (set by _freeze())
        self._plan_index: int = 0
        self._backend_heuristics: Optional[List] = None  # heur modes for a backend plan
        self._cpp_plans_created: bool = False  # C++ create_execution_plans ran
        self._compiled_plans: Dict[int, Any] = {}  # plan_index -> CompiledPlan (python plans)
        self._plan_kernel_entries: Dict[int, List[Any]] = {}  # plan_index -> compiled_cache entries its build used (serialize)
        self._cpp_bog_done: bool = False  # C++ build_operation_graph ran
        self._cpp_tensors: Dict[int, Any] = {}  # IR uid -> lowered C++ tensor
        self._reserved_uids: set = set()  # user-specified uids _alloc_uid must skip
        self._ambiguous_names: set = set()  # duplicate labels: excluded from the name index
        self._candidates: Optional[List["BaseEngine"]] = None  # manifest matches + registered
        self._facts: Dict[Any, Any] = {}  # analyzer callable -> its record; see _facts_for()
        self._backend_declined: Optional[Exception] = None  # why the backend has no entries
        self._backend_entries: Optional[List[Any]] = None  # backend_plan_entries(), once
        self._backend_mode_spans: List[Any] = []  # (mode, lo, hi) over the C++ plan list
        self._barred_names: set = set()  # deselect_engines()
        self._workspace_limit: Optional[int] = None  # deselect_workspace_greater_than()
        self._note_filters: List[Any] = []  # (kind, note, keep) from the classic note filters
        self._plan_pinned: bool = False  # select_plan() => the walk is strict
        # Backend operand order + the reusable pointer array handed to execute.
        # Both are properties of the frozen graph, so they outlive any one call.
        self._sorted_uids: Optional[List[int]] = None
        self._declared_layout_native = None  # DeclaredLayout for _sorted_uids; see _declared_layout
        self._slot_of_uid = None  # uid -> slot of that order, built with the layout
        self._ordered_binding_schema = None  # native, bounded metadata cache; never retains caller buffers
        self._ordered_binding_uids = None
        self._selected_engine_cache = None  # (plan config, engine); see selected_engine

    # =========================================================================
    # Routing
    # =========================================================================

    @property
    def plans(self) -> List[Any]:
        """The ranked plan list (list[PlanConfig]) from create_execution_plans()."""
        return list(self._plans)

    @property
    def _selected_plan_config(self) -> Optional[Any]:
        if not self._plans or not 0 <= self._plan_index < len(self._plans):
            return None
        cfg = self._plans[self._plan_index]
        return cfg if is_python_engine(cfg.engine_id) else None

    def _engine_for(self, cfg) -> Optional["BaseEngine"]:
        """The python engine that owns ``cfg``'s id, or None for a backend entry."""
        if cfg is None or not is_python_engine(cfg.engine_id):
            return None
        owners = self._owners_for_id(cfg.engine_id)
        if not owners:
            raise KeyError(f"no python engine declares id {cfg.engine_id}")
        if len(owners) > 1:  # registration proves this cannot happen; assert it anyway
            raise KeyError(f"id {cfg.engine_id} is declared by {[e.name for e in owners]} — dispatch is ambiguous")
        return owners[0]

    def _owners_for_id(self, engine_id: int) -> List["BaseEngine"]:
        """The candidate engine answering for ``engine_id``, if any.

        The single owner lookup: dispatch, the ranking's output validation and
        replay all go through it. An id names exactly one engine (the manifest
        hands each slot one), so this is an equality test — and it must stay the
        same test the manifest fallback below makes."""
        out = [engine for engine in self._candidate_engines() if engine.engine_id == engine_id]
        if not out:
            # Not a candidate for THIS graph, but the id may still name an
            # in-tree engine: the manifest decodes it without anything being
            # registered. That is what lets create_execution_plan() replay a
            # recorded (engine_id, knobs) on a fresh graph.
            from .engines import manifest

            engine = manifest.engine_for_id(engine_id)
            if engine is not None:
                out.append(engine)
        return out

    def _barred_indices(self) -> set:
        """Plan INDICES excluded by ``deselect_engines()`` (classic: match by
        name substring) or by a note filter. Per index, not per engine: one
        engine proposes several knob configurations and a name may match only
        one of them."""
        barred = set()
        if self._barred_names:
            barred |= {i for i in range(len(self._plans)) if any(p in self.get_plan_name_at_index(i) for p in self._barred_names)}
        # Same rule the backend applies per note (plans.h::filter_behavior_notes):
        # bar a plan that HAS the note when deselecting, or LACKS it when selecting.
        for kind, note, keep in self._note_filters:
            for i, cfg in enumerate(self._plans):
                if self._engine_for(cfg) is None:
                    continue  # backend entry: the backend already filtered it
                if (note in self._notes_of(cfg, kind)) != keep:
                    barred.add(i)
        return barred

    @property
    def selected_engine(self) -> Optional["BaseEngine"]:
        """The python engine for the currently selected plan entry, or None for
        the backend path. Populated after create_execution_plans().

        Public API, read by integrations after planning: ``None`` means the
        cuDNN backend serves the selected entry; otherwise ``.name`` is the
        engine's manifest name (``sdpa_fwd_prefill_sm100`` /
        ``sdpa_fwd_prefill_sm107`` / ``sdpa_fwd_prefill_sm120``, ...), a stable
        identifier -- manifest names and slots are append-only, never renamed
        or reused -- so a caller may compare against one.
        ``get_plan_name_at_index`` reports the same name with the plan's
        public knobs appended in brackets.

        Cached, because ``execute()`` asks on every call and answering means
        walking every registered engine for the one declaring this id, re-deriving
        something that only ``select_plan`` can change. Keyed on
        the config OBJECT, so replanning invalidates it without needing a hook
        on every writer of ``_plan_index``.
        """
        cfg = self._selected_plan_config
        cached = self._selected_engine_cache
        if cached is not None and cached[0] is cfg:
            return cached[1]
        engine = self._engine_for(cfg)
        self._selected_engine_cache = (cfg, engine)
        return engine

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def tensor(
        self,
        # classic public tensor() signature, POSITIONALLY IDENTICAL (guarded by
        # test_api_signature_parity); classic unset sentinels (NOT_SET, -1,
        # reordering NONE) are normalized to None below
        dim: List[int],
        stride: Optional[List[int]] = None,
        data_type: Any = None,
        is_virtual: bool = False,
        is_pass_by_value: bool = False,
        ragged_offset: Optional[Tensor] = None,
        reordering_type: Any = None,
        name: str = "",
        uid: Optional[int] = None,
        ragged_offset_multiplier: int = 1,
        alignment_value: int = 1,
        **kwargs,
    ) -> Tensor:
        """Create a tensor.

        `alignment_value` promises every VALUE the tensor holds is a multiple of
        it; it constrains the contents, not the layout, and is not validated."""
        if not name:
            name = f"tensor_{len(self._tensors)}"
        if data_type is not None and getattr(data_type, "name", None) == "NOT_SET":
            data_type = None
        if reordering_type is not None and getattr(reordering_type, "name", None) == "NONE":
            reordering_type = None
        if uid == -1:  # classic unset sentinel
            uid = None

        if uid is not None:
            # User-owned uid, same rule as set_uid (_reuid_tensor): classic
            # tensors have no uid until assigned, so a user uid may land on an
            # eagerly auto-assigned one — the user wins, the auto holder is
            # renumbered; colliding with another USER uid is an error.
            holder = self._tensor_by_uid.get(uid)
            if holder is not None:
                if holder.uid_assigned:
                    raise ValueError(f"uid {uid} is already user-assigned to tensor {holder.name!r}")
                fresh = self._alloc_uid()
                self._tensor_by_uid[fresh] = holder
                if holder.uid in self._data_bindings:
                    self._data_bindings[fresh] = self._data_bindings.pop(holder.uid)
                holder.uid = fresh
            self._reserved_uids.add(uid)

        dim = list(dim)  # classic API accepts torch.Size / tuples
        t = Tensor(
            name=name,
            dim=dim,
            stride=list(stride) if stride else _row_major_stride(dim),
            data_type=data_type or (self._context.intermediate_data_type if is_virtual else self._context.io_data_type),
            is_virtual=is_virtual,
            is_pass_by_value=is_pass_by_value,
            ragged_offset=ragged_offset,
            reordering_type=reordering_type,
            ragged_offset_multiplier=ragged_offset_multiplier,
            alignment_value=alignment_value,
            uid=uid if uid is not None else self._alloc_uid(),
            uid_assigned=uid is not None,
            dim_assigned=True,  # graph inputs: the user specified the layout
            stride_assigned=stride is not None,
            **kwargs,
        )
        self._register_tensor(t)
        return t

    def tensor_like(self, template: Any, name: str = "", is_virtual: bool = False) -> Tensor:
        """Create tensor from another IR tensor or a DLPack object (e.g. torch).

        Classic parity: CPU (host) framework tensors become pass-by-value, like
        the C++ tensor_like (is_pass_by_value = device == CPU)."""
        if isinstance(template, Tensor):
            return self.tensor(
                dim=list(template.dim),
                stride=list(template.stride),
                data_type=template.data_type,
                is_virtual=is_virtual,
                is_pass_by_value=template.is_pass_by_value,
                name=name,
            )
        # Element strides via the DLPack protocol (classic tensor_like reads the
        # DLPack capsule). torch's .stride() is element units, but e.g. CuPy
        # exposes byte-unit .strides — so normalize any non-torch DLPack object
        # through torch.from_dlpack first.
        if hasattr(template, "stride") and callable(getattr(template, "stride", None)):
            dim = list(template.shape)
            stride = list(template.stride())
        else:
            try:
                import torch as _torch

                _view = _torch.from_dlpack(template)
                dim = list(_view.shape)
                stride = list(_view.stride())
            except Exception:  # noqa: BLE001 — no torch / exotic dlpack: assume dense
                dim = list(template.shape)
                stride = _row_major_stride(dim)

        data_type = None
        try:
            import flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes; import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

            data_type = cudnn.datatypes._torch_to_cudnn_data_type(template.dtype)
        except Exception:
            pass

        is_pbv = bool(getattr(getattr(template, "device", None), "type", None) == "cpu")
        return self.tensor(dim=dim, stride=stride, data_type=data_type, is_virtual=is_virtual, is_pass_by_value=is_pbv, name=name)

    def tensor_scalar(self, value: Any, scalar_type: Any, name: str = "") -> Tensor:
        """Create a pass-by-value scalar tensor (classic tensor_scalar parity)."""
        if not name:
            name = f"scalar_{len(self._tensors)}"
        t = Tensor(
            name=name,
            dim=[1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            is_pass_by_value=True,
            pass_by_value=value,
            scalar_type=scalar_type,
            uid=self._alloc_uid(),
        )
        self._register_tensor(t)
        return t

    def _check_mutable(self, what: str) -> None:
        if self._frozen:
            raise RuntimeError(f"cannot {what} after lowering/planning — the graph is frozen (planning is one-shot; build a new graph)")
        # a mutation while merely validated (python-engine graphs stay mutable
        # until planning) must re-validate later — never run on stale inference,
        # and never plan on candidates matched against the pre-mutation graph
        # (validate() may have cached them before the freeze).
        self._is_validated = False
        self._candidates = None

    def _freeze(self) -> None:
        """Freeze the ENTIRE public graph surface.

        Called at lowering and at planning, whichever happens first.

        Frozen-ness is ONE flag, on the graph. Every mutation route the public
        API offers goes through _check_mutable (the chained setters via
        Tensor._guard / Node._guard, and the op builders), so the flag alone is
        the guard. The structures the caller could otherwise mutate behind the
        API's back are made immutable in their own right rather than watched:
        node.inputs/outputs/params become MappingProxy views and dim/stride
        become tuples. The inspection surface stays fully readable for engines."""
        if self._frozen:
            return
        from types import MappingProxyType

        for node in self._nodes:
            node.inputs = MappingProxyType(dict(node.inputs))
            node.outputs = MappingProxyType(dict(node.outputs))
            node.params = MappingProxyType(dict(node.params))
        for t in self._tensor_by_uid.values():
            t.dim = tuple(t.dim) if t.dim else t.dim
            t.stride = tuple(t.stride) if t.stride else t.stride
        self._frozen = True

    def _rename_tensor(self, t: Tensor, name: str) -> None:
        """Atomic rename keeping the name index coherent. Classic parity: names
        are labels, so renaming ONTO an existing label is legal — the name just
        becomes ambiguous and leaves the unique-name index."""
        if name == t.name:
            return
        # Freeze-guarded like every other setter. A name is a label, but it is
        # also a variant-pack key: a compiled plan may be holding the name it
        # was built with, and the lowered graph keeps the old one, so a rename
        # after planning leaves two answers to "which tensor is 'q'" -- and
        # swapping two names would silently rebind buffers. Nothing needs to
        # rename a planned graph; build another one.
        self._check_mutable("rename a tensor")
        if self._tensors.get(t.name) is t:
            del self._tensors[t.name]
        object.__setattr__(t, "name", name)
        if name in self._tensors or name in self._ambiguous_names:
            self._tensors.pop(name, None)
            self._ambiguous_names.add(name)
        else:
            self._tensors[name] = t

    def _reuid_tensor(self, t: Tensor, uid: int) -> None:
        """Atomic re-uid keeping indexes/bindings coherent.

        Classic parity: classic tensors have NO uid until set_uid, while the IR
        assigns eagerly — so a user set_uid may land on an auto-assigned uid.
        The user wins: the auto holder is silently renumbered (auto uids are
        internal until lowering). Two USER-assigned uids colliding is an error.
        """
        if uid == t.uid:
            if not self._frozen:  # same-value set_uid is a no-op (classic allows it anytime)
                t.uid_assigned = True
            return
        self._check_mutable("re-uid a tensor")
        holder = self._tensor_by_uid.get(uid)
        if holder is not None:
            if holder.uid_assigned:
                raise ValueError(f"uid {uid} is already user-assigned to tensor {holder.name!r}")
            fresh = self._alloc_uid()  # renumber the auto holder
            self._tensor_by_uid[fresh] = holder
            if holder.uid in self._data_bindings:
                self._data_bindings[fresh] = self._data_bindings.pop(holder.uid)
            holder.uid = fresh
        self._tensor_by_uid.pop(t.uid, None)
        if t.uid in self._data_bindings:
            self._data_bindings[uid] = self._data_bindings.pop(t.uid)
        t.uid = uid
        t.uid_assigned = True
        self._reserved_uids.add(uid)
        self._tensor_by_uid[uid] = t

    def _alloc_uid(self) -> int:
        # Skip uids the user reserved via tensor(uid=...) — the Python IR owns
        # the whole uid namespace (see the uid-ownership note in _lower_to_cpp).
        while self._next_uid in self._reserved_uids:
            self._next_uid += 1
        uid = self._next_uid
        self._next_uid += 1
        return uid

    # Classic C++ auto-names op outputs "<node>::<OUTPUT_ENUM>" (graph_interface.h
    # output_tensor calls). Ports whose name differs from the classic enum are
    # mapped here so canonical names (wrapper.Graph lookups, JSON dumps) match.
    _CLASSIC_OUT_SUFFIX = {
        "inv_var": "INV_VARIANCE",
        "mean": "MEAN",
        "next_running_mean": "NEXT_RUNNING_MEAN",
        "next_running_var": "NEXT_RUNNING_VAR",
        "DScale": "DSCALE",
        "DBias": "DBIAS",
    }

    def _get_name(self, op: str, name: str) -> str:
        self._check_mutable(f"add a {op} op")
        if name:
            return name
        count = self._node_count.get(op, 0)
        self._node_count[op] = count + 1
        return f"{op}.{count}"

    def _make_output(self, name: str) -> Tensor:
        """Create a virtual output tensor. data_type is left unset: validate()'s
        inference assigns io/intermediate by the FINAL virtual state (classic
        semantics — a user set_output(True) without set_data_type gets io)."""
        return Tensor(
            name=name,
            is_virtual=True,
            uid=self._alloc_uid(),
        )

    def _register_tensor(self, t: Tensor) -> None:
        t.owner = weakref.ref(self)
        # Classic parity: names are debug LABELS — duplicates are legal
        # (pycudnnTest builds two 'weight' tensors). uid is the identity; the
        # name index serves only names that remain unique, and name-keyed
        # lookups on an ambiguous name raise instead of guessing.
        if t.name in self._tensors or t.name in self._ambiguous_names:
            self._tensors.pop(t.name, None)
            self._ambiguous_names.add(t.name)
        else:
            self._tensors[t.name] = t
        self._tensor_by_uid[t.uid] = t

    def _ensure_tensor(self, arg: Any, name: str = "") -> Tensor:
        """Convert arg to a Tensor descriptor if it isn't one already.

        If arg is a framework tensor (torch, jax, cupy, etc.), creates a
        descriptor via tensor_like() and stores the data binding for execute().
        """
        if isinstance(arg, Tensor):
            return arg
        desc = self.tensor_like(arg, name=name)
        self._data_bindings[desc.uid] = arg
        return desc

    # =========================================================================
    # Operations
    # =========================================================================

    def matmul(
        self,
        A: Any,
        B: Any,
        compute_data_type: Any = None,
        padding: float = 0.0,
        name: str = "",
    ) -> Tensor:
        """Matrix multiplication: C = A @ B.

        A and B can be Tensor descriptors or framework tensors (torch, jax, etc.).
        """
        name = self._get_name("matmul", name)
        A = self._ensure_tensor(A, name=f"{name}::A")
        B = self._ensure_tensor(B, name=f"{name}::B")

        node = Node(name, NodeType.MATMUL, compute_data_type or self._context.compute_data_type)
        node.inputs["A"] = A
        node.inputs["B"] = B
        node.params["padding"] = padding

        C = self._make_output(f"{name}::C")
        node.outputs["C"] = C
        self._register_tensor(C)

        self._nodes.append(node)
        return C

    # ---- Pointwise ops ------------------------------------------------------
    # ``params["mode"]`` is the op kind == the C++ pygraph method name (the
    # pointwise_mode enum is not exposed to Python, and the method name IS the
    # canonical semantic name), so lowering is a direct getattr dispatch — no
    # mode<->method mapping table to maintain. Extra scalar attributes
    # (negative_slope / clips / swish_beta / axis) live in params and are
    # forwarded at lowering; ops that take them get explicit builders below,
    # the uniform rest are generated from _POINTWISE_TENSOR_ARGS (the table
    # mirrors the pybind signatures — tensor-argument names per op — so both
    # positional and the classic keyword call styles work).

    _POINTWISE_TENSOR_ARGS: "dict[str, tuple]" = {
        # unary
        **{
            op: ("input",)
            for op in (
                "abs",
                "ceil",
                "cos",
                "elu",
                "erf",
                "exp",
                "floor",
                "gelu",
                "gelu_approx_tanh",
                "identity",
                "log",
                "logical_not",
                "neg",
                "reciprocal",
                "rsqrt",
                "sigmoid",
                "sin",
                "softplus",
                "sqrt",
                "tan",
                "tanh",
            )
        },
        # binary
        **{op: ("a", "b") for op in ("add", "add_square", "div", "logical_and", "logical_or", "mul", "sub")},
        **{op: ("input0", "input1") for op in ("max", "min", "mod", "pow")},
        **{op: ("input", "comparison") for op in ("cmp_eq", "cmp_ge", "cmp_gt", "cmp_le", "cmp_lt", "cmp_neq")},
        "bias": ("input", "bias"),
        "scale": ("input", "scale"),
        # backward (loss, input) -> dinput
        **{
            op: ("loss", "input")
            for op in (
                "elu_backward",
                "gelu_approx_tanh_backward",
                "gelu_backward",
                "sigmoid_backward",
                "softplus_backward",
                "tanh_backward",
            )
        },
        # ternary
        "binary_select": ("input0", "input1", "mask"),
    }
    # scalar attributes forwarded from params to the C++ call at lowering
    _POINTWISE_EXTRA_PARAMS = ("negative_slope", "lower_clip", "upper_clip", "swish_beta", "axis")

    def _pointwise(self, mode: str, inputs: list, name: str, compute_data_type: Any = None, extra_params: Optional[dict] = None) -> Tensor:
        """Internal helper for pointwise ops. ``mode`` == C++ pygraph method name."""
        inputs = [self._ensure_tensor(t, name=f"{name}::IN_{i}") for i, t in enumerate(inputs)]
        node = Node(name, NodeType.POINTWISE, compute_data_type or self._context.compute_data_type)
        node.params["mode"] = mode
        if extra_params:
            node.params.update({k: v for k, v in extra_params.items() if v is not None})
        for i, t in enumerate(inputs):
            node.inputs[f"IN_{i}"] = t

        out = self._make_output(f"{name}::OUT_0")
        node.outputs["OUT_0"] = out
        self._register_tensor(out)

        self._nodes.append(node)
        return out

    # Pointwise ops with extra scalar attributes: explicit builders.

    def relu(
        self, input: Any, negative_slope: Any = None, lower_clip: Any = None, upper_clip: Any = None, name: str = "", compute_data_type: Any = None
    ) -> Tensor:
        """ReLU (optionally leaky via negative_slope, and/or clipped)."""
        return self._pointwise(
            "relu", [input], self._get_name("relu", name), compute_data_type, dict(negative_slope=negative_slope, lower_clip=lower_clip, upper_clip=upper_clip)
        )

    def leaky_relu(self, input: Any, negative_slope: Any, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Leaky ReLU."""
        return self._pointwise("leaky_relu", [input], self._get_name("leaky_relu", name), compute_data_type, dict(negative_slope=negative_slope))

    def swish(self, input: Any, swish_beta: Any = None, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Swish / SiLU."""
        return self._pointwise("swish", [input], self._get_name("swish", name), compute_data_type, dict(swish_beta=swish_beta))

    def gen_index(self, input: Any, axis: int, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Generate index along an axis."""
        return self._pointwise("gen_index", [input], self._get_name("gen_index", name), compute_data_type, dict(axis=axis))

    def relu_backward(
        self, loss: Any, input: Any, negative_slope: Any = None, lower_clip: Any = None, upper_clip: Any = None, name: str = "", compute_data_type: Any = None
    ) -> Tensor:
        """ReLU backward."""
        return self._pointwise(
            "relu_backward",
            [loss, input],
            self._get_name("relu_backward", name),
            compute_data_type,
            dict(negative_slope=negative_slope, lower_clip=lower_clip, upper_clip=upper_clip),
        )

    def leaky_relu_backward(self, loss: Any, input: Any, negative_slope: Any, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Leaky ReLU backward."""
        return self._pointwise(
            "leaky_relu_backward", [loss, input], self._get_name("leaky_relu_backward", name), compute_data_type, dict(negative_slope=negative_slope)
        )

    def swish_backward(self, loss: Any, input: Any, swish_beta: Any = None, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Swish backward."""
        return self._pointwise("swish_backward", [loss, input], self._get_name("swish_backward", name), compute_data_type, dict(swish_beta=swish_beta))

    # NOTE: reduction / block-scale / MoE / conv / norms / structural ops are all
    # declared in _STRUCTURED_OPS (module tail) — one table entry per op, one
    # generic lowering branch. Only ops whose call shape doesn't fit the table
    # (matmul's positional ergonomics, sdpa's conditional kwargs) stay explicit.

    # NOTE: the sdpa family (sdpa / sdpa_backward / sdpa_fp8 / sdpa_mxfp8 /
    # sdpa_fp8_backward / sdpa_mxfp8_backward) is declared in _CAPTURED_OPS
    # (module tail): kwargs are captured generically — tensors become named
    # ports (port == C++ kwarg), scalars/enums/callbacks go to params verbatim,
    # dropout tuples are flattened per element — and lowering forwards them
    # verbatim, so the full C++ kwarg surface (~130 args across variants) is
    # supported without hand-mirroring each argument.

    # =========================================================================
    # Inspection
    # =========================================================================

    @property
    def nodes(self) -> List[Node]:
        """All nodes in the graph (a copy — the graph's own list is not a
        public mutation path)."""
        return list(self._nodes)

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors by name (a copy — see nodes)."""
        return dict(self._tensors)

    @property
    def context(self) -> GraphContext:
        """Graph context."""
        return self._context

    def find_tensor(self, name_or_uid: Union[str, int]) -> Optional[Tensor]:
        """Find tensor by name or UID."""
        if isinstance(name_or_uid, int):
            return self._tensor_by_uid.get(name_or_uid)
        if name_or_uid in self._ambiguous_names:
            raise ValueError(f"tensor name {name_or_uid!r} is ambiguous (duplicate labels are legal; look up by uid or Tensor)")
        return self._tensors.get(name_or_uid)

    def get_node(self, name: str) -> Optional[Node]:
        """Find node by name."""
        return next((n for n in self._nodes if n.name == name), None)

    def get_inputs(self) -> List[Tensor]:
        """Get non-virtual input tensors."""
        produced = {t.uid for n in self._nodes for t in n.outputs.values() if t}
        return [t for t in self._tensors.values() if not t.is_virtual and t.uid not in produced]

    def get_outputs(self) -> List[Tensor]:
        """Get non-virtual output tensors."""
        return [t for t in self._tensors.values() if not t.is_virtual and any(t.uid == o.uid for n in self._nodes for o in n.outputs.values() if o)]

    def inspect(self) -> Dict[str, Any]:
        """Return graph structure for inspection."""
        return {
            "context": {
                "io_data_type": str(self._context.io_data_type),
                "compute_data_type": str(self._context.compute_data_type),
            },
            "nodes": [
                {
                    "name": n.name,
                    "type": n.node_type.name,
                    "inputs": {k: v.name for k, v in n.inputs.items()},
                    "outputs": {k: v.name for k, v in n.outputs.items()},
                    "params": n.params,
                }
                for n in self._nodes
            ],
            "tensors": {
                name: {"dim": t.dim, "stride": t.stride, "dtype": str(t.data_type), "is_virtual": t.is_virtual, "uid": t.uid}
                for name, t in self._tensors.items()
            },
        }

    # =========================================================================
    # Build & Execute
    # =========================================================================

    def validate(self) -> None:
        """Validate graph and infer properties.

        Classic parity: op outputs stay VIRTUAL unless the user marks them
        with set_output(True). A leaf output is NOT auto-marked — discarding
        an op result (e.g. the Stats of a training SDPA) is legal classic
        usage, and auto-marking it would make its uid required in the variant
        pack.
        """
        try:
            for node in self._nodes:
                node.infer_properties(self._context)
                # Table-driven shape inference, topologically: builder-time infer
                # only sees graph-input dims; chained ops (e.g. conv on a virtual
                # relu output) get their output dims here, once inputs are known.
                spec_entry = _STRUCTURED_BY_TYPE.get(node.node_type) or _CAPTURED_BY_TYPE.get(node.node_type)
                if spec_entry:
                    _, spec = spec_entry
                    infer = spec.get("infer", {})
                    # builder-time dims of a "reinfer" port are provisional: they depend on a sibling the caller may set later
                    reinfer = spec.get("reinfer", ())
                    for oport, out_t in node.outputs.items():
                        if out_t is not None and (not out_t.dim or (oport in reinfer and not out_t.dim_assigned)):
                            try:
                                d = infer.get(oport, lambda n: None)(node)
                            except Exception:  # noqa: BLE001 — best-effort
                                d = None
                            if d:
                                reinferred = bool(out_t.dim)
                                out_t.dim = list(d)
                                if not (reinferred and out_t.stride_assigned):  # a re-inferred port keeps the caller's stride
                                    out_t.stride = _row_major_stride(out_t.dim)
                node.validate()
        except ValueError:
            # Inference can reject shapes before the family validator runs
            # (e.g. matmul batch broadcasting). Let it report its typed semantic
            # error; if it declines or accepts, preserve the original rejection.
            if self._backend_lowerable() and self._lowered_graph is None:
                validator = self._python_native_validator()
                if validator is not None:
                    validator(self)
            raise
        for t in self._tensors.values():
            if t.dim and not t.stride:  # classic: stride optional, row-major inferred
                t.stride = _row_major_stride(t.dim)
            if not t.is_pass_by_value:
                t.validate()
        # Classic parity: C++ validation happens HERE, so a config the backend
        # rejects raises from validate() where callers catch it to skip. Skipped
        # only for a graph the backend has no lowering for (GDN/KDA/...), or when
        # the caller registered its own engine — the pre-existing exemption.
        #
        # A graph whose engine family declares a python-native validator, and for
        # which the manifest offers a python engine, validates natively instead
        # (issue #704): the eager C++ round-trip couples a graph a FROST engine
        # fully serves to the installed backend's version (an attribute the
        # backend is too old to *validate* but will never execute). The family's
        # validator runs its version- and arch-agnostic semantic rules with the
        # classic error types and returns False -- classic lowering -- when the
        # graph holds a node it does not cover. The backend's own verdict is
        # deferred to planning, where a decline is recorded (backend_plan_entries)
        # and surfaced by plan() only if no python engine proposes a plan either.
        if self._backend_lowerable() and self._lowered_graph is None:
            validator = self._python_native_validator()
            if not (validator is not None and validator(self)):
                self._lowered_graph = self._lower_to_cpp()
                try:
                    self._lowered_graph.validate()
                    self._verify_uid_ownership()
                except Exception:
                    # A rejected lowering must not survive: the next validate()
                    # would find _lowered_graph set, skip the backend check, and mark
                    # the rejected graph valid.
                    self._reset_lowered_state()
                    raise
        # Only a graph that passed EVERY check above is validated: a rejection
        # (python-native or C++) leaves the flag False so build()/plan() re-run
        # validate() and raise again instead of planning a rejected graph.
        self._is_validated = True

    def _python_native_validator(self):
        """The graph's family validator when validate() may skip the eager C++
        lowering: the graph belongs to one engine family, that family declares a
        python-native validator (manifest.EngineFamily.validator), AND the manifest
        offers a python engine for it (frost engines available and enabled).
        None otherwise: without a candidate the backend is the only possible
        server, so classic timing -- raise its rejection from validate() -- must
        hold. The validator itself still returns False for a graph holding a node
        it does not cover, which also means classic lowering."""
        from .engines import manifest

        if not self._nodes:
            return None
        family = manifest.family_for(self)
        if family is None:
            return None
        validator = manifest.resolve_validator(family)
        if validator is None or not self._candidate_engines():
            return None
        return validator

    def _backend_guard_reason(self) -> Optional[str]:
        """A family's reason the backend must NOT be consulted for this graph at planning, or None
        (``manifest.EngineFamily.backend_guard``: ``guard(graph, facts) -> Optional[str]``).  Answered from
        the family's facts -- un-memoized while the graph is still mutable, exactly as ``_facts_for`` --
        and recorded as the backend's decline by the planning sequence, ``_finalize_backend_layout`` (before
        the first C++ lowering) and ``backend_plan_entries`` (before the heuristics query).  The two paths
        that create backend plans OUTSIDE that sequence refuse through ``_refuse_guarded_backend``: an
        explicit ``create_execution_plan(<backend engine id>, knobs)`` (``_append_backend_plan``) and a late
        backend lowering for a classic query (``_lower_backend_plan``).  ``key()``, ``serialize()`` and a
        direct ``build_operation_graph()`` only lower (C++ lowering, validate, build_operation_graph): no
        plan is created there, and that part of the sequence completes on the guarded domain."""
        from .engines import manifest

        family = manifest.family_for(self) if self._nodes else None
        if family is None:
            return None
        guard, analyzer = manifest.resolve_backend_guard(family), manifest.resolve_analyzer(family)
        if guard is None or analyzer is None:
            return None
        facts = self._facts_for(analyzer)
        return None if facts is None else guard(self, facts)

    def build_operation_graph(self) -> None:
        """Validate the graph; lower to C++ when no python engines are registered.

        Backend selection is deferred to create_execution_plans() (the heuristics
        stage). With python engines registered, nothing is lowered here (a graph
        routed to a python engine never touches C++). Without them — the classic
        sequencing — lowering happens now, so plan-configuration and query
        methods (deselect_engines, get_engine_count, ...) work between
        build_operation_graph() and create_execution_plans(), exactly as on the
        classic API (they delegate to the lowered C++ graph via __getattr__).
        """
        if not self._is_validated:
            self.validate()
        if self._lowered_graph is not None and not self._cpp_bog_done:
            self._lowered_graph.build_operation_graph()
            self._cpp_bog_done = True
            self._sync_ir_shapes_from_backend()

    def _sync_ir_shapes_from_backend(self) -> None:
        """After the backend's shape/layout inference (build_operation_graph),
        reflect the REAL dim/stride back into the IR tensors. The IR's own
        inferred strides are provisional row-major; the backend applies
        classic per-op layout inference (channels-last conv etc.), and
        consumers of the IR (wrapper.Graph buffer allocation, engines,
        introspection) must see the layout that will actually execute.

        USER-assigned dim/stride are never overwritten: they describe the
        caller's actual buffers and are the authoritative API-level layout.
        Some composite nodes remap their C++ input descriptors in place to
        the backend's internal convention (the fp8/mxfp8 SDPA nodes swap K's
        dim/stride to the KT = (B, H, D, S) view the backend wants — same
        memory, transposed description); reflecting that back would corrupt
        the IR's user-facing (B, H, S, D) contract that graph analyzers and
        python engines consume."""
        for ir_uid, cpp_t in self._cpp_tensors.items():
            ir = self._tensor_by_uid.get(ir_uid)
            if ir is None:
                continue
            try:
                d, st = cpp_t.get_dim(), cpp_t.get_stride()
            except Exception:  # noqa: BLE001 — some tensors have no dims (scalars)
                continue
            if d and not ir.dim_assigned:
                object.__setattr__(ir, "dim", tuple(d))  # sealed (graph is frozen)
            if st and not ir.stride_assigned:
                object.__setattr__(ir, "stride", tuple(st))

    def create_execution_plans(self, heuristics: Optional[List] = None) -> None:
        """Build the ranked execution-plan list (the dispatch stage).

        Both sides are collected here and ranked into ONE flat list of
        ``PlanConfig(engine_id, knobs)``: the python engines that claim the
        graph (discovered from ``engines.manifest`` — no registration call, no
        environment variable) and the backend's own ranked recommendation
        (``backend_plan_entries()``, [] when the backend declined the graph or
        is not installed). ``engines.heuristics.rank`` decides the order;
        ``build_plans()`` walks it.

        Args:
            heuristics: cuDNN heuristic modes for the backend's recommendation.
        """
        if not self._is_validated:
            self.validate()

        from .engines.heuristics import rank

        # One-shot planning (classic conformance: the C++ graph never supported
        # re-planning — a second call there appends plans by accident, and no
        # user re-plans). Plan once; to plan differently, build a new graph
        # (IR construction is microseconds). Autotune re-selects WITHIN this
        # plan set via select_plan(). Explicit state flag, not an
        # is-the-list-nonempty proxy.
        if self._planning_done:
            raise RuntimeError(
                "create_execution_plans() was already called on this graph; planning is one-shot — build a new graph to re-plan, or use select_plan() to switch plans"
            )
        self._backend_heuristics = heuristics  # read by backend_plan_entries()

        # Finalize -> freeze -> analyze, in that order, BEFORE any ranking runs.
        # build_operation_graph() is where the backend's own layout inference
        # lands (a no-op for a graph that was never lowered), and it writes
        # through object.__setattr__ precisely to bypass the freeze — so the
        # snapshot has to come after it, not merely after validate(). Doing it
        # here rather than inside the heuristics keeps the epoch boundary out of
        # overridable policy: every engine probe and the ranking then read one
        # set of facts describing a graph that can no longer change.
        self._finalize_backend_layout()
        self._freeze()
        self._attach_facts()

        plans = list(rank(self, self._candidate_engines(), self.backend_plan_entries(), self._backend_heuristics))
        # Validate the FINAL router output: every entry must name an engine this
        # graph can actually dispatch to.
        from .engines.engine_ids import is_python_engine

        known = {e.engine_id for e in self._candidate_engines()}
        for cfg in plans:
            if is_python_engine(cfg.engine_id) and not self._owners_for_id(cfg.engine_id):
                raise ValueError(f"heuristics produced a plan for unknown python engine_id {cfg.engine_id} (known: {sorted(known)})")
        if not plans:
            # Say WHY, both sides, or the user is left guessing which side had nothing: the backend's own
            # rejection and every python engine's reason (a failure-path re-run of check_support; the
            # engines that accepted the graph but whose family proposed nothing are named too).
            from .engines.heuristics import decline_reasons

            why = []
            if self._backend_declined is not None:
                why.append(f"the backend declined: {self._backend_declined}")
            declined, accepted = decline_reasons(self._candidate_engines(), self)
            if declined:
                why.append("python engines declined: " + "; ".join(msg if msg.startswith(name) else f"{name}: {msg}" for name, msg in declined.items()))
            if accepted:
                why.append("python engines that accepted the graph but proposed no plan: " + ", ".join(accepted))
            raise cudnn_graph_not_supported("no engine — python or backend — proposed a plan for this graph" + (f" ({'; '.join(why)})" if why else ""))
        self._plans = plans
        self._planning_done = True
        self._plan_index = 0

    def _candidate_engines(self) -> List["BaseEngine"]:
        """The python engines this graph may dispatch to: the in-tree manifest
        family whose coarse key matches, and nothing else — an engine is known
        because the library ships it, not because someone handed it over.
        (Candidate ORDER is not rank — ``heuristics.rank`` ranks.) Cached: the
        graph is frozen at planning time anyway."""
        if self._candidates is None:
            from .engines import manifest

            self._candidates = list(manifest.engines_for(self))
        return self._candidates

    def _finalize_backend_layout(self) -> None:
        """Let the backend's layout inference land before the graph is frozen.

        Lowering and reflecting strides back is how the GRAPH learns what it
        will execute with, so it runs for every backend-lowerable graph -- a
        path that skipped it for graphs a python engine might claim is what
        left the snapshot unenforceable, since _sync_ir_shapes_from_backend
        writes through object.__setattr__ to bypass the freeze.

        A failure here is the backend DECLINING (a python engine may still
        serve the graph), recorded as backend_plan_entries() records one. Not
        routed through that, which also runs the C++ plan query that
        the heuristics may never ask for.
        """
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        if not self._backend_lowerable():  # no backend lowering for this op at all
            return
        reason = self._backend_guard_reason()
        if reason is not None:
            # A family-declared domain where consulting the backend is unsafe (a measured crash of its
            # planner): recorded as the backend's decline, never lowered (EngineFamily.backend_guard).
            _LOG.info("backend not consulted for this graph: %s", reason)
            self._backend_declined = cudnn_graph_not_supported(reason)
            return
        try:
            self._lower_backend_graph()
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError, ImportError, AttributeError) as exc:
            _LOG.warning("backend could not build this graph, treating as a decline: %s", exc)
            self._backend_declined = _detached_exception(exc)
            self._reset_lowered_state()

    def _attach_facts(self) -> None:
        """Describe this frozen graph in its family's vocabulary.

        Part of planning, not a call anyone makes. Runs AFTER _freeze(): a
        snapshot of a graph that can still change means chasing every mutation
        point, and missing one leaves facts describing a graph that is gone.
        A graph no family claims carries no payload.
        """
        from .engines import manifest

        family = manifest.family_for(self)
        if family is not None and family.offered_ids():
            analyzer = manifest.resolve_analyzer(family)
            if analyzer is not None:
                self._facts_for(analyzer)

    def _facts_for(self, analyzer):
        """The record ``analyzer`` produced for this graph, computing it once.

        Keyed by the analyzer itself so the ranking and the engine reach ONE
        record with no name to keep in sync -- two extractions of a graph is
        how a feature vector drifts from what the kernel does.
        """
        if not self._frozen:
            # Still mutable (a direct engine probe before planning): answer, but
            # do not remember. Memoizing a graph that can still change is what
            # made facts go stale across two validate() calls.
            return analyzer(self)
        if analyzer not in self._facts:
            self._facts[analyzer] = analyzer(self)
        return self._facts[analyzer]

    def _unlowerable_node(self) -> Optional[Node]:
        """The first node with no C++ lowering, or None when the whole graph has one.

        This — not "does a python engine claim the graph" — is what decides
        whether ``validate()`` lowers eagerly. Classic error timing depends on
        it: a config the backend rejects (alibi with a right bound, bottom-right
        causal with dropout, ...) must raise ``cudnnGraphNotSupportedError`` from
        ``validate()``, where callers catch it to skip. An op the backend has no
        node for at all (GDN/KDA/...) must NOT be lowered — the lowering loop
        silently skips such nodes and would hand C++ an incomplete graph.
        """
        # The cuDNN backend block-scale descriptor currently accepts the
        # matrix-style block-size vector, but not the rank-5
        # (1, C-block, 1, 1, 1) vector used by convolution. Let the Frost
        # convolution engine consume that public graph directly instead of
        # failing backend lowering before engine selection. Block-scale matmul
        # keeps its existing C++ lowering path.
        has_convolution = any(node.node_type == NodeType.CONV_FPROP for node in self._nodes)
        for node in self._nodes:
            if has_convolution and node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE and len(node.params.get("block_size") or ()) == 5:
                return node
            if node.node_type in (NodeType.MATMUL, NodeType.POINTWISE):
                continue
            spec_entry = _CAPTURED_BY_TYPE.get(node.node_type) or _STRUCTURED_BY_TYPE.get(node.node_type)
            if spec_entry is None:
                return node  # no lowering branch at all
            if spec_entry[1].get("python_only"):
                return node  # declared python-only: lowering raises by design
            if any(_python_only_attr_is_set(attr, node.params.get(attr)) for attr in spec_entry[1].get("python_only_attrs", ())):
                return node  # an op attribute the backend has no field for is SET: python engines only
            if any(node.outputs.get(port) is not None for port in spec_entry[1].get("python_only_out_kwargs", ())):
                return node  # an output the backend cannot produce (sf_o) is requested: python engines only
            if any(node.inputs.get(port) is not None for port in spec_entry[1].get("python_only_in_kwargs", ())):
                return node  # an input the backend has no field for (sdpa_mxfp8 scale_o) is bound: python engines only
        return None

    def _backend_lowerable(self) -> bool:
        return self._unlowerable_node() is None

    def backend_plan_entries(self) -> List[Any]:
        """The backend's own ranked ``(engine_id, knobs)`` recommendation, as
        PlanConfig entries carrying their C++ plan index.

        Public API: an integration that must keep the cuDNN backend reachable
        for a graph (a Stats layout only the backend serves everywhere) asks
        this after ``create_execution_plans()`` and reads ``[]`` as "the
        backend does not serve it" -- the answer is the backend's own (or the
        family's recorded backend guard), independent of which python engines
        are offered.

        Returns [] for the two things that legitimately mean "no backend entries":
        the backend DECLINED the graph (``cudnnGraphNotSupportedError``) or is not
        installed at all (``AttributeError``/``ImportError`` reaching for it).
        Everything else — a CUDA failure, a bad handle, a backend API error, all
        of which the binding surfaces as ``RuntimeError`` — propagates: swallowing
        those would silently run a python engine on a failing device and call it a
        routing decision.

        The query costs a real ``create_execution_plans`` on the lowered graph;
        there is no cheaper way to obtain a RANKED list, since
        ``get_engine_and_knobs_at_index`` indexes the plan list, not the engine list.

        Answered ONCE per graph: a second C++ create_execution_plans() appends
        to the same plan list (``enqueue_engine_configs`` -> ``back_inserter``),
        so re-querying would report every backend plan twice. The heuristics may
        therefore call it freely to place the entries where it wants.
        """
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        from .engines.base import PlanConfig

        if self._backend_entries is not None:
            return self._backend_entries

        unlowerable = self._unlowerable_node()
        if unlowerable is not None:
            # Known statically: the backend has no node for this op (GDN/KDA/...).
            # Asking anyway costs a lowering that must fail, once per graph, and
            # reports an expected routing outcome as a warning. Worded as the
            # backend words it — callers match on "No valid engine configs".
            name = unlowerable.node_type.name
            self._backend_declined = cudnn_graph_not_supported(
                f"No valid engine configs for {name}: {name.lower()} has no cuDNN backend lowering; it runs on a python engine"
            )
            self._backend_entries = []
            return self._backend_entries

        reason = self._backend_guard_reason()
        if reason is not None:
            # A family-declared domain where consulting the backend is unsafe (a measured crash of its
            # planner): recorded as the backend's decline, never queried (EngineFamily.backend_guard).
            self._backend_declined = cudnn_graph_not_supported(reason)
            self._backend_entries = []
            return self._backend_entries

        try:
            # Can the backend express this graph at all? Only the types the
            # BINDING raises count; an AssertionError/KeyError/TypeError here is
            # our own translator bug and must not read as a decline.
            self._lower_backend_graph()
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError, ImportError, AttributeError) as exc:
            self._backend_declined = _detached_exception(exc)
            # RuntimeError here is overloaded by the binding: a rejected
            # descriptor (cannot represent) and a failing device look the same.
            # Treat it as a decline so a python engine can still serve the
            # graph, but say so at WARNING and re-raise it if nothing does.
            _LOG.warning("backend could not lower this graph, treating as a decline: %s", exc)
            # Roll back: a half-lowered graph makes a later build_operation_graph()
            # walk into the descriptor that just failed.
            self._reset_lowered_state()
            return self._backend_entries
        try:
            # "Which engines does it offer?" — here only an unsupported-graph
            # answer is a decline. A CUDA failure, a bad handle or a backend API
            # error arrives as RuntimeError and must NOT be read as routing:
            # swallowing it would run a python engine on a failing device and
            # call that a decision.
            self._create_backend_plans()
        except cudnn.cudnnGraphNotSupportedError as exc:
            # Lowering succeeded, so the only decline left is "no engine for it";
            # an AttributeError here is a binding mismatch, not an absent backend.
            self._backend_declined = _detached_exception(exc)
            _LOG.debug("backend has no engine for this graph: %s", exc)
            self._backend_entries = []
            return self._backend_entries
        from .engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

        # One entry per DISTINCT (engine, knobs). Graph::create_execution_plans
        # checks override_heuristics_query() and returns before reading the mode
        # -- deterministic SDPA backward and FP8 backward override -- so every
        # per-mode call appends the same config. First index wins; it is the one
        # whose mode span is real.
        entries, seen = [], set()
        for i in range(self._lowered_graph.get_execution_plan_count()):
            engine_id, knobs = self._lowered_graph.get_engine_and_knobs_at_index(i)
            key = (engine_id, repr(knobs))
            if key in seen:
                continue
            seen.add(key)
            entries.append(PlanConfig(engine_id, knobs, cpp_index=i, mode=self._mode_of_backend_plan(i)))
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        asked_oss = any(h == cudnn.heur_mode.OPENSOURCE for h in (self._backend_heuristics or []))
        if asked_oss or (not entries and self._lowered_graph.get_engine_count()):
            # OSS candidates are registered in the backend's CANDIDATE space and
            # never surface as plans, so they cannot be enumerated: one
            # delegating entry, built and run by C++ itself. FIRST, because
            # Graph::build_plans tries the OSS engine before engine_configs and
            # returns as soon as it builds — ranking it after the concrete plans
            # would hand an OPENSOURCE caller a native kernel instead.
            entries.insert(0, PlanConfig(BACKEND_HEURISTIC_ENGINE_ID))
        self._backend_entries = entries
        return entries

    def get_execution_plan_count(self) -> int:
        """The number of ranked plans for this graph — ONE list, python engines
        and backend engines alike (``graph.plans``).

        The classic at-index APIs (``get_plan_name_at_index`` /
        ``build_plan_at_index`` / ``execute_plan_at_index`` /
        ``get_workspace_size_plan_at_index``) address this same list, so code
        that loops over the count picks up python engines with no change.
        """
        if self._planning_done:
            return len(self._plans)
        if self._lowered_graph is not None:
            # classic pre-planning sequencing: delegate, C++ reports its state
            return self._lowered_graph.get_execution_plan_count()
        return 0  # classic: an unplanned graph has zero plans (not an error)

    def get_plan_name_at_index(self, index: int) -> str:
        """Name of the plan at ``index`` in the ranked list. A python plan
        reports its engine name plus its public knobs when it has any
        (``sdpa_fwd_prefill_sm100[TILE_M=128, TILE_N=128]``, sorted by knob
        name so the same plan always prints the same); a backend plan reports
        the backend's own name."""
        if not self._planning_done:
            return self._lowered_graph.get_plan_name_at_index(index)
        cfg = self._plans[self._check_plan_index(index)]
        eng = self._engine_for(cfg)
        if eng is not None:
            from .engines.base import public_knobs_repr

            public = eng.knobs_to_public(cfg.knobs)
            return f"{eng.name}[{public_knobs_repr(public)}]" if public else eng.name
        cfg = self._materialize_backend_plan(index)
        if cfg.cpp_index is None:
            # Delegating entry: the backend holds candidates it does not expose
            # as plans, so there is no C++ index to name (indexing an empty
            # engine_configs is unchecked on the C++ side).
            return "backend_heuristics"
        self._lower_backend_plan()
        return self._lowered_graph.get_plan_name_at_index(cfg.cpp_index)

    def select_plan(self, index: int) -> "pygraph":
        """Pin the plan at ``index`` in the ranked list (``graph.plans``).

        A pin is STRICT: ``build_plans()`` starts there and a decline raises
        instead of quietly running a different plan."""
        if not self._planning_done:
            raise RuntimeError("call create_execution_plans() before select_plan()")
        if not 0 <= index < len(self._plans):
            raise IndexError(f"plan index {index} out of range for {len(self._plans)} plan(s) (graph.plans)")
        self._plan_index = index
        self._plan_pinned = True
        self._is_built = False
        return self

    def deselect_workspace_greater_than(self, workspace_limit: int) -> "pygraph":
        """Classic: exclude plans needing more than ``workspace_limit`` bytes.

        Recorded here as well as forwarded, because a python plan's workspace is
        known only to its CompiledPlan and the backend cannot filter it."""
        self._workspace_limit = workspace_limit
        if self._lowered_graph is not None:
            self._lowered_graph.deselect_workspace_greater_than(workspace_limit)
        return self

    def deselect_engines(self, engine_names: List[str]) -> "pygraph":
        """Classic: exclude plans whose name contains any of ``engine_names``.

        Applies to the whole ranked list — a python engine name excludes that
        engine exactly as a backend engine name excludes a backend plan."""
        self._barred_names.update(engine_names)
        if self._lowered_graph is not None:
            self._lowered_graph.deselect_engines(list(engine_names))
        if self._is_built and self._planning_done and self._plan_index in self._barred_indices():
            self._is_built = False  # the built plan is no longer allowed to run
        return self

    def deselect_behavior_notes(self, notes: List[Any]) -> "pygraph":
        """Classic: exclude plans carrying any of ``notes``."""
        return self._filter_notes("behavior", notes, keep=False)

    def select_behavior_notes(self, notes: List[Any]) -> "pygraph":
        """Classic: keep only plans carrying ``notes``."""
        return self._filter_notes("behavior", notes, keep=True)

    def deselect_numeric_notes(self, notes: List[Any]) -> "pygraph":
        """Classic: exclude plans carrying any of ``notes``."""
        return self._filter_notes("numerical", notes, keep=False)

    def select_numeric_notes(self, notes: List[Any]) -> "pygraph":
        """Classic: keep only plans carrying ``notes``."""
        return self._filter_notes("numerical", notes, keep=True)

    def _filter_notes(self, kind: str, notes: List[Any], *, keep: bool) -> "pygraph":
        """Record a note filter and forward it to the backend.

        Recorded here as well as forwarded because a python plan's notes are
        declared by its engine, not by the backend's engine config — without
        this the four classic note filters silently skipped every python plan.
        """
        self._note_filters.extend((kind, note, keep) for note in notes)
        # Only once the backend HAS plans: its filters mark indices into
        # engine_configs (plans.h::filter_behavior_notes), so a filter forwarded
        # while that list is empty marks nothing and is silently lost. Anything
        # set earlier is replayed by _create_backend_plans().
        if self._backend_has_plans():
            getattr(self._lowered_graph, f"{'select' if keep else 'deselect'}_{'behavior' if kind == 'behavior' else 'numeric'}_notes")(list(notes))
        if self._is_built and self._planning_done and self._plan_index in self._barred_indices():
            self._is_built = False  # the built plan is no longer allowed to run
        return self

    def _backend_has_plans(self) -> bool:
        """Whether the backend has anything for a note filter to mark. Its
        filters index engine_configs, so one forwarded earlier marks nothing."""
        return self._lowered_graph is not None and (self._cpp_plans_created or bool(self._lowered_graph.get_execution_plan_count()))

    def _forward_note_filters(self) -> None:
        """Push every recorded note filter at the backend. Idempotent — the C++
        side ORs into barred_indices — so replaying is safe."""
        for kind, note, keep in self._note_filters:
            getattr(self._lowered_graph, f"{'select' if keep else 'deselect'}_{'behavior' if kind == 'behavior' else 'numeric'}_notes")([note])

    def _notes_of(self, cfg: Any, kind: str) -> tuple:
        """The declared notes of a python plan; () for a backend entry, whose
        notes the backend owns and filters itself."""
        eng = self._engine_for(cfg)
        return tuple(getattr(eng, f"{kind}_notes", ()) or ()) if eng is not None else ()

    def _resolve_stream(self, handle: Any) -> Any:
        """Stream for a supplied handle (classic set_stream semantics). A failed
        query on a SUPPLIED handle is a correctness error and raises — never a
        silent fall-back to another stream. No handle -> None (kernel wrappers
        fall back to the default stream)."""
        if handle is None:
            return None
        return cudnn.get_stream(handle)

    def _build_context(self, handle: Any = None) -> Any:
        h = handle if handle is not None else self._handle
        return ExecutionContext(handle=h, stream=self._resolve_stream(h))

    def _backend_handle_for_lowering(self, cpp_kwargs: Dict[str, Any]) -> Optional[int]:
        """The raw handle the C++ graph is constructed with: the caller's, else the
        process default for this thread and device. None only on the deviceless
        (``device_property``) AOT path, which lowers without any handle. The graph
        never owns a handle (Rule 8); ``self._handle`` stays as the caller set it, so
        execute-time stream resolution is unchanged."""
        if self._handle is not None:
            return to_backend_handle(self._handle)
        if cpp_kwargs.get("device_property") is not None:
            return None
        return _default_backend_handle()

    def _reset_lowered_state(self) -> None:
        """Drop every artifact of a C++ lowering (graph, tensors, BOG/plan flags,
        backend entries) so a later lowering starts from the IR, not from a
        half-built or rejected descriptor."""
        self._lowered_graph = None
        self._cpp_tensors.clear()
        self._cpp_bog_done = False
        self._cpp_plans_created = False
        self._backend_mode_spans.clear()
        self._backend_entries = []

    def _verify_uid_ownership(self) -> None:
        # Verify the uid-ownership invariant (see _lower_to_cpp): every C++
        # tensor must carry exactly its IR uid. An assertion — not a silent
        # translation — so a lowering path that forgets to push a uid fails
        # loudly in tests instead of mis-binding buffers (a swapped
        # multi-output pairing writes past the smaller buffer: corruption).
        for ir_uid, cpp_t in self._cpp_tensors.items():
            cpp_uid = cpp_t.get_uid()
            if cpp_uid != ir_uid:
                raise AssertionError(f"uid ownership violated: IR tensor uid {ir_uid} lowered to C++ uid {cpp_uid} — a lowering path failed to push the uid")

    def _lower_backend_graph(self) -> None:
        """Lower to C++ and build its operation graph (once).

        Kept separate from plan creation because the two failures mean different
        things: a failure HERE is "the backend cannot represent this graph"
        (a node it has no lowering for, a dtype/layout its descriptors reject),
        which is a decline; a failure in plan creation is about engines."""
        if self._lowered_graph is None:
            self._lowered_graph = self._lower_to_cpp()
            self._lowered_graph.validate()
            self._verify_uid_ownership()
            # Replay the plan filters: with a python engine registered nothing is
            # lowered at build_operation_graph() time, so a deselect_* call made
            # at that stage reached only pygraph.
            if self._barred_names:
                self._lowered_graph.deselect_engines(list(self._barred_names))
            if self._workspace_limit is not None:
                self._lowered_graph.deselect_workspace_greater_than(self._workspace_limit)

        if not self._cpp_bog_done:
            self._lowered_graph.build_operation_graph()
            self._cpp_bog_done = True
            self._sync_ir_shapes_from_backend()

    def _create_backend_plans(self) -> None:
        """Run the backend's heuristics for the lowered graph (once), ONE MODE AT A TIME.

        A call per mode rather than one call listing them all. C++ appends each
        query to the same plan list, so asking separately and reading
        get_execution_plan_count() after each is what says which entries came
        from which mode — and ranking needs that, since "the backend's mode-A
        entries ahead of ours, its fallbacks behind" is not expressible against
        one opaque list.

        A mode with no configs raises; that is not a decline, since another mode
        may still have entries (an OPENSOURCE-only query legitimately leaves the
        cuDNN modes empty). Only every mode failing means the backend has
        nothing, and the last error is re-raised so the caller reports why.

        "Succeeded" is tracked per call, not read off the spans: an OPENSOURCE
        query registers a C++ OSS candidate without adding a plan, so it leaves
        no span, and judging by spans would discard the delegate it earned.
        """
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        if self._cpp_plans_created:
            return
        from .engines.heuristics import default_modes

        # The SAME default the ranking assumes, or the backend gets queried for
        # modes no family will place.
        modes = self._backend_heuristics or default_modes()
        at, failure, any_ok = self._lowered_graph.get_execution_plan_count(), None, False
        for mode in modes:
            try:
                self._lowered_graph.create_execution_plans([mode])
            except cudnn.cudnnGraphNotSupportedError as exc:
                failure = exc
                continue
            any_ok = True
            now = self._lowered_graph.get_execution_plan_count()
            if now > at:
                self._backend_mode_spans.append((mode, at, now))
                at = now
        if not any_ok and failure is not None:
            raise failure
        self._cpp_plans_created = True
        self._forward_note_filters()  # deferred from _filter_notes(): the plans exist now

    def _mode_of_backend_plan(self, cpp_index: int):
        """The heuristic mode whose query produced the backend plan at ``cpp_index``."""
        for mode, lo, hi in self._backend_mode_spans:
            if lo <= cpp_index < hi:
                return mode
        return None

    def _refuse_guarded_backend(self) -> None:
        """Raise the family's backend guard as a typed decline before a backend PLAN is created or queried.

        The planning sequence records the guard as the backend's decline and never asks
        (``_finalize_backend_layout``, ``backend_plan_entries``).  Two paths create backend plans outside
        that sequence and refuse here instead: an explicit ``create_execution_plan(<backend engine id>,
        knobs)`` (``_append_backend_plan`` -- one C++ engine config, measured to crash exactly as the
        heuristics query does on the guarded domain) and a late backend lowering for a classic query
        (``_lower_backend_plan`` -- the heuristics query itself).  So an explicit diagnostic pin of a
        backend plan on a guarded graph is a decline that names the reason, never a dead process; the
        reason is recorded as the backend's decline as the planning sequence would have."""
        reason = self._backend_guard_reason()
        if reason is not None:
            self._backend_declined = cudnn_graph_not_supported(reason)
            raise cudnn_graph_not_supported(reason)

    def _lower_backend_plan(self) -> None:
        """Lower to C++ (if not already) and create the backend plans (once)."""
        self._refuse_guarded_backend()
        self._lower_backend_graph()
        self._create_backend_plans()

    def _check_plan_index(self, index: int) -> int:
        """Reject an out-of-range index instead of letting python's negative
        indexing quietly run the last plan."""
        if not isinstance(index, int) or not 0 <= index < len(self._plans):
            raise IndexError(f"plan index {index} out of range for {len(self._plans)} plan(s)")
        return index

    def _materialize_backend_plan(self, index: int):
        """Ensure the backend entry at ``index`` has a real C++ plan index.

        A replayed entry (``cpp_index=None``, from custom heuristics or an
        autotune result) does not exist in the backend's list until it is
        appended. Every entry point that needs a concrete plan — check_support,
        the build walk, workspace, behaviour notes, execute — goes through here,
        so none of them can be reached with an entry the backend has never seen.
        Returns the (possibly updated) PlanConfig."""
        from .engines.base import PlanConfig

        from .engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

        cfg = self._plans[index]
        if cfg.cpp_index is not None or cfg.engine_id == BACKEND_HEURISTIC_ENGINE_ID:
            return cfg  # already placed, or the delegating entry C++ owns
        cfg = PlanConfig(cfg.engine_id, cfg.knobs, cpp_index=self._append_backend_plan(cfg.engine_id, cfg.knobs))
        self._plans[index] = cfg
        return cfg

    def _append_backend_plan(self, engine_id: int, knobs: Any) -> int:
        """Append one explicit ``(engine_id, knobs)`` plan to the BACKEND's list
        and return the C++ index it landed at.

        ``Graph::create_execution_plan`` appends (``enqueue_engine_configs`` ->
        ``back_inserter``), and ``build_plans(HEURISTICS_CHOICE)`` short-circuits
        once a candidate exists (plans.h), so the appended plan is neither built
        nor selected by the plain calls. Everything downstream therefore has to
        address it BY INDEX — which is why this returns one."""
        self._refuse_guarded_backend()  # a backend record replayed onto a guarded graph: a decline, not a crash
        self._lower_backend_graph()  # heuristics would only add plans nobody asked for
        at = self._lowered_graph.get_execution_plan_count()
        self._lowered_graph.create_execution_plan(engine_id, knobs or {})
        self._forward_note_filters()  # an explicitly appended config is filterable too
        return at

    def check_support(self) -> None:
        """Check that the graph can be served.

        A python plan re-affirms its engine's check_support() (already passed
        when it proposed the plan); a backend plan asks C++.

        A decline from the SELECTED entry is only fatal when no other entry
        could serve the graph. The backend's ``check_support()`` is aggregate —
        it answers for the backend as a whole, not for one plan — so letting it
        raise here would abort ``build()`` before the walk ever reached a python
        entry that can run. Tolerating it and letting ``build_plans()`` walk is
        what makes "backend first, python next" actually reachable; if nothing
        builds, the walk raises with every failure listed.
        """
        from .engines.base import decline_types

        eng = self.selected_engine
        if eng is not None:
            eng.check_support(self)
            return
        if self._plans and self._plans[self._plan_index].cpp_index is None:
            self._materialize_backend_plan(self._plan_index)  # else C++ is asked about a plan it has never seen
        elif self._lowered_graph is None:
            self._lower_backend_plan()
        try:
            self._lowered_graph.check_support()
        except decline_types() as exc:
            others = [i for i, cfg in enumerate(self._plans) if i != self._plan_index and self._engine_for(cfg) is not None]
            if self._plan_pinned or not others:
                raise
            self._backend_declined = _detached_exception(exc)
            _LOG.info("backend check_support declined the graph (%s); the plan walk still has %d python entr(y|ies)", exc, len(others))

    def build_plans(self, *args, ctx: Any = None, **kwargs) -> None:
        """Walk the ranked plan list from the selected index and finalize the
        first entry that builds.

        One rule for both sides: look the entry's ``engine_id`` up, try to build
        it, and on a decline move to the next entry. A python plan JIT-compiles
        here (cached on the graph, reused across executions); a backend plan is
        finalized by the backend. Only a decline
        (``NotImplementedError`` / ``cudnnGraphNotSupportedError``) advances the
        walk — any other exception is a bug in that engine and propagates.

        An explicit ``select_plan(i)`` is strict: a decline there or that
        plan being barred raises instead of silently selecting another plan.
        ALL visits the whole list and retains the selected plan when it builds;
        declines from other entries do not invalidate the pin.
        """
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        from .engines.base import decline_types

        if not self._planning_done:
            self.create_execution_plans()
        # ALL means "build every entry" (autotune times each index afterwards);
        # the default stops at the first that builds.
        build_all = any(a == cudnn.build_plan_policy.ALL for a in args) or kwargs.get("policy") == cudnn.build_plan_policy.ALL
        strict = self._plan_pinned
        barred = self._barred_indices()  # once: resolving names can lower the backend
        failures = []
        previous_index = self._plan_index
        selected_index = None
        self._is_built = False  # requalify the selection against the current filters
        for index in range(0 if build_all else previous_index, len(self._plans)):
            if index in barred:
                if strict and index == previous_index:  # select_plan and deselect_engines contradict each other
                    raise ValueError(
                        f"plan {index} ({self.get_plan_name_at_index(index)!r}) is pinned by select_plan() but "
                        f"excluded by deselect_engines(); drop one of the two instructions"
                    )
                continue
            try:
                self._build_plan_at(index, *args, ctx=ctx, **kwargs)
                if self._workspace_limit is not None and self._engine_for(self._plans[index]) is not None:
                    need = self._compiled_plans[index].get_workspace_size()
                    if need > self._workspace_limit:
                        raise cudnn_graph_not_supported(f"needs {need} workspace bytes, over the {self._workspace_limit} limit")
            except decline_types() as exc:
                if strict and index == previous_index:
                    raise
                failures.append(f"[{index}] {self.get_plan_name_at_index(index)}: {exc}")
                _LOG.info("plan %d declined at build time (%s); trying the next entry", index, exc)
                continue
            if not build_all:
                self._plan_index = index
                self._is_built = True
                return
            if selected_index is None or index == previous_index:
                selected_index = index
        if selected_index is not None:
            if self._engine_for(self._plans[selected_index]) is None:
                # Each native build changes C++'s candidate. Re-select the
                # retained, already-built plan so delegated queries, serialization
                # and CUDA-graph APIs agree with indexed execution.
                self._build_plan_at(selected_index, *args, ctx=ctx, **kwargs)
            self._plan_index, self._is_built = selected_index, True
            return
        if self._backend_declined is not None and not failures:
            raise _detached_exception(self._backend_declined)  # nothing else ran: the backend's failure IS the answer
        # The recorded backend decline too: a list with no backend entry (the backend declined, or a family guard kept
        # it out) whose python plans all failed to build would otherwise read as if the backend had never been asked.
        why = f"\n  the backend declined: {self._backend_declined}" if self._backend_declined is not None else ""
        raise cudnn_graph_not_supported("no plan in the list could be built:\n  " + "\n  ".join(failures or ["the plan list is empty"]) + why)

    def _build_plan_at(self, index: int, *args, ctx: Any = None, **kwargs) -> None:
        """Build one entry — the single place the two sides diverge.

        ``ctx`` is the caller's ExecutionContext when the build was triggered
        from ``execute(..., handle=...)``: a JIT engine compiles for the device
        and stream it will actually run on, so the handle must reach it here and
        not be replaced by the graph's own."""
        cfg = self._plans[index]
        eng = self._engine_for(cfg)
        if eng is not None:
            if index not in self._compiled_plans:
                from .frost import compiled_cache

                with compiled_cache.record_entries() as entries:
                    self._compiled_plans[index] = eng.build_plan(self, cfg, ctx or self._build_context())
                self._plan_kernel_entries[index] = entries
            self._prepare_ordered_binding_schema()
            return
        cfg = self._materialize_backend_plan(index)
        if cfg.cpp_index is None:  # delegating entry: the backend picks
            self._lower_backend_plan()
            self._lowered_graph.build_plans(*args, **kwargs)
        else:
            self._lower_backend_plan()
            self._lowered_graph.build_plan_at_index(cfg.cpp_index)
        self._prepare_ordered_binding_schema()

    def build(self, heuristics: Optional[List] = None, ctx: Any = None) -> None:
        """Convenience: validate -> build_operation_graph -> create_execution_plans
        -> check_support -> build_plans, in sequence.

        ``ctx`` is forwarded to whichever plan the walk builds, so an auto-build
        triggered from ``execute(..., handle=...)`` compiles for the caller's
        device and stream."""
        if not self._is_validated:
            self.validate()

        self.build_operation_graph()
        if not self._planning_done:  # never silently re-plan: preserves select_plan()
            self.create_execution_plans(heuristics)
        self.check_support()
        self.build_plans(ctx=ctx)

    def get_workspace_size(self, handle=None, override_uids=None, override_shapes=None, override_strides=None) -> int:
        """Workspace bytes for the selected plan. The classic overload args -- a
        handle and the dynamic-shape overrides -- pass through on the backend path."""
        if not self._is_built:
            raise RuntimeError("Call build() first")

        if self.selected_engine is not None:
            # A compiled python plan's workspace is normally a property of the
            # plan. A shape-dependent one (frost split-K partials) says so by
            # exposing get_workspace_size_for_shapes, which follows the overrides.
            plan = self._compiled_plans[self._plan_index]
            if override_shapes is not None:
                sized = getattr(plan, "get_workspace_size_for_shapes", None)
                if sized is not None:
                    return sized(override_uids, override_shapes)
            return plan.get_workspace_size()

        # Same reason execute() addresses by index; the overload args pass through.
        cfg = self._materialize_backend_plan(self._plan_index) if self._plans else None
        no_overload = handle is None and override_uids is None and override_shapes is None and override_strides is None
        if no_overload and cfg is not None and cfg.cpp_index is not None:
            return self._lowered_graph.get_workspace_size_plan_at_index(cfg.cpp_index)
        return self._lowered_graph.get_workspace_size(to_backend_handle(handle), override_uids, override_shapes, override_strides)

    def get_workspace_size_plan_at_index(self, index: int, handle=None, override_uids=None, override_shapes=None, override_strides=None) -> int:
        """Workspace bytes for the plan at ``index`` in the ranked list."""
        if not self._planning_done:  # e.g. a deserialized graph: C++ owns the list
            return self._lowered_graph.get_workspace_size_plan_at_index(index, to_backend_handle(handle), override_uids, override_shapes, override_strides)
        self._reject_if_barred(self._check_plan_index(index))
        cfg = self._plans[index]
        if self._engine_for(cfg) is not None:
            # Shape-dependent plans consult the overrides — see get_workspace_size().
            if index not in self._compiled_plans:
                self._build_plan_at(index)
            plan = self._compiled_plans[index]
            if override_shapes is not None:
                sized = getattr(plan, "get_workspace_size_for_shapes", None)
                if sized is not None:
                    return sized(override_uids, override_shapes)
            return plan.get_workspace_size()
        cfg = self._materialize_backend_plan(index)
        if cfg.cpp_index is None:  # delegating entry
            return self._lowered_graph.get_workspace_size(to_backend_handle(handle), override_uids, override_shapes, override_strides)
        return self._lowered_graph.get_workspace_size_plan_at_index(cfg.cpp_index, to_backend_handle(handle), override_uids, override_shapes, override_strides)

    def _reject_if_barred(self, index: int) -> None:
        """The ranked list is never filtered (indices stay stable), so every
        path that reaches a plan BY INDEX has to apply the exclusions itself."""
        if index in self._barred_indices():
            raise ValueError(f"plan {index} ({self.get_plan_name_at_index(index)!r}) is excluded by deselect_engines()/a note filter")

    def build_plan_at_index(self, index: int) -> "pygraph":
        """Build the plan at ``index`` and select it.

        Selecting is what makes the autotune idiom work: build index i, then
        get_workspace_size()/execute() refer to i rather than to whatever the
        walk last settled on."""
        if not self._planning_done:
            return self._lowered_graph.build_plan_at_index(index)
        self._check_plan_index(index)
        self._reject_if_barred(index)
        self._build_plan_at(index)
        if self._workspace_limit is not None and self._engine_for(self._plans[index]) is not None:
            need = self._compiled_plans[index].get_workspace_size()
            if need > self._workspace_limit:
                raise cudnn_graph_not_supported(f"plan {index} needs {need} workspace bytes, over the {self._workspace_limit} limit")
        self._plan_index, self._is_built = index, True
        return self

    def get_engine_and_knobs_at_index(self, index: int):
        """The ``(engine_id, knobs)`` of the plan at ``index`` in the ranked list.

        Answered from the unified list, NOT forwarded to C++ with a unified
        index — that would report the backend's entry for a python plan's slot.
        The pair is what ``create_execution_plan()`` replays, so it must name the
        same engine the caller just looked at.

        ``knobs`` is always a ``{cudnn.knob_type: int}`` dict (empty when the
        plan has no tuning axes, never ``None``), for backend and python plans
        alike: one record shape for autotuners to persist."""
        from .engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

        if not self._planning_done:
            return self._lowered_graph.get_engine_and_knobs_at_index(index)
        cfg = self._plans[self._check_plan_index(index)]
        if cfg.engine_id == BACKEND_HEURISTIC_ENGINE_ID:
            raise NotImplementedError(
                f"plan {index} delegates to the backend's own choice among candidates it does not expose "
                f"as plans (heur_mode.OPENSOURCE); there is no (engine_id, knobs) pair to replay"
            )
        eng = self._engine_for(cfg)
        if eng is not None:
            return (cfg.engine_id, eng.knobs_to_public(cfg.knobs))
        return (cfg.engine_id, dict(cfg.knobs) if cfg.knobs else {})

    def get_behavior_notes_for_plan_at_index(self, index: int, *args, **kwargs):
        """Classic backend behaviour notes for the plan at ``index``.

        Backend-only by construction: a python plan has no ``BehaviorNote_t``.
        Saying so is better than forwarding the index and returning some other
        plan's notes."""
        if not self._planning_done:
            return self._lowered_graph.get_behavior_notes_for_plan_at_index(index, *args, **kwargs)
        cfg = self._plans[self._check_plan_index(index)]
        if self._engine_for(cfg) is not None:
            return list(self._notes_of(cfg, "behavior"))
        cfg = self._materialize_backend_plan(index)
        if cfg.cpp_index is None:
            raise NotImplementedError(f"plan {index} delegates to the backend's own choice and has no position in its plan list")
        return self._lowered_graph.get_behavior_notes_for_plan_at_index(cfg.cpp_index, *args, **kwargs)

    def get_behavior_notes(self, *args, **kwargs):
        """Classic behaviour notes for the SELECTED plan — backend-only, for the
        same reason as the at-index query."""
        eng = self.selected_engine
        if eng is not None:
            return list(self._notes_of(self._selected_plan_config, "behavior"))
        if not self._planning_done:  # e.g. a deserialized graph: C++ owns the built plan
            return self._lowered_graph.get_behavior_notes(*args, **kwargs)
        self._lower_backend_plan()
        return self._lowered_graph.get_behavior_notes(*args, **kwargs)

    def populate_cuda_graph(self, handle, variant_pack, workspace, cuda_graph):
        """Classic CUDA-graph capture — the backend's, so a python plan declines.

        Capture itself is not the obstacle (the python engines run on the
        execute-time handle's stream); this API records the BACKEND's plan."""
        return self._cuda_graph_call("populate_cuda_graph", to_backend_handle(handle), variant_pack, workspace, cuda_graph)

    def update_cuda_graph(self, handle, variant_pack, workspace, cuda_graph):
        """Classic CUDA-graph update — backend plans only, as above."""
        return self._cuda_graph_call("update_cuda_graph", to_backend_handle(handle), variant_pack, workspace, cuda_graph)

    def _cuda_graph_call(self, name: str, *args, **kwargs):
        eng = self.selected_engine
        if eng is not None:
            raise cudnn_graph_not_supported(f"{name}() records a cuDNN backend plan; the selected plan is served by the python engine {eng.name!r}")
        if self._planning_done:
            self._lower_backend_plan()
        return getattr(self._lowered_graph, name)(*args, **kwargs)

    def key(self, *args, **kwargs):
        """Classic cache key. The backend computes it, so the graph must lower —
        and it does NOT yet distinguish which python engine serves the graph."""
        if self._lowered_graph is None:  # a deserialized graph already has one, and no op tree to rebuild
            node = self._unlowerable_node()
            if node is not None:
                raise cudnn_graph_not_supported(f"key() is the cuDNN backend's cache key; {node.node_type.name} has no backend lowering")
            if not self._is_validated:
                self.validate()  # lowering freezes the IR; validate() is what infers into it first
            self._lower_backend_graph()
        return self._lowered_graph.key(*args, **kwargs)

    def create_execution_plan(self, engine_id: int, knobs: Any = None) -> "pygraph":
        """APPEND one plan for an explicit ``(engine_id, knobs)`` to the ranked list.

        This is the deterministic-replay entry point: an autotune result or a
        perf sweep records ``(engine_id, knobs)`` and rebuilds exactly that plan
        later. It appends to ``graph.plans``, so the classic idiom
        ``create_execution_plan(...)`` then ``get_execution_plan_count() - 1``
        addresses the plan just added — for a python engine id as well as a
        backend one, which is the whole point of one id space.

        ``knobs`` is the public ``{cudnn.knob_type: int}`` dict that
        ``get_engine_and_knobs_at_index`` reported (``None`` / ``{}`` for a plan
        without tuning axes). A python engine's native knob object is accepted
        too, for callers that already hold one.
        """
        from .engines.base import PlanConfig
        from .engines.engine_ids import is_python_engine

        if not self._is_validated:
            self.validate()
        if is_python_engine(engine_id):
            owners = self._owners_for_id(engine_id)
            if not owners:
                raise ValueError(f"no python engine on this graph owns engine_id {engine_id}")
            if len(owners) > 1:
                raise ValueError(f"engine_id {engine_id} is owned by {[e.name for e in owners]} — ambiguous dispatch")
            if knobs is None or isinstance(knobs, dict):
                knobs = owners[0].knobs_from_public(knobs or {})
            entry = PlanConfig(engine_id, knobs)
        else:
            entry = PlanConfig(engine_id, knobs, cpp_index=self._append_backend_plan(engine_id, knobs))
        self._plans = list(self._plans) + [entry]
        self._planning_done = True
        # Same contract as create_execution_plans(): a plan refers to the graph
        # as it is now, so the graph stops accepting nodes. Without this a
        # replay on a fresh graph left it mutable while _is_built said the plan
        # was ready, and a node added afterwards silently diverged from it.
        self._freeze()
        return self

    def execute_plan_at_index(self, tensor_dict, workspace=None, index: int = 0, handle=None, *args, tensor_uids=None, **kwargs) -> None:
        """Execute one ranked plan without changing the graph's selected plan."""
        if not self._planning_done:  # e.g. a deserialized graph: C++ owns the list
            if tensor_uids is not None:
                pack = self._normalize_ordered(tensor_dict, tensor_uids, workspace, *args, **kwargs)
                return self._lowered_graph._execute_ordered_pack(pack.native, pack.workspace, to_backend_handle(handle) or 0, index)
            uid_to_data = self._uid_to_data(tensor_dict)
            var_pack, ws_ptr = self._native_var_pack(uid_to_data, workspace)
            return self._lowered_graph._execute_plan_at_index(var_pack, ws_ptr, index, to_backend_handle(handle), *args, **kwargs)
        self._reject_if_barred(self._check_plan_index(index))
        return self._execute(tensor_dict, workspace, handle, *args, tensor_uids=tensor_uids, index=index, **kwargs)

    def execute(
        self,
        tensor_dict: Union[Dict[Union[str, int, Tensor], Any], List[Any], Tuple[Any, ...]],
        workspace: Any = None,
        handle: Optional[Handle] = None,
        override_uids: Any = None,
        override_shapes: Any = None,
        override_strides: Any = None,
        tensor_uids: Any = None,
    ) -> None:
        """Execute the selected plan.

        Both python engines and the backend path write results directly into the
        caller-provided output tensors (in-place). Automatically calls build()
        if it hasn't run yet. Dispatch is a single check on the plan's engine id.

        Args:
            tensor_dict: Dict mapping tensors (by Tensor, name, or uid) to data.
                         Must include both input and output tensors.
                         With ``tensor_uids``, a tuple/list of buffers instead.
                         No additional preparation call is required.
            workspace: Workspace buffer (python engines receive it via the
                       ExecutionContext; plans that need one require it)
            handle: cuDNN handle; kernels launch on its stream (classic
                    ``set_stream`` semantics, both python engines and backend)
            override_uids/shapes/strides: runtime geometry for the backend or a
                         compatible VariantPack plan; legacy uid-map plans raise
                         rather than ignoring these arguments
            tensor_uids: optional tuple/list of distinct operand uids, one per
                         buffer in ``tensor_dict``; order need not be sorted.
                         Unused bindings are ignored, as in the mapping form;
                         duplicate uids are rejected. Explicit buffers override
                         auto-bound inputs. Metadata
                         is cached internally by value, while buffer observations,
                         workspace and the handle's stream remain per call.
        """
        return self._execute(tensor_dict, workspace, handle, override_uids, override_shapes, override_strides, tensor_uids)

    def _execute(
        self,
        tensor_dict,
        workspace=None,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
        tensor_uids=None,
        *,
        index=None,
    ):
        # Keep the caller's choice local across binding and launch. Temporarily
        # selecting/restoring an index lets overlapping callers run each other's
        # plan and can corrupt the selection left for ordinary execute().
        explicit_index = index is not None
        if not explicit_index and not self._is_built:
            # A JIT engine must compile for the device/stream it will run on, so
            # build the caller context here. Only here: a steady-state execute()
            # otherwise discarded this (a cudnnGetStream round-trip + an
            # ExecutionContext alloc) on every already-built call.
            caller_ctx = self._build_context(handle) if handle is not None else None
            if not self._planning_done:
                self.create_execution_plans()
            self.build(ctx=caller_ctx)

        if not explicit_index:
            index = self._plan_index

        ordered = tensor_uids is not None
        uid_to_data = None if ordered else self._uid_to_data(tensor_dict)
        # Backend overrides use its uid-map overload. Python plans must consume
        # them through VariantPack; a legacy uid-map executor cannot honor them.
        overriding = override_uids is not None or override_shapes is not None or override_strides is not None
        eng = self._engine_for(self._plans[index]) if explicit_index else self.selected_engine

        if eng is not None:  # python engine (plan id in the reserved region)
            h = handle if handle is not None else self._handle
            stream = self._resolve_stream(h)
            # A JIT engine launches through the driver, which reads the calling
            # thread's context stack; an autograd worker has none. The handle's
            # device decides when the stream names no context.
            ensure_current_context(stream, h.device.ordinal if h is not None else None)
            if index not in self._compiled_plans:
                # compile with the CALLER's context (execute-supplied handle
                # and its stream reach the JIT build)
                ctx = ExecutionContext(handle=h, stream=stream, workspace=workspace)
                self._compiled_plans[index] = eng.build_plan(self, self._plans[index], ctx)
                if not explicit_index:
                    self._is_built = True
            plan = self._compiled_plans[index]
            native_ordered = getattr(plan, "_ordered_native", None) if ordered else None
            if native_ordered is not None:
                schema = self._prepare_ordered_binding_schema()
                if schema is None:
                    raise ValueError("The graph has no operand layout for ordered execution")
                result = plan._execute_ordered(self, schema, tensor_dict, tensor_uids, workspace, override_uids, override_shapes, override_strides, stream)
                if result is None:
                    return
                # A producer without the exchange protocol is observed once by
                # the existing Python completion path, then uses the same plan.
                pack = self._finish_ordered(schema, result, workspace, override_uids)
                plan.execute(self, pack, ExecutionContext(handle=h, stream=stream, workspace=workspace))
                return
            ctx = ExecutionContext(handle=h, stream=stream, workspace=workspace)
            # Overrides go INTO the pack rather than around it: they describe
            # what this execute runs, so an engine reading the pack agrees with
            # the backend without knowing they exist.
            if plan.takes_variant_pack:
                pack = (
                    self._normalize_ordered(tensor_dict, tensor_uids, workspace, override_uids, override_shapes, override_strides)
                    if ordered
                    else self._normalize(uid_to_data, workspace, override_uids, override_shapes, override_strides)
                )
                plan.execute(self, pack, ctx)
            else:
                if overriding:
                    raise ValueError(f"{eng.name}: this plan does not support execute-time shape or stride overrides")
                if ordered:
                    # Legacy uid-map plans retain the same caller contract.
                    # Validate the ordered form before constructing their map.
                    self._normalize_ordered(tensor_dict, tensor_uids, workspace, None, None, None)
                    uid_to_data = dict(self._data_bindings)
                    uid_to_data.update(zip(tensor_uids, tensor_dict))
                plan.execute(self, uid_to_data, ctx)
            return

        variant_pack = None if ordered or overriding else self._normalize(uid_to_data, workspace)

        # Backend path. Address the plan the WALK built, not the backend's own
        # selection: they differ once the walk has skipped an entry.
        cfg = self._materialize_backend_plan(index) if self._plans else None
        cpp_index = cfg.cpp_index if cfg is not None else None

        if ordered:
            schema = self._prepare_ordered_binding_schema()
            if schema is None:
                raise ValueError("The graph has no operand layout for ordered execution")
            result = self._lowered_graph._execute_ordered(
                schema,
                tensor_dict,
                tensor_uids,
                self._data_bindings,
                workspace,
                override_uids,
                override_shapes,
                override_strides,
                to_backend_handle(handle) or 0,
                -1 if cpp_index is None else cpp_index,
            )
            if result is None:
                return
            variant_pack = self._finish_ordered(schema, result, workspace, override_uids)
            self._lowered_graph._execute_ordered_pack(
                variant_pack.native, variant_pack.workspace, to_backend_handle(handle) or 0, -1 if cpp_index is None else cpp_index
            )
            return

        # C++ turns a uid map into sorted pointers anyway (graph_interface.h,
        # "uid map -> extract sorted ptrs, delegate to the sorted_ptrs
        # implementation"), so handing it the sorted array directly skips one
        # dict build here, one map copy in pybind, and one hash lookup per
        # operand there.
        if variant_pack is not None:
            self._lowered_graph._execute_with_raw_ptrs(
                variant_pack.address,
                len(variant_pack),
                variant_pack.workspace,
                to_backend_handle(handle) or 0,
                -1 if cpp_index is None else cpp_index,
            )
            return

        var_pack, ws_ptr = self._native_var_pack(uid_to_data, workspace)
        if cpp_index is not None:
            self._lowered_graph._execute_plan_at_index(var_pack, ws_ptr, cpp_index, to_backend_handle(handle), override_uids, override_shapes, override_strides)
            return
        self._lowered_graph._execute(var_pack, ws_ptr, to_backend_handle(handle), override_uids, override_shapes, override_strides)

    def _variant_pack_uids(self) -> Optional[List[int]]:
        """The graph's caller-filled variant_pack, ASCENDING by uid.

        Taken from the lowered graph whenever there is one: C++ is the only
        side that can see every user slot, including the ones a walk over node
        ports cannot name (a tensor's ragged_offset hangs off the Tensor, not
        off a port) and correctly excluding the slots the graph fills itself
        (pass-by-value scalars it already knows, slice replacement
        destinations, cached workspace modifications).

        A python-only graph — gdn / kda / gdn2, which cannot lower by
        construction — has no C++ side, so its variant_pack come from the IR. The
        two never have to agree: each side indexes the layout it was given.
        """
        order = self._sorted_uids
        if order is not None:
            return order
        lowered = self._lowered_graph
        if lowered is not None:
            try:
                # The order lives in the variant-pack template, which C++ builds
                # lazily inside execute; the query itself does not trigger it, so
                # ask explicitly or it answers with an empty list.
                lowered._prepare_variant_pack_template()
                order = list(lowered._get_variant_pack_uids_sorted())
            except Exception:  # noqa: BLE001 — no template available yet
                order = []
        else:
            # Every tensor wired to a port, virtual or not. is_virtual is a
            # statement about the BACKEND's lowering — an intermediate it fuses
            # away — and a python-only op never lowers, so it does not mean
            # "the caller supplies nothing": a gdn graph marks its own O virtual
            # and the caller passes a buffer for it regardless. A slot nobody
            # fills stays empty; which ports are optional is the engine's own
            # business, and it already reads them with .get().  A port's ragged
            # offset hangs off its tensor, not a port; it is a caller slot too
            # (engines that address padded THD layouts read it on device).
            ports = [t for node in self._nodes for t in list(node.inputs.values()) + list(node.outputs.values()) if t is not None]
            order = sorted({t.uid for t in ports} | {t.ragged_offset.uid for t in ports if getattr(t, "ragged_offset", None) is not None})
        if not order:
            return None
        self._sorted_uids = order
        return order

    def _prepare_ordered_binding_schema(self):
        if self._ordered_binding_schema is None:
            order = self._variant_pack_uids()
            if order is not None:
                self._ordered_binding_uids = tuple(order)
                self._ordered_binding_schema = _pybind_module._OrderedBindingSchema(order, self._declared_layout(order), self._lowered_graph is not None)
        return self._ordered_binding_schema

    def _observe_unread(self, native, entries, order):
        """The existing observation fallback, shared by map and ordered calls."""
        from_graph = []
        for i, data in entries:
            if data is None:
                continue
            if type(data) is int:
                from_graph.append(i)
            ptr, tensor = self._describe(data, order[i])
            span = _observed_span(data)
            if span is not None:
                span *= _producer_itemsize(data, tensor.data_type)
            dev = getattr(data, "device", None)
            dev_type, dev_id = (-1, -1)
            if dev is not None and getattr(dev, "type", None) is not None:
                dev_type, dev_id = (2, int(dev.index or 0)) if dev.type == "cuda" else (1, 0)
            native.set_operand(
                i,
                ptr,
                tuple(tensor.dim),
                tuple(tensor.stride),
                *_dlpack_code_bits(tensor.data_type),
                _dlpack_lanes(tensor.data_type),
                -1 if span is None else span,
                dev_type,
                dev_id,
            )
        return from_graph

    def _workspace_extent_fallback(self, workspace):
        if type(workspace) is int:
            return workspace, None  # A raw address carries no observed capacity.
        workspace_ptr, workspace_tensor = self._describe(workspace, -1)
        if not _is_dense(workspace_tensor.dim, workspace_tensor.stride):
            raise ValueError(f"the workspace buffer must be contiguous; got dim {tuple(workspace_tensor.dim)} stride {tuple(workspace_tensor.stride)}")
        return workspace_ptr, _byte_size(workspace_tensor)

    def _normalize_ordered(self, buffers, tensor_uids, workspace, override_uids=None, override_shapes=None, override_strides=None):
        schema = self._prepare_ordered_binding_schema()
        if schema is None:
            raise ValueError("The graph has no operand layout for ordered execution")
        result = schema.read(buffers, tensor_uids, self._data_bindings, workspace, override_uids, override_shapes, override_strides)
        return self._finish_ordered(schema, result, workspace, override_uids)

    def _finish_ordered(self, schema, result, workspace, override_uids):
        native, unread, extent, described = result
        if unread:
            from_graph = self._observe_unread(native, unread, self._ordered_binding_uids)
            described = from_graph + schema.finish(native, from_graph)
        if extent is None:
            extent = self._workspace_extent_fallback(workspace)
        overridden = tuple(self._slot_of_uid[uid] for uid in override_uids) if override_uids else ()
        return VariantPack(self._ordered_binding_uids, native, extent[0], extent[1], tuple(described), overridden)

    def _normalize(self, uid_to_data: Dict[int, Any], workspace: Any, override_uids=None, override_shapes=None, override_strides=None):
        """Turn the caller's variant pack into :class:`VariantPack`, once.

        This is the ONLY place a caller's object is inspected. Everything below
        — the backend and every python engine — reads the pointers and Tensors
        built here, so the two paths cannot disagree about what the caller
        passed. Returns None when the operand layout is not known yet, which
        puts the caller back on the uid-map path.

        Overrides are applied here, to the slot, because they are part of the
        same answer: ``override_shapes`` says the caller allocated at a cache
        shape and is running a smaller one this call, so the pack must describe
        the shape about to run rather than the allocation. An engine that reads
        the pack then honours them without knowing the concept exists — which
        is the difference between one answer and two, since the backend
        re-describes the tensor from the overrides either way.
        """
        order = self._variant_pack_uids()
        if order is None:
            return None
        native = _pybind_module.VariantPackNative(len(order))
        # One crossing for the whole pack, uid lookups included. What comes back
        # is the slots whose producer publishes no exchange vtable; those are
        # described here without taking the rest down with them.
        unread = native.read_from(uid_to_data, order)
        # The backend's layout is exactly the slots it REQUIRES, so a hole is the
        # caller's mistake. A python-only graph's layout is every wired port,
        # including optional ones, where a hole means "not requested".
        strict = self._lowered_graph is not None
        # A bare address borrows the graph's axis order; keep that distinction
        # separate from the producer span/device, which stay unknown for it.
        from_graph = self._observe_unread(native, ((i, uid_to_data.get(order[i])) for i in unread), order) if unread else []
        if strict:
            hole = native.first_unfilled()
            if hole >= 0:
                uid = order[hole]
                declared = self._tensor_by_uid.get(uid)
                name = f" ({declared.name!r})" if declared is not None and declared.name else ""
                raise ValueError(f"the variant pack is missing a buffer for tensor uid {uid}{name}")
        # The declaration is the contract. The backend reads only the pointer,
        # so a caller may bind a 2-D matrix to a [1, m, k] tensor, a flat blob
        # to a reordered scale tensor, a 0-d scalar to (1, 1, 1): the graph
        # says what the bytes mean. A DENSE buffer of other extents that covers
        # the declared bytes is therefore re-described AS the declaration --
        # what a bare address gets -- so an engine reading the pack answers the
        # way the backend does. A buffer with the declared extents but its own
        # strides, a strided view, or one too small for the declaration keeps
        # its own description; the engine decides. Reordered scale blobs retain
        # their physical extents for capacity checks. The rule runs natively, one
        # crossing per pack: this is on every execute's critical path.
        from_graph.extend(native.describe_from(self._declared_layout(order), from_graph))
        if override_uids:
            # The backend refuses a partial override; a short list must not
            # quietly mean "keep the rest" here.
            if len(override_shapes or ()) != len(override_uids) or len(override_strides or ()) != len(override_uids):
                raise ValueError(
                    f"override_uids, override_shapes and override_strides must name the same tensors: got "
                    f"{len(override_uids)}, {len(override_shapes or ())} and {len(override_strides or ())} entries"
                )
            # One crossing for every override: the native side turns each element geometry into
            # storage-slot geometry (fp4 packing), re-expresses it in the buffer's own axis order and
            # applies it with the declared dtype (see VariantPackNative::override_many).
            layout = self._declared_layout(order)
            slot_of = self._slot_of_uid
            try:
                indices = [slot_of[uid] for uid in override_uids]
            except KeyError as exc:
                raise ValueError(f"override_uids names tensor uid {exc.args[0]}, which is not an operand of this graph") from None
            native.override_many(layout, indices, [list(s) for s in override_shapes], [list(s) if s else [] for s in override_strides])
        # The workspace has no uid, so it is not an operand — but an engine has
        # to bounds-check its carves, and reading its size here is the same read
        # every other buffer gets rather than a second probe further down.
        workspace_ptr, workspace_bytes = 0, 0
        if workspace is not None:
            extent = _pybind_module.read_buffer_extent(workspace)
            if extent is None:  # a bare address, a non-dense buffer, or no vtable
                # An engine carves the workspace by byte offset, so a byte
                # COUNT is only a byte RANGE when the buffer is dense.
                workspace_ptr, workspace_bytes = self._workspace_extent_fallback(workspace)
            else:
                workspace_ptr, workspace_bytes = extent
        return VariantPack(tuple(order), native, workspace_ptr, workspace_bytes, tuple(from_graph), tuple(indices) if override_uids else ())

    def _declared_layout(self, order: List[int]):
        """The storage-slot geometry each slot of ``order`` was declared with,
        built once per graph: the declaration is fixed once the graph is, and
        ``storage_geometry`` per operand per execute was a measurable share of
        the host path. Reset with ``_sorted_uids``."""
        layout = self._declared_layout_native
        if layout is None:
            layout = _pybind_module.DeclaredLayout(len(order))
            self._slot_of_uid = {uid: i for i, uid in enumerate(order)}
            for i, uid in enumerate(order):
                declared = self._tensor_by_uid.get(uid)
                if declared is None:
                    continue
                storage = storage_geometry(declared.dim, declared.stride, declared.data_type) if declared.dim else None
                if storage is None:
                    layout.set_dtype(i, *_dlpack_code_bits(declared.data_type), _dlpack_lanes(declared.data_type))  # overrides still speak its dtype
                    continue
                layout.set(
                    i,
                    list(storage[0]),
                    list(storage[1]),
                    storage_slot_bytes(declared.data_type) or 0,
                    *_dlpack_code_bits(declared.data_type),
                    _dlpack_lanes(declared.data_type),
                    declared.get_reordering_type() == _pybind_module.tensor_reordering.F8_128x4,
                )
            self._declared_layout_native = layout
        return layout

    def _describe(self, data: Any, uid: int):
        """``(pointer, Tensor)`` for one caller buffer.

        The Tensor carries the buffer's OWN dim/stride/data_type, which need
        not match what the graph declared. ``_normalize`` then re-describes a
        slot FROM the declaration when the two disagree and the buffer covers
        it (the declaration is the contract, as for the backend), so what an
        engine reads is the declaration unless the buffer is smaller.

        Every framework publishes the same four facts under a different
        spelling, so this asks for each spelling in turn. Two differences are
        the only ones that matter, and both are handled below: strides are in
        BYTES for the array-interface family and in ELEMENTS for torch/DLPack,
        and an absent stride means dense row-major rather than unknown.

        Ordered by what it costs to ask: attribute reads before the DLPack
        capsule round trip. The expensive case is torch bfloat16, whose
        ``__cuda_array_interface__`` raises because the protocol cannot spell
        bf16, which is why the last branch is last and not the only one.

        A bare address carries no geometry, so it borrows the graph's: the
        backend has always accepted a raw pointer, and an engine that reads the
        pack for its extents would otherwise fail on an operand form the
        backend takes -- one call, two answers, decided by plan selection.
        """
        if type(data) is int:  # bare device address
            declared = self._tensor_by_uid.get(uid)
            if declared is None or not declared.dim:
                return data, Tensor(uid=uid)
            # The slot speaks storage slots (fp4: two elements per slot), like
            # every other description in the pack.
            storage = storage_geometry(declared.dim, declared.stride, declared.data_type)
            dim, stride = storage if storage is not None else (tuple(declared.dim), tuple(declared.stride))
            return data, describing_tensor(uid, tuple(dim), tuple(stride), declared.data_type)
        dim = getattr(data, "shape", None)

        # torch: pointer from data_ptr(), strides from stride() in ELEMENTS
        if dim is not None and hasattr(data, "data_ptr") and callable(getattr(data, "stride", None)):
            return data.data_ptr(), describing_tensor(uid, tuple(dim), tuple(data.stride()), _buffer_dtype_to_cudnn(data.dtype))

        # cupy / numba: pointer from .data.ptr, strides from .strides in BYTES
        ptr = getattr(getattr(data, "data", None), "ptr", None)
        if dim is not None and ptr is not None:
            itemsize = data.dtype.itemsize
            strides = getattr(data, "strides", None)
            stride = tuple(s // itemsize for s in strides) if strides else _row_major_stride(dim)
            return int(ptr), describing_tensor(uid, tuple(dim), stride, _buffer_dtype_to_cudnn(data.dtype))

        # jax and bare DLPack producers: one capsule read, which is also the
        # only reader that gets cupy's byte strides right without knowing it is
        # cupy. It declines two different ways and they are NOT the same
        # answer, so they are told apart: "no protocol" means a pointer is all
        # this buffer will ever yield, while "dtype I cannot name" (fp8, fp4,
        # anything sub-byte) still has a real dim and stride worth keeping —
        # the torch branch above records those dtypes, and a buffer should not
        # be described differently for having come from jax.
        from .frost.buffers import _dlpack_geometry

        geometry = _dlpack_geometry(data)
        if geometry is None:  # neither __dlpack__ nor __cuda_array_interface__
            return self._device_pointer(data), Tensor(uid=uid, dim=tuple(dim) if dim is not None else [])
        ptr, dims, strides, name, _device = geometry
        # data_type is None for a dtype with no cuDNN enum (fp4 and friends)
        return ptr, describing_tensor(uid, tuple(dims), tuple(strides) if strides else _row_major_stride(dims), _buffer_dtype_to_cudnn(name))

    @staticmethod
    def _device_pointer(data: Any) -> int:
        if type(data) is int:
            return data
        if hasattr(data, "data_ptr"):
            return data.data_ptr()
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        return cudnn._pybind_module._get_data_ptr(data)  # dlpack fallback

    def _uid_to_data(self, tensor_dict) -> Dict[int, Any]:
        """Normalize a variant pack keyed by Tensor / name / uid to uid -> data,
        starting from the graph's auto-bound inputs (user keys win)."""
        uid_to_data = dict(self._data_bindings)
        for key, data in tensor_dict.items():
            if key is None:
                continue  # classic API tolerates None keys (optional tensors)
            if isinstance(key, Tensor):
                uid = key.uid
            elif isinstance(key, str):
                if key in self._ambiguous_names:
                    raise ValueError(f"tensor name {key!r} is ambiguous (duplicate labels); key the variant pack by uid or Tensor")
                uid = self._tensors[key].uid
            elif isinstance(key, int):
                uid = key
            else:  # a lowered C++ tensor (advanced/interop) — trust its uid
                uid = key.get_uid()
            uid_to_data[uid] = data
        return uid_to_data

    def _native_var_pack(self, uid_to_data: Dict[int, Any], workspace: Any):
        """uid -> device pointer, plus the workspace pointer, for the backend."""
        from .datatypes import _is_torch_tensor

        def _ptr(d):
            if type(d) is int:
                return d
            if _is_torch_tensor(d) or hasattr(d, "data_ptr"):
                return d.data_ptr()
            import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

            return cudnn._pybind_module._get_data_ptr(d)  # dlpack fallback

        return {uid: _ptr(d) for uid, d in uid_to_data.items()}, (_ptr(workspace) if workspace is not None else 0)

    def get_cuda_graph_retention(self):
        """The CUDA graph retention mode in effect for this graph: the one passed as
        ``cuda_graph_retention``, else the ``CUDNN_CUDA_GRAPH_RETENTION`` environment variable, else
        ``cuda_graph_retention.PER_GRAPH``. Available before the graph is lowered."""
        lowered = self.__dict__.get("_lowered_graph")
        if lowered is not None:
            return lowered.get_cuda_graph_retention()
        return _pybind_module._effective_cuda_graph_retention(self._cpp_graph_kwargs.get("cuda_graph_retention"))

    def __getattr__(self, name: str):
        # Plan-configuration and query methods (deselect_engines,
        # get_engine_and_knobs_at_index, key, populate_cuda_graph, ...) operate
        # on the lowered C++ graph — delegate to it. Only reached when normal
        # attribute lookup fails, i.e. for names this class doesn't define.
        lowered = self.__dict__.get("_lowered_graph")
        if lowered is not None and not name.startswith("_") and hasattr(lowered, name):
            return getattr(lowered, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
            + ("" if lowered is not None else " (graph not lowered yet — call build_operation_graph() first)")
        )

    def __repr__(self) -> str:
        if self._lowered_graph is not None:
            return repr(self._lowered_graph)  # classic JSON dump
        import json

        return json.dumps(self.inspect(), default=str, indent=2)

    def serialize(self, *, as_bytes: bool = False, include_kernels: bool = True):
        """Serialize the graph's selected execution plan.

        Returns a list of byte values by default. Set ``as_bytes=True`` to
        return the same binary data as compact ``bytes`` without constructing
        a Python list. ``deserialize`` accepts either representation.

        A backend plan serializes in the cuDNN backend's format. A python-engine
        plan (FROST, ...) serializes as the graph plus its ``(engine_id, knobs)``
        and, with ``include_kernels``, its compiled kernels (see
        ``cudnn._plan_serdes``); only ``pygraph.deserialize`` reads that format.
        """
        if self._planning_done and self.selected_engine is not None:
            from . import _plan_serdes

            if not self._is_built or self._plan_index not in self._compiled_plans:
                raise RuntimeError("serialize(): build the selected plan first (build_plans())")
            blob = _plan_serdes.serialize(self, include_kernels=include_kernels)
            return blob if as_bytes else list(blob)
        if self._lowered_graph is None:
            # Backend-selected or unplanned: the cuDNN graph format — lower on demand.
            node = self._unlowerable_node()
            if node is not None:
                # As key(): the format has no field for what the backend cannot
                # lower, so a blob would silently drop it (a SET softmax_precision
                # would deserialize as the f32 pipeline).
                raise cudnn_graph_not_supported(f"serialize() is the cuDNN backend's graph format; the {node.node_type.name} node has no backend lowering")
            self.validate()
            if self._lowered_graph is None:  # python engines registered
                self._lowered_graph = self._lower_to_cpp()
                self._lowered_graph.validate()
                self._verify_uid_ownership()
        return self._lowered_graph.serialize(as_bytes=as_bytes)

    def deserialize(self, *args, **kwargs) -> None:
        """Deserialize a graph. This is the one genuinely ambiguous classic
        overload -- ``(data)`` or ``(handle, data, enforce_precompiled=...)`` --
        so it stays a passthrough. The handle can arrive as the first positional
        or as the ``handle_`` keyword (the pybind overload's name); unwrap a
        Handle to its backend int and leave the ``data`` blob (the other overload)
        untouched. Binary data may be ``bytes`` or the legacy list of byte
        values; JSON strings retain their structural-graph meaning. A
        python-engine blob from ``serialize()`` is rebuilt here, into an empty
        graph, and its plan built."""
        python_plan = _python_plan_deserialize_args(args, kwargs)
        if python_plan is not None:
            from . import _plan_serdes

            handle, data, enforce_precompiled = python_plan
            _plan_serdes.deserialize(self, data, handle=handle, enforce_precompiled=enforce_precompiled)
            self._sorted_uids = None
            return
        if args and isinstance(args[0], Handle):
            args = (args[0].backend_handle,) + args[1:]
        if isinstance(kwargs.get("handle_"), Handle):
            kwargs["handle_"] = kwargs["handle_"].backend_handle
        if self._lowered_graph is None:
            import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

            if self._nodes:  # deserializing into a built-up graph: lower it
                self.validate()
                self._lowered_graph = self._lower_to_cpp()
            else:
                # Fresh container: forward only the fields that should survive
                # container replacement. Datatypes are deliberately NOT forwarded —
                # a deserialized plan carries its own context; forcing FLOAT here
                # would change existing one-argument deserialize(blob) callers.
                deser_kwargs = {}
                for _k in ("name", "kernel_cache", "device_property", "cuda_graph_retention"):
                    if _k in self._cpp_graph_kwargs:
                        deser_kwargs[_k] = self._cpp_graph_kwargs[_k]
                backend_handle = self._backend_handle_for_lowering(deser_kwargs)
                if backend_handle is not None:
                    deser_kwargs["handle"] = backend_handle
                self._lowered_graph = cudnn._pybind_module.backend_graph(**deser_kwargs)
        self._lowered_graph.deserialize(*args, **kwargs)
        self._is_built = True
        # The loaded graph carries its own variant_pack, so an order cached while
        # this container held a different graph no longer describes it.
        self._sorted_uids = None
        self._declared_layout_native = None
        self._slot_of_uid = None
        self._ordered_binding_schema = None
        self._ordered_binding_uids = None

    def _lower_to_cpp(self) -> Any:
        """Lower Python graph to C++ (the internal ``_pybind_module.backend_graph``)."""
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
        from .datatypes import _library_type  # torch dtype -> cudnn enum (classic parity)

        self._freeze()  # the lowered graph mirrors the IR from here on

        # The C++ graph rejects None (wants the enum). io_data_type may be unset
        # (block-scale tensors carry their own dtypes), but intermediate/compute
        # default to FLOAT — matching cudnn.graph() — so cuDNN can infer virtual
        # (intermediate) tensor dtypes during build.
        pg_kwargs = dict(self._cpp_graph_kwargs)
        if self._context.io_data_type is not None:
            pg_kwargs["io_data_type"] = _library_type(self._context.io_data_type)
        pg_kwargs["intermediate_data_type"] = _library_type(self._context.intermediate_data_type or cudnn.data_type.FLOAT)
        pg_kwargs["compute_data_type"] = _library_type(self._context.compute_data_type or cudnn.data_type.FLOAT)
        backend_handle = self._backend_handle_for_lowering(pg_kwargs)
        if backend_handle is not None:
            pg_kwargs["handle"] = backend_handle
        graph = cudnn._pybind_module.backend_graph(**pg_kwargs)

        tensor_map: Dict[int, Any] = {}

        def lower_tensor(t: Tensor) -> Any:
            if t.uid in tensor_map:
                return tensor_map[t.uid]
            if t.pass_by_value is not None and t.scalar_type is not None:
                cpp = graph.tensor_scalar(t.pass_by_value, t.scalar_type)
                cpp.set_uid(t.uid)
                tensor_map[t.uid] = cpp
                return cpp
            mk_kwargs = dict(
                dim=t.dim,
                stride=t.stride,
                is_virtual=t.is_virtual,
                is_pass_by_value=t.is_pass_by_value,
                name=t.name,
                # Always propagate the IR uid so execute()'s variant pack (keyed
                # by IR uid) matches; otherwise cuDNN assigns its own and the
                # buffers never bind. IR uids are unique and positive.
                uid=t.uid,
            )
            if t.data_type is not None:  # else NOT_SET → cuDNN infers from the
                mk_kwargs["data_type"] = _library_type(t.data_type)  # graph intermediate default
            if t.reordering_type is not None:  # e.g. F8_128x4 for block-scale SFs
                mk_kwargs["reordering_type"] = t.reordering_type
            if t.ragged_offset is not None:
                mk_kwargs["ragged_offset"] = lower_tensor(t.ragged_offset)
                if t.ragged_offset_multiplier not in (None, 1):  # non-default only
                    mk_kwargs["ragged_offset_multiplier"] = t.ragged_offset_multiplier
            cpp = graph._make_tensor(**mk_kwargs)
            tensor_map[t.uid] = cpp
            return cpp

        def push_output_attrs(out_t: Tensor, cpp_t: Any) -> None:
            # Every attribute a user may set on an OP OUTPUT via the classic
            # setter chain must be pushed here (inputs get theirs through
            # _make_tensor kwargs in lower_tensor). Missing one corrupts
            # silently: no multiplier -> wrong GPU addresses (cudaErrorMisalignedAddress);
            # no reordering -> the backend rejects or misreads the layout.
            # dim/stride: USER-assigned only — inferred IR strides are
            # provisional row-major; the backend keeps its classic per-op
            # layout inference (channels-last conv etc.) when the user did
            # not pin one.
            # the label too: classic renames act on the SAME object the cpp
            # graph holds, so the lowered graph carries the user's name
            if out_t.name:
                cpp_t.set_name(out_t.name)
            if out_t.dim_assigned and out_t.dim:
                cpp_t.set_dim(out_t.dim)
            if out_t.stride_assigned and out_t.stride:
                cpp_t.set_stride(out_t.stride)
            if out_t.ragged_offset is not None:
                cpp_t.set_ragged_offset(lower_tensor(out_t.ragged_offset))
                if out_t.ragged_offset_multiplier not in (None, 1):
                    cpp_t.set_ragged_offset_multiplier(out_t.ragged_offset_multiplier)
            if out_t.reordering_type is not None:
                cpp_t.set_reordering_type(out_t.reordering_type)
            if not out_t.is_virtual:
                cpp_t.set_output(True)
            if out_t.data_type:
                cpp_t.set_data_type(out_t.data_type)

        for node in self._nodes:
            for t in node.inputs.values():
                if t:
                    lower_tensor(t)

            if node.node_type == NodeType.MATMUL:
                mm_kw = dict(A=tensor_map[node.inputs["A"].uid], B=tensor_map[node.inputs["B"].uid], padding=node.params.get("padding", 0.0), name=node.name)
                if node.compute_data_type is not None:
                    mm_kw["compute_data_type"] = _library_type(node.compute_data_type)
                cpp_out = graph.matmul(**mm_kw)
            elif node.node_type == NodeType.POINTWISE:
                # params["mode"] IS the C++ pygraph method name — direct
                # dispatch; scalar attributes (clips/negative_slope/...) are
                # forwarded as keywords, tensors positionally (they lead every
                # pointwise signature).
                inputs = [tensor_map[t.uid] for t in node.inputs.values()]
                extra = {k: node.params[k] for k in self._POINTWISE_EXTRA_PARAMS if k in node.params}
                if node.compute_data_type is not None:
                    extra["compute_data_type"] = _library_type(node.compute_data_type)
                cpp_out = getattr(graph, node.params["mode"])(*inputs, name=node.name, **extra)
            elif node.node_type in _CAPTURED_BY_TYPE:
                # Captured op (sdpa family): rebuild the original kwargs —
                # tensor ports (port == C++ kwarg) map through tensor_map,
                # scalar params forward verbatim, dropout reassembles from its
                # flattened elements — and call the C++ method once.
                method, spec = _CAPTURED_BY_TYPE[node.node_type]
                kw = {"name": node.name}
                if node.compute_data_type is not None:
                    kw["compute_data_type"] = _library_type(node.compute_data_type)
                python_only = spec.get("python_only_attrs", ())
                for pk, pv in node.params.items():
                    if pk.startswith("_") or pk.startswith("dropout_") or pk in python_only:
                        continue  # python-only attrs never reach C++ (_unlowerable_node keeps a SET one off the backend)
                    # user callbacks (score_mod, ...) get a shimmed graph so
                    # closures over IR tensors keep working (see _CallbackGraphShim)
                    kw[pk] = _wrap_callback(pv, lower_tensor) if callable(pv) else pv
                python_only_ins = spec.get("python_only_in_kwargs", ())
                for port, t in node.inputs.items():
                    if not port.startswith("dropout_") and port not in python_only_ins:
                        kw[port] = tensor_map[t.uid]  # python-only inputs never reach C++ (_unlowerable_node keeps a SET one off the backend)
                python_only_outs = spec.get("python_only_out_kwargs", ())
                for port in spec.get("out_kwargs", ()):
                    if port in node.outputs and port not in python_only_outs:  # classic passes these descriptors as args
                        kw[port] = lower_tensor(node.outputs[port])
                n_drop = node.params.get("_dropout_n")
                if n_drop:
                    kw["dropout"] = tuple(
                        tensor_map[node.inputs[f"dropout_{i}"].uid] if f"dropout_{i}" in node.inputs else node.params[f"dropout_{i}"] for i in range(n_drop)
                    )
                result = getattr(graph, method)(**kw)
                cpp_outs = list(result) if isinstance(result, (list, tuple)) else [result]
                for oport, cpp_t in zip(spec["outputs"], cpp_outs):
                    out_t = node.outputs.get(oport)
                    if out_t is None or cpp_t is None:
                        continue
                    tensor_map[out_t.uid] = cpp_t
                    # sdpa-family output layout is user-chosen: the C++ node
                    # REQUIRES O's dim/stride before validate (BSHD vs BHSD) —
                    # push whatever the IR carries (inferred or user-set).
                    if out_t.dim:
                        cpp_t.set_dim(out_t.dim)
                    if out_t.stride:
                        cpp_t.set_stride(out_t.stride)
                    push_output_attrs(out_t, cpp_t)
                continue
            elif node.node_type in _STRUCTURED_BY_TYPE:
                # Generic structured op (norms / reduction / block-scale / MoE /
                # conv / structural): input ports are named after the C++
                # kwargs, so lowering is kwargs assembly + one call + zipping
                # the returned tuple with the declared output ports.
                method, spec = _STRUCTURED_BY_TYPE[node.node_type]
                if spec.get("python_only"):
                    raise cudnn.cudnnGraphNotSupportedError(
                        f"[cudnn_frontend] Error: No valid engine configs for {method.upper()}: "
                        f"{method} has no cuDNN backend lowering; register a python engine "
                        f"(e.g. cudnn.linear_attention.cutile.GdnCuTileEngine)"
                    )
                kw = {"name": node.name}
                if not spec.get("no_cdt") and node.compute_data_type is not None:
                    kw["compute_data_type"] = _library_type(node.compute_data_type)
                list_ports = spec.get("list_inputs", ())
                for port, t in node.inputs.items():
                    if any(port.startswith(f"{lp}_") for lp in list_ports):
                        continue  # collected below
                    kw[port] = tensor_map[t.uid]
                for lp in list_ports:
                    n = node.params.get(f"_n_{lp}", 0)
                    if n:
                        kw[lp] = [tensor_map[node.inputs[f"{lp}_{i}"].uid] for i in range(n)]
                for ak in spec.get("attrs", ()):
                    if ak in node.params:
                        kw[ak] = node.params[ak]
                result = getattr(graph, method)(**kw)
                cpp_outs = list(result) if isinstance(result, (list, tuple)) else [result]
                push_dims = spec.get("push_output_dims", False)
                for oport, cpp_t in zip(spec["outputs"], cpp_outs):
                    out_t = node.outputs.get(oport)
                    if out_t is None or cpp_t is None:
                        continue
                    tensor_map[out_t.uid] = cpp_t
                    if push_dims and out_t.dim:  # ops whose output dims cuDNN can't infer
                        cpp_t.set_dim(out_t.dim)
                        # stride only when USER-assigned: pushing the IR's
                        # provisional row-major stride into an (e.g.) NHWC
                        # graph makes the backend reject the fusion (classic
                        # infers the stride when the user sets only dims)
                        if out_t.stride_assigned and out_t.stride:
                            cpp_t.set_stride(out_t.stride)
                    push_output_attrs(out_t, cpp_t)
                continue
            else:
                continue

            # Map output
            for out_t in node.outputs.values():
                tensor_map[out_t.uid] = cpp_out
                push_output_attrs(out_t, cpp_out)

        # ---- uid ownership -------------------------------------------------
        # The Python IR owns the whole uid namespace: every IR tensor gets a uid
        # eagerly at creation (_alloc_uid, or user-specified via tensor(uid=)),
        # and lowering pushes ALL of them explicitly to C++ — inputs via
        # _make_tensor(uid=), op-created outputs/virtuals via set_uid here. The
        # C++ FE's build-time auto-assignment therefore NEVER triggers for
        # graphs built through the Python pygraph (its enumeration order is not
        # deterministic for multi-output ops, so relying on it mis-binds
        # buffers). Mixed construction — adding ops directly to the lowered C++
        # graph — is unsupported: a graph is either pure-Python or pure-C++.
        for ir_uid, cpp_t in tensor_map.items():
            cpp_t.set_uid(ir_uid)

        self._cpp_tensors = tensor_map
        return graph


def _install_pointwise_builders() -> None:
    """Generate the uniform pointwise builders from _POINTWISE_TENSOR_ARGS.

    Each builder accepts its tensors positionally OR by the classic pybind
    keyword names (e.g. ``g.bias(input=x, bias=b)``, ``g.max(input0=a,
    input1=b)``), matching the C++ pygraph API surface exactly. Ops with extra
    scalar attributes (relu / leaky_relu / swish / gen_index + backwards) have
    explicit builders on the class instead.
    """

    def make(op: str, argnames: tuple):
        def builder(self, *args, name: str = "", compute_data_type: Any = None, **kwargs):
            tensors = list(args)
            for an in argnames[len(args) :]:
                if an not in kwargs:
                    raise TypeError(f"{op}() missing tensor argument {an!r}")
                tensors.append(kwargs.pop(an))
            if len(tensors) != len(argnames) or kwargs:
                bad = kwargs or f"{len(tensors)} tensors"
                raise TypeError(f"{op}() expects tensor arguments {argnames}; got unexpected {bad}")
            return self._pointwise(op, tensors, self._get_name(op, name), compute_data_type)

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"Element-wise {op}({', '.join(argnames)})."
        return builder

    for op, argnames in pygraph._POINTWISE_TENSOR_ARGS.items():
        if not hasattr(pygraph, op):  # explicit builders (relu, ...) win
            setattr(pygraph, op, make(op, argnames))


_install_pointwise_builders()


# ---------------------------------------------------------------------------
# Structured ops, declaratively: norms, reduction, block-scale, MoE, conv, and
# the structural ops — everything except matmul (positional ergonomics) and
# sdpa (conditional kwarg assembly), which stay explicit.
#
# One table entry per op:
#   node_type          NodeType member (engines match on this)
#   inputs             ordered tensor ports == the C++ pybind kwarg names
#   list_inputs        ports taking a LIST of tensors (indexed ports + count)
#   attrs              scalar/enum/list params stored in node.params verbatim
#                      and forwarded as keywords at lowering
#   outputs            output ports, in C++ return order
#   maybe              per-output presence predicate over the node (an output
#                      whose predicate is False is skipped, and comes back None)
#   infer              per-output IR-side shape inference (introspection; cuDNN
#                      re-infers at build) — best-effort, None on failure
#   reinfer            output ports whose builder-time inferred dims validate()
#                      recomputes (they follow a sibling the caller may set later)
#   dtype_like         per-output data_type copied from a named input port
#   push_output_dims   True for ops whose output dims cuDNN cannot infer
#                      (dgrad/wgrad/reduction/reshape/...): IR dims are pushed
#   no_cdt             True for bindings without a compute_data_type kwarg
#
# Builders are generated: tensors positionally or by port name, attrs by
# keyword, plus a reserved ``out_dims`` kwarg (dims list for a single output,
# or {port: dims} for several) for the ambiguous-shape ops.
# ---------------------------------------------------------------------------


def _like(port):  # output dims mirror an input port
    return lambda node: (node.inputs[port].dim if port in node.inputs else None)


def _stats_like(port, keep_axes):  # input-port dims with all but keep_axes reduced to 1
    def infer(node):
        d = node.inputs[port].dim if port in node.inputs else None
        return [x if i in keep_axes else 1 for i, x in enumerate(d)] if d else None

    return infer


def _conv_fprop_dims(node):
    x, w = node.inputs["image"].dim, node.inputs["weight"].dim
    sp = len(x) - 2
    sym = node.params.get("padding")
    pre = node.params.get("pre_padding") or sym or [0] * sp
    post = node.params.get("post_padding") or sym or [0] * sp
    stride = node.params.get("stride") or [1] * sp
    dil = node.params.get("dilation") or [1] * sp
    out = [x[0], w[0]]
    for i in range(sp):
        eff = (w[i + 2] - 1) * dil[i] + 1
        out.append((x[i + 2] + pre[i] + post[i] - eff) // stride[i] + 1)
    return out


def _conv_dgrad_dims(node):
    dy, w = node.inputs["loss"].dim, node.inputs["filter"].dim
    sp = len(dy) - 2
    sym = node.params.get("padding")
    pre = node.params.get("pre_padding") or sym or [0] * sp
    post = node.params.get("post_padding") or sym or [0] * sp
    stride = node.params.get("stride") or [1] * sp
    dil = node.params.get("dilation") or [1] * sp
    # Reverse of fprop — ambiguous for strided conv; out_dims/set_dim overrides.
    out = [dy[0], w[1]]
    for i in range(sp):
        eff = (w[i + 2] - 1) * dil[i] + 1
        out.append((dy[i + 2] - 1) * stride[i] + eff - pre[i] - post[i])
    return out


def _slice_dims(node):  # output extent of each python slice over the input dims
    d = node.inputs["input"].dim
    sls = node.params.get("slices")
    if not d or not sls:
        return None
    return [len(range(*sl.indices(int(n)))) for sl, n in zip(sls, d)]


def _moe_bwd_dweight_dims(node):
    do, tok, fto = (node.inputs[p].dim for p in ("doutput", "token", "first_token_offset"))
    return [fto[0], tok[-1], do[-1]]  # [E, H, N]


def _linear_attention_final_state_dims(node):
    # [N, HO, V, K], one row per sequence -- or, when the state is addressed
    # through a ``state_indices`` pool-slot table, the caller's pool
    # [N_pool, HO, V, K], since the final state is written back into it in place.
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    cu = node.inputs.get("cu_seqlens")
    if cu is None or not cu.dim:
        return None
    rows = cu.dim[0] - 1
    state0 = node.inputs.get("initial_state")
    if node.inputs.get("state_indices") is not None and state0 is not None and state0.dim:
        rows = state0.dim[0]
    return [rows, max(q[-2], v[-2]), v[-1], q[-1]]


def _linear_attention_summary_final_dims(node):
    # [N, HO, V, K]
    k, v, g = node.inputs["k"].dim, node.inputs["v"].dim, node.inputs["g"].dim
    cu = node.inputs.get("cu_seqlens")
    if cu is None or not cu.dim:
        return None
    return [cu.dim[0] - 1, g[1], v[-1], k[-1]]


def _linear_attention_transition_dims(node):
    # [N, HO, K, K]
    k, g = node.inputs["k"].dim, node.inputs["g"].dim
    cu = node.inputs.get("cu_seqlens")
    if cu is None or not cu.dim:
        return None
    return [cu.dim[0] - 1, g[1], k[-1], k[-1]]


def linear_attention_summary_bwd_dims(node):
    # [N, HO, V, K]
    k, g, do = node.inputs["k"].dim, node.inputs["g"].dim, node.inputs["dO"].dim
    cu = node.inputs.get("cu_seqlens")
    if cu is None or not cu.dim:
        return None
    return [cu.dim[0] - 1, g[1], do[-1], k[-1]]


def _linear_attention_state_checkpoints_dims(node):
    n = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    cu = node.inputs["cu_seqlens"].dim if node.inputs.get("cu_seqlens") is not None else None
    if not n or not q or not v or not cu:
        return None
    return [max((v[0] * v[1] if len(v) == 4 else v[0]) // n + (cu[0] - 1), 1), max(q[-2], v[-2]), v[-1], q[-1]]


def _linear_attention_o_dims(node):
    # [total_T, HO, V]: the output lives at the gate heads (HO = max(q, v))
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    return [v[0] * v[1] if len(v) == 4 else v[0], max(q[-2], v[-2]), v[-1]]


def _gdp_o_dims(node):
    # [total_T, HO, V]: O follows q's rows; k/v carry the num_householder expansion (expanded rows or token-major)
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    return [q[0], max(q[-2], v[-2]), v[-1]]


def _mamba2_transition_dims(node):
    # [N, HO]: the Mamba-2 piece transition is one scalar (log2 of the decay product) per sequence and head
    g = node.inputs["g"].dim
    cu = node.inputs.get("cu_seqlens")
    if cu is None or not cu.dim:
        return None
    return [cu.dim[0] - 1, g[1]]


def _block_quant_scale_dims(node):
    d = list(node.inputs["input"].dim)
    bs = node.params.get("block_size")
    axis = node.params.get("axis")
    axis = len(d) - 1 if axis in (None, -1) else axis
    d[axis] = (d[axis] + bs - 1) // bs
    return d


def _norm_stats_like(first_axis):  # the C++ node's rule: input dims, 1 wherever scale is not 1 (from first_axis on)
    def infer(node):
        d = node.inputs["input"].dim if "input" in node.inputs else None
        s = node.inputs["scale"].dim if node.inputs.get("scale") is not None else None
        if not d:
            return None
        if not s:  # no scale (rmsnorm, #188): every axis after the batch axis is normalized
            return [x if i == 0 else 1 for i, x in enumerate(d)]
        return [1 if i >= first_axis and s[i] != 1 else x for i, x in enumerate(d)]

    return infer


def _layernorm_stat_like(other):  # the C++ layernorm node's rule: an unset stat takes the dims set on the other one
    def infer(node):
        t = node.outputs.get(other)
        return list(t.dim) if t is not None and t.dim_assigned and t.dim else _norm_stats_like(0)(node)

    return infer


_NORM_FWD_INFER = {"Y": _like("input"), "mean": _layernorm_stat_like("inv_var"), "inv_var": _layernorm_stat_like("mean")}
_ADANORM_FWD_INFER = {"Y": _like("input"), "mean": _norm_stats_like(1), "inv_var": _norm_stats_like(1)}
_NORM_BWD_INFER = {"DX": _like("input"), "DScale": _like("scale"), "DBias": _like("scale")}


def _training_phase(node):  # norm stats exist only in TRAINING forward phase
    phase = node.params.get("norm_forward_phase")
    return getattr(phase, "name", str(phase)).upper() != "INFERENCE"


_NORM_FWD_MAYBE = {"mean": _training_phase, "inv_var": _training_phase}


def _has_scale(node):  # rmsnorm/layernorm backward without a scale have no DScale or DBias (#188)
    return node.inputs.get("scale") is not None


_STRUCTURED_OPS = {
    # ---- norms --------------------------------------------------------------
    "rmsnorm": dict(
        node_type=NodeType.RMSNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "inv_var"),
        maybe={"inv_var": _training_phase},
        infer={"Y": _like("input"), "inv_var": _norm_stats_like(0)},
    ),
    "rmsnorm_backward": dict(
        node_type=NodeType.RMSNORM_BWD,
        inputs=("grad", "input", "scale", "inv_variance"),
        attrs=("has_dbias",),
        outputs=("DX", "DScale", "DBias"),
        # classic rmsnorm_backward names its outputs ::Dscale/::Dbias (mixed
        # case), unlike the other norm backwards (::DSCALE/::DBIAS)
        out_suffix={"DScale": "Dscale", "DBias": "Dbias"},
        maybe={"DScale": _has_scale, "DBias": lambda n: _has_scale(n) and n.params.get("has_dbias", True) is not False},
        infer=_NORM_BWD_INFER,
    ),
    "layernorm": dict(
        node_type=NodeType.LAYERNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer=_NORM_FWD_INFER,
        reinfer=("mean", "inv_var"),
    ),
    "layernorm_backward": dict(
        node_type=NodeType.LAYERNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        maybe={"DScale": _has_scale, "DBias": _has_scale},
        infer=_NORM_BWD_INFER,
    ),
    "adalayernorm": dict(
        node_type=NodeType.ADALAYERNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer=_ADANORM_FWD_INFER,
    ),
    "adalayernorm_backward": dict(
        node_type=NodeType.ADALAYERNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "instancenorm": dict(
        node_type=NodeType.INSTANCENORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer={"Y": _like("input"), "mean": _stats_like("input", (0, 1)), "inv_var": _stats_like("input", (0, 1))},
    ),
    "instancenorm_backward": dict(
        node_type=NodeType.INSTANCENORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "batchnorm": dict(
        node_type=NodeType.BATCHNORM,
        inputs=("input", "scale", "bias", "in_running_mean", "in_running_var", "epsilon", "momentum"),
        list_inputs=("peer_stats",),
        outputs=("Y", "mean", "inv_var", "next_running_mean", "next_running_var"),
        maybe={
            "next_running_mean": lambda n: "in_running_mean" in n.inputs,
            "next_running_var": lambda n: "in_running_var" in n.inputs,
        },
        infer={
            "Y": _like("input"),
            "mean": _stats_like("input", (1,)),
            "inv_var": _stats_like("input", (1,)),
            "next_running_mean": _stats_like("input", (1,)),
            "next_running_var": _stats_like("input", (1,)),
        },
    ),
    "batchnorm_inference": dict(
        node_type=NodeType.BATCHNORM_INFERENCE,
        inputs=("input", "mean", "inv_variance", "scale", "bias"),
        outputs=("Y",),
        infer={"Y": _like("input")},
    ),
    "batchnorm_backward": dict(
        node_type=NodeType.BATCHNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        list_inputs=("peer_stats",),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "genstats": dict(
        node_type=NodeType.GENSTATS,
        inputs=("input",),
        outputs=("SUM", "SQ_SUM"),
        infer={"SUM": _stats_like("input", (1,)), "SQ_SUM": _stats_like("input", (1,))},
    ),
    # ---- reduction / block-scale / MoE --------------------------------------
    "reduction": dict(
        node_type=NodeType.REDUCTION,
        inputs=("input", "group_offset"),
        attrs=("mode",),
        outputs=("OUT_0",),
        push_output_dims=True,  # cuDNN needs the reduced output dims explicitly
    ),
    "block_scale_dequantize": dict(
        node_type=NodeType.BLOCK_SCALE_DEQUANTIZE,
        inputs=("input", "descale"),
        attrs=("block_size", "is_negative_scale"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("input")},
    ),
    "block_scale_quantize": dict(
        node_type=NodeType.BLOCK_SCALE_QUANTIZE,
        inputs=("input", "group_offset"),
        attrs=("block_size", "axis", "transpose"),
        outputs=("Y", "scale"),
        infer={"Y": _like("input"), "scale": _block_quant_scale_dims},
    ),
    "moe_grouped_matmul": dict(
        node_type=NodeType.MOE_GROUPED_MATMUL,
        inputs=("token", "weight", "first_token_offset", "token_index", "token_ks"),
        keyword_inputs=("top_k_scores",),
        attrs=("mode", "top_k"),
        outputs=("OUT_0",),
        infer={
            "OUT_0": lambda n: [
                1,
                n.inputs[
                    (
                        "top_k_scores"
                        if n.params.get("mode") == cudnn.moe_grouped_matmul_mode.COMBINE
                        else "token_index" if n.params.get("mode") == cudnn.moe_grouped_matmul_mode.GATHER else "token"
                    )
                ].dim[-2],
                n.inputs["weight"].dim[-1],
            ]
        },
    ),
    "moe_grouped_matmul_bwd": dict(
        node_type=NodeType.MOE_GROUPED_MATMUL_BWD,
        inputs=("doutput", "token", "first_token_offset"),
        outputs=("dweight",),
        infer={"dweight": _moe_bwd_dweight_dims},
        push_output_dims=True,
    ),
    # ---- linear attention ----------------------------------------------------
    "gdn": dict(
        node_type=NodeType.GDN,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias", "state_indices"),
        attrs=(
            "scale",
            "output_final_state",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("O", "final_state", "state_checkpoints"),
        maybe={
            "final_state": lambda n: bool(n.params.get("output_final_state", False)),
            "state_checkpoints": lambda n: bool(n.params.get("checkpoint_every_n_tokens") or 0),
        },
        infer={"O": _linear_attention_o_dims, "final_state": _linear_attention_final_state_dims, "state_checkpoints": _linear_attention_state_checkpoints_dims},
        python_only=True,
    ),
    "gdn_bwd": dict(
        node_type=NodeType.GDN_BWD,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "dO", "state_checkpoints", "initial_state", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "scale",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("dQ", "dK", "dV", "dG", "dBeta", "d_initial_state", "d_a_log", "d_dt_bias"),
        maybe={
            "d_initial_state": lambda n: "initial_state" in n.inputs,
            "d_a_log": lambda n: "a_log" in n.inputs,
            "d_dt_bias": lambda n: "dt_bias" in n.inputs,
        },
        infer={
            "dQ": _like("q"),
            "dK": _like("k"),
            "dV": _like("v"),
            "dG": _like("g"),
            "dBeta": _like("beta"),
            "d_initial_state": _like("initial_state"),
            "d_a_log": _like("a_log"),
            "d_dt_bias": _like("dt_bias"),
        },
        dtype_like={"d_initial_state": "initial_state", "d_a_log": "a_log", "d_dt_bias": "dt_bias"},
        python_only=True,
    ),
    "gdn_summary": dict(
        node_type=NodeType.GDN_SUMMARY,
        inputs=("k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias"),
        attrs=("output_transition", "use_qk_l2norm", "use_beta_sigmoid", "allow_neg_eigval", "safe_gate", "gate_domain", "batch_invariant"),
        outputs=("final_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"final_state": _linear_attention_summary_final_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "gdn_summary_bwd": dict(
        node_type=NodeType.GDN_SUMMARY_BWD,
        inputs=("q", "k", "g", "beta", "cu_seqlens", "dO", "d_final_state", "a_log", "dt_bias"),
        attrs=("output_transition", "scale", "use_qk_l2norm", "use_beta_sigmoid", "allow_neg_eigval", "safe_gate", "gate_domain", "batch_invariant"),
        outputs=("d_initial_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"d_initial_state": linear_attention_summary_bwd_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "mamba2": dict(
        node_type=NodeType.MAMBA2,
        inputs=("q", "k", "v", "g", "a", "cu_seqlens", "initial_state", "dt_bias", "d", "z", "state_indices"),
        attrs=(
            "output_final_state",
            "checkpoint_every_n_tokens",
            "dt_softplus",
            "dt_min",
            "dt_max",
            "output_pre_gate",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("O", "final_state", "state_checkpoints", "O_pre"),
        maybe={
            "final_state": lambda n: bool(n.params.get("output_final_state", False)),
            "state_checkpoints": lambda n: bool(n.params.get("checkpoint_every_n_tokens") or 0),
            "O_pre": lambda n: bool(n.params.get("output_pre_gate", False)) and "z" in n.inputs,
        },
        infer={
            "O": _linear_attention_o_dims,
            "final_state": _linear_attention_final_state_dims,
            "state_checkpoints": _linear_attention_state_checkpoints_dims,
            "O_pre": _linear_attention_o_dims,
        },
        python_only=True,
    ),
    "mamba2_bwd": dict(
        node_type=NodeType.MAMBA2_BWD,
        inputs=("q", "k", "v", "g", "a", "cu_seqlens", "dO", "state_checkpoints", "initial_state", "d_final_state", "dt_bias", "d", "z", "O_pre"),
        attrs=(
            "checkpoint_every_n_tokens",
            "dt_softplus",
            "dt_min",
            "dt_max",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("dQ", "dK", "dV", "dG", "dA", "d_initial_state", "d_dt_bias", "dD", "dZ"),
        maybe={
            "d_initial_state": lambda n: "initial_state" in n.inputs,
            "d_dt_bias": lambda n: "dt_bias" in n.inputs,
            "dD": lambda n: "d" in n.inputs,
            "dZ": lambda n: "z" in n.inputs,
        },
        infer={
            "dQ": _like("q"),
            "dK": _like("k"),
            "dV": _like("v"),
            "dG": _like("g"),
            "dA": _like("a"),
            "d_initial_state": _like("initial_state"),
            "d_dt_bias": _like("dt_bias"),
            "dD": _like("d"),
            "dZ": _like("z"),
        },
        dtype_like={"d_initial_state": "initial_state", "dA": "a", "d_dt_bias": "dt_bias", "dD": "d", "dZ": "z"},
        python_only=True,
    ),
    "mamba2_summary": dict(
        node_type=NodeType.MAMBA2_SUMMARY,
        inputs=("k", "v", "g", "a", "cu_seqlens", "initial_state", "dt_bias"),
        attrs=("output_transition", "dt_softplus", "dt_min", "dt_max", "batch_invariant"),
        outputs=("final_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"final_state": _linear_attention_summary_final_dims, "transition": _mamba2_transition_dims},
        python_only=True,
    ),
    "mamba2_summary_bwd": dict(
        node_type=NodeType.MAMBA2_SUMMARY_BWD,
        inputs=("q", "k", "g", "a", "cu_seqlens", "dO", "d_final_state", "dt_bias"),
        attrs=("output_transition", "dt_softplus", "dt_min", "dt_max", "batch_invariant"),
        outputs=("d_initial_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"d_initial_state": linear_attention_summary_bwd_dims, "transition": _mamba2_transition_dims},
        python_only=True,
    ),
    "gdp": dict(
        node_type=NodeType.GDP,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias", "state_indices"),
        attrs=(
            "num_householder",
            "scale",
            "output_final_state",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("O", "final_state", "state_checkpoints"),
        maybe={
            "final_state": lambda n: bool(n.params.get("output_final_state", False)),
            "state_checkpoints": lambda n: bool(n.params.get("checkpoint_every_n_tokens") or 0),
        },
        infer={"O": _gdp_o_dims, "final_state": _linear_attention_final_state_dims, "state_checkpoints": _linear_attention_state_checkpoints_dims},
        python_only=True,
    ),
    "gdp_bwd": dict(
        node_type=NodeType.GDP_BWD,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "dO", "state_checkpoints", "initial_state", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "num_householder",
            "scale",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("dQ", "dK", "dV", "dG", "dBeta", "d_initial_state", "d_a_log", "d_dt_bias"),
        maybe={
            "d_initial_state": lambda n: "initial_state" in n.inputs,
            "d_a_log": lambda n: "a_log" in n.inputs,
            "d_dt_bias": lambda n: "dt_bias" in n.inputs,
        },
        infer={
            "dQ": _like("q"),
            "dK": _like("k"),
            "dV": _like("v"),
            "dG": _like("g"),
            "dBeta": _like("beta"),
            "d_initial_state": _like("initial_state"),
            "d_a_log": _like("a_log"),
            "d_dt_bias": _like("dt_bias"),
        },
        dtype_like={"d_initial_state": "initial_state", "d_a_log": "a_log", "d_dt_bias": "dt_bias"},
        python_only=True,
    ),
    "gdp_summary": dict(
        node_type=NodeType.GDP_SUMMARY,
        inputs=("k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias"),
        attrs=("num_householder", "output_transition", "use_qk_l2norm", "use_beta_sigmoid", "allow_neg_eigval", "safe_gate", "gate_domain", "batch_invariant"),
        outputs=("final_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"final_state": _linear_attention_summary_final_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "gdp_summary_bwd": dict(
        node_type=NodeType.GDP_SUMMARY_BWD,
        inputs=("q", "k", "g", "beta", "cu_seqlens", "dO", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "num_householder",
            "output_transition",
            "scale",
            "use_qk_l2norm",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "batch_invariant",
        ),
        outputs=("d_initial_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"d_initial_state": linear_attention_summary_bwd_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "kda": dict(
        node_type=NodeType.KDA,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias", "state_indices"),
        attrs=(
            "scale",
            "output_final_state",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("O", "final_state", "state_checkpoints"),
        maybe={
            "final_state": lambda n: bool(n.params.get("output_final_state", False)),
            "state_checkpoints": lambda n: bool(n.params.get("checkpoint_every_n_tokens") or 0),
        },
        infer={"O": _linear_attention_o_dims, "final_state": _linear_attention_final_state_dims, "state_checkpoints": _linear_attention_state_checkpoints_dims},
        python_only=True,
    ),
    "kda_bwd": dict(
        node_type=NodeType.KDA_BWD,
        inputs=("q", "k", "v", "g", "beta", "cu_seqlens", "dO", "state_checkpoints", "initial_state", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "scale",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("dQ", "dK", "dV", "dG", "dBeta", "d_initial_state", "d_a_log", "d_dt_bias"),
        maybe={
            "d_initial_state": lambda n: "initial_state" in n.inputs,
            "d_a_log": lambda n: "a_log" in n.inputs,
            "d_dt_bias": lambda n: "dt_bias" in n.inputs,
        },
        infer={
            "dQ": _like("q"),
            "dK": _like("k"),
            "dV": _like("v"),
            "dG": _like("g"),
            "dBeta": _like("beta"),
            "d_initial_state": _like("initial_state"),
            "d_a_log": _like("a_log"),
            "d_dt_bias": _like("dt_bias"),
        },
        dtype_like={"d_initial_state": "initial_state", "d_a_log": "a_log", "d_dt_bias": "dt_bias"},
        python_only=True,
    ),
    "kda_summary": dict(
        node_type=NodeType.KDA_SUMMARY,
        inputs=("k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias"),
        attrs=(
            "output_transition",
            "use_qk_l2norm",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
        ),
        outputs=("final_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"final_state": _linear_attention_summary_final_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "kda_summary_bwd": dict(
        node_type=NodeType.KDA_SUMMARY_BWD,
        inputs=("q", "k", "g", "beta", "cu_seqlens", "dO", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "output_transition",
            "scale",
            "use_qk_l2norm",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
        ),
        outputs=("d_initial_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"d_initial_state": linear_attention_summary_bwd_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "gdn2": dict(
        node_type=NodeType.GDN2,
        inputs=("q", "k", "v", "g", "beta", "w", "cu_seqlens", "initial_state", "a_log", "dt_bias", "state_indices"),
        attrs=(
            "scale",
            "output_final_state",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "beta_guard",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("O", "final_state", "state_checkpoints"),
        maybe={
            "final_state": lambda n: bool(n.params.get("output_final_state", False)),
            "state_checkpoints": lambda n: bool(n.params.get("checkpoint_every_n_tokens") or 0),
        },
        infer={"O": _linear_attention_o_dims, "final_state": _linear_attention_final_state_dims, "state_checkpoints": _linear_attention_state_checkpoints_dims},
        python_only=True,
    ),
    "gdn2_bwd": dict(
        node_type=NodeType.GDN2_BWD,
        inputs=("q", "k", "v", "g", "beta", "w", "cu_seqlens", "dO", "state_checkpoints", "initial_state", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "scale",
            "use_qk_l2norm",
            "checkpoint_every_n_tokens",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "beta_guard",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
            "overwrite_initial_state",
        ),
        outputs=("dQ", "dK", "dV", "dG", "dBeta", "dW", "d_initial_state", "d_a_log", "d_dt_bias"),
        maybe={
            "d_initial_state": lambda n: "initial_state" in n.inputs,
            "d_a_log": lambda n: "a_log" in n.inputs,
            "d_dt_bias": lambda n: "dt_bias" in n.inputs,
        },
        infer={
            "dQ": _like("q"),
            "dK": _like("k"),
            "dV": _like("v"),
            "dG": _like("g"),
            "dBeta": _like("beta"),
            "dW": _like("w"),
            "d_initial_state": _like("initial_state"),
            "d_a_log": _like("a_log"),
            "d_dt_bias": _like("dt_bias"),
        },
        dtype_like={"d_initial_state": "initial_state", "d_a_log": "a_log", "d_dt_bias": "dt_bias"},
        python_only=True,
    ),
    "gdn2_summary": dict(
        node_type=NodeType.GDN2_SUMMARY,
        inputs=("k", "v", "g", "beta", "w", "cu_seqlens", "initial_state", "a_log", "dt_bias"),
        attrs=(
            "output_transition",
            "use_qk_l2norm",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "beta_guard",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
        ),
        outputs=("final_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"final_state": _linear_attention_summary_final_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    "gdn2_summary_bwd": dict(
        node_type=NodeType.GDN2_SUMMARY_BWD,
        inputs=("q", "k", "g", "beta", "cu_seqlens", "dO", "d_final_state", "a_log", "dt_bias"),
        attrs=(
            "output_transition",
            "scale",
            "use_qk_l2norm",
            "use_beta_sigmoid",
            "allow_neg_eigval",
            "beta_guard",
            "safe_gate",
            "gate_domain",
            "gate_lower_bound",
            "batch_invariant",
        ),
        outputs=("d_initial_state", "transition"),
        maybe={"transition": lambda n: bool(n.params.get("output_transition", False))},
        infer={"d_initial_state": linear_attention_summary_bwd_dims, "transition": _linear_attention_transition_dims},
        python_only=True,
    ),
    # ---- convolution ---------------------------------------------------------
    "conv_fprop": dict(
        node_type=NodeType.CONV_FPROP,
        inputs=("image", "weight"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("Y",),
        infer={"Y": _conv_fprop_dims},
    ),
    "conv_dgrad": dict(
        node_type=NodeType.CONV_DGRAD,
        inputs=("loss", "filter"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("DX",),
        infer={"DX": _conv_dgrad_dims},
        push_output_dims=True,  # dgrad output dims are ambiguous for strided conv
    ),
    "conv_wgrad": dict(
        node_type=NodeType.CONV_WGRAD,
        inputs=("image", "loss"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("DW",),
        push_output_dims=True,  # wgrad output (filter) dims are not inferable
    ),
    # ---- structural -----------------------------------------------------------
    "reshape": dict(
        node_type=NodeType.RESHAPE,
        inputs=("input",),
        attrs=("reshape_mode",),
        outputs=("OUT_0",),
        push_output_dims=True,  # target shape comes from out_dims / set_dim
        no_cdt=True,
    ),
    "slice": dict(
        node_type=NodeType.SLICE,
        inputs=("input",),
        attrs=("slices",),
        outputs=("OUT_0",),
        infer={"OUT_0": _slice_dims},
        dtype_like={"OUT_0": "input"},  # classic: slice output dtype == input's
    ),
    "transpose": dict(
        node_type=NodeType.TRANSPOSE,
        inputs=("input",),
        attrs=("permutation",),
        outputs=("OUT_0",),
        infer={"OUT_0": lambda n: ([n.inputs["input"].dim[i] for i in n.params["permutation"]] if n.inputs["input"].dim else None)},
    ),
    "concatenate": dict(
        node_type=NodeType.CONCATENATE,
        inputs=(),
        list_inputs=("inputs",),
        attrs=("axis", "in_place_index"),
        outputs=("OUT_0",),
        no_cdt=True,
    ),
    "rope": dict(
        node_type=NodeType.ROPE,
        inputs=("input", "freqs"),
        attrs=("output_scale", "rope_dim"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("input")},
    ),
    "rope_backward": dict(
        node_type=NodeType.ROPE_BWD,
        inputs=("dY", "freqs"),
        attrs=("output_scale", "rope_dim"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("dY")},
    ),
}

# node_type -> (method name, spec), for the generic lowering branch
_STRUCTURED_BY_TYPE = {spec["node_type"]: (op, spec) for op, spec in _STRUCTURED_OPS.items()}


def _install_structured_builders() -> None:
    """Generate builders for _STRUCTURED_OPS.

    Call style: tensors positionally (in declared port order) or by port name;
    attrs by keyword; ``out_dims`` sets output dims explicitly (a dims list for
    single-output ops, or {port: dims}) for shapes cuDNN cannot infer."""

    def make(op: str, spec: dict):
        input_ports = spec["inputs"]
        list_ports = spec.get("list_inputs", ())
        attr_kws = spec.get("attrs", ())
        infer = spec.get("infer", {})
        maybe = spec.get("maybe", {})

        def builder(self, *args, name: str = "", compute_data_type: Any = None, out_dims: Any = None, **kwargs):
            name_ = self._get_name(op, name)
            node = Node(name_, spec["node_type"], compute_data_type or self._context.compute_data_type)
            # classic positional order: tensor ports first, then attrs
            n_p = len(input_ports)
            if len(args) > n_p + len(attr_kws):
                raise TypeError(f"{op}() takes at most {n_p + len(attr_kws)} positional arguments ({input_ports} + {attr_kws})")
            for ak, v in zip(attr_kws, args[n_p:]):
                if ak in kwargs:
                    raise TypeError(f"{op}() got multiple values for {ak!r}")
                kwargs[ak] = v
            args = args[:n_p]
            for port, v in zip(input_ports, args):
                if v is not None:  # an optional port passed positionally as None (e.g. scale, #188)
                    node.inputs[port] = self._ensure_tensor(v, name=f"{name_}::{port}")
            for port in input_ports[len(args) :]:
                v = kwargs.pop(port, None)
                if v is not None:
                    node.inputs[port] = self._ensure_tensor(v, name=f"{name_}::{port}")
            for port in spec.get("keyword_inputs", ()):
                v = kwargs.pop(port, None)
                if v is not None:
                    node.inputs[port] = self._ensure_tensor(v, name=f"{name_}::{port}")
            for lp in list_ports:
                vs = kwargs.pop(lp, None) or []
                for i, v in enumerate(vs):
                    node.inputs[f"{lp}_{i}"] = self._ensure_tensor(v, name=f"{name_}::{lp}_{i}")
                if vs:
                    node.params[f"_n_{lp}"] = len(vs)
            for ak in attr_kws:
                v = kwargs.pop(ak, None)
                if v is not None:
                    node.params[ak] = v
            if kwargs:
                raise TypeError(f"{op}() got unexpected arguments {sorted(kwargs)}; tensor ports are {input_ports}, attrs are {attr_kws}")
            if out_dims is not None and not isinstance(out_dims, dict):
                out_dims = {spec["outputs"][0]: out_dims}
            dtype_like = spec.get("dtype_like", {})
            outs = []
            for oport in spec["outputs"]:
                cond = maybe.get(oport)
                if cond is not None and not cond(node):
                    outs.append(None)  # classic returns None for absent outputs
                    continue
                o = self._make_output(f"{name_}::{spec.get('out_suffix', {}).get(oport) or self._CLASSIC_OUT_SUFFIX.get(oport, oport)}")
                src = dtype_like.get(oport)
                if src and src in node.inputs:
                    o.data_type = node.inputs[src].data_type
                d = (out_dims or {}).get(oport)
                if d is None:
                    try:  # best-effort IR-side inference; C++ validates at build
                        d = infer.get(oport, lambda n: None)(node)
                    except Exception:  # noqa: BLE001
                        d = None
                if d:
                    o.dim = list(d)
                    o.stride = _row_major_stride(o.dim)
                node.outputs[oport] = o
                self._register_tensor(o)
                outs.append(o)
            self._nodes.append(node)
            return outs[0] if len(outs) == 1 else list(outs)  # classic multi-output ops return a LIST

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"{op}({', '.join(input_ports)}) -> ({', '.join(spec['outputs'])})."
        return builder

    for op, spec in _STRUCTURED_OPS.items():
        setattr(pygraph, op, make(op, spec))


_install_structured_builders()


# ---------------------------------------------------------------------------
# Captured ops (the sdpa family): the kwarg surface is huge (~130 args across
# the six variants, including tensor-or-float args, dropout tuples, and
# score_mod callbacks), so builders capture ALL kwargs generically instead of
# hand-mirroring each one — tensors become named ports (port == C++ kwarg),
# everything else goes to params verbatim, dropout tuples are flattened per
# element. Lowering rebuilds the kwargs and makes one C++ call. The node stays
# first-class: engines read node.inputs["q"] / node.params["use_causal_mask"].
# ---------------------------------------------------------------------------


class _CallbackGraphShim:
    """Wraps the C++ graph handed to user callbacks (score_mod & co.) during
    lowering. Classic code passes the SAME object at build and callback time, so
    closures over user-created tensors just work; post-flip the user's closures
    capture IR Tensors while the callback receives the C++ graph. The shim
    translates any IR Tensor argument to its lowered C++ tensor at the call
    site, so existing callback code runs unchanged."""

    def __init__(self, target, lower_tensor):
        self._target = target
        self._lower = lower_tensor

    def _xlate(self, v):
        if isinstance(v, Tensor):
            # closure-captured helper tensors may not feed any node: lower on demand
            return self._lower(v)
        if isinstance(v, (list, tuple)):
            return type(v)(self._xlate(x) for x in v)
        return v

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            return attr(*[self._xlate(a) for a in args], **{k: self._xlate(v) for k, v in kwargs.items()})

        return call


def _wrap_callback(fn, lower_tensor):
    """Wrap a user callback param (e.g. score_mod) so the C++ graph it receives
    is shimmed (see _CallbackGraphShim) and stray IR-Tensor args translate."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        cpp_graph_t = cudnn._pybind_module.backend_graph

        def conv(v):
            if isinstance(v, cpp_graph_t):
                return _CallbackGraphShim(v, lower_tensor)
            if isinstance(v, Tensor):
                return lower_tensor(v)
            return v

        return fn(*[conv(a) for a in args], **{k: conv(v) for k, v in kwargs.items()})

    return wrapped


def _stats_expected(params):
    if params.get("generate_stats") is not None:
        return bool(params["generate_stats"])
    return not params.get("is_inference", True)


def _sdpa_o_dims(node):  # O: q dims with v's head dim
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    return list(q[:-1]) + [v[-1]]


def _sdpa_stats_dims(node):  # Stats: q dims with last dim 1
    return list(node.inputs["q"].dim[:-1]) + [1]


_AMAX = lambda node: [1, 1, 1, 1]  # noqa: E731 — fp8 amax side outputs

_CAPTURED_OPS = {
    "sdpa": dict(
        node_type=NodeType.SDPA,
        pos=("q", "k", "v"),
        outputs=("O", "Stats"),
        # kwargs whose tensors are semantically OUTPUTS of the node (the classic
        # API passes their descriptors as arguments): recorded in node.outputs so
        # engines see correct producer/consumer direction.
        out_kwargs=("rng_dump", "score_max", "score_sum_exp"),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims},
        # Op attributes the cuDNN backend has no field for: never forwarded to
        # C++; when SET they make the node backend-unlowerable so only a python
        # engine that honors them can serve the graph. `softmax_precision`
        # (cudnn.data_type.FLOAT | HALF, default FLOAT) asks for the softmax
        # accumulator precision -- numerics-changing, hence an op attribute
        # rather than a tuning knob.  `attn_scale_prefolded` (bool, default False)
        # states that the caller pre-multiplied Q by attn_scale * log2(e): the
        # engine applies no softmax scale (attn_scale must stay unset); False
        # counts as unset.  Served by the cc 10.7 python engines only.
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),
        # ``kv_mask``: a per-key mask ``(1|B, 1, 1, S_kv)`` (broadcast over the heads
        # and the query rows) the backend has no field for -- an INPUT python engines
        # fold in-kernel; a graph that binds it is python-engines-only (see
        # sdpa_mxfp8's scale_o).  BOOLEAN: False = the key is masked for every query,
        # like a key past ``seq_len_kv`` (a query whose every key is masked reads
        # O = 0 and Stats = -inf, the FROST empty-row contract).  The Q/K/V dtype or
        # FLOAT: an ADDITIVE per-key vector added to the scaled scores (the models'
        # finite -1e9 fold as a value; S_kv % 8 == 0, 16-byte strides).
        python_only_in_kwargs=("kv_mask",),
    ),
    "sdpa_backward": dict(
        node_type=NodeType.SDPA_BWD,
        # The forward's two softmax attributes are FORWARD-ONLY (the backward recomputes P from the
        # forward's unscaled contract).  Declared python-only here too so a SET value keeps the node off
        # the backend (the C++ binding has no such argument) and the analyzer's typed decline surfaces.
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),
        pos=("q", "k", "v", "o", "dO", "stats"),
        outputs=("dQ", "dK", "dV"),
        out_kwargs=("dBias", "dSink_token", "rng_dump"),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v")},
    ),
    "sdpa_fp8": dict(
        node_type=NodeType.SDPA_FP8,
        pos=("q", "k", "v", "descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"),
        outputs=("O", "Stats", "Amax_S", "Amax_O"),
        # ``sf_o``: block-scaled O scale factors (O declared FP4_E2M1 -> one
        # E4M3 scale per 16 d elements; O FP8_E4M3 + sf_o -> one UE8M0 scale
        # per 32). The caller passes its descriptor like rng_dump; the cuDNN
        # backend has no field for it, so a graph that sets it is served by
        # python engines only (see python_only_out_kwargs / _unlowerable_node).
        out_kwargs=("rng_dump", "score_max", "score_sum_exp", "sf_o"),
        python_only_out_kwargs=("sf_o",),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims, "Amax_S": _AMAX, "Amax_O": _AMAX},
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),  # see "sdpa"
    ),
    "sdpa_fp8_backward": dict(
        node_type=NodeType.SDPA_FP8_BWD,
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),  # forward-only, see "sdpa_backward"
        pos=(
            "q",
            "k",
            "v",
            "o",
            "dO",
            "stats",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_dO",
            "descale_s",
            "descale_dP",
            "scale_s",
            "scale_dQ",
            "scale_dK",
            "scale_dV",
            "scale_dP",
        ),
        outputs=("dQ", "dK", "dV", "amax_dQ", "amax_dK", "amax_dV", "amax_dP"),
        out_kwargs=("dSink_token",),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v"), "amax_dQ": _AMAX, "amax_dK": _AMAX, "amax_dV": _AMAX, "amax_dP": _AMAX},
    ),
    # mxfp8 variants (schemas match the bindings exactly).  The forward infers
    # its output dims the way sdpa / sdpa_fp8 do -- O / Stats from q and v,
    # Amax_O the [1, 1, 1, 1] scalar -- so an UNREQUESTED Amax_O (virtual, never
    # set_dim'd) passes the IR-level Tensor.validate() and stays a port the
    # python engines fold out (has_amax_o=False) instead of failing validate()
    # with "dims not set".  The mechanism, exactly: builder-time / validate-time
    # inference fills dim + a row-major stride on every output the caller has
    # NOT dimensioned; a caller's set_dim / set_stride overwrites that IR value
    # (it runs after the builder); and the sdpa-family arm of _lower_to_cpp
    # pushes WHATEVER the IR carries -- inferred or user-set -- so C++ receives
    # a virtual [1, 1, 1, 1] Amax_O for the formerly-failing undeclared case
    # (the state sdpa_fp8 has always produced) and a fully declared graph lowers
    # exactly as before.  Only push_output_attrs is user-assigned-only; it is not
    # what dimensions an sdpa-family output.  Shared hazard, inherited from sdpa
    # / sdpa_fp8 rather than new here: set_dim WITHOUT set_stride keeps the
    # provisional row-major stride of the INFERRED dims (Tensor.set_dim does not
    # drop a non-user-assigned stride, and validate()'s "stride optional" fill
    # fires only on an EMPTY stride), so dims that disagree with q[:-1]+[v[-1]]
    # / q[:-1]+[1] -- an inconsistent graph -- carry a stale stride.  Declare
    # dim AND stride, as every in-tree caller does.  The backward carries no
    # infer=; its output dims come from set_dim, as before.
    "sdpa_mxfp8": dict(
        node_type=NodeType.SDPA_MXFP8,
        pos=("q", "k", "v", "descale_q", "descale_k", "descale_v"),
        outputs=("O", "Stats", "Amax_O"),
        # Block-scaled O, as on sdpa_fp8: ``sf_o`` (E4M3 scales per 16 d for an
        # FP4_E2M1 O, UE8M0 per 32 d for an FP8_E4M3 O) is an OUTPUT the cuDNN
        # backend has no field for, and ``scale_o`` -- the FP4 global scale the
        # epilogue folds into O (required for an FP4 O, optional with an
        # UE8M0-scaled E4M3 O) -- an INPUT it has none for either: setting one
        # makes the node python-engines-only (python_only_* / _unlowerable_node).
        out_kwargs=("sf_o",),
        python_only_out_kwargs=("sf_o",),
        python_only_in_kwargs=("scale_o",),
        # The same two op attributes as "sdpa".  Without this entry a SET
        # softmax_precision would be forwarded to the C++ sdpa_mxfp8 (which has
        # no such argument) instead of keeping the node on the python engines.
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims, "Amax_O": _AMAX},
    ),
    "sdpa_mxfp8_backward": dict(
        node_type=NodeType.SDPA_MXFP8_BWD,
        python_only_attrs=("softmax_precision", "attn_scale_prefolded"),  # forward-only, see "sdpa_backward"
        pos=(
            "q",
            "q_T",
            "k",
            "k_T",
            "v",
            "o_f16",
            "dO_f16",
            "dO",
            "dO_T",
            "stats",
            "descale_q",
            "descale_q_T",
            "descale_k",
            "descale_k_T",
            "descale_v",
            "descale_dO",
            "descale_dO_T",
        ),
        outputs=("dQ", "dK", "dV", "amax_dQ", "amax_dK", "amax_dV"),
        out_kwargs=("dSink_token",),
    ),
}

_CAPTURED_BY_TYPE = {spec["node_type"]: (op, spec) for op, spec in _CAPTURED_OPS.items()}


def _install_captured_builders() -> None:
    """Generate the sdpa-family builders (generic kwarg capture)."""

    def _tensorish(v):
        return isinstance(v, Tensor) or hasattr(v, "__dlpack__")

    def make(op: str, spec: dict):
        pos = spec.get("pos", ())
        infer = spec.get("infer", {})
        maybe = spec.get("maybe", {})

        def builder(self, *args, name: str = "", compute_data_type: Any = None, out_dims: Any = None, **kwargs):
            name_ = self._get_name(op, name)
            node = Node(name_, spec["node_type"], compute_data_type or self._context.compute_data_type)
            if len(args) > len(pos):
                raise TypeError(f"{op}() takes at most {len(pos)} positional arguments {pos}")
            for k, v in zip(pos, args):
                if k in kwargs:
                    raise TypeError(f"{op}() got multiple values for {k!r}")
                kwargs[k] = v
            drop = kwargs.pop("dropout", None)
            out_kwargs = spec.get("out_kwargs", ())
            for k, v in kwargs.items():
                if v is None:
                    continue
                if _tensorish(v):
                    if k in out_kwargs:  # semantically an OUTPUT of this node
                        node.outputs[k] = self._ensure_tensor(v, name=f"{name_}::{k}")
                    else:
                        node.inputs[k] = self._ensure_tensor(v, name=f"{name_}::{k}")
                else:  # scalar / enum / callback — forwarded verbatim at lowering
                    node.params[k] = v
            if drop is not None:
                node.params["_dropout_n"] = len(drop)
                for i, e in enumerate(drop):
                    if _tensorish(e):
                        node.inputs[f"dropout_{i}"] = self._ensure_tensor(e, name=f"{name_}::dropout_{i}")
                    else:
                        node.params[f"dropout_{i}"] = e
            if out_dims is not None and not isinstance(out_dims, dict):
                out_dims = {spec["outputs"][0]: out_dims}
            rets = []
            for oport in spec["outputs"]:
                cond = maybe.get(oport)
                if cond is not None and not cond(node.params):
                    rets.append(None)  # e.g. Stats in inference mode (classic returns None)
                    continue
                o = self._make_output(f"{name_}::{spec.get('out_suffix', {}).get(oport) or self._CLASSIC_OUT_SUFFIX.get(oport, oport)}")
                d = (out_dims or {}).get(oport)
                if d is None:
                    try:
                        d = infer.get(oport, lambda n: None)(node)
                    except Exception:  # noqa: BLE001
                        d = None
                if d:
                    o.dim = list(d)
                    o.stride = _row_major_stride(o.dim)
                node.outputs[oport] = o
                self._register_tensor(o)
                rets.append(o)
            self._nodes.append(node)
            return list(rets)  # always full arity; classic returns a LIST

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"{op}(...) -> {spec['outputs']} (generic kwarg capture; see _CAPTURED_OPS)."
        return builder

    for op, spec in _CAPTURED_OPS.items():
        setattr(pygraph, op, make(op, spec))


_install_captured_builders()
