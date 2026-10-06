"""Prepared PCIe oneshot all-reduce launchers.

The declaration stores only immutable launch metadata.  The corresponding
materialized state retains the caller's already-established native runtime,
registered buffers, and resolved CuTe launchers.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
)
import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.compile_plan import load_programs

from .pcie_oneshot import IPC_SLAB_ALIGNMENT, _align_up, _eager_storage_shards
from ._tuning import PcieConfig, PcieQuery, TUNING

_PLAIN_SURFACES = frozenset(
    {
        "OneshotAllReduce.all_reduce",
        "OneshotAllReducePool.all_reduce",
    }
)
_FUSED_SURFACES = frozenset(
    {
        "OneshotAllReduce.all_reduce_fused_add_rms_norm",
        "OneshotAllReducePool.all_reduce_fused_add_rms_norm",
    }
)


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[dtype]


def _input_from_call(call: Mapping[str, object]) -> torch.Tensor:
    inp = call.get("inp")
    if not isinstance(inp, torch.Tensor):
        raise TypeError("oneshot preparation call requires its actual inp tensor")
    return inp


def _state(runtime):
    try:
        return runtime._ext._state(runtime._ptr)
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "oneshot preparation requires a live native oneshot runtime"
        ) from exc


@dataclass(frozen=True)
class _InvocationTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    _stride: tuple[int, ...]
    device: torch.device
    alignment: int = 16

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def element_size(self):
        return self.dtype.itemsize

    def stride(self):
        return self._stride


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _owned_resident_layout(runtime, native) -> tuple[int, int, int]:
    """Return the native runtime's actual locally-owned allocation layout.

    The extension owns the signal metadata size; the Python runtime owns the
    rank-data tensor and retains each locally exported IPC slab in
    ``_owned_buffers``.  This intentionally does not infer ownership from
    imported peer pointers or a pool-only convenience attribute.
    """
    rank_data = runtime.rank_data
    rank_data_nbytes = int(rank_data.numel() * rank_data.element_size())
    eager_bytes = native.eager_buffer_bytes
    if eager_bytes is None:
        return rank_data_nbytes, 0, 0
    shards = _eager_storage_shards(
        int(runtime.world_size), tuple(bool(value) for value in native.transport_policy)
    )
    signal_nbytes = _align_up(int(runtime._ext.meta_size()), IPC_SLAB_ALIGNMENT)
    eager_nbytes = _align_up(int(eager_bytes) * shards, IPC_SLAB_ALIGNMENT)
    slab_nbytes = signal_nbytes + 2 * eager_nbytes
    return rank_data_nbytes, slab_nbytes, len(runtime._owned_buffers)


def _plain_launcher_metadata(backend, native, inp):
    graph = backend._plain_graph_plan(native, inp)
    eager_transport, eager_threads, eager_blocks = backend._plain_launch_config(
        native,
        inp,
        graph_channel=False,
    )
    return {
        "dtype": _dtype_name(inp.dtype),
        "transport": graph.transport,
        "threads": graph.threads,
        "blocks": graph.blocks,
        "eager_transport": eager_transport,
        "eager_threads": eager_threads,
        "eager_blocks": eager_blocks,
        "size_packs": graph.size_packs,
        "stage_input": graph.stage_input,
        "device_index": graph.device_index,
        "remote_push_region_packs": graph.remote_push_region_packs,
    }


def query_from_metadata(
    runtime,
    *,
    surface: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    strides: tuple[int, ...] | None = None,
    alignment: int = 16,
) -> PcieQuery:
    """Extract a native declaration from immutable invocation metadata only."""
    tensor = _InvocationTensor(
        tuple(shape),
        dtype,
        _contiguous_strides(tuple(shape)) if strides is None else tuple(strides),
        runtime.device,
        alignment,
    )
    if surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {surface!r}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("oneshot invocation dtype is unsupported")
    if (
        tensor.numel() * tensor.element_size() > runtime.max_size
        or (tensor.numel() * tensor.element_size()) % 16
    ):
        raise ValueError("oneshot invocation exceeds its established runtime capacity")
    native = _state(runtime)
    backend = runtime._ext
    stage_input = native.eager_tables is not None
    variants = ((False, 0), (True, 0), (True, 1)) if stage_input else ((False, 0),)
    if surface in _PLAIN_SURFACES:
        launcher = _plain_launcher_metadata(backend, native, tensor)
        registered = not launcher["stage_input"]
    else:
        if tensor.ndim == 0 or tensor.shape[-1] * tensor.element_size() % 16:
            raise ValueError(
                "fused oneshot input last dimension must occupy 16-byte packs"
            )
        mode, single_cta, register_normalize, threads = backend._fused_launch_config(
            native, tensor
        )
        pack_elems = 16 // tensor.element_size()
        hidden_packs = int(tensor.shape[-1]) // pack_elems
        rows = tensor.numel() // int(tensor.shape[-1])
        if rows > 36:
            raise ValueError("fused allreduce RMSNorm supports at most 36 rows")
        min_ctas = (hidden_packs + threads * 3 - 1) // (threads * 3)
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        ctas = override if override > 0 else max(max(1, 3 // rows), min_ctas)
        ctas = max(1, min(ctas, 36 // rows))
        launcher = {
            "dtype": _dtype_name(dtype),
            "mode": mode,
            "single_cta": single_cta,
            "register_normalize": register_normalize,
            "threads": threads,
            "device_index": backend._device_index(runtime.device),
            "hidden_packs": hidden_packs,
            "rows": rows,
            "ctas_per_row": ctas,
            "blocks": rows * ctas,
            "stage_input": stage_input,
        }
        registered = not stage_input
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    setup = {
        "shape": tensor.shape,
        "stride": tensor.stride(),
        "dtype": _dtype_name(dtype),
        "device_index": backend._device_index(runtime.device),
        "elements": tensor.numel(),
        "element_size": tensor.element_size(),
        "rows": tensor.numel() // tensor.shape[-1],
        "rank_data_bytes": int(runtime.rank_data_bytes),
        "rank_data_nbytes": rank_data_nbytes,
        "owned_slab_nbytes": slab_nbytes,
        "owned_slab_count": owned_slab_count,
        "eager_buffer_bytes": None
        if native.eager_buffer_bytes is None
        else int(native.eager_buffer_bytes),
        "eager_storage_shards": _eager_storage_shards(
            int(runtime.world_size),
            tuple(bool(value) for value in native.transport_policy),
        ),
        "registered": registered,
        "variants": variants,
    }
    return PcieQuery(
        surface=surface,
        world_size=int(runtime.world_size),
        rank=int(runtime.rank),
        topology="pcie_ipc",
        call=FrozenMapping(launcher),
        setup=FrozenMapping(setup),
    )


def query_from_runtime(runtime, *, surface, call) -> PcieQuery:
    """Extract the exact immutable launcher contract from a live channel."""
    if surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {surface!r}")
    values = dict(call)
    inp = _input_from_call(values)
    native = _state(runtime)
    if inp.device != runtime.device:
        raise ValueError("oneshot input device differs from its native runtime")
    if not runtime.should_allreduce(inp):
        raise ValueError("input does not satisfy PCIe oneshot requirements")
    if surface in _FUSED_SURFACES and (
        inp.ndim == 0 or inp.shape[-1] * inp.element_size() % 16
    ):
        raise ValueError("fused oneshot input last dimension must occupy 16-byte packs")
    backend = runtime._ext
    dtype = _dtype_name(inp.dtype)
    stage_input = native.eager_tables is not None
    variants = ((False, 0), (True, 0), (True, 1)) if stage_input else ((False, 0),)
    if surface in _PLAIN_SURFACES:
        launcher = _plain_launcher_metadata(backend, native, inp)
    else:
        mode, single_cta, register_normalize, threads = backend._fused_launch_config(
            native, inp
        )
        pack_elems = 16 // inp.element_size()
        hidden_packs = int(inp.shape[-1]) // pack_elems
        rows = inp.numel() // int(inp.shape[-1])
        if rows > 36:
            raise ValueError("fused allreduce RMSNorm supports at most 36 rows")
        min_ctas = (hidden_packs + threads * 3 - 1) // (threads * 3)
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        ctas = override if override > 0 else max(max(1, 3 // rows), min_ctas)
        ctas = max(1, min(ctas, 36 // rows))
        launcher = {
            "dtype": dtype,
            "mode": mode,
            "single_cta": single_cta,
            "register_normalize": register_normalize,
            "threads": threads,
            "device_index": backend._device_index(inp.device),
            "hidden_packs": hidden_packs,
            "rows": rows,
            "ctas_per_row": ctas,
            "blocks": rows * ctas,
            "stage_input": stage_input,
        }
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    setup = {
        "shape": tuple(inp.shape),
        "stride": tuple(inp.stride()),
        "dtype": dtype,
        "device_index": backend._device_index(inp.device),
        "elements": int(inp.numel()),
        "element_size": int(inp.element_size()),
        "rows": 1 if inp.ndim == 0 else int(inp.numel() // inp.shape[-1]),
        "rank_data_bytes": int(runtime.rank_data_bytes),
        "rank_data_nbytes": rank_data_nbytes,
        "owned_slab_nbytes": slab_nbytes,
        "owned_slab_count": owned_slab_count,
        "eager_buffer_bytes": None
        if native.eager_buffer_bytes is None
        else int(native.eager_buffer_bytes),
        "eager_storage_shards": _eager_storage_shards(
            int(runtime.world_size),
            tuple(bool(value) for value in native.transport_policy),
        ),
        "registered": not stage_input,
        "variants": variants,
    }
    return PcieQuery(
        surface=surface,
        world_size=int(runtime.world_size),
        rank=int(runtime.rank),
        topology="pcie_ipc",
        call=FrozenMapping(launcher),
        setup=FrozenMapping(setup),
    )


@program_cache(scope="preparation")
def compile_oneshot_surface(query_payload, ordinal):
    """Compile all fixed slot variants represented by metadata only."""
    payload = dict(query_payload)
    query = PcieQuery(
        surface=payload["surface"],
        world_size=payload["world_size"],
        rank=payload["rank"],
        topology=payload["topology"],
        call=FrozenMapping(payload["call"]),
        setup=FrozenMapping(payload["setup"]),
    )
    call, setup = dict(query.call), dict(query.setup)
    variants = tuple(tuple(value) for value in setup["variants"])
    with torch.cuda.device(int(ordinal)):
        if query.surface in _PLAIN_SURFACES:
            from ._oneshot_cute import get_oneshot_launcher

            return {
                variant: get_oneshot_launcher(
                    call["dtype"],
                    query.world_size,
                    query.rank,
                    call["stage_input"],
                    variant[0],
                    variant[1],
                    call["transport"] if variant[0] else call["eager_transport"],
                    call["threads"] if variant[0] else call["eager_threads"],
                    call["device_index"],
                )
                for variant in variants
            }
        if query.surface in _FUSED_SURFACES:
            from ._oneshot_cute import get_fused_oneshot_launcher

            return {
                variant: get_fused_oneshot_launcher(
                    call["dtype"],
                    query.world_size,
                    query.rank,
                    call["mode"],
                    call["single_cta"],
                    call["register_normalize"],
                    variant[0],
                    variant[1],
                    call["threads"],
                    call["device_index"],
                )
                for variant in variants
            }
    raise ValueError(f"unsupported oneshot preparation surface {query.surface!r}")


@dataclass
class _OneshotExecutionState:
    query: PcieQuery
    runtime: object
    launchers: Mapping[tuple[bool, int], object]
    _bound_input: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected = tuple(tuple(variant) for variant in self.query.setup["variants"])
        missing = tuple(
            variant for variant in expected if self.launchers.get(variant) is None
        )
        if missing:
            raise RuntimeError(
                f"oneshot preparation did not retain required slot variants: {missing}"
            )

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("PCIe plan belongs to a different native runtime owner")
        if (runtime.rank, runtime.world_size) != (
            self.query.rank,
            self.query.world_size,
        ):
            raise ValueError("PCIe runtime identity differs from preparation")

    def _check_input(self, inp):
        setup = self.query.setup
        if (
            tuple(inp.shape),
            tuple(inp.stride()),
            _dtype_name(inp.dtype),
            inp.device.index,
        ) != (
            tuple(setup["shape"]),
            tuple(setup["stride"]),
            setup["dtype"],
            setup["device_index"],
        ):
            raise ValueError("oneshot input metadata differs from the prepared plan")

    def bind_input(self, inp: torch.Tensor) -> None:
        self._check_input(inp)
        self.runtime._prepare_prepared_surface(self.query, inp)
        # Staged runtimes copy each serving input into their fixed IPC slabs;
        # retaining the disposable priming tensor would make it look like
        # serving depends on that tensor.  Direct registered runtimes instead
        # have a real producer-owned pointer contract and keep that binding.
        self._bound_input = inp if self.runtime._eager_ptrs is None else None

    def _require_bound_input(self, inp):
        self._check_input(inp)
        if self._bound_input is None:
            if self.runtime._eager_ptrs is not None:
                return
            raise RuntimeError("oneshot plan was not bound during preparation")

    def run_plain(self, inp, out):
        self._require_bound_input(inp)
        return self.runtime._run_prepared_plain(inp, out, self.query, self.launchers)

    def run_fused(self, inp, residual, weight, out, residual_out, epsilon):
        self._require_bound_input(inp)
        return self.runtime._run_prepared_fused(
            inp,
            residual,
            weight,
            out,
            residual_out,
            epsilon,
            self.query,
            self.launchers,
        )


def _prepare_plain_call(
    state: _OneshotExecutionState,
    *,
    inp: torch.Tensor,
    out: torch.Tensor,
    produce=None,
    reset=None,
    restore=None,
    owners=(),
) -> PreparedCall:
    """Bind and prime a caller-owned plain all-reduce without a public warmup API."""
    state.bind_input(inp)
    return PreparedCall(
        run=lambda: state.run_plain(inp, out),
        output=out,
        produce=produce,
        reset=reset,
        restore=restore,
        owners=tuple(owners),
    )


def _prepare_fused_call(
    state: _OneshotExecutionState,
    *,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    epsilon: float,
    produce=None,
    reset=None,
    restore=None,
    owners=(),
) -> PreparedCall:
    """Bind and prime the complete fused arithmetic with its actual resources."""
    state.bind_input(inp)
    return PreparedCall(
        run=lambda: state.run_fused(inp, residual, weight, out, residual_out, epsilon),
        output=(out, residual_out),
        produce=produce,
        reset=reset,
        restore=restore,
        owners=tuple(owners),
    )


def plan(
    query: PcieQuery,
    *,
    runtime,
    invocation=FrozenMapping(),
    override: PcieConfig | None = None,
) -> Plan:
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    if query.surface not in _PLAIN_SURFACES | _FUSED_SURFACES:
        raise ValueError(f"unsupported oneshot preparation surface {query.surface!r}")
    if runtime is None:
        raise TypeError(
            "oneshot declarations require the existing native runtime owner"
        )
    if invocation:
        raise ValueError("oneshot invocation semantics belong in PcieQuery")

    def materialize(selection, device):
        launchers = load_programs(
            compile_oneshot_surface(TUNING.encode_query(query), device.ordinal)
        )
        return _OneshotExecutionState(query, runtime, MappingProxyType(launchers))

    native = _state(runtime)
    rank_data_nbytes, slab_nbytes, owned_slab_count = _owned_resident_layout(
        runtime, native
    )
    resident = rank_data_nbytes + slab_nbytes * owned_slab_count

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._oneshot_preparation:compile_oneshot_surface",
                TUNING.encode_query(query),
                device.ordinal,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(
                    ("pcie-oneshot-runtime", id(runtime)),
                    resident,
                    resident,
                ),
            ),
        ),
        _materialize=materialize,
        _device=runtime.device,
    )
