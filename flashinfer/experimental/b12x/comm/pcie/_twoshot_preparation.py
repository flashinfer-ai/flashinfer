"""Prepared launch ownership for PCIe two-shot collectives.

The IPC slabs are created by the communicator factory.  This module only
captures their immutable ABI and resolves the exact native launchers before a
runtime call is admitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping, MemoryRequirements, PersistentMemory, Plan, PreparedCall,
    current_plan,
)

from ._tuning import PcieConfig, PcieQuery, TUNING

_FP8_SURFACES = frozenset({
    "TwoShotReduceScatter.reduce_scatter_fp8",
    "TwoShotReduceScatter.all_gather_fp8",
})
_BF16_SURFACES = frozenset({
    "PCIeTwoShotBF16.reduce_scatter",
    "PCIeTwoShotBF16.all_gather",
    "PCIeTwoShotBF16.all_reduce",
})
_SURFACES = _FP8_SURFACES | _BF16_SURFACES


def _alignment(tensor: torch.Tensor) -> int:
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def _operation(surface: str) -> str:
    if surface.endswith("reduce_scatter_fp8") or surface.endswith("reduce_scatter"):
        return "reduce_scatter"
    if surface.endswith("all_gather_fp8") or surface.endswith("all_gather"):
        return "all_gather"
    if surface.endswith("all_reduce"):
        return "all_reduce"
    raise ValueError(f"unsupported two-shot preparation surface {surface!r}")


def _tensor(values: Mapping[str, object], name: str) -> torch.Tensor:
    value = values.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"two-shot preparation requires actual {name} tensor")
    return value


def query_from_runtime(runtime, *, surface: str, call) -> PcieQuery:
    """Normalize exact tensor/launch ABI without retaining tensors or groups."""
    if surface not in _SURFACES:
        raise ValueError(f"two-shot preparation does not own {surface!r}")
    values = dict(call)
    payload = _tensor(values, "payload" if surface != "PCIeTwoShotBF16.all_reduce" else "inp")
    if payload.device != runtime.device:
        raise ValueError("two-shot payload device differs from runtime")
    if payload.ndim == 0 or payload.numel() % int(runtime.row_elems):
        raise ValueError("two-shot payload must contain complete native rows")
    rows = payload.numel() // int(runtime.row_elems)
    operation = _operation(surface)
    if operation == "reduce_scatter" and rows % int(runtime.world_size):
        raise ValueError("two-shot reduce-scatter rows must divide world size")
    if surface in _FP8_SURFACES:
        scale = _tensor(values, "scale")
        if scale.device != runtime.device or scale.numel() != rows:
            raise ValueError("FP8 two-shot scale differs from payload rows/runtime")
        if payload.dtype not in (torch.float8_e4m3fn, torch.uint8) or scale.dtype is not torch.float32:
            raise TypeError("FP8 two-shot requires E4M3/uint8 payload and float32 scale")
        dtype = "float8_e4m3fn" if payload.dtype is torch.float8_e4m3fn else "uint8"
        scale_dtype = str(scale.dtype).removeprefix("torch.")
        scale_shape, scale_stride, scale_alignment = tuple(scale.shape), tuple(scale.stride()), _alignment(scale)
    else:
        if payload.dtype is not torch.bfloat16:
            raise TypeError("BF16 two-shot payload must be bfloat16")
        dtype, scale_dtype = "bfloat16", None
        scale_shape, scale_stride, scale_alignment = None, None, None
    threads = int(values.get("threads", 512))
    block_limit = int(values.get("block_limit", 64))
    if threads <= 0 or threads > 512 or threads % 32:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if block_limit <= 0 or block_limit > 64:
        raise ValueError("block_limit must be in [1, 64]")
    output = values.get("out")
    if output is not None and not isinstance(output, torch.Tensor):
        raise TypeError("two-shot output must be a tensor when supplied")
    call_metadata = FrozenMapping({
        "operation": operation, "dtype": dtype, "scale_dtype": scale_dtype,
        "rows": int(rows), "row_elems": int(runtime.row_elems),
        "shape": tuple(payload.shape), "stride": tuple(payload.stride()),
        "alignment": _alignment(payload), "scale_shape": scale_shape,
        "scale_stride": scale_stride, "scale_alignment": scale_alignment,
        "output_shape": None if output is None else tuple(output.shape),
        "output_stride": None if output is None else tuple(output.stride()),
        "output_dtype": None if output is None else str(output.dtype).removeprefix("torch."),
        "output_alignment": None if output is None else _alignment(output),
        "threads": threads, "block_limit": block_limit,
        "device_slot_variants": ((False, 0), (True, 0), (True, 1)),
    })
    # _OwnedSharedBuffer stores only pointers.  The pointer delta is exactly
    # the aligned signal prefix, followed by both fixed protocol slots.
    setup = FrozenMapping({
        "max_rows": int(runtime.max_rows), "pack_stride": int(runtime._pack_stride),
        "slot_bytes": int(runtime._slot_bytes), "slots": 2,
        "signal_ptr_count": len(runtime._signal_ptrs),
        "staging_ptr_count": tuple(len(slot) for slot in runtime._staging_ptrs),
        "slab_nbytes": 2 * int(runtime._slot_bytes)
        + int(runtime._staging_ptrs[0][runtime.rank] - runtime._signal_ptrs[runtime.rank]),
    })
    return PcieQuery(surface=surface, world_size=int(runtime.world_size), rank=int(runtime.rank),
                     topology="pcie_ipc", call=call_metadata, setup=setup)


def query_from_metadata(
    runtime, *, surface: str, shape: tuple[int, ...], dtype: torch.dtype,
    strides: tuple[int, ...] | None = None, alignment: int = 16,
) -> PcieQuery:
    """Declare a BF16 all-reduce from producer metadata without an activation."""
    if surface != "PCIeTwoShotBF16.all_reduce":
        raise ValueError(f"unsupported metadata two-shot surface {surface!r}")
    if dtype is not torch.bfloat16:
        raise TypeError("BF16 two-shot metadata requires bfloat16")
    if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("two-shot metadata requires a positive shape")
    count = 1
    for extent in shape:
        count *= extent
    if count % int(runtime.row_elems):
        raise ValueError("two-shot metadata must contain complete native rows")
    contiguous = []
    stride = 1
    for extent in reversed(shape):
        contiguous.append(stride)
        stride *= extent
    contiguous = tuple(reversed(contiguous))
    if strides is None:
        strides = contiguous
    if tuple(strides) != contiguous:
        raise ValueError("two-shot metadata requires contiguous payload layout")
    if alignment < 16 or alignment & (alignment - 1):
        raise ValueError("two-shot metadata alignment must be a power of two >= 16")
    rows = count // int(runtime.row_elems)
    call_metadata = FrozenMapping({
        "operation": "all_reduce", "dtype": "bfloat16", "scale_dtype": None,
        "rows": rows, "row_elems": int(runtime.row_elems), "shape": tuple(shape),
        "stride": tuple(strides), "alignment": alignment, "scale_shape": None,
        "scale_stride": None, "scale_alignment": None, "output_shape": tuple(shape),
        "output_stride": tuple(strides), "output_dtype": "bfloat16",
        "output_alignment": alignment, "threads": 512, "block_limit": 64,
        "device_slot_variants": ((False, 0), (True, 0), (True, 1)),
    })
    setup = FrozenMapping({
        "max_rows": int(runtime.max_rows), "pack_stride": int(runtime._pack_stride),
        "slot_bytes": int(runtime._slot_bytes), "slots": 2,
        "signal_ptr_count": len(runtime._signal_ptrs),
        "staging_ptr_count": tuple(len(slot) for slot in runtime._staging_ptrs),
        "slab_nbytes": 2 * int(runtime._slot_bytes)
        + int(runtime._staging_ptrs[0][runtime.rank] - runtime._signal_ptrs[runtime.rank]),
    })
    if rows > int(runtime.max_rows):
        raise ValueError("two-shot metadata exceeds existing runtime capacity")
    return PcieQuery(
        surface=surface, world_size=int(runtime.world_size), rank=int(runtime.rank),
        topology="pcie_ipc", call=call_metadata, setup=setup,
    )


def _query(payload) -> PcieQuery:
    values = dict(payload)
    return PcieQuery(surface=values["surface"], world_size=values["world_size"], rank=values["rank"],
                     topology=values["topology"], call=FrozenMapping(values["call"]), setup=FrozenMapping(values["setup"]))


@program_cache(scope="preparation")
def compile_twoshot_surface(query_payload, ordinal: int):
    """Resolve every native slot specialization represented by the declaration."""
    query = _query(query_payload)
    call = query.call
    with torch.cuda.device(int(ordinal)):
        if query.surface in _FP8_SURFACES:
            from ._twoshot_cute import get_twoshot_launcher
            return {(bool(device_slot), int(bias)): get_twoshot_launcher(
                str(call["operation"]), query.world_size, query.rank, bool(device_slot), int(bias),
                int(call["threads"]), int(call["row_elems"]), int(ordinal))
                    for device_slot, bias in call["device_slot_variants"]}
        from ._twoshot_bf16_cute import get_twoshot_bf16_allreduce_launcher, get_twoshot_bf16_launcher
        getter = get_twoshot_bf16_allreduce_launcher if call["operation"] == "all_reduce" else get_twoshot_bf16_launcher
        return {(bool(device_slot), int(bias)): (
            getter(query.world_size, query.rank, bool(device_slot), int(bias), int(call["threads"]), int(call["row_elems"]), int(ordinal))
            if call["operation"] == "all_reduce" else getter(str(call["operation"]), query.world_size, query.rank, bool(device_slot), int(bias), int(call["threads"]), int(call["row_elems"]), int(ordinal))
        ) for device_slot, bias in call["device_slot_variants"]}


@dataclass(frozen=True)
class _TwoShotExecutionState:
    query: PcieQuery
    runtime: object
    launchers: Mapping[tuple[bool, int], object]

    def require_runtime(self, runtime) -> None:
        if runtime is not self.runtime:
            raise ValueError("two-shot plan belongs to another IPC runtime")
        if (runtime.rank, runtime.world_size, runtime.row_elems) != (self.query.rank, self.query.world_size, self.query.call["row_elems"]):
            raise ValueError("two-shot runtime ABI differs from preparation")

    def launcher(self, device_slot_selection: bool, slot_bias: int):
        try:
            return self.launchers[(bool(device_slot_selection), int(slot_bias) & 1)]
        except KeyError as exc:
            raise RuntimeError("two-shot prepared plan misses requested slot launcher") from exc

    def run(self, payload, scale, out, *, threads: int, block_limit: int):
        self.require_runtime(self.runtime)
        call = self.query.call
        if (tuple(payload.shape), tuple(payload.stride()), _alignment(payload)) != (
            tuple(call["shape"]), tuple(call["stride"]), call["alignment"],
        ):
            raise ValueError("two-shot payload metadata differs from preparation")
        if call["scale_shape"] is not None and (
            scale is None
            or (tuple(scale.shape), tuple(scale.stride()), _alignment(scale)) != (
                tuple(call["scale_shape"]), tuple(call["scale_stride"]), call["scale_alignment"],
            )
        ):
            raise ValueError("two-shot scale metadata differs from preparation")
        if call["output_shape"] is not None and (
            tuple(out.shape), tuple(out.stride()), _alignment(out)
        ) != (tuple(call["output_shape"]), tuple(call["output_stride"]), call["output_alignment"]):
            raise ValueError("two-shot output metadata differs from preparation")
        return self.runtime._launch_prepared(
            payload, scale, out, state=self, threads=threads, block_limit=block_limit,
        )

def prepared_call(state: object, *, payload, out, scale=None) -> PreparedCall:
    """Prime the actual collective against caller-owned source/output buffers."""
    if not isinstance(state, _TwoShotExecutionState):
        raise TypeError("two-shot prepared call requires materialized state")
    if out is None:
        raise ValueError("two-shot priming requires a caller-owned output")
    operation = state.query.call["operation"]
    if state.query.surface in _FP8_SURFACES:
        if scale is None:
            raise TypeError("FP8 two-shot priming requires scale")
        run_scale = scale
    else:
        if scale is not None:
            raise TypeError("BF16 two-shot priming does not accept scale")
        run_scale = None
    return PreparedCall(
        run=lambda: state.run(payload, run_scale, out,
                              threads=state.query.call["threads"],
                              block_limit=state.query.call["block_limit"]),
        output=out,
        capture_safe=False,
    )


def plan(query: PcieQuery, *, runtime, invocation: FrozenMapping = FrozenMapping(), override: PcieConfig | None = None) -> Plan:
    if not isinstance(query, PcieQuery) or query.surface not in _SURFACES:
        raise TypeError("two-shot plan requires a two-shot PcieQuery")
    if runtime is None or (runtime.rank, runtime.world_size) != (query.rank, query.world_size):
        raise ValueError("two-shot query does not match its existing IPC runtime")
    if invocation:
        raise ValueError("two-shot invocation semantics belong in PcieQuery")
    payload = TUNING.encode_query(query)
    def jobs(config, device):
        return (CompileJob.create("b12x.comm.pcie._twoshot_preparation:compile_twoshot_surface", payload, device.ordinal),)
    def memory(config, device):
        required = int(query.setup["slab_nbytes"])
        return MemoryRequirements(persistent=(PersistentMemory((current_plan(), "pcie_twoshot_channel"), required, required),))
    def materialize(selection, device):
        return _TwoShotExecutionState(query, runtime, MappingProxyType(load_programs(compile_twoshot_surface(payload, device.ordinal))))
    return Plan(contract=TUNING, query=query, invocation=FrozenMapping(invocation), override=override,
                _compile_jobs=jobs, _memory_requirements=memory, _materialize=materialize, _device=runtime.device)


__all__ = ["plan", "prepared_call", "query_from_runtime"]
