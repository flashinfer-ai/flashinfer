"""Prepared launch ownership for the fixed PCIe DMA all-reduce channel."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
    current_plan,
)

from ._dma_kernels import DmaLaunchers, compile_launchers
from ._tuning import PcieConfig, PcieQuery, TUNING

_SURFACE = "DmaAllReduce.all_reduce"


def _require(call: FrozenMapping, *names: str) -> dict[str, object]:
    values = dict(call)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"DMA preparation call metadata misses {missing}")
    return values


def query_from_runtime(runtime, *, surface: str, call) -> PcieQuery:
    """Snapshot immutable DMA channel metadata without serializing IPC owners."""

    if surface != _SURFACE:
        raise ValueError(f"DMA preparation does not own PCIe surface {surface!r}")
    values = FrozenMapping(
        {
            "max_bytes": int(runtime.max_bytes),
            "wire_mode": str(runtime._fp8),
            "pieces_override": int(runtime._pieces_override),
            "a2a_chunks_override": int(runtime._a2a_chunks_override),
            **dict(call),
        }
    )
    _require(values, "max_bytes", "wire_mode", "pieces_override", "a2a_chunks_override")
    if int(values["max_bytes"]) != int(runtime.max_bytes):
        raise ValueError("DMA preparation max_bytes differs from its runtime channel")
    if str(values["wire_mode"]) != runtime._fp8:
        raise ValueError("DMA preparation wire mode differs from its runtime channel")
    if int(values["pieces_override"]) != runtime._pieces_override or int(
        values["a2a_chunks_override"]
    ) != runtime._a2a_chunks_override:
        raise ValueError("DMA preparation crossover metadata differs from its runtime channel")
    setup = FrozenMapping({
        "shard_capacity": int(runtime.shard_capacity),
        "flag_slots": 256,
        "flag_stride": 128,
        "steps": 2 * (int(runtime.world_size) - 1),
        "slab_nbytes": 256 * 128
        + 2 * (int(runtime.world_size) - 1) * int(runtime.shard_capacity),
        "stage_nbytes": 0 if runtime._fp8_stage is None else int(runtime._fp8_stage.numel()),
        "stage_stride": int(runtime._fp8_stage_stride),
    })
    return PcieQuery(
        surface=surface,
        world_size=int(runtime.world_size),
        rank=int(runtime.rank),
        topology="pcie_ipc",
        call=values,
        setup=setup,
    )


def query_from_metadata(
    runtime, *, shape: tuple[int, ...], dtype: torch.dtype,
    strides: tuple[int, ...] | None = None, alignment: int = 16,
) -> PcieQuery:
    """Snapshot a concrete DMA invocation without allocating an activation."""
    if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("DMA metadata requires a positive shape")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("DMA metadata dtype is unsupported")
    count = 1
    for extent in shape:
        count *= extent
    contiguous = []
    stride = 1
    for extent in reversed(shape):
        contiguous.append(stride)
        stride *= extent
    contiguous = tuple(reversed(contiguous))
    if strides is None:
        strides = contiguous
    if tuple(strides) != contiguous or alignment < 16 or alignment & (alignment - 1):
        raise ValueError("DMA metadata requires contiguous aligned input")
    if count % (int(runtime.world_size) * 8):
        raise ValueError("DMA metadata does not satisfy native shard alignment")
    if count * dtype.itemsize > int(runtime.max_bytes):
        raise ValueError("DMA metadata exceeds existing runtime capacity")
    return query_from_runtime(
        runtime,
        surface=_SURFACE,
        call=FrozenMapping({
            "shape": tuple(shape), "dtype": str(dtype).removeprefix("torch."),
            "strides": tuple(strides), "alignment": alignment,
        }),
    )


@program_cache(scope="preparation")
def compile_dma_surface(query_payload, ordinal: int) -> DmaLaunchers:
    """Compiler-pool factory using only query metadata and native factories."""

    import torch

    query = PcieQuery(**dict(query_payload))
    values = _require(query.call, "wire_mode")
    with torch.cuda.device(int(ordinal)):
        return compile_launchers(
            world_size=query.world_size, wire_mode=str(values["wire_mode"])
        )


@dataclass(frozen=True)
class _DmaExecutionState:
    query: PcieQuery
    runtime: object
    launchers: DmaLaunchers

    def require_runtime(self, runtime) -> None:
        if runtime is not self.runtime:
            raise ValueError("DMA plan belongs to a different native runtime channel")
        if (runtime.rank, runtime.world_size) != (self.query.rank, self.query.world_size):
            raise ValueError("DMA runtime rank or world size differs from preparation")

    def run(self, inp, *, out=None):
        return self.runtime._run_prepared(inp, out=out, state=self)

    def prime(self, inp, *, out) -> None:
        """Prime the actual caller-owned collective transport after loading code."""
        if self.runtime._fp8 and (
            inp.dtype is not torch.bfloat16
            or inp.numel() // self.runtime.world_size % 128
        ):
            raise ValueError(
                "compressed DMA preparation requires a BF16 priming input with "
                "128-element shard alignment"
            )
        self.runtime._prime_prepared(self)
        self.run(inp, out=out)

def prepared_call(state: object, *, inp, out) -> PreparedCall:
    """Build a priming call from actual channel input/output buffers."""

    if not isinstance(state, _DmaExecutionState):
        raise TypeError("DMA prepared call requires its materialized state")
    if out is None:
        raise ValueError("DMA collective priming requires an actual output buffer")
    return PreparedCall(run=lambda: state.prime(inp, out=out), capture_safe=False)


def plan(
    query: PcieQuery,
    *,
    runtime,
    invocation: FrozenMapping = FrozenMapping(),
    override: PcieConfig | None = None,
) -> Plan:
    """Declare the caller-owned DMA channel; no channel resource is created."""

    if not isinstance(query, PcieQuery) or query.surface != _SURFACE:
        raise TypeError("DMA plan requires a DmaAllReduce PcieQuery")
    if runtime is None:
        raise TypeError("DMA declarations require the existing IPC runtime channel")
    if invocation:
        raise ValueError("DMA invocation semantics belong in PcieQuery.call")
    if (runtime.rank, runtime.world_size) != (query.rank, query.world_size):
        raise ValueError("DMA query does not match its runtime channel")

    payload = TUNING.encode_query(query)

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.comm.pcie._dma_preparation:compile_dma_surface",
                payload,
                device.ordinal,
            ),
        )

    def memory(config, device):
        del config, device
        required = (
            int(query.setup["slab_nbytes"])
            + int(query.setup["stage_nbytes"])
            + 2 * int(query.setup["flag_slots"]) * 4
        )
        # The runtime owns its IPC slab/stage before declaration; preparation
        # retains and reports the exact pre-existing resident envelope.
        return MemoryRequirements(
            persistent=(
                PersistentMemory(
                    key=(current_plan(), "pcie_dma_channel"),
                    required_nbytes=required,
                    resident_nbytes=required,
                ),
            )
        )

    def materialize(selection, device):
        from b12x._lib.compile_plan import load_programs
        del selection
        launchers = load_programs(compile_dma_surface(payload, device.ordinal))
        runtime._kernels.install(launchers)
        return _DmaExecutionState(query, runtime, launchers)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=runtime.device,
    )


__all__ = ["plan", "prepared_call", "query_from_runtime"]
