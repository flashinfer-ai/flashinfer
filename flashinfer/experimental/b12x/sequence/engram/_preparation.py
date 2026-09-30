"""Prepared native Engram hashing, lookup compilation, and owned geometry."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x._lib.scratch_layout import SCRATCH_ALIGN_BYTES, align_up
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    current_plan,
    current_prepared_state,
)
from ._impl import Caps, _State
from b12x.preparation.types import _owned_tensor_nbytes
from ._tuning import EngramQuery, TUNING
from .geometry import Geometry, build_geometry


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype
    alignment: int = 16

    def data_ptr(self):
        return self.alignment


def _pointers():
    return {
        "ids": _CompilePointer(torch.int64), "mask": _CompilePointer(torch.bool),
        "token_map": _CompilePointer(torch.int64), "starts": _CompilePointer(torch.int32),
        "slots": _CompilePointer(torch.int32), "history": _CompilePointer(torch.int64),
        "num_seqs": _CompilePointer(torch.int32), "num_tokens": _CompilePointer(torch.int32),
        "compressed": _CompilePointer(torch.int64), "request_ids": _CompilePointer(torch.int32),
        "multipliers": _CompilePointer(torch.int64),
        "primes": _CompilePointer(torch.int64), "offsets": _CompilePointer(torch.int64),
        "hashes": _CompilePointer(torch.int64), "weight": _CompilePointer(torch.float8_e4m3fn),
        "scales": _CompilePointer(torch.uint8), "out": _CompilePointer(torch.bfloat16),
    }


@program_cache(scope="preparation")
def compile_engram(query_payload, config_payload, ordinal):
    """Compile only the native operation admitted by this declaration."""
    from . import _kernels as kernels
    from ..ple_hash import _kernels as hash_kernels

    query = EngramQuery(**dict(query_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, TUNING.decode_config(FrozenMapping(config_payload)), None)
    p = _pointers()
    t = query.max_tokens
    shard_rows = (query.table_rows + query.tp_size - 1) // query.tp_size
    shard_start = query.tp_rank * shard_rows
    shard_end = (query.tp_rank + 1) * shard_rows
    with torch.cuda.device(ordinal):
        if query.operation == "lookup":
            gather = ()
            if query.compact_rows:
                from .._shared._gds import compile_gather
                gather = (compile_gather(ordinal),)
            return (kernels._lookup.warmup(
                p["weight"], p["scales"], p["hashes"], p["num_tokens"], p["out"], t,
                T=t, ROWS=query.table_rows, START=shard_start, END=shard_end,
                COMPACT=query.compact_rows, RESIDENT_SCALES=query.resident_scales, num_warps=4, grid=(t, 24, 1),
            ), *gather)
        return (
            kernels._compress.warmup(
                p["ids"], p["mask"], p["token_map"], p["num_tokens"], p["compressed"],
                V=query.vocab_size, num_warps=1, grid=(t, 1, 1),
            ),
            hash_kernels._request_ids_kernel.warmup(
                p["starts"], p["num_seqs"], p["num_tokens"], p["request_ids"],
                MAX_TOKENS=t, num_warps=1, grid=(t, 1, 1),
            ),
            kernels._hash.warmup(
                p["compressed"], p["starts"], p["slots"], p["history"], p["num_tokens"],
                p["request_ids"], p["multipliers"], p["primes"], p["offsets"], p["hashes"],
                query.pad_id, num_warps=1, grid=(t, 24, 1),
            ),
        )


def make_plan(caps: Caps, *, token_map, geometry: Geometry | None = None,
              invocation=FrozenMapping(), override=None):
    unknown = set(invocation) - {"operation", "compact_rows", "resident_scales"}
    if unknown:
        raise ValueError(f"unknown Engram invocation fields: {sorted(unknown)!r}")
    operation = invocation.get("operation", "hash")
    if operation == "lookup" and "compact_rows" not in invocation:
        raise ValueError("lookup declarations require an explicit compact_rows coordinate")
    compact_rows = invocation.get("compact_rows", False)
    if operation not in ("hash", "lookup"):
        raise ValueError("Engram operation must be 'hash' or 'lookup'")
    if type(compact_rows) is not bool:
        raise TypeError("Engram compact_rows must be boolean")
    if operation == "hash" and compact_rows:
        raise ValueError("hash declarations cannot select lookup compaction")
    geometry = geometry or build_geometry()
    token_map = tuple(token_map)
    if caps.layer_id not in geometry.layer_ids:
        raise ValueError("layer_id is not present in hash geometry")
    if len(token_map) <= 2 or len(token_map) != caps.vocab_size or set(token_map) != set(range(geometry.compressed_vocab_size)):
        raise ValueError("token_map must cover the exact compressed vocabulary")
    expected = build_geometry(layer_ids=geometry.layer_ids, base_table_size=geometry.primes[0][0], compressed_vocab_size=geometry.compressed_vocab_size)
    if geometry != expected:
        raise ValueError("geometry must match PCG64 and globally unreused prime construction")
    index = geometry.layer_ids.index(caps.layer_id)
    rows = geometry.num_embeddings[index]
    shard_rows = (rows + caps.tp_size - 1) // caps.tp_size
    query = EngramQuery(max_tokens=caps.max_tokens, max_seqs=caps.max_seqs, max_requests=caps.max_requests,
                        vocab_size=caps.vocab_size, layer_id=caps.layer_id, tp_size=caps.tp_size,
                        tp_rank=caps.tp_rank, compressed_vocab_size=geometry.compressed_vocab_size,
                        table_rows=rows, operation=operation, compact_rows=compact_rows,
                        pad_id=token_map[2], resident_scales=invocation.get("resident_scales", False))
    def memory(config, device):
        del config, device
        scratch = ()
        persistent = ()
        if operation == "hash":
            scratch_bytes = align_up(caps.max_tokens * 8, SCRATCH_ALIGN_BYTES) + caps.max_tokens * 4
            geometry_bytes = (len(token_map) + len(geometry.multipliers[index]) + len(geometry.primes[index]) + len(geometry.offsets[index])) * 8
            scratch = (scratch_buffer_spec("engram", nbytes=scratch_bytes, device=caps.device),)
            state = current_prepared_state()
            resident = 0
            if isinstance(state, _State):
                geometry_bytes = resident = _owned_tensor_nbytes((
                    state.token_map, state.multipliers, state.primes, state.offsets,
                ))
            persistent = (PersistentMemory((current_plan(), "engram-geometry"), geometry_bytes, resident),)
        return MemoryRequirements(scratch=scratch, persistent=persistent)

    def materialize(selection, device):
        del device
        spec = memory(selection.config, None).scratch
        geometry_tensors = (
            torch.tensor(token_map, dtype=torch.int64, device=caps.device),
            torch.tensor(geometry.multipliers[index], dtype=torch.int64, device=caps.device),
            torch.tensor(geometry.primes[index], dtype=torch.int64, device=caps.device),
            torch.tensor(geometry.offsets[index], dtype=torch.int64, device=caps.device),
        ) if operation == "hash" else (None, None, None, None)
        return _State(
            caps, geometry if operation == "hash" else None, *geometry_tensors,
            query.pad_id, rows, caps.tp_rank * shard_rows, (caps.tp_rank + 1) * shard_rows,
            shard_rows, operation, compact_rows, spec,
            compile_engram(TUNING.encode_query(query), TUNING.encode_config(selection.config), caps.device.index),
            query.resident_scales,
        )

    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.sequence.engram._preparation:compile_engram", TUNING.encode_query(query),
            TUNING.encode_config(config), device.ordinal),),
        _memory_requirements=memory, _materialize=materialize, _device=caps.device,
    )
