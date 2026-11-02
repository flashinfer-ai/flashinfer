"""Native PLE hash and inline-dequantized table lookup preparation."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.sequence.ple_hash._preparation import (
    _CompilePointer,
    _HashState,
    _alignment,
    compile_hash,
    query_from_geometry,
    invocation_from_tensors as hash_invocation_from_tensors,
)
from b12x.sequence.ple_hash._tuning import PleHashQuery, TUNING as HASH_TUNING
from b12x.sequence.ple_hash.geometry import _GeometryInputs
from ._contracts import _bind, storage_layout
from ._tuning import PleEmbeddingConfig, PleEmbeddingQuery, TUNING


_LOOKUP_OPERANDS = ("weight", "weight_scale", "weight_scale_2", "out")


def invocation_from_tensors(
    *,
    weight=None,
    weight_scale=None,
    weight_scale_2=None,
    disk_table=None,
    out,
    **hash_tensors,
):
    if disk_table is not None:
        weight = disk_table.weight
        if disk_table.weight_scale is not None:
            weight_scale = disk_table.weight_scale
    invocation = dict(hash_invocation_from_tensors(out=out, **hash_tensors))
    alignments = list(invocation["input_alignments"])
    # The hash output is an internal aligned scratch view, not the BF16 output.
    alignments[8] = alignments[9]
    invocation["input_alignments"] = tuple(alignments)
    invocation["lookup_alignments"] = tuple(
        16 if value is None else _alignment(value)
        for value in (weight, weight_scale, weight_scale_2, out)
    )
    return FrozenMapping(invocation)


def _hash_query(query):
    return PleHashQuery(
        **{field.name: getattr(query, field.name) for field in fields(PleHashQuery)}
    )


def _lookup_constants(query):
    sizes, offsets, _, padded = query.geometry
    shard_size = padded // query.tp_size
    return (
        query.max_tokens,
        query.head_count,
        query.head_dim,
        query.embedding_dim,
        offsets[-1] + sizes[-1],
        query.tp_rank * shard_size,
        (query.tp_rank + 1) * shard_size,
        128,
        query.table_memory == "io_uring",
    )


@program_cache(scope="preparation")
def compile_embedding(query_payload, config_payload, ordinal):
    from . import _kernels as kernels

    query = PleEmbeddingQuery(**dict(query_payload))
    TUNING.validate_config(
        query, PleEmbeddingConfig.from_config(FrozenMapping(config_payload)), None
    )
    hash_query = _hash_query(query)
    hash_programs = compile_hash(
        HASH_TUNING.encode_query(hash_query), {"backend": "triton"}, ordinal
    )
    weight_dtype = {
        "bf16": torch.bfloat16,
        "fp8_e4m3_per_tensor": torch.float8_e4m3fn,
        "nvfp4_group16": torch.uint8,
    }[query.quant_mode]
    dtypes = (
        weight_dtype,
        torch.float8_e4m3fn if query.quant_mode == "nvfp4_group16" else torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    )
    pointers = tuple(
        _CompilePointer(dtype, alignment)
        for dtype, alignment in zip(dtypes, query.lookup_alignments)
    )
    ids = _CompilePointer(torch.int64, query.input_alignments[8])
    num_tokens = _CompilePointer(torch.int32, query.input_alignments[4])
    if query.quant_mode == "bf16":
        kernel, operands = (
            kernels._bf16_lookup_kernel,
            (pointers[0], ids, num_tokens, pointers[3]),
        )
    elif query.quant_mode == "fp8_e4m3_per_tensor":
        kernel, operands = (
            kernels._fp8_lookup_kernel,
            (pointers[0], pointers[1], ids, num_tokens, pointers[3]),
        )
    else:
        kernel, operands = (
            kernels._nvfp4_lookup_kernel,
            (*pointers[:3], ids, num_tokens, pointers[3]),
        )
    with torch.cuda.device(ordinal):
        lookup = kernel.warmup(
            *operands,
            *_lookup_constants(query),
            grid=(query.max_tokens, query.head_count, (query.head_dim + 127) // 128),
            num_warps=4,
        )
    gather = None
    if query.table_memory == "io_uring":
        from .._shared._gds import compile_gather

        gather = compile_gather(ordinal)
    return hash_programs, lookup, gather


@dataclass(frozen=True)
class _EmbeddingState:
    query: PleEmbeddingQuery
    layout: object
    hash_state: _HashState
    lookup: object
    lookup_constants: tuple
    grid_tail: tuple[int, int]
    disk_gather: object

    def _check_lookup(self, weight, weight_scale, weight_scale_2, out):
        values = (weight, weight_scale, weight_scale_2, out)
        required = (
            True,
            self.query.quant_mode != "bf16",
            self.query.quant_mode == "nvfp4_group16",
            True,
        )
        for name, tensor, alignment, present in zip(
            _LOOKUP_OPERANDS, values, self.query.lookup_alignments, required
        ):
            if (tensor is not None) != present:
                raise ValueError(f"PLE {name} presence differs from preparation")
            if tensor is not None and (
                tensor.device != self.layout.caps.device
                or _alignment(tensor) != alignment
            ):
                raise ValueError(
                    f"PLE {name} device/alignment differs from preparation"
                )

    def bind(self, **kwargs):
        binding = _bind(self.layout, _hash_state=self.hash_state, **kwargs)
        weight, scale = binding.weight, binding.weight_scale
        if binding.disk_table is not None:
            weight = binding.disk_table.weight
            if binding.disk_table.weight_scale is not None:
                scale = binding.disk_table.weight_scale
        self._check_lookup(weight, scale, binding.weight_scale_2, binding.out)
        return binding

    def run_lookup(
        self, weight, weight_scale, weight_scale_2, ids, num_tokens, out, *, token_count
    ):
        self._check_lookup(weight, weight_scale, weight_scale_2, out)
        q = self.query
        if not 0 <= token_count <= q.max_tokens:
            raise ValueError("PLE token count exceeds prepared capacity")
        if not token_count:
            return
        if q.quant_mode == "bf16":
            operands = (weight, ids, num_tokens, out)
        elif q.quant_mode == "fp8_e4m3_per_tensor":
            operands = (weight, weight_scale, ids, num_tokens, out)
        else:
            operands = (weight, weight_scale, weight_scale_2, ids, num_tokens, out)
        with torch.cuda.device(self.layout.caps.device):
            self.lookup[(token_count, *self.grid_tail)](
                *operands, *self.lookup_constants
            )

    def run_tensors(
        self,
        weight,
        weight_scale,
        weight_scale_2,
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        scratch,
        out,
        *,
        token_count,
    ):
        from ._kernels import _pipeline_scratch_views

        layout = self.layout
        hash_layout = layout._hash_layout.layout
        ids, request_ids = _pipeline_scratch_views(
            scratch,
            max_tokens=self.query.max_tokens,
            head_count=self.query.head_count,
            ids_offset_bytes=layout._layout.ids_offset_bytes,
            request_ids_offset_bytes=layout._layout.hash_scratch_offset_bytes
            + hash_layout.request_ids_offset_bytes,
        )
        self.hash_state.run_tensors(
            token_ids,
            query_start_loc,
            committed_history,
            num_seqs,
            num_tokens,
            multipliers,
            prime_sizes,
            table_offsets,
            ids,
            request_ids,
            token_count=token_count,
        )
        self.run_lookup(
            weight,
            weight_scale,
            weight_scale_2,
            ids,
            num_tokens,
            out,
            token_count=token_count,
        )

    def run(self, binding, *, token_count=None):
        if binding._state is not self.layout:
            raise ValueError("PLE binding belongs to another prepared plan")
        count = self.query.max_tokens if token_count is None else token_count
        if not 0 <= count <= self.query.max_tokens:
            raise ValueError("PLE token count exceeds prepared capacity")
        if binding.disk_table is not None:
            binding.disk_table._run(binding, state=self, token_count=count)
        else:
            geometry = binding._hash_binding.geometry
            self.run_tensors(
                binding.weight,
                binding.weight_scale,
                binding.weight_scale_2,
                binding.token_ids,
                binding.query_start_loc,
                binding.committed_history,
                binding.num_seqs,
                binding.num_tokens,
                geometry.multipliers,
                geometry.prime_sizes,
                geometry.table_offsets,
                binding.scratch,
                binding.out,
                token_count=count,
            )
        return binding.out[:count]


def make_plan(
    caps, *, geometry, prime_sizes, table_offsets, multipliers, invocation, override
):
    invocation = FrozenMapping(invocation)
    if set(invocation) - {"input_alignments", "lookup_alignments"}:
        raise ValueError("unknown PLE embedding invocation metadata")
    inputs = _GeometryInputs(
        caps,
        geometry=geometry,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        multipliers=multipliers,
    )
    layout = storage_layout(caps, geometry=inputs.geometry)
    hash_query = query_from_geometry(layout._hash_layout.caps, inputs, invocation)
    query = PleEmbeddingQuery(
        **HASH_TUNING.encode_query(hash_query),
        quant_mode=caps.quant_mode,
        table_memory=caps.table_memory,
        output_dtype="bfloat16",
        embedding_dim=caps.embedding_dim,
        tp_size=caps.tp_size,
        tp_rank=caps.tp_rank,
        lookup_alignments=invocation.get("lookup_alignments", (16, 16, 16, 16)),
    )

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.sequence.ple_embedding._preparation:compile_embedding",
                TUNING.encode_query(query),
                config.to_dict(),
                device.ordinal,
            ),
        )

    def memory(config, device):
        return MemoryRequirements(
            scratch=layout.scratch_specs(), persistent=(inputs.memory(caps.device),)
        )

    def materialize(selection, device):
        hash_programs, lookup, gather = compile_embedding(
            TUNING.encode_query(query), selection.config.to_dict(), device.ordinal
        )
        hash_state = _HashState(
            hash_query,
            layout._hash_layout,
            inputs.materialize(caps.device),
            hash_programs,
        )
        return _EmbeddingState(
            query,
            layout,
            hash_state,
            lookup,
            _lookup_constants(query),
            (query.head_count, (query.head_dim + 127) // 128),
            gather,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )
