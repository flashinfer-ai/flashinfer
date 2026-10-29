"""Prepared PLE hashing with separately owned checkpoint geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._contracts import Caps, _bind, _materialize_layout
from ._tuning import OPERANDS, PleHashConfig, PleHashQuery, TUNING
from .geometry import _GeometryInputs


_I64 = frozenset(
    (
        "token_ids",
        "committed_history",
        "multipliers",
        "prime_sizes",
        "table_offsets",
        "out",
    )
)


def _alignment(tensor):
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_tensors(
    *,
    token_ids,
    query_start_loc,
    committed_history,
    num_seqs,
    num_tokens,
    out,
    scratch=None,
    **unused,
):
    values = dict(
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=out,
    )
    alignments = {name: _alignment(value) for name, value in values.items()}
    if scratch is not None:
        if isinstance(scratch, Mapping):
            scratch = scratch["ple_hash"]
        elif isinstance(scratch, Sequence) and not isinstance(scratch, torch.Tensor):
            if len(scratch) != 1:
                raise ValueError("PLE hash uses one scratch buffer")
            scratch = scratch[0]
        alignments.update(request_ids=_alignment(scratch))
    return FrozenMapping(
        {"input_alignments": tuple(alignments.get(name, 16) for name in OPERANDS)}
    )


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype
    alignment: int

    def data_ptr(self):
        return self.alignment


@program_cache(scope="preparation")
def compile_hash(query_payload, config_payload, ordinal):
    from . import _kernels as kernels

    query = PleHashQuery(**dict(query_payload))
    TUNING.validate_config(
        query, PleHashConfig.from_config(FrozenMapping(config_payload)), None
    )
    p = {
        name: _CompilePointer(torch.int64 if name in _I64 else torch.int32, alignment)
        for name, alignment in zip(OPERANDS, query.input_alignments)
    }
    m = query.max_tokens
    with torch.cuda.device(ordinal):
        return (
            kernels._request_ids_kernel.warmup(
                p["query_start_loc"],
                p["num_seqs"],
                p["num_tokens"],
                p["request_ids"],
                MAX_TOKENS=m,
                num_warps=1,
                grid=(m,),
            ),
            kernels._hash_ids_kernel.warmup(
                p["token_ids"],
                p["query_start_loc"],
                p["committed_history"],
                p["num_tokens"],
                p["request_ids"],
                p["multipliers"],
                p["prime_sizes"],
                p["table_offsets"],
                p["out"],
                query.eos_token_id,
                MAX_TOKENS=m,
                MAX_ORDER=query.max_order,
                HEADS_PER_ORDER=query.heads_per_order,
                HEAD_COUNT=query.head_count,
                num_warps=1,
                grid=(m, query.head_count),
            ),
        )


def _binding_tensors(binding):
    geometry = binding.geometry
    return (
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        binding.num_seqs,
        binding.num_tokens,
        geometry.multipliers,
        geometry.prime_sizes,
        geometry.table_offsets,
        binding.out,
        binding.request_ids,
    )


@dataclass(frozen=True)
class _HashState:
    query: PleHashQuery
    layout: object
    geometry: object
    programs: tuple

    def _check(self, tensors):
        for name, tensor, alignment in zip(
            OPERANDS, tensors, self.query.input_alignments
        ):
            if tensor.device != self.layout.caps.device:
                raise ValueError(f"PLE hash {name} device differs from preparation")
            if tensor.numel() and _alignment(tensor) != alignment:
                raise ValueError(f"PLE hash {name} alignment differs from preparation")

    def bind(self, **kwargs):
        binding = _bind(self.layout, _geometry=self.geometry, **kwargs)
        self._check(_binding_tensors(binding))
        return binding

    def run(self, binding, *, token_count=None):
        if binding._state is not self.layout:
            raise ValueError("hash binding belongs to another prepared layout")
        self.run_tensors(*_binding_tensors(binding), token_count=token_count)
        return binding.out

    def run_tensors(
        self,
        token_ids,
        query_start_loc,
        committed_history,
        num_seqs,
        num_tokens,
        multipliers,
        prime_sizes,
        table_offsets,
        out,
        request_ids,
        *,
        token_count=None,
    ):
        self._check(
            (
                token_ids,
                query_start_loc,
                committed_history,
                num_seqs,
                num_tokens,
                multipliers,
                prime_sizes,
                table_offsets,
                out,
                request_ids,
            )
        )
        q, p = self.query, self.programs
        count = q.max_tokens if token_count is None else int(token_count)
        if not 0 <= count <= q.max_tokens:
            raise ValueError("hash launch bound exceeds prepared capacity")
        with torch.cuda.device(self.layout.caps.device):
            p[0][(count, 1, 1)](
                query_start_loc, num_seqs, num_tokens, request_ids, q.max_tokens
            )
            p[1][(count, q.head_count, 1)](
                token_ids,
                query_start_loc,
                committed_history,
                num_tokens,
                request_ids,
                multipliers,
                prime_sizes,
                table_offsets,
                out,
                q.eos_token_id,
                q.max_tokens,
                q.max_order,
                q.heads_per_order,
                q.head_count,
            )


def query_from_geometry(caps, inputs, invocation):
    alignments = list(invocation.get("input_alignments", (16,) * len(OPERANDS)))
    if len(alignments) != len(OPERANDS):
        raise ValueError("invalid PLE hash alignment coordinates")
    for index, source in zip((6, 7, 5), inputs.sources):
        alignments[index] = (
            _alignment(source)
            if source is not None
            and source.device == caps.device
            and source.is_contiguous()
            else 16
        )
    return PleHashQuery(
        max_tokens=caps.max_tokens,
        max_seqs=caps.max_seqs,
        vocab_size=caps.vocab_size,
        eos_token_id=caps.eos_token_id,
        max_order=caps.max_order,
        heads_per_order=caps.heads_per_order,
        base_table_size=caps.base_table_size,
        dense_layer_ordinal=caps.dense_layer_ordinal,
        table_alignment=caps.table_alignment,
        geometry=inputs.geometry.key(),
        input_alignments=tuple(alignments),
    )


def make_plan(
    caps: Caps,
    *,
    geometry,
    prime_sizes,
    table_offsets,
    multipliers,
    invocation,
    override,
):
    invocation = FrozenMapping(invocation)
    if set(invocation) - {"input_alignments"}:
        raise ValueError("unknown PLE hash invocation metadata")
    inputs = _GeometryInputs(
        caps,
        geometry=geometry,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        multipliers=multipliers,
    )
    query = query_from_geometry(caps, inputs, invocation)

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.sequence.ple_hash._preparation:compile_hash",
                TUNING.encode_query(query),
                config.to_dict(),
                device.ordinal,
            ),
        )

    def memory(config, device):
        return MemoryRequirements(
            scratch=_materialize_layout(caps, inputs.geometry, config).scratch_specs(),
            persistent=(inputs.memory(caps.device),),
        )

    def materialize(selection, device):
        return _HashState(
            query,
            _materialize_layout(caps, inputs.geometry, selection.config),
            inputs.materialize(caps.device),
            compile_hash(
                TUNING.encode_query(query), selection.config.to_dict(), device.ordinal
            ),
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
