"""Prepared contiguous batched and packed-varlen attention declarations."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from b12x._lib.scratch import scratch_buffer_spec

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping, MemoryRequirements, PersistentMemory, Plan,
    current_plan, current_prepared_state,
)
from b12x.preparation.types import require_prepared

from ._tuning import TUNING, VarlenAttentionConfig, VarlenAttentionQuery


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")




def _batched_payload(query_payload, invocation):
    from b12x.attention._shared.contiguous import api as contiguous

    query = VarlenAttentionQuery(**dict(query_payload))
    values = dict(invocation)
    return query, (
        tuple(values["q_shape"]), tuple(values["k_shape"]), tuple(values["v_shape"]),
        getattr(torch, values["dtype"]),
        bool(values["causal"]), int(values["window_size_left"]),
        int(values["window_size_right"]), bool(values["has_attention_sink_bias"]),
    ), contiguous


def _varlen_payload(query_payload, invocation):
    from b12x.attention._shared.contiguous import api as contiguous

    query = VarlenAttentionQuery(**dict(query_payload))
    values = dict(invocation)
    return query, (
        tuple(values["q_shape"]), tuple(values["k_shape"]), tuple(values["v_shape"]),
        tuple(values["cu_seqlens_q_shape"]), tuple(values["cu_seqlens_k_shape"]),
        getattr(torch, values["dtype"]),
        bool(values["causal"]), int(values["window_size_left"]),
        int(values["window_size_right"]), bool(values["has_attention_sink_bias"]),
        int(values["max_seqlen_q"]), int(values["max_seqlen_k"]),
        bool(values.get("block_sparse", False)),
        bool(values.get("per_segment_tiles", False)),
    ), contiguous


@program_cache(scope="preparation")
def compile_batched_attention(query_payload, invocation, config_payload, ordinal):
    query, values, contiguous = _batched_payload(query_payload, invocation)
    config = VarlenAttentionConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    q_shape, k_shape, v_shape, dtype, causal, left, right, has_sink = values
    with torch.cuda.device(ordinal):
        return contiguous._compile_attention(
            q_shape, k_shape, v_shape, dtype, causal, left, right, has_sink,
            config.tile_m, config.tile_n,
        )


@program_cache(scope="preparation")
def compile_varlen_attention(query_payload, invocation, config_payload, ordinal):
    query, values, contiguous = _varlen_payload(query_payload, invocation)
    config = VarlenAttentionConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    (*shapes, dtype, causal, left, right, has_sink, max_q, max_k, block_sparse,
     per_segment_tiles) = values
    q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape = shapes
    with torch.cuda.device(ordinal):
        return contiguous._compile_varlen_attention(
            q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape, dtype, causal,
            left, right, has_sink, max_q, max_k, config.tile_m, config.tile_n,
            block_sparse=bool(block_sparse),
            num_q_tiles=int(invocation["num_q_tiles"]),
            total_blocks_cap=int(invocation["total_blocks_cap"]),
            per_segment_tiles=bool(per_segment_tiles),
        )


@dataclass(frozen=True)
class BatchedBinding:
    plan: Plan | None
    binding: object
    sink_source: torch.Tensor | None = None
    sink_storage: torch.Tensor | None = None


@dataclass(frozen=True)
class VarlenBinding:
    plan: Plan | None
    binding: object
    sink_source: torch.Tensor | None = None
    sink_storage: torch.Tensor | None = None
    block_indices: torch.Tensor | None = None
    block_offsets: torch.Tensor | None = None


@dataclass(frozen=True)
class _BatchedState:
    plan: object
    scratch_plan: object
    sink_storage: torch.Tensor | None

    def _sink(self, attention_sink_bias):
        from b12x.attention._shared.contiguous import api as contiguous

        sink = contiguous._validate_attention_sink_bias_metadata(
            attention_sink_bias, q_shape=self.plan.q_shape, device=self.plan.device,
        )
        if sink is None:
            if self.plan.has_attention_sink_bias:
                raise ValueError("attention_sink_bias is required by this prepared plan")
            return self.sink_storage
        if not self.plan.has_attention_sink_bias:
            raise ValueError("attention_sink_bias is not enabled for this prepared plan")
        if self.sink_storage is not None:
            self.sink_storage.copy_(sink)
            return self.sink_storage
        if sink.dtype != torch.float32 or not sink.is_contiguous():
            raise ValueError("prepared attention sink requires conversion storage")
        return sink

    def bind(self, *, plan: Plan | None = None, scratch, q, k, v,
             softmax_scale=None, attention_sink_bias=None):
        sink = self._sink(attention_sink_bias)
        binding = self.scratch_plan.bind(
            scratch=scratch, q=q, k=k, v=v, softmax_scale=softmax_scale,
            attention_sink_bias=sink if self.plan.has_attention_sink_bias else None,
        )
        if not self.plan.has_attention_sink_bias:
            binding = replace(binding, attention_sink_bias=sink)
        return BatchedBinding(
            plan, binding,
            attention_sink_bias if self.sink_storage is not None and attention_sink_bias is not None else None,
            self.sink_storage if attention_sink_bias is not None else None,
        )

    def run(self, binding):
        if binding.sink_source is not None:
            binding.sink_storage.copy_(binding.sink_source)
        from b12x.attention._shared.contiguous.api import b12x_attention_forward
        return b12x_attention_forward(binding=binding.binding)


@dataclass(frozen=True)
class _VarlenState:
    plan: object
    scratch_plan: object
    sink_storage: torch.Tensor | None

    def _sink(self, attention_sink_bias):
        from b12x.attention._shared.contiguous import api as contiguous

        sink = contiguous._validate_attention_sink_bias_metadata(
            attention_sink_bias, q_shape=self.plan.q_shape, device=self.plan.device,
        )
        if sink is None:
            if self.plan.has_attention_sink_bias:
                raise ValueError("attention_sink_bias is required by this prepared plan")
            return self.sink_storage
        if not self.plan.has_attention_sink_bias:
            raise ValueError("attention_sink_bias is not enabled for this prepared plan")
        if self.sink_storage is not None:
            self.sink_storage.copy_(sink)
            return self.sink_storage
        if sink.dtype != torch.float32 or not sink.is_contiguous():
            raise ValueError("prepared attention sink requires conversion storage")
        return sink

    def bind(self, *, plan: Plan | None = None, scratch, q, k, v, cu_seqlens_q,
             cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None,
             causal=None, window_size=None, attention_sink_bias=None,
             block_indices=None, block_offsets=None):
        if self.plan.block_sparse:
            for name, tensor, capacity in (
                ("block_indices", block_indices, self.plan.total_blocks_cap),
                ("block_offsets", block_offsets, self.plan.num_q_tiles + 1),
            ):
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Sparse attention requires {name}")
                if (
                    tensor.dtype != torch.int32 or tensor.device != self.plan.device
                    or tensor.ndim != 1 or not tensor.is_contiguous()
                    or tensor.numel() < capacity
                ):
                    raise ValueError(
                        f"{name} must be contiguous CUDA int32 with capacity {capacity}"
                    )
        sink = self._sink(attention_sink_bias)
        binding = self.scratch_plan.bind(
            scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal,
            window_size=window_size,
            attention_sink_bias=sink if self.plan.has_attention_sink_bias else None,
        )
        if not self.plan.has_attention_sink_bias:
            binding = replace(binding, attention_sink_bias=sink)
        return VarlenBinding(
            plan, binding,
            attention_sink_bias if self.sink_storage is not None and attention_sink_bias is not None else None,
            self.sink_storage if attention_sink_bias is not None else None,
            block_indices,
            block_offsets,
        )

    def run(self, binding):
        if binding.sink_source is not None:
            binding.sink_storage.copy_(binding.sink_source)
        from b12x.attention._shared.contiguous.api import b12x_varlen_attention_forward
        return b12x_varlen_attention_forward(
            binding=binding.binding,
            block_indices=getattr(binding, "block_indices", None),
            block_offsets=getattr(binding, "block_offsets", None),
        )


def _explicit_max_seqlen(value, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be an explicit host integer for prepared varlen attention")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    resolved = int(value)
    if resolved != value or resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return resolved


def _batched_invocation(q, k, v, *, causal, window_size, attention_sink_bias):
    from b12x.attention._shared.contiguous import api as contiguous

    q_shape, k_shape, v_shape, device, dtype = contiguous._validate_forward_inputs(q, k, v)
    sink = contiguous._validate_attention_sink_bias_metadata(
        attention_sink_bias, q_shape=q_shape, device=device,
    )
    left, right = contiguous._normalize_window_size(window_size)
    batch_dims, seqlen_q, q_heads, q_dim = contiguous._seq_dims(q_shape)
    _, seqlen_k, kv_heads, _ = contiguous._seq_dims(k_shape)
    query = VarlenAttentionQuery(
        variant="batched", dtype=_dtype_name(dtype), causal=bool(causal),
        batch_size=batch_dims[0] if batch_dims else 1, q_heads=q_heads, kv_heads=kv_heads,
        q_head_dim=q_dim, v_head_dim=v_shape[-1],
        query_rows=seqlen_q * (batch_dims[0] if batch_dims else 1),
        kv_rows=seqlen_k * (batch_dims[0] if batch_dims else 1),
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
    )
    return query, FrozenMapping({
        "q_shape": q_shape, "k_shape": k_shape, "v_shape": v_shape,
        "dtype": _dtype_name(dtype),
        "causal": bool(causal), "window_size_left": left, "window_size_right": right,
        "has_attention_sink_bias": sink is not None,
        "sink_requires_copy": contiguous._attention_sink_requires_copy(sink),
    })


def _varlen_invocation(q, k, v, cu_seqlens_q, cu_seqlens_k, *, max_seqlen_q, max_seqlen_k,
                       causal, window_size, attention_sink_bias, block_sparse=False,
                       per_segment_tiles=False, num_q_tiles=0, total_blocks_cap=0):
    from b12x.attention._shared.contiguous import api as contiguous

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape, device, dtype = contiguous._validate_varlen_inputs(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
    )
    sink = contiguous._validate_attention_sink_bias_metadata(
        attention_sink_bias, q_shape=q_shape, device=device,
    )
    max_q = _explicit_max_seqlen(max_seqlen_q, name="max_seqlen_q")
    max_k = _explicit_max_seqlen(max_seqlen_k, name="max_seqlen_k")
    left, right = contiguous._normalize_window_size(window_size)
    total_q, q_heads, q_dim = q_shape
    total_k, kv_heads, _ = k_shape
    query = VarlenAttentionQuery(
        variant="varlen", dtype=_dtype_name(dtype), causal=bool(causal), batch_size=cu_q_shape[0] - 1,
        q_heads=q_heads, kv_heads=kv_heads, q_head_dim=q_dim, v_head_dim=v_shape[-1],
        query_rows=total_q, kv_rows=total_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
        block_sparse=bool(block_sparse),
        per_segment_tiles=bool(per_segment_tiles),
    )
    return query, FrozenMapping({
        "q_shape": q_shape, "k_shape": k_shape, "v_shape": v_shape,
        "cu_seqlens_q_shape": cu_q_shape, "cu_seqlens_k_shape": cu_k_shape,
        "dtype": _dtype_name(dtype),
        "causal": bool(causal), "window_size_left": left, "window_size_right": right,
        "has_attention_sink_bias": sink is not None,
        "sink_requires_copy": contiguous._attention_sink_requires_copy(sink),
        "max_seqlen_q": max_q, "max_seqlen_k": max_k,
        "block_sparse": bool(block_sparse),
        "per_segment_tiles": bool(per_segment_tiles),
        "num_q_tiles": int(num_q_tiles),
        "total_blocks_cap": int(total_blocks_cap),
    })


def _sink_memory(invocation, device):
    values = dict(invocation)
    from b12x.preparation.types import _owned_tensor_nbytes
    if not values["has_attention_sink_bias"]:
        from b12x.attention._shared.contiguous.api import _ATTENTION_SINK_PLACEHOLDERS
        resident = _owned_tensor_nbytes((_ATTENTION_SINK_PLACEHOLDERS.get(device.ordinal),))
        return (PersistentMemory(
            ("contiguous_attention.sink_placeholder", device.ordinal), 4, resident,
        ),)
    if values["sink_requires_copy"]:
        state = current_prepared_state()
        resident = _owned_tensor_nbytes((state.sink_storage,)) if isinstance(state, (_BatchedState, _VarlenState)) else 0
        return (PersistentMemory(
            ("contiguous_attention.sink_conversion", current_plan(), device.ordinal),
            int(values["q_shape"][-2]) * 4, resident,
        ),)
    return ()


def _sink_storage(invocation, device):
    from b12x.attention._shared.contiguous import api as contiguous

    values = dict(invocation)
    if not values["has_attention_sink_bias"]:
        return contiguous._attention_sink_placeholder(device.ordinal)
    if values["sink_requires_copy"]:
        return torch.empty(
            (int(values["q_shape"][-2]),), dtype=torch.float32,
            device=torch.device("cuda", device.ordinal),
        )
    return None


def _batched_memory(invocation, device):
    from b12x.attention._shared.contiguous import api as contiguous

    values = dict(invocation)
    layout = contiguous._attention_scratch_layout(
        q_shape=tuple(values["q_shape"]), v_shape=tuple(values["v_shape"]),
        dtype=getattr(torch, values["dtype"]),
    )
    return MemoryRequirements(scratch=(scratch_buffer_spec(
        "contiguous_attention.scratch", nbytes=layout.nbytes,
        device=torch.device("cuda", device.ordinal),
    ),), persistent=_sink_memory(invocation, device))


def _varlen_memory(invocation, device):
    from b12x.attention._shared.contiguous import api as contiguous

    values = dict(invocation)
    layout = contiguous._attention_scratch_layout(
        q_shape=tuple(values["q_shape"]), v_shape=tuple(values["v_shape"]),
        dtype=getattr(torch, values["dtype"]),
    )
    return MemoryRequirements(scratch=(scratch_buffer_spec(
        "varlen_contiguous_attention.scratch", nbytes=layout.nbytes,
        device=torch.device("cuda", device.ordinal),
    ),), persistent=_sink_memory(invocation, device))


def plan_batched(q, k, v, *, causal=True, window_size=None, attention_sink_bias=None, override=None):
    query, invocation = _batched_invocation(
        q, k, v, causal=causal, window_size=window_size,
        attention_sink_bias=attention_sink_bias,
    )

    def materialize(selection, device):
        _, values, contiguous = _batched_payload(TUNING.encode_query(replace(query, exhaustive=False)), invocation)
        q_shape, k_shape, v_shape, dtype, selected_causal, left, right, has_sink = values
        concrete = contiguous._get_attention_plan(
            q_shape, k_shape, v_shape, device.ordinal, dtype, selected_causal, left, right,
            has_sink, selection.config.tile_m, selection.config.tile_n,
        )
        return _BatchedState(
            concrete, contiguous.plan_attention_scratch(concrete),
            _sink_storage(invocation, device),
        )

    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.attention.varlen._preparation:compile_batched_attention",
            TUNING.encode_query(replace(query, exhaustive=False)), invocation.to_dict(), config.to_dict(), device.ordinal,
        ),),
        _memory_requirements=lambda config, device: _batched_memory(invocation, device),
        _materialize=materialize, _device=q.device,
    )
def plan(q, k, v, cu_seqlens_q, cu_seqlens_k=None, *, max_seqlen_q, max_seqlen_k,
         causal=False, window_size=None, attention_sink_bias=None, override=None,
         block_sparse=False, num_q_tiles=0, total_blocks_cap=0,
         per_segment_tiles=False):
    """Prepare row and segment capacities for packed varlen attention.

    Sparse lists use the override's tile sizes, or the default configuration's
    sizes when no override is supplied. Tuning preserves this CSR geometry.
    Lists are authoritative; causal and local masks are not reapplied. Offsets
    must be monotonic, begin at zero, and stay within total_blocks_cap. Listed
    K block IDs must be valid for each segment using that list. Per-segment
    lists require every segment's starting Q offset to be tile_m-aligned.
    Callers producing mutable CSR tensors must preserve these value invariants
    across replay. Binding validates storage metadata without reading GPU values.

    Bindings may use fewer packed rows or segments than the planning tensors.
    Head dimensions, dtype, and contiguous layouts remain fixed. GPU cumulative
    lengths must describe the bound tensors; supplied maximum lengths must not
    exceed the prepared limits. Live shapes never select or compile a program.
    """
    query, invocation = _varlen_invocation(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k, causal=causal, window_size=window_size,
        attention_sink_bias=attention_sink_bias, block_sparse=block_sparse,
        per_segment_tiles=per_segment_tiles, num_q_tiles=num_q_tiles,
        total_blocks_cap=total_blocks_cap,
    )

    if block_sparse:
        from ._tuning import _default_config

        config = override if override is not None else _default_config(query, None)
        if not isinstance(config, VarlenAttentionConfig):
            config = VarlenAttentionConfig.from_config(FrozenMapping(config))
        query = replace(query, block_tile_m=config.tile_m, block_tile_n=config.tile_n)
        TUNING.validate_config(query, config, None)
        required_tiles = (
            (query.query_rows + config.tile_m - 1) // config.tile_m
            if per_segment_tiles else (query.max_seqlen_q + config.tile_m - 1) // config.tile_m
        )
        if type(num_q_tiles) is not int or num_q_tiles < max(1, required_tiles):
            raise ValueError(f"Sparse attention requires at least {max(1, required_tiles)} query tiles")
        if type(total_blocks_cap) is not int or total_blocks_cap < 0:
            raise ValueError("Sparse attention block capacity must be a nonnegative integer")

    def materialize(selection, device):
        _, values, contiguous = _varlen_payload(TUNING.encode_query(replace(query, exhaustive=False)), invocation)
        (*shapes, dtype, selected_causal, left, right, has_sink, max_q, max_k,
         block_sparse, per_segment_tiles) = values
        q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape = shapes
        concrete = contiguous._get_varlen_attention_plan(
            q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape, device.ordinal, dtype,
            selected_causal, left, right, has_sink, max_q, max_k,
            selection.config.tile_m, selection.config.tile_n,
            block_sparse=bool(block_sparse),
            num_q_tiles=int(num_q_tiles),
            total_blocks_cap=int(total_blocks_cap),
            per_segment_tiles=bool(per_segment_tiles),
        )
        return _VarlenState(
            concrete, contiguous.plan_varlen_attention_scratch(concrete),
            _sink_storage(invocation, device),
        )
    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.attention.varlen._preparation:compile_varlen_attention",
            TUNING.encode_query(replace(query, exhaustive=False)), invocation.to_dict(), config.to_dict(), device.ordinal,
        ),),
        _memory_requirements=lambda config, device: _varlen_memory(invocation, device),
        _materialize=materialize, _device=q.device,
    )


def bind_batched(plan, **kwargs):
    return require_prepared(plan, "attention.varlen", kwargs["q"].device).bind(plan=plan, **kwargs)


def bind(plan, **kwargs):
    return require_prepared(plan, "attention.varlen", kwargs["q"].device).bind(plan=plan, **kwargs)


def run_batched(binding):
    if not isinstance(binding, BatchedBinding):
        raise TypeError("run_batched requires a prepared BatchedBinding")
    return require_prepared(binding.plan, "attention.varlen", binding.binding.q.device).run(binding)


def run(binding):
    if not isinstance(binding, VarlenBinding):
        raise TypeError("run requires a prepared VarlenBinding")
    return require_prepared(binding.plan, "attention.varlen", binding.binding.q.device).run(binding)
