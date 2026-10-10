"""Prepared GLM C4 pool production and FP8 index-query rotation.

The caller owns paged index records and per-sequence BF16 key/gate tails.
Main-token slots address model pages of 256 or more tokens; each page maps to
one or more 64-entry pooled-index pages. Device row boundaries and negative
state-slot IDs admit padded requests without host-side decisions.
"""

from dataclasses import dataclass

import torch
import triton

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)
from b12x.preparation.types import plan_from_handle, require_prepared

from . import _glm_pool_kernels as kernels


@dataclass(frozen=True, kw_only=True)
class PoolQuery:
    mode: str
    rows: int
    requests: int = 0
    tail_capacity: int = 4
    model_block_size: int = 256
    parent_stride_pages: int = 1
    strides: tuple[int, ...] = ()


_CONTRACT = make_fixed_contract(
    component_id="attention.dsa_indexer.glm_pool",
    query_type=PoolQuery,
    backend="triton",
)


@dataclass(frozen=True)
class _Pointer:
    dtype: torch.dtype

    def data_ptr(self):
        return 16


def _meta(query):
    return (64, 128, 4, query.tail_capacity, 448.0, 128)


@program_cache(scope="preparation")
def compile_pool(payload, ordinal):
    query = PoolQuery(**dict(payload))
    bf16 = _Pointer(torch.bfloat16)
    fp8 = _Pointer(torch.float8_e4m3fn)
    fp32 = _Pointer(torch.float32)
    i32, i64 = _Pointer(torch.int32), _Pointer(torch.int64)
    with torch.cuda.device(ordinal):
        if query.mode == "quantize":
            return (
                kernels._fwht_quant_kernel.warmup(
                    bf16,
                    fp8,
                    fp32,
                    fp32,
                    query.rows,
                    128,
                    448.0,
                    4,
                    True,
                    1 / 64,
                    num_warps=1,
                    grid=(triton.cdiv(query.rows, 4),),
                ),
            )
        pointers = (fp8, fp32, bf16, i64, i32, bf16, bf16, bf16, i64, i64)
        page_stride, *strides = query.strides
        scalar = (
            page_stride,
            query.model_block_size,
            query.parent_stride_pages,
            *strides,
        )
        if query.mode == "decode":
            return (
                kernels._decode_update_kernel.warmup(
                    *pointers,
                    *scalar,
                    *_meta(query),
                    True,
                    num_warps=4,
                    grid=(query.requests,),
                ),
            )
        pool = kernels._prefill_pool_kernel.warmup(
            *pointers,
            0,
            *scalar,
            *_meta(query),
            num_warps=4,
            grid=(query.requests, triton.cdiv(query.rows, 4)),
        )
        tail = kernels._prefill_tail_kernel.warmup(
            bf16,
            i64,
            i32,
            bf16,
            bf16,
            i64,
            0,
            *strides[:5],
            128,
            query.tail_capacity,
            128,
            num_warps=4,
            grid=(query.requests,),
        )
        return pool, tail


def _check_tensors(values, device):
    if any(value.device != device or value.data_ptr() % 16 for value in values):
        raise ValueError(
            "GLM pool operands must share a device and have 16-byte-aligned addresses"
        )


def _validate_quant(query, output, scales, weights):
    if query.ndim != 2 or query.shape[1] != 128 or query.dtype != torch.bfloat16:
        raise ValueError("GLM index queries require BF16 [rows,128]")
    if output.shape != query.shape or output.dtype != torch.float8_e4m3fn:
        raise ValueError("GLM rotated queries require matching FP8 output")
    if any(
        t.shape != (query.shape[0],) or t.dtype != torch.float32
        for t in (scales, weights)
    ):
        raise ValueError("GLM query scales and head weights require FP32 [rows]")
    values = (query, output, scales, weights)
    if any(not value.is_contiguous() for value in values) or query.shape[0] <= 0:
        raise ValueError("GLM query quantization requires nonempty contiguous operands")
    _check_tensors(values, query.device)


def _pool_strides(cache, tail, key, gate, ape):
    return (
        cache.stride(0),
        *tail.stride()[:3],
        key.stride(0),
        gate.stride(0),
        ape.stride(0),
    )


def _validate_pool(cache, tail, state_slots, starts, key, gate, ape, slots, positions):
    if cache.ndim != 3 or cache.shape[1:] != (64, 132) or cache.dtype != torch.uint8:
        raise ValueError("GLM index cache requires uint8 [pages,64,132]")
    if cache.stride()[1:] != (132, 1) or cache.stride(0) % 16:
        raise ValueError(
            "GLM index-cache pages require contiguous records and a 16-byte-aligned stride"
        )
    if (
        tail.ndim != 4
        or tail.shape[1] != 2
        or tail.shape[3] != 128
        or tail.shape[2] < 4
        or tail.shape[2] % 4
        or tail.stride(-1) != 1
        or tail.dtype != torch.bfloat16
    ):
        raise ValueError(
            "GLM tails require BF16 [slots,2,ring,128] with ring divisible by four"
        )
    if key.ndim != 2 or key.shape[1] != 128 or key.shape[0] <= 0:
        raise ValueError("GLM pool keys require [rows,128]")
    if (
        key.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or gate.shape != key.shape
    ):
        raise ValueError("GLM pool keys and gates require matching BF16 shapes")
    if key.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("GLM pool features must be contiguous within each row")
    if ape.shape != (4, 128) or ape.dtype != torch.bfloat16 or ape.stride(-1) != 1:
        raise ValueError("GLM pool position bias requires BF16 [4,128]")
    if (
        state_slots.ndim != 1
        or state_slots.dtype != torch.int64
        or state_slots.numel() <= 0
        or starts.shape != (state_slots.numel() + 1,)
        or starts.dtype != torch.int32
    ):
        raise ValueError(
            "GLM pool requests require int64 state slots and int32 row boundaries"
        )
    if any(
        t.shape != (key.shape[0],) or t.dtype != torch.int64 for t in (slots, positions)
    ):
        raise ValueError("GLM token slots and positions require int64 [rows]")
    if any(not t.is_contiguous() for t in (state_slots, starts, slots, positions)):
        raise ValueError("GLM pool metadata must be contiguous")
    _check_tensors(
        (cache, tail, state_slots, starts, key, gate, ape, slots, positions), key.device
    )


@dataclass(frozen=True)
class _State:
    query: PoolQuery
    programs: tuple

    def quantize(self, source, output, scales, weights):
        _validate_quant(source, output, scales, weights)
        if self.query.mode != "quantize" or source.shape[0] != self.query.rows:
            raise ValueError("GLM query shape differs from preparation")
        self.programs[0][(triton.cdiv(self.query.rows, 4), 1, 1)](
            source,
            output,
            scales,
            weights,
            self.query.rows,
            128,
            448.0,
            4,
            True,
            1 / 64,
        )

    def update(
        self, cache, tail, state_slots, starts, key, gate, ape, slots, positions
    ):
        _validate_pool(
            cache, tail, state_slots, starts, key, gate, ape, slots, positions
        )
        q = self.query
        if (
            q.mode not in ("decode", "prefill")
            or key.shape[0] != q.rows
            or state_slots.numel() != q.requests
            or tail.shape[2] != q.tail_capacity
            or _pool_strides(cache, tail, key, gate, ape) != q.strides
        ):
            raise ValueError("GLM pool shape or strides differ from preparation")
        pointers = (
            cache.view(torch.float8_e4m3fn),
            cache.view(torch.float32),
            tail,
            state_slots,
            starts,
            key,
            gate,
            ape,
            slots,
            positions,
        )
        page_stride, *strides = q.strides
        scalar = (page_stride, q.model_block_size, q.parent_stride_pages, *strides)
        if q.mode == "decode":
            self.programs[0][(q.requests, 1, 1)](*pointers, *scalar, *_meta(q), True)
        else:
            self.programs[0][(q.requests, triton.cdiv(q.rows, 4), 1)](
                *pointers, 0, *scalar, *_meta(q)
            )
            self.programs[1][(q.requests, 1, 1)](
                tail,
                state_slots,
                starts,
                key,
                gate,
                positions,
                0,
                *strides[:5],
                128,
                q.tail_capacity,
                128,
            )


def _plan(query, device):
    payload = _CONTRACT.encode_query(query)
    return Plan(
        contract=_CONTRACT,
        query=query,
        invocation=FrozenMapping(),
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.attention.dsa_indexer.glm_pool:compile_pool",
                payload,
                detected.ordinal,
            ),
        ),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=lambda selection, detected: _State(
            query, compile_pool(payload, detected.ordinal)
        ),
        _device=device,
    )


def plan_queries(source, output, scales, weights):
    """Declare rotated FP8 query rows and in-place scale-adjusted head weights."""
    _validate_quant(source, output, scales, weights)
    return _plan(PoolQuery(mode="quantize", rows=source.shape[0]), source.device)


def plan_pools(
    cache,
    tail,
    state_slots,
    starts,
    key,
    gate,
    ape,
    slots,
    positions,
    *,
    mode,
    model_block_size=256,
    parent_stride_pages=1,
):
    """Declare ordered decode or parallel prefill pool production."""
    _validate_pool(cache, tail, state_slots, starts, key, gate, ape, slots, positions)
    if mode not in ("decode", "prefill"):
        raise ValueError("GLM pool mode must be decode or prefill")
    if (
        type(model_block_size) is not int
        or model_block_size <= 0
        or model_block_size % 256
        or type(parent_stride_pages) is not int
        or parent_stride_pages < model_block_size // 256
    ):
        raise ValueError(
            "GLM model pages require whole C4 pages within the parent stride"
        )
    return _plan(
        PoolQuery(
            mode=mode,
            rows=key.shape[0],
            requests=state_slots.numel(),
            tail_capacity=tail.shape[2],
            model_block_size=model_block_size,
            parent_stride_pages=parent_stride_pages,
            strides=_pool_strides(cache, tail, key, gate, ape),
        ),
        key.device,
    )


@torch.library.custom_op(
    "b12x::glm_index_query", mutates_args=("output", "scales", "weights")
)
def _quantize(
    source: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    handle: int,
) -> None:
    require_prepared(
        plan_from_handle(handle), _CONTRACT.component_id, source.device
    ).quantize(source, output, scales, weights)


@_quantize.register_fake
def _fake_quantize(source, output, scales, weights, handle):
    pass


@torch.library.custom_op("b12x::glm_index_pool", mutates_args=("cache", "tail"))
def _update(
    cache: torch.Tensor,
    tail: torch.Tensor,
    state_slots: torch.Tensor,
    starts: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    handle: int,
) -> None:
    require_prepared(
        plan_from_handle(handle), _CONTRACT.component_id, key.device
    ).update(cache, tail, state_slots, starts, key, gate, ape, slots, positions)


@_update.register_fake
def _fake_update(
    cache, tail, state_slots, starts, key, gate, ape, slots, positions, handle
):
    pass


def quantize_queries(source, output, scales, weights, *, plan):
    """Rotate/quantize query rows; multiply weights by scales and 1/sqrt(128*32)."""
    _quantize(source, output, scales, weights, plan.handle)


def update_pools(
    cache, tail, state_slots, starts, key, gate, ape, slots, positions, *, plan
):
    """Update sequence tails and write completed pools without allocating storage."""
    _update(
        cache, tail, state_slots, starts, key, gate, ape, slots, positions, plan.handle
    )
