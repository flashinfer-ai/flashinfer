from __future__ import annotations

import torch
import triton
import triton.language as tl

_K5_GROUP = 64
_K5_LATENT_DIM = 512
_K5_ROPE_DIM = 64
_K5_BITS = 5
_K5_TILE_BYTES = 30_848
_K5_S_COL_OFFSET = 20_480
_K5_ZP_OFFSET = 21_504
_K5_S_ROW_OFFSET = 22_528
_K5_ROPE_OFFSET = 22_656
_GLM_FP8_RECORD_BYTES = 656
_GLM_FP8_SCALE_OFFSET = 512
_GLM_FP8_ROPE_OFFSET = 528
_FP8_E4M3_MAX = 448.0
_TL_K5_GROUP = tl.constexpr(_K5_GROUP)
_TL_K5_ROPE_DIM = tl.constexpr(_K5_ROPE_DIM)
_TL_K5_BITS = tl.constexpr(_K5_BITS)
_TL_K5_S_COL_OFFSET = tl.constexpr(_K5_S_COL_OFFSET)
_TL_K5_ZP_OFFSET = tl.constexpr(_K5_ZP_OFFSET)
_TL_K5_S_ROW_OFFSET = tl.constexpr(_K5_S_ROW_OFFSET)
_TL_K5_ROPE_OFFSET = tl.constexpr(_K5_ROPE_OFFSET)
_TL_GLM_FP8_SCALE_OFFSET = tl.constexpr(_GLM_FP8_SCALE_OFFSET)
_TL_GLM_FP8_ROPE_OFFSET = tl.constexpr(_GLM_FP8_ROPE_OFFSET)
_TL_FP8_E4M3_MAX = tl.constexpr(_FP8_E4M3_MAX)


@triton.jit
def _unpack_k5(payload, value_indices, mask):
    bit_positions = value_indices * _TL_K5_BITS
    byte_offsets = bit_positions // 8
    shifts = bit_positions % 8
    low = tl.load(payload + byte_offsets, mask=mask, other=0).to(tl.uint32)
    high = tl.load(payload + byte_offsets + 1, mask=mask, other=0).to(tl.uint32)
    return ((low | (high << 8)) >> shifts) & 31


@triton.jit
def _stage_k5_as_fp8_records_kernel(
    physical_slots_ptr,
    k5_cache_ptr,
    block_to_pool_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    cache_stride_block: tl.constexpr,
    latent_pool_stride_slot: tl.constexpr,
    latent_pool_stride_token: tl.constexpr,
    rope_pool_stride_slot: tl.constexpr,
    rope_pool_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    num_blocks,
    num_pool_slots,
):
    row = tl.program_id(0)
    physical_slot = tl.load(physical_slots_ptr + row)
    block = physical_slot // _TL_K5_GROUP
    token = physical_slot % _TL_K5_GROUP
    valid = (physical_slot >= 0) & (block >= 0) & (block < num_blocks)
    safe_block = tl.where(valid, block, 0)
    pool_slot = tl.load(
        block_to_pool_slot_ptr + safe_block,
        mask=valid,
        other=-1,
    )
    valid &= pool_slot < num_pool_slots

    if valid:
        exact = pool_slot >= 0
        body = ~exact
        safe_pool_slot = tl.maximum(pool_slot, 0)

        scale_groups = tl.arange(0, 4)
        group_dims = tl.arange(0, 128)
        dims = scale_groups[:, None] * 128 + group_dims[None, :]
        record = k5_cache_ptr + block * cache_stride_block
        indices = dims * _TL_K5_GROUP + token
        codes = _unpack_k5(record, indices, body).to(tl.float32)
        fp16_record = record.to(tl.pointer_type(tl.float16))
        s_col = tl.load(
            fp16_record + _TL_K5_S_COL_OFFSET // 2 + dims,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            fp16_record + _TL_K5_ZP_OFFSET // 2 + dims,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        s_row = tl.load(
            fp16_record + _TL_K5_S_ROW_OFFSET // 2 + token,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        body_latent = (codes * s_col + zero) * s_row
        exact_latent = tl.load(
            latent_pool_ptr
            + safe_pool_slot * latent_pool_stride_slot
            + token * latent_pool_stride_token
            + dims,
            mask=exact,
            other=0.0,
        ).to(tl.float32)
        latent = (
            tl.where(exact, exact_latent, body_latent).to(tl.bfloat16).to(tl.float32)
        )

        amax = tl.max(tl.abs(latent), axis=1)
        scales = tl.where(amax > 0.0, amax / _TL_FP8_E4M3_MAX, 1.0)
        quantized = tl.maximum(
            tl.minimum(latent / scales[:, None], _TL_FP8_E4M3_MAX),
            -_TL_FP8_E4M3_MAX,
        ).to(tl.float8e4nv)
        output_record = output_ptr + row * output_stride_row
        fp8_output = output_record.to(tl.pointer_type(tl.float8e4nv))
        tl.store(fp8_output + dims, quantized)
        fp32_output = output_record.to(tl.pointer_type(tl.float32))
        tl.store(
            fp32_output + _TL_GLM_FP8_SCALE_OFFSET // 4 + scale_groups,
            scales,
        )

        rope_dims = tl.arange(0, _TL_K5_ROPE_DIM)
        body_rope = tl.load(
            (record + _TL_K5_ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
            + token * _TL_K5_ROPE_DIM
            + rope_dims,
            mask=body,
            other=0.0,
        )
        exact_rope = tl.load(
            rope_pool_ptr
            + safe_pool_slot * rope_pool_stride_slot
            + token * rope_pool_stride_token
            + rope_dims,
            mask=exact,
            other=0.0,
        )
        rope = tl.where(exact, exact_rope, body_rope)
        tl.store(
            (output_record + _TL_GLM_FP8_ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
            + rope_dims,
            rope,
        )


def stage_k5_as_fp8_records(
    physical_slots: torch.Tensor,
    k5_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Stage KVarN5 MLA tiles as SparkInfer GLM FP8 cache records.

    Invalid physical slots leave the corresponding output rows unchanged.
    """
    if physical_slots.ndim != 1 or physical_slots.dtype != torch.int32:
        raise ValueError("physical_slots must be a flat int32 tensor")
    if not physical_slots.is_contiguous():
        raise ValueError("physical_slots must be contiguous")
    if k5_cache.dtype != torch.uint8 or k5_cache.ndim != 3:
        raise ValueError("k5_cache must be a rank-3 uint8 tensor")
    if not k5_cache.is_contiguous():
        raise ValueError("k5_cache must be contiguous")
    if k5_cache.stride(0) != _K5_TILE_BYTES:
        raise ValueError(
            f"KVarN5 cache block stride must be {_K5_TILE_BYTES} bytes, "
            f"got {k5_cache.stride(0)}"
        )
    if block_to_pool_slot.ndim != 1 or block_to_pool_slot.dtype != torch.int32:
        raise ValueError("block_to_pool_slot must be a flat int32 tensor")
    if not block_to_pool_slot.is_contiguous():
        raise ValueError("block_to_pool_slot must be contiguous")
    if block_to_pool_slot.numel() != k5_cache.shape[0]:
        raise ValueError("block_to_pool_slot must contain one entry per cache block")
    if latent_pool.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("latent_pool must use BF16 or float8_e4m3fn")
    if latent_pool.ndim != 3 or tuple(latent_pool.shape[1:]) != (
        _K5_GROUP,
        _K5_LATENT_DIM,
    ):
        raise ValueError("latent_pool must have shape (slots,64,512)")
    if not latent_pool.is_contiguous():
        raise ValueError("latent_pool must be contiguous")
    if rope_pool.dtype != torch.bfloat16 or rope_pool.ndim != 3:
        raise ValueError("rope_pool must be a rank-3 BF16 tensor")
    if tuple(rope_pool.shape[1:]) != (_K5_GROUP, _K5_ROPE_DIM):
        raise ValueError("rope_pool must have shape (slots,64,64)")
    if not rope_pool.is_contiguous():
        raise ValueError("rope_pool must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("latent_pool and rope_pool must have equal slot capacities")
    if output.dtype != torch.uint8 or output.ndim != 2:
        raise ValueError("output must be a rank-2 uint8 tensor")
    if output.shape != (physical_slots.numel(), _GLM_FP8_RECORD_BYTES):
        raise ValueError(
            f"output must have shape ({physical_slots.numel()},{_GLM_FP8_RECORD_BYTES})"
        )
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    tensors = (
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
    )
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all KVarN5 staging tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("KVarN5 staging tensors must be CUDA tensors")
    aligned_tensors = (
        (physical_slots, 4, "physical_slots"),
        (k5_cache, 2, "k5_cache"),
        (block_to_pool_slot, 4, "block_to_pool_slot"),
        (latent_pool, latent_pool.element_size(), "latent_pool"),
        (rope_pool, 2, "rope_pool"),
        (output, 16, "output"),
    )
    for tensor, alignment, name in aligned_tensors:
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    if physical_slots.numel() == 0:
        return

    _stage_k5_as_fp8_records_kernel[(physical_slots.numel(),)](
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
        k5_cache.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        output.stride(0),
        num_blocks=k5_cache.shape[0],
        num_pool_slots=latent_pool.shape[0],
        num_warps=4,
    )


__all__ = ["stage_k5_as_fp8_records"]
