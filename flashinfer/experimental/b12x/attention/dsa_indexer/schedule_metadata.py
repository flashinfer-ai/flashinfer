"""Paged-MQA schedule metadata helpers."""

from __future__ import annotations

import triton
import triton.language as tl
import torch


_MAX_TRITON_BATCH = 256
_MAX_TRITON_SMS = 512


@triton.jit
def _build_paged_mqa_schedule_triton(
    context_lens_ptr,
    schedule_ptr,
    batch_size,
    next_n,
    num_sms,
    context_lens_row_stride,
    schedule_row_stride,
    split_kv,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SMS: tl.constexpr,
):
    q_offsets = tl.arange(0, BLOCK_BATCH)
    if IS_CONTEXT_LENS_2D:
        lens_ptrs = (
            context_lens_ptr + q_offsets * context_lens_row_stride + (next_n - 1)
        )
    else:
        lens_ptrs = context_lens_ptr + q_offsets

    q_mask = q_offsets < batch_size
    context_lens = tl.load(lens_ptrs, mask=q_mask, other=0).to(tl.int32)
    context_lens = tl.maximum(context_lens, 0)
    num_segments = (context_lens + split_kv - 1) // split_kv
    prefix_sum = tl.cumsum(num_segments, axis=0)
    total_segments = tl.sum(num_segments, axis=0)

    sm_offsets = tl.arange(0, BLOCK_SMS)
    sm_mask = sm_offsets <= num_sms
    segments_per_sm = total_segments // num_sms
    segment_remainder = total_segments % num_sms
    segment_starts = sm_offsets * segments_per_sm + tl.minimum(
        sm_offsets, segment_remainder
    )

    completed_rows = (prefix_sum[None, :] <= segment_starts[:, None]) & q_mask[None, :]
    q_idx = tl.sum(completed_rows.to(tl.int32), axis=1)
    prev_q_idx = tl.maximum(q_idx - 1, 0)
    q_slots = tl.arange(0, BLOCK_BATCH)[None, :]
    prev_prefix_mask = q_mask[None, :] & (q_slots == prev_q_idx[:, None])
    prev_prefix_sum = tl.sum(tl.where(prev_prefix_mask, prefix_sum[None, :], 0), axis=1)
    prev_prefix_sum = tl.where(q_idx > 0, prev_prefix_sum, 0)
    kv_split_idx = segment_starts - prev_prefix_sum

    tl.store(schedule_ptr + sm_offsets * schedule_row_stride, q_idx, mask=sm_mask)
    tl.store(
        schedule_ptr + sm_offsets * schedule_row_stride + 1, kv_split_idx, mask=sm_mask
    )


def supports_triton_paged_mqa_schedule_metadata(
    context_lens: torch.Tensor,
    *,
    num_sms: int,
) -> bool:
    if context_lens.device.type != "cuda":
        return False
    batch_size = int(context_lens.shape[0]) if context_lens.ndim else 0
    if batch_size > _MAX_TRITON_BATCH:
        return False
    if num_sms <= 0 or num_sms > _MAX_TRITON_SMS:
        return False
    return True


def build_paged_mqa_schedule_metadata_torch(
    context_lens: torch.Tensor,
    *,
    block_kv: int,
    num_sms: int,
    pages_per_split: int,
    out: torch.Tensor,
) -> torch.Tensor:
    schedule_lens = context_lens[:, -1] if context_lens.ndim == 2 else context_lens
    schedule_lens = schedule_lens.to(torch.int64).clamp_min_(0)
    if schedule_lens.numel() == 0:
        out.zero_()
        return out

    split_kv = int(block_kv) * int(pages_per_split)
    num_segments = torch.div(
        schedule_lens + (split_kv - 1), split_kv, rounding_mode="floor"
    )
    prefix_sum = torch.cumsum(num_segments, dim=0, dtype=torch.int64)
    total_segments = prefix_sum[-1]

    sm_indices = torch.arange(
        num_sms + 1, dtype=torch.int64, device=context_lens.device
    )
    segments_per_sm = torch.div(total_segments, num_sms, rounding_mode="floor")
    segment_remainder = torch.remainder(total_segments, num_sms)
    segment_starts = sm_indices * segments_per_sm + torch.minimum(
        sm_indices,
        segment_remainder.expand_as(sm_indices),
    )
    q_idx = torch.searchsorted(prefix_sum, segment_starts, right=True)
    prev_q_idx = (q_idx - 1).clamp_min_(0)
    prev_prefix = prefix_sum.gather(0, prev_q_idx.clamp_max_(prefix_sum.shape[0] - 1))
    prev_prefix = torch.where(q_idx > 0, prev_prefix, torch.zeros_like(prev_prefix))
    kv_split_idx = segment_starts - prev_prefix

    out[:, 0].copy_(q_idx.to(torch.int32))
    out[:, 1].copy_(kv_split_idx.to(torch.int32))
    return out


def build_paged_mqa_schedule_metadata_triton(
    context_lens: torch.Tensor,
    *,
    block_kv: int,
    num_sms: int,
    pages_per_split: int,
    out: torch.Tensor,
) -> torch.Tensor:
    if not supports_triton_paged_mqa_schedule_metadata(context_lens, num_sms=num_sms):
        return build_paged_mqa_schedule_metadata_torch(
            context_lens,
            block_kv=block_kv,
            num_sms=num_sms,
            pages_per_split=pages_per_split,
            out=out,
        )

    batch_size = int(context_lens.shape[0])
    block_batch = max(triton.next_power_of_2(max(batch_size, 1)), 1)
    block_sms = max(triton.next_power_of_2(num_sms + 1), 1)
    num_warps = 8 if block_sms > 256 else 4 if block_sms > 128 else 2

    _build_paged_mqa_schedule_triton[(1,)](
        context_lens,
        out,
        batch_size,
        int(context_lens.shape[1]) if context_lens.ndim == 2 else 0,
        num_sms,
        int(context_lens.stride(0)) if context_lens.ndim == 2 else 0,
        int(out.stride(0)),
        int(block_kv) * int(pages_per_split),
        IS_CONTEXT_LENS_2D=context_lens.ndim == 2,
        BLOCK_BATCH=block_batch,
        BLOCK_SMS=block_sms,
        num_warps=num_warps,
    )
    return out
