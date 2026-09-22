"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""CPU planning for the experimental Hopper 64-token VSA backend."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseMetadata:
    indices: torch.Tensor
    counts: torch.Tensor
    order: torch.Tensor
    num_heads: int
    qo_len: int
    kv_len: int
    sm_scale: float
    schedule: str


def prepare_metadata(
    block_mask_map,
    block_row_sz,
    block_col_sz,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    *,
    causal=False,
    pos_encoding_mode="NONE",
    use_fp16_qk_reduction=False,
    logits_soft_cap=None,
    sm_scale=None,
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16,
):
    """Snapshot and validate descriptors once, before any attention launch.

    CUDA descriptors are copied synchronously to the CPU here. No caller-owned
    tensor is retained; later descriptor edits take effect only after replanning.
    """
    if num_qo_heads != num_kv_heads or num_qo_heads <= 0:
        raise ValueError("vsa_sm90_blk64 requires equal positive Q/KV head counts")
    if head_dim != 128:
        raise ValueError("vsa_sm90_blk64 requires head_dim=128")
    if q_data_type != torch.bfloat16 or kv_data_type != torch.bfloat16:
        raise ValueError("vsa_sm90_blk64 requires BF16 Q/K/V")
    if causal or pos_encoding_mode != "NONE" or use_fp16_qk_reduction:
        raise ValueError(
            "vsa_sm90_blk64 supports noncausal attention, no positional encoding, "
            "and FP32 QK accumulation"
        )
    if logits_soft_cap is not None and logits_soft_cap != 0:
        raise ValueError("vsa_sm90_blk64 does not support logits_soft_cap")
    scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else float(sm_scale)
    if (
        not math.isfinite(scale)
        or abs(scale * 1.4426950216293335) > torch.finfo(torch.float32).max
    ):
        raise ValueError("sm_scale must have a finite FP32 log2-scaled representation")
    if block_mask_map.dtype != torch.bool or block_mask_map.ndim != 3:
        raise ValueError("block_mask_map must be a boolean [H, MB, NB] tensor")
    h, mb, nb = block_mask_map.shape
    if h != num_kv_heads or mb <= 0 or nb <= 0:
        raise ValueError("block_mask_map must have matching heads and nonempty extents")
    for name, tensor, shape in (
        ("block_row_sz", block_row_sz, (h, mb)),
        ("block_col_sz", block_col_sz, (h, nb)),
    ):
        if tensor.shape != shape or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer tensor of shape {shape}")
        if not bool((tensor.to(device="cpu") == 64).all()):
            raise ValueError(
                "vsa_sm90_blk64 requires all row/column block sizes to be 64"
            )
    if max(mb, nb) > torch.iinfo(torch.int32).max // 64:
        raise ValueError("sequence lengths must fit the kernel's int32 shape ABI")

    mask = block_mask_map.to(device="cpu").contiguous()
    counts = mask.sum(dim=-1, dtype=torch.int32)
    if bool((counts == 0).any()):
        raise ValueError("vsa_sm90_blk64 does not support empty sparse rows")
    capacity = int(counts.max())
    # This backend supports at most 64 selected KV blocks.
    if capacity > 64:
        raise ValueError("vsa_sm90_blk64 supports at most 64 KV blocks per query block")
    indices = torch.argsort(~mask, dim=-1, stable=True)[..., :capacity]
    valid = torch.arange(capacity) < counts.unsqueeze(-1)
    indices = torch.where(valid, indices, -1).to(torch.int32).contiguous()

    blocks = h * mb
    if scale <= 1e-4:
        schedule = "general"
    elif capacity == 1:
        schedule = "single"
    elif blocks <= 64:
        schedule = "dsplit"
    elif capacity >= 12:
        schedule = "pipelined"
    else:
        schedule = "general"

    order = torch.empty((0,), dtype=torch.int64)
    if blocks > 264 and not bool((counts[:, :1] == counts).all()):
        linear = torch.arange(blocks, dtype=torch.int64).reshape(h, mb)
        if blocks <= 1056:
            order = linear.flatten()[
                torch.argsort(counts.flatten(), descending=True, stable=True)
            ]
        else:
            perm = torch.argsort(counts, dim=1, descending=True, stable=True)
            order = torch.gather(linear, 1, perm).flatten()
    return SparseMetadata(indices, counts, order, h, mb * 64, nb * 64, scale, schedule)
