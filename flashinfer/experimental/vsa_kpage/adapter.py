# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""FI plan/run adaptation for the cuDNN Frontend KPage forward kernels."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from flashinfer.utils import check_shape_dtype_device, get_compute_capability


@dataclass
class KPagePlan:
    layout: Any
    M: int
    N: int
    num_qo_heads: int
    num_kv_heads: int
    device: torch.device
    sm_scale: float


def prepare(
    *,
    backend,
    device,
    indptr,
    indices,
    block_mask,
    q2k_indices,
    q2k_num,
    kv_block_lens,
    q_block_lens,
    M,
    N,
    R,
    C,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    q_data_type,
    kv_data_type,
    o_data_type,
    mask,
    packed_mask,
    causal,
    use_fp16_qk_reduction,
    pos_encoding_mode,
    logits_soft_cap,
    sm_scale,
    kv_splits,
    use_clc,
    q_scale,
    k_scale,
    v_scale,
):
    """Validate before ordinary VSA's packed-GQA conversion, then build pages."""
    from packaging.version import Version
    import cutlass

    if Version(cutlass.__version__) < Version("4.6.2"):
        raise RuntimeError("KPage requires nvidia-cutlass-dsl >= 4.6.2")
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        raise ValueError("KPage requires SM100/SM103")
    block = 64 if backend == "vsa_sm100_blk64" else 128
    if block != R or block != C:
        raise ValueError(f"{backend} with KPage requires R == C == {block}")
    if min(M, N, num_qo_heads, num_kv_heads) <= 0:
        raise ValueError("KPage sequence lengths and head counts must be positive")
    if head_dim != 128:
        raise ValueError("KPage requires head_dim=128")
    if any(t != torch.bfloat16 for t in (q_data_type, kv_data_type, o_data_type)):
        raise ValueError("KPage requires BF16 q, k, v and output")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if block == 64:
        if num_qo_heads != num_kv_heads:
            raise ValueError("blk64 KPage supports MHA only")
        if M % 64 or N % 64:
            raise ValueError("blk64 KPage requires M and N to be multiples of 64")
        if q_block_lens is not None:
            raise ValueError("blk64 KPage does not support q_block_lens")
    if mask is not None or packed_mask is not None or causal:
        raise ValueError("KPage supports non-causal block-level sparsity only")
    if pos_encoding_mode != "NONE" or (
        logits_soft_cap is not None and logits_soft_cap != 0
    ):
        raise ValueError("KPage does not support position encoding or logits_soft_cap")
    if kv_splits not in (None, 1) or use_clc is not None:
        raise ValueError("KPage requires kv_splits=1 (or None) and use_clc=None")
    if any(s is not None for s in (q_scale, k_scale, v_scale)):
        raise ValueError("KPage does not support FP8 scale tensors")
    if use_fp16_qk_reduction:
        raise ValueError("KPage does not support use_fp16_qk_reduction")
    scale = 1 / math.sqrt(head_dim) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale):
        raise ValueError("KPage sm_scale must be finite")

    mb, nb = (M + block - 1) // block, (N + block - 1) // block

    def metadata(t, name, shape):
        if t.dtype != torch.int32 or tuple(t.shape) != shape:
            raise ValueError(f"{name} must be int32 with shape {shape}")
        if t.device != device:
            raise ValueError(f"{name} must be on {device}")
        return t.contiguous()

    # Do not deduplicate native index rows through a boolean mask: the source
    # validator rejects repeated active KV IDs rather than changing attention.
    if q2k_indices is not None:
        if block_mask is not None or indptr is not None or indices is not None:
            raise ValueError(
                "provide q2k_indices, block_mask, or BSR, not multiple sparse patterns"
            )
        # FI's public direct indices omit batch; also accept the source BSA's
        # explicit singleton batch dimension to ease migration.
        if q2k_indices.ndim == 3:
            q2k_indices = q2k_indices.unsqueeze(0)
        if q2k_num is not None and q2k_num.ndim == 2:
            q2k_num = q2k_num.unsqueeze(0)
        if q2k_indices.ndim != 4 or tuple(q2k_indices.shape[:3]) != (
            1,
            num_qo_heads,
            mb,
        ):
            raise ValueError(
                f"q2k_indices must have shape [1, {num_qo_heads}, {mb}, capacity]"
            )
        cap = q2k_indices.shape[-1]
        if cap < 1:
            raise ValueError("q2k_indices capacity must be positive")
        idx = metadata(q2k_indices, "q2k_indices", (1, num_qo_heads, mb, cap))
        counts = (
            torch.full((1, num_qo_heads, mb), cap, dtype=torch.int32, device=device)
            if q2k_num is None
            else metadata(q2k_num, "q2k_num", (1, num_qo_heads, mb))
        )
    else:
        if q2k_num is not None:
            raise ValueError("q2k_num requires q2k_indices")
        from flashinfer.sparse import _block_mask_to_vsa_index, _bsr_to_vsa_index

        if block_mask is not None:
            if indptr is not None or indices is not None:
                raise ValueError("provide block_mask or BSR, not both")
            if block_mask.dtype != torch.bool or block_mask.ndim != 3:
                raise ValueError("block_mask must be a rank-3 boolean tensor")
            if tuple(block_mask.shape) == (num_kv_heads, mb, nb):
                block_mask = block_mask.repeat_interleave(
                    num_qo_heads // num_kv_heads, dim=0
                )
            if tuple(block_mask.shape) != (num_qo_heads, mb, nb):
                raise ValueError(
                    f"block_mask must have shape [{num_qo_heads}, {mb}, {nb}] or [{num_kv_heads}, {mb}, {nb}]"
                )
            idx, counts = _block_mask_to_vsa_index(block_mask, device)
        else:
            if indptr is None or indices is None:
                raise ValueError(
                    "KPage requires q2k_indices, block_mask, or BSR indptr/indices"
                )
            if indptr.dtype not in (torch.int32, torch.int64) or indices.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("BSR indptr and indices must be integer tensors")
            ptr = indptr.cpu()
            if ptr.ndim != 1 or ptr.numel() != mb + 1 or indices.ndim != 1:
                raise ValueError("invalid BSR shapes")
            if (
                int(ptr[0]) != 0
                or int(ptr[-1]) != indices.numel()
                or bool((ptr[1:] < ptr[:-1]).any())
            ):
                raise ValueError("invalid BSR row pointers")
            if bool((ptr[1:] - ptr[:-1] > nb).any()):
                raise ValueError("BSR row has more entries than KV blocks")
            idx, counts = _bsr_to_vsa_index(
                indptr, indices, mb, nb, num_qo_heads, device
            )

    kv_sizes = (
        torch.full((nb,), block, dtype=torch.int32, device=device)
        if kv_block_lens is None
        else metadata(kv_block_lens, "kv_block_lens", (nb,))
    )
    q_sizes = (
        torch.full((mb,), block, dtype=torch.int32, device=device)
        if q_block_lens is None
        else metadata(q_block_lens, "q_block_lens", (mb,))
    )
    from .kpage._layout import prepare_layout

    with torch.cuda.device(device):
        layout = prepare_layout(
            idx,
            counts,
            kv_sizes,
            q_sizes,
            seqlen_q=M,
            seqlen_k=N,
            validate=True,
            allow_empty_block_nums=True,
            sparse_block_size=block,
        )
        # FI exposes forward only. Do not retain adjacency solely for a future
        # backward build after forward walk tables have been materialized.
        if layout._tables is not None:
            layout._tables.bwd_metadata = None
    return KPagePlan(layout, M, N, num_qo_heads, num_kv_heads, device, scale)


def run(plan, q, k, v, *, out=None, lse=None, return_lse=False):
    """NHD views into the source BHSD executor; write directly to FI buffers."""
    for name, tensor, shape in (
        ("q", q, (plan.M, plan.num_qo_heads, 128)),
        ("k", k, (plan.N, plan.num_kv_heads, 128)),
        ("v", v, (plan.N, plan.num_kv_heads, 128)),
    ):
        check_shape_dtype_device(tensor, shape, torch.bfloat16, plan.device, name)
        if not tensor.is_contiguous():
            raise ValueError(f"KPage requires compact NHD {name}")
    shape = (plan.M, plan.num_qo_heads, 128)
    if out is None:
        out = torch.empty(shape, dtype=q.dtype, device=plan.device)
    check_shape_dtype_device(out, shape, q.dtype, plan.device, "out")
    if not out.is_contiguous():
        raise ValueError("KPage requires contiguous out")
    for tensor in (q, k, v):
        if torch._C._overlaps(out, tensor):
            raise ValueError("KPage out must not overlap q, k or v")
    if lse is None:
        lse = torch.empty(
            (plan.M, plan.num_qo_heads), dtype=torch.float32, device=plan.device
        )
    check_shape_dtype_device(
        lse, (plan.M, plan.num_qo_heads), torch.float32, plan.device, "lse"
    )
    if not lse.is_contiguous():
        raise ValueError("KPage requires contiguous lse")
    if any(torch._C._overlaps(lse, t) for t in (q, k, v, out)):
        raise ValueError("KPage lse must not overlap inputs or out")
    from .kpage._interface import forward

    with torch.cuda.device(plan.device):
        forward(
            *(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
            plan.layout,
            plan.sm_scale,
            out=out.transpose(0, 1).unsqueeze(0),
            lse=lse.transpose(0, 1).unsqueeze(0),
        )
    return (out, lse) if return_lse else out
