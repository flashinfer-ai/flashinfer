# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side launch plumbing for the 1CTA token-sparse (DSA) MLA decode kernel.

The DSA kernel is the ``throughput_latency_1cta`` q64 keeps-MMA-AB FP8 schedule
run with one "batch" per query token: ``cache_seqs[t]`` is the token's active
top-k width and ``page_offsets[:, t]`` its row indices into a flat row pool
with ``page_size == 1``.  K/V tiles are fetched four rows per Gather4 TMA
through raw ``CUtensorMap``s encoded here: for the E4M3 cache a latent
128 B / SWIZZLE_128B and a RoPE 64 B / SWIZZLE_64B box over the ``[N, 576]``
pool; for the NVFP4 cache additionally a 64 B / SWIZZLE_64B box (the RoPE
pack, fed straight to the mxf4nvf4 MMA) and a 128 B / SWIZZLE_128B box (two
latent slices) over the ``[N, 320]`` data pool.
"""

from __future__ import annotations

import ctypes
import functools

import torch

HEAD_DIM = 576
LATENT_DIM = 512
ROPE_DIM = 64
V_DIM = 512
_DESCRIPTOR_BYTES = 128
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_PROFILE = "h64_keeps_mma_ab"
# Keyed by (device, data pointer, numel[, raw pool pointer, numel, row bytes]);
# never evicted: a captured CUDA graph keeps the device address of its
# descriptor tensor, and a key that is reused by a later buffer of the same
# address and size needs the same maps (128 B each).
_DESCRIPTOR_CACHE: dict[tuple, torch.Tensor] = {}


def _encode_pool_gather4_descriptor_host(
    pool: torch.Tensor, *, box_bytes: int, swizzle: str, row_bytes: int = HEAD_DIM
) -> torch.Tensor:
    """Encode a ``[row_bytes, rows(pool)]`` uint8 Gather4 map with a ``[box_bytes, 1]`` box.

    Row coordinates outside ``[0, rows)`` (``-1`` slots, stale indices) are
    zero-filled by the TMA unit.
    """

    from cuda.bindings import driver as cuda_drv

    swz = {
        "128B": cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        "64B": cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        "none": cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    }[swizzle]
    err, tensor_map = cuda_drv.cuTensorMapEncodeTiled(
        cuda_drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        pool.data_ptr(),
        [cuda_drv.cuuint64_t(row_bytes), cuda_drv.cuuint64_t(pool.shape[0])],
        [cuda_drv.cuuint64_t(row_bytes)],
        [cuda_drv.cuuint32_t(box_bytes), cuda_drv.cuuint32_t(1)],
        [cuda_drv.cuuint32_t(1), cuda_drv.cuuint32_t(1)],
        cuda_drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swz,
        cuda_drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda_drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled for DSA Gather4 failed: {err}")
    return torch.tensor(
        list(ctypes.string_at(tensor_map.getPtr(), _DESCRIPTOR_BYTES)),
        dtype=torch.uint8,
    )


def gather4_descriptors(
    kv_pool: torch.Tensor,
    raw_data_pool: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the cached device tensor ``[latent map | rope map (| NVFP4 rope map | NVFP4 latent map)]``.

    The NVFP4 maps cover the ``[N, 320]`` TRT-LLM data pool: the 64-byte
    SWIZZLE_64B box lands the RoPE pack as a K-major UMMA operand, the 128-byte
    SWIZZLE_128B box lands two latent slices per raw stage (the dequant's
    per-token 16-byte reads of 8 tokens x 128 B per warp stay bank-free).
    """

    key = (kv_pool.device.index or 0, kv_pool.data_ptr(), kv_pool.numel())
    if raw_data_pool is not None:
        key = key + (
            raw_data_pool.data_ptr(),
            raw_data_pool.numel(),
            raw_data_pool.shape[-1],
        )
    cached = _DESCRIPTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSA Gather4 descriptors for this KV pool were not built before CUDA "
            "graph capture; run the same pool tensor once eagerly first"
        )
    maps = [
        _encode_pool_gather4_descriptor_host(kv_pool, box_bytes=128, swizzle="128B"),
        _encode_pool_gather4_descriptor_host(kv_pool, box_bytes=64, swizzle="64B"),
    ]
    if raw_data_pool is not None:
        # Offset 256: the 64-byte RoPE box; 384: the 128-byte latent box.
        for box, swz in ((64, "64B"), (128, "128B")):
            maps.append(
                _encode_pool_gather4_descriptor_host(
                    raw_data_pool,
                    box_bytes=box,
                    swizzle=swz,
                    row_bytes=raw_data_pool.shape[-1],
                )
            )
    host = torch.cat(maps)
    device_tensor = host.to(device=kv_pool.device, non_blocking=False)
    if device_tensor.data_ptr() % 64:
        raise RuntimeError("DSA Gather4 descriptor tensor must be 64-byte aligned")
    _DESCRIPTOR_CACHE[key] = device_tensor
    return device_tensor


@functools.lru_cache(maxsize=None)
def max_active_clusters_1cta(device_index: int) -> int:
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    with torch.cuda.device(device_index):
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
        return int(
            cutlass_utils.HardwareInfo(device_index).get_max_active_clusters(1, stream)
        )


@functools.lru_cache(maxsize=None)
def compile_dsa_sparse_kernel(
    device_index: int,
    *,
    num_heads: int,
    sparse_mla_top_k: int,
    enable_skip_correction: bool,
    nvfp4: bool = False,
    residual_dim: int = 64,
):
    """Compile one token-sparse 1CTA specialization (JIT key = the argument list)."""

    import cutlass
    import cutlass.cute as cute

    from .kernel import ThroughputLatencyMlaDecodeTs

    # The q64 keeps-MMA-AB schedule tiles 64 flat query rows per CTA; H=128
    # runs as two physical "query positions" of 64 heads over the same token.
    phys_heads = 64
    phys_sq = num_heads // phys_heads
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=phys_heads,
        seq_len_q=phys_sq,
        seq_len_k=sparse_mla_top_k,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
        page_size=1,
        max_active_clusters=max_active_clusters_1cta(device_index),
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=_PROFILE,
        reduction_mode=None,
        logical_num_heads=num_heads,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=None,
        explicit_persistent=None,
        mask_type="dense",
        enable_skip_correction=enable_skip_correction,
        is_dynamic_token_sparse=True,
        is_dynamic_token_sparse_nvfp4=nvfp4,
        nvfp4_residual_dim=residual_dim,
    )
    cfg = kernel._make_config()
    if int(cfg.num_ctas_per_seq_kv) != 1:
        raise NotImplementedError("token-sparse 1CTA MLA does not support split-KV yet")

    total_q = cute.sym_int()
    pool_rows = cute.sym_int()
    table_columns = cute.sym_int()
    table_row_stride = cute.sym_int64(divisibility=1)
    e4m3 = cutlass.Float8E4M3FN
    rope_dim = ROPE_DIM
    head_dim = HEAD_DIM
    q_stride = (head_dim, 1, phys_heads * head_dim, num_heads * head_dim)
    q_latent_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (phys_heads, LATENT_DIM, phys_sq, total_q),
        stride=q_stride,
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (phys_heads, rope_dim, phys_sq, total_q),
        stride=q_stride,
        assumed_align=16,
    )
    c_latent_fake = cute.runtime.make_fake_tensor(
        e4m3,
        (1, LATENT_DIM, pool_rows),
        stride=(head_dim, 1, head_dim),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        e4m3, (1, rope_dim, pool_rows), stride=(head_dim, 1, head_dim), assumed_align=16
    )
    page_offsets_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (table_columns, total_q),
        stride=(1, table_row_stride),
        assumed_align=4,
    )
    out_stride = (V_DIM, 1, phys_heads * V_DIM, num_heads * V_DIM)
    out_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (phys_heads, V_DIM, phys_sq, total_q),
        stride=out_stride,
        assumed_align=16,
    )
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (phys_heads, phys_sq, total_q),
        stride=(1, phys_heads, num_heads),
        assumed_align=16,
    )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (total_q,), stride_order=(0,), assumed_align=4
    )
    descriptors_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        ((4 if nvfp4 else 2) * _DESCRIPTOR_BYTES,),
        stride_order=(0,),
        assumed_align=64,
    )
    scale_pool_fake = None
    if nvfp4:
        scale_bytes = cute.sym_int()
        scale_pool_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (scale_bytes,), stride_order=(0,), assumed_align=8
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        compiled = cute.compile[cute.FrontendNext](
            kernel,
            q_latent_fake,
            q_rope_fake,
            c_latent_fake,
            c_rope_fake,
            page_offsets_fake,
            out_fake,
            lse_fake,
            None,  # workspace
            cutlass.Int32(1),  # split_kv
            cache_seqs_fake,
            None,  # cu_seqlens_q
            None,  # block_split_kvs
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            cutlass.Float32(8.0 if enable_skip_correction else 0.0),
            descriptors_fake,
            scale_pool_fake,
            cutlass.Float32(1.0),
            stream_fake,
            options=_COMPILE_OPTIONS,
        )
    return compiled


def launch_dsa_sparse_kernel(
    compiled,
    *,
    query: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    top_k_lens: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    nvfp4_data_pool: torch.Tensor | None = None,
    nvfp4_scale_pool: torch.Tensor | None = None,
    nvfp4_dequant_scale: float = 1.0,
) -> None:
    """Bind ``[T, H, 576]`` packed inputs to the one-token-per-batch 1CTA ABI and launch."""

    total_q, num_heads, head_dim = query.shape
    phys_heads = 64
    phys_sq = num_heads // phys_heads
    q4 = query.view(total_q, phys_sq, phys_heads, head_dim)
    q_latent = q4[..., :LATENT_DIM].permute(2, 3, 1, 0)
    q_rope = q4[..., LATENT_DIM:].permute(2, 3, 1, 0)
    pool3 = kv_pool.view(-1, 1, head_dim)
    c_latent = pool3[..., :LATENT_DIM].permute(1, 2, 0)
    c_rope = pool3[..., LATENT_DIM:].permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)  # [Kmax, T], stride (1, Kmax)
    out_kernel = out.view(total_q, phys_sq, phys_heads, V_DIM).permute(2, 3, 1, 0)
    lse_kernel = lse.view(total_q, phys_sq, phys_heads).permute(2, 1, 0)
    descriptors = gather4_descriptors(kv_pool, nvfp4_data_pool)
    scale_pool_kernel = None if nvfp4_scale_pool is None else nvfp4_scale_pool.view(-1)
    compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_offsets,
        out_kernel,
        lse_kernel,
        None,
        1,
        top_k_lens,
        None,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        float(skip_corr_threshold),
        descriptors,
        scale_pool_kernel,
        float(nvfp4_dequant_scale),
    )
