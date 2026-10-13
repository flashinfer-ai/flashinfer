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

"""Host-side launch plumbing for the token-sparse (DeepSeek V4) 2CTA MLA kernel.

This module owns everything between validated public arguments and the
compiled CuTe DSL kernel: the raw Gather4 TMA descriptor pair and its LRU
cache, the compile specification (fake tensors and the positional ABI), the
fused RoPE/FP8 epilogue output layouts, and the per-launch argument assembly.
The public adapter in ``flashinfer/attention/prims_ts/dsv4.py`` validates
inputs, picks the scheduler, and calls into here.

``cutlass`` and the kernel/config modules are imported lazily so that
``flashinfer.attention.prims_ts`` stays importable without ``nvidia-cutlass-dsl``.
"""

from __future__ import annotations

import ctypes
import functools
from collections import OrderedDict
from typing import Optional

import torch


HEADS = 128
HEAD_DIM = 512
# Routing slots ``[0, SWA_TOPK)`` address the sliding-window pool.  The value
# is handed to ``MlaDecodeTs(sparse_swa_topk=...)``, whose config constructor
# checks it against the QK tile; it is spelled here because importing the
# config module pulls in ``cutlass`` at import time.
SWA_TOPK = 128
# The fused RoPE/FP8 epilogue receives O as one flat [T * 128 * 512] E4M3 tensor
# whose extent is a DSL Int32 dynamic shape; T * 65536 must stay below 2**31.
MAX_ROPE_QUANT_TOTAL_Q = (2**31 - 1) // (HEADS * HEAD_DIM)
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_RAW_GATHER4_DESCRIPTOR_BYTES = 128
# Encoded descriptor pairs are keyed by pool identity.  Serving reuses a
# handful of KV pools, so a small LRU bounds device memory when pools are
# reallocated.
_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE: OrderedDict[
    tuple[tuple[int, int, int], tuple[int, int, int]], torch.Tensor
] = OrderedDict()
_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE_CAPACITY = 16


@functools.lru_cache(maxsize=None)
def max_active_clusters_2cta(device_index: int) -> int:
    """Return the resident 2CTA cluster count of ``device_index``."""

    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    with torch.cuda.device(device_index):
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
        return int(
            cutlass_utils.HardwareInfo(device_index).get_max_active_clusters(2, stream)
        )


def _encode_raw_gather4_descriptor_host(pool: torch.Tensor) -> torch.Tensor:
    """Encode the token-sparse ``[512, INT_MAX]`` Gather4 map (one page slot per transaction) on the host."""

    from cuda.bindings import driver as cuda_drv

    err, tensor_map = cuda_drv.cuTensorMapEncodeTiled(
        cuda_drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        pool.data_ptr(),
        [
            cuda_drv.cuuint64_t(HEAD_DIM),
            cuda_drv.cuuint64_t((1 << 31) - 1),
        ],
        [cuda_drv.cuuint64_t(HEAD_DIM)],
        [cuda_drv.cuuint32_t(128), cuda_drv.cuuint32_t(1)],
        [cuda_drv.cuuint32_t(1), cuda_drv.cuuint32_t(1)],
        cuda_drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda_drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda_drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled for DSV4 Gather4 failed: {err}")

    return torch.tensor(
        list(ctypes.string_at(tensor_map.getPtr(), _RAW_GATHER4_DESCRIPTOR_BYTES)),
        dtype=torch.uint8,
    )


def gather4_descriptor_pair(
    sliding_window_kv_pool: torch.Tensor, compressed_kv_pool: torch.Tensor
) -> torch.Tensor:
    """Return the cached 256-byte device tensor holding the SWA and compressed Gather4 maps."""

    swa_key = (
        sliding_window_kv_pool.device.index or 0,
        sliding_window_kv_pool.data_ptr(),
        sliding_window_kv_pool.numel(),
    )
    compressed_key = (
        compressed_kv_pool.device.index or 0,
        compressed_kv_pool.data_ptr(),
        compressed_kv_pool.numel(),
    )
    key = (swa_key, compressed_key)
    cached = _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.get(key)
    if cached is not None:
        _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.move_to_end(key)
        return cached
    if torch.cuda.is_current_stream_capturing():
        # The miss path below performs a pageable H2D copy and may synchronize
        # the device on eviction; neither is legal under stream capture.
        raise RuntimeError(
            "DSV4 Gather4 descriptors for this KV pool pair were not built before "
            "CUDA graph capture; run the same pool tensors once eagerly first"
        )

    # Assemble the immutable 256-byte pair on the host and publish it with one
    # H2D copy; mutating a device-resident tensor map would need a
    # ``fence.proxy.tensormap`` before its first TMA use.
    host_descriptor_pair = torch.cat(
        (
            _encode_raw_gather4_descriptor_host(sliding_window_kv_pool),
            _encode_raw_gather4_descriptor_host(compressed_kv_pool),
        )
    )
    descriptor_pair = host_descriptor_pair.to(
        device=sliding_window_kv_pool.device, non_blocking=False
    )
    if descriptor_pair.data_ptr() % 64:
        raise RuntimeError("DSV4 Gather4 descriptor pair must be 64-byte aligned")
    if (
        len(_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE)
        >= _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE_CAPACITY
    ):
        # An evicted pair may still be read by an in-flight launch; drain the
        # device before releasing it.  Eviction only happens when a pool is
        # reallocated, so this synchronization is off the steady-state path.
        torch.cuda.synchronize(sliding_window_kv_pool.device)
        _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.popitem(last=False)
    _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE[key] = descriptor_pair
    return descriptor_pair


def rope_quant_output_shapes(
    total_q: int, *, ue8m0: bool
) -> tuple[tuple[int, int, int, int], tuple[int, int, int], torch.dtype]:
    """Return ``(out_shape, out_scale_shape, out_scale_dtype)`` of the fused RoPE/FP8 epilogue.

    ``out`` is the grouped physical E4M3 layout ``[16, T, 8, 512]``.  FP32
    scales are one value per (head, D128 block) in ``[16, 32, pad4(T)]``;
    UE8M0 scales are one INT32 word per head in ``[16, 8, pad4(T)]`` packing
    the four block exponent bytes.
    """

    if total_q > MAX_ROPE_QUANT_TOTAL_Q:
        raise ValueError(
            "the fused RoPE/FP8 output is addressed as a flat T*128*512 Int32 "
            f"extent; total packed queries must be <= {MAX_ROPE_QUANT_TOTAL_Q}, "
            f"got {total_q}"
        )
    head_groups = HEADS // 8
    padded_q = (total_q + 3) // 4 * 4
    out_shape = (head_groups, total_q, 8, HEAD_DIM)
    if ue8m0:
        return out_shape, (head_groups, 8, padded_q), torch.int32
    return out_shape, (head_groups, 8 * 4, padded_q), torch.float32


@functools.lru_cache(maxsize=None)
def compile_token_sparse_kernel(
    device_index: int,
    *,
    enable_skip_correction: bool,
    fuses_inv_rope_fp8_quant: bool,
    stores_lse: bool,
    is_persistent: bool,
    uses_ue8m0_scale_o: bool,
):
    """Compile one token-sparse MLA launch specialization.

    The JIT key is exactly the argument list; packed ``B/T/maxQ/Kmax`` are
    runtime fields and enter it only through ``is_persistent``.  A server
    whose ``B * max_seq_len_q`` crosses the resident-wave boundary therefore
    owns at most two variants and should warm both before graph capture.
    """

    import cutlass
    import cutlass.cute as cute

    from .kernel import MlaDecodeTs

    kernel = MlaDecodeTs(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=max_active_clusters_2cta(device_index),
        page_size=1,
        is_persistent=is_persistent,
        is_var_seq=True,
        is_var_split_kv=False,
        static_split_kv=1,
        static_seq_len_k=None,
        qkv_dtype="e4m3",
        out_dtype="e4m3" if fuses_inv_rope_fp8_quant else "bf16",
        rope_dim=0,
        num_heads=HEADS,
        # Hq=M=128 fixes the group ratio at one.  Runtime maxQ/B arrive
        # through the metadata/scalar ABI and only steer the scheduler.
        seq_len_q=1,
        batch_size=1,
        mask_type="causal",
        is_dynamic_token_sparse=True,
        sparse_swa_topk=SWA_TOPK,
        enable_skip_correction=enable_skip_correction,
        dsv4_fuses_inv_rope_fp8_quant=fuses_inv_rope_fp8_quant,
        dsv4_uses_ue8m0_scale_o=uses_ue8m0_scale_o,
        stores_lse=stores_lse,
    )
    compressed_rows = cute.sym_int()
    swa_rows = cute.sym_int()
    runtime_batch = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    runtime_total_q = cute.sym_int()
    runtime_sparse_capacity = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (HEADS, HEAD_DIM, runtime_total_q),
        stride=(HEAD_DIM, 1, HEADS * HEAD_DIM),
        assumed_align=16,
    )
    compressed_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, HEAD_DIM, compressed_rows),
        stride=(HEAD_DIM, 1, HEAD_DIM),
        assumed_align=16,
    )
    swa_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, HEAD_DIM, swa_rows),
        stride=(HEAD_DIM, 1, HEAD_DIM),
        assumed_align=16,
    )
    physical_indices_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_sparse_capacity, runtime_total_q),
        stride=(1, runtime_sparse_capacity),
        assumed_align=16,
    )
    if fuses_inv_rope_fp8_quant:
        # The physical output is [H/8, T, 8, D]; the epilogue owns the
        # non-affine head mapping, so the ABI sees one flat element span.
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (runtime_total_q * HEADS * HEAD_DIM,),
            stride_order=(0,),
            assumed_align=16,
        )
        runtime_cos_sin_elts = cute.sym_int()
        runtime_scale_elts = cute.sym_int()
        inv_rope_cos_sin_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (runtime_cos_sin_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
        # FP32 scales are one value per (head, D128 block); UE8M0 scales are
        # one INT32 word per head packing the four block exponent bytes.
        dsv4_o_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32 if uses_ue8m0_scale_o else cutlass.Float32,
            (runtime_scale_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
    else:
        out_fake = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (HEADS, HEAD_DIM, runtime_total_q),
            stride=(HEAD_DIM, 1, HEADS * HEAD_DIM),
            assumed_align=16,
        )
        inv_rope_cos_sin_fake = None
        dsv4_o_scale_fake = None
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (HEADS, runtime_total_q),
        stride=(1, HEADS),
        assumed_align=16,
    )
    raw_seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_batch,), stride_order=(0,), assumed_align=16
    )
    sparse_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_total_q,), stride_order=(0,), assumed_align=16
    )
    cu_seqlens_q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (runtime_num_q_offsets,),
        stride_order=(0,),
        assumed_align=4,
    )
    raw_tma_descriptor_pair_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (2 * _RAW_GATHER4_DESCRIPTOR_BYTES,),
        stride_order=(0,),
        assumed_align=64,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_specializations: tuple[object, ...] = (cute.FrontendNext,)
    with torch.cuda.device(device_index):
        return cute.compile[compile_specializations](
            kernel,
            q_fake,
            q_fake,
            compressed_cache_fake,
            compressed_cache_fake,
            physical_indices_fake,
            out_fake,
            lse_fake,
            None,
            raw_tma_descriptor_pair_fake,
            cutlass.Int32(1),
            raw_seq_lens_fake,
            sparse_lens_fake,
            cu_seqlens_q_fake,
            None,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            swa_cache_fake,
            cutlass.Int32(1),
            cutlass.Float32(8.0 if enable_skip_correction else 0.0),
            inv_rope_cos_sin_fake,
            dsv4_o_scale_fake,
            stream_fake,
            options=_COMPILE_OPTIONS,
        )


def launch_token_sparse_kernel(
    compiled,
    *,
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    max_seq_len_q: int,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> None:
    """Bind validated tensors to the positional ABI of ``compile_token_sparse_kernel`` and launch.

    ``out`` is the BF16 ``[T, 128, 512]`` output, or the grouped E4M3
    ``[16, T, 8, 512]`` output when ``inv_rope_cos_sin_cache`` and
    ``out_scale`` select the fused RoPE/FP8 epilogue.
    """

    fuses_inv_rope_fp8_quant = inv_rope_cos_sin_cache is not None
    if fuses_inv_rope_fp8_quant != (out_scale is not None):
        raise ValueError(
            "inv_rope_cos_sin_cache and out_scale must be given together for the "
            "fused RoPE/FP8 epilogue"
        )
    q_kernel = query.permute(1, 2, 0)
    compressed_kernel = compressed_kv_pool.view(-1, 1, HEAD_DIM).permute(1, 2, 0)
    swa_kernel = sliding_window_kv_pool.view(-1, 1, HEAD_DIM).permute(1, 2, 0)
    raw_tma_descriptor_pair = gather4_descriptor_pair(
        sliding_window_kv_pool, compressed_kv_pool
    )
    if fuses_inv_rope_fp8_quant:
        out_kernel = out.view(-1)
        inv_rope_kernel = inv_rope_cos_sin_cache.view(-1)
        out_scale_kernel = out_scale.view(-1)
    else:
        out_kernel = out.permute(1, 2, 0)
        inv_rope_kernel = None
        out_scale_kernel = None
    compiled(
        q_kernel,
        q_kernel,
        compressed_kernel,
        compressed_kernel,
        sparse_indices.transpose(0, 1),
        out_kernel,
        lse.transpose(0, 1),
        None,
        raw_tma_descriptor_pair,
        1,
        seq_lens,
        sparse_topk_lens,
        cum_seq_lens_q,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        swa_kernel,
        max_seq_len_q,
        float(skip_corr_threshold),
        inv_rope_kernel,
        out_scale_kernel,
    )
