/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/sparse_pre_indexer.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

// A power of two's exponent, or -1 for anything else: the kernel divides by
// these three, and a shift is the only form of that division worth having.
int32_t shift_of(int64_t v) {
  if (v <= 0 || (v & (v - 1)) != 0) return -1;
  int32_t sh = 0;
  while ((int64_t{1} << sh) < v) ++sh;
  return sh;
}

}  // namespace

void qsa_pre_indexer(TensorView q, TensorView k, TensorView positions, TensorView cos_sin_cache,
                     TensorView q_norm_weight, TensorView k_norm_weight, double eps,
                     TensorView q_out, TensorView state_cache, TensorView state_slots,
                     TensorView state_block_table, TensorView query_start_loc,
                     TensorView logical_positions, TensorView compressed_cache,
                     TensorView compressed_slots, TensorView work_metadata, int64_t compress_ratio,
                     int64_t mrope_h, int64_t mrope_w, bool is_k_mrope, bool cache_has_rope_pos) {
  CHECK_DEVICE(k, q);
  CHECK_DEVICE(positions, q);
  CHECK_DEVICE(cos_sin_cache, q);
  CHECK_DEVICE(q_norm_weight, q);
  CHECK_DEVICE(k_norm_weight, q);
  CHECK_DEVICE(q_out, q);
  CHECK_DEVICE(state_cache, q);
  CHECK_DEVICE(state_slots, q);
  CHECK_DEVICE(state_block_table, q);
  CHECK_DEVICE(query_start_loc, q);
  CHECK_DEVICE(logical_positions, q);
  CHECK_DEVICE(compressed_cache, q);
  CHECK_DEVICE(compressed_slots, q);
  CHECK_DEVICE(work_metadata, q);
  CHECK_DIM(2, q);                 // [tokens, heads * head_dim]
  CHECK_DIM(2, k);                 // [tokens, head_dim]
  CHECK_DIM(3, q_out);             // [tokens, heads, head_dim]
  CHECK_DIM(4, state_cache);       // [blocks, ring, 1, head_dim (+ coordinates)]
  CHECK_DIM(4, compressed_cache);  // [blocks, page, 1, head_dim]
  CHECK_DIM(2, state_block_table);
  CHECK_DIM(2, work_metadata);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS(q_out);
  CHECK_LAST_DIM_CONTIGUOUS(state_cache);
  CHECK_LAST_DIM_CONTIGUOUS(compressed_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);
  CHECK_CONTIGUOUS(q_norm_weight);
  CHECK_CONTIGUOUS(k_norm_weight);
  CHECK_CONTIGUOUS(state_slots);
  CHECK_CONTIGUOUS(compressed_slots);
  CHECK_CONTIGUOUS(logical_positions);
  CHECK_CONTIGUOUS(query_start_loc);
  CHECK_CONTIGUOUS(work_metadata);

  const int64_t num_tokens = q.size(0);
  if (num_tokens == 0) return;
  const int64_t num_q_heads = q_out.size(1);
  const int64_t head_dim = q_out.size(2);
  TVM_FFI_ICHECK(head_dim == 128 || head_dim == 256)
      << "qsa_pre_indexer builds a head dimension of 128 or 256, got " << head_dim;
  TVM_FFI_ICHECK_GT(compress_ratio, 0) << "compress_ratio must be positive";
  TVM_FFI_ICHECK_EQ(q.size(1), num_q_heads * head_dim) << "q must hold every head of a token";
  TVM_FFI_ICHECK_EQ(k.size(1), head_dim) << "k and q_out head_dim must match";
  TVM_FFI_ICHECK_EQ(q_out.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(compressed_cache.size(3), head_dim)
      << "the compressed cache and q_out head_dim must match";
  // Both caches are strided by their first two axes and the row is read from
  // there, so a second KV head would land on top of the first.
  TVM_FFI_ICHECK_EQ(state_cache.size(2), 1) << "the ring holds one KV head";
  TVM_FFI_ICHECK_EQ(compressed_cache.size(2), 1) << "the compressed cache holds one KV head";
  // The rotary table is addressed as a flat run of rows of half the head
  // dimension; a row of any other width silently shifts every position.
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(cos_sin_cache.ndim() - 1), head_dim / 2)
      << "the rotary table is pair-major, so a row is head_dim / 2 wide";
  // Three int64 coordinates sit after the row, in units of the row's own type.
  const int64_t coord_width =
      cache_has_rope_pos ? 3 * static_cast<int64_t>(sizeof(int64_t)) / get_element_size(state_cache)
                         : 0;
  TVM_FFI_ICHECK_GE(state_cache.size(3), head_dim + coord_width)
      << "the ring holds a row of head_dim, and its coordinates after it";
  TVM_FFI_ICHECK_EQ(work_metadata.size(1), 2) << "a work item is a request and its index";
  TVM_FFI_ICHECK_EQ(state_slots.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(compressed_slots.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(logical_positions.size(0), num_tokens);
  // Both are walked by a token stride, so a shorter axis is read past its end.
  TVM_FFI_ICHECK_EQ(k.size(0), num_tokens) << "one raw key per token";
  TVM_FFI_ICHECK_EQ(positions.size(positions.ndim() - 1), num_tokens)
      << "one rotary position per token";
  // A work item names a request and the kernel reads that request's end, so the
  // table has to run one past the last of them.
  TVM_FFI_ICHECK_EQ(query_start_loc.size(0), state_block_table.size(0) + 1)
      << "query_start_loc runs one entry past the requests the ring table holds";

  // Every half-width tensor is read through one pointer type.
  TVM_FFI_ICHECK_EQ(k.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(cos_sin_cache.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(q_norm_weight.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(k_norm_weight.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(q_out.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(state_cache.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(compressed_cache.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(positions.dtype(), dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK_EQ(state_slots.dtype(), dl_int64) << "state_slots must be int64";
  TVM_FFI_ICHECK_EQ(compressed_slots.dtype(), dl_int64) << "compressed_slots must be int64";
  TVM_FFI_ICHECK_EQ(logical_positions.dtype(), dl_int64) << "logical_positions must be int64";
  TVM_FFI_ICHECK_EQ(query_start_loc.dtype(), dl_int32) << "query_start_loc must be int32";
  TVM_FFI_ICHECK_EQ(state_block_table.dtype(), dl_int32) << "state_block_table must be int32";
  TVM_FFI_ICHECK_EQ(work_metadata.dtype(), dl_int32) << "work_metadata must be int32";

  const bool pos_2d = positions.ndim() == 2;
  TVM_FFI_ICHECK(pos_2d || positions.ndim() == 1) << "positions is one axis or three";
  if (pos_2d) {
    TVM_FFI_ICHECK_EQ(positions.size(0), 3) << "a three-axis position tensor has three rows";
  }

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    QSAPreIndexerParams<c_type> p{};
    p.q = static_cast<const c_type*>(q.data_ptr());
    p.q_stride_token = q.stride(0);
    p.k = static_cast<const c_type*>(k.data_ptr());
    p.k_stride_token = k.stride(0);
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.pos_stride_axis = pos_2d ? positions.stride(0) : 0;
    p.pos_stride_token = pos_2d ? positions.stride(1) : positions.stride(0);
    p.cos_sin = static_cast<const c_type*>(cos_sin_cache.data_ptr());
    p.q_norm_weight = static_cast<const c_type*>(q_norm_weight.data_ptr());
    p.k_norm_weight = static_cast<const c_type*>(k_norm_weight.data_ptr());
    p.eps = static_cast<float>(eps);
    p.q_out = static_cast<c_type*>(q_out.data_ptr());
    p.q_out_stride_token = q_out.stride(0);
    p.q_out_stride_head = q_out.stride(1);
    p.state_cache = static_cast<c_type*>(state_cache.data_ptr());
    p.state_stride_block = state_cache.stride(0);
    p.state_stride_token = state_cache.stride(1);
    p.state_slots = static_cast<const int64_t*>(state_slots.data_ptr());
    p.state_table = static_cast<const int32_t*>(state_block_table.data_ptr());
    p.state_table_stride_req = state_block_table.stride(0);
    p.query_start_loc = static_cast<const int32_t*>(query_start_loc.data_ptr());
    p.logical_positions = static_cast<const int64_t*>(logical_positions.data_ptr());
    p.compressed_slots = static_cast<const int64_t*>(compressed_slots.data_ptr());
    p.work_metadata = static_cast<const int32_t*>(work_metadata.data_ptr());
    p.compressed_cache = static_cast<c_type*>(compressed_cache.data_ptr());
    p.compressed_stride_block = compressed_cache.stride(0);
    p.compressed_stride_token = compressed_cache.stride(1);
    p.num_tokens = static_cast<int32_t>(num_tokens);
    p.num_state_blocks = static_cast<int32_t>(state_cache.size(0));
    p.num_compressed_blocks = static_cast<int32_t>(compressed_cache.size(0));
    p.num_k_work = static_cast<int32_t>(work_metadata.size(0));
    p.num_q_heads = static_cast<int32_t>(num_q_heads);
    p.compress_ratio = static_cast<int32_t>(compress_ratio);
    p.state_size = static_cast<int32_t>(state_cache.size(1));
    p.comp_page_size = static_cast<int32_t>(compressed_cache.size(1));
    p.mrope_h = static_cast<int32_t>(mrope_h);
    p.mrope_w = static_cast<int32_t>(mrope_w);
    p.ratio_shift = shift_of(compress_ratio);
    p.state_shift = shift_of(p.state_size);
    p.comp_shift = shift_of(p.comp_page_size);
    p.inv_ratio = 1.f / static_cast<float>(compress_ratio);

    const cudaError_t status =
        head_dim == 128
            ? QSAPreIndexer<128, c_type>(p, is_k_mrope, pos_2d, cache_has_rope_pos, stream)
            : QSAPreIndexer<256, c_type>(p, is_k_mrope, pos_2d, cache_has_rope_pos, stream);
    TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
        << "qsa_pre_indexer: unsupported rotary configuration (three-axis positions need a "
           "three-axis key)";
    TVM_FFI_ICHECK(status == cudaSuccess) << "QSAPreIndexer failed: " << cudaGetErrorString(status);
    return true;
  });
}
