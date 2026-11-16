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
#include <algorithm>
#include <cstdint>
#include <flashinfer/fp8_kv_widen.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

constexpr int kThreads = 256;

template <typename DTypeKV, typename DTypeO>
void LaunchGatherWiden(const DTypeKV* src, DTypeO* dst, const int32_t* page_indices,
                       int64_t num_pages, int64_t page_elems, cudaStream_t stream) {
  const int64_t num_vecs = page_elems / 16;
  if (num_pages == 0 || num_vecs == 0) {
    return;
  }
  const int blocks_y =
      static_cast<int>(std::min<int64_t>((num_vecs + kThreads - 1) / kThreads, 64));
  const dim3 grid(static_cast<unsigned>(num_pages), static_cast<unsigned>(blocks_y));
  fp8_kv_widen::GatherWidenKernel<DTypeKV, DTypeO>
      <<<grid, kThreads, 0, stream>>>(src, dst, page_indices, page_elems);
}

}  // namespace

// Gather the pages named by page_indices in logical order and widen each FP8
// element to the destination 16-bit dtype. The destination cache is compact:
// page j of dst holds src page page_indices[j].
void gather_widen_fp8_kv(TensorView k_src, TensorView v_src, TensorView k_dst, TensorView v_dst,
                         TensorView page_indices) {
  CHECK_INPUT(k_src);
  CHECK_INPUT(v_src);
  CHECK_INPUT(k_dst);
  CHECK_INPUT(v_dst);
  CHECK_INPUT(page_indices);
  CHECK_DEVICE(k_src, v_src);
  CHECK_DEVICE(k_src, k_dst);
  CHECK_DEVICE(k_src, v_dst);
  CHECK_DEVICE(k_src, page_indices);
  TVM_FFI_ICHECK_EQ(page_indices.dtype(), dl_int32)
      << "page_indices must be int32, got " << page_indices.dtype();
  CHECK_SAME_DTYPE(k_src, v_src);
  CHECK_SAME_DTYPE(k_dst, v_dst);

  const int64_t num_pages = page_indices.size(0);
  TVM_FFI_ICHECK_EQ(k_dst.size(0), num_pages) << "k_dst page count must match page_indices";
  TVM_FFI_ICHECK_EQ(v_dst.size(0), num_pages) << "v_dst page count must match page_indices";
  TVM_FFI_ICHECK_EQ(k_src.size(0), v_src.size(0))
      << "k_src and v_src must have the same number of pages";

  if (num_pages == 0) {
    return;
  }
  // The source cache is the caller's page pool; page_indices selects the pages
  // that are gathered into the compact destination.
  const int64_t k_page_elems = k_src.numel() / k_src.size(0);
  const int64_t v_page_elems = v_src.numel() / v_src.size(0);
  TVM_FFI_ICHECK_EQ(k_dst.numel(), num_pages * k_page_elems)
      << "k_dst must hold num_pages * k_page_elems values";
  TVM_FFI_ICHECK_EQ(v_dst.numel(), num_pages * v_page_elems)
      << "v_dst must hold num_pages * v_page_elems values";
  TVM_FFI_ICHECK(k_page_elems % 16 == 0)
      << "k page size must be a multiple of 16 elements, got " << k_page_elems;
  TVM_FFI_ICHECK(v_page_elems % 16 == 0)
      << "v page size must be a multiple of 16 elements, got " << v_page_elems;

  ffi::CUDADeviceGuard device_guard(k_src.device().device_id);
  const cudaStream_t stream = get_stream(k_src.device());
  const int32_t* indices = static_cast<const int32_t*>(page_indices.data_ptr());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(k_src.dtype(), kv_t, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k_dst.dtype(), o_t, [&] {
      LaunchGatherWiden(static_cast<const kv_t*>(k_src.data_ptr()),
                        static_cast<o_t*>(k_dst.data_ptr()), indices, num_pages, k_page_elems,
                        stream);
      LaunchGatherWiden(static_cast<const kv_t*>(v_src.data_ptr()),
                        static_cast<o_t*>(v_dst.data_ptr()), indices, num_pages, v_page_elems,
                        stream);
      return true;
    });
  });

  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess)
      << "gather_widen_fp8_kv failed: " << cudaGetErrorString(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather_widen_fp8_kv, gather_widen_fp8_kv);
