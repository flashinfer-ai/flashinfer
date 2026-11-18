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
#include <flashinfer/mla/batch_invariant_sparse.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

namespace {

namespace bs = mla::bi_sparse;

void check_dtype(TensorView t, DLDataTypeCode code, int bits, const char* name) {
  TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
      << name << " has an unexpected dtype";
}

}  // namespace

// One row per query token; q and cache are both bf16 or both uint8 (e4m3).
void mla_batch_invariant_sparse(TensorView q, TensorView cache, TensorView idx, TensorView lens,
                                TensorView ws, TensorView out, double c, double s2,
                                int64_t head_block, int64_t gather_warps, int64_t tiles_per_unit,
                                int64_t pv_pass, int64_t mma_split) {
  CHECK_DIM(3, q);
  CHECK_DIM(2, cache);
  CHECK_DIM(2, idx);
  CHECK_DIM(3, out);
  const int64_t rows = q.size(0), heads = q.size(1), D = q.size(2), width = idx.size(1);
  TVM_FFI_ICHECK(D == 576 || D == 512) << "the query rows must be 576 or 512 wide";
  TVM_FFI_ICHECK(cache.size(1) == D) << "the cache rows must be as wide as the query rows";
  TVM_FFI_ICHECK(idx.size(0) == rows && out.size(0) == rows && out.size(1) == heads &&
                 out.size(2) == bs::kLatent);
  TVM_FFI_ICHECK(heads >= 1 && heads <= 128 && width >= 1);
  TVM_FFI_ICHECK(head_block == 16 || head_block == 32 || head_block == 64);
  TVM_FFI_ICHECK(gather_warps == 4 || gather_warps == 8);
  TVM_FFI_ICHECK(gather_warps * 16 >= head_block);
  TVM_FFI_ICHECK(pv_pass == 64 || pv_pass == 128);
  TVM_FFI_ICHECK(mma_split == 1 || (mma_split == 4 && head_block == 16));
  TVM_FFI_ICHECK(tiles_per_unit == 0 || tiles_per_unit == 1 || tiles_per_unit == 2 ||
                 tiles_per_unit == 4 || tiles_per_unit == 8);
  check_dtype(out, kDLBfloat, 16, "out");
  check_dtype(idx, kDLInt, 32, "the index list");
  check_dtype(ws, kDLFloat, 32, "the workspace");
  const bool fp8 = cache.dtype().code == kDLUInt && cache.dtype().bits == 8;
  if (!fp8) check_dtype(cache, kDLBfloat, 16, "the cache");
  TVM_FFI_ICHECK(q.dtype().code == cache.dtype().code && q.dtype().bits == cache.dtype().bits)
      << "the query and the cache must have one dtype";
  for (TensorView t : {q, cache, idx, ws, out}) {
    CHECK_CONTIGUOUS(t);
    CHECK_DEVICE(t, q);
  }
  for (TensorView t : {q, cache, ws, out}) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
        << "the query, the cache, the workspace and the output must start on a 16-byte boundary";
  }
  const int32_t* lens_ptr = nullptr;
  if (lens.numel() != 0) {
    check_dtype(lens, kDLInt, 32, "the live counts");
    CHECK_CONTIGUOUS(lens);
    CHECK_DEVICE(lens, q);
    TVM_FFI_ICHECK(lens.numel() == rows);
    lens_ptr = static_cast<const int32_t*>(lens.data_ptr());
  }
  const int tiles = static_cast<int>((width + bs::kTile - 1) / bs::kTile);
  TVM_FFI_ICHECK(ws.numel() >= bs::WorkspaceFloats(rows, heads, tiles, tiles_per_unit))
      << "the workspace is too small";
  const float cf = static_cast<float>(c);
  bs::Params p{cache.data_ptr(),
               q.data_ptr(),
               static_cast<const int32_t*>(idx.data_ptr()),
               lens_ptr,
               static_cast<float*>(ws.data_ptr()),
               static_cast<uint16_t*>(out.data_ptr()),
               cf,
               -cf,
               static_cast<float>(s2),
               static_cast<int>(rows),
               static_cast<int>(heads),
               static_cast<int>(width),
               tiles,
               static_cast<int>(head_block),
               static_cast<int>(tiles_per_unit),
               cache.size(0)};
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaError_t status =
      bs::Launch(p, static_cast<int>(D - bs::kLatent), fp8, static_cast<int>(gather_warps),
                 static_cast<int>(pv_pass), static_cast<int>(mma_split), get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "the batch-invariant sparse MLA kernel failed: " << cudaGetErrorString(status);
}

// y[i] = ex2.approx.ftz.f32(x[i]): the device primitive, for the exact CPU model.
void mla_batch_invariant_sparse_ex2(TensorView x, TensorView y) {
  check_dtype(x, kDLFloat, 32, "x");
  check_dtype(y, kDLFloat, 32, "y");
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(y);
  CHECK_DEVICE(x, y);
  TVM_FFI_ICHECK(x.numel() == y.numel());
  ffi::CUDADeviceGuard device_guard(x.device().device_id);
  cudaError_t status =
      bs::LaunchEx2(static_cast<const float*>(x.data_ptr()), static_cast<float*>(y.data_ptr()),
                    static_cast<int>(x.numel()), get_stream(x.device()));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_batch_invariant_sparse, mla_batch_invariant_sparse);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_batch_invariant_sparse_ex2, mla_batch_invariant_sparse_ex2);
