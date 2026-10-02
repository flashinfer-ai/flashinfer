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

// Shared tvm-ffi launcher for the Cake SM100/SM103 block-sparse attention
// profiles.  flashinfer/jit/cake_vsa.py compiles this translation unit once per
// (profile, architecture) and embeds that profile's cubin; the profile is
// selected through the defines below, so the ten kernel profiles share one
// host source.
//
//   CAKE_VSA_KERNEL           kernel symbol (identifier)
//   CAKE_VSA_THREADS          thread-block size
//   CAKE_VSA_SMEM_BYTES       dynamic shared memory
//   CAKE_VSA_ABI              1 = block-mask profiles, 2 = block-64 direct
//                             profiles, 3 = FP16 direct (q2k) profile
//   CAKE_VSA_FP16             Q/K/V/out are float16 (default bfloat16)
//   CAKE_VSA_Q_LAYOUT         0 = 4D 64-column-split TMA, 1 = 3D TMA,
//                             2 = plain pointer                       (ABI 1)
//   CAKE_VSA_Q_BOX_ROWS       Q TMA box rows (64 or 128)              (ABI 1, layout 0)
//   CAKE_VSA_Q_BOX_SPLIT      Q TMA box 64-column splits (1 or 2)     (ABI 1, layout 0)
//   CAKE_VSA_KV_LAYOUT        0 = 4D 64-column-split TMA, 1 = 3D TMA  (ABI 1)
//   CAKE_VSA_HEAD_DIM         enforced trailing extent of q/k/v       (ABI 1, optional)
//   CAKE_VSA_OUT_ROW_ELEMS    out elements per (query block, head)    (ABI 1)
//   CAKE_VSA_SELECTED_BLOCKS  scalar selected_blocks follows nb       (ABI 1)
//   CAKE_VSA_BSR_INDICES      int32 BSR columns replace the uint8 mask
//                             and total_tiles follows selected_blocks (ABI 1)
//   CAKE_VSA_OUT_BOX_COLS     out TMA box columns (64 or 32)          (ABI 2)

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

#ifndef CAKE_VSA_KERNEL
#error "CAKE_VSA_KERNEL must name the embedded kernel symbol"
#endif
#ifndef CAKE_VSA_THREADS
#error "CAKE_VSA_THREADS must give the thread-block size"
#endif
#ifndef CAKE_VSA_SMEM_BYTES
#error "CAKE_VSA_SMEM_BYTES must give the dynamic shared memory size"
#endif
#ifndef CAKE_VSA_ABI
#error "CAKE_VSA_ABI must select the launcher ABI (1, 2 or 3)"
#endif

#define CAKE_VSA_STRINGIFY_(x) #x
#define CAKE_VSA_STRINGIFY(x) CAKE_VSA_STRINGIFY_(x)

TVM_FFI_EMBED_CUBIN(cake_vsa_kernel);

namespace flashinfer {
namespace cake_vsa {
// Internal linkage for everything below: this file is compiled once per (profile, arch) into
// separate modules. A static kernel handle inside an inline function would be an STB_GNU_UNIQUE
// symbol that the dynamic loader unifies across modules, making a later-loaded profile launch
// the first module's kernel.
namespace {

using tvm::ffi::TensorView;

#if defined(CAKE_VSA_FP16) || CAKE_VSA_ABI == 3
constexpr DLDataType kIoDtype = dl_float16;
constexpr CUtensorMapDataType kIoTmaDtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#else
constexpr DLDataType kIoDtype = dl_bfloat16;
constexpr CUtensorMapDataType kIoTmaDtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
#endif
constexpr int kIoBytes = 2;

inline void check_input(const TensorView& t, DLDataType dtype, const char* name) {
  check_cuda_tensor(t, name);
  check_dtype(t, dtype, name);
  check_contiguous(t, name);
}

inline void CheckI32(int64_t value, const char* name) {
  TVM_FFI_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                    value <= std::numeric_limits<int32_t>::max(),
                ValueError)
      << "scalar '" << name << "' value " << value << " is outside the int32 range";
}

inline int64_t HostExtent(int64_t a, int64_t b, int64_t c = 1) {
  int64_t extent = 1;
  for (int64_t factor : {a, b, c}) {
    TVM_FFI_CHECK(factor >= 0, ValueError)
        << "host extent factors must be non-negative, got " << factor;
    if (factor != 0) {
      TVM_FFI_CHECK(extent <= std::numeric_limits<int64_t>::max() / factor, ValueError)
          << "host extent overflows int64";
    }
    extent *= factor;
  }
  return extent;
}

inline void check_extent(const TensorView& t, int64_t extent, const char* name) {
  TVM_FFI_CHECK(t.numel() >= extent, ValueError)
      << name << " requires at least " << extent << " elements, got " << t.numel();
}

// Trailing dimensions of a TMA source: d1 is the innermost extent.
struct Trailing {
  int64_t d1;
  int64_t d2;
  int64_t d3;
};

inline Trailing TrailingDims(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source '" << name << "' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  Trailing d{t.size(t.ndim() - 1), t.size(t.ndim() - 2), t.size(t.ndim() - 3)};
  TVM_FFI_CHECK(d.d1 > 0 && d.d2 > 0 && d.d3 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  return d;
}

inline void RequireSplit64(int64_t d1, const char* name) {
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source '" << name << "' extent " << d1 << " must divide exactly by 64";
}

inline CUtensorMap EncodeTiled(const TensorView& t, const char* name, CUtensorMapDataType dtype,
                               uint32_t rank, const uint64_t* global_dim,
                               const uint64_t* global_strides, const uint32_t* box_dim,
                               CUtensorMapSwizzle swizzle) {
  for (uint32_t axis = 0; axis < rank; ++axis) {
    TVM_FFI_CHECK(global_dim[axis] > 0, ValueError)
        << "TMA descriptor for '" << name << "' resolved a non-positive global dim";
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box exceeds resolved global dims for '" << name << "'";
  }
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(&tm, dtype, rank, t.data_ptr(), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (" << rank << "D, '" << name << "') failed: CUresult=" << (int)r;
  return tm;
}

// [rows, heads, head_dim] viewed as (64 columns, rows, heads, head_dim / 64).
inline CUtensorMap EncodeQSplit(const TensorView& t, uint32_t box_rows, uint32_t box_split) {
  Trailing d = TrailingDims(t, "q");
  RequireSplit64(d.d1, "q");
  uint64_t global_dim[4] = {64u, (uint64_t)d.d3, (uint64_t)d.d2, (uint64_t)(d.d1 / 64)};
  uint64_t global_strides[3] = {(uint64_t)(d.d2 * d.d1 * kIoBytes), (uint64_t)(d.d1 * kIoBytes),
                                (uint64_t)(64 * kIoBytes)};
  uint32_t box_dim[4] = {64u, box_rows, 1u, box_split};
  return EncodeTiled(t, "q", kIoTmaDtype, 4, global_dim, global_strides, box_dim,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// [rows, heads, head_dim] viewed as (64 columns, heads, rows, head_dim / 64).
inline CUtensorMap EncodeQHeadMajorSplit(const TensorView& t) {
  Trailing d = TrailingDims(t, "q");
  RequireSplit64(d.d1, "q");
  uint64_t global_dim[4] = {64u, (uint64_t)d.d2, (uint64_t)d.d3, (uint64_t)(d.d1 / 64)};
  uint64_t global_strides[3] = {(uint64_t)(d.d1 * kIoBytes), (uint64_t)(d.d2 * d.d1 * kIoBytes),
                                (uint64_t)(64 * kIoBytes)};
  uint32_t box_dim[4] = {64u, 1u, 64u, 2u};
  return EncodeTiled(t, "q", kIoTmaDtype, 4, global_dim, global_strides, box_dim,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// [rows, heads, head_dim] viewed as (64 columns, rows, head_dim / 64, heads).
inline CUtensorMap EncodeKvSplit(const TensorView& t, const char* name) {
  Trailing d = TrailingDims(t, name);
  RequireSplit64(d.d1, name);
  uint64_t global_dim[4] = {64u, (uint64_t)d.d3, (uint64_t)(d.d1 / 64), (uint64_t)d.d2};
  uint64_t global_strides[3] = {(uint64_t)(d.d2 * d.d1 * kIoBytes), (uint64_t)(64 * kIoBytes),
                                (uint64_t)(d.d1 * kIoBytes)};
  uint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  return EncodeTiled(t, name, kIoTmaDtype, 4, global_dim, global_strides, box_dim,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// [rows, heads, head_dim] viewed as (head_dim, rows, heads) with a 64 x 64 box.
inline CUtensorMap EncodeDense3D(const TensorView& t, const char* name) {
  Trailing d = TrailingDims(t, name);
  uint64_t global_dim[3] = {(uint64_t)d.d1, (uint64_t)d.d3, (uint64_t)d.d2};
  uint64_t global_strides[2] = {(uint64_t)(d.d2 * d.d1 * kIoBytes), (uint64_t)(d.d1 * kIoBytes)};
  uint32_t box_dim[3] = {64u, 64u, 1u};
  return EncodeTiled(t, name, kIoTmaDtype, 3, global_dim, global_strides, box_dim,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

#if CAKE_VSA_ABI == 2
// [rows, heads, 128] output viewed as (128, rows, heads).
inline CUtensorMap EncodeOut(const TensorView& t) {
  Trailing d = TrailingDims(t, "out");
  uint64_t global_dim[3] = {128u, (uint64_t)d.d3, (uint64_t)d.d2};
  uint64_t global_strides[2] = {(uint64_t)(d.d2 * d.d1 * kIoBytes), (uint64_t)(d.d1 * kIoBytes)};
  uint32_t box_dim[3] = {CAKE_VSA_OUT_BOX_COLS, 64u, 1u};
  return EncodeTiled(t, "out", kIoTmaDtype, 3, global_dim, global_strides, box_dim,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}
#endif

#if CAKE_VSA_ABI == 3
// [.., pages, blocks, 1024 bytes] block scales viewed as (1024, blocks, pages).
inline CUtensorMap EncodeScale(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source '" << name << "' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t s3 = t.stride(t.ndim() - 3);
  int64_t s4 = t.stride(t.ndim() - 4);
  TVM_FFI_CHECK(s3 > 0 && s4 > 0, ValueError)
      << "TMA source '" << name << "' physical strides must be positive";
  uint64_t global_dim[3] = {1024u, (uint64_t)d3, (uint64_t)d4};
  uint64_t global_strides[2] = {(uint64_t)s3, (uint64_t)s4};
  uint32_t box_dim[3] = {256u, 1u, 1u};
  return EncodeTiled(t, name, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, global_dim, global_strides,
                     box_dim, CU_TENSOR_MAP_SWIZZLE_NONE);
}
#endif

inline void Launch(const TensorView& q, void** kargs, int64_t grid_x, int64_t grid_y,
                   int64_t grid_z) {
  TVM_FFI_CHECK(grid_x > 0 && grid_x <= std::numeric_limits<uint32_t>::max() && grid_y > 0 &&
                    grid_y <= std::numeric_limits<uint32_t>::max() && grid_z > 0 &&
                    grid_z <= std::numeric_limits<uint32_t>::max(),
                ValueError)
      << "launch grid dimensions must fit uint32_t, got (" << grid_x << ", " << grid_y << ", "
      << grid_z << ")";
  static auto kernel = EmbedCubinModule_cake_vsa_kernel::Global()->mod.GetKernelWithMaxDynamicSharedMemory(
      CAKE_VSA_STRINGIFY(CAKE_VSA_KERNEL), CAKE_VSA_SMEM_BYTES);
  const cudaStream_t stream = get_stream(q.device());
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(CAKE_VSA_THREADS, 1u, 1u);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, CAKE_VSA_SMEM_BYTES));
}

#if CAKE_VSA_ABI == 1

#ifndef CAKE_VSA_OUT_ROW_ELEMS
#error "CAKE_VSA_OUT_ROW_ELEMS must give the output elements per (query block, head)"
#endif
#if CAKE_VSA_Q_LAYOUT == 0 && !(defined(CAKE_VSA_Q_BOX_ROWS) && defined(CAKE_VSA_Q_BOX_SPLIT))
#error "CAKE_VSA_Q_BOX_ROWS and CAKE_VSA_Q_BOX_SPLIT must describe the Q TMA box"
#endif

inline void check_head_dim(const TensorView& t, const char* name) {
#ifdef CAKE_VSA_HEAD_DIM
  TVM_FFI_CHECK(t.ndim() == 3, ValueError) << name << " must have rank 3, got " << t.ndim();
  TVM_FFI_CHECK(t.size(-1) == CAKE_VSA_HEAD_DIM, ValueError)
      << name << " dimension -1 must equal " << CAKE_VSA_HEAD_DIM << ", got " << t.size(-1);
#else
  (void)t;
  (void)name;
#endif
}

void Run(TensorView q, TensorView k, TensorView v, TensorView out, TensorView lse,
         TensorView temperature_lse, TensorView mask, int64_t mb, int64_t nb,
#ifdef CAKE_VSA_SELECTED_BLOCKS
         int64_t selected_blocks,
#endif
#ifdef CAKE_VSA_BSR_INDICES
         int64_t total_tiles,
#endif
         int64_t num_q_heads, int64_t num_kv_heads, double softmax_scale_log2,
         double lse_temperature_scale, int64_t return_softmax_lse, int64_t return_temperature_lse,
         int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  check_input(q, kIoDtype, "q");
  check_input(k, kIoDtype, "k");
  check_input(v, kIoDtype, "v");
  check_input(out, kIoDtype, "out");
  check_input(lse, dl_float32, "lse");
  check_input(temperature_lse, dl_float32, "temperature_lse");
#ifdef CAKE_VSA_BSR_INDICES
  check_input(mask, dl_int32, "bsr_indices");
#else
  check_input(mask, dl_uint8, "block_mask");
#endif
  CheckI32(mb, "mb");
  CheckI32(nb, "nb");
#ifdef CAKE_VSA_SELECTED_BLOCKS
  CheckI32(selected_blocks, "selected_blocks");
#endif
#ifdef CAKE_VSA_BSR_INDICES
  CheckI32(total_tiles, "total_tiles");
#endif
  CheckI32(num_q_heads, "num_q_heads");
  CheckI32(num_kv_heads, "num_kv_heads");
  CheckI32(return_softmax_lse, "return_softmax_lse");
  CheckI32(return_temperature_lse, "return_temperature_lse");
  for (const TensorView* t : {&k, &v, &out, &lse, &temperature_lse, &mask}) {
    check_same_device(*t, q, "tensor argument", "q");
  }
  check_head_dim(q, "q");
  check_head_dim(k, "k");
  check_head_dim(v, "v");
  check_extent(out, HostExtent(CAKE_VSA_OUT_ROW_ELEMS, num_q_heads, mb), "out");
  if (return_softmax_lse != 0) {
    check_extent(lse, HostExtent(128, num_q_heads, mb), "lse");
  }
  if (return_temperature_lse != 0) {
    check_extent(temperature_lse, HostExtent(128, num_q_heads, mb), "temperature_lse");
  }
#ifdef CAKE_VSA_BSR_INDICES
  check_extent(mask, HostExtent(mb, selected_blocks), "bsr_indices");
  TVM_FFI_CHECK(selected_blocks == 6, ValueError)
      << "selected_blocks must equal 6, got " << selected_blocks;
  TVM_FFI_CHECK(total_tiles == HostExtent(mb, num_q_heads), ValueError)
      << "total_tiles must equal " << HostExtent(mb, num_q_heads) << ", got " << total_tiles;
#else
  check_extent(mask, HostExtent(num_q_heads, mb, nb), "block_mask");
#endif

  const DLDevice dev = q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  // Bind the primary context for the tensor-map encoders.
  TVM_FFI_CHECK(cudaSetDevice(dev.device_id) == cudaSuccess, RuntimeError)
      << "cudaSetDevice(" << dev.device_id << ") failed";
#if CAKE_VSA_Q_LAYOUT == 0
  CUtensorMap p_q = EncodeQSplit(q, CAKE_VSA_Q_BOX_ROWS, CAKE_VSA_Q_BOX_SPLIT);
#elif CAKE_VSA_Q_LAYOUT == 1
  CUtensorMap p_q = EncodeDense3D(q, "q");
#else
  check_extent(q, HostExtent(CAKE_VSA_OUT_ROW_ELEMS, num_q_heads, mb), "q");
  void* p_q = q.data_ptr();
#endif
#if CAKE_VSA_KV_LAYOUT == 0
  CUtensorMap p_k = EncodeKvSplit(k, "k");
  CUtensorMap p_v = EncodeKvSplit(v, "v");
#else
  CUtensorMap p_k = EncodeDense3D(k, "k");
  CUtensorMap p_v = EncodeDense3D(v, "v");
#endif
  void* p_out = out.data_ptr();
  void* p_lse = lse.data_ptr();
  void* p_temperature_lse = temperature_lse.data_ptr();
  void* p_mask = mask.data_ptr();
  int32_t v_mb = (int32_t)mb;
  int32_t v_nb = (int32_t)nb;
#ifdef CAKE_VSA_SELECTED_BLOCKS
  int32_t v_selected_blocks = (int32_t)selected_blocks;
#endif
#ifdef CAKE_VSA_BSR_INDICES
  int32_t v_total_tiles = (int32_t)total_tiles;
#endif
  int32_t v_num_q_heads = (int32_t)num_q_heads;
  int32_t v_num_kv_heads = (int32_t)num_kv_heads;
  float v_softmax_scale_log2 = (float)softmax_scale_log2;
  float v_lse_temperature_scale = (float)lse_temperature_scale;
  int32_t v_return_softmax_lse = (int32_t)return_softmax_lse;
  int32_t v_return_temperature_lse = (int32_t)return_temperature_lse;
  void* kargs[] = {&p_q,
                   &p_k,
                   &p_v,
                   &p_out,
                   &p_lse,
                   &p_temperature_lse,
                   &p_mask,
                   &v_mb,
                   &v_nb,
#ifdef CAKE_VSA_SELECTED_BLOCKS
                   &v_selected_blocks,
#endif
#ifdef CAKE_VSA_BSR_INDICES
                   &v_total_tiles,
#endif
                   &v_num_q_heads,
                   &v_num_kv_heads,
                   &v_softmax_scale_log2,
                   &v_lse_temperature_scale,
                   &v_return_softmax_lse,
                   &v_return_temperature_lse};
  Launch(q, kargs, grid_x, grid_y, grid_z);
}

#elif CAKE_VSA_ABI == 2

#ifndef CAKE_VSA_OUT_BOX_COLS
#error "CAKE_VSA_OUT_BOX_COLS must give the output TMA box columns"
#endif

void Run(TensorView q, TensorView k, TensorView v, TensorView out, TensorView lse,
         TensorView q2k_indices, TensorView q2k_num, TensorView kv_block_lens,
         int64_t max_kv_blocks, int64_t sequence_q, int64_t query_blocks, int64_t total_tiles,
         int64_t tiles_per_cta, int64_t num_heads, double softmax_scale_log2, int64_t return_lse,
         int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  check_input(q, kIoDtype, "q");
  check_input(k, kIoDtype, "k");
  check_input(v, kIoDtype, "v");
  check_input(out, kIoDtype, "out");
  check_input(lse, dl_float32, "lse");
  check_input(q2k_indices, dl_int32, "q2k_indices");
  check_input(q2k_num, dl_int32, "q2k_num");
  check_input(kv_block_lens, dl_int32, "kv_block_lens");
  CheckI32(max_kv_blocks, "max_kv_blocks");
  CheckI32(sequence_q, "sequence_q");
  CheckI32(query_blocks, "query_blocks");
  CheckI32(total_tiles, "total_tiles");
  CheckI32(tiles_per_cta, "tiles_per_cta");
  CheckI32(num_heads, "num_heads");
  CheckI32(return_lse, "return_lse");
  for (const TensorView* t : {&k, &v, &out, &lse, &q2k_indices, &q2k_num, &kv_block_lens}) {
    check_same_device(*t, q, "tensor argument", "q");
  }
  TVM_FFI_CHECK(out.ndim() == 3, ValueError) << "out must have rank 3, got " << out.ndim();
  TVM_FFI_CHECK(out.size(-1) == 128, ValueError)
      << "out dimension -1 must equal 128, got " << out.size(-1);

  const DLDevice dev = q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_CHECK(cudaSetDevice(dev.device_id) == cudaSuccess, RuntimeError)
      << "cudaSetDevice(" << dev.device_id << ") failed";
  CUtensorMap p_q = EncodeQHeadMajorSplit(q);
  CUtensorMap p_k = EncodeKvSplit(k, "k");
  CUtensorMap p_v = EncodeKvSplit(v, "v");
  CUtensorMap p_out = EncodeOut(out);
  void* p_lse = lse.data_ptr();
  void* p_q2k_indices = q2k_indices.data_ptr();
  void* p_q2k_num = q2k_num.data_ptr();
  void* p_kv_block_lens = kv_block_lens.data_ptr();
  int32_t v_max_kv_blocks = (int32_t)max_kv_blocks;
  int32_t v_sequence_q = (int32_t)sequence_q;
  int32_t v_query_blocks = (int32_t)query_blocks;
  int32_t v_total_tiles = (int32_t)total_tiles;
  int32_t v_tiles_per_cta = (int32_t)tiles_per_cta;
  int32_t v_num_heads = (int32_t)num_heads;
  float v_softmax_scale_log2 = (float)softmax_scale_log2;
  int32_t v_return_lse = (int32_t)return_lse;
  void* kargs[] = {&p_q,          &p_k,           &p_v,          &p_out,
                   &p_lse,        &p_q2k_indices, &p_q2k_num,    &p_kv_block_lens,
                   &v_max_kv_blocks, &v_sequence_q, &v_query_blocks, &v_total_tiles,
                   &v_tiles_per_cta, &v_num_heads,  &v_softmax_scale_log2, &v_return_lse};
  Launch(q, kargs, grid_x, grid_y, grid_z);
}

#elif CAKE_VSA_ABI == 3

void Run(TensorView q, TensorView k, TensorView k_scale, TensorView v, TensorView v_scale,
         TensorView out, TensorView lse, TensorView temperature_lse, TensorView q2k_indices,
         TensorView cu_seqlens_q, TensorView cu_seqlens_k, TensorView q_offsets,
         TensorView kv_lens, TensorView page_table, int64_t total_q, int64_t num_q_heads,
         int64_t num_kv_heads, int64_t topk, int64_t batch_size, int64_t uniform_q_len,
         int64_t max_pages, int64_t causal, int64_t derive_q_offset, double softmax_scale_log2,
         double k_global_scale, double v_global_scale, double lse_temperature_scale,
         int64_t return_softmax_lse, int64_t return_temperature_lse, int64_t grid_x,
         int64_t grid_y, int64_t grid_z) {
  check_input(q, kIoDtype, "q");
  check_input(k, kIoDtype, "k");
  check_input(k_scale, dl_uint8, "k_scale");
  check_input(v, kIoDtype, "v");
  check_input(v_scale, dl_uint8, "v_scale");
  check_input(out, kIoDtype, "out");
  check_input(lse, dl_float32, "lse");
  check_input(temperature_lse, dl_float32, "temperature_lse");
  check_input(q2k_indices, dl_int32, "q2k_indices");
  check_input(cu_seqlens_q, dl_int32, "cu_seqlens_q");
  check_input(cu_seqlens_k, dl_int32, "cu_seqlens_k");
  check_input(q_offsets, dl_int32, "q_offsets");
  check_input(kv_lens, dl_int32, "kv_lens");
  check_input(page_table, dl_int32, "page_table");
  CheckI32(total_q, "total_q");
  CheckI32(num_q_heads, "num_q_heads");
  CheckI32(num_kv_heads, "num_kv_heads");
  CheckI32(topk, "topk");
  CheckI32(batch_size, "batch_size");
  CheckI32(uniform_q_len, "uniform_q_len");
  CheckI32(max_pages, "max_pages");
  CheckI32(causal, "causal");
  CheckI32(derive_q_offset, "derive_q_offset");
  CheckI32(return_softmax_lse, "return_softmax_lse");
  CheckI32(return_temperature_lse, "return_temperature_lse");
  for (const TensorView* t : {&k, &k_scale, &v, &v_scale, &out, &lse, &temperature_lse,
                              &q2k_indices, &cu_seqlens_q, &cu_seqlens_k, &q_offsets, &kv_lens,
                              &page_table}) {
    check_same_device(*t, q, "tensor argument", "q");
  }
  TVM_FFI_CHECK(batch_size >= 1, ValueError) << "batch_size must be >= 1, got " << batch_size;

  const DLDevice dev = q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_CHECK(cudaSetDevice(dev.device_id) == cudaSuccess, RuntimeError)
      << "cudaSetDevice(" << dev.device_id << ") failed";
  CUtensorMap p_q = EncodeQSplit(q, 128u, 2u);
  CUtensorMap p_k = EncodeKvSplit(k, "k");
  CUtensorMap p_k_scale = EncodeScale(k_scale, "k_scale");
  CUtensorMap p_v = EncodeKvSplit(v, "v");
  CUtensorMap p_v_scale = EncodeScale(v_scale, "v_scale");
  void* p_out = out.data_ptr();
  void* p_lse = lse.data_ptr();
  void* p_temperature_lse = temperature_lse.data_ptr();
  void* p_q2k_indices = q2k_indices.data_ptr();
  void* p_cu_seqlens_q = cu_seqlens_q.data_ptr();
  void* p_cu_seqlens_k = cu_seqlens_k.data_ptr();
  void* p_q_offsets = q_offsets.data_ptr();
  void* p_kv_lens = kv_lens.data_ptr();
  void* p_page_table = page_table.data_ptr();
  int32_t v_total_q = (int32_t)total_q;
  int32_t v_num_q_heads = (int32_t)num_q_heads;
  int32_t v_num_kv_heads = (int32_t)num_kv_heads;
  int32_t v_topk = (int32_t)topk;
  int32_t v_batch_size = (int32_t)batch_size;
  int32_t v_uniform_q_len = (int32_t)uniform_q_len;
  int32_t v_max_pages = (int32_t)max_pages;
  int32_t v_causal = (int32_t)causal;
  int32_t v_derive_q_offset = (int32_t)derive_q_offset;
  float v_softmax_scale_log2 = (float)softmax_scale_log2;
  float v_k_global_scale = (float)k_global_scale;
  float v_v_global_scale = (float)v_global_scale;
  float v_lse_temperature_scale = (float)lse_temperature_scale;
  int32_t v_return_softmax_lse = (int32_t)return_softmax_lse;
  int32_t v_return_temperature_lse = (int32_t)return_temperature_lse;
  void* kargs[] = {&p_q,
                   &p_k,
                   &p_k_scale,
                   &p_v,
                   &p_v_scale,
                   &p_out,
                   &p_lse,
                   &p_temperature_lse,
                   &p_q2k_indices,
                   &p_cu_seqlens_q,
                   &p_cu_seqlens_k,
                   &p_q_offsets,
                   &p_kv_lens,
                   &p_page_table,
                   &v_total_q,
                   &v_num_q_heads,
                   &v_num_kv_heads,
                   &v_topk,
                   &v_batch_size,
                   &v_uniform_q_len,
                   &v_max_pages,
                   &v_causal,
                   &v_derive_q_offset,
                   &v_softmax_scale_log2,
                   &v_k_global_scale,
                   &v_v_global_scale,
                   &v_lse_temperature_scale,
                   &v_return_softmax_lse,
                   &v_return_temperature_lse};
  Launch(q, kargs, grid_x, grid_y, grid_z);
}

#else
#error "CAKE_VSA_ABI must be 1, 2 or 3"
#endif

}  // namespace
}  // namespace cake_vsa
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_vsa::Run);
