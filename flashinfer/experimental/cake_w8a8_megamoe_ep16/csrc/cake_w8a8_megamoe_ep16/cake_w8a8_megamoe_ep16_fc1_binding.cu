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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), stage `fc1`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_fc1_kernel.cu): the persistent
// SwapAB FC1 GEMM with fused SwiGLU and MXFP8 output quantization. The binding
// checks the tensor metadata of the route contract, encodes the four TMA
// tensor maps on the host and passes them by value (A over the k-tile-major
// w13 payload in 256-row boxes, B over the row-major activation rows), opts
// the kernel into its dynamic shared memory once, resolves the environment
// stream and performs one launch carrying the programmatic-stream-serialization
// attribute (the local_pack -> fc1 edge of the route's programmatic dependent
// launch chain); the block size and the dynamic shared memory are the generated
// program's constants.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

// Device entry point of the kernel translation unit (manifest declaration).
// clang-format off
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_fc1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ C_fp8, uint8_t* __restrict__ SF_out, int* __restrict__ route_map, float* __restrict__ route_weights, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ expert_row_offsets, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int N, int K_tiles, int grid_m, int grid_n, int rows_per_expert);
// clang-format on

namespace cake_w8a8_megamoe_ep16_fc1 {

constexpr int64_t kLocalExperts = 32;
constexpr int64_t kPackedFc1Rows = 10240;  // gate || up rows of w13
constexpr int64_t kHidden = 3072;
constexpr int64_t kIntermediate = 5120;
constexpr int64_t kMaxRowsPerExpert = 128;   // packed FC1 input rows per expert
constexpr int64_t kW13ScaleTiles = 960;      // 256x128 scale tiles per expert of w13
constexpr int64_t kFc1InputScaleTiles = 24;  // 128x128 scale tiles per expert of the input
constexpr int64_t kScaleTileRows256 = 32;    // a 256x128 scale tile is [32, 32] bytes
constexpr int64_t kScaleTileRows128 = 16;    // a 128x128 scale tile is [16, 32] bytes
constexpr int64_t kScaleTileColumns = 32;
constexpr int64_t kFc2InputScaleBytes = 655360;  // packed scale plane of the FC1 output
constexpr int64_t kTotalExpertRows = 4096;       // 32 experts x 128 packed rows
constexpr int64_t kMaxRoutes = 8192;             // 16 ranks x 64 tokens x top-8
constexpr int64_t kFc1Tiles = 1280;              // 32 experts x 40 N256 tiles
constexpr int64_t kK128 = 128;
constexpr cuuint32_t kW13BoxRows = 256;  // k-tile-major w13 payload: one R x 128 B block per box
constexpr unsigned kBlockThreads = 192;
constexpr unsigned kDynamicSmemBytes = 162816;

// Require the exact extents of a route-fixed tensor.
template <int Rank>
void check_extents(const TensorView& t, const int64_t (&extents)[Rank], const char* name) {
  TVM_FFI_CHECK(t.ndim() == Rank, ValueError)
      << name << " must be " << Rank << "-D, got ndim=" << t.ndim();
  for (int axis = 0; axis < Rank; ++axis) {
    TVM_FFI_CHECK(t.size(axis) == extents[axis], ValueError)
        << name << ".size(" << axis << ") must be " << extents[axis] << ", got " << t.size(axis);
  }
}

// Require the element count of a route-fixed buffer.
void check_numel(const TensorView& t, int64_t numel, const char* name) {
  TVM_FFI_CHECK(t.numel() == numel, ValueError)
      << name << " must hold " << numel << " elements, got " << t.numel();
}

// 4-D tensor map of the k-tile-major uint8 [outer, rows, k] weight payload
// (manifest route_contract.weight_layout_rule: byte (e, r, c) lives at
// (((e * rows / R + r / R) * (k / 128) + c / 128) * R + r % R) * 128 + c % 128,
// i.e. [outer][rows / R][k / 128][R][128 B]), viewed as
// [128, R, k / 128, outer * rows / R] with 128-byte swizzle: global_dim
// {128, R, k / 128, outer * rows / R}, byte strides {128, R * 128, R * k}, box
// {128, R, 1, 1}; one box is one contiguous R x 128-byte block (the manifest's
// A tma record).
CUtensorMap encode_weight(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' needs at least 2 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t k = t.size(t.ndim() - 1);
  const int64_t rows = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(k > 0 && rows > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  TVM_FFI_CHECK(k % kK128 == 0, ValueError)
      << "TMA source '" << name << "' extent " << k << " must divide exactly by " << kK128;
  const int64_t outer = t.numel() / (k * rows);
  TVM_FFI_CHECK((outer * rows) % box_rows == 0, ValueError)
      << "TMA source '" << name << "' row count " << outer * rows
      << " must divide exactly by the box height " << box_rows;
  const cuuint64_t global_dim[4] = {128, static_cast<cuuint64_t>(box_rows),
                                    static_cast<cuuint64_t>(k / kK128),
                                    static_cast<cuuint64_t>(outer * rows / box_rows)};
  const cuuint64_t global_strides[3] = {128, static_cast<cuuint64_t>(box_rows) * 128,
                                        static_cast<cuuint64_t>(box_rows) * k};
  const cuuint32_t box_dim[4] = {128, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

// 4-D tensor map of a row-major uint8 [outer, rows, k] activation operand, viewed
// as [128, rows, k / 128, outer] with 128-byte swizzle; one box covers 128 bytes
// of K for `box_rows` rows.
CUtensorMap encode_operand(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' needs at least 2 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t k = t.size(t.ndim() - 1);
  const int64_t rows = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(k > 0 && rows > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  TVM_FFI_CHECK(k % kK128 == 0, ValueError)
      << "TMA source '" << name << "' extent " << k << " must divide exactly by " << kK128;
  const int64_t outer = t.numel() / (k * rows);
  const cuuint64_t global_dim[4] = {128, static_cast<cuuint64_t>(rows),
                                    static_cast<cuuint64_t>(k / kK128),
                                    static_cast<cuuint64_t>(outer)};
  const cuuint64_t global_strides[3] = {static_cast<cuuint64_t>(k), 128,
                                        static_cast<cuuint64_t>(rows * k)};
  const cuuint32_t box_dim[4] = {128, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

// 4-D tensor map of a row-major uint8 [outer, tiles, rows, 32] scale plane, read as
// [32, rows, tiles, outer] without swizzle; one box covers 32 bytes for `box_rows` rows.
CUtensorMap encode_scales(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source '" << name << "' needs at least 3 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t d1 = t.size(t.ndim() - 1);
  const int64_t d2 = t.size(t.ndim() - 2);
  const int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  const int64_t outer = t.numel() / (d1 * d2 * d3);
  const cuuint64_t global_dim[4] = {static_cast<cuuint64_t>(d1), static_cast<cuuint64_t>(d2),
                                    static_cast<cuuint64_t>(d3), static_cast<cuuint64_t>(outer)};
  const cuuint64_t global_strides[3] = {static_cast<cuuint64_t>(d1),
                                        static_cast<cuuint64_t>(d1 * d2),
                                        static_cast<cuuint64_t>(d1 * d2 * d3)};
  const cuuint32_t box_dim[4] = {32, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

void Run(TensorView A, TensorView B, TensorView SFA, TensorView SFB, TensorView C_fp8,
         TensorView SF_out, TensorView route_map, TensorView route_weights, TensorView tile_expert,
         TensorView tile_m_local, TensorView expert_row_offsets, TensorView scale_gate,
         TensorView clamp_limit, TensorView act_alpha, TensorView act_beta, int64_t N,
         int64_t K_tiles, int64_t grid_m, int64_t grid_n, int64_t rows_per_expert, int64_t grid_x) {
  check_cuda_tensor(A, "A");
  const DLDevice dev = A.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);
  {
    // The encoders need a current driver context on this thread: bind the primary
    // context once per thread per device (the runtime launch binds it itself).
    static thread_local uint64_t bound_devices = 0;
    const uint64_t device_bit = dev.device_id < 64 ? uint64_t{1} << dev.device_id : uint64_t{0};
    if (device_bit == 0 || (bound_devices & device_bit) == 0) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      bound_devices |= device_bit;
    }
  }

  check_dtype(A, dl_uint8, "A");
  check_contiguous(A, "A");
  check_cuda_tensor(B, "B");
  check_dtype(B, dl_uint8, "B");
  check_contiguous(B, "B");
  check_same_device(B, A, "B", "A");
  check_cuda_tensor(SFA, "SFA");
  check_dtype(SFA, dl_uint8, "SFA");
  check_contiguous(SFA, "SFA");
  check_same_device(SFA, A, "SFA", "A");
  check_cuda_tensor(SFB, "SFB");
  check_dtype(SFB, dl_uint8, "SFB");
  check_contiguous(SFB, "SFB");
  check_same_device(SFB, A, "SFB", "A");
  check_cuda_tensor(C_fp8, "C_fp8");
  check_dtype(C_fp8, dl_float8_e4m3fn, "C_fp8");
  check_contiguous(C_fp8, "C_fp8");
  check_same_device(C_fp8, A, "C_fp8", "A");
  check_cuda_tensor(SF_out, "SF_out");
  check_dtype(SF_out, dl_uint8, "SF_out");
  check_contiguous(SF_out, "SF_out");
  check_same_device(SF_out, A, "SF_out", "A");
  check_cuda_tensor(route_map, "route_map");
  check_dtype(route_map, dl_int32, "route_map");
  check_contiguous(route_map, "route_map");
  check_same_device(route_map, A, "route_map", "A");
  check_cuda_tensor(route_weights, "route_weights");
  check_dtype(route_weights, dl_float32, "route_weights");
  check_contiguous(route_weights, "route_weights");
  check_same_device(route_weights, A, "route_weights", "A");
  check_cuda_tensor(tile_expert, "tile_expert");
  check_dtype(tile_expert, dl_int32, "tile_expert");
  check_contiguous(tile_expert, "tile_expert");
  check_same_device(tile_expert, A, "tile_expert", "A");
  check_cuda_tensor(tile_m_local, "tile_m_local");
  check_dtype(tile_m_local, dl_int32, "tile_m_local");
  check_contiguous(tile_m_local, "tile_m_local");
  check_same_device(tile_m_local, A, "tile_m_local", "A");
  check_cuda_tensor(expert_row_offsets, "expert_row_offsets");
  check_dtype(expert_row_offsets, dl_int32, "expert_row_offsets");
  check_contiguous(expert_row_offsets, "expert_row_offsets");
  check_same_device(expert_row_offsets, A, "expert_row_offsets", "A");
  check_cuda_tensor(scale_gate, "scale_gate");
  check_dtype(scale_gate, dl_float32, "scale_gate");
  check_contiguous(scale_gate, "scale_gate");
  check_same_device(scale_gate, A, "scale_gate", "A");
  check_cuda_tensor(clamp_limit, "clamp_limit");
  check_dtype(clamp_limit, dl_float32, "clamp_limit");
  check_contiguous(clamp_limit, "clamp_limit");
  check_same_device(clamp_limit, A, "clamp_limit", "A");
  check_cuda_tensor(act_alpha, "act_alpha");
  check_dtype(act_alpha, dl_float32, "act_alpha");
  check_contiguous(act_alpha, "act_alpha");
  check_same_device(act_alpha, A, "act_alpha", "A");
  check_cuda_tensor(act_beta, "act_beta");
  check_dtype(act_beta, dl_float32, "act_beta");
  check_contiguous(act_beta, "act_beta");
  check_same_device(act_beta, A, "act_beta", "A");

  check_extents(A, {kLocalExperts, kPackedFc1Rows, kHidden}, "A");
  check_extents(B, {kLocalExperts, kMaxRowsPerExpert, kHidden}, "B");
  check_extents(SFA, {kLocalExperts, kW13ScaleTiles, kScaleTileRows256, kScaleTileColumns}, "SFA");
  check_extents(SFB, {kLocalExperts, kFc1InputScaleTiles, kScaleTileRows128, kScaleTileColumns},
                "SFB");
  check_extents(C_fp8, {kLocalExperts, kMaxRowsPerExpert, kIntermediate}, "C_fp8");
  check_numel(SF_out, kFc2InputScaleBytes, "SF_out");
  check_numel(route_map, kTotalExpertRows, "route_map");
  check_numel(route_weights, kMaxRoutes, "route_weights");
  check_numel(tile_expert, kFc1Tiles, "tile_expert");
  check_numel(tile_m_local, kFc1Tiles, "tile_m_local");
  check_numel(expert_row_offsets, kLocalExperts, "expert_row_offsets");
  check_numel(scale_gate, kLocalExperts, "scale_gate");
  check_numel(clamp_limit, kLocalExperts, "clamp_limit");
  check_numel(act_alpha, kLocalExperts, "act_alpha");
  check_numel(act_beta, kLocalExperts, "act_beta");
  TVM_FFI_CHECK(N == A.size(1), ValueError)
      << "N must equal the packed FC1 row count A.size(1)=" << A.size(1) << ", got " << N;
  TVM_FFI_CHECK(K_tiles * kK128 == A.size(2), ValueError)
      << "K_tiles * 128 must equal A.size(2)=" << A.size(2) << ", got K_tiles=" << K_tiles;
  TVM_FFI_CHECK(grid_m == tile_expert.numel(), ValueError)
      << "grid_m must equal the tile map length " << tile_expert.numel() << ", got " << grid_m;
  TVM_FFI_CHECK(grid_n >= 1 && grid_n <= 2147483647LL, ValueError)
      << "grid_n must be in [1, 2147483647], got " << grid_n;
  TVM_FFI_CHECK(rows_per_expert == B.size(1), ValueError)
      << "rows_per_expert must equal B.size(1)=" << B.size(1) << ", got " << rows_per_expert;
  TVM_FFI_CHECK(grid_x > 0 && grid_x <= 2147483647LL, ValueError)
      << "grid_x must be in [1, 2147483647], got " << grid_x;

  const cudaStream_t stream = get_stream(dev);

  // Opt the kernel into its dynamic shared memory once; function-local static
  // initialization already runs once and thread-safely.
  static const bool smem_ready = [] {
    TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_fc1),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicSmemBytes)));
    return true;
  }();
  (void)smem_ready;

  CUtensorMap tma_A = encode_weight(A, kW13BoxRows, "A");
  CUtensorMap tma_B = encode_operand(B, 32, "B");
  CUtensorMap tma_SFA = encode_scales(SFA, 32, "SFA");
  CUtensorMap tma_SFB = encode_scales(SFB, 16, "SFB");
  uint8_t* p_C_fp8 = static_cast<uint8_t*>(C_fp8.data_ptr());
  uint8_t* p_SF_out = static_cast<uint8_t*>(SF_out.data_ptr());
  int* p_route_map = static_cast<int*>(route_map.data_ptr());
  float* p_route_weights = static_cast<float*>(route_weights.data_ptr());
  int* p_tile_expert = static_cast<int*>(tile_expert.data_ptr());
  int* p_tile_m_local = static_cast<int*>(tile_m_local.data_ptr());
  int* p_expert_row_offsets = static_cast<int*>(expert_row_offsets.data_ptr());
  float* p_scale_gate = static_cast<float*>(scale_gate.data_ptr());
  float* p_clamp_limit = static_cast<float*>(clamp_limit.data_ptr());
  float* p_act_alpha = static_cast<float*>(act_alpha.data_ptr());
  float* p_act_beta = static_cast<float*>(act_beta.data_ptr());
  int32_t v_N = static_cast<int32_t>(N);
  int32_t v_K_tiles = static_cast<int32_t>(K_tiles);
  int32_t v_grid_m = static_cast<int32_t>(grid_m);
  int32_t v_grid_n = static_cast<int32_t>(grid_n);
  int32_t v_rows_per_expert = static_cast<int32_t>(rows_per_expert);
  void* kargs[] = {&tma_A,
                   &tma_B,
                   &tma_SFA,
                   &tma_SFB,
                   &p_C_fp8,
                   &p_SF_out,
                   &p_route_map,
                   &p_route_weights,
                   &p_tile_expert,
                   &p_tile_m_local,
                   &p_expert_row_offsets,
                   &p_scale_gate,
                   &p_clamp_limit,
                   &p_act_alpha,
                   &p_act_beta,
                   &v_N,
                   &v_K_tiles,
                   &v_grid_m,
                   &v_grid_n,
                   &v_rows_per_expert};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  // Programmatic dependent launch (manifest launch.use_pdl): the program
  // prefills its first expert-weight tile through the static tile map, then
  // executes griddepcontrol.wait before it reads local_pack's packed operands
  // and route map, so the attribute only overlaps its prologue with the
  // primary's tail. In eager streams and under CUDA-graph capture this launch
  // is the local_pack -> fc1 programmatic edge.
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_fc1), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_fc1) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_fc1

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fc1, cake_w8a8_megamoe_ep16_fc1::Run);
