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
 *
 * Derived from the per-program host shims of NVlabs/kda (260927-kda-for-kda),
 * Copyright (c) 2026 KDA Team, MIT License (LICENSE.kda-for-kda.txt).
 */

// TVM FFI host launcher shared by the four retained SM103a KDA PTX programs.
//
// flashinfer/kda_kernels/ptx/static_runtime.py prepends one #define per entry
// of csrc/kda/ptx/programs.json before compiling this file for a program:
//
//   KDA_PTX_MODULE_IDENT        embedded cubin identifier (token)
//   KDA_PTX_KERNEL_NAME         kernel symbol (string literal)
//   KDA_PTX_DYNAMIC_SMEM_BYTES  dynamic shared memory per CTA
//   KDA_PTX_V_BOX_ROWS          token rows per TMA box of v (32 or 16)
//   KDA_PTX_OUT_BOX_DEPTH       64-column tiles per TMA box of out (1 or 2)
//   KDA_PTX_HANDOFF_FLAGS       1 when the kernel takes a handoff-flag address
//   KDA_PTX_CLUSTER_X           CTA cluster width (1 or 2)
//
// Every program shares one argument plan. A program without handoff flags
// accepts and ignores that scalar so the host passes the same argument list
// to all four.

#include <cuda.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#define KDA_PTX_EMBED_CUBIN_(ident) TVM_FFI_EMBED_CUBIN(ident)
#define KDA_PTX_EMBED_CUBIN(ident) KDA_PTX_EMBED_CUBIN_(ident)
#define KDA_PTX_EMBED_MODULE_(ident) EmbedCubinModule_##ident
#define KDA_PTX_EMBED_MODULE(ident) KDA_PTX_EMBED_MODULE_(ident)

KDA_PTX_EMBED_CUBIN(KDA_PTX_MODULE_IDENT);

namespace ptx_host_shim {

using tvm::ffi::TensorView;

constexpr int kNumTmaBuffers = 6;
constexpr uint64_t kBf16Bytes = 2;

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(const TensorView& t, const TensorView& reference,
                                const char* name, const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name << ": got cuda:"
      << t.device().device_id << " versus cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && d.lanes == 1, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=1), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError) << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis) << ", expected "
          << expected;
    }
  }
}

// Per-launch TMA descriptor table. The caller owns a device buffer with one
// 64-byte-aligned CUtensorMap slot per TMA argument. An upload call (outside
// CUDA Graph capture) encodes every descriptor and copies the table on the
// launch stream; launch calls pass pointers into it and do no host-to-device
// work. The kernels acquire each descriptor with fence.proxy.tensormap before
// use, so a table may be rewritten between launches.
inline void UploadTmaTable(char* table, const CUtensorMap* maps, size_t count,
                           cudaStream_t stream) {
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUresult result =
      cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuStreamIsCapturing for the TMA descriptor table failed: CUresult="
      << static_cast<int>(result);
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "the TMA descriptor table must be uploaded outside CUDA Graph capture";
  result = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(table), maps,
                             count * sizeof(CUtensorMap), reinterpret_cast<CUstream>(stream));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuMemcpyHtoDAsync for the TMA descriptor table failed: CUresult="
      << static_cast<int>(result);
}

inline void CheckTmaSource(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
}

// Encode one BF16 tiled descriptor. The box may exceed the global extent only
// along `token_dim`, where TMA zero-fills out-of-bounds rows; `token_dim < 0`
// checks every dimension.
inline CUtensorMap EncodeTiled(const char* name, uint32_t rank, void* base,
                               const uint64_t* global_dim, const uint64_t* global_strides,
                               const uint32_t* box_dim, int token_dim,
                               CUtensorMapSwizzle swizzle) {
  for (uint32_t i = 0; i < rank; ++i) {
    TVM_FFI_CHECK(global_dim[i] > 0, ValueError)
        << "TMA descriptor for '" << name << "' resolved a non-positive global dim";
    TVM_FFI_CHECK((int)i == token_dim || box_dim[i] <= global_dim[i], ValueError)
        << "TMA box exceeds resolved global dims for '" << name << "'";
  }
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, base, global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (" << rank << "D, '" << name << "') failed: CUresult=" << (int)r;
  return tm;
}

// q/k/out [tokens, heads, 128] as 4D (64-column tile, token, head, tile).
inline CUtensorMap EncodeHeadTiles(const TensorView& t, const char* name, uint32_t box_depth) {
  CheckTmaSource(t, name);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  int64_t tokens = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source '" << name << "' extent " << d1 << " must divide exactly by 64";
  uint64_t global_dim[4] = {64u, (uint64_t)tokens, (uint64_t)d2, (uint64_t)(d1 / 64)};
  uint64_t global_strides[3] = {(uint64_t)(d2 * d1) * kBf16Bytes, (uint64_t)d1 * kBf16Bytes,
                                64u * kBf16Bytes};
  uint32_t box_dim[4] = {64u, 32u, 1u, box_depth};
  return EncodeTiled(name, 4, t.data_ptr(), global_dim, global_strides, box_dim, 1,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// v/g [tokens, heads, 128] as 3D (column, head, token).
inline CUtensorMap EncodeRows(const TensorView& t, const char* name, uint32_t box_cols,
                              uint32_t box_rows, CUtensorMapSwizzle swizzle) {
  CheckTmaSource(t, name);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  int64_t tokens = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)d1, (uint64_t)d2, (uint64_t)tokens};
  uint64_t global_strides[2] = {(uint64_t)d1 * kBf16Bytes, (uint64_t)(d1 * d2) * kBf16Bytes};
  uint32_t box_dim[3] = {box_cols, 1u, box_rows};
  return EncodeTiled(name, 3, t.data_ptr(), global_dim, global_strides, box_dim, 2, swizzle);
}

// beta [tokens, heads (padded to 8)] as 2D (head, token); the head stride is
// the physical step so a padded copy can be bound.
inline CUtensorMap EncodeBeta(const TensorView& t) {
  const char* name = "beta_tma";
  CheckTmaSource(t, name);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source '" << name << "' trailing dims must be positive";
  int64_t tokens = t.numel() / d1;
  CheckDenseLeadingFold(t, 1, name);
  int64_t s2 = t.stride(t.ndim() - 2);
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source '" << name << "' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)d1, (uint64_t)tokens};
  uint64_t global_strides[1] = {(uint64_t)s2 * kBf16Bytes};
  uint32_t box_dim[2] = {8u, 32u};
  return EncodeTiled(name, 2, t.data_ptr(), global_dim, global_strides, box_dim, -1,
                     CU_TENSOR_MAP_SWIZZLE_NONE);
}

struct TensorContract {
  const TensorView* tensor;
  const char* name;
  int code;
  int bits;
  bool contiguous;
};

void Run(TensorView arg_q, TensorView arg_q_tma, TensorView arg_k, TensorView arg_k_tma,
         TensorView arg_v, TensorView arg_v_tma, TensorView arg_g, TensorView arg_g_tma,
         TensorView arg_beta, TensorView arg_beta_tma, TensorView arg_A_log,
         TensorView arg_dt_bias, TensorView arg_piece_table, TensorView arg_cta_table,
         TensorView arg_initial_state, TensorView arg_out, TensorView arg_out_tma,
         TensorView arg_final_state, int64_t arg_num_heads, double arg_scale,
         double arg_lower_bound, int64_t arg_failure_addr, int64_t arg_generation,
         double arg_detect_threshold_log2, int64_t arg_handoff_flags_addr,
         TensorView arg_tma_table, int64_t arg_tma_upload, int64_t grid_x, int64_t grid_y,
         int64_t grid_z) {
  const TensorContract contracts[] = {
      {&arg_q, "q", kDLBfloat, 16, true},
      {&arg_q_tma, "q_tma", kDLBfloat, 16, true},
      {&arg_k, "k", kDLBfloat, 16, true},
      {&arg_k_tma, "k_tma", kDLBfloat, 16, true},
      {&arg_v, "v", kDLBfloat, 16, true},
      {&arg_v_tma, "v_tma", kDLBfloat, 16, true},
      {&arg_g, "g", kDLBfloat, 16, true},
      {&arg_g_tma, "g_tma", kDLBfloat, 16, true},
      {&arg_beta, "beta", kDLBfloat, 16, true},
      {&arg_beta_tma, "beta_tma", kDLBfloat, 16, false},
      {&arg_A_log, "A_log", kDLFloat, 32, true},
      {&arg_dt_bias, "dt_bias", kDLFloat, 32, true},
      {&arg_piece_table, "piece_table", kDLInt, 32, true},
      {&arg_cta_table, "cta_table", kDLInt, 32, true},
      {&arg_initial_state, "initial_state", kDLFloat, 32, true},
      {&arg_out, "out", kDLBfloat, 16, true},
      {&arg_out_tma, "out_tma", kDLBfloat, 16, true},
      {&arg_final_state, "final_state", kDLFloat, 32, true},
      {&arg_tma_table, "tma_table", kDLUInt, 8, true},
  };
  for (const TensorContract& c : contracts) {
    CheckCudaTensor(*c.tensor, c.name);
    CheckDtype(*c.tensor, c.name, c.code, c.bits);
    if (c.contiguous) {
      CheckContiguous(*c.tensor, c.name);
    }
    if (c.tensor != &arg_q) {
      CheckSameCudaDevice(*c.tensor, arg_q, c.name, "q");
    }
  }
  TVM_FFI_CHECK(arg_num_heads >= INT32_MIN && arg_num_heads <= INT32_MAX, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads << " is outside i32 range";
  TVM_FFI_CHECK(arg_generation >= INT32_MIN && arg_generation <= INT32_MAX, ValueError)
      << "scalar 'generation' value " << arg_generation << " is outside i32 range";
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y << ", "
      << grid_z << ")";
#if KDA_PTX_CLUSTER_X > 1
  TVM_FFI_CHECK(grid_x % KDA_PTX_CLUSTER_X == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (" << KDA_PTX_CLUSTER_X << ", 1, 1)";
#endif
  TVM_FFI_CHECK(arg_tma_table.numel() >= kNumTmaBuffers * (int64_t)sizeof(CUtensorMap) &&
                    reinterpret_cast<uintptr_t>(arg_tma_table.data_ptr()) % 64 == 0,
                ValueError)
      << "tma_table must hold " << kNumTmaBuffers << " 64-byte-aligned CUtensorMap slots";

  DLDevice dev = arg_q.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  char* tma_slots = static_cast<char*>(arg_tma_table.data_ptr());
  if (arg_tma_upload) {
    CUtensorMap tma_maps[kNumTmaBuffers] = {
        EncodeHeadTiles(arg_q_tma, "q_tma", 2u),
        EncodeHeadTiles(arg_k_tma, "k_tma", 2u),
        EncodeRows(arg_v_tma, "v_tma", 64u, KDA_PTX_V_BOX_ROWS, CU_TENSOR_MAP_SWIZZLE_128B),
        EncodeRows(arg_g_tma, "g_tma", 128u, 32u, CU_TENSOR_MAP_SWIZZLE_NONE),
        EncodeBeta(arg_beta_tma),
        EncodeHeadTiles(arg_out_tma, "out_tma", KDA_PTX_OUT_BOX_DEPTH),
    };
    UploadTmaTable(tma_slots, tma_maps, kNumTmaBuffers, stream);
    return;
  }

  void* p_q = arg_q.data_ptr();
  void* p_q_tma = tma_slots + 0 * sizeof(CUtensorMap);
  void* p_k = arg_k.data_ptr();
  void* p_k_tma = tma_slots + 1 * sizeof(CUtensorMap);
  void* p_v = arg_v.data_ptr();
  void* p_v_tma = tma_slots + 2 * sizeof(CUtensorMap);
  void* p_g = arg_g.data_ptr();
  void* p_g_tma = tma_slots + 3 * sizeof(CUtensorMap);
  void* p_beta = arg_beta.data_ptr();
  void* p_beta_tma = tma_slots + 4 * sizeof(CUtensorMap);
  void* p_A_log = arg_A_log.data_ptr();
  void* p_dt_bias = arg_dt_bias.data_ptr();
  void* p_piece_table = arg_piece_table.data_ptr();
  void* p_cta_table = arg_cta_table.data_ptr();
  void* p_initial_state = arg_initial_state.data_ptr();
  void* p_out = arg_out.data_ptr();
  void* p_out_tma = tma_slots + 5 * sizeof(CUtensorMap);
  void* p_final_state = arg_final_state.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  float v_scale = (float)arg_scale;
  float v_lower_bound = (float)arg_lower_bound;
  int64_t v_failure_addr = arg_failure_addr;
  int32_t v_generation = (int32_t)arg_generation;
  float v_detect_threshold_log2 = (float)arg_detect_threshold_log2;
  int64_t v_handoff_flags_addr = arg_handoff_flags_addr;
  (void)v_handoff_flags_addr;
  void* kargs[] = {&p_q,
                   &p_q_tma,
                   &p_k,
                   &p_k_tma,
                   &p_v,
                   &p_v_tma,
                   &p_g,
                   &p_g_tma,
                   &p_beta,
                   &p_beta_tma,
                   &p_A_log,
                   &p_dt_bias,
                   &p_piece_table,
                   &p_cta_table,
                   &p_initial_state,
                   &p_out,
                   &p_out_tma,
                   &p_final_state,
                   &v_num_heads,
                   &v_scale,
                   &v_lower_bound,
                   &v_failure_addr,
                   &v_generation,
                   &v_detect_threshold_log2,
#if KDA_PTX_HANDOFF_FLAGS
                   &v_handoff_flags_addr,
#endif
  };

  static auto kernel =
      KDA_PTX_EMBED_MODULE(KDA_PTX_MODULE_IDENT)::Global()->mod.GetKernelWithMaxDynamicSharedMemory(
          KDA_PTX_KERNEL_NAME, KDA_PTX_DYNAMIC_SMEM_BYTES);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(1024u, 1u, 1u);

#if KDA_PTX_CLUSTER_X > 1
  tvm::ffi::cuda_api::LaunchConfig config;
  int n = 0;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attrs[2];
  attrs[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[n].value.clusterDim.x = KDA_PTX_CLUSTER_X;
  attrs[n].value.clusterDim.y = 1u;
  attrs[n].value.clusterDim.z = 1u;
  ++n;
  attrs[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  attrs[n].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  ++n;
  config.gridDimX = grid.x;
  config.gridDimY = grid.y;
  config.gridDimZ = grid.z;
  config.blockDimX = block.x;
  config.blockDimY = block.y;
  config.blockDimZ = block.z;
  config.sharedMemBytes = KDA_PTX_DYNAMIC_SMEM_BYTES;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#else
  cudaLaunchAttribute attrs[2];
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = KDA_PTX_CLUSTER_X;
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
  config.gridDim = {grid.x, grid.y, grid.z};
  config.blockDim = {block.x, block.y, block.z};
  config.dynamicSmemBytes = KDA_PTX_DYNAMIC_SMEM_BYTES;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(kargs, config));
#else
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(kargs, grid, block, stream, KDA_PTX_DYNAMIC_SMEM_BYTES));
#endif
}

}  // namespace ptx_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, ptx_host_shim::Run);
