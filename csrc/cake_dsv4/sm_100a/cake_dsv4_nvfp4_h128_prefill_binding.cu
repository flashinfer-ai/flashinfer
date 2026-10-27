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

// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_dsv4_nvfp4_h128_prefill_sm100a'.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

#include "common/cake_dsv4_nvfp4_host_common.cuh"
#include "common/cake_dsv4_nvfp4_prefill_host_common.cuh"

extern "C" __global__ void kernel_cake_dsv4_nvfp4_h128_prefill_sm100a(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, const __grid_constant__ CUtensorMap tmap_o, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int max_q_len, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens);


namespace cake_host_shim_af33cc401fb404a7 {

using namespace cake_host_shim_common;
using tvm::ffi::TensorView;

void Run(TensorView arg_tmap_q, TensorView arg_tmap_q_sf, TensorView arg_tmap_swa_kv, TensorView arg_tmap_compressed_kv, TensorView arg_tmap_swa_sf, TensorView arg_tmap_compressed_sf, TensorView arg_tmap_o, TensorView arg_O, TensorView arg_LSE, TensorView arg_swa_indices, TensorView arg_compressed_indices, TensorView arg_sparse_topk_lens, TensorView arg_sinks, TensorView arg_bmm1_scale, TensorView arg_bmm2_scale, int64_t arg_num_heads, int64_t arg_swa_index_stride, int64_t arg_compressed_index_stride, int64_t arg_sparse_topk_lens_offset, int64_t arg_num_query_tokens, int64_t arg_sparse_topk, int64_t arg_has_sinks, int64_t arg_total_work_items, int64_t arg_max_q_len, int64_t arg_swa_page_log2, int64_t arg_swa_pitch_units, int64_t arg_swa_footer_units, int64_t arg_compressed_page_log2, int64_t arg_compressed_pitch_units, int64_t arg_compressed_footer_units, int64_t arg_swa_width, int64_t arg_compressed_width, TensorView arg_extra_topk_lens, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_tmap_q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  {
    // Bind the primary context for the tensor-map encoders once per thread and device.
    static thread_local int encoder_context_device = -1;
    if (encoder_context_device != dev.device_id) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      encoder_context_device = dev.device_id;
    }
  }
  check_cuda_tensor(arg_tmap_q, "tmap_q");
  check_dtype(arg_tmap_q, DLDataType{kDLUInt, 8, 1}, "tmap_q");
  check_contiguous(arg_tmap_q, "tmap_q");
  check_cuda_tensor(arg_tmap_q_sf, "tmap_q_sf");
  check_dtype(arg_tmap_q_sf, DLDataType{kDLUInt, 8, 1}, "tmap_q_sf");
  check_contiguous(arg_tmap_q_sf, "tmap_q_sf");
  check_cuda_tensor(arg_tmap_swa_kv, "tmap_swa_kv");
  check_dtype(arg_tmap_swa_kv, DLDataType{kDLUInt, 8, 1}, "tmap_swa_kv");
  check_cuda_tensor(arg_tmap_compressed_kv, "tmap_compressed_kv");
  check_dtype(arg_tmap_compressed_kv, DLDataType{kDLUInt, 8, 1}, "tmap_compressed_kv");
  check_cuda_tensor(arg_tmap_swa_sf, "tmap_swa_sf");
  check_dtype(arg_tmap_swa_sf, DLDataType{kDLUInt, 8, 1}, "tmap_swa_sf");
  check_cuda_tensor(arg_tmap_compressed_sf, "tmap_compressed_sf");
  check_dtype(arg_tmap_compressed_sf, DLDataType{kDLUInt, 8, 1}, "tmap_compressed_sf");
  check_cuda_tensor(arg_tmap_o, "tmap_o");
  check_dtype(arg_tmap_o, DLDataType{kDLBfloat, 16, 1}, "tmap_o");
  check_contiguous(arg_tmap_o, "tmap_o");
  check_cuda_tensor(arg_O, "O");
  check_dtype(arg_O, DLDataType{kDLBfloat, 16, 1}, "O");
  check_contiguous(arg_O, "O");
  check_cuda_tensor(arg_LSE, "LSE");
  check_dtype(arg_LSE, DLDataType{kDLFloat, 32, 1}, "LSE");
  check_contiguous(arg_LSE, "LSE");
  check_cuda_tensor(arg_swa_indices, "swa_indices");
  check_dtype(arg_swa_indices, DLDataType{kDLInt, 32, 1}, "swa_indices");
  check_cuda_tensor(arg_compressed_indices, "compressed_indices");
  check_dtype(arg_compressed_indices, DLDataType{kDLInt, 32, 1}, "compressed_indices");
  check_cuda_tensor(arg_sparse_topk_lens, "sparse_topk_lens");
  check_dtype(arg_sparse_topk_lens, DLDataType{kDLInt, 32, 1}, "sparse_topk_lens");
  check_contiguous(arg_sparse_topk_lens, "sparse_topk_lens");
  check_cuda_tensor(arg_sinks, "sinks");
  check_dtype(arg_sinks, DLDataType{kDLFloat, 32, 1}, "sinks");
  check_contiguous(arg_sinks, "sinks");
  check_cuda_tensor(arg_bmm1_scale, "bmm1_scale");
  check_dtype(arg_bmm1_scale, DLDataType{kDLFloat, 32, 1}, "bmm1_scale");
  check_contiguous(arg_bmm1_scale, "bmm1_scale");
  check_cuda_tensor(arg_bmm2_scale, "bmm2_scale");
  check_dtype(arg_bmm2_scale, DLDataType{kDLFloat, 32, 1}, "bmm2_scale");
  check_contiguous(arg_bmm2_scale, "bmm2_scale");
  check_cuda_tensor(arg_extra_topk_lens, "extra_topk_lens");
  check_dtype(arg_extra_topk_lens, DLDataType{kDLInt, 32, 1}, "extra_topk_lens");
  check_contiguous(arg_extra_topk_lens, "extra_topk_lens");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_swa_index_stride >= -2147483648LL && arg_swa_index_stride <= 2147483647LL, ValueError)
      << "scalar 'swa_index_stride' value " << arg_swa_index_stride
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_compressed_index_stride >= -2147483648LL && arg_compressed_index_stride <= 2147483647LL, ValueError)
      << "scalar 'compressed_index_stride' value " << arg_compressed_index_stride
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sparse_topk_lens_offset >= -2147483648LL && arg_sparse_topk_lens_offset <= 2147483647LL, ValueError)
      << "scalar 'sparse_topk_lens_offset' value " << arg_sparse_topk_lens_offset
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_query_tokens >= -2147483648LL && arg_num_query_tokens <= 2147483647LL, ValueError)
      << "scalar 'num_query_tokens' value " << arg_num_query_tokens
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sparse_topk >= -2147483648LL && arg_sparse_topk <= 2147483647LL, ValueError)
      << "scalar 'sparse_topk' value " << arg_sparse_topk
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_has_sinks >= -2147483648LL && arg_has_sinks <= 2147483647LL, ValueError)
      << "scalar 'has_sinks' value " << arg_has_sinks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_work_items >= -2147483648LL && arg_total_work_items <= 2147483647LL, ValueError)
      << "scalar 'total_work_items' value " << arg_total_work_items
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_q_len >= -2147483648LL && arg_max_q_len <= 2147483647LL, ValueError)
      << "scalar 'max_q_len' value " << arg_max_q_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_swa_page_log2 >= -2147483648LL && arg_swa_page_log2 <= 2147483647LL, ValueError)
      << "scalar 'swa_page_log2' value " << arg_swa_page_log2
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_swa_pitch_units >= -2147483648LL && arg_swa_pitch_units <= 2147483647LL, ValueError)
      << "scalar 'swa_pitch_units' value " << arg_swa_pitch_units
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_swa_footer_units >= -2147483648LL && arg_swa_footer_units <= 2147483647LL, ValueError)
      << "scalar 'swa_footer_units' value " << arg_swa_footer_units
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_compressed_page_log2 >= -2147483648LL && arg_compressed_page_log2 <= 2147483647LL, ValueError)
      << "scalar 'compressed_page_log2' value " << arg_compressed_page_log2
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_compressed_pitch_units >= -2147483648LL && arg_compressed_pitch_units <= 2147483647LL, ValueError)
      << "scalar 'compressed_pitch_units' value " << arg_compressed_pitch_units
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_compressed_footer_units >= -2147483648LL && arg_compressed_footer_units <= 2147483647LL, ValueError)
      << "scalar 'compressed_footer_units' value " << arg_compressed_footer_units
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_swa_width >= -2147483648LL && arg_swa_width <= 2147483647LL, ValueError)
      << "scalar 'swa_width' value " << arg_swa_width
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_compressed_width >= -2147483648LL && arg_compressed_width <= 2147483647LL, ValueError)
      << "scalar 'compressed_width' value " << arg_compressed_width
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_tmap_q_sf, arg_tmap_q, "tmap_q_sf", "tmap_q");
  check_same_device(arg_tmap_swa_kv, arg_tmap_q, "tmap_swa_kv", "tmap_q");
  check_same_device(arg_tmap_compressed_kv, arg_tmap_q, "tmap_compressed_kv", "tmap_q");
  check_same_device(arg_tmap_swa_sf, arg_tmap_q, "tmap_swa_sf", "tmap_q");
  check_same_device(arg_tmap_compressed_sf, arg_tmap_q, "tmap_compressed_sf", "tmap_q");
  check_same_device(arg_tmap_o, arg_tmap_q, "tmap_o", "tmap_q");
  check_same_device(arg_O, arg_tmap_q, "O", "tmap_q");
  check_same_device(arg_LSE, arg_tmap_q, "LSE", "tmap_q");
  check_same_device(arg_swa_indices, arg_tmap_q, "swa_indices", "tmap_q");
  check_same_device(arg_compressed_indices, arg_tmap_q, "compressed_indices", "tmap_q");
  check_same_device(arg_sparse_topk_lens, arg_tmap_q, "sparse_topk_lens", "tmap_q");
  check_same_device(arg_sinks, arg_tmap_q, "sinks", "tmap_q");
  check_same_device(arg_bmm1_scale, arg_tmap_q, "bmm1_scale", "tmap_q");
  check_same_device(arg_bmm2_scale, arg_tmap_q, "bmm2_scale", "tmap_q");
  check_same_device(arg_extra_topk_lens, arg_tmap_q, "extra_topk_lens", "tmap_q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  TVM_FFI_CHECK(grid_x % 2 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_tmap_q = EncodeTmaPrefill_tmap_q(arg_tmap_q);
  CUtensorMap p_tmap_q_sf = EncodeTmaPrefill_tmap_q_sf(arg_tmap_q_sf);
  CUtensorMap p_tmap_swa_kv = EncodeTmaPrefill_tmap_swa_kv(arg_tmap_swa_kv);
  CUtensorMap p_tmap_compressed_kv = EncodeTmaPrefill_tmap_compressed_kv(arg_tmap_compressed_kv);
  CUtensorMap p_tmap_swa_sf = EncodeTmaPrefill_tmap_swa_sf(arg_tmap_swa_sf);
  CUtensorMap p_tmap_compressed_sf = EncodeTmaPrefill_tmap_compressed_sf(arg_tmap_compressed_sf);
  CUtensorMap p_tmap_o = EncodeTmaPrefill_tmap_o(arg_tmap_o);
  __nv_bfloat16* p_O = static_cast<__nv_bfloat16*>(static_cast<void*>(static_cast<char*>(arg_O.data_ptr()) + arg_O.byte_offset()));
  float* p_LSE = static_cast<float*>(static_cast<void*>(static_cast<char*>(arg_LSE.data_ptr()) + arg_LSE.byte_offset()));
  int* p_swa_indices = static_cast<int*>(static_cast<void*>(static_cast<char*>(arg_swa_indices.data_ptr()) + arg_swa_indices.byte_offset()));
  int* p_compressed_indices = static_cast<int*>(static_cast<void*>(static_cast<char*>(arg_compressed_indices.data_ptr()) + arg_compressed_indices.byte_offset()));
  int* p_sparse_topk_lens = static_cast<int*>(static_cast<void*>(static_cast<char*>(arg_sparse_topk_lens.data_ptr()) + arg_sparse_topk_lens.byte_offset()));
  float* p_sinks = static_cast<float*>(static_cast<void*>(static_cast<char*>(arg_sinks.data_ptr()) + arg_sinks.byte_offset()));
  float* p_bmm1_scale = static_cast<float*>(static_cast<void*>(static_cast<char*>(arg_bmm1_scale.data_ptr()) + arg_bmm1_scale.byte_offset()));
  float* p_bmm2_scale = static_cast<float*>(static_cast<void*>(static_cast<char*>(arg_bmm2_scale.data_ptr()) + arg_bmm2_scale.byte_offset()));
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_swa_index_stride = (int32_t)arg_swa_index_stride;
  int32_t v_compressed_index_stride = (int32_t)arg_compressed_index_stride;
  int32_t v_sparse_topk_lens_offset = (int32_t)arg_sparse_topk_lens_offset;
  int32_t v_num_query_tokens = (int32_t)arg_num_query_tokens;
  int32_t v_sparse_topk = (int32_t)arg_sparse_topk;
  int32_t v_has_sinks = (int32_t)arg_has_sinks;
  int32_t v_total_work_items = (int32_t)arg_total_work_items;
  int32_t v_max_q_len = (int32_t)arg_max_q_len;
  int32_t v_swa_page_log2 = (int32_t)arg_swa_page_log2;
  int32_t v_swa_pitch_units = (int32_t)arg_swa_pitch_units;
  int32_t v_swa_footer_units = (int32_t)arg_swa_footer_units;
  int32_t v_compressed_page_log2 = (int32_t)arg_compressed_page_log2;
  int32_t v_compressed_pitch_units = (int32_t)arg_compressed_pitch_units;
  int32_t v_compressed_footer_units = (int32_t)arg_compressed_footer_units;
  int32_t v_swa_width = (int32_t)arg_swa_width;
  int32_t v_compressed_width = (int32_t)arg_compressed_width;
  int* p_extra_topk_lens = static_cast<int*>(static_cast<void*>(static_cast<char*>(arg_extra_topk_lens.data_ptr()) + arg_extra_topk_lens.byte_offset()));
  void* kargs[] = {&p_tmap_q, &p_tmap_q_sf, &p_tmap_swa_kv, &p_tmap_compressed_kv, &p_tmap_swa_sf, &p_tmap_compressed_sf, &p_tmap_o, &p_O, &p_LSE, &p_swa_indices, &p_compressed_indices, &p_sparse_topk_lens, &p_sinks, &p_bmm1_scale, &p_bmm2_scale, &v_num_heads, &v_swa_index_stride, &v_compressed_index_stride, &v_sparse_topk_lens_offset, &v_num_query_tokens, &v_sparse_topk, &v_has_sinks, &v_total_work_items, &v_max_q_len, &v_swa_page_log2, &v_swa_pitch_units, &v_swa_footer_units, &v_compressed_page_log2, &v_compressed_pitch_units, &v_compressed_footer_units, &v_swa_width, &v_compressed_width, &p_extra_topk_lens};

  static const bool smem_ready = CakeSetMaxDynamicSmem(
      reinterpret_cast<const void*>(kernel_cake_dsv4_nvfp4_h128_prefill_sm100a), 227328);
  (void)smem_ready;
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(640u, 1u, 1u);

  // Launch attributes of this kernel; the launch itself is the export's shared CakeLaunchDirect.
  cudaLaunchAttribute attrs[2]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = 2u;
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
  CakeLaunchDirect(reinterpret_cast<const void*>(kernel_cake_dsv4_nvfp4_h128_prefill_sm100a), "kernel_cake_dsv4_nvfp4_h128_prefill_sm100a", grid, block, 227328u, stream, kargs, attrs, n);
}

}  // namespace cake_host_shim_af33cc401fb404a7

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_af33cc401fb404a7::Run);
