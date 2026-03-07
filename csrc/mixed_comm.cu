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
#include <array>
#include <set>
#include <tuple>

#include "flashinfer/comm/mixed_comm.cuh"
#include "tvm/ffi/container/map.h"
#include "tvm_ffi_utils.h"

namespace {

using flashinfer::mixed_comm::ACCESS_BYTES;
using flashinfer::mixed_comm::ceil_div;
using flashinfer::mixed_comm::fused_allreduce_allgather_multi_node;
using flashinfer::mixed_comm::fused_allreduce_allgather_single_node;
using flashinfer::mixed_comm::fused_reducescatter_allreduce_multi_node;
using flashinfer::mixed_comm::fused_reducescatter_allreduce_single_node;
using flashinfer::mixed_comm::is_mixed_mode;
using flashinfer::mixed_comm::is_opt_bytes_mode;
using flashinfer::mixed_comm::is_opt_waits_mode;
using flashinfer::mixed_comm::MixedCommArgs;
using flashinfer::mixed_comm::MixedCommMode;
using flashinfer::mixed_comm::mod;
using flashinfer::mixed_comm::NUM_MIXED_COMM_MODES;
using flashinfer::mixed_comm::round_down;
using flashinfer::mixed_comm::round_up;
using flashinfer::mixed_comm::WARP_SIZE;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;

thread_local int dummy_smem_size = -1;

int get_dummy_smem_size(int device_id) {
  // Use the dummy shared memory size to avoid two CTAs running on the same SM
  if (dummy_smem_size == -1) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    int max_smem_size = device_prop.sharedMemPerMultiprocessor;
    int resv_smem_size = device_prop.reservedSharedMemPerBlock;
    dummy_smem_size = std::min(max_smem_size / 2, max_smem_size - resv_smem_size);
  }
  return dummy_smem_size;
}

template <int elems_per_thd>
size_t get_num_accesses(size_t num_elems, int grid_size, int local_tp_size, int inter_tp_size,
                        int dp_size, int mode) {
  size_t num_accesses = ceil_div(num_elems, grid_size * elems_per_thd);
  if (is_opt_bytes_mode(static_cast<MixedCommMode>(mode))) {
    num_accesses = round_up(num_accesses, local_tp_size * inter_tp_size);
  } else if (is_mixed_mode(static_cast<MixedCommMode>(mode))) {
    num_accesses = round_up(num_accesses, local_tp_size);
  }
  num_accesses *= dp_size;
  return num_accesses;
}

template <typename T>
int get_max_block_size(T* kernel) {
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return round_down<WARP_SIZE>(attr.maxThreadsPerBlock);
}

template <typename T>
int get_block_size(T* kernel, Optional<int> block_size_raw, size_t num_accesses, int num_nodes) {
  int max_block_size = get_max_block_size(kernel);
  TVM_FFI_ICHECK_GE(max_block_size, num_nodes)
      << "max_block_size must be greater than or equal to num_nodes";
  int block_size;
  if (block_size_raw.has_value()) {
    block_size = block_size_raw.value();
    TVM_FFI_ICHECK_EQ(mod<WARP_SIZE>(block_size), 0) << "block_size must be divisible by WARP_SIZE";
    TVM_FFI_ICHECK_GE(block_size, num_nodes)
        << "block_size must be greater than or equal to num_nodes";
    TVM_FFI_ICHECK_LE(block_size, max_block_size)
        << "block_size must be less than or equal to max_block_size";
  } else {
    block_size =
        round_up<WARP_SIZE>(ceil_div(num_accesses, ceil_div(num_accesses, max_block_size)));
  }
  return block_size;
}

#define DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, ...)                 \
  [&]() -> bool {                                                                    \
    if (local_tp_size == 1 && local_dp_size == 2) {                                  \
      constexpr int LOCAL_TP_SIZE = 1;                                               \
      constexpr int LOCAL_DP_SIZE = 2;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 2 && local_dp_size == 1) {                           \
      constexpr int LOCAL_TP_SIZE = 2;                                               \
      constexpr int LOCAL_DP_SIZE = 1;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 1 && local_dp_size == 4) {                           \
      constexpr int LOCAL_TP_SIZE = 1;                                               \
      constexpr int LOCAL_DP_SIZE = 4;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 2 && local_dp_size == 2) {                           \
      constexpr int LOCAL_TP_SIZE = 2;                                               \
      constexpr int LOCAL_DP_SIZE = 2;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 4 && local_dp_size == 1) {                           \
      constexpr int LOCAL_TP_SIZE = 4;                                               \
      constexpr int LOCAL_DP_SIZE = 1;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 1 && local_dp_size == 8) {                           \
      constexpr int LOCAL_TP_SIZE = 1;                                               \
      constexpr int LOCAL_DP_SIZE = 8;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 2 && local_dp_size == 4) {                           \
      constexpr int LOCAL_TP_SIZE = 2;                                               \
      constexpr int LOCAL_DP_SIZE = 4;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 4 && local_dp_size == 2) {                           \
      constexpr int LOCAL_TP_SIZE = 4;                                               \
      constexpr int LOCAL_DP_SIZE = 2;                                               \
      return __VA_ARGS__();                                                          \
    } else if (local_tp_size == 8 && local_dp_size == 1) {                           \
      constexpr int LOCAL_TP_SIZE = 8;                                               \
      constexpr int LOCAL_DP_SIZE = 1;                                               \
      return __VA_ARGS__();                                                          \
    } else {                                                                         \
      TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__                                   \
                            << " failed to dispatch local_tp_size=" << local_tp_size \
                            << " and local_dp_size=" << local_dp_size;               \
      return false;                                                                  \
    }                                                                                \
  }()

#define DISPATCH_MODE_SINGLE_NODE(mode, ...)                                                 \
  [&]() -> bool {                                                                            \
    switch (mode) {                                                                          \
      case MixedCommMode::OPT_WAITS_SIGNAL_MC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_WAITS_SIGNAL_MC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_SIGNAL_UC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_WAITS_SIGNAL_UC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT1_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT1_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT1_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT1_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_SIGNAL_MC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_BYTES_SIGNAL_MC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_SIGNAL_UC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_BYTES_SIGNAL_UC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT1_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT1_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT1_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT1_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      default: {                                                                             \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch mode=" << mode; \
        return false;                                                                        \
      }                                                                                      \
    }                                                                                        \
  }()

#define DISPATCH_MODE_MULTI_NODE(mode, ...)                                                  \
  [&]() -> bool {                                                                            \
    switch (mode) {                                                                          \
      case MixedCommMode::OPT_WAITS_SIGNAL_MC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_WAITS_SIGNAL_MC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_SIGNAL_UC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_WAITS_SIGNAL_UC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT1_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT1_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT1_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT1_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT2_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT2_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_WAITS_LAMPORT2_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_WAITS_LAMPORT2_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_SIGNAL_MC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_BYTES_SIGNAL_MC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_SIGNAL_UC: {                                             \
        constexpr int MODE = MixedCommMode::OPT_BYTES_SIGNAL_UC;                             \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT1_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT1_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT1_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT1_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT2_MC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT2_MC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES_LAMPORT2_UC: {                                           \
        constexpr int MODE = MixedCommMode::OPT_BYTES_LAMPORT2_UC;                           \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_SIGNAL_MC: {                                                 \
        constexpr int MODE = MixedCommMode::MIXED_SIGNAL_MC;                                 \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_SIGNAL_UC: {                                                 \
        constexpr int MODE = MixedCommMode::MIXED_SIGNAL_UC;                                 \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_LAMPORT1_MC: {                                               \
        constexpr int MODE = MixedCommMode::MIXED_LAMPORT1_MC;                               \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_LAMPORT1_UC: {                                               \
        constexpr int MODE = MixedCommMode::MIXED_LAMPORT1_UC;                               \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_LAMPORT2_MC: {                                               \
        constexpr int MODE = MixedCommMode::MIXED_LAMPORT2_MC;                               \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED_LAMPORT2_UC: {                                               \
        constexpr int MODE = MixedCommMode::MIXED_LAMPORT2_UC;                               \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      default: {                                                                             \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch mode=" << mode; \
        return false;                                                                        \
      }                                                                                      \
    }                                                                                        \
  }()

#define LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args, smem_size, \
                      stream)                                                                      \
  [&]() -> bool {                                                                                  \
    int block_size = get_block_size(kernel, block_size_raw, num_accesses, num_nodes);              \
    cudaError_t status_smem =                                                                      \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);      \
    TVM_FFI_ICHECK_EQ(status_smem, cudaSuccess)                                                    \
        << "Failed to set max dynamic shared memory size, error: "                                 \
        << cudaGetErrorString(status_smem);                                                        \
    cudaError_t status_launch = cudaLaunchCooperativeKernel(                                       \
        reinterpret_cast<void const*>(kernel), grid_size, block_size, args, smem_size, stream);    \
    TVM_FFI_ICHECK_EQ(status_launch, cudaSuccess)                                                  \
        << "Failed to launch cooperative kernel, error: " << cudaGetErrorString(status_launch);    \
    return true;                                                                                   \
  }()

Map<String, int> get_block_size_range(DLDataType dtype, int local_tp_rank, int local_tp_size,
                                      int local_dp_rank, int local_dp_size, int inter_tp_rank,
                                      int inter_tp_size, int inter_dp_rank, int inter_dp_size) {
  int num_nodes = inter_tp_size * inter_dp_size;
  int min_val = round_up<WARP_SIZE>(num_nodes);
  std::array<int, NUM_MIXED_COMM_MODES> max_val_ar_ag;
  std::array<int, NUM_MIXED_COMM_MODES> max_val_rs_ar;
  std::fill(max_val_ar_ag.begin(), max_val_ar_ag.end(), 0);
  std::fill(max_val_rs_ar.begin(), max_val_rs_ar.end(), 0);

#define GET_SINGLE_MAX_BLOCK_SIZE(mode)                                                    \
  do {                                                                                     \
    max_val_ar_ag[mode] = get_max_block_size(                                              \
        fused_allreduce_allgather_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, mode, T>);     \
    max_val_rs_ar[mode] = get_max_block_size(                                              \
        fused_reducescatter_allreduce_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, mode, T>); \
  } while (false)

#define GET_MULTI_MAX_BLOCK_SIZE(mode)                                                    \
  do {                                                                                    \
    max_val_ar_ag[mode] = get_max_block_size(                                             \
        fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, mode, T>);     \
    max_val_rs_ar[mode] = get_max_block_size(                                             \
        fused_reducescatter_allreduce_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, mode, T>); \
  } while (false)

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, T, [&] {
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_SIGNAL_MC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_SIGNAL_UC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT1_MC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT1_UC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_SIGNAL_MC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_SIGNAL_UC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT1_MC);
        GET_SINGLE_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT1_UC);
      } else {
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_SIGNAL_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_SIGNAL_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT1_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT1_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT2_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_WAITS_LAMPORT2_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_SIGNAL_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_SIGNAL_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT1_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT1_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT2_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::OPT_BYTES_LAMPORT2_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_SIGNAL_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_SIGNAL_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_LAMPORT1_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_LAMPORT1_UC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_LAMPORT2_MC);
        GET_MULTI_MAX_BLOCK_SIZE(MixedCommMode::MIXED_LAMPORT2_UC);
      }
      return true;
    });
  });

#undef GET_SINGLE_MAX_BLOCK_SIZE
#undef GET_MULTI_MAX_BLOCK_SIZE

  Map<String, int> outputs;
  outputs.Set("min_val", min_val);
  for (int mode = 0; mode < NUM_MIXED_COMM_MODES; ++mode) {
    outputs.Set("max_val_ar_ag_" + std::to_string(mode), max_val_ar_ag[mode]);
    outputs.Set("max_val_rs_ar_" + std::to_string(mode), max_val_rs_ar[mode]);
  }
  return outputs;
}

void fused_allreduce_allgather(TensorView x_out, TensorView x_in, void* mem_data, void* mem_signal,
                               void* uc_data_array, void* mc_data_full, void* mc_data_tp,
                               void* mc_signal_full, void* mc_signal_tp, void* ns_data,
                               void* ns_signal, void* index_info, void* reset_info,
                               int local_tp_rank, int local_tp_size, int local_dp_rank,
                               int local_dp_size, int inter_tp_rank, int inter_tp_size,
                               int inter_dp_rank, int inter_dp_size, int grid_size, int mode,
                               Optional<int> block_size_raw) {
  CHECK_CONTIGUOUS(x_out);
  CHECK_CONTIGUOUS(x_in);
  CHECK_CUDA(x_out);
  CHECK_DEVICE(x_out, x_in);
  int num_nodes = inter_tp_size * inter_dp_size;
  int dp_size = local_dp_size * inter_dp_size;
  TVM_FFI_ICHECK_EQ(x_out.dtype(), x_in.dtype()) << "x_out and x_in must have the same dtype";
  TVM_FFI_ICHECK_EQ(x_out.ndim(), x_in.ndim())
      << "x_out and x_in must have the same number of dimensions";
  TVM_FFI_ICHECK_EQ(x_out.size(0), x_in.size(0) * dp_size)
      << "The 0th dimension of x_out must be equal to the 0th dimension of x_in multiplied by "
         "dp_size";
  for (int i = 1; i < x_out.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(x_out.size(i), x_in.size(i))
        << "The " << i << "th dimension of x_out must be equal to the " << i
        << "th dimension of x_in";
  }
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  size_t num_elems_in = x_in.numel();
  size_t num_elems_out = x_out.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    MixedCommArgs<T> mixed_comm_args = {
        .x_out = reinterpret_cast<T*>(x_out.data_ptr()),
        .x_in = reinterpret_cast<T const*>(x_in.data_ptr()),
        .mem_data_buffer = reinterpret_cast<T* const*>(mem_data),
        .mem_signal_buffer = reinterpret_cast<uint32_t* const*>(mem_signal),
        .uc_data_array_buffer = reinterpret_cast<T* const*>(uc_data_array),
        .mc_data_full_buffer = reinterpret_cast<T* const*>(mc_data_full),
        .mc_data_tp_buffer = reinterpret_cast<T* const*>(mc_data_tp),
        .mc_signal_full_buffer = reinterpret_cast<uint32_t* const*>(mc_signal_full),
        .mc_signal_tp_buffer = reinterpret_cast<uint32_t* const*>(mc_signal_tp),
        .ns_data_buffer = reinterpret_cast<T* const*>(ns_data),
        .ns_signal_buffer = reinterpret_cast<uint64_t* const*>(ns_signal),
        .index_info = reinterpret_cast<uint32_t*>(index_info),
        .reset_info = reinterpret_cast<int4*>(reset_info),
        .num_elems_in = num_elems_in,
        .num_elems_out = num_elems_out,
        .local_tp_rank = local_tp_rank,
        .local_dp_rank = local_dp_rank,
        .inter_tp_rank = inter_tp_rank,
        .inter_dp_rank = inter_dp_rank,
        .inter_tp_size = inter_tp_size,
        .inter_dp_size = inter_dp_size,
    };
    void* args[] = {&mixed_comm_args};
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    TVM_FFI_ICHECK_EQ(mod<elems_per_thd>(num_elems_in), 0)
        << "The number of elements in x_in must be divisible by " << elems_per_thd;
    size_t num_accesses = get_num_accesses<elems_per_thd>(num_elems_in, grid_size, local_tp_size,
                                                          inter_tp_size, dp_size, mode);
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        return DISPATCH_MODE_SINGLE_NODE(mode, [&] {
          auto kernel =
              fused_allreduce_allgather_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      } else {
        return DISPATCH_MODE_MULTI_NODE(mode, [&] {
          auto kernel = fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      }
    });
  });
}

void fused_reducescatter_allreduce(TensorView x_out, TensorView x_in, void* mem_data,
                                   void* mem_signal, void* uc_data_array, void* mc_data_full,
                                   void* mc_data_tp, void* mc_signal_full, void* mc_signal_tp,
                                   void* ns_data, void* ns_signal, void* index_info,
                                   void* reset_info, int local_tp_rank, int local_tp_size,
                                   int local_dp_rank, int local_dp_size, int inter_tp_rank,
                                   int inter_tp_size, int inter_dp_rank, int inter_dp_size,
                                   int grid_size, int mode, Optional<int> block_size_raw) {
  CHECK_CONTIGUOUS(x_out);
  CHECK_CONTIGUOUS(x_in);
  CHECK_CUDA(x_out);
  CHECK_DEVICE(x_out, x_in);
  int num_nodes = inter_tp_size * inter_dp_size;
  int dp_size = local_dp_size * inter_dp_size;
  TVM_FFI_ICHECK_EQ(x_out.dtype(), x_in.dtype()) << "x_out and x_in must have the same dtype";
  TVM_FFI_ICHECK_EQ(x_out.ndim(), x_in.ndim())
      << "x_out and x_in must have the same number of dimensions";
  TVM_FFI_ICHECK_EQ(x_out.size(0) * dp_size, x_in.size(0))
      << "The 0th dimension of x_out multiplied by dp_size must be equal to the 0th dimension of "
         "x_in";
  for (int i = 1; i < x_out.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(x_out.size(i), x_in.size(i))
        << "The " << i << "th dimension of x_out must be equal to the " << i
        << "th dimension of x_in";
  }
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  size_t num_elems_in = x_in.numel();
  size_t num_elems_out = x_out.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    MixedCommArgs<T> mixed_comm_args = {
        .x_out = reinterpret_cast<T*>(x_out.data_ptr()),
        .x_in = reinterpret_cast<T const*>(x_in.data_ptr()),
        .mem_data_buffer = reinterpret_cast<T* const*>(mem_data),
        .mem_signal_buffer = reinterpret_cast<uint32_t* const*>(mem_signal),
        .uc_data_array_buffer = reinterpret_cast<T* const*>(uc_data_array),
        .mc_data_full_buffer = reinterpret_cast<T* const*>(mc_data_full),
        .mc_data_tp_buffer = reinterpret_cast<T* const*>(mc_data_tp),
        .mc_signal_full_buffer = reinterpret_cast<uint32_t* const*>(mc_signal_full),
        .mc_signal_tp_buffer = reinterpret_cast<uint32_t* const*>(mc_signal_tp),
        .ns_data_buffer = reinterpret_cast<T* const*>(ns_data),
        .ns_signal_buffer = reinterpret_cast<uint64_t* const*>(ns_signal),
        .index_info = reinterpret_cast<uint32_t*>(index_info),
        .reset_info = reinterpret_cast<int4*>(reset_info),
        .num_elems_in = num_elems_in,
        .num_elems_out = num_elems_out,
        .local_tp_rank = local_tp_rank,
        .local_dp_rank = local_dp_rank,
        .inter_tp_rank = inter_tp_rank,
        .inter_dp_rank = inter_dp_rank,
        .inter_tp_size = inter_tp_size,
        .inter_dp_size = inter_dp_size,
    };
    void* args[] = {&mixed_comm_args};
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    TVM_FFI_ICHECK_EQ(mod<elems_per_thd>(num_elems_out), 0)
        << "The number of elements in x_out must be divisible by " << elems_per_thd;
    size_t num_accesses = get_num_accesses<elems_per_thd>(num_elems_out, grid_size, local_tp_size,
                                                          inter_tp_size, dp_size, mode);
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        return DISPATCH_MODE_SINGLE_NODE(mode, [&] {
          auto kernel =
              fused_reducescatter_allreduce_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      } else {
        return DISPATCH_MODE_MULTI_NODE(mode, [&] {
          auto kernel =
              fused_reducescatter_allreduce_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      }
    });
  });
}

#undef DISPATCH_LOCAL_TP_DP_SIZE
#undef DISPATCH_MODE_SINGLE_NODE
#undef DISPATCH_MODE_MULTI_NODE
#undef LAUNCH_KERNEL

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_block_size_range, get_block_size_range);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_allreduce_allgather, fused_allreduce_allgather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_reducescatter_allreduce, fused_reducescatter_allreduce);

}  // namespace
