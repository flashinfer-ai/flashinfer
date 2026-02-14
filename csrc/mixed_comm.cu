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
using flashinfer::mixed_comm::MixedCommMode;
using flashinfer::mixed_comm::mod;
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
  if (mode == MixedCommMode::OPT_BYTES) {
    num_accesses = round_up(num_accesses, local_tp_size * inter_tp_size);
  } else if (mode == MixedCommMode::MIXED) {
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
      case MixedCommMode::OPT_WAITS: {                                                       \
        constexpr int MODE = MixedCommMode::OPT_WAITS;                                       \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES: {                                                       \
        constexpr int MODE = MixedCommMode::OPT_BYTES;                                       \
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
      case MixedCommMode::OPT_WAITS: {                                                       \
        constexpr int MODE = MixedCommMode::OPT_WAITS;                                       \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::OPT_BYTES: {                                                       \
        constexpr int MODE = MixedCommMode::OPT_BYTES;                                       \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      case MixedCommMode::MIXED: {                                                           \
        constexpr int MODE = MixedCommMode::MIXED;                                           \
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
  int max_val_ar_ag_opt_waits;
  int max_val_ar_ag_opt_bytes;
  int max_val_ar_ag_mixed;
  int max_val_rs_ar_opt_waits;
  int max_val_rs_ar_opt_bytes;
  int max_val_rs_ar_mixed;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, T, [&] {
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        max_val_ar_ag_opt_waits =
            get_max_block_size(fused_allreduce_allgather_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                                     MixedCommMode::OPT_WAITS, T>);
        max_val_ar_ag_opt_bytes =
            get_max_block_size(fused_allreduce_allgather_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                                     MixedCommMode::OPT_BYTES, T>);
        max_val_ar_ag_mixed = 0;
        max_val_rs_ar_opt_waits = get_max_block_size(
            fused_reducescatter_allreduce_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                      MixedCommMode::OPT_WAITS, T>);
        max_val_rs_ar_opt_bytes = get_max_block_size(
            fused_reducescatter_allreduce_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                      MixedCommMode::OPT_BYTES, T>);
        max_val_rs_ar_mixed = 0;
      } else {
        max_val_ar_ag_opt_waits =
            get_max_block_size(fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                                    MixedCommMode::OPT_WAITS, T>);
        max_val_ar_ag_opt_bytes =
            get_max_block_size(fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                                    MixedCommMode::OPT_BYTES, T>);
        max_val_ar_ag_mixed =
            get_max_block_size(fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                                    MixedCommMode::MIXED, T>);
        max_val_rs_ar_opt_waits = get_max_block_size(
            fused_reducescatter_allreduce_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                     MixedCommMode::OPT_WAITS, T>);
        max_val_rs_ar_opt_bytes = get_max_block_size(
            fused_reducescatter_allreduce_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                     MixedCommMode::OPT_BYTES, T>);
        max_val_rs_ar_mixed = get_max_block_size(
            fused_reducescatter_allreduce_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE,
                                                     MixedCommMode::MIXED, T>);
      }
      return true;
    });
  });
  Map<String, int> outputs;
  outputs.Set("min_val", min_val);
  outputs.Set("max_val_ar_ag_0", max_val_ar_ag_opt_waits);
  outputs.Set("max_val_ar_ag_1", max_val_ar_ag_opt_bytes);
  outputs.Set("max_val_ar_ag_2", max_val_ar_ag_mixed);
  outputs.Set("max_val_rs_ar_0", max_val_rs_ar_opt_waits);
  outputs.Set("max_val_rs_ar_1", max_val_rs_ar_opt_bytes);
  outputs.Set("max_val_rs_ar_2", max_val_rs_ar_mixed);
  return outputs;
}

void fused_allreduce_allgather(TensorView x_out, TensorView x_in, void* mem_data_raw,
                               void* mem_signal_raw, void* mc_data_full_raw, void* mc_data_tp_raw,
                               void* mc_signal_full_raw, void* mc_signal_tp_raw,
                               Optional<TensorView> ns_data_raw, Optional<TensorView> ns_signal_raw,
                               int local_tp_rank, int local_tp_size, int local_dp_rank,
                               int local_dp_size, int inter_tp_rank, int inter_tp_size,
                               int inter_dp_rank, int inter_dp_size, int grid_size, int mode,
                               Optional<int> block_size_raw) {
  CHECK_CONTIGUOUS(x_out);
  CHECK_CONTIGUOUS(x_in);
  CHECK_CUDA(x_out);
  CHECK_DEVICE(x_out, x_in);
  int num_nodes = inter_tp_size * inter_dp_size;
  if (num_nodes > 1) {
    TVM_FFI_ICHECK(ns_data_raw.has_value()) << "ns_data must be provided when num_nodes > 1";
    TVM_FFI_ICHECK(ns_signal_raw.has_value()) << "ns_signal must be provided when num_nodes > 1";
    CHECK_CONTIGUOUS(ns_data_raw.value());
    CHECK_CONTIGUOUS(ns_signal_raw.value());
    CHECK_DEVICE(x_out, ns_data_raw.value());
    CHECK_DEVICE(x_out, ns_signal_raw.value());
  }
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
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    T* x_out_ptr = reinterpret_cast<T*>(x_out.data_ptr());
    T const* x_in_ptr = reinterpret_cast<T const*>(x_in.data_ptr());
    T* mem_data = reinterpret_cast<T*>(mem_data_raw);
    uint32_t* mem_signal = reinterpret_cast<uint32_t*>(mem_signal_raw);
    T* mc_data_full = reinterpret_cast<T*>(mc_data_full_raw);
    T* mc_data_tp = reinterpret_cast<T*>(mc_data_tp_raw);
    uint32_t* mc_signal_full = reinterpret_cast<uint32_t*>(mc_signal_full_raw);
    uint32_t* mc_signal_tp = reinterpret_cast<uint32_t*>(mc_signal_tp_raw);
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    TVM_FFI_ICHECK_EQ(mod<elems_per_thd>(num_elems_in), 0)
        << "The number of elements in x_in must be divisible by " << elems_per_thd;
    size_t num_accesses = get_num_accesses<elems_per_thd>(num_elems_in, grid_size, local_tp_size,
                                                          inter_tp_size, dp_size, mode);
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        void* args[] = {&x_out_ptr,    &x_in_ptr,      &mem_data,       &mem_signal,
                        &mc_data_full, &mc_data_tp,    &mc_signal_full, &mc_signal_tp,
                        &num_elems_in, &local_tp_rank, &local_dp_rank};
        return DISPATCH_MODE_SINGLE_NODE(mode, [&] {
          auto kernel =
              fused_allreduce_allgather_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      } else {
        T* ns_data_ptr = reinterpret_cast<T*>(ns_data_raw.value().data_ptr());
        uint64_t* ns_signal_ptr = reinterpret_cast<uint64_t*>(ns_signal_raw.value().data_ptr());
        void* args[] = {&x_out_ptr,     &x_in_ptr,      &mem_data,       &mem_signal,
                        &mc_data_full,  &mc_data_tp,    &mc_signal_full, &mc_signal_tp,
                        &ns_data_ptr,   &ns_signal_ptr, &num_elems_in,   &local_tp_rank,
                        &local_dp_rank, &inter_tp_rank, &inter_dp_rank,  &inter_tp_size,
                        &inter_dp_size};
        return DISPATCH_MODE_MULTI_NODE(mode, [&] {
          auto kernel = fused_allreduce_allgather_multi_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      }
    });
  });
}

void fused_reducescatter_allreduce(
    TensorView x_out, TensorView x_in, void* mem_data_raw, void* mem_signal_raw,
    void* uc_data_array_raw, void* mc_data_full_raw, void* mc_data_tp_raw, void* mc_signal_full_raw,
    Optional<TensorView> ns_data_raw, Optional<TensorView> ns_signal_raw, int local_tp_rank,
    int local_tp_size, int local_dp_rank, int local_dp_size, int inter_tp_rank, int inter_tp_size,
    int inter_dp_rank, int inter_dp_size, int grid_size, int mode, Optional<int> block_size_raw) {
  CHECK_CONTIGUOUS(x_out);
  CHECK_CONTIGUOUS(x_in);
  CHECK_CUDA(x_out);
  CHECK_DEVICE(x_out, x_in);
  int num_nodes = inter_tp_size * inter_dp_size;
  if (num_nodes > 1) {
    TVM_FFI_ICHECK(ns_data_raw.has_value()) << "ns_data must be provided when num_nodes > 1";
    TVM_FFI_ICHECK(ns_signal_raw.has_value()) << "ns_signal must be provided when num_nodes > 1";
    CHECK_CONTIGUOUS(ns_data_raw.value());
    CHECK_CONTIGUOUS(ns_signal_raw.value());
    CHECK_DEVICE(x_out, ns_data_raw.value());
    CHECK_DEVICE(x_out, ns_signal_raw.value());
  }
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
  size_t num_elems_out = x_out.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    T* x_out_ptr = reinterpret_cast<T*>(x_out.data_ptr());
    T const* x_in_ptr = reinterpret_cast<T const*>(x_in.data_ptr());
    T* mem_data = reinterpret_cast<T*>(mem_data_raw);
    uint32_t* mem_signal = reinterpret_cast<uint32_t*>(mem_signal_raw);
    T* const* uc_data_array = reinterpret_cast<T* const*>(uc_data_array_raw);
    T* mc_data_full = reinterpret_cast<T*>(mc_data_full_raw);
    T* mc_data_tp = reinterpret_cast<T*>(mc_data_tp_raw);
    uint32_t* mc_signal_full = reinterpret_cast<uint32_t*>(mc_signal_full_raw);
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    TVM_FFI_ICHECK_EQ(mod<elems_per_thd>(num_elems_out), 0)
        << "The number of elements in x_out must be divisible by " << elems_per_thd;
    size_t num_accesses = get_num_accesses<elems_per_thd>(num_elems_out, grid_size, local_tp_size,
                                                          inter_tp_size, dp_size, mode);
    return DISPATCH_LOCAL_TP_DP_SIZE(local_tp_size, local_dp_size, [&] {
      if (num_nodes == 1) {
        void* args[] = {&x_out_ptr,     &x_in_ptr,      &mem_data,     &mem_signal,
                        &uc_data_array, &mc_data_full,  &mc_data_tp,   &mc_signal_full,
                        &num_elems_out, &local_tp_rank, &local_dp_rank};
        return DISPATCH_MODE_SINGLE_NODE(mode, [&] {
          auto kernel =
              fused_reducescatter_allreduce_single_node<LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE, T>;
          return LAUNCH_KERNEL(kernel, block_size_raw, num_accesses, num_nodes, grid_size, args,
                               smem_size, stream);
        });
      } else {
        T* ns_data_ptr = reinterpret_cast<T*>(ns_data_raw.value().data_ptr());
        uint64_t* ns_signal_ptr = reinterpret_cast<uint64_t*>(ns_signal_raw.value().data_ptr());
        void* args[] = {&x_out_ptr,     &x_in_ptr,      &mem_data,      &mem_signal,
                        &uc_data_array, &mc_data_full,  &mc_data_tp,    &mc_signal_full,
                        &ns_data_ptr,   &ns_signal_ptr, &num_elems_out, &local_tp_rank,
                        &local_dp_rank, &inter_tp_rank, &inter_dp_rank, &inter_tp_size,
                        &inter_dp_size};
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
