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

#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <unordered_map>

#include "flashinfer/comm/mixed_comm_decl.cuh"
#include "tvm/ffi/container/map.h"
#include "tvm_ffi_utils.h"

namespace {

using flashinfer::mixed_comm::ACCESS_BYTES;
using flashinfer::mixed_comm::allgather_kernel;
using flashinfer::mixed_comm::allreduce_kernel;
using flashinfer::mixed_comm::ceil_div;
using flashinfer::mixed_comm::floor_div;
using flashinfer::mixed_comm::fused_allreduce_allgather_kernel;
using flashinfer::mixed_comm::fused_reducescatter_allreduce_kernel;
using flashinfer::mixed_comm::is_valid_block_y;
using flashinfer::mixed_comm::is_valid_mode;
using flashinfer::mixed_comm::is_valid_op;
using flashinfer::mixed_comm::MixedCommArgs;
using flashinfer::mixed_comm::MixedCommMode;
using flashinfer::mixed_comm::MixedCommOp;
using flashinfer::mixed_comm::mod;
using flashinfer::mixed_comm::reducescatter_kernel;
using flashinfer::mixed_comm::round_down;
using flashinfer::mixed_comm::round_up;
using flashinfer::mixed_comm::WARP_SIZE;
using tvm::ffi::Map;
using tvm::ffi::Shape;

constexpr int nvshmemx_uniqueid_t_size = sizeof(nvshmemx_uniqueid_t);

void ns_get_unique_id(TensorView uid) {
  CHECK_CONTIGUOUS(uid);
  TVM_FFI_ICHECK_EQ(uid.numel() * get_element_size(uid), nvshmemx_uniqueid_t_size);
  TVM_FFI_ICHECK_EQ(uid.device().device_type, kDLCPU);
  nvshmemx_uniqueid_t* uid_ptr = reinterpret_cast<nvshmemx_uniqueid_t*>(uid.data_ptr());
  *uid_ptr = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_get_uniqueid(uid_ptr);
}

int64_t ns_unique_id_size() { return nvshmemx_uniqueid_t_size; }

int64_t ns_init(TensorView uid, int64_t rank, int64_t world_size) {
  CHECK_CONTIGUOUS(uid);
  TVM_FFI_ICHECK_EQ(uid.numel() * get_element_size(uid), nvshmemx_uniqueid_t_size);
  TVM_FFI_ICHECK_EQ(uid.device().device_type, kDLCPU);
  nvshmemx_uniqueid_t* uid_ptr = reinterpret_cast<nvshmemx_uniqueid_t*>(uid.data_ptr());
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, uid_ptr, &attr);
  return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void ns_finalize() { nvshmem_finalize(); }

int64_t ns_my_pe() { return nvshmem_my_pe(); }

int64_t ns_n_pes() { return nvshmem_n_pes(); }

int64_t ns_local_my_pe() { return nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE); }

int64_t ns_local_n_pes() { return nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE); }

struct NVSHMEMNDAlloc {
  void AllocData(DLTensor* tensor) {
    size_t size = tvm::ffi::GetDataSize(*tensor);
    tensor->data = nvshmem_malloc(size);
    TVM_FFI_ICHECK_NE(tensor->data, nullptr) << "nvshmem_malloc failed. size: " << size;
  }
  void FreeData(DLTensor* tensor) { nvshmem_free(tensor->data); }
};

Tensor ns_malloc_tensor(Shape shape, DLDataType dtype, int device_id) {
  return Tensor::FromNDAlloc(NVSHMEMNDAlloc(), tvm::ffi::Shape(shape), dtype,
                             DLDevice{kDLCUDA, device_id});
}

thread_local std::unordered_map<int, int> dummy_smem_size_map;

int get_dummy_smem_size(int device_id) {
  // Use the dummy shared memory size to avoid two CTAs running on the same SM
  auto it = dummy_smem_size_map.find(device_id);
  if (it == dummy_smem_size_map.end()) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    int max_smem_size = device_prop.sharedMemPerMultiprocessor;
    int resv_smem_size = device_prop.reservedSharedMemPerBlock;
    int dummy_smem_size = std::min(max_smem_size / 2, max_smem_size - resv_smem_size);
    it = dummy_smem_size_map.emplace(device_id, dummy_smem_size).first;
  }
  return it->second;
}

#define SET_LOCAL_TP_DP(local_tp_size, local_dp_size, ...) \
  do {                                                     \
    constexpr int LOCAL_TP_SIZE = local_tp_size;           \
    constexpr int LOCAL_DP_SIZE = local_dp_size;           \
    constexpr bool USE_LOCAL_TP = local_tp_size > 1;       \
    constexpr bool USE_LOCAL_DP = local_dp_size > 1;       \
    return __VA_ARGS__();                                  \
  } while (false)

#define DISPATCH_LOCAL_TP_DP(local_tp_size, local_dp_size, ...)                          \
  [&]() -> bool {                                                                        \
    if (local_tp_size == 1 && local_dp_size == 2) {                                      \
      SET_LOCAL_TP_DP(1, 2, __VA_ARGS__);                                                \
    } else if (local_tp_size == 2 && local_dp_size == 1) {                               \
      SET_LOCAL_TP_DP(2, 1, __VA_ARGS__);                                                \
    } else if (local_tp_size == 1 && local_dp_size == 4) {                               \
      SET_LOCAL_TP_DP(1, 4, __VA_ARGS__);                                                \
    } else if (local_tp_size == 2 && local_dp_size == 2) {                               \
      SET_LOCAL_TP_DP(2, 2, __VA_ARGS__);                                                \
    } else if (local_tp_size == 4 && local_dp_size == 1) {                               \
      SET_LOCAL_TP_DP(4, 1, __VA_ARGS__);                                                \
    } else if (local_tp_size == 1 && local_dp_size == 8) {                               \
      SET_LOCAL_TP_DP(1, 8, __VA_ARGS__);                                                \
    } else if (local_tp_size == 2 && local_dp_size == 4) {                               \
      SET_LOCAL_TP_DP(2, 4, __VA_ARGS__);                                                \
    } else if (local_tp_size == 4 && local_dp_size == 2) {                               \
      SET_LOCAL_TP_DP(4, 2, __VA_ARGS__);                                                \
    } else if (local_tp_size == 8 && local_dp_size == 1) {                               \
      SET_LOCAL_TP_DP(8, 1, __VA_ARGS__);                                                \
    } else {                                                                             \
      TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__                                       \
                            << " invalid combination of local_tp_size=" << local_tp_size \
                            << " and local_dp_size=" << local_dp_size;                   \
      return false;                                                                      \
    }                                                                                    \
  }()

#define SET_INTER_TP_DP(use_inter_tp, use_inter_dp, ...) \
  do {                                                   \
    constexpr bool USE_INTER_TP = use_inter_tp;          \
    constexpr bool USE_INTER_DP = use_inter_dp;          \
    return __VA_ARGS__();                                \
  } while (false)

#define DISPATCH_INTER_TP_DP(inter_tp_size, inter_dp_size, ...) \
  [&]() -> bool {                                               \
    if (inter_tp_size == 1) {                                   \
      if (inter_dp_size == 1) {                                 \
        SET_INTER_TP_DP(false, false, __VA_ARGS__);             \
      } else {                                                  \
        SET_INTER_TP_DP(false, true, __VA_ARGS__);              \
      }                                                         \
    } else {                                                    \
      if (inter_dp_size == 1) {                                 \
        SET_INTER_TP_DP(true, false, __VA_ARGS__);              \
      } else {                                                  \
        SET_INTER_TP_DP(true, true, __VA_ARGS__);               \
      }                                                         \
    }                                                           \
  }()

#define CASE_MODE(mode, ...)             \
  case mode: {                           \
    constexpr MixedCommMode MODE = mode; \
    return __VA_ARGS__();                \
  }

#define DISPATCH_MODE(mode, ...)                                                  \
  [&]() -> bool {                                                                 \
    switch (mode) {                                                               \
      CASE_MODE(MixedCommMode::OPT_WAITS_MC, __VA_ARGS__)                         \
      CASE_MODE(MixedCommMode::OPT_WAITS_UC, __VA_ARGS__)                         \
      CASE_MODE(MixedCommMode::OPT_BYTES1_MC, __VA_ARGS__)                        \
      CASE_MODE(MixedCommMode::OPT_BYTES1_UC, __VA_ARGS__)                        \
      CASE_MODE(MixedCommMode::OPT_BYTES2_MC, __VA_ARGS__)                        \
      CASE_MODE(MixedCommMode::OPT_BYTES2_UC, __VA_ARGS__)                        \
      default:                                                                    \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " invalid mode=" << mode; \
        return false;                                                             \
    }                                                                             \
  }()

#define CASE_BLOCK_Y(block_size_y, ...)        \
  case block_size_y: {                         \
    constexpr int BLOCK_SIZE_Y = block_size_y; \
    return __VA_ARGS__();                      \
  }

#define DISPATCH_BLOCK_Y(block_size_y, ...)                                                       \
  [&]() -> bool {                                                                                 \
    switch (block_size_y) {                                                                       \
      CASE_BLOCK_Y(1, __VA_ARGS__)                                                                \
      CASE_BLOCK_Y(2, __VA_ARGS__)                                                                \
      CASE_BLOCK_Y(4, __VA_ARGS__)                                                                \
      CASE_BLOCK_Y(8, __VA_ARGS__)                                                                \
      default:                                                                                    \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " invalid block_size_y=" << block_size_y; \
        return false;                                                                             \
    }                                                                                             \
  }()

#define LAUNCH_KERNEL(kernel_name, num_elems_all)                                                 \
  [&]() -> bool {                                                                                 \
    TVM_FFI_ICHECK_EQ(mod<args.elems_per_thd>(num_elems_all), 0)                                  \
        << "num_elems_all must be divisible by " << args.elems_per_thd;                           \
    TVM_FFI_ICHECK_GE(args.block_size_x, args.inter_size)                                         \
        << "block_size_x must be greater than or equal to inter_size";                            \
    if constexpr (is_valid_op<USE_LOCAL_TP, USE_LOCAL_DP, USE_INTER_TP, USE_INTER_DP>(op) &&      \
                  is_valid_mode<USE_LOCAL_TP, USE_INTER_TP>(MODE) &&                              \
                  is_valid_block_y(BLOCK_SIZE_Y, LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE)) {           \
      auto* kernel = kernel_name<BLOCK_SIZE_Y, LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP,        \
                                 USE_INTER_DP, MODE, T>;                                          \
      void* kernel_args[] = {&args};                                                              \
      cudaError_t status_smem =                                                                   \
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);   \
      TVM_FFI_ICHECK_EQ(status_smem, cudaSuccess)                                                 \
          << "Failed to set max dynamic shared memory size, error: "                              \
          << cudaGetErrorString(status_smem);                                                     \
      dim3 block_size(args.block_size_x, args.block_size_y);                                      \
      cudaError_t status_launch =                                                                 \
          cudaLaunchCooperativeKernel(reinterpret_cast<void const*>(kernel), args.grid_size,      \
                                      block_size, kernel_args, smem_size, stream);                \
      TVM_FFI_ICHECK_EQ(status_launch, cudaSuccess)                                               \
          << "Failed to launch cooperative kernel, error: " << cudaGetErrorString(status_launch); \
      return true;                                                                                \
    } else {                                                                                      \
      TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " invalid combination of op=" << op         \
                            << " and mode=" << MODE;                                              \
      return false;                                                                               \
    }                                                                                             \
  }()

Map<int, int> get_max_block_size(DLDataType dtype, int local_tp_size, int local_dp_size,
                                 int inter_tp_size, int inter_dp_size, int op_val, int mode_val) {
#define CASE_OP(op, kernel_name)                                                             \
  case op: {                                                                                 \
    if constexpr (is_valid_op<USE_LOCAL_TP, USE_LOCAL_DP, USE_INTER_TP, USE_INTER_DP>(op) && \
                  is_valid_mode<USE_LOCAL_TP, USE_INTER_TP>(MODE) &&                         \
                  is_valid_block_y(BLOCK_SIZE_Y, LOCAL_TP_SIZE, LOCAL_DP_SIZE, MODE)) {      \
      cudaFuncAttributes attr;                                                               \
      cudaFuncGetAttributes(&attr, kernel_name<BLOCK_SIZE_Y, LOCAL_TP_SIZE, LOCAL_DP_SIZE,   \
                                               USE_INTER_TP, USE_INTER_DP, MODE, T>);        \
      max_block_size_dict.Set(block_size_y, round_down<WARP_SIZE>(attr.maxThreadsPerBlock)); \
      return true;                                                                           \
    } else {                                                                                 \
      TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " invalid op=" << op;                  \
      return false;                                                                          \
    }                                                                                        \
  }
  MixedCommOp op = static_cast<MixedCommOp>(op_val);
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  Map<int, int> max_block_size_dict;
  for (int i = 0; i < 4; ++i) {
    int block_size_y = 1 << i;
    if (is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode)) {
      DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, T, [&] {
        return DISPATCH_LOCAL_TP_DP(local_tp_size, local_dp_size, [&] {
          return DISPATCH_INTER_TP_DP(inter_tp_size, inter_dp_size, [&] {
            return DISPATCH_MODE(mode, [&] {
              return DISPATCH_BLOCK_Y(block_size_y, [&] {
                switch (op) {
                  CASE_OP(MixedCommOp::ALLREDUCE, allreduce_kernel)
                  CASE_OP(MixedCommOp::ALLGATHER, allgather_kernel)
                  CASE_OP(MixedCommOp::REDUCESCATTER, reducescatter_kernel)
                  CASE_OP(MixedCommOp::ALLREDUCE_ALLGATHER, fused_allreduce_allgather_kernel)
                  CASE_OP(MixedCommOp::REDUCESCATTER_ALLREDUCE,
                          fused_reducescatter_allreduce_kernel)
                  default:
                    TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " invalid op=" << op;
                    return false;
                }
              });
            });
          });
        });
      });
    }
  }
#undef CASE_OP
  return max_block_size_dict;
}

void check_inputs(TensorView x_out, TensorView x_in, int coef_out, int coef_in) {
  CHECK_CONTIGUOUS(x_out);
  CHECK_CONTIGUOUS(x_in);
  CHECK_CUDA(x_out);
  CHECK_DEVICE(x_out, x_in);
  TVM_FFI_ICHECK_EQ(x_out.dtype(), x_in.dtype()) << "x_out and x_in must have the same dtype";
  TVM_FFI_ICHECK_EQ(x_out.ndim(), x_in.ndim())
      << "x_out and x_in must have the same number of dimensions";
  TVM_FFI_ICHECK_EQ(x_out.size(0) * coef_in, x_in.size(0) * coef_out)
      << "The ratio of 0th dimension of x_out and x_in must be " << coef_out << " : " << coef_in;
  for (int i = 1; i < x_out.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(x_out.size(i), x_in.size(i))
        << "The " << i << "th dimension of x_out must be equal to the " << i
        << "th dimension of x_in";
  }
}

void allreduce(TensorView x_out, TensorView x_in, void* uc_buffer_array, void* mem_buffer,
               void* mc_buffer, void* ns_buffer, int vm_buffer_bytes_base, int64_t ns_data_bytes,
               int ns_signal_bytes, int grid_size, Map<int, int> max_block_size_dict,
               int min_block_size, int min_num_steps, int local_rank, int local_size,
               int inter_rank, int inter_size, int mode_val) {
  constexpr MixedCommOp op = MixedCommOp::ALLREDUCE;
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  check_inputs(x_out, x_in, 1, 1);
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  int64_t num_elems_all = x_in.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    return DISPATCH_LOCAL_TP_DP(local_size, 1, [&] {
      return DISPATCH_INTER_TP_DP(inter_size, 1, [&] {
        return DISPATCH_MODE(mode, [&] {
          MixedCommArgs<LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP, USE_INTER_DP, MODE, T> args(
              /*x_out_all=*/x_out.data_ptr(),
              /*x_in_all=*/x_in.data_ptr(),
              /*uc_buffer_array_all=*/uc_buffer_array,
              /*mem_buffer_all=*/mem_buffer,
              /*mc_buffer_full_all=*/mc_buffer,
              /*mc_buffer_tp_all=*/nullptr,
              /*ns_buffer_all=*/ns_buffer,
              /*vm_buffer_bytes=*/vm_buffer_bytes_base,
              /*ns_data_bytes_all=*/ns_data_bytes,
              /*ns_signal_bytes_all=*/ns_signal_bytes,
              /*grid_size=*/grid_size,
              /*max_block_size_dict=*/max_block_size_dict,
              /*min_block_size=*/min_block_size,
              /*min_num_steps=*/min_num_steps,
              /*num_elems_all=*/num_elems_all,
              /*local_tp_rank=*/local_rank,
              /*local_dp_rank=*/0,
              /*inter_tp_rank=*/inter_rank,
              /*inter_dp_rank=*/0,
              /*inter_tp_size=*/inter_size,
              /*inter_dp_size=*/1);
          return DISPATCH_BLOCK_Y(args.block_size_y,
                                  [&] { return LAUNCH_KERNEL(allreduce_kernel, num_elems_all); });
        });
      });
    });
  });
}

void allgather(TensorView x_out, TensorView x_in, void* uc_buffer_array, void* mem_buffer,
               void* mc_buffer, void* ns_buffer, int vm_buffer_bytes_base, int64_t ns_data_bytes,
               int ns_signal_bytes, int grid_size, Map<int, int> max_block_size_dict,
               int min_block_size, int min_num_steps, int local_rank, int local_size,
               int inter_rank, int inter_size, int mode_val) {
  constexpr MixedCommOp op = MixedCommOp::ALLGATHER;
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  check_inputs(x_out, x_in, local_size * inter_size, 1);
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  int64_t num_elems_in_all = x_in.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    return DISPATCH_LOCAL_TP_DP(1, local_size, [&] {
      return DISPATCH_INTER_TP_DP(1, inter_size, [&] {
        return DISPATCH_MODE(mode, [&] {
          MixedCommArgs<LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP, USE_INTER_DP, MODE, T> args(
              /*x_out_all=*/x_out.data_ptr(),
              /*x_in_all=*/x_in.data_ptr(),
              /*uc_buffer_array_all=*/uc_buffer_array,
              /*mem_buffer_all=*/mem_buffer,
              /*mc_buffer_full_all=*/mc_buffer,
              /*mc_buffer_tp_all=*/nullptr,
              /*ns_buffer_all=*/ns_buffer,
              /*vm_buffer_bytes=*/vm_buffer_bytes_base,
              /*ns_data_bytes_all=*/ns_data_bytes,
              /*ns_signal_bytes_all=*/ns_signal_bytes,
              /*grid_size=*/grid_size,
              /*max_block_size_dict=*/max_block_size_dict,
              /*min_block_size=*/min_block_size,
              /*min_num_steps=*/min_num_steps,
              /*num_elems_all=*/num_elems_in_all,
              /*local_tp_rank=*/0,
              /*local_dp_rank=*/local_rank,
              /*inter_tp_rank=*/0,
              /*inter_dp_rank=*/inter_rank,
              /*inter_tp_size=*/1,
              /*inter_dp_size=*/inter_size);
          return DISPATCH_BLOCK_Y(
              args.block_size_y, [&] { return LAUNCH_KERNEL(allgather_kernel, num_elems_in_all); });
        });
      });
    });
  });
}

void reducescatter(TensorView x_out, TensorView x_in, void* uc_buffer_array, void* mem_buffer,
                   void* mc_buffer, void* ns_buffer, int vm_buffer_bytes_base,
                   int64_t ns_data_bytes, int ns_signal_bytes, int grid_size,
                   Map<int, int> max_block_size_dict, int min_block_size, int min_num_steps,
                   int local_rank, int local_size, int inter_rank, int inter_size, int mode_val) {
  constexpr MixedCommOp op = MixedCommOp::REDUCESCATTER;
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  check_inputs(x_out, x_in, 1, local_size * inter_size);
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  int64_t num_elems_out_all = x_out.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    return DISPATCH_LOCAL_TP_DP(1, local_size, [&] {
      return DISPATCH_INTER_TP_DP(1, inter_size, [&] {
        return DISPATCH_MODE(mode, [&] {
          MixedCommArgs<LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP, USE_INTER_DP, MODE, T> args(
              /*x_out_all=*/x_out.data_ptr(),
              /*x_in_all=*/x_in.data_ptr(),
              /*uc_buffer_array_all=*/uc_buffer_array,
              /*mem_buffer_all=*/mem_buffer,
              /*mc_buffer_full_all=*/mc_buffer,
              /*mc_buffer_tp_all=*/nullptr,
              /*ns_buffer_all=*/ns_buffer,
              /*vm_buffer_bytes=*/vm_buffer_bytes_base,
              /*ns_data_bytes_all=*/ns_data_bytes,
              /*ns_signal_bytes_all=*/ns_signal_bytes,
              /*grid_size=*/grid_size,
              /*max_block_size_dict=*/max_block_size_dict,
              /*min_block_size=*/min_block_size,
              /*min_num_steps=*/min_num_steps,
              /*num_elems_all=*/num_elems_out_all,
              /*local_tp_rank=*/0,
              /*local_dp_rank=*/local_rank,
              /*inter_tp_rank=*/0,
              /*inter_dp_rank=*/inter_rank,
              /*inter_tp_size=*/1,
              /*inter_dp_size=*/inter_size);
          return DISPATCH_BLOCK_Y(args.block_size_y, [&] {
            return LAUNCH_KERNEL(reducescatter_kernel, num_elems_out_all);
          });
        });
      });
    });
  });
}

void fused_allreduce_allgather(TensorView x_out, TensorView x_in, void* uc_buffer_array,
                               void* mem_buffer, void* mc_buffer_full, void* mc_buffer_tp,
                               void* ns_buffer, int vm_buffer_bytes_base, int64_t ns_data_bytes,
                               int ns_signal_bytes, int grid_size,
                               Map<int, int> max_block_size_dict, int min_block_size,
                               int min_num_steps, int local_tp_rank, int local_tp_size,
                               int local_dp_rank, int local_dp_size, int inter_tp_rank,
                               int inter_tp_size, int inter_dp_rank, int inter_dp_size,
                               int mode_val) {
  constexpr MixedCommOp op = MixedCommOp::ALLREDUCE_ALLGATHER;
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  check_inputs(x_out, x_in, local_dp_size * inter_dp_size, 1);
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  int64_t num_elems_in_all = x_in.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    return DISPATCH_LOCAL_TP_DP(local_tp_size, local_dp_size, [&] {
      return DISPATCH_INTER_TP_DP(inter_tp_size, inter_dp_size, [&] {
        return DISPATCH_MODE(mode, [&] {
          MixedCommArgs<LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP, USE_INTER_DP, MODE, T> args(
              /*x_out_all=*/x_out.data_ptr(),
              /*x_in_all=*/x_in.data_ptr(),
              /*uc_buffer_array_all=*/uc_buffer_array,
              /*mem_buffer_all=*/mem_buffer,
              /*mc_buffer_full_all=*/mc_buffer_full,
              /*mc_buffer_tp_all=*/mc_buffer_tp,
              /*ns_buffer_all=*/ns_buffer,
              /*vm_buffer_bytes=*/vm_buffer_bytes_base,
              /*ns_data_bytes_all=*/ns_data_bytes,
              /*ns_signal_bytes_all=*/ns_signal_bytes,
              /*grid_size=*/grid_size,
              /*max_block_size_dict=*/max_block_size_dict,
              /*min_block_size=*/min_block_size,
              /*min_num_steps=*/min_num_steps,
              /*num_elems_all=*/num_elems_in_all,
              /*local_tp_rank=*/local_tp_rank,
              /*local_dp_rank=*/local_dp_rank,
              /*inter_tp_rank=*/inter_tp_rank,
              /*inter_dp_rank=*/inter_dp_rank,
              /*inter_tp_size=*/inter_tp_size,
              /*inter_dp_size=*/inter_dp_size);
          return DISPATCH_BLOCK_Y(args.block_size_y, [&] {
            return LAUNCH_KERNEL(fused_allreduce_allgather_kernel, num_elems_in_all);
          });
        });
      });
    });
  });
}

void fused_reducescatter_allreduce(TensorView x_out, TensorView x_in, void* uc_buffer_array,
                                   void* mem_buffer, void* mc_buffer_full, void* mc_buffer_tp,
                                   void* ns_buffer, int vm_buffer_bytes_base, int64_t ns_data_bytes,
                                   int ns_signal_bytes, int grid_size,
                                   Map<int, int> max_block_size_dict, int min_block_size,
                                   int min_num_steps, int local_tp_rank, int local_tp_size,
                                   int local_dp_rank, int local_dp_size, int inter_tp_rank,
                                   int inter_tp_size, int inter_dp_rank, int inter_dp_size,
                                   int mode_val) {
  constexpr MixedCommOp op = MixedCommOp::REDUCESCATTER_ALLREDUCE;
  MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  check_inputs(x_out, x_in, 1, local_dp_size * inter_dp_size);
  cudaStream_t stream = get_stream(x_out.device());
  int smem_size = get_dummy_smem_size(x_out.device().device_id);
  int64_t num_elems_out_all = x_out.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x_out.dtype(), T, [&] {
    return DISPATCH_LOCAL_TP_DP(local_tp_size, local_dp_size, [&] {
      return DISPATCH_INTER_TP_DP(inter_tp_size, inter_dp_size, [&] {
        return DISPATCH_MODE(mode, [&] {
          MixedCommArgs<LOCAL_TP_SIZE, LOCAL_DP_SIZE, USE_INTER_TP, USE_INTER_DP, MODE, T> args(
              /*x_out_all=*/x_out.data_ptr(),
              /*x_in_all=*/x_in.data_ptr(),
              /*uc_buffer_array_all=*/uc_buffer_array,
              /*mem_buffer_all=*/mem_buffer,
              /*mc_buffer_full_all=*/mc_buffer_full,
              /*mc_buffer_tp_all=*/mc_buffer_tp,
              /*ns_buffer_all=*/ns_buffer,
              /*vm_buffer_bytes=*/vm_buffer_bytes_base,
              /*ns_data_bytes_all=*/ns_data_bytes,
              /*ns_signal_bytes_all=*/ns_signal_bytes,
              /*grid_size=*/grid_size,
              /*max_block_size_dict=*/max_block_size_dict,
              /*min_block_size=*/min_block_size,
              /*min_num_steps=*/min_num_steps,
              /*num_elems_all=*/num_elems_out_all,
              /*local_tp_rank=*/local_tp_rank,
              /*local_dp_rank=*/local_dp_rank,
              /*inter_tp_rank=*/inter_tp_rank,
              /*inter_dp_rank=*/inter_dp_rank,
              /*inter_tp_size=*/inter_tp_size,
              /*inter_dp_size=*/inter_dp_size);
          return DISPATCH_BLOCK_Y(args.block_size_y, [&] {
            return LAUNCH_KERNEL(fused_reducescatter_allreduce_kernel, num_elems_out_all);
          });
        });
      });
    });
  });
}

#undef SET_LOCAL_TP_DP
#undef DISPATCH_LOCAL_TP_DP
#undef SET_INTER_TP_DP
#undef DISPATCH_INTER_TP_DP
#undef CASE_MODE
#undef DISPATCH_MODE
#undef CASE_BLOCK_Y
#undef DISPATCH_BLOCK_Y
#undef LAUNCH_KERNEL

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_get_unique_id, ns_get_unique_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_unique_id_size, ns_unique_id_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_init, ns_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_finalize, ns_finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_my_pe, ns_my_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_n_pes, ns_n_pes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_local_my_pe, ns_local_my_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_local_n_pes, ns_local_n_pes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_malloc, ns_malloc_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_max_block_size, get_max_block_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce, allreduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allgather, allgather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reducescatter, reducescatter);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_allreduce_allgather, fused_allreduce_allgather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_reducescatter_allreduce, fused_reducescatter_allreduce);

}  // namespace
