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

#pragma once

#include "flashinfer/comm/mixed_comm_decl.cuh"

namespace flashinfer {
namespace mixed_comm {

template <typename T>
__device__ __forceinline__ void copy_data(T* outputs, T const* inputs) {
  *reinterpret_cast<uint4*>(outputs) = *reinterpret_cast<uint4 const*>(inputs);
}

template <typename T>
__device__ __forceinline__ void load_volatile(T* outputs, T const* inputs) {
  uint4* dst = reinterpret_cast<uint4*>(outputs);
  uint4 const* src = reinterpret_cast<uint4 const*>(inputs);
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(dst->x), "=r"(dst->y), "=r"(dst->z), "=r"(dst->w)
               : "l"(src)
               : "memory");
}

__device__ __forceinline__ void multicast_add_signal(uint32_t* mc_signal) {
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
               :
               : "l"(mc_signal), "n"(1)
               : "memory");
}

__device__ __forceinline__ void multicast_wait_signal(uint32_t* mem_signal, uint32_t val) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref_signal(*mem_signal);
  while (ref_signal.load(cuda::memory_order_acquire) != val);
}

template <typename T>
__device__ __forceinline__ void multicast_load_reduce(T* outputs, T const* inputs) {
  uint4* dst = reinterpret_cast<uint4*>(outputs);
  uint4 const* src = reinterpret_cast<uint4 const*>(inputs);
  if constexpr (std::is_same_v<T, nv_half>) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst->x), "=r"(dst->y), "=r"(dst->z), "=r"(dst->w)
        : "l"(src)
        : "memory");
  } else {
    static_assert(std::is_same_v<T, nv_bfloat16>, "Invalid data type");
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst->x), "=r"(dst->y), "=r"(dst->z), "=r"(dst->w)
        : "l"(src)
        : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void multicast_store(T* outputs, T const* inputs) {
  float4* dst = reinterpret_cast<float4*>(outputs);
  float4 const* src = reinterpret_cast<float4 const*>(inputs);
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(dst), "f"(src->x), "f"(src->y), "f"(src->z), "f"(src->w)
               : "memory");
}

template <int group_size, typename T>
__device__ __forceinline__ void unicast_store(T* const* outputs, T const* inputs, int64_t ofst) {
#pragma unroll
  for (int i = 0; i < group_size; ++i) {
    copy_data(outputs[i] + ofst, inputs);
  }
}

template <int group_size, typename T>
__device__ __forceinline__ void unicast_store(T* const* outputs, T const* inputs, int64_t ofst,
                                              int stride) {
#pragma unroll
  for (int i = 0; i < group_size; ++i) {
    copy_data(outputs[i] + ofst, inputs);
    ofst += stride;
  }
}

template <int elems_per_thd, typename T_in, typename T_out>
__device__ __forceinline__ void convert_dtype(T_out* outputs, T_in const* inputs) {
  static_assert(!std::is_same_v<T_in, T_out>, "T_in and T_out must be different");
  static_assert(elems_per_thd % 2 == 0, "elems_per_thd must be even");
  if constexpr (std::is_same_v<T_in, float>) {
#pragma unroll
    for (int i = 0; i < elems_per_thd; i += 2) {
      if constexpr (std::is_same_v<T_out, nv_half>) {
        *reinterpret_cast<nv_half2*>(outputs + i) =
            __float22half2_rn(*reinterpret_cast<float2 const*>(inputs + i));
      } else {
        static_assert(std::is_same_v<T_out, nv_bfloat16>, "Invalid data type");
        *reinterpret_cast<nv_bfloat162*>(outputs + i) =
            __float22bfloat162_rn(*reinterpret_cast<float2 const*>(inputs + i));
      }
    }
  } else {
    static_assert(std::is_same_v<T_out, float>, "Invalid data type");
#pragma unroll
    for (int i = 0; i < elems_per_thd; i += 2) {
      if constexpr (std::is_same_v<T_in, nv_half>) {
        *reinterpret_cast<float2*>(outputs + i) =
            __half22float2(*reinterpret_cast<nv_half2 const*>(inputs + i));
      } else {
        static_assert(std::is_same_v<T_in, nv_bfloat16>, "Invalid data type");
        *reinterpret_cast<float2*>(outputs + i) =
            __bfloat1622float2(*reinterpret_cast<nv_bfloat162 const*>(inputs + i));
      }
    }
  }
}

template <int elems_per_thd, typename T>
__device__ __forceinline__ void accumulate(T_ACC* outputs, T const* inputs) {
  T_ACC reg_cvt[elems_per_thd];
  convert_dtype<elems_per_thd>(reg_cvt, inputs);
#pragma unroll
  for (int i = 0; i < elems_per_thd; ++i) {
    outputs[i] += reg_cvt[i];
  }
}

template <typename T>
__device__ __forceinline__ void vm_receive_data(T* outputs, T const* inputs) {
  do {
    load_volatile(outputs, inputs);
  } while (has_neg_zero(outputs));
}

template <int reduce_size, typename T, bool init_is_local = true>
__device__ __forceinline__ void vm_reduce_data(T* reg_data_out, T const* data_init,
                                               T const* data_remain, int stride) {
  if constexpr (reduce_size == 1) {
    if constexpr (init_is_local) {
      copy_data(reg_data_out, data_init);
    } else {
      vm_receive_data(reg_data_out, data_init);
    }
  } else {
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    T_ACC reg_acc[elems_per_thd];
    T reg_data_tmp[elems_per_thd];
    T* reg_data = reg_data_out;
    T* reg_data_next = reg_data_tmp;
    bool invalid;
    do {
      if constexpr (init_is_local) {
        copy_data(reg_data, data_init);
      } else {
        load_volatile(reg_data, data_init);
      }
      T const* data_val = data_remain;
      load_volatile(reg_data_next, data_val);
      if constexpr (init_is_local) {
        invalid = false;
      } else {
        invalid = has_neg_zero(reg_data);
      }
      convert_dtype<elems_per_thd>(reg_acc, reg_data);
#pragma unroll
      for (int i = 2; i < reduce_size; ++i) {
        cuda::std::swap(reg_data, reg_data_next);
        data_val += stride;
        load_volatile(reg_data_next, data_val);
        invalid = has_neg_zero(reg_data) ? true : invalid;
        accumulate<elems_per_thd>(reg_acc, reg_data);
      }
      invalid = has_neg_zero(reg_data_next) ? true : invalid;
      accumulate<elems_per_thd>(reg_acc, reg_data_next);
      convert_dtype<elems_per_thd>(reg_data_out, reg_acc);
    } while (invalid);
  }
}

template <int reduce_size, typename T>
__device__ __forceinline__ void vm_reduce_data(T* reg_data_out, T const* data, int stride) {
  vm_reduce_data<reduce_size, T, /*init_is_local=*/false>(reg_data_out, data, data + stride,
                                                          stride);
}

template <bool use_reduce, typename T, bool init_loaded = false>
__device__ __forceinline__ void ns_reduce_data(T* reg_data_out, T const* data, int stride,
                                               int reduce_size) {
  if constexpr (!use_reduce) {
    if constexpr (!init_loaded) {
      copy_data(reg_data_out, data);
    }
  } else {
    constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
    T_ACC reg_acc[elems_per_thd];
    T reg_data_tmp[elems_per_thd];
    T* reg_data = reg_data_out;
    T* reg_data_next = reg_data_tmp;
    if constexpr (!init_loaded) {
      copy_data(reg_data, data);
      data += stride;
      copy_data(reg_data_next, data);
      convert_dtype<elems_per_thd>(reg_acc, reg_data);
    } else {
      copy_data(reg_data_next, data);
      convert_dtype<elems_per_thd>(reg_acc, reg_data_out);
    }
    for (int i = 2; i < reduce_size; ++i) {
      cuda::std::swap(reg_data, reg_data_next);
      data += stride;
      copy_data(reg_data_next, data);
      accumulate<elems_per_thd>(reg_acc, reg_data);
    }
    accumulate<elems_per_thd>(reg_acc, reg_data_next);
    convert_dtype<elems_per_thd>(reg_data_out, reg_acc);
  }
}

template <int local_size, typename T>
__device__ __forceinline__ void set_uc_data_array(T* outputs[local_size], T* const* inputs,
                                                  int64_t ofst) {
#pragma unroll
  for (int i = 0; i < floor_div<2>(local_size); ++i) {
    reinterpret_cast<uint4*>(outputs)[i] = reinterpret_cast<uint4 const*>(inputs)[i];
  }
#pragma unroll
  for (int i = 0; i < local_size; ++i) {
    outputs[i] += ofst;
  }
}

template <int block_size_y, typename T>
__device__ __forceinline__ void run_first_block_y(int tid_y, T func) {
  if constexpr (block_size_y == 1) {
    func();
  } else {
    if (tid_y == 0) {
      func();
    }
  }
}

#define CIRCULAR_ADD(val, size, stride)                \
  do {                                                 \
    val = (val == size - stride) ? 0 : (val + stride); \
  } while (false)

#define CIRCULAR_MINUS(val, size, stride)                \
  do {                                                   \
    val = (val == 0) ? (size - stride) : (val - stride); \
  } while (false)

#define CIRCULAR_MAX_CLIP(val, size)         \
  do {                                       \
    val = (val < size) ? val : (val - size); \
  } while (false)

#define CIRCULAR_MIN_CLIP(val, size)      \
  do {                                    \
    val = (val < 0) ? (val + size) : val; \
  } while (false)

#define INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size)                                   \
  int const& bid = blockIdx.x;                                                               \
  int const& tid_x = threadIdx.x;                                                            \
  int const& tid_y = threadIdx.y;                                                            \
  int const base_data_idx = tid_x * args.elems_per_thd;                                      \
  int tid;                                                                                   \
  if constexpr (block_size_y == 1) {                                                         \
    tid = tid_x;                                                                             \
  } else {                                                                                   \
    tid = tid_x + tid_y * blockDim.x;                                                        \
  }                                                                                          \
  T reg_data_array[2][args.elems_per_thd];                                                   \
  T* reg_data = reg_data_array[0];                                                           \
  T* reg_data_prev = reg_data_array[1];                                                      \
  int64_t const ofst_elems_base = bid * args.max_num_elems;                                  \
  int64_t const ofst_elems = ofst_elems_base + base_data_idx;                                \
  T* x_out = args.x_out_all + ofst_elems;                                                    \
  T const* x_in = args.x_in_all + ofst_elems;                                                \
  int64_t num_elems_remain = min(args.num_elems_all - ofst_elems_base, args.max_num_elems);  \
  int64_t num_elems_remain_prev;                                                             \
  BufferInfo<args.use_inter, T> buffer_info(args.buffer_info + bid, args.vm_buffer_size_all, \
                                            args.ns_data_size_all, args.ns_signal_size_all); \
  int64_t const vm_ofst_data_base = static_cast<int64_t>(bid) * args.vm_buffer_size;         \
  int64_t const vm_ofst_data = vm_ofst_data_base + base_data_idx;                            \
  T* mem_data = args.mem_buffer_all + vm_ofst_data;                                          \
  T* ns_data;                                                                                \
  uint64_t* ns_signal;                                                                       \
  if constexpr (args.use_inter) {                                                            \
    ns_data = args.ns_data_all + (static_cast<int64_t>(bid) * ns_data_size + base_data_idx); \
    ns_signal = args.ns_signal_all + bid * 2;                                                \
  }                                                                                          \
  auto func_vm_reset_buffer = [&buffer_info, args, base_data_idx, vm_buffer_size,            \
                               mem_data](int64_t ofst_buffer) -> void {                      \
    buffer_info.reset_vm_buffer(mem_data + (ofst_buffer - base_data_idx), base_data_idx,     \
                                args.elems_per_blk);                                         \
    buffer_info.update_vm_reset_size(vm_buffer_size);                                        \
  }

#define UPDATE_BASIC_VARIABLES(x_stride)      \
  do {                                        \
    cuda::std::swap(reg_data, reg_data_prev); \
    num_elems_remain_prev = num_elems_remain; \
    num_elems_remain -= x_stride;             \
    buffer_info.update_buffer_index();        \
  } while (false)

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_waits_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size = args.local_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
  T *mc_data, *uc_data_array[args.local_size];
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data_self);
  }

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, mc_data, uc_data_array](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      copy_data(reg_data, x_in);
      replace_neg_zero(reg_data);
      if constexpr (args.use_multicast) {
        multicast_store(mc_data + ofst_buffer, reg_data);
      } else {
        unicast_store<args.local_size>(uc_data_array, reg_data, ofst_buffer);
      }
    }
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, mem_data](
                                               T* reg_data, int64_t num_elems,
                                               int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, mem_data + ofst_buffer, args.elems_per_blk);
      copy_data(x_out, reg_data);
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and write outputs (previous)
    func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_waits_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size = args.local_size * args.elems_per_blk;
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
  T *mc_data, *uc_data_array[args.local_size];
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data_self);
  }

  auto func_ns_read_inputs_send_data = [&x_in, args, base_data_idx, tid, ns_data, ns_signal](
                                           int64_t num_elems, int64_t ofst_data,
                                           int ofst_signal) -> void {
    T* dst = ns_data + (ofst_data + args.inter_rank * args.elems_per_blk);
    if (base_data_idx < num_elems) {
      copy_data(dst, x_in);
    }
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
      T* ns_dst_src = dst - base_data_idx;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    x_in += args.elems_per_blk;
  };
  auto func_ns_reduce_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, ns_data, ns_signal, mc_data, uc_data_array](
          T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
          int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    if (base_data_idx < num_elems) {
      ns_reduce_data<args.use_inter>(reg_data, ns_data + ns_ofst_data, args.elems_per_blk,
                                     args.inter_size);
      replace_neg_zero(reg_data);
      if constexpr (args.use_multicast) {
        multicast_store(mc_data + vm_ofst_buffer, reg_data);
      } else {
        unicast_store<args.local_size>(uc_data_array, reg_data, vm_ofst_buffer);
      }
    }
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, mem_data](
                                               T* reg_data, int64_t num_elems,
                                               int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, mem_data + ofst_buffer, args.elems_per_blk);
      copy_data(x_out, reg_data);
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data across different nodes
  func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce data across different nodes and send data in the same node
  func_ns_reduce_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                   buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                   buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data across different nodes
    func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                  buffer_info.ns_ofst_signal());
    // Reduce data in the same node and write outputs (previous)
    func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.vm_ofst_buffer_prev());
    // Reduce data across different nodes and send data in the same node
    func_ns_reduce_data_vm_send_data(
        reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
        buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes1_multicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * local_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_multicast && !args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_rank * args.elems_per_blk;
  int x_stride = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = x_stride * 2;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + x_ofst_chunk);
  T* mc_data_2 = mc_data + x_stride;
  T* mem_data_2 = mem_data + x_stride;
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, x_stride, mem_data](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T* dst = mem_data + ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size; i += block_size_y) {
      if (ofst_copy + base_data_idx < num_elems) {
        copy_data(dst + ofst_copy, x_in + ofst_copy);
      }
      ofst_copy += stride_copy;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_send_data = [args, base_data_idx, tid, tid_y, x_ofst_chunk, mc_data,
                                   mc_data_2, mem_signal, mc_signal](T* reg_data, int64_t num_elems,
                                                                     int64_t ofst_buffer) -> void {
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (x_ofst_chunk + base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_data + ofst_buffer);
        replace_neg_zero(reg_data);
        multicast_store(mc_data_2 + ofst_buffer, reg_data);
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T const* src = mem_data_2 + ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size; i += block_size_y) {
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src + ofst_copy);
        copy_data(x_out + ofst_copy, reg_data);
      }
      ofst_copy += stride_copy;
    }
    x_out += x_stride;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce and send data in the same node
  func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data in the same node
    func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes1_unicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * (local_size - 1) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && !args.use_multicast && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int x_ofst_chunk = args.local_rank * args.elems_per_blk;
  int x_stride = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = x_stride - args.elems_per_blk;
  int vm_buffer_size = vm_buffer_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);
  T* vm_x_out = x_out + x_ofst_chunk;
  T const* vm_x_in = x_in + x_ofst_chunk;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* mem_data_2 = mem_data + vm_buffer_size_base;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, x_ofst_chunk, x_stride,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    int ofst_copy_in = x_ofst_chunk;
    int64_t ofst_copy_out = ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size - 1; ++i) {
      CIRCULAR_ADD(ofst_copy_in, x_stride, args.elems_per_blk);
      if (ofst_copy_in + base_data_idx < num_elems) {
        copy_data(reg_data, x_in + ofst_copy_in);
        replace_neg_zero(reg_data);
        copy_data(uc_data_array[i] + ofst_copy_out, reg_data);
      }
      ofst_copy_out += args.elems_per_blk;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_send_data =
      [&vm_x_out, &vm_x_in, args, base_data_idx, x_ofst_chunk, x_stride, vm_buffer_size_base,
       mem_data, uc_data_array](T* reg_data, int64_t num_elems, int64_t ofst_buffer) -> void {
    if (x_ofst_chunk + base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, vm_x_in, mem_data + ofst_buffer,
                                      args.elems_per_blk);
      replace_neg_zero(reg_data);
      unicast_store<args.local_size - 1>(uc_data_array, reg_data, ofst_buffer + vm_buffer_size_base,
                                         args.elems_per_blk);
      copy_data(vm_x_out, reg_data);
    }
    vm_x_out += x_stride;
    vm_x_in += x_stride;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, x_ofst_chunk, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    T const* src = mem_data_2 + ofst_buffer;
    int ofst_copy = x_ofst_chunk;
#pragma unroll
    for (int i = 0; i < args.local_size - 1; ++i) {
      CIRCULAR_MINUS(ofst_copy, x_stride, args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src);
        copy_data(x_out + ofst_copy, reg_data);
      }
      src += args.elems_per_blk;
    }
    x_out += x_stride;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce data and send data in the same node
  func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Reduce data and send data in the same node
    func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes1_multicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * local_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_multicast && args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_rank * args.elems_per_blk;
  int x_stride = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = x_stride * 2;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + x_ofst_chunk);
  T* mc_data_2 = mc_data + x_stride;
  T* mem_data_2 = mem_data + x_stride;
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, x_stride, mem_data](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T* dst = mem_data + ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size; i += block_size_y) {
      if (ofst_copy + base_data_idx < num_elems) {
        copy_data(dst + ofst_copy, x_in + ofst_copy);
      }
      ofst_copy += stride_copy;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_ns_send_data =
      [args, base_data_idx, tid, tid_y, x_ofst_chunk, mc_data, mem_signal, mc_signal, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    num_elems -= x_ofst_chunk;
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    T* dst = ns_data + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_data + vm_ofst_buffer);
        copy_data(dst, reg_data);
      }
    });
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
      T* ns_dst_src = dst - base_data_idx;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
  };
  auto func_ns_reduce_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk, ns_data, ns_signal, mc_data_2](
          T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
          int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        ns_reduce_data<args.use_inter>(reg_data, ns_data + ns_ofst_data, args.elems_per_blk,
                                       args.inter_size);
        replace_neg_zero(reg_data);
        multicast_store(mc_data_2 + vm_ofst_buffer, reg_data);
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T const* src = mem_data_2 + ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size; i += block_size_y) {
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src + ofst_copy);
        copy_data(x_out + ofst_copy, reg_data);
      }
      ofst_copy += stride_copy;
    }
    x_out += x_stride;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Reduce data across different nodes and send data in the same node (previous)
    func_ns_reduce_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and send data in the same node
  func_ns_reduce_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                   buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                   buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes1_unicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * (local_size - 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && !args.use_multicast && args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int x_ofst_chunk = args.local_rank * args.elems_per_blk;
  int x_stride = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = x_stride - args.elems_per_blk;
  int vm_buffer_size = vm_buffer_size_base * 2;
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  T* vm_x_out = x_out + x_ofst_chunk;
  T const* vm_x_in = x_in + x_ofst_chunk;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* mem_data_2 = mem_data + vm_buffer_size_base;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, x_ofst_chunk, x_stride,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    int ofst_copy_in = x_ofst_chunk;
    int64_t ofst_copy_out = ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size - 1; ++i) {
      CIRCULAR_ADD(ofst_copy_in, x_stride, args.elems_per_blk);
      if (ofst_copy_in + base_data_idx < num_elems) {
        copy_data(reg_data, x_in + ofst_copy_in);
        replace_neg_zero(reg_data);
        copy_data(uc_data_array[i] + ofst_copy_out, reg_data);
      }
      ofst_copy_out += args.elems_per_blk;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_ns_send_data =
      [&vm_x_in, args, base_data_idx, tid, x_ofst_chunk, x_stride, mem_data, ns_data, ns_signal](
          T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
          int ns_ofst_signal) -> void {
    num_elems -= x_ofst_chunk;
    T* dst = ns_data + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
    if (base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, vm_x_in, mem_data + vm_ofst_buffer,
                                      args.elems_per_blk);
      copy_data(dst, reg_data);
    }
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
      T* ns_dst_src = dst - base_data_idx;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    vm_x_in += x_stride;
  };
  auto func_ns_reduce_data_vm_send_data =
      [&buffer_info, &vm_x_out, args, base_data_idx, tid, x_ofst_chunk, x_stride,
       vm_buffer_size_base, ns_data, ns_signal,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    if (base_data_idx < num_elems) {
      ns_reduce_data<args.use_inter>(reg_data, ns_data + ns_ofst_data, args.elems_per_blk,
                                     args.inter_size);
      replace_neg_zero(reg_data);
      unicast_store<args.local_size - 1>(uc_data_array, reg_data,
                                         vm_ofst_buffer + vm_buffer_size_base, args.elems_per_blk);
      copy_data(vm_x_out, reg_data);
    }
    vm_x_out += x_stride;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, x_ofst_chunk, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int ofst_copy = x_ofst_chunk;
    T const* src = mem_data_2 + ofst_buffer;
#pragma unroll
    for (int i = 0; i < args.local_size - 1; ++i) {
      CIRCULAR_MINUS(ofst_copy, x_stride, args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src);
        copy_data(x_out + ofst_copy, reg_data);
      }
      src += args.elems_per_blk;
    }
    x_out += x_stride;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Reduce data across different nodes and send data in the same node (previous)
    func_ns_reduce_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and send data in the same node
  func_ns_reduce_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                   buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                   buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes2_multicast(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * world_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   2 * (inter_size - 1) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes2_mode(mode) && args.use_multicast && args.use_inter, "Invalid mode");
  int x_ofst_chunk_local = args.local_rank * args.elems_per_blk;
  int x_ofst_chunk = args.world_rank * args.elems_per_blk;
  int x_stride_local = args.local_size * args.elems_per_blk;
  int x_stride = args.world_size * args.elems_per_blk;
  int vm_buffer_size_base = x_stride * 2;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  int ns_data_size_base = (args.inter_size - 1) * args.elems_per_blk;
  int ns_data_size = ns_data_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + x_ofst_chunk_local);
  T* mc_data_2 = mc_data + x_stride;
  T* mem_data_2 = mem_data + x_stride;
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);
  T* ns_data_2 = ns_data + ns_data_size_base;
  uint64_t* ns_signal_2 = ns_signal + 1;

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, x_stride, x_stride_local,
                              mem_data](int64_t num_elems, int64_t ofst_buffer) -> void {
    int peer_inter_rank = args.inter_rank;
    int ofst_copy_base = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T* dst = mem_data + (ofst_buffer + ofst_copy_base);
    for (int i = 0; i < args.inter_size; ++i) {
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
      int ofst_copy = ofst_copy_base + peer_inter_rank * x_stride_local;
#pragma unroll
      for (int j = 0; j < args.local_size; j += block_size_y) {
        if (ofst_copy + base_data_idx < num_elems) {
          copy_data(dst, x_in + ofst_copy);
        }
        ofst_copy += stride_copy;
        dst += stride_copy;
      }
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_ns_send_data =
      [args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local, ns_data_size_base,
       mc_data, mem_signal, mc_signal, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    num_elems -= x_ofst_chunk_local;
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    T* dst = ns_data + ns_ofst_data;
    T const* mc_src = mc_data + vm_ofst_buffer;
    run_first_block_y<block_size_y>(tid_y, [&] {
      int peer_inter_rank = args.inter_rank;
      for (int i = 0; i < args.inter_size - 1; ++i) {
        CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
        if (peer_inter_rank * x_stride_local + base_data_idx < num_elems) {
          multicast_load_reduce(reg_data, mc_src);
          copy_data(dst + i * args.elems_per_blk, reg_data);
        }
        mc_src += x_stride_local;
      }
    });
    __syncthreads();
    if (tid < args.inter_size - 1) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = dst - base_data_idx + tid * args.elems_per_blk;
      T* ns_dst = ns_src + ns_data_size_base;
      int ns_msg_elems = floor_div<args.elems_per_thd>(min(
          num_elems - peer_inter_rank * x_stride_local, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      if (ns_msg_elems > 0) {
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (args.inter_rank * x_stride_local + base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_src);
      }
    });
  };
  auto func_ns_reduce_send_data = [args, base_data_idx, tid, tid_y, x_ofst_chunk, ns_data_size_base,
                                   ns_data_2, ns_signal, ns_signal_2,
                                   mc_data_2](T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                              int ns_ofst_signal, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (num_elems > 0) {
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
      __syncthreads();
      T* src = ns_data_2 + ns_ofst_data;
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          ns_reduce_data<args.use_inter, T, /*init_loaded=*/true>(reg_data, src, args.elems_per_blk,
                                                                  args.inter_size);
          replace_neg_zero(reg_data);
          copy_data(src, reg_data);
        }
      });
      __syncthreads();
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      if (tid < args.inter_size - 1) {
        int peer_inter_rank = args.inter_rank - (tid + 1);
        CIRCULAR_MIN_CLIP(peer_inter_rank, args.inter_size);
        T* ns_src = src - base_data_idx;
        T* ns_dst = ns_src - ns_data_size_base + tid * args.elems_per_blk;
        int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal_2 + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          multicast_store(mc_data_2 + vm_ofst_buffer, reg_data);
        }
      });
    }
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local, ns_data,
       ns_signal, ns_signal_2, mc_data_2](T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                          int ns_ofst_signal, int ns_ofst_reset,
                                          int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk_local;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(2);
      int ns_num_senders =
          min(ceil_div(num_elems, x_stride_local), static_cast<int64_t>(args.inter_size)) -
          (num_elems > args.inter_rank * x_stride_local);
      if (ns_num_senders > 0) {
        nvshmem_signal_wait_until(ns_signal_2 + ns_ofst_signal, NVSHMEM_CMP_EQ, ns_num_senders);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      int peer_inter_rank = args.inter_rank;
      T* dst = mc_data_2 + vm_ofst_buffer;
      T const* src = ns_data + ns_ofst_data;
      for (int i = 0; i < args.inter_size - 1; ++i) {
        CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
        dst += x_stride_local;
        if (peer_inter_rank * x_stride_local + base_data_idx < num_elems) {
          copy_data(reg_data, src);
          multicast_store(dst, reg_data);
        }
        src += args.elems_per_blk;
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride_local,
                                             x_stride, mem_data_2](T* reg_data, int64_t num_elems,
                                                                   int64_t ofst_buffer) -> void {
    int peer_inter_rank = args.inter_rank;
    int ofst_copy_base = tid_y * args.elems_per_blk;
    int stride_copy = block_size_y * args.elems_per_blk;
    T const* src = mem_data_2 + (ofst_buffer + ofst_copy_base);
    for (int i = 0; i < args.inter_size; ++i) {
      int ofst_copy = ofst_copy_base + peer_inter_rank * x_stride_local;
#pragma unroll
      for (int j = 0; j < args.local_size; j += block_size_y) {
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src);
          copy_data(x_out + ofst_copy, reg_data);
        }
        ofst_copy += stride_copy;
        src += stride_copy;
      }
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
    }
    x_out += x_stride;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce and send data across different nodes
  func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                           buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Receive data across different nodes and send data in the same node (previous)
    func_ns_receive_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data across different nodes
    func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                             buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allreduce_opt_bytes2_unicast(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   2 * inter_size * (local_size - 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   2 * (inter_size - 1) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes2_mode(mode) && !args.use_multicast && args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int x_ofst_chunk_local = args.local_rank * args.elems_per_blk;
  int x_ofst_chunk = args.world_rank * args.elems_per_blk;
  int x_stride_local = args.local_size * args.elems_per_blk;
  int x_stride = args.world_size * args.elems_per_blk;
  int vm_buffer_size_base_local = x_stride_local - args.elems_per_blk;
  int vm_buffer_size_base = args.inter_size * vm_buffer_size_base_local;
  int vm_buffer_size = vm_buffer_size_base * 2;
  int ns_data_size_base = x_stride - vm_buffer_size_base;
  int ns_data_size = ns_data_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  T* ns_x_out = x_out + x_ofst_chunk;
  T* vm_x_out = x_out + x_ofst_chunk_local;
  T const* ns_x_in = x_in + x_ofst_chunk;
  T const* vm_x_in = x_in + x_ofst_chunk_local;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* mem_data_2 = mem_data + vm_buffer_size_base;
  T* ns_data_2 = ns_data + ns_data_size_base;
  uint64_t* ns_signal_2 = ns_signal + 1;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, x_stride_local, x_stride,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    int peer_inter_rank = args.inter_rank;
    int64_t ofst_copy_out = ofst_buffer;
    for (int i = 0; i < args.inter_size; ++i) {
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
      int ofst_copy_in_base = peer_inter_rank * x_stride_local;
      int peer_local_rank = args.local_rank;
#pragma unroll
      for (int j = 0; j < args.local_size - 1; ++j) {
        CIRCULAR_ADD(peer_local_rank, args.local_size, 1);
        int ofst_copy_in = ofst_copy_in_base + peer_local_rank * args.elems_per_blk;
        if (ofst_copy_in + base_data_idx < num_elems) {
          copy_data(reg_data, x_in + ofst_copy_in);
          replace_neg_zero(reg_data);
          copy_data(uc_data_array[j] + ofst_copy_out, reg_data);
        }
        ofst_copy_out += args.elems_per_blk;
      }
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_ns_send_data =
      [&vm_x_in, args, base_data_idx, tid, x_ofst_chunk_local, x_stride_local, x_stride,
       vm_buffer_size_base_local, ns_data_size_base, mem_data, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    num_elems -= x_ofst_chunk_local;
    int peer_inter_rank = args.inter_rank;
    T* dst = ns_data + ns_ofst_data;
    T const* src = mem_data + vm_ofst_buffer;
    for (int i = 0; i < args.inter_size - 1; ++i) {
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
      if (peer_inter_rank * x_stride_local + base_data_idx < num_elems) {
        vm_reduce_data<args.local_size>(reg_data, vm_x_in + peer_inter_rank * x_stride_local, src,
                                        args.elems_per_blk);
        copy_data(dst + i * args.elems_per_blk, reg_data);
      }
      src += vm_buffer_size_base_local;
    }
    vm_x_in += x_stride;
    __syncthreads();
    if (tid < args.inter_size - 1) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = dst - base_data_idx + tid * args.elems_per_blk;
      T* ns_dst = ns_src + ns_data_size_base;
      int ns_msg_elems = floor_div<args.elems_per_thd>(min(
          num_elems - peer_inter_rank * x_stride_local, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      if (ns_msg_elems > 0) {
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
  };
  auto func_ns_reduce_send_data =
      [&ns_x_in, &ns_x_out, args, base_data_idx, tid, x_ofst_chunk, x_stride,
       vm_buffer_size_base_local, vm_buffer_size_base, ns_data_size_base, ns_data_2, ns_signal,
       ns_signal_2, mem_data, uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                             int ns_ofst_signal, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (base_data_idx < num_elems) {
      T const* src =
          mem_data + (vm_ofst_buffer + (args.inter_size - 1) * vm_buffer_size_base_local);
      vm_reduce_data<args.local_size>(reg_data, ns_x_in, src, args.elems_per_blk);
    }
    ns_x_in += x_stride;
    if (num_elems > 0) {
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
      __syncthreads();
      T* src = ns_data_2 + ns_ofst_data;
      if (base_data_idx < num_elems) {
        ns_reduce_data<args.use_inter, T, /*init_loaded=*/true>(reg_data, src, args.elems_per_blk,
                                                                args.inter_size);
        replace_neg_zero(reg_data);
        copy_data(src, reg_data);
      }
      __syncthreads();
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      if (tid < args.inter_size - 1) {
        int peer_inter_rank = args.inter_rank - (tid + 1);
        CIRCULAR_MIN_CLIP(peer_inter_rank, args.inter_size);
        T* ns_src = src - base_data_idx;
        T* ns_dst = ns_src - ns_data_size_base + tid * args.elems_per_blk;
        int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal_2 + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      if (base_data_idx < num_elems) {
        unicast_store<args.local_size - 1>(
            uc_data_array, reg_data, vm_ofst_buffer + vm_buffer_size_base, args.elems_per_blk);
        copy_data(ns_x_out, reg_data);
      }
    }
    ns_x_out += x_stride;
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, &vm_x_out, args, base_data_idx, tid, x_ofst_chunk_local, x_stride_local,
       x_stride, vm_buffer_size_base_local, vm_buffer_size_base, ns_data, ns_signal, ns_signal_2,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk_local;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(2);
      int ns_num_senders =
          min(ceil_div(num_elems, x_stride_local), static_cast<int64_t>(args.inter_size)) -
          (num_elems > args.inter_rank * x_stride_local);
      if (ns_num_senders > 0) {
        nvshmem_signal_wait_until(ns_signal_2 + ns_ofst_signal, NVSHMEM_CMP_EQ, ns_num_senders);
      }
    }
    __syncthreads();
    int peer_inter_rank = args.inter_rank;
    int64_t ofst_copy_uc = vm_ofst_buffer + vm_buffer_size_base;
    T const* src = ns_data + ns_ofst_data;
    for (int i = 0; i < args.inter_size - 1; ++i) {
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
      ofst_copy_uc += vm_buffer_size_base_local;
      int ofst_copy_out = peer_inter_rank * x_stride_local;
      if (ofst_copy_out + base_data_idx < num_elems) {
        copy_data(reg_data, src);
        unicast_store<args.local_size - 1>(uc_data_array, reg_data, ofst_copy_uc,
                                           args.elems_per_blk);
        copy_data(vm_x_out + ofst_copy_out, reg_data);
      }
      src += args.elems_per_blk;
    }
    vm_x_out += x_stride;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, x_stride_local, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int peer_inter_rank = args.inter_rank;
    T const* src = mem_data_2 + ofst_buffer;
    for (int i = 0; i < args.inter_size; ++i) {
      int peer_local_rank = args.local_rank;
      int ofst_copy_base = peer_inter_rank * x_stride_local;
#pragma unroll
      for (int j = 0; j < args.local_size - 1; ++j) {
        CIRCULAR_MINUS(peer_local_rank, args.local_size, 1);
        int ofst_copy = ofst_copy_base + peer_local_rank * args.elems_per_blk;
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src);
          copy_data(x_out + ofst_copy, reg_data);
        }
        src += args.elems_per_blk;
      }
      CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
    }
    x_out += x_stride;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce and send data across different nodes
  func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                           buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Receive data across different nodes and send data in the same node (previous)
    func_ns_receive_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data across different nodes
    func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                             buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void allreduce_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args) {
  constexpr MixedCommOp op = MixedCommOp::ALLREDUCE;
  static_assert(is_valid_op<args.use_local_tp, args.use_local_dp, use_inter_tp, use_inter_dp>(op),
                "Invalid op");
  static_assert(is_valid_mode<args.use_local_tp, use_inter_tp>(mode), "Invalid mode");
  static_assert(is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode),
                "Invalid block_size_y");
  if constexpr (is_opt_waits_mode(mode)) {
    if constexpr (!args.use_inter) {
      allreduce_opt_waits_single<block_size_y>(args);
    } else {
      allreduce_opt_waits_multi<block_size_y>(args);
    }
  } else if constexpr (is_opt_bytes1_mode(mode)) {
    if constexpr (!args.use_inter) {
      if constexpr (args.use_multicast) {
        allreduce_opt_bytes1_multicast_single<block_size_y>(args);
      } else {
        allreduce_opt_bytes1_unicast_single<block_size_y>(args);
      }
    } else {
      if constexpr (args.use_multicast) {
        allreduce_opt_bytes1_multicast_multi<block_size_y>(args);
      } else {
        allreduce_opt_bytes1_unicast_multi<block_size_y>(args);
      }
    }
  } else {
    static_assert(is_opt_bytes2_mode(mode) && args.use_inter, "Invalid mode");
    if constexpr (args.use_multicast) {
      allreduce_opt_bytes2_multicast<block_size_y>(args);
    } else {
      allreduce_opt_bytes2_unicast<block_size_y>(args);
    }
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allgather_multicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  static_assert(args.use_multicast && !args.use_inter, "Invalid mode");
  int vm_buffer_size = args.local_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, tid_y, mc_data](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        copy_data(reg_data, x_in);
        replace_neg_zero(reg_data);
        multicast_store(mc_data + ofst_buffer, reg_data);
      }
    });
    x_in += args.elems_per_blk;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, mem_data](
                                                T* reg_data, int64_t num_elems,
                                                int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int64_t stride_copy_out = block_size_y * args.num_elems_all;
      int stride_copy_in = block_size_y * args.elems_per_blk;
      T* dst = x_out + tid_y * args.num_elems_all;
      T const* src = mem_data + (ofst_buffer + tid_y * args.elems_per_blk);
#pragma unroll
      for (int i = 0; i < args.local_size; i += block_size_y) {
        vm_receive_data(reg_data, src);
        copy_data(dst, reg_data);
        dst += stride_copy_out;
        src += stride_copy_in;
      }
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allgather_unicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_size - 1) * elems_per_blk * sizeof(T)
  static_assert(!args.use_multicast && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size = (args.local_size - 1) * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);
  T* vm_x_out = x_out + args.local_rank * args.num_elems_all;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);

  auto func_vm_read_inputs_send_data = [&x_in, &vm_x_out, args, base_data_idx, uc_data_array](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      copy_data(reg_data, x_in);
      replace_neg_zero(reg_data);
      unicast_store<args.local_size - 1>(uc_data_array, reg_data, ofst_buffer, args.elems_per_blk);
      copy_data(vm_x_out, reg_data);
    }
    x_in += args.elems_per_blk;
    vm_x_out += args.elems_per_blk;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, mem_data](
                                                T* reg_data, int64_t num_elems,
                                                int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_local_rank = args.local_rank;
      T const* src = mem_data + ofst_buffer;
#pragma unroll
      for (int i = 0; i < args.local_size - 1; ++i) {
        CIRCULAR_MINUS(peer_local_rank, args.local_size, 1);
        vm_receive_data(reg_data, src);
        copy_data(x_out + peer_local_rank * args.num_elems_all, reg_data);
        src += args.elems_per_blk;
      }
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allgather_multicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   world_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(args.use_multicast && args.use_inter, "Invalid mode");
  int vm_buffer_size_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size = args.inter_size * vm_buffer_size_local;
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  int64_t num_elems_out_all_local = args.local_size * args.num_elems_all;

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);

  auto func_ns_read_inputs_send_data = [&x_in, args, base_data_idx, tid, tid_y, ns_data, ns_signal,
                                        mc_data](T* reg_data, int64_t num_elems,
                                                 int64_t ns_ofst_data, int ns_ofst_signal,
                                                 int64_t vm_ofst_buffer) -> void {
    T* src = ns_data + ns_ofst_data;
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        copy_data(reg_data, x_in);
        replace_neg_zero(reg_data);
        copy_data(src, reg_data);
      }
    });
    __syncthreads();
    if (tid < args.inter_size - 1 && num_elems > 0) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = src - base_data_idx;
      T* ns_dst = ns_src + (tid + 1) * args.elems_per_blk;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        multicast_store(mc_data + vm_ofst_buffer, reg_data);
      }
    });
    x_in += args.elems_per_blk;
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, vm_buffer_size_local, ns_data, ns_signal,
       mc_data](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        T* dst = mc_data + vm_ofst_buffer;
        T const* src = ns_data + ns_ofst_data;
        for (int i = 0; i < args.inter_size - 1; ++i) {
          dst += vm_buffer_size_local;
          src += args.elems_per_blk;
          copy_data(reg_data, src);
          multicast_store(dst, reg_data);
        }
      }
    });
  };
  auto func_vm_receive_data_write_outputs =
      [&x_out, args, base_data_idx, tid_y, num_elems_out_all_local, mem_data](
          T* reg_data, int64_t num_elems, int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      int64_t stride_copy_out = block_size_y * args.num_elems_all;
      int stride_copy_in = block_size_y * args.elems_per_blk;
      T* dst_base = x_out + tid_y * args.num_elems_all;
      T const* src = mem_data + (ofst_buffer + tid_y * args.elems_per_blk);
      for (int i = 0; i < args.inter_size; ++i) {
        T* dst = dst_base + peer_inter_rank * num_elems_out_all_local;
#pragma unroll
        for (int j = 0; j < args.local_size; j += block_size_y) {
          vm_receive_data(reg_data, src);
          copy_data(dst, reg_data);
          dst += stride_copy_out;
          src += stride_copy_in;
        }
        CIRCULAR_MINUS(peer_inter_rank, args.inter_size, 1);
      }
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data across different nodes
  func_ns_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data across different nodes
    func_ns_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                  buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Receive data across different nodes and send data in the same node
    func_ns_receive_data_vm_send_data(
        reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
        buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void allgather_unicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   inter_size * (local_size - 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(!args.use_multicast && args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size_local = (args.local_size - 1) * args.elems_per_blk;
  int vm_buffer_size = args.inter_size * vm_buffer_size_local;
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  T* ns_x_out = x_out + args.world_rank * args.num_elems_all;
  T* vm_x_out = x_out + args.local_rank * args.num_elems_all;
  int64_t num_elems_out_all_local = args.local_size * args.num_elems_all;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);

  auto func_ns_read_inputs_send_data = [&x_in, &ns_x_out, args, base_data_idx, tid, ns_data,
                                        ns_signal, uc_data_array](
                                           T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                           int ns_ofst_signal, int64_t vm_ofst_buffer) -> void {
    T* src = ns_data + ns_ofst_data;
    if (base_data_idx < num_elems) {
      copy_data(reg_data, x_in);
      replace_neg_zero(reg_data);
      copy_data(src, reg_data);
    }
    __syncthreads();
    if (tid < args.inter_size - 1 && num_elems > 0) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = src - base_data_idx;
      T* ns_dst = ns_src + (tid + 1) * args.elems_per_blk;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    if (base_data_idx < num_elems) {
      unicast_store<args.local_size - 1>(uc_data_array, reg_data, vm_ofst_buffer,
                                         args.elems_per_blk);
      copy_data(ns_x_out, reg_data);
    }
    x_in += args.elems_per_blk;
    ns_x_out += args.elems_per_blk;
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, &vm_x_out, args, base_data_idx, tid, vm_buffer_size_local,
       num_elems_out_all_local, ns_data, ns_signal,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      int64_t ofst_copy = vm_ofst_buffer;
      T const* src = ns_data + ns_ofst_data;
      for (int i = 0; i < args.inter_size - 1; ++i) {
        CIRCULAR_MINUS(peer_inter_rank, args.inter_size, 1);
        ofst_copy += vm_buffer_size_local;
        src += args.elems_per_blk;
        copy_data(reg_data, src);
        unicast_store<args.local_size - 1>(uc_data_array, reg_data, ofst_copy, args.elems_per_blk);
        copy_data(vm_x_out + peer_inter_rank * num_elems_out_all_local, reg_data);
      }
    }
    vm_x_out += args.elems_per_blk;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, num_elems_out_all_local,
                                             mem_data](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      T const* src = mem_data + ofst_buffer;
      for (int i = 0; i < args.inter_size; ++i) {
        int peer_local_rank = args.local_rank;
        T* dst = x_out + peer_inter_rank * num_elems_out_all_local;
#pragma unroll
        for (int j = 0; j < args.local_size - 1; ++j) {
          CIRCULAR_MINUS(peer_local_rank, args.local_size, 1);
          vm_receive_data(reg_data, src);
          copy_data(dst + peer_local_rank * args.num_elems_all, reg_data);
          src += args.elems_per_blk;
        }
        CIRCULAR_MINUS(peer_inter_rank, args.inter_size, 1);
      }
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data across different nodes
  func_ns_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data across different nodes
    func_ns_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                  buffer_info.ns_ofst_signal(), buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Receive data across different nodes and send data in the same node
    func_ns_receive_data_vm_send_data(
        reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
        buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void allgather_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args) {
  constexpr MixedCommOp op = MixedCommOp::ALLGATHER;
  static_assert(is_valid_op<args.use_local_tp, args.use_local_dp, use_inter_tp, use_inter_dp>(op),
                "Invalid op");
  static_assert(is_valid_mode<args.use_local_tp, use_inter_tp>(mode), "Invalid mode");
  static_assert(is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode),
                "Invalid block_size_y");
  if constexpr (!args.use_inter) {
    if constexpr (args.use_multicast) {
      allgather_multicast_single<block_size_y>(args);
    } else {
      allgather_unicast_single<block_size_y>(args);
    }
  } else {
    if constexpr (args.use_multicast) {
      allgather_multicast_multi<block_size_y>(args);
    } else {
      allgather_unicast_multi<block_size_y>(args);
    }
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void reducescatter_multicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  static_assert(args.use_multicast && !args.use_inter, "Invalid mode");
  int vm_buffer_size_base = args.local_size * args.elems_per_blk;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, tid, mem_data, mc_signal](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int stride_copy_out = block_size_y * args.elems_per_blk;
      int64_t stride_copy_in = block_size_y * args.num_elems_all;
      T* dst = mem_data + (ofst_buffer + tid_y * args.elems_per_blk);
      T const* src = x_in + tid_y * args.num_elems_all;
#pragma unroll
      for (int i = 0; i < args.local_size; i += block_size_y) {
        copy_data(dst, src);
        dst += stride_copy_out;
        src += stride_copy_in;
      }
    }
    x_in += args.elems_per_blk;
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + ofst_buffer));
    }
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, tid, tid_y, mc_data,
                                            mem_signal](T* reg_data, int64_t num_elems,
                                                        int64_t ofst_buffer) -> void {
    if (tid == 0) {
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_data + ofst_buffer);
        copy_data(x_out, reg_data);
      }
    });
    x_out += args.elems_per_blk;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    int64_t vm_ofst_buffer_reset = buffer_info.vm_ofst_buffer_prev();
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reset previous buffer (previous)
    func_vm_reset_buffer(vm_ofst_buffer_reset);
    // Reduce data in the same node and write outputs
    func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  }
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void reducescatter_unicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_size - 1) * elems_per_blk * sizeof(T)
  static_assert(!args.use_multicast && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size = (args.local_size - 1) * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);
  T const* vm_x_in = x_in + args.local_rank * args.num_elems_all;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, uc_data_array](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_local_rank = args.local_rank;
      int64_t ofst_copy = ofst_buffer;
#pragma unroll
      for (int i = 0; i < args.local_size - 1; ++i) {
        CIRCULAR_ADD(peer_local_rank, args.local_size, 1);
        copy_data(reg_data, x_in + peer_local_rank * args.num_elems_all);
        replace_neg_zero(reg_data);
        copy_data(uc_data_array[i] + ofst_copy, reg_data);
        ofst_copy += args.elems_per_blk;
      }
    }
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, &vm_x_in, args, base_data_idx, mem_data](
                                               T* reg_data, int64_t num_elems,
                                               int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, vm_x_in, mem_data + ofst_buffer,
                                      args.elems_per_blk);
      copy_data(x_out, reg_data);
    }
    vm_x_in += args.elems_per_blk;
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and write outputs (previous)
    func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void reducescatter_multicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   world_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   2 * (inter_size - 1) * elems_per_blk * sizeof(T)
  static_assert(args.use_multicast && args.use_inter, "Invalid mode");
  int vm_buffer_size_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = args.inter_size * vm_buffer_size_local;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  int ns_data_size_base = (args.inter_size - 1) * args.elems_per_blk;
  int ns_data_size = ns_data_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  int64_t num_elems_in_all_local = args.local_size * args.num_elems_all;

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);
  T* ns_data_2 = ns_data + ns_data_size_base;

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, num_elems_in_all_local, mem_data](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      int stride_copy_out = block_size_y * args.elems_per_blk;
      int64_t stride_copy_in = block_size_y * args.num_elems_all;
      T* dst = mem_data + (ofst_buffer + tid_y * args.elems_per_blk);
      T const* src_base = x_in + tid_y * args.num_elems_all;
      for (int i = 0; i < args.inter_size; ++i) {
        CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
        T const* src = src_base + peer_inter_rank * num_elems_in_all_local;
#pragma unroll
        for (int j = 0; j < args.local_size; j += block_size_y) {
          copy_data(dst, src);
          dst += stride_copy_out;
          src += stride_copy_in;
        }
      }
    }
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_ns_send_data =
      [args, base_data_idx, tid, tid_y, vm_buffer_size_local, ns_data_size_base, mc_data,
       mem_signal, mc_signal, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    T* dst = ns_data + ns_ofst_data;
    T* mc_src = mc_data + vm_ofst_buffer;
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        for (int i = 0; i < args.inter_size - 1; ++i) {
          multicast_load_reduce(reg_data, mc_src);
          copy_data(dst + i * args.elems_per_blk, reg_data);
          mc_src += vm_buffer_size_local;
        }
      }
    });
    __syncthreads();
    if (tid < args.inter_size - 1 && num_elems > 0) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = dst - base_data_idx + tid * args.elems_per_blk;
      T* ns_dst = ns_src + ns_data_size_base;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_src);
      }
    });
  };
  auto func_ns_reduce_data_write_outputs = [&buffer_info, &x_out, args, base_data_idx, tid, tid_y,
                                            ns_data_2, ns_signal](
                                               T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                               int ns_ofst_signal, int ns_ofst_reset) -> void {
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        ns_reduce_data<args.use_inter, T, /*init_loaded=*/true>(
            reg_data, ns_data_2 + ns_ofst_data, args.elems_per_blk, args.inter_size);
        copy_data(x_out, reg_data);
      }
    });
    x_out += args.elems_per_blk;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Reduce data across different nodes and write outputs (previous)
    func_ns_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.ns_ofst_data_prev(),
                                      buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset);
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and write outputs
  func_ns_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(),
                                    buffer_info.ns_ofst_signal_prev());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void reducescatter_unicast_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   inter_size * (local_size - 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   2 * (inter_size - 1) * elems_per_blk * sizeof(T)
  static_assert(!args.use_multicast && args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int vm_buffer_size_local = (args.local_size - 1) * args.elems_per_blk;
  int vm_buffer_size = args.inter_size * vm_buffer_size_local;
  int ns_data_size_base = (args.inter_size - 1) * args.elems_per_blk;
  int ns_data_size = ns_data_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);
  T const* ns_x_in = x_in + args.world_rank * args.num_elems_all;
  T const* vm_x_in = x_in + args.local_rank * args.num_elems_all;
  int64_t num_elems_in_all_local = args.local_size * args.num_elems_all;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* ns_data_2 = ns_data + ns_data_size_base;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, num_elems_in_all_local,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      int64_t ofst_copy = ofst_buffer;
      for (int i = 0; i < args.inter_size; ++i) {
        CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
        T const* src = x_in + peer_inter_rank * num_elems_in_all_local;
        int peer_local_rank = args.local_rank;
#pragma unroll
        for (int j = 0; j < args.local_size - 1; ++j) {
          CIRCULAR_ADD(peer_local_rank, args.local_size, 1);
          copy_data(reg_data, src + peer_local_rank * args.num_elems_all);
          replace_neg_zero(reg_data);
          copy_data(uc_data_array[j] + ofst_copy, reg_data);
          ofst_copy += args.elems_per_blk;
        }
      }
    }
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_ns_send_data =
      [&vm_x_in, args, base_data_idx, tid, vm_buffer_size_local, ns_data_size_base,
       num_elems_in_all_local, mem_data, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    T* dst = ns_data + ns_ofst_data;
    T const* src = mem_data + vm_ofst_buffer;
    if (base_data_idx < num_elems) {
      int peer_inter_rank = args.inter_rank;
      for (int i = 0; i < args.inter_size - 1; ++i) {
        CIRCULAR_ADD(peer_inter_rank, args.inter_size, 1);
        vm_reduce_data<args.local_size>(
            reg_data, vm_x_in + peer_inter_rank * num_elems_in_all_local, src, args.elems_per_blk);
        copy_data(dst + i * args.elems_per_blk, reg_data);
        src += vm_buffer_size_local;
      }
    }
    vm_x_in += args.elems_per_blk;
    __syncthreads();
    if (tid < args.inter_size - 1 && num_elems > 0) {
      int peer_inter_rank = args.inter_rank + tid + 1;
      CIRCULAR_MAX_CLIP(peer_inter_rank, args.inter_size);
      T* ns_src = dst - base_data_idx + tid * args.elems_per_blk;
      T* ns_dst = ns_src + ns_data_size_base;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + peer_inter_rank * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
  };
  auto func_ns_reduce_data_write_outputs =
      [&buffer_info, &x_out, &ns_x_in, args, base_data_idx, tid, vm_buffer_size_local, ns_data_2,
       ns_signal, mem_data](T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                            int ns_ofst_signal, int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      T const* src = mem_data + (vm_ofst_buffer + (args.inter_size - 1) * vm_buffer_size_local);
      vm_reduce_data<args.local_size>(reg_data, ns_x_in, src, args.elems_per_blk);
    }
    ns_x_in += args.elems_per_blk;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    if (base_data_idx < num_elems) {
      ns_reduce_data<args.use_inter, T, /*init_loaded=*/true>(reg_data, ns_data_2 + ns_ofst_data,
                                                              args.elems_per_blk, args.inter_size);
      copy_data(x_out, reg_data);
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce data in the same node and send data across different nodes
  func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                   buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data across different nodes and write outputs (previous)
    func_ns_reduce_data_write_outputs(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and write outputs
  func_ns_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void reducescatter_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args) {
  constexpr MixedCommOp op = MixedCommOp::REDUCESCATTER;
  static_assert(is_valid_op<args.use_local_tp, args.use_local_dp, use_inter_tp, use_inter_dp>(op),
                "Invalid op");
  static_assert(is_valid_mode<args.use_local_tp, use_inter_tp>(mode), "Invalid mode");
  static_assert(is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode),
                "Invalid block_size_y");
  if constexpr (args.use_multicast) {
    if constexpr (!args.use_inter) {
      reducescatter_multicast_single<block_size_y>(args);
    } else {
      reducescatter_multicast_multi<block_size_y>(args);
    }
  } else {
    if constexpr (!args.use_inter) {
      reducescatter_unicast_single<block_size_y>(args);
    } else {
      reducescatter_unicast_multi<block_size_y>(args);
    }
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_waits_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && !args.use_inter, "Invalid mode");
  int vm_buffer_size = args.local_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
  T *mc_data, *uc_data_array[args.local_size];
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data_self);
  }

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, tid_y, mc_data, uc_data_array](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        copy_data(reg_data, x_in);
        replace_neg_zero(reg_data);
        if constexpr (args.use_multicast) {
          multicast_store(mc_data + ofst_buffer, reg_data);
        } else {
          unicast_store<args.local_size>(uc_data_array, reg_data, ofst_buffer);
        }
      }
    });
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, tid_y, mem_data](
                                               T* reg_data, int64_t num_elems,
                                               int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int stride_reduce = local_tp_size * args.elems_per_blk;
      int64_t stride_copy_out = block_size_y * args.num_elems_all;
      int stride_copy_in = block_size_y * stride_reduce;
      T* dst = x_out + tid_y * args.num_elems_all;
      T const* src = mem_data + (ofst_buffer + tid_y * stride_reduce);
#pragma unroll
      for (int i = 0; i < local_dp_size; i += block_size_y) {
        vm_reduce_data<local_tp_size>(reg_data, src, args.elems_per_blk);
        copy_data(dst, reg_data);
        dst += stride_copy_out;
        src += stride_copy_in;
      }
    }
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and write outputs (previous)
    func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_waits_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   inter_dp_size * local_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && args.use_inter, "Invalid mode");
  if constexpr (local_dp_size == 1) {
    static_assert(block_size_y == 1, "Invalid block_size_y");
    int vm_buffer_size = args.local_size * args.elems_per_blk;
    int ns_data_size = args.inter_size * args.elems_per_blk;
    INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

    int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
    T *mc_data, *uc_data_array[args.local_size];
    if constexpr (args.use_multicast) {
      mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
    } else {
      set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all,
                                         vm_ofst_data_self);
    }

    auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, mc_data, uc_data_array](
                                             T* reg_data, int64_t num_elems,
                                             int64_t ofst_buffer) -> void {
      if (base_data_idx < num_elems) {
        copy_data(reg_data, x_in);
        replace_neg_zero(reg_data);
        if constexpr (args.use_multicast) {
          multicast_store(mc_data + ofst_buffer, reg_data);
        } else {
          unicast_store<args.local_size>(uc_data_array, reg_data, ofst_buffer);
        }
      }
      x_in += args.elems_per_blk;
    };
    auto func_vm_reduce_data_ns_send_data =
        [args, base_data_idx, tid, mem_data, ns_data, ns_signal](
            T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
            int ns_ofst_signal) -> void {
      T* dst = ns_data + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
      if (base_data_idx < num_elems) {
        vm_reduce_data<args.local_size>(reg_data, mem_data + vm_ofst_buffer, args.elems_per_blk);
        copy_data(dst, reg_data);
      }
      __syncthreads();
      if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
        T* ns_dst_src = dst - base_data_idx;
        int ns_msg_elems =
            floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
        int peer_world_rank = args.local_rank + tid * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ns_ofst_signal,
                                  1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    };
    auto func_ns_reduce_data_write_outputs = [&buffer_info, &x_out, args, base_data_idx, tid,
                                              ns_data, ns_signal](
                                                 T* reg_data, int64_t num_elems, int64_t ofst_data,
                                                 int ofst_signal, int ofst_reset) -> void {
      if (tid == 0) {
        buffer_info.reset_ns_signal(ns_signal + ofst_reset);
        buffer_info.update_ns_reset_size(1);
        if (num_elems > 0) {
          nvshmem_signal_wait_until(ns_signal + ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
        }
      }
      __syncthreads();
      if (base_data_idx < num_elems) {
        int stride_reduce = args.inter_tp_size * args.elems_per_blk;
        T* dst = x_out;
        T const* src = ns_data + ofst_data;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          ns_reduce_data<use_inter_tp>(reg_data, src, args.elems_per_blk, args.inter_tp_size);
          copy_data(dst, reg_data);
          dst += args.num_elems_all;
          src += stride_reduce;
        }
      }
      x_out += args.elems_per_blk;
    };

    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and send data across different nodes
    func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                     buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Main loop
    for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
      int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
      UPDATE_BASIC_VARIABLES(args.elems_per_blk);
      // Read inputs and send data in the same node
      func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
      // Reduce data in the same node and send data across different nodes
      func_vm_reduce_data_ns_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(),
                                       buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal());
      // Reduce data across different nodes and write outputs (previous)
      func_ns_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                        buffer_info.ns_ofst_data_prev(),
                                        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset);
      // Reset previous buffer
      func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    }
    // Reduce data across different nodes and write outputs
    func_ns_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                      buffer_info.ns_ofst_signal(),
                                      buffer_info.ns_ofst_signal_prev());
    buffer_info.write_gmem(tid);
  } else {
    int vm_buffer_size = args.inter_dp_size * args.local_size * args.elems_per_blk;
    int ns_data_size = args.inter_size * args.elems_per_blk;
    INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

    int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
    T *mc_data, *uc_data_array[args.local_size];
    if constexpr (args.use_multicast) {
      mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
    } else {
      set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all,
                                         vm_ofst_data_self);
    }

    auto func_ns_read_inputs_send_data = [&x_in, args, base_data_idx, tid, tid_y, ns_data,
                                          ns_signal](int64_t num_elems, int64_t ofst_data,
                                                     int ofst_signal) -> void {
      T* dst = ns_data + (ofst_data + args.inter_rank * args.elems_per_blk);
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          copy_data(dst, x_in);
        }
      });
      __syncthreads();
      if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
        T* ns_dst_src = dst - base_data_idx;
        int ns_msg_elems =
            floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
        int peer_world_rank = args.local_rank + tid * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      x_in += args.elems_per_blk;
    };
    auto func_ns_reduce_data_vm_send_data =
        [&buffer_info, args, base_data_idx, tid, tid_y, ns_data, ns_signal, mc_data, uc_data_array](
            T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
            int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
      if (tid == 0) {
        buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
        buffer_info.update_ns_reset_size(1);
        if (num_elems > 0) {
          nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ,
                                    args.inter_size - 1);
        }
      }
      __syncthreads();
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          int64_t ofst_copy = vm_ofst_buffer;
          int stride_copy = args.local_size * args.elems_per_blk;
          int stride_reduce = args.inter_tp_size * args.elems_per_blk;
          T const* src = ns_data + ns_ofst_data;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            ns_reduce_data<use_inter_tp>(reg_data, src, args.elems_per_blk, args.inter_tp_size);
            replace_neg_zero(reg_data);
            if constexpr (args.use_multicast) {
              multicast_store(mc_data + ofst_copy, reg_data);
            } else {
              unicast_store<args.local_size>(uc_data_array, reg_data, ofst_copy);
            }
            ofst_copy += stride_copy;
            src += stride_reduce;
          }
        }
      });
    };
    auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, tid_y, mem_data](
                                                 T* reg_data, int64_t num_elems,
                                                 int64_t ofst_buffer) -> void {
      if (base_data_idx < num_elems) {
        int stride_reduce = local_tp_size * args.elems_per_blk;
        int64_t stride_copy_out = block_size_y * args.num_elems_all;
        int stride_copy_in = block_size_y * stride_reduce;
        T* dst = x_out + tid_y * args.num_elems_all;
        T const* src = mem_data + (ofst_buffer + tid_y * stride_reduce);
        for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_dp_size; j += block_size_y) {
            vm_reduce_data<local_tp_size>(reg_data, src, args.elems_per_blk);
            copy_data(dst, reg_data);
            dst += stride_copy_out;
            src += stride_copy_in;
          }
        }
      }
      x_out += args.elems_per_blk;
    };

    // Read inputs and send data across different nodes
    func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                  buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Reduce data across different nodes and send data in the same node
    func_ns_reduce_data_vm_send_data(
        reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
        buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
    // Main loop
    for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
      UPDATE_BASIC_VARIABLES(args.elems_per_blk);
      // Read inputs and send data across different nodes
      func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal());
      // Reduce data in the same node and write outputs (previous)
      func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                        buffer_info.vm_ofst_buffer_prev());
      // Reduce data across different nodes and send data in the same node
      func_ns_reduce_data_vm_send_data(
          reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
          buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
      // Reset previous buffer
      func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    }
    // Reduce data in the same node and write outputs
    func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    buffer_info.write_gmem(tid);
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_bytes1_multicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_tp_size + local_size) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_multicast && !args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = x_stride;
  int vm_buffer_size_base_2 = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* mc_data = args.mc_buffer_tp_all + (vm_ofst_data + x_ofst_chunk);
  T* mc_data_2 = args.mc_buffer_full_all +
                 (vm_ofst_data + vm_buffer_size_base_1 + args.local_rank * args.elems_per_blk);
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_tp_all + vm_ofst_signal);

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, x_stride, mem_data](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    T* dst = mem_data + ofst_buffer;
    if constexpr (block_size_y > local_tp_size) {
      if (tid_y < local_tp_size) {
        if (ofst_copy + base_data_idx < num_elems) {
          copy_data(dst + ofst_copy, x_in + ofst_copy);
        }
      }
    } else {
      int stride_copy = block_size_y * args.elems_per_blk;
#pragma unroll
      for (int i = 0; i < local_tp_size; i += block_size_y) {
        if (ofst_copy + base_data_idx < num_elems) {
          copy_data(dst + ofst_copy, x_in + ofst_copy);
        }
        ofst_copy += stride_copy;
      }
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_send_data = [base_data_idx, tid, tid_y, x_ofst_chunk, mc_data, mc_data_2,
                                   mem_signal, mc_signal](T* reg_data, int64_t num_elems,
                                                          int64_t ofst_buffer) -> void {
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(local_tp_size));
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (x_ofst_chunk + base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_data + ofst_buffer);
        replace_neg_zero(reg_data);
        multicast_store(mc_data_2 + ofst_buffer, reg_data);
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    if constexpr (block_size_y > local_tp_size) {
      int tid_y_dp = floor_div<local_tp_size>(tid_y);
      int tid_y_tp = mod<local_tp_size>(tid_y);
      int ofst_copy = tid_y_tp * args.elems_per_blk;
      T* dst = x_out + (tid_y_dp * args.num_elems_all + ofst_copy);
      T const* src = mem_data_2 + (ofst_buffer + tid_y * args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
        int64_t stride_copy_out = stride_dp_size * args.num_elems_all;
        int stride_copy_in = stride_dp_size * x_stride;
#pragma unroll
        for (int i = 0; i < local_dp_size; i += stride_dp_size) {
          vm_receive_data(reg_data, src);
          copy_data(dst, reg_data);
          dst += stride_copy_out;
          src += stride_copy_in;
        }
      }
    } else {
      int ofst_copy_base = tid_y * args.elems_per_blk;
      int stride_copy = block_size_y * args.elems_per_blk;
      T* dst = x_out;
      T const* src = mem_data_2 + ofst_buffer;
#pragma unroll
      for (int i = 0; i < local_dp_size; ++i) {
        int ofst_copy = ofst_copy_base;
#pragma unroll
        for (int j = 0; j < local_tp_size; j += block_size_y) {
          if (ofst_copy + base_data_idx < num_elems) {
            vm_receive_data(reg_data, src + ofst_copy);
            copy_data(dst + ofst_copy, reg_data);
            ofst_copy += stride_copy;
          }
        }
        dst += args.num_elems_all;
        src += x_stride;
      }
    }
    x_out += x_stride;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce and send data in the same node
  func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data in the same node
    func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_bytes1_unicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_tp_size + local_size - 2) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && !args.use_multicast && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = x_stride - args.elems_per_blk;
  int vm_buffer_size_base_2 = (args.local_size - 1) * args.elems_per_blk;
  int vm_buffer_size = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);
  T* vm_x_out = x_out + x_ofst_chunk + args.local_dp_rank * args.num_elems_all;
  T const* vm_x_in = x_in + x_ofst_chunk;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, x_ofst_chunk, x_stride,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    int ofst_copy_in = x_ofst_chunk;
    int64_t ofst_copy_out = ofst_buffer;
    T* const* uc_data_array_val = uc_data_array + (local_dp_size - 1) * local_tp_size;
#pragma unroll
    for (int i = 0; i < local_tp_size - 1; ++i) {
      CIRCULAR_ADD(ofst_copy_in, x_stride, args.elems_per_blk);
      if (ofst_copy_in + base_data_idx < num_elems) {
        copy_data(reg_data, x_in + ofst_copy_in);
        replace_neg_zero(reg_data);
        copy_data(uc_data_array_val[i] + ofst_copy_out, reg_data);
      }
      ofst_copy_out += args.elems_per_blk;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_send_data =
      [&vm_x_out, &vm_x_in, args, base_data_idx, x_ofst_chunk, x_stride, vm_buffer_size_base_1,
       mem_data, uc_data_array](T* reg_data, int64_t num_elems, int64_t ofst_buffer) -> void {
    if (x_ofst_chunk + base_data_idx < num_elems) {
      vm_reduce_data<local_tp_size>(reg_data, vm_x_in, mem_data + ofst_buffer, args.elems_per_blk);
      replace_neg_zero(reg_data);
      unicast_store<args.local_size - 1>(uc_data_array, reg_data,
                                         ofst_buffer + vm_buffer_size_base_1, args.elems_per_blk);
      copy_data(vm_x_out, reg_data);
    }
    vm_x_out += x_stride;
    vm_x_in += x_stride;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, x_ofst_chunk, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    T const* src = mem_data_2 + ofst_buffer;
    int peer_local_dp_rank = args.local_dp_rank;
#pragma unroll
    for (int i = 0; i < local_dp_size - 1; ++i) {
      CIRCULAR_MINUS(peer_local_dp_rank, local_dp_size, 1);
      T* dst = x_out + peer_local_dp_rank * args.num_elems_all;
      int ofst_copy = x_ofst_chunk;
#pragma unroll
      for (int j = 0; j < local_tp_size; ++j) {
        CIRCULAR_MINUS(ofst_copy, x_stride, args.elems_per_blk);
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src);
          copy_data(dst + ofst_copy, reg_data);
        }
        src += args.elems_per_blk;
      }
    }
    T* dst = x_out + args.local_dp_rank * args.num_elems_all;
    int ofst_copy = x_ofst_chunk;
#pragma unroll
    for (int i = 0; i < local_tp_size - 1; ++i) {
      CIRCULAR_MINUS(ofst_copy, x_stride, args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src);
        copy_data(dst + ofst_copy, reg_data);
      }
      src += args.elems_per_blk;
    }
    x_out += x_stride;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce data and send data in the same node
  func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Reduce data and send data in the same node
    func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_bytes1_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_tp_size * (1 + dp_size) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = x_stride;
  int vm_buffer_size_base_2_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base_2 = args.inter_dp_size * vm_buffer_size_base_2_local;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size;
  if constexpr (args.use_multicast) {
    vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  } else {
    vm_buffer_size = vm_buffer_size_base;
  }
  int ns_data_size = args.inter_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T *mc_data, *mc_data_2, *uc_data_array[args.local_size];
  uint32_t *mem_signal, *mc_signal;
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_tp_all + (vm_ofst_data + x_ofst_chunk);
    mc_data_2 = args.mc_buffer_full_all +
                (vm_ofst_data + vm_buffer_size_base_1 + args.local_rank * args.elems_per_blk);
    int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
    mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
    mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_tp_all + vm_ofst_signal);
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  }
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;

  auto func_vm_read_inputs_reduce_data_ns_send_data =
      [&x_in, args, base_data_idx, tid, tid_y, x_ofst_chunk, x_stride, mem_data, mc_data,
       uc_data_array, mem_signal, mc_signal, ns_data,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    if constexpr (args.use_multicast) {
      int ofst_copy = tid_y * args.elems_per_blk;
      T* dst = mem_data + vm_ofst_buffer;
      if constexpr (block_size_y > local_tp_size) {
        if (tid_y < local_tp_size) {
          if (ofst_copy + base_data_idx < num_elems) {
            copy_data(dst + ofst_copy, x_in + ofst_copy);
          }
        }
      } else {
        int stride_copy = block_size_y * args.elems_per_blk;
#pragma unroll
        for (int i = 0; i < local_tp_size; i += block_size_y) {
          if (ofst_copy + base_data_idx < num_elems) {
            copy_data(dst + ofst_copy, x_in + ofst_copy);
          }
          ofst_copy += stride_copy;
        }
      }
      __syncthreads();
      if (tid == 0) {
        multicast_add_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
        multicast_wait_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
            NEG_ZERO_UINT32 + static_cast<uint32_t>(local_tp_size));
      }
      __syncthreads();
    } else {
      int64_t ofst_copy_out = vm_ofst_buffer + args.local_tp_rank * args.elems_per_blk;
      int ofst_copy_in = tid_y * args.elems_per_blk;
      T* const* uc_data_array_val = uc_data_array + (local_dp_size - 1) * local_tp_size;
      if constexpr (block_size_y > local_tp_size) {
        if (tid_y < local_tp_size) {
          if (ofst_copy_in + base_data_idx < num_elems) {
            int idx = tid_y - args.local_tp_rank - 1;
            CIRCULAR_MIN_CLIP(idx, local_tp_size);
            copy_data(reg_data, x_in + ofst_copy_in);
            replace_neg_zero(reg_data);
            copy_data(uc_data_array_val[idx] + ofst_copy_out, reg_data);
          }
        }
      } else {
        int stride_copy = block_size_y * args.elems_per_blk;
#pragma unroll
        for (int i = 0; i < local_tp_size; i += block_size_y) {
          if (ofst_copy_in + base_data_idx < num_elems) {
            int idx = tid_y + i - args.local_tp_rank - 1;
            CIRCULAR_MIN_CLIP(idx, local_tp_size);
            copy_data(reg_data, x_in + ofst_copy_in);
            replace_neg_zero(reg_data);
            copy_data(uc_data_array_val[idx] + ofst_copy_out, reg_data);
          }
          ofst_copy_in += stride_copy;
        }
      }
    }
    x_in += x_stride;
    num_elems -= x_ofst_chunk;
    T* dst = ns_data + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        if constexpr (args.use_multicast) {
          multicast_load_reduce(reg_data, mc_data + vm_ofst_buffer);
        } else {
          vm_reduce_data<local_tp_size>(reg_data, mem_data + vm_ofst_buffer, args.elems_per_blk);
        }
        copy_data(dst, reg_data);
      }
    });
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
      T* ns_dst_src = dst - base_data_idx;
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
  };
  auto func_ns_reduce_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk, vm_buffer_size_base_1,
       vm_buffer_size_base_2_local, ns_data, ns_signal, mc_data_2,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        int stride_reduce = args.inter_tp_size * args.elems_per_blk;
        if constexpr (args.use_multicast) {
          T* dst = mc_data_2 + vm_ofst_buffer;
          T const* src = ns_data + ns_ofst_data;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            ns_reduce_data<use_inter_tp>(reg_data, src, args.elems_per_blk, args.inter_tp_size);
            replace_neg_zero(reg_data);
            multicast_store(dst, reg_data);
            dst += vm_buffer_size_base_2_local;
            src += stride_reduce;
          }
        } else {
          int64_t ofst_copy =
              vm_ofst_buffer + vm_buffer_size_base_1 + args.local_rank * args.elems_per_blk;
          T const* src = ns_data + ns_ofst_data;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            ns_reduce_data<use_inter_tp>(reg_data, src, args.elems_per_blk, args.inter_tp_size);
            replace_neg_zero(reg_data);
            unicast_store<args.local_size>(uc_data_array, reg_data, ofst_copy);
            ofst_copy += vm_buffer_size_base_2_local;
            src += stride_reduce;
          }
        }
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    if constexpr (block_size_y > local_tp_size) {
      int tid_y_dp = floor_div<local_tp_size>(tid_y);
      int tid_y_tp = mod<local_tp_size>(tid_y);
      int ofst_copy = tid_y_tp * args.elems_per_blk;
      T* dst = x_out + (tid_y_dp * args.num_elems_all + ofst_copy);
      T const* src = mem_data_2 + (ofst_buffer + tid_y * args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
        int64_t stride_copy_out = stride_dp_size * args.num_elems_all;
        int stride_copy_in = stride_dp_size * x_stride;
        for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_dp_size; j += stride_dp_size) {
            vm_receive_data(reg_data, src);
            copy_data(dst, reg_data);
            dst += stride_copy_out;
            src += stride_copy_in;
          }
        }
      }
    } else {
      int ofst_copy_base = tid_y * args.elems_per_blk;
      int stride_copy = block_size_y * args.elems_per_blk;
      T* dst = x_out;
      T const* src = mem_data_2 + ofst_buffer;
      for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          int ofst_copy = ofst_copy_base;
#pragma unroll
          for (int k = 0; k < local_tp_size; k += block_size_y) {
            if (ofst_copy + base_data_idx < num_elems) {
              vm_receive_data(reg_data, src + ofst_copy);
              copy_data(dst + ofst_copy, reg_data);
              ofst_copy += stride_copy;
            }
          }
          dst += args.num_elems_all;
          src += x_stride;
        }
      }
    }
    x_out += x_stride;
  };

  // Read inputs, reduce data in the same node and send data across different nodes
  func_vm_read_inputs_reduce_data_ns_send_data(
      reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
      buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs, reduce data in the same node and send data across different nodes
    func_vm_read_inputs_reduce_data_ns_send_data(
        reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
        buffer_info.ns_ofst_signal());
    // Reduce data across different nodes and send data in the same node (previous)
    func_ns_reduce_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and send data in the same node
  func_ns_reduce_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                   buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                   buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_allreduce_allgather_opt_bytes2(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   tp_size * (1 + dp_size) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   (inter_tp_size + inter_size) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes2_mode(mode) && args.use_inter, "Invalid mode");
  int x_ofst_chunk_local = args.local_tp_rank * args.elems_per_blk;
  int x_ofst_chunk = args.tp_rank * args.elems_per_blk;
  int x_stride_local = local_tp_size * args.elems_per_blk;
  int x_stride = args.tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = x_stride;
  int vm_buffer_size_base_2_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base_2 = args.inter_size * vm_buffer_size_base_2_local;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size;
  if constexpr (args.use_multicast) {
    vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  } else {
    vm_buffer_size = vm_buffer_size_base;
  }
  int ns_data_size_base_1 = args.inter_tp_size * args.elems_per_blk;
  int ns_data_size_base_2 = args.inter_size * args.elems_per_blk;
  int ns_data_size = ns_data_size_base_1 + ns_data_size_base_2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T *mc_data, *mc_data_2, *uc_data_array[args.local_size];
  uint32_t *mem_signal, *mc_signal;
  if constexpr (args.use_multicast) {
    if constexpr (local_tp_size > 1) {
      mc_data = args.mc_buffer_tp_all + (vm_ofst_data + x_ofst_chunk_local);
    }
    mc_data_2 = args.mc_buffer_full_all +
                (vm_ofst_data + vm_buffer_size_base_1 + args.local_rank * args.elems_per_blk);
    int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
    if constexpr (local_tp_size > 1) {
      mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
      mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_tp_all + vm_ofst_signal);
    }
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  }
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;
  T* ns_data_2 = ns_data + ns_data_size_base_1;
  uint64_t* ns_signal_2 = ns_signal + 1;

  auto func_vm_read_inputs_reduce_data_ns_send_data =
      [&x_in, args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local, x_stride,
       ns_data_size_base_1, mem_data, mc_data, uc_data_array, mem_signal, mc_signal, ns_data,
       ns_data_2, ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer,
                             int64_t ns_ofst_data, int ns_ofst_signal) -> void {
    if constexpr (args.use_multicast) {
      int ofst_copy = tid_y * args.elems_per_blk;
      T* dst = mem_data + vm_ofst_buffer;
      if constexpr (block_size_y > local_tp_size) {
        int stride_copy = local_tp_size * args.elems_per_blk;
        if (tid_y < local_tp_size) {
          for (int i = 0; i < args.inter_tp_size; ++i) {
            if (ofst_copy + base_data_idx < num_elems) {
              copy_data(dst + ofst_copy, x_in + ofst_copy);
            }
            ofst_copy += stride_copy;
          }
        }
      } else {
        int stride_copy = block_size_y * args.elems_per_blk;
        for (int i = 0; i < args.inter_tp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_tp_size; j += block_size_y) {
            if (ofst_copy + base_data_idx < num_elems) {
              copy_data(dst + ofst_copy, x_in + ofst_copy);
            }
            ofst_copy += stride_copy;
          }
        }
      }
      if constexpr (local_tp_size > 1) {
        __syncthreads();
        if (tid == 0) {
          multicast_add_signal(
              reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
          multicast_wait_signal(
              reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
              NEG_ZERO_UINT32 + static_cast<uint32_t>(local_tp_size));
        }
        __syncthreads();
      }
    } else {
      int64_t ofst_copy_out = vm_ofst_buffer + args.local_tp_rank * args.elems_per_blk;
      int ofst_copy_in = tid_y * args.elems_per_blk;
      T* const* uc_data_array_val = uc_data_array + (local_dp_size - 1) * local_tp_size;
      if constexpr (block_size_y > local_tp_size) {
        int stride_copy = local_tp_size * args.elems_per_blk;
        if (tid_y < local_tp_size) {
          for (int i = 0; i < args.inter_tp_size; ++i) {
            if (ofst_copy_in + base_data_idx < num_elems) {
              int idx = tid_y - args.local_tp_rank - 1;
              CIRCULAR_MIN_CLIP(idx, local_tp_size);
              copy_data(reg_data, x_in + ofst_copy_in);
              replace_neg_zero(reg_data);
              copy_data(uc_data_array_val[idx] + ofst_copy_out, reg_data);
            }
            ofst_copy_in += stride_copy;
            ofst_copy_out += x_stride_local;
          }
        }
      } else {
        int stride_copy = block_size_y * args.elems_per_blk;
        for (int i = 0; i < args.inter_tp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_tp_size; j += block_size_y) {
            if (ofst_copy_in + base_data_idx < num_elems) {
              int idx = tid_y + j - args.local_tp_rank - 1;
              CIRCULAR_MIN_CLIP(idx, local_tp_size);
              copy_data(reg_data, x_in + ofst_copy_in);
              replace_neg_zero(reg_data);
              copy_data(uc_data_array_val[idx] + ofst_copy_out, reg_data);
            }
            ofst_copy_in += stride_copy;
          }
          ofst_copy_out += x_stride_local;
        }
      }
    }
    x_in += x_stride;
    num_elems -= x_ofst_chunk_local;
    T* dst_self = ns_data + (ns_ofst_data + args.inter_tp_rank * args.elems_per_blk);
    T* dst_base = ns_data_2 + (ns_ofst_data + args.inter_dp_rank * ns_data_size_base_1);
    run_first_block_y<block_size_y>(tid_y, [&] {
      int ofst_copy = 0;
      int stride_reduce = local_tp_size * args.elems_per_blk;
      for (int i = 0; i < args.inter_tp_size; ++i) {
        T* dst = (i == args.inter_tp_rank) ? dst_self : (dst_base + i * args.elems_per_blk);
        if (ofst_copy + base_data_idx < num_elems) {
          if constexpr (args.use_multicast) {
            if constexpr (local_tp_size > 1) {
              multicast_load_reduce(reg_data, mc_data + (vm_ofst_buffer + ofst_copy));
            } else {
              copy_data(reg_data, mem_data + (vm_ofst_buffer + ofst_copy));
            }
          } else {
            vm_reduce_data<local_tp_size>(reg_data, mem_data + (vm_ofst_buffer + ofst_copy),
                                          args.elems_per_blk);
          }
          copy_data(dst, reg_data);
        }
        ofst_copy += stride_reduce;
      }
    });
    __syncthreads();
    if (tid < args.inter_tp_size && tid != args.inter_tp_rank) {
      T* ns_dst = dst_self - base_data_idx;
      T const* ns_src = dst_base - base_data_idx + tid * args.elems_per_blk;
      int ns_msg_elems = floor_div<args.elems_per_thd>(
          min(num_elems - tid * x_stride_local, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank =
          args.local_rank + (tid + args.inter_dp_rank * args.inter_tp_size) * args.local_size;
      if (ns_msg_elems > 0) {
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
  };
  auto func_ns_reduce_send_data =
      [args, base_data_idx, tid, tid_y, x_ofst_chunk, ns_data, ns_data_2, ns_signal, ns_signal_2](
          T* reg_data, int64_t num_elems, int64_t ofst_data, int ofst_signal) -> void {
    num_elems -= x_ofst_chunk;
    if (num_elems > 0) {
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal + ofst_signal, NVSHMEM_CMP_EQ, args.inter_tp_size - 1);
      }
      __syncthreads();
      T* dst = ns_data_2 + (ofst_data + args.inter_rank * args.elems_per_blk);
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          ns_reduce_data<use_inter_tp>(reg_data, ns_data + ofst_data, args.elems_per_blk,
                                       args.inter_tp_size);
          replace_neg_zero(reg_data);
          copy_data(dst, reg_data);
        }
      });
      __syncthreads();
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      if (tid < args.inter_size && tid != args.inter_rank) {
        T* ns_dst_src = dst - base_data_idx;
        int peer_world_rank = args.local_rank + tid * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal_2 + ofst_signal,
                                  1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local,
       vm_buffer_size_base_1, vm_buffer_size_base_2_local, ns_data_2, ns_signal, ns_signal_2,
       mc_data_2, uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data,
                                 int ns_ofst_signal, int ns_ofst_reset,
                                 int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk_local;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(2);
      int ns_num_senders = args.inter_dp_size * min(ceil_div(num_elems, x_stride_local),
                                                    static_cast<int64_t>(args.inter_tp_size)) -
                           (num_elems > args.inter_tp_rank * x_stride_local);
      if (ns_num_senders > 0) {
        nvshmem_signal_wait_until(ns_signal_2 + ns_ofst_signal, NVSHMEM_CMP_EQ, ns_num_senders);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      T const* src = ns_data_2 + ns_ofst_data;
      if constexpr (args.use_multicast) {
        T* dst = mc_data_2 + vm_ofst_buffer;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            if (j * x_stride_local + base_data_idx < num_elems) {
              copy_data(reg_data, src);
              multicast_store(dst, reg_data);
            }
            dst += vm_buffer_size_base_2_local;
            src += args.elems_per_blk;
          }
        }
      } else {
        int64_t ofst_copy =
            vm_ofst_buffer + vm_buffer_size_base_1 + args.local_rank * args.elems_per_blk;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            if (j * x_stride_local + base_data_idx < num_elems) {
              copy_data(reg_data, src);
              unicast_store<args.local_size>(uc_data_array, reg_data, ofst_copy);
            }
            ofst_copy += vm_buffer_size_base_2_local;
            src += args.elems_per_blk;
          }
        }
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int stride_copy_in = block_size_y * args.elems_per_blk;
    T const* src = mem_data_2 + (ofst_buffer + tid_y * args.elems_per_blk);
    if constexpr (block_size_y > local_tp_size) {
      constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
      int tid_y_dp = floor_div<local_tp_size>(tid_y);
      int tid_y_tp = mod<local_tp_size>(tid_y);
      for (int i = 0; i < args.inter_dp_size; ++i) {
        for (int j = 0; j < args.inter_tp_size; ++j) {
          int ofst_copy_out = (tid_y_tp + j * local_tp_size) * args.elems_per_blk;
#pragma unroll
          for (int k = 0; k < local_dp_size; k += stride_dp_size) {
            T* dst = x_out + (tid_y_dp + k + i * local_dp_size) * args.num_elems_all;
            if (ofst_copy_out + base_data_idx < num_elems) {
              vm_receive_data(reg_data, src);
              copy_data(dst + ofst_copy_out, reg_data);
            }
            src += stride_copy_in;
          }
        }
      }
    } else {
      for (int i = 0; i < args.inter_dp_size; ++i) {
        for (int j = 0; j < args.inter_tp_size; ++j) {
#pragma unroll
          for (int k = 0; k < local_dp_size; ++k) {
            T* dst = x_out + (k + i * local_dp_size) * args.num_elems_all;
#pragma unroll
            for (int l = 0; l < local_tp_size; l += block_size_y) {
              int ofst_copy_out = (tid_y + l + j * local_tp_size) * args.elems_per_blk;
              if (ofst_copy_out + base_data_idx < num_elems) {
                vm_receive_data(reg_data, src);
                copy_data(dst + ofst_copy_out, reg_data);
              }
              src += stride_copy_in;
            }
          }
        }
      }
    }
    x_out += x_stride;
  };

  // Read inputs, reduce data in the same node, and send data across different nodes
  func_vm_read_inputs_reduce_data_ns_send_data(
      reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
      buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce and send data across different nodes
  func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                           buffer_info.ns_ofst_signal());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs, reduce data in the same node, and send data across different nodes
    func_vm_read_inputs_reduce_data_ns_send_data(
        reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
        buffer_info.ns_ofst_signal());
    // Receive data across different nodes and send data in the same node (previous)
    func_ns_receive_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data across different nodes
    func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                             buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void fused_allreduce_allgather_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args) {
  constexpr MixedCommOp op = MixedCommOp::ALLREDUCE_ALLGATHER;
  static_assert(is_valid_op<args.use_local_tp, args.use_local_dp, use_inter_tp, use_inter_dp>(op),
                "Invalid op");
  static_assert(is_valid_mode<args.use_local_tp, use_inter_tp>(mode), "Invalid mode");
  static_assert(is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode),
                "Invalid block_size_y");
  if constexpr (is_opt_waits_mode(mode)) {
    if constexpr (!args.use_inter) {
      fused_allreduce_allgather_opt_waits_single<block_size_y>(args);
    } else {
      fused_allreduce_allgather_opt_waits_multi<block_size_y>(args);
    }
  } else if constexpr (is_opt_bytes1_mode(mode)) {
    if constexpr (!args.use_inter) {
      if constexpr (args.use_multicast) {
        fused_allreduce_allgather_opt_bytes1_multicast_single<block_size_y>(args);
      } else {
        fused_allreduce_allgather_opt_bytes1_unicast_single<block_size_y>(args);
      }
    } else {
      fused_allreduce_allgather_opt_bytes1_multi<block_size_y>(args);
    }
  } else {
    static_assert(is_opt_bytes2_mode(mode) && args.use_inter, "Invalid mode");
    fused_allreduce_allgather_opt_bytes2<block_size_y>(args);
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_waits_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && !args.use_inter, "Invalid mode");
  int vm_buffer_size = args.local_size * args.elems_per_blk;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all,
                                     vm_ofst_data + args.local_tp_rank * args.elems_per_blk);

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, tid_y, uc_data_array](
                                           T* reg_data, int64_t num_elems,
                                           int64_t ofst_buffer) -> void {
    if (base_data_idx < num_elems) {
      int peer_local_dp_rank_base = args.local_dp_rank + tid_y + 1;
      int stride_copy_base = local_tp_size * args.elems_per_blk;
      int64_t ofst_copy = ofst_buffer + tid_y * stride_copy_base;
      int stride_copy = block_size_y * stride_copy_base;
      T* const* uc_data_array_val = uc_data_array + tid_y * local_tp_size;
#pragma unroll
      for (int i = 0; i < local_dp_size; i += block_size_y) {
        int peer_local_dp_rank = peer_local_dp_rank_base + i;
        CIRCULAR_MAX_CLIP(peer_local_dp_rank, local_dp_size);
        copy_data(reg_data, x_in + peer_local_dp_rank * args.num_elems_all);
        replace_neg_zero(reg_data);
        unicast_store<local_tp_size>(uc_data_array_val, reg_data, ofst_copy);
        ofst_copy += stride_copy;
        uc_data_array_val += block_size_y * local_tp_size;
      }
    }
    x_in += args.elems_per_blk;
  };
  auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, tid_y, mem_data](
                                               T* reg_data, int64_t num_elems,
                                               int64_t ofst_buffer) -> void {
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        vm_reduce_data<args.local_size>(reg_data, mem_data + ofst_buffer, args.elems_per_blk);
        copy_data(x_out, reg_data);
      }
    });
    x_out += args.elems_per_blk;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
    UPDATE_BASIC_VARIABLES(args.elems_per_blk);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reduce data in the same node and write outputs (previous)
    func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                      buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data in the same node and write outputs
  func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_waits_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   inter_dp_size * local_size * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   (inter_dp_size + inter_size) * elems_per_blk * sizeof(T)
  static_assert(is_opt_waits_mode(mode) && args.use_inter, "Invalid mode");
  if constexpr (local_dp_size == 1) {
    static_assert(block_size_y == 1, "Invalid block_size_y");
    int vm_buffer_size = args.local_size * args.elems_per_blk;
    int ns_data_size_base_1 = args.inter_dp_size * args.elems_per_blk;
    int ns_data_size_base_2 = args.inter_size * args.elems_per_blk;
    int ns_data_size = ns_data_size_base_1 + ns_data_size_base_2;
    INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

    int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
    T *mc_data, *uc_data_array[args.local_size];
    if constexpr (args.use_multicast) {
      mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
    } else {
      set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all,
                                         vm_ofst_data_self);
    }
    T* ns_data_2 = ns_data + ns_data_size_base_1;

    auto func_ns_read_inputs_send_data = [&x_in, args, base_data_idx, tid, ns_data, ns_data_2,
                                          ns_signal](int64_t num_elems, int64_t ofst_data,
                                                     int ofst_signal) -> void {
      T* dst_self = ns_data_2 + (ofst_data + args.inter_rank * args.elems_per_blk);
      if (base_data_idx < num_elems) {
        T* dst_base = ns_data + ofst_data;
        T const* src = x_in;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          T* dst = (i == args.inter_dp_rank) ? dst_self : dst_base;
          copy_data(dst, src);
          dst_base += args.elems_per_blk;
          src += args.num_elems_all;
        }
      }
      __syncthreads();
      if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
        int peer_inter_dp_rank = tid / args.inter_tp_size;
        int peer_inter_tp_rank = tid - peer_inter_dp_rank * args.inter_tp_size;
        T* ns_dst = dst_self - base_data_idx;
        T const* ns_src =
            (peer_inter_dp_rank == args.inter_dp_rank)
                ? ns_dst
                : (ns_data - base_data_idx + ofst_data + peer_inter_dp_rank * args.elems_per_blk);
        int ns_msg_elems =
            floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
        int peer_world_rank = args.local_rank + tid * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      x_in += args.elems_per_blk;
    };
    auto func_ns_reduce_data_vm_send_data =
        [&buffer_info, args, base_data_idx, tid, ns_data_2, ns_signal, mc_data, uc_data_array](
            T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
            int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
      if (tid == 0) {
        buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
        buffer_info.update_ns_reset_size(1);
        if (num_elems > 0) {
          nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ,
                                    args.inter_size - 1);
        }
      }
      __syncthreads();
      if (base_data_idx < num_elems) {
        ns_reduce_data<args.use_inter>(reg_data, ns_data_2 + ns_ofst_data, args.elems_per_blk,
                                       args.inter_size);
        replace_neg_zero(reg_data);
        if constexpr (args.use_multicast) {
          multicast_store(mc_data + vm_ofst_buffer, reg_data);
        } else {
          unicast_store<args.local_size>(uc_data_array, reg_data, vm_ofst_buffer);
        }
      }
    };
    auto func_vm_reduce_data_write_outputs = [&x_out, args, base_data_idx, mem_data](
                                                 T* reg_data, int64_t num_elems,
                                                 int64_t ofst_buffer) -> void {
      if (base_data_idx < num_elems) {
        vm_reduce_data<args.local_size>(reg_data, mem_data + ofst_buffer, args.elems_per_blk);
        copy_data(x_out, reg_data);
      }
      x_out += args.elems_per_blk;
    };

    // Read inputs and send data across different nodes
    func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                  buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Reduce data across different nodes and send data in the same node
    func_ns_reduce_data_vm_send_data(
        reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
        buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
    // Main loop
    for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
      UPDATE_BASIC_VARIABLES(args.elems_per_blk);
      // Read inputs and send data across different nodes
      func_ns_read_inputs_send_data(num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal());
      // Reduce data in the same node and write outputs (previous)
      func_vm_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                        buffer_info.vm_ofst_buffer_prev());
      // Reduce data across different nodes and send data in the same node
      func_ns_reduce_data_vm_send_data(
          reg_data, num_elems_remain, buffer_info.ns_ofst_data(), buffer_info.ns_ofst_signal(),
          buffer_info.ns_ofst_signal_prev(), buffer_info.vm_ofst_buffer());
      // Reset previous buffer
      func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    }
    // Reduce data in the same node and write outputs
    func_vm_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    buffer_info.write_gmem(tid);
  } else {
    constexpr bool use_multicast = args.use_multicast && local_tp_size == 1;
    int vm_buffer_size_base_local = args.local_size * args.elems_per_blk;
    int vm_buffer_size_base = args.inter_dp_size * vm_buffer_size_base_local;
    int vm_buffer_size;
    if constexpr (use_multicast) {
      vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
    } else {
      vm_buffer_size = vm_buffer_size_base;
    }
    int ns_data_size_base_1 = args.inter_dp_size * args.elems_per_blk;
    int ns_data_size_base_2 = args.inter_size * args.elems_per_blk;
    int ns_data_size = ns_data_size_base_1 + ns_data_size_base_2;
    INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

    T *mc_data, *uc_data_array[args.local_size];
    uint32_t *mem_signal, *mc_signal;
    if constexpr (use_multicast) {
      int64_t vm_ofst_data_self = vm_ofst_data + args.local_rank * args.elems_per_blk;
      mc_data = args.mc_buffer_full_all + vm_ofst_data_self;
      int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
      mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
      mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);
    } else {
      int64_t vm_ofst_data_self = vm_ofst_data + args.local_tp_rank * args.elems_per_blk;
      set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all,
                                         vm_ofst_data_self);
    }
    T* ns_data_2 = ns_data + ns_data_size_base_1;

    auto func_vm_read_inputs_reduce_data_ns_send_data =
        [&x_in, args, base_data_idx, tid, tid_y, vm_buffer_size_base_local, mem_data, mc_data,
         uc_data_array, mem_signal, mc_signal, ns_data, ns_data_2,
         ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                    int ns_ofst_signal) -> void {
      if constexpr (use_multicast) {
        if (base_data_idx < num_elems) {
          int stride_copy_out = block_size_y * args.elems_per_blk;
          int64_t stride_copy_in = block_size_y * args.num_elems_all;
          T* dst = mem_data + (vm_ofst_buffer + tid_y * args.elems_per_blk);
          T const* src = x_in + tid_y * args.num_elems_all;
          for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
            for (int j = 0; j < local_dp_size; j += block_size_y) {
              copy_data(dst, src);
              dst += stride_copy_out;
              src += stride_copy_in;
            }
          }
        }
        __syncthreads();
        if (tid == 0) {
          multicast_add_signal(
              reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
          multicast_wait_signal(
              reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
              NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
        }
        __syncthreads();
      } else {
        if (base_data_idx < num_elems) {
          int peer_local_dp_rank_base = args.local_dp_rank + tid_y + 1;
          int stride_copy_base = local_tp_size * args.elems_per_blk;
          int64_t ofst_copy_out = vm_ofst_buffer + tid_y * stride_copy_base;
          int stride_copy_out = block_size_y * stride_copy_base;
          int64_t ofst_copy_in = local_dp_size * args.num_elems_all;
          T const* src = x_in;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            T* const* uc_data_array_val = uc_data_array + tid_y * local_tp_size;
#pragma unroll
            for (int j = 0; j < local_dp_size; j += block_size_y) {
              int peer_local_dp_rank = peer_local_dp_rank_base + j;
              CIRCULAR_MAX_CLIP(peer_local_dp_rank, local_dp_size);
              copy_data(reg_data, src + peer_local_dp_rank * args.num_elems_all);
              replace_neg_zero(reg_data);
              unicast_store<local_tp_size>(uc_data_array_val, reg_data, ofst_copy_out);
              ofst_copy_out += stride_copy_out;
              uc_data_array_val += block_size_y * local_tp_size;
            }
            src += ofst_copy_in;
          }
        }
      }
      T* dst_self = ns_data_2 + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
      T* dst_base = ns_data + ns_ofst_data;
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          if constexpr (use_multicast) {
            T const* src = mc_data + vm_ofst_buffer;
            for (int i = 0; i < args.inter_dp_size; ++i) {
              T* dst = (i == args.inter_dp_rank) ? dst_self : dst_base;
              multicast_load_reduce(reg_data, src);
              copy_data(dst, reg_data);
              dst_base += args.elems_per_blk;
              src += vm_buffer_size_base_local;
            }
          } else {
            T const* src = mem_data + vm_ofst_buffer;
            for (int i = 0; i < args.inter_dp_size; ++i) {
              T* dst = (i == args.inter_dp_rank) ? dst_self : dst_base;
              vm_reduce_data<args.local_size>(reg_data, src, args.elems_per_blk);
              copy_data(dst, reg_data);
              dst_base += args.elems_per_blk;
              src += vm_buffer_size_base_local;
            }
          }
        }
      });
      __syncthreads();
      if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
        int peer_inter_dp_rank = tid / args.inter_tp_size;
        int peer_inter_tp_rank = tid - peer_inter_dp_rank * args.inter_tp_size;
        T* ns_dst = dst_self - base_data_idx;
        T const* ns_src = (peer_inter_dp_rank == args.inter_dp_rank)
                              ? ns_dst
                              : (ns_data - base_data_idx + ns_ofst_data +
                                 peer_inter_dp_rank * args.elems_per_blk);
        int ns_msg_elems =
            floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
        int peer_world_rank = args.local_rank + tid * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      x_in += args.elems_per_blk;
    };
    auto func_ns_reduce_data_write_outputs = [&buffer_info, &x_out, args, base_data_idx, tid, tid_y,
                                              ns_data_2, ns_signal](
                                                 T* reg_data, int64_t num_elems, int64_t ofst_data,
                                                 int ofst_signal, int ofst_reset) -> void {
      if (tid == 0) {
        buffer_info.reset_ns_signal(ns_signal + ofst_reset);
        buffer_info.update_ns_reset_size(1);
        if (num_elems > 0) {
          nvshmem_signal_wait_until(ns_signal + ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
        }
      }
      __syncthreads();
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          ns_reduce_data<args.use_inter>(reg_data, ns_data_2 + ofst_data, args.elems_per_blk,
                                         args.inter_size);
          copy_data(x_out, reg_data);
        }
        x_out += args.elems_per_blk;
      });
    };

    // Read inputs, reduce data in the same node, and send data across different nodes
    func_vm_read_inputs_reduce_data_ns_send_data(
        reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
        buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Main loop
    for (int64_t t = args.max_num_elems; t > args.elems_per_blk; t -= args.elems_per_blk) {
      int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
      UPDATE_BASIC_VARIABLES(args.elems_per_blk);
      // Read inputs, reduce data in the same node, and send data across different nodes
      func_vm_read_inputs_reduce_data_ns_send_data(
          reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
          buffer_info.ns_ofst_signal());
      // Reduce data across different nodes and write outputs (previous)
      func_ns_reduce_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                        buffer_info.ns_ofst_data_prev(),
                                        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset);
      // Reset previous buffer
      func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    }
    // Reduce data across different nodes and write outputs
    func_ns_reduce_data_write_outputs(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                      buffer_info.ns_ofst_signal(),
                                      buffer_info.ns_ofst_signal_prev());
    buffer_info.write_gmem(tid);
  }
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_bytes1_multicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_size + local_tp_size) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_multicast && !args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base_2 = x_stride;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);

  T* mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);
  T* mc_data_2 = args.mc_buffer_tp_all + (vm_ofst_data + vm_buffer_size_base_1 + x_ofst_chunk);
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;
  int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
  uint32_t* mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
  uint32_t* mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);

  auto func_vm_read_inputs = [&x_in, args, base_data_idx, tid_y, x_stride, mem_data](
                                 int64_t num_elems, int64_t ofst_buffer) -> void {
    if constexpr (block_size_y > local_tp_size) {
      int tid_y_dp = floor_div<local_tp_size>(tid_y);
      int tid_y_tp = mod<local_tp_size>(tid_y);
      int ofst_copy = tid_y_tp * args.elems_per_blk;
      T* dst = mem_data + (ofst_buffer + tid_y * args.elems_per_blk);
      T const* src = x_in + (tid_y_dp * args.num_elems_all + ofst_copy);
      if (ofst_copy + base_data_idx < num_elems) {
        constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
        int stride_copy_out = stride_dp_size * x_stride;
        int64_t stride_copy_in = stride_dp_size * args.num_elems_all;
#pragma unroll
        for (int i = 0; i < local_dp_size; i += stride_dp_size) {
          copy_data(dst, src);
          dst += stride_copy_out;
          src += stride_copy_in;
        }
      }
    } else {
      int ofst_copy_base = tid_y * args.elems_per_blk;
      int stride_copy = block_size_y * args.elems_per_blk;
      T* dst = mem_data + ofst_buffer;
      T const* src = x_in;
#pragma unroll
      for (int i = 0; i < local_dp_size; ++i) {
        int ofst_copy = ofst_copy_base;
#pragma unroll
        for (int j = 0; j < local_tp_size; j += block_size_y) {
          if (ofst_copy + base_data_idx < num_elems) {
            copy_data(dst + ofst_copy, src + ofst_copy);
            ofst_copy += stride_copy;
          }
        }
        dst += x_stride;
        src += args.num_elems_all;
      }
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_send_data = [args, base_data_idx, tid, tid_y, x_ofst_chunk, mc_data,
                                   mc_data_2, mem_signal, mc_signal](T* reg_data, int64_t num_elems,
                                                                     int64_t ofst_buffer) -> void {
    __syncthreads();
    if (tid == 0) {
      multicast_add_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + ofst_buffer));
      multicast_wait_signal(
          reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + ofst_buffer),
          NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (x_ofst_chunk + base_data_idx < num_elems) {
        multicast_load_reduce(reg_data, mc_data + ofst_buffer);
        replace_neg_zero(reg_data);
        multicast_store(mc_data_2 + ofst_buffer, reg_data);
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    T const* src = mem_data_2 + ofst_buffer;
    if constexpr (block_size_y > local_tp_size) {
      if (tid_y < local_tp_size) {
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src + ofst_copy);
          copy_data(x_out + ofst_copy, reg_data);
        }
      }
    } else {
      int stride_copy = block_size_y * args.elems_per_blk;
#pragma unroll
      for (int i = 0; i < local_tp_size; i += block_size_y) {
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src + ofst_copy);
          copy_data(x_out + ofst_copy, reg_data);
        }
        ofst_copy += stride_copy;
      }
    }
    x_out += x_stride;
  };

  // Read inputs
  func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reduce and send data in the same node
  func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs
    func_vm_read_inputs(num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data in the same node
    func_vm_reduce_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_bytes1_unicast_single(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   (local_size + local_tp_size - 2) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && !args.use_multicast && !args.use_inter, "Invalid mode");
  static_assert(block_size_y == 1, "Invalid block_size_y");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = (args.local_size - 1) * args.elems_per_blk;
  int vm_buffer_size_base_2 = x_stride - args.elems_per_blk;
  int vm_buffer_size = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  INIT_BASIC_VARIABLES(vm_buffer_size, 0);
  T* vm_x_out = x_out + x_ofst_chunk;
  T const* vm_x_in = x_in + x_ofst_chunk + args.local_dp_rank * args.num_elems_all;

  T* uc_data_array[args.local_size];
  set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;

  auto func_vm_read_inputs_send_data = [&x_in, args, base_data_idx, x_ofst_chunk, x_stride,
                                        uc_data_array](T* reg_data, int64_t num_elems,
                                                       int64_t ofst_buffer) -> void {
    int64_t ofst_copy_out = ofst_buffer;
    int peer_local_dp_rank = args.local_dp_rank;
    T* const* uc_data_array_val = uc_data_array;
#pragma unroll
    for (int i = 0; i < local_dp_size - 1; ++i) {
      CIRCULAR_ADD(peer_local_dp_rank, local_dp_size, 1);
      T const* src = x_in + peer_local_dp_rank * args.num_elems_all;
      int ofst_copy_in = x_ofst_chunk;
#pragma unroll
      for (int j = 0; j < local_tp_size; ++j) {
        CIRCULAR_ADD(ofst_copy_in, x_stride, args.elems_per_blk);
        if (ofst_copy_in + base_data_idx < num_elems) {
          copy_data(reg_data, src + ofst_copy_in);
          replace_neg_zero(reg_data);
          copy_data(uc_data_array_val[j] + ofst_copy_out, reg_data);
        }
        ofst_copy_out += args.elems_per_blk;
      }
      uc_data_array_val += local_tp_size;
    }
    T const* src = x_in + args.local_dp_rank * args.num_elems_all;
    int ofst_copy_in = x_ofst_chunk;
#pragma unroll
    for (int i = 0; i < local_tp_size - 1; ++i) {
      CIRCULAR_ADD(ofst_copy_in, x_stride, args.elems_per_blk);
      if (ofst_copy_in + base_data_idx < num_elems) {
        copy_data(reg_data, src + ofst_copy_in);
        replace_neg_zero(reg_data);
        copy_data(uc_data_array_val[i] + ofst_copy_out, reg_data);
      }
      ofst_copy_out += args.elems_per_blk;
    }
    x_in += x_stride;
  };
  auto func_vm_reduce_data_send_data =
      [&vm_x_out, &vm_x_in, args, base_data_idx, x_ofst_chunk, x_stride, vm_buffer_size_base_1,
       mem_data, uc_data_array](T* reg_data, int64_t num_elems, int64_t ofst_buffer) -> void {
    if (x_ofst_chunk + base_data_idx < num_elems) {
      vm_reduce_data<args.local_size>(reg_data, vm_x_in, mem_data + ofst_buffer,
                                      args.elems_per_blk);
      replace_neg_zero(reg_data);
      unicast_store<local_tp_size - 1>(uc_data_array + (local_dp_size - 1) * local_tp_size,
                                       reg_data, ofst_buffer + vm_buffer_size_base_1,
                                       args.elems_per_blk);
      copy_data(vm_x_out, reg_data);
    }
    vm_x_out += x_stride;
    vm_x_in += x_stride;
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, x_ofst_chunk, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    T const* src = mem_data_2 + ofst_buffer;
    int ofst_copy = x_ofst_chunk;
#pragma unroll
    for (int i = 0; i < local_tp_size - 1; ++i) {
      CIRCULAR_MINUS(ofst_copy, x_stride, args.elems_per_blk);
      if (ofst_copy + base_data_idx < num_elems) {
        vm_receive_data(reg_data, src);
        copy_data(x_out + ofst_copy, reg_data);
      }
      src += args.elems_per_blk;
    }
    x_out += x_stride;
  };

  // Read inputs and send data in the same node
  func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce data and send data in the same node
  func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs and send data in the same node
    func_vm_read_inputs_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
    // Reduce data and send data in the same node
    func_vm_reduce_data_send_data(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  }
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_bytes1_multi(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   local_tp_size * (dp_size + 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   (inter_dp_size + inter_size) * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes1_mode(mode) && args.use_inter, "Invalid mode");
  int x_ofst_chunk = args.local_tp_rank * args.elems_per_blk;
  int x_stride = local_tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = args.inter_dp_size * vm_buffer_size_base_1_local;
  int vm_buffer_size_base_2 = x_stride;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size;
  if constexpr (args.use_multicast) {
    vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  } else {
    vm_buffer_size = vm_buffer_size_base;
  }
  int ns_data_size_base_1 = args.inter_dp_size * args.elems_per_blk;
  int ns_data_size_base_2 = args.inter_size * args.elems_per_blk;
  int ns_data_size = ns_data_size_base_1 + ns_data_size_base_2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T *mc_data, *mc_data_2, *uc_data_array[args.local_size];
  uint32_t *mem_signal, *mc_signal;
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);
    mc_data_2 = args.mc_buffer_tp_all + (vm_ofst_data + vm_buffer_size_base_1 + x_ofst_chunk);
    int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
    mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
    mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  }
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;
  T* ns_data_2 = ns_data + ns_data_size_base_1;

  auto func_vm_read_inputs_reduce_data_ns_send_data =
      [&x_in, args, base_data_idx, tid, tid_y, x_ofst_chunk, x_stride, vm_buffer_size_base_1_local,
       mem_data, mc_data, uc_data_array, mem_signal, mc_signal, ns_data, ns_data_2,
       ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer, int64_t ns_ofst_data,
                  int ns_ofst_signal) -> void {
    if constexpr (args.use_multicast) {
      if constexpr (block_size_y > local_tp_size) {
        int tid_y_dp = floor_div<local_tp_size>(tid_y);
        int tid_y_tp = mod<local_tp_size>(tid_y);
        int ofst_copy = tid_y_tp * args.elems_per_blk;
        T* dst = mem_data + (vm_ofst_buffer + tid_y * args.elems_per_blk);
        T const* src = x_in + (tid_y_dp * args.num_elems_all + ofst_copy);
        if (ofst_copy + base_data_idx < num_elems) {
          constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
          int stride_copy_out = stride_dp_size * x_stride;
          int64_t stride_copy_in = stride_dp_size * args.num_elems_all;
          for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
            for (int j = 0; j < local_dp_size; j += stride_dp_size) {
              copy_data(dst, src);
              dst += stride_copy_out;
              src += stride_copy_in;
            }
          }
        }
      } else {
        int ofst_copy_base = tid_y * args.elems_per_blk;
        int stride_copy = block_size_y * args.elems_per_blk;
        T* dst = mem_data + vm_ofst_buffer;
        T const* src = x_in;
        for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_dp_size; ++j) {
            int ofst_copy = ofst_copy_base;
#pragma unroll
            for (int k = 0; k < local_tp_size; k += block_size_y) {
              if (ofst_copy + base_data_idx < num_elems) {
                copy_data(dst + ofst_copy, src + ofst_copy);
                ofst_copy += stride_copy;
              }
            }
            dst += x_stride;
            src += args.num_elems_all;
          }
        }
      }
      __syncthreads();
      if (tid == 0) {
        multicast_add_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
        multicast_wait_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
            NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
      }
      __syncthreads();
    } else {
      if constexpr (block_size_y > local_tp_size) {
        int tid_y_dp = floor_div<local_tp_size>(tid_y);
        int tid_y_tp = mod<local_tp_size>(tid_y);
        int64_t ofst_copy_out = vm_ofst_buffer + args.local_rank * args.elems_per_blk;
        int ofst_copy_in = tid_y_tp * args.elems_per_blk;
        T const* src = x_in + (tid_y_dp * args.num_elems_all + ofst_copy_in);
        if (ofst_copy_in + base_data_idx < num_elems) {
          constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
          int64_t stride_copy_in = stride_dp_size * args.num_elems_all;
          int idx_tp = tid_y_tp - args.local_tp_rank - 1;
          CIRCULAR_MIN_CLIP(idx_tp, local_tp_size);
          for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
            for (int j = 0; j < local_dp_size; j += stride_dp_size) {
              int idx_dp = tid_y_dp + j - args.local_dp_rank - 1;
              CIRCULAR_MIN_CLIP(idx_dp, local_dp_size);
              int idx = idx_tp + idx_dp * local_tp_size;
              copy_data(reg_data, src);
              replace_neg_zero(reg_data);
              copy_data(uc_data_array[idx] + ofst_copy_out, reg_data);
              src += stride_copy_in;
            }
            ofst_copy_out += vm_buffer_size_base_1_local;
          }
        }
      } else {
        int ofst_copy_base = tid_y * args.elems_per_blk;
        int stride_copy = block_size_y * args.elems_per_blk;
        int64_t ofst_copy_out = vm_ofst_buffer + args.local_rank * args.elems_per_blk;
        T const* src = x_in;
        for (int i = 0; i < args.inter_dp_size; ++i) {
#pragma unroll
          for (int j = 0; j < local_dp_size; ++j) {
            int idx_dp = j - args.local_dp_rank - 1;
            CIRCULAR_MIN_CLIP(idx_dp, local_dp_size);
            int ofst_copy_in = ofst_copy_base;
#pragma unroll
            for (int k = 0; k < local_tp_size; k += block_size_y) {
              if (ofst_copy_in + base_data_idx < num_elems) {
                int idx_tp = tid_y + k - args.local_tp_rank - 1;
                CIRCULAR_MIN_CLIP(idx_tp, local_tp_size);
                int idx = idx_tp + idx_dp * local_tp_size;
                copy_data(reg_data, src + ofst_copy_in);
                replace_neg_zero(reg_data);
                copy_data(uc_data_array[idx] + ofst_copy_out, reg_data);
                ofst_copy_in += stride_copy;
              }
            }
            src += args.num_elems_all;
          }
          ofst_copy_out += vm_buffer_size_base_1_local;
        }
      }
    }
    x_in += x_stride;
    num_elems -= x_ofst_chunk;
    T* dst_self = ns_data_2 + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
    T* dst_base = ns_data + ns_ofst_data;
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        if constexpr (args.use_multicast) {
          T const* src = mc_data + vm_ofst_buffer;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            T* dst = (i == args.inter_dp_rank) ? dst_self : dst_base;
            multicast_load_reduce(reg_data, src);
            copy_data(dst, reg_data);
            dst_base += args.elems_per_blk;
            src += vm_buffer_size_base_1_local;
          }
        } else {
          T const* src = mem_data + vm_ofst_buffer;
          for (int i = 0; i < args.inter_dp_size; ++i) {
            T* dst = (i == args.inter_dp_rank) ? dst_self : dst_base;
            vm_reduce_data<args.local_size>(reg_data, src, args.elems_per_blk);
            copy_data(dst, reg_data);
            dst_base += args.elems_per_blk;
            src += vm_buffer_size_base_1_local;
          }
        }
      }
    });
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank && num_elems > 0) {
      int peer_inter_dp_rank = tid / args.inter_tp_size;
      int peer_inter_tp_rank = tid - peer_inter_dp_rank * args.inter_tp_size;
      T* ns_dst = dst_self - base_data_idx;
      T const* ns_src =
          (peer_inter_dp_rank == args.inter_dp_rank)
              ? ns_dst
              : (ns_data - base_data_idx + ns_ofst_data + peer_inter_dp_rank * args.elems_per_blk);
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
  };
  auto func_ns_reduce_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk, vm_buffer_size_base_1,
       ns_data_2, ns_signal, mc_data_2,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(1);
      if (num_elems > 0) {
        nvshmem_signal_wait_until(ns_signal + ns_ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      if (base_data_idx < num_elems) {
        ns_reduce_data<args.use_inter>(reg_data, ns_data_2 + ns_ofst_data, args.elems_per_blk,
                                       args.inter_size);
        replace_neg_zero(reg_data);
        if constexpr (args.use_multicast) {
          multicast_store(mc_data_2 + vm_ofst_buffer, reg_data);
        } else {
          unicast_store<local_tp_size>(uc_data_array + (local_dp_size - 1) * local_tp_size,
                                       reg_data,
                                       vm_ofst_buffer + vm_buffer_size_base_1 + x_ofst_chunk);
        }
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride,
                                             mem_data_2](T* reg_data, int64_t num_elems,
                                                         int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    T const* src = mem_data_2 + ofst_buffer;
    if constexpr (block_size_y > local_tp_size) {
      if (tid_y < local_tp_size) {
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src + ofst_copy);
          copy_data(x_out + ofst_copy, reg_data);
        }
      }
    } else {
      int stride_copy = block_size_y * args.elems_per_blk;
#pragma unroll
      for (int i = 0; i < local_tp_size; i += block_size_y) {
        if (ofst_copy + base_data_idx < num_elems) {
          vm_receive_data(reg_data, src + ofst_copy);
          copy_data(x_out + ofst_copy, reg_data);
        }
        ofst_copy += stride_copy;
      }
    }
    x_out += x_stride;
  };

  // Read inputs, reduce data in the same node, and send data across different nodes
  func_vm_read_inputs_reduce_data_ns_send_data(
      reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
      buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs, reduce data in the same node, and send data across different nodes
    func_vm_read_inputs_reduce_data_ns_send_data(
        reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
        buffer_info.ns_ofst_signal());
    // Reduce data across different nodes and send data in the same node (previous)
    func_ns_reduce_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Reduce data across different nodes and send data in the same node
  func_ns_reduce_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                   buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                   buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__device__ __forceinline__ void fused_reducescatter_allreduce_opt_bytes2(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const& args) {
  // Virtual Memory Data Buffer Size:
  //   tp_size * (dp_size + 1) * elems_per_blk * sizeof(T)
  // NVSHMEM Data Buffer Size:
  //   2 * inter_size * elems_per_blk * sizeof(T)
  static_assert(is_opt_bytes2_mode(mode) && args.use_inter, "Invalid mode");
  int x_ofst_chunk_local = args.local_tp_rank * args.elems_per_blk;
  int x_ofst_chunk = args.tp_rank * args.elems_per_blk;
  int x_stride_local = local_tp_size * args.elems_per_blk;
  int x_stride = args.tp_size * args.elems_per_blk;
  int vm_buffer_size_base_1_local = args.local_size * args.elems_per_blk;
  int vm_buffer_size_base_1 = args.inter_size * vm_buffer_size_base_1_local;
  int vm_buffer_size_base_2 = x_stride;
  int vm_buffer_size_base = vm_buffer_size_base_1 + vm_buffer_size_base_2;
  int vm_buffer_size;
  if constexpr (args.use_multicast) {
    vm_buffer_size = vm_buffer_size_base + args.elems_per_thd;
  } else {
    vm_buffer_size = vm_buffer_size_base;
  }
  int ns_data_size_base = args.inter_size * args.elems_per_blk;
  int ns_data_size = ns_data_size_base * 2;
  INIT_BASIC_VARIABLES(vm_buffer_size, ns_data_size);

  T *mc_data, *mc_data_2, *uc_data_array[args.local_size];
  uint32_t *mem_signal, *mc_signal;
  if constexpr (args.use_multicast) {
    mc_data = args.mc_buffer_full_all + (vm_ofst_data + args.local_rank * args.elems_per_blk);
    if constexpr (local_tp_size > 1) {
      mc_data_2 =
          args.mc_buffer_tp_all + (vm_ofst_data + vm_buffer_size_base_1 + x_ofst_chunk_local);
    }
    int64_t vm_ofst_signal = vm_ofst_data_base + vm_buffer_size_base;
    mem_signal = reinterpret_cast<uint32_t*>(args.mem_buffer_all + vm_ofst_signal);
    mc_signal = reinterpret_cast<uint32_t*>(args.mc_buffer_full_all + vm_ofst_signal);
  } else {
    set_uc_data_array<args.local_size>(uc_data_array, args.uc_buffer_array_all, vm_ofst_data);
  }
  T* mem_data_2 = mem_data + vm_buffer_size_base_1;
  T* ns_data_2 = ns_data + ns_data_size_base;
  uint64_t* ns_signal_2 = ns_signal + 1;

  auto func_vm_read_inputs_reduce_data_ns_send_data =
      [&x_in, args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local, x_stride,
       vm_buffer_size_base_1_local, mem_data, mc_data, uc_data_array, mem_signal, mc_signal,
       ns_data, ns_data_2, ns_signal](T* reg_data, int64_t num_elems, int64_t vm_ofst_buffer,
                                      int64_t ns_ofst_data, int ns_ofst_signal) -> void {
    if constexpr (args.use_multicast) {
      T* dst = mem_data + (vm_ofst_buffer + tid_y * args.elems_per_blk);
      int stride_copy_out = block_size_y * args.elems_per_blk;
      if constexpr (block_size_y > local_tp_size) {
        constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
        int tid_y_dp = floor_div<local_tp_size>(tid_y);
        int tid_y_tp = mod<local_tp_size>(tid_y);
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            int ofst_copy_in = (tid_y_tp + j * local_tp_size) * args.elems_per_blk;
#pragma unroll
            for (int k = 0; k < local_dp_size; k += stride_dp_size) {
              T const* src = x_in + (tid_y_dp + k + i * local_dp_size) * args.num_elems_all;
              if (ofst_copy_in + base_data_idx < num_elems) {
                copy_data(dst, src + ofst_copy_in);
              }
              dst += stride_copy_out;
            }
          }
        }
      } else {
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
#pragma unroll
            for (int k = 0; k < local_dp_size; ++k) {
              T const* src = x_in + (k + i * local_dp_size) * args.num_elems_all;
#pragma unroll
              for (int l = 0; l < local_tp_size; l += block_size_y) {
                int ofst_copy_in = (tid_y + l + j * local_tp_size) * args.elems_per_blk;
                if (ofst_copy_in + base_data_idx < num_elems) {
                  copy_data(dst, src + ofst_copy_in);
                }
                dst += stride_copy_out;
              }
            }
          }
        }
      }
      __syncthreads();
      if (tid == 0) {
        multicast_add_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mc_signal) + vm_ofst_buffer));
        multicast_wait_signal(
            reinterpret_cast<uint32_t*>(reinterpret_cast<T*>(mem_signal) + vm_ofst_buffer),
            NEG_ZERO_UINT32 + static_cast<uint32_t>(args.local_size));
      }
      __syncthreads();
    } else {
      int64_t ofst_copy_out = vm_ofst_buffer + args.local_rank * args.elems_per_blk;
      if constexpr (block_size_y > local_tp_size) {
        constexpr int stride_dp_size = floor_div<local_tp_size>(block_size_y);
        int tid_y_dp = floor_div<local_tp_size>(tid_y);
        int tid_y_tp = mod<local_tp_size>(tid_y);
        int idx_tp = tid_y_tp - args.local_tp_rank - 1;
        CIRCULAR_MIN_CLIP(idx_tp, local_tp_size);
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            int ofst_copy_in = (tid_y_tp + j * local_tp_size) * args.elems_per_blk;
#pragma unroll
            for (int k = 0; k < local_dp_size; k += stride_dp_size) {
              T const* src = x_in + (tid_y_dp + k + i * local_dp_size) * args.num_elems_all;
              int idx_dp = tid_y_dp + k - args.local_dp_rank - 1;
              CIRCULAR_MIN_CLIP(idx_dp, local_dp_size);
              int idx = idx_tp + idx_dp * local_tp_size;
              if (ofst_copy_in + base_data_idx < num_elems) {
                copy_data(reg_data, src + ofst_copy_in);
                replace_neg_zero(reg_data);
                copy_data(uc_data_array[idx] + ofst_copy_out, reg_data);
              }
            }
            ofst_copy_out += vm_buffer_size_base_1_local;
          }
        }
      } else {
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
#pragma unroll
            for (int k = 0; k < local_dp_size; ++k) {
              T const* src = x_in + (k + i * local_dp_size) * args.num_elems_all;
              int idx_dp = k - args.local_dp_rank - 1;
              CIRCULAR_MIN_CLIP(idx_dp, local_dp_size);
#pragma unroll
              for (int l = 0; l < local_tp_size; l += block_size_y) {
                int ofst_copy_in = (tid_y + l + j * local_tp_size) * args.elems_per_blk;
                int idx_tp = tid_y + l - args.local_tp_rank - 1;
                CIRCULAR_MIN_CLIP(idx_tp, local_tp_size);
                int idx = idx_tp + idx_dp * local_tp_size;
                if (ofst_copy_in + base_data_idx < num_elems) {
                  copy_data(reg_data, src + ofst_copy_in);
                  replace_neg_zero(reg_data);
                  copy_data(uc_data_array[idx] + ofst_copy_out, reg_data);
                }
              }
            }
            ofst_copy_out += vm_buffer_size_base_1_local;
          }
        }
      }
    }
    x_in += x_stride;
    num_elems -= x_ofst_chunk_local;
    T* dst_self = ns_data_2 + (ns_ofst_data + args.inter_rank * args.elems_per_blk);
    T* dst_base = ns_data + ns_ofst_data;
    run_first_block_y<block_size_y>(tid_y, [&] {
      if constexpr (args.use_multicast) {
        T const* src = mc_data + vm_ofst_buffer;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            if (j * x_stride_local + base_data_idx < num_elems) {
              T* dst = (i == args.inter_dp_rank && j == args.inter_tp_rank) ? dst_self : dst_base;
              multicast_load_reduce(reg_data, src);
              copy_data(dst, reg_data);
            }
            dst_base += args.elems_per_blk;
            src += vm_buffer_size_base_1_local;
          }
        }
      } else {
        T const* src = mem_data + vm_ofst_buffer;
        for (int i = 0; i < args.inter_dp_size; ++i) {
          for (int j = 0; j < args.inter_tp_size; ++j) {
            if (j * x_stride_local + base_data_idx < num_elems) {
              T* dst = (i == args.inter_dp_rank && j == args.inter_tp_rank) ? dst_self : dst_base;
              vm_reduce_data<args.local_size>(reg_data, src, args.elems_per_blk);
              copy_data(dst, reg_data);
            }
            dst_base += args.elems_per_blk;
            src += vm_buffer_size_base_1_local;
          }
        }
      }
    });
    __syncthreads();
    if (tid < args.inter_size && tid != args.inter_rank) {
      int peer_inter_dp_rank = tid / args.inter_tp_size;
      int peer_inter_tp_rank = tid - peer_inter_dp_rank * args.inter_tp_size;
      T* ns_dst = dst_self - base_data_idx;
      T const* ns_src = (ns_data - base_data_idx + ns_ofst_data + tid * args.elems_per_blk);
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems - peer_inter_tp_rank * x_stride_local,
                                            static_cast<int64_t>(args.elems_per_blk)));
      int peer_world_rank = args.local_rank + tid * args.local_size;
      if (ns_msg_elems > 0) {
        nvshmem_put128_signal_nbi(ns_dst, ns_src, ns_msg_elems, ns_signal + ns_ofst_signal, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
  };
  auto func_ns_reduce_send_data =
      [args, base_data_idx, tid, tid_y, x_ofst_chunk, ns_data, ns_data_2, ns_signal, ns_signal_2](
          T* reg_data, int64_t num_elems, int64_t ofst_data, int ofst_signal) -> void {
    num_elems -= x_ofst_chunk;
    if (num_elems > 0) {
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal + ofst_signal, NVSHMEM_CMP_EQ, args.inter_size - 1);
      }
      __syncthreads();
      T* dst = ns_data + (ofst_data + args.inter_rank * args.elems_per_blk);
      run_first_block_y<block_size_y>(tid_y, [&] {
        if (base_data_idx < num_elems) {
          ns_reduce_data<args.use_inter>(reg_data, ns_data_2 + ofst_data, args.elems_per_blk,
                                         args.inter_size);
          replace_neg_zero(reg_data);
          copy_data(dst, reg_data);
        }
      });
      __syncthreads();
      int ns_msg_elems =
          floor_div<args.elems_per_thd>(min(num_elems, static_cast<int64_t>(args.elems_per_blk)));
      if (tid < args.inter_tp_size && tid != args.inter_tp_rank) {
        T* ns_dst_src = dst - base_data_idx;
        int peer_world_rank =
            args.local_rank + (tid + args.inter_dp_rank * args.inter_tp_size) * args.local_size;
        nvshmem_put128_signal_nbi(ns_dst_src, ns_dst_src, ns_msg_elems, ns_signal_2 + ofst_signal,
                                  1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
    }
  };
  auto func_ns_receive_data_vm_send_data =
      [&buffer_info, args, base_data_idx, tid, tid_y, x_ofst_chunk_local, x_stride_local,
       vm_buffer_size_base_1, ns_data, ns_signal, ns_signal_2, mem_data_2, mc_data_2,
       uc_data_array](T* reg_data, int64_t num_elems, int64_t ns_ofst_data, int ns_ofst_signal,
                      int ns_ofst_reset, int64_t vm_ofst_buffer) -> void {
    num_elems -= x_ofst_chunk_local;
    if (tid == 0) {
      buffer_info.reset_ns_signal(ns_signal + ns_ofst_reset);
      buffer_info.update_ns_reset_size(2);
      int ns_num_senders =
          min(ceil_div(num_elems, x_stride_local), static_cast<int64_t>(args.inter_tp_size)) -
          (num_elems > args.inter_tp_rank * x_stride_local);
      if (ns_num_senders > 0) {
        nvshmem_signal_wait_until(ns_signal_2 + ns_ofst_signal, NVSHMEM_CMP_EQ, ns_num_senders);
      }
    }
    __syncthreads();
    run_first_block_y<block_size_y>(tid_y, [&] {
      T const* src =
          ns_data + ns_ofst_data + args.inter_dp_rank * args.inter_tp_size * args.elems_per_blk;
      if constexpr (args.use_multicast) {
        int ofst_copy = 0;
        for (int i = 0; i < args.inter_tp_size; ++i) {
          if (ofst_copy + base_data_idx < num_elems) {
            copy_data(reg_data, src);
            if constexpr (local_tp_size > 1) {
              multicast_store(mc_data_2 + (vm_ofst_buffer + ofst_copy), reg_data);
            } else {
              copy_data(mem_data_2 + (vm_ofst_buffer + ofst_copy), reg_data);
            }
          }
          ofst_copy += x_stride_local;
          src += args.elems_per_blk;
        }
      } else {
        int64_t ofst_copy_base = vm_ofst_buffer + vm_buffer_size_base_1 + x_ofst_chunk_local;
        int ofst_copy = 0;
        for (int i = 0; i < args.inter_tp_size; ++i) {
          if (ofst_copy + base_data_idx < num_elems) {
            copy_data(reg_data, src);
            unicast_store<local_tp_size>(uc_data_array + (local_dp_size - 1) * local_tp_size,
                                         reg_data, ofst_copy_base + ofst_copy);
          }
          ofst_copy += x_stride_local;
          src += args.elems_per_blk;
        }
      }
    });
  };
  auto func_vm_receive_data_write_outputs = [&x_out, args, base_data_idx, tid_y, x_stride_local,
                                             x_stride, mem_data_2](T* reg_data, int64_t num_elems,
                                                                   int64_t ofst_buffer) -> void {
    int ofst_copy = tid_y * args.elems_per_blk;
    T const* src = mem_data_2 + ofst_buffer;
    if constexpr (block_size_y > local_tp_size) {
      int stride_copy = local_tp_size * args.elems_per_blk;
      if (tid_y < local_tp_size) {
        for (int i = 0; i < args.inter_tp_size; ++i) {
          if (ofst_copy + base_data_idx < num_elems) {
            vm_receive_data(reg_data, src + ofst_copy);
            copy_data(x_out + ofst_copy, reg_data);
          }
          ofst_copy += stride_copy;
        }
      }
    } else {
      int stride_copy = block_size_y * args.elems_per_blk;
      for (int i = 0; i < args.inter_tp_size; ++i) {
#pragma unroll
        for (int j = 0; j < local_tp_size; j += block_size_y) {
          if (ofst_copy + base_data_idx < num_elems) {
            vm_receive_data(reg_data, src + ofst_copy);
            copy_data(x_out + ofst_copy, reg_data);
          }
          ofst_copy += stride_copy;
        }
      }
    }
    x_out += x_stride;
  };

  // Read inputs, reduce data in the same node, and send data across different nodes
  func_vm_read_inputs_reduce_data_ns_send_data(
      reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
      buffer_info.ns_ofst_signal());
  // Reset previous buffer
  func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  // Reduce and send data across different nodes
  func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                           buffer_info.ns_ofst_signal());
  // Main loop
  for (int64_t t = args.max_num_elems; t > x_stride; t -= x_stride) {
    int ns_ofst_signal_reset = buffer_info.ns_ofst_signal_prev();
    UPDATE_BASIC_VARIABLES(x_stride);
    // Read inputs, reduce data in the same node, and send data across different nodes
    func_vm_read_inputs_reduce_data_ns_send_data(
        reg_data, num_elems_remain, buffer_info.vm_ofst_buffer(), buffer_info.ns_ofst_data(),
        buffer_info.ns_ofst_signal());
    // Receive data across different nodes and send data in the same node (previous)
    func_ns_receive_data_vm_send_data(
        reg_data_prev, num_elems_remain_prev, buffer_info.ns_ofst_data_prev(),
        buffer_info.ns_ofst_signal_prev(), ns_ofst_signal_reset, buffer_info.vm_ofst_buffer_prev());
    // Receive data in the same node and write outputs (previous)
    func_vm_receive_data_write_outputs(reg_data_prev, num_elems_remain_prev,
                                       buffer_info.vm_ofst_buffer_prev());
    // Reduce and send data across different nodes
    func_ns_reduce_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                             buffer_info.ns_ofst_signal());
    // Reset previous buffer
    func_vm_reset_buffer(buffer_info.vm_ofst_buffer_prev());
  }
  // Receive data across different nodes and send data in the same node
  func_ns_receive_data_vm_send_data(reg_data, num_elems_remain, buffer_info.ns_ofst_data(),
                                    buffer_info.ns_ofst_signal(), buffer_info.ns_ofst_signal_prev(),
                                    buffer_info.vm_ofst_buffer());
  // Receive data in the same node and write outputs
  func_vm_receive_data_write_outputs(reg_data, num_elems_remain, buffer_info.vm_ofst_buffer());
  buffer_info.write_gmem(tid);
}

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void fused_reducescatter_allreduce_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args) {
  constexpr MixedCommOp op = MixedCommOp::REDUCESCATTER_ALLREDUCE;
  static_assert(is_valid_op<args.use_local_tp, args.use_local_dp, use_inter_tp, use_inter_dp>(op),
                "Invalid op");
  static_assert(is_valid_mode<args.use_local_tp, use_inter_tp>(mode), "Invalid mode");
  static_assert(is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode),
                "Invalid block_size_y");
  if constexpr (is_opt_waits_mode(mode)) {
    if constexpr (!args.use_inter) {
      fused_reducescatter_allreduce_opt_waits_single<block_size_y>(args);
    } else {
      fused_reducescatter_allreduce_opt_waits_multi<block_size_y>(args);
    }
  } else if constexpr (is_opt_bytes1_mode(mode)) {
    if constexpr (!args.use_inter) {
      if constexpr (args.use_multicast) {
        fused_reducescatter_allreduce_opt_bytes1_multicast_single<block_size_y>(args);
      } else {
        fused_reducescatter_allreduce_opt_bytes1_unicast_single<block_size_y>(args);
      }
    } else {
      fused_reducescatter_allreduce_opt_bytes1_multi<block_size_y>(args);
    }
  } else {
    static_assert(is_opt_bytes2_mode(mode) && args.use_inter, "Invalid mode");
    fused_reducescatter_allreduce_opt_bytes2<block_size_y>(args);
  }
}

#undef CIRCULAR_ADD
#undef CIRCULAR_MINUS
#undef CIRCULAR_MAX_CLIP
#undef CIRCULAR_MIN_CLIP
#undef INIT_BASIC_VARIABLES
#undef UPDATE_BASIC_VARIABLES

}  // namespace mixed_comm
}  // namespace flashinfer
