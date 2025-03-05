/*
 * Copyright (c) 2025 by FlashInfer team.
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

#ifndef FLASHINFER_PROFILER_CUH_
#define FLASHINFER_PROFILER_CUH_
#include <cuda.h>

namespace flashinfer {

__device__ __forceinline__ uint32_t get_block_idx() {
  return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

__device__ __forceinline__ uint32_t get_thread_idx() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

constexpr uint32_t BLOCK_GROUP_IDX_MASK = 0xFFFFF;
constexpr uint32_t EVENT_IDX_MASK = 0x3FF;
constexpr uint32_t BEGIN_END_MASK = 0x3;

constexpr uint32_t EVENT_IDX_SHIFT = 2;
constexpr uint32_t BLOCK_GROUP_IDX_SHIFT = 12;

constexpr uint32_t EVENT_BEGIN = 0x0;
constexpr uint32_t EVENT_END = 0x1;
constexpr uint32_t EVENT_INSTANT = 0x2;

__device__ __forceinline__ uint32_t encode_tag(uint32_t block_group_idx, uint32_t event_idx,
                                               uint32_t event_type) {
  return (block_group_idx << BLOCK_GROUP_IDX_SHIFT) | (event_idx << EVENT_IDX_SHIFT) | event_type;
}

__device__ __forceinline__ uint32_t get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}

struct ProfilerEntry {
  union {
    struct {
      uint32_t nblocks;
      uint32_t ngroups;
    };
    struct {
      uint32_t tag;
      uint32_t delta_time;
    };
    uint64_t raw;
  };
};

#ifdef FLASHINFER_ENABLE_PROFILER
#define PROFILER_CLOSURE_PARAMS_DECL \
  uint64_t* profiler_write_ptr;      \
  uint32_t profiler_write_stride;    \
  uint32_t profiler_entry_tag_base;  \
  bool profiler_write_thread_predicate;

#define PROFILER_PARAMS_DECL uint64_t* profiler_buffer;

#define PROFILER_INIT(params, smem_storage, closure, group_idx, num_groups,                     \
                      write_thread_predicate)                                                   \
  volatile ProfilerEntry entry;                                                                 \
  if (get_block_idx() == 0 && get_thread_idx() == 0) {                                          \
    entry.nblocks = get_num_blocks();                                                           \
    entry.ngroups = num_groups;                                                                 \
    params.profiler_buffer[0] = entry.raw;                                                      \
  }                                                                                             \
  closure.profiler_write_ptr =                                                                  \
      params.profiler_buffer + 1 + get_block_idx() * num_groups + group_idx;                    \
  closure.profiler_write_stride = get_num_blocks() * num_groups;                                \
  closure.profiler_entry_tag_base = encode_tag(get_block_idx() * num_groups + group_idx, 0, 0); \
  closure.profiler_write_thread_predicate = write_thread_predicate;

#define PROFILER_EVENT_START(closure, event)                                                  \
  if (closure.profiler_write_thread_predicate) {                                              \
    entry.tag =                                                                               \
        closure.profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_BEGIN; \
    entry.delta_time = get_timestamp();                                                       \
    *closure.profiler_write_ptr = entry.raw;                                                  \
    closure.profiler_write_ptr += closure.profiler_write_stride;                              \
  }                                                                                           \
  __threadfence_block();

#define PROFILER_EVENT_END(closure, event)                                                  \
  __threadfence_block();                                                                    \
  if (closure.profiler_write_thread_predicate) {                                            \
    entry.tag =                                                                             \
        closure.profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_END; \
    entry.delta_time = get_timestamp();                                                     \
    *closure.profiler_write_ptr = entry.raw;                                                \
    closure.profiler_write_ptr += closure.profiler_write_stride;                            \
  }

#define PROFILER_EVENT_INSTANT(closure, event)                                                  \
  __threadfence_block();                                                                        \
  if (closure.profiler_write_thread_predicate) {                                                \
    entry.tag =                                                                                 \
        closure.profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_INSTANT; \
    entry.delta_time = get_timestamp();                                                         \
    *closure.profiler_write_ptr = entry.raw;                                                    \
  }                                                                                             \
  __threadfence_block();

#else

#define PROFILER_CLOSURE_PARAMS_DECL
#define PROFILER_PARAMS_DECL
#define PROFILER_INIT(params, smem_storage, closure, group_idx, num_groups, write_thread_predicate)
#define PROFILER_EVENT_START(closure, event)
#define PROFILER_EVENT_END(closure, event)
#define PROFILER_EVENT_INSTANT(closure, event)

#endif

}  // namespace flashinfer

#endif  // FLASHINFER_PROFILER_CUH_
