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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_all_gather_matmul_device_common.cuh"
// The common preamble's fallback helper this module's helper library does not provide; the shared
// header carries another member's library definition of the name (kept translation-unit-local).
__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(THREADS) void kernel_cake_all_gather_matmul_a0785a42337a41e57f18(
    unsigned int* __restrict__ inp, long long* __restrict__ payload_peers,
    long long* __restrict__ signal_peers, unsigned int* __restrict__ counters,
    unsigned int ready_target, int rank, int rows, int chunk_rows, int num_chunks) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int peer_slot = blockIdx.y;
  unsigned int* destination = reinterpret_cast<unsigned int*>(payload_peers[peer_slot]);
  unsigned int* signal = reinterpret_cast<unsigned int*>(signal_peers[peer_slot]);
  unsigned int last_ticket = (unsigned int)(gridDim.x - 1);
#pragma unroll 1
  for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    int row_begin = chunk_idx * chunk_rows;
    int rows_left = rows - row_begin;
    int copy_rows = ((rows_left > chunk_rows) ? chunk_rows : rows_left);
    int vec_begin = row_begin * 1024;
    int vec_end = vec_begin + copy_rows * 1024;
    int slice_count = copy_rows * 256;
    int stride_ks = gridDim.x * 128;
#pragma unroll 1
    for (int s = 0; s < 4; s++) {
      int slice_base = vec_begin + s * 256;
      int base_ks = blockIdx.x * 128 + tid;
      while (slice_count > base_ks + 7 * stride_ks) {
        uint32_t _sysv_ld_0[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_0[0]), "=r"(_sysv_ld_0[1]), "=r"(_sysv_ld_0[2]),
                       "=r"(_sysv_ld_0[3])
                     : "l"(inp + ((slice_base + base_ks / 256 * 1024 + base_ks % 256) * 4))
                     : "memory");
        uint32_t _sysv_ld_1[4];
        asm volatile(
            "ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(_sysv_ld_1[0]), "=r"(_sysv_ld_1[1]), "=r"(_sysv_ld_1[2]), "=r"(_sysv_ld_1[3])
            : "l"(inp +
                  ((slice_base + (base_ks + stride_ks) / 256 * 1024 + (base_ks + stride_ks) % 256) *
                   4))
            : "memory");
        uint32_t _sysv_ld_2[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]),
                       "=r"(_sysv_ld_2[3])
                     : "l"(inp + ((slice_base + (base_ks + 2 * stride_ks) / 256 * 1024 +
                                   (base_ks + 2 * stride_ks) % 256) *
                                  4))
                     : "memory");
        uint32_t _sysv_ld_3[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_3[0]), "=r"(_sysv_ld_3[1]), "=r"(_sysv_ld_3[2]),
                       "=r"(_sysv_ld_3[3])
                     : "l"(inp + ((slice_base + (base_ks + 3 * stride_ks) / 256 * 1024 +
                                   (base_ks + 3 * stride_ks) % 256) *
                                  4))
                     : "memory");
        uint32_t _sysv_ld_4[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_4[0]), "=r"(_sysv_ld_4[1]), "=r"(_sysv_ld_4[2]),
                       "=r"(_sysv_ld_4[3])
                     : "l"(inp + ((slice_base + (base_ks + 4 * stride_ks) / 256 * 1024 +
                                   (base_ks + 4 * stride_ks) % 256) *
                                  4))
                     : "memory");
        uint32_t _sysv_ld_5[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_5[0]), "=r"(_sysv_ld_5[1]), "=r"(_sysv_ld_5[2]),
                       "=r"(_sysv_ld_5[3])
                     : "l"(inp + ((slice_base + (base_ks + 5 * stride_ks) / 256 * 1024 +
                                   (base_ks + 5 * stride_ks) % 256) *
                                  4))
                     : "memory");
        uint32_t _sysv_ld_6[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_6[0]), "=r"(_sysv_ld_6[1]), "=r"(_sysv_ld_6[2]),
                       "=r"(_sysv_ld_6[3])
                     : "l"(inp + ((slice_base + (base_ks + 6 * stride_ks) / 256 * 1024 +
                                   (base_ks + 6 * stride_ks) % 256) *
                                  4))
                     : "memory");
        uint32_t _sysv_ld_7[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_7[0]), "=r"(_sysv_ld_7[1]), "=r"(_sysv_ld_7[2]),
                       "=r"(_sysv_ld_7[3])
                     : "l"(inp + ((slice_base + (base_ks + 7 * stride_ks) / 256 * 1024 +
                                   (base_ks + 7 * stride_ks) % 256) *
                                  4))
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + base_ks / 256 * 1024 + base_ks % 256) * 4)),
                     "r"((_sysv_ld_0)[0]), "r"((_sysv_ld_0)[1]), "r"((_sysv_ld_0)[2]),
                     "r"((_sysv_ld_0)[3])
                     : "memory");
        asm volatile(
            "st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                destination +
                ((slice_base + (base_ks + stride_ks) / 256 * 1024 + (base_ks + stride_ks) % 256) *
                 4)),
            "r"((_sysv_ld_1)[0]), "r"((_sysv_ld_1)[1]), "r"((_sysv_ld_1)[2]), "r"((_sysv_ld_1)[3])
            : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 2 * stride_ks) / 256 * 1024 +
                                         (base_ks + 2 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_2)[0]), "r"((_sysv_ld_2)[1]), "r"((_sysv_ld_2)[2]),
                     "r"((_sysv_ld_2)[3])
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 3 * stride_ks) / 256 * 1024 +
                                         (base_ks + 3 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_3)[0]), "r"((_sysv_ld_3)[1]), "r"((_sysv_ld_3)[2]),
                     "r"((_sysv_ld_3)[3])
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 4 * stride_ks) / 256 * 1024 +
                                         (base_ks + 4 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_4)[0]), "r"((_sysv_ld_4)[1]), "r"((_sysv_ld_4)[2]),
                     "r"((_sysv_ld_4)[3])
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 5 * stride_ks) / 256 * 1024 +
                                         (base_ks + 5 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_5)[0]), "r"((_sysv_ld_5)[1]), "r"((_sysv_ld_5)[2]),
                     "r"((_sysv_ld_5)[3])
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 6 * stride_ks) / 256 * 1024 +
                                         (base_ks + 6 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_6)[0]), "r"((_sysv_ld_6)[1]), "r"((_sysv_ld_6)[2]),
                     "r"((_sysv_ld_6)[3])
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         destination + ((slice_base + (base_ks + 7 * stride_ks) / 256 * 1024 +
                                         (base_ks + 7 * stride_ks) % 256) *
                                        4)),
                     "r"((_sysv_ld_7)[0]), "r"((_sysv_ld_7)[1]), "r"((_sysv_ld_7)[2]),
                     "r"((_sysv_ld_7)[3])
                     : "memory");
        base_ks = base_ks + 8 * stride_ks;
      }
#pragma unroll 1
      for (int i = base_ks; i < slice_count; i += stride_ks) {
        int vector_ks = slice_base + i / 256 * 1024 + i % 256;
        uint32_t _sysv_ld_8[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(_sysv_ld_8[0]), "=r"(_sysv_ld_8[1]), "=r"(_sysv_ld_8[2]),
                       "=r"(_sysv_ld_8[3])
                     : "l"(inp + (vector_ks * 4))
                     : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(destination +
                                                                               (vector_ks * 4)),
                     "r"((_sysv_ld_8)[0]), "r"((_sysv_ld_8)[1]), "r"((_sysv_ld_8)[2]),
                     "r"((_sysv_ld_8)[3])
                     : "memory");
      }
      __syncthreads();
      if (warp == 0) {
        if (elect_sync()) {
          uint32_t _atomic_inc_old_0;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_0)
                       : "l"(&counters[(peer_slot * num_chunks + chunk_idx) * 4 + s]),
                         "r"(static_cast<uint32_t>(last_ticket))
                       : "memory");
          if (_atomic_inc_old_0 == last_ticket) {
            __threadfence_system();
            asm volatile(
                "st.relaxed.sys.u32 [%0], %1;" ::"l"((reinterpret_cast<unsigned int*>(signal) +
                                                      ((rank * num_chunks + chunk_idx) * 4 + s))),
                "r"(static_cast<unsigned int>(ready_target))
                : "memory");
          }
        }
      }
    }
  }
}

}  // extern "C"
