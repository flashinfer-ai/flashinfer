/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// SM120 GLM full-tile TMA transport. FP8 MMA and softmax stay unchanged.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmha/hopper/arrive_wait.h>
#include <fmha/smem_tile.h>
#include <fmha/smem_tile_v.h>

#include <stdexcept>

namespace fmha {

#if CUDART_VERSION >= 12000
using Sm120_tensor_map = CUtensorMap;
#else
// Older toolkits still parse this header for non-SM120 kernels.
struct alignas(64) Sm120_tensor_map {
  uint64_t opaque[16];
};
#endif
static_assert(sizeof(Sm120_tensor_map) == 128, "CUDA tensor maps require 128 bytes");

template <typename Params>
struct Sm120_tma_params : Params {
  Sm120_tensor_map glm_q, glm_k, glm_v;
};

template <typename Params>
inline Sm120_tma_params<Params> make_sm120_tma_params(Params const& params, int total_q,
                                                      int total_kv) {
#if CUDART_VERSION >= 12080
  int64_t const row_stride = params.is_s_padded ? int64_t(params.h) * 256 : 256;
  if (params.d != 256 || params.dv != 256 || params.h_kv != params.h ||
      params.q_stride_in_bytes != row_stride ||
      params.k_stride_in_bytes != params.q_stride_in_bytes ||
      params.v_stride_in_bytes != params.q_stride_in_bytes) {
    throw std::invalid_argument(
        "SM120 GLM TMA requires padded BSHD or packed head-major Q/K/V with D=256");
  }
  Sm120_tma_params<Params> result;
  static_cast<Params&>(result) = params;
  using Encode = decltype(&cuTensorMapEncodeTiled);
  static Encode encode = [] {
    void* ptr = nullptr;
    auto status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000,
                                                   cudaEnableDefault, nullptr);
    if (status != cudaSuccess || !ptr) throw std::runtime_error("TMA encoder unavailable");
    return reinterpret_cast<Encode>(ptr);
  }();
  // Split D into two 128-byte halves. The S axis precedes the half axis in
  // shared memory, giving two swizzled 64x128 tiles with one TMA operation.
  uint32_t box[5] = {128, 64, 2, 1, 1}, steps[5] = {1, 1, 1, 1, 1};
  auto set = [&](CUtensorMap& desc, void* ptr, int total_tokens) {
    uint64_t const tokens = params.is_s_padded ? params.s : total_tokens;
    uint64_t dims[5] = {128, tokens, 2, uint64_t(params.h),
                        uint64_t(params.is_s_padded ? params.b : 1)};
    uint64_t strides[4] = {uint64_t(row_stride), 128, params.is_s_padded ? 256 : tokens * 256,
                           tokens * params.h * 256};
    auto status = encode(&desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, ptr, dims, strides, box, steps,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) throw std::runtime_error("SM120 GLM TMA descriptor failed");
  };
  set(result.glm_q, params.q_ptr, total_q);
  set(result.glm_k, params.k_ptr, total_kv);
  set(result.glm_v, params.v_ptr, total_kv);
  return result;
#else
  throw std::runtime_error("SM120 GLM TMA requires CUDA 12.8 or newer");
#endif
}

// Each 64x256 tile is two swizzled 64x128 tiles. Move the column-half bit
// above the six row bits; the original row XOR is already the TMA swizzle.
inline __device__ uint32_t glm_tma_offset(uint32_t offset) {
  return (offset & 127) | ((offset & 0x3f00) >> 1) | ((offset & 128) << 6);
}

template <typename Base>
struct Sm120_tma_qk : Base {
  using Fragment = typename Base::Fragment;
  using Mma_tile = typename Base::Mma_tile;
  inline __device__ Sm120_tma_qk(void* smem, int tidx) : Base(smem, tidx) {
    this->smem_read_offset_ = glm_tma_offset(this->smem_read_offset_);
  }
  template <int N>
  inline __device__ void load(Fragment (&frag)[N], int ki) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      uint32_t offset = (this->smem_read_offset_ ^ glm_tma_offset(ki * 32)) + i * 16 * 128;
      uint4 tmp;
      ldsm(tmp, this->smem_ + offset);
      frag[i].reg(0) = tmp.x;
      frag[i].reg(1) = tmp.y;
      frag[i].reg(2) = tmp.z;
      frag[i].reg(3) = tmp.w;
    }
  }
};

template <typename Base>
struct Sm120_tma_v : Base {
  using Fragment = typename Base::Fragment;
  inline __device__ Sm120_tma_v(void* smem, int tidx) : Base(smem, tidx) {
    int k = tidx % 32;
    // Preserve FMHA's P fragment K ordering: 0,1,8,9 rather than 0,1,2,3.
    int row = (k & 17) | ((k & 2) << 2) | ((k & 12) >> 1);
    this->smem_read_offset_ = row * 128 + (row % 8) * 16;
  }
  inline __device__ void load_one(Fragment& frag, int ni, int ki) {
    uint32_t addr =
        this->smem_ + (this->smem_read_offset_ ^ glm_tma_offset(ni * 16)) + ki * 32 * 128;
    uint32_t a, b, c, d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3},[%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "r"(addr));
#else
    asm volatile("trap;");
#endif
    frag.reg(0) = a;
    frag.reg(1) = c;
    frag.reg(2) = b;
    frag.reg(3) = d;
  }
  template <int N>
  inline __device__ void load(Fragment (&frag)[N], int ki) {
#pragma unroll
    for (int ni = 0; ni < N; ++ni) load_one(frag[ni], ni, ki);
  }
};

inline __device__ uint32_t glm_tma_elect() {
  uint32_t elected = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.b32 %0, 1, 0, p; }"
               : "=r"(elected));
#endif
  return elected;
}

struct Sm120_tma_transport {
  Arrive_wait barriers;
  uint32_t base;
  int head, batch;
  inline __device__ Sm120_tma_transport(void* smem, uint64_t* bars, int head, int batch,
                                        bool init = true)
      : barriers(bars), base(__nvvm_get_smem_pointer(smem)), head(head), batch(batch) {
    if (!init) return;
    if (threadIdx.x == 0) {
      bar_create(bars, 1);
      bar_create(bars + 1, 1);
      fence_view_async_shared();
    }
    __syncthreads();
  }
  inline __device__ void load(Sm120_tensor_map const& desc, int tile, int row, int bar) {
    if (threadIdx.x < 32) {
      uint32_t elected = glm_tma_elect();
      auto barrier = __nvvm_get_smem_pointer(barriers.get_bar_addr(bar));
      uint32_t dst = base + tile * 16384;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
      asm volatile(
          "{ .reg .pred p; setp.ne.u32 p, %6, 0; "
          "@p cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1, {0, %2, 0, %3, %4}], [%5]; }" ::"r"(dst),
          "l"(reinterpret_cast<uint64_t>(&desc)), "r"(row), "r"(head), "r"(batch), "r"(barrier),
          "r"(elected)
          : "memory");
#endif
    }
  }
  inline __device__ void expect(int bar, int bytes) {
    if (threadIdx.x < 32) barriers.bar_arrive_set_transactioncnt(bar, bytes, glm_tma_elect());
  }
  inline __device__ void wait(int bar, int phase) {
    barriers.bar_wait(bar, phase);
    asm volatile("" ::: "memory");
  }
};
}  // namespace fmha
