#pragma once
#include <cuda_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <stdexcept>
#include <string>

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ inline int ld_acquire(const int32_t* __restrict__ addr) {
  int res;
  asm volatile("ld.acquire.cta.b32 %0, [%1];" : "=r"(res) : "l"(addr) : "memory");
  return res;
}

// topk kernel v2
template <typename T>
__host__ __device__ __forceinline__ T divup(T a, T b) {
  return (a + b - 1) / b;
}

// PTX functions
__device__ __forceinline__ uint32_t getLaneId() {
  uint32_t laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

__device__ __forceinline__ uint32_t getWarpId() {
  uint32_t warpid;
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}
__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

// Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
#pragma unroll
  for (int i = 1; i <= 16; i <<= 1)  // 16 = LANE_COUNT >> 1
  {
    const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
    if (getLaneId() >= i) val += t;
  }

  return val;
}

template <int num_threads>
__device__ __forceinline__ uint32_t InclusiveWarpDownScan(uint32_t val) {
#pragma unroll
  for (int i = 1; i <= (num_threads >> 1); i <<= 1)  // 16 = LANE_COUNT >> 1
  {
    const uint32_t t = __shfl_down_sync(0xffffffff, val, i, 32);
    if (getLaneId() < num_threads - i) val += t;
  }

  return val;
}
__device__ inline float2 explicit_load_float2(const float2* ptr) {
  float2 res;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v2.f32 {%0,%1}, [%2];"
      : "=f"(res.x), "=f"(res.y)
      : "l"(ptr)
      : "memory");
  return res;
}

__device__ __forceinline__ void reduce_shared(int* addr, int val) {
  asm("red.relaxed.cta.shared::cta.add.s32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}
template <auto* kernel_func, size_t smem_bytes>
void setup_kernel_smem_once() {
  static const cudaError_t result = []() -> cudaError_t {
    auto func_ptr = kernel_func;

    return cudaFuncSetAttribute(func_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }();
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ") +
        cudaGetErrorString(result));
  }
}
template <auto* kernel_func>
void setup_non_portable_clusters_once() {
  static const cudaError_t result = []() -> cudaError_t {
    auto func_ptr = kernel_func;
    return cudaFuncSetAttribute(func_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }();
  if (result != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s \n", cudaGetErrorString(result));
    throw std::runtime_error("cuda set non portable cluster error");
  }
}

__device__ inline void mbarrier_wait(uint64_t* mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
      "%1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}" ::"l"(mbar_addr),
      "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_init(uint64_t* mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_arrive(uint64_t* mbar_addr, int expect_size) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"l"(mbar_addr),
      "r"(expect_size)
      : "memory");
}

__device__ inline void mbarrier_cpy(int* src, int* dst, int num_bytes, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"l"(dst),
      "l"(src), "r"(num_bytes), "l"(mbar)
      : "memory");
}

__device__ inline int mbarrier_arrive_no_expect_tx(uint64_t* mbar) {
  int tok;
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];"
               : "=r"(tok)
               : "l"(mbar)
               : "memory");
  return tok;
}

__host__ __device__ inline void print_bits(uint32_t bits) {
  for (int i = 0; i < 32; i++) {
    printf("%d", (bits >> (31 - i)) & 1);
  }
}

// run_vectorized_v2 / run_vectorized_v4 / run_vectorized
//
// Helpers that iterate over a 1-D logits array using vectorised loads and invoke
// a per-element callback f(float value, int global_index).
//
// Template parameters (shared across all three variants):
//   NumBlocks    – number of blocks in the cooperative cluster assigned to this
//                  sequence.  The grid stride is NumBlocks * Threads (or
//                  NumBlocks * Threads * VecType for the vectorised inner loop),
//                  so that each block processes a disjoint, interleaved slice.
//   Threads      – number of threads per block (typically 1024).  Each thread
//                  handles one float2 / float4 element per inner-loop iteration.
//   UnRollFactor – static unroll depth for the inner loop.  The loop body is
//                  replicated UnRollFactor times by the compiler, reducing loop
//                  overhead and improving instruction-level parallelism.
//   VecType      – (run_vectorized only) vector width: 2 → float2 loads,
//                  4 → float4 loads.  Wider loads reduce memory transaction
//                  overhead on L2/DRAM.  Must be 2 or 4.
//   F            – callable with signature void(float value, int index).
//
// The main loop covers the largest aligned prefix of seq_len that fits an exact
// number of full grid strides × UnRollFactor tiles.  run_vectorized then falls
// through to a scalar tail loop that handles the remaining elements, ensuring
// every index in [0, seq_len) is visited exactly once.

// float2 vectorised inner loop (2 floats per load).
template <int NumBlocks, int Threads, int UnRollFactor, typename F>
__device__ __forceinline__ void run_vectorized_v2(const float2* logits, const int seq_len,
                                                  const int block_id, F f) {
  constexpr int ElemPerBlock = Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;

  for (int t = 0; t < seq_len / (2 * GridStride * UnRollFactor); t++) {
#pragma unroll
    for (int j = 0; j < UnRollFactor; j++) {
      int offset =
          t * GridStride * UnRollFactor + j * GridStride + block_id * ElemPerBlock + threadIdx.x;
      float2 val = logits[offset];
      f(val.x, offset * 2);
      f(val.y, offset * 2 + 1);
    }
  }
}

// float4 vectorised inner loop (4 floats per load).
template <int NumBlocks, int Threads, int UnRollFactor, typename F>
__device__ __forceinline__ void run_vectorized_v4(const float4* logits, const int seq_len,
                                                  const int block_id, F f) {
  constexpr int ElemPerBlock = Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;

  for (int t = 0; t < seq_len / (4 * GridStride * UnRollFactor); t++) {
#pragma unroll
    for (int j = 0; j < UnRollFactor; j++) {
      int offset =
          t * GridStride * UnRollFactor + j * GridStride + block_id * ElemPerBlock + threadIdx.x;
      float4 val = logits[offset];
      f(val.x, offset * 4);
      f(val.y, offset * 4 + 1);
      f(val.z, offset * 4 + 2);
      f(val.w, offset * 4 + 3);
    }
  }
}

// Dispatch to the float2 or float4 inner loop based on VecType, then handle
// any tail elements that don't fill a complete vectorised tile with a scalar loop.
template <int NumBlocks, int Threads, int UnrollFactor, int VecType, typename F>
__device__ __forceinline__ void run_vectorized(const float* logits, const int seq_len,
                                               const int block_id, F f) {
  static_assert(VecType == 2 || VecType == 4, "expected VecType == 2 or 4");
  if (VecType == 2) {
    run_vectorized_v2<NumBlocks, Threads, UnrollFactor>((float2*)logits, seq_len, block_id, f);
  } else if (VecType == 4) {
    run_vectorized_v4<NumBlocks, Threads, UnrollFactor>((float4*)logits, seq_len, block_id, f);
  }

  // Scalar tail: cover elements not reached by the aligned vectorised loop above.
  constexpr int ElemPerBlock = VecType * Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;
  int leftover_offset = (seq_len / (GridStride * UnrollFactor)) * GridStride * UnrollFactor;
  for (int i = leftover_offset + threadIdx.x + block_id * Threads; i < seq_len;
       i += Threads * NumBlocks) {
    f(logits[i], i);
  }
}

__device__ __forceinline__ int cum_sum(int* s_hist_buf) {
  constexpr int RADIX = 256;
  const int warp_idx = threadIdx.x / 32;

  __shared__ int reduce_buf[8];
  int val = 0;
  if (threadIdx.x < RADIX) {
    val = s_hist_buf[threadIdx.x];
    val = InclusiveWarpDownScan<32>(val);
    if (getLaneId() == 0 && warp_idx < 8) {
      reduce_buf[warp_idx] = val;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int cum_val = InclusiveWarpDownScan<8>(threadIdx.x < 8 ? reduce_buf[threadIdx.x] : 0);
    __syncwarp();
    if (threadIdx.x < 8) {
      reduce_buf[threadIdx.x] = cum_val;
    }
  }
  __syncthreads();
  if (warp_idx < 7) {
    val += reduce_buf[warp_idx + 1];
  }
  return val;
}
