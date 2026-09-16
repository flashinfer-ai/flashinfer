// Native launch bridge; generated device and TVM-FFI sources remain unchanged.
#include <algorithm>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"
#include "cake_ulysses_dispatch.cuh"

#ifndef CAKE_PEER_POINTER_TABLE_DECLARED
#define CAKE_PEER_POINTER_TABLE_DECLARED
template<typename T, int Capacity = 8> struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
#endif
extern "C" __global__ void kernel_cake_ulysses_a2a_060d10a295b39a821b1c(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_0a5f70d2046773fe2222(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_3997ed9b5eff01a95e7b(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_3e15799aefc2a6dd67fd(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_424cc8b9443592d967af(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_529ae577ff5c50923a78(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_54deb052ac6ae2b15662(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_5f26453ba41d53a97ac1(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_73c8c0e7219ae91ecfe6(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_81d93e801f61b0d2b4e0(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_81eb3da8eb3723c0e75a(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_9d0ea74ee5d92c570a9f(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_9d71bccb13996aa8f2e5(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_aa2652f6707abed77ee5(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_b22486e951c99364ee15(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_b4d01efa1606195e00e8(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_b538db2a3106c4836810(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_db6770e67a1e623ddc47(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_e5068bdd090c78c81537(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_e7639882cc46546b168f(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_ef0d2d5321e43f0d4d84(float* __restrict__ inp, CakePeerPointerTable<float> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_f492358b1a26be5cc57e(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_f5f6d5e72126f2d6800b(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_f612b1f375cd6096585f(__half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D);

namespace flashinfer::comm::ulysses {
static_assert(kMaxBlocks == 36);
static_assert(kUlyssesThreads == 512);
static_assert(sizeof(RankData) == 64 && alignof(RankData) == 16);
static_assert(sizeof(RankSignals) == 64 && alignof(RankSignals) == 16);
static_assert(sizeof(Signal) == 3456 && alignof(Signal) == 128);
static_assert(offsetof(Signal, self_counter) == 0);
static_assert(offsetof(Signal, peer_counter) == 1152);

cudaError_t LaunchGeneratedUlysses(UlyssesA2A* fa, void* inp, int dtype,
                                   int B, int S_local, int H_local, int D,
                                   int mode, cudaStream_t stream) {
  const void* kernel = nullptr;
  size_t smem = 0;
  switch (dtype) {
    case float16_code:
      switch (fa->world_size_) {
        case 2:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_db6770e67a1e623ddc47
                             : (const void*)kernel_cake_ulysses_a2a_e7639882cc46546b168f;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 4:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_0a5f70d2046773fe2222
                             : (const void*)kernel_cake_ulysses_a2a_424cc8b9443592d967af;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        case 6:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_e5068bdd090c78c81537
                             : (const void*)kernel_cake_ulysses_a2a_f612b1f375cd6096585f;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 8:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_060d10a295b39a821b1c
                             : (const void*)kernel_cake_ulysses_a2a_9d0ea74ee5d92c570a9f;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        default: return cudaErrorInvalidValue;
      }
      break;
    case bfloat16_code:
      switch (fa->world_size_) {
        case 2:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_f5f6d5e72126f2d6800b
                             : (const void*)kernel_cake_ulysses_a2a_f492358b1a26be5cc57e;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 4:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_9d71bccb13996aa8f2e5
                             : (const void*)kernel_cake_ulysses_a2a_b4d01efa1606195e00e8;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        case 6:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_529ae577ff5c50923a78
                             : (const void*)kernel_cake_ulysses_a2a_81d93e801f61b0d2b4e0;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 8:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_aa2652f6707abed77ee5
                             : (const void*)kernel_cake_ulysses_a2a_81eb3da8eb3723c0e75a;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        default: return cudaErrorInvalidValue;
      }
      break;
    case float32_code:
      switch (fa->world_size_) {
        case 2:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_73c8c0e7219ae91ecfe6
                             : (const void*)kernel_cake_ulysses_a2a_54deb052ac6ae2b15662;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 4:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_ef0d2d5321e43f0d4d84
                             : (const void*)kernel_cake_ulysses_a2a_3e15799aefc2a6dd67fd;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        case 6:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_5f26453ba41d53a97ac1
                             : (const void*)kernel_cake_ulysses_a2a_b538db2a3106c4836810;
          smem = mode == 0 ? 0u
                           : 0u;
          break;
        case 8:
          kernel = mode == 0 ? (const void*)kernel_cake_ulysses_a2a_b22486e951c99364ee15
                             : (const void*)kernel_cake_ulysses_a2a_3997ed9b5eff01a95e7b;
          smem = mode == 0 ? 128u
                           : 128u;
          break;
        default: return cudaErrorInvalidValue;
      }
      break;
    default: return cudaErrorInvalidValue;
  }
  // The handle already owns the exact by-value eight-entry tables. Their
  // active peer addresses and signal epochs remain unchanged across calls.
  void* args[] = {&inp, &fa->out_ptrs_, &fa->sg_, &fa->self_sg_, &fa->rank_,
                  &B, &S_local, &H_local, &D};
  const int64_t rows = static_cast<int64_t>(B) * fa->world_size_ * S_local;
  const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(kMaxBlocks, rows));
  return cudaLaunchKernel(kernel, dim3(blocks, 1, 1), dim3(kUlyssesThreads, 1, 1),
                          args, smem, stream);
}
}  // namespace flashinfer::comm::ulysses
