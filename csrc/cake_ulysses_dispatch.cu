// Native launch bridge for the generated Ulysses kernels (rendered by the Cake
// source exporter from the delivered program table; do not edit by hand).
// Preserve the include order used by the validated launch bridge.
// clang-format off
#include <algorithm>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"
#include "cake_ulysses_dispatch.cuh"
// clang-format on

#ifndef CAKE_PEER_POINTER_TABLE_DECLARED
#define CAKE_PEER_POINTER_TABLE_DECLARED
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable {
  T* ptrs[Capacity];
};
#endif
extern "C" __global__ void kernel_cake_ulysses_a2a_45915ee873e8d754fc8a(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_b3e242482bd4ee11df46(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_2862f9aa8f469dc4a895(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_b3fa2ce10ba84e438405(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_a43ae84038aed9942618(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_d2e1d5cc0834259b4b8a(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_f01765220b0ea3e0a458(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_8f16ac9ca3b30bbdb8ad(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_646d36c61be114175d4f(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_67f08c8843354b114fa0(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_329637d2bca31c68941f(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_1b864007ccae50b2c64b(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_7f52c5383930c70fcdfc(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_13d5bcbfe2305fa3b33f(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_37f3eb80fcfbdc94ec58(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_6c0f4daca24cb49e50b3(
    __nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_aa7f336404576b7995f9(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_7d3317288d67a76a904a(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_5e2b99e811e0f2a39e8c(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_0bfaf1875f1eae295499(
    __half* __restrict__ inp, CakePeerPointerTable<__half> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_9f53afbc054ade876d63(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_aa3de711d5012d241018(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_3b65cc535a8bb3d2f951(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);
extern "C" __global__ void kernel_cake_ulysses_a2a_80ac1f368b5e2e162a4f(
    float* __restrict__ inp, CakePeerPointerTable<float> staging_peers,
    CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self,
    int32_t pg_rank, int B, int S_local, int H_local, int D);

namespace flashinfer::comm::ulysses {
static_assert(kMaxBlocks == 36);
static_assert(kUlyssesThreads == 512);
static_assert(sizeof(RankData) == 64 && alignof(RankData) == 16);
static_assert(sizeof(RankSignals) == 64 && alignof(RankSignals) == 16);
static_assert(sizeof(Signal) == 3456 && alignof(Signal) == 128);
static_assert(offsetof(Signal, self_counter) == 0);
static_assert(offsetof(Signal, peer_counter) == 1152);

namespace {
struct GeneratedUlyssesKernel {
  int dtype_code;
  int world_size;
  int mode;
  unsigned dynamic_smem_bytes;
  const void* kernel;
};
// (dtype, world size, mode) -> generated kernel; one row per delivered route.
const GeneratedUlyssesKernel kGeneratedUlyssesKernels[] = {
    {bfloat16_code, 2, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_45915ee873e8d754fc8a)},
    {bfloat16_code, 4, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_b3e242482bd4ee11df46)},
    {bfloat16_code, 6, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_2862f9aa8f469dc4a895)},
    {bfloat16_code, 8, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_b3fa2ce10ba84e438405)},
    {float16_code, 2, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_a43ae84038aed9942618)},
    {float16_code, 4, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_d2e1d5cc0834259b4b8a)},
    {float16_code, 6, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_f01765220b0ea3e0a458)},
    {float16_code, 8, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_8f16ac9ca3b30bbdb8ad)},
    {float32_code, 2, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_646d36c61be114175d4f)},
    {float32_code, 4, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_67f08c8843354b114fa0)},
    {float32_code, 6, 0, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_329637d2bca31c68941f)},
    {float32_code, 8, 0, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_1b864007ccae50b2c64b)},
    {bfloat16_code, 2, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_7f52c5383930c70fcdfc)},
    {bfloat16_code, 4, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_13d5bcbfe2305fa3b33f)},
    {bfloat16_code, 6, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_37f3eb80fcfbdc94ec58)},
    {bfloat16_code, 8, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_6c0f4daca24cb49e50b3)},
    {float16_code, 2, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_aa7f336404576b7995f9)},
    {float16_code, 4, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_7d3317288d67a76a904a)},
    {float16_code, 6, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_5e2b99e811e0f2a39e8c)},
    {float16_code, 8, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_0bfaf1875f1eae295499)},
    {float32_code, 2, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_9f53afbc054ade876d63)},
    {float32_code, 4, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_aa3de711d5012d241018)},
    {float32_code, 6, 1, 0u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_3b65cc535a8bb3d2f951)},
    {float32_code, 8, 1, 128u,
     reinterpret_cast<const void*>(kernel_cake_ulysses_a2a_80ac1f368b5e2e162a4f)},
};
}  // namespace

cudaError_t LaunchGeneratedUlysses(UlyssesA2A* fa, void* inp, int dtype, int B, int S_local,
                                   int H_local, int D, int mode, cudaStream_t stream) {
  for (const GeneratedUlyssesKernel& entry : kGeneratedUlyssesKernels) {
    if (entry.dtype_code != dtype || entry.world_size != fa->world_size_ || entry.mode != mode) {
      continue;
    }
    // The handle already owns the exact by-value eight-entry tables. Their
    // active peer addresses and signal epochs remain unchanged across calls.
    void* args[] = {&inp,     &fa->out_ptrs_, &fa->sg_, &fa->self_sg_, &fa->rank_, &B,
                    &S_local, &H_local,       &D};
    const int64_t rows = static_cast<int64_t>(B) * fa->world_size_ * S_local;
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(kMaxBlocks, rows));
    return cudaLaunchKernel(entry.kernel, dim3(blocks, 1, 1), dim3(kUlyssesThreads, 1, 1), args,
                            entry.dynamic_smem_bytes, stream);
  }
  return cudaErrorInvalidValue;
}
}  // namespace flashinfer::comm::ulysses
