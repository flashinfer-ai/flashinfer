// Test-only delays and observations around the production STSM -> shared -> global path.
#include <tvm/ffi/container/array.h>

#include <array>
#include <flashinfer/attention/hopper/epilogue.cuh>
#include <flashinfer/attention/hopper/kernel_traits.cuh>

using namespace cute;
struct ProbeVariant {};
using Traits = flashinfer::AttentionKernelTraits<true, 256, 256, 128, 64, 2, cutlass::bfloat16_t,
                                                 cutlass::bfloat16_t, cutlass::bfloat16_t, int32_t,
                                                 ProbeVariant>;
using Epilogue = flashinfer::CollectiveEpilogue<Traits>;
static_assert(Traits::NUM_THREADS == 384 && Traits::NUM_MMA_THREADS == 256);

struct ProbeContext {
  uint64_t* events;
  unsigned* done;
  int consumer;

  __device__ void before_stsm() const {
    if ((consumer & 31) == 0) events[consumer / 32] = clock64();
    if (consumer < 128) {
      uint64_t start = clock64();
      start = __shfl_sync(0xffffffff, start, 0);
      uint64_t now;
      do {
        now = clock64();
        now = __shfl_sync(0xffffffff, now, 0);
      } while (now - start < (1ULL << 22));
    }
    if ((consumer & 31) == 0) events[8 + consumer / 32] = clock64();
    asm volatile("" ::: "memory");
  }

  __device__ void after_stsm() const {
    if ((consumer & 31) == 0) atomicExch(done + consumer / 32, 1u);
  }

  template <typename T>
  __device__ void before_global(T const* address) const {
    if (consumer == 190) {
      unsigned mask = 0;
      for (int warp = 0; warp < 4; ++warp) mask |= (atomicAdd(done + warp, 0u) != 0) << warp;
      events[19] = mask;
      events[16] = clock64();
      unsigned bits, pointer = static_cast<unsigned>(__cvta_generic_to_shared(address));
      asm volatile("ld.shared.u16 %0, [%1];" : "=r"(bits) : "r"(pointer) : "memory");
      events[18] = bits;
      events[17] = clock64();
    }
  }
};

// Generated from the current production class with common hooks only; every
// production barrier is retained, including a reintroduced bug.
#include "epilogue_test_impl.cuh"

// Keep the FFI Tensor alias out of the CuTe template's unqualified Tensor lookup.
#include "tvm_ffi_utils.h"

__global__ __launch_bounds__(384) void probe_kernel(cutlass::bfloat16_t* output, uint64_t* events) {
  extern __shared__ __align__(128) unsigned char bytes[];
  auto& storage = *reinterpret_cast<Traits::SharedStorage*>(bytes);
  auto* done = reinterpret_cast<unsigned*>(bytes + sizeof(Traits::SharedStorage));
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 21; ++i) events[i] = 0;
    for (int i = 0; i < 8; ++i) done[i] = 0;
  }
  for (int i = tid; i < 128 * 8 * 256; i += 384) reinterpret_cast<uint16_t*>(output)[i] = 0x4040;
  for (int i = tid; i < cosize(typename Epilogue::SmemLayoutO{}); i += 384)
    reinterpret_cast<uint16_t*>(storage.smem_o.data())[i] = 0xbf80;
  __syncthreads();
  if (tid >= 128) {
    Traits::TiledMmaPV mma;
    auto accum = partition_fragment_C(mma, select<0, 1>(Traits::TileShape_PDV{}));
    fill(accum, 2.0f);
    auto lse = make_tensor<float>(make_shape(Int<2>{}));
    clear(lse);
    using TestEpilogue = flashinfer::TestEpilogue<Traits>;
    TestEpilogue::Params params{output, flashinfer::get_gmem_layout(128, 8, 256, 2048, 256),
                                nullptr, flashinfer::get_lse_gmem_layout(128, 8)};
    auto coord = make_tuple(0, 6, 1, 0, 0, 128, 128, 0);
    TestEpilogue{}.store(params, accum, lse, storage, mma, tid - 128, coord,
                         ProbeContext{events, done, tid - 128});
  }
  __syncthreads();
  if (tid == 0) events[20] = 1;
}

// Compute the writer/reader from the actual CuTe partitions rather than a guessed
// row-to-warp mapping. STSM ownership and shared-to-global ownership are different.
tvm::ffi::Array<int64_t> mapping() {
  Traits::TiledMmaPV mma;
  Epilogue::TiledCopyO copy;
  auto identity = make_identity_tensor(make_shape(Int<128>{}, Int<256>{}));
  int owner = -1, reader = -1;
  for (int thread = 0; thread < 256; ++thread) {
    auto writes = mma.get_thread_slice(thread).partition_C(identity);
    auto reads = copy.get_slice(thread).partition_D(identity);
    for (int i = 0; i < size(writes); ++i) {
      if (get<0>(writes(i)) == 5 && get<1>(writes(i)) == 240) {
        TVM_FFI_ICHECK_EQ(owner, -1);
        owner = thread;
      }
    }
    for (int i = 0; i < size(reads); ++i) {
      if (get<0>(reads(i)) == 5 && get<1>(reads(i)) == 240) {
        TVM_FFI_ICHECK_EQ(reader, -1);
        reader = thread;
      }
    }
  }
  alignas(1024) std::array<cutlass::bfloat16_t, 128 * 256> memory;
  auto shared = make_tensor(make_smem_ptr(memory.data()), Epilogue::SmemLayoutO{});
  return {owner, reader, &shared(5, 240) - memory.data()};
}

void run_probe(tvm::ffi::TensorView output, tvm::ffi::TensorView events) {
  CHECK_INPUT(output);
  CHECK_INPUT(events);
  CHECK_DEVICE(output, events);
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == 128 && output.size(1) == 8 &&
                 output.size(2) == 256 && output.dtype().code == kDLBfloat &&
                 output.dtype().bits == 16);
  TVM_FFI_ICHECK(events.ndim() == 1 && events.size(0) == 21 && events.dtype().code == kDLInt &&
                 events.dtype().bits == 64);
  tvm::ffi::CUDADeviceGuard device_guard(output.device().device_id);
  constexpr int shared_bytes = sizeof(Traits::SharedStorage) + 8 * sizeof(unsigned);
  auto status =
      cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  probe_kernel<<<1, 384, shared_bytes, get_stream(output.device())>>>(
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      static_cast<uint64_t*>(events.data_ptr()));
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mapping, mapping);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_probe, run_probe);
