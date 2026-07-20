// Ported from b12x b12x/distributed/pcie_dma.cu @ 7bfc9455 (2026-07-04) -- one-time curated port.
// PCIe ring allreduce transport primitives.
//
// Prefill-size allreduce is bandwidth-bound and the per-GPU x16 link is the
// invariant bottleneck (2*(N-1)/N * S egress per rank for any algorithm).
// On this fabric CE peer copies sustain ~56 GB/s while SM peer reads run at
// ~27 GB/s and SM peer writes at ~3 GB/s, so the data plane is CE
// (cudaMemcpyAsync peer copies) and the SM only synchronizes and reduces:
//
//   copy chunk -> peer scratch (CE, stream-ordered)
//   set_flag   -> peer flag    (tiny SM kernel, monotonic device counter)
//   wait_flag  -> local flag   (spin kernel, graph-replay safe)
//   add        -> accumulate received chunk into the working buffer
//
// Flag values come from device-resident monotonic counters so captured
// graphs replay without host-side value patching, mirroring the oneshot
// barrier's self_counter scheme.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      _message << cudaGetErrorString(e) << "\n"                         \
               << __FILE__ << ':' << __LINE__;                          \
      throw std::runtime_error(_message.str());                        \
    }                                                                   \
  } while (0)

namespace pcie_dma {

using FlagType = unsigned int;

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__global__ void set_flag_kernel(FlagType* peer_flag, FlagType* local_counter) {
  const FlagType value = *local_counter + 1;
  *local_counter = value;
  // The CE copy this flag publishes completed in stream order before this
  // kernel launched; the system fence orders any outstanding writes.
  __threadfence_system();
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(value), "l"(peer_flag));
}

__global__ void wait_flag_kernel(FlagType* flag, FlagType* expected_counter) {
  const FlagType expected = *expected_counter + 1;
  *expected_counter = expected;
  FlagType observed;
  do {
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(observed) : "l"(flag));
  } while (static_cast<int>(observed - expected) < 0);
}

// dst = a + b; with a == dst this is the in-place accumulate, and with
// a == the caller's input it doubles as the first-touch accumulate that
// removes the upfront out.copy_(input) from the critical path.
template <typename T>
__global__ void __launch_bounds__(256, 1) add_kernel(T* __restrict__ dst,
                                                     const T* __restrict__ a_src,
                                                     const T* __restrict__ b_src,
                                                     long long packs) {
  using Pack = uint4;
  Pack* dst_p = reinterpret_cast<Pack*>(dst);
  const Pack* a_p = reinterpret_cast<const Pack*>(a_src);
  const Pack* b_p = reinterpret_cast<const Pack*>(b_src);
  constexpr int kElems = sizeof(Pack) / sizeof(T);
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs;
       idx += gridDim.x * blockDim.x) {
    Pack a = a_p[idx];
    Pack b = b_p[idx];
    T* av = reinterpret_cast<T*>(&a);
    const T* bv = reinterpret_cast<const T*>(&b);
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      av[e] = from_float<T>(to_float(av[e]) + to_float(bv[e]));
    }
    dst_p[idx] = a;
  }
}

// ---- FP8 wire path (quantize-once all-to-all) ----
//
// E4M3 payload with one fp32 amax scale per 128 elements, scales packed
// contiguously after the payload so a single CE copy ships both. Values are
// quantized exactly once on the way in (reduce-scatter) and once on the way
// out (broadcast); accumulation runs in fp32, so wire precision drops while
// summation precision rises versus the bf16 ring's per-hop bf16 adds.

constexpr int kQuantBlock = 128;
constexpr float kFp8Max = 448.0f;

struct __align__(16) SrcPtrs {
  const void* ptrs[8];
};

__device__ __forceinline__ float4 load_bf16x4(const nv_bfloat16* src) {
  const auto* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
  const float2 lo = __bfloat1622float2(src2[0]);
  const float2 hi = __bfloat1622float2(src2[1]);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void store_bf16x4(nv_bfloat16* dst, const float4 v) {
  auto* dst2 = reinterpret_cast<__nv_bfloat162*>(dst);
  dst2[0] = __float22bfloat162_rn(make_float2(v.x, v.y));
  dst2[1] = __float22bfloat162_rn(make_float2(v.z, v.w));
}

__device__ __forceinline__ float4 load_fp8x4(const __nv_fp8_e4m3* src) {
  return static_cast<float4>(
      *reinterpret_cast<const __nv_fp8x4_e4m3*>(src));
}

__device__ __forceinline__ void store_fp8x4(
    __nv_fp8_e4m3* dst, const float4 v) {
  *reinterpret_cast<__nv_fp8x4_e4m3*>(dst) = __nv_fp8x4_e4m3(v);
}

// One warp per 128-element block: 4 elements per lane, warp amax, scale,
// convert, store.
__global__ void __launch_bounds__(256, 1) quant_kernel(
    const nv_bfloat16* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ payload,
    float* __restrict__ scales,
    long long blocks) {
  const int warps_per_cta = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  for (long long block = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
       block < blocks; block += static_cast<long long>(gridDim.x) * warps_per_cta) {
    const long long elem0 = block * kQuantBlock + lane * 4;
    const float4 v = load_bf16x4(src + elem0);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                       fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    const float scale = amax > 0.0f ? amax / kFp8Max : 1.0f;
    if (lane == 0) scales[block] = scale;
    const float inv_scale = 1.0f / scale;
    store_fp8x4(
        payload + elem0,
        make_float4(v.x * inv_scale, v.y * inv_scale,
                    v.z * inv_scale, v.w * inv_scale));
  }
}

// out = inp + sum_j dequant(payload_j) in fp32, single pass over out.
__global__ void __launch_bounds__(256, 1) dequant_accum_kernel(
    nv_bfloat16* __restrict__ out,
    const nv_bfloat16* __restrict__ inp,
    SrcPtrs payloads,
    SrcPtrs scales,
    int nsrc,
    long long blocks) {
  const int warps_per_cta = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  for (long long block = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
       block < blocks; block += static_cast<long long>(gridDim.x) * warps_per_cta) {
    const long long elem0 = block * kQuantBlock + lane * 4;
    float4 acc = load_bf16x4(inp + elem0);
    for (int j = 0; j < nsrc; ++j) {
      const __nv_fp8_e4m3* p =
          reinterpret_cast<const __nv_fp8_e4m3*>(payloads.ptrs[j]) + elem0;
      const float scale = reinterpret_cast<const float*>(scales.ptrs[j])[block];
      const float4 v = load_fp8x4(p);
      acc.x += v.x * scale;
      acc.y += v.y * scale;
      acc.z += v.z * scale;
      acc.w += v.w * scale;
    }
    store_bf16x4(out + elem0, acc);
  }
}

__global__ void __launch_bounds__(256, 1) dequant_store_kernel(
    nv_bfloat16* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ payload,
    const float* __restrict__ scales,
    long long blocks) {
  const int warps_per_cta = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  for (long long block = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
       block < blocks; block += static_cast<long long>(gridDim.x) * warps_per_cta) {
    const long long elem0 = block * kQuantBlock + lane * 4;
    const float scale = scales[block];
    const float4 v = load_fp8x4(payload + elem0);
    store_bf16x4(
        out + elem0,
        make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale));
  }
}

// Add this rank's BF16 contribution to an incoming FP8 partial and emit the
// next hop's FP8 partial in one pass.  Only the final reduce-scatter hop also
// materializes BF16; intermediate partials stay compressed between ranks.
template <bool StoreBf16>
__global__ void __launch_bounds__(256, 1) dequant_add_quant_kernel(
    nv_bfloat16* __restrict__ out,
    const nv_bfloat16* __restrict__ local,
    const __nv_fp8_e4m3* __restrict__ payload_in,
    const float* __restrict__ scales_in,
    __nv_fp8_e4m3* __restrict__ payload_out,
    float* __restrict__ scales_out,
    long long blocks) {
  const int warps_per_cta = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  for (long long block = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
       block < blocks; block += static_cast<long long>(gridDim.x) * warps_per_cta) {
    const long long elem0 = block * kQuantBlock + lane * 4;
    const float scale_in = scales_in[block];
    const float4 a = load_bf16x4(local + elem0);
    const float4 b = load_fp8x4(payload_in + elem0);
    const float4 v = make_float4(
        a.x + b.x * scale_in, a.y + b.y * scale_in,
        a.z + b.z * scale_in, a.w + b.w * scale_in);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                       fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    const float scale_out = amax > 0.0f ? amax / kFp8Max : 1.0f;
    if (lane == 0) scales_out[block] = scale_out;
    const float inv_scale = 1.0f / scale_out;
    store_fp8x4(
        payload_out + elem0,
        make_float4(v.x * inv_scale, v.y * inv_scale,
                    v.z * inv_scale, v.w * inv_scale));
    if constexpr (StoreBf16) store_bf16x4(out + elem0, v);
  }
}

}  // namespace pcie_dma

static void dma_copy(int64_t dst_ptr, int64_t src_ptr, int64_t bytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDA_SUCCESS(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr),
                                     reinterpret_cast<const void*>(src_ptr),
                                     static_cast<size_t>(bytes),
                                     cudaMemcpyDeviceToDevice, stream));
}

static void dma_set_flag(int64_t peer_flag_ptr, int64_t counter_ptr) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pcie_dma::set_flag_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<pcie_dma::FlagType*>(peer_flag_ptr),
      reinterpret_cast<pcie_dma::FlagType*>(counter_ptr));
}

static void dma_wait_flag(int64_t flag_ptr, int64_t counter_ptr) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pcie_dma::wait_flag_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<pcie_dma::FlagType*>(flag_ptr),
      reinterpret_cast<pcie_dma::FlagType*>(counter_ptr));
}

static void dma_add(int64_t dst_ptr, int64_t a_ptr, int64_t b_ptr,
                    int64_t elems, int64_t dtype_code) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const int threads = 256;
  const long long packs16 = elems * (dtype_code == 2 ? 4 : 2) / 16;
  const int blocks = static_cast<int>(
      std::max<long long>(1, std::min<long long>(64, (packs16 + threads - 1) / threads)));
  if (dtype_code == 0) {
    const long long packs = elems / 8;
    pcie_dma::add_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(dst_ptr),
        reinterpret_cast<const nv_bfloat16*>(a_ptr),
        reinterpret_cast<const nv_bfloat16*>(b_ptr), packs);
  } else if (dtype_code == 1) {
    const long long packs = elems / 8;
    pcie_dma::add_kernel<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(dst_ptr), reinterpret_cast<const half*>(a_ptr),
        reinterpret_cast<const half*>(b_ptr), packs);
  } else {
    const long long packs = elems / 4;
    pcie_dma::add_kernel<float><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float*>(dst_ptr), reinterpret_cast<const float*>(a_ptr),
        reinterpret_cast<const float*>(b_ptr), packs);
  }
}

static int _fp8_blocks_grid(long long blocks, int threads) {
  const int warps = threads / 32;
  return static_cast<int>(std::max<long long>(
      1, std::min<long long>(64, (blocks + warps - 1) / warps)));
}

static void dma_quant(int64_t src_ptr, int64_t payload_ptr, int64_t scales_ptr,
                      int64_t elems) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const long long blocks = elems / pcie_dma::kQuantBlock;
  const int threads = 256;
  pcie_dma::quant_kernel<<<_fp8_blocks_grid(blocks, threads), threads, 0, stream>>>(
      reinterpret_cast<const nv_bfloat16*>(src_ptr),
      reinterpret_cast<__nv_fp8_e4m3*>(payload_ptr),
      reinterpret_cast<float*>(scales_ptr), blocks);
}

static void dma_dequant_accum(int64_t out_ptr, int64_t inp_ptr,
                              const std::vector<int64_t>& payload_ptrs,
                              const std::vector<int64_t>& scale_ptrs,
                              int64_t elems) {
  TORCH_CHECK(payload_ptrs.size() == scale_ptrs.size());
  TORCH_CHECK(payload_ptrs.size() <= 8);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pcie_dma::SrcPtrs payloads{};
  pcie_dma::SrcPtrs scales{};
  for (size_t j = 0; j < payload_ptrs.size(); ++j) {
    payloads.ptrs[j] = reinterpret_cast<const void*>(payload_ptrs[j]);
    scales.ptrs[j] = reinterpret_cast<const void*>(scale_ptrs[j]);
  }
  const long long blocks = elems / pcie_dma::kQuantBlock;
  const int threads = 256;
  pcie_dma::dequant_accum_kernel<<<_fp8_blocks_grid(blocks, threads), threads, 0,
                                   stream>>>(
      reinterpret_cast<nv_bfloat16*>(out_ptr),
      reinterpret_cast<const nv_bfloat16*>(inp_ptr), payloads, scales,
      static_cast<int>(payload_ptrs.size()), blocks);
}

static void dma_dequant_store(int64_t out_ptr, int64_t payload_ptr,
                              int64_t scales_ptr, int64_t elems) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const long long blocks = elems / pcie_dma::kQuantBlock;
  const int threads = 256;
  pcie_dma::dequant_store_kernel<<<_fp8_blocks_grid(blocks, threads), threads, 0,
                                   stream>>>(
      reinterpret_cast<nv_bfloat16*>(out_ptr),
      reinterpret_cast<const __nv_fp8_e4m3*>(payload_ptr),
      reinterpret_cast<const float*>(scales_ptr), blocks);
}

static void dma_dequant_add_quant(
    int64_t out_ptr, int64_t local_ptr, int64_t payload_in_ptr,
    int64_t scales_in_ptr, int64_t payload_out_ptr, int64_t scales_out_ptr,
    int64_t elems, bool store_bf16) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const long long blocks = elems / pcie_dma::kQuantBlock;
  const int threads = 256;
  const int grid = _fp8_blocks_grid(blocks, threads);
#define LAUNCH_DEQUANT_ADD_QUANT(STORE)                                      \
  pcie_dma::dequant_add_quant_kernel<STORE><<<grid, threads, 0, stream>>>(  \
      reinterpret_cast<nv_bfloat16*>(out_ptr),                              \
      reinterpret_cast<const nv_bfloat16*>(local_ptr),                      \
      reinterpret_cast<const __nv_fp8_e4m3*>(payload_in_ptr),               \
      reinterpret_cast<const float*>(scales_in_ptr),                        \
      reinterpret_cast<__nv_fp8_e4m3*>(payload_out_ptr),                    \
      reinterpret_cast<float*>(scales_out_ptr), blocks)
  if (store_bf16) {
    LAUNCH_DEQUANT_ADD_QUANT(true);
  } else {
    LAUNCH_DEQUANT_ADD_QUANT(false);
  }
#undef LAUNCH_DEQUANT_ADD_QUANT
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dma_quant", &dma_quant, "quantize bf16 to e4m3 with per-128 scales");
  m.def("dma_dequant_accum", &dma_dequant_accum,
        "out = inp + sum of dequantized sources (fp32 accumulate)");
  m.def("dma_dequant_store", &dma_dequant_store, "dequantize e4m3 to bf16");
  m.def("dma_dequant_add_quant", &dma_dequant_add_quant,
        "add a BF16 contribution to an FP8 partial and requantize");
  m.def("dma_copy", &dma_copy, "CE peer copy on the current stream");
  m.def("dma_set_flag", &dma_set_flag, "publish a monotonic flag to a peer");
  m.def("dma_wait_flag", &dma_wait_flag, "wait for a monotonic peer flag");
  m.def("dma_add", &dma_add, "elementwise add src into dst");
}
