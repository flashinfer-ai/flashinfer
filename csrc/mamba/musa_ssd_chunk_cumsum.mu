// Native MUSA SSD chunk-cumsum kernel.
//
// This is the shape-specialized implementation from the FlashInfer MUSA
// dashboard problem ``flashinfer_musa_ssd_chunk_cumsum_h64_c128``.  A block
// owns one chunk and a small group of heads.  Four-token vector loads keep
// the input and output accesses aligned; a warp scan computes the exclusive
// prefix of A * dt for each head.

#include <torch/all.h>
#include <torch/extension.h>
#include "torch_musa/csrc/core/MUSAStream.h"

#include <cstdint>

namespace {
constexpr int kHeads = 64;
constexpr int kChunk = 128;
constexpr int kPitch = 132;

template <int HPB, int NTHREAD>
__global__ __launch_bounds__(NTHREAD) void chunk_cumsum_kernel(
    const float* __restrict__ dt, const float* __restrict__ A,
    const float* __restrict__ bias, float* __restrict__ dA_out,
    float* __restrict__ dt_out) {
  constexpr int kVIter = (kChunk * HPB) / (NTHREAD * 4);
  constexpr int kTStride = (NTHREAD * 4) / HPB;
  constexpr int kWarps = NTHREAD / 32;
  constexpr int kHeadsPerWarp = HPB / kWarps;

  __shared__ float sm[HPB * kPitch];
  const int tid = threadIdx.x;
  const int chunk = blockIdx.x;
  const int head_group = blockIdx.y;
  const int h0 = (tid * 4) & (HPB - 1);
  const int t0 = tid / (HPB / 4);
  const float4 b4 = *reinterpret_cast<const float4*>(bias + head_group * HPB + h0);
  const float* src = dt + static_cast<size_t>(chunk) * kChunk * kHeads +
                     head_group * HPB + h0;

  float4 values[kVIter];
#pragma unroll
  for (int i = 0; i < kVIter; ++i) {
    values[i] = *reinterpret_cast<const float4*>(
        src + static_cast<size_t>(t0 + i * kTStride) * kHeads);
  }
#pragma unroll
  for (int i = 0; i < kVIter; ++i) {
    const int t = t0 + i * kTStride;
    const float x0 = values[i].x + b4.x;
    const float x1 = values[i].y + b4.y;
    const float x2 = values[i].z + b4.z;
    const float x3 = values[i].w + b4.w;
    sm[(h0 + 0) * kPitch + t] =
        x0 > 20.0f ? x0 : __logf(1.0f + __expf(x0));
    sm[(h0 + 1) * kPitch + t] =
        x1 > 20.0f ? x1 : __logf(1.0f + __expf(x1));
    sm[(h0 + 2) * kPitch + t] =
        x2 > 20.0f ? x2 : __logf(1.0f + __expf(x2));
    sm[(h0 + 3) * kPitch + t] =
        x3 > 20.0f ? x3 : __logf(1.0f + __expf(x3));
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int k = 0; k < kHeadsPerWarp; ++k) {
    const int h = warp + k * kWarps;
    const int global_head = head_group * HPB + h;
    const size_t out_base =
        (static_cast<size_t>(global_head) * gridDim.x + chunk) * kChunk;
    float* dA_chunk = dA_out + out_base;
    float* dt_chunk = dt_out + out_base;
    const float a = A[head_group * HPB + h];
    const float4 s = *reinterpret_cast<const float4*>(
        &sm[h * kPitch + lane * 4]);
    *reinterpret_cast<float4*>(&dt_chunk[lane * 4]) = s;

    const float p0 = s.x * a;
    const float p1 = p0 + s.y * a;
    const float p2 = p1 + s.z * a;
    const float p3 = p2 + s.w * a;
    float running = p3;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float neighbor = __shfl_up_sync(0xffffffffu, running, offset);
      if (lane >= offset) running += neighbor;
    }
    const float exclusive = running - p3;
    float4 prefix;
    prefix.x = p0 + exclusive;
    prefix.y = p1 + exclusive;
    prefix.z = p2 + exclusive;
    prefix.w = p3 + exclusive;
    *reinterpret_cast<float4*>(&dA_chunk[lane * 4]) = prefix;
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> musa_ssd_chunk_cumsum(
    const torch::Tensor& dt, const torch::Tensor& A,
    const torch::Tensor& bias) {
  TORCH_CHECK(dt.dim() == 2 && dt.size(1) == kHeads &&
                  dt.scalar_type() == torch::kFloat,
              "SSD cumsum dt must be [tokens,64] float32");
  TORCH_CHECK(A.device() == dt.device() && bias.device() == dt.device(),
              "SSD cumsum inputs must be on the same MUSA device");
  TORCH_CHECK(A.dim() == 1 && A.size(0) == kHeads &&
                  A.scalar_type() == torch::kFloat && A.is_contiguous(),
              "SSD cumsum A must be contiguous float32 [64]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == kHeads &&
                  bias.scalar_type() == torch::kFloat && bias.is_contiguous(),
              "SSD cumsum bias must be contiguous float32 [64]");
  TORCH_CHECK(dt.is_contiguous() && dt.size(0) % kChunk == 0,
              "SSD cumsum dt must be contiguous and have full 128-token chunks");

  const int nchunks = static_cast<int>(dt.size(0) / kChunk);
  auto dA_out = torch::empty({kHeads, nchunks, kChunk}, dt.options());
  auto dt_out = torch::empty({kHeads, nchunks, kChunk}, dt.options());
  float* dA_ptr = dA_out.data_ptr<float>();
  float* dt_ptr = dt_out.data_ptr<float>();
  const auto stream = c10::musa::getCurrentMUSAStream();
  if (nchunks <= 8) {
    chunk_cumsum_kernel<16, 128><<<dim3(nchunks, kHeads / 16), 128, 0,
                                    stream>>>(
        dt.data_ptr<float>(), A.data_ptr<float>(), bias.data_ptr<float>(),
        dA_ptr, dt_ptr);
  } else {
    chunk_cumsum_kernel<64, 256><<<dim3(nchunks, 1), 256, 0, stream>>>(
        dt.data_ptr<float>(), A.data_ptr<float>(), bias.data_ptr<float>(),
        dA_ptr, dt_ptr);
  }
  TORCH_CHECK(musaGetLastError() == musaSuccess,
              "SSD cumsum native launch failed");
  return std::make_tuple(dA_out, dt_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("musa_ssd_chunk_cumsum", &musa_ssd_chunk_cumsum,
        "Native MUSA SSD chunk cumsum");
}
