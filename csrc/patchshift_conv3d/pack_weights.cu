/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#include <algorithm>
#include <flashinfer/conv3d/patchshift/weight_layout.cuh>

#include "launcher.cuh"

namespace flashinfer::conv3d::patchshift::host {
namespace {

template <int TileM>
__device__ __forceinline__ void StorePackedWeight(Element value, Element* packed, int kout, int td,
                                                  int kh, int kw, int c, int c32_groups) {
  int k_tile = kout / TileM;
  int m = kout - k_tile * TileM;
  int c32_group = c / 32;
  int within_c32 = c - c32_group * 32;
  int kg = within_c32 / 16;
  int kk = within_c32 - kg * 16;
  int supergroup = td * c32_groups + c32_group;
  size_t offset =
      patchshift::PackedWeightOffset(k_tile, supergroup, kh, kw, kg, m, kk, 3 * c32_groups, TileM);
  packed[offset] = value;
}

__global__ void PackWeightsKernel(const Element* __restrict__ weight,
                                  Element* __restrict__ packed_m128,
                                  Element* __restrict__ packed_m64,
                                  Element* __restrict__ packed_m32, int c_size, int k_size,
                                  int64_t stride_k, int64_t stride_c, int64_t stride_t,
                                  int64_t stride_r, int64_t stride_s) {
  size_t total = size_t(k_size) * c_size * 3 * 3 * 3;
  int c32_groups = patchshift::C32Groups(c_size);
  for (size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x; linear < total;
       linear += size_t(blockDim.x) * gridDim.x) {
    size_t index = linear;
    int s = int(index % 3);
    index /= 3;
    int r = int(index % 3);
    index /= 3;
    int t = int(index % 3);
    index /= 3;
    int c = int(index % size_t(c_size));
    int k = int(index / size_t(c_size));
    size_t source = size_t(int64_t(k) * stride_k + int64_t(c) * stride_c + int64_t(t) * stride_t +
                           int64_t(r) * stride_r + int64_t(s) * stride_s);
    Element value = weight[source];
    StorePackedWeight<128>(value, packed_m128, k, t, r, s, c, c32_groups);
    StorePackedWeight<64>(value, packed_m64, k, t, r, s, c, c32_groups);
    StorePackedWeight<32>(value, packed_m32, k, t, r, s, c, c32_groups);
  }
}

}  // namespace

Status PackWeights(const Element* weight, Element* packed_m128, Element* packed_m64,
                   Element* packed_m32, const Conv3dProblem& problem, int64_t stride_k,
                   int64_t stride_c, int64_t stride_t, int64_t stride_r, int64_t stride_s,
                   cudaStream_t stream) {
  cudaError_t error = cudaMemsetAsync(
      packed_m128, 0, patchshift::PackedWeightNumel(problem.c, problem.k, 128) * sizeof(Element),
      stream);
  if (error != cudaSuccess) return Status::Cuda(error);
  error = cudaMemsetAsync(packed_m64, 0,
                          patchshift::PackedWeightNumel(problem.c, problem.k, 64) * sizeof(Element),
                          stream);
  if (error != cudaSuccess) return Status::Cuda(error);
  error = cudaMemsetAsync(packed_m32, 0,
                          patchshift::PackedWeightNumel(problem.c, problem.k, 32) * sizeof(Element),
                          stream);
  if (error != cudaSuccess) return Status::Cuda(error);

  size_t total = size_t(problem.k) * problem.c * 3 * 3 * 3;
  int blocks = int(std::min<size_t>((total + 255) / 256, 65535));
  PackWeightsKernel<<<blocks, 256, 0, stream>>>(weight, packed_m128, packed_m64, packed_m32,
                                                problem.c, problem.k, stride_k, stride_c, stride_t,
                                                stride_r, stride_s);
  return Status::Cuda(cudaGetLastError());
}

}  // namespace flashinfer::conv3d::patchshift::host
