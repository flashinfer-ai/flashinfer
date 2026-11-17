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
// Nemotron 3.5 Super NVFP4-KV decode, "base" kernel: 1, 3 or 5 query tokens per request, or
// requests of different query lengths.
#include "flashinfer/attention/nm35_nvfp4/decode_base_kernel.cuh"
#include "flashinfer/attention/nm35_nvfp4/host_cuda.cuh"

namespace flashinfer {
namespace nm35_nvfp4 {

namespace base = decode_base;

void launchDecodeBase(Call const& c, DecodePlan const& plan, void* streamPtr) {
  auto const stream = static_cast<cudaStream_t>(streamPtr);
  DecodeParams const p = makeDecodeParams(c, plan);
  dim3 const mainGrid(plan.ctas);
  switch (plan.g) {
    case 1:
      launchKernel(base::nm35_nvfp4_decode_base_g1, mainGrid, base::Cfg<1>::Threads,
                   base::Cfg<1>::SmemBytes, stream, false, p);
      break;
    case 3:
      launchKernel(base::nm35_nvfp4_decode_base_g3, mainGrid, base::Cfg<3>::Threads,
                   base::Cfg<3>::SmemBytes, stream, false, p);
      break;
    default:
      launchKernel(base::nm35_nvfp4_decode_base_g5, mainGrid, base::Cfg<5>::Threads,
                   base::Cfg<5>::SmemBytes, stream, false, p);
      break;
  }
  // Combine grid: one CTA per (virtual request, KV head) x (query token, query head of the group).
  dim3 const combineGrid(plan.virt * kNumHeadsKv, plan.g * kHeadsPerKv);
  launchKernel(base::nm35_nvfp4_decode_base_combine, combineGrid, base::kCombineThreads, 0, stream,
               p.pdlMode != 0, p);
}

}  // namespace nm35_nvfp4
}  // namespace flashinfer
