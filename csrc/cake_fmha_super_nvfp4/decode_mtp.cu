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
// Nemotron 3.5 Super NVFP4-KV decode, "mtp" kernel: 2, 4, 6 or 8 query tokens per request.
#include "decode_mtp_kernel.cuh"
#include "host_cuda.cuh"

namespace flashinfer {
namespace super_nvfp4 {

namespace mtp = decode_mtp;

void launchDecodeMtp(Call const& c, DecodePlan const& plan, void* streamPtr) {
  auto const stream = static_cast<cudaStream_t>(streamPtr);
  DecodeParams const p = makeDecodeParams(c, plan);
  dim3 const mainGrid(plan.ctas);
  switch (plan.g) {
    case 2:
      launchKernel(mtp::super_nvfp4_decode_mtp_g2, mainGrid, mtp::Cfg<2>::Threads,
                   mtp::Cfg<2>::SmemBytes, stream, false, p);
      break;
    case 4:
      launchKernel(mtp::super_nvfp4_decode_mtp_g4, mainGrid, mtp::Cfg<4>::Threads,
                   mtp::Cfg<4>::SmemBytes, stream, false, p);
      break;
    default:
      launchKernel(mtp::super_nvfp4_decode_mtp_g8, mainGrid, mtp::Cfg<8>::Threads,
                   mtp::Cfg<8>::SmemBytes, stream, false, p);
      break;
  }
  // A request split across CTAs leaves one partial result per piece. Pick the combine variant
  // by the expected pieces per request: one CTA per output column with 10, 5 or 3 loads per
  // warp, or (few pieces) one warp per column.
  int const pieces = int((plan.ctas / 2 + plan.virt - 1) / plan.virt) + 1;
  bool const overlap = p.pdlMode != 0;
  dim3 const columnGrid(plan.virt * kNumHeadsKv, plan.g * kHeadsPerKv);
  if (pieces > 40) {
    launchKernel(mtp::super_nvfp4_decode_mtp_combine_c10, columnGrid, mtp::kCombineThreads, 0,
                 stream, overlap, p);
  } else if (pieces > 24) {
    launchKernel(mtp::super_nvfp4_decode_mtp_combine_c5, columnGrid, mtp::kCombineThreads, 0,
                 stream, overlap, p);
  } else if (pieces > 12) {
    launchKernel(mtp::super_nvfp4_decode_mtp_combine_c3, columnGrid, mtp::kCombineThreads, 0,
                 stream, overlap, p);
  } else {
    dim3 const warpGrid(plan.virt * kNumHeadsKv, plan.g * kHeadsPerKv / 8);
    launchKernel(mtp::super_nvfp4_decode_mtp_combine_w12, warpGrid, mtp::kCombineThreads, 0, stream,
                 overlap, p);
  }
}

}  // namespace super_nvfp4
}  // namespace flashinfer
