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
// Nemotron 3.5 Super NVFP4-KV prefill. A conversion kernel first dequantizes every visible NVFP4
// page to FP8 in the workspace; FP8 attention kernels then run on that compact cache.
#include <cuda.h>
#include <cuda_runtime.h>

#include "flashinfer/attention/cake_nvfp4/host_cuda.cuh"
#include "flashinfer/attention/cake_nvfp4/prefill/attn.cuh"
#include "flashinfer/attention/cake_nvfp4/prefill/attnp.cuh"
#include "flashinfer/attention/cake_nvfp4/prefill/comb.cuh"
#include "flashinfer/attention/cake_nvfp4/prefill/conv.cuh"
#include "flashinfer/attention/cake_nvfp4/prefill/pmax.cuh"

namespace flashinfer {
namespace cake_nvfp4 {

namespace {

namespace conv = prefill::conv;
namespace plain = prefill::attnp;
namespace split = prefill::attn;
namespace rowmax = prefill::pmax;
namespace combine = prefill::comb;

CUtensorMap byteMap(void const* ptr, std::array<uint64_t, 4> shape, std::array<uint64_t, 3> strides,
                    std::array<uint32_t, 4> box) {
  return tensorMap(ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, shape, strides, box,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
}

}  // namespace

void launchPrefill(Call const& c, Device const& dev, void* streamPtr) {
  auto const stream = static_cast<cudaStream_t>(streamPtr);
  // The gate admits one request, so its pages are the whole compact cache.
  PrefillSchedule const schedule = makePrefillSchedule(c.maxQ, dev.smCount);
  int const qTiles = (c.maxQ + kPrefillQTile - 1) / kPrefillQTile;
  int const cachePages = int(prefillCachePages(c.maxKv));

  // Workspace: FP8 K pages | FP8 V pages | split partial O | split partial (max, sum) | row max.
  auto* const k8 = static_cast<uint8_t*>(c.workspace);
  auto* const v8 = k8 + int64_t(cachePages) * kPrefillFp8PageBytes;
  auto* const partO = reinterpret_cast<float*>(v8 + int64_t(cachePages) * kPrefillFp8PageBytes);
  auto* const partMl = partO + int64_t(2) * schedule.nrem * kPrefillQTile * kHeadDim;
  auto* const rowMax = partMl + int64_t(2) * schedule.nrem * kPrefillQTile * 2;

  auto* const pageTable = const_cast<int*>(c.pageTable);
  auto* const kvLengths = const_cast<int*>(c.seqLensKv);
  auto* const qIndptr = const_cast<int*>(c.cumSeqLensQ);
  auto asWords = [](void const* ptr) { return static_cast<int*>(const_cast<void*>(ptr)); };

  // Encode the descriptors before the first launch: encoding is the step that can fail, and a
  // failure must leave the workspace untouched.
  uint64_t const rows = c.sumQ;
  uint64_t const pages = cachePages;
  CUtensorMap const qMap = byteMap(c.q, {128, 1, 32, rows}, {128, 128, 4096}, {128, 1, 1, 128});
  CUtensorMap const kMap = byteMap(k8, {128, 128, 2, pages}, {128, 16384, 32768}, {128, 128, 1, 1});
  CUtensorMap const vMap = byteMap(v8, {128, 128, 2, pages}, {128, 16384, 32768}, {128, 128, 1, 1});

  // The conversion kernel reads 32-bit words: strides are in words.
  launchKernel(conv::cake_nvfp4_prefill_conv, dim3(unsigned(cachePages) * kNumHeadsKv),
               conv::kThreads, 0, stream, false, asWords(c.k), asWords(c.v), asWords(c.kSf),
               asWords(c.vSf), k8, v8, int(c.kStridePage / 4), int(c.kStrideHead / 4),
               int(c.kSfStridePage / 4), int(c.kSfStrideHead / 4), c.pageTable, c.seqLensKv,
               int(c.poolPages), cachePages, int(c.maxPagesPerSeq));

  auto* const out = static_cast<uint8_t*>(c.out);
  // The attention kernels index the compact cache: the table stride is its pages per sequence.
  int const tableStride = cachePages;

  int const plainTiles = schedule.plainTiles(qTiles * kNumHeadsQ);
  if (plainTiles > 0) {
    prefill::TensorMaps<3> const maps{{qMap, kMap, vMap}};
    launchKernel(plain::cake_nvfp4_prefill_attnp, dim3(plainTiles), plain::kThreads,
                 plain::kSharedBytes, stream, false, out, pageTable, qIndptr, kvLengths,
                 tableStride, qTiles, /*tile_start=*/0, c.scaleSoftmaxLog2, c.outputScale,
                 c.scaleSoftmaxLog2Ptr, c.outputScalePtr, maps);
  }
  if (schedule.nrem > 0) {
    prefill::TensorMaps<2> const maxMaps{{qMap, kMap}};
    launchKernel(rowmax::cake_nvfp4_prefill_pmax, dim3(schedule.nrem * schedule.pchunks),
                 rowmax::kThreads, rowmax::kSharedBytes, stream, false, rowMax, pageTable, qIndptr,
                 kvLengths, tableStride, qTiles, schedule.nctas, schedule.qf, schedule.tileBase,
                 schedule.xFrac, schedule.pchunks, maxMaps);
    prefill::TensorMaps<3> const maps{{qMap, kMap, vMap}};
    launchKernel(split::cake_nvfp4_prefill_attn, dim3(schedule.nctas), split::kThreads,
                 split::kSharedBytes, stream, false, out, partO, partMl, rowMax, schedule.pchunks,
                 pageTable, qIndptr, kvLengths, tableStride, qTiles, schedule.nctas, schedule.qf,
                 schedule.tileBase, schedule.nrem, schedule.xFrac, c.scaleSoftmaxLog2,
                 c.outputScale, c.scaleSoftmaxLog2Ptr, c.outputScalePtr, maps);
    launchKernel(combine::cake_nvfp4_prefill_comb, dim3(schedule.nrem), combine::kThreads, 0,
                 stream, false, out, partO, partMl, qIndptr, qTiles, schedule.nctas, schedule.qf,
                 schedule.tileBase, c.scaleSoftmaxLog2, c.outputScale, c.scaleSoftmaxLog2Ptr,
                 c.outputScalePtr);
  }
}

}  // namespace cake_nvfp4
}  // namespace flashinfer
