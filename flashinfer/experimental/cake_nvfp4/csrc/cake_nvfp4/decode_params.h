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
#pragma once
// Parameter block of the decode kernels of the Nemotron 3.5 Super NVFP4-KV backend: "mtp" and
// "base" take the same block, by value.
#include <cuda.h>

#include <cstdint>

#include "host.h"

namespace flashinfer {
namespace cake_nvfp4 {

constexpr int kPageSizeLog2 = 7;
static_assert((1 << kPageSizeLog2) == kPageSize, "page size");

struct DecodeParams {
  CUtensorMap tmaQ;  // u8 [query row][kvHead][16 heads][128]: box 128 x 16 x 1 x G, 128B swizzle
  CUtensorMap tmaK;  // u8 [page][kvHead][128 tok][64 B packed FP4]: box 64 x 128, 64B swizzle
  CUtensorMap tmaV;  // 16U4_ALIGN16B [page][kvHead][128 tok][128 FP4]: box 128 x 128, 128B swizzle
  CUtensorMap tmaKSf;  // u8 [page][kvHead][8][128 B]: box 128 x 8
  CUtensorMap tmaVSf;
  uint8_t* out;  // [requests * queryTokens][32][128] E4M3
  float* ws;     // piece O | piece (max, sum) | unit piece ranges | per-CTA KV segment lists
  int32_t const* pageTable;  // [requests][maxPagesPerSeq]
  int32_t const* seqLens;    // [requests]
  int64_t wsStatsOffset;     // in floats
  int64_t wsRangeOffset;  // in floats: int2 {first range, last range} per (virtual request, kvHead)
  int64_t wsSegsOffset;   // in floats (16-byte aligned)
  int32_t maxSegsPerCta;
  int32_t numVirt;  // virtual requests: requests * numGroups; each fuses G consecutive query tokens
  int32_t numGroups;    // queryTokens / G
  int32_t queryTokens;  // T: query tokens per request (MTP + 1)
  int32_t maxPagesPerSeq;
  float scaleSoftmaxLog2;
  float outputScale;
  float const* scaleSoftmaxLog2Ptr;
  float const* outputScalePtr;
  int32_t const* cumSeqLensQ;  // optional packed/ragged query offsets
  // 0: the combine kernel starts after the main kernel has finished; 1: it overlaps the main
  // kernel (programmatic dependent launch) and waits where it first reads the workspace.
  int32_t pdlMode;
};

}  // namespace cake_nvfp4
}  // namespace flashinfer
