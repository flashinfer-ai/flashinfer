/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// .16dp32bitx2.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumColsBetweenHalfWarps>
inline __device__ int ldst16dp32bitTmemColIdx(int tidx) {
  // The base is (tidx % 32) / 16) * NumColsBetweenHalfWarps.
  return ((tidx & 0x1f) >> 4) * NumColsBetweenHalfWarps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int StrideBetweenWarps>
inline __device__ int ldst16dp32bitTmemRowIdx(int tidx) {
  // The base is (tidx / 32) * StrideBetweenWarps + (tidx % 16).
  return (tidx / 32) * StrideBetweenWarps + (tidx & 0xf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// Helper function for .16dp128bit and .16dp256bit.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int StrideBetweenWarps, int NumTmemCols>
inline __device__ int ldst16dpTmemRowIdx(int offset) {
  // This code computes "offset / (NumTmemCols * 2) * StrideBetweenWarps".

  // The number of columns multiplied by 2.
  auto constexpr NumTmemCols2{NumTmemCols * 2};
  // Extract the index of the instruction (128bit and 256bit span two rows => NumTmemCols * 2).
  auto instIdx = offset / NumTmemCols2;
  // Compute the stride.
  return instIdx * StrideBetweenWarps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// .16dp256bit.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int ldst16dp256bitTmemColIdx(int tidx) {
  // The base is (tidx % 4) * 2.
  return (tidx & 0x03) * 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The col/row functions below have the same template signature to simplify the calling code in
// the generator even if that's a bit awkward below :)

template <int, int NumTmemCols>
inline __device__ int ldst16dp256bitTmemColIdx(int base, int offset) {
  // The number of columns multiplied by 2.
  auto constexpr NumTmemCols2{NumTmemCols * 2};
  // The mask to extract NumTmemCols*2. The right mask excludes 0b11 as we div. by 4 and mul. by 8.
  auto constexpr offsetMask{(NumTmemCols2 - 1) & 0xffc};
  // The loop offset is (offset % NumTmemCols2) / 4 * 8 + (offset % NumTmemCols2) % 2.
  return base + (offset & offsetMask) * 2 + (offset & 0x01);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int StrideBetweenWarps>
inline __device__ int ldst16dp256bitTmemRowIdx(int tidx) {
  // The base is (tidx / 32) * StrideBetweenWarps + (tidx % 32) / 4.
  return (tidx / 32) * StrideBetweenWarps + (tidx & 0x1f) / 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int StrideBetweenWarps, int NumTmemCols>
inline __device__ int ldst16dp256bitTmemRowIdx(int base, int offset) {
  // The loop offset is (offset / (NumTmemCols * 2)) * StrideBetweenWarps + (offset % 4 / 2) * 8.
  return base + ldst16dpTmemRowIdx<StrideBetweenWarps, NumTmemCols>(offset) + (offset & 0x02) * 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// M=128 2CTA UTCMMA with 2x2 warpgroup layout.
//
// Warp0 TMEM Row [0  -  31] Col [0 - 31] holds M [0  - 31] N [0  - 31]
// Warp1 TMEM Row [32 -  63] Col [0 - 31] holds M [32 - 63] N [0  - 31]
// Warp2 TMEM Row [64 -  95] Col [0 - 31] holds M [0  - 31] N [32 - 63]
// Warp3 TMEM Row [96 - 127] Col [0 - 31] holds M [32 - 63] N [32 - 63]
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// Map tmemD's colIdx to the actual data's colIdx.
template <int InstN>
inline __device__ int warpGrp2x2TmemColIdxToDataColIdx(int warpGrpThreadIdx, int tmemColIdx) {
  // The dataD's colIdx = (warpGrpThreadIdx / 64) * (InstN / 2) + tmemColIdx.
  return (warpGrpThreadIdx / 64) * (InstN / 2) + tmemColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Map tmemD's rowIdx to the actual data's rowIdx.
inline __device__ int warpGrp2x2TmemRowIdxToDataRowIdx(int tmemRowIdx) {
  // The dataD's rowIdx = tmemRowIdx % 64.
  return tmemRowIdx & 0x3f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dev
}  // namespace trtllm
