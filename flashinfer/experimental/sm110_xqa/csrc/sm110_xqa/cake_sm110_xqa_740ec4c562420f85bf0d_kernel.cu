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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_sm110_xqa_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SOURCE_BACKING_OFF 0
#define SMEM_SOURCE_BACKING_STAGE_BYTES 167168
#define SMEM_SOURCE_BACKING_STRIDE 167168
#define SMEM_TOTAL 167168
#define THREADS 512

#if !defined(__CUDACC_RTC__)
#include <stddef.h>
#endif
struct __align__(8) KVCacheList {
    void* pool;
    const int* page_list;
    const int* sequence_lengths;
    unsigned int max_pages;
};
static_assert(sizeof(KVCacheList) == 32, "KVCacheList size");
static_assert(__alignof__(KVCacheList) == 8, "KVCacheList alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KVCacheList, pool) == 0, "KVCacheList.pool offset");
static_assert(offsetof(KVCacheList, page_list) == 8, "KVCacheList.page_list offset");
static_assert(offsetof(KVCacheList, sequence_lengths) == 16, "KVCacheList.sequence_lengths offset");
static_assert(offsetof(KVCacheList, max_pages) == 24, "KVCacheList.max_pages offset");
#endif

template <typename, unsigned int> struct Vec;
using LdGrain = Vec<unsigned int, 4U>;
template <>
struct __align__(16) Vec<unsigned int, 4U> {
    unsigned int data[4];
};
static_assert(sizeof(LdGrain) == 16, "LdGrain size");
static_assert(__alignof__(LdGrain) == 16, "LdGrain alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(LdGrain, data) == 0, "LdGrain.data offset");
#endif

template <typename, unsigned int, unsigned int, bool> struct Array2D;
using QSmemBuffer = Array2D<LdGrain, 32U, 64U, true>;
template <>
struct __align__(128) Array2D<LdGrain, 32U, 64U, true> {
    LdGrain data[32][64];
    template <bool swizzle = false>
    __device__ inline LdGrain& at(unsigned int r, unsigned int c)
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        LdGrain& ret = ((*this).data[0] + 0)[r * 64 + c_swizzled];
        return ret;
    }
    template <bool swizzle = false>
    __device__ inline const LdGrain& at(unsigned int r, unsigned int c) const
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        const LdGrain& ret = ((*this).data[0] + 0)[r * 64 + c_swizzled];
        return ret;
    }
};
static_assert(sizeof(QSmemBuffer) == 32768, "QSmemBuffer size");
static_assert(__alignof__(QSmemBuffer) == 128, "QSmemBuffer alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(QSmemBuffer, data) == 0, "QSmemBuffer.data offset");
#endif

using KSmemBuffer = Array2D<LdGrain, 64U, 4U, true>;
template <>
struct __align__(128) Array2D<LdGrain, 64U, 4U, true> {
    LdGrain data[64][4];
    template <bool swizzle = false>
    __device__ inline LdGrain& at(unsigned int r, unsigned int c)
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r / 2 % 4 : c);
        LdGrain& result = ((*this).data[0] + 0)[r * 4 + c_swizzled];
        return result;
    }
    template <bool swizzle = false>
    __device__ inline const LdGrain& at(unsigned int r, unsigned int c) const
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r / 2 % 4 : c);
        const LdGrain& ret = ((*this).data[0] + 0)[r * 4 + c_swizzled];
        return ret;
    }
};
static_assert(sizeof(KSmemBuffer) == 4096, "KSmemBuffer size");
static_assert(__alignof__(KSmemBuffer) == 128, "KSmemBuffer alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KSmemBuffer, data) == 0, "KSmemBuffer.data offset");
#endif

using XSmemBuffer = Array2D<LdGrain, 32U, 8U, true>;
template <>
struct __align__(128) Array2D<LdGrain, 32U, 8U, true> {
    LdGrain data[32][8];
    template <bool swizzle = false>
    __device__ inline LdGrain& at(unsigned int r, unsigned int c)
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        LdGrain& result = ((*this).data[0] + 0)[r * 8 + c_swizzled];
        return result;
    }
    template <bool swizzle = false>
    __device__ inline const LdGrain& at(unsigned int r, unsigned int c) const
    {
        unsigned int const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        const LdGrain& ret = ((*this).data[0] + 0)[r * 8 + c_swizzled];
        return ret;
    }
};
static_assert(sizeof(XSmemBuffer) == 4096, "XSmemBuffer size");
static_assert(__alignof__(XSmemBuffer) == 128, "XSmemBuffer alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(XSmemBuffer, data) == 0, "XSmemBuffer.data offset");
#endif

using VSmemBufferFP8 = Array2D<LdGrain, 32U, 32U, true>;
template <>
struct __align__(128) Array2D<LdGrain, 32U, 32U, true> {
    LdGrain data[32][32];
    template <bool swizzle = false>
    __device__ inline LdGrain& at(unsigned int r, unsigned int c)
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        LdGrain& result = ((*this).data[0] + 0)[r * 32 + c_swizzled];
        return result;
    }
    template <bool swizzle = false>
    __device__ inline const LdGrain& at(unsigned int r, unsigned int c) const
    {
        uint32_t const c_swizzled = ((swizzle) ? c ^ r % 8 : c);
        const LdGrain& ret = ((*this).data[0] + 0)[r * 32 + c_swizzled];
        return ret;
    }
};
static_assert(sizeof(VSmemBufferFP8) == 16384, "VSmemBufferFP8 size");
static_assert(__alignof__(VSmemBufferFP8) == 128, "VSmemBufferFP8 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(VSmemBufferFP8, data) == 0, "VSmemBufferFP8.data offset");
#endif

struct __align__(16) SMemWarpRowMax {
    float data[1][8][4];
};
static_assert(sizeof(SMemWarpRowMax) == 128, "SMemWarpRowMax size");
static_assert(__alignof__(SMemWarpRowMax) == 16, "SMemWarpRowMax alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SMemWarpRowMax, data) == 0, "SMemWarpRowMax.data offset");
#endif

struct __align__(8) CtaBarrier {
    unsigned long long mBar;
};
static_assert(sizeof(CtaBarrier) == 8, "CtaBarrier size");
static_assert(__alignof__(CtaBarrier) == 8, "CtaBarrier alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(CtaBarrier, mBar) == 0, "CtaBarrier.mBar offset");
#endif

struct __align__(8) CtaBarrierPair {
    CtaBarrier produced;
    CtaBarrier consumed;
};
static_assert(sizeof(CtaBarrierPair) == 16, "CtaBarrierPair size");
static_assert(__alignof__(CtaBarrierPair) == 8, "CtaBarrierPair alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(CtaBarrierPair, produced) == 0, "CtaBarrierPair.produced offset");
static_assert(offsetof(CtaBarrierPair, consumed) == 8, "CtaBarrierPair.consumed offset");
#endif

struct __align__(128) SharedMem {
    QSmemBuffer q[1][1];
    KSmemBuffer k[8][2];
    XSmemBuffer x[1][8];
    VSmemBufferFP8 v[1][1][2];
    SMemWarpRowMax warpRowMax[1][8];
    SMemWarpRowMax warpRowSum[1][8];
    SMemWarpRowMax ctaRowMax[1][8];
    CtaBarrier qBarrier[1];
    CtaBarrier qReuseBarrier[1];
    CtaBarrierPair xBarriers[1][8];
    CtaBarrier otherBarriers[3];
    CtaBarrier kPartialBarriers[4];
};
static_assert(sizeof(SharedMem) == 167168, "SharedMem size");
static_assert(__alignof__(SharedMem) == 128, "SharedMem alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SharedMem, q) == 0, "SharedMem.q offset");
static_assert(offsetof(SharedMem, k) == 32768, "SharedMem.k offset");
static_assert(offsetof(SharedMem, x) == 98304, "SharedMem.x offset");
static_assert(offsetof(SharedMem, v) == 131072, "SharedMem.v offset");
static_assert(offsetof(SharedMem, warpRowMax) == 163840, "SharedMem.warpRowMax offset");
static_assert(offsetof(SharedMem, warpRowSum) == 164864, "SharedMem.warpRowSum offset");
static_assert(offsetof(SharedMem, ctaRowMax) == 165888, "SharedMem.ctaRowMax offset");
static_assert(offsetof(SharedMem, qBarrier) == 166912, "SharedMem.qBarrier offset");
static_assert(offsetof(SharedMem, qReuseBarrier) == 166920, "SharedMem.qReuseBarrier offset");
static_assert(offsetof(SharedMem, xBarriers) == 166928, "SharedMem.xBarriers offset");
static_assert(offsetof(SharedMem, otherBarriers) == 167056, "SharedMem.otherBarriers offset");
static_assert(offsetof(SharedMem, kPartialBarriers) == 167080, "SharedMem.kPartialBarriers offset");
#endif

struct __align__(4) QHeadTokenMap {
    unsigned int q_heads;
    unsigned int head_group_size;
    unsigned int row_begin;
    __device__ inline unsigned int operator()(unsigned int idxHeadTokenLocal) const
    {
        idxHeadTokenLocal = idxHeadTokenLocal + (*this).row_begin;
        uint32_t const tokenIdx = idxHeadTokenLocal / (*this).head_group_size;
        uint32_t const headIdx = idxHeadTokenLocal % (*this).head_group_size;
        uint32_t const result = tokenIdx * (*this).q_heads + headIdx;
        return result;
    }
};
static_assert(sizeof(QHeadTokenMap) == 12, "QHeadTokenMap size");
static_assert(__alignof__(QHeadTokenMap) == 4, "QHeadTokenMap alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(QHeadTokenMap, q_heads) == 0, "QHeadTokenMap.q_heads offset");
static_assert(offsetof(QHeadTokenMap, head_group_size) == 4, "QHeadTokenMap.head_group_size offset");
static_assert(offsetof(QHeadTokenMap, row_begin) == 8, "QHeadTokenMap.row_begin offset");
#endif

using OutputHead = Vec<__half, 512U>;
template <>
struct __align__(16) Vec<__half, 512U> {
    __half data[512];
};
static_assert(sizeof(OutputHead) == 1024, "OutputHead size");
static_assert(__alignof__(OutputHead) == 16, "OutputHead alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(OutputHead, data) == 0, "OutputHead.data offset");
#endif

struct __align__(8) TinyPtrConstIOHead {
    const OutputHead* base;
    unsigned int offset;
    __device__ inline operator const OutputHead*() const
    {
        const OutputHead* const result = (*this).base + (*this).offset;
        return result;
    }
    __device__ inline TinyPtrConstIOHead operator+(unsigned int i) const
    {
        TinyPtrConstIOHead const result{(*this).base, (*this).offset + i};
        return result;
    }
};
static_assert(sizeof(TinyPtrConstIOHead) == 16, "TinyPtrConstIOHead size");
static_assert(__alignof__(TinyPtrConstIOHead) == 8, "TinyPtrConstIOHead alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(TinyPtrConstIOHead, base) == 0, "TinyPtrConstIOHead.base offset");
static_assert(offsetof(TinyPtrConstIOHead, offset) == 8, "TinyPtrConstIOHead.offset offset");
#endif

using CacheHeadFp8 = Vec<uint8_t, 512U>;
template <>
struct __align__(16) Vec<uint8_t, 512U> {
    uint8_t data[512];
};
static_assert(sizeof(CacheHeadFp8) == 512, "CacheHeadFp8 size");
static_assert(__alignof__(CacheHeadFp8) == 16, "CacheHeadFp8 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(CacheHeadFp8, data) == 0, "CacheHeadFp8.data offset");
#endif

using KVCachePageIndices1 = Vec<int, 1U>;
template <>
struct __align__(4) Vec<int, 1U> {
    int data[1];
};
static_assert(sizeof(KVCachePageIndices1) == 4, "KVCachePageIndices1 size");
static_assert(__alignof__(KVCachePageIndices1) == 4, "KVCachePageIndices1 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KVCachePageIndices1, data) == 0, "KVCachePageIndices1.data offset");
#endif

struct __align__(8) HeadPtrConstFP8Page1 {
    const CacheHeadFp8* pool;
    KVCachePageIndices1 pageIndices;
    unsigned int nbKHeads;
    unsigned int offset;
    unsigned int tokensPerPageLog2;
    const CacheHeadFp8* base;
    unsigned int mValidMask;
    __device__ inline unsigned int validMask() const
    {
        uint32_t const result = (*this).mValidMask;
        return result;
    }
    __device__ inline const CacheHeadFp8* operator+(unsigned int i) const
    {
        const CacheHeadFp8* const result = (*this).base + ((unsigned long long)i * (unsigned long long)(*this).nbKHeads);
        return result;
    }
};
static_assert(sizeof(HeadPtrConstFP8Page1) == 40, "HeadPtrConstFP8Page1 size");
static_assert(__alignof__(HeadPtrConstFP8Page1) == 8, "HeadPtrConstFP8Page1 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(HeadPtrConstFP8Page1, pool) == 0, "HeadPtrConstFP8Page1.pool offset");
static_assert(offsetof(HeadPtrConstFP8Page1, pageIndices) == 8, "HeadPtrConstFP8Page1.pageIndices offset");
static_assert(offsetof(HeadPtrConstFP8Page1, nbKHeads) == 12, "HeadPtrConstFP8Page1.nbKHeads offset");
static_assert(offsetof(HeadPtrConstFP8Page1, offset) == 16, "HeadPtrConstFP8Page1.offset offset");
static_assert(offsetof(HeadPtrConstFP8Page1, tokensPerPageLog2) == 20, "HeadPtrConstFP8Page1.tokensPerPageLog2 offset");
static_assert(offsetof(HeadPtrConstFP8Page1, base) == 24, "HeadPtrConstFP8Page1.base offset");
static_assert(offsetof(HeadPtrConstFP8Page1, mValidMask) == 32, "HeadPtrConstFP8Page1.mValidMask offset");
#endif

struct __align__(1) Warp {
};
static_assert(sizeof(Warp) == 1, "Warp size");
static_assert(__alignof__(Warp) == 1, "Warp alignment");

struct __align__(1) KIdentityHeadMap {
    __device__ inline unsigned int operator()(unsigned int x) const
    {
        return x;
    }
};
static_assert(sizeof(KIdentityHeadMap) == 1, "KIdentityHeadMap size");
static_assert(__alignof__(KIdentityHeadMap) == 1, "KIdentityHeadMap alignment");

using InstAcc = Array2D<float, 2U, 2U, true>;
template <>
struct __align__(16) Array2D<float, 2U, 2U, true> {
    float data[2][2];
};
static_assert(sizeof(InstAcc) == 16, "InstAcc size");
static_assert(__alignof__(InstAcc) == 16, "InstAcc alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(InstAcc, data) == 0, "InstAcc.data offset");
#endif

using WarpAcc = Array2D<InstAcc, 2U, 8U, true>;
template <>
struct __align__(128) Array2D<InstAcc, 2U, 8U, true> {
    InstAcc data[2][8];
};
static_assert(sizeof(WarpAcc) == 256, "WarpAcc size");
static_assert(__alignof__(WarpAcc) == 128, "WarpAcc alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(WarpAcc, data) == 0, "WarpAcc.data offset");
#endif

using QuadRegRowMax = Vec<float, 4U>;
template <>
struct __align__(16) Vec<float, 4U> {
    float data[4];
};
static_assert(sizeof(QuadRegRowMax) == 16, "QuadRegRowMax size");
static_assert(__alignof__(QuadRegRowMax) == 16, "QuadRegRowMax alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(QuadRegRowMax, data) == 0, "QuadRegRowMax.data offset");
#endif

using GemmOutRegTile = Array2D<__half2, 4U, 8U, true>;
template <>
struct __align__(128) Array2D<__half2, 4U, 8U, true> {
    __half2 data[4][8];
};
static_assert(sizeof(GemmOutRegTile) == 128, "GemmOutRegTile size");
static_assert(__alignof__(GemmOutRegTile) == 128, "GemmOutRegTile alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(GemmOutRegTile, data) == 0, "GemmOutRegTile.data offset");
#endif

using ThrdRegRowMax = Vec<float, 1U>;
template <>
struct __align__(4) Vec<float, 1U> {
    float data[1];
};
static_assert(sizeof(ThrdRegRowMax) == 4, "ThrdRegRowMax size");
static_assert(__alignof__(ThrdRegRowMax) == 4, "ThrdRegRowMax alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(ThrdRegRowMax, data) == 0, "ThrdRegRowMax.data offset");
#endif

using UniformRescaleMask = Vec<unsigned int, 1U>;
template <>
struct __align__(4) Vec<unsigned int, 1U> {
    unsigned int data[1];
};
static_assert(sizeof(UniformRescaleMask) == 4, "UniformRescaleMask size");
static_assert(__alignof__(UniformRescaleMask) == 4, "UniformRescaleMask alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(UniformRescaleMask, data) == 0, "UniformRescaleMask.data offset");
#endif

template <unsigned int, unsigned int> struct InstInMat;
using InstInMat22 = InstInMat<2U, 2U>;
template <>
struct __align__(4) InstInMat<2U, 2U> {
    unsigned int data[2][2];
};
static_assert(sizeof(InstInMat22) == 16, "InstInMat22 size");
static_assert(__alignof__(InstInMat22) == 4, "InstInMat22 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(InstInMat22, data) == 0, "InstInMat22.data offset");
#endif

using SourceSlice22x2 = Array2D<InstInMat22, 2U, 1U, true>;
template <>
struct __align__(32) Array2D<InstInMat22, 2U, 1U, true> {
    InstInMat22 data[2][1];
};
static_assert(sizeof(SourceSlice22x2) == 32, "SourceSlice22x2 size");
static_assert(__alignof__(SourceSlice22x2) == 32, "SourceSlice22x2 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceSlice22x2, data) == 0, "SourceSlice22x2.data offset");
#endif

using InstInMat41 = InstInMat<4U, 1U>;
template <>
struct __align__(4) InstInMat<4U, 1U> {
    unsigned int data[4][1];
};
static_assert(sizeof(InstInMat41) == 16, "InstInMat41 size");
static_assert(__alignof__(InstInMat41) == 4, "InstInMat41 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(InstInMat41, data) == 0, "InstInMat41.data offset");
#endif

using SourceSlice41x2 = Array2D<InstInMat41, 2U, 1U, true>;
template <>
struct __align__(32) Array2D<InstInMat41, 2U, 1U, true> {
    InstInMat41 data[2][1];
};
static_assert(sizeof(SourceSlice41x2) == 32, "SourceSlice41x2 size");
static_assert(__alignof__(SourceSlice41x2) == 32, "SourceSlice41x2 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceSlice41x2, data) == 0, "SourceSlice41x2.data offset");
#endif

using InstInMat42 = InstInMat<4U, 2U>;
template <>
struct __align__(4) InstInMat<4U, 2U> {
    unsigned int data[4][2];
};
static_assert(sizeof(InstInMat42) == 32, "InstInMat42 size");
static_assert(__alignof__(InstInMat42) == 4, "InstInMat42 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(InstInMat42, data) == 0, "InstInMat42.data offset");
#endif

using SourceSlice42x2 = Array2D<InstInMat42, 2U, 1U, true>;
template <>
struct __align__(64) Array2D<InstInMat42, 2U, 1U, true> {
    InstInMat42 data[2][1];
};
static_assert(sizeof(SourceSlice42x2) == 64, "SourceSlice42x2 size");
static_assert(__alignof__(SourceSlice42x2) == 64, "SourceSlice42x2 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceSlice42x2, data) == 0, "SourceSlice42x2.data offset");
#endif

using SourceRowSumAcc = Vec<InstAcc, 2U>;
template <>
struct __align__(16) Vec<InstAcc, 2U> {
    InstAcc data[2];
};
static_assert(sizeof(SourceRowSumAcc) == 32, "SourceRowSumAcc size");
static_assert(__alignof__(SourceRowSumAcc) == 16, "SourceRowSumAcc alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceRowSumAcc, data) == 0, "SourceRowSumAcc.data offset");
#endif

struct __align__(4) VGroupedHeadOffsetMap {
    unsigned int dst_head_offset;
    __device__ inline unsigned int operator()(unsigned int x) const
    {
        KIdentityHeadMap const localHeadIdxMap{};
        uint32_t const result = localHeadIdxMap((*this).dst_head_offset + x);
        return result;
    }
};
static_assert(sizeof(VGroupedHeadOffsetMap) == 4, "VGroupedHeadOffsetMap size");
static_assert(__alignof__(VGroupedHeadOffsetMap) == 4, "VGroupedHeadOffsetMap alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(VGroupedHeadOffsetMap, dst_head_offset) == 0, "VGroupedHeadOffsetMap.dst_head_offset offset");
#endif

using Half2Scale1 = Vec<__half2, 1U>;
template <>
struct __align__(4) Vec<__half2, 1U> {
    __half2 data[1];
};
static_assert(sizeof(Half2Scale1) == 4, "Half2Scale1 size");
static_assert(__alignof__(Half2Scale1) == 4, "Half2Scale1 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(Half2Scale1, data) == 0, "Half2Scale1.data offset");
#endif

using SourceVCacheWordF16 = Vec<unsigned int, 2U>;
template <>
struct __align__(8) Vec<unsigned int, 2U> {
    unsigned int data[2];
};
static_assert(sizeof(SourceVCacheWordF16) == 8, "SourceVCacheWordF16 size");
static_assert(__alignof__(SourceVCacheWordF16) == 8, "SourceVCacheWordF16 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceVCacheWordF16, data) == 0, "SourceVCacheWordF16.data offset");
#endif

using Half2Scale4 = Vec<__half2, 4U>;
template <>
struct __align__(16) Vec<__half2, 4U> {
    __half2 data[4];
};
static_assert(sizeof(Half2Scale4) == 16, "Half2Scale4 size");
static_assert(__alignof__(Half2Scale4) == 16, "Half2Scale4 alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(Half2Scale4, data) == 0, "Half2Scale4.data offset");
#endif

using SourceOutputSeg = Vec<__half2, 2U>;
template <>
struct __align__(8) Vec<__half2, 2U> {
    __half2 data[2];
};
static_assert(sizeof(SourceOutputSeg) == 8, "SourceOutputSeg size");
static_assert(__alignof__(SourceOutputSeg) == 8, "SourceOutputSeg alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceOutputSeg, data) == 0, "SourceOutputSeg.data offset");
#endif

using SourceOutputReorderedSeg = Vec<__half, 4U>;
template <>
struct __align__(8) Vec<__half, 4U> {
    __half data[4];
};
static_assert(sizeof(SourceOutputReorderedSeg) == 8, "SourceOutputReorderedSeg size");
static_assert(__alignof__(SourceOutputReorderedSeg) == 8, "SourceOutputReorderedSeg alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(SourceOutputReorderedSeg, data) == 0, "SourceOutputReorderedSeg.data offset");
#endif

using OutputGrain = Vec<__half, 8U>;
template <>
struct __align__(16) Vec<__half, 8U> {
    __half data[8];
};
static_assert(sizeof(OutputGrain) == 16, "OutputGrain size");
static_assert(__alignof__(OutputGrain) == 16, "OutputGrain alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(OutputGrain, data) == 0, "OutputGrain.data offset");
#endif

namespace xqa_shared_compute_v102 {
__device__ inline unsigned int laneId()
{
    uint32_t id;
    asm("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
template <bool isFull>
__device__ inline void copyHeadsAsyncMultiWarp(unsigned int idxWarp, QSmemBuffer& dst, const TinyPtrConstIOHead& src, unsigned int nbAvailHeads, const QHeadTokenMap& localHeadIdxMap)
{
    uint32_t const tid_1 = 32 * idxWarp + xqa_shared_compute_v102::laneId();
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t const idxGrain = 256 * i + tid_1;
        if (idxGrain >= 2048) {
            break;
        }
        uint32_t const idxHeadLocal = idxGrain / 64;
        uint32_t const idxGrainInsideHead = idxGrain % 64;
        bool const isHeadInBound = isFull || idxHeadLocal < nbAvailHeads;
        const OutputHead* const pSrcHead = static_cast<const OutputHead*>(((src + localHeadIdxMap(idxHeadLocal))));
        bool const isValidPage = (unsigned long long)pSrcHead != 0;
        const LdGrain* const pSrc = reinterpret_cast<const LdGrain*>(pSrcHead) + idxGrainInsideHead;
        LdGrain* const pDst = &dst.template at<true>(idxHeadLocal, idxGrainInsideHead);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "l"(__cvta_generic_to_shared(pDst)), "l"(pSrc), "r"((isValidPage && isHeadInBound) ? 16 : 0));
    }
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace k_fp8_page128 {
__device__ inline HeadPtrConstFP8Page1 constructHeadPtrPage1(const CacheHeadFp8* pool, KVCachePageIndices1 pageIndices, unsigned int nbKHeads, unsigned int offset, unsigned int tokensPerPageLog2)
{
    const CacheHeadFp8* const base = ((((unsigned int)pageIndices.data[0] & 2147483648u) != 0) ? pool : pool + ((unsigned int)pageIndices.data[0] * (nbKHeads * (1 << tokensPerPageLog2)) + offset));
    uint32_t const mValidMask = ((((unsigned int)pageIndices.data[0] & 2147483648u) != 0) ? 0 : 4294967295u);
    HeadPtrConstFP8Page1 const result{pool, pageIndices, nbKHeads, offset, tokensPerPageLog2, base, mValidMask};
    return result;
}
} // namespace k_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace k_fp8_page128 {
__device__ inline HeadPtrConstFP8Page1 makePagedHeadPtr(const CacheHeadFp8* pool, KVCachePageIndices1 pageIndices, unsigned int nbKHeads, unsigned int offset, unsigned int sliceByteOffset, unsigned int tokensPerPageLog2)
{
    HeadPtrConstFP8Page1 const result = xqa_shared_compute_v102::k_fp8_page128::constructHeadPtrPage1(pool, pageIndices, nbKHeads, offset, tokensPerPageLog2);
    return result;
}
} // namespace k_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace k_fp8_page128 {
__device__ inline unsigned int srcTileValidMask(const HeadPtrConstFP8Page1& src)
{
    uint32_t const result = src.validMask();
    return result;
}
} // namespace k_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace k_fp8_page128 {
template <bool isFull>
__device__ inline void copyPartialHeadsAsync(const Warp& warp_1, KSmemBuffer& dst, unsigned int dstHeadOffset, const HeadPtrConstFP8Page1& src, unsigned int idxPart, unsigned int nbAvailHeads, const KIdentityHeadMap& localHeadIdxMap)
{
    uint32_t const warpLane = xqa_shared_compute_v102::laneId();
    uint32_t const segIdx = warpLane / 4;
    uint32_t const segLane = warpLane % 4;
    uint32_t const tileValidMask = xqa_shared_compute_v102::k_fp8_page128::srcTileValidMask(src);
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t const idxHeadLocal = 8 * i + segIdx;
        bool const isHeadInBound = isFull || idxHeadLocal < nbAvailHeads;
        uint32_t const idxGrainInsideHead = 4 * idxPart + segLane;
        const CacheHeadFp8* const pSrcHead = (src + localHeadIdxMap(idxHeadLocal));
        bool const isValidPage = 1;
        const LdGrain* const pSrc = reinterpret_cast<const LdGrain*>(pSrcHead) + idxGrainInsideHead;
        LdGrain* const pDst = &dst.template at<true>(dstHeadOffset + idxHeadLocal, segLane);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "l"(__cvta_generic_to_shared(pDst)), "l"(pSrc), "r"(((tileValidMask & ((isValidPage && isHeadInBound) ? 16 : 0)) != 0) ? 16 : 0));
    }
}
} // namespace k_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline LdGrain ldmatrix_4x(const Warp& warp_1, const LdGrain* row)
{
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(row)))
        : "memory");
    LdGrain result{{a, b, c, d}};
    return result;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_q {
__device__ inline InstInMat22 loadInstInMat(const Warp& warp_1, const QSmemBuffer& src, unsigned int rowOffset, unsigned int colOffset)
{
    uint32_t const idx = xqa_shared_compute_v102::laneId() / 8;
    uint32_t const idxKEx = idx / 2;
    uint32_t const idxMNEx = idx % 2;
    uint32_t const srcIdxKEx = idxKEx;
    uint32_t const srcIdxMNEx = idxMNEx;
    uint32_t rowLane = xqa_shared_compute_v102::laneId();
    const LdGrain& grain = src.template at<true>(rowOffset + 8 * srcIdxMNEx + rowLane % 8, colOffset + srcIdxKEx);
    const LdGrain* const ptr = &grain;
    LdGrain const data = xqa_shared_compute_v102::ldmatrix_4x(warp_1, ptr);
    InstInMat22 dst;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst.data[i / 2][i % 2] = data.data[i];
    }
    return dst;
}
} // namespace load_q
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_q {
__device__ inline SourceSlice22x2 loadMatrix(const Warp& warp_1, const QSmemBuffer& src, unsigned int rowBeg, unsigned int colBeg)
{
    SourceSlice22x2 dst;
    #pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 1; j++) {
            dst.data[i][j] = xqa_shared_compute_v102::load_q::loadInstInMat(warp_1, src, rowBeg + 16 * i, colBeg + 2 * j);
        }
    }
    return dst;
}
} // namespace load_q
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_k_fp8 {
__device__ inline InstInMat41 loadInstInMat(const Warp& warp_1, const KSmemBuffer& src, unsigned int rowOffset, unsigned int colOffset)
{
    uint32_t const idx = xqa_shared_compute_v102::laneId() / 8;
    uint32_t const idxKEx = idx;
    uint32_t const idxMNEx = 0;
    uint32_t const srcIdxKEx = idxMNEx;
    uint32_t const srcIdxMNEx = idxKEx;
    uint32_t rowLane = xqa_shared_compute_v102::laneId();
    const LdGrain& grain = src.template at<true>(rowOffset + 8 * srcIdxMNEx + rowLane % 8, colOffset + srcIdxKEx);
    const LdGrain* const ptr = &grain;
    LdGrain const data = xqa_shared_compute_v102::ldmatrix_4x(warp_1, ptr);
    InstInMat41 dst;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst.data[i][0] = data.data[i];
    }
    return dst;
}
} // namespace load_k_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_k_fp8 {
__device__ inline SourceSlice41x2 loadMatrix(const Warp& warp_1, const KSmemBuffer& src, unsigned int rowBeg, unsigned int colBeg)
{
    SourceSlice41x2 dst;
    #pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 1; j++) {
            dst.data[i][j] = xqa_shared_compute_v102::load_k_fp8::loadInstInMat(warp_1, src, rowBeg + 32 * i, colBeg + j);
        }
    }
    return dst;
}
} // namespace load_k_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace qk_fp8 {
__device__ inline SourceSlice42x2 makeKSlice(const SourceSlice41x2& kSliceOrig)
{
    SourceSlice42x2 ret;
    #pragma unroll
    for (uint32_t m = 0; m < 2; m++) {
        #pragma unroll
        for (uint32_t n = 0; n < 1; n++) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < 1; j++) {
                    uint32_t data0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(data0) : "h"((uint16_t)kSliceOrig.data[m][n].data[i][j]));
                    uint32_t data1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(data1) : "h"((uint16_t)(kSliceOrig.data[m][n].data[i][j] >> 16)));
                    ret.data[m][n].data[i][j * 2] = data0;
                    ret.data[m][n].data[i][j * 2 + 1] = data1;
                }
            }
        }
    }
    return ret;
}
} // namespace qk_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace qk_fp8 {
__device__ inline void smemQKPartGemm(const Warp& warp_1, WarpAcc& acc, const QSmemBuffer& q, unsigned int qColBeg, const KSmemBuffer& k)
{
    #pragma unroll 2
    for (uint32_t s = 0; s < 4; s++) {
        SourceSlice22x2 const qSlice = xqa_shared_compute_v102::load_q::loadMatrix(warp_1, q, 0, qColBeg + 2 * s);
        SourceSlice41x2 const kSliceOrig = xqa_shared_compute_v102::load_k_fp8::loadMatrix(warp_1, k, 0, s);
        SourceSlice42x2 const kSlice = xqa_shared_compute_v102::qk_fp8::makeKSlice(kSliceOrig);
        #pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            #pragma unroll
            for (uint32_t j = 0; j < 2; j++) {
                InstInMat22 const matrixA = qSlice.data[i][0];
                InstInMat42 const matrixB = kSlice.data[j][0];
                #pragma unroll
                for (uint32_t n = 0; n < 4; n++) {
                    unsigned int const b[2][1]{{matrixB.data[n][0]}, {matrixB.data[n][1]}};
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc.data[i][j * 4 + n].data)[0][0]), "+f"((acc.data[i][j * 4 + n].data)[0][1]), "+f"((acc.data[i][j * 4 + n].data)[1][0]), "+f"((acc.data[i][j * 4 + n].data)[1][1])
                        : "r"((matrixA.data)[0][0]), "r"((matrixA.data)[0][1]), "r"((matrixA.data)[1][0]), "r"((matrixA.data)[1][1]), "r"((b)[0][0]), "r"((b)[1][0]));
                }
            }
        }
    }
    return;
}
} // namespace qk_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline void rescaleAcc(const Warp& warp_1, WarpAcc& acc, float scale)
{
    #pragma unroll
    for (uint32_t m = 0; m < 2; m++) {
        #pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            #pragma unroll
            for (uint32_t n = 0; n < 8; n++) {
                #pragma unroll
                for (uint32_t j = 0; j < 2; j++) {
                    acc.data[m][n].data[i][j] = acc.data[m][n].data[i][j] * scale;
                }
            }
        }
    }
    return;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline QuadRegRowMax warpTileOnlineSoftmax(const Warp& warp_1, const QuadRegRowMax& rowMaxHint, WarpAcc& acc)
{
    QuadRegRowMax rowMax = rowMaxHint;
    #pragma unroll
    for (uint32_t n = 0; n < 8; n++) {
        #pragma unroll
        for (uint32_t j = 0; j < 2; j++) {
            #pragma unroll
            for (uint32_t m = 0; m < 2; m++) {
                #pragma unroll
                for (uint32_t i = 0; i < 2; i++) {
                    float localMax = fmaxf(rowMax.data[m * 2 + i], acc.data[m][n].data[i][j]);
                    rowMax.data[m * 2 + i] = localMax;
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t row = 0; row < 4; row++) {
        float shuffled = __shfl_xor_sync(0xFFFFFFFF, rowMax.data[row], 2);
        float quadMax = fmaxf(rowMax.data[row], shuffled);
        rowMax.data[row] = quadMax;
    }
    #pragma unroll
    for (uint32_t row_1 = 0; row_1 < 4; row_1++) {
        float shuffled_1 = __shfl_xor_sync(0xFFFFFFFF, rowMax.data[row_1], 1);
        float quadMax_1 = fmaxf(rowMax.data[row_1], shuffled_1);
        rowMax.data[row_1] = quadMax_1;
    }
    #pragma unroll
    for (uint32_t m_1 = 0; m_1 < 2; m_1++) {
        #pragma unroll
        for (uint32_t i_1 = 0; i_1 < 2; i_1++) {
            float const maxVal = rowMax.data[m_1 * 2 + i_1];
            float const bias = maxVal * 1.4426950408889634f;
            #pragma unroll
            for (uint32_t n_1 = 0; n_1 < 8; n_1++) {
                #pragma unroll
                for (uint32_t j_1 = 0; j_1 < 2; j_1++) {
                    float exponent = approx_exp2(acc.data[m_1][n_1].data[i_1][j_1] * 1.4426950408889634f - bias);
                    acc.data[m_1][n_1].data[i_1][j_1] = exponent;
                }
            }
        }
    }
    return rowMax;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace output_tile {
__device__ inline GemmOutRegTile toFp16(const WarpAcc& acc)
{
    GemmOutRegTile dst;
    #pragma unroll
    for (uint32_t m = 0; m < 2; m++) {
        #pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            #pragma unroll
            for (uint32_t n = 0; n < 8; n++) {
                #pragma unroll
                for (uint32_t j = 0; j < 2; j += 2) {
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2((acc.data[m][n].data[i] + j)[_lp*2 + 0], (acc.data[m][n].data[i] + j)[_lp*2+1 + 0]));
                        (reinterpret_cast<unsigned int*>(dst.data[m * 2 + i] + (n * 2 + j) / 2))[_lp] = *(uint32_t*)&_h2;
                    }
                }
            }
        }
    }
    return dst;
}
} // namespace output_tile
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace output_tile {
__device__ inline QuadRegRowMax computeRowSum(const Warp& warp_1, const GemmOutRegTile& src)
{
    SourceRowSumAcc acc{};
    __half2 const b[2][1]{{::__float2half2_rn(1.0f)}, {::__float2half2_rn(1.0f)}};
    #pragma unroll
    for (uint32_t n = 0; n < 4; n++) {
        #pragma unroll
        for (uint32_t m = 0; m < 2; m++) {
            __half2 const a[2][2]{{src.data[m * 2][n * 2], src.data[m * 2 + 1][n * 2]}, {src.data[m * 2][n * 2 + 1], src.data[m * 2 + 1][n * 2 + 1]}};
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc.data[m].data)[0][0]), "+f"((acc.data[m].data)[0][1]), "+f"((acc.data[m].data)[1][0]), "+f"((acc.data[m].data)[1][1])
                : "r"((reinterpret_cast<const unsigned int (&)[2][2]>(a))[0][0]), "r"((reinterpret_cast<const unsigned int (&)[2][2]>(a))[0][1]), "r"((reinterpret_cast<const unsigned int (&)[2][2]>(a))[1][0]), "r"((reinterpret_cast<const unsigned int (&)[2][2]>(a))[1][1]), "r"((reinterpret_cast<const unsigned int (&)[2][1]>(b))[0][0]), "r"((reinterpret_cast<const unsigned int (&)[2][1]>(b))[1][0]));
        }
    }
    QuadRegRowMax rowSum;
    #pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 2; j++) {
            rowSum.data[i * 2 + j] = acc.data[i].data[j][0];
        }
        rowSum.data[i * 2] = acc.data[i].data[0][0];
        rowSum.data[i * 2 + 1] = acc.data[i].data[1][0];
    }
    #pragma unroll
    for (uint32_t i_1 = 0; i_1 < 4; i_1++) {
        uint32_t lane_1 = xqa_shared_compute_v102::laneId();
        const float lane0Val = __shfl_sync(15 << lane_1 / 4 * 4, rowSum.data[i_1], 0, 4);
        rowSum.data[i_1] = lane0Val;
    }
    return rowSum;
}
} // namespace output_tile
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace output_tile {
__device__ inline void storeOrderedGemmOutTile(const Warp& warp_1, XSmemBuffer& dst, const GemmOutRegTile& src)
{
    uint32_t const lane_1 = xqa_shared_compute_v102::laneId();
    #pragma unroll
    for (uint32_t m = 0; m < 4; m++) {
        #pragma unroll
        for (uint32_t n = 0; n < 2; n++) {
            LdGrain& grain = dst.template at<true>(8 * m + lane_1 % 8, 4 * n + lane_1 / 8);
            LdGrain* const p = &grain;
            LdGrain data;
            #pragma unroll
            for (uint32_t i = 0; i < 1; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    data.data[i * 4 + j] = reinterpret_cast<const unsigned int&>(src.data[m + i][n * 4 + j]);
                }
            }
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "l"(__cvta_generic_to_shared(p)), "r"(*reinterpret_cast<const uint32_t*>(&data.data[0])), "r"(*reinterpret_cast<const uint32_t*>(&data.data[1])), "r"(*reinterpret_cast<const uint32_t*>(&data.data[2])), "r"(*reinterpret_cast<const uint32_t*>(&data.data[3]))
                : "memory");
        }
    }
    return;
}
} // namespace output_tile
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace v_fp8_page128 {
template <bool isFull>
__device__ inline void copyPartialHeadsAsync(const Warp& warp_1, VSmemBufferFP8& dst, unsigned int dstHeadOffset, const HeadPtrConstFP8Page1& src, unsigned int idxPart, unsigned int nbAvailHeads, const VGroupedHeadOffsetMap& localHeadIdxMap)
{
    uint32_t const warpLane = xqa_shared_compute_v102::laneId();
    uint32_t const segIdx = warpLane / 32;
    uint32_t const segLane = warpLane % 32;
    uint32_t const tileValidMask = xqa_shared_compute_v102::k_fp8_page128::srcTileValidMask(src);
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t const idxHeadLocal = i + segIdx;
        bool const isHeadInBound = isFull || idxHeadLocal < nbAvailHeads;
        uint32_t const idxGrainInsideHead = 32 * idxPart + segLane;
        const CacheHeadFp8* const pSrcHead = (src + localHeadIdxMap(idxHeadLocal));
        bool const isValidPage = 1;
        const LdGrain* const pSrc = reinterpret_cast<const LdGrain*>(pSrcHead) + idxGrainInsideHead;
        LdGrain* const pDst = &dst.template at<true>(dstHeadOffset + idxHeadLocal, segLane);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "l"(__cvta_generic_to_shared(pDst)), "l"(pSrc), "r"(((tileValidMask & ((isValidPage && isHeadInBound) ? 16 : 0)) != 0) ? 16 : 0));
    }
}
} // namespace v_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace v_fp8_page128 {
__device__ inline void copyHeadsAsync(unsigned int idxWarp, VSmemBufferFP8& dst, const HeadPtrConstFP8Page1& src, unsigned int nbAvailHeads, const KIdentityHeadMap& localHeadIdxMap)
{
    Warp const warp_1{};
    uint32_t const dstHeadOffset = 4 * idxWarp;
    uint32_t const warpNbAvailHeads = ((dstHeadOffset < nbAvailHeads) ? nbAvailHeads - dstHeadOffset : 0);
    VGroupedHeadOffsetMap const offsetHeadIdxMap{dstHeadOffset};
    xqa_shared_compute_v102::v_fp8_page128::copyPartialHeadsAsync<false>(warp_1, dst, dstHeadOffset, src, 0, warpNbAvailHeads, offsetHeadIdxMap);
}
} // namespace v_fp8_page128
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline float replicateValForQuad(const Warp& warp_1, const ThrdRegRowMax& src, unsigned int idxMat8)
{
    uint32_t i = idxMat8 / 4;
    uint32_t j = idxMat8 % 4;
    uint32_t lane_1 = xqa_shared_compute_v102::laneId();
    float result;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(result) : "f"(src.data[i]), "r"(8 * j + lane_1 / 4));
    return result;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline void rescaleAcc(const Warp& warp_1, WarpAcc& acc, const UniformRescaleMask& rescaleMask, const ThrdRegRowMax& rowScales)
{
    #pragma unroll
    for (uint32_t m = 0; m < 2; m++) {
        #pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            uint32_t r = m * 2 + i;
            float scale = xqa_shared_compute_v102::replicateValForQuad(warp_1, rowScales, r);
            #pragma unroll
            for (uint32_t n = 0; n < 8; n++) {
                #pragma unroll
                for (uint32_t j = 0; j < 2; j++) {
                    acc.data[m][n].data[i][j] = acc.data[m][n].data[i][j] * scale;
                }
            }
        }
    }
    return;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace half2_convert {
__device__ inline Half2Scale1 __float2half2_rn(const ThrdRegRowMax& a)
{
    Half2Scale1 result;
    #pragma unroll
    for (uint32_t i = 0; i < 1; i++) {
        result.data[i] = ::__float2half2_rn(a.data[i]);
    }
    return result;
}
} // namespace half2_convert
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace quad_view {
__device__ inline QuadRegRowMax replicateForQuad(const Warp& warp_1, const ThrdRegRowMax& src)
{
    QuadRegRowMax dst{};
    #pragma unroll
    for (uint32_t i = 0; i < 1; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t lane_1 = xqa_shared_compute_v102::laneId();
            float shuffled;
            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(shuffled) : "f"(src.data[i]), "r"(8 * j + lane_1 / 4));
            dst.data[i * 4 + j] = shuffled;
        }
    }
    return dst;
}
} // namespace quad_view
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline InstInMat22 loadInstInMat(const Warp& warp_1, const XSmemBuffer& src, unsigned int rowOffset, unsigned int colOffset)
{
    uint32_t const idx = xqa_shared_compute_v102::laneId() / 8;
    uint32_t const idxKEx = idx / 2;
    uint32_t const idxMNEx = idx % 2;
    uint32_t const srcIdxKEx = idxKEx;
    uint32_t const srcIdxMNEx = idxMNEx;
    uint32_t rowLane = xqa_shared_compute_v102::laneId();
    const LdGrain& grain = src.template at<true>(rowOffset + 8 * srcIdxMNEx + rowLane % 8, colOffset + srcIdxKEx);
    const LdGrain* const ptr = &grain;
    LdGrain const data = xqa_shared_compute_v102::ldmatrix_4x(warp_1, ptr);
    InstInMat22 dst;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst.data[i / 2][i % 2] = data.data[i];
    }
    return dst;
}
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_x {
__device__ inline SourceSlice22x2 loadMatrix(const Warp& warp_1, const XSmemBuffer& src, unsigned int rowBeg, unsigned int colBeg)
{
    SourceSlice22x2 dst;
    #pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 1; j++) {
            dst.data[i][j] = xqa_shared_compute_v102::loadInstInMat(warp_1, src, rowBeg + 16 * i, colBeg + 2 * j);
        }
    }
    return dst;
}
} // namespace load_x
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace ldmatrix_transposed {
__device__ inline LdGrain ldmatrix_4x(const Warp& warp_1, const LdGrain* row)
{
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(row)))
        : "memory");
    LdGrain result{{a, b, c, d}};
    return result;
}
} // namespace ldmatrix_transposed
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_v_fp8 {
__device__ inline InstInMat22 loadInstInMat(const Warp& warp_1, const VSmemBufferFP8& src, unsigned int rowOffset, unsigned int colOffset)
{
    uint32_t const idx = xqa_shared_compute_v102::laneId() / 8;
    uint32_t const idxKEx = idx / 2;
    uint32_t const idxMNEx = idx % 2;
    uint32_t const srcIdxKEx = idxKEx;
    uint32_t const srcIdxMNEx = idxMNEx;
    uint32_t rowLane = xqa_shared_compute_v102::laneId();
    const LdGrain& grain = src.template at<true>(rowOffset + 8 * srcIdxMNEx + rowLane % 8, colOffset + srcIdxKEx);
    const LdGrain* const ptr = &grain;
    LdGrain const data = xqa_shared_compute_v102::ldmatrix_transposed::ldmatrix_4x(warp_1, ptr);
    InstInMat22 dst;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst.data[i / 2][i % 2] = data.data[i];
    }
    return dst;
}
} // namespace load_v_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace load_v_fp8 {
__device__ inline SourceSlice22x2 loadMatrix(const Warp& warp_1, const VSmemBufferFP8& src, unsigned int rowBeg, unsigned int colBeg)
{
    SourceSlice22x2 dst;
    #pragma unroll
    for (uint32_t i = 0; i < 1; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < 2; j++) {
            dst.data[j][i] = xqa_shared_compute_v102::load_v_fp8::loadInstInMat(warp_1, src, rowBeg + 16 * i, colBeg + 2 * j);
        }
    }
    return dst;
}
} // namespace load_v_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace pv_fp8 {
__device__ inline SourceVCacheWordF16 convertVCacheWordToF16(unsigned int i8data)
{
    SourceVCacheWordF16 ret;
    uint32_t src;
    asm("prmt.b32 %0, %1, %2, 0x3120;" : "=r"(src) : "r"(i8data), "r"(0));
    uint32_t dst0;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(dst0) : "h"((uint16_t)src));
    uint32_t dst1;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(dst1) : "h"((uint16_t)(src >> 16)));
    ret.data[0] = dst0;
    ret.data[1] = dst1;
    return ret;
}
} // namespace pv_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace pv_fp8 {
__device__ inline SourceSlice42x2 makeVSlice(const SourceSlice22x2& vSliceOrig)
{
    SourceSlice42x2 ret;
    #pragma unroll
    for (uint32_t m = 0; m < 2; m++) {
        #pragma unroll
        for (uint32_t n = 0; n < 1; n++) {
            const InstInMat22& src = vSliceOrig.data[m][n];
            InstInMat42& dst = ret.data[m][n];
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < 2; j++) {
                    SourceVCacheWordF16 const data = xqa_shared_compute_v102::pv_fp8::convertVCacheWordToF16(src.data[i][j]);
                    #pragma unroll
                    for (uint32_t e = 0; e < 2; e++) {
                        dst.data[i * 2 + e][j] = data.data[e];
                    }
                }
            }
        }
    }
    return ret;
}
} // namespace pv_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace pv_fp8 {
__device__ inline void smemXVPartGemm(const Warp& warp_1, WarpAcc& acc, bool skipXRowRescale, UniformRescaleMask xRowNeedRescaleMask, ThrdRegRowMax xRowScales, const XSmemBuffer& x, unsigned int idxVTilePerXTile, const VSmemBufferFP8& vt, unsigned int idxNSplit)
{
    Half2Scale4 xRowScalesQuad;
    Half2Scale1 const xRowScalesF16 = xqa_shared_compute_v102::half2_convert::__float2half2_rn(xRowScales);
    reinterpret_cast<QuadRegRowMax&>(xRowScalesQuad) = xqa_shared_compute_v102::quad_view::replicateForQuad(warp_1, reinterpret_cast<const ThrdRegRowMax&>(xRowScalesF16));
    #pragma unroll
    for (uint32_t s = 0; s < 2; s++) {
        uint32_t const colBeg = idxVTilePerXTile * 4 + 2 * s;
        SourceSlice22x2 xSlice = xqa_shared_compute_v102::load_x::loadMatrix(warp_1, x, 0, colBeg);
        #pragma unroll
        for (uint32_t m = 0; m < 2; m++) {
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++) {
                uint32_t const r = m * 2 + i;
                #pragma unroll
                for (uint32_t n = 0; n < 1; n++) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; j++) {
                        __half2& elem = reinterpret_cast<__half2&>(xSlice.data[m][n].data[j][i]);
                        elem = ((skipXRowRescale) ? elem : elem * xRowScalesQuad.data[r]);
                    }
                }
            }
        }
        uint32_t const rowBeg = 16 * s;
        SourceSlice22x2 const vSliceOrig = xqa_shared_compute_v102::load_v_fp8::loadMatrix(warp_1, vt, rowBeg, 4 * idxNSplit);
        SourceSlice42x2 const vSlice = xqa_shared_compute_v102::pv_fp8::makeVSlice(vSliceOrig);
        #pragma unroll
        for (uint32_t i_1 = 0; i_1 < 2; i_1++) {
            #pragma unroll
            for (uint32_t j_1 = 0; j_1 < 2; j_1++) {
                const InstInMat42& vInMat = vSlice.data[j_1][0];
                #pragma unroll
                for (uint32_t n_1 = 0; n_1 < 4; n_1++) {
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc.data[i_1][j_1 * 4 + n_1].data)[0][0]), "+f"((acc.data[i_1][j_1 * 4 + n_1].data)[0][1]), "+f"((acc.data[i_1][j_1 * 4 + n_1].data)[1][0]), "+f"((acc.data[i_1][j_1 * 4 + n_1].data)[1][1])
                        : "r"((xSlice.data[i_1][0].data)[0][0]), "r"((xSlice.data[i_1][0].data)[0][1]), "r"((xSlice.data[i_1][0].data)[1][0]), "r"((xSlice.data[i_1][0].data)[1][1]), "r"((reinterpret_cast<const unsigned int (&)[2][1]>(vInMat.data[n_1]))[0][0]), "r"((reinterpret_cast<const unsigned int (&)[2][1]>(vInMat.data[n_1]))[1][0]));
                }
            }
        }
    }
    return;
}
} // namespace pv_fp8
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
namespace output_tile {
__device__ inline void reorderAndStoreGemmOutTile(const Warp& warp_1, XSmemBuffer& dst, const GemmOutRegTile& src)
{
    uint32_t const lane_1 = xqa_shared_compute_v102::laneId();
    #pragma unroll
    for (uint32_t m = 0; m < 4; m++) {
        #pragma unroll
        for (uint32_t n = 0; n < 4; n++) {
            uint32_t const idxRowLocal = xqa_shared_compute_v102::laneId() / 4;
            uint32_t const idxSegLocal = xqa_shared_compute_v102::laneId() % 4;
            SourceOutputSeg seg;
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                seg.data[e] = src.data[m][n * 2 + e];
            }
            SourceOutputReorderedSeg reorderedSeg;
            #pragma unroll
            for (uint32_t e_1 = 0; e_1 < 2; e_1++) {
                reorderedSeg.data[e_1] = seg.data[e_1].x;
                reorderedSeg.data[2 + e_1] = seg.data[e_1].y;
            }
            reinterpret_cast<SourceVCacheWordF16&>(dst.template at<true>(8 * m + idxRowLocal, n * 2 + idxSegLocal / 2).data[idxSegLocal % 2 * 2]) = reinterpret_cast<SourceVCacheWordF16&>(reorderedSeg);
        }
    }
    return;
}
} // namespace output_tile
} // namespace xqa_shared_compute_v102

namespace xqa_shared_compute_v102 {
__device__ inline void copyOutputToGlobalMem(const Warp& warp_1, OutputHead* dst, unsigned int nbQHeads, unsigned int headGrpSize, unsigned int idxHeadGrpOffset, unsigned int nbValidHeadTokens, uint2 dstOffset, const XSmemBuffer& src)
{
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        unsigned int flatIdx = 32 * i + xqa_shared_compute_v102::laneId();
        unsigned int r = flatIdx / 8;
        unsigned int c = flatIdx % 8;
        LdGrain const data = src.template at<true>(r, c);
        unsigned int m = dstOffset.y + r;
        unsigned int n = dstOffset.x / 8 + c;
        if (r >= nbValidHeadTokens) {
            break;
        }
        unsigned int tokenIdx = m / headGrpSize;
        unsigned int headIdx = m % headGrpSize;
        unsigned int idxHead = idxHeadGrpOffset + tokenIdx * nbQHeads + headIdx;
        OutputGrain const outVec = reinterpret_cast<const OutputGrain&>(data);
        reinterpret_cast<OutputGrain*>(dst + idxHead)[n] = outVec;
    }
}
} // namespace xqa_shared_compute_v102

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_sm110_xqa_740ec4c562420f85bf0d(unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const __half* __restrict__ q, const unsigned int* __restrict__ mask, const float* __restrict__ attention_sinks, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale, unsigned int* __restrict__ semaphores, void* __restrict__ scratch)
{
    const int tid = threadIdx.x + 256 * (threadIdx.y + 1 * threadIdx.z);
    const int warp_x = make_warp_uniform(threadIdx.x / 32);
    const int warp_z = make_warp_uniform(threadIdx.z);
    const uint32_t warp = warp_x + 8 * warp_z;
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 166912;
    #define q_ready_addr (mbar_base + 0)
    #define q_reuse_addr (mbar_base + 8)
    #define x_produced_addr (mbar_base + 16)
    #define x_consumed_addr (mbar_base + 24)
    #define v_ready_addr (mbar_base + 144)
    #define v_reuse_addr (mbar_base + 160)
    #define k_partial_addr (mbar_base + 168)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* source_backing = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int source_backing_addr = smem + 0;
    SharedMem& smem_1 = *reinterpret_cast<SharedMem*>(source_backing);
    unsigned int actual_q = q_seq_len;
    unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
    if ((unsigned long long)q_cu_seq_lens != 0) {
        request_offset = q_cu_seq_lens[blockIdx.z];
        actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
    }
    unsigned int q_heads = num_kv_heads * head_group_size;
    unsigned int blocks_per_group = (unsigned int)gridDim.y / num_kv_heads;
    unsigned int head_group = (unsigned int)blockIdx.y / blocks_per_group;
    unsigned int row_begin = (unsigned int)blockIdx.y % blocks_per_group * 32;
    unsigned int _min_0 = ((actual_q * head_group_size - row_begin) < ((unsigned int)32) ? (actual_q * head_group_size - row_begin) : ((unsigned int)32));
    unsigned int valid_rows = ((row_begin <= actual_q * head_group_size) ? _min_0 : (unsigned int)0);
    unsigned int mask_row_halfwords = (q_seq_len + 31) / 32 * 2;
    const unsigned int* mask_request = mask + (request_offset * ((q_seq_len + 31) / 32));

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 25 barriers)
    // Mbarriers at smem_raw[166912..167112)

    // CTA-linear mbarrier record 0: threads=1, record_width=2
    if (tid < 1) {
        // record[0] q_ready: init_count=256
        mbarrier_init_generic(&reinterpret_cast<uint64_t (*)[2]>(smem_raw + 166912)[static_cast<uint32_t>(tid)][0], 256);
        // record[1] q_reuse: init_count=256
        mbarrier_init_generic(&reinterpret_cast<uint64_t (*)[2]>(smem_raw + 166912)[static_cast<uint32_t>(tid)][1], 256);
    }
    // CTA-linear mbarrier record 1: threads=8, record_width=2
    if (tid < 8) {
        // record[0] x_produced: stage=thread_index, init_count=32
        mbarrier_init_generic(&reinterpret_cast<uint64_t (*)[2]>(smem_raw + 166928)[static_cast<uint32_t>(tid)][0], 32);
        // record[1] x_consumed: stage=thread_index, init_count=256
        mbarrier_init_generic(&reinterpret_cast<uint64_t (*)[2]>(smem_raw + 166928)[static_cast<uint32_t>(tid)][1], 256);
    }

    // Kernel pre-init-sync ops
    if (tid < 256) { smem_1.ctaRowMax[0][tid / 32].data[0][(unsigned int)(tid % 32) / 4][tid % 4] = -1e+30f; }
    // CTA-linear mbarrier record 2: threads=2, record_width=1
    if (tid < 2) {
        // record[0] v_ready: stage=thread_index, init_count=256
        mbarrier_init_generic(&reinterpret_cast<uint64_t*>(smem_raw + 167056)[static_cast<uint32_t>(tid)], 256);
    }
    // CTA-linear mbarrier record 3: threads=1, record_width=1
    if (tid < 1) {
        // record[0] v_reuse: init_count=256
        mbarrier_init_generic(&reinterpret_cast<uint64_t*>(smem_raw + 167072)[static_cast<uint32_t>(tid)], 256);
    }
    // CTA-linear mbarrier record 4: threads=4, record_width=1
    if (tid < 4) {
        // record[0] k_partial: stage=thread_index, init_count=32
        mbarrier_init_generic(&reinterpret_cast<uint64_t*>(smem_raw + 167080)[static_cast<uint32_t>(tid)], 32);
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: qk ----
    if (warp_z == 0) {
        { // qk_main
            int warp_id_in_role = warp_x;
            unsigned int warp_0 = warp_id_in_role;
            unsigned int lane_1 = lane;
            unsigned int row = (warp_0 * 32 + lane_1) / 8;
            unsigned int line = (warp_0 * 32 + lane_1) % 8;
            unsigned int head_token = row_begin + row;
            unsigned int source_head = (request_offset + head_token / head_group_size) * q_heads + head_group * head_group_size + head_token % head_group_size;
            if (row < valid_rows) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(q + (source_head * 512 + line * 64)))); }
            QHeadTokenMap const localQHeadTokenIdxMap{q_heads, head_group_size, row_begin};
            uint32_t const idxHeadTokenBeg = q_heads * request_offset + head_group * head_group_size;
            TinyPtrConstIOHead const q_src{reinterpret_cast<const OutputHead*>(q), idxHeadTokenBeg};
            QSmemBuffer& q_dst = reinterpret_cast<QSmemBuffer*>(smem_1.q[0] + 0)[0];
            if (valid_rows == 32) {
                xqa_shared_compute_v102::copyHeadsAsyncMultiWarp<true>(warp_0, q_dst, q_src, valid_rows, localQHeadTokenIdxMap);
            } else {
                xqa_shared_compute_v102::copyHeadsAsyncMultiWarp<false>(warp_0, q_dst, q_src, valid_rows, localQHeadTokenIdxMap);
            }
            asm volatile(
                "{\n\t"
                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                "}"
                :: "r"(q_ready_addr) : "memory");
            unsigned int _vec_load_0[1];
            {
                uint32_t _scalar_bits_0;
                uint64_t _l2_evict_last_policy_1;
                asm("createpolicy.fractional.L2::evict_last.b64 %0;"
                    : "=l"(_l2_evict_last_policy_1));
                asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
                    : "=r"(_scalar_bits_0)
                    : "l"((const void*)(kv_cache_list.sequence_lengths + (blockIdx.z))), "l"(_l2_evict_last_policy_1));
                _vec_load_0[0] = (unsigned int)_scalar_bits_0;
            }
            unsigned int seq_iters = (_vec_load_0[0] + 511) / 512;
            unsigned int k_prefetch_base = 0;
            unsigned int slice_rows = (_vec_load_0[0] + blocks_per_group - 1) / blocks_per_group;
            unsigned int slice_begin = (unsigned int)blockIdx.y % blocks_per_group * slice_rows;
            unsigned int _min_1 = ((slice_begin + slice_rows) < (_vec_load_0[0]) ? (slice_begin + slice_rows) : (_vec_load_0[0]));
            unsigned int slice_end = ((slice_begin < _vec_load_0[0]) ? _min_1 : slice_begin);
            unsigned int slice_lines = (slice_end - slice_begin) * (unsigned int)(512 * ((1) ? 1 : 2) / 128);
            for (unsigned int it = 0; it < (slice_lines + 255) / 256; it++) {
                unsigned int index = it * 256 + (warp_0 * 32 + lane_1);
                unsigned int row_0 = slice_begin + index / (unsigned int)(512 * ((1) ? 1 : 2) / 128);
                unsigned int line_1 = index % (unsigned int)(512 * ((1) ? 1 : 2) / 128);
                bool valid = index < slice_lines;
                unsigned int head_row = k_prefetch_base + row_0;
                unsigned int page_slot = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + row_0 / 128;
                int _vec_load_1[1];
                {
                    _vec_load_1[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + ((valid) ? page_slot : (unsigned int)0));
                }
                int page_id = _vec_load_1[0];
                valid = valid && page_id >= 0;
                head_row = ((valid) ? ((unsigned int)page_id * 128 + row_0 % 128) * num_kv_heads + head_group : (unsigned int)0);
                if (valid) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(reinterpret_cast<const uint8_t*>(kv_cache_list.pool) + (head_row * 512 + line_1 * (unsigned int)(128 / ((1) ? 1 : 2)))))); }
            }
            unsigned int mask_begin = (_vec_load_0[0] - actual_q) / 512;
            float qk_scale = attention_scale;
            qk_scale = qk_scale * k_cache_scale;
            unsigned int cache_base = 0;
            int page[1];
            unsigned int page_index[1];
            bool help_first = warp_0 >= 4 && _vec_load_0[0] <= warp_0 * 64 && _vec_load_0[0] > (warp_0 - 4) * 64;
            unsigned int first_warp = ((help_first) ? warp_0 - 4 : warp_0);
            unsigned int first_part = ((help_first) ? (unsigned int)4 : (unsigned int)0);
            page_index[0] = first_warp / 2;
            if (page_index[0] < (_vec_load_0[0] + 127) / 128) {
                unsigned int page_list_offset = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + page_index[0];
                int _vec_load_2[1];
                {
                    _vec_load_2[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset);
                }
                page[0] = _vec_load_2[0];
            } else {
                page[0] = -1;
            }
            uint32_t const seqOffset = 64 * first_warp;
            uint32_t const idxHeadBeg = (seqOffset & 127) * num_kv_heads + head_group;
            uint32_t const nbHeadsAvail = ((seqOffset < _vec_load_0[0]) ? _vec_load_0[0] - seqOffset : 0);
            HeadPtrConstFP8Page1 const k_src = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page[0]}}, num_kv_heads, idxHeadBeg, 0, 7);
            KSmemBuffer& k_dst = reinterpret_cast<KSmemBuffer*>(smem_1.k[warp_0] + 0)[0];
            if (1 < seq_iters) {
                xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<true>(Warp{}, k_dst, 0, k_src, first_part, 64, KIdentityHeadMap{});
            } else {
                xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<false>(Warp{}, k_dst, 0, k_src, first_part, nbHeadsAvail, KIdentityHeadMap{});
            }
            asm volatile("cp.async.commit_group;");
            mbarrier_wait_hint(q_ready_addr, 0, 4294967295u);
            unsigned int row_2 = lane;
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                unsigned int grain = 2 * warp_0 + (unsigned int)(16 * n);
                LdGrain const lo = reinterpret_cast<const LdGrain*>(&(smem_1.q[0][0]).template at<true>(row_2, grain))[0];
                LdGrain const hi = reinterpret_cast<const LdGrain*>(&(smem_1.q[0][0]).template at<true>(row_2, grain + 1))[0];
                (&(smem_1.q[0][0]).template at<true>(row_2, grain))[0] = LdGrain{{lo.data[0], lo.data[2], hi.data[0], hi.data[2]}};
                (&(smem_1.q[0][0]).template at<true>(row_2, grain + 1))[0] = LdGrain{{lo.data[1], lo.data[3], hi.data[1], hi.data[3]}};
            }
            unsigned long long _mbarrier_arrival_token_2;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_2)
                : "l"(smem_raw + ((q_ready_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            mbarrier_wait_hint(q_ready_addr, 1, 4294967295u);
            for (unsigned int seq = 0; seq < seq_iters; seq++) {
                bool idle = _vec_load_0[0] <= seq * 512 + warp_0 * 64;
                bool partner_active = warp_0 >= 4 && _vec_load_0[0] > seq * 512 + (warp_0 - 4) * 64;
                bool help = idle && partner_active;
                if (idle && !partner_active) {
                    break;
                }
                bool helped = warp_0 < 4 && _vec_load_0[0] <= seq * 512 + (warp_0 + 4) * 64;
                bool finish_half = help || helped;
                unsigned int token_warp = ((help) ? warp_0 - 4 : warp_0);
                unsigned int part_base = ((help) ? (unsigned int)4 : (unsigned int)0);
                if (help && seq > 0) {
                    asm volatile("cp.async.wait_group 0;");
                    page_index[0] = token_warp / 2 + seq * 4;
                    if (page_index[0] < (_vec_load_0[0] + 127) / 128) {
                        unsigned int page_list_offset_1 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + page_index[0];
                        int _vec_load_3[1];
                        {
                            _vec_load_3[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset_1);
                        }
                        page[0] = _vec_load_3[0];
                    } else {
                        page[0] = -1;
                    }
                    uint32_t const seqOffset_1 = 512 * seq + 64 * token_warp;
                    uint32_t const idxHeadBeg_1 = (seqOffset_1 & 127) * num_kv_heads + head_group;
                    uint32_t const nbHeadsAvail_1 = ((seqOffset_1 < _vec_load_0[0]) ? _vec_load_0[0] - seqOffset_1 : 0);
                    HeadPtrConstFP8Page1 const k_src_1 = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page[0]}}, num_kv_heads, idxHeadBeg_1, 0, 7);
                    KSmemBuffer& k_dst_1 = reinterpret_cast<KSmemBuffer*>(smem_1.k[warp_0] + 0)[0];
                    if (seq + 1 < seq_iters) {
                        xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<true>(Warp{}, k_dst_1, 0, k_src_1, part_base, 64, KIdentityHeadMap{});
                    } else {
                        xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<false>(Warp{}, k_dst_1, 0, k_src_1, part_base, nbHeadsAvail_1, KIdentityHeadMap{});
                    }
                    asm volatile("cp.async.commit_group;");
                }
                WarpAcc acc{};
                #pragma unroll 1
                for (unsigned int part = 0; part < 4; part++) {
                    bool last_of_half = part == 3;
                    unsigned int next_seq = ((last_of_half) ? ((finish_half) ? seq + 1 : seq) : seq);
                    unsigned int next_part = ((last_of_half) ? ((finish_half) ? (unsigned int)0 : (unsigned int)4) : part_base + part + 1);
                    uint32_t const seqOffset_2 = 512 * next_seq + 64 * token_warp;
                    uint32_t const idxHeadBeg_2 = (seqOffset_2 & 127) * num_kv_heads + head_group;
                    uint32_t const nbHeadsAvail_2 = ((seqOffset_2 < _vec_load_0[0]) ? _vec_load_0[0] - seqOffset_2 : 0);
                    HeadPtrConstFP8Page1 const k_src_2 = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page[0]}}, num_kv_heads, idxHeadBeg_2, 0, 7);
                    KSmemBuffer& k_dst_2 = reinterpret_cast<KSmemBuffer*>(smem_1.k[warp_0] + (part + 1) % 2)[0];
                    if (next_seq + 1 < seq_iters) {
                        xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<true>(Warp{}, k_dst_2, 0, k_src_2, next_part, 64, KIdentityHeadMap{});
                    } else {
                        xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<false>(Warp{}, k_dst_2, 0, k_src_2, next_part, nbHeadsAvail_2, KIdentityHeadMap{});
                    }
                    if (finish_half && part == 2) {
                        page_index[0] = page_index[0] + 4;
                        if (page_index[0] < (_vec_load_0[0] + 127) / 128) {
                            unsigned int page_list_offset_2 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + page_index[0];
                            int _vec_load_4[1];
                            {
                                _vec_load_4[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset_2);
                            }
                            page[0] = _vec_load_4[0];
                        } else {
                            page[0] = -1;
                        }
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 1;");
                    if (part == 0) {
                    }
                    const QSmemBuffer& smemQ = reinterpret_cast<const QSmemBuffer*>(smem_1.q[0] + 0)[0];
                    const KSmemBuffer& smemKPart = reinterpret_cast<const KSmemBuffer*>(smem_1.k[warp_0] + part % 2)[0];
                    uint32_t const smemQOffset = (part_base + part) * 8;
                    xqa_shared_compute_v102::qk_fp8::smemQKPartGemm(Warp{}, acc, smemQ, smemQOffset, smemKPart);
                }
                if (!finish_half) {
                    #pragma unroll 1
                    for (unsigned int part_1 = 4; part_1 < 8; part_1++) {
                        uint32_t const seqOffset_3 = 512 * (seq + (part_1 + 1) / 8) + 64 * warp_0;
                        uint32_t const idxHeadBeg_3 = (seqOffset_3 & 127) * num_kv_heads + head_group;
                        uint32_t const nbHeadsAvail_3 = ((seqOffset_3 < _vec_load_0[0]) ? _vec_load_0[0] - seqOffset_3 : 0);
                        HeadPtrConstFP8Page1 const k_src_3 = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page[0]}}, num_kv_heads, idxHeadBeg_3, 0, 7);
                        KSmemBuffer& k_dst_3 = reinterpret_cast<KSmemBuffer*>(smem_1.k[warp_0] + (part_1 + 1) % 2)[0];
                        if (seq + (part_1 + 1) / 8 + 1 < seq_iters) {
                            xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<true>(Warp{}, k_dst_3, 0, k_src_3, (part_1 + 1) % 8, 64, KIdentityHeadMap{});
                        } else {
                            xqa_shared_compute_v102::k_fp8_page128::copyPartialHeadsAsync<false>(Warp{}, k_dst_3, 0, k_src_3, (part_1 + 1) % 8, nbHeadsAvail_3, KIdentityHeadMap{});
                        }
                        if ((part_1 + 1) % 8 == 7) {
                            page_index[0] = page_index[0] + 4;
                            if (page_index[0] < (_vec_load_0[0] + 127) / 128) {
                                unsigned int page_list_offset_3 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + page_index[0];
                                int _vec_load_5[1];
                                {
                                    _vec_load_5[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset_3);
                                }
                                page[0] = _vec_load_5[0];
                            } else {
                                page[0] = -1;
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 1;");
                        const QSmemBuffer& smemQ_1 = reinterpret_cast<const QSmemBuffer*>(smem_1.q[0] + 0)[0];
                        const KSmemBuffer& smemKPart_1 = reinterpret_cast<const KSmemBuffer*>(smem_1.k[warp_0] + part_1 % 2)[0];
                        uint32_t const smemQOffset_1 = part_1 * 8;
                        xqa_shared_compute_v102::qk_fp8::smemQKPartGemm(Warp{}, acc, smemQ_1, smemQOffset_1, smemKPart_1);
                    }
                }
                if (help) {
                    asm volatile("cp.async.wait_group 0;");
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][0].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[512 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][1].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[1024 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][2].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[1536 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][3].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[2048 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][4].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[2560 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][5].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[3072 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][6].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[3584 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[0][7].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[4096 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][0].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[4608 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][1].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[5120 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][2].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[5632 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][3].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[6144 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][4].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[6656 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][5].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[7168 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][6].data[0] + 0)[0];
                    reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0] + 0)[7680 + lane_1 * 16])[0] = reinterpret_cast<int4*>(acc.data[1][7].data[0] + 0)[0];
                    unsigned long long _mbarrier_arrival_token_3;
                    asm volatile(
                        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                        : "=l"(_mbarrier_arrival_token_3)
                        : "l"(smem_raw + ((k_partial_addr + (warp_0 - 4) * 8) - smem)), "r"((uint32_t)(1)) : "memory");
                    break;
                }
                float partial_scratch[4];
                if (helped) {
                    mbarrier_wait_hint(k_partial_addr + (warp_0) * 8, 0, 4294967295u);
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[lane_1 * 16])[0];
                    acc.data[0][0].data[0][0] = acc.data[0][0].data[0][0] + partial_scratch[0];
                    acc.data[0][0].data[0][1] = acc.data[0][0].data[0][1] + partial_scratch[1];
                    acc.data[0][0].data[1][0] = acc.data[0][0].data[1][0] + partial_scratch[2];
                    acc.data[0][0].data[1][1] = acc.data[0][0].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[512 + lane_1 * 16])[0];
                    acc.data[0][1].data[0][0] = acc.data[0][1].data[0][0] + partial_scratch[0];
                    acc.data[0][1].data[0][1] = acc.data[0][1].data[0][1] + partial_scratch[1];
                    acc.data[0][1].data[1][0] = acc.data[0][1].data[1][0] + partial_scratch[2];
                    acc.data[0][1].data[1][1] = acc.data[0][1].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[1024 + lane_1 * 16])[0];
                    acc.data[0][2].data[0][0] = acc.data[0][2].data[0][0] + partial_scratch[0];
                    acc.data[0][2].data[0][1] = acc.data[0][2].data[0][1] + partial_scratch[1];
                    acc.data[0][2].data[1][0] = acc.data[0][2].data[1][0] + partial_scratch[2];
                    acc.data[0][2].data[1][1] = acc.data[0][2].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[1536 + lane_1 * 16])[0];
                    acc.data[0][3].data[0][0] = acc.data[0][3].data[0][0] + partial_scratch[0];
                    acc.data[0][3].data[0][1] = acc.data[0][3].data[0][1] + partial_scratch[1];
                    acc.data[0][3].data[1][0] = acc.data[0][3].data[1][0] + partial_scratch[2];
                    acc.data[0][3].data[1][1] = acc.data[0][3].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[2048 + lane_1 * 16])[0];
                    acc.data[0][4].data[0][0] = acc.data[0][4].data[0][0] + partial_scratch[0];
                    acc.data[0][4].data[0][1] = acc.data[0][4].data[0][1] + partial_scratch[1];
                    acc.data[0][4].data[1][0] = acc.data[0][4].data[1][0] + partial_scratch[2];
                    acc.data[0][4].data[1][1] = acc.data[0][4].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[2560 + lane_1 * 16])[0];
                    acc.data[0][5].data[0][0] = acc.data[0][5].data[0][0] + partial_scratch[0];
                    acc.data[0][5].data[0][1] = acc.data[0][5].data[0][1] + partial_scratch[1];
                    acc.data[0][5].data[1][0] = acc.data[0][5].data[1][0] + partial_scratch[2];
                    acc.data[0][5].data[1][1] = acc.data[0][5].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[3072 + lane_1 * 16])[0];
                    acc.data[0][6].data[0][0] = acc.data[0][6].data[0][0] + partial_scratch[0];
                    acc.data[0][6].data[0][1] = acc.data[0][6].data[0][1] + partial_scratch[1];
                    acc.data[0][6].data[1][0] = acc.data[0][6].data[1][0] + partial_scratch[2];
                    acc.data[0][6].data[1][1] = acc.data[0][6].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[3584 + lane_1 * 16])[0];
                    acc.data[0][7].data[0][0] = acc.data[0][7].data[0][0] + partial_scratch[0];
                    acc.data[0][7].data[0][1] = acc.data[0][7].data[0][1] + partial_scratch[1];
                    acc.data[0][7].data[1][0] = acc.data[0][7].data[1][0] + partial_scratch[2];
                    acc.data[0][7].data[1][1] = acc.data[0][7].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[4096 + lane_1 * 16])[0];
                    acc.data[1][0].data[0][0] = acc.data[1][0].data[0][0] + partial_scratch[0];
                    acc.data[1][0].data[0][1] = acc.data[1][0].data[0][1] + partial_scratch[1];
                    acc.data[1][0].data[1][0] = acc.data[1][0].data[1][0] + partial_scratch[2];
                    acc.data[1][0].data[1][1] = acc.data[1][0].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[4608 + lane_1 * 16])[0];
                    acc.data[1][1].data[0][0] = acc.data[1][1].data[0][0] + partial_scratch[0];
                    acc.data[1][1].data[0][1] = acc.data[1][1].data[0][1] + partial_scratch[1];
                    acc.data[1][1].data[1][0] = acc.data[1][1].data[1][0] + partial_scratch[2];
                    acc.data[1][1].data[1][1] = acc.data[1][1].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[5120 + lane_1 * 16])[0];
                    acc.data[1][2].data[0][0] = acc.data[1][2].data[0][0] + partial_scratch[0];
                    acc.data[1][2].data[0][1] = acc.data[1][2].data[0][1] + partial_scratch[1];
                    acc.data[1][2].data[1][0] = acc.data[1][2].data[1][0] + partial_scratch[2];
                    acc.data[1][2].data[1][1] = acc.data[1][2].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[5632 + lane_1 * 16])[0];
                    acc.data[1][3].data[0][0] = acc.data[1][3].data[0][0] + partial_scratch[0];
                    acc.data[1][3].data[0][1] = acc.data[1][3].data[0][1] + partial_scratch[1];
                    acc.data[1][3].data[1][0] = acc.data[1][3].data[1][0] + partial_scratch[2];
                    acc.data[1][3].data[1][1] = acc.data[1][3].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[6144 + lane_1 * 16])[0];
                    acc.data[1][4].data[0][0] = acc.data[1][4].data[0][0] + partial_scratch[0];
                    acc.data[1][4].data[0][1] = acc.data[1][4].data[0][1] + partial_scratch[1];
                    acc.data[1][4].data[1][0] = acc.data[1][4].data[1][0] + partial_scratch[2];
                    acc.data[1][4].data[1][1] = acc.data[1][4].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[6656 + lane_1 * 16])[0];
                    acc.data[1][5].data[0][0] = acc.data[1][5].data[0][0] + partial_scratch[0];
                    acc.data[1][5].data[0][1] = acc.data[1][5].data[0][1] + partial_scratch[1];
                    acc.data[1][5].data[1][0] = acc.data[1][5].data[1][0] + partial_scratch[2];
                    acc.data[1][5].data[1][1] = acc.data[1][5].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[7168 + lane_1 * 16])[0];
                    acc.data[1][6].data[0][0] = acc.data[1][6].data[0][0] + partial_scratch[0];
                    acc.data[1][6].data[0][1] = acc.data[1][6].data[0][1] + partial_scratch[1];
                    acc.data[1][6].data[1][0] = acc.data[1][6].data[1][0] + partial_scratch[2];
                    acc.data[1][6].data[1][1] = acc.data[1][6].data[1][1] + partial_scratch[3];
                    reinterpret_cast<int4*>(partial_scratch + 0)[0] = reinterpret_cast<int4*>(&reinterpret_cast<uint8_t*>(smem_1.k[warp_0 + 4] + 0)[7680 + lane_1 * 16])[0];
                    acc.data[1][7].data[0][0] = acc.data[1][7].data[0][0] + partial_scratch[0];
                    acc.data[1][7].data[0][1] = acc.data[1][7].data[0][1] + partial_scratch[1];
                    acc.data[1][7].data[1][0] = acc.data[1][7].data[1][0] + partial_scratch[2];
                    acc.data[1][7].data[1][1] = acc.data[1][7].data[1][1] + partial_scratch[3];
                }
                if (qk_scale != 1.0f) {
                    xqa_shared_compute_v102::rescaleAcc(Warp{}, acc, qk_scale);
                }
                mbarrier_wait_hint(x_consumed_addr + (warp_0) * 16, seq % 2, 4294967295u);
                QuadRegRowMax initRowMaxQuad = reinterpret_cast<const QuadRegRowMax*>(smem_1.ctaRowMax[0][warp_0].data[0][lane_1 / 4 * 4 / 4] + lane_1 / 4 * 4 % 4)[0];
                if (mask_begin <= seq) {
                    unsigned int valid_cols = _vec_load_0[0] - (seq * 512 + warp_0 * 64);
                    #pragma unroll
                    for (int m = 0; m < 2; m++) {
                        #pragma unroll
                        for (int i = 0; i < 2; i++) {
                            unsigned int _min_2 = (((row_begin + (unsigned int)(m * 16) + lane_1 / 4 + (unsigned int)(i * 8)) / head_group_size) < (actual_q - 1) ? ((row_begin + (unsigned int)(m * 16) + lane_1 / 4 + (unsigned int)(i * 8)) / head_group_size) : (actual_q - 1));
                            unsigned int token_row = _min_2;
                            #pragma unroll
                            for (int mask_n = 0; mask_n < 4; mask_n++) {
                                unsigned int first_col = (unsigned int)(mask_n * 16) + lane_1 % 4 * 2;
                                unsigned int last_col = first_col + 9;
                                unsigned int _min_3 = ((first_col + actual_q - valid_cols) < (actual_q - 1) ? (first_col + actual_q - valid_cols) : (actual_q - 1));
                                unsigned int pos0 = ((valid_cols > first_col + actual_q) ? (unsigned int)0 : _min_3);
                                unsigned int _min_4 = ((last_col + actual_q - valid_cols) < (actual_q - 1) ? (last_col + actual_q - valid_cols) : (actual_q - 1));
                                unsigned int pos1 = ((valid_cols > last_col + actual_q) ? (unsigned int)0 : _min_4);
                                uint16_t _vec_load_6[1];
                                {
                                    _vec_load_6[0] = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint16_t*>(mask_request) + (token_row * mask_row_halfwords + pos0 / 16));
                                }
                                uint16_t _vec_load_7[1];
                                {
                                    _vec_load_7[0] = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint16_t*>(mask_request) + (token_row * mask_row_halfwords + pos1 / 16));
                                }
                                uint32_t _pack_u16x2_0;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_0) : "h"(_vec_load_6[0]), "h"(_vec_load_7[0]));
                                unsigned int packed = _pack_u16x2_0;
                                #pragma unroll
                                for (int nj = 0; nj < 2; nj++) {
                                    #pragma unroll
                                    for (int j = 0; j < 2; j++) {
                                        unsigned int col = (unsigned int)((mask_n * 2 + nj) * 8) + lane_1 % 4 * 2 + (unsigned int)j;
                                        bool visible = ((valid_cols > col + actual_q) ? (unsigned int)1 : packed & (unsigned int)1 << col + actual_q - valid_cols - pos0 / 16 * 16) != 0;
                                        acc.data[m][mask_n * 2 + nj].data[i][j] = ((visible && col < valid_cols) ? acc.data[m][mask_n * 2 + nj].data[i][j] : -CUDART_INF_F);
                                    }
                                }
                            }
                        }
                    }
                }
                QuadRegRowMax const regRowMax = xqa_shared_compute_v102::warpTileOnlineSoftmax(Warp{}, initRowMaxQuad, acc);
                GemmOutRegTile const fp16Acc = xqa_shared_compute_v102::output_tile::toFp16(acc);
                QuadRegRowMax const regRowSum = xqa_shared_compute_v102::output_tile::computeRowSum(Warp{}, fp16Acc);
                xqa_shared_compute_v102::output_tile::storeOrderedGemmOutTile(Warp{}, smem_1.x[0][warp_0], fp16Acc);
                if (lane_1 % 4 == 0) {
                    reinterpret_cast<int4*>(smem_1.warpRowMax[0][warp_0].data[0][lane_1 / 4 * 4 / 4] + lane_1 / 4 * 4 % 4)[0] = reinterpret_cast<int4 const*>(regRowMax.data + 0)[0];
                    reinterpret_cast<int4*>(smem_1.warpRowSum[0][warp_0].data[0][lane_1 / 4 * 4 / 4] + lane_1 / 4 * 4 % 4)[0] = reinterpret_cast<int4 const*>(regRowSum.data + 0)[0];
                }
                unsigned long long _mbarrier_arrival_token_4;
                asm volatile(
                    "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                    : "=l"(_mbarrier_arrival_token_4)
                    : "l"(smem_raw + ((x_produced_addr + (warp_0) * 16) - smem)), "r"((uint32_t)(1)) : "memory");
            }
        }
    // ---- Role: pv ----
    } else if (warp_z == 1) {
        { // pv_main
            int warp_id_in_role_1 = warp_x;
            unsigned int warp_0_1 = warp_id_in_role_1;
            unsigned int lane_1_1 = lane;
            unsigned int out_row = (warp_0_1 * 32 + lane_1_1) / 8;
            unsigned int out_line = (warp_0_1 * 32 + lane_1_1) % 8;
            unsigned int out_token = row_begin + out_row;
            unsigned int out_head_row = (request_offset + out_token / head_group_size) * q_heads + head_group * head_group_size + out_token % head_group_size;
            if (out_row < valid_rows) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(output + (out_head_row * 512 + out_line * 64)))); }
            unsigned int row_1 = (warp_0_1 * 32 + lane_1_1) / 8;
            unsigned int line_2 = (warp_0_1 * 32 + lane_1_1) % 8;
            unsigned int head_token_1 = row_begin + row_1;
            unsigned int source_head_1 = (request_offset + head_token_1 / head_group_size) * q_heads + head_group * head_group_size + head_token_1 % head_group_size;
            if (row_1 < valid_rows) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(q + (source_head_1 * 512 + line_2 * 64)))); }
            unsigned int _vec_load_8[1];
            {
                uint32_t _scalar_bits_0;
                uint64_t _l2_evict_last_policy_1;
                asm("createpolicy.fractional.L2::evict_last.b64 %0;"
                    : "=l"(_l2_evict_last_policy_1));
                asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
                    : "=r"(_scalar_bits_0)
                    : "l"((const void*)(kv_cache_list.sequence_lengths + (blockIdx.z))), "l"(_l2_evict_last_policy_1));
                _vec_load_8[0] = (unsigned int)_scalar_bits_0;
            }
            unsigned int seq_iters_1 = (_vec_load_8[0] + 511) / 512;
            unsigned long long _mbarrier_arrival_token_2;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_2)
                : "l"(smem_raw + ((x_consumed_addr + (warp_0_1) * 16) - smem)), "r"((uint32_t)(8)) : "memory");
            unsigned int v_prefetch_base = 0;
            unsigned int slice_rows_1 = (_vec_load_8[0] + blocks_per_group - 1) / blocks_per_group;
            unsigned int slice_begin_1 = (unsigned int)blockIdx.y % blocks_per_group * slice_rows_1;
            unsigned int _min_5 = ((slice_begin_1 + slice_rows_1) < (_vec_load_8[0]) ? (slice_begin_1 + slice_rows_1) : (_vec_load_8[0]));
            unsigned int slice_end_1 = ((slice_begin_1 < _vec_load_8[0]) ? _min_5 : slice_begin_1);
            unsigned int slice_lines_1 = (slice_end_1 - slice_begin_1) * (unsigned int)(512 * ((1) ? 1 : 2) / 128);
            for (unsigned int it_1 = 0; it_1 < (slice_lines_1 + 255) / 256; it_1++) {
                unsigned int index_1 = it_1 * 256 + (warp_0_1 * 32 + lane_1_1);
                unsigned int row_0_1 = slice_begin_1 + index_1 / (unsigned int)(512 * ((1) ? 1 : 2) / 128);
                unsigned int line_1_1 = index_1 % (unsigned int)(512 * ((1) ? 1 : 2) / 128);
                bool valid_1 = index_1 < slice_lines_1;
                unsigned int head_row_1 = v_prefetch_base + row_0_1;
                unsigned int page_slot_1 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + row_0_1 / 128;
                int _vec_load_9[1];
                {
                    _vec_load_9[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + ((valid_1) ? page_slot_1 : (unsigned int)0));
                }
                int page_id_1 = _vec_load_9[0];
                valid_1 = valid_1 && page_id_1 >= 0;
                head_row_1 = ((valid_1) ? ((unsigned int)page_id_1 * 128 + row_0_1 % 128) * num_kv_heads + head_group : (unsigned int)0);
                if (valid_1) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(reinterpret_cast<const uint8_t*>(kv_cache_list.pool) + (head_row_1 * 512 + line_1_1 * (unsigned int)(128 / ((1) ? 1 : 2)))))); }
            }
            unsigned int cache_base_1 = 0;
            int page_1[1];
            unsigned int page_index_1[1];
            page_index_1[0] = 0;
            if (page_index_1[0] < (_vec_load_8[0] + 127) / 128) {
                unsigned int page_list_offset_4 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + page_index_1[0];
                int _vec_load_10[1];
                {
                    _vec_load_10[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset_4);
                }
                page_1[0] = _vec_load_10[0];
            } else {
                page_1[0] = -1;
            }
            VSmemBufferFP8& v_dst = reinterpret_cast<VSmemBufferFP8*>(smem_1.v[0][0] + 0)[0];
            uint32_t const seqOffset_4 = 0;
            uint32_t const idxHeadBeg_4 = (seqOffset_4 & 127) * num_kv_heads + head_group;
            HeadPtrConstFP8Page1 const v_src = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page_1[0]}}, num_kv_heads, idxHeadBeg_4, 0, 7);
            uint32_t const nbHeadsAvail_4 = ((1 < seq_iters_1) ? 32 : ((seqOffset_4 < _vec_load_8[0]) ? _vec_load_8[0] - seqOffset_4 : 0));
            xqa_shared_compute_v102::v_fp8_page128::copyHeadsAsync(warp_0_1, v_dst, v_src, nbHeadsAvail_4, KIdentityHeadMap{});
            asm volatile(
                "{\n\t"
                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                "}"
                :: "r"(v_ready_addr) : "memory");
            float maximum = -1e+30f;
            float total = 0.0f;
            WarpAcc acc_1{};
            unsigned long long _mbarrier_arrival_token_3;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_3)
                : "l"(smem_raw + ((v_reuse_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            unsigned int stat_index = lane_1_1 % 8 * 4 + lane_1_1 / 8;
            bool x_produced_phase = 0;
            for (unsigned int seq_1 = 0; seq_1 < seq_iters_1; seq_1++) {
                #pragma unroll
                for (int tile = 0; tile < 8; tile++) {
                    if (_vec_load_8[0] <= seq_1 * 512 + (unsigned int)(tile * 64)) {
                        break;
                    }
                    const XSmemBuffer& smemXTile = reinterpret_cast<const XSmemBuffer*>(smem_1.x[0] + tile)[0];
                    ThrdRegRowMax xRowScales;
                    UniformRescaleMask xRowNeedRescaleMask;
                    bool skip_scale;
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        uint32_t _mbar_token_0 = mbarrier_test_wait(v_reuse_addr, half);
                        mbarrier_wait_token_hint(v_reuse_addr, half, _mbar_token_0, 4294967295u);
                        VSmemBufferFP8& v_dst_1 = reinterpret_cast<VSmemBufferFP8*>(smem_1.v[0][0] + (half + 1) % 2)[0];
                        uint32_t const seqOffset_5 = 512 * (seq_1 + (unsigned int)((tile + (half + 1) / 2) / 8)) + 64 * (unsigned int)((tile + (half + 1) / 2) % 8) + 32 * (unsigned int)((half + 1) % 2);
                        uint32_t const idxHeadBeg_5 = (seqOffset_5 & 127) * num_kv_heads + head_group;
                        HeadPtrConstFP8Page1 const v_src_1 = xqa_shared_compute_v102::k_fp8_page128::makePagedHeadPtr(reinterpret_cast<const CacheHeadFp8*>(reinterpret_cast<const uint8_t*>(kv_cache_list.pool)), KVCachePageIndices1{{page_1[0]}}, num_kv_heads, idxHeadBeg_5, 0, 7);
                        uint32_t const nbHeadsAvail_5 = ((seq_1 + (unsigned int)((tile + (half + 1) / 2) / 8) + 1 < seq_iters_1) ? 32 : ((seqOffset_5 < _vec_load_8[0]) ? _vec_load_8[0] - seqOffset_5 : 0));
                        xqa_shared_compute_v102::v_fp8_page128::copyHeadsAsync(warp_0_1, v_dst_1, v_src_1, nbHeadsAvail_5, KIdentityHeadMap{});
                        if ((tile + (half + 1) / 2) % 8 % 2 == 1 && (half + 1) % 2 == 1) {
                            page_index_1[0] = page_index_1[0] + 1;
                            if (page_index_1[0] < (_vec_load_8[0] + 127) / 128) {
                                unsigned int page_list_offset_5 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + page_index_1[0];
                                int _vec_load_11[1];
                                {
                                    _vec_load_11[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + page_list_offset_5);
                                }
                                page_1[0] = _vec_load_11[0];
                            } else {
                                page_1[0] = -1;
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(v_ready_addr + ((half + 1) % 2) * 8) : "memory");
                        uint32_t _mbar_token_1;
                        asm volatile("{ .reg .pred p; "
                            "mbarrier.test_wait.parity.acquire.cta.b64 p, [%1], %2; "
                            "selp.b32 %0, 1, 0, p; }"
                            : "=r"(_mbar_token_1) : "l"(&smem_1.otherBarriers[half].mBar), "r"(tile % 2) : "memory");
                        if (half == 0) {
                            mbarrier_wait_hint(x_produced_addr + (tile) * 16, x_produced_phase, 4294967295u);
                            if (tile == 0) {
                            }
                            float tile_max[1];
                            float tile_sum[1];
                            const void* _smem_load_vec_ptr_4 = reinterpret_cast<const void*>(smem_1.warpRowMax[0][tile].data[0][stat_index / 4] + stat_index % 4);
                            uint32_t _smem_load_vec_addr_4 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_vec_ptr_4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tile_max[0])) : "r"(_smem_load_vec_addr_4));
                            const void* _smem_load_vec_ptr_5 = reinterpret_cast<const void*>(smem_1.warpRowSum[0][tile].data[0][stat_index / 4] + stat_index % 4);
                            uint32_t _smem_load_vec_addr_5 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_vec_ptr_5));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tile_sum[0])) : "r"(_smem_load_vec_addr_5));
                            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, tile_max[0] != maximum);
                            unsigned int need = _vote_0;
                            if (need == 0) {
                                skip_scale = 1;
                            } else {
                                float previous = maximum;
                                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, tile_max[0] > previous);
                                unsigned int acc_need = _vote_1;
                                unsigned int x_need = need & ~acc_need;
                                xRowNeedRescaleMask.data[0] = x_need;
                                float _fmax_0 = fmaxf(previous, tile_max[0]);
                                maximum = _fmax_0;
                                skip_scale = x_need == 0;
                                if ((unsigned int)tile == warp_0_1) {
                                    smem_1.ctaRowMax[0][tile].data[0][stat_index / 4][stat_index % 4] = maximum;
                                }
                                float _expf_0 = __expf(previous - maximum);
                                float acc_scale = _expf_0;
                                total = total * acc_scale;
                                UniformRescaleMask const accRowNeedRescaleMask{{acc_need}};
                                ThrdRegRowMax const accRowScales{{acc_scale}};
                                xqa_shared_compute_v102::rescaleAcc(Warp{}, acc_1, accRowNeedRescaleMask, accRowScales);
                                if (skip_scale) {
                                    xRowScales = xRowScales;
                                } else {
                                    float _expf_1 = __expf(tile_max[0] - maximum);
                                    xRowScales = ThrdRegRowMax{{_expf_1}};
                                }
                                tile_sum[0] = ((skip_scale) ? tile_sum[0] : tile_sum[0] * xRowScales.data[0]);
                            }
                            total = total + tile_sum[0];
                        }
                        mbarrier_wait_token_generic_hint(&smem_1.otherBarriers[half].mBar, tile % 2, _mbar_token_1, 4294967295u);
                        if (tile == 0 && half == 0) {
                        }
                        const VSmemBufferFP8& smemVTile = reinterpret_cast<const VSmemBufferFP8*>(smem_1.v[0][0] + half)[0];
                        xqa_shared_compute_v102::pv_fp8::smemXVPartGemm(Warp{}, acc_1, skip_scale, xRowNeedRescaleMask, xRowScales, smemXTile, (unsigned int)half, smemVTile, warp_0_1);
                        unsigned long long _mbarrier_arrival_token_6;
                        asm volatile(
                            "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                            : "=l"(_mbarrier_arrival_token_6)
                            : "l"(smem_raw + ((v_reuse_addr) - smem)), "r"((uint32_t)(1)) : "memory");
                    }
                    unsigned long long _mbarrier_arrival_token_7;
                    asm volatile(
                        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                        : "=l"(_mbarrier_arrival_token_7)
                        : "l"(smem_raw + ((x_consumed_addr + (tile) * 16) - smem)), "r"((uint32_t)(1)) : "memory");
                }
                x_produced_phase = !x_produced_phase;
            }
            if (seq_iters_1 > 0) {
                float _rcp_0 = __frcp_rn(total);
                float reciprocal = _rcp_0;
                reciprocal = reciprocal * v_cache_scale;
                UniformRescaleMask const accRowNeedRescaleMask_1{{4294967295u}};
                ThrdRegRowMax const accRowScales_1{{reciprocal}};
                xqa_shared_compute_v102::rescaleAcc(Warp{}, acc_1, accRowNeedRescaleMask_1, accRowScales_1);
            }
            GemmOutRegTile const outTile = xqa_shared_compute_v102::output_tile::toFp16(acc_1);
            XSmemBuffer& outSwizzleBuffer = smem_1.x[0][warp_0_1];
            __syncthreads();
            xqa_shared_compute_v102::output_tile::reorderAndStoreGemmOutTile(Warp{}, outSwizzleBuffer, outTile);
            __syncwarp();
            OutputHead* const output_dst = reinterpret_cast<OutputHead*>(output) + (request_offset * q_heads);
            const XSmemBuffer& output_src = reinterpret_cast<const XSmemBuffer*>(smem_1.x[0] + warp_0_1)[0];
            xqa_shared_compute_v102::copyOutputToGlobalMem(Warp{}, output_dst, q_heads, head_group_size, head_group * head_group_size, valid_rows, uint2{warp_0_1 * 64, row_begin}, output_src);
        }
    }

    // Cleanup
}

} // extern "C"
