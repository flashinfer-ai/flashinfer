/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Minimal warp-level top-K reduction lifted from
// tensorrt_llm/cpp/.../kernels/moeTopKFuncs.cuh - only the bits used by the
// non-grouped, MaxNumExperts=256 / Topk=8 path. Standalone (no
// tensorrt_llm/common/config.h dependency).
//
// Pulled this in instead of pulling the whole TRT-LLM header so the
// extension builds without the full repo include path.
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace mega_topk
{

namespace cg = cooperative_groups;
constexpr int kWARP_SIZE = 32;

template <typename T_>
struct TopKRedType
{
    using T = T_;
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, int>, "Top K reduction here only specialised for float / int");
    using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
    static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
    static constexpr int kMaxIdx = 65535;
    TypeCmp compValIdx;

    static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0)
    {
        auto valueBits = cub::Traits<T>::TwiddleIn(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
        TypeCmp compactTmp = valueBits;
        compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
        return compactTmp;
    }

    static __host__ __device__ void unpack(T& value, int32_t& index, TypeCmp cmp)
    {
        index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));
        auto compactTmp = cmp >> kMoveBits;
        auto valueBits
            = cub::Traits<T>::TwiddleOut(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
        value = reinterpret_cast<T&>(valueBits);
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(T val, int32_t idx)
        : compValIdx(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compValIdx;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<kWARP_SIZE> const& warp)
    {
        return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
    }
};

#define MEGA_TOPK_SWAP(I, J)                                                                                           \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        topK[I].compValIdx = pairMax;                                                                                  \
        topK[J].compValIdx = pairMin;                                                                                  \
    }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType>
{
    static __device__ void run(RedType*) {}
};

template <typename RedType>
struct Sort<4, RedType>
{
    static __device__ void run(RedType* topK)
    {
        MEGA_TOPK_SWAP(0, 2);
        MEGA_TOPK_SWAP(1, 3);
        MEGA_TOPK_SWAP(0, 1);
        MEGA_TOPK_SWAP(2, 3);
        MEGA_TOPK_SWAP(1, 2);
    }
};

// Single-value-per-thread warp top-K (used in the final merge stage).
template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue, int actualK = K)
{
    static_assert(K > 0 && K < kWARP_SIZE);
    using RedType = TopKRedType<Type>;
    RedType topK{value, idx};
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < actualK; ++kk)
    {
        topK = kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
        packedMax = topK.reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
}

// N-values-per-thread warp top-K (used for the 128-experts-per-warp stage).
template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N], Type const minValue, int actualK = K)
{
    static_assert(K > 0 && K < kWARP_SIZE);
    static_assert(N == 4, "this specialisation: 4 candidates per thread (128 per warp)");
    using RedType = TopKRedType<Type>;

    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
        topK[nn] = RedType{value[nn], idx[nn]};
    Sort<N, RedType>::run(topK);

    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < actualK; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
        for (int nn = 0; nn < N; ++nn)
        {
            topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
        }
        packedMax = topK[0].reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
}

#undef MEGA_TOPK_SWAP

} // namespace mega_topk
