// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_ATTENTION_VARIANT_HELPER_H
#define FLASHINFER_ATTENTION_VARIANT_HELPER_H

#include "gpu_iface/platform.hpp"

#include <cstdint>

namespace flashinfer
{

#define REGISTER_QUERY_TRANSFORM(params, q, ...)                               \
    template <typename Params, typename T>                                     \
    __device__ __forceinline__ T QueryTransform(const Params &params,          \
                                                void *q_smem)                  \
    {                                                                          \
        __VA_ARGS__                                                            \
    }

#define REGISTER_KEY_TRANSFORM(params, k, ...)                                 \
    template <typename Params, typename T>                                     \
    __device__ __forceinline__ T KeyTransform(const Params &params,            \
                                              void *k_smem)                    \
    {                                                                          \
        __VA_ARGS__                                                            \
    }

#define REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx,   \
                                  qo_head_idx, kv_head_idx, ...)               \
    template <typename Params, typename T>                                     \
    __device__ __forceinline__ T LogitsTransform(                              \
        const Params &params, T logits, uint32_t batch_idx, uint32_t qo_idx,   \
        uint32_t kv_idx, uint32_t qo_head_idx, uint32_t kv_head_idx)           \
    {                                                                          \
        __VA_ARGS__                                                            \
    }

#define REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx,   \
                             kv_head_idx, ...)                                 \
    template <typename Params>                                                 \
    __device__ __forceinline__ bool LogitsMask(                                \
        const Params &params, uint32_t batch_idx, uint32_t qo_idx,             \
        uint32_t kv_idx, uint32_t qo_head_idx, uint32_t kv_head_idx)           \
    {                                                                          \
        __VA_ARGS__                                                            \
    }

struct AttentionVariantBase
{
    constexpr static bool use_softmax = true;
    REGISTER_LOGITS_TRANSFORM(params,
                              logits,
                              batch_idx,
                              qo_idx,
                              kv_idx,
                              qo_head_idx,
                              kv_head_idx,
                              { return logits; })

    REGISTER_LOGITS_MASK(params,
                         batch_idx,
                         qo_idx,
                         kv_idx,
                         qo_head_idx,
                         kv_head_idx,
                         { return true; })
};

} // namespace flashinfer

#endif // FLASHINFER_ATTENTION_VARIANT_HELPER_H
