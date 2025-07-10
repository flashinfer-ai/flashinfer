// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once

#include "enums.hpp"
#include "gpu_iface/exception.h"

#define DISPATCH_USE_FP16_QK_REDUCTION(use_fp16_qk_reduction,                  \
                                       USE_FP16_QK_REDUCTION, ...)             \
    if (use_fp16_qk_reduction) {                                               \
        FLASHINFER_ERROR("FP16_QK_REDUCTION disabled at compile time");        \
    }                                                                          \
    else {                                                                     \
        constexpr bool USE_FP16_QK_REDUCTION = false;                          \
        __VA_ARGS__                                                            \
    }

#define DISPATCH_NUM_MMA_Q(num_mma_q, NUM_MMA_Q, ...)                          \
    if (num_mma_q == 1) {                                                      \
        constexpr size_t NUM_MMA_Q = 1;                                        \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (num_mma_q == 2) {                                                 \
        constexpr size_t NUM_MMA_Q = 2;                                        \
        __VA_ARGS__                                                            \
    }                                                                          \
    else {                                                                     \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported num_mma_q: " << num_mma_q;                     \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }

#define DISPATCH_NUM_MMA_KV(max_mma_kv, NUM_MMA_KV, ...)                       \
    if (max_mma_kv >= 8) {                                                     \
        constexpr size_t NUM_MMA_KV = 8;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (max_mma_kv >= 4) {                                                \
        constexpr size_t NUM_MMA_KV = 4;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (max_mma_kv >= 2) {                                                \
        constexpr size_t NUM_MMA_KV = 2;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (max_mma_kv >= 1) {                                                \
        constexpr size_t NUM_MMA_KV = 1;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else {                                                                     \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported max_mma_kv: " << max_mma_kv;                   \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }

#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)                       \
    switch (cta_tile_q) {                                                      \
    case 128:                                                                  \
    {                                                                          \
        constexpr uint32_t CTA_TILE_Q = 128;                                   \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 64:                                                                   \
    {                                                                          \
        constexpr uint32_t CTA_TILE_Q = 64;                                    \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 16:                                                                   \
    {                                                                          \
        constexpr uint32_t CTA_TILE_Q = 16;                                    \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    default:                                                                   \
    {                                                                          \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported cta_tile_q: " << cta_tile_q;                   \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }                                                                          \
    }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...)                   \
    if (group_size == 1) {                                                     \
        constexpr size_t GROUP_SIZE = 1;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (group_size == 2) {                                                \
        constexpr size_t GROUP_SIZE = 2;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (group_size == 3) {                                                \
        constexpr size_t GROUP_SIZE = 3;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (group_size == 4) {                                                \
        constexpr size_t GROUP_SIZE = 4;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else if (group_size == 8) {                                                \
        constexpr size_t GROUP_SIZE = 8;                                       \
        __VA_ARGS__                                                            \
    }                                                                          \
    else {                                                                     \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported group_size: " << group_size;                   \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }

#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)                          \
    switch (mask_mode) {                                                       \
    case MaskMode::kNone:                                                      \
    {                                                                          \
        constexpr MaskMode MASK_MODE = MaskMode::kNone;                        \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case MaskMode::kCausal:                                                    \
    {                                                                          \
        constexpr MaskMode MASK_MODE = MaskMode::kCausal;                      \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case MaskMode::kCustom:                                                    \
    {                                                                          \
        constexpr MaskMode MASK_MODE = MaskMode::kCustom;                      \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    default:                                                                   \
    {                                                                          \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported mask_mode: " << int(mask_mode);                \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }                                                                          \
    }

// convert head_dim to compile-time constant
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                             \
    switch (head_dim) {                                                        \
    case 64:                                                                   \
    {                                                                          \
        constexpr size_t HEAD_DIM = 64;                                        \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 128:                                                                  \
    {                                                                          \
        constexpr size_t HEAD_DIM = 128;                                       \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 256:                                                                  \
    {                                                                          \
        constexpr size_t HEAD_DIM = 256;                                       \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 512:                                                                  \
    {                                                                          \
        constexpr size_t HEAD_DIM = 512;                                       \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    default:                                                                   \
    {                                                                          \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported head_dim: " << head_dim;                       \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }                                                                          \
    }

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)  \
    switch (pos_encoding_mode) {                                               \
    case PosEncodingMode::kNone:                                               \
    {                                                                          \
        constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;  \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case PosEncodingMode::kRoPELlama:                                          \
    {                                                                          \
        constexpr PosEncodingMode POS_ENCODING_MODE =                          \
            PosEncodingMode::kRoPELlama;                                       \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case PosEncodingMode::kALiBi:                                              \
    {                                                                          \
        constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi; \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    default:                                                                   \
    {                                                                          \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported pos_encoding_mode: "                           \
                << int(pos_encoding_mode);                                     \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }                                                                          \
    }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...)     \
    switch (aligned_vec_size) {                                                \
    case 16:                                                                   \
    {                                                                          \
        constexpr size_t ALIGNED_VEC_SIZE = 16;                                \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 8:                                                                    \
    {                                                                          \
        constexpr size_t ALIGNED_VEC_SIZE = 8;                                 \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 4:                                                                    \
    {                                                                          \
        constexpr size_t ALIGNED_VEC_SIZE = 4;                                 \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 2:                                                                    \
    {                                                                          \
        constexpr size_t ALIGNED_VEC_SIZE = 2;                                 \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    case 1:                                                                    \
    {                                                                          \
        constexpr size_t ALIGNED_VEC_SIZE = 1;                                 \
        __VA_ARGS__                                                            \
        break;                                                                 \
    }                                                                          \
    default:                                                                   \
    {                                                                          \
        std::ostringstream err_msg;                                            \
        err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;       \
        FLASHINFER_ERROR(err_msg.str());                                       \
    }                                                                          \
    }

#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity,          \
                                                    NUM_STAGES_SMEM, ...)      \
    if (compute_capacity.first >= 8) {                                         \
        constexpr uint32_t NUM_STAGES_SMEM = 2;                                \
        __VA_ARGS__                                                            \
    }                                                                          \
    else {                                                                     \
        constexpr uint32_t NUM_STAGES_SMEM = 1;                                \
        __VA_ARGS__                                                            \
    }
