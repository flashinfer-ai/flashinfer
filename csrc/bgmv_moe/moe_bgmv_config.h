#pragma once

#include <cstdint>

/*
 * BGMV MoE dimension configuration and forward declarations.
 *
 * Defines the set of (narrow, wide) dimension pairs that are compiled.
 * narrow = LoRA rank (8, 16, 32, 64)
 * wide = hidden/intermediate dimension of the model
 *
 * Models covered:
 *   Qwen3-30B-A3B:              gate_up=(2048,768),  down=(768,2048)
 *   Qwen3.5-35B-A3B:            gate_up=(2048,768),  down=(768,2048)  [256 experts, top-8]
 *   Gemma-4-26B-A4B:            gate_up=(2816,2112), down=(2112,2816) [128 experts, top-8]
 *   Nemotron-Nano-3-30B-A3B:    gate_up=(2688,1856), down=(1856,2688)
 *   Nemotron-3-Super-120B-A12B: gate_up=(4096,2688), down=(2688,4096)
 *   Large MoE (128 experts):    gate_up=(3072,5888), down=(2944,3072)
 *
 * TP-sharded dimensions also included.
 * All wide values must satisfy: wide % 32 == 0.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

// clang-format off

#define FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 384)   \
    f(in_T, out_T, W_T, narrow, 736)   \
    f(in_T, out_T, W_T, narrow, 768)   \
    f(in_T, out_T, W_T, narrow, 1024)  \
    f(in_T, out_T, W_T, narrow, 1344)  \
    f(in_T, out_T, W_T, narrow, 1472)  \
    f(in_T, out_T, W_T, narrow, 1536)  \
    f(in_T, out_T, W_T, narrow, 1856)  \
    f(in_T, out_T, W_T, narrow, 2048)  \
    f(in_T, out_T, W_T, narrow, 2112)  \
    f(in_T, out_T, W_T, narrow, 2688)  \
    f(in_T, out_T, W_T, narrow, 2816)  \
    f(in_T, out_T, W_T, narrow, 2880)  \
    f(in_T, out_T, W_T, narrow, 2944)  \
    f(in_T, out_T, W_T, narrow, 3072)  \
    f(in_T, out_T, W_T, narrow, 4096)  \
    f(in_T, out_T, W_T, narrow, 5120)  \
    f(in_T, out_T, W_T, narrow, 5888)  \
    f(in_T, out_T, W_T, narrow, 7168)  \
    f(in_T, out_T, W_T, narrow, 8192)  \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 28672)

#define FOR_MOE_ALL_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 64)

// clang-format on

// ===== Forward declarations =====

template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void moe_bgmv_shrink_sliced(out_T* __restrict__ Y, const in_T* __restrict__ X,
                            W_T** __restrict__ w_ptr, const int64_t* __restrict__ sorted_token_ids,
                            const int64_t* __restrict__ expert_ids,
                            const int64_t* __restrict__ lora_indices, int64_t num_pairs,
                            int64_t num_slices, int64_t num_experts, int64_t num_tokens,
                            int64_t lora_stride, float scale);

template <int feat_in, int feat_out, typename in_T, typename W_T>
void moe_bgmv_expand_sliced(float* __restrict__ Y, const in_T* __restrict__ X,
                            W_T** __restrict__ w_ptr, const int64_t* __restrict__ sorted_token_ids,
                            const int64_t* __restrict__ expert_ids,
                            const int64_t* __restrict__ lora_indices,
                            const float* __restrict__ topk_weights,
                            const int64_t* __restrict__ slice_start_loc, int64_t num_pairs,
                            int64_t num_slices, int64_t num_experts, int64_t total_feat_out,
                            int32_t current_feat_out, int64_t num_tokens, int64_t lora_stride,
                            float scale);
