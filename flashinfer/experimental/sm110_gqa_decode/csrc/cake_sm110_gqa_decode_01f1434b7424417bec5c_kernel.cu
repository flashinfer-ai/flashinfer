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

#include "cake_sm110_gqa_decode_device_abi.cuh"

#define NUM_MAIN_STAGES 1
#define SMEM_MAXIMA_OFF 0
#define SMEM_MAXIMA_STAGE_BYTES 128
#define SMEM_MAXIMA_STRIDE 128
#define SMEM_SUMS_OFF 128
#define SMEM_SUMS_STAGE_BYTES 128
#define SMEM_SUMS_STRIDE 128
#define SMEM_PROBS_OFF 256
#define SMEM_PROBS_STAGE_BYTES 512
#define SMEM_PROBS_STRIDE 512
#define SMEM_PARTIALS_OFF 768
#define SMEM_PARTIALS_STAGE_BYTES 16384
#define SMEM_PARTIALS_STRIDE 16384
#define SMEM_TOTAL 17152
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>
#include "cake_sm110_gqa_decode_device_helpers.cuh"
extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_sm110_gqa_decode_01f1434b7424417bec5c(__half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ V, __half* __restrict__ O, int* __restrict__ sequence_lengths, float softmax_scale_log2, long long Q_stride_p0, long long Q_stride_p1, long long Q_stride_p2, long long K_stride_p0, long long K_stride_p1, long long K_stride_p2, long long V_stride_p0, long long V_stride_p1, long long V_stride_p2, long long K_size_p2, long long V_size_p2)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* maxima = reinterpret_cast<float*>(smem_raw + 0);
    const int maxima_addr = smem + 0;
    float* sums = reinterpret_cast<float*>(smem_raw + 128);
    const int sums_addr = smem + 128;
    __half* probs = reinterpret_cast<__half*>(smem_raw + 256);
    const int probs_addr = smem + 256;
    float* partials = reinterpret_cast<float*>(smem_raw + 768);
    const int partials_addr = smem + 768;

    // === Task calls (dependency order) ===
    int tile = blockIdx.x;
    int batch = tile / 8;
    int kv_head = tile - batch * 8;
    int thread = tid;
    int warp_1 = warp;
    int lane_1 = lane;
    long long batch64 = (long long)batch;
    long long kv_head64 = (long long)kv_head;
    long long q_stride_batch = Q_stride_p0;
    long long q_stride_head = Q_stride_p1;
    long long q_stride_kv_head = Q_stride_p2;
    long long q_base = q_stride_batch * batch64 + q_stride_kv_head * kv_head64;
    long long k_stride_batch = K_stride_p0;
    long long k_stride_head = K_stride_p1;
    long long k_stride_key = K_stride_p2;
    long long k_base = k_stride_batch * batch64 + k_stride_head * kv_head64;
    long long v_stride_batch = V_stride_p0;
    long long v_stride_head = V_stride_p1;
    long long v_stride_key = V_stride_p2;
    long long v_base = v_stride_batch * batch64 + v_stride_head * kv_head64;
    int kq = thread / 8;
    int q8 = thread - kq * 8;
    int kg = thread / 16;
    int ch = thread - kg * 16;
    int o_row = (batch * 32 + kv_head * 4) * 128;
    int seqlen = sequence_lengths[batch];
    int k_capacity = (int)K_size_p2;
    int v_capacity = (int)V_size_p2;
    float q[64];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + (q_base + q_stride_head * (long long)g + (long long)(q8 * 16 + h * 8)) + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&q[g * 16 + h * 8 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
        }
    }
    unsigned int k_words[16];
    k_words[0] = 0;
    k_words[1] = 0;
    k_words[2] = 0;
    k_words[3] = 0;
    k_words[4] = 0;
    k_words[5] = 0;
    k_words[6] = 0;
    k_words[7] = 0;
    k_words[8] = 0;
    k_words[9] = 0;
    k_words[10] = 0;
    k_words[11] = 0;
    k_words[12] = 0;
    k_words[13] = 0;
    k_words[14] = 0;
    k_words[15] = 0;
    unsigned int v_words[16];
    v_words[0] = 0;
    v_words[1] = 0;
    v_words[2] = 0;
    v_words[3] = 0;
    v_words[4] = 0;
    v_words[5] = 0;
    v_words[6] = 0;
    v_words[7] = 0;
    v_words[8] = 0;
    v_words[9] = 0;
    v_words[10] = 0;
    v_words[11] = 0;
    v_words[12] = 0;
    v_words[13] = 0;
    v_words[14] = 0;
    v_words[15] = 0;
    int key = kg * 4;
    if (key < 8 && key < v_capacity) {
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key + (long long)(ch * 8)) + 0);
            v_words[0 + 0] = _uv4_1.x;
            v_words[0 + 1] = _uv4_1.y;
            v_words[0 + 2] = _uv4_1.z;
            v_words[0 + 3] = _uv4_1.w;
        }
    }
    int key_0 = kg * 4 + 1;
    if (key_0 < 8 && key_0 < v_capacity) {
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_0 + (long long)(ch * 8)) + 0);
            v_words[4 + 0] = _uv4_2.x;
            v_words[4 + 1] = _uv4_2.y;
            v_words[4 + 2] = _uv4_2.z;
            v_words[4 + 3] = _uv4_2.w;
        }
    }
    int key_1 = kg * 4 + 2;
    if (key_1 < 8 && key_1 < v_capacity) {
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_1 + (long long)(ch * 8)) + 0);
            v_words[8 + 0] = _uv4_3.x;
            v_words[8 + 1] = _uv4_3.y;
            v_words[8 + 2] = _uv4_3.z;
            v_words[8 + 3] = _uv4_3.w;
        }
    }
    int key_2 = kg * 4 + 3;
    if (key_2 < 8 && key_2 < v_capacity) {
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_2 + (long long)(ch * 8)) + 0);
            v_words[12 + 0] = _uv4_4.x;
            v_words[12 + 1] = _uv4_4.y;
            v_words[12 + 2] = _uv4_4.z;
            v_words[12 + 3] = _uv4_4.w;
        }
    }
    float v[8];
    if (seqlen == 1) {
        if (kg == 0) {
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&v[_pair * 2]))
                    : "r"(v_words[_pair]));
            }
            #pragma unroll
            for (int g_1 = 0; g_1 < 4; g_1++) {
                {
                    __half2 _pk[4];
                    _pk[0] = __floats2half2_rn(v[0 + 0], v[0 + 1]);
                    _pk[1] = __floats2half2_rn(v[0 + 2], v[0 + 3]);
                    _pk[2] = __floats2half2_rn(v[0 + 4], v[0 + 5]);
                    _pk[3] = __floats2half2_rn(v[0 + 6], v[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__half*)(O + (o_row + g_1 * 128 + ch * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    } else {
        int key_3 = kq * 2;
        if (key_3 >= 0 && key_3 < seqlen && key_3 < k_capacity) {
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(K + (k_base + k_stride_key * (long long)key_3 + (long long)(q8 * 16)) + 0);
                k_words[0 + 0] = _uv4_5.x;
                k_words[0 + 1] = _uv4_5.y;
                k_words[0 + 2] = _uv4_5.z;
                k_words[0 + 3] = _uv4_5.w;
            }
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(K + (k_base + k_stride_key * (long long)key_3 + (long long)(q8 * 16 + 8)) + 0);
                k_words[4 + 0] = _uv4_6.x;
                k_words[4 + 1] = _uv4_6.y;
                k_words[4 + 2] = _uv4_6.z;
                k_words[4 + 3] = _uv4_6.w;
            }
        }
        int key_4 = kq * 2 + 1;
        if (key_4 >= 0 && key_4 < seqlen && key_4 < k_capacity) {
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(K + (k_base + k_stride_key * (long long)key_4 + (long long)(q8 * 16)) + 0);
                k_words[8 + 0] = _uv4_7.x;
                k_words[8 + 1] = _uv4_7.y;
                k_words[8 + 2] = _uv4_7.z;
                k_words[8 + 3] = _uv4_7.w;
            }
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(K + (k_base + k_stride_key * (long long)key_4 + (long long)(q8 * 16 + 8)) + 0);
                k_words[12 + 0] = _uv4_8.x;
                k_words[12 + 1] = _uv4_8.y;
                k_words[12 + 2] = _uv4_8.z;
                k_words[12 + 3] = _uv4_8.w;
            }
        }
        int key_5 = kg * 4;
        if (key_5 >= 8 && key_5 < seqlen && key_5 < v_capacity) {
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_5 + (long long)(ch * 8)) + 0);
                v_words[0 + 0] = _uv4_9.x;
                v_words[0 + 1] = _uv4_9.y;
                v_words[0 + 2] = _uv4_9.z;
                v_words[0 + 3] = _uv4_9.w;
            }
        }
        int key_6 = kg * 4 + 1;
        if (key_6 >= 8 && key_6 < seqlen && key_6 < v_capacity) {
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_6 + (long long)(ch * 8)) + 0);
                v_words[4 + 0] = _uv4_10.x;
                v_words[4 + 1] = _uv4_10.y;
                v_words[4 + 2] = _uv4_10.z;
                v_words[4 + 3] = _uv4_10.w;
            }
        }
        int key_7 = kg * 4 + 2;
        if (key_7 >= 8 && key_7 < seqlen && key_7 < v_capacity) {
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_7 + (long long)(ch * 8)) + 0);
                v_words[8 + 0] = _uv4_11.x;
                v_words[8 + 1] = _uv4_11.y;
                v_words[8 + 2] = _uv4_11.z;
                v_words[8 + 3] = _uv4_11.w;
            }
        }
        int key_8 = kg * 4 + 3;
        if (key_8 >= 8 && key_8 < seqlen && key_8 < v_capacity) {
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(V + (v_base + v_stride_key * (long long)key_8 + (long long)(ch * 8)) + 0);
                v_words[12 + 0] = _uv4_12.x;
                v_words[12 + 1] = _uv4_12.y;
                v_words[12 + 2] = _uv4_12.z;
                v_words[12 + 3] = _uv4_12.w;
            }
        }
        float k[16];
        float scores[8];
        scores[0] = 0.0f;
        scores[1] = 0.0f;
        scores[2] = 0.0f;
        scores[3] = 0.0f;
        scores[4] = 0.0f;
        scores[5] = 0.0f;
        scores[6] = 0.0f;
        scores[7] = 0.0f;
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&k[_pair * 2]))
                : "r"(k_words[_pair]));
        }
        #pragma unroll
        for (int g_2 = 0; g_2 < 4; g_2++) {
            #pragma unroll
            for (int d = 0; d < 16; d++) {
                scores[g_2] = __fmaf_rn(q[g_2 * 16 + d], k[d], scores[g_2]);
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&k[_pair * 2]))
                : "r"(k_words[8 + _pair]));
        }
        #pragma unroll
        for (int g_3 = 0; g_3 < 4; g_3++) {
            #pragma unroll
            for (int d_1 = 0; d_1 < 16; d_1++) {
                scores[4 + g_3] = __fmaf_rn(q[g_3 * 16 + d_1], k[d_1], scores[4 + g_3]);
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, scores[i], 1);
            scores[i] = scores[i] + _shfl_xor_sync_13;
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, scores[i_1], 2);
            scores[i_1] = scores[i_1] + _shfl_xor_sync_14;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; i_2++) {
            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, scores[i_2], 4);
            scores[i_2] = scores[i_2] + _shfl_xor_sync_15;
        }
        int key_9 = kq * 2;
        if (key_9 >= seqlen) {
            #pragma unroll
            for (int g_4 = 0; g_4 < 4; g_4++) {
                scores[g_4] = -CUDART_INF_F;
            }
        }
        int key_10 = kq * 2 + 1;
        if (key_10 >= seqlen) {
            #pragma unroll
            for (int g_5 = 0; g_5 < 4; g_5++) {
                scores[4 + g_5] = -CUDART_INF_F;
            }
        }
        #pragma unroll
        for (int g_6 = 0; g_6 < 4; g_6++) {
            float warp_max = scores[g_6];
            #pragma unroll
            for (int j = 1; j < 2; j++) {
                warp_max = fmaxf(warp_max, scores[j * 4 + g_6]);
            }
            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, warp_max, 8);
            warp_max = fmaxf(warp_max, _shfl_xor_sync_16);
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, warp_max, 16);
            warp_max = fmaxf(warp_max, _shfl_xor_sync_17);
            if (lane_1 == 0) {
                maxima[warp_1 * 4 + g_6] = warp_max;
            }
        }
        __syncthreads();
        float probabilities[8];
        #pragma unroll
        for (int g_7 = 0; g_7 < 4; g_7++) {
            float tile_max = maxima[g_7];
            #pragma unroll
            for (int w = 1; w < 8; w++) {
                tile_max = fmaxf(tile_max, maxima[w * 4 + g_7]);
            }
            float safe_max = ((tile_max == -CUDART_INF_F) ? 0.0f : tile_max);
            float max_scaled = safe_max * softmax_scale_log2;
            float score_bias = ((seqlen > 0) ? -max_scaled : -CUDART_INF_F);
            #pragma unroll
            for (int j_1 = 0; j_1 < 2; j_1++) {
                float _exp2_18 = approx_exp2(__fmaf_rn(scores[j_1 * 4 + g_7], softmax_scale_log2, score_bias));
                probabilities[j_1 * 4 + g_7] = _exp2_18;
            }
        }
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(probabilities[_lp*2 + 0], probabilities[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int g_8 = 0; g_8 < 4; g_8++) {
            float partial_sum = 0.0f;
            if (q8 == 0) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 2; j_2++) {
                    partial_sum += probabilities[j_2 * 4 + g_8];
                }
            }
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, partial_sum, 8);
            partial_sum += _shfl_xor_sync_19;
            float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, partial_sum, 16);
            partial_sum += _shfl_xor_sync_20;
            if (lane_1 == 0) {
                sums[warp_1 * 4 + g_8] = partial_sum;
            }
        }
        if (q8 == 0) {
            #pragma unroll
            for (int h_1 = 0; h_1 < 1; h_1++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probs_addr + (unsigned int)(kq * 16 + h_1 * 16)), "r"(packed[h_1 * 4]), "r"(packed[h_1 * 4 + 1]), "r"(packed[h_1 * 4 + 2]), "r"(packed[h_1 * 4 + 3]) : "memory");
            }
        }
        __syncthreads();
        unsigned int words[8];
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; i_3++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[i_3 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&words[(i_3 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(i_3 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(i_3 * 4) + 3]))
                : "r"(probs_addr + (unsigned int)(kg * 32 + i_3 * 16)));
        }
        float _unpack_f16_0[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_0[_pair * 2]))
                : "r"(words[_pair]));
        }
        float out[32];
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
        out[3] = 0.0f;
        out[4] = 0.0f;
        out[5] = 0.0f;
        out[6] = 0.0f;
        out[7] = 0.0f;
        out[8] = 0.0f;
        out[9] = 0.0f;
        out[10] = 0.0f;
        out[11] = 0.0f;
        out[12] = 0.0f;
        out[13] = 0.0f;
        out[14] = 0.0f;
        out[15] = 0.0f;
        out[16] = 0.0f;
        out[17] = 0.0f;
        out[18] = 0.0f;
        out[19] = 0.0f;
        out[20] = 0.0f;
        out[21] = 0.0f;
        out[22] = 0.0f;
        out[23] = 0.0f;
        out[24] = 0.0f;
        out[25] = 0.0f;
        out[26] = 0.0f;
        out[27] = 0.0f;
        out[28] = 0.0f;
        out[29] = 0.0f;
        out[30] = 0.0f;
        out[31] = 0.0f;
        int key_11 = kg * 4;
        if (key_11 < seqlen) {
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&v[_pair * 2]))
                    : "r"(v_words[_pair]));
            }
            #pragma unroll
            for (int g_9 = 0; g_9 < 4; g_9++) {
                #pragma unroll
                for (int d_2 = 0; d_2 < 8; d_2++) {
                    out[g_9 * 8 + d_2] = __fmaf_rn(_unpack_f16_0[g_9], v[d_2], out[g_9 * 8 + d_2]);
                }
            }
        }
        int key_12 = kg * 4 + 1;
        if (key_12 < seqlen) {
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&v[_pair * 2]))
                    : "r"(v_words[4 + _pair]));
            }
            #pragma unroll
            for (int g_10 = 0; g_10 < 4; g_10++) {
                #pragma unroll
                for (int d_3 = 0; d_3 < 8; d_3++) {
                    out[g_10 * 8 + d_3] = __fmaf_rn(_unpack_f16_0[4 + g_10], v[d_3], out[g_10 * 8 + d_3]);
                }
            }
        }
        int key_13 = kg * 4 + 2;
        if (key_13 < seqlen) {
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&v[_pair * 2]))
                    : "r"(v_words[8 + _pair]));
            }
            #pragma unroll
            for (int g_11 = 0; g_11 < 4; g_11++) {
                #pragma unroll
                for (int d_4 = 0; d_4 < 8; d_4++) {
                    out[g_11 * 8 + d_4] = __fmaf_rn(_unpack_f16_0[8 + g_11], v[d_4], out[g_11 * 8 + d_4]);
                }
            }
        }
        int key_14 = kg * 4 + 3;
        if (key_14 < seqlen) {
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&v[_pair * 2]))
                    : "r"(v_words[12 + _pair]));
            }
            #pragma unroll
            for (int g_12 = 0; g_12 < 4; g_12++) {
                #pragma unroll
                for (int d_5 = 0; d_5 < 8; d_5++) {
                    out[g_12 * 8 + d_5] = __fmaf_rn(_unpack_f16_0[12 + g_12], v[d_5], out[g_12 * 8 + d_5]);
                }
            }
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 32; i_4++) {
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, out[i_4], 16);
            out[i_4] = out[i_4] + _shfl_xor_sync_21;
        }
        if (lane_1 < 16) {
            #pragma unroll
            for (int g_13 = 0; g_13 < 4; g_13++) {
                #pragma unroll
                for (int h_2 = 0; h_2 < 2; h_2++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partials_addr + (unsigned int)(((warp_1 * 4 + g_13) * 128 + ch * 8 + h_2 * 4) * 4)), "f"(out[g_13 * 8 + h_2 * 4]), "f"(out[g_13 * 8 + h_2 * 4 + 1]), "f"(out[g_13 * 8 + h_2 * 4 + 2]), "f"(out[g_13 * 8 + h_2 * 4 + 3]) : "memory");
                }
            }
        }
        __syncthreads();
        if (thread < 64) {
            float acc[8];
            #pragma unroll
            for (int h_3 = 0; h_3 < 2; h_3++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[h_3 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(h_3 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(h_3 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(h_3 * 4) + 3]))
                    : "r"(partials_addr + (unsigned int)((kg * 128 + ch * 8 + h_3 * 4) * 4)));
            }
            #pragma unroll
            for (int w_1 = 1; w_1 < 8; w_1++) {
                float part[8];
                #pragma unroll
                for (int h_4 = 0; h_4 < 2; h_4++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[h_4 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(h_4 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(h_4 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(h_4 * 4) + 3]))
                        : "r"(partials_addr + (unsigned int)(((w_1 * 4 + kg) * 128 + ch * 8 + h_4 * 4) * 4)));
                }
                #pragma unroll
                for (int i_5 = 0; i_5 < 8; i_5++) {
                    acc[i_5] = acc[i_5] + part[i_5];
                }
            }
            float block_sum = sums[kg];
            #pragma unroll
            for (int w_2 = 1; w_2 < 8; w_2++) {
                block_sum += sums[w_2 * 4 + kg];
            }
            float _rcp_22 = approx_rcp(block_sum);
            float inv_sum = ((block_sum > 0.0f && block_sum == block_sum) ? _rcp_22 : 0.0f);
            {
                const float2 _prescale2_23 = {inv_sum, inv_sum};
                #if __CUDA_ARCH__ >= 1000
                #pragma unroll
                for (int _ps = 0; _ps < 4; _ps++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&acc[0])[_ps], _prescale2_23);
                #else
                #pragma unroll
                for (int _ps = 0; _ps < 8; _ps++)
                    acc[0 + _ps] *= inv_sum;
                #endif
                __half2 _pk[4];
                _pk[0] = __floats2half2_rn(acc[0 + 0], acc[0 + 1]);
                _pk[1] = __floats2half2_rn(acc[0 + 2], acc[0 + 3]);
                _pk[2] = __floats2half2_rn(acc[0 + 4], acc[0 + 5]);
                _pk[3] = __floats2half2_rn(acc[0 + 6], acc[0 + 7]);
                *reinterpret_cast<uint4*>(&((__half*)(O + (o_row + kg * 128 + ch * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
