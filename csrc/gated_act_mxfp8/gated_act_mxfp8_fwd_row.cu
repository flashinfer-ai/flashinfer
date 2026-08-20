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
#include <stdint.h>

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}

#define GATED_MXFP8_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROW_OUTPUT_OFF 0
#define SMEM_ROW_OUTPUT_STAGE_BYTES 4096
#define SMEM_ROW_OUTPUT_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_gated_act_mxfp8_fwd_row_direct_128x64(__nv_bfloat16* __restrict__ gated_input, __grid_constant__ CUtensorMap const row_output_tma, uint8_t* __restrict__ row_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* row_output = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int row_output_addr = smem + 0;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int half = tid_0 & 1;
    int blk = tid_0 >> 1 & 3;
    int row = tid_0 >> 3;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    unsigned int gate_words[8];
    unsigned int up_words[8];
    float output_pair_values[2];
    unsigned int packed_bf16[8];
    float scaled_quad[4];
    unsigned int packed_fp8[4];
    float x0 = 0.0f;
    float x1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    unsigned int x0_bits = 0;
    unsigned int x1_bits = 0;
    unsigned int up0_bits = 0;
    unsigned int up1_bits = 0;
    float scaled_value0 = 0.0f;
    float scaled_value1 = 0.0f;
    float scaled_value2 = 0.0f;
    float scaled_value3 = 0.0f;
    unsigned int scaled_bits0 = 0;
    unsigned int scaled_bits1 = 0;
    unsigned int scaled_bits2 = 0;
    unsigned int scaled_bits3 = 0;
    #pragma unroll
    for (int stage = 0; stage < 1; stage++) {
        int grow = by * 32 + stage * 32 + row;
        int col = bx * 128 + blk * 32 + half * 16;
        int gate_index = grow * (2 * K) + col;
        int up_index = gate_index + K;
        {
            const void* _v8p_0 = (const void*)(gated_input + (gate_index));
            uint32_t _v8_0_0[8];
            asm volatile(
                "ld.global.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
            *(&gate_words[0 + 0]) = _v8_0_0[0];
            *(&gate_words[0 + 1]) = _v8_0_0[1];
            *(&gate_words[0 + 2]) = _v8_0_0[2];
            *(&gate_words[0 + 3]) = _v8_0_0[3];
            *(&gate_words[0 + 4]) = _v8_0_0[4];
            *(&gate_words[0 + 5]) = _v8_0_0[5];
            *(&gate_words[0 + 6]) = _v8_0_0[6];
            *(&gate_words[0 + 7]) = _v8_0_0[7];
        }
        {
            const void* _v8p_1 = (const void*)(gated_input + (up_index));
            uint32_t _v8_1_0[8];
            asm volatile(
                "ld.global.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
            *(&up_words[0 + 0]) = _v8_1_0[0];
            *(&up_words[0 + 1]) = _v8_1_0[1];
            *(&up_words[0 + 2]) = _v8_1_0[2];
            *(&up_words[0 + 3]) = _v8_1_0[3];
            *(&up_words[0 + 4]) = _v8_1_0[4];
            *(&up_words[0 + 5]) = _v8_1_0[5];
            *(&up_words[0 + 6]) = _v8_1_0[6];
            *(&up_words[0 + 7]) = _v8_1_0[7];
        }
        unsigned int amax_pair = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; pair++) {
            unsigned int gate_word = gate_words[pair];
            unsigned int up_word = up_words[pair];
            x0_bits = (gate_word & 65535) << 16;
            x1_bits = gate_word & 4294901760;
            up0_bits = (up_word & 65535) << 16;
            up1_bits = up_word & 4294901760;
            x0 = reinterpret_cast<float*>(&x0_bits)[0];
            x1 = reinterpret_cast<float*>(&x1_bits)[0];
            up0 = reinterpret_cast<float*>(&up0_bits)[0];
            up1 = reinterpret_cast<float*>(&up1_bits)[0];
            float _exp2_noftz_0;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_0) : "f"((-x0) * 1.4426950408889634f));
            float exp0 = _exp2_noftz_0;
            float _exp2_noftz_1;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_1) : "f"((-x1) * 1.4426950408889634f));
            float exp1 = _exp2_noftz_1;
            float _rcp_rn_0;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_0) : "f"(1.0f + exp0));
            float sigmoid0 = _rcp_rn_0;
            float _rcp_rn_1;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_1) : "f"(1.0f + exp1));
            float sigmoid1 = _rcp_rn_1;
            float2 _f2_0 = make_float2(x0, x1);
            float2 _f2_1 = make_float2(sigmoid0, sigmoid1);
            float2 _f32x2_mul_rn_0;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
            float2 act_pair = _f32x2_mul_rn_0;
            float2 _f2_2 = make_float2(up0, up1);
            float2 _f32x2_mul_rn_1;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_1) : "l"(*(const unsigned long long*)&act_pair), "l"(*(const unsigned long long*)&_f2_2));
            float2 output_pair = _f32x2_mul_rn_1;
            output_pair_values[0] = output_pair.x;
            output_pair_values[1] = output_pair.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair_values[_lp*2 + 0], output_pair_values[_lp*2+1 + 0]));
                packed_bf16[_lp + (pair)] = *(uint32_t*)&_bf2;
            }
            uint32_t _bf16x2_abs_max_nan_0;
            asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_0) : "r"(amax_pair), "r"(packed_bf16[pair]));
            amax_pair = _bf16x2_abs_max_nan_0;
        }
        amax_pair = amax_pair & 2147450879;
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax_pair, 1);
        unsigned int peer_amax = _shfl_xor_0;
        uint32_t _bf16x2_max_nan_0;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_0) : "r"(amax_pair), "r"(peer_amax));
        amax_pair = _bf16x2_max_nan_0;
        uint32_t _bf16x2_max_nan_1;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_1) : "r"(amax_pair), "r"(amax_pair >> 16));
        amax_pair = _bf16x2_max_nan_1;
        unsigned int amax_f32_bits = (amax_pair & 65535) << 16;
        int scale_i32 = (int)(amax_f32_bits + 2031616 >> 23) - 8;
        if (scale_i32 < 0) {
            scale_i32 = 0;
        }
        unsigned int exponent_bits = amax_f32_bits & 2139095040;
        if (exponent_bits == 2139095040) {
            scale_i32 = 255;
        }
        unsigned int scale = (unsigned int)scale_i32;
        int scale_col = bx * 4 + blk;
        int num_scale_col_blocks = K / 128;
        int scale_index = ((grow >> 7) * num_scale_col_blocks + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
        if (half == 0) {
            *((unsigned char*)(row_scales + scale_index)) = (unsigned char)(scale);
        }
        unsigned int inv_bf16 = 254 - scale << 7;
        if (scale == 255) {
            inv_bf16 = 32704;
        }
        unsigned int inv_bf16x2 = inv_bf16 | inv_bf16 << 16;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t _bf16x2_mul_0;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed_bf16[2 * q]), "r"(inv_bf16x2));
            unsigned int scaled0 = _bf16x2_mul_0;
            uint32_t _bf16x2_mul_1;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed_bf16[2 * q + 1]), "r"(inv_bf16x2));
            unsigned int scaled1 = _bf16x2_mul_1;
            scaled_bits0 = (scaled0 & 65535) << 16;
            scaled_bits1 = scaled0 & 4294901760;
            scaled_bits2 = (scaled1 & 65535) << 16;
            scaled_bits3 = scaled1 & 4294901760;
            scaled_value0 = reinterpret_cast<float*>(&scaled_bits0)[0];
            scaled_value1 = reinterpret_cast<float*>(&scaled_bits1)[0];
            scaled_value2 = reinterpret_cast<float*>(&scaled_bits2)[0];
            scaled_value3 = reinterpret_cast<float*>(&scaled_bits3)[0];
            scaled_quad[0] = scaled_value0;
            scaled_quad[1] = scaled_value1;
            scaled_quad[2] = scaled_value2;
            scaled_quad[3] = scaled_value3;
            {
                uint32_t _packed;
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_packed) : "f"(scaled_quad[0]), "f"(scaled_quad[1]), "f"(scaled_quad[2]), "f"(scaled_quad[3]));
                packed_fp8[(q) + 0] = _packed;
            }
        }
        int word_quad = blk * 2 + half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((row_output_addr + (unsigned int)(stage * 4096) + (unsigned int)(row * 128 + word_quad * 16))), "r"(packed_fp8[0]), "r"(packed_fp8[1]), "r"(packed_fp8[2]), "r"(packed_fp8[3]) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                tma_store_2d(&row_output_tma, bx * 128, by * 32 + stage * 32, row_output_addr + (unsigned int)(stage * 4096));
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
}

} // extern "C"

#undef GATED_MXFP8_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROW_OUTPUT_OFF
#undef SMEM_ROW_OUTPUT_STAGE_BYTES
#undef SMEM_ROW_OUTPUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef row_output_addr
