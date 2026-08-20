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
#define SMEM_ROW_ACT_OFF 0
#define SMEM_ROW_ACT_STAGE_BYTES 2048
#define SMEM_ROW_ACT_STRIDE 2048
#define SMEM_ROW_GATE_OFF 2048
#define SMEM_ROW_GATE_STAGE_BYTES 2048
#define SMEM_ROW_GATE_STRIDE 2048
#define SMEM_TOTAL 4096
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gated_act_mxfp8_bwd_row_direct_64x64(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad_h, __grid_constant__ CUtensorMap const row_act_tma, __grid_constant__ CUtensorMap const row_gate_tma, uint8_t* __restrict__ row_scales, int M, int K)
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
    uint8_t* row_act = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int row_act_addr = smem + 0;
    uint8_t* row_gate = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int row_gate_addr = smem + 2048;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int half = tid_0 & 1;
    int blk = tid_0 >> 1 & 1;
    int row = tid_0 >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    unsigned int gate_words[8];
    unsigned int up_words[8];
    unsigned int grad_words[8];
    float pair_values[2];
    unsigned int packed_act[8];
    unsigned int packed_gate[8];
    float scaled_quad[4];
    unsigned int packed_fp8_act[4];
    unsigned int packed_fp8_gate[4];
    float x0 = 0.0f;
    float x1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float grad0 = 0.0f;
    float grad1 = 0.0f;
    unsigned int bits0 = 0;
    unsigned int bits1 = 0;
    unsigned int bits2 = 0;
    unsigned int bits3 = 0;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;
    #pragma unroll
    for (int stage = 0; stage < 1; stage++) {
        int grow = by * 32 + stage * 32 + row;
        int col = bx * 64 + blk * 32 + half * 16;
        int gate_index = grow * (2 * K) + col;
        int grad_index = grow * K + col;
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
            const void* _v8p_1 = (const void*)(gated_input + (gate_index + K));
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
        {
            const void* _v8p_2 = (const void*)(grad_h + (grad_index));
            uint32_t _v8_2_0[8];
            asm volatile(
                "ld.global.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
            *(&grad_words[0 + 0]) = _v8_2_0[0];
            *(&grad_words[0 + 1]) = _v8_2_0[1];
            *(&grad_words[0 + 2]) = _v8_2_0[2];
            *(&grad_words[0 + 3]) = _v8_2_0[3];
            *(&grad_words[0 + 4]) = _v8_2_0[4];
            *(&grad_words[0 + 5]) = _v8_2_0[5];
            *(&grad_words[0 + 6]) = _v8_2_0[6];
            *(&grad_words[0 + 7]) = _v8_2_0[7];
        }
        unsigned int amax_act = 0;
        unsigned int amax_gate = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; pair++) {
            unsigned int gate_word = gate_words[pair];
            unsigned int up_word = up_words[pair];
            unsigned int grad_word = grad_words[pair];
            bits0 = (gate_word & 65535) << 16;
            bits1 = gate_word & 4294901760;
            x0 = reinterpret_cast<float*>(&bits0)[0];
            x1 = reinterpret_cast<float*>(&bits1)[0];
            bits0 = (up_word & 65535) << 16;
            bits1 = up_word & 4294901760;
            up0 = reinterpret_cast<float*>(&bits0)[0];
            up1 = reinterpret_cast<float*>(&bits1)[0];
            bits0 = (grad_word & 65535) << 16;
            bits1 = grad_word & 4294901760;
            grad0 = reinterpret_cast<float*>(&bits0)[0];
            grad1 = reinterpret_cast<float*>(&bits1)[0];
            float _exp2_noftz_0;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_0) : "f"((-x0) * 1.4426950408889634f));
            float _rcp_rn_0;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_0) : "f"(1.0f + _exp2_noftz_0));
            float _exp2_noftz_1;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_1) : "f"((-x1) * 1.4426950408889634f));
            float _rcp_rn_1;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_1) : "f"(1.0f + _exp2_noftz_1));
            float2 _f2_0 = make_float2(_rcp_rn_0, _rcp_rn_1);
            float2 sigmoid = _f2_0;
            float2 _f2_1 = make_float2(x0, x1);
            float2 _f32x2_mul_rn_0;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&sigmoid));
            float2 act = _f32x2_mul_rn_0;
            float2 _f2_2 = make_float2(1.0f, 1.0f);
            float2 _f32x2_sub_rn_0;
            asm volatile("sub.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_sub_rn_0) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&sigmoid));
            float2 one_minus = _f32x2_sub_rn_0;
            float2 _f32x2_fma_rn_0;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&_f32x2_fma_rn_0) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&one_minus), "l"(*(const unsigned long long*)&sigmoid));
            float2 dact = _f32x2_fma_rn_0;
            float2 _f2_3 = make_float2(grad0, grad1);
            float2 grad_pair = _f2_3;
            float2 _f32x2_mul_rn_1;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_1) : "l"(*(const unsigned long long*)&dact), "l"(*(const unsigned long long*)&grad_pair));
            float2 _f2_4 = make_float2(up0, up1);
            float2 _f32x2_mul_rn_2;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_2) : "l"(*(const unsigned long long*)&_f32x2_mul_rn_1), "l"(*(const unsigned long long*)&_f2_4));
            float2 dgate = _f32x2_mul_rn_2;
            float2 _f32x2_mul_rn_3;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_3) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&grad_pair));
            float2 dup = _f32x2_mul_rn_3;
            pair_values[0] = dgate.x;
            pair_values[1] = dgate.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values[_lp*2 + 0], pair_values[_lp*2+1 + 0]));
                packed_act[_lp + (pair)] = *(uint32_t*)&_bf2;
            }
            uint32_t _bf16x2_abs_max_nan_0;
            asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_0) : "r"(amax_act), "r"(packed_act[pair]));
            amax_act = _bf16x2_abs_max_nan_0;
            pair_values[0] = dup.x;
            pair_values[1] = dup.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values[_lp*2 + 0], pair_values[_lp*2+1 + 0]));
                packed_gate[_lp + (pair)] = *(uint32_t*)&_bf2;
            }
            uint32_t _bf16x2_abs_max_nan_1;
            asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_1) : "r"(amax_gate), "r"(packed_gate[pair]));
            amax_gate = _bf16x2_abs_max_nan_1;
        }
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax_act & 2147450879, 1);
        unsigned int peer_act = _shfl_xor_0;
        uint32_t _bf16x2_max_nan_0;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_0) : "r"(amax_act & 2147450879), "r"(peer_act));
        amax_act = _bf16x2_max_nan_0;
        uint32_t _bf16x2_max_nan_1;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_1) : "r"(amax_act), "r"(amax_act >> 16));
        amax_act = _bf16x2_max_nan_1;
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, amax_gate & 2147450879, 1);
        unsigned int peer_gate = _shfl_xor_1;
        uint32_t _bf16x2_max_nan_2;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_2) : "r"(amax_gate & 2147450879), "r"(peer_gate));
        amax_gate = _bf16x2_max_nan_2;
        uint32_t _bf16x2_max_nan_3;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_3) : "r"(amax_gate), "r"(amax_gate >> 16));
        amax_gate = _bf16x2_max_nan_3;
        unsigned int act_bits = (amax_act & 65535) << 16;
        unsigned int gate_bits = (amax_gate & 65535) << 16;
        int act_scale_i32 = (int)(act_bits + 2031616 >> 23) - 8;
        int gate_scale_i32 = (int)(gate_bits + 2031616 >> 23) - 8;
        if (act_scale_i32 < 0) {
            act_scale_i32 = 0;
        }
        if (gate_scale_i32 < 0) {
            gate_scale_i32 = 0;
        }
        unsigned int act_exponent = act_bits & 2139095040;
        unsigned int gate_exponent = gate_bits & 2139095040;
        if (act_exponent == 2139095040) {
            act_scale_i32 = 255;
        }
        if (gate_exponent == 2139095040) {
            gate_scale_i32 = 255;
        }
        unsigned int act_scale = (unsigned int)act_scale_i32;
        unsigned int gate_scale = (unsigned int)gate_scale_i32;
        int scale_col = bx * 2 + blk;
        int num_scale_col_blocks = K / 64;
        int scale_base = ((grow >> 7) * num_scale_col_blocks + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
        int gate_scale_col = scale_col + K / 32;
        int gate_scale_index = ((grow >> 7) * num_scale_col_blocks + (gate_scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (gate_scale_col & 3);
        if (half == 0) {
            *((unsigned char*)(row_scales + scale_base)) = (unsigned char)(act_scale);
            *((unsigned char*)(row_scales + gate_scale_index)) = (unsigned char)(gate_scale);
        }
        unsigned int inv_act = 254 - act_scale << 7;
        unsigned int inv_gate = 254 - gate_scale << 7;
        if (act_scale == 255) {
            inv_act = 32704;
        }
        if (gate_scale == 255) {
            inv_gate = 32704;
        }
        inv_act = inv_act | inv_act << 16;
        inv_gate = inv_gate | inv_gate << 16;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t _bf16x2_mul_0;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed_act[2 * q]), "r"(inv_act));
            unsigned int scaled_act0 = _bf16x2_mul_0;
            uint32_t _bf16x2_mul_1;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed_act[2 * q + 1]), "r"(inv_act));
            unsigned int scaled_act1 = _bf16x2_mul_1;
            bits0 = (scaled_act0 & 65535) << 16;
            bits1 = scaled_act0 & 4294901760;
            bits2 = (scaled_act1 & 65535) << 16;
            bits3 = scaled_act1 & 4294901760;
            value0 = reinterpret_cast<float*>(&bits0)[0];
            value1 = reinterpret_cast<float*>(&bits1)[0];
            value2 = reinterpret_cast<float*>(&bits2)[0];
            value3 = reinterpret_cast<float*>(&bits3)[0];
            scaled_quad[0] = value0;
            scaled_quad[1] = value1;
            scaled_quad[2] = value2;
            scaled_quad[3] = value3;
            {
                uint32_t _packed;
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_packed) : "f"(scaled_quad[0]), "f"(scaled_quad[1]), "f"(scaled_quad[2]), "f"(scaled_quad[3]));
                packed_fp8_act[(q) + 0] = _packed;
            }
            uint32_t _bf16x2_mul_2;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(packed_gate[2 * q]), "r"(inv_gate));
            unsigned int scaled_gate0 = _bf16x2_mul_2;
            uint32_t _bf16x2_mul_3;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(packed_gate[2 * q + 1]), "r"(inv_gate));
            unsigned int scaled_gate1 = _bf16x2_mul_3;
            bits0 = (scaled_gate0 & 65535) << 16;
            bits1 = scaled_gate0 & 4294901760;
            bits2 = (scaled_gate1 & 65535) << 16;
            bits3 = scaled_gate1 & 4294901760;
            value0 = reinterpret_cast<float*>(&bits0)[0];
            value1 = reinterpret_cast<float*>(&bits1)[0];
            value2 = reinterpret_cast<float*>(&bits2)[0];
            value3 = reinterpret_cast<float*>(&bits3)[0];
            scaled_quad[0] = value0;
            scaled_quad[1] = value1;
            scaled_quad[2] = value2;
            scaled_quad[3] = value3;
            {
                uint32_t _packed;
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_packed) : "f"(scaled_quad[0]), "f"(scaled_quad[1]), "f"(scaled_quad[2]), "f"(scaled_quad[3]));
                packed_fp8_gate[(q) + 0] = _packed;
            }
        }
        int word_quad = blk * 2 + half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((row_act_addr + (unsigned int)(stage * 2048) + (unsigned int)(row * 64 + word_quad * 16))), "r"(packed_fp8_act[0]), "r"(packed_fp8_act[1]), "r"(packed_fp8_act[2]), "r"(packed_fp8_act[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((row_gate_addr + (unsigned int)(stage * 2048) + (unsigned int)(row * 64 + word_quad * 16))), "r"(packed_fp8_gate[0]), "r"(packed_fp8_gate[1]), "r"(packed_fp8_gate[2]), "r"(packed_fp8_gate[3]) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                tma_store_2d(&row_act_tma, bx * 64, by * 32 + stage * 32, row_act_addr + (unsigned int)(stage * 2048));
                tma_store_2d(&row_gate_tma, bx * 64, by * 32 + stage * 32, row_gate_addr + (unsigned int)(stage * 2048));
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
#undef SMEM_ROW_ACT_OFF
#undef SMEM_ROW_ACT_STAGE_BYTES
#undef SMEM_ROW_ACT_STRIDE
#undef SMEM_ROW_GATE_OFF
#undef SMEM_ROW_GATE_STAGE_BYTES
#undef SMEM_ROW_GATE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef row_act_addr
#undef row_gate_addr
