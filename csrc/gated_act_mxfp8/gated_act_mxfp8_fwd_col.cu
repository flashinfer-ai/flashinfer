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

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}

__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
#define SMEM_GATE_SMEM_OFF 1024
#define SMEM_GATE_SMEM_STAGE_BYTES 4096
#define SMEM_GATE_SMEM_STRIDE 4096
#define SMEM_UP_SMEM_OFF 5120
#define SMEM_UP_SMEM_STAGE_BYTES 4096
#define SMEM_UP_SMEM_STRIDE 4096
#define SMEM_COL_OUTPUT_OFF 9216
#define SMEM_COL_OUTPUT_STAGE_BYTES 2048
#define SMEM_COL_OUTPUT_STRIDE 2048
#define SMEM_SUB_AMAX_OFF 11264
#define SMEM_SUB_AMAX_STAGE_BYTES 256
#define SMEM_SUB_AMAX_STRIDE 256
#define SMEM_TOTAL 11520
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gated_act_mxfp8_fwd_col_staged_64x64(__grid_constant__ CUtensorMap const gate_tma, __grid_constant__ CUtensorMap const up_tma, __grid_constant__ CUtensorMap const col_output_tma, uint8_t* __restrict__ col_scales, int M, int K)
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
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int gate_smem_addr = smem + 1024;
    __nv_bfloat16* up_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int up_smem_addr = smem + 5120;
    uint8_t* col_output = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int col_output_addr = smem + 9216;
    unsigned int* sub_amax = reinterpret_cast<unsigned int*>(smem_raw + 11264);
    const int sub_amax_addr = smem + 11264;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // input_full: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 0, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define input_full_addr (mbar_base + 0)

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int col = tid_0 & 63;
    int ty = tid_0 >> 6;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    unsigned int partial_word[1];
    float pair_values[2];
    unsigned int packed_act[8];
    float scaled_quad[4];
    unsigned int packed_fp8[4];
    float x0 = 0.0f;
    float x1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    unsigned int bits0 = 0;
    unsigned int bits1 = 0;
    unsigned int bits2 = 0;
    unsigned int bits3 = 0;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;
    if (warp == 0) {
        if (elect_sync()) {
            tma_2d_gmem2smem(gate_smem_addr, &gate_tma, bx * 64, by * 32, input_full_addr);
            tma_2d_gmem2smem(up_smem_addr, &up_tma, bx * 64, by * 32, input_full_addr);
        }
    }
    if (tid_0 == 0) {
        mbarrier_arrive_expect_tx(input_full_addr, 8192);
    } else {
        mbarrier_arrive(input_full_addr);
    }
    #pragma unroll
    for (int stage = 0; stage < 1; stage++) {
        if (stage + 1 < 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    tma_2d_gmem2smem(gate_smem_addr + (unsigned int)((stage + 1) * 4096), &gate_tma, bx * 64, by * 32 + (stage + 1) * 32, input_full_addr + (stage + 1) * 8);
                    tma_2d_gmem2smem(up_smem_addr + (unsigned int)((stage + 1) * 4096), &up_tma, bx * 64, by * 32 + (stage + 1) * 32, input_full_addr + (stage + 1) * 8);
                }
            }
            if (tid_0 == 0) {
                mbarrier_arrive_expect_tx(input_full_addr + (stage + 1) * 8, 8192);
            } else {
                mbarrier_arrive(input_full_addr + (stage + 1) * 8);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_wait(input_full_addr + (stage) * 8, 0);
        unsigned int amax_act = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; pair++) {
            int rlo = ty * 16 + 2 * pair;
            int elem_offset = stage * 2048 + rlo * 64 + col;
            float _cvt_f32_0 = __bfloat162float(gate_smem[elem_offset]);
            x0 = _cvt_f32_0;
            x1 = (float)gate_smem[elem_offset + 64];
            float _cvt_f32_1 = __bfloat162float(up_smem[elem_offset]);
            up0 = _cvt_f32_1;
            up1 = (float)up_smem[elem_offset + 64];
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
            float2 _f2_2 = make_float2(up0, up1);
            float2 _f32x2_mul_rn_1;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&_f2_2));
            float2 output = _f32x2_mul_rn_1;
            pair_values[0] = output.x;
            pair_values[1] = output.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values[_lp*2 + 0], pair_values[_lp*2+1 + 0]));
                packed_act[_lp + (pair)] = *(uint32_t*)&_bf2;
            }
            uint32_t _bf16x2_abs_max_nan_0;
            asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_0) : "r"(amax_act), "r"(packed_act[pair]));
            amax_act = _bf16x2_abs_max_nan_0;
        }
        amax_act = amax_act & 2147450879;
        uint32_t _bf16x2_max_nan_0;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_0) : "r"(amax_act), "r"(amax_act >> 16));
        amax_act = _bf16x2_max_nan_0;
        int sub_addr = sub_amax_addr + (unsigned int)(col * 4);
        if (ty > 0) {
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(sub_addr), "r"(amax_act));
        }
        __syncthreads();
        if (ty == 0) {
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&partial_word[0])) : "r"(sub_addr));
            uint32_t _bf16x2_max_nan_1;
            asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_1) : "r"(amax_act), "r"(partial_word[0]));
            amax_act = _bf16x2_max_nan_1;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(sub_addr), "r"(amax_act));
        }
        __syncthreads();
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&partial_word[0])) : "r"(sub_addr));
        amax_act = partial_word[0];
        unsigned int amax_bits = (amax_act & 65535) << 16;
        int scale_i32 = (int)(amax_bits + 2031616 >> 23) - 8;
        if (scale_i32 < 0) {
            scale_i32 = 0;
        }
        unsigned int exponent_bits = amax_bits & 2139095040;
        if (exponent_bits == 2139095040) {
            scale_i32 = 255;
        }
        unsigned int scale = (unsigned int)scale_i32;
        int row_tile = by + stage;
        int out_col = bx * 64 + col;
        int num_scale_blocks = M / 128;
        int scale_index = ((out_col >> 7) * num_scale_blocks + (row_tile >> 2)) * 512 + (out_col & 31) * 16 + (out_col >> 5 & 3) * 4 + (row_tile & 3);
        if (ty == 0) {
            *((unsigned char*)(col_scales + scale_index)) = (unsigned char)(scale);
        }
        unsigned int inv = 254 - scale << 7;
        if (scale == 255) {
            inv = 32704;
        }
        inv = inv | inv << 16;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t _bf16x2_mul_0;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed_act[2 * q]), "r"(inv));
            unsigned int scaled0 = _bf16x2_mul_0;
            uint32_t _bf16x2_mul_1;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed_act[2 * q + 1]), "r"(inv));
            unsigned int scaled1 = _bf16x2_mul_1;
            bits0 = (scaled0 & 65535) << 16;
            bits1 = scaled0 & 4294901760;
            bits2 = (scaled1 & 65535) << 16;
            bits3 = scaled1 & 4294901760;
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
                packed_fp8[(q) + 0] = _packed;
            }
        }
        int word_base = ty * 4;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((col_output_addr + (unsigned int)(stage * 2048) + (unsigned int)(col * 32 + word_base * 4))), "r"(packed_fp8[0]), "r"(packed_fp8[1]), "r"(packed_fp8[2]), "r"(packed_fp8[3]) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                tma_store_2d(&col_output_tma, by * 32 + stage * 32, bx * 64, col_output_addr + (unsigned int)(stage * 2048));
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int barrier_stage = 0; barrier_stage < 1; barrier_stage++) {
                asm volatile(
                    "mbarrier.inval.shared::cta.b64 [%0];"
                    :: "r"(input_full_addr + barrier_stage * 8) : "memory");
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#undef GATED_MXFP8_INF
#undef NUM_MAIN_STAGES
#undef SMEM_COL_OUTPUT_OFF
#undef SMEM_COL_OUTPUT_STAGE_BYTES
#undef SMEM_COL_OUTPUT_STRIDE
#undef SMEM_GATE_SMEM_OFF
#undef SMEM_GATE_SMEM_STAGE_BYTES
#undef SMEM_GATE_SMEM_STRIDE
#undef SMEM_SUB_AMAX_OFF
#undef SMEM_SUB_AMAX_STAGE_BYTES
#undef SMEM_SUB_AMAX_STRIDE
#undef SMEM_TOTAL
#undef SMEM_UP_SMEM_OFF
#undef SMEM_UP_SMEM_STAGE_BYTES
#undef SMEM_UP_SMEM_STRIDE
#undef THREADS
#undef col_output_addr
#undef gate_smem_addr
#undef input_full_addr
#undef sub_amax_addr
#undef up_smem_addr
