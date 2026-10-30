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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 640
#define SMEM_PARTIALS_STRIDE 640
#define SMEM_TOTAL 640
#define THREADS 128
#define LAUNCH_BOUNDS_THREADS 128
#define LAUNCH_BOUNDS_MIN_BLOCKS 4
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_prepare__t128(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ x, float* __restrict__ stats, __nv_bfloat16* __restrict__ db, float* __restrict__ gstats, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = warp;
    int lane_1 = lane;
    int group = tid / 128;
    int gtid = tid - group * 128;
    long long col0 = (long long)(gtid * 16);
    int it = 0;
    float zero = 0.0f;
    #pragma unroll 1
    for (int row0 = bid; row0 < T; row0 += num_bids) {
        int row = row0 + group;
        float accs[20];
        #pragma unroll
        for (int q = 0; q < 20; q++) {
            accs[q] = zero;
        }
        if (row < T) {
            long long row64 = (long long)row;
            long long xbase = row64 * 16384 + col0;
            long long bbase = row64 * 4096 + col0;
            unsigned int cache[144];
            #pragma unroll
            for (int r = 0; r < 9; r++) {
                #pragma unroll
                for (int v = 0; v < 2; v++) {
                    if (r < 4) {
                        {
                            asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                                : "=r"(cache[(r * 2 + v) * 8 + 0]), "=r"(cache[(r * 2 + v) * 8 + 1]), "=r"(cache[(r * 2 + v) * 8 + 2]), "=r"(cache[(r * 2 + v) * 8 + 3]), "=r"(cache[(r * 2 + v) * 8 + 4]), "=r"(cache[(r * 2 + v) * 8 + 5]), "=r"(cache[(r * 2 + v) * 8 + 6]), "=r"(cache[(r * 2 + v) * 8 + 7]) : "l"((const void*)((const char*)(dy + (xbase + (long long)(r * 4096 + v * 2048)) + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    } else if (r == 4) {
                        {
                            asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                                : "=r"(cache[(r * 2 + v) * 8 + 0]), "=r"(cache[(r * 2 + v) * 8 + 1]), "=r"(cache[(r * 2 + v) * 8 + 2]), "=r"(cache[(r * 2 + v) * 8 + 3]), "=r"(cache[(r * 2 + v) * 8 + 4]), "=r"(cache[(r * 2 + v) * 8 + 5]), "=r"(cache[(r * 2 + v) * 8 + 6]), "=r"(cache[(r * 2 + v) * 8 + 7]) : "l"((const void*)((const char*)(b + (bbase + (long long)(v * 2048)) + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    } else {
                        {
                            asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                                : "=r"(cache[(r * 2 + v) * 8 + 0]), "=r"(cache[(r * 2 + v) * 8 + 1]), "=r"(cache[(r * 2 + v) * 8 + 2]), "=r"(cache[(r * 2 + v) * 8 + 3]), "=r"(cache[(r * 2 + v) * 8 + 4]), "=r"(cache[(r * 2 + v) * 8 + 5]), "=r"(cache[(r * 2 + v) * 8 + 6]), "=r"(cache[(r * 2 + v) * 8 + 7]) : "l"((const void*)((const char*)(x + (xbase + (long long)((r - 5) * 4096 + v * 2048)) + 0) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
            long long sbase = row64 * 64;
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(stats + (sbase + 28) + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(stats + (sbase + 32) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float post[4];
            post[0] = _vec_load_0[1];
            post[1] = _vec_load_0[2];
            post[2] = _vec_load_0[3];
            post[3] = _vec_load_1[0];
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                unsigned int dbw[8];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(cache[v_1 * 8 + k]));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(post[0]), "f"(cvt_f32.x));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(post[0]), "f"(cvt_f32.y));
                    float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cache[(2 + v_1) * 8 + k]));
                    float _fma_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(post[1]), "f"(cvt_f32_0.x), "f"(_mul_0));
                    float _fma_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(post[1]), "f"(cvt_f32_0.y), "f"(_mul_1));
                    float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(cache[(4 + v_1) * 8 + k]));
                    float _fma_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(post[2]), "f"(cvt_f32_1.x), "f"(_fma_0));
                    float _fma_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(post[2]), "f"(cvt_f32_1.y), "f"(_fma_1));
                    float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(cache[(6 + v_1) * 8 + k]));
                    float _fma_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(post[3]), "f"(cvt_f32_2.x), "f"(_fma_2));
                    float _fma_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(post[3]), "f"(cvt_f32_2.y), "f"(_fma_3));
                    unsigned int hword = __as_u32(__float22bfloat162_rn(make_float2(_fma_5, _fma_6)));
                    dbw[k] = hword;
                }
                {
                    const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(dbw);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(reinterpret_cast<unsigned int*>(db + (bbase + (long long)(v_1 * 2048))))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                }
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 2; v_2++) {
                #pragma unroll
                for (int k_1 = 0; k_1 < 8; k_1++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float _dot_8;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_8) : "r"(cache[(j * 2 + v_2) * 8 + k_1]), "r"(cache[(8 + v_2) * 8 + k_1]), "f"(accs[j]));
                        accs[j] = _dot_8;
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            float _dot_9;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_9) : "r"(cache[((5 + i) * 2 + v_2) * 8 + k_1]), "r"(cache[(j * 2 + v_2) * 8 + k_1]), "f"(accs[4 + 4 * i + j]));
                            accs[4 + 4 * i + j] = _dot_9;
                        }
                    }
                }
            }
        }
        float zero_0 = 0.0f;
        float zero_1 = 0.0f;
        float zero_2 = 0.0f;
        float zero_3 = 0.0f;
        float zero_4 = 0.0f;
        float zero_5 = 0.0f;
        float zero_6 = 0.0f;
        float zero_7 = 0.0f;
        float zero_8 = 0.0f;
        float zero_9 = 0.0f;
        float zero_10 = 0.0f;
        float zero_11 = 0.0f;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? accs[0] : accs[16]);
        float keep = ((upper != 0) ? accs[16] : accs[0]);
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(keep), "f"(_shfl_xor_sync_10));
        float send_12 = ((upper != 0) ? accs[1] : accs[17]);
        float keep_13 = ((upper != 0) ? accs[17] : accs[1]);
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, send_12, 16);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(keep_13), "f"(_shfl_xor_sync_11));
        float send_14 = ((upper != 0) ? accs[2] : accs[18]);
        float keep_15 = ((upper != 0) ? accs[18] : accs[2]);
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, send_14, 16);
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(keep_15), "f"(_shfl_xor_sync_12));
        float send_16 = ((upper != 0) ? accs[3] : accs[19]);
        float keep_17 = ((upper != 0) ? accs[19] : accs[3]);
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, send_16, 16);
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(keep_17), "f"(_shfl_xor_sync_13));
        float send_18 = ((upper != 0) ? accs[4] : zero_0);
        float keep_19 = ((upper != 0) ? zero_0 : accs[4]);
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, send_18, 16);
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(keep_19), "f"(_shfl_xor_sync_14));
        float send_20 = ((upper != 0) ? accs[5] : zero_1);
        float keep_21 = ((upper != 0) ? zero_1 : accs[5]);
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, send_20, 16);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(keep_21), "f"(_shfl_xor_sync_15));
        float send_22 = ((upper != 0) ? accs[6] : zero_2);
        float keep_23 = ((upper != 0) ? zero_2 : accs[6]);
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, send_22, 16);
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(keep_23), "f"(_shfl_xor_sync_16));
        float send_24 = ((upper != 0) ? accs[7] : zero_3);
        float keep_25 = ((upper != 0) ? zero_3 : accs[7]);
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, send_24, 16);
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(keep_25), "f"(_shfl_xor_sync_17));
        float send_26 = ((upper != 0) ? accs[8] : zero_4);
        float keep_27 = ((upper != 0) ? zero_4 : accs[8]);
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, send_26, 16);
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(keep_27), "f"(_shfl_xor_sync_18));
        float send_28 = ((upper != 0) ? accs[9] : zero_5);
        float keep_29 = ((upper != 0) ? zero_5 : accs[9]);
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, send_28, 16);
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(keep_29), "f"(_shfl_xor_sync_19));
        float send_30 = ((upper != 0) ? accs[10] : zero_6);
        float keep_31 = ((upper != 0) ? zero_6 : accs[10]);
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, send_30, 16);
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(keep_31), "f"(_shfl_xor_sync_20));
        float send_32 = ((upper != 0) ? accs[11] : zero_7);
        float keep_33 = ((upper != 0) ? zero_7 : accs[11]);
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, send_32, 16);
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(keep_33), "f"(_shfl_xor_sync_21));
        float send_34 = ((upper != 0) ? accs[12] : zero_8);
        float keep_35 = ((upper != 0) ? zero_8 : accs[12]);
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, send_34, 16);
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(keep_35), "f"(_shfl_xor_sync_22));
        float send_36 = ((upper != 0) ? accs[13] : zero_9);
        float keep_37 = ((upper != 0) ? zero_9 : accs[13]);
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, send_36, 16);
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(keep_37), "f"(_shfl_xor_sync_23));
        float send_38 = ((upper != 0) ? accs[14] : zero_10);
        float keep_39 = ((upper != 0) ? zero_10 : accs[14]);
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, send_38, 16);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(keep_39), "f"(_shfl_xor_sync_24));
        float send_40 = ((upper != 0) ? accs[15] : zero_11);
        float keep_41 = ((upper != 0) ? zero_11 : accs[15]);
        float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, send_40, 16);
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(keep_41), "f"(_shfl_xor_sync_25));
        int upper_42 = lane_1 & 8;
        float send_43 = ((upper_42 != 0) ? _add_0 : _add_8);
        float keep_44 = ((upper_42 != 0) ? _add_8 : _add_0);
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, send_43, 8);
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(keep_44), "f"(_shfl_xor_sync_26));
        float send_45 = ((upper_42 != 0) ? _add_1 : _add_9);
        float keep_46 = ((upper_42 != 0) ? _add_9 : _add_1);
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, send_45, 8);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(keep_46), "f"(_shfl_xor_sync_27));
        float send_47 = ((upper_42 != 0) ? _add_2 : _add_10);
        float keep_48 = ((upper_42 != 0) ? _add_10 : _add_2);
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, send_47, 8);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(keep_48), "f"(_shfl_xor_sync_28));
        float send_49 = ((upper_42 != 0) ? _add_3 : _add_11);
        float keep_50 = ((upper_42 != 0) ? _add_11 : _add_3);
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, send_49, 8);
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(keep_50), "f"(_shfl_xor_sync_29));
        float send_51 = ((upper_42 != 0) ? _add_4 : _add_12);
        float keep_52 = ((upper_42 != 0) ? _add_12 : _add_4);
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, send_51, 8);
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(keep_52), "f"(_shfl_xor_sync_30));
        float send_53 = ((upper_42 != 0) ? _add_5 : _add_13);
        float keep_54 = ((upper_42 != 0) ? _add_13 : _add_5);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, send_53, 8);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(keep_54), "f"(_shfl_xor_sync_31));
        float send_55 = ((upper_42 != 0) ? _add_6 : _add_14);
        float keep_56 = ((upper_42 != 0) ? _add_14 : _add_6);
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, send_55, 8);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(keep_56), "f"(_shfl_xor_sync_32));
        float send_57 = ((upper_42 != 0) ? _add_7 : _add_15);
        float keep_58 = ((upper_42 != 0) ? _add_15 : _add_7);
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, send_57, 8);
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(keep_58), "f"(_shfl_xor_sync_33));
        int upper_59 = lane_1 & 4;
        float send_60 = ((upper_59 != 0) ? _add_16 : _add_20);
        float keep_61 = ((upper_59 != 0) ? _add_20 : _add_16);
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, send_60, 4);
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(keep_61), "f"(_shfl_xor_sync_34));
        float send_62 = ((upper_59 != 0) ? _add_17 : _add_21);
        float keep_63 = ((upper_59 != 0) ? _add_21 : _add_17);
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, send_62, 4);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(keep_63), "f"(_shfl_xor_sync_35));
        float send_64 = ((upper_59 != 0) ? _add_18 : _add_22);
        float keep_65 = ((upper_59 != 0) ? _add_22 : _add_18);
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, send_64, 4);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(keep_65), "f"(_shfl_xor_sync_36));
        float send_66 = ((upper_59 != 0) ? _add_19 : _add_23);
        float keep_67 = ((upper_59 != 0) ? _add_23 : _add_19);
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, send_66, 4);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(keep_67), "f"(_shfl_xor_sync_37));
        int upper_68 = lane_1 & 2;
        float send_69 = ((upper_68 != 0) ? _add_24 : _add_26);
        float keep_70 = ((upper_68 != 0) ? _add_26 : _add_24);
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, send_69, 2);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(keep_70), "f"(_shfl_xor_sync_38));
        float send_71 = ((upper_68 != 0) ? _add_25 : _add_27);
        float keep_72 = ((upper_68 != 0) ? _add_27 : _add_25);
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, send_71, 2);
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(keep_72), "f"(_shfl_xor_sync_39));
        int upper_73 = lane_1 & 1;
        float send_74 = ((upper_73 != 0) ? _add_28 : _add_29);
        float keep_75 = ((upper_73 != 0) ? _add_29 : _add_28);
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, send_74, 1);
        float red;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(red) : "f"(keep_75), "f"(_shfl_xor_sync_40));
        int par = (it & 1) * 80;
        if (lane_1 < 20) {
            partials[par + warp_0 * 20 + lane_1] = red;
        }
        __syncthreads();
        it += 1;
        if (row < T) {
            if (gtid < 32) {
                float vals[1];
                vals[0] = zero;
                if (lane_1 < 20) {
                    float p0 = partials[par + group * 4 * 20 + lane_1];
                    vals[0] = p0;
                    #pragma unroll
                    for (int w = 1; w < 4; w++) {
                        float pw = partials[par + (group * 4 + w) * 20 + lane_1];
                        float _add_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(vals[0]), "f"(pw));
                        vals[0] = _add_41;
                    }
                }
                long long gidx = (long long)row * 32 + (long long)lane_1;
                gstats[gidx] = vals[0];
            }
        }
    }
}

} // extern "C"
