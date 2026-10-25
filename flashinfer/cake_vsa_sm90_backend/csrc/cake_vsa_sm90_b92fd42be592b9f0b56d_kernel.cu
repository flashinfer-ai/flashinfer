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
#define SMEM_RED_OFF 0
#define SMEM_RED_STAGE_BYTES 16
#define SMEM_RED_STRIDE 16
#define SMEM_VMAX_SMEM_OFF 16
#define SMEM_VMAX_SMEM_STAGE_BYTES 2048
#define SMEM_VMAX_SMEM_STRIDE 2048
#define SMEM_TOTAL 2176
#define THREADS 128

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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_vsa_sm90_b92fd42be592b9f0b56d(__nv_bfloat16* __restrict__ q_in, __nv_bfloat16* __restrict__ k_in, __nv_bfloat16* __restrict__ v_in, uint8_t* __restrict__ q_out, uint8_t* __restrict__ k_out, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_amax, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int grid_heads, int k_scale_groups, int chunks_per_bh, int q_chunks, int k_chunks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;
    float* vmax_smem = reinterpret_cast<float*>(smem_raw + 16);
    const int vmax_smem_addr = smem + 16;

    // === Task calls (declaration order) ===
    int head = bid % grid_heads;
    int batch_chunk = bid / grid_heads;
    int chunk = batch_chunk % chunks_per_bh;
    int batch = batch_chunk / chunks_per_bh;
    int bh = batch * num_heads + head;
    int kv_bh = batch * num_kv_heads + head;
    int warp_0 = warp;
    int lane_1 = lane;
    int half = lane_1 / 8;
    int sub = lane_1 % 8;
    int channel0 = sub * 16;
    int token0 = chunk * 64 + warp_0 * 16 + half;
    int q_active = ((chunk < q_chunks && head < num_heads) ? 1 : 0);
    int kv_active = ((chunk < k_chunks && head < num_kv_heads) ? 1 : 0);
    unsigned int qw[32];
    unsigned int kw[32];
    unsigned int vw[32];
    int q_off[4];
    int q_ok[4];
    int k_off[4];
    int k_ok[4];
    if (q_active != 0) {
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int token = token0 + p * 4;
            int in_range = ((token < seqlen_q) ? 1 : 0);
            int load_token = ((in_range != 0) ? token : seqlen_q - 1);
            int offset = ((batch * seqlen_q + load_token) * num_heads + head) * 128 + channel0;
            q_ok[p] = in_range;
            q_off[p] = offset;
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q_in + q_off[p] + 0);
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&qw[p * 8]);
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
    }
    if (kv_active != 0) {
        #pragma unroll
        for (int p_1 = 0; p_1 < 4; p_1++) {
            int token_1 = token0 + p_1 * 4;
            int in_range_1 = ((token_1 < seqlen_k) ? 1 : 0);
            int load_token_1 = ((in_range_1 != 0) ? token_1 : seqlen_k - 1);
            int offset_1 = ((batch * seqlen_k + load_token_1) * num_kv_heads + head) * 128 + channel0;
            k_ok[p_1] = in_range_1;
            k_off[p_1] = offset_1;
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k_in + k_off[p_1] + 0);
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&kw[p_1 * 8]);
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(v_in + k_off[p_1] + 0);
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&vw[p_1 * 8]);
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vdst_2[_blk] = _vptr_2[_blk];
                }
            }
        }
    }
    if (q_active != 0) {
        #pragma unroll
        for (int p_2 = 0; p_2 < 4; p_2++) {
            int token_2 = token0 + p_2 * 4;
            uint32_t _bf16x2_abs_0;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(qw[p_2 * 8]));
            unsigned int packed = _bf16x2_abs_0;
            #pragma unroll
            for (int j = 1; j < 8; j++) {
                uint32_t _bf16x2_abs_1;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(qw[p_2 * 8 + j]));
                uint32_t _bf16x2_max_0;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(packed), "r"(_bf16x2_abs_1));
                packed = _bf16x2_max_0;
            }
            float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(packed));
            float amax = fmaxf(_cvt_f32_0.x, _cvt_f32_0.y);
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, amax, 4);
            amax = fmaxf(amax, _shfl_xor_sync_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
            amax = fmaxf(amax, _shfl_xor_sync_4);
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
            amax = fmaxf(amax, _shfl_xor_sync_5);
            amax = fmaxf(amax, 1e-06f);
            float values[16];
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(qw[p_2 * 8 + j_1]));
                values[2 * j_1] = _cvt_f32_1.x;
                values[2 * j_1 + 1] = _cvt_f32_1.y;
            }
            float _mul_operand_0 = 448.0f / amax;
            values[0] = mul_f32_rn_noftz(values[0], _mul_operand_0);
            values[1] = mul_f32_rn_noftz(values[1], _mul_operand_0);
            values[2] = mul_f32_rn_noftz(values[2], _mul_operand_0);
            values[3] = mul_f32_rn_noftz(values[3], _mul_operand_0);
            values[4] = mul_f32_rn_noftz(values[4], _mul_operand_0);
            values[5] = mul_f32_rn_noftz(values[5], _mul_operand_0);
            values[6] = mul_f32_rn_noftz(values[6], _mul_operand_0);
            values[7] = mul_f32_rn_noftz(values[7], _mul_operand_0);
            values[8] = mul_f32_rn_noftz(values[8], _mul_operand_0);
            values[9] = mul_f32_rn_noftz(values[9], _mul_operand_0);
            values[10] = mul_f32_rn_noftz(values[10], _mul_operand_0);
            values[11] = mul_f32_rn_noftz(values[11], _mul_operand_0);
            values[12] = mul_f32_rn_noftz(values[12], _mul_operand_0);
            values[13] = mul_f32_rn_noftz(values[13], _mul_operand_0);
            values[14] = mul_f32_rn_noftz(values[14], _mul_operand_0);
            values[15] = mul_f32_rn_noftz(values[15], _mul_operand_0);
            if (q_ok[p_2] != 0) {
                unsigned int _packed_0[4];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values[0]), "f"(values[1]),
                                           "f"(values[2]), "f"(values[3]));
                    _packed_0[0] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values[4]), "f"(values[5]),
                                           "f"(values[6]), "f"(values[7]));
                    _packed_0[1] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values[8]), "f"(values[9]),
                                           "f"(values[10]), "f"(values[11]));
                    _packed_0[2] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values[12]), "f"(values[13]),
                                           "f"(values[14]), "f"(values[15]));
                    _packed_0[3] = _packed;
                }
                reinterpret_cast<int4*>(q_out + q_off[p_2])[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
            if (q_ok[p_2] != 0 && sub == 0) {
                q_scale[bh * seqlen_q + token_2] = amax / 448.0f;
            }
        }
    }
    if (kv_active != 0) {
        int group = warp_0;
        int group_index = chunk * 4 + group;
        unsigned int kmax = 0;
        unsigned int vmax[8];
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            vmax[j_2] = 0;
        }
        #pragma unroll
        for (int p_3 = 0; p_3 < 4; p_3++) {
            uint32_t _bf16x2_abs_2;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(kw[p_3 * 8]));
            unsigned int kabs = _bf16x2_abs_2;
            #pragma unroll
            for (int j_3 = 1; j_3 < 8; j_3++) {
                uint32_t _bf16x2_abs_3;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(kw[p_3 * 8 + j_3]));
                uint32_t _bf16x2_max_1;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(kabs), "r"(_bf16x2_abs_3));
                kabs = _bf16x2_max_1;
            }
            unsigned int masked = ((k_ok[p_3] != 0) ? kabs : 0);
            uint32_t _bf16x2_max_2;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(kmax), "r"(masked));
            kmax = _bf16x2_max_2;
            #pragma unroll
            for (int j_4 = 0; j_4 < 8; j_4++) {
                uint32_t _bf16x2_abs_4;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(vw[p_3 * 8 + j_4]));
                unsigned int vmasked = ((k_ok[p_3] != 0) ? _bf16x2_abs_4 : 0);
                uint32_t _bf16x2_max_3;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(vmax[j_4]), "r"(vmasked));
                vmax[j_4] = _bf16x2_max_3;
            }
        }
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(kmax));
        float group_max = fmaxf(_cvt_f32_2.x, _cvt_f32_2.y);
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, group_max, 16);
        group_max = fmaxf(group_max, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, group_max, 8);
        group_max = fmaxf(group_max, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, group_max, 4);
        group_max = fmaxf(group_max, _shfl_xor_sync_8);
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
        group_max = fmaxf(group_max, _shfl_xor_sync_9);
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
        group_max = fmaxf(group_max, _shfl_xor_sync_10);
        group_max = fmaxf(group_max, 1e-06f);
        #pragma unroll
        for (int p_4 = 0; p_4 < 4; p_4++) {
            float values_1[16];
            #pragma unroll
            for (int j_5 = 0; j_5 < 8; j_5++) {
                float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(kw[p_4 * 8 + j_5]));
                values_1[2 * j_5] = _cvt_f32_3.x;
                values_1[2 * j_5 + 1] = _cvt_f32_3.y;
            }
            float _mul_operand_1 = 448.0f / group_max;
            values_1[0] = mul_f32_rn_noftz(values_1[0], _mul_operand_1);
            values_1[1] = mul_f32_rn_noftz(values_1[1], _mul_operand_1);
            values_1[2] = mul_f32_rn_noftz(values_1[2], _mul_operand_1);
            values_1[3] = mul_f32_rn_noftz(values_1[3], _mul_operand_1);
            values_1[4] = mul_f32_rn_noftz(values_1[4], _mul_operand_1);
            values_1[5] = mul_f32_rn_noftz(values_1[5], _mul_operand_1);
            values_1[6] = mul_f32_rn_noftz(values_1[6], _mul_operand_1);
            values_1[7] = mul_f32_rn_noftz(values_1[7], _mul_operand_1);
            values_1[8] = mul_f32_rn_noftz(values_1[8], _mul_operand_1);
            values_1[9] = mul_f32_rn_noftz(values_1[9], _mul_operand_1);
            values_1[10] = mul_f32_rn_noftz(values_1[10], _mul_operand_1);
            values_1[11] = mul_f32_rn_noftz(values_1[11], _mul_operand_1);
            values_1[12] = mul_f32_rn_noftz(values_1[12], _mul_operand_1);
            values_1[13] = mul_f32_rn_noftz(values_1[13], _mul_operand_1);
            values_1[14] = mul_f32_rn_noftz(values_1[14], _mul_operand_1);
            values_1[15] = mul_f32_rn_noftz(values_1[15], _mul_operand_1);
            if (k_ok[p_4] != 0) {
                unsigned int _packed_1[4];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values_1[0]), "f"(values_1[1]),
                                           "f"(values_1[2]), "f"(values_1[3]));
                    _packed_1[0] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values_1[4]), "f"(values_1[5]),
                                           "f"(values_1[6]), "f"(values_1[7]));
                    _packed_1[1] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values_1[8]), "f"(values_1[9]),
                                           "f"(values_1[10]), "f"(values_1[11]));
                    _packed_1[2] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(values_1[12]), "f"(values_1[13]),
                                           "f"(values_1[14]), "f"(values_1[15]));
                    _packed_1[3] = _packed;
                }
                reinterpret_cast<int4*>(k_out + k_off[p_4])[0] = reinterpret_cast<int4*>(_packed_1)[0];
            }
        }
        if (lane_1 == 0 && group_index < k_scale_groups) {
            k_scale[kv_bh * k_scale_groups + group_index] = group_max / 448.0f;
        }
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            unsigned int _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_6], 8);
            uint32_t _bf16x2_max_4;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(vmax[j_6]), "r"(_shfl_xor_sync_11));
            vmax[j_6] = _bf16x2_max_4;
        }
        #pragma unroll
        for (int j_7 = 0; j_7 < 8; j_7++) {
            unsigned int _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_7], 16);
            uint32_t _bf16x2_max_5;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(vmax[j_7]), "r"(_shfl_xor_sync_12));
            vmax[j_7] = _bf16x2_max_5;
        }
        if (half == 0) {
            #pragma unroll
            for (int j_8 = 0; j_8 < 8; j_8++) {
                float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vmax[j_8]));
                vmax_smem[warp_0 * 128 + channel0 + 2 * j_8] = _cvt_f32_4.x;
                vmax_smem[warp_0 * 128 + channel0 + 2 * j_8 + 1] = _cvt_f32_4.y;
            }
        }
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        if (tid < 128) {
            float cta_max = vmax_smem[tid];
            #pragma unroll
            for (int other = 1; other < 4; other++) {
                cta_max = fmaxf(cta_max, vmax_smem[other * 128 + tid]);
            }
            atomicMax(&reinterpret_cast<int*>(v_amax)[kv_bh * 128 + tid], (int)__as_u32(cta_max));
        }
    }
}

} // extern "C"
