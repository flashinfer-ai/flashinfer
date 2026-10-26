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
#define NUM_MAIN_1_STAGES 1
#define SMEM_AMAX_PART_OFF 0
#define SMEM_AMAX_PART_STAGE_BYTES 64
#define SMEM_AMAX_PART_STRIDE 64
#define SMEM_AMAX_PART_Q_OFF 64
#define SMEM_AMAX_PART_Q_STAGE_BYTES 32
#define SMEM_AMAX_PART_Q_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_fp8_attention_bf0833d889569b5406df(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q8, unsigned int* __restrict__ k8, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, int seqlen, int num_heads, int num_blocks, int q_stride_b, int q_stride_s, int q_stride_h, int k_stride_b, int k_stride_s, int k_stride_h, int v_stride_b, int v_stride_s, int v_stride_h)
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
    float* amax_part = reinterpret_cast<float*>(smem_raw + 0);
    const int amax_part_addr = smem + 0;
    float* amax_part_q = reinterpret_cast<float*>(smem_raw + 64);
    const int amax_part_q_addr = smem + 64;

    // === Task calls (declaration order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    int blk = bid % num_blocks;
    int bh = bid / num_blocks;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int s_pad = num_blocks * 128;
    int tok_base = blk * 128;
    int q_base = batch * q_stride_b + head * q_stride_h + col16;
    int k_base = batch * k_stride_b + head * k_stride_h + col16;
    float kvals[64];
    for (int ps = 0; ps < 4; ps++) {
        int tok = tok_base + ps * 32 + trow;
        int tok_c = ((tok < seqlen) ? tok : seqlen - 1);
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + (k_base + tok_c * k_stride_s) + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        for (int e = 0; e < 16; e++) {
            float kf = _vec_load_0[e];
            kvals[ps * 16 + e] = kf;
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float mk[16];
    for (int quad = 0; quad < 4; quad++) {
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (bh * 128 + col16 + quad * 4) + 0);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
        }
        for (int e_1 = 0; e_1 < 4; e_1++) {
            mk[quad * 4 + e_1] = _vec_load_1[e_1];
        }
    }
    float kmax0 = 0.0f;
    float kmax1 = 0.0f;
    for (int ps_1 = 0; ps_1 < 4; ps_1++) {
        int tok_1 = tok_base + ps_1 * 32 + trow;
        int valid = ((tok_1 < seqlen) ? 1 : 0);
        for (int e_2 = 0; e_2 < 16; e_2++) {
            float kc = kvals[ps_1 * 16 + e_2] - mk[e_2];
            kvals[ps_1 * 16 + e_2] = ((valid == 1) ? kc : 0.0f);
            if (ps_1 >= 2) {
                kmax1 = fmaxf(kmax1, fabsf(kvals[ps_1 * 16 + e_2]));
            } else {
                kmax0 = fmaxf(kmax0, fabsf(kvals[ps_1 * 16 + e_2]));
            }
        }
    }
    for (int stage = 0; stage < 5; stage++) {
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, kmax0, 16 >> stage);
        kmax0 = fmaxf(kmax0, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, kmax1, 16 >> stage);
        kmax1 = fmaxf(kmax1, _shfl_xor_sync_3);
    }
    if (lane == 0) {
        amax_part[2 * warp] = kmax0;
        amax_part[2 * warp + 1] = kmax1;
    }
    __syncthreads();
    kmax0 = amax_part[2 * (lane & 7)];
    kmax1 = amax_part[2 * (lane & 7) + 1];
    for (int stage_1 = 0; stage_1 < 3; stage_1++) {
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, kmax0, 4 >> stage_1);
        kmax0 = fmaxf(kmax0, _shfl_xor_sync_4);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, kmax1, 4 >> stage_1);
        kmax1 = fmaxf(kmax1, _shfl_xor_sync_5);
    }
    kmax0 = fmaxf(kmax0, 1e-12f);
    kmax1 = fmaxf(kmax1, 1e-12f);
    float kinv0 = __fdiv_rn(127.0f, kmax0);
    float kinv1 = __fdiv_rn(127.0f, kmax1);
    if (tid == 0) {
        float _fdiv_6 = __fdiv_rn(kmax0, 127.0f);
        *(reinterpret_cast<float*>(k_scale + (2 * bid)) + (0)) = _fdiv_6;
        float _fdiv_7 = __fdiv_rn(kmax1, 127.0f);
        *(reinterpret_cast<float*>(k_scale + (2 * bid + 1)) + (0)) = _fdiv_7;
    }
    for (int ps_2 = 0; ps_2 < 4; ps_2++) {
        int tok2 = tok_base + ps_2 * 32 + trow;
        unsigned int kw[4];
        float k_x00 = fminf(fmaxf(kvals[ps_2 * 16] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m00 = k_x00 + 1.2582912e+07f;
        float k_x01 = fminf(fmaxf(kvals[ps_2 * 16 + 1] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m01 = k_x01 + 1.2582912e+07f;
        float k_x02 = fminf(fmaxf(kvals[ps_2 * 16 + 2] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m02 = k_x02 + 1.2582912e+07f;
        float k_x03 = fminf(fmaxf(kvals[ps_2 * 16 + 3] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m03 = k_x03 + 1.2582912e+07f;
        uint32_t _prmt_9;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_9) : "r"(__as_u32(k_m00)), "r"(__as_u32(k_m01)));
        uint32_t _prmt_10;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_10) : "r"(__as_u32(k_m02)), "r"(__as_u32(k_m03)));
        uint32_t _prmt_8;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_8) : "r"(_prmt_9), "r"(_prmt_10));
        kw[0] = _prmt_8;
        float k_x10 = fminf(fmaxf(kvals[ps_2 * 16 + 4] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m10 = k_x10 + 1.2582912e+07f;
        float k_x11 = fminf(fmaxf(kvals[ps_2 * 16 + 4 + 1] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m11 = k_x11 + 1.2582912e+07f;
        float k_x12 = fminf(fmaxf(kvals[ps_2 * 16 + 4 + 2] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m12 = k_x12 + 1.2582912e+07f;
        float k_x13 = fminf(fmaxf(kvals[ps_2 * 16 + 4 + 3] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m13 = k_x13 + 1.2582912e+07f;
        uint32_t _prmt_12;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_12) : "r"(__as_u32(k_m10)), "r"(__as_u32(k_m11)));
        uint32_t _prmt_13;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_13) : "r"(__as_u32(k_m12)), "r"(__as_u32(k_m13)));
        uint32_t _prmt_11;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_11) : "r"(_prmt_12), "r"(_prmt_13));
        kw[1] = _prmt_11;
        float k_x20 = fminf(fmaxf(kvals[ps_2 * 16 + 8] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m20 = k_x20 + 1.2582912e+07f;
        float k_x21 = fminf(fmaxf(kvals[ps_2 * 16 + 8 + 1] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m21 = k_x21 + 1.2582912e+07f;
        float k_x22 = fminf(fmaxf(kvals[ps_2 * 16 + 8 + 2] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m22 = k_x22 + 1.2582912e+07f;
        float k_x23 = fminf(fmaxf(kvals[ps_2 * 16 + 8 + 3] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m23 = k_x23 + 1.2582912e+07f;
        uint32_t _prmt_15;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_15) : "r"(__as_u32(k_m20)), "r"(__as_u32(k_m21)));
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_16) : "r"(__as_u32(k_m22)), "r"(__as_u32(k_m23)));
        uint32_t _prmt_14;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_14) : "r"(_prmt_15), "r"(_prmt_16));
        kw[2] = _prmt_14;
        float k_x30 = fminf(fmaxf(kvals[ps_2 * 16 + 12] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m30 = k_x30 + 1.2582912e+07f;
        float k_x31 = fminf(fmaxf(kvals[ps_2 * 16 + 12 + 1] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m31 = k_x31 + 1.2582912e+07f;
        float k_x32 = fminf(fmaxf(kvals[ps_2 * 16 + 12 + 2] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m32 = k_x32 + 1.2582912e+07f;
        float k_x33 = fminf(fmaxf(kvals[ps_2 * 16 + 12 + 3] * ((ps_2 >= 2) ? kinv1 : kinv0), -127.0f), 127.0f);
        float k_m33 = k_x33 + 1.2582912e+07f;
        uint32_t _prmt_18;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_18) : "r"(__as_u32(k_m30)), "r"(__as_u32(k_m31)));
        uint32_t _prmt_19;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_19) : "r"(__as_u32(k_m32)), "r"(__as_u32(k_m33)));
        uint32_t _prmt_17;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_17) : "r"(_prmt_18), "r"(_prmt_19));
        kw[3] = _prmt_17;
        reinterpret_cast<int4*>(k8 + ((bh * s_pad + tok2) * 128 + col16 >> 2))[0] = reinterpret_cast<int4*>(kw)[0];
    }
    float qvals[64];
    float qmax = 0.0f;
    for (int ps_3 = 0; ps_3 < 4; ps_3++) {
        int tok_2 = tok_base + ps_3 * 32 + trow;
        int valid_1 = ((tok_2 < seqlen) ? 1 : 0);
        int tok_c_1 = ((tok_2 < seqlen) ? tok_2 : seqlen - 1);
        float _vec_load_2[16];
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(Q + (q_base + tok_c_1 * q_stride_s) + 0);
            uint4 _vld_20[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_20[_blk] = _vptr_20[_blk];
                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                }
            }
        }
        for (int e_3 = 0; e_3 < 16; e_3++) {
            float qf = _vec_load_2[e_3];
            qvals[ps_3 * 16 + e_3] = ((valid_1 == 1) ? qf : 0.0f);
            qmax = fmaxf(qmax, fabsf(qvals[ps_3 * 16 + e_3]));
        }
    }
    for (int stage_2 = 0; stage_2 < 5; stage_2++) {
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, qmax, 16 >> stage_2);
        qmax = fmaxf(qmax, _shfl_xor_sync_21);
    }
    if (lane == 0) {
        amax_part_q[warp] = qmax;
    }
    __syncthreads();
    qmax = amax_part_q[lane & 7];
    for (int stage_3 = 0; stage_3 < 3; stage_3++) {
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, qmax, 4 >> stage_3);
        qmax = fmaxf(qmax, _shfl_xor_sync_22);
    }
    qmax = fmaxf(qmax, 1e-12f);
    float qinv = __fdiv_rn(127.0f, qmax);
    if (tid == 0) {
        float _fdiv_23 = __fdiv_rn(qmax, 127.0f);
        *(reinterpret_cast<float*>(q_scale + bid) + (0)) = _fdiv_23;
    }
    for (int ps_4 = 0; ps_4 < 4; ps_4++) {
        int tok_3 = tok_base + ps_4 * 32 + trow;
        unsigned int qw[4];
        float q_x00 = fminf(fmaxf(qvals[ps_4 * 16] * qinv, -127.0f), 127.0f);
        float q_m00 = q_x00 + 1.2582912e+07f;
        float q_x01 = fminf(fmaxf(qvals[ps_4 * 16 + 1] * qinv, -127.0f), 127.0f);
        float q_m01 = q_x01 + 1.2582912e+07f;
        float q_x02 = fminf(fmaxf(qvals[ps_4 * 16 + 2] * qinv, -127.0f), 127.0f);
        float q_m02 = q_x02 + 1.2582912e+07f;
        float q_x03 = fminf(fmaxf(qvals[ps_4 * 16 + 3] * qinv, -127.0f), 127.0f);
        float q_m03 = q_x03 + 1.2582912e+07f;
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_25) : "r"(__as_u32(q_m00)), "r"(__as_u32(q_m01)));
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_26) : "r"(__as_u32(q_m02)), "r"(__as_u32(q_m03)));
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_24) : "r"(_prmt_25), "r"(_prmt_26));
        qw[0] = _prmt_24;
        float q_x10 = fminf(fmaxf(qvals[ps_4 * 16 + 4] * qinv, -127.0f), 127.0f);
        float q_m10 = q_x10 + 1.2582912e+07f;
        float q_x11 = fminf(fmaxf(qvals[ps_4 * 16 + 4 + 1] * qinv, -127.0f), 127.0f);
        float q_m11 = q_x11 + 1.2582912e+07f;
        float q_x12 = fminf(fmaxf(qvals[ps_4 * 16 + 4 + 2] * qinv, -127.0f), 127.0f);
        float q_m12 = q_x12 + 1.2582912e+07f;
        float q_x13 = fminf(fmaxf(qvals[ps_4 * 16 + 4 + 3] * qinv, -127.0f), 127.0f);
        float q_m13 = q_x13 + 1.2582912e+07f;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_28) : "r"(__as_u32(q_m10)), "r"(__as_u32(q_m11)));
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_29) : "r"(__as_u32(q_m12)), "r"(__as_u32(q_m13)));
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_27) : "r"(_prmt_28), "r"(_prmt_29));
        qw[1] = _prmt_27;
        float q_x20 = fminf(fmaxf(qvals[ps_4 * 16 + 8] * qinv, -127.0f), 127.0f);
        float q_m20 = q_x20 + 1.2582912e+07f;
        float q_x21 = fminf(fmaxf(qvals[ps_4 * 16 + 8 + 1] * qinv, -127.0f), 127.0f);
        float q_m21 = q_x21 + 1.2582912e+07f;
        float q_x22 = fminf(fmaxf(qvals[ps_4 * 16 + 8 + 2] * qinv, -127.0f), 127.0f);
        float q_m22 = q_x22 + 1.2582912e+07f;
        float q_x23 = fminf(fmaxf(qvals[ps_4 * 16 + 8 + 3] * qinv, -127.0f), 127.0f);
        float q_m23 = q_x23 + 1.2582912e+07f;
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_31) : "r"(__as_u32(q_m20)), "r"(__as_u32(q_m21)));
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_32) : "r"(__as_u32(q_m22)), "r"(__as_u32(q_m23)));
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_30) : "r"(_prmt_31), "r"(_prmt_32));
        qw[2] = _prmt_30;
        float q_x30 = fminf(fmaxf(qvals[ps_4 * 16 + 12] * qinv, -127.0f), 127.0f);
        float q_m30 = q_x30 + 1.2582912e+07f;
        float q_x31 = fminf(fmaxf(qvals[ps_4 * 16 + 12 + 1] * qinv, -127.0f), 127.0f);
        float q_m31 = q_x31 + 1.2582912e+07f;
        float q_x32 = fminf(fmaxf(qvals[ps_4 * 16 + 12 + 2] * qinv, -127.0f), 127.0f);
        float q_m32 = q_x32 + 1.2582912e+07f;
        float q_x33 = fminf(fmaxf(qvals[ps_4 * 16 + 12 + 3] * qinv, -127.0f), 127.0f);
        float q_m33 = q_x33 + 1.2582912e+07f;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_34) : "r"(__as_u32(q_m30)), "r"(__as_u32(q_m31)));
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_35) : "r"(__as_u32(q_m32)), "r"(__as_u32(q_m33)));
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_33) : "r"(_prmt_34), "r"(_prmt_35));
        qw[3] = _prmt_33;
        reinterpret_cast<int4*>(q8 + ((bh * s_pad + tok_3) * 128 + col16 >> 2))[0] = reinterpret_cast<int4*>(qw)[0];
    }
    int cg = tid >> 5;
    int chunk = tid >> 3 & 3;
    int half = tid >> 2 & 1;
    int t4 = tid & 3;
    float vs[16];
    for (int quad_1 = 0; quad_1 < 4; quad_1++) {
        float _vec_load_3[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (bh * 128 + cg * 16 + quad_1 * 4) + 0);
            _vec_load_3[0 + 0] = _v4.x;
            _vec_load_3[0 + 1] = _v4.y;
            _vec_load_3[0 + 2] = _v4.z;
            _vec_load_3[0 + 3] = _v4.w;
        }
        for (int e_4 = 0; e_4 < 4; e_4++) {
            vs[quad_1 * 4 + e_4] = _vec_load_3[e_4];
        }
    }
    int v_base = batch * v_stride_b + head * v_stride_h + cg * 16;
    float vq[64];
    for (int i = 0; i < 4; i++) {
        int key_local = chunk * 32 + half * 16 + (i >> 1) * 8 + 2 * t4 + (i & 1);
        int vtok = tok_base + key_local;
        int vvalid = ((vtok < seqlen) ? 1 : 0);
        int vtok_c = ((vtok < seqlen) ? vtok : seqlen - 1);
        float _vec_load_4[16];
        {
            const uint4* _vptr_37 = reinterpret_cast<const uint4*>(V + (v_base + vtok_c * v_stride_s) + 0);
            uint4 _vld_37[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_37[_blk] = _vptr_37[_blk];
                uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                }
            }
        }
        for (int e_5 = 0; e_5 < 16; e_5++) {
            float vf = _vec_load_4[e_5];
            float _fdiv_38 = __fdiv_rn(vf, vs[e_5]);
            vq[i * 16 + e_5] = ((vvalid == 1) ? _fdiv_38 : 0.0f);
        }
    }
    int vt_word = (bh * 128 + cg * 16) * s_pad + tok_base + chunk * 32 + half * 16 + 4 * t4 >> 2;
    for (int e_6 = 0; e_6 < 16; e_6++) {
        float four[4];
        for (int i_1 = 0; i_1 < 4; i_1++) {
            four[i_1] = vq[i_1 * 16 + e_6];
        }
        unsigned int word[1];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(four[0]), "f"(four[1]),
                                   "f"(four[2]), "f"(four[3]));
            word[0] = _packed;
        }
        *(reinterpret_cast<unsigned int*>(vt8 + (vt_word + e_6 * (s_pad >> 2))) + (0)) = word[0];
    }
}

} // extern "C"
