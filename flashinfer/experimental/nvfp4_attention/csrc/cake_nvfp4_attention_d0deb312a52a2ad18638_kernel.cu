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
kernel_cake_nvfp4_attention_d0deb312a52a2ad18638(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, float* __restrict__ mean_k, float* __restrict__ q_mean, float* __restrict__ q_amax, float* __restrict__ qm_amax, float* __restrict__ k_amax, unsigned int* __restrict__ q8, unsigned int* __restrict__ k8, unsigned int* __restrict__ qm8, unsigned int* __restrict__ vt4, uint8_t* __restrict__ v_sf, int seq_len, int seq_pad, int num_heads, int num_qblocks, int num_kblocks, int num_items, long long q_stride_b, long long q_stride_h, long long q_stride_s, long long k_stride_b, long long k_stride_h, long long k_stride_s, long long v_stride_b, long long v_stride_h, long long v_stride_s)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    int vwarp = tid >> 5;
    int vlane = tid & 31;
    int vblk = vwarp & 1;
    int vdp = (vwarp >> 1) * 16 + (vlane & 15);
    int vhalf = vlane >> 4;
    unsigned long long qsb = (unsigned long long)q_stride_b;
    unsigned long long qsh = (unsigned long long)q_stride_h;
    unsigned long long qss = (unsigned long long)q_stride_s;
    unsigned long long ksb = (unsigned long long)k_stride_b;
    unsigned long long ksh = (unsigned long long)k_stride_h;
    unsigned long long kss = (unsigned long long)k_stride_s;
    unsigned long long vsb = (unsigned long long)v_stride_b;
    unsigned long long vsh = (unsigned long long)v_stride_h;
    unsigned long long vss = (unsigned long long)v_stride_s;
    #pragma unroll 1
    for (int work = bid; work < num_items; work += num_bids) {
        int gblk = work / num_heads;
        int head = work % num_heads;
        int nblk = seq_pad / 32;
        int batch = gblk / nblk;
        int lblk = gblk % nblk;
        int tok_base = lblk * 32;
        int tok = tok_base + trow;
        int ptok = gblk * 32 + trow;
        int valid = ((tok < seq_len) ? 1 : 0);
        int tok_c = ((tok < seq_len) ? tok : seq_len - 1);
        unsigned long long b64 = (unsigned long long)batch;
        unsigned long long h64 = (unsigned long long)head;
        unsigned long long t64 = (unsigned long long)tok_c;
        unsigned long long c64 = (unsigned long long)col16;
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + (b64 * qsb + h64 * qsh + t64 * qss + c64) + 0);
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
        float _vec_load_1[16];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + (b64 * ksb + h64 * ksh + t64 * kss + c64) + 0);
            uint4 _vld_1[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        int vlive[8];
        unsigned long long vbase = b64 * vsb + h64 * vsh + (unsigned long long)(vdp * 2);
        vlive[0] = ((tok_base + (vblk * 4 + vhalf * 2) < seq_len) ? 1 : 0);
        float _vec_load_2[2];
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (vblk * 4 + vhalf * 2) < seq_len) ? tok_base + (vblk * 4 + vhalf * 2) : seq_len - 1)) * vss) + 0);
            (&_vec_load_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) << 16);
            (&_vec_load_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) & 0xffff0000u);
        }
        vlive[1] = ((tok_base + (vblk * 4 + vhalf * 2 + 1) < seq_len) ? 1 : 0);
        float _vec_load_3[2];
        {
            uint32_t _bf16x2_bits_3;
            _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (vblk * 4 + vhalf * 2 + 1) < seq_len) ? tok_base + (vblk * 4 + vhalf * 2 + 1) : seq_len - 1)) * vss) + 0);
            (&_vec_load_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) << 16);
            (&_vec_load_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) & 0xffff0000u);
        }
        vlive[2] = ((tok_base + (8 + vblk * 4 + vhalf * 2) < seq_len) ? 1 : 0);
        float _vec_load_4[2];
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (8 + vblk * 4 + vhalf * 2) < seq_len) ? tok_base + (8 + vblk * 4 + vhalf * 2) : seq_len - 1)) * vss) + 0);
            (&_vec_load_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) << 16);
            (&_vec_load_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) & 0xffff0000u);
        }
        vlive[3] = ((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? 1 : 0);
        float _vec_load_5[2];
        {
            uint32_t _bf16x2_bits_5;
            _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? tok_base + (8 + vblk * 4 + vhalf * 2 + 1) : seq_len - 1)) * vss) + 0);
            (&_vec_load_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) << 16);
            (&_vec_load_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) & 0xffff0000u);
        }
        vlive[4] = ((tok_base + (16 + vblk * 4 + vhalf * 2) < seq_len) ? 1 : 0);
        float _vec_load_6[2];
        {
            uint32_t _bf16x2_bits_6;
            _bf16x2_bits_6 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (16 + vblk * 4 + vhalf * 2) < seq_len) ? tok_base + (16 + vblk * 4 + vhalf * 2) : seq_len - 1)) * vss) + 0);
            (&_vec_load_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) << 16);
            (&_vec_load_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) & 0xffff0000u);
        }
        vlive[5] = ((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? 1 : 0);
        float _vec_load_7[2];
        {
            uint32_t _bf16x2_bits_7;
            _bf16x2_bits_7 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? tok_base + (16 + vblk * 4 + vhalf * 2 + 1) : seq_len - 1)) * vss) + 0);
            (&_vec_load_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) << 16);
            (&_vec_load_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) & 0xffff0000u);
        }
        vlive[6] = ((tok_base + (24 + vblk * 4 + vhalf * 2) < seq_len) ? 1 : 0);
        float _vec_load_8[2];
        {
            uint32_t _bf16x2_bits_8;
            _bf16x2_bits_8 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (24 + vblk * 4 + vhalf * 2) < seq_len) ? tok_base + (24 + vblk * 4 + vhalf * 2) : seq_len - 1)) * vss) + 0);
            (&_vec_load_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) << 16);
            (&_vec_load_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) & 0xffff0000u);
        }
        vlive[7] = ((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? 1 : 0);
        float _vec_load_9[2];
        {
            uint32_t _bf16x2_bits_9;
            _bf16x2_bits_9 = *reinterpret_cast<const uint32_t*>(V + (vbase + (unsigned long long)(((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < seq_len) ? tok_base + (24 + vblk * 4 + vhalf * 2 + 1) : seq_len - 1)) * vss) + 0);
            (&_vec_load_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_9) << 16);
            (&_vec_load_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_9) & 0xffff0000u);
        }
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        int qt = batch * num_qblocks + (tok >> 7);
        int is_first = ((valid == 1 && (tok & 127) == 0) ? 1 : 0);
        int row_words = (ptok * num_heads + head) * 32 + sub * 4;
        float qa = q_amax[qt * num_heads + head];
        float qma = qm_amax[qt * num_heads + head];
        float inv_q = __fdiv_rn(127.0f, fmaxf(qa, (qa + qma) * 0.5f));
        int qmbase = (qt * num_heads + head) * 128 + col16;
        float o8v[16];
        float qc[16];
        for (int quad = 0; quad < 4; quad++) {
            float _vec_load_10[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(q_mean + (qmbase + quad * 4) + 0);
                _vec_load_10[0 + 0] = _v4.x;
                _vec_load_10[0 + 1] = _v4.y;
                _vec_load_10[0 + 2] = _v4.z;
                _vec_load_10[0 + 3] = _v4.w;
            }
            for (int e = 0; e < 4; e++) {
                float om = fminf(fmaxf(_vec_load_10[e] * inv_q, -127.0f), 127.0f);
                o8v[quad * 4 + e] = om + 1.2582912e+07f - 1.2582912e+07f;
                float qf = _vec_load_0[quad * 4 + e];
                qc[quad * 4 + e] = ((valid == 1) ? __fmaf_rn(qf, inv_q, -o8v[quad * 4 + e]) : 0.0f);
            }
        }
        unsigned int qw[4];
        float _min_0 = fminf(fmaxf(qc[0], -127.0f), 127.0f);
        float _min_1 = fminf(fmaxf(qc[1], -127.0f), 127.0f);
        float _min_2 = fminf(fmaxf(qc[2], -127.0f), 127.0f);
        float _min_3 = fminf(fmaxf(qc[3], -127.0f), 127.0f);
        float _min_4 = fminf(fmaxf(qc[4], -127.0f), 127.0f);
        float _min_5 = fminf(fmaxf(qc[5], -127.0f), 127.0f);
        float _min_6 = fminf(fmaxf(qc[6], -127.0f), 127.0f);
        float _min_7 = fminf(fmaxf(qc[7], -127.0f), 127.0f);
        float _min_8 = fminf(fmaxf(qc[8], -127.0f), 127.0f);
        float _min_9 = fminf(fmaxf(qc[9], -127.0f), 127.0f);
        float _min_10 = fminf(fmaxf(qc[10], -127.0f), 127.0f);
        float _min_11 = fminf(fmaxf(qc[11], -127.0f), 127.0f);
        float _min_12 = fminf(fmaxf(qc[12], -127.0f), 127.0f);
        float _min_13 = fminf(fmaxf(qc[13], -127.0f), 127.0f);
        float _min_14 = fminf(fmaxf(qc[14], -127.0f), 127.0f);
        float _min_15 = fminf(fmaxf(qc[15], -127.0f), 127.0f);
        uint32_t _prmt_12;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_12) : "r"(__as_u32(_min_0 + 1.2582912e+07f)), "r"(__as_u32(_min_1 + 1.2582912e+07f)));
        uint32_t _prmt_13;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_13) : "r"(__as_u32(_min_2 + 1.2582912e+07f)), "r"(__as_u32(_min_3 + 1.2582912e+07f)));
        uint32_t _prmt_11;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_11) : "r"(_prmt_12), "r"(_prmt_13));
        qw[0] = _prmt_11;
        uint32_t _prmt_15;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_15) : "r"(__as_u32(_min_4 + 1.2582912e+07f)), "r"(__as_u32(_min_5 + 1.2582912e+07f)));
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_16) : "r"(__as_u32(_min_6 + 1.2582912e+07f)), "r"(__as_u32(_min_7 + 1.2582912e+07f)));
        uint32_t _prmt_14;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_14) : "r"(_prmt_15), "r"(_prmt_16));
        qw[1] = _prmt_14;
        uint32_t _prmt_18;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_18) : "r"(__as_u32(_min_8 + 1.2582912e+07f)), "r"(__as_u32(_min_9 + 1.2582912e+07f)));
        uint32_t _prmt_19;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_19) : "r"(__as_u32(_min_10 + 1.2582912e+07f)), "r"(__as_u32(_min_11 + 1.2582912e+07f)));
        uint32_t _prmt_17;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_17) : "r"(_prmt_18), "r"(_prmt_19));
        qw[2] = _prmt_17;
        uint32_t _prmt_21;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_21) : "r"(__as_u32(_min_12 + 1.2582912e+07f)), "r"(__as_u32(_min_13 + 1.2582912e+07f)));
        uint32_t _prmt_22;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_22) : "r"(__as_u32(_min_14 + 1.2582912e+07f)), "r"(__as_u32(_min_15 + 1.2582912e+07f)));
        uint32_t _prmt_20;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_20) : "r"(_prmt_21), "r"(_prmt_22));
        qw[3] = _prmt_20;
        reinterpret_cast<int4*>(q8 + row_words)[0] = reinterpret_cast<int4*>(qw)[0];
        if (is_first == 1) {
            unsigned int mw[4];
            float _min_16 = fminf(fmaxf(o8v[0], -127.0f), 127.0f);
            float _min_17 = fminf(fmaxf(o8v[1], -127.0f), 127.0f);
            float _min_18 = fminf(fmaxf(o8v[2], -127.0f), 127.0f);
            float _min_19 = fminf(fmaxf(o8v[3], -127.0f), 127.0f);
            float _min_20 = fminf(fmaxf(o8v[4], -127.0f), 127.0f);
            float _min_21 = fminf(fmaxf(o8v[5], -127.0f), 127.0f);
            float _min_22 = fminf(fmaxf(o8v[6], -127.0f), 127.0f);
            float _min_23 = fminf(fmaxf(o8v[7], -127.0f), 127.0f);
            float _min_24 = fminf(fmaxf(o8v[8], -127.0f), 127.0f);
            float _min_25 = fminf(fmaxf(o8v[9], -127.0f), 127.0f);
            float _min_26 = fminf(fmaxf(o8v[10], -127.0f), 127.0f);
            float _min_27 = fminf(fmaxf(o8v[11], -127.0f), 127.0f);
            float _min_28 = fminf(fmaxf(o8v[12], -127.0f), 127.0f);
            float _min_29 = fminf(fmaxf(o8v[13], -127.0f), 127.0f);
            float _min_30 = fminf(fmaxf(o8v[14], -127.0f), 127.0f);
            float _min_31 = fminf(fmaxf(o8v[15], -127.0f), 127.0f);
            uint32_t _prmt_24;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_24) : "r"(__as_u32(_min_16 + 1.2582912e+07f)), "r"(__as_u32(_min_17 + 1.2582912e+07f)));
            uint32_t _prmt_25;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_25) : "r"(__as_u32(_min_18 + 1.2582912e+07f)), "r"(__as_u32(_min_19 + 1.2582912e+07f)));
            uint32_t _prmt_23;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_23) : "r"(_prmt_24), "r"(_prmt_25));
            mw[0] = _prmt_23;
            uint32_t _prmt_27;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_27) : "r"(__as_u32(_min_20 + 1.2582912e+07f)), "r"(__as_u32(_min_21 + 1.2582912e+07f)));
            uint32_t _prmt_28;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_28) : "r"(__as_u32(_min_22 + 1.2582912e+07f)), "r"(__as_u32(_min_23 + 1.2582912e+07f)));
            uint32_t _prmt_26;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_26) : "r"(_prmt_27), "r"(_prmt_28));
            mw[1] = _prmt_26;
            uint32_t _prmt_30;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_30) : "r"(__as_u32(_min_24 + 1.2582912e+07f)), "r"(__as_u32(_min_25 + 1.2582912e+07f)));
            uint32_t _prmt_31;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_31) : "r"(__as_u32(_min_26 + 1.2582912e+07f)), "r"(__as_u32(_min_27 + 1.2582912e+07f)));
            uint32_t _prmt_29;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_29) : "r"(_prmt_30), "r"(_prmt_31));
            mw[2] = _prmt_29;
            uint32_t _prmt_33;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_33) : "r"(__as_u32(_min_28 + 1.2582912e+07f)), "r"(__as_u32(_min_29 + 1.2582912e+07f)));
            uint32_t _prmt_34;
            asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_34) : "r"(__as_u32(_min_30 + 1.2582912e+07f)), "r"(__as_u32(_min_31 + 1.2582912e+07f)));
            uint32_t _prmt_32;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_32) : "r"(_prmt_33), "r"(_prmt_34));
            mw[3] = _prmt_32;
            int qmrow = qt * num_heads + head;
            reinterpret_cast<int4*>(qm8 + (qmrow * 32 + sub * 4))[0] = reinterpret_cast<int4*>(mw)[0];
        }
        int mbase = (batch * num_heads + head) * 128 + col16;
        float kc[16];
        for (int quad_1 = 0; quad_1 < 4; quad_1++) {
            float _vec_load_11[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad_1 * 4) + 0);
                _vec_load_11[0 + 0] = _v4.x;
                _vec_load_11[0 + 1] = _v4.y;
                _vec_load_11[0 + 2] = _v4.z;
                _vec_load_11[0 + 3] = _v4.w;
            }
            for (int e_1 = 0; e_1 < 4; e_1++) {
                float kf = _vec_load_1[quad_1 * 4 + e_1];
                kc[quad_1 * 4 + e_1] = ((valid == 1) ? kf - _vec_load_11[e_1] : 0.0f);
            }
        }
        float inv_k = __fdiv_rn(127.0f, k_amax[head * num_kblocks + (ptok >> 7)]);
        unsigned int kw[4];
        float _min_32 = fminf(fmaxf(kc[0] * inv_k, -127.0f), 127.0f);
        float _min_33 = fminf(fmaxf(kc[1] * inv_k, -127.0f), 127.0f);
        float _min_34 = fminf(fmaxf(kc[2] * inv_k, -127.0f), 127.0f);
        float _min_35 = fminf(fmaxf(kc[3] * inv_k, -127.0f), 127.0f);
        float _min_36 = fminf(fmaxf(kc[4] * inv_k, -127.0f), 127.0f);
        float _min_37 = fminf(fmaxf(kc[5] * inv_k, -127.0f), 127.0f);
        float _min_38 = fminf(fmaxf(kc[6] * inv_k, -127.0f), 127.0f);
        float _min_39 = fminf(fmaxf(kc[7] * inv_k, -127.0f), 127.0f);
        float _min_40 = fminf(fmaxf(kc[8] * inv_k, -127.0f), 127.0f);
        float _min_41 = fminf(fmaxf(kc[9] * inv_k, -127.0f), 127.0f);
        float _min_42 = fminf(fmaxf(kc[10] * inv_k, -127.0f), 127.0f);
        float _min_43 = fminf(fmaxf(kc[11] * inv_k, -127.0f), 127.0f);
        float _min_44 = fminf(fmaxf(kc[12] * inv_k, -127.0f), 127.0f);
        float _min_45 = fminf(fmaxf(kc[13] * inv_k, -127.0f), 127.0f);
        float _min_46 = fminf(fmaxf(kc[14] * inv_k, -127.0f), 127.0f);
        float _min_47 = fminf(fmaxf(kc[15] * inv_k, -127.0f), 127.0f);
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_37) : "r"(__as_u32(_min_32 + 1.2582912e+07f)), "r"(__as_u32(_min_33 + 1.2582912e+07f)));
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_38) : "r"(__as_u32(_min_34 + 1.2582912e+07f)), "r"(__as_u32(_min_35 + 1.2582912e+07f)));
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_36) : "r"(_prmt_37), "r"(_prmt_38));
        kw[0] = _prmt_36;
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_40) : "r"(__as_u32(_min_36 + 1.2582912e+07f)), "r"(__as_u32(_min_37 + 1.2582912e+07f)));
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_41) : "r"(__as_u32(_min_38 + 1.2582912e+07f)), "r"(__as_u32(_min_39 + 1.2582912e+07f)));
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_39) : "r"(_prmt_40), "r"(_prmt_41));
        kw[1] = _prmt_39;
        uint32_t _prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_43) : "r"(__as_u32(_min_40 + 1.2582912e+07f)), "r"(__as_u32(_min_41 + 1.2582912e+07f)));
        uint32_t _prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_44) : "r"(__as_u32(_min_42 + 1.2582912e+07f)), "r"(__as_u32(_min_43 + 1.2582912e+07f)));
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_42) : "r"(_prmt_43), "r"(_prmt_44));
        kw[2] = _prmt_42;
        uint32_t _prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_46) : "r"(__as_u32(_min_44 + 1.2582912e+07f)), "r"(__as_u32(_min_45 + 1.2582912e+07f)));
        uint32_t _prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_47) : "r"(__as_u32(_min_46 + 1.2582912e+07f)), "r"(__as_u32(_min_47 + 1.2582912e+07f)));
        uint32_t _prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_45) : "r"(_prmt_46), "r"(_prmt_47));
        kw[3] = _prmt_45;
        reinterpret_cast<int4*>(k8 + row_words)[0] = reinterpret_cast<int4*>(kw)[0];
        float va[8];
        float vb[8];
        va[0] = _vec_load_2[0];
        vb[0] = _vec_load_2[1];
        va[1] = _vec_load_3[0];
        vb[1] = _vec_load_3[1];
        va[2] = _vec_load_4[0];
        vb[2] = _vec_load_4[1];
        va[3] = _vec_load_5[0];
        vb[3] = _vec_load_5[1];
        va[4] = _vec_load_6[0];
        vb[4] = _vec_load_6[1];
        va[5] = _vec_load_7[0];
        vb[5] = _vec_load_7[1];
        va[6] = _vec_load_8[0];
        vb[6] = _vec_load_8[1];
        va[7] = _vec_load_9[0];
        vb[7] = _vec_load_9[1];
        va[0] = ((vlive[0] == 1) ? va[0] : 0.0f);
        vb[0] = ((vlive[0] == 1) ? vb[0] : 0.0f);
        va[1] = ((vlive[1] == 1) ? va[1] : 0.0f);
        vb[1] = ((vlive[1] == 1) ? vb[1] : 0.0f);
        va[2] = ((vlive[2] == 1) ? va[2] : 0.0f);
        vb[2] = ((vlive[2] == 1) ? vb[2] : 0.0f);
        va[3] = ((vlive[3] == 1) ? va[3] : 0.0f);
        vb[3] = ((vlive[3] == 1) ? vb[3] : 0.0f);
        va[4] = ((vlive[4] == 1) ? va[4] : 0.0f);
        vb[4] = ((vlive[4] == 1) ? vb[4] : 0.0f);
        va[5] = ((vlive[5] == 1) ? va[5] : 0.0f);
        vb[5] = ((vlive[5] == 1) ? vb[5] : 0.0f);
        va[6] = ((vlive[6] == 1) ? va[6] : 0.0f);
        vb[6] = ((vlive[6] == 1) ? vb[6] : 0.0f);
        va[7] = ((vlive[7] == 1) ? va[7] : 0.0f);
        vb[7] = ((vlive[7] == 1) ? vb[7] : 0.0f);
        float amax_a = 0.0f;
        float amax_b = 0.0f;
        for (int ii = 0; ii < 8; ii++) {
            amax_a = fmaxf(amax_a, fabsf(va[ii]));
            amax_b = fmaxf(amax_b, fabsf(vb[ii]));
        }
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, amax_a, 16);
        amax_a = fmaxf(amax_a, _shfl_xor_sync_48);
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, amax_b, 16);
        amax_b = fmaxf(amax_b, _shfl_xor_sync_49);
        float vsf_a = fmaxf(amax_a * 0.16666667f, 0.001953125f);
        float vsf_b = fmaxf(amax_b * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vsf_b), "f"(vsf_a));
        uint16_t _e4m3x2_decode_50 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
        uint32_t _f16x2_decode_50;
        float vdec_a;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_50) : "h"(_e4m3x2_decode_50));
        uint16_t _f16_decode_50 = (uint16_t)_f16x2_decode_50;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a) : "h"(_f16_decode_50));
        uint16_t _e4m3x2_decode_51 = (uint16_t)((unsigned int)_e4m3x2_f32_0 >> 8 & 0xFFu);
        uint32_t _f16x2_decode_51;
        float vdec_b;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_51) : "h"(_e4m3x2_decode_51));
        uint16_t _f16_decode_51 = (uint16_t)_f16x2_decode_51;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b) : "h"(_f16_decode_51));
        float vinv_a = __fdiv_rn(1.0f, vdec_a);
        float vinv_b = __fdiv_rn(1.0f, vdec_b);
        for (int ii_1 = 0; ii_1 < 8; ii_1++) {
            va[ii_1] = va[ii_1] * vinv_a;
            vb[ii_1] = vb[ii_1] * vinv_b;
        }
        unsigned int vwa[1];
        unsigned int vwb[1];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa[0]) : "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]), "f"(va[4]), "f"(va[5]), "f"(va[6]), "f"(va[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb[0]) : "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]), "f"(vb[4]), "f"(vb[5]), "f"(vb[6]), "f"(vb[7]));
        int ptok_base = gblk * 32;
        int vrow_words = num_kblocks * 128 >> 3;
        int vt_word = (head * 128 + vdp * 2) * vrow_words + (ptok_base >> 3) + vblk * 2 + vhalf;
        *(reinterpret_cast<unsigned int*>(vt4 + vt_word) + (0)) = vwa[0];
        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word + vrow_words)) + (0)) = vwb[0];
        if (vhalf == 0) {
            int vsf_base = ((head * num_kblocks + (ptok_base >> 7)) * 128 + vdp * 2) * 8 + (((ptok_base & 127) >> 4) + vblk);
            uint16_t _cvt_satfinite_e4m3x2_f32_52;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(0.0f), "f"(vsf_a));
            *reinterpret_cast<unsigned char*>(v_sf + vsf_base) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_52 & 127));
            uint16_t _cvt_satfinite_e4m3x2_f32_53;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(0.0f), "f"(vsf_b));
            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_53 & 127));
        }
    }
}

} // extern "C"
