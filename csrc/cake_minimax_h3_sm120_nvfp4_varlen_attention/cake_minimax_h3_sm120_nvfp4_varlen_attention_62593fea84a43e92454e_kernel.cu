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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_62593fea84a43e92454e(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tok_seg, int* __restrict__ cu_seqlens, int* __restrict__ seg_tile_begin, float* __restrict__ mean_k, float* __restrict__ q_mean, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ qm4, uint8_t* __restrict__ qm_sf, unsigned int* __restrict__ vt4, uint8_t* __restrict__ v_sf, int total_tokens, int total_padded, int num_heads, int num_kblocks)
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
    int num_tblk = total_padded / 32;
    int vwarp = tid >> 5;
    int vlane = tid & 31;
    int vblk = vwarp & 1;
    int vdp = (vwarp >> 1) * 16 + (vlane & 15);
    int vhalf = vlane >> 4;
    #pragma unroll 1
    for (int work = bid; work < num_tblk * num_heads; work += num_bids) {
        int tblk = work / num_heads;
        int head = work % num_heads;
        int tok_base = tblk * 32;
        int tok = tok_base + trow;
        int valid = ((tok < total_tokens) ? 1 : 0);
        int tok_c = ((tok < total_tokens) ? tok : total_tokens - 1);
        int base = (tok_c * num_heads + head) * 128 + col16;
        float kv[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + base + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&kv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        int seg = tok_seg[tok_c];
        int vlive[8];
        vlive[0] = ((tok_base + (vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float vl_0[2];
        {
            uint32_t _bf16x2_bits_1;
            _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) << 16);
            (&vl_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) & 0xffff0000u);
        }
        vlive[1] = ((tok_base + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float vl_1[2];
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) << 16);
            (&vl_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) & 0xffff0000u);
        }
        vlive[2] = ((tok_base + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float vl_2[2];
        {
            uint32_t _bf16x2_bits_3;
            _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (8 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) << 16);
            (&vl_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) & 0xffff0000u);
        }
        vlive[3] = ((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float vl_3[2];
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (8 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) << 16);
            (&vl_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) & 0xffff0000u);
        }
        vlive[4] = ((tok_base + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float vl_4[2];
        {
            uint32_t _bf16x2_bits_5;
            _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (16 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) << 16);
            (&vl_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) & 0xffff0000u);
        }
        vlive[5] = ((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float vl_5[2];
        {
            uint32_t _bf16x2_bits_6;
            _bf16x2_bits_6 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (16 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) << 16);
            (&vl_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) & 0xffff0000u);
        }
        vlive[6] = ((tok_base + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float vl_6[2];
        {
            uint32_t _bf16x2_bits_7;
            _bf16x2_bits_7 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? tok_base + (24 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) << 16);
            (&vl_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) & 0xffff0000u);
        }
        vlive[7] = ((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float vl_7[2];
        {
            uint32_t _bf16x2_bits_8;
            _bf16x2_bits_8 = *reinterpret_cast<const uint32_t*>(V + ((((tok_base + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tok_base + (24 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
            (&vl_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) << 16);
            (&vl_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) & 0xffff0000u);
        }
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        int mbase = (seg * num_heads + head) * 128 + col16;
        float kc[16];
        for (int quad = 0; quad < 4; quad++) {
            float _vec_load_11[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad * 4) + 0);
                _vec_load_11[0 + 0] = _v4.x;
                _vec_load_11[0 + 1] = _v4.y;
                _vec_load_11[0 + 2] = _v4.z;
                _vec_load_11[0 + 3] = _v4.w;
            }
            for (int e = 0; e < 4; e++) {
                float kf = kv[quad * 4 + e];
                kc[quad * 4 + e] = ((valid == 1) ? kf - _vec_load_11[e] : 0.0f);
            }
        }
        unsigned int kw[2];
        float ksf_a[1];
        float max_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(kc[0]), fabsf(kc[1])), fabsf(kc[2])), fabsf(kc[3])), fabsf(kc[4])), fabsf(kc[5])), fabsf(kc[6])), fabsf(kc[7])), fabsf(kc[8])), fabsf(kc[9])), fabsf(kc[10])), fabsf(kc[11])), fabsf(kc[12])), fabsf(kc[13])), fabsf(kc[14])), fabsf(kc[15]));
        uint16_t _cvt_satfinite_e4m3x2_f32_10;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(0.0f), "f"(max_1 * 0.16666667f));
        uint16_t _e4m3x2_decode_11 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_10 & 0xFFu);
        uint32_t _f16x2_decode_11;
        float fp8_decode;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_11) : "h"(_e4m3x2_decode_11));
        uint16_t _f16_decode_11 = (uint16_t)_f16x2_decode_11;
        asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode) : "h"(_f16_decode_11));
        float fdiv_rn = __fdiv_rn(1.0f, fp8_decode);
        kc[0] = kc[0] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[1] = kc[1] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[2] = kc[2] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[3] = kc[3] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[4] = kc[4] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[5] = kc[5] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[6] = kc[6] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[7] = kc[7] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[8] = kc[8] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[9] = kc[9] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[10] = kc[10] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[11] = kc[11] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[12] = kc[12] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[13] = kc[13] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[14] = kc[14] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        kc[15] = kc[15] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[0]) : "f"(kc[0]), "f"(kc[1]), "f"(kc[2]), "f"(kc[3]), "f"(kc[4]), "f"(kc[5]), "f"(kc[6]), "f"(kc[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[1]) : "f"(kc[8]), "f"(kc[9]), "f"(kc[10]), "f"(kc[11]), "f"(kc[12]), "f"(kc[13]), "f"(kc[14]), "f"(kc[15]));
        ksf_a[0] = max_1 * 0.16666667f;
        unsigned int kstore[4];
        unsigned int _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
        kstore[0] = _shfl_xor_sync_12;
        unsigned int _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
        kstore[1] = _shfl_xor_sync_13;
        kstore[2] = kw[0];
        kstore[3] = kw[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_14;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(0.0f), "f"(ksf_a[0]));
        *reinterpret_cast<unsigned char*>(k_sf + (((head * num_kblocks + (tok >> 7)) * 128 + (tok & 127)) * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_14 & 127));
        if (valid == 1) {
            if ((sub & 1) == 1) {
                reinterpret_cast<int4*>(k4 + (((tok * num_heads + head) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
            }
        }
        float va[8];
        float vb[8];
        va[0] = vl_0[0];
        vb[0] = vl_0[1];
        va[1] = vl_1[0];
        vb[1] = vl_1[1];
        va[2] = vl_2[0];
        vb[2] = vl_2[1];
        va[3] = vl_3[0];
        vb[3] = vl_3[1];
        va[4] = vl_4[0];
        vb[4] = vl_4[1];
        va[5] = vl_5[0];
        vb[5] = vl_5[1];
        va[6] = vl_6[0];
        vb[6] = vl_6[1];
        va[7] = vl_7[0];
        vb[7] = vl_7[1];
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
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, amax_a, 16);
        amax_a = fmaxf(amax_a, _shfl_xor_sync_15);
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, amax_b, 16);
        amax_b = fmaxf(amax_b, _shfl_xor_sync_16);
        float vsf_a = amax_a * 0.16666667f;
        float vsf_b = amax_b * 0.16666667f;
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vsf_b), "f"(vsf_a));
        uint16_t _e4m3x2_decode_17 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
        uint32_t _f16x2_decode_17;
        float vdec_a;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_17) : "h"(_e4m3x2_decode_17));
        uint16_t _f16_decode_17 = (uint16_t)_f16x2_decode_17;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a) : "h"(_f16_decode_17));
        uint16_t _e4m3x2_decode_18 = (uint16_t)((unsigned int)_e4m3x2_f32_0 >> 8 & 0xFFu);
        uint32_t _f16x2_decode_18;
        float vdec_b;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_18) : "h"(_e4m3x2_decode_18));
        uint16_t _f16_decode_18 = (uint16_t)_f16x2_decode_18;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b) : "h"(_f16_decode_18));
        float _fdiv_19 = __fdiv_rn(1.0f, vdec_a);
        float vinv_a = ((vdec_a != 0.0f) ? _fdiv_19 : 0.0f);
        float _fdiv_20 = __fdiv_rn(1.0f, vdec_b);
        float vinv_b = ((vdec_b != 0.0f) ? _fdiv_20 : 0.0f);
        for (int ii_1 = 0; ii_1 < 8; ii_1++) {
            va[ii_1] = va[ii_1] * vinv_a;
            vb[ii_1] = vb[ii_1] * vinv_b;
        }
        unsigned int vwa[1];
        unsigned int vwb[1];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa[0]) : "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]), "f"(va[4]), "f"(va[5]), "f"(va[6]), "f"(va[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb[0]) : "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]), "f"(vb[4]), "f"(vb[5]), "f"(vb[6]), "f"(vb[7]));
        int row_words = total_padded >> 3;
        int vt_word = (head * 128 + vdp * 2) * row_words + (tok_base >> 3) + vblk * 2 + vhalf;
        *(reinterpret_cast<unsigned int*>(vt4 + vt_word) + (0)) = vwa[0];
        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word + row_words)) + (0)) = vwb[0];
        if (vhalf == 0) {
            int vsf_base = ((head * num_kblocks + (tok_base >> 7)) * 128 + vdp * 2) * 8 + (((tok_base & 127) >> 4) + vblk);
            uint16_t _cvt_satfinite_e4m3x2_f32_21;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(0.0f), "f"(vsf_a));
            *reinterpret_cast<unsigned char*>(v_sf + vsf_base) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_21 & 127));
            uint16_t _cvt_satfinite_e4m3x2_f32_22;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_22) : "f"(0.0f), "f"(vsf_b));
            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_22 & 127));
        }
    }
}

} // extern "C"
