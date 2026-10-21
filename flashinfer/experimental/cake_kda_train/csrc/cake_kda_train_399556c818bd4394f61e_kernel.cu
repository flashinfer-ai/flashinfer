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
#define SMEM_GRAW_OFF 0
#define SMEM_GRAW_STAGE_BYTES 32768
#define SMEM_GRAW_STRIDE 32768
#define SMEM_GTOT_OFF 32768
#define SMEM_GTOT_STAGE_BYTES 4096
#define SMEM_GTOT_STRIDE 4096
#define SMEM_TOTAL 36864
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_399556c818bd4394f61e(float* __restrict__ g, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ aqk, __nv_bfloat16* __restrict__ akk, __nv_bfloat16* __restrict__ do_, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ ke_out, __nv_bfloat16* __restrict__ qe_out, __nv_bfloat16* __restrict__ aqk_tril, __nv_bfloat16* __restrict__ akk_int, __nv_bfloat16* __restrict__ do_int, __nv_bfloat16* __restrict__ v_int, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group, int copy_saved)
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
    float* graw = reinterpret_cast<float*>(smem_raw + 0);
    const int graw_addr = smem + 0;
    float* gtot = reinterpret_cast<float*>(smem_raw + 32768);
    const int gtot_addr = smem + 32768;

    // === Task calls (dependency order) ===
    int bx = blockIdx.x;
    int chunk = bx / 2;
    int half = bx - chunk * 2;
    int head = blockIdx.y;
    int qk_head = head / group;
    int warp_0 = warp;
    int lane_1 = lane;
    int d0 = lane_1 * 4;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    long long int0 = (long long)chunk * 64;
    int clast = clen - 1;
    if (clast < 0) {
        clast = 0;
    }
    int rs = num_heads * 128;
    int qs = num_qk_heads * 128;
    long long ibase = (int0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long ebase = (ext0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long qbase = (ext0 * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d0;
    long long bbase = ext0 * (long long)num_heads + (long long)head;
    float x[32];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        int t_u = warp_0 * 8 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long gidx = ebase + (long long)(t_c * rs);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(g + gidx);
            x[u * 4 + 0] = _v4.x;
            x[u * 4 + 1] = _v4.y;
            x[u * 4 + 2] = _v4.z;
            x[u * 4 + 3] = _v4.w;
        }
    }
    float run[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        run[e] = 0.0f;
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        int t_u_1 = warp_0 * 8 + u_1;
        float gq[4];
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            float gate_u = 0.0f;
            if (t_u_1 < clen) {
                gate_u = x[u_1 * 4 + e_1];
            }
            gq[e_1] = gate_u;
            run[e_1] = run[e_1] + gate_u;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u_1 * 128 + d0) * 4)), "f"(gq[0]), "f"(gq[1]), "f"(gq[2]), "f"(gq[3]) : "memory");
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((warp_0 * 128 + d0) * 4)), "f"(run[0]), "f"(run[1]), "f"(run[2]), "f"(run[3]) : "memory");
    __syncthreads();
    int g2 = half * 4 + warp_0 / 2;
    int sub = warp_0 - warp_0 / 2 * 2;
    float off[4];
    float total[4];
    float gk_n[4];
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        off[e_2] = 0.0f;
        total[e_2] = 0.0f;
    }
    #pragma unroll
    for (int gi = 0; gi < 8; gi++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(gtot_addr + (unsigned int)((gi * 128 + d0) * 4)));
        if (g2 > gi) {
            #pragma unroll
            for (int e_3 = 0; e_3 < 4; e_3++) {
                off[e_3] = off[e_3] + tg[e_3];
            }
        }
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            total[e_4] = total[e_4] + tg[e_4];
        }
    }
    #pragma unroll
    for (int e_5 = 0; e_5 < 4; e_5++) {
        gk_n[e_5] = total[e_5] * 1.4426950408889634f;
    }
    float acc[4];
    #pragma unroll
    for (int e_6 = 0; e_6 < 4; e_6++) {
        acc[e_6] = 0.0f;
    }
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        if (sub > 0) {
            float gq0[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq0[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((g2 * 8 + p) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                acc[e_7] = acc[e_7] + gq0[e_7];
            }
        }
    }
    #pragma unroll
    for (int u_2 = 0; u_2 < 4; u_2++) {
        int t_u_2 = g2 * 8 + sub * 4 + u_2;
        int t_c_1 = t_u_2;
        if (t_u_2 > clast) {
            t_c_1 = clast;
        }
        long long iidx = ibase + (long long)(t_u_2 * rs);
        long long eidx = ebase + (long long)(t_c_1 * rs);
        long long qidx = qbase + (long long)(t_c_1 * qs);
        float bt_l = (float)beta[bbase + (long long)(t_c_1 * num_heads)];
        float kv[4];
        float qv[4];
        float vv[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k + qidx);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&kv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&kv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q + qidx);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&qv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&qv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(v + eidx);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&vv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&vv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        float gq2[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&gq2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2[(0) + 3]))
            : "r"(graw_addr + (unsigned int)((t_u_2 * 128 + d0) * 4)));
        float bt = 0.0f;
        if (t_u_2 < clen) {
            bt = bt_l;
        }
        float gkt[4];
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            acc[e_8] = acc[e_8] + gq2[e_8];
            gkt[e_8] = (off[e_8] + acc[e_8]) * 1.4426950408889634f;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
        }
        float eg[4];
        #pragma unroll
        for (int e_9 = 0; e_9 < 4; e_9++) {
            float _exp2_0 = approx_exp2(gkt[e_9]);
            eg[e_9] = _exp2_0;
        }
        float vsel[4];
        float vb[4];
        #pragma unroll
        for (int e_10 = 0; e_10 < 4; e_10++) {
            float v_e = 0.0f;
            if (t_u_2 < clen) {
                v_e = vv[e_10];
            }
            vsel[e_10] = v_e;
            vb[e_10] = v_e * bt;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(vb[0 + 0], vb[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(vb[0 + 2], vb[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(vb_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(vsel[0 + 0], vsel[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(vsel[0 + 2], vsel[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(v_int + iidx))[0]) = _pk2;
        }
        float kb[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            float k_e = 0.0f;
            if (t_u_2 < clen) {
                k_e = kv[e_11];
            }
            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)(k_e * bt);
            float _cvt_f32_0 = (float)_cvt_bf16_0;
            kb[e_11] = _cvt_f32_0 * eg[e_11];
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kb[0 + 0], kb[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kb[0 + 2], kb[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kb_out + iidx))[0]) = _pk2;
        }
        float kg[4];
        #pragma unroll
        for (int e_12 = 0; e_12 < 4; e_12++) {
            float k_e_1 = 0.0f;
            if (t_u_2 < clen) {
                k_e_1 = kv[e_12];
            }
            float _exp2_1 = approx_exp2(gk_n[e_12] - gkt[e_12]);
            kg[e_12] = k_e_1 * _exp2_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kg[0 + 0], kg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kg[0 + 2], kg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kg_out + iidx))[0]) = _pk2;
        }
        float ke[4];
        #pragma unroll
        for (int e_13 = 0; e_13 < 4; e_13++) {
            float k_e_2 = 0.0f;
            if (t_u_2 < clen) {
                k_e_2 = kv[e_13];
            }
            ke[e_13] = k_e_2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(ke[0 + 0], ke[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ke[0 + 2], ke[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(ke_out + iidx))[0]) = _pk2;
        }
        float qg[4];
        #pragma unroll
        for (int e_14 = 0; e_14 < 4; e_14++) {
            float q_e = 0.0f;
            if (t_u_2 < clen) {
                q_e = qv[e_14];
            }
            qg[e_14] = q_e * eg[e_14];
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(qg[0 + 0], qg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qg[0 + 2], qg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qg_out + iidx))[0]) = _pk2;
        }
        float qe[4];
        #pragma unroll
        for (int e_15 = 0; e_15 < 4; e_15++) {
            float q_e_1 = 0.0f;
            if (t_u_2 < clen) {
                q_e_1 = qv[e_15];
            }
            qe[e_15] = q_e_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(qe[0 + 0], qe[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qe[0 + 2], qe[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qe_out + iidx))[0]) = _pk2;
        }
        if (lane_1 == 0) {
            beta_int[(int0 + (long long)t_u_2) * (long long)num_heads + (long long)head] = bt;
        }
    }
    unsigned int dwa[8];
    int t_u_3 = g2 * 8 + sub * 4;
    int t_c_2 = t_u_3;
    if (t_u_3 > clast) {
        t_c_2 = clast;
    }
    long long eidx_d = ebase + (long long)(t_c_2 * rs);
    {
        uint2 _vld_4;
        _vld_4 = *reinterpret_cast<const uint2*>(do_ + eidx_d);
        uint2* _vdst_4 = reinterpret_cast<uint2*>(&dwa[0]);
        *_vdst_4 = _vld_4;
    }
    int t_u_2_1 = g2 * 8 + sub * 4 + 1;
    int t_c_3 = t_u_2_1;
    if (t_u_2_1 > clast) {
        t_c_3 = clast;
    }
    long long eidx_d_4 = ebase + (long long)(t_c_3 * rs);
    {
        uint2 _vld_5;
        _vld_5 = *reinterpret_cast<const uint2*>(do_ + eidx_d_4);
        uint2* _vdst_5 = reinterpret_cast<uint2*>(&dwa[2]);
        *_vdst_5 = _vld_5;
    }
    int t_u_5 = g2 * 8 + sub * 4 + 2;
    int t_c_6 = t_u_5;
    if (t_u_5 > clast) {
        t_c_6 = clast;
    }
    long long eidx_d_7 = ebase + (long long)(t_c_6 * rs);
    {
        uint2 _vld_6;
        _vld_6 = *reinterpret_cast<const uint2*>(do_ + eidx_d_7);
        uint2* _vdst_6 = reinterpret_cast<uint2*>(&dwa[4]);
        *_vdst_6 = _vld_6;
    }
    int t_u_8 = g2 * 8 + sub * 4 + 3;
    int t_c_9 = t_u_8;
    if (t_u_8 > clast) {
        t_c_9 = clast;
    }
    long long eidx_d_10 = ebase + (long long)(t_c_9 * rs);
    {
        uint2 _vld_7;
        _vld_7 = *reinterpret_cast<const uint2*>(do_ + eidx_d_10);
        uint2* _vdst_7 = reinterpret_cast<uint2*>(&dwa[6]);
        *_vdst_7 = _vld_7;
    }
    int t_u_11 = g2 * 8 + sub * 4;
    long long iidx_d = ibase + (long long)(t_u_11 * rs);
    float dsel[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[_pair]) << 16);
        (&dsel[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[_pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_16 = 0; e_16 < 4; e_16++) {
        float d_e = 0.0f;
        if (t_u_11 < clen) {
            d_e = dsel[e_16];
        }
        dsel[e_16] = d_e;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(dsel[0 + 0], dsel[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(dsel[0 + 2], dsel[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx_d))[0]) = _pk2;
    }
    int t_u_12 = g2 * 8 + sub * 4 + 1;
    long long iidx_d_13 = ibase + (long long)(t_u_12 * rs);
    float dsel_14[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[2 + _pair]) << 16);
        (&dsel_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[2 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_17 = 0; e_17 < 4; e_17++) {
        float d_e_1 = 0.0f;
        if (t_u_12 < clen) {
            d_e_1 = dsel_14[e_17];
        }
        dsel_14[e_17] = d_e_1;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(dsel_14[0 + 0], dsel_14[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(dsel_14[0 + 2], dsel_14[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx_d_13))[0]) = _pk2;
    }
    int t_u_15 = g2 * 8 + sub * 4 + 2;
    long long iidx_d_16 = ibase + (long long)(t_u_15 * rs);
    float dsel_17[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[4 + _pair]) << 16);
        (&dsel_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[4 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_18 = 0; e_18 < 4; e_18++) {
        float d_e_2 = 0.0f;
        if (t_u_15 < clen) {
            d_e_2 = dsel_17[e_18];
        }
        dsel_17[e_18] = d_e_2;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(dsel_17[0 + 0], dsel_17[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(dsel_17[0 + 2], dsel_17[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx_d_16))[0]) = _pk2;
    }
    int t_u_18 = g2 * 8 + sub * 4 + 3;
    long long iidx_d_19 = ibase + (long long)(t_u_18 * rs);
    float dsel_20[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[6 + _pair]) << 16);
        (&dsel_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[6 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_19 = 0; e_19 < 4; e_19++) {
        float d_e_3 = 0.0f;
        if (t_u_18 < clen) {
            d_e_3 = dsel_20[e_19];
        }
        dsel_20[e_19] = d_e_3;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(dsel_20[0 + 0], dsel_20[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(dsel_20[0 + 2], dsel_20[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx_d_19))[0]) = _pk2;
    }
    if (clen == 0) {
        float zq[4];
        #pragma unroll
        for (int e_20 = 0; e_20 < 4; e_20++) {
            zq[e_20] = 0.0f;
        }
        #pragma unroll
        for (int u_3 = 0; u_3 < 4; u_3++) {
            int t_z = g2 * 8 + sub * 4 + u_3;
            long long zidx = ibase + (long long)(t_z * rs);
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(zq[0 + 0], zq[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zq[0 + 2], zq[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(v_new + zidx))[0]) = _pk2;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(zq[0 + 0], zq[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zq[0 + 2], zq[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dv2 + zidx))[0]) = _pk2;
            }
        }
    }
    if (copy_saved != 0) {
        unsigned int aw[4];
        unsigned int akw[4];
        int qd = half * 512 + tid;
        int t_a = qd / 16;
        int s0 = (qd - t_a * 16) * 4;
        int t_ac = t_a;
        if (t_a > clast) {
            t_ac = clast;
        }
        long long eidx_a = ((ext0 + (long long)t_ac) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(aqk + eidx_a);
            uint2* _vdst_8 = reinterpret_cast<uint2*>(&aw[0]);
            *_vdst_8 = _vld_8;
        }
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(akk + eidx_a);
            uint2* _vdst_9 = reinterpret_cast<uint2*>(&akw[0]);
            *_vdst_9 = _vld_9;
        }
        int qd_0 = half * 512 + tid + 256;
        int t_a_1 = qd_0 / 16;
        int s0_2 = (qd_0 - t_a_1 * 16) * 4;
        int t_ac_3 = t_a_1;
        if (t_a_1 > clast) {
            t_ac_3 = clast;
        }
        long long eidx_a_4 = ((ext0 + (long long)t_ac_3) * (long long)num_heads + (long long)head) * 64 + (long long)s0_2;
        {
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(aqk + eidx_a_4);
            uint2* _vdst_10 = reinterpret_cast<uint2*>(&aw[2]);
            *_vdst_10 = _vld_10;
        }
        {
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(akk + eidx_a_4);
            uint2* _vdst_11 = reinterpret_cast<uint2*>(&akw[2]);
            *_vdst_11 = _vld_11;
        }
        int qd_5 = half * 512 + tid;
        int t_a_6 = qd_5 / 16;
        int s0_7 = (qd_5 - t_a_6 * 16) * 4;
        long long iidx_a = ((int0 + (long long)t_a_6) * (long long)num_heads + (long long)head) * 64 + (long long)s0_7;
        float raw[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&raw[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(aw[_pair]) << 16);
            (&raw[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(aw[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int e_21 = 0; e_21 < 4; e_21++) {
            float val = 0.0f;
            if (t_a_6 < clen) {
                if (t_a_6 >= s0_7 + e_21) {
                    val = raw[e_21];
                }
            }
            raw[e_21] = val;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(raw[0 + 0], raw[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(raw[0 + 2], raw[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(aqk_tril + iidx_a))[0]) = _pk2;
        }
        float rawk[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&rawk[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(akw[_pair]) << 16);
            (&rawk[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(akw[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int e_22 = 0; e_22 < 4; e_22++) {
            float valk = 0.0f;
            if (t_a_6 < clen) {
                valk = rawk[e_22];
            }
            rawk[e_22] = valk;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(rawk[0 + 0], rawk[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(rawk[0 + 2], rawk[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(akk_int + iidx_a))[0]) = _pk2;
        }
        int qd_8 = half * 512 + tid + 256;
        int t_a_9 = qd_8 / 16;
        int s0_10 = (qd_8 - t_a_9 * 16) * 4;
        long long iidx_a_11 = ((int0 + (long long)t_a_9) * (long long)num_heads + (long long)head) * 64 + (long long)s0_10;
        float raw_12[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&raw_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(aw[2 + _pair]) << 16);
            (&raw_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(aw[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int e_23 = 0; e_23 < 4; e_23++) {
            float val_1 = 0.0f;
            if (t_a_9 < clen) {
                if (t_a_9 >= s0_10 + e_23) {
                    val_1 = raw_12[e_23];
                }
            }
            raw_12[e_23] = val_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(raw_12[0 + 0], raw_12[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(raw_12[0 + 2], raw_12[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(aqk_tril + iidx_a_11))[0]) = _pk2;
        }
        float rawk_13[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&rawk_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(akw[2 + _pair]) << 16);
            (&rawk_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(akw[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int e_24 = 0; e_24 < 4; e_24++) {
            float valk_1 = 0.0f;
            if (t_a_9 < clen) {
                valk_1 = rawk_13[e_24];
            }
            rawk_13[e_24] = valk_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(rawk_13[0 + 0], rawk_13[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(rawk_13[0 + 2], rawk_13[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(akk_int + iidx_a_11))[0]) = _pk2;
        }
    }
}

} // extern "C"

