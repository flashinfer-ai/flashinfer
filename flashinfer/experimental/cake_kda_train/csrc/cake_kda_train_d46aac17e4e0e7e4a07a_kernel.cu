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
#define THREADS 1024

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_kda_train_d46aac17e4e0e7e4a07a(float* __restrict__ g, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ aqk, __nv_bfloat16* __restrict__ akk, __nv_bfloat16* __restrict__ do_, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ ke_out, __nv_bfloat16* __restrict__ qe_out, __nv_bfloat16* __restrict__ aqk_tril, __nv_bfloat16* __restrict__ akk_int, __nv_bfloat16* __restrict__ do_int, __nv_bfloat16* __restrict__ v_int, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group, int copy_saved)
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
    int chunk = blockIdx.x;
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
    float gates[8];
    #pragma unroll
    for (int u = 0; u < 2; u++) {
        int t_u = warp_0 * 2 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long gidx = ebase + (long long)(t_c * rs);
        float x[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(g + gidx);
            x[0 + 0] = _v4.x;
            x[0 + 1] = _v4.y;
            x[0 + 2] = _v4.z;
            x[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float gate_u = 0.0f;
            if (t_u < clen) {
                gate_u = x[e];
            }
            gates[u * 4 + e] = gate_u;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u * 128 + d0) * 4)), "f"(gates[u * 4]), "f"(gates[u * 4 + 1]), "f"(gates[u * 4 + 2]), "f"(gates[u * 4 + 3]) : "memory");
    }
    float bt_l[2];
    #pragma unroll
    for (int u_1 = 0; u_1 < 2; u_1++) {
        int t_c_1 = warp_0 * 2 + u_1;
        if (t_c_1 > clast) {
            t_c_1 = clast;
        }
        bt_l[u_1] = (float)beta[bbase + (long long)(t_c_1 * num_heads)];
    }
    int t_a = tid / 16;
    int s0 = (tid - t_a * 16) * 4;
    int t_ac = t_a;
    if (t_a > clast) {
        t_ac = clast;
    }
    long long iidx_a = ((int0 + (long long)t_a) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
    long long eidx_a = ((ext0 + (long long)t_ac) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
    float raw[4];
    #pragma unroll
    for (int e_1 = 0; e_1 < 4; e_1++) {
        raw[e_1] = 0.0f;
    }
    if (copy_saved != 0) {
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(aqk + eidx_a);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&raw[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&raw[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
    }
    __syncthreads();
    int grp = warp_0 / 4;
    int pos0 = (warp_0 - grp * 4) * 2;
    float local[4];
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        local[e_2] = 0.0f;
    }
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        if (pos0 >= p) {
            float gp[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gp[0])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((grp * 8 + p) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_3 = 0; e_3 < 4; e_3++) {
                local[e_3] = local[e_3] + gp[e_3];
            }
        }
    }
    #pragma unroll
    for (int e_4 = 0; e_4 < 4; e_4++) {
        gates[e_4] = local[e_4];
        gates[4 + e_4] = local[e_4] + gates[4 + e_4];
    }
    if (pos0 == 6) {
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((grp * 128 + d0) * 4)), "f"(gates[4]), "f"(gates[5]), "f"(gates[6]), "f"(gates[7]) : "memory");
    }
    __syncthreads();
    float off[4];
    float total[4];
    float gk_n[4];
    #pragma unroll
    for (int e_5 = 0; e_5 < 4; e_5++) {
        off[e_5] = 0.0f;
        total[e_5] = 0.0f;
    }
    #pragma unroll
    for (int gi = 0; gi < 8; gi++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(gtot_addr + (unsigned int)((gi * 128 + d0) * 4)));
        if (grp > gi) {
            #pragma unroll
            for (int e_6 = 0; e_6 < 4; e_6++) {
                off[e_6] = off[e_6] + tg[e_6];
            }
        }
        #pragma unroll
        for (int e_7 = 0; e_7 < 4; e_7++) {
            total[e_7] = total[e_7] + tg[e_7];
        }
    }
    #pragma unroll
    for (int e_8 = 0; e_8 < 4; e_8++) {
        gk_n[e_8] = total[e_8] * 1.4426950408889634f;
    }
    #pragma unroll
    for (int u_2 = 0; u_2 < 2; u_2++) {
        int t_u_1 = warp_0 * 2 + u_2;
        int t_c_2 = t_u_1;
        if (t_u_1 > clast) {
            t_c_2 = clast;
        }
        long long iidx = ibase + (long long)(t_u_1 * rs);
        long long eidx = ebase + (long long)(t_c_2 * rs);
        long long qidx = qbase + (long long)(t_c_2 * qs);
        float kv[4];
        float qv[4];
        float vv[4];
        float dov[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k + qidx);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&kv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&kv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q + qidx);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&qv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&qv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(v + eidx);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&vv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&vv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(do_ + eidx);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&dov[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&dov[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        float bt = 0.0f;
        if (t_u_1 < clen) {
            bt = bt_l[u_2];
        }
        float gkt[4];
        #pragma unroll
        for (int e_9 = 0; e_9 < 4; e_9++) {
            gkt[e_9] = (off[e_9] + gates[u_2 * 4 + e_9]) * 1.4426950408889634f;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
        }
        float eg[4];
        #pragma unroll
        for (int e_10 = 0; e_10 < 4; e_10++) {
            float _exp2_0 = approx_exp2(gkt[e_10]);
            eg[e_10] = _exp2_0;
        }
        float vsel[4];
        float vb[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            float v_e = 0.0f;
            if (t_u_1 < clen) {
                v_e = vv[e_11];
            }
            vsel[e_11] = v_e;
            vb[e_11] = v_e * bt;
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
        for (int e_12 = 0; e_12 < 4; e_12++) {
            float k_e = 0.0f;
            if (t_u_1 < clen) {
                k_e = kv[e_12];
            }
            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)(k_e * bt);
            float _cvt_f32_0 = (float)_cvt_bf16_0;
            kb[e_12] = _cvt_f32_0 * eg[e_12];
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
        for (int e_13 = 0; e_13 < 4; e_13++) {
            float k_e_1 = 0.0f;
            if (t_u_1 < clen) {
                k_e_1 = kv[e_13];
            }
            float _exp2_1 = approx_exp2(gk_n[e_13] - gkt[e_13]);
            kg[e_13] = k_e_1 * _exp2_1;
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
        for (int e_14 = 0; e_14 < 4; e_14++) {
            float k_e_2 = 0.0f;
            if (t_u_1 < clen) {
                k_e_2 = kv[e_14];
            }
            ke[e_14] = k_e_2;
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
        for (int e_15 = 0; e_15 < 4; e_15++) {
            float q_e = 0.0f;
            if (t_u_1 < clen) {
                q_e = qv[e_15];
            }
            qg[e_15] = q_e * eg[e_15];
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
        for (int e_16 = 0; e_16 < 4; e_16++) {
            float q_e_1 = 0.0f;
            if (t_u_1 < clen) {
                q_e_1 = qv[e_16];
            }
            qe[e_16] = q_e_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(qe[0 + 0], qe[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qe[0 + 2], qe[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qe_out + iidx))[0]) = _pk2;
        }
        float dsel[4];
        #pragma unroll
        for (int e_17 = 0; e_17 < 4; e_17++) {
            float d_e = 0.0f;
            if (t_u_1 < clen) {
                d_e = dov[e_17];
            }
            dsel[e_17] = d_e;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(dsel[0 + 0], dsel[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(dsel[0 + 2], dsel[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx))[0]) = _pk2;
        }
        if (lane_1 == 0) {
            beta_int[(int0 + (long long)t_u_1) * (long long)num_heads + (long long)head] = bt;
        }
    }
    if (clen == 0) {
        float zq[4];
        #pragma unroll
        for (int e_18 = 0; e_18 < 4; e_18++) {
            zq[e_18] = 0.0f;
        }
        #pragma unroll
        for (int u_3 = 0; u_3 < 2; u_3++) {
            int t_z = warp_0 * 2 + u_3;
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
        #pragma unroll
        for (int e_19 = 0; e_19 < 4; e_19++) {
            float val = 0.0f;
            if (t_a < clen) {
                if (t_a >= s0 + e_19) {
                    val = raw[e_19];
                }
            }
            raw[e_19] = val;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(raw[0 + 0], raw[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(raw[0 + 2], raw[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(aqk_tril + iidx_a))[0]) = _pk2;
        }
        float rawk[4];
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(akk + eidx_a);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&rawk[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&rawk[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int e_20 = 0; e_20 < 4; e_20++) {
            float valk = 0.0f;
            if (t_a < clen) {
                valk = rawk[e_20];
            }
            rawk[e_20] = valk;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(rawk[0 + 0], rawk[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(rawk[0 + 2], rawk[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(akk_int + iidx_a))[0]) = _pk2;
        }
    }
}

} // extern "C"

