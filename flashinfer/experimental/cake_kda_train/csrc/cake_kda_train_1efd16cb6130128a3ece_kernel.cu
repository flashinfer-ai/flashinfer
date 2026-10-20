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
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_1efd16cb6130128a3ece(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ g, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ v_new, int num_qk_heads, int num_heads, int group)
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
    int by = blockIdx.y;
    int nbx = (int)gridDim.x;
    int lin = by * nbx + bx;
    int chunk = lin / num_heads;
    int head = lin - chunk * num_heads;
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
    float gq[16];
    unsigned int qw[8];
    unsigned int kw[8];
    unsigned int vw[8];
    float btv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) {
        int t_u = warp_0 * 4 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long eidx = ebase + (long long)(t_c * rs);
        long long qidx = qbase + (long long)(t_c * qs);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(g + eidx);
            gq[u * 4 + 0] = _v4.x;
            gq[u * 4 + 1] = _v4.y;
            gq[u * 4 + 2] = _v4.z;
            gq[u * 4 + 3] = _v4.w;
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(q + qidx);
            uint2* _vdst_1 = reinterpret_cast<uint2*>(&qw[u * 2]);
            *_vdst_1 = _vld_1;
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k + qidx);
            uint2* _vdst_2 = reinterpret_cast<uint2*>(&kw[u * 2]);
            *_vdst_2 = _vld_2;
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(v + eidx);
            uint2* _vdst_3 = reinterpret_cast<uint2*>(&vw[u * 2]);
            *_vdst_3 = _vld_3;
        }
        btv[u] = (float)beta[bbase + (long long)(t_c * num_heads)];
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 4; u_1++) {
        int t_u_1 = warp_0 * 4 + u_1;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float g_u = 0.0f;
            if (t_u_1 < clen) {
                g_u = gq[u_1 * 4 + e];
            }
            gq[u_1 * 4 + e] = g_u;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u_1 * 128 + d0) * 4)), "f"(gq[u_1 * 4]), "f"(gq[u_1 * 4 + 1]), "f"(gq[u_1 * 4 + 2]), "f"(gq[u_1 * 4 + 3]) : "memory");
        float bt_u = 0.0f;
        if (t_u_1 < clen) {
            bt_u = btv[u_1];
        }
        btv[u_1] = bt_u;
    }
    __syncthreads();
    int grp = warp_0 / 2;
    int pos0 = (warp_0 - grp * 2) * 4;
    float local[4];
    #pragma unroll
    for (int e_1 = 0; e_1 < 4; e_1++) {
        local[e_1] = 0.0f;
    }
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        if (pos0 >= p) {
            float gp[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gp[0])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((grp * 8 + p) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                local[e_2] = local[e_2] + gp[e_2];
            }
        }
    }
    if (pos0 == 4) {
        float tot[4];
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            tot[e_3] = local[e_3];
        }
        #pragma unroll
        for (int u_2 = 1; u_2 < 4; u_2++) {
            float gu[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gu[0])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((warp_0 * 4 + u_2) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                tot[e_4] = tot[e_4] + gu[e_4];
            }
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((grp * 128 + d0) * 4)), "f"(tot[0]), "f"(tot[1]), "f"(tot[2]), "f"(tot[3]) : "memory");
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
    float zero[4];
    float acc[4];
    #pragma unroll
    for (int e_9 = 0; e_9 < 4; e_9++) {
        zero[e_9] = 0.0f;
        acc[e_9] = local[e_9];
    }
    #pragma unroll
    for (int u_3 = 0; u_3 < 4; u_3++) {
        int t_u_2 = warp_0 * 4 + u_3;
        long long iidx = ibase + (long long)(t_u_2 * rs);
        if (u_3 > 0) {
            float gu2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gu2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 3]))
                : "r"(graw_addr + (unsigned int)((t_u_2 * 128 + d0) * 4)));
            #pragma unroll
            for (int e_10 = 0; e_10 < 4; e_10++) {
                acc[e_10] = acc[e_10] + gu2[e_10];
            }
        }
        float _unpack_bf16_0[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((qw + u_3 * 2)[_pair]) << 16);
            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((qw + u_3 * 2)[_pair]) & 0xffff0000u);
        }
        float _unpack_bf16_1[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((kw + u_3 * 2)[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((kw + u_3 * 2)[_pair]) & 0xffff0000u);
        }
        float _unpack_bf16_2[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((vw + u_3 * 2)[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((vw + u_3 * 2)[_pair]) & 0xffff0000u);
        }
        float bt = btv[u_3];
        float gkt[4];
        float vb[4];
        float kb[4];
        float kg[4];
        float qg[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            gkt[e_11] = (off[e_11] + acc[e_11]) * 1.4426950408889634f;
            float _exp2_0 = approx_exp2(gkt[e_11]);
            float e_g = _exp2_0;
            float q_e = 0.0f;
            float k_e = 0.0f;
            float v_e = 0.0f;
            if (t_u_2 < clen) {
                q_e = _unpack_bf16_0[e_11];
                k_e = _unpack_bf16_1[e_11];
                v_e = _unpack_bf16_2[e_11];
            }
            vb[e_11] = v_e * bt;
            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)(k_e * bt);
            float _cvt_f32_0 = (float)_cvt_bf16_0;
            kb[e_11] = _cvt_f32_0 * e_g;
            float _exp2_1 = approx_exp2(gk_n[e_11] - gkt[e_11]);
            kg[e_11] = k_e * _exp2_1;
            qg[e_11] = q_e * e_g;
        }
        if (u_3 == 3) {
            if (warp_0 == 15) {
                {
                    float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
                    *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
                }
            }
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
            _pk[0] = __floats2bfloat162_rn(kb[0 + 0], kb[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kb[0 + 2], kb[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kb_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kg[0 + 0], kg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kg[0 + 2], kg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kg_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(qg[0 + 0], qg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qg[0 + 2], qg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qg_out + iidx))[0]) = _pk2;
        }
        if (clen == 0) {
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(zero[0 + 0], zero[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zero[0 + 2], zero[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(v_new + iidx))[0]) = _pk2;
            }
        }
    }
}

} // extern "C"

