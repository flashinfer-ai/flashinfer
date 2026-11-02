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
kernel_cake_kda_train_0e7c693f766175f4406c(float* __restrict__ g, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ kg_out, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group)
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

    // === Task calls (declaration order) ===
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
        if (warp_0 > gi) {
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
        gk_n[e_5] = total[e_5] * 1.442695f;
    }
    float acc[4];
    #pragma unroll
    for (int e_6 = 0; e_6 < 4; e_6++) {
        acc[e_6] = 0.0f;
    }
    #pragma unroll
    for (int u_2 = 0; u_2 < 8; u_2++) {
        int t_u_2 = warp_0 * 8 + u_2;
        int t_c_1 = t_u_2;
        if (t_u_2 > clast) {
            t_c_1 = clast;
        }
        long long iidx = ibase + (long long)(t_u_2 * rs);
        long long qidx = qbase + (long long)(t_c_1 * qs);
        float bt_l = (float)beta[bbase + (long long)(t_c_1 * num_heads)];
        float kv[4];
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
        for (int e_7 = 0; e_7 < 4; e_7++) {
            acc[e_7] = acc[e_7] + gq2[e_7];
            gkt[e_7] = (off[e_7] + acc[e_7]) * 1.442695f;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
        }
        float eg[4];
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            float _exp2_2 = approx_exp2(gkt[e_8]);
            eg[e_8] = _exp2_2;
        }
        float kg[4];
        #pragma unroll
        for (int e_9 = 0; e_9 < 4; e_9++) {
            float k_e = 0.0f;
            if (t_u_2 < clen) {
                k_e = kv[e_9];
            }
            float _exp2_3 = approx_exp2(gk_n[e_9] - gkt[e_9]);
            kg[e_9] = k_e * _exp2_3;
        }
        unsigned int _packed_0[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 0], kg[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 2], kg[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(kg_out + iidx)[0] = reinterpret_cast<int2*>(_packed_0)[0];
        if (lane_1 == 0) {
            beta_int[(int0 + (long long)t_u_2) * (long long)num_heads + (long long)head] = bt;
        }
    }
    if (clen == 0) {
        float zq[4];
        #pragma unroll
        for (int e_10 = 0; e_10 < 4; e_10++) {
            zq[e_10] = 0.0f;
        }
        #pragma unroll
        for (int u_3 = 0; u_3 < 8; u_3++) {
            int t_z = warp_0 * 8 + u_3;
            long long zidx = ibase + (long long)(t_z * rs);
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 0], zq[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 2], zq[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(v_new + zidx)[0] = reinterpret_cast<int2*>(_packed_1)[0];
            unsigned int _packed_2[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 0], zq[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 2], zq[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dv2 + zidx)[0] = reinterpret_cast<int2*>(_packed_2)[0];
        }
    }
}

} // extern "C"
