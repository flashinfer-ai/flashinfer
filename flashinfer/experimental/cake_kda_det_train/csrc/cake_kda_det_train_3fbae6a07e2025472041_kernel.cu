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

#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_det_train_3fbae6a07e2025472041(__nv_bfloat16* __restrict__ g_raw, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ kg_out, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group, float lower_bound, long long g_raw_row_stride, long long g_raw_head_stride, __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_raw, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ qk_rstd, int num_tokens, int q_raw_row_stride, int q_raw_head_stride, int k_raw_row_stride, int k_raw_head_stride, int read_rstd, int write_rstd)
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

    // Prologue
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
    long long qbase = (ext0 * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d0;
    long long bbase = ext0 * (long long)num_heads + (long long)head;
    long long qrbase = (long long)qk_head * (long long)q_raw_head_stride + (long long)d0;
    long long krbase = (long long)qk_head * (long long)k_raw_head_stride + (long long)d0;
    int l2n_lead = 0;
    if (head == qk_head * group) {
        l2n_lead = 1;
    }
    float rg_rate = __nv_expf(A_log[head]);
    float rg_clr;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(rg_clr) : "f"(lower_bound), "f"(rg_rate));
    float bias[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        bias[e] = dt_bias[head * 128 + d0 + e];
    }
    long long rbase = (long long)head * g_raw_head_stride + (long long)d0;
    float x[32];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        int t_u = warp_0 * 8 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long gidx = rbase + (ext0 + (long long)t_c) * g_raw_row_stride;
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(g_raw + gidx);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x[u * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&x[u * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
    }
    float run[4];
    #pragma unroll
    for (int e_1 = 0; e_1 < 4; e_1++) {
        run[e_1] = 0.0f;
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        int t_u_1 = warp_0 * 8 + u_1;
        float gq[4];
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            float rg_u;
            asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(rg_u) : "f"(x[u_1 * 4 + e_2]), "f"(bias[e_2]));
            float rg_z;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(rg_z) : "f"(rg_rate), "f"(rg_u));
            float _neg_1;
            asm("neg.f32 %0, %1;" : "=f"(_neg_1) : "f"(rg_z));
            float rg_e = __nv_expf(_neg_1);
            float rg_den;
            asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(rg_den) : "f"(1.0f), "f"(rg_e));
            float rg_quot = __fdiv_rn(lower_bound, rg_den);
            float rg_gam = ((t_u_1 < clen) ? rg_quot : 0.0f);
            gq[e_2] = rg_gam;
            run[e_2] = run[e_2] + rg_gam;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u_1 * 128 + d0) * 4)), "f"(gq[0]), "f"(gq[1]), "f"(gq[2]), "f"(gq[3]) : "memory");
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((warp_0 * 128 + d0) * 4)), "f"(run[0]), "f"(run[1]), "f"(run[2]), "f"(run[3]) : "memory");
    int t_c0 = warp_0 * 8;
    if (t_c0 > clast) {
        t_c0 = clast;
    }
    float kv_n[4];
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(k_raw + krbase + (ext0 + (long long)t_c0) * (long long)k_raw_row_stride);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&kv_n[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&kv_n[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
    }
    float qv_n[4];
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(q_raw + qrbase + (ext0 + (long long)t_c0) * (long long)q_raw_row_stride);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&qv_n[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
            (&qv_n[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
    }
    float lr_r = 0.0f;
    if (read_rstd != 0) {
        lr_r = qk_rstd[(0 * (long long)num_tokens + (long long)(ext0 + (long long)t_c0)) * (long long)num_qk_heads + (long long)qk_head];
    }
    float rq_n = lr_r;
    float lr_r_0 = 0.0f;
    if (read_rstd != 0) {
        lr_r_0 = qk_rstd[(1 * (long long)num_tokens + (long long)(ext0 + (long long)t_c0)) * (long long)num_qk_heads + (long long)qk_head];
    }
    float rk_n = lr_r_0;
    float bt_n = (float)beta[bbase + (long long)(t_c0 * num_heads)];
    __syncthreads();
    float off[4];
    float total[4];
    float gk_n[4];
    #pragma unroll
    for (int e_3 = 0; e_3 < 4; e_3++) {
        off[e_3] = 0.0f;
        total[e_3] = 0.0f;
    }
    #pragma unroll
    for (int gi = 0; gi < 8; gi++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(gtot_addr + (unsigned int)((gi * 128 + d0) * 4)));
        if (warp_0 > gi) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                off[e_4] = off[e_4] + tg[e_4];
            }
        }
        #pragma unroll
        for (int e_5 = 0; e_5 < 4; e_5++) {
            total[e_5] = total[e_5] + tg[e_5];
        }
    }
    #pragma unroll
    for (int e_6 = 0; e_6 < 4; e_6++) {
        gk_n[e_6] = total[e_6] * 1.442695f;
    }
    float acc[4];
    #pragma unroll
    for (int e_7 = 0; e_7 < 4; e_7++) {
        acc[e_7] = 0.0f;
    }
    #pragma unroll
    for (int u_2 = 0; u_2 < 8; u_2++) {
        int t_u_2 = warp_0 * 8 + u_2;
        long long iidx = ibase + (long long)(t_u_2 * rs);
        float bt_l = bt_n;
        float kv[4];
        float qv[4];
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            kv[e_8] = kv_n[e_8];
            qv[e_8] = qv_n[e_8];
        }
        float rq = rq_n;
        float rk = rk_n;
        if (u_2 < 7) {
            int t_c1 = t_u_2 + 1;
            if (t_c1 > clast) {
                t_c1 = clast;
            }
            {
                uint2 _vld_4;
                _vld_4 = *reinterpret_cast<const uint2*>(k_raw + krbase + (ext0 + (long long)t_c1) * (long long)k_raw_row_stride);
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&kv_n[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&kv_n[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_5;
                _vld_5 = *reinterpret_cast<const uint2*>(q_raw + qrbase + (ext0 + (long long)t_c1) * (long long)q_raw_row_stride);
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&qv_n[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&qv_n[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
            float lr_r_1 = 0.0f;
            if (read_rstd != 0) {
                lr_r_1 = qk_rstd[(0 * (long long)num_tokens + (long long)(ext0 + (long long)t_c1)) * (long long)num_qk_heads + (long long)qk_head];
            }
            float rq_1 = lr_r_1;
            float lr_r_2 = 0.0f;
            if (read_rstd != 0) {
                lr_r_2 = qk_rstd[(1 * (long long)num_tokens + (long long)(ext0 + (long long)t_c1)) * (long long)num_qk_heads + (long long)qk_head];
            }
            float rk_1 = lr_r_2;
            rq_n = rq_1;
            rk_n = rk_1;
            bt_n = (float)beta[bbase + (long long)(t_c1 * num_heads)];
        }
        int t_n = t_u_2;
        if (t_u_2 > clast) {
            t_n = clast;
        }
        long long nidx = qbase + (long long)(t_u_2 * qs);
        float r = 0.0f;
        if (read_rstd != 0) {
            r = rq;
        } else {
            float s;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(qv[0]), "f"(qv[0]));
            float _fma_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(qv[1]), "f"(qv[1]), "f"(s));
            s = _fma_6;
            float _fma_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(qv[2]), "f"(qv[2]), "f"(s));
            s = _fma_7;
            float _fma_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(qv[3]), "f"(qv[3]), "f"(s));
            s = _fma_8;
            float s_even = __shfl_xor_sync(0xFFFFFFFF, s, 1);
            float c = s_even;
            float _fma_9;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(qv[0]), "f"(qv[0]), "f"(c));
            c = _fma_9;
            float _fma_10;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(qv[1]), "f"(qv[1]), "f"(c));
            c = _fma_10;
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(qv[2]), "f"(qv[2]), "f"(c));
            c = _fma_11;
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(qv[3]), "f"(qv[3]), "f"(c));
            c = _fma_12;
            float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, c, 16);
            float _add_13;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(c), "f"(_shfl_xor_sync_14));
            c = _add_13;
            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, c, 8);
            float _add_15;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(c), "f"(_shfl_xor_sync_16));
            c = _add_15;
            float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
            float _add_17;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(c), "f"(_shfl_xor_sync_18));
            c = _add_17;
            float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, c, 2);
            float _add_19;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(c), "f"(_shfl_xor_sync_20));
            c = _add_19;
            float c_odd = __shfl_xor_sync(0xFFFFFFFF, c, 1);
            if ((lane_1 & 1) == 0) {
                c = c_odd;
            }
            float _add_23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(c), "f"(1e-06f));
            float _sqrt_22;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_22) : "f"(_add_23));
            float _fdiv_21 = __fdiv_rn(1.0f, _sqrt_22);
            r = _fdiv_21;
            if (write_rstd != 0) {
                if (l2n_lead != 0) {
                    if (t_u_2 < clen) {
                        if (lane_1 == 0) {
                            *(reinterpret_cast<float*>(qk_rstd + ((0 * (long long)num_tokens + (long long)(ext0 + (long long)t_n)) * (long long)num_qk_heads + (long long)qk_head)) + (0)) = r;
                        }
                    }
                }
            }
        }
        float y[4];
        y[0] = mul_f32_rn_noftz(qv[0], r);
        y[1] = mul_f32_rn_noftz(qv[1], r);
        y[2] = mul_f32_rn_noftz(qv[2], r);
        y[3] = mul_f32_rn_noftz(qv[3], r);
        unsigned int yb[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
            yb[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 2], y[_lp*2+1 + 2]));
            yb[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&qv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yb[_pair]) << 16);
            (&qv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yb[_pair]) & 0xffff0000u);
        }
        if (l2n_lead != 0) {
            if (t_u_2 < clen) {
                reinterpret_cast<int2*>(qn + nidx)[0] = reinterpret_cast<int2*>(yb)[0];
            }
        }
        float r_0 = 0.0f;
        if (read_rstd != 0) {
            r_0 = rk;
        } else {
            float s_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(kv[0]), "f"(kv[0]));
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(kv[1]), "f"(kv[1]), "f"(s_1));
            s_1 = _fma_24;
            float _fma_25;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(kv[2]), "f"(kv[2]), "f"(s_1));
            s_1 = _fma_25;
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(kv[3]), "f"(kv[3]), "f"(s_1));
            s_1 = _fma_26;
            float s_even_1 = __shfl_xor_sync(0xFFFFFFFF, s_1, 1);
            float c_1 = s_even_1;
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(kv[0]), "f"(kv[0]), "f"(c_1));
            c_1 = _fma_27;
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(kv[1]), "f"(kv[1]), "f"(c_1));
            c_1 = _fma_28;
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(kv[2]), "f"(kv[2]), "f"(c_1));
            c_1 = _fma_29;
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(kv[3]), "f"(kv[3]), "f"(c_1));
            c_1 = _fma_30;
            float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, c_1, 16);
            float _add_31;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(c_1), "f"(_shfl_xor_sync_32));
            c_1 = _add_31;
            float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, c_1, 8);
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(c_1), "f"(_shfl_xor_sync_34));
            c_1 = _add_33;
            float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, c_1, 4);
            float _add_35;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(c_1), "f"(_shfl_xor_sync_36));
            c_1 = _add_35;
            float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, c_1, 2);
            float _add_37;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(c_1), "f"(_shfl_xor_sync_38));
            c_1 = _add_37;
            float c_odd_1 = __shfl_xor_sync(0xFFFFFFFF, c_1, 1);
            if ((lane_1 & 1) == 0) {
                c_1 = c_odd_1;
            }
            float _add_41;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(c_1), "f"(1e-06f));
            float _sqrt_40;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_40) : "f"(_add_41));
            float _fdiv_39 = __fdiv_rn(1.0f, _sqrt_40);
            r_0 = _fdiv_39;
            if (write_rstd != 0) {
                if (l2n_lead != 0) {
                    if (t_u_2 < clen) {
                        if (lane_1 == 0) {
                            *(reinterpret_cast<float*>(qk_rstd + ((1 * (long long)num_tokens + (long long)(ext0 + (long long)t_n)) * (long long)num_qk_heads + (long long)qk_head)) + (0)) = r_0;
                        }
                    }
                }
            }
        }
        float y_0[4];
        y_0[0] = mul_f32_rn_noftz(kv[0], r_0);
        y_0[1] = mul_f32_rn_noftz(kv[1], r_0);
        y_0[2] = mul_f32_rn_noftz(kv[2], r_0);
        y_0[3] = mul_f32_rn_noftz(kv[3], r_0);
        unsigned int yb_0[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_0[_lp*2 + 0], y_0[_lp*2+1 + 0]));
            yb_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_0[_lp*2 + 2], y_0[_lp*2+1 + 2]));
            yb_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&kv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yb_0[_pair]) << 16);
            (&kv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yb_0[_pair]) & 0xffff0000u);
        }
        if (l2n_lead != 0) {
            if (t_u_2 < clen) {
                reinterpret_cast<int2*>(kn + nidx)[0] = reinterpret_cast<int2*>(yb_0)[0];
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
        for (int e_9 = 0; e_9 < 4; e_9++) {
            acc[e_9] = acc[e_9] + gq2[e_9];
            gkt[e_9] = (off[e_9] + acc[e_9]) * 1.442695f;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
        }
        float eg[4];
        #pragma unroll
        for (int e_10 = 0; e_10 < 4; e_10++) {
            float _exp2_42 = approx_exp2(gkt[e_10]);
            eg[e_10] = _exp2_42;
        }
        float kg[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            float k_e = 0.0f;
            if (t_u_2 < clen) {
                k_e = kv[e_11];
            }
            float _exp2_43 = approx_exp2(gk_n[e_11] - gkt[e_11]);
            kg[e_11] = k_e * _exp2_43;
        }
        unsigned int _packed_1[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 0], kg[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 2], kg[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(kg_out + iidx)[0] = reinterpret_cast<int2*>(_packed_1)[0];
        if (lane_1 == 0) {
            beta_int[(int0 + (long long)t_u_2) * (long long)num_heads + (long long)head] = bt;
        }
    }
    if (clen == 0) {
        float zq[4];
        #pragma unroll
        for (int e_12 = 0; e_12 < 4; e_12++) {
            zq[e_12] = 0.0f;
        }
        #pragma unroll
        for (int u_3 = 0; u_3 < 8; u_3++) {
            int t_z = warp_0 * 8 + u_3;
            long long zidx = ibase + (long long)(t_z * rs);
            unsigned int _packed_4[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 0], zq[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zq[_lp*2 + 2], zq[_lp*2+1 + 2]));
                _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(v_new + zidx)[0] = reinterpret_cast<int2*>(_packed_4)[0];
            reinterpret_cast<int2*>(dv2 + zidx)[0] = reinterpret_cast<int2*>(_packed_4)[0];
        }
    }
}

} // extern "C"
