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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_a8921482bf37aaee0133(float* __restrict__ g, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ do_, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ do_int, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group, __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_raw, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ qk_rstd, int num_tokens, int q_raw_row_stride, int q_raw_head_stride, int k_raw_row_stride, int k_raw_head_stride, int read_rstd, int write_rstd)
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
    long long ebase_g = (ext0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    float x[32];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        int t_u = warp_0 * 8 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long gidx = ebase_g + (long long)(t_c * rs);
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
        float bt_l = (float)beta[bbase + (long long)(t_c_1 * num_heads)];
        float kv[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k_raw + krbase + (ext0 + (long long)t_c_1) * (long long)k_raw_row_stride);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&kv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&kv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float qv[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q_raw + qrbase + (ext0 + (long long)t_c_1) * (long long)q_raw_row_stride);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&qv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&qv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        float lr_r = 0.0f;
        if (read_rstd != 0) {
            lr_r = qk_rstd[(0 * (long long)num_tokens + (long long)(ext0 + (long long)t_c_1)) * (long long)num_qk_heads + (long long)qk_head];
        }
        float rq = lr_r;
        float lr_r_0 = 0.0f;
        if (read_rstd != 0) {
            lr_r_0 = qk_rstd[(1 * (long long)num_tokens + (long long)(ext0 + (long long)t_c_1)) * (long long)num_qk_heads + (long long)qk_head];
        }
        float rk = lr_r_0;
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
            float _fma_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(qv[1]), "f"(qv[1]), "f"(s));
            s = _fma_3;
            float _fma_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(qv[2]), "f"(qv[2]), "f"(s));
            s = _fma_4;
            float _fma_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(qv[3]), "f"(qv[3]), "f"(s));
            s = _fma_5;
            float s_even = __shfl_xor_sync(0xFFFFFFFF, s, 1);
            float c = s_even;
            float _fma_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(qv[0]), "f"(qv[0]), "f"(c));
            c = _fma_6;
            float _fma_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(qv[1]), "f"(qv[1]), "f"(c));
            c = _fma_7;
            float _fma_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(qv[2]), "f"(qv[2]), "f"(c));
            c = _fma_8;
            float _fma_9;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(qv[3]), "f"(qv[3]), "f"(c));
            c = _fma_9;
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, c, 16);
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(c), "f"(_shfl_xor_sync_11));
            c = _add_10;
            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, c, 8);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(c), "f"(_shfl_xor_sync_13));
            c = _add_12;
            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(c), "f"(_shfl_xor_sync_15));
            c = _add_14;
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, c, 2);
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(c), "f"(_shfl_xor_sync_17));
            c = _add_16;
            float c_odd = __shfl_xor_sync(0xFFFFFFFF, c, 1);
            if ((lane_1 & 1) == 0) {
                c = c_odd;
            }
            float _add_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(c), "f"(1e-06f));
            float _sqrt_19;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_19) : "f"(_add_20));
            float _fdiv_18 = __fdiv_rn(1.0f, _sqrt_19);
            r = _fdiv_18;
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
            float _fma_21;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(kv[1]), "f"(kv[1]), "f"(s_1));
            s_1 = _fma_21;
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(kv[2]), "f"(kv[2]), "f"(s_1));
            s_1 = _fma_22;
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(kv[3]), "f"(kv[3]), "f"(s_1));
            s_1 = _fma_23;
            float s_even_1 = __shfl_xor_sync(0xFFFFFFFF, s_1, 1);
            float c_1 = s_even_1;
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(kv[0]), "f"(kv[0]), "f"(c_1));
            c_1 = _fma_24;
            float _fma_25;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(kv[1]), "f"(kv[1]), "f"(c_1));
            c_1 = _fma_25;
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(kv[2]), "f"(kv[2]), "f"(c_1));
            c_1 = _fma_26;
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(kv[3]), "f"(kv[3]), "f"(c_1));
            c_1 = _fma_27;
            float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, c_1, 16);
            float _add_28;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(c_1), "f"(_shfl_xor_sync_29));
            c_1 = _add_28;
            float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, c_1, 8);
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(c_1), "f"(_shfl_xor_sync_31));
            c_1 = _add_30;
            float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, c_1, 4);
            float _add_32;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(c_1), "f"(_shfl_xor_sync_33));
            c_1 = _add_32;
            float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, c_1, 2);
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(c_1), "f"(_shfl_xor_sync_35));
            c_1 = _add_34;
            float c_odd_1 = __shfl_xor_sync(0xFFFFFFFF, c_1, 1);
            if ((lane_1 & 1) == 0) {
                c_1 = c_odd_1;
            }
            float _add_38;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(c_1), "f"(1e-06f));
            float _sqrt_37;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_37) : "f"(_add_38));
            float _fdiv_36 = __fdiv_rn(1.0f, _sqrt_37);
            r_0 = _fdiv_36;
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
            float _exp2_39 = approx_exp2(gkt[e_8]);
            eg[e_8] = _exp2_39;
        }
        float kb[4];
        #pragma unroll
        for (int e_9 = 0; e_9 < 4; e_9++) {
            float k_e = 0.0f;
            if (t_u_2 < clen) {
                k_e = kv[e_9];
            }
            kb[e_9] = (float)(__nv_bfloat16)(k_e * bt) * eg[e_9];
        }
        unsigned int _packed_0[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb[_lp*2 + 0], kb[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb[_lp*2 + 2], kb[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(kb_out + iidx)[0] = reinterpret_cast<int2*>(_packed_0)[0];
        float kg[4];
        #pragma unroll
        for (int e_10 = 0; e_10 < 4; e_10++) {
            float k_e_1 = 0.0f;
            if (t_u_2 < clen) {
                k_e_1 = kv[e_10];
            }
            float _exp2_40 = approx_exp2(gk_n[e_10] - gkt[e_10]);
            kg[e_10] = k_e_1 * _exp2_40;
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
        float qg[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            float q_e = 0.0f;
            if (t_u_2 < clen) {
                q_e = qv[e_11];
            }
            qg[e_11] = q_e * eg[e_11];
        }
        unsigned int _packed_2[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg[_lp*2 + 0], qg[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg[_lp*2 + 2], qg[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(qg_out + iidx)[0] = reinterpret_cast<int2*>(_packed_2)[0];
        if (lane_1 == 0) {
            beta_int[(int0 + (long long)t_u_2) * (long long)num_heads + (long long)head] = bt;
        }
    }
    long long ebase = (ext0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    unsigned int dwa[16];
    int t_u_0 = warp_0 * 8;
    int t_c_0 = t_u_0;
    if (t_u_0 > clast) {
        t_c_0 = clast;
    }
    long long eidx_d_0 = ebase + (long long)(t_c_0 * rs);
    {
        uint2 _vld_41;
        _vld_41 = *reinterpret_cast<const uint2*>(do_ + eidx_d_0);
        uint2* _vdst_41 = reinterpret_cast<uint2*>(&dwa[0]);
        *_vdst_41 = _vld_41;
    }
    int t_u_1_1 = warp_0 * 8 + 1;
    int t_c_1_1 = t_u_1_1;
    if (t_u_1_1 > clast) {
        t_c_1_1 = clast;
    }
    long long eidx_d_1 = ebase + (long long)(t_c_1_1 * rs);
    {
        uint2 _vld_42;
        _vld_42 = *reinterpret_cast<const uint2*>(do_ + eidx_d_1);
        uint2* _vdst_42 = reinterpret_cast<uint2*>(&dwa[2]);
        *_vdst_42 = _vld_42;
    }
    int t_u_2_1 = warp_0 * 8 + 2;
    int t_c_2 = t_u_2_1;
    if (t_u_2_1 > clast) {
        t_c_2 = clast;
    }
    long long eidx_d_2 = ebase + (long long)(t_c_2 * rs);
    {
        uint2 _vld_43;
        _vld_43 = *reinterpret_cast<const uint2*>(do_ + eidx_d_2);
        uint2* _vdst_43 = reinterpret_cast<uint2*>(&dwa[4]);
        *_vdst_43 = _vld_43;
    }
    int t_u_3 = warp_0 * 8 + 3;
    int t_c_3 = t_u_3;
    if (t_u_3 > clast) {
        t_c_3 = clast;
    }
    long long eidx_d_3 = ebase + (long long)(t_c_3 * rs);
    {
        uint2 _vld_44;
        _vld_44 = *reinterpret_cast<const uint2*>(do_ + eidx_d_3);
        uint2* _vdst_44 = reinterpret_cast<uint2*>(&dwa[6]);
        *_vdst_44 = _vld_44;
    }
    int t_u_4 = warp_0 * 8 + 4;
    int t_c_4 = t_u_4;
    if (t_u_4 > clast) {
        t_c_4 = clast;
    }
    long long eidx_d_4 = ebase + (long long)(t_c_4 * rs);
    {
        uint2 _vld_45;
        _vld_45 = *reinterpret_cast<const uint2*>(do_ + eidx_d_4);
        uint2* _vdst_45 = reinterpret_cast<uint2*>(&dwa[8]);
        *_vdst_45 = _vld_45;
    }
    int t_u_5 = warp_0 * 8 + 5;
    int t_c_5 = t_u_5;
    if (t_u_5 > clast) {
        t_c_5 = clast;
    }
    long long eidx_d_5 = ebase + (long long)(t_c_5 * rs);
    {
        uint2 _vld_46;
        _vld_46 = *reinterpret_cast<const uint2*>(do_ + eidx_d_5);
        uint2* _vdst_46 = reinterpret_cast<uint2*>(&dwa[10]);
        *_vdst_46 = _vld_46;
    }
    int t_u_6 = warp_0 * 8 + 6;
    int t_c_6 = t_u_6;
    if (t_u_6 > clast) {
        t_c_6 = clast;
    }
    long long eidx_d_6 = ebase + (long long)(t_c_6 * rs);
    {
        uint2 _vld_47;
        _vld_47 = *reinterpret_cast<const uint2*>(do_ + eidx_d_6);
        uint2* _vdst_47 = reinterpret_cast<uint2*>(&dwa[12]);
        *_vdst_47 = _vld_47;
    }
    int t_u_7 = warp_0 * 8 + 7;
    int t_c_7 = t_u_7;
    if (t_u_7 > clast) {
        t_c_7 = clast;
    }
    long long eidx_d_7 = ebase + (long long)(t_c_7 * rs);
    {
        uint2 _vld_48;
        _vld_48 = *reinterpret_cast<const uint2*>(do_ + eidx_d_7);
        uint2* _vdst_48 = reinterpret_cast<uint2*>(&dwa[14]);
        *_vdst_48 = _vld_48;
    }
    int t_u_8 = warp_0 * 8;
    long long iidx_d_0 = ibase + (long long)(t_u_8 * rs);
    float dsel_0[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[_pair]) << 16);
        (&dsel_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[_pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_0 = 0; e_0 < 4; e_0++) {
        float d_e_0 = 0.0f;
        if (t_u_8 < clen) {
            d_e_0 = dsel_0[e_0];
        }
        dsel_0[e_0] = d_e_0;
    }
    unsigned int _packed_3_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_0[_lp*2 + 0], dsel_0[_lp*2+1 + 0]));
        _packed_3_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_0[_lp*2 + 2], dsel_0[_lp*2+1 + 2]));
        _packed_3_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_0)[0] = reinterpret_cast<int2*>(_packed_3_0)[0];
    int t_u_9 = warp_0 * 8 + 1;
    long long iidx_d_1 = ibase + (long long)(t_u_9 * rs);
    float dsel_1[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[2 + _pair]) << 16);
        (&dsel_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[2 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_1_1 = 0; e_1_1 < 4; e_1_1++) {
        float d_e_1 = 0.0f;
        if (t_u_9 < clen) {
            d_e_1 = dsel_1[e_1_1];
        }
        dsel_1[e_1_1] = d_e_1;
    }
    unsigned int _packed_3_1[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_1[_lp*2 + 0], dsel_1[_lp*2+1 + 0]));
        _packed_3_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_1[_lp*2 + 2], dsel_1[_lp*2+1 + 2]));
        _packed_3_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_1)[0] = reinterpret_cast<int2*>(_packed_3_1)[0];
    int t_u_10 = warp_0 * 8 + 2;
    long long iidx_d_2 = ibase + (long long)(t_u_10 * rs);
    float dsel_2[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[4 + _pair]) << 16);
        (&dsel_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[4 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_2_1 = 0; e_2_1 < 4; e_2_1++) {
        float d_e_2 = 0.0f;
        if (t_u_10 < clen) {
            d_e_2 = dsel_2[e_2_1];
        }
        dsel_2[e_2_1] = d_e_2;
    }
    unsigned int _packed_3_2[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_2[_lp*2 + 0], dsel_2[_lp*2+1 + 0]));
        _packed_3_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_2[_lp*2 + 2], dsel_2[_lp*2+1 + 2]));
        _packed_3_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_2)[0] = reinterpret_cast<int2*>(_packed_3_2)[0];
    int t_u_11 = warp_0 * 8 + 3;
    long long iidx_d_3 = ibase + (long long)(t_u_11 * rs);
    float dsel_3[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[6 + _pair]) << 16);
        (&dsel_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[6 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_3_1 = 0; e_3_1 < 4; e_3_1++) {
        float d_e_3 = 0.0f;
        if (t_u_11 < clen) {
            d_e_3 = dsel_3[e_3_1];
        }
        dsel_3[e_3_1] = d_e_3;
    }
    unsigned int _packed_3_3[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_3[_lp*2 + 0], dsel_3[_lp*2+1 + 0]));
        _packed_3_3[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_3[_lp*2 + 2], dsel_3[_lp*2+1 + 2]));
        _packed_3_3[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_3)[0] = reinterpret_cast<int2*>(_packed_3_3)[0];
    int t_u_12 = warp_0 * 8 + 4;
    long long iidx_d_4 = ibase + (long long)(t_u_12 * rs);
    float dsel_4[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[8 + _pair]) << 16);
        (&dsel_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[8 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_4_1 = 0; e_4_1 < 4; e_4_1++) {
        float d_e_4 = 0.0f;
        if (t_u_12 < clen) {
            d_e_4 = dsel_4[e_4_1];
        }
        dsel_4[e_4_1] = d_e_4;
    }
    unsigned int _packed_3_4[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_4[_lp*2 + 0], dsel_4[_lp*2+1 + 0]));
        _packed_3_4[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_4[_lp*2 + 2], dsel_4[_lp*2+1 + 2]));
        _packed_3_4[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_4)[0] = reinterpret_cast<int2*>(_packed_3_4)[0];
    int t_u_13 = warp_0 * 8 + 5;
    long long iidx_d_5 = ibase + (long long)(t_u_13 * rs);
    float dsel_5[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[10 + _pair]) << 16);
        (&dsel_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[10 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_5_1 = 0; e_5_1 < 4; e_5_1++) {
        float d_e_5 = 0.0f;
        if (t_u_13 < clen) {
            d_e_5 = dsel_5[e_5_1];
        }
        dsel_5[e_5_1] = d_e_5;
    }
    unsigned int _packed_3_5[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_5[_lp*2 + 0], dsel_5[_lp*2+1 + 0]));
        _packed_3_5[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_5[_lp*2 + 2], dsel_5[_lp*2+1 + 2]));
        _packed_3_5[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_5)[0] = reinterpret_cast<int2*>(_packed_3_5)[0];
    int t_u_14 = warp_0 * 8 + 6;
    long long iidx_d_6 = ibase + (long long)(t_u_14 * rs);
    float dsel_6[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[12 + _pair]) << 16);
        (&dsel_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[12 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_6_1 = 0; e_6_1 < 4; e_6_1++) {
        float d_e_6 = 0.0f;
        if (t_u_14 < clen) {
            d_e_6 = dsel_6[e_6_1];
        }
        dsel_6[e_6_1] = d_e_6;
    }
    unsigned int _packed_3_6[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_6[_lp*2 + 0], dsel_6[_lp*2+1 + 0]));
        _packed_3_6[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_6[_lp*2 + 2], dsel_6[_lp*2+1 + 2]));
        _packed_3_6[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_6)[0] = reinterpret_cast<int2*>(_packed_3_6)[0];
    int t_u_15 = warp_0 * 8 + 7;
    long long iidx_d_7 = ibase + (long long)(t_u_15 * rs);
    float dsel_7[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&dsel_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dwa[14 + _pair]) << 16);
        (&dsel_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dwa[14 + _pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int e_7_1 = 0; e_7_1 < 4; e_7_1++) {
        float d_e_7 = 0.0f;
        if (t_u_15 < clen) {
            d_e_7 = dsel_7[e_7_1];
        }
        dsel_7[e_7_1] = d_e_7;
    }
    unsigned int _packed_3_7[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_7[_lp*2 + 0], dsel_7[_lp*2+1 + 0]));
        _packed_3_7[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dsel_7[_lp*2 + 2], dsel_7[_lp*2+1 + 2]));
        _packed_3_7[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(do_int + iidx_d_7)[0] = reinterpret_cast<int2*>(_packed_3_7)[0];
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
