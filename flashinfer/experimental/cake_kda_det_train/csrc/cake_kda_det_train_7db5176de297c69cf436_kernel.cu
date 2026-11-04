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

#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_kda_det_train_7db5176de297c69cf436(float* __restrict__ g, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, float* __restrict__ beta_int, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ dv2, int num_qk_heads, int num_heads, int group, __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_raw, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ qk_rstd, int num_tokens, int q_raw_row_stride, int q_raw_head_stride, int k_raw_row_stride, int k_raw_head_stride, int read_rstd, int write_rstd)
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
    unsigned int kw[16];
    unsigned int qw[16];
    float rq[8];
    float rk[8];
    float btl[8];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        int t_u = warp_0 * 8 + u;
        int t_c_0 = t_u;
        if (t_u > clast) {
            t_c_0 = clast;
        }
        int t_c = t_c_0;
        {
            float4 _v4 = *reinterpret_cast<const float4*>(g + ebase_g + (long long)(t_c * rs));
            x[u * 4 + 0] = _v4.x;
            x[u * 4 + 1] = _v4.y;
            x[u * 4 + 2] = _v4.z;
            x[u * 4 + 3] = _v4.w;
        }
        long long ridx = qrbase + (ext0 + (long long)t_c) * (long long)q_raw_row_stride;
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(q_raw + ridx);
            uint2* _vdst_1 = reinterpret_cast<uint2*>(&qw[2 * u]);
            *_vdst_1 = _vld_1;
        }
        long long ridx_0 = krbase + (ext0 + (long long)t_c) * (long long)k_raw_row_stride;
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k_raw + ridx_0);
            uint2* _vdst_2 = reinterpret_cast<uint2*>(&kw[2 * u]);
            *_vdst_2 = _vld_2;
        }
        float lr_r = 0.0f;
        if (read_rstd != 0) {
            lr_r = qk_rstd[(0 * (long long)num_tokens + (long long)(ext0 + (long long)t_c)) * (long long)num_qk_heads + (long long)qk_head];
        }
        rq[u] = lr_r;
        float lr_r_0 = 0.0f;
        if (read_rstd != 0) {
            lr_r_0 = qk_rstd[(1 * (long long)num_tokens + (long long)(ext0 + (long long)t_c)) * (long long)num_qk_heads + (long long)qk_head];
        }
        rk[u] = lr_r_0;
        btl[u] = (float)beta[bbase + (long long)(t_c * num_heads)];
    }
    int t_u_0 = warp_0 * 8;
    long long nidx_0 = qbase + (long long)(t_u_0 * qs);
    bool st_ok_0 = l2n_lead != 0 && t_u_0 < clen;
    float xv[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
        (&xv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
    }
    float y[4];
    y[0] = mul_f32_rn_noftz(xv[0], rq[0]);
    y[1] = mul_f32_rn_noftz(xv[1], rq[0]);
    y[2] = mul_f32_rn_noftz(xv[2], rq[0]);
    y[3] = mul_f32_rn_noftz(xv[3], rq[0]);
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
    qw[0] = yb[0];
    qw[1] = yb[1];
    if (st_ok_0) {
        reinterpret_cast<int2*>(qn + nidx_0)[0] = reinterpret_cast<int2*>(yb)[0];
    }
    float xv_0[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) << 16);
        (&xv_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) & 0xffff0000u);
    }
    float y_0[4];
    y_0[0] = mul_f32_rn_noftz(xv_0[0], rk[0]);
    y_0[1] = mul_f32_rn_noftz(xv_0[1], rk[0]);
    y_0[2] = mul_f32_rn_noftz(xv_0[2], rk[0]);
    y_0[3] = mul_f32_rn_noftz(xv_0[3], rk[0]);
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
    kw[0] = yb_0[0];
    kw[1] = yb_0[1];
    if (st_ok_0) {
        reinterpret_cast<int2*>(kn + nidx_0)[0] = reinterpret_cast<int2*>(yb_0)[0];
    }
    int t_u_1 = warp_0 * 8 + 1;
    long long nidx_1 = qbase + (long long)(t_u_1 * qs);
    bool st_ok_1 = l2n_lead != 0 && t_u_1 < clen;
    float xv_1[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[2 + _pair]) << 16);
        (&xv_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[2 + _pair]) & 0xffff0000u);
    }
    float y_1[4];
    y_1[0] = mul_f32_rn_noftz(xv_1[0], rq[1]);
    y_1[1] = mul_f32_rn_noftz(xv_1[1], rq[1]);
    y_1[2] = mul_f32_rn_noftz(xv_1[2], rq[1]);
    y_1[3] = mul_f32_rn_noftz(xv_1[3], rq[1]);
    unsigned int yb_1[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_1[_lp*2 + 0], y_1[_lp*2+1 + 0]));
        yb_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_1[_lp*2 + 2], y_1[_lp*2+1 + 2]));
        yb_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[2] = yb_1[0];
    qw[3] = yb_1[1];
    if (st_ok_1) {
        reinterpret_cast<int2*>(qn + nidx_1)[0] = reinterpret_cast<int2*>(yb_1)[0];
    }
    float xv_2[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[2 + _pair]) << 16);
        (&xv_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[2 + _pair]) & 0xffff0000u);
    }
    float y_2[4];
    y_2[0] = mul_f32_rn_noftz(xv_2[0], rk[1]);
    y_2[1] = mul_f32_rn_noftz(xv_2[1], rk[1]);
    y_2[2] = mul_f32_rn_noftz(xv_2[2], rk[1]);
    y_2[3] = mul_f32_rn_noftz(xv_2[3], rk[1]);
    unsigned int yb_2[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_2[_lp*2 + 0], y_2[_lp*2+1 + 0]));
        yb_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_2[_lp*2 + 2], y_2[_lp*2+1 + 2]));
        yb_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[2] = yb_2[0];
    kw[3] = yb_2[1];
    if (st_ok_1) {
        reinterpret_cast<int2*>(kn + nidx_1)[0] = reinterpret_cast<int2*>(yb_2)[0];
    }
    int t_u_2 = warp_0 * 8 + 2;
    long long nidx_2 = qbase + (long long)(t_u_2 * qs);
    bool st_ok_2 = l2n_lead != 0 && t_u_2 < clen;
    float xv_3[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[4 + _pair]) << 16);
        (&xv_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[4 + _pair]) & 0xffff0000u);
    }
    float y_3[4];
    y_3[0] = mul_f32_rn_noftz(xv_3[0], rq[2]);
    y_3[1] = mul_f32_rn_noftz(xv_3[1], rq[2]);
    y_3[2] = mul_f32_rn_noftz(xv_3[2], rq[2]);
    y_3[3] = mul_f32_rn_noftz(xv_3[3], rq[2]);
    unsigned int yb_3[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_3[_lp*2 + 0], y_3[_lp*2+1 + 0]));
        yb_3[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_3[_lp*2 + 2], y_3[_lp*2+1 + 2]));
        yb_3[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[4] = yb_3[0];
    qw[5] = yb_3[1];
    if (st_ok_2) {
        reinterpret_cast<int2*>(qn + nidx_2)[0] = reinterpret_cast<int2*>(yb_3)[0];
    }
    float xv_4[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[4 + _pair]) << 16);
        (&xv_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[4 + _pair]) & 0xffff0000u);
    }
    float y_4[4];
    y_4[0] = mul_f32_rn_noftz(xv_4[0], rk[2]);
    y_4[1] = mul_f32_rn_noftz(xv_4[1], rk[2]);
    y_4[2] = mul_f32_rn_noftz(xv_4[2], rk[2]);
    y_4[3] = mul_f32_rn_noftz(xv_4[3], rk[2]);
    unsigned int yb_4[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_4[_lp*2 + 0], y_4[_lp*2+1 + 0]));
        yb_4[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_4[_lp*2 + 2], y_4[_lp*2+1 + 2]));
        yb_4[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[4] = yb_4[0];
    kw[5] = yb_4[1];
    if (st_ok_2) {
        reinterpret_cast<int2*>(kn + nidx_2)[0] = reinterpret_cast<int2*>(yb_4)[0];
    }
    int t_u_3 = warp_0 * 8 + 3;
    long long nidx_3 = qbase + (long long)(t_u_3 * qs);
    bool st_ok_3 = l2n_lead != 0 && t_u_3 < clen;
    float xv_5[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[6 + _pair]) << 16);
        (&xv_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[6 + _pair]) & 0xffff0000u);
    }
    float y_5[4];
    y_5[0] = mul_f32_rn_noftz(xv_5[0], rq[3]);
    y_5[1] = mul_f32_rn_noftz(xv_5[1], rq[3]);
    y_5[2] = mul_f32_rn_noftz(xv_5[2], rq[3]);
    y_5[3] = mul_f32_rn_noftz(xv_5[3], rq[3]);
    unsigned int yb_5[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_5[_lp*2 + 0], y_5[_lp*2+1 + 0]));
        yb_5[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_5[_lp*2 + 2], y_5[_lp*2+1 + 2]));
        yb_5[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[6] = yb_5[0];
    qw[7] = yb_5[1];
    if (st_ok_3) {
        reinterpret_cast<int2*>(qn + nidx_3)[0] = reinterpret_cast<int2*>(yb_5)[0];
    }
    float xv_6[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[6 + _pair]) << 16);
        (&xv_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[6 + _pair]) & 0xffff0000u);
    }
    float y_6[4];
    y_6[0] = mul_f32_rn_noftz(xv_6[0], rk[3]);
    y_6[1] = mul_f32_rn_noftz(xv_6[1], rk[3]);
    y_6[2] = mul_f32_rn_noftz(xv_6[2], rk[3]);
    y_6[3] = mul_f32_rn_noftz(xv_6[3], rk[3]);
    unsigned int yb_6[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_6[_lp*2 + 0], y_6[_lp*2+1 + 0]));
        yb_6[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_6[_lp*2 + 2], y_6[_lp*2+1 + 2]));
        yb_6[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[6] = yb_6[0];
    kw[7] = yb_6[1];
    if (st_ok_3) {
        reinterpret_cast<int2*>(kn + nidx_3)[0] = reinterpret_cast<int2*>(yb_6)[0];
    }
    int t_u_4 = warp_0 * 8 + 4;
    long long nidx_4 = qbase + (long long)(t_u_4 * qs);
    bool st_ok_4 = l2n_lead != 0 && t_u_4 < clen;
    float xv_7[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
        (&xv_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
    }
    float y_7[4];
    y_7[0] = mul_f32_rn_noftz(xv_7[0], rq[4]);
    y_7[1] = mul_f32_rn_noftz(xv_7[1], rq[4]);
    y_7[2] = mul_f32_rn_noftz(xv_7[2], rq[4]);
    y_7[3] = mul_f32_rn_noftz(xv_7[3], rq[4]);
    unsigned int yb_7[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_7[_lp*2 + 0], y_7[_lp*2+1 + 0]));
        yb_7[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_7[_lp*2 + 2], y_7[_lp*2+1 + 2]));
        yb_7[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[8] = yb_7[0];
    qw[9] = yb_7[1];
    if (st_ok_4) {
        reinterpret_cast<int2*>(qn + nidx_4)[0] = reinterpret_cast<int2*>(yb_7)[0];
    }
    float xv_8[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[8 + _pair]) << 16);
        (&xv_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[8 + _pair]) & 0xffff0000u);
    }
    float y_8[4];
    y_8[0] = mul_f32_rn_noftz(xv_8[0], rk[4]);
    y_8[1] = mul_f32_rn_noftz(xv_8[1], rk[4]);
    y_8[2] = mul_f32_rn_noftz(xv_8[2], rk[4]);
    y_8[3] = mul_f32_rn_noftz(xv_8[3], rk[4]);
    unsigned int yb_8[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_8[_lp*2 + 0], y_8[_lp*2+1 + 0]));
        yb_8[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_8[_lp*2 + 2], y_8[_lp*2+1 + 2]));
        yb_8[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[8] = yb_8[0];
    kw[9] = yb_8[1];
    if (st_ok_4) {
        reinterpret_cast<int2*>(kn + nidx_4)[0] = reinterpret_cast<int2*>(yb_8)[0];
    }
    int t_u_5 = warp_0 * 8 + 5;
    long long nidx_5 = qbase + (long long)(t_u_5 * qs);
    bool st_ok_5 = l2n_lead != 0 && t_u_5 < clen;
    float xv_9[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[10 + _pair]) << 16);
        (&xv_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[10 + _pair]) & 0xffff0000u);
    }
    float y_9[4];
    y_9[0] = mul_f32_rn_noftz(xv_9[0], rq[5]);
    y_9[1] = mul_f32_rn_noftz(xv_9[1], rq[5]);
    y_9[2] = mul_f32_rn_noftz(xv_9[2], rq[5]);
    y_9[3] = mul_f32_rn_noftz(xv_9[3], rq[5]);
    unsigned int yb_9[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_9[_lp*2 + 0], y_9[_lp*2+1 + 0]));
        yb_9[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_9[_lp*2 + 2], y_9[_lp*2+1 + 2]));
        yb_9[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[10] = yb_9[0];
    qw[11] = yb_9[1];
    if (st_ok_5) {
        reinterpret_cast<int2*>(qn + nidx_5)[0] = reinterpret_cast<int2*>(yb_9)[0];
    }
    float xv_10[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[10 + _pair]) << 16);
        (&xv_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[10 + _pair]) & 0xffff0000u);
    }
    float y_10[4];
    y_10[0] = mul_f32_rn_noftz(xv_10[0], rk[5]);
    y_10[1] = mul_f32_rn_noftz(xv_10[1], rk[5]);
    y_10[2] = mul_f32_rn_noftz(xv_10[2], rk[5]);
    y_10[3] = mul_f32_rn_noftz(xv_10[3], rk[5]);
    unsigned int yb_10[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_10[_lp*2 + 0], y_10[_lp*2+1 + 0]));
        yb_10[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_10[_lp*2 + 2], y_10[_lp*2+1 + 2]));
        yb_10[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[10] = yb_10[0];
    kw[11] = yb_10[1];
    if (st_ok_5) {
        reinterpret_cast<int2*>(kn + nidx_5)[0] = reinterpret_cast<int2*>(yb_10)[0];
    }
    int t_u_6 = warp_0 * 8 + 6;
    long long nidx_6 = qbase + (long long)(t_u_6 * qs);
    bool st_ok_6 = l2n_lead != 0 && t_u_6 < clen;
    float xv_11[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[12 + _pair]) << 16);
        (&xv_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[12 + _pair]) & 0xffff0000u);
    }
    float y_11[4];
    y_11[0] = mul_f32_rn_noftz(xv_11[0], rq[6]);
    y_11[1] = mul_f32_rn_noftz(xv_11[1], rq[6]);
    y_11[2] = mul_f32_rn_noftz(xv_11[2], rq[6]);
    y_11[3] = mul_f32_rn_noftz(xv_11[3], rq[6]);
    unsigned int yb_11[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_11[_lp*2 + 0], y_11[_lp*2+1 + 0]));
        yb_11[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_11[_lp*2 + 2], y_11[_lp*2+1 + 2]));
        yb_11[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[12] = yb_11[0];
    qw[13] = yb_11[1];
    if (st_ok_6) {
        reinterpret_cast<int2*>(qn + nidx_6)[0] = reinterpret_cast<int2*>(yb_11)[0];
    }
    float xv_12[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[12 + _pair]) << 16);
        (&xv_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[12 + _pair]) & 0xffff0000u);
    }
    float y_12[4];
    y_12[0] = mul_f32_rn_noftz(xv_12[0], rk[6]);
    y_12[1] = mul_f32_rn_noftz(xv_12[1], rk[6]);
    y_12[2] = mul_f32_rn_noftz(xv_12[2], rk[6]);
    y_12[3] = mul_f32_rn_noftz(xv_12[3], rk[6]);
    unsigned int yb_12[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_12[_lp*2 + 0], y_12[_lp*2+1 + 0]));
        yb_12[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_12[_lp*2 + 2], y_12[_lp*2+1 + 2]));
        yb_12[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[12] = yb_12[0];
    kw[13] = yb_12[1];
    if (st_ok_6) {
        reinterpret_cast<int2*>(kn + nidx_6)[0] = reinterpret_cast<int2*>(yb_12)[0];
    }
    int t_u_7 = warp_0 * 8 + 7;
    long long nidx_7 = qbase + (long long)(t_u_7 * qs);
    bool st_ok_7 = l2n_lead != 0 && t_u_7 < clen;
    float xv_13[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[14 + _pair]) << 16);
        (&xv_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[14 + _pair]) & 0xffff0000u);
    }
    float y_13[4];
    y_13[0] = mul_f32_rn_noftz(xv_13[0], rq[7]);
    y_13[1] = mul_f32_rn_noftz(xv_13[1], rq[7]);
    y_13[2] = mul_f32_rn_noftz(xv_13[2], rq[7]);
    y_13[3] = mul_f32_rn_noftz(xv_13[3], rq[7]);
    unsigned int yb_13[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_13[_lp*2 + 0], y_13[_lp*2+1 + 0]));
        yb_13[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_13[_lp*2 + 2], y_13[_lp*2+1 + 2]));
        yb_13[_lp + 1] = *(uint32_t*)&_bf2;
    }
    qw[14] = yb_13[0];
    qw[15] = yb_13[1];
    if (st_ok_7) {
        reinterpret_cast<int2*>(qn + nidx_7)[0] = reinterpret_cast<int2*>(yb_13)[0];
    }
    float xv_14[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&xv_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[14 + _pair]) << 16);
        (&xv_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[14 + _pair]) & 0xffff0000u);
    }
    float y_14[4];
    y_14[0] = mul_f32_rn_noftz(xv_14[0], rk[7]);
    y_14[1] = mul_f32_rn_noftz(xv_14[1], rk[7]);
    y_14[2] = mul_f32_rn_noftz(xv_14[2], rk[7]);
    y_14[3] = mul_f32_rn_noftz(xv_14[3], rk[7]);
    unsigned int yb_14[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_14[_lp*2 + 0], y_14[_lp*2+1 + 0]));
        yb_14[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_14[_lp*2 + 2], y_14[_lp*2+1 + 2]));
        yb_14[_lp + 1] = *(uint32_t*)&_bf2;
    }
    kw[14] = yb_14[0];
    kw[15] = yb_14[1];
    if (st_ok_7) {
        reinterpret_cast<int2*>(kn + nidx_7)[0] = reinterpret_cast<int2*>(yb_14)[0];
    }
    float run[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        run[e] = 0.0f;
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        int t_u_8 = warp_0 * 8 + u_1;
        float gq[4];
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            float gate_u = 0.0f;
            if (t_u_8 < clen) {
                gate_u = x[u_1 * 4 + e_1];
            }
            gq[e_1] = gate_u;
            run[e_1] = run[e_1] + gate_u;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u_8 * 128 + d0) * 4)), "f"(gq[0]), "f"(gq[1]), "f"(gq[2]), "f"(gq[3]) : "memory");
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
    int t_u_8_1 = warp_0 * 8;
    long long iidx_0 = ibase + (long long)(t_u_8_1 * rs);
    float kv_0[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) << 16);
        (&kv_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) & 0xffff0000u);
    }
    float gq2_0[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_0[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_8_1 * 128 + d0) * 4)));
    float bt_0 = 0.0f;
    if (t_u_8_1 < clen) {
        bt_0 = btl[0];
    }
    float gkt_0[4];
    #pragma unroll
    for (int e_0 = 0; e_0 < 4; e_0++) {
        acc[e_0] = acc[e_0] + gq2_0[e_0];
        gkt_0[e_0] = (off[e_0] + acc[e_0]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_0[0 + 0], gkt_0[0 + 1], gkt_0[0 + 2], gkt_0[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_0) + 0) = _v4;
    }
    float eg_0[4];
    #pragma unroll
    for (int e_1_1 = 0; e_1_1 < 4; e_1_1++) {
        float _exp2_3 = approx_exp2(gkt_0[e_1_1]);
        eg_0[e_1_1] = _exp2_3;
    }
    float kb_0[4];
    #pragma unroll
    for (int e_2_1 = 0; e_2_1 < 4; e_2_1++) {
        float k_e_0 = 0.0f;
        if (t_u_8_1 < clen) {
            k_e_0 = kv_0[e_2_1];
        }
        kb_0[e_2_1] = (float)(__nv_bfloat16)(k_e_0 * bt_0) * eg_0[e_2_1];
    }
    unsigned int _packed_0_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_0[_lp*2 + 0], kb_0[_lp*2+1 + 0]));
        _packed_0_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_0[_lp*2 + 2], kb_0[_lp*2+1 + 2]));
        _packed_0_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_0)[0] = reinterpret_cast<int2*>(_packed_0_0)[0];
    float kg_0[4];
    #pragma unroll
    for (int e_3_1 = 0; e_3_1 < 4; e_3_1++) {
        float k_e_1 = 0.0f;
        if (t_u_8_1 < clen) {
            k_e_1 = kv_0[e_3_1];
        }
        float _exp2_4 = approx_exp2(gk_n[e_3_1] - gkt_0[e_3_1]);
        kg_0[e_3_1] = k_e_1 * _exp2_4;
    }
    unsigned int _packed_1_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_0[_lp*2 + 0], kg_0[_lp*2+1 + 0]));
        _packed_1_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_0[_lp*2 + 2], kg_0[_lp*2+1 + 2]));
        _packed_1_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_0)[0] = reinterpret_cast<int2*>(_packed_1_0)[0];
    float qv_0[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
        (&qv_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
    }
    float qg_0[4];
    #pragma unroll
    for (int e_4_1 = 0; e_4_1 < 4; e_4_1++) {
        float q_e_0 = 0.0f;
        if (t_u_8_1 < clen) {
            q_e_0 = qv_0[e_4_1];
        }
        qg_0[e_4_1] = q_e_0 * eg_0[e_4_1];
    }
    unsigned int _packed_2_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_0[_lp*2 + 0], qg_0[_lp*2+1 + 0]));
        _packed_2_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_0[_lp*2 + 2], qg_0[_lp*2+1 + 2]));
        _packed_2_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_0)[0] = reinterpret_cast<int2*>(_packed_2_0)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_8_1) * (long long)num_heads + (long long)head] = bt_0;
    }
    int t_u_9 = warp_0 * 8 + 1;
    long long iidx_1 = ibase + (long long)(t_u_9 * rs);
    float kv_1[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[2 + _pair]) << 16);
        (&kv_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[2 + _pair]) & 0xffff0000u);
    }
    float gq2_1[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_1[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_9 * 128 + d0) * 4)));
    float bt_1 = 0.0f;
    if (t_u_9 < clen) {
        bt_1 = btl[1];
    }
    float gkt_1[4];
    #pragma unroll
    for (int e_5_1 = 0; e_5_1 < 4; e_5_1++) {
        acc[e_5_1] = acc[e_5_1] + gq2_1[e_5_1];
        gkt_1[e_5_1] = (off[e_5_1] + acc[e_5_1]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_1[0 + 0], gkt_1[0 + 1], gkt_1[0 + 2], gkt_1[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_1) + 0) = _v4;
    }
    float eg_1[4];
    #pragma unroll
    for (int e_6_1 = 0; e_6_1 < 4; e_6_1++) {
        float _exp2_5 = approx_exp2(gkt_1[e_6_1]);
        eg_1[e_6_1] = _exp2_5;
    }
    float kb_1[4];
    #pragma unroll
    for (int e_7 = 0; e_7 < 4; e_7++) {
        float k_e_2 = 0.0f;
        if (t_u_9 < clen) {
            k_e_2 = kv_1[e_7];
        }
        kb_1[e_7] = (float)(__nv_bfloat16)(k_e_2 * bt_1) * eg_1[e_7];
    }
    unsigned int _packed_0_1[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_1[_lp*2 + 0], kb_1[_lp*2+1 + 0]));
        _packed_0_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_1[_lp*2 + 2], kb_1[_lp*2+1 + 2]));
        _packed_0_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_1)[0] = reinterpret_cast<int2*>(_packed_0_1)[0];
    float kg_1[4];
    #pragma unroll
    for (int e_8 = 0; e_8 < 4; e_8++) {
        float k_e_3 = 0.0f;
        if (t_u_9 < clen) {
            k_e_3 = kv_1[e_8];
        }
        float _exp2_6 = approx_exp2(gk_n[e_8] - gkt_1[e_8]);
        kg_1[e_8] = k_e_3 * _exp2_6;
    }
    unsigned int _packed_1_1[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_1[_lp*2 + 0], kg_1[_lp*2+1 + 0]));
        _packed_1_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_1[_lp*2 + 2], kg_1[_lp*2+1 + 2]));
        _packed_1_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_1)[0] = reinterpret_cast<int2*>(_packed_1_1)[0];
    float qv_1[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[2 + _pair]) << 16);
        (&qv_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[2 + _pair]) & 0xffff0000u);
    }
    float qg_1[4];
    #pragma unroll
    for (int e_9 = 0; e_9 < 4; e_9++) {
        float q_e_1 = 0.0f;
        if (t_u_9 < clen) {
            q_e_1 = qv_1[e_9];
        }
        qg_1[e_9] = q_e_1 * eg_1[e_9];
    }
    unsigned int _packed_2_1[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_1[_lp*2 + 0], qg_1[_lp*2+1 + 0]));
        _packed_2_1[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_1[_lp*2 + 2], qg_1[_lp*2+1 + 2]));
        _packed_2_1[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_1)[0] = reinterpret_cast<int2*>(_packed_2_1)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_9) * (long long)num_heads + (long long)head] = bt_1;
    }
    int t_u_10 = warp_0 * 8 + 2;
    long long iidx_2 = ibase + (long long)(t_u_10 * rs);
    float kv_2[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[4 + _pair]) << 16);
        (&kv_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[4 + _pair]) & 0xffff0000u);
    }
    float gq2_2[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_2[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_10 * 128 + d0) * 4)));
    float bt_2 = 0.0f;
    if (t_u_10 < clen) {
        bt_2 = btl[2];
    }
    float gkt_2[4];
    #pragma unroll
    for (int e_10 = 0; e_10 < 4; e_10++) {
        acc[e_10] = acc[e_10] + gq2_2[e_10];
        gkt_2[e_10] = (off[e_10] + acc[e_10]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_2[0 + 0], gkt_2[0 + 1], gkt_2[0 + 2], gkt_2[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_2) + 0) = _v4;
    }
    float eg_2[4];
    #pragma unroll
    for (int e_11 = 0; e_11 < 4; e_11++) {
        float _exp2_7 = approx_exp2(gkt_2[e_11]);
        eg_2[e_11] = _exp2_7;
    }
    float kb_2[4];
    #pragma unroll
    for (int e_12 = 0; e_12 < 4; e_12++) {
        float k_e_4 = 0.0f;
        if (t_u_10 < clen) {
            k_e_4 = kv_2[e_12];
        }
        kb_2[e_12] = (float)(__nv_bfloat16)(k_e_4 * bt_2) * eg_2[e_12];
    }
    unsigned int _packed_0_2[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_2[_lp*2 + 0], kb_2[_lp*2+1 + 0]));
        _packed_0_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_2[_lp*2 + 2], kb_2[_lp*2+1 + 2]));
        _packed_0_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_2)[0] = reinterpret_cast<int2*>(_packed_0_2)[0];
    float kg_2[4];
    #pragma unroll
    for (int e_13 = 0; e_13 < 4; e_13++) {
        float k_e_5 = 0.0f;
        if (t_u_10 < clen) {
            k_e_5 = kv_2[e_13];
        }
        float _exp2_8 = approx_exp2(gk_n[e_13] - gkt_2[e_13]);
        kg_2[e_13] = k_e_5 * _exp2_8;
    }
    unsigned int _packed_1_2[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_2[_lp*2 + 0], kg_2[_lp*2+1 + 0]));
        _packed_1_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_2[_lp*2 + 2], kg_2[_lp*2+1 + 2]));
        _packed_1_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_2)[0] = reinterpret_cast<int2*>(_packed_1_2)[0];
    float qv_2[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[4 + _pair]) << 16);
        (&qv_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[4 + _pair]) & 0xffff0000u);
    }
    float qg_2[4];
    #pragma unroll
    for (int e_14 = 0; e_14 < 4; e_14++) {
        float q_e_2 = 0.0f;
        if (t_u_10 < clen) {
            q_e_2 = qv_2[e_14];
        }
        qg_2[e_14] = q_e_2 * eg_2[e_14];
    }
    unsigned int _packed_2_2[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_2[_lp*2 + 0], qg_2[_lp*2+1 + 0]));
        _packed_2_2[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_2[_lp*2 + 2], qg_2[_lp*2+1 + 2]));
        _packed_2_2[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_2)[0] = reinterpret_cast<int2*>(_packed_2_2)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_10) * (long long)num_heads + (long long)head] = bt_2;
    }
    int t_u_11 = warp_0 * 8 + 3;
    long long iidx_3 = ibase + (long long)(t_u_11 * rs);
    float kv_3[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[6 + _pair]) << 16);
        (&kv_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[6 + _pair]) & 0xffff0000u);
    }
    float gq2_3[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_3[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_11 * 128 + d0) * 4)));
    float bt_3 = 0.0f;
    if (t_u_11 < clen) {
        bt_3 = btl[3];
    }
    float gkt_3[4];
    #pragma unroll
    for (int e_15 = 0; e_15 < 4; e_15++) {
        acc[e_15] = acc[e_15] + gq2_3[e_15];
        gkt_3[e_15] = (off[e_15] + acc[e_15]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_3[0 + 0], gkt_3[0 + 1], gkt_3[0 + 2], gkt_3[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_3) + 0) = _v4;
    }
    float eg_3[4];
    #pragma unroll
    for (int e_16 = 0; e_16 < 4; e_16++) {
        float _exp2_9 = approx_exp2(gkt_3[e_16]);
        eg_3[e_16] = _exp2_9;
    }
    float kb_3[4];
    #pragma unroll
    for (int e_17 = 0; e_17 < 4; e_17++) {
        float k_e_6 = 0.0f;
        if (t_u_11 < clen) {
            k_e_6 = kv_3[e_17];
        }
        kb_3[e_17] = (float)(__nv_bfloat16)(k_e_6 * bt_3) * eg_3[e_17];
    }
    unsigned int _packed_0_3[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_3[_lp*2 + 0], kb_3[_lp*2+1 + 0]));
        _packed_0_3[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_3[_lp*2 + 2], kb_3[_lp*2+1 + 2]));
        _packed_0_3[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_3)[0] = reinterpret_cast<int2*>(_packed_0_3)[0];
    float kg_3[4];
    #pragma unroll
    for (int e_18 = 0; e_18 < 4; e_18++) {
        float k_e_7 = 0.0f;
        if (t_u_11 < clen) {
            k_e_7 = kv_3[e_18];
        }
        float _exp2_10 = approx_exp2(gk_n[e_18] - gkt_3[e_18]);
        kg_3[e_18] = k_e_7 * _exp2_10;
    }
    unsigned int _packed_1_3[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_3[_lp*2 + 0], kg_3[_lp*2+1 + 0]));
        _packed_1_3[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_3[_lp*2 + 2], kg_3[_lp*2+1 + 2]));
        _packed_1_3[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_3)[0] = reinterpret_cast<int2*>(_packed_1_3)[0];
    float qv_3[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[6 + _pair]) << 16);
        (&qv_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[6 + _pair]) & 0xffff0000u);
    }
    float qg_3[4];
    #pragma unroll
    for (int e_19 = 0; e_19 < 4; e_19++) {
        float q_e_3 = 0.0f;
        if (t_u_11 < clen) {
            q_e_3 = qv_3[e_19];
        }
        qg_3[e_19] = q_e_3 * eg_3[e_19];
    }
    unsigned int _packed_2_3[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_3[_lp*2 + 0], qg_3[_lp*2+1 + 0]));
        _packed_2_3[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_3[_lp*2 + 2], qg_3[_lp*2+1 + 2]));
        _packed_2_3[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_3)[0] = reinterpret_cast<int2*>(_packed_2_3)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_11) * (long long)num_heads + (long long)head] = bt_3;
    }
    int t_u_12 = warp_0 * 8 + 4;
    long long iidx_4 = ibase + (long long)(t_u_12 * rs);
    float kv_4[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[8 + _pair]) << 16);
        (&kv_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[8 + _pair]) & 0xffff0000u);
    }
    float gq2_4[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_4[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_12 * 128 + d0) * 4)));
    float bt_4 = 0.0f;
    if (t_u_12 < clen) {
        bt_4 = btl[4];
    }
    float gkt_4[4];
    #pragma unroll
    for (int e_20 = 0; e_20 < 4; e_20++) {
        acc[e_20] = acc[e_20] + gq2_4[e_20];
        gkt_4[e_20] = (off[e_20] + acc[e_20]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_4[0 + 0], gkt_4[0 + 1], gkt_4[0 + 2], gkt_4[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_4) + 0) = _v4;
    }
    float eg_4[4];
    #pragma unroll
    for (int e_21 = 0; e_21 < 4; e_21++) {
        float _exp2_11 = approx_exp2(gkt_4[e_21]);
        eg_4[e_21] = _exp2_11;
    }
    float kb_4[4];
    #pragma unroll
    for (int e_22 = 0; e_22 < 4; e_22++) {
        float k_e_8 = 0.0f;
        if (t_u_12 < clen) {
            k_e_8 = kv_4[e_22];
        }
        kb_4[e_22] = (float)(__nv_bfloat16)(k_e_8 * bt_4) * eg_4[e_22];
    }
    unsigned int _packed_0_4[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_4[_lp*2 + 0], kb_4[_lp*2+1 + 0]));
        _packed_0_4[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_4[_lp*2 + 2], kb_4[_lp*2+1 + 2]));
        _packed_0_4[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_4)[0] = reinterpret_cast<int2*>(_packed_0_4)[0];
    float kg_4[4];
    #pragma unroll
    for (int e_23 = 0; e_23 < 4; e_23++) {
        float k_e_9 = 0.0f;
        if (t_u_12 < clen) {
            k_e_9 = kv_4[e_23];
        }
        float _exp2_12 = approx_exp2(gk_n[e_23] - gkt_4[e_23]);
        kg_4[e_23] = k_e_9 * _exp2_12;
    }
    unsigned int _packed_1_4[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_4[_lp*2 + 0], kg_4[_lp*2+1 + 0]));
        _packed_1_4[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_4[_lp*2 + 2], kg_4[_lp*2+1 + 2]));
        _packed_1_4[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_4)[0] = reinterpret_cast<int2*>(_packed_1_4)[0];
    float qv_4[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
        (&qv_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
    }
    float qg_4[4];
    #pragma unroll
    for (int e_24 = 0; e_24 < 4; e_24++) {
        float q_e_4 = 0.0f;
        if (t_u_12 < clen) {
            q_e_4 = qv_4[e_24];
        }
        qg_4[e_24] = q_e_4 * eg_4[e_24];
    }
    unsigned int _packed_2_4[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_4[_lp*2 + 0], qg_4[_lp*2+1 + 0]));
        _packed_2_4[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_4[_lp*2 + 2], qg_4[_lp*2+1 + 2]));
        _packed_2_4[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_4)[0] = reinterpret_cast<int2*>(_packed_2_4)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_12) * (long long)num_heads + (long long)head] = bt_4;
    }
    int t_u_13 = warp_0 * 8 + 5;
    long long iidx_5 = ibase + (long long)(t_u_13 * rs);
    float kv_5[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[10 + _pair]) << 16);
        (&kv_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[10 + _pair]) & 0xffff0000u);
    }
    float gq2_5[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_5[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_13 * 128 + d0) * 4)));
    float bt_5 = 0.0f;
    if (t_u_13 < clen) {
        bt_5 = btl[5];
    }
    float gkt_5[4];
    #pragma unroll
    for (int e_25 = 0; e_25 < 4; e_25++) {
        acc[e_25] = acc[e_25] + gq2_5[e_25];
        gkt_5[e_25] = (off[e_25] + acc[e_25]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_5[0 + 0], gkt_5[0 + 1], gkt_5[0 + 2], gkt_5[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_5) + 0) = _v4;
    }
    float eg_5[4];
    #pragma unroll
    for (int e_26 = 0; e_26 < 4; e_26++) {
        float _exp2_13 = approx_exp2(gkt_5[e_26]);
        eg_5[e_26] = _exp2_13;
    }
    float kb_5[4];
    #pragma unroll
    for (int e_27 = 0; e_27 < 4; e_27++) {
        float k_e_10 = 0.0f;
        if (t_u_13 < clen) {
            k_e_10 = kv_5[e_27];
        }
        kb_5[e_27] = (float)(__nv_bfloat16)(k_e_10 * bt_5) * eg_5[e_27];
    }
    unsigned int _packed_0_5[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_5[_lp*2 + 0], kb_5[_lp*2+1 + 0]));
        _packed_0_5[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_5[_lp*2 + 2], kb_5[_lp*2+1 + 2]));
        _packed_0_5[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_5)[0] = reinterpret_cast<int2*>(_packed_0_5)[0];
    float kg_5[4];
    #pragma unroll
    for (int e_28 = 0; e_28 < 4; e_28++) {
        float k_e_11 = 0.0f;
        if (t_u_13 < clen) {
            k_e_11 = kv_5[e_28];
        }
        float _exp2_14 = approx_exp2(gk_n[e_28] - gkt_5[e_28]);
        kg_5[e_28] = k_e_11 * _exp2_14;
    }
    unsigned int _packed_1_5[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_5[_lp*2 + 0], kg_5[_lp*2+1 + 0]));
        _packed_1_5[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_5[_lp*2 + 2], kg_5[_lp*2+1 + 2]));
        _packed_1_5[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_5)[0] = reinterpret_cast<int2*>(_packed_1_5)[0];
    float qv_5[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[10 + _pair]) << 16);
        (&qv_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[10 + _pair]) & 0xffff0000u);
    }
    float qg_5[4];
    #pragma unroll
    for (int e_29 = 0; e_29 < 4; e_29++) {
        float q_e_5 = 0.0f;
        if (t_u_13 < clen) {
            q_e_5 = qv_5[e_29];
        }
        qg_5[e_29] = q_e_5 * eg_5[e_29];
    }
    unsigned int _packed_2_5[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_5[_lp*2 + 0], qg_5[_lp*2+1 + 0]));
        _packed_2_5[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_5[_lp*2 + 2], qg_5[_lp*2+1 + 2]));
        _packed_2_5[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_5)[0] = reinterpret_cast<int2*>(_packed_2_5)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_13) * (long long)num_heads + (long long)head] = bt_5;
    }
    int t_u_14 = warp_0 * 8 + 6;
    long long iidx_6 = ibase + (long long)(t_u_14 * rs);
    float kv_6[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[12 + _pair]) << 16);
        (&kv_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[12 + _pair]) & 0xffff0000u);
    }
    float gq2_6[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_6[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_14 * 128 + d0) * 4)));
    float bt_6 = 0.0f;
    if (t_u_14 < clen) {
        bt_6 = btl[6];
    }
    float gkt_6[4];
    #pragma unroll
    for (int e_30 = 0; e_30 < 4; e_30++) {
        acc[e_30] = acc[e_30] + gq2_6[e_30];
        gkt_6[e_30] = (off[e_30] + acc[e_30]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_6[0 + 0], gkt_6[0 + 1], gkt_6[0 + 2], gkt_6[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_6) + 0) = _v4;
    }
    float eg_6[4];
    #pragma unroll
    for (int e_31 = 0; e_31 < 4; e_31++) {
        float _exp2_15 = approx_exp2(gkt_6[e_31]);
        eg_6[e_31] = _exp2_15;
    }
    float kb_6[4];
    #pragma unroll
    for (int e_32 = 0; e_32 < 4; e_32++) {
        float k_e_12 = 0.0f;
        if (t_u_14 < clen) {
            k_e_12 = kv_6[e_32];
        }
        kb_6[e_32] = (float)(__nv_bfloat16)(k_e_12 * bt_6) * eg_6[e_32];
    }
    unsigned int _packed_0_6[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_6[_lp*2 + 0], kb_6[_lp*2+1 + 0]));
        _packed_0_6[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_6[_lp*2 + 2], kb_6[_lp*2+1 + 2]));
        _packed_0_6[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_6)[0] = reinterpret_cast<int2*>(_packed_0_6)[0];
    float kg_6[4];
    #pragma unroll
    for (int e_33 = 0; e_33 < 4; e_33++) {
        float k_e_13 = 0.0f;
        if (t_u_14 < clen) {
            k_e_13 = kv_6[e_33];
        }
        float _exp2_16 = approx_exp2(gk_n[e_33] - gkt_6[e_33]);
        kg_6[e_33] = k_e_13 * _exp2_16;
    }
    unsigned int _packed_1_6[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_6[_lp*2 + 0], kg_6[_lp*2+1 + 0]));
        _packed_1_6[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_6[_lp*2 + 2], kg_6[_lp*2+1 + 2]));
        _packed_1_6[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_6)[0] = reinterpret_cast<int2*>(_packed_1_6)[0];
    float qv_6[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[12 + _pair]) << 16);
        (&qv_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[12 + _pair]) & 0xffff0000u);
    }
    float qg_6[4];
    #pragma unroll
    for (int e_34 = 0; e_34 < 4; e_34++) {
        float q_e_6 = 0.0f;
        if (t_u_14 < clen) {
            q_e_6 = qv_6[e_34];
        }
        qg_6[e_34] = q_e_6 * eg_6[e_34];
    }
    unsigned int _packed_2_6[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_6[_lp*2 + 0], qg_6[_lp*2+1 + 0]));
        _packed_2_6[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_6[_lp*2 + 2], qg_6[_lp*2+1 + 2]));
        _packed_2_6[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_6)[0] = reinterpret_cast<int2*>(_packed_2_6)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_14) * (long long)num_heads + (long long)head] = bt_6;
    }
    int t_u_15 = warp_0 * 8 + 7;
    long long iidx_7 = ibase + (long long)(t_u_15 * rs);
    float kv_7[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&kv_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[14 + _pair]) << 16);
        (&kv_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[14 + _pair]) & 0xffff0000u);
    }
    float gq2_7[4];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&gq2_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq2_7[(0) + 3]))
        : "r"(graw_addr + (unsigned int)((t_u_15 * 128 + d0) * 4)));
    float bt_7 = 0.0f;
    if (t_u_15 < clen) {
        bt_7 = btl[7];
    }
    float gkt_7[4];
    #pragma unroll
    for (int e_35 = 0; e_35 < 4; e_35++) {
        acc[e_35] = acc[e_35] + gq2_7[e_35];
        gkt_7[e_35] = (off[e_35] + acc[e_35]) * 1.442695f;
    }
    {
        float4 _v4 = make_float4(gkt_7[0 + 0], gkt_7[0 + 1], gkt_7[0 + 2], gkt_7[0 + 3]);
        *reinterpret_cast<float4*>((gk_out + iidx_7) + 0) = _v4;
    }
    float eg_7[4];
    #pragma unroll
    for (int e_36 = 0; e_36 < 4; e_36++) {
        float _exp2_17 = approx_exp2(gkt_7[e_36]);
        eg_7[e_36] = _exp2_17;
    }
    float kb_7[4];
    #pragma unroll
    for (int e_37 = 0; e_37 < 4; e_37++) {
        float k_e_14 = 0.0f;
        if (t_u_15 < clen) {
            k_e_14 = kv_7[e_37];
        }
        kb_7[e_37] = (float)(__nv_bfloat16)(k_e_14 * bt_7) * eg_7[e_37];
    }
    unsigned int _packed_0_7[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_7[_lp*2 + 0], kb_7[_lp*2+1 + 0]));
        _packed_0_7[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_7[_lp*2 + 2], kb_7[_lp*2+1 + 2]));
        _packed_0_7[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kb_out + iidx_7)[0] = reinterpret_cast<int2*>(_packed_0_7)[0];
    float kg_7[4];
    #pragma unroll
    for (int e_38 = 0; e_38 < 4; e_38++) {
        float k_e_15 = 0.0f;
        if (t_u_15 < clen) {
            k_e_15 = kv_7[e_38];
        }
        float _exp2_18 = approx_exp2(gk_n[e_38] - gkt_7[e_38]);
        kg_7[e_38] = k_e_15 * _exp2_18;
    }
    unsigned int _packed_1_7[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_7[_lp*2 + 0], kg_7[_lp*2+1 + 0]));
        _packed_1_7[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_7[_lp*2 + 2], kg_7[_lp*2+1 + 2]));
        _packed_1_7[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(kg_out + iidx_7)[0] = reinterpret_cast<int2*>(_packed_1_7)[0];
    float qv_7[4];
    #pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
        (&qv_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[14 + _pair]) << 16);
        (&qv_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[14 + _pair]) & 0xffff0000u);
    }
    float qg_7[4];
    #pragma unroll
    for (int e_39 = 0; e_39 < 4; e_39++) {
        float q_e_7 = 0.0f;
        if (t_u_15 < clen) {
            q_e_7 = qv_7[e_39];
        }
        qg_7[e_39] = q_e_7 * eg_7[e_39];
    }
    unsigned int _packed_2_7[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_7[_lp*2 + 0], qg_7[_lp*2+1 + 0]));
        _packed_2_7[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_7[_lp*2 + 2], qg_7[_lp*2+1 + 2]));
        _packed_2_7[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(qg_out + iidx_7)[0] = reinterpret_cast<int2*>(_packed_2_7)[0];
    if (lane_1 == 0) {
        beta_int[(int0 + (long long)t_u_15) * (long long)num_heads + (long long)head] = bt_7;
    }
    if (clen == 0) {
        float zq[4];
        #pragma unroll
        for (int e_40 = 0; e_40 < 4; e_40++) {
            zq[e_40] = 0.0f;
        }
        #pragma unroll
        for (int u_2 = 0; u_2 < 8; u_2++) {
            int t_z = warp_0 * 8 + u_2;
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
