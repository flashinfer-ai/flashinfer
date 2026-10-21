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
#define SMEM_X_S_OFF 0
#define SMEM_X_S_STAGE_BYTES 34816
#define SMEM_X_S_STRIDE 34816
#define SMEM_G_S_OFF 34816
#define SMEM_G_S_STAGE_BYTES 34816
#define SMEM_G_S_STRIDE 34816
#define SMEM_BETA_S_OFF 69632
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_DIAG_S_OFF 69888
#define SMEM_DIAG_S_STAGE_BYTES 5120
#define SMEM_DIAG_S_STRIDE 5120
#define SMEM_AKKW_S_OFF 0
#define SMEM_AKKW_S_STAGE_BYTES 18432
#define SMEM_AKKW_S_STRIDE 18432
#define SMEM_AKKO_S_OFF 34816
#define SMEM_AKKO_S_STAGE_BYTES 18432
#define SMEM_AKKO_S_STRIDE 18432
#define SMEM_SCR_S_OFF 53248
#define SMEM_SCR_S_STAGE_BYTES 4608
#define SMEM_SCR_S_STRIDE 4608
#define SMEM_TOTAL 75008
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_6e17ba1f5255e2a02f99(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ gk, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ aqk_ext, __nv_bfloat16* __restrict__ akk_ext, __nv_bfloat16* __restrict__ aqk_int, __nv_bfloat16* __restrict__ akk_int, int num_heads, int num_qk_heads, int group, int write_ext, int write_int, float scale)
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
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int x_s_addr = smem + 0;
    float* g_s = reinterpret_cast<float*>(smem_raw + 34816);
    const int g_s_addr = smem + 34816;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 69632);
    const int beta_s_addr = smem + 69632;
    float* diag_s = reinterpret_cast<float*>(smem_raw + 69888);
    const int diag_s_addr = smem + 69888;
    float* akkw_s = reinterpret_cast<float*>(smem_raw + 0);
    const int akkw_s_addr = smem + 0;
    float* akko_s = reinterpret_cast<float*>(smem_raw + 34816);
    const int akko_s_addr = smem + 34816;
    float* scr_s = reinterpret_cast<float*>(smem_raw + 53248);
    const int scr_s_addr = smem + 53248;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    int clast = clen - 1;
    if (clast < 0) {
        clast = 0;
    }
    long long abase = (ext0 * (long long)num_heads + (long long)head) * 64;
    long long ibase = ((long long)row0 * (long long)num_heads + (long long)head) * 64;
    int ars = num_heads * 64;
    if (tid_0 < 64) {
        float b_v = 0.0f;
        if (tid_0 < clen) {
            __nv_bfloat16 b_raw = beta[(ext0 + (long long)tid_0) * (long long)num_heads + (long long)head];
            float _cvt_f32_0 = (float)b_raw;
            b_v = _cvt_f32_0;
        }
        beta_s[tid_0] = b_v;
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int e1 = (r * 256 + tid_0) * 8;
        int t_b = e1 / 128;
        int d_b = e1 - t_b * 128;
        int t_c = t_b;
        if (t_b > clast) {
            t_c = clast;
        }
        long long qk_idx = ((ext0 + (long long)t_c) * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d_b;
        float qv[8];
        float kv[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + qk_idx);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + qk_idx);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&kv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            float q_e = 0.0f;
            float k_e = 0.0f;
            if (t_b < clen) {
                q_e = qv[e];
                k_e = kv[e];
            }
            qv[e] = q_e;
            kv[e] = k_e;
        }
        unsigned int qv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qv[_lp*2 + 0], qv[_lp*2+1 + 0]));
            qv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        unsigned int kv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kv[_lp*2 + 0], kv[_lp*2+1 + 0]));
            kv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_s_addr + (unsigned int)((t_b * 136 + d_b) * 2)), "r"(qv_bf16[0]), "r"(qv_bf16[1]), "r"(qv_bf16[2]), "r"(qv_bf16[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_s_addr + (unsigned int)(((64 + t_b) * 136 + d_b) * 2)), "r"(kv_bf16[0]), "r"(kv_bf16[1]), "r"(kv_bf16[2]), "r"(kv_bf16[3]) : "memory");
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        int e4 = (r_1 * 256 + tid_0) * 4;
        int t_g = e4 / 128;
        int d_g = e4 - t_g * 128;
        long long g_idx = (((long long)row0 + (long long)t_g) * (long long)num_heads + (long long)head) * 128 + (long long)d_g;
        float gv[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(gk + g_idx);
            gv[0 + 0] = _v4.x;
            gv[0 + 1] = _v4.y;
            gv[0 + 2] = _v4.z;
            gv[0 + 3] = _v4.w;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(g_s_addr + (unsigned int)((t_g * 136 + d_g) * 4)), "f"(gv[0]), "f"(gv[1]), "f"(gv[2]), "f"(gv[3]) : "memory");
    }
    __syncthreads();
    int grp = lane / 4;
    int tq = lane % 4;
    float acc_q[8];
    float acc_k[8];
    float acc_q1[4];
    float acc_k1[4];
    float acc_q3[12];
    float acc_k3[12];
    int trow_g = 48;
    int srow_g = 24;
    if (warp == 5) {
        trow_g = 32;
        srow_g = 0;
    }
    if (warp == 6) {
        srow_g = 0;
    }
    if (warp <= 4) {
        int i_0 = warp;
        int j_0 = warp;
        int diag_0 = 1;
        float shift_0 = 21.64042561333445f;
        if (warp == 4) {
            i_0 = 1;
            j_0 = 0;
            diag_0 = 0;
            shift_0 = 0.0f;
        }
        int arow_0 = i_0 * 16;
        acc_q[0] = 0.0f;
        acc_q[1] = 0.0f;
        acc_q[2] = 0.0f;
        acc_q[3] = 0.0f;
        acc_q[4] = 0.0f;
        acc_q[5] = 0.0f;
        acc_q[6] = 0.0f;
        acc_q[7] = 0.0f;
        acc_k[0] = 0.0f;
        acc_k[1] = 0.0f;
        acc_k[2] = 0.0f;
        acc_k[3] = 0.0f;
        acc_k[4] = 0.0f;
        acc_k[5] = 0.0f;
        acc_k[6] = 0.0f;
        acc_k[7] = 0.0f;
        #pragma unroll 1
        for (int kb = 0; kb < 16; kb += 4) {
            #pragma unroll
            for (int ku = 0; ku < 4; ku++) {
                int d = kb * 8 + (ku * 8 + tq * 2);
                float an[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&an[0])), "=r"(*reinterpret_cast<uint32_t*>(&an[(0) + 1]))
                    : "r"(g_s_addr + (unsigned int)((arow_0 * 136 + d) * 4)));
                an[0] = an[0] - shift_0;
                an[1] = an[1] - shift_0;
                float gA[4];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gA[0])), "=r"(*reinterpret_cast<uint32_t*>(&gA[(0) + 1]))
                    : "r"(g_s_addr + (unsigned int)(((i_0 * 16 + grp) * 136 + d) * 4)));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gA[2])), "=r"(*reinterpret_cast<uint32_t*>(&gA[(2) + 1]))
                    : "r"(g_s_addr + (unsigned int)(((i_0 * 16 + grp + 8) * 136 + d) * 4)));
                float eA[4];
                float _exp2_0 = approx_exp2(gA[0] - an[0]);
                eA[0] = _exp2_0;
                float _exp2_1 = approx_exp2(gA[2] - an[0]);
                eA[1] = _exp2_1;
                float _exp2_2 = approx_exp2(gA[1] - an[1]);
                eA[2] = _exp2_2;
                float _exp2_3 = approx_exp2(gA[3] - an[1]);
                eA[3] = _exp2_3;
                unsigned int qw[2];
                unsigned int kw[2];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw[0])) : "r"(x_s_addr + (unsigned int)(((i_0 * 16 + grp) * 136 + d) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw[1])) : "r"(x_s_addr + (unsigned int)(((i_0 * 16 + grp + 8) * 136 + d) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw[0])) : "r"(x_s_addr + (unsigned int)(((64 + i_0 * 16 + grp) * 136 + d) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw[1])) : "r"(x_s_addr + (unsigned int)(((64 + i_0 * 16 + grp + 8) * 136 + d) * 2)));
                float _unpack_bf16_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) & 0xffff0000u);
                }
                float aq[4];
                float ak[4];
                aq[0] = _unpack_bf16_0[0] * eA[0];
                aq[1] = _unpack_bf16_0[2] * eA[1];
                aq[2] = _unpack_bf16_0[1] * eA[2];
                aq[3] = _unpack_bf16_0[3] * eA[3];
                ak[0] = _unpack_bf16_1[0] * eA[0];
                ak[1] = _unpack_bf16_1[2] * eA[1];
                ak[2] = _unpack_bf16_1[1] * eA[2];
                ak[3] = _unpack_bf16_1[3] * eA[3];
                unsigned int aqr[4];
                unsigned int akr[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    aqr[_lp] = __float_as_uint(aq[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    akr[_lp] = __float_as_uint(ak[_lp + 0]);
                }
                #pragma unroll
                for (int nn = 0; nn < 2; nn++) {
                    int s = j_0 * 16 + nn * 8 + grp;
                    float gB[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gB[0])), "=r"(*reinterpret_cast<uint32_t*>(&gB[(0) + 1]))
                        : "r"(g_s_addr + (unsigned int)((s * 136 + d) * 4)));
                    unsigned int bw[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bw[0])) : "r"(x_s_addr + (unsigned int)(((64 + s) * 136 + d) * 2)));
                    float _unpack_bf16_2[2];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(bw[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(bw[_pair]) & 0xffff0000u);
                    }
                    float b[2];
                    float _exp2_4 = approx_exp2(an[0] - gB[0]);
                    b[0] = _unpack_bf16_2[0] * _exp2_4;
                    float _exp2_5 = approx_exp2(an[1] - gB[1]);
                    b[1] = _unpack_bf16_2[1] * _exp2_5;
                    unsigned int br[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br[_lp] = __float_as_uint(b[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_q + nn * 4)[0]), "+f"((acc_q + nn * 4)[1]), "+f"((acc_q + nn * 4)[2]), "+f"((acc_q + nn * 4)[3])
                        : "r"(aqr[0]), "r"(aqr[1]), "r"(aqr[2]), "r"(aqr[3]), "r"(br[0]), "r"(br[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_k + nn * 4)[0]), "+f"((acc_k + nn * 4)[1]), "+f"((acc_k + nn * 4)[2]), "+f"((acc_k + nn * 4)[3])
                        : "r"(akr[0]), "r"(akr[1]), "r"(akr[2]), "r"(akr[3]), "r"(br[0]), "r"(br[1]));
                }
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int t_d = i_0 * 16 + grp + half * 8;
            #pragma unroll
            for (int nn_1 = 0; nn_1 < 2; nn_1++) {
                int s_d = j_0 * 16 + nn_1 * 8 + tq * 2;
                float v0 = acc_q[nn_1 * 4 + half * 2] * scale;
                float v1 = acc_q[nn_1 * 4 + half * 2 + 1] * scale;
                if (diag_0 != 0) {
                    if (s_d > t_d) {
                        v0 = 0.0f;
                    }
                    if (t_d < s_d + 1) {
                        v1 = 0.0f;
                    }
                }
                float pr[2];
                pr[0] = v0;
                pr[1] = v1;
                if (write_ext != 0) {
                    if (t_d < clen) {
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(pr[0 + 0], pr[0 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d * ars) + (long long)s_d)))[0]) = _pk;
                        }
                    }
                }
                if (write_int != 0) {
                    float w0 = v0;
                    float w1 = v1;
                    if (t_d >= clen) {
                        w0 = 0.0f;
                        w1 = 0.0f;
                    }
                    float pi[2];
                    pi[0] = w0;
                    pi[1] = w1;
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(pi[0 + 0], pi[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_int + (ibase + (long long)(t_d * ars) + (long long)s_d)))[0]) = _pk;
                    }
                }
            }
        }
        if (warp < 4) {
            float zero8[8];
            zero8[0] = 0.0f;
            zero8[1] = 0.0f;
            zero8[2] = 0.0f;
            zero8[3] = 0.0f;
            zero8[4] = 0.0f;
            zero8[5] = 0.0f;
            zero8[6] = 0.0f;
            zero8[7] = 0.0f;
            int rr = lane / 2;
            int part = lane - rr * 2;
            int t_d_1 = i_0 * 16 + rr;
            int nchunks = (3 - i_0) * 2;
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                int cc = part + c * 2;
                if (cc < nchunks) {
                    int col = (i_0 + 1) * 16 + cc * 8;
                    if (write_ext != 0) {
                        if (t_d_1 < clen) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(zero8[0 + 0], zero8[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(zero8[0 + 2], zero8[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(zero8[0 + 4], zero8[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(zero8[0 + 6], zero8[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d_1 * ars) + (long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                    if (write_int != 0) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(zero8[0 + 0], zero8[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(zero8[0 + 2], zero8[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(zero8[0 + 4], zero8[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(zero8[0 + 6], zero8[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(aqk_int + (ibase + (long long)(t_d_1 * ars) + (long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int half_1 = 0; half_1 < 2; half_1++) {
                int t_d_0 = i_0 * 16 + grp + half_1 * 8;
                float bt_d = beta_s[t_d_0];
                #pragma unroll
                for (int nn_2 = 0; nn_2 < 2; nn_2++) {
                    int s_d_1 = i_0 * 16 + nn_2 * 8 + tq * 2;
                    float v0_1 = acc_k[nn_2 * 4 + half_1 * 2] * bt_d;
                    float v1_1 = acc_k[nn_2 * 4 + half_1 * 2 + 1] * bt_d;
                    if (s_d_1 >= t_d_0) {
                        v0_1 = 0.0f;
                    }
                    if (t_d_0 <= s_d_1 + 1) {
                        v1_1 = 0.0f;
                    }
                    float pr_1[2];
                    pr_1[0] = v0_1;
                    pr_1[1] = v1_1;
                    asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(diag_s_addr + (unsigned int)((i_0 * 16 * 20 + (grp + half_1 * 8) * 20 + nn_2 * 8 + tq * 2) * 4)), "f"(pr_1[0]), "f"(pr_1[1]) : "memory");
                }
            }
            __syncwarp();
            int c_1 = lane % 16;
            int dbase = i_0 * 16 * 20;
            float Ai[16];
            #pragma unroll
            for (int r_2 = 0; r_2 < 16; r_2++) {
                float a_r = 0.0f;
                if (lane < 16) {
                    float raw_r = diag_s[dbase + r_2 * 20 + c_1];
                    a_r = -raw_r;
                }
                Ai[r_2] = a_r;
            }
            #pragma unroll
            for (int i2 = 2; i2 < 16; i2++) {
                float a_c = Ai[i2];
                float tot = a_c;
                #pragma unroll
                for (int jj = 0; jj < i2; jj++) {
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, a_c, jj);
                    float a_j = _shfl_0;
                    tot = __fmaf_rn(a_j, Ai[jj], tot);
                }
                Ai[i2] = tot;
            }
            if (lane < 16) {
                #pragma unroll
                for (int r_3 = 0; r_3 < 16; r_3++) {
                    float v_r = Ai[r_3];
                    if (c_1 == r_3) {
                        v_r = v_r + 1.0f;
                    }
                    diag_s[dbase + r_3 * 20 + c_1] = v_r;
                }
            }
        }
        if (warp == 4) {
            acc_q1[0] = 0.0f;
            acc_q1[1] = 0.0f;
            acc_q1[2] = 0.0f;
            acc_q1[3] = 0.0f;
            acc_k1[0] = 0.0f;
            acc_k1[1] = 0.0f;
            acc_k1[2] = 0.0f;
            acc_k1[3] = 0.0f;
            #pragma unroll 1
            for (int kb_1 = 0; kb_1 < 16; kb_1 += 4) {
                #pragma unroll
                for (int ku_1 = 0; ku_1 < 4; ku_1++) {
                    int d_1 = kb_1 * 8 + (ku_1 * 8 + tq * 2);
                    float an_1[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&an_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_1[(0) + 1]))
                        : "r"(g_s_addr + (unsigned int)((4352 + d_1) * 4)));
                    float gA_1[4];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gA_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gA_1[(0) + 1]))
                        : "r"(g_s_addr + (unsigned int)(((32 + grp) * 136 + d_1) * 4)));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gA_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&gA_1[(2) + 1]))
                        : "r"(g_s_addr + (unsigned int)(((32 + grp + 8) * 136 + d_1) * 4)));
                    float eA_1[4];
                    float _exp2_6 = approx_exp2(gA_1[0] - an_1[0]);
                    eA_1[0] = _exp2_6;
                    float _exp2_7 = approx_exp2(gA_1[2] - an_1[0]);
                    eA_1[1] = _exp2_7;
                    float _exp2_8 = approx_exp2(gA_1[1] - an_1[1]);
                    eA_1[2] = _exp2_8;
                    float _exp2_9 = approx_exp2(gA_1[3] - an_1[1]);
                    eA_1[3] = _exp2_9;
                    unsigned int qw_1[2];
                    unsigned int kw_1[2];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[0])) : "r"(x_s_addr + (unsigned int)(((32 + grp) * 136 + d_1) * 2)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[1])) : "r"(x_s_addr + (unsigned int)(((32 + grp + 8) * 136 + d_1) * 2)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_1[0])) : "r"(x_s_addr + (unsigned int)(((96 + grp) * 136 + d_1) * 2)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_1[1])) : "r"(x_s_addr + (unsigned int)(((96 + grp + 8) * 136 + d_1) * 2)));
                    float _unpack_bf16_3[4];
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) & 0xffff0000u);
                    }
                    float _unpack_bf16_4[4];
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) & 0xffff0000u);
                    }
                    float aq_1[4];
                    float ak_1[4];
                    aq_1[0] = _unpack_bf16_3[0] * eA_1[0];
                    aq_1[1] = _unpack_bf16_3[2] * eA_1[1];
                    aq_1[2] = _unpack_bf16_3[1] * eA_1[2];
                    aq_1[3] = _unpack_bf16_3[3] * eA_1[3];
                    ak_1[0] = _unpack_bf16_4[0] * eA_1[0];
                    ak_1[1] = _unpack_bf16_4[2] * eA_1[1];
                    ak_1[2] = _unpack_bf16_4[1] * eA_1[2];
                    ak_1[3] = _unpack_bf16_4[3] * eA_1[3];
                    unsigned int aqr_1[4];
                    unsigned int akr_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        aqr_1[_lp] = __float_as_uint(aq_1[_lp + 0]);
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        akr_1[_lp] = __float_as_uint(ak_1[_lp + 0]);
                    }
                    #pragma unroll
                    for (int nn_3 = 0; nn_3 < 1; nn_3++) {
                        int s_1 = 24 + nn_3 * 8 + grp;
                        float gB_1[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gB_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gB_1[(0) + 1]))
                            : "r"(g_s_addr + (unsigned int)((s_1 * 136 + d_1) * 4)));
                        unsigned int bw_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bw_1[0])) : "r"(x_s_addr + (unsigned int)(((64 + s_1) * 136 + d_1) * 2)));
                        float _unpack_bf16_5[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(bw_1[_pair]) << 16);
                            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(bw_1[_pair]) & 0xffff0000u);
                        }
                        float b_1[2];
                        float _exp2_10 = approx_exp2(an_1[0] - gB_1[0]);
                        b_1[0] = _unpack_bf16_5[0] * _exp2_10;
                        float _exp2_11 = approx_exp2(an_1[1] - gB_1[1]);
                        b_1[1] = _unpack_bf16_5[1] * _exp2_11;
                        unsigned int br_1[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            br_1[_lp] = __float_as_uint(b_1[_lp + 0]);
                        }
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"((acc_q1 + nn_3 * 4)[0]), "+f"((acc_q1 + nn_3 * 4)[1]), "+f"((acc_q1 + nn_3 * 4)[2]), "+f"((acc_q1 + nn_3 * 4)[3])
                            : "r"(aqr_1[0]), "r"(aqr_1[1]), "r"(aqr_1[2]), "r"(aqr_1[3]), "r"(br_1[0]), "r"(br_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"((acc_k1 + nn_3 * 4)[0]), "+f"((acc_k1 + nn_3 * 4)[1]), "+f"((acc_k1 + nn_3 * 4)[2]), "+f"((acc_k1 + nn_3 * 4)[3])
                            : "r"(akr_1[0]), "r"(akr_1[1]), "r"(akr_1[2]), "r"(akr_1[3]), "r"(br_1[0]), "r"(br_1[1]));
                    }
                }
            }
            #pragma unroll
            for (int half_2 = 0; half_2 < 2; half_2++) {
                int t_d_2 = 32 + grp + half_2 * 8;
                #pragma unroll
                for (int nn_4 = 0; nn_4 < 1; nn_4++) {
                    int s_d_2 = 24 + nn_4 * 8 + tq * 2;
                    float v0_2 = acc_q1[nn_4 * 4 + half_2 * 2] * scale;
                    float v1_2 = acc_q1[nn_4 * 4 + half_2 * 2 + 1] * scale;
                    float pr_2[2];
                    pr_2[0] = v0_2;
                    pr_2[1] = v1_2;
                    if (write_ext != 0) {
                        if (t_d_2 < clen) {
                            {
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(pr_2[0 + 0], pr_2[0 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d_2 * ars) + (long long)s_d_2)))[0]) = _pk;
                            }
                        }
                    }
                    if (write_int != 0) {
                        float w0_1 = v0_2;
                        float w1_1 = v1_2;
                        if (t_d_2 >= clen) {
                            w0_1 = 0.0f;
                            w1_1 = 0.0f;
                        }
                        float pi_1[2];
                        pi_1[0] = w0_1;
                        pi_1[1] = w1_1;
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(pi_1[0 + 0], pi_1[0 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_int + (ibase + (long long)(t_d_2 * ars) + (long long)s_d_2)))[0]) = _pk;
                        }
                    }
                }
            }
        }
    }
    if (warp >= 5) {
        acc_q3[0] = 0.0f;
        acc_q3[1] = 0.0f;
        acc_q3[2] = 0.0f;
        acc_q3[3] = 0.0f;
        acc_q3[4] = 0.0f;
        acc_q3[5] = 0.0f;
        acc_q3[6] = 0.0f;
        acc_q3[7] = 0.0f;
        acc_q3[8] = 0.0f;
        acc_q3[9] = 0.0f;
        acc_q3[10] = 0.0f;
        acc_q3[11] = 0.0f;
        acc_k3[0] = 0.0f;
        acc_k3[1] = 0.0f;
        acc_k3[2] = 0.0f;
        acc_k3[3] = 0.0f;
        acc_k3[4] = 0.0f;
        acc_k3[5] = 0.0f;
        acc_k3[6] = 0.0f;
        acc_k3[7] = 0.0f;
        acc_k3[8] = 0.0f;
        acc_k3[9] = 0.0f;
        acc_k3[10] = 0.0f;
        acc_k3[11] = 0.0f;
        #pragma unroll 1
        for (int kb_2 = 0; kb_2 < 16; kb_2 += 4) {
            #pragma unroll
            for (int ku_2 = 0; ku_2 < 4; ku_2++) {
                int d_2 = kb_2 * 8 + (ku_2 * 8 + tq * 2);
                float an_2[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&an_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_2[(0) + 1]))
                    : "r"(g_s_addr + (unsigned int)((trow_g * 136 + d_2) * 4)));
                float gA_2[4];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gA_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gA_2[(0) + 1]))
                    : "r"(g_s_addr + (unsigned int)(((trow_g + grp) * 136 + d_2) * 4)));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gA_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&gA_2[(2) + 1]))
                    : "r"(g_s_addr + (unsigned int)(((trow_g + grp + 8) * 136 + d_2) * 4)));
                float eA_2[4];
                float _exp2_12 = approx_exp2(gA_2[0] - an_2[0]);
                eA_2[0] = _exp2_12;
                float _exp2_13 = approx_exp2(gA_2[2] - an_2[0]);
                eA_2[1] = _exp2_13;
                float _exp2_14 = approx_exp2(gA_2[1] - an_2[1]);
                eA_2[2] = _exp2_14;
                float _exp2_15 = approx_exp2(gA_2[3] - an_2[1]);
                eA_2[3] = _exp2_15;
                unsigned int qw_2[2];
                unsigned int kw_2[2];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_2[0])) : "r"(x_s_addr + (unsigned int)(((trow_g + grp) * 136 + d_2) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_2[1])) : "r"(x_s_addr + (unsigned int)(((trow_g + grp + 8) * 136 + d_2) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_2[0])) : "r"(x_s_addr + (unsigned int)(((64 + trow_g + grp) * 136 + d_2) * 2)));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_2[1])) : "r"(x_s_addr + (unsigned int)(((64 + trow_g + grp + 8) * 136 + d_2) * 2)));
                float _unpack_bf16_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_2[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_2[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_2[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_2[_pair]) & 0xffff0000u);
                }
                float aq_2[4];
                float ak_2[4];
                aq_2[0] = _unpack_bf16_6[0] * eA_2[0];
                aq_2[1] = _unpack_bf16_6[2] * eA_2[1];
                aq_2[2] = _unpack_bf16_6[1] * eA_2[2];
                aq_2[3] = _unpack_bf16_6[3] * eA_2[3];
                ak_2[0] = _unpack_bf16_7[0] * eA_2[0];
                ak_2[1] = _unpack_bf16_7[2] * eA_2[1];
                ak_2[2] = _unpack_bf16_7[1] * eA_2[2];
                ak_2[3] = _unpack_bf16_7[3] * eA_2[3];
                unsigned int aqr_2[4];
                unsigned int akr_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    aqr_2[_lp] = __float_as_uint(aq_2[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    akr_2[_lp] = __float_as_uint(ak_2[_lp + 0]);
                }
                #pragma unroll
                for (int nn_5 = 0; nn_5 < 3; nn_5++) {
                    int s_2 = srow_g + nn_5 * 8 + grp;
                    float gB_2[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gB_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gB_2[(0) + 1]))
                        : "r"(g_s_addr + (unsigned int)((s_2 * 136 + d_2) * 4)));
                    unsigned int bw_2[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bw_2[0])) : "r"(x_s_addr + (unsigned int)(((64 + s_2) * 136 + d_2) * 2)));
                    float _unpack_bf16_8[2];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(bw_2[_pair]) << 16);
                        (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(bw_2[_pair]) & 0xffff0000u);
                    }
                    float b_2[2];
                    float _exp2_16 = approx_exp2(an_2[0] - gB_2[0]);
                    b_2[0] = _unpack_bf16_8[0] * _exp2_16;
                    float _exp2_17 = approx_exp2(an_2[1] - gB_2[1]);
                    b_2[1] = _unpack_bf16_8[1] * _exp2_17;
                    unsigned int br_2[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br_2[_lp] = __float_as_uint(b_2[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_q3 + nn_5 * 4)[0]), "+f"((acc_q3 + nn_5 * 4)[1]), "+f"((acc_q3 + nn_5 * 4)[2]), "+f"((acc_q3 + nn_5 * 4)[3])
                        : "r"(aqr_2[0]), "r"(aqr_2[1]), "r"(aqr_2[2]), "r"(aqr_2[3]), "r"(br_2[0]), "r"(br_2[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_k3 + nn_5 * 4)[0]), "+f"((acc_k3 + nn_5 * 4)[1]), "+f"((acc_k3 + nn_5 * 4)[2]), "+f"((acc_k3 + nn_5 * 4)[3])
                        : "r"(akr_2[0]), "r"(akr_2[1]), "r"(akr_2[2]), "r"(akr_2[3]), "r"(br_2[0]), "r"(br_2[1]));
                }
            }
        }
        #pragma unroll
        for (int half_3 = 0; half_3 < 2; half_3++) {
            int t_d_3 = trow_g + grp + half_3 * 8;
            #pragma unroll
            for (int nn_6 = 0; nn_6 < 3; nn_6++) {
                int s_d_3 = srow_g + nn_6 * 8 + tq * 2;
                float v0_3 = acc_q3[nn_6 * 4 + half_3 * 2] * scale;
                float v1_3 = acc_q3[nn_6 * 4 + half_3 * 2 + 1] * scale;
                float pr_3[2];
                pr_3[0] = v0_3;
                pr_3[1] = v1_3;
                if (write_ext != 0) {
                    if (t_d_3 < clen) {
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(pr_3[0 + 0], pr_3[0 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d_3 * ars) + (long long)s_d_3)))[0]) = _pk;
                        }
                    }
                }
                if (write_int != 0) {
                    float w0_2 = v0_3;
                    float w1_2 = v1_3;
                    if (t_d_3 >= clen) {
                        w0_2 = 0.0f;
                        w1_2 = 0.0f;
                    }
                    float pi_2[2];
                    pi_2[0] = w0_2;
                    pi_2[1] = w1_2;
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(pi_2[0 + 0], pi_2[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_int + (ibase + (long long)(t_d_3 * ars) + (long long)s_d_3)))[0]) = _pk;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (warp == 4) {
        #pragma unroll
        for (int half_4 = 0; half_4 < 2; half_4++) {
            int t_d_4 = 16 + grp + half_4 * 8;
            float bt_d_1 = beta_s[t_d_4];
            #pragma unroll
            for (int nn_7 = 0; nn_7 < 2; nn_7++) {
                float pr_4[2];
                pr_4[0] = acc_k[nn_7 * 4 + half_4 * 2] * bt_d_1;
                pr_4[1] = acc_k[nn_7 * 4 + half_4 * 2 + 1] * bt_d_1;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akkw_s_addr + (unsigned int)((t_d_4 * 72 + nn_7 * 8 + tq * 2) * 4)), "f"(pr_4[0]), "f"(pr_4[1]) : "memory");
            }
        }
        #pragma unroll
        for (int half_5 = 0; half_5 < 2; half_5++) {
            int t_d_5 = 32 + grp + half_5 * 8;
            float bt_d_2 = beta_s[t_d_5];
            #pragma unroll
            for (int nn_8 = 0; nn_8 < 1; nn_8++) {
                float pr_5[2];
                pr_5[0] = acc_k1[nn_8 * 4 + half_5 * 2] * bt_d_2;
                pr_5[1] = acc_k1[nn_8 * 4 + half_5 * 2 + 1] * bt_d_2;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akkw_s_addr + (unsigned int)((t_d_5 * 72 + 24 + nn_8 * 8 + tq * 2) * 4)), "f"(pr_5[0]), "f"(pr_5[1]) : "memory");
            }
        }
    }
    if (warp >= 5) {
        #pragma unroll
        for (int half_6 = 0; half_6 < 2; half_6++) {
            int t_d_6 = trow_g + grp + half_6 * 8;
            float bt_d_3 = beta_s[t_d_6];
            #pragma unroll
            for (int nn_9 = 0; nn_9 < 3; nn_9++) {
                float pr_6[2];
                pr_6[0] = acc_k3[nn_9 * 4 + half_6 * 2] * bt_d_3;
                pr_6[1] = acc_k3[nn_9 * 4 + half_6 * 2 + 1] * bt_d_3;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akkw_s_addr + (unsigned int)((t_d_6 * 72 + srow_g + nn_9 * 8 + tq * 2) * 4)), "f"(pr_6[0]), "f"(pr_6[1]) : "memory");
            }
        }
    }
    __syncthreads();
    float acc[8];
    if (warp == 0) {
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            float a[4];
            a[0] = diag_s[320 + grp * 20 + kk * 8 + tq];
            a[1] = diag_s[320 + (grp + 8) * 20 + kk * 8 + tq];
            a[2] = diag_s[320 + grp * 20 + kk * 8 + tq + 4];
            a[3] = diag_s[320 + (grp + 8) * 20 + kk * 8 + tq + 4];
            unsigned int ar[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar[_lp] = __float_as_uint(a[_lp + 0]);
            }
            #pragma unroll
            for (int nn_10 = 0; nn_10 < 2; nn_10++) {
                float b_3[2];
                b_3[0] = akkw_s[1152 + (kk * 8 + tq) * 72 + nn_10 * 8 + grp];
                b_3[1] = akkw_s[1152 + (kk * 8 + tq + 4) * 72 + nn_10 * 8 + grp];
                unsigned int br_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_3[_lp] = __float_as_uint(b_3[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_10 * 4)[0]), "+f"((acc + nn_10 * 4)[1]), "+f"((acc + nn_10 * 4)[2]), "+f"((acc + nn_10 * 4)[3])
                    : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br_3[0]), "r"(br_3[1]));
            }
        }
        #pragma unroll
        for (int nn_11 = 0; nn_11 < 2; nn_11++) {
            #pragma unroll
            for (int half_7 = 0; half_7 < 2; half_7++) {
                float v0_4 = acc[nn_11 * 4 + half_7 * 2];
                float v1_4 = acc[nn_11 * 4 + half_7 * 2 + 1];
                float pr_7[2];
                pr_7[0] = v0_4;
                pr_7[1] = v1_4;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)(((grp + half_7 * 8) * 24 + nn_11 * 8 + tq * 2) * 4)), "f"(pr_7[0]), "f"(pr_7[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_1 = 0; kk_1 < 2; kk_1++) {
            float a_1[4];
            a_1[0] = scr_s[grp * 24 + kk_1 * 8 + tq];
            a_1[1] = scr_s[(grp + 8) * 24 + kk_1 * 8 + tq];
            a_1[2] = scr_s[grp * 24 + kk_1 * 8 + tq + 4];
            a_1[3] = scr_s[(grp + 8) * 24 + kk_1 * 8 + tq + 4];
            unsigned int ar_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_1[_lp] = __float_as_uint(a_1[_lp + 0]);
            }
            #pragma unroll
            for (int nn_12 = 0; nn_12 < 2; nn_12++) {
                float b_4[2];
                b_4[0] = diag_s[(kk_1 * 8 + tq) * 20 + nn_12 * 8 + grp];
                b_4[1] = diag_s[(kk_1 * 8 + tq + 4) * 20 + nn_12 * 8 + grp];
                unsigned int br_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_4[_lp] = __float_as_uint(b_4[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_12 * 4)[0]), "+f"((acc + nn_12 * 4)[1]), "+f"((acc + nn_12 * 4)[2]), "+f"((acc + nn_12 * 4)[3])
                    : "r"(ar_1[0]), "r"(ar_1[1]), "r"(ar_1[2]), "r"(ar_1[3]), "r"(br_4[0]), "r"(br_4[1]));
            }
        }
        #pragma unroll
        for (int nn_13 = 0; nn_13 < 2; nn_13++) {
            #pragma unroll
            for (int half_8 = 0; half_8 < 2; half_8++) {
                float v0_5 = acc[nn_13 * 4 + half_8 * 2];
                float v1_5 = acc[nn_13 * 4 + half_8 * 2 + 1];
                v0_5 = -v0_5;
                v1_5 = -v1_5;
                float pr_8[2];
                pr_8[0] = v0_5;
                pr_8[1] = v1_5;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((1152 + (grp + half_8 * 8) * 72 + nn_13 * 8 + tq * 2) * 4)), "f"(pr_8[0]), "f"(pr_8[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_2 = 0; kk_2 < 2; kk_2++) {
            float a_2[4];
            a_2[0] = akkw_s[2304 + grp * 72 + kk_2 * 8 + tq];
            a_2[1] = akkw_s[2304 + (grp + 8) * 72 + kk_2 * 8 + tq];
            a_2[2] = akkw_s[2304 + grp * 72 + kk_2 * 8 + tq + 4];
            a_2[3] = akkw_s[2304 + (grp + 8) * 72 + kk_2 * 8 + tq + 4];
            unsigned int ar_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_2[_lp] = __float_as_uint(a_2[_lp + 0]);
            }
            #pragma unroll
            for (int nn_14 = 0; nn_14 < 2; nn_14++) {
                float b_5[2];
                b_5[0] = diag_s[(kk_2 * 8 + tq) * 20 + nn_14 * 8 + grp];
                b_5[1] = diag_s[(kk_2 * 8 + tq + 4) * 20 + nn_14 * 8 + grp];
                unsigned int br_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_5[_lp] = __float_as_uint(b_5[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_14 * 4)[0]), "+f"((acc + nn_14 * 4)[1]), "+f"((acc + nn_14 * 4)[2]), "+f"((acc + nn_14 * 4)[3])
                    : "r"(ar_2[0]), "r"(ar_2[1]), "r"(ar_2[2]), "r"(ar_2[3]), "r"(br_5[0]), "r"(br_5[1]));
            }
        }
        #pragma unroll
        for (int m = 1; m < 2; m++) {
            #pragma unroll
            for (int kk_3 = 0; kk_3 < 2; kk_3++) {
                float a_3[4];
                a_3[0] = akkw_s[2304 + m * 16 + grp * 72 + kk_3 * 8 + tq];
                a_3[1] = akkw_s[2304 + m * 16 + (grp + 8) * 72 + kk_3 * 8 + tq];
                a_3[2] = akkw_s[2304 + m * 16 + grp * 72 + kk_3 * 8 + tq + 4];
                a_3[3] = akkw_s[2304 + m * 16 + (grp + 8) * 72 + kk_3 * 8 + tq + 4];
                unsigned int ar_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    ar_3[_lp] = __float_as_uint(a_3[_lp + 0]);
                }
                #pragma unroll
                for (int nn_15 = 0; nn_15 < 2; nn_15++) {
                    float b_6[2];
                    b_6[0] = akko_s[m * 16 * 72 + (kk_3 * 8 + tq) * 72 + nn_15 * 8 + grp];
                    b_6[1] = akko_s[m * 16 * 72 + (kk_3 * 8 + tq + 4) * 72 + nn_15 * 8 + grp];
                    unsigned int br_6[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br_6[_lp] = __float_as_uint(b_6[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nn_15 * 4)[0]), "+f"((acc + nn_15 * 4)[1]), "+f"((acc + nn_15 * 4)[2]), "+f"((acc + nn_15 * 4)[3])
                        : "r"(ar_3[0]), "r"(ar_3[1]), "r"(ar_3[2]), "r"(ar_3[3]), "r"(br_6[0]), "r"(br_6[1]));
                }
            }
        }
        #pragma unroll
        for (int nn_16 = 0; nn_16 < 2; nn_16++) {
            #pragma unroll
            for (int half_9 = 0; half_9 < 2; half_9++) {
                float v0_6 = acc[nn_16 * 4 + half_9 * 2];
                float v1_6 = acc[nn_16 * 4 + half_9 * 2 + 1];
                float pr_9[2];
                pr_9[0] = v0_6;
                pr_9[1] = v1_6;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)(((grp + half_9 * 8) * 24 + nn_16 * 8 + tq * 2) * 4)), "f"(pr_9[0]), "f"(pr_9[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_4 = 0; kk_4 < 2; kk_4++) {
            float a_4[4];
            a_4[0] = diag_s[640 + grp * 20 + kk_4 * 8 + tq];
            a_4[1] = diag_s[640 + (grp + 8) * 20 + kk_4 * 8 + tq];
            a_4[2] = diag_s[640 + grp * 20 + kk_4 * 8 + tq + 4];
            a_4[3] = diag_s[640 + (grp + 8) * 20 + kk_4 * 8 + tq + 4];
            unsigned int ar_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_4[_lp] = __float_as_uint(a_4[_lp + 0]);
            }
            #pragma unroll
            for (int nn_17 = 0; nn_17 < 2; nn_17++) {
                float b_7[2];
                b_7[0] = scr_s[(kk_4 * 8 + tq) * 24 + nn_17 * 8 + grp];
                b_7[1] = scr_s[(kk_4 * 8 + tq + 4) * 24 + nn_17 * 8 + grp];
                unsigned int br_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_7[_lp] = __float_as_uint(b_7[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_17 * 4)[0]), "+f"((acc + nn_17 * 4)[1]), "+f"((acc + nn_17 * 4)[2]), "+f"((acc + nn_17 * 4)[3])
                    : "r"(ar_4[0]), "r"(ar_4[1]), "r"(ar_4[2]), "r"(ar_4[3]), "r"(br_7[0]), "r"(br_7[1]));
            }
        }
        #pragma unroll
        for (int nn_18 = 0; nn_18 < 2; nn_18++) {
            #pragma unroll
            for (int half_10 = 0; half_10 < 2; half_10++) {
                float v0_7 = acc[nn_18 * 4 + half_10 * 2];
                float v1_7 = acc[nn_18 * 4 + half_10 * 2 + 1];
                v0_7 = -v0_7;
                v1_7 = -v1_7;
                float pr_10[2];
                pr_10[0] = v0_7;
                pr_10[1] = v1_7;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((2304 + (grp + half_10 * 8) * 72 + nn_18 * 8 + tq * 2) * 4)), "f"(pr_10[0]), "f"(pr_10[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_5 = 0; kk_5 < 2; kk_5++) {
            float a_5[4];
            a_5[0] = akkw_s[3456 + grp * 72 + kk_5 * 8 + tq];
            a_5[1] = akkw_s[3456 + (grp + 8) * 72 + kk_5 * 8 + tq];
            a_5[2] = akkw_s[3456 + grp * 72 + kk_5 * 8 + tq + 4];
            a_5[3] = akkw_s[3456 + (grp + 8) * 72 + kk_5 * 8 + tq + 4];
            unsigned int ar_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_5[_lp] = __float_as_uint(a_5[_lp + 0]);
            }
            #pragma unroll
            for (int nn_19 = 0; nn_19 < 2; nn_19++) {
                float b_8[2];
                b_8[0] = diag_s[(kk_5 * 8 + tq) * 20 + nn_19 * 8 + grp];
                b_8[1] = diag_s[(kk_5 * 8 + tq + 4) * 20 + nn_19 * 8 + grp];
                unsigned int br_8[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_8[_lp] = __float_as_uint(b_8[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_19 * 4)[0]), "+f"((acc + nn_19 * 4)[1]), "+f"((acc + nn_19 * 4)[2]), "+f"((acc + nn_19 * 4)[3])
                    : "r"(ar_5[0]), "r"(ar_5[1]), "r"(ar_5[2]), "r"(ar_5[3]), "r"(br_8[0]), "r"(br_8[1]));
            }
        }
        #pragma unroll
        for (int m_1 = 1; m_1 < 3; m_1++) {
            #pragma unroll
            for (int kk_6 = 0; kk_6 < 2; kk_6++) {
                float a_6[4];
                a_6[0] = akkw_s[3456 + m_1 * 16 + grp * 72 + kk_6 * 8 + tq];
                a_6[1] = akkw_s[3456 + m_1 * 16 + (grp + 8) * 72 + kk_6 * 8 + tq];
                a_6[2] = akkw_s[3456 + m_1 * 16 + grp * 72 + kk_6 * 8 + tq + 4];
                a_6[3] = akkw_s[3456 + m_1 * 16 + (grp + 8) * 72 + kk_6 * 8 + tq + 4];
                unsigned int ar_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    ar_6[_lp] = __float_as_uint(a_6[_lp + 0]);
                }
                #pragma unroll
                for (int nn_20 = 0; nn_20 < 2; nn_20++) {
                    float b_9[2];
                    b_9[0] = akko_s[m_1 * 16 * 72 + (kk_6 * 8 + tq) * 72 + nn_20 * 8 + grp];
                    b_9[1] = akko_s[m_1 * 16 * 72 + (kk_6 * 8 + tq + 4) * 72 + nn_20 * 8 + grp];
                    unsigned int br_9[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br_9[_lp] = __float_as_uint(b_9[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nn_20 * 4)[0]), "+f"((acc + nn_20 * 4)[1]), "+f"((acc + nn_20 * 4)[2]), "+f"((acc + nn_20 * 4)[3])
                        : "r"(ar_6[0]), "r"(ar_6[1]), "r"(ar_6[2]), "r"(ar_6[3]), "r"(br_9[0]), "r"(br_9[1]));
                }
            }
        }
        #pragma unroll
        for (int nn_21 = 0; nn_21 < 2; nn_21++) {
            #pragma unroll
            for (int half_11 = 0; half_11 < 2; half_11++) {
                float v0_8 = acc[nn_21 * 4 + half_11 * 2];
                float v1_8 = acc[nn_21 * 4 + half_11 * 2 + 1];
                float pr_11[2];
                pr_11[0] = v0_8;
                pr_11[1] = v1_8;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)(((grp + half_11 * 8) * 24 + nn_21 * 8 + tq * 2) * 4)), "f"(pr_11[0]), "f"(pr_11[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_7 = 0; kk_7 < 2; kk_7++) {
            float a_7[4];
            a_7[0] = diag_s[960 + grp * 20 + kk_7 * 8 + tq];
            a_7[1] = diag_s[960 + (grp + 8) * 20 + kk_7 * 8 + tq];
            a_7[2] = diag_s[960 + grp * 20 + kk_7 * 8 + tq + 4];
            a_7[3] = diag_s[960 + (grp + 8) * 20 + kk_7 * 8 + tq + 4];
            unsigned int ar_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_7[_lp] = __float_as_uint(a_7[_lp + 0]);
            }
            #pragma unroll
            for (int nn_22 = 0; nn_22 < 2; nn_22++) {
                float b_10[2];
                b_10[0] = scr_s[(kk_7 * 8 + tq) * 24 + nn_22 * 8 + grp];
                b_10[1] = scr_s[(kk_7 * 8 + tq + 4) * 24 + nn_22 * 8 + grp];
                unsigned int br_10[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_10[_lp] = __float_as_uint(b_10[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_22 * 4)[0]), "+f"((acc + nn_22 * 4)[1]), "+f"((acc + nn_22 * 4)[2]), "+f"((acc + nn_22 * 4)[3])
                    : "r"(ar_7[0]), "r"(ar_7[1]), "r"(ar_7[2]), "r"(ar_7[3]), "r"(br_10[0]), "r"(br_10[1]));
            }
        }
        #pragma unroll
        for (int nn_23 = 0; nn_23 < 2; nn_23++) {
            #pragma unroll
            for (int half_12 = 0; half_12 < 2; half_12++) {
                float v0_9 = acc[nn_23 * 4 + half_12 * 2];
                float v1_9 = acc[nn_23 * 4 + half_12 * 2 + 1];
                v0_9 = -v0_9;
                v1_9 = -v1_9;
                float pr_12[2];
                pr_12[0] = v0_9;
                pr_12[1] = v1_9;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((3456 + (grp + half_12 * 8) * 72 + nn_23 * 8 + tq * 2) * 4)), "f"(pr_12[0]), "f"(pr_12[1]) : "memory");
            }
        }
        __syncwarp();
    }
    if (warp == 1) {
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_8 = 0; kk_8 < 2; kk_8++) {
            float a_8[4];
            a_8[0] = diag_s[640 + grp * 20 + kk_8 * 8 + tq];
            a_8[1] = diag_s[640 + (grp + 8) * 20 + kk_8 * 8 + tq];
            a_8[2] = diag_s[640 + grp * 20 + kk_8 * 8 + tq + 4];
            a_8[3] = diag_s[640 + (grp + 8) * 20 + kk_8 * 8 + tq + 4];
            unsigned int ar_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_8[_lp] = __float_as_uint(a_8[_lp + 0]);
            }
            #pragma unroll
            for (int nn_24 = 0; nn_24 < 2; nn_24++) {
                float b_11[2];
                b_11[0] = akkw_s[2320 + (kk_8 * 8 + tq) * 72 + nn_24 * 8 + grp];
                b_11[1] = akkw_s[2320 + (kk_8 * 8 + tq + 4) * 72 + nn_24 * 8 + grp];
                unsigned int br_11[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_11[_lp] = __float_as_uint(b_11[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_24 * 4)[0]), "+f"((acc + nn_24 * 4)[1]), "+f"((acc + nn_24 * 4)[2]), "+f"((acc + nn_24 * 4)[3])
                    : "r"(ar_8[0]), "r"(ar_8[1]), "r"(ar_8[2]), "r"(ar_8[3]), "r"(br_11[0]), "r"(br_11[1]));
            }
        }
        #pragma unroll
        for (int nn_25 = 0; nn_25 < 2; nn_25++) {
            #pragma unroll
            for (int half_13 = 0; half_13 < 2; half_13++) {
                float v0_10 = acc[nn_25 * 4 + half_13 * 2];
                float v1_10 = acc[nn_25 * 4 + half_13 * 2 + 1];
                float pr_13[2];
                pr_13[0] = v0_10;
                pr_13[1] = v1_10;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)((384 + (grp + half_13 * 8) * 24 + nn_25 * 8 + tq * 2) * 4)), "f"(pr_13[0]), "f"(pr_13[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_9 = 0; kk_9 < 2; kk_9++) {
            float a_9[4];
            a_9[0] = scr_s[384 + grp * 24 + kk_9 * 8 + tq];
            a_9[1] = scr_s[384 + (grp + 8) * 24 + kk_9 * 8 + tq];
            a_9[2] = scr_s[384 + grp * 24 + kk_9 * 8 + tq + 4];
            a_9[3] = scr_s[384 + (grp + 8) * 24 + kk_9 * 8 + tq + 4];
            unsigned int ar_9[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_9[_lp] = __float_as_uint(a_9[_lp + 0]);
            }
            #pragma unroll
            for (int nn_26 = 0; nn_26 < 2; nn_26++) {
                float b_12[2];
                b_12[0] = diag_s[320 + (kk_9 * 8 + tq) * 20 + nn_26 * 8 + grp];
                b_12[1] = diag_s[320 + (kk_9 * 8 + tq + 4) * 20 + nn_26 * 8 + grp];
                unsigned int br_12[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_12[_lp] = __float_as_uint(b_12[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_26 * 4)[0]), "+f"((acc + nn_26 * 4)[1]), "+f"((acc + nn_26 * 4)[2]), "+f"((acc + nn_26 * 4)[3])
                    : "r"(ar_9[0]), "r"(ar_9[1]), "r"(ar_9[2]), "r"(ar_9[3]), "r"(br_12[0]), "r"(br_12[1]));
            }
        }
        #pragma unroll
        for (int nn_27 = 0; nn_27 < 2; nn_27++) {
            #pragma unroll
            for (int half_14 = 0; half_14 < 2; half_14++) {
                float v0_11 = acc[nn_27 * 4 + half_14 * 2];
                float v1_11 = acc[nn_27 * 4 + half_14 * 2 + 1];
                v0_11 = -v0_11;
                v1_11 = -v1_11;
                float pr_14[2];
                pr_14[0] = v0_11;
                pr_14[1] = v1_11;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((2320 + (grp + half_14 * 8) * 72 + nn_27 * 8 + tq * 2) * 4)), "f"(pr_14[0]), "f"(pr_14[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_10 = 0; kk_10 < 2; kk_10++) {
            float a_10[4];
            a_10[0] = akkw_s[3472 + grp * 72 + kk_10 * 8 + tq];
            a_10[1] = akkw_s[3472 + (grp + 8) * 72 + kk_10 * 8 + tq];
            a_10[2] = akkw_s[3472 + grp * 72 + kk_10 * 8 + tq + 4];
            a_10[3] = akkw_s[3472 + (grp + 8) * 72 + kk_10 * 8 + tq + 4];
            unsigned int ar_10[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_10[_lp] = __float_as_uint(a_10[_lp + 0]);
            }
            #pragma unroll
            for (int nn_28 = 0; nn_28 < 2; nn_28++) {
                float b_13[2];
                b_13[0] = diag_s[320 + (kk_10 * 8 + tq) * 20 + nn_28 * 8 + grp];
                b_13[1] = diag_s[320 + (kk_10 * 8 + tq + 4) * 20 + nn_28 * 8 + grp];
                unsigned int br_13[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_13[_lp] = __float_as_uint(b_13[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_28 * 4)[0]), "+f"((acc + nn_28 * 4)[1]), "+f"((acc + nn_28 * 4)[2]), "+f"((acc + nn_28 * 4)[3])
                    : "r"(ar_10[0]), "r"(ar_10[1]), "r"(ar_10[2]), "r"(ar_10[3]), "r"(br_13[0]), "r"(br_13[1]));
            }
        }
        #pragma unroll
        for (int m_2 = 2; m_2 < 3; m_2++) {
            #pragma unroll
            for (int kk_11 = 0; kk_11 < 2; kk_11++) {
                float a_11[4];
                a_11[0] = akkw_s[3456 + m_2 * 16 + grp * 72 + kk_11 * 8 + tq];
                a_11[1] = akkw_s[3456 + m_2 * 16 + (grp + 8) * 72 + kk_11 * 8 + tq];
                a_11[2] = akkw_s[3456 + m_2 * 16 + grp * 72 + kk_11 * 8 + tq + 4];
                a_11[3] = akkw_s[3456 + m_2 * 16 + (grp + 8) * 72 + kk_11 * 8 + tq + 4];
                unsigned int ar_11[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    ar_11[_lp] = __float_as_uint(a_11[_lp + 0]);
                }
                #pragma unroll
                for (int nn_29 = 0; nn_29 < 2; nn_29++) {
                    float b_14[2];
                    b_14[0] = akko_s[m_2 * 16 * 72 + 16 + (kk_11 * 8 + tq) * 72 + nn_29 * 8 + grp];
                    b_14[1] = akko_s[m_2 * 16 * 72 + 16 + (kk_11 * 8 + tq + 4) * 72 + nn_29 * 8 + grp];
                    unsigned int br_14[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br_14[_lp] = __float_as_uint(b_14[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nn_29 * 4)[0]), "+f"((acc + nn_29 * 4)[1]), "+f"((acc + nn_29 * 4)[2]), "+f"((acc + nn_29 * 4)[3])
                        : "r"(ar_11[0]), "r"(ar_11[1]), "r"(ar_11[2]), "r"(ar_11[3]), "r"(br_14[0]), "r"(br_14[1]));
                }
            }
        }
        #pragma unroll
        for (int nn_30 = 0; nn_30 < 2; nn_30++) {
            #pragma unroll
            for (int half_15 = 0; half_15 < 2; half_15++) {
                float v0_12 = acc[nn_30 * 4 + half_15 * 2];
                float v1_12 = acc[nn_30 * 4 + half_15 * 2 + 1];
                float pr_15[2];
                pr_15[0] = v0_12;
                pr_15[1] = v1_12;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)((384 + (grp + half_15 * 8) * 24 + nn_30 * 8 + tq * 2) * 4)), "f"(pr_15[0]), "f"(pr_15[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_12 = 0; kk_12 < 2; kk_12++) {
            float a_12[4];
            a_12[0] = diag_s[960 + grp * 20 + kk_12 * 8 + tq];
            a_12[1] = diag_s[960 + (grp + 8) * 20 + kk_12 * 8 + tq];
            a_12[2] = diag_s[960 + grp * 20 + kk_12 * 8 + tq + 4];
            a_12[3] = diag_s[960 + (grp + 8) * 20 + kk_12 * 8 + tq + 4];
            unsigned int ar_12[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_12[_lp] = __float_as_uint(a_12[_lp + 0]);
            }
            #pragma unroll
            for (int nn_31 = 0; nn_31 < 2; nn_31++) {
                float b_15[2];
                b_15[0] = scr_s[384 + (kk_12 * 8 + tq) * 24 + nn_31 * 8 + grp];
                b_15[1] = scr_s[384 + (kk_12 * 8 + tq + 4) * 24 + nn_31 * 8 + grp];
                unsigned int br_15[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_15[_lp] = __float_as_uint(b_15[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_31 * 4)[0]), "+f"((acc + nn_31 * 4)[1]), "+f"((acc + nn_31 * 4)[2]), "+f"((acc + nn_31 * 4)[3])
                    : "r"(ar_12[0]), "r"(ar_12[1]), "r"(ar_12[2]), "r"(ar_12[3]), "r"(br_15[0]), "r"(br_15[1]));
            }
        }
        #pragma unroll
        for (int nn_32 = 0; nn_32 < 2; nn_32++) {
            #pragma unroll
            for (int half_16 = 0; half_16 < 2; half_16++) {
                float v0_13 = acc[nn_32 * 4 + half_16 * 2];
                float v1_13 = acc[nn_32 * 4 + half_16 * 2 + 1];
                v0_13 = -v0_13;
                v1_13 = -v1_13;
                float pr_16[2];
                pr_16[0] = v0_13;
                pr_16[1] = v1_13;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((3472 + (grp + half_16 * 8) * 72 + nn_32 * 8 + tq * 2) * 4)), "f"(pr_16[0]), "f"(pr_16[1]) : "memory");
            }
        }
        __syncwarp();
    }
    if (warp == 2) {
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_13 = 0; kk_13 < 2; kk_13++) {
            float a_13[4];
            a_13[0] = diag_s[960 + grp * 20 + kk_13 * 8 + tq];
            a_13[1] = diag_s[960 + (grp + 8) * 20 + kk_13 * 8 + tq];
            a_13[2] = diag_s[960 + grp * 20 + kk_13 * 8 + tq + 4];
            a_13[3] = diag_s[960 + (grp + 8) * 20 + kk_13 * 8 + tq + 4];
            unsigned int ar_13[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_13[_lp] = __float_as_uint(a_13[_lp + 0]);
            }
            #pragma unroll
            for (int nn_33 = 0; nn_33 < 2; nn_33++) {
                float b_16[2];
                b_16[0] = akkw_s[3488 + (kk_13 * 8 + tq) * 72 + nn_33 * 8 + grp];
                b_16[1] = akkw_s[3488 + (kk_13 * 8 + tq + 4) * 72 + nn_33 * 8 + grp];
                unsigned int br_16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_16[_lp] = __float_as_uint(b_16[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_33 * 4)[0]), "+f"((acc + nn_33 * 4)[1]), "+f"((acc + nn_33 * 4)[2]), "+f"((acc + nn_33 * 4)[3])
                    : "r"(ar_13[0]), "r"(ar_13[1]), "r"(ar_13[2]), "r"(ar_13[3]), "r"(br_16[0]), "r"(br_16[1]));
            }
        }
        #pragma unroll
        for (int nn_34 = 0; nn_34 < 2; nn_34++) {
            #pragma unroll
            for (int half_17 = 0; half_17 < 2; half_17++) {
                float v0_14 = acc[nn_34 * 4 + half_17 * 2];
                float v1_14 = acc[nn_34 * 4 + half_17 * 2 + 1];
                float pr_17[2];
                pr_17[0] = v0_14;
                pr_17[1] = v1_14;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)((768 + (grp + half_17 * 8) * 24 + nn_34 * 8 + tq * 2) * 4)), "f"(pr_17[0]), "f"(pr_17[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_14 = 0; kk_14 < 2; kk_14++) {
            float a_14[4];
            a_14[0] = scr_s[768 + grp * 24 + kk_14 * 8 + tq];
            a_14[1] = scr_s[768 + (grp + 8) * 24 + kk_14 * 8 + tq];
            a_14[2] = scr_s[768 + grp * 24 + kk_14 * 8 + tq + 4];
            a_14[3] = scr_s[768 + (grp + 8) * 24 + kk_14 * 8 + tq + 4];
            unsigned int ar_14[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_14[_lp] = __float_as_uint(a_14[_lp + 0]);
            }
            #pragma unroll
            for (int nn_35 = 0; nn_35 < 2; nn_35++) {
                float b_17[2];
                b_17[0] = diag_s[640 + (kk_14 * 8 + tq) * 20 + nn_35 * 8 + grp];
                b_17[1] = diag_s[640 + (kk_14 * 8 + tq + 4) * 20 + nn_35 * 8 + grp];
                unsigned int br_17[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_17[_lp] = __float_as_uint(b_17[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_35 * 4)[0]), "+f"((acc + nn_35 * 4)[1]), "+f"((acc + nn_35 * 4)[2]), "+f"((acc + nn_35 * 4)[3])
                    : "r"(ar_14[0]), "r"(ar_14[1]), "r"(ar_14[2]), "r"(ar_14[3]), "r"(br_17[0]), "r"(br_17[1]));
            }
        }
        #pragma unroll
        for (int nn_36 = 0; nn_36 < 2; nn_36++) {
            #pragma unroll
            for (int half_18 = 0; half_18 < 2; half_18++) {
                float v0_15 = acc[nn_36 * 4 + half_18 * 2];
                float v1_15 = acc[nn_36 * 4 + half_18 * 2 + 1];
                v0_15 = -v0_15;
                v1_15 = -v1_15;
                float pr_18[2];
                pr_18[0] = v0_15;
                pr_18[1] = v1_15;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((3488 + (grp + half_18 * 8) * 72 + nn_36 * 8 + tq * 2) * 4)), "f"(pr_18[0]), "f"(pr_18[1]) : "memory");
            }
        }
        __syncwarp();
    }
    __syncthreads();
    #pragma unroll
    for (int r_4 = 0; r_4 < 2; r_4++) {
        int gi = r_4 * 256 + tid_0;
        int t_e = gi / 8;
        int s0 = (gi - t_e * 8) * 8;
        int bi = t_e / 16;
        int bj = s0 / 16;
        int upper = 0;
        if (bj > bi) {
            upper = 1;
        }
        unsigned int src = akko_s_addr + (unsigned int)((t_e * 72 + s0) * 4);
        if (bj == bi) {
            src = diag_s_addr + (unsigned int)((bi * 16 * 20 + (t_e - bi * 16) * 20 + (s0 - bj * 16)) * 4);
        }
        float vk[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vk[0])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 3]))
            : "r"(src));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vk[4])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 3]))
            : "r"(src + 16));
        #pragma unroll
        for (int e_1 = 0; e_1 < 8; e_1++) {
            float vk_e = vk[e_1];
            if (upper != 0) {
                vk_e = 0.0f;
            }
            vk[e_1] = vk_e;
        }
        if (write_ext != 0) {
            if (t_e < clen) {
                long long eidx = abase + (long long)(t_e * ars) + (long long)s0;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(vk[0 + 0], vk[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(vk[0 + 2], vk[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(vk[0 + 4], vk[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(vk[0 + 6], vk[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(akk_ext + eidx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
        if (write_int != 0) {
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2++) {
                float vk2_e = vk[e_2];
                if (t_e >= clen) {
                    vk2_e = 0.0f;
                }
                vk[e_2] = vk2_e;
            }
            long long iidx = ibase + (long long)(t_e * ars) + (long long)s0;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vk[0 + 0], vk[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vk[0 + 2], vk[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vk[0 + 4], vk[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vk[0 + 6], vk[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(akk_int + iidx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

