/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_DAQK_S_OFF 0
#define SMEM_DAQK_S_STAGE_BYTES 17408
#define SMEM_DAQK_S_STRIDE 17408
#define SMEM_DAKK_S_OFF 17408
#define SMEM_DAKK_S_STAGE_BYTES 17408
#define SMEM_DAKK_S_STRIDE 17408
#define SMEM_G_S_OFF 34816
#define SMEM_G_S_STAGE_BYTES 33792
#define SMEM_G_S_STRIDE 33792
#define SMEM_K_S_OFF 68608
#define SMEM_K_S_STAGE_BYTES 17408
#define SMEM_K_S_STRIDE 17408
#define SMEM_Q_S_OFF 86016
#define SMEM_Q_S_STAGE_BYTES 17408
#define SMEM_Q_S_STRIDE 17408
#define SMEM_BETA_S_OFF 103424
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_TOTAL 103680
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_chunk_train_8f902df21bdc21f29ba4(float* __restrict__ dAqk, float* __restrict__ dAkk, float* __restrict__ gk, __nv_bfloat16* __restrict__ k_e, __nv_bfloat16* __restrict__ q_e, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, float* __restrict__ dq_out, float* __restrict__ dk_out, float* __restrict__ dg_out, float* __restrict__ db_out, int num_heads, int num_qk_heads, int group)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* dAqk_s = reinterpret_cast<float*>(smem_raw + 0);
    const int dAqk_s_addr = smem + 0;
    float* dAkk_s = reinterpret_cast<float*>(smem_raw + 17408);
    const int dAkk_s_addr = smem + 17408;
    float* g_s = reinterpret_cast<float*>(smem_raw + 34816);
    const int g_s_addr = smem + 34816;
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int k_s_addr = smem + 68608;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 86016);
    const int q_s_addr = smem + 86016;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 103424);
    const int beta_s_addr = smem + 103424;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    if (tid_0 < 64) {
        beta_s[tid_0] = beta[(row0 + tid_0) * num_heads + head];
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        int e0 = (r * 256 + tid_0) * 8;
        int t_a = e0 / 64;
        int s_a = e0 - t_a * 64;
        int gidx = ((row0 + t_a) * num_heads + head) * 64 + s_a;
        float va[8];
        float vb[8];
        {
            unsigned _ldv8_0_0;
            unsigned _ldv8_0_1;
            unsigned _ldv8_0_2;
            unsigned _ldv8_0_3;
            unsigned _ldv8_0_4;
            unsigned _ldv8_0_5;
            unsigned _ldv8_0_6;
            unsigned _ldv8_0_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(dAqk + (gidx))) : "memory");
            va[0 + 0] = __uint_as_float(_ldv8_0_0);
            va[0 + 1] = __uint_as_float(_ldv8_0_1);
            va[0 + 2] = __uint_as_float(_ldv8_0_2);
            va[0 + 3] = __uint_as_float(_ldv8_0_3);
            va[0 + 4] = __uint_as_float(_ldv8_0_4);
            va[0 + 5] = __uint_as_float(_ldv8_0_5);
            va[0 + 6] = __uint_as_float(_ldv8_0_6);
            va[0 + 7] = __uint_as_float(_ldv8_0_7);
        }
        {
            unsigned _ldv8_1_0;
            unsigned _ldv8_1_1;
            unsigned _ldv8_1_2;
            unsigned _ldv8_1_3;
            unsigned _ldv8_1_4;
            unsigned _ldv8_1_5;
            unsigned _ldv8_1_6;
            unsigned _ldv8_1_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(dAkk + (gidx))) : "memory");
            vb[0 + 0] = __uint_as_float(_ldv8_1_0);
            vb[0 + 1] = __uint_as_float(_ldv8_1_1);
            vb[0 + 2] = __uint_as_float(_ldv8_1_2);
            vb[0 + 3] = __uint_as_float(_ldv8_1_3);
            vb[0 + 4] = __uint_as_float(_ldv8_1_4);
            vb[0 + 5] = __uint_as_float(_ldv8_1_5);
            vb[0 + 6] = __uint_as_float(_ldv8_1_6);
            vb[0 + 7] = __uint_as_float(_ldv8_1_7);
        }
        #pragma unroll
        for (int q4 = 0; q4 < 2; q4++) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAqk_s_addr + (unsigned int)((t_a * 68 + s_a + q4 * 4) * 4)), "f"(va[q4 * 4]), "f"(va[q4 * 4 + 1]), "f"(va[q4 * 4 + 2]), "f"(va[q4 * 4 + 3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAkk_s_addr + (unsigned int)((t_a * 68 + s_a + q4 * 4) * 4)), "f"(vb[q4 * 4]), "f"(vb[q4 * 4 + 1]), "f"(vb[q4 * 4 + 2]), "f"(vb[q4 * 4 + 3]) : "memory");
        }
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 4; r_1++) {
        int e1 = (r_1 * 256 + tid_0) * 8;
        int t_b = e1 / 128;
        int d_b = e1 - t_b * 128;
        int gidx2 = ((row0 + t_b) * num_heads + head) * 128 + d_b;
        float gv[8];
        float kv[8];
        float qv[8];
        {
            unsigned _ldv8_2_0;
            unsigned _ldv8_2_1;
            unsigned _ldv8_2_2;
            unsigned _ldv8_2_3;
            unsigned _ldv8_2_4;
            unsigned _ldv8_2_5;
            unsigned _ldv8_2_6;
            unsigned _ldv8_2_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(gk + (gidx2))) : "memory");
            gv[0 + 0] = __uint_as_float(_ldv8_2_0);
            gv[0 + 1] = __uint_as_float(_ldv8_2_1);
            gv[0 + 2] = __uint_as_float(_ldv8_2_2);
            gv[0 + 3] = __uint_as_float(_ldv8_2_3);
            gv[0 + 4] = __uint_as_float(_ldv8_2_4);
            gv[0 + 5] = __uint_as_float(_ldv8_2_5);
            gv[0 + 6] = __uint_as_float(_ldv8_2_6);
            gv[0 + 7] = __uint_as_float(_ldv8_2_7);
        }
        int qk_idx2 = ((row0 + t_b) * num_qk_heads + qk_head) * 128 + d_b;
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(k_e + qk_idx2);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&kv[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kv[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q_e + qk_idx2);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&qv[0 + _blk * 8 + _pair * 2])[0]), "=f"((&qv[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        uint32_t kv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kv[_lp*2 + 0], kv[_lp*2+1 + 0]));
            kv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t qv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qv[_lp*2 + 0], qv[_lp*2+1 + 0]));
            qv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int q4_1 = 0; q4_1 < 2; q4_1++) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(g_s_addr + (unsigned int)((t_b * 132 + d_b + q4_1 * 4) * 4)), "f"(gv[q4_1 * 4]), "f"(gv[q4_1 * 4 + 1]), "f"(gv[q4_1 * 4 + 2]), "f"(gv[q4_1 * 4 + 3]) : "memory");
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(k_s_addr + (unsigned int)((t_b * 136 + d_b) * 2)), "r"(kv_bf16[0]), "r"(kv_bf16[1]), "r"(kv_bf16[2]), "r"(kv_bf16[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_s_addr + (unsigned int)((t_b * 136 + d_b) * 2)), "r"(qv_bf16[0]), "r"(qv_bf16[1]), "r"(qv_bf16[2]), "r"(qv_bf16[3]) : "memory");
    }
    __syncthreads();
    int i = warp % 4;
    int kb_half = warp / 4;
    int grp = lane / 4;
    int tq = lane % 4;
    int r0 = i * 16;
    float db_lo = 0.0f;
    float db_hi = 0.0f;
    #pragma unroll
    for (int kbi = 0; kbi < 2; kbi++) {
        int kb = kb_half * 2 + kbi;
        int c_base = kb * 32;
        float aq_off[16];
        float ak_off[16];
        float aq_dg[16];
        float ak_dg[16];
        float at_off[16];
        float at_dg[16];
        aq_off[0] = 0.0f;
        aq_off[1] = 0.0f;
        aq_off[2] = 0.0f;
        aq_off[3] = 0.0f;
        aq_off[4] = 0.0f;
        aq_off[5] = 0.0f;
        aq_off[6] = 0.0f;
        aq_off[7] = 0.0f;
        aq_off[8] = 0.0f;
        aq_off[9] = 0.0f;
        aq_off[10] = 0.0f;
        aq_off[11] = 0.0f;
        aq_off[12] = 0.0f;
        aq_off[13] = 0.0f;
        aq_off[14] = 0.0f;
        aq_off[15] = 0.0f;
        ak_off[0] = 0.0f;
        ak_off[1] = 0.0f;
        ak_off[2] = 0.0f;
        ak_off[3] = 0.0f;
        ak_off[4] = 0.0f;
        ak_off[5] = 0.0f;
        ak_off[6] = 0.0f;
        ak_off[7] = 0.0f;
        ak_off[8] = 0.0f;
        ak_off[9] = 0.0f;
        ak_off[10] = 0.0f;
        ak_off[11] = 0.0f;
        ak_off[12] = 0.0f;
        ak_off[13] = 0.0f;
        ak_off[14] = 0.0f;
        ak_off[15] = 0.0f;
        aq_dg[0] = 0.0f;
        aq_dg[1] = 0.0f;
        aq_dg[2] = 0.0f;
        aq_dg[3] = 0.0f;
        aq_dg[4] = 0.0f;
        aq_dg[5] = 0.0f;
        aq_dg[6] = 0.0f;
        aq_dg[7] = 0.0f;
        aq_dg[8] = 0.0f;
        aq_dg[9] = 0.0f;
        aq_dg[10] = 0.0f;
        aq_dg[11] = 0.0f;
        aq_dg[12] = 0.0f;
        aq_dg[13] = 0.0f;
        aq_dg[14] = 0.0f;
        aq_dg[15] = 0.0f;
        ak_dg[0] = 0.0f;
        ak_dg[1] = 0.0f;
        ak_dg[2] = 0.0f;
        ak_dg[3] = 0.0f;
        ak_dg[4] = 0.0f;
        ak_dg[5] = 0.0f;
        ak_dg[6] = 0.0f;
        ak_dg[7] = 0.0f;
        ak_dg[8] = 0.0f;
        ak_dg[9] = 0.0f;
        ak_dg[10] = 0.0f;
        ak_dg[11] = 0.0f;
        ak_dg[12] = 0.0f;
        ak_dg[13] = 0.0f;
        ak_dg[14] = 0.0f;
        ak_dg[15] = 0.0f;
        at_off[0] = 0.0f;
        at_off[1] = 0.0f;
        at_off[2] = 0.0f;
        at_off[3] = 0.0f;
        at_off[4] = 0.0f;
        at_off[5] = 0.0f;
        at_off[6] = 0.0f;
        at_off[7] = 0.0f;
        at_off[8] = 0.0f;
        at_off[9] = 0.0f;
        at_off[10] = 0.0f;
        at_off[11] = 0.0f;
        at_off[12] = 0.0f;
        at_off[13] = 0.0f;
        at_off[14] = 0.0f;
        at_off[15] = 0.0f;
        at_dg[0] = 0.0f;
        at_dg[1] = 0.0f;
        at_dg[2] = 0.0f;
        at_dg[3] = 0.0f;
        at_dg[4] = 0.0f;
        at_dg[5] = 0.0f;
        at_dg[6] = 0.0f;
        at_dg[7] = 0.0f;
        at_dg[8] = 0.0f;
        at_dg[9] = 0.0f;
        at_dg[10] = 0.0f;
        at_dg[11] = 0.0f;
        at_dg[12] = 0.0f;
        at_dg[13] = 0.0f;
        at_dg[14] = 0.0f;
        at_dg[15] = 0.0f;
        float gn_b[4];
        float gmid_b[4];
        float gnl_b[4];
        #pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            int kd_b = c_base + nn * 8 + grp;
            gn_b[nn] = g_s[r0 * 132 + kd_b];
            gmid_b[nn] = g_s[(r0 + 8) * 132 + kd_b];
            gnl_b[nn] = g_s[(r0 + 16 - 1) * 132 + kd_b];
        }
        #pragma unroll 1
        for (int j = 0; j < i; j++) {
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int s0 = j * 16 + kk * 8;
                float aqv[4];
                float akv[4];
                #pragma unroll
                for (int wd = 0; wd < 4; wd++) {
                    int rr = r0 + grp + wd % 2 * 8;
                    int cc = s0 + tq + wd / 2 * 4;
                    aqv[wd] = dAqk_s[rr * 68 + cc];
                    akv[wd] = dAkk_s[rr * 68 + cc];
                }
                unsigned int aqr[4];
                unsigned int akr[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    aqr[_lp] = __float_as_uint(aqv[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    akr[_lp] = __float_as_uint(akv[_lp + 0]);
                }
                #pragma unroll
                for (int nn_1 = 0; nn_1 < 4; nn_1++) {
                    int kd = c_base + nn_1 * 8 + grp;
                    float bv[2];
                    #pragma unroll
                    for (int wd_1 = 0; wd_1 < 2; wd_1++) {
                        int ss = s0 + tq + wd_1 * 4;
                        __nv_bfloat16 _k_s_v0 = k_s[ss * 136 + kd];
                        float _cvt_f32_0 = __bfloat162float(_k_s_v0);
                        float _exp2_0 = approx_exp2(gn_b[nn_1] - g_s[ss * 132 + kd]);
                        bv[wd_1] = _cvt_f32_0 * _exp2_0;
                    }
                    unsigned int br[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br[_lp] = __float_as_uint(bv[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((aq_off + nn_1 * 4)[0]), "+f"((aq_off + nn_1 * 4)[1]), "+f"((aq_off + nn_1 * 4)[2]), "+f"((aq_off + nn_1 * 4)[3])
                        : "r"(aqr[0]), "r"(aqr[1]), "r"(aqr[2]), "r"(aqr[3]), "r"(br[0]), "r"(br[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((ak_off + nn_1 * 4)[0]), "+f"((ak_off + nn_1 * 4)[1]), "+f"((ak_off + nn_1 * 4)[2]), "+f"((ak_off + nn_1 * 4)[3])
                        : "r"(akr[0]), "r"(akr[1]), "r"(akr[2]), "r"(akr[3]), "r"(br[0]), "r"(br[1]));
                }
            }
        }
        #pragma unroll
        for (int kk_1 = 0; kk_1 < 2; kk_1++) {
            int s0d = r0 + kk_1 * 8;
            float aqd[4];
            float akd[4];
            #pragma unroll
            for (int wd_2 = 0; wd_2 < 4; wd_2++) {
                int rr_1 = r0 + grp + wd_2 % 2 * 8;
                int cc_1 = s0d + tq + wd_2 / 2 * 4;
                float va_ = dAqk_s[rr_1 * 68 + cc_1];
                float vk_ = dAkk_s[rr_1 * 68 + cc_1];
                if (cc_1 > rr_1) {
                    va_ = 0.0f;
                    vk_ = 0.0f;
                }
                aqd[wd_2] = va_;
                akd[wd_2] = vk_;
            }
            unsigned int aqdr[4];
            unsigned int akdr[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aqdr[_lp] = __float_as_uint(aqd[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                akdr[_lp] = __float_as_uint(akd[_lp + 0]);
            }
            #pragma unroll
            for (int nn_2 = 0; nn_2 < 4; nn_2++) {
                int kd_1 = c_base + nn_2 * 8 + grp;
                float bvd[2];
                #pragma unroll
                for (int wd_3 = 0; wd_3 < 2; wd_3++) {
                    int ss_1 = s0d + tq + wd_3 * 4;
                    __nv_bfloat16 _k_s_v1 = k_s[ss_1 * 136 + kd_1];
                    float _cvt_f32_1 = __bfloat162float(_k_s_v1);
                    float _exp2_1 = approx_exp2(gmid_b[nn_2] - g_s[ss_1 * 132 + kd_1]);
                    bvd[wd_3] = _cvt_f32_1 * _exp2_1;
                }
                unsigned int brd[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    brd[_lp] = __float_as_uint(bvd[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((aq_dg + nn_2 * 4)[0]), "+f"((aq_dg + nn_2 * 4)[1]), "+f"((aq_dg + nn_2 * 4)[2]), "+f"((aq_dg + nn_2 * 4)[3])
                    : "r"(aqdr[0]), "r"(aqdr[1]), "r"(aqdr[2]), "r"(aqdr[3]), "r"(brd[0]), "r"(brd[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((ak_dg + nn_2 * 4)[0]), "+f"((ak_dg + nn_2 * 4)[1]), "+f"((ak_dg + nn_2 * 4)[2]), "+f"((ak_dg + nn_2 * 4)[3])
                    : "r"(akdr[0]), "r"(akdr[1]), "r"(akdr[2]), "r"(akdr[3]), "r"(brd[0]), "r"(brd[1]));
            }
        }
        #pragma unroll 1
        for (int j_1 = i + 1; j_1 < 4; j_1++) {
            #pragma unroll
            for (int kk_2 = 0; kk_2 < 2; kk_2++) {
                int s0t = j_1 * 16 + kk_2 * 8;
                float atq[4];
                float atk[4];
                #pragma unroll
                for (int wd_4 = 0; wd_4 < 4; wd_4++) {
                    int tt = r0 + grp + wd_4 % 2 * 8;
                    int ss2 = s0t + tq + wd_4 / 2 * 4;
                    atq[wd_4] = dAqk_s[ss2 * 68 + tt];
                    atk[wd_4] = dAkk_s[ss2 * 68 + tt];
                }
                unsigned int atqr[4];
                unsigned int atkr[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    atqr[_lp] = __float_as_uint(atq[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    atkr[_lp] = __float_as_uint(atk[_lp + 0]);
                }
                #pragma unroll
                for (int nn_3 = 0; nn_3 < 4; nn_3++) {
                    int kd_2 = c_base + nn_3 * 8 + grp;
                    float b1[2];
                    float b2[2];
                    #pragma unroll
                    for (int wd_5 = 0; wd_5 < 2; wd_5++) {
                        int ss_2 = s0t + tq + wd_5 * 4;
                        float _exp2_2 = approx_exp2(g_s[ss_2 * 132 + kd_2] - gnl_b[nn_3]);
                        float ef = _exp2_2;
                        __nv_bfloat16 _q_s_v2 = q_s[ss_2 * 136 + kd_2];
                        float _cvt_f32_2 = __bfloat162float(_q_s_v2);
                        b1[wd_5] = _cvt_f32_2 * ef;
                        __nv_bfloat16 _kraw__kb_s_v3 = k_s[ss_2 * 136 + kd_2];
                        float _cvt_f32_3 = __bfloat162float(_kraw__kb_s_v3);
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_cvt_f32_3 * beta_s[ss_2]);
                        __nv_bfloat16 _kb_s_v3 = _cvt_bf16_0;
                        float _cvt_f32_4 = __bfloat162float(_kb_s_v3);
                        b2[wd_5] = _cvt_f32_4 * ef;
                    }
                    unsigned int b1r[2];
                    unsigned int b2r[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        b1r[_lp] = __float_as_uint(b1[_lp + 0]);
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        b2r[_lp] = __float_as_uint(b2[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((at_off + nn_3 * 4)[0]), "+f"((at_off + nn_3 * 4)[1]), "+f"((at_off + nn_3 * 4)[2]), "+f"((at_off + nn_3 * 4)[3])
                        : "r"(atqr[0]), "r"(atqr[1]), "r"(atqr[2]), "r"(atqr[3]), "r"(b1r[0]), "r"(b1r[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((at_off + nn_3 * 4)[0]), "+f"((at_off + nn_3 * 4)[1]), "+f"((at_off + nn_3 * 4)[2]), "+f"((at_off + nn_3 * 4)[3])
                        : "r"(atkr[0]), "r"(atkr[1]), "r"(atkr[2]), "r"(atkr[3]), "r"(b2r[0]), "r"(b2r[1]));
                }
            }
        }
        #pragma unroll
        for (int kk_3 = 0; kk_3 < 2; kk_3++) {
            int s0e = r0 + kk_3 * 8;
            float aeq[4];
            float aek[4];
            #pragma unroll
            for (int wd_6 = 0; wd_6 < 4; wd_6++) {
                int tt_1 = r0 + grp + wd_6 % 2 * 8;
                int ss3 = s0e + tq + wd_6 / 2 * 4;
                float vq_ = dAqk_s[ss3 * 68 + tt_1];
                float vk2_ = dAkk_s[ss3 * 68 + tt_1];
                if (ss3 < tt_1) {
                    vq_ = 0.0f;
                    vk2_ = 0.0f;
                }
                aeq[wd_6] = vq_;
                aek[wd_6] = vk2_;
            }
            unsigned int aeqr[4];
            unsigned int aekr[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aeqr[_lp] = __float_as_uint(aeq[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aekr[_lp] = __float_as_uint(aek[_lp + 0]);
            }
            #pragma unroll
            for (int nn_4 = 0; nn_4 < 4; nn_4++) {
                int kd_3 = c_base + nn_4 * 8 + grp;
                float b1d[2];
                float b2d[2];
                #pragma unroll
                for (int wd_7 = 0; wd_7 < 2; wd_7++) {
                    int ss_3 = s0e + tq + wd_7 * 4;
                    float _exp2_3 = approx_exp2(g_s[ss_3 * 132 + kd_3] - gmid_b[nn_4]);
                    float efd = _exp2_3;
                    __nv_bfloat16 _q_s_v4 = q_s[ss_3 * 136 + kd_3];
                    float _cvt_f32_5 = __bfloat162float(_q_s_v4);
                    b1d[wd_7] = _cvt_f32_5 * efd;
                    __nv_bfloat16 _kraw__kb_s_v5 = k_s[ss_3 * 136 + kd_3];
                    float _cvt_f32_6 = __bfloat162float(_kraw__kb_s_v5);
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_cvt_f32_6 * beta_s[ss_3]);
                    __nv_bfloat16 _kb_s_v5 = _cvt_bf16_1;
                    float _cvt_f32_7 = __bfloat162float(_kb_s_v5);
                    b2d[wd_7] = _cvt_f32_7 * efd;
                }
                unsigned int b1dr[2];
                unsigned int b2dr[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b1dr[_lp] = __float_as_uint(b1d[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b2dr[_lp] = __float_as_uint(b2d[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_dg + nn_4 * 4)[0]), "+f"((at_dg + nn_4 * 4)[1]), "+f"((at_dg + nn_4 * 4)[2]), "+f"((at_dg + nn_4 * 4)[3])
                    : "r"(aeqr[0]), "r"(aeqr[1]), "r"(aeqr[2]), "r"(aeqr[3]), "r"(b1dr[0]), "r"(b1dr[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_dg + nn_4 * 4)[0]), "+f"((at_dg + nn_4 * 4)[1]), "+f"((at_dg + nn_4 * 4)[2]), "+f"((at_dg + nn_4 * 4)[3])
                    : "r"(aekr[0]), "r"(aekr[1]), "r"(aekr[2]), "r"(aekr[3]), "r"(b2dr[0]), "r"(b2dr[1]));
            }
        }
        #pragma unroll
        for (int nn_5 = 0; nn_5 < 4; nn_5++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int t_loc = r0 + grp + half * 8;
                int grow = row0 + t_loc;
                int gbase = (grow * num_heads + head) * 128;
                int col0 = c_base + nn_5 * 8 + tq * 2;
                float dqf[2];
                float dkf[2];
                float dgf[2];
                {
                    float2 _v2_5 = *reinterpret_cast<const float2*>(dq_f + gbase + col0);
                    dqf[0] = _v2_5.x;
                    dqf[0 + 1] = _v2_5.y;
                }
                {
                    float2 _v2_6 = *reinterpret_cast<const float2*>(dk_f + gbase + col0);
                    dkf[0] = _v2_6.x;
                    dkf[0 + 1] = _v2_6.y;
                }
                {
                    float2 _v2_7 = *reinterpret_cast<const float2*>(dg_f + gbase + col0);
                    dgf[0] = _v2_7.x;
                    dgf[0 + 1] = _v2_7.y;
                }
                float dq_o[2];
                float dk_o[2];
                float dg_o[2];
                float bt_t = beta_s[t_loc];
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int col = col0 + e;
                    const int reg = nn_5 * 4 + half * 2 + e;
                    float gtk = g_s[t_loc * 132 + col];
                    float gn_c = g_s[r0 * 132 + col];
                    float gmid_c = g_s[(r0 + 8) * 132 + col];
                    float gnl_c = g_s[(r0 + 16 - 1) * 132 + col];
                    float _exp2_4 = approx_exp2(gtk - gn_c);
                    float e_off = _exp2_4;
                    float _exp2_5 = approx_exp2(gtk - gmid_c);
                    float e_dg = _exp2_5;
                    float dq2 = aq_off[reg] * e_off + aq_dg[reg] * e_dg;
                    float dk2 = ak_off[reg] * e_off + ak_dg[reg] * e_dg;
                    __nv_bfloat16 _k_s_v6 = k_s[t_loc * 136 + col];
                    float _cvt_f32_8 = __bfloat162float(_k_s_v6);
                    float kt = _cvt_f32_8;
                    if (half == 0) {
                        db_lo = db_lo + dk2 * kt;
                    } else {
                        db_hi = db_hi + dk2 * kt;
                    }
                    dk2 = dk2 * bt_t;
                    __nv_bfloat16 _q_s_v7 = q_s[t_loc * 136 + col];
                    float _cvt_f32_9 = __bfloat162float(_q_s_v7);
                    float dg2 = _cvt_f32_9 * dq2;
                    float _exp2_6 = approx_exp2(gnl_c - gtk);
                    float _exp2_7 = approx_exp2(gmid_c - gtk);
                    float dkt = at_off[reg] * _exp2_6 + at_dg[reg] * _exp2_7;
                    dq_o[e] = dq2 + dqf[e];
                    dg_o[e] = dg2 + (dk2 - dkt) * kt + dgf[e];
                    dk_o[e] = dk2 + dkf[e] + dkt;
                }
                {
                    float2 _v2 = make_float2(dq_o[0 + 0], dq_o[0 + 1]);
                    *reinterpret_cast<float2*>(dq_out + (gbase + col0) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(dk_o[0 + 0], dk_o[0 + 1]);
                    *reinterpret_cast<float2*>(dk_out + (gbase + col0) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(dg_o[0 + 0], dg_o[0 + 1]);
                    *reinterpret_cast<float2*>(dg_out + (gbase + col0) + 0) = _v2;
                }
            }
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, db_lo, 1);
    db_lo = db_lo + _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, db_lo, 2);
    db_lo = db_lo + _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, db_hi, 1);
    db_hi = db_hi + _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, db_hi, 2);
    db_hi = db_hi + _shfl_xor_3;
    __syncthreads();
    if (tq == 0) {
        if (kb_half == 1) {
            beta_s[r0 + grp] = db_lo;
            beta_s[r0 + grp + 8] = db_hi;
        }
    }
    __syncthreads();
    if (tq == 0) {
        if (kb_half == 0) {
            int idx_lo = (row0 + r0 + grp) * num_heads + head;
            int idx_hi = (row0 + r0 + grp + 8) * num_heads + head;
            db_out[idx_lo] = db_f[idx_lo] + (db_lo + beta_s[r0 + grp]);
            db_out[idx_hi] = db_f[idx_hi] + (db_hi + beta_s[r0 + grp + 8]);
        }
    }
}

} // extern "C"
