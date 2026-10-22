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
#define SMEM_DAQK_S_OFF 0
#define SMEM_DAQK_S_STAGE_BYTES 18432
#define SMEM_DAQK_S_STRIDE 18432
#define SMEM_DAKK_S_OFF 18432
#define SMEM_DAKK_S_STAGE_BYTES 18432
#define SMEM_DAKK_S_STRIDE 18432
#define SMEM_REC_S_OFF 36864
#define SMEM_REC_S_STAGE_BYTES 67584
#define SMEM_REC_S_STRIDE 67584
#define SMEM_BETA_S_OFF 104448
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_TOTAL 104704
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_30cb08c6a4c77c18cc5c(float* __restrict__ dAqk, float* __restrict__ dAkk, float* __restrict__ gk, __nv_bfloat16* __restrict__ k_e, __nv_bfloat16* __restrict__ q_e, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_out, float* __restrict__ db_out, int num_heads, int num_qk_heads, int group)
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
    float* dAqk_s = reinterpret_cast<float*>(smem_raw + 0);
    const int dAqk_s_addr = smem + 0;
    float* dAkk_s = reinterpret_cast<float*>(smem_raw + 18432);
    const int dAkk_s_addr = smem + 18432;
    unsigned int* rec_s = reinterpret_cast<unsigned int*>(smem_raw + 36864);
    const int rec_s_addr = smem + 36864;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 104448);
    const int beta_s_addr = smem + 104448;
    int clen_setup = chunk_len[blockIdx.x];
    if (__syncthreads_and(clen_setup == 0)) return;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    int rs = num_heads * 128;
    int qrs = num_qk_heads * 128;
    long long gbase_c = ((long long)row0 * (long long)num_heads + (long long)head) * 128;
    long long qbase_c = ((long long)row0 * (long long)num_qk_heads + (long long)qk_head) * 128;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    if (tid_0 < 64) {
        beta_s[tid_0] = beta[(row0 + tid_0) * num_heads + head];
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int e0 = (r * 256 + tid_0) * 4;
        int t_a = e0 / 64;
        int s_a = e0 - t_a * 64;
        int gidx = ((row0 + t_a) * num_heads + head) * 64 + s_a;
        float va[4];
        float vb[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dAqk + gidx);
            va[0 + 0] = _v4.x;
            va[0 + 1] = _v4.y;
            va[0 + 2] = _v4.z;
            va[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dAkk + gidx);
            vb[0 + 0] = _v4.x;
            vb[0 + 1] = _v4.y;
            vb[0 + 2] = _v4.z;
            vb[0 + 3] = _v4.w;
        }
        int pos_a = t_a * 72 + (s_a ^ t_a & 4);
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAqk_s_addr + (unsigned int)(pos_a * 4)), "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]) : "memory");
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAkk_s_addr + (unsigned int)(pos_a * 4)), "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]) : "memory");
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 16; r_1++) {
        int e1 = (r_1 * 256 + tid_0) * 2;
        int t_b = e1 / 128;
        int d_b = e1 - t_b * 128;
        long long gidx2 = gbase_c + (long long)(t_b * rs + d_b);
        float gv[2];
        float kv[2];
        float qv[2];
        {
            float2 _v2_2 = *reinterpret_cast<const float2*>(gk + gidx2);
            gv[0] = _v2_2.x;
            gv[0 + 1] = _v2_2.y;
        }
        long long qk_idx2 = qbase_c + (long long)(t_b * qrs + d_b);
        {
            uint32_t _bf16x2_bits_3;
            _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(k_e + qk_idx2);
            (&kv[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) << 16);
            (&kv[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) & 0xffff0000u);
        }
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(q_e + qk_idx2);
            (&qv[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) << 16);
            (&qv[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) & 0xffff0000u);
        }
        unsigned int g0w = __as_u32(gv[0]);
        unsigned int g1w = __as_u32(gv[1]);
        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(kv[0], qv[0]));
        unsigned int kq0 = __as_u32(_bf16x2_0);
        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(kv[1], qv[1]));
        unsigned int kq1 = __as_u32(_bf16x2_1);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rec_s_addr + (unsigned int)((t_b * 132 + d_b) * 8)), "r"(g0w), "r"(kq0), "r"(g1w), "r"(kq1) : "memory");
    }
    __syncthreads();
    int i = warp % 4;
    int kb_half = warp / 4;
    int grp = lane / 4;
    int tq = lane % 4;
    int r0 = i * 16;
    int sw_g = grp & 4;
    float db_lo = 0.0f;
    float db_hi = 0.0f;
    #pragma unroll
    for (int kbi = 0; kbi < 2; kbi++) {
        int kb = kb_half * 2 + kbi;
        int c_base = kb * 32;
        float gn_b[4];
        float gc_b[4];
        float gnl_b[4];
        #pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            int kd_b = c_base + nn * 8 + grp;
            unsigned int rn_b[2];
            unsigned int rl_b[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rn_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&rn_b[(0) + 1]))
                : "r"(rec_s_addr + (unsigned int)((r0 * 132 + kd_b) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rl_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&rl_b[(0) + 1]))
                : "r"(rec_s_addr + (unsigned int)(((r0 + 16 - 1) * 132 + kd_b) * 8)));
            gn_b[nn] = __uint_as_float(rn_b[0]);
            gc_b[nn] = gn_b[nn] - 21.640425324000002f;
            gnl_b[nn] = __uint_as_float(rl_b[0]);
        }
        float at_off[16];
        float at_dg[16];
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
        #pragma unroll 1
        for (int j = i + 1; j < 4; j++) {
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int s0t = j * 16 + kk * 8;
                float atq[4];
                float atk[4];
                #pragma unroll
                for (int wd = 0; wd < 4; wd++) {
                    int tt = r0 + grp + wd % 2 * 8 ^ 4 * (wd / 2);
                    int ss2 = s0t + tq + wd / 2 * 4;
                    atq[wd] = dAqk_s[ss2 * 72 + tt];
                    atk[wd] = dAkk_s[ss2 * 72 + tt];
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
                float bt_k[2];
                #pragma unroll
                for (int wd_1 = 0; wd_1 < 2; wd_1++) {
                    bt_k[wd_1] = beta_s[s0t + tq + wd_1 * 4];
                }
                #pragma unroll
                for (int nn_1 = 0; nn_1 < 4; nn_1++) {
                    int kd = c_base + nn_1 * 8 + grp;
                    float b1[2];
                    float b2[2];
                    #pragma unroll
                    for (int wd_2 = 0; wd_2 < 2; wd_2++) {
                        int ss = s0t + tq + wd_2 * 4;
                        unsigned int rec[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rec[0])), "=r"(*reinterpret_cast<uint32_t*>(&rec[(0) + 1]))
                            : "r"(rec_s_addr + (unsigned int)((ss * 132 + kd) * 8)));
                        float _exp2_0 = approx_exp2(__uint_as_float(rec[0]) - gnl_b[nn_1]);
                        float ef = _exp2_0;
                        b1[wd_2] = __uint_as_float(rec[1] & 4294901760u) * ef;
                        __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)(__uint_as_float(rec[1] << 16) * bt_k[wd_2]);
                        __nv_bfloat16 _kb_s_v3 = _cvt_bf16_0;
                        float _cvt_f32_0 = (float)_kb_s_v3;
                        b2[wd_2] = _cvt_f32_0 * ef;
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
                        : "+f"((at_off + nn_1 * 4)[0]), "+f"((at_off + nn_1 * 4)[1]), "+f"((at_off + nn_1 * 4)[2]), "+f"((at_off + nn_1 * 4)[3])
                        : "r"(atqr[0]), "r"(atqr[1]), "r"(atqr[2]), "r"(atqr[3]), "r"(b1r[0]), "r"(b1r[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((at_off + nn_1 * 4)[0]), "+f"((at_off + nn_1 * 4)[1]), "+f"((at_off + nn_1 * 4)[2]), "+f"((at_off + nn_1 * 4)[3])
                        : "r"(atkr[0]), "r"(atkr[1]), "r"(atkr[2]), "r"(atkr[3]), "r"(b2r[0]), "r"(b2r[1]));
                }
            }
        }
        #pragma unroll
        for (int kk_1 = 0; kk_1 < 2; kk_1++) {
            int s0e = r0 + kk_1 * 8;
            float aeq[4];
            float aek[4];
            #pragma unroll
            for (int wd_3 = 0; wd_3 < 4; wd_3++) {
                int tt_1 = r0 + grp + wd_3 % 2 * 8;
                int tt_s = tt_1 ^ 4 * (wd_3 / 2);
                int ss3 = s0e + tq + wd_3 / 2 * 4;
                float vq_ = dAqk_s[ss3 * 72 + tt_s];
                float vk2_ = dAkk_s[ss3 * 72 + tt_s];
                if (ss3 < tt_1) {
                    vq_ = 0.0f;
                    vk2_ = 0.0f;
                }
                aeq[wd_3] = vq_;
                aek[wd_3] = vk2_;
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
            float bt_d[2];
            #pragma unroll
            for (int wd_4 = 0; wd_4 < 2; wd_4++) {
                bt_d[wd_4] = beta_s[s0e + tq + wd_4 * 4];
            }
            #pragma unroll
            for (int nn_2 = 0; nn_2 < 4; nn_2++) {
                int kd_1 = c_base + nn_2 * 8 + grp;
                float b1d[2];
                float b2d[2];
                #pragma unroll
                for (int wd_5 = 0; wd_5 < 2; wd_5++) {
                    int ss_1 = s0e + tq + wd_5 * 4;
                    unsigned int recd[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&recd[0])), "=r"(*reinterpret_cast<uint32_t*>(&recd[(0) + 1]))
                        : "r"(rec_s_addr + (unsigned int)((ss_1 * 132 + kd_1) * 8)));
                    float _exp2_1 = approx_exp2(__uint_as_float(recd[0]) - gc_b[nn_2]);
                    float efd = _exp2_1;
                    b1d[wd_5] = __uint_as_float(recd[1] & 4294901760u) * efd;
                    __nv_bfloat16 _cvt_bf16_1 = (__nv_bfloat16)(__uint_as_float(recd[1] << 16) * bt_d[wd_5]);
                    __nv_bfloat16 _kb_s_v5 = _cvt_bf16_1;
                    float _cvt_f32_1 = (float)_kb_s_v5;
                    b2d[wd_5] = _cvt_f32_1 * efd;
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
                    : "+f"((at_dg + nn_2 * 4)[0]), "+f"((at_dg + nn_2 * 4)[1]), "+f"((at_dg + nn_2 * 4)[2]), "+f"((at_dg + nn_2 * 4)[3])
                    : "r"(aeqr[0]), "r"(aeqr[1]), "r"(aeqr[2]), "r"(aeqr[3]), "r"(b1dr[0]), "r"(b1dr[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_dg + nn_2 * 4)[0]), "+f"((at_dg + nn_2 * 4)[1]), "+f"((at_dg + nn_2 * 4)[2]), "+f"((at_dg + nn_2 * 4)[3])
                    : "r"(aekr[0]), "r"(aekr[1]), "r"(aekr[2]), "r"(aekr[3]), "r"(b2dr[0]), "r"(b2dr[1]));
            }
        }
        float dkt[16];
        #pragma unroll
        for (int nn_3 = 0; nn_3 < 4; nn_3++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int t_loc_f = r0 + grp + half * 8;
                int col0_f = c_base + nn_3 * 8 + tq * 2;
                unsigned int rt_f[4];
                unsigned int rn_f[4];
                unsigned int rl_f[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rt_f[0])), "=r"(*reinterpret_cast<uint32_t*>(&rt_f[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rt_f[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rt_f[(0) + 3]))
                    : "r"(rec_s_addr + (unsigned int)((t_loc_f * 132 + col0_f) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rn_f[0])), "=r"(*reinterpret_cast<uint32_t*>(&rn_f[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rn_f[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rn_f[(0) + 3]))
                    : "r"(rec_s_addr + (unsigned int)((r0 * 132 + col0_f) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rl_f[0])), "=r"(*reinterpret_cast<uint32_t*>(&rl_f[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rl_f[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rl_f[(0) + 3]))
                    : "r"(rec_s_addr + (unsigned int)(((r0 + 16 - 1) * 132 + col0_f) * 8)));
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int reg_f = nn_3 * 4 + half * 2 + e;
                    float gtk_f = __uint_as_float(rt_f[2 * e]);
                    float gc_f = __uint_as_float(rn_f[2 * e]) - 21.640425324000002f;
                    float gnl_f = __uint_as_float(rl_f[2 * e]);
                    float _exp2_2 = approx_exp2(gnl_f - gtk_f);
                    float _exp2_3 = approx_exp2(gc_f - gtk_f);
                    dkt[reg_f] = __fmaf_rn(at_off[reg_f], _exp2_2, at_dg[reg_f] * _exp2_3);
                }
            }
        }
        float aq_off[16];
        float ak_off[16];
        float aq_dg[16];
        float ak_dg[16];
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
        #pragma unroll 1
        for (int j_1 = 0; j_1 < i; j_1++) {
            #pragma unroll
            for (int kk_2 = 0; kk_2 < 2; kk_2++) {
                int s0 = j_1 * 16 + kk_2 * 8;
                float aqv[4];
                float akv[4];
                #pragma unroll
                for (int wd_6 = 0; wd_6 < 4; wd_6++) {
                    int rr = r0 + grp + wd_6 % 2 * 8;
                    int cc = s0 + tq + wd_6 / 2 * 4 ^ sw_g;
                    aqv[wd_6] = dAqk_s[rr * 72 + cc];
                    akv[wd_6] = dAkk_s[rr * 72 + cc];
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
                for (int nn_4 = 0; nn_4 < 4; nn_4++) {
                    int kd_2 = c_base + nn_4 * 8 + grp;
                    float bv[2];
                    #pragma unroll
                    for (int wd_7 = 0; wd_7 < 2; wd_7++) {
                        int ss_2 = s0 + tq + wd_7 * 4;
                        unsigned int rec1[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rec1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rec1[(0) + 1]))
                            : "r"(rec_s_addr + (unsigned int)((ss_2 * 132 + kd_2) * 8)));
                        float _exp2_4 = approx_exp2(gn_b[nn_4] - __uint_as_float(rec1[0]));
                        bv[wd_7] = __uint_as_float(rec1[1] << 16) * _exp2_4;
                    }
                    unsigned int br[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br[_lp] = __float_as_uint(bv[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((aq_off + nn_4 * 4)[0]), "+f"((aq_off + nn_4 * 4)[1]), "+f"((aq_off + nn_4 * 4)[2]), "+f"((aq_off + nn_4 * 4)[3])
                        : "r"(aqr[0]), "r"(aqr[1]), "r"(aqr[2]), "r"(aqr[3]), "r"(br[0]), "r"(br[1]));
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((ak_off + nn_4 * 4)[0]), "+f"((ak_off + nn_4 * 4)[1]), "+f"((ak_off + nn_4 * 4)[2]), "+f"((ak_off + nn_4 * 4)[3])
                        : "r"(akr[0]), "r"(akr[1]), "r"(akr[2]), "r"(akr[3]), "r"(br[0]), "r"(br[1]));
                }
            }
        }
        #pragma unroll
        for (int kk_3 = 0; kk_3 < 2; kk_3++) {
            int s0d = r0 + kk_3 * 8;
            float aqd[4];
            float akd[4];
            #pragma unroll
            for (int wd_8 = 0; wd_8 < 4; wd_8++) {
                int rr_1 = r0 + grp + wd_8 % 2 * 8;
                int cc_1 = s0d + tq + wd_8 / 2 * 4;
                int cc_s = cc_1 ^ sw_g;
                float va_ = dAqk_s[rr_1 * 72 + cc_s];
                float vk_ = dAkk_s[rr_1 * 72 + cc_s];
                if (cc_1 > rr_1) {
                    va_ = 0.0f;
                    vk_ = 0.0f;
                }
                aqd[wd_8] = va_;
                akd[wd_8] = vk_;
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
            for (int nn_5 = 0; nn_5 < 4; nn_5++) {
                int kd_3 = c_base + nn_5 * 8 + grp;
                float bvd[2];
                #pragma unroll
                for (int wd_9 = 0; wd_9 < 2; wd_9++) {
                    int ss_3 = s0d + tq + wd_9 * 4;
                    unsigned int rec2[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rec2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rec2[(0) + 1]))
                        : "r"(rec_s_addr + (unsigned int)((ss_3 * 132 + kd_3) * 8)));
                    float _exp2_5 = approx_exp2(gc_b[nn_5] - __uint_as_float(rec2[0]));
                    bvd[wd_9] = __uint_as_float(rec2[1] << 16) * _exp2_5;
                }
                unsigned int brd[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    brd[_lp] = __float_as_uint(bvd[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((aq_dg + nn_5 * 4)[0]), "+f"((aq_dg + nn_5 * 4)[1]), "+f"((aq_dg + nn_5 * 4)[2]), "+f"((aq_dg + nn_5 * 4)[3])
                    : "r"(aqdr[0]), "r"(aqdr[1]), "r"(aqdr[2]), "r"(aqdr[3]), "r"(brd[0]), "r"(brd[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((ak_dg + nn_5 * 4)[0]), "+f"((ak_dg + nn_5 * 4)[1]), "+f"((ak_dg + nn_5 * 4)[2]), "+f"((ak_dg + nn_5 * 4)[3])
                    : "r"(akdr[0]), "r"(akdr[1]), "r"(akdr[2]), "r"(akdr[3]), "r"(brd[0]), "r"(brd[1]));
            }
        }
        #pragma unroll
        for (int nn_6 = 0; nn_6 < 4; nn_6++) {
            #pragma unroll
            for (int half_1 = 0; half_1 < 2; half_1++) {
                int t_loc = r0 + grp + half_1 * 8;
                int col0 = c_base + nn_6 * 8 + tq * 2;
                long long gidx3 = gbase_c + (long long)(t_loc * rs + col0);
                float dqf[2];
                float dkf[2];
                float dgf[2];
                {
                    float2 _v2_5 = *reinterpret_cast<const float2*>(dq_f + gidx3);
                    dqf[0] = _v2_5.x;
                    dqf[0 + 1] = _v2_5.y;
                }
                {
                    float2 _v2_6 = *reinterpret_cast<const float2*>(dk_f + gidx3);
                    dkf[0] = _v2_6.x;
                    dkf[0 + 1] = _v2_6.y;
                }
                {
                    float2 _v2_7 = *reinterpret_cast<const float2*>(dg_f + gidx3);
                    dgf[0] = _v2_7.x;
                    dgf[0 + 1] = _v2_7.y;
                }
                unsigned int rt[4];
                unsigned int rn[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rt[0])), "=r"(*reinterpret_cast<uint32_t*>(&rt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rt[(0) + 3]))
                    : "r"(rec_s_addr + (unsigned int)((t_loc * 132 + col0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rn[0])), "=r"(*reinterpret_cast<uint32_t*>(&rn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rn[(0) + 3]))
                    : "r"(rec_s_addr + (unsigned int)((r0 * 132 + col0) * 8)));
                float dq_o[2];
                float dk_o[2];
                float dg_o[2];
                float bt_t = beta_s[t_loc];
                #pragma unroll
                for (int e_1 = 0; e_1 < 2; e_1++) {
                    const int reg = nn_6 * 4 + half_1 * 2 + e_1;
                    float gtk = __uint_as_float(rt[2 * e_1]);
                    float gn_c = __uint_as_float(rn[2 * e_1]);
                    float gc_c = gn_c - 21.640425324000002f;
                    float _exp2_6 = approx_exp2(gtk - gn_c);
                    float e_off = _exp2_6;
                    float _exp2_7 = approx_exp2(gtk - gc_c);
                    float e_dg = _exp2_7;
                    float dq2 = __fmaf_rn(aq_off[reg], e_off, aq_dg[reg] * e_dg);
                    float dk2 = __fmaf_rn(ak_off[reg], e_off, ak_dg[reg] * e_dg);
                    float kt = __uint_as_float(rt[2 * e_1 + 1] << 16);
                    if (half_1 == 0) {
                        db_lo = __fmaf_rn(dk2, kt, db_lo);
                    } else {
                        db_hi = __fmaf_rn(dk2, kt, db_hi);
                    }
                    dk2 = dk2 * bt_t;
                    float qt = __uint_as_float(rt[2 * e_1 + 1] & 4294901760u);
                    float dg2 = qt * dq2;
                    dq_o[e_1] = dq2 + dqf[e_1];
                    dg_o[e_1] = __fmaf_rn(dk2 - dkt[reg], kt, dg2) + dgf[e_1];
                    dk_o[e_1] = dk2 + dkf[e_1] + dkt[reg];
                }
                {
                    float2 _v2 = make_float2(dg_o[0 + 0], dg_o[0 + 1]);
                    *reinterpret_cast<float2*>((dg_out + gidx3) + 0) = _v2;
                }
                if (t_loc < clen) {
                    long long qindex = (ext0 + (long long)t_loc) * (long long)qrs + (long long)(qk_head * 128 + col0);
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(dq_o[0 + 0], dq_o[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq_ext + qindex))[0]) = _pk;
                    }
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(dk_o[0 + 0], dk_o[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk_ext + qindex))[0]) = _pk;
                    }
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

