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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_ONE_STAGES 1
#define NUM_TWO_STAGES 2
#define NUM_Q_PIPE_STAGES 3
#define NUM_OPS_PIPE_STAGES 4
#define SMEM_GK_P_OFF 0
#define SMEM_GK_P_STAGE_BYTES 8192
#define SMEM_GK_P_STRIDE 8192
#define SMEM_K_P_OFF 32768
#define SMEM_K_P_STAGE_BYTES 8192
#define SMEM_K_P_STRIDE 8192
#define SMEM_Q_P_OFF 68608
#define SMEM_Q_P_STAGE_BYTES 1024
#define SMEM_Q_P_STRIDE 1024
#define SMEM_B_RING_OFF 49152
#define SMEM_B_RING_STAGE_BYTES 8192
#define SMEM_B_RING_STRIDE 8192
#define SMEM_L1_S_OFF 65536
#define SMEM_L1_S_STAGE_BYTES 1024
#define SMEM_L1_S_STRIDE 1024
#define SMEM_L2_S_OFF 66560
#define SMEM_L2_S_STAGE_BYTES 2048
#define SMEM_L2_S_STRIDE 2048
#define SMEM_L3_S_OFF 68608
#define SMEM_L3_S_STAGE_BYTES 3072
#define SMEM_L3_S_STRIDE 3072
#define SMEM_AII_S_OFF 71680
#define SMEM_AII_S_STAGE_BYTES 1024
#define SMEM_AII_S_STRIDE 1024
#define SMEM_XCH_S_OFF 71680
#define SMEM_XCH_S_STAGE_BYTES 4096
#define SMEM_XCH_S_STRIDE 4096
#define SMEM_TOT_S_OFF 75264
#define SMEM_TOT_S_STAGE_BYTES 512
#define SMEM_TOT_S_STRIDE 512
#define SMEM_BETA_S_OFF 75776
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_TOTAL 76160
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}






__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_c9395e21a813579c4bfc(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap g, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ aqk_ext, __nv_bfloat16* __restrict__ akk_ext, __nv_bfloat16* __restrict__ aqk_int, __nv_bfloat16* __restrict__ akk_int, int num_heads, int num_qk_heads, int group, int write_ext, int write_int, float scale, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ v_new, int write_prep)
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

    const int mbar_base = smem + 76032;
    #define ops_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 32)
    #define ring_empty_addr (mbar_base + 56)
    #define acc_done_addr (mbar_base + 72)
    #define hop_addr (mbar_base + 80)
    #define lvl1_done_addr (mbar_base + 88)
    #define lvl2_done_addr (mbar_base + 96)
    #define chain_done_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* gk_p = reinterpret_cast<float*>(smem_raw + 0);
    const int gk_p_addr = smem + 0;
    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int k_p_addr = smem + 32768;
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int q_p_addr = smem + 68608;
    float* b_ring = reinterpret_cast<float*>(smem_raw + 49152);
    const int b_ring_addr = smem + 49152;
    float* l1_s = reinterpret_cast<float*>(smem_raw + 65536);
    const int l1_s_addr = smem + 65536;
    float* l2_s = reinterpret_cast<float*>(smem_raw + 66560);
    const int l2_s_addr = smem + 66560;
    float* l3_s = reinterpret_cast<float*>(smem_raw + 68608);
    const int l3_s_addr = smem + 68608;
    float* aii_s = reinterpret_cast<float*>(smem_raw + 71680);
    const int aii_s_addr = smem + 71680;
    float* xch_s = reinterpret_cast<float*>(smem_raw + 71680);
    const int xch_s_addr = smem + 71680;
    float* tot_s = reinterpret_cast<float*>(smem_raw + 75264);
    const int tot_s_addr = smem + 75264;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 75776);
    const int beta_s_addr = smem + 75776;
    int chunk_s = blockIdx.x;
    int clen_s = chunk_len[chunk_s];
    int ext0_s = chunk_bos[chunk_s];
    if (clen_s == 0) {
        if (write_int != 0) {
            int tid_s = tid;
            float zero8[8];
            zero8[0] = 0.0f;
            zero8[1] = 0.0f;
            zero8[2] = 0.0f;
            zero8[3] = 0.0f;
            zero8[4] = 0.0f;
            zero8[5] = 0.0f;
            zero8[6] = 0.0f;
            zero8[7] = 0.0f;
            int head_s = blockIdx.y;
            long long pbase = ((long long)chunk_s * 64 * (long long)num_heads + (long long)head_s) * 64;
            int prs = num_heads * 64;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                int gi = r * 256 + tid_s;
                int prow = gi / 8;
                int pcol = (gi - prow * 8) * 8;
                long long pidx = pbase + (long long)(prow * prs) + (long long)pcol;
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 0], zero8[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 2], zero8[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 4], zero8[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 6], zero8[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(aqk_int + pidx)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 0], zero8[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 2], zero8[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 4], zero8[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 6], zero8[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_int + pidx)[0] = reinterpret_cast<int4*>(_packed_1)[0];
            }
        }
        if (write_prep != 0) {
            int tid_p = tid;
            float pz8[8];
            pz8[0] = 0.0f;
            pz8[1] = 0.0f;
            pz8[2] = 0.0f;
            pz8[3] = 0.0f;
            pz8[4] = 0.0f;
            pz8[5] = 0.0f;
            pz8[6] = 0.0f;
            pz8[7] = 0.0f;
            int head_p = blockIdx.y;
            long long pbase_p = ((long long)chunk_s * 64 * (long long)num_heads + (long long)head_p) * 128;
            int prs_p = num_heads * 128;
            #pragma unroll
            for (int r_1 = 0; r_1 < 4; r_1++) {
                int gi_p = r_1 * 256 + tid_p;
                int prow_p = gi_p / 16;
                int pcol_p = (gi_p - prow_p * 16) * 8;
                long long pidx_p = pbase_p + (long long)(prow_p * prs_p) + (long long)pcol_p;
                unsigned int _packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 0], pz8[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 2], pz8[_lp*2+1 + 2]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 4], pz8[_lp*2+1 + 4]));
                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 6], pz8[_lp*2+1 + 6]));
                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(kb_out + pidx_p)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                unsigned int _packed_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 0], pz8[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 2], pz8[_lp*2+1 + 2]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 4], pz8[_lp*2+1 + 4]));
                    _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 6], pz8[_lp*2+1 + 6]));
                    _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(kg_out + pidx_p)[0] = reinterpret_cast<int4*>(_packed_3)[0];
                unsigned int _packed_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 0], pz8[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 2], pz8[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 4], pz8[_lp*2+1 + 4]));
                    _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 6], pz8[_lp*2+1 + 6]));
                    _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(qg_out + pidx_p)[0] = reinterpret_cast<int4*>(_packed_4)[0];
                unsigned int _packed_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 0], pz8[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 2], pz8[_lp*2+1 + 2]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 4], pz8[_lp*2+1 + 4]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pz8[_lp*2 + 6], pz8[_lp*2+1 + 6]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(v_new + pidx_p)[0] = reinterpret_cast<int4*>(_packed_5)[0];
            }
            if (tid_p < 32) {
                float pz4[4];
                pz4[0] = 0.0f;
                pz4[1] = 0.0f;
                pz4[2] = 0.0f;
                pz4[3] = 0.0f;
                long long gidx_p = pbase_p + (long long)(prs_p * 63) + (long long)(tid_p * 4);
                {
                    float4 _v4 = make_float4(pz4[0 + 0], pz4[0 + 1], pz4[0 + 2], pz4[0 + 3]);
                    *reinterpret_cast<float4*>((gk_out + gidx_p) + 0) = _v4;
                }
            }
        }
    }
    if (__syncthreads_and(clen_s == 0)) return;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[76032..76144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ops_pipe' ---
            // ops_full: 4 barriers, init_count=1
            mbarrier_init(smem + 76032, 1);
            mbarrier_init(smem + 76040, 1);
            mbarrier_init(smem + 76048, 1);
            mbarrier_init(smem + 76056, 1);
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 76064, 1);
            mbarrier_init(smem + 76072, 1);
            mbarrier_init(smem + 76080, 1);
            // --- pipeline 'two' ---
            // ring_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 76088, 1);
            mbarrier_init(smem + 76096, 1);
            // --- pipeline 'one' ---
            // acc_done: 1 barriers, init_count=1
            mbarrier_init(smem + 76104, 1);
            // hop: 1 barriers, init_count=1
            mbarrier_init(smem + 76112, 1);
            // lvl1_done: 1 barriers, init_count=1
            mbarrier_init(smem + 76120, 1);
            // lvl2_done: 1 barriers, init_count=1
            mbarrier_init(smem + 76128, 1);
            // chain_done: 1 barriers, init_count=1
            mbarrier_init(smem + 76136, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 76144);
    if (warp == 0) {
        int _tmem_hold = smem + 76144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (declaration order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    int w = warp;
    int lane_1 = lane;
    int q4 = w % 4;
    int h4 = w / 4;
    int row_base = q4 * 32 << 16;
    int ext0_i = ext0_s;
    long long ext0 = (long long)ext0_i;
    int clen = clen_s;
    long long abase = (ext0 * (long long)num_heads + (long long)head) * 64;
    long long ibase = ((long long)row0 * (long long)num_heads + (long long)head) * 64;
    int ars = num_heads * 64;
    if (warp == 3) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 1024);
            tma_3d_gmem2smem(q_p_addr, (&q), 0, qk_head, ext0_i, q_full_addr);
            mbarrier_arrive_expect_tx(q_full_addr + 8, 1024);
            tma_3d_gmem2smem(q_p_addr + 1024, (&q), 32, qk_head, ext0_i, q_full_addr + 8);
            mbarrier_arrive_expect_tx(q_full_addr + 16, 1024);
            tma_3d_gmem2smem(q_p_addr + 2048, (&q), 64, qk_head, ext0_i, q_full_addr + 16);
            mbarrier_arrive_expect_tx(ops_full_addr, 16384);
            tma_3d_gmem2smem(gk_p_addr, (&g), 0, head, ext0_i, ops_full_addr);
            tma_3d_gmem2smem(k_p_addr, (&k), 0, qk_head, ext0_i, ops_full_addr);
            mbarrier_arrive_expect_tx(ops_full_addr + 8, 8192);
            tma_3d_gmem2smem(gk_p_addr + 8192, (&g), 32, head, ext0_i, ops_full_addr + 8);
            mbarrier_arrive_expect_tx(ops_full_addr + 16, 16384);
            tma_3d_gmem2smem(gk_p_addr + 16384, (&g), 64, head, ext0_i, ops_full_addr + 16);
            tma_3d_gmem2smem(k_p_addr + 8192, (&k), 64, qk_head, ext0_i, ops_full_addr + 16);
            mbarrier_arrive_expect_tx(ops_full_addr + 24, 8192);
            tma_3d_gmem2smem(gk_p_addr + 24576, (&g), 96, head, ext0_i, ops_full_addr + 24);
        }
    }
    if (tid_0 < 64) {
        float b_v = 0.0f;
        if (tid_0 < clen) {
            __nv_bfloat16 b_raw = beta[(ext0 + (long long)tid_0) * (long long)num_heads + (long long)head];
            b_v = (float)b_raw;
        }
        beta_s[tid_0] = b_v;
    }
    int sl = lane_1 % 16;
    int hh = lane_1 / 16;
    int p8 = sl & 7;
    int atom = (sl >> 3) * 1024 + p8 * 128;
    int cpair = 4 * h4 + 2 * hh;
    int g_lane = atom + ((cpair ^ p8) << 4);
    int kl0 = atom + ((2 * h4 + hh ^ p8) << 4);
    int kl1 = atom + ((4 + 2 * h4 + hh ^ p8) << 4);
    int ql = sl * 64 + ((2 * h4 + hh ^ sl >> 1 & 3) << 4);
    int an_lane = cpair << 4;
    int th = hh * 8;
    int scol = q4 * 16;
    int s_row = scol + sl;
    int gk0 = (int)gk_p_addr;
    int k_addr = (int)k_p_addr;
    int q_addr = (int)q_p_addr + ql;
    int b_addr = (int)b_ring_addr + g_lane;
    unsigned int a_taddr = taddr + (unsigned int)row_base + (unsigned int)(64 + 16 * h4);
    unsigned int qs = 0;
    int fmask = 50748;
    if (write_prep == 0) {
        fmask = 0;
    }
    unsigned int _phase_ops_full_0 = 0;
    mbarrier_wait(ops_full_addr, _phase_ops_full_0);
    _phase_ops_full_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned int _phase_ops_full_1 = 0;
    mbarrier_wait(ops_full_addr + 8, _phase_ops_full_1);
    _phase_ops_full_1 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned int _phase_ops_full_2 = 0;
    mbarrier_wait(ops_full_addr + 16, _phase_ops_full_2);
    _phase_ops_full_2 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    unsigned int _phase_ops_full_3 = 0;
    mbarrier_wait(ops_full_addr + 24, _phase_ops_full_3);
    _phase_ops_full_3 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    int rb = gk0 + (lane_1 >> 3) * 8192 + w * 1024;
    int c8v = lane_1 & 7;
    int gwv = w;
    int xc = (int)xch_s_addr + lane_1 * 16;
    int t_0 = gwv * 8;
    float gq[32];
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&gq[p * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(p * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(p * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(p * 4) + 3]))
            : "r"(rb + p * 128 + ((c8v ^ p) << 4)));
    }
    if (clen < t_0 + 8) {
        #pragma unroll
        for (int p_1 = 0; p_1 < 8; p_1++) {
            int t_p = t_0 + p_1;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float g_e = 0.0f;
                if (t_p < clen) {
                    g_e = gq[p_1 * 4 + e];
                }
                gq[p_1 * 4 + e] = g_e;
            }
        }
    }
    #pragma unroll
    for (int e_1 = 0; e_1 < 4; e_1 += 2) {
        float2 acc2 = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int p_2 = 0; p_2 < 8; p_2++) {
            acc2 = add_f32x2(acc2, make_float2(gq[p_2 * 4 + e_1], gq[p_2 * 4 + e_1 + 1]));
            gq[p_2 * 4 + e_1] = acc2.x;
            gq[p_2 * 4 + e_1 + 1] = acc2.y;
        }
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xc + gwv * 512), "f"(gq[28]), "f"(gq[29]), "f"(gq[30]), "f"(gq[31]) : "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    float2 off0 = make_float2(0.0f, 0.0f);
    float2 off1 = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int gi_1 = 0; gi_1 < 7; gi_1++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(xc + gi_1 * 512));
        if (gwv > gi_1) {
            off0 = add_f32x2(off0, make_float2(tg[0], tg[1]));
            off1 = add_f32x2(off1, make_float2(tg[2], tg[3]));
        }
    }
    float2 _f2_0 = make_float2(1.442695f, 1.442695f);
    #pragma unroll
    for (int p_3 = 0; p_3 < 8; p_3++) {
        float gkr[4];
        float2 _f2_1 = make_float2(gq[p_3 * 4], gq[p_3 * 4 + 1]);
        float2 _f2_2 = make_float2(gq[p_3 * 4 + 2], gq[p_3 * 4 + 3]);
        float2 _f32x2_mul_a_0 = add_f32x2(off0, _f2_1);
        float2 r0;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&_f2_0));
        float2 _f32x2_mul_a_1 = add_f32x2(off1, _f2_2);
        float2 r1;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&_f2_0));
        gkr[0] = r0.x;
        gkr[1] = r0.y;
        gkr[2] = r1.x;
        gkr[3] = r1.y;
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(rb + p_3 * 128 + ((c8v ^ p_3) << 4)), "f"(gkr[0]), "f"(gkr[1]), "f"(gkr[2]), "f"(gkr[3]) : "memory");
    }
    if (gwv == 7) {
        float2 _f2_3 = make_float2(gq[28], gq[29]);
        float2 _f2_4 = make_float2(gq[30], gq[31]);
        float2 tl2 = add_f32x2(off0, _f2_3);
        float2 th2 = add_f32x2(off1, _f2_4);
        float tot4[4];
        tot4[0] = tl2.x;
        tot4[1] = tl2.y;
        tot4[2] = th2.x;
        tot4[3] = th2.y;
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xc + 3584), "f"(tot4[0]), "f"(tot4[1]), "f"(tot4[2]), "f"(tot4[3]) : "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (write_prep != 0) {
        if (tid_0 < 32) {
            float g63[4];
            int g63_c = (tid_0 & 7 ^ 7) * 16;
            int g63_a = gk0 + (tid_0 >> 3) * 8192 + 8064 + g63_c;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g63[0])), "=r"(*reinterpret_cast<uint32_t*>(&g63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g63[(0) + 3]))
                : "r"(g63_a));
            long long g63_o = ibase * 2 + (long long)(ars * 2 * 63 + tid_0 * 4);
            {
                float4 _v4 = make_float4(g63[0 + 0], g63[0 + 1], g63[0 + 2], g63[0 + 3]);
                *reinterpret_cast<float4*>((gk_out + g63_o) + 0) = _v4;
            }
        }
    }
    unsigned int _phase_q_full = 0;
    unsigned int _phase_ring_empty_0 = 1;
    unsigned int _phase_acc_done_0 = 0;
    unsigned int _phase_ring_empty_1 = 1;
    #pragma unroll 1
    for (int i = 0; i < 4; i++) {
        int fill = fmask >> i * 4 + q4 & 1;
        int fq = q4 - (146 >> i * 2 & 3);
        int f_row = h4 * 8 + (lane_1 >> 2);
        int f_c = lane_1 & 3;
        int p8f = f_row & 7;
        int t_f = i * 16 + f_row;
        float bt_r[1];
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bt_r[0])) : "r"((int)beta_s_addr + t_f * 4));
        float bt_f = bt_r[0];
        int v_f = 0;
        if (t_f < clen) {
            v_f = 1;
        }
        unsigned int m_f = (unsigned int)(-v_f);
        int rowb = i * 2048 + f_row * 128;
        int c4 = (f_c ^ fq << 2) << 4;
        int gk_i = gk0 + rowb + (c4 ^ p8f << 4);
        int t_i = (int)tot_s_addr + (fq * 16 + f_c * 4 << 2);
        float rcp_f = 1.442695f;
        int cq = (f_c >> 1 ^ fq << 1) << 4;
        int h8 = (f_c & 1) * 8;
        int k_i = k_addr + rowb + (cq ^ p8f << 4) + h8;
        int k_i1 = k_i ^ 64;
        int q_i = (int)q_p_addr + f_row * 64 + (cq ^ (f_row >> 1 & 3) << 4) + h8;
        long long o_i = ibase * 2 + (long long)(t_f * (ars * 2) + (fq * 16 + f_c * 4));
        __nv_bfloat16* kb_p = kb_out + o_i;
        __nv_bfloat16* kg_p = kg_out + o_i;
        __nv_bfloat16* qg_p = qg_out + o_i;
        int bq = q4 - i - 1 & 3;
        if (fill != 0) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            float gk[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gk[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk[(0) + 3]))
                : "r"(gk_i));
            float tot[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tot[0])), "=r"(*reinterpret_cast<uint32_t*>(&tot[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tot[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tot[(0) + 3]))
                : "r"(t_i));
            unsigned int kw[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw[(0) + 1]))
                : "r"(k_i));
            unsigned int ks[2];
            ks[0] = kw[0] & m_f;
            ks[1] = kw[1] & m_f;
            float _unpack_bf16_0[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ks[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ks[_pair]) & 0xffff0000u);
            }
            float kb[4];
            float kg[4];
            float2 _f2_5 = make_float2(bt_f, bt_f);
            unsigned int qw[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(0) + 1]))
                : "r"(q_i + (int)qs * 1024));
            unsigned int qsel[2];
            qsel[0] = qw[0] & m_f;
            qsel[1] = qw[1] & m_f;
            float _unpack_bf16_1[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qsel[_pair]) << 16);
                (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qsel[_pair]) & 0xffff0000u);
            }
            float qg[4];
            float e0 = approx_exp2(gk[0]);
            float e1 = approx_exp2(gk[1]);
            float2 _f2_6 = make_float2(e0, e1);
            float2 _f32x2_mul_a_2 = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
            float2 kbt;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f2_5));
            float kb0 = (float)(__nv_bfloat16)kbt.x;
            float kb1 = (float)(__nv_bfloat16)kbt.y;
            float2 _f32x2_mul_a_3 = make_float2(kb0, kb1);
            float2 kb2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&_f2_6));
            kb[0] = kb2.x;
            kb[1] = kb2.y;
            float _fma_0;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(tot[0]), "f"(rcp_f), "f"(-gk[0]));
            float _fma_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(tot[1]), "f"(rcp_f), "f"(-gk[1]));
            float x0 = approx_exp2(_fma_0);
            float x1 = approx_exp2(_fma_1);
            float2 _f32x2_mul_a_4 = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
            float2 _f32x2_mul_b_5 = make_float2(x0, x1);
            float2 kg2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            kg[0] = kg2.x;
            kg[1] = kg2.y;
            float2 _f32x2_mul_a_6 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
            float2 qg2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f2_6));
            qg[0] = qg2.x;
            qg[1] = qg2.y;
            float e0_0 = approx_exp2(gk[2]);
            float e1_1 = approx_exp2(gk[3]);
            float2 _f2_7 = make_float2(e0_0, e1_1);
            float2 _f32x2_mul_a_7 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
            float2 kbt_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f2_5));
            float kb0_3 = (float)(__nv_bfloat16)kbt_2.x;
            float kb1_4 = (float)(__nv_bfloat16)kbt_2.y;
            float2 _f32x2_mul_a_8 = make_float2(kb0_3, kb1_4);
            float2 kb2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f2_7));
            kb[2] = kb2_5.x;
            kb[3] = kb2_5.y;
            float _fma_2;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(tot[2]), "f"(rcp_f), "f"(-gk[2]));
            float _fma_3;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(tot[3]), "f"(rcp_f), "f"(-gk[3]));
            float x0_6 = approx_exp2(_fma_2);
            float x1_7 = approx_exp2(_fma_3);
            float2 _f32x2_mul_a_9 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
            float2 _f32x2_mul_b_10 = make_float2(x0_6, x1_7);
            float2 kg2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
            kg[2] = kg2_8.x;
            kg[3] = kg2_8.y;
            float2 _f32x2_mul_a_11 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
            float2 qg2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&_f2_7));
            qg[2] = qg2_9.x;
            qg[3] = qg2_9.y;
            unsigned int _packed_6[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb[_lp*2 + 0], kb[_lp*2+1 + 0]));
                _packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb[_lp*2 + 2], kb[_lp*2+1 + 2]));
                _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_12 = kb_p + 0;
                __builtin_assume(__isGlobal(_gst_12));
                reinterpret_cast<int2*>(_gst_12)[0] = reinterpret_cast<int2*>(_packed_6)[0];
            }
            unsigned int _packed_7[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 0], kg[_lp*2+1 + 0]));
                _packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg[_lp*2 + 2], kg[_lp*2+1 + 2]));
                _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_13 = kg_p + 0;
                __builtin_assume(__isGlobal(_gst_13));
                reinterpret_cast<int2*>(_gst_13)[0] = reinterpret_cast<int2*>(_packed_7)[0];
            }
            unsigned int _packed_8[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg[_lp*2 + 0], qg[_lp*2+1 + 0]));
                _packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg[_lp*2 + 2], qg[_lp*2+1 + 2]));
                _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_14 = qg_p + 0;
                __builtin_assume(__isGlobal(_gst_14));
                reinterpret_cast<int2*>(_gst_14)[0] = reinterpret_cast<int2*>(_packed_8)[0];
            }
        }
        mbarrier_wait(ring_empty_addr, _phase_ring_empty_0);
        _phase_ring_empty_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (bq < 2) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        int gk_step = gk0;
        int k_step = k_addr + kl0;
        float an[8];
        int aa = gk_step + i * 2048 + an_lane;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an[0])), "=r"(*reinterpret_cast<uint32_t*>(&an[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an[(0) + 3]))
            : "r"(aa));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an[4])), "=r"(*reinterpret_cast<uint32_t*>(&an[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an[(4) + 3]))
            : "r"(aa + 16));
        if (q4 <= i) {
            int use_gc = 0;
            if (q4 == i) {
                use_gc = 1;
            }
            int s_off = q4 * 2048;
            int ga = gk_step + g_lane + s_off;
            int ka = k_step + s_off;
            float g_0[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(0) + 3]))
                : "r"(ga));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0[(4) + 3]))
                : "r"(ga ^ 16));
            unsigned int kw_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_1[(0) + 3]))
                : "r"(ka));
            float _unpack_bf16_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) << 16);
                (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) & 0xffff0000u);
            }
            float shift = 0.0f;
            if (use_gc != 0) {
                shift = 21.640425f;
            }
            float2 _f2_8 = make_float2(shift, shift);
            float v[8];
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2 += 2) {
                float2 d2 = sub_f32x2(sub_f32x2(make_float2(an[e_2], an[e_2 + 1]), _f2_8), make_float2(g_0[e_2], g_0[e_2 + 1]));
                float x0_1 = approx_exp2(d2.x);
                float x1_1 = approx_exp2(d2.y);
                float2 _f32x2_mul_a_15 = make_float2(x0_1, x1_1);
                float2 _f32x2_mul_b_16 = make_float2(_unpack_bf16_2[e_2], _unpack_bf16_2[e_2 + 1]);
                float2 v2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
                v[e_2] = v2.x;
                v[e_2 + 1] = v2.y;
            }
            if (s_row >= clen) {
                #pragma unroll
                for (int e_3 = 0; e_3 < 8; e_3++) {
                    v[e_3] = 0.0f;
                }
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(a_taddr), "r"(*reinterpret_cast<const uint32_t*>(&v[0])), "r"(*reinterpret_cast<const uint32_t*>(&v[1])), "r"(*reinterpret_cast<const uint32_t*>(&v[2])), "r"(*reinterpret_cast<const uint32_t*>(&v[3])), "r"(*reinterpret_cast<const uint32_t*>(&v[4])), "r"(*reinterpret_cast<const uint32_t*>(&v[5])), "r"(*reinterpret_cast<const uint32_t*>(&v[6])), "r"(*reinterpret_cast<const uint32_t*>(&v[7])));
        }
        if (bq < 2) {
            int t_off = i * 2048;
            int ga_1 = gk_step + g_lane + t_off;
            int qa = q_addr + (int)qs * 1024;
            int ka_1 = k_step + t_off;
            int sa = b_addr + bq * 2048;
            float g_0_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(0) + 3]))
                : "r"(ga_1));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_1[(4) + 3]))
                : "r"(ga_1 ^ 16));
            unsigned int qw_1[4];
            unsigned int kw_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 3]))
                : "r"(qa));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_2[(0) + 3]))
                : "r"(ka_1));
            float shift_1 = 0.0f;
            if (bq != 0) {
                shift_1 = 21.640425f;
            }
            float2 _f2_9 = make_float2(shift_1, shift_1);
            #pragma unroll
            for (int e_4 = 0; e_4 < 8; e_4 += 2) {
                float2 d2_1 = sub_f32x2(make_float2(g_0_1[e_4], g_0_1[e_4 + 1]), sub_f32x2(make_float2(an[e_4], an[e_4 + 1]), _f2_9));
                float _exp2_17 = approx_exp2(d2_1.x);
                g_0_1[e_4] = _exp2_17;
                float _exp2_18 = approx_exp2(d2_1.y);
                g_0_1[e_4 + 1] = _exp2_18;
            }
            if (clen <= i * 16 + sl) {
                #pragma unroll
                for (int e_5 = 0; e_5 < 8; e_5++) {
                    g_0_1[e_5] = 0.0f;
                }
            }
            float _unpack_bf16_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) << 16);
                (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) & 0xffff0000u);
            }
            float vq[8];
            #pragma unroll
            for (int e_6 = 0; e_6 < 8; e_6 += 2) {
                float2 _f32x2_mul_a_19 = make_float2(_unpack_bf16_3[e_6], _unpack_bf16_3[e_6 + 1]);
                float2 _f32x2_mul_b_20 = make_float2(g_0_1[e_6], g_0_1[e_6 + 1]);
                float2 q2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                vq[e_6] = q2.x;
                vq[e_6 + 1] = q2.y;
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa), "f"(vq[0]), "f"(vq[1]), "f"(vq[2]), "f"(vq[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa ^ 16), "f"(vq[4]), "f"(vq[5]), "f"(vq[6]), "f"(vq[7]) : "memory");
            float _unpack_bf16_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_2[_pair]) << 16);
                (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_2[_pair]) & 0xffff0000u);
            }
            float vk[8];
            #pragma unroll
            for (int e_7 = 0; e_7 < 8; e_7 += 2) {
                float2 _f32x2_mul_a_21 = make_float2(_unpack_bf16_4[e_7], _unpack_bf16_4[e_7 + 1]);
                float2 _f32x2_mul_b_22 = make_float2(g_0_1[e_7], g_0_1[e_7 + 1]);
                float2 k2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                vk[e_7] = k2.x;
                vk[e_7 + 1] = k2.y;
            }
            int ksa = sa + 4096;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa), "f"(vk[0]), "f"(vk[1]), "f"(vk[2]), "f"(vk[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa ^ 16), "f"(vk[4]), "f"(vk[5]), "f"(vk[6]), "f"(vk[7]) : "memory");
        }
        if (i >= 1) {
            mbarrier_wait_hint(acc_done_addr, _phase_acc_done_0, 1000000);
            _phase_acc_done_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tbase = (i - 1) * 16;
            if (q4 <= i - 1) {
                int col = h4 * 32;
                if (q4 == i - 1) {
                    col = col + 16;
                }
                float _tmem_load_0[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)col));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (h4 == 0) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        _tmem_load_0[j] = _tmem_load_0[j] * scale;
                    }
                    if (q4 == i - 1) {
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            if (sl > th + j_1) {
                                _tmem_load_0[j_1] = 0.0f;
                            }
                        }
                    }
                    int sl_0 = lane_1 % 16;
                    int th_1 = lane_1 / 16 * 8;
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>((int)(b_ring_addr + 8192) + w * 1152 + ((th_1 + j_2) * 18 + sl_0) * 4);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[j_2]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int tl = lane_1 / 2;
                    int hh_2 = lane_1 % 2 * 8;
                    int raddr = (int)(b_ring_addr + 8192) + w * 1152 + (tl * 18 + hh_2) * 4;
                    float vals[8];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals[(0) + 1]))
                        : "r"(raddr));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals[(2) + 1]))
                        : "r"(raddr + 8));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals[(4) + 1]))
                        : "r"(raddr + 16));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals[(6) + 1]))
                        : "r"(raddr + 24));
                    int t_d = tbase + tl;
                    int col_3 = scol + hh_2;
                    if (write_ext != 0) {
                        if (t_d < clen) {
                            unsigned int _packed_9[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                                _packed_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 2], vals[_lp*2+1 + 2]));
                                _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 4], vals[_lp*2+1 + 4]));
                                _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 6], vals[_lp*2+1 + 6]));
                                _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(aqk_ext + (abase + (long long)(t_d * ars) + (long long)col_3))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                        }
                    }
                    if (write_int != 0) {
                        if (t_d >= clen) {
                            #pragma unroll
                            for (int e_8 = 0; e_8 < 8; e_8++) {
                                vals[e_8] = 0.0f;
                            }
                        }
                        unsigned int _packed_10[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 2], vals[_lp*2+1 + 2]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 4], vals[_lp*2+1 + 4]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 6], vals[_lp*2+1 + 6]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(aqk_int + (ibase + (long long)(t_d * ars) + (long long)col_3))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                } else {
                    float bt[8];
                    int baddr = (int)beta_s_addr + (tbase + th) * 4;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bt[0])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(0) + 3]))
                        : "r"(baddr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bt[4])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bt[(4) + 3]))
                        : "r"(baddr + 16));
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        _tmem_load_0[j_3] = _tmem_load_0[j_3] * bt[j_3];
                    }
                    int c4_0 = sl >> 2;
                    int low = (sl & 3) * 4;
                    if (q4 == i - 1) {
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            if (sl >= th + j_4) {
                                _tmem_load_0[j_4] = 0.0f;
                            }
                        }
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 8; j_5++) {
                            _tmem_load_0[j_5] = -_tmem_load_0[j_5];
                        }
                        int base_t = (int)(aii_s_addr + (unsigned int)(q4 * 1024)) + th * 64 + low;
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            {
                                uint32_t _addr_24 = static_cast<uint32_t>(base_t + j_6 * 64 + ((c4_0 ^ j_6 >> 1) << 4));
                                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[j_6]) : "memory");
                            }
                        }
                    } else if (i - 1 == 2) {
                        int base_t_1 = (int)l2_s_addr + th * 128 + low;
                        int c8 = q4 * 4 + c4_0;
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 8; j_7++) {
                            {
                                uint32_t _addr_25 = static_cast<uint32_t>(base_t_1 + j_7 * 128 + ((c8 ^ j_7) << 4));
                                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[j_7]) : "memory");
                            }
                        }
                    } else {
                        int lb = (int)l1_s_addr;
                        if (i - 1 == 3) {
                            lb = (int)l3_s_addr + q4 * 1024;
                        }
                        int base_t_2 = lb + th * 64 + low;
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 8; j_8++) {
                            {
                                uint32_t _addr_26 = static_cast<uint32_t>(base_t_2 + j_8 * 64 + ((c4_0 ^ j_8 >> 1) << 4));
                                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[j_8]) : "memory");
                            }
                        }
                    }
                }
            } else if (h4 == 0) {
                float zero8_1[8];
                zero8_1[0] = 0.0f;
                zero8_1[1] = 0.0f;
                zero8_1[2] = 0.0f;
                zero8_1[3] = 0.0f;
                zero8_1[4] = 0.0f;
                zero8_1[5] = 0.0f;
                zero8_1[6] = 0.0f;
                zero8_1[7] = 0.0f;
                int t_d_1 = tbase + lane_1 / 2;
                int col_1 = scol + lane_1 % 2 * 8;
                if (write_ext != 0) {
                    if (t_d_1 < clen) {
                        unsigned int _packed_11[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 0], zero8_1[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 2], zero8_1[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 4], zero8_1[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 6], zero8_1[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(aqk_ext + (abase + (long long)(t_d_1 * ars) + (long long)col_1))[0] = reinterpret_cast<int4*>(_packed_11)[0];
                    }
                }
                if (write_int != 0) {
                    unsigned int _packed_12[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 0], zero8_1[_lp*2+1 + 0]));
                        _packed_12[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 2], zero8_1[_lp*2+1 + 2]));
                        _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 4], zero8_1[_lp*2+1 + 4]));
                        _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_1[_lp*2 + 6], zero8_1[_lp*2+1 + 6]));
                        _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(aqk_int + (ibase + (long long)(t_d_1 * ars) + (long long)col_1))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                }
            } else {
                float zero8_2[8];
                zero8_2[0] = 0.0f;
                zero8_2[1] = 0.0f;
                zero8_2[2] = 0.0f;
                zero8_2[3] = 0.0f;
                zero8_2[4] = 0.0f;
                zero8_2[5] = 0.0f;
                zero8_2[6] = 0.0f;
                zero8_2[7] = 0.0f;
                int t_d_2 = tbase + lane_1 / 2;
                int col_2 = scol + lane_1 % 2 * 8;
                if (write_ext != 0) {
                    if (t_d_2 < clen) {
                        unsigned int _packed_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 0], zero8_2[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 2], zero8_2[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 4], zero8_2[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 6], zero8_2[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_2 * ars) + (long long)col_2))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                    }
                }
                if (write_int != 0) {
                    unsigned int _packed_14[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 0], zero8_2[_lp*2+1 + 0]));
                        _packed_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 2], zero8_2[_lp*2+1 + 2]));
                        _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 4], zero8_2[_lp*2+1 + 4]));
                        _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_2[_lp*2 + 6], zero8_2[_lp*2+1 + 6]));
                        _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_2 * ars) + (long long)col_2))[0] = reinterpret_cast<int4*>(_packed_14)[0];
                }
            }
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        if (warp == 3) {
            if (elect_sync()) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_27 = (((b_ring_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_27 = (((b_ring_addr) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 68159760;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_acc), "r"(_smem_desc_lo_27), "r"(tmem_acc + 64), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                int i_n = i;
                if (i_n < 4) {
                    mbarrier_arrive_expect_tx(q_full_addr + (qs) * 8, 1024);
                    tma_3d_gmem2smem(q_p_addr + qs * 1024, (&q), 96, qk_head, ext0_i + i_n * 16, q_full_addr + (qs) * 8);
                }
            }
        }
        qs += 1;
        if (qs == 3) { qs = 0; _phase_q_full ^= 1; }
        int bq_0 = q4 - i - 1 & 3;
        if (fill != 0) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            float gk_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 3]))
                : "r"(gk_i + 8192));
            float tot_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tot_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tot_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tot_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tot_1[(0) + 3]))
                : "r"(t_i + 128));
            unsigned int kw_3[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_3[(0) + 1]))
                : "r"(k_i1));
            unsigned int ks_1[2];
            ks_1[0] = kw_3[0] & m_f;
            ks_1[1] = kw_3[1] & m_f;
            float _unpack_bf16_5[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ks_1[_pair]) << 16);
                (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ks_1[_pair]) & 0xffff0000u);
            }
            float kb_1[4];
            float kg_1[4];
            float2 _f2_10 = make_float2(bt_f, bt_f);
            unsigned int qw_2[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(0) + 1]))
                : "r"(q_i + (int)qs * 1024));
            unsigned int qsel_1[2];
            qsel_1[0] = qw_2[0] & m_f;
            qsel_1[1] = qw_2[1] & m_f;
            float _unpack_bf16_6[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qsel_1[_pair]) << 16);
                (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qsel_1[_pair]) & 0xffff0000u);
            }
            float qg_1[4];
            float e0_1 = approx_exp2(gk_1[0]);
            float e1_2 = approx_exp2(gk_1[1]);
            float2 _f2_11 = make_float2(e0_1, e1_2);
            float2 _f32x2_mul_a_28 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
            float2 kbt_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f2_10));
            float kb0_1 = (float)(__nv_bfloat16)kbt_1.x;
            float kb1_1 = (float)(__nv_bfloat16)kbt_1.y;
            float2 _f32x2_mul_a_29 = make_float2(kb0_1, kb1_1);
            float2 kb2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_f2_11));
            kb_1[0] = kb2_1.x;
            kb_1[1] = kb2_1.y;
            float _fma_4;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(tot_1[0]), "f"(rcp_f), "f"(-gk_1[0]));
            float _fma_5;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(tot_1[1]), "f"(rcp_f), "f"(-gk_1[1]));
            float x0_2 = approx_exp2(_fma_4);
            float x1_2 = approx_exp2(_fma_5);
            float2 _f32x2_mul_a_30 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
            float2 _f32x2_mul_b_31 = make_float2(x0_2, x1_2);
            float2 kg2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
            kg_1[0] = kg2_1.x;
            kg_1[1] = kg2_1.y;
            float2 _f32x2_mul_a_32 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
            float2 qg2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f2_11));
            qg_1[0] = qg2_1.x;
            qg_1[1] = qg2_1.y;
            float e0_0_1 = approx_exp2(gk_1[2]);
            float e1_1_1 = approx_exp2(gk_1[3]);
            float2 _f2_12 = make_float2(e0_0_1, e1_1_1);
            float2 _f32x2_mul_a_33 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
            float2 kbt_2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f2_10));
            float kb0_3_1 = (float)(__nv_bfloat16)kbt_2_1.x;
            float kb1_4_1 = (float)(__nv_bfloat16)kbt_2_1.y;
            float2 _f32x2_mul_a_34 = make_float2(kb0_3_1, kb1_4_1);
            float2 kb2_5_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_5_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f2_12));
            kb_1[2] = kb2_5_1.x;
            kb_1[3] = kb2_5_1.y;
            float _fma_6;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(tot_1[2]), "f"(rcp_f), "f"(-gk_1[2]));
            float _fma_7;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(tot_1[3]), "f"(rcp_f), "f"(-gk_1[3]));
            float x0_6_1 = approx_exp2(_fma_6);
            float x1_7_1 = approx_exp2(_fma_7);
            float2 _f32x2_mul_a_35 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
            float2 _f32x2_mul_b_36 = make_float2(x0_6_1, x1_7_1);
            float2 kg2_8_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_8_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
            kg_1[2] = kg2_8_1.x;
            kg_1[3] = kg2_8_1.y;
            float2 _f32x2_mul_a_37 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
            float2 qg2_9_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_f2_12));
            qg_1[2] = qg2_9_1.x;
            qg_1[3] = qg2_9_1.y;
            unsigned int _packed_15[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_1[_lp*2 + 0], kb_1[_lp*2+1 + 0]));
                _packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_1[_lp*2 + 2], kb_1[_lp*2+1 + 2]));
                _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_38 = kb_p + 32;
                __builtin_assume(__isGlobal(_gst_38));
                reinterpret_cast<int2*>(_gst_38)[0] = reinterpret_cast<int2*>(_packed_15)[0];
            }
            unsigned int _packed_16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_1[_lp*2 + 0], kg_1[_lp*2+1 + 0]));
                _packed_16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_1[_lp*2 + 2], kg_1[_lp*2+1 + 2]));
                _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_39 = kg_p + 32;
                __builtin_assume(__isGlobal(_gst_39));
                reinterpret_cast<int2*>(_gst_39)[0] = reinterpret_cast<int2*>(_packed_16)[0];
            }
            unsigned int _packed_17[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_1[_lp*2 + 0], qg_1[_lp*2+1 + 0]));
                _packed_17[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_1[_lp*2 + 2], qg_1[_lp*2+1 + 2]));
                _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_40 = qg_p + 32;
                __builtin_assume(__isGlobal(_gst_40));
                reinterpret_cast<int2*>(_gst_40)[0] = reinterpret_cast<int2*>(_packed_17)[0];
            }
        }
        mbarrier_wait(ring_empty_addr + 8, _phase_ring_empty_1);
        _phase_ring_empty_1 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (bq_0 < 2) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        int gk_step_1 = gk0 + 8192;
        int k_step_2 = k_addr + kl1;
        float an_3[8];
        int aa_4 = gk_step_1 + i * 2048 + an_lane;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(0) + 3]))
            : "r"(aa_4));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_3[(4) + 3]))
            : "r"(aa_4 + 16));
        if (q4 <= i) {
            int use_gc_1 = 0;
            if (q4 == i) {
                use_gc_1 = 1;
            }
            int s_off_1 = q4 * 2048;
            int ga_2 = gk_step_1 + g_lane + s_off_1;
            int ka_2 = k_step_2 + s_off_1;
            float g_0_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(0) + 3]))
                : "r"(ga_2));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_2[(4) + 3]))
                : "r"(ga_2 ^ 16));
            unsigned int kw_4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_4[(0) + 3]))
                : "r"(ka_2));
            float _unpack_bf16_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_4[_pair]) << 16);
                (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_4[_pair]) & 0xffff0000u);
            }
            float shift_2 = 0.0f;
            if (use_gc_1 != 0) {
                shift_2 = 21.640425f;
            }
            float2 _f2_13 = make_float2(shift_2, shift_2);
            float v_1[8];
            #pragma unroll
            for (int e_9 = 0; e_9 < 8; e_9 += 2) {
                float2 d2_2 = sub_f32x2(sub_f32x2(make_float2(an_3[e_9], an_3[e_9 + 1]), _f2_13), make_float2(g_0_2[e_9], g_0_2[e_9 + 1]));
                float x0_3 = approx_exp2(d2_2.x);
                float x1_3 = approx_exp2(d2_2.y);
                float2 _f32x2_mul_a_41 = make_float2(x0_3, x1_3);
                float2 _f32x2_mul_b_42 = make_float2(_unpack_bf16_7[e_9], _unpack_bf16_7[e_9 + 1]);
                float2 v2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
                v_1[e_9] = v2_1.x;
                v_1[e_9 + 1] = v2_1.y;
            }
            if (s_row >= clen) {
                #pragma unroll
                for (int e_10 = 0; e_10 < 8; e_10++) {
                    v_1[e_10] = 0.0f;
                }
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(a_taddr + 32), "r"(*reinterpret_cast<const uint32_t*>(&v_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&v_1[7])));
        }
        if (bq_0 < 2) {
            int t_off_1 = i * 2048;
            int ga_3 = gk_step_1 + g_lane + t_off_1;
            int qa_1 = q_addr + (int)qs * 1024;
            int ka_3 = k_step_2 + t_off_1;
            int sa_1 = b_addr + 8192 + bq_0 * 2048;
            float g_0_3[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(0) + 3]))
                : "r"(ga_3));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_3[(4) + 3]))
                : "r"(ga_3 ^ 16));
            unsigned int qw_3[4];
            unsigned int kw_5[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 3]))
                : "r"(qa_1));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_5[(0) + 3]))
                : "r"(ka_3));
            float shift_3 = 0.0f;
            if (bq_0 != 0) {
                shift_3 = 21.640425f;
            }
            float2 _f2_14 = make_float2(shift_3, shift_3);
            #pragma unroll
            for (int e_11 = 0; e_11 < 8; e_11 += 2) {
                float2 d2_3 = sub_f32x2(make_float2(g_0_3[e_11], g_0_3[e_11 + 1]), sub_f32x2(make_float2(an_3[e_11], an_3[e_11 + 1]), _f2_14));
                float _exp2_43 = approx_exp2(d2_3.x);
                g_0_3[e_11] = _exp2_43;
                float _exp2_44 = approx_exp2(d2_3.y);
                g_0_3[e_11 + 1] = _exp2_44;
            }
            if (clen <= i * 16 + sl) {
                #pragma unroll
                for (int e_12 = 0; e_12 < 8; e_12++) {
                    g_0_3[e_12] = 0.0f;
                }
            }
            float _unpack_bf16_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_3[_pair]) << 16);
                (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_3[_pair]) & 0xffff0000u);
            }
            float vq_1[8];
            #pragma unroll
            for (int e_13 = 0; e_13 < 8; e_13 += 2) {
                float2 _f32x2_mul_a_45 = make_float2(_unpack_bf16_8[e_13], _unpack_bf16_8[e_13 + 1]);
                float2 _f32x2_mul_b_46 = make_float2(g_0_3[e_13], g_0_3[e_13 + 1]);
                float2 q2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
                vq_1[e_13] = q2_1.x;
                vq_1[e_13 + 1] = q2_1.y;
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_1), "f"(vq_1[0]), "f"(vq_1[1]), "f"(vq_1[2]), "f"(vq_1[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_1 ^ 16), "f"(vq_1[4]), "f"(vq_1[5]), "f"(vq_1[6]), "f"(vq_1[7]) : "memory");
            float _unpack_bf16_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_5[_pair]) << 16);
                (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_5[_pair]) & 0xffff0000u);
            }
            float vk_1[8];
            #pragma unroll
            for (int e_14 = 0; e_14 < 8; e_14 += 2) {
                float2 _f32x2_mul_a_47 = make_float2(_unpack_bf16_9[e_14], _unpack_bf16_9[e_14 + 1]);
                float2 _f32x2_mul_b_48 = make_float2(g_0_3[e_14], g_0_3[e_14 + 1]);
                float2 k2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
                vk_1[e_14] = k2_1.x;
                vk_1[e_14 + 1] = k2_1.y;
            }
            int ksa_1 = sa_1 + 4096;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_1), "f"(vk_1[0]), "f"(vk_1[1]), "f"(vk_1[2]), "f"(vk_1[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_1 ^ 16), "f"(vk_1[4]), "f"(vk_1[5]), "f"(vk_1[6]), "f"(vk_1[7]) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        if (warp == 3) {
            if (elect_sync()) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_49 = (((b_ring_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_49 = (((b_ring_addr) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 68159760;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_acc), "r"(_smem_desc_lo_49), "r"(tmem_acc + 96), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                asm volatile("tcgen05.fence::before_thread_sync;");
                int i_n_1 = i + 1;
                if (i_n_1 < 4) {
                    mbarrier_arrive_expect_tx(q_full_addr + (qs) * 8, 1024);
                    tma_3d_gmem2smem(q_p_addr + qs * 1024, (&q), 0, qk_head, ext0_i + i_n_1 * 16, q_full_addr + (qs) * 8);
                }
            }
        }
        qs += 1;
        if (qs == 3) { qs = 0; _phase_q_full ^= 1; }
        int bq_5 = q4 - i - 1 & 3;
        if (fill != 0) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            float gk_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 3]))
                : "r"(gk_i + 16384));
            float tot_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tot_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tot_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tot_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tot_2[(0) + 3]))
                : "r"(t_i + 256));
            unsigned int kw_6[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_6[(0) + 1]))
                : "r"(k_i + 8192));
            unsigned int ks_2[2];
            ks_2[0] = kw_6[0] & m_f;
            ks_2[1] = kw_6[1] & m_f;
            float _unpack_bf16_10[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ks_2[_pair]) << 16);
                (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ks_2[_pair]) & 0xffff0000u);
            }
            float kb_2[4];
            float kg_2[4];
            float2 _f2_15 = make_float2(bt_f, bt_f);
            unsigned int qw_4[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(0) + 1]))
                : "r"(q_i + (int)qs * 1024));
            unsigned int qsel_2[2];
            qsel_2[0] = qw_4[0] & m_f;
            qsel_2[1] = qw_4[1] & m_f;
            float _unpack_bf16_11[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qsel_2[_pair]) << 16);
                (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qsel_2[_pair]) & 0xffff0000u);
            }
            float qg_2[4];
            float e0_2 = approx_exp2(gk_2[0]);
            float e1_3 = approx_exp2(gk_2[1]);
            float2 _f2_16 = make_float2(e0_2, e1_3);
            float2 _f32x2_mul_a_50 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
            float2 kbt_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f2_15));
            float kb0_2 = (float)(__nv_bfloat16)kbt_3.x;
            float kb1_2 = (float)(__nv_bfloat16)kbt_3.y;
            float2 _f32x2_mul_a_51 = make_float2(kb0_2, kb1_2);
            float2 kb2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&_f2_16));
            kb_2[0] = kb2_2.x;
            kb_2[1] = kb2_2.y;
            float _fma_8;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(tot_2[0]), "f"(rcp_f), "f"(-gk_2[0]));
            float _fma_9;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(tot_2[1]), "f"(rcp_f), "f"(-gk_2[1]));
            float x0_4 = approx_exp2(_fma_8);
            float x1_4 = approx_exp2(_fma_9);
            float2 _f32x2_mul_a_52 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
            float2 _f32x2_mul_b_53 = make_float2(x0_4, x1_4);
            float2 kg2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            kg_2[0] = kg2_2.x;
            kg_2[1] = kg2_2.y;
            float2 _f32x2_mul_a_54 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
            float2 qg2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f2_16));
            qg_2[0] = qg2_2.x;
            qg_2[1] = qg2_2.y;
            float e0_0_2 = approx_exp2(gk_2[2]);
            float e1_1_2 = approx_exp2(gk_2[3]);
            float2 _f2_17 = make_float2(e0_0_2, e1_1_2);
            float2 _f32x2_mul_a_55 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
            float2 kbt_2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_f2_15));
            float kb0_3_2 = (float)(__nv_bfloat16)kbt_2_2.x;
            float kb1_4_2 = (float)(__nv_bfloat16)kbt_2_2.y;
            float2 _f32x2_mul_a_56 = make_float2(kb0_3_2, kb1_4_2);
            float2 kb2_5_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_5_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f2_17));
            kb_2[2] = kb2_5_2.x;
            kb_2[3] = kb2_5_2.y;
            float _fma_10;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(tot_2[2]), "f"(rcp_f), "f"(-gk_2[2]));
            float _fma_11;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(tot_2[3]), "f"(rcp_f), "f"(-gk_2[3]));
            float x0_6_2 = approx_exp2(_fma_10);
            float x1_7_2 = approx_exp2(_fma_11);
            float2 _f32x2_mul_a_57 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
            float2 _f32x2_mul_b_58 = make_float2(x0_6_2, x1_7_2);
            float2 kg2_8_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_8_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
            kg_2[2] = kg2_8_2.x;
            kg_2[3] = kg2_8_2.y;
            float2 _f32x2_mul_a_59 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
            float2 qg2_9_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_9_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f2_17));
            qg_2[2] = qg2_9_2.x;
            qg_2[3] = qg2_9_2.y;
            unsigned int _packed_18[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_2[_lp*2 + 0], kb_2[_lp*2+1 + 0]));
                _packed_18[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_2[_lp*2 + 2], kb_2[_lp*2+1 + 2]));
                _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_60 = kb_p + 64;
                __builtin_assume(__isGlobal(_gst_60));
                reinterpret_cast<int2*>(_gst_60)[0] = reinterpret_cast<int2*>(_packed_18)[0];
            }
            unsigned int _packed_19[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_2[_lp*2 + 0], kg_2[_lp*2+1 + 0]));
                _packed_19[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_2[_lp*2 + 2], kg_2[_lp*2+1 + 2]));
                _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_61 = kg_p + 64;
                __builtin_assume(__isGlobal(_gst_61));
                reinterpret_cast<int2*>(_gst_61)[0] = reinterpret_cast<int2*>(_packed_19)[0];
            }
            unsigned int _packed_20[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_2[_lp*2 + 0], qg_2[_lp*2+1 + 0]));
                _packed_20[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_2[_lp*2 + 2], qg_2[_lp*2+1 + 2]));
                _packed_20[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_62 = qg_p + 64;
                __builtin_assume(__isGlobal(_gst_62));
                reinterpret_cast<int2*>(_gst_62)[0] = reinterpret_cast<int2*>(_packed_20)[0];
            }
        }
        mbarrier_wait(ring_empty_addr, _phase_ring_empty_0);
        _phase_ring_empty_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (bq_5 < 2) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        int gk_step_6 = gk0 + 16384;
        int k_step_7 = k_addr + 8192 + kl0;
        float an_8[8];
        int aa_9 = gk_step_6 + i * 2048 + an_lane;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(0) + 3]))
            : "r"(aa_9));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_8[(4) + 3]))
            : "r"(aa_9 + 16));
        if (q4 <= i) {
            int use_gc_2 = 0;
            if (q4 == i) {
                use_gc_2 = 1;
            }
            int s_off_2 = q4 * 2048;
            int ga_4 = gk_step_6 + g_lane + s_off_2;
            int ka_4 = k_step_7 + s_off_2;
            float g_0_4[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(0) + 3]))
                : "r"(ga_4));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_4[(4) + 3]))
                : "r"(ga_4 ^ 16));
            unsigned int kw_7[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_7[(0) + 3]))
                : "r"(ka_4));
            float _unpack_bf16_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_7[_pair]) << 16);
                (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_7[_pair]) & 0xffff0000u);
            }
            float shift_4 = 0.0f;
            if (use_gc_2 != 0) {
                shift_4 = 21.640425f;
            }
            float2 _f2_18 = make_float2(shift_4, shift_4);
            float v_2[8];
            #pragma unroll
            for (int e_15 = 0; e_15 < 8; e_15 += 2) {
                float2 d2_4 = sub_f32x2(sub_f32x2(make_float2(an_8[e_15], an_8[e_15 + 1]), _f2_18), make_float2(g_0_4[e_15], g_0_4[e_15 + 1]));
                float x0_5 = approx_exp2(d2_4.x);
                float x1_5 = approx_exp2(d2_4.y);
                float2 _f32x2_mul_a_63 = make_float2(x0_5, x1_5);
                float2 _f32x2_mul_b_64 = make_float2(_unpack_bf16_12[e_15], _unpack_bf16_12[e_15 + 1]);
                float2 v2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                v_2[e_15] = v2_2.x;
                v_2[e_15 + 1] = v2_2.y;
            }
            if (s_row >= clen) {
                #pragma unroll
                for (int e_16 = 0; e_16 < 8; e_16++) {
                    v_2[e_16] = 0.0f;
                }
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(a_taddr), "r"(*reinterpret_cast<const uint32_t*>(&v_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&v_2[7])));
        }
        if (bq_5 < 2) {
            int t_off_2 = i * 2048;
            int ga_5 = gk_step_6 + g_lane + t_off_2;
            int qa_2 = q_addr + (int)qs * 1024;
            int ka_5 = k_step_7 + t_off_2;
            int sa_2 = b_addr + bq_5 * 2048;
            float g_0_5[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(0) + 3]))
                : "r"(ga_5));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_5[(4) + 3]))
                : "r"(ga_5 ^ 16));
            unsigned int qw_5[4];
            unsigned int kw_8[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 3]))
                : "r"(qa_2));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_8[(0) + 3]))
                : "r"(ka_5));
            float shift_5 = 0.0f;
            if (bq_5 != 0) {
                shift_5 = 21.640425f;
            }
            float2 _f2_19 = make_float2(shift_5, shift_5);
            #pragma unroll
            for (int e_17 = 0; e_17 < 8; e_17 += 2) {
                float2 d2_5 = sub_f32x2(make_float2(g_0_5[e_17], g_0_5[e_17 + 1]), sub_f32x2(make_float2(an_8[e_17], an_8[e_17 + 1]), _f2_19));
                float _exp2_65 = approx_exp2(d2_5.x);
                g_0_5[e_17] = _exp2_65;
                float _exp2_66 = approx_exp2(d2_5.y);
                g_0_5[e_17 + 1] = _exp2_66;
            }
            if (clen <= i * 16 + sl) {
                #pragma unroll
                for (int e_18 = 0; e_18 < 8; e_18++) {
                    g_0_5[e_18] = 0.0f;
                }
            }
            float _unpack_bf16_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_5[_pair]) << 16);
                (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_5[_pair]) & 0xffff0000u);
            }
            float vq_2[8];
            #pragma unroll
            for (int e_19 = 0; e_19 < 8; e_19 += 2) {
                float2 _f32x2_mul_a_67 = make_float2(_unpack_bf16_13[e_19], _unpack_bf16_13[e_19 + 1]);
                float2 _f32x2_mul_b_68 = make_float2(g_0_5[e_19], g_0_5[e_19 + 1]);
                float2 q2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
                vq_2[e_19] = q2_2.x;
                vq_2[e_19 + 1] = q2_2.y;
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_2), "f"(vq_2[0]), "f"(vq_2[1]), "f"(vq_2[2]), "f"(vq_2[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_2 ^ 16), "f"(vq_2[4]), "f"(vq_2[5]), "f"(vq_2[6]), "f"(vq_2[7]) : "memory");
            float _unpack_bf16_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_8[_pair]) << 16);
                (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_8[_pair]) & 0xffff0000u);
            }
            float vk_2[8];
            #pragma unroll
            for (int e_20 = 0; e_20 < 8; e_20 += 2) {
                float2 _f32x2_mul_a_69 = make_float2(_unpack_bf16_14[e_20], _unpack_bf16_14[e_20 + 1]);
                float2 _f32x2_mul_b_70 = make_float2(g_0_5[e_20], g_0_5[e_20 + 1]);
                float2 k2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
                vk_2[e_20] = k2_2.x;
                vk_2[e_20 + 1] = k2_2.y;
            }
            int ksa_2 = sa_2 + 4096;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_2), "f"(vk_2[0]), "f"(vk_2[1]), "f"(vk_2[2]), "f"(vk_2[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_2 ^ 16), "f"(vk_2[4]), "f"(vk_2[5]), "f"(vk_2[6]), "f"(vk_2[7]) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        if (warp == 3) {
            if (elect_sync()) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_71 = (((b_ring_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_71 = (((b_ring_addr) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 68159760;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_acc), "r"(_smem_desc_lo_71), "r"(tmem_acc + 64), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                int i_n_2 = i + 1;
                if (i_n_2 < 4) {
                    mbarrier_arrive_expect_tx(q_full_addr + (qs) * 8, 1024);
                    tma_3d_gmem2smem(q_p_addr + qs * 1024, (&q), 32, qk_head, ext0_i + i_n_2 * 16, q_full_addr + (qs) * 8);
                }
            }
        }
        qs += 1;
        if (qs == 3) { qs = 0; _phase_q_full ^= 1; }
        int bq_10 = q4 - i - 1 & 3;
        if (fill != 0) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            float gk_3[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 3]))
                : "r"(gk_i + 24576));
            float tot_3[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tot_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&tot_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tot_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tot_3[(0) + 3]))
                : "r"(t_i + 384));
            unsigned int kw_9[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_9[(0) + 1]))
                : "r"(k_i1 + 8192));
            unsigned int ks_3[2];
            ks_3[0] = kw_9[0] & m_f;
            ks_3[1] = kw_9[1] & m_f;
            float _unpack_bf16_15[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ks_3[_pair]) << 16);
                (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ks_3[_pair]) & 0xffff0000u);
            }
            float kb_3[4];
            float kg_3[4];
            float2 _f2_20 = make_float2(bt_f, bt_f);
            unsigned int qw_6[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(0) + 1]))
                : "r"(q_i + (int)qs * 1024));
            unsigned int qsel_3[2];
            qsel_3[0] = qw_6[0] & m_f;
            qsel_3[1] = qw_6[1] & m_f;
            float _unpack_bf16_16[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qsel_3[_pair]) << 16);
                (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qsel_3[_pair]) & 0xffff0000u);
            }
            float qg_3[4];
            float e0_3 = approx_exp2(gk_3[0]);
            float e1_4 = approx_exp2(gk_3[1]);
            float2 _f2_21 = make_float2(e0_3, e1_4);
            float2 _f32x2_mul_a_72 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
            float2 kbt_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f2_20));
            float kb0_4 = (float)(__nv_bfloat16)kbt_4.x;
            float kb1_3 = (float)(__nv_bfloat16)kbt_4.y;
            float2 _f32x2_mul_a_73 = make_float2(kb0_4, kb1_3);
            float2 kb2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_f2_21));
            kb_3[0] = kb2_3.x;
            kb_3[1] = kb2_3.y;
            float _fma_12;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(tot_3[0]), "f"(rcp_f), "f"(-gk_3[0]));
            float _fma_13;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(tot_3[1]), "f"(rcp_f), "f"(-gk_3[1]));
            float x0_7 = approx_exp2(_fma_12);
            float x1_6 = approx_exp2(_fma_13);
            float2 _f32x2_mul_a_74 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
            float2 _f32x2_mul_b_75 = make_float2(x0_7, x1_6);
            float2 kg2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
            kg_3[0] = kg2_3.x;
            kg_3[1] = kg2_3.y;
            float2 _f32x2_mul_a_76 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
            float2 qg2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f2_21));
            qg_3[0] = qg2_3.x;
            qg_3[1] = qg2_3.y;
            float e0_0_3 = approx_exp2(gk_3[2]);
            float e1_1_3 = approx_exp2(gk_3[3]);
            float2 _f2_22 = make_float2(e0_0_3, e1_1_3);
            float2 _f32x2_mul_a_77 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
            float2 kbt_2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kbt_2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f2_20));
            float kb0_3_3 = (float)(__nv_bfloat16)kbt_2_3.x;
            float kb1_4_3 = (float)(__nv_bfloat16)kbt_2_3.y;
            float2 _f32x2_mul_a_78 = make_float2(kb0_3_3, kb1_4_3);
            float2 kb2_5_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kb2_5_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f2_22));
            kb_3[2] = kb2_5_3.x;
            kb_3[3] = kb2_5_3.y;
            float _fma_14;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(tot_3[2]), "f"(rcp_f), "f"(-gk_3[2]));
            float _fma_15;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(tot_3[3]), "f"(rcp_f), "f"(-gk_3[3]));
            float x0_6_3 = approx_exp2(_fma_14);
            float x1_7_3 = approx_exp2(_fma_15);
            float2 _f32x2_mul_a_79 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
            float2 _f32x2_mul_b_80 = make_float2(x0_6_3, x1_7_3);
            float2 kg2_8_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kg2_8_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
            kg_3[2] = kg2_8_3.x;
            kg_3[3] = kg2_8_3.y;
            float2 _f32x2_mul_a_81 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
            float2 qg2_9_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&qg2_9_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f2_22));
            qg_3[2] = qg2_9_3.x;
            qg_3[3] = qg2_9_3.y;
            unsigned int _packed_21[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_3[_lp*2 + 0], kb_3[_lp*2+1 + 0]));
                _packed_21[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb_3[_lp*2 + 2], kb_3[_lp*2+1 + 2]));
                _packed_21[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_82 = kb_p + 96;
                __builtin_assume(__isGlobal(_gst_82));
                reinterpret_cast<int2*>(_gst_82)[0] = reinterpret_cast<int2*>(_packed_21)[0];
            }
            unsigned int _packed_22[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_3[_lp*2 + 0], kg_3[_lp*2+1 + 0]));
                _packed_22[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg_3[_lp*2 + 2], kg_3[_lp*2+1 + 2]));
                _packed_22[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_83 = kg_p + 96;
                __builtin_assume(__isGlobal(_gst_83));
                reinterpret_cast<int2*>(_gst_83)[0] = reinterpret_cast<int2*>(_packed_22)[0];
            }
            unsigned int _packed_23[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_3[_lp*2 + 0], qg_3[_lp*2+1 + 0]));
                _packed_23[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qg_3[_lp*2 + 2], qg_3[_lp*2+1 + 2]));
                _packed_23[_lp + 1] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_84 = qg_p + 96;
                __builtin_assume(__isGlobal(_gst_84));
                reinterpret_cast<int2*>(_gst_84)[0] = reinterpret_cast<int2*>(_packed_23)[0];
            }
        }
        mbarrier_wait(ring_empty_addr + 8, _phase_ring_empty_1);
        _phase_ring_empty_1 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (bq_10 < 2) {
            mbarrier_wait(q_full_addr + (qs) * 8, _phase_q_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        int gk_step_11 = gk0 + 24576;
        int k_step_12 = k_addr + 8192 + kl1;
        float an_13[8];
        int aa_14 = gk_step_11 + i * 2048 + an_lane;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(0) + 3]))
            : "r"(aa_14));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&an_13[(4) + 3]))
            : "r"(aa_14 + 16));
        if (q4 <= i) {
            int use_gc_3 = 0;
            if (q4 == i) {
                use_gc_3 = 1;
            }
            int s_off_3 = q4 * 2048;
            int ga_6 = gk_step_11 + g_lane + s_off_3;
            int ka_6 = k_step_12 + s_off_3;
            float g_0_6[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(0) + 3]))
                : "r"(ga_6));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_6[(4) + 3]))
                : "r"(ga_6 ^ 16));
            unsigned int kw_10[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_10[(0) + 3]))
                : "r"(ka_6));
            float _unpack_bf16_17[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_10[_pair]) << 16);
                (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_10[_pair]) & 0xffff0000u);
            }
            float shift_6 = 0.0f;
            if (use_gc_3 != 0) {
                shift_6 = 21.640425f;
            }
            float2 _f2_23 = make_float2(shift_6, shift_6);
            float v_3[8];
            #pragma unroll
            for (int e_21 = 0; e_21 < 8; e_21 += 2) {
                float2 d2_6 = sub_f32x2(sub_f32x2(make_float2(an_13[e_21], an_13[e_21 + 1]), _f2_23), make_float2(g_0_6[e_21], g_0_6[e_21 + 1]));
                float x0_8 = approx_exp2(d2_6.x);
                float x1_8 = approx_exp2(d2_6.y);
                float2 _f32x2_mul_a_85 = make_float2(x0_8, x1_8);
                float2 _f32x2_mul_b_86 = make_float2(_unpack_bf16_17[e_21], _unpack_bf16_17[e_21 + 1]);
                float2 v2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                v_3[e_21] = v2_3.x;
                v_3[e_21 + 1] = v2_3.y;
            }
            if (s_row >= clen) {
                #pragma unroll
                for (int e_22 = 0; e_22 < 8; e_22++) {
                    v_3[e_22] = 0.0f;
                }
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(a_taddr + 32), "r"(*reinterpret_cast<const uint32_t*>(&v_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&v_3[7])));
        }
        if (bq_10 < 2) {
            int t_off_3 = i * 2048;
            int ga_7 = gk_step_11 + g_lane + t_off_3;
            int qa_3 = q_addr + (int)qs * 1024;
            int ka_7 = k_step_12 + t_off_3;
            int sa_3 = b_addr + 8192 + bq_10 * 2048;
            float g_0_7[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(0) + 3]))
                : "r"(ga_7));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_0_7[(4) + 3]))
                : "r"(ga_7 ^ 16));
            unsigned int qw_7[4];
            unsigned int kw_11[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 3]))
                : "r"(qa_3));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&kw_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&kw_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw_11[(0) + 3]))
                : "r"(ka_7));
            float shift_7 = 0.0f;
            if (bq_10 != 0) {
                shift_7 = 21.640425f;
            }
            float2 _f2_24 = make_float2(shift_7, shift_7);
            #pragma unroll
            for (int e_23 = 0; e_23 < 8; e_23 += 2) {
                float2 d2_7 = sub_f32x2(make_float2(g_0_7[e_23], g_0_7[e_23 + 1]), sub_f32x2(make_float2(an_13[e_23], an_13[e_23 + 1]), _f2_24));
                float _exp2_87 = approx_exp2(d2_7.x);
                g_0_7[e_23] = _exp2_87;
                float _exp2_88 = approx_exp2(d2_7.y);
                g_0_7[e_23 + 1] = _exp2_88;
            }
            if (clen <= i * 16 + sl) {
                #pragma unroll
                for (int e_24 = 0; e_24 < 8; e_24++) {
                    g_0_7[e_24] = 0.0f;
                }
            }
            float _unpack_bf16_18[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_7[_pair]) << 16);
                (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_7[_pair]) & 0xffff0000u);
            }
            float vq_3[8];
            #pragma unroll
            for (int e_25 = 0; e_25 < 8; e_25 += 2) {
                float2 _f32x2_mul_a_89 = make_float2(_unpack_bf16_18[e_25], _unpack_bf16_18[e_25 + 1]);
                float2 _f32x2_mul_b_90 = make_float2(g_0_7[e_25], g_0_7[e_25 + 1]);
                float2 q2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                vq_3[e_25] = q2_3.x;
                vq_3[e_25 + 1] = q2_3.y;
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_3), "f"(vq_3[0]), "f"(vq_3[1]), "f"(vq_3[2]), "f"(vq_3[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sa_3 ^ 16), "f"(vq_3[4]), "f"(vq_3[5]), "f"(vq_3[6]), "f"(vq_3[7]) : "memory");
            float _unpack_bf16_19[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_11[_pair]) << 16);
                (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_11[_pair]) & 0xffff0000u);
            }
            float vk_3[8];
            #pragma unroll
            for (int e_26 = 0; e_26 < 8; e_26 += 2) {
                float2 _f32x2_mul_a_91 = make_float2(_unpack_bf16_19[e_26], _unpack_bf16_19[e_26 + 1]);
                float2 _f32x2_mul_b_92 = make_float2(g_0_7[e_26], g_0_7[e_26 + 1]);
                float2 k2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&k2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                vk_3[e_26] = k2_3.x;
                vk_3[e_26 + 1] = k2_3.y;
            }
            int ksa_3 = sa_3 + 4096;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_3), "f"(vk_3[0]), "f"(vk_3[1]), "f"(vk_3[2]), "f"(vk_3[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ksa_3 ^ 16), "f"(vk_3[4]), "f"(vk_3[5]), "f"(vk_3[6]), "f"(vk_3[7]) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        if (warp == 3) {
            if (elect_sync()) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_93 = (((b_ring_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_93 = (((b_ring_addr) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 68159760;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_acc), "r"(_smem_desc_lo_93), "r"(tmem_acc + 96), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                tcgen05_commit(acc_done_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                int i_n_3 = i + 1;
                if (i_n_3 < 4) {
                    mbarrier_arrive_expect_tx(q_full_addr + (qs) * 8, 1024);
                    tma_3d_gmem2smem(q_p_addr + qs * 1024, (&q), 64, qk_head, ext0_i + i_n_3 * 16, q_full_addr + (qs) * 8);
                }
            }
        }
        qs += 1;
        if (qs == 3) { qs = 0; _phase_q_full ^= 1; }
    }
    mbarrier_wait_hint(acc_done_addr, _phase_acc_done_0, 1000000);
    _phase_acc_done_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    int tbase_1 = 48;
    if (q4 <= 3) {
        int col_4 = h4 * 32;
        if (q4 == 3) {
            col_4 = col_4 + 16;
        }
        float _tmem_load_1[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
            : "r"(taddr + (unsigned int)row_base + (unsigned int)col_4));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (h4 == 0) {
            #pragma unroll
            for (int j_9 = 0; j_9 < 8; j_9++) {
                _tmem_load_1[j_9] = _tmem_load_1[j_9] * scale;
            }
            if (q4 == 3) {
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    if (sl > th + j_10) {
                        _tmem_load_1[j_10] = 0.0f;
                    }
                }
            }
            int sl_0_1 = lane_1 % 16;
            int th_1_1 = lane_1 / 16 * 8;
            #pragma unroll
            for (int j_11 = 0; j_11 < 8; j_11++) {
                {
                    uint32_t _addr_94 = static_cast<uint32_t>(gk0 + w * 1152 + ((th_1_1 + j_11) * 18 + sl_0_1) * 4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(_tmem_load_1[j_11]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tl_1 = lane_1 / 2;
            int hh_2_1 = lane_1 % 2 * 8;
            int raddr_1 = gk0 + w * 1152 + (tl_1 * 18 + hh_2_1) * 4;
            float vals_1[8];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[(0) + 1]))
                : "r"(raddr_1));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[(2) + 1]))
                : "r"(raddr_1 + 8));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[(4) + 1]))
                : "r"(raddr_1 + 16));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[(6) + 1]))
                : "r"(raddr_1 + 24));
            int t_d_3 = tbase_1 + tl_1;
            int col_3_1 = scol + hh_2_1;
            if (write_ext != 0) {
                if (t_d_3 < clen) {
                    unsigned int _packed_24[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                        _packed_24[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 2], vals_1[_lp*2+1 + 2]));
                        _packed_24[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 4], vals_1[_lp*2+1 + 4]));
                        _packed_24[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 6], vals_1[_lp*2+1 + 6]));
                        _packed_24[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(aqk_ext + (abase + (long long)(t_d_3 * ars) + (long long)col_3_1))[0] = reinterpret_cast<int4*>(_packed_24)[0];
                }
            }
            if (write_int != 0) {
                if (t_d_3 >= clen) {
                    #pragma unroll
                    for (int e_27 = 0; e_27 < 8; e_27++) {
                        vals_1[e_27] = 0.0f;
                    }
                }
                unsigned int _packed_25[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                    _packed_25[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 2], vals_1[_lp*2+1 + 2]));
                    _packed_25[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 4], vals_1[_lp*2+1 + 4]));
                    _packed_25[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 6], vals_1[_lp*2+1 + 6]));
                    _packed_25[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(aqk_int + (ibase + (long long)(t_d_3 * ars) + (long long)col_3_1))[0] = reinterpret_cast<int4*>(_packed_25)[0];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
        } else {
            float bt_1[8];
            int baddr_1 = (int)beta_s_addr + (tbase_1 + th) * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bt_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(0) + 3]))
                : "r"(baddr_1));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bt_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bt_1[(4) + 3]))
                : "r"(baddr_1 + 16));
            #pragma unroll
            for (int j_12 = 0; j_12 < 8; j_12++) {
                _tmem_load_1[j_12] = _tmem_load_1[j_12] * bt_1[j_12];
            }
            int c4_1 = sl >> 2;
            int low_1 = (sl & 3) * 4;
            if (q4 == 3) {
                #pragma unroll
                for (int j_13 = 0; j_13 < 8; j_13++) {
                    if (sl >= th + j_13) {
                        _tmem_load_1[j_13] = 0.0f;
                    }
                }
                #pragma unroll
                for (int j_14 = 0; j_14 < 8; j_14++) {
                    _tmem_load_1[j_14] = -_tmem_load_1[j_14];
                }
                int base_t_3 = (int)(aii_s_addr + (unsigned int)(q4 * 1024)) + th * 64 + low_1;
                #pragma unroll
                for (int j_15 = 0; j_15 < 8; j_15++) {
                    {
                        uint32_t _addr_95 = static_cast<uint32_t>(base_t_3 + j_15 * 64 + ((c4_1 ^ j_15 >> 1) << 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(_tmem_load_1[j_15]) : "memory");
                    }
                }
            } else {
                int lb_1 = (int)l1_s_addr;
                lb_1 = (int)l3_s_addr + q4 * 1024;
                int base_t_4 = lb_1 + th * 64 + low_1;
                #pragma unroll
                for (int j_16 = 0; j_16 < 8; j_16++) {
                    {
                        uint32_t _addr_96 = static_cast<uint32_t>(base_t_4 + j_16 * 64 + ((c4_1 ^ j_16 >> 1) << 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(_tmem_load_1[j_16]) : "memory");
                    }
                }
            }
        }
    } else if (h4 == 0) {
        float zero8_3[8];
        zero8_3[0] = 0.0f;
        zero8_3[1] = 0.0f;
        zero8_3[2] = 0.0f;
        zero8_3[3] = 0.0f;
        zero8_3[4] = 0.0f;
        zero8_3[5] = 0.0f;
        zero8_3[6] = 0.0f;
        zero8_3[7] = 0.0f;
        int t_d_4 = tbase_1 + lane_1 / 2;
        int col_5 = scol + lane_1 % 2 * 8;
        if (write_ext != 0) {
            if (t_d_4 < clen) {
                unsigned int _packed_26[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 0], zero8_3[_lp*2+1 + 0]));
                    _packed_26[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 2], zero8_3[_lp*2+1 + 2]));
                    _packed_26[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 4], zero8_3[_lp*2+1 + 4]));
                    _packed_26[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 6], zero8_3[_lp*2+1 + 6]));
                    _packed_26[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(aqk_ext + (abase + (long long)(t_d_4 * ars) + (long long)col_5))[0] = reinterpret_cast<int4*>(_packed_26)[0];
            }
        }
        if (write_int != 0) {
            unsigned int _packed_27[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 0], zero8_3[_lp*2+1 + 0]));
                _packed_27[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 2], zero8_3[_lp*2+1 + 2]));
                _packed_27[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 4], zero8_3[_lp*2+1 + 4]));
                _packed_27[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_3[_lp*2 + 6], zero8_3[_lp*2+1 + 6]));
                _packed_27[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(aqk_int + (ibase + (long long)(t_d_4 * ars) + (long long)col_5))[0] = reinterpret_cast<int4*>(_packed_27)[0];
        }
    } else {
        float zero8_4[8];
        zero8_4[0] = 0.0f;
        zero8_4[1] = 0.0f;
        zero8_4[2] = 0.0f;
        zero8_4[3] = 0.0f;
        zero8_4[4] = 0.0f;
        zero8_4[5] = 0.0f;
        zero8_4[6] = 0.0f;
        zero8_4[7] = 0.0f;
        int t_d_5 = tbase_1 + lane_1 / 2;
        int col_6 = scol + lane_1 % 2 * 8;
        if (write_ext != 0) {
            if (t_d_5 < clen) {
                unsigned int _packed_28[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 0], zero8_4[_lp*2+1 + 0]));
                    _packed_28[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 2], zero8_4[_lp*2+1 + 2]));
                    _packed_28[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 4], zero8_4[_lp*2+1 + 4]));
                    _packed_28[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 6], zero8_4[_lp*2+1 + 6]));
                    _packed_28[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_5 * ars) + (long long)col_6))[0] = reinterpret_cast<int4*>(_packed_28)[0];
            }
        }
        if (write_int != 0) {
            unsigned int _packed_29[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 0], zero8_4[_lp*2+1 + 0]));
                _packed_29[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 2], zero8_4[_lp*2+1 + 2]));
                _packed_29[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 4], zero8_4[_lp*2+1 + 4]));
                _packed_29[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8_4[_lp*2 + 6], zero8_4[_lp*2+1 + 6]));
                _packed_29[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_5 * ars) + (long long)col_6))[0] = reinterpret_cast<int4*>(_packed_29)[0];
        }
    }
    if (h4 == 0) {
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.arrive 3, 256;" ::: "memory");
        float zero16[16];
        zero16[0] = 0.0f;
        zero16[1] = 0.0f;
        zero16[2] = 0.0f;
        zero16[3] = 0.0f;
        zero16[4] = 0.0f;
        zero16[5] = 0.0f;
        zero16[6] = 0.0f;
        zero16[7] = 0.0f;
        zero16[8] = 0.0f;
        zero16[9] = 0.0f;
        zero16[10] = 0.0f;
        zero16[11] = 0.0f;
        zero16[12] = 0.0f;
        zero16[13] = 0.0f;
        zero16[14] = 0.0f;
        zero16[15] = 0.0f;
        if (q4 != 0) {
            tmem_st_x16_f32(taddr + (unsigned int)row_base, zero16);
        }
        if (q4 != 1) {
            tmem_st_x16_f32(taddr + (unsigned int)row_base + 16, zero16);
        }
        if (q4 == 3) {
            tmem_st_x16_f32(taddr + (unsigned int)row_base + 32, zero16);
        }
        if (q4 != 3) {
            tmem_st_x16_f32(taddr + (unsigned int)row_base + 48, zero16);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    float Ai[16];
    if (h4 != 0) {
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
        float lcol[16];
        int c4_2 = (lane_1 & 15) >> 2;
        int tile_q = (int)(aii_s_addr + (unsigned int)(q4 * 1024));
        int cbase = tile_q + (lane_1 & 3) * 4;
        #pragma unroll
        for (int r_2 = 0; r_2 < 16; r_2++) {
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lcol[r_2])) : "r"(cbase + r_2 * 64 + ((c4_2 ^ r_2 >> 1 & 3) << 4)));
        }
        #pragma unroll
        for (int r_3 = 0; r_3 < 16; r_3++) {
            float a_r = 0.0f;
            if (lane_1 < 16) {
                a_r = lcol[r_3];
            }
            Ai[r_3] = a_r;
        }
        float ra[4];
        float rb_0[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra[(0) + 3]))
            : "r"(tile_q + 144));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_0[(0) + 3]))
            : "r"(tile_q + 208));
        float a_c = Ai[2];
        float b_c = Ai[3];
        float2 tot2 = make_float2(a_c, b_c);
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            float a_j = ra[jj];
            float b_j = rb_0[jj];
            tot2 = fma_f32x2_rn_ftz(make_float2(a_j, b_j), make_float2(Ai[jj], Ai[jj]), tot2);
        }
        Ai[2] = tot2.x;
        float b_i = rb_0[2];
        float tot_b = tot2.y;
        tot_b = __fmaf_rn(b_i, Ai[2], tot_b);
        Ai[3] = tot_b;
        float ra_1[4];
        float rb_2[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_1[(0) + 3]))
            : "r"(tile_q + 288));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(0) + 3]))
            : "r"(tile_q + 352));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_2[(4) + 3]))
            : "r"(tile_q + 368));
        float a_c_3 = Ai[4];
        float b_c_4 = Ai[5];
        float2 tot2_5 = make_float2(a_c_3, b_c_4);
        #pragma unroll
        for (int jj_1 = 0; jj_1 < 4; jj_1++) {
            float a_j_1 = ra_1[jj_1];
            float b_j_1 = rb_2[jj_1];
            tot2_5 = fma_f32x2_rn_ftz(make_float2(a_j_1, b_j_1), make_float2(Ai[jj_1], Ai[jj_1]), tot2_5);
        }
        Ai[4] = tot2_5.x;
        float b_i_6 = rb_2[4];
        float tot_b_7 = tot2_5.y;
        tot_b_7 = __fmaf_rn(b_i_6, Ai[4], tot_b_7);
        Ai[5] = tot_b_7;
        float ra_8[8];
        float rb_9[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(0) + 3]))
            : "r"(tile_q + 432));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_8[(4) + 3]))
            : "r"(tile_q + 416));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(0) + 3]))
            : "r"(tile_q + 496));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_9[(4) + 3]))
            : "r"(tile_q + 480));
        float a_c_10 = Ai[6];
        float b_c_11 = Ai[7];
        float2 tot2_12 = make_float2(a_c_10, b_c_11);
        #pragma unroll
        for (int jj_2 = 0; jj_2 < 6; jj_2++) {
            float a_j_2 = ra_8[jj_2];
            float b_j_2 = rb_9[jj_2];
            tot2_12 = fma_f32x2_rn_ftz(make_float2(a_j_2, b_j_2), make_float2(Ai[jj_2], Ai[jj_2]), tot2_12);
        }
        Ai[6] = tot2_12.x;
        float b_i_13 = rb_9[6];
        float tot_b_14 = tot2_12.y;
        tot_b_14 = __fmaf_rn(b_i_13, Ai[6], tot_b_14);
        Ai[7] = tot_b_14;
        float ra_15[8];
        float rb_16[12];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(0) + 3]))
            : "r"(tile_q + 512));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_15[(4) + 3]))
            : "r"(tile_q + 528));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(0) + 3]))
            : "r"(tile_q + 576));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(4) + 3]))
            : "r"(tile_q + 592));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_16[(8) + 3]))
            : "r"(tile_q + 608));
        float a_c_17 = Ai[8];
        float b_c_18 = Ai[9];
        float2 tot2_19 = make_float2(a_c_17, b_c_18);
        #pragma unroll
        for (int jj_3 = 0; jj_3 < 8; jj_3++) {
            float a_j_3 = ra_15[jj_3];
            float b_j_3 = rb_16[jj_3];
            tot2_19 = fma_f32x2_rn_ftz(make_float2(a_j_3, b_j_3), make_float2(Ai[jj_3], Ai[jj_3]), tot2_19);
        }
        Ai[8] = tot2_19.x;
        float b_i_20 = rb_16[8];
        float tot_b_21 = tot2_19.y;
        tot_b_21 = __fmaf_rn(b_i_20, Ai[8], tot_b_21);
        Ai[9] = tot_b_21;
        float ra_22[12];
        float rb_23[12];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(0) + 3]))
            : "r"(tile_q + 656));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(4) + 3]))
            : "r"(tile_q + 640));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_22[(8) + 3]))
            : "r"(tile_q + 688));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(0) + 3]))
            : "r"(tile_q + 720));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(4) + 3]))
            : "r"(tile_q + 704));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_23[(8) + 3]))
            : "r"(tile_q + 752));
        float a_c_24 = Ai[10];
        float b_c_25 = Ai[11];
        float2 tot2_26 = make_float2(a_c_24, b_c_25);
        #pragma unroll
        for (int jj_4 = 0; jj_4 < 10; jj_4++) {
            float a_j_4 = ra_22[jj_4];
            float b_j_4 = rb_23[jj_4];
            tot2_26 = fma_f32x2_rn_ftz(make_float2(a_j_4, b_j_4), make_float2(Ai[jj_4], Ai[jj_4]), tot2_26);
        }
        Ai[10] = tot2_26.x;
        float b_i_27 = rb_23[10];
        float tot_b_28 = tot2_26.y;
        tot_b_28 = __fmaf_rn(b_i_27, Ai[10], tot_b_28);
        Ai[11] = tot_b_28;
        float ra_29[12];
        float rb_30[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(0) + 3]))
            : "r"(tile_q + 800));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(4) + 3]))
            : "r"(tile_q + 816));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_29[(8) + 3]))
            : "r"(tile_q + 768));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(0) + 3]))
            : "r"(tile_q + 864));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(4) + 3]))
            : "r"(tile_q + 880));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(8) + 3]))
            : "r"(tile_q + 832));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_30[12])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_30[(12) + 3]))
            : "r"(tile_q + 848));
        float a_c_31 = Ai[12];
        float b_c_32 = Ai[13];
        float2 tot2_33 = make_float2(a_c_31, b_c_32);
        #pragma unroll
        for (int jj_5 = 0; jj_5 < 12; jj_5++) {
            float a_j_5 = ra_29[jj_5];
            float b_j_5 = rb_30[jj_5];
            tot2_33 = fma_f32x2_rn_ftz(make_float2(a_j_5, b_j_5), make_float2(Ai[jj_5], Ai[jj_5]), tot2_33);
        }
        Ai[12] = tot2_33.x;
        float b_i_34 = rb_30[12];
        float tot_b_35 = tot2_33.y;
        tot_b_35 = __fmaf_rn(b_i_34, Ai[12], tot_b_35);
        Ai[13] = tot_b_35;
        float ra_36[16];
        float rb_37[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(0) + 3]))
            : "r"(tile_q + 944));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(4) + 3]))
            : "r"(tile_q + 928));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(8) + 3]))
            : "r"(tile_q + 912));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ra_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ra_36[(12) + 3]))
            : "r"(tile_q + 896));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(0) + 3]))
            : "r"(tile_q + 1008));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(4) + 3]))
            : "r"(tile_q + 992));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_37[8])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(8) + 3]))
            : "r"(tile_q + 976));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rb_37[12])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rb_37[(12) + 3]))
            : "r"(tile_q + 960));
        float a_c_38 = Ai[14];
        float b_c_39 = Ai[15];
        float2 tot2_40 = make_float2(a_c_38, b_c_39);
        #pragma unroll
        for (int jj_6 = 0; jj_6 < 14; jj_6++) {
            float a_j_6 = ra_36[jj_6];
            float b_j_6 = rb_37[jj_6];
            tot2_40 = fma_f32x2_rn_ftz(make_float2(a_j_6, b_j_6), make_float2(Ai[jj_6], Ai[jj_6]), tot2_40);
        }
        Ai[14] = tot2_40.x;
        float b_i_41 = rb_37[14];
        float tot_b_42 = tot2_40.y;
        tot_b_42 = __fmaf_rn(b_i_41, Ai[14], tot_b_42);
        Ai[15] = tot_b_42;
        #pragma unroll
        for (int r_4 = 0; r_4 < 16; r_4++) {
            float v_r = Ai[r_4];
            if (lane_1 == r_4) {
                v_r = v_r + 1.0f;
            }
            if (lane_1 >= 16) {
                v_r = 0.0f;
            }
            Ai[r_4] = v_r;
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (lane_1 < 16) {
            #pragma unroll
            for (int r_5 = 0; r_5 < 16; r_5++) {
                {
                    uint32_t _addr_97 = static_cast<uint32_t>(cbase + r_5 * 64 + ((c4_2 ^ r_5 >> 1 & 3) << 4));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(Ai[r_5]) : "memory");
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 3, 256;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
        tmem_st_x16_f32(taddr + (unsigned int)row_base + (unsigned int)(q4 * 16), Ai);
        if (q4 < 2) {
            float zero16_1[16];
            zero16_1[0] = 0.0f;
            zero16_1[1] = 0.0f;
            zero16_1[2] = 0.0f;
            zero16_1[3] = 0.0f;
            zero16_1[4] = 0.0f;
            zero16_1[5] = 0.0f;
            zero16_1[6] = 0.0f;
            zero16_1[7] = 0.0f;
            zero16_1[8] = 0.0f;
            zero16_1[9] = 0.0f;
            zero16_1[10] = 0.0f;
            zero16_1[11] = 0.0f;
            zero16_1[12] = 0.0f;
            zero16_1[13] = 0.0f;
            zero16_1[14] = 0.0f;
            zero16_1[15] = 0.0f;
            tmem_st_x16_f32(taddr + (unsigned int)row_base + 32, zero16_1);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    unsigned int _phase_hop_0 = 0;
    if (warp == 3) {
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_98 = (((l1_s_addr) >> 4) & 0x7FFF) + (0) * 64;
#else
            int _smem_desc_lo_98 = (((l1_s_addr) >> 4) & 0x3FFF) + (0) * 64;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x80004020;\n\t"
            "mov.b32 id, 67373328;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (64))), "r"(_smem_desc_lo_98), "r"(tmem_acc), "r"(0) : "memory");
            tcgen05_commit(hop_addr);
            mbarrier_wait(hop_addr, _phase_hop_0);
            _phase_hop_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_99 = (((aii_s_addr) >> 4) & 0x7FFF) + (1) * 64;
#else
            int _smem_desc_lo_99 = (((aii_s_addr) >> 4) & 0x3FFF) + (1) * 64;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x80004020;\n\t"
            "mov.b32 id, %4;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (16))), "r"(_smem_desc_lo_99), "r"(tmem_acc + 64), "r"(1), "r"(((uint32_t)(67373328) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
            tcgen05_commit(hop_addr);
            tcgen05_commit(lvl1_done_addr);
            mbarrier_wait(hop_addr, _phase_hop_0);
            _phase_hop_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_100 = (((l2_s_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
            int _smem_desc_lo_100 = (((l2_s_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 67373328;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (80))), "r"(_smem_desc_lo_100), "r"(tmem_acc), "r"(0) : "memory");
            tcgen05_commit(hop_addr);
            mbarrier_wait(hop_addr, _phase_hop_0);
            _phase_hop_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_101 = (((aii_s_addr) >> 4) & 0x7FFF) + (2) * 64;
#else
            int _smem_desc_lo_101 = (((aii_s_addr) >> 4) & 0x3FFF) + (2) * 64;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x80004020;\n\t"
            "mov.b32 id, %4;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (32))), "r"(_smem_desc_lo_101), "r"(tmem_acc + 80), "r"(1), "r"(((uint32_t)(67373328) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
            tcgen05_commit(hop_addr);
            tcgen05_commit(lvl2_done_addr);
            mbarrier_wait(hop_addr, _phase_hop_0);
            _phase_hop_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_102 = (((l3_s_addr) >> 4) & 0x7FFF) + (0) * 192;
#else
            int _smem_desc_lo_102 = (((l3_s_addr) >> 4) & 0x3FFF) + (0) * 192;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x80004020;\n\t"
            "mov.b32 id, 67373328;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 62;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 62;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (96))), "r"(_smem_desc_lo_102), "r"(tmem_acc), "r"(0) : "memory");
            tcgen05_commit(hop_addr);
            mbarrier_wait(hop_addr, _phase_hop_0);
            _phase_hop_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
            int _smem_desc_lo_103 = (((aii_s_addr) >> 4) & 0x7FFF) + (3) * 64;
#else
            int _smem_desc_lo_103 = (((aii_s_addr) >> 4) & 0x3FFF) + (3) * 64;
#endif
            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x80004020;\n\t"
            "mov.b32 id, %4;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"((tmem_acc + (48))), "r"(_smem_desc_lo_103), "r"(tmem_acc + 96), "r"(1), "r"(((uint32_t)(67373328) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
            tcgen05_commit(chain_done_addr);
        }
    }
    int tr_base = gk0 + w * 1152;
    unsigned int _phase_lvl1_done_0 = 0;
    if (h4 == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (q4 <= 0) {
            float _tmem_load_2[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                : "r"(taddr + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int sl_0_2 = lane_1 % 16;
            int th_1_2 = lane_1 / 16 * 8;
            #pragma unroll
            for (int j_17 = 0; j_17 < 8; j_17++) {
                {
                    uint32_t _addr_104 = static_cast<uint32_t>(tr_base + ((th_1_2 + j_17) * 18 + sl_0_2) * 4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(_tmem_load_2[j_17]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tl_2 = lane_1 / 2;
            int hh_2_2 = lane_1 % 2 * 8;
            int raddr_2 = tr_base + (tl_2 * 18 + hh_2_2) * 4;
            float vals_2[8];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[(0) + 1]))
                : "r"(raddr_2));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[(2) + 1]))
                : "r"(raddr_2 + 8));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[(4) + 1]))
                : "r"(raddr_2 + 16));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[(6) + 1]))
                : "r"(raddr_2 + 24));
            int t_d_6 = tl_2;
            int col_7 = scol + hh_2_2;
            if (write_ext != 0) {
                if (t_d_6 < clen) {
                    unsigned int _packed_30[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 0], vals_2[_lp*2+1 + 0]));
                        _packed_30[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 2], vals_2[_lp*2+1 + 2]));
                        _packed_30[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 4], vals_2[_lp*2+1 + 4]));
                        _packed_30[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 6], vals_2[_lp*2+1 + 6]));
                        _packed_30[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_6 * ars) + (long long)col_7))[0] = reinterpret_cast<int4*>(_packed_30)[0];
                }
            }
            if (write_int != 0) {
                if (t_d_6 >= clen) {
                    #pragma unroll
                    for (int e_28 = 0; e_28 < 8; e_28++) {
                        vals_2[e_28] = 0.0f;
                    }
                }
                unsigned int _packed_31[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 0], vals_2[_lp*2+1 + 0]));
                    _packed_31[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 2], vals_2[_lp*2+1 + 2]));
                    _packed_31[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 4], vals_2[_lp*2+1 + 4]));
                    _packed_31[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 6], vals_2[_lp*2+1 + 6]));
                    _packed_31[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_6 * ars) + (long long)col_7))[0] = reinterpret_cast<int4*>(_packed_31)[0];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        mbarrier_wait_hint(lvl1_done_addr, _phase_lvl1_done_0, 1000000);
        _phase_lvl1_done_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (q4 <= 1) {
            float _tmem_load_3[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                : "r"(taddr + (unsigned int)row_base + 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int sl_0_3 = lane_1 % 16;
            int th_1_3 = lane_1 / 16 * 8;
            #pragma unroll
            for (int j_18 = 0; j_18 < 8; j_18++) {
                {
                    uint32_t _addr_105 = static_cast<uint32_t>(tr_base + ((th_1_3 + j_18) * 18 + sl_0_3) * 4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(_tmem_load_3[j_18]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tl_3 = lane_1 / 2;
            int hh_2_3 = lane_1 % 2 * 8;
            int raddr_3 = tr_base + (tl_3 * 18 + hh_2_3) * 4;
            float vals_3[8];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[(0) + 1]))
                : "r"(raddr_3));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[(2) + 1]))
                : "r"(raddr_3 + 8));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[(4) + 1]))
                : "r"(raddr_3 + 16));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[(6) + 1]))
                : "r"(raddr_3 + 24));
            int t_d_7 = 16 + tl_3;
            int col_8 = scol + hh_2_3;
            if (write_ext != 0) {
                if (t_d_7 < clen) {
                    unsigned int _packed_32[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 0], vals_3[_lp*2+1 + 0]));
                        _packed_32[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 2], vals_3[_lp*2+1 + 2]));
                        _packed_32[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 4], vals_3[_lp*2+1 + 4]));
                        _packed_32[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 6], vals_3[_lp*2+1 + 6]));
                        _packed_32[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_7 * ars) + (long long)col_8))[0] = reinterpret_cast<int4*>(_packed_32)[0];
                }
            }
            if (write_int != 0) {
                if (t_d_7 >= clen) {
                    #pragma unroll
                    for (int e_29 = 0; e_29 < 8; e_29++) {
                        vals_3[e_29] = 0.0f;
                    }
                }
                unsigned int _packed_33[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 0], vals_3[_lp*2+1 + 0]));
                    _packed_33[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 2], vals_3[_lp*2+1 + 2]));
                    _packed_33[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 4], vals_3[_lp*2+1 + 4]));
                    _packed_33[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 6], vals_3[_lp*2+1 + 6]));
                    _packed_33[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_7 * ars) + (long long)col_8))[0] = reinterpret_cast<int4*>(_packed_33)[0];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
    }
    unsigned int _phase_lvl2_done_0 = 0;
    unsigned int _phase_chain_done_0 = 0;
    if (h4 != 0) {
        mbarrier_wait_hint(lvl2_done_addr, _phase_lvl2_done_0, 1000000);
        _phase_lvl2_done_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (q4 <= 2) {
            float _tmem_load_4[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                : "r"(taddr + (unsigned int)row_base + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int sl_0_4 = lane_1 % 16;
            int th_1_4 = lane_1 / 16 * 8;
            #pragma unroll
            for (int j_19 = 0; j_19 < 8; j_19++) {
                {
                    uint32_t _addr_106 = static_cast<uint32_t>(tr_base + ((th_1_4 + j_19) * 18 + sl_0_4) * 4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(_tmem_load_4[j_19]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tl_4 = lane_1 / 2;
            int hh_2_4 = lane_1 % 2 * 8;
            int raddr_4 = tr_base + (tl_4 * 18 + hh_2_4) * 4;
            float vals_4[8];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_4[(0) + 1]))
                : "r"(raddr_4));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_4[(2) + 1]))
                : "r"(raddr_4 + 8));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_4[(4) + 1]))
                : "r"(raddr_4 + 16));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_4[(6) + 1]))
                : "r"(raddr_4 + 24));
            int t_d_8 = 32 + tl_4;
            int col_9 = scol + hh_2_4;
            if (write_ext != 0) {
                if (t_d_8 < clen) {
                    unsigned int _packed_34[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 0], vals_4[_lp*2+1 + 0]));
                        _packed_34[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 2], vals_4[_lp*2+1 + 2]));
                        _packed_34[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 4], vals_4[_lp*2+1 + 4]));
                        _packed_34[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 6], vals_4[_lp*2+1 + 6]));
                        _packed_34[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_8 * ars) + (long long)col_9))[0] = reinterpret_cast<int4*>(_packed_34)[0];
                }
            }
            if (write_int != 0) {
                if (t_d_8 >= clen) {
                    #pragma unroll
                    for (int e_30 = 0; e_30 < 8; e_30++) {
                        vals_4[e_30] = 0.0f;
                    }
                }
                unsigned int _packed_35[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 0], vals_4[_lp*2+1 + 0]));
                    _packed_35[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 2], vals_4[_lp*2+1 + 2]));
                    _packed_35[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 4], vals_4[_lp*2+1 + 4]));
                    _packed_35[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 6], vals_4[_lp*2+1 + 6]));
                    _packed_35[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_8 * ars) + (long long)col_9))[0] = reinterpret_cast<int4*>(_packed_35)[0];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        mbarrier_wait_hint(chain_done_addr, _phase_chain_done_0, 1000000);
        _phase_chain_done_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (q4 <= 3) {
            float _tmem_load_5[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
                : "r"(taddr + (unsigned int)row_base + 48));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int sl_0_5 = lane_1 % 16;
            int th_1_5 = lane_1 / 16 * 8;
            #pragma unroll
            for (int j_20 = 0; j_20 < 8; j_20++) {
                {
                    uint32_t _addr_107 = static_cast<uint32_t>(tr_base + ((th_1_5 + j_20) * 18 + sl_0_5) * 4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(_tmem_load_5[j_20]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tl_5 = lane_1 / 2;
            int hh_2_5 = lane_1 % 2 * 8;
            int raddr_5 = tr_base + (tl_5 * 18 + hh_2_5) * 4;
            float vals_5[8];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_5[(0) + 1]))
                : "r"(raddr_5));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_5[(2) + 1]))
                : "r"(raddr_5 + 8));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_5[(4) + 1]))
                : "r"(raddr_5 + 16));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vals_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_5[(6) + 1]))
                : "r"(raddr_5 + 24));
            int t_d_9 = 48 + tl_5;
            int col_10 = scol + hh_2_5;
            if (write_ext != 0) {
                if (t_d_9 < clen) {
                    unsigned int _packed_36[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 0], vals_5[_lp*2+1 + 0]));
                        _packed_36[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 2], vals_5[_lp*2+1 + 2]));
                        _packed_36[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 4], vals_5[_lp*2+1 + 4]));
                        _packed_36[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 6], vals_5[_lp*2+1 + 6]));
                        _packed_36[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(akk_ext + (abase + (long long)(t_d_9 * ars) + (long long)col_10))[0] = reinterpret_cast<int4*>(_packed_36)[0];
                }
            }
            if (write_int != 0) {
                if (t_d_9 >= clen) {
                    #pragma unroll
                    for (int e_31 = 0; e_31 < 8; e_31++) {
                        vals_5[e_31] = 0.0f;
                    }
                }
                unsigned int _packed_37[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 0], vals_5[_lp*2+1 + 0]));
                    _packed_37[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 2], vals_5[_lp*2+1 + 2]));
                    _packed_37[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 4], vals_5[_lp*2+1 + 4]));
                    _packed_37[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_5[_lp*2 + 6], vals_5[_lp*2+1 + 6]));
                    _packed_37[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(akk_int + (ibase + (long long)(t_d_9 * ars) + (long long)col_10))[0] = reinterpret_cast<int4*>(_packed_37)[0];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (w == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
}

} // extern "C"
