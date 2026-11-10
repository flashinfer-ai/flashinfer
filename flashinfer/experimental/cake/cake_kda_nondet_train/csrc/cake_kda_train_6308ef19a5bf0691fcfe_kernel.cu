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

#define LAUNCH_MIN_BLOCKS 2

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


extern "C" __device__ float __nv_expf(float);


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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_6308ef19a5bf0691fcfe(const __grid_constant__ CUtensorMap dAqk, const __grid_constant__ CUtensorMap dAkk, const __grid_constant__ CUtensorMap gk, const __grid_constant__ CUtensorMap k_e, const __grid_constant__ CUtensorMap q_e, const __grid_constant__ CUtensorMap dq_f_t, const __grid_constant__ CUtensorMap dk_f_t, const __grid_constant__ CUtensorMap dg_f_t, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ g_raw, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_raw, float* __restrict__ qk_rstd, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, __nv_bfloat16* __restrict__ dg_raw, __nv_bfloat16* __restrict__ dbeta_ext, float* __restrict__ dA_part, float* __restrict__ db_part, int num_heads, int num_qk_heads, int group, float lower_bound, long long g_raw_row_stride, long long g_raw_head_stride, int num_tokens, int q_raw_row_stride, int q_raw_head_stride, int k_raw_row_stride, int k_raw_head_stride)
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

    const int mbar_base = smem + 114944;
    #define ops_full_addr (mbar_base + 0)
    #define blk_full_addr (mbar_base + 32)
    #define blk_empty_addr (mbar_base + 48)
    #define p1blk_empty_addr (mbar_base + 64)
    #define ring_full_addr (mbar_base + 72)
    #define ring_empty_addr (mbar_base + 88)
    #define p2_done_addr (mbar_base + 104)
    #define fold_done_addr (mbar_base + 112)
    #define p1a_ready_addr (mbar_base + 120)
    #define p1a_t2_drained_addr (mbar_base + 128)
    #define p1b_ready_addr (mbar_base + 136)
    #define s16_done_addr (mbar_base + 144)
    #define epi_full_addr (mbar_base + 152)
    #define epi_empty_addr (mbar_base + 184)
    #define kq_ready_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* gk_p = reinterpret_cast<float*>(smem_raw + 0);
    const int gk_p_addr = smem + 0;
    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int k_p_addr = smem + 32768;
    float* ring = reinterpret_cast<float*>(smem_raw + 49152);
    const int ring_addr = smem + 49152;
    float* dab = reinterpret_cast<float*>(smem_raw + 81920);
    const int dab_addr = smem + 81920;
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int q_p_addr = smem + 98304;
    float* tiles = reinterpret_cast<float*>(smem_raw + 106496);
    const int tiles_addr = smem + 106496;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 114688);
    const int beta_s_addr = smem + 114688;
    float* pool_f32 = reinterpret_cast<float*>(smem_raw + 0);
    const int pool_f32_addr = smem + 0;

    // Kernel setup ops
    int clen_setup = chunk_len[blockIdx.x];
    int kq0_setup = chunk_bos[blockIdx.x];
    if (__syncthreads_and(clen_setup == 0)) return;

    // Mbarrier init (31 barriers at smem_raw[114944..115192))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'four' ---
            // ops_full: 4 barriers, init_count=1
            mbarrier_init(smem + 114944, 1);
            mbarrier_init(smem + 114952, 1);
            mbarrier_init(smem + 114960, 1);
            mbarrier_init(smem + 114968, 1);
            // --- pipeline 'two' ---
            // blk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 114976, 1);
            mbarrier_init(smem + 114984, 1);
            // blk_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 114992, 256);
            mbarrier_init(smem + 115000, 256);
            // --- pipeline 'one' ---
            // p1blk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 115008, 1);
            // --- pipeline 'two' ---
            // ring_full: 2 barriers, init_count=256
            mbarrier_init(smem + 115016, 256);
            mbarrier_init(smem + 115024, 256);
            // ring_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 115032, 1);
            mbarrier_init(smem + 115040, 1);
            // --- pipeline 'one' ---
            // p2_done: 1 barriers, init_count=1
            mbarrier_init(smem + 115048, 1);
            // fold_done: 1 barriers, init_count=256
            mbarrier_init(smem + 115056, 256);
            // p1a_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 115064, 1);
            // p1a_t2_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 115072, 256);
            // p1b_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 115080, 1);
            // s16_done: 1 barriers, init_count=1
            mbarrier_init(smem + 115088, 1);
            // --- pipeline 'four' ---
            // epi_full: 4 barriers, init_count=1
            mbarrier_init(smem + 115096, 1);
            mbarrier_init(smem + 115104, 1);
            mbarrier_init(smem + 115112, 1);
            mbarrier_init(smem + 115120, 1);
            // epi_empty: 4 barriers, init_count=128
            mbarrier_init(smem + 115128, 128);
            mbarrier_init(smem + 115136, 128);
            mbarrier_init(smem + 115144, 128);
            mbarrier_init(smem + 115152, 128);
            // kq_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 115160, 1);
            mbarrier_init(smem + 115168, 1);
            mbarrier_init(smem + 115176, 1);
            mbarrier_init(smem + 115184, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 115192);
    if (warp == 0) {
        int _tmem_hold = smem + 115192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // ---- Role: compute ----
    if (warp <= 7) {
        { // compute_main
            int chunk = blockIdx.x;
            int hv = blockIdx.y;
            int qk_head = hv / group;
            int row0 = chunk * 64;
            int tid_0 = tid;
            int lane_1 = lane;
            int w = warp;
            int qd = w % 4;
            int th = w / 4;
            int d_lane = qd * 32 + lane_1;
            int dh_lane = qd / 2;
            int dcol = d_lane % 64;
            int row_base = qd * 32 << 16;
            int rs_g = num_heads * 128;
            int qrs = num_qk_heads * 128;
            long long ext0 = (long long)chunk_bos[chunk];
            int clen = chunk_len[chunk];
            long long gbase_t = (ext0 * (long long)num_heads + (long long)hv) * 128;
            __nv_bfloat16* draw_base = dg_raw + (gbase_t + (long long)d_lane);
            long long qoff0 = ext0 * (long long)qrs + (long long)(qk_head * 128 + d_lane);
            __nv_bfloat16* dq_base = dq_ext + qoff0;
            __nv_bfloat16* dk_base = dk_ext + qoff0;
            int l2n_tok = tid_0 >> 1 & 63;
            long long l2n_pf_row = ext0 + (long long)l2n_tok;
            int l2n_pf_col = 64 * (tid_0 & 1);
            bool l2n_pf_live = l2n_tok < clen;
            if (tid_0 < 128 && l2n_pf_live) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(q_raw + (l2n_pf_row * (long long)q_raw_row_stride + (long long)(qk_head * q_raw_head_stride + l2n_pf_col))))); }
            if (tid_0 >= 128 && l2n_pf_live) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(k_raw + (l2n_pf_row * (long long)k_raw_row_stride + (long long)(qk_head * k_raw_head_stride + l2n_pf_col))))); }
            if ((tid_0 & 1) == 0 && l2n_pf_live) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(qk_rstd + (((long long)(tid_0 >> 7) * (long long)num_tokens + l2n_pf_row) * (long long)num_qk_heads + (long long)qk_head)))); }
            float off_acc[1];
            off_acc[0] = 0.0f;
            if (tid_0 < 64) {
                beta_s[tid_0] = beta[(row0 + tid_0) * num_heads + hv];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(ops_full_addr, 0);
            mbarrier_wait(blk_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (clen < 16) {
                mbarrier_wait(kq_ready_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int sbase = 8 * th;
            int gbase = pool_f32_addr + (unsigned int)(qd * 8192 + sbase * 128);
            int kbase = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase * 128);
            int qbase = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float g_s[8];
            float q_s[8];
            float kb_s[8];
            float bv[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv[4])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4 + 16)));
            float r[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r[0])) : "r"((gbase + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[0] = r[0];
            float r_0[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_0[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[1] = r_0[0];
            unsigned int r_1[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1[0])) : "r"((qbase + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[0] = __uint_as_float(r_1[0] << 16);
            unsigned int r_2[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2[0])) : "r"((qbase + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[1] = __uint_as_float(r_2[0] << 16);
            unsigned int r_3[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_3[0])) : "r"((kbase + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_0 = __uint_as_float(r_3[0] << 16);
            unsigned int r_4[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_4[0])) : "r"((kbase + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_0 = __uint_as_float(r_4[0] << 16);
            float2 _f32x2_mul_a_0 = make_float2(k_s0_1_0, k_s1_1_0);
            float2 _f32x2_mul_b_1 = make_float2(bv[0], bv[1]);
            float2 kp_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_1));
            __nv_bfloat16 kb0_1_0 = (__nv_bfloat16)kp_1_0.x;
            __nv_bfloat16 kb1_1_0 = (__nv_bfloat16)kp_1_0.y;
            kb_s[0] = (float)kb0_1_0;
            kb_s[1] = (float)kb1_1_0;
            float r_5[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_5[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[2] = r_5[0];
            float r_6[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_6[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[3] = r_6[0];
            unsigned int r_7[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_7[0])) : "r"((qbase + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[2] = __uint_as_float(r_7[0] << 16);
            unsigned int r_8[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_8[0])) : "r"((qbase + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[3] = __uint_as_float(r_8[0] << 16);
            unsigned int r_9[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_9[0])) : "r"((kbase + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_1 = __uint_as_float(r_9[0] << 16);
            unsigned int r_10[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_10[0])) : "r"((kbase + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_1 = __uint_as_float(r_10[0] << 16);
            float2 _f32x2_mul_a_2 = make_float2(k_s0_1_1, k_s1_1_1);
            float2 _f32x2_mul_b_3 = make_float2(bv[2], bv[3]);
            float2 kp_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
            __nv_bfloat16 kb0_1_1 = (__nv_bfloat16)kp_1_1.x;
            __nv_bfloat16 kb1_1_1 = (__nv_bfloat16)kp_1_1.y;
            kb_s[2] = (float)kb0_1_1;
            kb_s[3] = (float)kb1_1_1;
            float r_11[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_11[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[4] = r_11[0];
            float r_12[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_12[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[5] = r_12[0];
            unsigned int r_13[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_13[0])) : "r"((qbase + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[4] = __uint_as_float(r_13[0] << 16);
            unsigned int r_14[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_14[0])) : "r"((qbase + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[5] = __uint_as_float(r_14[0] << 16);
            unsigned int r_15[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_15[0])) : "r"((kbase + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_2 = __uint_as_float(r_15[0] << 16);
            unsigned int r_16[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_16[0])) : "r"((kbase + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_2 = __uint_as_float(r_16[0] << 16);
            float2 _f32x2_mul_a_4 = make_float2(k_s0_1_2, k_s1_1_2);
            float2 _f32x2_mul_b_5 = make_float2(bv[4], bv[5]);
            float2 kp_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            __nv_bfloat16 kb0_1_2 = (__nv_bfloat16)kp_1_2.x;
            __nv_bfloat16 kb1_1_2 = (__nv_bfloat16)kp_1_2.y;
            kb_s[4] = (float)kb0_1_2;
            kb_s[5] = (float)kb1_1_2;
            float r_17[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_17[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[6] = r_17[0];
            float r_18[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_18[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[7] = r_18[0];
            unsigned int r_19[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_19[0])) : "r"((qbase + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[6] = __uint_as_float(r_19[0] << 16);
            unsigned int r_20[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_20[0])) : "r"((qbase + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[7] = __uint_as_float(r_20[0] << 16);
            unsigned int r_21[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_21[0])) : "r"((kbase + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_3 = __uint_as_float(r_21[0] << 16);
            unsigned int r_22[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_22[0])) : "r"((kbase + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_3 = __uint_as_float(r_22[0] << 16);
            float2 _f32x2_mul_a_6 = make_float2(k_s0_1_3, k_s1_1_3);
            float2 _f32x2_mul_b_7 = make_float2(bv[6], bv[7]);
            float2 kp_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
            __nv_bfloat16 kb0_1_3 = (__nv_bfloat16)kp_1_3.x;
            __nv_bfloat16 kb1_1_3 = (__nv_bfloat16)kp_1_3.y;
            kb_s[6] = (float)kb0_1_3;
            kb_s[7] = (float)kb1_1_3;
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv[4];
                int kk = w * 4;
                int a = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[0])) : "r"(a));
                tv[0] = ((lane_1 > (kk & 15)) ? 0.0f : tv[0]);
                {
                    uint32_t _addr_8 = static_cast<uint32_t>(a);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(tv[0]) : "memory");
                }
                int kk_r_0 = w * 4 + 1;
                int a_r_0 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_r_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_r_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[1])) : "r"(a_r_0));
                tv[1] = ((lane_1 > (kk_r_0 & 15)) ? 0.0f : tv[1]);
                {
                    uint32_t _addr_9 = static_cast<uint32_t>(a_r_0);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(tv[1]) : "memory");
                }
                int kk_r_1 = w * 4 + 2;
                int a_r_1 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_r_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_r_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[2])) : "r"(a_r_1));
                tv[2] = ((lane_1 > (kk_r_1 & 15)) ? 0.0f : tv[2]);
                {
                    uint32_t _addr_10 = static_cast<uint32_t>(a_r_1);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(tv[2]) : "memory");
                }
                int kk_r_2 = w * 4 + 3;
                int a_r_2 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_r_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_r_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[3])) : "r"(a_r_2));
                tv[3] = ((lane_1 > (kk_r_2 & 15)) ? 0.0f : tv[3]);
                {
                    uint32_t _addr_11 = static_cast<uint32_t>(a_r_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(tv[3]) : "memory");
                }
                int ta = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta), "f"(tv[0]), "f"(tv[1]), "f"(tv[2]), "f"(tv[3]) : "memory");
            }
            float r_23[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_23[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d = r_23[0] - 21.640425f;
            float2 _f2_0 = make_float2(a_d, a_d);
            float vq[8];
            float vk[8];
            float2 d2_1_0 = sub_f32x2(make_float2(g_s[0], g_s[1]), _f2_0);
            float ev0_1_0 = approx_exp2(d2_1_0.x);
            float ev1_1_0 = approx_exp2(d2_1_0.y);
            float2 _f2_1 = make_float2(ev0_1_0, ev1_1_0);
            float2 _f32x2_mul_a_12 = make_float2(q_s[0], q_s[1]);
            float2 vq2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f2_1));
            float2 _f32x2_mul_a_13 = make_float2(kb_s[0], kb_s[1]);
            float2 vk2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_f2_1));
            vq[0] = vq2_1_0.x;
            vq[1] = vq2_1_0.y;
            vk[0] = vk2_1_0.x;
            vk[1] = vk2_1_0.y;
            float2 d2_1_1 = sub_f32x2(make_float2(g_s[2], g_s[3]), _f2_0);
            float ev0_1_1 = approx_exp2(d2_1_1.x);
            float ev1_1_1 = approx_exp2(d2_1_1.y);
            float2 _f2_2 = make_float2(ev0_1_1, ev1_1_1);
            float2 _f32x2_mul_a_14 = make_float2(q_s[2], q_s[3]);
            float2 vq2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f2_2));
            float2 _f32x2_mul_a_15 = make_float2(kb_s[2], kb_s[3]);
            float2 vk2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f2_2));
            vq[2] = vq2_1_1.x;
            vq[3] = vq2_1_1.y;
            vk[2] = vk2_1_1.x;
            vk[3] = vk2_1_1.y;
            float2 d2_1_2 = sub_f32x2(make_float2(g_s[4], g_s[5]), _f2_0);
            float ev0_1_2 = approx_exp2(d2_1_2.x);
            float ev1_1_2 = approx_exp2(d2_1_2.y);
            float2 _f2_3 = make_float2(ev0_1_2, ev1_1_2);
            float2 _f32x2_mul_a_16 = make_float2(q_s[4], q_s[5]);
            float2 vq2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f2_3));
            float2 _f32x2_mul_a_17 = make_float2(kb_s[4], kb_s[5]);
            float2 vk2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_f2_3));
            vq[4] = vq2_1_2.x;
            vq[5] = vq2_1_2.y;
            vk[4] = vk2_1_2.x;
            vk[5] = vk2_1_2.y;
            float2 d2_1_3 = sub_f32x2(make_float2(g_s[6], g_s[7]), _f2_0);
            float ev0_1_3 = approx_exp2(d2_1_3.x);
            float ev1_1_3 = approx_exp2(d2_1_3.y);
            float2 _f2_4 = make_float2(ev0_1_3, ev1_1_3);
            float2 _f32x2_mul_a_18 = make_float2(q_s[6], q_s[7]);
            float2 vq2_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f2_4));
            float2 _f32x2_mul_a_19 = make_float2(kb_s[6], kb_s[7]);
            float2 vk2_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f2_4));
            vq[6] = vq2_1_3.x;
            vq[7] = vq2_1_3.y;
            vk[6] = vk2_1_3.x;
            vk[7] = vk2_1_3.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq[0]), "f"(vq[1]), "f"(vq[2]), "f"(vq[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk[0]), "f"(vk[1]), "f"(vk[2]), "f"(vk[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq[4]), "f"(vq[5]), "f"(vq[6]), "f"(vq[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk[4]), "f"(vk[5]), "f"(vk[6]), "f"(vk[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(blk_empty_addr);
            mbarrier_wait(ops_full_addr + 8, 0);
            mbarrier_wait(blk_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (clen < 32) {
                mbarrier_wait(kq_ready_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int sbase_56 = 16 + 8 * th;
            int gbase_57 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_56 * 128);
            int kbase_58 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_56 * 128);
            int qbase_59 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float g_s_60[8];
            float q_s_61[8];
            float kb_s_62[8];
            float bv_63[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_56 * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_63[4])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_63[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_56 * 4 + 16)));
            float r_24[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_24[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[0] = r_24[0];
            float r_25[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_25[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[1] = r_25[0];
            unsigned int r_26[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_26[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[0] = __uint_as_float(r_26[0] << 16);
            unsigned int r_27[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_27[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[1] = __uint_as_float(r_27[0] << 16);
            unsigned int r_28[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_28[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_0 = __uint_as_float(r_28[0] << 16);
            unsigned int r_29[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_29[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_0 = __uint_as_float(r_29[0] << 16);
            float2 _f32x2_mul_a_20 = make_float2(k_s0_0, k_s1_0);
            float2 _f32x2_mul_b_21 = make_float2(bv_63[0], bv_63[1]);
            float2 kp_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            __nv_bfloat16 kb0_0 = (__nv_bfloat16)kp_0.x;
            __nv_bfloat16 kb1_0 = (__nv_bfloat16)kp_0.y;
            kb_s_62[0] = (float)kb0_0;
            kb_s_62[1] = (float)kb1_0;
            float r_30[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_30[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[2] = r_30[0];
            float r_31[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_31[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[3] = r_31[0];
            unsigned int r_32[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_32[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[2] = __uint_as_float(r_32[0] << 16);
            unsigned int r_33[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_33[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[3] = __uint_as_float(r_33[0] << 16);
            unsigned int r_34[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_34[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1 = __uint_as_float(r_34[0] << 16);
            unsigned int r_35[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_35[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1 = __uint_as_float(r_35[0] << 16);
            float2 _f32x2_mul_a_22 = make_float2(k_s0_1, k_s1_1);
            float2 _f32x2_mul_b_23 = make_float2(bv_63[2], bv_63[3]);
            float2 kp_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            __nv_bfloat16 kb0_1 = (__nv_bfloat16)kp_1.x;
            __nv_bfloat16 kb1_1 = (__nv_bfloat16)kp_1.y;
            kb_s_62[2] = (float)kb0_1;
            kb_s_62[3] = (float)kb1_1;
            float r_36[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_36[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[4] = r_36[0];
            float r_37[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_37[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[5] = r_37[0];
            unsigned int r_38[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_38[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[4] = __uint_as_float(r_38[0] << 16);
            unsigned int r_39[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_39[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[5] = __uint_as_float(r_39[0] << 16);
            unsigned int r_40[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_40[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_2 = __uint_as_float(r_40[0] << 16);
            unsigned int r_41[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_41[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_2 = __uint_as_float(r_41[0] << 16);
            float2 _f32x2_mul_a_24 = make_float2(k_s0_2, k_s1_2);
            float2 _f32x2_mul_b_25 = make_float2(bv_63[4], bv_63[5]);
            float2 kp_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
            __nv_bfloat16 kb0_2 = (__nv_bfloat16)kp_2.x;
            __nv_bfloat16 kb1_2 = (__nv_bfloat16)kp_2.y;
            kb_s_62[4] = (float)kb0_2;
            kb_s_62[5] = (float)kb1_2;
            float r_42[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_42[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[6] = r_42[0];
            float r_43[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_43[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[7] = r_43[0];
            unsigned int r_44[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_44[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[6] = __uint_as_float(r_44[0] << 16);
            unsigned int r_45[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_45[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[7] = __uint_as_float(r_45[0] << 16);
            unsigned int r_46[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_46[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_3 = __uint_as_float(r_46[0] << 16);
            unsigned int r_47[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_47[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_3 = __uint_as_float(r_47[0] << 16);
            float2 _f32x2_mul_a_26 = make_float2(k_s0_3, k_s1_3);
            float2 _f32x2_mul_b_27 = make_float2(bv_63[6], bv_63[7]);
            float2 kp_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            __nv_bfloat16 kb0_3 = (__nv_bfloat16)kp_3.x;
            __nv_bfloat16 kb1_3 = (__nv_bfloat16)kp_3.y;
            kb_s_62[6] = (float)kb0_3;
            kb_s_62[7] = (float)kb1_3;
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_1[4];
                int kk_1 = w * 4;
                int a_1 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[0])) : "r"(a_1));
                int kk_0 = w * 4 + 1;
                int a_1_1 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[1])) : "r"(a_1_1));
                int kk_2 = w * 4 + 2;
                int a_3 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[2])) : "r"(a_3));
                int kk_4 = w * 4 + 3;
                int a_5 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[3])) : "r"(a_5));
                int ta_1 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_1), "f"(tv_1[0]), "f"(tv_1[1]), "f"(tv_1[2]), "f"(tv_1[3]) : "memory");
            }
            float r_48[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_48[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_109 = r_48[0];
            float2 _f2_5 = make_float2(a_d_109, a_d_109);
            float vq_110[8];
            float vk_111[8];
            float2 d2_0 = sub_f32x2(make_float2(g_s_60[0], g_s_60[1]), _f2_5);
            float ev0_0 = approx_exp2(d2_0.x);
            float ev1_0 = approx_exp2(d2_0.y);
            float2 _f2_6 = make_float2(ev0_0, ev1_0);
            float2 _f32x2_mul_a_28 = make_float2(q_s_61[0], q_s_61[1]);
            float2 vq2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f2_6));
            float2 _f32x2_mul_a_29 = make_float2(kb_s_62[0], kb_s_62[1]);
            float2 vk2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_f2_6));
            vq_110[0] = vq2_0.x;
            vq_110[1] = vq2_0.y;
            vk_111[0] = vk2_0.x;
            vk_111[1] = vk2_0.y;
            float2 d2_1 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_5);
            float ev0_1 = approx_exp2(d2_1.x);
            float ev1_1 = approx_exp2(d2_1.y);
            float2 _f2_7 = make_float2(ev0_1, ev1_1);
            float2 _f32x2_mul_a_30 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f2_7));
            float2 _f32x2_mul_a_31 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f2_7));
            vq_110[2] = vq2_1.x;
            vq_110[3] = vq2_1.y;
            vk_111[2] = vk2_1.x;
            vk_111[3] = vk2_1.y;
            float2 d2_2 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_5);
            float ev0_2 = approx_exp2(d2_2.x);
            float ev1_2 = approx_exp2(d2_2.y);
            float2 _f2_8 = make_float2(ev0_2, ev1_2);
            float2 _f32x2_mul_a_32 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f2_8));
            float2 _f32x2_mul_a_33 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f2_8));
            vq_110[4] = vq2_2.x;
            vq_110[5] = vq2_2.y;
            vk_111[4] = vk2_2.x;
            vk_111[5] = vk2_2.y;
            float2 d2_3 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_5);
            float ev0_3 = approx_exp2(d2_3.x);
            float ev1_3 = approx_exp2(d2_3.y);
            float2 _f2_9 = make_float2(ev0_3, ev1_3);
            float2 _f32x2_mul_a_34 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f2_9));
            float2 _f32x2_mul_a_35 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f2_9));
            vq_110[6] = vq2_3.x;
            vq_110[7] = vq2_3.y;
            vk_111[6] = vk2_3.x;
            vk_111[7] = vk2_3.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_110[0]), "f"(vq_110[1]), "f"(vq_110[2]), "f"(vq_110[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_111[0]), "f"(vk_111[1]), "f"(vk_111[2]), "f"(vk_111[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_110[4]), "f"(vq_110[5]), "f"(vq_110[6]), "f"(vq_110[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_111[4]), "f"(vk_111[5]), "f"(vk_111[6]), "f"(vk_111[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_2[4];
                int kk_3 = w * 4;
                int a_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[0])) : "r"(a_2));
                tv_2[0] = ((lane_1 > (kk_3 & 15)) ? 0.0f : tv_2[0]);
                {
                    uint32_t _addr_36 = static_cast<uint32_t>(a_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(tv_2[0]) : "memory");
                }
                int kk_r_3 = w * 4 + 1;
                int a_r_3 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_r_3 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_r_3 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[1])) : "r"(a_r_3));
                tv_2[1] = ((lane_1 > (kk_r_3 & 15)) ? 0.0f : tv_2[1]);
                {
                    uint32_t _addr_37 = static_cast<uint32_t>(a_r_3);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(tv_2[1]) : "memory");
                }
                int kk_r_4 = w * 4 + 2;
                int a_r_4 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_r_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_r_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[2])) : "r"(a_r_4));
                tv_2[2] = ((lane_1 > (kk_r_4 & 15)) ? 0.0f : tv_2[2]);
                {
                    uint32_t _addr_38 = static_cast<uint32_t>(a_r_4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(tv_2[2]) : "memory");
                }
                int kk_r_5 = w * 4 + 3;
                int a_r_5 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_r_5 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_r_5 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[3])) : "r"(a_r_5));
                tv_2[3] = ((lane_1 > (kk_r_5 & 15)) ? 0.0f : tv_2[3]);
                {
                    uint32_t _addr_39 = static_cast<uint32_t>(a_r_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(tv_2[3]) : "memory");
                }
                int ta_2 = (tiles_addr + 4096 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_2), "f"(tv_2[0]), "f"(tv_2[1]), "f"(tv_2[2]), "f"(tv_2[3]) : "memory");
            }
            float r_49[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_49[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_133 = r_49[0] - 21.640425f;
            float2 _f2_10 = make_float2(a_d_133, a_d_133);
            float vq_134[8];
            float vk_135[8];
            float2 d2_4 = sub_f32x2(make_float2(g_s_60[0], g_s_60[1]), _f2_10);
            float ev0_4 = approx_exp2(d2_4.x);
            float ev1_4 = approx_exp2(d2_4.y);
            float2 _f2_11 = make_float2(ev0_4, ev1_4);
            float2 _f32x2_mul_a_40 = make_float2(q_s_61[0], q_s_61[1]);
            float2 vq2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f2_11));
            float2 _f32x2_mul_a_41 = make_float2(kb_s_62[0], kb_s_62[1]);
            float2 vk2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_f2_11));
            vq_134[0] = vq2_4.x;
            vq_134[1] = vq2_4.y;
            vk_135[0] = vk2_4.x;
            vk_135[1] = vk2_4.y;
            float2 d2_5 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_10);
            float ev0_5 = approx_exp2(d2_5.x);
            float ev1_5 = approx_exp2(d2_5.y);
            float2 _f2_12 = make_float2(ev0_5, ev1_5);
            float2 _f32x2_mul_a_42 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f2_12));
            float2 _f32x2_mul_a_43 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_f2_12));
            vq_134[2] = vq2_5.x;
            vq_134[3] = vq2_5.y;
            vk_135[2] = vk2_5.x;
            vk_135[3] = vk2_5.y;
            float2 d2_6 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_10);
            float ev0_6 = approx_exp2(d2_6.x);
            float ev1_6 = approx_exp2(d2_6.y);
            float2 _f2_13 = make_float2(ev0_6, ev1_6);
            float2 _f32x2_mul_a_44 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f2_13));
            float2 _f32x2_mul_a_45 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f2_13));
            vq_134[4] = vq2_6.x;
            vq_134[5] = vq2_6.y;
            vk_135[4] = vk2_6.x;
            vk_135[5] = vk2_6.y;
            float2 d2_7 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_10);
            float ev0_7 = approx_exp2(d2_7.x);
            float ev1_7 = approx_exp2(d2_7.y);
            float2 _f2_14 = make_float2(ev0_7, ev1_7);
            float2 _f32x2_mul_a_46 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f2_14));
            float2 _f32x2_mul_a_47 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_f2_14));
            vq_134[6] = vq2_7.x;
            vq_134[7] = vq2_7.y;
            vk_135[6] = vk2_7.x;
            vk_135[7] = vk2_7.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_134[0]), "f"(vq_134[1]), "f"(vq_134[2]), "f"(vq_134[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_135[0]), "f"(vk_135[1]), "f"(vk_135[2]), "f"(vk_135[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_134[4]), "f"(vq_134[5]), "f"(vq_134[6]), "f"(vq_134[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_135[4]), "f"(vk_135[5]), "f"(vk_135[6]), "f"(vk_135[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(blk_empty_addr + 8);
            mbarrier_wait(ops_full_addr + 16, 0);
            mbarrier_wait(blk_full_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (clen < 48) {
                mbarrier_wait(kq_ready_addr + 16, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int sbase_156 = 32 + 8 * th;
            int gbase_157 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_156 * 128);
            int kbase_158 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_156 * 128);
            int qbase_159 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float g_s_160[8];
            float q_s_161[8];
            float kb_s_162[8];
            float bv_163[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_163[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_156 * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_163[4])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_163[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_156 * 4 + 16)));
            float r_50[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_50[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[0] = r_50[0];
            float r_51[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_51[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[1] = r_51[0];
            unsigned int r_52[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_52[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[0] = __uint_as_float(r_52[0] << 16);
            unsigned int r_53[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_53[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[1] = __uint_as_float(r_53[0] << 16);
            unsigned int r_54[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_54[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_4 = __uint_as_float(r_54[0] << 16);
            unsigned int r_55[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_55[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_4 = __uint_as_float(r_55[0] << 16);
            float2 _f32x2_mul_a_48 = make_float2(k_s0_4, k_s1_4);
            float2 _f32x2_mul_b_49 = make_float2(bv_163[0], bv_163[1]);
            float2 kp_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
            __nv_bfloat16 kb0_4 = (__nv_bfloat16)kp_4.x;
            __nv_bfloat16 kb1_4 = (__nv_bfloat16)kp_4.y;
            kb_s_162[0] = (float)kb0_4;
            kb_s_162[1] = (float)kb1_4;
            float r_56[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_56[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[2] = r_56[0];
            float r_57[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_57[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[3] = r_57[0];
            unsigned int r_58[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_58[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[2] = __uint_as_float(r_58[0] << 16);
            unsigned int r_59[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_59[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[3] = __uint_as_float(r_59[0] << 16);
            unsigned int r_60[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_60[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_5 = __uint_as_float(r_60[0] << 16);
            unsigned int r_61[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_61[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_5 = __uint_as_float(r_61[0] << 16);
            float2 _f32x2_mul_a_50 = make_float2(k_s0_5, k_s1_5);
            float2 _f32x2_mul_b_51 = make_float2(bv_163[2], bv_163[3]);
            float2 kp_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            __nv_bfloat16 kb0_5 = (__nv_bfloat16)kp_5.x;
            __nv_bfloat16 kb1_5 = (__nv_bfloat16)kp_5.y;
            kb_s_162[2] = (float)kb0_5;
            kb_s_162[3] = (float)kb1_5;
            float r_62[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_62[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[4] = r_62[0];
            float r_63[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_63[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[5] = r_63[0];
            unsigned int r_64[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_64[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[4] = __uint_as_float(r_64[0] << 16);
            unsigned int r_65[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_65[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[5] = __uint_as_float(r_65[0] << 16);
            unsigned int r_66[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_66[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_6 = __uint_as_float(r_66[0] << 16);
            unsigned int r_67[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_67[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_6 = __uint_as_float(r_67[0] << 16);
            float2 _f32x2_mul_a_52 = make_float2(k_s0_6, k_s1_6);
            float2 _f32x2_mul_b_53 = make_float2(bv_163[4], bv_163[5]);
            float2 kp_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            __nv_bfloat16 kb0_6 = (__nv_bfloat16)kp_6.x;
            __nv_bfloat16 kb1_6 = (__nv_bfloat16)kp_6.y;
            kb_s_162[4] = (float)kb0_6;
            kb_s_162[5] = (float)kb1_6;
            float r_68[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_68[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[6] = r_68[0];
            float r_69[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_69[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[7] = r_69[0];
            unsigned int r_70[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_70[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[6] = __uint_as_float(r_70[0] << 16);
            unsigned int r_71[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_71[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[7] = __uint_as_float(r_71[0] << 16);
            unsigned int r_72[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_72[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_7 = __uint_as_float(r_72[0] << 16);
            unsigned int r_73[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_73[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_7 = __uint_as_float(r_73[0] << 16);
            float2 _f32x2_mul_a_54 = make_float2(k_s0_7, k_s1_7);
            float2 _f32x2_mul_b_55 = make_float2(bv_163[6], bv_163[7]);
            float2 kp_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
            __nv_bfloat16 kb0_7 = (__nv_bfloat16)kp_7.x;
            __nv_bfloat16 kb1_7 = (__nv_bfloat16)kp_7.y;
            kb_s_162[6] = (float)kb0_7;
            kb_s_162[7] = (float)kb1_7;
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_3[4];
                int kk_5 = w * 4;
                int a_4 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[0])) : "r"(a_4));
                int kk_0_1 = w * 4 + 1;
                int a_1_2 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[1])) : "r"(a_1_2));
                int kk_2_1 = w * 4 + 2;
                int a_3_1 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[2])) : "r"(a_3_1));
                int kk_4_1 = w * 4 + 3;
                int a_5_1 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_4_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[3])) : "r"(a_5_1));
                int ta_3 = (tiles_addr + 6144 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_3), "f"(tv_3[0]), "f"(tv_3[1]), "f"(tv_3[2]), "f"(tv_3[3]) : "memory");
            }
            float r_74[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_74[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_209 = r_74[0];
            float2 _f2_15 = make_float2(a_d_209, a_d_209);
            float vq_210[8];
            float vk_211[8];
            float2 d2_8 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_15);
            float ev0_8 = approx_exp2(d2_8.x);
            float ev1_8 = approx_exp2(d2_8.y);
            float2 _f2_16 = make_float2(ev0_8, ev1_8);
            float2 _f32x2_mul_a_56 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f2_16));
            float2 _f32x2_mul_a_57 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f2_16));
            vq_210[0] = vq2_8.x;
            vq_210[1] = vq2_8.y;
            vk_211[0] = vk2_8.x;
            vk_211[1] = vk2_8.y;
            float2 d2_9 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_15);
            float ev0_9 = approx_exp2(d2_9.x);
            float ev1_9 = approx_exp2(d2_9.y);
            float2 _f2_17 = make_float2(ev0_9, ev1_9);
            float2 _f32x2_mul_a_58 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f2_17));
            float2 _f32x2_mul_a_59 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f2_17));
            vq_210[2] = vq2_9.x;
            vq_210[3] = vq2_9.y;
            vk_211[2] = vk2_9.x;
            vk_211[3] = vk2_9.y;
            float2 d2_10 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_15);
            float ev0_10 = approx_exp2(d2_10.x);
            float ev1_10 = approx_exp2(d2_10.y);
            float2 _f2_18 = make_float2(ev0_10, ev1_10);
            float2 _f32x2_mul_a_60 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f2_18));
            float2 _f32x2_mul_a_61 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_f2_18));
            vq_210[4] = vq2_10.x;
            vq_210[5] = vq2_10.y;
            vk_211[4] = vk2_10.x;
            vk_211[5] = vk2_10.y;
            float2 d2_11 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_15);
            float ev0_11 = approx_exp2(d2_11.x);
            float ev1_11 = approx_exp2(d2_11.y);
            float2 _f2_19 = make_float2(ev0_11, ev1_11);
            float2 _f32x2_mul_a_62 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f2_19));
            float2 _f32x2_mul_a_63 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f2_19));
            vq_210[6] = vq2_11.x;
            vq_210[7] = vq2_11.y;
            vk_211[6] = vk2_11.x;
            vk_211[7] = vk2_11.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_210[0]), "f"(vq_210[1]), "f"(vq_210[2]), "f"(vq_210[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_211[0]), "f"(vk_211[1]), "f"(vk_211[2]), "f"(vk_211[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_210[4]), "f"(vq_210[5]), "f"(vq_210[6]), "f"(vq_210[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_211[4]), "f"(vk_211[5]), "f"(vk_211[6]), "f"(vk_211[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_4[4];
                int kk_6 = w * 4;
                int a_6 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_6 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_6 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[0])) : "r"(a_6));
                int kk_0_2 = w * 4 + 1;
                int a_1_3 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[1])) : "r"(a_1_3));
                int kk_2_2 = w * 4 + 2;
                int a_3_2 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[2])) : "r"(a_3_2));
                int kk_4_2 = w * 4 + 3;
                int a_5_2 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[3])) : "r"(a_5_2));
                int ta_4 = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_4), "f"(tv_4[0]), "f"(tv_4[1]), "f"(tv_4[2]), "f"(tv_4[3]) : "memory");
            }
            float r_75[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_75[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_233 = r_75[0];
            float2 _f2_20 = make_float2(a_d_233, a_d_233);
            float vq_234[8];
            float vk_235[8];
            float2 d2_12 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_20);
            float ev0_12 = approx_exp2(d2_12.x);
            float ev1_12 = approx_exp2(d2_12.y);
            float2 _f2_21 = make_float2(ev0_12, ev1_12);
            float2 _f32x2_mul_a_64 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f2_21));
            float2 _f32x2_mul_a_65 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f2_21));
            vq_234[0] = vq2_12.x;
            vq_234[1] = vq2_12.y;
            vk_235[0] = vk2_12.x;
            vk_235[1] = vk2_12.y;
            float2 d2_13 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_20);
            float ev0_13 = approx_exp2(d2_13.x);
            float ev1_13 = approx_exp2(d2_13.y);
            float2 _f2_22 = make_float2(ev0_13, ev1_13);
            float2 _f32x2_mul_a_66 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f2_22));
            float2 _f32x2_mul_a_67 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f2_22));
            vq_234[2] = vq2_13.x;
            vq_234[3] = vq2_13.y;
            vk_235[2] = vk2_13.x;
            vk_235[3] = vk2_13.y;
            float2 d2_14 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_20);
            float ev0_14 = approx_exp2(d2_14.x);
            float ev1_14 = approx_exp2(d2_14.y);
            float2 _f2_23 = make_float2(ev0_14, ev1_14);
            float2 _f32x2_mul_a_68 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f2_23));
            float2 _f32x2_mul_a_69 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f2_23));
            vq_234[4] = vq2_14.x;
            vq_234[5] = vq2_14.y;
            vk_235[4] = vk2_14.x;
            vk_235[5] = vk2_14.y;
            float2 d2_15 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_20);
            float ev0_15 = approx_exp2(d2_15.x);
            float ev1_15 = approx_exp2(d2_15.y);
            float2 _f2_24 = make_float2(ev0_15, ev1_15);
            float2 _f32x2_mul_a_70 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f2_24));
            float2 _f32x2_mul_a_71 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f2_24));
            vq_234[6] = vq2_15.x;
            vq_234[7] = vq2_15.y;
            vk_235[6] = vk2_15.x;
            vk_235[7] = vk2_15.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_234[0]), "f"(vq_234[1]), "f"(vq_234[2]), "f"(vq_234[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_235[0]), "f"(vk_235[1]), "f"(vk_235[2]), "f"(vk_235[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_234[4]), "f"(vq_234[5]), "f"(vq_234[6]), "f"(vq_234[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_235[4]), "f"(vk_235[5]), "f"(vk_235[6]), "f"(vk_235[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_5[4];
                int kk_7 = w * 4;
                int a_7 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[0])) : "r"(a_7));
                tv_5[0] = ((lane_1 > (kk_7 & 15)) ? 0.0f : tv_5[0]);
                {
                    uint32_t _addr_72 = static_cast<uint32_t>(a_7);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(tv_5[0]) : "memory");
                }
                int kk_r_6 = w * 4 + 1;
                int a_r_6 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_r_6 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_r_6 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[1])) : "r"(a_r_6));
                tv_5[1] = ((lane_1 > (kk_r_6 & 15)) ? 0.0f : tv_5[1]);
                {
                    uint32_t _addr_73 = static_cast<uint32_t>(a_r_6);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(tv_5[1]) : "memory");
                }
                int kk_r_7 = w * 4 + 2;
                int a_r_7 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_r_7 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_r_7 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[2])) : "r"(a_r_7));
                tv_5[2] = ((lane_1 > (kk_r_7 & 15)) ? 0.0f : tv_5[2]);
                {
                    uint32_t _addr_74 = static_cast<uint32_t>(a_r_7);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(tv_5[2]) : "memory");
                }
                int kk_r_8 = w * 4 + 3;
                int a_r_8 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_r_8 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_r_8 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[3])) : "r"(a_r_8));
                tv_5[3] = ((lane_1 > (kk_r_8 & 15)) ? 0.0f : tv_5[3]);
                {
                    uint32_t _addr_75 = static_cast<uint32_t>(a_r_8);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(tv_5[3]) : "memory");
                }
                int ta_5 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_5), "f"(tv_5[0]), "f"(tv_5[1]), "f"(tv_5[2]), "f"(tv_5[3]) : "memory");
            }
            float r_76[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_76[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_257 = r_76[0] - 21.640425f;
            float2 _f2_25 = make_float2(a_d_257, a_d_257);
            float vq_258[8];
            float vk_259[8];
            float2 d2_16 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_25);
            float ev0_16 = approx_exp2(d2_16.x);
            float ev1_16 = approx_exp2(d2_16.y);
            float2 _f2_26 = make_float2(ev0_16, ev1_16);
            float2 _f32x2_mul_a_76 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_16;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f2_26));
            float2 _f32x2_mul_a_77 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_16;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f2_26));
            vq_258[0] = vq2_16.x;
            vq_258[1] = vq2_16.y;
            vk_259[0] = vk2_16.x;
            vk_259[1] = vk2_16.y;
            float2 d2_17 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_25);
            float ev0_17 = approx_exp2(d2_17.x);
            float ev1_17 = approx_exp2(d2_17.y);
            float2 _f2_27 = make_float2(ev0_17, ev1_17);
            float2 _f32x2_mul_a_78 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_17;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f2_27));
            float2 _f32x2_mul_a_79 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_17;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f2_27));
            vq_258[2] = vq2_17.x;
            vq_258[3] = vq2_17.y;
            vk_259[2] = vk2_17.x;
            vk_259[3] = vk2_17.y;
            float2 d2_18 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_25);
            float ev0_18 = approx_exp2(d2_18.x);
            float ev1_18 = approx_exp2(d2_18.y);
            float2 _f2_28 = make_float2(ev0_18, ev1_18);
            float2 _f32x2_mul_a_80 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_18;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f2_28));
            float2 _f32x2_mul_a_81 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_18;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f2_28));
            vq_258[4] = vq2_18.x;
            vq_258[5] = vq2_18.y;
            vk_259[4] = vk2_18.x;
            vk_259[5] = vk2_18.y;
            float2 d2_19 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_25);
            float ev0_19 = approx_exp2(d2_19.x);
            float ev1_19 = approx_exp2(d2_19.y);
            float2 _f2_29 = make_float2(ev0_19, ev1_19);
            float2 _f32x2_mul_a_82 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_19;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f2_29));
            float2 _f32x2_mul_a_83 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_19;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f2_29));
            vq_258[6] = vq2_19.x;
            vq_258[7] = vq2_19.y;
            vk_259[6] = vk2_19.x;
            vk_259[7] = vk2_19.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_258[0]), "f"(vq_258[1]), "f"(vq_258[2]), "f"(vq_258[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_259[0]), "f"(vk_259[1]), "f"(vk_259[2]), "f"(vk_259[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_258[4]), "f"(vq_258[5]), "f"(vq_258[6]), "f"(vq_258[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_259[4]), "f"(vk_259[5]), "f"(vk_259[6]), "f"(vk_259[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ops_full_addr + 24, 0);
            mbarrier_wait(blk_full_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (clen < 64) {
                mbarrier_wait(kq_ready_addr + 24, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int sbase_280 = 48 + 8 * th;
            int gbase_281 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_280 * 128);
            int kbase_282 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_280 * 128);
            int qbase_283 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float g_s_284[8];
            float q_s_285[8];
            float kb_s_286[8];
            float bv_287[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_287[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_280 * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_287[4])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_287[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_280 * 4 + 16)));
            float r_77[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_77[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[0] = r_77[0];
            float r_78[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_78[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[1] = r_78[0];
            unsigned int r_79[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_79[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[0] = __uint_as_float(r_79[0] << 16);
            unsigned int r_80[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_80[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[1] = __uint_as_float(r_80[0] << 16);
            unsigned int r_81[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_81[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_8 = __uint_as_float(r_81[0] << 16);
            unsigned int r_82[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_82[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_8 = __uint_as_float(r_82[0] << 16);
            float2 _f32x2_mul_a_84 = make_float2(k_s0_8, k_s1_8);
            float2 _f32x2_mul_b_85 = make_float2(bv_287[0], bv_287[1]);
            float2 kp_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
            __nv_bfloat16 kb0_8 = (__nv_bfloat16)kp_8.x;
            __nv_bfloat16 kb1_8 = (__nv_bfloat16)kp_8.y;
            kb_s_286[0] = (float)kb0_8;
            kb_s_286[1] = (float)kb1_8;
            float r_83[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_83[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[2] = r_83[0];
            float r_84[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_84[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[3] = r_84[0];
            unsigned int r_85[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_85[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[2] = __uint_as_float(r_85[0] << 16);
            unsigned int r_86[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_86[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[3] = __uint_as_float(r_86[0] << 16);
            unsigned int r_87[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_87[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_9 = __uint_as_float(r_87[0] << 16);
            unsigned int r_88[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_88[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_9 = __uint_as_float(r_88[0] << 16);
            float2 _f32x2_mul_a_86 = make_float2(k_s0_9, k_s1_9);
            float2 _f32x2_mul_b_87 = make_float2(bv_287[2], bv_287[3]);
            float2 kp_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
            __nv_bfloat16 kb0_9 = (__nv_bfloat16)kp_9.x;
            __nv_bfloat16 kb1_9 = (__nv_bfloat16)kp_9.y;
            kb_s_286[2] = (float)kb0_9;
            kb_s_286[3] = (float)kb1_9;
            float r_89[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_89[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[4] = r_89[0];
            float r_90[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_90[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[5] = r_90[0];
            unsigned int r_91[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_91[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[4] = __uint_as_float(r_91[0] << 16);
            unsigned int r_92[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_92[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[5] = __uint_as_float(r_92[0] << 16);
            unsigned int r_93[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_93[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_10 = __uint_as_float(r_93[0] << 16);
            unsigned int r_94[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_94[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_10 = __uint_as_float(r_94[0] << 16);
            float2 _f32x2_mul_a_88 = make_float2(k_s0_10, k_s1_10);
            float2 _f32x2_mul_b_89 = make_float2(bv_287[4], bv_287[5]);
            float2 kp_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
            __nv_bfloat16 kb0_10 = (__nv_bfloat16)kp_10.x;
            __nv_bfloat16 kb1_10 = (__nv_bfloat16)kp_10.y;
            kb_s_286[4] = (float)kb0_10;
            kb_s_286[5] = (float)kb1_10;
            float r_95[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_95[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[6] = r_95[0];
            float r_96[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_96[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[7] = r_96[0];
            unsigned int r_97[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_97[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[6] = __uint_as_float(r_97[0] << 16);
            unsigned int r_98[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_98[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[7] = __uint_as_float(r_98[0] << 16);
            unsigned int r_99[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_99[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_11 = __uint_as_float(r_99[0] << 16);
            unsigned int r_100[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_100[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_11 = __uint_as_float(r_100[0] << 16);
            float2 _f32x2_mul_a_90 = make_float2(k_s0_11, k_s1_11);
            float2 _f32x2_mul_b_91 = make_float2(bv_287[6], bv_287[7]);
            float2 kp_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
            __nv_bfloat16 kb0_11 = (__nv_bfloat16)kp_11.x;
            __nv_bfloat16 kb1_11 = (__nv_bfloat16)kp_11.y;
            kb_s_286[6] = (float)kb0_11;
            kb_s_286[7] = (float)kb1_11;
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_6[4];
                int kk_8 = w * 4;
                int a_8 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[0])) : "r"(a_8));
                int kk_0_3 = w * 4 + 1;
                int a_1_4 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_0_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[1])) : "r"(a_1_4));
                int kk_2_3 = w * 4 + 2;
                int a_3_3 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_2_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[2])) : "r"(a_3_3));
                int kk_4_3 = w * 4 + 3;
                int a_5_3 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_4_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[3])) : "r"(a_5_3));
                int ta_6 = (tiles_addr + 4096 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_6), "f"(tv_6[0]), "f"(tv_6[1]), "f"(tv_6[2]), "f"(tv_6[3]) : "memory");
            }
            float r_101[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_101[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_333 = r_101[0];
            float2 _f2_30 = make_float2(a_d_333, a_d_333);
            float vq_334[8];
            float vk_335[8];
            float2 d2_20 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_30);
            float ev0_20 = approx_exp2(d2_20.x);
            float ev1_20 = approx_exp2(d2_20.y);
            float2 _f2_31 = make_float2(ev0_20, ev1_20);
            float2 _f32x2_mul_a_92 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_20;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f2_31));
            float2 _f32x2_mul_a_93 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_20;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f2_31));
            vq_334[0] = vq2_20.x;
            vq_334[1] = vq2_20.y;
            vk_335[0] = vk2_20.x;
            vk_335[1] = vk2_20.y;
            float2 d2_21 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_30);
            float ev0_21 = approx_exp2(d2_21.x);
            float ev1_21 = approx_exp2(d2_21.y);
            float2 _f2_32 = make_float2(ev0_21, ev1_21);
            float2 _f32x2_mul_a_94 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_21;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&_f2_32));
            float2 _f32x2_mul_a_95 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_21;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f2_32));
            vq_334[2] = vq2_21.x;
            vq_334[3] = vq2_21.y;
            vk_335[2] = vk2_21.x;
            vk_335[3] = vk2_21.y;
            float2 d2_22 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_30);
            float ev0_22 = approx_exp2(d2_22.x);
            float ev1_22 = approx_exp2(d2_22.y);
            float2 _f2_33 = make_float2(ev0_22, ev1_22);
            float2 _f32x2_mul_a_96 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_22;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f2_33));
            float2 _f32x2_mul_a_97 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_22;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f2_33));
            vq_334[4] = vq2_22.x;
            vq_334[5] = vq2_22.y;
            vk_335[4] = vk2_22.x;
            vk_335[5] = vk2_22.y;
            float2 d2_23 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_30);
            float ev0_23 = approx_exp2(d2_23.x);
            float ev1_23 = approx_exp2(d2_23.y);
            float2 _f2_34 = make_float2(ev0_23, ev1_23);
            float2 _f32x2_mul_a_98 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_f2_34));
            float2 _f32x2_mul_a_99 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f2_34));
            vq_334[6] = vq2_23.x;
            vq_334[7] = vq2_23.y;
            vk_335[6] = vk2_23.x;
            vk_335[7] = vk2_23.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_334[0]), "f"(vq_334[1]), "f"(vq_334[2]), "f"(vq_334[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_335[0]), "f"(vk_335[1]), "f"(vk_335[2]), "f"(vk_335[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_334[4]), "f"(vq_334[5]), "f"(vq_334[6]), "f"(vq_334[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_335[4]), "f"(vk_335[5]), "f"(vk_335[6]), "f"(vk_335[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_7[4];
                int kk_9 = w * 4;
                int a_9 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_9 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_9 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[0])) : "r"(a_9));
                int kk_0_4 = w * 4 + 1;
                int a_1_5 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[1])) : "r"(a_1_5));
                int kk_2_4 = w * 4 + 2;
                int a_3_4 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[2])) : "r"(a_3_4));
                int kk_4_4 = w * 4 + 3;
                int a_5_4 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[3])) : "r"(a_5_4));
                int ta_7 = (tiles_addr + 6144 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_7), "f"(tv_7[0]), "f"(tv_7[1]), "f"(tv_7[2]), "f"(tv_7[3]) : "memory");
            }
            float r_102[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_102[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_357 = r_102[0];
            float2 _f2_35 = make_float2(a_d_357, a_d_357);
            float vq_358[8];
            float vk_359[8];
            float2 d2_24 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_35);
            float ev0_24 = approx_exp2(d2_24.x);
            float ev1_24 = approx_exp2(d2_24.y);
            float2 _f2_36 = make_float2(ev0_24, ev1_24);
            float2 _f32x2_mul_a_100 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_24;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f2_36));
            float2 _f32x2_mul_a_101 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_24;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f2_36));
            vq_358[0] = vq2_24.x;
            vq_358[1] = vq2_24.y;
            vk_359[0] = vk2_24.x;
            vk_359[1] = vk2_24.y;
            float2 d2_25 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_35);
            float ev0_25 = approx_exp2(d2_25.x);
            float ev1_25 = approx_exp2(d2_25.y);
            float2 _f2_37 = make_float2(ev0_25, ev1_25);
            float2 _f32x2_mul_a_102 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_25;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f2_37));
            float2 _f32x2_mul_a_103 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_25;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f2_37));
            vq_358[2] = vq2_25.x;
            vq_358[3] = vq2_25.y;
            vk_359[2] = vk2_25.x;
            vk_359[3] = vk2_25.y;
            float2 d2_26 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_35);
            float ev0_26 = approx_exp2(d2_26.x);
            float ev1_26 = approx_exp2(d2_26.y);
            float2 _f2_38 = make_float2(ev0_26, ev1_26);
            float2 _f32x2_mul_a_104 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f2_38));
            float2 _f32x2_mul_a_105 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f2_38));
            vq_358[4] = vq2_26.x;
            vq_358[5] = vq2_26.y;
            vk_359[4] = vk2_26.x;
            vk_359[5] = vk2_26.y;
            float2 d2_27 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_35);
            float ev0_27 = approx_exp2(d2_27.x);
            float ev1_27 = approx_exp2(d2_27.y);
            float2 _f2_39 = make_float2(ev0_27, ev1_27);
            float2 _f32x2_mul_a_106 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_27;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&_f2_39));
            float2 _f32x2_mul_a_107 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_27;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f2_39));
            vq_358[6] = vq2_27.x;
            vq_358[7] = vq2_27.y;
            vk_359[6] = vk2_27.x;
            vk_359[7] = vk2_27.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_358[0]), "f"(vq_358[1]), "f"(vq_358[2]), "f"(vq_358[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_359[0]), "f"(vk_359[1]), "f"(vk_359[2]), "f"(vk_359[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_358[4]), "f"(vq_358[5]), "f"(vq_358[6]), "f"(vq_358[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_359[4]), "f"(vk_359[5]), "f"(vk_359[6]), "f"(vk_359[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_8[4];
                int kk_10 = w * 4;
                int a_10 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_10 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_10 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[0])) : "r"(a_10));
                int kk_0_5 = w * 4 + 1;
                int a_1_6 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[1])) : "r"(a_1_6));
                int kk_2_5 = w * 4 + 2;
                int a_3_5 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[2])) : "r"(a_3_5));
                int kk_4_5 = w * 4 + 3;
                int a_5_5 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[3])) : "r"(a_5_5));
                int ta_8 = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_8), "f"(tv_8[0]), "f"(tv_8[1]), "f"(tv_8[2]), "f"(tv_8[3]) : "memory");
            }
            float r_103[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_103[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_381 = r_103[0];
            float2 _f2_40 = make_float2(a_d_381, a_d_381);
            float vq_382[8];
            float vk_383[8];
            float2 d2_28 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_40);
            float ev0_28 = approx_exp2(d2_28.x);
            float ev1_28 = approx_exp2(d2_28.y);
            float2 _f2_41 = make_float2(ev0_28, ev1_28);
            float2 _f32x2_mul_a_108 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_28;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_f2_41));
            float2 _f32x2_mul_a_109 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_28;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f2_41));
            vq_382[0] = vq2_28.x;
            vq_382[1] = vq2_28.y;
            vk_383[0] = vk2_28.x;
            vk_383[1] = vk2_28.y;
            float2 d2_29 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_40);
            float ev0_29 = approx_exp2(d2_29.x);
            float ev1_29 = approx_exp2(d2_29.y);
            float2 _f2_42 = make_float2(ev0_29, ev1_29);
            float2 _f32x2_mul_a_110 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_29;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&_f2_42));
            float2 _f32x2_mul_a_111 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_29;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_f2_42));
            vq_382[2] = vq2_29.x;
            vq_382[3] = vq2_29.y;
            vk_383[2] = vk2_29.x;
            vk_383[3] = vk2_29.y;
            float2 d2_30 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_40);
            float ev0_30 = approx_exp2(d2_30.x);
            float ev1_30 = approx_exp2(d2_30.y);
            float2 _f2_43 = make_float2(ev0_30, ev1_30);
            float2 _f32x2_mul_a_112 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_30;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&_f2_43));
            float2 _f32x2_mul_a_113 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_30;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f2_43));
            vq_382[4] = vq2_30.x;
            vq_382[5] = vq2_30.y;
            vk_383[4] = vk2_30.x;
            vk_383[5] = vk2_30.y;
            float2 d2_31 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_40);
            float ev0_31 = approx_exp2(d2_31.x);
            float ev1_31 = approx_exp2(d2_31.y);
            float2 _f2_44 = make_float2(ev0_31, ev1_31);
            float2 _f32x2_mul_a_114 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_31;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_f2_44));
            float2 _f32x2_mul_a_115 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_31;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_f2_44));
            vq_382[6] = vq2_31.x;
            vq_382[7] = vq2_31.y;
            vk_383[6] = vk2_31.x;
            vk_383[7] = vk2_31.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_382[0]), "f"(vq_382[1]), "f"(vq_382[2]), "f"(vq_382[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_383[0]), "f"(vk_383[1]), "f"(vk_383[2]), "f"(vk_383[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_382[4]), "f"(vq_382[5]), "f"(vq_382[6]), "f"(vq_382[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_383[4]), "f"(vk_383[5]), "f"(vk_383[6]), "f"(vk_383[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_9[4];
                int kk_11 = w * 4;
                int a_11 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[0])) : "r"(a_11));
                tv_9[0] = ((lane_1 > (kk_11 & 15)) ? 0.0f : tv_9[0]);
                {
                    uint32_t _addr_116 = static_cast<uint32_t>(a_11);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(tv_9[0]) : "memory");
                }
                int kk_r_9 = w * 4 + 1;
                int a_r_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_r_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_r_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[1])) : "r"(a_r_9));
                tv_9[1] = ((lane_1 > (kk_r_9 & 15)) ? 0.0f : tv_9[1]);
                {
                    uint32_t _addr_117 = static_cast<uint32_t>(a_r_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(tv_9[1]) : "memory");
                }
                int kk_r_10 = w * 4 + 2;
                int a_r_10 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_r_10 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_r_10 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[2])) : "r"(a_r_10));
                tv_9[2] = ((lane_1 > (kk_r_10 & 15)) ? 0.0f : tv_9[2]);
                {
                    uint32_t _addr_118 = static_cast<uint32_t>(a_r_10);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(tv_9[2]) : "memory");
                }
                int kk_r_11 = w * 4 + 3;
                int a_r_11 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_r_11 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_r_11 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[3])) : "r"(a_r_11));
                tv_9[3] = ((lane_1 > (kk_r_11 & 15)) ? 0.0f : tv_9[3]);
                {
                    uint32_t _addr_119 = static_cast<uint32_t>(a_r_11);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(tv_9[3]) : "memory");
                }
                int ta_9 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_9), "f"(tv_9[0]), "f"(tv_9[1]), "f"(tv_9[2]), "f"(tv_9[3]) : "memory");
            }
            float r_104[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_104[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_405 = r_104[0] - 21.640425f;
            float2 _f2_45 = make_float2(a_d_405, a_d_405);
            float vq_406[8];
            float vk_407[8];
            float2 d2_32 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_45);
            float ev0_32 = approx_exp2(d2_32.x);
            float ev1_32 = approx_exp2(d2_32.y);
            float2 _f2_46 = make_float2(ev0_32, ev1_32);
            float2 _f32x2_mul_a_120 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_32;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&_f2_46));
            float2 _f32x2_mul_a_121 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_32;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f2_46));
            vq_406[0] = vq2_32.x;
            vq_406[1] = vq2_32.y;
            vk_407[0] = vk2_32.x;
            vk_407[1] = vk2_32.y;
            float2 d2_33 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_45);
            float ev0_33 = approx_exp2(d2_33.x);
            float ev1_33 = approx_exp2(d2_33.y);
            float2 _f2_47 = make_float2(ev0_33, ev1_33);
            float2 _f32x2_mul_a_122 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_33;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&_f2_47));
            float2 _f32x2_mul_a_123 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_33;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_f2_47));
            vq_406[2] = vq2_33.x;
            vq_406[3] = vq2_33.y;
            vk_407[2] = vk2_33.x;
            vk_407[3] = vk2_33.y;
            float2 d2_34 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_45);
            float ev0_34 = approx_exp2(d2_34.x);
            float ev1_34 = approx_exp2(d2_34.y);
            float2 _f2_48 = make_float2(ev0_34, ev1_34);
            float2 _f32x2_mul_a_124 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_34;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&_f2_48));
            float2 _f32x2_mul_a_125 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_34;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f2_48));
            vq_406[4] = vq2_34.x;
            vq_406[5] = vq2_34.y;
            vk_407[4] = vk2_34.x;
            vk_407[5] = vk2_34.y;
            float2 d2_35 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_45);
            float ev0_35 = approx_exp2(d2_35.x);
            float ev1_35 = approx_exp2(d2_35.y);
            float2 _f2_49 = make_float2(ev0_35, ev1_35);
            float2 _f32x2_mul_a_126 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_35;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&_f2_49));
            float2 _f32x2_mul_a_127 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_35;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&_f2_49));
            vq_406[6] = vq2_35.x;
            vq_406[7] = vq2_35.y;
            vk_407[6] = vk2_35.x;
            vk_407[7] = vk2_35.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_406[0]), "f"(vq_406[1]), "f"(vq_406[2]), "f"(vq_406[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_407[0]), "f"(vk_407[1]), "f"(vk_407[2]), "f"(vk_407[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_406[4]), "f"(vq_406[5]), "f"(vq_406[6]), "f"(vq_406[7]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_407[4]), "f"(vk_407[5]), "f"(vk_407[6]), "f"(vk_407[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_428 = 8 * th;
            int gbase_429 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_428 * 128);
            int kbase_430 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_428 * 128);
            float r_105[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_105[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0 = r_105[0];
            float r_106[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_106[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1 = r_106[0];
            float gs[8];
            float ks[8];
            float r_107[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_107[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[0] = r_107[0];
            unsigned int r_108[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_108[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[0] = __uint_as_float(r_108[0] << 16);
            float r_109[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_109[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[1] = r_109[0];
            unsigned int r_110[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_110[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[1] = __uint_as_float(r_110[0] << 16);
            float r_111[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_111[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[2] = r_111[0];
            unsigned int r_112[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_112[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[2] = __uint_as_float(r_112[0] << 16);
            float r_113[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_113[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[3] = r_113[0];
            unsigned int r_114[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_114[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[3] = __uint_as_float(r_114[0] << 16);
            float r_115[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_115[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[4] = r_115[0];
            unsigned int r_116[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_116[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[4] = __uint_as_float(r_116[0] << 16);
            float r_117[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_117[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[5] = r_117[0];
            unsigned int r_118[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_118[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[5] = __uint_as_float(r_118[0] << 16);
            float r_119[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_119[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[6] = r_119[0];
            unsigned int r_120[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_120[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[6] = __uint_as_float(r_120[0] << 16);
            float r_121[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_121[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[7] = r_121[0];
            unsigned int r_122[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_122[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[7] = __uint_as_float(r_122[0] << 16);
            float2 _f2_50 = make_float2(a0, a0);
            float v_1_0[8];
            float2 d2_36 = sub_f32x2(_f2_50, make_float2(gs[0], gs[1]));
            float e0_1_0 = approx_exp2(d2_36.x);
            float e1_1_0 = approx_exp2(d2_36.y);
            float2 _f32x2_mul_a_128 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_129 = make_float2(e0_1_0, e1_1_0);
            float2 v2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
            v_1_0[0] = v2_1_0.x;
            v_1_0[1] = v2_1_0.y;
            float2 d2_37 = sub_f32x2(_f2_50, make_float2(gs[2], gs[3]));
            float e0_1_1 = approx_exp2(d2_37.x);
            float e1_1_1 = approx_exp2(d2_37.y);
            float2 _f32x2_mul_a_130 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_131 = make_float2(e0_1_1, e1_1_1);
            float2 v2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
            v_1_0[2] = v2_1_1.x;
            v_1_0[3] = v2_1_1.y;
            float2 d2_38 = sub_f32x2(_f2_50, make_float2(gs[4], gs[5]));
            float e0_1_2 = approx_exp2(d2_38.x);
            float e1_1_2 = approx_exp2(d2_38.y);
            float2 _f32x2_mul_a_132 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_133 = make_float2(e0_1_2, e1_1_2);
            float2 v2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
            v_1_0[4] = v2_1_2.x;
            v_1_0[5] = v2_1_2.y;
            float2 d2_39 = sub_f32x2(_f2_50, make_float2(gs[6], gs[7]));
            float e0_1_3 = approx_exp2(d2_39.x);
            float e1_1_3 = approx_exp2(d2_39.y);
            float2 _f32x2_mul_a_134 = make_float2(ks[6], ks[7]);
            float2 _f32x2_mul_b_135 = make_float2(e0_1_3, e1_1_3);
            float2 v2_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
            v_1_0[6] = v2_1_3.x;
            v_1_0[7] = v2_1_3.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_0[0]), "f"(v_1_0[1]), "f"(v_1_0[2]), "f"(v_1_0[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_0[4]), "f"(v_1_0[5]), "f"(v_1_0[6]), "f"(v_1_0[7]) : "memory");
            float2 _f2_51 = make_float2(a1, a1);
            float v_1_1[8];
            float2 d2_40 = sub_f32x2(_f2_51, make_float2(gs[0], gs[1]));
            float e0_1_4 = approx_exp2(d2_40.x);
            float e1_1_4 = approx_exp2(d2_40.y);
            float2 _f32x2_mul_a_136 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_137 = make_float2(e0_1_4, e1_1_4);
            float2 v2_1_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
            v_1_1[0] = v2_1_4.x;
            v_1_1[1] = v2_1_4.y;
            float2 d2_41 = sub_f32x2(_f2_51, make_float2(gs[2], gs[3]));
            float e0_1_5 = approx_exp2(d2_41.x);
            float e1_1_5 = approx_exp2(d2_41.y);
            float2 _f32x2_mul_a_138 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_139 = make_float2(e0_1_5, e1_1_5);
            float2 v2_1_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
            v_1_1[2] = v2_1_5.x;
            v_1_1[3] = v2_1_5.y;
            float2 d2_42 = sub_f32x2(_f2_51, make_float2(gs[4], gs[5]));
            float e0_1_6 = approx_exp2(d2_42.x);
            float e1_1_6 = approx_exp2(d2_42.y);
            float2 _f32x2_mul_a_140 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_141 = make_float2(e0_1_6, e1_1_6);
            float2 v2_1_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
            v_1_1[4] = v2_1_6.x;
            v_1_1[5] = v2_1_6.y;
            float2 d2_43 = sub_f32x2(_f2_51, make_float2(gs[6], gs[7]));
            float e0_1_7 = approx_exp2(d2_43.x);
            float e1_1_7 = approx_exp2(d2_43.y);
            float2 _f32x2_mul_a_142 = make_float2(ks[6], ks[7]);
            float2 _f32x2_mul_b_143 = make_float2(e0_1_7, e1_1_7);
            float2 v2_1_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
            v_1_1[6] = v2_1_7.x;
            v_1_1[7] = v2_1_7.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_1[0]), "f"(v_1_1[1]), "f"(v_1_1[2]), "f"(v_1_1[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_1[4]), "f"(v_1_1[5]), "f"(v_1_1[6]), "f"(v_1_1[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_479 = 16 + 8 * th;
            int gbase_480 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_479 * 128);
            int kbase_481 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_479 * 128);
            float r_123[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_123[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_483 = r_123[0];
            float r_124[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_124[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1_485 = r_124[0];
            float gs_486[8];
            float ks_487[8];
            float r_125[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_125[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[0] = r_125[0];
            unsigned int r_126[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_126[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[0] = __uint_as_float(r_126[0] << 16);
            float r_127[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_127[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[1] = r_127[0];
            unsigned int r_128[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_128[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[1] = __uint_as_float(r_128[0] << 16);
            float r_129[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_129[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[2] = r_129[0];
            unsigned int r_130[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_130[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[2] = __uint_as_float(r_130[0] << 16);
            float r_131[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_131[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[3] = r_131[0];
            unsigned int r_132[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_132[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[3] = __uint_as_float(r_132[0] << 16);
            float r_133[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_133[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[4] = r_133[0];
            unsigned int r_134[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_134[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[4] = __uint_as_float(r_134[0] << 16);
            float r_135[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_135[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[5] = r_135[0];
            unsigned int r_136[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_136[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[5] = __uint_as_float(r_136[0] << 16);
            float r_137[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_137[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[6] = r_137[0];
            unsigned int r_138[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_138[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[6] = __uint_as_float(r_138[0] << 16);
            float r_139[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_139[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[7] = r_139[0];
            unsigned int r_140[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_140[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[7] = __uint_as_float(r_140[0] << 16);
            float2 _f2_52 = make_float2(a0_483, a0_483);
            float v_0[8];
            float2 d2_44 = sub_f32x2(_f2_52, make_float2(gs_486[0], gs_486[1]));
            float e0_0 = approx_exp2(d2_44.x);
            float e1_0 = approx_exp2(d2_44.y);
            float2 _f32x2_mul_a_144 = make_float2(ks_487[0], ks_487[1]);
            float2 _f32x2_mul_b_145 = make_float2(e0_0, e1_0);
            float2 v2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
            v_0[0] = v2_0.x;
            v_0[1] = v2_0.y;
            float2 d2_45 = sub_f32x2(_f2_52, make_float2(gs_486[2], gs_486[3]));
            float e0_1 = approx_exp2(d2_45.x);
            float e1_1 = approx_exp2(d2_45.y);
            float2 _f32x2_mul_a_146 = make_float2(ks_487[2], ks_487[3]);
            float2 _f32x2_mul_b_147 = make_float2(e0_1, e1_1);
            float2 v2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
            v_0[2] = v2_1.x;
            v_0[3] = v2_1.y;
            float2 d2_46 = sub_f32x2(_f2_52, make_float2(gs_486[4], gs_486[5]));
            float e0_2 = approx_exp2(d2_46.x);
            float e1_2 = approx_exp2(d2_46.y);
            float2 _f32x2_mul_a_148 = make_float2(ks_487[4], ks_487[5]);
            float2 _f32x2_mul_b_149 = make_float2(e0_2, e1_2);
            float2 v2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
            v_0[4] = v2_2.x;
            v_0[5] = v2_2.y;
            float2 d2_47 = sub_f32x2(_f2_52, make_float2(gs_486[6], gs_486[7]));
            float e0_3 = approx_exp2(d2_47.x);
            float e1_3 = approx_exp2(d2_47.y);
            float2 _f32x2_mul_a_150 = make_float2(ks_487[6], ks_487[7]);
            float2 _f32x2_mul_b_151 = make_float2(e0_3, e1_3);
            float2 v2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
            v_0[6] = v2_3.x;
            v_0[7] = v2_3.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_0[0]), "f"(v_0[1]), "f"(v_0[2]), "f"(v_0[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_0[4]), "f"(v_0[5]), "f"(v_0[6]), "f"(v_0[7]) : "memory");
            float2 _f2_53 = make_float2(a1_485, a1_485);
            float v_1[8];
            float2 d2_48 = sub_f32x2(_f2_53, make_float2(gs_486[0], gs_486[1]));
            float e0_4 = approx_exp2(d2_48.x);
            float e1_4 = approx_exp2(d2_48.y);
            float2 _f32x2_mul_a_152 = make_float2(ks_487[0], ks_487[1]);
            float2 _f32x2_mul_b_153 = make_float2(e0_4, e1_4);
            float2 v2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
            v_1[0] = v2_4.x;
            v_1[1] = v2_4.y;
            float2 d2_49 = sub_f32x2(_f2_53, make_float2(gs_486[2], gs_486[3]));
            float e0_5 = approx_exp2(d2_49.x);
            float e1_5 = approx_exp2(d2_49.y);
            float2 _f32x2_mul_a_154 = make_float2(ks_487[2], ks_487[3]);
            float2 _f32x2_mul_b_155 = make_float2(e0_5, e1_5);
            float2 v2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
            v_1[2] = v2_5.x;
            v_1[3] = v2_5.y;
            float2 d2_50 = sub_f32x2(_f2_53, make_float2(gs_486[4], gs_486[5]));
            float e0_6 = approx_exp2(d2_50.x);
            float e1_6 = approx_exp2(d2_50.y);
            float2 _f32x2_mul_a_156 = make_float2(ks_487[4], ks_487[5]);
            float2 _f32x2_mul_b_157 = make_float2(e0_6, e1_6);
            float2 v2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
            v_1[4] = v2_6.x;
            v_1[5] = v2_6.y;
            float2 d2_51 = sub_f32x2(_f2_53, make_float2(gs_486[6], gs_486[7]));
            float e0_7 = approx_exp2(d2_51.x);
            float e1_7 = approx_exp2(d2_51.y);
            float2 _f32x2_mul_a_158 = make_float2(ks_487[6], ks_487[7]);
            float2 _f32x2_mul_b_159 = make_float2(e0_7, e1_7);
            float2 v2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
            v_1[6] = v2_7.x;
            v_1[7] = v2_7.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1[0]), "f"(v_1[1]), "f"(v_1[2]), "f"(v_1[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1[4]), "f"(v_1[5]), "f"(v_1[6]), "f"(v_1[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_538 = 32 + 8 * th;
            int gbase_539 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_538 * 128);
            int kbase_540 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_538 * 128);
            float r_141[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_141[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_542 = r_141[0];
            float a1_543 = 0.0f;
            float gs_544[8];
            float ks_545[8];
            float r_142[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_142[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[0] = r_142[0];
            unsigned int r_143[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_143[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[0] = __uint_as_float(r_143[0] << 16);
            float r_144[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_144[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[1] = r_144[0];
            unsigned int r_145[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_145[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[1] = __uint_as_float(r_145[0] << 16);
            float r_146[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_146[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[2] = r_146[0];
            unsigned int r_147[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_147[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[2] = __uint_as_float(r_147[0] << 16);
            float r_148[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_148[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[3] = r_148[0];
            unsigned int r_149[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_149[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[3] = __uint_as_float(r_149[0] << 16);
            float r_150[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_150[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[4] = r_150[0];
            unsigned int r_151[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_151[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[4] = __uint_as_float(r_151[0] << 16);
            float r_152[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_152[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[5] = r_152[0];
            unsigned int r_153[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_153[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[5] = __uint_as_float(r_153[0] << 16);
            float r_154[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_154[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[6] = r_154[0];
            unsigned int r_155[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_155[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[6] = __uint_as_float(r_155[0] << 16);
            float r_156[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_156[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[7] = r_156[0];
            unsigned int r_157[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_157[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[7] = __uint_as_float(r_157[0] << 16);
            float2 _f2_54 = make_float2(a0_542, a0_542);
            float v_562[8];
            float2 d2_52 = sub_f32x2(_f2_54, make_float2(gs_544[0], gs_544[1]));
            float e0_8 = approx_exp2(d2_52.x);
            float e1_8 = approx_exp2(d2_52.y);
            float2 _f32x2_mul_a_160 = make_float2(ks_545[0], ks_545[1]);
            float2 _f32x2_mul_b_161 = make_float2(e0_8, e1_8);
            float2 v2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
            v_562[0] = v2_8.x;
            v_562[1] = v2_8.y;
            float2 d2_53 = sub_f32x2(_f2_54, make_float2(gs_544[2], gs_544[3]));
            float e0_9 = approx_exp2(d2_53.x);
            float e1_9 = approx_exp2(d2_53.y);
            float2 _f32x2_mul_a_162 = make_float2(ks_545[2], ks_545[3]);
            float2 _f32x2_mul_b_163 = make_float2(e0_9, e1_9);
            float2 v2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
            v_562[2] = v2_9.x;
            v_562[3] = v2_9.y;
            float2 d2_54 = sub_f32x2(_f2_54, make_float2(gs_544[4], gs_544[5]));
            float e0_10 = approx_exp2(d2_54.x);
            float e1_10 = approx_exp2(d2_54.y);
            float2 _f32x2_mul_a_164 = make_float2(ks_545[4], ks_545[5]);
            float2 _f32x2_mul_b_165 = make_float2(e0_10, e1_10);
            float2 v2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
            v_562[4] = v2_10.x;
            v_562[5] = v2_10.y;
            float2 d2_55 = sub_f32x2(_f2_54, make_float2(gs_544[6], gs_544[7]));
            float e0_11 = approx_exp2(d2_55.x);
            float e1_11 = approx_exp2(d2_55.y);
            float2 _f32x2_mul_a_166 = make_float2(ks_545[6], ks_545[7]);
            float2 _f32x2_mul_b_167 = make_float2(e0_11, e1_11);
            float2 v2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
            v_562[6] = v2_11.x;
            v_562[7] = v2_11.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_562[0]), "f"(v_562[1]), "f"(v_562[2]), "f"(v_562[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_562[4]), "f"(v_562[5]), "f"(v_562[6]), "f"(v_562[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(p2_done_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (32 + 8 * th)));
            float _tmem_load_1[8];
            tmem_ld_x8(&_tmem_load_1[0], taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_158[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_158[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl = r_158[0];
            float r_159[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_159[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc = r_159[0] - 21.640425f;
            float dkt[8];
            float pan[8];
            float2 _f2_55 = make_float2(gc, gc);
            float2 _f2_56 = make_float2(gnl, gnl);
            int t0_1_0 = 32 + 8 * th;
            int t1_1_0 = 32 + 8 * th + 1;
            float r_160[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_160[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_0 = r_160[0];
            float r_161[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_161[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_0 = r_161[0];
            unsigned int r_162[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_162[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_0 = __uint_as_float(r_162[0] << 16);
            unsigned int r_163[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_163[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_0 = __uint_as_float(r_163[0] << 16);
            float2 _f2_57 = make_float2(g_t0_1_0, g_t1_1_0);
            float2 dc2_1_0 = sub_f32x2(_f2_55, _f2_57);
            float e2_0 = approx_exp2(dc2_1_0.x);
            float e2_2_0 = approx_exp2(dc2_1_0.y);
            float2 dn2_1_0 = sub_f32x2(_f2_56, _f2_57);
            float en_0 = approx_exp2(dn2_1_0.x);
            float en_2_0 = approx_exp2(dn2_1_0.y);
            float2 _f32x2_mul_a_168 = make_float2(k_t0_1_0, k_t1_1_0);
            float2 _f32x2_mul_b_169 = make_float2(e2_0, e2_2_0);
            float2 p2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
            float dl = __uint_as_float(__as_u32(p2_1_0.x) & 4294959104u) - p2_1_0.x;
            float _rcp_170 = approx_rcp(k_t0_1_0);
            float ft = ((dl == 0.0f) ? e2_0 : __fmaf_rn(dl, _rcp_170, e2_0));
            float dl_0 = __uint_as_float(__as_u32(p2_1_0.y) & 4294959104u) - p2_1_0.y;
            float _rcp_171 = approx_rcp(k_t1_1_0);
            float ft_0 = ((dl_0 == 0.0f) ? e2_2_0 : __fmaf_rn(dl_0, _rcp_171, e2_2_0));
            dkt[0] = __fmaf_rn(_tmem_load_1[0], en_0, _tmem_load_0[0] * ft);
            dkt[1] = __fmaf_rn(_tmem_load_1[1], en_2_0, _tmem_load_0[1] * ft_0);
            pan[0] = p2_1_0.x;
            pan[1] = p2_1_0.y;
            int t0_1_1 = 32 + 8 * th + 2;
            int t1_1_1 = 32 + 8 * th + 2 + 1;
            float r_164[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_164[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_1 = r_164[0];
            float r_165[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_165[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_1 = r_165[0];
            unsigned int r_166[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_166[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_1 = __uint_as_float(r_166[0] << 16);
            unsigned int r_167[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_167[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_1 = __uint_as_float(r_167[0] << 16);
            float2 _f2_58 = make_float2(g_t0_1_1, g_t1_1_1);
            float2 dc2_1_1 = sub_f32x2(_f2_55, _f2_58);
            float e2_1 = approx_exp2(dc2_1_1.x);
            float e2_2_1 = approx_exp2(dc2_1_1.y);
            float2 dn2_1_1 = sub_f32x2(_f2_56, _f2_58);
            float en_1 = approx_exp2(dn2_1_1.x);
            float en_2_1 = approx_exp2(dn2_1_1.y);
            float2 _f32x2_mul_a_172 = make_float2(k_t0_1_1, k_t1_1_1);
            float2 _f32x2_mul_b_173 = make_float2(e2_1, e2_2_1);
            float2 p2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
            float dl_1 = __uint_as_float(__as_u32(p2_1_1.x) & 4294959104u) - p2_1_1.x;
            float _rcp_174 = approx_rcp(k_t0_1_1);
            float ft_1 = ((dl_1 == 0.0f) ? e2_1 : __fmaf_rn(dl_1, _rcp_174, e2_1));
            float dl_2 = __uint_as_float(__as_u32(p2_1_1.y) & 4294959104u) - p2_1_1.y;
            float _rcp_175 = approx_rcp(k_t1_1_1);
            float ft_2 = ((dl_2 == 0.0f) ? e2_2_1 : __fmaf_rn(dl_2, _rcp_175, e2_2_1));
            dkt[2] = __fmaf_rn(_tmem_load_1[2], en_1, _tmem_load_0[2] * ft_1);
            dkt[3] = __fmaf_rn(_tmem_load_1[3], en_2_1, _tmem_load_0[3] * ft_2);
            pan[2] = p2_1_1.x;
            pan[3] = p2_1_1.y;
            int t0_1_2 = 32 + 8 * th + 4;
            int t1_1_2 = 32 + 8 * th + 4 + 1;
            float r_168[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_168[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_2 = r_168[0];
            float r_169[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_169[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_2 = r_169[0];
            unsigned int r_170[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_170[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_2 = __uint_as_float(r_170[0] << 16);
            unsigned int r_171[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_171[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_2 = __uint_as_float(r_171[0] << 16);
            float2 _f2_59 = make_float2(g_t0_1_2, g_t1_1_2);
            float2 dc2_1_2 = sub_f32x2(_f2_55, _f2_59);
            float e2_2 = approx_exp2(dc2_1_2.x);
            float e2_2_2 = approx_exp2(dc2_1_2.y);
            float2 dn2_1_2 = sub_f32x2(_f2_56, _f2_59);
            float en_2 = approx_exp2(dn2_1_2.x);
            float en_2_2 = approx_exp2(dn2_1_2.y);
            float2 _f32x2_mul_a_176 = make_float2(k_t0_1_2, k_t1_1_2);
            float2 _f32x2_mul_b_177 = make_float2(e2_2, e2_2_2);
            float2 p2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
            float dl_3 = __uint_as_float(__as_u32(p2_1_2.x) & 4294959104u) - p2_1_2.x;
            float _rcp_178 = approx_rcp(k_t0_1_2);
            float ft_3 = ((dl_3 == 0.0f) ? e2_2 : __fmaf_rn(dl_3, _rcp_178, e2_2));
            float dl_4 = __uint_as_float(__as_u32(p2_1_2.y) & 4294959104u) - p2_1_2.y;
            float _rcp_179 = approx_rcp(k_t1_1_2);
            float ft_4 = ((dl_4 == 0.0f) ? e2_2_2 : __fmaf_rn(dl_4, _rcp_179, e2_2_2));
            dkt[4] = __fmaf_rn(_tmem_load_1[4], en_2, _tmem_load_0[4] * ft_3);
            dkt[5] = __fmaf_rn(_tmem_load_1[5], en_2_2, _tmem_load_0[5] * ft_4);
            pan[4] = p2_1_2.x;
            pan[5] = p2_1_2.y;
            int t0_1_3 = 32 + 8 * th + 6;
            int t1_1_3 = 32 + 8 * th + 6 + 1;
            float r_172[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_172[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_3 = r_172[0];
            float r_173[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_173[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_3 = r_173[0];
            unsigned int r_174[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_174[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_3 = __uint_as_float(r_174[0] << 16);
            unsigned int r_175[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_175[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_3 = __uint_as_float(r_175[0] << 16);
            float2 _f2_60 = make_float2(g_t0_1_3, g_t1_1_3);
            float2 dc2_1_3 = sub_f32x2(_f2_55, _f2_60);
            float e2_3 = approx_exp2(dc2_1_3.x);
            float e2_2_3 = approx_exp2(dc2_1_3.y);
            float2 dn2_1_3 = sub_f32x2(_f2_56, _f2_60);
            float en_3 = approx_exp2(dn2_1_3.x);
            float en_2_3 = approx_exp2(dn2_1_3.y);
            float2 _f32x2_mul_a_180 = make_float2(k_t0_1_3, k_t1_1_3);
            float2 _f32x2_mul_b_181 = make_float2(e2_3, e2_2_3);
            float2 p2_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&_f32x2_mul_b_181));
            float dl_5 = __uint_as_float(__as_u32(p2_1_3.x) & 4294959104u) - p2_1_3.x;
            float _rcp_182 = approx_rcp(k_t0_1_3);
            float ft_5 = ((dl_5 == 0.0f) ? e2_3 : __fmaf_rn(dl_5, _rcp_182, e2_3));
            float dl_6 = __uint_as_float(__as_u32(p2_1_3.y) & 4294959104u) - p2_1_3.y;
            float _rcp_183 = approx_rcp(k_t1_1_3);
            float ft_6 = ((dl_6 == 0.0f) ? e2_2_3 : __fmaf_rn(dl_6, _rcp_183, e2_2_3));
            dkt[6] = __fmaf_rn(_tmem_load_1[6], en_3, _tmem_load_0[6] * ft_5);
            dkt[7] = __fmaf_rn(_tmem_load_1[7], en_2_3, _tmem_load_0[7] * ft_6);
            pan[6] = p2_1_3.x;
            pan[7] = p2_1_3.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th), dkt);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan[0]), "f"(pan[1]), "f"(pan[2]), "f"(pan[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan[4]), "f"(pan[5]), "f"(pan[6]), "f"(pan[7]) : "memory");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (48 + 8 * th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_176[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_176[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_637 = r_176[0];
            float r_177[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_177[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_639 = r_177[0] - 21.640425f;
            float dkt_640[8];
            float pan_641[8];
            float2 _f2_61 = make_float2(gc_639, gc_639);
            float2 _f2_62 = make_float2(gnl_637, gnl_637);
            int t0_0 = 48 + 8 * th;
            int t1_0 = 48 + 8 * th + 1;
            float r_178[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_178[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_0 = r_178[0];
            float r_179[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_179[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_0 = r_179[0];
            unsigned int r_180[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_180[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_0 = __uint_as_float(r_180[0] << 16);
            unsigned int r_181[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_181[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_0 = __uint_as_float(r_181[0] << 16);
            float2 dc2_0 = sub_f32x2(_f2_61, make_float2(g_t0_0, g_t1_0));
            float e2_0_0 = approx_exp2(dc2_0.x);
            float e2_1_0 = approx_exp2(dc2_0.y);
            float2 _f32x2_mul_a_184 = make_float2(k_t0_0, k_t1_0);
            float2 _f32x2_mul_b_185 = make_float2(e2_0_0, e2_1_0);
            float2 p2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&_f32x2_mul_b_185));
            float dl_7 = __uint_as_float(__as_u32(p2_0.x) & 4294959104u) - p2_0.x;
            float _rcp_186 = approx_rcp(k_t0_0);
            float ft_7 = ((dl_7 == 0.0f) ? e2_0_0 : __fmaf_rn(dl_7, _rcp_186, e2_0_0));
            float dl_8 = __uint_as_float(__as_u32(p2_0.y) & 4294959104u) - p2_0.y;
            float _rcp_187 = approx_rcp(k_t1_0);
            float ft_8 = ((dl_8 == 0.0f) ? e2_1_0 : __fmaf_rn(dl_8, _rcp_187, e2_1_0));
            dkt_640[0] = _tmem_load_2[0] * ft_7;
            dkt_640[1] = _tmem_load_2[1] * ft_8;
            pan_641[0] = p2_0.x;
            pan_641[1] = p2_0.y;
            int t0_1 = 48 + 8 * th + 2;
            int t1_1 = 48 + 8 * th + 2 + 1;
            float r_182[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_182[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1 = r_182[0];
            float r_183[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_183[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1 = r_183[0];
            unsigned int r_184[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_184[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1 = __uint_as_float(r_184[0] << 16);
            unsigned int r_185[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_185[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1 = __uint_as_float(r_185[0] << 16);
            float2 dc2_1 = sub_f32x2(_f2_61, make_float2(g_t0_1, g_t1_1));
            float e2_0_1 = approx_exp2(dc2_1.x);
            float e2_1_1 = approx_exp2(dc2_1.y);
            float2 _f32x2_mul_a_188 = make_float2(k_t0_1, k_t1_1);
            float2 _f32x2_mul_b_189 = make_float2(e2_0_1, e2_1_1);
            float2 p2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
            float dl_9 = __uint_as_float(__as_u32(p2_1.x) & 4294959104u) - p2_1.x;
            float _rcp_190 = approx_rcp(k_t0_1);
            float ft_9 = ((dl_9 == 0.0f) ? e2_0_1 : __fmaf_rn(dl_9, _rcp_190, e2_0_1));
            float dl_10 = __uint_as_float(__as_u32(p2_1.y) & 4294959104u) - p2_1.y;
            float _rcp_191 = approx_rcp(k_t1_1);
            float ft_10 = ((dl_10 == 0.0f) ? e2_1_1 : __fmaf_rn(dl_10, _rcp_191, e2_1_1));
            dkt_640[2] = _tmem_load_2[2] * ft_9;
            dkt_640[3] = _tmem_load_2[3] * ft_10;
            pan_641[2] = p2_1.x;
            pan_641[3] = p2_1.y;
            int t0_2 = 48 + 8 * th + 4;
            int t1_2 = 48 + 8 * th + 4 + 1;
            float r_186[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_186[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_2 = r_186[0];
            float r_187[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_187[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_2 = r_187[0];
            unsigned int r_188[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_188[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_2 = __uint_as_float(r_188[0] << 16);
            unsigned int r_189[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_189[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_2 = __uint_as_float(r_189[0] << 16);
            float2 dc2_2 = sub_f32x2(_f2_61, make_float2(g_t0_2, g_t1_2));
            float e2_0_2 = approx_exp2(dc2_2.x);
            float e2_1_2 = approx_exp2(dc2_2.y);
            float2 _f32x2_mul_a_192 = make_float2(k_t0_2, k_t1_2);
            float2 _f32x2_mul_b_193 = make_float2(e2_0_2, e2_1_2);
            float2 p2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&_f32x2_mul_b_193));
            float dl_11 = __uint_as_float(__as_u32(p2_2.x) & 4294959104u) - p2_2.x;
            float _rcp_194 = approx_rcp(k_t0_2);
            float ft_11 = ((dl_11 == 0.0f) ? e2_0_2 : __fmaf_rn(dl_11, _rcp_194, e2_0_2));
            float dl_12 = __uint_as_float(__as_u32(p2_2.y) & 4294959104u) - p2_2.y;
            float _rcp_195 = approx_rcp(k_t1_2);
            float ft_12 = ((dl_12 == 0.0f) ? e2_1_2 : __fmaf_rn(dl_12, _rcp_195, e2_1_2));
            dkt_640[4] = _tmem_load_2[4] * ft_11;
            dkt_640[5] = _tmem_load_2[5] * ft_12;
            pan_641[4] = p2_2.x;
            pan_641[5] = p2_2.y;
            int t0_3 = 48 + 8 * th + 6;
            int t1_3 = 48 + 8 * th + 6 + 1;
            float r_190[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_190[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_3 = r_190[0];
            float r_191[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_191[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_3 = r_191[0];
            unsigned int r_192[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_192[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_3 = __uint_as_float(r_192[0] << 16);
            unsigned int r_193[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_193[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_3 = __uint_as_float(r_193[0] << 16);
            float2 dc2_3 = sub_f32x2(_f2_61, make_float2(g_t0_3, g_t1_3));
            float e2_0_3 = approx_exp2(dc2_3.x);
            float e2_1_3 = approx_exp2(dc2_3.y);
            float2 _f32x2_mul_a_196 = make_float2(k_t0_3, k_t1_3);
            float2 _f32x2_mul_b_197 = make_float2(e2_0_3, e2_1_3);
            float2 p2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
            float dl_13 = __uint_as_float(__as_u32(p2_3.x) & 4294959104u) - p2_3.x;
            float _rcp_198 = approx_rcp(k_t0_3);
            float ft_13 = ((dl_13 == 0.0f) ? e2_0_3 : __fmaf_rn(dl_13, _rcp_198, e2_0_3));
            float dl_14 = __uint_as_float(__as_u32(p2_3.y) & 4294959104u) - p2_3.y;
            float _rcp_199 = approx_rcp(k_t1_3);
            float ft_14 = ((dl_14 == 0.0f) ? e2_1_3 : __fmaf_rn(dl_14, _rcp_199, e2_1_3));
            dkt_640[6] = _tmem_load_2[6] * ft_13;
            dkt_640[7] = _tmem_load_2[7] * ft_14;
            pan_641[6] = p2_3.x;
            pan_641[7] = p2_3.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th), dkt_640);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_641[0]), "f"(pan_641[1]), "f"(pan_641[2]), "f"(pan_641[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_641[4]), "f"(pan_641[5]), "f"(pan_641[6]), "f"(pan_641[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + (unsigned int)row_base + (unsigned int)(64 + 8 * th));
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_194[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_194[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_699 = r_194[0];
            float r_195[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_195[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_701 = r_195[0] - 21.640425f;
            float dkt_702[8];
            float pan_703[8];
            float2 _f2_63 = make_float2(gc_701, gc_701);
            float2 _f2_64 = make_float2(gnl_699, gnl_699);
            int t0_4 = 8 * th;
            int t1_4 = 8 * th + 1;
            float r_196[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_196[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_4 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_4 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_4 = r_196[0];
            float r_197[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_197[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_4 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_4 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_4 = r_197[0];
            unsigned int r_198[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_198[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_4 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_4 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_4 = __uint_as_float(r_198[0] << 16);
            unsigned int r_199[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_199[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_4 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_4 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_4 = __uint_as_float(r_199[0] << 16);
            float2 _f2_65 = make_float2(g_t0_4, g_t1_4);
            float2 dc2_4 = sub_f32x2(_f2_63, _f2_65);
            float e2_0_4 = approx_exp2(dc2_4.x);
            float e2_1_4 = approx_exp2(dc2_4.y);
            float2 dn2_0 = sub_f32x2(_f2_64, _f2_65);
            float en_0_0 = approx_exp2(dn2_0.x);
            float en_1_0 = approx_exp2(dn2_0.y);
            float2 _f32x2_mul_a_200 = make_float2(k_t0_4, k_t1_4);
            float2 _f32x2_mul_b_201 = make_float2(e2_0_4, e2_1_4);
            float2 p2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
            float dl_15 = __uint_as_float(__as_u32(p2_4.x) & 4294959104u) - p2_4.x;
            float _rcp_202 = approx_rcp(k_t0_4);
            float ft_15 = ((dl_15 == 0.0f) ? e2_0_4 : __fmaf_rn(dl_15, _rcp_202, e2_0_4));
            float dl_16 = __uint_as_float(__as_u32(p2_4.y) & 4294959104u) - p2_4.y;
            float _rcp_203 = approx_rcp(k_t1_4);
            float ft_16 = ((dl_16 == 0.0f) ? e2_1_4 : __fmaf_rn(dl_16, _rcp_203, e2_1_4));
            dkt_702[0] = __fmaf_rn(_tmem_load_4[0], en_0_0, _tmem_load_3[0] * ft_15);
            dkt_702[1] = __fmaf_rn(_tmem_load_4[1], en_1_0, _tmem_load_3[1] * ft_16);
            pan_703[0] = p2_4.x;
            pan_703[1] = p2_4.y;
            int t0_5 = 8 * th + 2;
            int t1_5 = 8 * th + 2 + 1;
            float r_200[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_200[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_5 = r_200[0];
            float r_201[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_201[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_5 = r_201[0];
            unsigned int r_202[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_202[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_5 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_5 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_5 = __uint_as_float(r_202[0] << 16);
            unsigned int r_203[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_203[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_5 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_5 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_5 = __uint_as_float(r_203[0] << 16);
            float2 _f2_66 = make_float2(g_t0_5, g_t1_5);
            float2 dc2_5 = sub_f32x2(_f2_63, _f2_66);
            float e2_0_5 = approx_exp2(dc2_5.x);
            float e2_1_5 = approx_exp2(dc2_5.y);
            float2 dn2_1 = sub_f32x2(_f2_64, _f2_66);
            float en_0_1 = approx_exp2(dn2_1.x);
            float en_1_1 = approx_exp2(dn2_1.y);
            float2 _f32x2_mul_a_204 = make_float2(k_t0_5, k_t1_5);
            float2 _f32x2_mul_b_205 = make_float2(e2_0_5, e2_1_5);
            float2 p2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
            float dl_17 = __uint_as_float(__as_u32(p2_5.x) & 4294959104u) - p2_5.x;
            float _rcp_206 = approx_rcp(k_t0_5);
            float ft_17 = ((dl_17 == 0.0f) ? e2_0_5 : __fmaf_rn(dl_17, _rcp_206, e2_0_5));
            float dl_18 = __uint_as_float(__as_u32(p2_5.y) & 4294959104u) - p2_5.y;
            float _rcp_207 = approx_rcp(k_t1_5);
            float ft_18 = ((dl_18 == 0.0f) ? e2_1_5 : __fmaf_rn(dl_18, _rcp_207, e2_1_5));
            dkt_702[2] = __fmaf_rn(_tmem_load_4[2], en_0_1, _tmem_load_3[2] * ft_17);
            dkt_702[3] = __fmaf_rn(_tmem_load_4[3], en_1_1, _tmem_load_3[3] * ft_18);
            pan_703[2] = p2_5.x;
            pan_703[3] = p2_5.y;
            int t0_6 = 8 * th + 4;
            int t1_6 = 8 * th + 4 + 1;
            float r_204[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_204[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_6 = r_204[0];
            float r_205[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_205[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_6 = r_205[0];
            unsigned int r_206[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_206[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_6 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_6 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_6 = __uint_as_float(r_206[0] << 16);
            unsigned int r_207[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_207[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_6 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_6 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_6 = __uint_as_float(r_207[0] << 16);
            float2 _f2_67 = make_float2(g_t0_6, g_t1_6);
            float2 dc2_6 = sub_f32x2(_f2_63, _f2_67);
            float e2_0_6 = approx_exp2(dc2_6.x);
            float e2_1_6 = approx_exp2(dc2_6.y);
            float2 dn2_2 = sub_f32x2(_f2_64, _f2_67);
            float en_0_2 = approx_exp2(dn2_2.x);
            float en_1_2 = approx_exp2(dn2_2.y);
            float2 _f32x2_mul_a_208 = make_float2(k_t0_6, k_t1_6);
            float2 _f32x2_mul_b_209 = make_float2(e2_0_6, e2_1_6);
            float2 p2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
            float dl_19 = __uint_as_float(__as_u32(p2_6.x) & 4294959104u) - p2_6.x;
            float _rcp_210 = approx_rcp(k_t0_6);
            float ft_19 = ((dl_19 == 0.0f) ? e2_0_6 : __fmaf_rn(dl_19, _rcp_210, e2_0_6));
            float dl_20 = __uint_as_float(__as_u32(p2_6.y) & 4294959104u) - p2_6.y;
            float _rcp_211 = approx_rcp(k_t1_6);
            float ft_20 = ((dl_20 == 0.0f) ? e2_1_6 : __fmaf_rn(dl_20, _rcp_211, e2_1_6));
            dkt_702[4] = __fmaf_rn(_tmem_load_4[4], en_0_2, _tmem_load_3[4] * ft_19);
            dkt_702[5] = __fmaf_rn(_tmem_load_4[5], en_1_2, _tmem_load_3[5] * ft_20);
            pan_703[4] = p2_6.x;
            pan_703[5] = p2_6.y;
            int t0_7 = 8 * th + 6;
            int t1_7 = 8 * th + 6 + 1;
            float r_208[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_208[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_7 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_7 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_7 = r_208[0];
            float r_209[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_209[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_7 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_7 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_7 = r_209[0];
            unsigned int r_210[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_210[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_7 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_7 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_7 = __uint_as_float(r_210[0] << 16);
            unsigned int r_211[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_211[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_7 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_7 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_7 = __uint_as_float(r_211[0] << 16);
            float2 _f2_68 = make_float2(g_t0_7, g_t1_7);
            float2 dc2_7 = sub_f32x2(_f2_63, _f2_68);
            float e2_0_7 = approx_exp2(dc2_7.x);
            float e2_1_7 = approx_exp2(dc2_7.y);
            float2 dn2_3 = sub_f32x2(_f2_64, _f2_68);
            float en_0_3 = approx_exp2(dn2_3.x);
            float en_1_3 = approx_exp2(dn2_3.y);
            float2 _f32x2_mul_a_212 = make_float2(k_t0_7, k_t1_7);
            float2 _f32x2_mul_b_213 = make_float2(e2_0_7, e2_1_7);
            float2 p2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
            float dl_21 = __uint_as_float(__as_u32(p2_7.x) & 4294959104u) - p2_7.x;
            float _rcp_214 = approx_rcp(k_t0_7);
            float ft_21 = ((dl_21 == 0.0f) ? e2_0_7 : __fmaf_rn(dl_21, _rcp_214, e2_0_7));
            float dl_22 = __uint_as_float(__as_u32(p2_7.y) & 4294959104u) - p2_7.y;
            float _rcp_215 = approx_rcp(k_t1_7);
            float ft_22 = ((dl_22 == 0.0f) ? e2_1_7 : __fmaf_rn(dl_22, _rcp_215, e2_1_7));
            dkt_702[6] = __fmaf_rn(_tmem_load_4[6], en_0_3, _tmem_load_3[6] * ft_21);
            dkt_702[7] = __fmaf_rn(_tmem_load_4[7], en_1_3, _tmem_load_3[7] * ft_22);
            pan_703[6] = p2_7.x;
            pan_703[7] = p2_7.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(8 * th), dkt_702);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[0]), "f"(pan_703[1]), "f"(pan_703[2]), "f"(pan_703[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[4]), "f"(pan_703[5]), "f"(pan_703[6]), "f"(pan_703[7]) : "memory");
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (16 + 8 * th)));
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_212[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_212[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_773 = r_212[0];
            float r_213[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_213[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_775 = r_213[0] - 21.640425f;
            float dkt_776[8];
            float pan_777[8];
            float2 _f2_69 = make_float2(gc_775, gc_775);
            float2 _f2_70 = make_float2(gnl_773, gnl_773);
            int t0_8 = 16 + 8 * th;
            int t1_8 = 16 + 8 * th + 1;
            float r_214[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_214[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_8 = r_214[0];
            float r_215[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_215[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_8 = r_215[0];
            unsigned int r_216[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_216[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_8 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_8 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_8 = __uint_as_float(r_216[0] << 16);
            unsigned int r_217[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_217[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_8 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_8 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_8 = __uint_as_float(r_217[0] << 16);
            float2 _f2_71 = make_float2(g_t0_8, g_t1_8);
            float2 dc2_8 = sub_f32x2(_f2_69, _f2_71);
            float e2_0_8 = approx_exp2(dc2_8.x);
            float e2_1_8 = approx_exp2(dc2_8.y);
            float2 dn2_4 = sub_f32x2(_f2_70, _f2_71);
            float en_0_4 = approx_exp2(dn2_4.x);
            float en_1_4 = approx_exp2(dn2_4.y);
            float2 _f32x2_mul_a_216 = make_float2(k_t0_8, k_t1_8);
            float2 _f32x2_mul_b_217 = make_float2(e2_0_8, e2_1_8);
            float2 p2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
            float dl_23 = __uint_as_float(__as_u32(p2_8.x) & 4294959104u) - p2_8.x;
            float _rcp_218 = approx_rcp(k_t0_8);
            float ft_23 = ((dl_23 == 0.0f) ? e2_0_8 : __fmaf_rn(dl_23, _rcp_218, e2_0_8));
            float dl_24 = __uint_as_float(__as_u32(p2_8.y) & 4294959104u) - p2_8.y;
            float _rcp_219 = approx_rcp(k_t1_8);
            float ft_24 = ((dl_24 == 0.0f) ? e2_1_8 : __fmaf_rn(dl_24, _rcp_219, e2_1_8));
            dkt_776[0] = __fmaf_rn(_tmem_load_6[0], en_0_4, _tmem_load_5[0] * ft_23);
            dkt_776[1] = __fmaf_rn(_tmem_load_6[1], en_1_4, _tmem_load_5[1] * ft_24);
            pan_777[0] = p2_8.x;
            pan_777[1] = p2_8.y;
            int t0_9 = 16 + 8 * th + 2;
            int t1_9 = 16 + 8 * th + 2 + 1;
            float r_218[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_218[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_9 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_9 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_9 = r_218[0];
            float r_219[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_219[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_9 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_9 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_9 = r_219[0];
            unsigned int r_220[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_220[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_9 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_9 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_9 = __uint_as_float(r_220[0] << 16);
            unsigned int r_221[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_221[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_9 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_9 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_9 = __uint_as_float(r_221[0] << 16);
            float2 _f2_72 = make_float2(g_t0_9, g_t1_9);
            float2 dc2_9 = sub_f32x2(_f2_69, _f2_72);
            float e2_0_9 = approx_exp2(dc2_9.x);
            float e2_1_9 = approx_exp2(dc2_9.y);
            float2 dn2_5 = sub_f32x2(_f2_70, _f2_72);
            float en_0_5 = approx_exp2(dn2_5.x);
            float en_1_5 = approx_exp2(dn2_5.y);
            float2 _f32x2_mul_a_220 = make_float2(k_t0_9, k_t1_9);
            float2 _f32x2_mul_b_221 = make_float2(e2_0_9, e2_1_9);
            float2 p2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
            float dl_25 = __uint_as_float(__as_u32(p2_9.x) & 4294959104u) - p2_9.x;
            float _rcp_222 = approx_rcp(k_t0_9);
            float ft_25 = ((dl_25 == 0.0f) ? e2_0_9 : __fmaf_rn(dl_25, _rcp_222, e2_0_9));
            float dl_26 = __uint_as_float(__as_u32(p2_9.y) & 4294959104u) - p2_9.y;
            float _rcp_223 = approx_rcp(k_t1_9);
            float ft_26 = ((dl_26 == 0.0f) ? e2_1_9 : __fmaf_rn(dl_26, _rcp_223, e2_1_9));
            dkt_776[2] = __fmaf_rn(_tmem_load_6[2], en_0_5, _tmem_load_5[2] * ft_25);
            dkt_776[3] = __fmaf_rn(_tmem_load_6[3], en_1_5, _tmem_load_5[3] * ft_26);
            pan_777[2] = p2_9.x;
            pan_777[3] = p2_9.y;
            int t0_10 = 16 + 8 * th + 4;
            int t1_10 = 16 + 8 * th + 4 + 1;
            float r_222[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_222[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_10 = r_222[0];
            float r_223[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_223[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_10 = r_223[0];
            unsigned int r_224[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_224[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_10 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_10 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_10 = __uint_as_float(r_224[0] << 16);
            unsigned int r_225[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_225[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_10 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_10 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_10 = __uint_as_float(r_225[0] << 16);
            float2 _f2_73 = make_float2(g_t0_10, g_t1_10);
            float2 dc2_10 = sub_f32x2(_f2_69, _f2_73);
            float e2_0_10 = approx_exp2(dc2_10.x);
            float e2_1_10 = approx_exp2(dc2_10.y);
            float2 dn2_6 = sub_f32x2(_f2_70, _f2_73);
            float en_0_6 = approx_exp2(dn2_6.x);
            float en_1_6 = approx_exp2(dn2_6.y);
            float2 _f32x2_mul_a_224 = make_float2(k_t0_10, k_t1_10);
            float2 _f32x2_mul_b_225 = make_float2(e2_0_10, e2_1_10);
            float2 p2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
            float dl_27 = __uint_as_float(__as_u32(p2_10.x) & 4294959104u) - p2_10.x;
            float _rcp_226 = approx_rcp(k_t0_10);
            float ft_27 = ((dl_27 == 0.0f) ? e2_0_10 : __fmaf_rn(dl_27, _rcp_226, e2_0_10));
            float dl_28 = __uint_as_float(__as_u32(p2_10.y) & 4294959104u) - p2_10.y;
            float _rcp_227 = approx_rcp(k_t1_10);
            float ft_28 = ((dl_28 == 0.0f) ? e2_1_10 : __fmaf_rn(dl_28, _rcp_227, e2_1_10));
            dkt_776[4] = __fmaf_rn(_tmem_load_6[4], en_0_6, _tmem_load_5[4] * ft_27);
            dkt_776[5] = __fmaf_rn(_tmem_load_6[5], en_1_6, _tmem_load_5[5] * ft_28);
            pan_777[4] = p2_10.x;
            pan_777[5] = p2_10.y;
            int t0_11 = 16 + 8 * th + 6;
            int t1_11 = 16 + 8 * th + 6 + 1;
            float r_226[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_226[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_11 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_11 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_11 = r_226[0];
            float r_227[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_227[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_11 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_11 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_11 = r_227[0];
            unsigned int r_228[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_228[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_11 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_11 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_11 = __uint_as_float(r_228[0] << 16);
            unsigned int r_229[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_229[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_11 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_11 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_11 = __uint_as_float(r_229[0] << 16);
            float2 _f2_74 = make_float2(g_t0_11, g_t1_11);
            float2 dc2_11 = sub_f32x2(_f2_69, _f2_74);
            float e2_0_11 = approx_exp2(dc2_11.x);
            float e2_1_11 = approx_exp2(dc2_11.y);
            float2 dn2_7 = sub_f32x2(_f2_70, _f2_74);
            float en_0_7 = approx_exp2(dn2_7.x);
            float en_1_7 = approx_exp2(dn2_7.y);
            float2 _f32x2_mul_a_228 = make_float2(k_t0_11, k_t1_11);
            float2 _f32x2_mul_b_229 = make_float2(e2_0_11, e2_1_11);
            float2 p2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
            float dl_29 = __uint_as_float(__as_u32(p2_11.x) & 4294959104u) - p2_11.x;
            float _rcp_230 = approx_rcp(k_t0_11);
            float ft_29 = ((dl_29 == 0.0f) ? e2_0_11 : __fmaf_rn(dl_29, _rcp_230, e2_0_11));
            float dl_30 = __uint_as_float(__as_u32(p2_11.y) & 4294959104u) - p2_11.y;
            float _rcp_231 = approx_rcp(k_t1_11);
            float ft_30 = ((dl_30 == 0.0f) ? e2_1_11 : __fmaf_rn(dl_30, _rcp_231, e2_1_11));
            dkt_776[6] = __fmaf_rn(_tmem_load_6[6], en_0_7, _tmem_load_5[6] * ft_29);
            dkt_776[7] = __fmaf_rn(_tmem_load_6[7], en_1_7, _tmem_load_5[7] * ft_30);
            pan_777[6] = p2_11.x;
            pan_777[7] = p2_11.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th), dkt_776);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[0]), "f"(pan_777[1]), "f"(pan_777[2]), "f"(pan_777[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[4]), "f"(pan_777[5]), "f"(pan_777[6]), "f"(pan_777[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_538_0 = 8 * th;
            int gbase_539_0 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_538_0 * 128);
            int kbase_540_0 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_538_0 * 128);
            float r_230[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_230[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_542_0 = r_230[0];
            float a1_543_0 = 0.0f;
            float gs_544_0[8];
            float ks_545_0[8];
            float r_231[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_231[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[0] = r_231[0];
            unsigned int r_232[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_232[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[0] = __uint_as_float(r_232[0] << 16);
            float r_233[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_233[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[1] = r_233[0];
            unsigned int r_234[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_234[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[1] = __uint_as_float(r_234[0] << 16);
            float r_235[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_235[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[2] = r_235[0];
            unsigned int r_236[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_236[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[2] = __uint_as_float(r_236[0] << 16);
            float r_237[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_237[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[3] = r_237[0];
            unsigned int r_238[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_238[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[3] = __uint_as_float(r_238[0] << 16);
            float r_239[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_239[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[4] = r_239[0];
            unsigned int r_240[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_240[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[4] = __uint_as_float(r_240[0] << 16);
            float r_241[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_241[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[5] = r_241[0];
            unsigned int r_242[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_242[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[5] = __uint_as_float(r_242[0] << 16);
            float r_243[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_243[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[6] = r_243[0];
            unsigned int r_244[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_244[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[6] = __uint_as_float(r_244[0] << 16);
            float r_245[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_245[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[7] = r_245[0];
            unsigned int r_246[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_246[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[7] = __uint_as_float(r_246[0] << 16);
            float2 _f2_54_0 = make_float2(a0_542_0, a0_542_0);
            float v_562_0[8];
            float2 d2_56 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[0], gs_544_0[1]));
            float e0_12 = approx_exp2(d2_56.x);
            float e1_12 = approx_exp2(d2_56.y);
            float2 _f32x2_mul_a_232 = make_float2(ks_545_0[0], ks_545_0[1]);
            float2 _f32x2_mul_b_233 = make_float2(e0_12, e1_12);
            float2 v2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&_f32x2_mul_b_233));
            v_562_0[0] = v2_12.x;
            v_562_0[1] = v2_12.y;
            float2 d2_57 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[2], gs_544_0[3]));
            float e0_13 = approx_exp2(d2_57.x);
            float e1_13 = approx_exp2(d2_57.y);
            float2 _f32x2_mul_a_234 = make_float2(ks_545_0[2], ks_545_0[3]);
            float2 _f32x2_mul_b_235 = make_float2(e0_13, e1_13);
            float2 v2_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
            v_562_0[2] = v2_13.x;
            v_562_0[3] = v2_13.y;
            float2 d2_58 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[4], gs_544_0[5]));
            float e0_14 = approx_exp2(d2_58.x);
            float e1_14 = approx_exp2(d2_58.y);
            float2 _f32x2_mul_a_236 = make_float2(ks_545_0[4], ks_545_0[5]);
            float2 _f32x2_mul_b_237 = make_float2(e0_14, e1_14);
            float2 v2_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&_f32x2_mul_b_237));
            v_562_0[4] = v2_14.x;
            v_562_0[5] = v2_14.y;
            float2 d2_59 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[6], gs_544_0[7]));
            float e0_15 = approx_exp2(d2_59.x);
            float e1_15 = approx_exp2(d2_59.y);
            float2 _f32x2_mul_a_238 = make_float2(ks_545_0[6], ks_545_0[7]);
            float2 _f32x2_mul_b_239 = make_float2(e0_15, e1_15);
            float2 v2_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
            v_562_0[6] = v2_15.x;
            v_562_0[7] = v2_15.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_562_0[0]), "f"(v_562_0[1]), "f"(v_562_0[2]), "f"(v_562_0[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_562_0[4]), "f"(v_562_0[5]), "f"(v_562_0[6]), "f"(v_562_0[7]) : "memory");
            mbarrier_wait(blk_full_addr, 0);
            mbarrier_wait(blk_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                int kk_3_0 = w * 4;
                if (lane_1 > (kk_3_0 & 15)) {
                    {
                        uint32_t _addr_240 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_3_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_3_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_240), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_1 = w * 4 + 1;
                if (lane_1 > (kk_3_1 & 15)) {
                    {
                        uint32_t _addr_241 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_3_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_3_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_241), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_2 = w * 4 + 2;
                if (lane_1 > (kk_3_2 & 15)) {
                    {
                        uint32_t _addr_242 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_3_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_3_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_242), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_3 = w * 4 + 3;
                if (lane_1 > (kk_3_3 & 15)) {
                    {
                        uint32_t _addr_243 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_3_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_3_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_243), "f"(0.0f) : "memory");
                    }
                }
            }
            if (lane_1 < 16) {
                int kk_3_4 = w * 4;
                if (lane_1 > (kk_3_4 & 15)) {
                    {
                        uint32_t _addr_244 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_244), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_5 = w * 4 + 1;
                if (lane_1 > (kk_3_5 & 15)) {
                    {
                        uint32_t _addr_245 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3_5 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3_5 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_245), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_6 = w * 4 + 2;
                if (lane_1 > (kk_3_6 & 15)) {
                    {
                        uint32_t _addr_246 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3_6 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3_6 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_246), "f"(0.0f) : "memory");
                    }
                }
                int kk_3_7 = w * 4 + 3;
                if (lane_1 > (kk_3_7 & 15)) {
                    {
                        uint32_t _addr_247 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3_7 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3_7 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_247), "f"(0.0f) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(fold_done_addr);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(p1a_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + (th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_247[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_247[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn = r_247[0];
            float gc_888 = gn - 21.640425f;
            int eb = 98304 + th * -32768;
            int ew = eb >> 2;
            int qbase_889 = pool_f32_addr + (unsigned int)(eb + 12288 + dh_lane * 1024);
            int sb8 = 32 + 8 * th;
            int gb8 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8 * 128);
            int kb8 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8 * 128);
            float dbp[8];
            float _tmem_load_7[8];
            tmem_ld_x8(&_tmem_load_7[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            float _tmem_load_8[8];
            tmem_ld_x8(&_tmem_load_8[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th));
            float _tmem_load_9[8];
            tmem_ld_x8(&_tmem_load_9[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th));
            float _tmem_load_10[8];
            tmem_ld_x8(&_tmem_load_10[0], taddr + (unsigned int)row_base + (unsigned int)(208 + 8 * th));
            float _tmem_load_11[8];
            tmem_ld_x8(&_tmem_load_11[0], taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th));
            float btv[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (32 + 8 * th))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv[4])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (32 + 8 * th) + 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_run[1];
            s_run[0] = 0.0f;
            int t_1_0 = 32 + 8 * th + 7;
            float r_248[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_248[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_0 = r_248[0];
            unsigned int r_249[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_249[0])) : "r"((kb8 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_0 = __uint_as_float(r_249[0] << 16);
            unsigned int r_250[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_250[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_0 = __uint_as_float(r_250[0] << 16);
            float q_t_2_0 = ((t_1_0 >= clen) ? 0.0f : q_t_1_0);
            float dqf_1_0 = pool_f32[ew + 896 + d_lane];
            float dkf_1_0 = pool_f32[ew + 1920 + d_lane];
            float dgf_1_0 = pool_f32[ew + 2944 + d_lane];
            float bt_t_1_0 = btv[7];
            float e_off_1_0 = approx_exp2(g_t_1_0 - gn);
            float e_dg_1_0 = approx_exp2(g_t_1_0 - gc_888);
            float dq2_1_0 = __fmaf_rn(_tmem_load_7[7], e_off_1_0, _tmem_load_9[7] * e_dg_1_0);
            float dk2_1_0 = __fmaf_rn(_tmem_load_8[7], e_off_1_0, _tmem_load_10[7] * e_dg_1_0);
            dbp[7] = dk2_1_0 * k_t_1_0;
            float dk2b_1_0 = dk2_1_0 * bt_t_1_0;
            float dq_o_1_0 = dq2_1_0 + dqf_1_0;
            float dk_o_1_0 = dk2b_1_0 + dkf_1_0 + _tmem_load_11[7];
            float qe = q_t_2_0 * e_dg_1_0;
            float kp = k_t_1_0 * bt_t_1_0;
            __nv_bfloat16 kb = (__nv_bfloat16)kp;
            float kbe = (float)kb * e_dg_1_0;
            float cr = __fmaf_rn(__uint_as_float(__as_u32(qe) & 4294959104u) - qe, _tmem_load_9[7], (__uint_as_float(__as_u32(kbe) & 4294959104u) - kp * e_dg_1_0) * _tmem_load_10[7]);
            float dg_o_1_0 = __fmaf_rn(q_t_2_0, dq2_1_0, __fmaf_rn(dk2b_1_0 - _tmem_load_11[7], k_t_1_0, cr)) + dgf_1_0;
            s_run[0] = s_run[0] + dg_o_1_0;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 7)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (896 + d_lane)] = dq_o_1_0;
            pool_f32[(98304 - 32768 * th >> 2) + (1920 + d_lane)] = dk_o_1_0;
            int t_1_1 = 32 + 8 * th + 6;
            float r_251[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_251[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_1 = r_251[0];
            unsigned int r_252[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_252[0])) : "r"((kb8 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_1 = __uint_as_float(r_252[0] << 16);
            unsigned int r_253[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_253[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_1 = __uint_as_float(r_253[0] << 16);
            float q_t_2_1 = ((t_1_1 >= clen) ? 0.0f : q_t_1_1);
            float dqf_1_1 = pool_f32[ew + 768 + d_lane];
            float dkf_1_1 = pool_f32[ew + 1792 + d_lane];
            float dgf_1_1 = pool_f32[ew + 2816 + d_lane];
            float bt_t_1_1 = btv[6];
            float e_off_1_1 = approx_exp2(g_t_1_1 - gn);
            float e_dg_1_1 = approx_exp2(g_t_1_1 - gc_888);
            float dq2_1_1 = __fmaf_rn(_tmem_load_7[6], e_off_1_1, _tmem_load_9[6] * e_dg_1_1);
            float dk2_1_1 = __fmaf_rn(_tmem_load_8[6], e_off_1_1, _tmem_load_10[6] * e_dg_1_1);
            dbp[6] = dk2_1_1 * k_t_1_1;
            float dk2b_1_1 = dk2_1_1 * bt_t_1_1;
            float dq_o_1_1 = dq2_1_1 + dqf_1_1;
            float dk_o_1_1 = dk2b_1_1 + dkf_1_1 + _tmem_load_11[6];
            float qe_0 = q_t_2_1 * e_dg_1_1;
            float kp_12 = k_t_1_1 * bt_t_1_1;
            __nv_bfloat16 kb_0 = (__nv_bfloat16)kp_12;
            float kbe_0 = (float)kb_0 * e_dg_1_1;
            float cr_0 = __fmaf_rn(__uint_as_float(__as_u32(qe_0) & 4294959104u) - qe_0, _tmem_load_9[6], (__uint_as_float(__as_u32(kbe_0) & 4294959104u) - kp_12 * e_dg_1_1) * _tmem_load_10[6]);
            float dg_o_1_1 = __fmaf_rn(q_t_2_1, dq2_1_1, __fmaf_rn(dk2b_1_1 - _tmem_load_11[6], k_t_1_1, cr_0)) + dgf_1_1;
            s_run[0] = s_run[0] + dg_o_1_1;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 6)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (768 + d_lane)] = dq_o_1_1;
            pool_f32[(98304 - 32768 * th >> 2) + (1792 + d_lane)] = dk_o_1_1;
            int t_1_2 = 32 + 8 * th + 5;
            float r_254[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_254[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_2 = r_254[0];
            unsigned int r_255[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_255[0])) : "r"((kb8 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_2 = __uint_as_float(r_255[0] << 16);
            unsigned int r_256[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_256[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_2 = __uint_as_float(r_256[0] << 16);
            float q_t_2_2 = ((t_1_2 >= clen) ? 0.0f : q_t_1_2);
            float dqf_1_2 = pool_f32[ew + 640 + d_lane];
            float dkf_1_2 = pool_f32[ew + 1664 + d_lane];
            float dgf_1_2 = pool_f32[ew + 2688 + d_lane];
            float bt_t_1_2 = btv[5];
            float e_off_1_2 = approx_exp2(g_t_1_2 - gn);
            float e_dg_1_2 = approx_exp2(g_t_1_2 - gc_888);
            float dq2_1_2 = __fmaf_rn(_tmem_load_7[5], e_off_1_2, _tmem_load_9[5] * e_dg_1_2);
            float dk2_1_2 = __fmaf_rn(_tmem_load_8[5], e_off_1_2, _tmem_load_10[5] * e_dg_1_2);
            dbp[5] = dk2_1_2 * k_t_1_2;
            float dk2b_1_2 = dk2_1_2 * bt_t_1_2;
            float dq_o_1_2 = dq2_1_2 + dqf_1_2;
            float dk_o_1_2 = dk2b_1_2 + dkf_1_2 + _tmem_load_11[5];
            float qe_1 = q_t_2_2 * e_dg_1_2;
            float kp_13 = k_t_1_2 * bt_t_1_2;
            __nv_bfloat16 kb_1 = (__nv_bfloat16)kp_13;
            float kbe_1 = (float)kb_1 * e_dg_1_2;
            float cr_1 = __fmaf_rn(__uint_as_float(__as_u32(qe_1) & 4294959104u) - qe_1, _tmem_load_9[5], (__uint_as_float(__as_u32(kbe_1) & 4294959104u) - kp_13 * e_dg_1_2) * _tmem_load_10[5]);
            float dg_o_1_2 = __fmaf_rn(q_t_2_2, dq2_1_2, __fmaf_rn(dk2b_1_2 - _tmem_load_11[5], k_t_1_2, cr_1)) + dgf_1_2;
            s_run[0] = s_run[0] + dg_o_1_2;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 5)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (640 + d_lane)] = dq_o_1_2;
            pool_f32[(98304 - 32768 * th >> 2) + (1664 + d_lane)] = dk_o_1_2;
            int t_1_3 = 32 + 8 * th + 4;
            float r_257[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_257[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_3 = r_257[0];
            unsigned int r_258[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_258[0])) : "r"((kb8 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_3 = __uint_as_float(r_258[0] << 16);
            unsigned int r_259[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_259[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_3 = __uint_as_float(r_259[0] << 16);
            float q_t_2_3 = ((t_1_3 >= clen) ? 0.0f : q_t_1_3);
            float dqf_1_3 = pool_f32[ew + 512 + d_lane];
            float dkf_1_3 = pool_f32[ew + 1536 + d_lane];
            float dgf_1_3 = pool_f32[ew + 2560 + d_lane];
            float bt_t_1_3 = btv[4];
            float e_off_1_3 = approx_exp2(g_t_1_3 - gn);
            float e_dg_1_3 = approx_exp2(g_t_1_3 - gc_888);
            float dq2_1_3 = __fmaf_rn(_tmem_load_7[4], e_off_1_3, _tmem_load_9[4] * e_dg_1_3);
            float dk2_1_3 = __fmaf_rn(_tmem_load_8[4], e_off_1_3, _tmem_load_10[4] * e_dg_1_3);
            dbp[4] = dk2_1_3 * k_t_1_3;
            float dk2b_1_3 = dk2_1_3 * bt_t_1_3;
            float dq_o_1_3 = dq2_1_3 + dqf_1_3;
            float dk_o_1_3 = dk2b_1_3 + dkf_1_3 + _tmem_load_11[4];
            float qe_2 = q_t_2_3 * e_dg_1_3;
            float kp_14 = k_t_1_3 * bt_t_1_3;
            __nv_bfloat16 kb_2 = (__nv_bfloat16)kp_14;
            float kbe_2 = (float)kb_2 * e_dg_1_3;
            float cr_2 = __fmaf_rn(__uint_as_float(__as_u32(qe_2) & 4294959104u) - qe_2, _tmem_load_9[4], (__uint_as_float(__as_u32(kbe_2) & 4294959104u) - kp_14 * e_dg_1_3) * _tmem_load_10[4]);
            float dg_o_1_3 = __fmaf_rn(q_t_2_3, dq2_1_3, __fmaf_rn(dk2b_1_3 - _tmem_load_11[4], k_t_1_3, cr_2)) + dgf_1_3;
            s_run[0] = s_run[0] + dg_o_1_3;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 4)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (512 + d_lane)] = dq_o_1_3;
            pool_f32[(98304 - 32768 * th >> 2) + (1536 + d_lane)] = dk_o_1_3;
            int t_1_4 = 32 + 8 * th + 3;
            float r_260[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_260[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_4 = r_260[0];
            unsigned int r_261[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_261[0])) : "r"((kb8 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_4 = __uint_as_float(r_261[0] << 16);
            unsigned int r_262[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_262[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_4 = __uint_as_float(r_262[0] << 16);
            float q_t_2_4 = ((t_1_4 >= clen) ? 0.0f : q_t_1_4);
            float dqf_1_4 = pool_f32[ew + 384 + d_lane];
            float dkf_1_4 = pool_f32[ew + 1408 + d_lane];
            float dgf_1_4 = pool_f32[ew + 2432 + d_lane];
            float bt_t_1_4 = btv[3];
            float e_off_1_4 = approx_exp2(g_t_1_4 - gn);
            float e_dg_1_4 = approx_exp2(g_t_1_4 - gc_888);
            float dq2_1_4 = __fmaf_rn(_tmem_load_7[3], e_off_1_4, _tmem_load_9[3] * e_dg_1_4);
            float dk2_1_4 = __fmaf_rn(_tmem_load_8[3], e_off_1_4, _tmem_load_10[3] * e_dg_1_4);
            dbp[3] = dk2_1_4 * k_t_1_4;
            float dk2b_1_4 = dk2_1_4 * bt_t_1_4;
            float dq_o_1_4 = dq2_1_4 + dqf_1_4;
            float dk_o_1_4 = dk2b_1_4 + dkf_1_4 + _tmem_load_11[3];
            float qe_3 = q_t_2_4 * e_dg_1_4;
            float kp_15 = k_t_1_4 * bt_t_1_4;
            __nv_bfloat16 kb_3 = (__nv_bfloat16)kp_15;
            float kbe_3 = (float)kb_3 * e_dg_1_4;
            float cr_3 = __fmaf_rn(__uint_as_float(__as_u32(qe_3) & 4294959104u) - qe_3, _tmem_load_9[3], (__uint_as_float(__as_u32(kbe_3) & 4294959104u) - kp_15 * e_dg_1_4) * _tmem_load_10[3]);
            float dg_o_1_4 = __fmaf_rn(q_t_2_4, dq2_1_4, __fmaf_rn(dk2b_1_4 - _tmem_load_11[3], k_t_1_4, cr_3)) + dgf_1_4;
            s_run[0] = s_run[0] + dg_o_1_4;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 3)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (384 + d_lane)] = dq_o_1_4;
            pool_f32[(98304 - 32768 * th >> 2) + (1408 + d_lane)] = dk_o_1_4;
            int t_1_5 = 32 + 8 * th + 2;
            float r_263[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_263[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_5 = r_263[0];
            unsigned int r_264[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_264[0])) : "r"((kb8 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_5 = __uint_as_float(r_264[0] << 16);
            unsigned int r_265[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_265[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_5 = __uint_as_float(r_265[0] << 16);
            float q_t_2_5 = ((t_1_5 >= clen) ? 0.0f : q_t_1_5);
            float dqf_1_5 = pool_f32[ew + 256 + d_lane];
            float dkf_1_5 = pool_f32[ew + 1280 + d_lane];
            float dgf_1_5 = pool_f32[ew + 2304 + d_lane];
            float bt_t_1_5 = btv[2];
            float e_off_1_5 = approx_exp2(g_t_1_5 - gn);
            float e_dg_1_5 = approx_exp2(g_t_1_5 - gc_888);
            float dq2_1_5 = __fmaf_rn(_tmem_load_7[2], e_off_1_5, _tmem_load_9[2] * e_dg_1_5);
            float dk2_1_5 = __fmaf_rn(_tmem_load_8[2], e_off_1_5, _tmem_load_10[2] * e_dg_1_5);
            dbp[2] = dk2_1_5 * k_t_1_5;
            float dk2b_1_5 = dk2_1_5 * bt_t_1_5;
            float dq_o_1_5 = dq2_1_5 + dqf_1_5;
            float dk_o_1_5 = dk2b_1_5 + dkf_1_5 + _tmem_load_11[2];
            float qe_4 = q_t_2_5 * e_dg_1_5;
            float kp_16 = k_t_1_5 * bt_t_1_5;
            __nv_bfloat16 kb_4 = (__nv_bfloat16)kp_16;
            float kbe_4 = (float)kb_4 * e_dg_1_5;
            float cr_4 = __fmaf_rn(__uint_as_float(__as_u32(qe_4) & 4294959104u) - qe_4, _tmem_load_9[2], (__uint_as_float(__as_u32(kbe_4) & 4294959104u) - kp_16 * e_dg_1_5) * _tmem_load_10[2]);
            float dg_o_1_5 = __fmaf_rn(q_t_2_5, dq2_1_5, __fmaf_rn(dk2b_1_5 - _tmem_load_11[2], k_t_1_5, cr_4)) + dgf_1_5;
            s_run[0] = s_run[0] + dg_o_1_5;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 2)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (256 + d_lane)] = dq_o_1_5;
            pool_f32[(98304 - 32768 * th >> 2) + (1280 + d_lane)] = dk_o_1_5;
            int t_1_6 = 32 + 8 * th + 1;
            float r_266[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_266[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_6 = r_266[0];
            unsigned int r_267[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_267[0])) : "r"((kb8 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_6 = __uint_as_float(r_267[0] << 16);
            unsigned int r_268[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_268[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_6 = __uint_as_float(r_268[0] << 16);
            float q_t_2_6 = ((t_1_6 >= clen) ? 0.0f : q_t_1_6);
            float dqf_1_6 = pool_f32[ew + 128 + d_lane];
            float dkf_1_6 = pool_f32[ew + 1152 + d_lane];
            float dgf_1_6 = pool_f32[ew + 2176 + d_lane];
            float bt_t_1_6 = btv[1];
            float e_off_1_6 = approx_exp2(g_t_1_6 - gn);
            float e_dg_1_6 = approx_exp2(g_t_1_6 - gc_888);
            float dq2_1_6 = __fmaf_rn(_tmem_load_7[1], e_off_1_6, _tmem_load_9[1] * e_dg_1_6);
            float dk2_1_6 = __fmaf_rn(_tmem_load_8[1], e_off_1_6, _tmem_load_10[1] * e_dg_1_6);
            dbp[1] = dk2_1_6 * k_t_1_6;
            float dk2b_1_6 = dk2_1_6 * bt_t_1_6;
            float dq_o_1_6 = dq2_1_6 + dqf_1_6;
            float dk_o_1_6 = dk2b_1_6 + dkf_1_6 + _tmem_load_11[1];
            float qe_5 = q_t_2_6 * e_dg_1_6;
            float kp_17 = k_t_1_6 * bt_t_1_6;
            __nv_bfloat16 kb_5 = (__nv_bfloat16)kp_17;
            float kbe_5 = (float)kb_5 * e_dg_1_6;
            float cr_5 = __fmaf_rn(__uint_as_float(__as_u32(qe_5) & 4294959104u) - qe_5, _tmem_load_9[1], (__uint_as_float(__as_u32(kbe_5) & 4294959104u) - kp_17 * e_dg_1_6) * _tmem_load_10[1]);
            float dg_o_1_6 = __fmaf_rn(q_t_2_6, dq2_1_6, __fmaf_rn(dk2b_1_6 - _tmem_load_11[1], k_t_1_6, cr_5)) + dgf_1_6;
            s_run[0] = s_run[0] + dg_o_1_6;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 1)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + (128 + d_lane)] = dq_o_1_6;
            pool_f32[(98304 - 32768 * th >> 2) + (1152 + d_lane)] = dk_o_1_6;
            int t_1_7 = 32 + 8 * th;
            float r_269[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_269[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1_7 = r_269[0];
            unsigned int r_270[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_270[0])) : "r"((kb8 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1_7 = __uint_as_float(r_270[0] << 16);
            unsigned int r_271[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_271[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1_7 = __uint_as_float(r_271[0] << 16);
            float q_t_2_7 = ((t_1_7 >= clen) ? 0.0f : q_t_1_7);
            float dqf_1_7 = pool_f32[ew + d_lane];
            float dkf_1_7 = pool_f32[ew + 1024 + d_lane];
            float dgf_1_7 = pool_f32[ew + 2048 + d_lane];
            float bt_t_1_7 = btv[0];
            float e_off_1_7 = approx_exp2(g_t_1_7 - gn);
            float e_dg_1_7 = approx_exp2(g_t_1_7 - gc_888);
            float dq2_1_7 = __fmaf_rn(_tmem_load_7[0], e_off_1_7, _tmem_load_9[0] * e_dg_1_7);
            float dk2_1_7 = __fmaf_rn(_tmem_load_8[0], e_off_1_7, _tmem_load_10[0] * e_dg_1_7);
            dbp[0] = dk2_1_7 * k_t_1_7;
            float dk2b_1_7 = dk2_1_7 * bt_t_1_7;
            float dq_o_1_7 = dq2_1_7 + dqf_1_7;
            float dk_o_1_7 = dk2b_1_7 + dkf_1_7 + _tmem_load_11[0];
            float qe_6 = q_t_2_7 * e_dg_1_7;
            float kp_18 = k_t_1_7 * bt_t_1_7;
            __nv_bfloat16 kb_6 = (__nv_bfloat16)kp_18;
            float kbe_6 = (float)kb_6 * e_dg_1_7;
            float cr_6 = __fmaf_rn(__uint_as_float(__as_u32(qe_6) & 4294959104u) - qe_6, _tmem_load_9[0], (__uint_as_float(__as_u32(kbe_6) & 4294959104u) - kp_18 * e_dg_1_7) * _tmem_load_10[0]);
            float dg_o_1_7 = __fmaf_rn(q_t_2_7, dq2_1_7, __fmaf_rn(dk2b_1_7 - _tmem_load_11[0], k_t_1_7, cr_6)) + dgf_1_7;
            s_run[0] = s_run[0] + dg_o_1_7;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            pool_f32[(98304 - 32768 * th >> 2) + d_lane] = dq_o_1_7;
            pool_f32[(98304 - 32768 * th >> 2) + (1024 + d_lane)] = dk_o_1_7;
            float tot = s_run[0];
            int b4 = lane_1 >> 4 & 1;
            int b3 = lane_1 >> 3 & 1;
            int b2 = lane_1 >> 2 & 1;
            float v4[4];
            float send_1_0 = ((b4 == 0) ? dbp[4] : dbp[0]);
            float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 16);
            v4[0] = ((b4 == 0) ? dbp[0] : dbp[4]) + _shfl_xor_sync_248;
            float send_1_1 = ((b4 == 0) ? dbp[5] : dbp[1]);
            float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 16);
            v4[1] = ((b4 == 0) ? dbp[1] : dbp[5]) + _shfl_xor_sync_249;
            float send_1_2 = ((b4 == 0) ? dbp[6] : dbp[2]);
            float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 16);
            v4[2] = ((b4 == 0) ? dbp[2] : dbp[6]) + _shfl_xor_sync_250;
            float send_1_3 = ((b4 == 0) ? dbp[7] : dbp[3]);
            float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 16);
            v4[3] = ((b4 == 0) ? dbp[3] : dbp[7]) + _shfl_xor_sync_251;
            float v2_1044[2];
            float send_1_4 = ((b3 == 0) ? v4[2] : v4[0]);
            float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 8);
            v2_1044[0] = ((b3 == 0) ? v4[0] : v4[2]) + _shfl_xor_sync_252;
            float send_1_5 = ((b3 == 0) ? v4[3] : v4[1]);
            float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 8);
            v2_1044[1] = ((b3 == 0) ? v4[1] : v4[3]) + _shfl_xor_sync_253;
            float send_1047 = ((b2 == 0) ? v2_1044[1] : v2_1044[0]);
            float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, send_1047, 4);
            float v1 = ((b2 == 0) ? v2_1044[0] : v2_1044[1]) + _shfl_xor_sync_254;
            float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
            v1 = v1 + _shfl_xor_sync_255;
            float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
            v1 = v1 + _shfl_xor_sync_256;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1a_t2_drained_addr);
            int ew_0 = 98304 - 32768 * th >> 2;
            float nz = __uint_as_float((unsigned int)num_heads >> 31 | 2147483648u);
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int i = 0; i < 2; i++) {
                int j = qd + 4 * i;
                bool live = 32 + 8 * th + j < clen;
                int _min_257 = ((32 + 8 * th + j) < (clen - 1) ? (32 + 8 * th + j) : (clen - 1));
                long long row = ext0 + (long long)_min_257;
                float gq[4];
                float gk_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_0 + 128 * j + 4 * lane_1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gk_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_0[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_0 + 1024 + 128 * j + 4 * lane_1))));
                if (i == 1) {
                    mbarrier_arrive(epi_empty_addr + (th) * 8);
                }
                float xq[4];
                float xk[4];
                float rq[1];
                float rk[1];
                {
                    uint2 _vld_258;
                    _vld_258 = *reinterpret_cast<const uint2*>(q_raw + (row * (long long)q_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_258 = reinterpret_cast<uint32_t*>(&_vld_258);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_258[_pair]) << 16);
                        (&xq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_258[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rq[0] = *reinterpret_cast<const float*>(qk_rstd + ((0 + row) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                {
                    uint2 _vld_259;
                    _vld_259 = *reinterpret_cast<const uint2*>(k_raw + (row * (long long)k_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_259 = reinterpret_cast<uint32_t*>(&_vld_259);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xk[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_259[_pair]) << 16);
                        (&xk[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_259[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rk[0] = *reinterpret_cast<const float*>(qk_rstd + (((long long)num_tokens + row) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                long long dxo = row * (long long)qrs + (long long)(qk_head * 128 + 4 * lane_1);
                float y[4];
                float p[4];
                y[0] = mul_f32_rn_noftz(xq[0], rq[0]);
                float _fma_260;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(gq[0]), "f"(y[0]), "f"(nz));
                p[0] = _fma_260;
                y[1] = mul_f32_rn_noftz(xq[1], rq[0]);
                float _fma_261;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(gq[1]), "f"(y[1]), "f"(nz));
                p[1] = _fma_261;
                y[2] = mul_f32_rn_noftz(xq[2], rq[0]);
                float _fma_262;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(gq[2]), "f"(y[2]), "f"(nz));
                p[2] = _fma_262;
                y[3] = mul_f32_rn_noftz(xq[3], rq[0]);
                float _fma_263;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(gq[3]), "f"(y[3]), "f"(nz));
                p[3] = _fma_263;
                float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, p[0], 4);
                float _add_264;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(p[0]), "f"(_shfl_xor_sync_265));
                p[0] = _add_264;
                float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, p[1], 4);
                float _add_266;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(p[1]), "f"(_shfl_xor_sync_267));
                p[1] = _add_266;
                float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, p[2], 4);
                float _add_268;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(p[2]), "f"(_shfl_xor_sync_269));
                p[2] = _add_268;
                float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, p[3], 4);
                float _add_270;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(p[3]), "f"(_shfl_xor_sync_271));
                p[3] = _add_270;
                float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, p[0], 2);
                float _add_272;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(p[0]), "f"(_shfl_xor_sync_273));
                p[0] = _add_272;
                float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, p[1], 2);
                float _add_274;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(p[1]), "f"(_shfl_xor_sync_275));
                p[1] = _add_274;
                float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, p[2], 2);
                float _add_276;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(p[2]), "f"(_shfl_xor_sync_277));
                p[2] = _add_276;
                float _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, p[3], 2);
                float _add_278;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(p[3]), "f"(_shfl_xor_sync_279));
                p[3] = _add_278;
                float _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, p[0], 1);
                float _add_280;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(p[0]), "f"(_shfl_xor_sync_281));
                p[0] = _add_280;
                float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, p[1], 1);
                float _add_282;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(p[1]), "f"(_shfl_xor_sync_283));
                p[1] = _add_282;
                float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, p[2], 1);
                float _add_284;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(p[2]), "f"(_shfl_xor_sync_285));
                p[2] = _add_284;
                float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, p[3], 1);
                float _add_286;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(p[3]), "f"(_shfl_xor_sync_287));
                p[3] = _add_286;
                float _add_288;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(p[0]), "f"(p[2]));
                float _add_289;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(p[1]), "f"(p[3]));
                float sq;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq) : "f"(_add_288), "f"(_add_289));
                float _shfl_sync_292 = __shfl_sync(0xFFFFFFFF, sq, 0);
                float _shfl_sync_293 = __shfl_sync(0xFFFFFFFF, sq, 8);
                float _add_291;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(_shfl_sync_292), "f"(_shfl_sync_293));
                float _shfl_sync_294 = __shfl_sync(0xFFFFFFFF, sq, 16);
                float _add_290;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_add_291), "f"(_shfl_sync_294));
                float _shfl_sync_295 = __shfl_sync(0xFFFFFFFF, sq, 24);
                float dot;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot) : "f"(_add_290), "f"(_shfl_sync_295));
                float o[4];
                float _fma_297;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(y[0]), "f"(dot), "f"(nz));
                float _sub_296;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_296) : "f"(gq[0]), "f"(_fma_297));
                o[0] = mul_f32_rn_noftz(rq[0], _sub_296);
                float _fma_299;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(y[1]), "f"(dot), "f"(nz));
                float _sub_298;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(gq[1]), "f"(_fma_299));
                o[1] = mul_f32_rn_noftz(rq[0], _sub_298);
                float _fma_301;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(y[2]), "f"(dot), "f"(nz));
                float _sub_300;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(gq[2]), "f"(_fma_301));
                o[2] = mul_f32_rn_noftz(rq[0], _sub_300);
                float _fma_303;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(y[3]), "f"(dot), "f"(nz));
                float _sub_302;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_302) : "f"(gq[3]), "f"(_fma_303));
                o[3] = mul_f32_rn_noftz(rq[0], _sub_302);
                unsigned int ob[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 0], o[_lp*2+1 + 0]));
                    ob[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dq_ext + dxo), "r"(ob[(0) + 0]), "r"(ob[(0) + 1]), "r"((unsigned int)(live)) : "memory");
                float y_0[4];
                float p_0[4];
                y_0[0] = mul_f32_rn_noftz(xk[0], rk[0]);
                float _fma_304;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(gk_0[0]), "f"(y_0[0]), "f"(nz));
                p_0[0] = _fma_304;
                y_0[1] = mul_f32_rn_noftz(xk[1], rk[0]);
                float _fma_305;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(gk_0[1]), "f"(y_0[1]), "f"(nz));
                p_0[1] = _fma_305;
                y_0[2] = mul_f32_rn_noftz(xk[2], rk[0]);
                float _fma_306;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(gk_0[2]), "f"(y_0[2]), "f"(nz));
                p_0[2] = _fma_306;
                y_0[3] = mul_f32_rn_noftz(xk[3], rk[0]);
                float _fma_307;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(gk_0[3]), "f"(y_0[3]), "f"(nz));
                p_0[3] = _fma_307;
                float _shfl_xor_sync_309 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 4);
                float _add_308;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_308) : "f"(p_0[0]), "f"(_shfl_xor_sync_309));
                p_0[0] = _add_308;
                float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 4);
                float _add_310;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_310) : "f"(p_0[1]), "f"(_shfl_xor_sync_311));
                p_0[1] = _add_310;
                float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 4);
                float _add_312;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(p_0[2]), "f"(_shfl_xor_sync_313));
                p_0[2] = _add_312;
                float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 4);
                float _add_314;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_314) : "f"(p_0[3]), "f"(_shfl_xor_sync_315));
                p_0[3] = _add_314;
                float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 2);
                float _add_316;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(p_0[0]), "f"(_shfl_xor_sync_317));
                p_0[0] = _add_316;
                float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 2);
                float _add_318;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_318) : "f"(p_0[1]), "f"(_shfl_xor_sync_319));
                p_0[1] = _add_318;
                float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 2);
                float _add_320;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(p_0[2]), "f"(_shfl_xor_sync_321));
                p_0[2] = _add_320;
                float _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 2);
                float _add_322;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(p_0[3]), "f"(_shfl_xor_sync_323));
                p_0[3] = _add_322;
                float _shfl_xor_sync_325 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 1);
                float _add_324;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(p_0[0]), "f"(_shfl_xor_sync_325));
                p_0[0] = _add_324;
                float _shfl_xor_sync_327 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 1);
                float _add_326;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(p_0[1]), "f"(_shfl_xor_sync_327));
                p_0[1] = _add_326;
                float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 1);
                float _add_328;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(p_0[2]), "f"(_shfl_xor_sync_329));
                p_0[2] = _add_328;
                float _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 1);
                float _add_330;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(p_0[3]), "f"(_shfl_xor_sync_331));
                p_0[3] = _add_330;
                float _add_332;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(p_0[0]), "f"(p_0[2]));
                float _add_333;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_333) : "f"(p_0[1]), "f"(p_0[3]));
                float sq_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_0) : "f"(_add_332), "f"(_add_333));
                float _shfl_sync_336 = __shfl_sync(0xFFFFFFFF, sq_0, 0);
                float _shfl_sync_337 = __shfl_sync(0xFFFFFFFF, sq_0, 8);
                float _add_335;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(_shfl_sync_336), "f"(_shfl_sync_337));
                float _shfl_sync_338 = __shfl_sync(0xFFFFFFFF, sq_0, 16);
                float _add_334;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(_add_335), "f"(_shfl_sync_338));
                float _shfl_sync_339 = __shfl_sync(0xFFFFFFFF, sq_0, 24);
                float dot_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_0) : "f"(_add_334), "f"(_shfl_sync_339));
                float o_0[4];
                float _fma_341;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(y_0[0]), "f"(dot_0), "f"(nz));
                float _sub_340;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_340) : "f"(gk_0[0]), "f"(_fma_341));
                o_0[0] = mul_f32_rn_noftz(rk[0], _sub_340);
                float _fma_343;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(y_0[1]), "f"(dot_0), "f"(nz));
                float _sub_342;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_342) : "f"(gk_0[1]), "f"(_fma_343));
                o_0[1] = mul_f32_rn_noftz(rk[0], _sub_342);
                float _fma_345;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(y_0[2]), "f"(dot_0), "f"(nz));
                float _sub_344;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_344) : "f"(gk_0[2]), "f"(_fma_345));
                o_0[2] = mul_f32_rn_noftz(rk[0], _sub_344);
                float _fma_347;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(y_0[3]), "f"(dot_0), "f"(nz));
                float _sub_346;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_346) : "f"(gk_0[3]), "f"(_fma_347));
                o_0[3] = mul_f32_rn_noftz(rk[0], _sub_346);
                unsigned int ob_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_0[_lp*2 + 0], o_0[_lp*2+1 + 0]));
                    ob_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dk_ext + dxo), "r"(ob_0[(0) + 0]), "r"(ob_0[(0) + 1]), "r"((unsigned int)(live)) : "memory");
            }
            int pf_row = 32 + 8 * th + (lane_1 & 7) + 16 * (lane_1 >> 3);
            long long pf_off = (ext0 + (long long)pf_row) * g_raw_row_stride + (long long)hv * g_raw_head_stride;
            if (lane_1 < 16 && pf_row < clen) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(g_raw + (pf_off + (long long)(32 * qd))))); }
            if (lane_1 == 16) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(A_log + hv))); }
            if (lane_1 == 17) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dt_bias + (hv * 128 + 32 * qd)))); }
            mbarrier_wait(epi_full_addr + (th) * 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_272[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_272[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_0 = r_272[0];
            float gc_0 = gn_0 - 21.640425f;
            int eb_0 = 98304 + th * -32768;
            int ew_1 = eb_0 >> 2;
            int qbase_0 = pool_f32_addr + (unsigned int)(eb_0 + 12288 + dh_lane * 1024);
            int sb8_0 = 48 + 8 * th;
            int gb8_0 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_0 * 128);
            int kb8_0 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_0 * 128);
            int c_qa = 160 + 8 * th;
            int c_ka = 176 + 8 * th;
            int c_qd = 224 + 8 * th;
            int c_kd = 240 + 8 * th;
            int c_dk = 48 + 8 * th;
            int c_bp = c_ka;
            float s_run_0[1];
            s_run_0[0] = 0.0f;
            #pragma unroll 1
            for (int k = 0; k < 8; k++) {
                int j_1 = 7 - k;
                float a_qd[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd + j_1)));
                float a_kd[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd + j_1)));
                float a_dk[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk + j_1)));
                float a_qa[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qa[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qa + j_1)));
                float a_ka[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_ka[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_ka + j_1)));
                float btr[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_0 + j_1))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t = sb8_0 + j_1;
                float r_273[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_273[0])) : "r"((gb8_0 + (lane_1 / 32 * 8192 + 128 * j_1 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_1 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t = r_273[0];
                unsigned int r_274[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_274[0])) : "r"((kb8_0 + (dcol / 64 * 8192 + 128 * j_1 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_1 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t = __uint_as_float(r_274[0] << 16);
                unsigned int r_275[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_275[0])) : "r"((qbase_0 + (dcol / 64 * 1024 + 128 * j_1 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_1 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t = __uint_as_float(r_275[0] << 16);
                float q_m = ((t >= clen) ? 0.0f : q_t);
                float dqf = pool_f32[ew_1 + 128 * j_1 + d_lane];
                float dkf = pool_f32[ew_1 + (1024 + 128 * j_1) + d_lane];
                float dgf = pool_f32[ew_1 + (2048 + 128 * j_1) + d_lane];
                float bt_t = btr[0];
                float e_dg = approx_exp2(g_t - gc_0);
                float e_off = approx_exp2(g_t - gn_0);
                float dq2 = __fmaf_rn(a_qa[0], e_off, a_qd[0] * e_dg);
                float dk2 = __fmaf_rn(a_ka[0], e_off, a_kd[0] * e_dg);
                float dbv[1];
                dbv[0] = dk2 * k_t;
                float dk2b = dk2 * bt_t;
                float dq_o = dq2 + dqf;
                float dk_o = dk2b + dkf + a_dk[0];
                float qe_7 = q_m * e_dg;
                float kp_19 = k_t * bt_t;
                __nv_bfloat16 kb_7 = (__nv_bfloat16)kp_19;
                float kbe_7 = (float)kb_7 * e_dg;
                float cr_7 = __fmaf_rn(__uint_as_float(__as_u32(qe_7) & 4294959104u) - qe_7, a_qd[0], (__uint_as_float(__as_u32(kbe_7) & 4294959104u) - kp_19 * e_dg) * a_kd[0]);
                float dg_o = __fmaf_rn(q_m, dq2, __fmaf_rn(dk2b - a_dk[0], k_t, cr_7)) + dgf;
                s_run_0[0] = s_run_0[0] + dg_o;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd + j_1)), "f"(s_run_0[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp + j_1)), "f"(dbv[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                pool_f32[(98304 - 32768 * th >> 2) + (128 * j_1 + d_lane)] = dq_o;
                pool_f32[(98304 - 32768 * th >> 2) + (1024 + 128 * j_1 + d_lane)] = dk_o;
            }
            float tot_0 = s_run_0[0];
            float dbp_0[8];
            tmem_ld_x8(&dbp_0[0], taddr + (unsigned int)row_base + (unsigned int)c_bp);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_0 = lane_1 >> 4 & 1;
            int b3_0 = lane_1 >> 3 & 1;
            int b2_0 = lane_1 >> 2 & 1;
            float v4_0[4];
            float send_0 = ((b4_0 == 0) ? dbp_0[4] : dbp_0[0]);
            float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
            v4_0[0] = ((b4_0 == 0) ? dbp_0[0] : dbp_0[4]) + _shfl_xor_sync_348;
            float send_1 = ((b4_0 == 0) ? dbp_0[5] : dbp_0[1]);
            float _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
            v4_0[1] = ((b4_0 == 0) ? dbp_0[1] : dbp_0[5]) + _shfl_xor_sync_349;
            float send_2 = ((b4_0 == 0) ? dbp_0[6] : dbp_0[2]);
            float _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
            v4_0[2] = ((b4_0 == 0) ? dbp_0[2] : dbp_0[6]) + _shfl_xor_sync_350;
            float send_3 = ((b4_0 == 0) ? dbp_0[7] : dbp_0[3]);
            float _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
            v4_0[3] = ((b4_0 == 0) ? dbp_0[3] : dbp_0[7]) + _shfl_xor_sync_351;
            float v2[2];
            float send2_0 = ((b3_0 == 0) ? v4_0[2] : v4_0[0]);
            float _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, send2_0, 8);
            v2[0] = ((b3_0 == 0) ? v4_0[0] : v4_0[2]) + _shfl_xor_sync_352;
            float send2_1 = ((b3_0 == 0) ? v4_0[3] : v4_0[1]);
            float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, send2_1, 8);
            v2[1] = ((b3_0 == 0) ? v4_0[1] : v4_0[3]) + _shfl_xor_sync_353;
            float send1 = ((b2_0 == 0) ? v2[1] : v2[0]);
            float _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, send1, 4);
            float v1_0 = ((b2_0 == 0) ? v2[0] : v2[1]) + _shfl_xor_sync_354;
            float _shfl_xor_sync_355 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 2);
            v1_0 = v1_0 + _shfl_xor_sync_355;
            float _shfl_xor_sync_356 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 1);
            v1_0 = v1_0 + _shfl_xor_sync_356;
            asm volatile("tcgen05.fence::before_thread_sync;");
            int ew_2 = 98304 - 32768 * th >> 2;
            float nz_0 = __uint_as_float((unsigned int)num_heads >> 31 | 2147483648u);
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 2; i_1++) {
                int j_2 = qd + 4 * i_1;
                bool live_1 = 48 + 8 * th + j_2 < clen;
                int _min_357 = ((48 + 8 * th + j_2) < (clen - 1) ? (48 + 8 * th + j_2) : (clen - 1));
                long long row_1 = ext0 + (long long)_min_357;
                float gq_1[4];
                float gk_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gq_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq_1[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_2 + 128 * j_2 + 4 * lane_1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_1[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_2 + 1024 + 128 * j_2 + 4 * lane_1))));
                if (i_1 == 1) {
                    mbarrier_arrive(epi_empty_addr + (th) * 8);
                }
                float xq_1[4];
                float xk_1[4];
                float rq_1[1];
                float rk_1[1];
                {
                    uint2 _vld_358;
                    _vld_358 = *reinterpret_cast<const uint2*>(q_raw + (row_1 * (long long)q_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_358 = reinterpret_cast<uint32_t*>(&_vld_358);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xq_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_358[_pair]) << 16);
                        (&xq_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_358[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rq_1[0] = *reinterpret_cast<const float*>(qk_rstd + ((0 + row_1) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                {
                    uint2 _vld_359;
                    _vld_359 = *reinterpret_cast<const uint2*>(k_raw + (row_1 * (long long)k_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_359 = reinterpret_cast<uint32_t*>(&_vld_359);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xk_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_359[_pair]) << 16);
                        (&xk_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_359[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rk_1[0] = *reinterpret_cast<const float*>(qk_rstd + (((long long)num_tokens + row_1) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                long long dxo_1 = row_1 * (long long)qrs + (long long)(qk_head * 128 + 4 * lane_1);
                float y_1[4];
                float p_1[4];
                y_1[0] = mul_f32_rn_noftz(xq_1[0], rq_1[0]);
                float _fma_360;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(gq_1[0]), "f"(y_1[0]), "f"(nz_0));
                p_1[0] = _fma_360;
                y_1[1] = mul_f32_rn_noftz(xq_1[1], rq_1[0]);
                float _fma_361;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(gq_1[1]), "f"(y_1[1]), "f"(nz_0));
                p_1[1] = _fma_361;
                y_1[2] = mul_f32_rn_noftz(xq_1[2], rq_1[0]);
                float _fma_362;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(gq_1[2]), "f"(y_1[2]), "f"(nz_0));
                p_1[2] = _fma_362;
                y_1[3] = mul_f32_rn_noftz(xq_1[3], rq_1[0]);
                float _fma_363;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(gq_1[3]), "f"(y_1[3]), "f"(nz_0));
                p_1[3] = _fma_363;
                float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, p_1[0], 4);
                float _add_364;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(p_1[0]), "f"(_shfl_xor_sync_365));
                p_1[0] = _add_364;
                float _shfl_xor_sync_367 = __shfl_xor_sync(0xFFFFFFFF, p_1[1], 4);
                float _add_366;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(p_1[1]), "f"(_shfl_xor_sync_367));
                p_1[1] = _add_366;
                float _shfl_xor_sync_369 = __shfl_xor_sync(0xFFFFFFFF, p_1[2], 4);
                float _add_368;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(p_1[2]), "f"(_shfl_xor_sync_369));
                p_1[2] = _add_368;
                float _shfl_xor_sync_371 = __shfl_xor_sync(0xFFFFFFFF, p_1[3], 4);
                float _add_370;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_370) : "f"(p_1[3]), "f"(_shfl_xor_sync_371));
                p_1[3] = _add_370;
                float _shfl_xor_sync_373 = __shfl_xor_sync(0xFFFFFFFF, p_1[0], 2);
                float _add_372;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_372) : "f"(p_1[0]), "f"(_shfl_xor_sync_373));
                p_1[0] = _add_372;
                float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, p_1[1], 2);
                float _add_374;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(p_1[1]), "f"(_shfl_xor_sync_375));
                p_1[1] = _add_374;
                float _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, p_1[2], 2);
                float _add_376;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(p_1[2]), "f"(_shfl_xor_sync_377));
                p_1[2] = _add_376;
                float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, p_1[3], 2);
                float _add_378;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(p_1[3]), "f"(_shfl_xor_sync_379));
                p_1[3] = _add_378;
                float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, p_1[0], 1);
                float _add_380;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_380) : "f"(p_1[0]), "f"(_shfl_xor_sync_381));
                p_1[0] = _add_380;
                float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, p_1[1], 1);
                float _add_382;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_382) : "f"(p_1[1]), "f"(_shfl_xor_sync_383));
                p_1[1] = _add_382;
                float _shfl_xor_sync_385 = __shfl_xor_sync(0xFFFFFFFF, p_1[2], 1);
                float _add_384;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_384) : "f"(p_1[2]), "f"(_shfl_xor_sync_385));
                p_1[2] = _add_384;
                float _shfl_xor_sync_387 = __shfl_xor_sync(0xFFFFFFFF, p_1[3], 1);
                float _add_386;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_386) : "f"(p_1[3]), "f"(_shfl_xor_sync_387));
                p_1[3] = _add_386;
                float _add_388;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_388) : "f"(p_1[0]), "f"(p_1[2]));
                float _add_389;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_389) : "f"(p_1[1]), "f"(p_1[3]));
                float sq_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_1) : "f"(_add_388), "f"(_add_389));
                float _shfl_sync_392 = __shfl_sync(0xFFFFFFFF, sq_1, 0);
                float _shfl_sync_393 = __shfl_sync(0xFFFFFFFF, sq_1, 8);
                float _add_391;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_391) : "f"(_shfl_sync_392), "f"(_shfl_sync_393));
                float _shfl_sync_394 = __shfl_sync(0xFFFFFFFF, sq_1, 16);
                float _add_390;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_390) : "f"(_add_391), "f"(_shfl_sync_394));
                float _shfl_sync_395 = __shfl_sync(0xFFFFFFFF, sq_1, 24);
                float dot_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_1) : "f"(_add_390), "f"(_shfl_sync_395));
                float o_1[4];
                float _fma_397;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(y_1[0]), "f"(dot_1), "f"(nz_0));
                float _sub_396;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_396) : "f"(gq_1[0]), "f"(_fma_397));
                o_1[0] = mul_f32_rn_noftz(rq_1[0], _sub_396);
                float _fma_399;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(y_1[1]), "f"(dot_1), "f"(nz_0));
                float _sub_398;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_398) : "f"(gq_1[1]), "f"(_fma_399));
                o_1[1] = mul_f32_rn_noftz(rq_1[0], _sub_398);
                float _fma_401;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(y_1[2]), "f"(dot_1), "f"(nz_0));
                float _sub_400;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(gq_1[2]), "f"(_fma_401));
                o_1[2] = mul_f32_rn_noftz(rq_1[0], _sub_400);
                float _fma_403;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(y_1[3]), "f"(dot_1), "f"(nz_0));
                float _sub_402;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_402) : "f"(gq_1[3]), "f"(_fma_403));
                o_1[3] = mul_f32_rn_noftz(rq_1[0], _sub_402);
                unsigned int ob_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1[_lp*2 + 0], o_1[_lp*2+1 + 0]));
                    ob_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dq_ext + dxo_1), "r"(ob_1[(0) + 0]), "r"(ob_1[(0) + 1]), "r"((unsigned int)(live_1)) : "memory");
                float y_1_1[4];
                float p_1_1[4];
                y_1_1[0] = mul_f32_rn_noftz(xk_1[0], rk_1[0]);
                float _fma_404;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(gk_1[0]), "f"(y_1_1[0]), "f"(nz_0));
                p_1_1[0] = _fma_404;
                y_1_1[1] = mul_f32_rn_noftz(xk_1[1], rk_1[0]);
                float _fma_405;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(gk_1[1]), "f"(y_1_1[1]), "f"(nz_0));
                p_1_1[1] = _fma_405;
                y_1_1[2] = mul_f32_rn_noftz(xk_1[2], rk_1[0]);
                float _fma_406;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(gk_1[2]), "f"(y_1_1[2]), "f"(nz_0));
                p_1_1[2] = _fma_406;
                y_1_1[3] = mul_f32_rn_noftz(xk_1[3], rk_1[0]);
                float _fma_407;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(gk_1[3]), "f"(y_1_1[3]), "f"(nz_0));
                p_1_1[3] = _fma_407;
                float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[0], 4);
                float _add_408;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(p_1_1[0]), "f"(_shfl_xor_sync_409));
                p_1_1[0] = _add_408;
                float _shfl_xor_sync_411 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[1], 4);
                float _add_410;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_410) : "f"(p_1_1[1]), "f"(_shfl_xor_sync_411));
                p_1_1[1] = _add_410;
                float _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[2], 4);
                float _add_412;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_412) : "f"(p_1_1[2]), "f"(_shfl_xor_sync_413));
                p_1_1[2] = _add_412;
                float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[3], 4);
                float _add_414;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_414) : "f"(p_1_1[3]), "f"(_shfl_xor_sync_415));
                p_1_1[3] = _add_414;
                float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[0], 2);
                float _add_416;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_416) : "f"(p_1_1[0]), "f"(_shfl_xor_sync_417));
                p_1_1[0] = _add_416;
                float _shfl_xor_sync_419 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[1], 2);
                float _add_418;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(p_1_1[1]), "f"(_shfl_xor_sync_419));
                p_1_1[1] = _add_418;
                float _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[2], 2);
                float _add_420;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_420) : "f"(p_1_1[2]), "f"(_shfl_xor_sync_421));
                p_1_1[2] = _add_420;
                float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[3], 2);
                float _add_422;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_422) : "f"(p_1_1[3]), "f"(_shfl_xor_sync_423));
                p_1_1[3] = _add_422;
                float _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[0], 1);
                float _add_424;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_424) : "f"(p_1_1[0]), "f"(_shfl_xor_sync_425));
                p_1_1[0] = _add_424;
                float _shfl_xor_sync_427 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[1], 1);
                float _add_426;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_426) : "f"(p_1_1[1]), "f"(_shfl_xor_sync_427));
                p_1_1[1] = _add_426;
                float _shfl_xor_sync_429 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[2], 1);
                float _add_428;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_428) : "f"(p_1_1[2]), "f"(_shfl_xor_sync_429));
                p_1_1[2] = _add_428;
                float _shfl_xor_sync_431 = __shfl_xor_sync(0xFFFFFFFF, p_1_1[3], 1);
                float _add_430;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_430) : "f"(p_1_1[3]), "f"(_shfl_xor_sync_431));
                p_1_1[3] = _add_430;
                float _add_432;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_432) : "f"(p_1_1[0]), "f"(p_1_1[2]));
                float _add_433;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(p_1_1[1]), "f"(p_1_1[3]));
                float sq_1_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_1_1) : "f"(_add_432), "f"(_add_433));
                float _shfl_sync_436 = __shfl_sync(0xFFFFFFFF, sq_1_1, 0);
                float _shfl_sync_437 = __shfl_sync(0xFFFFFFFF, sq_1_1, 8);
                float _add_435;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_435) : "f"(_shfl_sync_436), "f"(_shfl_sync_437));
                float _shfl_sync_438 = __shfl_sync(0xFFFFFFFF, sq_1_1, 16);
                float _add_434;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(_add_435), "f"(_shfl_sync_438));
                float _shfl_sync_439 = __shfl_sync(0xFFFFFFFF, sq_1_1, 24);
                float dot_1_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_1_1) : "f"(_add_434), "f"(_shfl_sync_439));
                float o_1_1[4];
                float _fma_441;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(y_1_1[0]), "f"(dot_1_1), "f"(nz_0));
                float _sub_440;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_440) : "f"(gk_1[0]), "f"(_fma_441));
                o_1_1[0] = mul_f32_rn_noftz(rk_1[0], _sub_440);
                float _fma_443;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(y_1_1[1]), "f"(dot_1_1), "f"(nz_0));
                float _sub_442;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_442) : "f"(gk_1[1]), "f"(_fma_443));
                o_1_1[1] = mul_f32_rn_noftz(rk_1[0], _sub_442);
                float _fma_445;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(y_1_1[2]), "f"(dot_1_1), "f"(nz_0));
                float _sub_444;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_444) : "f"(gk_1[2]), "f"(_fma_445));
                o_1_1[2] = mul_f32_rn_noftz(rk_1[0], _sub_444);
                float _fma_447;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(y_1_1[3]), "f"(dot_1_1), "f"(nz_0));
                float _sub_446;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(gk_1[3]), "f"(_fma_447));
                o_1_1[3] = mul_f32_rn_noftz(rk_1[0], _sub_446);
                unsigned int ob_1_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_1_1[_lp*2 + 0], o_1_1[_lp*2+1 + 0]));
                    ob_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dk_ext + dxo_1), "r"(ob_1_1[(0) + 0]), "r"(ob_1_1[(0) + 1]), "r"((unsigned int)(live_1)) : "memory");
            }
            float gacc[8];
            float rg_rate = __nv_expf(A_log[hv]);
            float rg_clr;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(rg_clr) : "f"(lower_bound), "f"(rg_rate));
            gacc[0] = 0.0f;
            gacc[1] = 0.0f;
            gacc[2] = 0.0f;
            gacc[3] = 0.0f;
            gacc[4] = rg_rate;
            gacc[5] = rg_clr;
            gacc[6] = dt_bias[hv * 128 + d_lane];
            gacc[7] = 0.0f;
            float raw_t3[8];
            long long rb = ext0 * g_raw_row_stride + (long long)hv * g_raw_head_stride + (long long)d_lane;
            int t_c_0 = ((48 + 8 * th < clen) ? 48 + 8 * th : clen - 1);
            raw_t3[0] = (float)g_raw[rb + (long long)t_c_0 * g_raw_row_stride];
            int t_c_1 = ((48 + 8 * th + 1 < clen) ? 48 + 8 * th + 1 : clen - 1);
            raw_t3[1] = (float)g_raw[rb + (long long)t_c_1 * g_raw_row_stride];
            int t_c_2 = ((48 + 8 * th + 2 < clen) ? 48 + 8 * th + 2 : clen - 1);
            raw_t3[2] = (float)g_raw[rb + (long long)t_c_2 * g_raw_row_stride];
            int t_c_3 = ((48 + 8 * th + 3 < clen) ? 48 + 8 * th + 3 : clen - 1);
            raw_t3[3] = (float)g_raw[rb + (long long)t_c_3 * g_raw_row_stride];
            int t_c_4 = ((48 + 8 * th + 4 < clen) ? 48 + 8 * th + 4 : clen - 1);
            raw_t3[4] = (float)g_raw[rb + (long long)t_c_4 * g_raw_row_stride];
            int t_c_5 = ((48 + 8 * th + 5 < clen) ? 48 + 8 * th + 5 : clen - 1);
            raw_t3[5] = (float)g_raw[rb + (long long)t_c_5 * g_raw_row_stride];
            int t_c_6 = ((48 + 8 * th + 6 < clen) ? 48 + 8 * th + 6 : clen - 1);
            raw_t3[6] = (float)g_raw[rb + (long long)t_c_6 * g_raw_row_stride];
            int t_c_7 = ((48 + 8 * th + 7 < clen) ? 48 + 8 * th + 7 : clen - 1);
            raw_t3[7] = (float)g_raw[rb + (long long)t_c_7 * g_raw_row_stride];
            float raw_t2[8];
            long long rb_0 = ext0 * g_raw_row_stride + (long long)hv * g_raw_head_stride + (long long)d_lane;
            int t_c_8 = ((32 + 8 * th < clen) ? 32 + 8 * th : clen - 1);
            raw_t2[0] = (float)g_raw[rb_0 + (long long)t_c_8 * g_raw_row_stride];
            int t_c_9 = ((32 + 8 * th + 1 < clen) ? 32 + 8 * th + 1 : clen - 1);
            raw_t2[1] = (float)g_raw[rb_0 + (long long)t_c_9 * g_raw_row_stride];
            int t_c_10 = ((32 + 8 * th + 2 < clen) ? 32 + 8 * th + 2 : clen - 1);
            raw_t2[2] = (float)g_raw[rb_0 + (long long)t_c_10 * g_raw_row_stride];
            int t_c_11 = ((32 + 8 * th + 3 < clen) ? 32 + 8 * th + 3 : clen - 1);
            raw_t2[3] = (float)g_raw[rb_0 + (long long)t_c_11 * g_raw_row_stride];
            int t_c_12 = ((32 + 8 * th + 4 < clen) ? 32 + 8 * th + 4 : clen - 1);
            raw_t2[4] = (float)g_raw[rb_0 + (long long)t_c_12 * g_raw_row_stride];
            int t_c_13 = ((32 + 8 * th + 5 < clen) ? 32 + 8 * th + 5 : clen - 1);
            raw_t2[5] = (float)g_raw[rb_0 + (long long)t_c_13 * g_raw_row_stride];
            int t_c_14 = ((32 + 8 * th + 6 < clen) ? 32 + 8 * th + 6 : clen - 1);
            raw_t2[6] = (float)g_raw[rb_0 + (long long)t_c_14 * g_raw_row_stride];
            int t_c_15 = ((32 + 8 * th + 7 < clen) ? 32 + 8 * th + 7 : clen - 1);
            raw_t2[7] = (float)g_raw[rb_0 + (long long)t_c_15 * g_raw_row_stride];
            {
                uint32_t _addr_448 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_448), "f"(tot_0) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (qd == 0) {
                asm volatile("barrier.sync 2, 64;" ::: "memory");
            }
            if (qd == 1) {
                asm volatile("barrier.sync 3, 64;" ::: "memory");
            }
            if (qd == 2) {
                asm volatile("barrier.sync 4, 64;" ::: "memory");
            }
            if (qd == 3) {
                asm volatile("barrier.sync 5, 64;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            float t_other[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_other[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + (1 - th) * 32 + lane_1))));
            float t_hi = ((th == 1) ? tot_0 : t_other[0]);
            float t_lo = ((th == 1) ? t_other[0] : tot_0);
            float offset = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi);
            float sfx[8];
            tmem_ld_x8(&sfx[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_a = gacc[0];
            float c_a = gacc[1];
            float s_b = gacc[2];
            float c_b = gacc[3];
            float u8[8];
            float dgr8[8];
            float2 dyg2_0 = add_f32x2(make_float2(offset, offset), make_float2(sfx[0], sfx[1]));
            float2 rp_u = add_f32x2(make_float2(raw_t3[0], raw_t3[1]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_449 = make_float2(gacc[4], gacc[4]);
            float2 rp_z;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z) : "l"(*(const unsigned long long*)&_f32x2_mul_a_449), "l"(*(const unsigned long long*)&rp_u));
            float _abs_452;
            asm("abs.f32 %0, %1;" : "=f"(_abs_452) : "f"(rp_z.x));
            float _neg_451;
            asm("neg.f32 %0, %1;" : "=f"(_neg_451) : "f"(_abs_452));
            float _exp_450 = __nv_expf(_neg_451);
            float _abs_455;
            asm("abs.f32 %0, %1;" : "=f"(_abs_455) : "f"(rp_z.y));
            float _neg_454;
            asm("neg.f32 %0, %1;" : "=f"(_neg_454) : "f"(_abs_455));
            float _exp_453 = __nv_expf(_neg_454);
            float2 rp_e = make_float2(_exp_450, _exp_453);
            float2 rp_dn = add_f32x2(make_float2(1.0f, 1.0f), rp_e);
            float2 rp_den;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den) : "l"(*(const unsigned long long*)&rp_dn), "l"(*(const unsigned long long*)&rp_dn));
            float _rcp_456 = approx_rcp(rp_den.x);
            float _rcp_457 = approx_rcp(rp_den.y);
            float2 _f32x2_mul_b_458 = make_float2(_rcp_456, _rcp_457);
            float2 rp_sp;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp) : "l"(*(const unsigned long long*)&rp_e), "l"(*(const unsigned long long*)&_f32x2_mul_b_458));
            float2 rp_t;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t) : "l"(*(const unsigned long long*)&dyg2_0), "l"(*(const unsigned long long*)&rp_sp));
            float2 _f32x2_mul_b_459 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr) : "l"(*(const unsigned long long*)&rp_t), "l"(*(const unsigned long long*)&_f32x2_mul_b_459));
            u8[0] = rp_u.x;
            u8[1] = rp_u.y;
            dgr8[0] = rp_dgr.x;
            dgr8[1] = rp_dgr.y;
            float2 dyg2_1 = add_f32x2(make_float2(offset, offset), make_float2(sfx[2], sfx[3]));
            float2 rp_u_0 = add_f32x2(make_float2(raw_t3[2], raw_t3[3]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_460 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_460), "l"(*(const unsigned long long*)&rp_u_0));
            float _abs_463;
            asm("abs.f32 %0, %1;" : "=f"(_abs_463) : "f"(rp_z_0.x));
            float _neg_462;
            asm("neg.f32 %0, %1;" : "=f"(_neg_462) : "f"(_abs_463));
            float _exp_461 = __nv_expf(_neg_462);
            float _abs_466;
            asm("abs.f32 %0, %1;" : "=f"(_abs_466) : "f"(rp_z_0.y));
            float _neg_465;
            asm("neg.f32 %0, %1;" : "=f"(_neg_465) : "f"(_abs_466));
            float _exp_464 = __nv_expf(_neg_465);
            float2 rp_e_0 = make_float2(_exp_461, _exp_464);
            float2 rp_dn_0 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_0);
            float2 rp_den_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_0) : "l"(*(const unsigned long long*)&rp_dn_0), "l"(*(const unsigned long long*)&rp_dn_0));
            float _rcp_467 = approx_rcp(rp_den_0.x);
            float _rcp_468 = approx_rcp(rp_den_0.y);
            float2 _f32x2_mul_b_469 = make_float2(_rcp_467, _rcp_468);
            float2 rp_sp_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_0) : "l"(*(const unsigned long long*)&rp_e_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_469));
            float2 rp_t_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_0) : "l"(*(const unsigned long long*)&dyg2_1), "l"(*(const unsigned long long*)&rp_sp_0));
            float2 _f32x2_mul_b_470 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_0) : "l"(*(const unsigned long long*)&rp_t_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_470));
            u8[2] = rp_u_0.x;
            u8[3] = rp_u_0.y;
            dgr8[2] = rp_dgr_0.x;
            dgr8[3] = rp_dgr_0.y;
            float2 dyg2_2 = add_f32x2(make_float2(offset, offset), make_float2(sfx[4], sfx[5]));
            float2 rp_u_1 = add_f32x2(make_float2(raw_t3[4], raw_t3[5]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_471 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_471), "l"(*(const unsigned long long*)&rp_u_1));
            float _abs_474;
            asm("abs.f32 %0, %1;" : "=f"(_abs_474) : "f"(rp_z_1.x));
            float _neg_473;
            asm("neg.f32 %0, %1;" : "=f"(_neg_473) : "f"(_abs_474));
            float _exp_472 = __nv_expf(_neg_473);
            float _abs_477;
            asm("abs.f32 %0, %1;" : "=f"(_abs_477) : "f"(rp_z_1.y));
            float _neg_476;
            asm("neg.f32 %0, %1;" : "=f"(_neg_476) : "f"(_abs_477));
            float _exp_475 = __nv_expf(_neg_476);
            float2 rp_e_1 = make_float2(_exp_472, _exp_475);
            float2 rp_dn_1 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_1);
            float2 rp_den_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_1) : "l"(*(const unsigned long long*)&rp_dn_1), "l"(*(const unsigned long long*)&rp_dn_1));
            float _rcp_478 = approx_rcp(rp_den_1.x);
            float _rcp_479 = approx_rcp(rp_den_1.y);
            float2 _f32x2_mul_b_480 = make_float2(_rcp_478, _rcp_479);
            float2 rp_sp_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_1) : "l"(*(const unsigned long long*)&rp_e_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_480));
            float2 rp_t_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_1) : "l"(*(const unsigned long long*)&dyg2_2), "l"(*(const unsigned long long*)&rp_sp_1));
            float2 _f32x2_mul_b_481 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_1) : "l"(*(const unsigned long long*)&rp_t_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_481));
            u8[4] = rp_u_1.x;
            u8[5] = rp_u_1.y;
            dgr8[4] = rp_dgr_1.x;
            dgr8[5] = rp_dgr_1.y;
            float2 dyg2_3 = add_f32x2(make_float2(offset, offset), make_float2(sfx[6], sfx[7]));
            float2 rp_u_2 = add_f32x2(make_float2(raw_t3[6], raw_t3[7]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_482 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_482), "l"(*(const unsigned long long*)&rp_u_2));
            float _abs_485;
            asm("abs.f32 %0, %1;" : "=f"(_abs_485) : "f"(rp_z_2.x));
            float _neg_484;
            asm("neg.f32 %0, %1;" : "=f"(_neg_484) : "f"(_abs_485));
            float _exp_483 = __nv_expf(_neg_484);
            float _abs_488;
            asm("abs.f32 %0, %1;" : "=f"(_abs_488) : "f"(rp_z_2.y));
            float _neg_487;
            asm("neg.f32 %0, %1;" : "=f"(_neg_487) : "f"(_abs_488));
            float _exp_486 = __nv_expf(_neg_487);
            float2 rp_e_2 = make_float2(_exp_483, _exp_486);
            float2 rp_dn_2 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_2);
            float2 rp_den_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_2) : "l"(*(const unsigned long long*)&rp_dn_2), "l"(*(const unsigned long long*)&rp_dn_2));
            float _rcp_489 = approx_rcp(rp_den_2.x);
            float _rcp_490 = approx_rcp(rp_den_2.y);
            float2 _f32x2_mul_b_491 = make_float2(_rcp_489, _rcp_490);
            float2 rp_sp_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_2) : "l"(*(const unsigned long long*)&rp_e_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_491));
            float2 rp_t_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_2) : "l"(*(const unsigned long long*)&dyg2_3), "l"(*(const unsigned long long*)&rp_sp_2));
            float2 _f32x2_mul_b_492 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_2) : "l"(*(const unsigned long long*)&rp_t_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_492));
            u8[6] = rp_u_2.x;
            u8[7] = rp_u_2.y;
            dgr8[6] = rp_dgr_2.x;
            dgr8[7] = rp_dgr_2.y;
            int odd = lane & 1;
            unsigned int sel = ((odd == 1) ? 12918 : 21520);
            unsigned int pw[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dgr8[_lp*2 + 0], dgr8[_lp*2+1 + 0]));
                pw[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int pv_0 = __shfl_xor_sync(0xFFFFFFFF, pw[0], 1);
            unsigned int pq_0[1];
            uint32_t _prmt_493;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_493) : "r"(pw[0]), "r"(pv_0), "r"(sel));
            pq_0[0] = _prmt_493;
            int t_q_0 = 48 + 8 * th + odd;
            if (t_q_0 < clen) {
                {
                    auto _gst_494 = draw_base + (t_q_0 * rs_g - odd);
                    __builtin_assume(__isGlobal(_gst_494));
                    *(reinterpret_cast<unsigned int*>(_gst_494) + (0)) = pq_0[0];
                }
            }
            unsigned int pv_1 = __shfl_xor_sync(0xFFFFFFFF, pw[1], 1);
            unsigned int pq_1[1];
            uint32_t _prmt_495;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_495) : "r"(pw[1]), "r"(pv_1), "r"(sel));
            pq_1[0] = _prmt_495;
            int t_q_1 = 48 + 8 * th + 2 + odd;
            if (t_q_1 < clen) {
                {
                    auto _gst_496 = draw_base + (t_q_1 * rs_g - odd);
                    __builtin_assume(__isGlobal(_gst_496));
                    *(reinterpret_cast<unsigned int*>(_gst_496) + (0)) = pq_1[0];
                }
            }
            unsigned int pv_2 = __shfl_xor_sync(0xFFFFFFFF, pw[2], 1);
            unsigned int pq_2[1];
            uint32_t _prmt_497;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_497) : "r"(pw[2]), "r"(pv_2), "r"(sel));
            pq_2[0] = _prmt_497;
            int t_q_2 = 48 + 8 * th + 4 + odd;
            if (t_q_2 < clen) {
                {
                    auto _gst_498 = draw_base + (t_q_2 * rs_g - odd);
                    __builtin_assume(__isGlobal(_gst_498));
                    *(reinterpret_cast<unsigned int*>(_gst_498) + (0)) = pq_2[0];
                }
            }
            unsigned int pv_3 = __shfl_xor_sync(0xFFFFFFFF, pw[3], 1);
            unsigned int pq_3[1];
            uint32_t _prmt_499;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_499) : "r"(pw[3]), "r"(pv_3), "r"(sel));
            pq_3[0] = _prmt_499;
            int t_q_3 = 48 + 8 * th + 6 + odd;
            if (t_q_3 < clen) {
                {
                    auto _gst_500 = draw_base + (t_q_3 * rs_g - odd);
                    __builtin_assume(__isGlobal(_gst_500));
                    *(reinterpret_cast<unsigned int*>(_gst_500) + (0)) = pq_3[0];
                }
            }
            int t_a_0 = 48 + 8 * th;
            dgr8[0] = ((t_a_0 < clen) ? dgr8[0] : 0.0f);
            u8[0] = ((t_a_0 < clen) ? u8[0] : 0.0f);
            int t_a_1 = 48 + 8 * th + 1;
            dgr8[1] = ((t_a_1 < clen) ? dgr8[1] : 0.0f);
            u8[1] = ((t_a_1 < clen) ? u8[1] : 0.0f);
            int t_a_2 = 48 + 8 * th + 2;
            dgr8[2] = ((t_a_2 < clen) ? dgr8[2] : 0.0f);
            u8[2] = ((t_a_2 < clen) ? u8[2] : 0.0f);
            int t_a_3 = 48 + 8 * th + 3;
            dgr8[3] = ((t_a_3 < clen) ? dgr8[3] : 0.0f);
            u8[3] = ((t_a_3 < clen) ? u8[3] : 0.0f);
            int t_a_4 = 48 + 8 * th + 4;
            dgr8[4] = ((t_a_4 < clen) ? dgr8[4] : 0.0f);
            u8[4] = ((t_a_4 < clen) ? u8[4] : 0.0f);
            int t_a_5 = 48 + 8 * th + 5;
            dgr8[5] = ((t_a_5 < clen) ? dgr8[5] : 0.0f);
            u8[5] = ((t_a_5 < clen) ? u8[5] : 0.0f);
            int t_a_6 = 48 + 8 * th + 6;
            dgr8[6] = ((t_a_6 < clen) ? dgr8[6] : 0.0f);
            u8[6] = ((t_a_6 < clen) ? u8[6] : 0.0f);
            int t_a_7 = 48 + 8 * th + 7;
            dgr8[7] = ((t_a_7 < clen) ? dgr8[7] : 0.0f);
            u8[7] = ((t_a_7 < clen) ? u8[7] : 0.0f);
            float pb01;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb01) : "f"(dgr8[0]), "f"(dgr8[1]));
            float pb23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb23) : "f"(dgr8[2]), "f"(dgr8[3]));
            float pb45;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb45) : "f"(dgr8[4]), "f"(dgr8[5]));
            float pb67;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb67) : "f"(dgr8[6]), "f"(dgr8[7]));
            float pb03;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb03) : "f"(pb01), "f"(pb23));
            float pb47;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb47) : "f"(pb45), "f"(pb67));
            float pb;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb) : "f"(pb03), "f"(pb47));
            float pm0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm0) : "f"(dgr8[0]), "f"(u8[0]));
            float pa0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa0) : "f"(dgr8[1]), "f"(u8[1]), "f"(pm0));
            float pm2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm2) : "f"(dgr8[2]), "f"(u8[2]));
            float pa1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa1) : "f"(dgr8[3]), "f"(u8[3]), "f"(pm2));
            float pm4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm4) : "f"(dgr8[4]), "f"(u8[4]));
            float pa2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa2) : "f"(dgr8[5]), "f"(u8[5]), "f"(pm4));
            float pm6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm6) : "f"(dgr8[6]), "f"(u8[6]));
            float pa3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa3) : "f"(dgr8[7]), "f"(u8[7]), "f"(pm6));
            float pa01;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa01) : "f"(pa0), "f"(pa1));
            float pa23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa23) : "f"(pa2), "f"(pa3));
            float pa;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa) : "f"(pa01), "f"(pa23));
            float nm_t;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t) : "f"(s_a), "f"(pa));
            float _sub_501;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_501) : "f"(s_a), "f"(nm_t));
            float nm_es;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es) : "f"(_sub_501), "f"(pa));
            float _sub_502;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_502) : "f"(pa), "f"(nm_t));
            float nm_ex;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex) : "f"(_sub_502), "f"(s_a));
            float _abs_504;
            asm("abs.f32 %0, %1;" : "=f"(_abs_504) : "f"(s_a));
            float _abs_505;
            asm("abs.f32 %0, %1;" : "=f"(_abs_505) : "f"(pa));
            uint32_t _setp_503_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_503_bits) : "f"(_abs_504), "f"(_abs_505));
            bool _setp_503 = _setp_503_bits != 0;
            float nm_err = ((_setp_503) ? nm_es : nm_ex);
            float _add_506;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_506) : "f"(c_a), "f"(nm_err));
            c_a = _add_506;
            s_a = nm_t;
            float nm_t_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_0) : "f"(s_b), "f"(pb));
            float _sub_507;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_507) : "f"(s_b), "f"(nm_t_0));
            float nm_es_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_0) : "f"(_sub_507), "f"(pb));
            float _sub_508;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_508) : "f"(pb), "f"(nm_t_0));
            float nm_ex_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_0) : "f"(_sub_508), "f"(s_b));
            float _abs_510;
            asm("abs.f32 %0, %1;" : "=f"(_abs_510) : "f"(s_b));
            float _abs_511;
            asm("abs.f32 %0, %1;" : "=f"(_abs_511) : "f"(pb));
            uint32_t _setp_509_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_509_bits) : "f"(_abs_510), "f"(_abs_511));
            bool _setp_509 = _setp_509_bits != 0;
            float nm_err_0 = ((_setp_509) ? nm_es_0 : nm_ex_0);
            float _add_512;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_512) : "f"(c_b), "f"(nm_err_0));
            c_b = _add_512;
            s_b = nm_t_0;
            gacc[0] = s_a;
            gacc[1] = c_a;
            gacc[2] = s_b;
            gacc[3] = c_b;
            off_acc[0] = off_acc[0] + t_hi + t_lo;
            float _tmem_load_18[4];
            tmem_ld_x4(&_tmem_load_18[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * (1 - th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float t_hi_0 = ((th == 1) ? tot : _tmem_load_18[0]);
            float t_lo_0 = ((th == 1) ? _tmem_load_18[0] : tot);
            float offset_0 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_0);
            float sfx_0[8];
            tmem_ld_x8(&sfx_0[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_a_0 = gacc[0];
            float c_a_0 = gacc[1];
            float s_b_0 = gacc[2];
            float c_b_0 = gacc[3];
            float u8_0[8];
            float dgr8_0[8];
            float2 dyg2_4 = add_f32x2(make_float2(offset_0, offset_0), make_float2(sfx_0[0], sfx_0[1]));
            float2 rp_u_3 = add_f32x2(make_float2(raw_t2[0], raw_t2[1]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_513 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_513), "l"(*(const unsigned long long*)&rp_u_3));
            float _abs_516;
            asm("abs.f32 %0, %1;" : "=f"(_abs_516) : "f"(rp_z_3.x));
            float _neg_515;
            asm("neg.f32 %0, %1;" : "=f"(_neg_515) : "f"(_abs_516));
            float _exp_514 = __nv_expf(_neg_515);
            float _abs_519;
            asm("abs.f32 %0, %1;" : "=f"(_abs_519) : "f"(rp_z_3.y));
            float _neg_518;
            asm("neg.f32 %0, %1;" : "=f"(_neg_518) : "f"(_abs_519));
            float _exp_517 = __nv_expf(_neg_518);
            float2 rp_e_3 = make_float2(_exp_514, _exp_517);
            float2 rp_dn_3 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_3);
            float2 rp_den_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_3) : "l"(*(const unsigned long long*)&rp_dn_3), "l"(*(const unsigned long long*)&rp_dn_3));
            float _rcp_520 = approx_rcp(rp_den_3.x);
            float _rcp_521 = approx_rcp(rp_den_3.y);
            float2 _f32x2_mul_b_522 = make_float2(_rcp_520, _rcp_521);
            float2 rp_sp_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_3) : "l"(*(const unsigned long long*)&rp_e_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_522));
            float2 rp_t_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_3) : "l"(*(const unsigned long long*)&dyg2_4), "l"(*(const unsigned long long*)&rp_sp_3));
            float2 _f32x2_mul_b_523 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_3) : "l"(*(const unsigned long long*)&rp_t_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_523));
            u8_0[0] = rp_u_3.x;
            u8_0[1] = rp_u_3.y;
            dgr8_0[0] = rp_dgr_3.x;
            dgr8_0[1] = rp_dgr_3.y;
            float2 dyg2_5 = add_f32x2(make_float2(offset_0, offset_0), make_float2(sfx_0[2], sfx_0[3]));
            float2 rp_u_4 = add_f32x2(make_float2(raw_t2[2], raw_t2[3]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_524 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_524), "l"(*(const unsigned long long*)&rp_u_4));
            float _abs_527;
            asm("abs.f32 %0, %1;" : "=f"(_abs_527) : "f"(rp_z_4.x));
            float _neg_526;
            asm("neg.f32 %0, %1;" : "=f"(_neg_526) : "f"(_abs_527));
            float _exp_525 = __nv_expf(_neg_526);
            float _abs_530;
            asm("abs.f32 %0, %1;" : "=f"(_abs_530) : "f"(rp_z_4.y));
            float _neg_529;
            asm("neg.f32 %0, %1;" : "=f"(_neg_529) : "f"(_abs_530));
            float _exp_528 = __nv_expf(_neg_529);
            float2 rp_e_4 = make_float2(_exp_525, _exp_528);
            float2 rp_dn_4 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_4);
            float2 rp_den_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_4) : "l"(*(const unsigned long long*)&rp_dn_4), "l"(*(const unsigned long long*)&rp_dn_4));
            float _rcp_531 = approx_rcp(rp_den_4.x);
            float _rcp_532 = approx_rcp(rp_den_4.y);
            float2 _f32x2_mul_b_533 = make_float2(_rcp_531, _rcp_532);
            float2 rp_sp_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_4) : "l"(*(const unsigned long long*)&rp_e_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_533));
            float2 rp_t_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_4) : "l"(*(const unsigned long long*)&dyg2_5), "l"(*(const unsigned long long*)&rp_sp_4));
            float2 _f32x2_mul_b_534 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_4) : "l"(*(const unsigned long long*)&rp_t_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_534));
            u8_0[2] = rp_u_4.x;
            u8_0[3] = rp_u_4.y;
            dgr8_0[2] = rp_dgr_4.x;
            dgr8_0[3] = rp_dgr_4.y;
            float2 dyg2_6 = add_f32x2(make_float2(offset_0, offset_0), make_float2(sfx_0[4], sfx_0[5]));
            float2 rp_u_5 = add_f32x2(make_float2(raw_t2[4], raw_t2[5]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_535 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_535), "l"(*(const unsigned long long*)&rp_u_5));
            float _abs_538;
            asm("abs.f32 %0, %1;" : "=f"(_abs_538) : "f"(rp_z_5.x));
            float _neg_537;
            asm("neg.f32 %0, %1;" : "=f"(_neg_537) : "f"(_abs_538));
            float _exp_536 = __nv_expf(_neg_537);
            float _abs_541;
            asm("abs.f32 %0, %1;" : "=f"(_abs_541) : "f"(rp_z_5.y));
            float _neg_540;
            asm("neg.f32 %0, %1;" : "=f"(_neg_540) : "f"(_abs_541));
            float _exp_539 = __nv_expf(_neg_540);
            float2 rp_e_5 = make_float2(_exp_536, _exp_539);
            float2 rp_dn_5 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_5);
            float2 rp_den_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_5) : "l"(*(const unsigned long long*)&rp_dn_5), "l"(*(const unsigned long long*)&rp_dn_5));
            float _rcp_542 = approx_rcp(rp_den_5.x);
            float _rcp_543 = approx_rcp(rp_den_5.y);
            float2 _f32x2_mul_b_544 = make_float2(_rcp_542, _rcp_543);
            float2 rp_sp_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_5) : "l"(*(const unsigned long long*)&rp_e_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_544));
            float2 rp_t_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_5) : "l"(*(const unsigned long long*)&dyg2_6), "l"(*(const unsigned long long*)&rp_sp_5));
            float2 _f32x2_mul_b_545 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_5) : "l"(*(const unsigned long long*)&rp_t_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_545));
            u8_0[4] = rp_u_5.x;
            u8_0[5] = rp_u_5.y;
            dgr8_0[4] = rp_dgr_5.x;
            dgr8_0[5] = rp_dgr_5.y;
            float2 dyg2_7 = add_f32x2(make_float2(offset_0, offset_0), make_float2(sfx_0[6], sfx_0[7]));
            float2 rp_u_6 = add_f32x2(make_float2(raw_t2[6], raw_t2[7]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_546 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_546), "l"(*(const unsigned long long*)&rp_u_6));
            float _abs_549;
            asm("abs.f32 %0, %1;" : "=f"(_abs_549) : "f"(rp_z_6.x));
            float _neg_548;
            asm("neg.f32 %0, %1;" : "=f"(_neg_548) : "f"(_abs_549));
            float _exp_547 = __nv_expf(_neg_548);
            float _abs_552;
            asm("abs.f32 %0, %1;" : "=f"(_abs_552) : "f"(rp_z_6.y));
            float _neg_551;
            asm("neg.f32 %0, %1;" : "=f"(_neg_551) : "f"(_abs_552));
            float _exp_550 = __nv_expf(_neg_551);
            float2 rp_e_6 = make_float2(_exp_547, _exp_550);
            float2 rp_dn_6 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_6);
            float2 rp_den_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_6) : "l"(*(const unsigned long long*)&rp_dn_6), "l"(*(const unsigned long long*)&rp_dn_6));
            float _rcp_553 = approx_rcp(rp_den_6.x);
            float _rcp_554 = approx_rcp(rp_den_6.y);
            float2 _f32x2_mul_b_555 = make_float2(_rcp_553, _rcp_554);
            float2 rp_sp_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_6) : "l"(*(const unsigned long long*)&rp_e_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_555));
            float2 rp_t_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_6) : "l"(*(const unsigned long long*)&dyg2_7), "l"(*(const unsigned long long*)&rp_sp_6));
            float2 _f32x2_mul_b_556 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_6) : "l"(*(const unsigned long long*)&rp_t_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_556));
            u8_0[6] = rp_u_6.x;
            u8_0[7] = rp_u_6.y;
            dgr8_0[6] = rp_dgr_6.x;
            dgr8_0[7] = rp_dgr_6.y;
            int odd_0 = lane & 1;
            unsigned int sel_0 = ((odd_0 == 1) ? 12918 : 21520);
            unsigned int pw_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dgr8_0[_lp*2 + 0], dgr8_0[_lp*2+1 + 0]));
                pw_0[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int pv_4 = __shfl_xor_sync(0xFFFFFFFF, pw_0[0], 1);
            unsigned int pq_4[1];
            uint32_t _prmt_557;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_557) : "r"(pw_0[0]), "r"(pv_4), "r"(sel_0));
            pq_4[0] = _prmt_557;
            int t_q_4 = 32 + 8 * th + odd_0;
            if (t_q_4 < clen) {
                {
                    auto _gst_558 = draw_base + (t_q_4 * rs_g - odd_0);
                    __builtin_assume(__isGlobal(_gst_558));
                    *(reinterpret_cast<unsigned int*>(_gst_558) + (0)) = pq_4[0];
                }
            }
            unsigned int pv_5 = __shfl_xor_sync(0xFFFFFFFF, pw_0[1], 1);
            unsigned int pq_5[1];
            uint32_t _prmt_559;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_559) : "r"(pw_0[1]), "r"(pv_5), "r"(sel_0));
            pq_5[0] = _prmt_559;
            int t_q_5 = 32 + 8 * th + 2 + odd_0;
            if (t_q_5 < clen) {
                {
                    auto _gst_560 = draw_base + (t_q_5 * rs_g - odd_0);
                    __builtin_assume(__isGlobal(_gst_560));
                    *(reinterpret_cast<unsigned int*>(_gst_560) + (0)) = pq_5[0];
                }
            }
            unsigned int pv_6 = __shfl_xor_sync(0xFFFFFFFF, pw_0[2], 1);
            unsigned int pq_6[1];
            uint32_t _prmt_561;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_561) : "r"(pw_0[2]), "r"(pv_6), "r"(sel_0));
            pq_6[0] = _prmt_561;
            int t_q_6 = 32 + 8 * th + 4 + odd_0;
            if (t_q_6 < clen) {
                {
                    auto _gst_562 = draw_base + (t_q_6 * rs_g - odd_0);
                    __builtin_assume(__isGlobal(_gst_562));
                    *(reinterpret_cast<unsigned int*>(_gst_562) + (0)) = pq_6[0];
                }
            }
            unsigned int pv_7 = __shfl_xor_sync(0xFFFFFFFF, pw_0[3], 1);
            unsigned int pq_7[1];
            uint32_t _prmt_563;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_563) : "r"(pw_0[3]), "r"(pv_7), "r"(sel_0));
            pq_7[0] = _prmt_563;
            int t_q_7 = 32 + 8 * th + 6 + odd_0;
            if (t_q_7 < clen) {
                {
                    auto _gst_564 = draw_base + (t_q_7 * rs_g - odd_0);
                    __builtin_assume(__isGlobal(_gst_564));
                    *(reinterpret_cast<unsigned int*>(_gst_564) + (0)) = pq_7[0];
                }
            }
            int t_a_8 = 32 + 8 * th;
            dgr8_0[0] = ((t_a_8 < clen) ? dgr8_0[0] : 0.0f);
            u8_0[0] = ((t_a_8 < clen) ? u8_0[0] : 0.0f);
            int t_a_9 = 32 + 8 * th + 1;
            dgr8_0[1] = ((t_a_9 < clen) ? dgr8_0[1] : 0.0f);
            u8_0[1] = ((t_a_9 < clen) ? u8_0[1] : 0.0f);
            int t_a_10 = 32 + 8 * th + 2;
            dgr8_0[2] = ((t_a_10 < clen) ? dgr8_0[2] : 0.0f);
            u8_0[2] = ((t_a_10 < clen) ? u8_0[2] : 0.0f);
            int t_a_11 = 32 + 8 * th + 3;
            dgr8_0[3] = ((t_a_11 < clen) ? dgr8_0[3] : 0.0f);
            u8_0[3] = ((t_a_11 < clen) ? u8_0[3] : 0.0f);
            int t_a_12 = 32 + 8 * th + 4;
            dgr8_0[4] = ((t_a_12 < clen) ? dgr8_0[4] : 0.0f);
            u8_0[4] = ((t_a_12 < clen) ? u8_0[4] : 0.0f);
            int t_a_13 = 32 + 8 * th + 5;
            dgr8_0[5] = ((t_a_13 < clen) ? dgr8_0[5] : 0.0f);
            u8_0[5] = ((t_a_13 < clen) ? u8_0[5] : 0.0f);
            int t_a_14 = 32 + 8 * th + 6;
            dgr8_0[6] = ((t_a_14 < clen) ? dgr8_0[6] : 0.0f);
            u8_0[6] = ((t_a_14 < clen) ? u8_0[6] : 0.0f);
            int t_a_15 = 32 + 8 * th + 7;
            dgr8_0[7] = ((t_a_15 < clen) ? dgr8_0[7] : 0.0f);
            u8_0[7] = ((t_a_15 < clen) ? u8_0[7] : 0.0f);
            float pb01_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb01_0) : "f"(dgr8_0[0]), "f"(dgr8_0[1]));
            float pb23_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb23_0) : "f"(dgr8_0[2]), "f"(dgr8_0[3]));
            float pb45_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb45_0) : "f"(dgr8_0[4]), "f"(dgr8_0[5]));
            float pb67_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb67_0) : "f"(dgr8_0[6]), "f"(dgr8_0[7]));
            float pb03_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb03_0) : "f"(pb01_0), "f"(pb23_0));
            float pb47_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb47_0) : "f"(pb45_0), "f"(pb67_0));
            float pb_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb_0) : "f"(pb03_0), "f"(pb47_0));
            float pm0_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm0_0) : "f"(dgr8_0[0]), "f"(u8_0[0]));
            float pa0_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa0_0) : "f"(dgr8_0[1]), "f"(u8_0[1]), "f"(pm0_0));
            float pm2_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm2_0) : "f"(dgr8_0[2]), "f"(u8_0[2]));
            float pa1_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa1_0) : "f"(dgr8_0[3]), "f"(u8_0[3]), "f"(pm2_0));
            float pm4_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm4_0) : "f"(dgr8_0[4]), "f"(u8_0[4]));
            float pa2_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa2_0) : "f"(dgr8_0[5]), "f"(u8_0[5]), "f"(pm4_0));
            float pm6_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm6_0) : "f"(dgr8_0[6]), "f"(u8_0[6]));
            float pa3_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa3_0) : "f"(dgr8_0[7]), "f"(u8_0[7]), "f"(pm6_0));
            float pa01_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa01_0) : "f"(pa0_0), "f"(pa1_0));
            float pa23_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa23_0) : "f"(pa2_0), "f"(pa3_0));
            float pa_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa_0) : "f"(pa01_0), "f"(pa23_0));
            float nm_t_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_1) : "f"(s_a_0), "f"(pa_0));
            float _sub_565;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(s_a_0), "f"(nm_t_1));
            float nm_es_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_1) : "f"(_sub_565), "f"(pa_0));
            float _sub_566;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_566) : "f"(pa_0), "f"(nm_t_1));
            float nm_ex_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_1) : "f"(_sub_566), "f"(s_a_0));
            float _abs_568;
            asm("abs.f32 %0, %1;" : "=f"(_abs_568) : "f"(s_a_0));
            float _abs_569;
            asm("abs.f32 %0, %1;" : "=f"(_abs_569) : "f"(pa_0));
            uint32_t _setp_567_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_567_bits) : "f"(_abs_568), "f"(_abs_569));
            bool _setp_567 = _setp_567_bits != 0;
            float nm_err_1 = ((_setp_567) ? nm_es_1 : nm_ex_1);
            float _add_570;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(c_a_0), "f"(nm_err_1));
            c_a_0 = _add_570;
            s_a_0 = nm_t_1;
            float nm_t_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_2) : "f"(s_b_0), "f"(pb_0));
            float _sub_571;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_571) : "f"(s_b_0), "f"(nm_t_2));
            float nm_es_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_2) : "f"(_sub_571), "f"(pb_0));
            float _sub_572;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_572) : "f"(pb_0), "f"(nm_t_2));
            float nm_ex_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_2) : "f"(_sub_572), "f"(s_b_0));
            float _abs_574;
            asm("abs.f32 %0, %1;" : "=f"(_abs_574) : "f"(s_b_0));
            float _abs_575;
            asm("abs.f32 %0, %1;" : "=f"(_abs_575) : "f"(pb_0));
            uint32_t _setp_573_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_573_bits) : "f"(_abs_574), "f"(_abs_575));
            bool _setp_573 = _setp_573_bits != 0;
            float nm_err_2 = ((_setp_573) ? nm_es_2 : nm_ex_2);
            float _add_576;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_576) : "f"(c_b_0), "f"(nm_err_2));
            c_b_0 = _add_576;
            s_b_0 = nm_t_2;
            gacc[0] = s_a_0;
            gacc[1] = c_a_0;
            gacc[2] = s_b_0;
            gacc[3] = c_b_0;
            off_acc[0] = off_acc[0] + t_hi_0 + t_lo_0;
            {
                uint32_t _addr_577 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_577), "f"(off_acc[0]) : "memory");
            }
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th), gacc);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float raw_t1[8];
            long long rb_1 = ext0 * g_raw_row_stride + (long long)hv * g_raw_head_stride + (long long)d_lane;
            int t_c_16 = ((16 + 8 * th < clen) ? 16 + 8 * th : clen - 1);
            raw_t1[0] = (float)g_raw[rb_1 + (long long)t_c_16 * g_raw_row_stride];
            int t_c_17 = ((16 + 8 * th + 1 < clen) ? 16 + 8 * th + 1 : clen - 1);
            raw_t1[1] = (float)g_raw[rb_1 + (long long)t_c_17 * g_raw_row_stride];
            int t_c_18 = ((16 + 8 * th + 2 < clen) ? 16 + 8 * th + 2 : clen - 1);
            raw_t1[2] = (float)g_raw[rb_1 + (long long)t_c_18 * g_raw_row_stride];
            int t_c_19 = ((16 + 8 * th + 3 < clen) ? 16 + 8 * th + 3 : clen - 1);
            raw_t1[3] = (float)g_raw[rb_1 + (long long)t_c_19 * g_raw_row_stride];
            int t_c_20 = ((16 + 8 * th + 4 < clen) ? 16 + 8 * th + 4 : clen - 1);
            raw_t1[4] = (float)g_raw[rb_1 + (long long)t_c_20 * g_raw_row_stride];
            int t_c_21 = ((16 + 8 * th + 5 < clen) ? 16 + 8 * th + 5 : clen - 1);
            raw_t1[5] = (float)g_raw[rb_1 + (long long)t_c_21 * g_raw_row_stride];
            int t_c_22 = ((16 + 8 * th + 6 < clen) ? 16 + 8 * th + 6 : clen - 1);
            raw_t1[6] = (float)g_raw[rb_1 + (long long)t_c_22 * g_raw_row_stride];
            int t_c_23 = ((16 + 8 * th + 7 < clen) ? 16 + 8 * th + 7 : clen - 1);
            raw_t1[7] = (float)g_raw[rb_1 + (long long)t_c_23 * g_raw_row_stride];
            mbarrier_wait(p1b_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + (2 + th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_276[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_276[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1 = r_276[0];
            float gc_1 = gn_1 - 21.640425f;
            int eb_1 = 81920 + th * -32768;
            int ew_3 = eb_1 >> 2;
            int qbase_1 = pool_f32_addr + (unsigned int)(eb_1 + 12288 + dh_lane * 1024);
            int sb8_1 = 16 + 8 * th;
            int gb8_1 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1 * 128);
            int kb8_1 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1 * 128);
            int c_qa_0 = 64 + 8 * th;
            int c_ka_0 = 80 + 8 * th;
            int c_qd_0 = 96 + 8 * th;
            int c_kd_0 = 112 + 8 * th;
            int c_dk_0 = 16 + 8 * th;
            int c_bp_0 = c_ka_0;
            float s_run_1[1];
            s_run_1[0] = 0.0f;
            #pragma unroll 1
            for (int k_1 = 0; k_1 < 8; k_1++) {
                int j_3 = 7 - k_1;
                float a_qd_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_0 + j_3)));
                float a_kd_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd_0 + j_3)));
                float a_dk_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk_0 + j_3)));
                float a_qa_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qa_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qa_0 + j_3)));
                float a_ka_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_ka_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_ka_0 + j_3)));
                float btr_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr_1[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_1 + j_3))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t_1 = sb8_1 + j_3;
                float r_277[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_277[0])) : "r"((gb8_1 + (lane_1 / 32 * 8192 + 128 * j_3 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_3 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t_1 = r_277[0];
                unsigned int r_278[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_278[0])) : "r"((kb8_1 + (dcol / 64 * 8192 + 128 * j_3 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_3 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t_1 = __uint_as_float(r_278[0] << 16);
                unsigned int r_279[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_279[0])) : "r"((qbase_1 + (dcol / 64 * 1024 + 128 * j_3 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_3 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t_1 = __uint_as_float(r_279[0] << 16);
                float q_m_1 = ((t_1 >= clen) ? 0.0f : q_t_1);
                float dqf_1 = pool_f32[ew_3 + 128 * j_3 + d_lane];
                float dkf_1 = pool_f32[ew_3 + (1024 + 128 * j_3) + d_lane];
                float dgf_1 = pool_f32[ew_3 + (2048 + 128 * j_3) + d_lane];
                float bt_t_1 = btr_1[0];
                float e_dg_1 = approx_exp2(g_t_1 - gc_1);
                float e_off_1 = approx_exp2(g_t_1 - gn_1);
                float dq2_1 = __fmaf_rn(a_qa_1[0], e_off_1, a_qd_1[0] * e_dg_1);
                float dk2_1 = __fmaf_rn(a_ka_1[0], e_off_1, a_kd_1[0] * e_dg_1);
                float dbv_1[1];
                dbv_1[0] = dk2_1 * k_t_1;
                float dk2b_1 = dk2_1 * bt_t_1;
                float dq_o_1 = dq2_1 + dqf_1;
                float dk_o_1 = dk2b_1 + dkf_1 + a_dk_1[0];
                float qe_8 = q_m_1 * e_dg_1;
                float kp_20 = k_t_1 * bt_t_1;
                __nv_bfloat16 kb_8 = (__nv_bfloat16)kp_20;
                float kbe_8 = (float)kb_8 * e_dg_1;
                float cr_8 = __fmaf_rn(__uint_as_float(__as_u32(qe_8) & 4294959104u) - qe_8, a_qd_1[0], (__uint_as_float(__as_u32(kbe_8) & 4294959104u) - kp_20 * e_dg_1) * a_kd_1[0]);
                float dg_o_1 = __fmaf_rn(q_m_1, dq2_1, __fmaf_rn(dk2b_1 - a_dk_1[0], k_t_1, cr_8)) + dgf_1;
                s_run_1[0] = s_run_1[0] + dg_o_1;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_0 + j_3)), "f"(s_run_1[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp_0 + j_3)), "f"(dbv_1[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                pool_f32[(81920 - 32768 * th >> 2) + (128 * j_3 + d_lane)] = dq_o_1;
                pool_f32[(81920 - 32768 * th >> 2) + (1024 + 128 * j_3 + d_lane)] = dk_o_1;
            }
            float tot_1 = s_run_1[0];
            float dbp_1[8];
            tmem_ld_x8(&dbp_1[0], taddr + (unsigned int)row_base + (unsigned int)c_bp_0);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_1 = lane_1 >> 4 & 1;
            int b3_1 = lane_1 >> 3 & 1;
            int b2_1 = lane_1 >> 2 & 1;
            float v4_1[4];
            float send_4 = ((b4_1 == 0) ? dbp_1[4] : dbp_1[0]);
            float _shfl_xor_sync_578 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
            v4_1[0] = ((b4_1 == 0) ? dbp_1[0] : dbp_1[4]) + _shfl_xor_sync_578;
            float send_5 = ((b4_1 == 0) ? dbp_1[5] : dbp_1[1]);
            float _shfl_xor_sync_579 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
            v4_1[1] = ((b4_1 == 0) ? dbp_1[1] : dbp_1[5]) + _shfl_xor_sync_579;
            float send_6 = ((b4_1 == 0) ? dbp_1[6] : dbp_1[2]);
            float _shfl_xor_sync_580 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
            v4_1[2] = ((b4_1 == 0) ? dbp_1[2] : dbp_1[6]) + _shfl_xor_sync_580;
            float send_7 = ((b4_1 == 0) ? dbp_1[7] : dbp_1[3]);
            float _shfl_xor_sync_581 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
            v4_1[3] = ((b4_1 == 0) ? dbp_1[3] : dbp_1[7]) + _shfl_xor_sync_581;
            float v2_16[2];
            float send2_2 = ((b3_1 == 0) ? v4_1[2] : v4_1[0]);
            float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, send2_2, 8);
            v2_16[0] = ((b3_1 == 0) ? v4_1[0] : v4_1[2]) + _shfl_xor_sync_582;
            float send2_3 = ((b3_1 == 0) ? v4_1[3] : v4_1[1]);
            float _shfl_xor_sync_583 = __shfl_xor_sync(0xFFFFFFFF, send2_3, 8);
            v2_16[1] = ((b3_1 == 0) ? v4_1[1] : v4_1[3]) + _shfl_xor_sync_583;
            float send1_0 = ((b2_1 == 0) ? v2_16[1] : v2_16[0]);
            float _shfl_xor_sync_584 = __shfl_xor_sync(0xFFFFFFFF, send1_0, 4);
            float v1_1 = ((b2_1 == 0) ? v2_16[0] : v2_16[1]) + _shfl_xor_sync_584;
            float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
            v1_1 = v1_1 + _shfl_xor_sync_585;
            float _shfl_xor_sync_586 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
            v1_1 = v1_1 + _shfl_xor_sync_586;
            asm volatile("tcgen05.fence::before_thread_sync;");
            int ew_4 = 81920 - 32768 * th >> 2;
            float nz_1 = __uint_as_float((unsigned int)num_heads >> 31 | 2147483648u);
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int i_2 = 0; i_2 < 2; i_2++) {
                int j_4 = qd + 4 * i_2;
                bool live_2 = 16 + 8 * th + j_4 < clen;
                int _min_587 = ((16 + 8 * th + j_4) < (clen - 1) ? (16 + 8 * th + j_4) : (clen - 1));
                long long row_2 = ext0 + (long long)_min_587;
                float gq_2[4];
                float gk_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gq_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq_2[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_4 + 128 * j_4 + 4 * lane_1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_2[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_4 + 1024 + 128 * j_4 + 4 * lane_1))));
                if (i_2 == 1) {
                    mbarrier_arrive(epi_empty_addr + (2 + th) * 8);
                }
                float xq_2[4];
                float xk_2[4];
                float rq_2[1];
                float rk_2[1];
                {
                    uint2 _vld_588;
                    _vld_588 = *reinterpret_cast<const uint2*>(q_raw + (row_2 * (long long)q_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_588 = reinterpret_cast<uint32_t*>(&_vld_588);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xq_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_588[_pair]) << 16);
                        (&xq_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_588[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rq_2[0] = *reinterpret_cast<const float*>(qk_rstd + ((0 + row_2) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                {
                    uint2 _vld_589;
                    _vld_589 = *reinterpret_cast<const uint2*>(k_raw + (row_2 * (long long)k_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_589 = reinterpret_cast<uint32_t*>(&_vld_589);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xk_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_589[_pair]) << 16);
                        (&xk_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_589[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rk_2[0] = *reinterpret_cast<const float*>(qk_rstd + (((long long)num_tokens + row_2) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                long long dxo_2 = row_2 * (long long)qrs + (long long)(qk_head * 128 + 4 * lane_1);
                float y_2[4];
                float p_2[4];
                y_2[0] = mul_f32_rn_noftz(xq_2[0], rq_2[0]);
                float _fma_590;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_590) : "f"(gq_2[0]), "f"(y_2[0]), "f"(nz_1));
                p_2[0] = _fma_590;
                y_2[1] = mul_f32_rn_noftz(xq_2[1], rq_2[0]);
                float _fma_591;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_591) : "f"(gq_2[1]), "f"(y_2[1]), "f"(nz_1));
                p_2[1] = _fma_591;
                y_2[2] = mul_f32_rn_noftz(xq_2[2], rq_2[0]);
                float _fma_592;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_592) : "f"(gq_2[2]), "f"(y_2[2]), "f"(nz_1));
                p_2[2] = _fma_592;
                y_2[3] = mul_f32_rn_noftz(xq_2[3], rq_2[0]);
                float _fma_593;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(gq_2[3]), "f"(y_2[3]), "f"(nz_1));
                p_2[3] = _fma_593;
                float _shfl_xor_sync_595 = __shfl_xor_sync(0xFFFFFFFF, p_2[0], 4);
                float _add_594;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_594) : "f"(p_2[0]), "f"(_shfl_xor_sync_595));
                p_2[0] = _add_594;
                float _shfl_xor_sync_597 = __shfl_xor_sync(0xFFFFFFFF, p_2[1], 4);
                float _add_596;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_596) : "f"(p_2[1]), "f"(_shfl_xor_sync_597));
                p_2[1] = _add_596;
                float _shfl_xor_sync_599 = __shfl_xor_sync(0xFFFFFFFF, p_2[2], 4);
                float _add_598;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_598) : "f"(p_2[2]), "f"(_shfl_xor_sync_599));
                p_2[2] = _add_598;
                float _shfl_xor_sync_601 = __shfl_xor_sync(0xFFFFFFFF, p_2[3], 4);
                float _add_600;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_600) : "f"(p_2[3]), "f"(_shfl_xor_sync_601));
                p_2[3] = _add_600;
                float _shfl_xor_sync_603 = __shfl_xor_sync(0xFFFFFFFF, p_2[0], 2);
                float _add_602;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_602) : "f"(p_2[0]), "f"(_shfl_xor_sync_603));
                p_2[0] = _add_602;
                float _shfl_xor_sync_605 = __shfl_xor_sync(0xFFFFFFFF, p_2[1], 2);
                float _add_604;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_604) : "f"(p_2[1]), "f"(_shfl_xor_sync_605));
                p_2[1] = _add_604;
                float _shfl_xor_sync_607 = __shfl_xor_sync(0xFFFFFFFF, p_2[2], 2);
                float _add_606;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_606) : "f"(p_2[2]), "f"(_shfl_xor_sync_607));
                p_2[2] = _add_606;
                float _shfl_xor_sync_609 = __shfl_xor_sync(0xFFFFFFFF, p_2[3], 2);
                float _add_608;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_608) : "f"(p_2[3]), "f"(_shfl_xor_sync_609));
                p_2[3] = _add_608;
                float _shfl_xor_sync_611 = __shfl_xor_sync(0xFFFFFFFF, p_2[0], 1);
                float _add_610;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_610) : "f"(p_2[0]), "f"(_shfl_xor_sync_611));
                p_2[0] = _add_610;
                float _shfl_xor_sync_613 = __shfl_xor_sync(0xFFFFFFFF, p_2[1], 1);
                float _add_612;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_612) : "f"(p_2[1]), "f"(_shfl_xor_sync_613));
                p_2[1] = _add_612;
                float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, p_2[2], 1);
                float _add_614;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(p_2[2]), "f"(_shfl_xor_sync_615));
                p_2[2] = _add_614;
                float _shfl_xor_sync_617 = __shfl_xor_sync(0xFFFFFFFF, p_2[3], 1);
                float _add_616;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_616) : "f"(p_2[3]), "f"(_shfl_xor_sync_617));
                p_2[3] = _add_616;
                float _add_618;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_618) : "f"(p_2[0]), "f"(p_2[2]));
                float _add_619;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_619) : "f"(p_2[1]), "f"(p_2[3]));
                float sq_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_2) : "f"(_add_618), "f"(_add_619));
                float _shfl_sync_622 = __shfl_sync(0xFFFFFFFF, sq_2, 0);
                float _shfl_sync_623 = __shfl_sync(0xFFFFFFFF, sq_2, 8);
                float _add_621;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_621) : "f"(_shfl_sync_622), "f"(_shfl_sync_623));
                float _shfl_sync_624 = __shfl_sync(0xFFFFFFFF, sq_2, 16);
                float _add_620;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_620) : "f"(_add_621), "f"(_shfl_sync_624));
                float _shfl_sync_625 = __shfl_sync(0xFFFFFFFF, sq_2, 24);
                float dot_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_2) : "f"(_add_620), "f"(_shfl_sync_625));
                float o_2[4];
                float _fma_627;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(y_2[0]), "f"(dot_2), "f"(nz_1));
                float _sub_626;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_626) : "f"(gq_2[0]), "f"(_fma_627));
                o_2[0] = mul_f32_rn_noftz(rq_2[0], _sub_626);
                float _fma_629;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(y_2[1]), "f"(dot_2), "f"(nz_1));
                float _sub_628;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_628) : "f"(gq_2[1]), "f"(_fma_629));
                o_2[1] = mul_f32_rn_noftz(rq_2[0], _sub_628);
                float _fma_631;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(y_2[2]), "f"(dot_2), "f"(nz_1));
                float _sub_630;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_630) : "f"(gq_2[2]), "f"(_fma_631));
                o_2[2] = mul_f32_rn_noftz(rq_2[0], _sub_630);
                float _fma_633;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_633) : "f"(y_2[3]), "f"(dot_2), "f"(nz_1));
                float _sub_632;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_632) : "f"(gq_2[3]), "f"(_fma_633));
                o_2[3] = mul_f32_rn_noftz(rq_2[0], _sub_632);
                unsigned int ob_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_2[_lp*2 + 0], o_2[_lp*2+1 + 0]));
                    ob_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dq_ext + dxo_2), "r"(ob_2[(0) + 0]), "r"(ob_2[(0) + 1]), "r"((unsigned int)(live_2)) : "memory");
                float y_2_1[4];
                float p_2_1[4];
                y_2_1[0] = mul_f32_rn_noftz(xk_2[0], rk_2[0]);
                float _fma_634;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_634) : "f"(gk_2[0]), "f"(y_2_1[0]), "f"(nz_1));
                p_2_1[0] = _fma_634;
                y_2_1[1] = mul_f32_rn_noftz(xk_2[1], rk_2[0]);
                float _fma_635;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_635) : "f"(gk_2[1]), "f"(y_2_1[1]), "f"(nz_1));
                p_2_1[1] = _fma_635;
                y_2_1[2] = mul_f32_rn_noftz(xk_2[2], rk_2[0]);
                float _fma_636;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_636) : "f"(gk_2[2]), "f"(y_2_1[2]), "f"(nz_1));
                p_2_1[2] = _fma_636;
                y_2_1[3] = mul_f32_rn_noftz(xk_2[3], rk_2[0]);
                float _fma_637;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_637) : "f"(gk_2[3]), "f"(y_2_1[3]), "f"(nz_1));
                p_2_1[3] = _fma_637;
                float _shfl_xor_sync_639 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[0], 4);
                float _add_638;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_638) : "f"(p_2_1[0]), "f"(_shfl_xor_sync_639));
                p_2_1[0] = _add_638;
                float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[1], 4);
                float _add_640;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_640) : "f"(p_2_1[1]), "f"(_shfl_xor_sync_641));
                p_2_1[1] = _add_640;
                float _shfl_xor_sync_643 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[2], 4);
                float _add_642;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_642) : "f"(p_2_1[2]), "f"(_shfl_xor_sync_643));
                p_2_1[2] = _add_642;
                float _shfl_xor_sync_645 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[3], 4);
                float _add_644;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(p_2_1[3]), "f"(_shfl_xor_sync_645));
                p_2_1[3] = _add_644;
                float _shfl_xor_sync_647 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[0], 2);
                float _add_646;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_646) : "f"(p_2_1[0]), "f"(_shfl_xor_sync_647));
                p_2_1[0] = _add_646;
                float _shfl_xor_sync_649 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[1], 2);
                float _add_648;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_648) : "f"(p_2_1[1]), "f"(_shfl_xor_sync_649));
                p_2_1[1] = _add_648;
                float _shfl_xor_sync_651 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[2], 2);
                float _add_650;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_650) : "f"(p_2_1[2]), "f"(_shfl_xor_sync_651));
                p_2_1[2] = _add_650;
                float _shfl_xor_sync_653 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[3], 2);
                float _add_652;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_652) : "f"(p_2_1[3]), "f"(_shfl_xor_sync_653));
                p_2_1[3] = _add_652;
                float _shfl_xor_sync_655 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[0], 1);
                float _add_654;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(p_2_1[0]), "f"(_shfl_xor_sync_655));
                p_2_1[0] = _add_654;
                float _shfl_xor_sync_657 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[1], 1);
                float _add_656;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_656) : "f"(p_2_1[1]), "f"(_shfl_xor_sync_657));
                p_2_1[1] = _add_656;
                float _shfl_xor_sync_659 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[2], 1);
                float _add_658;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(p_2_1[2]), "f"(_shfl_xor_sync_659));
                p_2_1[2] = _add_658;
                float _shfl_xor_sync_661 = __shfl_xor_sync(0xFFFFFFFF, p_2_1[3], 1);
                float _add_660;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_660) : "f"(p_2_1[3]), "f"(_shfl_xor_sync_661));
                p_2_1[3] = _add_660;
                float _add_662;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_662) : "f"(p_2_1[0]), "f"(p_2_1[2]));
                float _add_663;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_663) : "f"(p_2_1[1]), "f"(p_2_1[3]));
                float sq_2_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_2_1) : "f"(_add_662), "f"(_add_663));
                float _shfl_sync_666 = __shfl_sync(0xFFFFFFFF, sq_2_1, 0);
                float _shfl_sync_667 = __shfl_sync(0xFFFFFFFF, sq_2_1, 8);
                float _add_665;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_665) : "f"(_shfl_sync_666), "f"(_shfl_sync_667));
                float _shfl_sync_668 = __shfl_sync(0xFFFFFFFF, sq_2_1, 16);
                float _add_664;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_664) : "f"(_add_665), "f"(_shfl_sync_668));
                float _shfl_sync_669 = __shfl_sync(0xFFFFFFFF, sq_2_1, 24);
                float dot_2_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_2_1) : "f"(_add_664), "f"(_shfl_sync_669));
                float o_2_1[4];
                float _fma_671;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_671) : "f"(y_2_1[0]), "f"(dot_2_1), "f"(nz_1));
                float _sub_670;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_670) : "f"(gk_2[0]), "f"(_fma_671));
                o_2_1[0] = mul_f32_rn_noftz(rk_2[0], _sub_670);
                float _fma_673;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_673) : "f"(y_2_1[1]), "f"(dot_2_1), "f"(nz_1));
                float _sub_672;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_672) : "f"(gk_2[1]), "f"(_fma_673));
                o_2_1[1] = mul_f32_rn_noftz(rk_2[0], _sub_672);
                float _fma_675;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_675) : "f"(y_2_1[2]), "f"(dot_2_1), "f"(nz_1));
                float _sub_674;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_674) : "f"(gk_2[2]), "f"(_fma_675));
                o_2_1[2] = mul_f32_rn_noftz(rk_2[0], _sub_674);
                float _fma_677;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_677) : "f"(y_2_1[3]), "f"(dot_2_1), "f"(nz_1));
                float _sub_676;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_676) : "f"(gk_2[3]), "f"(_fma_677));
                o_2_1[3] = mul_f32_rn_noftz(rk_2[0], _sub_676);
                unsigned int ob_2_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_2_1[_lp*2 + 0], o_2_1[_lp*2+1 + 0]));
                    ob_2_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dk_ext + dxo_2), "r"(ob_2_1[(0) + 0]), "r"(ob_2_1[(0) + 1]), "r"((unsigned int)(live_2)) : "memory");
            }
            tmem_ld_x8(&gacc[0], taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_678 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_678), "f"(tot_1) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (qd == 0) {
                asm volatile("barrier.sync 2, 64;" ::: "memory");
            }
            if (qd == 1) {
                asm volatile("barrier.sync 3, 64;" ::: "memory");
            }
            if (qd == 2) {
                asm volatile("barrier.sync 4, 64;" ::: "memory");
            }
            if (qd == 3) {
                asm volatile("barrier.sync 5, 64;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            float t_other_0[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_other_0[0])) : "r"(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + (1 - th) * 32 + lane_1))));
            float t_hi_1 = ((th == 1) ? tot_1 : t_other_0[0]);
            float t_lo_1 = ((th == 1) ? t_other_0[0] : tot_1);
            float offset_1 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1);
            float sfx_1[8];
            tmem_ld_x8(&sfx_1[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_a_1 = gacc[0];
            float c_a_1 = gacc[1];
            float s_b_1 = gacc[2];
            float c_b_1 = gacc[3];
            float u8_1[8];
            float dgr8_1[8];
            float2 dyg2_8 = add_f32x2(make_float2(offset_1, offset_1), make_float2(sfx_1[0], sfx_1[1]));
            float2 rp_u_7 = add_f32x2(make_float2(raw_t1[0], raw_t1[1]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_679 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_679), "l"(*(const unsigned long long*)&rp_u_7));
            float _abs_682;
            asm("abs.f32 %0, %1;" : "=f"(_abs_682) : "f"(rp_z_7.x));
            float _neg_681;
            asm("neg.f32 %0, %1;" : "=f"(_neg_681) : "f"(_abs_682));
            float _exp_680 = __nv_expf(_neg_681);
            float _abs_685;
            asm("abs.f32 %0, %1;" : "=f"(_abs_685) : "f"(rp_z_7.y));
            float _neg_684;
            asm("neg.f32 %0, %1;" : "=f"(_neg_684) : "f"(_abs_685));
            float _exp_683 = __nv_expf(_neg_684);
            float2 rp_e_7 = make_float2(_exp_680, _exp_683);
            float2 rp_dn_7 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_7);
            float2 rp_den_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_7) : "l"(*(const unsigned long long*)&rp_dn_7), "l"(*(const unsigned long long*)&rp_dn_7));
            float _rcp_686 = approx_rcp(rp_den_7.x);
            float _rcp_687 = approx_rcp(rp_den_7.y);
            float2 _f32x2_mul_b_688 = make_float2(_rcp_686, _rcp_687);
            float2 rp_sp_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_7) : "l"(*(const unsigned long long*)&rp_e_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_688));
            float2 rp_t_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_7) : "l"(*(const unsigned long long*)&dyg2_8), "l"(*(const unsigned long long*)&rp_sp_7));
            float2 _f32x2_mul_b_689 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_7) : "l"(*(const unsigned long long*)&rp_t_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_689));
            u8_1[0] = rp_u_7.x;
            u8_1[1] = rp_u_7.y;
            dgr8_1[0] = rp_dgr_7.x;
            dgr8_1[1] = rp_dgr_7.y;
            float2 dyg2_9 = add_f32x2(make_float2(offset_1, offset_1), make_float2(sfx_1[2], sfx_1[3]));
            float2 rp_u_8 = add_f32x2(make_float2(raw_t1[2], raw_t1[3]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_690 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_690), "l"(*(const unsigned long long*)&rp_u_8));
            float _abs_693;
            asm("abs.f32 %0, %1;" : "=f"(_abs_693) : "f"(rp_z_8.x));
            float _neg_692;
            asm("neg.f32 %0, %1;" : "=f"(_neg_692) : "f"(_abs_693));
            float _exp_691 = __nv_expf(_neg_692);
            float _abs_696;
            asm("abs.f32 %0, %1;" : "=f"(_abs_696) : "f"(rp_z_8.y));
            float _neg_695;
            asm("neg.f32 %0, %1;" : "=f"(_neg_695) : "f"(_abs_696));
            float _exp_694 = __nv_expf(_neg_695);
            float2 rp_e_8 = make_float2(_exp_691, _exp_694);
            float2 rp_dn_8 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_8);
            float2 rp_den_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_8) : "l"(*(const unsigned long long*)&rp_dn_8), "l"(*(const unsigned long long*)&rp_dn_8));
            float _rcp_697 = approx_rcp(rp_den_8.x);
            float _rcp_698 = approx_rcp(rp_den_8.y);
            float2 _f32x2_mul_b_699 = make_float2(_rcp_697, _rcp_698);
            float2 rp_sp_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_8) : "l"(*(const unsigned long long*)&rp_e_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_699));
            float2 rp_t_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_8) : "l"(*(const unsigned long long*)&dyg2_9), "l"(*(const unsigned long long*)&rp_sp_8));
            float2 _f32x2_mul_b_700 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_8) : "l"(*(const unsigned long long*)&rp_t_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_700));
            u8_1[2] = rp_u_8.x;
            u8_1[3] = rp_u_8.y;
            dgr8_1[2] = rp_dgr_8.x;
            dgr8_1[3] = rp_dgr_8.y;
            float2 dyg2_10 = add_f32x2(make_float2(offset_1, offset_1), make_float2(sfx_1[4], sfx_1[5]));
            float2 rp_u_9 = add_f32x2(make_float2(raw_t1[4], raw_t1[5]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_701 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_701), "l"(*(const unsigned long long*)&rp_u_9));
            float _abs_704;
            asm("abs.f32 %0, %1;" : "=f"(_abs_704) : "f"(rp_z_9.x));
            float _neg_703;
            asm("neg.f32 %0, %1;" : "=f"(_neg_703) : "f"(_abs_704));
            float _exp_702 = __nv_expf(_neg_703);
            float _abs_707;
            asm("abs.f32 %0, %1;" : "=f"(_abs_707) : "f"(rp_z_9.y));
            float _neg_706;
            asm("neg.f32 %0, %1;" : "=f"(_neg_706) : "f"(_abs_707));
            float _exp_705 = __nv_expf(_neg_706);
            float2 rp_e_9 = make_float2(_exp_702, _exp_705);
            float2 rp_dn_9 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_9);
            float2 rp_den_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_9) : "l"(*(const unsigned long long*)&rp_dn_9), "l"(*(const unsigned long long*)&rp_dn_9));
            float _rcp_708 = approx_rcp(rp_den_9.x);
            float _rcp_709 = approx_rcp(rp_den_9.y);
            float2 _f32x2_mul_b_710 = make_float2(_rcp_708, _rcp_709);
            float2 rp_sp_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_9) : "l"(*(const unsigned long long*)&rp_e_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_710));
            float2 rp_t_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_9) : "l"(*(const unsigned long long*)&dyg2_10), "l"(*(const unsigned long long*)&rp_sp_9));
            float2 _f32x2_mul_b_711 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_9) : "l"(*(const unsigned long long*)&rp_t_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_711));
            u8_1[4] = rp_u_9.x;
            u8_1[5] = rp_u_9.y;
            dgr8_1[4] = rp_dgr_9.x;
            dgr8_1[5] = rp_dgr_9.y;
            float2 dyg2_11 = add_f32x2(make_float2(offset_1, offset_1), make_float2(sfx_1[6], sfx_1[7]));
            float2 rp_u_10 = add_f32x2(make_float2(raw_t1[6], raw_t1[7]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_712 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_712), "l"(*(const unsigned long long*)&rp_u_10));
            float _abs_715;
            asm("abs.f32 %0, %1;" : "=f"(_abs_715) : "f"(rp_z_10.x));
            float _neg_714;
            asm("neg.f32 %0, %1;" : "=f"(_neg_714) : "f"(_abs_715));
            float _exp_713 = __nv_expf(_neg_714);
            float _abs_718;
            asm("abs.f32 %0, %1;" : "=f"(_abs_718) : "f"(rp_z_10.y));
            float _neg_717;
            asm("neg.f32 %0, %1;" : "=f"(_neg_717) : "f"(_abs_718));
            float _exp_716 = __nv_expf(_neg_717);
            float2 rp_e_10 = make_float2(_exp_713, _exp_716);
            float2 rp_dn_10 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_10);
            float2 rp_den_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_10) : "l"(*(const unsigned long long*)&rp_dn_10), "l"(*(const unsigned long long*)&rp_dn_10));
            float _rcp_719 = approx_rcp(rp_den_10.x);
            float _rcp_720 = approx_rcp(rp_den_10.y);
            float2 _f32x2_mul_b_721 = make_float2(_rcp_719, _rcp_720);
            float2 rp_sp_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_10) : "l"(*(const unsigned long long*)&rp_e_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_721));
            float2 rp_t_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_10) : "l"(*(const unsigned long long*)&dyg2_11), "l"(*(const unsigned long long*)&rp_sp_10));
            float2 _f32x2_mul_b_722 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_10) : "l"(*(const unsigned long long*)&rp_t_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_722));
            u8_1[6] = rp_u_10.x;
            u8_1[7] = rp_u_10.y;
            dgr8_1[6] = rp_dgr_10.x;
            dgr8_1[7] = rp_dgr_10.y;
            int odd_1 = lane & 1;
            unsigned int sel_1 = ((odd_1 == 1) ? 12918 : 21520);
            unsigned int pw_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dgr8_1[_lp*2 + 0], dgr8_1[_lp*2+1 + 0]));
                pw_1[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int pv_8 = __shfl_xor_sync(0xFFFFFFFF, pw_1[0], 1);
            unsigned int pq_8[1];
            uint32_t _prmt_723;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_723) : "r"(pw_1[0]), "r"(pv_8), "r"(sel_1));
            pq_8[0] = _prmt_723;
            int t_q_8 = 16 + 8 * th + odd_1;
            if (t_q_8 < clen) {
                {
                    auto _gst_724 = draw_base + (t_q_8 * rs_g - odd_1);
                    __builtin_assume(__isGlobal(_gst_724));
                    *(reinterpret_cast<unsigned int*>(_gst_724) + (0)) = pq_8[0];
                }
            }
            unsigned int pv_9 = __shfl_xor_sync(0xFFFFFFFF, pw_1[1], 1);
            unsigned int pq_9[1];
            uint32_t _prmt_725;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_725) : "r"(pw_1[1]), "r"(pv_9), "r"(sel_1));
            pq_9[0] = _prmt_725;
            int t_q_9 = 16 + 8 * th + 2 + odd_1;
            if (t_q_9 < clen) {
                {
                    auto _gst_726 = draw_base + (t_q_9 * rs_g - odd_1);
                    __builtin_assume(__isGlobal(_gst_726));
                    *(reinterpret_cast<unsigned int*>(_gst_726) + (0)) = pq_9[0];
                }
            }
            unsigned int pv_10 = __shfl_xor_sync(0xFFFFFFFF, pw_1[2], 1);
            unsigned int pq_10[1];
            uint32_t _prmt_727;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_727) : "r"(pw_1[2]), "r"(pv_10), "r"(sel_1));
            pq_10[0] = _prmt_727;
            int t_q_10 = 16 + 8 * th + 4 + odd_1;
            if (t_q_10 < clen) {
                {
                    auto _gst_728 = draw_base + (t_q_10 * rs_g - odd_1);
                    __builtin_assume(__isGlobal(_gst_728));
                    *(reinterpret_cast<unsigned int*>(_gst_728) + (0)) = pq_10[0];
                }
            }
            unsigned int pv_11 = __shfl_xor_sync(0xFFFFFFFF, pw_1[3], 1);
            unsigned int pq_11[1];
            uint32_t _prmt_729;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_729) : "r"(pw_1[3]), "r"(pv_11), "r"(sel_1));
            pq_11[0] = _prmt_729;
            int t_q_11 = 16 + 8 * th + 6 + odd_1;
            if (t_q_11 < clen) {
                {
                    auto _gst_730 = draw_base + (t_q_11 * rs_g - odd_1);
                    __builtin_assume(__isGlobal(_gst_730));
                    *(reinterpret_cast<unsigned int*>(_gst_730) + (0)) = pq_11[0];
                }
            }
            int t_a_16 = 16 + 8 * th;
            dgr8_1[0] = ((t_a_16 < clen) ? dgr8_1[0] : 0.0f);
            u8_1[0] = ((t_a_16 < clen) ? u8_1[0] : 0.0f);
            int t_a_17 = 16 + 8 * th + 1;
            dgr8_1[1] = ((t_a_17 < clen) ? dgr8_1[1] : 0.0f);
            u8_1[1] = ((t_a_17 < clen) ? u8_1[1] : 0.0f);
            int t_a_18 = 16 + 8 * th + 2;
            dgr8_1[2] = ((t_a_18 < clen) ? dgr8_1[2] : 0.0f);
            u8_1[2] = ((t_a_18 < clen) ? u8_1[2] : 0.0f);
            int t_a_19 = 16 + 8 * th + 3;
            dgr8_1[3] = ((t_a_19 < clen) ? dgr8_1[3] : 0.0f);
            u8_1[3] = ((t_a_19 < clen) ? u8_1[3] : 0.0f);
            int t_a_20 = 16 + 8 * th + 4;
            dgr8_1[4] = ((t_a_20 < clen) ? dgr8_1[4] : 0.0f);
            u8_1[4] = ((t_a_20 < clen) ? u8_1[4] : 0.0f);
            int t_a_21 = 16 + 8 * th + 5;
            dgr8_1[5] = ((t_a_21 < clen) ? dgr8_1[5] : 0.0f);
            u8_1[5] = ((t_a_21 < clen) ? u8_1[5] : 0.0f);
            int t_a_22 = 16 + 8 * th + 6;
            dgr8_1[6] = ((t_a_22 < clen) ? dgr8_1[6] : 0.0f);
            u8_1[6] = ((t_a_22 < clen) ? u8_1[6] : 0.0f);
            int t_a_23 = 16 + 8 * th + 7;
            dgr8_1[7] = ((t_a_23 < clen) ? dgr8_1[7] : 0.0f);
            u8_1[7] = ((t_a_23 < clen) ? u8_1[7] : 0.0f);
            float pb01_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb01_1) : "f"(dgr8_1[0]), "f"(dgr8_1[1]));
            float pb23_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb23_1) : "f"(dgr8_1[2]), "f"(dgr8_1[3]));
            float pb45_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb45_1) : "f"(dgr8_1[4]), "f"(dgr8_1[5]));
            float pb67_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb67_1) : "f"(dgr8_1[6]), "f"(dgr8_1[7]));
            float pb03_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb03_1) : "f"(pb01_1), "f"(pb23_1));
            float pb47_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb47_1) : "f"(pb45_1), "f"(pb67_1));
            float pb_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb_1) : "f"(pb03_1), "f"(pb47_1));
            float pm0_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm0_1) : "f"(dgr8_1[0]), "f"(u8_1[0]));
            float pa0_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa0_1) : "f"(dgr8_1[1]), "f"(u8_1[1]), "f"(pm0_1));
            float pm2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm2_1) : "f"(dgr8_1[2]), "f"(u8_1[2]));
            float pa1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa1_1) : "f"(dgr8_1[3]), "f"(u8_1[3]), "f"(pm2_1));
            float pm4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm4_1) : "f"(dgr8_1[4]), "f"(u8_1[4]));
            float pa2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa2_1) : "f"(dgr8_1[5]), "f"(u8_1[5]), "f"(pm4_1));
            float pm6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm6_1) : "f"(dgr8_1[6]), "f"(u8_1[6]));
            float pa3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa3_1) : "f"(dgr8_1[7]), "f"(u8_1[7]), "f"(pm6_1));
            float pa01_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa01_1) : "f"(pa0_1), "f"(pa1_1));
            float pa23_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa23_1) : "f"(pa2_1), "f"(pa3_1));
            float pa_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa_1) : "f"(pa01_1), "f"(pa23_1));
            float nm_t_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_3) : "f"(s_a_1), "f"(pa_1));
            float _sub_731;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_731) : "f"(s_a_1), "f"(nm_t_3));
            float nm_es_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_3) : "f"(_sub_731), "f"(pa_1));
            float _sub_732;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_732) : "f"(pa_1), "f"(nm_t_3));
            float nm_ex_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_3) : "f"(_sub_732), "f"(s_a_1));
            float _abs_734;
            asm("abs.f32 %0, %1;" : "=f"(_abs_734) : "f"(s_a_1));
            float _abs_735;
            asm("abs.f32 %0, %1;" : "=f"(_abs_735) : "f"(pa_1));
            uint32_t _setp_733_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_733_bits) : "f"(_abs_734), "f"(_abs_735));
            bool _setp_733 = _setp_733_bits != 0;
            float nm_err_3 = ((_setp_733) ? nm_es_3 : nm_ex_3);
            float _add_736;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_736) : "f"(c_a_1), "f"(nm_err_3));
            c_a_1 = _add_736;
            s_a_1 = nm_t_3;
            float nm_t_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_4) : "f"(s_b_1), "f"(pb_1));
            float _sub_737;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_737) : "f"(s_b_1), "f"(nm_t_4));
            float nm_es_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_4) : "f"(_sub_737), "f"(pb_1));
            float _sub_738;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_738) : "f"(pb_1), "f"(nm_t_4));
            float nm_ex_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_4) : "f"(_sub_738), "f"(s_b_1));
            float _abs_740;
            asm("abs.f32 %0, %1;" : "=f"(_abs_740) : "f"(s_b_1));
            float _abs_741;
            asm("abs.f32 %0, %1;" : "=f"(_abs_741) : "f"(pb_1));
            uint32_t _setp_739_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_739_bits) : "f"(_abs_740), "f"(_abs_741));
            bool _setp_739 = _setp_739_bits != 0;
            float nm_err_4 = ((_setp_739) ? nm_es_4 : nm_ex_4);
            float _add_742;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_742) : "f"(c_b_1), "f"(nm_err_4));
            c_b_1 = _add_742;
            s_b_1 = nm_t_4;
            gacc[0] = s_a_1;
            gacc[1] = c_a_1;
            gacc[2] = s_b_1;
            gacc[3] = c_b_1;
            off_acc[0] = off_acc[0] + t_hi_1 + t_lo_1;
            {
                uint32_t _addr_743 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_743), "f"(off_acc[0]) : "memory");
            }
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th), gacc);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float raw_t0[8];
            long long rb_2 = ext0 * g_raw_row_stride + (long long)hv * g_raw_head_stride + (long long)d_lane;
            int t_c_24 = ((8 * th < clen) ? 8 * th : clen - 1);
            raw_t0[0] = (float)g_raw[rb_2 + (long long)t_c_24 * g_raw_row_stride];
            int t_c_25 = ((8 * th + 1 < clen) ? 8 * th + 1 : clen - 1);
            raw_t0[1] = (float)g_raw[rb_2 + (long long)t_c_25 * g_raw_row_stride];
            int t_c_26 = ((8 * th + 2 < clen) ? 8 * th + 2 : clen - 1);
            raw_t0[2] = (float)g_raw[rb_2 + (long long)t_c_26 * g_raw_row_stride];
            int t_c_27 = ((8 * th + 3 < clen) ? 8 * th + 3 : clen - 1);
            raw_t0[3] = (float)g_raw[rb_2 + (long long)t_c_27 * g_raw_row_stride];
            int t_c_28 = ((8 * th + 4 < clen) ? 8 * th + 4 : clen - 1);
            raw_t0[4] = (float)g_raw[rb_2 + (long long)t_c_28 * g_raw_row_stride];
            int t_c_29 = ((8 * th + 5 < clen) ? 8 * th + 5 : clen - 1);
            raw_t0[5] = (float)g_raw[rb_2 + (long long)t_c_29 * g_raw_row_stride];
            int t_c_30 = ((8 * th + 6 < clen) ? 8 * th + 6 : clen - 1);
            raw_t0[6] = (float)g_raw[rb_2 + (long long)t_c_30 * g_raw_row_stride];
            int t_c_31 = ((8 * th + 7 < clen) ? 8 * th + 7 : clen - 1);
            raw_t0[7] = (float)g_raw[rb_2 + (long long)t_c_31 * g_raw_row_stride];
            mbarrier_wait(epi_full_addr + (th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_280[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_280[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_2 = r_280[0];
            float gc_2 = gn_2 - 21.640425f;
            int eb_2 = 98304 + th * -32768;
            int ew_5 = eb_2 >> 2;
            int qbase_2 = pool_f32_addr + (unsigned int)(eb_2 + 12288 + dh_lane * 1024);
            int sb8_2 = 8 * th;
            int gb8_2 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_2 * 128);
            int kb8_2 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_2 * 128);
            int c_qa_1 = 8 * th;
            int c_ka_1 = 8 * th;
            int c_qd_1 = 128 + 8 * th;
            int c_kd_1 = 144 + 8 * th;
            int c_dk_1 = 8 * th;
            int c_bp_1 = c_kd_1;
            float s_run_2[1];
            s_run_2[0] = 0.0f;
            #pragma unroll 1
            for (int k_2 = 0; k_2 < 8; k_2++) {
                int j_5 = 7 - k_2;
                float a_qd_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_1 + j_5)));
                float a_kd_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd_1 + j_5)));
                float a_dk_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk_1 + j_5)));
                float btr_2[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr_2[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_2 + j_5))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t_2 = sb8_2 + j_5;
                float r_281[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_281[0])) : "r"((gb8_2 + (lane_1 / 32 * 8192 + 128 * j_5 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_5 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t_2 = r_281[0];
                unsigned int r_282[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_282[0])) : "r"((kb8_2 + (dcol / 64 * 8192 + 128 * j_5 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_5 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t_2 = __uint_as_float(r_282[0] << 16);
                unsigned int r_283[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_283[0])) : "r"((qbase_2 + (dcol / 64 * 1024 + 128 * j_5 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_5 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t_2 = __uint_as_float(r_283[0] << 16);
                float q_m_2 = ((t_2 >= clen) ? 0.0f : q_t_2);
                float dqf_2 = pool_f32[ew_5 + 128 * j_5 + d_lane];
                float dkf_2 = pool_f32[ew_5 + (1024 + 128 * j_5) + d_lane];
                float dgf_2 = pool_f32[ew_5 + (2048 + 128 * j_5) + d_lane];
                float bt_t_2 = btr_2[0];
                float e_dg_2 = approx_exp2(g_t_2 - gc_2);
                float dq2_2 = a_qd_2[0] * e_dg_2;
                float dk2_2 = a_kd_2[0] * e_dg_2;
                float dbv_2[1];
                dbv_2[0] = dk2_2 * k_t_2;
                float dk2b_2 = dk2_2 * bt_t_2;
                float dq_o_2 = dq2_2 + dqf_2;
                float dk_o_2 = dk2b_2 + dkf_2 + a_dk_2[0];
                float qe_9 = q_m_2 * e_dg_2;
                float kp_21 = k_t_2 * bt_t_2;
                __nv_bfloat16 kb_9 = (__nv_bfloat16)kp_21;
                float kbe_9 = (float)kb_9 * e_dg_2;
                float cr_9 = __fmaf_rn(__uint_as_float(__as_u32(qe_9) & 4294959104u) - qe_9, a_qd_2[0], (__uint_as_float(__as_u32(kbe_9) & 4294959104u) - kp_21 * e_dg_2) * a_kd_2[0]);
                float dg_o_2 = __fmaf_rn(q_m_2, dq2_2, __fmaf_rn(dk2b_2 - a_dk_2[0], k_t_2, cr_9)) + dgf_2;
                s_run_2[0] = s_run_2[0] + dg_o_2;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_1 + j_5)), "f"(s_run_2[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp_1 + j_5)), "f"(dbv_2[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                pool_f32[(98304 - 32768 * th >> 2) + (128 * j_5 + d_lane)] = dq_o_2;
                pool_f32[(98304 - 32768 * th >> 2) + (1024 + 128 * j_5 + d_lane)] = dk_o_2;
            }
            float tot_2 = s_run_2[0];
            float dbp_2[8];
            tmem_ld_x8(&dbp_2[0], taddr + (unsigned int)row_base + (unsigned int)c_bp_1);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_2 = lane_1 >> 4 & 1;
            int b3_2 = lane_1 >> 3 & 1;
            int b2_2 = lane_1 >> 2 & 1;
            float v4_2[4];
            float send_8 = ((b4_2 == 0) ? dbp_2[4] : dbp_2[0]);
            float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
            v4_2[0] = ((b4_2 == 0) ? dbp_2[0] : dbp_2[4]) + _shfl_xor_sync_744;
            float send_9 = ((b4_2 == 0) ? dbp_2[5] : dbp_2[1]);
            float _shfl_xor_sync_745 = __shfl_xor_sync(0xFFFFFFFF, send_9, 16);
            v4_2[1] = ((b4_2 == 0) ? dbp_2[1] : dbp_2[5]) + _shfl_xor_sync_745;
            float send_10 = ((b4_2 == 0) ? dbp_2[6] : dbp_2[2]);
            float _shfl_xor_sync_746 = __shfl_xor_sync(0xFFFFFFFF, send_10, 16);
            v4_2[2] = ((b4_2 == 0) ? dbp_2[2] : dbp_2[6]) + _shfl_xor_sync_746;
            float send_11 = ((b4_2 == 0) ? dbp_2[7] : dbp_2[3]);
            float _shfl_xor_sync_747 = __shfl_xor_sync(0xFFFFFFFF, send_11, 16);
            v4_2[3] = ((b4_2 == 0) ? dbp_2[3] : dbp_2[7]) + _shfl_xor_sync_747;
            float v2_17[2];
            float send2_4 = ((b3_2 == 0) ? v4_2[2] : v4_2[0]);
            float _shfl_xor_sync_748 = __shfl_xor_sync(0xFFFFFFFF, send2_4, 8);
            v2_17[0] = ((b3_2 == 0) ? v4_2[0] : v4_2[2]) + _shfl_xor_sync_748;
            float send2_5 = ((b3_2 == 0) ? v4_2[3] : v4_2[1]);
            float _shfl_xor_sync_749 = __shfl_xor_sync(0xFFFFFFFF, send2_5, 8);
            v2_17[1] = ((b3_2 == 0) ? v4_2[1] : v4_2[3]) + _shfl_xor_sync_749;
            float send1_1 = ((b2_2 == 0) ? v2_17[1] : v2_17[0]);
            float _shfl_xor_sync_750 = __shfl_xor_sync(0xFFFFFFFF, send1_1, 4);
            float v1_2 = ((b2_2 == 0) ? v2_17[0] : v2_17[1]) + _shfl_xor_sync_750;
            float _shfl_xor_sync_751 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 2);
            v1_2 = v1_2 + _shfl_xor_sync_751;
            float _shfl_xor_sync_752 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 1);
            v1_2 = v1_2 + _shfl_xor_sync_752;
            int ew_6 = 98304 - 32768 * th >> 2;
            float nz_2 = __uint_as_float((unsigned int)num_heads >> 31 | 2147483648u);
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int i_3 = 0; i_3 < 2; i_3++) {
                int j_6 = qd + 4 * i_3;
                bool live_3 = 8 * th + j_6 < clen;
                int _min_753 = ((8 * th + j_6) < (clen - 1) ? (8 * th + j_6) : (clen - 1));
                long long row_3 = ext0 + (long long)_min_753;
                float gq_3[4];
                float gk_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gq_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gq_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gq_3[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_6 + 128 * j_6 + 4 * lane_1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gk_3[(0) + 3]))
                    : "r"(pool_f32_addr + (unsigned int)(4 * (ew_6 + 1024 + 128 * j_6 + 4 * lane_1))));
                float xq_3[4];
                float xk_3[4];
                float rq_3[1];
                float rk_3[1];
                {
                    uint2 _vld_754;
                    _vld_754 = *reinterpret_cast<const uint2*>(q_raw + (row_3 * (long long)q_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_754 = reinterpret_cast<uint32_t*>(&_vld_754);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xq_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) << 16);
                        (&xq_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rq_3[0] = *reinterpret_cast<const float*>(qk_rstd + ((0 + row_3) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                {
                    uint2 _vld_755;
                    _vld_755 = *reinterpret_cast<const uint2*>(k_raw + (row_3 * (long long)k_raw_row_stride + (long long)(qk_head * 128 + 4 * lane_1)) + 0);
                    uint32_t* _vpairs_755 = reinterpret_cast<uint32_t*>(&_vld_755);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&xk_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) << 16);
                        (&xk_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) & 0xffff0000u);
                    }
                }
                {
                    rk_3[0] = *reinterpret_cast<const float*>(qk_rstd + (((long long)num_tokens + row_3) * (long long)num_qk_heads + (long long)qk_head) + 0);
                }
                long long dxo_3 = row_3 * (long long)qrs + (long long)(qk_head * 128 + 4 * lane_1);
                float y_3[4];
                float p_3[4];
                y_3[0] = mul_f32_rn_noftz(xq_3[0], rq_3[0]);
                float _fma_756;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_756) : "f"(gq_3[0]), "f"(y_3[0]), "f"(nz_2));
                p_3[0] = _fma_756;
                y_3[1] = mul_f32_rn_noftz(xq_3[1], rq_3[0]);
                float _fma_757;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_757) : "f"(gq_3[1]), "f"(y_3[1]), "f"(nz_2));
                p_3[1] = _fma_757;
                y_3[2] = mul_f32_rn_noftz(xq_3[2], rq_3[0]);
                float _fma_758;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_758) : "f"(gq_3[2]), "f"(y_3[2]), "f"(nz_2));
                p_3[2] = _fma_758;
                y_3[3] = mul_f32_rn_noftz(xq_3[3], rq_3[0]);
                float _fma_759;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_759) : "f"(gq_3[3]), "f"(y_3[3]), "f"(nz_2));
                p_3[3] = _fma_759;
                float _shfl_xor_sync_761 = __shfl_xor_sync(0xFFFFFFFF, p_3[0], 4);
                float _add_760;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_760) : "f"(p_3[0]), "f"(_shfl_xor_sync_761));
                p_3[0] = _add_760;
                float _shfl_xor_sync_763 = __shfl_xor_sync(0xFFFFFFFF, p_3[1], 4);
                float _add_762;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_762) : "f"(p_3[1]), "f"(_shfl_xor_sync_763));
                p_3[1] = _add_762;
                float _shfl_xor_sync_765 = __shfl_xor_sync(0xFFFFFFFF, p_3[2], 4);
                float _add_764;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_764) : "f"(p_3[2]), "f"(_shfl_xor_sync_765));
                p_3[2] = _add_764;
                float _shfl_xor_sync_767 = __shfl_xor_sync(0xFFFFFFFF, p_3[3], 4);
                float _add_766;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_766) : "f"(p_3[3]), "f"(_shfl_xor_sync_767));
                p_3[3] = _add_766;
                float _shfl_xor_sync_769 = __shfl_xor_sync(0xFFFFFFFF, p_3[0], 2);
                float _add_768;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_768) : "f"(p_3[0]), "f"(_shfl_xor_sync_769));
                p_3[0] = _add_768;
                float _shfl_xor_sync_771 = __shfl_xor_sync(0xFFFFFFFF, p_3[1], 2);
                float _add_770;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_770) : "f"(p_3[1]), "f"(_shfl_xor_sync_771));
                p_3[1] = _add_770;
                float _shfl_xor_sync_773 = __shfl_xor_sync(0xFFFFFFFF, p_3[2], 2);
                float _add_772;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_772) : "f"(p_3[2]), "f"(_shfl_xor_sync_773));
                p_3[2] = _add_772;
                float _shfl_xor_sync_775 = __shfl_xor_sync(0xFFFFFFFF, p_3[3], 2);
                float _add_774;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_774) : "f"(p_3[3]), "f"(_shfl_xor_sync_775));
                p_3[3] = _add_774;
                float _shfl_xor_sync_777 = __shfl_xor_sync(0xFFFFFFFF, p_3[0], 1);
                float _add_776;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_776) : "f"(p_3[0]), "f"(_shfl_xor_sync_777));
                p_3[0] = _add_776;
                float _shfl_xor_sync_779 = __shfl_xor_sync(0xFFFFFFFF, p_3[1], 1);
                float _add_778;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_778) : "f"(p_3[1]), "f"(_shfl_xor_sync_779));
                p_3[1] = _add_778;
                float _shfl_xor_sync_781 = __shfl_xor_sync(0xFFFFFFFF, p_3[2], 1);
                float _add_780;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_780) : "f"(p_3[2]), "f"(_shfl_xor_sync_781));
                p_3[2] = _add_780;
                float _shfl_xor_sync_783 = __shfl_xor_sync(0xFFFFFFFF, p_3[3], 1);
                float _add_782;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_782) : "f"(p_3[3]), "f"(_shfl_xor_sync_783));
                p_3[3] = _add_782;
                float _add_784;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_784) : "f"(p_3[0]), "f"(p_3[2]));
                float _add_785;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_785) : "f"(p_3[1]), "f"(p_3[3]));
                float sq_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_3) : "f"(_add_784), "f"(_add_785));
                float _shfl_sync_788 = __shfl_sync(0xFFFFFFFF, sq_3, 0);
                float _shfl_sync_789 = __shfl_sync(0xFFFFFFFF, sq_3, 8);
                float _add_787;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_787) : "f"(_shfl_sync_788), "f"(_shfl_sync_789));
                float _shfl_sync_790 = __shfl_sync(0xFFFFFFFF, sq_3, 16);
                float _add_786;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_786) : "f"(_add_787), "f"(_shfl_sync_790));
                float _shfl_sync_791 = __shfl_sync(0xFFFFFFFF, sq_3, 24);
                float dot_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_3) : "f"(_add_786), "f"(_shfl_sync_791));
                float o_3[4];
                float _fma_793;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_793) : "f"(y_3[0]), "f"(dot_3), "f"(nz_2));
                float _sub_792;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_792) : "f"(gq_3[0]), "f"(_fma_793));
                o_3[0] = mul_f32_rn_noftz(rq_3[0], _sub_792);
                float _fma_795;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(y_3[1]), "f"(dot_3), "f"(nz_2));
                float _sub_794;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_794) : "f"(gq_3[1]), "f"(_fma_795));
                o_3[1] = mul_f32_rn_noftz(rq_3[0], _sub_794);
                float _fma_797;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(y_3[2]), "f"(dot_3), "f"(nz_2));
                float _sub_796;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_796) : "f"(gq_3[2]), "f"(_fma_797));
                o_3[2] = mul_f32_rn_noftz(rq_3[0], _sub_796);
                float _fma_799;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(y_3[3]), "f"(dot_3), "f"(nz_2));
                float _sub_798;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_798) : "f"(gq_3[3]), "f"(_fma_799));
                o_3[3] = mul_f32_rn_noftz(rq_3[0], _sub_798);
                unsigned int ob_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_3[_lp*2 + 0], o_3[_lp*2+1 + 0]));
                    ob_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dq_ext + dxo_3), "r"(ob_3[(0) + 0]), "r"(ob_3[(0) + 1]), "r"((unsigned int)(live_3)) : "memory");
                float y_3_1[4];
                float p_3_1[4];
                y_3_1[0] = mul_f32_rn_noftz(xk_3[0], rk_3[0]);
                float _fma_800;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(gk_3[0]), "f"(y_3_1[0]), "f"(nz_2));
                p_3_1[0] = _fma_800;
                y_3_1[1] = mul_f32_rn_noftz(xk_3[1], rk_3[0]);
                float _fma_801;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(gk_3[1]), "f"(y_3_1[1]), "f"(nz_2));
                p_3_1[1] = _fma_801;
                y_3_1[2] = mul_f32_rn_noftz(xk_3[2], rk_3[0]);
                float _fma_802;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_802) : "f"(gk_3[2]), "f"(y_3_1[2]), "f"(nz_2));
                p_3_1[2] = _fma_802;
                y_3_1[3] = mul_f32_rn_noftz(xk_3[3], rk_3[0]);
                float _fma_803;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_803) : "f"(gk_3[3]), "f"(y_3_1[3]), "f"(nz_2));
                p_3_1[3] = _fma_803;
                float _shfl_xor_sync_805 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[0], 4);
                float _add_804;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_804) : "f"(p_3_1[0]), "f"(_shfl_xor_sync_805));
                p_3_1[0] = _add_804;
                float _shfl_xor_sync_807 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[1], 4);
                float _add_806;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_806) : "f"(p_3_1[1]), "f"(_shfl_xor_sync_807));
                p_3_1[1] = _add_806;
                float _shfl_xor_sync_809 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[2], 4);
                float _add_808;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_808) : "f"(p_3_1[2]), "f"(_shfl_xor_sync_809));
                p_3_1[2] = _add_808;
                float _shfl_xor_sync_811 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[3], 4);
                float _add_810;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_810) : "f"(p_3_1[3]), "f"(_shfl_xor_sync_811));
                p_3_1[3] = _add_810;
                float _shfl_xor_sync_813 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[0], 2);
                float _add_812;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_812) : "f"(p_3_1[0]), "f"(_shfl_xor_sync_813));
                p_3_1[0] = _add_812;
                float _shfl_xor_sync_815 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[1], 2);
                float _add_814;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_814) : "f"(p_3_1[1]), "f"(_shfl_xor_sync_815));
                p_3_1[1] = _add_814;
                float _shfl_xor_sync_817 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[2], 2);
                float _add_816;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_816) : "f"(p_3_1[2]), "f"(_shfl_xor_sync_817));
                p_3_1[2] = _add_816;
                float _shfl_xor_sync_819 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[3], 2);
                float _add_818;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_818) : "f"(p_3_1[3]), "f"(_shfl_xor_sync_819));
                p_3_1[3] = _add_818;
                float _shfl_xor_sync_821 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[0], 1);
                float _add_820;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_820) : "f"(p_3_1[0]), "f"(_shfl_xor_sync_821));
                p_3_1[0] = _add_820;
                float _shfl_xor_sync_823 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[1], 1);
                float _add_822;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_822) : "f"(p_3_1[1]), "f"(_shfl_xor_sync_823));
                p_3_1[1] = _add_822;
                float _shfl_xor_sync_825 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[2], 1);
                float _add_824;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_824) : "f"(p_3_1[2]), "f"(_shfl_xor_sync_825));
                p_3_1[2] = _add_824;
                float _shfl_xor_sync_827 = __shfl_xor_sync(0xFFFFFFFF, p_3_1[3], 1);
                float _add_826;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_826) : "f"(p_3_1[3]), "f"(_shfl_xor_sync_827));
                p_3_1[3] = _add_826;
                float _add_828;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_828) : "f"(p_3_1[0]), "f"(p_3_1[2]));
                float _add_829;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_829) : "f"(p_3_1[1]), "f"(p_3_1[3]));
                float sq_3_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_3_1) : "f"(_add_828), "f"(_add_829));
                float _shfl_sync_832 = __shfl_sync(0xFFFFFFFF, sq_3_1, 0);
                float _shfl_sync_833 = __shfl_sync(0xFFFFFFFF, sq_3_1, 8);
                float _add_831;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_831) : "f"(_shfl_sync_832), "f"(_shfl_sync_833));
                float _shfl_sync_834 = __shfl_sync(0xFFFFFFFF, sq_3_1, 16);
                float _add_830;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_830) : "f"(_add_831), "f"(_shfl_sync_834));
                float _shfl_sync_835 = __shfl_sync(0xFFFFFFFF, sq_3_1, 24);
                float dot_3_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_3_1) : "f"(_add_830), "f"(_shfl_sync_835));
                float o_3_1[4];
                float _fma_837;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(y_3_1[0]), "f"(dot_3_1), "f"(nz_2));
                float _sub_836;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_836) : "f"(gk_3[0]), "f"(_fma_837));
                o_3_1[0] = mul_f32_rn_noftz(rk_3[0], _sub_836);
                float _fma_839;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(y_3_1[1]), "f"(dot_3_1), "f"(nz_2));
                float _sub_838;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_838) : "f"(gk_3[1]), "f"(_fma_839));
                o_3_1[1] = mul_f32_rn_noftz(rk_3[0], _sub_838);
                float _fma_841;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(y_3_1[2]), "f"(dot_3_1), "f"(nz_2));
                float _sub_840;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_840) : "f"(gk_3[2]), "f"(_fma_841));
                o_3_1[2] = mul_f32_rn_noftz(rk_3[0], _sub_840);
                float _fma_843;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(y_3_1[3]), "f"(dot_3_1), "f"(nz_2));
                float _sub_842;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_842) : "f"(gk_3[3]), "f"(_fma_843));
                o_3_1[3] = mul_f32_rn_noftz(rk_3[0], _sub_842);
                unsigned int ob_3_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_3_1[_lp*2 + 0], o_3_1[_lp*2+1 + 0]));
                    ob_3_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(dk_ext + dxo_3), "r"(ob_3_1[(0) + 0]), "r"(ob_3_1[(0) + 1]), "r"((unsigned int)(live_3)) : "memory");
            }
            tmem_ld_x8(&gacc[0], taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_844 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_844), "f"(tot_2) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (qd == 0) {
                asm volatile("barrier.sync 2, 64;" ::: "memory");
            }
            if (qd == 1) {
                asm volatile("barrier.sync 3, 64;" ::: "memory");
            }
            if (qd == 2) {
                asm volatile("barrier.sync 4, 64;" ::: "memory");
            }
            if (qd == 3) {
                asm volatile("barrier.sync 5, 64;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            float t_other_1[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_other_1[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + (1 - th) * 32 + lane_1))));
            float t_hi_2 = ((th == 1) ? tot_2 : t_other_1[0]);
            float t_lo_2 = ((th == 1) ? t_other_1[0] : tot_2);
            float offset_2 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_2);
            float sfx_2[8];
            tmem_ld_x8(&sfx_2[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_a_2 = gacc[0];
            float c_a_2 = gacc[1];
            float s_b_2 = gacc[2];
            float c_b_2 = gacc[3];
            float u8_2[8];
            float dgr8_2[8];
            float2 dyg2_12 = add_f32x2(make_float2(offset_2, offset_2), make_float2(sfx_2[0], sfx_2[1]));
            float2 rp_u_11 = add_f32x2(make_float2(raw_t0[0], raw_t0[1]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_845 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_845), "l"(*(const unsigned long long*)&rp_u_11));
            float _abs_848;
            asm("abs.f32 %0, %1;" : "=f"(_abs_848) : "f"(rp_z_11.x));
            float _neg_847;
            asm("neg.f32 %0, %1;" : "=f"(_neg_847) : "f"(_abs_848));
            float _exp_846 = __nv_expf(_neg_847);
            float _abs_851;
            asm("abs.f32 %0, %1;" : "=f"(_abs_851) : "f"(rp_z_11.y));
            float _neg_850;
            asm("neg.f32 %0, %1;" : "=f"(_neg_850) : "f"(_abs_851));
            float _exp_849 = __nv_expf(_neg_850);
            float2 rp_e_11 = make_float2(_exp_846, _exp_849);
            float2 rp_dn_11 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_11);
            float2 rp_den_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_11) : "l"(*(const unsigned long long*)&rp_dn_11), "l"(*(const unsigned long long*)&rp_dn_11));
            float _rcp_852 = approx_rcp(rp_den_11.x);
            float _rcp_853 = approx_rcp(rp_den_11.y);
            float2 _f32x2_mul_b_854 = make_float2(_rcp_852, _rcp_853);
            float2 rp_sp_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_11) : "l"(*(const unsigned long long*)&rp_e_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_854));
            float2 rp_t_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_11) : "l"(*(const unsigned long long*)&dyg2_12), "l"(*(const unsigned long long*)&rp_sp_11));
            float2 _f32x2_mul_b_855 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_11) : "l"(*(const unsigned long long*)&rp_t_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_855));
            u8_2[0] = rp_u_11.x;
            u8_2[1] = rp_u_11.y;
            dgr8_2[0] = rp_dgr_11.x;
            dgr8_2[1] = rp_dgr_11.y;
            float2 dyg2_13 = add_f32x2(make_float2(offset_2, offset_2), make_float2(sfx_2[2], sfx_2[3]));
            float2 rp_u_12 = add_f32x2(make_float2(raw_t0[2], raw_t0[3]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_856 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_856), "l"(*(const unsigned long long*)&rp_u_12));
            float _abs_859;
            asm("abs.f32 %0, %1;" : "=f"(_abs_859) : "f"(rp_z_12.x));
            float _neg_858;
            asm("neg.f32 %0, %1;" : "=f"(_neg_858) : "f"(_abs_859));
            float _exp_857 = __nv_expf(_neg_858);
            float _abs_862;
            asm("abs.f32 %0, %1;" : "=f"(_abs_862) : "f"(rp_z_12.y));
            float _neg_861;
            asm("neg.f32 %0, %1;" : "=f"(_neg_861) : "f"(_abs_862));
            float _exp_860 = __nv_expf(_neg_861);
            float2 rp_e_12 = make_float2(_exp_857, _exp_860);
            float2 rp_dn_12 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_12);
            float2 rp_den_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_12) : "l"(*(const unsigned long long*)&rp_dn_12), "l"(*(const unsigned long long*)&rp_dn_12));
            float _rcp_863 = approx_rcp(rp_den_12.x);
            float _rcp_864 = approx_rcp(rp_den_12.y);
            float2 _f32x2_mul_b_865 = make_float2(_rcp_863, _rcp_864);
            float2 rp_sp_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_12) : "l"(*(const unsigned long long*)&rp_e_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_865));
            float2 rp_t_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_12) : "l"(*(const unsigned long long*)&dyg2_13), "l"(*(const unsigned long long*)&rp_sp_12));
            float2 _f32x2_mul_b_866 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_12) : "l"(*(const unsigned long long*)&rp_t_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_866));
            u8_2[2] = rp_u_12.x;
            u8_2[3] = rp_u_12.y;
            dgr8_2[2] = rp_dgr_12.x;
            dgr8_2[3] = rp_dgr_12.y;
            float2 dyg2_14 = add_f32x2(make_float2(offset_2, offset_2), make_float2(sfx_2[4], sfx_2[5]));
            float2 rp_u_13 = add_f32x2(make_float2(raw_t0[4], raw_t0[5]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_867 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_867), "l"(*(const unsigned long long*)&rp_u_13));
            float _abs_870;
            asm("abs.f32 %0, %1;" : "=f"(_abs_870) : "f"(rp_z_13.x));
            float _neg_869;
            asm("neg.f32 %0, %1;" : "=f"(_neg_869) : "f"(_abs_870));
            float _exp_868 = __nv_expf(_neg_869);
            float _abs_873;
            asm("abs.f32 %0, %1;" : "=f"(_abs_873) : "f"(rp_z_13.y));
            float _neg_872;
            asm("neg.f32 %0, %1;" : "=f"(_neg_872) : "f"(_abs_873));
            float _exp_871 = __nv_expf(_neg_872);
            float2 rp_e_13 = make_float2(_exp_868, _exp_871);
            float2 rp_dn_13 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_13);
            float2 rp_den_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_13) : "l"(*(const unsigned long long*)&rp_dn_13), "l"(*(const unsigned long long*)&rp_dn_13));
            float _rcp_874 = approx_rcp(rp_den_13.x);
            float _rcp_875 = approx_rcp(rp_den_13.y);
            float2 _f32x2_mul_b_876 = make_float2(_rcp_874, _rcp_875);
            float2 rp_sp_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_13) : "l"(*(const unsigned long long*)&rp_e_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_876));
            float2 rp_t_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_13) : "l"(*(const unsigned long long*)&dyg2_14), "l"(*(const unsigned long long*)&rp_sp_13));
            float2 _f32x2_mul_b_877 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_13) : "l"(*(const unsigned long long*)&rp_t_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_877));
            u8_2[4] = rp_u_13.x;
            u8_2[5] = rp_u_13.y;
            dgr8_2[4] = rp_dgr_13.x;
            dgr8_2[5] = rp_dgr_13.y;
            float2 dyg2_15 = add_f32x2(make_float2(offset_2, offset_2), make_float2(sfx_2[6], sfx_2[7]));
            float2 rp_u_14 = add_f32x2(make_float2(raw_t0[6], raw_t0[7]), make_float2(gacc[6], gacc[6]));
            float2 _f32x2_mul_a_878 = make_float2(gacc[4], gacc[4]);
            float2 rp_z_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_z_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_878), "l"(*(const unsigned long long*)&rp_u_14));
            float _abs_881;
            asm("abs.f32 %0, %1;" : "=f"(_abs_881) : "f"(rp_z_14.x));
            float _neg_880;
            asm("neg.f32 %0, %1;" : "=f"(_neg_880) : "f"(_abs_881));
            float _exp_879 = __nv_expf(_neg_880);
            float _abs_884;
            asm("abs.f32 %0, %1;" : "=f"(_abs_884) : "f"(rp_z_14.y));
            float _neg_883;
            asm("neg.f32 %0, %1;" : "=f"(_neg_883) : "f"(_abs_884));
            float _exp_882 = __nv_expf(_neg_883);
            float2 rp_e_14 = make_float2(_exp_879, _exp_882);
            float2 rp_dn_14 = add_f32x2(make_float2(1.0f, 1.0f), rp_e_14);
            float2 rp_den_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_den_14) : "l"(*(const unsigned long long*)&rp_dn_14), "l"(*(const unsigned long long*)&rp_dn_14));
            float _rcp_885 = approx_rcp(rp_den_14.x);
            float _rcp_886 = approx_rcp(rp_den_14.y);
            float2 _f32x2_mul_b_887 = make_float2(_rcp_885, _rcp_886);
            float2 rp_sp_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_sp_14) : "l"(*(const unsigned long long*)&rp_e_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_887));
            float2 rp_t_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_t_14) : "l"(*(const unsigned long long*)&dyg2_15), "l"(*(const unsigned long long*)&rp_sp_14));
            float2 _f32x2_mul_b_888 = make_float2(gacc[5], gacc[5]);
            float2 rp_dgr_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&rp_dgr_14) : "l"(*(const unsigned long long*)&rp_t_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_888));
            u8_2[6] = rp_u_14.x;
            u8_2[7] = rp_u_14.y;
            dgr8_2[6] = rp_dgr_14.x;
            dgr8_2[7] = rp_dgr_14.y;
            int odd_2 = lane & 1;
            unsigned int sel_2 = ((odd_2 == 1) ? 12918 : 21520);
            unsigned int pw_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dgr8_2[_lp*2 + 0], dgr8_2[_lp*2+1 + 0]));
                pw_2[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int pv_12 = __shfl_xor_sync(0xFFFFFFFF, pw_2[0], 1);
            unsigned int pq_12[1];
            uint32_t _prmt_889;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_889) : "r"(pw_2[0]), "r"(pv_12), "r"(sel_2));
            pq_12[0] = _prmt_889;
            int t_q_12 = 8 * th + odd_2;
            if (t_q_12 < clen) {
                {
                    auto _gst_890 = draw_base + (t_q_12 * rs_g - odd_2);
                    __builtin_assume(__isGlobal(_gst_890));
                    *(reinterpret_cast<unsigned int*>(_gst_890) + (0)) = pq_12[0];
                }
            }
            unsigned int pv_13 = __shfl_xor_sync(0xFFFFFFFF, pw_2[1], 1);
            unsigned int pq_13[1];
            uint32_t _prmt_891;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_891) : "r"(pw_2[1]), "r"(pv_13), "r"(sel_2));
            pq_13[0] = _prmt_891;
            int t_q_13 = 8 * th + 2 + odd_2;
            if (t_q_13 < clen) {
                {
                    auto _gst_892 = draw_base + (t_q_13 * rs_g - odd_2);
                    __builtin_assume(__isGlobal(_gst_892));
                    *(reinterpret_cast<unsigned int*>(_gst_892) + (0)) = pq_13[0];
                }
            }
            unsigned int pv_14 = __shfl_xor_sync(0xFFFFFFFF, pw_2[2], 1);
            unsigned int pq_14[1];
            uint32_t _prmt_893;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_893) : "r"(pw_2[2]), "r"(pv_14), "r"(sel_2));
            pq_14[0] = _prmt_893;
            int t_q_14 = 8 * th + 4 + odd_2;
            if (t_q_14 < clen) {
                {
                    auto _gst_894 = draw_base + (t_q_14 * rs_g - odd_2);
                    __builtin_assume(__isGlobal(_gst_894));
                    *(reinterpret_cast<unsigned int*>(_gst_894) + (0)) = pq_14[0];
                }
            }
            unsigned int pv_15 = __shfl_xor_sync(0xFFFFFFFF, pw_2[3], 1);
            unsigned int pq_15[1];
            uint32_t _prmt_895;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_895) : "r"(pw_2[3]), "r"(pv_15), "r"(sel_2));
            pq_15[0] = _prmt_895;
            int t_q_15 = 8 * th + 6 + odd_2;
            if (t_q_15 < clen) {
                {
                    auto _gst_896 = draw_base + (t_q_15 * rs_g - odd_2);
                    __builtin_assume(__isGlobal(_gst_896));
                    *(reinterpret_cast<unsigned int*>(_gst_896) + (0)) = pq_15[0];
                }
            }
            int t_a_24 = 8 * th;
            dgr8_2[0] = ((t_a_24 < clen) ? dgr8_2[0] : 0.0f);
            u8_2[0] = ((t_a_24 < clen) ? u8_2[0] : 0.0f);
            int t_a_25 = 8 * th + 1;
            dgr8_2[1] = ((t_a_25 < clen) ? dgr8_2[1] : 0.0f);
            u8_2[1] = ((t_a_25 < clen) ? u8_2[1] : 0.0f);
            int t_a_26 = 8 * th + 2;
            dgr8_2[2] = ((t_a_26 < clen) ? dgr8_2[2] : 0.0f);
            u8_2[2] = ((t_a_26 < clen) ? u8_2[2] : 0.0f);
            int t_a_27 = 8 * th + 3;
            dgr8_2[3] = ((t_a_27 < clen) ? dgr8_2[3] : 0.0f);
            u8_2[3] = ((t_a_27 < clen) ? u8_2[3] : 0.0f);
            int t_a_28 = 8 * th + 4;
            dgr8_2[4] = ((t_a_28 < clen) ? dgr8_2[4] : 0.0f);
            u8_2[4] = ((t_a_28 < clen) ? u8_2[4] : 0.0f);
            int t_a_29 = 8 * th + 5;
            dgr8_2[5] = ((t_a_29 < clen) ? dgr8_2[5] : 0.0f);
            u8_2[5] = ((t_a_29 < clen) ? u8_2[5] : 0.0f);
            int t_a_30 = 8 * th + 6;
            dgr8_2[6] = ((t_a_30 < clen) ? dgr8_2[6] : 0.0f);
            u8_2[6] = ((t_a_30 < clen) ? u8_2[6] : 0.0f);
            int t_a_31 = 8 * th + 7;
            dgr8_2[7] = ((t_a_31 < clen) ? dgr8_2[7] : 0.0f);
            u8_2[7] = ((t_a_31 < clen) ? u8_2[7] : 0.0f);
            float pb01_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb01_2) : "f"(dgr8_2[0]), "f"(dgr8_2[1]));
            float pb23_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb23_2) : "f"(dgr8_2[2]), "f"(dgr8_2[3]));
            float pb45_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb45_2) : "f"(dgr8_2[4]), "f"(dgr8_2[5]));
            float pb67_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb67_2) : "f"(dgr8_2[6]), "f"(dgr8_2[7]));
            float pb03_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb03_2) : "f"(pb01_2), "f"(pb23_2));
            float pb47_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb47_2) : "f"(pb45_2), "f"(pb67_2));
            float pb_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pb_2) : "f"(pb03_2), "f"(pb47_2));
            float pm0_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm0_2) : "f"(dgr8_2[0]), "f"(u8_2[0]));
            float pa0_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa0_2) : "f"(dgr8_2[1]), "f"(u8_2[1]), "f"(pm0_2));
            float pm2_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm2_2) : "f"(dgr8_2[2]), "f"(u8_2[2]));
            float pa1_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa1_2) : "f"(dgr8_2[3]), "f"(u8_2[3]), "f"(pm2_2));
            float pm4_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm4_2) : "f"(dgr8_2[4]), "f"(u8_2[4]));
            float pa2_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa2_2) : "f"(dgr8_2[5]), "f"(u8_2[5]), "f"(pm4_2));
            float pm6_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(pm6_2) : "f"(dgr8_2[6]), "f"(u8_2[6]));
            float pa3_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pa3_2) : "f"(dgr8_2[7]), "f"(u8_2[7]), "f"(pm6_2));
            float pa01_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa01_2) : "f"(pa0_2), "f"(pa1_2));
            float pa23_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa23_2) : "f"(pa2_2), "f"(pa3_2));
            float pa_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(pa_2) : "f"(pa01_2), "f"(pa23_2));
            float nm_t_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_5) : "f"(s_a_2), "f"(pa_2));
            float _sub_897;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_897) : "f"(s_a_2), "f"(nm_t_5));
            float nm_es_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_5) : "f"(_sub_897), "f"(pa_2));
            float _sub_898;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_898) : "f"(pa_2), "f"(nm_t_5));
            float nm_ex_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_5) : "f"(_sub_898), "f"(s_a_2));
            float _abs_900;
            asm("abs.f32 %0, %1;" : "=f"(_abs_900) : "f"(s_a_2));
            float _abs_901;
            asm("abs.f32 %0, %1;" : "=f"(_abs_901) : "f"(pa_2));
            uint32_t _setp_899_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_899_bits) : "f"(_abs_900), "f"(_abs_901));
            bool _setp_899 = _setp_899_bits != 0;
            float nm_err_5 = ((_setp_899) ? nm_es_5 : nm_ex_5);
            float _add_902;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_902) : "f"(c_a_2), "f"(nm_err_5));
            c_a_2 = _add_902;
            s_a_2 = nm_t_5;
            float nm_t_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_6) : "f"(s_b_2), "f"(pb_2));
            float _sub_903;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_903) : "f"(s_b_2), "f"(nm_t_6));
            float nm_es_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_6) : "f"(_sub_903), "f"(pb_2));
            float _sub_904;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_904) : "f"(pb_2), "f"(nm_t_6));
            float nm_ex_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_6) : "f"(_sub_904), "f"(s_b_2));
            float _abs_906;
            asm("abs.f32 %0, %1;" : "=f"(_abs_906) : "f"(s_b_2));
            float _abs_907;
            asm("abs.f32 %0, %1;" : "=f"(_abs_907) : "f"(pb_2));
            uint32_t _setp_905_bits;
            asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_905_bits) : "f"(_abs_906), "f"(_abs_907));
            bool _setp_905 = _setp_905_bits != 0;
            float nm_err_6 = ((_setp_905) ? nm_es_6 : nm_ex_6);
            float _add_908;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_908) : "f"(c_b_2), "f"(nm_err_6));
            c_b_2 = _add_908;
            s_b_2 = nm_t_6;
            gacc[0] = s_a_2;
            gacc[1] = c_a_2;
            gacc[2] = s_b_2;
            gacc[3] = c_b_2;
            off_acc[0] = off_acc[0] + t_hi_2 + t_lo_2;
            long long pidx = (((long long)chunk * (long long)num_heads + (long long)hv) * 128 + (long long)d_lane) * 2;
            int m_addr = pool_f32_addr + (unsigned int)(qd * 8192 + lane_1 * 16);
            float other[4];
            if (th == 1) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(m_addr), "f"(gacc[0]), "f"(gacc[1]), "f"(gacc[2]), "f"(gacc[3]) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (qd == 0) {
                asm volatile("barrier.sync 2, 64;" ::: "memory");
            }
            if (qd == 1) {
                asm volatile("barrier.sync 3, 64;" ::: "memory");
            }
            if (qd == 2) {
                asm volatile("barrier.sync 4, 64;" ::: "memory");
            }
            if (qd == 3) {
                asm volatile("barrier.sync 5, 64;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (th == 0) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                    : "r"(m_addr));
                float s_a_3 = gacc[0];
                float c_a_3 = gacc[1];
                float s_b_3 = gacc[2];
                float c_b_3 = gacc[3];
                float ts_s;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(s_a_3), "f"(other[0]));
                float ts_bb;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(s_a_3));
                float _sub_910;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_910) : "f"(ts_s), "f"(ts_bb));
                float _sub_909;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_909) : "f"(s_a_3), "f"(_sub_910));
                float _sub_911;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_911) : "f"(other[0]), "f"(ts_bb));
                float ts_e;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_909), "f"(_sub_911));
                float _add_912;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_912) : "f"(c_a_3), "f"(other[1]));
                float pm_c;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c) : "f"(_add_912), "f"(ts_e));
                float ts_s_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(ts_s), "f"(pm_c));
                float ts_bb_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(ts_s));
                float _sub_914;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_914) : "f"(ts_s_0), "f"(ts_bb_0));
                float _sub_913;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_913) : "f"(ts_s), "f"(_sub_914));
                float _sub_915;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_915) : "f"(pm_c), "f"(ts_bb_0));
                float ts_e_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_913), "f"(_sub_915));
                s_a_3 = ts_s_0;
                c_a_3 = ts_e_0;
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(s_b_3), "f"(other[2]));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(s_b_3));
                float _sub_917;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_917) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_916;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_916) : "f"(s_b_3), "f"(_sub_917));
                float _sub_918;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_918) : "f"(other[2]), "f"(ts_bb_1));
                float ts_e_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_916), "f"(_sub_918));
                float _add_919;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_919) : "f"(c_b_3), "f"(other[3]));
                float pm_c_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_0) : "f"(_add_919), "f"(ts_e_1));
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(ts_s_1), "f"(pm_c_0));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(ts_s_1));
                float _sub_921;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_921) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_920;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_920) : "f"(ts_s_1), "f"(_sub_921));
                float _sub_922;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_922) : "f"(pm_c_0), "f"(ts_bb_2));
                float ts_e_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_920), "f"(_sub_922));
                s_b_3 = ts_s_2;
                c_b_3 = ts_e_2;
                gacc[0] = s_a_3;
                gacc[1] = c_a_3;
                gacc[2] = s_b_3;
                gacc[3] = c_b_3;
                {
                    float2 _v2 = make_float2(gacc[0 + 0], gacc[0 + 1]);
                    *reinterpret_cast<float2*>((dA_part + pidx) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(gacc[2 + 0], gacc[2 + 1]);
                    *reinterpret_cast<float2*>((db_part + pidx) + 0) = _v2;
                }
            }
            if ((lane_1 & 3) == 0) {
                int e_star = lane_1 >> 2 & 7;
                pool_f32[28160 + qd * 64 + 8 * th + e_star] = v1_2;
                pool_f32[28160 + qd * 64 + 16 + 8 * th + e_star] = v1_1;
                pool_f32[28160 + qd * 64 + 32 + 8 * th + e_star] = v1;
                pool_f32[28160 + qd * 64 + 48 + 8 * th + e_star] = v1_0;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (tid_0 < clen) {
                int idx = (row0 + tid_0) * num_heads + hv;
                float total = pool_f32[28160 + tid_0] + pool_f32[28224 + tid_0];
                total = total + pool_f32[28288 + tid_0];
                total = total + pool_f32[28352 + tid_0];
                long long tok0 = (long long)chunk_bos[blockIdx.x];
                dbeta_ext[(tok0 + (long long)tid_0) * (long long)num_heads + (long long)hv] = (__nv_bfloat16)(db_f[idx] + total);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (w == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: helper ----
    if (warp == 8) {
        { // helper_main
            int chunk_h = blockIdx.x;
            int hv_h = blockIdx.y;
            int qk_head_h = hv_h / group;
            int row0_h = chunk_h * 64;
            if (elect_sync()) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(ops_full_addr, 12288);
                tma_3d_gmem2smem(gk_p_addr, (&gk), 0, hv_h, row0_h, ops_full_addr);
                tma_3d_gmem2smem(gk_p_addr + 8192, (&gk), 32, hv_h, row0_h, ops_full_addr);
                tma_3d_gmem2smem(gk_p_addr + 16384, (&gk), 64, hv_h, row0_h, ops_full_addr);
                tma_3d_gmem2smem(gk_p_addr + 24576, (&gk), 96, hv_h, row0_h, ops_full_addr);
                tma_3d_gmem2smem(k_p_addr, (&k_e), 0, qk_head_h, kq0_setup, ops_full_addr);
                tma_3d_gmem2smem(k_p_addr + 8192, (&k_e), 64, qk_head_h, kq0_setup, ops_full_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(ops_full_addr + 8, 12288);
                tma_3d_gmem2smem(gk_p_addr + 2048, (&gk), 0, hv_h, row0_h + 16, ops_full_addr + 8);
                tma_3d_gmem2smem(gk_p_addr + 8192 + 2048, (&gk), 32, hv_h, row0_h + 16, ops_full_addr + 8);
                tma_3d_gmem2smem(gk_p_addr + 16384 + 2048, (&gk), 64, hv_h, row0_h + 16, ops_full_addr + 8);
                tma_3d_gmem2smem(gk_p_addr + 24576 + 2048, (&gk), 96, hv_h, row0_h + 16, ops_full_addr + 8);
                tma_3d_gmem2smem(k_p_addr + 2048, (&k_e), 0, qk_head_h, kq0_setup + 16, ops_full_addr + 8);
                tma_3d_gmem2smem(k_p_addr + 8192 + 2048, (&k_e), 64, qk_head_h, kq0_setup + 16, ops_full_addr + 8);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(ops_full_addr + 16, 12288);
                tma_3d_gmem2smem(gk_p_addr + 4096, (&gk), 0, hv_h, row0_h + 32, ops_full_addr + 16);
                tma_3d_gmem2smem(gk_p_addr + 8192 + 4096, (&gk), 32, hv_h, row0_h + 32, ops_full_addr + 16);
                tma_3d_gmem2smem(gk_p_addr + 16384 + 4096, (&gk), 64, hv_h, row0_h + 32, ops_full_addr + 16);
                tma_3d_gmem2smem(gk_p_addr + 24576 + 4096, (&gk), 96, hv_h, row0_h + 32, ops_full_addr + 16);
                tma_3d_gmem2smem(k_p_addr + 4096, (&k_e), 0, qk_head_h, kq0_setup + 32, ops_full_addr + 16);
                tma_3d_gmem2smem(k_p_addr + 8192 + 4096, (&k_e), 64, qk_head_h, kq0_setup + 32, ops_full_addr + 16);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(ops_full_addr + 24, 12288);
                tma_3d_gmem2smem(gk_p_addr + 6144, (&gk), 0, hv_h, row0_h + 48, ops_full_addr + 24);
                tma_3d_gmem2smem(gk_p_addr + 8192 + 6144, (&gk), 32, hv_h, row0_h + 48, ops_full_addr + 24);
                tma_3d_gmem2smem(gk_p_addr + 16384 + 6144, (&gk), 64, hv_h, row0_h + 48, ops_full_addr + 24);
                tma_3d_gmem2smem(gk_p_addr + 24576 + 6144, (&gk), 96, hv_h, row0_h + 48, ops_full_addr + 24);
                tma_3d_gmem2smem(k_p_addr + 6144, (&k_e), 0, qk_head_h, kq0_setup + 48, ops_full_addr + 24);
                tma_3d_gmem2smem(k_p_addr + 8192 + 6144, (&k_e), 64, qk_head_h, kq0_setup + 48, ops_full_addr + 24);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr, 12288);
                tma_3d_gmem2smem(dab_addr, (&dAqk), 0, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 2048, (&dAkk), 0, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096, (&dAqk), 32, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096 + 2048, (&dAkk), 32, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr, (&q_e), 0, qk_head_h, kq0_setup, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 1024, (&q_e), 0, qk_head_h, kq0_setup + 8, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 2048, (&q_e), 64, qk_head_h, kq0_setup, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 2048 + 1024, (&q_e), 64, qk_head_h, kq0_setup + 8, blk_full_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr + 8, 12288);
                tma_3d_gmem2smem(dab_addr + 8192, (&dAqk), 0, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 2048, (&dAkk), 0, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096, (&dAqk), 32, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096 + 2048, (&dAkk), 32, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 4096, (&q_e), 0, qk_head_h, kq0_setup + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 4096 + 1024, (&q_e), 0, qk_head_h, kq0_setup + 16 + 8, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 6144, (&q_e), 64, qk_head_h, kq0_setup + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 6144 + 1024, (&q_e), 64, qk_head_h, kq0_setup + 16 + 8, blk_full_addr + 8);
                if (clen_setup < 16) {
                    mbarrier_wait(ops_full_addr, 0);
                    mbarrier_wait(blk_full_addr, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float z[4];
                    z[0] = 0.0f;
                    z[1] = 0.0f;
                    z[2] = 0.0f;
                    z[3] = 0.0f;
                    #pragma unroll 1
                    for (int r_284 = 0; r_284 < 16; r_284++) {
                        int t_3 = r_284;
                        if (t_3 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_284 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr);
                }
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_0 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_1 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _smem_desc_lo_0 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_1 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (64))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(blk_empty_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr, 12288);
                tma_3d_gmem2smem(dab_addr, (&dAqk), 0, hv_h, row0_h + 32, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 2048, (&dAkk), 0, hv_h, row0_h + 32, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096, (&dAqk), 32, hv_h, row0_h + 32, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096 + 2048, (&dAkk), 32, hv_h, row0_h + 32, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr, (&q_e), 0, qk_head_h, kq0_setup + 32, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 1024, (&q_e), 0, qk_head_h, kq0_setup + 32 + 8, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 2048, (&q_e), 64, qk_head_h, kq0_setup + 32, blk_full_addr);
                tma_3d_gmem2smem(q_p_addr + 2048 + 1024, (&q_e), 64, qk_head_h, kq0_setup + 32 + 8, blk_full_addr);
                if (clen_setup < 32) {
                    mbarrier_wait(ops_full_addr + 8, 0);
                    mbarrier_wait(blk_full_addr + 8, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float z_1[4];
                    z_1[0] = 0.0f;
                    z_1[1] = 0.0f;
                    z_1[2] = 0.0f;
                    z_1[3] = 0.0f;
                    #pragma unroll 1
                    for (int r_285 = 0; r_285 < 16; r_285++) {
                        int t_4 = 16 + r_285;
                        if (t_4 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_285 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 8);
                }
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_2 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_3 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _smem_desc_lo_2 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_3 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_4 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_5 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                int _smem_desc_lo_4 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_5 = (((tiles_addr) >> 4) & 0x3FFF) + (2) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_acc + (80))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(blk_empty_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr + 8, 12288);
                tma_3d_gmem2smem(dab_addr + 8192, (&dAqk), 0, hv_h, row0_h + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 2048, (&dAkk), 0, hv_h, row0_h + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096, (&dAqk), 32, hv_h, row0_h + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096 + 2048, (&dAkk), 32, hv_h, row0_h + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 4096, (&q_e), 0, qk_head_h, kq0_setup + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 4096 + 1024, (&q_e), 0, qk_head_h, kq0_setup + 48 + 8, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 6144, (&q_e), 64, qk_head_h, kq0_setup + 48, blk_full_addr + 8);
                tma_3d_gmem2smem(q_p_addr + 6144 + 1024, (&q_e), 64, qk_head_h, kq0_setup + 48 + 8, blk_full_addr + 8);
                if (clen_setup < 48) {
                    mbarrier_wait(ops_full_addr + 16, 0);
                    mbarrier_wait(blk_full_addr, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float z_2[4];
                    z_2[0] = 0.0f;
                    z_2[1] = 0.0f;
                    z_2[2] = 0.0f;
                    z_2[3] = 0.0f;
                    #pragma unroll 1
                    for (int r_286 = 0; r_286 < 16; r_286++) {
                        int t_5 = 32 + r_286;
                        if (t_5 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_5 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_5 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_286 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 16);
                }
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_6 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_7 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                int _smem_desc_lo_6 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_7 = (((tiles_addr) >> 4) & 0x3FFF) + (3) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_8 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_9 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _smem_desc_lo_8 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_9 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_acc + (16))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_10 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_11 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _smem_desc_lo_10 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_11 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"((tmem_acc + (96))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                if (clen_setup < 64) {
                    mbarrier_wait(ops_full_addr + 24, 0);
                    mbarrier_wait(blk_full_addr + 8, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float z_3[4];
                    z_3[0] = 0.0f;
                    z_3[1] = 0.0f;
                    z_3[2] = 0.0f;
                    z_3[3] = 0.0f;
                    #pragma unroll 1
                    for (int r_287 = 0; r_287 < 16; r_287++) {
                        int t_6 = 48 + r_287;
                        if (t_6 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_6 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_6 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_287 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 24);
                }
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_12 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_13 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                int _smem_desc_lo_12 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_13 = (((tiles_addr) >> 4) & 0x3FFF) + (2) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"(tmem_acc), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_14 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_15 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                int _smem_desc_lo_14 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_15 = (((tiles_addr) >> 4) & 0x3FFF) + (3) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"((tmem_acc + (16))), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_16 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_17 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _smem_desc_lo_16 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_17 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"((tmem_acc + (32))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_18 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_19 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _smem_desc_lo_18 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_19 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_acc + (112))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                tcgen05_commit(p2_done_addr);
                mbarrier_wait(p2_done_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(epi_empty_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 98304, (&dq_f_t), 0, hv_h, row0_h + 32, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 102400, (&dk_f_t), 0, hv_h, row0_h + 32, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 106496, (&dg_f_t), 0, hv_h, row0_h + 32, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup + 32, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup + 32, epi_full_addr);
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(0)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 40)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(64)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 40)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(0)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 48)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(64)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 48)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(0)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 56)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_e))), "r"((int)(64)), "r"((int)(qk_head_h)), "r"((int)(kq0_setup + 56)) : "memory");
            }
            int pf_lane = tid & 31;
            long long pf_base = ((long long)(row0_h + (pf_lane >> 2)) * (long long)num_heads + (long long)hv_h) * 128 + (long long)((pf_lane & 3) * 32);
            long long pf_off_2_0 = pf_base + (long long)num_heads * 5120;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off_2_0)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off_2_0)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off_2_0)));
            long long pf_off_2_1 = pf_base + (long long)num_heads * 6144;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off_2_1)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off_2_1)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off_2_1)));
            long long pf_off_2_2 = pf_base + (long long)num_heads * 7168;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off_2_2)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off_2_2)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off_2_2)));
            if (elect_sync()) {
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_20 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_21 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_20 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_21 = (((dab_addr) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"((tmem_acc + (128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_22 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_23 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_22 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_23 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_22), "r"(_smem_desc_lo_23), "r"((tmem_acc + (144))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_24 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_25 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_24 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_25 = (((dab_addr) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"((tmem_acc + (160))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_26 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_27 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_26 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_27 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_26), "r"(_smem_desc_lo_27), "r"((tmem_acc + (176))), "r"(0) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_28 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_29 = (((dab_addr + 64) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_28 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_29 = (((dab_addr + 64) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_28), "r"(_smem_desc_lo_29), "r"((tmem_acc + (128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_30 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_31 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_30 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_31 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_30), "r"(_smem_desc_lo_31), "r"((tmem_acc + (144))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_32 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_33 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_32 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_33 = (((dab_addr + 64) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_32), "r"(_smem_desc_lo_33), "r"((tmem_acc + (160))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_34 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_35 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_34 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_35 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_34), "r"(_smem_desc_lo_35), "r"((tmem_acc + (176))), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_36 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_37 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_36 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_37 = (((dab_addr + 4096) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_36), "r"(_smem_desc_lo_37), "r"((tmem_acc + (160))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_38 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_39 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_38 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_39 = (((dab_addr + 6144) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_38), "r"(_smem_desc_lo_39), "r"((tmem_acc + (176))), "r"(1) : "memory");
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_40 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_41 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_40 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_41 = (((dab_addr + 4096) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_40), "r"(_smem_desc_lo_41), "r"((tmem_acc + (192))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_42 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_43 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_42 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_43 = (((dab_addr + 6144) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_42), "r"(_smem_desc_lo_43), "r"((tmem_acc + (208))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_44 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_45 = (((dab_addr + 4160) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_44 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_45 = (((dab_addr + 4160) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_44), "r"(_smem_desc_lo_45), "r"((tmem_acc + (224))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_46 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_47 = (((dab_addr + 6208) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_46 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_47 = (((dab_addr + 6208) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_46), "r"(_smem_desc_lo_47), "r"((tmem_acc + (240))), "r"(0) : "memory");
                tcgen05_commit(p1a_ready_addr);
                tcgen05_commit(ring_empty_addr + 8);
                tcgen05_commit(p1blk_empty_addr);
                mbarrier_wait(p1blk_empty_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr, 8192);
                tma_3d_gmem2smem(dab_addr, (&dAqk), 0, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 2048, (&dAkk), 0, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096, (&dAqk), 32, hv_h, row0_h, blk_full_addr);
                tma_3d_gmem2smem(dab_addr + 4096 + 2048, (&dAkk), 32, hv_h, row0_h, blk_full_addr);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(blk_full_addr + 8, 8192);
                tma_3d_gmem2smem(dab_addr + 8192, (&dAqk), 0, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 2048, (&dAkk), 0, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096, (&dAqk), 32, hv_h, row0_h + 16, blk_full_addr + 8);
                tma_3d_gmem2smem(dab_addr + 8192 + 4096 + 2048, (&dAkk), 32, hv_h, row0_h + 16, blk_full_addr + 8);
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(fold_done_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_48 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_49 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_48 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_49 = (((dab_addr + 64) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_48), "r"(_smem_desc_lo_49), "r"((tmem_acc + (96))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_50 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_51 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_50 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_51 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_50), "r"(_smem_desc_lo_51), "r"((tmem_acc + (112))), "r"(0) : "memory");
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_52 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_53 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_52 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_53 = (((dab_addr) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_52), "r"(_smem_desc_lo_53), "r"((tmem_acc + (64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_54 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_55 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _smem_desc_lo_54 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _smem_desc_lo_55 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (1) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_54), "r"(_smem_desc_lo_55), "r"((tmem_acc + (80))), "r"(0) : "memory");
                tcgen05_commit(s16_done_addr);
                mbarrier_wait(s16_done_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(epi_empty_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 65536, (&dq_f_t), 0, hv_h, row0_h + 40, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 69632, (&dk_f_t), 0, hv_h, row0_h + 40, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 73728, (&dg_f_t), 0, hv_h, row0_h + 40, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 40, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 40, epi_full_addr + 8);
                mbarrier_wait(epi_empty_addr, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 98304, (&dq_f_t), 0, hv_h, row0_h + 48, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 102400, (&dk_f_t), 0, hv_h, row0_h + 48, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 106496, (&dg_f_t), 0, hv_h, row0_h + 48, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup + 48, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup + 48, epi_full_addr);
                mbarrier_wait(p1a_t2_drained_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(epi_empty_addr + 8, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 65536, (&dq_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 69632, (&dk_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 73728, (&dg_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_56 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_57 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_56 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_57 = (((dab_addr) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_56), "r"(_smem_desc_lo_57), "r"((tmem_acc + (128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_58 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_59 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _smem_desc_lo_58 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _smem_desc_lo_59 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (0) * 512;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_58), "r"(_smem_desc_lo_59), "r"((tmem_acc + (144))), "r"(0) : "memory");
                tcgen05_commit(p1b_ready_addr);
                mbarrier_wait(p1b_ready_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(epi_empty_addr + 16, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 16, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 81920, (&dq_f_t), 0, hv_h, row0_h + 16, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 86016, (&dk_f_t), 0, hv_h, row0_h + 16, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 90112, (&dg_f_t), 0, hv_h, row0_h + 16, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 94208, (&q_e), 0, qk_head_h, kq0_setup + 16, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 95232, (&q_e), 64, qk_head_h, kq0_setup + 16, epi_full_addr + 16);
                mbarrier_wait(epi_empty_addr + 24, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 24, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 49152, (&dq_f_t), 0, hv_h, row0_h + 24, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 53248, (&dk_f_t), 0, hv_h, row0_h + 24, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 57344, (&dg_f_t), 0, hv_h, row0_h + 24, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 61440, (&q_e), 0, qk_head_h, kq0_setup + 24, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 62464, (&q_e), 64, qk_head_h, kq0_setup + 24, epi_full_addr + 24);
                mbarrier_wait(epi_empty_addr, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 98304, (&dq_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 102400, (&dk_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 106496, (&dg_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup, epi_full_addr);
                mbarrier_wait(epi_empty_addr + 8, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 65536, (&dq_f_t), 0, hv_h, row0_h + 8, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 69632, (&dk_f_t), 0, hv_h, row0_h + 8, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 73728, (&dg_f_t), 0, hv_h, row0_h + 8, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 8, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 8, epi_full_addr + 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
