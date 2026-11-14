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

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_det_train_aa6b5c23c97e1cb9880a(const __grid_constant__ CUtensorMap dAqk, const __grid_constant__ CUtensorMap dAkk, const __grid_constant__ CUtensorMap gk, const __grid_constant__ CUtensorMap k_e, const __grid_constant__ CUtensorMap q_e, const __grid_constant__ CUtensorMap dq_f_t, const __grid_constant__ CUtensorMap dk_f_t, const __grid_constant__ CUtensorMap dg_f_t, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_ext, __nv_bfloat16* __restrict__ dbeta_ext, int num_heads, int num_qk_heads, int group)
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
            float* dg_base = dg_ext + (gbase_t + (long long)d_lane);
            long long qoff0 = ext0 * (long long)qrs + (long long)(qk_head * 128 + d_lane);
            __nv_bfloat16* dq_base = dq_ext + qoff0;
            __nv_bfloat16* dk_base = dk_ext + qoff0;
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
            unsigned int r_4[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_4[0])) : "r"((qbase + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[0] = __uint_as_float(r_4[0] << 16);
            unsigned int r_5[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_5[0])) : "r"((qbase + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[1] = __uint_as_float(r_5[0] << 16);
            unsigned int r_6[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_6[0])) : "r"((kbase + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_0 = __uint_as_float(r_6[0] << 16);
            unsigned int r_7[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_7[0])) : "r"((kbase + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_0 = __uint_as_float(r_7[0] << 16);
            float2 _f32x2_mul_a_0 = make_float2(k_s0_1_0, k_s1_1_0);
            float2 _f32x2_mul_b_1 = make_float2(bv[0], bv[1]);
            float2 kp_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_1));
            __nv_bfloat16 kb0_1_0 = (__nv_bfloat16)kp_1_0.x;
            __nv_bfloat16 kb1_1_0 = (__nv_bfloat16)kp_1_0.y;
            kb_s[0] = (float)kb0_1_0;
            kb_s[1] = (float)kb1_1_0;
            float r_8[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_8[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[2] = r_8[0];
            float r_9[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_9[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[3] = r_9[0];
            unsigned int r_10[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_10[0])) : "r"((qbase + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[2] = __uint_as_float(r_10[0] << 16);
            unsigned int r_11[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_11[0])) : "r"((qbase + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[3] = __uint_as_float(r_11[0] << 16);
            unsigned int r_12[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_12[0])) : "r"((kbase + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_1 = __uint_as_float(r_12[0] << 16);
            unsigned int r_13[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_13[0])) : "r"((kbase + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_1 = __uint_as_float(r_13[0] << 16);
            float2 _f32x2_mul_a_2 = make_float2(k_s0_1_1, k_s1_1_1);
            float2 _f32x2_mul_b_3 = make_float2(bv[2], bv[3]);
            float2 kp_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
            __nv_bfloat16 kb0_1_1 = (__nv_bfloat16)kp_1_1.x;
            __nv_bfloat16 kb1_1_1 = (__nv_bfloat16)kp_1_1.y;
            kb_s[2] = (float)kb0_1_1;
            kb_s[3] = (float)kb1_1_1;
            float r_14[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_14[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[4] = r_14[0];
            float r_15[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_15[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[5] = r_15[0];
            unsigned int r_16[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_16[0])) : "r"((qbase + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[4] = __uint_as_float(r_16[0] << 16);
            unsigned int r_17[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_17[0])) : "r"((qbase + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[5] = __uint_as_float(r_17[0] << 16);
            unsigned int r_18[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_18[0])) : "r"((kbase + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_2 = __uint_as_float(r_18[0] << 16);
            unsigned int r_19[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_19[0])) : "r"((kbase + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_2 = __uint_as_float(r_19[0] << 16);
            float2 _f32x2_mul_a_4 = make_float2(k_s0_1_2, k_s1_1_2);
            float2 _f32x2_mul_b_5 = make_float2(bv[4], bv[5]);
            float2 kp_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            __nv_bfloat16 kb0_1_2 = (__nv_bfloat16)kp_1_2.x;
            __nv_bfloat16 kb1_1_2 = (__nv_bfloat16)kp_1_2.y;
            kb_s[4] = (float)kb0_1_2;
            kb_s[5] = (float)kb1_1_2;
            float r_20[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_20[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[6] = r_20[0];
            float r_21[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_21[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[7] = r_21[0];
            unsigned int r_22[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_22[0])) : "r"((qbase + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[6] = __uint_as_float(r_22[0] << 16);
            unsigned int r_23[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_23[0])) : "r"((qbase + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[7] = __uint_as_float(r_23[0] << 16);
            unsigned int r_24[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_24[0])) : "r"((kbase + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1_3 = __uint_as_float(r_24[0] << 16);
            unsigned int r_25[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_25[0])) : "r"((kbase + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1_3 = __uint_as_float(r_25[0] << 16);
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
            float r_26[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_26[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d = r_26[0] - 21.640425f;
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
            float r_27[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_27[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[0] = r_27[0];
            float r_28[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_28[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[1] = r_28[0];
            unsigned int r_29[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_29[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[0] = __uint_as_float(r_29[0] << 16);
            unsigned int r_30[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_30[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[1] = __uint_as_float(r_30[0] << 16);
            unsigned int r_31[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_31[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_0 = __uint_as_float(r_31[0] << 16);
            unsigned int r_32[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_32[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_0 = __uint_as_float(r_32[0] << 16);
            float2 _f32x2_mul_a_20 = make_float2(k_s0_0, k_s1_0);
            float2 _f32x2_mul_b_21 = make_float2(bv_63[0], bv_63[1]);
            float2 kp_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            __nv_bfloat16 kb0_0 = (__nv_bfloat16)kp_0.x;
            __nv_bfloat16 kb1_0 = (__nv_bfloat16)kp_0.y;
            kb_s_62[0] = (float)kb0_0;
            kb_s_62[1] = (float)kb1_0;
            float r_33[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_33[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[2] = r_33[0];
            float r_34[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_34[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[3] = r_34[0];
            unsigned int r_35[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_35[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[2] = __uint_as_float(r_35[0] << 16);
            unsigned int r_36[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_36[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[3] = __uint_as_float(r_36[0] << 16);
            unsigned int r_37[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_37[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_1 = __uint_as_float(r_37[0] << 16);
            unsigned int r_38[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_38[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_1 = __uint_as_float(r_38[0] << 16);
            float2 _f32x2_mul_a_22 = make_float2(k_s0_1, k_s1_1);
            float2 _f32x2_mul_b_23 = make_float2(bv_63[2], bv_63[3]);
            float2 kp_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            __nv_bfloat16 kb0_1 = (__nv_bfloat16)kp_1.x;
            __nv_bfloat16 kb1_1 = (__nv_bfloat16)kp_1.y;
            kb_s_62[2] = (float)kb0_1;
            kb_s_62[3] = (float)kb1_1;
            float r_39[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_39[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[4] = r_39[0];
            float r_41[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_41[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[5] = r_41[0];
            unsigned int r_42[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_42[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[4] = __uint_as_float(r_42[0] << 16);
            unsigned int r_43[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_43[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[5] = __uint_as_float(r_43[0] << 16);
            unsigned int r_44[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_44[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_2 = __uint_as_float(r_44[0] << 16);
            unsigned int r_45[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_45[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_2 = __uint_as_float(r_45[0] << 16);
            float2 _f32x2_mul_a_24 = make_float2(k_s0_2, k_s1_2);
            float2 _f32x2_mul_b_25 = make_float2(bv_63[4], bv_63[5]);
            float2 kp_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
            __nv_bfloat16 kb0_2 = (__nv_bfloat16)kp_2.x;
            __nv_bfloat16 kb1_2 = (__nv_bfloat16)kp_2.y;
            kb_s_62[4] = (float)kb0_2;
            kb_s_62[5] = (float)kb1_2;
            float r_46[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_46[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[6] = r_46[0];
            float r_47[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_47[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[7] = r_47[0];
            unsigned int r_48[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_48[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[6] = __uint_as_float(r_48[0] << 16);
            unsigned int r_49[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_49[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[7] = __uint_as_float(r_49[0] << 16);
            unsigned int r_50[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_50[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_3 = __uint_as_float(r_50[0] << 16);
            unsigned int r_51[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_51[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_3 = __uint_as_float(r_51[0] << 16);
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
            float r_52[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_52[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_109 = r_52[0];
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
            float2 d2_2 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_5);
            float ev0_2 = approx_exp2(d2_2.x);
            float ev1_2 = approx_exp2(d2_2.y);
            float2 _f2_7 = make_float2(ev0_2, ev1_2);
            float2 _f32x2_mul_a_30 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f2_7));
            float2 _f32x2_mul_a_31 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f2_7));
            vq_110[2] = vq2_2.x;
            vq_110[3] = vq2_2.y;
            vk_111[2] = vk2_2.x;
            vk_111[3] = vk2_2.y;
            float2 d2_3 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_5);
            float ev0_3 = approx_exp2(d2_3.x);
            float ev1_3 = approx_exp2(d2_3.y);
            float2 _f2_8 = make_float2(ev0_3, ev1_3);
            float2 _f32x2_mul_a_32 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f2_8));
            float2 _f32x2_mul_a_33 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f2_8));
            vq_110[4] = vq2_3.x;
            vq_110[5] = vq2_3.y;
            vk_111[4] = vk2_3.x;
            vk_111[5] = vk2_3.y;
            float2 d2_4 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_5);
            float ev0_4 = approx_exp2(d2_4.x);
            float ev1_4 = approx_exp2(d2_4.y);
            float2 _f2_9 = make_float2(ev0_4, ev1_4);
            float2 _f32x2_mul_a_34 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f2_9));
            float2 _f32x2_mul_a_35 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f2_9));
            vq_110[6] = vq2_4.x;
            vq_110[7] = vq2_4.y;
            vk_111[6] = vk2_4.x;
            vk_111[7] = vk2_4.y;
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
            float r_53[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_53[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_133 = r_53[0] - 21.640425f;
            float2 _f2_10 = make_float2(a_d_133, a_d_133);
            float vq_134[8];
            float vk_135[8];
            float2 d2_5 = sub_f32x2(make_float2(g_s_60[0], g_s_60[1]), _f2_10);
            float ev0_5 = approx_exp2(d2_5.x);
            float ev1_5 = approx_exp2(d2_5.y);
            float2 _f2_11 = make_float2(ev0_5, ev1_5);
            float2 _f32x2_mul_a_40 = make_float2(q_s_61[0], q_s_61[1]);
            float2 vq2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f2_11));
            float2 _f32x2_mul_a_41 = make_float2(kb_s_62[0], kb_s_62[1]);
            float2 vk2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_f2_11));
            vq_134[0] = vq2_5.x;
            vq_134[1] = vq2_5.y;
            vk_135[0] = vk2_5.x;
            vk_135[1] = vk2_5.y;
            float2 d2_6 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_10);
            float ev0_6 = approx_exp2(d2_6.x);
            float ev1_6 = approx_exp2(d2_6.y);
            float2 _f2_12 = make_float2(ev0_6, ev1_6);
            float2 _f32x2_mul_a_42 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f2_12));
            float2 _f32x2_mul_a_43 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_f2_12));
            vq_134[2] = vq2_6.x;
            vq_134[3] = vq2_6.y;
            vk_135[2] = vk2_6.x;
            vk_135[3] = vk2_6.y;
            float2 d2_7 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_10);
            float ev0_7 = approx_exp2(d2_7.x);
            float ev1_7 = approx_exp2(d2_7.y);
            float2 _f2_13 = make_float2(ev0_7, ev1_7);
            float2 _f32x2_mul_a_44 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f2_13));
            float2 _f32x2_mul_a_45 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f2_13));
            vq_134[4] = vq2_7.x;
            vq_134[5] = vq2_7.y;
            vk_135[4] = vk2_7.x;
            vk_135[5] = vk2_7.y;
            float2 d2_8 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_10);
            float ev0_8 = approx_exp2(d2_8.x);
            float ev1_8 = approx_exp2(d2_8.y);
            float2 _f2_14 = make_float2(ev0_8, ev1_8);
            float2 _f32x2_mul_a_46 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f2_14));
            float2 _f32x2_mul_a_47 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_f2_14));
            vq_134[6] = vq2_8.x;
            vq_134[7] = vq2_8.y;
            vk_135[6] = vk2_8.x;
            vk_135[7] = vk2_8.y;
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
            float r_54[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_54[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[0] = r_54[0];
            float r_55[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_55[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[1] = r_55[0];
            unsigned int r_56[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_56[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[0] = __uint_as_float(r_56[0] << 16);
            unsigned int r_57[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_57[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[1] = __uint_as_float(r_57[0] << 16);
            unsigned int r_58[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_58[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_4 = __uint_as_float(r_58[0] << 16);
            unsigned int r_59[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_59[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_4 = __uint_as_float(r_59[0] << 16);
            float2 _f32x2_mul_a_48 = make_float2(k_s0_4, k_s1_4);
            float2 _f32x2_mul_b_49 = make_float2(bv_163[0], bv_163[1]);
            float2 kp_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
            __nv_bfloat16 kb0_4 = (__nv_bfloat16)kp_4.x;
            __nv_bfloat16 kb1_4 = (__nv_bfloat16)kp_4.y;
            kb_s_162[0] = (float)kb0_4;
            kb_s_162[1] = (float)kb1_4;
            float r_60[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_60[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[2] = r_60[0];
            float r_61[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_61[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[3] = r_61[0];
            unsigned int r_62[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_62[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[2] = __uint_as_float(r_62[0] << 16);
            unsigned int r_63[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_63[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[3] = __uint_as_float(r_63[0] << 16);
            unsigned int r_64[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_64[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_5 = __uint_as_float(r_64[0] << 16);
            unsigned int r_65[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_65[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_5 = __uint_as_float(r_65[0] << 16);
            float2 _f32x2_mul_a_50 = make_float2(k_s0_5, k_s1_5);
            float2 _f32x2_mul_b_51 = make_float2(bv_163[2], bv_163[3]);
            float2 kp_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            __nv_bfloat16 kb0_5 = (__nv_bfloat16)kp_5.x;
            __nv_bfloat16 kb1_5 = (__nv_bfloat16)kp_5.y;
            kb_s_162[2] = (float)kb0_5;
            kb_s_162[3] = (float)kb1_5;
            float r_66[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_66[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[4] = r_66[0];
            float r_67[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_67[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[5] = r_67[0];
            unsigned int r_68[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_68[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[4] = __uint_as_float(r_68[0] << 16);
            unsigned int r_69[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_69[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[5] = __uint_as_float(r_69[0] << 16);
            unsigned int r_70[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_70[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_6 = __uint_as_float(r_70[0] << 16);
            unsigned int r_71[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_71[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_6 = __uint_as_float(r_71[0] << 16);
            float2 _f32x2_mul_a_52 = make_float2(k_s0_6, k_s1_6);
            float2 _f32x2_mul_b_53 = make_float2(bv_163[4], bv_163[5]);
            float2 kp_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            __nv_bfloat16 kb0_6 = (__nv_bfloat16)kp_6.x;
            __nv_bfloat16 kb1_6 = (__nv_bfloat16)kp_6.y;
            kb_s_162[4] = (float)kb0_6;
            kb_s_162[5] = (float)kb1_6;
            float r_72[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_72[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[6] = r_72[0];
            float r_73[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_73[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[7] = r_73[0];
            unsigned int r_74[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_74[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[6] = __uint_as_float(r_74[0] << 16);
            unsigned int r_75[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_75[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[7] = __uint_as_float(r_75[0] << 16);
            unsigned int r_76[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_76[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_7 = __uint_as_float(r_76[0] << 16);
            unsigned int r_77[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_77[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_7 = __uint_as_float(r_77[0] << 16);
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
            float r_78[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_78[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_209 = r_78[0];
            float2 _f2_15 = make_float2(a_d_209, a_d_209);
            float vq_210[8];
            float vk_211[8];
            float2 d2_9 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_15);
            float ev0_9 = approx_exp2(d2_9.x);
            float ev1_9 = approx_exp2(d2_9.y);
            float2 _f2_16 = make_float2(ev0_9, ev1_9);
            float2 _f32x2_mul_a_56 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f2_16));
            float2 _f32x2_mul_a_57 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f2_16));
            vq_210[0] = vq2_9.x;
            vq_210[1] = vq2_9.y;
            vk_211[0] = vk2_9.x;
            vk_211[1] = vk2_9.y;
            float2 d2_10 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_15);
            float ev0_10 = approx_exp2(d2_10.x);
            float ev1_10 = approx_exp2(d2_10.y);
            float2 _f2_17 = make_float2(ev0_10, ev1_10);
            float2 _f32x2_mul_a_58 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f2_17));
            float2 _f32x2_mul_a_59 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f2_17));
            vq_210[2] = vq2_10.x;
            vq_210[3] = vq2_10.y;
            vk_211[2] = vk2_10.x;
            vk_211[3] = vk2_10.y;
            float2 d2_11 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_15);
            float ev0_11 = approx_exp2(d2_11.x);
            float ev1_11 = approx_exp2(d2_11.y);
            float2 _f2_18 = make_float2(ev0_11, ev1_11);
            float2 _f32x2_mul_a_60 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f2_18));
            float2 _f32x2_mul_a_61 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_f2_18));
            vq_210[4] = vq2_11.x;
            vq_210[5] = vq2_11.y;
            vk_211[4] = vk2_11.x;
            vk_211[5] = vk2_11.y;
            float2 d2_12 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_15);
            float ev0_12 = approx_exp2(d2_12.x);
            float ev1_12 = approx_exp2(d2_12.y);
            float2 _f2_19 = make_float2(ev0_12, ev1_12);
            float2 _f32x2_mul_a_62 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f2_19));
            float2 _f32x2_mul_a_63 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f2_19));
            vq_210[6] = vq2_12.x;
            vq_210[7] = vq2_12.y;
            vk_211[6] = vk2_12.x;
            vk_211[7] = vk2_12.y;
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
            float r_79[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_79[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_233 = r_79[0];
            float2 _f2_20 = make_float2(a_d_233, a_d_233);
            float vq_234[8];
            float vk_235[8];
            float2 d2_13 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_20);
            float ev0_13 = approx_exp2(d2_13.x);
            float ev1_13 = approx_exp2(d2_13.y);
            float2 _f2_21 = make_float2(ev0_13, ev1_13);
            float2 _f32x2_mul_a_64 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f2_21));
            float2 _f32x2_mul_a_65 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f2_21));
            vq_234[0] = vq2_13.x;
            vq_234[1] = vq2_13.y;
            vk_235[0] = vk2_13.x;
            vk_235[1] = vk2_13.y;
            float2 d2_14 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_20);
            float ev0_14 = approx_exp2(d2_14.x);
            float ev1_14 = approx_exp2(d2_14.y);
            float2 _f2_22 = make_float2(ev0_14, ev1_14);
            float2 _f32x2_mul_a_66 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f2_22));
            float2 _f32x2_mul_a_67 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f2_22));
            vq_234[2] = vq2_14.x;
            vq_234[3] = vq2_14.y;
            vk_235[2] = vk2_14.x;
            vk_235[3] = vk2_14.y;
            float2 d2_15 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_20);
            float ev0_15 = approx_exp2(d2_15.x);
            float ev1_15 = approx_exp2(d2_15.y);
            float2 _f2_23 = make_float2(ev0_15, ev1_15);
            float2 _f32x2_mul_a_68 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f2_23));
            float2 _f32x2_mul_a_69 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f2_23));
            vq_234[4] = vq2_15.x;
            vq_234[5] = vq2_15.y;
            vk_235[4] = vk2_15.x;
            vk_235[5] = vk2_15.y;
            float2 d2_16 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_20);
            float ev0_16 = approx_exp2(d2_16.x);
            float ev1_16 = approx_exp2(d2_16.y);
            float2 _f2_24 = make_float2(ev0_16, ev1_16);
            float2 _f32x2_mul_a_70 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_16;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f2_24));
            float2 _f32x2_mul_a_71 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_16;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f2_24));
            vq_234[6] = vq2_16.x;
            vq_234[7] = vq2_16.y;
            vk_235[6] = vk2_16.x;
            vk_235[7] = vk2_16.y;
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
            float r_80[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_80[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_257 = r_80[0] - 21.640425f;
            float2 _f2_25 = make_float2(a_d_257, a_d_257);
            float vq_258[8];
            float vk_259[8];
            float2 d2_17 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_25);
            float ev0_17 = approx_exp2(d2_17.x);
            float ev1_17 = approx_exp2(d2_17.y);
            float2 _f2_26 = make_float2(ev0_17, ev1_17);
            float2 _f32x2_mul_a_76 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_17;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f2_26));
            float2 _f32x2_mul_a_77 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_17;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f2_26));
            vq_258[0] = vq2_17.x;
            vq_258[1] = vq2_17.y;
            vk_259[0] = vk2_17.x;
            vk_259[1] = vk2_17.y;
            float2 d2_18 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_25);
            float ev0_18 = approx_exp2(d2_18.x);
            float ev1_18 = approx_exp2(d2_18.y);
            float2 _f2_27 = make_float2(ev0_18, ev1_18);
            float2 _f32x2_mul_a_78 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_18;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f2_27));
            float2 _f32x2_mul_a_79 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_18;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f2_27));
            vq_258[2] = vq2_18.x;
            vq_258[3] = vq2_18.y;
            vk_259[2] = vk2_18.x;
            vk_259[3] = vk2_18.y;
            float2 d2_19 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_25);
            float ev0_19 = approx_exp2(d2_19.x);
            float ev1_19 = approx_exp2(d2_19.y);
            float2 _f2_28 = make_float2(ev0_19, ev1_19);
            float2 _f32x2_mul_a_80 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_19;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f2_28));
            float2 _f32x2_mul_a_81 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_19;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f2_28));
            vq_258[4] = vq2_19.x;
            vq_258[5] = vq2_19.y;
            vk_259[4] = vk2_19.x;
            vk_259[5] = vk2_19.y;
            float2 d2_20 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_25);
            float ev0_20 = approx_exp2(d2_20.x);
            float ev1_20 = approx_exp2(d2_20.y);
            float2 _f2_29 = make_float2(ev0_20, ev1_20);
            float2 _f32x2_mul_a_82 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_20;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f2_29));
            float2 _f32x2_mul_a_83 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_20;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f2_29));
            vq_258[6] = vq2_20.x;
            vq_258[7] = vq2_20.y;
            vk_259[6] = vk2_20.x;
            vk_259[7] = vk2_20.y;
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
            float r_81[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_81[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[0] = r_81[0];
            float r_82[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_82[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[1] = r_82[0];
            unsigned int r_83[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_83[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[0] = __uint_as_float(r_83[0] << 16);
            unsigned int r_84[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_84[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[1] = __uint_as_float(r_84[0] << 16);
            unsigned int r_85[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_85[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_8 = __uint_as_float(r_85[0] << 16);
            unsigned int r_86[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_86[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_8 = __uint_as_float(r_86[0] << 16);
            float2 _f32x2_mul_a_84 = make_float2(k_s0_8, k_s1_8);
            float2 _f32x2_mul_b_85 = make_float2(bv_287[0], bv_287[1]);
            float2 kp_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
            __nv_bfloat16 kb0_8 = (__nv_bfloat16)kp_8.x;
            __nv_bfloat16 kb1_8 = (__nv_bfloat16)kp_8.y;
            kb_s_286[0] = (float)kb0_8;
            kb_s_286[1] = (float)kb1_8;
            float r_87[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_87[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[2] = r_87[0];
            float r_88[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_88[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[3] = r_88[0];
            unsigned int r_89[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_89[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[2] = __uint_as_float(r_89[0] << 16);
            unsigned int r_90[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_90[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[3] = __uint_as_float(r_90[0] << 16);
            unsigned int r_91[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_91[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_9 = __uint_as_float(r_91[0] << 16);
            unsigned int r_92[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_92[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_9 = __uint_as_float(r_92[0] << 16);
            float2 _f32x2_mul_a_86 = make_float2(k_s0_9, k_s1_9);
            float2 _f32x2_mul_b_87 = make_float2(bv_287[2], bv_287[3]);
            float2 kp_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
            __nv_bfloat16 kb0_9 = (__nv_bfloat16)kp_9.x;
            __nv_bfloat16 kb1_9 = (__nv_bfloat16)kp_9.y;
            kb_s_286[2] = (float)kb0_9;
            kb_s_286[3] = (float)kb1_9;
            float r_93[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_93[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[4] = r_93[0];
            float r_94[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_94[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[5] = r_94[0];
            unsigned int r_95[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_95[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[4] = __uint_as_float(r_95[0] << 16);
            unsigned int r_96[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_96[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[5] = __uint_as_float(r_96[0] << 16);
            unsigned int r_97[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_97[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_10 = __uint_as_float(r_97[0] << 16);
            unsigned int r_98[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_98[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_10 = __uint_as_float(r_98[0] << 16);
            float2 _f32x2_mul_a_88 = make_float2(k_s0_10, k_s1_10);
            float2 _f32x2_mul_b_89 = make_float2(bv_287[4], bv_287[5]);
            float2 kp_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
            __nv_bfloat16 kb0_10 = (__nv_bfloat16)kp_10.x;
            __nv_bfloat16 kb1_10 = (__nv_bfloat16)kp_10.y;
            kb_s_286[4] = (float)kb0_10;
            kb_s_286[5] = (float)kb1_10;
            float r_99[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_99[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[6] = r_99[0];
            float r_100[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_100[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[7] = r_100[0];
            unsigned int r_101[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_101[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[6] = __uint_as_float(r_101[0] << 16);
            unsigned int r_102[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_102[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[7] = __uint_as_float(r_102[0] << 16);
            unsigned int r_103[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_103[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_11 = __uint_as_float(r_103[0] << 16);
            unsigned int r_104[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_104[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_11 = __uint_as_float(r_104[0] << 16);
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
            float r_105[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_105[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_333 = r_105[0];
            float2 _f2_30 = make_float2(a_d_333, a_d_333);
            float vq_334[8];
            float vk_335[8];
            float2 d2_21 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_30);
            float ev0_21 = approx_exp2(d2_21.x);
            float ev1_21 = approx_exp2(d2_21.y);
            float2 _f2_31 = make_float2(ev0_21, ev1_21);
            float2 _f32x2_mul_a_92 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_21;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f2_31));
            float2 _f32x2_mul_a_93 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_21;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f2_31));
            vq_334[0] = vq2_21.x;
            vq_334[1] = vq2_21.y;
            vk_335[0] = vk2_21.x;
            vk_335[1] = vk2_21.y;
            float2 d2_22 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_30);
            float ev0_22 = approx_exp2(d2_22.x);
            float ev1_22 = approx_exp2(d2_22.y);
            float2 _f2_32 = make_float2(ev0_22, ev1_22);
            float2 _f32x2_mul_a_94 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_22;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&_f2_32));
            float2 _f32x2_mul_a_95 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_22;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f2_32));
            vq_334[2] = vq2_22.x;
            vq_334[3] = vq2_22.y;
            vk_335[2] = vk2_22.x;
            vk_335[3] = vk2_22.y;
            float2 d2_23 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_30);
            float ev0_23 = approx_exp2(d2_23.x);
            float ev1_23 = approx_exp2(d2_23.y);
            float2 _f2_33 = make_float2(ev0_23, ev1_23);
            float2 _f32x2_mul_a_96 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f2_33));
            float2 _f32x2_mul_a_97 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f2_33));
            vq_334[4] = vq2_23.x;
            vq_334[5] = vq2_23.y;
            vk_335[4] = vk2_23.x;
            vk_335[5] = vk2_23.y;
            float2 d2_24 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_30);
            float ev0_24 = approx_exp2(d2_24.x);
            float ev1_24 = approx_exp2(d2_24.y);
            float2 _f2_34 = make_float2(ev0_24, ev1_24);
            float2 _f32x2_mul_a_98 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_24;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_f2_34));
            float2 _f32x2_mul_a_99 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_24;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f2_34));
            vq_334[6] = vq2_24.x;
            vq_334[7] = vq2_24.y;
            vk_335[6] = vk2_24.x;
            vk_335[7] = vk2_24.y;
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
            float r_106[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_106[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_357 = r_106[0];
            float2 _f2_35 = make_float2(a_d_357, a_d_357);
            float vq_358[8];
            float vk_359[8];
            float2 d2_25 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_35);
            float ev0_25 = approx_exp2(d2_25.x);
            float ev1_25 = approx_exp2(d2_25.y);
            float2 _f2_36 = make_float2(ev0_25, ev1_25);
            float2 _f32x2_mul_a_100 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_25;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f2_36));
            float2 _f32x2_mul_a_101 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_25;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f2_36));
            vq_358[0] = vq2_25.x;
            vq_358[1] = vq2_25.y;
            vk_359[0] = vk2_25.x;
            vk_359[1] = vk2_25.y;
            float2 d2_26 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_35);
            float ev0_26 = approx_exp2(d2_26.x);
            float ev1_26 = approx_exp2(d2_26.y);
            float2 _f2_37 = make_float2(ev0_26, ev1_26);
            float2 _f32x2_mul_a_102 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f2_37));
            float2 _f32x2_mul_a_103 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f2_37));
            vq_358[2] = vq2_26.x;
            vq_358[3] = vq2_26.y;
            vk_359[2] = vk2_26.x;
            vk_359[3] = vk2_26.y;
            float2 d2_27 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_35);
            float ev0_27 = approx_exp2(d2_27.x);
            float ev1_27 = approx_exp2(d2_27.y);
            float2 _f2_38 = make_float2(ev0_27, ev1_27);
            float2 _f32x2_mul_a_104 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_27;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f2_38));
            float2 _f32x2_mul_a_105 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_27;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f2_38));
            vq_358[4] = vq2_27.x;
            vq_358[5] = vq2_27.y;
            vk_359[4] = vk2_27.x;
            vk_359[5] = vk2_27.y;
            float2 d2_28 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_35);
            float ev0_28 = approx_exp2(d2_28.x);
            float ev1_28 = approx_exp2(d2_28.y);
            float2 _f2_39 = make_float2(ev0_28, ev1_28);
            float2 _f32x2_mul_a_106 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_28;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&_f2_39));
            float2 _f32x2_mul_a_107 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_28;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f2_39));
            vq_358[6] = vq2_28.x;
            vq_358[7] = vq2_28.y;
            vk_359[6] = vk2_28.x;
            vk_359[7] = vk2_28.y;
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
            float r_107[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_107[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_381 = r_107[0];
            float2 _f2_40 = make_float2(a_d_381, a_d_381);
            float vq_382[8];
            float vk_383[8];
            float2 d2_29 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_40);
            float ev0_29 = approx_exp2(d2_29.x);
            float ev1_29 = approx_exp2(d2_29.y);
            float2 _f2_41 = make_float2(ev0_29, ev1_29);
            float2 _f32x2_mul_a_108 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_29;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_f2_41));
            float2 _f32x2_mul_a_109 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_29;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f2_41));
            vq_382[0] = vq2_29.x;
            vq_382[1] = vq2_29.y;
            vk_383[0] = vk2_29.x;
            vk_383[1] = vk2_29.y;
            float2 d2_30 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_40);
            float ev0_30 = approx_exp2(d2_30.x);
            float ev1_30 = approx_exp2(d2_30.y);
            float2 _f2_42 = make_float2(ev0_30, ev1_30);
            float2 _f32x2_mul_a_110 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_30;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&_f2_42));
            float2 _f32x2_mul_a_111 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_30;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_f2_42));
            vq_382[2] = vq2_30.x;
            vq_382[3] = vq2_30.y;
            vk_383[2] = vk2_30.x;
            vk_383[3] = vk2_30.y;
            float2 d2_31 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_40);
            float ev0_31 = approx_exp2(d2_31.x);
            float ev1_31 = approx_exp2(d2_31.y);
            float2 _f2_43 = make_float2(ev0_31, ev1_31);
            float2 _f32x2_mul_a_112 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_31;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&_f2_43));
            float2 _f32x2_mul_a_113 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_31;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f2_43));
            vq_382[4] = vq2_31.x;
            vq_382[5] = vq2_31.y;
            vk_383[4] = vk2_31.x;
            vk_383[5] = vk2_31.y;
            float2 d2_32 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_40);
            float ev0_32 = approx_exp2(d2_32.x);
            float ev1_32 = approx_exp2(d2_32.y);
            float2 _f2_44 = make_float2(ev0_32, ev1_32);
            float2 _f32x2_mul_a_114 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_32;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_f2_44));
            float2 _f32x2_mul_a_115 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_32;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_f2_44));
            vq_382[6] = vq2_32.x;
            vq_382[7] = vq2_32.y;
            vk_383[6] = vk2_32.x;
            vk_383[7] = vk2_32.y;
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
            float r_109[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_109[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_405 = r_109[0] - 21.640425f;
            float2 _f2_45 = make_float2(a_d_405, a_d_405);
            float vq_406[8];
            float vk_407[8];
            float2 d2_33 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_45);
            float ev0_33 = approx_exp2(d2_33.x);
            float ev1_33 = approx_exp2(d2_33.y);
            float2 _f2_46 = make_float2(ev0_33, ev1_33);
            float2 _f32x2_mul_a_120 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_33;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&_f2_46));
            float2 _f32x2_mul_a_121 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_33;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f2_46));
            vq_406[0] = vq2_33.x;
            vq_406[1] = vq2_33.y;
            vk_407[0] = vk2_33.x;
            vk_407[1] = vk2_33.y;
            float2 d2_34 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_45);
            float ev0_34 = approx_exp2(d2_34.x);
            float ev1_34 = approx_exp2(d2_34.y);
            float2 _f2_47 = make_float2(ev0_34, ev1_34);
            float2 _f32x2_mul_a_122 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_34;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&_f2_47));
            float2 _f32x2_mul_a_123 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_34;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_f2_47));
            vq_406[2] = vq2_34.x;
            vq_406[3] = vq2_34.y;
            vk_407[2] = vk2_34.x;
            vk_407[3] = vk2_34.y;
            float2 d2_35 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_45);
            float ev0_35 = approx_exp2(d2_35.x);
            float ev1_35 = approx_exp2(d2_35.y);
            float2 _f2_48 = make_float2(ev0_35, ev1_35);
            float2 _f32x2_mul_a_124 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_35;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&_f2_48));
            float2 _f32x2_mul_a_125 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_35;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f2_48));
            vq_406[4] = vq2_35.x;
            vq_406[5] = vq2_35.y;
            vk_407[4] = vk2_35.x;
            vk_407[5] = vk2_35.y;
            float2 d2_36 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_45);
            float ev0_36 = approx_exp2(d2_36.x);
            float ev1_36 = approx_exp2(d2_36.y);
            float2 _f2_49 = make_float2(ev0_36, ev1_36);
            float2 _f32x2_mul_a_126 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_36;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&_f2_49));
            float2 _f32x2_mul_a_127 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_36;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&_f2_49));
            vq_406[6] = vq2_36.x;
            vq_406[7] = vq2_36.y;
            vk_407[6] = vk2_36.x;
            vk_407[7] = vk2_36.y;
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
            float r_110[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_110[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0 = r_110[0];
            float r_111[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_111[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1 = r_111[0];
            float gs[8];
            float ks[8];
            float r_112[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_112[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[0] = r_112[0];
            unsigned int r_113[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_113[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[0] = __uint_as_float(r_113[0] << 16);
            float r_114[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_114[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[1] = r_114[0];
            unsigned int r_115[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_115[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[1] = __uint_as_float(r_115[0] << 16);
            float r_116[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_116[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[2] = r_116[0];
            unsigned int r_117[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_117[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[2] = __uint_as_float(r_117[0] << 16);
            float r_118[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_118[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[3] = r_118[0];
            unsigned int r_119[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_119[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[3] = __uint_as_float(r_119[0] << 16);
            float r_120[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_120[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[4] = r_120[0];
            unsigned int r_121[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_121[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[4] = __uint_as_float(r_121[0] << 16);
            float r_122[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_122[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[5] = r_122[0];
            unsigned int r_123[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_123[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[5] = __uint_as_float(r_123[0] << 16);
            float r_124[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_124[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[6] = r_124[0];
            unsigned int r_125[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_125[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[6] = __uint_as_float(r_125[0] << 16);
            float r_126[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_126[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[7] = r_126[0];
            unsigned int r_127[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_127[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[7] = __uint_as_float(r_127[0] << 16);
            float2 _f2_50 = make_float2(a0, a0);
            float v_1_0[8];
            float2 d2_37 = sub_f32x2(_f2_50, make_float2(gs[0], gs[1]));
            float e0_1_0 = approx_exp2(d2_37.x);
            float e1_1_0 = approx_exp2(d2_37.y);
            float2 _f32x2_mul_a_128 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_129 = make_float2(e0_1_0, e1_1_0);
            float2 v2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
            v_1_0[0] = v2_1_0.x;
            v_1_0[1] = v2_1_0.y;
            float2 d2_38 = sub_f32x2(_f2_50, make_float2(gs[2], gs[3]));
            float e0_1_1 = approx_exp2(d2_38.x);
            float e1_1_1 = approx_exp2(d2_38.y);
            float2 _f32x2_mul_a_130 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_131 = make_float2(e0_1_1, e1_1_1);
            float2 v2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
            v_1_0[2] = v2_1_1.x;
            v_1_0[3] = v2_1_1.y;
            float2 d2_39 = sub_f32x2(_f2_50, make_float2(gs[4], gs[5]));
            float e0_1_2 = approx_exp2(d2_39.x);
            float e1_1_2 = approx_exp2(d2_39.y);
            float2 _f32x2_mul_a_132 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_133 = make_float2(e0_1_2, e1_1_2);
            float2 v2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
            v_1_0[4] = v2_1_2.x;
            v_1_0[5] = v2_1_2.y;
            float2 d2_40 = sub_f32x2(_f2_50, make_float2(gs[6], gs[7]));
            float e0_1_3 = approx_exp2(d2_40.x);
            float e1_1_3 = approx_exp2(d2_40.y);
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
            float2 d2_41 = sub_f32x2(_f2_51, make_float2(gs[0], gs[1]));
            float e0_1_4 = approx_exp2(d2_41.x);
            float e1_1_4 = approx_exp2(d2_41.y);
            float2 _f32x2_mul_a_136 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_137 = make_float2(e0_1_4, e1_1_4);
            float2 v2_1_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
            v_1_1[0] = v2_1_4.x;
            v_1_1[1] = v2_1_4.y;
            float2 d2_42 = sub_f32x2(_f2_51, make_float2(gs[2], gs[3]));
            float e0_1_5 = approx_exp2(d2_42.x);
            float e1_1_5 = approx_exp2(d2_42.y);
            float2 _f32x2_mul_a_138 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_139 = make_float2(e0_1_5, e1_1_5);
            float2 v2_1_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
            v_1_1[2] = v2_1_5.x;
            v_1_1[3] = v2_1_5.y;
            float2 d2_43 = sub_f32x2(_f2_51, make_float2(gs[4], gs[5]));
            float e0_1_6 = approx_exp2(d2_43.x);
            float e1_1_6 = approx_exp2(d2_43.y);
            float2 _f32x2_mul_a_140 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_141 = make_float2(e0_1_6, e1_1_6);
            float2 v2_1_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
            v_1_1[4] = v2_1_6.x;
            v_1_1[5] = v2_1_6.y;
            float2 d2_44 = sub_f32x2(_f2_51, make_float2(gs[6], gs[7]));
            float e0_1_7 = approx_exp2(d2_44.x);
            float e1_1_7 = approx_exp2(d2_44.y);
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
            int sbase_428_0 = 16 + 8 * th;
            int gbase_429_0 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_428_0 * 128);
            int kbase_430_0 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_428_0 * 128);
            float r_128[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_128[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_0 = r_128[0];
            float r_129[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_129[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1_0 = r_129[0];
            float gs_0[8];
            float ks_0[8];
            float r_130[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_130[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[0] = r_130[0];
            unsigned int r_131[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_131[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[0] = __uint_as_float(r_131[0] << 16);
            float r_133[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_133[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[1] = r_133[0];
            unsigned int r_134[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_134[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[1] = __uint_as_float(r_134[0] << 16);
            float r_135[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_135[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[2] = r_135[0];
            unsigned int r_136[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_136[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[2] = __uint_as_float(r_136[0] << 16);
            float r_137[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_137[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[3] = r_137[0];
            unsigned int r_138[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_138[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[3] = __uint_as_float(r_138[0] << 16);
            float r_139[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_139[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[4] = r_139[0];
            unsigned int r_140[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_140[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[4] = __uint_as_float(r_140[0] << 16);
            float r_141[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_141[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[5] = r_141[0];
            unsigned int r_142[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_142[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[5] = __uint_as_float(r_142[0] << 16);
            float r_143[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_143[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[6] = r_143[0];
            unsigned int r_144[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_144[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[6] = __uint_as_float(r_144[0] << 16);
            float r_145[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_145[0])) : "r"((gbase_429_0 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_0[7] = r_145[0];
            unsigned int r_146[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_146[0])) : "r"((kbase_430_0 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_0[7] = __uint_as_float(r_146[0] << 16);
            float2 _f2_52 = make_float2(a0_0, a0_0);
            float v_1_2[8];
            float2 d2_45 = sub_f32x2(_f2_52, make_float2(gs_0[0], gs_0[1]));
            float e0_1_8 = approx_exp2(d2_45.x);
            float e1_1_8 = approx_exp2(d2_45.y);
            float2 _f32x2_mul_a_144 = make_float2(ks_0[0], ks_0[1]);
            float2 _f32x2_mul_b_145 = make_float2(e0_1_8, e1_1_8);
            float2 v2_1_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
            v_1_2[0] = v2_1_8.x;
            v_1_2[1] = v2_1_8.y;
            float2 d2_46 = sub_f32x2(_f2_52, make_float2(gs_0[2], gs_0[3]));
            float e0_1_9 = approx_exp2(d2_46.x);
            float e1_1_9 = approx_exp2(d2_46.y);
            float2 _f32x2_mul_a_146 = make_float2(ks_0[2], ks_0[3]);
            float2 _f32x2_mul_b_147 = make_float2(e0_1_9, e1_1_9);
            float2 v2_1_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
            v_1_2[2] = v2_1_9.x;
            v_1_2[3] = v2_1_9.y;
            float2 d2_47 = sub_f32x2(_f2_52, make_float2(gs_0[4], gs_0[5]));
            float e0_1_10 = approx_exp2(d2_47.x);
            float e1_1_10 = approx_exp2(d2_47.y);
            float2 _f32x2_mul_a_148 = make_float2(ks_0[4], ks_0[5]);
            float2 _f32x2_mul_b_149 = make_float2(e0_1_10, e1_1_10);
            float2 v2_1_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
            v_1_2[4] = v2_1_10.x;
            v_1_2[5] = v2_1_10.y;
            float2 d2_48 = sub_f32x2(_f2_52, make_float2(gs_0[6], gs_0[7]));
            float e0_1_11 = approx_exp2(d2_48.x);
            float e1_1_11 = approx_exp2(d2_48.y);
            float2 _f32x2_mul_a_150 = make_float2(ks_0[6], ks_0[7]);
            float2 _f32x2_mul_b_151 = make_float2(e0_1_11, e1_1_11);
            float2 v2_1_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
            v_1_2[6] = v2_1_11.x;
            v_1_2[7] = v2_1_11.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_2[0]), "f"(v_1_2[1]), "f"(v_1_2[2]), "f"(v_1_2[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_2[4]), "f"(v_1_2[5]), "f"(v_1_2[6]), "f"(v_1_2[7]) : "memory");
            float2 _f2_53 = make_float2(a1_0, a1_0);
            float v_1_3[8];
            float2 d2_49 = sub_f32x2(_f2_53, make_float2(gs_0[0], gs_0[1]));
            float e0_1_12 = approx_exp2(d2_49.x);
            float e1_1_12 = approx_exp2(d2_49.y);
            float2 _f32x2_mul_a_152 = make_float2(ks_0[0], ks_0[1]);
            float2 _f32x2_mul_b_153 = make_float2(e0_1_12, e1_1_12);
            float2 v2_1_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
            v_1_3[0] = v2_1_12.x;
            v_1_3[1] = v2_1_12.y;
            float2 d2_50 = sub_f32x2(_f2_53, make_float2(gs_0[2], gs_0[3]));
            float e0_1_13 = approx_exp2(d2_50.x);
            float e1_1_13 = approx_exp2(d2_50.y);
            float2 _f32x2_mul_a_154 = make_float2(ks_0[2], ks_0[3]);
            float2 _f32x2_mul_b_155 = make_float2(e0_1_13, e1_1_13);
            float2 v2_1_13;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
            v_1_3[2] = v2_1_13.x;
            v_1_3[3] = v2_1_13.y;
            float2 d2_51 = sub_f32x2(_f2_53, make_float2(gs_0[4], gs_0[5]));
            float e0_1_14 = approx_exp2(d2_51.x);
            float e1_1_14 = approx_exp2(d2_51.y);
            float2 _f32x2_mul_a_156 = make_float2(ks_0[4], ks_0[5]);
            float2 _f32x2_mul_b_157 = make_float2(e0_1_14, e1_1_14);
            float2 v2_1_14;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
            v_1_3[4] = v2_1_14.x;
            v_1_3[5] = v2_1_14.y;
            float2 d2_52 = sub_f32x2(_f2_53, make_float2(gs_0[6], gs_0[7]));
            float e0_1_15 = approx_exp2(d2_52.x);
            float e1_1_15 = approx_exp2(d2_52.y);
            float2 _f32x2_mul_a_158 = make_float2(ks_0[6], ks_0[7]);
            float2 _f32x2_mul_b_159 = make_float2(e0_1_15, e1_1_15);
            float2 v2_1_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
            v_1_3[6] = v2_1_15.x;
            v_1_3[7] = v2_1_15.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_3[0]), "f"(v_1_3[1]), "f"(v_1_3[2]), "f"(v_1_3[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1_3[4]), "f"(v_1_3[5]), "f"(v_1_3[6]), "f"(v_1_3[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_538 = 32 + 8 * th;
            int gbase_539 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_538 * 128);
            int kbase_540 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_538 * 128);
            float r_147[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_147[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_542 = r_147[0];
            float a1_543 = 0.0f;
            float gs_544[8];
            float ks_545[8];
            float r_148[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_148[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[0] = r_148[0];
            unsigned int r_149[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_149[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[0] = __uint_as_float(r_149[0] << 16);
            float r_150[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_150[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[1] = r_150[0];
            unsigned int r_151[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_151[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[1] = __uint_as_float(r_151[0] << 16);
            float r_152[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_152[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[2] = r_152[0];
            unsigned int r_153[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_153[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[2] = __uint_as_float(r_153[0] << 16);
            float r_154[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_154[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[3] = r_154[0];
            unsigned int r_155[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_155[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[3] = __uint_as_float(r_155[0] << 16);
            float r_156[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_156[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[4] = r_156[0];
            unsigned int r_157[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_157[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[4] = __uint_as_float(r_157[0] << 16);
            float r_158[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_158[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[5] = r_158[0];
            unsigned int r_159[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_159[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[5] = __uint_as_float(r_159[0] << 16);
            float r_160[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_160[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[6] = r_160[0];
            unsigned int r_161[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_161[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[6] = __uint_as_float(r_161[0] << 16);
            float r_162[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_162[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[7] = r_162[0];
            unsigned int r_163[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_163[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[7] = __uint_as_float(r_163[0] << 16);
            float2 _f2_54 = make_float2(a0_542, a0_542);
            float v_562[8];
            float2 d2_53 = sub_f32x2(_f2_54, make_float2(gs_544[0], gs_544[1]));
            float e0_0 = approx_exp2(d2_53.x);
            float e1_0 = approx_exp2(d2_53.y);
            float2 _f32x2_mul_a_160 = make_float2(ks_545[0], ks_545[1]);
            float2 _f32x2_mul_b_161 = make_float2(e0_0, e1_0);
            float2 v2_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
            v_562[0] = v2_0.x;
            v_562[1] = v2_0.y;
            float2 d2_54 = sub_f32x2(_f2_54, make_float2(gs_544[2], gs_544[3]));
            float e0_1 = approx_exp2(d2_54.x);
            float e1_1 = approx_exp2(d2_54.y);
            float2 _f32x2_mul_a_162 = make_float2(ks_545[2], ks_545[3]);
            float2 _f32x2_mul_b_163 = make_float2(e0_1, e1_1);
            float2 v2_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
            v_562[2] = v2_1.x;
            v_562[3] = v2_1.y;
            float2 d2_55 = sub_f32x2(_f2_54, make_float2(gs_544[4], gs_544[5]));
            float e0_2 = approx_exp2(d2_55.x);
            float e1_2 = approx_exp2(d2_55.y);
            float2 _f32x2_mul_a_164 = make_float2(ks_545[4], ks_545[5]);
            float2 _f32x2_mul_b_165 = make_float2(e0_2, e1_2);
            float2 v2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
            v_562[4] = v2_2.x;
            v_562[5] = v2_2.y;
            float2 d2_56 = sub_f32x2(_f2_54, make_float2(gs_544[6], gs_544[7]));
            float e0_3 = approx_exp2(d2_56.x);
            float e1_3 = approx_exp2(d2_56.y);
            float2 _f32x2_mul_a_166 = make_float2(ks_545[6], ks_545[7]);
            float2 _f32x2_mul_b_167 = make_float2(e0_3, e1_3);
            float2 v2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
            v_562[6] = v2_3.x;
            v_562[7] = v2_3.y;
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
            float r_164[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_164[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl = r_164[0];
            float r_165[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_165[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc = r_165[0] - 21.640425f;
            float dkt[8];
            float pan[8];
            float2 _f2_55 = make_float2(gc, gc);
            float2 _f2_56 = make_float2(gnl, gnl);
            int t0_1_0 = 32 + 8 * th;
            int t1_1_0 = 32 + 8 * th + 1;
            float r_166[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_166[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_0 = r_166[0];
            float r_167[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_167[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_0 = r_167[0];
            unsigned int r_168[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_168[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_0 = __uint_as_float(r_168[0] << 16);
            unsigned int r_169[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_169[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_0 = __uint_as_float(r_169[0] << 16);
            float2 _f2_57 = make_float2(g_t0_1_0, g_t1_1_0);
            float2 dc2_1_0 = sub_f32x2(_f2_55, _f2_57);
            float e2_3 = approx_exp2(dc2_1_0.x);
            float e2_2_0 = approx_exp2(dc2_1_0.y);
            float2 dn2_1_0 = sub_f32x2(_f2_56, _f2_57);
            float en_3 = approx_exp2(dn2_1_0.x);
            float en_2_0 = approx_exp2(dn2_1_0.y);
            float2 _f32x2_mul_a_168 = make_float2(k_t0_1_0, k_t1_1_0);
            float2 _f32x2_mul_b_169 = make_float2(e2_3, e2_2_0);
            float2 p2_1_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
            float dl = __uint_as_float(__as_u32(p2_1_0.x) & 4294959104u) - p2_1_0.x;
            float _rcp_170 = approx_rcp(k_t0_1_0);
            float ft = ((dl == 0.0f) ? e2_3 : __fmaf_rn(dl, _rcp_170, e2_3));
            float dl_0 = __uint_as_float(__as_u32(p2_1_0.y) & 4294959104u) - p2_1_0.y;
            float _rcp_171 = approx_rcp(k_t1_1_0);
            float ft_0 = ((dl_0 == 0.0f) ? e2_2_0 : __fmaf_rn(dl_0, _rcp_171, e2_2_0));
            dkt[0] = __fmaf_rn(_tmem_load_1[0], en_3, _tmem_load_0[0] * ft);
            dkt[1] = __fmaf_rn(_tmem_load_1[1], en_2_0, _tmem_load_0[1] * ft_0);
            pan[0] = p2_1_0.x;
            pan[1] = p2_1_0.y;
            int t0_1_1 = 32 + 8 * th + 2;
            int t1_1_1 = 32 + 8 * th + 2 + 1;
            float r_170[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_170[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_1 = r_170[0];
            float r_171[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_171[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_1 = r_171[0];
            unsigned int r_172[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_172[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_1 = __uint_as_float(r_172[0] << 16);
            unsigned int r_173[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_173[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_1 = __uint_as_float(r_173[0] << 16);
            float2 _f2_58 = make_float2(g_t0_1_1, g_t1_1_1);
            float2 dc2_1_1 = sub_f32x2(_f2_55, _f2_58);
            float e2_4 = approx_exp2(dc2_1_1.x);
            float e2_2_1 = approx_exp2(dc2_1_1.y);
            float2 dn2_1_1 = sub_f32x2(_f2_56, _f2_58);
            float en_4 = approx_exp2(dn2_1_1.x);
            float en_2_1 = approx_exp2(dn2_1_1.y);
            float2 _f32x2_mul_a_172 = make_float2(k_t0_1_1, k_t1_1_1);
            float2 _f32x2_mul_b_173 = make_float2(e2_4, e2_2_1);
            float2 p2_1_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
            float dl_1 = __uint_as_float(__as_u32(p2_1_1.x) & 4294959104u) - p2_1_1.x;
            float _rcp_174 = approx_rcp(k_t0_1_1);
            float ft_1 = ((dl_1 == 0.0f) ? e2_4 : __fmaf_rn(dl_1, _rcp_174, e2_4));
            float dl_2 = __uint_as_float(__as_u32(p2_1_1.y) & 4294959104u) - p2_1_1.y;
            float _rcp_175 = approx_rcp(k_t1_1_1);
            float ft_2 = ((dl_2 == 0.0f) ? e2_2_1 : __fmaf_rn(dl_2, _rcp_175, e2_2_1));
            dkt[2] = __fmaf_rn(_tmem_load_1[2], en_4, _tmem_load_0[2] * ft_1);
            dkt[3] = __fmaf_rn(_tmem_load_1[3], en_2_1, _tmem_load_0[3] * ft_2);
            pan[2] = p2_1_1.x;
            pan[3] = p2_1_1.y;
            int t0_1_2 = 32 + 8 * th + 4;
            int t1_1_2 = 32 + 8 * th + 4 + 1;
            float r_174[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_174[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_2 = r_174[0];
            float r_175[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_175[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_2 = r_175[0];
            unsigned int r_176[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_176[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_2 = __uint_as_float(r_176[0] << 16);
            unsigned int r_177[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_177[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_2 = __uint_as_float(r_177[0] << 16);
            float2 _f2_59 = make_float2(g_t0_1_2, g_t1_1_2);
            float2 dc2_1_2 = sub_f32x2(_f2_55, _f2_59);
            float e2_5 = approx_exp2(dc2_1_2.x);
            float e2_2_2 = approx_exp2(dc2_1_2.y);
            float2 dn2_1_2 = sub_f32x2(_f2_56, _f2_59);
            float en_5 = approx_exp2(dn2_1_2.x);
            float en_2_2 = approx_exp2(dn2_1_2.y);
            float2 _f32x2_mul_a_176 = make_float2(k_t0_1_2, k_t1_1_2);
            float2 _f32x2_mul_b_177 = make_float2(e2_5, e2_2_2);
            float2 p2_1_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
            float dl_3 = __uint_as_float(__as_u32(p2_1_2.x) & 4294959104u) - p2_1_2.x;
            float _rcp_178 = approx_rcp(k_t0_1_2);
            float ft_3 = ((dl_3 == 0.0f) ? e2_5 : __fmaf_rn(dl_3, _rcp_178, e2_5));
            float dl_4 = __uint_as_float(__as_u32(p2_1_2.y) & 4294959104u) - p2_1_2.y;
            float _rcp_179 = approx_rcp(k_t1_1_2);
            float ft_4 = ((dl_4 == 0.0f) ? e2_2_2 : __fmaf_rn(dl_4, _rcp_179, e2_2_2));
            dkt[4] = __fmaf_rn(_tmem_load_1[4], en_5, _tmem_load_0[4] * ft_3);
            dkt[5] = __fmaf_rn(_tmem_load_1[5], en_2_2, _tmem_load_0[5] * ft_4);
            pan[4] = p2_1_2.x;
            pan[5] = p2_1_2.y;
            int t0_1_3 = 32 + 8 * th + 6;
            int t1_1_3 = 32 + 8 * th + 6 + 1;
            float r_178[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_178[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1_3 = r_178[0];
            float r_179[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_179[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1_3 = r_179[0];
            unsigned int r_180[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_180[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1_3 = __uint_as_float(r_180[0] << 16);
            unsigned int r_181[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_181[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1_3 = __uint_as_float(r_181[0] << 16);
            float2 _f2_60 = make_float2(g_t0_1_3, g_t1_1_3);
            float2 dc2_1_3 = sub_f32x2(_f2_55, _f2_60);
            float e2_6 = approx_exp2(dc2_1_3.x);
            float e2_2_3 = approx_exp2(dc2_1_3.y);
            float2 dn2_1_3 = sub_f32x2(_f2_56, _f2_60);
            float en_6 = approx_exp2(dn2_1_3.x);
            float en_2_3 = approx_exp2(dn2_1_3.y);
            float2 _f32x2_mul_a_180 = make_float2(k_t0_1_3, k_t1_1_3);
            float2 _f32x2_mul_b_181 = make_float2(e2_6, e2_2_3);
            float2 p2_1_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&_f32x2_mul_b_181));
            float dl_5 = __uint_as_float(__as_u32(p2_1_3.x) & 4294959104u) - p2_1_3.x;
            float _rcp_182 = approx_rcp(k_t0_1_3);
            float ft_5 = ((dl_5 == 0.0f) ? e2_6 : __fmaf_rn(dl_5, _rcp_182, e2_6));
            float dl_6 = __uint_as_float(__as_u32(p2_1_3.y) & 4294959104u) - p2_1_3.y;
            float _rcp_183 = approx_rcp(k_t1_1_3);
            float ft_6 = ((dl_6 == 0.0f) ? e2_2_3 : __fmaf_rn(dl_6, _rcp_183, e2_2_3));
            dkt[6] = __fmaf_rn(_tmem_load_1[6], en_6, _tmem_load_0[6] * ft_5);
            dkt[7] = __fmaf_rn(_tmem_load_1[7], en_2_3, _tmem_load_0[7] * ft_6);
            pan[6] = p2_1_3.x;
            pan[7] = p2_1_3.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th), dkt);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan[0]), "f"(pan[1]), "f"(pan[2]), "f"(pan[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan[4]), "f"(pan[5]), "f"(pan[6]), "f"(pan[7]) : "memory");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (48 + 8 * th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_182[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_182[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_637 = r_182[0];
            float r_183[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_183[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_639 = r_183[0] - 21.640425f;
            float dkt_640[8];
            float pan_641[8];
            float2 _f2_61 = make_float2(gc_639, gc_639);
            float2 _f2_62 = make_float2(gnl_637, gnl_637);
            int t0_0 = 48 + 8 * th;
            int t1_0 = 48 + 8 * th + 1;
            float r_184[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_184[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_0 = r_184[0];
            float r_185[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_185[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_0 = r_185[0];
            unsigned int r_186[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_186[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_0 = __uint_as_float(r_186[0] << 16);
            unsigned int r_187[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_187[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_0 = __uint_as_float(r_187[0] << 16);
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
            int t0_2 = 48 + 8 * th + 2;
            int t1_2 = 48 + 8 * th + 2 + 1;
            float r_188[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_188[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_2 = r_188[0];
            float r_189[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_189[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_2 = r_189[0];
            unsigned int r_190[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_190[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_2 = __uint_as_float(r_190[0] << 16);
            unsigned int r_191[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_191[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_2 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_2 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_2 = __uint_as_float(r_191[0] << 16);
            float2 dc2_2 = sub_f32x2(_f2_61, make_float2(g_t0_2, g_t1_2));
            float e2_0_1 = approx_exp2(dc2_2.x);
            float e2_1_1 = approx_exp2(dc2_2.y);
            float2 _f32x2_mul_a_188 = make_float2(k_t0_2, k_t1_2);
            float2 _f32x2_mul_b_189 = make_float2(e2_0_1, e2_1_1);
            float2 p2_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
            float dl_9 = __uint_as_float(__as_u32(p2_2.x) & 4294959104u) - p2_2.x;
            float _rcp_190 = approx_rcp(k_t0_2);
            float ft_9 = ((dl_9 == 0.0f) ? e2_0_1 : __fmaf_rn(dl_9, _rcp_190, e2_0_1));
            float dl_10 = __uint_as_float(__as_u32(p2_2.y) & 4294959104u) - p2_2.y;
            float _rcp_191 = approx_rcp(k_t1_2);
            float ft_10 = ((dl_10 == 0.0f) ? e2_1_1 : __fmaf_rn(dl_10, _rcp_191, e2_1_1));
            dkt_640[2] = _tmem_load_2[2] * ft_9;
            dkt_640[3] = _tmem_load_2[3] * ft_10;
            pan_641[2] = p2_2.x;
            pan_641[3] = p2_2.y;
            int t0_3 = 48 + 8 * th + 4;
            int t1_3 = 48 + 8 * th + 4 + 1;
            float r_192[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_192[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_3 = r_192[0];
            float r_193[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_193[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_3 = r_193[0];
            unsigned int r_194[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_194[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_3 = __uint_as_float(r_194[0] << 16);
            unsigned int r_195[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_195[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_3 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_3 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_3 = __uint_as_float(r_195[0] << 16);
            float2 dc2_3 = sub_f32x2(_f2_61, make_float2(g_t0_3, g_t1_3));
            float e2_0_2 = approx_exp2(dc2_3.x);
            float e2_1_2 = approx_exp2(dc2_3.y);
            float2 _f32x2_mul_a_192 = make_float2(k_t0_3, k_t1_3);
            float2 _f32x2_mul_b_193 = make_float2(e2_0_2, e2_1_2);
            float2 p2_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&_f32x2_mul_b_193));
            float dl_11 = __uint_as_float(__as_u32(p2_3.x) & 4294959104u) - p2_3.x;
            float _rcp_194 = approx_rcp(k_t0_3);
            float ft_11 = ((dl_11 == 0.0f) ? e2_0_2 : __fmaf_rn(dl_11, _rcp_194, e2_0_2));
            float dl_12 = __uint_as_float(__as_u32(p2_3.y) & 4294959104u) - p2_3.y;
            float _rcp_195 = approx_rcp(k_t1_3);
            float ft_12 = ((dl_12 == 0.0f) ? e2_1_2 : __fmaf_rn(dl_12, _rcp_195, e2_1_2));
            dkt_640[4] = _tmem_load_2[4] * ft_11;
            dkt_640[5] = _tmem_load_2[5] * ft_12;
            pan_641[4] = p2_3.x;
            pan_641[5] = p2_3.y;
            int t0_4 = 48 + 8 * th + 6;
            int t1_4 = 48 + 8 * th + 6 + 1;
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
            float2 dc2_4 = sub_f32x2(_f2_61, make_float2(g_t0_4, g_t1_4));
            float e2_0_3 = approx_exp2(dc2_4.x);
            float e2_1_3 = approx_exp2(dc2_4.y);
            float2 _f32x2_mul_a_196 = make_float2(k_t0_4, k_t1_4);
            float2 _f32x2_mul_b_197 = make_float2(e2_0_3, e2_1_3);
            float2 p2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
            float dl_13 = __uint_as_float(__as_u32(p2_4.x) & 4294959104u) - p2_4.x;
            float _rcp_198 = approx_rcp(k_t0_4);
            float ft_13 = ((dl_13 == 0.0f) ? e2_0_3 : __fmaf_rn(dl_13, _rcp_198, e2_0_3));
            float dl_14 = __uint_as_float(__as_u32(p2_4.y) & 4294959104u) - p2_4.y;
            float _rcp_199 = approx_rcp(k_t1_4);
            float ft_14 = ((dl_14 == 0.0f) ? e2_1_3 : __fmaf_rn(dl_14, _rcp_199, e2_1_3));
            dkt_640[6] = _tmem_load_2[6] * ft_13;
            dkt_640[7] = _tmem_load_2[7] * ft_14;
            pan_641[6] = p2_4.x;
            pan_641[7] = p2_4.y;
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
            float r_200[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_200[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_699 = r_200[0];
            float r_201[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_201[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_701 = r_201[0] - 21.640425f;
            float dkt_702[8];
            float pan_703[8];
            float2 _f2_63 = make_float2(gc_701, gc_701);
            float2 _f2_64 = make_float2(gnl_699, gnl_699);
            int t0_5 = 8 * th;
            int t1_5 = 8 * th + 1;
            float r_202[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_202[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_5 = r_202[0];
            float r_203[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_203[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_5 = r_203[0];
            unsigned int r_204[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_204[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_5 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_5 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_5 = __uint_as_float(r_204[0] << 16);
            unsigned int r_205[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_205[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_5 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_5 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_5 = __uint_as_float(r_205[0] << 16);
            float2 _f2_65 = make_float2(g_t0_5, g_t1_5);
            float2 dc2_5 = sub_f32x2(_f2_63, _f2_65);
            float e2_0_4 = approx_exp2(dc2_5.x);
            float e2_1_4 = approx_exp2(dc2_5.y);
            float2 dn2_0 = sub_f32x2(_f2_64, _f2_65);
            float en_0_0 = approx_exp2(dn2_0.x);
            float en_1_0 = approx_exp2(dn2_0.y);
            float2 _f32x2_mul_a_200 = make_float2(k_t0_5, k_t1_5);
            float2 _f32x2_mul_b_201 = make_float2(e2_0_4, e2_1_4);
            float2 p2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
            float dl_15 = __uint_as_float(__as_u32(p2_5.x) & 4294959104u) - p2_5.x;
            float _rcp_202 = approx_rcp(k_t0_5);
            float ft_15 = ((dl_15 == 0.0f) ? e2_0_4 : __fmaf_rn(dl_15, _rcp_202, e2_0_4));
            float dl_16 = __uint_as_float(__as_u32(p2_5.y) & 4294959104u) - p2_5.y;
            float _rcp_203 = approx_rcp(k_t1_5);
            float ft_16 = ((dl_16 == 0.0f) ? e2_1_4 : __fmaf_rn(dl_16, _rcp_203, e2_1_4));
            dkt_702[0] = __fmaf_rn(_tmem_load_4[0], en_0_0, _tmem_load_3[0] * ft_15);
            dkt_702[1] = __fmaf_rn(_tmem_load_4[1], en_1_0, _tmem_load_3[1] * ft_16);
            pan_703[0] = p2_5.x;
            pan_703[1] = p2_5.y;
            int t0_6 = 8 * th + 2;
            int t1_6 = 8 * th + 2 + 1;
            float r_206[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_206[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_6 = r_206[0];
            float r_207[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_207[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_6 = r_207[0];
            unsigned int r_209[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_209[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_6 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_6 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_6 = __uint_as_float(r_209[0] << 16);
            unsigned int r_210[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_210[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_6 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_6 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_6 = __uint_as_float(r_210[0] << 16);
            float2 _f2_66 = make_float2(g_t0_6, g_t1_6);
            float2 dc2_6 = sub_f32x2(_f2_63, _f2_66);
            float e2_0_5 = approx_exp2(dc2_6.x);
            float e2_1_5 = approx_exp2(dc2_6.y);
            float2 dn2_2 = sub_f32x2(_f2_64, _f2_66);
            float en_0_1 = approx_exp2(dn2_2.x);
            float en_1_1 = approx_exp2(dn2_2.y);
            float2 _f32x2_mul_a_204 = make_float2(k_t0_6, k_t1_6);
            float2 _f32x2_mul_b_205 = make_float2(e2_0_5, e2_1_5);
            float2 p2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
            float dl_17 = __uint_as_float(__as_u32(p2_6.x) & 4294959104u) - p2_6.x;
            float _rcp_206 = approx_rcp(k_t0_6);
            float ft_17 = ((dl_17 == 0.0f) ? e2_0_5 : __fmaf_rn(dl_17, _rcp_206, e2_0_5));
            float dl_18 = __uint_as_float(__as_u32(p2_6.y) & 4294959104u) - p2_6.y;
            float _rcp_207 = approx_rcp(k_t1_6);
            float ft_18 = ((dl_18 == 0.0f) ? e2_1_5 : __fmaf_rn(dl_18, _rcp_207, e2_1_5));
            dkt_702[2] = __fmaf_rn(_tmem_load_4[2], en_0_1, _tmem_load_3[2] * ft_17);
            dkt_702[3] = __fmaf_rn(_tmem_load_4[3], en_1_1, _tmem_load_3[3] * ft_18);
            pan_703[2] = p2_6.x;
            pan_703[3] = p2_6.y;
            int t0_7 = 8 * th + 4;
            int t1_7 = 8 * th + 4 + 1;
            float r_211[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_211[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_7 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_7 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_7 = r_211[0];
            float r_212[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_212[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_7 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_7 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_7 = r_212[0];
            unsigned int r_213[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_213[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_7 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_7 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_7 = __uint_as_float(r_213[0] << 16);
            unsigned int r_214[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_214[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_7 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_7 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_7 = __uint_as_float(r_214[0] << 16);
            float2 _f2_67 = make_float2(g_t0_7, g_t1_7);
            float2 dc2_7 = sub_f32x2(_f2_63, _f2_67);
            float e2_0_6 = approx_exp2(dc2_7.x);
            float e2_1_6 = approx_exp2(dc2_7.y);
            float2 dn2_3 = sub_f32x2(_f2_64, _f2_67);
            float en_0_2 = approx_exp2(dn2_3.x);
            float en_1_2 = approx_exp2(dn2_3.y);
            float2 _f32x2_mul_a_208 = make_float2(k_t0_7, k_t1_7);
            float2 _f32x2_mul_b_209 = make_float2(e2_0_6, e2_1_6);
            float2 p2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
            float dl_19 = __uint_as_float(__as_u32(p2_7.x) & 4294959104u) - p2_7.x;
            float _rcp_210 = approx_rcp(k_t0_7);
            float ft_19 = ((dl_19 == 0.0f) ? e2_0_6 : __fmaf_rn(dl_19, _rcp_210, e2_0_6));
            float dl_20 = __uint_as_float(__as_u32(p2_7.y) & 4294959104u) - p2_7.y;
            float _rcp_211 = approx_rcp(k_t1_7);
            float ft_20 = ((dl_20 == 0.0f) ? e2_1_6 : __fmaf_rn(dl_20, _rcp_211, e2_1_6));
            dkt_702[4] = __fmaf_rn(_tmem_load_4[4], en_0_2, _tmem_load_3[4] * ft_19);
            dkt_702[5] = __fmaf_rn(_tmem_load_4[5], en_1_2, _tmem_load_3[5] * ft_20);
            pan_703[4] = p2_7.x;
            pan_703[5] = p2_7.y;
            int t0_8 = 8 * th + 6;
            int t1_8 = 8 * th + 6 + 1;
            float r_215[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_215[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_8 = r_215[0];
            float r_216[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_216[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_8 = r_216[0];
            unsigned int r_217[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_217[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_8 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_8 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_8 = __uint_as_float(r_217[0] << 16);
            unsigned int r_218[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_218[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_8 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_8 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_8 = __uint_as_float(r_218[0] << 16);
            float2 _f2_68 = make_float2(g_t0_8, g_t1_8);
            float2 dc2_8 = sub_f32x2(_f2_63, _f2_68);
            float e2_0_7 = approx_exp2(dc2_8.x);
            float e2_1_7 = approx_exp2(dc2_8.y);
            float2 dn2_4 = sub_f32x2(_f2_64, _f2_68);
            float en_0_3 = approx_exp2(dn2_4.x);
            float en_1_3 = approx_exp2(dn2_4.y);
            float2 _f32x2_mul_a_212 = make_float2(k_t0_8, k_t1_8);
            float2 _f32x2_mul_b_213 = make_float2(e2_0_7, e2_1_7);
            float2 p2_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
            float dl_21 = __uint_as_float(__as_u32(p2_8.x) & 4294959104u) - p2_8.x;
            float _rcp_214 = approx_rcp(k_t0_8);
            float ft_21 = ((dl_21 == 0.0f) ? e2_0_7 : __fmaf_rn(dl_21, _rcp_214, e2_0_7));
            float dl_22 = __uint_as_float(__as_u32(p2_8.y) & 4294959104u) - p2_8.y;
            float _rcp_215 = approx_rcp(k_t1_8);
            float ft_22 = ((dl_22 == 0.0f) ? e2_1_7 : __fmaf_rn(dl_22, _rcp_215, e2_1_7));
            dkt_702[6] = __fmaf_rn(_tmem_load_4[6], en_0_3, _tmem_load_3[6] * ft_21);
            dkt_702[7] = __fmaf_rn(_tmem_load_4[7], en_1_3, _tmem_load_3[7] * ft_22);
            pan_703[6] = p2_8.x;
            pan_703[7] = p2_8.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(8 * th), dkt_702);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[0]), "f"(pan_703[1]), "f"(pan_703[2]), "f"(pan_703[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[4]), "f"(pan_703[5]), "f"(pan_703[6]), "f"(pan_703[7]) : "memory");
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (16 + 8 * th)));
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_219[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_219[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_773 = r_219[0];
            float r_220[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_220[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_775 = r_220[0] - 21.640425f;
            float dkt_776[8];
            float pan_777[8];
            float2 _f2_69 = make_float2(gc_775, gc_775);
            float2 _f2_70 = make_float2(gnl_773, gnl_773);
            int t0_9 = 16 + 8 * th;
            int t1_9 = 16 + 8 * th + 1;
            float r_221[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_221[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_9 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_9 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_9 = r_221[0];
            float r_222[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_222[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_9 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_9 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_9 = r_222[0];
            unsigned int r_223[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_223[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_9 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_9 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_9 = __uint_as_float(r_223[0] << 16);
            unsigned int r_224[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_224[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_9 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_9 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_9 = __uint_as_float(r_224[0] << 16);
            float2 _f2_71 = make_float2(g_t0_9, g_t1_9);
            float2 dc2_9 = sub_f32x2(_f2_69, _f2_71);
            float e2_0_8 = approx_exp2(dc2_9.x);
            float e2_1_8 = approx_exp2(dc2_9.y);
            float2 dn2_5 = sub_f32x2(_f2_70, _f2_71);
            float en_0_4 = approx_exp2(dn2_5.x);
            float en_1_4 = approx_exp2(dn2_5.y);
            float2 _f32x2_mul_a_216 = make_float2(k_t0_9, k_t1_9);
            float2 _f32x2_mul_b_217 = make_float2(e2_0_8, e2_1_8);
            float2 p2_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
            float dl_23 = __uint_as_float(__as_u32(p2_9.x) & 4294959104u) - p2_9.x;
            float _rcp_218 = approx_rcp(k_t0_9);
            float ft_23 = ((dl_23 == 0.0f) ? e2_0_8 : __fmaf_rn(dl_23, _rcp_218, e2_0_8));
            float dl_24 = __uint_as_float(__as_u32(p2_9.y) & 4294959104u) - p2_9.y;
            float _rcp_219 = approx_rcp(k_t1_9);
            float ft_24 = ((dl_24 == 0.0f) ? e2_1_8 : __fmaf_rn(dl_24, _rcp_219, e2_1_8));
            dkt_776[0] = __fmaf_rn(_tmem_load_6[0], en_0_4, _tmem_load_5[0] * ft_23);
            dkt_776[1] = __fmaf_rn(_tmem_load_6[1], en_1_4, _tmem_load_5[1] * ft_24);
            pan_777[0] = p2_9.x;
            pan_777[1] = p2_9.y;
            int t0_10 = 16 + 8 * th + 2;
            int t1_10 = 16 + 8 * th + 2 + 1;
            float r_225[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_225[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_10 = r_225[0];
            float r_226[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_226[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_10 = r_226[0];
            unsigned int r_227[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_227[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_10 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_10 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_10 = __uint_as_float(r_227[0] << 16);
            unsigned int r_228[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_228[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_10 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_10 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_10 = __uint_as_float(r_228[0] << 16);
            float2 _f2_72 = make_float2(g_t0_10, g_t1_10);
            float2 dc2_10 = sub_f32x2(_f2_69, _f2_72);
            float e2_0_9 = approx_exp2(dc2_10.x);
            float e2_1_9 = approx_exp2(dc2_10.y);
            float2 dn2_6 = sub_f32x2(_f2_70, _f2_72);
            float en_0_5 = approx_exp2(dn2_6.x);
            float en_1_5 = approx_exp2(dn2_6.y);
            float2 _f32x2_mul_a_220 = make_float2(k_t0_10, k_t1_10);
            float2 _f32x2_mul_b_221 = make_float2(e2_0_9, e2_1_9);
            float2 p2_10;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
            float dl_25 = __uint_as_float(__as_u32(p2_10.x) & 4294959104u) - p2_10.x;
            float _rcp_222 = approx_rcp(k_t0_10);
            float ft_25 = ((dl_25 == 0.0f) ? e2_0_9 : __fmaf_rn(dl_25, _rcp_222, e2_0_9));
            float dl_26 = __uint_as_float(__as_u32(p2_10.y) & 4294959104u) - p2_10.y;
            float _rcp_223 = approx_rcp(k_t1_10);
            float ft_26 = ((dl_26 == 0.0f) ? e2_1_9 : __fmaf_rn(dl_26, _rcp_223, e2_1_9));
            dkt_776[2] = __fmaf_rn(_tmem_load_6[2], en_0_5, _tmem_load_5[2] * ft_25);
            dkt_776[3] = __fmaf_rn(_tmem_load_6[3], en_1_5, _tmem_load_5[3] * ft_26);
            pan_777[2] = p2_10.x;
            pan_777[3] = p2_10.y;
            int t0_11 = 16 + 8 * th + 4;
            int t1_11 = 16 + 8 * th + 4 + 1;
            float r_229[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_229[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_11 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_11 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_11 = r_229[0];
            float r_230[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_230[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_11 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_11 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_11 = r_230[0];
            unsigned int r_231[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_231[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_11 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_11 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_11 = __uint_as_float(r_231[0] << 16);
            unsigned int r_233[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_233[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_11 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_11 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_11 = __uint_as_float(r_233[0] << 16);
            float2 _f2_73 = make_float2(g_t0_11, g_t1_11);
            float2 dc2_11 = sub_f32x2(_f2_69, _f2_73);
            float e2_0_10 = approx_exp2(dc2_11.x);
            float e2_1_10 = approx_exp2(dc2_11.y);
            float2 dn2_7 = sub_f32x2(_f2_70, _f2_73);
            float en_0_6 = approx_exp2(dn2_7.x);
            float en_1_6 = approx_exp2(dn2_7.y);
            float2 _f32x2_mul_a_224 = make_float2(k_t0_11, k_t1_11);
            float2 _f32x2_mul_b_225 = make_float2(e2_0_10, e2_1_10);
            float2 p2_11;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
            float dl_27 = __uint_as_float(__as_u32(p2_11.x) & 4294959104u) - p2_11.x;
            float _rcp_226 = approx_rcp(k_t0_11);
            float ft_27 = ((dl_27 == 0.0f) ? e2_0_10 : __fmaf_rn(dl_27, _rcp_226, e2_0_10));
            float dl_28 = __uint_as_float(__as_u32(p2_11.y) & 4294959104u) - p2_11.y;
            float _rcp_227 = approx_rcp(k_t1_11);
            float ft_28 = ((dl_28 == 0.0f) ? e2_1_10 : __fmaf_rn(dl_28, _rcp_227, e2_1_10));
            dkt_776[4] = __fmaf_rn(_tmem_load_6[4], en_0_6, _tmem_load_5[4] * ft_27);
            dkt_776[5] = __fmaf_rn(_tmem_load_6[5], en_1_6, _tmem_load_5[5] * ft_28);
            pan_777[4] = p2_11.x;
            pan_777[5] = p2_11.y;
            int t0_12 = 16 + 8 * th + 6;
            int t1_12 = 16 + 8 * th + 6 + 1;
            float r_234[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_234[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_12 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_12 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_12 = r_234[0];
            float r_235[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_235[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_12 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_12 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_12 = r_235[0];
            unsigned int r_236[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_236[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_12 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_12 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_12 = __uint_as_float(r_236[0] << 16);
            unsigned int r_237[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_237[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_12 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_12 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_12 = __uint_as_float(r_237[0] << 16);
            float2 _f2_74 = make_float2(g_t0_12, g_t1_12);
            float2 dc2_12 = sub_f32x2(_f2_69, _f2_74);
            float e2_0_11 = approx_exp2(dc2_12.x);
            float e2_1_11 = approx_exp2(dc2_12.y);
            float2 dn2_8 = sub_f32x2(_f2_70, _f2_74);
            float en_0_7 = approx_exp2(dn2_8.x);
            float en_1_7 = approx_exp2(dn2_8.y);
            float2 _f32x2_mul_a_228 = make_float2(k_t0_12, k_t1_12);
            float2 _f32x2_mul_b_229 = make_float2(e2_0_11, e2_1_11);
            float2 p2_12;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
            float dl_29 = __uint_as_float(__as_u32(p2_12.x) & 4294959104u) - p2_12.x;
            float _rcp_230 = approx_rcp(k_t0_12);
            float ft_29 = ((dl_29 == 0.0f) ? e2_0_11 : __fmaf_rn(dl_29, _rcp_230, e2_0_11));
            float dl_30 = __uint_as_float(__as_u32(p2_12.y) & 4294959104u) - p2_12.y;
            float _rcp_231 = approx_rcp(k_t1_12);
            float ft_30 = ((dl_30 == 0.0f) ? e2_1_11 : __fmaf_rn(dl_30, _rcp_231, e2_1_11));
            dkt_776[6] = __fmaf_rn(_tmem_load_6[6], en_0_7, _tmem_load_5[6] * ft_29);
            dkt_776[7] = __fmaf_rn(_tmem_load_6[7], en_1_7, _tmem_load_5[7] * ft_30);
            pan_777[6] = p2_12.x;
            pan_777[7] = p2_12.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th), dkt_776);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[0]), "f"(pan_777[1]), "f"(pan_777[2]), "f"(pan_777[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[4]), "f"(pan_777[5]), "f"(pan_777[6]), "f"(pan_777[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_538_0 = 8 * th;
            int gbase_539_0 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_538_0 * 128);
            int kbase_540_0 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_538_0 * 128);
            float r_238[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_238[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_542_0 = r_238[0];
            float a1_543_0 = 0.0f;
            float gs_544_0[8];
            float ks_545_0[8];
            float r_239[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_239[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[0] = r_239[0];
            unsigned int r_240[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_240[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[0] = __uint_as_float(r_240[0] << 16);
            float r_241[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_241[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[1] = r_241[0];
            unsigned int r_242[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_242[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[1] = __uint_as_float(r_242[0] << 16);
            float r_243[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_243[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[2] = r_243[0];
            unsigned int r_244[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_244[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[2] = __uint_as_float(r_244[0] << 16);
            float r_245[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_245[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[3] = r_245[0];
            unsigned int r_246[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_246[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[3] = __uint_as_float(r_246[0] << 16);
            float r_247[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_247[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[4] = r_247[0];
            unsigned int r_248[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_248[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[4] = __uint_as_float(r_248[0] << 16);
            float r_249[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_249[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[5] = r_249[0];
            unsigned int r_250[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_250[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[5] = __uint_as_float(r_250[0] << 16);
            float r_251[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_251[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[6] = r_251[0];
            unsigned int r_252[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_252[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[6] = __uint_as_float(r_252[0] << 16);
            float r_253[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_253[0])) : "r"((gbase_539_0 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544_0[7] = r_253[0];
            unsigned int r_254[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_254[0])) : "r"((kbase_540_0 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545_0[7] = __uint_as_float(r_254[0] << 16);
            float2 _f2_54_0 = make_float2(a0_542_0, a0_542_0);
            float v_562_0[8];
            float2 d2_57 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[0], gs_544_0[1]));
            float e0_4 = approx_exp2(d2_57.x);
            float e1_4 = approx_exp2(d2_57.y);
            float2 _f32x2_mul_a_232 = make_float2(ks_545_0[0], ks_545_0[1]);
            float2 _f32x2_mul_b_233 = make_float2(e0_4, e1_4);
            float2 v2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&_f32x2_mul_b_233));
            v_562_0[0] = v2_4.x;
            v_562_0[1] = v2_4.y;
            float2 d2_58 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[2], gs_544_0[3]));
            float e0_5 = approx_exp2(d2_58.x);
            float e1_5 = approx_exp2(d2_58.y);
            float2 _f32x2_mul_a_234 = make_float2(ks_545_0[2], ks_545_0[3]);
            float2 _f32x2_mul_b_235 = make_float2(e0_5, e1_5);
            float2 v2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
            v_562_0[2] = v2_5.x;
            v_562_0[3] = v2_5.y;
            float2 d2_59 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[4], gs_544_0[5]));
            float e0_6 = approx_exp2(d2_59.x);
            float e1_6 = approx_exp2(d2_59.y);
            float2 _f32x2_mul_a_236 = make_float2(ks_545_0[4], ks_545_0[5]);
            float2 _f32x2_mul_b_237 = make_float2(e0_6, e1_6);
            float2 v2_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&_f32x2_mul_b_237));
            v_562_0[4] = v2_6.x;
            v_562_0[5] = v2_6.y;
            float2 d2_60 = sub_f32x2(_f2_54_0, make_float2(gs_544_0[6], gs_544_0[7]));
            float e0_7 = approx_exp2(d2_60.x);
            float e1_7 = approx_exp2(d2_60.y);
            float2 _f32x2_mul_a_238 = make_float2(ks_545_0[6], ks_545_0[7]);
            float2 _f32x2_mul_b_239 = make_float2(e0_7, e1_7);
            float2 v2_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
            v_562_0[6] = v2_7.x;
            v_562_0[7] = v2_7.y;
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
            float r_255[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_255[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn = r_255[0];
            float gc_0 = gn - 21.640425f;
            int eb = 98304 + th * -32768;
            int ew = eb >> 2;
            int qbase_0 = pool_f32_addr + (unsigned int)(eb + 12288 + dh_lane * 1024);
            int sb8 = 32 + 8 * th;
            int gb8 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8 * 128);
            int kb8 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8 * 128);
            int c_qa = 128 + 8 * th;
            int c_ka = 144 + 8 * th;
            int c_qd = 192 + 8 * th;
            int c_kd = 208 + 8 * th;
            int c_dk = 32 + 8 * th;
            int c_bp = c_ka;
            float s_run[1];
            s_run[0] = 0.0f;
            #pragma unroll 1
            for (int k = 0; k < 8; k++) {
                int j = 7 - k;
                float a_qd[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd + j)));
                float a_kd[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd + j)));
                float a_dk[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk + j)));
                float a_qa[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qa[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qa + j)));
                float a_ka[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_ka[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_ka + j)));
                float btr[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8 + j))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t = sb8 + j;
                float r_257[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_257[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 128 * j + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t = r_257[0];
                unsigned int r_258[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_258[0])) : "r"((kb8 + (dcol / 64 * 8192 + 128 * j + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t = __uint_as_float(r_258[0] << 16);
                unsigned int r_259[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_259[0])) : "r"((qbase_0 + (dcol / 64 * 1024 + 128 * j + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t = __uint_as_float(r_259[0] << 16);
                float q_m = ((t >= clen) ? 0.0f : q_t);
                float dqf = pool_f32[ew + 128 * j + d_lane];
                float dkf = pool_f32[ew + (1024 + 128 * j) + d_lane];
                float dgf = pool_f32[ew + (2048 + 128 * j) + d_lane];
                float bt_t = btr[0];
                float e_dg = approx_exp2(g_t - gc_0);
                float e_off = approx_exp2(g_t - gn);
                float dq2 = __fmaf_rn(a_qa[0], e_off, a_qd[0] * e_dg);
                float dk2 = __fmaf_rn(a_ka[0], e_off, a_kd[0] * e_dg);
                float dbv[1];
                dbv[0] = dk2 * k_t;
                float dk2b = dk2 * bt_t;
                float dq_o = dq2 + dqf;
                float dk_o = dk2b + dkf + a_dk[0];
                float qe = q_m * e_dg;
                float kp = k_t * bt_t;
                __nv_bfloat16 kb = (__nv_bfloat16)kp;
                float kbe = (float)kb * e_dg;
                float cr = __fmaf_rn(__uint_as_float(__as_u32(qe) & 4294959104u) - qe, a_qd[0], (__uint_as_float(__as_u32(kbe) & 4294959104u) - kp * e_dg) * a_kd[0]);
                float dg_o = __fmaf_rn(q_m, dq2, __fmaf_rn(dk2b - a_dk[0], k_t, cr)) + dgf;
                s_run[0] = s_run[0] + dg_o;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd + j)), "f"(s_run[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp + j)), "f"(dbv[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int qoff = t * qrs;
                if (t < clen) {
                    {
                        auto _gst_248 = dq_base + qoff;
                        __builtin_assume(__isGlobal(_gst_248));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_248) + (0)) = (__nv_bfloat16)dq_o;
                    }
                    {
                        auto _gst_249 = dk_base + qoff;
                        __builtin_assume(__isGlobal(_gst_249));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_249) + (0)) = (__nv_bfloat16)dk_o;
                    }
                }
            }
            float tot = s_run[0];
            float dbp[8];
            tmem_ld_x8(&dbp[0], taddr + (unsigned int)row_base + (unsigned int)c_bp);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4 = lane_1 >> 4 & 1;
            int b3 = lane_1 >> 3 & 1;
            int b2 = lane_1 >> 2 & 1;
            float v4[4];
            float send_0 = ((b4 == 0) ? dbp[4] : dbp[0]);
            float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
            v4[0] = ((b4 == 0) ? dbp[0] : dbp[4]) + _shfl_xor_sync_250;
            float send_2 = ((b4 == 0) ? dbp[5] : dbp[1]);
            float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
            v4[1] = ((b4 == 0) ? dbp[1] : dbp[5]) + _shfl_xor_sync_251;
            float send_3 = ((b4 == 0) ? dbp[6] : dbp[2]);
            float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
            v4[2] = ((b4 == 0) ? dbp[2] : dbp[6]) + _shfl_xor_sync_252;
            float send_4 = ((b4 == 0) ? dbp[7] : dbp[3]);
            float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
            v4[3] = ((b4 == 0) ? dbp[3] : dbp[7]) + _shfl_xor_sync_253;
            float v2[2];
            float send2_0 = ((b3 == 0) ? v4[2] : v4[0]);
            float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, send2_0, 8);
            v2[0] = ((b3 == 0) ? v4[0] : v4[2]) + _shfl_xor_sync_254;
            float send2_1 = ((b3 == 0) ? v4[3] : v4[1]);
            float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, send2_1, 8);
            v2[1] = ((b3 == 0) ? v4[1] : v4[3]) + _shfl_xor_sync_255;
            float send1 = ((b2 == 0) ? v2[1] : v2[0]);
            float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, send1, 4);
            float v1 = ((b2 == 0) ? v2[0] : v2[1]) + _shfl_xor_sync_256;
            float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
            v1 = v1 + _shfl_xor_sync_257;
            float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
            v1 = v1 + _shfl_xor_sync_258;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1a_t2_drained_addr);
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            mbarrier_wait(epi_full_addr + (th) * 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_260[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_260[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_0 = r_260[0];
            float gc_1 = gn_0 - 21.640425f;
            int eb_0 = 98304 + th * -32768;
            int ew_0 = eb_0 >> 2;
            int qbase_1 = pool_f32_addr + (unsigned int)(eb_0 + 12288 + dh_lane * 1024);
            int sb8_0 = 48 + 8 * th;
            int gb8_0 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_0 * 128);
            int kb8_0 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_0 * 128);
            int c_qa_0 = 160 + 8 * th;
            int c_ka_0 = 176 + 8 * th;
            int c_qd_0 = 224 + 8 * th;
            int c_kd_0 = 240 + 8 * th;
            int c_dk_0 = 48 + 8 * th;
            int c_bp_0 = c_ka_0;
            float s_run_0[1];
            s_run_0[0] = 0.0f;
            #pragma unroll 1
            for (int k_1 = 0; k_1 < 8; k_1++) {
                int j_1 = 7 - k_1;
                float a_qd_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_0 + j_1)));
                float a_kd_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd_0 + j_1)));
                float a_dk_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk_0 + j_1)));
                float a_qa_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qa_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qa_0 + j_1)));
                float a_ka_1[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_ka_1[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_ka_0 + j_1)));
                float btr_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr_1[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_0 + j_1))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t_1 = sb8_0 + j_1;
                float r_261[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_261[0])) : "r"((gb8_0 + (lane_1 / 32 * 8192 + 128 * j_1 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_1 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t_1 = r_261[0];
                unsigned int r_262[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_262[0])) : "r"((kb8_0 + (dcol / 64 * 8192 + 128 * j_1 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_1 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t_1 = __uint_as_float(r_262[0] << 16);
                unsigned int r_263[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_263[0])) : "r"((qbase_1 + (dcol / 64 * 1024 + 128 * j_1 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_1 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t_1 = __uint_as_float(r_263[0] << 16);
                float q_m_1 = ((t_1 >= clen) ? 0.0f : q_t_1);
                float dqf_1 = pool_f32[ew_0 + 128 * j_1 + d_lane];
                float dkf_1 = pool_f32[ew_0 + (1024 + 128 * j_1) + d_lane];
                float dgf_1 = pool_f32[ew_0 + (2048 + 128 * j_1) + d_lane];
                float bt_t_1 = btr_1[0];
                float e_dg_1 = approx_exp2(g_t_1 - gc_1);
                float e_off_1 = approx_exp2(g_t_1 - gn_0);
                float dq2_1 = __fmaf_rn(a_qa_1[0], e_off_1, a_qd_1[0] * e_dg_1);
                float dk2_1 = __fmaf_rn(a_ka_1[0], e_off_1, a_kd_1[0] * e_dg_1);
                float dbv_1[1];
                dbv_1[0] = dk2_1 * k_t_1;
                float dk2b_1 = dk2_1 * bt_t_1;
                float dq_o_1 = dq2_1 + dqf_1;
                float dk_o_1 = dk2b_1 + dkf_1 + a_dk_1[0];
                float qe_1 = q_m_1 * e_dg_1;
                float kp_12 = k_t_1 * bt_t_1;
                __nv_bfloat16 kb_1 = (__nv_bfloat16)kp_12;
                float kbe_1 = (float)kb_1 * e_dg_1;
                float cr_1 = __fmaf_rn(__uint_as_float(__as_u32(qe_1) & 4294959104u) - qe_1, a_qd_1[0], (__uint_as_float(__as_u32(kbe_1) & 4294959104u) - kp_12 * e_dg_1) * a_kd_1[0]);
                float dg_o_1 = __fmaf_rn(q_m_1, dq2_1, __fmaf_rn(dk2b_1 - a_dk_1[0], k_t_1, cr_1)) + dgf_1;
                s_run_0[0] = s_run_0[0] + dg_o_1;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_0 + j_1)), "f"(s_run_0[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp_0 + j_1)), "f"(dbv_1[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int qoff_1 = t_1 * qrs;
                if (t_1 < clen) {
                    {
                        auto _gst_259 = dq_base + qoff_1;
                        __builtin_assume(__isGlobal(_gst_259));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_259) + (0)) = (__nv_bfloat16)dq_o_1;
                    }
                    {
                        auto _gst_260 = dk_base + qoff_1;
                        __builtin_assume(__isGlobal(_gst_260));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_260) + (0)) = (__nv_bfloat16)dk_o_1;
                    }
                }
            }
            float tot_0 = s_run_0[0];
            float dbp_0[8];
            tmem_ld_x8(&dbp_0[0], taddr + (unsigned int)row_base + (unsigned int)c_bp_0);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_0 = lane_1 >> 4 & 1;
            int b3_0 = lane_1 >> 3 & 1;
            int b2_0 = lane_1 >> 2 & 1;
            float v4_0[4];
            float send_5 = ((b4_0 == 0) ? dbp_0[4] : dbp_0[0]);
            float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
            v4_0[0] = ((b4_0 == 0) ? dbp_0[0] : dbp_0[4]) + _shfl_xor_sync_261;
            float send_6 = ((b4_0 == 0) ? dbp_0[5] : dbp_0[1]);
            float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
            v4_0[1] = ((b4_0 == 0) ? dbp_0[1] : dbp_0[5]) + _shfl_xor_sync_262;
            float send_7 = ((b4_0 == 0) ? dbp_0[6] : dbp_0[2]);
            float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
            v4_0[2] = ((b4_0 == 0) ? dbp_0[2] : dbp_0[6]) + _shfl_xor_sync_263;
            float send_8 = ((b4_0 == 0) ? dbp_0[7] : dbp_0[3]);
            float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
            v4_0[3] = ((b4_0 == 0) ? dbp_0[3] : dbp_0[7]) + _shfl_xor_sync_264;
            float v2_8[2];
            float send2_2 = ((b3_0 == 0) ? v4_0[2] : v4_0[0]);
            float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, send2_2, 8);
            v2_8[0] = ((b3_0 == 0) ? v4_0[0] : v4_0[2]) + _shfl_xor_sync_265;
            float send2_3 = ((b3_0 == 0) ? v4_0[3] : v4_0[1]);
            float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, send2_3, 8);
            v2_8[1] = ((b3_0 == 0) ? v4_0[1] : v4_0[3]) + _shfl_xor_sync_266;
            float send1_0 = ((b2_0 == 0) ? v2_8[1] : v2_8[0]);
            float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, send1_0, 4);
            float v1_0 = ((b2_0 == 0) ? v2_8[0] : v2_8[1]) + _shfl_xor_sync_267;
            float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 2);
            v1_0 = v1_0 + _shfl_xor_sync_268;
            float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 1);
            v1_0 = v1_0 + _shfl_xor_sync_269;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            {
                uint32_t _addr_270 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_270), "f"(tot_0) : "memory");
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
            float _tmem_load_17[8];
            tmem_ld_x8(&_tmem_load_17[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_0 = 48 + 8 * th;
            int goff_1_0 = t_0 * rs_g;
            float dg_1_0 = offset + _tmem_load_17[0];
            if (t_0 < clen) {
                {
                    auto _gst_271 = dg_base + goff_1_0;
                    __builtin_assume(__isGlobal(_gst_271));
                    *(reinterpret_cast<float*>(_gst_271) + (0)) = dg_1_0;
                }
            }
            int t_2 = 48 + 8 * th + 1;
            int goff_1_1 = t_2 * rs_g;
            float dg_1_1 = offset + _tmem_load_17[1];
            if (t_2 < clen) {
                {
                    auto _gst_272 = dg_base + goff_1_1;
                    __builtin_assume(__isGlobal(_gst_272));
                    *(reinterpret_cast<float*>(_gst_272) + (0)) = dg_1_1;
                }
            }
            int t_3 = 48 + 8 * th + 2;
            int goff_1_2 = t_3 * rs_g;
            float dg_1_2 = offset + _tmem_load_17[2];
            if (t_3 < clen) {
                {
                    auto _gst_273 = dg_base + goff_1_2;
                    __builtin_assume(__isGlobal(_gst_273));
                    *(reinterpret_cast<float*>(_gst_273) + (0)) = dg_1_2;
                }
            }
            int t_4 = 48 + 8 * th + 3;
            int goff_1_3 = t_4 * rs_g;
            float dg_1_3 = offset + _tmem_load_17[3];
            if (t_4 < clen) {
                {
                    auto _gst_274 = dg_base + goff_1_3;
                    __builtin_assume(__isGlobal(_gst_274));
                    *(reinterpret_cast<float*>(_gst_274) + (0)) = dg_1_3;
                }
            }
            int t_5 = 48 + 8 * th + 4;
            int goff_1_4 = t_5 * rs_g;
            float dg_1_4 = offset + _tmem_load_17[4];
            if (t_5 < clen) {
                {
                    auto _gst_275 = dg_base + goff_1_4;
                    __builtin_assume(__isGlobal(_gst_275));
                    *(reinterpret_cast<float*>(_gst_275) + (0)) = dg_1_4;
                }
            }
            int t_6 = 48 + 8 * th + 5;
            int goff_1_5 = t_6 * rs_g;
            float dg_1_5 = offset + _tmem_load_17[5];
            if (t_6 < clen) {
                {
                    auto _gst_276 = dg_base + goff_1_5;
                    __builtin_assume(__isGlobal(_gst_276));
                    *(reinterpret_cast<float*>(_gst_276) + (0)) = dg_1_5;
                }
            }
            int t_7 = 48 + 8 * th + 6;
            int goff_1_6 = t_7 * rs_g;
            float dg_1_6 = offset + _tmem_load_17[6];
            if (t_7 < clen) {
                {
                    auto _gst_277 = dg_base + goff_1_6;
                    __builtin_assume(__isGlobal(_gst_277));
                    *(reinterpret_cast<float*>(_gst_277) + (0)) = dg_1_6;
                }
            }
            int t_8 = 48 + 8 * th + 7;
            int goff_1_7 = t_8 * rs_g;
            float dg_1_7 = offset + _tmem_load_17[7];
            if (t_8 < clen) {
                {
                    auto _gst_278 = dg_base + goff_1_7;
                    __builtin_assume(__isGlobal(_gst_278));
                    *(reinterpret_cast<float*>(_gst_278) + (0)) = dg_1_7;
                }
            }
            off_acc[0] = off_acc[0] + t_hi + t_lo;
            float _tmem_load_18[4];
            tmem_ld_x4(&_tmem_load_18[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * (1 - th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float t_hi_1248 = ((th == 1) ? tot : _tmem_load_18[0]);
            float t_lo_1249 = ((th == 1) ? _tmem_load_18[0] : tot);
            float offset_1250 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1248);
            float _tmem_load_19[8];
            tmem_ld_x8(&_tmem_load_19[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_9 = 32 + 8 * th;
            int goff_0 = t_9 * rs_g;
            float dg_0 = offset_1250 + _tmem_load_19[0];
            if (t_9 < clen) {
                {
                    auto _gst_279 = dg_base + goff_0;
                    __builtin_assume(__isGlobal(_gst_279));
                    *(reinterpret_cast<float*>(_gst_279) + (0)) = dg_0;
                }
            }
            int t_10 = 32 + 8 * th + 1;
            int goff_1 = t_10 * rs_g;
            float dg_1 = offset_1250 + _tmem_load_19[1];
            if (t_10 < clen) {
                {
                    auto _gst_280 = dg_base + goff_1;
                    __builtin_assume(__isGlobal(_gst_280));
                    *(reinterpret_cast<float*>(_gst_280) + (0)) = dg_1;
                }
            }
            int t_11 = 32 + 8 * th + 2;
            int goff_2 = t_11 * rs_g;
            float dg_2 = offset_1250 + _tmem_load_19[2];
            if (t_11 < clen) {
                {
                    auto _gst_281 = dg_base + goff_2;
                    __builtin_assume(__isGlobal(_gst_281));
                    *(reinterpret_cast<float*>(_gst_281) + (0)) = dg_2;
                }
            }
            int t_12 = 32 + 8 * th + 3;
            int goff_3 = t_12 * rs_g;
            float dg_3 = offset_1250 + _tmem_load_19[3];
            if (t_12 < clen) {
                {
                    auto _gst_282 = dg_base + goff_3;
                    __builtin_assume(__isGlobal(_gst_282));
                    *(reinterpret_cast<float*>(_gst_282) + (0)) = dg_3;
                }
            }
            int t_13 = 32 + 8 * th + 4;
            int goff_4 = t_13 * rs_g;
            float dg_4 = offset_1250 + _tmem_load_19[4];
            if (t_13 < clen) {
                {
                    auto _gst_283 = dg_base + goff_4;
                    __builtin_assume(__isGlobal(_gst_283));
                    *(reinterpret_cast<float*>(_gst_283) + (0)) = dg_4;
                }
            }
            int t_14 = 32 + 8 * th + 5;
            int goff_5 = t_14 * rs_g;
            float dg_5 = offset_1250 + _tmem_load_19[5];
            if (t_14 < clen) {
                {
                    auto _gst_284 = dg_base + goff_5;
                    __builtin_assume(__isGlobal(_gst_284));
                    *(reinterpret_cast<float*>(_gst_284) + (0)) = dg_5;
                }
            }
            int t_15 = 32 + 8 * th + 6;
            int goff_6 = t_15 * rs_g;
            float dg_6 = offset_1250 + _tmem_load_19[6];
            if (t_15 < clen) {
                {
                    auto _gst_285 = dg_base + goff_6;
                    __builtin_assume(__isGlobal(_gst_285));
                    *(reinterpret_cast<float*>(_gst_285) + (0)) = dg_6;
                }
            }
            int t_16 = 32 + 8 * th + 7;
            int goff_7 = t_16 * rs_g;
            float dg_7 = offset_1250 + _tmem_load_19[7];
            if (t_16 < clen) {
                {
                    auto _gst_286 = dg_base + goff_7;
                    __builtin_assume(__isGlobal(_gst_286));
                    *(reinterpret_cast<float*>(_gst_286) + (0)) = dg_7;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1248 + t_lo_1249;
            {
                uint32_t _addr_287 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_287), "f"(off_acc[0]) : "memory");
            }
            mbarrier_wait(p1b_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + (2 + th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_264[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_264[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1 = r_264[0];
            float gc_2 = gn_1 - 21.640425f;
            int eb_1 = 81920 + th * -32768;
            int ew_1 = eb_1 >> 2;
            int qbase_2 = pool_f32_addr + (unsigned int)(eb_1 + 12288 + dh_lane * 1024);
            int sb8_1 = 16 + 8 * th;
            int gb8_1 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1 * 128);
            int kb8_1 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1 * 128);
            int c_qa_1 = 64 + 8 * th;
            int c_ka_1 = 80 + 8 * th;
            int c_qd_1 = 96 + 8 * th;
            int c_kd_1 = 112 + 8 * th;
            int c_dk_1 = 16 + 8 * th;
            int c_bp_1 = c_ka_1;
            float s_run_1[1];
            s_run_1[0] = 0.0f;
            #pragma unroll 1
            for (int k_2 = 0; k_2 < 8; k_2++) {
                int j_2 = 7 - k_2;
                float a_qd_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_1 + j_2)));
                float a_kd_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd_1 + j_2)));
                float a_dk_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk_1 + j_2)));
                float a_qa_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qa_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qa_1 + j_2)));
                float a_ka_2[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_ka_2[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_ka_1 + j_2)));
                float btr_2[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr_2[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_1 + j_2))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t_17 = sb8_1 + j_2;
                float r_265[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_265[0])) : "r"((gb8_1 + (lane_1 / 32 * 8192 + 128 * j_2 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_2 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t_2 = r_265[0];
                unsigned int r_266[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_266[0])) : "r"((kb8_1 + (dcol / 64 * 8192 + 128 * j_2 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_2 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t_2 = __uint_as_float(r_266[0] << 16);
                unsigned int r_267[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_267[0])) : "r"((qbase_2 + (dcol / 64 * 1024 + 128 * j_2 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_2 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t_2 = __uint_as_float(r_267[0] << 16);
                float q_m_2 = ((t_17 >= clen) ? 0.0f : q_t_2);
                float dqf_2 = pool_f32[ew_1 + 128 * j_2 + d_lane];
                float dkf_2 = pool_f32[ew_1 + (1024 + 128 * j_2) + d_lane];
                float dgf_2 = pool_f32[ew_1 + (2048 + 128 * j_2) + d_lane];
                float bt_t_2 = btr_2[0];
                float e_dg_2 = approx_exp2(g_t_2 - gc_2);
                float e_off_2 = approx_exp2(g_t_2 - gn_1);
                float dq2_2 = __fmaf_rn(a_qa_2[0], e_off_2, a_qd_2[0] * e_dg_2);
                float dk2_2 = __fmaf_rn(a_ka_2[0], e_off_2, a_kd_2[0] * e_dg_2);
                float dbv_2[1];
                dbv_2[0] = dk2_2 * k_t_2;
                float dk2b_2 = dk2_2 * bt_t_2;
                float dq_o_2 = dq2_2 + dqf_2;
                float dk_o_2 = dk2b_2 + dkf_2 + a_dk_2[0];
                float qe_2 = q_m_2 * e_dg_2;
                float kp_13 = k_t_2 * bt_t_2;
                __nv_bfloat16 kb_2 = (__nv_bfloat16)kp_13;
                float kbe_2 = (float)kb_2 * e_dg_2;
                float cr_2 = __fmaf_rn(__uint_as_float(__as_u32(qe_2) & 4294959104u) - qe_2, a_qd_2[0], (__uint_as_float(__as_u32(kbe_2) & 4294959104u) - kp_13 * e_dg_2) * a_kd_2[0]);
                float dg_o_2 = __fmaf_rn(q_m_2, dq2_2, __fmaf_rn(dk2b_2 - a_dk_2[0], k_t_2, cr_2)) + dgf_2;
                s_run_1[0] = s_run_1[0] + dg_o_2;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_1 + j_2)), "f"(s_run_1[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp_1 + j_2)), "f"(dbv_2[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int qoff_2 = t_17 * qrs;
                if (t_17 < clen) {
                    {
                        auto _gst_288 = dq_base + qoff_2;
                        __builtin_assume(__isGlobal(_gst_288));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_288) + (0)) = (__nv_bfloat16)dq_o_2;
                    }
                    {
                        auto _gst_289 = dk_base + qoff_2;
                        __builtin_assume(__isGlobal(_gst_289));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_289) + (0)) = (__nv_bfloat16)dk_o_2;
                    }
                }
            }
            float tot_1 = s_run_1[0];
            float dbp_1[8];
            tmem_ld_x8(&dbp_1[0], taddr + (unsigned int)row_base + (unsigned int)c_bp_1);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_1 = lane_1 >> 4 & 1;
            int b3_1 = lane_1 >> 3 & 1;
            int b2_1 = lane_1 >> 2 & 1;
            float v4_1[4];
            float send_9 = ((b4_1 == 0) ? dbp_1[4] : dbp_1[0]);
            float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, send_9, 16);
            v4_1[0] = ((b4_1 == 0) ? dbp_1[0] : dbp_1[4]) + _shfl_xor_sync_290;
            float send_10 = ((b4_1 == 0) ? dbp_1[5] : dbp_1[1]);
            float _shfl_xor_sync_291 = __shfl_xor_sync(0xFFFFFFFF, send_10, 16);
            v4_1[1] = ((b4_1 == 0) ? dbp_1[1] : dbp_1[5]) + _shfl_xor_sync_291;
            float send_11 = ((b4_1 == 0) ? dbp_1[6] : dbp_1[2]);
            float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, send_11, 16);
            v4_1[2] = ((b4_1 == 0) ? dbp_1[2] : dbp_1[6]) + _shfl_xor_sync_292;
            float send_12 = ((b4_1 == 0) ? dbp_1[7] : dbp_1[3]);
            float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, send_12, 16);
            v4_1[3] = ((b4_1 == 0) ? dbp_1[3] : dbp_1[7]) + _shfl_xor_sync_293;
            float v2_9[2];
            float send2_4 = ((b3_1 == 0) ? v4_1[2] : v4_1[0]);
            float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, send2_4, 8);
            v2_9[0] = ((b3_1 == 0) ? v4_1[0] : v4_1[2]) + _shfl_xor_sync_294;
            float send2_5 = ((b3_1 == 0) ? v4_1[3] : v4_1[1]);
            float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, send2_5, 8);
            v2_9[1] = ((b3_1 == 0) ? v4_1[1] : v4_1[3]) + _shfl_xor_sync_295;
            float send1_1 = ((b2_1 == 0) ? v2_9[1] : v2_9[0]);
            float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, send1_1, 4);
            float v1_1 = ((b2_1 == 0) ? v2_9[0] : v2_9[1]) + _shfl_xor_sync_296;
            float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
            v1_1 = v1_1 + _shfl_xor_sync_297;
            float _shfl_xor_sync_298 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
            v1_1 = v1_1 + _shfl_xor_sync_298;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (2 + th) * 8);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_299 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_299), "f"(tot_1) : "memory");
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
            float t_hi_1486 = ((th == 1) ? tot_1 : t_other_0[0]);
            float t_lo_1487 = ((th == 1) ? t_other_0[0] : tot_1);
            float offset_1488 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1486);
            float _tmem_load_25[8];
            tmem_ld_x8(&_tmem_load_25[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_17_1 = 16 + 8 * th;
            int goff_8 = t_17_1 * rs_g;
            float dg_8 = offset_1488 + _tmem_load_25[0];
            if (t_17_1 < clen) {
                {
                    auto _gst_300 = dg_base + goff_8;
                    __builtin_assume(__isGlobal(_gst_300));
                    *(reinterpret_cast<float*>(_gst_300) + (0)) = dg_8;
                }
            }
            int t_18 = 16 + 8 * th + 1;
            int goff_9 = t_18 * rs_g;
            float dg_9 = offset_1488 + _tmem_load_25[1];
            if (t_18 < clen) {
                {
                    auto _gst_301 = dg_base + goff_9;
                    __builtin_assume(__isGlobal(_gst_301));
                    *(reinterpret_cast<float*>(_gst_301) + (0)) = dg_9;
                }
            }
            int t_19 = 16 + 8 * th + 2;
            int goff_10 = t_19 * rs_g;
            float dg_10 = offset_1488 + _tmem_load_25[2];
            if (t_19 < clen) {
                {
                    auto _gst_302 = dg_base + goff_10;
                    __builtin_assume(__isGlobal(_gst_302));
                    *(reinterpret_cast<float*>(_gst_302) + (0)) = dg_10;
                }
            }
            int t_20 = 16 + 8 * th + 3;
            int goff_11 = t_20 * rs_g;
            float dg_11 = offset_1488 + _tmem_load_25[3];
            if (t_20 < clen) {
                {
                    auto _gst_303 = dg_base + goff_11;
                    __builtin_assume(__isGlobal(_gst_303));
                    *(reinterpret_cast<float*>(_gst_303) + (0)) = dg_11;
                }
            }
            int t_21 = 16 + 8 * th + 4;
            int goff_12 = t_21 * rs_g;
            float dg_12 = offset_1488 + _tmem_load_25[4];
            if (t_21 < clen) {
                {
                    auto _gst_304 = dg_base + goff_12;
                    __builtin_assume(__isGlobal(_gst_304));
                    *(reinterpret_cast<float*>(_gst_304) + (0)) = dg_12;
                }
            }
            int t_22 = 16 + 8 * th + 5;
            int goff_13 = t_22 * rs_g;
            float dg_13 = offset_1488 + _tmem_load_25[5];
            if (t_22 < clen) {
                {
                    auto _gst_305 = dg_base + goff_13;
                    __builtin_assume(__isGlobal(_gst_305));
                    *(reinterpret_cast<float*>(_gst_305) + (0)) = dg_13;
                }
            }
            int t_23 = 16 + 8 * th + 6;
            int goff_14 = t_23 * rs_g;
            float dg_14 = offset_1488 + _tmem_load_25[6];
            if (t_23 < clen) {
                {
                    auto _gst_306 = dg_base + goff_14;
                    __builtin_assume(__isGlobal(_gst_306));
                    *(reinterpret_cast<float*>(_gst_306) + (0)) = dg_14;
                }
            }
            int t_24 = 16 + 8 * th + 7;
            int goff_15 = t_24 * rs_g;
            float dg_15 = offset_1488 + _tmem_load_25[7];
            if (t_24 < clen) {
                {
                    auto _gst_307 = dg_base + goff_15;
                    __builtin_assume(__isGlobal(_gst_307));
                    *(reinterpret_cast<float*>(_gst_307) + (0)) = dg_15;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1486 + t_lo_1487;
            {
                uint32_t _addr_308 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_308), "f"(off_acc[0]) : "memory");
            }
            mbarrier_wait(epi_full_addr + (th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_268[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_268[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_2 = r_268[0];
            float gc_3 = gn_2 - 21.640425f;
            int eb_2 = 98304 + th * -32768;
            int ew_2 = eb_2 >> 2;
            int qbase_3 = pool_f32_addr + (unsigned int)(eb_2 + 12288 + dh_lane * 1024);
            int sb8_2 = 8 * th;
            int gb8_2 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_2 * 128);
            int kb8_2 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_2 * 128);
            int c_qa_2 = 8 * th;
            int c_ka_2 = 8 * th;
            int c_qd_2 = 128 + 8 * th;
            int c_kd_2 = 144 + 8 * th;
            int c_dk_2 = 8 * th;
            int c_bp_2 = c_kd_2;
            float s_run_2[1];
            s_run_2[0] = 0.0f;
            #pragma unroll 1
            for (int k_3 = 0; k_3 < 8; k_3++) {
                int j_3 = 7 - k_3;
                float a_qd_3[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_qd_3[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_2 + j_3)));
                float a_kd_3[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_kd_3[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_kd_2 + j_3)));
                float a_dk_3[1];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    " {%0}, [%1];"
                    : "=f"(a_dk_3[0])
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(c_dk_2 + j_3)));
                float btr_3[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&btr_3[0])) : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (sb8_2 + j_3))));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int t_25 = sb8_2 + j_3;
                float r_269[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_269[0])) : "r"((gb8_2 + (lane_1 / 32 * 8192 + 128 * j_3 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 * j_3 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
                float g_t_3 = r_269[0];
                unsigned int r_270[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_270[0])) : "r"((kb8_2 + (dcol / 64 * 8192 + 128 * j_3 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 * j_3 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float k_t_3 = __uint_as_float(r_270[0] << 16);
                unsigned int r_271[1];
                asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_271[0])) : "r"((qbase_3 + (dcol / 64 * 1024 + 128 * j_3 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 * j_3 + dcol % 64 * 2 >> 7 & 7) << 4))));
                float q_t_3 = __uint_as_float(r_271[0] << 16);
                float q_m_3 = ((t_25 >= clen) ? 0.0f : q_t_3);
                float dqf_3 = pool_f32[ew_2 + 128 * j_3 + d_lane];
                float dkf_3 = pool_f32[ew_2 + (1024 + 128 * j_3) + d_lane];
                float dgf_3 = pool_f32[ew_2 + (2048 + 128 * j_3) + d_lane];
                float bt_t_3 = btr_3[0];
                float e_dg_3 = approx_exp2(g_t_3 - gc_3);
                float dq2_3 = a_qd_3[0] * e_dg_3;
                float dk2_3 = a_kd_3[0] * e_dg_3;
                float dbv_3[1];
                dbv_3[0] = dk2_3 * k_t_3;
                float dk2b_3 = dk2_3 * bt_t_3;
                float dq_o_3 = dq2_3 + dqf_3;
                float dk_o_3 = dk2b_3 + dkf_3 + a_dk_3[0];
                float qe_3 = q_m_3 * e_dg_3;
                float kp_14 = k_t_3 * bt_t_3;
                __nv_bfloat16 kb_3 = (__nv_bfloat16)kp_14;
                float kbe_3 = (float)kb_3 * e_dg_3;
                float cr_3 = __fmaf_rn(__uint_as_float(__as_u32(qe_3) & 4294959104u) - qe_3, a_qd_3[0], (__uint_as_float(__as_u32(kbe_3) & 4294959104u) - kp_14 * e_dg_3) * a_kd_3[0]);
                float dg_o_3 = __fmaf_rn(q_m_3, dq2_3, __fmaf_rn(dk2b_3 - a_dk_3[0], k_t_3, cr_3)) + dgf_3;
                s_run_2[0] = s_run_2[0] + dg_o_3;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_qd_2 + j_3)), "f"(s_run_2[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)row_base + (unsigned int)(c_bp_2 + j_3)), "f"(dbv_3[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int qoff_3 = t_25 * qrs;
                if (t_25 < clen) {
                    {
                        auto _gst_309 = dq_base + qoff_3;
                        __builtin_assume(__isGlobal(_gst_309));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_309) + (0)) = (__nv_bfloat16)dq_o_3;
                    }
                    {
                        auto _gst_310 = dk_base + qoff_3;
                        __builtin_assume(__isGlobal(_gst_310));
                        *(reinterpret_cast<__nv_bfloat16*>(_gst_310) + (0)) = (__nv_bfloat16)dk_o_3;
                    }
                }
            }
            float tot_2 = s_run_2[0];
            float dbp_2[8];
            tmem_ld_x8(&dbp_2[0], taddr + (unsigned int)row_base + (unsigned int)c_bp_2);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int b4_2 = lane_1 >> 4 & 1;
            int b3_2 = lane_1 >> 3 & 1;
            int b2_2 = lane_1 >> 2 & 1;
            float v4_2[4];
            float send_13 = ((b4_2 == 0) ? dbp_2[4] : dbp_2[0]);
            float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, send_13, 16);
            v4_2[0] = ((b4_2 == 0) ? dbp_2[0] : dbp_2[4]) + _shfl_xor_sync_311;
            float send_14 = ((b4_2 == 0) ? dbp_2[5] : dbp_2[1]);
            float _shfl_xor_sync_312 = __shfl_xor_sync(0xFFFFFFFF, send_14, 16);
            v4_2[1] = ((b4_2 == 0) ? dbp_2[1] : dbp_2[5]) + _shfl_xor_sync_312;
            float send_15 = ((b4_2 == 0) ? dbp_2[6] : dbp_2[2]);
            float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, send_15, 16);
            v4_2[2] = ((b4_2 == 0) ? dbp_2[2] : dbp_2[6]) + _shfl_xor_sync_313;
            float send_16 = ((b4_2 == 0) ? dbp_2[7] : dbp_2[3]);
            float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, send_16, 16);
            v4_2[3] = ((b4_2 == 0) ? dbp_2[3] : dbp_2[7]) + _shfl_xor_sync_314;
            float v2_10[2];
            float send2_6 = ((b3_2 == 0) ? v4_2[2] : v4_2[0]);
            float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, send2_6, 8);
            v2_10[0] = ((b3_2 == 0) ? v4_2[0] : v4_2[2]) + _shfl_xor_sync_315;
            float send2_7 = ((b3_2 == 0) ? v4_2[3] : v4_2[1]);
            float _shfl_xor_sync_316 = __shfl_xor_sync(0xFFFFFFFF, send2_7, 8);
            v2_10[1] = ((b3_2 == 0) ? v4_2[1] : v4_2[3]) + _shfl_xor_sync_316;
            float send1_2 = ((b2_2 == 0) ? v2_10[1] : v2_10[0]);
            float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, send1_2, 4);
            float v1_2 = ((b2_2 == 0) ? v2_10[0] : v2_10[1]) + _shfl_xor_sync_317;
            float _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 2);
            v1_2 = v1_2 + _shfl_xor_sync_318;
            float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 1);
            v1_2 = v1_2 + _shfl_xor_sync_319;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_320 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_320), "f"(tot_2) : "memory");
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
            float t_hi_1708 = ((th == 1) ? tot_2 : t_other_1[0]);
            float t_lo_1709 = ((th == 1) ? t_other_1[0] : tot_2);
            float offset_1710 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1708);
            float _tmem_load_29[8];
            tmem_ld_x8(&_tmem_load_29[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_25_1 = 8 * th;
            int goff_16 = t_25_1 * rs_g;
            float dg_16 = offset_1710 + _tmem_load_29[0];
            if (t_25_1 < clen) {
                {
                    auto _gst_321 = dg_base + goff_16;
                    __builtin_assume(__isGlobal(_gst_321));
                    *(reinterpret_cast<float*>(_gst_321) + (0)) = dg_16;
                }
            }
            int t_26 = 8 * th + 1;
            int goff_17 = t_26 * rs_g;
            float dg_17 = offset_1710 + _tmem_load_29[1];
            if (t_26 < clen) {
                {
                    auto _gst_322 = dg_base + goff_17;
                    __builtin_assume(__isGlobal(_gst_322));
                    *(reinterpret_cast<float*>(_gst_322) + (0)) = dg_17;
                }
            }
            int t_27 = 8 * th + 2;
            int goff_18 = t_27 * rs_g;
            float dg_18 = offset_1710 + _tmem_load_29[2];
            if (t_27 < clen) {
                {
                    auto _gst_323 = dg_base + goff_18;
                    __builtin_assume(__isGlobal(_gst_323));
                    *(reinterpret_cast<float*>(_gst_323) + (0)) = dg_18;
                }
            }
            int t_28 = 8 * th + 3;
            int goff_19 = t_28 * rs_g;
            float dg_19 = offset_1710 + _tmem_load_29[3];
            if (t_28 < clen) {
                {
                    auto _gst_324 = dg_base + goff_19;
                    __builtin_assume(__isGlobal(_gst_324));
                    *(reinterpret_cast<float*>(_gst_324) + (0)) = dg_19;
                }
            }
            int t_29 = 8 * th + 4;
            int goff_20 = t_29 * rs_g;
            float dg_20 = offset_1710 + _tmem_load_29[4];
            if (t_29 < clen) {
                {
                    auto _gst_325 = dg_base + goff_20;
                    __builtin_assume(__isGlobal(_gst_325));
                    *(reinterpret_cast<float*>(_gst_325) + (0)) = dg_20;
                }
            }
            int t_30 = 8 * th + 5;
            int goff_21 = t_30 * rs_g;
            float dg_21 = offset_1710 + _tmem_load_29[5];
            if (t_30 < clen) {
                {
                    auto _gst_326 = dg_base + goff_21;
                    __builtin_assume(__isGlobal(_gst_326));
                    *(reinterpret_cast<float*>(_gst_326) + (0)) = dg_21;
                }
            }
            int t_31 = 8 * th + 6;
            int goff_22 = t_31 * rs_g;
            float dg_22 = offset_1710 + _tmem_load_29[6];
            if (t_31 < clen) {
                {
                    auto _gst_327 = dg_base + goff_22;
                    __builtin_assume(__isGlobal(_gst_327));
                    *(reinterpret_cast<float*>(_gst_327) + (0)) = dg_22;
                }
            }
            int t_32 = 8 * th + 7;
            int goff_23 = t_32 * rs_g;
            float dg_23 = offset_1710 + _tmem_load_29[7];
            if (t_32 < clen) {
                {
                    auto _gst_328 = dg_base + goff_23;
                    __builtin_assume(__isGlobal(_gst_328));
                    *(reinterpret_cast<float*>(_gst_328) + (0)) = dg_23;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1708 + t_lo_1709;
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
                    for (int r_1 = 0; r_1 < 16; r_1++) {
                        int t_33 = r_1;
                        if (t_33 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_33 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_33 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr);
                }
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _smem_desc_lo_0 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_1 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
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
                    for (int r_2 = 0; r_2 < 16; r_2++) {
                        int t_34 = 16 + r_2;
                        if (t_34 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_34 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_34 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 8);
                }
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _smem_desc_lo_2 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_3 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
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
                int _smem_desc_lo_4 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_5 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
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
                    for (int r_3 = 0; r_3 < 16; r_3++) {
                        int t_35 = 32 + r_3;
                        if (t_35 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_35 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_35 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 16);
                }
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _smem_desc_lo_6 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_7 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
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
                int _smem_desc_lo_8 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_9 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
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
                int _smem_desc_lo_10 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_11 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
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
                    for (int r_40 = 0; r_40 < 16; r_40++) {
                        int t_36 = 48 + r_40;
                        if (t_36 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_36 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_36 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_40 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 24);
                }
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _smem_desc_lo_12 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_13 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
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
                int _smem_desc_lo_14 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_15 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
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
                int _smem_desc_lo_16 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_17 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
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
                int _smem_desc_lo_18 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_19 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
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
                int _smem_desc_lo_20 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_21 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_22 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_23 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_24 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_25 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_26 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_27 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_28 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_29 = (((dab_addr + 64) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_30 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_31 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_32 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_33 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_34 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_35 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_36 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_37 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_38 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_39 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_40 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_41 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_42 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_43 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_44 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_45 = (((dab_addr + 4160) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_46 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_47 = (((dab_addr + 6208) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_48 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_49 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_50 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_51 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_52 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_53 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
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
                int _smem_desc_lo_54 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _smem_desc_lo_55 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
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
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 65536, (&dq_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 69632, (&dk_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 73728, (&dg_f_t), 0, hv_h, row0_h + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
                int _smem_desc_lo_56 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_57 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
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
                int _smem_desc_lo_58 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _smem_desc_lo_59 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
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
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                tma_3d_gmem2smem(pool_f32_addr + 98304, (&dq_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 102400, (&dk_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 106496, (&dg_f_t), 0, hv_h, row0_h, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup, epi_full_addr);
                mbarrier_wait(epi_empty_addr + 8, 1);
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
