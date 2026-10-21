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
#define TMEM_NCOLS 256
#define TMEM_ACC_OFFSET 0
#define NUM_ONE_STAGES 1
#define NUM_TWO_STAGES 2
#define NUM_FOUR_STAGES 4
#define SMEM_GK_P_OFF 0
#define SMEM_GK_P_STAGE_BYTES 8192
#define SMEM_GK_P_STRIDE 8192
#define SMEM_K_P_OFF 32768
#define SMEM_K_P_STAGE_BYTES 8192
#define SMEM_K_P_STRIDE 8192
#define SMEM_RING_OFF 49152
#define SMEM_RING_STAGE_BYTES 16384
#define SMEM_RING_STRIDE 16384
#define SMEM_DAB_OFF 81920
#define SMEM_DAB_STAGE_BYTES 8192
#define SMEM_DAB_STRIDE 8192
#define SMEM_Q_P_OFF 98304
#define SMEM_Q_P_STAGE_BYTES 2048
#define SMEM_Q_P_STRIDE 2048
#define SMEM_TILES_OFF 106496
#define SMEM_TILES_STAGE_BYTES 2048
#define SMEM_TILES_STRIDE 2048
#define SMEM_BETA_S_OFF 114688
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_POOL_F32_OFF 0
#define SMEM_POOL_F32_STAGE_BYTES 114944
#define SMEM_POOL_F32_STRIDE 114944
#define SMEM_TOTAL 115200
#define THREADS 288
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

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_cc51220bb4a9a37152ad(const __grid_constant__ CUtensorMap dAqk, const __grid_constant__ CUtensorMap dAkk, const __grid_constant__ CUtensorMap gk, const __grid_constant__ CUtensorMap k_e, const __grid_constant__ CUtensorMap q_e, const __grid_constant__ CUtensorMap dq_f_t, const __grid_constant__ CUtensorMap dk_f_t, const __grid_constant__ CUtensorMap dg_f_t, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_out, float* __restrict__ db_out, int num_heads, int num_qk_heads, int group)
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

    // Kernel setup ops
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
    int clen_setup = chunk_len[blockIdx.x];
    int kq0_setup = chunk_bos[blockIdx.x];
    if (__syncthreads_and(clen_setup == 0)) return;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 31 barriers)
    // Mbarriers at smem_raw[114944..115192)

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

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

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
            long long gbase_c = ((long long)row0 * (long long)num_heads + (long long)hv) * 128;
            long long ext0 = (long long)chunk_bos[chunk];
            int clen = chunk_len[chunk];
            float* dg_base = dg_out + (gbase_c + (long long)d_lane);
            long long qoff0 = ext0 * (long long)qrs + (long long)(qk_head * 128 + d_lane);
            __nv_bfloat16* dq_base = dq_ext + qoff0;
            __nv_bfloat16* dk_base = dk_ext + qoff0;
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
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv[4];
                int kk = w * 4;
                int a = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[0])) : "r"(a));
                tv[0] = ((lane_1 > (kk & 15)) ? 0.0f : tv[0]);
                {
                    uint32_t _addr_0 = static_cast<uint32_t>(a);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(tv[0]) : "memory");
                }
                int kk_0 = w * 4 + 1;
                int a_1 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[1])) : "r"(a_1));
                tv[1] = ((lane_1 > (kk_0 & 15)) ? 0.0f : tv[1]);
                {
                    uint32_t _addr_1 = static_cast<uint32_t>(a_1);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(tv[1]) : "memory");
                }
                int kk_2 = w * 4 + 2;
                int a_3 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[2])) : "r"(a_3));
                tv[2] = ((lane_1 > (kk_2 & 15)) ? 0.0f : tv[2]);
                {
                    uint32_t _addr_2 = static_cast<uint32_t>(a_3);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(tv[2]) : "memory");
                }
                int kk_4 = w * 4 + 3;
                int a_5 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[3])) : "r"(a_5));
                tv[3] = ((lane_1 > (kk_4 & 15)) ? 0.0f : tv[3]);
                {
                    uint32_t _addr_3 = static_cast<uint32_t>(a_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(tv[3]) : "memory");
                }
                int ta = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta), "f"(tv[0]), "f"(tv[1]), "f"(tv[2]), "f"(tv[3]) : "memory");
            }
            float r[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d = r[0] - 21.640425324000002f;
            int sbase = 8 * th;
            int gbase = pool_f32_addr + (unsigned int)(qd * 8192 + sbase * 128);
            int kbase = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase * 128);
            int qbase = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float2 _f2_0 = make_float2(a_d, a_d);
            float bv[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4)));
            float vq[4];
            float vk[4];
            float r_2[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2[0])) : "r"((gbase + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0 = r_2[0];
            float r_3[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_3[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1 = r_3[0];
            unsigned int r_4[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_4[0])) : "r"((qbase + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0 = __uint_as_float(r_4[0] << 16);
            unsigned int r_5[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_5[0])) : "r"((qbase + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1 = __uint_as_float(r_5[0] << 16);
            unsigned int r_6[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_6[0])) : "r"((kbase + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0 = __uint_as_float(r_6[0] << 16);
            unsigned int r_7[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_7[0])) : "r"((kbase + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1 = __uint_as_float(r_7[0] << 16);
            float2 _f2_1 = make_float2(g_s0, g_s1);
            float2 d2 = sub_f32x2(_f2_1, _f2_0);
            float _exp2_0 = approx_exp2(d2.x);
            float ev0 = _exp2_0;
            float _exp2_1 = approx_exp2(d2.y);
            float ev1 = _exp2_1;
            float2 _f2_2 = make_float2(k_s0, k_s1);
            float2 _f2_3 = make_float2(bv[0], bv[1]);
            float2 kp;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_3));
            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)kp.x;
            __nv_bfloat16 kb0 = _cvt_bf16_0;
            __nv_bfloat16 _cvt_bf16_1 = (__nv_bfloat16)kp.y;
            __nv_bfloat16 kb1 = _cvt_bf16_1;
            float2 _f2_4 = make_float2(ev0, ev1);
            float2 _f2_5 = make_float2(q_s0, q_s1);
            float2 vq2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&_f2_4));
            float _cvt_f32_0 = (float)kb0;
            float _cvt_f32_1 = (float)kb1;
            float2 _f2_6 = make_float2(_cvt_f32_0, _cvt_f32_1);
            float2 vk2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_4));
            vq[0] = vq2.x;
            vq[1] = vq2.y;
            vk[0] = vk2.x;
            vk[1] = vk2.y;
            float r_8[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_8[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_9 = r_8[0];
            float r_10[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_10[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_11 = r_10[0];
            unsigned int r_12[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_12[0])) : "r"((qbase + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_13 = __uint_as_float(r_12[0] << 16);
            unsigned int r_14[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_14[0])) : "r"((qbase + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_15 = __uint_as_float(r_14[0] << 16);
            unsigned int r_16[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_16[0])) : "r"((kbase + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_17 = __uint_as_float(r_16[0] << 16);
            unsigned int r_18[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_18[0])) : "r"((kbase + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_19 = __uint_as_float(r_18[0] << 16);
            float2 _f2_7 = make_float2(g_s0_9, g_s1_11);
            float2 d2_20 = sub_f32x2(_f2_7, _f2_0);
            float _exp2_2 = approx_exp2(d2_20.x);
            float ev0_21 = _exp2_2;
            float _exp2_3 = approx_exp2(d2_20.y);
            float ev1_22 = _exp2_3;
            float2 _f2_8 = make_float2(k_s0_17, k_s1_19);
            float2 _f2_9 = make_float2(bv[2], bv[3]);
            float2 kp_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_23) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_9));
            __nv_bfloat16 _cvt_bf16_2 = (__nv_bfloat16)kp_23.x;
            __nv_bfloat16 kb0_24 = _cvt_bf16_2;
            __nv_bfloat16 _cvt_bf16_3 = (__nv_bfloat16)kp_23.y;
            __nv_bfloat16 kb1_25 = _cvt_bf16_3;
            float2 _f2_10 = make_float2(ev0_21, ev1_22);
            float2 _f2_11 = make_float2(q_s0_13, q_s1_15);
            float2 vq2_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_26) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
            float _cvt_f32_2 = (float)kb0_24;
            float _cvt_f32_3 = (float)kb1_25;
            float2 _f2_12 = make_float2(_cvt_f32_2, _cvt_f32_3);
            float2 vk2_27;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_27) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
            vq[2] = vq2_26.x;
            vq[3] = vq2_26.y;
            vk[2] = vk2_27.x;
            vk[3] = vk2_27.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq[0]), "f"(vq[1]), "f"(vq[2]), "f"(vq[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk[0]), "f"(vk[1]), "f"(vk[2]), "f"(vk[3]) : "memory");
            float bv_28[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_28[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4 + 16)));
            float vq_29[4];
            float vk_30[4];
            float r_31[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_31[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_32 = r_31[0];
            float r_33[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_33[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_34 = r_33[0];
            unsigned int r_35[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_35[0])) : "r"((qbase + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_36 = __uint_as_float(r_35[0] << 16);
            unsigned int r_37[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_37[0])) : "r"((qbase + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_38 = __uint_as_float(r_37[0] << 16);
            unsigned int r_39[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_39[0])) : "r"((kbase + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_40 = __uint_as_float(r_39[0] << 16);
            unsigned int r_41[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_41[0])) : "r"((kbase + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_42 = __uint_as_float(r_41[0] << 16);
            float2 _f2_13 = make_float2(g_s0_32, g_s1_34);
            float2 d2_43 = sub_f32x2(_f2_13, _f2_0);
            float _exp2_4 = approx_exp2(d2_43.x);
            float ev0_44 = _exp2_4;
            float _exp2_5 = approx_exp2(d2_43.y);
            float ev1_45 = _exp2_5;
            float2 _f2_14 = make_float2(k_s0_40, k_s1_42);
            float2 _f2_15 = make_float2(bv_28[0], bv_28[1]);
            float2 kp_46;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_46) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_15));
            __nv_bfloat16 _cvt_bf16_4 = (__nv_bfloat16)kp_46.x;
            __nv_bfloat16 kb0_47 = _cvt_bf16_4;
            __nv_bfloat16 _cvt_bf16_5 = (__nv_bfloat16)kp_46.y;
            __nv_bfloat16 kb1_48 = _cvt_bf16_5;
            float2 _f2_16 = make_float2(ev0_44, ev1_45);
            float2 _f2_17 = make_float2(q_s0_36, q_s1_38);
            float2 vq2_49;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_49) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&_f2_16));
            float _cvt_f32_4 = (float)kb0_47;
            float _cvt_f32_5 = (float)kb1_48;
            float2 _f2_18 = make_float2(_cvt_f32_4, _cvt_f32_5);
            float2 vk2_50;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_50) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_16));
            vq_29[0] = vq2_49.x;
            vq_29[1] = vq2_49.y;
            vk_30[0] = vk2_50.x;
            vk_30[1] = vk2_50.y;
            float r_51[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_51[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_52 = r_51[0];
            float r_53[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_53[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_54 = r_53[0];
            unsigned int r_55[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_55[0])) : "r"((qbase + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_56 = __uint_as_float(r_55[0] << 16);
            unsigned int r_57[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_57[0])) : "r"((qbase + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_58 = __uint_as_float(r_57[0] << 16);
            unsigned int r_59[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_59[0])) : "r"((kbase + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_60 = __uint_as_float(r_59[0] << 16);
            unsigned int r_61[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_61[0])) : "r"((kbase + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_62 = __uint_as_float(r_61[0] << 16);
            float2 _f2_19 = make_float2(g_s0_52, g_s1_54);
            float2 d2_63 = sub_f32x2(_f2_19, _f2_0);
            float _exp2_6 = approx_exp2(d2_63.x);
            float ev0_64 = _exp2_6;
            float _exp2_7 = approx_exp2(d2_63.y);
            float ev1_65 = _exp2_7;
            float2 _f2_20 = make_float2(k_s0_60, k_s1_62);
            float2 _f2_21 = make_float2(bv_28[2], bv_28[3]);
            float2 kp_66;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_66) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&_f2_21));
            __nv_bfloat16 _cvt_bf16_6 = (__nv_bfloat16)kp_66.x;
            __nv_bfloat16 kb0_67 = _cvt_bf16_6;
            __nv_bfloat16 _cvt_bf16_7 = (__nv_bfloat16)kp_66.y;
            __nv_bfloat16 kb1_68 = _cvt_bf16_7;
            float2 _f2_22 = make_float2(ev0_64, ev1_65);
            float2 _f2_23 = make_float2(q_s0_56, q_s1_58);
            float2 vq2_69;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_69) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&_f2_22));
            float _cvt_f32_6 = (float)kb0_67;
            float _cvt_f32_7 = (float)kb1_68;
            float2 _f2_24 = make_float2(_cvt_f32_6, _cvt_f32_7);
            float2 vk2_70;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_70) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_22));
            vq_29[2] = vq2_69.x;
            vq_29[3] = vq2_69.y;
            vk_30[2] = vk2_70.x;
            vk_30[3] = vk2_70.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_29[0]), "f"(vq_29[1]), "f"(vq_29[2]), "f"(vq_29[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_30[0]), "f"(vk_30[1]), "f"(vk_30[2]), "f"(vk_30[3]) : "memory");
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
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_1[4];
                int kk_1 = w * 4;
                int a_2 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[0])) : "r"(a_2));
                int kk_0_1 = w * 4 + 1;
                int a_1_1 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_0_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[1])) : "r"(a_1_1));
                int kk_2_1 = w * 4 + 2;
                int a_3_1 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_2_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[2])) : "r"(a_3_1));
                int kk_4_1 = w * 4 + 3;
                int a_5_1 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_4_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_1[3])) : "r"(a_5_1));
                int ta_1 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_1), "f"(tv_1[0]), "f"(tv_1[1]), "f"(tv_1[2]), "f"(tv_1[3]) : "memory");
            }
            float r_71[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_71[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_72 = r_71[0];
            int sbase_73 = 16 + 8 * th;
            int gbase_74 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_73 * 128);
            int kbase_75 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_73 * 128);
            int qbase_76 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_25 = make_float2(a_d_72, a_d_72);
            float bv_77[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_77[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_73 * 4)));
            float vq_78[4];
            float vk_79[4];
            float r_80[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_80[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_81 = r_80[0];
            float r_82[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_82[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_83 = r_82[0];
            unsigned int r_84[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_84[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_85 = __uint_as_float(r_84[0] << 16);
            unsigned int r_86[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_86[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_87 = __uint_as_float(r_86[0] << 16);
            unsigned int r_88[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_88[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_89 = __uint_as_float(r_88[0] << 16);
            unsigned int r_90[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_90[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_91 = __uint_as_float(r_90[0] << 16);
            float2 _f2_26 = make_float2(g_s0_81, g_s1_83);
            float2 d2_92 = sub_f32x2(_f2_26, _f2_25);
            float _exp2_8 = approx_exp2(d2_92.x);
            float ev0_93 = _exp2_8;
            float _exp2_9 = approx_exp2(d2_92.y);
            float ev1_94 = _exp2_9;
            float2 _f2_27 = make_float2(k_s0_89, k_s1_91);
            float2 _f2_28 = make_float2(bv_77[0], bv_77[1]);
            float2 kp_95;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_95) : "l"(*(const unsigned long long*)&_f2_27), "l"(*(const unsigned long long*)&_f2_28));
            __nv_bfloat16 _cvt_bf16_8 = (__nv_bfloat16)kp_95.x;
            __nv_bfloat16 kb0_96 = _cvt_bf16_8;
            __nv_bfloat16 _cvt_bf16_9 = (__nv_bfloat16)kp_95.y;
            __nv_bfloat16 kb1_97 = _cvt_bf16_9;
            float2 _f2_29 = make_float2(ev0_93, ev1_94);
            float2 _f2_30 = make_float2(q_s0_85, q_s1_87);
            float2 vq2_98;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_98) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&_f2_29));
            float _cvt_f32_8 = (float)kb0_96;
            float _cvt_f32_9 = (float)kb1_97;
            float2 _f2_31 = make_float2(_cvt_f32_8, _cvt_f32_9);
            float2 vk2_99;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_99) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&_f2_29));
            vq_78[0] = vq2_98.x;
            vq_78[1] = vq2_98.y;
            vk_79[0] = vk2_99.x;
            vk_79[1] = vk2_99.y;
            float r_100[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_100[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_101 = r_100[0];
            float r_102[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_102[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_103 = r_102[0];
            unsigned int r_104[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_104[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_105 = __uint_as_float(r_104[0] << 16);
            unsigned int r_106[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_106[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_107 = __uint_as_float(r_106[0] << 16);
            unsigned int r_108[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_108[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_109 = __uint_as_float(r_108[0] << 16);
            unsigned int r_110[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_110[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_111 = __uint_as_float(r_110[0] << 16);
            float2 _f2_32 = make_float2(g_s0_101, g_s1_103);
            float2 d2_112 = sub_f32x2(_f2_32, _f2_25);
            float _exp2_10 = approx_exp2(d2_112.x);
            float ev0_113 = _exp2_10;
            float _exp2_11 = approx_exp2(d2_112.y);
            float ev1_114 = _exp2_11;
            float2 _f2_33 = make_float2(k_s0_109, k_s1_111);
            float2 _f2_34 = make_float2(bv_77[2], bv_77[3]);
            float2 kp_115;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_115) : "l"(*(const unsigned long long*)&_f2_33), "l"(*(const unsigned long long*)&_f2_34));
            __nv_bfloat16 _cvt_bf16_10 = (__nv_bfloat16)kp_115.x;
            __nv_bfloat16 kb0_116 = _cvt_bf16_10;
            __nv_bfloat16 _cvt_bf16_11 = (__nv_bfloat16)kp_115.y;
            __nv_bfloat16 kb1_117 = _cvt_bf16_11;
            float2 _f2_35 = make_float2(ev0_113, ev1_114);
            float2 _f2_36 = make_float2(q_s0_105, q_s1_107);
            float2 vq2_118;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_118) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_35));
            float _cvt_f32_10 = (float)kb0_116;
            float _cvt_f32_11 = (float)kb1_117;
            float2 _f2_37 = make_float2(_cvt_f32_10, _cvt_f32_11);
            float2 vk2_119;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_119) : "l"(*(const unsigned long long*)&_f2_37), "l"(*(const unsigned long long*)&_f2_35));
            vq_78[2] = vq2_118.x;
            vq_78[3] = vq2_118.y;
            vk_79[2] = vk2_119.x;
            vk_79[3] = vk2_119.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_78[0]), "f"(vq_78[1]), "f"(vq_78[2]), "f"(vq_78[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_79[0]), "f"(vk_79[1]), "f"(vk_79[2]), "f"(vk_79[3]) : "memory");
            float bv_120[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_120[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_73 * 4 + 16)));
            float vq_121[4];
            float vk_122[4];
            float r_123[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_123[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_124 = r_123[0];
            float r_125[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_125[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_126 = r_125[0];
            unsigned int r_127[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_127[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_128 = __uint_as_float(r_127[0] << 16);
            unsigned int r_129[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_129[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_130 = __uint_as_float(r_129[0] << 16);
            unsigned int r_131[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_131[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_132 = __uint_as_float(r_131[0] << 16);
            unsigned int r_133[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_133[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_134 = __uint_as_float(r_133[0] << 16);
            float2 _f2_38 = make_float2(g_s0_124, g_s1_126);
            float2 d2_135 = sub_f32x2(_f2_38, _f2_25);
            float _exp2_12 = approx_exp2(d2_135.x);
            float ev0_136 = _exp2_12;
            float _exp2_13 = approx_exp2(d2_135.y);
            float ev1_137 = _exp2_13;
            float2 _f2_39 = make_float2(k_s0_132, k_s1_134);
            float2 _f2_40 = make_float2(bv_120[0], bv_120[1]);
            float2 kp_138;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_138) : "l"(*(const unsigned long long*)&_f2_39), "l"(*(const unsigned long long*)&_f2_40));
            __nv_bfloat16 _cvt_bf16_12 = (__nv_bfloat16)kp_138.x;
            __nv_bfloat16 kb0_139 = _cvt_bf16_12;
            __nv_bfloat16 _cvt_bf16_13 = (__nv_bfloat16)kp_138.y;
            __nv_bfloat16 kb1_140 = _cvt_bf16_13;
            float2 _f2_41 = make_float2(ev0_136, ev1_137);
            float2 _f2_42 = make_float2(q_s0_128, q_s1_130);
            float2 vq2_141;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_141) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_41));
            float _cvt_f32_12 = (float)kb0_139;
            float _cvt_f32_13 = (float)kb1_140;
            float2 _f2_43 = make_float2(_cvt_f32_12, _cvt_f32_13);
            float2 vk2_142;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_142) : "l"(*(const unsigned long long*)&_f2_43), "l"(*(const unsigned long long*)&_f2_41));
            vq_121[0] = vq2_141.x;
            vq_121[1] = vq2_141.y;
            vk_122[0] = vk2_142.x;
            vk_122[1] = vk2_142.y;
            float r_143[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_143[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_144 = r_143[0];
            float r_145[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_145[0])) : "r"((gbase_74 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_146 = r_145[0];
            unsigned int r_147[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_147[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_148 = __uint_as_float(r_147[0] << 16);
            unsigned int r_149[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_149[0])) : "r"((qbase_76 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_150 = __uint_as_float(r_149[0] << 16);
            unsigned int r_151[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_151[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_152 = __uint_as_float(r_151[0] << 16);
            unsigned int r_153[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_153[0])) : "r"((kbase_75 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_154 = __uint_as_float(r_153[0] << 16);
            float2 _f2_44 = make_float2(g_s0_144, g_s1_146);
            float2 d2_155 = sub_f32x2(_f2_44, _f2_25);
            float _exp2_14 = approx_exp2(d2_155.x);
            float ev0_156 = _exp2_14;
            float _exp2_15 = approx_exp2(d2_155.y);
            float ev1_157 = _exp2_15;
            float2 _f2_45 = make_float2(k_s0_152, k_s1_154);
            float2 _f2_46 = make_float2(bv_120[2], bv_120[3]);
            float2 kp_158;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_158) : "l"(*(const unsigned long long*)&_f2_45), "l"(*(const unsigned long long*)&_f2_46));
            __nv_bfloat16 _cvt_bf16_14 = (__nv_bfloat16)kp_158.x;
            __nv_bfloat16 kb0_159 = _cvt_bf16_14;
            __nv_bfloat16 _cvt_bf16_15 = (__nv_bfloat16)kp_158.y;
            __nv_bfloat16 kb1_160 = _cvt_bf16_15;
            float2 _f2_47 = make_float2(ev0_156, ev1_157);
            float2 _f2_48 = make_float2(q_s0_148, q_s1_150);
            float2 vq2_161;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_161) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_47));
            float _cvt_f32_14 = (float)kb0_159;
            float _cvt_f32_15 = (float)kb1_160;
            float2 _f2_49 = make_float2(_cvt_f32_14, _cvt_f32_15);
            float2 vk2_162;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_162) : "l"(*(const unsigned long long*)&_f2_49), "l"(*(const unsigned long long*)&_f2_47));
            vq_121[2] = vq2_161.x;
            vq_121[3] = vq2_161.y;
            vk_122[2] = vk2_162.x;
            vk_122[3] = vk2_162.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_121[0]), "f"(vq_121[1]), "f"(vq_121[2]), "f"(vq_121[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_122[0]), "f"(vk_122[1]), "f"(vk_122[2]), "f"(vk_122[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_2[4];
                int kk_3 = w * 4;
                int a_4 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[0])) : "r"(a_4));
                tv_2[0] = ((lane_1 > (kk_3 & 15)) ? 0.0f : tv_2[0]);
                {
                    uint32_t _addr_4 = static_cast<uint32_t>(a_4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(tv_2[0]) : "memory");
                }
                int kk_0_2 = w * 4 + 1;
                int a_1_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[1])) : "r"(a_1_2));
                tv_2[1] = ((lane_1 > (kk_0_2 & 15)) ? 0.0f : tv_2[1]);
                {
                    uint32_t _addr_5 = static_cast<uint32_t>(a_1_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(tv_2[1]) : "memory");
                }
                int kk_2_2 = w * 4 + 2;
                int a_3_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[2])) : "r"(a_3_2));
                tv_2[2] = ((lane_1 > (kk_2_2 & 15)) ? 0.0f : tv_2[2]);
                {
                    uint32_t _addr_6 = static_cast<uint32_t>(a_3_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(tv_2[2]) : "memory");
                }
                int kk_4_2 = w * 4 + 3;
                int a_5_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[3])) : "r"(a_5_2));
                tv_2[3] = ((lane_1 > (kk_4_2 & 15)) ? 0.0f : tv_2[3]);
                {
                    uint32_t _addr_7 = static_cast<uint32_t>(a_5_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(tv_2[3]) : "memory");
                }
                int ta_2 = (tiles_addr + 4096 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_2), "f"(tv_2[0]), "f"(tv_2[1]), "f"(tv_2[2]), "f"(tv_2[3]) : "memory");
            }
            float r_163[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_163[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_164 = r_163[0] - 21.640425324000002f;
            int sbase_165 = 16 + 8 * th;
            int gbase_166 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_165 * 128);
            int kbase_167 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_165 * 128);
            int qbase_168 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_50 = make_float2(a_d_164, a_d_164);
            float bv_169[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_169[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_169[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_169[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_169[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_165 * 4)));
            float vq_170[4];
            float vk_171[4];
            float r_172[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_172[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_173 = r_172[0];
            float r_174[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_174[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_175 = r_174[0];
            unsigned int r_176[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_176[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_177 = __uint_as_float(r_176[0] << 16);
            unsigned int r_178[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_178[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_179 = __uint_as_float(r_178[0] << 16);
            unsigned int r_180[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_180[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_181 = __uint_as_float(r_180[0] << 16);
            unsigned int r_182[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_182[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_183 = __uint_as_float(r_182[0] << 16);
            float2 _f2_51 = make_float2(g_s0_173, g_s1_175);
            float2 d2_184 = sub_f32x2(_f2_51, _f2_50);
            float _exp2_16 = approx_exp2(d2_184.x);
            float ev0_185 = _exp2_16;
            float _exp2_17 = approx_exp2(d2_184.y);
            float ev1_186 = _exp2_17;
            float2 _f2_52 = make_float2(k_s0_181, k_s1_183);
            float2 _f2_53 = make_float2(bv_169[0], bv_169[1]);
            float2 kp_187;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_187) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f2_53));
            __nv_bfloat16 _cvt_bf16_16 = (__nv_bfloat16)kp_187.x;
            __nv_bfloat16 kb0_188 = _cvt_bf16_16;
            __nv_bfloat16 _cvt_bf16_17 = (__nv_bfloat16)kp_187.y;
            __nv_bfloat16 kb1_189 = _cvt_bf16_17;
            float2 _f2_54 = make_float2(ev0_185, ev1_186);
            float2 _f2_55 = make_float2(q_s0_177, q_s1_179);
            float2 vq2_190;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_190) : "l"(*(const unsigned long long*)&_f2_55), "l"(*(const unsigned long long*)&_f2_54));
            float _cvt_f32_16 = (float)kb0_188;
            float _cvt_f32_17 = (float)kb1_189;
            float2 _f2_56 = make_float2(_cvt_f32_16, _cvt_f32_17);
            float2 vk2_191;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_191) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f2_54));
            vq_170[0] = vq2_190.x;
            vq_170[1] = vq2_190.y;
            vk_171[0] = vk2_191.x;
            vk_171[1] = vk2_191.y;
            float r_192[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_192[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_193 = r_192[0];
            float r_194[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_194[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_195 = r_194[0];
            unsigned int r_196[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_196[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_197 = __uint_as_float(r_196[0] << 16);
            unsigned int r_198[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_198[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_199 = __uint_as_float(r_198[0] << 16);
            unsigned int r_200[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_200[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_201 = __uint_as_float(r_200[0] << 16);
            unsigned int r_202[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_202[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_203 = __uint_as_float(r_202[0] << 16);
            float2 _f2_57 = make_float2(g_s0_193, g_s1_195);
            float2 d2_204 = sub_f32x2(_f2_57, _f2_50);
            float _exp2_18 = approx_exp2(d2_204.x);
            float ev0_205 = _exp2_18;
            float _exp2_19 = approx_exp2(d2_204.y);
            float ev1_206 = _exp2_19;
            float2 _f2_58 = make_float2(k_s0_201, k_s1_203);
            float2 _f2_59 = make_float2(bv_169[2], bv_169[3]);
            float2 kp_207;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_207) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f2_59));
            __nv_bfloat16 _cvt_bf16_18 = (__nv_bfloat16)kp_207.x;
            __nv_bfloat16 kb0_208 = _cvt_bf16_18;
            __nv_bfloat16 _cvt_bf16_19 = (__nv_bfloat16)kp_207.y;
            __nv_bfloat16 kb1_209 = _cvt_bf16_19;
            float2 _f2_60 = make_float2(ev0_205, ev1_206);
            float2 _f2_61 = make_float2(q_s0_197, q_s1_199);
            float2 vq2_210;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_210) : "l"(*(const unsigned long long*)&_f2_61), "l"(*(const unsigned long long*)&_f2_60));
            float _cvt_f32_18 = (float)kb0_208;
            float _cvt_f32_19 = (float)kb1_209;
            float2 _f2_62 = make_float2(_cvt_f32_18, _cvt_f32_19);
            float2 vk2_211;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_211) : "l"(*(const unsigned long long*)&_f2_62), "l"(*(const unsigned long long*)&_f2_60));
            vq_170[2] = vq2_210.x;
            vq_170[3] = vq2_210.y;
            vk_171[2] = vk2_211.x;
            vk_171[3] = vk2_211.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_170[0]), "f"(vq_170[1]), "f"(vq_170[2]), "f"(vq_170[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_171[0]), "f"(vk_171[1]), "f"(vk_171[2]), "f"(vk_171[3]) : "memory");
            float bv_212[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_212[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_212[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_212[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_212[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_165 * 4 + 16)));
            float vq_213[4];
            float vk_214[4];
            float r_215[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_215[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_216 = r_215[0];
            float r_217[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_217[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_218 = r_217[0];
            unsigned int r_219[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_219[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_220 = __uint_as_float(r_219[0] << 16);
            unsigned int r_221[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_221[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_222 = __uint_as_float(r_221[0] << 16);
            unsigned int r_223[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_223[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_224 = __uint_as_float(r_223[0] << 16);
            unsigned int r_225[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_225[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_226 = __uint_as_float(r_225[0] << 16);
            float2 _f2_63 = make_float2(g_s0_216, g_s1_218);
            float2 d2_227 = sub_f32x2(_f2_63, _f2_50);
            float _exp2_20 = approx_exp2(d2_227.x);
            float ev0_228 = _exp2_20;
            float _exp2_21 = approx_exp2(d2_227.y);
            float ev1_229 = _exp2_21;
            float2 _f2_64 = make_float2(k_s0_224, k_s1_226);
            float2 _f2_65 = make_float2(bv_212[0], bv_212[1]);
            float2 kp_230;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_230) : "l"(*(const unsigned long long*)&_f2_64), "l"(*(const unsigned long long*)&_f2_65));
            __nv_bfloat16 _cvt_bf16_20 = (__nv_bfloat16)kp_230.x;
            __nv_bfloat16 kb0_231 = _cvt_bf16_20;
            __nv_bfloat16 _cvt_bf16_21 = (__nv_bfloat16)kp_230.y;
            __nv_bfloat16 kb1_232 = _cvt_bf16_21;
            float2 _f2_66 = make_float2(ev0_228, ev1_229);
            float2 _f2_67 = make_float2(q_s0_220, q_s1_222);
            float2 vq2_233;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_233) : "l"(*(const unsigned long long*)&_f2_67), "l"(*(const unsigned long long*)&_f2_66));
            float _cvt_f32_20 = (float)kb0_231;
            float _cvt_f32_21 = (float)kb1_232;
            float2 _f2_68 = make_float2(_cvt_f32_20, _cvt_f32_21);
            float2 vk2_234;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_234) : "l"(*(const unsigned long long*)&_f2_68), "l"(*(const unsigned long long*)&_f2_66));
            vq_213[0] = vq2_233.x;
            vq_213[1] = vq2_233.y;
            vk_214[0] = vk2_234.x;
            vk_214[1] = vk2_234.y;
            float r_235[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_235[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_236 = r_235[0];
            float r_237[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_237[0])) : "r"((gbase_166 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_238 = r_237[0];
            unsigned int r_239[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_239[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_240 = __uint_as_float(r_239[0] << 16);
            unsigned int r_241[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_241[0])) : "r"((qbase_168 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_242 = __uint_as_float(r_241[0] << 16);
            unsigned int r_243[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_243[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_244 = __uint_as_float(r_243[0] << 16);
            unsigned int r_245[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_245[0])) : "r"((kbase_167 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_246 = __uint_as_float(r_245[0] << 16);
            float2 _f2_69 = make_float2(g_s0_236, g_s1_238);
            float2 d2_247 = sub_f32x2(_f2_69, _f2_50);
            float _exp2_22 = approx_exp2(d2_247.x);
            float ev0_248 = _exp2_22;
            float _exp2_23 = approx_exp2(d2_247.y);
            float ev1_249 = _exp2_23;
            float2 _f2_70 = make_float2(k_s0_244, k_s1_246);
            float2 _f2_71 = make_float2(bv_212[2], bv_212[3]);
            float2 kp_250;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_250) : "l"(*(const unsigned long long*)&_f2_70), "l"(*(const unsigned long long*)&_f2_71));
            __nv_bfloat16 _cvt_bf16_22 = (__nv_bfloat16)kp_250.x;
            __nv_bfloat16 kb0_251 = _cvt_bf16_22;
            __nv_bfloat16 _cvt_bf16_23 = (__nv_bfloat16)kp_250.y;
            __nv_bfloat16 kb1_252 = _cvt_bf16_23;
            float2 _f2_72 = make_float2(ev0_248, ev1_249);
            float2 _f2_73 = make_float2(q_s0_240, q_s1_242);
            float2 vq2_253;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_253) : "l"(*(const unsigned long long*)&_f2_73), "l"(*(const unsigned long long*)&_f2_72));
            float _cvt_f32_22 = (float)kb0_251;
            float _cvt_f32_23 = (float)kb1_252;
            float2 _f2_74 = make_float2(_cvt_f32_22, _cvt_f32_23);
            float2 vk2_254;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_254) : "l"(*(const unsigned long long*)&_f2_74), "l"(*(const unsigned long long*)&_f2_72));
            vq_213[2] = vq2_253.x;
            vq_213[3] = vq2_253.y;
            vk_214[2] = vk2_254.x;
            vk_214[3] = vk2_254.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_213[0]), "f"(vq_213[1]), "f"(vq_213[2]), "f"(vq_213[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_214[0]), "f"(vk_214[1]), "f"(vk_214[2]), "f"(vk_214[3]) : "memory");
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
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_3[4];
                int kk_5 = w * 4;
                int a_6 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_5 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_5 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[0])) : "r"(a_6));
                int kk_0_3 = w * 4 + 1;
                int a_1_3 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[1])) : "r"(a_1_3));
                int kk_2_3 = w * 4 + 2;
                int a_3_3 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[2])) : "r"(a_3_3));
                int kk_4_3 = w * 4 + 3;
                int a_5_3 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_4_3 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4_3 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_3[3])) : "r"(a_5_3));
                int ta_3 = (tiles_addr + 6144 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_3), "f"(tv_3[0]), "f"(tv_3[1]), "f"(tv_3[2]), "f"(tv_3[3]) : "memory");
            }
            float r_255[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_255[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_256 = r_255[0];
            int sbase_257 = 32 + 8 * th;
            int gbase_258 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_257 * 128);
            int kbase_259 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_257 * 128);
            int qbase_260 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float2 _f2_75 = make_float2(a_d_256, a_d_256);
            float bv_261[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_261[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_261[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_261[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_261[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_257 * 4)));
            float vq_262[4];
            float vk_263[4];
            float r_264[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_264[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_265 = r_264[0];
            float r_266[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_266[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_267 = r_266[0];
            unsigned int r_268[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_268[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_269 = __uint_as_float(r_268[0] << 16);
            unsigned int r_270[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_270[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_271 = __uint_as_float(r_270[0] << 16);
            unsigned int r_272[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_272[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_273 = __uint_as_float(r_272[0] << 16);
            unsigned int r_274[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_274[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_275 = __uint_as_float(r_274[0] << 16);
            float2 _f2_76 = make_float2(g_s0_265, g_s1_267);
            float2 d2_276 = sub_f32x2(_f2_76, _f2_75);
            float _exp2_24 = approx_exp2(d2_276.x);
            float ev0_277 = _exp2_24;
            float _exp2_25 = approx_exp2(d2_276.y);
            float ev1_278 = _exp2_25;
            float2 _f2_77 = make_float2(k_s0_273, k_s1_275);
            float2 _f2_78 = make_float2(bv_261[0], bv_261[1]);
            float2 kp_279;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_279) : "l"(*(const unsigned long long*)&_f2_77), "l"(*(const unsigned long long*)&_f2_78));
            __nv_bfloat16 _cvt_bf16_24 = (__nv_bfloat16)kp_279.x;
            __nv_bfloat16 kb0_280 = _cvt_bf16_24;
            __nv_bfloat16 _cvt_bf16_25 = (__nv_bfloat16)kp_279.y;
            __nv_bfloat16 kb1_281 = _cvt_bf16_25;
            float2 _f2_79 = make_float2(ev0_277, ev1_278);
            float2 _f2_80 = make_float2(q_s0_269, q_s1_271);
            float2 vq2_282;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_282) : "l"(*(const unsigned long long*)&_f2_80), "l"(*(const unsigned long long*)&_f2_79));
            float _cvt_f32_24 = (float)kb0_280;
            float _cvt_f32_25 = (float)kb1_281;
            float2 _f2_81 = make_float2(_cvt_f32_24, _cvt_f32_25);
            float2 vk2_283;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_283) : "l"(*(const unsigned long long*)&_f2_81), "l"(*(const unsigned long long*)&_f2_79));
            vq_262[0] = vq2_282.x;
            vq_262[1] = vq2_282.y;
            vk_263[0] = vk2_283.x;
            vk_263[1] = vk2_283.y;
            float r_284[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_284[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_285 = r_284[0];
            float r_286[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_286[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_287 = r_286[0];
            unsigned int r_288[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_288[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_289 = __uint_as_float(r_288[0] << 16);
            unsigned int r_290[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_290[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_291 = __uint_as_float(r_290[0] << 16);
            unsigned int r_292[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_292[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_293 = __uint_as_float(r_292[0] << 16);
            unsigned int r_294[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_294[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_295 = __uint_as_float(r_294[0] << 16);
            float2 _f2_82 = make_float2(g_s0_285, g_s1_287);
            float2 d2_296 = sub_f32x2(_f2_82, _f2_75);
            float _exp2_26 = approx_exp2(d2_296.x);
            float ev0_297 = _exp2_26;
            float _exp2_27 = approx_exp2(d2_296.y);
            float ev1_298 = _exp2_27;
            float2 _f2_83 = make_float2(k_s0_293, k_s1_295);
            float2 _f2_84 = make_float2(bv_261[2], bv_261[3]);
            float2 kp_299;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_299) : "l"(*(const unsigned long long*)&_f2_83), "l"(*(const unsigned long long*)&_f2_84));
            __nv_bfloat16 _cvt_bf16_26 = (__nv_bfloat16)kp_299.x;
            __nv_bfloat16 kb0_300 = _cvt_bf16_26;
            __nv_bfloat16 _cvt_bf16_27 = (__nv_bfloat16)kp_299.y;
            __nv_bfloat16 kb1_301 = _cvt_bf16_27;
            float2 _f2_85 = make_float2(ev0_297, ev1_298);
            float2 _f2_86 = make_float2(q_s0_289, q_s1_291);
            float2 vq2_302;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_302) : "l"(*(const unsigned long long*)&_f2_86), "l"(*(const unsigned long long*)&_f2_85));
            float _cvt_f32_26 = (float)kb0_300;
            float _cvt_f32_27 = (float)kb1_301;
            float2 _f2_87 = make_float2(_cvt_f32_26, _cvt_f32_27);
            float2 vk2_303;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_303) : "l"(*(const unsigned long long*)&_f2_87), "l"(*(const unsigned long long*)&_f2_85));
            vq_262[2] = vq2_302.x;
            vq_262[3] = vq2_302.y;
            vk_263[2] = vk2_303.x;
            vk_263[3] = vk2_303.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_262[0]), "f"(vq_262[1]), "f"(vq_262[2]), "f"(vq_262[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_263[0]), "f"(vk_263[1]), "f"(vk_263[2]), "f"(vk_263[3]) : "memory");
            float bv_304[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_304[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_304[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_304[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_304[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_257 * 4 + 16)));
            float vq_305[4];
            float vk_306[4];
            float r_307[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_307[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_308 = r_307[0];
            float r_309[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_309[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_310 = r_309[0];
            unsigned int r_311[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_311[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_312 = __uint_as_float(r_311[0] << 16);
            unsigned int r_313[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_313[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_314 = __uint_as_float(r_313[0] << 16);
            unsigned int r_315[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_315[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_316 = __uint_as_float(r_315[0] << 16);
            unsigned int r_317[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_317[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_318 = __uint_as_float(r_317[0] << 16);
            float2 _f2_88 = make_float2(g_s0_308, g_s1_310);
            float2 d2_319 = sub_f32x2(_f2_88, _f2_75);
            float _exp2_28 = approx_exp2(d2_319.x);
            float ev0_320 = _exp2_28;
            float _exp2_29 = approx_exp2(d2_319.y);
            float ev1_321 = _exp2_29;
            float2 _f2_89 = make_float2(k_s0_316, k_s1_318);
            float2 _f2_90 = make_float2(bv_304[0], bv_304[1]);
            float2 kp_322;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_322) : "l"(*(const unsigned long long*)&_f2_89), "l"(*(const unsigned long long*)&_f2_90));
            __nv_bfloat16 _cvt_bf16_28 = (__nv_bfloat16)kp_322.x;
            __nv_bfloat16 kb0_323 = _cvt_bf16_28;
            __nv_bfloat16 _cvt_bf16_29 = (__nv_bfloat16)kp_322.y;
            __nv_bfloat16 kb1_324 = _cvt_bf16_29;
            float2 _f2_91 = make_float2(ev0_320, ev1_321);
            float2 _f2_92 = make_float2(q_s0_312, q_s1_314);
            float2 vq2_325;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_325) : "l"(*(const unsigned long long*)&_f2_92), "l"(*(const unsigned long long*)&_f2_91));
            float _cvt_f32_28 = (float)kb0_323;
            float _cvt_f32_29 = (float)kb1_324;
            float2 _f2_93 = make_float2(_cvt_f32_28, _cvt_f32_29);
            float2 vk2_326;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_326) : "l"(*(const unsigned long long*)&_f2_93), "l"(*(const unsigned long long*)&_f2_91));
            vq_305[0] = vq2_325.x;
            vq_305[1] = vq2_325.y;
            vk_306[0] = vk2_326.x;
            vk_306[1] = vk2_326.y;
            float r_327[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_327[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_328 = r_327[0];
            float r_329[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_329[0])) : "r"((gbase_258 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_330 = r_329[0];
            unsigned int r_331[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_331[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_332 = __uint_as_float(r_331[0] << 16);
            unsigned int r_333[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_333[0])) : "r"((qbase_260 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_334 = __uint_as_float(r_333[0] << 16);
            unsigned int r_335[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_335[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_336 = __uint_as_float(r_335[0] << 16);
            unsigned int r_337[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_337[0])) : "r"((kbase_259 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_338 = __uint_as_float(r_337[0] << 16);
            float2 _f2_94 = make_float2(g_s0_328, g_s1_330);
            float2 d2_339 = sub_f32x2(_f2_94, _f2_75);
            float _exp2_30 = approx_exp2(d2_339.x);
            float ev0_340 = _exp2_30;
            float _exp2_31 = approx_exp2(d2_339.y);
            float ev1_341 = _exp2_31;
            float2 _f2_95 = make_float2(k_s0_336, k_s1_338);
            float2 _f2_96 = make_float2(bv_304[2], bv_304[3]);
            float2 kp_342;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_342) : "l"(*(const unsigned long long*)&_f2_95), "l"(*(const unsigned long long*)&_f2_96));
            __nv_bfloat16 _cvt_bf16_30 = (__nv_bfloat16)kp_342.x;
            __nv_bfloat16 kb0_343 = _cvt_bf16_30;
            __nv_bfloat16 _cvt_bf16_31 = (__nv_bfloat16)kp_342.y;
            __nv_bfloat16 kb1_344 = _cvt_bf16_31;
            float2 _f2_97 = make_float2(ev0_340, ev1_341);
            float2 _f2_98 = make_float2(q_s0_332, q_s1_334);
            float2 vq2_345;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_345) : "l"(*(const unsigned long long*)&_f2_98), "l"(*(const unsigned long long*)&_f2_97));
            float _cvt_f32_30 = (float)kb0_343;
            float _cvt_f32_31 = (float)kb1_344;
            float2 _f2_99 = make_float2(_cvt_f32_30, _cvt_f32_31);
            float2 vk2_346;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_346) : "l"(*(const unsigned long long*)&_f2_99), "l"(*(const unsigned long long*)&_f2_97));
            vq_305[2] = vq2_345.x;
            vq_305[3] = vq2_345.y;
            vk_306[2] = vk2_346.x;
            vk_306[3] = vk2_346.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_305[0]), "f"(vq_305[1]), "f"(vq_305[2]), "f"(vq_305[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_306[0]), "f"(vk_306[1]), "f"(vk_306[2]), "f"(vk_306[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_4[4];
                int kk_6 = w * 4;
                int a_7 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_6 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_6 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[0])) : "r"(a_7));
                int kk_0_4 = w * 4 + 1;
                int a_1_4 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[1])) : "r"(a_1_4));
                int kk_2_4 = w * 4 + 2;
                int a_3_4 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[2])) : "r"(a_3_4));
                int kk_4_4 = w * 4 + 3;
                int a_5_4 = (dab_addr + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_4 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_4 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_4[3])) : "r"(a_5_4));
                int ta_4 = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_4), "f"(tv_4[0]), "f"(tv_4[1]), "f"(tv_4[2]), "f"(tv_4[3]) : "memory");
            }
            float r_347[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_347[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_348 = r_347[0];
            int sbase_349 = 32 + 8 * th;
            int gbase_350 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_349 * 128);
            int kbase_351 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_349 * 128);
            int qbase_352 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float2 _f2_100 = make_float2(a_d_348, a_d_348);
            float bv_353[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_353[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_353[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_353[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_353[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_349 * 4)));
            float vq_354[4];
            float vk_355[4];
            float r_356[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_356[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_357 = r_356[0];
            float r_358[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_358[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_359 = r_358[0];
            unsigned int r_360[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_360[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_361 = __uint_as_float(r_360[0] << 16);
            unsigned int r_362[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_362[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_363 = __uint_as_float(r_362[0] << 16);
            unsigned int r_364[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_364[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_365 = __uint_as_float(r_364[0] << 16);
            unsigned int r_366[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_366[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_367 = __uint_as_float(r_366[0] << 16);
            float2 _f2_101 = make_float2(g_s0_357, g_s1_359);
            float2 d2_368 = sub_f32x2(_f2_101, _f2_100);
            float _exp2_32 = approx_exp2(d2_368.x);
            float ev0_369 = _exp2_32;
            float _exp2_33 = approx_exp2(d2_368.y);
            float ev1_370 = _exp2_33;
            float2 _f2_102 = make_float2(k_s0_365, k_s1_367);
            float2 _f2_103 = make_float2(bv_353[0], bv_353[1]);
            float2 kp_371;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_371) : "l"(*(const unsigned long long*)&_f2_102), "l"(*(const unsigned long long*)&_f2_103));
            __nv_bfloat16 _cvt_bf16_32 = (__nv_bfloat16)kp_371.x;
            __nv_bfloat16 kb0_372 = _cvt_bf16_32;
            __nv_bfloat16 _cvt_bf16_33 = (__nv_bfloat16)kp_371.y;
            __nv_bfloat16 kb1_373 = _cvt_bf16_33;
            float2 _f2_104 = make_float2(ev0_369, ev1_370);
            float2 _f2_105 = make_float2(q_s0_361, q_s1_363);
            float2 vq2_374;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_374) : "l"(*(const unsigned long long*)&_f2_105), "l"(*(const unsigned long long*)&_f2_104));
            float _cvt_f32_32 = (float)kb0_372;
            float _cvt_f32_33 = (float)kb1_373;
            float2 _f2_106 = make_float2(_cvt_f32_32, _cvt_f32_33);
            float2 vk2_375;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_375) : "l"(*(const unsigned long long*)&_f2_106), "l"(*(const unsigned long long*)&_f2_104));
            vq_354[0] = vq2_374.x;
            vq_354[1] = vq2_374.y;
            vk_355[0] = vk2_375.x;
            vk_355[1] = vk2_375.y;
            float r_376[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_376[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_377 = r_376[0];
            float r_378[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_378[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_379 = r_378[0];
            unsigned int r_380[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_380[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_381 = __uint_as_float(r_380[0] << 16);
            unsigned int r_382[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_382[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_383 = __uint_as_float(r_382[0] << 16);
            unsigned int r_384[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_384[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_385 = __uint_as_float(r_384[0] << 16);
            unsigned int r_386[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_386[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_387 = __uint_as_float(r_386[0] << 16);
            float2 _f2_107 = make_float2(g_s0_377, g_s1_379);
            float2 d2_388 = sub_f32x2(_f2_107, _f2_100);
            float _exp2_34 = approx_exp2(d2_388.x);
            float ev0_389 = _exp2_34;
            float _exp2_35 = approx_exp2(d2_388.y);
            float ev1_390 = _exp2_35;
            float2 _f2_108 = make_float2(k_s0_385, k_s1_387);
            float2 _f2_109 = make_float2(bv_353[2], bv_353[3]);
            float2 kp_391;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_391) : "l"(*(const unsigned long long*)&_f2_108), "l"(*(const unsigned long long*)&_f2_109));
            __nv_bfloat16 _cvt_bf16_34 = (__nv_bfloat16)kp_391.x;
            __nv_bfloat16 kb0_392 = _cvt_bf16_34;
            __nv_bfloat16 _cvt_bf16_35 = (__nv_bfloat16)kp_391.y;
            __nv_bfloat16 kb1_393 = _cvt_bf16_35;
            float2 _f2_110 = make_float2(ev0_389, ev1_390);
            float2 _f2_111 = make_float2(q_s0_381, q_s1_383);
            float2 vq2_394;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_394) : "l"(*(const unsigned long long*)&_f2_111), "l"(*(const unsigned long long*)&_f2_110));
            float _cvt_f32_34 = (float)kb0_392;
            float _cvt_f32_35 = (float)kb1_393;
            float2 _f2_112 = make_float2(_cvt_f32_34, _cvt_f32_35);
            float2 vk2_395;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_395) : "l"(*(const unsigned long long*)&_f2_112), "l"(*(const unsigned long long*)&_f2_110));
            vq_354[2] = vq2_394.x;
            vq_354[3] = vq2_394.y;
            vk_355[2] = vk2_395.x;
            vk_355[3] = vk2_395.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_354[0]), "f"(vq_354[1]), "f"(vq_354[2]), "f"(vq_354[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_355[0]), "f"(vk_355[1]), "f"(vk_355[2]), "f"(vk_355[3]) : "memory");
            float bv_396[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_396[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_396[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_396[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_396[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_349 * 4 + 16)));
            float vq_397[4];
            float vk_398[4];
            float r_399[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_399[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_400 = r_399[0];
            float r_401[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_401[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_402 = r_401[0];
            unsigned int r_403[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_403[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_404 = __uint_as_float(r_403[0] << 16);
            unsigned int r_405[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_405[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_406 = __uint_as_float(r_405[0] << 16);
            unsigned int r_407[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_407[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_408 = __uint_as_float(r_407[0] << 16);
            unsigned int r_409[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_409[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_410 = __uint_as_float(r_409[0] << 16);
            float2 _f2_113 = make_float2(g_s0_400, g_s1_402);
            float2 d2_411 = sub_f32x2(_f2_113, _f2_100);
            float _exp2_36 = approx_exp2(d2_411.x);
            float ev0_412 = _exp2_36;
            float _exp2_37 = approx_exp2(d2_411.y);
            float ev1_413 = _exp2_37;
            float2 _f2_114 = make_float2(k_s0_408, k_s1_410);
            float2 _f2_115 = make_float2(bv_396[0], bv_396[1]);
            float2 kp_414;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_414) : "l"(*(const unsigned long long*)&_f2_114), "l"(*(const unsigned long long*)&_f2_115));
            __nv_bfloat16 _cvt_bf16_36 = (__nv_bfloat16)kp_414.x;
            __nv_bfloat16 kb0_415 = _cvt_bf16_36;
            __nv_bfloat16 _cvt_bf16_37 = (__nv_bfloat16)kp_414.y;
            __nv_bfloat16 kb1_416 = _cvt_bf16_37;
            float2 _f2_116 = make_float2(ev0_412, ev1_413);
            float2 _f2_117 = make_float2(q_s0_404, q_s1_406);
            float2 vq2_417;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_417) : "l"(*(const unsigned long long*)&_f2_117), "l"(*(const unsigned long long*)&_f2_116));
            float _cvt_f32_36 = (float)kb0_415;
            float _cvt_f32_37 = (float)kb1_416;
            float2 _f2_118 = make_float2(_cvt_f32_36, _cvt_f32_37);
            float2 vk2_418;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_418) : "l"(*(const unsigned long long*)&_f2_118), "l"(*(const unsigned long long*)&_f2_116));
            vq_397[0] = vq2_417.x;
            vq_397[1] = vq2_417.y;
            vk_398[0] = vk2_418.x;
            vk_398[1] = vk2_418.y;
            float r_419[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_419[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_420 = r_419[0];
            float r_421[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_421[0])) : "r"((gbase_350 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_422 = r_421[0];
            unsigned int r_423[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_423[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_424 = __uint_as_float(r_423[0] << 16);
            unsigned int r_425[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_425[0])) : "r"((qbase_352 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_426 = __uint_as_float(r_425[0] << 16);
            unsigned int r_427[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_427[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_428 = __uint_as_float(r_427[0] << 16);
            unsigned int r_429[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_429[0])) : "r"((kbase_351 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_430 = __uint_as_float(r_429[0] << 16);
            float2 _f2_119 = make_float2(g_s0_420, g_s1_422);
            float2 d2_431 = sub_f32x2(_f2_119, _f2_100);
            float _exp2_38 = approx_exp2(d2_431.x);
            float ev0_432 = _exp2_38;
            float _exp2_39 = approx_exp2(d2_431.y);
            float ev1_433 = _exp2_39;
            float2 _f2_120 = make_float2(k_s0_428, k_s1_430);
            float2 _f2_121 = make_float2(bv_396[2], bv_396[3]);
            float2 kp_434;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_434) : "l"(*(const unsigned long long*)&_f2_120), "l"(*(const unsigned long long*)&_f2_121));
            __nv_bfloat16 _cvt_bf16_38 = (__nv_bfloat16)kp_434.x;
            __nv_bfloat16 kb0_435 = _cvt_bf16_38;
            __nv_bfloat16 _cvt_bf16_39 = (__nv_bfloat16)kp_434.y;
            __nv_bfloat16 kb1_436 = _cvt_bf16_39;
            float2 _f2_122 = make_float2(ev0_432, ev1_433);
            float2 _f2_123 = make_float2(q_s0_424, q_s1_426);
            float2 vq2_437;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_437) : "l"(*(const unsigned long long*)&_f2_123), "l"(*(const unsigned long long*)&_f2_122));
            float _cvt_f32_38 = (float)kb0_435;
            float _cvt_f32_39 = (float)kb1_436;
            float2 _f2_124 = make_float2(_cvt_f32_38, _cvt_f32_39);
            float2 vk2_438;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_438) : "l"(*(const unsigned long long*)&_f2_124), "l"(*(const unsigned long long*)&_f2_122));
            vq_397[2] = vq2_437.x;
            vq_397[3] = vq2_437.y;
            vk_398[2] = vk2_438.x;
            vk_398[3] = vk2_438.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_397[0]), "f"(vq_397[1]), "f"(vq_397[2]), "f"(vq_397[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_398[0]), "f"(vk_398[1]), "f"(vk_398[2]), "f"(vk_398[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_5[4];
                int kk_7 = w * 4;
                int a_8 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[0])) : "r"(a_8));
                tv_5[0] = ((lane_1 > (kk_7 & 15)) ? 0.0f : tv_5[0]);
                {
                    uint32_t _addr_8 = static_cast<uint32_t>(a_8);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(tv_5[0]) : "memory");
                }
                int kk_0_5 = w * 4 + 1;
                int a_1_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[1])) : "r"(a_1_5));
                tv_5[1] = ((lane_1 > (kk_0_5 & 15)) ? 0.0f : tv_5[1]);
                {
                    uint32_t _addr_9 = static_cast<uint32_t>(a_1_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(tv_5[1]) : "memory");
                }
                int kk_2_5 = w * 4 + 2;
                int a_3_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[2])) : "r"(a_3_5));
                tv_5[2] = ((lane_1 > (kk_2_5 & 15)) ? 0.0f : tv_5[2]);
                {
                    uint32_t _addr_10 = static_cast<uint32_t>(a_3_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(tv_5[2]) : "memory");
                }
                int kk_4_5 = w * 4 + 3;
                int a_5_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[3])) : "r"(a_5_5));
                tv_5[3] = ((lane_1 > (kk_4_5 & 15)) ? 0.0f : tv_5[3]);
                {
                    uint32_t _addr_11 = static_cast<uint32_t>(a_5_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(tv_5[3]) : "memory");
                }
                int ta_5 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_5), "f"(tv_5[0]), "f"(tv_5[1]), "f"(tv_5[2]), "f"(tv_5[3]) : "memory");
            }
            float r_439[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_439[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_440 = r_439[0] - 21.640425324000002f;
            int sbase_441 = 32 + 8 * th;
            int gbase_442 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_441 * 128);
            int kbase_443 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_441 * 128);
            int qbase_444 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float2 _f2_125 = make_float2(a_d_440, a_d_440);
            float bv_445[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_445[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_445[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_445[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_445[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_441 * 4)));
            float vq_446[4];
            float vk_447[4];
            float r_448[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_448[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_449 = r_448[0];
            float r_450[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_450[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_451 = r_450[0];
            unsigned int r_452[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_452[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_453 = __uint_as_float(r_452[0] << 16);
            unsigned int r_454[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_454[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_455 = __uint_as_float(r_454[0] << 16);
            unsigned int r_456[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_456[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_457 = __uint_as_float(r_456[0] << 16);
            unsigned int r_458[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_458[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_459 = __uint_as_float(r_458[0] << 16);
            float2 _f2_126 = make_float2(g_s0_449, g_s1_451);
            float2 d2_460 = sub_f32x2(_f2_126, _f2_125);
            float _exp2_40 = approx_exp2(d2_460.x);
            float ev0_461 = _exp2_40;
            float _exp2_41 = approx_exp2(d2_460.y);
            float ev1_462 = _exp2_41;
            float2 _f2_127 = make_float2(k_s0_457, k_s1_459);
            float2 _f2_128 = make_float2(bv_445[0], bv_445[1]);
            float2 kp_463;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_463) : "l"(*(const unsigned long long*)&_f2_127), "l"(*(const unsigned long long*)&_f2_128));
            __nv_bfloat16 _cvt_bf16_40 = (__nv_bfloat16)kp_463.x;
            __nv_bfloat16 kb0_464 = _cvt_bf16_40;
            __nv_bfloat16 _cvt_bf16_41 = (__nv_bfloat16)kp_463.y;
            __nv_bfloat16 kb1_465 = _cvt_bf16_41;
            float2 _f2_129 = make_float2(ev0_461, ev1_462);
            float2 _f2_130 = make_float2(q_s0_453, q_s1_455);
            float2 vq2_466;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_466) : "l"(*(const unsigned long long*)&_f2_130), "l"(*(const unsigned long long*)&_f2_129));
            float _cvt_f32_40 = (float)kb0_464;
            float _cvt_f32_41 = (float)kb1_465;
            float2 _f2_131 = make_float2(_cvt_f32_40, _cvt_f32_41);
            float2 vk2_467;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_467) : "l"(*(const unsigned long long*)&_f2_131), "l"(*(const unsigned long long*)&_f2_129));
            vq_446[0] = vq2_466.x;
            vq_446[1] = vq2_466.y;
            vk_447[0] = vk2_467.x;
            vk_447[1] = vk2_467.y;
            float r_468[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_468[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_469 = r_468[0];
            float r_470[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_470[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_471 = r_470[0];
            unsigned int r_472[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_472[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_473 = __uint_as_float(r_472[0] << 16);
            unsigned int r_474[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_474[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_475 = __uint_as_float(r_474[0] << 16);
            unsigned int r_476[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_476[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_477 = __uint_as_float(r_476[0] << 16);
            unsigned int r_478[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_478[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_479 = __uint_as_float(r_478[0] << 16);
            float2 _f2_132 = make_float2(g_s0_469, g_s1_471);
            float2 d2_480 = sub_f32x2(_f2_132, _f2_125);
            float _exp2_42 = approx_exp2(d2_480.x);
            float ev0_481 = _exp2_42;
            float _exp2_43 = approx_exp2(d2_480.y);
            float ev1_482 = _exp2_43;
            float2 _f2_133 = make_float2(k_s0_477, k_s1_479);
            float2 _f2_134 = make_float2(bv_445[2], bv_445[3]);
            float2 kp_483;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_483) : "l"(*(const unsigned long long*)&_f2_133), "l"(*(const unsigned long long*)&_f2_134));
            __nv_bfloat16 _cvt_bf16_42 = (__nv_bfloat16)kp_483.x;
            __nv_bfloat16 kb0_484 = _cvt_bf16_42;
            __nv_bfloat16 _cvt_bf16_43 = (__nv_bfloat16)kp_483.y;
            __nv_bfloat16 kb1_485 = _cvt_bf16_43;
            float2 _f2_135 = make_float2(ev0_481, ev1_482);
            float2 _f2_136 = make_float2(q_s0_473, q_s1_475);
            float2 vq2_486;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_486) : "l"(*(const unsigned long long*)&_f2_136), "l"(*(const unsigned long long*)&_f2_135));
            float _cvt_f32_42 = (float)kb0_484;
            float _cvt_f32_43 = (float)kb1_485;
            float2 _f2_137 = make_float2(_cvt_f32_42, _cvt_f32_43);
            float2 vk2_487;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_487) : "l"(*(const unsigned long long*)&_f2_137), "l"(*(const unsigned long long*)&_f2_135));
            vq_446[2] = vq2_486.x;
            vq_446[3] = vq2_486.y;
            vk_447[2] = vk2_487.x;
            vk_447[3] = vk2_487.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_446[0]), "f"(vq_446[1]), "f"(vq_446[2]), "f"(vq_446[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_447[0]), "f"(vk_447[1]), "f"(vk_447[2]), "f"(vk_447[3]) : "memory");
            float bv_488[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_488[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_488[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_488[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_488[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_441 * 4 + 16)));
            float vq_489[4];
            float vk_490[4];
            float r_491[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_491[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_492 = r_491[0];
            float r_493[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_493[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_494 = r_493[0];
            unsigned int r_495[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_495[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_496 = __uint_as_float(r_495[0] << 16);
            unsigned int r_497[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_497[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_498 = __uint_as_float(r_497[0] << 16);
            unsigned int r_499[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_499[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_500 = __uint_as_float(r_499[0] << 16);
            unsigned int r_501[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_501[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_502 = __uint_as_float(r_501[0] << 16);
            float2 _f2_138 = make_float2(g_s0_492, g_s1_494);
            float2 d2_503 = sub_f32x2(_f2_138, _f2_125);
            float _exp2_44 = approx_exp2(d2_503.x);
            float ev0_504 = _exp2_44;
            float _exp2_45 = approx_exp2(d2_503.y);
            float ev1_505 = _exp2_45;
            float2 _f2_139 = make_float2(k_s0_500, k_s1_502);
            float2 _f2_140 = make_float2(bv_488[0], bv_488[1]);
            float2 kp_506;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_506) : "l"(*(const unsigned long long*)&_f2_139), "l"(*(const unsigned long long*)&_f2_140));
            __nv_bfloat16 _cvt_bf16_44 = (__nv_bfloat16)kp_506.x;
            __nv_bfloat16 kb0_507 = _cvt_bf16_44;
            __nv_bfloat16 _cvt_bf16_45 = (__nv_bfloat16)kp_506.y;
            __nv_bfloat16 kb1_508 = _cvt_bf16_45;
            float2 _f2_141 = make_float2(ev0_504, ev1_505);
            float2 _f2_142 = make_float2(q_s0_496, q_s1_498);
            float2 vq2_509;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_509) : "l"(*(const unsigned long long*)&_f2_142), "l"(*(const unsigned long long*)&_f2_141));
            float _cvt_f32_44 = (float)kb0_507;
            float _cvt_f32_45 = (float)kb1_508;
            float2 _f2_143 = make_float2(_cvt_f32_44, _cvt_f32_45);
            float2 vk2_510;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_510) : "l"(*(const unsigned long long*)&_f2_143), "l"(*(const unsigned long long*)&_f2_141));
            vq_489[0] = vq2_509.x;
            vq_489[1] = vq2_509.y;
            vk_490[0] = vk2_510.x;
            vk_490[1] = vk2_510.y;
            float r_511[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_511[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_512 = r_511[0];
            float r_513[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_513[0])) : "r"((gbase_442 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_514 = r_513[0];
            unsigned int r_515[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_515[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_516 = __uint_as_float(r_515[0] << 16);
            unsigned int r_517[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_517[0])) : "r"((qbase_444 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_518 = __uint_as_float(r_517[0] << 16);
            unsigned int r_519[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_519[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_520 = __uint_as_float(r_519[0] << 16);
            unsigned int r_521[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_521[0])) : "r"((kbase_443 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_522 = __uint_as_float(r_521[0] << 16);
            float2 _f2_144 = make_float2(g_s0_512, g_s1_514);
            float2 d2_523 = sub_f32x2(_f2_144, _f2_125);
            float _exp2_46 = approx_exp2(d2_523.x);
            float ev0_524 = _exp2_46;
            float _exp2_47 = approx_exp2(d2_523.y);
            float ev1_525 = _exp2_47;
            float2 _f2_145 = make_float2(k_s0_520, k_s1_522);
            float2 _f2_146 = make_float2(bv_488[2], bv_488[3]);
            float2 kp_526;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_526) : "l"(*(const unsigned long long*)&_f2_145), "l"(*(const unsigned long long*)&_f2_146));
            __nv_bfloat16 _cvt_bf16_46 = (__nv_bfloat16)kp_526.x;
            __nv_bfloat16 kb0_527 = _cvt_bf16_46;
            __nv_bfloat16 _cvt_bf16_47 = (__nv_bfloat16)kp_526.y;
            __nv_bfloat16 kb1_528 = _cvt_bf16_47;
            float2 _f2_147 = make_float2(ev0_524, ev1_525);
            float2 _f2_148 = make_float2(q_s0_516, q_s1_518);
            float2 vq2_529;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_529) : "l"(*(const unsigned long long*)&_f2_148), "l"(*(const unsigned long long*)&_f2_147));
            float _cvt_f32_46 = (float)kb0_527;
            float _cvt_f32_47 = (float)kb1_528;
            float2 _f2_149 = make_float2(_cvt_f32_46, _cvt_f32_47);
            float2 vk2_530;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_530) : "l"(*(const unsigned long long*)&_f2_149), "l"(*(const unsigned long long*)&_f2_147));
            vq_489[2] = vq2_529.x;
            vq_489[3] = vq2_529.y;
            vk_490[2] = vk2_530.x;
            vk_490[3] = vk2_530.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_489[0]), "f"(vq_489[1]), "f"(vq_489[2]), "f"(vq_489[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_490[0]), "f"(vk_490[1]), "f"(vk_490[2]), "f"(vk_490[3]) : "memory");
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
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_6[4];
                int kk_8 = w * 4;
                int a_9 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_8 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_8 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[0])) : "r"(a_9));
                int kk_0_6 = w * 4 + 1;
                int a_1_6 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_0_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[1])) : "r"(a_1_6));
                int kk_2_6 = w * 4 + 2;
                int a_3_6 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_2_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[2])) : "r"(a_3_6));
                int kk_4_6 = w * 4 + 3;
                int a_5_6 = (dab_addr + 8192 + (unsigned int)(lane_1 / 32 * 4096 + kk_4_6 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4_6 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_6[3])) : "r"(a_5_6));
                int ta_6 = (tiles_addr + 4096 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_6), "f"(tv_6[0]), "f"(tv_6[1]), "f"(tv_6[2]), "f"(tv_6[3]) : "memory");
            }
            float r_531[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_531[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_532 = r_531[0];
            int sbase_533 = 48 + 8 * th;
            int gbase_534 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_533 * 128);
            int kbase_535 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_533 * 128);
            int qbase_536 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_150 = make_float2(a_d_532, a_d_532);
            float bv_537[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_537[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_537[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_537[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_537[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_533 * 4)));
            float vq_538[4];
            float vk_539[4];
            float r_540[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_540[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_541 = r_540[0];
            float r_542[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_542[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_543 = r_542[0];
            unsigned int r_544[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_544[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_545 = __uint_as_float(r_544[0] << 16);
            unsigned int r_546[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_546[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_547 = __uint_as_float(r_546[0] << 16);
            unsigned int r_548[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_548[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_549 = __uint_as_float(r_548[0] << 16);
            unsigned int r_550[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_550[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_551 = __uint_as_float(r_550[0] << 16);
            float2 _f2_151 = make_float2(g_s0_541, g_s1_543);
            float2 d2_552 = sub_f32x2(_f2_151, _f2_150);
            float _exp2_48 = approx_exp2(d2_552.x);
            float ev0_553 = _exp2_48;
            float _exp2_49 = approx_exp2(d2_552.y);
            float ev1_554 = _exp2_49;
            float2 _f2_152 = make_float2(k_s0_549, k_s1_551);
            float2 _f2_153 = make_float2(bv_537[0], bv_537[1]);
            float2 kp_555;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_555) : "l"(*(const unsigned long long*)&_f2_152), "l"(*(const unsigned long long*)&_f2_153));
            __nv_bfloat16 _cvt_bf16_48 = (__nv_bfloat16)kp_555.x;
            __nv_bfloat16 kb0_556 = _cvt_bf16_48;
            __nv_bfloat16 _cvt_bf16_49 = (__nv_bfloat16)kp_555.y;
            __nv_bfloat16 kb1_557 = _cvt_bf16_49;
            float2 _f2_154 = make_float2(ev0_553, ev1_554);
            float2 _f2_155 = make_float2(q_s0_545, q_s1_547);
            float2 vq2_558;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_558) : "l"(*(const unsigned long long*)&_f2_155), "l"(*(const unsigned long long*)&_f2_154));
            float _cvt_f32_48 = (float)kb0_556;
            float _cvt_f32_49 = (float)kb1_557;
            float2 _f2_156 = make_float2(_cvt_f32_48, _cvt_f32_49);
            float2 vk2_559;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_559) : "l"(*(const unsigned long long*)&_f2_156), "l"(*(const unsigned long long*)&_f2_154));
            vq_538[0] = vq2_558.x;
            vq_538[1] = vq2_558.y;
            vk_539[0] = vk2_559.x;
            vk_539[1] = vk2_559.y;
            float r_560[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_560[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_561 = r_560[0];
            float r_562[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_562[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_563 = r_562[0];
            unsigned int r_564[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_564[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_565 = __uint_as_float(r_564[0] << 16);
            unsigned int r_566[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_566[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_567 = __uint_as_float(r_566[0] << 16);
            unsigned int r_568[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_568[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_569 = __uint_as_float(r_568[0] << 16);
            unsigned int r_570[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_570[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_571 = __uint_as_float(r_570[0] << 16);
            float2 _f2_157 = make_float2(g_s0_561, g_s1_563);
            float2 d2_572 = sub_f32x2(_f2_157, _f2_150);
            float _exp2_50 = approx_exp2(d2_572.x);
            float ev0_573 = _exp2_50;
            float _exp2_51 = approx_exp2(d2_572.y);
            float ev1_574 = _exp2_51;
            float2 _f2_158 = make_float2(k_s0_569, k_s1_571);
            float2 _f2_159 = make_float2(bv_537[2], bv_537[3]);
            float2 kp_575;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_575) : "l"(*(const unsigned long long*)&_f2_158), "l"(*(const unsigned long long*)&_f2_159));
            __nv_bfloat16 _cvt_bf16_50 = (__nv_bfloat16)kp_575.x;
            __nv_bfloat16 kb0_576 = _cvt_bf16_50;
            __nv_bfloat16 _cvt_bf16_51 = (__nv_bfloat16)kp_575.y;
            __nv_bfloat16 kb1_577 = _cvt_bf16_51;
            float2 _f2_160 = make_float2(ev0_573, ev1_574);
            float2 _f2_161 = make_float2(q_s0_565, q_s1_567);
            float2 vq2_578;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_578) : "l"(*(const unsigned long long*)&_f2_161), "l"(*(const unsigned long long*)&_f2_160));
            float _cvt_f32_50 = (float)kb0_576;
            float _cvt_f32_51 = (float)kb1_577;
            float2 _f2_162 = make_float2(_cvt_f32_50, _cvt_f32_51);
            float2 vk2_579;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_579) : "l"(*(const unsigned long long*)&_f2_162), "l"(*(const unsigned long long*)&_f2_160));
            vq_538[2] = vq2_578.x;
            vq_538[3] = vq2_578.y;
            vk_539[2] = vk2_579.x;
            vk_539[3] = vk2_579.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_538[0]), "f"(vq_538[1]), "f"(vq_538[2]), "f"(vq_538[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_539[0]), "f"(vk_539[1]), "f"(vk_539[2]), "f"(vk_539[3]) : "memory");
            float bv_580[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_580[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_580[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_580[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_580[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_533 * 4 + 16)));
            float vq_581[4];
            float vk_582[4];
            float r_583[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_583[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_584 = r_583[0];
            float r_585[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_585[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_586 = r_585[0];
            unsigned int r_587[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_587[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_588 = __uint_as_float(r_587[0] << 16);
            unsigned int r_589[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_589[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_590 = __uint_as_float(r_589[0] << 16);
            unsigned int r_591[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_591[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_592 = __uint_as_float(r_591[0] << 16);
            unsigned int r_593[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_593[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_594 = __uint_as_float(r_593[0] << 16);
            float2 _f2_163 = make_float2(g_s0_584, g_s1_586);
            float2 d2_595 = sub_f32x2(_f2_163, _f2_150);
            float _exp2_52 = approx_exp2(d2_595.x);
            float ev0_596 = _exp2_52;
            float _exp2_53 = approx_exp2(d2_595.y);
            float ev1_597 = _exp2_53;
            float2 _f2_164 = make_float2(k_s0_592, k_s1_594);
            float2 _f2_165 = make_float2(bv_580[0], bv_580[1]);
            float2 kp_598;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_598) : "l"(*(const unsigned long long*)&_f2_164), "l"(*(const unsigned long long*)&_f2_165));
            __nv_bfloat16 _cvt_bf16_52 = (__nv_bfloat16)kp_598.x;
            __nv_bfloat16 kb0_599 = _cvt_bf16_52;
            __nv_bfloat16 _cvt_bf16_53 = (__nv_bfloat16)kp_598.y;
            __nv_bfloat16 kb1_600 = _cvt_bf16_53;
            float2 _f2_166 = make_float2(ev0_596, ev1_597);
            float2 _f2_167 = make_float2(q_s0_588, q_s1_590);
            float2 vq2_601;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_601) : "l"(*(const unsigned long long*)&_f2_167), "l"(*(const unsigned long long*)&_f2_166));
            float _cvt_f32_52 = (float)kb0_599;
            float _cvt_f32_53 = (float)kb1_600;
            float2 _f2_168 = make_float2(_cvt_f32_52, _cvt_f32_53);
            float2 vk2_602;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_602) : "l"(*(const unsigned long long*)&_f2_168), "l"(*(const unsigned long long*)&_f2_166));
            vq_581[0] = vq2_601.x;
            vq_581[1] = vq2_601.y;
            vk_582[0] = vk2_602.x;
            vk_582[1] = vk2_602.y;
            float r_603[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_603[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_604 = r_603[0];
            float r_605[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_605[0])) : "r"((gbase_534 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_606 = r_605[0];
            unsigned int r_607[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_607[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_608 = __uint_as_float(r_607[0] << 16);
            unsigned int r_609[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_609[0])) : "r"((qbase_536 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_610 = __uint_as_float(r_609[0] << 16);
            unsigned int r_611[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_611[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_612 = __uint_as_float(r_611[0] << 16);
            unsigned int r_613[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_613[0])) : "r"((kbase_535 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_614 = __uint_as_float(r_613[0] << 16);
            float2 _f2_169 = make_float2(g_s0_604, g_s1_606);
            float2 d2_615 = sub_f32x2(_f2_169, _f2_150);
            float _exp2_54 = approx_exp2(d2_615.x);
            float ev0_616 = _exp2_54;
            float _exp2_55 = approx_exp2(d2_615.y);
            float ev1_617 = _exp2_55;
            float2 _f2_170 = make_float2(k_s0_612, k_s1_614);
            float2 _f2_171 = make_float2(bv_580[2], bv_580[3]);
            float2 kp_618;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_618) : "l"(*(const unsigned long long*)&_f2_170), "l"(*(const unsigned long long*)&_f2_171));
            __nv_bfloat16 _cvt_bf16_54 = (__nv_bfloat16)kp_618.x;
            __nv_bfloat16 kb0_619 = _cvt_bf16_54;
            __nv_bfloat16 _cvt_bf16_55 = (__nv_bfloat16)kp_618.y;
            __nv_bfloat16 kb1_620 = _cvt_bf16_55;
            float2 _f2_172 = make_float2(ev0_616, ev1_617);
            float2 _f2_173 = make_float2(q_s0_608, q_s1_610);
            float2 vq2_621;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_621) : "l"(*(const unsigned long long*)&_f2_173), "l"(*(const unsigned long long*)&_f2_172));
            float _cvt_f32_54 = (float)kb0_619;
            float _cvt_f32_55 = (float)kb1_620;
            float2 _f2_174 = make_float2(_cvt_f32_54, _cvt_f32_55);
            float2 vk2_622;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_622) : "l"(*(const unsigned long long*)&_f2_174), "l"(*(const unsigned long long*)&_f2_172));
            vq_581[2] = vq2_621.x;
            vq_581[3] = vq2_621.y;
            vk_582[2] = vk2_622.x;
            vk_582[3] = vk2_622.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_581[0]), "f"(vq_581[1]), "f"(vq_581[2]), "f"(vq_581[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_582[0]), "f"(vk_582[1]), "f"(vk_582[2]), "f"(vk_582[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_7[4];
                int kk_9 = w * 4;
                int a_10 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_9 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_9 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[0])) : "r"(a_10));
                int kk_0_7 = w * 4 + 1;
                int a_1_7 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_7 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_7 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[1])) : "r"(a_1_7));
                int kk_2_7 = w * 4 + 2;
                int a_3_7 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_7 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_7 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[2])) : "r"(a_3_7));
                int kk_4_7 = w * 4 + 3;
                int a_5_7 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_7 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_7 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_7[3])) : "r"(a_5_7));
                int ta_7 = (tiles_addr + 6144 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_7), "f"(tv_7[0]), "f"(tv_7[1]), "f"(tv_7[2]), "f"(tv_7[3]) : "memory");
            }
            float r_623[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_623[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_624 = r_623[0];
            int sbase_625 = 48 + 8 * th;
            int gbase_626 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_625 * 128);
            int kbase_627 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_625 * 128);
            int qbase_628 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_175 = make_float2(a_d_624, a_d_624);
            float bv_629[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_629[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_629[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_629[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_629[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_625 * 4)));
            float vq_630[4];
            float vk_631[4];
            float r_632[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_632[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_633 = r_632[0];
            float r_634[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_634[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_635 = r_634[0];
            unsigned int r_636[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_636[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_637 = __uint_as_float(r_636[0] << 16);
            unsigned int r_638[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_638[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_639 = __uint_as_float(r_638[0] << 16);
            unsigned int r_640[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_640[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_641 = __uint_as_float(r_640[0] << 16);
            unsigned int r_642[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_642[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_643 = __uint_as_float(r_642[0] << 16);
            float2 _f2_176 = make_float2(g_s0_633, g_s1_635);
            float2 d2_644 = sub_f32x2(_f2_176, _f2_175);
            float _exp2_56 = approx_exp2(d2_644.x);
            float ev0_645 = _exp2_56;
            float _exp2_57 = approx_exp2(d2_644.y);
            float ev1_646 = _exp2_57;
            float2 _f2_177 = make_float2(k_s0_641, k_s1_643);
            float2 _f2_178 = make_float2(bv_629[0], bv_629[1]);
            float2 kp_647;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_647) : "l"(*(const unsigned long long*)&_f2_177), "l"(*(const unsigned long long*)&_f2_178));
            __nv_bfloat16 _cvt_bf16_56 = (__nv_bfloat16)kp_647.x;
            __nv_bfloat16 kb0_648 = _cvt_bf16_56;
            __nv_bfloat16 _cvt_bf16_57 = (__nv_bfloat16)kp_647.y;
            __nv_bfloat16 kb1_649 = _cvt_bf16_57;
            float2 _f2_179 = make_float2(ev0_645, ev1_646);
            float2 _f2_180 = make_float2(q_s0_637, q_s1_639);
            float2 vq2_650;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_650) : "l"(*(const unsigned long long*)&_f2_180), "l"(*(const unsigned long long*)&_f2_179));
            float _cvt_f32_56 = (float)kb0_648;
            float _cvt_f32_57 = (float)kb1_649;
            float2 _f2_181 = make_float2(_cvt_f32_56, _cvt_f32_57);
            float2 vk2_651;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_651) : "l"(*(const unsigned long long*)&_f2_181), "l"(*(const unsigned long long*)&_f2_179));
            vq_630[0] = vq2_650.x;
            vq_630[1] = vq2_650.y;
            vk_631[0] = vk2_651.x;
            vk_631[1] = vk2_651.y;
            float r_652[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_652[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_653 = r_652[0];
            float r_654[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_654[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_655 = r_654[0];
            unsigned int r_656[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_656[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_657 = __uint_as_float(r_656[0] << 16);
            unsigned int r_658[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_658[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_659 = __uint_as_float(r_658[0] << 16);
            unsigned int r_660[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_660[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_661 = __uint_as_float(r_660[0] << 16);
            unsigned int r_662[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_662[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_663 = __uint_as_float(r_662[0] << 16);
            float2 _f2_182 = make_float2(g_s0_653, g_s1_655);
            float2 d2_664 = sub_f32x2(_f2_182, _f2_175);
            float _exp2_58 = approx_exp2(d2_664.x);
            float ev0_665 = _exp2_58;
            float _exp2_59 = approx_exp2(d2_664.y);
            float ev1_666 = _exp2_59;
            float2 _f2_183 = make_float2(k_s0_661, k_s1_663);
            float2 _f2_184 = make_float2(bv_629[2], bv_629[3]);
            float2 kp_667;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_667) : "l"(*(const unsigned long long*)&_f2_183), "l"(*(const unsigned long long*)&_f2_184));
            __nv_bfloat16 _cvt_bf16_58 = (__nv_bfloat16)kp_667.x;
            __nv_bfloat16 kb0_668 = _cvt_bf16_58;
            __nv_bfloat16 _cvt_bf16_59 = (__nv_bfloat16)kp_667.y;
            __nv_bfloat16 kb1_669 = _cvt_bf16_59;
            float2 _f2_185 = make_float2(ev0_665, ev1_666);
            float2 _f2_186 = make_float2(q_s0_657, q_s1_659);
            float2 vq2_670;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_670) : "l"(*(const unsigned long long*)&_f2_186), "l"(*(const unsigned long long*)&_f2_185));
            float _cvt_f32_58 = (float)kb0_668;
            float _cvt_f32_59 = (float)kb1_669;
            float2 _f2_187 = make_float2(_cvt_f32_58, _cvt_f32_59);
            float2 vk2_671;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_671) : "l"(*(const unsigned long long*)&_f2_187), "l"(*(const unsigned long long*)&_f2_185));
            vq_630[2] = vq2_670.x;
            vq_630[3] = vq2_670.y;
            vk_631[2] = vk2_671.x;
            vk_631[3] = vk2_671.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_630[0]), "f"(vq_630[1]), "f"(vq_630[2]), "f"(vq_630[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_631[0]), "f"(vk_631[1]), "f"(vk_631[2]), "f"(vk_631[3]) : "memory");
            float bv_672[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_672[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_672[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_672[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_672[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_625 * 4 + 16)));
            float vq_673[4];
            float vk_674[4];
            float r_675[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_675[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_676 = r_675[0];
            float r_677[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_677[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_678 = r_677[0];
            unsigned int r_679[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_679[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_680 = __uint_as_float(r_679[0] << 16);
            unsigned int r_681[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_681[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_682 = __uint_as_float(r_681[0] << 16);
            unsigned int r_683[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_683[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_684 = __uint_as_float(r_683[0] << 16);
            unsigned int r_685[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_685[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_686 = __uint_as_float(r_685[0] << 16);
            float2 _f2_188 = make_float2(g_s0_676, g_s1_678);
            float2 d2_687 = sub_f32x2(_f2_188, _f2_175);
            float _exp2_60 = approx_exp2(d2_687.x);
            float ev0_688 = _exp2_60;
            float _exp2_61 = approx_exp2(d2_687.y);
            float ev1_689 = _exp2_61;
            float2 _f2_189 = make_float2(k_s0_684, k_s1_686);
            float2 _f2_190 = make_float2(bv_672[0], bv_672[1]);
            float2 kp_690;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_690) : "l"(*(const unsigned long long*)&_f2_189), "l"(*(const unsigned long long*)&_f2_190));
            __nv_bfloat16 _cvt_bf16_60 = (__nv_bfloat16)kp_690.x;
            __nv_bfloat16 kb0_691 = _cvt_bf16_60;
            __nv_bfloat16 _cvt_bf16_61 = (__nv_bfloat16)kp_690.y;
            __nv_bfloat16 kb1_692 = _cvt_bf16_61;
            float2 _f2_191 = make_float2(ev0_688, ev1_689);
            float2 _f2_192 = make_float2(q_s0_680, q_s1_682);
            float2 vq2_693;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_693) : "l"(*(const unsigned long long*)&_f2_192), "l"(*(const unsigned long long*)&_f2_191));
            float _cvt_f32_60 = (float)kb0_691;
            float _cvt_f32_61 = (float)kb1_692;
            float2 _f2_193 = make_float2(_cvt_f32_60, _cvt_f32_61);
            float2 vk2_694;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_694) : "l"(*(const unsigned long long*)&_f2_193), "l"(*(const unsigned long long*)&_f2_191));
            vq_673[0] = vq2_693.x;
            vq_673[1] = vq2_693.y;
            vk_674[0] = vk2_694.x;
            vk_674[1] = vk2_694.y;
            float r_695[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_695[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_696 = r_695[0];
            float r_697[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_697[0])) : "r"((gbase_626 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_698 = r_697[0];
            unsigned int r_699[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_699[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_700 = __uint_as_float(r_699[0] << 16);
            unsigned int r_701[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_701[0])) : "r"((qbase_628 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_702 = __uint_as_float(r_701[0] << 16);
            unsigned int r_703[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_703[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_704 = __uint_as_float(r_703[0] << 16);
            unsigned int r_705[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_705[0])) : "r"((kbase_627 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_706 = __uint_as_float(r_705[0] << 16);
            float2 _f2_194 = make_float2(g_s0_696, g_s1_698);
            float2 d2_707 = sub_f32x2(_f2_194, _f2_175);
            float _exp2_62 = approx_exp2(d2_707.x);
            float ev0_708 = _exp2_62;
            float _exp2_63 = approx_exp2(d2_707.y);
            float ev1_709 = _exp2_63;
            float2 _f2_195 = make_float2(k_s0_704, k_s1_706);
            float2 _f2_196 = make_float2(bv_672[2], bv_672[3]);
            float2 kp_710;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_710) : "l"(*(const unsigned long long*)&_f2_195), "l"(*(const unsigned long long*)&_f2_196));
            __nv_bfloat16 _cvt_bf16_62 = (__nv_bfloat16)kp_710.x;
            __nv_bfloat16 kb0_711 = _cvt_bf16_62;
            __nv_bfloat16 _cvt_bf16_63 = (__nv_bfloat16)kp_710.y;
            __nv_bfloat16 kb1_712 = _cvt_bf16_63;
            float2 _f2_197 = make_float2(ev0_708, ev1_709);
            float2 _f2_198 = make_float2(q_s0_700, q_s1_702);
            float2 vq2_713;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_713) : "l"(*(const unsigned long long*)&_f2_198), "l"(*(const unsigned long long*)&_f2_197));
            float _cvt_f32_62 = (float)kb0_711;
            float _cvt_f32_63 = (float)kb1_712;
            float2 _f2_199 = make_float2(_cvt_f32_62, _cvt_f32_63);
            float2 vk2_714;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_714) : "l"(*(const unsigned long long*)&_f2_199), "l"(*(const unsigned long long*)&_f2_197));
            vq_673[2] = vq2_713.x;
            vq_673[3] = vq2_713.y;
            vk_674[2] = vk2_714.x;
            vk_674[3] = vk2_714.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_673[0]), "f"(vq_673[1]), "f"(vq_673[2]), "f"(vq_673[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_674[0]), "f"(vk_674[1]), "f"(vk_674[2]), "f"(vk_674[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_8[4];
                int kk_10 = w * 4;
                int a_11 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_10 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_10 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[0])) : "r"(a_11));
                int kk_0_8 = w * 4 + 1;
                int a_1_8 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_0_8 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_0_8 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[1])) : "r"(a_1_8));
                int kk_2_8 = w * 4 + 2;
                int a_3_8 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_2_8 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_2_8 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[2])) : "r"(a_3_8));
                int kk_4_8 = w * 4 + 3;
                int a_5_8 = (dab_addr + 8192 + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_4_8 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_4_8 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_8[3])) : "r"(a_5_8));
                int ta_8 = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_8), "f"(tv_8[0]), "f"(tv_8[1]), "f"(tv_8[2]), "f"(tv_8[3]) : "memory");
            }
            float r_715[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_715[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_716 = r_715[0];
            int sbase_717 = 48 + 8 * th;
            int gbase_718 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_717 * 128);
            int kbase_719 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_717 * 128);
            int qbase_720 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_200 = make_float2(a_d_716, a_d_716);
            float bv_721[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_721[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_721[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_721[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_721[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_717 * 4)));
            float vq_722[4];
            float vk_723[4];
            float r_724[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_724[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_725 = r_724[0];
            float r_726[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_726[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_727 = r_726[0];
            unsigned int r_728[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_728[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_729 = __uint_as_float(r_728[0] << 16);
            unsigned int r_730[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_730[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_731 = __uint_as_float(r_730[0] << 16);
            unsigned int r_732[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_732[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_733 = __uint_as_float(r_732[0] << 16);
            unsigned int r_734[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_734[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_735 = __uint_as_float(r_734[0] << 16);
            float2 _f2_201 = make_float2(g_s0_725, g_s1_727);
            float2 d2_736 = sub_f32x2(_f2_201, _f2_200);
            float _exp2_64 = approx_exp2(d2_736.x);
            float ev0_737 = _exp2_64;
            float _exp2_65 = approx_exp2(d2_736.y);
            float ev1_738 = _exp2_65;
            float2 _f2_202 = make_float2(k_s0_733, k_s1_735);
            float2 _f2_203 = make_float2(bv_721[0], bv_721[1]);
            float2 kp_739;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_739) : "l"(*(const unsigned long long*)&_f2_202), "l"(*(const unsigned long long*)&_f2_203));
            __nv_bfloat16 _cvt_bf16_64 = (__nv_bfloat16)kp_739.x;
            __nv_bfloat16 kb0_740 = _cvt_bf16_64;
            __nv_bfloat16 _cvt_bf16_65 = (__nv_bfloat16)kp_739.y;
            __nv_bfloat16 kb1_741 = _cvt_bf16_65;
            float2 _f2_204 = make_float2(ev0_737, ev1_738);
            float2 _f2_205 = make_float2(q_s0_729, q_s1_731);
            float2 vq2_742;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_742) : "l"(*(const unsigned long long*)&_f2_205), "l"(*(const unsigned long long*)&_f2_204));
            float _cvt_f32_64 = (float)kb0_740;
            float _cvt_f32_65 = (float)kb1_741;
            float2 _f2_206 = make_float2(_cvt_f32_64, _cvt_f32_65);
            float2 vk2_743;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_743) : "l"(*(const unsigned long long*)&_f2_206), "l"(*(const unsigned long long*)&_f2_204));
            vq_722[0] = vq2_742.x;
            vq_722[1] = vq2_742.y;
            vk_723[0] = vk2_743.x;
            vk_723[1] = vk2_743.y;
            float r_744[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_744[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_745 = r_744[0];
            float r_746[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_746[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_747 = r_746[0];
            unsigned int r_748[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_748[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_749 = __uint_as_float(r_748[0] << 16);
            unsigned int r_750[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_750[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_751 = __uint_as_float(r_750[0] << 16);
            unsigned int r_752[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_752[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_753 = __uint_as_float(r_752[0] << 16);
            unsigned int r_754[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_754[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_755 = __uint_as_float(r_754[0] << 16);
            float2 _f2_207 = make_float2(g_s0_745, g_s1_747);
            float2 d2_756 = sub_f32x2(_f2_207, _f2_200);
            float _exp2_66 = approx_exp2(d2_756.x);
            float ev0_757 = _exp2_66;
            float _exp2_67 = approx_exp2(d2_756.y);
            float ev1_758 = _exp2_67;
            float2 _f2_208 = make_float2(k_s0_753, k_s1_755);
            float2 _f2_209 = make_float2(bv_721[2], bv_721[3]);
            float2 kp_759;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_759) : "l"(*(const unsigned long long*)&_f2_208), "l"(*(const unsigned long long*)&_f2_209));
            __nv_bfloat16 _cvt_bf16_66 = (__nv_bfloat16)kp_759.x;
            __nv_bfloat16 kb0_760 = _cvt_bf16_66;
            __nv_bfloat16 _cvt_bf16_67 = (__nv_bfloat16)kp_759.y;
            __nv_bfloat16 kb1_761 = _cvt_bf16_67;
            float2 _f2_210 = make_float2(ev0_757, ev1_758);
            float2 _f2_211 = make_float2(q_s0_749, q_s1_751);
            float2 vq2_762;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_762) : "l"(*(const unsigned long long*)&_f2_211), "l"(*(const unsigned long long*)&_f2_210));
            float _cvt_f32_66 = (float)kb0_760;
            float _cvt_f32_67 = (float)kb1_761;
            float2 _f2_212 = make_float2(_cvt_f32_66, _cvt_f32_67);
            float2 vk2_763;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_763) : "l"(*(const unsigned long long*)&_f2_212), "l"(*(const unsigned long long*)&_f2_210));
            vq_722[2] = vq2_762.x;
            vq_722[3] = vq2_762.y;
            vk_723[2] = vk2_763.x;
            vk_723[3] = vk2_763.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_722[0]), "f"(vq_722[1]), "f"(vq_722[2]), "f"(vq_722[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_723[0]), "f"(vk_723[1]), "f"(vk_723[2]), "f"(vk_723[3]) : "memory");
            float bv_764[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_764[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_764[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_764[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_764[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_717 * 4 + 16)));
            float vq_765[4];
            float vk_766[4];
            float r_767[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_767[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_768 = r_767[0];
            float r_769[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_769[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_770 = r_769[0];
            unsigned int r_771[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_771[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_772 = __uint_as_float(r_771[0] << 16);
            unsigned int r_773[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_773[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_774 = __uint_as_float(r_773[0] << 16);
            unsigned int r_775[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_775[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_776 = __uint_as_float(r_775[0] << 16);
            unsigned int r_777[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_777[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_778 = __uint_as_float(r_777[0] << 16);
            float2 _f2_213 = make_float2(g_s0_768, g_s1_770);
            float2 d2_779 = sub_f32x2(_f2_213, _f2_200);
            float _exp2_68 = approx_exp2(d2_779.x);
            float ev0_780 = _exp2_68;
            float _exp2_69 = approx_exp2(d2_779.y);
            float ev1_781 = _exp2_69;
            float2 _f2_214 = make_float2(k_s0_776, k_s1_778);
            float2 _f2_215 = make_float2(bv_764[0], bv_764[1]);
            float2 kp_782;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_782) : "l"(*(const unsigned long long*)&_f2_214), "l"(*(const unsigned long long*)&_f2_215));
            __nv_bfloat16 _cvt_bf16_68 = (__nv_bfloat16)kp_782.x;
            __nv_bfloat16 kb0_783 = _cvt_bf16_68;
            __nv_bfloat16 _cvt_bf16_69 = (__nv_bfloat16)kp_782.y;
            __nv_bfloat16 kb1_784 = _cvt_bf16_69;
            float2 _f2_216 = make_float2(ev0_780, ev1_781);
            float2 _f2_217 = make_float2(q_s0_772, q_s1_774);
            float2 vq2_785;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_785) : "l"(*(const unsigned long long*)&_f2_217), "l"(*(const unsigned long long*)&_f2_216));
            float _cvt_f32_68 = (float)kb0_783;
            float _cvt_f32_69 = (float)kb1_784;
            float2 _f2_218 = make_float2(_cvt_f32_68, _cvt_f32_69);
            float2 vk2_786;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_786) : "l"(*(const unsigned long long*)&_f2_218), "l"(*(const unsigned long long*)&_f2_216));
            vq_765[0] = vq2_785.x;
            vq_765[1] = vq2_785.y;
            vk_766[0] = vk2_786.x;
            vk_766[1] = vk2_786.y;
            float r_787[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_787[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_788 = r_787[0];
            float r_789[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_789[0])) : "r"((gbase_718 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_790 = r_789[0];
            unsigned int r_791[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_791[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_792 = __uint_as_float(r_791[0] << 16);
            unsigned int r_793[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_793[0])) : "r"((qbase_720 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_794 = __uint_as_float(r_793[0] << 16);
            unsigned int r_795[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_795[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_796 = __uint_as_float(r_795[0] << 16);
            unsigned int r_797[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_797[0])) : "r"((kbase_719 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_798 = __uint_as_float(r_797[0] << 16);
            float2 _f2_219 = make_float2(g_s0_788, g_s1_790);
            float2 d2_799 = sub_f32x2(_f2_219, _f2_200);
            float _exp2_70 = approx_exp2(d2_799.x);
            float ev0_800 = _exp2_70;
            float _exp2_71 = approx_exp2(d2_799.y);
            float ev1_801 = _exp2_71;
            float2 _f2_220 = make_float2(k_s0_796, k_s1_798);
            float2 _f2_221 = make_float2(bv_764[2], bv_764[3]);
            float2 kp_802;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_802) : "l"(*(const unsigned long long*)&_f2_220), "l"(*(const unsigned long long*)&_f2_221));
            __nv_bfloat16 _cvt_bf16_70 = (__nv_bfloat16)kp_802.x;
            __nv_bfloat16 kb0_803 = _cvt_bf16_70;
            __nv_bfloat16 _cvt_bf16_71 = (__nv_bfloat16)kp_802.y;
            __nv_bfloat16 kb1_804 = _cvt_bf16_71;
            float2 _f2_222 = make_float2(ev0_800, ev1_801);
            float2 _f2_223 = make_float2(q_s0_792, q_s1_794);
            float2 vq2_805;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_805) : "l"(*(const unsigned long long*)&_f2_223), "l"(*(const unsigned long long*)&_f2_222));
            float _cvt_f32_70 = (float)kb0_803;
            float _cvt_f32_71 = (float)kb1_804;
            float2 _f2_224 = make_float2(_cvt_f32_70, _cvt_f32_71);
            float2 vk2_806;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_806) : "l"(*(const unsigned long long*)&_f2_224), "l"(*(const unsigned long long*)&_f2_222));
            vq_765[2] = vq2_805.x;
            vq_765[3] = vq2_805.y;
            vk_766[2] = vk2_806.x;
            vk_766[3] = vk2_806.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_765[0]), "f"(vq_765[1]), "f"(vq_765[2]), "f"(vq_765[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_766[0]), "f"(vk_766[1]), "f"(vk_766[2]), "f"(vk_766[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                float tv_9[4];
                int kk_11 = w * 4;
                int a_12 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[0])) : "r"(a_12));
                tv_9[0] = ((lane_1 > (kk_11 & 15)) ? 0.0f : tv_9[0]);
                {
                    uint32_t _addr_12 = static_cast<uint32_t>(a_12);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(tv_9[0]) : "memory");
                }
                int kk_0_9 = w * 4 + 1;
                int a_1_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_0_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_0_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[1])) : "r"(a_1_9));
                tv_9[1] = ((lane_1 > (kk_0_9 & 15)) ? 0.0f : tv_9[1]);
                {
                    uint32_t _addr_13 = static_cast<uint32_t>(a_1_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(tv_9[1]) : "memory");
                }
                int kk_2_9 = w * 4 + 2;
                int a_3_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_2_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_2_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[2])) : "r"(a_3_9));
                tv_9[2] = ((lane_1 > (kk_2_9 & 15)) ? 0.0f : tv_9[2]);
                {
                    uint32_t _addr_14 = static_cast<uint32_t>(a_3_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(tv_9[2]) : "memory");
                }
                int kk_4_9 = w * 4 + 3;
                int a_5_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_4_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_4_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[3])) : "r"(a_5_9));
                tv_9[3] = ((lane_1 > (kk_4_9 & 15)) ? 0.0f : tv_9[3]);
                {
                    uint32_t _addr_15 = static_cast<uint32_t>(a_5_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(tv_9[3]) : "memory");
                }
                int ta_9 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_9), "f"(tv_9[0]), "f"(tv_9[1]), "f"(tv_9[2]), "f"(tv_9[3]) : "memory");
            }
            float r_807[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_807[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_808 = r_807[0] - 21.640425324000002f;
            int sbase_809 = 48 + 8 * th;
            int gbase_810 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_809 * 128);
            int kbase_811 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_809 * 128);
            int qbase_812 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float2 _f2_225 = make_float2(a_d_808, a_d_808);
            float bv_813[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_813[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_813[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_813[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_813[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_809 * 4)));
            float vq_814[4];
            float vk_815[4];
            float r_816[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_816[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_817 = r_816[0];
            float r_818[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_818[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_819 = r_818[0];
            unsigned int r_820[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_820[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_821 = __uint_as_float(r_820[0] << 16);
            unsigned int r_822[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_822[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_823 = __uint_as_float(r_822[0] << 16);
            unsigned int r_824[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_824[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_825 = __uint_as_float(r_824[0] << 16);
            unsigned int r_826[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_826[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_827 = __uint_as_float(r_826[0] << 16);
            float2 _f2_226 = make_float2(g_s0_817, g_s1_819);
            float2 d2_828 = sub_f32x2(_f2_226, _f2_225);
            float _exp2_72 = approx_exp2(d2_828.x);
            float ev0_829 = _exp2_72;
            float _exp2_73 = approx_exp2(d2_828.y);
            float ev1_830 = _exp2_73;
            float2 _f2_227 = make_float2(k_s0_825, k_s1_827);
            float2 _f2_228 = make_float2(bv_813[0], bv_813[1]);
            float2 kp_831;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_831) : "l"(*(const unsigned long long*)&_f2_227), "l"(*(const unsigned long long*)&_f2_228));
            __nv_bfloat16 _cvt_bf16_72 = (__nv_bfloat16)kp_831.x;
            __nv_bfloat16 kb0_832 = _cvt_bf16_72;
            __nv_bfloat16 _cvt_bf16_73 = (__nv_bfloat16)kp_831.y;
            __nv_bfloat16 kb1_833 = _cvt_bf16_73;
            float2 _f2_229 = make_float2(ev0_829, ev1_830);
            float2 _f2_230 = make_float2(q_s0_821, q_s1_823);
            float2 vq2_834;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_834) : "l"(*(const unsigned long long*)&_f2_230), "l"(*(const unsigned long long*)&_f2_229));
            float _cvt_f32_72 = (float)kb0_832;
            float _cvt_f32_73 = (float)kb1_833;
            float2 _f2_231 = make_float2(_cvt_f32_72, _cvt_f32_73);
            float2 vk2_835;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_835) : "l"(*(const unsigned long long*)&_f2_231), "l"(*(const unsigned long long*)&_f2_229));
            vq_814[0] = vq2_834.x;
            vq_814[1] = vq2_834.y;
            vk_815[0] = vk2_835.x;
            vk_815[1] = vk2_835.y;
            float r_836[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_836[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_837 = r_836[0];
            float r_838[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_838[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_839 = r_838[0];
            unsigned int r_840[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_840[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_841 = __uint_as_float(r_840[0] << 16);
            unsigned int r_842[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_842[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_843 = __uint_as_float(r_842[0] << 16);
            unsigned int r_844[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_844[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_845 = __uint_as_float(r_844[0] << 16);
            unsigned int r_846[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_846[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_847 = __uint_as_float(r_846[0] << 16);
            float2 _f2_232 = make_float2(g_s0_837, g_s1_839);
            float2 d2_848 = sub_f32x2(_f2_232, _f2_225);
            float _exp2_74 = approx_exp2(d2_848.x);
            float ev0_849 = _exp2_74;
            float _exp2_75 = approx_exp2(d2_848.y);
            float ev1_850 = _exp2_75;
            float2 _f2_233 = make_float2(k_s0_845, k_s1_847);
            float2 _f2_234 = make_float2(bv_813[2], bv_813[3]);
            float2 kp_851;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_851) : "l"(*(const unsigned long long*)&_f2_233), "l"(*(const unsigned long long*)&_f2_234));
            __nv_bfloat16 _cvt_bf16_74 = (__nv_bfloat16)kp_851.x;
            __nv_bfloat16 kb0_852 = _cvt_bf16_74;
            __nv_bfloat16 _cvt_bf16_75 = (__nv_bfloat16)kp_851.y;
            __nv_bfloat16 kb1_853 = _cvt_bf16_75;
            float2 _f2_235 = make_float2(ev0_849, ev1_850);
            float2 _f2_236 = make_float2(q_s0_841, q_s1_843);
            float2 vq2_854;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_854) : "l"(*(const unsigned long long*)&_f2_236), "l"(*(const unsigned long long*)&_f2_235));
            float _cvt_f32_74 = (float)kb0_852;
            float _cvt_f32_75 = (float)kb1_853;
            float2 _f2_237 = make_float2(_cvt_f32_74, _cvt_f32_75);
            float2 vk2_855;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_855) : "l"(*(const unsigned long long*)&_f2_237), "l"(*(const unsigned long long*)&_f2_235));
            vq_814[2] = vq2_854.x;
            vq_814[3] = vq2_854.y;
            vk_815[2] = vk2_855.x;
            vk_815[3] = vk2_855.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_814[0]), "f"(vq_814[1]), "f"(vq_814[2]), "f"(vq_814[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_815[0]), "f"(vk_815[1]), "f"(vk_815[2]), "f"(vk_815[3]) : "memory");
            float bv_856[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_856[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_856[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_856[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_856[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_809 * 4 + 16)));
            float vq_857[4];
            float vk_858[4];
            float r_859[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_859[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_860 = r_859[0];
            float r_861[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_861[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_862 = r_861[0];
            unsigned int r_863[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_863[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_864 = __uint_as_float(r_863[0] << 16);
            unsigned int r_865[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_865[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_866 = __uint_as_float(r_865[0] << 16);
            unsigned int r_867[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_867[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_868 = __uint_as_float(r_867[0] << 16);
            unsigned int r_869[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_869[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_870 = __uint_as_float(r_869[0] << 16);
            float2 _f2_238 = make_float2(g_s0_860, g_s1_862);
            float2 d2_871 = sub_f32x2(_f2_238, _f2_225);
            float _exp2_76 = approx_exp2(d2_871.x);
            float ev0_872 = _exp2_76;
            float _exp2_77 = approx_exp2(d2_871.y);
            float ev1_873 = _exp2_77;
            float2 _f2_239 = make_float2(k_s0_868, k_s1_870);
            float2 _f2_240 = make_float2(bv_856[0], bv_856[1]);
            float2 kp_874;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_874) : "l"(*(const unsigned long long*)&_f2_239), "l"(*(const unsigned long long*)&_f2_240));
            __nv_bfloat16 _cvt_bf16_76 = (__nv_bfloat16)kp_874.x;
            __nv_bfloat16 kb0_875 = _cvt_bf16_76;
            __nv_bfloat16 _cvt_bf16_77 = (__nv_bfloat16)kp_874.y;
            __nv_bfloat16 kb1_876 = _cvt_bf16_77;
            float2 _f2_241 = make_float2(ev0_872, ev1_873);
            float2 _f2_242 = make_float2(q_s0_864, q_s1_866);
            float2 vq2_877;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_877) : "l"(*(const unsigned long long*)&_f2_242), "l"(*(const unsigned long long*)&_f2_241));
            float _cvt_f32_76 = (float)kb0_875;
            float _cvt_f32_77 = (float)kb1_876;
            float2 _f2_243 = make_float2(_cvt_f32_76, _cvt_f32_77);
            float2 vk2_878;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_878) : "l"(*(const unsigned long long*)&_f2_243), "l"(*(const unsigned long long*)&_f2_241));
            vq_857[0] = vq2_877.x;
            vq_857[1] = vq2_877.y;
            vk_858[0] = vk2_878.x;
            vk_858[1] = vk2_878.y;
            float r_879[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_879[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s0_880 = r_879[0];
            float r_881[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_881[0])) : "r"((gbase_810 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s1_882 = r_881[0];
            unsigned int r_883[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_883[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s0_884 = __uint_as_float(r_883[0] << 16);
            unsigned int r_885[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_885[0])) : "r"((qbase_812 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s1_886 = __uint_as_float(r_885[0] << 16);
            unsigned int r_887[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_887[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_888 = __uint_as_float(r_887[0] << 16);
            unsigned int r_889[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_889[0])) : "r"((kbase_811 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_890 = __uint_as_float(r_889[0] << 16);
            float2 _f2_244 = make_float2(g_s0_880, g_s1_882);
            float2 d2_891 = sub_f32x2(_f2_244, _f2_225);
            float _exp2_78 = approx_exp2(d2_891.x);
            float ev0_892 = _exp2_78;
            float _exp2_79 = approx_exp2(d2_891.y);
            float ev1_893 = _exp2_79;
            float2 _f2_245 = make_float2(k_s0_888, k_s1_890);
            float2 _f2_246 = make_float2(bv_856[2], bv_856[3]);
            float2 kp_894;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_894) : "l"(*(const unsigned long long*)&_f2_245), "l"(*(const unsigned long long*)&_f2_246));
            __nv_bfloat16 _cvt_bf16_78 = (__nv_bfloat16)kp_894.x;
            __nv_bfloat16 kb0_895 = _cvt_bf16_78;
            __nv_bfloat16 _cvt_bf16_79 = (__nv_bfloat16)kp_894.y;
            __nv_bfloat16 kb1_896 = _cvt_bf16_79;
            float2 _f2_247 = make_float2(ev0_892, ev1_893);
            float2 _f2_248 = make_float2(q_s0_884, q_s1_886);
            float2 vq2_897;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_897) : "l"(*(const unsigned long long*)&_f2_248), "l"(*(const unsigned long long*)&_f2_247));
            float _cvt_f32_78 = (float)kb0_895;
            float _cvt_f32_79 = (float)kb1_896;
            float2 _f2_249 = make_float2(_cvt_f32_78, _cvt_f32_79);
            float2 vk2_898;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_898) : "l"(*(const unsigned long long*)&_f2_249), "l"(*(const unsigned long long*)&_f2_247));
            vq_857[2] = vq2_897.x;
            vq_857[3] = vq2_897.y;
            vk_858[2] = vk2_898.x;
            vk_858[3] = vk2_898.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_857[0]), "f"(vq_857[1]), "f"(vq_857[2]), "f"(vq_857[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_858[0]), "f"(vk_858[1]), "f"(vk_858[2]), "f"(vk_858[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_899 = 8 * th;
            int gbase_900 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_899 * 128);
            int kbase_901 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_899 * 128);
            float r_902[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_902[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0 = r_902[0];
            float r_903[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_903[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1 = r_903[0];
            float gs[4];
            float ks[4];
            float r_904[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_904[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[0] = r_904[0];
            unsigned int r_905[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_905[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[0] = __uint_as_float(r_905[0] << 16);
            float r_906[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_906[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[1] = r_906[0];
            unsigned int r_907[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_907[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[1] = __uint_as_float(r_907[0] << 16);
            float r_908[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_908[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[2] = r_908[0];
            unsigned int r_909[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_909[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[2] = __uint_as_float(r_909[0] << 16);
            float r_910[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_910[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[3] = r_910[0];
            unsigned int r_911[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_911[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[3] = __uint_as_float(r_911[0] << 16);
            float2 _f2_250 = make_float2(a0, a0);
            float v[4];
            float2 _f2_251 = make_float2(gs[0], gs[1]);
            float2 d2_912 = sub_f32x2(_f2_250, _f2_251);
            float _exp2_80 = approx_exp2(d2_912.x);
            float e0 = _exp2_80;
            float _exp2_81 = approx_exp2(d2_912.y);
            float e1 = _exp2_81;
            float2 _f2_252 = make_float2(ks[0], ks[1]);
            float2 _f2_253 = make_float2(e0, e1);
            float2 v2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2) : "l"(*(const unsigned long long*)&_f2_252), "l"(*(const unsigned long long*)&_f2_253));
            v[0] = v2.x;
            v[1] = v2.y;
            float2 _f2_254 = make_float2(gs[2], gs[3]);
            float2 d2_913 = sub_f32x2(_f2_250, _f2_254);
            float _exp2_82 = approx_exp2(d2_913.x);
            float e0_914 = _exp2_82;
            float _exp2_83 = approx_exp2(d2_913.y);
            float e1_915 = _exp2_83;
            float2 _f2_255 = make_float2(ks[2], ks[3]);
            float2 _f2_256 = make_float2(e0_914, e1_915);
            float2 v2_916;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_916) : "l"(*(const unsigned long long*)&_f2_255), "l"(*(const unsigned long long*)&_f2_256));
            v[2] = v2_916.x;
            v[3] = v2_916.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
            float2 _f2_257 = make_float2(a1, a1);
            float v_917[4];
            float2 _f2_258 = make_float2(gs[0], gs[1]);
            float2 d2_918 = sub_f32x2(_f2_257, _f2_258);
            float _exp2_84 = approx_exp2(d2_918.x);
            float e0_919 = _exp2_84;
            float _exp2_85 = approx_exp2(d2_918.y);
            float e1_920 = _exp2_85;
            float2 _f2_259 = make_float2(ks[0], ks[1]);
            float2 _f2_260 = make_float2(e0_919, e1_920);
            float2 v2_921;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_921) : "l"(*(const unsigned long long*)&_f2_259), "l"(*(const unsigned long long*)&_f2_260));
            v_917[0] = v2_921.x;
            v_917[1] = v2_921.y;
            float2 _f2_261 = make_float2(gs[2], gs[3]);
            float2 d2_922 = sub_f32x2(_f2_257, _f2_261);
            float _exp2_86 = approx_exp2(d2_922.x);
            float e0_923 = _exp2_86;
            float _exp2_87 = approx_exp2(d2_922.y);
            float e1_924 = _exp2_87;
            float2 _f2_262 = make_float2(ks[2], ks[3]);
            float2 _f2_263 = make_float2(e0_923, e1_924);
            float2 v2_925;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_925) : "l"(*(const unsigned long long*)&_f2_262), "l"(*(const unsigned long long*)&_f2_263));
            v_917[2] = v2_925.x;
            v_917[3] = v2_925.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_917[0]), "f"(v_917[1]), "f"(v_917[2]), "f"(v_917[3]) : "memory");
            float gs_926[4];
            float ks_927[4];
            float r_928[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_928[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_926[0] = r_928[0];
            unsigned int r_929[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_929[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_927[0] = __uint_as_float(r_929[0] << 16);
            float r_930[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_930[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_926[1] = r_930[0];
            unsigned int r_931[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_931[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_927[1] = __uint_as_float(r_931[0] << 16);
            float r_932[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_932[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_926[2] = r_932[0];
            unsigned int r_933[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_933[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_927[2] = __uint_as_float(r_933[0] << 16);
            float r_934[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_934[0])) : "r"((gbase_900 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_926[3] = r_934[0];
            unsigned int r_935[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_935[0])) : "r"((kbase_901 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_927[3] = __uint_as_float(r_935[0] << 16);
            float2 _f2_264 = make_float2(a0, a0);
            float v_936[4];
            float2 _f2_265 = make_float2(gs_926[0], gs_926[1]);
            float2 d2_937 = sub_f32x2(_f2_264, _f2_265);
            float _exp2_88 = approx_exp2(d2_937.x);
            float e0_938 = _exp2_88;
            float _exp2_89 = approx_exp2(d2_937.y);
            float e1_939 = _exp2_89;
            float2 _f2_266 = make_float2(ks_927[0], ks_927[1]);
            float2 _f2_267 = make_float2(e0_938, e1_939);
            float2 v2_940;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_940) : "l"(*(const unsigned long long*)&_f2_266), "l"(*(const unsigned long long*)&_f2_267));
            v_936[0] = v2_940.x;
            v_936[1] = v2_940.y;
            float2 _f2_268 = make_float2(gs_926[2], gs_926[3]);
            float2 d2_941 = sub_f32x2(_f2_264, _f2_268);
            float _exp2_90 = approx_exp2(d2_941.x);
            float e0_942 = _exp2_90;
            float _exp2_91 = approx_exp2(d2_941.y);
            float e1_943 = _exp2_91;
            float2 _f2_269 = make_float2(ks_927[2], ks_927[3]);
            float2 _f2_270 = make_float2(e0_942, e1_943);
            float2 v2_944;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_944) : "l"(*(const unsigned long long*)&_f2_269), "l"(*(const unsigned long long*)&_f2_270));
            v_936[2] = v2_944.x;
            v_936[3] = v2_944.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_936[0]), "f"(v_936[1]), "f"(v_936[2]), "f"(v_936[3]) : "memory");
            float2 _f2_271 = make_float2(a1, a1);
            float v_945[4];
            float2 _f2_272 = make_float2(gs_926[0], gs_926[1]);
            float2 d2_946 = sub_f32x2(_f2_271, _f2_272);
            float _exp2_92 = approx_exp2(d2_946.x);
            float e0_947 = _exp2_92;
            float _exp2_93 = approx_exp2(d2_946.y);
            float e1_948 = _exp2_93;
            float2 _f2_273 = make_float2(ks_927[0], ks_927[1]);
            float2 _f2_274 = make_float2(e0_947, e1_948);
            float2 v2_949;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_949) : "l"(*(const unsigned long long*)&_f2_273), "l"(*(const unsigned long long*)&_f2_274));
            v_945[0] = v2_949.x;
            v_945[1] = v2_949.y;
            float2 _f2_275 = make_float2(gs_926[2], gs_926[3]);
            float2 d2_950 = sub_f32x2(_f2_271, _f2_275);
            float _exp2_94 = approx_exp2(d2_950.x);
            float e0_951 = _exp2_94;
            float _exp2_95 = approx_exp2(d2_950.y);
            float e1_952 = _exp2_95;
            float2 _f2_276 = make_float2(ks_927[2], ks_927[3]);
            float2 _f2_277 = make_float2(e0_951, e1_952);
            float2 v2_953;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_953) : "l"(*(const unsigned long long*)&_f2_276), "l"(*(const unsigned long long*)&_f2_277));
            v_945[2] = v2_953.x;
            v_945[3] = v2_953.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_945[0]), "f"(v_945[1]), "f"(v_945[2]), "f"(v_945[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_954 = 16 + 8 * th;
            int gbase_955 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_954 * 128);
            int kbase_956 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_954 * 128);
            float r_957[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_957[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_958 = r_957[0];
            float r_959[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_959[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1_960 = r_959[0];
            float gs_961[4];
            float ks_962[4];
            float r_963[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_963[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_961[0] = r_963[0];
            unsigned int r_964[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_964[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_962[0] = __uint_as_float(r_964[0] << 16);
            float r_965[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_965[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_961[1] = r_965[0];
            unsigned int r_966[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_966[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_962[1] = __uint_as_float(r_966[0] << 16);
            float r_967[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_967[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_961[2] = r_967[0];
            unsigned int r_968[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_968[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_962[2] = __uint_as_float(r_968[0] << 16);
            float r_969[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_969[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_961[3] = r_969[0];
            unsigned int r_970[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_970[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_962[3] = __uint_as_float(r_970[0] << 16);
            float2 _f2_278 = make_float2(a0_958, a0_958);
            float v_971[4];
            float2 _f2_279 = make_float2(gs_961[0], gs_961[1]);
            float2 d2_972 = sub_f32x2(_f2_278, _f2_279);
            float _exp2_96 = approx_exp2(d2_972.x);
            float e0_973 = _exp2_96;
            float _exp2_97 = approx_exp2(d2_972.y);
            float e1_974 = _exp2_97;
            float2 _f2_280 = make_float2(ks_962[0], ks_962[1]);
            float2 _f2_281 = make_float2(e0_973, e1_974);
            float2 v2_975;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_975) : "l"(*(const unsigned long long*)&_f2_280), "l"(*(const unsigned long long*)&_f2_281));
            v_971[0] = v2_975.x;
            v_971[1] = v2_975.y;
            float2 _f2_282 = make_float2(gs_961[2], gs_961[3]);
            float2 d2_976 = sub_f32x2(_f2_278, _f2_282);
            float _exp2_98 = approx_exp2(d2_976.x);
            float e0_977 = _exp2_98;
            float _exp2_99 = approx_exp2(d2_976.y);
            float e1_978 = _exp2_99;
            float2 _f2_283 = make_float2(ks_962[2], ks_962[3]);
            float2 _f2_284 = make_float2(e0_977, e1_978);
            float2 v2_979;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_979) : "l"(*(const unsigned long long*)&_f2_283), "l"(*(const unsigned long long*)&_f2_284));
            v_971[2] = v2_979.x;
            v_971[3] = v2_979.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_971[0]), "f"(v_971[1]), "f"(v_971[2]), "f"(v_971[3]) : "memory");
            float2 _f2_285 = make_float2(a1_960, a1_960);
            float v_980[4];
            float2 _f2_286 = make_float2(gs_961[0], gs_961[1]);
            float2 d2_981 = sub_f32x2(_f2_285, _f2_286);
            float _exp2_100 = approx_exp2(d2_981.x);
            float e0_982 = _exp2_100;
            float _exp2_101 = approx_exp2(d2_981.y);
            float e1_983 = _exp2_101;
            float2 _f2_287 = make_float2(ks_962[0], ks_962[1]);
            float2 _f2_288 = make_float2(e0_982, e1_983);
            float2 v2_984;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_984) : "l"(*(const unsigned long long*)&_f2_287), "l"(*(const unsigned long long*)&_f2_288));
            v_980[0] = v2_984.x;
            v_980[1] = v2_984.y;
            float2 _f2_289 = make_float2(gs_961[2], gs_961[3]);
            float2 d2_985 = sub_f32x2(_f2_285, _f2_289);
            float _exp2_102 = approx_exp2(d2_985.x);
            float e0_986 = _exp2_102;
            float _exp2_103 = approx_exp2(d2_985.y);
            float e1_987 = _exp2_103;
            float2 _f2_290 = make_float2(ks_962[2], ks_962[3]);
            float2 _f2_291 = make_float2(e0_986, e1_987);
            float2 v2_988;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_988) : "l"(*(const unsigned long long*)&_f2_290), "l"(*(const unsigned long long*)&_f2_291));
            v_980[2] = v2_988.x;
            v_980[3] = v2_988.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_980[0]), "f"(v_980[1]), "f"(v_980[2]), "f"(v_980[3]) : "memory");
            float gs_989[4];
            float ks_990[4];
            float r_991[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_991[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_989[0] = r_991[0];
            unsigned int r_992[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_992[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_990[0] = __uint_as_float(r_992[0] << 16);
            float r_993[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_993[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_989[1] = r_993[0];
            unsigned int r_994[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_994[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_990[1] = __uint_as_float(r_994[0] << 16);
            float r_995[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_995[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_989[2] = r_995[0];
            unsigned int r_996[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_996[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_990[2] = __uint_as_float(r_996[0] << 16);
            float r_997[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_997[0])) : "r"((gbase_955 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_989[3] = r_997[0];
            unsigned int r_998[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_998[0])) : "r"((kbase_956 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_990[3] = __uint_as_float(r_998[0] << 16);
            float2 _f2_292 = make_float2(a0_958, a0_958);
            float v_999[4];
            float2 _f2_293 = make_float2(gs_989[0], gs_989[1]);
            float2 d2_1000 = sub_f32x2(_f2_292, _f2_293);
            float _exp2_104 = approx_exp2(d2_1000.x);
            float e0_1001 = _exp2_104;
            float _exp2_105 = approx_exp2(d2_1000.y);
            float e1_1002 = _exp2_105;
            float2 _f2_294 = make_float2(ks_990[0], ks_990[1]);
            float2 _f2_295 = make_float2(e0_1001, e1_1002);
            float2 v2_1003;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1003) : "l"(*(const unsigned long long*)&_f2_294), "l"(*(const unsigned long long*)&_f2_295));
            v_999[0] = v2_1003.x;
            v_999[1] = v2_1003.y;
            float2 _f2_296 = make_float2(gs_989[2], gs_989[3]);
            float2 d2_1004 = sub_f32x2(_f2_292, _f2_296);
            float _exp2_106 = approx_exp2(d2_1004.x);
            float e0_1005 = _exp2_106;
            float _exp2_107 = approx_exp2(d2_1004.y);
            float e1_1006 = _exp2_107;
            float2 _f2_297 = make_float2(ks_990[2], ks_990[3]);
            float2 _f2_298 = make_float2(e0_1005, e1_1006);
            float2 v2_1007;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1007) : "l"(*(const unsigned long long*)&_f2_297), "l"(*(const unsigned long long*)&_f2_298));
            v_999[2] = v2_1007.x;
            v_999[3] = v2_1007.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_999[0]), "f"(v_999[1]), "f"(v_999[2]), "f"(v_999[3]) : "memory");
            float2 _f2_299 = make_float2(a1_960, a1_960);
            float v_1008[4];
            float2 _f2_300 = make_float2(gs_989[0], gs_989[1]);
            float2 d2_1009 = sub_f32x2(_f2_299, _f2_300);
            float _exp2_108 = approx_exp2(d2_1009.x);
            float e0_1010 = _exp2_108;
            float _exp2_109 = approx_exp2(d2_1009.y);
            float e1_1011 = _exp2_109;
            float2 _f2_301 = make_float2(ks_990[0], ks_990[1]);
            float2 _f2_302 = make_float2(e0_1010, e1_1011);
            float2 v2_1012;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1012) : "l"(*(const unsigned long long*)&_f2_301), "l"(*(const unsigned long long*)&_f2_302));
            v_1008[0] = v2_1012.x;
            v_1008[1] = v2_1012.y;
            float2 _f2_303 = make_float2(gs_989[2], gs_989[3]);
            float2 d2_1013 = sub_f32x2(_f2_299, _f2_303);
            float _exp2_110 = approx_exp2(d2_1013.x);
            float e0_1014 = _exp2_110;
            float _exp2_111 = approx_exp2(d2_1013.y);
            float e1_1015 = _exp2_111;
            float2 _f2_304 = make_float2(ks_990[2], ks_990[3]);
            float2 _f2_305 = make_float2(e0_1014, e1_1015);
            float2 v2_1016;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1016) : "l"(*(const unsigned long long*)&_f2_304), "l"(*(const unsigned long long*)&_f2_305));
            v_1008[2] = v2_1016.x;
            v_1008[3] = v2_1016.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1008[0]), "f"(v_1008[1]), "f"(v_1008[2]), "f"(v_1008[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_1017 = 32 + 8 * th;
            int gbase_1018 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_1017 * 128);
            int kbase_1019 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_1017 * 128);
            float r_1020[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1020[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_1021 = r_1020[0];
            float a1_1022 = 0.0f;
            float gs_1023[4];
            float ks_1024[4];
            float r_1025[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1025[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1023[0] = r_1025[0];
            unsigned int r_1026[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1026[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1024[0] = __uint_as_float(r_1026[0] << 16);
            float r_1027[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1027[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1023[1] = r_1027[0];
            unsigned int r_1028[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1028[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1024[1] = __uint_as_float(r_1028[0] << 16);
            float r_1029[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1029[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1023[2] = r_1029[0];
            unsigned int r_1030[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1030[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1024[2] = __uint_as_float(r_1030[0] << 16);
            float r_1031[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1031[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1023[3] = r_1031[0];
            unsigned int r_1032[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1032[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1024[3] = __uint_as_float(r_1032[0] << 16);
            float2 _f2_306 = make_float2(a0_1021, a0_1021);
            float v_1033[4];
            float2 _f2_307 = make_float2(gs_1023[0], gs_1023[1]);
            float2 d2_1034 = sub_f32x2(_f2_306, _f2_307);
            float _exp2_112 = approx_exp2(d2_1034.x);
            float e0_1035 = _exp2_112;
            float _exp2_113 = approx_exp2(d2_1034.y);
            float e1_1036 = _exp2_113;
            float2 _f2_308 = make_float2(ks_1024[0], ks_1024[1]);
            float2 _f2_309 = make_float2(e0_1035, e1_1036);
            float2 v2_1037;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1037) : "l"(*(const unsigned long long*)&_f2_308), "l"(*(const unsigned long long*)&_f2_309));
            v_1033[0] = v2_1037.x;
            v_1033[1] = v2_1037.y;
            float2 _f2_310 = make_float2(gs_1023[2], gs_1023[3]);
            float2 d2_1038 = sub_f32x2(_f2_306, _f2_310);
            float _exp2_114 = approx_exp2(d2_1038.x);
            float e0_1039 = _exp2_114;
            float _exp2_115 = approx_exp2(d2_1038.y);
            float e1_1040 = _exp2_115;
            float2 _f2_311 = make_float2(ks_1024[2], ks_1024[3]);
            float2 _f2_312 = make_float2(e0_1039, e1_1040);
            float2 v2_1041;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1041) : "l"(*(const unsigned long long*)&_f2_311), "l"(*(const unsigned long long*)&_f2_312));
            v_1033[2] = v2_1041.x;
            v_1033[3] = v2_1041.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_1033[0]), "f"(v_1033[1]), "f"(v_1033[2]), "f"(v_1033[3]) : "memory");
            float gs_1042[4];
            float ks_1043[4];
            float r_1044[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1044[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1042[0] = r_1044[0];
            unsigned int r_1045[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1045[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1043[0] = __uint_as_float(r_1045[0] << 16);
            float r_1046[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1046[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1042[1] = r_1046[0];
            unsigned int r_1047[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1047[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1043[1] = __uint_as_float(r_1047[0] << 16);
            float r_1048[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1048[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1042[2] = r_1048[0];
            unsigned int r_1049[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1049[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1043[2] = __uint_as_float(r_1049[0] << 16);
            float r_1050[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1050[0])) : "r"((gbase_1018 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1042[3] = r_1050[0];
            unsigned int r_1051[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1051[0])) : "r"((kbase_1019 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1043[3] = __uint_as_float(r_1051[0] << 16);
            float2 _f2_313 = make_float2(a0_1021, a0_1021);
            float v_1052[4];
            float2 _f2_314 = make_float2(gs_1042[0], gs_1042[1]);
            float2 d2_1053 = sub_f32x2(_f2_313, _f2_314);
            float _exp2_116 = approx_exp2(d2_1053.x);
            float e0_1054 = _exp2_116;
            float _exp2_117 = approx_exp2(d2_1053.y);
            float e1_1055 = _exp2_117;
            float2 _f2_315 = make_float2(ks_1043[0], ks_1043[1]);
            float2 _f2_316 = make_float2(e0_1054, e1_1055);
            float2 v2_1056;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1056) : "l"(*(const unsigned long long*)&_f2_315), "l"(*(const unsigned long long*)&_f2_316));
            v_1052[0] = v2_1056.x;
            v_1052[1] = v2_1056.y;
            float2 _f2_317 = make_float2(gs_1042[2], gs_1042[3]);
            float2 d2_1057 = sub_f32x2(_f2_313, _f2_317);
            float _exp2_118 = approx_exp2(d2_1057.x);
            float e0_1058 = _exp2_118;
            float _exp2_119 = approx_exp2(d2_1057.y);
            float e1_1059 = _exp2_119;
            float2 _f2_318 = make_float2(ks_1043[2], ks_1043[3]);
            float2 _f2_319 = make_float2(e0_1058, e1_1059);
            float2 v2_1060;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1060) : "l"(*(const unsigned long long*)&_f2_318), "l"(*(const unsigned long long*)&_f2_319));
            v_1052[2] = v2_1060.x;
            v_1052[3] = v2_1060.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1052[0]), "f"(v_1052[1]), "f"(v_1052[2]), "f"(v_1052[3]) : "memory");
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
            float r_1061[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1061[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl = r_1061[0];
            float r_1062[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1062[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc = r_1062[0] - 21.640425324000002f;
            float dkt[8];
            float pan[8];
            float2 _f2_320 = make_float2(gc, gc);
            float2 _f2_321 = make_float2(gnl, gnl);
            int t0 = 32 + 8 * th;
            int t1 = 32 + 8 * th + 1;
            float r_1063[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1063[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0 = r_1063[0];
            float r_1064[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1064[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1 = r_1064[0];
            unsigned int r_1065[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1065[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0 = __uint_as_float(r_1065[0] << 16);
            unsigned int r_1066[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1066[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1 = __uint_as_float(r_1066[0] << 16);
            float2 _f2_322 = make_float2(g_t0, g_t1);
            float2 dc2 = sub_f32x2(_f2_320, _f2_322);
            float _exp2_120 = approx_exp2(dc2.x);
            float e2_0 = _exp2_120;
            float _exp2_121 = approx_exp2(dc2.y);
            float e2_1 = _exp2_121;
            float2 dn2 = sub_f32x2(_f2_321, _f2_322);
            float _exp2_122 = approx_exp2(dn2.x);
            float en_0 = _exp2_122;
            float _exp2_123 = approx_exp2(dn2.y);
            float en_1 = _exp2_123;
            dkt[0] = __fmaf_rn(_tmem_load_1[0], en_0, _tmem_load_0[0] * e2_0);
            dkt[1] = __fmaf_rn(_tmem_load_1[1], en_1, _tmem_load_0[1] * e2_1);
            float2 _f2_323 = make_float2(k_t0, k_t1);
            float2 _f2_324 = make_float2(e2_0, e2_1);
            float2 p2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2) : "l"(*(const unsigned long long*)&_f2_323), "l"(*(const unsigned long long*)&_f2_324));
            pan[0] = p2.x;
            pan[1] = p2.y;
            int t0_1067 = 32 + 8 * th + 2;
            int t1_1068 = 32 + 8 * th + 2 + 1;
            float r_1069[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1069[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1067 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1067 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1070 = r_1069[0];
            float r_1071[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1071[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1068 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1068 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1072 = r_1071[0];
            unsigned int r_1073[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1073[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1067 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1067 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1074 = __uint_as_float(r_1073[0] << 16);
            unsigned int r_1075[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1075[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1068 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1068 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1076 = __uint_as_float(r_1075[0] << 16);
            float2 _f2_325 = make_float2(g_t0_1070, g_t1_1072);
            float2 dc2_1077 = sub_f32x2(_f2_320, _f2_325);
            float _exp2_124 = approx_exp2(dc2_1077.x);
            float e2_0_1078 = _exp2_124;
            float _exp2_125 = approx_exp2(dc2_1077.y);
            float e2_1_1079 = _exp2_125;
            float2 dn2_1080 = sub_f32x2(_f2_321, _f2_325);
            float _exp2_126 = approx_exp2(dn2_1080.x);
            float en_0_1081 = _exp2_126;
            float _exp2_127 = approx_exp2(dn2_1080.y);
            float en_1_1082 = _exp2_127;
            dkt[2] = __fmaf_rn(_tmem_load_1[2], en_0_1081, _tmem_load_0[2] * e2_0_1078);
            dkt[3] = __fmaf_rn(_tmem_load_1[3], en_1_1082, _tmem_load_0[3] * e2_1_1079);
            float2 _f2_326 = make_float2(k_t0_1074, k_t1_1076);
            float2 _f2_327 = make_float2(e2_0_1078, e2_1_1079);
            float2 p2_1083;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1083) : "l"(*(const unsigned long long*)&_f2_326), "l"(*(const unsigned long long*)&_f2_327));
            pan[2] = p2_1083.x;
            pan[3] = p2_1083.y;
            int t0_1084 = 32 + 8 * th + 4;
            int t1_1085 = 32 + 8 * th + 4 + 1;
            float r_1086[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1086[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1084 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1084 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1087 = r_1086[0];
            float r_1088[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1088[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1085 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1085 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1089 = r_1088[0];
            unsigned int r_1090[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1090[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1084 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1084 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1091 = __uint_as_float(r_1090[0] << 16);
            unsigned int r_1092[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1092[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1085 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1085 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1093 = __uint_as_float(r_1092[0] << 16);
            float2 _f2_328 = make_float2(g_t0_1087, g_t1_1089);
            float2 dc2_1094 = sub_f32x2(_f2_320, _f2_328);
            float _exp2_128 = approx_exp2(dc2_1094.x);
            float e2_0_1095 = _exp2_128;
            float _exp2_129 = approx_exp2(dc2_1094.y);
            float e2_1_1096 = _exp2_129;
            float2 dn2_1097 = sub_f32x2(_f2_321, _f2_328);
            float _exp2_130 = approx_exp2(dn2_1097.x);
            float en_0_1098 = _exp2_130;
            float _exp2_131 = approx_exp2(dn2_1097.y);
            float en_1_1099 = _exp2_131;
            dkt[4] = __fmaf_rn(_tmem_load_1[4], en_0_1098, _tmem_load_0[4] * e2_0_1095);
            dkt[5] = __fmaf_rn(_tmem_load_1[5], en_1_1099, _tmem_load_0[5] * e2_1_1096);
            float2 _f2_329 = make_float2(k_t0_1091, k_t1_1093);
            float2 _f2_330 = make_float2(e2_0_1095, e2_1_1096);
            float2 p2_1100;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1100) : "l"(*(const unsigned long long*)&_f2_329), "l"(*(const unsigned long long*)&_f2_330));
            pan[4] = p2_1100.x;
            pan[5] = p2_1100.y;
            int t0_1101 = 32 + 8 * th + 6;
            int t1_1102 = 32 + 8 * th + 6 + 1;
            float r_1103[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1103[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1101 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1101 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1104 = r_1103[0];
            float r_1105[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1105[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1102 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1102 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1106 = r_1105[0];
            unsigned int r_1107[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1107[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1101 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1101 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1108 = __uint_as_float(r_1107[0] << 16);
            unsigned int r_1109[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1109[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1102 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1102 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1110 = __uint_as_float(r_1109[0] << 16);
            float2 _f2_331 = make_float2(g_t0_1104, g_t1_1106);
            float2 dc2_1111 = sub_f32x2(_f2_320, _f2_331);
            float _exp2_132 = approx_exp2(dc2_1111.x);
            float e2_0_1112 = _exp2_132;
            float _exp2_133 = approx_exp2(dc2_1111.y);
            float e2_1_1113 = _exp2_133;
            float2 dn2_1114 = sub_f32x2(_f2_321, _f2_331);
            float _exp2_134 = approx_exp2(dn2_1114.x);
            float en_0_1115 = _exp2_134;
            float _exp2_135 = approx_exp2(dn2_1114.y);
            float en_1_1116 = _exp2_135;
            dkt[6] = __fmaf_rn(_tmem_load_1[6], en_0_1115, _tmem_load_0[6] * e2_0_1112);
            dkt[7] = __fmaf_rn(_tmem_load_1[7], en_1_1116, _tmem_load_0[7] * e2_1_1113);
            float2 _f2_332 = make_float2(k_t0_1108, k_t1_1110);
            float2 _f2_333 = make_float2(e2_0_1112, e2_1_1113);
            float2 p2_1117;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1117) : "l"(*(const unsigned long long*)&_f2_332), "l"(*(const unsigned long long*)&_f2_333));
            pan[6] = p2_1117.x;
            pan[7] = p2_1117.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th), dkt);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan[0]), "f"(pan[1]), "f"(pan[2]), "f"(pan[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan[4]), "f"(pan[5]), "f"(pan[6]), "f"(pan[7]) : "memory");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (48 + 8 * th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_1118[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1118[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_1119 = r_1118[0];
            float r_1120[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1120[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_1121 = r_1120[0] - 21.640425324000002f;
            float dkt_1122[8];
            float pan_1123[8];
            float2 _f2_334 = make_float2(gc_1121, gc_1121);
            float2 _f2_335 = make_float2(gnl_1119, gnl_1119);
            int t0_1124 = 48 + 8 * th;
            int t1_1125 = 48 + 8 * th + 1;
            float r_1126[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1126[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1124 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1124 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1127 = r_1126[0];
            float r_1128[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1128[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1125 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1125 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1129 = r_1128[0];
            unsigned int r_1130[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1130[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1124 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1124 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1131 = __uint_as_float(r_1130[0] << 16);
            unsigned int r_1132[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1132[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1125 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1125 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1133 = __uint_as_float(r_1132[0] << 16);
            float2 _f2_336 = make_float2(g_t0_1127, g_t1_1129);
            float2 dc2_1134 = sub_f32x2(_f2_334, _f2_336);
            float _exp2_136 = approx_exp2(dc2_1134.x);
            float e2_0_1135 = _exp2_136;
            float _exp2_137 = approx_exp2(dc2_1134.y);
            float e2_1_1136 = _exp2_137;
            dkt_1122[0] = _tmem_load_2[0] * e2_0_1135;
            dkt_1122[1] = _tmem_load_2[1] * e2_1_1136;
            float2 _f2_337 = make_float2(k_t0_1131, k_t1_1133);
            float2 _f2_338 = make_float2(e2_0_1135, e2_1_1136);
            float2 p2_1137;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1137) : "l"(*(const unsigned long long*)&_f2_337), "l"(*(const unsigned long long*)&_f2_338));
            pan_1123[0] = p2_1137.x;
            pan_1123[1] = p2_1137.y;
            int t0_1138 = 48 + 8 * th + 2;
            int t1_1139 = 48 + 8 * th + 2 + 1;
            float r_1140[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1140[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1138 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1138 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1141 = r_1140[0];
            float r_1142[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1142[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1139 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1139 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1143 = r_1142[0];
            unsigned int r_1144[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1144[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1138 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1138 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1145 = __uint_as_float(r_1144[0] << 16);
            unsigned int r_1146[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1146[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1139 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1139 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1147 = __uint_as_float(r_1146[0] << 16);
            float2 _f2_339 = make_float2(g_t0_1141, g_t1_1143);
            float2 dc2_1148 = sub_f32x2(_f2_334, _f2_339);
            float _exp2_138 = approx_exp2(dc2_1148.x);
            float e2_0_1149 = _exp2_138;
            float _exp2_139 = approx_exp2(dc2_1148.y);
            float e2_1_1150 = _exp2_139;
            dkt_1122[2] = _tmem_load_2[2] * e2_0_1149;
            dkt_1122[3] = _tmem_load_2[3] * e2_1_1150;
            float2 _f2_340 = make_float2(k_t0_1145, k_t1_1147);
            float2 _f2_341 = make_float2(e2_0_1149, e2_1_1150);
            float2 p2_1151;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1151) : "l"(*(const unsigned long long*)&_f2_340), "l"(*(const unsigned long long*)&_f2_341));
            pan_1123[2] = p2_1151.x;
            pan_1123[3] = p2_1151.y;
            int t0_1152 = 48 + 8 * th + 4;
            int t1_1153 = 48 + 8 * th + 4 + 1;
            float r_1154[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1154[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1152 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1152 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1155 = r_1154[0];
            float r_1156[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1156[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1153 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1153 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1157 = r_1156[0];
            unsigned int r_1158[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1158[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1152 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1152 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1159 = __uint_as_float(r_1158[0] << 16);
            unsigned int r_1160[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1160[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1153 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1153 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1161 = __uint_as_float(r_1160[0] << 16);
            float2 _f2_342 = make_float2(g_t0_1155, g_t1_1157);
            float2 dc2_1162 = sub_f32x2(_f2_334, _f2_342);
            float _exp2_140 = approx_exp2(dc2_1162.x);
            float e2_0_1163 = _exp2_140;
            float _exp2_141 = approx_exp2(dc2_1162.y);
            float e2_1_1164 = _exp2_141;
            dkt_1122[4] = _tmem_load_2[4] * e2_0_1163;
            dkt_1122[5] = _tmem_load_2[5] * e2_1_1164;
            float2 _f2_343 = make_float2(k_t0_1159, k_t1_1161);
            float2 _f2_344 = make_float2(e2_0_1163, e2_1_1164);
            float2 p2_1165;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1165) : "l"(*(const unsigned long long*)&_f2_343), "l"(*(const unsigned long long*)&_f2_344));
            pan_1123[4] = p2_1165.x;
            pan_1123[5] = p2_1165.y;
            int t0_1166 = 48 + 8 * th + 6;
            int t1_1167 = 48 + 8 * th + 6 + 1;
            float r_1168[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1168[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1166 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1166 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1169 = r_1168[0];
            float r_1170[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1170[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1167 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1167 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1171 = r_1170[0];
            unsigned int r_1172[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1172[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1166 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1166 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1173 = __uint_as_float(r_1172[0] << 16);
            unsigned int r_1174[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1174[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1167 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1167 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1175 = __uint_as_float(r_1174[0] << 16);
            float2 _f2_345 = make_float2(g_t0_1169, g_t1_1171);
            float2 dc2_1176 = sub_f32x2(_f2_334, _f2_345);
            float _exp2_142 = approx_exp2(dc2_1176.x);
            float e2_0_1177 = _exp2_142;
            float _exp2_143 = approx_exp2(dc2_1176.y);
            float e2_1_1178 = _exp2_143;
            dkt_1122[6] = _tmem_load_2[6] * e2_0_1177;
            dkt_1122[7] = _tmem_load_2[7] * e2_1_1178;
            float2 _f2_346 = make_float2(k_t0_1173, k_t1_1175);
            float2 _f2_347 = make_float2(e2_0_1177, e2_1_1178);
            float2 p2_1179;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1179) : "l"(*(const unsigned long long*)&_f2_346), "l"(*(const unsigned long long*)&_f2_347));
            pan_1123[6] = p2_1179.x;
            pan_1123[7] = p2_1179.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th), dkt_1122);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1123[0]), "f"(pan_1123[1]), "f"(pan_1123[2]), "f"(pan_1123[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1123[4]), "f"(pan_1123[5]), "f"(pan_1123[6]), "f"(pan_1123[7]) : "memory");
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
            float r_1180[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1180[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_1181 = r_1180[0];
            float r_1182[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1182[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_1183 = r_1182[0] - 21.640425324000002f;
            float dkt_1184[8];
            float pan_1185[8];
            float2 _f2_348 = make_float2(gc_1183, gc_1183);
            float2 _f2_349 = make_float2(gnl_1181, gnl_1181);
            int t0_1186 = 8 * th;
            int t1_1187 = 8 * th + 1;
            float r_1188[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1188[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1186 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1186 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1189 = r_1188[0];
            float r_1190[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1190[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1187 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1187 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1191 = r_1190[0];
            unsigned int r_1192[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1192[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1186 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1186 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1193 = __uint_as_float(r_1192[0] << 16);
            unsigned int r_1194[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1194[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1187 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1187 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1195 = __uint_as_float(r_1194[0] << 16);
            float2 _f2_350 = make_float2(g_t0_1189, g_t1_1191);
            float2 dc2_1196 = sub_f32x2(_f2_348, _f2_350);
            float _exp2_144 = approx_exp2(dc2_1196.x);
            float e2_0_1197 = _exp2_144;
            float _exp2_145 = approx_exp2(dc2_1196.y);
            float e2_1_1198 = _exp2_145;
            float2 dn2_1199 = sub_f32x2(_f2_349, _f2_350);
            float _exp2_146 = approx_exp2(dn2_1199.x);
            float en_0_1200 = _exp2_146;
            float _exp2_147 = approx_exp2(dn2_1199.y);
            float en_1_1201 = _exp2_147;
            dkt_1184[0] = __fmaf_rn(_tmem_load_4[0], en_0_1200, _tmem_load_3[0] * e2_0_1197);
            dkt_1184[1] = __fmaf_rn(_tmem_load_4[1], en_1_1201, _tmem_load_3[1] * e2_1_1198);
            float2 _f2_351 = make_float2(k_t0_1193, k_t1_1195);
            float2 _f2_352 = make_float2(e2_0_1197, e2_1_1198);
            float2 p2_1202;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1202) : "l"(*(const unsigned long long*)&_f2_351), "l"(*(const unsigned long long*)&_f2_352));
            pan_1185[0] = p2_1202.x;
            pan_1185[1] = p2_1202.y;
            int t0_1203 = 8 * th + 2;
            int t1_1204 = 8 * th + 2 + 1;
            float r_1205[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1205[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1203 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1203 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1206 = r_1205[0];
            float r_1207[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1207[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1204 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1204 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1208 = r_1207[0];
            unsigned int r_1209[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1209[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1203 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1203 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1210 = __uint_as_float(r_1209[0] << 16);
            unsigned int r_1211[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1211[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1204 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1204 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1212 = __uint_as_float(r_1211[0] << 16);
            float2 _f2_353 = make_float2(g_t0_1206, g_t1_1208);
            float2 dc2_1213 = sub_f32x2(_f2_348, _f2_353);
            float _exp2_148 = approx_exp2(dc2_1213.x);
            float e2_0_1214 = _exp2_148;
            float _exp2_149 = approx_exp2(dc2_1213.y);
            float e2_1_1215 = _exp2_149;
            float2 dn2_1216 = sub_f32x2(_f2_349, _f2_353);
            float _exp2_150 = approx_exp2(dn2_1216.x);
            float en_0_1217 = _exp2_150;
            float _exp2_151 = approx_exp2(dn2_1216.y);
            float en_1_1218 = _exp2_151;
            dkt_1184[2] = __fmaf_rn(_tmem_load_4[2], en_0_1217, _tmem_load_3[2] * e2_0_1214);
            dkt_1184[3] = __fmaf_rn(_tmem_load_4[3], en_1_1218, _tmem_load_3[3] * e2_1_1215);
            float2 _f2_354 = make_float2(k_t0_1210, k_t1_1212);
            float2 _f2_355 = make_float2(e2_0_1214, e2_1_1215);
            float2 p2_1219;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1219) : "l"(*(const unsigned long long*)&_f2_354), "l"(*(const unsigned long long*)&_f2_355));
            pan_1185[2] = p2_1219.x;
            pan_1185[3] = p2_1219.y;
            int t0_1220 = 8 * th + 4;
            int t1_1221 = 8 * th + 4 + 1;
            float r_1222[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1222[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1220 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1220 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1223 = r_1222[0];
            float r_1224[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1224[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1221 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1221 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1225 = r_1224[0];
            unsigned int r_1226[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1226[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1220 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1220 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1227 = __uint_as_float(r_1226[0] << 16);
            unsigned int r_1228[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1228[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1221 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1221 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1229 = __uint_as_float(r_1228[0] << 16);
            float2 _f2_356 = make_float2(g_t0_1223, g_t1_1225);
            float2 dc2_1230 = sub_f32x2(_f2_348, _f2_356);
            float _exp2_152 = approx_exp2(dc2_1230.x);
            float e2_0_1231 = _exp2_152;
            float _exp2_153 = approx_exp2(dc2_1230.y);
            float e2_1_1232 = _exp2_153;
            float2 dn2_1233 = sub_f32x2(_f2_349, _f2_356);
            float _exp2_154 = approx_exp2(dn2_1233.x);
            float en_0_1234 = _exp2_154;
            float _exp2_155 = approx_exp2(dn2_1233.y);
            float en_1_1235 = _exp2_155;
            dkt_1184[4] = __fmaf_rn(_tmem_load_4[4], en_0_1234, _tmem_load_3[4] * e2_0_1231);
            dkt_1184[5] = __fmaf_rn(_tmem_load_4[5], en_1_1235, _tmem_load_3[5] * e2_1_1232);
            float2 _f2_357 = make_float2(k_t0_1227, k_t1_1229);
            float2 _f2_358 = make_float2(e2_0_1231, e2_1_1232);
            float2 p2_1236;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1236) : "l"(*(const unsigned long long*)&_f2_357), "l"(*(const unsigned long long*)&_f2_358));
            pan_1185[4] = p2_1236.x;
            pan_1185[5] = p2_1236.y;
            int t0_1237 = 8 * th + 6;
            int t1_1238 = 8 * th + 6 + 1;
            float r_1239[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1239[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1237 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1237 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1240 = r_1239[0];
            float r_1241[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1241[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1238 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1238 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1242 = r_1241[0];
            unsigned int r_1243[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1243[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1237 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1237 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1244 = __uint_as_float(r_1243[0] << 16);
            unsigned int r_1245[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1245[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1238 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1238 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1246 = __uint_as_float(r_1245[0] << 16);
            float2 _f2_359 = make_float2(g_t0_1240, g_t1_1242);
            float2 dc2_1247 = sub_f32x2(_f2_348, _f2_359);
            float _exp2_156 = approx_exp2(dc2_1247.x);
            float e2_0_1248 = _exp2_156;
            float _exp2_157 = approx_exp2(dc2_1247.y);
            float e2_1_1249 = _exp2_157;
            float2 dn2_1250 = sub_f32x2(_f2_349, _f2_359);
            float _exp2_158 = approx_exp2(dn2_1250.x);
            float en_0_1251 = _exp2_158;
            float _exp2_159 = approx_exp2(dn2_1250.y);
            float en_1_1252 = _exp2_159;
            dkt_1184[6] = __fmaf_rn(_tmem_load_4[6], en_0_1251, _tmem_load_3[6] * e2_0_1248);
            dkt_1184[7] = __fmaf_rn(_tmem_load_4[7], en_1_1252, _tmem_load_3[7] * e2_1_1249);
            float2 _f2_360 = make_float2(k_t0_1244, k_t1_1246);
            float2 _f2_361 = make_float2(e2_0_1248, e2_1_1249);
            float2 p2_1253;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1253) : "l"(*(const unsigned long long*)&_f2_360), "l"(*(const unsigned long long*)&_f2_361));
            pan_1185[6] = p2_1253.x;
            pan_1185[7] = p2_1253.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(8 * th), dkt_1184);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1185[0]), "f"(pan_1185[1]), "f"(pan_1185[2]), "f"(pan_1185[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1185[4]), "f"(pan_1185[5]), "f"(pan_1185[6]), "f"(pan_1185[7]) : "memory");
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (16 + 8 * th)));
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_1254[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1254[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_1255 = r_1254[0];
            float r_1256[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1256[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_1257 = r_1256[0] - 21.640425324000002f;
            float dkt_1258[8];
            float pan_1259[8];
            float2 _f2_362 = make_float2(gc_1257, gc_1257);
            float2 _f2_363 = make_float2(gnl_1255, gnl_1255);
            int t0_1260 = 16 + 8 * th;
            int t1_1261 = 16 + 8 * th + 1;
            float r_1262[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1262[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1260 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1260 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1263 = r_1262[0];
            float r_1264[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1264[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1261 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1261 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1265 = r_1264[0];
            unsigned int r_1266[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1266[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1260 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1260 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1267 = __uint_as_float(r_1266[0] << 16);
            unsigned int r_1268[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1268[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1261 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1261 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1269 = __uint_as_float(r_1268[0] << 16);
            float2 _f2_364 = make_float2(g_t0_1263, g_t1_1265);
            float2 dc2_1270 = sub_f32x2(_f2_362, _f2_364);
            float _exp2_160 = approx_exp2(dc2_1270.x);
            float e2_0_1271 = _exp2_160;
            float _exp2_161 = approx_exp2(dc2_1270.y);
            float e2_1_1272 = _exp2_161;
            float2 dn2_1273 = sub_f32x2(_f2_363, _f2_364);
            float _exp2_162 = approx_exp2(dn2_1273.x);
            float en_0_1274 = _exp2_162;
            float _exp2_163 = approx_exp2(dn2_1273.y);
            float en_1_1275 = _exp2_163;
            dkt_1258[0] = __fmaf_rn(_tmem_load_6[0], en_0_1274, _tmem_load_5[0] * e2_0_1271);
            dkt_1258[1] = __fmaf_rn(_tmem_load_6[1], en_1_1275, _tmem_load_5[1] * e2_1_1272);
            float2 _f2_365 = make_float2(k_t0_1267, k_t1_1269);
            float2 _f2_366 = make_float2(e2_0_1271, e2_1_1272);
            float2 p2_1276;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1276) : "l"(*(const unsigned long long*)&_f2_365), "l"(*(const unsigned long long*)&_f2_366));
            pan_1259[0] = p2_1276.x;
            pan_1259[1] = p2_1276.y;
            int t0_1277 = 16 + 8 * th + 2;
            int t1_1278 = 16 + 8 * th + 2 + 1;
            float r_1279[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1279[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1277 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1277 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1280 = r_1279[0];
            float r_1281[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1281[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1278 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1278 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1282 = r_1281[0];
            unsigned int r_1283[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1283[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1277 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1277 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1284 = __uint_as_float(r_1283[0] << 16);
            unsigned int r_1285[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1285[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1278 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1278 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1286 = __uint_as_float(r_1285[0] << 16);
            float2 _f2_367 = make_float2(g_t0_1280, g_t1_1282);
            float2 dc2_1287 = sub_f32x2(_f2_362, _f2_367);
            float _exp2_164 = approx_exp2(dc2_1287.x);
            float e2_0_1288 = _exp2_164;
            float _exp2_165 = approx_exp2(dc2_1287.y);
            float e2_1_1289 = _exp2_165;
            float2 dn2_1290 = sub_f32x2(_f2_363, _f2_367);
            float _exp2_166 = approx_exp2(dn2_1290.x);
            float en_0_1291 = _exp2_166;
            float _exp2_167 = approx_exp2(dn2_1290.y);
            float en_1_1292 = _exp2_167;
            dkt_1258[2] = __fmaf_rn(_tmem_load_6[2], en_0_1291, _tmem_load_5[2] * e2_0_1288);
            dkt_1258[3] = __fmaf_rn(_tmem_load_6[3], en_1_1292, _tmem_load_5[3] * e2_1_1289);
            float2 _f2_368 = make_float2(k_t0_1284, k_t1_1286);
            float2 _f2_369 = make_float2(e2_0_1288, e2_1_1289);
            float2 p2_1293;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1293) : "l"(*(const unsigned long long*)&_f2_368), "l"(*(const unsigned long long*)&_f2_369));
            pan_1259[2] = p2_1293.x;
            pan_1259[3] = p2_1293.y;
            int t0_1294 = 16 + 8 * th + 4;
            int t1_1295 = 16 + 8 * th + 4 + 1;
            float r_1296[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1296[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1294 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1294 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1297 = r_1296[0];
            float r_1298[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1298[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1295 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1295 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1299 = r_1298[0];
            unsigned int r_1300[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1300[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1294 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1294 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1301 = __uint_as_float(r_1300[0] << 16);
            unsigned int r_1302[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1302[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1295 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1295 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1303 = __uint_as_float(r_1302[0] << 16);
            float2 _f2_370 = make_float2(g_t0_1297, g_t1_1299);
            float2 dc2_1304 = sub_f32x2(_f2_362, _f2_370);
            float _exp2_168 = approx_exp2(dc2_1304.x);
            float e2_0_1305 = _exp2_168;
            float _exp2_169 = approx_exp2(dc2_1304.y);
            float e2_1_1306 = _exp2_169;
            float2 dn2_1307 = sub_f32x2(_f2_363, _f2_370);
            float _exp2_170 = approx_exp2(dn2_1307.x);
            float en_0_1308 = _exp2_170;
            float _exp2_171 = approx_exp2(dn2_1307.y);
            float en_1_1309 = _exp2_171;
            dkt_1258[4] = __fmaf_rn(_tmem_load_6[4], en_0_1308, _tmem_load_5[4] * e2_0_1305);
            dkt_1258[5] = __fmaf_rn(_tmem_load_6[5], en_1_1309, _tmem_load_5[5] * e2_1_1306);
            float2 _f2_371 = make_float2(k_t0_1301, k_t1_1303);
            float2 _f2_372 = make_float2(e2_0_1305, e2_1_1306);
            float2 p2_1310;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1310) : "l"(*(const unsigned long long*)&_f2_371), "l"(*(const unsigned long long*)&_f2_372));
            pan_1259[4] = p2_1310.x;
            pan_1259[5] = p2_1310.y;
            int t0_1311 = 16 + 8 * th + 6;
            int t1_1312 = 16 + 8 * th + 6 + 1;
            float r_1313[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1313[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_1311 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_1311 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_1314 = r_1313[0];
            float r_1315[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1315[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_1312 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_1312 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_1316 = r_1315[0];
            unsigned int r_1317[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1317[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_1311 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_1311 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_1318 = __uint_as_float(r_1317[0] << 16);
            unsigned int r_1319[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1319[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_1312 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_1312 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_1320 = __uint_as_float(r_1319[0] << 16);
            float2 _f2_373 = make_float2(g_t0_1314, g_t1_1316);
            float2 dc2_1321 = sub_f32x2(_f2_362, _f2_373);
            float _exp2_172 = approx_exp2(dc2_1321.x);
            float e2_0_1322 = _exp2_172;
            float _exp2_173 = approx_exp2(dc2_1321.y);
            float e2_1_1323 = _exp2_173;
            float2 dn2_1324 = sub_f32x2(_f2_363, _f2_373);
            float _exp2_174 = approx_exp2(dn2_1324.x);
            float en_0_1325 = _exp2_174;
            float _exp2_175 = approx_exp2(dn2_1324.y);
            float en_1_1326 = _exp2_175;
            dkt_1258[6] = __fmaf_rn(_tmem_load_6[6], en_0_1325, _tmem_load_5[6] * e2_0_1322);
            dkt_1258[7] = __fmaf_rn(_tmem_load_6[7], en_1_1326, _tmem_load_5[7] * e2_1_1323);
            float2 _f2_374 = make_float2(k_t0_1318, k_t1_1320);
            float2 _f2_375 = make_float2(e2_0_1322, e2_1_1323);
            float2 p2_1327;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_1327) : "l"(*(const unsigned long long*)&_f2_374), "l"(*(const unsigned long long*)&_f2_375));
            pan_1259[6] = p2_1327.x;
            pan_1259[7] = p2_1327.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th), dkt_1258);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1259[0]), "f"(pan_1259[1]), "f"(pan_1259[2]), "f"(pan_1259[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_1259[4]), "f"(pan_1259[5]), "f"(pan_1259[6]), "f"(pan_1259[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_1328 = 8 * th;
            int gbase_1329 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_1328 * 128);
            int kbase_1330 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_1328 * 128);
            float r_1331[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1331[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_1332 = r_1331[0];
            float a1_1333 = 0.0f;
            float gs_1334[4];
            float ks_1335[4];
            float r_1336[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1336[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1334[0] = r_1336[0];
            unsigned int r_1337[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1337[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1335[0] = __uint_as_float(r_1337[0] << 16);
            float r_1338[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1338[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1334[1] = r_1338[0];
            unsigned int r_1339[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1339[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1335[1] = __uint_as_float(r_1339[0] << 16);
            float r_1340[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1340[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1334[2] = r_1340[0];
            unsigned int r_1341[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1341[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1335[2] = __uint_as_float(r_1341[0] << 16);
            float r_1342[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1342[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1334[3] = r_1342[0];
            unsigned int r_1343[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1343[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1335[3] = __uint_as_float(r_1343[0] << 16);
            float2 _f2_376 = make_float2(a0_1332, a0_1332);
            float v_1344[4];
            float2 _f2_377 = make_float2(gs_1334[0], gs_1334[1]);
            float2 d2_1345 = sub_f32x2(_f2_376, _f2_377);
            float _exp2_176 = approx_exp2(d2_1345.x);
            float e0_1346 = _exp2_176;
            float _exp2_177 = approx_exp2(d2_1345.y);
            float e1_1347 = _exp2_177;
            float2 _f2_378 = make_float2(ks_1335[0], ks_1335[1]);
            float2 _f2_379 = make_float2(e0_1346, e1_1347);
            float2 v2_1348;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1348) : "l"(*(const unsigned long long*)&_f2_378), "l"(*(const unsigned long long*)&_f2_379));
            v_1344[0] = v2_1348.x;
            v_1344[1] = v2_1348.y;
            float2 _f2_380 = make_float2(gs_1334[2], gs_1334[3]);
            float2 d2_1349 = sub_f32x2(_f2_376, _f2_380);
            float _exp2_178 = approx_exp2(d2_1349.x);
            float e0_1350 = _exp2_178;
            float _exp2_179 = approx_exp2(d2_1349.y);
            float e1_1351 = _exp2_179;
            float2 _f2_381 = make_float2(ks_1335[2], ks_1335[3]);
            float2 _f2_382 = make_float2(e0_1350, e1_1351);
            float2 v2_1352;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1352) : "l"(*(const unsigned long long*)&_f2_381), "l"(*(const unsigned long long*)&_f2_382));
            v_1344[2] = v2_1352.x;
            v_1344[3] = v2_1352.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_1344[0]), "f"(v_1344[1]), "f"(v_1344[2]), "f"(v_1344[3]) : "memory");
            float gs_1353[4];
            float ks_1354[4];
            float r_1355[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1355[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1353[0] = r_1355[0];
            unsigned int r_1356[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1356[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1354[0] = __uint_as_float(r_1356[0] << 16);
            float r_1357[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1357[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1353[1] = r_1357[0];
            unsigned int r_1358[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1358[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1354[1] = __uint_as_float(r_1358[0] << 16);
            float r_1359[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1359[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1353[2] = r_1359[0];
            unsigned int r_1360[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1360[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1354[2] = __uint_as_float(r_1360[0] << 16);
            float r_1361[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1361[0])) : "r"((gbase_1329 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1353[3] = r_1361[0];
            unsigned int r_1362[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1362[0])) : "r"((kbase_1330 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1354[3] = __uint_as_float(r_1362[0] << 16);
            float2 _f2_383 = make_float2(a0_1332, a0_1332);
            float v_1363[4];
            float2 _f2_384 = make_float2(gs_1353[0], gs_1353[1]);
            float2 d2_1364 = sub_f32x2(_f2_383, _f2_384);
            float _exp2_180 = approx_exp2(d2_1364.x);
            float e0_1365 = _exp2_180;
            float _exp2_181 = approx_exp2(d2_1364.y);
            float e1_1366 = _exp2_181;
            float2 _f2_385 = make_float2(ks_1354[0], ks_1354[1]);
            float2 _f2_386 = make_float2(e0_1365, e1_1366);
            float2 v2_1367;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1367) : "l"(*(const unsigned long long*)&_f2_385), "l"(*(const unsigned long long*)&_f2_386));
            v_1363[0] = v2_1367.x;
            v_1363[1] = v2_1367.y;
            float2 _f2_387 = make_float2(gs_1353[2], gs_1353[3]);
            float2 d2_1368 = sub_f32x2(_f2_383, _f2_387);
            float _exp2_182 = approx_exp2(d2_1368.x);
            float e0_1369 = _exp2_182;
            float _exp2_183 = approx_exp2(d2_1368.y);
            float e1_1370 = _exp2_183;
            float2 _f2_388 = make_float2(ks_1354[2], ks_1354[3]);
            float2 _f2_389 = make_float2(e0_1369, e1_1370);
            float2 v2_1371;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_1371) : "l"(*(const unsigned long long*)&_f2_388), "l"(*(const unsigned long long*)&_f2_389));
            v_1363[2] = v2_1371.x;
            v_1363[3] = v2_1371.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1363[0]), "f"(v_1363[1]), "f"(v_1363[2]), "f"(v_1363[3]) : "memory");
            mbarrier_wait(blk_full_addr, 0);
            mbarrier_wait(blk_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                int kk_12 = w * 4;
                if (lane_1 > (kk_12 & 15)) {
                    {
                        uint32_t _addr_16 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_12 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_12 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(0.0f) : "memory");
                    }
                }
                int kk_0_10 = w * 4 + 1;
                if (lane_1 > (kk_0_10 & 15)) {
                    {
                        uint32_t _addr_17 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(0.0f) : "memory");
                    }
                }
                int kk_1_1 = w * 4 + 2;
                if (lane_1 > (kk_1_1 & 15)) {
                    {
                        uint32_t _addr_18 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(0.0f) : "memory");
                    }
                }
                int kk_2_10 = w * 4 + 3;
                if (lane_1 > (kk_2_10 & 15)) {
                    {
                        uint32_t _addr_19 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(0.0f) : "memory");
                    }
                }
            }
            if (lane_1 < 16) {
                int kk_13 = w * 4;
                if (lane_1 > (kk_13 & 15)) {
                    {
                        uint32_t _addr_20 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_13 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_13 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(0.0f) : "memory");
                    }
                }
                int kk_0_11 = w * 4 + 1;
                if (lane_1 > (kk_0_11 & 15)) {
                    {
                        uint32_t _addr_21 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_11 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_11 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(0.0f) : "memory");
                    }
                }
                int kk_1_2 = w * 4 + 2;
                if (lane_1 > (kk_1_2 & 15)) {
                    {
                        uint32_t _addr_22 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_1_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_1_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(0.0f) : "memory");
                    }
                }
                int kk_2_11 = w * 4 + 3;
                if (lane_1 > (kk_2_11 & 15)) {
                    {
                        uint32_t _addr_23 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_11 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_11 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(0.0f) : "memory");
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
            float r_1372[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1372[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn = r_1372[0];
            float gc_1373 = gn - 21.640425324000002f;
            int eb = 98304 + th * -32768;
            int ew = eb >> 2;
            int qbase_1374 = pool_f32_addr + (unsigned int)(eb + 12288 + dh_lane * 1024);
            int sb8 = 32 + 8 * th;
            int gb8 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8 * 128);
            int kb8 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8 * 128);
            float dbp[8];
            float _tmem_load_7[4];
            tmem_ld_x4(&_tmem_load_7[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            float _tmem_load_8[4];
            tmem_ld_x4(&_tmem_load_8[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th));
            float _tmem_load_9[4];
            tmem_ld_x4(&_tmem_load_9[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th));
            float _tmem_load_10[4];
            tmem_ld_x4(&_tmem_load_10[0], taddr + (unsigned int)row_base + (unsigned int)(208 + 8 * th));
            float _tmem_load_11[4];
            tmem_ld_x4(&_tmem_load_11[0], taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th));
            float btv[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (32 + 8 * th))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t = 32 + 8 * th;
            float r_1375[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1375[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t = r_1375[0];
            unsigned int r_1376[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1376[0])) : "r"((kb8 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t = __uint_as_float(r_1376[0] << 16);
            unsigned int r_1377[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1377[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t = __uint_as_float(r_1377[0] << 16);
            float q_t_1378 = ((t >= clen) ? 0.0f : q_t);
            float dqf = pool_f32[ew + d_lane];
            float dkf = pool_f32[ew + 1024 + d_lane];
            float dgf = pool_f32[ew + 2048 + d_lane];
            float bt_t = btv[0];
            float _exp2_184 = approx_exp2(g_t - gn);
            float e_off = _exp2_184;
            float _exp2_185 = approx_exp2(g_t - gc_1373);
            float e_dg = _exp2_185;
            float dq2 = __fmaf_rn(_tmem_load_7[0], e_off, _tmem_load_9[0] * e_dg);
            float dk2 = __fmaf_rn(_tmem_load_8[0], e_off, _tmem_load_10[0] * e_dg);
            dbp[0] = dk2 * k_t;
            float dk2b = dk2 * bt_t;
            float dq_o = dq2 + dqf;
            float dk_o = dk2b + dkf + _tmem_load_11[0];
            float dg_o = __fmaf_rn(q_t_1378, dq2, (dk2b - _tmem_load_11[0]) * k_t) + dgf;
            int goff = t * rs_g;
            int qoff = t * qrs;
            *(reinterpret_cast<float*>(dg_base + goff) + (0)) = dg_o;
            if (t < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff) + (0)) = __float2bfloat16_rn(dq_o);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff) + (0)) = __float2bfloat16_rn(dk_o);
            }
            int t_1379 = 32 + 8 * th + 1;
            float r_1380[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1380[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1381 = r_1380[0];
            unsigned int r_1382[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1382[0])) : "r"((kb8 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1383 = __uint_as_float(r_1382[0] << 16);
            unsigned int r_1384[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1384[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1385 = __uint_as_float(r_1384[0] << 16);
            float q_t_1386 = ((t_1379 >= clen) ? 0.0f : q_t_1385);
            float dqf_1387 = pool_f32[ew + 128 + d_lane];
            float dkf_1388 = pool_f32[ew + 1152 + d_lane];
            float dgf_1389 = pool_f32[ew + 2176 + d_lane];
            float bt_t_1390 = btv[1];
            float _exp2_186 = approx_exp2(g_t_1381 - gn);
            float e_off_1391 = _exp2_186;
            float _exp2_187 = approx_exp2(g_t_1381 - gc_1373);
            float e_dg_1392 = _exp2_187;
            float dq2_1393 = __fmaf_rn(_tmem_load_7[1], e_off_1391, _tmem_load_9[1] * e_dg_1392);
            float dk2_1394 = __fmaf_rn(_tmem_load_8[1], e_off_1391, _tmem_load_10[1] * e_dg_1392);
            dbp[1] = dk2_1394 * k_t_1383;
            float dk2b_1395 = dk2_1394 * bt_t_1390;
            float dq_o_1396 = dq2_1393 + dqf_1387;
            float dk_o_1397 = dk2b_1395 + dkf_1388 + _tmem_load_11[1];
            float dg_o_1398 = __fmaf_rn(q_t_1386, dq2_1393, (dk2b_1395 - _tmem_load_11[1]) * k_t_1383) + dgf_1389;
            int goff_1399 = t_1379 * rs_g;
            int qoff_1400 = t_1379 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1399) + (0)) = dg_o_1398;
            if (t_1379 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1400) + (0)) = __float2bfloat16_rn(dq_o_1396);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1400) + (0)) = __float2bfloat16_rn(dk_o_1397);
            }
            int t_1401 = 32 + 8 * th + 2;
            float r_1402[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1402[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1403 = r_1402[0];
            unsigned int r_1404[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1404[0])) : "r"((kb8 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1405 = __uint_as_float(r_1404[0] << 16);
            unsigned int r_1406[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1406[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1407 = __uint_as_float(r_1406[0] << 16);
            float q_t_1408 = ((t_1401 >= clen) ? 0.0f : q_t_1407);
            float dqf_1409 = pool_f32[ew + 256 + d_lane];
            float dkf_1410 = pool_f32[ew + 1280 + d_lane];
            float dgf_1411 = pool_f32[ew + 2304 + d_lane];
            float bt_t_1412 = btv[2];
            float _exp2_188 = approx_exp2(g_t_1403 - gn);
            float e_off_1413 = _exp2_188;
            float _exp2_189 = approx_exp2(g_t_1403 - gc_1373);
            float e_dg_1414 = _exp2_189;
            float dq2_1415 = __fmaf_rn(_tmem_load_7[2], e_off_1413, _tmem_load_9[2] * e_dg_1414);
            float dk2_1416 = __fmaf_rn(_tmem_load_8[2], e_off_1413, _tmem_load_10[2] * e_dg_1414);
            dbp[2] = dk2_1416 * k_t_1405;
            float dk2b_1417 = dk2_1416 * bt_t_1412;
            float dq_o_1418 = dq2_1415 + dqf_1409;
            float dk_o_1419 = dk2b_1417 + dkf_1410 + _tmem_load_11[2];
            float dg_o_1420 = __fmaf_rn(q_t_1408, dq2_1415, (dk2b_1417 - _tmem_load_11[2]) * k_t_1405) + dgf_1411;
            int goff_1421 = t_1401 * rs_g;
            int qoff_1422 = t_1401 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1421) + (0)) = dg_o_1420;
            if (t_1401 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1422) + (0)) = __float2bfloat16_rn(dq_o_1418);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1422) + (0)) = __float2bfloat16_rn(dk_o_1419);
            }
            int t_1423 = 32 + 8 * th + 3;
            float r_1424[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1424[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1425 = r_1424[0];
            unsigned int r_1426[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1426[0])) : "r"((kb8 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1427 = __uint_as_float(r_1426[0] << 16);
            unsigned int r_1428[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1428[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1429 = __uint_as_float(r_1428[0] << 16);
            float q_t_1430 = ((t_1423 >= clen) ? 0.0f : q_t_1429);
            float dqf_1431 = pool_f32[ew + 384 + d_lane];
            float dkf_1432 = pool_f32[ew + 1408 + d_lane];
            float dgf_1433 = pool_f32[ew + 2432 + d_lane];
            float bt_t_1434 = btv[3];
            float _exp2_190 = approx_exp2(g_t_1425 - gn);
            float e_off_1435 = _exp2_190;
            float _exp2_191 = approx_exp2(g_t_1425 - gc_1373);
            float e_dg_1436 = _exp2_191;
            float dq2_1437 = __fmaf_rn(_tmem_load_7[3], e_off_1435, _tmem_load_9[3] * e_dg_1436);
            float dk2_1438 = __fmaf_rn(_tmem_load_8[3], e_off_1435, _tmem_load_10[3] * e_dg_1436);
            dbp[3] = dk2_1438 * k_t_1427;
            float dk2b_1439 = dk2_1438 * bt_t_1434;
            float dq_o_1440 = dq2_1437 + dqf_1431;
            float dk_o_1441 = dk2b_1439 + dkf_1432 + _tmem_load_11[3];
            float dg_o_1442 = __fmaf_rn(q_t_1430, dq2_1437, (dk2b_1439 - _tmem_load_11[3]) * k_t_1427) + dgf_1433;
            int goff_1443 = t_1423 * rs_g;
            int qoff_1444 = t_1423 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1443) + (0)) = dg_o_1442;
            if (t_1423 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1444) + (0)) = __float2bfloat16_rn(dq_o_1440);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1444) + (0)) = __float2bfloat16_rn(dk_o_1441);
            }
            float _tmem_load_12[4];
            tmem_ld_x4(&_tmem_load_12[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 4));
            float _tmem_load_13[4];
            tmem_ld_x4(&_tmem_load_13[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th + 4));
            float _tmem_load_14[4];
            tmem_ld_x4(&_tmem_load_14[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 4));
            float _tmem_load_15[4];
            tmem_ld_x4(&_tmem_load_15[0], taddr + (unsigned int)row_base + (unsigned int)(208 + 8 * th + 4));
            float _tmem_load_16[4];
            tmem_ld_x4(&_tmem_load_16[0], taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th + 4));
            float btv_1445[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1445[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1445[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1445[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1445[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (32 + 8 * th + 4))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1446 = 32 + 8 * th + 4;
            float r_1447[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1447[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1448 = r_1447[0];
            unsigned int r_1449[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1449[0])) : "r"((kb8 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1450 = __uint_as_float(r_1449[0] << 16);
            unsigned int r_1451[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1451[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1452 = __uint_as_float(r_1451[0] << 16);
            float q_t_1453 = ((t_1446 >= clen) ? 0.0f : q_t_1452);
            float dqf_1454 = pool_f32[ew + 512 + d_lane];
            float dkf_1455 = pool_f32[ew + 1536 + d_lane];
            float dgf_1456 = pool_f32[ew + 2560 + d_lane];
            float bt_t_1457 = btv_1445[0];
            float _exp2_192 = approx_exp2(g_t_1448 - gn);
            float e_off_1458 = _exp2_192;
            float _exp2_193 = approx_exp2(g_t_1448 - gc_1373);
            float e_dg_1459 = _exp2_193;
            float dq2_1460 = __fmaf_rn(_tmem_load_12[0], e_off_1458, _tmem_load_14[0] * e_dg_1459);
            float dk2_1461 = __fmaf_rn(_tmem_load_13[0], e_off_1458, _tmem_load_15[0] * e_dg_1459);
            dbp[4] = dk2_1461 * k_t_1450;
            float dk2b_1462 = dk2_1461 * bt_t_1457;
            float dq_o_1463 = dq2_1460 + dqf_1454;
            float dk_o_1464 = dk2b_1462 + dkf_1455 + _tmem_load_16[0];
            float dg_o_1465 = __fmaf_rn(q_t_1453, dq2_1460, (dk2b_1462 - _tmem_load_16[0]) * k_t_1450) + dgf_1456;
            int goff_1466 = t_1446 * rs_g;
            int qoff_1467 = t_1446 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1466) + (0)) = dg_o_1465;
            if (t_1446 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1467) + (0)) = __float2bfloat16_rn(dq_o_1463);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1467) + (0)) = __float2bfloat16_rn(dk_o_1464);
            }
            int t_1468 = 32 + 8 * th + 4 + 1;
            float r_1469[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1469[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1470 = r_1469[0];
            unsigned int r_1471[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1471[0])) : "r"((kb8 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1472 = __uint_as_float(r_1471[0] << 16);
            unsigned int r_1473[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1473[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1474 = __uint_as_float(r_1473[0] << 16);
            float q_t_1475 = ((t_1468 >= clen) ? 0.0f : q_t_1474);
            float dqf_1476 = pool_f32[ew + 640 + d_lane];
            float dkf_1477 = pool_f32[ew + 1664 + d_lane];
            float dgf_1478 = pool_f32[ew + 2688 + d_lane];
            float bt_t_1479 = btv_1445[1];
            float _exp2_194 = approx_exp2(g_t_1470 - gn);
            float e_off_1480 = _exp2_194;
            float _exp2_195 = approx_exp2(g_t_1470 - gc_1373);
            float e_dg_1481 = _exp2_195;
            float dq2_1482 = __fmaf_rn(_tmem_load_12[1], e_off_1480, _tmem_load_14[1] * e_dg_1481);
            float dk2_1483 = __fmaf_rn(_tmem_load_13[1], e_off_1480, _tmem_load_15[1] * e_dg_1481);
            dbp[5] = dk2_1483 * k_t_1472;
            float dk2b_1484 = dk2_1483 * bt_t_1479;
            float dq_o_1485 = dq2_1482 + dqf_1476;
            float dk_o_1486 = dk2b_1484 + dkf_1477 + _tmem_load_16[1];
            float dg_o_1487 = __fmaf_rn(q_t_1475, dq2_1482, (dk2b_1484 - _tmem_load_16[1]) * k_t_1472) + dgf_1478;
            int goff_1488 = t_1468 * rs_g;
            int qoff_1489 = t_1468 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1488) + (0)) = dg_o_1487;
            if (t_1468 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1489) + (0)) = __float2bfloat16_rn(dq_o_1485);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1489) + (0)) = __float2bfloat16_rn(dk_o_1486);
            }
            int t_1490 = 32 + 8 * th + 4 + 2;
            float r_1491[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1491[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1492 = r_1491[0];
            unsigned int r_1493[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1493[0])) : "r"((kb8 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1494 = __uint_as_float(r_1493[0] << 16);
            unsigned int r_1495[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1495[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1496 = __uint_as_float(r_1495[0] << 16);
            float q_t_1497 = ((t_1490 >= clen) ? 0.0f : q_t_1496);
            float dqf_1498 = pool_f32[ew + 768 + d_lane];
            float dkf_1499 = pool_f32[ew + 1792 + d_lane];
            float dgf_1500 = pool_f32[ew + 2816 + d_lane];
            float bt_t_1501 = btv_1445[2];
            float _exp2_196 = approx_exp2(g_t_1492 - gn);
            float e_off_1502 = _exp2_196;
            float _exp2_197 = approx_exp2(g_t_1492 - gc_1373);
            float e_dg_1503 = _exp2_197;
            float dq2_1504 = __fmaf_rn(_tmem_load_12[2], e_off_1502, _tmem_load_14[2] * e_dg_1503);
            float dk2_1505 = __fmaf_rn(_tmem_load_13[2], e_off_1502, _tmem_load_15[2] * e_dg_1503);
            dbp[6] = dk2_1505 * k_t_1494;
            float dk2b_1506 = dk2_1505 * bt_t_1501;
            float dq_o_1507 = dq2_1504 + dqf_1498;
            float dk_o_1508 = dk2b_1506 + dkf_1499 + _tmem_load_16[2];
            float dg_o_1509 = __fmaf_rn(q_t_1497, dq2_1504, (dk2b_1506 - _tmem_load_16[2]) * k_t_1494) + dgf_1500;
            int goff_1510 = t_1490 * rs_g;
            int qoff_1511 = t_1490 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1510) + (0)) = dg_o_1509;
            if (t_1490 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1511) + (0)) = __float2bfloat16_rn(dq_o_1507);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1511) + (0)) = __float2bfloat16_rn(dk_o_1508);
            }
            int t_1512 = 32 + 8 * th + 4 + 3;
            float r_1513[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1513[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1514 = r_1513[0];
            unsigned int r_1515[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1515[0])) : "r"((kb8 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1516 = __uint_as_float(r_1515[0] << 16);
            unsigned int r_1517[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1517[0])) : "r"((qbase_1374 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1518 = __uint_as_float(r_1517[0] << 16);
            float q_t_1519 = ((t_1512 >= clen) ? 0.0f : q_t_1518);
            float dqf_1520 = pool_f32[ew + 896 + d_lane];
            float dkf_1521 = pool_f32[ew + 1920 + d_lane];
            float dgf_1522 = pool_f32[ew + 2944 + d_lane];
            float bt_t_1523 = btv_1445[3];
            float _exp2_198 = approx_exp2(g_t_1514 - gn);
            float e_off_1524 = _exp2_198;
            float _exp2_199 = approx_exp2(g_t_1514 - gc_1373);
            float e_dg_1525 = _exp2_199;
            float dq2_1526 = __fmaf_rn(_tmem_load_12[3], e_off_1524, _tmem_load_14[3] * e_dg_1525);
            float dk2_1527 = __fmaf_rn(_tmem_load_13[3], e_off_1524, _tmem_load_15[3] * e_dg_1525);
            dbp[7] = dk2_1527 * k_t_1516;
            float dk2b_1528 = dk2_1527 * bt_t_1523;
            float dq_o_1529 = dq2_1526 + dqf_1520;
            float dk_o_1530 = dk2b_1528 + dkf_1521 + _tmem_load_16[3];
            float dg_o_1531 = __fmaf_rn(q_t_1519, dq2_1526, (dk2b_1528 - _tmem_load_16[3]) * k_t_1516) + dgf_1522;
            int goff_1532 = t_1512 * rs_g;
            int qoff_1533 = t_1512 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1532) + (0)) = dg_o_1531;
            if (t_1512 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1533) + (0)) = __float2bfloat16_rn(dq_o_1529);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1533) + (0)) = __float2bfloat16_rn(dk_o_1530);
            }
            int b4 = lane_1 >> 4 & 1;
            int b3 = lane_1 >> 3 & 1;
            int b2 = lane_1 >> 2 & 1;
            float v4[4];
            float send = ((b4 == 0) ? dbp[4] : dbp[0]);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            v4[0] = ((b4 == 0) ? dbp[0] : dbp[4]) + _shfl_xor_0;
            float send_1534 = ((b4 == 0) ? dbp[5] : dbp[1]);
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1534, 16);
            v4[1] = ((b4 == 0) ? dbp[1] : dbp[5]) + _shfl_xor_1;
            float send_1535 = ((b4 == 0) ? dbp[6] : dbp[2]);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1535, 16);
            v4[2] = ((b4 == 0) ? dbp[2] : dbp[6]) + _shfl_xor_2;
            float send_1536 = ((b4 == 0) ? dbp[7] : dbp[3]);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1536, 16);
            v4[3] = ((b4 == 0) ? dbp[3] : dbp[7]) + _shfl_xor_3;
            float v2_1537[2];
            float send_1538 = ((b3 == 0) ? v4[2] : v4[0]);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1538, 8);
            v2_1537[0] = ((b3 == 0) ? v4[0] : v4[2]) + _shfl_xor_4;
            float send_1539 = ((b3 == 0) ? v4[3] : v4[1]);
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1539, 8);
            v2_1537[1] = ((b3 == 0) ? v4[1] : v4[3]) + _shfl_xor_5;
            float send_1540 = ((b2 == 0) ? v2_1537[1] : v2_1537[0]);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1540, 4);
            float v1 = ((b2 == 0) ? v2_1537[0] : v2_1537[1]) + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
            v1 = v1 + _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
            v1 = v1 + _shfl_xor_8;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1a_t2_drained_addr);
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            mbarrier_wait(epi_full_addr + (th) * 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1541[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1541[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1542 = r_1541[0];
            float gc_1543 = gn_1542 - 21.640425324000002f;
            int eb_1544 = 98304 + th * -32768;
            int ew_1545 = eb_1544 >> 2;
            int qbase_1546 = pool_f32_addr + (unsigned int)(eb_1544 + 12288 + dh_lane * 1024);
            int sb8_1547 = 48 + 8 * th;
            int gb8_1548 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1547 * 128);
            int kb8_1549 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1547 * 128);
            float dbp_1550[8];
            float _tmem_load_17[4];
            tmem_ld_x4(&_tmem_load_17[0], taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th));
            float _tmem_load_18[4];
            tmem_ld_x4(&_tmem_load_18[0], taddr + (unsigned int)row_base + (unsigned int)(176 + 8 * th));
            float _tmem_load_19[4];
            tmem_ld_x4(&_tmem_load_19[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th));
            float _tmem_load_20[4];
            tmem_ld_x4(&_tmem_load_20[0], taddr + (unsigned int)row_base + (unsigned int)(240 + 8 * th));
            float _tmem_load_21[4];
            tmem_ld_x4(&_tmem_load_21[0], taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th));
            float btv_1551[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1551[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1551[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1551[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1551[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (48 + 8 * th))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1552 = 48 + 8 * th;
            float r_1553[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1553[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1554 = r_1553[0];
            unsigned int r_1555[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1555[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1556 = __uint_as_float(r_1555[0] << 16);
            unsigned int r_1557[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1557[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1558 = __uint_as_float(r_1557[0] << 16);
            float q_t_1559 = ((t_1552 >= clen) ? 0.0f : q_t_1558);
            float dqf_1560 = pool_f32[ew_1545 + d_lane];
            float dkf_1561 = pool_f32[ew_1545 + 1024 + d_lane];
            float dgf_1562 = pool_f32[ew_1545 + 2048 + d_lane];
            float bt_t_1563 = btv_1551[0];
            float _exp2_200 = approx_exp2(g_t_1554 - gn_1542);
            float e_off_1564 = _exp2_200;
            float _exp2_201 = approx_exp2(g_t_1554 - gc_1543);
            float e_dg_1565 = _exp2_201;
            float dq2_1566 = __fmaf_rn(_tmem_load_17[0], e_off_1564, _tmem_load_19[0] * e_dg_1565);
            float dk2_1567 = __fmaf_rn(_tmem_load_18[0], e_off_1564, _tmem_load_20[0] * e_dg_1565);
            dbp_1550[0] = dk2_1567 * k_t_1556;
            float dk2b_1568 = dk2_1567 * bt_t_1563;
            float dq_o_1569 = dq2_1566 + dqf_1560;
            float dk_o_1570 = dk2b_1568 + dkf_1561 + _tmem_load_21[0];
            float dg_o_1571 = __fmaf_rn(q_t_1559, dq2_1566, (dk2b_1568 - _tmem_load_21[0]) * k_t_1556) + dgf_1562;
            int goff_1572 = t_1552 * rs_g;
            int qoff_1573 = t_1552 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1572) + (0)) = dg_o_1571;
            if (t_1552 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1573) + (0)) = __float2bfloat16_rn(dq_o_1569);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1573) + (0)) = __float2bfloat16_rn(dk_o_1570);
            }
            int t_1574 = 48 + 8 * th + 1;
            float r_1575[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1575[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1576 = r_1575[0];
            unsigned int r_1577[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1577[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1578 = __uint_as_float(r_1577[0] << 16);
            unsigned int r_1579[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1579[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1580 = __uint_as_float(r_1579[0] << 16);
            float q_t_1581 = ((t_1574 >= clen) ? 0.0f : q_t_1580);
            float dqf_1582 = pool_f32[ew_1545 + 128 + d_lane];
            float dkf_1583 = pool_f32[ew_1545 + 1152 + d_lane];
            float dgf_1584 = pool_f32[ew_1545 + 2176 + d_lane];
            float bt_t_1585 = btv_1551[1];
            float _exp2_202 = approx_exp2(g_t_1576 - gn_1542);
            float e_off_1586 = _exp2_202;
            float _exp2_203 = approx_exp2(g_t_1576 - gc_1543);
            float e_dg_1587 = _exp2_203;
            float dq2_1588 = __fmaf_rn(_tmem_load_17[1], e_off_1586, _tmem_load_19[1] * e_dg_1587);
            float dk2_1589 = __fmaf_rn(_tmem_load_18[1], e_off_1586, _tmem_load_20[1] * e_dg_1587);
            dbp_1550[1] = dk2_1589 * k_t_1578;
            float dk2b_1590 = dk2_1589 * bt_t_1585;
            float dq_o_1591 = dq2_1588 + dqf_1582;
            float dk_o_1592 = dk2b_1590 + dkf_1583 + _tmem_load_21[1];
            float dg_o_1593 = __fmaf_rn(q_t_1581, dq2_1588, (dk2b_1590 - _tmem_load_21[1]) * k_t_1578) + dgf_1584;
            int goff_1594 = t_1574 * rs_g;
            int qoff_1595 = t_1574 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1594) + (0)) = dg_o_1593;
            if (t_1574 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1595) + (0)) = __float2bfloat16_rn(dq_o_1591);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1595) + (0)) = __float2bfloat16_rn(dk_o_1592);
            }
            int t_1596 = 48 + 8 * th + 2;
            float r_1597[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1597[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1598 = r_1597[0];
            unsigned int r_1599[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1599[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1600 = __uint_as_float(r_1599[0] << 16);
            unsigned int r_1601[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1601[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1602 = __uint_as_float(r_1601[0] << 16);
            float q_t_1603 = ((t_1596 >= clen) ? 0.0f : q_t_1602);
            float dqf_1604 = pool_f32[ew_1545 + 256 + d_lane];
            float dkf_1605 = pool_f32[ew_1545 + 1280 + d_lane];
            float dgf_1606 = pool_f32[ew_1545 + 2304 + d_lane];
            float bt_t_1607 = btv_1551[2];
            float _exp2_204 = approx_exp2(g_t_1598 - gn_1542);
            float e_off_1608 = _exp2_204;
            float _exp2_205 = approx_exp2(g_t_1598 - gc_1543);
            float e_dg_1609 = _exp2_205;
            float dq2_1610 = __fmaf_rn(_tmem_load_17[2], e_off_1608, _tmem_load_19[2] * e_dg_1609);
            float dk2_1611 = __fmaf_rn(_tmem_load_18[2], e_off_1608, _tmem_load_20[2] * e_dg_1609);
            dbp_1550[2] = dk2_1611 * k_t_1600;
            float dk2b_1612 = dk2_1611 * bt_t_1607;
            float dq_o_1613 = dq2_1610 + dqf_1604;
            float dk_o_1614 = dk2b_1612 + dkf_1605 + _tmem_load_21[2];
            float dg_o_1615 = __fmaf_rn(q_t_1603, dq2_1610, (dk2b_1612 - _tmem_load_21[2]) * k_t_1600) + dgf_1606;
            int goff_1616 = t_1596 * rs_g;
            int qoff_1617 = t_1596 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1616) + (0)) = dg_o_1615;
            if (t_1596 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1617) + (0)) = __float2bfloat16_rn(dq_o_1613);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1617) + (0)) = __float2bfloat16_rn(dk_o_1614);
            }
            int t_1618 = 48 + 8 * th + 3;
            float r_1619[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1619[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1620 = r_1619[0];
            unsigned int r_1621[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1621[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1622 = __uint_as_float(r_1621[0] << 16);
            unsigned int r_1623[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1623[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1624 = __uint_as_float(r_1623[0] << 16);
            float q_t_1625 = ((t_1618 >= clen) ? 0.0f : q_t_1624);
            float dqf_1626 = pool_f32[ew_1545 + 384 + d_lane];
            float dkf_1627 = pool_f32[ew_1545 + 1408 + d_lane];
            float dgf_1628 = pool_f32[ew_1545 + 2432 + d_lane];
            float bt_t_1629 = btv_1551[3];
            float _exp2_206 = approx_exp2(g_t_1620 - gn_1542);
            float e_off_1630 = _exp2_206;
            float _exp2_207 = approx_exp2(g_t_1620 - gc_1543);
            float e_dg_1631 = _exp2_207;
            float dq2_1632 = __fmaf_rn(_tmem_load_17[3], e_off_1630, _tmem_load_19[3] * e_dg_1631);
            float dk2_1633 = __fmaf_rn(_tmem_load_18[3], e_off_1630, _tmem_load_20[3] * e_dg_1631);
            dbp_1550[3] = dk2_1633 * k_t_1622;
            float dk2b_1634 = dk2_1633 * bt_t_1629;
            float dq_o_1635 = dq2_1632 + dqf_1626;
            float dk_o_1636 = dk2b_1634 + dkf_1627 + _tmem_load_21[3];
            float dg_o_1637 = __fmaf_rn(q_t_1625, dq2_1632, (dk2b_1634 - _tmem_load_21[3]) * k_t_1622) + dgf_1628;
            int goff_1638 = t_1618 * rs_g;
            int qoff_1639 = t_1618 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1638) + (0)) = dg_o_1637;
            if (t_1618 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1639) + (0)) = __float2bfloat16_rn(dq_o_1635);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1639) + (0)) = __float2bfloat16_rn(dk_o_1636);
            }
            float _tmem_load_22[4];
            tmem_ld_x4(&_tmem_load_22[0], taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th + 4));
            float _tmem_load_23[4];
            tmem_ld_x4(&_tmem_load_23[0], taddr + (unsigned int)row_base + (unsigned int)(176 + 8 * th + 4));
            float _tmem_load_24[4];
            tmem_ld_x4(&_tmem_load_24[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 4));
            float _tmem_load_25[4];
            tmem_ld_x4(&_tmem_load_25[0], taddr + (unsigned int)row_base + (unsigned int)(240 + 8 * th + 4));
            float _tmem_load_26[4];
            tmem_ld_x4(&_tmem_load_26[0], taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th + 4));
            float btv_1640[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1640[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1640[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1640[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1640[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (48 + 8 * th + 4))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1641 = 48 + 8 * th + 4;
            float r_1642[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1642[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1643 = r_1642[0];
            unsigned int r_1644[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1644[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1645 = __uint_as_float(r_1644[0] << 16);
            unsigned int r_1646[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1646[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1647 = __uint_as_float(r_1646[0] << 16);
            float q_t_1648 = ((t_1641 >= clen) ? 0.0f : q_t_1647);
            float dqf_1649 = pool_f32[ew_1545 + 512 + d_lane];
            float dkf_1650 = pool_f32[ew_1545 + 1536 + d_lane];
            float dgf_1651 = pool_f32[ew_1545 + 2560 + d_lane];
            float bt_t_1652 = btv_1640[0];
            float _exp2_208 = approx_exp2(g_t_1643 - gn_1542);
            float e_off_1653 = _exp2_208;
            float _exp2_209 = approx_exp2(g_t_1643 - gc_1543);
            float e_dg_1654 = _exp2_209;
            float dq2_1655 = __fmaf_rn(_tmem_load_22[0], e_off_1653, _tmem_load_24[0] * e_dg_1654);
            float dk2_1656 = __fmaf_rn(_tmem_load_23[0], e_off_1653, _tmem_load_25[0] * e_dg_1654);
            dbp_1550[4] = dk2_1656 * k_t_1645;
            float dk2b_1657 = dk2_1656 * bt_t_1652;
            float dq_o_1658 = dq2_1655 + dqf_1649;
            float dk_o_1659 = dk2b_1657 + dkf_1650 + _tmem_load_26[0];
            float dg_o_1660 = __fmaf_rn(q_t_1648, dq2_1655, (dk2b_1657 - _tmem_load_26[0]) * k_t_1645) + dgf_1651;
            int goff_1661 = t_1641 * rs_g;
            int qoff_1662 = t_1641 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1661) + (0)) = dg_o_1660;
            if (t_1641 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1662) + (0)) = __float2bfloat16_rn(dq_o_1658);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1662) + (0)) = __float2bfloat16_rn(dk_o_1659);
            }
            int t_1663 = 48 + 8 * th + 4 + 1;
            float r_1664[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1664[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1665 = r_1664[0];
            unsigned int r_1666[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1666[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1667 = __uint_as_float(r_1666[0] << 16);
            unsigned int r_1668[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1668[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1669 = __uint_as_float(r_1668[0] << 16);
            float q_t_1670 = ((t_1663 >= clen) ? 0.0f : q_t_1669);
            float dqf_1671 = pool_f32[ew_1545 + 640 + d_lane];
            float dkf_1672 = pool_f32[ew_1545 + 1664 + d_lane];
            float dgf_1673 = pool_f32[ew_1545 + 2688 + d_lane];
            float bt_t_1674 = btv_1640[1];
            float _exp2_210 = approx_exp2(g_t_1665 - gn_1542);
            float e_off_1675 = _exp2_210;
            float _exp2_211 = approx_exp2(g_t_1665 - gc_1543);
            float e_dg_1676 = _exp2_211;
            float dq2_1677 = __fmaf_rn(_tmem_load_22[1], e_off_1675, _tmem_load_24[1] * e_dg_1676);
            float dk2_1678 = __fmaf_rn(_tmem_load_23[1], e_off_1675, _tmem_load_25[1] * e_dg_1676);
            dbp_1550[5] = dk2_1678 * k_t_1667;
            float dk2b_1679 = dk2_1678 * bt_t_1674;
            float dq_o_1680 = dq2_1677 + dqf_1671;
            float dk_o_1681 = dk2b_1679 + dkf_1672 + _tmem_load_26[1];
            float dg_o_1682 = __fmaf_rn(q_t_1670, dq2_1677, (dk2b_1679 - _tmem_load_26[1]) * k_t_1667) + dgf_1673;
            int goff_1683 = t_1663 * rs_g;
            int qoff_1684 = t_1663 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1683) + (0)) = dg_o_1682;
            if (t_1663 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1684) + (0)) = __float2bfloat16_rn(dq_o_1680);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1684) + (0)) = __float2bfloat16_rn(dk_o_1681);
            }
            int t_1685 = 48 + 8 * th + 4 + 2;
            float r_1686[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1686[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1687 = r_1686[0];
            unsigned int r_1688[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1688[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1689 = __uint_as_float(r_1688[0] << 16);
            unsigned int r_1690[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1690[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1691 = __uint_as_float(r_1690[0] << 16);
            float q_t_1692 = ((t_1685 >= clen) ? 0.0f : q_t_1691);
            float dqf_1693 = pool_f32[ew_1545 + 768 + d_lane];
            float dkf_1694 = pool_f32[ew_1545 + 1792 + d_lane];
            float dgf_1695 = pool_f32[ew_1545 + 2816 + d_lane];
            float bt_t_1696 = btv_1640[2];
            float _exp2_212 = approx_exp2(g_t_1687 - gn_1542);
            float e_off_1697 = _exp2_212;
            float _exp2_213 = approx_exp2(g_t_1687 - gc_1543);
            float e_dg_1698 = _exp2_213;
            float dq2_1699 = __fmaf_rn(_tmem_load_22[2], e_off_1697, _tmem_load_24[2] * e_dg_1698);
            float dk2_1700 = __fmaf_rn(_tmem_load_23[2], e_off_1697, _tmem_load_25[2] * e_dg_1698);
            dbp_1550[6] = dk2_1700 * k_t_1689;
            float dk2b_1701 = dk2_1700 * bt_t_1696;
            float dq_o_1702 = dq2_1699 + dqf_1693;
            float dk_o_1703 = dk2b_1701 + dkf_1694 + _tmem_load_26[2];
            float dg_o_1704 = __fmaf_rn(q_t_1692, dq2_1699, (dk2b_1701 - _tmem_load_26[2]) * k_t_1689) + dgf_1695;
            int goff_1705 = t_1685 * rs_g;
            int qoff_1706 = t_1685 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1705) + (0)) = dg_o_1704;
            if (t_1685 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1706) + (0)) = __float2bfloat16_rn(dq_o_1702);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1706) + (0)) = __float2bfloat16_rn(dk_o_1703);
            }
            int t_1707 = 48 + 8 * th + 4 + 3;
            float r_1708[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1708[0])) : "r"((gb8_1548 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1709 = r_1708[0];
            unsigned int r_1710[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1710[0])) : "r"((kb8_1549 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1711 = __uint_as_float(r_1710[0] << 16);
            unsigned int r_1712[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1712[0])) : "r"((qbase_1546 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1713 = __uint_as_float(r_1712[0] << 16);
            float q_t_1714 = ((t_1707 >= clen) ? 0.0f : q_t_1713);
            float dqf_1715 = pool_f32[ew_1545 + 896 + d_lane];
            float dkf_1716 = pool_f32[ew_1545 + 1920 + d_lane];
            float dgf_1717 = pool_f32[ew_1545 + 2944 + d_lane];
            float bt_t_1718 = btv_1640[3];
            float _exp2_214 = approx_exp2(g_t_1709 - gn_1542);
            float e_off_1719 = _exp2_214;
            float _exp2_215 = approx_exp2(g_t_1709 - gc_1543);
            float e_dg_1720 = _exp2_215;
            float dq2_1721 = __fmaf_rn(_tmem_load_22[3], e_off_1719, _tmem_load_24[3] * e_dg_1720);
            float dk2_1722 = __fmaf_rn(_tmem_load_23[3], e_off_1719, _tmem_load_25[3] * e_dg_1720);
            dbp_1550[7] = dk2_1722 * k_t_1711;
            float dk2b_1723 = dk2_1722 * bt_t_1718;
            float dq_o_1724 = dq2_1721 + dqf_1715;
            float dk_o_1725 = dk2b_1723 + dkf_1716 + _tmem_load_26[3];
            float dg_o_1726 = __fmaf_rn(q_t_1714, dq2_1721, (dk2b_1723 - _tmem_load_26[3]) * k_t_1711) + dgf_1717;
            int goff_1727 = t_1707 * rs_g;
            int qoff_1728 = t_1707 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1727) + (0)) = dg_o_1726;
            if (t_1707 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1728) + (0)) = __float2bfloat16_rn(dq_o_1724);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1728) + (0)) = __float2bfloat16_rn(dk_o_1725);
            }
            int b4_1729 = lane_1 >> 4 & 1;
            int b3_1730 = lane_1 >> 3 & 1;
            int b2_1731 = lane_1 >> 2 & 1;
            float v4_1732[4];
            float send_1733 = ((b4_1729 == 0) ? dbp_1550[4] : dbp_1550[0]);
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1733, 16);
            v4_1732[0] = ((b4_1729 == 0) ? dbp_1550[0] : dbp_1550[4]) + _shfl_xor_9;
            float send_1734 = ((b4_1729 == 0) ? dbp_1550[5] : dbp_1550[1]);
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1734, 16);
            v4_1732[1] = ((b4_1729 == 0) ? dbp_1550[1] : dbp_1550[5]) + _shfl_xor_10;
            float send_1735 = ((b4_1729 == 0) ? dbp_1550[6] : dbp_1550[2]);
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1735, 16);
            v4_1732[2] = ((b4_1729 == 0) ? dbp_1550[2] : dbp_1550[6]) + _shfl_xor_11;
            float send_1736 = ((b4_1729 == 0) ? dbp_1550[7] : dbp_1550[3]);
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1736, 16);
            v4_1732[3] = ((b4_1729 == 0) ? dbp_1550[3] : dbp_1550[7]) + _shfl_xor_12;
            float v2_1737[2];
            float send_1738 = ((b3_1730 == 0) ? v4_1732[2] : v4_1732[0]);
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1738, 8);
            v2_1737[0] = ((b3_1730 == 0) ? v4_1732[0] : v4_1732[2]) + _shfl_xor_13;
            float send_1739 = ((b3_1730 == 0) ? v4_1732[3] : v4_1732[1]);
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1739, 8);
            v2_1737[1] = ((b3_1730 == 0) ? v4_1732[1] : v4_1732[3]) + _shfl_xor_14;
            float send_1740 = ((b2_1731 == 0) ? v2_1737[1] : v2_1737[0]);
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_1740, 4);
            float v1_1741 = ((b2_1731 == 0) ? v2_1737[0] : v2_1737[1]) + _shfl_xor_15;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, v1_1741, 2);
            v1_1741 = v1_1741 + _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, v1_1741, 1);
            v1_1741 = v1_1741 + _shfl_xor_17;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            mbarrier_wait(p1b_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + (2 + th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1742[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1742[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1743 = r_1742[0];
            float gc_1744 = gn_1743 - 21.640425324000002f;
            int eb_1745 = 81920 + th * -32768;
            int ew_1746 = eb_1745 >> 2;
            int qbase_1747 = pool_f32_addr + (unsigned int)(eb_1745 + 12288 + dh_lane * 1024);
            int sb8_1748 = 16 + 8 * th;
            int gb8_1749 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1748 * 128);
            int kb8_1750 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1748 * 128);
            float dbp_1751[8];
            float _tmem_load_27[4];
            tmem_ld_x4(&_tmem_load_27[0], taddr + (unsigned int)row_base + (unsigned int)(64 + 8 * th));
            float _tmem_load_28[4];
            tmem_ld_x4(&_tmem_load_28[0], taddr + (unsigned int)row_base + (unsigned int)(80 + 8 * th));
            float _tmem_load_29[4];
            tmem_ld_x4(&_tmem_load_29[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th));
            float _tmem_load_30[4];
            tmem_ld_x4(&_tmem_load_30[0], taddr + (unsigned int)row_base + (unsigned int)(112 + 8 * th));
            float _tmem_load_31[4];
            tmem_ld_x4(&_tmem_load_31[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            float btv_1752[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1752[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1752[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1752[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1752[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (16 + 8 * th))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1753 = 16 + 8 * th;
            float r_1754[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1754[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1755 = r_1754[0];
            unsigned int r_1756[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1756[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1757 = __uint_as_float(r_1756[0] << 16);
            unsigned int r_1758[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1758[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1759 = __uint_as_float(r_1758[0] << 16);
            float q_t_1760 = ((t_1753 >= clen) ? 0.0f : q_t_1759);
            float dqf_1761 = pool_f32[ew_1746 + d_lane];
            float dkf_1762 = pool_f32[ew_1746 + 1024 + d_lane];
            float dgf_1763 = pool_f32[ew_1746 + 2048 + d_lane];
            float bt_t_1764 = btv_1752[0];
            float _exp2_216 = approx_exp2(g_t_1755 - gn_1743);
            float e_off_1765 = _exp2_216;
            float _exp2_217 = approx_exp2(g_t_1755 - gc_1744);
            float e_dg_1766 = _exp2_217;
            float dq2_1767 = __fmaf_rn(_tmem_load_27[0], e_off_1765, _tmem_load_29[0] * e_dg_1766);
            float dk2_1768 = __fmaf_rn(_tmem_load_28[0], e_off_1765, _tmem_load_30[0] * e_dg_1766);
            dbp_1751[0] = dk2_1768 * k_t_1757;
            float dk2b_1769 = dk2_1768 * bt_t_1764;
            float dq_o_1770 = dq2_1767 + dqf_1761;
            float dk_o_1771 = dk2b_1769 + dkf_1762 + _tmem_load_31[0];
            float dg_o_1772 = __fmaf_rn(q_t_1760, dq2_1767, (dk2b_1769 - _tmem_load_31[0]) * k_t_1757) + dgf_1763;
            int goff_1773 = t_1753 * rs_g;
            int qoff_1774 = t_1753 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1773) + (0)) = dg_o_1772;
            if (t_1753 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1774) + (0)) = __float2bfloat16_rn(dq_o_1770);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1774) + (0)) = __float2bfloat16_rn(dk_o_1771);
            }
            int t_1775 = 16 + 8 * th + 1;
            float r_1776[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1776[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1777 = r_1776[0];
            unsigned int r_1778[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1778[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1779 = __uint_as_float(r_1778[0] << 16);
            unsigned int r_1780[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1780[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1781 = __uint_as_float(r_1780[0] << 16);
            float q_t_1782 = ((t_1775 >= clen) ? 0.0f : q_t_1781);
            float dqf_1783 = pool_f32[ew_1746 + 128 + d_lane];
            float dkf_1784 = pool_f32[ew_1746 + 1152 + d_lane];
            float dgf_1785 = pool_f32[ew_1746 + 2176 + d_lane];
            float bt_t_1786 = btv_1752[1];
            float _exp2_218 = approx_exp2(g_t_1777 - gn_1743);
            float e_off_1787 = _exp2_218;
            float _exp2_219 = approx_exp2(g_t_1777 - gc_1744);
            float e_dg_1788 = _exp2_219;
            float dq2_1789 = __fmaf_rn(_tmem_load_27[1], e_off_1787, _tmem_load_29[1] * e_dg_1788);
            float dk2_1790 = __fmaf_rn(_tmem_load_28[1], e_off_1787, _tmem_load_30[1] * e_dg_1788);
            dbp_1751[1] = dk2_1790 * k_t_1779;
            float dk2b_1791 = dk2_1790 * bt_t_1786;
            float dq_o_1792 = dq2_1789 + dqf_1783;
            float dk_o_1793 = dk2b_1791 + dkf_1784 + _tmem_load_31[1];
            float dg_o_1794 = __fmaf_rn(q_t_1782, dq2_1789, (dk2b_1791 - _tmem_load_31[1]) * k_t_1779) + dgf_1785;
            int goff_1795 = t_1775 * rs_g;
            int qoff_1796 = t_1775 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1795) + (0)) = dg_o_1794;
            if (t_1775 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1796) + (0)) = __float2bfloat16_rn(dq_o_1792);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1796) + (0)) = __float2bfloat16_rn(dk_o_1793);
            }
            int t_1797 = 16 + 8 * th + 2;
            float r_1798[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1798[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1799 = r_1798[0];
            unsigned int r_1800[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1800[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1801 = __uint_as_float(r_1800[0] << 16);
            unsigned int r_1802[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1802[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1803 = __uint_as_float(r_1802[0] << 16);
            float q_t_1804 = ((t_1797 >= clen) ? 0.0f : q_t_1803);
            float dqf_1805 = pool_f32[ew_1746 + 256 + d_lane];
            float dkf_1806 = pool_f32[ew_1746 + 1280 + d_lane];
            float dgf_1807 = pool_f32[ew_1746 + 2304 + d_lane];
            float bt_t_1808 = btv_1752[2];
            float _exp2_220 = approx_exp2(g_t_1799 - gn_1743);
            float e_off_1809 = _exp2_220;
            float _exp2_221 = approx_exp2(g_t_1799 - gc_1744);
            float e_dg_1810 = _exp2_221;
            float dq2_1811 = __fmaf_rn(_tmem_load_27[2], e_off_1809, _tmem_load_29[2] * e_dg_1810);
            float dk2_1812 = __fmaf_rn(_tmem_load_28[2], e_off_1809, _tmem_load_30[2] * e_dg_1810);
            dbp_1751[2] = dk2_1812 * k_t_1801;
            float dk2b_1813 = dk2_1812 * bt_t_1808;
            float dq_o_1814 = dq2_1811 + dqf_1805;
            float dk_o_1815 = dk2b_1813 + dkf_1806 + _tmem_load_31[2];
            float dg_o_1816 = __fmaf_rn(q_t_1804, dq2_1811, (dk2b_1813 - _tmem_load_31[2]) * k_t_1801) + dgf_1807;
            int goff_1817 = t_1797 * rs_g;
            int qoff_1818 = t_1797 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1817) + (0)) = dg_o_1816;
            if (t_1797 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1818) + (0)) = __float2bfloat16_rn(dq_o_1814);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1818) + (0)) = __float2bfloat16_rn(dk_o_1815);
            }
            int t_1819 = 16 + 8 * th + 3;
            float r_1820[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1820[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1821 = r_1820[0];
            unsigned int r_1822[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1822[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1823 = __uint_as_float(r_1822[0] << 16);
            unsigned int r_1824[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1824[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1825 = __uint_as_float(r_1824[0] << 16);
            float q_t_1826 = ((t_1819 >= clen) ? 0.0f : q_t_1825);
            float dqf_1827 = pool_f32[ew_1746 + 384 + d_lane];
            float dkf_1828 = pool_f32[ew_1746 + 1408 + d_lane];
            float dgf_1829 = pool_f32[ew_1746 + 2432 + d_lane];
            float bt_t_1830 = btv_1752[3];
            float _exp2_222 = approx_exp2(g_t_1821 - gn_1743);
            float e_off_1831 = _exp2_222;
            float _exp2_223 = approx_exp2(g_t_1821 - gc_1744);
            float e_dg_1832 = _exp2_223;
            float dq2_1833 = __fmaf_rn(_tmem_load_27[3], e_off_1831, _tmem_load_29[3] * e_dg_1832);
            float dk2_1834 = __fmaf_rn(_tmem_load_28[3], e_off_1831, _tmem_load_30[3] * e_dg_1832);
            dbp_1751[3] = dk2_1834 * k_t_1823;
            float dk2b_1835 = dk2_1834 * bt_t_1830;
            float dq_o_1836 = dq2_1833 + dqf_1827;
            float dk_o_1837 = dk2b_1835 + dkf_1828 + _tmem_load_31[3];
            float dg_o_1838 = __fmaf_rn(q_t_1826, dq2_1833, (dk2b_1835 - _tmem_load_31[3]) * k_t_1823) + dgf_1829;
            int goff_1839 = t_1819 * rs_g;
            int qoff_1840 = t_1819 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1839) + (0)) = dg_o_1838;
            if (t_1819 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1840) + (0)) = __float2bfloat16_rn(dq_o_1836);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1840) + (0)) = __float2bfloat16_rn(dk_o_1837);
            }
            float _tmem_load_32[4];
            tmem_ld_x4(&_tmem_load_32[0], taddr + (unsigned int)row_base + (unsigned int)(64 + 8 * th + 4));
            float _tmem_load_33[4];
            tmem_ld_x4(&_tmem_load_33[0], taddr + (unsigned int)row_base + (unsigned int)(80 + 8 * th + 4));
            float _tmem_load_34[4];
            tmem_ld_x4(&_tmem_load_34[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 4));
            float _tmem_load_35[4];
            tmem_ld_x4(&_tmem_load_35[0], taddr + (unsigned int)row_base + (unsigned int)(112 + 8 * th + 4));
            float _tmem_load_36[4];
            tmem_ld_x4(&_tmem_load_36[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th + 4));
            float btv_1841[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1841[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1841[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1841[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1841[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (16 + 8 * th + 4))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1842 = 16 + 8 * th + 4;
            float r_1843[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1843[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1844 = r_1843[0];
            unsigned int r_1845[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1845[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1846 = __uint_as_float(r_1845[0] << 16);
            unsigned int r_1847[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1847[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1848 = __uint_as_float(r_1847[0] << 16);
            float q_t_1849 = ((t_1842 >= clen) ? 0.0f : q_t_1848);
            float dqf_1850 = pool_f32[ew_1746 + 512 + d_lane];
            float dkf_1851 = pool_f32[ew_1746 + 1536 + d_lane];
            float dgf_1852 = pool_f32[ew_1746 + 2560 + d_lane];
            float bt_t_1853 = btv_1841[0];
            float _exp2_224 = approx_exp2(g_t_1844 - gn_1743);
            float e_off_1854 = _exp2_224;
            float _exp2_225 = approx_exp2(g_t_1844 - gc_1744);
            float e_dg_1855 = _exp2_225;
            float dq2_1856 = __fmaf_rn(_tmem_load_32[0], e_off_1854, _tmem_load_34[0] * e_dg_1855);
            float dk2_1857 = __fmaf_rn(_tmem_load_33[0], e_off_1854, _tmem_load_35[0] * e_dg_1855);
            dbp_1751[4] = dk2_1857 * k_t_1846;
            float dk2b_1858 = dk2_1857 * bt_t_1853;
            float dq_o_1859 = dq2_1856 + dqf_1850;
            float dk_o_1860 = dk2b_1858 + dkf_1851 + _tmem_load_36[0];
            float dg_o_1861 = __fmaf_rn(q_t_1849, dq2_1856, (dk2b_1858 - _tmem_load_36[0]) * k_t_1846) + dgf_1852;
            int goff_1862 = t_1842 * rs_g;
            int qoff_1863 = t_1842 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1862) + (0)) = dg_o_1861;
            if (t_1842 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1863) + (0)) = __float2bfloat16_rn(dq_o_1859);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1863) + (0)) = __float2bfloat16_rn(dk_o_1860);
            }
            int t_1864 = 16 + 8 * th + 4 + 1;
            float r_1865[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1865[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1866 = r_1865[0];
            unsigned int r_1867[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1867[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1868 = __uint_as_float(r_1867[0] << 16);
            unsigned int r_1869[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1869[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1870 = __uint_as_float(r_1869[0] << 16);
            float q_t_1871 = ((t_1864 >= clen) ? 0.0f : q_t_1870);
            float dqf_1872 = pool_f32[ew_1746 + 640 + d_lane];
            float dkf_1873 = pool_f32[ew_1746 + 1664 + d_lane];
            float dgf_1874 = pool_f32[ew_1746 + 2688 + d_lane];
            float bt_t_1875 = btv_1841[1];
            float _exp2_226 = approx_exp2(g_t_1866 - gn_1743);
            float e_off_1876 = _exp2_226;
            float _exp2_227 = approx_exp2(g_t_1866 - gc_1744);
            float e_dg_1877 = _exp2_227;
            float dq2_1878 = __fmaf_rn(_tmem_load_32[1], e_off_1876, _tmem_load_34[1] * e_dg_1877);
            float dk2_1879 = __fmaf_rn(_tmem_load_33[1], e_off_1876, _tmem_load_35[1] * e_dg_1877);
            dbp_1751[5] = dk2_1879 * k_t_1868;
            float dk2b_1880 = dk2_1879 * bt_t_1875;
            float dq_o_1881 = dq2_1878 + dqf_1872;
            float dk_o_1882 = dk2b_1880 + dkf_1873 + _tmem_load_36[1];
            float dg_o_1883 = __fmaf_rn(q_t_1871, dq2_1878, (dk2b_1880 - _tmem_load_36[1]) * k_t_1868) + dgf_1874;
            int goff_1884 = t_1864 * rs_g;
            int qoff_1885 = t_1864 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1884) + (0)) = dg_o_1883;
            if (t_1864 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1885) + (0)) = __float2bfloat16_rn(dq_o_1881);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1885) + (0)) = __float2bfloat16_rn(dk_o_1882);
            }
            int t_1886 = 16 + 8 * th + 4 + 2;
            float r_1887[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1887[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1888 = r_1887[0];
            unsigned int r_1889[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1889[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1890 = __uint_as_float(r_1889[0] << 16);
            unsigned int r_1891[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1891[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1892 = __uint_as_float(r_1891[0] << 16);
            float q_t_1893 = ((t_1886 >= clen) ? 0.0f : q_t_1892);
            float dqf_1894 = pool_f32[ew_1746 + 768 + d_lane];
            float dkf_1895 = pool_f32[ew_1746 + 1792 + d_lane];
            float dgf_1896 = pool_f32[ew_1746 + 2816 + d_lane];
            float bt_t_1897 = btv_1841[2];
            float _exp2_228 = approx_exp2(g_t_1888 - gn_1743);
            float e_off_1898 = _exp2_228;
            float _exp2_229 = approx_exp2(g_t_1888 - gc_1744);
            float e_dg_1899 = _exp2_229;
            float dq2_1900 = __fmaf_rn(_tmem_load_32[2], e_off_1898, _tmem_load_34[2] * e_dg_1899);
            float dk2_1901 = __fmaf_rn(_tmem_load_33[2], e_off_1898, _tmem_load_35[2] * e_dg_1899);
            dbp_1751[6] = dk2_1901 * k_t_1890;
            float dk2b_1902 = dk2_1901 * bt_t_1897;
            float dq_o_1903 = dq2_1900 + dqf_1894;
            float dk_o_1904 = dk2b_1902 + dkf_1895 + _tmem_load_36[2];
            float dg_o_1905 = __fmaf_rn(q_t_1893, dq2_1900, (dk2b_1902 - _tmem_load_36[2]) * k_t_1890) + dgf_1896;
            int goff_1906 = t_1886 * rs_g;
            int qoff_1907 = t_1886 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1906) + (0)) = dg_o_1905;
            if (t_1886 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1907) + (0)) = __float2bfloat16_rn(dq_o_1903);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1907) + (0)) = __float2bfloat16_rn(dk_o_1904);
            }
            int t_1908 = 16 + 8 * th + 4 + 3;
            float r_1909[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1909[0])) : "r"((gb8_1749 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1910 = r_1909[0];
            unsigned int r_1911[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1911[0])) : "r"((kb8_1750 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1912 = __uint_as_float(r_1911[0] << 16);
            unsigned int r_1913[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1913[0])) : "r"((qbase_1747 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1914 = __uint_as_float(r_1913[0] << 16);
            float q_t_1915 = ((t_1908 >= clen) ? 0.0f : q_t_1914);
            float dqf_1916 = pool_f32[ew_1746 + 896 + d_lane];
            float dkf_1917 = pool_f32[ew_1746 + 1920 + d_lane];
            float dgf_1918 = pool_f32[ew_1746 + 2944 + d_lane];
            float bt_t_1919 = btv_1841[3];
            float _exp2_230 = approx_exp2(g_t_1910 - gn_1743);
            float e_off_1920 = _exp2_230;
            float _exp2_231 = approx_exp2(g_t_1910 - gc_1744);
            float e_dg_1921 = _exp2_231;
            float dq2_1922 = __fmaf_rn(_tmem_load_32[3], e_off_1920, _tmem_load_34[3] * e_dg_1921);
            float dk2_1923 = __fmaf_rn(_tmem_load_33[3], e_off_1920, _tmem_load_35[3] * e_dg_1921);
            dbp_1751[7] = dk2_1923 * k_t_1912;
            float dk2b_1924 = dk2_1923 * bt_t_1919;
            float dq_o_1925 = dq2_1922 + dqf_1916;
            float dk_o_1926 = dk2b_1924 + dkf_1917 + _tmem_load_36[3];
            float dg_o_1927 = __fmaf_rn(q_t_1915, dq2_1922, (dk2b_1924 - _tmem_load_36[3]) * k_t_1912) + dgf_1918;
            int goff_1928 = t_1908 * rs_g;
            int qoff_1929 = t_1908 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1928) + (0)) = dg_o_1927;
            if (t_1908 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1929) + (0)) = __float2bfloat16_rn(dq_o_1925);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1929) + (0)) = __float2bfloat16_rn(dk_o_1926);
            }
            int b4_1930 = lane_1 >> 4 & 1;
            int b3_1931 = lane_1 >> 3 & 1;
            int b2_1932 = lane_1 >> 2 & 1;
            float v4_1933[4];
            float send_1934 = ((b4_1930 == 0) ? dbp_1751[4] : dbp_1751[0]);
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, send_1934, 16);
            v4_1933[0] = ((b4_1930 == 0) ? dbp_1751[0] : dbp_1751[4]) + _shfl_xor_18;
            float send_1935 = ((b4_1930 == 0) ? dbp_1751[5] : dbp_1751[1]);
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, send_1935, 16);
            v4_1933[1] = ((b4_1930 == 0) ? dbp_1751[1] : dbp_1751[5]) + _shfl_xor_19;
            float send_1936 = ((b4_1930 == 0) ? dbp_1751[6] : dbp_1751[2]);
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, send_1936, 16);
            v4_1933[2] = ((b4_1930 == 0) ? dbp_1751[2] : dbp_1751[6]) + _shfl_xor_20;
            float send_1937 = ((b4_1930 == 0) ? dbp_1751[7] : dbp_1751[3]);
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, send_1937, 16);
            v4_1933[3] = ((b4_1930 == 0) ? dbp_1751[3] : dbp_1751[7]) + _shfl_xor_21;
            float v2_1938[2];
            float send_1939 = ((b3_1931 == 0) ? v4_1933[2] : v4_1933[0]);
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, send_1939, 8);
            v2_1938[0] = ((b3_1931 == 0) ? v4_1933[0] : v4_1933[2]) + _shfl_xor_22;
            float send_1940 = ((b3_1931 == 0) ? v4_1933[3] : v4_1933[1]);
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, send_1940, 8);
            v2_1938[1] = ((b3_1931 == 0) ? v4_1933[1] : v4_1933[3]) + _shfl_xor_23;
            float send_1941 = ((b2_1932 == 0) ? v2_1938[1] : v2_1938[0]);
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, send_1941, 4);
            float v1_1942 = ((b2_1932 == 0) ? v2_1938[0] : v2_1938[1]) + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, v1_1942, 2);
            v1_1942 = v1_1942 + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, v1_1942, 1);
            v1_1942 = v1_1942 + _shfl_xor_26;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (2 + th) * 8);
            mbarrier_wait(epi_full_addr + (th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1943[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1943[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1944 = r_1943[0];
            float gc_1945 = gn_1944 - 21.640425324000002f;
            int eb_1946 = 98304 + th * -32768;
            int ew_1947 = eb_1946 >> 2;
            int qbase_1948 = pool_f32_addr + (unsigned int)(eb_1946 + 12288 + dh_lane * 1024);
            int sb8_1949 = 8 * th;
            int gb8_1950 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1949 * 128);
            int kb8_1951 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1949 * 128);
            float dbp_1952[8];
            float _tmem_load_37[4];
            tmem_ld_x4(&_tmem_load_37[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            float _tmem_load_38[4];
            tmem_ld_x4(&_tmem_load_38[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th));
            float _tmem_load_39[4];
            tmem_ld_x4(&_tmem_load_39[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th));
            float btv_1953[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1953[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1953[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1953[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1953[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (8 * th))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1954 = 8 * th;
            float r_1955[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1955[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1956 = r_1955[0];
            unsigned int r_1957[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1957[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1958 = __uint_as_float(r_1957[0] << 16);
            unsigned int r_1959[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1959[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1960 = __uint_as_float(r_1959[0] << 16);
            float q_t_1961 = ((t_1954 >= clen) ? 0.0f : q_t_1960);
            float dqf_1962 = pool_f32[ew_1947 + d_lane];
            float dkf_1963 = pool_f32[ew_1947 + 1024 + d_lane];
            float dgf_1964 = pool_f32[ew_1947 + 2048 + d_lane];
            float bt_t_1965 = btv_1953[0];
            float _exp2_232 = approx_exp2(g_t_1956 - gn_1944);
            float e_off_1966 = _exp2_232;
            float _exp2_233 = approx_exp2(g_t_1956 - gc_1945);
            float e_dg_1967 = _exp2_233;
            float dq2_1968 = _tmem_load_37[0] * e_dg_1967;
            float dk2_1969 = _tmem_load_38[0] * e_dg_1967;
            dbp_1952[0] = dk2_1969 * k_t_1958;
            float dk2b_1970 = dk2_1969 * bt_t_1965;
            float dq_o_1971 = dq2_1968 + dqf_1962;
            float dk_o_1972 = dk2b_1970 + dkf_1963 + _tmem_load_39[0];
            float dg_o_1973 = __fmaf_rn(q_t_1961, dq2_1968, (dk2b_1970 - _tmem_load_39[0]) * k_t_1958) + dgf_1964;
            int goff_1974 = t_1954 * rs_g;
            int qoff_1975 = t_1954 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1974) + (0)) = dg_o_1973;
            if (t_1954 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1975) + (0)) = __float2bfloat16_rn(dq_o_1971);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1975) + (0)) = __float2bfloat16_rn(dk_o_1972);
            }
            int t_1976 = 8 * th + 1;
            float r_1977[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1977[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1978 = r_1977[0];
            unsigned int r_1979[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1979[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1980 = __uint_as_float(r_1979[0] << 16);
            unsigned int r_1981[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1981[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1982 = __uint_as_float(r_1981[0] << 16);
            float q_t_1983 = ((t_1976 >= clen) ? 0.0f : q_t_1982);
            float dqf_1984 = pool_f32[ew_1947 + 128 + d_lane];
            float dkf_1985 = pool_f32[ew_1947 + 1152 + d_lane];
            float dgf_1986 = pool_f32[ew_1947 + 2176 + d_lane];
            float bt_t_1987 = btv_1953[1];
            float _exp2_234 = approx_exp2(g_t_1978 - gn_1944);
            float e_off_1988 = _exp2_234;
            float _exp2_235 = approx_exp2(g_t_1978 - gc_1945);
            float e_dg_1989 = _exp2_235;
            float dq2_1990 = _tmem_load_37[1] * e_dg_1989;
            float dk2_1991 = _tmem_load_38[1] * e_dg_1989;
            dbp_1952[1] = dk2_1991 * k_t_1980;
            float dk2b_1992 = dk2_1991 * bt_t_1987;
            float dq_o_1993 = dq2_1990 + dqf_1984;
            float dk_o_1994 = dk2b_1992 + dkf_1985 + _tmem_load_39[1];
            float dg_o_1995 = __fmaf_rn(q_t_1983, dq2_1990, (dk2b_1992 - _tmem_load_39[1]) * k_t_1980) + dgf_1986;
            int goff_1996 = t_1976 * rs_g;
            int qoff_1997 = t_1976 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1996) + (0)) = dg_o_1995;
            if (t_1976 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1997) + (0)) = __float2bfloat16_rn(dq_o_1993);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1997) + (0)) = __float2bfloat16_rn(dk_o_1994);
            }
            int t_1998 = 8 * th + 2;
            float r_1999[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1999[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2000 = r_1999[0];
            unsigned int r_2001[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2001[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2002 = __uint_as_float(r_2001[0] << 16);
            unsigned int r_2003[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2003[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2004 = __uint_as_float(r_2003[0] << 16);
            float q_t_2005 = ((t_1998 >= clen) ? 0.0f : q_t_2004);
            float dqf_2006 = pool_f32[ew_1947 + 256 + d_lane];
            float dkf_2007 = pool_f32[ew_1947 + 1280 + d_lane];
            float dgf_2008 = pool_f32[ew_1947 + 2304 + d_lane];
            float bt_t_2009 = btv_1953[2];
            float _exp2_236 = approx_exp2(g_t_2000 - gn_1944);
            float e_off_2010 = _exp2_236;
            float _exp2_237 = approx_exp2(g_t_2000 - gc_1945);
            float e_dg_2011 = _exp2_237;
            float dq2_2012 = _tmem_load_37[2] * e_dg_2011;
            float dk2_2013 = _tmem_load_38[2] * e_dg_2011;
            dbp_1952[2] = dk2_2013 * k_t_2002;
            float dk2b_2014 = dk2_2013 * bt_t_2009;
            float dq_o_2015 = dq2_2012 + dqf_2006;
            float dk_o_2016 = dk2b_2014 + dkf_2007 + _tmem_load_39[2];
            float dg_o_2017 = __fmaf_rn(q_t_2005, dq2_2012, (dk2b_2014 - _tmem_load_39[2]) * k_t_2002) + dgf_2008;
            int goff_2018 = t_1998 * rs_g;
            int qoff_2019 = t_1998 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2018) + (0)) = dg_o_2017;
            if (t_1998 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2019) + (0)) = __float2bfloat16_rn(dq_o_2015);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2019) + (0)) = __float2bfloat16_rn(dk_o_2016);
            }
            int t_2020 = 8 * th + 3;
            float r_2021[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2021[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2022 = r_2021[0];
            unsigned int r_2023[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2023[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2024 = __uint_as_float(r_2023[0] << 16);
            unsigned int r_2025[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2025[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2026 = __uint_as_float(r_2025[0] << 16);
            float q_t_2027 = ((t_2020 >= clen) ? 0.0f : q_t_2026);
            float dqf_2028 = pool_f32[ew_1947 + 384 + d_lane];
            float dkf_2029 = pool_f32[ew_1947 + 1408 + d_lane];
            float dgf_2030 = pool_f32[ew_1947 + 2432 + d_lane];
            float bt_t_2031 = btv_1953[3];
            float _exp2_238 = approx_exp2(g_t_2022 - gn_1944);
            float e_off_2032 = _exp2_238;
            float _exp2_239 = approx_exp2(g_t_2022 - gc_1945);
            float e_dg_2033 = _exp2_239;
            float dq2_2034 = _tmem_load_37[3] * e_dg_2033;
            float dk2_2035 = _tmem_load_38[3] * e_dg_2033;
            dbp_1952[3] = dk2_2035 * k_t_2024;
            float dk2b_2036 = dk2_2035 * bt_t_2031;
            float dq_o_2037 = dq2_2034 + dqf_2028;
            float dk_o_2038 = dk2b_2036 + dkf_2029 + _tmem_load_39[3];
            float dg_o_2039 = __fmaf_rn(q_t_2027, dq2_2034, (dk2b_2036 - _tmem_load_39[3]) * k_t_2024) + dgf_2030;
            int goff_2040 = t_2020 * rs_g;
            int qoff_2041 = t_2020 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2040) + (0)) = dg_o_2039;
            if (t_2020 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2041) + (0)) = __float2bfloat16_rn(dq_o_2037);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2041) + (0)) = __float2bfloat16_rn(dk_o_2038);
            }
            float _tmem_load_40[4];
            tmem_ld_x4(&_tmem_load_40[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 4));
            float _tmem_load_41[4];
            tmem_ld_x4(&_tmem_load_41[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th + 4));
            float _tmem_load_42[4];
            tmem_ld_x4(&_tmem_load_42[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th + 4));
            float btv_2042[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_2042[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_2042[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_2042[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_2042[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (8 * th + 4))));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_2043 = 8 * th + 4;
            float r_2044[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2044[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2045 = r_2044[0];
            unsigned int r_2046[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2046[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2047 = __uint_as_float(r_2046[0] << 16);
            unsigned int r_2048[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2048[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2049 = __uint_as_float(r_2048[0] << 16);
            float q_t_2050 = ((t_2043 >= clen) ? 0.0f : q_t_2049);
            float dqf_2051 = pool_f32[ew_1947 + 512 + d_lane];
            float dkf_2052 = pool_f32[ew_1947 + 1536 + d_lane];
            float dgf_2053 = pool_f32[ew_1947 + 2560 + d_lane];
            float bt_t_2054 = btv_2042[0];
            float _exp2_240 = approx_exp2(g_t_2045 - gn_1944);
            float e_off_2055 = _exp2_240;
            float _exp2_241 = approx_exp2(g_t_2045 - gc_1945);
            float e_dg_2056 = _exp2_241;
            float dq2_2057 = _tmem_load_40[0] * e_dg_2056;
            float dk2_2058 = _tmem_load_41[0] * e_dg_2056;
            dbp_1952[4] = dk2_2058 * k_t_2047;
            float dk2b_2059 = dk2_2058 * bt_t_2054;
            float dq_o_2060 = dq2_2057 + dqf_2051;
            float dk_o_2061 = dk2b_2059 + dkf_2052 + _tmem_load_42[0];
            float dg_o_2062 = __fmaf_rn(q_t_2050, dq2_2057, (dk2b_2059 - _tmem_load_42[0]) * k_t_2047) + dgf_2053;
            int goff_2063 = t_2043 * rs_g;
            int qoff_2064 = t_2043 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2063) + (0)) = dg_o_2062;
            if (t_2043 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2064) + (0)) = __float2bfloat16_rn(dq_o_2060);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2064) + (0)) = __float2bfloat16_rn(dk_o_2061);
            }
            int t_2065 = 8 * th + 4 + 1;
            float r_2066[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2066[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2067 = r_2066[0];
            unsigned int r_2068[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2068[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2069 = __uint_as_float(r_2068[0] << 16);
            unsigned int r_2070[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2070[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2071 = __uint_as_float(r_2070[0] << 16);
            float q_t_2072 = ((t_2065 >= clen) ? 0.0f : q_t_2071);
            float dqf_2073 = pool_f32[ew_1947 + 640 + d_lane];
            float dkf_2074 = pool_f32[ew_1947 + 1664 + d_lane];
            float dgf_2075 = pool_f32[ew_1947 + 2688 + d_lane];
            float bt_t_2076 = btv_2042[1];
            float _exp2_242 = approx_exp2(g_t_2067 - gn_1944);
            float e_off_2077 = _exp2_242;
            float _exp2_243 = approx_exp2(g_t_2067 - gc_1945);
            float e_dg_2078 = _exp2_243;
            float dq2_2079 = _tmem_load_40[1] * e_dg_2078;
            float dk2_2080 = _tmem_load_41[1] * e_dg_2078;
            dbp_1952[5] = dk2_2080 * k_t_2069;
            float dk2b_2081 = dk2_2080 * bt_t_2076;
            float dq_o_2082 = dq2_2079 + dqf_2073;
            float dk_o_2083 = dk2b_2081 + dkf_2074 + _tmem_load_42[1];
            float dg_o_2084 = __fmaf_rn(q_t_2072, dq2_2079, (dk2b_2081 - _tmem_load_42[1]) * k_t_2069) + dgf_2075;
            int goff_2085 = t_2065 * rs_g;
            int qoff_2086 = t_2065 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2085) + (0)) = dg_o_2084;
            if (t_2065 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2086) + (0)) = __float2bfloat16_rn(dq_o_2082);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2086) + (0)) = __float2bfloat16_rn(dk_o_2083);
            }
            int t_2087 = 8 * th + 4 + 2;
            float r_2088[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2088[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2089 = r_2088[0];
            unsigned int r_2090[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2090[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2091 = __uint_as_float(r_2090[0] << 16);
            unsigned int r_2092[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2092[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2093 = __uint_as_float(r_2092[0] << 16);
            float q_t_2094 = ((t_2087 >= clen) ? 0.0f : q_t_2093);
            float dqf_2095 = pool_f32[ew_1947 + 768 + d_lane];
            float dkf_2096 = pool_f32[ew_1947 + 1792 + d_lane];
            float dgf_2097 = pool_f32[ew_1947 + 2816 + d_lane];
            float bt_t_2098 = btv_2042[2];
            float _exp2_244 = approx_exp2(g_t_2089 - gn_1944);
            float e_off_2099 = _exp2_244;
            float _exp2_245 = approx_exp2(g_t_2089 - gc_1945);
            float e_dg_2100 = _exp2_245;
            float dq2_2101 = _tmem_load_40[2] * e_dg_2100;
            float dk2_2102 = _tmem_load_41[2] * e_dg_2100;
            dbp_1952[6] = dk2_2102 * k_t_2091;
            float dk2b_2103 = dk2_2102 * bt_t_2098;
            float dq_o_2104 = dq2_2101 + dqf_2095;
            float dk_o_2105 = dk2b_2103 + dkf_2096 + _tmem_load_42[2];
            float dg_o_2106 = __fmaf_rn(q_t_2094, dq2_2101, (dk2b_2103 - _tmem_load_42[2]) * k_t_2091) + dgf_2097;
            int goff_2107 = t_2087 * rs_g;
            int qoff_2108 = t_2087 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2107) + (0)) = dg_o_2106;
            if (t_2087 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2108) + (0)) = __float2bfloat16_rn(dq_o_2104);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2108) + (0)) = __float2bfloat16_rn(dk_o_2105);
            }
            int t_2109 = 8 * th + 4 + 3;
            float r_2110[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2110[0])) : "r"((gb8_1950 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_2111 = r_2110[0];
            unsigned int r_2112[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2112[0])) : "r"((kb8_1951 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_2113 = __uint_as_float(r_2112[0] << 16);
            unsigned int r_2114[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2114[0])) : "r"((qbase_1948 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_2115 = __uint_as_float(r_2114[0] << 16);
            float q_t_2116 = ((t_2109 >= clen) ? 0.0f : q_t_2115);
            float dqf_2117 = pool_f32[ew_1947 + 896 + d_lane];
            float dkf_2118 = pool_f32[ew_1947 + 1920 + d_lane];
            float dgf_2119 = pool_f32[ew_1947 + 2944 + d_lane];
            float bt_t_2120 = btv_2042[3];
            float _exp2_246 = approx_exp2(g_t_2111 - gn_1944);
            float e_off_2121 = _exp2_246;
            float _exp2_247 = approx_exp2(g_t_2111 - gc_1945);
            float e_dg_2122 = _exp2_247;
            float dq2_2123 = _tmem_load_40[3] * e_dg_2122;
            float dk2_2124 = _tmem_load_41[3] * e_dg_2122;
            dbp_1952[7] = dk2_2124 * k_t_2113;
            float dk2b_2125 = dk2_2124 * bt_t_2120;
            float dq_o_2126 = dq2_2123 + dqf_2117;
            float dk_o_2127 = dk2b_2125 + dkf_2118 + _tmem_load_42[3];
            float dg_o_2128 = __fmaf_rn(q_t_2116, dq2_2123, (dk2b_2125 - _tmem_load_42[3]) * k_t_2113) + dgf_2119;
            int goff_2129 = t_2109 * rs_g;
            int qoff_2130 = t_2109 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_2129) + (0)) = dg_o_2128;
            if (t_2109 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_2130) + (0)) = __float2bfloat16_rn(dq_o_2126);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_2130) + (0)) = __float2bfloat16_rn(dk_o_2127);
            }
            int b4_2131 = lane_1 >> 4 & 1;
            int b3_2132 = lane_1 >> 3 & 1;
            int b2_2133 = lane_1 >> 2 & 1;
            float v4_2134[4];
            float send_2135 = ((b4_2131 == 0) ? dbp_1952[4] : dbp_1952[0]);
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, send_2135, 16);
            v4_2134[0] = ((b4_2131 == 0) ? dbp_1952[0] : dbp_1952[4]) + _shfl_xor_27;
            float send_2136 = ((b4_2131 == 0) ? dbp_1952[5] : dbp_1952[1]);
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, send_2136, 16);
            v4_2134[1] = ((b4_2131 == 0) ? dbp_1952[1] : dbp_1952[5]) + _shfl_xor_28;
            float send_2137 = ((b4_2131 == 0) ? dbp_1952[6] : dbp_1952[2]);
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, send_2137, 16);
            v4_2134[2] = ((b4_2131 == 0) ? dbp_1952[2] : dbp_1952[6]) + _shfl_xor_29;
            float send_2138 = ((b4_2131 == 0) ? dbp_1952[7] : dbp_1952[3]);
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, send_2138, 16);
            v4_2134[3] = ((b4_2131 == 0) ? dbp_1952[3] : dbp_1952[7]) + _shfl_xor_30;
            float v2_2139[2];
            float send_2140 = ((b3_2132 == 0) ? v4_2134[2] : v4_2134[0]);
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, send_2140, 8);
            v2_2139[0] = ((b3_2132 == 0) ? v4_2134[0] : v4_2134[2]) + _shfl_xor_31;
            float send_2141 = ((b3_2132 == 0) ? v4_2134[3] : v4_2134[1]);
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, send_2141, 8);
            v2_2139[1] = ((b3_2132 == 0) ? v4_2134[1] : v4_2134[3]) + _shfl_xor_32;
            float send_2142 = ((b2_2133 == 0) ? v2_2139[1] : v2_2139[0]);
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, send_2142, 4);
            float v1_2143 = ((b2_2133 == 0) ? v2_2139[0] : v2_2139[1]) + _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, v1_2143, 2);
            v1_2143 = v1_2143 + _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, v1_2143, 1);
            v1_2143 = v1_2143 + _shfl_xor_35;
            if ((lane_1 & 3) == 0) {
                int e_star = lane_1 >> 2 & 7;
                pool_f32[28160 + qd * 64 + 8 * th + e_star] = v1_2143;
                pool_f32[28160 + qd * 64 + 16 + 8 * th + e_star] = v1_1942;
                pool_f32[28160 + qd * 64 + 32 + 8 * th + e_star] = v1;
                pool_f32[28160 + qd * 64 + 48 + 8 * th + e_star] = v1_1741;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (tid_0 < 64) {
                int idx = (row0 + tid_0) * num_heads + hv;
                float total = pool_f32[28160 + tid_0] + pool_f32[28224 + tid_0];
                total = total + pool_f32[28288 + tid_0];
                total = total + pool_f32[28352 + tid_0];
                db_out[idx] = db_f[idx] + total;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (w == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
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
                        int t_1 = r_1;
                        if (t_1 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_1 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 16)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 32)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 48)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 64)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 80)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 96)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_1 * 128 + 112)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_1 * 128)), "f"(z[0]), "f"(z[1]), "f"(z[2]), "f"(z[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr);
                }
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_0 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_0 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _mma_a_lo_0 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_0 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (64))), "r"(0));
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
                    for (int r_9 = 0; r_9 < 16; r_9++) {
                        int t_2 = 16 + r_9;
                        if (t_2 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_9 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 8);
                }
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_1 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_1 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _mma_a_lo_1 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_1 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_2 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_2 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                int _mma_a_lo_2 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_2 = (((tiles_addr) >> 4) & 0x3FFF) + (2) * 128;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_acc + (80))), "r"(0));
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
                    for (int r_11 = 0; r_11 < 16; r_11++) {
                        int t_3 = 32 + r_11;
                        if (t_3 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_11 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 16);
                }
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_3 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_3 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                int _mma_a_lo_3 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_3 = (((tiles_addr) >> 4) & 0x3FFF) + (3) * 128;
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_acc), "r"(1));
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_4 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_4 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _mma_a_lo_4 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_4 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_acc + (16))), "r"(0));
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_5 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_5 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _mma_a_lo_5 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_5 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_acc + (96))), "r"(0));
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
                    for (int r_13 = 0; r_13 < 16; r_13++) {
                        int t_4 = 48 + r_13;
                        if (t_4 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_13 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(kq_ready_addr + 24);
                }
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_6 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_6 = (((tiles_addr) >> 4) & 0x7FFF) + (2) * 128;
#else
                int _mma_a_lo_6 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_6 = (((tiles_addr) >> 4) & 0x3FFF) + (2) * 128;
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_acc), "r"(1));
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_7 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_7 = (((tiles_addr) >> 4) & 0x7FFF) + (3) * 128;
#else
                int _mma_a_lo_7 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_7 = (((tiles_addr) >> 4) & 0x3FFF) + (3) * 128;
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_acc + (16))), "r"(1));
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_8 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_8 = (((tiles_addr) >> 4) & 0x7FFF) + (0) * 128;
#else
                int _mma_a_lo_8 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_8 = (((tiles_addr) >> 4) & 0x3FFF) + (0) * 128;
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
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"((tmem_acc + (32))), "r"(0));
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_9 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_9 = (((tiles_addr) >> 4) & 0x7FFF) + (1) * 128;
#else
                int _mma_a_lo_9 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_9 = (((tiles_addr) >> 4) & 0x3FFF) + (1) * 128;
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
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_acc + (112))), "r"(0));
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
            long long pf_off = pf_base + (long long)num_heads * 5120;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off)));
            long long pf_off_0 = pf_base + (long long)num_heads * 6144;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off_0)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off_0)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off_0)));
            long long pf_off_1 = pf_base + (long long)num_heads * 7168;
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_f + pf_off_1)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dk_f + pf_off_1)));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dg_f + pf_off_1)));
            if (elect_sync()) {
                mbarrier_wait(ring_full_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_10 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_10 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_10 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_10 = (((dab_addr) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_acc + (128))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_11 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_11 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_11 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_11 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_acc + (144))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_12 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_12 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_12 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_12 = (((dab_addr) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"((tmem_acc + (160))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_13 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_13 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_13 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_13 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_13), "r"((tmem_acc + (176))), "r"(0));
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_14 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_14 = (((dab_addr + 64) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_14 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_14 = (((dab_addr + 64) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"((tmem_acc + (128))), "r"(1));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_15 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_15 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_15 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_15 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_15), "r"((tmem_acc + (144))), "r"(1));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_16 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_16 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_16 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_16 = (((dab_addr + 64) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"((tmem_acc + (160))), "r"(1));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_17 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_17 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_17 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_17 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_17), "r"(_mma_b_lo_17), "r"((tmem_acc + (176))), "r"(1));
                tcgen05_commit(ring_empty_addr + 8);
                mbarrier_wait(ring_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_18 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_18 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_18 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_18 = (((dab_addr + 4096) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_18), "r"(_mma_b_lo_18), "r"((tmem_acc + (160))), "r"(1));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_19 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_19 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_19 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_19 = (((dab_addr + 6144) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_19), "r"(_mma_b_lo_19), "r"((tmem_acc + (176))), "r"(1));
                tcgen05_commit(ring_empty_addr);
                mbarrier_wait(ring_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_20 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_20 = (((dab_addr + 4096) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_20 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_20 = (((dab_addr + 4096) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_20), "r"(_mma_b_lo_20), "r"((tmem_acc + (192))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_21 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_21 = (((dab_addr + 6144) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_21 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_21 = (((dab_addr + 6144) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_21), "r"(_mma_b_lo_21), "r"((tmem_acc + (208))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_22 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_22 = (((dab_addr + 4160) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_22 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_22 = (((dab_addr + 4160) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_22), "r"(_mma_b_lo_22), "r"((tmem_acc + (224))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_23 = (((ring_addr + 64) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_23 = (((dab_addr + 6208) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_23 = (((ring_addr + 64) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_23 = (((dab_addr + 6208) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_23), "r"(_mma_b_lo_23), "r"((tmem_acc + (240))), "r"(0));
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
                int _mma_a_lo_24 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_24 = (((dab_addr + 64) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_24 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_24 = (((dab_addr + 64) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_24), "r"(_mma_b_lo_24), "r"((tmem_acc + (96))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_25 = (((ring_addr + 64) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_25 = (((dab_addr + 2112) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_25 = (((ring_addr + 64) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_25 = (((dab_addr + 2112) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_25), "r"(_mma_b_lo_25), "r"((tmem_acc + (112))), "r"(0));
                mbarrier_wait(ring_full_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_26 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_26 = (((dab_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_26 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_26 = (((dab_addr) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_26), "r"(_mma_b_lo_26), "r"((tmem_acc + (64))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_27 = (((ring_addr) >> 4) & 0x7FFF) + (1) * 1024;
                int _mma_b_lo_27 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (1) * 512;
#else
                int _mma_a_lo_27 = (((ring_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_27 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (1) * 512;
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
                    :: "r"(_mma_a_lo_27), "r"(_mma_b_lo_27), "r"((tmem_acc + (80))), "r"(0));
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
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_28 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_28 = (((dab_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_28 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_28 = (((dab_addr) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_28), "r"(_mma_b_lo_28), "r"((tmem_acc + (128))), "r"(0));
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_29 = (((ring_addr) >> 4) & 0x7FFF) + (0) * 1024;
                int _mma_b_lo_29 = (((dab_addr + 2048) >> 4) & 0x7FFF) + (0) * 512;
#else
                int _mma_a_lo_29 = (((ring_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_29 = (((dab_addr + 2048) >> 4) & 0x3FFF) + (0) * 512;
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
                    :: "r"(_mma_a_lo_29), "r"(_mma_b_lo_29), "r"((tmem_acc + (144))), "r"(0));
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

