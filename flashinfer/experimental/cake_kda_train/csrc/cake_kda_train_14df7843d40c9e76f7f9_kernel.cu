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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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
kernel_cake_kda_train_14df7843d40c9e76f7f9(const __grid_constant__ CUtensorMap dAqk, const __grid_constant__ CUtensorMap dAkk, const __grid_constant__ CUtensorMap gk, const __grid_constant__ CUtensorMap k_e, const __grid_constant__ CUtensorMap q_e, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_out, float* __restrict__ db_out, int num_heads, int num_qk_heads, int group)
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
            // epi_empty: 4 barriers, init_count=256
            mbarrier_init(smem + 115128, 256);
            mbarrier_init(smem + 115136, 256);
            mbarrier_init(smem + 115144, 256);
            mbarrier_init(smem + 115152, 256);
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
            float bv[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4)));
            float vq[4];
            float vk[4];
            float r_2[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2[0])) : "r"((gbase + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s = r_2[0];
            unsigned int r_3[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_3[0])) : "r"((qbase + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s = __uint_as_float(r_3[0] << 16);
            unsigned int r_4[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_4[0])) : "r"((kbase + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s = __uint_as_float(r_4[0] << 16);
            float _exp2_0 = approx_exp2(g_s - a_d);
            float ev = _exp2_0;
            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)(k_s * bv[0]);
            __nv_bfloat16 kb = _cvt_bf16_0;
            vq[0] = q_s * ev;
            float _cvt_f32_0 = (float)kb;
            vk[0] = _cvt_f32_0 * ev;
            float r_5[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_5[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_6 = r_5[0];
            unsigned int r_7[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_7[0])) : "r"((qbase + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_8 = __uint_as_float(r_7[0] << 16);
            unsigned int r_9[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_9[0])) : "r"((kbase + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_10 = __uint_as_float(r_9[0] << 16);
            float _exp2_1 = approx_exp2(g_s_6 - a_d);
            float ev_11 = _exp2_1;
            __nv_bfloat16 _cvt_bf16_1 = (__nv_bfloat16)(k_s_10 * bv[1]);
            __nv_bfloat16 kb_12 = _cvt_bf16_1;
            vq[1] = q_s_8 * ev_11;
            float _cvt_f32_1 = (float)kb_12;
            vk[1] = _cvt_f32_1 * ev_11;
            float r_13[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_13[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_14 = r_13[0];
            unsigned int r_15[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_15[0])) : "r"((qbase + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_16 = __uint_as_float(r_15[0] << 16);
            unsigned int r_17[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_17[0])) : "r"((kbase + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_18 = __uint_as_float(r_17[0] << 16);
            float _exp2_2 = approx_exp2(g_s_14 - a_d);
            float ev_19 = _exp2_2;
            __nv_bfloat16 _cvt_bf16_2 = (__nv_bfloat16)(k_s_18 * bv[2]);
            __nv_bfloat16 kb_20 = _cvt_bf16_2;
            vq[2] = q_s_16 * ev_19;
            float _cvt_f32_2 = (float)kb_20;
            vk[2] = _cvt_f32_2 * ev_19;
            float r_21[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_21[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_22 = r_21[0];
            unsigned int r_23[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_23[0])) : "r"((qbase + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_24 = __uint_as_float(r_23[0] << 16);
            unsigned int r_25[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_25[0])) : "r"((kbase + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_26 = __uint_as_float(r_25[0] << 16);
            float _exp2_3 = approx_exp2(g_s_22 - a_d);
            float ev_27 = _exp2_3;
            __nv_bfloat16 _cvt_bf16_3 = (__nv_bfloat16)(k_s_26 * bv[3]);
            __nv_bfloat16 kb_28 = _cvt_bf16_3;
            vq[3] = q_s_24 * ev_27;
            float _cvt_f32_3 = (float)kb_28;
            vk[3] = _cvt_f32_3 * ev_27;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq[0]), "f"(vq[1]), "f"(vq[2]), "f"(vq[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk[0]), "f"(vk[1]), "f"(vk[2]), "f"(vk[3]) : "memory");
            float bv_29[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_29[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase * 4 + 16)));
            float vq_30[4];
            float vk_31[4];
            float r_32[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_32[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_33 = r_32[0];
            unsigned int r_34[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_34[0])) : "r"((qbase + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_35 = __uint_as_float(r_34[0] << 16);
            unsigned int r_36[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_36[0])) : "r"((kbase + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_37 = __uint_as_float(r_36[0] << 16);
            float _exp2_4 = approx_exp2(g_s_33 - a_d);
            float ev_38 = _exp2_4;
            __nv_bfloat16 _cvt_bf16_4 = (__nv_bfloat16)(k_s_37 * bv_29[0]);
            __nv_bfloat16 kb_39 = _cvt_bf16_4;
            vq_30[0] = q_s_35 * ev_38;
            float _cvt_f32_4 = (float)kb_39;
            vk_31[0] = _cvt_f32_4 * ev_38;
            float r_40[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_40[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_41 = r_40[0];
            unsigned int r_42[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_42[0])) : "r"((qbase + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_43 = __uint_as_float(r_42[0] << 16);
            unsigned int r_44[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_44[0])) : "r"((kbase + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_45 = __uint_as_float(r_44[0] << 16);
            float _exp2_5 = approx_exp2(g_s_41 - a_d);
            float ev_46 = _exp2_5;
            __nv_bfloat16 _cvt_bf16_5 = (__nv_bfloat16)(k_s_45 * bv_29[1]);
            __nv_bfloat16 kb_47 = _cvt_bf16_5;
            vq_30[1] = q_s_43 * ev_46;
            float _cvt_f32_5 = (float)kb_47;
            vk_31[1] = _cvt_f32_5 * ev_46;
            float r_48[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_48[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_49 = r_48[0];
            unsigned int r_50[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_50[0])) : "r"((qbase + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_51 = __uint_as_float(r_50[0] << 16);
            unsigned int r_52[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_52[0])) : "r"((kbase + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_53 = __uint_as_float(r_52[0] << 16);
            float _exp2_6 = approx_exp2(g_s_49 - a_d);
            float ev_54 = _exp2_6;
            __nv_bfloat16 _cvt_bf16_6 = (__nv_bfloat16)(k_s_53 * bv_29[2]);
            __nv_bfloat16 kb_55 = _cvt_bf16_6;
            vq_30[2] = q_s_51 * ev_54;
            float _cvt_f32_6 = (float)kb_55;
            vk_31[2] = _cvt_f32_6 * ev_54;
            float r_56[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_56[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_57 = r_56[0];
            unsigned int r_58[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_58[0])) : "r"((qbase + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_59 = __uint_as_float(r_58[0] << 16);
            unsigned int r_60[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_60[0])) : "r"((kbase + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_61 = __uint_as_float(r_60[0] << 16);
            float _exp2_7 = approx_exp2(g_s_57 - a_d);
            float ev_62 = _exp2_7;
            __nv_bfloat16 _cvt_bf16_7 = (__nv_bfloat16)(k_s_61 * bv_29[3]);
            __nv_bfloat16 kb_63 = _cvt_bf16_7;
            vq_30[3] = q_s_59 * ev_62;
            float _cvt_f32_7 = (float)kb_63;
            vk_31[3] = _cvt_f32_7 * ev_62;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_30[0]), "f"(vq_30[1]), "f"(vq_30[2]), "f"(vq_30[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_31[0]), "f"(vk_31[1]), "f"(vk_31[2]), "f"(vk_31[3]) : "memory");
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
            float r_64[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_64[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_65 = r_64[0];
            int sbase_66 = 16 + 8 * th;
            int gbase_67 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_66 * 128);
            int kbase_68 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_66 * 128);
            int qbase_69 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_70[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_70[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_70[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_70[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_70[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_66 * 4)));
            float vq_71[4];
            float vk_72[4];
            float r_73[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_73[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_74 = r_73[0];
            unsigned int r_75[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_75[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_76 = __uint_as_float(r_75[0] << 16);
            unsigned int r_77[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_77[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_78 = __uint_as_float(r_77[0] << 16);
            float _exp2_8 = approx_exp2(g_s_74 - a_d_65);
            float ev_79 = _exp2_8;
            __nv_bfloat16 _cvt_bf16_8 = (__nv_bfloat16)(k_s_78 * bv_70[0]);
            __nv_bfloat16 kb_80 = _cvt_bf16_8;
            vq_71[0] = q_s_76 * ev_79;
            float _cvt_f32_8 = (float)kb_80;
            vk_72[0] = _cvt_f32_8 * ev_79;
            float r_81[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_81[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_82 = r_81[0];
            unsigned int r_83[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_83[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_84 = __uint_as_float(r_83[0] << 16);
            unsigned int r_85[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_85[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_86 = __uint_as_float(r_85[0] << 16);
            float _exp2_9 = approx_exp2(g_s_82 - a_d_65);
            float ev_87 = _exp2_9;
            __nv_bfloat16 _cvt_bf16_9 = (__nv_bfloat16)(k_s_86 * bv_70[1]);
            __nv_bfloat16 kb_88 = _cvt_bf16_9;
            vq_71[1] = q_s_84 * ev_87;
            float _cvt_f32_9 = (float)kb_88;
            vk_72[1] = _cvt_f32_9 * ev_87;
            float r_89[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_89[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_90 = r_89[0];
            unsigned int r_91[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_91[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_92 = __uint_as_float(r_91[0] << 16);
            unsigned int r_93[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_93[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_94 = __uint_as_float(r_93[0] << 16);
            float _exp2_10 = approx_exp2(g_s_90 - a_d_65);
            float ev_95 = _exp2_10;
            __nv_bfloat16 _cvt_bf16_10 = (__nv_bfloat16)(k_s_94 * bv_70[2]);
            __nv_bfloat16 kb_96 = _cvt_bf16_10;
            vq_71[2] = q_s_92 * ev_95;
            float _cvt_f32_10 = (float)kb_96;
            vk_72[2] = _cvt_f32_10 * ev_95;
            float r_97[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_97[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_98 = r_97[0];
            unsigned int r_99[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_99[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_100 = __uint_as_float(r_99[0] << 16);
            unsigned int r_101[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_101[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_102 = __uint_as_float(r_101[0] << 16);
            float _exp2_11 = approx_exp2(g_s_98 - a_d_65);
            float ev_103 = _exp2_11;
            __nv_bfloat16 _cvt_bf16_11 = (__nv_bfloat16)(k_s_102 * bv_70[3]);
            __nv_bfloat16 kb_104 = _cvt_bf16_11;
            vq_71[3] = q_s_100 * ev_103;
            float _cvt_f32_11 = (float)kb_104;
            vk_72[3] = _cvt_f32_11 * ev_103;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_71[0]), "f"(vq_71[1]), "f"(vq_71[2]), "f"(vq_71[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_72[0]), "f"(vk_72[1]), "f"(vk_72[2]), "f"(vk_72[3]) : "memory");
            float bv_105[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_105[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_105[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_105[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_105[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_66 * 4 + 16)));
            float vq_106[4];
            float vk_107[4];
            float r_108[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_108[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_109 = r_108[0];
            unsigned int r_110[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_110[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_111 = __uint_as_float(r_110[0] << 16);
            unsigned int r_112[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_112[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_113 = __uint_as_float(r_112[0] << 16);
            float _exp2_12 = approx_exp2(g_s_109 - a_d_65);
            float ev_114 = _exp2_12;
            __nv_bfloat16 _cvt_bf16_12 = (__nv_bfloat16)(k_s_113 * bv_105[0]);
            __nv_bfloat16 kb_115 = _cvt_bf16_12;
            vq_106[0] = q_s_111 * ev_114;
            float _cvt_f32_12 = (float)kb_115;
            vk_107[0] = _cvt_f32_12 * ev_114;
            float r_116[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_116[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_117 = r_116[0];
            unsigned int r_118[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_118[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_119 = __uint_as_float(r_118[0] << 16);
            unsigned int r_120[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_120[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_121 = __uint_as_float(r_120[0] << 16);
            float _exp2_13 = approx_exp2(g_s_117 - a_d_65);
            float ev_122 = _exp2_13;
            __nv_bfloat16 _cvt_bf16_13 = (__nv_bfloat16)(k_s_121 * bv_105[1]);
            __nv_bfloat16 kb_123 = _cvt_bf16_13;
            vq_106[1] = q_s_119 * ev_122;
            float _cvt_f32_13 = (float)kb_123;
            vk_107[1] = _cvt_f32_13 * ev_122;
            float r_124[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_124[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_125 = r_124[0];
            unsigned int r_126[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_126[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_127 = __uint_as_float(r_126[0] << 16);
            unsigned int r_128[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_128[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_129 = __uint_as_float(r_128[0] << 16);
            float _exp2_14 = approx_exp2(g_s_125 - a_d_65);
            float ev_130 = _exp2_14;
            __nv_bfloat16 _cvt_bf16_14 = (__nv_bfloat16)(k_s_129 * bv_105[2]);
            __nv_bfloat16 kb_131 = _cvt_bf16_14;
            vq_106[2] = q_s_127 * ev_130;
            float _cvt_f32_14 = (float)kb_131;
            vk_107[2] = _cvt_f32_14 * ev_130;
            float r_132[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_132[0])) : "r"((gbase_67 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_133 = r_132[0];
            unsigned int r_134[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_134[0])) : "r"((qbase_69 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_135 = __uint_as_float(r_134[0] << 16);
            unsigned int r_136[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_136[0])) : "r"((kbase_68 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_137 = __uint_as_float(r_136[0] << 16);
            float _exp2_15 = approx_exp2(g_s_133 - a_d_65);
            float ev_138 = _exp2_15;
            __nv_bfloat16 _cvt_bf16_15 = (__nv_bfloat16)(k_s_137 * bv_105[3]);
            __nv_bfloat16 kb_139 = _cvt_bf16_15;
            vq_106[3] = q_s_135 * ev_138;
            float _cvt_f32_15 = (float)kb_139;
            vk_107[3] = _cvt_f32_15 * ev_138;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_106[0]), "f"(vq_106[1]), "f"(vq_106[2]), "f"(vq_106[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_107[0]), "f"(vk_107[1]), "f"(vk_107[2]), "f"(vk_107[3]) : "memory");
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
            float r_140[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_140[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_141 = r_140[0] - 21.640425324000002f;
            int sbase_142 = 16 + 8 * th;
            int gbase_143 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_142 * 128);
            int kbase_144 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_142 * 128);
            int qbase_145 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_146[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_146[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_146[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_146[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_146[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_142 * 4)));
            float vq_147[4];
            float vk_148[4];
            float r_149[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_149[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_150 = r_149[0];
            unsigned int r_151[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_151[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_152 = __uint_as_float(r_151[0] << 16);
            unsigned int r_153[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_153[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_154 = __uint_as_float(r_153[0] << 16);
            float _exp2_16 = approx_exp2(g_s_150 - a_d_141);
            float ev_155 = _exp2_16;
            __nv_bfloat16 _cvt_bf16_16 = (__nv_bfloat16)(k_s_154 * bv_146[0]);
            __nv_bfloat16 kb_156 = _cvt_bf16_16;
            vq_147[0] = q_s_152 * ev_155;
            float _cvt_f32_16 = (float)kb_156;
            vk_148[0] = _cvt_f32_16 * ev_155;
            float r_157[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_157[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_158 = r_157[0];
            unsigned int r_159[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_159[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_160 = __uint_as_float(r_159[0] << 16);
            unsigned int r_161[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_161[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_162 = __uint_as_float(r_161[0] << 16);
            float _exp2_17 = approx_exp2(g_s_158 - a_d_141);
            float ev_163 = _exp2_17;
            __nv_bfloat16 _cvt_bf16_17 = (__nv_bfloat16)(k_s_162 * bv_146[1]);
            __nv_bfloat16 kb_164 = _cvt_bf16_17;
            vq_147[1] = q_s_160 * ev_163;
            float _cvt_f32_17 = (float)kb_164;
            vk_148[1] = _cvt_f32_17 * ev_163;
            float r_165[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_165[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_166 = r_165[0];
            unsigned int r_167[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_167[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_168 = __uint_as_float(r_167[0] << 16);
            unsigned int r_169[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_169[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_170 = __uint_as_float(r_169[0] << 16);
            float _exp2_18 = approx_exp2(g_s_166 - a_d_141);
            float ev_171 = _exp2_18;
            __nv_bfloat16 _cvt_bf16_18 = (__nv_bfloat16)(k_s_170 * bv_146[2]);
            __nv_bfloat16 kb_172 = _cvt_bf16_18;
            vq_147[2] = q_s_168 * ev_171;
            float _cvt_f32_18 = (float)kb_172;
            vk_148[2] = _cvt_f32_18 * ev_171;
            float r_173[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_173[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_174 = r_173[0];
            unsigned int r_175[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_175[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_176 = __uint_as_float(r_175[0] << 16);
            unsigned int r_177[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_177[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_178 = __uint_as_float(r_177[0] << 16);
            float _exp2_19 = approx_exp2(g_s_174 - a_d_141);
            float ev_179 = _exp2_19;
            __nv_bfloat16 _cvt_bf16_19 = (__nv_bfloat16)(k_s_178 * bv_146[3]);
            __nv_bfloat16 kb_180 = _cvt_bf16_19;
            vq_147[3] = q_s_176 * ev_179;
            float _cvt_f32_19 = (float)kb_180;
            vk_148[3] = _cvt_f32_19 * ev_179;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_147[0]), "f"(vq_147[1]), "f"(vq_147[2]), "f"(vq_147[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_148[0]), "f"(vk_148[1]), "f"(vk_148[2]), "f"(vk_148[3]) : "memory");
            float bv_181[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_181[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_181[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_181[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_181[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_142 * 4 + 16)));
            float vq_182[4];
            float vk_183[4];
            float r_184[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_184[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_185 = r_184[0];
            unsigned int r_186[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_186[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_187 = __uint_as_float(r_186[0] << 16);
            unsigned int r_188[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_188[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_189 = __uint_as_float(r_188[0] << 16);
            float _exp2_20 = approx_exp2(g_s_185 - a_d_141);
            float ev_190 = _exp2_20;
            __nv_bfloat16 _cvt_bf16_20 = (__nv_bfloat16)(k_s_189 * bv_181[0]);
            __nv_bfloat16 kb_191 = _cvt_bf16_20;
            vq_182[0] = q_s_187 * ev_190;
            float _cvt_f32_20 = (float)kb_191;
            vk_183[0] = _cvt_f32_20 * ev_190;
            float r_192[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_192[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_193 = r_192[0];
            unsigned int r_194[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_194[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_195 = __uint_as_float(r_194[0] << 16);
            unsigned int r_196[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_196[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_197 = __uint_as_float(r_196[0] << 16);
            float _exp2_21 = approx_exp2(g_s_193 - a_d_141);
            float ev_198 = _exp2_21;
            __nv_bfloat16 _cvt_bf16_21 = (__nv_bfloat16)(k_s_197 * bv_181[1]);
            __nv_bfloat16 kb_199 = _cvt_bf16_21;
            vq_182[1] = q_s_195 * ev_198;
            float _cvt_f32_21 = (float)kb_199;
            vk_183[1] = _cvt_f32_21 * ev_198;
            float r_200[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_200[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_201 = r_200[0];
            unsigned int r_202[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_202[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_203 = __uint_as_float(r_202[0] << 16);
            unsigned int r_204[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_204[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_205 = __uint_as_float(r_204[0] << 16);
            float _exp2_22 = approx_exp2(g_s_201 - a_d_141);
            float ev_206 = _exp2_22;
            __nv_bfloat16 _cvt_bf16_22 = (__nv_bfloat16)(k_s_205 * bv_181[2]);
            __nv_bfloat16 kb_207 = _cvt_bf16_22;
            vq_182[2] = q_s_203 * ev_206;
            float _cvt_f32_22 = (float)kb_207;
            vk_183[2] = _cvt_f32_22 * ev_206;
            float r_208[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_208[0])) : "r"((gbase_143 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_209 = r_208[0];
            unsigned int r_210[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_210[0])) : "r"((qbase_145 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_211 = __uint_as_float(r_210[0] << 16);
            unsigned int r_212[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_212[0])) : "r"((kbase_144 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_213 = __uint_as_float(r_212[0] << 16);
            float _exp2_23 = approx_exp2(g_s_209 - a_d_141);
            float ev_214 = _exp2_23;
            __nv_bfloat16 _cvt_bf16_23 = (__nv_bfloat16)(k_s_213 * bv_181[3]);
            __nv_bfloat16 kb_215 = _cvt_bf16_23;
            vq_182[3] = q_s_211 * ev_214;
            float _cvt_f32_23 = (float)kb_215;
            vk_183[3] = _cvt_f32_23 * ev_214;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_182[0]), "f"(vq_182[1]), "f"(vq_182[2]), "f"(vq_182[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_183[0]), "f"(vk_183[1]), "f"(vk_183[2]), "f"(vk_183[3]) : "memory");
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
            float r_216[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_216[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_217 = r_216[0];
            int sbase_218 = 32 + 8 * th;
            int gbase_219 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_218 * 128);
            int kbase_220 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_218 * 128);
            int qbase_221 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float bv_222[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_222[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_222[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_222[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_222[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_218 * 4)));
            float vq_223[4];
            float vk_224[4];
            float r_225[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_225[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_226 = r_225[0];
            unsigned int r_227[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_227[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_228 = __uint_as_float(r_227[0] << 16);
            unsigned int r_229[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_229[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_230 = __uint_as_float(r_229[0] << 16);
            float _exp2_24 = approx_exp2(g_s_226 - a_d_217);
            float ev_231 = _exp2_24;
            __nv_bfloat16 _cvt_bf16_24 = (__nv_bfloat16)(k_s_230 * bv_222[0]);
            __nv_bfloat16 kb_232 = _cvt_bf16_24;
            vq_223[0] = q_s_228 * ev_231;
            float _cvt_f32_24 = (float)kb_232;
            vk_224[0] = _cvt_f32_24 * ev_231;
            float r_233[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_233[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_234 = r_233[0];
            unsigned int r_235[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_235[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_236 = __uint_as_float(r_235[0] << 16);
            unsigned int r_237[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_237[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_238 = __uint_as_float(r_237[0] << 16);
            float _exp2_25 = approx_exp2(g_s_234 - a_d_217);
            float ev_239 = _exp2_25;
            __nv_bfloat16 _cvt_bf16_25 = (__nv_bfloat16)(k_s_238 * bv_222[1]);
            __nv_bfloat16 kb_240 = _cvt_bf16_25;
            vq_223[1] = q_s_236 * ev_239;
            float _cvt_f32_25 = (float)kb_240;
            vk_224[1] = _cvt_f32_25 * ev_239;
            float r_241[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_241[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_242 = r_241[0];
            unsigned int r_243[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_243[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_244 = __uint_as_float(r_243[0] << 16);
            unsigned int r_245[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_245[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_246 = __uint_as_float(r_245[0] << 16);
            float _exp2_26 = approx_exp2(g_s_242 - a_d_217);
            float ev_247 = _exp2_26;
            __nv_bfloat16 _cvt_bf16_26 = (__nv_bfloat16)(k_s_246 * bv_222[2]);
            __nv_bfloat16 kb_248 = _cvt_bf16_26;
            vq_223[2] = q_s_244 * ev_247;
            float _cvt_f32_26 = (float)kb_248;
            vk_224[2] = _cvt_f32_26 * ev_247;
            float r_249[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_249[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_250 = r_249[0];
            unsigned int r_251[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_251[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_252 = __uint_as_float(r_251[0] << 16);
            unsigned int r_253[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_253[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_254 = __uint_as_float(r_253[0] << 16);
            float _exp2_27 = approx_exp2(g_s_250 - a_d_217);
            float ev_255 = _exp2_27;
            __nv_bfloat16 _cvt_bf16_27 = (__nv_bfloat16)(k_s_254 * bv_222[3]);
            __nv_bfloat16 kb_256 = _cvt_bf16_27;
            vq_223[3] = q_s_252 * ev_255;
            float _cvt_f32_27 = (float)kb_256;
            vk_224[3] = _cvt_f32_27 * ev_255;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_223[0]), "f"(vq_223[1]), "f"(vq_223[2]), "f"(vq_223[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_224[0]), "f"(vk_224[1]), "f"(vk_224[2]), "f"(vk_224[3]) : "memory");
            float bv_257[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_257[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_257[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_257[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_257[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_218 * 4 + 16)));
            float vq_258[4];
            float vk_259[4];
            float r_260[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_260[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_261 = r_260[0];
            unsigned int r_262[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_262[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_263 = __uint_as_float(r_262[0] << 16);
            unsigned int r_264[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_264[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_265 = __uint_as_float(r_264[0] << 16);
            float _exp2_28 = approx_exp2(g_s_261 - a_d_217);
            float ev_266 = _exp2_28;
            __nv_bfloat16 _cvt_bf16_28 = (__nv_bfloat16)(k_s_265 * bv_257[0]);
            __nv_bfloat16 kb_267 = _cvt_bf16_28;
            vq_258[0] = q_s_263 * ev_266;
            float _cvt_f32_28 = (float)kb_267;
            vk_259[0] = _cvt_f32_28 * ev_266;
            float r_268[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_268[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_269 = r_268[0];
            unsigned int r_270[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_270[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_271 = __uint_as_float(r_270[0] << 16);
            unsigned int r_272[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_272[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_273 = __uint_as_float(r_272[0] << 16);
            float _exp2_29 = approx_exp2(g_s_269 - a_d_217);
            float ev_274 = _exp2_29;
            __nv_bfloat16 _cvt_bf16_29 = (__nv_bfloat16)(k_s_273 * bv_257[1]);
            __nv_bfloat16 kb_275 = _cvt_bf16_29;
            vq_258[1] = q_s_271 * ev_274;
            float _cvt_f32_29 = (float)kb_275;
            vk_259[1] = _cvt_f32_29 * ev_274;
            float r_276[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_276[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_277 = r_276[0];
            unsigned int r_278[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_278[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_279 = __uint_as_float(r_278[0] << 16);
            unsigned int r_280[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_280[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_281 = __uint_as_float(r_280[0] << 16);
            float _exp2_30 = approx_exp2(g_s_277 - a_d_217);
            float ev_282 = _exp2_30;
            __nv_bfloat16 _cvt_bf16_30 = (__nv_bfloat16)(k_s_281 * bv_257[2]);
            __nv_bfloat16 kb_283 = _cvt_bf16_30;
            vq_258[2] = q_s_279 * ev_282;
            float _cvt_f32_30 = (float)kb_283;
            vk_259[2] = _cvt_f32_30 * ev_282;
            float r_284[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_284[0])) : "r"((gbase_219 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_285 = r_284[0];
            unsigned int r_286[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_286[0])) : "r"((qbase_221 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_287 = __uint_as_float(r_286[0] << 16);
            unsigned int r_288[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_288[0])) : "r"((kbase_220 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_289 = __uint_as_float(r_288[0] << 16);
            float _exp2_31 = approx_exp2(g_s_285 - a_d_217);
            float ev_290 = _exp2_31;
            __nv_bfloat16 _cvt_bf16_31 = (__nv_bfloat16)(k_s_289 * bv_257[3]);
            __nv_bfloat16 kb_291 = _cvt_bf16_31;
            vq_258[3] = q_s_287 * ev_290;
            float _cvt_f32_31 = (float)kb_291;
            vk_259[3] = _cvt_f32_31 * ev_290;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_258[0]), "f"(vq_258[1]), "f"(vq_258[2]), "f"(vq_258[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_259[0]), "f"(vk_259[1]), "f"(vk_259[2]), "f"(vk_259[3]) : "memory");
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
            float r_292[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_292[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_293 = r_292[0];
            int sbase_294 = 32 + 8 * th;
            int gbase_295 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_294 * 128);
            int kbase_296 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_294 * 128);
            int qbase_297 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float bv_298[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_298[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_298[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_298[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_298[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_294 * 4)));
            float vq_299[4];
            float vk_300[4];
            float r_301[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_301[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_302 = r_301[0];
            unsigned int r_303[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_303[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_304 = __uint_as_float(r_303[0] << 16);
            unsigned int r_305[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_305[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_306 = __uint_as_float(r_305[0] << 16);
            float _exp2_32 = approx_exp2(g_s_302 - a_d_293);
            float ev_307 = _exp2_32;
            __nv_bfloat16 _cvt_bf16_32 = (__nv_bfloat16)(k_s_306 * bv_298[0]);
            __nv_bfloat16 kb_308 = _cvt_bf16_32;
            vq_299[0] = q_s_304 * ev_307;
            float _cvt_f32_32 = (float)kb_308;
            vk_300[0] = _cvt_f32_32 * ev_307;
            float r_309[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_309[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_310 = r_309[0];
            unsigned int r_311[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_311[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_312 = __uint_as_float(r_311[0] << 16);
            unsigned int r_313[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_313[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_314 = __uint_as_float(r_313[0] << 16);
            float _exp2_33 = approx_exp2(g_s_310 - a_d_293);
            float ev_315 = _exp2_33;
            __nv_bfloat16 _cvt_bf16_33 = (__nv_bfloat16)(k_s_314 * bv_298[1]);
            __nv_bfloat16 kb_316 = _cvt_bf16_33;
            vq_299[1] = q_s_312 * ev_315;
            float _cvt_f32_33 = (float)kb_316;
            vk_300[1] = _cvt_f32_33 * ev_315;
            float r_317[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_317[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_318 = r_317[0];
            unsigned int r_319[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_319[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_320 = __uint_as_float(r_319[0] << 16);
            unsigned int r_321[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_321[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_322 = __uint_as_float(r_321[0] << 16);
            float _exp2_34 = approx_exp2(g_s_318 - a_d_293);
            float ev_323 = _exp2_34;
            __nv_bfloat16 _cvt_bf16_34 = (__nv_bfloat16)(k_s_322 * bv_298[2]);
            __nv_bfloat16 kb_324 = _cvt_bf16_34;
            vq_299[2] = q_s_320 * ev_323;
            float _cvt_f32_34 = (float)kb_324;
            vk_300[2] = _cvt_f32_34 * ev_323;
            float r_325[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_325[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_326 = r_325[0];
            unsigned int r_327[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_327[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_328 = __uint_as_float(r_327[0] << 16);
            unsigned int r_329[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_329[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_330 = __uint_as_float(r_329[0] << 16);
            float _exp2_35 = approx_exp2(g_s_326 - a_d_293);
            float ev_331 = _exp2_35;
            __nv_bfloat16 _cvt_bf16_35 = (__nv_bfloat16)(k_s_330 * bv_298[3]);
            __nv_bfloat16 kb_332 = _cvt_bf16_35;
            vq_299[3] = q_s_328 * ev_331;
            float _cvt_f32_35 = (float)kb_332;
            vk_300[3] = _cvt_f32_35 * ev_331;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_299[0]), "f"(vq_299[1]), "f"(vq_299[2]), "f"(vq_299[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_300[0]), "f"(vk_300[1]), "f"(vk_300[2]), "f"(vk_300[3]) : "memory");
            float bv_333[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_333[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_333[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_333[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_333[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_294 * 4 + 16)));
            float vq_334[4];
            float vk_335[4];
            float r_336[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_336[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_337 = r_336[0];
            unsigned int r_338[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_338[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_339 = __uint_as_float(r_338[0] << 16);
            unsigned int r_340[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_340[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_341 = __uint_as_float(r_340[0] << 16);
            float _exp2_36 = approx_exp2(g_s_337 - a_d_293);
            float ev_342 = _exp2_36;
            __nv_bfloat16 _cvt_bf16_36 = (__nv_bfloat16)(k_s_341 * bv_333[0]);
            __nv_bfloat16 kb_343 = _cvt_bf16_36;
            vq_334[0] = q_s_339 * ev_342;
            float _cvt_f32_36 = (float)kb_343;
            vk_335[0] = _cvt_f32_36 * ev_342;
            float r_344[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_344[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_345 = r_344[0];
            unsigned int r_346[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_346[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_347 = __uint_as_float(r_346[0] << 16);
            unsigned int r_348[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_348[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_349 = __uint_as_float(r_348[0] << 16);
            float _exp2_37 = approx_exp2(g_s_345 - a_d_293);
            float ev_350 = _exp2_37;
            __nv_bfloat16 _cvt_bf16_37 = (__nv_bfloat16)(k_s_349 * bv_333[1]);
            __nv_bfloat16 kb_351 = _cvt_bf16_37;
            vq_334[1] = q_s_347 * ev_350;
            float _cvt_f32_37 = (float)kb_351;
            vk_335[1] = _cvt_f32_37 * ev_350;
            float r_352[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_352[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_353 = r_352[0];
            unsigned int r_354[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_354[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_355 = __uint_as_float(r_354[0] << 16);
            unsigned int r_356[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_356[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_357 = __uint_as_float(r_356[0] << 16);
            float _exp2_38 = approx_exp2(g_s_353 - a_d_293);
            float ev_358 = _exp2_38;
            __nv_bfloat16 _cvt_bf16_38 = (__nv_bfloat16)(k_s_357 * bv_333[2]);
            __nv_bfloat16 kb_359 = _cvt_bf16_38;
            vq_334[2] = q_s_355 * ev_358;
            float _cvt_f32_38 = (float)kb_359;
            vk_335[2] = _cvt_f32_38 * ev_358;
            float r_360[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_360[0])) : "r"((gbase_295 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_361 = r_360[0];
            unsigned int r_362[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_362[0])) : "r"((qbase_297 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_363 = __uint_as_float(r_362[0] << 16);
            unsigned int r_364[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_364[0])) : "r"((kbase_296 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_365 = __uint_as_float(r_364[0] << 16);
            float _exp2_39 = approx_exp2(g_s_361 - a_d_293);
            float ev_366 = _exp2_39;
            __nv_bfloat16 _cvt_bf16_39 = (__nv_bfloat16)(k_s_365 * bv_333[3]);
            __nv_bfloat16 kb_367 = _cvt_bf16_39;
            vq_334[3] = q_s_363 * ev_366;
            float _cvt_f32_39 = (float)kb_367;
            vk_335[3] = _cvt_f32_39 * ev_366;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_334[0]), "f"(vq_334[1]), "f"(vq_334[2]), "f"(vq_334[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_335[0]), "f"(vk_335[1]), "f"(vk_335[2]), "f"(vk_335[3]) : "memory");
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
            float r_368[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_368[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_369 = r_368[0] - 21.640425324000002f;
            int sbase_370 = 32 + 8 * th;
            int gbase_371 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_370 * 128);
            int kbase_372 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_370 * 128);
            int qbase_373 = pool_f32_addr + (unsigned int)(98304 + dh_lane * 2048 + th * 1024);
            float bv_374[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_374[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_374[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_374[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_374[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_370 * 4)));
            float vq_375[4];
            float vk_376[4];
            float r_377[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_377[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_378 = r_377[0];
            unsigned int r_379[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_379[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_380 = __uint_as_float(r_379[0] << 16);
            unsigned int r_381[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_381[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_382 = __uint_as_float(r_381[0] << 16);
            float _exp2_40 = approx_exp2(g_s_378 - a_d_369);
            float ev_383 = _exp2_40;
            __nv_bfloat16 _cvt_bf16_40 = (__nv_bfloat16)(k_s_382 * bv_374[0]);
            __nv_bfloat16 kb_384 = _cvt_bf16_40;
            vq_375[0] = q_s_380 * ev_383;
            float _cvt_f32_40 = (float)kb_384;
            vk_376[0] = _cvt_f32_40 * ev_383;
            float r_385[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_385[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_386 = r_385[0];
            unsigned int r_387[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_387[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_388 = __uint_as_float(r_387[0] << 16);
            unsigned int r_389[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_389[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_390 = __uint_as_float(r_389[0] << 16);
            float _exp2_41 = approx_exp2(g_s_386 - a_d_369);
            float ev_391 = _exp2_41;
            __nv_bfloat16 _cvt_bf16_41 = (__nv_bfloat16)(k_s_390 * bv_374[1]);
            __nv_bfloat16 kb_392 = _cvt_bf16_41;
            vq_375[1] = q_s_388 * ev_391;
            float _cvt_f32_41 = (float)kb_392;
            vk_376[1] = _cvt_f32_41 * ev_391;
            float r_393[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_393[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_394 = r_393[0];
            unsigned int r_395[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_395[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_396 = __uint_as_float(r_395[0] << 16);
            unsigned int r_397[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_397[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_398 = __uint_as_float(r_397[0] << 16);
            float _exp2_42 = approx_exp2(g_s_394 - a_d_369);
            float ev_399 = _exp2_42;
            __nv_bfloat16 _cvt_bf16_42 = (__nv_bfloat16)(k_s_398 * bv_374[2]);
            __nv_bfloat16 kb_400 = _cvt_bf16_42;
            vq_375[2] = q_s_396 * ev_399;
            float _cvt_f32_42 = (float)kb_400;
            vk_376[2] = _cvt_f32_42 * ev_399;
            float r_401[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_401[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_402 = r_401[0];
            unsigned int r_403[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_403[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_404 = __uint_as_float(r_403[0] << 16);
            unsigned int r_405[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_405[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_406 = __uint_as_float(r_405[0] << 16);
            float _exp2_43 = approx_exp2(g_s_402 - a_d_369);
            float ev_407 = _exp2_43;
            __nv_bfloat16 _cvt_bf16_43 = (__nv_bfloat16)(k_s_406 * bv_374[3]);
            __nv_bfloat16 kb_408 = _cvt_bf16_43;
            vq_375[3] = q_s_404 * ev_407;
            float _cvt_f32_43 = (float)kb_408;
            vk_376[3] = _cvt_f32_43 * ev_407;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_375[0]), "f"(vq_375[1]), "f"(vq_375[2]), "f"(vq_375[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_376[0]), "f"(vk_376[1]), "f"(vk_376[2]), "f"(vk_376[3]) : "memory");
            float bv_409[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_409[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_409[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_409[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_409[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_370 * 4 + 16)));
            float vq_410[4];
            float vk_411[4];
            float r_412[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_412[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_413 = r_412[0];
            unsigned int r_414[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_414[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_415 = __uint_as_float(r_414[0] << 16);
            unsigned int r_416[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_416[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_417 = __uint_as_float(r_416[0] << 16);
            float _exp2_44 = approx_exp2(g_s_413 - a_d_369);
            float ev_418 = _exp2_44;
            __nv_bfloat16 _cvt_bf16_44 = (__nv_bfloat16)(k_s_417 * bv_409[0]);
            __nv_bfloat16 kb_419 = _cvt_bf16_44;
            vq_410[0] = q_s_415 * ev_418;
            float _cvt_f32_44 = (float)kb_419;
            vk_411[0] = _cvt_f32_44 * ev_418;
            float r_420[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_420[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_421 = r_420[0];
            unsigned int r_422[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_422[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_423 = __uint_as_float(r_422[0] << 16);
            unsigned int r_424[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_424[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_425 = __uint_as_float(r_424[0] << 16);
            float _exp2_45 = approx_exp2(g_s_421 - a_d_369);
            float ev_426 = _exp2_45;
            __nv_bfloat16 _cvt_bf16_45 = (__nv_bfloat16)(k_s_425 * bv_409[1]);
            __nv_bfloat16 kb_427 = _cvt_bf16_45;
            vq_410[1] = q_s_423 * ev_426;
            float _cvt_f32_45 = (float)kb_427;
            vk_411[1] = _cvt_f32_45 * ev_426;
            float r_428[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_428[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_429 = r_428[0];
            unsigned int r_430[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_430[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_431 = __uint_as_float(r_430[0] << 16);
            unsigned int r_432[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_432[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_433 = __uint_as_float(r_432[0] << 16);
            float _exp2_46 = approx_exp2(g_s_429 - a_d_369);
            float ev_434 = _exp2_46;
            __nv_bfloat16 _cvt_bf16_46 = (__nv_bfloat16)(k_s_433 * bv_409[2]);
            __nv_bfloat16 kb_435 = _cvt_bf16_46;
            vq_410[2] = q_s_431 * ev_434;
            float _cvt_f32_46 = (float)kb_435;
            vk_411[2] = _cvt_f32_46 * ev_434;
            float r_436[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_436[0])) : "r"((gbase_371 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_437 = r_436[0];
            unsigned int r_438[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_438[0])) : "r"((qbase_373 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_439 = __uint_as_float(r_438[0] << 16);
            unsigned int r_440[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_440[0])) : "r"((kbase_372 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_441 = __uint_as_float(r_440[0] << 16);
            float _exp2_47 = approx_exp2(g_s_437 - a_d_369);
            float ev_442 = _exp2_47;
            __nv_bfloat16 _cvt_bf16_47 = (__nv_bfloat16)(k_s_441 * bv_409[3]);
            __nv_bfloat16 kb_443 = _cvt_bf16_47;
            vq_410[3] = q_s_439 * ev_442;
            float _cvt_f32_47 = (float)kb_443;
            vk_411[3] = _cvt_f32_47 * ev_442;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_410[0]), "f"(vq_410[1]), "f"(vq_410[2]), "f"(vq_410[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_411[0]), "f"(vk_411[1]), "f"(vk_411[2]), "f"(vk_411[3]) : "memory");
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
            float r_444[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_444[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_445 = r_444[0];
            int sbase_446 = 48 + 8 * th;
            int gbase_447 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_446 * 128);
            int kbase_448 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_446 * 128);
            int qbase_449 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_450[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_450[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_450[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_450[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_450[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_446 * 4)));
            float vq_451[4];
            float vk_452[4];
            float r_453[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_453[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_454 = r_453[0];
            unsigned int r_455[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_455[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_456 = __uint_as_float(r_455[0] << 16);
            unsigned int r_457[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_457[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_458 = __uint_as_float(r_457[0] << 16);
            float _exp2_48 = approx_exp2(g_s_454 - a_d_445);
            float ev_459 = _exp2_48;
            __nv_bfloat16 _cvt_bf16_48 = (__nv_bfloat16)(k_s_458 * bv_450[0]);
            __nv_bfloat16 kb_460 = _cvt_bf16_48;
            vq_451[0] = q_s_456 * ev_459;
            float _cvt_f32_48 = (float)kb_460;
            vk_452[0] = _cvt_f32_48 * ev_459;
            float r_461[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_461[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_462 = r_461[0];
            unsigned int r_463[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_463[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_464 = __uint_as_float(r_463[0] << 16);
            unsigned int r_465[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_465[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_466 = __uint_as_float(r_465[0] << 16);
            float _exp2_49 = approx_exp2(g_s_462 - a_d_445);
            float ev_467 = _exp2_49;
            __nv_bfloat16 _cvt_bf16_49 = (__nv_bfloat16)(k_s_466 * bv_450[1]);
            __nv_bfloat16 kb_468 = _cvt_bf16_49;
            vq_451[1] = q_s_464 * ev_467;
            float _cvt_f32_49 = (float)kb_468;
            vk_452[1] = _cvt_f32_49 * ev_467;
            float r_469[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_469[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_470 = r_469[0];
            unsigned int r_471[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_471[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_472 = __uint_as_float(r_471[0] << 16);
            unsigned int r_473[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_473[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_474 = __uint_as_float(r_473[0] << 16);
            float _exp2_50 = approx_exp2(g_s_470 - a_d_445);
            float ev_475 = _exp2_50;
            __nv_bfloat16 _cvt_bf16_50 = (__nv_bfloat16)(k_s_474 * bv_450[2]);
            __nv_bfloat16 kb_476 = _cvt_bf16_50;
            vq_451[2] = q_s_472 * ev_475;
            float _cvt_f32_50 = (float)kb_476;
            vk_452[2] = _cvt_f32_50 * ev_475;
            float r_477[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_477[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_478 = r_477[0];
            unsigned int r_479[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_479[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_480 = __uint_as_float(r_479[0] << 16);
            unsigned int r_481[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_481[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_482 = __uint_as_float(r_481[0] << 16);
            float _exp2_51 = approx_exp2(g_s_478 - a_d_445);
            float ev_483 = _exp2_51;
            __nv_bfloat16 _cvt_bf16_51 = (__nv_bfloat16)(k_s_482 * bv_450[3]);
            __nv_bfloat16 kb_484 = _cvt_bf16_51;
            vq_451[3] = q_s_480 * ev_483;
            float _cvt_f32_51 = (float)kb_484;
            vk_452[3] = _cvt_f32_51 * ev_483;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_451[0]), "f"(vq_451[1]), "f"(vq_451[2]), "f"(vq_451[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_452[0]), "f"(vk_452[1]), "f"(vk_452[2]), "f"(vk_452[3]) : "memory");
            float bv_485[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_485[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_485[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_485[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_485[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_446 * 4 + 16)));
            float vq_486[4];
            float vk_487[4];
            float r_488[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_488[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_489 = r_488[0];
            unsigned int r_490[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_490[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_491 = __uint_as_float(r_490[0] << 16);
            unsigned int r_492[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_492[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_493 = __uint_as_float(r_492[0] << 16);
            float _exp2_52 = approx_exp2(g_s_489 - a_d_445);
            float ev_494 = _exp2_52;
            __nv_bfloat16 _cvt_bf16_52 = (__nv_bfloat16)(k_s_493 * bv_485[0]);
            __nv_bfloat16 kb_495 = _cvt_bf16_52;
            vq_486[0] = q_s_491 * ev_494;
            float _cvt_f32_52 = (float)kb_495;
            vk_487[0] = _cvt_f32_52 * ev_494;
            float r_496[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_496[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_497 = r_496[0];
            unsigned int r_498[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_498[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_499 = __uint_as_float(r_498[0] << 16);
            unsigned int r_500[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_500[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_501 = __uint_as_float(r_500[0] << 16);
            float _exp2_53 = approx_exp2(g_s_497 - a_d_445);
            float ev_502 = _exp2_53;
            __nv_bfloat16 _cvt_bf16_53 = (__nv_bfloat16)(k_s_501 * bv_485[1]);
            __nv_bfloat16 kb_503 = _cvt_bf16_53;
            vq_486[1] = q_s_499 * ev_502;
            float _cvt_f32_53 = (float)kb_503;
            vk_487[1] = _cvt_f32_53 * ev_502;
            float r_504[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_504[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_505 = r_504[0];
            unsigned int r_506[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_506[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_507 = __uint_as_float(r_506[0] << 16);
            unsigned int r_508[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_508[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_509 = __uint_as_float(r_508[0] << 16);
            float _exp2_54 = approx_exp2(g_s_505 - a_d_445);
            float ev_510 = _exp2_54;
            __nv_bfloat16 _cvt_bf16_54 = (__nv_bfloat16)(k_s_509 * bv_485[2]);
            __nv_bfloat16 kb_511 = _cvt_bf16_54;
            vq_486[2] = q_s_507 * ev_510;
            float _cvt_f32_54 = (float)kb_511;
            vk_487[2] = _cvt_f32_54 * ev_510;
            float r_512[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_512[0])) : "r"((gbase_447 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_513 = r_512[0];
            unsigned int r_514[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_514[0])) : "r"((qbase_449 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_515 = __uint_as_float(r_514[0] << 16);
            unsigned int r_516[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_516[0])) : "r"((kbase_448 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_517 = __uint_as_float(r_516[0] << 16);
            float _exp2_55 = approx_exp2(g_s_513 - a_d_445);
            float ev_518 = _exp2_55;
            __nv_bfloat16 _cvt_bf16_55 = (__nv_bfloat16)(k_s_517 * bv_485[3]);
            __nv_bfloat16 kb_519 = _cvt_bf16_55;
            vq_486[3] = q_s_515 * ev_518;
            float _cvt_f32_55 = (float)kb_519;
            vk_487[3] = _cvt_f32_55 * ev_518;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_486[0]), "f"(vq_486[1]), "f"(vq_486[2]), "f"(vq_486[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_487[0]), "f"(vk_487[1]), "f"(vk_487[2]), "f"(vk_487[3]) : "memory");
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
            float r_520[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_520[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_521 = r_520[0];
            int sbase_522 = 48 + 8 * th;
            int gbase_523 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_522 * 128);
            int kbase_524 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_522 * 128);
            int qbase_525 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_526[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_526[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_526[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_526[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_526[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_522 * 4)));
            float vq_527[4];
            float vk_528[4];
            float r_529[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_529[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_530 = r_529[0];
            unsigned int r_531[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_531[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_532 = __uint_as_float(r_531[0] << 16);
            unsigned int r_533[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_533[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_534 = __uint_as_float(r_533[0] << 16);
            float _exp2_56 = approx_exp2(g_s_530 - a_d_521);
            float ev_535 = _exp2_56;
            __nv_bfloat16 _cvt_bf16_56 = (__nv_bfloat16)(k_s_534 * bv_526[0]);
            __nv_bfloat16 kb_536 = _cvt_bf16_56;
            vq_527[0] = q_s_532 * ev_535;
            float _cvt_f32_56 = (float)kb_536;
            vk_528[0] = _cvt_f32_56 * ev_535;
            float r_537[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_537[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_538 = r_537[0];
            unsigned int r_539[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_539[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_540 = __uint_as_float(r_539[0] << 16);
            unsigned int r_541[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_541[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_542 = __uint_as_float(r_541[0] << 16);
            float _exp2_57 = approx_exp2(g_s_538 - a_d_521);
            float ev_543 = _exp2_57;
            __nv_bfloat16 _cvt_bf16_57 = (__nv_bfloat16)(k_s_542 * bv_526[1]);
            __nv_bfloat16 kb_544 = _cvt_bf16_57;
            vq_527[1] = q_s_540 * ev_543;
            float _cvt_f32_57 = (float)kb_544;
            vk_528[1] = _cvt_f32_57 * ev_543;
            float r_545[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_545[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_546 = r_545[0];
            unsigned int r_547[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_547[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_548 = __uint_as_float(r_547[0] << 16);
            unsigned int r_549[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_549[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_550 = __uint_as_float(r_549[0] << 16);
            float _exp2_58 = approx_exp2(g_s_546 - a_d_521);
            float ev_551 = _exp2_58;
            __nv_bfloat16 _cvt_bf16_58 = (__nv_bfloat16)(k_s_550 * bv_526[2]);
            __nv_bfloat16 kb_552 = _cvt_bf16_58;
            vq_527[2] = q_s_548 * ev_551;
            float _cvt_f32_58 = (float)kb_552;
            vk_528[2] = _cvt_f32_58 * ev_551;
            float r_553[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_553[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_554 = r_553[0];
            unsigned int r_555[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_555[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_556 = __uint_as_float(r_555[0] << 16);
            unsigned int r_557[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_557[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_558 = __uint_as_float(r_557[0] << 16);
            float _exp2_59 = approx_exp2(g_s_554 - a_d_521);
            float ev_559 = _exp2_59;
            __nv_bfloat16 _cvt_bf16_59 = (__nv_bfloat16)(k_s_558 * bv_526[3]);
            __nv_bfloat16 kb_560 = _cvt_bf16_59;
            vq_527[3] = q_s_556 * ev_559;
            float _cvt_f32_59 = (float)kb_560;
            vk_528[3] = _cvt_f32_59 * ev_559;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_527[0]), "f"(vq_527[1]), "f"(vq_527[2]), "f"(vq_527[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_528[0]), "f"(vk_528[1]), "f"(vk_528[2]), "f"(vk_528[3]) : "memory");
            float bv_561[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_561[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_561[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_561[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_561[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_522 * 4 + 16)));
            float vq_562[4];
            float vk_563[4];
            float r_564[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_564[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_565 = r_564[0];
            unsigned int r_566[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_566[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_567 = __uint_as_float(r_566[0] << 16);
            unsigned int r_568[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_568[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_569 = __uint_as_float(r_568[0] << 16);
            float _exp2_60 = approx_exp2(g_s_565 - a_d_521);
            float ev_570 = _exp2_60;
            __nv_bfloat16 _cvt_bf16_60 = (__nv_bfloat16)(k_s_569 * bv_561[0]);
            __nv_bfloat16 kb_571 = _cvt_bf16_60;
            vq_562[0] = q_s_567 * ev_570;
            float _cvt_f32_60 = (float)kb_571;
            vk_563[0] = _cvt_f32_60 * ev_570;
            float r_572[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_572[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_573 = r_572[0];
            unsigned int r_574[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_574[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_575 = __uint_as_float(r_574[0] << 16);
            unsigned int r_576[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_576[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_577 = __uint_as_float(r_576[0] << 16);
            float _exp2_61 = approx_exp2(g_s_573 - a_d_521);
            float ev_578 = _exp2_61;
            __nv_bfloat16 _cvt_bf16_61 = (__nv_bfloat16)(k_s_577 * bv_561[1]);
            __nv_bfloat16 kb_579 = _cvt_bf16_61;
            vq_562[1] = q_s_575 * ev_578;
            float _cvt_f32_61 = (float)kb_579;
            vk_563[1] = _cvt_f32_61 * ev_578;
            float r_580[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_580[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_581 = r_580[0];
            unsigned int r_582[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_582[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_583 = __uint_as_float(r_582[0] << 16);
            unsigned int r_584[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_584[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_585 = __uint_as_float(r_584[0] << 16);
            float _exp2_62 = approx_exp2(g_s_581 - a_d_521);
            float ev_586 = _exp2_62;
            __nv_bfloat16 _cvt_bf16_62 = (__nv_bfloat16)(k_s_585 * bv_561[2]);
            __nv_bfloat16 kb_587 = _cvt_bf16_62;
            vq_562[2] = q_s_583 * ev_586;
            float _cvt_f32_62 = (float)kb_587;
            vk_563[2] = _cvt_f32_62 * ev_586;
            float r_588[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_588[0])) : "r"((gbase_523 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_589 = r_588[0];
            unsigned int r_590[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_590[0])) : "r"((qbase_525 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_591 = __uint_as_float(r_590[0] << 16);
            unsigned int r_592[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_592[0])) : "r"((kbase_524 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_593 = __uint_as_float(r_592[0] << 16);
            float _exp2_63 = approx_exp2(g_s_589 - a_d_521);
            float ev_594 = _exp2_63;
            __nv_bfloat16 _cvt_bf16_63 = (__nv_bfloat16)(k_s_593 * bv_561[3]);
            __nv_bfloat16 kb_595 = _cvt_bf16_63;
            vq_562[3] = q_s_591 * ev_594;
            float _cvt_f32_63 = (float)kb_595;
            vk_563[3] = _cvt_f32_63 * ev_594;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_562[0]), "f"(vq_562[1]), "f"(vq_562[2]), "f"(vq_562[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_563[0]), "f"(vk_563[1]), "f"(vk_563[2]), "f"(vk_563[3]) : "memory");
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
            float r_596[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_596[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_597 = r_596[0];
            int sbase_598 = 48 + 8 * th;
            int gbase_599 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_598 * 128);
            int kbase_600 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_598 * 128);
            int qbase_601 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_602[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_602[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_602[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_602[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_602[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_598 * 4)));
            float vq_603[4];
            float vk_604[4];
            float r_605[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_605[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_606 = r_605[0];
            unsigned int r_607[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_607[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_608 = __uint_as_float(r_607[0] << 16);
            unsigned int r_609[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_609[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_610 = __uint_as_float(r_609[0] << 16);
            float _exp2_64 = approx_exp2(g_s_606 - a_d_597);
            float ev_611 = _exp2_64;
            __nv_bfloat16 _cvt_bf16_64 = (__nv_bfloat16)(k_s_610 * bv_602[0]);
            __nv_bfloat16 kb_612 = _cvt_bf16_64;
            vq_603[0] = q_s_608 * ev_611;
            float _cvt_f32_64 = (float)kb_612;
            vk_604[0] = _cvt_f32_64 * ev_611;
            float r_613[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_613[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_614 = r_613[0];
            unsigned int r_615[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_615[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_616 = __uint_as_float(r_615[0] << 16);
            unsigned int r_617[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_617[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_618 = __uint_as_float(r_617[0] << 16);
            float _exp2_65 = approx_exp2(g_s_614 - a_d_597);
            float ev_619 = _exp2_65;
            __nv_bfloat16 _cvt_bf16_65 = (__nv_bfloat16)(k_s_618 * bv_602[1]);
            __nv_bfloat16 kb_620 = _cvt_bf16_65;
            vq_603[1] = q_s_616 * ev_619;
            float _cvt_f32_65 = (float)kb_620;
            vk_604[1] = _cvt_f32_65 * ev_619;
            float r_621[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_621[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_622 = r_621[0];
            unsigned int r_623[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_623[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_624 = __uint_as_float(r_623[0] << 16);
            unsigned int r_625[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_625[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_626 = __uint_as_float(r_625[0] << 16);
            float _exp2_66 = approx_exp2(g_s_622 - a_d_597);
            float ev_627 = _exp2_66;
            __nv_bfloat16 _cvt_bf16_66 = (__nv_bfloat16)(k_s_626 * bv_602[2]);
            __nv_bfloat16 kb_628 = _cvt_bf16_66;
            vq_603[2] = q_s_624 * ev_627;
            float _cvt_f32_66 = (float)kb_628;
            vk_604[2] = _cvt_f32_66 * ev_627;
            float r_629[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_629[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_630 = r_629[0];
            unsigned int r_631[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_631[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_632 = __uint_as_float(r_631[0] << 16);
            unsigned int r_633[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_633[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_634 = __uint_as_float(r_633[0] << 16);
            float _exp2_67 = approx_exp2(g_s_630 - a_d_597);
            float ev_635 = _exp2_67;
            __nv_bfloat16 _cvt_bf16_67 = (__nv_bfloat16)(k_s_634 * bv_602[3]);
            __nv_bfloat16 kb_636 = _cvt_bf16_67;
            vq_603[3] = q_s_632 * ev_635;
            float _cvt_f32_67 = (float)kb_636;
            vk_604[3] = _cvt_f32_67 * ev_635;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_603[0]), "f"(vq_603[1]), "f"(vq_603[2]), "f"(vq_603[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_604[0]), "f"(vk_604[1]), "f"(vk_604[2]), "f"(vk_604[3]) : "memory");
            float bv_637[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_637[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_637[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_637[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_637[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_598 * 4 + 16)));
            float vq_638[4];
            float vk_639[4];
            float r_640[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_640[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_641 = r_640[0];
            unsigned int r_642[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_642[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_643 = __uint_as_float(r_642[0] << 16);
            unsigned int r_644[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_644[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_645 = __uint_as_float(r_644[0] << 16);
            float _exp2_68 = approx_exp2(g_s_641 - a_d_597);
            float ev_646 = _exp2_68;
            __nv_bfloat16 _cvt_bf16_68 = (__nv_bfloat16)(k_s_645 * bv_637[0]);
            __nv_bfloat16 kb_647 = _cvt_bf16_68;
            vq_638[0] = q_s_643 * ev_646;
            float _cvt_f32_68 = (float)kb_647;
            vk_639[0] = _cvt_f32_68 * ev_646;
            float r_648[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_648[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_649 = r_648[0];
            unsigned int r_650[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_650[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_651 = __uint_as_float(r_650[0] << 16);
            unsigned int r_652[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_652[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_653 = __uint_as_float(r_652[0] << 16);
            float _exp2_69 = approx_exp2(g_s_649 - a_d_597);
            float ev_654 = _exp2_69;
            __nv_bfloat16 _cvt_bf16_69 = (__nv_bfloat16)(k_s_653 * bv_637[1]);
            __nv_bfloat16 kb_655 = _cvt_bf16_69;
            vq_638[1] = q_s_651 * ev_654;
            float _cvt_f32_69 = (float)kb_655;
            vk_639[1] = _cvt_f32_69 * ev_654;
            float r_656[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_656[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_657 = r_656[0];
            unsigned int r_658[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_658[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_659 = __uint_as_float(r_658[0] << 16);
            unsigned int r_660[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_660[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_661 = __uint_as_float(r_660[0] << 16);
            float _exp2_70 = approx_exp2(g_s_657 - a_d_597);
            float ev_662 = _exp2_70;
            __nv_bfloat16 _cvt_bf16_70 = (__nv_bfloat16)(k_s_661 * bv_637[2]);
            __nv_bfloat16 kb_663 = _cvt_bf16_70;
            vq_638[2] = q_s_659 * ev_662;
            float _cvt_f32_70 = (float)kb_663;
            vk_639[2] = _cvt_f32_70 * ev_662;
            float r_664[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_664[0])) : "r"((gbase_599 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_665 = r_664[0];
            unsigned int r_666[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_666[0])) : "r"((qbase_601 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_667 = __uint_as_float(r_666[0] << 16);
            unsigned int r_668[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_668[0])) : "r"((kbase_600 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_669 = __uint_as_float(r_668[0] << 16);
            float _exp2_71 = approx_exp2(g_s_665 - a_d_597);
            float ev_670 = _exp2_71;
            __nv_bfloat16 _cvt_bf16_71 = (__nv_bfloat16)(k_s_669 * bv_637[3]);
            __nv_bfloat16 kb_671 = _cvt_bf16_71;
            vq_638[3] = q_s_667 * ev_670;
            float _cvt_f32_71 = (float)kb_671;
            vk_639[3] = _cvt_f32_71 * ev_670;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_638[0]), "f"(vq_638[1]), "f"(vq_638[2]), "f"(vq_638[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_639[0]), "f"(vk_639[1]), "f"(vk_639[2]), "f"(vk_639[3]) : "memory");
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
            float r_672[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_672[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_673 = r_672[0] - 21.640425324000002f;
            int sbase_674 = 48 + 8 * th;
            int gbase_675 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_674 * 128);
            int kbase_676 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_674 * 128);
            int qbase_677 = pool_f32_addr + (unsigned int)(98304 + (2 + dh_lane) * 2048 + th * 1024);
            float bv_678[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_678[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_678[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_678[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_678[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_674 * 4)));
            float vq_679[4];
            float vk_680[4];
            float r_681[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_681[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_682 = r_681[0];
            unsigned int r_683[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_683[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_684 = __uint_as_float(r_683[0] << 16);
            unsigned int r_685[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_685[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_686 = __uint_as_float(r_685[0] << 16);
            float _exp2_72 = approx_exp2(g_s_682 - a_d_673);
            float ev_687 = _exp2_72;
            __nv_bfloat16 _cvt_bf16_72 = (__nv_bfloat16)(k_s_686 * bv_678[0]);
            __nv_bfloat16 kb_688 = _cvt_bf16_72;
            vq_679[0] = q_s_684 * ev_687;
            float _cvt_f32_72 = (float)kb_688;
            vk_680[0] = _cvt_f32_72 * ev_687;
            float r_689[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_689[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_690 = r_689[0];
            unsigned int r_691[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_691[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_692 = __uint_as_float(r_691[0] << 16);
            unsigned int r_693[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_693[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_694 = __uint_as_float(r_693[0] << 16);
            float _exp2_73 = approx_exp2(g_s_690 - a_d_673);
            float ev_695 = _exp2_73;
            __nv_bfloat16 _cvt_bf16_73 = (__nv_bfloat16)(k_s_694 * bv_678[1]);
            __nv_bfloat16 kb_696 = _cvt_bf16_73;
            vq_679[1] = q_s_692 * ev_695;
            float _cvt_f32_73 = (float)kb_696;
            vk_680[1] = _cvt_f32_73 * ev_695;
            float r_697[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_697[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_698 = r_697[0];
            unsigned int r_699[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_699[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_700 = __uint_as_float(r_699[0] << 16);
            unsigned int r_701[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_701[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_702 = __uint_as_float(r_701[0] << 16);
            float _exp2_74 = approx_exp2(g_s_698 - a_d_673);
            float ev_703 = _exp2_74;
            __nv_bfloat16 _cvt_bf16_74 = (__nv_bfloat16)(k_s_702 * bv_678[2]);
            __nv_bfloat16 kb_704 = _cvt_bf16_74;
            vq_679[2] = q_s_700 * ev_703;
            float _cvt_f32_74 = (float)kb_704;
            vk_680[2] = _cvt_f32_74 * ev_703;
            float r_705[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_705[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_706 = r_705[0];
            unsigned int r_707[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_707[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_708 = __uint_as_float(r_707[0] << 16);
            unsigned int r_709[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_709[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_710 = __uint_as_float(r_709[0] << 16);
            float _exp2_75 = approx_exp2(g_s_706 - a_d_673);
            float ev_711 = _exp2_75;
            __nv_bfloat16 _cvt_bf16_75 = (__nv_bfloat16)(k_s_710 * bv_678[3]);
            __nv_bfloat16 kb_712 = _cvt_bf16_75;
            vq_679[3] = q_s_708 * ev_711;
            float _cvt_f32_75 = (float)kb_712;
            vk_680[3] = _cvt_f32_75 * ev_711;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(vq_679[0]), "f"(vq_679[1]), "f"(vq_679[2]), "f"(vq_679[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_680[0]), "f"(vk_680[1]), "f"(vk_680[2]), "f"(vk_680[3]) : "memory");
            float bv_713[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bv_713[0])), "=r"(*reinterpret_cast<uint32_t*>(&bv_713[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bv_713[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bv_713[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + sbase_674 * 4 + 16)));
            float vq_714[4];
            float vk_715[4];
            float r_716[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_716[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_717 = r_716[0];
            unsigned int r_718[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_718[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_719 = __uint_as_float(r_718[0] << 16);
            unsigned int r_720[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_720[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_721 = __uint_as_float(r_720[0] << 16);
            float _exp2_76 = approx_exp2(g_s_717 - a_d_673);
            float ev_722 = _exp2_76;
            __nv_bfloat16 _cvt_bf16_76 = (__nv_bfloat16)(k_s_721 * bv_713[0]);
            __nv_bfloat16 kb_723 = _cvt_bf16_76;
            vq_714[0] = q_s_719 * ev_722;
            float _cvt_f32_76 = (float)kb_723;
            vk_715[0] = _cvt_f32_76 * ev_722;
            float r_724[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_724[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_725 = r_724[0];
            unsigned int r_726[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_726[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_727 = __uint_as_float(r_726[0] << 16);
            unsigned int r_728[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_728[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_729 = __uint_as_float(r_728[0] << 16);
            float _exp2_77 = approx_exp2(g_s_725 - a_d_673);
            float ev_730 = _exp2_77;
            __nv_bfloat16 _cvt_bf16_77 = (__nv_bfloat16)(k_s_729 * bv_713[1]);
            __nv_bfloat16 kb_731 = _cvt_bf16_77;
            vq_714[1] = q_s_727 * ev_730;
            float _cvt_f32_77 = (float)kb_731;
            vk_715[1] = _cvt_f32_77 * ev_730;
            float r_732[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_732[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_733 = r_732[0];
            unsigned int r_734[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_734[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_735 = __uint_as_float(r_734[0] << 16);
            unsigned int r_736[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_736[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_737 = __uint_as_float(r_736[0] << 16);
            float _exp2_78 = approx_exp2(g_s_733 - a_d_673);
            float ev_738 = _exp2_78;
            __nv_bfloat16 _cvt_bf16_78 = (__nv_bfloat16)(k_s_737 * bv_713[2]);
            __nv_bfloat16 kb_739 = _cvt_bf16_78;
            vq_714[2] = q_s_735 * ev_738;
            float _cvt_f32_78 = (float)kb_739;
            vk_715[2] = _cvt_f32_78 * ev_738;
            float r_740[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_740[0])) : "r"((gbase_675 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_s_741 = r_740[0];
            unsigned int r_742[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_742[0])) : "r"((qbase_677 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_s_743 = __uint_as_float(r_742[0] << 16);
            unsigned int r_744[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_744[0])) : "r"((kbase_676 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s_745 = __uint_as_float(r_744[0] << 16);
            float _exp2_79 = approx_exp2(g_s_741 - a_d_673);
            float ev_746 = _exp2_79;
            __nv_bfloat16 _cvt_bf16_79 = (__nv_bfloat16)(k_s_745 * bv_713[3]);
            __nv_bfloat16 kb_747 = _cvt_bf16_79;
            vq_714[3] = q_s_743 * ev_746;
            float _cvt_f32_79 = (float)kb_747;
            vk_715[3] = _cvt_f32_79 * ev_746;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vq_714[0]), "f"(vq_714[1]), "f"(vq_714[2]), "f"(vq_714[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(vk_715[0]), "f"(vk_715[1]), "f"(vk_715[2]), "f"(vk_715[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_748 = 8 * th;
            int gbase_749 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_748 * 128);
            int kbase_750 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_748 * 128);
            float r_751[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_751[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0 = r_751[0];
            float r_752[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_752[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1 = r_752[0];
            float gs[4];
            float ks[4];
            float r_753[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_753[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[0] = r_753[0];
            unsigned int r_754[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_754[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[0] = __uint_as_float(r_754[0] << 16);
            float r_755[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_755[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[1] = r_755[0];
            unsigned int r_756[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_756[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[1] = __uint_as_float(r_756[0] << 16);
            float r_757[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_757[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[2] = r_757[0];
            unsigned int r_758[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_758[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[2] = __uint_as_float(r_758[0] << 16);
            float r_759[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_759[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[3] = r_759[0];
            unsigned int r_760[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_760[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[3] = __uint_as_float(r_760[0] << 16);
            float v[4];
            float _exp2_80 = approx_exp2(a0 - gs[0]);
            v[0] = ks[0] * _exp2_80;
            float _exp2_81 = approx_exp2(a0 - gs[1]);
            v[1] = ks[1] * _exp2_81;
            float _exp2_82 = approx_exp2(a0 - gs[2]);
            v[2] = ks[2] * _exp2_82;
            float _exp2_83 = approx_exp2(a0 - gs[3]);
            v[3] = ks[3] * _exp2_83;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
            float v_761[4];
            float _exp2_84 = approx_exp2(a1 - gs[0]);
            v_761[0] = ks[0] * _exp2_84;
            float _exp2_85 = approx_exp2(a1 - gs[1]);
            v_761[1] = ks[1] * _exp2_85;
            float _exp2_86 = approx_exp2(a1 - gs[2]);
            v_761[2] = ks[2] * _exp2_86;
            float _exp2_87 = approx_exp2(a1 - gs[3]);
            v_761[3] = ks[3] * _exp2_87;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_761[0]), "f"(v_761[1]), "f"(v_761[2]), "f"(v_761[3]) : "memory");
            float gs_762[4];
            float ks_763[4];
            float r_764[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_764[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_762[0] = r_764[0];
            unsigned int r_765[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_765[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_763[0] = __uint_as_float(r_765[0] << 16);
            float r_766[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_766[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_762[1] = r_766[0];
            unsigned int r_767[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_767[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_763[1] = __uint_as_float(r_767[0] << 16);
            float r_768[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_768[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_762[2] = r_768[0];
            unsigned int r_769[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_769[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_763[2] = __uint_as_float(r_769[0] << 16);
            float r_770[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_770[0])) : "r"((gbase_749 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_762[3] = r_770[0];
            unsigned int r_771[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_771[0])) : "r"((kbase_750 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_763[3] = __uint_as_float(r_771[0] << 16);
            float v_772[4];
            float _exp2_88 = approx_exp2(a0 - gs_762[0]);
            v_772[0] = ks_763[0] * _exp2_88;
            float _exp2_89 = approx_exp2(a0 - gs_762[1]);
            v_772[1] = ks_763[1] * _exp2_89;
            float _exp2_90 = approx_exp2(a0 - gs_762[2]);
            v_772[2] = ks_763[2] * _exp2_90;
            float _exp2_91 = approx_exp2(a0 - gs_762[3]);
            v_772[3] = ks_763[3] * _exp2_91;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_772[0]), "f"(v_772[1]), "f"(v_772[2]), "f"(v_772[3]) : "memory");
            float v_773[4];
            float _exp2_92 = approx_exp2(a1 - gs_762[0]);
            v_773[0] = ks_763[0] * _exp2_92;
            float _exp2_93 = approx_exp2(a1 - gs_762[1]);
            v_773[1] = ks_763[1] * _exp2_93;
            float _exp2_94 = approx_exp2(a1 - gs_762[2]);
            v_773[2] = ks_763[2] * _exp2_94;
            float _exp2_95 = approx_exp2(a1 - gs_762[3]);
            v_773[3] = ks_763[3] * _exp2_95;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_773[0]), "f"(v_773[1]), "f"(v_773[2]), "f"(v_773[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_774 = 16 + 8 * th;
            int gbase_775 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_774 * 128);
            int kbase_776 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_774 * 128);
            float r_777[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_777[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_778 = r_777[0];
            float r_779[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_779[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1_780 = r_779[0];
            float gs_781[4];
            float ks_782[4];
            float r_783[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_783[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_781[0] = r_783[0];
            unsigned int r_784[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_784[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_782[0] = __uint_as_float(r_784[0] << 16);
            float r_785[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_785[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_781[1] = r_785[0];
            unsigned int r_786[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_786[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_782[1] = __uint_as_float(r_786[0] << 16);
            float r_787[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_787[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_781[2] = r_787[0];
            unsigned int r_788[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_788[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_782[2] = __uint_as_float(r_788[0] << 16);
            float r_789[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_789[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_781[3] = r_789[0];
            unsigned int r_790[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_790[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_782[3] = __uint_as_float(r_790[0] << 16);
            float v_791[4];
            float _exp2_96 = approx_exp2(a0_778 - gs_781[0]);
            v_791[0] = ks_782[0] * _exp2_96;
            float _exp2_97 = approx_exp2(a0_778 - gs_781[1]);
            v_791[1] = ks_782[1] * _exp2_97;
            float _exp2_98 = approx_exp2(a0_778 - gs_781[2]);
            v_791[2] = ks_782[2] * _exp2_98;
            float _exp2_99 = approx_exp2(a0_778 - gs_781[3]);
            v_791[3] = ks_782[3] * _exp2_99;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_791[0]), "f"(v_791[1]), "f"(v_791[2]), "f"(v_791[3]) : "memory");
            float v_792[4];
            float _exp2_100 = approx_exp2(a1_780 - gs_781[0]);
            v_792[0] = ks_782[0] * _exp2_100;
            float _exp2_101 = approx_exp2(a1_780 - gs_781[1]);
            v_792[1] = ks_782[1] * _exp2_101;
            float _exp2_102 = approx_exp2(a1_780 - gs_781[2]);
            v_792[2] = ks_782[2] * _exp2_102;
            float _exp2_103 = approx_exp2(a1_780 - gs_781[3]);
            v_792[3] = ks_782[3] * _exp2_103;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_792[0]), "f"(v_792[1]), "f"(v_792[2]), "f"(v_792[3]) : "memory");
            float gs_793[4];
            float ks_794[4];
            float r_795[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_795[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_793[0] = r_795[0];
            unsigned int r_796[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_796[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_794[0] = __uint_as_float(r_796[0] << 16);
            float r_797[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_797[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_793[1] = r_797[0];
            unsigned int r_798[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_798[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_794[1] = __uint_as_float(r_798[0] << 16);
            float r_799[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_799[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_793[2] = r_799[0];
            unsigned int r_800[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_800[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_794[2] = __uint_as_float(r_800[0] << 16);
            float r_801[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_801[0])) : "r"((gbase_775 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_793[3] = r_801[0];
            unsigned int r_802[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_802[0])) : "r"((kbase_776 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_794[3] = __uint_as_float(r_802[0] << 16);
            float v_803[4];
            float _exp2_104 = approx_exp2(a0_778 - gs_793[0]);
            v_803[0] = ks_794[0] * _exp2_104;
            float _exp2_105 = approx_exp2(a0_778 - gs_793[1]);
            v_803[1] = ks_794[1] * _exp2_105;
            float _exp2_106 = approx_exp2(a0_778 - gs_793[2]);
            v_803[2] = ks_794[2] * _exp2_106;
            float _exp2_107 = approx_exp2(a0_778 - gs_793[3]);
            v_803[3] = ks_794[3] * _exp2_107;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_803[0]), "f"(v_803[1]), "f"(v_803[2]), "f"(v_803[3]) : "memory");
            float v_804[4];
            float _exp2_108 = approx_exp2(a1_780 - gs_793[0]);
            v_804[0] = ks_794[0] * _exp2_108;
            float _exp2_109 = approx_exp2(a1_780 - gs_793[1]);
            v_804[1] = ks_794[1] * _exp2_109;
            float _exp2_110 = approx_exp2(a1_780 - gs_793[2]);
            v_804[2] = ks_794[2] * _exp2_110;
            float _exp2_111 = approx_exp2(a1_780 - gs_793[3]);
            v_804[3] = ks_794[3] * _exp2_111;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_804[0]), "f"(v_804[1]), "f"(v_804[2]), "f"(v_804[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_805 = 32 + 8 * th;
            int gbase_806 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_805 * 128);
            int kbase_807 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_805 * 128);
            float r_808[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_808[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_809 = r_808[0];
            float a1_810 = 0.0f;
            float gs_811[4];
            float ks_812[4];
            float r_813[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_813[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_811[0] = r_813[0];
            unsigned int r_814[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_814[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_812[0] = __uint_as_float(r_814[0] << 16);
            float r_815[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_815[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_811[1] = r_815[0];
            unsigned int r_816[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_816[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_812[1] = __uint_as_float(r_816[0] << 16);
            float r_817[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_817[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_811[2] = r_817[0];
            unsigned int r_818[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_818[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_812[2] = __uint_as_float(r_818[0] << 16);
            float r_819[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_819[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_811[3] = r_819[0];
            unsigned int r_820[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_820[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_812[3] = __uint_as_float(r_820[0] << 16);
            float v_821[4];
            float _exp2_112 = approx_exp2(a0_809 - gs_811[0]);
            v_821[0] = ks_812[0] * _exp2_112;
            float _exp2_113 = approx_exp2(a0_809 - gs_811[1]);
            v_821[1] = ks_812[1] * _exp2_113;
            float _exp2_114 = approx_exp2(a0_809 - gs_811[2]);
            v_821[2] = ks_812[2] * _exp2_114;
            float _exp2_115 = approx_exp2(a0_809 - gs_811[3]);
            v_821[3] = ks_812[3] * _exp2_115;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_821[0]), "f"(v_821[1]), "f"(v_821[2]), "f"(v_821[3]) : "memory");
            float gs_822[4];
            float ks_823[4];
            float r_824[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_824[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_822[0] = r_824[0];
            unsigned int r_825[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_825[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_823[0] = __uint_as_float(r_825[0] << 16);
            float r_826[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_826[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_822[1] = r_826[0];
            unsigned int r_827[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_827[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_823[1] = __uint_as_float(r_827[0] << 16);
            float r_828[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_828[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_822[2] = r_828[0];
            unsigned int r_829[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_829[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_823[2] = __uint_as_float(r_829[0] << 16);
            float r_830[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_830[0])) : "r"((gbase_806 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_822[3] = r_830[0];
            unsigned int r_831[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_831[0])) : "r"((kbase_807 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_823[3] = __uint_as_float(r_831[0] << 16);
            float v_832[4];
            float _exp2_116 = approx_exp2(a0_809 - gs_822[0]);
            v_832[0] = ks_823[0] * _exp2_116;
            float _exp2_117 = approx_exp2(a0_809 - gs_822[1]);
            v_832[1] = ks_823[1] * _exp2_117;
            float _exp2_118 = approx_exp2(a0_809 - gs_822[2]);
            v_832[2] = ks_823[2] * _exp2_118;
            float _exp2_119 = approx_exp2(a0_809 - gs_822[3]);
            v_832[3] = ks_823[3] * _exp2_119;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_832[0]), "f"(v_832[1]), "f"(v_832[2]), "f"(v_832[3]) : "memory");
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
            float r_833[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_833[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl = r_833[0];
            float r_834[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_834[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc = r_834[0] - 21.640425324000002f;
            float dkt[8];
            float pan[8];
            int t = 32 + 8 * th;
            float r_835[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_835[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t = r_835[0];
            unsigned int r_836[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_836[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t = __uint_as_float(r_836[0] << 16);
            float _exp2_120 = approx_exp2(gc - g_t);
            float e2 = _exp2_120;
            float _exp2_121 = approx_exp2(gnl - g_t);
            dkt[0] = __fmaf_rn(_tmem_load_1[0], _exp2_121, _tmem_load_0[0] * e2);
            pan[0] = k_t * e2;
            int t_837 = 32 + 8 * th + 1;
            float r_838[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_838[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_837 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_837 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_839 = r_838[0];
            unsigned int r_840[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_840[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_837 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_837 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_841 = __uint_as_float(r_840[0] << 16);
            float _exp2_122 = approx_exp2(gc - g_t_839);
            float e2_842 = _exp2_122;
            float _exp2_123 = approx_exp2(gnl - g_t_839);
            dkt[1] = __fmaf_rn(_tmem_load_1[1], _exp2_123, _tmem_load_0[1] * e2_842);
            pan[1] = k_t_841 * e2_842;
            int t_843 = 32 + 8 * th + 2;
            float r_844[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_844[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_843 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_843 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_845 = r_844[0];
            unsigned int r_846[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_846[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_843 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_843 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_847 = __uint_as_float(r_846[0] << 16);
            float _exp2_124 = approx_exp2(gc - g_t_845);
            float e2_848 = _exp2_124;
            float _exp2_125 = approx_exp2(gnl - g_t_845);
            dkt[2] = __fmaf_rn(_tmem_load_1[2], _exp2_125, _tmem_load_0[2] * e2_848);
            pan[2] = k_t_847 * e2_848;
            int t_849 = 32 + 8 * th + 3;
            float r_850[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_850[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_849 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_849 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_851 = r_850[0];
            unsigned int r_852[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_852[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_849 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_849 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_853 = __uint_as_float(r_852[0] << 16);
            float _exp2_126 = approx_exp2(gc - g_t_851);
            float e2_854 = _exp2_126;
            float _exp2_127 = approx_exp2(gnl - g_t_851);
            dkt[3] = __fmaf_rn(_tmem_load_1[3], _exp2_127, _tmem_load_0[3] * e2_854);
            pan[3] = k_t_853 * e2_854;
            int t_855 = 32 + 8 * th + 4;
            float r_856[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_856[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_855 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_855 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_857 = r_856[0];
            unsigned int r_858[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_858[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_855 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_855 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_859 = __uint_as_float(r_858[0] << 16);
            float _exp2_128 = approx_exp2(gc - g_t_857);
            float e2_860 = _exp2_128;
            float _exp2_129 = approx_exp2(gnl - g_t_857);
            dkt[4] = __fmaf_rn(_tmem_load_1[4], _exp2_129, _tmem_load_0[4] * e2_860);
            pan[4] = k_t_859 * e2_860;
            int t_861 = 32 + 8 * th + 5;
            float r_862[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_862[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_861 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_861 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_863 = r_862[0];
            unsigned int r_864[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_864[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_861 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_861 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_865 = __uint_as_float(r_864[0] << 16);
            float _exp2_130 = approx_exp2(gc - g_t_863);
            float e2_866 = _exp2_130;
            float _exp2_131 = approx_exp2(gnl - g_t_863);
            dkt[5] = __fmaf_rn(_tmem_load_1[5], _exp2_131, _tmem_load_0[5] * e2_866);
            pan[5] = k_t_865 * e2_866;
            int t_867 = 32 + 8 * th + 6;
            float r_868[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_868[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_867 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_867 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_869 = r_868[0];
            unsigned int r_870[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_870[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_867 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_867 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_871 = __uint_as_float(r_870[0] << 16);
            float _exp2_132 = approx_exp2(gc - g_t_869);
            float e2_872 = _exp2_132;
            float _exp2_133 = approx_exp2(gnl - g_t_869);
            dkt[6] = __fmaf_rn(_tmem_load_1[6], _exp2_133, _tmem_load_0[6] * e2_872);
            pan[6] = k_t_871 * e2_872;
            int t_873 = 32 + 8 * th + 7;
            float r_874[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_874[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_873 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_873 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_875 = r_874[0];
            unsigned int r_876[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_876[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_873 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_873 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_877 = __uint_as_float(r_876[0] << 16);
            float _exp2_134 = approx_exp2(gc - g_t_875);
            float e2_878 = _exp2_134;
            float _exp2_135 = approx_exp2(gnl - g_t_875);
            dkt[7] = __fmaf_rn(_tmem_load_1[7], _exp2_135, _tmem_load_0[7] * e2_878);
            pan[7] = k_t_877 * e2_878;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th), dkt);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan[0]), "f"(pan[1]), "f"(pan[2]), "f"(pan[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan[4]), "f"(pan[5]), "f"(pan[6]), "f"(pan[7]) : "memory");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (48 + 8 * th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_879[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_879[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_880 = r_879[0];
            float r_881[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_881[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_882 = r_881[0] - 21.640425324000002f;
            float dkt_883[8];
            float pan_884[8];
            int t_885 = 48 + 8 * th;
            float r_886[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_886[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_885 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_885 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_887 = r_886[0];
            unsigned int r_888[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_888[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_885 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_885 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_889 = __uint_as_float(r_888[0] << 16);
            float _exp2_136 = approx_exp2(gc_882 - g_t_887);
            float e2_890 = _exp2_136;
            dkt_883[0] = _tmem_load_2[0] * e2_890;
            pan_884[0] = k_t_889 * e2_890;
            int t_891 = 48 + 8 * th + 1;
            float r_892[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_892[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_891 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_891 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_893 = r_892[0];
            unsigned int r_894[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_894[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_891 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_891 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_895 = __uint_as_float(r_894[0] << 16);
            float _exp2_137 = approx_exp2(gc_882 - g_t_893);
            float e2_896 = _exp2_137;
            dkt_883[1] = _tmem_load_2[1] * e2_896;
            pan_884[1] = k_t_895 * e2_896;
            int t_897 = 48 + 8 * th + 2;
            float r_898[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_898[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_897 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_897 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_899 = r_898[0];
            unsigned int r_900[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_900[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_897 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_897 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_901 = __uint_as_float(r_900[0] << 16);
            float _exp2_138 = approx_exp2(gc_882 - g_t_899);
            float e2_902 = _exp2_138;
            dkt_883[2] = _tmem_load_2[2] * e2_902;
            pan_884[2] = k_t_901 * e2_902;
            int t_903 = 48 + 8 * th + 3;
            float r_904[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_904[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_903 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_903 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_905 = r_904[0];
            unsigned int r_906[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_906[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_903 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_903 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_907 = __uint_as_float(r_906[0] << 16);
            float _exp2_139 = approx_exp2(gc_882 - g_t_905);
            float e2_908 = _exp2_139;
            dkt_883[3] = _tmem_load_2[3] * e2_908;
            pan_884[3] = k_t_907 * e2_908;
            int t_909 = 48 + 8 * th + 4;
            float r_910[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_910[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_909 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_909 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_911 = r_910[0];
            unsigned int r_912[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_912[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_909 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_909 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_913 = __uint_as_float(r_912[0] << 16);
            float _exp2_140 = approx_exp2(gc_882 - g_t_911);
            float e2_914 = _exp2_140;
            dkt_883[4] = _tmem_load_2[4] * e2_914;
            pan_884[4] = k_t_913 * e2_914;
            int t_915 = 48 + 8 * th + 5;
            float r_916[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_916[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_915 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_915 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_917 = r_916[0];
            unsigned int r_918[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_918[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_915 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_915 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_919 = __uint_as_float(r_918[0] << 16);
            float _exp2_141 = approx_exp2(gc_882 - g_t_917);
            float e2_920 = _exp2_141;
            dkt_883[5] = _tmem_load_2[5] * e2_920;
            pan_884[5] = k_t_919 * e2_920;
            int t_921 = 48 + 8 * th + 6;
            float r_922[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_922[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_921 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_921 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_923 = r_922[0];
            unsigned int r_924[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_924[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_921 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_921 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_925 = __uint_as_float(r_924[0] << 16);
            float _exp2_142 = approx_exp2(gc_882 - g_t_923);
            float e2_926 = _exp2_142;
            dkt_883[6] = _tmem_load_2[6] * e2_926;
            pan_884[6] = k_t_925 * e2_926;
            int t_927 = 48 + 8 * th + 7;
            float r_928[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_928[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_927 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_927 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_929 = r_928[0];
            unsigned int r_930[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_930[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_927 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_927 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_931 = __uint_as_float(r_930[0] << 16);
            float _exp2_143 = approx_exp2(gc_882 - g_t_929);
            float e2_932 = _exp2_143;
            dkt_883[7] = _tmem_load_2[7] * e2_932;
            pan_884[7] = k_t_931 * e2_932;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th), dkt_883);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_884[0]), "f"(pan_884[1]), "f"(pan_884[2]), "f"(pan_884[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_884[4]), "f"(pan_884[5]), "f"(pan_884[6]), "f"(pan_884[7]) : "memory");
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
            float r_933[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_933[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_934 = r_933[0];
            float r_935[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_935[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_936 = r_935[0] - 21.640425324000002f;
            float dkt_937[8];
            float pan_938[8];
            int t_939 = 8 * th;
            float r_940[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_940[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_939 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_939 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_941 = r_940[0];
            unsigned int r_942[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_942[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_939 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_939 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_943 = __uint_as_float(r_942[0] << 16);
            float _exp2_144 = approx_exp2(gc_936 - g_t_941);
            float e2_944 = _exp2_144;
            float _exp2_145 = approx_exp2(gnl_934 - g_t_941);
            dkt_937[0] = __fmaf_rn(_tmem_load_4[0], _exp2_145, _tmem_load_3[0] * e2_944);
            pan_938[0] = k_t_943 * e2_944;
            int t_945 = 8 * th + 1;
            float r_946[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_946[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_945 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_945 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_947 = r_946[0];
            unsigned int r_948[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_948[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_945 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_945 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_949 = __uint_as_float(r_948[0] << 16);
            float _exp2_146 = approx_exp2(gc_936 - g_t_947);
            float e2_950 = _exp2_146;
            float _exp2_147 = approx_exp2(gnl_934 - g_t_947);
            dkt_937[1] = __fmaf_rn(_tmem_load_4[1], _exp2_147, _tmem_load_3[1] * e2_950);
            pan_938[1] = k_t_949 * e2_950;
            int t_951 = 8 * th + 2;
            float r_952[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_952[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_951 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_951 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_953 = r_952[0];
            unsigned int r_954[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_954[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_951 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_951 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_955 = __uint_as_float(r_954[0] << 16);
            float _exp2_148 = approx_exp2(gc_936 - g_t_953);
            float e2_956 = _exp2_148;
            float _exp2_149 = approx_exp2(gnl_934 - g_t_953);
            dkt_937[2] = __fmaf_rn(_tmem_load_4[2], _exp2_149, _tmem_load_3[2] * e2_956);
            pan_938[2] = k_t_955 * e2_956;
            int t_957 = 8 * th + 3;
            float r_958[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_958[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_957 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_957 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_959 = r_958[0];
            unsigned int r_960[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_960[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_957 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_957 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_961 = __uint_as_float(r_960[0] << 16);
            float _exp2_150 = approx_exp2(gc_936 - g_t_959);
            float e2_962 = _exp2_150;
            float _exp2_151 = approx_exp2(gnl_934 - g_t_959);
            dkt_937[3] = __fmaf_rn(_tmem_load_4[3], _exp2_151, _tmem_load_3[3] * e2_962);
            pan_938[3] = k_t_961 * e2_962;
            int t_963 = 8 * th + 4;
            float r_964[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_964[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_963 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_963 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_965 = r_964[0];
            unsigned int r_966[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_966[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_963 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_963 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_967 = __uint_as_float(r_966[0] << 16);
            float _exp2_152 = approx_exp2(gc_936 - g_t_965);
            float e2_968 = _exp2_152;
            float _exp2_153 = approx_exp2(gnl_934 - g_t_965);
            dkt_937[4] = __fmaf_rn(_tmem_load_4[4], _exp2_153, _tmem_load_3[4] * e2_968);
            pan_938[4] = k_t_967 * e2_968;
            int t_969 = 8 * th + 5;
            float r_970[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_970[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_969 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_969 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_971 = r_970[0];
            unsigned int r_972[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_972[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_969 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_969 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_973 = __uint_as_float(r_972[0] << 16);
            float _exp2_154 = approx_exp2(gc_936 - g_t_971);
            float e2_974 = _exp2_154;
            float _exp2_155 = approx_exp2(gnl_934 - g_t_971);
            dkt_937[5] = __fmaf_rn(_tmem_load_4[5], _exp2_155, _tmem_load_3[5] * e2_974);
            pan_938[5] = k_t_973 * e2_974;
            int t_975 = 8 * th + 6;
            float r_976[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_976[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_975 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_975 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_977 = r_976[0];
            unsigned int r_978[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_978[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_975 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_975 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_979 = __uint_as_float(r_978[0] << 16);
            float _exp2_156 = approx_exp2(gc_936 - g_t_977);
            float e2_980 = _exp2_156;
            float _exp2_157 = approx_exp2(gnl_934 - g_t_977);
            dkt_937[6] = __fmaf_rn(_tmem_load_4[6], _exp2_157, _tmem_load_3[6] * e2_980);
            pan_938[6] = k_t_979 * e2_980;
            int t_981 = 8 * th + 7;
            float r_982[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_982[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_981 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_981 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_983 = r_982[0];
            unsigned int r_984[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_984[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_981 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_981 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_985 = __uint_as_float(r_984[0] << 16);
            float _exp2_158 = approx_exp2(gc_936 - g_t_983);
            float e2_986 = _exp2_158;
            float _exp2_159 = approx_exp2(gnl_934 - g_t_983);
            dkt_937[7] = __fmaf_rn(_tmem_load_4[7], _exp2_159, _tmem_load_3[7] * e2_986);
            pan_938[7] = k_t_985 * e2_986;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(8 * th), dkt_937);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan_938[0]), "f"(pan_938[1]), "f"(pan_938[2]), "f"(pan_938[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_938[4]), "f"(pan_938[5]), "f"(pan_938[6]), "f"(pan_938[7]) : "memory");
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (16 + 8 * th)));
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_987[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_987[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_988 = r_987[0];
            float r_989[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_989[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_990 = r_989[0] - 21.640425324000002f;
            float dkt_991[8];
            float pan_992[8];
            int t_993 = 16 + 8 * th;
            float r_994[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_994[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_993 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_993 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_995 = r_994[0];
            unsigned int r_996[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_996[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_993 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_993 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_997 = __uint_as_float(r_996[0] << 16);
            float _exp2_160 = approx_exp2(gc_990 - g_t_995);
            float e2_998 = _exp2_160;
            float _exp2_161 = approx_exp2(gnl_988 - g_t_995);
            dkt_991[0] = __fmaf_rn(_tmem_load_6[0], _exp2_161, _tmem_load_5[0] * e2_998);
            pan_992[0] = k_t_997 * e2_998;
            int t_999 = 16 + 8 * th + 1;
            float r_1000[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1000[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_999 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_999 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1001 = r_1000[0];
            unsigned int r_1002[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1002[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_999 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_999 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1003 = __uint_as_float(r_1002[0] << 16);
            float _exp2_162 = approx_exp2(gc_990 - g_t_1001);
            float e2_1004 = _exp2_162;
            float _exp2_163 = approx_exp2(gnl_988 - g_t_1001);
            dkt_991[1] = __fmaf_rn(_tmem_load_6[1], _exp2_163, _tmem_load_5[1] * e2_1004);
            pan_992[1] = k_t_1003 * e2_1004;
            int t_1005 = 16 + 8 * th + 2;
            float r_1006[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1006[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1005 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1005 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1007 = r_1006[0];
            unsigned int r_1008[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1008[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1005 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1005 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1009 = __uint_as_float(r_1008[0] << 16);
            float _exp2_164 = approx_exp2(gc_990 - g_t_1007);
            float e2_1010 = _exp2_164;
            float _exp2_165 = approx_exp2(gnl_988 - g_t_1007);
            dkt_991[2] = __fmaf_rn(_tmem_load_6[2], _exp2_165, _tmem_load_5[2] * e2_1010);
            pan_992[2] = k_t_1009 * e2_1010;
            int t_1011 = 16 + 8 * th + 3;
            float r_1012[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1012[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1011 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1011 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1013 = r_1012[0];
            unsigned int r_1014[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1014[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1011 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1011 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1015 = __uint_as_float(r_1014[0] << 16);
            float _exp2_166 = approx_exp2(gc_990 - g_t_1013);
            float e2_1016 = _exp2_166;
            float _exp2_167 = approx_exp2(gnl_988 - g_t_1013);
            dkt_991[3] = __fmaf_rn(_tmem_load_6[3], _exp2_167, _tmem_load_5[3] * e2_1016);
            pan_992[3] = k_t_1015 * e2_1016;
            int t_1017 = 16 + 8 * th + 4;
            float r_1018[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1018[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1017 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1017 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1019 = r_1018[0];
            unsigned int r_1020[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1020[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1017 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1017 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1021 = __uint_as_float(r_1020[0] << 16);
            float _exp2_168 = approx_exp2(gc_990 - g_t_1019);
            float e2_1022 = _exp2_168;
            float _exp2_169 = approx_exp2(gnl_988 - g_t_1019);
            dkt_991[4] = __fmaf_rn(_tmem_load_6[4], _exp2_169, _tmem_load_5[4] * e2_1022);
            pan_992[4] = k_t_1021 * e2_1022;
            int t_1023 = 16 + 8 * th + 5;
            float r_1024[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1024[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1023 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1023 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1025 = r_1024[0];
            unsigned int r_1026[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1026[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1023 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1023 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1027 = __uint_as_float(r_1026[0] << 16);
            float _exp2_170 = approx_exp2(gc_990 - g_t_1025);
            float e2_1028 = _exp2_170;
            float _exp2_171 = approx_exp2(gnl_988 - g_t_1025);
            dkt_991[5] = __fmaf_rn(_tmem_load_6[5], _exp2_171, _tmem_load_5[5] * e2_1028);
            pan_992[5] = k_t_1027 * e2_1028;
            int t_1029 = 16 + 8 * th + 6;
            float r_1030[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1030[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1029 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1029 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1031 = r_1030[0];
            unsigned int r_1032[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1032[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1029 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1029 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1033 = __uint_as_float(r_1032[0] << 16);
            float _exp2_172 = approx_exp2(gc_990 - g_t_1031);
            float e2_1034 = _exp2_172;
            float _exp2_173 = approx_exp2(gnl_988 - g_t_1031);
            dkt_991[6] = __fmaf_rn(_tmem_load_6[6], _exp2_173, _tmem_load_5[6] * e2_1034);
            pan_992[6] = k_t_1033 * e2_1034;
            int t_1035 = 16 + 8 * th + 7;
            float r_1036[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1036[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1035 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1035 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1037 = r_1036[0];
            unsigned int r_1038[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1038[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1035 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1035 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1039 = __uint_as_float(r_1038[0] << 16);
            float _exp2_174 = approx_exp2(gc_990 - g_t_1037);
            float e2_1040 = _exp2_174;
            float _exp2_175 = approx_exp2(gnl_988 - g_t_1037);
            dkt_991[7] = __fmaf_rn(_tmem_load_6[7], _exp2_175, _tmem_load_5[7] * e2_1040);
            pan_992[7] = k_t_1039 * e2_1040;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th), dkt_991);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_992[0]), "f"(pan_992[1]), "f"(pan_992[2]), "f"(pan_992[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_992[4]), "f"(pan_992[5]), "f"(pan_992[6]), "f"(pan_992[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_1041 = 8 * th;
            int gbase_1042 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_1041 * 128);
            int kbase_1043 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_1041 * 128);
            float r_1044[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1044[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_1045 = r_1044[0];
            float a1_1046 = 0.0f;
            float gs_1047[4];
            float ks_1048[4];
            float r_1049[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1049[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1047[0] = r_1049[0];
            unsigned int r_1050[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1050[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1048[0] = __uint_as_float(r_1050[0] << 16);
            float r_1051[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1051[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1047[1] = r_1051[0];
            unsigned int r_1052[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1052[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1048[1] = __uint_as_float(r_1052[0] << 16);
            float r_1053[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1053[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1047[2] = r_1053[0];
            unsigned int r_1054[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1054[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1048[2] = __uint_as_float(r_1054[0] << 16);
            float r_1055[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1055[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1047[3] = r_1055[0];
            unsigned int r_1056[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1056[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1048[3] = __uint_as_float(r_1056[0] << 16);
            float v_1057[4];
            float _exp2_176 = approx_exp2(a0_1045 - gs_1047[0]);
            v_1057[0] = ks_1048[0] * _exp2_176;
            float _exp2_177 = approx_exp2(a0_1045 - gs_1047[1]);
            v_1057[1] = ks_1048[1] * _exp2_177;
            float _exp2_178 = approx_exp2(a0_1045 - gs_1047[2]);
            v_1057[2] = ks_1048[2] * _exp2_178;
            float _exp2_179 = approx_exp2(a0_1045 - gs_1047[3]);
            v_1057[3] = ks_1048[3] * _exp2_179;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_1057[0]), "f"(v_1057[1]), "f"(v_1057[2]), "f"(v_1057[3]) : "memory");
            float gs_1058[4];
            float ks_1059[4];
            float r_1060[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1060[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1058[0] = r_1060[0];
            unsigned int r_1061[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1061[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1059[0] = __uint_as_float(r_1061[0] << 16);
            float r_1062[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1062[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1058[1] = r_1062[0];
            unsigned int r_1063[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1063[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1059[1] = __uint_as_float(r_1063[0] << 16);
            float r_1064[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1064[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1058[2] = r_1064[0];
            unsigned int r_1065[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1065[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1059[2] = __uint_as_float(r_1065[0] << 16);
            float r_1066[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1066[0])) : "r"((gbase_1042 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_1058[3] = r_1066[0];
            unsigned int r_1067[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1067[0])) : "r"((kbase_1043 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_1059[3] = __uint_as_float(r_1067[0] << 16);
            float v_1068[4];
            float _exp2_180 = approx_exp2(a0_1045 - gs_1058[0]);
            v_1068[0] = ks_1059[0] * _exp2_180;
            float _exp2_181 = approx_exp2(a0_1045 - gs_1058[1]);
            v_1068[1] = ks_1059[1] * _exp2_181;
            float _exp2_182 = approx_exp2(a0_1045 - gs_1058[2]);
            v_1068[2] = ks_1059[2] * _exp2_182;
            float _exp2_183 = approx_exp2(a0_1045 - gs_1058[3]);
            v_1068[3] = ks_1059[3] * _exp2_183;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_1068[0]), "f"(v_1068[1]), "f"(v_1068[2]), "f"(v_1068[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(p1a_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr, 0);
            mbarrier_wait(epi_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1069[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1069[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn = r_1069[0];
            float gc_1070 = gn - 21.640425324000002f;
            int eb = 98304 + th * -32768;
            int ew = eb >> 2;
            int qbase_1071 = pool_f32_addr + (unsigned int)(eb + 12288 + dh_lane * 1024);
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1072 = 32 + 8 * th;
            float r_1073[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1073[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1072 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1072 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1074 = r_1073[0];
            unsigned int r_1075[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1075[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1072 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1072 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1076 = __uint_as_float(r_1075[0] << 16);
            unsigned int r_1077[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1077[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t = __uint_as_float(r_1077[0] << 16);
            float q_t_1078 = ((t_1072 >= clen) ? 0.0f : q_t);
            float dqf = pool_f32[ew + d_lane];
            float dkf = pool_f32[ew + 1024 + d_lane];
            float dgf = pool_f32[ew + 2048 + d_lane];
            float bt_t = beta_s[t_1072];
            float _exp2_184 = approx_exp2(g_t_1074 - gn);
            float e_off = _exp2_184;
            float _exp2_185 = approx_exp2(g_t_1074 - gc_1070);
            float e_dg = _exp2_185;
            float dq2 = __fmaf_rn(_tmem_load_7[0], e_off, _tmem_load_9[0] * e_dg);
            float dk2 = __fmaf_rn(_tmem_load_8[0], e_off, _tmem_load_10[0] * e_dg);
            dbp[0] = dk2 * k_t_1076;
            float dk2b = dk2 * bt_t;
            float dq_o = dq2 + dqf;
            float dk_o = dk2b + dkf + _tmem_load_11[0];
            float dg_o = __fmaf_rn(q_t_1078, dq2, (dk2b - _tmem_load_11[0]) * k_t_1076) + dgf;
            int goff = t_1072 * rs_g;
            int qoff = t_1072 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff) + (0)) = dg_o;
            if (t_1072 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff) + (0)) = __float2bfloat16_rn(dq_o);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff) + (0)) = __float2bfloat16_rn(dk_o);
            }
            int t_1079 = 32 + 8 * th + 1;
            float r_1080[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1080[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1079 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1079 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1081 = r_1080[0];
            unsigned int r_1082[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1082[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1079 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1079 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1083 = __uint_as_float(r_1082[0] << 16);
            unsigned int r_1084[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1084[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1085 = __uint_as_float(r_1084[0] << 16);
            float q_t_1086 = ((t_1079 >= clen) ? 0.0f : q_t_1085);
            float dqf_1087 = pool_f32[ew + 128 + d_lane];
            float dkf_1088 = pool_f32[ew + 1152 + d_lane];
            float dgf_1089 = pool_f32[ew + 2176 + d_lane];
            float bt_t_1090 = beta_s[t_1079];
            float _exp2_186 = approx_exp2(g_t_1081 - gn);
            float e_off_1091 = _exp2_186;
            float _exp2_187 = approx_exp2(g_t_1081 - gc_1070);
            float e_dg_1092 = _exp2_187;
            float dq2_1093 = __fmaf_rn(_tmem_load_7[1], e_off_1091, _tmem_load_9[1] * e_dg_1092);
            float dk2_1094 = __fmaf_rn(_tmem_load_8[1], e_off_1091, _tmem_load_10[1] * e_dg_1092);
            dbp[1] = dk2_1094 * k_t_1083;
            float dk2b_1095 = dk2_1094 * bt_t_1090;
            float dq_o_1096 = dq2_1093 + dqf_1087;
            float dk_o_1097 = dk2b_1095 + dkf_1088 + _tmem_load_11[1];
            float dg_o_1098 = __fmaf_rn(q_t_1086, dq2_1093, (dk2b_1095 - _tmem_load_11[1]) * k_t_1083) + dgf_1089;
            int goff_1099 = t_1079 * rs_g;
            int qoff_1100 = t_1079 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1099) + (0)) = dg_o_1098;
            if (t_1079 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1100) + (0)) = __float2bfloat16_rn(dq_o_1096);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1100) + (0)) = __float2bfloat16_rn(dk_o_1097);
            }
            int t_1101 = 32 + 8 * th + 2;
            float r_1102[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1102[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1101 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1101 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1103 = r_1102[0];
            unsigned int r_1104[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1104[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1101 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1101 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1105 = __uint_as_float(r_1104[0] << 16);
            unsigned int r_1106[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1106[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1107 = __uint_as_float(r_1106[0] << 16);
            float q_t_1108 = ((t_1101 >= clen) ? 0.0f : q_t_1107);
            float dqf_1109 = pool_f32[ew + 256 + d_lane];
            float dkf_1110 = pool_f32[ew + 1280 + d_lane];
            float dgf_1111 = pool_f32[ew + 2304 + d_lane];
            float bt_t_1112 = beta_s[t_1101];
            float _exp2_188 = approx_exp2(g_t_1103 - gn);
            float e_off_1113 = _exp2_188;
            float _exp2_189 = approx_exp2(g_t_1103 - gc_1070);
            float e_dg_1114 = _exp2_189;
            float dq2_1115 = __fmaf_rn(_tmem_load_7[2], e_off_1113, _tmem_load_9[2] * e_dg_1114);
            float dk2_1116 = __fmaf_rn(_tmem_load_8[2], e_off_1113, _tmem_load_10[2] * e_dg_1114);
            dbp[2] = dk2_1116 * k_t_1105;
            float dk2b_1117 = dk2_1116 * bt_t_1112;
            float dq_o_1118 = dq2_1115 + dqf_1109;
            float dk_o_1119 = dk2b_1117 + dkf_1110 + _tmem_load_11[2];
            float dg_o_1120 = __fmaf_rn(q_t_1108, dq2_1115, (dk2b_1117 - _tmem_load_11[2]) * k_t_1105) + dgf_1111;
            int goff_1121 = t_1101 * rs_g;
            int qoff_1122 = t_1101 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1121) + (0)) = dg_o_1120;
            if (t_1101 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1122) + (0)) = __float2bfloat16_rn(dq_o_1118);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1122) + (0)) = __float2bfloat16_rn(dk_o_1119);
            }
            int t_1123 = 32 + 8 * th + 3;
            float r_1124[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1124[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1123 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1123 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1125 = r_1124[0];
            unsigned int r_1126[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1126[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1123 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1123 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1127 = __uint_as_float(r_1126[0] << 16);
            unsigned int r_1128[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1128[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1129 = __uint_as_float(r_1128[0] << 16);
            float q_t_1130 = ((t_1123 >= clen) ? 0.0f : q_t_1129);
            float dqf_1131 = pool_f32[ew + 384 + d_lane];
            float dkf_1132 = pool_f32[ew + 1408 + d_lane];
            float dgf_1133 = pool_f32[ew + 2432 + d_lane];
            float bt_t_1134 = beta_s[t_1123];
            float _exp2_190 = approx_exp2(g_t_1125 - gn);
            float e_off_1135 = _exp2_190;
            float _exp2_191 = approx_exp2(g_t_1125 - gc_1070);
            float e_dg_1136 = _exp2_191;
            float dq2_1137 = __fmaf_rn(_tmem_load_7[3], e_off_1135, _tmem_load_9[3] * e_dg_1136);
            float dk2_1138 = __fmaf_rn(_tmem_load_8[3], e_off_1135, _tmem_load_10[3] * e_dg_1136);
            dbp[3] = dk2_1138 * k_t_1127;
            float dk2b_1139 = dk2_1138 * bt_t_1134;
            float dq_o_1140 = dq2_1137 + dqf_1131;
            float dk_o_1141 = dk2b_1139 + dkf_1132 + _tmem_load_11[3];
            float dg_o_1142 = __fmaf_rn(q_t_1130, dq2_1137, (dk2b_1139 - _tmem_load_11[3]) * k_t_1127) + dgf_1133;
            int goff_1143 = t_1123 * rs_g;
            int qoff_1144 = t_1123 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1143) + (0)) = dg_o_1142;
            if (t_1123 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1144) + (0)) = __float2bfloat16_rn(dq_o_1140);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1144) + (0)) = __float2bfloat16_rn(dk_o_1141);
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1145 = 32 + 8 * th + 4;
            float r_1146[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1146[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1145 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1145 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1147 = r_1146[0];
            unsigned int r_1148[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1148[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1145 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1145 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1149 = __uint_as_float(r_1148[0] << 16);
            unsigned int r_1150[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1150[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1151 = __uint_as_float(r_1150[0] << 16);
            float q_t_1152 = ((t_1145 >= clen) ? 0.0f : q_t_1151);
            float dqf_1153 = pool_f32[ew + 512 + d_lane];
            float dkf_1154 = pool_f32[ew + 1536 + d_lane];
            float dgf_1155 = pool_f32[ew + 2560 + d_lane];
            float bt_t_1156 = beta_s[t_1145];
            float _exp2_192 = approx_exp2(g_t_1147 - gn);
            float e_off_1157 = _exp2_192;
            float _exp2_193 = approx_exp2(g_t_1147 - gc_1070);
            float e_dg_1158 = _exp2_193;
            float dq2_1159 = __fmaf_rn(_tmem_load_12[0], e_off_1157, _tmem_load_14[0] * e_dg_1158);
            float dk2_1160 = __fmaf_rn(_tmem_load_13[0], e_off_1157, _tmem_load_15[0] * e_dg_1158);
            dbp[4] = dk2_1160 * k_t_1149;
            float dk2b_1161 = dk2_1160 * bt_t_1156;
            float dq_o_1162 = dq2_1159 + dqf_1153;
            float dk_o_1163 = dk2b_1161 + dkf_1154 + _tmem_load_16[0];
            float dg_o_1164 = __fmaf_rn(q_t_1152, dq2_1159, (dk2b_1161 - _tmem_load_16[0]) * k_t_1149) + dgf_1155;
            int goff_1165 = t_1145 * rs_g;
            int qoff_1166 = t_1145 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1165) + (0)) = dg_o_1164;
            if (t_1145 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1166) + (0)) = __float2bfloat16_rn(dq_o_1162);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1166) + (0)) = __float2bfloat16_rn(dk_o_1163);
            }
            int t_1167 = 32 + 8 * th + 4 + 1;
            float r_1168[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1168[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1167 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1167 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1169 = r_1168[0];
            unsigned int r_1170[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1170[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1167 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1167 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1171 = __uint_as_float(r_1170[0] << 16);
            unsigned int r_1172[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1172[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1173 = __uint_as_float(r_1172[0] << 16);
            float q_t_1174 = ((t_1167 >= clen) ? 0.0f : q_t_1173);
            float dqf_1175 = pool_f32[ew + 640 + d_lane];
            float dkf_1176 = pool_f32[ew + 1664 + d_lane];
            float dgf_1177 = pool_f32[ew + 2688 + d_lane];
            float bt_t_1178 = beta_s[t_1167];
            float _exp2_194 = approx_exp2(g_t_1169 - gn);
            float e_off_1179 = _exp2_194;
            float _exp2_195 = approx_exp2(g_t_1169 - gc_1070);
            float e_dg_1180 = _exp2_195;
            float dq2_1181 = __fmaf_rn(_tmem_load_12[1], e_off_1179, _tmem_load_14[1] * e_dg_1180);
            float dk2_1182 = __fmaf_rn(_tmem_load_13[1], e_off_1179, _tmem_load_15[1] * e_dg_1180);
            dbp[5] = dk2_1182 * k_t_1171;
            float dk2b_1183 = dk2_1182 * bt_t_1178;
            float dq_o_1184 = dq2_1181 + dqf_1175;
            float dk_o_1185 = dk2b_1183 + dkf_1176 + _tmem_load_16[1];
            float dg_o_1186 = __fmaf_rn(q_t_1174, dq2_1181, (dk2b_1183 - _tmem_load_16[1]) * k_t_1171) + dgf_1177;
            int goff_1187 = t_1167 * rs_g;
            int qoff_1188 = t_1167 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1187) + (0)) = dg_o_1186;
            if (t_1167 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1188) + (0)) = __float2bfloat16_rn(dq_o_1184);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1188) + (0)) = __float2bfloat16_rn(dk_o_1185);
            }
            int t_1189 = 32 + 8 * th + 4 + 2;
            float r_1190[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1190[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1189 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1189 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1191 = r_1190[0];
            unsigned int r_1192[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1192[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1189 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1189 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1193 = __uint_as_float(r_1192[0] << 16);
            unsigned int r_1194[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1194[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1195 = __uint_as_float(r_1194[0] << 16);
            float q_t_1196 = ((t_1189 >= clen) ? 0.0f : q_t_1195);
            float dqf_1197 = pool_f32[ew + 768 + d_lane];
            float dkf_1198 = pool_f32[ew + 1792 + d_lane];
            float dgf_1199 = pool_f32[ew + 2816 + d_lane];
            float bt_t_1200 = beta_s[t_1189];
            float _exp2_196 = approx_exp2(g_t_1191 - gn);
            float e_off_1201 = _exp2_196;
            float _exp2_197 = approx_exp2(g_t_1191 - gc_1070);
            float e_dg_1202 = _exp2_197;
            float dq2_1203 = __fmaf_rn(_tmem_load_12[2], e_off_1201, _tmem_load_14[2] * e_dg_1202);
            float dk2_1204 = __fmaf_rn(_tmem_load_13[2], e_off_1201, _tmem_load_15[2] * e_dg_1202);
            dbp[6] = dk2_1204 * k_t_1193;
            float dk2b_1205 = dk2_1204 * bt_t_1200;
            float dq_o_1206 = dq2_1203 + dqf_1197;
            float dk_o_1207 = dk2b_1205 + dkf_1198 + _tmem_load_16[2];
            float dg_o_1208 = __fmaf_rn(q_t_1196, dq2_1203, (dk2b_1205 - _tmem_load_16[2]) * k_t_1193) + dgf_1199;
            int goff_1209 = t_1189 * rs_g;
            int qoff_1210 = t_1189 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1209) + (0)) = dg_o_1208;
            if (t_1189 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1210) + (0)) = __float2bfloat16_rn(dq_o_1206);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1210) + (0)) = __float2bfloat16_rn(dk_o_1207);
            }
            int t_1211 = 32 + 8 * th + 4 + 3;
            float r_1212[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1212[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1211 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1211 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1213 = r_1212[0];
            unsigned int r_1214[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1214[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1211 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1211 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1215 = __uint_as_float(r_1214[0] << 16);
            unsigned int r_1216[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1216[0])) : "r"((qbase_1071 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1217 = __uint_as_float(r_1216[0] << 16);
            float q_t_1218 = ((t_1211 >= clen) ? 0.0f : q_t_1217);
            float dqf_1219 = pool_f32[ew + 896 + d_lane];
            float dkf_1220 = pool_f32[ew + 1920 + d_lane];
            float dgf_1221 = pool_f32[ew + 2944 + d_lane];
            float bt_t_1222 = beta_s[t_1211];
            float _exp2_198 = approx_exp2(g_t_1213 - gn);
            float e_off_1223 = _exp2_198;
            float _exp2_199 = approx_exp2(g_t_1213 - gc_1070);
            float e_dg_1224 = _exp2_199;
            float dq2_1225 = __fmaf_rn(_tmem_load_12[3], e_off_1223, _tmem_load_14[3] * e_dg_1224);
            float dk2_1226 = __fmaf_rn(_tmem_load_13[3], e_off_1223, _tmem_load_15[3] * e_dg_1224);
            dbp[7] = dk2_1226 * k_t_1215;
            float dk2b_1227 = dk2_1226 * bt_t_1222;
            float dq_o_1228 = dq2_1225 + dqf_1219;
            float dk_o_1229 = dk2b_1227 + dkf_1220 + _tmem_load_16[3];
            float dg_o_1230 = __fmaf_rn(q_t_1218, dq2_1225, (dk2b_1227 - _tmem_load_16[3]) * k_t_1215) + dgf_1221;
            int goff_1231 = t_1211 * rs_g;
            int qoff_1232 = t_1211 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1231) + (0)) = dg_o_1230;
            if (t_1211 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1232) + (0)) = __float2bfloat16_rn(dq_o_1228);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1232) + (0)) = __float2bfloat16_rn(dk_o_1229);
            }
            int b4 = lane_1 >> 4 & 1;
            int b3 = lane_1 >> 3 & 1;
            int b2 = lane_1 >> 2 & 1;
            float v4[4];
            float send = ((b4 == 0) ? dbp[4] : dbp[0]);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            v4[0] = ((b4 == 0) ? dbp[0] : dbp[4]) + _shfl_xor_0;
            float send_1233 = ((b4 == 0) ? dbp[5] : dbp[1]);
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1233, 16);
            v4[1] = ((b4 == 0) ? dbp[1] : dbp[5]) + _shfl_xor_1;
            float send_1234 = ((b4 == 0) ? dbp[6] : dbp[2]);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1234, 16);
            v4[2] = ((b4 == 0) ? dbp[2] : dbp[6]) + _shfl_xor_2;
            float send_1235 = ((b4 == 0) ? dbp[7] : dbp[3]);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1235, 16);
            v4[3] = ((b4 == 0) ? dbp[3] : dbp[7]) + _shfl_xor_3;
            float v2[2];
            float send_1236 = ((b3 == 0) ? v4[2] : v4[0]);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1236, 8);
            v2[0] = ((b3 == 0) ? v4[0] : v4[2]) + _shfl_xor_4;
            float send_1237 = ((b3 == 0) ? v4[3] : v4[1]);
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1237, 8);
            v2[1] = ((b3 == 0) ? v4[1] : v4[3]) + _shfl_xor_5;
            float send_1238 = ((b2 == 0) ? v2[1] : v2[0]);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1238, 4);
            float v1 = ((b2 == 0) ? v2[0] : v2[1]) + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
            v1 = v1 + _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
            v1 = v1 + _shfl_xor_8;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1a_t2_drained_addr);
            mbarrier_arrive(epi_empty_addr);
            mbarrier_arrive(epi_empty_addr + 8);
            mbarrier_wait(epi_full_addr, 1);
            mbarrier_wait(epi_full_addr + 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1239[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1239[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1240 = r_1239[0];
            float gc_1241 = gn_1240 - 21.640425324000002f;
            int eb_1242 = 98304 + th * -32768;
            int ew_1243 = eb_1242 >> 2;
            int qbase_1244 = pool_f32_addr + (unsigned int)(eb_1242 + 12288 + dh_lane * 1024);
            float dbp_1245[8];
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1246 = 48 + 8 * th;
            float r_1247[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1247[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1246 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1246 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1248 = r_1247[0];
            unsigned int r_1249[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1249[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1246 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1246 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1250 = __uint_as_float(r_1249[0] << 16);
            unsigned int r_1251[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1251[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1252 = __uint_as_float(r_1251[0] << 16);
            float q_t_1253 = ((t_1246 >= clen) ? 0.0f : q_t_1252);
            float dqf_1254 = pool_f32[ew_1243 + d_lane];
            float dkf_1255 = pool_f32[ew_1243 + 1024 + d_lane];
            float dgf_1256 = pool_f32[ew_1243 + 2048 + d_lane];
            float bt_t_1257 = beta_s[t_1246];
            float _exp2_200 = approx_exp2(g_t_1248 - gn_1240);
            float e_off_1258 = _exp2_200;
            float _exp2_201 = approx_exp2(g_t_1248 - gc_1241);
            float e_dg_1259 = _exp2_201;
            float dq2_1260 = __fmaf_rn(_tmem_load_17[0], e_off_1258, _tmem_load_19[0] * e_dg_1259);
            float dk2_1261 = __fmaf_rn(_tmem_load_18[0], e_off_1258, _tmem_load_20[0] * e_dg_1259);
            dbp_1245[0] = dk2_1261 * k_t_1250;
            float dk2b_1262 = dk2_1261 * bt_t_1257;
            float dq_o_1263 = dq2_1260 + dqf_1254;
            float dk_o_1264 = dk2b_1262 + dkf_1255 + _tmem_load_21[0];
            float dg_o_1265 = __fmaf_rn(q_t_1253, dq2_1260, (dk2b_1262 - _tmem_load_21[0]) * k_t_1250) + dgf_1256;
            int goff_1266 = t_1246 * rs_g;
            int qoff_1267 = t_1246 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1266) + (0)) = dg_o_1265;
            if (t_1246 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1267) + (0)) = __float2bfloat16_rn(dq_o_1263);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1267) + (0)) = __float2bfloat16_rn(dk_o_1264);
            }
            int t_1268 = 48 + 8 * th + 1;
            float r_1269[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1269[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1268 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1268 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1270 = r_1269[0];
            unsigned int r_1271[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1271[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1268 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1268 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1272 = __uint_as_float(r_1271[0] << 16);
            unsigned int r_1273[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1273[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1274 = __uint_as_float(r_1273[0] << 16);
            float q_t_1275 = ((t_1268 >= clen) ? 0.0f : q_t_1274);
            float dqf_1276 = pool_f32[ew_1243 + 128 + d_lane];
            float dkf_1277 = pool_f32[ew_1243 + 1152 + d_lane];
            float dgf_1278 = pool_f32[ew_1243 + 2176 + d_lane];
            float bt_t_1279 = beta_s[t_1268];
            float _exp2_202 = approx_exp2(g_t_1270 - gn_1240);
            float e_off_1280 = _exp2_202;
            float _exp2_203 = approx_exp2(g_t_1270 - gc_1241);
            float e_dg_1281 = _exp2_203;
            float dq2_1282 = __fmaf_rn(_tmem_load_17[1], e_off_1280, _tmem_load_19[1] * e_dg_1281);
            float dk2_1283 = __fmaf_rn(_tmem_load_18[1], e_off_1280, _tmem_load_20[1] * e_dg_1281);
            dbp_1245[1] = dk2_1283 * k_t_1272;
            float dk2b_1284 = dk2_1283 * bt_t_1279;
            float dq_o_1285 = dq2_1282 + dqf_1276;
            float dk_o_1286 = dk2b_1284 + dkf_1277 + _tmem_load_21[1];
            float dg_o_1287 = __fmaf_rn(q_t_1275, dq2_1282, (dk2b_1284 - _tmem_load_21[1]) * k_t_1272) + dgf_1278;
            int goff_1288 = t_1268 * rs_g;
            int qoff_1289 = t_1268 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1288) + (0)) = dg_o_1287;
            if (t_1268 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1289) + (0)) = __float2bfloat16_rn(dq_o_1285);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1289) + (0)) = __float2bfloat16_rn(dk_o_1286);
            }
            int t_1290 = 48 + 8 * th + 2;
            float r_1291[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1291[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1290 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1290 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1292 = r_1291[0];
            unsigned int r_1293[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1293[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1290 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1290 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1294 = __uint_as_float(r_1293[0] << 16);
            unsigned int r_1295[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1295[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1296 = __uint_as_float(r_1295[0] << 16);
            float q_t_1297 = ((t_1290 >= clen) ? 0.0f : q_t_1296);
            float dqf_1298 = pool_f32[ew_1243 + 256 + d_lane];
            float dkf_1299 = pool_f32[ew_1243 + 1280 + d_lane];
            float dgf_1300 = pool_f32[ew_1243 + 2304 + d_lane];
            float bt_t_1301 = beta_s[t_1290];
            float _exp2_204 = approx_exp2(g_t_1292 - gn_1240);
            float e_off_1302 = _exp2_204;
            float _exp2_205 = approx_exp2(g_t_1292 - gc_1241);
            float e_dg_1303 = _exp2_205;
            float dq2_1304 = __fmaf_rn(_tmem_load_17[2], e_off_1302, _tmem_load_19[2] * e_dg_1303);
            float dk2_1305 = __fmaf_rn(_tmem_load_18[2], e_off_1302, _tmem_load_20[2] * e_dg_1303);
            dbp_1245[2] = dk2_1305 * k_t_1294;
            float dk2b_1306 = dk2_1305 * bt_t_1301;
            float dq_o_1307 = dq2_1304 + dqf_1298;
            float dk_o_1308 = dk2b_1306 + dkf_1299 + _tmem_load_21[2];
            float dg_o_1309 = __fmaf_rn(q_t_1297, dq2_1304, (dk2b_1306 - _tmem_load_21[2]) * k_t_1294) + dgf_1300;
            int goff_1310 = t_1290 * rs_g;
            int qoff_1311 = t_1290 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1310) + (0)) = dg_o_1309;
            if (t_1290 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1311) + (0)) = __float2bfloat16_rn(dq_o_1307);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1311) + (0)) = __float2bfloat16_rn(dk_o_1308);
            }
            int t_1312 = 48 + 8 * th + 3;
            float r_1313[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1313[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1312 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1312 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1314 = r_1313[0];
            unsigned int r_1315[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1315[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1312 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1312 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1316 = __uint_as_float(r_1315[0] << 16);
            unsigned int r_1317[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1317[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1318 = __uint_as_float(r_1317[0] << 16);
            float q_t_1319 = ((t_1312 >= clen) ? 0.0f : q_t_1318);
            float dqf_1320 = pool_f32[ew_1243 + 384 + d_lane];
            float dkf_1321 = pool_f32[ew_1243 + 1408 + d_lane];
            float dgf_1322 = pool_f32[ew_1243 + 2432 + d_lane];
            float bt_t_1323 = beta_s[t_1312];
            float _exp2_206 = approx_exp2(g_t_1314 - gn_1240);
            float e_off_1324 = _exp2_206;
            float _exp2_207 = approx_exp2(g_t_1314 - gc_1241);
            float e_dg_1325 = _exp2_207;
            float dq2_1326 = __fmaf_rn(_tmem_load_17[3], e_off_1324, _tmem_load_19[3] * e_dg_1325);
            float dk2_1327 = __fmaf_rn(_tmem_load_18[3], e_off_1324, _tmem_load_20[3] * e_dg_1325);
            dbp_1245[3] = dk2_1327 * k_t_1316;
            float dk2b_1328 = dk2_1327 * bt_t_1323;
            float dq_o_1329 = dq2_1326 + dqf_1320;
            float dk_o_1330 = dk2b_1328 + dkf_1321 + _tmem_load_21[3];
            float dg_o_1331 = __fmaf_rn(q_t_1319, dq2_1326, (dk2b_1328 - _tmem_load_21[3]) * k_t_1316) + dgf_1322;
            int goff_1332 = t_1312 * rs_g;
            int qoff_1333 = t_1312 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1332) + (0)) = dg_o_1331;
            if (t_1312 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1333) + (0)) = __float2bfloat16_rn(dq_o_1329);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1333) + (0)) = __float2bfloat16_rn(dk_o_1330);
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1334 = 48 + 8 * th + 4;
            float r_1335[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1335[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1334 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1334 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1336 = r_1335[0];
            unsigned int r_1337[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1337[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1334 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1334 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1338 = __uint_as_float(r_1337[0] << 16);
            unsigned int r_1339[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1339[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1340 = __uint_as_float(r_1339[0] << 16);
            float q_t_1341 = ((t_1334 >= clen) ? 0.0f : q_t_1340);
            float dqf_1342 = pool_f32[ew_1243 + 512 + d_lane];
            float dkf_1343 = pool_f32[ew_1243 + 1536 + d_lane];
            float dgf_1344 = pool_f32[ew_1243 + 2560 + d_lane];
            float bt_t_1345 = beta_s[t_1334];
            float _exp2_208 = approx_exp2(g_t_1336 - gn_1240);
            float e_off_1346 = _exp2_208;
            float _exp2_209 = approx_exp2(g_t_1336 - gc_1241);
            float e_dg_1347 = _exp2_209;
            float dq2_1348 = __fmaf_rn(_tmem_load_22[0], e_off_1346, _tmem_load_24[0] * e_dg_1347);
            float dk2_1349 = __fmaf_rn(_tmem_load_23[0], e_off_1346, _tmem_load_25[0] * e_dg_1347);
            dbp_1245[4] = dk2_1349 * k_t_1338;
            float dk2b_1350 = dk2_1349 * bt_t_1345;
            float dq_o_1351 = dq2_1348 + dqf_1342;
            float dk_o_1352 = dk2b_1350 + dkf_1343 + _tmem_load_26[0];
            float dg_o_1353 = __fmaf_rn(q_t_1341, dq2_1348, (dk2b_1350 - _tmem_load_26[0]) * k_t_1338) + dgf_1344;
            int goff_1354 = t_1334 * rs_g;
            int qoff_1355 = t_1334 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1354) + (0)) = dg_o_1353;
            if (t_1334 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1355) + (0)) = __float2bfloat16_rn(dq_o_1351);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1355) + (0)) = __float2bfloat16_rn(dk_o_1352);
            }
            int t_1356 = 48 + 8 * th + 4 + 1;
            float r_1357[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1357[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1356 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1356 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1358 = r_1357[0];
            unsigned int r_1359[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1359[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1356 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1356 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1360 = __uint_as_float(r_1359[0] << 16);
            unsigned int r_1361[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1361[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1362 = __uint_as_float(r_1361[0] << 16);
            float q_t_1363 = ((t_1356 >= clen) ? 0.0f : q_t_1362);
            float dqf_1364 = pool_f32[ew_1243 + 640 + d_lane];
            float dkf_1365 = pool_f32[ew_1243 + 1664 + d_lane];
            float dgf_1366 = pool_f32[ew_1243 + 2688 + d_lane];
            float bt_t_1367 = beta_s[t_1356];
            float _exp2_210 = approx_exp2(g_t_1358 - gn_1240);
            float e_off_1368 = _exp2_210;
            float _exp2_211 = approx_exp2(g_t_1358 - gc_1241);
            float e_dg_1369 = _exp2_211;
            float dq2_1370 = __fmaf_rn(_tmem_load_22[1], e_off_1368, _tmem_load_24[1] * e_dg_1369);
            float dk2_1371 = __fmaf_rn(_tmem_load_23[1], e_off_1368, _tmem_load_25[1] * e_dg_1369);
            dbp_1245[5] = dk2_1371 * k_t_1360;
            float dk2b_1372 = dk2_1371 * bt_t_1367;
            float dq_o_1373 = dq2_1370 + dqf_1364;
            float dk_o_1374 = dk2b_1372 + dkf_1365 + _tmem_load_26[1];
            float dg_o_1375 = __fmaf_rn(q_t_1363, dq2_1370, (dk2b_1372 - _tmem_load_26[1]) * k_t_1360) + dgf_1366;
            int goff_1376 = t_1356 * rs_g;
            int qoff_1377 = t_1356 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1376) + (0)) = dg_o_1375;
            if (t_1356 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1377) + (0)) = __float2bfloat16_rn(dq_o_1373);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1377) + (0)) = __float2bfloat16_rn(dk_o_1374);
            }
            int t_1378 = 48 + 8 * th + 4 + 2;
            float r_1379[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1379[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1378 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1378 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1380 = r_1379[0];
            unsigned int r_1381[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1381[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1378 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1378 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1382 = __uint_as_float(r_1381[0] << 16);
            unsigned int r_1383[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1383[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1384 = __uint_as_float(r_1383[0] << 16);
            float q_t_1385 = ((t_1378 >= clen) ? 0.0f : q_t_1384);
            float dqf_1386 = pool_f32[ew_1243 + 768 + d_lane];
            float dkf_1387 = pool_f32[ew_1243 + 1792 + d_lane];
            float dgf_1388 = pool_f32[ew_1243 + 2816 + d_lane];
            float bt_t_1389 = beta_s[t_1378];
            float _exp2_212 = approx_exp2(g_t_1380 - gn_1240);
            float e_off_1390 = _exp2_212;
            float _exp2_213 = approx_exp2(g_t_1380 - gc_1241);
            float e_dg_1391 = _exp2_213;
            float dq2_1392 = __fmaf_rn(_tmem_load_22[2], e_off_1390, _tmem_load_24[2] * e_dg_1391);
            float dk2_1393 = __fmaf_rn(_tmem_load_23[2], e_off_1390, _tmem_load_25[2] * e_dg_1391);
            dbp_1245[6] = dk2_1393 * k_t_1382;
            float dk2b_1394 = dk2_1393 * bt_t_1389;
            float dq_o_1395 = dq2_1392 + dqf_1386;
            float dk_o_1396 = dk2b_1394 + dkf_1387 + _tmem_load_26[2];
            float dg_o_1397 = __fmaf_rn(q_t_1385, dq2_1392, (dk2b_1394 - _tmem_load_26[2]) * k_t_1382) + dgf_1388;
            int goff_1398 = t_1378 * rs_g;
            int qoff_1399 = t_1378 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1398) + (0)) = dg_o_1397;
            if (t_1378 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1399) + (0)) = __float2bfloat16_rn(dq_o_1395);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1399) + (0)) = __float2bfloat16_rn(dk_o_1396);
            }
            int t_1400 = 48 + 8 * th + 4 + 3;
            float r_1401[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1401[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1400 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1400 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1402 = r_1401[0];
            unsigned int r_1403[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1403[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1400 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1400 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1404 = __uint_as_float(r_1403[0] << 16);
            unsigned int r_1405[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1405[0])) : "r"((qbase_1244 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1406 = __uint_as_float(r_1405[0] << 16);
            float q_t_1407 = ((t_1400 >= clen) ? 0.0f : q_t_1406);
            float dqf_1408 = pool_f32[ew_1243 + 896 + d_lane];
            float dkf_1409 = pool_f32[ew_1243 + 1920 + d_lane];
            float dgf_1410 = pool_f32[ew_1243 + 2944 + d_lane];
            float bt_t_1411 = beta_s[t_1400];
            float _exp2_214 = approx_exp2(g_t_1402 - gn_1240);
            float e_off_1412 = _exp2_214;
            float _exp2_215 = approx_exp2(g_t_1402 - gc_1241);
            float e_dg_1413 = _exp2_215;
            float dq2_1414 = __fmaf_rn(_tmem_load_22[3], e_off_1412, _tmem_load_24[3] * e_dg_1413);
            float dk2_1415 = __fmaf_rn(_tmem_load_23[3], e_off_1412, _tmem_load_25[3] * e_dg_1413);
            dbp_1245[7] = dk2_1415 * k_t_1404;
            float dk2b_1416 = dk2_1415 * bt_t_1411;
            float dq_o_1417 = dq2_1414 + dqf_1408;
            float dk_o_1418 = dk2b_1416 + dkf_1409 + _tmem_load_26[3];
            float dg_o_1419 = __fmaf_rn(q_t_1407, dq2_1414, (dk2b_1416 - _tmem_load_26[3]) * k_t_1404) + dgf_1410;
            int goff_1420 = t_1400 * rs_g;
            int qoff_1421 = t_1400 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1420) + (0)) = dg_o_1419;
            if (t_1400 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1421) + (0)) = __float2bfloat16_rn(dq_o_1417);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1421) + (0)) = __float2bfloat16_rn(dk_o_1418);
            }
            int b4_1422 = lane_1 >> 4 & 1;
            int b3_1423 = lane_1 >> 3 & 1;
            int b2_1424 = lane_1 >> 2 & 1;
            float v4_1425[4];
            float send_1426 = ((b4_1422 == 0) ? dbp_1245[4] : dbp_1245[0]);
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1426, 16);
            v4_1425[0] = ((b4_1422 == 0) ? dbp_1245[0] : dbp_1245[4]) + _shfl_xor_9;
            float send_1427 = ((b4_1422 == 0) ? dbp_1245[5] : dbp_1245[1]);
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1427, 16);
            v4_1425[1] = ((b4_1422 == 0) ? dbp_1245[1] : dbp_1245[5]) + _shfl_xor_10;
            float send_1428 = ((b4_1422 == 0) ? dbp_1245[6] : dbp_1245[2]);
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1428, 16);
            v4_1425[2] = ((b4_1422 == 0) ? dbp_1245[2] : dbp_1245[6]) + _shfl_xor_11;
            float send_1429 = ((b4_1422 == 0) ? dbp_1245[7] : dbp_1245[3]);
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1429, 16);
            v4_1425[3] = ((b4_1422 == 0) ? dbp_1245[3] : dbp_1245[7]) + _shfl_xor_12;
            float v2_1430[2];
            float send_1431 = ((b3_1423 == 0) ? v4_1425[2] : v4_1425[0]);
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1431, 8);
            v2_1430[0] = ((b3_1423 == 0) ? v4_1425[0] : v4_1425[2]) + _shfl_xor_13;
            float send_1432 = ((b3_1423 == 0) ? v4_1425[3] : v4_1425[1]);
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1432, 8);
            v2_1430[1] = ((b3_1423 == 0) ? v4_1425[1] : v4_1425[3]) + _shfl_xor_14;
            float send_1433 = ((b2_1424 == 0) ? v2_1430[1] : v2_1430[0]);
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_1433, 4);
            float v1_1434 = ((b2_1424 == 0) ? v2_1430[0] : v2_1430[1]) + _shfl_xor_15;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, v1_1434, 2);
            v1_1434 = v1_1434 + _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, v1_1434, 1);
            v1_1434 = v1_1434 + _shfl_xor_17;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr);
            mbarrier_arrive(epi_empty_addr + 8);
            mbarrier_wait(p1b_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + 16, 0);
            mbarrier_wait(epi_full_addr + 24, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1435[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1435[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1436 = r_1435[0];
            float gc_1437 = gn_1436 - 21.640425324000002f;
            int eb_1438 = 81920 + th * -32768;
            int ew_1439 = eb_1438 >> 2;
            int qbase_1440 = pool_f32_addr + (unsigned int)(eb_1438 + 12288 + dh_lane * 1024);
            float dbp_1441[8];
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1442 = 16 + 8 * th;
            float r_1443[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1443[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1442 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1442 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1444 = r_1443[0];
            unsigned int r_1445[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1445[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1442 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1442 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1446 = __uint_as_float(r_1445[0] << 16);
            unsigned int r_1447[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1447[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1448 = __uint_as_float(r_1447[0] << 16);
            float q_t_1449 = ((t_1442 >= clen) ? 0.0f : q_t_1448);
            float dqf_1450 = pool_f32[ew_1439 + d_lane];
            float dkf_1451 = pool_f32[ew_1439 + 1024 + d_lane];
            float dgf_1452 = pool_f32[ew_1439 + 2048 + d_lane];
            float bt_t_1453 = beta_s[t_1442];
            float _exp2_216 = approx_exp2(g_t_1444 - gn_1436);
            float e_off_1454 = _exp2_216;
            float _exp2_217 = approx_exp2(g_t_1444 - gc_1437);
            float e_dg_1455 = _exp2_217;
            float dq2_1456 = __fmaf_rn(_tmem_load_27[0], e_off_1454, _tmem_load_29[0] * e_dg_1455);
            float dk2_1457 = __fmaf_rn(_tmem_load_28[0], e_off_1454, _tmem_load_30[0] * e_dg_1455);
            dbp_1441[0] = dk2_1457 * k_t_1446;
            float dk2b_1458 = dk2_1457 * bt_t_1453;
            float dq_o_1459 = dq2_1456 + dqf_1450;
            float dk_o_1460 = dk2b_1458 + dkf_1451 + _tmem_load_31[0];
            float dg_o_1461 = __fmaf_rn(q_t_1449, dq2_1456, (dk2b_1458 - _tmem_load_31[0]) * k_t_1446) + dgf_1452;
            int goff_1462 = t_1442 * rs_g;
            int qoff_1463 = t_1442 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1462) + (0)) = dg_o_1461;
            if (t_1442 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1463) + (0)) = __float2bfloat16_rn(dq_o_1459);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1463) + (0)) = __float2bfloat16_rn(dk_o_1460);
            }
            int t_1464 = 16 + 8 * th + 1;
            float r_1465[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1465[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1464 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1464 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1466 = r_1465[0];
            unsigned int r_1467[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1467[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1464 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1464 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1468 = __uint_as_float(r_1467[0] << 16);
            unsigned int r_1469[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1469[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1470 = __uint_as_float(r_1469[0] << 16);
            float q_t_1471 = ((t_1464 >= clen) ? 0.0f : q_t_1470);
            float dqf_1472 = pool_f32[ew_1439 + 128 + d_lane];
            float dkf_1473 = pool_f32[ew_1439 + 1152 + d_lane];
            float dgf_1474 = pool_f32[ew_1439 + 2176 + d_lane];
            float bt_t_1475 = beta_s[t_1464];
            float _exp2_218 = approx_exp2(g_t_1466 - gn_1436);
            float e_off_1476 = _exp2_218;
            float _exp2_219 = approx_exp2(g_t_1466 - gc_1437);
            float e_dg_1477 = _exp2_219;
            float dq2_1478 = __fmaf_rn(_tmem_load_27[1], e_off_1476, _tmem_load_29[1] * e_dg_1477);
            float dk2_1479 = __fmaf_rn(_tmem_load_28[1], e_off_1476, _tmem_load_30[1] * e_dg_1477);
            dbp_1441[1] = dk2_1479 * k_t_1468;
            float dk2b_1480 = dk2_1479 * bt_t_1475;
            float dq_o_1481 = dq2_1478 + dqf_1472;
            float dk_o_1482 = dk2b_1480 + dkf_1473 + _tmem_load_31[1];
            float dg_o_1483 = __fmaf_rn(q_t_1471, dq2_1478, (dk2b_1480 - _tmem_load_31[1]) * k_t_1468) + dgf_1474;
            int goff_1484 = t_1464 * rs_g;
            int qoff_1485 = t_1464 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1484) + (0)) = dg_o_1483;
            if (t_1464 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1485) + (0)) = __float2bfloat16_rn(dq_o_1481);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1485) + (0)) = __float2bfloat16_rn(dk_o_1482);
            }
            int t_1486 = 16 + 8 * th + 2;
            float r_1487[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1487[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1486 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1486 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1488 = r_1487[0];
            unsigned int r_1489[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1489[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1486 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1486 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1490 = __uint_as_float(r_1489[0] << 16);
            unsigned int r_1491[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1491[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1492 = __uint_as_float(r_1491[0] << 16);
            float q_t_1493 = ((t_1486 >= clen) ? 0.0f : q_t_1492);
            float dqf_1494 = pool_f32[ew_1439 + 256 + d_lane];
            float dkf_1495 = pool_f32[ew_1439 + 1280 + d_lane];
            float dgf_1496 = pool_f32[ew_1439 + 2304 + d_lane];
            float bt_t_1497 = beta_s[t_1486];
            float _exp2_220 = approx_exp2(g_t_1488 - gn_1436);
            float e_off_1498 = _exp2_220;
            float _exp2_221 = approx_exp2(g_t_1488 - gc_1437);
            float e_dg_1499 = _exp2_221;
            float dq2_1500 = __fmaf_rn(_tmem_load_27[2], e_off_1498, _tmem_load_29[2] * e_dg_1499);
            float dk2_1501 = __fmaf_rn(_tmem_load_28[2], e_off_1498, _tmem_load_30[2] * e_dg_1499);
            dbp_1441[2] = dk2_1501 * k_t_1490;
            float dk2b_1502 = dk2_1501 * bt_t_1497;
            float dq_o_1503 = dq2_1500 + dqf_1494;
            float dk_o_1504 = dk2b_1502 + dkf_1495 + _tmem_load_31[2];
            float dg_o_1505 = __fmaf_rn(q_t_1493, dq2_1500, (dk2b_1502 - _tmem_load_31[2]) * k_t_1490) + dgf_1496;
            int goff_1506 = t_1486 * rs_g;
            int qoff_1507 = t_1486 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1506) + (0)) = dg_o_1505;
            if (t_1486 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1507) + (0)) = __float2bfloat16_rn(dq_o_1503);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1507) + (0)) = __float2bfloat16_rn(dk_o_1504);
            }
            int t_1508 = 16 + 8 * th + 3;
            float r_1509[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1509[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1508 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1508 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1510 = r_1509[0];
            unsigned int r_1511[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1511[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1508 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1508 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1512 = __uint_as_float(r_1511[0] << 16);
            unsigned int r_1513[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1513[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1514 = __uint_as_float(r_1513[0] << 16);
            float q_t_1515 = ((t_1508 >= clen) ? 0.0f : q_t_1514);
            float dqf_1516 = pool_f32[ew_1439 + 384 + d_lane];
            float dkf_1517 = pool_f32[ew_1439 + 1408 + d_lane];
            float dgf_1518 = pool_f32[ew_1439 + 2432 + d_lane];
            float bt_t_1519 = beta_s[t_1508];
            float _exp2_222 = approx_exp2(g_t_1510 - gn_1436);
            float e_off_1520 = _exp2_222;
            float _exp2_223 = approx_exp2(g_t_1510 - gc_1437);
            float e_dg_1521 = _exp2_223;
            float dq2_1522 = __fmaf_rn(_tmem_load_27[3], e_off_1520, _tmem_load_29[3] * e_dg_1521);
            float dk2_1523 = __fmaf_rn(_tmem_load_28[3], e_off_1520, _tmem_load_30[3] * e_dg_1521);
            dbp_1441[3] = dk2_1523 * k_t_1512;
            float dk2b_1524 = dk2_1523 * bt_t_1519;
            float dq_o_1525 = dq2_1522 + dqf_1516;
            float dk_o_1526 = dk2b_1524 + dkf_1517 + _tmem_load_31[3];
            float dg_o_1527 = __fmaf_rn(q_t_1515, dq2_1522, (dk2b_1524 - _tmem_load_31[3]) * k_t_1512) + dgf_1518;
            int goff_1528 = t_1508 * rs_g;
            int qoff_1529 = t_1508 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1528) + (0)) = dg_o_1527;
            if (t_1508 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1529) + (0)) = __float2bfloat16_rn(dq_o_1525);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1529) + (0)) = __float2bfloat16_rn(dk_o_1526);
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
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1530 = 16 + 8 * th + 4;
            float r_1531[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1531[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1530 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1530 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1532 = r_1531[0];
            unsigned int r_1533[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1533[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1530 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1530 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1534 = __uint_as_float(r_1533[0] << 16);
            unsigned int r_1535[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1535[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1536 = __uint_as_float(r_1535[0] << 16);
            float q_t_1537 = ((t_1530 >= clen) ? 0.0f : q_t_1536);
            float dqf_1538 = pool_f32[ew_1439 + 512 + d_lane];
            float dkf_1539 = pool_f32[ew_1439 + 1536 + d_lane];
            float dgf_1540 = pool_f32[ew_1439 + 2560 + d_lane];
            float bt_t_1541 = beta_s[t_1530];
            float _exp2_224 = approx_exp2(g_t_1532 - gn_1436);
            float e_off_1542 = _exp2_224;
            float _exp2_225 = approx_exp2(g_t_1532 - gc_1437);
            float e_dg_1543 = _exp2_225;
            float dq2_1544 = __fmaf_rn(_tmem_load_32[0], e_off_1542, _tmem_load_34[0] * e_dg_1543);
            float dk2_1545 = __fmaf_rn(_tmem_load_33[0], e_off_1542, _tmem_load_35[0] * e_dg_1543);
            dbp_1441[4] = dk2_1545 * k_t_1534;
            float dk2b_1546 = dk2_1545 * bt_t_1541;
            float dq_o_1547 = dq2_1544 + dqf_1538;
            float dk_o_1548 = dk2b_1546 + dkf_1539 + _tmem_load_36[0];
            float dg_o_1549 = __fmaf_rn(q_t_1537, dq2_1544, (dk2b_1546 - _tmem_load_36[0]) * k_t_1534) + dgf_1540;
            int goff_1550 = t_1530 * rs_g;
            int qoff_1551 = t_1530 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1550) + (0)) = dg_o_1549;
            if (t_1530 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1551) + (0)) = __float2bfloat16_rn(dq_o_1547);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1551) + (0)) = __float2bfloat16_rn(dk_o_1548);
            }
            int t_1552 = 16 + 8 * th + 4 + 1;
            float r_1553[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1553[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1552 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1552 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1554 = r_1553[0];
            unsigned int r_1555[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1555[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1552 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1552 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1556 = __uint_as_float(r_1555[0] << 16);
            unsigned int r_1557[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1557[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1558 = __uint_as_float(r_1557[0] << 16);
            float q_t_1559 = ((t_1552 >= clen) ? 0.0f : q_t_1558);
            float dqf_1560 = pool_f32[ew_1439 + 640 + d_lane];
            float dkf_1561 = pool_f32[ew_1439 + 1664 + d_lane];
            float dgf_1562 = pool_f32[ew_1439 + 2688 + d_lane];
            float bt_t_1563 = beta_s[t_1552];
            float _exp2_226 = approx_exp2(g_t_1554 - gn_1436);
            float e_off_1564 = _exp2_226;
            float _exp2_227 = approx_exp2(g_t_1554 - gc_1437);
            float e_dg_1565 = _exp2_227;
            float dq2_1566 = __fmaf_rn(_tmem_load_32[1], e_off_1564, _tmem_load_34[1] * e_dg_1565);
            float dk2_1567 = __fmaf_rn(_tmem_load_33[1], e_off_1564, _tmem_load_35[1] * e_dg_1565);
            dbp_1441[5] = dk2_1567 * k_t_1556;
            float dk2b_1568 = dk2_1567 * bt_t_1563;
            float dq_o_1569 = dq2_1566 + dqf_1560;
            float dk_o_1570 = dk2b_1568 + dkf_1561 + _tmem_load_36[1];
            float dg_o_1571 = __fmaf_rn(q_t_1559, dq2_1566, (dk2b_1568 - _tmem_load_36[1]) * k_t_1556) + dgf_1562;
            int goff_1572 = t_1552 * rs_g;
            int qoff_1573 = t_1552 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1572) + (0)) = dg_o_1571;
            if (t_1552 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1573) + (0)) = __float2bfloat16_rn(dq_o_1569);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1573) + (0)) = __float2bfloat16_rn(dk_o_1570);
            }
            int t_1574 = 16 + 8 * th + 4 + 2;
            float r_1575[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1575[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1574 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1574 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1576 = r_1575[0];
            unsigned int r_1577[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1577[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1574 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1574 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1578 = __uint_as_float(r_1577[0] << 16);
            unsigned int r_1579[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1579[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1580 = __uint_as_float(r_1579[0] << 16);
            float q_t_1581 = ((t_1574 >= clen) ? 0.0f : q_t_1580);
            float dqf_1582 = pool_f32[ew_1439 + 768 + d_lane];
            float dkf_1583 = pool_f32[ew_1439 + 1792 + d_lane];
            float dgf_1584 = pool_f32[ew_1439 + 2816 + d_lane];
            float bt_t_1585 = beta_s[t_1574];
            float _exp2_228 = approx_exp2(g_t_1576 - gn_1436);
            float e_off_1586 = _exp2_228;
            float _exp2_229 = approx_exp2(g_t_1576 - gc_1437);
            float e_dg_1587 = _exp2_229;
            float dq2_1588 = __fmaf_rn(_tmem_load_32[2], e_off_1586, _tmem_load_34[2] * e_dg_1587);
            float dk2_1589 = __fmaf_rn(_tmem_load_33[2], e_off_1586, _tmem_load_35[2] * e_dg_1587);
            dbp_1441[6] = dk2_1589 * k_t_1578;
            float dk2b_1590 = dk2_1589 * bt_t_1585;
            float dq_o_1591 = dq2_1588 + dqf_1582;
            float dk_o_1592 = dk2b_1590 + dkf_1583 + _tmem_load_36[2];
            float dg_o_1593 = __fmaf_rn(q_t_1581, dq2_1588, (dk2b_1590 - _tmem_load_36[2]) * k_t_1578) + dgf_1584;
            int goff_1594 = t_1574 * rs_g;
            int qoff_1595 = t_1574 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1594) + (0)) = dg_o_1593;
            if (t_1574 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1595) + (0)) = __float2bfloat16_rn(dq_o_1591);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1595) + (0)) = __float2bfloat16_rn(dk_o_1592);
            }
            int t_1596 = 16 + 8 * th + 4 + 3;
            float r_1597[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1597[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1596 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1596 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1598 = r_1597[0];
            unsigned int r_1599[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1599[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1596 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1596 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1600 = __uint_as_float(r_1599[0] << 16);
            unsigned int r_1601[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1601[0])) : "r"((qbase_1440 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1602 = __uint_as_float(r_1601[0] << 16);
            float q_t_1603 = ((t_1596 >= clen) ? 0.0f : q_t_1602);
            float dqf_1604 = pool_f32[ew_1439 + 896 + d_lane];
            float dkf_1605 = pool_f32[ew_1439 + 1920 + d_lane];
            float dgf_1606 = pool_f32[ew_1439 + 2944 + d_lane];
            float bt_t_1607 = beta_s[t_1596];
            float _exp2_230 = approx_exp2(g_t_1598 - gn_1436);
            float e_off_1608 = _exp2_230;
            float _exp2_231 = approx_exp2(g_t_1598 - gc_1437);
            float e_dg_1609 = _exp2_231;
            float dq2_1610 = __fmaf_rn(_tmem_load_32[3], e_off_1608, _tmem_load_34[3] * e_dg_1609);
            float dk2_1611 = __fmaf_rn(_tmem_load_33[3], e_off_1608, _tmem_load_35[3] * e_dg_1609);
            dbp_1441[7] = dk2_1611 * k_t_1600;
            float dk2b_1612 = dk2_1611 * bt_t_1607;
            float dq_o_1613 = dq2_1610 + dqf_1604;
            float dk_o_1614 = dk2b_1612 + dkf_1605 + _tmem_load_36[3];
            float dg_o_1615 = __fmaf_rn(q_t_1603, dq2_1610, (dk2b_1612 - _tmem_load_36[3]) * k_t_1600) + dgf_1606;
            int goff_1616 = t_1596 * rs_g;
            int qoff_1617 = t_1596 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1616) + (0)) = dg_o_1615;
            if (t_1596 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1617) + (0)) = __float2bfloat16_rn(dq_o_1613);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1617) + (0)) = __float2bfloat16_rn(dk_o_1614);
            }
            int b4_1618 = lane_1 >> 4 & 1;
            int b3_1619 = lane_1 >> 3 & 1;
            int b2_1620 = lane_1 >> 2 & 1;
            float v4_1621[4];
            float send_1622 = ((b4_1618 == 0) ? dbp_1441[4] : dbp_1441[0]);
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, send_1622, 16);
            v4_1621[0] = ((b4_1618 == 0) ? dbp_1441[0] : dbp_1441[4]) + _shfl_xor_18;
            float send_1623 = ((b4_1618 == 0) ? dbp_1441[5] : dbp_1441[1]);
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, send_1623, 16);
            v4_1621[1] = ((b4_1618 == 0) ? dbp_1441[1] : dbp_1441[5]) + _shfl_xor_19;
            float send_1624 = ((b4_1618 == 0) ? dbp_1441[6] : dbp_1441[2]);
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, send_1624, 16);
            v4_1621[2] = ((b4_1618 == 0) ? dbp_1441[2] : dbp_1441[6]) + _shfl_xor_20;
            float send_1625 = ((b4_1618 == 0) ? dbp_1441[7] : dbp_1441[3]);
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, send_1625, 16);
            v4_1621[3] = ((b4_1618 == 0) ? dbp_1441[3] : dbp_1441[7]) + _shfl_xor_21;
            float v2_1626[2];
            float send_1627 = ((b3_1619 == 0) ? v4_1621[2] : v4_1621[0]);
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, send_1627, 8);
            v2_1626[0] = ((b3_1619 == 0) ? v4_1621[0] : v4_1621[2]) + _shfl_xor_22;
            float send_1628 = ((b3_1619 == 0) ? v4_1621[3] : v4_1621[1]);
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, send_1628, 8);
            v2_1626[1] = ((b3_1619 == 0) ? v4_1621[1] : v4_1621[3]) + _shfl_xor_23;
            float send_1629 = ((b2_1620 == 0) ? v2_1626[1] : v2_1626[0]);
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, send_1629, 4);
            float v1_1630 = ((b2_1620 == 0) ? v2_1626[0] : v2_1626[1]) + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, v1_1630, 2);
            v1_1630 = v1_1630 + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, v1_1630, 1);
            v1_1630 = v1_1630 + _shfl_xor_26;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + 16);
            mbarrier_arrive(epi_empty_addr + 24);
            mbarrier_wait(epi_full_addr, 0);
            mbarrier_wait(epi_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1631[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1631[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1632 = r_1631[0];
            float gc_1633 = gn_1632 - 21.640425324000002f;
            int eb_1634 = 98304 + th * -32768;
            int ew_1635 = eb_1634 >> 2;
            int qbase_1636 = pool_f32_addr + (unsigned int)(eb_1634 + 12288 + dh_lane * 1024);
            float dbp_1637[8];
            float _tmem_load_37[4];
            tmem_ld_x4(&_tmem_load_37[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            float _tmem_load_38[4];
            tmem_ld_x4(&_tmem_load_38[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th));
            float _tmem_load_39[4];
            tmem_ld_x4(&_tmem_load_39[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1638 = 8 * th;
            float r_1639[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1639[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1638 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1638 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1640 = r_1639[0];
            unsigned int r_1641[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1641[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1638 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1638 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1642 = __uint_as_float(r_1641[0] << 16);
            unsigned int r_1643[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1643[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1644 = __uint_as_float(r_1643[0] << 16);
            float q_t_1645 = ((t_1638 >= clen) ? 0.0f : q_t_1644);
            float dqf_1646 = pool_f32[ew_1635 + d_lane];
            float dkf_1647 = pool_f32[ew_1635 + 1024 + d_lane];
            float dgf_1648 = pool_f32[ew_1635 + 2048 + d_lane];
            float bt_t_1649 = beta_s[t_1638];
            float _exp2_232 = approx_exp2(g_t_1640 - gn_1632);
            float e_off_1650 = _exp2_232;
            float _exp2_233 = approx_exp2(g_t_1640 - gc_1633);
            float e_dg_1651 = _exp2_233;
            float dq2_1652 = _tmem_load_37[0] * e_dg_1651;
            float dk2_1653 = _tmem_load_38[0] * e_dg_1651;
            dbp_1637[0] = dk2_1653 * k_t_1642;
            float dk2b_1654 = dk2_1653 * bt_t_1649;
            float dq_o_1655 = dq2_1652 + dqf_1646;
            float dk_o_1656 = dk2b_1654 + dkf_1647 + _tmem_load_39[0];
            float dg_o_1657 = __fmaf_rn(q_t_1645, dq2_1652, (dk2b_1654 - _tmem_load_39[0]) * k_t_1642) + dgf_1648;
            int goff_1658 = t_1638 * rs_g;
            int qoff_1659 = t_1638 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1658) + (0)) = dg_o_1657;
            if (t_1638 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1659) + (0)) = __float2bfloat16_rn(dq_o_1655);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1659) + (0)) = __float2bfloat16_rn(dk_o_1656);
            }
            int t_1660 = 8 * th + 1;
            float r_1661[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1661[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1660 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1660 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1662 = r_1661[0];
            unsigned int r_1663[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1663[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1660 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1660 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1664 = __uint_as_float(r_1663[0] << 16);
            unsigned int r_1665[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1665[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1666 = __uint_as_float(r_1665[0] << 16);
            float q_t_1667 = ((t_1660 >= clen) ? 0.0f : q_t_1666);
            float dqf_1668 = pool_f32[ew_1635 + 128 + d_lane];
            float dkf_1669 = pool_f32[ew_1635 + 1152 + d_lane];
            float dgf_1670 = pool_f32[ew_1635 + 2176 + d_lane];
            float bt_t_1671 = beta_s[t_1660];
            float _exp2_234 = approx_exp2(g_t_1662 - gn_1632);
            float e_off_1672 = _exp2_234;
            float _exp2_235 = approx_exp2(g_t_1662 - gc_1633);
            float e_dg_1673 = _exp2_235;
            float dq2_1674 = _tmem_load_37[1] * e_dg_1673;
            float dk2_1675 = _tmem_load_38[1] * e_dg_1673;
            dbp_1637[1] = dk2_1675 * k_t_1664;
            float dk2b_1676 = dk2_1675 * bt_t_1671;
            float dq_o_1677 = dq2_1674 + dqf_1668;
            float dk_o_1678 = dk2b_1676 + dkf_1669 + _tmem_load_39[1];
            float dg_o_1679 = __fmaf_rn(q_t_1667, dq2_1674, (dk2b_1676 - _tmem_load_39[1]) * k_t_1664) + dgf_1670;
            int goff_1680 = t_1660 * rs_g;
            int qoff_1681 = t_1660 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1680) + (0)) = dg_o_1679;
            if (t_1660 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1681) + (0)) = __float2bfloat16_rn(dq_o_1677);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1681) + (0)) = __float2bfloat16_rn(dk_o_1678);
            }
            int t_1682 = 8 * th + 2;
            float r_1683[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1683[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1682 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1682 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1684 = r_1683[0];
            unsigned int r_1685[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1685[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1682 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1682 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1686 = __uint_as_float(r_1685[0] << 16);
            unsigned int r_1687[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1687[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1688 = __uint_as_float(r_1687[0] << 16);
            float q_t_1689 = ((t_1682 >= clen) ? 0.0f : q_t_1688);
            float dqf_1690 = pool_f32[ew_1635 + 256 + d_lane];
            float dkf_1691 = pool_f32[ew_1635 + 1280 + d_lane];
            float dgf_1692 = pool_f32[ew_1635 + 2304 + d_lane];
            float bt_t_1693 = beta_s[t_1682];
            float _exp2_236 = approx_exp2(g_t_1684 - gn_1632);
            float e_off_1694 = _exp2_236;
            float _exp2_237 = approx_exp2(g_t_1684 - gc_1633);
            float e_dg_1695 = _exp2_237;
            float dq2_1696 = _tmem_load_37[2] * e_dg_1695;
            float dk2_1697 = _tmem_load_38[2] * e_dg_1695;
            dbp_1637[2] = dk2_1697 * k_t_1686;
            float dk2b_1698 = dk2_1697 * bt_t_1693;
            float dq_o_1699 = dq2_1696 + dqf_1690;
            float dk_o_1700 = dk2b_1698 + dkf_1691 + _tmem_load_39[2];
            float dg_o_1701 = __fmaf_rn(q_t_1689, dq2_1696, (dk2b_1698 - _tmem_load_39[2]) * k_t_1686) + dgf_1692;
            int goff_1702 = t_1682 * rs_g;
            int qoff_1703 = t_1682 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1702) + (0)) = dg_o_1701;
            if (t_1682 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1703) + (0)) = __float2bfloat16_rn(dq_o_1699);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1703) + (0)) = __float2bfloat16_rn(dk_o_1700);
            }
            int t_1704 = 8 * th + 3;
            float r_1705[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1705[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1704 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1704 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1706 = r_1705[0];
            unsigned int r_1707[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1707[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1704 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1704 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1708 = __uint_as_float(r_1707[0] << 16);
            unsigned int r_1709[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1709[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1710 = __uint_as_float(r_1709[0] << 16);
            float q_t_1711 = ((t_1704 >= clen) ? 0.0f : q_t_1710);
            float dqf_1712 = pool_f32[ew_1635 + 384 + d_lane];
            float dkf_1713 = pool_f32[ew_1635 + 1408 + d_lane];
            float dgf_1714 = pool_f32[ew_1635 + 2432 + d_lane];
            float bt_t_1715 = beta_s[t_1704];
            float _exp2_238 = approx_exp2(g_t_1706 - gn_1632);
            float e_off_1716 = _exp2_238;
            float _exp2_239 = approx_exp2(g_t_1706 - gc_1633);
            float e_dg_1717 = _exp2_239;
            float dq2_1718 = _tmem_load_37[3] * e_dg_1717;
            float dk2_1719 = _tmem_load_38[3] * e_dg_1717;
            dbp_1637[3] = dk2_1719 * k_t_1708;
            float dk2b_1720 = dk2_1719 * bt_t_1715;
            float dq_o_1721 = dq2_1718 + dqf_1712;
            float dk_o_1722 = dk2b_1720 + dkf_1713 + _tmem_load_39[3];
            float dg_o_1723 = __fmaf_rn(q_t_1711, dq2_1718, (dk2b_1720 - _tmem_load_39[3]) * k_t_1708) + dgf_1714;
            int goff_1724 = t_1704 * rs_g;
            int qoff_1725 = t_1704 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1724) + (0)) = dg_o_1723;
            if (t_1704 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1725) + (0)) = __float2bfloat16_rn(dq_o_1721);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1725) + (0)) = __float2bfloat16_rn(dk_o_1722);
            }
            float _tmem_load_40[4];
            tmem_ld_x4(&_tmem_load_40[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 4));
            float _tmem_load_41[4];
            tmem_ld_x4(&_tmem_load_41[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th + 4));
            float _tmem_load_42[4];
            tmem_ld_x4(&_tmem_load_42[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th + 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1726 = 8 * th + 4;
            float r_1727[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1727[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1726 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1726 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1728 = r_1727[0];
            unsigned int r_1729[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1729[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1726 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1726 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1730 = __uint_as_float(r_1729[0] << 16);
            unsigned int r_1731[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1731[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1732 = __uint_as_float(r_1731[0] << 16);
            float q_t_1733 = ((t_1726 >= clen) ? 0.0f : q_t_1732);
            float dqf_1734 = pool_f32[ew_1635 + 512 + d_lane];
            float dkf_1735 = pool_f32[ew_1635 + 1536 + d_lane];
            float dgf_1736 = pool_f32[ew_1635 + 2560 + d_lane];
            float bt_t_1737 = beta_s[t_1726];
            float _exp2_240 = approx_exp2(g_t_1728 - gn_1632);
            float e_off_1738 = _exp2_240;
            float _exp2_241 = approx_exp2(g_t_1728 - gc_1633);
            float e_dg_1739 = _exp2_241;
            float dq2_1740 = _tmem_load_40[0] * e_dg_1739;
            float dk2_1741 = _tmem_load_41[0] * e_dg_1739;
            dbp_1637[4] = dk2_1741 * k_t_1730;
            float dk2b_1742 = dk2_1741 * bt_t_1737;
            float dq_o_1743 = dq2_1740 + dqf_1734;
            float dk_o_1744 = dk2b_1742 + dkf_1735 + _tmem_load_42[0];
            float dg_o_1745 = __fmaf_rn(q_t_1733, dq2_1740, (dk2b_1742 - _tmem_load_42[0]) * k_t_1730) + dgf_1736;
            int goff_1746 = t_1726 * rs_g;
            int qoff_1747 = t_1726 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1746) + (0)) = dg_o_1745;
            if (t_1726 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1747) + (0)) = __float2bfloat16_rn(dq_o_1743);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1747) + (0)) = __float2bfloat16_rn(dk_o_1744);
            }
            int t_1748 = 8 * th + 4 + 1;
            float r_1749[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1749[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1748 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1748 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1750 = r_1749[0];
            unsigned int r_1751[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1751[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1748 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1748 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1752 = __uint_as_float(r_1751[0] << 16);
            unsigned int r_1753[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1753[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1754 = __uint_as_float(r_1753[0] << 16);
            float q_t_1755 = ((t_1748 >= clen) ? 0.0f : q_t_1754);
            float dqf_1756 = pool_f32[ew_1635 + 640 + d_lane];
            float dkf_1757 = pool_f32[ew_1635 + 1664 + d_lane];
            float dgf_1758 = pool_f32[ew_1635 + 2688 + d_lane];
            float bt_t_1759 = beta_s[t_1748];
            float _exp2_242 = approx_exp2(g_t_1750 - gn_1632);
            float e_off_1760 = _exp2_242;
            float _exp2_243 = approx_exp2(g_t_1750 - gc_1633);
            float e_dg_1761 = _exp2_243;
            float dq2_1762 = _tmem_load_40[1] * e_dg_1761;
            float dk2_1763 = _tmem_load_41[1] * e_dg_1761;
            dbp_1637[5] = dk2_1763 * k_t_1752;
            float dk2b_1764 = dk2_1763 * bt_t_1759;
            float dq_o_1765 = dq2_1762 + dqf_1756;
            float dk_o_1766 = dk2b_1764 + dkf_1757 + _tmem_load_42[1];
            float dg_o_1767 = __fmaf_rn(q_t_1755, dq2_1762, (dk2b_1764 - _tmem_load_42[1]) * k_t_1752) + dgf_1758;
            int goff_1768 = t_1748 * rs_g;
            int qoff_1769 = t_1748 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1768) + (0)) = dg_o_1767;
            if (t_1748 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1769) + (0)) = __float2bfloat16_rn(dq_o_1765);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1769) + (0)) = __float2bfloat16_rn(dk_o_1766);
            }
            int t_1770 = 8 * th + 4 + 2;
            float r_1771[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1771[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1770 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1770 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1772 = r_1771[0];
            unsigned int r_1773[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1773[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1770 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1770 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1774 = __uint_as_float(r_1773[0] << 16);
            unsigned int r_1775[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1775[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1776 = __uint_as_float(r_1775[0] << 16);
            float q_t_1777 = ((t_1770 >= clen) ? 0.0f : q_t_1776);
            float dqf_1778 = pool_f32[ew_1635 + 768 + d_lane];
            float dkf_1779 = pool_f32[ew_1635 + 1792 + d_lane];
            float dgf_1780 = pool_f32[ew_1635 + 2816 + d_lane];
            float bt_t_1781 = beta_s[t_1770];
            float _exp2_244 = approx_exp2(g_t_1772 - gn_1632);
            float e_off_1782 = _exp2_244;
            float _exp2_245 = approx_exp2(g_t_1772 - gc_1633);
            float e_dg_1783 = _exp2_245;
            float dq2_1784 = _tmem_load_40[2] * e_dg_1783;
            float dk2_1785 = _tmem_load_41[2] * e_dg_1783;
            dbp_1637[6] = dk2_1785 * k_t_1774;
            float dk2b_1786 = dk2_1785 * bt_t_1781;
            float dq_o_1787 = dq2_1784 + dqf_1778;
            float dk_o_1788 = dk2b_1786 + dkf_1779 + _tmem_load_42[2];
            float dg_o_1789 = __fmaf_rn(q_t_1777, dq2_1784, (dk2b_1786 - _tmem_load_42[2]) * k_t_1774) + dgf_1780;
            int goff_1790 = t_1770 * rs_g;
            int qoff_1791 = t_1770 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1790) + (0)) = dg_o_1789;
            if (t_1770 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1791) + (0)) = __float2bfloat16_rn(dq_o_1787);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1791) + (0)) = __float2bfloat16_rn(dk_o_1788);
            }
            int t_1792 = 8 * th + 4 + 3;
            float r_1793[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1793[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t_1792 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t_1792 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1794 = r_1793[0];
            unsigned int r_1795[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1795[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t_1792 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t_1792 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1796 = __uint_as_float(r_1795[0] << 16);
            unsigned int r_1797[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1797[0])) : "r"((qbase_1636 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1798 = __uint_as_float(r_1797[0] << 16);
            float q_t_1799 = ((t_1792 >= clen) ? 0.0f : q_t_1798);
            float dqf_1800 = pool_f32[ew_1635 + 896 + d_lane];
            float dkf_1801 = pool_f32[ew_1635 + 1920 + d_lane];
            float dgf_1802 = pool_f32[ew_1635 + 2944 + d_lane];
            float bt_t_1803 = beta_s[t_1792];
            float _exp2_246 = approx_exp2(g_t_1794 - gn_1632);
            float e_off_1804 = _exp2_246;
            float _exp2_247 = approx_exp2(g_t_1794 - gc_1633);
            float e_dg_1805 = _exp2_247;
            float dq2_1806 = _tmem_load_40[3] * e_dg_1805;
            float dk2_1807 = _tmem_load_41[3] * e_dg_1805;
            dbp_1637[7] = dk2_1807 * k_t_1796;
            float dk2b_1808 = dk2_1807 * bt_t_1803;
            float dq_o_1809 = dq2_1806 + dqf_1800;
            float dk_o_1810 = dk2b_1808 + dkf_1801 + _tmem_load_42[3];
            float dg_o_1811 = __fmaf_rn(q_t_1799, dq2_1806, (dk2b_1808 - _tmem_load_42[3]) * k_t_1796) + dgf_1802;
            int goff_1812 = t_1792 * rs_g;
            int qoff_1813 = t_1792 * qrs;
            *(reinterpret_cast<float*>(dg_base + goff_1812) + (0)) = dg_o_1811;
            if (t_1792 < clen) {
                *(reinterpret_cast<__nv_bfloat16*>(dq_base + qoff_1813) + (0)) = __float2bfloat16_rn(dq_o_1809);
                *(reinterpret_cast<__nv_bfloat16*>(dk_base + qoff_1813) + (0)) = __float2bfloat16_rn(dk_o_1810);
            }
            int b4_1814 = lane_1 >> 4 & 1;
            int b3_1815 = lane_1 >> 3 & 1;
            int b2_1816 = lane_1 >> 2 & 1;
            float v4_1817[4];
            float send_1818 = ((b4_1814 == 0) ? dbp_1637[4] : dbp_1637[0]);
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, send_1818, 16);
            v4_1817[0] = ((b4_1814 == 0) ? dbp_1637[0] : dbp_1637[4]) + _shfl_xor_27;
            float send_1819 = ((b4_1814 == 0) ? dbp_1637[5] : dbp_1637[1]);
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, send_1819, 16);
            v4_1817[1] = ((b4_1814 == 0) ? dbp_1637[1] : dbp_1637[5]) + _shfl_xor_28;
            float send_1820 = ((b4_1814 == 0) ? dbp_1637[6] : dbp_1637[2]);
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, send_1820, 16);
            v4_1817[2] = ((b4_1814 == 0) ? dbp_1637[2] : dbp_1637[6]) + _shfl_xor_29;
            float send_1821 = ((b4_1814 == 0) ? dbp_1637[7] : dbp_1637[3]);
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, send_1821, 16);
            v4_1817[3] = ((b4_1814 == 0) ? dbp_1637[3] : dbp_1637[7]) + _shfl_xor_30;
            float v2_1822[2];
            float send_1823 = ((b3_1815 == 0) ? v4_1817[2] : v4_1817[0]);
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, send_1823, 8);
            v2_1822[0] = ((b3_1815 == 0) ? v4_1817[0] : v4_1817[2]) + _shfl_xor_31;
            float send_1824 = ((b3_1815 == 0) ? v4_1817[3] : v4_1817[1]);
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, send_1824, 8);
            v2_1822[1] = ((b3_1815 == 0) ? v4_1817[1] : v4_1817[3]) + _shfl_xor_32;
            float send_1825 = ((b2_1816 == 0) ? v2_1822[1] : v2_1822[0]);
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, send_1825, 4);
            float v1_1826 = ((b2_1816 == 0) ? v2_1822[0] : v2_1822[1]) + _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, v1_1826, 2);
            v1_1826 = v1_1826 + _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, v1_1826, 1);
            v1_1826 = v1_1826 + _shfl_xor_35;
            if ((lane_1 & 3) == 0) {
                int e_star = lane_1 >> 2 & 7;
                pool_f32[28160 + qd * 64 + 8 * th + e_star] = v1_1826;
                pool_f32[28160 + qd * 64 + 16 + 8 * th + e_star] = v1_1630;
                pool_f32[28160 + qd * 64 + 32 + 8 * th + e_star] = v1;
                pool_f32[28160 + qd * 64 + 48 + 8 * th + e_star] = v1_1434;
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
                    for (int r_6 = 0; r_6 < 16; r_6++) {
                        int t_2 = 16 + r_6;
                        if (t_2 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_6 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
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
                    for (int r_8 = 0; r_8 < 16; r_8++) {
                        int t_3 = 32 + r_8;
                        if (t_3 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_8 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
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
                    for (int r_10 = 0; r_10 < 16; r_10++) {
                        int t_4 = 48 + r_10;
                        if (t_4 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_10 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
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
                long long fidx = ((long long)(row0_h + 32) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98304, dq_f + fidx, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102400, dk_f + fidx, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 106496, dg_f + fidx, 512, epi_full_addr);
                long long fidx_0 = ((long long)(row0_h + 33) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98816, dq_f + fidx_0, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102912, dk_f + fidx_0, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107008, dg_f + fidx_0, 512, epi_full_addr);
                long long fidx_1 = ((long long)(row0_h + 34) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99328, dq_f + fidx_1, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103424, dk_f + fidx_1, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107520, dg_f + fidx_1, 512, epi_full_addr);
                long long fidx_2 = ((long long)(row0_h + 35) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99840, dq_f + fidx_2, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103936, dk_f + fidx_2, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108032, dg_f + fidx_2, 512, epi_full_addr);
                long long fidx_3 = ((long long)(row0_h + 36) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100352, dq_f + fidx_3, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104448, dk_f + fidx_3, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108544, dg_f + fidx_3, 512, epi_full_addr);
                long long fidx_4 = ((long long)(row0_h + 37) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100864, dq_f + fidx_4, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104960, dk_f + fidx_4, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109056, dg_f + fidx_4, 512, epi_full_addr);
                long long fidx_5 = ((long long)(row0_h + 38) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101376, dq_f + fidx_5, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105472, dk_f + fidx_5, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109568, dg_f + fidx_5, 512, epi_full_addr);
                long long fidx_6 = ((long long)(row0_h + 39) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101888, dq_f + fidx_6, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105984, dk_f + fidx_6, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 110080, dg_f + fidx_6, 512, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup + 32, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup + 32, epi_full_addr);
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
                long long fidx_7 = ((long long)(row0_h + 40) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 65536, dq_f + fidx_7, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 69632, dk_f + fidx_7, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73728, dg_f + fidx_7, 512, epi_full_addr + 8);
                long long fidx_8 = ((long long)(row0_h + 41) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66048, dq_f + fidx_8, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70144, dk_f + fidx_8, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74240, dg_f + fidx_8, 512, epi_full_addr + 8);
                long long fidx_9 = ((long long)(row0_h + 42) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66560, dq_f + fidx_9, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70656, dk_f + fidx_9, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74752, dg_f + fidx_9, 512, epi_full_addr + 8);
                long long fidx_10 = ((long long)(row0_h + 43) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67072, dq_f + fidx_10, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71168, dk_f + fidx_10, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75264, dg_f + fidx_10, 512, epi_full_addr + 8);
                long long fidx_11 = ((long long)(row0_h + 44) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67584, dq_f + fidx_11, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71680, dk_f + fidx_11, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75776, dg_f + fidx_11, 512, epi_full_addr + 8);
                long long fidx_12 = ((long long)(row0_h + 45) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68096, dq_f + fidx_12, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72192, dk_f + fidx_12, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76288, dg_f + fidx_12, 512, epi_full_addr + 8);
                long long fidx_13 = ((long long)(row0_h + 46) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68608, dq_f + fidx_13, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72704, dk_f + fidx_13, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76800, dg_f + fidx_13, 512, epi_full_addr + 8);
                long long fidx_14 = ((long long)(row0_h + 47) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 69120, dq_f + fidx_14, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73216, dk_f + fidx_14, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 77312, dg_f + fidx_14, 512, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 40, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 40, epi_full_addr + 8);
                mbarrier_wait(p1a_t2_drained_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
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
                mbarrier_wait(epi_empty_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                long long fidx_15 = ((long long)(row0_h + 48) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98304, dq_f + fidx_15, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102400, dk_f + fidx_15, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 106496, dg_f + fidx_15, 512, epi_full_addr);
                long long fidx_16 = ((long long)(row0_h + 49) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98816, dq_f + fidx_16, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102912, dk_f + fidx_16, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107008, dg_f + fidx_16, 512, epi_full_addr);
                long long fidx_17 = ((long long)(row0_h + 50) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99328, dq_f + fidx_17, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103424, dk_f + fidx_17, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107520, dg_f + fidx_17, 512, epi_full_addr);
                long long fidx_18 = ((long long)(row0_h + 51) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99840, dq_f + fidx_18, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103936, dk_f + fidx_18, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108032, dg_f + fidx_18, 512, epi_full_addr);
                long long fidx_19 = ((long long)(row0_h + 52) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100352, dq_f + fidx_19, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104448, dk_f + fidx_19, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108544, dg_f + fidx_19, 512, epi_full_addr);
                long long fidx_20 = ((long long)(row0_h + 53) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100864, dq_f + fidx_20, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104960, dk_f + fidx_20, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109056, dg_f + fidx_20, 512, epi_full_addr);
                long long fidx_21 = ((long long)(row0_h + 54) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101376, dq_f + fidx_21, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105472, dk_f + fidx_21, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109568, dg_f + fidx_21, 512, epi_full_addr);
                long long fidx_22 = ((long long)(row0_h + 55) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101888, dq_f + fidx_22, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105984, dk_f + fidx_22, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 110080, dg_f + fidx_22, 512, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup + 48, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup + 48, epi_full_addr);
                mbarrier_wait(epi_empty_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                long long fidx_23 = ((long long)(row0_h + 56) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 65536, dq_f + fidx_23, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 69632, dk_f + fidx_23, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73728, dg_f + fidx_23, 512, epi_full_addr + 8);
                long long fidx_24 = ((long long)(row0_h + 57) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66048, dq_f + fidx_24, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70144, dk_f + fidx_24, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74240, dg_f + fidx_24, 512, epi_full_addr + 8);
                long long fidx_25 = ((long long)(row0_h + 58) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66560, dq_f + fidx_25, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70656, dk_f + fidx_25, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74752, dg_f + fidx_25, 512, epi_full_addr + 8);
                long long fidx_26 = ((long long)(row0_h + 59) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67072, dq_f + fidx_26, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71168, dk_f + fidx_26, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75264, dg_f + fidx_26, 512, epi_full_addr + 8);
                long long fidx_27 = ((long long)(row0_h + 60) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67584, dq_f + fidx_27, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71680, dk_f + fidx_27, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75776, dg_f + fidx_27, 512, epi_full_addr + 8);
                long long fidx_28 = ((long long)(row0_h + 61) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68096, dq_f + fidx_28, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72192, dk_f + fidx_28, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76288, dg_f + fidx_28, 512, epi_full_addr + 8);
                long long fidx_29 = ((long long)(row0_h + 62) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68608, dq_f + fidx_29, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72704, dk_f + fidx_29, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76800, dg_f + fidx_29, 512, epi_full_addr + 8);
                long long fidx_30 = ((long long)(row0_h + 63) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 69120, dq_f + fidx_30, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73216, dk_f + fidx_30, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 77312, dg_f + fidx_30, 512, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 56, epi_full_addr + 8);
                mbarrier_wait(p1b_ready_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(epi_empty_addr + 16, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 16, 14336);
                long long fidx_31 = ((long long)(row0_h + 16) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 81920, dq_f + fidx_31, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 86016, dk_f + fidx_31, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 90112, dg_f + fidx_31, 512, epi_full_addr + 16);
                long long fidx_32 = ((long long)(row0_h + 17) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 82432, dq_f + fidx_32, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 86528, dk_f + fidx_32, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 90624, dg_f + fidx_32, 512, epi_full_addr + 16);
                long long fidx_33 = ((long long)(row0_h + 18) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 82944, dq_f + fidx_33, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 87040, dk_f + fidx_33, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 91136, dg_f + fidx_33, 512, epi_full_addr + 16);
                long long fidx_34 = ((long long)(row0_h + 19) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 83456, dq_f + fidx_34, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 87552, dk_f + fidx_34, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 91648, dg_f + fidx_34, 512, epi_full_addr + 16);
                long long fidx_35 = ((long long)(row0_h + 20) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 83968, dq_f + fidx_35, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 88064, dk_f + fidx_35, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 92160, dg_f + fidx_35, 512, epi_full_addr + 16);
                long long fidx_36 = ((long long)(row0_h + 21) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 84480, dq_f + fidx_36, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 88576, dk_f + fidx_36, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 92672, dg_f + fidx_36, 512, epi_full_addr + 16);
                long long fidx_37 = ((long long)(row0_h + 22) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 84992, dq_f + fidx_37, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 89088, dk_f + fidx_37, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 93184, dg_f + fidx_37, 512, epi_full_addr + 16);
                long long fidx_38 = ((long long)(row0_h + 23) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 85504, dq_f + fidx_38, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 89600, dk_f + fidx_38, 512, epi_full_addr + 16);
                cp_async_bulk_gmem2smem(pool_f32_addr + 93696, dg_f + fidx_38, 512, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 94208, (&q_e), 0, qk_head_h, kq0_setup + 16, epi_full_addr + 16);
                tma_3d_gmem2smem(pool_f32_addr + 95232, (&q_e), 64, qk_head_h, kq0_setup + 16, epi_full_addr + 16);
                mbarrier_wait(epi_empty_addr + 24, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 24, 14336);
                long long fidx_39 = ((long long)(row0_h + 24) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 49152, dq_f + fidx_39, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 53248, dk_f + fidx_39, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 57344, dg_f + fidx_39, 512, epi_full_addr + 24);
                long long fidx_40 = ((long long)(row0_h + 25) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 49664, dq_f + fidx_40, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 53760, dk_f + fidx_40, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 57856, dg_f + fidx_40, 512, epi_full_addr + 24);
                long long fidx_41 = ((long long)(row0_h + 26) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 50176, dq_f + fidx_41, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 54272, dk_f + fidx_41, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 58368, dg_f + fidx_41, 512, epi_full_addr + 24);
                long long fidx_42 = ((long long)(row0_h + 27) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 50688, dq_f + fidx_42, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 54784, dk_f + fidx_42, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 58880, dg_f + fidx_42, 512, epi_full_addr + 24);
                long long fidx_43 = ((long long)(row0_h + 28) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 51200, dq_f + fidx_43, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 55296, dk_f + fidx_43, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 59392, dg_f + fidx_43, 512, epi_full_addr + 24);
                long long fidx_44 = ((long long)(row0_h + 29) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 51712, dq_f + fidx_44, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 55808, dk_f + fidx_44, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 59904, dg_f + fidx_44, 512, epi_full_addr + 24);
                long long fidx_45 = ((long long)(row0_h + 30) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 52224, dq_f + fidx_45, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 56320, dk_f + fidx_45, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 60416, dg_f + fidx_45, 512, epi_full_addr + 24);
                long long fidx_46 = ((long long)(row0_h + 31) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 52736, dq_f + fidx_46, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 56832, dk_f + fidx_46, 512, epi_full_addr + 24);
                cp_async_bulk_gmem2smem(pool_f32_addr + 60928, dg_f + fidx_46, 512, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 61440, (&q_e), 0, qk_head_h, kq0_setup + 24, epi_full_addr + 24);
                tma_3d_gmem2smem(pool_f32_addr + 62464, (&q_e), 64, qk_head_h, kq0_setup + 24, epi_full_addr + 24);
                mbarrier_wait(epi_empty_addr, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr, 14336);
                long long fidx_47 = ((long long)row0_h * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98304, dq_f + fidx_47, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102400, dk_f + fidx_47, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 106496, dg_f + fidx_47, 512, epi_full_addr);
                long long fidx_48 = ((long long)(row0_h + 1) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 98816, dq_f + fidx_48, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 102912, dk_f + fidx_48, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107008, dg_f + fidx_48, 512, epi_full_addr);
                long long fidx_49 = ((long long)(row0_h + 2) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99328, dq_f + fidx_49, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103424, dk_f + fidx_49, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 107520, dg_f + fidx_49, 512, epi_full_addr);
                long long fidx_50 = ((long long)(row0_h + 3) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 99840, dq_f + fidx_50, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 103936, dk_f + fidx_50, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108032, dg_f + fidx_50, 512, epi_full_addr);
                long long fidx_51 = ((long long)(row0_h + 4) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100352, dq_f + fidx_51, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104448, dk_f + fidx_51, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 108544, dg_f + fidx_51, 512, epi_full_addr);
                long long fidx_52 = ((long long)(row0_h + 5) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 100864, dq_f + fidx_52, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 104960, dk_f + fidx_52, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109056, dg_f + fidx_52, 512, epi_full_addr);
                long long fidx_53 = ((long long)(row0_h + 6) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101376, dq_f + fidx_53, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105472, dk_f + fidx_53, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 109568, dg_f + fidx_53, 512, epi_full_addr);
                long long fidx_54 = ((long long)(row0_h + 7) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 101888, dq_f + fidx_54, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 105984, dk_f + fidx_54, 512, epi_full_addr);
                cp_async_bulk_gmem2smem(pool_f32_addr + 110080, dg_f + fidx_54, 512, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 110592, (&q_e), 0, qk_head_h, kq0_setup, epi_full_addr);
                tma_3d_gmem2smem(pool_f32_addr + 111616, (&q_e), 64, qk_head_h, kq0_setup, epi_full_addr);
                mbarrier_wait(epi_empty_addr + 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(epi_full_addr + 8, 14336);
                long long fidx_55 = ((long long)(row0_h + 8) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 65536, dq_f + fidx_55, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 69632, dk_f + fidx_55, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73728, dg_f + fidx_55, 512, epi_full_addr + 8);
                long long fidx_56 = ((long long)(row0_h + 9) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66048, dq_f + fidx_56, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70144, dk_f + fidx_56, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74240, dg_f + fidx_56, 512, epi_full_addr + 8);
                long long fidx_57 = ((long long)(row0_h + 10) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 66560, dq_f + fidx_57, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 70656, dk_f + fidx_57, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 74752, dg_f + fidx_57, 512, epi_full_addr + 8);
                long long fidx_58 = ((long long)(row0_h + 11) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67072, dq_f + fidx_58, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71168, dk_f + fidx_58, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75264, dg_f + fidx_58, 512, epi_full_addr + 8);
                long long fidx_59 = ((long long)(row0_h + 12) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 67584, dq_f + fidx_59, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 71680, dk_f + fidx_59, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 75776, dg_f + fidx_59, 512, epi_full_addr + 8);
                long long fidx_60 = ((long long)(row0_h + 13) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68096, dq_f + fidx_60, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72192, dk_f + fidx_60, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76288, dg_f + fidx_60, 512, epi_full_addr + 8);
                long long fidx_61 = ((long long)(row0_h + 14) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 68608, dq_f + fidx_61, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 72704, dk_f + fidx_61, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 76800, dg_f + fidx_61, 512, epi_full_addr + 8);
                long long fidx_62 = ((long long)(row0_h + 15) * (long long)num_heads + (long long)hv_h) * 128;
                cp_async_bulk_gmem2smem(pool_f32_addr + 69120, dq_f + fidx_62, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 73216, dk_f + fidx_62, 512, epi_full_addr + 8);
                cp_async_bulk_gmem2smem(pool_f32_addr + 77312, dg_f + fidx_62, 512, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 77824, (&q_e), 0, qk_head_h, kq0_setup + 8, epi_full_addr + 8);
                tma_3d_gmem2smem(pool_f32_addr + 78848, (&q_e), 64, qk_head_h, kq0_setup + 8, epi_full_addr + 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"

