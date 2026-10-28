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
kernel_cake_kda_train_aa6b5c23c97e1cb9880a(const __grid_constant__ CUtensorMap dAqk, const __grid_constant__ CUtensorMap dAkk, const __grid_constant__ CUtensorMap gk, const __grid_constant__ CUtensorMap k_e, const __grid_constant__ CUtensorMap q_e, const __grid_constant__ CUtensorMap dq_f_t, const __grid_constant__ CUtensorMap dk_f_t, const __grid_constant__ CUtensorMap dg_f_t, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_ext, __nv_bfloat16* __restrict__ dbeta_ext, int num_heads, int num_qk_heads, int group)
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
            float r_2[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_2[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[1] = r_2[0];
            unsigned int r_3[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_3[0])) : "r"((qbase + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[0] = __uint_as_float(r_3[0] << 16);
            unsigned int r_4[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_4[0])) : "r"((qbase + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[1] = __uint_as_float(r_4[0] << 16);
            unsigned int r_5[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_5[0])) : "r"((kbase + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0 = __uint_as_float(r_5[0] << 16);
            unsigned int r_6[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_6[0])) : "r"((kbase + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1 = __uint_as_float(r_6[0] << 16);
            float2 _f32x2_mul_a_0 = make_float2(k_s0, k_s1);
            float2 _f32x2_mul_b_1 = make_float2(bv[0], bv[1]);
            float2 kp;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_1));
            __nv_bfloat16 kb0 = (__nv_bfloat16)kp.x;
            __nv_bfloat16 kb1 = (__nv_bfloat16)kp.y;
            kb_s[0] = (float)kb0;
            kb_s[1] = (float)kb1;
            float r_7[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_7[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[2] = r_7[0];
            float r_8[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_8[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[3] = r_8[0];
            unsigned int r_9[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_9[0])) : "r"((qbase + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[2] = __uint_as_float(r_9[0] << 16);
            unsigned int r_10[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_10[0])) : "r"((qbase + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[3] = __uint_as_float(r_10[0] << 16);
            unsigned int r_11[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_11[0])) : "r"((kbase + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_12 = __uint_as_float(r_11[0] << 16);
            unsigned int r_13[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_13[0])) : "r"((kbase + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_14 = __uint_as_float(r_13[0] << 16);
            float2 _f32x2_mul_a_2 = make_float2(k_s0_12, k_s1_14);
            float2 _f32x2_mul_b_3 = make_float2(bv[2], bv[3]);
            float2 kp_15;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
            __nv_bfloat16 kb0_16 = (__nv_bfloat16)kp_15.x;
            __nv_bfloat16 kb1_17 = (__nv_bfloat16)kp_15.y;
            kb_s[2] = (float)kb0_16;
            kb_s[3] = (float)kb1_17;
            float r_18[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_18[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[4] = r_18[0];
            float r_19[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_19[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[5] = r_19[0];
            unsigned int r_20[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_20[0])) : "r"((qbase + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[4] = __uint_as_float(r_20[0] << 16);
            unsigned int r_21[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_21[0])) : "r"((qbase + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[5] = __uint_as_float(r_21[0] << 16);
            unsigned int r_22[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_22[0])) : "r"((kbase + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_23 = __uint_as_float(r_22[0] << 16);
            unsigned int r_24[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_24[0])) : "r"((kbase + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_25 = __uint_as_float(r_24[0] << 16);
            float2 _f32x2_mul_a_4 = make_float2(k_s0_23, k_s1_25);
            float2 _f32x2_mul_b_5 = make_float2(bv[4], bv[5]);
            float2 kp_26;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            __nv_bfloat16 kb0_27 = (__nv_bfloat16)kp_26.x;
            __nv_bfloat16 kb1_28 = (__nv_bfloat16)kp_26.y;
            kb_s[4] = (float)kb0_27;
            kb_s[5] = (float)kb1_28;
            float r_29[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_29[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[6] = r_29[0];
            float r_30[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_30[0])) : "r"((gbase + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s[7] = r_30[0];
            unsigned int r_31[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_31[0])) : "r"((qbase + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[6] = __uint_as_float(r_31[0] << 16);
            unsigned int r_32[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_32[0])) : "r"((qbase + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s[7] = __uint_as_float(r_32[0] << 16);
            unsigned int r_33[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_33[0])) : "r"((kbase + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_34 = __uint_as_float(r_33[0] << 16);
            unsigned int r_35[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_35[0])) : "r"((kbase + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_36 = __uint_as_float(r_35[0] << 16);
            float2 _f32x2_mul_a_6 = make_float2(k_s0_34, k_s1_36);
            float2 _f32x2_mul_b_7 = make_float2(bv[6], bv[7]);
            float2 kp_37;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
            __nv_bfloat16 kb0_38 = (__nv_bfloat16)kp_37.x;
            __nv_bfloat16 kb1_39 = (__nv_bfloat16)kp_37.y;
            kb_s[6] = (float)kb0_38;
            kb_s[7] = (float)kb1_39;
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
                int kk_0 = w * 4 + 1;
                int a_1 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[1])) : "r"(a_1));
                tv[1] = ((lane_1 > (kk_0 & 15)) ? 0.0f : tv[1]);
                {
                    uint32_t _addr_9 = static_cast<uint32_t>(a_1);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(tv[1]) : "memory");
                }
                int kk_2 = w * 4 + 2;
                int a_3 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[2])) : "r"(a_3));
                tv[2] = ((lane_1 > (kk_2 & 15)) ? 0.0f : tv[2]);
                {
                    uint32_t _addr_10 = static_cast<uint32_t>(a_3);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(tv[2]) : "memory");
                }
                int kk_4 = w * 4 + 3;
                int a_5 = (dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_4 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[3])) : "r"(a_5));
                tv[3] = ((lane_1 > (kk_4 & 15)) ? 0.0f : tv[3]);
                {
                    uint32_t _addr_11 = static_cast<uint32_t>(a_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(tv[3]) : "memory");
                }
                int ta = (tiles_addr + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta), "f"(tv[0]), "f"(tv[1]), "f"(tv[2]), "f"(tv[3]) : "memory");
            }
            float r_40[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_40[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d = r_40[0] - 21.640425f;
            float2 _f2_0 = make_float2(a_d, a_d);
            float vq[8];
            float vk[8];
            float2 d2 = sub_f32x2(make_float2(g_s[0], g_s[1]), _f2_0);
            float ev0 = approx_exp2(d2.x);
            float ev1 = approx_exp2(d2.y);
            float2 _f2_1 = make_float2(ev0, ev1);
            float2 _f32x2_mul_a_12 = make_float2(q_s[0], q_s[1]);
            float2 vq2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f2_1));
            float2 _f32x2_mul_a_13 = make_float2(kb_s[0], kb_s[1]);
            float2 vk2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_f2_1));
            vq[0] = vq2.x;
            vq[1] = vq2.y;
            vk[0] = vk2.x;
            vk[1] = vk2.y;
            float2 d2_41 = sub_f32x2(make_float2(g_s[2], g_s[3]), _f2_0);
            float ev0_42 = approx_exp2(d2_41.x);
            float ev1_43 = approx_exp2(d2_41.y);
            float2 _f2_2 = make_float2(ev0_42, ev1_43);
            float2 _f32x2_mul_a_14 = make_float2(q_s[2], q_s[3]);
            float2 vq2_44;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f2_2));
            float2 _f32x2_mul_a_15 = make_float2(kb_s[2], kb_s[3]);
            float2 vk2_45;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f2_2));
            vq[2] = vq2_44.x;
            vq[3] = vq2_44.y;
            vk[2] = vk2_45.x;
            vk[3] = vk2_45.y;
            float2 d2_46 = sub_f32x2(make_float2(g_s[4], g_s[5]), _f2_0);
            float ev0_47 = approx_exp2(d2_46.x);
            float ev1_48 = approx_exp2(d2_46.y);
            float2 _f2_3 = make_float2(ev0_47, ev1_48);
            float2 _f32x2_mul_a_16 = make_float2(q_s[4], q_s[5]);
            float2 vq2_49;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f2_3));
            float2 _f32x2_mul_a_17 = make_float2(kb_s[4], kb_s[5]);
            float2 vk2_50;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_f2_3));
            vq[4] = vq2_49.x;
            vq[5] = vq2_49.y;
            vk[4] = vk2_50.x;
            vk[5] = vk2_50.y;
            float2 d2_51 = sub_f32x2(make_float2(g_s[6], g_s[7]), _f2_0);
            float ev0_52 = approx_exp2(d2_51.x);
            float ev1_53 = approx_exp2(d2_51.y);
            float2 _f2_4 = make_float2(ev0_52, ev1_53);
            float2 _f32x2_mul_a_18 = make_float2(q_s[6], q_s[7]);
            float2 vq2_54;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f2_4));
            float2 _f32x2_mul_a_19 = make_float2(kb_s[6], kb_s[7]);
            float2 vk2_55;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f2_4));
            vq[6] = vq2_54.x;
            vq[7] = vq2_54.y;
            vk[6] = vk2_55.x;
            vk[7] = vk2_55.y;
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
            float r_64[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_64[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[0] = r_64[0];
            float r_65[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_65[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[1] = r_65[0];
            unsigned int r_66[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_66[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[0] = __uint_as_float(r_66[0] << 16);
            unsigned int r_67[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_67[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[1] = __uint_as_float(r_67[0] << 16);
            unsigned int r_68[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_68[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_69 = __uint_as_float(r_68[0] << 16);
            unsigned int r_70[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_70[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_71 = __uint_as_float(r_70[0] << 16);
            float2 _f32x2_mul_a_20 = make_float2(k_s0_69, k_s1_71);
            float2 _f32x2_mul_b_21 = make_float2(bv_63[0], bv_63[1]);
            float2 kp_72;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            __nv_bfloat16 kb0_73 = (__nv_bfloat16)kp_72.x;
            __nv_bfloat16 kb1_74 = (__nv_bfloat16)kp_72.y;
            kb_s_62[0] = (float)kb0_73;
            kb_s_62[1] = (float)kb1_74;
            float r_75[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_75[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[2] = r_75[0];
            float r_76[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_76[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[3] = r_76[0];
            unsigned int r_77[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_77[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[2] = __uint_as_float(r_77[0] << 16);
            unsigned int r_78[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_78[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[3] = __uint_as_float(r_78[0] << 16);
            unsigned int r_79[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_79[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_80 = __uint_as_float(r_79[0] << 16);
            unsigned int r_81[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_81[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_82 = __uint_as_float(r_81[0] << 16);
            float2 _f32x2_mul_a_22 = make_float2(k_s0_80, k_s1_82);
            float2 _f32x2_mul_b_23 = make_float2(bv_63[2], bv_63[3]);
            float2 kp_83;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            __nv_bfloat16 kb0_84 = (__nv_bfloat16)kp_83.x;
            __nv_bfloat16 kb1_85 = (__nv_bfloat16)kp_83.y;
            kb_s_62[2] = (float)kb0_84;
            kb_s_62[3] = (float)kb1_85;
            float r_86[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_86[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[4] = r_86[0];
            float r_87[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_87[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[5] = r_87[0];
            unsigned int r_88[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_88[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[4] = __uint_as_float(r_88[0] << 16);
            unsigned int r_89[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_89[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[5] = __uint_as_float(r_89[0] << 16);
            unsigned int r_90[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_90[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_91 = __uint_as_float(r_90[0] << 16);
            unsigned int r_92[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_92[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_93 = __uint_as_float(r_92[0] << 16);
            float2 _f32x2_mul_a_24 = make_float2(k_s0_91, k_s1_93);
            float2 _f32x2_mul_b_25 = make_float2(bv_63[4], bv_63[5]);
            float2 kp_94;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
            __nv_bfloat16 kb0_95 = (__nv_bfloat16)kp_94.x;
            __nv_bfloat16 kb1_96 = (__nv_bfloat16)kp_94.y;
            kb_s_62[4] = (float)kb0_95;
            kb_s_62[5] = (float)kb1_96;
            float r_97[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_97[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[6] = r_97[0];
            float r_98[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_98[0])) : "r"((gbase_57 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_60[7] = r_98[0];
            unsigned int r_99[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_99[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[6] = __uint_as_float(r_99[0] << 16);
            unsigned int r_100[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_100[0])) : "r"((qbase_59 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_61[7] = __uint_as_float(r_100[0] << 16);
            unsigned int r_101[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_101[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_102 = __uint_as_float(r_101[0] << 16);
            unsigned int r_103[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_103[0])) : "r"((kbase_58 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_104 = __uint_as_float(r_103[0] << 16);
            float2 _f32x2_mul_a_26 = make_float2(k_s0_102, k_s1_104);
            float2 _f32x2_mul_b_27 = make_float2(bv_63[6], bv_63[7]);
            float2 kp_105;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            __nv_bfloat16 kb0_106 = (__nv_bfloat16)kp_105.x;
            __nv_bfloat16 kb1_107 = (__nv_bfloat16)kp_105.y;
            kb_s_62[6] = (float)kb0_106;
            kb_s_62[7] = (float)kb1_107;
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
            float r_108[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_108[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_109 = r_108[0];
            float2 _f2_5 = make_float2(a_d_109, a_d_109);
            float vq_110[8];
            float vk_111[8];
            float2 d2_112 = sub_f32x2(make_float2(g_s_60[0], g_s_60[1]), _f2_5);
            float ev0_113 = approx_exp2(d2_112.x);
            float ev1_114 = approx_exp2(d2_112.y);
            float2 _f2_6 = make_float2(ev0_113, ev1_114);
            float2 _f32x2_mul_a_28 = make_float2(q_s_61[0], q_s_61[1]);
            float2 vq2_115;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_115) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f2_6));
            float2 _f32x2_mul_a_29 = make_float2(kb_s_62[0], kb_s_62[1]);
            float2 vk2_116;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_f2_6));
            vq_110[0] = vq2_115.x;
            vq_110[1] = vq2_115.y;
            vk_111[0] = vk2_116.x;
            vk_111[1] = vk2_116.y;
            float2 d2_117 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_5);
            float ev0_118 = approx_exp2(d2_117.x);
            float ev1_119 = approx_exp2(d2_117.y);
            float2 _f2_7 = make_float2(ev0_118, ev1_119);
            float2 _f32x2_mul_a_30 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_120;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f2_7));
            float2 _f32x2_mul_a_31 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_121;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_121) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f2_7));
            vq_110[2] = vq2_120.x;
            vq_110[3] = vq2_120.y;
            vk_111[2] = vk2_121.x;
            vk_111[3] = vk2_121.y;
            float2 d2_122 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_5);
            float ev0_123 = approx_exp2(d2_122.x);
            float ev1_124 = approx_exp2(d2_122.y);
            float2 _f2_8 = make_float2(ev0_123, ev1_124);
            float2 _f32x2_mul_a_32 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_125;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_125) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f2_8));
            float2 _f32x2_mul_a_33 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_126;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f2_8));
            vq_110[4] = vq2_125.x;
            vq_110[5] = vq2_125.y;
            vk_111[4] = vk2_126.x;
            vk_111[5] = vk2_126.y;
            float2 d2_127 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_5);
            float ev0_128 = approx_exp2(d2_127.x);
            float ev1_129 = approx_exp2(d2_127.y);
            float2 _f2_9 = make_float2(ev0_128, ev1_129);
            float2 _f32x2_mul_a_34 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_130;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_130) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f2_9));
            float2 _f32x2_mul_a_35 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_131;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_131) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f2_9));
            vq_110[6] = vq2_130.x;
            vq_110[7] = vq2_130.y;
            vk_111[6] = vk2_131.x;
            vk_111[7] = vk2_131.y;
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
                int a_4 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_3 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[0])) : "r"(a_4));
                tv_2[0] = ((lane_1 > (kk_3 & 15)) ? 0.0f : tv_2[0]);
                {
                    uint32_t _addr_36 = static_cast<uint32_t>(a_4);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(tv_2[0]) : "memory");
                }
                int kk_0_2 = w * 4 + 1;
                int a_1_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[1])) : "r"(a_1_2));
                tv_2[1] = ((lane_1 > (kk_0_2 & 15)) ? 0.0f : tv_2[1]);
                {
                    uint32_t _addr_37 = static_cast<uint32_t>(a_1_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(tv_2[1]) : "memory");
                }
                int kk_2_2 = w * 4 + 2;
                int a_3_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[2])) : "r"(a_3_2));
                tv_2[2] = ((lane_1 > (kk_2_2 & 15)) ? 0.0f : tv_2[2]);
                {
                    uint32_t _addr_38 = static_cast<uint32_t>(a_3_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(tv_2[2]) : "memory");
                }
                int kk_4_2 = w * 4 + 3;
                int a_5_2 = (dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_4_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_2[3])) : "r"(a_5_2));
                tv_2[3] = ((lane_1 > (kk_4_2 & 15)) ? 0.0f : tv_2[3]);
                {
                    uint32_t _addr_39 = static_cast<uint32_t>(a_5_2);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(tv_2[3]) : "memory");
                }
                int ta_2 = (tiles_addr + 4096 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_2), "f"(tv_2[0]), "f"(tv_2[1]), "f"(tv_2[2]), "f"(tv_2[3]) : "memory");
            }
            float r_132[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_132[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_133 = r_132[0] - 21.640425f;
            float2 _f2_10 = make_float2(a_d_133, a_d_133);
            float vq_134[8];
            float vk_135[8];
            float2 d2_136 = sub_f32x2(make_float2(g_s_60[0], g_s_60[1]), _f2_10);
            float ev0_137 = approx_exp2(d2_136.x);
            float ev1_138 = approx_exp2(d2_136.y);
            float2 _f2_11 = make_float2(ev0_137, ev1_138);
            float2 _f32x2_mul_a_40 = make_float2(q_s_61[0], q_s_61[1]);
            float2 vq2_139;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_139) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f2_11));
            float2 _f32x2_mul_a_41 = make_float2(kb_s_62[0], kb_s_62[1]);
            float2 vk2_140;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_140) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_f2_11));
            vq_134[0] = vq2_139.x;
            vq_134[1] = vq2_139.y;
            vk_135[0] = vk2_140.x;
            vk_135[1] = vk2_140.y;
            float2 d2_141 = sub_f32x2(make_float2(g_s_60[2], g_s_60[3]), _f2_10);
            float ev0_142 = approx_exp2(d2_141.x);
            float ev1_143 = approx_exp2(d2_141.y);
            float2 _f2_12 = make_float2(ev0_142, ev1_143);
            float2 _f32x2_mul_a_42 = make_float2(q_s_61[2], q_s_61[3]);
            float2 vq2_144;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_144) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f2_12));
            float2 _f32x2_mul_a_43 = make_float2(kb_s_62[2], kb_s_62[3]);
            float2 vk2_145;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_145) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_f2_12));
            vq_134[2] = vq2_144.x;
            vq_134[3] = vq2_144.y;
            vk_135[2] = vk2_145.x;
            vk_135[3] = vk2_145.y;
            float2 d2_146 = sub_f32x2(make_float2(g_s_60[4], g_s_60[5]), _f2_10);
            float ev0_147 = approx_exp2(d2_146.x);
            float ev1_148 = approx_exp2(d2_146.y);
            float2 _f2_13 = make_float2(ev0_147, ev1_148);
            float2 _f32x2_mul_a_44 = make_float2(q_s_61[4], q_s_61[5]);
            float2 vq2_149;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_149) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f2_13));
            float2 _f32x2_mul_a_45 = make_float2(kb_s_62[4], kb_s_62[5]);
            float2 vk2_150;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_150) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f2_13));
            vq_134[4] = vq2_149.x;
            vq_134[5] = vq2_149.y;
            vk_135[4] = vk2_150.x;
            vk_135[5] = vk2_150.y;
            float2 d2_151 = sub_f32x2(make_float2(g_s_60[6], g_s_60[7]), _f2_10);
            float ev0_152 = approx_exp2(d2_151.x);
            float ev1_153 = approx_exp2(d2_151.y);
            float2 _f2_14 = make_float2(ev0_152, ev1_153);
            float2 _f32x2_mul_a_46 = make_float2(q_s_61[6], q_s_61[7]);
            float2 vq2_154;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_154) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f2_14));
            float2 _f32x2_mul_a_47 = make_float2(kb_s_62[6], kb_s_62[7]);
            float2 vk2_155;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_155) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_f2_14));
            vq_134[6] = vq2_154.x;
            vq_134[7] = vq2_154.y;
            vk_135[6] = vk2_155.x;
            vk_135[7] = vk2_155.y;
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
            float r_164[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_164[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[0] = r_164[0];
            float r_165[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_165[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[1] = r_165[0];
            unsigned int r_166[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_166[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[0] = __uint_as_float(r_166[0] << 16);
            unsigned int r_167[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_167[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[1] = __uint_as_float(r_167[0] << 16);
            unsigned int r_168[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_168[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_169 = __uint_as_float(r_168[0] << 16);
            unsigned int r_170[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_170[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_171 = __uint_as_float(r_170[0] << 16);
            float2 _f32x2_mul_a_48 = make_float2(k_s0_169, k_s1_171);
            float2 _f32x2_mul_b_49 = make_float2(bv_163[0], bv_163[1]);
            float2 kp_172;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_172) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
            __nv_bfloat16 kb0_173 = (__nv_bfloat16)kp_172.x;
            __nv_bfloat16 kb1_174 = (__nv_bfloat16)kp_172.y;
            kb_s_162[0] = (float)kb0_173;
            kb_s_162[1] = (float)kb1_174;
            float r_175[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_175[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[2] = r_175[0];
            float r_176[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_176[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[3] = r_176[0];
            unsigned int r_177[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_177[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[2] = __uint_as_float(r_177[0] << 16);
            unsigned int r_178[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_178[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[3] = __uint_as_float(r_178[0] << 16);
            unsigned int r_179[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_179[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_180 = __uint_as_float(r_179[0] << 16);
            unsigned int r_181[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_181[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_182 = __uint_as_float(r_181[0] << 16);
            float2 _f32x2_mul_a_50 = make_float2(k_s0_180, k_s1_182);
            float2 _f32x2_mul_b_51 = make_float2(bv_163[2], bv_163[3]);
            float2 kp_183;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_183) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            __nv_bfloat16 kb0_184 = (__nv_bfloat16)kp_183.x;
            __nv_bfloat16 kb1_185 = (__nv_bfloat16)kp_183.y;
            kb_s_162[2] = (float)kb0_184;
            kb_s_162[3] = (float)kb1_185;
            float r_186[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_186[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[4] = r_186[0];
            float r_187[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_187[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[5] = r_187[0];
            unsigned int r_188[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_188[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[4] = __uint_as_float(r_188[0] << 16);
            unsigned int r_189[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_189[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[5] = __uint_as_float(r_189[0] << 16);
            unsigned int r_190[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_190[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_191 = __uint_as_float(r_190[0] << 16);
            unsigned int r_192[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_192[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_193 = __uint_as_float(r_192[0] << 16);
            float2 _f32x2_mul_a_52 = make_float2(k_s0_191, k_s1_193);
            float2 _f32x2_mul_b_53 = make_float2(bv_163[4], bv_163[5]);
            float2 kp_194;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_194) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            __nv_bfloat16 kb0_195 = (__nv_bfloat16)kp_194.x;
            __nv_bfloat16 kb1_196 = (__nv_bfloat16)kp_194.y;
            kb_s_162[4] = (float)kb0_195;
            kb_s_162[5] = (float)kb1_196;
            float r_197[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_197[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[6] = r_197[0];
            float r_198[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_198[0])) : "r"((gbase_157 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_160[7] = r_198[0];
            unsigned int r_199[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_199[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[6] = __uint_as_float(r_199[0] << 16);
            unsigned int r_200[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_200[0])) : "r"((qbase_159 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_161[7] = __uint_as_float(r_200[0] << 16);
            unsigned int r_201[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_201[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_202 = __uint_as_float(r_201[0] << 16);
            unsigned int r_203[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_203[0])) : "r"((kbase_158 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_204 = __uint_as_float(r_203[0] << 16);
            float2 _f32x2_mul_a_54 = make_float2(k_s0_202, k_s1_204);
            float2 _f32x2_mul_b_55 = make_float2(bv_163[6], bv_163[7]);
            float2 kp_205;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_205) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
            __nv_bfloat16 kb0_206 = (__nv_bfloat16)kp_205.x;
            __nv_bfloat16 kb1_207 = (__nv_bfloat16)kp_205.y;
            kb_s_162[6] = (float)kb0_206;
            kb_s_162[7] = (float)kb1_207;
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
            float r_208[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_208[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_209 = r_208[0];
            float2 _f2_15 = make_float2(a_d_209, a_d_209);
            float vq_210[8];
            float vk_211[8];
            float2 d2_212 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_15);
            float ev0_213 = approx_exp2(d2_212.x);
            float ev1_214 = approx_exp2(d2_212.y);
            float2 _f2_16 = make_float2(ev0_213, ev1_214);
            float2 _f32x2_mul_a_56 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_215;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_215) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f2_16));
            float2 _f32x2_mul_a_57 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_216;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_216) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f2_16));
            vq_210[0] = vq2_215.x;
            vq_210[1] = vq2_215.y;
            vk_211[0] = vk2_216.x;
            vk_211[1] = vk2_216.y;
            float2 d2_217 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_15);
            float ev0_218 = approx_exp2(d2_217.x);
            float ev1_219 = approx_exp2(d2_217.y);
            float2 _f2_17 = make_float2(ev0_218, ev1_219);
            float2 _f32x2_mul_a_58 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_220;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_220) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f2_17));
            float2 _f32x2_mul_a_59 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_221;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_221) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f2_17));
            vq_210[2] = vq2_220.x;
            vq_210[3] = vq2_220.y;
            vk_211[2] = vk2_221.x;
            vk_211[3] = vk2_221.y;
            float2 d2_222 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_15);
            float ev0_223 = approx_exp2(d2_222.x);
            float ev1_224 = approx_exp2(d2_222.y);
            float2 _f2_18 = make_float2(ev0_223, ev1_224);
            float2 _f32x2_mul_a_60 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_225;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_225) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f2_18));
            float2 _f32x2_mul_a_61 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_226;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_226) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_f2_18));
            vq_210[4] = vq2_225.x;
            vq_210[5] = vq2_225.y;
            vk_211[4] = vk2_226.x;
            vk_211[5] = vk2_226.y;
            float2 d2_227 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_15);
            float ev0_228 = approx_exp2(d2_227.x);
            float ev1_229 = approx_exp2(d2_227.y);
            float2 _f2_19 = make_float2(ev0_228, ev1_229);
            float2 _f32x2_mul_a_62 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_230;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_230) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f2_19));
            float2 _f32x2_mul_a_63 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_231;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_231) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f2_19));
            vq_210[6] = vq2_230.x;
            vq_210[7] = vq2_230.y;
            vk_211[6] = vk2_231.x;
            vk_211[7] = vk2_231.y;
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
            float r_232[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_232[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_233 = r_232[0];
            float2 _f2_20 = make_float2(a_d_233, a_d_233);
            float vq_234[8];
            float vk_235[8];
            float2 d2_236 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_20);
            float ev0_237 = approx_exp2(d2_236.x);
            float ev1_238 = approx_exp2(d2_236.y);
            float2 _f2_21 = make_float2(ev0_237, ev1_238);
            float2 _f32x2_mul_a_64 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_239;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_239) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f2_21));
            float2 _f32x2_mul_a_65 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_240;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_240) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f2_21));
            vq_234[0] = vq2_239.x;
            vq_234[1] = vq2_239.y;
            vk_235[0] = vk2_240.x;
            vk_235[1] = vk2_240.y;
            float2 d2_241 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_20);
            float ev0_242 = approx_exp2(d2_241.x);
            float ev1_243 = approx_exp2(d2_241.y);
            float2 _f2_22 = make_float2(ev0_242, ev1_243);
            float2 _f32x2_mul_a_66 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_244;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_244) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f2_22));
            float2 _f32x2_mul_a_67 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_245;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_245) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f2_22));
            vq_234[2] = vq2_244.x;
            vq_234[3] = vq2_244.y;
            vk_235[2] = vk2_245.x;
            vk_235[3] = vk2_245.y;
            float2 d2_246 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_20);
            float ev0_247 = approx_exp2(d2_246.x);
            float ev1_248 = approx_exp2(d2_246.y);
            float2 _f2_23 = make_float2(ev0_247, ev1_248);
            float2 _f32x2_mul_a_68 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_249;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_249) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f2_23));
            float2 _f32x2_mul_a_69 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_250;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_250) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f2_23));
            vq_234[4] = vq2_249.x;
            vq_234[5] = vq2_249.y;
            vk_235[4] = vk2_250.x;
            vk_235[5] = vk2_250.y;
            float2 d2_251 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_20);
            float ev0_252 = approx_exp2(d2_251.x);
            float ev1_253 = approx_exp2(d2_251.y);
            float2 _f2_24 = make_float2(ev0_252, ev1_253);
            float2 _f32x2_mul_a_70 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_254;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_254) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f2_24));
            float2 _f32x2_mul_a_71 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_255;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_255) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f2_24));
            vq_234[6] = vq2_254.x;
            vq_234[7] = vq2_254.y;
            vk_235[6] = vk2_255.x;
            vk_235[7] = vk2_255.y;
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
                int a_8 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_7 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[0])) : "r"(a_8));
                tv_5[0] = ((lane_1 > (kk_7 & 15)) ? 0.0f : tv_5[0]);
                {
                    uint32_t _addr_72 = static_cast<uint32_t>(a_8);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(tv_5[0]) : "memory");
                }
                int kk_0_5 = w * 4 + 1;
                int a_1_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_0_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[1])) : "r"(a_1_5));
                tv_5[1] = ((lane_1 > (kk_0_5 & 15)) ? 0.0f : tv_5[1]);
                {
                    uint32_t _addr_73 = static_cast<uint32_t>(a_1_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(tv_5[1]) : "memory");
                }
                int kk_2_5 = w * 4 + 2;
                int a_3_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_2_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[2])) : "r"(a_3_5));
                tv_5[2] = ((lane_1 > (kk_2_5 & 15)) ? 0.0f : tv_5[2]);
                {
                    uint32_t _addr_74 = static_cast<uint32_t>(a_3_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(tv_5[2]) : "memory");
                }
                int kk_4_5 = w * 4 + 3;
                int a_5_5 = (dab_addr + (unsigned int)((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 ^ ((32 + lane_1) / 32 * 4096 + kk_4_5 * 128 + (32 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_5[3])) : "r"(a_5_5));
                tv_5[3] = ((lane_1 > (kk_4_5 & 15)) ? 0.0f : tv_5[3]);
                {
                    uint32_t _addr_75 = static_cast<uint32_t>(a_5_5);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(tv_5[3]) : "memory");
                }
                int ta_5 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_5), "f"(tv_5[0]), "f"(tv_5[1]), "f"(tv_5[2]), "f"(tv_5[3]) : "memory");
            }
            float r_256[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_256[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_257 = r_256[0] - 21.640425f;
            float2 _f2_25 = make_float2(a_d_257, a_d_257);
            float vq_258[8];
            float vk_259[8];
            float2 d2_260 = sub_f32x2(make_float2(g_s_160[0], g_s_160[1]), _f2_25);
            float ev0_261 = approx_exp2(d2_260.x);
            float ev1_262 = approx_exp2(d2_260.y);
            float2 _f2_26 = make_float2(ev0_261, ev1_262);
            float2 _f32x2_mul_a_76 = make_float2(q_s_161[0], q_s_161[1]);
            float2 vq2_263;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_263) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f2_26));
            float2 _f32x2_mul_a_77 = make_float2(kb_s_162[0], kb_s_162[1]);
            float2 vk2_264;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_264) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f2_26));
            vq_258[0] = vq2_263.x;
            vq_258[1] = vq2_263.y;
            vk_259[0] = vk2_264.x;
            vk_259[1] = vk2_264.y;
            float2 d2_265 = sub_f32x2(make_float2(g_s_160[2], g_s_160[3]), _f2_25);
            float ev0_266 = approx_exp2(d2_265.x);
            float ev1_267 = approx_exp2(d2_265.y);
            float2 _f2_27 = make_float2(ev0_266, ev1_267);
            float2 _f32x2_mul_a_78 = make_float2(q_s_161[2], q_s_161[3]);
            float2 vq2_268;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_268) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f2_27));
            float2 _f32x2_mul_a_79 = make_float2(kb_s_162[2], kb_s_162[3]);
            float2 vk2_269;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_269) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f2_27));
            vq_258[2] = vq2_268.x;
            vq_258[3] = vq2_268.y;
            vk_259[2] = vk2_269.x;
            vk_259[3] = vk2_269.y;
            float2 d2_270 = sub_f32x2(make_float2(g_s_160[4], g_s_160[5]), _f2_25);
            float ev0_271 = approx_exp2(d2_270.x);
            float ev1_272 = approx_exp2(d2_270.y);
            float2 _f2_28 = make_float2(ev0_271, ev1_272);
            float2 _f32x2_mul_a_80 = make_float2(q_s_161[4], q_s_161[5]);
            float2 vq2_273;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_273) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f2_28));
            float2 _f32x2_mul_a_81 = make_float2(kb_s_162[4], kb_s_162[5]);
            float2 vk2_274;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_274) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f2_28));
            vq_258[4] = vq2_273.x;
            vq_258[5] = vq2_273.y;
            vk_259[4] = vk2_274.x;
            vk_259[5] = vk2_274.y;
            float2 d2_275 = sub_f32x2(make_float2(g_s_160[6], g_s_160[7]), _f2_25);
            float ev0_276 = approx_exp2(d2_275.x);
            float ev1_277 = approx_exp2(d2_275.y);
            float2 _f2_29 = make_float2(ev0_276, ev1_277);
            float2 _f32x2_mul_a_82 = make_float2(q_s_161[6], q_s_161[7]);
            float2 vq2_278;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_278) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f2_29));
            float2 _f32x2_mul_a_83 = make_float2(kb_s_162[6], kb_s_162[7]);
            float2 vk2_279;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_279) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f2_29));
            vq_258[6] = vq2_278.x;
            vq_258[7] = vq2_278.y;
            vk_259[6] = vk2_279.x;
            vk_259[7] = vk2_279.y;
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
            float r_288[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_288[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[0] = r_288[0];
            float r_289[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_289[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[1] = r_289[0];
            unsigned int r_290[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_290[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[0] = __uint_as_float(r_290[0] << 16);
            unsigned int r_291[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_291[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[1] = __uint_as_float(r_291[0] << 16);
            unsigned int r_292[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_292[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_293 = __uint_as_float(r_292[0] << 16);
            unsigned int r_294[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_294[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_295 = __uint_as_float(r_294[0] << 16);
            float2 _f32x2_mul_a_84 = make_float2(k_s0_293, k_s1_295);
            float2 _f32x2_mul_b_85 = make_float2(bv_287[0], bv_287[1]);
            float2 kp_296;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_296) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
            __nv_bfloat16 kb0_297 = (__nv_bfloat16)kp_296.x;
            __nv_bfloat16 kb1_298 = (__nv_bfloat16)kp_296.y;
            kb_s_286[0] = (float)kb0_297;
            kb_s_286[1] = (float)kb1_298;
            float r_299[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_299[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[2] = r_299[0];
            float r_300[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_300[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[3] = r_300[0];
            unsigned int r_301[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_301[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[2] = __uint_as_float(r_301[0] << 16);
            unsigned int r_302[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_302[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[3] = __uint_as_float(r_302[0] << 16);
            unsigned int r_303[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_303[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_304 = __uint_as_float(r_303[0] << 16);
            unsigned int r_305[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_305[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_306 = __uint_as_float(r_305[0] << 16);
            float2 _f32x2_mul_a_86 = make_float2(k_s0_304, k_s1_306);
            float2 _f32x2_mul_b_87 = make_float2(bv_287[2], bv_287[3]);
            float2 kp_307;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_307) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
            __nv_bfloat16 kb0_308 = (__nv_bfloat16)kp_307.x;
            __nv_bfloat16 kb1_309 = (__nv_bfloat16)kp_307.y;
            kb_s_286[2] = (float)kb0_308;
            kb_s_286[3] = (float)kb1_309;
            float r_310[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_310[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[4] = r_310[0];
            float r_311[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_311[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[5] = r_311[0];
            unsigned int r_312[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_312[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[4] = __uint_as_float(r_312[0] << 16);
            unsigned int r_313[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_313[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[5] = __uint_as_float(r_313[0] << 16);
            unsigned int r_314[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_314[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_315 = __uint_as_float(r_314[0] << 16);
            unsigned int r_316[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_316[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_317 = __uint_as_float(r_316[0] << 16);
            float2 _f32x2_mul_a_88 = make_float2(k_s0_315, k_s1_317);
            float2 _f32x2_mul_b_89 = make_float2(bv_287[4], bv_287[5]);
            float2 kp_318;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_318) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
            __nv_bfloat16 kb0_319 = (__nv_bfloat16)kp_318.x;
            __nv_bfloat16 kb1_320 = (__nv_bfloat16)kp_318.y;
            kb_s_286[4] = (float)kb0_319;
            kb_s_286[5] = (float)kb1_320;
            float r_321[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_321[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[6] = r_321[0];
            float r_322[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_322[0])) : "r"((gbase_281 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            g_s_284[7] = r_322[0];
            unsigned int r_323[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_323[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[6] = __uint_as_float(r_323[0] << 16);
            unsigned int r_324[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_324[0])) : "r"((qbase_283 + (dcol / 64 * 2048 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 2048 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            q_s_285[7] = __uint_as_float(r_324[0] << 16);
            unsigned int r_325[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_325[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s0_326 = __uint_as_float(r_325[0] << 16);
            unsigned int r_327[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_327[0])) : "r"((kbase_282 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_s1_328 = __uint_as_float(r_327[0] << 16);
            float2 _f32x2_mul_a_90 = make_float2(k_s0_326, k_s1_328);
            float2 _f32x2_mul_b_91 = make_float2(bv_287[6], bv_287[7]);
            float2 kp_329;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kp_329) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
            __nv_bfloat16 kb0_330 = (__nv_bfloat16)kp_329.x;
            __nv_bfloat16 kb1_331 = (__nv_bfloat16)kp_329.y;
            kb_s_286[6] = (float)kb0_330;
            kb_s_286[7] = (float)kb1_331;
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
            float r_332[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_332[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_333 = r_332[0];
            float2 _f2_30 = make_float2(a_d_333, a_d_333);
            float vq_334[8];
            float vk_335[8];
            float2 d2_336 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_30);
            float ev0_337 = approx_exp2(d2_336.x);
            float ev1_338 = approx_exp2(d2_336.y);
            float2 _f2_31 = make_float2(ev0_337, ev1_338);
            float2 _f32x2_mul_a_92 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_339;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_339) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f2_31));
            float2 _f32x2_mul_a_93 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_340;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_340) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f2_31));
            vq_334[0] = vq2_339.x;
            vq_334[1] = vq2_339.y;
            vk_335[0] = vk2_340.x;
            vk_335[1] = vk2_340.y;
            float2 d2_341 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_30);
            float ev0_342 = approx_exp2(d2_341.x);
            float ev1_343 = approx_exp2(d2_341.y);
            float2 _f2_32 = make_float2(ev0_342, ev1_343);
            float2 _f32x2_mul_a_94 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_344;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_344) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&_f2_32));
            float2 _f32x2_mul_a_95 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_345;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_345) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f2_32));
            vq_334[2] = vq2_344.x;
            vq_334[3] = vq2_344.y;
            vk_335[2] = vk2_345.x;
            vk_335[3] = vk2_345.y;
            float2 d2_346 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_30);
            float ev0_347 = approx_exp2(d2_346.x);
            float ev1_348 = approx_exp2(d2_346.y);
            float2 _f2_33 = make_float2(ev0_347, ev1_348);
            float2 _f32x2_mul_a_96 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_349;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_349) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f2_33));
            float2 _f32x2_mul_a_97 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_350;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_350) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f2_33));
            vq_334[4] = vq2_349.x;
            vq_334[5] = vq2_349.y;
            vk_335[4] = vk2_350.x;
            vk_335[5] = vk2_350.y;
            float2 d2_351 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_30);
            float ev0_352 = approx_exp2(d2_351.x);
            float ev1_353 = approx_exp2(d2_351.y);
            float2 _f2_34 = make_float2(ev0_352, ev1_353);
            float2 _f32x2_mul_a_98 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_354;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_354) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_f2_34));
            float2 _f32x2_mul_a_99 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_355;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_355) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f2_34));
            vq_334[6] = vq2_354.x;
            vq_334[7] = vq2_354.y;
            vk_335[6] = vk2_355.x;
            vk_335[7] = vk2_355.y;
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
            float r_356[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_356[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_357 = r_356[0];
            float2 _f2_35 = make_float2(a_d_357, a_d_357);
            float vq_358[8];
            float vk_359[8];
            float2 d2_360 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_35);
            float ev0_361 = approx_exp2(d2_360.x);
            float ev1_362 = approx_exp2(d2_360.y);
            float2 _f2_36 = make_float2(ev0_361, ev1_362);
            float2 _f32x2_mul_a_100 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_363;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_363) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f2_36));
            float2 _f32x2_mul_a_101 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_364;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_364) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f2_36));
            vq_358[0] = vq2_363.x;
            vq_358[1] = vq2_363.y;
            vk_359[0] = vk2_364.x;
            vk_359[1] = vk2_364.y;
            float2 d2_365 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_35);
            float ev0_366 = approx_exp2(d2_365.x);
            float ev1_367 = approx_exp2(d2_365.y);
            float2 _f2_37 = make_float2(ev0_366, ev1_367);
            float2 _f32x2_mul_a_102 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_368;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_368) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f2_37));
            float2 _f32x2_mul_a_103 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_369;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_369) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f2_37));
            vq_358[2] = vq2_368.x;
            vq_358[3] = vq2_368.y;
            vk_359[2] = vk2_369.x;
            vk_359[3] = vk2_369.y;
            float2 d2_370 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_35);
            float ev0_371 = approx_exp2(d2_370.x);
            float ev1_372 = approx_exp2(d2_370.y);
            float2 _f2_38 = make_float2(ev0_371, ev1_372);
            float2 _f32x2_mul_a_104 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_373;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_373) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f2_38));
            float2 _f32x2_mul_a_105 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_374;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_374) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f2_38));
            vq_358[4] = vq2_373.x;
            vq_358[5] = vq2_373.y;
            vk_359[4] = vk2_374.x;
            vk_359[5] = vk2_374.y;
            float2 d2_375 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_35);
            float ev0_376 = approx_exp2(d2_375.x);
            float ev1_377 = approx_exp2(d2_375.y);
            float2 _f2_39 = make_float2(ev0_376, ev1_377);
            float2 _f32x2_mul_a_106 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_378;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_378) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&_f2_39));
            float2 _f32x2_mul_a_107 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_379;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_379) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f2_39));
            vq_358[6] = vq2_378.x;
            vq_358[7] = vq2_378.y;
            vk_359[6] = vk2_379.x;
            vk_359[7] = vk2_379.y;
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
            float r_380[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_380[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_381 = r_380[0];
            float2 _f2_40 = make_float2(a_d_381, a_d_381);
            float vq_382[8];
            float vk_383[8];
            float2 d2_384 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_40);
            float ev0_385 = approx_exp2(d2_384.x);
            float ev1_386 = approx_exp2(d2_384.y);
            float2 _f2_41 = make_float2(ev0_385, ev1_386);
            float2 _f32x2_mul_a_108 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_387;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_387) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_f2_41));
            float2 _f32x2_mul_a_109 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_388;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_388) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f2_41));
            vq_382[0] = vq2_387.x;
            vq_382[1] = vq2_387.y;
            vk_383[0] = vk2_388.x;
            vk_383[1] = vk2_388.y;
            float2 d2_389 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_40);
            float ev0_390 = approx_exp2(d2_389.x);
            float ev1_391 = approx_exp2(d2_389.y);
            float2 _f2_42 = make_float2(ev0_390, ev1_391);
            float2 _f32x2_mul_a_110 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_392;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_392) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&_f2_42));
            float2 _f32x2_mul_a_111 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_393;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_393) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_f2_42));
            vq_382[2] = vq2_392.x;
            vq_382[3] = vq2_392.y;
            vk_383[2] = vk2_393.x;
            vk_383[3] = vk2_393.y;
            float2 d2_394 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_40);
            float ev0_395 = approx_exp2(d2_394.x);
            float ev1_396 = approx_exp2(d2_394.y);
            float2 _f2_43 = make_float2(ev0_395, ev1_396);
            float2 _f32x2_mul_a_112 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_397;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_397) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&_f2_43));
            float2 _f32x2_mul_a_113 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_398;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_398) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f2_43));
            vq_382[4] = vq2_397.x;
            vq_382[5] = vq2_397.y;
            vk_383[4] = vk2_398.x;
            vk_383[5] = vk2_398.y;
            float2 d2_399 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_40);
            float ev0_400 = approx_exp2(d2_399.x);
            float ev1_401 = approx_exp2(d2_399.y);
            float2 _f2_44 = make_float2(ev0_400, ev1_401);
            float2 _f32x2_mul_a_114 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_402;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_402) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_f2_44));
            float2 _f32x2_mul_a_115 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_403;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_403) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_f2_44));
            vq_382[6] = vq2_402.x;
            vq_382[7] = vq2_402.y;
            vk_383[6] = vk2_403.x;
            vk_383[7] = vk2_403.y;
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
                int a_12 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_11 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[0])) : "r"(a_12));
                tv_9[0] = ((lane_1 > (kk_11 & 15)) ? 0.0f : tv_9[0]);
                {
                    uint32_t _addr_116 = static_cast<uint32_t>(a_12);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(tv_9[0]) : "memory");
                }
                int kk_0_9 = w * 4 + 1;
                int a_1_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_0_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_0_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[1])) : "r"(a_1_9));
                tv_9[1] = ((lane_1 > (kk_0_9 & 15)) ? 0.0f : tv_9[1]);
                {
                    uint32_t _addr_117 = static_cast<uint32_t>(a_1_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(tv_9[1]) : "memory");
                }
                int kk_2_9 = w * 4 + 2;
                int a_3_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_2_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_2_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[2])) : "r"(a_3_9));
                tv_9[2] = ((lane_1 > (kk_2_9 & 15)) ? 0.0f : tv_9[2]);
                {
                    uint32_t _addr_118 = static_cast<uint32_t>(a_3_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(tv_9[2]) : "memory");
                }
                int kk_4_9 = w * 4 + 3;
                int a_5_9 = (dab_addr + 8192 + (unsigned int)((48 + lane_1) / 32 * 4096 + kk_4_9 * 128 + (48 + lane_1) % 32 * 4 ^ ((48 + lane_1) / 32 * 4096 + kk_4_9 * 128 + (48 + lane_1) % 32 * 4 >> 7 & 7) << 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv_9[3])) : "r"(a_5_9));
                tv_9[3] = ((lane_1 > (kk_4_9 & 15)) ? 0.0f : tv_9[3]);
                {
                    uint32_t _addr_119 = static_cast<uint32_t>(a_5_9);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(tv_9[3]) : "memory");
                }
                int ta_9 = (tiles_addr + 2048 + (unsigned int)(w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 ^ (w * 4 / 32 * 2048 + lane_1 * 128 + w * 4 % 32 * 4 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ta_9), "f"(tv_9[0]), "f"(tv_9[1]), "f"(tv_9[2]), "f"(tv_9[3]) : "memory");
            }
            float r_404[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_404[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a_d_405 = r_404[0] - 21.640425f;
            float2 _f2_45 = make_float2(a_d_405, a_d_405);
            float vq_406[8];
            float vk_407[8];
            float2 d2_408 = sub_f32x2(make_float2(g_s_284[0], g_s_284[1]), _f2_45);
            float ev0_409 = approx_exp2(d2_408.x);
            float ev1_410 = approx_exp2(d2_408.y);
            float2 _f2_46 = make_float2(ev0_409, ev1_410);
            float2 _f32x2_mul_a_120 = make_float2(q_s_285[0], q_s_285[1]);
            float2 vq2_411;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_411) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&_f2_46));
            float2 _f32x2_mul_a_121 = make_float2(kb_s_286[0], kb_s_286[1]);
            float2 vk2_412;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_412) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f2_46));
            vq_406[0] = vq2_411.x;
            vq_406[1] = vq2_411.y;
            vk_407[0] = vk2_412.x;
            vk_407[1] = vk2_412.y;
            float2 d2_413 = sub_f32x2(make_float2(g_s_284[2], g_s_284[3]), _f2_45);
            float ev0_414 = approx_exp2(d2_413.x);
            float ev1_415 = approx_exp2(d2_413.y);
            float2 _f2_47 = make_float2(ev0_414, ev1_415);
            float2 _f32x2_mul_a_122 = make_float2(q_s_285[2], q_s_285[3]);
            float2 vq2_416;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_416) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&_f2_47));
            float2 _f32x2_mul_a_123 = make_float2(kb_s_286[2], kb_s_286[3]);
            float2 vk2_417;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_417) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_f2_47));
            vq_406[2] = vq2_416.x;
            vq_406[3] = vq2_416.y;
            vk_407[2] = vk2_417.x;
            vk_407[3] = vk2_417.y;
            float2 d2_418 = sub_f32x2(make_float2(g_s_284[4], g_s_284[5]), _f2_45);
            float ev0_419 = approx_exp2(d2_418.x);
            float ev1_420 = approx_exp2(d2_418.y);
            float2 _f2_48 = make_float2(ev0_419, ev1_420);
            float2 _f32x2_mul_a_124 = make_float2(q_s_285[4], q_s_285[5]);
            float2 vq2_421;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_421) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&_f2_48));
            float2 _f32x2_mul_a_125 = make_float2(kb_s_286[4], kb_s_286[5]);
            float2 vk2_422;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_422) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f2_48));
            vq_406[4] = vq2_421.x;
            vq_406[5] = vq2_421.y;
            vk_407[4] = vk2_422.x;
            vk_407[5] = vk2_422.y;
            float2 d2_423 = sub_f32x2(make_float2(g_s_284[6], g_s_284[7]), _f2_45);
            float ev0_424 = approx_exp2(d2_423.x);
            float ev1_425 = approx_exp2(d2_423.y);
            float2 _f2_49 = make_float2(ev0_424, ev1_425);
            float2 _f32x2_mul_a_126 = make_float2(q_s_285[6], q_s_285[7]);
            float2 vq2_426;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vq2_426) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&_f2_49));
            float2 _f32x2_mul_a_127 = make_float2(kb_s_286[6], kb_s_286[7]);
            float2 vk2_427;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&vk2_427) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&_f2_49));
            vq_406[6] = vq2_426.x;
            vq_406[7] = vq2_426.y;
            vk_407[6] = vk2_427.x;
            vk_407[7] = vk2_427.y;
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
            float r_431[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_431[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0 = r_431[0];
            float r_432[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_432[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1 = r_432[0];
            float gs[8];
            float ks[8];
            float r_433[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_433[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[0] = r_433[0];
            unsigned int r_434[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_434[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[0] = __uint_as_float(r_434[0] << 16);
            float r_435[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_435[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[1] = r_435[0];
            unsigned int r_436[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_436[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[1] = __uint_as_float(r_436[0] << 16);
            float r_437[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_437[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[2] = r_437[0];
            unsigned int r_438[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_438[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[2] = __uint_as_float(r_438[0] << 16);
            float r_439[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_439[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[3] = r_439[0];
            unsigned int r_440[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_440[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[3] = __uint_as_float(r_440[0] << 16);
            float r_441[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_441[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[4] = r_441[0];
            unsigned int r_442[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_442[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[4] = __uint_as_float(r_442[0] << 16);
            float r_443[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_443[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[5] = r_443[0];
            unsigned int r_444[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_444[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[5] = __uint_as_float(r_444[0] << 16);
            float r_445[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_445[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[6] = r_445[0];
            unsigned int r_446[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_446[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[6] = __uint_as_float(r_446[0] << 16);
            float r_447[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_447[0])) : "r"((gbase_429 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs[7] = r_447[0];
            unsigned int r_448[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_448[0])) : "r"((kbase_430 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks[7] = __uint_as_float(r_448[0] << 16);
            float2 _f2_50 = make_float2(a0, a0);
            float v[8];
            float2 d2_449 = sub_f32x2(_f2_50, make_float2(gs[0], gs[1]));
            float e0 = approx_exp2(d2_449.x);
            float e1 = approx_exp2(d2_449.y);
            float2 _f32x2_mul_a_128 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_129 = make_float2(e0, e1);
            float2 v2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
            v[0] = v2.x;
            v[1] = v2.y;
            float2 d2_450 = sub_f32x2(_f2_50, make_float2(gs[2], gs[3]));
            float e0_451 = approx_exp2(d2_450.x);
            float e1_452 = approx_exp2(d2_450.y);
            float2 _f32x2_mul_a_130 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_131 = make_float2(e0_451, e1_452);
            float2 v2_453;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_453) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
            v[2] = v2_453.x;
            v[3] = v2_453.y;
            float2 d2_454 = sub_f32x2(_f2_50, make_float2(gs[4], gs[5]));
            float e0_455 = approx_exp2(d2_454.x);
            float e1_456 = approx_exp2(d2_454.y);
            float2 _f32x2_mul_a_132 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_133 = make_float2(e0_455, e1_456);
            float2 v2_457;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_457) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
            v[4] = v2_457.x;
            v[5] = v2_457.y;
            float2 d2_458 = sub_f32x2(_f2_50, make_float2(gs[6], gs[7]));
            float e0_459 = approx_exp2(d2_458.x);
            float e1_460 = approx_exp2(d2_458.y);
            float2 _f32x2_mul_a_134 = make_float2(ks[6], ks[7]);
            float2 _f32x2_mul_b_135 = make_float2(e0_459, e1_460);
            float2 v2_461;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_461) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
            v[6] = v2_461.x;
            v[7] = v2_461.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]) : "memory");
            float2 _f2_51 = make_float2(a1, a1);
            float v_462[8];
            float2 d2_463 = sub_f32x2(_f2_51, make_float2(gs[0], gs[1]));
            float e0_464 = approx_exp2(d2_463.x);
            float e1_465 = approx_exp2(d2_463.y);
            float2 _f32x2_mul_a_136 = make_float2(ks[0], ks[1]);
            float2 _f32x2_mul_b_137 = make_float2(e0_464, e1_465);
            float2 v2_466;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_466) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
            v_462[0] = v2_466.x;
            v_462[1] = v2_466.y;
            float2 d2_467 = sub_f32x2(_f2_51, make_float2(gs[2], gs[3]));
            float e0_468 = approx_exp2(d2_467.x);
            float e1_469 = approx_exp2(d2_467.y);
            float2 _f32x2_mul_a_138 = make_float2(ks[2], ks[3]);
            float2 _f32x2_mul_b_139 = make_float2(e0_468, e1_469);
            float2 v2_470;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_470) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
            v_462[2] = v2_470.x;
            v_462[3] = v2_470.y;
            float2 d2_471 = sub_f32x2(_f2_51, make_float2(gs[4], gs[5]));
            float e0_472 = approx_exp2(d2_471.x);
            float e1_473 = approx_exp2(d2_471.y);
            float2 _f32x2_mul_a_140 = make_float2(ks[4], ks[5]);
            float2 _f32x2_mul_b_141 = make_float2(e0_472, e1_473);
            float2 v2_474;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_474) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
            v_462[4] = v2_474.x;
            v_462[5] = v2_474.y;
            float2 d2_475 = sub_f32x2(_f2_51, make_float2(gs[6], gs[7]));
            float e0_476 = approx_exp2(d2_475.x);
            float e1_477 = approx_exp2(d2_475.y);
            float2 _f32x2_mul_a_142 = make_float2(ks[6], ks[7]);
            float2 _f32x2_mul_b_143 = make_float2(e0_476, e1_477);
            float2 v2_478;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_478) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
            v_462[6] = v2_478.x;
            v_462[7] = v2_478.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_462[0]), "f"(v_462[1]), "f"(v_462[2]), "f"(v_462[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_462[4]), "f"(v_462[5]), "f"(v_462[6]), "f"(v_462[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr);
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_479 = 16 + 8 * th;
            int gbase_480 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_479 * 128);
            int kbase_481 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_479 * 128);
            float r_482[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_482[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_483 = r_482[0];
            float r_484[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_484[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a1_485 = r_484[0];
            float gs_486[8];
            float ks_487[8];
            float r_488[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_488[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[0] = r_488[0];
            unsigned int r_489[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_489[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[0] = __uint_as_float(r_489[0] << 16);
            float r_490[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_490[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[1] = r_490[0];
            unsigned int r_491[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_491[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[1] = __uint_as_float(r_491[0] << 16);
            float r_492[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_492[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[2] = r_492[0];
            unsigned int r_493[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_493[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[2] = __uint_as_float(r_493[0] << 16);
            float r_494[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_494[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[3] = r_494[0];
            unsigned int r_495[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_495[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[3] = __uint_as_float(r_495[0] << 16);
            float r_496[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_496[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[4] = r_496[0];
            unsigned int r_497[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_497[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[4] = __uint_as_float(r_497[0] << 16);
            float r_498[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_498[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[5] = r_498[0];
            unsigned int r_499[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_499[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[5] = __uint_as_float(r_499[0] << 16);
            float r_500[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_500[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[6] = r_500[0];
            unsigned int r_501[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_501[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[6] = __uint_as_float(r_501[0] << 16);
            float r_502[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_502[0])) : "r"((gbase_480 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_486[7] = r_502[0];
            unsigned int r_503[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_503[0])) : "r"((kbase_481 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_487[7] = __uint_as_float(r_503[0] << 16);
            float2 _f2_52 = make_float2(a0_483, a0_483);
            float v_504[8];
            float2 d2_505 = sub_f32x2(_f2_52, make_float2(gs_486[0], gs_486[1]));
            float e0_506 = approx_exp2(d2_505.x);
            float e1_507 = approx_exp2(d2_505.y);
            float2 _f32x2_mul_a_144 = make_float2(ks_487[0], ks_487[1]);
            float2 _f32x2_mul_b_145 = make_float2(e0_506, e1_507);
            float2 v2_508;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_508) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
            v_504[0] = v2_508.x;
            v_504[1] = v2_508.y;
            float2 d2_509 = sub_f32x2(_f2_52, make_float2(gs_486[2], gs_486[3]));
            float e0_510 = approx_exp2(d2_509.x);
            float e1_511 = approx_exp2(d2_509.y);
            float2 _f32x2_mul_a_146 = make_float2(ks_487[2], ks_487[3]);
            float2 _f32x2_mul_b_147 = make_float2(e0_510, e1_511);
            float2 v2_512;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_512) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
            v_504[2] = v2_512.x;
            v_504[3] = v2_512.y;
            float2 d2_513 = sub_f32x2(_f2_52, make_float2(gs_486[4], gs_486[5]));
            float e0_514 = approx_exp2(d2_513.x);
            float e1_515 = approx_exp2(d2_513.y);
            float2 _f32x2_mul_a_148 = make_float2(ks_487[4], ks_487[5]);
            float2 _f32x2_mul_b_149 = make_float2(e0_514, e1_515);
            float2 v2_516;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_516) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
            v_504[4] = v2_516.x;
            v_504[5] = v2_516.y;
            float2 d2_517 = sub_f32x2(_f2_52, make_float2(gs_486[6], gs_486[7]));
            float e0_518 = approx_exp2(d2_517.x);
            float e1_519 = approx_exp2(d2_517.y);
            float2 _f32x2_mul_a_150 = make_float2(ks_487[6], ks_487[7]);
            float2 _f32x2_mul_b_151 = make_float2(e0_518, e1_519);
            float2 v2_520;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_520) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
            v_504[6] = v2_520.x;
            v_504[7] = v2_520.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_504[0]), "f"(v_504[1]), "f"(v_504[2]), "f"(v_504[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_504[4]), "f"(v_504[5]), "f"(v_504[6]), "f"(v_504[7]) : "memory");
            float2 _f2_53 = make_float2(a1_485, a1_485);
            float v_521[8];
            float2 d2_522 = sub_f32x2(_f2_53, make_float2(gs_486[0], gs_486[1]));
            float e0_523 = approx_exp2(d2_522.x);
            float e1_524 = approx_exp2(d2_522.y);
            float2 _f32x2_mul_a_152 = make_float2(ks_487[0], ks_487[1]);
            float2 _f32x2_mul_b_153 = make_float2(e0_523, e1_524);
            float2 v2_525;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_525) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
            v_521[0] = v2_525.x;
            v_521[1] = v2_525.y;
            float2 d2_526 = sub_f32x2(_f2_53, make_float2(gs_486[2], gs_486[3]));
            float e0_527 = approx_exp2(d2_526.x);
            float e1_528 = approx_exp2(d2_526.y);
            float2 _f32x2_mul_a_154 = make_float2(ks_487[2], ks_487[3]);
            float2 _f32x2_mul_b_155 = make_float2(e0_527, e1_528);
            float2 v2_529;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_529) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
            v_521[2] = v2_529.x;
            v_521[3] = v2_529.y;
            float2 d2_530 = sub_f32x2(_f2_53, make_float2(gs_486[4], gs_486[5]));
            float e0_531 = approx_exp2(d2_530.x);
            float e1_532 = approx_exp2(d2_530.y);
            float2 _f32x2_mul_a_156 = make_float2(ks_487[4], ks_487[5]);
            float2 _f32x2_mul_b_157 = make_float2(e0_531, e1_532);
            float2 v2_533;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_533) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
            v_521[4] = v2_533.x;
            v_521[5] = v2_533.y;
            float2 d2_534 = sub_f32x2(_f2_53, make_float2(gs_486[6], gs_486[7]));
            float e0_535 = approx_exp2(d2_534.x);
            float e1_536 = approx_exp2(d2_534.y);
            float2 _f32x2_mul_a_158 = make_float2(ks_487[6], ks_487[7]);
            float2 _f32x2_mul_b_159 = make_float2(e0_535, e1_536);
            float2 v2_537;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_537) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
            v_521[6] = v2_537.x;
            v_521[7] = v2_537.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(v_521[0]), "f"(v_521[1]), "f"(v_521[2]), "f"(v_521[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_521[4]), "f"(v_521[5]), "f"(v_521[6]), "f"(v_521[7]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(ring_full_addr + 8);
            mbarrier_wait(ring_empty_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_538 = 32 + 8 * th;
            int gbase_539 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_538 * 128);
            int kbase_540 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_538 * 128);
            float r_541[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_541[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_542 = r_541[0];
            float a1_543 = 0.0f;
            float gs_544[8];
            float ks_545[8];
            float r_546[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_546[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[0] = r_546[0];
            unsigned int r_547[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_547[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[0] = __uint_as_float(r_547[0] << 16);
            float r_548[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_548[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[1] = r_548[0];
            unsigned int r_549[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_549[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[1] = __uint_as_float(r_549[0] << 16);
            float r_550[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_550[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[2] = r_550[0];
            unsigned int r_551[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_551[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[2] = __uint_as_float(r_551[0] << 16);
            float r_552[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_552[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[3] = r_552[0];
            unsigned int r_553[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_553[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[3] = __uint_as_float(r_553[0] << 16);
            float r_554[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_554[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[4] = r_554[0];
            unsigned int r_555[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_555[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[4] = __uint_as_float(r_555[0] << 16);
            float r_556[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_556[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[5] = r_556[0];
            unsigned int r_557[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_557[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[5] = __uint_as_float(r_557[0] << 16);
            float r_558[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_558[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[6] = r_558[0];
            unsigned int r_559[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_559[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[6] = __uint_as_float(r_559[0] << 16);
            float r_560[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_560[0])) : "r"((gbase_539 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_544[7] = r_560[0];
            unsigned int r_561[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_561[0])) : "r"((kbase_540 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_545[7] = __uint_as_float(r_561[0] << 16);
            float2 _f2_54 = make_float2(a0_542, a0_542);
            float v_562[8];
            float2 d2_563 = sub_f32x2(_f2_54, make_float2(gs_544[0], gs_544[1]));
            float e0_564 = approx_exp2(d2_563.x);
            float e1_565 = approx_exp2(d2_563.y);
            float2 _f32x2_mul_a_160 = make_float2(ks_545[0], ks_545[1]);
            float2 _f32x2_mul_b_161 = make_float2(e0_564, e1_565);
            float2 v2_566;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_566) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
            v_562[0] = v2_566.x;
            v_562[1] = v2_566.y;
            float2 d2_567 = sub_f32x2(_f2_54, make_float2(gs_544[2], gs_544[3]));
            float e0_568 = approx_exp2(d2_567.x);
            float e1_569 = approx_exp2(d2_567.y);
            float2 _f32x2_mul_a_162 = make_float2(ks_545[2], ks_545[3]);
            float2 _f32x2_mul_b_163 = make_float2(e0_568, e1_569);
            float2 v2_570;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_570) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
            v_562[2] = v2_570.x;
            v_562[3] = v2_570.y;
            float2 d2_571 = sub_f32x2(_f2_54, make_float2(gs_544[4], gs_544[5]));
            float e0_572 = approx_exp2(d2_571.x);
            float e1_573 = approx_exp2(d2_571.y);
            float2 _f32x2_mul_a_164 = make_float2(ks_545[4], ks_545[5]);
            float2 _f32x2_mul_b_165 = make_float2(e0_572, e1_573);
            float2 v2_574;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_574) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
            v_562[4] = v2_574.x;
            v_562[5] = v2_574.y;
            float2 d2_575 = sub_f32x2(_f2_54, make_float2(gs_544[6], gs_544[7]));
            float e0_576 = approx_exp2(d2_575.x);
            float e1_577 = approx_exp2(d2_575.y);
            float2 _f32x2_mul_a_166 = make_float2(ks_545[6], ks_545[7]);
            float2 _f32x2_mul_b_167 = make_float2(e0_576, e1_577);
            float2 v2_578;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_578) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
            v_562[6] = v2_578.x;
            v_562[7] = v2_578.y;
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
            float r_579[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_579[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6016 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl = r_579[0];
            float r_580[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_580[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc = r_580[0] - 21.640425f;
            float dkt[8];
            float pan[8];
            float2 _f2_55 = make_float2(gc, gc);
            float2 _f2_56 = make_float2(gnl, gnl);
            int t0 = 32 + 8 * th;
            int t1 = 32 + 8 * th + 1;
            float r_581[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_581[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0 = r_581[0];
            float r_582[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_582[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1 = r_582[0];
            unsigned int r_583[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_583[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0 = __uint_as_float(r_583[0] << 16);
            unsigned int r_584[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_584[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1 = __uint_as_float(r_584[0] << 16);
            float2 _f2_57 = make_float2(g_t0, g_t1);
            float2 dc2 = sub_f32x2(_f2_55, _f2_57);
            float e2_0 = approx_exp2(dc2.x);
            float e2_1 = approx_exp2(dc2.y);
            float2 dn2 = sub_f32x2(_f2_56, _f2_57);
            float en_0 = approx_exp2(dn2.x);
            float en_1 = approx_exp2(dn2.y);
            dkt[0] = __fmaf_rn(_tmem_load_1[0], en_0, _tmem_load_0[0] * e2_0);
            dkt[1] = __fmaf_rn(_tmem_load_1[1], en_1, _tmem_load_0[1] * e2_1);
            float2 _f32x2_mul_a_168 = make_float2(k_t0, k_t1);
            float2 _f32x2_mul_b_169 = make_float2(e2_0, e2_1);
            float2 p2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
            pan[0] = p2.x;
            pan[1] = p2.y;
            int t0_585 = 32 + 8 * th + 2;
            int t1_586 = 32 + 8 * th + 2 + 1;
            float r_587[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_587[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_585 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_585 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_588 = r_587[0];
            float r_589[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_589[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_586 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_586 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_590 = r_589[0];
            unsigned int r_591[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_591[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_585 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_585 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_592 = __uint_as_float(r_591[0] << 16);
            unsigned int r_593[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_593[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_586 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_586 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_594 = __uint_as_float(r_593[0] << 16);
            float2 _f2_58 = make_float2(g_t0_588, g_t1_590);
            float2 dc2_595 = sub_f32x2(_f2_55, _f2_58);
            float e2_0_596 = approx_exp2(dc2_595.x);
            float e2_1_597 = approx_exp2(dc2_595.y);
            float2 dn2_598 = sub_f32x2(_f2_56, _f2_58);
            float en_0_599 = approx_exp2(dn2_598.x);
            float en_1_600 = approx_exp2(dn2_598.y);
            dkt[2] = __fmaf_rn(_tmem_load_1[2], en_0_599, _tmem_load_0[2] * e2_0_596);
            dkt[3] = __fmaf_rn(_tmem_load_1[3], en_1_600, _tmem_load_0[3] * e2_1_597);
            float2 _f32x2_mul_a_170 = make_float2(k_t0_592, k_t1_594);
            float2 _f32x2_mul_b_171 = make_float2(e2_0_596, e2_1_597);
            float2 p2_601;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_601) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
            pan[2] = p2_601.x;
            pan[3] = p2_601.y;
            int t0_602 = 32 + 8 * th + 4;
            int t1_603 = 32 + 8 * th + 4 + 1;
            float r_604[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_604[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_602 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_602 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_605 = r_604[0];
            float r_606[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_606[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_603 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_603 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_607 = r_606[0];
            unsigned int r_608[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_608[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_602 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_602 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_609 = __uint_as_float(r_608[0] << 16);
            unsigned int r_610[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_610[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_603 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_603 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_611 = __uint_as_float(r_610[0] << 16);
            float2 _f2_59 = make_float2(g_t0_605, g_t1_607);
            float2 dc2_612 = sub_f32x2(_f2_55, _f2_59);
            float e2_0_613 = approx_exp2(dc2_612.x);
            float e2_1_614 = approx_exp2(dc2_612.y);
            float2 dn2_615 = sub_f32x2(_f2_56, _f2_59);
            float en_0_616 = approx_exp2(dn2_615.x);
            float en_1_617 = approx_exp2(dn2_615.y);
            dkt[4] = __fmaf_rn(_tmem_load_1[4], en_0_616, _tmem_load_0[4] * e2_0_613);
            dkt[5] = __fmaf_rn(_tmem_load_1[5], en_1_617, _tmem_load_0[5] * e2_1_614);
            float2 _f32x2_mul_a_172 = make_float2(k_t0_609, k_t1_611);
            float2 _f32x2_mul_b_173 = make_float2(e2_0_613, e2_1_614);
            float2 p2_618;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_618) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
            pan[4] = p2_618.x;
            pan[5] = p2_618.y;
            int t0_619 = 32 + 8 * th + 6;
            int t1_620 = 32 + 8 * th + 6 + 1;
            float r_621[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_621[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_619 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_619 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_622 = r_621[0];
            float r_623[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_623[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_620 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_620 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_624 = r_623[0];
            unsigned int r_625[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_625[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_619 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_619 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_626 = __uint_as_float(r_625[0] << 16);
            unsigned int r_627[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_627[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_620 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_620 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_628 = __uint_as_float(r_627[0] << 16);
            float2 _f2_60 = make_float2(g_t0_622, g_t1_624);
            float2 dc2_629 = sub_f32x2(_f2_55, _f2_60);
            float e2_0_630 = approx_exp2(dc2_629.x);
            float e2_1_631 = approx_exp2(dc2_629.y);
            float2 dn2_632 = sub_f32x2(_f2_56, _f2_60);
            float en_0_633 = approx_exp2(dn2_632.x);
            float en_1_634 = approx_exp2(dn2_632.y);
            dkt[6] = __fmaf_rn(_tmem_load_1[6], en_0_633, _tmem_load_0[6] * e2_0_630);
            dkt[7] = __fmaf_rn(_tmem_load_1[7], en_1_634, _tmem_load_0[7] * e2_1_631);
            float2 _f32x2_mul_a_174 = make_float2(k_t0_626, k_t1_628);
            float2 _f32x2_mul_b_175 = make_float2(e2_0_630, e2_1_631);
            float2 p2_635;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_635) : "l"(*(const unsigned long long*)&_f32x2_mul_a_174), "l"(*(const unsigned long long*)&_f32x2_mul_b_175));
            pan[6] = p2_635.x;
            pan[7] = p2_635.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(32 + 8 * th), dkt);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan[0]), "f"(pan[1]), "f"(pan[2]), "f"(pan[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan[4]), "f"(pan[5]), "f"(pan[6]), "f"(pan[7]) : "memory");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (48 + 8 * th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_636[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_636[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 8064 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_637 = r_636[0];
            float r_638[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_638[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_639 = r_638[0] - 21.640425f;
            float dkt_640[8];
            float pan_641[8];
            float2 _f2_61 = make_float2(gc_639, gc_639);
            float2 _f2_62 = make_float2(gnl_637, gnl_637);
            int t0_642 = 48 + 8 * th;
            int t1_643 = 48 + 8 * th + 1;
            float r_644[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_644[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_642 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_642 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_645 = r_644[0];
            float r_646[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_646[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_643 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_643 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_647 = r_646[0];
            unsigned int r_648[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_648[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_642 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_642 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_649 = __uint_as_float(r_648[0] << 16);
            unsigned int r_650[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_650[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_643 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_643 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_651 = __uint_as_float(r_650[0] << 16);
            float2 dc2_652 = sub_f32x2(_f2_61, make_float2(g_t0_645, g_t1_647));
            float e2_0_653 = approx_exp2(dc2_652.x);
            float e2_1_654 = approx_exp2(dc2_652.y);
            dkt_640[0] = _tmem_load_2[0] * e2_0_653;
            dkt_640[1] = _tmem_load_2[1] * e2_1_654;
            float2 _f32x2_mul_a_176 = make_float2(k_t0_649, k_t1_651);
            float2 _f32x2_mul_b_177 = make_float2(e2_0_653, e2_1_654);
            float2 p2_655;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_655) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
            pan_641[0] = p2_655.x;
            pan_641[1] = p2_655.y;
            int t0_656 = 48 + 8 * th + 2;
            int t1_657 = 48 + 8 * th + 2 + 1;
            float r_658[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_658[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_656 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_656 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_659 = r_658[0];
            float r_660[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_660[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_657 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_657 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_661 = r_660[0];
            unsigned int r_662[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_662[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_656 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_656 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_663 = __uint_as_float(r_662[0] << 16);
            unsigned int r_664[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_664[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_657 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_657 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_665 = __uint_as_float(r_664[0] << 16);
            float2 dc2_666 = sub_f32x2(_f2_61, make_float2(g_t0_659, g_t1_661));
            float e2_0_667 = approx_exp2(dc2_666.x);
            float e2_1_668 = approx_exp2(dc2_666.y);
            dkt_640[2] = _tmem_load_2[2] * e2_0_667;
            dkt_640[3] = _tmem_load_2[3] * e2_1_668;
            float2 _f32x2_mul_a_178 = make_float2(k_t0_663, k_t1_665);
            float2 _f32x2_mul_b_179 = make_float2(e2_0_667, e2_1_668);
            float2 p2_669;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_669) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
            pan_641[2] = p2_669.x;
            pan_641[3] = p2_669.y;
            int t0_670 = 48 + 8 * th + 4;
            int t1_671 = 48 + 8 * th + 4 + 1;
            float r_672[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_672[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_670 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_670 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_673 = r_672[0];
            float r_674[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_674[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_671 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_671 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_675 = r_674[0];
            unsigned int r_676[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_676[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_670 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_670 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_677 = __uint_as_float(r_676[0] << 16);
            unsigned int r_678[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_678[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_671 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_671 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_679 = __uint_as_float(r_678[0] << 16);
            float2 dc2_680 = sub_f32x2(_f2_61, make_float2(g_t0_673, g_t1_675));
            float e2_0_681 = approx_exp2(dc2_680.x);
            float e2_1_682 = approx_exp2(dc2_680.y);
            dkt_640[4] = _tmem_load_2[4] * e2_0_681;
            dkt_640[5] = _tmem_load_2[5] * e2_1_682;
            float2 _f32x2_mul_a_180 = make_float2(k_t0_677, k_t1_679);
            float2 _f32x2_mul_b_181 = make_float2(e2_0_681, e2_1_682);
            float2 p2_683;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_683) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&_f32x2_mul_b_181));
            pan_641[4] = p2_683.x;
            pan_641[5] = p2_683.y;
            int t0_684 = 48 + 8 * th + 6;
            int t1_685 = 48 + 8 * th + 6 + 1;
            float r_686[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_686[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_684 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_684 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_687 = r_686[0];
            float r_688[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_688[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_685 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_685 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_689 = r_688[0];
            unsigned int r_690[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_690[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_684 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_684 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_691 = __uint_as_float(r_690[0] << 16);
            unsigned int r_692[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_692[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_685 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_685 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_693 = __uint_as_float(r_692[0] << 16);
            float2 dc2_694 = sub_f32x2(_f2_61, make_float2(g_t0_687, g_t1_689));
            float e2_0_695 = approx_exp2(dc2_694.x);
            float e2_1_696 = approx_exp2(dc2_694.y);
            dkt_640[6] = _tmem_load_2[6] * e2_0_695;
            dkt_640[7] = _tmem_load_2[7] * e2_1_696;
            float2 _f32x2_mul_a_182 = make_float2(k_t0_691, k_t1_693);
            float2 _f32x2_mul_b_183 = make_float2(e2_0_695, e2_1_696);
            float2 p2_697;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_697) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&_f32x2_mul_b_183));
            pan_641[6] = p2_697.x;
            pan_641[7] = p2_697.y;
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
            float r_698[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_698[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 1920 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_699 = r_698[0];
            float r_700[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_700[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_701 = r_700[0] - 21.640425f;
            float dkt_702[8];
            float pan_703[8];
            float2 _f2_63 = make_float2(gc_701, gc_701);
            float2 _f2_64 = make_float2(gnl_699, gnl_699);
            int t0_704 = 8 * th;
            int t1_705 = 8 * th + 1;
            float r_706[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_706[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_704 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_704 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_707 = r_706[0];
            float r_708[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_708[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_705 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_705 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_709 = r_708[0];
            unsigned int r_710[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_710[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_704 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_704 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_711 = __uint_as_float(r_710[0] << 16);
            unsigned int r_712[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_712[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_705 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_705 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_713 = __uint_as_float(r_712[0] << 16);
            float2 _f2_65 = make_float2(g_t0_707, g_t1_709);
            float2 dc2_714 = sub_f32x2(_f2_63, _f2_65);
            float e2_0_715 = approx_exp2(dc2_714.x);
            float e2_1_716 = approx_exp2(dc2_714.y);
            float2 dn2_717 = sub_f32x2(_f2_64, _f2_65);
            float en_0_718 = approx_exp2(dn2_717.x);
            float en_1_719 = approx_exp2(dn2_717.y);
            dkt_702[0] = __fmaf_rn(_tmem_load_4[0], en_0_718, _tmem_load_3[0] * e2_0_715);
            dkt_702[1] = __fmaf_rn(_tmem_load_4[1], en_1_719, _tmem_load_3[1] * e2_1_716);
            float2 _f32x2_mul_a_184 = make_float2(k_t0_711, k_t1_713);
            float2 _f32x2_mul_b_185 = make_float2(e2_0_715, e2_1_716);
            float2 p2_720;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_720) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&_f32x2_mul_b_185));
            pan_703[0] = p2_720.x;
            pan_703[1] = p2_720.y;
            int t0_721 = 8 * th + 2;
            int t1_722 = 8 * th + 2 + 1;
            float r_723[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_723[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_721 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_721 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_724 = r_723[0];
            float r_725[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_725[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_722 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_722 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_726 = r_725[0];
            unsigned int r_727[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_727[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_721 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_721 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_728 = __uint_as_float(r_727[0] << 16);
            unsigned int r_729[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_729[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_722 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_722 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_730 = __uint_as_float(r_729[0] << 16);
            float2 _f2_66 = make_float2(g_t0_724, g_t1_726);
            float2 dc2_731 = sub_f32x2(_f2_63, _f2_66);
            float e2_0_732 = approx_exp2(dc2_731.x);
            float e2_1_733 = approx_exp2(dc2_731.y);
            float2 dn2_734 = sub_f32x2(_f2_64, _f2_66);
            float en_0_735 = approx_exp2(dn2_734.x);
            float en_1_736 = approx_exp2(dn2_734.y);
            dkt_702[2] = __fmaf_rn(_tmem_load_4[2], en_0_735, _tmem_load_3[2] * e2_0_732);
            dkt_702[3] = __fmaf_rn(_tmem_load_4[3], en_1_736, _tmem_load_3[3] * e2_1_733);
            float2 _f32x2_mul_a_186 = make_float2(k_t0_728, k_t1_730);
            float2 _f32x2_mul_b_187 = make_float2(e2_0_732, e2_1_733);
            float2 p2_737;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_737) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&_f32x2_mul_b_187));
            pan_703[2] = p2_737.x;
            pan_703[3] = p2_737.y;
            int t0_738 = 8 * th + 4;
            int t1_739 = 8 * th + 4 + 1;
            float r_740[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_740[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_738 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_738 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_741 = r_740[0];
            float r_742[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_742[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_739 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_739 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_743 = r_742[0];
            unsigned int r_744[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_744[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_738 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_738 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_745 = __uint_as_float(r_744[0] << 16);
            unsigned int r_746[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_746[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_739 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_739 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_747 = __uint_as_float(r_746[0] << 16);
            float2 _f2_67 = make_float2(g_t0_741, g_t1_743);
            float2 dc2_748 = sub_f32x2(_f2_63, _f2_67);
            float e2_0_749 = approx_exp2(dc2_748.x);
            float e2_1_750 = approx_exp2(dc2_748.y);
            float2 dn2_751 = sub_f32x2(_f2_64, _f2_67);
            float en_0_752 = approx_exp2(dn2_751.x);
            float en_1_753 = approx_exp2(dn2_751.y);
            dkt_702[4] = __fmaf_rn(_tmem_load_4[4], en_0_752, _tmem_load_3[4] * e2_0_749);
            dkt_702[5] = __fmaf_rn(_tmem_load_4[5], en_1_753, _tmem_load_3[5] * e2_1_750);
            float2 _f32x2_mul_a_188 = make_float2(k_t0_745, k_t1_747);
            float2 _f32x2_mul_b_189 = make_float2(e2_0_749, e2_1_750);
            float2 p2_754;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_754) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
            pan_703[4] = p2_754.x;
            pan_703[5] = p2_754.y;
            int t0_755 = 8 * th + 6;
            int t1_756 = 8 * th + 6 + 1;
            float r_757[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_757[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_755 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_755 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_758 = r_757[0];
            float r_759[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_759[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_756 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_756 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_760 = r_759[0];
            unsigned int r_761[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_761[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_755 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_755 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_762 = __uint_as_float(r_761[0] << 16);
            unsigned int r_763[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_763[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_756 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_756 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_764 = __uint_as_float(r_763[0] << 16);
            float2 _f2_68 = make_float2(g_t0_758, g_t1_760);
            float2 dc2_765 = sub_f32x2(_f2_63, _f2_68);
            float e2_0_766 = approx_exp2(dc2_765.x);
            float e2_1_767 = approx_exp2(dc2_765.y);
            float2 dn2_768 = sub_f32x2(_f2_64, _f2_68);
            float en_0_769 = approx_exp2(dn2_768.x);
            float en_1_770 = approx_exp2(dn2_768.y);
            dkt_702[6] = __fmaf_rn(_tmem_load_4[6], en_0_769, _tmem_load_3[6] * e2_0_766);
            dkt_702[7] = __fmaf_rn(_tmem_load_4[7], en_1_770, _tmem_load_3[7] * e2_1_767);
            float2 _f32x2_mul_a_190 = make_float2(k_t0_762, k_t1_764);
            float2 _f32x2_mul_b_191 = make_float2(e2_0_766, e2_1_767);
            float2 p2_771;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_771) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&_f32x2_mul_b_191));
            pan_703[6] = p2_771.x;
            pan_703[7] = p2_771.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(8 * th), dkt_702);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[0]), "f"(pan_703[1]), "f"(pan_703[2]), "f"(pan_703[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_703[4]), "f"(pan_703[5]), "f"(pan_703[6]), "f"(pan_703[7]) : "memory");
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)row_base + (unsigned int)(64 + (16 + 8 * th)));
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float r_772[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_772[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 3968 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gnl_773 = r_772[0];
            float r_774[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_774[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gc_775 = r_774[0] - 21.640425f;
            float dkt_776[8];
            float pan_777[8];
            float2 _f2_69 = make_float2(gc_775, gc_775);
            float2 _f2_70 = make_float2(gnl_773, gnl_773);
            int t0_778 = 16 + 8 * th;
            int t1_779 = 16 + 8 * th + 1;
            float r_780[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_780[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_778 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_778 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_781 = r_780[0];
            float r_782[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_782[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_779 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_779 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_783 = r_782[0];
            unsigned int r_784[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_784[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_778 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_778 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_785 = __uint_as_float(r_784[0] << 16);
            unsigned int r_786[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_786[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_779 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_779 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_787 = __uint_as_float(r_786[0] << 16);
            float2 _f2_71 = make_float2(g_t0_781, g_t1_783);
            float2 dc2_788 = sub_f32x2(_f2_69, _f2_71);
            float e2_0_789 = approx_exp2(dc2_788.x);
            float e2_1_790 = approx_exp2(dc2_788.y);
            float2 dn2_791 = sub_f32x2(_f2_70, _f2_71);
            float en_0_792 = approx_exp2(dn2_791.x);
            float en_1_793 = approx_exp2(dn2_791.y);
            dkt_776[0] = __fmaf_rn(_tmem_load_6[0], en_0_792, _tmem_load_5[0] * e2_0_789);
            dkt_776[1] = __fmaf_rn(_tmem_load_6[1], en_1_793, _tmem_load_5[1] * e2_1_790);
            float2 _f32x2_mul_a_192 = make_float2(k_t0_785, k_t1_787);
            float2 _f32x2_mul_b_193 = make_float2(e2_0_789, e2_1_790);
            float2 p2_794;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_794) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&_f32x2_mul_b_193));
            pan_777[0] = p2_794.x;
            pan_777[1] = p2_794.y;
            int t0_795 = 16 + 8 * th + 2;
            int t1_796 = 16 + 8 * th + 2 + 1;
            float r_797[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_797[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_795 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_795 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_798 = r_797[0];
            float r_799[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_799[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_796 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_796 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_800 = r_799[0];
            unsigned int r_801[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_801[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_795 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_795 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_802 = __uint_as_float(r_801[0] << 16);
            unsigned int r_803[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_803[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_796 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_796 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_804 = __uint_as_float(r_803[0] << 16);
            float2 _f2_72 = make_float2(g_t0_798, g_t1_800);
            float2 dc2_805 = sub_f32x2(_f2_69, _f2_72);
            float e2_0_806 = approx_exp2(dc2_805.x);
            float e2_1_807 = approx_exp2(dc2_805.y);
            float2 dn2_808 = sub_f32x2(_f2_70, _f2_72);
            float en_0_809 = approx_exp2(dn2_808.x);
            float en_1_810 = approx_exp2(dn2_808.y);
            dkt_776[2] = __fmaf_rn(_tmem_load_6[2], en_0_809, _tmem_load_5[2] * e2_0_806);
            dkt_776[3] = __fmaf_rn(_tmem_load_6[3], en_1_810, _tmem_load_5[3] * e2_1_807);
            float2 _f32x2_mul_a_194 = make_float2(k_t0_802, k_t1_804);
            float2 _f32x2_mul_b_195 = make_float2(e2_0_806, e2_1_807);
            float2 p2_811;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_811) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&_f32x2_mul_b_195));
            pan_777[2] = p2_811.x;
            pan_777[3] = p2_811.y;
            int t0_812 = 16 + 8 * th + 4;
            int t1_813 = 16 + 8 * th + 4 + 1;
            float r_814[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_814[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_812 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_812 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_815 = r_814[0];
            float r_816[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_816[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_813 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_813 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_817 = r_816[0];
            unsigned int r_818[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_818[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_812 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_812 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_819 = __uint_as_float(r_818[0] << 16);
            unsigned int r_820[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_820[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_813 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_813 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_821 = __uint_as_float(r_820[0] << 16);
            float2 _f2_73 = make_float2(g_t0_815, g_t1_817);
            float2 dc2_822 = sub_f32x2(_f2_69, _f2_73);
            float e2_0_823 = approx_exp2(dc2_822.x);
            float e2_1_824 = approx_exp2(dc2_822.y);
            float2 dn2_825 = sub_f32x2(_f2_70, _f2_73);
            float en_0_826 = approx_exp2(dn2_825.x);
            float en_1_827 = approx_exp2(dn2_825.y);
            dkt_776[4] = __fmaf_rn(_tmem_load_6[4], en_0_826, _tmem_load_5[4] * e2_0_823);
            dkt_776[5] = __fmaf_rn(_tmem_load_6[5], en_1_827, _tmem_load_5[5] * e2_1_824);
            float2 _f32x2_mul_a_196 = make_float2(k_t0_819, k_t1_821);
            float2 _f32x2_mul_b_197 = make_float2(e2_0_823, e2_1_824);
            float2 p2_828;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_828) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
            pan_777[4] = p2_828.x;
            pan_777[5] = p2_828.y;
            int t0_829 = 16 + 8 * th + 6;
            int t1_830 = 16 + 8 * th + 6 + 1;
            float r_831[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_831[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t0_829 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t0_829 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t0_832 = r_831[0];
            float r_833[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_833[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + t1_830 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + t1_830 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t1_834 = r_833[0];
            unsigned int r_835[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_835[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t0_829 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t0_829 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t0_836 = __uint_as_float(r_835[0] << 16);
            unsigned int r_837[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_837[0])) : "r"((k_p_addr + (unsigned int)(dh_lane * 8192) + (unsigned int)(dcol / 64 * 8192 + t1_830 * 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + t1_830 * 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t1_838 = __uint_as_float(r_837[0] << 16);
            float2 _f2_74 = make_float2(g_t0_832, g_t1_834);
            float2 dc2_839 = sub_f32x2(_f2_69, _f2_74);
            float e2_0_840 = approx_exp2(dc2_839.x);
            float e2_1_841 = approx_exp2(dc2_839.y);
            float2 dn2_842 = sub_f32x2(_f2_70, _f2_74);
            float en_0_843 = approx_exp2(dn2_842.x);
            float en_1_844 = approx_exp2(dn2_842.y);
            dkt_776[6] = __fmaf_rn(_tmem_load_6[6], en_0_843, _tmem_load_5[6] * e2_0_840);
            dkt_776[7] = __fmaf_rn(_tmem_load_6[7], en_1_844, _tmem_load_5[7] * e2_1_841);
            float2 _f32x2_mul_a_198 = make_float2(k_t0_836, k_t1_838);
            float2 _f32x2_mul_b_199 = make_float2(e2_0_840, e2_1_841);
            float2 p2_845;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p2_845) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
            pan_777[6] = p2_845.x;
            pan_777[7] = p2_845.y;
            tmem_st_x8_f32(taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th), dkt_776);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 ^ ((16 + 8 * th) / 32 * 16384 + d_lane * 128 + (16 + 8 * th) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[0]), "f"(pan_777[1]), "f"(pan_777[2]), "f"(pan_777[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + (unsigned int)((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 ^ ((16 + 8 * th + 4) / 32 * 16384 + d_lane * 128 + (16 + 8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(pan_777[4]), "f"(pan_777[5]), "f"(pan_777[6]), "f"(pan_777[7]) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_wait(ring_empty_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int sbase_846 = 8 * th;
            int gbase_847 = pool_f32_addr + (unsigned int)(qd * 8192 + sbase_846 * 128);
            int kbase_848 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sbase_846 * 128);
            float r_849[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_849[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float a0_850 = r_849[0];
            float a1_851 = 0.0f;
            float gs_852[8];
            float ks_853[8];
            float r_854[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_854[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[0] = r_854[0];
            unsigned int r_855[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_855[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[0] = __uint_as_float(r_855[0] << 16);
            float r_856[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_856[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[1] = r_856[0];
            unsigned int r_857[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_857[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[1] = __uint_as_float(r_857[0] << 16);
            float r_858[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_858[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[2] = r_858[0];
            unsigned int r_859[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_859[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[2] = __uint_as_float(r_859[0] << 16);
            float r_860[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_860[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[3] = r_860[0];
            unsigned int r_861[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_861[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[3] = __uint_as_float(r_861[0] << 16);
            float r_862[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_862[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[4] = r_862[0];
            unsigned int r_863[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_863[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[4] = __uint_as_float(r_863[0] << 16);
            float r_864[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_864[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[5] = r_864[0];
            unsigned int r_865[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_865[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[5] = __uint_as_float(r_865[0] << 16);
            float r_866[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_866[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[6] = r_866[0];
            unsigned int r_867[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_867[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[6] = __uint_as_float(r_867[0] << 16);
            float r_868[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_868[0])) : "r"((gbase_847 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            gs_852[7] = r_868[0];
            unsigned int r_869[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_869[0])) : "r"((kbase_848 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            ks_853[7] = __uint_as_float(r_869[0] << 16);
            float2 _f2_75 = make_float2(a0_850, a0_850);
            float v_870[8];
            float2 d2_871 = sub_f32x2(_f2_75, make_float2(gs_852[0], gs_852[1]));
            float e0_872 = approx_exp2(d2_871.x);
            float e1_873 = approx_exp2(d2_871.y);
            float2 _f32x2_mul_a_200 = make_float2(ks_853[0], ks_853[1]);
            float2 _f32x2_mul_b_201 = make_float2(e0_872, e1_873);
            float2 v2_874;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_874) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
            v_870[0] = v2_874.x;
            v_870[1] = v2_874.y;
            float2 d2_875 = sub_f32x2(_f2_75, make_float2(gs_852[2], gs_852[3]));
            float e0_876 = approx_exp2(d2_875.x);
            float e1_877 = approx_exp2(d2_875.y);
            float2 _f32x2_mul_a_202 = make_float2(ks_853[2], ks_853[3]);
            float2 _f32x2_mul_b_203 = make_float2(e0_876, e1_877);
            float2 v2_878;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_878) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&_f32x2_mul_b_203));
            v_870[2] = v2_878.x;
            v_870[3] = v2_878.y;
            float2 d2_879 = sub_f32x2(_f2_75, make_float2(gs_852[4], gs_852[5]));
            float e0_880 = approx_exp2(d2_879.x);
            float e1_881 = approx_exp2(d2_879.y);
            float2 _f32x2_mul_a_204 = make_float2(ks_853[4], ks_853[5]);
            float2 _f32x2_mul_b_205 = make_float2(e0_880, e1_881);
            float2 v2_882;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_882) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
            v_870[4] = v2_882.x;
            v_870[5] = v2_882.y;
            float2 d2_883 = sub_f32x2(_f2_75, make_float2(gs_852[6], gs_852[7]));
            float e0_884 = approx_exp2(d2_883.x);
            float e1_885 = approx_exp2(d2_883.y);
            float2 _f32x2_mul_a_206 = make_float2(ks_853[6], ks_853[7]);
            float2 _f32x2_mul_b_207 = make_float2(e0_884, e1_885);
            float2 v2_886;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v2_886) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
            v_870[6] = v2_886.x;
            v_870[7] = v2_886.y;
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)(8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 ^ (8 * th / 32 * 16384 + d_lane * 128 + 8 * th % 32 * 4 >> 7 & 7) << 4))), "f"(v_870[0]), "f"(v_870[1]), "f"(v_870[2]), "f"(v_870[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((ring_addr + 16384 + (unsigned int)((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 ^ ((8 * th + 4) / 32 * 16384 + d_lane * 128 + (8 * th + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(v_870[4]), "f"(v_870[5]), "f"(v_870[6]), "f"(v_870[7]) : "memory");
            mbarrier_wait(blk_full_addr, 0);
            mbarrier_wait(blk_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane_1 < 16) {
                int kk_12 = w * 4;
                if (lane_1 > (kk_12 & 15)) {
                    {
                        uint32_t _addr_208 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_12 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_12 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_208), "f"(0.0f) : "memory");
                    }
                }
                int kk_0_10 = w * 4 + 1;
                if (lane_1 > (kk_0_10 & 15)) {
                    {
                        uint32_t _addr_209 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_0_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_0_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_209), "f"(0.0f) : "memory");
                    }
                }
                int kk_1_1 = w * 4 + 2;
                if (lane_1 > (kk_1_1 & 15)) {
                    {
                        uint32_t _addr_210 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_1_1 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_1_1 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_210), "f"(0.0f) : "memory");
                    }
                }
                int kk_2_10 = w * 4 + 3;
                if (lane_1 > (kk_2_10 & 15)) {
                    {
                        uint32_t _addr_211 = static_cast<uint32_t>((dab_addr + (unsigned int)(lane_1 / 32 * 4096 + kk_2_10 * 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 4096 + kk_2_10 * 128 + lane_1 % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_211), "f"(0.0f) : "memory");
                    }
                }
            }
            if (lane_1 < 16) {
                int kk_13 = w * 4;
                if (lane_1 > (kk_13 & 15)) {
                    {
                        uint32_t _addr_212 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_13 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_13 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_212), "f"(0.0f) : "memory");
                    }
                }
                int kk_0_11 = w * 4 + 1;
                if (lane_1 > (kk_0_11 & 15)) {
                    {
                        uint32_t _addr_213 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_0_11 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_0_11 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_213), "f"(0.0f) : "memory");
                    }
                }
                int kk_1_2 = w * 4 + 2;
                if (lane_1 > (kk_1_2 & 15)) {
                    {
                        uint32_t _addr_214 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_1_2 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_1_2 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_214), "f"(0.0f) : "memory");
                    }
                }
                int kk_2_11 = w * 4 + 3;
                if (lane_1 > (kk_2_11 & 15)) {
                    {
                        uint32_t _addr_215 = static_cast<uint32_t>((dab_addr + 8192 + (unsigned int)((16 + lane_1) / 32 * 4096 + kk_2_11 * 128 + (16 + lane_1) % 32 * 4 ^ ((16 + lane_1) / 32 * 4096 + kk_2_11 * 128 + (16 + lane_1) % 32 * 4 >> 7 & 7) << 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_215), "f"(0.0f) : "memory");
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
            float r_887[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_887[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 4096 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn = r_887[0];
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
            int t = 32 + 8 * th + 7;
            float r_890[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_890[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t = r_890[0];
            unsigned int r_891[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_891[0])) : "r"((kb8 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t = __uint_as_float(r_891[0] << 16);
            unsigned int r_892[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_892[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t = __uint_as_float(r_892[0] << 16);
            float q_t_893 = ((t >= clen) ? 0.0f : q_t);
            float dqf = pool_f32[ew + 896 + d_lane];
            float dkf = pool_f32[ew + 1920 + d_lane];
            float dgf = pool_f32[ew + 2944 + d_lane];
            float bt_t = btv[7];
            float e_off = approx_exp2(g_t - gn);
            float e_dg = approx_exp2(g_t - gc_888);
            float dq2 = __fmaf_rn(_tmem_load_7[7], e_off, _tmem_load_9[7] * e_dg);
            float dk2 = __fmaf_rn(_tmem_load_8[7], e_off, _tmem_load_10[7] * e_dg);
            dbp[7] = dk2 * k_t;
            float dk2b = dk2 * bt_t;
            float dq_o = dq2 + dqf;
            float dk_o = dk2b + dkf + _tmem_load_11[7];
            float dg_o = __fmaf_rn(q_t_893, dq2, (dk2b - _tmem_load_11[7]) * k_t) + dgf;
            s_run[0] = s_run[0] + dg_o;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 7)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff = t * qrs;
            if (t < clen) {
                {
                    auto _gst_216 = dq_base + qoff;
                    __builtin_assume(__isGlobal(_gst_216));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_216) + (0)) = (__nv_bfloat16)dq_o;
                }
                {
                    auto _gst_217 = dk_base + qoff;
                    __builtin_assume(__isGlobal(_gst_217));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_217) + (0)) = (__nv_bfloat16)dk_o;
                }
            }
            int t_894 = 32 + 8 * th + 6;
            float r_895[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_895[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_896 = r_895[0];
            unsigned int r_897[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_897[0])) : "r"((kb8 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_898 = __uint_as_float(r_897[0] << 16);
            unsigned int r_899[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_899[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_900 = __uint_as_float(r_899[0] << 16);
            float q_t_901 = ((t_894 >= clen) ? 0.0f : q_t_900);
            float dqf_902 = pool_f32[ew + 768 + d_lane];
            float dkf_903 = pool_f32[ew + 1792 + d_lane];
            float dgf_904 = pool_f32[ew + 2816 + d_lane];
            float bt_t_905 = btv[6];
            float e_off_906 = approx_exp2(g_t_896 - gn);
            float e_dg_907 = approx_exp2(g_t_896 - gc_888);
            float dq2_908 = __fmaf_rn(_tmem_load_7[6], e_off_906, _tmem_load_9[6] * e_dg_907);
            float dk2_909 = __fmaf_rn(_tmem_load_8[6], e_off_906, _tmem_load_10[6] * e_dg_907);
            dbp[6] = dk2_909 * k_t_898;
            float dk2b_910 = dk2_909 * bt_t_905;
            float dq_o_911 = dq2_908 + dqf_902;
            float dk_o_912 = dk2b_910 + dkf_903 + _tmem_load_11[6];
            float dg_o_913 = __fmaf_rn(q_t_901, dq2_908, (dk2b_910 - _tmem_load_11[6]) * k_t_898) + dgf_904;
            s_run[0] = s_run[0] + dg_o_913;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 6)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_914 = t_894 * qrs;
            if (t_894 < clen) {
                {
                    auto _gst_218 = dq_base + qoff_914;
                    __builtin_assume(__isGlobal(_gst_218));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_218) + (0)) = (__nv_bfloat16)dq_o_911;
                }
                {
                    auto _gst_219 = dk_base + qoff_914;
                    __builtin_assume(__isGlobal(_gst_219));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_219) + (0)) = (__nv_bfloat16)dk_o_912;
                }
            }
            int t_915 = 32 + 8 * th + 5;
            float r_916[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_916[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_917 = r_916[0];
            unsigned int r_918[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_918[0])) : "r"((kb8 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_919 = __uint_as_float(r_918[0] << 16);
            unsigned int r_920[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_920[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_921 = __uint_as_float(r_920[0] << 16);
            float q_t_922 = ((t_915 >= clen) ? 0.0f : q_t_921);
            float dqf_923 = pool_f32[ew + 640 + d_lane];
            float dkf_924 = pool_f32[ew + 1664 + d_lane];
            float dgf_925 = pool_f32[ew + 2688 + d_lane];
            float bt_t_926 = btv[5];
            float e_off_927 = approx_exp2(g_t_917 - gn);
            float e_dg_928 = approx_exp2(g_t_917 - gc_888);
            float dq2_929 = __fmaf_rn(_tmem_load_7[5], e_off_927, _tmem_load_9[5] * e_dg_928);
            float dk2_930 = __fmaf_rn(_tmem_load_8[5], e_off_927, _tmem_load_10[5] * e_dg_928);
            dbp[5] = dk2_930 * k_t_919;
            float dk2b_931 = dk2_930 * bt_t_926;
            float dq_o_932 = dq2_929 + dqf_923;
            float dk_o_933 = dk2b_931 + dkf_924 + _tmem_load_11[5];
            float dg_o_934 = __fmaf_rn(q_t_922, dq2_929, (dk2b_931 - _tmem_load_11[5]) * k_t_919) + dgf_925;
            s_run[0] = s_run[0] + dg_o_934;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 5)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_935 = t_915 * qrs;
            if (t_915 < clen) {
                {
                    auto _gst_220 = dq_base + qoff_935;
                    __builtin_assume(__isGlobal(_gst_220));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_220) + (0)) = (__nv_bfloat16)dq_o_932;
                }
                {
                    auto _gst_221 = dk_base + qoff_935;
                    __builtin_assume(__isGlobal(_gst_221));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_221) + (0)) = (__nv_bfloat16)dk_o_933;
                }
            }
            int t_936 = 32 + 8 * th + 4;
            float r_937[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_937[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_938 = r_937[0];
            unsigned int r_939[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_939[0])) : "r"((kb8 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_940 = __uint_as_float(r_939[0] << 16);
            unsigned int r_941[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_941[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_942 = __uint_as_float(r_941[0] << 16);
            float q_t_943 = ((t_936 >= clen) ? 0.0f : q_t_942);
            float dqf_944 = pool_f32[ew + 512 + d_lane];
            float dkf_945 = pool_f32[ew + 1536 + d_lane];
            float dgf_946 = pool_f32[ew + 2560 + d_lane];
            float bt_t_947 = btv[4];
            float e_off_948 = approx_exp2(g_t_938 - gn);
            float e_dg_949 = approx_exp2(g_t_938 - gc_888);
            float dq2_950 = __fmaf_rn(_tmem_load_7[4], e_off_948, _tmem_load_9[4] * e_dg_949);
            float dk2_951 = __fmaf_rn(_tmem_load_8[4], e_off_948, _tmem_load_10[4] * e_dg_949);
            dbp[4] = dk2_951 * k_t_940;
            float dk2b_952 = dk2_951 * bt_t_947;
            float dq_o_953 = dq2_950 + dqf_944;
            float dk_o_954 = dk2b_952 + dkf_945 + _tmem_load_11[4];
            float dg_o_955 = __fmaf_rn(q_t_943, dq2_950, (dk2b_952 - _tmem_load_11[4]) * k_t_940) + dgf_946;
            s_run[0] = s_run[0] + dg_o_955;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 4)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_956 = t_936 * qrs;
            if (t_936 < clen) {
                {
                    auto _gst_222 = dq_base + qoff_956;
                    __builtin_assume(__isGlobal(_gst_222));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_222) + (0)) = (__nv_bfloat16)dq_o_953;
                }
                {
                    auto _gst_223 = dk_base + qoff_956;
                    __builtin_assume(__isGlobal(_gst_223));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_223) + (0)) = (__nv_bfloat16)dk_o_954;
                }
            }
            int t_957 = 32 + 8 * th + 3;
            float r_958[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_958[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_959 = r_958[0];
            unsigned int r_960[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_960[0])) : "r"((kb8 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_961 = __uint_as_float(r_960[0] << 16);
            unsigned int r_962[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_962[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_963 = __uint_as_float(r_962[0] << 16);
            float q_t_964 = ((t_957 >= clen) ? 0.0f : q_t_963);
            float dqf_965 = pool_f32[ew + 384 + d_lane];
            float dkf_966 = pool_f32[ew + 1408 + d_lane];
            float dgf_967 = pool_f32[ew + 2432 + d_lane];
            float bt_t_968 = btv[3];
            float e_off_969 = approx_exp2(g_t_959 - gn);
            float e_dg_970 = approx_exp2(g_t_959 - gc_888);
            float dq2_971 = __fmaf_rn(_tmem_load_7[3], e_off_969, _tmem_load_9[3] * e_dg_970);
            float dk2_972 = __fmaf_rn(_tmem_load_8[3], e_off_969, _tmem_load_10[3] * e_dg_970);
            dbp[3] = dk2_972 * k_t_961;
            float dk2b_973 = dk2_972 * bt_t_968;
            float dq_o_974 = dq2_971 + dqf_965;
            float dk_o_975 = dk2b_973 + dkf_966 + _tmem_load_11[3];
            float dg_o_976 = __fmaf_rn(q_t_964, dq2_971, (dk2b_973 - _tmem_load_11[3]) * k_t_961) + dgf_967;
            s_run[0] = s_run[0] + dg_o_976;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 3)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_977 = t_957 * qrs;
            if (t_957 < clen) {
                {
                    auto _gst_224 = dq_base + qoff_977;
                    __builtin_assume(__isGlobal(_gst_224));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_224) + (0)) = (__nv_bfloat16)dq_o_974;
                }
                {
                    auto _gst_225 = dk_base + qoff_977;
                    __builtin_assume(__isGlobal(_gst_225));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_225) + (0)) = (__nv_bfloat16)dk_o_975;
                }
            }
            int t_978 = 32 + 8 * th + 2;
            float r_979[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_979[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_980 = r_979[0];
            unsigned int r_981[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_981[0])) : "r"((kb8 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_982 = __uint_as_float(r_981[0] << 16);
            unsigned int r_983[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_983[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_984 = __uint_as_float(r_983[0] << 16);
            float q_t_985 = ((t_978 >= clen) ? 0.0f : q_t_984);
            float dqf_986 = pool_f32[ew + 256 + d_lane];
            float dkf_987 = pool_f32[ew + 1280 + d_lane];
            float dgf_988 = pool_f32[ew + 2304 + d_lane];
            float bt_t_989 = btv[2];
            float e_off_990 = approx_exp2(g_t_980 - gn);
            float e_dg_991 = approx_exp2(g_t_980 - gc_888);
            float dq2_992 = __fmaf_rn(_tmem_load_7[2], e_off_990, _tmem_load_9[2] * e_dg_991);
            float dk2_993 = __fmaf_rn(_tmem_load_8[2], e_off_990, _tmem_load_10[2] * e_dg_991);
            dbp[2] = dk2_993 * k_t_982;
            float dk2b_994 = dk2_993 * bt_t_989;
            float dq_o_995 = dq2_992 + dqf_986;
            float dk_o_996 = dk2b_994 + dkf_987 + _tmem_load_11[2];
            float dg_o_997 = __fmaf_rn(q_t_985, dq2_992, (dk2b_994 - _tmem_load_11[2]) * k_t_982) + dgf_988;
            s_run[0] = s_run[0] + dg_o_997;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 2)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_998 = t_978 * qrs;
            if (t_978 < clen) {
                {
                    auto _gst_226 = dq_base + qoff_998;
                    __builtin_assume(__isGlobal(_gst_226));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_226) + (0)) = (__nv_bfloat16)dq_o_995;
                }
                {
                    auto _gst_227 = dk_base + qoff_998;
                    __builtin_assume(__isGlobal(_gst_227));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_227) + (0)) = (__nv_bfloat16)dk_o_996;
                }
            }
            int t_999 = 32 + 8 * th + 1;
            float r_1000[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1000[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1001 = r_1000[0];
            unsigned int r_1002[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1002[0])) : "r"((kb8 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1003 = __uint_as_float(r_1002[0] << 16);
            unsigned int r_1004[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1004[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1005 = __uint_as_float(r_1004[0] << 16);
            float q_t_1006 = ((t_999 >= clen) ? 0.0f : q_t_1005);
            float dqf_1007 = pool_f32[ew + 128 + d_lane];
            float dkf_1008 = pool_f32[ew + 1152 + d_lane];
            float dgf_1009 = pool_f32[ew + 2176 + d_lane];
            float bt_t_1010 = btv[1];
            float e_off_1011 = approx_exp2(g_t_1001 - gn);
            float e_dg_1012 = approx_exp2(g_t_1001 - gc_888);
            float dq2_1013 = __fmaf_rn(_tmem_load_7[1], e_off_1011, _tmem_load_9[1] * e_dg_1012);
            float dk2_1014 = __fmaf_rn(_tmem_load_8[1], e_off_1011, _tmem_load_10[1] * e_dg_1012);
            dbp[1] = dk2_1014 * k_t_1003;
            float dk2b_1015 = dk2_1014 * bt_t_1010;
            float dq_o_1016 = dq2_1013 + dqf_1007;
            float dk_o_1017 = dk2b_1015 + dkf_1008 + _tmem_load_11[1];
            float dg_o_1018 = __fmaf_rn(q_t_1006, dq2_1013, (dk2b_1015 - _tmem_load_11[1]) * k_t_1003) + dgf_1009;
            s_run[0] = s_run[0] + dg_o_1018;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th + 1)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1019 = t_999 * qrs;
            if (t_999 < clen) {
                {
                    auto _gst_228 = dq_base + qoff_1019;
                    __builtin_assume(__isGlobal(_gst_228));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_228) + (0)) = (__nv_bfloat16)dq_o_1016;
                }
                {
                    auto _gst_229 = dk_base + qoff_1019;
                    __builtin_assume(__isGlobal(_gst_229));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_229) + (0)) = (__nv_bfloat16)dk_o_1017;
                }
            }
            int t_1020 = 32 + 8 * th;
            float r_1021[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1021[0])) : "r"((gb8 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1022 = r_1021[0];
            unsigned int r_1023[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1023[0])) : "r"((kb8 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1024 = __uint_as_float(r_1023[0] << 16);
            unsigned int r_1025[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1025[0])) : "r"((qbase_889 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1026 = __uint_as_float(r_1025[0] << 16);
            float q_t_1027 = ((t_1020 >= clen) ? 0.0f : q_t_1026);
            float dqf_1028 = pool_f32[ew + d_lane];
            float dkf_1029 = pool_f32[ew + 1024 + d_lane];
            float dgf_1030 = pool_f32[ew + 2048 + d_lane];
            float bt_t_1031 = btv[0];
            float e_off_1032 = approx_exp2(g_t_1022 - gn);
            float e_dg_1033 = approx_exp2(g_t_1022 - gc_888);
            float dq2_1034 = __fmaf_rn(_tmem_load_7[0], e_off_1032, _tmem_load_9[0] * e_dg_1033);
            float dk2_1035 = __fmaf_rn(_tmem_load_8[0], e_off_1032, _tmem_load_10[0] * e_dg_1033);
            dbp[0] = dk2_1035 * k_t_1024;
            float dk2b_1036 = dk2_1035 * bt_t_1031;
            float dq_o_1037 = dq2_1034 + dqf_1028;
            float dk_o_1038 = dk2b_1036 + dkf_1029 + _tmem_load_11[0];
            float dg_o_1039 = __fmaf_rn(q_t_1027, dq2_1034, (dk2b_1036 - _tmem_load_11[0]) * k_t_1024) + dgf_1030;
            s_run[0] = s_run[0] + dg_o_1039;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th)), "f"(s_run[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1040 = t_1020 * qrs;
            if (t_1020 < clen) {
                {
                    auto _gst_230 = dq_base + qoff_1040;
                    __builtin_assume(__isGlobal(_gst_230));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_230) + (0)) = (__nv_bfloat16)dq_o_1037;
                }
                {
                    auto _gst_231 = dk_base + qoff_1040;
                    __builtin_assume(__isGlobal(_gst_231));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_231) + (0)) = (__nv_bfloat16)dk_o_1038;
                }
            }
            float tot = s_run[0];
            int b4 = lane_1 >> 4 & 1;
            int b3 = lane_1 >> 3 & 1;
            int b2 = lane_1 >> 2 & 1;
            float v4[4];
            float send = ((b4 == 0) ? dbp[4] : dbp[0]);
            float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            v4[0] = ((b4 == 0) ? dbp[0] : dbp[4]) + _shfl_xor_sync_232;
            float send_1041 = ((b4 == 0) ? dbp[5] : dbp[1]);
            float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, send_1041, 16);
            v4[1] = ((b4 == 0) ? dbp[1] : dbp[5]) + _shfl_xor_sync_233;
            float send_1042 = ((b4 == 0) ? dbp[6] : dbp[2]);
            float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, send_1042, 16);
            v4[2] = ((b4 == 0) ? dbp[2] : dbp[6]) + _shfl_xor_sync_234;
            float send_1043 = ((b4 == 0) ? dbp[7] : dbp[3]);
            float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, send_1043, 16);
            v4[3] = ((b4 == 0) ? dbp[3] : dbp[7]) + _shfl_xor_sync_235;
            float v2_1044[2];
            float send_1045 = ((b3 == 0) ? v4[2] : v4[0]);
            float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, send_1045, 8);
            v2_1044[0] = ((b3 == 0) ? v4[0] : v4[2]) + _shfl_xor_sync_236;
            float send_1046 = ((b3 == 0) ? v4[3] : v4[1]);
            float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, send_1046, 8);
            v2_1044[1] = ((b3 == 0) ? v4[1] : v4[3]) + _shfl_xor_sync_237;
            float send_1047 = ((b2 == 0) ? v2_1044[1] : v2_1044[0]);
            float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, send_1047, 4);
            float v1 = ((b2 == 0) ? v2_1044[0] : v2_1044[1]) + _shfl_xor_sync_238;
            float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
            v1 = v1 + _shfl_xor_sync_239;
            float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
            v1 = v1 + _shfl_xor_sync_240;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1a_t2_drained_addr);
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            mbarrier_wait(epi_full_addr + (th) * 8, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1048[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1048[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 6144 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1049 = r_1048[0];
            float gc_1050 = gn_1049 - 21.640425f;
            int eb_1051 = 98304 + th * -32768;
            int ew_1052 = eb_1051 >> 2;
            int qbase_1053 = pool_f32_addr + (unsigned int)(eb_1051 + 12288 + dh_lane * 1024);
            int sb8_1054 = 48 + 8 * th;
            int gb8_1055 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1054 * 128);
            int kb8_1056 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1054 * 128);
            float dbp_1057[8];
            float _tmem_load_12[8];
            tmem_ld_x8(&_tmem_load_12[0], taddr + (unsigned int)row_base + (unsigned int)(160 + 8 * th));
            float _tmem_load_13[8];
            tmem_ld_x8(&_tmem_load_13[0], taddr + (unsigned int)row_base + (unsigned int)(176 + 8 * th));
            float _tmem_load_14[8];
            tmem_ld_x8(&_tmem_load_14[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th));
            float _tmem_load_15[8];
            tmem_ld_x8(&_tmem_load_15[0], taddr + (unsigned int)row_base + (unsigned int)(240 + 8 * th));
            float _tmem_load_16[8];
            tmem_ld_x8(&_tmem_load_16[0], taddr + (unsigned int)row_base + (unsigned int)(48 + 8 * th));
            float btv_1058[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (48 + 8 * th))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[4])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1058[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (48 + 8 * th) + 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_run_1059[1];
            s_run_1059[0] = 0.0f;
            int t_1060 = 48 + 8 * th + 7;
            float r_1061[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1061[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1062 = r_1061[0];
            unsigned int r_1063[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1063[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1064 = __uint_as_float(r_1063[0] << 16);
            unsigned int r_1065[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1065[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1066 = __uint_as_float(r_1065[0] << 16);
            float q_t_1067 = ((t_1060 >= clen) ? 0.0f : q_t_1066);
            float dqf_1068 = pool_f32[ew_1052 + 896 + d_lane];
            float dkf_1069 = pool_f32[ew_1052 + 1920 + d_lane];
            float dgf_1070 = pool_f32[ew_1052 + 2944 + d_lane];
            float bt_t_1071 = btv_1058[7];
            float e_off_1072 = approx_exp2(g_t_1062 - gn_1049);
            float e_dg_1073 = approx_exp2(g_t_1062 - gc_1050);
            float dq2_1074 = __fmaf_rn(_tmem_load_12[7], e_off_1072, _tmem_load_14[7] * e_dg_1073);
            float dk2_1075 = __fmaf_rn(_tmem_load_13[7], e_off_1072, _tmem_load_15[7] * e_dg_1073);
            dbp_1057[7] = dk2_1075 * k_t_1064;
            float dk2b_1076 = dk2_1075 * bt_t_1071;
            float dq_o_1077 = dq2_1074 + dqf_1068;
            float dk_o_1078 = dk2b_1076 + dkf_1069 + _tmem_load_16[7];
            float dg_o_1079 = __fmaf_rn(q_t_1067, dq2_1074, (dk2b_1076 - _tmem_load_16[7]) * k_t_1064) + dgf_1070;
            s_run_1059[0] = s_run_1059[0] + dg_o_1079;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 7)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1080 = t_1060 * qrs;
            if (t_1060 < clen) {
                {
                    auto _gst_241 = dq_base + qoff_1080;
                    __builtin_assume(__isGlobal(_gst_241));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_241) + (0)) = (__nv_bfloat16)dq_o_1077;
                }
                {
                    auto _gst_242 = dk_base + qoff_1080;
                    __builtin_assume(__isGlobal(_gst_242));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_242) + (0)) = (__nv_bfloat16)dk_o_1078;
                }
            }
            int t_1081 = 48 + 8 * th + 6;
            float r_1082[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1082[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1083 = r_1082[0];
            unsigned int r_1084[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1084[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1085 = __uint_as_float(r_1084[0] << 16);
            unsigned int r_1086[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1086[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1087 = __uint_as_float(r_1086[0] << 16);
            float q_t_1088 = ((t_1081 >= clen) ? 0.0f : q_t_1087);
            float dqf_1089 = pool_f32[ew_1052 + 768 + d_lane];
            float dkf_1090 = pool_f32[ew_1052 + 1792 + d_lane];
            float dgf_1091 = pool_f32[ew_1052 + 2816 + d_lane];
            float bt_t_1092 = btv_1058[6];
            float e_off_1093 = approx_exp2(g_t_1083 - gn_1049);
            float e_dg_1094 = approx_exp2(g_t_1083 - gc_1050);
            float dq2_1095 = __fmaf_rn(_tmem_load_12[6], e_off_1093, _tmem_load_14[6] * e_dg_1094);
            float dk2_1096 = __fmaf_rn(_tmem_load_13[6], e_off_1093, _tmem_load_15[6] * e_dg_1094);
            dbp_1057[6] = dk2_1096 * k_t_1085;
            float dk2b_1097 = dk2_1096 * bt_t_1092;
            float dq_o_1098 = dq2_1095 + dqf_1089;
            float dk_o_1099 = dk2b_1097 + dkf_1090 + _tmem_load_16[6];
            float dg_o_1100 = __fmaf_rn(q_t_1088, dq2_1095, (dk2b_1097 - _tmem_load_16[6]) * k_t_1085) + dgf_1091;
            s_run_1059[0] = s_run_1059[0] + dg_o_1100;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 6)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1101 = t_1081 * qrs;
            if (t_1081 < clen) {
                {
                    auto _gst_243 = dq_base + qoff_1101;
                    __builtin_assume(__isGlobal(_gst_243));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_243) + (0)) = (__nv_bfloat16)dq_o_1098;
                }
                {
                    auto _gst_244 = dk_base + qoff_1101;
                    __builtin_assume(__isGlobal(_gst_244));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_244) + (0)) = (__nv_bfloat16)dk_o_1099;
                }
            }
            int t_1102 = 48 + 8 * th + 5;
            float r_1103[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1103[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1104 = r_1103[0];
            unsigned int r_1105[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1105[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1106 = __uint_as_float(r_1105[0] << 16);
            unsigned int r_1107[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1107[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1108 = __uint_as_float(r_1107[0] << 16);
            float q_t_1109 = ((t_1102 >= clen) ? 0.0f : q_t_1108);
            float dqf_1110 = pool_f32[ew_1052 + 640 + d_lane];
            float dkf_1111 = pool_f32[ew_1052 + 1664 + d_lane];
            float dgf_1112 = pool_f32[ew_1052 + 2688 + d_lane];
            float bt_t_1113 = btv_1058[5];
            float e_off_1114 = approx_exp2(g_t_1104 - gn_1049);
            float e_dg_1115 = approx_exp2(g_t_1104 - gc_1050);
            float dq2_1116 = __fmaf_rn(_tmem_load_12[5], e_off_1114, _tmem_load_14[5] * e_dg_1115);
            float dk2_1117 = __fmaf_rn(_tmem_load_13[5], e_off_1114, _tmem_load_15[5] * e_dg_1115);
            dbp_1057[5] = dk2_1117 * k_t_1106;
            float dk2b_1118 = dk2_1117 * bt_t_1113;
            float dq_o_1119 = dq2_1116 + dqf_1110;
            float dk_o_1120 = dk2b_1118 + dkf_1111 + _tmem_load_16[5];
            float dg_o_1121 = __fmaf_rn(q_t_1109, dq2_1116, (dk2b_1118 - _tmem_load_16[5]) * k_t_1106) + dgf_1112;
            s_run_1059[0] = s_run_1059[0] + dg_o_1121;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 5)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1122 = t_1102 * qrs;
            if (t_1102 < clen) {
                {
                    auto _gst_245 = dq_base + qoff_1122;
                    __builtin_assume(__isGlobal(_gst_245));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_245) + (0)) = (__nv_bfloat16)dq_o_1119;
                }
                {
                    auto _gst_246 = dk_base + qoff_1122;
                    __builtin_assume(__isGlobal(_gst_246));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_246) + (0)) = (__nv_bfloat16)dk_o_1120;
                }
            }
            int t_1123 = 48 + 8 * th + 4;
            float r_1124[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1124[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1125 = r_1124[0];
            unsigned int r_1126[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1126[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1127 = __uint_as_float(r_1126[0] << 16);
            unsigned int r_1128[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1128[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1129 = __uint_as_float(r_1128[0] << 16);
            float q_t_1130 = ((t_1123 >= clen) ? 0.0f : q_t_1129);
            float dqf_1131 = pool_f32[ew_1052 + 512 + d_lane];
            float dkf_1132 = pool_f32[ew_1052 + 1536 + d_lane];
            float dgf_1133 = pool_f32[ew_1052 + 2560 + d_lane];
            float bt_t_1134 = btv_1058[4];
            float e_off_1135 = approx_exp2(g_t_1125 - gn_1049);
            float e_dg_1136 = approx_exp2(g_t_1125 - gc_1050);
            float dq2_1137 = __fmaf_rn(_tmem_load_12[4], e_off_1135, _tmem_load_14[4] * e_dg_1136);
            float dk2_1138 = __fmaf_rn(_tmem_load_13[4], e_off_1135, _tmem_load_15[4] * e_dg_1136);
            dbp_1057[4] = dk2_1138 * k_t_1127;
            float dk2b_1139 = dk2_1138 * bt_t_1134;
            float dq_o_1140 = dq2_1137 + dqf_1131;
            float dk_o_1141 = dk2b_1139 + dkf_1132 + _tmem_load_16[4];
            float dg_o_1142 = __fmaf_rn(q_t_1130, dq2_1137, (dk2b_1139 - _tmem_load_16[4]) * k_t_1127) + dgf_1133;
            s_run_1059[0] = s_run_1059[0] + dg_o_1142;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 4)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1143 = t_1123 * qrs;
            if (t_1123 < clen) {
                {
                    auto _gst_247 = dq_base + qoff_1143;
                    __builtin_assume(__isGlobal(_gst_247));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_247) + (0)) = (__nv_bfloat16)dq_o_1140;
                }
                {
                    auto _gst_248 = dk_base + qoff_1143;
                    __builtin_assume(__isGlobal(_gst_248));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_248) + (0)) = (__nv_bfloat16)dk_o_1141;
                }
            }
            int t_1144 = 48 + 8 * th + 3;
            float r_1145[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1145[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1146 = r_1145[0];
            unsigned int r_1147[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1147[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1148 = __uint_as_float(r_1147[0] << 16);
            unsigned int r_1149[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1149[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1150 = __uint_as_float(r_1149[0] << 16);
            float q_t_1151 = ((t_1144 >= clen) ? 0.0f : q_t_1150);
            float dqf_1152 = pool_f32[ew_1052 + 384 + d_lane];
            float dkf_1153 = pool_f32[ew_1052 + 1408 + d_lane];
            float dgf_1154 = pool_f32[ew_1052 + 2432 + d_lane];
            float bt_t_1155 = btv_1058[3];
            float e_off_1156 = approx_exp2(g_t_1146 - gn_1049);
            float e_dg_1157 = approx_exp2(g_t_1146 - gc_1050);
            float dq2_1158 = __fmaf_rn(_tmem_load_12[3], e_off_1156, _tmem_load_14[3] * e_dg_1157);
            float dk2_1159 = __fmaf_rn(_tmem_load_13[3], e_off_1156, _tmem_load_15[3] * e_dg_1157);
            dbp_1057[3] = dk2_1159 * k_t_1148;
            float dk2b_1160 = dk2_1159 * bt_t_1155;
            float dq_o_1161 = dq2_1158 + dqf_1152;
            float dk_o_1162 = dk2b_1160 + dkf_1153 + _tmem_load_16[3];
            float dg_o_1163 = __fmaf_rn(q_t_1151, dq2_1158, (dk2b_1160 - _tmem_load_16[3]) * k_t_1148) + dgf_1154;
            s_run_1059[0] = s_run_1059[0] + dg_o_1163;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 3)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1164 = t_1144 * qrs;
            if (t_1144 < clen) {
                {
                    auto _gst_249 = dq_base + qoff_1164;
                    __builtin_assume(__isGlobal(_gst_249));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_249) + (0)) = (__nv_bfloat16)dq_o_1161;
                }
                {
                    auto _gst_250 = dk_base + qoff_1164;
                    __builtin_assume(__isGlobal(_gst_250));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_250) + (0)) = (__nv_bfloat16)dk_o_1162;
                }
            }
            int t_1165 = 48 + 8 * th + 2;
            float r_1166[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1166[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1167 = r_1166[0];
            unsigned int r_1168[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1168[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1169 = __uint_as_float(r_1168[0] << 16);
            unsigned int r_1170[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1170[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1171 = __uint_as_float(r_1170[0] << 16);
            float q_t_1172 = ((t_1165 >= clen) ? 0.0f : q_t_1171);
            float dqf_1173 = pool_f32[ew_1052 + 256 + d_lane];
            float dkf_1174 = pool_f32[ew_1052 + 1280 + d_lane];
            float dgf_1175 = pool_f32[ew_1052 + 2304 + d_lane];
            float bt_t_1176 = btv_1058[2];
            float e_off_1177 = approx_exp2(g_t_1167 - gn_1049);
            float e_dg_1178 = approx_exp2(g_t_1167 - gc_1050);
            float dq2_1179 = __fmaf_rn(_tmem_load_12[2], e_off_1177, _tmem_load_14[2] * e_dg_1178);
            float dk2_1180 = __fmaf_rn(_tmem_load_13[2], e_off_1177, _tmem_load_15[2] * e_dg_1178);
            dbp_1057[2] = dk2_1180 * k_t_1169;
            float dk2b_1181 = dk2_1180 * bt_t_1176;
            float dq_o_1182 = dq2_1179 + dqf_1173;
            float dk_o_1183 = dk2b_1181 + dkf_1174 + _tmem_load_16[2];
            float dg_o_1184 = __fmaf_rn(q_t_1172, dq2_1179, (dk2b_1181 - _tmem_load_16[2]) * k_t_1169) + dgf_1175;
            s_run_1059[0] = s_run_1059[0] + dg_o_1184;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 2)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1185 = t_1165 * qrs;
            if (t_1165 < clen) {
                {
                    auto _gst_251 = dq_base + qoff_1185;
                    __builtin_assume(__isGlobal(_gst_251));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_251) + (0)) = (__nv_bfloat16)dq_o_1182;
                }
                {
                    auto _gst_252 = dk_base + qoff_1185;
                    __builtin_assume(__isGlobal(_gst_252));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_252) + (0)) = (__nv_bfloat16)dk_o_1183;
                }
            }
            int t_1186 = 48 + 8 * th + 1;
            float r_1187[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1187[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1188 = r_1187[0];
            unsigned int r_1189[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1189[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1190 = __uint_as_float(r_1189[0] << 16);
            unsigned int r_1191[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1191[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1192 = __uint_as_float(r_1191[0] << 16);
            float q_t_1193 = ((t_1186 >= clen) ? 0.0f : q_t_1192);
            float dqf_1194 = pool_f32[ew_1052 + 128 + d_lane];
            float dkf_1195 = pool_f32[ew_1052 + 1152 + d_lane];
            float dgf_1196 = pool_f32[ew_1052 + 2176 + d_lane];
            float bt_t_1197 = btv_1058[1];
            float e_off_1198 = approx_exp2(g_t_1188 - gn_1049);
            float e_dg_1199 = approx_exp2(g_t_1188 - gc_1050);
            float dq2_1200 = __fmaf_rn(_tmem_load_12[1], e_off_1198, _tmem_load_14[1] * e_dg_1199);
            float dk2_1201 = __fmaf_rn(_tmem_load_13[1], e_off_1198, _tmem_load_15[1] * e_dg_1199);
            dbp_1057[1] = dk2_1201 * k_t_1190;
            float dk2b_1202 = dk2_1201 * bt_t_1197;
            float dq_o_1203 = dq2_1200 + dqf_1194;
            float dk_o_1204 = dk2b_1202 + dkf_1195 + _tmem_load_16[1];
            float dg_o_1205 = __fmaf_rn(q_t_1193, dq2_1200, (dk2b_1202 - _tmem_load_16[1]) * k_t_1190) + dgf_1196;
            s_run_1059[0] = s_run_1059[0] + dg_o_1205;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th + 1)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1206 = t_1186 * qrs;
            if (t_1186 < clen) {
                {
                    auto _gst_253 = dq_base + qoff_1206;
                    __builtin_assume(__isGlobal(_gst_253));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_253) + (0)) = (__nv_bfloat16)dq_o_1203;
                }
                {
                    auto _gst_254 = dk_base + qoff_1206;
                    __builtin_assume(__isGlobal(_gst_254));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_254) + (0)) = (__nv_bfloat16)dk_o_1204;
                }
            }
            int t_1207 = 48 + 8 * th;
            float r_1208[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1208[0])) : "r"((gb8_1055 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1209 = r_1208[0];
            unsigned int r_1210[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1210[0])) : "r"((kb8_1056 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1211 = __uint_as_float(r_1210[0] << 16);
            unsigned int r_1212[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1212[0])) : "r"((qbase_1053 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1213 = __uint_as_float(r_1212[0] << 16);
            float q_t_1214 = ((t_1207 >= clen) ? 0.0f : q_t_1213);
            float dqf_1215 = pool_f32[ew_1052 + d_lane];
            float dkf_1216 = pool_f32[ew_1052 + 1024 + d_lane];
            float dgf_1217 = pool_f32[ew_1052 + 2048 + d_lane];
            float bt_t_1218 = btv_1058[0];
            float e_off_1219 = approx_exp2(g_t_1209 - gn_1049);
            float e_dg_1220 = approx_exp2(g_t_1209 - gc_1050);
            float dq2_1221 = __fmaf_rn(_tmem_load_12[0], e_off_1219, _tmem_load_14[0] * e_dg_1220);
            float dk2_1222 = __fmaf_rn(_tmem_load_13[0], e_off_1219, _tmem_load_15[0] * e_dg_1220);
            dbp_1057[0] = dk2_1222 * k_t_1211;
            float dk2b_1223 = dk2_1222 * bt_t_1218;
            float dq_o_1224 = dq2_1221 + dqf_1215;
            float dk_o_1225 = dk2b_1223 + dkf_1216 + _tmem_load_16[0];
            float dg_o_1226 = __fmaf_rn(q_t_1214, dq2_1221, (dk2b_1223 - _tmem_load_16[0]) * k_t_1211) + dgf_1217;
            s_run_1059[0] = s_run_1059[0] + dg_o_1226;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th)), "f"(s_run_1059[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1227 = t_1207 * qrs;
            if (t_1207 < clen) {
                {
                    auto _gst_255 = dq_base + qoff_1227;
                    __builtin_assume(__isGlobal(_gst_255));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_255) + (0)) = (__nv_bfloat16)dq_o_1224;
                }
                {
                    auto _gst_256 = dk_base + qoff_1227;
                    __builtin_assume(__isGlobal(_gst_256));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_256) + (0)) = (__nv_bfloat16)dk_o_1225;
                }
            }
            float tot_1228 = s_run_1059[0];
            int b4_1229 = lane_1 >> 4 & 1;
            int b3_1230 = lane_1 >> 3 & 1;
            int b2_1231 = lane_1 >> 2 & 1;
            float v4_1232[4];
            float send_1233 = ((b4_1229 == 0) ? dbp_1057[4] : dbp_1057[0]);
            float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, send_1233, 16);
            v4_1232[0] = ((b4_1229 == 0) ? dbp_1057[0] : dbp_1057[4]) + _shfl_xor_sync_257;
            float send_1234 = ((b4_1229 == 0) ? dbp_1057[5] : dbp_1057[1]);
            float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, send_1234, 16);
            v4_1232[1] = ((b4_1229 == 0) ? dbp_1057[1] : dbp_1057[5]) + _shfl_xor_sync_258;
            float send_1235 = ((b4_1229 == 0) ? dbp_1057[6] : dbp_1057[2]);
            float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, send_1235, 16);
            v4_1232[2] = ((b4_1229 == 0) ? dbp_1057[2] : dbp_1057[6]) + _shfl_xor_sync_259;
            float send_1236 = ((b4_1229 == 0) ? dbp_1057[7] : dbp_1057[3]);
            float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, send_1236, 16);
            v4_1232[3] = ((b4_1229 == 0) ? dbp_1057[3] : dbp_1057[7]) + _shfl_xor_sync_260;
            float v2_1237[2];
            float send_1238 = ((b3_1230 == 0) ? v4_1232[2] : v4_1232[0]);
            float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, send_1238, 8);
            v2_1237[0] = ((b3_1230 == 0) ? v4_1232[0] : v4_1232[2]) + _shfl_xor_sync_261;
            float send_1239 = ((b3_1230 == 0) ? v4_1232[3] : v4_1232[1]);
            float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, send_1239, 8);
            v2_1237[1] = ((b3_1230 == 0) ? v4_1232[1] : v4_1232[3]) + _shfl_xor_sync_262;
            float send_1240 = ((b2_1231 == 0) ? v2_1237[1] : v2_1237[0]);
            float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, send_1240, 4);
            float v1_1241 = ((b2_1231 == 0) ? v2_1237[0] : v2_1237[1]) + _shfl_xor_sync_263;
            float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, v1_1241, 2);
            v1_1241 = v1_1241 + _shfl_xor_sync_264;
            float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, v1_1241, 1);
            v1_1241 = v1_1241 + _shfl_xor_sync_265;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (th) * 8);
            {
                uint32_t _addr_266 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_266), "f"(tot_1228) : "memory");
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
            float t_hi = ((th == 1) ? tot_1228 : t_other[0]);
            float t_lo = ((th == 1) ? t_other[0] : tot_1228);
            float offset = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi);
            float _tmem_load_17[8];
            tmem_ld_x8(&_tmem_load_17[0], taddr + (unsigned int)row_base + (unsigned int)(224 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1242 = 48 + 8 * th;
            int goff = t_1242 * rs_g;
            float dg = offset + _tmem_load_17[0];
            if (t_1242 < clen) {
                {
                    auto _gst_267 = dg_base + goff;
                    __builtin_assume(__isGlobal(_gst_267));
                    *(reinterpret_cast<float*>(_gst_267) + (0)) = dg;
                }
            }
            int t_1243 = 48 + 8 * th + 1;
            int goff_1244 = t_1243 * rs_g;
            float dg_1245 = offset + _tmem_load_17[1];
            if (t_1243 < clen) {
                {
                    auto _gst_268 = dg_base + goff_1244;
                    __builtin_assume(__isGlobal(_gst_268));
                    *(reinterpret_cast<float*>(_gst_268) + (0)) = dg_1245;
                }
            }
            int t_1246 = 48 + 8 * th + 2;
            int goff_1247 = t_1246 * rs_g;
            float dg_1248 = offset + _tmem_load_17[2];
            if (t_1246 < clen) {
                {
                    auto _gst_269 = dg_base + goff_1247;
                    __builtin_assume(__isGlobal(_gst_269));
                    *(reinterpret_cast<float*>(_gst_269) + (0)) = dg_1248;
                }
            }
            int t_1249 = 48 + 8 * th + 3;
            int goff_1250 = t_1249 * rs_g;
            float dg_1251 = offset + _tmem_load_17[3];
            if (t_1249 < clen) {
                {
                    auto _gst_270 = dg_base + goff_1250;
                    __builtin_assume(__isGlobal(_gst_270));
                    *(reinterpret_cast<float*>(_gst_270) + (0)) = dg_1251;
                }
            }
            int t_1252 = 48 + 8 * th + 4;
            int goff_1253 = t_1252 * rs_g;
            float dg_1254 = offset + _tmem_load_17[4];
            if (t_1252 < clen) {
                {
                    auto _gst_271 = dg_base + goff_1253;
                    __builtin_assume(__isGlobal(_gst_271));
                    *(reinterpret_cast<float*>(_gst_271) + (0)) = dg_1254;
                }
            }
            int t_1255 = 48 + 8 * th + 5;
            int goff_1256 = t_1255 * rs_g;
            float dg_1257 = offset + _tmem_load_17[5];
            if (t_1255 < clen) {
                {
                    auto _gst_272 = dg_base + goff_1256;
                    __builtin_assume(__isGlobal(_gst_272));
                    *(reinterpret_cast<float*>(_gst_272) + (0)) = dg_1257;
                }
            }
            int t_1258 = 48 + 8 * th + 6;
            int goff_1259 = t_1258 * rs_g;
            float dg_1260 = offset + _tmem_load_17[6];
            if (t_1258 < clen) {
                {
                    auto _gst_273 = dg_base + goff_1259;
                    __builtin_assume(__isGlobal(_gst_273));
                    *(reinterpret_cast<float*>(_gst_273) + (0)) = dg_1260;
                }
            }
            int t_1261 = 48 + 8 * th + 7;
            int goff_1262 = t_1261 * rs_g;
            float dg_1263 = offset + _tmem_load_17[7];
            if (t_1261 < clen) {
                {
                    auto _gst_274 = dg_base + goff_1262;
                    __builtin_assume(__isGlobal(_gst_274));
                    *(reinterpret_cast<float*>(_gst_274) + (0)) = dg_1263;
                }
            }
            off_acc[0] = off_acc[0] + t_hi + t_lo;
            float _tmem_load_18[4];
            tmem_ld_x4(&_tmem_load_18[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * (1 - th)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float t_hi_1264 = ((th == 1) ? tot : _tmem_load_18[0]);
            float t_lo_1265 = ((th == 1) ? _tmem_load_18[0] : tot);
            float offset_1266 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1264);
            float _tmem_load_19[8];
            tmem_ld_x8(&_tmem_load_19[0], taddr + (unsigned int)row_base + (unsigned int)(192 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1267 = 32 + 8 * th;
            int goff_1268 = t_1267 * rs_g;
            float dg_1269 = offset_1266 + _tmem_load_19[0];
            if (t_1267 < clen) {
                {
                    auto _gst_275 = dg_base + goff_1268;
                    __builtin_assume(__isGlobal(_gst_275));
                    *(reinterpret_cast<float*>(_gst_275) + (0)) = dg_1269;
                }
            }
            int t_1270 = 32 + 8 * th + 1;
            int goff_1271 = t_1270 * rs_g;
            float dg_1272 = offset_1266 + _tmem_load_19[1];
            if (t_1270 < clen) {
                {
                    auto _gst_276 = dg_base + goff_1271;
                    __builtin_assume(__isGlobal(_gst_276));
                    *(reinterpret_cast<float*>(_gst_276) + (0)) = dg_1272;
                }
            }
            int t_1273 = 32 + 8 * th + 2;
            int goff_1274 = t_1273 * rs_g;
            float dg_1275 = offset_1266 + _tmem_load_19[2];
            if (t_1273 < clen) {
                {
                    auto _gst_277 = dg_base + goff_1274;
                    __builtin_assume(__isGlobal(_gst_277));
                    *(reinterpret_cast<float*>(_gst_277) + (0)) = dg_1275;
                }
            }
            int t_1276 = 32 + 8 * th + 3;
            int goff_1277 = t_1276 * rs_g;
            float dg_1278 = offset_1266 + _tmem_load_19[3];
            if (t_1276 < clen) {
                {
                    auto _gst_278 = dg_base + goff_1277;
                    __builtin_assume(__isGlobal(_gst_278));
                    *(reinterpret_cast<float*>(_gst_278) + (0)) = dg_1278;
                }
            }
            int t_1279 = 32 + 8 * th + 4;
            int goff_1280 = t_1279 * rs_g;
            float dg_1281 = offset_1266 + _tmem_load_19[4];
            if (t_1279 < clen) {
                {
                    auto _gst_279 = dg_base + goff_1280;
                    __builtin_assume(__isGlobal(_gst_279));
                    *(reinterpret_cast<float*>(_gst_279) + (0)) = dg_1281;
                }
            }
            int t_1282 = 32 + 8 * th + 5;
            int goff_1283 = t_1282 * rs_g;
            float dg_1284 = offset_1266 + _tmem_load_19[5];
            if (t_1282 < clen) {
                {
                    auto _gst_280 = dg_base + goff_1283;
                    __builtin_assume(__isGlobal(_gst_280));
                    *(reinterpret_cast<float*>(_gst_280) + (0)) = dg_1284;
                }
            }
            int t_1285 = 32 + 8 * th + 6;
            int goff_1286 = t_1285 * rs_g;
            float dg_1287 = offset_1266 + _tmem_load_19[6];
            if (t_1285 < clen) {
                {
                    auto _gst_281 = dg_base + goff_1286;
                    __builtin_assume(__isGlobal(_gst_281));
                    *(reinterpret_cast<float*>(_gst_281) + (0)) = dg_1287;
                }
            }
            int t_1288 = 32 + 8 * th + 7;
            int goff_1289 = t_1288 * rs_g;
            float dg_1290 = offset_1266 + _tmem_load_19[7];
            if (t_1288 < clen) {
                {
                    auto _gst_282 = dg_base + goff_1289;
                    __builtin_assume(__isGlobal(_gst_282));
                    *(reinterpret_cast<float*>(_gst_282) + (0)) = dg_1290;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1264 + t_lo_1265;
            {
                uint32_t _addr_283 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_283), "f"(off_acc[0]) : "memory");
            }
            mbarrier_wait(p1b_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(epi_full_addr + (2 + th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1291[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1291[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 2048 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1292 = r_1291[0];
            float gc_1293 = gn_1292 - 21.640425f;
            int eb_1294 = 81920 + th * -32768;
            int ew_1295 = eb_1294 >> 2;
            int qbase_1296 = pool_f32_addr + (unsigned int)(eb_1294 + 12288 + dh_lane * 1024);
            int sb8_1297 = 16 + 8 * th;
            int gb8_1298 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1297 * 128);
            int kb8_1299 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1297 * 128);
            float dbp_1300[8];
            float _tmem_load_20[8];
            tmem_ld_x8(&_tmem_load_20[0], taddr + (unsigned int)row_base + (unsigned int)(64 + 8 * th));
            float _tmem_load_21[8];
            tmem_ld_x8(&_tmem_load_21[0], taddr + (unsigned int)row_base + (unsigned int)(80 + 8 * th));
            float _tmem_load_22[8];
            tmem_ld_x8(&_tmem_load_22[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th));
            float _tmem_load_23[8];
            tmem_ld_x8(&_tmem_load_23[0], taddr + (unsigned int)row_base + (unsigned int)(112 + 8 * th));
            float _tmem_load_24[8];
            tmem_ld_x8(&_tmem_load_24[0], taddr + (unsigned int)row_base + (unsigned int)(16 + 8 * th));
            float btv_1301[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (16 + 8 * th))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[4])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1301[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (16 + 8 * th) + 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_run_1302[1];
            s_run_1302[0] = 0.0f;
            int t_1303 = 16 + 8 * th + 7;
            float r_1304[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1304[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1305 = r_1304[0];
            unsigned int r_1306[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1306[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1307 = __uint_as_float(r_1306[0] << 16);
            unsigned int r_1308[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1308[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1309 = __uint_as_float(r_1308[0] << 16);
            float q_t_1310 = ((t_1303 >= clen) ? 0.0f : q_t_1309);
            float dqf_1311 = pool_f32[ew_1295 + 896 + d_lane];
            float dkf_1312 = pool_f32[ew_1295 + 1920 + d_lane];
            float dgf_1313 = pool_f32[ew_1295 + 2944 + d_lane];
            float bt_t_1314 = btv_1301[7];
            float e_off_1315 = approx_exp2(g_t_1305 - gn_1292);
            float e_dg_1316 = approx_exp2(g_t_1305 - gc_1293);
            float dq2_1317 = __fmaf_rn(_tmem_load_20[7], e_off_1315, _tmem_load_22[7] * e_dg_1316);
            float dk2_1318 = __fmaf_rn(_tmem_load_21[7], e_off_1315, _tmem_load_23[7] * e_dg_1316);
            dbp_1300[7] = dk2_1318 * k_t_1307;
            float dk2b_1319 = dk2_1318 * bt_t_1314;
            float dq_o_1320 = dq2_1317 + dqf_1311;
            float dk_o_1321 = dk2b_1319 + dkf_1312 + _tmem_load_24[7];
            float dg_o_1322 = __fmaf_rn(q_t_1310, dq2_1317, (dk2b_1319 - _tmem_load_24[7]) * k_t_1307) + dgf_1313;
            s_run_1302[0] = s_run_1302[0] + dg_o_1322;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 7)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1323 = t_1303 * qrs;
            if (t_1303 < clen) {
                {
                    auto _gst_284 = dq_base + qoff_1323;
                    __builtin_assume(__isGlobal(_gst_284));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_284) + (0)) = (__nv_bfloat16)dq_o_1320;
                }
                {
                    auto _gst_285 = dk_base + qoff_1323;
                    __builtin_assume(__isGlobal(_gst_285));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_285) + (0)) = (__nv_bfloat16)dk_o_1321;
                }
            }
            int t_1324 = 16 + 8 * th + 6;
            float r_1325[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1325[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1326 = r_1325[0];
            unsigned int r_1327[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1327[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1328 = __uint_as_float(r_1327[0] << 16);
            unsigned int r_1329[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1329[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1330 = __uint_as_float(r_1329[0] << 16);
            float q_t_1331 = ((t_1324 >= clen) ? 0.0f : q_t_1330);
            float dqf_1332 = pool_f32[ew_1295 + 768 + d_lane];
            float dkf_1333 = pool_f32[ew_1295 + 1792 + d_lane];
            float dgf_1334 = pool_f32[ew_1295 + 2816 + d_lane];
            float bt_t_1335 = btv_1301[6];
            float e_off_1336 = approx_exp2(g_t_1326 - gn_1292);
            float e_dg_1337 = approx_exp2(g_t_1326 - gc_1293);
            float dq2_1338 = __fmaf_rn(_tmem_load_20[6], e_off_1336, _tmem_load_22[6] * e_dg_1337);
            float dk2_1339 = __fmaf_rn(_tmem_load_21[6], e_off_1336, _tmem_load_23[6] * e_dg_1337);
            dbp_1300[6] = dk2_1339 * k_t_1328;
            float dk2b_1340 = dk2_1339 * bt_t_1335;
            float dq_o_1341 = dq2_1338 + dqf_1332;
            float dk_o_1342 = dk2b_1340 + dkf_1333 + _tmem_load_24[6];
            float dg_o_1343 = __fmaf_rn(q_t_1331, dq2_1338, (dk2b_1340 - _tmem_load_24[6]) * k_t_1328) + dgf_1334;
            s_run_1302[0] = s_run_1302[0] + dg_o_1343;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 6)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1344 = t_1324 * qrs;
            if (t_1324 < clen) {
                {
                    auto _gst_286 = dq_base + qoff_1344;
                    __builtin_assume(__isGlobal(_gst_286));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_286) + (0)) = (__nv_bfloat16)dq_o_1341;
                }
                {
                    auto _gst_287 = dk_base + qoff_1344;
                    __builtin_assume(__isGlobal(_gst_287));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_287) + (0)) = (__nv_bfloat16)dk_o_1342;
                }
            }
            int t_1345 = 16 + 8 * th + 5;
            float r_1346[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1346[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1347 = r_1346[0];
            unsigned int r_1348[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1348[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1349 = __uint_as_float(r_1348[0] << 16);
            unsigned int r_1350[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1350[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1351 = __uint_as_float(r_1350[0] << 16);
            float q_t_1352 = ((t_1345 >= clen) ? 0.0f : q_t_1351);
            float dqf_1353 = pool_f32[ew_1295 + 640 + d_lane];
            float dkf_1354 = pool_f32[ew_1295 + 1664 + d_lane];
            float dgf_1355 = pool_f32[ew_1295 + 2688 + d_lane];
            float bt_t_1356 = btv_1301[5];
            float e_off_1357 = approx_exp2(g_t_1347 - gn_1292);
            float e_dg_1358 = approx_exp2(g_t_1347 - gc_1293);
            float dq2_1359 = __fmaf_rn(_tmem_load_20[5], e_off_1357, _tmem_load_22[5] * e_dg_1358);
            float dk2_1360 = __fmaf_rn(_tmem_load_21[5], e_off_1357, _tmem_load_23[5] * e_dg_1358);
            dbp_1300[5] = dk2_1360 * k_t_1349;
            float dk2b_1361 = dk2_1360 * bt_t_1356;
            float dq_o_1362 = dq2_1359 + dqf_1353;
            float dk_o_1363 = dk2b_1361 + dkf_1354 + _tmem_load_24[5];
            float dg_o_1364 = __fmaf_rn(q_t_1352, dq2_1359, (dk2b_1361 - _tmem_load_24[5]) * k_t_1349) + dgf_1355;
            s_run_1302[0] = s_run_1302[0] + dg_o_1364;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 5)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1365 = t_1345 * qrs;
            if (t_1345 < clen) {
                {
                    auto _gst_288 = dq_base + qoff_1365;
                    __builtin_assume(__isGlobal(_gst_288));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_288) + (0)) = (__nv_bfloat16)dq_o_1362;
                }
                {
                    auto _gst_289 = dk_base + qoff_1365;
                    __builtin_assume(__isGlobal(_gst_289));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_289) + (0)) = (__nv_bfloat16)dk_o_1363;
                }
            }
            int t_1366 = 16 + 8 * th + 4;
            float r_1367[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1367[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1368 = r_1367[0];
            unsigned int r_1369[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1369[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1370 = __uint_as_float(r_1369[0] << 16);
            unsigned int r_1371[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1371[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1372 = __uint_as_float(r_1371[0] << 16);
            float q_t_1373 = ((t_1366 >= clen) ? 0.0f : q_t_1372);
            float dqf_1374 = pool_f32[ew_1295 + 512 + d_lane];
            float dkf_1375 = pool_f32[ew_1295 + 1536 + d_lane];
            float dgf_1376 = pool_f32[ew_1295 + 2560 + d_lane];
            float bt_t_1377 = btv_1301[4];
            float e_off_1378 = approx_exp2(g_t_1368 - gn_1292);
            float e_dg_1379 = approx_exp2(g_t_1368 - gc_1293);
            float dq2_1380 = __fmaf_rn(_tmem_load_20[4], e_off_1378, _tmem_load_22[4] * e_dg_1379);
            float dk2_1381 = __fmaf_rn(_tmem_load_21[4], e_off_1378, _tmem_load_23[4] * e_dg_1379);
            dbp_1300[4] = dk2_1381 * k_t_1370;
            float dk2b_1382 = dk2_1381 * bt_t_1377;
            float dq_o_1383 = dq2_1380 + dqf_1374;
            float dk_o_1384 = dk2b_1382 + dkf_1375 + _tmem_load_24[4];
            float dg_o_1385 = __fmaf_rn(q_t_1373, dq2_1380, (dk2b_1382 - _tmem_load_24[4]) * k_t_1370) + dgf_1376;
            s_run_1302[0] = s_run_1302[0] + dg_o_1385;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 4)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1386 = t_1366 * qrs;
            if (t_1366 < clen) {
                {
                    auto _gst_290 = dq_base + qoff_1386;
                    __builtin_assume(__isGlobal(_gst_290));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_290) + (0)) = (__nv_bfloat16)dq_o_1383;
                }
                {
                    auto _gst_291 = dk_base + qoff_1386;
                    __builtin_assume(__isGlobal(_gst_291));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_291) + (0)) = (__nv_bfloat16)dk_o_1384;
                }
            }
            int t_1387 = 16 + 8 * th + 3;
            float r_1388[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1388[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1389 = r_1388[0];
            unsigned int r_1390[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1390[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1391 = __uint_as_float(r_1390[0] << 16);
            unsigned int r_1392[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1392[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1393 = __uint_as_float(r_1392[0] << 16);
            float q_t_1394 = ((t_1387 >= clen) ? 0.0f : q_t_1393);
            float dqf_1395 = pool_f32[ew_1295 + 384 + d_lane];
            float dkf_1396 = pool_f32[ew_1295 + 1408 + d_lane];
            float dgf_1397 = pool_f32[ew_1295 + 2432 + d_lane];
            float bt_t_1398 = btv_1301[3];
            float e_off_1399 = approx_exp2(g_t_1389 - gn_1292);
            float e_dg_1400 = approx_exp2(g_t_1389 - gc_1293);
            float dq2_1401 = __fmaf_rn(_tmem_load_20[3], e_off_1399, _tmem_load_22[3] * e_dg_1400);
            float dk2_1402 = __fmaf_rn(_tmem_load_21[3], e_off_1399, _tmem_load_23[3] * e_dg_1400);
            dbp_1300[3] = dk2_1402 * k_t_1391;
            float dk2b_1403 = dk2_1402 * bt_t_1398;
            float dq_o_1404 = dq2_1401 + dqf_1395;
            float dk_o_1405 = dk2b_1403 + dkf_1396 + _tmem_load_24[3];
            float dg_o_1406 = __fmaf_rn(q_t_1394, dq2_1401, (dk2b_1403 - _tmem_load_24[3]) * k_t_1391) + dgf_1397;
            s_run_1302[0] = s_run_1302[0] + dg_o_1406;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 3)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1407 = t_1387 * qrs;
            if (t_1387 < clen) {
                {
                    auto _gst_292 = dq_base + qoff_1407;
                    __builtin_assume(__isGlobal(_gst_292));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_292) + (0)) = (__nv_bfloat16)dq_o_1404;
                }
                {
                    auto _gst_293 = dk_base + qoff_1407;
                    __builtin_assume(__isGlobal(_gst_293));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_293) + (0)) = (__nv_bfloat16)dk_o_1405;
                }
            }
            int t_1408 = 16 + 8 * th + 2;
            float r_1409[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1409[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1410 = r_1409[0];
            unsigned int r_1411[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1411[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1412 = __uint_as_float(r_1411[0] << 16);
            unsigned int r_1413[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1413[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1414 = __uint_as_float(r_1413[0] << 16);
            float q_t_1415 = ((t_1408 >= clen) ? 0.0f : q_t_1414);
            float dqf_1416 = pool_f32[ew_1295 + 256 + d_lane];
            float dkf_1417 = pool_f32[ew_1295 + 1280 + d_lane];
            float dgf_1418 = pool_f32[ew_1295 + 2304 + d_lane];
            float bt_t_1419 = btv_1301[2];
            float e_off_1420 = approx_exp2(g_t_1410 - gn_1292);
            float e_dg_1421 = approx_exp2(g_t_1410 - gc_1293);
            float dq2_1422 = __fmaf_rn(_tmem_load_20[2], e_off_1420, _tmem_load_22[2] * e_dg_1421);
            float dk2_1423 = __fmaf_rn(_tmem_load_21[2], e_off_1420, _tmem_load_23[2] * e_dg_1421);
            dbp_1300[2] = dk2_1423 * k_t_1412;
            float dk2b_1424 = dk2_1423 * bt_t_1419;
            float dq_o_1425 = dq2_1422 + dqf_1416;
            float dk_o_1426 = dk2b_1424 + dkf_1417 + _tmem_load_24[2];
            float dg_o_1427 = __fmaf_rn(q_t_1415, dq2_1422, (dk2b_1424 - _tmem_load_24[2]) * k_t_1412) + dgf_1418;
            s_run_1302[0] = s_run_1302[0] + dg_o_1427;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 2)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1428 = t_1408 * qrs;
            if (t_1408 < clen) {
                {
                    auto _gst_294 = dq_base + qoff_1428;
                    __builtin_assume(__isGlobal(_gst_294));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_294) + (0)) = (__nv_bfloat16)dq_o_1425;
                }
                {
                    auto _gst_295 = dk_base + qoff_1428;
                    __builtin_assume(__isGlobal(_gst_295));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_295) + (0)) = (__nv_bfloat16)dk_o_1426;
                }
            }
            int t_1429 = 16 + 8 * th + 1;
            float r_1430[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1430[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1431 = r_1430[0];
            unsigned int r_1432[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1432[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1433 = __uint_as_float(r_1432[0] << 16);
            unsigned int r_1434[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1434[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1435 = __uint_as_float(r_1434[0] << 16);
            float q_t_1436 = ((t_1429 >= clen) ? 0.0f : q_t_1435);
            float dqf_1437 = pool_f32[ew_1295 + 128 + d_lane];
            float dkf_1438 = pool_f32[ew_1295 + 1152 + d_lane];
            float dgf_1439 = pool_f32[ew_1295 + 2176 + d_lane];
            float bt_t_1440 = btv_1301[1];
            float e_off_1441 = approx_exp2(g_t_1431 - gn_1292);
            float e_dg_1442 = approx_exp2(g_t_1431 - gc_1293);
            float dq2_1443 = __fmaf_rn(_tmem_load_20[1], e_off_1441, _tmem_load_22[1] * e_dg_1442);
            float dk2_1444 = __fmaf_rn(_tmem_load_21[1], e_off_1441, _tmem_load_23[1] * e_dg_1442);
            dbp_1300[1] = dk2_1444 * k_t_1433;
            float dk2b_1445 = dk2_1444 * bt_t_1440;
            float dq_o_1446 = dq2_1443 + dqf_1437;
            float dk_o_1447 = dk2b_1445 + dkf_1438 + _tmem_load_24[1];
            float dg_o_1448 = __fmaf_rn(q_t_1436, dq2_1443, (dk2b_1445 - _tmem_load_24[1]) * k_t_1433) + dgf_1439;
            s_run_1302[0] = s_run_1302[0] + dg_o_1448;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th + 1)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1449 = t_1429 * qrs;
            if (t_1429 < clen) {
                {
                    auto _gst_296 = dq_base + qoff_1449;
                    __builtin_assume(__isGlobal(_gst_296));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_296) + (0)) = (__nv_bfloat16)dq_o_1446;
                }
                {
                    auto _gst_297 = dk_base + qoff_1449;
                    __builtin_assume(__isGlobal(_gst_297));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_297) + (0)) = (__nv_bfloat16)dk_o_1447;
                }
            }
            int t_1450 = 16 + 8 * th;
            float r_1451[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1451[0])) : "r"((gb8_1298 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1452 = r_1451[0];
            unsigned int r_1453[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1453[0])) : "r"((kb8_1299 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1454 = __uint_as_float(r_1453[0] << 16);
            unsigned int r_1455[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1455[0])) : "r"((qbase_1296 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1456 = __uint_as_float(r_1455[0] << 16);
            float q_t_1457 = ((t_1450 >= clen) ? 0.0f : q_t_1456);
            float dqf_1458 = pool_f32[ew_1295 + d_lane];
            float dkf_1459 = pool_f32[ew_1295 + 1024 + d_lane];
            float dgf_1460 = pool_f32[ew_1295 + 2048 + d_lane];
            float bt_t_1461 = btv_1301[0];
            float e_off_1462 = approx_exp2(g_t_1452 - gn_1292);
            float e_dg_1463 = approx_exp2(g_t_1452 - gc_1293);
            float dq2_1464 = __fmaf_rn(_tmem_load_20[0], e_off_1462, _tmem_load_22[0] * e_dg_1463);
            float dk2_1465 = __fmaf_rn(_tmem_load_21[0], e_off_1462, _tmem_load_23[0] * e_dg_1463);
            dbp_1300[0] = dk2_1465 * k_t_1454;
            float dk2b_1466 = dk2_1465 * bt_t_1461;
            float dq_o_1467 = dq2_1464 + dqf_1458;
            float dk_o_1468 = dk2b_1466 + dkf_1459 + _tmem_load_24[0];
            float dg_o_1469 = __fmaf_rn(q_t_1457, dq2_1464, (dk2b_1466 - _tmem_load_24[0]) * k_t_1454) + dgf_1460;
            s_run_1302[0] = s_run_1302[0] + dg_o_1469;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th)), "f"(s_run_1302[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1470 = t_1450 * qrs;
            if (t_1450 < clen) {
                {
                    auto _gst_298 = dq_base + qoff_1470;
                    __builtin_assume(__isGlobal(_gst_298));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_298) + (0)) = (__nv_bfloat16)dq_o_1467;
                }
                {
                    auto _gst_299 = dk_base + qoff_1470;
                    __builtin_assume(__isGlobal(_gst_299));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_299) + (0)) = (__nv_bfloat16)dk_o_1468;
                }
            }
            float tot_1471 = s_run_1302[0];
            int b4_1472 = lane_1 >> 4 & 1;
            int b3_1473 = lane_1 >> 3 & 1;
            int b2_1474 = lane_1 >> 2 & 1;
            float v4_1475[4];
            float send_1476 = ((b4_1472 == 0) ? dbp_1300[4] : dbp_1300[0]);
            float _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, send_1476, 16);
            v4_1475[0] = ((b4_1472 == 0) ? dbp_1300[0] : dbp_1300[4]) + _shfl_xor_sync_300;
            float send_1477 = ((b4_1472 == 0) ? dbp_1300[5] : dbp_1300[1]);
            float _shfl_xor_sync_301 = __shfl_xor_sync(0xFFFFFFFF, send_1477, 16);
            v4_1475[1] = ((b4_1472 == 0) ? dbp_1300[1] : dbp_1300[5]) + _shfl_xor_sync_301;
            float send_1478 = ((b4_1472 == 0) ? dbp_1300[6] : dbp_1300[2]);
            float _shfl_xor_sync_302 = __shfl_xor_sync(0xFFFFFFFF, send_1478, 16);
            v4_1475[2] = ((b4_1472 == 0) ? dbp_1300[2] : dbp_1300[6]) + _shfl_xor_sync_302;
            float send_1479 = ((b4_1472 == 0) ? dbp_1300[7] : dbp_1300[3]);
            float _shfl_xor_sync_303 = __shfl_xor_sync(0xFFFFFFFF, send_1479, 16);
            v4_1475[3] = ((b4_1472 == 0) ? dbp_1300[3] : dbp_1300[7]) + _shfl_xor_sync_303;
            float v2_1480[2];
            float send_1481 = ((b3_1473 == 0) ? v4_1475[2] : v4_1475[0]);
            float _shfl_xor_sync_304 = __shfl_xor_sync(0xFFFFFFFF, send_1481, 8);
            v2_1480[0] = ((b3_1473 == 0) ? v4_1475[0] : v4_1475[2]) + _shfl_xor_sync_304;
            float send_1482 = ((b3_1473 == 0) ? v4_1475[3] : v4_1475[1]);
            float _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, send_1482, 8);
            v2_1480[1] = ((b3_1473 == 0) ? v4_1475[1] : v4_1475[3]) + _shfl_xor_sync_305;
            float send_1483 = ((b2_1474 == 0) ? v2_1480[1] : v2_1480[0]);
            float _shfl_xor_sync_306 = __shfl_xor_sync(0xFFFFFFFF, send_1483, 4);
            float v1_1484 = ((b2_1474 == 0) ? v2_1480[0] : v2_1480[1]) + _shfl_xor_sync_306;
            float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, v1_1484, 2);
            v1_1484 = v1_1484 + _shfl_xor_sync_307;
            float _shfl_xor_sync_308 = __shfl_xor_sync(0xFFFFFFFF, v1_1484, 1);
            v1_1484 = v1_1484 + _shfl_xor_sync_308;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(epi_empty_addr + (2 + th) * 8);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_309 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_309), "f"(tot_1471) : "memory");
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
            float t_other_1485[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_other_1485[0])) : "r"(pool_f32_addr + (unsigned int)(112640 + 4 * (qd * 64 + (1 - th) * 32 + lane_1))));
            float t_hi_1486 = ((th == 1) ? tot_1471 : t_other_1485[0]);
            float t_lo_1487 = ((th == 1) ? t_other_1485[0] : tot_1471);
            float offset_1488 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1486);
            float _tmem_load_25[8];
            tmem_ld_x8(&_tmem_load_25[0], taddr + (unsigned int)row_base + (unsigned int)(96 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1489 = 16 + 8 * th;
            int goff_1490 = t_1489 * rs_g;
            float dg_1491 = offset_1488 + _tmem_load_25[0];
            if (t_1489 < clen) {
                {
                    auto _gst_310 = dg_base + goff_1490;
                    __builtin_assume(__isGlobal(_gst_310));
                    *(reinterpret_cast<float*>(_gst_310) + (0)) = dg_1491;
                }
            }
            int t_1492 = 16 + 8 * th + 1;
            int goff_1493 = t_1492 * rs_g;
            float dg_1494 = offset_1488 + _tmem_load_25[1];
            if (t_1492 < clen) {
                {
                    auto _gst_311 = dg_base + goff_1493;
                    __builtin_assume(__isGlobal(_gst_311));
                    *(reinterpret_cast<float*>(_gst_311) + (0)) = dg_1494;
                }
            }
            int t_1495 = 16 + 8 * th + 2;
            int goff_1496 = t_1495 * rs_g;
            float dg_1497 = offset_1488 + _tmem_load_25[2];
            if (t_1495 < clen) {
                {
                    auto _gst_312 = dg_base + goff_1496;
                    __builtin_assume(__isGlobal(_gst_312));
                    *(reinterpret_cast<float*>(_gst_312) + (0)) = dg_1497;
                }
            }
            int t_1498 = 16 + 8 * th + 3;
            int goff_1499 = t_1498 * rs_g;
            float dg_1500 = offset_1488 + _tmem_load_25[3];
            if (t_1498 < clen) {
                {
                    auto _gst_313 = dg_base + goff_1499;
                    __builtin_assume(__isGlobal(_gst_313));
                    *(reinterpret_cast<float*>(_gst_313) + (0)) = dg_1500;
                }
            }
            int t_1501 = 16 + 8 * th + 4;
            int goff_1502 = t_1501 * rs_g;
            float dg_1503 = offset_1488 + _tmem_load_25[4];
            if (t_1501 < clen) {
                {
                    auto _gst_314 = dg_base + goff_1502;
                    __builtin_assume(__isGlobal(_gst_314));
                    *(reinterpret_cast<float*>(_gst_314) + (0)) = dg_1503;
                }
            }
            int t_1504 = 16 + 8 * th + 5;
            int goff_1505 = t_1504 * rs_g;
            float dg_1506 = offset_1488 + _tmem_load_25[5];
            if (t_1504 < clen) {
                {
                    auto _gst_315 = dg_base + goff_1505;
                    __builtin_assume(__isGlobal(_gst_315));
                    *(reinterpret_cast<float*>(_gst_315) + (0)) = dg_1506;
                }
            }
            int t_1507 = 16 + 8 * th + 6;
            int goff_1508 = t_1507 * rs_g;
            float dg_1509 = offset_1488 + _tmem_load_25[6];
            if (t_1507 < clen) {
                {
                    auto _gst_316 = dg_base + goff_1508;
                    __builtin_assume(__isGlobal(_gst_316));
                    *(reinterpret_cast<float*>(_gst_316) + (0)) = dg_1509;
                }
            }
            int t_1510 = 16 + 8 * th + 7;
            int goff_1511 = t_1510 * rs_g;
            float dg_1512 = offset_1488 + _tmem_load_25[7];
            if (t_1510 < clen) {
                {
                    auto _gst_317 = dg_base + goff_1511;
                    __builtin_assume(__isGlobal(_gst_317));
                    *(reinterpret_cast<float*>(_gst_317) + (0)) = dg_1512;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1486 + t_lo_1487;
            {
                uint32_t _addr_318 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_318), "f"(off_acc[0]) : "memory");
            }
            mbarrier_wait(epi_full_addr + (th) * 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float r_1513[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1513[0])) : "r"((gk_p_addr + (unsigned int)(qd * 8192) + (unsigned int)(lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float gn_1514 = r_1513[0];
            float gc_1515 = gn_1514 - 21.640425f;
            int eb_1516 = 98304 + th * -32768;
            int ew_1517 = eb_1516 >> 2;
            int qbase_1518 = pool_f32_addr + (unsigned int)(eb_1516 + 12288 + dh_lane * 1024);
            int sb8_1519 = 8 * th;
            int gb8_1520 = pool_f32_addr + (unsigned int)(qd * 8192 + sb8_1519 * 128);
            int kb8_1521 = pool_f32_addr + (unsigned int)(32768 + dh_lane * 8192 + sb8_1519 * 128);
            float dbp_1522[8];
            float _tmem_load_26[8];
            tmem_ld_x8(&_tmem_load_26[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            float _tmem_load_27[8];
            tmem_ld_x8(&_tmem_load_27[0], taddr + (unsigned int)row_base + (unsigned int)(144 + 8 * th));
            float _tmem_load_28[8];
            tmem_ld_x8(&_tmem_load_28[0], taddr + (unsigned int)row_base + (unsigned int)(8 * th));
            float btv_1523[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[0])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(0) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (8 * th))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[4])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&btv_1523[(4) + 3]))
                : "r"(pool_f32_addr + (unsigned int)(114688 + 4 * (8 * th) + 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_run_1524[1];
            s_run_1524[0] = 0.0f;
            int t_1525 = 8 * th + 7;
            float r_1526[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1526[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 896 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1527 = r_1526[0];
            unsigned int r_1528[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1528[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1529 = __uint_as_float(r_1528[0] << 16);
            unsigned int r_1530[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1530[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 896 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 896 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1531 = __uint_as_float(r_1530[0] << 16);
            float q_t_1532 = ((t_1525 >= clen) ? 0.0f : q_t_1531);
            float dqf_1533 = pool_f32[ew_1517 + 896 + d_lane];
            float dkf_1534 = pool_f32[ew_1517 + 1920 + d_lane];
            float dgf_1535 = pool_f32[ew_1517 + 2944 + d_lane];
            float bt_t_1536 = btv_1523[7];
            float e_off_1537 = approx_exp2(g_t_1527 - gn_1514);
            float e_dg_1538 = approx_exp2(g_t_1527 - gc_1515);
            float dq2_1539 = _tmem_load_26[7] * e_dg_1538;
            float dk2_1540 = _tmem_load_27[7] * e_dg_1538;
            dbp_1522[7] = dk2_1540 * k_t_1529;
            float dk2b_1541 = dk2_1540 * bt_t_1536;
            float dq_o_1542 = dq2_1539 + dqf_1533;
            float dk_o_1543 = dk2b_1541 + dkf_1534 + _tmem_load_28[7];
            float dg_o_1544 = __fmaf_rn(q_t_1532, dq2_1539, (dk2b_1541 - _tmem_load_28[7]) * k_t_1529) + dgf_1535;
            s_run_1524[0] = s_run_1524[0] + dg_o_1544;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 7)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1545 = t_1525 * qrs;
            if (t_1525 < clen) {
                {
                    auto _gst_319 = dq_base + qoff_1545;
                    __builtin_assume(__isGlobal(_gst_319));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_319) + (0)) = (__nv_bfloat16)dq_o_1542;
                }
                {
                    auto _gst_320 = dk_base + qoff_1545;
                    __builtin_assume(__isGlobal(_gst_320));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_320) + (0)) = (__nv_bfloat16)dk_o_1543;
                }
            }
            int t_1546 = 8 * th + 6;
            float r_1547[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1547[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 768 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1548 = r_1547[0];
            unsigned int r_1549[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1549[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1550 = __uint_as_float(r_1549[0] << 16);
            unsigned int r_1551[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1551[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 768 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 768 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1552 = __uint_as_float(r_1551[0] << 16);
            float q_t_1553 = ((t_1546 >= clen) ? 0.0f : q_t_1552);
            float dqf_1554 = pool_f32[ew_1517 + 768 + d_lane];
            float dkf_1555 = pool_f32[ew_1517 + 1792 + d_lane];
            float dgf_1556 = pool_f32[ew_1517 + 2816 + d_lane];
            float bt_t_1557 = btv_1523[6];
            float e_off_1558 = approx_exp2(g_t_1548 - gn_1514);
            float e_dg_1559 = approx_exp2(g_t_1548 - gc_1515);
            float dq2_1560 = _tmem_load_26[6] * e_dg_1559;
            float dk2_1561 = _tmem_load_27[6] * e_dg_1559;
            dbp_1522[6] = dk2_1561 * k_t_1550;
            float dk2b_1562 = dk2_1561 * bt_t_1557;
            float dq_o_1563 = dq2_1560 + dqf_1554;
            float dk_o_1564 = dk2b_1562 + dkf_1555 + _tmem_load_28[6];
            float dg_o_1565 = __fmaf_rn(q_t_1553, dq2_1560, (dk2b_1562 - _tmem_load_28[6]) * k_t_1550) + dgf_1556;
            s_run_1524[0] = s_run_1524[0] + dg_o_1565;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 6)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1566 = t_1546 * qrs;
            if (t_1546 < clen) {
                {
                    auto _gst_321 = dq_base + qoff_1566;
                    __builtin_assume(__isGlobal(_gst_321));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_321) + (0)) = (__nv_bfloat16)dq_o_1563;
                }
                {
                    auto _gst_322 = dk_base + qoff_1566;
                    __builtin_assume(__isGlobal(_gst_322));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_322) + (0)) = (__nv_bfloat16)dk_o_1564;
                }
            }
            int t_1567 = 8 * th + 5;
            float r_1568[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1568[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 640 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1569 = r_1568[0];
            unsigned int r_1570[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1570[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1571 = __uint_as_float(r_1570[0] << 16);
            unsigned int r_1572[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1572[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 640 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 640 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1573 = __uint_as_float(r_1572[0] << 16);
            float q_t_1574 = ((t_1567 >= clen) ? 0.0f : q_t_1573);
            float dqf_1575 = pool_f32[ew_1517 + 640 + d_lane];
            float dkf_1576 = pool_f32[ew_1517 + 1664 + d_lane];
            float dgf_1577 = pool_f32[ew_1517 + 2688 + d_lane];
            float bt_t_1578 = btv_1523[5];
            float e_off_1579 = approx_exp2(g_t_1569 - gn_1514);
            float e_dg_1580 = approx_exp2(g_t_1569 - gc_1515);
            float dq2_1581 = _tmem_load_26[5] * e_dg_1580;
            float dk2_1582 = _tmem_load_27[5] * e_dg_1580;
            dbp_1522[5] = dk2_1582 * k_t_1571;
            float dk2b_1583 = dk2_1582 * bt_t_1578;
            float dq_o_1584 = dq2_1581 + dqf_1575;
            float dk_o_1585 = dk2b_1583 + dkf_1576 + _tmem_load_28[5];
            float dg_o_1586 = __fmaf_rn(q_t_1574, dq2_1581, (dk2b_1583 - _tmem_load_28[5]) * k_t_1571) + dgf_1577;
            s_run_1524[0] = s_run_1524[0] + dg_o_1586;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 5)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1587 = t_1567 * qrs;
            if (t_1567 < clen) {
                {
                    auto _gst_323 = dq_base + qoff_1587;
                    __builtin_assume(__isGlobal(_gst_323));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_323) + (0)) = (__nv_bfloat16)dq_o_1584;
                }
                {
                    auto _gst_324 = dk_base + qoff_1587;
                    __builtin_assume(__isGlobal(_gst_324));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_324) + (0)) = (__nv_bfloat16)dk_o_1585;
                }
            }
            int t_1588 = 8 * th + 4;
            float r_1589[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1589[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 512 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1590 = r_1589[0];
            unsigned int r_1591[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1591[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1592 = __uint_as_float(r_1591[0] << 16);
            unsigned int r_1593[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1593[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 512 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 512 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1594 = __uint_as_float(r_1593[0] << 16);
            float q_t_1595 = ((t_1588 >= clen) ? 0.0f : q_t_1594);
            float dqf_1596 = pool_f32[ew_1517 + 512 + d_lane];
            float dkf_1597 = pool_f32[ew_1517 + 1536 + d_lane];
            float dgf_1598 = pool_f32[ew_1517 + 2560 + d_lane];
            float bt_t_1599 = btv_1523[4];
            float e_off_1600 = approx_exp2(g_t_1590 - gn_1514);
            float e_dg_1601 = approx_exp2(g_t_1590 - gc_1515);
            float dq2_1602 = _tmem_load_26[4] * e_dg_1601;
            float dk2_1603 = _tmem_load_27[4] * e_dg_1601;
            dbp_1522[4] = dk2_1603 * k_t_1592;
            float dk2b_1604 = dk2_1603 * bt_t_1599;
            float dq_o_1605 = dq2_1602 + dqf_1596;
            float dk_o_1606 = dk2b_1604 + dkf_1597 + _tmem_load_28[4];
            float dg_o_1607 = __fmaf_rn(q_t_1595, dq2_1602, (dk2b_1604 - _tmem_load_28[4]) * k_t_1592) + dgf_1598;
            s_run_1524[0] = s_run_1524[0] + dg_o_1607;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 4)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1608 = t_1588 * qrs;
            if (t_1588 < clen) {
                {
                    auto _gst_325 = dq_base + qoff_1608;
                    __builtin_assume(__isGlobal(_gst_325));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_325) + (0)) = (__nv_bfloat16)dq_o_1605;
                }
                {
                    auto _gst_326 = dk_base + qoff_1608;
                    __builtin_assume(__isGlobal(_gst_326));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_326) + (0)) = (__nv_bfloat16)dk_o_1606;
                }
            }
            int t_1609 = 8 * th + 3;
            float r_1610[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1610[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 384 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1611 = r_1610[0];
            unsigned int r_1612[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1612[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1613 = __uint_as_float(r_1612[0] << 16);
            unsigned int r_1614[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1614[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 384 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 384 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1615 = __uint_as_float(r_1614[0] << 16);
            float q_t_1616 = ((t_1609 >= clen) ? 0.0f : q_t_1615);
            float dqf_1617 = pool_f32[ew_1517 + 384 + d_lane];
            float dkf_1618 = pool_f32[ew_1517 + 1408 + d_lane];
            float dgf_1619 = pool_f32[ew_1517 + 2432 + d_lane];
            float bt_t_1620 = btv_1523[3];
            float e_off_1621 = approx_exp2(g_t_1611 - gn_1514);
            float e_dg_1622 = approx_exp2(g_t_1611 - gc_1515);
            float dq2_1623 = _tmem_load_26[3] * e_dg_1622;
            float dk2_1624 = _tmem_load_27[3] * e_dg_1622;
            dbp_1522[3] = dk2_1624 * k_t_1613;
            float dk2b_1625 = dk2_1624 * bt_t_1620;
            float dq_o_1626 = dq2_1623 + dqf_1617;
            float dk_o_1627 = dk2b_1625 + dkf_1618 + _tmem_load_28[3];
            float dg_o_1628 = __fmaf_rn(q_t_1616, dq2_1623, (dk2b_1625 - _tmem_load_28[3]) * k_t_1613) + dgf_1619;
            s_run_1524[0] = s_run_1524[0] + dg_o_1628;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 3)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1629 = t_1609 * qrs;
            if (t_1609 < clen) {
                {
                    auto _gst_327 = dq_base + qoff_1629;
                    __builtin_assume(__isGlobal(_gst_327));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_327) + (0)) = (__nv_bfloat16)dq_o_1626;
                }
                {
                    auto _gst_328 = dk_base + qoff_1629;
                    __builtin_assume(__isGlobal(_gst_328));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_328) + (0)) = (__nv_bfloat16)dk_o_1627;
                }
            }
            int t_1630 = 8 * th + 2;
            float r_1631[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1631[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 256 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1632 = r_1631[0];
            unsigned int r_1633[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1633[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1634 = __uint_as_float(r_1633[0] << 16);
            unsigned int r_1635[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1635[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 256 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 256 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1636 = __uint_as_float(r_1635[0] << 16);
            float q_t_1637 = ((t_1630 >= clen) ? 0.0f : q_t_1636);
            float dqf_1638 = pool_f32[ew_1517 + 256 + d_lane];
            float dkf_1639 = pool_f32[ew_1517 + 1280 + d_lane];
            float dgf_1640 = pool_f32[ew_1517 + 2304 + d_lane];
            float bt_t_1641 = btv_1523[2];
            float e_off_1642 = approx_exp2(g_t_1632 - gn_1514);
            float e_dg_1643 = approx_exp2(g_t_1632 - gc_1515);
            float dq2_1644 = _tmem_load_26[2] * e_dg_1643;
            float dk2_1645 = _tmem_load_27[2] * e_dg_1643;
            dbp_1522[2] = dk2_1645 * k_t_1634;
            float dk2b_1646 = dk2_1645 * bt_t_1641;
            float dq_o_1647 = dq2_1644 + dqf_1638;
            float dk_o_1648 = dk2b_1646 + dkf_1639 + _tmem_load_28[2];
            float dg_o_1649 = __fmaf_rn(q_t_1637, dq2_1644, (dk2b_1646 - _tmem_load_28[2]) * k_t_1634) + dgf_1640;
            s_run_1524[0] = s_run_1524[0] + dg_o_1649;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 2)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1650 = t_1630 * qrs;
            if (t_1630 < clen) {
                {
                    auto _gst_329 = dq_base + qoff_1650;
                    __builtin_assume(__isGlobal(_gst_329));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_329) + (0)) = (__nv_bfloat16)dq_o_1647;
                }
                {
                    auto _gst_330 = dk_base + qoff_1650;
                    __builtin_assume(__isGlobal(_gst_330));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_330) + (0)) = (__nv_bfloat16)dk_o_1648;
                }
            }
            int t_1651 = 8 * th + 1;
            float r_1652[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1652[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + 128 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1653 = r_1652[0];
            unsigned int r_1654[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1654[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1655 = __uint_as_float(r_1654[0] << 16);
            unsigned int r_1656[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1656[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + 128 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + 128 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1657 = __uint_as_float(r_1656[0] << 16);
            float q_t_1658 = ((t_1651 >= clen) ? 0.0f : q_t_1657);
            float dqf_1659 = pool_f32[ew_1517 + 128 + d_lane];
            float dkf_1660 = pool_f32[ew_1517 + 1152 + d_lane];
            float dgf_1661 = pool_f32[ew_1517 + 2176 + d_lane];
            float bt_t_1662 = btv_1523[1];
            float e_off_1663 = approx_exp2(g_t_1653 - gn_1514);
            float e_dg_1664 = approx_exp2(g_t_1653 - gc_1515);
            float dq2_1665 = _tmem_load_26[1] * e_dg_1664;
            float dk2_1666 = _tmem_load_27[1] * e_dg_1664;
            dbp_1522[1] = dk2_1666 * k_t_1655;
            float dk2b_1667 = dk2_1666 * bt_t_1662;
            float dq_o_1668 = dq2_1665 + dqf_1659;
            float dk_o_1669 = dk2b_1667 + dkf_1660 + _tmem_load_28[1];
            float dg_o_1670 = __fmaf_rn(q_t_1658, dq2_1665, (dk2b_1667 - _tmem_load_28[1]) * k_t_1655) + dgf_1661;
            s_run_1524[0] = s_run_1524[0] + dg_o_1670;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th + 1)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1671 = t_1651 * qrs;
            if (t_1651 < clen) {
                {
                    auto _gst_331 = dq_base + qoff_1671;
                    __builtin_assume(__isGlobal(_gst_331));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_331) + (0)) = (__nv_bfloat16)dq_o_1668;
                }
                {
                    auto _gst_332 = dk_base + qoff_1671;
                    __builtin_assume(__isGlobal(_gst_332));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_332) + (0)) = (__nv_bfloat16)dk_o_1669;
                }
            }
            int t_1672 = 8 * th;
            float r_1673[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1673[0])) : "r"((gb8_1520 + (lane_1 / 32 * 8192 + lane_1 % 32 * 4 ^ (lane_1 / 32 * 8192 + lane_1 % 32 * 4 >> 7 & 7) << 4))));
            float g_t_1674 = r_1673[0];
            unsigned int r_1675[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1675[0])) : "r"((kb8_1521 + (dcol / 64 * 8192 + dcol % 64 * 2 ^ (dcol / 64 * 8192 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float k_t_1676 = __uint_as_float(r_1675[0] << 16);
            unsigned int r_1677[1];
            asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&r_1677[0])) : "r"((qbase_1518 + (dcol / 64 * 1024 + dcol % 64 * 2 ^ (dcol / 64 * 1024 + dcol % 64 * 2 >> 7 & 7) << 4))));
            float q_t_1678 = __uint_as_float(r_1677[0] << 16);
            float q_t_1679 = ((t_1672 >= clen) ? 0.0f : q_t_1678);
            float dqf_1680 = pool_f32[ew_1517 + d_lane];
            float dkf_1681 = pool_f32[ew_1517 + 1024 + d_lane];
            float dgf_1682 = pool_f32[ew_1517 + 2048 + d_lane];
            float bt_t_1683 = btv_1523[0];
            float e_off_1684 = approx_exp2(g_t_1674 - gn_1514);
            float e_dg_1685 = approx_exp2(g_t_1674 - gc_1515);
            float dq2_1686 = _tmem_load_26[0] * e_dg_1685;
            float dk2_1687 = _tmem_load_27[0] * e_dg_1685;
            dbp_1522[0] = dk2_1687 * k_t_1676;
            float dk2b_1688 = dk2_1687 * bt_t_1683;
            float dq_o_1689 = dq2_1686 + dqf_1680;
            float dk_o_1690 = dk2b_1688 + dkf_1681 + _tmem_load_28[0];
            float dg_o_1691 = __fmaf_rn(q_t_1679, dq2_1686, (dk2b_1688 - _tmem_load_28[0]) * k_t_1676) + dgf_1682;
            s_run_1524[0] = s_run_1524[0] + dg_o_1691;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th)), "f"(s_run_1524[0]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int qoff_1692 = t_1672 * qrs;
            if (t_1672 < clen) {
                {
                    auto _gst_333 = dq_base + qoff_1692;
                    __builtin_assume(__isGlobal(_gst_333));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_333) + (0)) = (__nv_bfloat16)dq_o_1689;
                }
                {
                    auto _gst_334 = dk_base + qoff_1692;
                    __builtin_assume(__isGlobal(_gst_334));
                    *(reinterpret_cast<__nv_bfloat16*>(_gst_334) + (0)) = (__nv_bfloat16)dk_o_1690;
                }
            }
            float tot_1693 = s_run_1524[0];
            int b4_1694 = lane_1 >> 4 & 1;
            int b3_1695 = lane_1 >> 3 & 1;
            int b2_1696 = lane_1 >> 2 & 1;
            float v4_1697[4];
            float send_1698 = ((b4_1694 == 0) ? dbp_1522[4] : dbp_1522[0]);
            float _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, send_1698, 16);
            v4_1697[0] = ((b4_1694 == 0) ? dbp_1522[0] : dbp_1522[4]) + _shfl_xor_sync_335;
            float send_1699 = ((b4_1694 == 0) ? dbp_1522[5] : dbp_1522[1]);
            float _shfl_xor_sync_336 = __shfl_xor_sync(0xFFFFFFFF, send_1699, 16);
            v4_1697[1] = ((b4_1694 == 0) ? dbp_1522[1] : dbp_1522[5]) + _shfl_xor_sync_336;
            float send_1700 = ((b4_1694 == 0) ? dbp_1522[6] : dbp_1522[2]);
            float _shfl_xor_sync_337 = __shfl_xor_sync(0xFFFFFFFF, send_1700, 16);
            v4_1697[2] = ((b4_1694 == 0) ? dbp_1522[2] : dbp_1522[6]) + _shfl_xor_sync_337;
            float send_1701 = ((b4_1694 == 0) ? dbp_1522[7] : dbp_1522[3]);
            float _shfl_xor_sync_338 = __shfl_xor_sync(0xFFFFFFFF, send_1701, 16);
            v4_1697[3] = ((b4_1694 == 0) ? dbp_1522[3] : dbp_1522[7]) + _shfl_xor_sync_338;
            float v2_1702[2];
            float send_1703 = ((b3_1695 == 0) ? v4_1697[2] : v4_1697[0]);
            float _shfl_xor_sync_339 = __shfl_xor_sync(0xFFFFFFFF, send_1703, 8);
            v2_1702[0] = ((b3_1695 == 0) ? v4_1697[0] : v4_1697[2]) + _shfl_xor_sync_339;
            float send_1704 = ((b3_1695 == 0) ? v4_1697[3] : v4_1697[1]);
            float _shfl_xor_sync_340 = __shfl_xor_sync(0xFFFFFFFF, send_1704, 8);
            v2_1702[1] = ((b3_1695 == 0) ? v4_1697[1] : v4_1697[3]) + _shfl_xor_sync_340;
            float send_1705 = ((b2_1696 == 0) ? v2_1702[1] : v2_1702[0]);
            float _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, send_1705, 4);
            float v1_1706 = ((b2_1696 == 0) ? v2_1702[0] : v2_1702[1]) + _shfl_xor_sync_341;
            float _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, v1_1706, 2);
            v1_1706 = v1_1706 + _shfl_xor_sync_342;
            float _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, v1_1706, 1);
            v1_1706 = v1_1706 + _shfl_xor_sync_343;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&off_acc[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1))));
            {
                uint32_t _addr_344 = static_cast<uint32_t>(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + th * 32 + lane_1)));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_344), "f"(tot_1693) : "memory");
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
            float t_other_1707[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_other_1707[0])) : "r"(pool_f32_addr + (unsigned int)(113664 + 4 * (qd * 64 + (1 - th) * 32 + lane_1))));
            float t_hi_1708 = ((th == 1) ? tot_1693 : t_other_1707[0]);
            float t_lo_1709 = ((th == 1) ? t_other_1707[0] : tot_1693);
            float offset_1710 = ((th == 1) ? off_acc[0] : off_acc[0] + t_hi_1708);
            float _tmem_load_29[8];
            tmem_ld_x8(&_tmem_load_29[0], taddr + (unsigned int)row_base + (unsigned int)(128 + 8 * th));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int t_1711 = 8 * th;
            int goff_1712 = t_1711 * rs_g;
            float dg_1713 = offset_1710 + _tmem_load_29[0];
            if (t_1711 < clen) {
                {
                    auto _gst_345 = dg_base + goff_1712;
                    __builtin_assume(__isGlobal(_gst_345));
                    *(reinterpret_cast<float*>(_gst_345) + (0)) = dg_1713;
                }
            }
            int t_1714 = 8 * th + 1;
            int goff_1715 = t_1714 * rs_g;
            float dg_1716 = offset_1710 + _tmem_load_29[1];
            if (t_1714 < clen) {
                {
                    auto _gst_346 = dg_base + goff_1715;
                    __builtin_assume(__isGlobal(_gst_346));
                    *(reinterpret_cast<float*>(_gst_346) + (0)) = dg_1716;
                }
            }
            int t_1717 = 8 * th + 2;
            int goff_1718 = t_1717 * rs_g;
            float dg_1719 = offset_1710 + _tmem_load_29[2];
            if (t_1717 < clen) {
                {
                    auto _gst_347 = dg_base + goff_1718;
                    __builtin_assume(__isGlobal(_gst_347));
                    *(reinterpret_cast<float*>(_gst_347) + (0)) = dg_1719;
                }
            }
            int t_1720 = 8 * th + 3;
            int goff_1721 = t_1720 * rs_g;
            float dg_1722 = offset_1710 + _tmem_load_29[3];
            if (t_1720 < clen) {
                {
                    auto _gst_348 = dg_base + goff_1721;
                    __builtin_assume(__isGlobal(_gst_348));
                    *(reinterpret_cast<float*>(_gst_348) + (0)) = dg_1722;
                }
            }
            int t_1723 = 8 * th + 4;
            int goff_1724 = t_1723 * rs_g;
            float dg_1725 = offset_1710 + _tmem_load_29[4];
            if (t_1723 < clen) {
                {
                    auto _gst_349 = dg_base + goff_1724;
                    __builtin_assume(__isGlobal(_gst_349));
                    *(reinterpret_cast<float*>(_gst_349) + (0)) = dg_1725;
                }
            }
            int t_1726 = 8 * th + 5;
            int goff_1727 = t_1726 * rs_g;
            float dg_1728 = offset_1710 + _tmem_load_29[5];
            if (t_1726 < clen) {
                {
                    auto _gst_350 = dg_base + goff_1727;
                    __builtin_assume(__isGlobal(_gst_350));
                    *(reinterpret_cast<float*>(_gst_350) + (0)) = dg_1728;
                }
            }
            int t_1729 = 8 * th + 6;
            int goff_1730 = t_1729 * rs_g;
            float dg_1731 = offset_1710 + _tmem_load_29[6];
            if (t_1729 < clen) {
                {
                    auto _gst_351 = dg_base + goff_1730;
                    __builtin_assume(__isGlobal(_gst_351));
                    *(reinterpret_cast<float*>(_gst_351) + (0)) = dg_1731;
                }
            }
            int t_1732 = 8 * th + 7;
            int goff_1733 = t_1732 * rs_g;
            float dg_1734 = offset_1710 + _tmem_load_29[7];
            if (t_1732 < clen) {
                {
                    auto _gst_352 = dg_base + goff_1733;
                    __builtin_assume(__isGlobal(_gst_352));
                    *(reinterpret_cast<float*>(_gst_352) + (0)) = dg_1734;
                }
            }
            off_acc[0] = off_acc[0] + t_hi_1708 + t_lo_1709;
            if ((lane_1 & 3) == 0) {
                int e_star = lane_1 >> 2 & 7;
                pool_f32[28160 + qd * 64 + 8 * th + e_star] = v1_1706;
                pool_f32[28160 + qd * 64 + 16 + 8 * th + e_star] = v1_1484;
                pool_f32[28160 + qd * 64 + 32 + 8 * th + e_star] = v1;
                pool_f32[28160 + qd * 64 + 48 + 8 * th + e_star] = v1_1241;
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
                    for (int r_12 = 0; r_12 < 16; r_12++) {
                        int t_2 = 16 + r_12;
                        if (t_2 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 16)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 32)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 48)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 64)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 80)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 96)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_2 * 128 + 112)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_12 * 128)), "f"(z_1[0]), "f"(z_1[1]), "f"(z_1[2]), "f"(z_1[3]) : "memory");
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
                    for (int r_14 = 0; r_14 < 16; r_14++) {
                        int t_3 = 32 + r_14;
                        if (t_3 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98304 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98320 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98336 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98352 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98368 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98384 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98400 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(98416 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100352 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 16)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100368 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 32)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100384 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 48)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100400 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 64)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100416 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 80)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100432 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 96)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100448 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_3 * 128 + 112)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(100464 + r_14 * 128)), "f"(z_2[0]), "f"(z_2[1]), "f"(z_2[2]), "f"(z_2[3]) : "memory");
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
                    for (int r_15 = 0; r_15 < 16; r_15++) {
                        int t_4 = 48 + r_15;
                        if (t_4 >= clen_setup) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102400 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102416 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102432 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102448 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102464 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102480 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102496 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(32768 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(102512 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104448 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 16)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104464 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 32)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104480 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 48)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104496 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 64)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104512 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 80)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104528 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 96)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104544 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(40960 + t_4 * 128 + 112)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pool_f32_addr + (unsigned int)(104560 + r_15 * 128)), "f"(z_3[0]), "f"(z_3[1]), "f"(z_3[2]), "f"(z_3[3]) : "memory");
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
