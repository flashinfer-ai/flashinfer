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
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_CLC_PIPELINE_STAGES 1
#define SMEM_QKO_OFF 1024
#define SMEM_QKO_STAGE_BYTES 65536
#define SMEM_QKO_STRIDE 65536
#define SMEM_K_DUAL_OFF 1024
#define SMEM_K_DUAL_STAGE_BYTES 65536
#define SMEM_K_DUAL_STRIDE 65536
#define SMEM_V_OFF 1024
#define SMEM_V_STAGE_BYTES 65536
#define SMEM_V_STRIDE 65536
#define SMEM_ROPE_OFF 197632
#define SMEM_ROPE_STAGE_BYTES 8192
#define SMEM_ROPE_STRIDE 8192
#define SMEM_EXCHANGE_OFF 205824
#define SMEM_EXCHANGE_STAGE_BYTES 16384
#define SMEM_EXCHANGE_STRIDE 16384
#define SMEM_PROBABILITY_OFF 222208
#define SMEM_PROBABILITY_STAGE_BYTES 8192
#define SMEM_PROBABILITY_STRIDE 8192
#define SMEM_VALIDITY_OFF 230400
#define SMEM_VALIDITY_STAGE_BYTES 24
#define SMEM_VALIDITY_STRIDE 24
#define SMEM_VALIDITY32_OFF 230400
#define SMEM_VALIDITY32_STAGE_BYTES 24
#define SMEM_VALIDITY32_STRIDE 24
#define SMEM_STATS_OFF 230528
#define SMEM_STATS_STAGE_BYTES 1024
#define SMEM_STATS_STRIDE 1024
#define SMEM_CLC_PAYLOAD_OFF 231552
#define SMEM_CLC_PAYLOAD_STAGE_BYTES 16
#define SMEM_CLC_PAYLOAD_STRIDE 16
#define SMEM_LEN_CELLS_OFF 231600
#define SMEM_LEN_CELLS_STAGE_BYTES 8
#define SMEM_LEN_CELLS_STRIDE 8
#define SMEM_TOTAL 231680
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

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



__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)







__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_train_257d4674f88939dbe09d(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap kv_latent, __nv_bfloat16* __restrict__ k_rope, const __grid_constant__ CUtensorMap out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int k_rope_stride, int k_rope_offset, int has_topk_length, int derive_length, float scale_log2)
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

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_copied_addr (mbar_base + 8)
    #define clc_full_addr (mbar_base + 16)
    #define clc_empty_addr (mbar_base + 24)
    #define len_ready_addr (mbar_base + 32)
    #define qk_done_addr (mbar_base + 48)
    #define sv_done_addr (mbar_base + 72)
    #define kv_ready_addr (mbar_base + 96)
    #define valid_ready_addr (mbar_base + 144)
    #define valid_free_addr (mbar_base + 168)
    #define p_free_addr (mbar_base + 192)
    #define so_ready_addr (mbar_base + 200)
    #define o_written_addr (mbar_base + 208)
    #define o_written_waited_addr (mbar_base + 216)
    #define qr_full_addr (mbar_base + 224)
    #define qr_copied_addr (mbar_base + 232)
    #define qr_done_addr (mbar_base + 240)
    #define kr_ready_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* qko = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int qko_addr = smem + 1024;
    __nv_bfloat16* k_dual = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int k_dual_addr = smem + 1024;
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int v_addr = smem + 1024;
    __nv_bfloat16* rope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int rope_addr = smem + 197632;
    float* exchange = reinterpret_cast<float*>(smem_raw + 205824);
    const int exchange_addr = smem + 205824;
    __nv_bfloat16* probability = reinterpret_cast<__nv_bfloat16*>(smem_raw + 222208);
    const int probability_addr = smem + 222208;
    uint8_t* validity = reinterpret_cast<uint8_t*>(smem_raw + 230400);
    const int validity_addr = smem + 230400;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int validity32_addr = smem + 230400;
    float* stats = reinterpret_cast<float*>(smem_raw + 230528);
    const int stats_addr = smem + 230528;
    unsigned int* clc_payload = reinterpret_cast<unsigned int*>(smem_raw + 231552);
    const int clc_payload_addr = smem + 231552;
    int* len_cells = reinterpret_cast<int*>(smem_raw + 231600);
    const int len_cells_addr = smem + 231600;
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_copied: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'clc_pipeline' ---
            // clc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // clc_empty: 1 barriers, init_count=346
            mbarrier_init(smem + 24, 346);
            // qr_full: 1 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            // qr_copied: 1 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // len_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // qk_done: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sv_done: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // kv_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // valid_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            mbarrier_init(smem + 160, 8);
            // valid_free: 3 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // p_free: 1 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            // so_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // o_written: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // o_written_waited: 1 barriers, init_count=4
            mbarrier_init(smem + 216, 4);
            // qr_done: 1 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            // kr_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 248, 64);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 2) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: softmax ----
    if (warp <= 3) {
        { // softmax_main
            int query = blockIdx.x;
            unsigned int outer_phase = 0;
            int cursor = 0;
            #pragma unroll 1
            for (int outer = 0; outer < num_queries; outer++) {
                int active = topk;
                if (derive_length != 0) {
                    int len_slot = outer & 1;
                    unsigned int len_phase = outer >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot) * 8, len_phase);
                    int _max_0 = ((len_cells[len_slot]) > (0) ? (len_cells[len_slot]) : (0));
                    int _min_0 = ((_max_0) < (topk) ? (_max_0) : (topk));
                    active = _min_0;
                } else if (has_topk_length != 0) {
                    int _max_1 = ((topk_length[query]) > (0) ? (topk_length[query]) : (0));
                    int _min_1 = ((_max_1) < (topk) ? (_max_1) : (topk));
                    active = _min_1;
                }
                int active_0 = active;
                int _max_2 = (((active_0 + 63) / 64) > (2) ? ((active_0 + 63) / 64) : (2));
                int blocks = _max_2;
                int row = warp * 32 + lane;
                unsigned int tmrow = tmem_tmem;
                float mi = -1e+30f;
                float li = 0.0f;
                float real_mi = -CAKE_INF;
                asm volatile("cp.async.bulk.wait_group 0;");
                mbarrier_arrive(o_written_addr);
                #pragma unroll 1
                for (int block = 0; block < blocks; block++) {
                    int slot = cursor % 3;
                    unsigned int phase = cursor / 3 & 1;
                    asm volatile("barrier.sync %0, 64;" :: "r"(1 + (warp & 1)) : "memory");
                    mbarrier_wait(qk_done_addr + (slot) * 8, phase);
                    mbarrier_wait(valid_ready_addr + (slot) * 8, phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float p[32];
                    float peer[32];
                    if (warp < 2) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31])
                            : "r"(tmrow + 400));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 432));
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 400));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31])
                            : "r"(tmrow + 432));
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_free_addr);
                    unsigned int p_mask = validity32[slot * 2 + warp / 2];
                    if ((p_mask & 1) == 0) {
                        p[0] = -CAKE_INF;
                    }
                    if ((p_mask >> 1 & 1) == 0) {
                        p[1] = -CAKE_INF;
                    }
                    if ((p_mask >> 2 & 1) == 0) {
                        p[2] = -CAKE_INF;
                    }
                    if ((p_mask >> 3 & 1) == 0) {
                        p[3] = -CAKE_INF;
                    }
                    if ((p_mask >> 4 & 1) == 0) {
                        p[4] = -CAKE_INF;
                    }
                    if ((p_mask >> 5 & 1) == 0) {
                        p[5] = -CAKE_INF;
                    }
                    if ((p_mask >> 6 & 1) == 0) {
                        p[6] = -CAKE_INF;
                    }
                    if ((p_mask >> 7 & 1) == 0) {
                        p[7] = -CAKE_INF;
                    }
                    if ((p_mask >> 8 & 1) == 0) {
                        p[8] = -CAKE_INF;
                    }
                    if ((p_mask >> 9 & 1) == 0) {
                        p[9] = -CAKE_INF;
                    }
                    if ((p_mask >> 10 & 1) == 0) {
                        p[10] = -CAKE_INF;
                    }
                    if ((p_mask >> 11 & 1) == 0) {
                        p[11] = -CAKE_INF;
                    }
                    if ((p_mask >> 12 & 1) == 0) {
                        p[12] = -CAKE_INF;
                    }
                    if ((p_mask >> 13 & 1) == 0) {
                        p[13] = -CAKE_INF;
                    }
                    if ((p_mask >> 14 & 1) == 0) {
                        p[14] = -CAKE_INF;
                    }
                    if ((p_mask >> 15 & 1) == 0) {
                        p[15] = -CAKE_INF;
                    }
                    if ((p_mask >> 16 & 1) == 0) {
                        p[16] = -CAKE_INF;
                    }
                    if ((p_mask >> 17 & 1) == 0) {
                        p[17] = -CAKE_INF;
                    }
                    if ((p_mask >> 18 & 1) == 0) {
                        p[18] = -CAKE_INF;
                    }
                    if ((p_mask >> 19 & 1) == 0) {
                        p[19] = -CAKE_INF;
                    }
                    if ((p_mask >> 20 & 1) == 0) {
                        p[20] = -CAKE_INF;
                    }
                    if ((p_mask >> 21 & 1) == 0) {
                        p[21] = -CAKE_INF;
                    }
                    if ((p_mask >> 22 & 1) == 0) {
                        p[22] = -CAKE_INF;
                    }
                    if ((p_mask >> 23 & 1) == 0) {
                        p[23] = -CAKE_INF;
                    }
                    if ((p_mask >> 24 & 1) == 0) {
                        p[24] = -CAKE_INF;
                    }
                    if ((p_mask >> 25 & 1) == 0) {
                        p[25] = -CAKE_INF;
                    }
                    if ((p_mask >> 26 & 1) == 0) {
                        p[26] = -CAKE_INF;
                    }
                    if ((p_mask >> 27 & 1) == 0) {
                        p[27] = -CAKE_INF;
                    }
                    if ((p_mask >> 28 & 1) == 0) {
                        p[28] = -CAKE_INF;
                    }
                    if ((p_mask >> 29 & 1) == 0) {
                        p[29] = -CAKE_INF;
                    }
                    if ((p_mask >> 30 & 1) == 0) {
                        p[30] = -CAKE_INF;
                    }
                    if ((p_mask >> 31 & 1) == 0) {
                        p[31] = -CAKE_INF;
                    }
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + lane * 4) * 4)), "f"(peer[0]), "f"(peer[1]), "f"(peer[2]), "f"(peer[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 128 + lane * 4) * 4)), "f"(peer[4]), "f"(peer[5]), "f"(peer[6]), "f"(peer[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 256 + lane * 4) * 4)), "f"(peer[8]), "f"(peer[9]), "f"(peer[10]), "f"(peer[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 384 + lane * 4) * 4)), "f"(peer[12]), "f"(peer[13]), "f"(peer[14]), "f"(peer[15]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 512 + lane * 4) * 4)), "f"(peer[16]), "f"(peer[17]), "f"(peer[18]), "f"(peer[19]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 640 + lane * 4) * 4)), "f"(peer[20]), "f"(peer[21]), "f"(peer[22]), "f"(peer[23]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 768 + lane * 4) * 4)), "f"(peer[24]), "f"(peer[25]), "f"(peer[26]), "f"(peer[27]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 896 + lane * 4) * 4)), "f"(peer[28]), "f"(peer[29]), "f"(peer[30]), "f"(peer[31]) : "memory");
                    asm volatile("barrier.sync %0, 64;" :: "r"(1 + (warp & 1)) : "memory");
                    float _exchange_reg_0[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_0[_lr] = _smem_ptr[(warp * 1024 + lane * 4) + _lr];
                    }
                    float2 _f2_0 = make_float2(p[0], p[1]);
                    float2 _f2_1 = make_float2(_exchange_reg_0[0], _exchange_reg_0[1]);
                    float2 pair_sum = add_f32x2(_f2_0, _f2_1);
                    p[0] = pair_sum.x;
                    p[1] = pair_sum.y;
                    float2 _f2_2 = make_float2(p[2], p[3]);
                    float2 _f2_3 = make_float2(_exchange_reg_0[2], _exchange_reg_0[3]);
                    float2 pair_sum_0 = add_f32x2(_f2_2, _f2_3);
                    p[2] = pair_sum_0.x;
                    p[3] = pair_sum_0.y;
                    float _exchange_reg_1[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_1[_lr] = _smem_ptr[(warp * 1024 + 128 + lane * 4) + _lr];
                    }
                    float2 _f2_4 = make_float2(p[4], p[5]);
                    float2 _f2_5 = make_float2(_exchange_reg_1[0], _exchange_reg_1[1]);
                    float2 pair_sum_1 = add_f32x2(_f2_4, _f2_5);
                    p[4] = pair_sum_1.x;
                    p[5] = pair_sum_1.y;
                    float2 _f2_6 = make_float2(p[6], p[7]);
                    float2 _f2_7 = make_float2(_exchange_reg_1[2], _exchange_reg_1[3]);
                    float2 pair_sum_2 = add_f32x2(_f2_6, _f2_7);
                    p[6] = pair_sum_2.x;
                    p[7] = pair_sum_2.y;
                    float _exchange_reg_2[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_2[_lr] = _smem_ptr[(warp * 1024 + 256 + lane * 4) + _lr];
                    }
                    float2 _f2_8 = make_float2(p[8], p[9]);
                    float2 _f2_9 = make_float2(_exchange_reg_2[0], _exchange_reg_2[1]);
                    float2 pair_sum_3 = add_f32x2(_f2_8, _f2_9);
                    p[8] = pair_sum_3.x;
                    p[9] = pair_sum_3.y;
                    float2 _f2_10 = make_float2(p[10], p[11]);
                    float2 _f2_11 = make_float2(_exchange_reg_2[2], _exchange_reg_2[3]);
                    float2 pair_sum_4 = add_f32x2(_f2_10, _f2_11);
                    p[10] = pair_sum_4.x;
                    p[11] = pair_sum_4.y;
                    float _exchange_reg_3[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_3[_lr] = _smem_ptr[(warp * 1024 + 384 + lane * 4) + _lr];
                    }
                    float2 _f2_12 = make_float2(p[12], p[13]);
                    float2 _f2_13 = make_float2(_exchange_reg_3[0], _exchange_reg_3[1]);
                    float2 pair_sum_5 = add_f32x2(_f2_12, _f2_13);
                    p[12] = pair_sum_5.x;
                    p[13] = pair_sum_5.y;
                    float2 _f2_14 = make_float2(p[14], p[15]);
                    float2 _f2_15 = make_float2(_exchange_reg_3[2], _exchange_reg_3[3]);
                    float2 pair_sum_6 = add_f32x2(_f2_14, _f2_15);
                    p[14] = pair_sum_6.x;
                    p[15] = pair_sum_6.y;
                    float _exchange_reg_4[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_4[_lr] = _smem_ptr[(warp * 1024 + 512 + lane * 4) + _lr];
                    }
                    float2 _f2_16 = make_float2(p[16], p[17]);
                    float2 _f2_17 = make_float2(_exchange_reg_4[0], _exchange_reg_4[1]);
                    float2 pair_sum_7 = add_f32x2(_f2_16, _f2_17);
                    p[16] = pair_sum_7.x;
                    p[17] = pair_sum_7.y;
                    float2 _f2_18 = make_float2(p[18], p[19]);
                    float2 _f2_19 = make_float2(_exchange_reg_4[2], _exchange_reg_4[3]);
                    float2 pair_sum_8 = add_f32x2(_f2_18, _f2_19);
                    p[18] = pair_sum_8.x;
                    p[19] = pair_sum_8.y;
                    float _exchange_reg_5[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_5[_lr] = _smem_ptr[(warp * 1024 + 640 + lane * 4) + _lr];
                    }
                    float2 _f2_20 = make_float2(p[20], p[21]);
                    float2 _f2_21 = make_float2(_exchange_reg_5[0], _exchange_reg_5[1]);
                    float2 pair_sum_9 = add_f32x2(_f2_20, _f2_21);
                    p[20] = pair_sum_9.x;
                    p[21] = pair_sum_9.y;
                    float2 _f2_22 = make_float2(p[22], p[23]);
                    float2 _f2_23 = make_float2(_exchange_reg_5[2], _exchange_reg_5[3]);
                    float2 pair_sum_10 = add_f32x2(_f2_22, _f2_23);
                    p[22] = pair_sum_10.x;
                    p[23] = pair_sum_10.y;
                    float _exchange_reg_6[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_6[_lr] = _smem_ptr[(warp * 1024 + 768 + lane * 4) + _lr];
                    }
                    float2 _f2_24 = make_float2(p[24], p[25]);
                    float2 _f2_25 = make_float2(_exchange_reg_6[0], _exchange_reg_6[1]);
                    float2 pair_sum_11 = add_f32x2(_f2_24, _f2_25);
                    p[24] = pair_sum_11.x;
                    p[25] = pair_sum_11.y;
                    float2 _f2_26 = make_float2(p[26], p[27]);
                    float2 _f2_27 = make_float2(_exchange_reg_6[2], _exchange_reg_6[3]);
                    float2 pair_sum_12 = add_f32x2(_f2_26, _f2_27);
                    p[26] = pair_sum_12.x;
                    p[27] = pair_sum_12.y;
                    float _exchange_reg_7[4];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(exchange);
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++)
                            _exchange_reg_7[_lr] = _smem_ptr[(warp * 1024 + 896 + lane * 4) + _lr];
                    }
                    float2 _f2_28 = make_float2(p[28], p[29]);
                    float2 _f2_29 = make_float2(_exchange_reg_7[0], _exchange_reg_7[1]);
                    float2 pair_sum_13 = add_f32x2(_f2_28, _f2_29);
                    p[28] = pair_sum_13.x;
                    p[29] = pair_sum_13.y;
                    float2 _f2_30 = make_float2(p[30], p[31]);
                    float2 _f2_31 = make_float2(_exchange_reg_7[2], _exchange_reg_7[3]);
                    float2 pair_sum_14 = add_f32x2(_f2_30, _f2_31);
                    p[30] = pair_sum_14.x;
                    p[31] = pair_sum_14.y;
                    mbarrier_arrive(valid_free_addr + (slot) * 8);
                    float local_max0 = -CAKE_INF;
                    float local_max1 = -CAKE_INF;
                    float _max_3 = max_noftz(local_max0, p[0]);
                    local_max0 = _max_3;
                    float _max_4 = max_noftz(local_max0, p[1]);
                    local_max0 = _max_4;
                    float _max_5 = max_noftz(local_max0, p[2]);
                    local_max0 = _max_5;
                    float _max_6 = max_noftz(local_max0, p[3]);
                    local_max0 = _max_6;
                    float _max_7 = max_noftz(local_max0, p[4]);
                    local_max0 = _max_7;
                    float _max_8 = max_noftz(local_max0, p[5]);
                    local_max0 = _max_8;
                    float _max_9 = max_noftz(local_max0, p[6]);
                    local_max0 = _max_9;
                    float _max_10 = max_noftz(local_max0, p[7]);
                    local_max0 = _max_10;
                    float _max_11 = max_noftz(local_max0, p[8]);
                    local_max0 = _max_11;
                    float _max_12 = max_noftz(local_max0, p[9]);
                    local_max0 = _max_12;
                    float _max_13 = max_noftz(local_max0, p[10]);
                    local_max0 = _max_13;
                    float _max_14 = max_noftz(local_max0, p[11]);
                    local_max0 = _max_14;
                    float _max_15 = max_noftz(local_max0, p[12]);
                    local_max0 = _max_15;
                    float _max_16 = max_noftz(local_max0, p[13]);
                    local_max0 = _max_16;
                    float _max_17 = max_noftz(local_max0, p[14]);
                    local_max0 = _max_17;
                    float _max_18 = max_noftz(local_max0, p[15]);
                    local_max0 = _max_18;
                    float _max_19 = max_noftz(local_max1, p[16]);
                    local_max1 = _max_19;
                    float _max_20 = max_noftz(local_max1, p[17]);
                    local_max1 = _max_20;
                    float _max_21 = max_noftz(local_max1, p[18]);
                    local_max1 = _max_21;
                    float _max_22 = max_noftz(local_max1, p[19]);
                    local_max1 = _max_22;
                    float _max_23 = max_noftz(local_max1, p[20]);
                    local_max1 = _max_23;
                    float _max_24 = max_noftz(local_max1, p[21]);
                    local_max1 = _max_24;
                    float _max_25 = max_noftz(local_max1, p[22]);
                    local_max1 = _max_25;
                    float _max_26 = max_noftz(local_max1, p[23]);
                    local_max1 = _max_26;
                    float _max_27 = max_noftz(local_max1, p[24]);
                    local_max1 = _max_27;
                    float _max_28 = max_noftz(local_max1, p[25]);
                    local_max1 = _max_28;
                    float _max_29 = max_noftz(local_max1, p[26]);
                    local_max1 = _max_29;
                    float _max_30 = max_noftz(local_max1, p[27]);
                    local_max1 = _max_30;
                    float _max_31 = max_noftz(local_max1, p[28]);
                    local_max1 = _max_31;
                    float _max_32 = max_noftz(local_max1, p[29]);
                    local_max1 = _max_32;
                    float _max_33 = max_noftz(local_max1, p[30]);
                    local_max1 = _max_33;
                    float _max_34 = max_noftz(local_max1, p[31]);
                    local_max1 = _max_34;
                    float _max_35 = max_noftz(local_max0, local_max1);
                    float cur_max = _max_35 * scale_log2;
                    int rescale = 0;
                    stats[row] = cur_max;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    float _max_36 = max_noftz(cur_max, stats[row ^ 64]);
                    cur_max = _max_36;
                    float _max_37 = max_noftz(real_mi, cur_max);
                    real_mi = _max_37;
                    int _vote_0 = __any_sync(0xFFFFFFFF, cur_max - mi > 6.0f);
                    rescale = _vote_0;
                    float factor = 1.0f;
                    float next_max = mi;
                    if (rescale != 0) {
                        float _max_38 = max_noftz(cur_max, mi);
                        next_max = _max_38;
                        float _exp2_0 = approx_exp2(mi - next_max);
                        factor = _exp2_0;
                    }
                    mi = next_max;
                    const float2 _fma_b2_0 = {scale_log2, scale_log2};
                    const float2 _fma_c2_1 = {-mi, -mi};
                    float2 _fma_pair_2 = fma_f32x2(make_float2(p[0], p[1]), _fma_b2_0, _fma_c2_1);
                    p[0] = _fma_pair_2.x;
                    p[1] = _fma_pair_2.y;
                    float2 _fma_pair_3 = fma_f32x2(make_float2(p[2], p[3]), _fma_b2_0, _fma_c2_1);
                    p[2] = _fma_pair_3.x;
                    p[3] = _fma_pair_3.y;
                    float2 _fma_pair_4 = fma_f32x2(make_float2(p[4], p[5]), _fma_b2_0, _fma_c2_1);
                    p[4] = _fma_pair_4.x;
                    p[5] = _fma_pair_4.y;
                    float2 _fma_pair_5 = fma_f32x2(make_float2(p[6], p[7]), _fma_b2_0, _fma_c2_1);
                    p[6] = _fma_pair_5.x;
                    p[7] = _fma_pair_5.y;
                    float2 _fma_pair_6 = fma_f32x2(make_float2(p[8], p[9]), _fma_b2_0, _fma_c2_1);
                    p[8] = _fma_pair_6.x;
                    p[9] = _fma_pair_6.y;
                    float2 _fma_pair_7 = fma_f32x2(make_float2(p[10], p[11]), _fma_b2_0, _fma_c2_1);
                    p[10] = _fma_pair_7.x;
                    p[11] = _fma_pair_7.y;
                    float2 _fma_pair_8 = fma_f32x2(make_float2(p[12], p[13]), _fma_b2_0, _fma_c2_1);
                    p[12] = _fma_pair_8.x;
                    p[13] = _fma_pair_8.y;
                    float2 _fma_pair_9 = fma_f32x2(make_float2(p[14], p[15]), _fma_b2_0, _fma_c2_1);
                    p[14] = _fma_pair_9.x;
                    p[15] = _fma_pair_9.y;
                    float2 _fma_pair_10 = fma_f32x2(make_float2(p[16], p[17]), _fma_b2_0, _fma_c2_1);
                    p[16] = _fma_pair_10.x;
                    p[17] = _fma_pair_10.y;
                    float2 _fma_pair_11 = fma_f32x2(make_float2(p[18], p[19]), _fma_b2_0, _fma_c2_1);
                    p[18] = _fma_pair_11.x;
                    p[19] = _fma_pair_11.y;
                    float2 _fma_pair_12 = fma_f32x2(make_float2(p[20], p[21]), _fma_b2_0, _fma_c2_1);
                    p[20] = _fma_pair_12.x;
                    p[21] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(p[22], p[23]), _fma_b2_0, _fma_c2_1);
                    p[22] = _fma_pair_13.x;
                    p[23] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(p[24], p[25]), _fma_b2_0, _fma_c2_1);
                    p[24] = _fma_pair_14.x;
                    p[25] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(p[26], p[27]), _fma_b2_0, _fma_c2_1);
                    p[26] = _fma_pair_15.x;
                    p[27] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(p[28], p[29]), _fma_b2_0, _fma_c2_1);
                    p[28] = _fma_pair_16.x;
                    p[29] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(p[30], p[31]), _fma_b2_0, _fma_c2_1);
                    p[30] = _fma_pair_17.x;
                    p[31] = _fma_pair_17.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        p[_le] = approx_exp2(p[_le]);
                    }
                    float2 _f2_32 = make_float2(0.0f, 0.0f);
                    float2 sum_pair = _f2_32;
                    float2 _f2_33 = make_float2(p[0], p[1]);
                    sum_pair = add_f32x2(sum_pair, _f2_33);
                    float2 _f2_34 = make_float2(p[2], p[3]);
                    sum_pair = add_f32x2(sum_pair, _f2_34);
                    float2 _f2_35 = make_float2(p[4], p[5]);
                    sum_pair = add_f32x2(sum_pair, _f2_35);
                    float2 _f2_36 = make_float2(p[6], p[7]);
                    sum_pair = add_f32x2(sum_pair, _f2_36);
                    float2 _f2_37 = make_float2(p[8], p[9]);
                    sum_pair = add_f32x2(sum_pair, _f2_37);
                    float2 _f2_38 = make_float2(p[10], p[11]);
                    sum_pair = add_f32x2(sum_pair, _f2_38);
                    float2 _f2_39 = make_float2(p[12], p[13]);
                    sum_pair = add_f32x2(sum_pair, _f2_39);
                    float2 _f2_40 = make_float2(p[14], p[15]);
                    sum_pair = add_f32x2(sum_pair, _f2_40);
                    float2 _f2_41 = make_float2(p[16], p[17]);
                    sum_pair = add_f32x2(sum_pair, _f2_41);
                    float2 _f2_42 = make_float2(p[18], p[19]);
                    sum_pair = add_f32x2(sum_pair, _f2_42);
                    float2 _f2_43 = make_float2(p[20], p[21]);
                    sum_pair = add_f32x2(sum_pair, _f2_43);
                    float2 _f2_44 = make_float2(p[22], p[23]);
                    sum_pair = add_f32x2(sum_pair, _f2_44);
                    float2 _f2_45 = make_float2(p[24], p[25]);
                    sum_pair = add_f32x2(sum_pair, _f2_45);
                    float2 _f2_46 = make_float2(p[26], p[27]);
                    sum_pair = add_f32x2(sum_pair, _f2_46);
                    float2 _f2_47 = make_float2(p[28], p[29]);
                    sum_pair = add_f32x2(sum_pair, _f2_47);
                    float2 _f2_48 = make_float2(p[30], p[31]);
                    sum_pair = add_f32x2(sum_pair, _f2_48);
                    float cur_sum = sum_pair.x + sum_pair.y;
                    float _fma_0 = __fmaf_rn(li, factor, cur_sum);
                    li = _fma_0;
                    if (block > 0) {
                        mbarrier_wait(sv_done_addr + ((cursor - 1) % 3) * 8, (cursor - 1) / 3 & 1);
                    }
                    uint32_t p_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                        p_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048) * 2)), "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]) : "memory");
                    uint32_t p_bf16_15[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 8], p[_lp*2+1 + 8]));
                        p_bf16_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 512) * 2)), "r"(p_bf16_15[0]), "r"(p_bf16_15[1]), "r"(p_bf16_15[2]), "r"(p_bf16_15[3]) : "memory");
                    uint32_t p_bf16_16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 16], p[_lp*2+1 + 16]));
                        p_bf16_16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1024) * 2)), "r"(p_bf16_16[0]), "r"(p_bf16_16[1]), "r"(p_bf16_16[2]), "r"(p_bf16_16[3]) : "memory");
                    uint32_t p_bf16_17[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 24], p[_lp*2+1 + 24]));
                        p_bf16_17[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1536) * 2)), "r"(p_bf16_17[0]), "r"(p_bf16_17[1]), "r"(p_bf16_17[2]), "r"(p_bf16_17[3]) : "memory");
                    if (block > 0 && rescale != 0) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(tmrow));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_18 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_18);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_0[_ls] = _tmem_load_0[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow, _tmem_load_0);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(tmrow + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_19 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_19);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_1[_ls] = _tmem_load_1[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 32, _tmem_load_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(tmrow + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_20 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_20);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_2[_ls] = _tmem_load_2[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 64, _tmem_load_2);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(tmrow + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_21 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_21);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_3[_ls] = _tmem_load_3[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 96, _tmem_load_3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                            : "r"(tmrow + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_22 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_22);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_4[_ls] = _tmem_load_4[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 128, _tmem_load_4);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(tmrow + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_23 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_23);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_5[_ls] = _tmem_load_5[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 160, _tmem_load_5);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(tmrow + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_24 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_24);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_6[_ls] = _tmem_load_6[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 192, _tmem_load_6);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                            : "r"(tmrow + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_25 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_25);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_7[_ls] = _tmem_load_7[_ls] * factor;
                        }
                        #endif
                        tmem_st_x32_f32(tmrow + 224, _tmem_load_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(so_ready_addr);
                    cursor = cursor + 1;
                }
                mbarrier_wait(o_written_waited_addr, outer_phase);
                if (real_mi == -CAKE_INF) {
                    li = 0.0f;
                    mi = -CAKE_INF;
                }
                stats[128 + row] = li;
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                li = li + stats[128 + (row ^ 64)];
                if (row < 64) {
                    float _log_0 = logf(li);
                    float _fma_1 = __fmaf_rn(mi, 0.6931471805599453f, _log_0);
                    float cur_lse = _fma_1;
                    if (real_mi == -CAKE_INF) {
                        cur_lse = -CAKE_INF;
                    }
                    lse[query * 64 + row] = cur_lse;
                }
                float output_scale = 1.0f / li;
                int _vote_1 = __any_sync(0xFFFFFFFF, li != 0.0f);
                int have_valid = _vote_1;
                if (have_valid == 0) {
                    output_scale = 1.0f;
                }
                mbarrier_wait(sv_done_addr + ((cursor - 1) % 3) * 8, (cursor - 1) / 3 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int o_slot = (cursor + 2) % 3;
                int head = row % 64;
                int query_head = query * 64 + head;
                long long o_row_base = (long long)query_head * 512;
                float ov[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31]), "=f"(ov[32]), "=f"(ov[33]), "=f"(ov[34]), "=f"(ov[35]), "=f"(ov[36]), "=f"(ov[37]), "=f"(ov[38]), "=f"(ov[39]), "=f"(ov[40]), "=f"(ov[41]), "=f"(ov[42]), "=f"(ov[43]), "=f"(ov[44]), "=f"(ov[45]), "=f"(ov[46]), "=f"(ov[47]), "=f"(ov[48]), "=f"(ov[49]), "=f"(ov[50]), "=f"(ov[51]), "=f"(ov[52]), "=f"(ov[53]), "=f"(ov[54]), "=f"(ov[55]), "=f"(ov[56]), "=f"(ov[57]), "=f"(ov[58]), "=f"(ov[59]), "=f"(ov[60]), "=f"(ov[61]), "=f"(ov[62]), "=f"(ov[63])
                        : "r"(tmrow));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov[0] = 0.0f;
                    ov[1] = 0.0f;
                    ov[2] = 0.0f;
                    ov[3] = 0.0f;
                    ov[4] = 0.0f;
                    ov[5] = 0.0f;
                    ov[6] = 0.0f;
                    ov[7] = 0.0f;
                    ov[8] = 0.0f;
                    ov[9] = 0.0f;
                    ov[10] = 0.0f;
                    ov[11] = 0.0f;
                    ov[12] = 0.0f;
                    ov[13] = 0.0f;
                    ov[14] = 0.0f;
                    ov[15] = 0.0f;
                    ov[16] = 0.0f;
                    ov[17] = 0.0f;
                    ov[18] = 0.0f;
                    ov[19] = 0.0f;
                    ov[20] = 0.0f;
                    ov[21] = 0.0f;
                    ov[22] = 0.0f;
                    ov[23] = 0.0f;
                    ov[24] = 0.0f;
                    ov[25] = 0.0f;
                    ov[26] = 0.0f;
                    ov[27] = 0.0f;
                    ov[28] = 0.0f;
                    ov[29] = 0.0f;
                    ov[30] = 0.0f;
                    ov[31] = 0.0f;
                    ov[32] = 0.0f;
                    ov[33] = 0.0f;
                    ov[34] = 0.0f;
                    ov[35] = 0.0f;
                    ov[36] = 0.0f;
                    ov[37] = 0.0f;
                    ov[38] = 0.0f;
                    ov[39] = 0.0f;
                    ov[40] = 0.0f;
                    ov[41] = 0.0f;
                    ov[42] = 0.0f;
                    ov[43] = 0.0f;
                    ov[44] = 0.0f;
                    ov[45] = 0.0f;
                    ov[46] = 0.0f;
                    ov[47] = 0.0f;
                    ov[48] = 0.0f;
                    ov[49] = 0.0f;
                    ov[50] = 0.0f;
                    ov[51] = 0.0f;
                    ov[52] = 0.0f;
                    ov[53] = 0.0f;
                    ov[54] = 0.0f;
                    ov[55] = 0.0f;
                    ov[56] = 0.0f;
                    ov[57] = 0.0f;
                    ov[58] = 0.0f;
                    ov[59] = 0.0f;
                    ov[60] = 0.0f;
                    ov[61] = 0.0f;
                    ov[62] = 0.0f;
                    ov[63] = 0.0f;
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_26 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_26);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++) {
                    ov[_ls] = ov[_ls] * output_scale;
                }
                #endif
                int out_col = row / 64 * 128;
                float residual[16];
                uint32_t ov_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 0], ov[_lp*2+1 + 0]));
                    ov_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_f32[_pair * 2])[0]), "=f"((&ov_bf16_f32[_pair * 2])[1])
                        : "r"(ov_bf16[_pair]));
                }
                residual[0] = ov[0] - ov_bf16_f32[0];
                residual[1] = ov[1] - ov_bf16_f32[1];
                residual[2] = ov[2] - ov_bf16_f32[2];
                residual[3] = ov[3] - ov_bf16_f32[3];
                residual[4] = ov[4] - ov_bf16_f32[4];
                residual[5] = ov[5] - ov_bf16_f32[5];
                residual[6] = ov[6] - ov_bf16_f32[6];
                residual[7] = ov[7] - ov_bf16_f32[7];
                int byte_offset = o_slot * 65536 + out_col * 128 + row % 64 * 128;
                int swizzled_offset = byte_offset ^ (byte_offset >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset), "r"(ov_bf16[0]), "r"(ov_bf16[1]), "r"(ov_bf16[2]), "r"(ov_bf16[3]) : "memory");
                uint32_t ov_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 8], ov[_lp*2+1 + 8]));
                    ov_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_1_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_1_f32[_pair * 2])[0]), "=f"((&ov_bf16_1_f32[_pair * 2])[1])
                        : "r"(ov_bf16_1[_pair]));
                }
                residual[8] = ov[8] - ov_bf16_1_f32[0];
                residual[9] = ov[9] - ov_bf16_1_f32[1];
                residual[10] = ov[10] - ov_bf16_1_f32[2];
                residual[11] = ov[11] - ov_bf16_1_f32[3];
                residual[12] = ov[12] - ov_bf16_1_f32[4];
                residual[13] = ov[13] - ov_bf16_1_f32[5];
                residual[14] = ov[14] - ov_bf16_1_f32[6];
                residual[15] = ov[15] - ov_bf16_1_f32[7];
                int byte_offset_2 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3 = byte_offset_2 ^ (byte_offset_2 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3), "r"(ov_bf16_1[0]), "r"(ov_bf16_1[1]), "r"(ov_bf16_1[2]), "r"(ov_bf16_1[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual[0 + 0], residual[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual[0 + 2], residual[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual[0 + 4], residual[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual[0 + 6], residual[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual[0 + 8], residual[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual[0 + 10], residual[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual[0 + 12], residual[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual[0 + 14], residual[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_4[16];
                uint32_t ov_bf16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 16], ov[_lp*2+1 + 16]));
                    ov_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_5_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_5_f32[_pair * 2])[0]), "=f"((&ov_bf16_5_f32[_pair * 2])[1])
                        : "r"(ov_bf16_5[_pair]));
                }
                residual_4[0] = ov[16] - ov_bf16_5_f32[0];
                residual_4[1] = ov[17] - ov_bf16_5_f32[1];
                residual_4[2] = ov[18] - ov_bf16_5_f32[2];
                residual_4[3] = ov[19] - ov_bf16_5_f32[3];
                residual_4[4] = ov[20] - ov_bf16_5_f32[4];
                residual_4[5] = ov[21] - ov_bf16_5_f32[5];
                residual_4[6] = ov[22] - ov_bf16_5_f32[6];
                residual_4[7] = ov[23] - ov_bf16_5_f32[7];
                int byte_offset_6 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 32;
                int swizzled_offset_7 = byte_offset_6 ^ (byte_offset_6 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_7), "r"(ov_bf16_5[0]), "r"(ov_bf16_5[1]), "r"(ov_bf16_5[2]), "r"(ov_bf16_5[3]) : "memory");
                uint32_t ov_bf16_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 24], ov[_lp*2+1 + 24]));
                    ov_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_8_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_8_f32[_pair * 2])[0]), "=f"((&ov_bf16_8_f32[_pair * 2])[1])
                        : "r"(ov_bf16_8[_pair]));
                }
                residual_4[8] = ov[24] - ov_bf16_8_f32[0];
                residual_4[9] = ov[25] - ov_bf16_8_f32[1];
                residual_4[10] = ov[26] - ov_bf16_8_f32[2];
                residual_4[11] = ov[27] - ov_bf16_8_f32[3];
                residual_4[12] = ov[28] - ov_bf16_8_f32[4];
                residual_4[13] = ov[29] - ov_bf16_8_f32[5];
                residual_4[14] = ov[30] - ov_bf16_8_f32[6];
                residual_4[15] = ov[31] - ov_bf16_8_f32[7];
                int byte_offset_9 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 48;
                int swizzled_offset_10 = byte_offset_9 ^ (byte_offset_9 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_10), "r"(ov_bf16_8[0]), "r"(ov_bf16_8[1]), "r"(ov_bf16_8[2]), "r"(ov_bf16_8[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_4[0 + 0], residual_4[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_4[0 + 2], residual_4[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_4[0 + 4], residual_4[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_4[0 + 6], residual_4[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_4[0 + 8], residual_4[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_4[0 + 10], residual_4[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_4[0 + 12], residual_4[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_4[0 + 14], residual_4[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_11[16];
                uint32_t ov_bf16_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 32], ov[_lp*2+1 + 32]));
                    ov_bf16_12[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_12_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_12_f32[_pair * 2])[0]), "=f"((&ov_bf16_12_f32[_pair * 2])[1])
                        : "r"(ov_bf16_12[_pair]));
                }
                residual_11[0] = ov[32] - ov_bf16_12_f32[0];
                residual_11[1] = ov[33] - ov_bf16_12_f32[1];
                residual_11[2] = ov[34] - ov_bf16_12_f32[2];
                residual_11[3] = ov[35] - ov_bf16_12_f32[3];
                residual_11[4] = ov[36] - ov_bf16_12_f32[4];
                residual_11[5] = ov[37] - ov_bf16_12_f32[5];
                residual_11[6] = ov[38] - ov_bf16_12_f32[6];
                residual_11[7] = ov[39] - ov_bf16_12_f32[7];
                int byte_offset_13 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 64;
                int swizzled_offset_14 = byte_offset_13 ^ (byte_offset_13 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_14), "r"(ov_bf16_12[0]), "r"(ov_bf16_12[1]), "r"(ov_bf16_12[2]), "r"(ov_bf16_12[3]) : "memory");
                uint32_t ov_bf16_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 40], ov[_lp*2+1 + 40]));
                    ov_bf16_15[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_15_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_15_f32[_pair * 2])[0]), "=f"((&ov_bf16_15_f32[_pair * 2])[1])
                        : "r"(ov_bf16_15[_pair]));
                }
                residual_11[8] = ov[40] - ov_bf16_15_f32[0];
                residual_11[9] = ov[41] - ov_bf16_15_f32[1];
                residual_11[10] = ov[42] - ov_bf16_15_f32[2];
                residual_11[11] = ov[43] - ov_bf16_15_f32[3];
                residual_11[12] = ov[44] - ov_bf16_15_f32[4];
                residual_11[13] = ov[45] - ov_bf16_15_f32[5];
                residual_11[14] = ov[46] - ov_bf16_15_f32[6];
                residual_11[15] = ov[47] - ov_bf16_15_f32[7];
                int byte_offset_16 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 80;
                int swizzled_offset_17 = byte_offset_16 ^ (byte_offset_16 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_17), "r"(ov_bf16_15[0]), "r"(ov_bf16_15[1]), "r"(ov_bf16_15[2]), "r"(ov_bf16_15[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_11[0 + 0], residual_11[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_11[0 + 2], residual_11[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_11[0 + 4], residual_11[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_11[0 + 6], residual_11[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_11[0 + 8], residual_11[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_11[0 + 10], residual_11[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_11[0 + 12], residual_11[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_11[0 + 14], residual_11[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col + 32 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_18[16];
                uint32_t ov_bf16_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 48], ov[_lp*2+1 + 48]));
                    ov_bf16_19[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_19_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_19_f32[_pair * 2])[0]), "=f"((&ov_bf16_19_f32[_pair * 2])[1])
                        : "r"(ov_bf16_19[_pair]));
                }
                residual_18[0] = ov[48] - ov_bf16_19_f32[0];
                residual_18[1] = ov[49] - ov_bf16_19_f32[1];
                residual_18[2] = ov[50] - ov_bf16_19_f32[2];
                residual_18[3] = ov[51] - ov_bf16_19_f32[3];
                residual_18[4] = ov[52] - ov_bf16_19_f32[4];
                residual_18[5] = ov[53] - ov_bf16_19_f32[5];
                residual_18[6] = ov[54] - ov_bf16_19_f32[6];
                residual_18[7] = ov[55] - ov_bf16_19_f32[7];
                int byte_offset_20 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 96;
                int swizzled_offset_21 = byte_offset_20 ^ (byte_offset_20 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_21), "r"(ov_bf16_19[0]), "r"(ov_bf16_19[1]), "r"(ov_bf16_19[2]), "r"(ov_bf16_19[3]) : "memory");
                uint32_t ov_bf16_22[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 56], ov[_lp*2+1 + 56]));
                    ov_bf16_22[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_bf16_22_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_bf16_22_f32[_pair * 2])[0]), "=f"((&ov_bf16_22_f32[_pair * 2])[1])
                        : "r"(ov_bf16_22[_pair]));
                }
                residual_18[8] = ov[56] - ov_bf16_22_f32[0];
                residual_18[9] = ov[57] - ov_bf16_22_f32[1];
                residual_18[10] = ov[58] - ov_bf16_22_f32[2];
                residual_18[11] = ov[59] - ov_bf16_22_f32[3];
                residual_18[12] = ov[60] - ov_bf16_22_f32[4];
                residual_18[13] = ov[61] - ov_bf16_22_f32[5];
                residual_18[14] = ov[62] - ov_bf16_22_f32[6];
                residual_18[15] = ov[63] - ov_bf16_22_f32[7];
                int byte_offset_23 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 112;
                int swizzled_offset_24 = byte_offset_23 ^ (byte_offset_23 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_24), "r"(ov_bf16_22[0]), "r"(ov_bf16_22[1]), "r"(ov_bf16_22[2]), "r"(ov_bf16_22[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_18[0 + 0], residual_18[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_18[0 + 2], residual_18[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_18[0 + 4], residual_18[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_18[0 + 6], residual_18[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_18[0 + 8], residual_18[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_18[0 + 10], residual_18[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_18[0 + 12], residual_18[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_18[0 + 14], residual_18[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col + 48 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk = warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk * 8192));
                    }
                }
                float ov_25[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_25[0]), "=f"(ov_25[1]), "=f"(ov_25[2]), "=f"(ov_25[3]), "=f"(ov_25[4]), "=f"(ov_25[5]), "=f"(ov_25[6]), "=f"(ov_25[7]), "=f"(ov_25[8]), "=f"(ov_25[9]), "=f"(ov_25[10]), "=f"(ov_25[11]), "=f"(ov_25[12]), "=f"(ov_25[13]), "=f"(ov_25[14]), "=f"(ov_25[15]), "=f"(ov_25[16]), "=f"(ov_25[17]), "=f"(ov_25[18]), "=f"(ov_25[19]), "=f"(ov_25[20]), "=f"(ov_25[21]), "=f"(ov_25[22]), "=f"(ov_25[23]), "=f"(ov_25[24]), "=f"(ov_25[25]), "=f"(ov_25[26]), "=f"(ov_25[27]), "=f"(ov_25[28]), "=f"(ov_25[29]), "=f"(ov_25[30]), "=f"(ov_25[31]), "=f"(ov_25[32]), "=f"(ov_25[33]), "=f"(ov_25[34]), "=f"(ov_25[35]), "=f"(ov_25[36]), "=f"(ov_25[37]), "=f"(ov_25[38]), "=f"(ov_25[39]), "=f"(ov_25[40]), "=f"(ov_25[41]), "=f"(ov_25[42]), "=f"(ov_25[43]), "=f"(ov_25[44]), "=f"(ov_25[45]), "=f"(ov_25[46]), "=f"(ov_25[47]), "=f"(ov_25[48]), "=f"(ov_25[49]), "=f"(ov_25[50]), "=f"(ov_25[51]), "=f"(ov_25[52]), "=f"(ov_25[53]), "=f"(ov_25[54]), "=f"(ov_25[55]), "=f"(ov_25[56]), "=f"(ov_25[57]), "=f"(ov_25[58]), "=f"(ov_25[59]), "=f"(ov_25[60]), "=f"(ov_25[61]), "=f"(ov_25[62]), "=f"(ov_25[63])
                        : "r"(tmrow + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_25[0] = 0.0f;
                    ov_25[1] = 0.0f;
                    ov_25[2] = 0.0f;
                    ov_25[3] = 0.0f;
                    ov_25[4] = 0.0f;
                    ov_25[5] = 0.0f;
                    ov_25[6] = 0.0f;
                    ov_25[7] = 0.0f;
                    ov_25[8] = 0.0f;
                    ov_25[9] = 0.0f;
                    ov_25[10] = 0.0f;
                    ov_25[11] = 0.0f;
                    ov_25[12] = 0.0f;
                    ov_25[13] = 0.0f;
                    ov_25[14] = 0.0f;
                    ov_25[15] = 0.0f;
                    ov_25[16] = 0.0f;
                    ov_25[17] = 0.0f;
                    ov_25[18] = 0.0f;
                    ov_25[19] = 0.0f;
                    ov_25[20] = 0.0f;
                    ov_25[21] = 0.0f;
                    ov_25[22] = 0.0f;
                    ov_25[23] = 0.0f;
                    ov_25[24] = 0.0f;
                    ov_25[25] = 0.0f;
                    ov_25[26] = 0.0f;
                    ov_25[27] = 0.0f;
                    ov_25[28] = 0.0f;
                    ov_25[29] = 0.0f;
                    ov_25[30] = 0.0f;
                    ov_25[31] = 0.0f;
                    ov_25[32] = 0.0f;
                    ov_25[33] = 0.0f;
                    ov_25[34] = 0.0f;
                    ov_25[35] = 0.0f;
                    ov_25[36] = 0.0f;
                    ov_25[37] = 0.0f;
                    ov_25[38] = 0.0f;
                    ov_25[39] = 0.0f;
                    ov_25[40] = 0.0f;
                    ov_25[41] = 0.0f;
                    ov_25[42] = 0.0f;
                    ov_25[43] = 0.0f;
                    ov_25[44] = 0.0f;
                    ov_25[45] = 0.0f;
                    ov_25[46] = 0.0f;
                    ov_25[47] = 0.0f;
                    ov_25[48] = 0.0f;
                    ov_25[49] = 0.0f;
                    ov_25[50] = 0.0f;
                    ov_25[51] = 0.0f;
                    ov_25[52] = 0.0f;
                    ov_25[53] = 0.0f;
                    ov_25[54] = 0.0f;
                    ov_25[55] = 0.0f;
                    ov_25[56] = 0.0f;
                    ov_25[57] = 0.0f;
                    ov_25[58] = 0.0f;
                    ov_25[59] = 0.0f;
                    ov_25[60] = 0.0f;
                    ov_25[61] = 0.0f;
                    ov_25[62] = 0.0f;
                    ov_25[63] = 0.0f;
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_27 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_25)[_ls], _scale2_27);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++) {
                    ov_25[_ls] = ov_25[_ls] * output_scale;
                }
                #endif
                int out_col_26 = row / 64 * 128 + 64;
                float residual_27[16];
                uint32_t ov_25_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 0], ov_25[_lp*2+1 + 0]));
                    ov_25_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16[_pair]));
                }
                residual_27[0] = ov_25[0] - ov_25_bf16_f32[0];
                residual_27[1] = ov_25[1] - ov_25_bf16_f32[1];
                residual_27[2] = ov_25[2] - ov_25_bf16_f32[2];
                residual_27[3] = ov_25[3] - ov_25_bf16_f32[3];
                residual_27[4] = ov_25[4] - ov_25_bf16_f32[4];
                residual_27[5] = ov_25[5] - ov_25_bf16_f32[5];
                residual_27[6] = ov_25[6] - ov_25_bf16_f32[6];
                residual_27[7] = ov_25[7] - ov_25_bf16_f32[7];
                int byte_offset_28 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128;
                int swizzled_offset_29 = byte_offset_28 ^ (byte_offset_28 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_29), "r"(ov_25_bf16[0]), "r"(ov_25_bf16[1]), "r"(ov_25_bf16[2]), "r"(ov_25_bf16[3]) : "memory");
                uint32_t ov_25_bf16_30[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 8], ov_25[_lp*2+1 + 8]));
                    ov_25_bf16_30[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_30_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_30_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_30_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_30[_pair]));
                }
                residual_27[8] = ov_25[8] - ov_25_bf16_30_f32[0];
                residual_27[9] = ov_25[9] - ov_25_bf16_30_f32[1];
                residual_27[10] = ov_25[10] - ov_25_bf16_30_f32[2];
                residual_27[11] = ov_25[11] - ov_25_bf16_30_f32[3];
                residual_27[12] = ov_25[12] - ov_25_bf16_30_f32[4];
                residual_27[13] = ov_25[13] - ov_25_bf16_30_f32[5];
                residual_27[14] = ov_25[14] - ov_25_bf16_30_f32[6];
                residual_27[15] = ov_25[15] - ov_25_bf16_30_f32[7];
                int byte_offset_31 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_32 = byte_offset_31 ^ (byte_offset_31 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_32), "r"(ov_25_bf16_30[0]), "r"(ov_25_bf16_30[1]), "r"(ov_25_bf16_30[2]), "r"(ov_25_bf16_30[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_27[0 + 0], residual_27[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_27[0 + 2], residual_27[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_27[0 + 4], residual_27[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_27[0 + 6], residual_27[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_27[0 + 8], residual_27[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_27[0 + 10], residual_27[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_27[0 + 12], residual_27[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_27[0 + 14], residual_27[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_26 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_33[16];
                uint32_t ov_25_bf16_34[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 16], ov_25[_lp*2+1 + 16]));
                    ov_25_bf16_34[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_34_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_34_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_34_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_34[_pair]));
                }
                residual_33[0] = ov_25[16] - ov_25_bf16_34_f32[0];
                residual_33[1] = ov_25[17] - ov_25_bf16_34_f32[1];
                residual_33[2] = ov_25[18] - ov_25_bf16_34_f32[2];
                residual_33[3] = ov_25[19] - ov_25_bf16_34_f32[3];
                residual_33[4] = ov_25[20] - ov_25_bf16_34_f32[4];
                residual_33[5] = ov_25[21] - ov_25_bf16_34_f32[5];
                residual_33[6] = ov_25[22] - ov_25_bf16_34_f32[6];
                residual_33[7] = ov_25[23] - ov_25_bf16_34_f32[7];
                int byte_offset_35 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_36 = byte_offset_35 ^ (byte_offset_35 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_36), "r"(ov_25_bf16_34[0]), "r"(ov_25_bf16_34[1]), "r"(ov_25_bf16_34[2]), "r"(ov_25_bf16_34[3]) : "memory");
                uint32_t ov_25_bf16_37[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 24], ov_25[_lp*2+1 + 24]));
                    ov_25_bf16_37[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_37_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_37_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_37_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_37[_pair]));
                }
                residual_33[8] = ov_25[24] - ov_25_bf16_37_f32[0];
                residual_33[9] = ov_25[25] - ov_25_bf16_37_f32[1];
                residual_33[10] = ov_25[26] - ov_25_bf16_37_f32[2];
                residual_33[11] = ov_25[27] - ov_25_bf16_37_f32[3];
                residual_33[12] = ov_25[28] - ov_25_bf16_37_f32[4];
                residual_33[13] = ov_25[29] - ov_25_bf16_37_f32[5];
                residual_33[14] = ov_25[30] - ov_25_bf16_37_f32[6];
                residual_33[15] = ov_25[31] - ov_25_bf16_37_f32[7];
                int byte_offset_38 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_39 = byte_offset_38 ^ (byte_offset_38 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_39), "r"(ov_25_bf16_37[0]), "r"(ov_25_bf16_37[1]), "r"(ov_25_bf16_37[2]), "r"(ov_25_bf16_37[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_33[0 + 0], residual_33[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_33[0 + 2], residual_33[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_33[0 + 4], residual_33[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_33[0 + 6], residual_33[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_33[0 + 8], residual_33[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_33[0 + 10], residual_33[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_33[0 + 12], residual_33[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_33[0 + 14], residual_33[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_26 + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_40[16];
                uint32_t ov_25_bf16_41[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 32], ov_25[_lp*2+1 + 32]));
                    ov_25_bf16_41[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_41_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_41_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_41_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_41[_pair]));
                }
                residual_40[0] = ov_25[32] - ov_25_bf16_41_f32[0];
                residual_40[1] = ov_25[33] - ov_25_bf16_41_f32[1];
                residual_40[2] = ov_25[34] - ov_25_bf16_41_f32[2];
                residual_40[3] = ov_25[35] - ov_25_bf16_41_f32[3];
                residual_40[4] = ov_25[36] - ov_25_bf16_41_f32[4];
                residual_40[5] = ov_25[37] - ov_25_bf16_41_f32[5];
                residual_40[6] = ov_25[38] - ov_25_bf16_41_f32[6];
                residual_40[7] = ov_25[39] - ov_25_bf16_41_f32[7];
                int byte_offset_42 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_43 = byte_offset_42 ^ (byte_offset_42 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_43), "r"(ov_25_bf16_41[0]), "r"(ov_25_bf16_41[1]), "r"(ov_25_bf16_41[2]), "r"(ov_25_bf16_41[3]) : "memory");
                uint32_t ov_25_bf16_44[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 40], ov_25[_lp*2+1 + 40]));
                    ov_25_bf16_44[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_44_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_44_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_44_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_44[_pair]));
                }
                residual_40[8] = ov_25[40] - ov_25_bf16_44_f32[0];
                residual_40[9] = ov_25[41] - ov_25_bf16_44_f32[1];
                residual_40[10] = ov_25[42] - ov_25_bf16_44_f32[2];
                residual_40[11] = ov_25[43] - ov_25_bf16_44_f32[3];
                residual_40[12] = ov_25[44] - ov_25_bf16_44_f32[4];
                residual_40[13] = ov_25[45] - ov_25_bf16_44_f32[5];
                residual_40[14] = ov_25[46] - ov_25_bf16_44_f32[6];
                residual_40[15] = ov_25[47] - ov_25_bf16_44_f32[7];
                int byte_offset_45 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_46 = byte_offset_45 ^ (byte_offset_45 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_46), "r"(ov_25_bf16_44[0]), "r"(ov_25_bf16_44[1]), "r"(ov_25_bf16_44[2]), "r"(ov_25_bf16_44[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_40[0 + 0], residual_40[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_40[0 + 2], residual_40[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_40[0 + 4], residual_40[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_40[0 + 6], residual_40[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_40[0 + 8], residual_40[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_40[0 + 10], residual_40[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_40[0 + 12], residual_40[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_40[0 + 14], residual_40[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_26 + 32 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_47[16];
                uint32_t ov_25_bf16_48[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 48], ov_25[_lp*2+1 + 48]));
                    ov_25_bf16_48[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_48_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_48_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_48_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_48[_pair]));
                }
                residual_47[0] = ov_25[48] - ov_25_bf16_48_f32[0];
                residual_47[1] = ov_25[49] - ov_25_bf16_48_f32[1];
                residual_47[2] = ov_25[50] - ov_25_bf16_48_f32[2];
                residual_47[3] = ov_25[51] - ov_25_bf16_48_f32[3];
                residual_47[4] = ov_25[52] - ov_25_bf16_48_f32[4];
                residual_47[5] = ov_25[53] - ov_25_bf16_48_f32[5];
                residual_47[6] = ov_25[54] - ov_25_bf16_48_f32[6];
                residual_47[7] = ov_25[55] - ov_25_bf16_48_f32[7];
                int byte_offset_49 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_50 = byte_offset_49 ^ (byte_offset_49 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_50), "r"(ov_25_bf16_48[0]), "r"(ov_25_bf16_48[1]), "r"(ov_25_bf16_48[2]), "r"(ov_25_bf16_48[3]) : "memory");
                uint32_t ov_25_bf16_51[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 56], ov_25[_lp*2+1 + 56]));
                    ov_25_bf16_51[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_25_bf16_51_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_25_bf16_51_f32[_pair * 2])[0]), "=f"((&ov_25_bf16_51_f32[_pair * 2])[1])
                        : "r"(ov_25_bf16_51[_pair]));
                }
                residual_47[8] = ov_25[56] - ov_25_bf16_51_f32[0];
                residual_47[9] = ov_25[57] - ov_25_bf16_51_f32[1];
                residual_47[10] = ov_25[58] - ov_25_bf16_51_f32[2];
                residual_47[11] = ov_25[59] - ov_25_bf16_51_f32[3];
                residual_47[12] = ov_25[60] - ov_25_bf16_51_f32[4];
                residual_47[13] = ov_25[61] - ov_25_bf16_51_f32[5];
                residual_47[14] = ov_25[62] - ov_25_bf16_51_f32[6];
                residual_47[15] = ov_25[63] - ov_25_bf16_51_f32[7];
                int byte_offset_52 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_53 = byte_offset_52 ^ (byte_offset_52 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_53), "r"(ov_25_bf16_51[0]), "r"(ov_25_bf16_51[1]), "r"(ov_25_bf16_51[2]), "r"(ov_25_bf16_51[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_47[0 + 0], residual_47[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_47[0 + 2], residual_47[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_47[0 + 4], residual_47[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_47[0 + 6], residual_47[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_47[0 + 8], residual_47[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_47[0 + 10], residual_47[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_47[0 + 12], residual_47[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_47[0 + 14], residual_47[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_26 + 48 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_1 = warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_1, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_1 * 8192));
                    }
                }
                float ov_54[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_54[0]), "=f"(ov_54[1]), "=f"(ov_54[2]), "=f"(ov_54[3]), "=f"(ov_54[4]), "=f"(ov_54[5]), "=f"(ov_54[6]), "=f"(ov_54[7]), "=f"(ov_54[8]), "=f"(ov_54[9]), "=f"(ov_54[10]), "=f"(ov_54[11]), "=f"(ov_54[12]), "=f"(ov_54[13]), "=f"(ov_54[14]), "=f"(ov_54[15]), "=f"(ov_54[16]), "=f"(ov_54[17]), "=f"(ov_54[18]), "=f"(ov_54[19]), "=f"(ov_54[20]), "=f"(ov_54[21]), "=f"(ov_54[22]), "=f"(ov_54[23]), "=f"(ov_54[24]), "=f"(ov_54[25]), "=f"(ov_54[26]), "=f"(ov_54[27]), "=f"(ov_54[28]), "=f"(ov_54[29]), "=f"(ov_54[30]), "=f"(ov_54[31]), "=f"(ov_54[32]), "=f"(ov_54[33]), "=f"(ov_54[34]), "=f"(ov_54[35]), "=f"(ov_54[36]), "=f"(ov_54[37]), "=f"(ov_54[38]), "=f"(ov_54[39]), "=f"(ov_54[40]), "=f"(ov_54[41]), "=f"(ov_54[42]), "=f"(ov_54[43]), "=f"(ov_54[44]), "=f"(ov_54[45]), "=f"(ov_54[46]), "=f"(ov_54[47]), "=f"(ov_54[48]), "=f"(ov_54[49]), "=f"(ov_54[50]), "=f"(ov_54[51]), "=f"(ov_54[52]), "=f"(ov_54[53]), "=f"(ov_54[54]), "=f"(ov_54[55]), "=f"(ov_54[56]), "=f"(ov_54[57]), "=f"(ov_54[58]), "=f"(ov_54[59]), "=f"(ov_54[60]), "=f"(ov_54[61]), "=f"(ov_54[62]), "=f"(ov_54[63])
                        : "r"(tmrow + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_54[0] = 0.0f;
                    ov_54[1] = 0.0f;
                    ov_54[2] = 0.0f;
                    ov_54[3] = 0.0f;
                    ov_54[4] = 0.0f;
                    ov_54[5] = 0.0f;
                    ov_54[6] = 0.0f;
                    ov_54[7] = 0.0f;
                    ov_54[8] = 0.0f;
                    ov_54[9] = 0.0f;
                    ov_54[10] = 0.0f;
                    ov_54[11] = 0.0f;
                    ov_54[12] = 0.0f;
                    ov_54[13] = 0.0f;
                    ov_54[14] = 0.0f;
                    ov_54[15] = 0.0f;
                    ov_54[16] = 0.0f;
                    ov_54[17] = 0.0f;
                    ov_54[18] = 0.0f;
                    ov_54[19] = 0.0f;
                    ov_54[20] = 0.0f;
                    ov_54[21] = 0.0f;
                    ov_54[22] = 0.0f;
                    ov_54[23] = 0.0f;
                    ov_54[24] = 0.0f;
                    ov_54[25] = 0.0f;
                    ov_54[26] = 0.0f;
                    ov_54[27] = 0.0f;
                    ov_54[28] = 0.0f;
                    ov_54[29] = 0.0f;
                    ov_54[30] = 0.0f;
                    ov_54[31] = 0.0f;
                    ov_54[32] = 0.0f;
                    ov_54[33] = 0.0f;
                    ov_54[34] = 0.0f;
                    ov_54[35] = 0.0f;
                    ov_54[36] = 0.0f;
                    ov_54[37] = 0.0f;
                    ov_54[38] = 0.0f;
                    ov_54[39] = 0.0f;
                    ov_54[40] = 0.0f;
                    ov_54[41] = 0.0f;
                    ov_54[42] = 0.0f;
                    ov_54[43] = 0.0f;
                    ov_54[44] = 0.0f;
                    ov_54[45] = 0.0f;
                    ov_54[46] = 0.0f;
                    ov_54[47] = 0.0f;
                    ov_54[48] = 0.0f;
                    ov_54[49] = 0.0f;
                    ov_54[50] = 0.0f;
                    ov_54[51] = 0.0f;
                    ov_54[52] = 0.0f;
                    ov_54[53] = 0.0f;
                    ov_54[54] = 0.0f;
                    ov_54[55] = 0.0f;
                    ov_54[56] = 0.0f;
                    ov_54[57] = 0.0f;
                    ov_54[58] = 0.0f;
                    ov_54[59] = 0.0f;
                    ov_54[60] = 0.0f;
                    ov_54[61] = 0.0f;
                    ov_54[62] = 0.0f;
                    ov_54[63] = 0.0f;
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_28 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_54)[_ls], _scale2_28);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++) {
                    ov_54[_ls] = ov_54[_ls] * output_scale;
                }
                #endif
                int out_col_55 = 256 + row / 64 * 128;
                float residual_56[16];
                uint32_t ov_54_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 0], ov_54[_lp*2+1 + 0]));
                    ov_54_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16[_pair]));
                }
                residual_56[0] = ov_54[0] - ov_54_bf16_f32[0];
                residual_56[1] = ov_54[1] - ov_54_bf16_f32[1];
                residual_56[2] = ov_54[2] - ov_54_bf16_f32[2];
                residual_56[3] = ov_54[3] - ov_54_bf16_f32[3];
                residual_56[4] = ov_54[4] - ov_54_bf16_f32[4];
                residual_56[5] = ov_54[5] - ov_54_bf16_f32[5];
                residual_56[6] = ov_54[6] - ov_54_bf16_f32[6];
                residual_56[7] = ov_54[7] - ov_54_bf16_f32[7];
                int byte_offset_57 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128;
                int swizzled_offset_58 = byte_offset_57 ^ (byte_offset_57 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_58), "r"(ov_54_bf16[0]), "r"(ov_54_bf16[1]), "r"(ov_54_bf16[2]), "r"(ov_54_bf16[3]) : "memory");
                uint32_t ov_54_bf16_59[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 8], ov_54[_lp*2+1 + 8]));
                    ov_54_bf16_59[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_59_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_59_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_59_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_59[_pair]));
                }
                residual_56[8] = ov_54[8] - ov_54_bf16_59_f32[0];
                residual_56[9] = ov_54[9] - ov_54_bf16_59_f32[1];
                residual_56[10] = ov_54[10] - ov_54_bf16_59_f32[2];
                residual_56[11] = ov_54[11] - ov_54_bf16_59_f32[3];
                residual_56[12] = ov_54[12] - ov_54_bf16_59_f32[4];
                residual_56[13] = ov_54[13] - ov_54_bf16_59_f32[5];
                residual_56[14] = ov_54[14] - ov_54_bf16_59_f32[6];
                residual_56[15] = ov_54[15] - ov_54_bf16_59_f32[7];
                int byte_offset_60 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_61 = byte_offset_60 ^ (byte_offset_60 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_61), "r"(ov_54_bf16_59[0]), "r"(ov_54_bf16_59[1]), "r"(ov_54_bf16_59[2]), "r"(ov_54_bf16_59[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_56[0 + 0], residual_56[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_56[0 + 2], residual_56[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_56[0 + 4], residual_56[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_56[0 + 6], residual_56[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_56[0 + 8], residual_56[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_56[0 + 10], residual_56[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_56[0 + 12], residual_56[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_56[0 + 14], residual_56[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_55 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_62[16];
                uint32_t ov_54_bf16_63[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 16], ov_54[_lp*2+1 + 16]));
                    ov_54_bf16_63[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_63_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_63_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_63_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_63[_pair]));
                }
                residual_62[0] = ov_54[16] - ov_54_bf16_63_f32[0];
                residual_62[1] = ov_54[17] - ov_54_bf16_63_f32[1];
                residual_62[2] = ov_54[18] - ov_54_bf16_63_f32[2];
                residual_62[3] = ov_54[19] - ov_54_bf16_63_f32[3];
                residual_62[4] = ov_54[20] - ov_54_bf16_63_f32[4];
                residual_62[5] = ov_54[21] - ov_54_bf16_63_f32[5];
                residual_62[6] = ov_54[22] - ov_54_bf16_63_f32[6];
                residual_62[7] = ov_54[23] - ov_54_bf16_63_f32[7];
                int byte_offset_64 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_65 = byte_offset_64 ^ (byte_offset_64 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_65), "r"(ov_54_bf16_63[0]), "r"(ov_54_bf16_63[1]), "r"(ov_54_bf16_63[2]), "r"(ov_54_bf16_63[3]) : "memory");
                uint32_t ov_54_bf16_66[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 24], ov_54[_lp*2+1 + 24]));
                    ov_54_bf16_66[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_66_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_66_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_66_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_66[_pair]));
                }
                residual_62[8] = ov_54[24] - ov_54_bf16_66_f32[0];
                residual_62[9] = ov_54[25] - ov_54_bf16_66_f32[1];
                residual_62[10] = ov_54[26] - ov_54_bf16_66_f32[2];
                residual_62[11] = ov_54[27] - ov_54_bf16_66_f32[3];
                residual_62[12] = ov_54[28] - ov_54_bf16_66_f32[4];
                residual_62[13] = ov_54[29] - ov_54_bf16_66_f32[5];
                residual_62[14] = ov_54[30] - ov_54_bf16_66_f32[6];
                residual_62[15] = ov_54[31] - ov_54_bf16_66_f32[7];
                int byte_offset_67 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_68 = byte_offset_67 ^ (byte_offset_67 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_68), "r"(ov_54_bf16_66[0]), "r"(ov_54_bf16_66[1]), "r"(ov_54_bf16_66[2]), "r"(ov_54_bf16_66[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_62[0 + 0], residual_62[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_62[0 + 2], residual_62[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_62[0 + 4], residual_62[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_62[0 + 6], residual_62[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_62[0 + 8], residual_62[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_62[0 + 10], residual_62[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_62[0 + 12], residual_62[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_62[0 + 14], residual_62[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_55 + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_69[16];
                uint32_t ov_54_bf16_70[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 32], ov_54[_lp*2+1 + 32]));
                    ov_54_bf16_70[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_70_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_70_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_70_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_70[_pair]));
                }
                residual_69[0] = ov_54[32] - ov_54_bf16_70_f32[0];
                residual_69[1] = ov_54[33] - ov_54_bf16_70_f32[1];
                residual_69[2] = ov_54[34] - ov_54_bf16_70_f32[2];
                residual_69[3] = ov_54[35] - ov_54_bf16_70_f32[3];
                residual_69[4] = ov_54[36] - ov_54_bf16_70_f32[4];
                residual_69[5] = ov_54[37] - ov_54_bf16_70_f32[5];
                residual_69[6] = ov_54[38] - ov_54_bf16_70_f32[6];
                residual_69[7] = ov_54[39] - ov_54_bf16_70_f32[7];
                int byte_offset_71 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_72 = byte_offset_71 ^ (byte_offset_71 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_72), "r"(ov_54_bf16_70[0]), "r"(ov_54_bf16_70[1]), "r"(ov_54_bf16_70[2]), "r"(ov_54_bf16_70[3]) : "memory");
                uint32_t ov_54_bf16_73[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 40], ov_54[_lp*2+1 + 40]));
                    ov_54_bf16_73[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_73_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_73_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_73_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_73[_pair]));
                }
                residual_69[8] = ov_54[40] - ov_54_bf16_73_f32[0];
                residual_69[9] = ov_54[41] - ov_54_bf16_73_f32[1];
                residual_69[10] = ov_54[42] - ov_54_bf16_73_f32[2];
                residual_69[11] = ov_54[43] - ov_54_bf16_73_f32[3];
                residual_69[12] = ov_54[44] - ov_54_bf16_73_f32[4];
                residual_69[13] = ov_54[45] - ov_54_bf16_73_f32[5];
                residual_69[14] = ov_54[46] - ov_54_bf16_73_f32[6];
                residual_69[15] = ov_54[47] - ov_54_bf16_73_f32[7];
                int byte_offset_74 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_75 = byte_offset_74 ^ (byte_offset_74 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_75), "r"(ov_54_bf16_73[0]), "r"(ov_54_bf16_73[1]), "r"(ov_54_bf16_73[2]), "r"(ov_54_bf16_73[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_69[0 + 0], residual_69[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_69[0 + 2], residual_69[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_69[0 + 4], residual_69[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_69[0 + 6], residual_69[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_69[0 + 8], residual_69[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_69[0 + 10], residual_69[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_69[0 + 12], residual_69[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_69[0 + 14], residual_69[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_55 + 32 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_76[16];
                uint32_t ov_54_bf16_77[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 48], ov_54[_lp*2+1 + 48]));
                    ov_54_bf16_77[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_77_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_77_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_77_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_77[_pair]));
                }
                residual_76[0] = ov_54[48] - ov_54_bf16_77_f32[0];
                residual_76[1] = ov_54[49] - ov_54_bf16_77_f32[1];
                residual_76[2] = ov_54[50] - ov_54_bf16_77_f32[2];
                residual_76[3] = ov_54[51] - ov_54_bf16_77_f32[3];
                residual_76[4] = ov_54[52] - ov_54_bf16_77_f32[4];
                residual_76[5] = ov_54[53] - ov_54_bf16_77_f32[5];
                residual_76[6] = ov_54[54] - ov_54_bf16_77_f32[6];
                residual_76[7] = ov_54[55] - ov_54_bf16_77_f32[7];
                int byte_offset_78 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_79 = byte_offset_78 ^ (byte_offset_78 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_79), "r"(ov_54_bf16_77[0]), "r"(ov_54_bf16_77[1]), "r"(ov_54_bf16_77[2]), "r"(ov_54_bf16_77[3]) : "memory");
                uint32_t ov_54_bf16_80[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 56], ov_54[_lp*2+1 + 56]));
                    ov_54_bf16_80[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_54_bf16_80_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_54_bf16_80_f32[_pair * 2])[0]), "=f"((&ov_54_bf16_80_f32[_pair * 2])[1])
                        : "r"(ov_54_bf16_80[_pair]));
                }
                residual_76[8] = ov_54[56] - ov_54_bf16_80_f32[0];
                residual_76[9] = ov_54[57] - ov_54_bf16_80_f32[1];
                residual_76[10] = ov_54[58] - ov_54_bf16_80_f32[2];
                residual_76[11] = ov_54[59] - ov_54_bf16_80_f32[3];
                residual_76[12] = ov_54[60] - ov_54_bf16_80_f32[4];
                residual_76[13] = ov_54[61] - ov_54_bf16_80_f32[5];
                residual_76[14] = ov_54[62] - ov_54_bf16_80_f32[6];
                residual_76[15] = ov_54[63] - ov_54_bf16_80_f32[7];
                int byte_offset_81 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_82 = byte_offset_81 ^ (byte_offset_81 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_82), "r"(ov_54_bf16_80[0]), "r"(ov_54_bf16_80[1]), "r"(ov_54_bf16_80[2]), "r"(ov_54_bf16_80[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_76[0 + 0], residual_76[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_76[0 + 2], residual_76[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_76[0 + 4], residual_76[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_76[0 + 6], residual_76[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_76[0 + 8], residual_76[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_76[0 + 10], residual_76[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_76[0 + 12], residual_76[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_76[0 + 14], residual_76[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_55 + 48 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_2 = 4 + warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk_2, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_2 * 8192));
                    }
                }
                float ov_83[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_83[0]), "=f"(ov_83[1]), "=f"(ov_83[2]), "=f"(ov_83[3]), "=f"(ov_83[4]), "=f"(ov_83[5]), "=f"(ov_83[6]), "=f"(ov_83[7]), "=f"(ov_83[8]), "=f"(ov_83[9]), "=f"(ov_83[10]), "=f"(ov_83[11]), "=f"(ov_83[12]), "=f"(ov_83[13]), "=f"(ov_83[14]), "=f"(ov_83[15]), "=f"(ov_83[16]), "=f"(ov_83[17]), "=f"(ov_83[18]), "=f"(ov_83[19]), "=f"(ov_83[20]), "=f"(ov_83[21]), "=f"(ov_83[22]), "=f"(ov_83[23]), "=f"(ov_83[24]), "=f"(ov_83[25]), "=f"(ov_83[26]), "=f"(ov_83[27]), "=f"(ov_83[28]), "=f"(ov_83[29]), "=f"(ov_83[30]), "=f"(ov_83[31]), "=f"(ov_83[32]), "=f"(ov_83[33]), "=f"(ov_83[34]), "=f"(ov_83[35]), "=f"(ov_83[36]), "=f"(ov_83[37]), "=f"(ov_83[38]), "=f"(ov_83[39]), "=f"(ov_83[40]), "=f"(ov_83[41]), "=f"(ov_83[42]), "=f"(ov_83[43]), "=f"(ov_83[44]), "=f"(ov_83[45]), "=f"(ov_83[46]), "=f"(ov_83[47]), "=f"(ov_83[48]), "=f"(ov_83[49]), "=f"(ov_83[50]), "=f"(ov_83[51]), "=f"(ov_83[52]), "=f"(ov_83[53]), "=f"(ov_83[54]), "=f"(ov_83[55]), "=f"(ov_83[56]), "=f"(ov_83[57]), "=f"(ov_83[58]), "=f"(ov_83[59]), "=f"(ov_83[60]), "=f"(ov_83[61]), "=f"(ov_83[62]), "=f"(ov_83[63])
                        : "r"(tmrow + 128 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_83[0] = 0.0f;
                    ov_83[1] = 0.0f;
                    ov_83[2] = 0.0f;
                    ov_83[3] = 0.0f;
                    ov_83[4] = 0.0f;
                    ov_83[5] = 0.0f;
                    ov_83[6] = 0.0f;
                    ov_83[7] = 0.0f;
                    ov_83[8] = 0.0f;
                    ov_83[9] = 0.0f;
                    ov_83[10] = 0.0f;
                    ov_83[11] = 0.0f;
                    ov_83[12] = 0.0f;
                    ov_83[13] = 0.0f;
                    ov_83[14] = 0.0f;
                    ov_83[15] = 0.0f;
                    ov_83[16] = 0.0f;
                    ov_83[17] = 0.0f;
                    ov_83[18] = 0.0f;
                    ov_83[19] = 0.0f;
                    ov_83[20] = 0.0f;
                    ov_83[21] = 0.0f;
                    ov_83[22] = 0.0f;
                    ov_83[23] = 0.0f;
                    ov_83[24] = 0.0f;
                    ov_83[25] = 0.0f;
                    ov_83[26] = 0.0f;
                    ov_83[27] = 0.0f;
                    ov_83[28] = 0.0f;
                    ov_83[29] = 0.0f;
                    ov_83[30] = 0.0f;
                    ov_83[31] = 0.0f;
                    ov_83[32] = 0.0f;
                    ov_83[33] = 0.0f;
                    ov_83[34] = 0.0f;
                    ov_83[35] = 0.0f;
                    ov_83[36] = 0.0f;
                    ov_83[37] = 0.0f;
                    ov_83[38] = 0.0f;
                    ov_83[39] = 0.0f;
                    ov_83[40] = 0.0f;
                    ov_83[41] = 0.0f;
                    ov_83[42] = 0.0f;
                    ov_83[43] = 0.0f;
                    ov_83[44] = 0.0f;
                    ov_83[45] = 0.0f;
                    ov_83[46] = 0.0f;
                    ov_83[47] = 0.0f;
                    ov_83[48] = 0.0f;
                    ov_83[49] = 0.0f;
                    ov_83[50] = 0.0f;
                    ov_83[51] = 0.0f;
                    ov_83[52] = 0.0f;
                    ov_83[53] = 0.0f;
                    ov_83[54] = 0.0f;
                    ov_83[55] = 0.0f;
                    ov_83[56] = 0.0f;
                    ov_83[57] = 0.0f;
                    ov_83[58] = 0.0f;
                    ov_83[59] = 0.0f;
                    ov_83[60] = 0.0f;
                    ov_83[61] = 0.0f;
                    ov_83[62] = 0.0f;
                    ov_83[63] = 0.0f;
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_29 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_83)[_ls], _scale2_29);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++) {
                    ov_83[_ls] = ov_83[_ls] * output_scale;
                }
                #endif
                int out_col_84 = 256 + row / 64 * 128 + 64;
                float residual_85[16];
                uint32_t ov_83_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 0], ov_83[_lp*2+1 + 0]));
                    ov_83_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16[_pair]));
                }
                residual_85[0] = ov_83[0] - ov_83_bf16_f32[0];
                residual_85[1] = ov_83[1] - ov_83_bf16_f32[1];
                residual_85[2] = ov_83[2] - ov_83_bf16_f32[2];
                residual_85[3] = ov_83[3] - ov_83_bf16_f32[3];
                residual_85[4] = ov_83[4] - ov_83_bf16_f32[4];
                residual_85[5] = ov_83[5] - ov_83_bf16_f32[5];
                residual_85[6] = ov_83[6] - ov_83_bf16_f32[6];
                residual_85[7] = ov_83[7] - ov_83_bf16_f32[7];
                int byte_offset_86 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128;
                int swizzled_offset_87 = byte_offset_86 ^ (byte_offset_86 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_87), "r"(ov_83_bf16[0]), "r"(ov_83_bf16[1]), "r"(ov_83_bf16[2]), "r"(ov_83_bf16[3]) : "memory");
                uint32_t ov_83_bf16_88[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 8], ov_83[_lp*2+1 + 8]));
                    ov_83_bf16_88[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_88_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_88_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_88_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_88[_pair]));
                }
                residual_85[8] = ov_83[8] - ov_83_bf16_88_f32[0];
                residual_85[9] = ov_83[9] - ov_83_bf16_88_f32[1];
                residual_85[10] = ov_83[10] - ov_83_bf16_88_f32[2];
                residual_85[11] = ov_83[11] - ov_83_bf16_88_f32[3];
                residual_85[12] = ov_83[12] - ov_83_bf16_88_f32[4];
                residual_85[13] = ov_83[13] - ov_83_bf16_88_f32[5];
                residual_85[14] = ov_83[14] - ov_83_bf16_88_f32[6];
                residual_85[15] = ov_83[15] - ov_83_bf16_88_f32[7];
                int byte_offset_89 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_90 = byte_offset_89 ^ (byte_offset_89 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_90), "r"(ov_83_bf16_88[0]), "r"(ov_83_bf16_88[1]), "r"(ov_83_bf16_88[2]), "r"(ov_83_bf16_88[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_85[0 + 0], residual_85[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_85[0 + 2], residual_85[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_85[0 + 4], residual_85[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_85[0 + 6], residual_85[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_85[0 + 8], residual_85[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_85[0 + 10], residual_85[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_85[0 + 12], residual_85[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_85[0 + 14], residual_85[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_84 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_91[16];
                uint32_t ov_83_bf16_92[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 16], ov_83[_lp*2+1 + 16]));
                    ov_83_bf16_92[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_92_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_92_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_92_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_92[_pair]));
                }
                residual_91[0] = ov_83[16] - ov_83_bf16_92_f32[0];
                residual_91[1] = ov_83[17] - ov_83_bf16_92_f32[1];
                residual_91[2] = ov_83[18] - ov_83_bf16_92_f32[2];
                residual_91[3] = ov_83[19] - ov_83_bf16_92_f32[3];
                residual_91[4] = ov_83[20] - ov_83_bf16_92_f32[4];
                residual_91[5] = ov_83[21] - ov_83_bf16_92_f32[5];
                residual_91[6] = ov_83[22] - ov_83_bf16_92_f32[6];
                residual_91[7] = ov_83[23] - ov_83_bf16_92_f32[7];
                int byte_offset_93 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_94 = byte_offset_93 ^ (byte_offset_93 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_94), "r"(ov_83_bf16_92[0]), "r"(ov_83_bf16_92[1]), "r"(ov_83_bf16_92[2]), "r"(ov_83_bf16_92[3]) : "memory");
                uint32_t ov_83_bf16_95[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 24], ov_83[_lp*2+1 + 24]));
                    ov_83_bf16_95[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_95_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_95_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_95_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_95[_pair]));
                }
                residual_91[8] = ov_83[24] - ov_83_bf16_95_f32[0];
                residual_91[9] = ov_83[25] - ov_83_bf16_95_f32[1];
                residual_91[10] = ov_83[26] - ov_83_bf16_95_f32[2];
                residual_91[11] = ov_83[27] - ov_83_bf16_95_f32[3];
                residual_91[12] = ov_83[28] - ov_83_bf16_95_f32[4];
                residual_91[13] = ov_83[29] - ov_83_bf16_95_f32[5];
                residual_91[14] = ov_83[30] - ov_83_bf16_95_f32[6];
                residual_91[15] = ov_83[31] - ov_83_bf16_95_f32[7];
                int byte_offset_96 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_97 = byte_offset_96 ^ (byte_offset_96 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_97), "r"(ov_83_bf16_95[0]), "r"(ov_83_bf16_95[1]), "r"(ov_83_bf16_95[2]), "r"(ov_83_bf16_95[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_91[0 + 0], residual_91[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_91[0 + 2], residual_91[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_91[0 + 4], residual_91[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_91[0 + 6], residual_91[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_91[0 + 8], residual_91[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_91[0 + 10], residual_91[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_91[0 + 12], residual_91[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_91[0 + 14], residual_91[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_84 + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_98[16];
                uint32_t ov_83_bf16_99[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 32], ov_83[_lp*2+1 + 32]));
                    ov_83_bf16_99[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_99_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_99_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_99_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_99[_pair]));
                }
                residual_98[0] = ov_83[32] - ov_83_bf16_99_f32[0];
                residual_98[1] = ov_83[33] - ov_83_bf16_99_f32[1];
                residual_98[2] = ov_83[34] - ov_83_bf16_99_f32[2];
                residual_98[3] = ov_83[35] - ov_83_bf16_99_f32[3];
                residual_98[4] = ov_83[36] - ov_83_bf16_99_f32[4];
                residual_98[5] = ov_83[37] - ov_83_bf16_99_f32[5];
                residual_98[6] = ov_83[38] - ov_83_bf16_99_f32[6];
                residual_98[7] = ov_83[39] - ov_83_bf16_99_f32[7];
                int byte_offset_100 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_101 = byte_offset_100 ^ (byte_offset_100 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_101), "r"(ov_83_bf16_99[0]), "r"(ov_83_bf16_99[1]), "r"(ov_83_bf16_99[2]), "r"(ov_83_bf16_99[3]) : "memory");
                uint32_t ov_83_bf16_102[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 40], ov_83[_lp*2+1 + 40]));
                    ov_83_bf16_102[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_102_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_102_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_102_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_102[_pair]));
                }
                residual_98[8] = ov_83[40] - ov_83_bf16_102_f32[0];
                residual_98[9] = ov_83[41] - ov_83_bf16_102_f32[1];
                residual_98[10] = ov_83[42] - ov_83_bf16_102_f32[2];
                residual_98[11] = ov_83[43] - ov_83_bf16_102_f32[3];
                residual_98[12] = ov_83[44] - ov_83_bf16_102_f32[4];
                residual_98[13] = ov_83[45] - ov_83_bf16_102_f32[5];
                residual_98[14] = ov_83[46] - ov_83_bf16_102_f32[6];
                residual_98[15] = ov_83[47] - ov_83_bf16_102_f32[7];
                int byte_offset_103 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_104 = byte_offset_103 ^ (byte_offset_103 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_104), "r"(ov_83_bf16_102[0]), "r"(ov_83_bf16_102[1]), "r"(ov_83_bf16_102[2]), "r"(ov_83_bf16_102[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_98[0 + 0], residual_98[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_98[0 + 2], residual_98[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_98[0 + 4], residual_98[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_98[0 + 6], residual_98[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_98[0 + 8], residual_98[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_98[0 + 10], residual_98[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_98[0 + 12], residual_98[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_98[0 + 14], residual_98[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_84 + 32 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                float residual_105[16];
                uint32_t ov_83_bf16_106[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 48], ov_83[_lp*2+1 + 48]));
                    ov_83_bf16_106[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_106_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_106_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_106_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_106[_pair]));
                }
                residual_105[0] = ov_83[48] - ov_83_bf16_106_f32[0];
                residual_105[1] = ov_83[49] - ov_83_bf16_106_f32[1];
                residual_105[2] = ov_83[50] - ov_83_bf16_106_f32[2];
                residual_105[3] = ov_83[51] - ov_83_bf16_106_f32[3];
                residual_105[4] = ov_83[52] - ov_83_bf16_106_f32[4];
                residual_105[5] = ov_83[53] - ov_83_bf16_106_f32[5];
                residual_105[6] = ov_83[54] - ov_83_bf16_106_f32[6];
                residual_105[7] = ov_83[55] - ov_83_bf16_106_f32[7];
                int byte_offset_107 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_108 = byte_offset_107 ^ (byte_offset_107 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_108), "r"(ov_83_bf16_106[0]), "r"(ov_83_bf16_106[1]), "r"(ov_83_bf16_106[2]), "r"(ov_83_bf16_106[3]) : "memory");
                uint32_t ov_83_bf16_109[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 56], ov_83[_lp*2+1 + 56]));
                    ov_83_bf16_109[_lp] = *(uint32_t*)&_bf2;
                }
                float ov_83_bf16_109_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&ov_83_bf16_109_f32[_pair * 2])[0]), "=f"((&ov_83_bf16_109_f32[_pair * 2])[1])
                        : "r"(ov_83_bf16_109[_pair]));
                }
                residual_105[8] = ov_83[56] - ov_83_bf16_109_f32[0];
                residual_105[9] = ov_83[57] - ov_83_bf16_109_f32[1];
                residual_105[10] = ov_83[58] - ov_83_bf16_109_f32[2];
                residual_105[11] = ov_83[59] - ov_83_bf16_109_f32[3];
                residual_105[12] = ov_83[60] - ov_83_bf16_109_f32[4];
                residual_105[13] = ov_83[61] - ov_83_bf16_109_f32[5];
                residual_105[14] = ov_83[62] - ov_83_bf16_109_f32[6];
                residual_105[15] = ov_83[63] - ov_83_bf16_109_f32[7];
                int byte_offset_110 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_111 = byte_offset_110 ^ (byte_offset_110 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_111), "r"(ov_83_bf16_109[0]), "r"(ov_83_bf16_109[1]), "r"(ov_83_bf16_109[2]), "r"(ov_83_bf16_109[3]) : "memory");
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(residual_105[0 + 0], residual_105[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(residual_105[0 + 2], residual_105[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(residual_105[0 + 4], residual_105[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(residual_105[0 + 6], residual_105[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(residual_105[0 + 8], residual_105[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(residual_105[0 + 10], residual_105[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(residual_105[0 + 12], residual_105[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(residual_105[0 + 14], residual_105[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(o_lo))[o_row_base + (long long)out_col_84 + 48 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_3 = 4 + warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_3, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_3 * 8192));
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                mbarrier_wait(clc_full_addr, outer_phase);
                uint32_t _clc_valid_0 = 0;
                uint32_t _clc_ctaid_x_0;
                uint32_t _clc_ctaid_y_0;
                uint32_t _clc_ctaid_z_0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_0 == 0) {
                    break;
                }
                query = _clc_ctaid_x_0;
                outer_phase = outer_phase ^ 1;
            }
            if (warp == 3) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 4 && warp <= 7) {
        { // producer_main
            int query_1 = blockIdx.x;
            unsigned int outer_phase_1 = 0;
            int cursor_1 = 0;
            int producer_warp = warp - 4;
            #pragma unroll 1
            for (int outer_1 = 0; outer_1 < num_queries; outer_1++) {
                int active_1 = topk;
                if (derive_length != 0) {
                    int len_slot_1 = outer_1 & 1;
                    unsigned int len_phase_1 = outer_1 >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot_1) * 8, len_phase_1);
                    int _max_39 = ((len_cells[len_slot_1]) > (0) ? (len_cells[len_slot_1]) : (0));
                    int _min_2 = ((_max_39) < (topk) ? (_max_39) : (topk));
                    active_1 = _min_2;
                } else if (has_topk_length != 0) {
                    int _max_40 = ((topk_length[query_1]) > (0) ? (topk_length[query_1]) : (0));
                    int _min_3 = ((_max_40) < (topk) ? (_max_40) : (topk));
                    active_1 = _min_3;
                }
                int active_0_1 = active_1;
                int _max_41 = (((active_0_1 + 63) / 64) > (2) ? ((active_0_1 + 63) / 64) : (2));
                int blocks_1 = _max_41;
                long long row_base = (long long)indices_offset + (long long)query_1 * (long long)idx_stride;
                #pragma unroll 1
                for (int block_1 = 0; block_1 < blocks_1; block_1++) {
                    if (elect_sync()) {
                        int idx[16];
                        int smallest = num_kv;
                        int largest = -1;
                        int whole_tile = active_0_1 >= block_1 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (whole_tile != 0) {
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_0 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + producer_warp * 4)) + 0);
                                int4 _ivld_0;
                                _ivld_0 = *_ivptr_0;
                                _vec_load_0[0 + 0] = _ivld_0.x;
                                _vec_load_0[0 + 1] = _ivld_0.y;
                                _vec_load_0[0 + 2] = _ivld_0.z;
                                _vec_load_0[0 + 3] = _ivld_0.w;
                            }
                            idx[0] = _vec_load_0[0];
                            idx[1] = _vec_load_0[1];
                            idx[2] = _vec_load_0[2];
                            idx[3] = _vec_load_0[3];
                            int _max_42 = ((_vec_load_0[0]) > (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _max_43 = ((_vec_load_0[2]) > (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _max_44 = ((_max_42) > (_max_43) ? (_max_42) : (_max_43));
                            int _max_45 = ((largest) > (_max_44) ? (largest) : (_max_44));
                            largest = _max_45;
                            int _min_4 = ((_vec_load_0[0]) < (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _min_5 = ((_vec_load_0[2]) < (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _min_6 = ((_min_4) < (_min_5) ? (_min_4) : (_min_5));
                            int _min_7 = ((smallest) < (_min_6) ? (smallest) : (_min_6));
                            smallest = _min_7;
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_1 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 16 + producer_warp * 4)) + 0);
                                int4 _ivld_1;
                                _ivld_1 = *_ivptr_1;
                                _vec_load_1[0 + 0] = _ivld_1.x;
                                _vec_load_1[0 + 1] = _ivld_1.y;
                                _vec_load_1[0 + 2] = _ivld_1.z;
                                _vec_load_1[0 + 3] = _ivld_1.w;
                            }
                            idx[4] = _vec_load_1[0];
                            idx[5] = _vec_load_1[1];
                            idx[6] = _vec_load_1[2];
                            idx[7] = _vec_load_1[3];
                            int _max_46 = ((_vec_load_1[0]) > (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _max_47 = ((_vec_load_1[2]) > (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _max_48 = ((_max_46) > (_max_47) ? (_max_46) : (_max_47));
                            int _max_49 = ((largest) > (_max_48) ? (largest) : (_max_48));
                            largest = _max_49;
                            int _min_8 = ((_vec_load_1[0]) < (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _min_9 = ((_vec_load_1[2]) < (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _min_10 = ((_min_8) < (_min_9) ? (_min_8) : (_min_9));
                            int _min_11 = ((smallest) < (_min_10) ? (smallest) : (_min_10));
                            smallest = _min_11;
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 32 + producer_warp * 4)) + 0);
                                int4 _ivld_2;
                                _ivld_2 = *_ivptr_2;
                                _vec_load_2[0 + 0] = _ivld_2.x;
                                _vec_load_2[0 + 1] = _ivld_2.y;
                                _vec_load_2[0 + 2] = _ivld_2.z;
                                _vec_load_2[0 + 3] = _ivld_2.w;
                            }
                            idx[8] = _vec_load_2[0];
                            idx[9] = _vec_load_2[1];
                            idx[10] = _vec_load_2[2];
                            idx[11] = _vec_load_2[3];
                            int _max_50 = ((_vec_load_2[0]) > (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _max_51 = ((_vec_load_2[2]) > (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _max_52 = ((_max_50) > (_max_51) ? (_max_50) : (_max_51));
                            int _max_53 = ((largest) > (_max_52) ? (largest) : (_max_52));
                            largest = _max_53;
                            int _min_12 = ((_vec_load_2[0]) < (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _min_13 = ((_vec_load_2[2]) < (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _min_14 = ((_min_12) < (_min_13) ? (_min_12) : (_min_13));
                            int _min_15 = ((smallest) < (_min_14) ? (smallest) : (_min_14));
                            smallest = _min_15;
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_3 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 48 + producer_warp * 4)) + 0);
                                int4 _ivld_3;
                                _ivld_3 = *_ivptr_3;
                                _vec_load_3[0 + 0] = _ivld_3.x;
                                _vec_load_3[0 + 1] = _ivld_3.y;
                                _vec_load_3[0 + 2] = _ivld_3.z;
                                _vec_load_3[0 + 3] = _ivld_3.w;
                            }
                            idx[12] = _vec_load_3[0];
                            idx[13] = _vec_load_3[1];
                            idx[14] = _vec_load_3[2];
                            idx[15] = _vec_load_3[3];
                            int _max_54 = ((_vec_load_3[0]) > (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _max_55 = ((_vec_load_3[2]) > (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _max_56 = ((_max_54) > (_max_55) ? (_max_54) : (_max_55));
                            int _max_57 = ((largest) > (_max_56) ? (largest) : (_max_56));
                            largest = _max_57;
                            int _min_16 = ((_vec_load_3[0]) < (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _min_17 = ((_vec_load_3[2]) < (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _min_18 = ((_min_16) < (_min_17) ? (_min_16) : (_min_17));
                            int _min_19 = ((smallest) < (_min_18) ? (smallest) : (_min_18));
                            smallest = _min_19;
                        } else {
                            int position = block_1 * 64 + producer_warp * 4;
                            int _min_20 = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int clamped = _min_20;
                            int value = indices[row_base + (long long)clamped];
                            if (position >= active_0_1) {
                                value = -1;
                            }
                            idx[0] = value;
                            int _max_58 = ((largest) > (value) ? (largest) : (value));
                            largest = _max_58;
                            int _min_21 = ((smallest) < (value) ? (smallest) : (value));
                            smallest = _min_21;
                            int position_0 = block_1 * 64 + producer_warp * 4 + 1;
                            int _min_22 = ((position_0) < (topk - 1) ? (position_0) : (topk - 1));
                            int clamped_1 = _min_22;
                            int value_2 = indices[row_base + (long long)clamped_1];
                            if (position_0 >= active_0_1) {
                                value_2 = -1;
                            }
                            idx[1] = value_2;
                            int _max_59 = ((largest) > (value_2) ? (largest) : (value_2));
                            largest = _max_59;
                            int _min_23 = ((smallest) < (value_2) ? (smallest) : (value_2));
                            smallest = _min_23;
                            int position_3 = block_1 * 64 + producer_warp * 4 + 2;
                            int _min_24 = ((position_3) < (topk - 1) ? (position_3) : (topk - 1));
                            int clamped_4 = _min_24;
                            int value_5 = indices[row_base + (long long)clamped_4];
                            if (position_3 >= active_0_1) {
                                value_5 = -1;
                            }
                            idx[2] = value_5;
                            int _max_60 = ((largest) > (value_5) ? (largest) : (value_5));
                            largest = _max_60;
                            int _min_25 = ((smallest) < (value_5) ? (smallest) : (value_5));
                            smallest = _min_25;
                            int position_6 = block_1 * 64 + producer_warp * 4 + 3;
                            int _min_26 = ((position_6) < (topk - 1) ? (position_6) : (topk - 1));
                            int clamped_7 = _min_26;
                            int value_8 = indices[row_base + (long long)clamped_7];
                            if (position_6 >= active_0_1) {
                                value_8 = -1;
                            }
                            idx[3] = value_8;
                            int _max_61 = ((largest) > (value_8) ? (largest) : (value_8));
                            largest = _max_61;
                            int _min_27 = ((smallest) < (value_8) ? (smallest) : (value_8));
                            smallest = _min_27;
                            int position_9 = block_1 * 64 + 16 + producer_warp * 4;
                            int _min_28 = ((position_9) < (topk - 1) ? (position_9) : (topk - 1));
                            int clamped_10 = _min_28;
                            int value_11 = indices[row_base + (long long)clamped_10];
                            if (position_9 >= active_0_1) {
                                value_11 = -1;
                            }
                            idx[4] = value_11;
                            int _max_62 = ((largest) > (value_11) ? (largest) : (value_11));
                            largest = _max_62;
                            int _min_29 = ((smallest) < (value_11) ? (smallest) : (value_11));
                            smallest = _min_29;
                            int position_12 = block_1 * 64 + 16 + producer_warp * 4 + 1;
                            int _min_30 = ((position_12) < (topk - 1) ? (position_12) : (topk - 1));
                            int clamped_13 = _min_30;
                            int value_14 = indices[row_base + (long long)clamped_13];
                            if (position_12 >= active_0_1) {
                                value_14 = -1;
                            }
                            idx[5] = value_14;
                            int _max_63 = ((largest) > (value_14) ? (largest) : (value_14));
                            largest = _max_63;
                            int _min_31 = ((smallest) < (value_14) ? (smallest) : (value_14));
                            smallest = _min_31;
                            int position_15 = block_1 * 64 + 16 + producer_warp * 4 + 2;
                            int _min_32 = ((position_15) < (topk - 1) ? (position_15) : (topk - 1));
                            int clamped_16 = _min_32;
                            int value_17 = indices[row_base + (long long)clamped_16];
                            if (position_15 >= active_0_1) {
                                value_17 = -1;
                            }
                            idx[6] = value_17;
                            int _max_64 = ((largest) > (value_17) ? (largest) : (value_17));
                            largest = _max_64;
                            int _min_33 = ((smallest) < (value_17) ? (smallest) : (value_17));
                            smallest = _min_33;
                            int position_18 = block_1 * 64 + 16 + producer_warp * 4 + 3;
                            int _min_34 = ((position_18) < (topk - 1) ? (position_18) : (topk - 1));
                            int clamped_19 = _min_34;
                            int value_20 = indices[row_base + (long long)clamped_19];
                            if (position_18 >= active_0_1) {
                                value_20 = -1;
                            }
                            idx[7] = value_20;
                            int _max_65 = ((largest) > (value_20) ? (largest) : (value_20));
                            largest = _max_65;
                            int _min_35 = ((smallest) < (value_20) ? (smallest) : (value_20));
                            smallest = _min_35;
                            int position_21 = block_1 * 64 + 32 + producer_warp * 4;
                            int _min_36 = ((position_21) < (topk - 1) ? (position_21) : (topk - 1));
                            int clamped_22 = _min_36;
                            int value_23 = indices[row_base + (long long)clamped_22];
                            if (position_21 >= active_0_1) {
                                value_23 = -1;
                            }
                            idx[8] = value_23;
                            int _max_66 = ((largest) > (value_23) ? (largest) : (value_23));
                            largest = _max_66;
                            int _min_37 = ((smallest) < (value_23) ? (smallest) : (value_23));
                            smallest = _min_37;
                            int position_24 = block_1 * 64 + 32 + producer_warp * 4 + 1;
                            int _min_38 = ((position_24) < (topk - 1) ? (position_24) : (topk - 1));
                            int clamped_25 = _min_38;
                            int value_26 = indices[row_base + (long long)clamped_25];
                            if (position_24 >= active_0_1) {
                                value_26 = -1;
                            }
                            idx[9] = value_26;
                            int _max_67 = ((largest) > (value_26) ? (largest) : (value_26));
                            largest = _max_67;
                            int _min_39 = ((smallest) < (value_26) ? (smallest) : (value_26));
                            smallest = _min_39;
                            int position_27 = block_1 * 64 + 32 + producer_warp * 4 + 2;
                            int _min_40 = ((position_27) < (topk - 1) ? (position_27) : (topk - 1));
                            int clamped_28 = _min_40;
                            int value_29 = indices[row_base + (long long)clamped_28];
                            if (position_27 >= active_0_1) {
                                value_29 = -1;
                            }
                            idx[10] = value_29;
                            int _max_68 = ((largest) > (value_29) ? (largest) : (value_29));
                            largest = _max_68;
                            int _min_41 = ((smallest) < (value_29) ? (smallest) : (value_29));
                            smallest = _min_41;
                            int position_30 = block_1 * 64 + 32 + producer_warp * 4 + 3;
                            int _min_42 = ((position_30) < (topk - 1) ? (position_30) : (topk - 1));
                            int clamped_31 = _min_42;
                            int value_32 = indices[row_base + (long long)clamped_31];
                            if (position_30 >= active_0_1) {
                                value_32 = -1;
                            }
                            idx[11] = value_32;
                            int _max_69 = ((largest) > (value_32) ? (largest) : (value_32));
                            largest = _max_69;
                            int _min_43 = ((smallest) < (value_32) ? (smallest) : (value_32));
                            smallest = _min_43;
                            int position_33 = block_1 * 64 + 48 + producer_warp * 4;
                            int _min_44 = ((position_33) < (topk - 1) ? (position_33) : (topk - 1));
                            int clamped_34 = _min_44;
                            int value_35 = indices[row_base + (long long)clamped_34];
                            if (position_33 >= active_0_1) {
                                value_35 = -1;
                            }
                            idx[12] = value_35;
                            int _max_70 = ((largest) > (value_35) ? (largest) : (value_35));
                            largest = _max_70;
                            int _min_45 = ((smallest) < (value_35) ? (smallest) : (value_35));
                            smallest = _min_45;
                            int position_36 = block_1 * 64 + 48 + producer_warp * 4 + 1;
                            int _min_46 = ((position_36) < (topk - 1) ? (position_36) : (topk - 1));
                            int clamped_37 = _min_46;
                            int value_38 = indices[row_base + (long long)clamped_37];
                            if (position_36 >= active_0_1) {
                                value_38 = -1;
                            }
                            idx[13] = value_38;
                            int _max_71 = ((largest) > (value_38) ? (largest) : (value_38));
                            largest = _max_71;
                            int _min_47 = ((smallest) < (value_38) ? (smallest) : (value_38));
                            smallest = _min_47;
                            int position_39 = block_1 * 64 + 48 + producer_warp * 4 + 2;
                            int _min_48 = ((position_39) < (topk - 1) ? (position_39) : (topk - 1));
                            int clamped_40 = _min_48;
                            int value_41 = indices[row_base + (long long)clamped_40];
                            if (position_39 >= active_0_1) {
                                value_41 = -1;
                            }
                            idx[14] = value_41;
                            int _max_72 = ((largest) > (value_41) ? (largest) : (value_41));
                            largest = _max_72;
                            int _min_49 = ((smallest) < (value_41) ? (smallest) : (value_41));
                            smallest = _min_49;
                            int position_42 = block_1 * 64 + 48 + producer_warp * 4 + 3;
                            int _min_50 = ((position_42) < (topk - 1) ? (position_42) : (topk - 1));
                            int clamped_43 = _min_50;
                            int value_44 = indices[row_base + (long long)clamped_43];
                            if (position_42 >= active_0_1) {
                                value_44 = -1;
                            }
                            idx[15] = value_44;
                            int _max_73 = ((largest) > (value_44) ? (largest) : (value_44));
                            largest = _max_73;
                            int _min_51 = ((smallest) < (value_44) ? (smallest) : (value_44));
                            smallest = _min_51;
                        }
                        int skip = (smallest == num_kv || largest == -1) && block_1 >= 3;
                        if (block_1 == 1) {
                            mbarrier_wait(q_copied_addr, outer_phase_1);
                        }
                        if (block_1 == 2) {
                            mbarrier_wait(o_written_addr, outer_phase_1);
                        }
                        int slot_1 = cursor_1 % 3;
                        unsigned int phase_1 = cursor_1 / 3 & 1;
                        mbarrier_wait(sv_done_addr + (slot_1) * 8, phase_1 ^ 1);
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048), "l"((&kv_latent)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096), "l"((&kv_latent)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144), "l"((&kv_latent)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (slot_1 * 2) * 8), "r"((uint32_t)(8192)) : "memory");
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "r"((uint32_t)(8192)) : "memory");
                        }
                        cursor_1 = cursor_1 + 1;
                    }
                }
                if (elect_sync()) {
                    if (blocks_1 <= 2) {
                        mbarrier_wait(o_written_addr, outer_phase_1);
                    }
                    mbarrier_arrive(o_written_waited_addr);
                }
                __syncwarp();
                mbarrier_wait(clc_full_addr, outer_phase_1);
                uint32_t _clc_valid_1 = 0;
                uint32_t _clc_ctaid_x_1;
                uint32_t _clc_ctaid_y_1;
                uint32_t _clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_1 == 0) {
                    break;
                }
                query_1 = _clc_ctaid_x_1;
                outer_phase_1 = outer_phase_1 ^ 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int query_2 = blockIdx.x;
            unsigned int outer_phase_2 = 0;
            int cursor_2 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int outer_2 = 0; outer_2 < num_queries; outer_2++) {
                    int active_2 = topk;
                    if (derive_length != 0) {
                        int len_slot_2 = outer_2 & 1;
                        unsigned int len_phase_2 = outer_2 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_2) * 8, len_phase_2);
                        int _max_74 = ((len_cells[len_slot_2]) > (0) ? (len_cells[len_slot_2]) : (0));
                        int _min_52 = ((_max_74) < (topk) ? (_max_74) : (topk));
                        active_2 = _min_52;
                    } else if (has_topk_length != 0) {
                        int _max_75 = ((topk_length[query_2]) > (0) ? (topk_length[query_2]) : (0));
                        int _min_53 = ((_max_75) < (topk) ? (_max_75) : (topk));
                        active_2 = _min_53;
                    }
                    int active_0_2 = active_2;
                    int _max_76 = (((active_0_2 + 63) / 64) > (2) ? ((active_0_2 + 63) / 64) : (2));
                    int blocks_2 = _max_76;
                    int q_slot = (cursor_2 + 1) % 3;
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(rope_addr), "l"((&q_rope)), "r"(0), "r"(0), "r"(0), "r"(query_2),
                           "r"(qr_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qko_addr + (unsigned int)(q_slot * 65536)), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(qr_full_addr, 8192);
                    mbarrier_wait(qr_full_addr, outer_phase_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(rope_addr)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(rope_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 384)), "l"(_tcgen05_cp_desc_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(rope_addr + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(rope_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 384 + 8)), "l"(_tcgen05_cp_desc_1)
                            : "memory");
                    }
                    tcgen05_commit(qr_copied_addr);
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    mbarrier_wait(q_full_addr, outer_phase_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256)), "l"(_tcgen05_cp_desc_2)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 8)), "l"(_tcgen05_cp_desc_3)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 16)), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 24)), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32)), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 8)), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 16)), "l"(_tcgen05_cp_desc_8)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 24)), "l"(_tcgen05_cp_desc_9)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64)), "l"(_tcgen05_cp_desc_10)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 8)), "l"(_tcgen05_cp_desc_11)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 16)), "l"(_tcgen05_cp_desc_12)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 24)), "l"(_tcgen05_cp_desc_13)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96)), "l"(_tcgen05_cp_desc_14)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 8)), "l"(_tcgen05_cp_desc_15)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 16)), "l"(_tcgen05_cp_desc_16)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 24)), "l"(_tcgen05_cp_desc_17)
                            : "memory");
                    }
                    tcgen05_commit(q_copied_addr);
                    #pragma unroll 1
                    for (int block_2 = 0; block_2 < blocks_2 + 1; block_2++) {
                        if (blocks_2 > block_2) {
                            int slot_2 = cursor_2 % 3;
                            unsigned int phase_2 = cursor_2 / 3 & 1;
                            mbarrier_wait(p_free_addr, cursor_2 & 1 ^ 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            mbarrier_wait(kr_ready_addr, cursor_2 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_0 = ((rope_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (400))), "r"(_mma_b_lo_0), "r"(tmem_tmem + 384), "r"(0));
                            tcgen05_commit(qr_done_addr);
                            if (block_2 == 0) {
                                mbarrier_wait(q_copied_addr, outer_phase_2);
                            }
                            mbarrier_arrive_expect_tx(kv_ready_addr + (slot_2 * 2) * 8, 32768);
                            mbarrier_wait(kv_ready_addr + (slot_2 * 2) * 8, phase_2);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_1 = (((k_dual_addr) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (400))), "r"(_mma_b_lo_1), "r"(tmem_tmem + 256), "r"(1));
                            mbarrier_arrive_expect_tx(kv_ready_addr + (slot_2 * 2 + 1) * 8, 32768);
                            mbarrier_wait(kv_ready_addr + (slot_2 * 2 + 1) * 8, phase_2);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_2 = (((k_dual_addr + 32768) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (400))), "r"(_mma_b_lo_2), "r"(tmem_tmem + 320), "r"(1));
                            tcgen05_commit(qk_done_addr + (slot_2) * 8);
                        }
                        if (block_2 > 0) {
                            int slot_3 = (cursor_2 - 1) % 3;
                            mbarrier_wait(so_ready_addr, cursor_2 - 1 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = (((probability_addr) >> 4) & 0x3FFF) | 0x400000;
                            int _mma_b_lo_3 = ((((v_addr) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem), "r"(((block_2 == 1) ? 0 : 1)));
                            int _mma_b_lo_4 = ((((v_addr + 32768) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_4), "r"((tmem_tmem + (128))), "r"(((block_2 == 1) ? 0 : 1)));
                            tcgen05_commit(sv_done_addr + (slot_3) * 8);
                        }
                        cursor_2 = cursor_2 + 1;
                    }
                    cursor_2 = cursor_2 - 1;
                    mbarrier_wait(clc_full_addr, outer_phase_2);
                    uint32_t _clc_valid_2 = 0;
                    uint32_t _clc_ctaid_x_2;
                    uint32_t _clc_ctaid_y_2;
                    uint32_t _clc_ctaid_z_2;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    query_2 = _clc_ctaid_x_2;
                    outer_phase_2 = outer_phase_2 ^ 1;
                }
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 9) {
        { // metadata_main
            if (lane < 8) {
                int query_3 = blockIdx.x;
                unsigned int outer_phase_3 = 0;
                int cursor_3 = 0;
                #pragma unroll 1
                for (int outer_3 = 0; outer_3 < num_queries; outer_3++) {
                    int active_3 = topk;
                    if (derive_length != 0) {
                        int len_slot_3 = outer_3 & 1;
                        unsigned int len_phase_3 = outer_3 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_3) * 8, len_phase_3);
                        int _max_80 = ((len_cells[len_slot_3]) > (0) ? (len_cells[len_slot_3]) : (0));
                        int _min_64 = ((_max_80) < (topk) ? (_max_80) : (topk));
                        active_3 = _min_64;
                    } else if (has_topk_length != 0) {
                        int _max_81 = ((topk_length[query_3]) > (0) ? (topk_length[query_3]) : (0));
                        int _min_65 = ((_max_81) < (topk) ? (_max_81) : (topk));
                        active_3 = _min_65;
                    }
                    int active_0_3 = active_3;
                    int _max_82 = (((active_0_3 + 63) / 64) > (2) ? ((active_0_3 + 63) / 64) : (2));
                    int blocks_3 = _max_82;
                    long long row_base_1 = (long long)indices_offset + (long long)query_3 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int block_3 = 0; block_3 < blocks_3; block_3++) {
                        unsigned int mask = 0;
                        int position_1 = block_3 * 64 + lane * 8;
                        int whole_tile_1 = active_0_3 >= block_3 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (whole_tile_1 != 0) {
                            int _vec_load_4[8];
                            {
                                uint32_t _iv_0_0;
                                uint32_t _iv_0_1;
                                uint32_t _iv_0_2;
                                uint32_t _iv_0_3;
                                uint32_t _iv_0_4;
                                uint32_t _iv_0_5;
                                uint32_t _iv_0_6;
                                uint32_t _iv_0_7;
                                asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(indices + (row_base_1 + (long long)position_1) + (0))) : "memory");
                                _vec_load_4[0 + 0] = (int32_t)_iv_0_0;
                                _vec_load_4[0 + 1] = (int32_t)_iv_0_1;
                                _vec_load_4[0 + 2] = (int32_t)_iv_0_2;
                                _vec_load_4[0 + 3] = (int32_t)_iv_0_3;
                                _vec_load_4[0 + 4] = (int32_t)_iv_0_4;
                                _vec_load_4[0 + 5] = (int32_t)_iv_0_5;
                                _vec_load_4[0 + 6] = (int32_t)_iv_0_6;
                                _vec_load_4[0 + 7] = (int32_t)_iv_0_7;
                            }
                            if (_vec_load_4[0] >= 0 && _vec_load_4[0] < num_kv) {
                                mask = mask | 1;
                            }
                            if (_vec_load_4[1] >= 0 && _vec_load_4[1] < num_kv) {
                                mask = mask | 2;
                            }
                            if (_vec_load_4[2] >= 0 && _vec_load_4[2] < num_kv) {
                                mask = mask | 4;
                            }
                            if (_vec_load_4[3] >= 0 && _vec_load_4[3] < num_kv) {
                                mask = mask | 8;
                            }
                            if (_vec_load_4[4] >= 0 && _vec_load_4[4] < num_kv) {
                                mask = mask | 16;
                            }
                            if (_vec_load_4[5] >= 0 && _vec_load_4[5] < num_kv) {
                                mask = mask | 32;
                            }
                            if (_vec_load_4[6] >= 0 && _vec_load_4[6] < num_kv) {
                                mask = mask | 64;
                            }
                            if (_vec_load_4[7] >= 0 && _vec_load_4[7] < num_kv) {
                                mask = mask | 128;
                            }
                        } else {
                            int _min_66 = ((position_1) < (topk - 1) ? (position_1) : (topk - 1));
                            int clamped_2 = _min_66;
                            int value_1 = indices[row_base_1 + (long long)clamped_2];
                            if (value_1 >= 0 && value_1 < num_kv && active_0_3 > position_1) {
                                mask = mask | 1;
                            }
                            int _min_67 = ((position_1 + 1) < (topk - 1) ? (position_1 + 1) : (topk - 1));
                            int clamped_0 = _min_67;
                            int value_1_1 = indices[row_base_1 + (long long)clamped_0];
                            if (value_1_1 >= 0 && value_1_1 < num_kv && active_0_3 > position_1 + 1) {
                                mask = mask | 2;
                            }
                            int _min_68 = ((position_1 + 2) < (topk - 1) ? (position_1 + 2) : (topk - 1));
                            int clamped_2_1 = _min_68;
                            int value_3 = indices[row_base_1 + (long long)clamped_2_1];
                            if (value_3 >= 0 && value_3 < num_kv && active_0_3 > position_1 + 2) {
                                mask = mask | 4;
                            }
                            int _min_69 = ((position_1 + 3) < (topk - 1) ? (position_1 + 3) : (topk - 1));
                            int clamped_4_1 = _min_69;
                            int value_5_1 = indices[row_base_1 + (long long)clamped_4_1];
                            if (value_5_1 >= 0 && value_5_1 < num_kv && active_0_3 > position_1 + 3) {
                                mask = mask | 8;
                            }
                            int _min_70 = ((position_1 + 4) < (topk - 1) ? (position_1 + 4) : (topk - 1));
                            int clamped_6 = _min_70;
                            int value_7 = indices[row_base_1 + (long long)clamped_6];
                            if (value_7 >= 0 && value_7 < num_kv && active_0_3 > position_1 + 4) {
                                mask = mask | 16;
                            }
                            int _min_71 = ((position_1 + 5) < (topk - 1) ? (position_1 + 5) : (topk - 1));
                            int clamped_8 = _min_71;
                            int value_9 = indices[row_base_1 + (long long)clamped_8];
                            if (value_9 >= 0 && value_9 < num_kv && active_0_3 > position_1 + 5) {
                                mask = mask | 32;
                            }
                            int _min_72 = ((position_1 + 6) < (topk - 1) ? (position_1 + 6) : (topk - 1));
                            int clamped_10_1 = _min_72;
                            int value_11_1 = indices[row_base_1 + (long long)clamped_10_1];
                            if (value_11_1 >= 0 && value_11_1 < num_kv && active_0_3 > position_1 + 6) {
                                mask = mask | 64;
                            }
                            int _min_73 = ((position_1 + 7) < (topk - 1) ? (position_1 + 7) : (topk - 1));
                            int clamped_12 = _min_73;
                            int value_13 = indices[row_base_1 + (long long)clamped_12];
                            if (value_13 >= 0 && value_13 < num_kv && active_0_3 > position_1 + 7) {
                                mask = mask | 128;
                            }
                        }
                        int slot_4 = cursor_3 % 3;
                        unsigned int phase_3 = cursor_3 / 3 & 1;
                        mbarrier_wait(valid_free_addr + (slot_4) * 8, phase_3 ^ 1);
                        validity[slot_4 * 8 + lane] = mask;
                        mbarrier_arrive(valid_ready_addr + (slot_4) * 8);
                        cursor_3 = cursor_3 + 1;
                    }
                    mbarrier_wait(clc_full_addr, outer_phase_3);
                    uint32_t _clc_valid_4 = 0;
                    uint32_t _clc_ctaid_x_4;
                    uint32_t _clc_ctaid_y_4;
                    uint32_t _clc_ctaid_z_4;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_4 == 0) {
                        break;
                    }
                    query_3 = _clc_ctaid_x_4;
                    outer_phase_3 = outer_phase_3 ^ 1;
                }
            } else if (lane == 8) {
                unsigned int outer_phase_4 = 0;
                #pragma unroll 1
                for (int outer_4 = 0; outer_4 < num_queries; outer_4++) {
                    mbarrier_wait(clc_empty_addr, outer_phase_4 ^ 1);
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(clc_payload_addr + 0 * 16 + 0 * 16), "r"(clc_full_addr + 0 * 8)
                        : "memory");
                    mbarrier_arrive_expect_tx(clc_full_addr, 16);
                    mbarrier_wait(clc_full_addr, outer_phase_4);
                    uint32_t _clc_valid_5 = 0;
                    uint32_t _clc_ctaid_x_5;
                    uint32_t _clc_ctaid_y_5;
                    uint32_t _clc_ctaid_z_5;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_5 == 0) {
                        break;
                    }
                    outer_phase_4 = outer_phase_4 ^ 1;
                }
            } else {
                if (lane >= 16) {
                    int query_4 = blockIdx.x;
                    unsigned int outer_phase_5 = 0;
                    int lane16 = lane - 16;
                    int scan_aligned = ((idx_stride | indices_offset) & 7) == 0;
                    #pragma unroll 1
                    for (int outer_5 = 0; outer_5 < num_queries; outer_5++) {
                        if (derive_length != 0) {
                            long long row_base_2 = (long long)indices_offset + (long long)query_4 * (long long)idx_stride;
                            int last = 0;
                            int n_full = topk / 128;
                            if (scan_aligned == 0) {
                                n_full = 0;
                            }
                            #pragma unroll 4
                            for (int g = 0; g < n_full; g++) {
                                int spos = g * 128 + lane16 * 8;
                                int _vec_load_5[8];
                                {
                                    uint32_t _iv_1_0;
                                    uint32_t _iv_1_1;
                                    uint32_t _iv_1_2;
                                    uint32_t _iv_1_3;
                                    uint32_t _iv_1_4;
                                    uint32_t _iv_1_5;
                                    uint32_t _iv_1_6;
                                    uint32_t _iv_1_7;
                                    asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos) + (0))) : "memory");
                                    _vec_load_5[0 + 0] = (int32_t)_iv_1_0;
                                    _vec_load_5[0 + 1] = (int32_t)_iv_1_1;
                                    _vec_load_5[0 + 2] = (int32_t)_iv_1_2;
                                    _vec_load_5[0 + 3] = (int32_t)_iv_1_3;
                                    _vec_load_5[0 + 4] = (int32_t)_iv_1_4;
                                    _vec_load_5[0 + 5] = (int32_t)_iv_1_5;
                                    _vec_load_5[0 + 6] = (int32_t)_iv_1_6;
                                    _vec_load_5[0 + 7] = (int32_t)_iv_1_7;
                                }
                                if (_vec_load_5[0] >= 0 && _vec_load_5[0] < num_kv) {
                                    last = spos + 1;
                                }
                                if (_vec_load_5[1] >= 0 && _vec_load_5[1] < num_kv) {
                                    last = spos + 1 + 1;
                                }
                                if (_vec_load_5[2] >= 0 && _vec_load_5[2] < num_kv) {
                                    last = spos + 2 + 1;
                                }
                                if (_vec_load_5[3] >= 0 && _vec_load_5[3] < num_kv) {
                                    last = spos + 3 + 1;
                                }
                                if (_vec_load_5[4] >= 0 && _vec_load_5[4] < num_kv) {
                                    last = spos + 4 + 1;
                                }
                                if (_vec_load_5[5] >= 0 && _vec_load_5[5] < num_kv) {
                                    last = spos + 5 + 1;
                                }
                                if (_vec_load_5[6] >= 0 && _vec_load_5[6] < num_kv) {
                                    last = spos + 6 + 1;
                                }
                                if (_vec_load_5[7] >= 0 && _vec_load_5[7] < num_kv) {
                                    last = spos + 7 + 1;
                                }
                            }
                            int tail0 = n_full * 128 + lane16;
                            int n_tail = (topk - n_full * 128 - lane16 + 15) / 16;
                            #pragma unroll 1
                            for (int t = 0; t < n_tail; t++) {
                                int pos = tail0 + t * 16;
                                int value_4 = indices[row_base_2 + (long long)pos];
                                if (value_4 >= 0 && value_4 < num_kv) {
                                    last = pos + 1;
                                }
                            }
                            int part = last;
                            int _shfl_xor_0 = __shfl_xor_sync(4294901760u, part, 8);
                            int peer_1 = _shfl_xor_0;
                            int _max_83 = ((part) > (peer_1) ? (part) : (peer_1));
                            part = _max_83;
                            int _shfl_xor_1 = __shfl_xor_sync(4294901760u, part, 4);
                            int peer_0 = _shfl_xor_1;
                            int _max_84 = ((part) > (peer_0) ? (part) : (peer_0));
                            part = _max_84;
                            int _shfl_xor_2 = __shfl_xor_sync(4294901760u, part, 2);
                            int peer_1_1 = _shfl_xor_2;
                            int _max_85 = ((part) > (peer_1_1) ? (part) : (peer_1_1));
                            part = _max_85;
                            int _shfl_xor_3 = __shfl_xor_sync(4294901760u, part, 1);
                            int peer_2 = _shfl_xor_3;
                            int _max_86 = ((part) > (peer_2) ? (part) : (peer_2));
                            part = _max_86;
                            if (lane == 16) {
                                len_cells[outer_5 & 1] = part;
                                topk_length[(long long)query_4] = part;
                                mbarrier_arrive(len_ready_addr + (outer_5 & 1) * 8);
                            }
                        }
                        mbarrier_wait(clc_full_addr, outer_phase_5);
                        uint32_t _clc_valid_6 = 0;
                        uint32_t _clc_ctaid_x_6;
                        uint32_t _clc_ctaid_y_6;
                        uint32_t _clc_ctaid_z_6;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_6), "=r"(_clc_ctaid_y_6), "=r"(_clc_ctaid_z_6), "=r"(_clc_valid_6)
                            : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                            : "memory");
                        mbarrier_arrive(clc_empty_addr);
                        if (_clc_valid_6 == 0) {
                            break;
                        }
                        query_4 = _clc_ctaid_x_6;
                        outer_phase_5 = outer_phase_5 ^ 1;
                    }
                }
            }
        }
    }
    // ---- Role: rope_load ----
    if (warp >= 10 && warp <= 11) {
        { // rope_load_main
            int query_5 = blockIdx.x;
            unsigned int outer_phase_6 = 0;
            int cursor_4 = 0;
            int thread = (warp - 10) * 32 + lane;
            int group = thread / 8;
            int col = thread % 8 * 8;
            #pragma unroll 1
            for (int outer_6 = 0; outer_6 < num_queries; outer_6++) {
                int active_4 = topk;
                if (derive_length != 0) {
                    int len_slot_4 = outer_6 & 1;
                    unsigned int len_phase_4 = outer_6 >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot_4) * 8, len_phase_4);
                    int _max_77 = ((len_cells[len_slot_4]) > (0) ? (len_cells[len_slot_4]) : (0));
                    int _min_54 = ((_max_77) < (topk) ? (_max_77) : (topk));
                    active_4 = _min_54;
                } else if (has_topk_length != 0) {
                    int _max_78 = ((topk_length[query_5]) > (0) ? (topk_length[query_5]) : (0));
                    int _min_55 = ((_max_78) < (topk) ? (_max_78) : (topk));
                    active_4 = _min_55;
                }
                int active_0_4 = active_4;
                int _max_79 = (((active_0_4 + 63) / 64) > (2) ? ((active_0_4 + 63) / 64) : (2));
                int blocks_4 = _max_79;
                long long row_base_3 = (long long)indices_offset + (long long)query_5 * (long long)idx_stride;
                #pragma unroll 1
                for (int block_4 = 0; block_4 < blocks_4; block_4++) {
                    int index_values[8];
                    int position_2 = block_4 * 64 + group;
                    int _min_56 = ((position_2) < (topk - 1) ? (position_2) : (topk - 1));
                    int clamped_3 = _min_56;
                    int value_6 = indices[row_base_3 + (long long)clamped_3];
                    if (position_2 >= active_0_4) {
                        value_6 = -1;
                    }
                    index_values[0] = value_6;
                    int position_0_1 = block_4 * 64 + group + 8;
                    int _min_57 = ((position_0_1) < (topk - 1) ? (position_0_1) : (topk - 1));
                    int clamped_1_1 = _min_57;
                    int value_2_1 = indices[row_base_3 + (long long)clamped_1_1];
                    if (position_0_1 >= active_0_4) {
                        value_2_1 = -1;
                    }
                    index_values[1] = value_2_1;
                    int position_3_1 = block_4 * 64 + group + 16;
                    int _min_58 = ((position_3_1) < (topk - 1) ? (position_3_1) : (topk - 1));
                    int clamped_4_2 = _min_58;
                    int value_5_2 = indices[row_base_3 + (long long)clamped_4_2];
                    if (position_3_1 >= active_0_4) {
                        value_5_2 = -1;
                    }
                    index_values[2] = value_5_2;
                    int position_6_1 = block_4 * 64 + group + 24;
                    int _min_59 = ((position_6_1) < (topk - 1) ? (position_6_1) : (topk - 1));
                    int clamped_7_1 = _min_59;
                    int value_8_1 = indices[row_base_3 + (long long)clamped_7_1];
                    if (position_6_1 >= active_0_4) {
                        value_8_1 = -1;
                    }
                    index_values[3] = value_8_1;
                    int position_9_1 = block_4 * 64 + group + 32;
                    int _min_60 = ((position_9_1) < (topk - 1) ? (position_9_1) : (topk - 1));
                    int clamped_10_2 = _min_60;
                    int value_11_2 = indices[row_base_3 + (long long)clamped_10_2];
                    if (position_9_1 >= active_0_4) {
                        value_11_2 = -1;
                    }
                    index_values[4] = value_11_2;
                    int position_12_1 = block_4 * 64 + group + 40;
                    int _min_61 = ((position_12_1) < (topk - 1) ? (position_12_1) : (topk - 1));
                    int clamped_13_1 = _min_61;
                    int value_14_1 = indices[row_base_3 + (long long)clamped_13_1];
                    if (position_12_1 >= active_0_4) {
                        value_14_1 = -1;
                    }
                    index_values[5] = value_14_1;
                    int position_15_1 = block_4 * 64 + group + 48;
                    int _min_62 = ((position_15_1) < (topk - 1) ? (position_15_1) : (topk - 1));
                    int clamped_16_1 = _min_62;
                    int value_17_1 = indices[row_base_3 + (long long)clamped_16_1];
                    if (position_15_1 >= active_0_4) {
                        value_17_1 = -1;
                    }
                    index_values[6] = value_17_1;
                    int position_18_1 = block_4 * 64 + group + 56;
                    int _min_63 = ((position_18_1) < (topk - 1) ? (position_18_1) : (topk - 1));
                    int clamped_19_1 = _min_63;
                    int value_20_1 = indices[row_base_3 + (long long)clamped_19_1];
                    if (position_18_1 >= active_0_4) {
                        value_20_1 = -1;
                    }
                    index_values[7] = value_20_1;
                    mbarrier_wait(qr_done_addr, cursor_4 & 1 ^ 1);
                    if (block_4 == 0) {
                        mbarrier_wait(qr_copied_addr, outer_phase_6);
                    }
                    int raw_offset = (col / 32 * 2048 + group * 32 + col % 32) * 2;
                    int swizzled_offset_1 = raw_offset ^ (raw_offset >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[0] * (long long)k_rope_stride + (long long)col)), "r"((index_values[0] >= 0 && index_values[0] < num_kv) ? 16 : 0));
                    int raw_offset_21 = (col / 32 * 2048 + (group + 8) * 32 + col % 32) * 2;
                    int swizzled_offset_22 = raw_offset_21 ^ (raw_offset_21 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_22), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[1] * (long long)k_rope_stride + (long long)col)), "r"((index_values[1] >= 0 && index_values[1] < num_kv) ? 16 : 0));
                    int raw_offset_23 = (col / 32 * 2048 + (group + 16) * 32 + col % 32) * 2;
                    int swizzled_offset_24_1 = raw_offset_23 ^ (raw_offset_23 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_24_1), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[2] * (long long)k_rope_stride + (long long)col)), "r"((index_values[2] >= 0 && index_values[2] < num_kv) ? 16 : 0));
                    int raw_offset_25 = (col / 32 * 2048 + (group + 24) * 32 + col % 32) * 2;
                    int swizzled_offset_26 = raw_offset_25 ^ (raw_offset_25 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_26), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[3] * (long long)k_rope_stride + (long long)col)), "r"((index_values[3] >= 0 && index_values[3] < num_kv) ? 16 : 0));
                    int raw_offset_27 = (col / 32 * 2048 + (group + 32) * 32 + col % 32) * 2;
                    int swizzled_offset_28 = raw_offset_27 ^ (raw_offset_27 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_28), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[4] * (long long)k_rope_stride + (long long)col)), "r"((index_values[4] >= 0 && index_values[4] < num_kv) ? 16 : 0));
                    int raw_offset_29 = (col / 32 * 2048 + (group + 40) * 32 + col % 32) * 2;
                    int swizzled_offset_30 = raw_offset_29 ^ (raw_offset_29 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_30), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[5] * (long long)k_rope_stride + (long long)col)), "r"((index_values[5] >= 0 && index_values[5] < num_kv) ? 16 : 0));
                    int raw_offset_31 = (col / 32 * 2048 + (group + 48) * 32 + col % 32) * 2;
                    int swizzled_offset_32_1 = raw_offset_31 ^ (raw_offset_31 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_32_1), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[6] * (long long)k_rope_stride + (long long)col)), "r"((index_values[6] >= 0 && index_values[6] < num_kv) ? 16 : 0));
                    int raw_offset_33 = (col / 32 * 2048 + (group + 56) * 32 + col % 32) * 2;
                    int swizzled_offset_34 = raw_offset_33 ^ (raw_offset_33 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_34), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[7] * (long long)k_rope_stride + (long long)col)), "r"((index_values[7] >= 0 && index_values[7] < num_kv) ? 16 : 0));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(kr_ready_addr) : "memory");
                    cursor_4 = cursor_4 + 1;
                }
                mbarrier_wait(clc_full_addr, outer_phase_6);
                uint32_t _clc_valid_3 = 0;
                uint32_t _clc_ctaid_x_3;
                uint32_t _clc_ctaid_y_3;
                uint32_t _clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_3 == 0) {
                    break;
                }
                query_5 = _clc_ctaid_x_3;
                outer_phase_6 = outer_phase_6 ^ 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
