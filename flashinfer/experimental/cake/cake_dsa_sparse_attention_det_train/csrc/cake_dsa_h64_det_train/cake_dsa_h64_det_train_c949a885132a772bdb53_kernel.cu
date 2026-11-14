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




__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
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
kernel_cake_dsa_h64_det_train_c949a885132a772bdb53(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap kv_latent, __nv_bfloat16* __restrict__ k_rope, const __grid_constant__ CUtensorMap out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int k_rope_stride, int k_rope_offset, int has_topk_length, int derive_length, float scale_log2)
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
    #define kv_ready_lo_addr (mbar_base + 96)
    #define kv_ready_hi_addr (mbar_base + 120)
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

    // Prologue
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

    // Kernel setup ops
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }

    // Mbarrier init (32 barriers at smem_raw[0..256))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'clc_pipeline' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_copied: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
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
            // --- pipeline 'score_pipe' ---
            // qk_done: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sv_done: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // kv_ready_lo: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // kv_ready_hi: 3 barriers, init_count=1
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
            // --- pipeline 'clc_pipeline' ---
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

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem = taddr;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: softmax ----
    if (warp <= 3) {
        { // softmax_main
            int query = blockIdx.x;
            unsigned int outer_phase = 0;
            int cursor = 0;
            unsigned int _phase_o_written_waited = 0;
            #pragma unroll 1
            for (int outer = 0; outer < num_queries; outer++) {
                int active = topk;
                if (derive_length != 0) {
                    int len_slot = outer % 2;
                    int _stage_len_ready = (int)((unsigned int)outer % 2);
                    int _parity_len_ready = (int)((unsigned int)outer / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready) * 8, _parity_len_ready);
                    int _max_1 = ((len_cells[len_slot]) > (0) ? (len_cells[len_slot]) : (0));
                    int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
                    active = _min_0;
                } else if (has_topk_length != 0) {
                    int _max_3 = ((topk_length[query]) > (0) ? (topk_length[query]) : (0));
                    int _min_2 = ((_max_3) < (topk) ? (_max_3) : (topk));
                    active = _min_2;
                }
                int active_0 = active;
                int blocks = (((active_0 + 63) / 64) > (2) ? ((active_0 + 63) / 64) : (2));
                int row = warp * 32 + lane;
                unsigned int tmrow = tmem_tmem;
                float mi = -1e+30f;
                float li = 0.0f;
                float real_mi = -CUDART_INF_F;
                asm volatile("cp.async.bulk.wait_group 0;");
                mbarrier_arrive(o_written_addr);
                #pragma unroll 1
                for (int block = 0; block < blocks; block++) {
                    int slot = cursor % 3;
                    asm volatile("barrier.sync %0, 64;" :: "r"(1 + (warp & 1)) : "memory");
                    int _stage_qk_done = (int)((unsigned int)cursor % 3);
                    int _parity_qk_done = (int)((unsigned int)cursor / 3 % 2);
                    mbarrier_wait(qk_done_addr + (_stage_qk_done) * 8, _parity_qk_done);
                    int _stage_valid_ready = (int)((unsigned int)cursor % 3);
                    int _parity_valid_ready = (int)((unsigned int)cursor / 3 % 2);
                    mbarrier_wait(valid_ready_addr + (_stage_valid_ready) * 8, _parity_valid_ready);
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
                        p[0] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 1 & 1) == 0) {
                        p[1] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 2 & 1) == 0) {
                        p[2] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 3 & 1) == 0) {
                        p[3] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 4 & 1) == 0) {
                        p[4] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 5 & 1) == 0) {
                        p[5] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 6 & 1) == 0) {
                        p[6] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 7 & 1) == 0) {
                        p[7] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 8 & 1) == 0) {
                        p[8] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 9 & 1) == 0) {
                        p[9] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 10 & 1) == 0) {
                        p[10] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 11 & 1) == 0) {
                        p[11] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 12 & 1) == 0) {
                        p[12] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 13 & 1) == 0) {
                        p[13] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 14 & 1) == 0) {
                        p[14] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 15 & 1) == 0) {
                        p[15] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 16 & 1) == 0) {
                        p[16] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 17 & 1) == 0) {
                        p[17] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 18 & 1) == 0) {
                        p[18] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 19 & 1) == 0) {
                        p[19] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 20 & 1) == 0) {
                        p[20] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 21 & 1) == 0) {
                        p[21] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 22 & 1) == 0) {
                        p[22] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 23 & 1) == 0) {
                        p[23] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 24 & 1) == 0) {
                        p[24] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 25 & 1) == 0) {
                        p[25] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 26 & 1) == 0) {
                        p[26] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 27 & 1) == 0) {
                        p[27] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 28 & 1) == 0) {
                        p[28] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 29 & 1) == 0) {
                        p[29] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 30 & 1) == 0) {
                        p[30] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 31 & 1) == 0) {
                        p[31] = -CUDART_INF_F;
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
                    float smem_regs[4];
                    #pragma unroll
                    for (int _lr = 0; _lr < 4; _lr++) {
                        smem_regs[_lr] = exchange[warp * 1024 + lane * 4 + _lr];
                    }
                    float2 pair_sum_1_0 = add_f32x2(make_float2(p[0], p[1]), make_float2(smem_regs[0], smem_regs[1]));
                    p[0] = pair_sum_1_0.x;
                    p[1] = pair_sum_1_0.y;
                    float2 pair_sum_1_1 = add_f32x2(make_float2(p[2], p[3]), make_float2(smem_regs[2], smem_regs[3]));
                    p[2] = pair_sum_1_1.x;
                    p[3] = pair_sum_1_1.y;
                    float smem_regs_0[4];
                    #pragma unroll
                    for (int _lr_1 = 0; _lr_1 < 4; _lr_1++) {
                        smem_regs_0[_lr_1] = exchange[warp * 1024 + 128 + lane * 4 + _lr_1];
                    }
                    float2 pair_sum_1_2 = add_f32x2(make_float2(p[4], p[5]), make_float2(smem_regs_0[0], smem_regs_0[1]));
                    p[4] = pair_sum_1_2.x;
                    p[5] = pair_sum_1_2.y;
                    float2 pair_sum_1_3 = add_f32x2(make_float2(p[6], p[7]), make_float2(smem_regs_0[2], smem_regs_0[3]));
                    p[6] = pair_sum_1_3.x;
                    p[7] = pair_sum_1_3.y;
                    float smem_regs_1[4];
                    #pragma unroll
                    for (int _lr_2 = 0; _lr_2 < 4; _lr_2++) {
                        smem_regs_1[_lr_2] = exchange[warp * 1024 + 256 + lane * 4 + _lr_2];
                    }
                    float2 pair_sum_1_4 = add_f32x2(make_float2(p[8], p[9]), make_float2(smem_regs_1[0], smem_regs_1[1]));
                    p[8] = pair_sum_1_4.x;
                    p[9] = pair_sum_1_4.y;
                    float2 pair_sum_1_5 = add_f32x2(make_float2(p[10], p[11]), make_float2(smem_regs_1[2], smem_regs_1[3]));
                    p[10] = pair_sum_1_5.x;
                    p[11] = pair_sum_1_5.y;
                    float smem_regs_3[4];
                    #pragma unroll
                    for (int _lr_3 = 0; _lr_3 < 4; _lr_3++) {
                        smem_regs_3[_lr_3] = exchange[warp * 1024 + 384 + lane * 4 + _lr_3];
                    }
                    float2 pair_sum_1_6 = add_f32x2(make_float2(p[12], p[13]), make_float2(smem_regs_3[0], smem_regs_3[1]));
                    p[12] = pair_sum_1_6.x;
                    p[13] = pair_sum_1_6.y;
                    float2 pair_sum_1_7 = add_f32x2(make_float2(p[14], p[15]), make_float2(smem_regs_3[2], smem_regs_3[3]));
                    p[14] = pair_sum_1_7.x;
                    p[15] = pair_sum_1_7.y;
                    float smem_regs_4[4];
                    #pragma unroll
                    for (int _lr_4 = 0; _lr_4 < 4; _lr_4++) {
                        smem_regs_4[_lr_4] = exchange[warp * 1024 + 512 + lane * 4 + _lr_4];
                    }
                    float2 pair_sum_1_8 = add_f32x2(make_float2(p[16], p[17]), make_float2(smem_regs_4[0], smem_regs_4[1]));
                    p[16] = pair_sum_1_8.x;
                    p[17] = pair_sum_1_8.y;
                    float2 pair_sum_1_9 = add_f32x2(make_float2(p[18], p[19]), make_float2(smem_regs_4[2], smem_regs_4[3]));
                    p[18] = pair_sum_1_9.x;
                    p[19] = pair_sum_1_9.y;
                    float smem_regs_5[4];
                    #pragma unroll
                    for (int _lr_5 = 0; _lr_5 < 4; _lr_5++) {
                        smem_regs_5[_lr_5] = exchange[warp * 1024 + 640 + lane * 4 + _lr_5];
                    }
                    float2 pair_sum_1_10 = add_f32x2(make_float2(p[20], p[21]), make_float2(smem_regs_5[0], smem_regs_5[1]));
                    p[20] = pair_sum_1_10.x;
                    p[21] = pair_sum_1_10.y;
                    float2 pair_sum_1_11 = add_f32x2(make_float2(p[22], p[23]), make_float2(smem_regs_5[2], smem_regs_5[3]));
                    p[22] = pair_sum_1_11.x;
                    p[23] = pair_sum_1_11.y;
                    float smem_regs_6[4];
                    #pragma unroll
                    for (int _lr_6 = 0; _lr_6 < 4; _lr_6++) {
                        smem_regs_6[_lr_6] = exchange[warp * 1024 + 768 + lane * 4 + _lr_6];
                    }
                    float2 pair_sum_1_12 = add_f32x2(make_float2(p[24], p[25]), make_float2(smem_regs_6[0], smem_regs_6[1]));
                    p[24] = pair_sum_1_12.x;
                    p[25] = pair_sum_1_12.y;
                    float2 pair_sum_1_13 = add_f32x2(make_float2(p[26], p[27]), make_float2(smem_regs_6[2], smem_regs_6[3]));
                    p[26] = pair_sum_1_13.x;
                    p[27] = pair_sum_1_13.y;
                    float smem_regs_7[4];
                    #pragma unroll
                    for (int _lr_7 = 0; _lr_7 < 4; _lr_7++) {
                        smem_regs_7[_lr_7] = exchange[warp * 1024 + 896 + lane * 4 + _lr_7];
                    }
                    float2 pair_sum_1_14 = add_f32x2(make_float2(p[28], p[29]), make_float2(smem_regs_7[0], smem_regs_7[1]));
                    p[28] = pair_sum_1_14.x;
                    p[29] = pair_sum_1_14.y;
                    float2 pair_sum_1_15 = add_f32x2(make_float2(p[30], p[31]), make_float2(smem_regs_7[2], smem_regs_7[3]));
                    p[30] = pair_sum_1_15.x;
                    p[31] = pair_sum_1_15.y;
                    mbarrier_arrive(valid_free_addr + (slot) * 8);
                    float local_max0 = -CUDART_INF_F;
                    float local_max1 = -CUDART_INF_F;
                    local_max0 = max_noftz(local_max0, p[0]);
                    local_max0 = max_noftz(local_max0, p[1]);
                    local_max0 = max_noftz(local_max0, p[2]);
                    local_max0 = max_noftz(local_max0, p[3]);
                    local_max0 = max_noftz(local_max0, p[4]);
                    local_max0 = max_noftz(local_max0, p[5]);
                    local_max0 = max_noftz(local_max0, p[6]);
                    local_max0 = max_noftz(local_max0, p[7]);
                    local_max0 = max_noftz(local_max0, p[8]);
                    local_max0 = max_noftz(local_max0, p[9]);
                    local_max0 = max_noftz(local_max0, p[10]);
                    local_max0 = max_noftz(local_max0, p[11]);
                    local_max0 = max_noftz(local_max0, p[12]);
                    local_max0 = max_noftz(local_max0, p[13]);
                    local_max0 = max_noftz(local_max0, p[14]);
                    local_max0 = max_noftz(local_max0, p[15]);
                    local_max1 = max_noftz(local_max1, p[16]);
                    local_max1 = max_noftz(local_max1, p[17]);
                    local_max1 = max_noftz(local_max1, p[18]);
                    local_max1 = max_noftz(local_max1, p[19]);
                    local_max1 = max_noftz(local_max1, p[20]);
                    local_max1 = max_noftz(local_max1, p[21]);
                    local_max1 = max_noftz(local_max1, p[22]);
                    local_max1 = max_noftz(local_max1, p[23]);
                    local_max1 = max_noftz(local_max1, p[24]);
                    local_max1 = max_noftz(local_max1, p[25]);
                    local_max1 = max_noftz(local_max1, p[26]);
                    local_max1 = max_noftz(local_max1, p[27]);
                    local_max1 = max_noftz(local_max1, p[28]);
                    local_max1 = max_noftz(local_max1, p[29]);
                    local_max1 = max_noftz(local_max1, p[30]);
                    local_max1 = max_noftz(local_max1, p[31]);
                    float cur_max = max_noftz(local_max0, local_max1) * scale_log2;
                    int rescale = 0;
                    stats[row] = cur_max;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    cur_max = max_noftz(cur_max, stats[row ^ 64]);
                    real_mi = max_noftz(real_mi, cur_max);
                    int _warp_vote_4 = __any_sync(0xFFFFFFFF, cur_max - mi > 6.0f);
                    rescale = _warp_vote_4;
                    float factor = 1.0f;
                    float next_max = mi;
                    if (rescale != 0) {
                        next_max = max_noftz(cur_max, mi);
                        float _exp2_5 = approx_exp2(mi - next_max);
                        factor = _exp2_5;
                    }
                    mi = next_max;
                    const float2 _fma_b2_6 = {scale_log2, scale_log2};
                    const float2 _fma_c2_7 = {-mi, -mi};
                    float2 _fma_pair_8 = fma_f32x2(make_float2(p[0], p[1]), _fma_b2_6, _fma_c2_7);
                    p[0] = _fma_pair_8.x;
                    p[1] = _fma_pair_8.y;
                    float2 _fma_pair_9 = fma_f32x2(make_float2(p[2], p[3]), _fma_b2_6, _fma_c2_7);
                    p[2] = _fma_pair_9.x;
                    p[3] = _fma_pair_9.y;
                    float2 _fma_pair_10 = fma_f32x2(make_float2(p[4], p[5]), _fma_b2_6, _fma_c2_7);
                    p[4] = _fma_pair_10.x;
                    p[5] = _fma_pair_10.y;
                    float2 _fma_pair_11 = fma_f32x2(make_float2(p[6], p[7]), _fma_b2_6, _fma_c2_7);
                    p[6] = _fma_pair_11.x;
                    p[7] = _fma_pair_11.y;
                    float2 _fma_pair_12 = fma_f32x2(make_float2(p[8], p[9]), _fma_b2_6, _fma_c2_7);
                    p[8] = _fma_pair_12.x;
                    p[9] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(p[10], p[11]), _fma_b2_6, _fma_c2_7);
                    p[10] = _fma_pair_13.x;
                    p[11] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(p[12], p[13]), _fma_b2_6, _fma_c2_7);
                    p[12] = _fma_pair_14.x;
                    p[13] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(p[14], p[15]), _fma_b2_6, _fma_c2_7);
                    p[14] = _fma_pair_15.x;
                    p[15] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(p[16], p[17]), _fma_b2_6, _fma_c2_7);
                    p[16] = _fma_pair_16.x;
                    p[17] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(p[18], p[19]), _fma_b2_6, _fma_c2_7);
                    p[18] = _fma_pair_17.x;
                    p[19] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(p[20], p[21]), _fma_b2_6, _fma_c2_7);
                    p[20] = _fma_pair_18.x;
                    p[21] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(p[22], p[23]), _fma_b2_6, _fma_c2_7);
                    p[22] = _fma_pair_19.x;
                    p[23] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(p[24], p[25]), _fma_b2_6, _fma_c2_7);
                    p[24] = _fma_pair_20.x;
                    p[25] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(p[26], p[27]), _fma_b2_6, _fma_c2_7);
                    p[26] = _fma_pair_21.x;
                    p[27] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(p[28], p[29]), _fma_b2_6, _fma_c2_7);
                    p[28] = _fma_pair_22.x;
                    p[29] = _fma_pair_22.y;
                    float2 _fma_pair_23 = fma_f32x2(make_float2(p[30], p[31]), _fma_b2_6, _fma_c2_7);
                    p[30] = _fma_pair_23.x;
                    p[31] = _fma_pair_23.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        p[_le] = approx_exp2(p[_le]);
                    }
                    float2 sum_pair = make_float2(0.0f, 0.0f);
                    sum_pair = add_f32x2(sum_pair, make_float2(p[0], p[1]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[2], p[3]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[4], p[5]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[6], p[7]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[8], p[9]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[10], p[11]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[12], p[13]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[14], p[15]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[16], p[17]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[18], p[19]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[20], p[21]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[22], p[23]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[24], p[25]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[26], p[27]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[28], p[29]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[30], p[31]));
                    float cur_sum = sum_pair.x + sum_pair.y;
                    li = __fmaf_rn(li, factor, cur_sum);
                    if (block > 0) {
                        int _stage_sv_done = (int)((unsigned int)(cursor - 1) % 3);
                        int _parity_sv_done = (int)((unsigned int)(cursor - 1) / 3 % 2);
                        mbarrier_wait(sv_done_addr + (_stage_sv_done) * 8, _parity_sv_done);
                    }
                    unsigned int p_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                        p_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048) * 2)), "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]) : "memory");
                    unsigned int p_bf16_r_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 8], p[_lp*2+1 + 8]));
                        p_bf16_r_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 512) * 2)), "r"(p_bf16_r_0[0]), "r"(p_bf16_r_0[1]), "r"(p_bf16_r_0[2]), "r"(p_bf16_r_0[3]) : "memory");
                    unsigned int p_bf16_r_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 16], p[_lp*2+1 + 16]));
                        p_bf16_r_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1024) * 2)), "r"(p_bf16_r_1[0]), "r"(p_bf16_r_1[1]), "r"(p_bf16_r_1[2]), "r"(p_bf16_r_1[3]) : "memory");
                    unsigned int p_bf16_r_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 24], p[_lp*2+1 + 24]));
                        p_bf16_r_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1536) * 2)), "r"(p_bf16_r_2[0]), "r"(p_bf16_r_2[1]), "r"(p_bf16_r_2[2]), "r"(p_bf16_r_2[3]) : "memory");
                    if (block > 0 && rescale != 0) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(tmrow));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_24 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_24);
                        tmem_st_x32_f32(tmrow, _tmem_load_0);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(tmrow + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_25 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_25);
                        tmem_st_x32_f32(tmrow + 32, _tmem_load_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(tmrow + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_26 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_26);
                        tmem_st_x32_f32(tmrow + 64, _tmem_load_2);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(tmrow + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_27 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_27);
                        tmem_st_x32_f32(tmrow + 96, _tmem_load_3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                            : "r"(tmrow + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_28 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_28);
                        tmem_st_x32_f32(tmrow + 128, _tmem_load_4);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(tmrow + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_29 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_29);
                        tmem_st_x32_f32(tmrow + 160, _tmem_load_5);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(tmrow + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_30 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_30);
                        tmem_st_x32_f32(tmrow + 192, _tmem_load_6);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                            : "r"(tmrow + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_31 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_31);
                        tmem_st_x32_f32(tmrow + 224, _tmem_load_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(so_ready_addr);
                    cursor += 1;
                }
                mbarrier_wait(o_written_waited_addr, _phase_o_written_waited);
                if (real_mi == -CUDART_INF_F) {
                    li = 0.0f;
                    mi = -CUDART_INF_F;
                }
                stats[128 + row] = li;
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                li = li + stats[128 + (row ^ 64)];
                if (row < 64) {
                    float _log_32 = logf(li);
                    float cur_lse = __fmaf_rn(mi, 0.6931472f, _log_32);
                    if (real_mi == -CUDART_INF_F) {
                        cur_lse = -CUDART_INF_F;
                    }
                    lse[query * 64 + row] = cur_lse;
                }
                float output_scale = 1.0f / li;
                int have_valid = __any_sync(0xFFFFFFFF, li != 0.0f);
                if (have_valid == 0) {
                    output_scale = 1.0f;
                }
                int _stage_sv_done_1 = (int)((unsigned int)(cursor - 1) % 3);
                int _parity_sv_done_1 = (int)((unsigned int)(cursor - 1) / 3 % 2);
                mbarrier_wait(sv_done_addr + (_stage_sv_done_1) * 8, _parity_sv_done_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int o_slot = (cursor + 2) % 3;
                int head = row % 64;
                int query_head = query * 64 + head;
                long long o_row_base = (long long)query_head * 512ll;
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
                const float2 _scale2_33 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_33);
                int out_col = row / 64 * 128;
                float res[16];
                unsigned int ov_bf16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 0], ov[_lp*2+1 + 0]));
                    ov_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_0[_pair]) << 16);
                    (&unpacked_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_0[_pair]) & 0xffff0000u);
                }
                res[0] = ov[0] - unpacked_0[0];
                res[1] = ov[1] - unpacked_0[1];
                res[2] = ov[2] - unpacked_0[2];
                res[3] = ov[3] - unpacked_0[3];
                res[4] = ov[4] - unpacked_0[4];
                res[5] = ov[5] - unpacked_0[5];
                res[6] = ov[6] - unpacked_0[6];
                res[7] = ov[7] - unpacked_0[7];
                int byte_offset_0 = o_slot * 65536 + out_col * 128 + row % 64 * 128;
                int swizzled_offset_0 = byte_offset_0 ^ (byte_offset_0 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_0), "r"(ov_bf16_0[0]), "r"(ov_bf16_0[1]), "r"(ov_bf16_0[2]), "r"(ov_bf16_0[3]) : "memory");
                unsigned int ov_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 8], ov[_lp*2+1 + 8]));
                    ov_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1[_pair]) << 16);
                    (&unpacked_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1[_pair]) & 0xffff0000u);
                }
                res[8] = ov[8] - unpacked_1[0];
                res[9] = ov[9] - unpacked_1[1];
                res[10] = ov[10] - unpacked_1[2];
                res[11] = ov[11] - unpacked_1[3];
                res[12] = ov[12] - unpacked_1[4];
                res[13] = ov[13] - unpacked_1[5];
                res[14] = ov[14] - unpacked_1[6];
                res[15] = ov[15] - unpacked_1[7];
                int byte_offset_1 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 16;
                int swizzled_offset_2 = byte_offset_1 ^ (byte_offset_1 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_2), "r"(ov_bf16_1[0]), "r"(ov_bf16_1[1]), "r"(ov_bf16_1[2]), "r"(ov_bf16_1[3]) : "memory");
                unsigned int packed[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 0], res[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 2], res[_lp*2+1 + 2]));
                    packed[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 4], res[_lp*2+1 + 4]));
                    packed[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 6], res[_lp*2+1 + 6]));
                    packed[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 8], res[_lp*2+1 + 8]));
                    packed[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 10], res[_lp*2+1 + 10]));
                    packed[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 12], res[_lp*2+1 + 12]));
                    packed[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 14], res[_lp*2+1 + 14]));
                    packed[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_34 = reinterpret_cast<const unsigned*>(packed);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col))), "r"(_raw_stv8_34[0]), "r"(_raw_stv8_34[1]), "r"(_raw_stv8_34[2]), "r"(_raw_stv8_34[3]), "r"(_raw_stv8_34[4]), "r"(_raw_stv8_34[5]), "r"(_raw_stv8_34[6]), "r"(_raw_stv8_34[7]) : "memory");
                }
                float res_0[16];
                unsigned int ov_bf16_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 16], ov[_lp*2+1 + 16]));
                    ov_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2[_pair]) << 16);
                    (&unpacked_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2[_pair]) & 0xffff0000u);
                }
                res_0[0] = ov[16] - unpacked_2[0];
                res_0[1] = ov[17] - unpacked_2[1];
                res_0[2] = ov[18] - unpacked_2[2];
                res_0[3] = ov[19] - unpacked_2[3];
                res_0[4] = ov[20] - unpacked_2[4];
                res_0[5] = ov[21] - unpacked_2[5];
                res_0[6] = ov[22] - unpacked_2[6];
                res_0[7] = ov[23] - unpacked_2[7];
                int byte_offset_2 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 32;
                int swizzled_offset_3 = byte_offset_2 ^ (byte_offset_2 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3), "r"(ov_bf16_2[0]), "r"(ov_bf16_2[1]), "r"(ov_bf16_2[2]), "r"(ov_bf16_2[3]) : "memory");
                unsigned int ov_bf16_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 24], ov[_lp*2+1 + 24]));
                    ov_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_3[_pair]) << 16);
                    (&unpacked_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_3[_pair]) & 0xffff0000u);
                }
                res_0[8] = ov[24] - unpacked_3[0];
                res_0[9] = ov[25] - unpacked_3[1];
                res_0[10] = ov[26] - unpacked_3[2];
                res_0[11] = ov[27] - unpacked_3[3];
                res_0[12] = ov[28] - unpacked_3[4];
                res_0[13] = ov[29] - unpacked_3[5];
                res_0[14] = ov[30] - unpacked_3[6];
                res_0[15] = ov[31] - unpacked_3[7];
                int byte_offset_3 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 48;
                int swizzled_offset_4 = byte_offset_3 ^ (byte_offset_3 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_4), "r"(ov_bf16_3[0]), "r"(ov_bf16_3[1]), "r"(ov_bf16_3[2]), "r"(ov_bf16_3[3]) : "memory");
                unsigned int packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 0], res_0[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 2], res_0[_lp*2+1 + 2]));
                    packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 4], res_0[_lp*2+1 + 4]));
                    packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 6], res_0[_lp*2+1 + 6]));
                    packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 8], res_0[_lp*2+1 + 8]));
                    packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 10], res_0[_lp*2+1 + 10]));
                    packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 12], res_0[_lp*2+1 + 12]));
                    packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 14], res_0[_lp*2+1 + 14]));
                    packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_35 = reinterpret_cast<const unsigned*>(packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 16ll))), "r"(_raw_stv8_35[0]), "r"(_raw_stv8_35[1]), "r"(_raw_stv8_35[2]), "r"(_raw_stv8_35[3]), "r"(_raw_stv8_35[4]), "r"(_raw_stv8_35[5]), "r"(_raw_stv8_35[6]), "r"(_raw_stv8_35[7]) : "memory");
                }
                float res_1[16];
                unsigned int ov_bf16_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 32], ov[_lp*2+1 + 32]));
                    ov_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_4[_pair]) << 16);
                    (&unpacked_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_4[_pair]) & 0xffff0000u);
                }
                res_1[0] = ov[32] - unpacked_4[0];
                res_1[1] = ov[33] - unpacked_4[1];
                res_1[2] = ov[34] - unpacked_4[2];
                res_1[3] = ov[35] - unpacked_4[3];
                res_1[4] = ov[36] - unpacked_4[4];
                res_1[5] = ov[37] - unpacked_4[5];
                res_1[6] = ov[38] - unpacked_4[6];
                res_1[7] = ov[39] - unpacked_4[7];
                int byte_offset_4 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 64;
                int swizzled_offset_5 = byte_offset_4 ^ (byte_offset_4 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_5), "r"(ov_bf16_4[0]), "r"(ov_bf16_4[1]), "r"(ov_bf16_4[2]), "r"(ov_bf16_4[3]) : "memory");
                unsigned int ov_bf16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 40], ov[_lp*2+1 + 40]));
                    ov_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) << 16);
                    (&unpacked_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) & 0xffff0000u);
                }
                res_1[8] = ov[40] - unpacked_5[0];
                res_1[9] = ov[41] - unpacked_5[1];
                res_1[10] = ov[42] - unpacked_5[2];
                res_1[11] = ov[43] - unpacked_5[3];
                res_1[12] = ov[44] - unpacked_5[4];
                res_1[13] = ov[45] - unpacked_5[5];
                res_1[14] = ov[46] - unpacked_5[6];
                res_1[15] = ov[47] - unpacked_5[7];
                int byte_offset_5 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 80;
                int swizzled_offset_6 = byte_offset_5 ^ (byte_offset_5 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_6), "r"(ov_bf16_5[0]), "r"(ov_bf16_5[1]), "r"(ov_bf16_5[2]), "r"(ov_bf16_5[3]) : "memory");
                unsigned int packed_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 0], res_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 2], res_1[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 4], res_1[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 6], res_1[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 8], res_1[_lp*2+1 + 8]));
                    packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 10], res_1[_lp*2+1 + 10]));
                    packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 12], res_1[_lp*2+1 + 12]));
                    packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 14], res_1[_lp*2+1 + 14]));
                    packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_36 = reinterpret_cast<const unsigned*>(packed_1);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 32ll))), "r"(_raw_stv8_36[0]), "r"(_raw_stv8_36[1]), "r"(_raw_stv8_36[2]), "r"(_raw_stv8_36[3]), "r"(_raw_stv8_36[4]), "r"(_raw_stv8_36[5]), "r"(_raw_stv8_36[6]), "r"(_raw_stv8_36[7]) : "memory");
                }
                float res_2[16];
                unsigned int ov_bf16_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 48], ov[_lp*2+1 + 48]));
                    ov_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_6[_pair]) << 16);
                    (&unpacked_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_6[_pair]) & 0xffff0000u);
                }
                res_2[0] = ov[48] - unpacked_6[0];
                res_2[1] = ov[49] - unpacked_6[1];
                res_2[2] = ov[50] - unpacked_6[2];
                res_2[3] = ov[51] - unpacked_6[3];
                res_2[4] = ov[52] - unpacked_6[4];
                res_2[5] = ov[53] - unpacked_6[5];
                res_2[6] = ov[54] - unpacked_6[6];
                res_2[7] = ov[55] - unpacked_6[7];
                int byte_offset_6 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 96;
                int swizzled_offset_7 = byte_offset_6 ^ (byte_offset_6 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_7), "r"(ov_bf16_6[0]), "r"(ov_bf16_6[1]), "r"(ov_bf16_6[2]), "r"(ov_bf16_6[3]) : "memory");
                unsigned int ov_bf16_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 56], ov[_lp*2+1 + 56]));
                    ov_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_7[_pair]) << 16);
                    (&unpacked_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_7[_pair]) & 0xffff0000u);
                }
                res_2[8] = ov[56] - unpacked_7[0];
                res_2[9] = ov[57] - unpacked_7[1];
                res_2[10] = ov[58] - unpacked_7[2];
                res_2[11] = ov[59] - unpacked_7[3];
                res_2[12] = ov[60] - unpacked_7[4];
                res_2[13] = ov[61] - unpacked_7[5];
                res_2[14] = ov[62] - unpacked_7[6];
                res_2[15] = ov[63] - unpacked_7[7];
                int byte_offset_7 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 112;
                int swizzled_offset_8 = byte_offset_7 ^ (byte_offset_7 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_8), "r"(ov_bf16_7[0]), "r"(ov_bf16_7[1]), "r"(ov_bf16_7[2]), "r"(ov_bf16_7[3]) : "memory");
                unsigned int packed_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 0], res_2[_lp*2+1 + 0]));
                    packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 2], res_2[_lp*2+1 + 2]));
                    packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 4], res_2[_lp*2+1 + 4]));
                    packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 6], res_2[_lp*2+1 + 6]));
                    packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 8], res_2[_lp*2+1 + 8]));
                    packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 10], res_2[_lp*2+1 + 10]));
                    packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 12], res_2[_lp*2+1 + 12]));
                    packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 14], res_2[_lp*2+1 + 14]));
                    packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(packed_2);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 48ll))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk = warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk * 8192));
                    }
                }
                float ov_0[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_0[0]), "=f"(ov_0[1]), "=f"(ov_0[2]), "=f"(ov_0[3]), "=f"(ov_0[4]), "=f"(ov_0[5]), "=f"(ov_0[6]), "=f"(ov_0[7]), "=f"(ov_0[8]), "=f"(ov_0[9]), "=f"(ov_0[10]), "=f"(ov_0[11]), "=f"(ov_0[12]), "=f"(ov_0[13]), "=f"(ov_0[14]), "=f"(ov_0[15]), "=f"(ov_0[16]), "=f"(ov_0[17]), "=f"(ov_0[18]), "=f"(ov_0[19]), "=f"(ov_0[20]), "=f"(ov_0[21]), "=f"(ov_0[22]), "=f"(ov_0[23]), "=f"(ov_0[24]), "=f"(ov_0[25]), "=f"(ov_0[26]), "=f"(ov_0[27]), "=f"(ov_0[28]), "=f"(ov_0[29]), "=f"(ov_0[30]), "=f"(ov_0[31]), "=f"(ov_0[32]), "=f"(ov_0[33]), "=f"(ov_0[34]), "=f"(ov_0[35]), "=f"(ov_0[36]), "=f"(ov_0[37]), "=f"(ov_0[38]), "=f"(ov_0[39]), "=f"(ov_0[40]), "=f"(ov_0[41]), "=f"(ov_0[42]), "=f"(ov_0[43]), "=f"(ov_0[44]), "=f"(ov_0[45]), "=f"(ov_0[46]), "=f"(ov_0[47]), "=f"(ov_0[48]), "=f"(ov_0[49]), "=f"(ov_0[50]), "=f"(ov_0[51]), "=f"(ov_0[52]), "=f"(ov_0[53]), "=f"(ov_0[54]), "=f"(ov_0[55]), "=f"(ov_0[56]), "=f"(ov_0[57]), "=f"(ov_0[58]), "=f"(ov_0[59]), "=f"(ov_0[60]), "=f"(ov_0[61]), "=f"(ov_0[62]), "=f"(ov_0[63])
                        : "r"(tmrow + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_0[0] = 0.0f;
                    ov_0[1] = 0.0f;
                    ov_0[2] = 0.0f;
                    ov_0[3] = 0.0f;
                    ov_0[4] = 0.0f;
                    ov_0[5] = 0.0f;
                    ov_0[6] = 0.0f;
                    ov_0[7] = 0.0f;
                    ov_0[8] = 0.0f;
                    ov_0[9] = 0.0f;
                    ov_0[10] = 0.0f;
                    ov_0[11] = 0.0f;
                    ov_0[12] = 0.0f;
                    ov_0[13] = 0.0f;
                    ov_0[14] = 0.0f;
                    ov_0[15] = 0.0f;
                    ov_0[16] = 0.0f;
                    ov_0[17] = 0.0f;
                    ov_0[18] = 0.0f;
                    ov_0[19] = 0.0f;
                    ov_0[20] = 0.0f;
                    ov_0[21] = 0.0f;
                    ov_0[22] = 0.0f;
                    ov_0[23] = 0.0f;
                    ov_0[24] = 0.0f;
                    ov_0[25] = 0.0f;
                    ov_0[26] = 0.0f;
                    ov_0[27] = 0.0f;
                    ov_0[28] = 0.0f;
                    ov_0[29] = 0.0f;
                    ov_0[30] = 0.0f;
                    ov_0[31] = 0.0f;
                    ov_0[32] = 0.0f;
                    ov_0[33] = 0.0f;
                    ov_0[34] = 0.0f;
                    ov_0[35] = 0.0f;
                    ov_0[36] = 0.0f;
                    ov_0[37] = 0.0f;
                    ov_0[38] = 0.0f;
                    ov_0[39] = 0.0f;
                    ov_0[40] = 0.0f;
                    ov_0[41] = 0.0f;
                    ov_0[42] = 0.0f;
                    ov_0[43] = 0.0f;
                    ov_0[44] = 0.0f;
                    ov_0[45] = 0.0f;
                    ov_0[46] = 0.0f;
                    ov_0[47] = 0.0f;
                    ov_0[48] = 0.0f;
                    ov_0[49] = 0.0f;
                    ov_0[50] = 0.0f;
                    ov_0[51] = 0.0f;
                    ov_0[52] = 0.0f;
                    ov_0[53] = 0.0f;
                    ov_0[54] = 0.0f;
                    ov_0[55] = 0.0f;
                    ov_0[56] = 0.0f;
                    ov_0[57] = 0.0f;
                    ov_0[58] = 0.0f;
                    ov_0[59] = 0.0f;
                    ov_0[60] = 0.0f;
                    ov_0[61] = 0.0f;
                    ov_0[62] = 0.0f;
                    ov_0[63] = 0.0f;
                }
                const float2 _scale2_38 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_0)[_ls], _scale2_38);
                int out_col_0 = row / 64 * 128 + 64;
                float res_3[16];
                unsigned int ov_bf16_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 0], ov_0[_lp*2+1 + 0]));
                    ov_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_8[_pair]) << 16);
                    (&unpacked_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_8[_pair]) & 0xffff0000u);
                }
                res_3[0] = ov_0[0] - unpacked_8[0];
                res_3[1] = ov_0[1] - unpacked_8[1];
                res_3[2] = ov_0[2] - unpacked_8[2];
                res_3[3] = ov_0[3] - unpacked_8[3];
                res_3[4] = ov_0[4] - unpacked_8[4];
                res_3[5] = ov_0[5] - unpacked_8[5];
                res_3[6] = ov_0[6] - unpacked_8[6];
                res_3[7] = ov_0[7] - unpacked_8[7];
                int byte_offset_8 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128;
                int swizzled_offset_9 = byte_offset_8 ^ (byte_offset_8 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_9), "r"(ov_bf16_8[0]), "r"(ov_bf16_8[1]), "r"(ov_bf16_8[2]), "r"(ov_bf16_8[3]) : "memory");
                unsigned int ov_bf16_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 8], ov_0[_lp*2+1 + 8]));
                    ov_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_9[_pair]) << 16);
                    (&unpacked_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_9[_pair]) & 0xffff0000u);
                }
                res_3[8] = ov_0[8] - unpacked_9[0];
                res_3[9] = ov_0[9] - unpacked_9[1];
                res_3[10] = ov_0[10] - unpacked_9[2];
                res_3[11] = ov_0[11] - unpacked_9[3];
                res_3[12] = ov_0[12] - unpacked_9[4];
                res_3[13] = ov_0[13] - unpacked_9[5];
                res_3[14] = ov_0[14] - unpacked_9[6];
                res_3[15] = ov_0[15] - unpacked_9[7];
                int byte_offset_9 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_10 = byte_offset_9 ^ (byte_offset_9 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_10), "r"(ov_bf16_9[0]), "r"(ov_bf16_9[1]), "r"(ov_bf16_9[2]), "r"(ov_bf16_9[3]) : "memory");
                unsigned int packed_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
                    packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 2], res_3[_lp*2+1 + 2]));
                    packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 4], res_3[_lp*2+1 + 4]));
                    packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 6], res_3[_lp*2+1 + 6]));
                    packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 8], res_3[_lp*2+1 + 8]));
                    packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 10], res_3[_lp*2+1 + 10]));
                    packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 12], res_3[_lp*2+1 + 12]));
                    packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 14], res_3[_lp*2+1 + 14]));
                    packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_39 = reinterpret_cast<const unsigned*>(packed_3);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0))), "r"(_raw_stv8_39[0]), "r"(_raw_stv8_39[1]), "r"(_raw_stv8_39[2]), "r"(_raw_stv8_39[3]), "r"(_raw_stv8_39[4]), "r"(_raw_stv8_39[5]), "r"(_raw_stv8_39[6]), "r"(_raw_stv8_39[7]) : "memory");
                }
                float res_4[16];
                unsigned int ov_bf16_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 16], ov_0[_lp*2+1 + 16]));
                    ov_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_10[_pair]) << 16);
                    (&unpacked_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_10[_pair]) & 0xffff0000u);
                }
                res_4[0] = ov_0[16] - unpacked_10[0];
                res_4[1] = ov_0[17] - unpacked_10[1];
                res_4[2] = ov_0[18] - unpacked_10[2];
                res_4[3] = ov_0[19] - unpacked_10[3];
                res_4[4] = ov_0[20] - unpacked_10[4];
                res_4[5] = ov_0[21] - unpacked_10[5];
                res_4[6] = ov_0[22] - unpacked_10[6];
                res_4[7] = ov_0[23] - unpacked_10[7];
                int byte_offset_10 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_11 = byte_offset_10 ^ (byte_offset_10 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_11), "r"(ov_bf16_10[0]), "r"(ov_bf16_10[1]), "r"(ov_bf16_10[2]), "r"(ov_bf16_10[3]) : "memory");
                unsigned int ov_bf16_11[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 24], ov_0[_lp*2+1 + 24]));
                    ov_bf16_11[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_11[_pair]) << 16);
                    (&unpacked_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_11[_pair]) & 0xffff0000u);
                }
                res_4[8] = ov_0[24] - unpacked_11[0];
                res_4[9] = ov_0[25] - unpacked_11[1];
                res_4[10] = ov_0[26] - unpacked_11[2];
                res_4[11] = ov_0[27] - unpacked_11[3];
                res_4[12] = ov_0[28] - unpacked_11[4];
                res_4[13] = ov_0[29] - unpacked_11[5];
                res_4[14] = ov_0[30] - unpacked_11[6];
                res_4[15] = ov_0[31] - unpacked_11[7];
                int byte_offset_11 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_12 = byte_offset_11 ^ (byte_offset_11 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_12), "r"(ov_bf16_11[0]), "r"(ov_bf16_11[1]), "r"(ov_bf16_11[2]), "r"(ov_bf16_11[3]) : "memory");
                unsigned int packed_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                    packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 2], res_4[_lp*2+1 + 2]));
                    packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 4], res_4[_lp*2+1 + 4]));
                    packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 6], res_4[_lp*2+1 + 6]));
                    packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 8], res_4[_lp*2+1 + 8]));
                    packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 10], res_4[_lp*2+1 + 10]));
                    packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 12], res_4[_lp*2+1 + 12]));
                    packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 14], res_4[_lp*2+1 + 14]));
                    packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_40 = reinterpret_cast<const unsigned*>(packed_4);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 16ll))), "r"(_raw_stv8_40[0]), "r"(_raw_stv8_40[1]), "r"(_raw_stv8_40[2]), "r"(_raw_stv8_40[3]), "r"(_raw_stv8_40[4]), "r"(_raw_stv8_40[5]), "r"(_raw_stv8_40[6]), "r"(_raw_stv8_40[7]) : "memory");
                }
                float res_5[16];
                unsigned int ov_bf16_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 32], ov_0[_lp*2+1 + 32]));
                    ov_bf16_12[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_12[_pair]) << 16);
                    (&unpacked_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_12[_pair]) & 0xffff0000u);
                }
                res_5[0] = ov_0[32] - unpacked_12[0];
                res_5[1] = ov_0[33] - unpacked_12[1];
                res_5[2] = ov_0[34] - unpacked_12[2];
                res_5[3] = ov_0[35] - unpacked_12[3];
                res_5[4] = ov_0[36] - unpacked_12[4];
                res_5[5] = ov_0[37] - unpacked_12[5];
                res_5[6] = ov_0[38] - unpacked_12[6];
                res_5[7] = ov_0[39] - unpacked_12[7];
                int byte_offset_12 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_13 = byte_offset_12 ^ (byte_offset_12 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_13), "r"(ov_bf16_12[0]), "r"(ov_bf16_12[1]), "r"(ov_bf16_12[2]), "r"(ov_bf16_12[3]) : "memory");
                unsigned int ov_bf16_13[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 40], ov_0[_lp*2+1 + 40]));
                    ov_bf16_13[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_13[_pair]) << 16);
                    (&unpacked_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_13[_pair]) & 0xffff0000u);
                }
                res_5[8] = ov_0[40] - unpacked_13[0];
                res_5[9] = ov_0[41] - unpacked_13[1];
                res_5[10] = ov_0[42] - unpacked_13[2];
                res_5[11] = ov_0[43] - unpacked_13[3];
                res_5[12] = ov_0[44] - unpacked_13[4];
                res_5[13] = ov_0[45] - unpacked_13[5];
                res_5[14] = ov_0[46] - unpacked_13[6];
                res_5[15] = ov_0[47] - unpacked_13[7];
                int byte_offset_13 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_14 = byte_offset_13 ^ (byte_offset_13 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_14), "r"(ov_bf16_13[0]), "r"(ov_bf16_13[1]), "r"(ov_bf16_13[2]), "r"(ov_bf16_13[3]) : "memory");
                unsigned int packed_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 0], res_5[_lp*2+1 + 0]));
                    packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 2], res_5[_lp*2+1 + 2]));
                    packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 4], res_5[_lp*2+1 + 4]));
                    packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 6], res_5[_lp*2+1 + 6]));
                    packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 8], res_5[_lp*2+1 + 8]));
                    packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 10], res_5[_lp*2+1 + 10]));
                    packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 12], res_5[_lp*2+1 + 12]));
                    packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 14], res_5[_lp*2+1 + 14]));
                    packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_41 = reinterpret_cast<const unsigned*>(packed_5);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 32ll))), "r"(_raw_stv8_41[0]), "r"(_raw_stv8_41[1]), "r"(_raw_stv8_41[2]), "r"(_raw_stv8_41[3]), "r"(_raw_stv8_41[4]), "r"(_raw_stv8_41[5]), "r"(_raw_stv8_41[6]), "r"(_raw_stv8_41[7]) : "memory");
                }
                float res_6[16];
                unsigned int ov_bf16_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 48], ov_0[_lp*2+1 + 48]));
                    ov_bf16_14[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_14[_pair]) << 16);
                    (&unpacked_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_14[_pair]) & 0xffff0000u);
                }
                res_6[0] = ov_0[48] - unpacked_14[0];
                res_6[1] = ov_0[49] - unpacked_14[1];
                res_6[2] = ov_0[50] - unpacked_14[2];
                res_6[3] = ov_0[51] - unpacked_14[3];
                res_6[4] = ov_0[52] - unpacked_14[4];
                res_6[5] = ov_0[53] - unpacked_14[5];
                res_6[6] = ov_0[54] - unpacked_14[6];
                res_6[7] = ov_0[55] - unpacked_14[7];
                int byte_offset_14 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_15 = byte_offset_14 ^ (byte_offset_14 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_15), "r"(ov_bf16_14[0]), "r"(ov_bf16_14[1]), "r"(ov_bf16_14[2]), "r"(ov_bf16_14[3]) : "memory");
                unsigned int ov_bf16_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 56], ov_0[_lp*2+1 + 56]));
                    ov_bf16_15[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_15[_pair]) << 16);
                    (&unpacked_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_15[_pair]) & 0xffff0000u);
                }
                res_6[8] = ov_0[56] - unpacked_15[0];
                res_6[9] = ov_0[57] - unpacked_15[1];
                res_6[10] = ov_0[58] - unpacked_15[2];
                res_6[11] = ov_0[59] - unpacked_15[3];
                res_6[12] = ov_0[60] - unpacked_15[4];
                res_6[13] = ov_0[61] - unpacked_15[5];
                res_6[14] = ov_0[62] - unpacked_15[6];
                res_6[15] = ov_0[63] - unpacked_15[7];
                int byte_offset_15 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_16 = byte_offset_15 ^ (byte_offset_15 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_16), "r"(ov_bf16_15[0]), "r"(ov_bf16_15[1]), "r"(ov_bf16_15[2]), "r"(ov_bf16_15[3]) : "memory");
                unsigned int packed_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 0], res_6[_lp*2+1 + 0]));
                    packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 2], res_6[_lp*2+1 + 2]));
                    packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 4], res_6[_lp*2+1 + 4]));
                    packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 6], res_6[_lp*2+1 + 6]));
                    packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 8], res_6[_lp*2+1 + 8]));
                    packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 10], res_6[_lp*2+1 + 10]));
                    packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 12], res_6[_lp*2+1 + 12]));
                    packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 14], res_6[_lp*2+1 + 14]));
                    packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_42 = reinterpret_cast<const unsigned*>(packed_6);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 48ll))), "r"(_raw_stv8_42[0]), "r"(_raw_stv8_42[1]), "r"(_raw_stv8_42[2]), "r"(_raw_stv8_42[3]), "r"(_raw_stv8_42[4]), "r"(_raw_stv8_42[5]), "r"(_raw_stv8_42[6]), "r"(_raw_stv8_42[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_1 = warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_1, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_1 * 8192));
                    }
                }
                float ov_1[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_1[0]), "=f"(ov_1[1]), "=f"(ov_1[2]), "=f"(ov_1[3]), "=f"(ov_1[4]), "=f"(ov_1[5]), "=f"(ov_1[6]), "=f"(ov_1[7]), "=f"(ov_1[8]), "=f"(ov_1[9]), "=f"(ov_1[10]), "=f"(ov_1[11]), "=f"(ov_1[12]), "=f"(ov_1[13]), "=f"(ov_1[14]), "=f"(ov_1[15]), "=f"(ov_1[16]), "=f"(ov_1[17]), "=f"(ov_1[18]), "=f"(ov_1[19]), "=f"(ov_1[20]), "=f"(ov_1[21]), "=f"(ov_1[22]), "=f"(ov_1[23]), "=f"(ov_1[24]), "=f"(ov_1[25]), "=f"(ov_1[26]), "=f"(ov_1[27]), "=f"(ov_1[28]), "=f"(ov_1[29]), "=f"(ov_1[30]), "=f"(ov_1[31]), "=f"(ov_1[32]), "=f"(ov_1[33]), "=f"(ov_1[34]), "=f"(ov_1[35]), "=f"(ov_1[36]), "=f"(ov_1[37]), "=f"(ov_1[38]), "=f"(ov_1[39]), "=f"(ov_1[40]), "=f"(ov_1[41]), "=f"(ov_1[42]), "=f"(ov_1[43]), "=f"(ov_1[44]), "=f"(ov_1[45]), "=f"(ov_1[46]), "=f"(ov_1[47]), "=f"(ov_1[48]), "=f"(ov_1[49]), "=f"(ov_1[50]), "=f"(ov_1[51]), "=f"(ov_1[52]), "=f"(ov_1[53]), "=f"(ov_1[54]), "=f"(ov_1[55]), "=f"(ov_1[56]), "=f"(ov_1[57]), "=f"(ov_1[58]), "=f"(ov_1[59]), "=f"(ov_1[60]), "=f"(ov_1[61]), "=f"(ov_1[62]), "=f"(ov_1[63])
                        : "r"(tmrow + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_1[0] = 0.0f;
                    ov_1[1] = 0.0f;
                    ov_1[2] = 0.0f;
                    ov_1[3] = 0.0f;
                    ov_1[4] = 0.0f;
                    ov_1[5] = 0.0f;
                    ov_1[6] = 0.0f;
                    ov_1[7] = 0.0f;
                    ov_1[8] = 0.0f;
                    ov_1[9] = 0.0f;
                    ov_1[10] = 0.0f;
                    ov_1[11] = 0.0f;
                    ov_1[12] = 0.0f;
                    ov_1[13] = 0.0f;
                    ov_1[14] = 0.0f;
                    ov_1[15] = 0.0f;
                    ov_1[16] = 0.0f;
                    ov_1[17] = 0.0f;
                    ov_1[18] = 0.0f;
                    ov_1[19] = 0.0f;
                    ov_1[20] = 0.0f;
                    ov_1[21] = 0.0f;
                    ov_1[22] = 0.0f;
                    ov_1[23] = 0.0f;
                    ov_1[24] = 0.0f;
                    ov_1[25] = 0.0f;
                    ov_1[26] = 0.0f;
                    ov_1[27] = 0.0f;
                    ov_1[28] = 0.0f;
                    ov_1[29] = 0.0f;
                    ov_1[30] = 0.0f;
                    ov_1[31] = 0.0f;
                    ov_1[32] = 0.0f;
                    ov_1[33] = 0.0f;
                    ov_1[34] = 0.0f;
                    ov_1[35] = 0.0f;
                    ov_1[36] = 0.0f;
                    ov_1[37] = 0.0f;
                    ov_1[38] = 0.0f;
                    ov_1[39] = 0.0f;
                    ov_1[40] = 0.0f;
                    ov_1[41] = 0.0f;
                    ov_1[42] = 0.0f;
                    ov_1[43] = 0.0f;
                    ov_1[44] = 0.0f;
                    ov_1[45] = 0.0f;
                    ov_1[46] = 0.0f;
                    ov_1[47] = 0.0f;
                    ov_1[48] = 0.0f;
                    ov_1[49] = 0.0f;
                    ov_1[50] = 0.0f;
                    ov_1[51] = 0.0f;
                    ov_1[52] = 0.0f;
                    ov_1[53] = 0.0f;
                    ov_1[54] = 0.0f;
                    ov_1[55] = 0.0f;
                    ov_1[56] = 0.0f;
                    ov_1[57] = 0.0f;
                    ov_1[58] = 0.0f;
                    ov_1[59] = 0.0f;
                    ov_1[60] = 0.0f;
                    ov_1[61] = 0.0f;
                    ov_1[62] = 0.0f;
                    ov_1[63] = 0.0f;
                }
                const float2 _scale2_43 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_43);
                int out_col_1 = 256 + row / 64 * 128;
                float res_7[16];
                unsigned int ov_bf16_16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 0], ov_1[_lp*2+1 + 0]));
                    ov_bf16_16[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_16[_pair]) << 16);
                    (&unpacked_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_16[_pair]) & 0xffff0000u);
                }
                res_7[0] = ov_1[0] - unpacked_16[0];
                res_7[1] = ov_1[1] - unpacked_16[1];
                res_7[2] = ov_1[2] - unpacked_16[2];
                res_7[3] = ov_1[3] - unpacked_16[3];
                res_7[4] = ov_1[4] - unpacked_16[4];
                res_7[5] = ov_1[5] - unpacked_16[5];
                res_7[6] = ov_1[6] - unpacked_16[6];
                res_7[7] = ov_1[7] - unpacked_16[7];
                int byte_offset_16 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128;
                int swizzled_offset_17 = byte_offset_16 ^ (byte_offset_16 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_17), "r"(ov_bf16_16[0]), "r"(ov_bf16_16[1]), "r"(ov_bf16_16[2]), "r"(ov_bf16_16[3]) : "memory");
                unsigned int ov_bf16_17[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 8], ov_1[_lp*2+1 + 8]));
                    ov_bf16_17[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_17[_pair]) << 16);
                    (&unpacked_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_17[_pair]) & 0xffff0000u);
                }
                res_7[8] = ov_1[8] - unpacked_17[0];
                res_7[9] = ov_1[9] - unpacked_17[1];
                res_7[10] = ov_1[10] - unpacked_17[2];
                res_7[11] = ov_1[11] - unpacked_17[3];
                res_7[12] = ov_1[12] - unpacked_17[4];
                res_7[13] = ov_1[13] - unpacked_17[5];
                res_7[14] = ov_1[14] - unpacked_17[6];
                res_7[15] = ov_1[15] - unpacked_17[7];
                int byte_offset_17 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_18 = byte_offset_17 ^ (byte_offset_17 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_18), "r"(ov_bf16_17[0]), "r"(ov_bf16_17[1]), "r"(ov_bf16_17[2]), "r"(ov_bf16_17[3]) : "memory");
                unsigned int packed_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 0], res_7[_lp*2+1 + 0]));
                    packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 2], res_7[_lp*2+1 + 2]));
                    packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 4], res_7[_lp*2+1 + 4]));
                    packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 6], res_7[_lp*2+1 + 6]));
                    packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 8], res_7[_lp*2+1 + 8]));
                    packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 10], res_7[_lp*2+1 + 10]));
                    packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 12], res_7[_lp*2+1 + 12]));
                    packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 14], res_7[_lp*2+1 + 14]));
                    packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_44 = reinterpret_cast<const unsigned*>(packed_7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1))), "r"(_raw_stv8_44[0]), "r"(_raw_stv8_44[1]), "r"(_raw_stv8_44[2]), "r"(_raw_stv8_44[3]), "r"(_raw_stv8_44[4]), "r"(_raw_stv8_44[5]), "r"(_raw_stv8_44[6]), "r"(_raw_stv8_44[7]) : "memory");
                }
                float res_8[16];
                unsigned int ov_bf16_18[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 16], ov_1[_lp*2+1 + 16]));
                    ov_bf16_18[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_18[_pair]) << 16);
                    (&unpacked_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_18[_pair]) & 0xffff0000u);
                }
                res_8[0] = ov_1[16] - unpacked_18[0];
                res_8[1] = ov_1[17] - unpacked_18[1];
                res_8[2] = ov_1[18] - unpacked_18[2];
                res_8[3] = ov_1[19] - unpacked_18[3];
                res_8[4] = ov_1[20] - unpacked_18[4];
                res_8[5] = ov_1[21] - unpacked_18[5];
                res_8[6] = ov_1[22] - unpacked_18[6];
                res_8[7] = ov_1[23] - unpacked_18[7];
                int byte_offset_18 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_19 = byte_offset_18 ^ (byte_offset_18 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_19), "r"(ov_bf16_18[0]), "r"(ov_bf16_18[1]), "r"(ov_bf16_18[2]), "r"(ov_bf16_18[3]) : "memory");
                unsigned int ov_bf16_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 24], ov_1[_lp*2+1 + 24]));
                    ov_bf16_19[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_19[_pair]) << 16);
                    (&unpacked_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_19[_pair]) & 0xffff0000u);
                }
                res_8[8] = ov_1[24] - unpacked_19[0];
                res_8[9] = ov_1[25] - unpacked_19[1];
                res_8[10] = ov_1[26] - unpacked_19[2];
                res_8[11] = ov_1[27] - unpacked_19[3];
                res_8[12] = ov_1[28] - unpacked_19[4];
                res_8[13] = ov_1[29] - unpacked_19[5];
                res_8[14] = ov_1[30] - unpacked_19[6];
                res_8[15] = ov_1[31] - unpacked_19[7];
                int byte_offset_19 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_20 = byte_offset_19 ^ (byte_offset_19 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_20), "r"(ov_bf16_19[0]), "r"(ov_bf16_19[1]), "r"(ov_bf16_19[2]), "r"(ov_bf16_19[3]) : "memory");
                unsigned int packed_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 0], res_8[_lp*2+1 + 0]));
                    packed_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 2], res_8[_lp*2+1 + 2]));
                    packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 4], res_8[_lp*2+1 + 4]));
                    packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 6], res_8[_lp*2+1 + 6]));
                    packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 8], res_8[_lp*2+1 + 8]));
                    packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 10], res_8[_lp*2+1 + 10]));
                    packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 12], res_8[_lp*2+1 + 12]));
                    packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 14], res_8[_lp*2+1 + 14]));
                    packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_45 = reinterpret_cast<const unsigned*>(packed_8);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 16ll))), "r"(_raw_stv8_45[0]), "r"(_raw_stv8_45[1]), "r"(_raw_stv8_45[2]), "r"(_raw_stv8_45[3]), "r"(_raw_stv8_45[4]), "r"(_raw_stv8_45[5]), "r"(_raw_stv8_45[6]), "r"(_raw_stv8_45[7]) : "memory");
                }
                float res_9[16];
                unsigned int ov_bf16_20[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 32], ov_1[_lp*2+1 + 32]));
                    ov_bf16_20[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_20[_pair]) << 16);
                    (&unpacked_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_20[_pair]) & 0xffff0000u);
                }
                res_9[0] = ov_1[32] - unpacked_20[0];
                res_9[1] = ov_1[33] - unpacked_20[1];
                res_9[2] = ov_1[34] - unpacked_20[2];
                res_9[3] = ov_1[35] - unpacked_20[3];
                res_9[4] = ov_1[36] - unpacked_20[4];
                res_9[5] = ov_1[37] - unpacked_20[5];
                res_9[6] = ov_1[38] - unpacked_20[6];
                res_9[7] = ov_1[39] - unpacked_20[7];
                int byte_offset_20 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_21 = byte_offset_20 ^ (byte_offset_20 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_21), "r"(ov_bf16_20[0]), "r"(ov_bf16_20[1]), "r"(ov_bf16_20[2]), "r"(ov_bf16_20[3]) : "memory");
                unsigned int ov_bf16_21[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 40], ov_1[_lp*2+1 + 40]));
                    ov_bf16_21[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_21[_pair]) << 16);
                    (&unpacked_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_21[_pair]) & 0xffff0000u);
                }
                res_9[8] = ov_1[40] - unpacked_21[0];
                res_9[9] = ov_1[41] - unpacked_21[1];
                res_9[10] = ov_1[42] - unpacked_21[2];
                res_9[11] = ov_1[43] - unpacked_21[3];
                res_9[12] = ov_1[44] - unpacked_21[4];
                res_9[13] = ov_1[45] - unpacked_21[5];
                res_9[14] = ov_1[46] - unpacked_21[6];
                res_9[15] = ov_1[47] - unpacked_21[7];
                int byte_offset_21 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_22 = byte_offset_21 ^ (byte_offset_21 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_22), "r"(ov_bf16_21[0]), "r"(ov_bf16_21[1]), "r"(ov_bf16_21[2]), "r"(ov_bf16_21[3]) : "memory");
                unsigned int packed_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 0], res_9[_lp*2+1 + 0]));
                    packed_9[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 2], res_9[_lp*2+1 + 2]));
                    packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 4], res_9[_lp*2+1 + 4]));
                    packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 6], res_9[_lp*2+1 + 6]));
                    packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 8], res_9[_lp*2+1 + 8]));
                    packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 10], res_9[_lp*2+1 + 10]));
                    packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 12], res_9[_lp*2+1 + 12]));
                    packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 14], res_9[_lp*2+1 + 14]));
                    packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_46 = reinterpret_cast<const unsigned*>(packed_9);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 32ll))), "r"(_raw_stv8_46[0]), "r"(_raw_stv8_46[1]), "r"(_raw_stv8_46[2]), "r"(_raw_stv8_46[3]), "r"(_raw_stv8_46[4]), "r"(_raw_stv8_46[5]), "r"(_raw_stv8_46[6]), "r"(_raw_stv8_46[7]) : "memory");
                }
                float res_10[16];
                unsigned int ov_bf16_22[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 48], ov_1[_lp*2+1 + 48]));
                    ov_bf16_22[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_22[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_22[_pair]) << 16);
                    (&unpacked_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_22[_pair]) & 0xffff0000u);
                }
                res_10[0] = ov_1[48] - unpacked_22[0];
                res_10[1] = ov_1[49] - unpacked_22[1];
                res_10[2] = ov_1[50] - unpacked_22[2];
                res_10[3] = ov_1[51] - unpacked_22[3];
                res_10[4] = ov_1[52] - unpacked_22[4];
                res_10[5] = ov_1[53] - unpacked_22[5];
                res_10[6] = ov_1[54] - unpacked_22[6];
                res_10[7] = ov_1[55] - unpacked_22[7];
                int byte_offset_22 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_23 = byte_offset_22 ^ (byte_offset_22 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_23), "r"(ov_bf16_22[0]), "r"(ov_bf16_22[1]), "r"(ov_bf16_22[2]), "r"(ov_bf16_22[3]) : "memory");
                unsigned int ov_bf16_23[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 56], ov_1[_lp*2+1 + 56]));
                    ov_bf16_23[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_23[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_23[_pair]) << 16);
                    (&unpacked_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_23[_pair]) & 0xffff0000u);
                }
                res_10[8] = ov_1[56] - unpacked_23[0];
                res_10[9] = ov_1[57] - unpacked_23[1];
                res_10[10] = ov_1[58] - unpacked_23[2];
                res_10[11] = ov_1[59] - unpacked_23[3];
                res_10[12] = ov_1[60] - unpacked_23[4];
                res_10[13] = ov_1[61] - unpacked_23[5];
                res_10[14] = ov_1[62] - unpacked_23[6];
                res_10[15] = ov_1[63] - unpacked_23[7];
                int byte_offset_23 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_24 = byte_offset_23 ^ (byte_offset_23 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_24), "r"(ov_bf16_23[0]), "r"(ov_bf16_23[1]), "r"(ov_bf16_23[2]), "r"(ov_bf16_23[3]) : "memory");
                unsigned int packed_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 0], res_10[_lp*2+1 + 0]));
                    packed_10[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 2], res_10[_lp*2+1 + 2]));
                    packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 4], res_10[_lp*2+1 + 4]));
                    packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 6], res_10[_lp*2+1 + 6]));
                    packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 8], res_10[_lp*2+1 + 8]));
                    packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 10], res_10[_lp*2+1 + 10]));
                    packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 12], res_10[_lp*2+1 + 12]));
                    packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 14], res_10[_lp*2+1 + 14]));
                    packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_47 = reinterpret_cast<const unsigned*>(packed_10);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 48ll))), "r"(_raw_stv8_47[0]), "r"(_raw_stv8_47[1]), "r"(_raw_stv8_47[2]), "r"(_raw_stv8_47[3]), "r"(_raw_stv8_47[4]), "r"(_raw_stv8_47[5]), "r"(_raw_stv8_47[6]), "r"(_raw_stv8_47[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_2 = 4 + warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk_2, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_2 * 8192));
                    }
                }
                float ov_2[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_2[0]), "=f"(ov_2[1]), "=f"(ov_2[2]), "=f"(ov_2[3]), "=f"(ov_2[4]), "=f"(ov_2[5]), "=f"(ov_2[6]), "=f"(ov_2[7]), "=f"(ov_2[8]), "=f"(ov_2[9]), "=f"(ov_2[10]), "=f"(ov_2[11]), "=f"(ov_2[12]), "=f"(ov_2[13]), "=f"(ov_2[14]), "=f"(ov_2[15]), "=f"(ov_2[16]), "=f"(ov_2[17]), "=f"(ov_2[18]), "=f"(ov_2[19]), "=f"(ov_2[20]), "=f"(ov_2[21]), "=f"(ov_2[22]), "=f"(ov_2[23]), "=f"(ov_2[24]), "=f"(ov_2[25]), "=f"(ov_2[26]), "=f"(ov_2[27]), "=f"(ov_2[28]), "=f"(ov_2[29]), "=f"(ov_2[30]), "=f"(ov_2[31]), "=f"(ov_2[32]), "=f"(ov_2[33]), "=f"(ov_2[34]), "=f"(ov_2[35]), "=f"(ov_2[36]), "=f"(ov_2[37]), "=f"(ov_2[38]), "=f"(ov_2[39]), "=f"(ov_2[40]), "=f"(ov_2[41]), "=f"(ov_2[42]), "=f"(ov_2[43]), "=f"(ov_2[44]), "=f"(ov_2[45]), "=f"(ov_2[46]), "=f"(ov_2[47]), "=f"(ov_2[48]), "=f"(ov_2[49]), "=f"(ov_2[50]), "=f"(ov_2[51]), "=f"(ov_2[52]), "=f"(ov_2[53]), "=f"(ov_2[54]), "=f"(ov_2[55]), "=f"(ov_2[56]), "=f"(ov_2[57]), "=f"(ov_2[58]), "=f"(ov_2[59]), "=f"(ov_2[60]), "=f"(ov_2[61]), "=f"(ov_2[62]), "=f"(ov_2[63])
                        : "r"(tmrow + 128 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_2[0] = 0.0f;
                    ov_2[1] = 0.0f;
                    ov_2[2] = 0.0f;
                    ov_2[3] = 0.0f;
                    ov_2[4] = 0.0f;
                    ov_2[5] = 0.0f;
                    ov_2[6] = 0.0f;
                    ov_2[7] = 0.0f;
                    ov_2[8] = 0.0f;
                    ov_2[9] = 0.0f;
                    ov_2[10] = 0.0f;
                    ov_2[11] = 0.0f;
                    ov_2[12] = 0.0f;
                    ov_2[13] = 0.0f;
                    ov_2[14] = 0.0f;
                    ov_2[15] = 0.0f;
                    ov_2[16] = 0.0f;
                    ov_2[17] = 0.0f;
                    ov_2[18] = 0.0f;
                    ov_2[19] = 0.0f;
                    ov_2[20] = 0.0f;
                    ov_2[21] = 0.0f;
                    ov_2[22] = 0.0f;
                    ov_2[23] = 0.0f;
                    ov_2[24] = 0.0f;
                    ov_2[25] = 0.0f;
                    ov_2[26] = 0.0f;
                    ov_2[27] = 0.0f;
                    ov_2[28] = 0.0f;
                    ov_2[29] = 0.0f;
                    ov_2[30] = 0.0f;
                    ov_2[31] = 0.0f;
                    ov_2[32] = 0.0f;
                    ov_2[33] = 0.0f;
                    ov_2[34] = 0.0f;
                    ov_2[35] = 0.0f;
                    ov_2[36] = 0.0f;
                    ov_2[37] = 0.0f;
                    ov_2[38] = 0.0f;
                    ov_2[39] = 0.0f;
                    ov_2[40] = 0.0f;
                    ov_2[41] = 0.0f;
                    ov_2[42] = 0.0f;
                    ov_2[43] = 0.0f;
                    ov_2[44] = 0.0f;
                    ov_2[45] = 0.0f;
                    ov_2[46] = 0.0f;
                    ov_2[47] = 0.0f;
                    ov_2[48] = 0.0f;
                    ov_2[49] = 0.0f;
                    ov_2[50] = 0.0f;
                    ov_2[51] = 0.0f;
                    ov_2[52] = 0.0f;
                    ov_2[53] = 0.0f;
                    ov_2[54] = 0.0f;
                    ov_2[55] = 0.0f;
                    ov_2[56] = 0.0f;
                    ov_2[57] = 0.0f;
                    ov_2[58] = 0.0f;
                    ov_2[59] = 0.0f;
                    ov_2[60] = 0.0f;
                    ov_2[61] = 0.0f;
                    ov_2[62] = 0.0f;
                    ov_2[63] = 0.0f;
                }
                const float2 _scale2_48 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_48);
                int out_col_2 = 256 + row / 64 * 128 + 64;
                float res_11[16];
                unsigned int ov_bf16_24[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 0], ov_2[_lp*2+1 + 0]));
                    ov_bf16_24[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_24[_pair]) << 16);
                    (&unpacked_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_24[_pair]) & 0xffff0000u);
                }
                res_11[0] = ov_2[0] - unpacked_24[0];
                res_11[1] = ov_2[1] - unpacked_24[1];
                res_11[2] = ov_2[2] - unpacked_24[2];
                res_11[3] = ov_2[3] - unpacked_24[3];
                res_11[4] = ov_2[4] - unpacked_24[4];
                res_11[5] = ov_2[5] - unpacked_24[5];
                res_11[6] = ov_2[6] - unpacked_24[6];
                res_11[7] = ov_2[7] - unpacked_24[7];
                int byte_offset_24 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128;
                int swizzled_offset_25 = byte_offset_24 ^ (byte_offset_24 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_25), "r"(ov_bf16_24[0]), "r"(ov_bf16_24[1]), "r"(ov_bf16_24[2]), "r"(ov_bf16_24[3]) : "memory");
                unsigned int ov_bf16_25[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 8], ov_2[_lp*2+1 + 8]));
                    ov_bf16_25[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_25[_pair]) << 16);
                    (&unpacked_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_25[_pair]) & 0xffff0000u);
                }
                res_11[8] = ov_2[8] - unpacked_25[0];
                res_11[9] = ov_2[9] - unpacked_25[1];
                res_11[10] = ov_2[10] - unpacked_25[2];
                res_11[11] = ov_2[11] - unpacked_25[3];
                res_11[12] = ov_2[12] - unpacked_25[4];
                res_11[13] = ov_2[13] - unpacked_25[5];
                res_11[14] = ov_2[14] - unpacked_25[6];
                res_11[15] = ov_2[15] - unpacked_25[7];
                int byte_offset_25 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_26 = byte_offset_25 ^ (byte_offset_25 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_26), "r"(ov_bf16_25[0]), "r"(ov_bf16_25[1]), "r"(ov_bf16_25[2]), "r"(ov_bf16_25[3]) : "memory");
                unsigned int packed_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 0], res_11[_lp*2+1 + 0]));
                    packed_11[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 2], res_11[_lp*2+1 + 2]));
                    packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 4], res_11[_lp*2+1 + 4]));
                    packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 6], res_11[_lp*2+1 + 6]));
                    packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 8], res_11[_lp*2+1 + 8]));
                    packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 10], res_11[_lp*2+1 + 10]));
                    packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 12], res_11[_lp*2+1 + 12]));
                    packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 14], res_11[_lp*2+1 + 14]));
                    packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_49 = reinterpret_cast<const unsigned*>(packed_11);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2))), "r"(_raw_stv8_49[0]), "r"(_raw_stv8_49[1]), "r"(_raw_stv8_49[2]), "r"(_raw_stv8_49[3]), "r"(_raw_stv8_49[4]), "r"(_raw_stv8_49[5]), "r"(_raw_stv8_49[6]), "r"(_raw_stv8_49[7]) : "memory");
                }
                float res_12[16];
                unsigned int ov_bf16_26[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 16], ov_2[_lp*2+1 + 16]));
                    ov_bf16_26[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_26[_pair]) << 16);
                    (&unpacked_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_26[_pair]) & 0xffff0000u);
                }
                res_12[0] = ov_2[16] - unpacked_26[0];
                res_12[1] = ov_2[17] - unpacked_26[1];
                res_12[2] = ov_2[18] - unpacked_26[2];
                res_12[3] = ov_2[19] - unpacked_26[3];
                res_12[4] = ov_2[20] - unpacked_26[4];
                res_12[5] = ov_2[21] - unpacked_26[5];
                res_12[6] = ov_2[22] - unpacked_26[6];
                res_12[7] = ov_2[23] - unpacked_26[7];
                int byte_offset_26 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_27 = byte_offset_26 ^ (byte_offset_26 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_27), "r"(ov_bf16_26[0]), "r"(ov_bf16_26[1]), "r"(ov_bf16_26[2]), "r"(ov_bf16_26[3]) : "memory");
                unsigned int ov_bf16_27[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 24], ov_2[_lp*2+1 + 24]));
                    ov_bf16_27[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_27[_pair]) << 16);
                    (&unpacked_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_27[_pair]) & 0xffff0000u);
                }
                res_12[8] = ov_2[24] - unpacked_27[0];
                res_12[9] = ov_2[25] - unpacked_27[1];
                res_12[10] = ov_2[26] - unpacked_27[2];
                res_12[11] = ov_2[27] - unpacked_27[3];
                res_12[12] = ov_2[28] - unpacked_27[4];
                res_12[13] = ov_2[29] - unpacked_27[5];
                res_12[14] = ov_2[30] - unpacked_27[6];
                res_12[15] = ov_2[31] - unpacked_27[7];
                int byte_offset_27 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_28 = byte_offset_27 ^ (byte_offset_27 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_28), "r"(ov_bf16_27[0]), "r"(ov_bf16_27[1]), "r"(ov_bf16_27[2]), "r"(ov_bf16_27[3]) : "memory");
                unsigned int packed_12[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 0], res_12[_lp*2+1 + 0]));
                    packed_12[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 2], res_12[_lp*2+1 + 2]));
                    packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 4], res_12[_lp*2+1 + 4]));
                    packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 6], res_12[_lp*2+1 + 6]));
                    packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 8], res_12[_lp*2+1 + 8]));
                    packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 10], res_12[_lp*2+1 + 10]));
                    packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 12], res_12[_lp*2+1 + 12]));
                    packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 14], res_12[_lp*2+1 + 14]));
                    packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_50 = reinterpret_cast<const unsigned*>(packed_12);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 16ll))), "r"(_raw_stv8_50[0]), "r"(_raw_stv8_50[1]), "r"(_raw_stv8_50[2]), "r"(_raw_stv8_50[3]), "r"(_raw_stv8_50[4]), "r"(_raw_stv8_50[5]), "r"(_raw_stv8_50[6]), "r"(_raw_stv8_50[7]) : "memory");
                }
                float res_13[16];
                unsigned int ov_bf16_28[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 32], ov_2[_lp*2+1 + 32]));
                    ov_bf16_28[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_28[_pair]) << 16);
                    (&unpacked_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_28[_pair]) & 0xffff0000u);
                }
                res_13[0] = ov_2[32] - unpacked_28[0];
                res_13[1] = ov_2[33] - unpacked_28[1];
                res_13[2] = ov_2[34] - unpacked_28[2];
                res_13[3] = ov_2[35] - unpacked_28[3];
                res_13[4] = ov_2[36] - unpacked_28[4];
                res_13[5] = ov_2[37] - unpacked_28[5];
                res_13[6] = ov_2[38] - unpacked_28[6];
                res_13[7] = ov_2[39] - unpacked_28[7];
                int byte_offset_28 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_29 = byte_offset_28 ^ (byte_offset_28 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_29), "r"(ov_bf16_28[0]), "r"(ov_bf16_28[1]), "r"(ov_bf16_28[2]), "r"(ov_bf16_28[3]) : "memory");
                unsigned int ov_bf16_29[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 40], ov_2[_lp*2+1 + 40]));
                    ov_bf16_29[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_29[_pair]) << 16);
                    (&unpacked_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_29[_pair]) & 0xffff0000u);
                }
                res_13[8] = ov_2[40] - unpacked_29[0];
                res_13[9] = ov_2[41] - unpacked_29[1];
                res_13[10] = ov_2[42] - unpacked_29[2];
                res_13[11] = ov_2[43] - unpacked_29[3];
                res_13[12] = ov_2[44] - unpacked_29[4];
                res_13[13] = ov_2[45] - unpacked_29[5];
                res_13[14] = ov_2[46] - unpacked_29[6];
                res_13[15] = ov_2[47] - unpacked_29[7];
                int byte_offset_29 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_30 = byte_offset_29 ^ (byte_offset_29 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_30), "r"(ov_bf16_29[0]), "r"(ov_bf16_29[1]), "r"(ov_bf16_29[2]), "r"(ov_bf16_29[3]) : "memory");
                unsigned int packed_13[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 0], res_13[_lp*2+1 + 0]));
                    packed_13[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 2], res_13[_lp*2+1 + 2]));
                    packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 4], res_13[_lp*2+1 + 4]));
                    packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 6], res_13[_lp*2+1 + 6]));
                    packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 8], res_13[_lp*2+1 + 8]));
                    packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 10], res_13[_lp*2+1 + 10]));
                    packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 12], res_13[_lp*2+1 + 12]));
                    packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 14], res_13[_lp*2+1 + 14]));
                    packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_51 = reinterpret_cast<const unsigned*>(packed_13);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 32ll))), "r"(_raw_stv8_51[0]), "r"(_raw_stv8_51[1]), "r"(_raw_stv8_51[2]), "r"(_raw_stv8_51[3]), "r"(_raw_stv8_51[4]), "r"(_raw_stv8_51[5]), "r"(_raw_stv8_51[6]), "r"(_raw_stv8_51[7]) : "memory");
                }
                float res_14[16];
                unsigned int ov_bf16_30[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 48], ov_2[_lp*2+1 + 48]));
                    ov_bf16_30[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_30[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_30[_pair]) << 16);
                    (&unpacked_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_30[_pair]) & 0xffff0000u);
                }
                res_14[0] = ov_2[48] - unpacked_30[0];
                res_14[1] = ov_2[49] - unpacked_30[1];
                res_14[2] = ov_2[50] - unpacked_30[2];
                res_14[3] = ov_2[51] - unpacked_30[3];
                res_14[4] = ov_2[52] - unpacked_30[4];
                res_14[5] = ov_2[53] - unpacked_30[5];
                res_14[6] = ov_2[54] - unpacked_30[6];
                res_14[7] = ov_2[55] - unpacked_30[7];
                int byte_offset_30 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_31 = byte_offset_30 ^ (byte_offset_30 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_31), "r"(ov_bf16_30[0]), "r"(ov_bf16_30[1]), "r"(ov_bf16_30[2]), "r"(ov_bf16_30[3]) : "memory");
                unsigned int ov_bf16_31[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 56], ov_2[_lp*2+1 + 56]));
                    ov_bf16_31[_lp] = *(uint32_t*)&_bf2;
                }
                float unpacked_31[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&unpacked_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_31[_pair]) << 16);
                    (&unpacked_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_31[_pair]) & 0xffff0000u);
                }
                res_14[8] = ov_2[56] - unpacked_31[0];
                res_14[9] = ov_2[57] - unpacked_31[1];
                res_14[10] = ov_2[58] - unpacked_31[2];
                res_14[11] = ov_2[59] - unpacked_31[3];
                res_14[12] = ov_2[60] - unpacked_31[4];
                res_14[13] = ov_2[61] - unpacked_31[5];
                res_14[14] = ov_2[62] - unpacked_31[6];
                res_14[15] = ov_2[63] - unpacked_31[7];
                int byte_offset_31 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_32 = byte_offset_31 ^ (byte_offset_31 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_32), "r"(ov_bf16_31[0]), "r"(ov_bf16_31[1]), "r"(ov_bf16_31[2]), "r"(ov_bf16_31[3]) : "memory");
                unsigned int packed_14[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 0], res_14[_lp*2+1 + 0]));
                    packed_14[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 2], res_14[_lp*2+1 + 2]));
                    packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 4], res_14[_lp*2+1 + 4]));
                    packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 6], res_14[_lp*2+1 + 6]));
                    packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 8], res_14[_lp*2+1 + 8]));
                    packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 10], res_14[_lp*2+1 + 10]));
                    packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 12], res_14[_lp*2+1 + 12]));
                    packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 14], res_14[_lp*2+1 + 14]));
                    packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_52 = reinterpret_cast<const unsigned*>(packed_14);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 48ll))), "r"(_raw_stv8_52[0]), "r"(_raw_stv8_52[1]), "r"(_raw_stv8_52[2]), "r"(_raw_stv8_52[3]), "r"(_raw_stv8_52[4]), "r"(_raw_stv8_52[5]), "r"(_raw_stv8_52[6]), "r"(_raw_stv8_52[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_3 = 4 + warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_3, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_3 * 8192));
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                mbarrier_wait(clc_full_addr, _phase_o_written_waited);
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
                _phase_o_written_waited ^= 1;
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
            unsigned int _phase_q_copied = 0;
            #pragma unroll 1
            for (int outer_1 = 0; outer_1 < num_queries; outer_1++) {
                int active_1 = topk;
                if (derive_length != 0) {
                    int len_slot_1 = outer_1 % 2;
                    int _stage_len_ready_1 = (int)((unsigned int)outer_1 % 2);
                    int _parity_len_ready_1 = (int)((unsigned int)outer_1 / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready_1) * 8, _parity_len_ready_1);
                    int _max_1_1 = ((len_cells[len_slot_1]) > (0) ? (len_cells[len_slot_1]) : (0));
                    int _min_0_1 = ((_max_1_1) < (topk) ? (_max_1_1) : (topk));
                    active_1 = _min_0_1;
                } else if (has_topk_length != 0) {
                    int _max_3_1 = ((topk_length[query_1]) > (0) ? (topk_length[query_1]) : (0));
                    int _min_2_1 = ((_max_3_1) < (topk) ? (_max_3_1) : (topk));
                    active_1 = _min_2_1;
                }
                int active_0_1 = active_1;
                int blocks_1 = (((active_0_1 + 63) / 64) > (2) ? ((active_0_1 + 63) / 64) : (2));
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
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + producer_warp * 4)) + 0);
                                int4 _ivld_4;
                                _ivld_4 = *_ivptr_4;
                                _vec_load_0[0 + 0] = _ivld_4.x;
                                _vec_load_0[0 + 1] = _ivld_4.y;
                                _vec_load_0[0 + 2] = _ivld_4.z;
                                _vec_load_0[0 + 3] = _ivld_4.w;
                            }
                            idx[0] = _vec_load_0[0];
                            idx[1] = _vec_load_0[1];
                            idx[2] = _vec_load_0[2];
                            idx[3] = _vec_load_0[3];
                            int _max_7 = ((_vec_load_0[0]) > (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _max_8 = ((_vec_load_0[2]) > (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _max_6 = ((_max_7) > (_max_8) ? (_max_7) : (_max_8));
                            int _max_5 = ((largest) > (_max_6) ? (largest) : (_max_6));
                            largest = _max_5;
                            int _min_11 = ((_vec_load_0[0]) < (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _min_12 = ((_vec_load_0[2]) < (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _min_10 = ((_min_11) < (_min_12) ? (_min_11) : (_min_12));
                            int _min_9 = ((smallest) < (_min_10) ? (smallest) : (_min_10));
                            smallest = _min_9;
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_13 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 16 + producer_warp * 4)) + 0);
                                int4 _ivld_13;
                                _ivld_13 = *_ivptr_13;
                                _vec_load_1[0 + 0] = _ivld_13.x;
                                _vec_load_1[0 + 1] = _ivld_13.y;
                                _vec_load_1[0 + 2] = _ivld_13.z;
                                _vec_load_1[0 + 3] = _ivld_13.w;
                            }
                            idx[4] = _vec_load_1[0];
                            idx[5] = _vec_load_1[1];
                            idx[6] = _vec_load_1[2];
                            idx[7] = _vec_load_1[3];
                            int _max_16 = ((_vec_load_1[0]) > (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _max_17 = ((_vec_load_1[2]) > (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _max_15 = ((_max_16) > (_max_17) ? (_max_16) : (_max_17));
                            int _max_14 = ((largest) > (_max_15) ? (largest) : (_max_15));
                            largest = _max_14;
                            int _min_20 = ((_vec_load_1[0]) < (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _min_21 = ((_vec_load_1[2]) < (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _min_19 = ((_min_20) < (_min_21) ? (_min_20) : (_min_21));
                            int _min_18 = ((smallest) < (_min_19) ? (smallest) : (_min_19));
                            smallest = _min_18;
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_22 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 32 + producer_warp * 4)) + 0);
                                int4 _ivld_22;
                                _ivld_22 = *_ivptr_22;
                                _vec_load_2[0 + 0] = _ivld_22.x;
                                _vec_load_2[0 + 1] = _ivld_22.y;
                                _vec_load_2[0 + 2] = _ivld_22.z;
                                _vec_load_2[0 + 3] = _ivld_22.w;
                            }
                            idx[8] = _vec_load_2[0];
                            idx[9] = _vec_load_2[1];
                            idx[10] = _vec_load_2[2];
                            idx[11] = _vec_load_2[3];
                            int _max_25 = ((_vec_load_2[0]) > (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _max_26 = ((_vec_load_2[2]) > (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _max_24 = ((_max_25) > (_max_26) ? (_max_25) : (_max_26));
                            int _max_23 = ((largest) > (_max_24) ? (largest) : (_max_24));
                            largest = _max_23;
                            int _min_29 = ((_vec_load_2[0]) < (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _min_30 = ((_vec_load_2[2]) < (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _min_28 = ((_min_29) < (_min_30) ? (_min_29) : (_min_30));
                            int _min_27 = ((smallest) < (_min_28) ? (smallest) : (_min_28));
                            smallest = _min_27;
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_31 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 48 + producer_warp * 4)) + 0);
                                int4 _ivld_31;
                                _ivld_31 = *_ivptr_31;
                                _vec_load_3[0 + 0] = _ivld_31.x;
                                _vec_load_3[0 + 1] = _ivld_31.y;
                                _vec_load_3[0 + 2] = _ivld_31.z;
                                _vec_load_3[0 + 3] = _ivld_31.w;
                            }
                            idx[12] = _vec_load_3[0];
                            idx[13] = _vec_load_3[1];
                            idx[14] = _vec_load_3[2];
                            idx[15] = _vec_load_3[3];
                            int _max_34 = ((_vec_load_3[0]) > (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _max_35 = ((_vec_load_3[2]) > (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _max_33 = ((_max_34) > (_max_35) ? (_max_34) : (_max_35));
                            int _max_32 = ((largest) > (_max_33) ? (largest) : (_max_33));
                            largest = _max_32;
                            int _min_38 = ((_vec_load_3[0]) < (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _min_39 = ((_vec_load_3[2]) < (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _min_37 = ((_min_38) < (_min_39) ? (_min_38) : (_min_39));
                            int _min_36 = ((smallest) < (_min_37) ? (smallest) : (_min_37));
                            smallest = _min_36;
                        } else {
                            int position_1_0 = block_1 * 64 + producer_warp * 4;
                            int clamped_3_0 = ((position_1_0) < (topk - 1) ? (position_1_0) : (topk - 1));
                            int value_4_0 = indices[row_base + (long long)clamped_3_0];
                            if (position_1_0 >= active_0_1) {
                                value_4_0 = -1;
                            }
                            idx[0] = value_4_0;
                            int _max_40 = ((largest) > (value_4_0) ? (largest) : (value_4_0));
                            largest = _max_40;
                            int _min_41 = ((smallest) < (value_4_0) ? (smallest) : (value_4_0));
                            smallest = _min_41;
                            int position_1_1 = block_1 * 64 + producer_warp * 4 + 1;
                            int clamped_3_1 = ((position_1_1) < (topk - 1) ? (position_1_1) : (topk - 1));
                            int value_4_1 = indices[row_base + (long long)clamped_3_1];
                            if (position_1_1 >= active_0_1) {
                                value_4_1 = -1;
                            }
                            idx[1] = value_4_1;
                            int _max_42 = ((largest) > (value_4_1) ? (largest) : (value_4_1));
                            largest = _max_42;
                            int _min_43 = ((smallest) < (value_4_1) ? (smallest) : (value_4_1));
                            smallest = _min_43;
                            int position_1_2 = block_1 * 64 + producer_warp * 4 + 2;
                            int clamped_3_2 = ((position_1_2) < (topk - 1) ? (position_1_2) : (topk - 1));
                            int value_4_2 = indices[row_base + (long long)clamped_3_2];
                            if (position_1_2 >= active_0_1) {
                                value_4_2 = -1;
                            }
                            idx[2] = value_4_2;
                            int _max_44 = ((largest) > (value_4_2) ? (largest) : (value_4_2));
                            largest = _max_44;
                            int _min_45 = ((smallest) < (value_4_2) ? (smallest) : (value_4_2));
                            smallest = _min_45;
                            int position_1_3 = block_1 * 64 + producer_warp * 4 + 3;
                            int clamped_3_3 = ((position_1_3) < (topk - 1) ? (position_1_3) : (topk - 1));
                            int value_4_3 = indices[row_base + (long long)clamped_3_3];
                            if (position_1_3 >= active_0_1) {
                                value_4_3 = -1;
                            }
                            idx[3] = value_4_3;
                            int _max_46 = ((largest) > (value_4_3) ? (largest) : (value_4_3));
                            largest = _max_46;
                            int _min_47 = ((smallest) < (value_4_3) ? (smallest) : (value_4_3));
                            smallest = _min_47;
                            int position_1_4 = block_1 * 64 + 16 + producer_warp * 4;
                            int clamped_3_4 = ((position_1_4) < (topk - 1) ? (position_1_4) : (topk - 1));
                            int value_4_4 = indices[row_base + (long long)clamped_3_4];
                            if (position_1_4 >= active_0_1) {
                                value_4_4 = -1;
                            }
                            idx[4] = value_4_4;
                            int _max_48 = ((largest) > (value_4_4) ? (largest) : (value_4_4));
                            largest = _max_48;
                            int _min_49 = ((smallest) < (value_4_4) ? (smallest) : (value_4_4));
                            smallest = _min_49;
                            int position_1_5 = block_1 * 64 + 16 + producer_warp * 4 + 1;
                            int clamped_3_5 = ((position_1_5) < (topk - 1) ? (position_1_5) : (topk - 1));
                            int value_4_5 = indices[row_base + (long long)clamped_3_5];
                            if (position_1_5 >= active_0_1) {
                                value_4_5 = -1;
                            }
                            idx[5] = value_4_5;
                            int _max_50 = ((largest) > (value_4_5) ? (largest) : (value_4_5));
                            largest = _max_50;
                            int _min_51 = ((smallest) < (value_4_5) ? (smallest) : (value_4_5));
                            smallest = _min_51;
                            int position_1_6 = block_1 * 64 + 16 + producer_warp * 4 + 2;
                            int clamped_3_6 = ((position_1_6) < (topk - 1) ? (position_1_6) : (topk - 1));
                            int value_4_6 = indices[row_base + (long long)clamped_3_6];
                            if (position_1_6 >= active_0_1) {
                                value_4_6 = -1;
                            }
                            idx[6] = value_4_6;
                            int _max_52 = ((largest) > (value_4_6) ? (largest) : (value_4_6));
                            largest = _max_52;
                            int _min_53 = ((smallest) < (value_4_6) ? (smallest) : (value_4_6));
                            smallest = _min_53;
                            int position_1_7 = block_1 * 64 + 16 + producer_warp * 4 + 3;
                            int clamped_3_7 = ((position_1_7) < (topk - 1) ? (position_1_7) : (topk - 1));
                            int value_4_7 = indices[row_base + (long long)clamped_3_7];
                            if (position_1_7 >= active_0_1) {
                                value_4_7 = -1;
                            }
                            idx[7] = value_4_7;
                            int _max_54 = ((largest) > (value_4_7) ? (largest) : (value_4_7));
                            largest = _max_54;
                            int _min_55 = ((smallest) < (value_4_7) ? (smallest) : (value_4_7));
                            smallest = _min_55;
                            int position_1_8 = block_1 * 64 + 32 + producer_warp * 4;
                            int clamped_3_8 = ((position_1_8) < (topk - 1) ? (position_1_8) : (topk - 1));
                            int value_4_8 = indices[row_base + (long long)clamped_3_8];
                            if (position_1_8 >= active_0_1) {
                                value_4_8 = -1;
                            }
                            idx[8] = value_4_8;
                            int _max_56 = ((largest) > (value_4_8) ? (largest) : (value_4_8));
                            largest = _max_56;
                            int _min_57 = ((smallest) < (value_4_8) ? (smallest) : (value_4_8));
                            smallest = _min_57;
                            int position_1_9 = block_1 * 64 + 32 + producer_warp * 4 + 1;
                            int clamped_3_9 = ((position_1_9) < (topk - 1) ? (position_1_9) : (topk - 1));
                            int value_4_9 = indices[row_base + (long long)clamped_3_9];
                            if (position_1_9 >= active_0_1) {
                                value_4_9 = -1;
                            }
                            idx[9] = value_4_9;
                            int _max_58 = ((largest) > (value_4_9) ? (largest) : (value_4_9));
                            largest = _max_58;
                            int _min_59 = ((smallest) < (value_4_9) ? (smallest) : (value_4_9));
                            smallest = _min_59;
                            int position_1_10 = block_1 * 64 + 32 + producer_warp * 4 + 2;
                            int clamped_3_10 = ((position_1_10) < (topk - 1) ? (position_1_10) : (topk - 1));
                            int value_4_10 = indices[row_base + (long long)clamped_3_10];
                            if (position_1_10 >= active_0_1) {
                                value_4_10 = -1;
                            }
                            idx[10] = value_4_10;
                            int _max_60 = ((largest) > (value_4_10) ? (largest) : (value_4_10));
                            largest = _max_60;
                            int _min_61 = ((smallest) < (value_4_10) ? (smallest) : (value_4_10));
                            smallest = _min_61;
                            int position_1_11 = block_1 * 64 + 32 + producer_warp * 4 + 3;
                            int clamped_3_11 = ((position_1_11) < (topk - 1) ? (position_1_11) : (topk - 1));
                            int value_4_11 = indices[row_base + (long long)clamped_3_11];
                            if (position_1_11 >= active_0_1) {
                                value_4_11 = -1;
                            }
                            idx[11] = value_4_11;
                            int _max_62 = ((largest) > (value_4_11) ? (largest) : (value_4_11));
                            largest = _max_62;
                            int _min_63 = ((smallest) < (value_4_11) ? (smallest) : (value_4_11));
                            smallest = _min_63;
                            int position_1_12 = block_1 * 64 + 48 + producer_warp * 4;
                            int clamped_3_12 = ((position_1_12) < (topk - 1) ? (position_1_12) : (topk - 1));
                            int value_4_12 = indices[row_base + (long long)clamped_3_12];
                            if (position_1_12 >= active_0_1) {
                                value_4_12 = -1;
                            }
                            idx[12] = value_4_12;
                            int _max_64 = ((largest) > (value_4_12) ? (largest) : (value_4_12));
                            largest = _max_64;
                            int _min_65 = ((smallest) < (value_4_12) ? (smallest) : (value_4_12));
                            smallest = _min_65;
                            int position_1_13 = block_1 * 64 + 48 + producer_warp * 4 + 1;
                            int clamped_3_13 = ((position_1_13) < (topk - 1) ? (position_1_13) : (topk - 1));
                            int value_4_13 = indices[row_base + (long long)clamped_3_13];
                            if (position_1_13 >= active_0_1) {
                                value_4_13 = -1;
                            }
                            idx[13] = value_4_13;
                            int _max_66 = ((largest) > (value_4_13) ? (largest) : (value_4_13));
                            largest = _max_66;
                            int _min_67 = ((smallest) < (value_4_13) ? (smallest) : (value_4_13));
                            smallest = _min_67;
                            int position_1_14 = block_1 * 64 + 48 + producer_warp * 4 + 2;
                            int clamped_3_14 = ((position_1_14) < (topk - 1) ? (position_1_14) : (topk - 1));
                            int value_4_14 = indices[row_base + (long long)clamped_3_14];
                            if (position_1_14 >= active_0_1) {
                                value_4_14 = -1;
                            }
                            idx[14] = value_4_14;
                            int _max_68 = ((largest) > (value_4_14) ? (largest) : (value_4_14));
                            largest = _max_68;
                            int _min_69 = ((smallest) < (value_4_14) ? (smallest) : (value_4_14));
                            smallest = _min_69;
                            int position_1_15 = block_1 * 64 + 48 + producer_warp * 4 + 3;
                            int clamped_3_15 = ((position_1_15) < (topk - 1) ? (position_1_15) : (topk - 1));
                            int value_4_15 = indices[row_base + (long long)clamped_3_15];
                            if (position_1_15 >= active_0_1) {
                                value_4_15 = -1;
                            }
                            idx[15] = value_4_15;
                            int _max_70 = ((largest) > (value_4_15) ? (largest) : (value_4_15));
                            largest = _max_70;
                            int _min_71 = ((smallest) < (value_4_15) ? (smallest) : (value_4_15));
                            smallest = _min_71;
                        }
                        int skip = (smallest == num_kv || largest == -1) && block_1 >= 3;
                        if (block_1 == 1) {
                            mbarrier_wait(q_copied_addr, _phase_q_copied);
                        }
                        if (block_1 == 2) {
                            mbarrier_wait(o_written_addr, _phase_q_copied);
                        }
                        int slot_1 = cursor_1 % 3;
                        int _stage_sv_done_2 = (int)((unsigned int)cursor_1 % 3);
                        int _parity_sv_done_2 = (int)((unsigned int)cursor_1 / 3 % 2) ^ 1;
                        mbarrier_wait(sv_done_addr + (_stage_sv_done_2) * 8, _parity_sv_done_2);
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048), "l"((&kv_latent)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096), "l"((&kv_latent)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144), "l"((&kv_latent)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_lo_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_hi_addr + (slot_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_lo_addr + (slot_1) * 8), "r"((uint32_t)(8192)) : "memory");
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_hi_addr + (slot_1) * 8), "r"((uint32_t)(8192)) : "memory");
                        }
                        cursor_1 += 1;
                    }
                }
                if (elect_sync()) {
                    if (blocks_1 <= 2) {
                        mbarrier_wait(o_written_addr, _phase_q_copied);
                    }
                    mbarrier_arrive(o_written_waited_addr);
                }
                __syncwarp();
                mbarrier_wait(clc_full_addr, _phase_q_copied);
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
                _phase_q_copied ^= 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int query_2 = blockIdx.x;
            unsigned int outer_phase_2 = 0;
            int cursor_2 = 0;
            unsigned int _phase_qr_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int outer_2 = 0; outer_2 < num_queries; outer_2++) {
                    int active_2 = topk;
                    if (derive_length != 0) {
                        int len_slot_2 = outer_2 % 2;
                        int _stage_len_ready_2 = (int)((unsigned int)outer_2 % 2);
                        int _parity_len_ready_2 = (int)((unsigned int)outer_2 / 2 % 2);
                        mbarrier_wait(len_ready_addr + (_stage_len_ready_2) * 8, _parity_len_ready_2);
                        int _max_1_2 = ((len_cells[len_slot_2]) > (0) ? (len_cells[len_slot_2]) : (0));
                        int _min_0_2 = ((_max_1_2) < (topk) ? (_max_1_2) : (topk));
                        active_2 = _min_0_2;
                    } else if (has_topk_length != 0) {
                        int _max_3_2 = ((topk_length[query_2]) > (0) ? (topk_length[query_2]) : (0));
                        int _min_2_2 = ((_max_3_2) < (topk) ? (_max_3_2) : (topk));
                        active_2 = _min_2_2;
                    }
                    int active_0_2 = active_2;
                    int blocks_2 = (((active_0_2 + 63) / 64) > (2) ? ((active_0_2 + 63) / 64) : (2));
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
                    mbarrier_wait(qr_full_addr, _phase_qr_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(rope_addr)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(rope_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 384 + 0)), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(rope_addr + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(rope_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(512)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 384 + 8)), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    tcgen05_commit(qr_copied_addr);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    mbarrier_wait(q_full_addr, _phase_qr_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 0)), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 8)), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 16)), "l"(_tcgen05_cp_desc_8)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 24)), "l"(_tcgen05_cp_desc_9)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 0)), "l"(_tcgen05_cp_desc_10)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 8)), "l"(_tcgen05_cp_desc_11)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 16)), "l"(_tcgen05_cp_desc_12)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 24)), "l"(_tcgen05_cp_desc_13)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 0)), "l"(_tcgen05_cp_desc_14)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 8)), "l"(_tcgen05_cp_desc_15)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 16)), "l"(_tcgen05_cp_desc_16)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 24)), "l"(_tcgen05_cp_desc_17)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 0)), "l"(_tcgen05_cp_desc_18)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 8)), "l"(_tcgen05_cp_desc_19)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_20 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_20 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 16)), "l"(_tcgen05_cp_desc_20)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 24)), "l"(_tcgen05_cp_desc_21)
                            : "memory");
                    }
                    tcgen05_commit(q_copied_addr);
                    #pragma unroll 1
                    for (int block_2 = 0; block_2 < blocks_2 + 1; block_2++) {
                        if (blocks_2 > block_2) {
                            int slot_2 = cursor_2 % 3;
                            int _parity_p_free = (int)((unsigned int)cursor_2 % 2) ^ 1;
                            mbarrier_wait(p_free_addr, _parity_p_free);
                            int _parity_kr_ready = (int)((unsigned int)cursor_2 % 2);
                            mbarrier_wait(kr_ready_addr, _parity_kr_ready);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_22 = ((rope_addr) >> 4) & 0x7FFF;
#else
                            int _smem_desc_lo_22 = ((rope_addr) >> 4) & 0x3FFF;
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
                    :: "r"((tmem_tmem + (400))), "r"(_smem_desc_lo_22), "r"(tmem_tmem + 384), "r"(0) : "memory");
                            tcgen05_commit(qr_done_addr);
                            if (block_2 == 0) {
                                mbarrier_wait(q_copied_addr, _phase_qr_full);
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_lo_addr + (slot_2) * 8, 32768);
                            int _stage_kv_ready_lo = (int)((unsigned int)cursor_2 % 3);
                            int _parity_kv_ready_lo = (int)((unsigned int)cursor_2 / 3 % 2);
                            mbarrier_wait(kv_ready_lo_addr + (_stage_kv_ready_lo) * 8, _parity_kv_ready_lo);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_23 = (((k_dual_addr) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_23 = (((k_dual_addr) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    :: "r"((tmem_tmem + (400))), "r"(_smem_desc_lo_23), "r"(tmem_tmem + 256), "r"(1) : "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_hi_addr + (slot_2) * 8, 32768);
                            int _stage_kv_ready_hi = (int)((unsigned int)cursor_2 % 3);
                            int _parity_kv_ready_hi = (int)((unsigned int)cursor_2 / 3 % 2);
                            mbarrier_wait(kv_ready_hi_addr + (_stage_kv_ready_hi) * 8, _parity_kv_ready_hi);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_24 = (((k_dual_addr + 32768) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_24 = (((k_dual_addr + 32768) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    :: "r"((tmem_tmem + (400))), "r"(_smem_desc_lo_24), "r"(tmem_tmem + 320), "r"(1) : "memory");
                            tcgen05_commit(qk_done_addr + (slot_2) * 8);
                        }
                        if (block_2 > 0) {
                            int slot_3 = (cursor_2 - 1) % 3;
                            int _parity_so_ready = (int)((unsigned int)cursor_2 % 2) ^ 1;
                            mbarrier_wait(so_ready_addr, _parity_so_ready);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x7FFF) | 0x400000;
                            int _smem_desc_lo_25 = ((((v_addr) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x3FFF) | 0x400000;
                            int _smem_desc_lo_25 = ((((v_addr) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_25), "r"(tmem_tmem), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_26 = ((((v_addr + 32768) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _smem_desc_lo_26 = ((((v_addr + 32768) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_26), "r"((tmem_tmem + (128))), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
                            tcgen05_commit(sv_done_addr + (slot_3) * 8);
                        }
                        if (blocks_2 > block_2) {
                            cursor_2 += 1;
                        }
                    }
                    mbarrier_wait(clc_full_addr, _phase_qr_full);
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
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    query_2 = _clc_ctaid_x_2;
                    _phase_qr_full ^= 1;
                }
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 9) {
        { // metadata_main
            unsigned int _phase_clc_full_0 = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full = 0;
            unsigned int _phase_clc_full_0_1 = 0;
            if (lane < 8) {
                int query_3 = blockIdx.x;
                unsigned int outer_phase_3 = 0;
                int cursor_3 = 0;
                #pragma unroll 1
                for (int outer_3 = 0; outer_3 < num_queries; outer_3++) {
                    int active_3 = topk;
                    if (derive_length != 0) {
                        int len_slot_3 = outer_3 % 2;
                        int _stage_len_ready_3 = (int)((unsigned int)outer_3 % 2);
                        int _parity_len_ready_3 = (int)((unsigned int)outer_3 / 2 % 2);
                        mbarrier_wait(len_ready_addr + (_stage_len_ready_3) * 8, _parity_len_ready_3);
                        int _max_1_3 = ((len_cells[len_slot_3]) > (0) ? (len_cells[len_slot_3]) : (0));
                        int _min_0_3 = ((_max_1_3) < (topk) ? (_max_1_3) : (topk));
                        active_3 = _min_0_3;
                    } else if (has_topk_length != 0) {
                        int _max_3_3 = ((topk_length[query_3]) > (0) ? (topk_length[query_3]) : (0));
                        int _min_2_3 = ((_max_3_3) < (topk) ? (_max_3_3) : (topk));
                        active_3 = _min_2_3;
                    }
                    int active_0_3 = active_3;
                    int blocks_3 = (((active_0_3 + 63) / 64) > (2) ? ((active_0_3 + 63) / 64) : (2));
                    long long row_base_1 = (long long)indices_offset + (long long)query_3 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int block_3 = 0; block_3 < blocks_3; block_3++) {
                        unsigned int mask = 0;
                        int position = block_3 * 64 + lane * 8;
                        int whole_tile_1 = active_0_3 >= block_3 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (whole_tile_1 != 0) {
                            int _vec_load_4[8];
                            {
                                uint32_t _iv_4_0;
                                uint32_t _iv_4_1;
                                uint32_t _iv_4_2;
                                uint32_t _iv_4_3;
                                uint32_t _iv_4_4;
                                uint32_t _iv_4_5;
                                uint32_t _iv_4_6;
                                uint32_t _iv_4_7;
                                asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(indices + (row_base_1 + (long long)position) + (0))) : "memory");
                                _vec_load_4[0 + 0] = (int32_t)_iv_4_0;
                                _vec_load_4[0 + 1] = (int32_t)_iv_4_1;
                                _vec_load_4[0 + 2] = (int32_t)_iv_4_2;
                                _vec_load_4[0 + 3] = (int32_t)_iv_4_3;
                                _vec_load_4[0 + 4] = (int32_t)_iv_4_4;
                                _vec_load_4[0 + 5] = (int32_t)_iv_4_5;
                                _vec_load_4[0 + 6] = (int32_t)_iv_4_6;
                                _vec_load_4[0 + 7] = (int32_t)_iv_4_7;
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
                            int clamped_3_24 = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int value_4_24 = indices[row_base_1 + (long long)clamped_3_24];
                            if (value_4_24 >= 0 && value_4_24 < num_kv && active_0_3 > position) {
                                mask = mask | 1;
                            }
                            int clamped_3_25 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                            int value_4_25 = indices[row_base_1 + (long long)clamped_3_25];
                            if (value_4_25 >= 0 && value_4_25 < num_kv && active_0_3 > position + 1) {
                                mask = mask | 2;
                            }
                            int clamped_3_26 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                            int value_4_26 = indices[row_base_1 + (long long)clamped_3_26];
                            if (value_4_26 >= 0 && value_4_26 < num_kv && active_0_3 > position + 2) {
                                mask = mask | 4;
                            }
                            int clamped_3_27 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                            int value_4_27 = indices[row_base_1 + (long long)clamped_3_27];
                            if (value_4_27 >= 0 && value_4_27 < num_kv && active_0_3 > position + 3) {
                                mask = mask | 8;
                            }
                            int clamped_3_28 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                            int value_4_28 = indices[row_base_1 + (long long)clamped_3_28];
                            if (value_4_28 >= 0 && value_4_28 < num_kv && active_0_3 > position + 4) {
                                mask = mask | 16;
                            }
                            int clamped_3_29 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                            int value_4_29 = indices[row_base_1 + (long long)clamped_3_29];
                            if (value_4_29 >= 0 && value_4_29 < num_kv && active_0_3 > position + 5) {
                                mask = mask | 32;
                            }
                            int clamped_3_30 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                            int value_4_30 = indices[row_base_1 + (long long)clamped_3_30];
                            if (value_4_30 >= 0 && value_4_30 < num_kv && active_0_3 > position + 6) {
                                mask = mask | 64;
                            }
                            int clamped_3_31 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                            int value_4_31 = indices[row_base_1 + (long long)clamped_3_31];
                            if (value_4_31 >= 0 && value_4_31 < num_kv && active_0_3 > position + 7) {
                                mask = mask | 128;
                            }
                        }
                        int slot_4 = cursor_3 % 3;
                        int _stage_valid_free = (int)((unsigned int)cursor_3 % 3);
                        int _parity_valid_free = (int)((unsigned int)cursor_3 / 3 % 2) ^ 1;
                        mbarrier_wait(valid_free_addr + (_stage_valid_free) * 8, _parity_valid_free);
                        validity[slot_4 * 8 + lane] = mask;
                        mbarrier_arrive(valid_ready_addr + (slot_4) * 8);
                        cursor_3 += 1;
                    }
                    mbarrier_wait(clc_full_addr, _phase_clc_full_0);
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
                    _phase_clc_full_0 ^= 1;
                }
            } else if (lane == 8) {
                unsigned int outer_phase_4 = 0;
                #pragma unroll 1
                for (int outer_4 = 0; outer_4 < num_queries; outer_4++) {
                    mbarrier_wait(clc_empty_addr, _phase_clc_empty);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(clc_payload_addr + 0 * 16 + 0 * 16), "r"(clc_full_addr + 0 * 8)
                        : "memory");
                    mbarrier_arrive_expect_tx(clc_full_addr, 16);
                    mbarrier_wait(clc_full_addr, _phase_clc_full);
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
                    _phase_clc_empty ^= 1;
                    _phase_clc_full ^= 1;
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
                                    uint32_t _iv_5_0;
                                    uint32_t _iv_5_1;
                                    uint32_t _iv_5_2;
                                    uint32_t _iv_5_3;
                                    uint32_t _iv_5_4;
                                    uint32_t _iv_5_5;
                                    uint32_t _iv_5_6;
                                    uint32_t _iv_5_7;
                                    asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos) + (0))) : "memory");
                                    _vec_load_5[0 + 0] = (int32_t)_iv_5_0;
                                    _vec_load_5[0 + 1] = (int32_t)_iv_5_1;
                                    _vec_load_5[0 + 2] = (int32_t)_iv_5_2;
                                    _vec_load_5[0 + 3] = (int32_t)_iv_5_3;
                                    _vec_load_5[0 + 4] = (int32_t)_iv_5_4;
                                    _vec_load_5[0 + 5] = (int32_t)_iv_5_5;
                                    _vec_load_5[0 + 6] = (int32_t)_iv_5_6;
                                    _vec_load_5[0 + 7] = (int32_t)_iv_5_7;
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
                                int value = indices[row_base_2 + (long long)pos];
                                if (value >= 0 && value < num_kv) {
                                    last = pos + 1;
                                }
                            }
                            int part = last;
                            int peer_3_0 = __shfl_xor_sync(4294901760u, part, 8);
                            int _max_6_1 = ((part) > (peer_3_0) ? (part) : (peer_3_0));
                            part = _max_6_1;
                            int peer_3_1 = __shfl_xor_sync(4294901760u, part, 4);
                            int _max_7_1 = ((part) > (peer_3_1) ? (part) : (peer_3_1));
                            part = _max_7_1;
                            int peer_3_2 = __shfl_xor_sync(4294901760u, part, 2);
                            int _max_8_1 = ((part) > (peer_3_2) ? (part) : (peer_3_2));
                            part = _max_8_1;
                            int peer_3_3 = __shfl_xor_sync(4294901760u, part, 1);
                            int _max_9 = ((part) > (peer_3_3) ? (part) : (peer_3_3));
                            part = _max_9;
                            if (lane == 16) {
                                len_cells[outer_5 & 1] = part;
                                topk_length[(long long)query_4] = part;
                                mbarrier_arrive(len_ready_addr + ((int)((unsigned int)outer_5 % 2)) * 8);
                            }
                        }
                        mbarrier_wait(clc_full_addr, _phase_clc_full_0_1);
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
                        _phase_clc_full_0_1 ^= 1;
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
            unsigned int _phase_qr_copied = 0;
            #pragma unroll 1
            for (int outer_6 = 0; outer_6 < num_queries; outer_6++) {
                int active_4 = topk;
                if (derive_length != 0) {
                    int len_slot_4 = outer_6 % 2;
                    int _stage_len_ready_4 = (int)((unsigned int)outer_6 % 2);
                    int _parity_len_ready_4 = (int)((unsigned int)outer_6 / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready_4) * 8, _parity_len_ready_4);
                    int _max_1_4 = ((len_cells[len_slot_4]) > (0) ? (len_cells[len_slot_4]) : (0));
                    int _min_0_4 = ((_max_1_4) < (topk) ? (_max_1_4) : (topk));
                    active_4 = _min_0_4;
                } else if (has_topk_length != 0) {
                    int _max_3_4 = ((topk_length[query_5]) > (0) ? (topk_length[query_5]) : (0));
                    int _min_2_4 = ((_max_3_4) < (topk) ? (_max_3_4) : (topk));
                    active_4 = _min_2_4;
                }
                int active_0_4 = active_4;
                int blocks_4 = (((active_0_4 + 63) / 64) > (2) ? ((active_0_4 + 63) / 64) : (2));
                long long row_base_3 = (long long)indices_offset + (long long)query_5 * (long long)idx_stride;
                #pragma unroll 1
                for (int block_4 = 0; block_4 < blocks_4; block_4++) {
                    int index_values[8];
                    int position_1_16 = block_4 * 64 + group;
                    int clamped_3_16 = ((position_1_16) < (topk - 1) ? (position_1_16) : (topk - 1));
                    int value_4_16 = indices[row_base_3 + (long long)clamped_3_16];
                    if (position_1_16 >= active_0_4) {
                        value_4_16 = -1;
                    }
                    index_values[0] = value_4_16;
                    int position_1_17 = block_4 * 64 + group + 8;
                    int clamped_3_17 = ((position_1_17) < (topk - 1) ? (position_1_17) : (topk - 1));
                    int value_4_17 = indices[row_base_3 + (long long)clamped_3_17];
                    if (position_1_17 >= active_0_4) {
                        value_4_17 = -1;
                    }
                    index_values[1] = value_4_17;
                    int position_1_18 = block_4 * 64 + group + 16;
                    int clamped_3_18 = ((position_1_18) < (topk - 1) ? (position_1_18) : (topk - 1));
                    int value_4_18 = indices[row_base_3 + (long long)clamped_3_18];
                    if (position_1_18 >= active_0_4) {
                        value_4_18 = -1;
                    }
                    index_values[2] = value_4_18;
                    int position_1_19 = block_4 * 64 + group + 24;
                    int clamped_3_19 = ((position_1_19) < (topk - 1) ? (position_1_19) : (topk - 1));
                    int value_4_19 = indices[row_base_3 + (long long)clamped_3_19];
                    if (position_1_19 >= active_0_4) {
                        value_4_19 = -1;
                    }
                    index_values[3] = value_4_19;
                    int position_1_20 = block_4 * 64 + group + 32;
                    int clamped_3_20 = ((position_1_20) < (topk - 1) ? (position_1_20) : (topk - 1));
                    int value_4_20 = indices[row_base_3 + (long long)clamped_3_20];
                    if (position_1_20 >= active_0_4) {
                        value_4_20 = -1;
                    }
                    index_values[4] = value_4_20;
                    int position_1_21 = block_4 * 64 + group + 40;
                    int clamped_3_21 = ((position_1_21) < (topk - 1) ? (position_1_21) : (topk - 1));
                    int value_4_21 = indices[row_base_3 + (long long)clamped_3_21];
                    if (position_1_21 >= active_0_4) {
                        value_4_21 = -1;
                    }
                    index_values[5] = value_4_21;
                    int position_1_22 = block_4 * 64 + group + 48;
                    int clamped_3_22 = ((position_1_22) < (topk - 1) ? (position_1_22) : (topk - 1));
                    int value_4_22 = indices[row_base_3 + (long long)clamped_3_22];
                    if (position_1_22 >= active_0_4) {
                        value_4_22 = -1;
                    }
                    index_values[6] = value_4_22;
                    int position_1_23 = block_4 * 64 + group + 56;
                    int clamped_3_23 = ((position_1_23) < (topk - 1) ? (position_1_23) : (topk - 1));
                    int value_4_23 = indices[row_base_3 + (long long)clamped_3_23];
                    if (position_1_23 >= active_0_4) {
                        value_4_23 = -1;
                    }
                    index_values[7] = value_4_23;
                    int _parity_qr_done = (int)((unsigned int)cursor_4 % 2) ^ 1;
                    mbarrier_wait(qr_done_addr, _parity_qr_done);
                    if (block_4 == 0) {
                        mbarrier_wait(qr_copied_addr, _phase_qr_copied);
                    }
                    int raw_offset_1_0 = (col / 32 * 2048 + group * 32 + col % 32) * 2;
                    int swizzled_offset_1_0 = raw_offset_1_0 ^ (raw_offset_1_0 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_0), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[0] * (long long)k_rope_stride + (long long)col)), "r"((index_values[0] >= 0 && index_values[0] < num_kv) ? 16 : 0));
                    int raw_offset_1_1 = (col / 32 * 2048 + (group + 8) * 32 + col % 32) * 2;
                    int swizzled_offset_1_1 = raw_offset_1_1 ^ (raw_offset_1_1 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_1), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[1] * (long long)k_rope_stride + (long long)col)), "r"((index_values[1] >= 0 && index_values[1] < num_kv) ? 16 : 0));
                    int raw_offset_1_2 = (col / 32 * 2048 + (group + 16) * 32 + col % 32) * 2;
                    int swizzled_offset_1_2 = raw_offset_1_2 ^ (raw_offset_1_2 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_2), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[2] * (long long)k_rope_stride + (long long)col)), "r"((index_values[2] >= 0 && index_values[2] < num_kv) ? 16 : 0));
                    int raw_offset_1_3 = (col / 32 * 2048 + (group + 24) * 32 + col % 32) * 2;
                    int swizzled_offset_1_3 = raw_offset_1_3 ^ (raw_offset_1_3 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_3), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[3] * (long long)k_rope_stride + (long long)col)), "r"((index_values[3] >= 0 && index_values[3] < num_kv) ? 16 : 0));
                    int raw_offset_1_4 = (col / 32 * 2048 + (group + 32) * 32 + col % 32) * 2;
                    int swizzled_offset_1_4 = raw_offset_1_4 ^ (raw_offset_1_4 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_4), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[4] * (long long)k_rope_stride + (long long)col)), "r"((index_values[4] >= 0 && index_values[4] < num_kv) ? 16 : 0));
                    int raw_offset_1_5 = (col / 32 * 2048 + (group + 40) * 32 + col % 32) * 2;
                    int swizzled_offset_1_5 = raw_offset_1_5 ^ (raw_offset_1_5 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_5), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[5] * (long long)k_rope_stride + (long long)col)), "r"((index_values[5] >= 0 && index_values[5] < num_kv) ? 16 : 0));
                    int raw_offset_1_6 = (col / 32 * 2048 + (group + 48) * 32 + col % 32) * 2;
                    int swizzled_offset_1_6 = raw_offset_1_6 ^ (raw_offset_1_6 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_6), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[6] * (long long)k_rope_stride + (long long)col)), "r"((index_values[6] >= 0 && index_values[6] < num_kv) ? 16 : 0));
                    int raw_offset_1_7 = (col / 32 * 2048 + (group + 56) * 32 + col % 32) * 2;
                    int swizzled_offset_1_7 = raw_offset_1_7 ^ (raw_offset_1_7 >> 7 & 3) << 4;
                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                        :: "r"(rope_addr + (unsigned int)swizzled_offset_1_7), "l"(k_rope + ((long long)k_rope_offset + (long long)index_values[7] * (long long)k_rope_stride + (long long)col)), "r"((index_values[7] >= 0 && index_values[7] < num_kv) ? 16 : 0));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(kr_ready_addr) : "memory");
                    cursor_4 += 1;
                }
                mbarrier_wait(clc_full_addr, _phase_qr_copied);
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
                _phase_qr_copied ^= 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
