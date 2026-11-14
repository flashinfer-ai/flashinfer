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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_46cdc056c891876ed545(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int derive_length, float scale_log2, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ indices_out, int num_docs, int search_iters, int causal, int convert)
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
    float* exchange = reinterpret_cast<float*>(smem_raw + 197632);
    const int exchange_addr = smem + 197632;
    __nv_bfloat16* probability = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int probability_addr = smem + 214016;
    uint8_t* validity = reinterpret_cast<uint8_t*>(smem_raw + 222208);
    const int validity_addr = smem + 222208;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int validity32_addr = smem + 222208;
    float* stats = reinterpret_cast<float*>(smem_raw + 222336);
    const int stats_addr = smem + 222336;
    unsigned int* clc_payload = reinterpret_cast<unsigned int*>(smem_raw + 223360);
    const int clc_payload_addr = smem + 223360;
    int* len_cells = reinterpret_cast<int*>(smem_raw + 223408);
    const int len_cells_addr = smem + 223408;
    int* bound_cells = reinterpret_cast<int*>(smem_raw + 223416);
    const int bound_cells_addr = smem + 223416;

    // Kernel setup ops
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (28 barriers at smem_raw[0..224))

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
            // clc_empty: 1 barriers, init_count=282
            mbarrier_init(smem + 24, 282);
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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 2) {
        int _tmem_hold = smem + 224;
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
                asm volatile("cp.async.bulk.wait_group.read 0;");
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
                            : "r"(tmrow + 384));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 416));
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 384));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31])
                            : "r"(tmrow + 416));
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
                float ov_0[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_0[0]), "=f"(ov_0[1]), "=f"(ov_0[2]), "=f"(ov_0[3]), "=f"(ov_0[4]), "=f"(ov_0[5]), "=f"(ov_0[6]), "=f"(ov_0[7]), "=f"(ov_0[8]), "=f"(ov_0[9]), "=f"(ov_0[10]), "=f"(ov_0[11]), "=f"(ov_0[12]), "=f"(ov_0[13]), "=f"(ov_0[14]), "=f"(ov_0[15]), "=f"(ov_0[16]), "=f"(ov_0[17]), "=f"(ov_0[18]), "=f"(ov_0[19]), "=f"(ov_0[20]), "=f"(ov_0[21]), "=f"(ov_0[22]), "=f"(ov_0[23]), "=f"(ov_0[24]), "=f"(ov_0[25]), "=f"(ov_0[26]), "=f"(ov_0[27]), "=f"(ov_0[28]), "=f"(ov_0[29]), "=f"(ov_0[30]), "=f"(ov_0[31]), "=f"(ov_0[32]), "=f"(ov_0[33]), "=f"(ov_0[34]), "=f"(ov_0[35]), "=f"(ov_0[36]), "=f"(ov_0[37]), "=f"(ov_0[38]), "=f"(ov_0[39]), "=f"(ov_0[40]), "=f"(ov_0[41]), "=f"(ov_0[42]), "=f"(ov_0[43]), "=f"(ov_0[44]), "=f"(ov_0[45]), "=f"(ov_0[46]), "=f"(ov_0[47]), "=f"(ov_0[48]), "=f"(ov_0[49]), "=f"(ov_0[50]), "=f"(ov_0[51]), "=f"(ov_0[52]), "=f"(ov_0[53]), "=f"(ov_0[54]), "=f"(ov_0[55]), "=f"(ov_0[56]), "=f"(ov_0[57]), "=f"(ov_0[58]), "=f"(ov_0[59]), "=f"(ov_0[60]), "=f"(ov_0[61]), "=f"(ov_0[62]), "=f"(ov_0[63])
                        : "r"(tmrow));
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
                const float2 _scale2_33 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_0)[_ls], _scale2_33);
                int out_col_0 = row / 64 * 128;
                float residual_0[16];
                unsigned int ov_bf16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 0], ov_0[_lp*2+1 + 0]));
                    ov_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_0[_pair]) << 16);
                    (&_unpack_bf16_0_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_0[_pair]) & 0xffff0000u);
                }
                residual_0[0] = ov_0[0] - _unpack_bf16_0_0[0];
                residual_0[1] = ov_0[1] - _unpack_bf16_0_0[1];
                residual_0[2] = ov_0[2] - _unpack_bf16_0_0[2];
                residual_0[3] = ov_0[3] - _unpack_bf16_0_0[3];
                residual_0[4] = ov_0[4] - _unpack_bf16_0_0[4];
                residual_0[5] = ov_0[5] - _unpack_bf16_0_0[5];
                residual_0[6] = ov_0[6] - _unpack_bf16_0_0[6];
                residual_0[7] = ov_0[7] - _unpack_bf16_0_0[7];
                int byte_offset_0 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128;
                int swizzled_offset_0 = byte_offset_0 ^ (byte_offset_0 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_0), "r"(ov_bf16_0[0]), "r"(ov_bf16_0[1]), "r"(ov_bf16_0[2]), "r"(ov_bf16_0[3]) : "memory");
                unsigned int ov_bf16_1_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 8], ov_0[_lp*2+1 + 8]));
                    ov_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_0[_pair]) << 16);
                    (&_unpack_bf16_1_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_0[_pair]) & 0xffff0000u);
                }
                residual_0[8] = ov_0[8] - _unpack_bf16_1_0[0];
                residual_0[9] = ov_0[9] - _unpack_bf16_1_0[1];
                residual_0[10] = ov_0[10] - _unpack_bf16_1_0[2];
                residual_0[11] = ov_0[11] - _unpack_bf16_1_0[3];
                residual_0[12] = ov_0[12] - _unpack_bf16_1_0[4];
                residual_0[13] = ov_0[13] - _unpack_bf16_1_0[5];
                residual_0[14] = ov_0[14] - _unpack_bf16_1_0[6];
                residual_0[15] = ov_0[15] - _unpack_bf16_1_0[7];
                int byte_offset_2_0 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3_0 = byte_offset_2_0 ^ (byte_offset_2_0 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3_0), "r"(ov_bf16_1_0[0]), "r"(ov_bf16_1_0[1]), "r"(ov_bf16_1_0[2]), "r"(ov_bf16_1_0[3]) : "memory");
                unsigned int _packed_0_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 0], residual_0[_lp*2+1 + 0]));
                    _packed_0_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 2], residual_0[_lp*2+1 + 2]));
                    _packed_0_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 4], residual_0[_lp*2+1 + 4]));
                    _packed_0_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 6], residual_0[_lp*2+1 + 6]));
                    _packed_0_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 8], residual_0[_lp*2+1 + 8]));
                    _packed_0_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 10], residual_0[_lp*2+1 + 10]));
                    _packed_0_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 12], residual_0[_lp*2+1 + 12]));
                    _packed_0_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_0[_lp*2 + 14], residual_0[_lp*2+1 + 14]));
                    _packed_0_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_34 = reinterpret_cast<const unsigned*>(_packed_0_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0))), "r"(_raw_stv8_34[0]), "r"(_raw_stv8_34[1]), "r"(_raw_stv8_34[2]), "r"(_raw_stv8_34[3]), "r"(_raw_stv8_34[4]), "r"(_raw_stv8_34[5]), "r"(_raw_stv8_34[6]), "r"(_raw_stv8_34[7]) : "memory");
                }
                float residual_r_0[16];
                unsigned int ov_bf16_2_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 16], ov_0[_lp*2+1 + 16]));
                    ov_bf16_2_0[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_0[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_0[_pair]) & 0xffff0000u);
                }
                residual_r_0[0] = ov_0[16] - _unpack_bf16_2[0];
                residual_r_0[1] = ov_0[17] - _unpack_bf16_2[1];
                residual_r_0[2] = ov_0[18] - _unpack_bf16_2[2];
                residual_r_0[3] = ov_0[19] - _unpack_bf16_2[3];
                residual_r_0[4] = ov_0[20] - _unpack_bf16_2[4];
                residual_r_0[5] = ov_0[21] - _unpack_bf16_2[5];
                residual_r_0[6] = ov_0[22] - _unpack_bf16_2[6];
                residual_r_0[7] = ov_0[23] - _unpack_bf16_2[7];
                int byte_offset_1_0 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_1_0 = byte_offset_1_0 ^ (byte_offset_1_0 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_0), "r"(ov_bf16_2_0[0]), "r"(ov_bf16_2_0[1]), "r"(ov_bf16_2_0[2]), "r"(ov_bf16_2_0[3]) : "memory");
                unsigned int ov_bf16_2_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 24], ov_0[_lp*2+1 + 24]));
                    ov_bf16_2_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_1[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_1[_pair]) & 0xffff0000u);
                }
                residual_r_0[8] = ov_0[24] - _unpack_bf16_3[0];
                residual_r_0[9] = ov_0[25] - _unpack_bf16_3[1];
                residual_r_0[10] = ov_0[26] - _unpack_bf16_3[2];
                residual_r_0[11] = ov_0[27] - _unpack_bf16_3[3];
                residual_r_0[12] = ov_0[28] - _unpack_bf16_3[4];
                residual_r_0[13] = ov_0[29] - _unpack_bf16_3[5];
                residual_r_0[14] = ov_0[30] - _unpack_bf16_3[6];
                residual_r_0[15] = ov_0[31] - _unpack_bf16_3[7];
                int byte_offset_1_1 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_1_1 = byte_offset_1_1 ^ (byte_offset_1_1 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_1), "r"(ov_bf16_2_1[0]), "r"(ov_bf16_2_1[1]), "r"(ov_bf16_2_1[2]), "r"(ov_bf16_2_1[3]) : "memory");
                unsigned int packed_r_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 0], residual_r_0[_lp*2+1 + 0]));
                    packed_r_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 2], residual_r_0[_lp*2+1 + 2]));
                    packed_r_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 4], residual_r_0[_lp*2+1 + 4]));
                    packed_r_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 6], residual_r_0[_lp*2+1 + 6]));
                    packed_r_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 8], residual_r_0[_lp*2+1 + 8]));
                    packed_r_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 10], residual_r_0[_lp*2+1 + 10]));
                    packed_r_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 12], residual_r_0[_lp*2+1 + 12]));
                    packed_r_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_0[_lp*2 + 14], residual_r_0[_lp*2+1 + 14]));
                    packed_r_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_35 = reinterpret_cast<const unsigned*>(packed_r_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 16ll))), "r"(_raw_stv8_35[0]), "r"(_raw_stv8_35[1]), "r"(_raw_stv8_35[2]), "r"(_raw_stv8_35[3]), "r"(_raw_stv8_35[4]), "r"(_raw_stv8_35[5]), "r"(_raw_stv8_35[6]), "r"(_raw_stv8_35[7]) : "memory");
                }
                float residual_r_1[16];
                unsigned int ov_bf16_2_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 32], ov_0[_lp*2+1 + 32]));
                    ov_bf16_2_2[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_2[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_2[_pair]) & 0xffff0000u);
                }
                residual_r_1[0] = ov_0[32] - _unpack_bf16_4[0];
                residual_r_1[1] = ov_0[33] - _unpack_bf16_4[1];
                residual_r_1[2] = ov_0[34] - _unpack_bf16_4[2];
                residual_r_1[3] = ov_0[35] - _unpack_bf16_4[3];
                residual_r_1[4] = ov_0[36] - _unpack_bf16_4[4];
                residual_r_1[5] = ov_0[37] - _unpack_bf16_4[5];
                residual_r_1[6] = ov_0[38] - _unpack_bf16_4[6];
                residual_r_1[7] = ov_0[39] - _unpack_bf16_4[7];
                int byte_offset_1_2 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_1_2 = byte_offset_1_2 ^ (byte_offset_1_2 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_2), "r"(ov_bf16_2_2[0]), "r"(ov_bf16_2_2[1]), "r"(ov_bf16_2_2[2]), "r"(ov_bf16_2_2[3]) : "memory");
                unsigned int ov_bf16_2_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 40], ov_0[_lp*2+1 + 40]));
                    ov_bf16_2_3[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_3[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_3[_pair]) & 0xffff0000u);
                }
                residual_r_1[8] = ov_0[40] - _unpack_bf16_5[0];
                residual_r_1[9] = ov_0[41] - _unpack_bf16_5[1];
                residual_r_1[10] = ov_0[42] - _unpack_bf16_5[2];
                residual_r_1[11] = ov_0[43] - _unpack_bf16_5[3];
                residual_r_1[12] = ov_0[44] - _unpack_bf16_5[4];
                residual_r_1[13] = ov_0[45] - _unpack_bf16_5[5];
                residual_r_1[14] = ov_0[46] - _unpack_bf16_5[6];
                residual_r_1[15] = ov_0[47] - _unpack_bf16_5[7];
                int byte_offset_1_3 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_1_3 = byte_offset_1_3 ^ (byte_offset_1_3 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_3), "r"(ov_bf16_2_3[0]), "r"(ov_bf16_2_3[1]), "r"(ov_bf16_2_3[2]), "r"(ov_bf16_2_3[3]) : "memory");
                unsigned int packed_r_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 0], residual_r_1[_lp*2+1 + 0]));
                    packed_r_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 2], residual_r_1[_lp*2+1 + 2]));
                    packed_r_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 4], residual_r_1[_lp*2+1 + 4]));
                    packed_r_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 6], residual_r_1[_lp*2+1 + 6]));
                    packed_r_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 8], residual_r_1[_lp*2+1 + 8]));
                    packed_r_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 10], residual_r_1[_lp*2+1 + 10]));
                    packed_r_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 12], residual_r_1[_lp*2+1 + 12]));
                    packed_r_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_1[_lp*2 + 14], residual_r_1[_lp*2+1 + 14]));
                    packed_r_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_36 = reinterpret_cast<const unsigned*>(packed_r_1);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 32ll))), "r"(_raw_stv8_36[0]), "r"(_raw_stv8_36[1]), "r"(_raw_stv8_36[2]), "r"(_raw_stv8_36[3]), "r"(_raw_stv8_36[4]), "r"(_raw_stv8_36[5]), "r"(_raw_stv8_36[6]), "r"(_raw_stv8_36[7]) : "memory");
                }
                float residual_r_2[16];
                unsigned int ov_bf16_2_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 48], ov_0[_lp*2+1 + 48]));
                    ov_bf16_2_4[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_4[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_4[_pair]) & 0xffff0000u);
                }
                residual_r_2[0] = ov_0[48] - _unpack_bf16_6[0];
                residual_r_2[1] = ov_0[49] - _unpack_bf16_6[1];
                residual_r_2[2] = ov_0[50] - _unpack_bf16_6[2];
                residual_r_2[3] = ov_0[51] - _unpack_bf16_6[3];
                residual_r_2[4] = ov_0[52] - _unpack_bf16_6[4];
                residual_r_2[5] = ov_0[53] - _unpack_bf16_6[5];
                residual_r_2[6] = ov_0[54] - _unpack_bf16_6[6];
                residual_r_2[7] = ov_0[55] - _unpack_bf16_6[7];
                int byte_offset_1_4 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_1_4 = byte_offset_1_4 ^ (byte_offset_1_4 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_4), "r"(ov_bf16_2_4[0]), "r"(ov_bf16_2_4[1]), "r"(ov_bf16_2_4[2]), "r"(ov_bf16_2_4[3]) : "memory");
                unsigned int ov_bf16_2_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 56], ov_0[_lp*2+1 + 56]));
                    ov_bf16_2_5[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_5[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_5[_pair]) & 0xffff0000u);
                }
                residual_r_2[8] = ov_0[56] - _unpack_bf16_7[0];
                residual_r_2[9] = ov_0[57] - _unpack_bf16_7[1];
                residual_r_2[10] = ov_0[58] - _unpack_bf16_7[2];
                residual_r_2[11] = ov_0[59] - _unpack_bf16_7[3];
                residual_r_2[12] = ov_0[60] - _unpack_bf16_7[4];
                residual_r_2[13] = ov_0[61] - _unpack_bf16_7[5];
                residual_r_2[14] = ov_0[62] - _unpack_bf16_7[6];
                residual_r_2[15] = ov_0[63] - _unpack_bf16_7[7];
                int byte_offset_1_5 = o_slot * 65536 + out_col_0 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_1_5 = byte_offset_1_5 ^ (byte_offset_1_5 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_5), "r"(ov_bf16_2_5[0]), "r"(ov_bf16_2_5[1]), "r"(ov_bf16_2_5[2]), "r"(ov_bf16_2_5[3]) : "memory");
                unsigned int packed_r_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 0], residual_r_2[_lp*2+1 + 0]));
                    packed_r_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 2], residual_r_2[_lp*2+1 + 2]));
                    packed_r_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 4], residual_r_2[_lp*2+1 + 4]));
                    packed_r_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 6], residual_r_2[_lp*2+1 + 6]));
                    packed_r_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 8], residual_r_2[_lp*2+1 + 8]));
                    packed_r_2[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 10], residual_r_2[_lp*2+1 + 10]));
                    packed_r_2[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 12], residual_r_2[_lp*2+1 + 12]));
                    packed_r_2[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_2[_lp*2 + 14], residual_r_2[_lp*2+1 + 14]));
                    packed_r_2[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(packed_r_2);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_0 + 48ll))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_0 = warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk_0, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_0 * 8192));
                    }
                }
                float ov_1[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_1[0]), "=f"(ov_1[1]), "=f"(ov_1[2]), "=f"(ov_1[3]), "=f"(ov_1[4]), "=f"(ov_1[5]), "=f"(ov_1[6]), "=f"(ov_1[7]), "=f"(ov_1[8]), "=f"(ov_1[9]), "=f"(ov_1[10]), "=f"(ov_1[11]), "=f"(ov_1[12]), "=f"(ov_1[13]), "=f"(ov_1[14]), "=f"(ov_1[15]), "=f"(ov_1[16]), "=f"(ov_1[17]), "=f"(ov_1[18]), "=f"(ov_1[19]), "=f"(ov_1[20]), "=f"(ov_1[21]), "=f"(ov_1[22]), "=f"(ov_1[23]), "=f"(ov_1[24]), "=f"(ov_1[25]), "=f"(ov_1[26]), "=f"(ov_1[27]), "=f"(ov_1[28]), "=f"(ov_1[29]), "=f"(ov_1[30]), "=f"(ov_1[31]), "=f"(ov_1[32]), "=f"(ov_1[33]), "=f"(ov_1[34]), "=f"(ov_1[35]), "=f"(ov_1[36]), "=f"(ov_1[37]), "=f"(ov_1[38]), "=f"(ov_1[39]), "=f"(ov_1[40]), "=f"(ov_1[41]), "=f"(ov_1[42]), "=f"(ov_1[43]), "=f"(ov_1[44]), "=f"(ov_1[45]), "=f"(ov_1[46]), "=f"(ov_1[47]), "=f"(ov_1[48]), "=f"(ov_1[49]), "=f"(ov_1[50]), "=f"(ov_1[51]), "=f"(ov_1[52]), "=f"(ov_1[53]), "=f"(ov_1[54]), "=f"(ov_1[55]), "=f"(ov_1[56]), "=f"(ov_1[57]), "=f"(ov_1[58]), "=f"(ov_1[59]), "=f"(ov_1[60]), "=f"(ov_1[61]), "=f"(ov_1[62]), "=f"(ov_1[63])
                        : "r"(tmrow + 64));
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
                const float2 _scale2_38 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_38);
                int out_col_1 = row / 64 * 128 + 64;
                float residual_1[16];
                unsigned int ov_bf16_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 0], ov_1[_lp*2+1 + 0]));
                    ov_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_3[_pair]) << 16);
                    (&_unpack_bf16_0_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_3[_pair]) & 0xffff0000u);
                }
                residual_1[0] = ov_1[0] - _unpack_bf16_0_1[0];
                residual_1[1] = ov_1[1] - _unpack_bf16_0_1[1];
                residual_1[2] = ov_1[2] - _unpack_bf16_0_1[2];
                residual_1[3] = ov_1[3] - _unpack_bf16_0_1[3];
                residual_1[4] = ov_1[4] - _unpack_bf16_0_1[4];
                residual_1[5] = ov_1[5] - _unpack_bf16_0_1[5];
                residual_1[6] = ov_1[6] - _unpack_bf16_0_1[6];
                residual_1[7] = ov_1[7] - _unpack_bf16_0_1[7];
                int byte_offset_3 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128;
                int swizzled_offset_2 = byte_offset_3 ^ (byte_offset_3 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_2), "r"(ov_bf16_3[0]), "r"(ov_bf16_3[1]), "r"(ov_bf16_3[2]), "r"(ov_bf16_3[3]) : "memory");
                unsigned int ov_bf16_1_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 8], ov_1[_lp*2+1 + 8]));
                    ov_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_1[_pair]) << 16);
                    (&_unpack_bf16_1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_1[_pair]) & 0xffff0000u);
                }
                residual_1[8] = ov_1[8] - _unpack_bf16_1_1[0];
                residual_1[9] = ov_1[9] - _unpack_bf16_1_1[1];
                residual_1[10] = ov_1[10] - _unpack_bf16_1_1[2];
                residual_1[11] = ov_1[11] - _unpack_bf16_1_1[3];
                residual_1[12] = ov_1[12] - _unpack_bf16_1_1[4];
                residual_1[13] = ov_1[13] - _unpack_bf16_1_1[5];
                residual_1[14] = ov_1[14] - _unpack_bf16_1_1[6];
                residual_1[15] = ov_1[15] - _unpack_bf16_1_1[7];
                int byte_offset_2_1 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3_1 = byte_offset_2_1 ^ (byte_offset_2_1 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3_1), "r"(ov_bf16_1_1[0]), "r"(ov_bf16_1_1[1]), "r"(ov_bf16_1_1[2]), "r"(ov_bf16_1_1[3]) : "memory");
                unsigned int _packed_0_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 0], residual_1[_lp*2+1 + 0]));
                    _packed_0_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 2], residual_1[_lp*2+1 + 2]));
                    _packed_0_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 4], residual_1[_lp*2+1 + 4]));
                    _packed_0_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 6], residual_1[_lp*2+1 + 6]));
                    _packed_0_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 8], residual_1[_lp*2+1 + 8]));
                    _packed_0_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 10], residual_1[_lp*2+1 + 10]));
                    _packed_0_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 12], residual_1[_lp*2+1 + 12]));
                    _packed_0_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 14], residual_1[_lp*2+1 + 14]));
                    _packed_0_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_39 = reinterpret_cast<const unsigned*>(_packed_0_1);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1))), "r"(_raw_stv8_39[0]), "r"(_raw_stv8_39[1]), "r"(_raw_stv8_39[2]), "r"(_raw_stv8_39[3]), "r"(_raw_stv8_39[4]), "r"(_raw_stv8_39[5]), "r"(_raw_stv8_39[6]), "r"(_raw_stv8_39[7]) : "memory");
                }
                float residual_r_3[16];
                unsigned int ov_bf16_2_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 16], ov_1[_lp*2+1 + 16]));
                    ov_bf16_2_6[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_6[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_6[_pair]) & 0xffff0000u);
                }
                residual_r_3[0] = ov_1[16] - _unpack_bf16_8[0];
                residual_r_3[1] = ov_1[17] - _unpack_bf16_8[1];
                residual_r_3[2] = ov_1[18] - _unpack_bf16_8[2];
                residual_r_3[3] = ov_1[19] - _unpack_bf16_8[3];
                residual_r_3[4] = ov_1[20] - _unpack_bf16_8[4];
                residual_r_3[5] = ov_1[21] - _unpack_bf16_8[5];
                residual_r_3[6] = ov_1[22] - _unpack_bf16_8[6];
                residual_r_3[7] = ov_1[23] - _unpack_bf16_8[7];
                int byte_offset_1_6 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_1_6 = byte_offset_1_6 ^ (byte_offset_1_6 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_6), "r"(ov_bf16_2_6[0]), "r"(ov_bf16_2_6[1]), "r"(ov_bf16_2_6[2]), "r"(ov_bf16_2_6[3]) : "memory");
                unsigned int ov_bf16_2_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 24], ov_1[_lp*2+1 + 24]));
                    ov_bf16_2_7[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_7[_pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_7[_pair]) & 0xffff0000u);
                }
                residual_r_3[8] = ov_1[24] - _unpack_bf16_9[0];
                residual_r_3[9] = ov_1[25] - _unpack_bf16_9[1];
                residual_r_3[10] = ov_1[26] - _unpack_bf16_9[2];
                residual_r_3[11] = ov_1[27] - _unpack_bf16_9[3];
                residual_r_3[12] = ov_1[28] - _unpack_bf16_9[4];
                residual_r_3[13] = ov_1[29] - _unpack_bf16_9[5];
                residual_r_3[14] = ov_1[30] - _unpack_bf16_9[6];
                residual_r_3[15] = ov_1[31] - _unpack_bf16_9[7];
                int byte_offset_1_7 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_1_7 = byte_offset_1_7 ^ (byte_offset_1_7 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_7), "r"(ov_bf16_2_7[0]), "r"(ov_bf16_2_7[1]), "r"(ov_bf16_2_7[2]), "r"(ov_bf16_2_7[3]) : "memory");
                unsigned int packed_r_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 0], residual_r_3[_lp*2+1 + 0]));
                    packed_r_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 2], residual_r_3[_lp*2+1 + 2]));
                    packed_r_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 4], residual_r_3[_lp*2+1 + 4]));
                    packed_r_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 6], residual_r_3[_lp*2+1 + 6]));
                    packed_r_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 8], residual_r_3[_lp*2+1 + 8]));
                    packed_r_3[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 10], residual_r_3[_lp*2+1 + 10]));
                    packed_r_3[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 12], residual_r_3[_lp*2+1 + 12]));
                    packed_r_3[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_3[_lp*2 + 14], residual_r_3[_lp*2+1 + 14]));
                    packed_r_3[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_40 = reinterpret_cast<const unsigned*>(packed_r_3);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 16ll))), "r"(_raw_stv8_40[0]), "r"(_raw_stv8_40[1]), "r"(_raw_stv8_40[2]), "r"(_raw_stv8_40[3]), "r"(_raw_stv8_40[4]), "r"(_raw_stv8_40[5]), "r"(_raw_stv8_40[6]), "r"(_raw_stv8_40[7]) : "memory");
                }
                float residual_r_4[16];
                unsigned int ov_bf16_2_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 32], ov_1[_lp*2+1 + 32]));
                    ov_bf16_2_8[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_8[_pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_8[_pair]) & 0xffff0000u);
                }
                residual_r_4[0] = ov_1[32] - _unpack_bf16_10[0];
                residual_r_4[1] = ov_1[33] - _unpack_bf16_10[1];
                residual_r_4[2] = ov_1[34] - _unpack_bf16_10[2];
                residual_r_4[3] = ov_1[35] - _unpack_bf16_10[3];
                residual_r_4[4] = ov_1[36] - _unpack_bf16_10[4];
                residual_r_4[5] = ov_1[37] - _unpack_bf16_10[5];
                residual_r_4[6] = ov_1[38] - _unpack_bf16_10[6];
                residual_r_4[7] = ov_1[39] - _unpack_bf16_10[7];
                int byte_offset_1_8 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_1_8 = byte_offset_1_8 ^ (byte_offset_1_8 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_8), "r"(ov_bf16_2_8[0]), "r"(ov_bf16_2_8[1]), "r"(ov_bf16_2_8[2]), "r"(ov_bf16_2_8[3]) : "memory");
                unsigned int ov_bf16_2_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 40], ov_1[_lp*2+1 + 40]));
                    ov_bf16_2_9[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_9[_pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_9[_pair]) & 0xffff0000u);
                }
                residual_r_4[8] = ov_1[40] - _unpack_bf16_11[0];
                residual_r_4[9] = ov_1[41] - _unpack_bf16_11[1];
                residual_r_4[10] = ov_1[42] - _unpack_bf16_11[2];
                residual_r_4[11] = ov_1[43] - _unpack_bf16_11[3];
                residual_r_4[12] = ov_1[44] - _unpack_bf16_11[4];
                residual_r_4[13] = ov_1[45] - _unpack_bf16_11[5];
                residual_r_4[14] = ov_1[46] - _unpack_bf16_11[6];
                residual_r_4[15] = ov_1[47] - _unpack_bf16_11[7];
                int byte_offset_1_9 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_1_9 = byte_offset_1_9 ^ (byte_offset_1_9 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_9), "r"(ov_bf16_2_9[0]), "r"(ov_bf16_2_9[1]), "r"(ov_bf16_2_9[2]), "r"(ov_bf16_2_9[3]) : "memory");
                unsigned int packed_r_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 0], residual_r_4[_lp*2+1 + 0]));
                    packed_r_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 2], residual_r_4[_lp*2+1 + 2]));
                    packed_r_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 4], residual_r_4[_lp*2+1 + 4]));
                    packed_r_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 6], residual_r_4[_lp*2+1 + 6]));
                    packed_r_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 8], residual_r_4[_lp*2+1 + 8]));
                    packed_r_4[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 10], residual_r_4[_lp*2+1 + 10]));
                    packed_r_4[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 12], residual_r_4[_lp*2+1 + 12]));
                    packed_r_4[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_4[_lp*2 + 14], residual_r_4[_lp*2+1 + 14]));
                    packed_r_4[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_41 = reinterpret_cast<const unsigned*>(packed_r_4);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 32ll))), "r"(_raw_stv8_41[0]), "r"(_raw_stv8_41[1]), "r"(_raw_stv8_41[2]), "r"(_raw_stv8_41[3]), "r"(_raw_stv8_41[4]), "r"(_raw_stv8_41[5]), "r"(_raw_stv8_41[6]), "r"(_raw_stv8_41[7]) : "memory");
                }
                float residual_r_5[16];
                unsigned int ov_bf16_2_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 48], ov_1[_lp*2+1 + 48]));
                    ov_bf16_2_10[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_10[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_10[_pair]) & 0xffff0000u);
                }
                residual_r_5[0] = ov_1[48] - _unpack_bf16_12[0];
                residual_r_5[1] = ov_1[49] - _unpack_bf16_12[1];
                residual_r_5[2] = ov_1[50] - _unpack_bf16_12[2];
                residual_r_5[3] = ov_1[51] - _unpack_bf16_12[3];
                residual_r_5[4] = ov_1[52] - _unpack_bf16_12[4];
                residual_r_5[5] = ov_1[53] - _unpack_bf16_12[5];
                residual_r_5[6] = ov_1[54] - _unpack_bf16_12[6];
                residual_r_5[7] = ov_1[55] - _unpack_bf16_12[7];
                int byte_offset_1_10 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_1_10 = byte_offset_1_10 ^ (byte_offset_1_10 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_10), "r"(ov_bf16_2_10[0]), "r"(ov_bf16_2_10[1]), "r"(ov_bf16_2_10[2]), "r"(ov_bf16_2_10[3]) : "memory");
                unsigned int ov_bf16_2_11[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 56], ov_1[_lp*2+1 + 56]));
                    ov_bf16_2_11[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_11[_pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_11[_pair]) & 0xffff0000u);
                }
                residual_r_5[8] = ov_1[56] - _unpack_bf16_13[0];
                residual_r_5[9] = ov_1[57] - _unpack_bf16_13[1];
                residual_r_5[10] = ov_1[58] - _unpack_bf16_13[2];
                residual_r_5[11] = ov_1[59] - _unpack_bf16_13[3];
                residual_r_5[12] = ov_1[60] - _unpack_bf16_13[4];
                residual_r_5[13] = ov_1[61] - _unpack_bf16_13[5];
                residual_r_5[14] = ov_1[62] - _unpack_bf16_13[6];
                residual_r_5[15] = ov_1[63] - _unpack_bf16_13[7];
                int byte_offset_1_11 = o_slot * 65536 + out_col_1 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_1_11 = byte_offset_1_11 ^ (byte_offset_1_11 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_11), "r"(ov_bf16_2_11[0]), "r"(ov_bf16_2_11[1]), "r"(ov_bf16_2_11[2]), "r"(ov_bf16_2_11[3]) : "memory");
                unsigned int packed_r_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 0], residual_r_5[_lp*2+1 + 0]));
                    packed_r_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 2], residual_r_5[_lp*2+1 + 2]));
                    packed_r_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 4], residual_r_5[_lp*2+1 + 4]));
                    packed_r_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 6], residual_r_5[_lp*2+1 + 6]));
                    packed_r_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 8], residual_r_5[_lp*2+1 + 8]));
                    packed_r_5[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 10], residual_r_5[_lp*2+1 + 10]));
                    packed_r_5[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 12], residual_r_5[_lp*2+1 + 12]));
                    packed_r_5[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_5[_lp*2 + 14], residual_r_5[_lp*2+1 + 14]));
                    packed_r_5[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_42 = reinterpret_cast<const unsigned*>(packed_r_5);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_1 + 48ll))), "r"(_raw_stv8_42[0]), "r"(_raw_stv8_42[1]), "r"(_raw_stv8_42[2]), "r"(_raw_stv8_42[3]), "r"(_raw_stv8_42[4]), "r"(_raw_stv8_42[5]), "r"(_raw_stv8_42[6]), "r"(_raw_stv8_42[7]) : "memory");
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
                float ov_2[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_2[0]), "=f"(ov_2[1]), "=f"(ov_2[2]), "=f"(ov_2[3]), "=f"(ov_2[4]), "=f"(ov_2[5]), "=f"(ov_2[6]), "=f"(ov_2[7]), "=f"(ov_2[8]), "=f"(ov_2[9]), "=f"(ov_2[10]), "=f"(ov_2[11]), "=f"(ov_2[12]), "=f"(ov_2[13]), "=f"(ov_2[14]), "=f"(ov_2[15]), "=f"(ov_2[16]), "=f"(ov_2[17]), "=f"(ov_2[18]), "=f"(ov_2[19]), "=f"(ov_2[20]), "=f"(ov_2[21]), "=f"(ov_2[22]), "=f"(ov_2[23]), "=f"(ov_2[24]), "=f"(ov_2[25]), "=f"(ov_2[26]), "=f"(ov_2[27]), "=f"(ov_2[28]), "=f"(ov_2[29]), "=f"(ov_2[30]), "=f"(ov_2[31]), "=f"(ov_2[32]), "=f"(ov_2[33]), "=f"(ov_2[34]), "=f"(ov_2[35]), "=f"(ov_2[36]), "=f"(ov_2[37]), "=f"(ov_2[38]), "=f"(ov_2[39]), "=f"(ov_2[40]), "=f"(ov_2[41]), "=f"(ov_2[42]), "=f"(ov_2[43]), "=f"(ov_2[44]), "=f"(ov_2[45]), "=f"(ov_2[46]), "=f"(ov_2[47]), "=f"(ov_2[48]), "=f"(ov_2[49]), "=f"(ov_2[50]), "=f"(ov_2[51]), "=f"(ov_2[52]), "=f"(ov_2[53]), "=f"(ov_2[54]), "=f"(ov_2[55]), "=f"(ov_2[56]), "=f"(ov_2[57]), "=f"(ov_2[58]), "=f"(ov_2[59]), "=f"(ov_2[60]), "=f"(ov_2[61]), "=f"(ov_2[62]), "=f"(ov_2[63])
                        : "r"(tmrow + 128));
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
                const float2 _scale2_43 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_43);
                int out_col_2 = 256 + row / 64 * 128;
                float residual_2[16];
                unsigned int ov_bf16_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 0], ov_2[_lp*2+1 + 0]));
                    ov_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_4[_pair]) << 16);
                    (&_unpack_bf16_0_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_4[_pair]) & 0xffff0000u);
                }
                residual_2[0] = ov_2[0] - _unpack_bf16_0_2[0];
                residual_2[1] = ov_2[1] - _unpack_bf16_0_2[1];
                residual_2[2] = ov_2[2] - _unpack_bf16_0_2[2];
                residual_2[3] = ov_2[3] - _unpack_bf16_0_2[3];
                residual_2[4] = ov_2[4] - _unpack_bf16_0_2[4];
                residual_2[5] = ov_2[5] - _unpack_bf16_0_2[5];
                residual_2[6] = ov_2[6] - _unpack_bf16_0_2[6];
                residual_2[7] = ov_2[7] - _unpack_bf16_0_2[7];
                int byte_offset_4 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128;
                int swizzled_offset_4 = byte_offset_4 ^ (byte_offset_4 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_4), "r"(ov_bf16_4[0]), "r"(ov_bf16_4[1]), "r"(ov_bf16_4[2]), "r"(ov_bf16_4[3]) : "memory");
                unsigned int ov_bf16_1_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 8], ov_2[_lp*2+1 + 8]));
                    ov_bf16_1_2[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_2[_pair]) << 16);
                    (&_unpack_bf16_1_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_2[_pair]) & 0xffff0000u);
                }
                residual_2[8] = ov_2[8] - _unpack_bf16_1_2[0];
                residual_2[9] = ov_2[9] - _unpack_bf16_1_2[1];
                residual_2[10] = ov_2[10] - _unpack_bf16_1_2[2];
                residual_2[11] = ov_2[11] - _unpack_bf16_1_2[3];
                residual_2[12] = ov_2[12] - _unpack_bf16_1_2[4];
                residual_2[13] = ov_2[13] - _unpack_bf16_1_2[5];
                residual_2[14] = ov_2[14] - _unpack_bf16_1_2[6];
                residual_2[15] = ov_2[15] - _unpack_bf16_1_2[7];
                int byte_offset_2_2 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3_2 = byte_offset_2_2 ^ (byte_offset_2_2 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3_2), "r"(ov_bf16_1_2[0]), "r"(ov_bf16_1_2[1]), "r"(ov_bf16_1_2[2]), "r"(ov_bf16_1_2[3]) : "memory");
                unsigned int _packed_0_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 0], residual_2[_lp*2+1 + 0]));
                    _packed_0_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 2], residual_2[_lp*2+1 + 2]));
                    _packed_0_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 4], residual_2[_lp*2+1 + 4]));
                    _packed_0_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 6], residual_2[_lp*2+1 + 6]));
                    _packed_0_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 8], residual_2[_lp*2+1 + 8]));
                    _packed_0_2[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 10], residual_2[_lp*2+1 + 10]));
                    _packed_0_2[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 12], residual_2[_lp*2+1 + 12]));
                    _packed_0_2[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_2[_lp*2 + 14], residual_2[_lp*2+1 + 14]));
                    _packed_0_2[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_44 = reinterpret_cast<const unsigned*>(_packed_0_2);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2))), "r"(_raw_stv8_44[0]), "r"(_raw_stv8_44[1]), "r"(_raw_stv8_44[2]), "r"(_raw_stv8_44[3]), "r"(_raw_stv8_44[4]), "r"(_raw_stv8_44[5]), "r"(_raw_stv8_44[6]), "r"(_raw_stv8_44[7]) : "memory");
                }
                float residual_r_6[16];
                unsigned int ov_bf16_2_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 16], ov_2[_lp*2+1 + 16]));
                    ov_bf16_2_12[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_12[_pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_12[_pair]) & 0xffff0000u);
                }
                residual_r_6[0] = ov_2[16] - _unpack_bf16_14[0];
                residual_r_6[1] = ov_2[17] - _unpack_bf16_14[1];
                residual_r_6[2] = ov_2[18] - _unpack_bf16_14[2];
                residual_r_6[3] = ov_2[19] - _unpack_bf16_14[3];
                residual_r_6[4] = ov_2[20] - _unpack_bf16_14[4];
                residual_r_6[5] = ov_2[21] - _unpack_bf16_14[5];
                residual_r_6[6] = ov_2[22] - _unpack_bf16_14[6];
                residual_r_6[7] = ov_2[23] - _unpack_bf16_14[7];
                int byte_offset_1_12 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_1_12 = byte_offset_1_12 ^ (byte_offset_1_12 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_12), "r"(ov_bf16_2_12[0]), "r"(ov_bf16_2_12[1]), "r"(ov_bf16_2_12[2]), "r"(ov_bf16_2_12[3]) : "memory");
                unsigned int ov_bf16_2_13[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 24], ov_2[_lp*2+1 + 24]));
                    ov_bf16_2_13[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_13[_pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_13[_pair]) & 0xffff0000u);
                }
                residual_r_6[8] = ov_2[24] - _unpack_bf16_15[0];
                residual_r_6[9] = ov_2[25] - _unpack_bf16_15[1];
                residual_r_6[10] = ov_2[26] - _unpack_bf16_15[2];
                residual_r_6[11] = ov_2[27] - _unpack_bf16_15[3];
                residual_r_6[12] = ov_2[28] - _unpack_bf16_15[4];
                residual_r_6[13] = ov_2[29] - _unpack_bf16_15[5];
                residual_r_6[14] = ov_2[30] - _unpack_bf16_15[6];
                residual_r_6[15] = ov_2[31] - _unpack_bf16_15[7];
                int byte_offset_1_13 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_1_13 = byte_offset_1_13 ^ (byte_offset_1_13 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_13), "r"(ov_bf16_2_13[0]), "r"(ov_bf16_2_13[1]), "r"(ov_bf16_2_13[2]), "r"(ov_bf16_2_13[3]) : "memory");
                unsigned int packed_r_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 0], residual_r_6[_lp*2+1 + 0]));
                    packed_r_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 2], residual_r_6[_lp*2+1 + 2]));
                    packed_r_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 4], residual_r_6[_lp*2+1 + 4]));
                    packed_r_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 6], residual_r_6[_lp*2+1 + 6]));
                    packed_r_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 8], residual_r_6[_lp*2+1 + 8]));
                    packed_r_6[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 10], residual_r_6[_lp*2+1 + 10]));
                    packed_r_6[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 12], residual_r_6[_lp*2+1 + 12]));
                    packed_r_6[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_6[_lp*2 + 14], residual_r_6[_lp*2+1 + 14]));
                    packed_r_6[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_45 = reinterpret_cast<const unsigned*>(packed_r_6);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 16ll))), "r"(_raw_stv8_45[0]), "r"(_raw_stv8_45[1]), "r"(_raw_stv8_45[2]), "r"(_raw_stv8_45[3]), "r"(_raw_stv8_45[4]), "r"(_raw_stv8_45[5]), "r"(_raw_stv8_45[6]), "r"(_raw_stv8_45[7]) : "memory");
                }
                float residual_r_7[16];
                unsigned int ov_bf16_2_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 32], ov_2[_lp*2+1 + 32]));
                    ov_bf16_2_14[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_14[_pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_14[_pair]) & 0xffff0000u);
                }
                residual_r_7[0] = ov_2[32] - _unpack_bf16_16[0];
                residual_r_7[1] = ov_2[33] - _unpack_bf16_16[1];
                residual_r_7[2] = ov_2[34] - _unpack_bf16_16[2];
                residual_r_7[3] = ov_2[35] - _unpack_bf16_16[3];
                residual_r_7[4] = ov_2[36] - _unpack_bf16_16[4];
                residual_r_7[5] = ov_2[37] - _unpack_bf16_16[5];
                residual_r_7[6] = ov_2[38] - _unpack_bf16_16[6];
                residual_r_7[7] = ov_2[39] - _unpack_bf16_16[7];
                int byte_offset_1_14 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_1_14 = byte_offset_1_14 ^ (byte_offset_1_14 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_14), "r"(ov_bf16_2_14[0]), "r"(ov_bf16_2_14[1]), "r"(ov_bf16_2_14[2]), "r"(ov_bf16_2_14[3]) : "memory");
                unsigned int ov_bf16_2_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 40], ov_2[_lp*2+1 + 40]));
                    ov_bf16_2_15[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_15[_pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_15[_pair]) & 0xffff0000u);
                }
                residual_r_7[8] = ov_2[40] - _unpack_bf16_17[0];
                residual_r_7[9] = ov_2[41] - _unpack_bf16_17[1];
                residual_r_7[10] = ov_2[42] - _unpack_bf16_17[2];
                residual_r_7[11] = ov_2[43] - _unpack_bf16_17[3];
                residual_r_7[12] = ov_2[44] - _unpack_bf16_17[4];
                residual_r_7[13] = ov_2[45] - _unpack_bf16_17[5];
                residual_r_7[14] = ov_2[46] - _unpack_bf16_17[6];
                residual_r_7[15] = ov_2[47] - _unpack_bf16_17[7];
                int byte_offset_1_15 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_1_15 = byte_offset_1_15 ^ (byte_offset_1_15 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_15), "r"(ov_bf16_2_15[0]), "r"(ov_bf16_2_15[1]), "r"(ov_bf16_2_15[2]), "r"(ov_bf16_2_15[3]) : "memory");
                unsigned int packed_r_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 0], residual_r_7[_lp*2+1 + 0]));
                    packed_r_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 2], residual_r_7[_lp*2+1 + 2]));
                    packed_r_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 4], residual_r_7[_lp*2+1 + 4]));
                    packed_r_7[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 6], residual_r_7[_lp*2+1 + 6]));
                    packed_r_7[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 8], residual_r_7[_lp*2+1 + 8]));
                    packed_r_7[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 10], residual_r_7[_lp*2+1 + 10]));
                    packed_r_7[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 12], residual_r_7[_lp*2+1 + 12]));
                    packed_r_7[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_7[_lp*2 + 14], residual_r_7[_lp*2+1 + 14]));
                    packed_r_7[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_46 = reinterpret_cast<const unsigned*>(packed_r_7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 32ll))), "r"(_raw_stv8_46[0]), "r"(_raw_stv8_46[1]), "r"(_raw_stv8_46[2]), "r"(_raw_stv8_46[3]), "r"(_raw_stv8_46[4]), "r"(_raw_stv8_46[5]), "r"(_raw_stv8_46[6]), "r"(_raw_stv8_46[7]) : "memory");
                }
                float residual_r_8[16];
                unsigned int ov_bf16_2_16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 48], ov_2[_lp*2+1 + 48]));
                    ov_bf16_2_16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_16[_pair]) << 16);
                    (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_16[_pair]) & 0xffff0000u);
                }
                residual_r_8[0] = ov_2[48] - _unpack_bf16_18[0];
                residual_r_8[1] = ov_2[49] - _unpack_bf16_18[1];
                residual_r_8[2] = ov_2[50] - _unpack_bf16_18[2];
                residual_r_8[3] = ov_2[51] - _unpack_bf16_18[3];
                residual_r_8[4] = ov_2[52] - _unpack_bf16_18[4];
                residual_r_8[5] = ov_2[53] - _unpack_bf16_18[5];
                residual_r_8[6] = ov_2[54] - _unpack_bf16_18[6];
                residual_r_8[7] = ov_2[55] - _unpack_bf16_18[7];
                int byte_offset_1_16 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_1_16 = byte_offset_1_16 ^ (byte_offset_1_16 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_16), "r"(ov_bf16_2_16[0]), "r"(ov_bf16_2_16[1]), "r"(ov_bf16_2_16[2]), "r"(ov_bf16_2_16[3]) : "memory");
                unsigned int ov_bf16_2_17[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 56], ov_2[_lp*2+1 + 56]));
                    ov_bf16_2_17[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_17[_pair]) << 16);
                    (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_17[_pair]) & 0xffff0000u);
                }
                residual_r_8[8] = ov_2[56] - _unpack_bf16_19[0];
                residual_r_8[9] = ov_2[57] - _unpack_bf16_19[1];
                residual_r_8[10] = ov_2[58] - _unpack_bf16_19[2];
                residual_r_8[11] = ov_2[59] - _unpack_bf16_19[3];
                residual_r_8[12] = ov_2[60] - _unpack_bf16_19[4];
                residual_r_8[13] = ov_2[61] - _unpack_bf16_19[5];
                residual_r_8[14] = ov_2[62] - _unpack_bf16_19[6];
                residual_r_8[15] = ov_2[63] - _unpack_bf16_19[7];
                int byte_offset_1_17 = o_slot * 65536 + out_col_2 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_1_17 = byte_offset_1_17 ^ (byte_offset_1_17 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_17), "r"(ov_bf16_2_17[0]), "r"(ov_bf16_2_17[1]), "r"(ov_bf16_2_17[2]), "r"(ov_bf16_2_17[3]) : "memory");
                unsigned int packed_r_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 0], residual_r_8[_lp*2+1 + 0]));
                    packed_r_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 2], residual_r_8[_lp*2+1 + 2]));
                    packed_r_8[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 4], residual_r_8[_lp*2+1 + 4]));
                    packed_r_8[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 6], residual_r_8[_lp*2+1 + 6]));
                    packed_r_8[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 8], residual_r_8[_lp*2+1 + 8]));
                    packed_r_8[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 10], residual_r_8[_lp*2+1 + 10]));
                    packed_r_8[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 12], residual_r_8[_lp*2+1 + 12]));
                    packed_r_8[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_8[_lp*2 + 14], residual_r_8[_lp*2+1 + 14]));
                    packed_r_8[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_47 = reinterpret_cast<const unsigned*>(packed_r_8);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_2 + 48ll))), "r"(_raw_stv8_47[0]), "r"(_raw_stv8_47[1]), "r"(_raw_stv8_47[2]), "r"(_raw_stv8_47[3]), "r"(_raw_stv8_47[4]), "r"(_raw_stv8_47[5]), "r"(_raw_stv8_47[6]), "r"(_raw_stv8_47[7]) : "memory");
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
                float ov_3[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_3[0]), "=f"(ov_3[1]), "=f"(ov_3[2]), "=f"(ov_3[3]), "=f"(ov_3[4]), "=f"(ov_3[5]), "=f"(ov_3[6]), "=f"(ov_3[7]), "=f"(ov_3[8]), "=f"(ov_3[9]), "=f"(ov_3[10]), "=f"(ov_3[11]), "=f"(ov_3[12]), "=f"(ov_3[13]), "=f"(ov_3[14]), "=f"(ov_3[15]), "=f"(ov_3[16]), "=f"(ov_3[17]), "=f"(ov_3[18]), "=f"(ov_3[19]), "=f"(ov_3[20]), "=f"(ov_3[21]), "=f"(ov_3[22]), "=f"(ov_3[23]), "=f"(ov_3[24]), "=f"(ov_3[25]), "=f"(ov_3[26]), "=f"(ov_3[27]), "=f"(ov_3[28]), "=f"(ov_3[29]), "=f"(ov_3[30]), "=f"(ov_3[31]), "=f"(ov_3[32]), "=f"(ov_3[33]), "=f"(ov_3[34]), "=f"(ov_3[35]), "=f"(ov_3[36]), "=f"(ov_3[37]), "=f"(ov_3[38]), "=f"(ov_3[39]), "=f"(ov_3[40]), "=f"(ov_3[41]), "=f"(ov_3[42]), "=f"(ov_3[43]), "=f"(ov_3[44]), "=f"(ov_3[45]), "=f"(ov_3[46]), "=f"(ov_3[47]), "=f"(ov_3[48]), "=f"(ov_3[49]), "=f"(ov_3[50]), "=f"(ov_3[51]), "=f"(ov_3[52]), "=f"(ov_3[53]), "=f"(ov_3[54]), "=f"(ov_3[55]), "=f"(ov_3[56]), "=f"(ov_3[57]), "=f"(ov_3[58]), "=f"(ov_3[59]), "=f"(ov_3[60]), "=f"(ov_3[61]), "=f"(ov_3[62]), "=f"(ov_3[63])
                        : "r"(tmrow + 192));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_3[0] = 0.0f;
                    ov_3[1] = 0.0f;
                    ov_3[2] = 0.0f;
                    ov_3[3] = 0.0f;
                    ov_3[4] = 0.0f;
                    ov_3[5] = 0.0f;
                    ov_3[6] = 0.0f;
                    ov_3[7] = 0.0f;
                    ov_3[8] = 0.0f;
                    ov_3[9] = 0.0f;
                    ov_3[10] = 0.0f;
                    ov_3[11] = 0.0f;
                    ov_3[12] = 0.0f;
                    ov_3[13] = 0.0f;
                    ov_3[14] = 0.0f;
                    ov_3[15] = 0.0f;
                    ov_3[16] = 0.0f;
                    ov_3[17] = 0.0f;
                    ov_3[18] = 0.0f;
                    ov_3[19] = 0.0f;
                    ov_3[20] = 0.0f;
                    ov_3[21] = 0.0f;
                    ov_3[22] = 0.0f;
                    ov_3[23] = 0.0f;
                    ov_3[24] = 0.0f;
                    ov_3[25] = 0.0f;
                    ov_3[26] = 0.0f;
                    ov_3[27] = 0.0f;
                    ov_3[28] = 0.0f;
                    ov_3[29] = 0.0f;
                    ov_3[30] = 0.0f;
                    ov_3[31] = 0.0f;
                    ov_3[32] = 0.0f;
                    ov_3[33] = 0.0f;
                    ov_3[34] = 0.0f;
                    ov_3[35] = 0.0f;
                    ov_3[36] = 0.0f;
                    ov_3[37] = 0.0f;
                    ov_3[38] = 0.0f;
                    ov_3[39] = 0.0f;
                    ov_3[40] = 0.0f;
                    ov_3[41] = 0.0f;
                    ov_3[42] = 0.0f;
                    ov_3[43] = 0.0f;
                    ov_3[44] = 0.0f;
                    ov_3[45] = 0.0f;
                    ov_3[46] = 0.0f;
                    ov_3[47] = 0.0f;
                    ov_3[48] = 0.0f;
                    ov_3[49] = 0.0f;
                    ov_3[50] = 0.0f;
                    ov_3[51] = 0.0f;
                    ov_3[52] = 0.0f;
                    ov_3[53] = 0.0f;
                    ov_3[54] = 0.0f;
                    ov_3[55] = 0.0f;
                    ov_3[56] = 0.0f;
                    ov_3[57] = 0.0f;
                    ov_3[58] = 0.0f;
                    ov_3[59] = 0.0f;
                    ov_3[60] = 0.0f;
                    ov_3[61] = 0.0f;
                    ov_3[62] = 0.0f;
                    ov_3[63] = 0.0f;
                }
                const float2 _scale2_48 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_3)[_ls], _scale2_48);
                int out_col_3 = 256 + row / 64 * 128 + 64;
                float residual_3[16];
                unsigned int ov_bf16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 0], ov_3[_lp*2+1 + 0]));
                    ov_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) << 16);
                    (&_unpack_bf16_0_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) & 0xffff0000u);
                }
                residual_3[0] = ov_3[0] - _unpack_bf16_0_3[0];
                residual_3[1] = ov_3[1] - _unpack_bf16_0_3[1];
                residual_3[2] = ov_3[2] - _unpack_bf16_0_3[2];
                residual_3[3] = ov_3[3] - _unpack_bf16_0_3[3];
                residual_3[4] = ov_3[4] - _unpack_bf16_0_3[4];
                residual_3[5] = ov_3[5] - _unpack_bf16_0_3[5];
                residual_3[6] = ov_3[6] - _unpack_bf16_0_3[6];
                residual_3[7] = ov_3[7] - _unpack_bf16_0_3[7];
                int byte_offset_5 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128;
                int swizzled_offset_5 = byte_offset_5 ^ (byte_offset_5 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_5), "r"(ov_bf16_5[0]), "r"(ov_bf16_5[1]), "r"(ov_bf16_5[2]), "r"(ov_bf16_5[3]) : "memory");
                unsigned int ov_bf16_1_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 8], ov_3[_lp*2+1 + 8]));
                    ov_bf16_1_3[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_3[_pair]) << 16);
                    (&_unpack_bf16_1_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1_3[_pair]) & 0xffff0000u);
                }
                residual_3[8] = ov_3[8] - _unpack_bf16_1_3[0];
                residual_3[9] = ov_3[9] - _unpack_bf16_1_3[1];
                residual_3[10] = ov_3[10] - _unpack_bf16_1_3[2];
                residual_3[11] = ov_3[11] - _unpack_bf16_1_3[3];
                residual_3[12] = ov_3[12] - _unpack_bf16_1_3[4];
                residual_3[13] = ov_3[13] - _unpack_bf16_1_3[5];
                residual_3[14] = ov_3[14] - _unpack_bf16_1_3[6];
                residual_3[15] = ov_3[15] - _unpack_bf16_1_3[7];
                int byte_offset_2_3 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3_3 = byte_offset_2_3 ^ (byte_offset_2_3 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3_3), "r"(ov_bf16_1_3[0]), "r"(ov_bf16_1_3[1]), "r"(ov_bf16_1_3[2]), "r"(ov_bf16_1_3[3]) : "memory");
                unsigned int _packed_0_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 0], residual_3[_lp*2+1 + 0]));
                    _packed_0_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 2], residual_3[_lp*2+1 + 2]));
                    _packed_0_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 4], residual_3[_lp*2+1 + 4]));
                    _packed_0_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 6], residual_3[_lp*2+1 + 6]));
                    _packed_0_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 8], residual_3[_lp*2+1 + 8]));
                    _packed_0_3[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 10], residual_3[_lp*2+1 + 10]));
                    _packed_0_3[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 12], residual_3[_lp*2+1 + 12]));
                    _packed_0_3[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_3[_lp*2 + 14], residual_3[_lp*2+1 + 14]));
                    _packed_0_3[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_49 = reinterpret_cast<const unsigned*>(_packed_0_3);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_3))), "r"(_raw_stv8_49[0]), "r"(_raw_stv8_49[1]), "r"(_raw_stv8_49[2]), "r"(_raw_stv8_49[3]), "r"(_raw_stv8_49[4]), "r"(_raw_stv8_49[5]), "r"(_raw_stv8_49[6]), "r"(_raw_stv8_49[7]) : "memory");
                }
                float residual_r_9[16];
                unsigned int ov_bf16_2_18[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 16], ov_3[_lp*2+1 + 16]));
                    ov_bf16_2_18[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_18[_pair]) << 16);
                    (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_18[_pair]) & 0xffff0000u);
                }
                residual_r_9[0] = ov_3[16] - _unpack_bf16_20[0];
                residual_r_9[1] = ov_3[17] - _unpack_bf16_20[1];
                residual_r_9[2] = ov_3[18] - _unpack_bf16_20[2];
                residual_r_9[3] = ov_3[19] - _unpack_bf16_20[3];
                residual_r_9[4] = ov_3[20] - _unpack_bf16_20[4];
                residual_r_9[5] = ov_3[21] - _unpack_bf16_20[5];
                residual_r_9[6] = ov_3[22] - _unpack_bf16_20[6];
                residual_r_9[7] = ov_3[23] - _unpack_bf16_20[7];
                int byte_offset_1_18 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_1_18 = byte_offset_1_18 ^ (byte_offset_1_18 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_18), "r"(ov_bf16_2_18[0]), "r"(ov_bf16_2_18[1]), "r"(ov_bf16_2_18[2]), "r"(ov_bf16_2_18[3]) : "memory");
                unsigned int ov_bf16_2_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 24], ov_3[_lp*2+1 + 24]));
                    ov_bf16_2_19[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_19[_pair]) << 16);
                    (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_19[_pair]) & 0xffff0000u);
                }
                residual_r_9[8] = ov_3[24] - _unpack_bf16_21[0];
                residual_r_9[9] = ov_3[25] - _unpack_bf16_21[1];
                residual_r_9[10] = ov_3[26] - _unpack_bf16_21[2];
                residual_r_9[11] = ov_3[27] - _unpack_bf16_21[3];
                residual_r_9[12] = ov_3[28] - _unpack_bf16_21[4];
                residual_r_9[13] = ov_3[29] - _unpack_bf16_21[5];
                residual_r_9[14] = ov_3[30] - _unpack_bf16_21[6];
                residual_r_9[15] = ov_3[31] - _unpack_bf16_21[7];
                int byte_offset_1_19 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_1_19 = byte_offset_1_19 ^ (byte_offset_1_19 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_19), "r"(ov_bf16_2_19[0]), "r"(ov_bf16_2_19[1]), "r"(ov_bf16_2_19[2]), "r"(ov_bf16_2_19[3]) : "memory");
                unsigned int packed_r_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 0], residual_r_9[_lp*2+1 + 0]));
                    packed_r_9[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 2], residual_r_9[_lp*2+1 + 2]));
                    packed_r_9[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 4], residual_r_9[_lp*2+1 + 4]));
                    packed_r_9[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 6], residual_r_9[_lp*2+1 + 6]));
                    packed_r_9[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 8], residual_r_9[_lp*2+1 + 8]));
                    packed_r_9[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 10], residual_r_9[_lp*2+1 + 10]));
                    packed_r_9[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 12], residual_r_9[_lp*2+1 + 12]));
                    packed_r_9[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_9[_lp*2 + 14], residual_r_9[_lp*2+1 + 14]));
                    packed_r_9[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_50 = reinterpret_cast<const unsigned*>(packed_r_9);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_3 + 16ll))), "r"(_raw_stv8_50[0]), "r"(_raw_stv8_50[1]), "r"(_raw_stv8_50[2]), "r"(_raw_stv8_50[3]), "r"(_raw_stv8_50[4]), "r"(_raw_stv8_50[5]), "r"(_raw_stv8_50[6]), "r"(_raw_stv8_50[7]) : "memory");
                }
                float residual_r_10[16];
                unsigned int ov_bf16_2_20[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 32], ov_3[_lp*2+1 + 32]));
                    ov_bf16_2_20[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_22[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_20[_pair]) << 16);
                    (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_20[_pair]) & 0xffff0000u);
                }
                residual_r_10[0] = ov_3[32] - _unpack_bf16_22[0];
                residual_r_10[1] = ov_3[33] - _unpack_bf16_22[1];
                residual_r_10[2] = ov_3[34] - _unpack_bf16_22[2];
                residual_r_10[3] = ov_3[35] - _unpack_bf16_22[3];
                residual_r_10[4] = ov_3[36] - _unpack_bf16_22[4];
                residual_r_10[5] = ov_3[37] - _unpack_bf16_22[5];
                residual_r_10[6] = ov_3[38] - _unpack_bf16_22[6];
                residual_r_10[7] = ov_3[39] - _unpack_bf16_22[7];
                int byte_offset_1_20 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_1_20 = byte_offset_1_20 ^ (byte_offset_1_20 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_20), "r"(ov_bf16_2_20[0]), "r"(ov_bf16_2_20[1]), "r"(ov_bf16_2_20[2]), "r"(ov_bf16_2_20[3]) : "memory");
                unsigned int ov_bf16_2_21[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 40], ov_3[_lp*2+1 + 40]));
                    ov_bf16_2_21[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_23[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_21[_pair]) << 16);
                    (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_21[_pair]) & 0xffff0000u);
                }
                residual_r_10[8] = ov_3[40] - _unpack_bf16_23[0];
                residual_r_10[9] = ov_3[41] - _unpack_bf16_23[1];
                residual_r_10[10] = ov_3[42] - _unpack_bf16_23[2];
                residual_r_10[11] = ov_3[43] - _unpack_bf16_23[3];
                residual_r_10[12] = ov_3[44] - _unpack_bf16_23[4];
                residual_r_10[13] = ov_3[45] - _unpack_bf16_23[5];
                residual_r_10[14] = ov_3[46] - _unpack_bf16_23[6];
                residual_r_10[15] = ov_3[47] - _unpack_bf16_23[7];
                int byte_offset_1_21 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_1_21 = byte_offset_1_21 ^ (byte_offset_1_21 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_21), "r"(ov_bf16_2_21[0]), "r"(ov_bf16_2_21[1]), "r"(ov_bf16_2_21[2]), "r"(ov_bf16_2_21[3]) : "memory");
                unsigned int packed_r_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 0], residual_r_10[_lp*2+1 + 0]));
                    packed_r_10[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 2], residual_r_10[_lp*2+1 + 2]));
                    packed_r_10[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 4], residual_r_10[_lp*2+1 + 4]));
                    packed_r_10[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 6], residual_r_10[_lp*2+1 + 6]));
                    packed_r_10[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 8], residual_r_10[_lp*2+1 + 8]));
                    packed_r_10[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 10], residual_r_10[_lp*2+1 + 10]));
                    packed_r_10[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 12], residual_r_10[_lp*2+1 + 12]));
                    packed_r_10[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_10[_lp*2 + 14], residual_r_10[_lp*2+1 + 14]));
                    packed_r_10[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_51 = reinterpret_cast<const unsigned*>(packed_r_10);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_3 + 32ll))), "r"(_raw_stv8_51[0]), "r"(_raw_stv8_51[1]), "r"(_raw_stv8_51[2]), "r"(_raw_stv8_51[3]), "r"(_raw_stv8_51[4]), "r"(_raw_stv8_51[5]), "r"(_raw_stv8_51[6]), "r"(_raw_stv8_51[7]) : "memory");
                }
                float residual_r_11[16];
                unsigned int ov_bf16_2_22[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 48], ov_3[_lp*2+1 + 48]));
                    ov_bf16_2_22[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_22[_pair]) << 16);
                    (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_22[_pair]) & 0xffff0000u);
                }
                residual_r_11[0] = ov_3[48] - _unpack_bf16_24[0];
                residual_r_11[1] = ov_3[49] - _unpack_bf16_24[1];
                residual_r_11[2] = ov_3[50] - _unpack_bf16_24[2];
                residual_r_11[3] = ov_3[51] - _unpack_bf16_24[3];
                residual_r_11[4] = ov_3[52] - _unpack_bf16_24[4];
                residual_r_11[5] = ov_3[53] - _unpack_bf16_24[5];
                residual_r_11[6] = ov_3[54] - _unpack_bf16_24[6];
                residual_r_11[7] = ov_3[55] - _unpack_bf16_24[7];
                int byte_offset_1_22 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_1_22 = byte_offset_1_22 ^ (byte_offset_1_22 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_22), "r"(ov_bf16_2_22[0]), "r"(ov_bf16_2_22[1]), "r"(ov_bf16_2_22[2]), "r"(ov_bf16_2_22[3]) : "memory");
                unsigned int ov_bf16_2_23[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 56], ov_3[_lp*2+1 + 56]));
                    ov_bf16_2_23[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_23[_pair]) << 16);
                    (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_2_23[_pair]) & 0xffff0000u);
                }
                residual_r_11[8] = ov_3[56] - _unpack_bf16_25[0];
                residual_r_11[9] = ov_3[57] - _unpack_bf16_25[1];
                residual_r_11[10] = ov_3[58] - _unpack_bf16_25[2];
                residual_r_11[11] = ov_3[59] - _unpack_bf16_25[3];
                residual_r_11[12] = ov_3[60] - _unpack_bf16_25[4];
                residual_r_11[13] = ov_3[61] - _unpack_bf16_25[5];
                residual_r_11[14] = ov_3[62] - _unpack_bf16_25[6];
                residual_r_11[15] = ov_3[63] - _unpack_bf16_25[7];
                int byte_offset_1_23 = o_slot * 65536 + out_col_3 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_1_23 = byte_offset_1_23 ^ (byte_offset_1_23 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_1_23), "r"(ov_bf16_2_23[0]), "r"(ov_bf16_2_23[1]), "r"(ov_bf16_2_23[2]), "r"(ov_bf16_2_23[3]) : "memory");
                unsigned int packed_r_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 0], residual_r_11[_lp*2+1 + 0]));
                    packed_r_11[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 2], residual_r_11[_lp*2+1 + 2]));
                    packed_r_11[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 4], residual_r_11[_lp*2+1 + 4]));
                    packed_r_11[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 6], residual_r_11[_lp*2+1 + 6]));
                    packed_r_11[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 8], residual_r_11[_lp*2+1 + 8]));
                    packed_r_11[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 10], residual_r_11[_lp*2+1 + 10]));
                    packed_r_11[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 12], residual_r_11[_lp*2+1 + 12]));
                    packed_r_11[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_r_11[_lp*2 + 14], residual_r_11[_lp*2+1 + 14]));
                    packed_r_11[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_52 = reinterpret_cast<const unsigned*>(packed_r_11);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_3 + 48ll))), "r"(_raw_stv8_52[0]), "r"(_raw_stv8_52[1]), "r"(_raw_stv8_52[2]), "r"(_raw_stv8_52[3]), "r"(_raw_stv8_52[4]), "r"(_raw_stv8_52[5]), "r"(_raw_stv8_52[6]), "r"(_raw_stv8_52[7]) : "memory");
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
            asm volatile("cp.async.bulk.wait_group 0;");
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
                long long row_out = (long long)query_1 * (long long)topk;
                int nxt[16];
                if (elect_sync()) {
                    if (convert != 0) {
                        int nb = 0;
                        int whole_out = active_0_1 >= nb * 64 + 64 && (topk & 3) == 0;
                        if (whole_out != 0) {
                            int off_3_0 = nb * 64 + producer_warp * 4;
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_0) + 0);
                                int4 _ivld_4;
                                _ivld_4 = *_ivptr_4;
                                _vec_load_0[0 + 0] = _ivld_4.x;
                                _vec_load_0[0 + 1] = _ivld_4.y;
                                _vec_load_0[0 + 2] = _ivld_4.z;
                                _vec_load_0[0 + 3] = _ivld_4.w;
                            }
                            nxt[0] = _vec_load_0[0];
                            nxt[1] = _vec_load_0[1];
                            nxt[2] = _vec_load_0[2];
                            nxt[3] = _vec_load_0[3];
                            int off_3_1 = nb * 64 + 16 + producer_warp * 4;
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_5 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_1) + 0);
                                int4 _ivld_5;
                                _ivld_5 = *_ivptr_5;
                                _vec_load_1[0 + 0] = _ivld_5.x;
                                _vec_load_1[0 + 1] = _ivld_5.y;
                                _vec_load_1[0 + 2] = _ivld_5.z;
                                _vec_load_1[0 + 3] = _ivld_5.w;
                            }
                            nxt[4] = _vec_load_1[0];
                            nxt[5] = _vec_load_1[1];
                            nxt[6] = _vec_load_1[2];
                            nxt[7] = _vec_load_1[3];
                            int off_3_2 = nb * 64 + 32 + producer_warp * 4;
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_6 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_2) + 0);
                                int4 _ivld_6;
                                _ivld_6 = *_ivptr_6;
                                _vec_load_2[0 + 0] = _ivld_6.x;
                                _vec_load_2[0 + 1] = _ivld_6.y;
                                _vec_load_2[0 + 2] = _ivld_6.z;
                                _vec_load_2[0 + 3] = _ivld_6.w;
                            }
                            nxt[8] = _vec_load_2[0];
                            nxt[9] = _vec_load_2[1];
                            nxt[10] = _vec_load_2[2];
                            nxt[11] = _vec_load_2[3];
                            int off_3_3 = nb * 64 + 48 + producer_warp * 4;
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_7 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_3) + 0);
                                int4 _ivld_7;
                                _ivld_7 = *_ivptr_7;
                                _vec_load_3[0 + 0] = _ivld_7.x;
                                _vec_load_3[0 + 1] = _ivld_7.y;
                                _vec_load_3[0 + 2] = _ivld_7.z;
                                _vec_load_3[0 + 3] = _ivld_7.w;
                            }
                            nxt[12] = _vec_load_3[0];
                            nxt[13] = _vec_load_3[1];
                            nxt[14] = _vec_load_3[2];
                            nxt[15] = _vec_load_3[3];
                        } else {
                            int off_3_4 = nb * 64 + producer_warp * 4;
                            int position_1_0 = off_3_4;
                            int clamped_3_0 = ((position_1_0) < (topk - 1) ? (position_1_0) : (topk - 1));
                            int value_4_0 = indices_out[row_out + (long long)clamped_3_0];
                            if (position_1_0 >= active_0_1) {
                                value_4_0 = -1;
                            }
                            nxt[0] = value_4_0;
                            int position_1_1 = off_3_4 + 1;
                            int clamped_3_1 = ((position_1_1) < (topk - 1) ? (position_1_1) : (topk - 1));
                            int value_4_1 = indices_out[row_out + (long long)clamped_3_1];
                            if (position_1_1 >= active_0_1) {
                                value_4_1 = -1;
                            }
                            nxt[1] = value_4_1;
                            int position_1_2 = off_3_4 + 2;
                            int clamped_3_2 = ((position_1_2) < (topk - 1) ? (position_1_2) : (topk - 1));
                            int value_4_2 = indices_out[row_out + (long long)clamped_3_2];
                            if (position_1_2 >= active_0_1) {
                                value_4_2 = -1;
                            }
                            nxt[2] = value_4_2;
                            int position_1_3 = off_3_4 + 3;
                            int clamped_3_3 = ((position_1_3) < (topk - 1) ? (position_1_3) : (topk - 1));
                            int value_4_3 = indices_out[row_out + (long long)clamped_3_3];
                            if (position_1_3 >= active_0_1) {
                                value_4_3 = -1;
                            }
                            nxt[3] = value_4_3;
                            int off_3_5 = nb * 64 + 16 + producer_warp * 4;
                            int position_1_4 = off_3_5;
                            int clamped_3_4 = ((position_1_4) < (topk - 1) ? (position_1_4) : (topk - 1));
                            int value_4_4 = indices_out[row_out + (long long)clamped_3_4];
                            if (position_1_4 >= active_0_1) {
                                value_4_4 = -1;
                            }
                            nxt[4] = value_4_4;
                            int position_1_5 = off_3_5 + 1;
                            int clamped_3_5 = ((position_1_5) < (topk - 1) ? (position_1_5) : (topk - 1));
                            int value_4_5 = indices_out[row_out + (long long)clamped_3_5];
                            if (position_1_5 >= active_0_1) {
                                value_4_5 = -1;
                            }
                            nxt[5] = value_4_5;
                            int position_1_6 = off_3_5 + 2;
                            int clamped_3_6 = ((position_1_6) < (topk - 1) ? (position_1_6) : (topk - 1));
                            int value_4_6 = indices_out[row_out + (long long)clamped_3_6];
                            if (position_1_6 >= active_0_1) {
                                value_4_6 = -1;
                            }
                            nxt[6] = value_4_6;
                            int position_1_7 = off_3_5 + 3;
                            int clamped_3_7 = ((position_1_7) < (topk - 1) ? (position_1_7) : (topk - 1));
                            int value_4_7 = indices_out[row_out + (long long)clamped_3_7];
                            if (position_1_7 >= active_0_1) {
                                value_4_7 = -1;
                            }
                            nxt[7] = value_4_7;
                            int off_3_6 = nb * 64 + 32 + producer_warp * 4;
                            int position_1_8 = off_3_6;
                            int clamped_3_8 = ((position_1_8) < (topk - 1) ? (position_1_8) : (topk - 1));
                            int value_4_8 = indices_out[row_out + (long long)clamped_3_8];
                            if (position_1_8 >= active_0_1) {
                                value_4_8 = -1;
                            }
                            nxt[8] = value_4_8;
                            int position_1_9 = off_3_6 + 1;
                            int clamped_3_9 = ((position_1_9) < (topk - 1) ? (position_1_9) : (topk - 1));
                            int value_4_9 = indices_out[row_out + (long long)clamped_3_9];
                            if (position_1_9 >= active_0_1) {
                                value_4_9 = -1;
                            }
                            nxt[9] = value_4_9;
                            int position_1_10 = off_3_6 + 2;
                            int clamped_3_10 = ((position_1_10) < (topk - 1) ? (position_1_10) : (topk - 1));
                            int value_4_10 = indices_out[row_out + (long long)clamped_3_10];
                            if (position_1_10 >= active_0_1) {
                                value_4_10 = -1;
                            }
                            nxt[10] = value_4_10;
                            int position_1_11 = off_3_6 + 3;
                            int clamped_3_11 = ((position_1_11) < (topk - 1) ? (position_1_11) : (topk - 1));
                            int value_4_11 = indices_out[row_out + (long long)clamped_3_11];
                            if (position_1_11 >= active_0_1) {
                                value_4_11 = -1;
                            }
                            nxt[11] = value_4_11;
                            int off_3_7 = nb * 64 + 48 + producer_warp * 4;
                            int position_1_12 = off_3_7;
                            int clamped_3_12 = ((position_1_12) < (topk - 1) ? (position_1_12) : (topk - 1));
                            int value_4_12 = indices_out[row_out + (long long)clamped_3_12];
                            if (position_1_12 >= active_0_1) {
                                value_4_12 = -1;
                            }
                            nxt[12] = value_4_12;
                            int position_1_13 = off_3_7 + 1;
                            int clamped_3_13 = ((position_1_13) < (topk - 1) ? (position_1_13) : (topk - 1));
                            int value_4_13 = indices_out[row_out + (long long)clamped_3_13];
                            if (position_1_13 >= active_0_1) {
                                value_4_13 = -1;
                            }
                            nxt[13] = value_4_13;
                            int position_1_14 = off_3_7 + 2;
                            int clamped_3_14 = ((position_1_14) < (topk - 1) ? (position_1_14) : (topk - 1));
                            int value_4_14 = indices_out[row_out + (long long)clamped_3_14];
                            if (position_1_14 >= active_0_1) {
                                value_4_14 = -1;
                            }
                            nxt[14] = value_4_14;
                            int position_1_15 = off_3_7 + 3;
                            int clamped_3_15 = ((position_1_15) < (topk - 1) ? (position_1_15) : (topk - 1));
                            int value_4_15 = indices_out[row_out + (long long)clamped_3_15];
                            if (position_1_15 >= active_0_1) {
                                value_4_15 = -1;
                            }
                            nxt[15] = value_4_15;
                        }
                    }
                }
                #pragma unroll 1
                for (int block_1 = 0; block_1 < blocks_1; block_1++) {
                    if (elect_sync()) {
                        int idx[16];
                        int smallest = num_kv;
                        int largest = -1;
                        int whole_tile = active_0_1 >= block_1 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (convert != 0) {
                            idx[0] = nxt[0];
                            idx[1] = nxt[1];
                            idx[2] = nxt[2];
                            idx[3] = nxt[3];
                            idx[4] = nxt[4];
                            idx[5] = nxt[5];
                            idx[6] = nxt[6];
                            idx[7] = nxt[7];
                            idx[8] = nxt[8];
                            idx[9] = nxt[9];
                            idx[10] = nxt[10];
                            idx[11] = nxt[11];
                            idx[12] = nxt[12];
                            idx[13] = nxt[13];
                            idx[14] = nxt[14];
                            idx[15] = nxt[15];
                            int _max_10 = ((idx[0]) > (idx[1]) ? (idx[0]) : (idx[1]));
                            int _max_11 = ((idx[2]) > (idx[3]) ? (idx[2]) : (idx[3]));
                            int _max_9 = ((_max_10) > (_max_11) ? (_max_10) : (_max_11));
                            int _max_8 = ((largest) > (_max_9) ? (largest) : (_max_9));
                            largest = _max_8;
                            int _min_14 = ((idx[0]) < (idx[1]) ? (idx[0]) : (idx[1]));
                            int _min_15 = ((idx[2]) < (idx[3]) ? (idx[2]) : (idx[3]));
                            int _min_13 = ((_min_14) < (_min_15) ? (_min_14) : (_min_15));
                            int _min_12 = ((smallest) < (_min_13) ? (smallest) : (_min_13));
                            smallest = _min_12;
                            int _max_18 = ((idx[4]) > (idx[5]) ? (idx[4]) : (idx[5]));
                            int _max_19 = ((idx[6]) > (idx[7]) ? (idx[6]) : (idx[7]));
                            int _max_17 = ((_max_18) > (_max_19) ? (_max_18) : (_max_19));
                            int _max_16 = ((largest) > (_max_17) ? (largest) : (_max_17));
                            largest = _max_16;
                            int _min_22 = ((idx[4]) < (idx[5]) ? (idx[4]) : (idx[5]));
                            int _min_23 = ((idx[6]) < (idx[7]) ? (idx[6]) : (idx[7]));
                            int _min_21 = ((_min_22) < (_min_23) ? (_min_22) : (_min_23));
                            int _min_20 = ((smallest) < (_min_21) ? (smallest) : (_min_21));
                            smallest = _min_20;
                            int _max_26 = ((idx[8]) > (idx[9]) ? (idx[8]) : (idx[9]));
                            int _max_27 = ((idx[10]) > (idx[11]) ? (idx[10]) : (idx[11]));
                            int _max_25 = ((_max_26) > (_max_27) ? (_max_26) : (_max_27));
                            int _max_24 = ((largest) > (_max_25) ? (largest) : (_max_25));
                            largest = _max_24;
                            int _min_30 = ((idx[8]) < (idx[9]) ? (idx[8]) : (idx[9]));
                            int _min_31 = ((idx[10]) < (idx[11]) ? (idx[10]) : (idx[11]));
                            int _min_29 = ((_min_30) < (_min_31) ? (_min_30) : (_min_31));
                            int _min_28 = ((smallest) < (_min_29) ? (smallest) : (_min_29));
                            smallest = _min_28;
                            int _max_34 = ((idx[12]) > (idx[13]) ? (idx[12]) : (idx[13]));
                            int _max_35 = ((idx[14]) > (idx[15]) ? (idx[14]) : (idx[15]));
                            int _max_33 = ((_max_34) > (_max_35) ? (_max_34) : (_max_35));
                            int _max_32 = ((largest) > (_max_33) ? (largest) : (_max_33));
                            largest = _max_32;
                            int _min_38 = ((idx[12]) < (idx[13]) ? (idx[12]) : (idx[13]));
                            int _min_39 = ((idx[14]) < (idx[15]) ? (idx[14]) : (idx[15]));
                            int _min_37 = ((_min_38) < (_min_39) ? (_min_38) : (_min_39));
                            int _min_36 = ((smallest) < (_min_37) ? (smallest) : (_min_37));
                            smallest = _min_36;
                            if (blocks_1 > block_1 + 1) {
                                int nb_1 = block_1 + 1;
                                int whole_out_1 = active_0_1 >= nb_1 * 64 + 64 && (topk & 3) == 0;
                                if (whole_out_1 != 0) {
                                    int off_3_8 = nb_1 * 64 + producer_warp * 4;
                                    int _vec_load_4[4];
                                    {
                                        const int4* _ivptr_40 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_8) + 0);
                                        int4 _ivld_40;
                                        _ivld_40 = *_ivptr_40;
                                        _vec_load_4[0 + 0] = _ivld_40.x;
                                        _vec_load_4[0 + 1] = _ivld_40.y;
                                        _vec_load_4[0 + 2] = _ivld_40.z;
                                        _vec_load_4[0 + 3] = _ivld_40.w;
                                    }
                                    nxt[0] = _vec_load_4[0];
                                    nxt[1] = _vec_load_4[1];
                                    nxt[2] = _vec_load_4[2];
                                    nxt[3] = _vec_load_4[3];
                                    int off_3_9 = nb_1 * 64 + 16 + producer_warp * 4;
                                    int _vec_load_5[4];
                                    {
                                        const int4* _ivptr_41 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_9) + 0);
                                        int4 _ivld_41;
                                        _ivld_41 = *_ivptr_41;
                                        _vec_load_5[0 + 0] = _ivld_41.x;
                                        _vec_load_5[0 + 1] = _ivld_41.y;
                                        _vec_load_5[0 + 2] = _ivld_41.z;
                                        _vec_load_5[0 + 3] = _ivld_41.w;
                                    }
                                    nxt[4] = _vec_load_5[0];
                                    nxt[5] = _vec_load_5[1];
                                    nxt[6] = _vec_load_5[2];
                                    nxt[7] = _vec_load_5[3];
                                    int off_3_10 = nb_1 * 64 + 32 + producer_warp * 4;
                                    int _vec_load_6[4];
                                    {
                                        const int4* _ivptr_42 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_10) + 0);
                                        int4 _ivld_42;
                                        _ivld_42 = *_ivptr_42;
                                        _vec_load_6[0 + 0] = _ivld_42.x;
                                        _vec_load_6[0 + 1] = _ivld_42.y;
                                        _vec_load_6[0 + 2] = _ivld_42.z;
                                        _vec_load_6[0 + 3] = _ivld_42.w;
                                    }
                                    nxt[8] = _vec_load_6[0];
                                    nxt[9] = _vec_load_6[1];
                                    nxt[10] = _vec_load_6[2];
                                    nxt[11] = _vec_load_6[3];
                                    int off_3_11 = nb_1 * 64 + 48 + producer_warp * 4;
                                    int _vec_load_7[4];
                                    {
                                        const int4* _ivptr_43 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_3_11) + 0);
                                        int4 _ivld_43;
                                        _ivld_43 = *_ivptr_43;
                                        _vec_load_7[0 + 0] = _ivld_43.x;
                                        _vec_load_7[0 + 1] = _ivld_43.y;
                                        _vec_load_7[0 + 2] = _ivld_43.z;
                                        _vec_load_7[0 + 3] = _ivld_43.w;
                                    }
                                    nxt[12] = _vec_load_7[0];
                                    nxt[13] = _vec_load_7[1];
                                    nxt[14] = _vec_load_7[2];
                                    nxt[15] = _vec_load_7[3];
                                } else {
                                    int off_3_12 = nb_1 * 64 + producer_warp * 4;
                                    int position_1_16 = off_3_12;
                                    int clamped_3_16 = ((position_1_16) < (topk - 1) ? (position_1_16) : (topk - 1));
                                    int value_4_16 = indices_out[row_out + (long long)clamped_3_16];
                                    if (position_1_16 >= active_0_1) {
                                        value_4_16 = -1;
                                    }
                                    nxt[0] = value_4_16;
                                    int position_1_17 = off_3_12 + 1;
                                    int clamped_3_17 = ((position_1_17) < (topk - 1) ? (position_1_17) : (topk - 1));
                                    int value_4_17 = indices_out[row_out + (long long)clamped_3_17];
                                    if (position_1_17 >= active_0_1) {
                                        value_4_17 = -1;
                                    }
                                    nxt[1] = value_4_17;
                                    int position_1_18 = off_3_12 + 2;
                                    int clamped_3_18 = ((position_1_18) < (topk - 1) ? (position_1_18) : (topk - 1));
                                    int value_4_18 = indices_out[row_out + (long long)clamped_3_18];
                                    if (position_1_18 >= active_0_1) {
                                        value_4_18 = -1;
                                    }
                                    nxt[2] = value_4_18;
                                    int position_1_19 = off_3_12 + 3;
                                    int clamped_3_19 = ((position_1_19) < (topk - 1) ? (position_1_19) : (topk - 1));
                                    int value_4_19 = indices_out[row_out + (long long)clamped_3_19];
                                    if (position_1_19 >= active_0_1) {
                                        value_4_19 = -1;
                                    }
                                    nxt[3] = value_4_19;
                                    int off_3_13 = nb_1 * 64 + 16 + producer_warp * 4;
                                    int position_1_20 = off_3_13;
                                    int clamped_3_20 = ((position_1_20) < (topk - 1) ? (position_1_20) : (topk - 1));
                                    int value_4_20 = indices_out[row_out + (long long)clamped_3_20];
                                    if (position_1_20 >= active_0_1) {
                                        value_4_20 = -1;
                                    }
                                    nxt[4] = value_4_20;
                                    int position_1_21 = off_3_13 + 1;
                                    int clamped_3_21 = ((position_1_21) < (topk - 1) ? (position_1_21) : (topk - 1));
                                    int value_4_21 = indices_out[row_out + (long long)clamped_3_21];
                                    if (position_1_21 >= active_0_1) {
                                        value_4_21 = -1;
                                    }
                                    nxt[5] = value_4_21;
                                    int position_1_22 = off_3_13 + 2;
                                    int clamped_3_22 = ((position_1_22) < (topk - 1) ? (position_1_22) : (topk - 1));
                                    int value_4_22 = indices_out[row_out + (long long)clamped_3_22];
                                    if (position_1_22 >= active_0_1) {
                                        value_4_22 = -1;
                                    }
                                    nxt[6] = value_4_22;
                                    int position_1_23 = off_3_13 + 3;
                                    int clamped_3_23 = ((position_1_23) < (topk - 1) ? (position_1_23) : (topk - 1));
                                    int value_4_23 = indices_out[row_out + (long long)clamped_3_23];
                                    if (position_1_23 >= active_0_1) {
                                        value_4_23 = -1;
                                    }
                                    nxt[7] = value_4_23;
                                    int off_3_14 = nb_1 * 64 + 32 + producer_warp * 4;
                                    int position_1_24 = off_3_14;
                                    int clamped_3_24 = ((position_1_24) < (topk - 1) ? (position_1_24) : (topk - 1));
                                    int value_4_24 = indices_out[row_out + (long long)clamped_3_24];
                                    if (position_1_24 >= active_0_1) {
                                        value_4_24 = -1;
                                    }
                                    nxt[8] = value_4_24;
                                    int position_1_25 = off_3_14 + 1;
                                    int clamped_3_25 = ((position_1_25) < (topk - 1) ? (position_1_25) : (topk - 1));
                                    int value_4_25 = indices_out[row_out + (long long)clamped_3_25];
                                    if (position_1_25 >= active_0_1) {
                                        value_4_25 = -1;
                                    }
                                    nxt[9] = value_4_25;
                                    int position_1_26 = off_3_14 + 2;
                                    int clamped_3_26 = ((position_1_26) < (topk - 1) ? (position_1_26) : (topk - 1));
                                    int value_4_26 = indices_out[row_out + (long long)clamped_3_26];
                                    if (position_1_26 >= active_0_1) {
                                        value_4_26 = -1;
                                    }
                                    nxt[10] = value_4_26;
                                    int position_1_27 = off_3_14 + 3;
                                    int clamped_3_27 = ((position_1_27) < (topk - 1) ? (position_1_27) : (topk - 1));
                                    int value_4_27 = indices_out[row_out + (long long)clamped_3_27];
                                    if (position_1_27 >= active_0_1) {
                                        value_4_27 = -1;
                                    }
                                    nxt[11] = value_4_27;
                                    int off_3_15 = nb_1 * 64 + 48 + producer_warp * 4;
                                    int position_1_28 = off_3_15;
                                    int clamped_3_28 = ((position_1_28) < (topk - 1) ? (position_1_28) : (topk - 1));
                                    int value_4_28 = indices_out[row_out + (long long)clamped_3_28];
                                    if (position_1_28 >= active_0_1) {
                                        value_4_28 = -1;
                                    }
                                    nxt[12] = value_4_28;
                                    int position_1_29 = off_3_15 + 1;
                                    int clamped_3_29 = ((position_1_29) < (topk - 1) ? (position_1_29) : (topk - 1));
                                    int value_4_29 = indices_out[row_out + (long long)clamped_3_29];
                                    if (position_1_29 >= active_0_1) {
                                        value_4_29 = -1;
                                    }
                                    nxt[13] = value_4_29;
                                    int position_1_30 = off_3_15 + 2;
                                    int clamped_3_30 = ((position_1_30) < (topk - 1) ? (position_1_30) : (topk - 1));
                                    int value_4_30 = indices_out[row_out + (long long)clamped_3_30];
                                    if (position_1_30 >= active_0_1) {
                                        value_4_30 = -1;
                                    }
                                    nxt[14] = value_4_30;
                                    int position_1_31 = off_3_15 + 3;
                                    int clamped_3_31 = ((position_1_31) < (topk - 1) ? (position_1_31) : (topk - 1));
                                    int value_4_31 = indices_out[row_out + (long long)clamped_3_31];
                                    if (position_1_31 >= active_0_1) {
                                        value_4_31 = -1;
                                    }
                                    nxt[15] = value_4_31;
                                }
                            }
                        } else if (whole_tile != 0) {
                            int _vec_load_8[4];
                            {
                                const int4* _ivptr_44 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + producer_warp * 4)) + 0);
                                int4 _ivld_44;
                                _ivld_44 = *_ivptr_44;
                                _vec_load_8[0 + 0] = _ivld_44.x;
                                _vec_load_8[0 + 1] = _ivld_44.y;
                                _vec_load_8[0 + 2] = _ivld_44.z;
                                _vec_load_8[0 + 3] = _ivld_44.w;
                            }
                            idx[0] = _vec_load_8[0];
                            idx[1] = _vec_load_8[1];
                            idx[2] = _vec_load_8[2];
                            idx[3] = _vec_load_8[3];
                            int _max_47 = ((_vec_load_8[0]) > (_vec_load_8[1]) ? (_vec_load_8[0]) : (_vec_load_8[1]));
                            int _max_48 = ((_vec_load_8[2]) > (_vec_load_8[3]) ? (_vec_load_8[2]) : (_vec_load_8[3]));
                            int _max_46 = ((_max_47) > (_max_48) ? (_max_47) : (_max_48));
                            int _max_45 = ((largest) > (_max_46) ? (largest) : (_max_46));
                            largest = _max_45;
                            int _min_51 = ((_vec_load_8[0]) < (_vec_load_8[1]) ? (_vec_load_8[0]) : (_vec_load_8[1]));
                            int _min_52 = ((_vec_load_8[2]) < (_vec_load_8[3]) ? (_vec_load_8[2]) : (_vec_load_8[3]));
                            int _min_50 = ((_min_51) < (_min_52) ? (_min_51) : (_min_52));
                            int _min_49 = ((smallest) < (_min_50) ? (smallest) : (_min_50));
                            smallest = _min_49;
                            int _vec_load_9[4];
                            {
                                const int4* _ivptr_53 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 16 + producer_warp * 4)) + 0);
                                int4 _ivld_53;
                                _ivld_53 = *_ivptr_53;
                                _vec_load_9[0 + 0] = _ivld_53.x;
                                _vec_load_9[0 + 1] = _ivld_53.y;
                                _vec_load_9[0 + 2] = _ivld_53.z;
                                _vec_load_9[0 + 3] = _ivld_53.w;
                            }
                            idx[4] = _vec_load_9[0];
                            idx[5] = _vec_load_9[1];
                            idx[6] = _vec_load_9[2];
                            idx[7] = _vec_load_9[3];
                            int _max_56 = ((_vec_load_9[0]) > (_vec_load_9[1]) ? (_vec_load_9[0]) : (_vec_load_9[1]));
                            int _max_57 = ((_vec_load_9[2]) > (_vec_load_9[3]) ? (_vec_load_9[2]) : (_vec_load_9[3]));
                            int _max_55 = ((_max_56) > (_max_57) ? (_max_56) : (_max_57));
                            int _max_54 = ((largest) > (_max_55) ? (largest) : (_max_55));
                            largest = _max_54;
                            int _min_60 = ((_vec_load_9[0]) < (_vec_load_9[1]) ? (_vec_load_9[0]) : (_vec_load_9[1]));
                            int _min_61 = ((_vec_load_9[2]) < (_vec_load_9[3]) ? (_vec_load_9[2]) : (_vec_load_9[3]));
                            int _min_59 = ((_min_60) < (_min_61) ? (_min_60) : (_min_61));
                            int _min_58 = ((smallest) < (_min_59) ? (smallest) : (_min_59));
                            smallest = _min_58;
                            int _vec_load_10[4];
                            {
                                const int4* _ivptr_62 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 32 + producer_warp * 4)) + 0);
                                int4 _ivld_62;
                                _ivld_62 = *_ivptr_62;
                                _vec_load_10[0 + 0] = _ivld_62.x;
                                _vec_load_10[0 + 1] = _ivld_62.y;
                                _vec_load_10[0 + 2] = _ivld_62.z;
                                _vec_load_10[0 + 3] = _ivld_62.w;
                            }
                            idx[8] = _vec_load_10[0];
                            idx[9] = _vec_load_10[1];
                            idx[10] = _vec_load_10[2];
                            idx[11] = _vec_load_10[3];
                            int _max_65 = ((_vec_load_10[0]) > (_vec_load_10[1]) ? (_vec_load_10[0]) : (_vec_load_10[1]));
                            int _max_66 = ((_vec_load_10[2]) > (_vec_load_10[3]) ? (_vec_load_10[2]) : (_vec_load_10[3]));
                            int _max_64 = ((_max_65) > (_max_66) ? (_max_65) : (_max_66));
                            int _max_63 = ((largest) > (_max_64) ? (largest) : (_max_64));
                            largest = _max_63;
                            int _min_69 = ((_vec_load_10[0]) < (_vec_load_10[1]) ? (_vec_load_10[0]) : (_vec_load_10[1]));
                            int _min_70 = ((_vec_load_10[2]) < (_vec_load_10[3]) ? (_vec_load_10[2]) : (_vec_load_10[3]));
                            int _min_68 = ((_min_69) < (_min_70) ? (_min_69) : (_min_70));
                            int _min_67 = ((smallest) < (_min_68) ? (smallest) : (_min_68));
                            smallest = _min_67;
                            int _vec_load_11[4];
                            {
                                const int4* _ivptr_71 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 48 + producer_warp * 4)) + 0);
                                int4 _ivld_71;
                                _ivld_71 = *_ivptr_71;
                                _vec_load_11[0 + 0] = _ivld_71.x;
                                _vec_load_11[0 + 1] = _ivld_71.y;
                                _vec_load_11[0 + 2] = _ivld_71.z;
                                _vec_load_11[0 + 3] = _ivld_71.w;
                            }
                            idx[12] = _vec_load_11[0];
                            idx[13] = _vec_load_11[1];
                            idx[14] = _vec_load_11[2];
                            idx[15] = _vec_load_11[3];
                            int _max_74 = ((_vec_load_11[0]) > (_vec_load_11[1]) ? (_vec_load_11[0]) : (_vec_load_11[1]));
                            int _max_75 = ((_vec_load_11[2]) > (_vec_load_11[3]) ? (_vec_load_11[2]) : (_vec_load_11[3]));
                            int _max_73 = ((_max_74) > (_max_75) ? (_max_74) : (_max_75));
                            int _max_72 = ((largest) > (_max_73) ? (largest) : (_max_73));
                            largest = _max_72;
                            int _min_78 = ((_vec_load_11[0]) < (_vec_load_11[1]) ? (_vec_load_11[0]) : (_vec_load_11[1]));
                            int _min_79 = ((_vec_load_11[2]) < (_vec_load_11[3]) ? (_vec_load_11[2]) : (_vec_load_11[3]));
                            int _min_77 = ((_min_78) < (_min_79) ? (_min_78) : (_min_79));
                            int _min_76 = ((smallest) < (_min_77) ? (smallest) : (_min_77));
                            smallest = _min_76;
                        } else {
                            int position_1_32 = block_1 * 64 + producer_warp * 4;
                            int clamped_3_32 = ((position_1_32) < (topk - 1) ? (position_1_32) : (topk - 1));
                            int value_4_32 = indices[row_base + (long long)clamped_3_32];
                            if (position_1_32 >= active_0_1) {
                                value_4_32 = -1;
                            }
                            idx[0] = value_4_32;
                            int _max_80 = ((largest) > (value_4_32) ? (largest) : (value_4_32));
                            largest = _max_80;
                            int _min_81 = ((smallest) < (value_4_32) ? (smallest) : (value_4_32));
                            smallest = _min_81;
                            int position_1_33 = block_1 * 64 + producer_warp * 4 + 1;
                            int clamped_3_33 = ((position_1_33) < (topk - 1) ? (position_1_33) : (topk - 1));
                            int value_4_33 = indices[row_base + (long long)clamped_3_33];
                            if (position_1_33 >= active_0_1) {
                                value_4_33 = -1;
                            }
                            idx[1] = value_4_33;
                            int _max_82 = ((largest) > (value_4_33) ? (largest) : (value_4_33));
                            largest = _max_82;
                            int _min_83 = ((smallest) < (value_4_33) ? (smallest) : (value_4_33));
                            smallest = _min_83;
                            int position_1_34 = block_1 * 64 + producer_warp * 4 + 2;
                            int clamped_3_34 = ((position_1_34) < (topk - 1) ? (position_1_34) : (topk - 1));
                            int value_4_34 = indices[row_base + (long long)clamped_3_34];
                            if (position_1_34 >= active_0_1) {
                                value_4_34 = -1;
                            }
                            idx[2] = value_4_34;
                            int _max_84 = ((largest) > (value_4_34) ? (largest) : (value_4_34));
                            largest = _max_84;
                            int _min_85 = ((smallest) < (value_4_34) ? (smallest) : (value_4_34));
                            smallest = _min_85;
                            int position_1_35 = block_1 * 64 + producer_warp * 4 + 3;
                            int clamped_3_35 = ((position_1_35) < (topk - 1) ? (position_1_35) : (topk - 1));
                            int value_4_35 = indices[row_base + (long long)clamped_3_35];
                            if (position_1_35 >= active_0_1) {
                                value_4_35 = -1;
                            }
                            idx[3] = value_4_35;
                            int _max_86 = ((largest) > (value_4_35) ? (largest) : (value_4_35));
                            largest = _max_86;
                            int _min_87 = ((smallest) < (value_4_35) ? (smallest) : (value_4_35));
                            smallest = _min_87;
                            int position_1_36 = block_1 * 64 + 16 + producer_warp * 4;
                            int clamped_3_36 = ((position_1_36) < (topk - 1) ? (position_1_36) : (topk - 1));
                            int value_4_36 = indices[row_base + (long long)clamped_3_36];
                            if (position_1_36 >= active_0_1) {
                                value_4_36 = -1;
                            }
                            idx[4] = value_4_36;
                            int _max_88 = ((largest) > (value_4_36) ? (largest) : (value_4_36));
                            largest = _max_88;
                            int _min_89 = ((smallest) < (value_4_36) ? (smallest) : (value_4_36));
                            smallest = _min_89;
                            int position_1_37 = block_1 * 64 + 16 + producer_warp * 4 + 1;
                            int clamped_3_37 = ((position_1_37) < (topk - 1) ? (position_1_37) : (topk - 1));
                            int value_4_37 = indices[row_base + (long long)clamped_3_37];
                            if (position_1_37 >= active_0_1) {
                                value_4_37 = -1;
                            }
                            idx[5] = value_4_37;
                            int _max_90 = ((largest) > (value_4_37) ? (largest) : (value_4_37));
                            largest = _max_90;
                            int _min_91 = ((smallest) < (value_4_37) ? (smallest) : (value_4_37));
                            smallest = _min_91;
                            int position_1_38 = block_1 * 64 + 16 + producer_warp * 4 + 2;
                            int clamped_3_38 = ((position_1_38) < (topk - 1) ? (position_1_38) : (topk - 1));
                            int value_4_38 = indices[row_base + (long long)clamped_3_38];
                            if (position_1_38 >= active_0_1) {
                                value_4_38 = -1;
                            }
                            idx[6] = value_4_38;
                            int _max_92 = ((largest) > (value_4_38) ? (largest) : (value_4_38));
                            largest = _max_92;
                            int _min_93 = ((smallest) < (value_4_38) ? (smallest) : (value_4_38));
                            smallest = _min_93;
                            int position_1_39 = block_1 * 64 + 16 + producer_warp * 4 + 3;
                            int clamped_3_39 = ((position_1_39) < (topk - 1) ? (position_1_39) : (topk - 1));
                            int value_4_39 = indices[row_base + (long long)clamped_3_39];
                            if (position_1_39 >= active_0_1) {
                                value_4_39 = -1;
                            }
                            idx[7] = value_4_39;
                            int _max_94 = ((largest) > (value_4_39) ? (largest) : (value_4_39));
                            largest = _max_94;
                            int _min_95 = ((smallest) < (value_4_39) ? (smallest) : (value_4_39));
                            smallest = _min_95;
                            int position_1_40 = block_1 * 64 + 32 + producer_warp * 4;
                            int clamped_3_40 = ((position_1_40) < (topk - 1) ? (position_1_40) : (topk - 1));
                            int value_4_40 = indices[row_base + (long long)clamped_3_40];
                            if (position_1_40 >= active_0_1) {
                                value_4_40 = -1;
                            }
                            idx[8] = value_4_40;
                            int _max_96 = ((largest) > (value_4_40) ? (largest) : (value_4_40));
                            largest = _max_96;
                            int _min_97 = ((smallest) < (value_4_40) ? (smallest) : (value_4_40));
                            smallest = _min_97;
                            int position_1_41 = block_1 * 64 + 32 + producer_warp * 4 + 1;
                            int clamped_3_41 = ((position_1_41) < (topk - 1) ? (position_1_41) : (topk - 1));
                            int value_4_41 = indices[row_base + (long long)clamped_3_41];
                            if (position_1_41 >= active_0_1) {
                                value_4_41 = -1;
                            }
                            idx[9] = value_4_41;
                            int _max_98 = ((largest) > (value_4_41) ? (largest) : (value_4_41));
                            largest = _max_98;
                            int _min_99 = ((smallest) < (value_4_41) ? (smallest) : (value_4_41));
                            smallest = _min_99;
                            int position_1_42 = block_1 * 64 + 32 + producer_warp * 4 + 2;
                            int clamped_3_42 = ((position_1_42) < (topk - 1) ? (position_1_42) : (topk - 1));
                            int value_4_42 = indices[row_base + (long long)clamped_3_42];
                            if (position_1_42 >= active_0_1) {
                                value_4_42 = -1;
                            }
                            idx[10] = value_4_42;
                            int _max_100 = ((largest) > (value_4_42) ? (largest) : (value_4_42));
                            largest = _max_100;
                            int _min_101 = ((smallest) < (value_4_42) ? (smallest) : (value_4_42));
                            smallest = _min_101;
                            int position_1_43 = block_1 * 64 + 32 + producer_warp * 4 + 3;
                            int clamped_3_43 = ((position_1_43) < (topk - 1) ? (position_1_43) : (topk - 1));
                            int value_4_43 = indices[row_base + (long long)clamped_3_43];
                            if (position_1_43 >= active_0_1) {
                                value_4_43 = -1;
                            }
                            idx[11] = value_4_43;
                            int _max_102 = ((largest) > (value_4_43) ? (largest) : (value_4_43));
                            largest = _max_102;
                            int _min_103 = ((smallest) < (value_4_43) ? (smallest) : (value_4_43));
                            smallest = _min_103;
                            int position_1_44 = block_1 * 64 + 48 + producer_warp * 4;
                            int clamped_3_44 = ((position_1_44) < (topk - 1) ? (position_1_44) : (topk - 1));
                            int value_4_44 = indices[row_base + (long long)clamped_3_44];
                            if (position_1_44 >= active_0_1) {
                                value_4_44 = -1;
                            }
                            idx[12] = value_4_44;
                            int _max_104 = ((largest) > (value_4_44) ? (largest) : (value_4_44));
                            largest = _max_104;
                            int _min_105 = ((smallest) < (value_4_44) ? (smallest) : (value_4_44));
                            smallest = _min_105;
                            int position_1_45 = block_1 * 64 + 48 + producer_warp * 4 + 1;
                            int clamped_3_45 = ((position_1_45) < (topk - 1) ? (position_1_45) : (topk - 1));
                            int value_4_45 = indices[row_base + (long long)clamped_3_45];
                            if (position_1_45 >= active_0_1) {
                                value_4_45 = -1;
                            }
                            idx[13] = value_4_45;
                            int _max_106 = ((largest) > (value_4_45) ? (largest) : (value_4_45));
                            largest = _max_106;
                            int _min_107 = ((smallest) < (value_4_45) ? (smallest) : (value_4_45));
                            smallest = _min_107;
                            int position_1_46 = block_1 * 64 + 48 + producer_warp * 4 + 2;
                            int clamped_3_46 = ((position_1_46) < (topk - 1) ? (position_1_46) : (topk - 1));
                            int value_4_46 = indices[row_base + (long long)clamped_3_46];
                            if (position_1_46 >= active_0_1) {
                                value_4_46 = -1;
                            }
                            idx[14] = value_4_46;
                            int _max_108 = ((largest) > (value_4_46) ? (largest) : (value_4_46));
                            largest = _max_108;
                            int _min_109 = ((smallest) < (value_4_46) ? (smallest) : (value_4_46));
                            smallest = _min_109;
                            int position_1_47 = block_1 * 64 + 48 + producer_warp * 4 + 3;
                            int clamped_3_47 = ((position_1_47) < (topk - 1) ? (position_1_47) : (topk - 1));
                            int value_4_47 = indices[row_base + (long long)clamped_3_47];
                            if (position_1_47 >= active_0_1) {
                                value_4_47 = -1;
                            }
                            idx[15] = value_4_47;
                            int _max_110 = ((largest) > (value_4_47) ? (largest) : (value_4_47));
                            largest = _max_110;
                            int _min_111 = ((smallest) < (value_4_47) ? (smallest) : (value_4_47));
                            smallest = _min_111;
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
            int next_valid = 1;
            int next_query = 0;
            unsigned int _phase_q_full = 0;
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
                    if (outer_2 == 0) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qko_addr + (unsigned int)(q_slot * 65536)), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(query_2),
                               "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    }
                    mbarrier_wait(q_full_addr, _phase_q_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 0)), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 8)), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 16)), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 0 + 24)), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 0)), "l"(_tcgen05_cp_desc_8)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 8)), "l"(_tcgen05_cp_desc_9)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 16)), "l"(_tcgen05_cp_desc_10)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 24)), "l"(_tcgen05_cp_desc_11)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 0)), "l"(_tcgen05_cp_desc_12)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 8)), "l"(_tcgen05_cp_desc_13)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 16)), "l"(_tcgen05_cp_desc_14)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 24)), "l"(_tcgen05_cp_desc_15)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 0)), "l"(_tcgen05_cp_desc_16)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 8)), "l"(_tcgen05_cp_desc_17)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 16)), "l"(_tcgen05_cp_desc_18)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 24)), "l"(_tcgen05_cp_desc_19)
                            : "memory");
                    }
                    tcgen05_commit(q_copied_addr);
                    #pragma unroll 1
                    for (int block_2 = 0; block_2 < blocks_2 + 1; block_2++) {
                        if (blocks_2 > block_2) {
                            int slot_2 = (cursor_2 + block_2) % 3;
                            int _parity_p_free = (int)((unsigned int)(cursor_2 + block_2) % 2) ^ 1;
                            mbarrier_wait(p_free_addr, _parity_p_free);
                            if (block_2 == 0) {
                                mbarrier_wait(q_copied_addr, _phase_q_full);
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_lo_addr + (slot_2) * 8, 32768);
                            int _stage_kv_ready_lo = (int)((unsigned int)(cursor_2 + block_2) % 3);
                            int _parity_kv_ready_lo = (int)((unsigned int)(cursor_2 + block_2) / 3 % 2);
                            mbarrier_wait(kv_ready_lo_addr + (_stage_kv_ready_lo) * 8, _parity_kv_ready_lo);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_20 = (((k_dual_addr) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_20 = (((k_dual_addr) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    :: "r"((tmem_tmem + (384))), "r"(_smem_desc_lo_20), "r"(tmem_tmem + 256), "r"(0) : "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_hi_addr + (slot_2) * 8, 32768);
                            int _stage_kv_ready_hi = (int)((unsigned int)(cursor_2 + block_2) % 3);
                            int _parity_kv_ready_hi = (int)((unsigned int)(cursor_2 + block_2) / 3 % 2);
                            mbarrier_wait(kv_ready_hi_addr + (_stage_kv_ready_hi) * 8, _parity_kv_ready_hi);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_21 = (((k_dual_addr + 32768) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_21 = (((k_dual_addr + 32768) >> 4) & 0x3FFF) + (slot_2) * 4096;
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
                    :: "r"((tmem_tmem + (384))), "r"(_smem_desc_lo_21), "r"(tmem_tmem + 320), "r"(1) : "memory");
                            tcgen05_commit(qk_done_addr + (slot_2) * 8);
                        }
                        if (block_2 > 0) {
                            int slot_3 = (cursor_2 + block_2 - 1) % 3;
                            int _parity_so_ready = (int)((unsigned int)(cursor_2 + block_2) % 2) ^ 1;
                            mbarrier_wait(so_ready_addr, _parity_so_ready);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x7FFF) | 0x400000;
                            int _smem_desc_lo_22 = ((((v_addr) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x3FFF) | 0x400000;
                            int _smem_desc_lo_22 = ((((v_addr) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_22), "r"(tmem_tmem), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_23 = ((((v_addr + 32768) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _smem_desc_lo_23 = ((((v_addr + 32768) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_23), "r"((tmem_tmem + (128))), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
                            tcgen05_commit(sv_done_addr + (slot_3) * 8);
                        }
                        if (block_2 == blocks_2 - 1) {
                            mbarrier_wait(clc_full_addr, _phase_q_full);
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
                            next_valid = _clc_valid_2;
                            next_query = _clc_ctaid_x_2;
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(clc_empty_addr);
                            if (next_valid != 0) {
                                int _stage_sv_done_3 = (int)((unsigned int)(cursor_2 + block_2 - 1) % 3);
                                int _parity_sv_done_3 = (int)((unsigned int)(cursor_2 + block_2 - 1) / 3 % 2);
                                mbarrier_wait(sv_done_addr + (_stage_sv_done_3) * 8, _parity_sv_done_3);
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(qko_addr + (unsigned int)((cursor_2 + block_2 + 2) % 3 * 65536)), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(next_query),
                                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive_expect_tx(q_full_addr, 65536);
                            }
                        }
                    }
                    cursor_2 += blocks_2;
                    if (next_valid == 0) {
                        break;
                    }
                    query_2 = next_query;
                    _phase_q_full ^= 1;
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
                    int bound = num_kv - 1;
                    if (derive_length != 0) {
                        bound = bound_cells[outer_3 & 1];
                    }
                    #pragma unroll 1
                    for (int block_3 = 0; block_3 < blocks_3; block_3++) {
                        unsigned int mask = 0;
                        int position = block_3 * 64 + lane * 8;
                        int whole_tile_1 = active_0_3 >= block_3 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (convert == 1) {
                            if (active_0_3 > position) {
                                mask = mask | 1;
                            }
                            if (active_0_3 > position + 1) {
                                mask = mask | 2;
                            }
                            if (active_0_3 > position + 2) {
                                mask = mask | 4;
                            }
                            if (active_0_3 > position + 3) {
                                mask = mask | 8;
                            }
                            if (active_0_3 > position + 4) {
                                mask = mask | 16;
                            }
                            if (active_0_3 > position + 5) {
                                mask = mask | 32;
                            }
                            if (active_0_3 > position + 6) {
                                mask = mask | 64;
                            }
                            if (active_0_3 > position + 7) {
                                mask = mask | 128;
                            }
                        } else if (whole_tile_1 != 0) {
                            int _vec_load_12[8];
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
                                _vec_load_12[0 + 0] = (int32_t)_iv_4_0;
                                _vec_load_12[0 + 1] = (int32_t)_iv_4_1;
                                _vec_load_12[0 + 2] = (int32_t)_iv_4_2;
                                _vec_load_12[0 + 3] = (int32_t)_iv_4_3;
                                _vec_load_12[0 + 4] = (int32_t)_iv_4_4;
                                _vec_load_12[0 + 5] = (int32_t)_iv_4_5;
                                _vec_load_12[0 + 6] = (int32_t)_iv_4_6;
                                _vec_load_12[0 + 7] = (int32_t)_iv_4_7;
                            }
                            if (_vec_load_12[0] >= 0 && _vec_load_12[0] <= bound) {
                                mask = mask | 1;
                            }
                            if (_vec_load_12[1] >= 0 && _vec_load_12[1] <= bound) {
                                mask = mask | 2;
                            }
                            if (_vec_load_12[2] >= 0 && _vec_load_12[2] <= bound) {
                                mask = mask | 4;
                            }
                            if (_vec_load_12[3] >= 0 && _vec_load_12[3] <= bound) {
                                mask = mask | 8;
                            }
                            if (_vec_load_12[4] >= 0 && _vec_load_12[4] <= bound) {
                                mask = mask | 16;
                            }
                            if (_vec_load_12[5] >= 0 && _vec_load_12[5] <= bound) {
                                mask = mask | 32;
                            }
                            if (_vec_load_12[6] >= 0 && _vec_load_12[6] <= bound) {
                                mask = mask | 64;
                            }
                            if (_vec_load_12[7] >= 0 && _vec_load_12[7] <= bound) {
                                mask = mask | 128;
                            }
                        } else {
                            int clamped_3_48 = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int value_4_48 = indices[row_base_1 + (long long)clamped_3_48];
                            if (value_4_48 >= 0 && value_4_48 <= bound && active_0_3 > position) {
                                mask = mask | 1;
                            }
                            int clamped_3_49 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                            int value_4_49 = indices[row_base_1 + (long long)clamped_3_49];
                            if (value_4_49 >= 0 && value_4_49 <= bound && active_0_3 > position + 1) {
                                mask = mask | 2;
                            }
                            int clamped_3_50 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                            int value_4_50 = indices[row_base_1 + (long long)clamped_3_50];
                            if (value_4_50 >= 0 && value_4_50 <= bound && active_0_3 > position + 2) {
                                mask = mask | 4;
                            }
                            int clamped_3_51 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                            int value_4_51 = indices[row_base_1 + (long long)clamped_3_51];
                            if (value_4_51 >= 0 && value_4_51 <= bound && active_0_3 > position + 3) {
                                mask = mask | 8;
                            }
                            int clamped_3_52 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                            int value_4_52 = indices[row_base_1 + (long long)clamped_3_52];
                            if (value_4_52 >= 0 && value_4_52 <= bound && active_0_3 > position + 4) {
                                mask = mask | 16;
                            }
                            int clamped_3_53 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                            int value_4_53 = indices[row_base_1 + (long long)clamped_3_53];
                            if (value_4_53 >= 0 && value_4_53 <= bound && active_0_3 > position + 5) {
                                mask = mask | 32;
                            }
                            int clamped_3_54 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                            int value_4_54 = indices[row_base_1 + (long long)clamped_3_54];
                            if (value_4_54 >= 0 && value_4_54 <= bound && active_0_3 > position + 6) {
                                mask = mask | 64;
                            }
                            int clamped_3_55 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                            int value_4_55 = indices[row_base_1 + (long long)clamped_3_55];
                            if (value_4_55 >= 0 && value_4_55 <= bound && active_0_3 > position + 7) {
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
                    query_3 = _clc_ctaid_x_3;
                    _phase_clc_full_0 ^= 1;
                }
            } else if (lane == 8) {
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
                    _phase_clc_empty ^= 1;
                    _phase_clc_full ^= 1;
                }
            } else {
                if (lane >= 16) {
                    int query_4 = blockIdx.x;
                    unsigned int outer_phase_4 = 0;
                    int lane16 = lane - 16;
                    int scan_aligned = ((idx_stride | indices_offset) & 7) == 0;
                    int out_vec = (topk & 3) == 0;
                    #pragma unroll 1
                    for (int outer_5 = 0; outer_5 < num_queries; outer_5++) {
                        if (derive_length != 0) {
                            long long row_base_2 = (long long)indices_offset + (long long)query_4 * (long long)idx_stride;
                            long long out_base = (long long)query_4 * (long long)topk;
                            int key_base = 0;
                            int bound_1 = num_kv - 1;
                            if (convert != 0) {
                                int lo = 0;
                                int hi = num_docs;
                                #pragma unroll 1
                                for (int it = 0; it < search_iters; it++) {
                                    int mid = (lo + hi) / 2;
                                    int _vec_load_13[1];
                                    {
                                        _vec_load_13[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)mid + 0);
                                    }
                                    int go_lo = (int)(mid > lo && _vec_load_13[0] <= query_4);
                                    int go_hi = (int)(mid > lo && _vec_load_13[0] > query_4);
                                    lo = ((go_lo != 0) ? mid : lo);
                                    hi = ((go_hi != 0) ? mid : hi);
                                }
                                int _vec_load_14[1];
                                {
                                    _vec_load_14[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)lo + 0);
                                }
                                int _vec_load_15[1];
                                {
                                    _vec_load_15[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)(lo + 1) + 0);
                                }
                                key_base = _vec_load_14[0];
                                int k_end = _vec_load_15[0];
                                bound_1 = k_end - key_base - 1;
                                if (causal != 0) {
                                    int _vec_load_16[1];
                                    {
                                        _vec_load_16[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)(lo + 1) + 0);
                                    }
                                    bound_1 = k_end - _vec_load_16[0] + query_4 - key_base;
                                }
                            }
                            int length = 0;
                            int n_full = topk / 128;
                            if (scan_aligned == 0) {
                                n_full = 0;
                            }
                            if (convert == 1) {
                                int base = 0;
                                int fill[4];
                                fill[0] = -1;
                                fill[1] = -1;
                                fill[2] = -1;
                                fill[3] = -1;
                                int n_batch = (n_full + 3) / 4;
                                #pragma unroll 1
                                for (int b = 0; b < n_batch; b++) {
                                    int group = ((b * 4) < (n_full - 1) ? (b * 4) : (n_full - 1));
                                    int spos = group * 128 + lane16 * 8;
                                    int _vec_load_17[8];
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
                                        _vec_load_17[0 + 0] = (int32_t)_iv_5_0;
                                        _vec_load_17[0 + 1] = (int32_t)_iv_5_1;
                                        _vec_load_17[0 + 2] = (int32_t)_iv_5_2;
                                        _vec_load_17[0 + 3] = (int32_t)_iv_5_3;
                                        _vec_load_17[0 + 4] = (int32_t)_iv_5_4;
                                        _vec_load_17[0 + 5] = (int32_t)_iv_5_5;
                                        _vec_load_17[0 + 6] = (int32_t)_iv_5_6;
                                        _vec_load_17[0 + 7] = (int32_t)_iv_5_7;
                                    }
                                    int group_0 = ((b * 4 + 1) < (n_full - 1) ? (b * 4 + 1) : (n_full - 1));
                                    int spos_1 = group_0 * 128 + lane16 * 8;
                                    int _vec_load_18[8];
                                    {
                                        uint32_t _iv_6_0;
                                        uint32_t _iv_6_1;
                                        uint32_t _iv_6_2;
                                        uint32_t _iv_6_3;
                                        uint32_t _iv_6_4;
                                        uint32_t _iv_6_5;
                                        uint32_t _iv_6_6;
                                        uint32_t _iv_6_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_1) + (0))) : "memory");
                                        _vec_load_18[0 + 0] = (int32_t)_iv_6_0;
                                        _vec_load_18[0 + 1] = (int32_t)_iv_6_1;
                                        _vec_load_18[0 + 2] = (int32_t)_iv_6_2;
                                        _vec_load_18[0 + 3] = (int32_t)_iv_6_3;
                                        _vec_load_18[0 + 4] = (int32_t)_iv_6_4;
                                        _vec_load_18[0 + 5] = (int32_t)_iv_6_5;
                                        _vec_load_18[0 + 6] = (int32_t)_iv_6_6;
                                        _vec_load_18[0 + 7] = (int32_t)_iv_6_7;
                                    }
                                    int group_2 = ((b * 4 + 2) < (n_full - 1) ? (b * 4 + 2) : (n_full - 1));
                                    int spos_3 = group_2 * 128 + lane16 * 8;
                                    int _vec_load_19[8];
                                    {
                                        uint32_t _iv_7_0;
                                        uint32_t _iv_7_1;
                                        uint32_t _iv_7_2;
                                        uint32_t _iv_7_3;
                                        uint32_t _iv_7_4;
                                        uint32_t _iv_7_5;
                                        uint32_t _iv_7_6;
                                        uint32_t _iv_7_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_3) + (0))) : "memory");
                                        _vec_load_19[0 + 0] = (int32_t)_iv_7_0;
                                        _vec_load_19[0 + 1] = (int32_t)_iv_7_1;
                                        _vec_load_19[0 + 2] = (int32_t)_iv_7_2;
                                        _vec_load_19[0 + 3] = (int32_t)_iv_7_3;
                                        _vec_load_19[0 + 4] = (int32_t)_iv_7_4;
                                        _vec_load_19[0 + 5] = (int32_t)_iv_7_5;
                                        _vec_load_19[0 + 6] = (int32_t)_iv_7_6;
                                        _vec_load_19[0 + 7] = (int32_t)_iv_7_7;
                                    }
                                    int group_4 = ((b * 4 + 3) < (n_full - 1) ? (b * 4 + 3) : (n_full - 1));
                                    int spos_5 = group_4 * 128 + lane16 * 8;
                                    int _vec_load_20[8];
                                    {
                                        uint32_t _iv_8_0;
                                        uint32_t _iv_8_1;
                                        uint32_t _iv_8_2;
                                        uint32_t _iv_8_3;
                                        uint32_t _iv_8_4;
                                        uint32_t _iv_8_5;
                                        uint32_t _iv_8_6;
                                        uint32_t _iv_8_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_8_0), "=r"(_iv_8_1), "=r"(_iv_8_2), "=r"(_iv_8_3), "=r"(_iv_8_4), "=r"(_iv_8_5), "=r"(_iv_8_6), "=r"(_iv_8_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_5) + (0))) : "memory");
                                        _vec_load_20[0 + 0] = (int32_t)_iv_8_0;
                                        _vec_load_20[0 + 1] = (int32_t)_iv_8_1;
                                        _vec_load_20[0 + 2] = (int32_t)_iv_8_2;
                                        _vec_load_20[0 + 3] = (int32_t)_iv_8_3;
                                        _vec_load_20[0 + 4] = (int32_t)_iv_8_4;
                                        _vec_load_20[0 + 5] = (int32_t)_iv_8_5;
                                        _vec_load_20[0 + 6] = (int32_t)_iv_8_6;
                                        _vec_load_20[0 + 7] = (int32_t)_iv_8_7;
                                    }
                                    if (n_full > b * 4) {
                                        int ok_0[8];
                                        int outv_0[8];
                                        unsigned int ballots_0[8];
                                        int prefix_0 = 0;
                                        int total_0 = 0;
                                        ok_0[0] = (int)(_vec_load_17[0] >= 0 && _vec_load_17[0] <= bound_1);
                                        outv_0[0] = -1;
                                        if (ok_0[0] != 0) {
                                            outv_0[0] = _vec_load_17[0] + key_base;
                                        }
                                        unsigned int _warp_vote_9 = __ballot_sync(4294901760u, ok_0[0] != 0);
                                        ballots_0[0] = _warp_vote_9;
                                        int _rank_in_mask_10 = (int)__popc(ballots_0[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_10;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[0]);
                                        ok_0[1] = (int)(_vec_load_17[1] >= 0 && _vec_load_17[1] <= bound_1);
                                        outv_0[1] = -1;
                                        if (ok_0[1] != 0) {
                                            outv_0[1] = _vec_load_17[1] + key_base;
                                        }
                                        unsigned int _warp_vote_11 = __ballot_sync(4294901760u, ok_0[1] != 0);
                                        ballots_0[1] = _warp_vote_11;
                                        int _rank_in_mask_12 = (int)__popc(ballots_0[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_12;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[1]);
                                        ok_0[2] = (int)(_vec_load_17[2] >= 0 && _vec_load_17[2] <= bound_1);
                                        outv_0[2] = -1;
                                        if (ok_0[2] != 0) {
                                            outv_0[2] = _vec_load_17[2] + key_base;
                                        }
                                        unsigned int _warp_vote_13 = __ballot_sync(4294901760u, ok_0[2] != 0);
                                        ballots_0[2] = _warp_vote_13;
                                        int _rank_in_mask_14 = (int)__popc(ballots_0[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_14;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[2]);
                                        ok_0[3] = (int)(_vec_load_17[3] >= 0 && _vec_load_17[3] <= bound_1);
                                        outv_0[3] = -1;
                                        if (ok_0[3] != 0) {
                                            outv_0[3] = _vec_load_17[3] + key_base;
                                        }
                                        unsigned int _warp_vote_15 = __ballot_sync(4294901760u, ok_0[3] != 0);
                                        ballots_0[3] = _warp_vote_15;
                                        int _rank_in_mask_16 = (int)__popc(ballots_0[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_16;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[3]);
                                        ok_0[4] = (int)(_vec_load_17[4] >= 0 && _vec_load_17[4] <= bound_1);
                                        outv_0[4] = -1;
                                        if (ok_0[4] != 0) {
                                            outv_0[4] = _vec_load_17[4] + key_base;
                                        }
                                        unsigned int _warp_vote_17 = __ballot_sync(4294901760u, ok_0[4] != 0);
                                        ballots_0[4] = _warp_vote_17;
                                        int _rank_in_mask_18 = (int)__popc(ballots_0[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_18;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[4]);
                                        ok_0[5] = (int)(_vec_load_17[5] >= 0 && _vec_load_17[5] <= bound_1);
                                        outv_0[5] = -1;
                                        if (ok_0[5] != 0) {
                                            outv_0[5] = _vec_load_17[5] + key_base;
                                        }
                                        unsigned int _warp_vote_19 = __ballot_sync(4294901760u, ok_0[5] != 0);
                                        ballots_0[5] = _warp_vote_19;
                                        int _rank_in_mask_20 = (int)__popc(ballots_0[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_20;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[5]);
                                        ok_0[6] = (int)(_vec_load_17[6] >= 0 && _vec_load_17[6] <= bound_1);
                                        outv_0[6] = -1;
                                        if (ok_0[6] != 0) {
                                            outv_0[6] = _vec_load_17[6] + key_base;
                                        }
                                        unsigned int _warp_vote_21 = __ballot_sync(4294901760u, ok_0[6] != 0);
                                        ballots_0[6] = _warp_vote_21;
                                        int _rank_in_mask_22 = (int)__popc(ballots_0[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_22;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[6]);
                                        ok_0[7] = (int)(_vec_load_17[7] >= 0 && _vec_load_17[7] <= bound_1);
                                        outv_0[7] = -1;
                                        if (ok_0[7] != 0) {
                                            outv_0[7] = _vec_load_17[7] + key_base;
                                        }
                                        unsigned int _warp_vote_23 = __ballot_sync(4294901760u, ok_0[7] != 0);
                                        ballots_0[7] = _warp_vote_23;
                                        int _rank_in_mask_24 = (int)__popc(ballots_0[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_0 = prefix_0 + _rank_in_mask_24;
                                        total_0 = total_0 + (int)(unsigned int)__popc(ballots_0[7]);
                                        if (out_vec != 0 && total_0 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_0[0 + 0], outv_0[0 + 1], outv_0[0 + 2], outv_0[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_0[4 + 0], outv_0[4 + 1], outv_0[4 + 2], outv_0[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4ll) = _iv4;
                                            }
                                        } else {
                                            int rank_0 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[0])), "r"((unsigned int)(ok_0[0] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[1])), "r"((unsigned int)(ok_0[1] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[2])), "r"((unsigned int)(ok_0[2] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[3])), "r"((unsigned int)(ok_0[3] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[4])), "r"((unsigned int)(ok_0[4] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[5])), "r"((unsigned int)(ok_0[5] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[6])), "r"((unsigned int)(ok_0[6] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_0 + rank_0)])), "r"((unsigned int)(outv_0[7])), "r"((unsigned int)(ok_0[7] != 0)) : "memory");
                                            rank_0 = rank_0 + ok_0[7];
                                        }
                                        base = base + total_0;
                                    }
                                    if (n_full > b * 4 + 1) {
                                        int ok_1[8];
                                        int outv_1[8];
                                        unsigned int ballots_1[8];
                                        int prefix_1 = 0;
                                        int total_1 = 0;
                                        ok_1[0] = (int)(_vec_load_18[0] >= 0 && _vec_load_18[0] <= bound_1);
                                        outv_1[0] = -1;
                                        if (ok_1[0] != 0) {
                                            outv_1[0] = _vec_load_18[0] + key_base;
                                        }
                                        unsigned int _warp_vote_25 = __ballot_sync(4294901760u, ok_1[0] != 0);
                                        ballots_1[0] = _warp_vote_25;
                                        int _rank_in_mask_26 = (int)__popc(ballots_1[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_26;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[0]);
                                        ok_1[1] = (int)(_vec_load_18[1] >= 0 && _vec_load_18[1] <= bound_1);
                                        outv_1[1] = -1;
                                        if (ok_1[1] != 0) {
                                            outv_1[1] = _vec_load_18[1] + key_base;
                                        }
                                        unsigned int _warp_vote_27 = __ballot_sync(4294901760u, ok_1[1] != 0);
                                        ballots_1[1] = _warp_vote_27;
                                        int _rank_in_mask_28 = (int)__popc(ballots_1[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_28;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[1]);
                                        ok_1[2] = (int)(_vec_load_18[2] >= 0 && _vec_load_18[2] <= bound_1);
                                        outv_1[2] = -1;
                                        if (ok_1[2] != 0) {
                                            outv_1[2] = _vec_load_18[2] + key_base;
                                        }
                                        unsigned int _warp_vote_29 = __ballot_sync(4294901760u, ok_1[2] != 0);
                                        ballots_1[2] = _warp_vote_29;
                                        int _rank_in_mask_30 = (int)__popc(ballots_1[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_30;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[2]);
                                        ok_1[3] = (int)(_vec_load_18[3] >= 0 && _vec_load_18[3] <= bound_1);
                                        outv_1[3] = -1;
                                        if (ok_1[3] != 0) {
                                            outv_1[3] = _vec_load_18[3] + key_base;
                                        }
                                        unsigned int _warp_vote_31 = __ballot_sync(4294901760u, ok_1[3] != 0);
                                        ballots_1[3] = _warp_vote_31;
                                        int _rank_in_mask_32 = (int)__popc(ballots_1[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_32;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[3]);
                                        ok_1[4] = (int)(_vec_load_18[4] >= 0 && _vec_load_18[4] <= bound_1);
                                        outv_1[4] = -1;
                                        if (ok_1[4] != 0) {
                                            outv_1[4] = _vec_load_18[4] + key_base;
                                        }
                                        unsigned int _warp_vote_33 = __ballot_sync(4294901760u, ok_1[4] != 0);
                                        ballots_1[4] = _warp_vote_33;
                                        int _rank_in_mask_34 = (int)__popc(ballots_1[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_34;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[4]);
                                        ok_1[5] = (int)(_vec_load_18[5] >= 0 && _vec_load_18[5] <= bound_1);
                                        outv_1[5] = -1;
                                        if (ok_1[5] != 0) {
                                            outv_1[5] = _vec_load_18[5] + key_base;
                                        }
                                        unsigned int _warp_vote_35 = __ballot_sync(4294901760u, ok_1[5] != 0);
                                        ballots_1[5] = _warp_vote_35;
                                        int _rank_in_mask_36 = (int)__popc(ballots_1[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_36;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[5]);
                                        ok_1[6] = (int)(_vec_load_18[6] >= 0 && _vec_load_18[6] <= bound_1);
                                        outv_1[6] = -1;
                                        if (ok_1[6] != 0) {
                                            outv_1[6] = _vec_load_18[6] + key_base;
                                        }
                                        unsigned int _warp_vote_37 = __ballot_sync(4294901760u, ok_1[6] != 0);
                                        ballots_1[6] = _warp_vote_37;
                                        int _rank_in_mask_38 = (int)__popc(ballots_1[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_38;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[6]);
                                        ok_1[7] = (int)(_vec_load_18[7] >= 0 && _vec_load_18[7] <= bound_1);
                                        outv_1[7] = -1;
                                        if (ok_1[7] != 0) {
                                            outv_1[7] = _vec_load_18[7] + key_base;
                                        }
                                        unsigned int _warp_vote_39 = __ballot_sync(4294901760u, ok_1[7] != 0);
                                        ballots_1[7] = _warp_vote_39;
                                        int _rank_in_mask_40 = (int)__popc(ballots_1[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_40;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[7]);
                                        if (out_vec != 0 && total_1 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_1[0 + 0], outv_1[0 + 1], outv_1[0 + 2], outv_1[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_1[4 + 0], outv_1[4 + 1], outv_1[4 + 2], outv_1[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4ll) = _iv4;
                                            }
                                        } else {
                                            int rank_1 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[0])), "r"((unsigned int)(ok_1[0] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[1])), "r"((unsigned int)(ok_1[1] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[2])), "r"((unsigned int)(ok_1[2] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[3])), "r"((unsigned int)(ok_1[3] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[4])), "r"((unsigned int)(ok_1[4] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[5])), "r"((unsigned int)(ok_1[5] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[6])), "r"((unsigned int)(ok_1[6] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[7])), "r"((unsigned int)(ok_1[7] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[7];
                                        }
                                        base = base + total_1;
                                    }
                                    if (n_full > b * 4 + 2) {
                                        int ok_2[8];
                                        int outv_2[8];
                                        unsigned int ballots_2[8];
                                        int prefix_2 = 0;
                                        int total_2 = 0;
                                        ok_2[0] = (int)(_vec_load_19[0] >= 0 && _vec_load_19[0] <= bound_1);
                                        outv_2[0] = -1;
                                        if (ok_2[0] != 0) {
                                            outv_2[0] = _vec_load_19[0] + key_base;
                                        }
                                        unsigned int _warp_vote_41 = __ballot_sync(4294901760u, ok_2[0] != 0);
                                        ballots_2[0] = _warp_vote_41;
                                        int _rank_in_mask_42 = (int)__popc(ballots_2[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_42;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[0]);
                                        ok_2[1] = (int)(_vec_load_19[1] >= 0 && _vec_load_19[1] <= bound_1);
                                        outv_2[1] = -1;
                                        if (ok_2[1] != 0) {
                                            outv_2[1] = _vec_load_19[1] + key_base;
                                        }
                                        unsigned int _warp_vote_43 = __ballot_sync(4294901760u, ok_2[1] != 0);
                                        ballots_2[1] = _warp_vote_43;
                                        int _rank_in_mask_44 = (int)__popc(ballots_2[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_44;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[1]);
                                        ok_2[2] = (int)(_vec_load_19[2] >= 0 && _vec_load_19[2] <= bound_1);
                                        outv_2[2] = -1;
                                        if (ok_2[2] != 0) {
                                            outv_2[2] = _vec_load_19[2] + key_base;
                                        }
                                        unsigned int _warp_vote_45 = __ballot_sync(4294901760u, ok_2[2] != 0);
                                        ballots_2[2] = _warp_vote_45;
                                        int _rank_in_mask_46 = (int)__popc(ballots_2[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_46;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[2]);
                                        ok_2[3] = (int)(_vec_load_19[3] >= 0 && _vec_load_19[3] <= bound_1);
                                        outv_2[3] = -1;
                                        if (ok_2[3] != 0) {
                                            outv_2[3] = _vec_load_19[3] + key_base;
                                        }
                                        unsigned int _warp_vote_47 = __ballot_sync(4294901760u, ok_2[3] != 0);
                                        ballots_2[3] = _warp_vote_47;
                                        int _rank_in_mask_48 = (int)__popc(ballots_2[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_48;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[3]);
                                        ok_2[4] = (int)(_vec_load_19[4] >= 0 && _vec_load_19[4] <= bound_1);
                                        outv_2[4] = -1;
                                        if (ok_2[4] != 0) {
                                            outv_2[4] = _vec_load_19[4] + key_base;
                                        }
                                        unsigned int _warp_vote_49 = __ballot_sync(4294901760u, ok_2[4] != 0);
                                        ballots_2[4] = _warp_vote_49;
                                        int _rank_in_mask_50 = (int)__popc(ballots_2[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_50;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[4]);
                                        ok_2[5] = (int)(_vec_load_19[5] >= 0 && _vec_load_19[5] <= bound_1);
                                        outv_2[5] = -1;
                                        if (ok_2[5] != 0) {
                                            outv_2[5] = _vec_load_19[5] + key_base;
                                        }
                                        unsigned int _warp_vote_51 = __ballot_sync(4294901760u, ok_2[5] != 0);
                                        ballots_2[5] = _warp_vote_51;
                                        int _rank_in_mask_52 = (int)__popc(ballots_2[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_52;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[5]);
                                        ok_2[6] = (int)(_vec_load_19[6] >= 0 && _vec_load_19[6] <= bound_1);
                                        outv_2[6] = -1;
                                        if (ok_2[6] != 0) {
                                            outv_2[6] = _vec_load_19[6] + key_base;
                                        }
                                        unsigned int _warp_vote_53 = __ballot_sync(4294901760u, ok_2[6] != 0);
                                        ballots_2[6] = _warp_vote_53;
                                        int _rank_in_mask_54 = (int)__popc(ballots_2[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_54;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[6]);
                                        ok_2[7] = (int)(_vec_load_19[7] >= 0 && _vec_load_19[7] <= bound_1);
                                        outv_2[7] = -1;
                                        if (ok_2[7] != 0) {
                                            outv_2[7] = _vec_load_19[7] + key_base;
                                        }
                                        unsigned int _warp_vote_55 = __ballot_sync(4294901760u, ok_2[7] != 0);
                                        ballots_2[7] = _warp_vote_55;
                                        int _rank_in_mask_56 = (int)__popc(ballots_2[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_56;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[7]);
                                        if (out_vec != 0 && total_2 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_2[0 + 0], outv_2[0 + 1], outv_2[0 + 2], outv_2[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_2[4 + 0], outv_2[4 + 1], outv_2[4 + 2], outv_2[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4ll) = _iv4;
                                            }
                                        } else {
                                            int rank_2 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[0])), "r"((unsigned int)(ok_2[0] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[1])), "r"((unsigned int)(ok_2[1] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[2])), "r"((unsigned int)(ok_2[2] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[3])), "r"((unsigned int)(ok_2[3] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[4])), "r"((unsigned int)(ok_2[4] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[5])), "r"((unsigned int)(ok_2[5] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[6])), "r"((unsigned int)(ok_2[6] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[7])), "r"((unsigned int)(ok_2[7] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[7];
                                        }
                                        base = base + total_2;
                                    }
                                    if (n_full > b * 4 + 3) {
                                        int ok_3[8];
                                        int outv_3[8];
                                        unsigned int ballots_3[8];
                                        int prefix_3 = 0;
                                        int total_3 = 0;
                                        ok_3[0] = (int)(_vec_load_20[0] >= 0 && _vec_load_20[0] <= bound_1);
                                        outv_3[0] = -1;
                                        if (ok_3[0] != 0) {
                                            outv_3[0] = _vec_load_20[0] + key_base;
                                        }
                                        unsigned int _warp_vote_57 = __ballot_sync(4294901760u, ok_3[0] != 0);
                                        ballots_3[0] = _warp_vote_57;
                                        int _rank_in_mask_58 = (int)__popc(ballots_3[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_58;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[0]);
                                        ok_3[1] = (int)(_vec_load_20[1] >= 0 && _vec_load_20[1] <= bound_1);
                                        outv_3[1] = -1;
                                        if (ok_3[1] != 0) {
                                            outv_3[1] = _vec_load_20[1] + key_base;
                                        }
                                        unsigned int _warp_vote_59 = __ballot_sync(4294901760u, ok_3[1] != 0);
                                        ballots_3[1] = _warp_vote_59;
                                        int _rank_in_mask_60 = (int)__popc(ballots_3[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_60;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[1]);
                                        ok_3[2] = (int)(_vec_load_20[2] >= 0 && _vec_load_20[2] <= bound_1);
                                        outv_3[2] = -1;
                                        if (ok_3[2] != 0) {
                                            outv_3[2] = _vec_load_20[2] + key_base;
                                        }
                                        unsigned int _warp_vote_61 = __ballot_sync(4294901760u, ok_3[2] != 0);
                                        ballots_3[2] = _warp_vote_61;
                                        int _rank_in_mask_62 = (int)__popc(ballots_3[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_62;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[2]);
                                        ok_3[3] = (int)(_vec_load_20[3] >= 0 && _vec_load_20[3] <= bound_1);
                                        outv_3[3] = -1;
                                        if (ok_3[3] != 0) {
                                            outv_3[3] = _vec_load_20[3] + key_base;
                                        }
                                        unsigned int _warp_vote_63 = __ballot_sync(4294901760u, ok_3[3] != 0);
                                        ballots_3[3] = _warp_vote_63;
                                        int _rank_in_mask_64 = (int)__popc(ballots_3[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_64;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[3]);
                                        ok_3[4] = (int)(_vec_load_20[4] >= 0 && _vec_load_20[4] <= bound_1);
                                        outv_3[4] = -1;
                                        if (ok_3[4] != 0) {
                                            outv_3[4] = _vec_load_20[4] + key_base;
                                        }
                                        unsigned int _warp_vote_65 = __ballot_sync(4294901760u, ok_3[4] != 0);
                                        ballots_3[4] = _warp_vote_65;
                                        int _rank_in_mask_66 = (int)__popc(ballots_3[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_66;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[4]);
                                        ok_3[5] = (int)(_vec_load_20[5] >= 0 && _vec_load_20[5] <= bound_1);
                                        outv_3[5] = -1;
                                        if (ok_3[5] != 0) {
                                            outv_3[5] = _vec_load_20[5] + key_base;
                                        }
                                        unsigned int _warp_vote_67 = __ballot_sync(4294901760u, ok_3[5] != 0);
                                        ballots_3[5] = _warp_vote_67;
                                        int _rank_in_mask_68 = (int)__popc(ballots_3[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_68;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[5]);
                                        ok_3[6] = (int)(_vec_load_20[6] >= 0 && _vec_load_20[6] <= bound_1);
                                        outv_3[6] = -1;
                                        if (ok_3[6] != 0) {
                                            outv_3[6] = _vec_load_20[6] + key_base;
                                        }
                                        unsigned int _warp_vote_69 = __ballot_sync(4294901760u, ok_3[6] != 0);
                                        ballots_3[6] = _warp_vote_69;
                                        int _rank_in_mask_70 = (int)__popc(ballots_3[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_70;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[6]);
                                        ok_3[7] = (int)(_vec_load_20[7] >= 0 && _vec_load_20[7] <= bound_1);
                                        outv_3[7] = -1;
                                        if (ok_3[7] != 0) {
                                            outv_3[7] = _vec_load_20[7] + key_base;
                                        }
                                        unsigned int _warp_vote_71 = __ballot_sync(4294901760u, ok_3[7] != 0);
                                        ballots_3[7] = _warp_vote_71;
                                        int _rank_in_mask_72 = (int)__popc(ballots_3[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_72;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[7]);
                                        if (out_vec != 0 && total_3 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_3[0 + 0], outv_3[0 + 1], outv_3[0 + 2], outv_3[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_3[4 + 0], outv_3[4 + 1], outv_3[4 + 2], outv_3[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4ll) = _iv4;
                                            }
                                        } else {
                                            int rank_3 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[0])), "r"((unsigned int)(ok_3[0] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[1])), "r"((unsigned int)(ok_3[1] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[2])), "r"((unsigned int)(ok_3[2] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[3])), "r"((unsigned int)(ok_3[3] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[4])), "r"((unsigned int)(ok_3[4] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[5])), "r"((unsigned int)(ok_3[5] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[6])), "r"((unsigned int)(ok_3[6] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[7])), "r"((unsigned int)(ok_3[7] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[7];
                                        }
                                        base = base + total_3;
                                    }
                                }
                                int tail0 = n_full * 128 + lane16;
                                int n_tail = (topk - n_full * 128 + 15) / 16;
                                #pragma unroll 1
                                for (int t = 0; t < n_tail; t++) {
                                    int pos = tail0 + t * 16;
                                    int value = -1;
                                    if (pos < topk) {
                                        value = indices[row_base_2 + (long long)pos];
                                    }
                                    int valid = (int)(value >= 0 && value <= bound_1);
                                    int converted = value + key_base;
                                    unsigned int ballot = __ballot_sync(4294901760u, valid != 0);
                                    int _rank_in_mask_73 = (int)__popc(ballot & ((1u << (unsigned int)lane) - 1u));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + _rank_in_mask_73)])), "r"((unsigned int)(converted)), "r"((unsigned int)(valid != 0)) : "memory");
                                    base = base + (int)(unsigned int)__popc(ballot);
                                }
                                if (out_vec != 0) {
                                    int head_1 = 4 - (base & 3) & 3;
                                    int hpos = base + lane16;
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)hpos])), "r"((unsigned int)(-1)), "r"((unsigned int)(lane16 < head_1 && hpos < topk)) : "memory");
                                    int base4 = base + head_1;
                                    int n_vec = (topk - base4) / 4;
                                    #pragma unroll 1
                                    for (int t_1 = 0; t_1 < (n_vec + 15) / 16; t_1++) {
                                        int chunk = t_1 * 16 + lane16;
                                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(indices_out + (out_base + (long long)(base4 + chunk * 4))), "r"(fill[(0) + 0]), "r"(fill[(0) + 1]), "r"(fill[(0) + 2]), "r"(fill[(0) + 3]), "r"((unsigned int)(chunk < n_vec)) : "memory");
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int t_2 = 0; t_2 < (topk - base + 15) / 16; t_2++) {
                                        int fpos = base + t_2 * 16 + lane16;
                                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)fpos])), "r"((unsigned int)(-1)), "r"((unsigned int)(fpos < topk)) : "memory");
                                    }
                                }
                                length = base;
                            } else if (convert == 2) {
                                int n_batch_1 = n_full / 4;
                                #pragma unroll 1
                                for (int b_1 = 0; b_1 < n_batch_1; b_1++) {
                                    int spos_2 = b_1 * 4 * 128 + lane16 * 8;
                                    int _vec_load_21[8];
                                    {
                                        uint32_t _iv_74_0;
                                        uint32_t _iv_74_1;
                                        uint32_t _iv_74_2;
                                        uint32_t _iv_74_3;
                                        uint32_t _iv_74_4;
                                        uint32_t _iv_74_5;
                                        uint32_t _iv_74_6;
                                        uint32_t _iv_74_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_74_0), "=r"(_iv_74_1), "=r"(_iv_74_2), "=r"(_iv_74_3), "=r"(_iv_74_4), "=r"(_iv_74_5), "=r"(_iv_74_6), "=r"(_iv_74_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_2) + (0))) : "memory");
                                        _vec_load_21[0 + 0] = (int32_t)_iv_74_0;
                                        _vec_load_21[0 + 1] = (int32_t)_iv_74_1;
                                        _vec_load_21[0 + 2] = (int32_t)_iv_74_2;
                                        _vec_load_21[0 + 3] = (int32_t)_iv_74_3;
                                        _vec_load_21[0 + 4] = (int32_t)_iv_74_4;
                                        _vec_load_21[0 + 5] = (int32_t)_iv_74_5;
                                        _vec_load_21[0 + 6] = (int32_t)_iv_74_6;
                                        _vec_load_21[0 + 7] = (int32_t)_iv_74_7;
                                    }
                                    int spos_0 = (b_1 * 4 + 1) * 128 + lane16 * 8;
                                    int _vec_load_22[8];
                                    {
                                        uint32_t _iv_75_0;
                                        uint32_t _iv_75_1;
                                        uint32_t _iv_75_2;
                                        uint32_t _iv_75_3;
                                        uint32_t _iv_75_4;
                                        uint32_t _iv_75_5;
                                        uint32_t _iv_75_6;
                                        uint32_t _iv_75_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_75_0), "=r"(_iv_75_1), "=r"(_iv_75_2), "=r"(_iv_75_3), "=r"(_iv_75_4), "=r"(_iv_75_5), "=r"(_iv_75_6), "=r"(_iv_75_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_0) + (0))) : "memory");
                                        _vec_load_22[0 + 0] = (int32_t)_iv_75_0;
                                        _vec_load_22[0 + 1] = (int32_t)_iv_75_1;
                                        _vec_load_22[0 + 2] = (int32_t)_iv_75_2;
                                        _vec_load_22[0 + 3] = (int32_t)_iv_75_3;
                                        _vec_load_22[0 + 4] = (int32_t)_iv_75_4;
                                        _vec_load_22[0 + 5] = (int32_t)_iv_75_5;
                                        _vec_load_22[0 + 6] = (int32_t)_iv_75_6;
                                        _vec_load_22[0 + 7] = (int32_t)_iv_75_7;
                                    }
                                    int spos_1_1 = (b_1 * 4 + 2) * 128 + lane16 * 8;
                                    int _vec_load_23[8];
                                    {
                                        uint32_t _iv_76_0;
                                        uint32_t _iv_76_1;
                                        uint32_t _iv_76_2;
                                        uint32_t _iv_76_3;
                                        uint32_t _iv_76_4;
                                        uint32_t _iv_76_5;
                                        uint32_t _iv_76_6;
                                        uint32_t _iv_76_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_76_0), "=r"(_iv_76_1), "=r"(_iv_76_2), "=r"(_iv_76_3), "=r"(_iv_76_4), "=r"(_iv_76_5), "=r"(_iv_76_6), "=r"(_iv_76_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_1_1) + (0))) : "memory");
                                        _vec_load_23[0 + 0] = (int32_t)_iv_76_0;
                                        _vec_load_23[0 + 1] = (int32_t)_iv_76_1;
                                        _vec_load_23[0 + 2] = (int32_t)_iv_76_2;
                                        _vec_load_23[0 + 3] = (int32_t)_iv_76_3;
                                        _vec_load_23[0 + 4] = (int32_t)_iv_76_4;
                                        _vec_load_23[0 + 5] = (int32_t)_iv_76_5;
                                        _vec_load_23[0 + 6] = (int32_t)_iv_76_6;
                                        _vec_load_23[0 + 7] = (int32_t)_iv_76_7;
                                    }
                                    int spos_2_1 = (b_1 * 4 + 3) * 128 + lane16 * 8;
                                    int _vec_load_24[8];
                                    {
                                        uint32_t _iv_77_0;
                                        uint32_t _iv_77_1;
                                        uint32_t _iv_77_2;
                                        uint32_t _iv_77_3;
                                        uint32_t _iv_77_4;
                                        uint32_t _iv_77_5;
                                        uint32_t _iv_77_6;
                                        uint32_t _iv_77_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_77_0), "=r"(_iv_77_1), "=r"(_iv_77_2), "=r"(_iv_77_3), "=r"(_iv_77_4), "=r"(_iv_77_5), "=r"(_iv_77_6), "=r"(_iv_77_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_2_1) + (0))) : "memory");
                                        _vec_load_24[0 + 0] = (int32_t)_iv_77_0;
                                        _vec_load_24[0 + 1] = (int32_t)_iv_77_1;
                                        _vec_load_24[0 + 2] = (int32_t)_iv_77_2;
                                        _vec_load_24[0 + 3] = (int32_t)_iv_77_3;
                                        _vec_load_24[0 + 4] = (int32_t)_iv_77_4;
                                        _vec_load_24[0 + 5] = (int32_t)_iv_77_5;
                                        _vec_load_24[0 + 6] = (int32_t)_iv_77_6;
                                        _vec_load_24[0 + 7] = (int32_t)_iv_77_7;
                                    }
                                    int outv[8];
                                    outv[0] = -1;
                                    if (_vec_load_21[0] >= 0 && _vec_load_21[0] <= bound_1) {
                                        outv[0] = _vec_load_21[0] + key_base;
                                        length = spos_2 + 1;
                                    }
                                    outv[1] = -1;
                                    if (_vec_load_21[1] >= 0 && _vec_load_21[1] <= bound_1) {
                                        outv[1] = _vec_load_21[1] + key_base;
                                        length = spos_2 + 1 + 1;
                                    }
                                    outv[2] = -1;
                                    if (_vec_load_21[2] >= 0 && _vec_load_21[2] <= bound_1) {
                                        outv[2] = _vec_load_21[2] + key_base;
                                        length = spos_2 + 2 + 1;
                                    }
                                    outv[3] = -1;
                                    if (_vec_load_21[3] >= 0 && _vec_load_21[3] <= bound_1) {
                                        outv[3] = _vec_load_21[3] + key_base;
                                        length = spos_2 + 3 + 1;
                                    }
                                    outv[4] = -1;
                                    if (_vec_load_21[4] >= 0 && _vec_load_21[4] <= bound_1) {
                                        outv[4] = _vec_load_21[4] + key_base;
                                        length = spos_2 + 4 + 1;
                                    }
                                    outv[5] = -1;
                                    if (_vec_load_21[5] >= 0 && _vec_load_21[5] <= bound_1) {
                                        outv[5] = _vec_load_21[5] + key_base;
                                        length = spos_2 + 5 + 1;
                                    }
                                    outv[6] = -1;
                                    if (_vec_load_21[6] >= 0 && _vec_load_21[6] <= bound_1) {
                                        outv[6] = _vec_load_21[6] + key_base;
                                        length = spos_2 + 6 + 1;
                                    }
                                    outv[7] = -1;
                                    if (_vec_load_21[7] >= 0 && _vec_load_21[7] <= bound_1) {
                                        outv[7] = _vec_load_21[7] + key_base;
                                        length = spos_2 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv[0 + 0], outv[0 + 1], outv[0 + 2], outv[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv[4 + 0], outv[4 + 1], outv[4 + 2], outv[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2 + 4ll) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_2] = outv[0];
                                        indices_out[out_base + (long long)(spos_2 + 1)] = outv[1];
                                        indices_out[out_base + (long long)(spos_2 + 2)] = outv[2];
                                        indices_out[out_base + (long long)(spos_2 + 3)] = outv[3];
                                        indices_out[out_base + (long long)(spos_2 + 4)] = outv[4];
                                        indices_out[out_base + (long long)(spos_2 + 5)] = outv[5];
                                        indices_out[out_base + (long long)(spos_2 + 6)] = outv[6];
                                        indices_out[out_base + (long long)(spos_2 + 7)] = outv[7];
                                    }
                                    int outv_3_1[8];
                                    outv_3_1[0] = -1;
                                    if (_vec_load_22[0] >= 0 && _vec_load_22[0] <= bound_1) {
                                        outv_3_1[0] = _vec_load_22[0] + key_base;
                                        length = spos_0 + 1;
                                    }
                                    outv_3_1[1] = -1;
                                    if (_vec_load_22[1] >= 0 && _vec_load_22[1] <= bound_1) {
                                        outv_3_1[1] = _vec_load_22[1] + key_base;
                                        length = spos_0 + 1 + 1;
                                    }
                                    outv_3_1[2] = -1;
                                    if (_vec_load_22[2] >= 0 && _vec_load_22[2] <= bound_1) {
                                        outv_3_1[2] = _vec_load_22[2] + key_base;
                                        length = spos_0 + 2 + 1;
                                    }
                                    outv_3_1[3] = -1;
                                    if (_vec_load_22[3] >= 0 && _vec_load_22[3] <= bound_1) {
                                        outv_3_1[3] = _vec_load_22[3] + key_base;
                                        length = spos_0 + 3 + 1;
                                    }
                                    outv_3_1[4] = -1;
                                    if (_vec_load_22[4] >= 0 && _vec_load_22[4] <= bound_1) {
                                        outv_3_1[4] = _vec_load_22[4] + key_base;
                                        length = spos_0 + 4 + 1;
                                    }
                                    outv_3_1[5] = -1;
                                    if (_vec_load_22[5] >= 0 && _vec_load_22[5] <= bound_1) {
                                        outv_3_1[5] = _vec_load_22[5] + key_base;
                                        length = spos_0 + 5 + 1;
                                    }
                                    outv_3_1[6] = -1;
                                    if (_vec_load_22[6] >= 0 && _vec_load_22[6] <= bound_1) {
                                        outv_3_1[6] = _vec_load_22[6] + key_base;
                                        length = spos_0 + 6 + 1;
                                    }
                                    outv_3_1[7] = -1;
                                    if (_vec_load_22[7] >= 0 && _vec_load_22[7] <= bound_1) {
                                        outv_3_1[7] = _vec_load_22[7] + key_base;
                                        length = spos_0 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_3_1[0 + 0], outv_3_1[0 + 1], outv_3_1[0 + 2], outv_3_1[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_0) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_3_1[4 + 0], outv_3_1[4 + 1], outv_3_1[4 + 2], outv_3_1[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_0 + 4ll) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_0] = outv_3_1[0];
                                        indices_out[out_base + (long long)(spos_0 + 1)] = outv_3_1[1];
                                        indices_out[out_base + (long long)(spos_0 + 2)] = outv_3_1[2];
                                        indices_out[out_base + (long long)(spos_0 + 3)] = outv_3_1[3];
                                        indices_out[out_base + (long long)(spos_0 + 4)] = outv_3_1[4];
                                        indices_out[out_base + (long long)(spos_0 + 5)] = outv_3_1[5];
                                        indices_out[out_base + (long long)(spos_0 + 6)] = outv_3_1[6];
                                        indices_out[out_base + (long long)(spos_0 + 7)] = outv_3_1[7];
                                    }
                                    int outv_4[8];
                                    outv_4[0] = -1;
                                    if (_vec_load_23[0] >= 0 && _vec_load_23[0] <= bound_1) {
                                        outv_4[0] = _vec_load_23[0] + key_base;
                                        length = spos_1_1 + 1;
                                    }
                                    outv_4[1] = -1;
                                    if (_vec_load_23[1] >= 0 && _vec_load_23[1] <= bound_1) {
                                        outv_4[1] = _vec_load_23[1] + key_base;
                                        length = spos_1_1 + 1 + 1;
                                    }
                                    outv_4[2] = -1;
                                    if (_vec_load_23[2] >= 0 && _vec_load_23[2] <= bound_1) {
                                        outv_4[2] = _vec_load_23[2] + key_base;
                                        length = spos_1_1 + 2 + 1;
                                    }
                                    outv_4[3] = -1;
                                    if (_vec_load_23[3] >= 0 && _vec_load_23[3] <= bound_1) {
                                        outv_4[3] = _vec_load_23[3] + key_base;
                                        length = spos_1_1 + 3 + 1;
                                    }
                                    outv_4[4] = -1;
                                    if (_vec_load_23[4] >= 0 && _vec_load_23[4] <= bound_1) {
                                        outv_4[4] = _vec_load_23[4] + key_base;
                                        length = spos_1_1 + 4 + 1;
                                    }
                                    outv_4[5] = -1;
                                    if (_vec_load_23[5] >= 0 && _vec_load_23[5] <= bound_1) {
                                        outv_4[5] = _vec_load_23[5] + key_base;
                                        length = spos_1_1 + 5 + 1;
                                    }
                                    outv_4[6] = -1;
                                    if (_vec_load_23[6] >= 0 && _vec_load_23[6] <= bound_1) {
                                        outv_4[6] = _vec_load_23[6] + key_base;
                                        length = spos_1_1 + 6 + 1;
                                    }
                                    outv_4[7] = -1;
                                    if (_vec_load_23[7] >= 0 && _vec_load_23[7] <= bound_1) {
                                        outv_4[7] = _vec_load_23[7] + key_base;
                                        length = spos_1_1 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_4[0 + 0], outv_4[0 + 1], outv_4[0 + 2], outv_4[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_1_1) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_4[4 + 0], outv_4[4 + 1], outv_4[4 + 2], outv_4[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_1_1 + 4ll) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_1_1] = outv_4[0];
                                        indices_out[out_base + (long long)(spos_1_1 + 1)] = outv_4[1];
                                        indices_out[out_base + (long long)(spos_1_1 + 2)] = outv_4[2];
                                        indices_out[out_base + (long long)(spos_1_1 + 3)] = outv_4[3];
                                        indices_out[out_base + (long long)(spos_1_1 + 4)] = outv_4[4];
                                        indices_out[out_base + (long long)(spos_1_1 + 5)] = outv_4[5];
                                        indices_out[out_base + (long long)(spos_1_1 + 6)] = outv_4[6];
                                        indices_out[out_base + (long long)(spos_1_1 + 7)] = outv_4[7];
                                    }
                                    int outv_5[8];
                                    outv_5[0] = -1;
                                    if (_vec_load_24[0] >= 0 && _vec_load_24[0] <= bound_1) {
                                        outv_5[0] = _vec_load_24[0] + key_base;
                                        length = spos_2_1 + 1;
                                    }
                                    outv_5[1] = -1;
                                    if (_vec_load_24[1] >= 0 && _vec_load_24[1] <= bound_1) {
                                        outv_5[1] = _vec_load_24[1] + key_base;
                                        length = spos_2_1 + 1 + 1;
                                    }
                                    outv_5[2] = -1;
                                    if (_vec_load_24[2] >= 0 && _vec_load_24[2] <= bound_1) {
                                        outv_5[2] = _vec_load_24[2] + key_base;
                                        length = spos_2_1 + 2 + 1;
                                    }
                                    outv_5[3] = -1;
                                    if (_vec_load_24[3] >= 0 && _vec_load_24[3] <= bound_1) {
                                        outv_5[3] = _vec_load_24[3] + key_base;
                                        length = spos_2_1 + 3 + 1;
                                    }
                                    outv_5[4] = -1;
                                    if (_vec_load_24[4] >= 0 && _vec_load_24[4] <= bound_1) {
                                        outv_5[4] = _vec_load_24[4] + key_base;
                                        length = spos_2_1 + 4 + 1;
                                    }
                                    outv_5[5] = -1;
                                    if (_vec_load_24[5] >= 0 && _vec_load_24[5] <= bound_1) {
                                        outv_5[5] = _vec_load_24[5] + key_base;
                                        length = spos_2_1 + 5 + 1;
                                    }
                                    outv_5[6] = -1;
                                    if (_vec_load_24[6] >= 0 && _vec_load_24[6] <= bound_1) {
                                        outv_5[6] = _vec_load_24[6] + key_base;
                                        length = spos_2_1 + 6 + 1;
                                    }
                                    outv_5[7] = -1;
                                    if (_vec_load_24[7] >= 0 && _vec_load_24[7] <= bound_1) {
                                        outv_5[7] = _vec_load_24[7] + key_base;
                                        length = spos_2_1 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_5[0 + 0], outv_5[0 + 1], outv_5[0 + 2], outv_5[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2_1) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_5[4 + 0], outv_5[4 + 1], outv_5[4 + 2], outv_5[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2_1 + 4ll) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_2_1] = outv_5[0];
                                        indices_out[out_base + (long long)(spos_2_1 + 1)] = outv_5[1];
                                        indices_out[out_base + (long long)(spos_2_1 + 2)] = outv_5[2];
                                        indices_out[out_base + (long long)(spos_2_1 + 3)] = outv_5[3];
                                        indices_out[out_base + (long long)(spos_2_1 + 4)] = outv_5[4];
                                        indices_out[out_base + (long long)(spos_2_1 + 5)] = outv_5[5];
                                        indices_out[out_base + (long long)(spos_2_1 + 6)] = outv_5[6];
                                        indices_out[out_base + (long long)(spos_2_1 + 7)] = outv_5[7];
                                    }
                                }
                                #pragma unroll 1
                                for (int r = 0; r < n_full - n_batch_1 * 4; r++) {
                                    int spos_4 = (n_batch_1 * 4 + r) * 128 + lane16 * 8;
                                    int _vec_load_25[8];
                                    {
                                        uint32_t _iv_78_0;
                                        uint32_t _iv_78_1;
                                        uint32_t _iv_78_2;
                                        uint32_t _iv_78_3;
                                        uint32_t _iv_78_4;
                                        uint32_t _iv_78_5;
                                        uint32_t _iv_78_6;
                                        uint32_t _iv_78_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_78_0), "=r"(_iv_78_1), "=r"(_iv_78_2), "=r"(_iv_78_3), "=r"(_iv_78_4), "=r"(_iv_78_5), "=r"(_iv_78_6), "=r"(_iv_78_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_4) + (0))) : "memory");
                                        _vec_load_25[0 + 0] = (int32_t)_iv_78_0;
                                        _vec_load_25[0 + 1] = (int32_t)_iv_78_1;
                                        _vec_load_25[0 + 2] = (int32_t)_iv_78_2;
                                        _vec_load_25[0 + 3] = (int32_t)_iv_78_3;
                                        _vec_load_25[0 + 4] = (int32_t)_iv_78_4;
                                        _vec_load_25[0 + 5] = (int32_t)_iv_78_5;
                                        _vec_load_25[0 + 6] = (int32_t)_iv_78_6;
                                        _vec_load_25[0 + 7] = (int32_t)_iv_78_7;
                                    }
                                    int outv_6[8];
                                    outv_6[0] = -1;
                                    if (_vec_load_25[0] >= 0 && _vec_load_25[0] <= bound_1) {
                                        outv_6[0] = _vec_load_25[0] + key_base;
                                        length = spos_4 + 1;
                                    }
                                    outv_6[1] = -1;
                                    if (_vec_load_25[1] >= 0 && _vec_load_25[1] <= bound_1) {
                                        outv_6[1] = _vec_load_25[1] + key_base;
                                        length = spos_4 + 1 + 1;
                                    }
                                    outv_6[2] = -1;
                                    if (_vec_load_25[2] >= 0 && _vec_load_25[2] <= bound_1) {
                                        outv_6[2] = _vec_load_25[2] + key_base;
                                        length = spos_4 + 2 + 1;
                                    }
                                    outv_6[3] = -1;
                                    if (_vec_load_25[3] >= 0 && _vec_load_25[3] <= bound_1) {
                                        outv_6[3] = _vec_load_25[3] + key_base;
                                        length = spos_4 + 3 + 1;
                                    }
                                    outv_6[4] = -1;
                                    if (_vec_load_25[4] >= 0 && _vec_load_25[4] <= bound_1) {
                                        outv_6[4] = _vec_load_25[4] + key_base;
                                        length = spos_4 + 4 + 1;
                                    }
                                    outv_6[5] = -1;
                                    if (_vec_load_25[5] >= 0 && _vec_load_25[5] <= bound_1) {
                                        outv_6[5] = _vec_load_25[5] + key_base;
                                        length = spos_4 + 5 + 1;
                                    }
                                    outv_6[6] = -1;
                                    if (_vec_load_25[6] >= 0 && _vec_load_25[6] <= bound_1) {
                                        outv_6[6] = _vec_load_25[6] + key_base;
                                        length = spos_4 + 6 + 1;
                                    }
                                    outv_6[7] = -1;
                                    if (_vec_load_25[7] >= 0 && _vec_load_25[7] <= bound_1) {
                                        outv_6[7] = _vec_load_25[7] + key_base;
                                        length = spos_4 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_6[0 + 0], outv_6[0 + 1], outv_6[0 + 2], outv_6[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_4) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_6[4 + 0], outv_6[4 + 1], outv_6[4 + 2], outv_6[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_4 + 4ll) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_4] = outv_6[0];
                                        indices_out[out_base + (long long)(spos_4 + 1)] = outv_6[1];
                                        indices_out[out_base + (long long)(spos_4 + 2)] = outv_6[2];
                                        indices_out[out_base + (long long)(spos_4 + 3)] = outv_6[3];
                                        indices_out[out_base + (long long)(spos_4 + 4)] = outv_6[4];
                                        indices_out[out_base + (long long)(spos_4 + 5)] = outv_6[5];
                                        indices_out[out_base + (long long)(spos_4 + 6)] = outv_6[6];
                                        indices_out[out_base + (long long)(spos_4 + 7)] = outv_6[7];
                                    }
                                }
                                int tail0_1 = n_full * 128 + lane16;
                                int n_tail_1 = (topk - n_full * 128 - lane16 + 15) / 16;
                                #pragma unroll 1
                                for (int t_3 = 0; t_3 < n_tail_1; t_3++) {
                                    int pos_1 = tail0_1 + t_3 * 16;
                                    int value_1 = indices[row_base_2 + (long long)pos_1];
                                    int converted_1 = -1;
                                    if (value_1 >= 0 && value_1 <= bound_1) {
                                        converted_1 = value_1 + key_base;
                                        length = pos_1 + 1;
                                    }
                                    indices_out[out_base + (long long)pos_1] = converted_1;
                                }
                            } else {
                                #pragma unroll 4
                                for (int g = 0; g < n_full; g++) {
                                    int spos_6 = g * 128 + lane16 * 8;
                                    int _vec_load_26[8];
                                    {
                                        uint32_t _iv_79_0;
                                        uint32_t _iv_79_1;
                                        uint32_t _iv_79_2;
                                        uint32_t _iv_79_3;
                                        uint32_t _iv_79_4;
                                        uint32_t _iv_79_5;
                                        uint32_t _iv_79_6;
                                        uint32_t _iv_79_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_79_0), "=r"(_iv_79_1), "=r"(_iv_79_2), "=r"(_iv_79_3), "=r"(_iv_79_4), "=r"(_iv_79_5), "=r"(_iv_79_6), "=r"(_iv_79_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_6) + (0))) : "memory");
                                        _vec_load_26[0 + 0] = (int32_t)_iv_79_0;
                                        _vec_load_26[0 + 1] = (int32_t)_iv_79_1;
                                        _vec_load_26[0 + 2] = (int32_t)_iv_79_2;
                                        _vec_load_26[0 + 3] = (int32_t)_iv_79_3;
                                        _vec_load_26[0 + 4] = (int32_t)_iv_79_4;
                                        _vec_load_26[0 + 5] = (int32_t)_iv_79_5;
                                        _vec_load_26[0 + 6] = (int32_t)_iv_79_6;
                                        _vec_load_26[0 + 7] = (int32_t)_iv_79_7;
                                    }
                                    if (_vec_load_26[0] >= 0 && _vec_load_26[0] <= bound_1) {
                                        length = spos_6 + 1;
                                    }
                                    if (_vec_load_26[1] >= 0 && _vec_load_26[1] <= bound_1) {
                                        length = spos_6 + 1 + 1;
                                    }
                                    if (_vec_load_26[2] >= 0 && _vec_load_26[2] <= bound_1) {
                                        length = spos_6 + 2 + 1;
                                    }
                                    if (_vec_load_26[3] >= 0 && _vec_load_26[3] <= bound_1) {
                                        length = spos_6 + 3 + 1;
                                    }
                                    if (_vec_load_26[4] >= 0 && _vec_load_26[4] <= bound_1) {
                                        length = spos_6 + 4 + 1;
                                    }
                                    if (_vec_load_26[5] >= 0 && _vec_load_26[5] <= bound_1) {
                                        length = spos_6 + 5 + 1;
                                    }
                                    if (_vec_load_26[6] >= 0 && _vec_load_26[6] <= bound_1) {
                                        length = spos_6 + 6 + 1;
                                    }
                                    if (_vec_load_26[7] >= 0 && _vec_load_26[7] <= bound_1) {
                                        length = spos_6 + 7 + 1;
                                    }
                                }
                                int tail0_2 = n_full * 128 + lane16;
                                int n_tail_2 = (topk - n_full * 128 - lane16 + 15) / 16;
                                #pragma unroll 1
                                for (int t_4 = 0; t_4 < n_tail_2; t_4++) {
                                    int pos_2 = tail0_2 + t_4 * 16;
                                    int value_2 = indices[row_base_2 + (long long)pos_2];
                                    if (value_2 >= 0 && value_2 <= bound_1) {
                                        length = pos_2 + 1;
                                    }
                                }
                            }
                            int part = length;
                            if (convert != 1) {
                                int peer_3_0 = __shfl_xor_sync(4294901760u, part, 8);
                                int _max_80_1 = ((part) > (peer_3_0) ? (part) : (peer_3_0));
                                part = _max_80_1;
                                int peer_3_1 = __shfl_xor_sync(4294901760u, part, 4);
                                int _max_81 = ((part) > (peer_3_1) ? (part) : (peer_3_1));
                                part = _max_81;
                                int peer_3_2 = __shfl_xor_sync(4294901760u, part, 2);
                                int _max_82_1 = ((part) > (peer_3_2) ? (part) : (peer_3_2));
                                part = _max_82_1;
                                int peer_3_3 = __shfl_xor_sync(4294901760u, part, 1);
                                int _max_83 = ((part) > (peer_3_3) ? (part) : (peer_3_3));
                                part = _max_83;
                            }
                            if (convert != 0) {
                                __syncwarp(4294901760u);
                            }
                            if (lane == 16) {
                                len_cells[outer_5 & 1] = part;
                                bound_cells[outer_5 & 1] = bound_1;
                                topk_length[(long long)query_4] = part;
                                mbarrier_arrive(len_ready_addr + ((int)((unsigned int)outer_5 % 2)) * 8);
                            }
                        }
                        mbarrier_wait(clc_full_addr, _phase_clc_full_0_1);
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
                        query_4 = _clc_ctaid_x_5;
                        _phase_clc_full_0_1 ^= 1;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
