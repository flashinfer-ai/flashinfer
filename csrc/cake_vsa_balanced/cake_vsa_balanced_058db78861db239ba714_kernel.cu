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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}




__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_balanced_058db78861db239ba714(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int* __restrict__ indptr, int* __restrict__ indices, int sequence_q, int query_blocks, int total_tiles, int num_heads, float softmax_scale_log2, int return_lse, unsigned int* __restrict__ queue_counters, int* __restrict__ order_debug, int debug_order)
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
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define s_empty_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 80)
    #define p_lastsplit_addr (mbar_base + 88)
    #define p_empty_addr (mbar_base + 96)
    #define corr_sig_addr (mbar_base + 104)
    #define corr_done_addr (mbar_base + 112)
    #define stats_full_addr (mbar_base + 120)
    #define o_full_addr (mbar_base + 128)
    #define tile_done_addr (mbar_base + 136)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)
    #define claim_gate_addr (mbar_base + 208)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 214016);
    const int scale_smem_addr = smem + 214016;
    unsigned int* partial_smem = reinterpret_cast<unsigned int*>(smem_raw + 215552);
    const int partial_smem_addr = smem + 215552;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 231936);
    const int work_token_words_addr = smem + 231936;
    unsigned int* hist_smem = reinterpret_cast<unsigned int*>(smem_raw + 232064);
    const int hist_smem_addr = smem + 232064;

    // Mbarrier init (27 barriers at smem_raw[0..216))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // s_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            // p_lastsplit: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // stats_full: 1 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=320
            mbarrier_init(smem + 176, 320);
            mbarrier_init(smem + 184, 320);
            mbarrier_init(smem + 192, 320);
            mbarrier_init(smem + 200, 320);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 320 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 0) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_scores = taddr;
    const unsigned int tmem_probabilities = taddr + 256;
    const unsigned int tmem_output = taddr + 128;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // softmax_main
            unsigned int work_stage_s = 0;
            int bid_t = blockIdx.x;
            int q_block_t = bid_t % query_blocks;
            int head_t = bid_t / query_blocks;
            int row_begin_t = indptr[q_block_t];
            int count_t = indptr[q_block_t + 1] - row_begin_t;
            int groups_t = (count_t + 4 - 1) / 4;
            unsigned int valid_t = 1;
            if (bid_t >= total_tiles) {
                valid_t = 0;
            }
            unsigned int gate_t = 1;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_corr_done_0 = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (int _tile_iter = 0; _tile_iter < total_tiles; _tile_iter++) {
                if (valid_t == 0) {
                    break;
                }
                int query_base = q_block_t * 64;
                int q_valid = sequence_q - query_base;
                if (q_valid > 64) {
                    q_valid = 64;
                }
                if (q_valid < 0) {
                    q_valid = 0;
                }
                const int warp_in_role = warp;
                const int tmem_row_origin = warp_in_role * 32;
                const int warp_col = warp_in_role / 2;
                int my_row = warp_in_role % 2 * 32 + lane;
                int stat_slot = warp_in_role * 32 + lane;
                int row_valid = ((my_row < q_valid) ? 1 : 0);
                float row_max = -CUDART_INF_F;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int group_index = 0; group_index < groups_t; group_index++) {
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int remaining = count_t - group_index * 4;
                    int valid_lo = ((remaining > warp_col && row_valid != 0) ? 64 : 0);
                    int valid_hi = ((remaining > warp_col + 2 && row_valid != 0) ? 64 : 0);
                    int valid_cols = valid_lo + valid_hi;
                    int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(score_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(score_addr + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                        : "r"(score_addr + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(score_addr + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(s_empty_addr);
                    if (valid_lo == 0) {
                        uint32_t _thresh_mask_0;
                        {
                            int _lim_1 = 64;
                            if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                            else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                            }
                        }
                        if (!(~_thresh_mask_0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                        if (!(~_thresh_mask_0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_2;
                        {
                            int _lim_3 = 32;
                            if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                            else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                            }
                        }
                        if (!(~_thresh_mask_2 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                        if (!(~_thresh_mask_2 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_4;
                        {
                            int _lim_5 = 0;
                            if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                            else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                            }
                        }
                        if (!(~_thresh_mask_4 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                        if (!(~_thresh_mask_4 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_6;
                        {
                            int _lim_7 = -32;
                            if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                            else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
                            }
                        }
                        if (!(~_thresh_mask_6 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                        if (!(~_thresh_mask_6 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    }
                    if (valid_hi == 0) {
                        uint32_t _thresh_mask_8;
                        {
                            int _lim_9 = 64;
                            if (_lim_9 <= 0) { _thresh_mask_8 = 0u; }
                            else if (_lim_9 >= 32) { _thresh_mask_8 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_8) : "r"(_lim_9));
                            }
                        }
                        if (!(_thresh_mask_8 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_10;
                        {
                            int _lim_11 = 32;
                            if (_lim_11 <= 0) { _thresh_mask_10 = 0u; }
                            else if (_lim_11 >= 32) { _thresh_mask_10 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_10) : "r"(_lim_11));
                            }
                        }
                        if (!(_thresh_mask_10 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_12;
                        {
                            int _lim_13 = 0;
                            if (_lim_13 <= 0) { _thresh_mask_12 = 0u; }
                            else if (_lim_13 >= 32) { _thresh_mask_12 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_12) : "r"(_lim_13));
                            }
                        }
                        if (!(_thresh_mask_12 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_12 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_14;
                        {
                            int _lim_15 = -32;
                            if (_lim_15 <= 0) { _thresh_mask_14 = 0u; }
                            else if (_lim_15 >= 32) { _thresh_mask_14 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_14) : "r"(_lim_15));
                            }
                        }
                        if (!(_thresh_mask_14 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    }
                    float2 _reg_reduce_max2_16 = {-CUDART_INF_F, -CUDART_INF_F};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_16);
                    float tile_max = row_max_reduce(_reg_reduce_max2_16);
                    if (valid_cols <= 0) {
                        tile_max = -CUDART_INF_F;
                    }
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                    float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale;
                    float selected_max;
                    if (acc_scale_log2 >= -8.0f) {
                        selected_max = row_max;
                        safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                        acc_scale = 1.0f;
                        new_max_scaled = safe_max * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        float _exp2_17 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_17 : 1.0f);
                    }
                    row_max = selected_max;
                    scale_smem[stat_slot] = acc_scale;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_sig_addr);
                    float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CUDART_INF_F);
                    const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_19 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_18, _fma_c2_19);
                    unsigned int packed_p_lo[32];
                    softmax_frag_exp2_cast(&_tmem_load_0[0], packed_p_lo, 0);
                    softmax_frag_exp2_cast(&_tmem_load_0[32], &packed_p_lo[16], 0);
                    int p_addr = taddr + (unsigned int)256 + (unsigned int)(tmem_row_origin << 16);
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_addr), "r"(packed_p_lo[0]), "r"(packed_p_lo[1]), "r"(packed_p_lo[2]), "r"(packed_p_lo[3]), "r"(packed_p_lo[4]), "r"(packed_p_lo[5]), "r"(packed_p_lo[6]), "r"(packed_p_lo[7]), "r"(packed_p_lo[8]), "r"(packed_p_lo[9]), "r"(packed_p_lo[10]), "r"(packed_p_lo[11]), "r"(packed_p_lo[12]), "r"(packed_p_lo[13]), "r"(packed_p_lo[14]), "r"(packed_p_lo[15]), "r"(packed_p_lo[16]), "r"(packed_p_lo[17]), "r"(packed_p_lo[18]), "r"(packed_p_lo[19]), "r"(packed_p_lo[20]), "r"(packed_p_lo[21]), "r"(packed_p_lo[22]), "r"(packed_p_lo[23]), "r"(packed_p_lo[24]), "r"(packed_p_lo[25]), "r"(packed_p_lo[26]), "r"(packed_p_lo[27]), "r"(packed_p_lo[28]), "r"(packed_p_lo[29]), "r"(packed_p_lo[30]), "r"(packed_p_lo[31]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr);
                    unsigned int packed_p_hi[32];
                    softmax_frag_exp2_cast(&_tmem_load_0[64], packed_p_hi, 0);
                    softmax_frag_exp2_cast(&_tmem_load_0[96], &packed_p_hi[16], 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_addr + 32), "r"(packed_p_hi[0]), "r"(packed_p_hi[1]), "r"(packed_p_hi[2]), "r"(packed_p_hi[3]), "r"(packed_p_hi[4]), "r"(packed_p_hi[5]), "r"(packed_p_hi[6]), "r"(packed_p_hi[7]), "r"(packed_p_hi[8]), "r"(packed_p_hi[9]), "r"(packed_p_hi[10]), "r"(packed_p_hi[11]), "r"(packed_p_hi[12]), "r"(packed_p_hi[13]), "r"(packed_p_hi[14]), "r"(packed_p_hi[15]), "r"(packed_p_hi[16]), "r"(packed_p_hi[17]), "r"(packed_p_hi[18]), "r"(packed_p_hi[19]), "r"(packed_p_hi[20]), "r"(packed_p_hi[21]), "r"(packed_p_hi[22]), "r"(packed_p_hi[23]), "r"(packed_p_hi[24]), "r"(packed_p_hi[25]), "r"(packed_p_hi[26]), "r"(packed_p_hi[27]), "r"(packed_p_hi[28]), "r"(packed_p_hi[29]), "r"(packed_p_hi[30]), "r"(packed_p_hi[31]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_lastsplit_addr);
                    float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[32], _tmem_load_0[33]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[34], _tmem_load_0[35]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[36], _tmem_load_0[37]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[38], _tmem_load_0[39]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[40], _tmem_load_0[41]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[42], _tmem_load_0[43]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[44], _tmem_load_0[45]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[46], _tmem_load_0[47]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[48], _tmem_load_0[49]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[50], _tmem_load_0[51]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[52], _tmem_load_0[53]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[54], _tmem_load_0[55]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[56], _tmem_load_0[57]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[58], _tmem_load_0[59]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[60], _tmem_load_0[61]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[62], _tmem_load_0[63]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[64], _tmem_load_0[65]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[66], _tmem_load_0[67]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[68], _tmem_load_0[69]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[70], _tmem_load_0[71]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[72], _tmem_load_0[73]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[74], _tmem_load_0[75]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[76], _tmem_load_0[77]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[78], _tmem_load_0[79]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[80], _tmem_load_0[81]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[82], _tmem_load_0[83]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[84], _tmem_load_0[85]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[86], _tmem_load_0[87]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[88], _tmem_load_0[89]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[90], _tmem_load_0[91]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[92], _tmem_load_0[93]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[94], _tmem_load_0[95]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[96], _tmem_load_0[97]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[98], _tmem_load_0[99]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[100], _tmem_load_0[101]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[102], _tmem_load_0[103]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[104], _tmem_load_0[105]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[106], _tmem_load_0[107]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[108], _tmem_load_0[109]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[110], _tmem_load_0[111]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[112], _tmem_load_0[113]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[114], _tmem_load_0[115]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[116], _tmem_load_0[117]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[118], _tmem_load_0[119]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[120], _tmem_load_0[121]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[122], _tmem_load_0[123]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[124], _tmem_load_0[125]));
                    _reg_reduce_sum2_20 = add_f32x2_rn_noftz(_reg_reduce_sum2_20, make_float2(_tmem_load_0[126], _tmem_load_0[127]));
                    float block_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                    row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                }
                scale_smem[128 + stat_slot] = row_sum;
                scale_smem[256 + stat_slot] = row_max;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(stats_full_addr);
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                unsigned int base = work_stage_s * 8;
                unsigned int valid_u = work_token_words[base];
                unsigned int q_block_u = work_token_words[base + 1];
                unsigned int head_u = work_token_words[base + 2];
                unsigned int row_begin_u = work_token_words[base + 3];
                unsigned int count_u = work_token_words[base + 4];
                unsigned int groups_u = work_token_words[base + 5];
                unsigned int gate_u = work_token_words[base + 7];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_t = valid_u;
                q_block_t = (int)q_block_u;
                count_t = (int)count_u;
                groups_t = (int)groups_u;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // correction_main
            unsigned int work_stage_c = 0;
            mbarrier_arrive(tile_done_addr);
            int bid_t_1 = blockIdx.x;
            int q_block_t_1 = bid_t_1 % query_blocks;
            int head_t_1 = bid_t_1 / query_blocks;
            int row_begin_t_1 = indptr[q_block_t_1];
            int count_t_1 = indptr[q_block_t_1 + 1] - row_begin_t_1;
            int groups_t_1 = (count_t_1 + 4 - 1) / 4;
            unsigned int valid_t_1 = 1;
            if (bid_t_1 >= total_tiles) {
                valid_t_1 = 0;
            }
            unsigned int gate_t_1 = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_stats_full_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (int _tile_iter_1 = 0; _tile_iter_1 < total_tiles; _tile_iter_1++) {
                if (valid_t_1 == 0) {
                    break;
                }
                int query_base_1 = q_block_t_1 * 64;
                int q_valid_1 = sequence_q - query_base_1;
                if (q_valid_1 > 64) {
                    q_valid_1 = 64;
                }
                if (q_valid_1 < 0) {
                    q_valid_1 = 0;
                }
                const int warp_in_role_1 = warp - 4;
                const int tmem_row_origin_1 = warp_in_role_1 * 32;
                int my_row_1 = warp_in_role_1 % 2 * 32 + lane;
                int stat_slot_1 = warp_in_role_1 * 32 + lane;
                int partner_slot = (warp_in_role_1 ^ 2) * 32 + lane;
                int row_addr = tmem_row_origin_1 << 16;
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                #pragma unroll 1
                for (int _group_index = 1; _group_index < groups_t_1; _group_index++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_1 = scale_smem[stat_slot_1];
                    int _warp_vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                    if (_warp_vote_0 != 0) {
                        float _tmem_load_1[128];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(taddr + 128 + (unsigned int)row_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                            : "r"(taddr + 128 + (unsigned int)row_addr + 32));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95])
                            : "r"(taddr + 128 + (unsigned int)row_addr + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                            : "r"(taddr + 128 + (unsigned int)row_addr + 96));
                        const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 64; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x128.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                            :: "r"(taddr + 128 + (unsigned int)row_addr), "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]), "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]), "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]), "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]), "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]), "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]), "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]), "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]), "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]), "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]), "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]), "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]), "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]), "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]), "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]), "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]), "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]), "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]), "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]), "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]), "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]), "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]), "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]), "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]), "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]), "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]), "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]), "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]), "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]), "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]), "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]), "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]), "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]), "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]), "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]), "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]), "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]), "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]), "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]), "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]), "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]), "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]), "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]), "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]), "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]), "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]), "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]), "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]), "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]), "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]), "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]), "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]), "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]), "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]), "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]), "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]), "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]), "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]), "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]), "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(stats_full_addr, _phase_stats_full_0);
                _phase_stats_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_sum = scale_smem[128 + stat_slot_1];
                float final_max = scale_smem[256 + stat_slot_1];
                float partner_sum = scale_smem[128 + partner_slot];
                float partner_max = scale_smem[256 + partner_slot];
                float max_total;
                asm("max.f32 %0, %1, %2;" : "=f"(max_total) : "f"(final_max), "f"(partner_max));
                float max_total_safe = ((max_total == -CUDART_INF_F) ? 0.0f : max_total);
                float _exp2_2 = approx_exp2((final_max - max_total_safe) * softmax_scale_log2);
                float local_scale = ((final_sum > 0.0f) ? _exp2_2 : 0.0f);
                float _exp2_3 = approx_exp2((partner_max - max_total_safe) * softmax_scale_log2);
                float partner_scale = ((partner_sum > 0.0f) ? _exp2_3 : 0.0f);
                float sum_total = __fmaf_rn(final_sum, local_scale, partner_sum * partner_scale);
                float _rcp_4 = approx_rcp(sum_total);
                float local_weight = ((sum_total > 0.0f) ? local_scale * _rcp_4 : 0.0f);
                float _tmem_load_2[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(taddr + 128 + (unsigned int)row_addr));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(taddr + 128 + (unsigned int)row_addr + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[64]), "=f"(_tmem_load_2[65]), "=f"(_tmem_load_2[66]), "=f"(_tmem_load_2[67]), "=f"(_tmem_load_2[68]), "=f"(_tmem_load_2[69]), "=f"(_tmem_load_2[70]), "=f"(_tmem_load_2[71]), "=f"(_tmem_load_2[72]), "=f"(_tmem_load_2[73]), "=f"(_tmem_load_2[74]), "=f"(_tmem_load_2[75]), "=f"(_tmem_load_2[76]), "=f"(_tmem_load_2[77]), "=f"(_tmem_load_2[78]), "=f"(_tmem_load_2[79]), "=f"(_tmem_load_2[80]), "=f"(_tmem_load_2[81]), "=f"(_tmem_load_2[82]), "=f"(_tmem_load_2[83]), "=f"(_tmem_load_2[84]), "=f"(_tmem_load_2[85]), "=f"(_tmem_load_2[86]), "=f"(_tmem_load_2[87]), "=f"(_tmem_load_2[88]), "=f"(_tmem_load_2[89]), "=f"(_tmem_load_2[90]), "=f"(_tmem_load_2[91]), "=f"(_tmem_load_2[92]), "=f"(_tmem_load_2[93]), "=f"(_tmem_load_2[94]), "=f"(_tmem_load_2[95])
                    : "r"(taddr + 128 + (unsigned int)row_addr + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[96]), "=f"(_tmem_load_2[97]), "=f"(_tmem_load_2[98]), "=f"(_tmem_load_2[99]), "=f"(_tmem_load_2[100]), "=f"(_tmem_load_2[101]), "=f"(_tmem_load_2[102]), "=f"(_tmem_load_2[103]), "=f"(_tmem_load_2[104]), "=f"(_tmem_load_2[105]), "=f"(_tmem_load_2[106]), "=f"(_tmem_load_2[107]), "=f"(_tmem_load_2[108]), "=f"(_tmem_load_2[109]), "=f"(_tmem_load_2[110]), "=f"(_tmem_load_2[111]), "=f"(_tmem_load_2[112]), "=f"(_tmem_load_2[113]), "=f"(_tmem_load_2[114]), "=f"(_tmem_load_2[115]), "=f"(_tmem_load_2[116]), "=f"(_tmem_load_2[117]), "=f"(_tmem_load_2[118]), "=f"(_tmem_load_2[119]), "=f"(_tmem_load_2[120]), "=f"(_tmem_load_2[121]), "=f"(_tmem_load_2[122]), "=f"(_tmem_load_2[123]), "=f"(_tmem_load_2[124]), "=f"(_tmem_load_2[125]), "=f"(_tmem_load_2[126]), "=f"(_tmem_load_2[127])
                    : "r"(taddr + 128 + (unsigned int)row_addr + 96));
                const float2 _scale2_5 = {local_weight, local_weight};
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_5);
                int query = query_base_1 + my_row_1;
                int output_row = (query * num_heads + head_t_1) * 128;
                if (warp_in_role_1 >= 2) {
                    unsigned int _tmem_load_2_bf16_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                        _tmem_load_2_bf16_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 0) * 16)), "r"(_tmem_load_2_bf16_0[0]), "r"(_tmem_load_2_bf16_0[1]), "r"(_tmem_load_2_bf16_0[2]), "r"(_tmem_load_2_bf16_0[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 8], _tmem_load_2[_lp*2+1 + 8]));
                        _tmem_load_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 1) * 16)), "r"(_tmem_load_2_bf16_1[0]), "r"(_tmem_load_2_bf16_1[1]), "r"(_tmem_load_2_bf16_1[2]), "r"(_tmem_load_2_bf16_1[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 16], _tmem_load_2[_lp*2+1 + 16]));
                        _tmem_load_2_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 2) * 16)), "r"(_tmem_load_2_bf16_2[0]), "r"(_tmem_load_2_bf16_2[1]), "r"(_tmem_load_2_bf16_2[2]), "r"(_tmem_load_2_bf16_2[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 24], _tmem_load_2[_lp*2+1 + 24]));
                        _tmem_load_2_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 3) * 16)), "r"(_tmem_load_2_bf16_3[0]), "r"(_tmem_load_2_bf16_3[1]), "r"(_tmem_load_2_bf16_3[2]), "r"(_tmem_load_2_bf16_3[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 32], _tmem_load_2[_lp*2+1 + 32]));
                        _tmem_load_2_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 4) * 16)), "r"(_tmem_load_2_bf16_4[0]), "r"(_tmem_load_2_bf16_4[1]), "r"(_tmem_load_2_bf16_4[2]), "r"(_tmem_load_2_bf16_4[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_5[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 40], _tmem_load_2[_lp*2+1 + 40]));
                        _tmem_load_2_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 5) * 16)), "r"(_tmem_load_2_bf16_5[0]), "r"(_tmem_load_2_bf16_5[1]), "r"(_tmem_load_2_bf16_5[2]), "r"(_tmem_load_2_bf16_5[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_6[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 48], _tmem_load_2[_lp*2+1 + 48]));
                        _tmem_load_2_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 6) * 16)), "r"(_tmem_load_2_bf16_6[0]), "r"(_tmem_load_2_bf16_6[1]), "r"(_tmem_load_2_bf16_6[2]), "r"(_tmem_load_2_bf16_6[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_7[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 56], _tmem_load_2[_lp*2+1 + 56]));
                        _tmem_load_2_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 7) * 16)), "r"(_tmem_load_2_bf16_7[0]), "r"(_tmem_load_2_bf16_7[1]), "r"(_tmem_load_2_bf16_7[2]), "r"(_tmem_load_2_bf16_7[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_8[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 64], _tmem_load_2[_lp*2+1 + 64]));
                        _tmem_load_2_bf16_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 8) * 16)), "r"(_tmem_load_2_bf16_8[0]), "r"(_tmem_load_2_bf16_8[1]), "r"(_tmem_load_2_bf16_8[2]), "r"(_tmem_load_2_bf16_8[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_9[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 72], _tmem_load_2[_lp*2+1 + 72]));
                        _tmem_load_2_bf16_9[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 9) * 16)), "r"(_tmem_load_2_bf16_9[0]), "r"(_tmem_load_2_bf16_9[1]), "r"(_tmem_load_2_bf16_9[2]), "r"(_tmem_load_2_bf16_9[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_10[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 80], _tmem_load_2[_lp*2+1 + 80]));
                        _tmem_load_2_bf16_10[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 10) * 16)), "r"(_tmem_load_2_bf16_10[0]), "r"(_tmem_load_2_bf16_10[1]), "r"(_tmem_load_2_bf16_10[2]), "r"(_tmem_load_2_bf16_10[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_11[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 88], _tmem_load_2[_lp*2+1 + 88]));
                        _tmem_load_2_bf16_11[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 11) * 16)), "r"(_tmem_load_2_bf16_11[0]), "r"(_tmem_load_2_bf16_11[1]), "r"(_tmem_load_2_bf16_11[2]), "r"(_tmem_load_2_bf16_11[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_12[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 96], _tmem_load_2[_lp*2+1 + 96]));
                        _tmem_load_2_bf16_12[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 12) * 16)), "r"(_tmem_load_2_bf16_12[0]), "r"(_tmem_load_2_bf16_12[1]), "r"(_tmem_load_2_bf16_12[2]), "r"(_tmem_load_2_bf16_12[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_13[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 104], _tmem_load_2[_lp*2+1 + 104]));
                        _tmem_load_2_bf16_13[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 13) * 16)), "r"(_tmem_load_2_bf16_13[0]), "r"(_tmem_load_2_bf16_13[1]), "r"(_tmem_load_2_bf16_13[2]), "r"(_tmem_load_2_bf16_13[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_14[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 112], _tmem_load_2[_lp*2+1 + 112]));
                        _tmem_load_2_bf16_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 14) * 16)), "r"(_tmem_load_2_bf16_14[0]), "r"(_tmem_load_2_bf16_14[1]), "r"(_tmem_load_2_bf16_14[2]), "r"(_tmem_load_2_bf16_14[3]) : "memory");
                    unsigned int _tmem_load_2_bf16_15[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 120], _tmem_load_2[_lp*2+1 + 120]));
                        _tmem_load_2_bf16_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 15) * 16)), "r"(_tmem_load_2_bf16_15[0]), "r"(_tmem_load_2_bf16_15[1]), "r"(_tmem_load_2_bf16_15[2]), "r"(_tmem_load_2_bf16_15[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp_in_role_1 < 2 && my_row_1 < q_valid_1) {
                    #pragma unroll
                    for (int offset = 0; offset < 128; offset += 8) {
                        unsigned int smem_regs[4];
                        #pragma unroll
                        for (int _lr = 0; _lr < 4; _lr++) {
                            smem_regs[_lr] = partial_smem[my_row_1 * 64 + (my_row_1 & 7 ^ offset / 8) * 4 + _lr];
                        }
                        float _unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(smem_regs[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(smem_regs[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int elem = 0; elem < 8; elem++) {
                            _tmem_load_2[offset + elem] = _tmem_load_2[offset + elem] + _unpack_bf16_0[elem];
                        }
                        unsigned int _packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_2 + offset)[_lp*2 + 0], (_tmem_load_2 + offset)[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_2 + offset)[_lp*2 + 2], (_tmem_load_2 + offset)[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_2 + offset)[_lp*2 + 4], (_tmem_load_2 + offset)[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_2 + offset)[_lp*2 + 6], (_tmem_load_2 + offset)[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(out + (output_row + offset))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    }
                    if (return_lse != 0) {
                        int stat_idx = query * num_heads + head_t_1;
                        float _log2_6;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(sum_total));
                        lse[stat_idx] = ((sum_total > 0.0f) ? __fmaf_rn(max_total_safe * softmax_scale_log2, 0.6931472f, _log2_6 * 0.6931472f) : -CUDART_INF_F);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(tile_done_addr);
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_1 = work_stage_c * 8;
                unsigned int valid_u_1 = work_token_words[base_1];
                unsigned int q_block_u_1 = work_token_words[base_1 + 1];
                unsigned int head_u_1 = work_token_words[base_1 + 2];
                unsigned int row_begin_u_1 = work_token_words[base_1 + 3];
                unsigned int count_u_1 = work_token_words[base_1 + 4];
                unsigned int groups_u_1 = work_token_words[base_1 + 5];
                unsigned int gate_u_1 = work_token_words[base_1 + 7];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_t_1 = valid_u_1;
                q_block_t_1 = (int)q_block_u_1;
                head_t_1 = (int)head_u_1;
                groups_t_1 = (int)groups_u_1;
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int kv_stage = 0;
            unsigned int work_stage_m = 0;
            int bid_t_2 = blockIdx.x;
            int q_block_t_2 = bid_t_2 % query_blocks;
            int head_t_2 = bid_t_2 / query_blocks;
            int row_begin_t_2 = indptr[q_block_t_2];
            int count_t_2 = indptr[q_block_t_2 + 1] - row_begin_t_2;
            int groups_t_2 = (count_t_2 + 4 - 1) / 4;
            unsigned int valid_t_2 = 1;
            if (bid_t_2 >= total_tiles) {
                valid_t_2 = 0;
            }
            unsigned int gate_t_2 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_tile_done_0 = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_lastsplit_0 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (int _tile_iter_2 = 0; _tile_iter_2 < total_tiles; _tile_iter_2++) {
                if (valid_t_2 == 0) {
                    break;
                }
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                int first_pv = 1;
                unsigned int k_stage = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0) : "memory");
                if (groups_t_2 == 1) {
                    elect_commit2(s_full_addr, q_empty_addr);
                } else {
                    elect_commit(s_full_addr);
                }
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                _phase_tile_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int group_index_1 = 0; group_index_1 < groups_t_2 - 1; group_index_1++) {
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                    k_stage = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores), "r"(0) : "memory");
                    if (group_index_1 + 2 == groups_t_2) {
                        elect_commit2(s_full_addr, q_empty_addr);
                    } else {
                        elect_commit(s_full_addr);
                    }
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; _phase_kv_full ^= 1; }
                    unsigned int v_stage = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_2), "r"(tmem_probabilities + 0), "r"(((first_pv) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                    _phase_p_lastsplit_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_3), "r"(tmem_probabilities + 0), "r"(1) : "memory");
                    first_pv = 0;
                    elect_commit(p_empty_addr);
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; _phase_kv_full ^= 1; }
                }
                unsigned int final_v_stage = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_4 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_4), "r"(tmem_probabilities + 0), "r"(((first_pv) ? 0 : 1)) : "memory");
                mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                _phase_p_lastsplit_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_5 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_5), "r"(tmem_probabilities + 0), "r"(1) : "memory");
                elect_commit2(o_full_addr, p_empty_addr);
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                _phase_s_empty_0 ^= 1;
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_2 = work_stage_m * 8;
                unsigned int valid_u_2 = work_token_words[base_2];
                unsigned int q_block_u_2 = work_token_words[base_2 + 1];
                unsigned int head_u_2 = work_token_words[base_2 + 2];
                unsigned int row_begin_u_2 = work_token_words[base_2 + 3];
                unsigned int count_u_2 = work_token_words[base_2 + 4];
                unsigned int groups_u_2 = work_token_words[base_2 + 5];
                unsigned int gate_u_2 = work_token_words[base_2 + 7];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_t_2 = valid_u_2;
                groups_t_2 = (int)groups_u_2;
            }
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            unsigned int load_stage = 0;
            unsigned int work_stage_l = 0;
            int bid_t_3 = blockIdx.x;
            int q_block_t_3 = bid_t_3 % query_blocks;
            int head_t_3 = bid_t_3 / query_blocks;
            int row_begin_t_3 = indptr[q_block_t_3];
            int count_t_3 = indptr[q_block_t_3 + 1] - row_begin_t_3;
            int groups_t_3 = (count_t_3 + 4 - 1) / 4;
            unsigned int valid_t_3 = 1;
            if (bid_t_3 >= total_tiles) {
                valid_t_3 = 0;
            }
            unsigned int gate_t_3 = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (int tile_iter = 0; tile_iter < total_tiles; tile_iter++) {
                if (valid_t_3 == 0) {
                    break;
                }
                int query_base_2 = q_block_t_3 * 64;
                int q_valid_2 = sequence_q - query_base_2;
                if (q_valid_2 > 64) {
                    q_valid_2 = 64;
                }
                if (q_valid_2 < 0) {
                    q_valid_2 = 0;
                }
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    tma_4d_gmem2smem(q_smem_addr, (&q), 0, head_t_3, query_base_2, 0, q_full_addr);
                }
                int entry0 = 0;
                int entry1 = entry0 + 1;
                int entry2 = entry0 + 2;
                int entry3 = entry0 + 3;
                int block0 = indices[row_begin_t_3 + entry0];
                int block1 = block0;
                if (entry1 < count_t_3) {
                    block1 = indices[row_begin_t_3 + entry1];
                }
                int block2 = block1;
                if (entry2 < count_t_3) {
                    block2 = indices[row_begin_t_3 + entry2];
                }
                int block3 = block2;
                if (entry3 < count_t_3) {
                    block3 = indices[row_begin_t_3 + entry3];
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                if (groups_t_3 > 1) {
                    int entry0_0 = 4;
                    int entry1_0 = entry0_0 + 1;
                    int entry2_0 = entry0_0 + 2;
                    int entry3_0 = entry0_0 + 3;
                    int block0_0 = indices[row_begin_t_3 + entry0_0];
                    int block1_0 = block0_0;
                    if (entry1_0 < count_t_3) {
                        block1_0 = indices[row_begin_t_3 + entry1_0];
                    }
                    int block2_0 = block1_0;
                    if (entry2_0 < count_t_3) {
                        block2_0 = indices[row_begin_t_3 + entry2_0];
                    }
                    int block3_0 = block2_0;
                    if (entry3_0 < count_t_3) {
                        block3_0 = indices[row_begin_t_3 + entry3_0];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0_0 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0_0 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1_0 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1_0 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2_0 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2_0 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3_0 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3_0 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    #pragma unroll 1
                    for (int group_index_2 = 0; group_index_2 < groups_t_3 - 2; group_index_2++) {
                        int entry0_1 = group_index_2 * 4;
                        int entry1_1 = entry0_1 + 1;
                        int entry2_1 = entry0_1 + 2;
                        int entry3_1 = entry0_1 + 3;
                        int block0_1 = indices[row_begin_t_3 + entry0_1];
                        int block1_1 = block0_1;
                        if (entry1_1 < count_t_3) {
                            block1_1 = indices[row_begin_t_3 + entry1_1];
                        }
                        int block2_1 = block1_1;
                        if (entry2_1 < count_t_3) {
                            block2_1 = indices[row_begin_t_3 + entry2_1];
                        }
                        int block3_1 = block2_1;
                        if (entry3_1 < count_t_3) {
                            block3_1 = indices[row_begin_t_3 + entry3_1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_1 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_1 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_1 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_1 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_1 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_1 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_1 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_1 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        int entry0_2 = (group_index_2 + 2) * 4;
                        int entry1_2 = entry0_2 + 1;
                        int entry2_2 = entry0_2 + 2;
                        int entry3_2 = entry0_2 + 3;
                        int block0_2 = indices[row_begin_t_3 + entry0_2];
                        int block1_2 = block0_2;
                        if (entry1_2 < count_t_3) {
                            block1_2 = indices[row_begin_t_3 + entry1_2];
                        }
                        int block2_2 = block1_2;
                        if (entry2_2 < count_t_3) {
                            block2_2 = indices[row_begin_t_3 + entry2_2];
                        }
                        int block3_2 = block2_2;
                        if (entry3_2 < count_t_3) {
                            block3_2 = indices[row_begin_t_3 + entry3_2];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0_2 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0_2 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1_2 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1_2 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2_2 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2_2 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3_2 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3_2 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                    int entry0_3 = (groups_t_3 - 2) * 4;
                    int entry1_3 = entry0_3 + 1;
                    int entry2_3 = entry0_3 + 2;
                    int entry3_3 = entry0_3 + 3;
                    int block0_3 = indices[row_begin_t_3 + entry0_3];
                    int block1_3 = block0_3;
                    if (entry1_3 < count_t_3) {
                        block1_3 = indices[row_begin_t_3 + entry1_3];
                    }
                    int block2_3 = block1_3;
                    if (entry2_3 < count_t_3) {
                        block2_3 = indices[row_begin_t_3 + entry2_3];
                    }
                    int block3_3 = block2_3;
                    if (entry3_3 < count_t_3) {
                        block3_3 = indices[row_begin_t_3 + entry3_3];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_3 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_3 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_3 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_3 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_3 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_3 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_3 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_3 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    if (gate_t_3 != 0) {
                        if (lane == 0) {
                            mbarrier_arrive(claim_gate_addr);
                        }
                    }
                    int entry0_4 = (groups_t_3 - 1) * 4;
                    int entry1_4 = entry0_4 + 1;
                    int entry2_4 = entry0_4 + 2;
                    int entry3_4 = entry0_4 + 3;
                    int block0_4 = indices[row_begin_t_3 + entry0_4];
                    int block1_4 = block0_4;
                    if (entry1_4 < count_t_3) {
                        block1_4 = indices[row_begin_t_3 + entry1_4];
                    }
                    int block2_4 = block1_4;
                    if (entry2_4 < count_t_3) {
                        block2_4 = indices[row_begin_t_3 + entry2_4];
                    }
                    int block3_4 = block2_4;
                    if (entry3_4 < count_t_3) {
                        block3_4 = indices[row_begin_t_3 + entry3_4];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_4 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_4 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_4 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_4 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_4 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_4 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_4 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_4 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                } else {
                    if (gate_t_3 != 0) {
                        if (lane == 0) {
                            mbarrier_arrive(claim_gate_addr);
                        }
                    }
                    int entry0_5 = 0;
                    int entry1_5 = entry0_5 + 1;
                    int entry2_5 = entry0_5 + 2;
                    int entry3_5 = entry0_5 + 3;
                    int block0_5 = indices[row_begin_t_3 + entry0_5];
                    int block1_5 = block0_5;
                    if (entry1_5 < count_t_3) {
                        block1_5 = indices[row_begin_t_3 + entry1_5];
                    }
                    int block2_5 = block1_5;
                    if (entry2_5 < count_t_3) {
                        block2_5 = indices[row_begin_t_3 + entry2_5];
                    }
                    int block3_5 = block2_5;
                    if (entry3_5 < count_t_3) {
                        block3_5 = indices[row_begin_t_3 + entry3_5];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_5 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_5 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_5 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_5 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_5 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_5 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_5 * 64, 0, head_t_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_5 * 64, 1, head_t_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_3);
                unsigned int base_3 = work_stage_l * 8;
                unsigned int valid_u_3 = work_token_words[base_3];
                unsigned int q_block_u_3 = work_token_words[base_3 + 1];
                unsigned int head_u_3 = work_token_words[base_3 + 2];
                unsigned int row_begin_u_3 = work_token_words[base_3 + 3];
                unsigned int count_u_3 = work_token_words[base_3 + 4];
                unsigned int groups_u_3 = work_token_words[base_3 + 5];
                unsigned int gate_u_3 = work_token_words[base_3 + 7];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_3 ^= 1; }
                valid_t_3 = valid_u_3;
                q_block_t_3 = (int)q_block_u_3;
                head_t_3 = (int)head_u_3;
                row_begin_t_3 = (int)row_begin_u_3;
                count_t_3 = (int)count_u_3;
                groups_t_3 = (int)groups_u_3;
                gate_t_3 = gate_u_3;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            int lane_1 = lane;
            int grid_i = gridDim.x;
            int bid_1 = blockIdx.x;
            unsigned int work_k = 0;
            hist_smem[lane_1] = 0;
            hist_smem[32 + lane_1] = 0;
            __syncwarp();
            int n_scan = (query_blocks + 31) / 32;
            int carry = indptr[0];
            int v_reg[8];
            v_reg[0] = 0;
            int qb_v_0 = lane_1;
            if (qb_v_0 < query_blocks) {
                v_reg[0] = indptr[qb_v_0 + 1];
            }
            v_reg[1] = 0;
            int qb_v_1 = 32 + lane_1;
            if (qb_v_1 < query_blocks) {
                v_reg[1] = indptr[qb_v_1 + 1];
            }
            v_reg[2] = 0;
            int qb_v_2 = 64 + lane_1;
            if (qb_v_2 < query_blocks) {
                v_reg[2] = indptr[qb_v_2 + 1];
            }
            v_reg[3] = 0;
            int qb_v_3 = 96 + lane_1;
            if (qb_v_3 < query_blocks) {
                v_reg[3] = indptr[qb_v_3 + 1];
            }
            v_reg[4] = 0;
            int qb_v_4 = 128 + lane_1;
            if (qb_v_4 < query_blocks) {
                v_reg[4] = indptr[qb_v_4 + 1];
            }
            v_reg[5] = 0;
            int qb_v_5 = 160 + lane_1;
            if (qb_v_5 < query_blocks) {
                v_reg[5] = indptr[qb_v_5 + 1];
            }
            v_reg[6] = 0;
            int qb_v_6 = 192 + lane_1;
            if (qb_v_6 < query_blocks) {
                v_reg[6] = indptr[qb_v_6 + 1];
            }
            v_reg[7] = 0;
            int qb_v_7 = 224 + lane_1;
            if (qb_v_7 < query_blocks) {
                v_reg[7] = indptr[qb_v_7 + 1];
            }
            int grp_reg[8];
            grp_reg[0] = -1;
            int prev_h_0 = __shfl_up_sync(0xFFFFFFFF, v_reg[0], 1, 32);
            if (lane_1 == 0) {
                prev_h_0 = carry;
            }
            int qb_h_0 = lane_1;
            if (qb_h_0 < query_blocks) {
                int grp = (v_reg[0] - prev_h_0 + 4 - 1) / 4;
                if (grp > 63) {
                    grp = 63;
                }
                int grp_h_0 = grp;
                grp_reg[0] = grp_h_0;
                atomicAdd(&hist_smem[grp_h_0], 1);
            }
            int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, v_reg[0], 31);
            carry = _shfl_sync_0;
            grp_reg[1] = -1;
            int prev_h_1 = __shfl_up_sync(0xFFFFFFFF, v_reg[1], 1, 32);
            if (lane_1 == 0) {
                prev_h_1 = carry;
            }
            int qb_h_1 = 32 + lane_1;
            if (qb_h_1 < query_blocks) {
                int grp_1 = (v_reg[1] - prev_h_1 + 4 - 1) / 4;
                if (grp_1 > 63) {
                    grp_1 = 63;
                }
                int grp_h_1 = grp_1;
                grp_reg[1] = grp_h_1;
                atomicAdd(&hist_smem[grp_h_1], 1);
            }
            int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, v_reg[1], 31);
            carry = _shfl_sync_1;
            grp_reg[2] = -1;
            int prev_h_2 = __shfl_up_sync(0xFFFFFFFF, v_reg[2], 1, 32);
            if (lane_1 == 0) {
                prev_h_2 = carry;
            }
            int qb_h_2 = 64 + lane_1;
            if (qb_h_2 < query_blocks) {
                int grp_2 = (v_reg[2] - prev_h_2 + 4 - 1) / 4;
                if (grp_2 > 63) {
                    grp_2 = 63;
                }
                int grp_h_2 = grp_2;
                grp_reg[2] = grp_h_2;
                atomicAdd(&hist_smem[grp_h_2], 1);
            }
            int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, v_reg[2], 31);
            carry = _shfl_sync_2;
            grp_reg[3] = -1;
            int prev_h_3 = __shfl_up_sync(0xFFFFFFFF, v_reg[3], 1, 32);
            if (lane_1 == 0) {
                prev_h_3 = carry;
            }
            int qb_h_3 = 96 + lane_1;
            if (qb_h_3 < query_blocks) {
                int grp_3 = (v_reg[3] - prev_h_3 + 4 - 1) / 4;
                if (grp_3 > 63) {
                    grp_3 = 63;
                }
                int grp_h_3 = grp_3;
                grp_reg[3] = grp_h_3;
                atomicAdd(&hist_smem[grp_h_3], 1);
            }
            int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, v_reg[3], 31);
            carry = _shfl_sync_3;
            grp_reg[4] = -1;
            int prev_h_4 = __shfl_up_sync(0xFFFFFFFF, v_reg[4], 1, 32);
            if (lane_1 == 0) {
                prev_h_4 = carry;
            }
            int qb_h_4 = 128 + lane_1;
            if (qb_h_4 < query_blocks) {
                int grp_4 = (v_reg[4] - prev_h_4 + 4 - 1) / 4;
                if (grp_4 > 63) {
                    grp_4 = 63;
                }
                int grp_h_4 = grp_4;
                grp_reg[4] = grp_h_4;
                atomicAdd(&hist_smem[grp_h_4], 1);
            }
            int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, v_reg[4], 31);
            carry = _shfl_sync_4;
            grp_reg[5] = -1;
            int prev_h_5 = __shfl_up_sync(0xFFFFFFFF, v_reg[5], 1, 32);
            if (lane_1 == 0) {
                prev_h_5 = carry;
            }
            int qb_h_5 = 160 + lane_1;
            if (qb_h_5 < query_blocks) {
                int grp_5 = (v_reg[5] - prev_h_5 + 4 - 1) / 4;
                if (grp_5 > 63) {
                    grp_5 = 63;
                }
                int grp_h_5 = grp_5;
                grp_reg[5] = grp_h_5;
                atomicAdd(&hist_smem[grp_h_5], 1);
            }
            int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, v_reg[5], 31);
            carry = _shfl_sync_5;
            grp_reg[6] = -1;
            int prev_h_6 = __shfl_up_sync(0xFFFFFFFF, v_reg[6], 1, 32);
            if (lane_1 == 0) {
                prev_h_6 = carry;
            }
            int qb_h_6 = 192 + lane_1;
            if (qb_h_6 < query_blocks) {
                int grp_6 = (v_reg[6] - prev_h_6 + 4 - 1) / 4;
                if (grp_6 > 63) {
                    grp_6 = 63;
                }
                int grp_h_6 = grp_6;
                grp_reg[6] = grp_h_6;
                atomicAdd(&hist_smem[grp_h_6], 1);
            }
            int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, v_reg[6], 31);
            carry = _shfl_sync_6;
            grp_reg[7] = -1;
            int prev_h_7 = __shfl_up_sync(0xFFFFFFFF, v_reg[7], 1, 32);
            if (lane_1 == 0) {
                prev_h_7 = carry;
            }
            int qb_h_7 = 224 + lane_1;
            if (qb_h_7 < query_blocks) {
                int grp_7 = (v_reg[7] - prev_h_7 + 4 - 1) / 4;
                if (grp_7 > 63) {
                    grp_7 = 63;
                }
                int grp_h_7 = grp_7;
                grp_reg[7] = grp_h_7;
                atomicAdd(&hist_smem[grp_h_7], 1);
            }
            int _shfl_sync_7 = __shfl_sync(0xFFFFFFFF, v_reg[7], 31);
            carry = _shfl_sync_7;
            #pragma unroll 1
            for (int hc = 8; hc < n_scan; hc += 8) {
                int v2[8];
                v2[0] = 0;
                int qb_c_0 = hc * 32 + lane_1;
                if (qb_c_0 < query_blocks) {
                    v2[0] = indptr[qb_c_0 + 1];
                }
                v2[1] = 0;
                int qb_c_1 = (hc + 1) * 32 + lane_1;
                if (qb_c_1 < query_blocks) {
                    v2[1] = indptr[qb_c_1 + 1];
                }
                v2[2] = 0;
                int qb_c_2 = (hc + 2) * 32 + lane_1;
                if (qb_c_2 < query_blocks) {
                    v2[2] = indptr[qb_c_2 + 1];
                }
                v2[3] = 0;
                int qb_c_3 = (hc + 3) * 32 + lane_1;
                if (qb_c_3 < query_blocks) {
                    v2[3] = indptr[qb_c_3 + 1];
                }
                v2[4] = 0;
                int qb_c_4 = (hc + 4) * 32 + lane_1;
                if (qb_c_4 < query_blocks) {
                    v2[4] = indptr[qb_c_4 + 1];
                }
                v2[5] = 0;
                int qb_c_5 = (hc + 5) * 32 + lane_1;
                if (qb_c_5 < query_blocks) {
                    v2[5] = indptr[qb_c_5 + 1];
                }
                v2[6] = 0;
                int qb_c_6 = (hc + 6) * 32 + lane_1;
                if (qb_c_6 < query_blocks) {
                    v2[6] = indptr[qb_c_6 + 1];
                }
                v2[7] = 0;
                int qb_c_7 = (hc + 7) * 32 + lane_1;
                if (qb_c_7 < query_blocks) {
                    v2[7] = indptr[qb_c_7 + 1];
                }
                int prev_c_0 = __shfl_up_sync(0xFFFFFFFF, v2[0], 1, 32);
                if (lane_1 == 0) {
                    prev_c_0 = carry;
                }
                int qb_c2_0 = hc * 32 + lane_1;
                if (qb_c2_0 < query_blocks) {
                    int grp_8 = (v2[0] - prev_c_0 + 4 - 1) / 4;
                    if (grp_8 > 63) {
                        grp_8 = 63;
                    }
                    atomicAdd(&hist_smem[grp_8], 1);
                }
                int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, v2[0], 31);
                carry = _shfl_sync_8;
                int prev_c_1 = __shfl_up_sync(0xFFFFFFFF, v2[1], 1, 32);
                if (lane_1 == 0) {
                    prev_c_1 = carry;
                }
                int qb_c2_1 = (hc + 1) * 32 + lane_1;
                if (qb_c2_1 < query_blocks) {
                    int grp_9 = (v2[1] - prev_c_1 + 4 - 1) / 4;
                    if (grp_9 > 63) {
                        grp_9 = 63;
                    }
                    atomicAdd(&hist_smem[grp_9], 1);
                }
                int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, v2[1], 31);
                carry = _shfl_sync_9;
                int prev_c_2 = __shfl_up_sync(0xFFFFFFFF, v2[2], 1, 32);
                if (lane_1 == 0) {
                    prev_c_2 = carry;
                }
                int qb_c2_2 = (hc + 2) * 32 + lane_1;
                if (qb_c2_2 < query_blocks) {
                    int grp_10 = (v2[2] - prev_c_2 + 4 - 1) / 4;
                    if (grp_10 > 63) {
                        grp_10 = 63;
                    }
                    atomicAdd(&hist_smem[grp_10], 1);
                }
                int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, v2[2], 31);
                carry = _shfl_sync_10;
                int prev_c_3 = __shfl_up_sync(0xFFFFFFFF, v2[3], 1, 32);
                if (lane_1 == 0) {
                    prev_c_3 = carry;
                }
                int qb_c2_3 = (hc + 3) * 32 + lane_1;
                if (qb_c2_3 < query_blocks) {
                    int grp_11 = (v2[3] - prev_c_3 + 4 - 1) / 4;
                    if (grp_11 > 63) {
                        grp_11 = 63;
                    }
                    atomicAdd(&hist_smem[grp_11], 1);
                }
                int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, v2[3], 31);
                carry = _shfl_sync_11;
                int prev_c_4 = __shfl_up_sync(0xFFFFFFFF, v2[4], 1, 32);
                if (lane_1 == 0) {
                    prev_c_4 = carry;
                }
                int qb_c2_4 = (hc + 4) * 32 + lane_1;
                if (qb_c2_4 < query_blocks) {
                    int grp_12 = (v2[4] - prev_c_4 + 4 - 1) / 4;
                    if (grp_12 > 63) {
                        grp_12 = 63;
                    }
                    atomicAdd(&hist_smem[grp_12], 1);
                }
                int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, v2[4], 31);
                carry = _shfl_sync_12;
                int prev_c_5 = __shfl_up_sync(0xFFFFFFFF, v2[5], 1, 32);
                if (lane_1 == 0) {
                    prev_c_5 = carry;
                }
                int qb_c2_5 = (hc + 5) * 32 + lane_1;
                if (qb_c2_5 < query_blocks) {
                    int grp_13 = (v2[5] - prev_c_5 + 4 - 1) / 4;
                    if (grp_13 > 63) {
                        grp_13 = 63;
                    }
                    atomicAdd(&hist_smem[grp_13], 1);
                }
                int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, v2[5], 31);
                carry = _shfl_sync_13;
                int prev_c_6 = __shfl_up_sync(0xFFFFFFFF, v2[6], 1, 32);
                if (lane_1 == 0) {
                    prev_c_6 = carry;
                }
                int qb_c2_6 = (hc + 6) * 32 + lane_1;
                if (qb_c2_6 < query_blocks) {
                    int grp_14 = (v2[6] - prev_c_6 + 4 - 1) / 4;
                    if (grp_14 > 63) {
                        grp_14 = 63;
                    }
                    atomicAdd(&hist_smem[grp_14], 1);
                }
                int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, v2[6], 31);
                carry = _shfl_sync_14;
                int prev_c_7 = __shfl_up_sync(0xFFFFFFFF, v2[7], 1, 32);
                if (lane_1 == 0) {
                    prev_c_7 = carry;
                }
                int qb_c2_7 = (hc + 7) * 32 + lane_1;
                if (qb_c2_7 < query_blocks) {
                    int grp_15 = (v2[7] - prev_c_7 + 4 - 1) / 4;
                    if (grp_15 > 63) {
                        grp_15 = 63;
                    }
                    atomicAdd(&hist_smem[grp_15], 1);
                }
                int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, v2[7], 31);
                carry = _shfl_sync_15;
            }
            __syncwarp();
            unsigned int h0 = hist_smem[lane_1 * 2];
            unsigned int h1 = hist_smem[lane_1 * 2 + 1];
            unsigned int lane_sum = h0 + h1;
            unsigned int lane_max = ((h0) > (h1) ? (h0) : (h1));
            unsigned int uniform_mask = __ballot_sync(0xFFFFFFFF, lane_max == (unsigned int)query_blocks);
            unsigned int uniform_sel = ((uniform_mask != 0) ? 1 : 0);
            unsigned int heads_u = (unsigned int)num_heads;
            uint32_t suffix_incl = lane_sum;
            { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, suffix_incl, 1, 32); if (lane + 1 < 32) { suffix_incl += peer; } }
            { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, suffix_incl, 2, 32); if (lane + 2 < 32) { suffix_incl += peer; } }
            { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, suffix_incl, 4, 32); if (lane + 4 < 32) { suffix_incl += peer; } }
            { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, suffix_incl, 8, 32); if (lane + 8 < 32) { suffix_incl += peer; } }
            { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, suffix_incl, 16, 32); if (lane + 16 < 32) { suffix_incl += peer; } }
            unsigned int excl = suffix_incl - lane_sum;
            unsigned int p1 = excl * heads_u;
            unsigned int p0 = (excl + h1) * heads_u;
            unsigned int n0 = h0 * heads_u;
            unsigned int n1 = h1 * heads_u;
            unsigned int gate_flag = 1;
            if (uniform_sel != 0) {
                gate_flag = 0;
            }
            if (total_tiles <= grid_i) {
                gate_flag = 0;
            }
            unsigned int below_mask = (1 << (unsigned int)lane_1) - 1;
            int last_pos = total_tiles + grid_i - 1;
            int n_pub = 1;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_claim_gate_0 = 0;
            #pragma unroll 1
            for (int _claim_i = 0; _claim_i < total_tiles + 1; _claim_i++) {
                mbarrier_wait_hint(work_empty_addr + (work_k) * 8, _phase_work_empty, 1000);
                if (gate_flag != 0) {
                    mbarrier_wait_hint(claim_gate_addr, _phase_claim_gate_0, 1000);
                }
                _phase_claim_gate_0 ^= 1;
                unsigned int ticket = (unsigned int)total_tiles;
                unsigned int valid_tok = 0;
                int tile_sel = 0;
                #pragma unroll 1
                for (int _draw_i = 0; _draw_i < total_tiles + grid_i; _draw_i++) {
                    unsigned int ticket_lane0 = (unsigned int)total_tiles;
                    if (total_tiles > grid_i) {
                        if (uniform_sel != 0) {
                            ticket_lane0 = (unsigned int)(bid_1 + n_pub * grid_i);
                        } else if (lane_1 == 0) {
                            unsigned int _atomic_fetch_add_16 = atomicAdd(&queue_counters[0], 1);
                            ticket_lane0 = _atomic_fetch_add_16;
                        }
                    }
                    unsigned int draw = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                    if (draw == (unsigned int)last_pos) {
                        if (lane_1 == 0) {
                            *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                        }
                    }
                    unsigned int valid_d = ((draw < (unsigned int)total_tiles) ? 1 : 0);
                    int hit = -1;
                    unsigned int p_sel = p0;
                    unsigned int n_sel = h0;
                    if (draw >= p0 && draw < p0 + n0) {
                        hit = lane_1 * 2;
                    }
                    if (draw >= p1 && draw < p1 + n1) {
                        hit = lane_1 * 2 + 1;
                        p_sel = p1;
                        n_sel = h1;
                    }
                    unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, hit >= 0);
                    int _find_first_set_17 = __ffs(hit_mask);
                    int hit_lane = ((_find_first_set_17 - 1) > (0) ? (_find_first_set_17 - 1) : (0));
                    int bucket = __shfl_sync(0xFFFFFFFF, hit, hit_lane);
                    unsigned int p_b = __shfl_sync(0xFFFFFFFF, p_sel, hit_lane);
                    unsigned int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, n_sel, hit_lane);
                    unsigned int n_b = ((_shfl_sync_18) > (1) ? (_shfl_sync_18) : (1));
                    unsigned int r_t = draw - p_b;
                    unsigned int head_sel = r_t / n_b;
                    unsigned int j_sel = r_t - head_sel * n_b;
                    unsigned int running = 0;
                    int q_found = 0;
                    unsigned int done_f = 0;
                    if (done_f == 0) {
                        int in_b_0 = 0;
                        if (grp_reg[0] == bucket) {
                            in_b_0 = 1;
                        }
                        unsigned int m_f_0 = __ballot_sync(0xFFFFFFFF, in_b_0 != 0);
                        unsigned int c_f_0 = (unsigned int)(unsigned int)__popc(m_f_0);
                        if (running + c_f_0 > j_sel) {
                            unsigned int want_f_0 = j_sel - running;
                            unsigned int rank_f_0 = (unsigned int)(unsigned int)__popc(m_f_0 & below_mask);
                            int mine_f_0 = 0;
                            if (in_b_0 != 0 && rank_f_0 == want_f_0) {
                                mine_f_0 = 1;
                            }
                            unsigned int sel_mask_0 = __ballot_sync(0xFFFFFFFF, mine_f_0 != 0);
                            int _find_first_set_19 = __ffs(sel_mask_0);
                            q_found = _find_first_set_19 - 1;
                            done_f = 1;
                        }
                        running = running + c_f_0;
                    }
                    if (done_f == 0) {
                        int in_b_1 = 0;
                        if (grp_reg[1] == bucket) {
                            in_b_1 = 1;
                        }
                        unsigned int m_f_1 = __ballot_sync(0xFFFFFFFF, in_b_1 != 0);
                        unsigned int c_f_1 = (unsigned int)(unsigned int)__popc(m_f_1);
                        if (running + c_f_1 > j_sel) {
                            unsigned int want_f_1 = j_sel - running;
                            unsigned int rank_f_1 = (unsigned int)(unsigned int)__popc(m_f_1 & below_mask);
                            int mine_f_1 = 0;
                            if (in_b_1 != 0 && rank_f_1 == want_f_1) {
                                mine_f_1 = 1;
                            }
                            unsigned int sel_mask_1 = __ballot_sync(0xFFFFFFFF, mine_f_1 != 0);
                            int _find_first_set_20 = __ffs(sel_mask_1);
                            q_found = 32 + (_find_first_set_20 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_1;
                    }
                    if (done_f == 0) {
                        int in_b_2 = 0;
                        if (grp_reg[2] == bucket) {
                            in_b_2 = 1;
                        }
                        unsigned int m_f_2 = __ballot_sync(0xFFFFFFFF, in_b_2 != 0);
                        unsigned int c_f_2 = (unsigned int)(unsigned int)__popc(m_f_2);
                        if (running + c_f_2 > j_sel) {
                            unsigned int want_f_2 = j_sel - running;
                            unsigned int rank_f_2 = (unsigned int)(unsigned int)__popc(m_f_2 & below_mask);
                            int mine_f_2 = 0;
                            if (in_b_2 != 0 && rank_f_2 == want_f_2) {
                                mine_f_2 = 1;
                            }
                            unsigned int sel_mask_2 = __ballot_sync(0xFFFFFFFF, mine_f_2 != 0);
                            int _find_first_set_21 = __ffs(sel_mask_2);
                            q_found = 64 + (_find_first_set_21 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_2;
                    }
                    if (done_f == 0) {
                        int in_b_3 = 0;
                        if (grp_reg[3] == bucket) {
                            in_b_3 = 1;
                        }
                        unsigned int m_f_3 = __ballot_sync(0xFFFFFFFF, in_b_3 != 0);
                        unsigned int c_f_3 = (unsigned int)(unsigned int)__popc(m_f_3);
                        if (running + c_f_3 > j_sel) {
                            unsigned int want_f_3 = j_sel - running;
                            unsigned int rank_f_3 = (unsigned int)(unsigned int)__popc(m_f_3 & below_mask);
                            int mine_f_3 = 0;
                            if (in_b_3 != 0 && rank_f_3 == want_f_3) {
                                mine_f_3 = 1;
                            }
                            unsigned int sel_mask_3 = __ballot_sync(0xFFFFFFFF, mine_f_3 != 0);
                            int _find_first_set_22 = __ffs(sel_mask_3);
                            q_found = 96 + (_find_first_set_22 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_3;
                    }
                    if (done_f == 0) {
                        int in_b_4 = 0;
                        if (grp_reg[4] == bucket) {
                            in_b_4 = 1;
                        }
                        unsigned int m_f_4 = __ballot_sync(0xFFFFFFFF, in_b_4 != 0);
                        unsigned int c_f_4 = (unsigned int)(unsigned int)__popc(m_f_4);
                        if (running + c_f_4 > j_sel) {
                            unsigned int want_f_4 = j_sel - running;
                            unsigned int rank_f_4 = (unsigned int)(unsigned int)__popc(m_f_4 & below_mask);
                            int mine_f_4 = 0;
                            if (in_b_4 != 0 && rank_f_4 == want_f_4) {
                                mine_f_4 = 1;
                            }
                            unsigned int sel_mask_4 = __ballot_sync(0xFFFFFFFF, mine_f_4 != 0);
                            int _find_first_set_23 = __ffs(sel_mask_4);
                            q_found = 128 + (_find_first_set_23 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_4;
                    }
                    if (done_f == 0) {
                        int in_b_5 = 0;
                        if (grp_reg[5] == bucket) {
                            in_b_5 = 1;
                        }
                        unsigned int m_f_5 = __ballot_sync(0xFFFFFFFF, in_b_5 != 0);
                        unsigned int c_f_5 = (unsigned int)(unsigned int)__popc(m_f_5);
                        if (running + c_f_5 > j_sel) {
                            unsigned int want_f_5 = j_sel - running;
                            unsigned int rank_f_5 = (unsigned int)(unsigned int)__popc(m_f_5 & below_mask);
                            int mine_f_5 = 0;
                            if (in_b_5 != 0 && rank_f_5 == want_f_5) {
                                mine_f_5 = 1;
                            }
                            unsigned int sel_mask_5 = __ballot_sync(0xFFFFFFFF, mine_f_5 != 0);
                            int _find_first_set_24 = __ffs(sel_mask_5);
                            q_found = 160 + (_find_first_set_24 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_5;
                    }
                    if (done_f == 0) {
                        int in_b_6 = 0;
                        if (grp_reg[6] == bucket) {
                            in_b_6 = 1;
                        }
                        unsigned int m_f_6 = __ballot_sync(0xFFFFFFFF, in_b_6 != 0);
                        unsigned int c_f_6 = (unsigned int)(unsigned int)__popc(m_f_6);
                        if (running + c_f_6 > j_sel) {
                            unsigned int want_f_6 = j_sel - running;
                            unsigned int rank_f_6 = (unsigned int)(unsigned int)__popc(m_f_6 & below_mask);
                            int mine_f_6 = 0;
                            if (in_b_6 != 0 && rank_f_6 == want_f_6) {
                                mine_f_6 = 1;
                            }
                            unsigned int sel_mask_6 = __ballot_sync(0xFFFFFFFF, mine_f_6 != 0);
                            int _find_first_set_25 = __ffs(sel_mask_6);
                            q_found = 192 + (_find_first_set_25 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_6;
                    }
                    if (done_f == 0) {
                        int in_b_7 = 0;
                        if (grp_reg[7] == bucket) {
                            in_b_7 = 1;
                        }
                        unsigned int m_f_7 = __ballot_sync(0xFFFFFFFF, in_b_7 != 0);
                        unsigned int c_f_7 = (unsigned int)(unsigned int)__popc(m_f_7);
                        if (running + c_f_7 > j_sel) {
                            unsigned int want_f_7 = j_sel - running;
                            unsigned int rank_f_7 = (unsigned int)(unsigned int)__popc(m_f_7 & below_mask);
                            int mine_f_7 = 0;
                            if (in_b_7 != 0 && rank_f_7 == want_f_7) {
                                mine_f_7 = 1;
                            }
                            unsigned int sel_mask_7 = __ballot_sync(0xFFFFFFFF, mine_f_7 != 0);
                            int _find_first_set_26 = __ffs(sel_mask_7);
                            q_found = 224 + (_find_first_set_26 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_7;
                    }
                    #pragma unroll 1
                    for (int fi2 = 8; fi2 < n_scan; fi2++) {
                        if (done_f == 0) {
                            int in_b2 = 0;
                            int qb_f = fi2 * 32 + lane_1;
                            if (qb_f < query_blocks) {
                                int cnt_f = indptr[qb_f + 1] - indptr[qb_f];
                                int grp_16 = (cnt_f + 4 - 1) / 4;
                                if (grp_16 > 63) {
                                    grp_16 = 63;
                                }
                                if (grp_16 == bucket) {
                                    in_b2 = 1;
                                }
                            }
                            unsigned int m_f2 = __ballot_sync(0xFFFFFFFF, in_b2 != 0);
                            unsigned int c_f2 = (unsigned int)(unsigned int)__popc(m_f2);
                            if (running + c_f2 > j_sel) {
                                unsigned int want_f2 = j_sel - running;
                                unsigned int rank_f2 = (unsigned int)(unsigned int)__popc(m_f2 & below_mask);
                                int mine_f2 = 0;
                                if (in_b2 != 0 && rank_f2 == want_f2) {
                                    mine_f2 = 1;
                                }
                                unsigned int sel_mask2 = __ballot_sync(0xFFFFFFFF, mine_f2 != 0);
                                int _find_first_set_27 = __ffs(sel_mask2);
                                q_found = fi2 * 32 + (_find_first_set_27 - 1);
                                done_f = 1;
                            }
                            running = running + c_f2;
                        }
                    }
                    int tile_map = (int)head_sel * query_blocks + q_found;
                    int tile_d = (int)draw;
                    unsigned int skip = 0;
                    if (valid_d != 0) {
                        if (uniform_sel == 0) {
                            tile_d = tile_map;
                            if (tile_d < grid_i) {
                                skip = 1;
                            }
                        }
                    }
                    ticket = draw;
                    valid_tok = valid_d;
                    tile_sel = tile_d;
                    if (skip == 0) {
                        break;
                    }
                }
                unsigned int token_base = work_k * 8;
                if (valid_tok != 0) {
                    int q_block_p = tile_sel % query_blocks;
                    int head_p = tile_sel / query_blocks;
                    int row_begin_p = indptr[q_block_p];
                    int count_p = indptr[q_block_p + 1] - row_begin_p;
                    int groups_p = (count_p + 4 - 1) / 4;
                    if (lane_1 == 0) {
                        work_token_words[token_base + 1] = (unsigned int)q_block_p;
                        work_token_words[token_base + 2] = (unsigned int)head_p;
                        work_token_words[token_base + 3] = (unsigned int)row_begin_p;
                        work_token_words[token_base + 4] = (unsigned int)count_p;
                        work_token_words[token_base + 5] = (unsigned int)groups_p;
                        work_token_words[token_base + 6] = ticket;
                        work_token_words[token_base + 7] = gate_flag;
                    }
                }
                if (lane_1 == 0) {
                    work_token_words[token_base] = valid_tok;
                    mbarrier_arrive(work_full_addr + (work_k) * 8);
                }
                if (lane_1 == 0) {
                    if (valid_tok != 0) {
                        if (debug_order != 0) {
                            order_debug[(int)ticket] = tile_sel;
                        }
                    }
                }
                n_pub = n_pub + 1;
                work_k += 1;
                if (work_k == 4) { work_k = 0; _phase_work_empty ^= 1; }
                if (valid_tok == 0) {
                    break;
                }
            }
            if (gate_flag == 0) {
                if (bid_1 < total_tiles) {
                    mbarrier_wait_hint(claim_gate_addr, 0, 1000);
                }
            }
        }
    }
    // ---- Role: empty ----
    if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
