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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#if __CUDA_ARCH__ == 1070
#define TMEM_NCOLS 256
#else
#define TMEM_NCOLS 512
#endif
#define TMEM_A_TMEM_OFFSET 0
#define TMEM_ACC_SMALL_OFFSET 128
#define TMEM_ACC_BIG_OFFSET 176
#define NUM_MAIN_STAGES 1
#define SMEM_RESIDUAL_OFF 0
#define SMEM_RESIDUAL_STAGE_BYTES 32768
#define SMEM_RESIDUAL_STRIDE 32768
#define SMEM_X_OFF 98304
#define SMEM_X_STAGE_BYTES 8192
#define SMEM_X_STRIDE 8192
#define SMEM_RESIDUAL_PAIRS_OFF 0
#define SMEM_RESIDUAL_PAIRS_STAGE_BYTES 98304
#define SMEM_RESIDUAL_PAIRS_STRIDE 98304
#define SMEM_X_PAIRS_OFF 98304
#define SMEM_X_PAIRS_STAGE_BYTES 24576
#define SMEM_X_PAIRS_STRIDE 24576
#define SMEM_FN_OFF 122880
#define SMEM_FN_STAGE_BYTES 36864
#define SMEM_FN_STRIDE 36864
#define SMEM_FN_PIECES_OFF 122880
#define SMEM_FN_PIECES_STAGE_BYTES 3072
#define SMEM_FN_PIECES_STRIDE 3072
#define SMEM_POST_OFF 196608
#define SMEM_POST_STAGE_BYTES 1024
#define SMEM_POST_STRIDE 1024
#define SMEM_COMB_OFF 198656
#define SMEM_COMB_STAGE_BYTES 4096
#define SMEM_COMB_STRIDE 4096
#define SMEM_HC_SUMS_OFF 206848
#define SMEM_HC_SUMS_STAGE_BYTES 512
#define SMEM_HC_SUMS_STRIDE 512
#define SMEM_TAIL_SCRATCH_OFF 207360
#define SMEM_TAIL_SCRATCH_STAGE_BYTES 256
#define SMEM_TAIL_SCRATCH_STRIDE 256
#define SMEM_TAIL_CONTROL_OFF 207360
#define SMEM_TAIL_CONTROL_STAGE_BYTES 256
#define SMEM_TAIL_CONTROL_STRIDE 256
#if __CUDA_ARCH__ == 1070
#define SMEM_QUEUE_OFF 207616
#else
#define SMEM_QUEUE_OFF 207576
#endif
#define SMEM_QUEUE_STAGE_BYTES 4
#define SMEM_QUEUE_STRIDE 4
#if __CUDA_ARCH__ == 1070
#else
#define SMEM_PAIR_SUMS_OFF 207616
#define SMEM_PAIR_SUMS_STAGE_BYTES 256
#define SMEM_PAIR_SUMS_STRIDE 256
#endif
#define SMEM_WEIGHT_WORDS_OFF 207872
#define SMEM_WEIGHT_WORDS_STAGE_BYTES 8192
#define SMEM_WEIGHT_WORDS_STRIDE 8192
#if __CUDA_ARCH__ == 1070
#define SMEM_EXHAUSTED_OFF 207680
#define SMEM_EXHAUSTED_STAGE_BYTES 64
#define SMEM_EXHAUSTED_STRIDE 64
#else
#define SMEM_EXHAUSTED_OFF 207580
#define SMEM_EXHAUSTED_STAGE_BYTES 4
#define SMEM_EXHAUSTED_STRIDE 4
#endif
#define SMEM_TOTAL 217088
#define THREADS 768
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




__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}


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



extern "C" __device__ float __nv_expf(float);



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
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

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_mhc_41024b807c02040677b3(const __grid_constant__ CUtensorMap residual_map, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap fnp_map, const __grid_constant__ CUtensorMap post_map, const __grid_constant__ CUtensorMap comb_map, const __grid_constant__ CUtensorMap new_residual_map, float* __restrict__ mix_scales, float* __restrict__ mix_bases, float* __restrict__ new_post_mix, float* __restrict__ new_comb_res_mix, __nv_bfloat16* __restrict__ rmsnorm_weight, __nv_bfloat16* __restrict__ new_residual, __nv_bfloat16* __restrict__ y_bf16, float* __restrict__ scratch, unsigned long long* __restrict__ split_barriers, unsigned long long* __restrict__ launch_epochs, unsigned int num_tokens, unsigned int num_splits, CakeFastDivmod split_div, unsigned int num_ctas, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, unsigned int num_sinkhorn_iters, float rmsnorm_eps, float rmsnorm_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 216064;
    #define full_io_addr (mbar_base + 0)
    #define empty_io_addr (mbar_base + 24)
    #define store_ready_addr (mbar_base + 48)
    #define full_fn_addr (mbar_base + 72)
    #define empty_fn_addr (mbar_base + 88)
    #define full_a_addr (mbar_base + 104)
    #define empty_a_addr (mbar_base + 136)
    #define big_full_addr (mbar_base + 168)
    #define big_empty_addr (mbar_base + 184)
    #define small_full_addr (mbar_base + 200)
    #define small_empty_addr (mbar_base + 216)
    #define full_coeff_addr (mbar_base + 232)
    #define empty_coeff_addr (mbar_base + 248)
    #define full_stats_addr (mbar_base + 264)
    #define empty_stats_addr (mbar_base + 272)
    #define mix_arrival_addr (mbar_base + 280)
    #define weight_full_addr (mbar_base + 288)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* residual = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int residual_addr = smem + 0;
    __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int x_addr = smem + 98304;
    unsigned int* residual_pairs = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int residual_pairs_addr = smem + 0;
    unsigned int* x_pairs = reinterpret_cast<unsigned int*>(smem_raw + 98304);
    const int x_pairs_addr = smem + 98304;
    __nv_bfloat16* fn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 122880);
    const int fn_addr = smem + 122880;
    __nv_bfloat16* fn_pieces = reinterpret_cast<__nv_bfloat16*>(smem_raw + 122880);
    const int fn_pieces_addr = smem + 122880;
    float* post = reinterpret_cast<float*>(smem_raw + 196608);
    const int post_addr = smem + 196608;
    float* comb = reinterpret_cast<float*>(smem_raw + 198656);
    const int comb_addr = smem + 198656;
    float* hc_sums = reinterpret_cast<float*>(smem_raw + 206848);
    const int hc_sums_addr = smem + 206848;
    float* tail_scratch = reinterpret_cast<float*>(smem_raw + 207360);
    const int tail_scratch_addr = smem + 207360;
    unsigned int* tail_control = reinterpret_cast<unsigned int*>(smem_raw + 207360);
    const int tail_control_addr = smem + 207360;
#if __CUDA_ARCH__ == 1070
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 207616);
    const int queue_addr = smem + 207616;
#else
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 207576);
    const int queue_addr = smem + 207576;
    float* pair_sums = reinterpret_cast<float*>(smem_raw + 207616);
    const int pair_sums_addr = smem + 207616;
#endif
    unsigned int* weight_words = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int weight_words_addr = smem + 207872;
#if __CUDA_ARCH__ == 1070
    unsigned int* exhausted = reinterpret_cast<unsigned int*>(smem_raw + 207680);
    const int exhausted_addr = smem + 207680;
#else
    unsigned int* exhausted = reinterpret_cast<unsigned int*>(smem_raw + 207580);
    const int exhausted_addr = smem + 207580;
#endif
    unsigned int lane_in_warp = threadIdx.x % 32;
    uint32_t _fast_div_q_0 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(64), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(64);
    uint32_t _fast_div_r_0 = (uint32_t)(64) - _fast_div_q_0 * (uint32_t)(split_div.divisor);
    if (warp == 0) {
        if (elect_sync()) {
            queue[0] = 0;
            exhausted[0] = 0;
#if __CUDA_ARCH__ == 1070
            exhausted[1] = 0;
            exhausted[2] = 0;
            exhausted[3] = 0;
            exhausted[4] = 0;
            exhausted[5] = 0;
            exhausted[6] = 0;
            exhausted[7] = 0;
            exhausted[8] = 0;
            exhausted[9] = 0;
            exhausted[10] = 0;
            exhausted[11] = 0;
            exhausted[12] = 0;
            exhausted[13] = 0;
            exhausted[14] = 0;
            exhausted[15] = 0;
#else
#endif
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&residual_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&x_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&fnp_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&post_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&comb_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&new_residual_map))) : "memory");
        }
    }

    // Mbarrier init (17 pipeline groups, 0 ordered-sequence groups, 37 barriers)
    // Mbarriers at smem_raw[216064..216360)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_io: 3 barriers, init_count=1
            mbarrier_init(smem + 216064, 1);
            mbarrier_init(smem + 216072, 1);
            mbarrier_init(smem + 216080, 1);
            // empty_io: 3 barriers, init_count=1
            mbarrier_init(smem + 216088, 1);
            mbarrier_init(smem + 216096, 1);
            mbarrier_init(smem + 216104, 1);
            // store_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 216112, 8);
            mbarrier_init(smem + 216120, 8);
            mbarrier_init(smem + 216128, 8);
            // full_coeff: 2 barriers, init_count=1
            mbarrier_init(smem + 216296, 1);
            mbarrier_init(smem + 216304, 1);
            // empty_coeff: 2 barriers, init_count=8
            mbarrier_init(smem + 216312, 8);
            mbarrier_init(smem + 216320, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 216136, 1);
            mbarrier_init(smem + 216144, 1);
            // empty_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 216152, 1);
            mbarrier_init(smem + 216160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_a: 4 barriers, init_count=128
            mbarrier_init(smem + 216168, 128);
            mbarrier_init(smem + 216176, 128);
            mbarrier_init(smem + 216184, 128);
            mbarrier_init(smem + 216192, 128);
            // empty_a: 4 barriers, init_count=1
            mbarrier_init(smem + 216200, 1);
            mbarrier_init(smem + 216208, 1);
            mbarrier_init(smem + 216216, 1);
            mbarrier_init(smem + 216224, 1);
            // big_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216232, 1);
            mbarrier_init(smem + 216240, 1);
            // big_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 216248, 4);
            mbarrier_init(smem + 216256, 4);
            // small_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216264, 1);
            mbarrier_init(smem + 216272, 1);
            // small_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 216280, 4);
            mbarrier_init(smem + 216288, 4);
            // full_stats: 1 barriers, init_count=8
            mbarrier_init(smem + 216328, 8);
            // empty_stats: 1 barriers, init_count=4
            mbarrier_init(smem + 216336, 4);
            // mix_arrival: 1 barriers, init_count=5
            mbarrier_init(smem + 216344, 5);
            // weight_full: 1 barriers, init_count=32
            mbarrier_init(smem + 216352, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

#if __CUDA_ARCH__ == 1070
    // TMEM alloc (256 columns, 224 used)
#else
    // TMEM alloc (512 columns, 272 used)
#endif
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216360);
    if (warp == 3) {
        int _tmem_hold = smem + 216360;
#if __CUDA_ARCH__ == 1070
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
#else
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
#endif
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_a_tmem = taddr;
    const int tmem_acc_small = taddr + 128;
    const int tmem_acc_big = taddr + 176;

    // ---- Role: control ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // control_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
            if (warp < 2) {
                unsigned int seq = 0;
                unsigned long long io_policy = (((num_tokens + 63) / 64 <= 64) ? 1364590687093260288 : 1152921504606846976);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int task = (unsigned int)bid; task < (num_tokens + 63) / 64 * num_splits; task += num_ctas) {
                        uint32_t fast_div_q = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task);
                        uint32_t fast_div_r = (uint32_t)(task) - fast_div_q * (uint32_t)(split_div.divisor);
                        #pragma unroll 1
                        for (unsigned int kb = 0; kb < _fast_div_q_0 + ((fast_div_r < _fast_div_r_0) ? 1 : 0); kb++) {
                            unsigned int hidden = (fast_div_r * _fast_div_q_0 + ((fast_div_r < _fast_div_r_0) ? fast_div_r : _fast_div_r_0)) * 64 + kb * 64;
                            if (warp == 0) {
                                unsigned int stage = seq % 3;
                                unsigned int _stage_empty_io = seq % 3;
                                unsigned int _parity_empty_io = seq / 3 % 2 ^ 1;
                                mbarrier_wait(empty_io_addr + (_stage_empty_io) * 8, _parity_empty_io);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(residual_addr + stage * 32768), "l"((&residual_map)), "r"(hidden), "r"(fast_div_q * 64), "r"(0),
                                       "r"(full_io_addr + (stage) * 8), "l"(io_policy) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(x_addr + stage * 8192), "l"((&x_map)), "r"(hidden), "r"(fast_div_q * 64),
                                       "r"(full_io_addr + (stage) * 8), "l"(io_policy) : "memory");
                                mbarrier_arrive_expect_tx(full_io_addr + (stage) * 8, 40960);
                            } else {
                                unsigned int stage_1 = seq % 2;
                                unsigned int _stage_empty_fn = seq % 2;
                                unsigned int _parity_empty_fn = seq / 2 % 2 ^ 1;
                                mbarrier_wait(empty_fn_addr + (_stage_empty_fn) * 8, _parity_empty_fn);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864), "l"((&fnp_map)), "r"(hidden), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 9216), "l"((&fnp_map)), "r"(hidden + 4096), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 18432), "l"((&fnp_map)), "r"(hidden + 8192), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 27648), "l"((&fnp_map)), "r"(hidden + 12288), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                mbarrier_arrive_expect_tx(full_fn_addr + (stage_1) * 8, 36864);
                            }
                            seq += 1;
                        }
                    }
                }
                __syncwarp();
            } else if (warp == 2) {
                unsigned int fseq = 0;
                unsigned int aseq = 0;
                unsigned int cseq = 0;
                unsigned int useq = 0;
                #pragma unroll 1
                for (unsigned int task_1 = (unsigned int)bid; task_1 < (num_tokens + 63) / 64 * num_splits; task_1 += num_ctas) {
                    uint32_t fast_div_q_1 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_1), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_1);
                    uint32_t fast_div_r_1 = (uint32_t)(task_1) - fast_div_q_1 * (uint32_t)(split_div.divisor);
                    unsigned int ustage = useq % 2;
                    unsigned int _stage_small_empty = useq % 2;
                    unsigned int _parity_small_empty = useq / 2 % 2 ^ 1;
                    mbarrier_wait(small_empty_addr + (_stage_small_empty) * 8, _parity_small_empty);
                    #pragma unroll 1
                    for (unsigned int kb_1 = 0; kb_1 < _fast_div_q_0 + ((fast_div_r_1 < _fast_div_r_0) ? 1 : 0); kb_1++) {
                        unsigned int fstage = fseq % 2;
                        unsigned int _stage_full_fn = fseq % 2;
                        unsigned int _parity_full_fn = fseq / 2 % 2;
                        mbarrier_wait(full_fn_addr + (_stage_full_fn) * 8, _parity_full_fn);
                        unsigned int cstage_2_0 = cseq % 2;
                        unsigned int _stage_big_empty = cseq % 2;
                        unsigned int _parity_big_empty = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (_stage_big_empty) * 8, _parity_big_empty);
                        unsigned int astage_1_0 = aseq % 4;
                        unsigned int _stage_full_a = aseq % 4;
                        unsigned int _parity_full_a = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a) * 8, _parity_full_a);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_0, 0x40004040, 67503248, ((kb_1 == 0) ? 0 : 1));
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_48_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_48_0, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_49_0, 0x40004040, 67503248, 0);
                        int _mma_b_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_49_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48)), (unsigned int)tmem_a_tmem + astage_1_0 * 32, _mma_b_lo_49_0, 0x40004040, 67503248, 0);
                        int _mma_b_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_1, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_48_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_48_1, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_49_1, 0x40004040, 67503248, 1);
                        int _mma_b_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_49_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 8), _mma_b_lo_49_1, 0x40004040, 67503248, 1);
                        int _mma_b_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_2, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_48_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_48_2, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_49_2, 0x40004040, 67503248, 1);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_49_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 16), _mma_b_lo_49_2, 0x40004040, 67503248, 1);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_3, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_48_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_48_3, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_49_3, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_49_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48)), (unsigned int)tmem_a_tmem + (astage_1_0 * 32 + 24), _mma_b_lo_49_3, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_1_0) * 8);
                        aseq += 1;
                        unsigned int astage_1_1 = aseq % 4;
                        unsigned int _stage_full_a_1 = aseq % 4;
                        unsigned int _parity_full_a_1 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_1) * 8, _parity_full_a_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_4, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_48_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_48_4, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_49_4, 0x40004040, 67503248, 1);
                        int _mma_b_lo_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_49_4 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48 + 24)), (unsigned int)tmem_a_tmem + astage_1_1 * 32, _mma_b_lo_49_4, 0x40004040, 67503248, 0);
                        int _mma_b_lo_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_5, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_48_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_48_5, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_49_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_49_5 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 8), _mma_b_lo_49_5, 0x40004040, 67503248, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_6, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_48_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_48_6, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_49_6, 0x40004040, 67503248, 1);
                        int _mma_b_lo_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_49_6 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 16), _mma_b_lo_49_6, 0x40004040, 67503248, 1);
                        int _mma_b_lo_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_7, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_48_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_48_7, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_49_7, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_49_7 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_0 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_1 * 32 + 24), _mma_b_lo_49_7, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_1_1) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage_2_0) * 8);
                        cseq += 1;
                        unsigned int cstage_2_1 = cseq % 2;
                        unsigned int _stage_big_empty_1 = cseq % 2;
                        unsigned int _parity_big_empty_1 = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (_stage_big_empty_1) * 8, _parity_big_empty_1);
                        unsigned int astage_1_2 = aseq % 4;
                        unsigned int _stage_full_a_2 = aseq % 4;
                        unsigned int _parity_full_a_2 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_2) * 8, _parity_full_a_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_8, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_48_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_48_8, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_49_8, 0x40004040, 67503248, 0);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_49_8 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48)), (unsigned int)tmem_a_tmem + astage_1_2 * 32, _mma_b_lo_49_8, 0x40004040, 67503248, 0);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_9, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_48_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_48_9, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_49_9, 0x40004040, 67503248, 1);
                        int _mma_b_lo_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_49_9 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 8), _mma_b_lo_49_9, 0x40004040, 67503248, 1);
                        int _mma_b_lo_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_10, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_48_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_48_10, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_49_10, 0x40004040, 67503248, 1);
                        int _mma_b_lo_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_49_10 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 16), _mma_b_lo_49_10, 0x40004040, 67503248, 1);
                        int _mma_b_lo_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_11, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_48_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_48_11, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_49_11, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_49_11 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48)), (unsigned int)tmem_a_tmem + (astage_1_2 * 32 + 24), _mma_b_lo_49_11, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_1_2) * 8);
                        aseq += 1;
                        unsigned int astage_1_3 = aseq % 4;
                        unsigned int _stage_full_a_3 = aseq % 4;
                        unsigned int _parity_full_a_3 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (_stage_full_a_3) * 8, _parity_full_a_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
#else
                        int _mma_b_lo_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_12, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
#else
                        int _mma_b_lo_48_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_48_12, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_49_12, 0x40004040, 67503248, 1);
                        int _mma_b_lo_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
#else
                        int _mma_b_lo_49_12 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48 + 24)), (unsigned int)tmem_a_tmem + astage_1_3 * 32, _mma_b_lo_49_12, 0x40004040, 67503248, 0);
                        int _mma_b_lo_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_13, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
#else
                        int _mma_b_lo_48_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_48_13, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_49_13, 0x40004040, 67503248, 1);
                        int _mma_b_lo_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
#else
                        int _mma_b_lo_49_13 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 8), _mma_b_lo_49_13, 0x40004040, 67503248, 1);
                        int _mma_b_lo_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_14, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
#else
                        int _mma_b_lo_48_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_48_14, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_49_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
#else
                        int _mma_b_lo_49_14 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 16), _mma_b_lo_49_14, 0x40004040, 67503248, 1);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 2) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_15, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_48_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
#else
                        int _mma_b_lo_48_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9 + 1) * 192);
#endif
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_48_15, 0x40004040, 67503248, 1);
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_49_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_49_15, 0x40004040, 67503248, 1);
#else
                        int _mma_b_lo_49_15 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x3FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2_1 * 48 + 24)), (unsigned int)tmem_a_tmem + (astage_1_3 * 32 + 24), _mma_b_lo_49_15, 0x40004040, 67503248, 1);
#endif
                        elect_commit(empty_a_addr + (astage_1_3) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage_2_1) * 8);
                        cseq += 1;
                        elect_commit(empty_fn_addr + (fstage) * 8);
                        fseq += 1;
                    }
                    elect_commit(small_full_addr + (ustage) * 8);
                    useq += 1;
                }
            } else {
                unsigned int io_seq = 0;
                unsigned int coeff_seq = 0;
                unsigned int tasks_done = 0;
                if ((unsigned int)bid < (num_tokens + 63) / 64 * num_splits) {
                    mbarrier_wait(empty_coeff_addr, 1);
                    if (elect_sync()) {
                        uint32_t fast_div_q_2 = (split_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)bid), split_div.multiplier) >> split_div.shift_right) : (uint32_t)((unsigned int)bid);
                        uint32_t fast_div_r_2 = (uint32_t)((unsigned int)bid) - fast_div_q_2 * (uint32_t)(split_div.divisor);
                        tma_2d_gmem2smem(post_addr, (&post_map), 0, fast_div_q_2 * 64, full_coeff_addr);
                        tma_2d_gmem2smem(comb_addr, (&comb_map), 0, fast_div_q_2 * 64, full_coeff_addr);
                        mbarrier_arrive_expect_tx(full_coeff_addr, 5120);
                    }
                    __syncwarp();
                    coeff_seq += 1;
                }
                #pragma unroll 1
                for (unsigned int task_2 = (unsigned int)bid; task_2 < (num_tokens + 63) / 64 * num_splits; task_2 += num_ctas) {
                    uint32_t fast_div_q_3 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2);
                    uint32_t fast_div_r_3 = (uint32_t)(task_2) - fast_div_q_3 * (uint32_t)(split_div.divisor);
                    unsigned int blocks = _fast_div_q_0 + ((fast_div_r_3 < _fast_div_r_0) ? 1 : 0);
                    #pragma unroll 1
                    for (unsigned int kb_2 = 0; kb_2 < blocks; kb_2++) {
                        unsigned int stage_2 = io_seq % 3;
                        unsigned int _stage_store_ready = io_seq % 3;
                        unsigned int _parity_store_ready = io_seq / 3 % 2;
                        mbarrier_wait(store_ready_addr + (_stage_store_ready) * 8, _parity_store_ready);
                        unsigned int prefetch = kb_2 + 1 == blocks && task_2 + num_ctas < (num_tokens + 63) / 64 * num_splits;
                        if (prefetch != 0) {
                            unsigned int _stage_empty_coeff = coeff_seq % 2;
                            unsigned int _parity_empty_coeff = coeff_seq / 2 % 2 ^ 1;
                            mbarrier_wait(empty_coeff_addr + (_stage_empty_coeff) * 8, _parity_empty_coeff);
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3}], [%4], %5;"
                                :: "l"((&new_residual_map)), "r"((fast_div_r_3 * _fast_div_q_0 + ((fast_div_r_3 < _fast_div_r_0) ? fast_div_r_3 : _fast_div_r_0)) * 64 + kb_2 * 64), "r"(fast_div_q_3 * 64), "r"(0), "r"(residual_addr + stage_2 * 32768), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            if (prefetch != 0) {
                                uint32_t fast_div_q_0 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2 + num_ctas), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2 + num_ctas);
                                uint32_t fast_div_r_0 = (uint32_t)(task_2 + num_ctas) - fast_div_q_0 * (uint32_t)(split_div.divisor);
                                tma_2d_gmem2smem(post_addr + coeff_seq % 2 * 1024, (&post_map), 0, fast_div_q_0 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                tma_2d_gmem2smem(comb_addr + coeff_seq % 2 * 4096, (&comb_map), 0, fast_div_q_0 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                mbarrier_arrive_expect_tx(full_coeff_addr + (coeff_seq % 2) * 8, 5120);
                            }
                            if (blocks > kb_2 + 1) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            } else {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            mbarrier_arrive(empty_io_addr + (stage_2) * 8);
                        }
                        if (prefetch != 0) {
                            coeff_seq += 1;
                        }
                        io_seq += 1;
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(mix_arrival_addr);
                    }
                    tasks_done += 1;
                }
            }
        }
    }
    // ---- Role: posts ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // posts_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
            unsigned int worker = warp / 4 - 1;
            unsigned int io_seq_1 = 0;
            unsigned int a_seq = worker;
            unsigned int coeff_seq_1 = 0;
            unsigned int first_row = warp % 4 * 16 + lane_in_warp / 4;
            unsigned int offs[8];
            unsigned int atom = worker;
            unsigned int pair = (unsigned int)(atom * 8) + lane_in_warp % 4 * 2;
            unsigned int row = first_row;
            offs[0] = row * 32 + (pair / 4 ^ row & 7) * 4 + pair % 4;
            unsigned int row_0 = first_row + 8;
            offs[1] = row_0 * 32 + (pair / 4 ^ row_0 & 7) * 4 + pair % 4;
            unsigned int pair_1 = (unsigned int)(atom * 8 + 1) + lane_in_warp % 4 * 2;
            unsigned int row_2 = first_row;
            offs[2] = row_2 * 32 + (pair_1 / 4 ^ row_2 & 7) * 4 + pair_1 % 4;
            unsigned int row_3 = first_row + 8;
            offs[3] = row_3 * 32 + (pair_1 / 4 ^ row_3 & 7) * 4 + pair_1 % 4;
            unsigned int atom_4 = worker + 2;
            unsigned int pair_5 = (unsigned int)(atom_4 * 8) + lane_in_warp % 4 * 2;
            unsigned int row_6 = first_row;
            offs[4] = row_6 * 32 + (pair_5 / 4 ^ row_6 & 7) * 4 + pair_5 % 4;
            unsigned int row_7 = first_row + 8;
            offs[5] = row_7 * 32 + (pair_5 / 4 ^ row_7 & 7) * 4 + pair_5 % 4;
            unsigned int pair_8 = (unsigned int)(atom_4 * 8 + 1) + lane_in_warp % 4 * 2;
            unsigned int row_9 = first_row;
            offs[6] = row_9 * 32 + (pair_8 / 4 ^ row_9 & 7) * 4 + pair_8 % 4;
            unsigned int row_10 = first_row + 8;
            offs[7] = row_10 * 32 + (pair_8 / 4 ^ row_10 & 7) * 4 + pair_8 % 4;
            unsigned int _phase_empty_stats_0 = 1;
            #pragma unroll 1
            for (unsigned int task_3 = (unsigned int)bid; task_3 < (num_tokens + 63) / 64 * num_splits; task_3 += num_ctas) {
                uint32_t fast_div_q_4 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_3), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_3);
                uint32_t fast_div_r_4 = (uint32_t)(task_3) - fast_div_q_4 * (uint32_t)(split_div.divisor);
                unsigned int coeff_stage = coeff_seq_1 % 2;
                unsigned int _stage_full_coeff = coeff_seq_1 % 2;
                unsigned int _parity_full_coeff = coeff_seq_1 / 2 % 2;
                mbarrier_wait(full_coeff_addr + (_stage_full_coeff) * 8, _parity_full_coeff);
                float post_coeff[8];
                float comb_coeff[32];
                #pragma unroll
                for (int ri = 0; ri < 2; ri++) {
                    unsigned int row_1 = first_row + (unsigned int)(ri * 8);
                    #pragma unroll
                    for (int route = 0; route < 4; route++) {
                        post_coeff[ri * 4 + route] = post[coeff_stage * 256 + row_1 * 4 + (unsigned int)route];
                    }
                    #pragma unroll
                    for (int ci = 0; ci < 16; ci++) {
                        comb_coeff[ri * 16 + ci] = comb[coeff_stage * 1024 + row_1 * 16 + ((unsigned int)(ci / 4) ^ row_1 / 2 & 3) * 4 + (unsigned int)(ci % 4)];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_coeff_addr + (coeff_stage) * 8);
                }
                coeff_seq_1 += 1;
                float hc_acc[2];
                float hc_comp[2];
                hc_acc[0] = 0.0f;
                hc_comp[0] = 0.0f;
                hc_acc[1] = 0.0f;
                hc_comp[1] = 0.0f;
                #pragma unroll 1
                for (unsigned int kb_3 = 0; kb_3 < _fast_div_q_0 + ((fast_div_r_4 < _fast_div_r_0) ? 1 : 0); kb_3++) {
                    unsigned int io_stage = io_seq_1 % 3;
                    unsigned int _stage_full_io = io_seq_1 % 3;
                    unsigned int _parity_full_io = io_seq_1 / 3 % 2;
                    mbarrier_wait(full_io_addr + (_stage_full_io) * 8, _parity_full_io);
                    unsigned int words[16];
                    unsigned int offset = offs[0];
                    unsigned int x_word = x_pairs[io_stage * 2048 + offset];
                    float2 xv = make_float2(__uint_as_float(x_word << 16), __uint_as_float(x_word & 4294901760u));
                    float rv[8];
                    unsigned int word = residual_pairs[io_stage * 8192 + offset];
                    rv[0] = __uint_as_float(word << 16);
                    rv[1] = __uint_as_float(word & 4294901760u);
                    unsigned int word_0 = residual_pairs[io_stage * 8192 + 2048 + offset];
                    rv[2] = __uint_as_float(word_0 << 16);
                    rv[3] = __uint_as_float(word_0 & 4294901760u);
                    unsigned int word_1 = residual_pairs[io_stage * 8192 + 4096 + offset];
                    rv[4] = __uint_as_float(word_1 << 16);
                    rv[5] = __uint_as_float(word_1 & 4294901760u);
                    unsigned int word_2 = residual_pairs[io_stage * 8192 + 6144 + offset];
                    rv[6] = __uint_as_float(word_2 << 16);
                    rv[7] = __uint_as_float(word_2 & 4294901760u);
                    float sq[2];
                    sq[0] = 0.0f;
                    sq[1] = 0.0f;
                    float coeff = post_coeff[0];
                    float2 _f32x2_mul_a_0 = make_float2(coeff, coeff);
                    float2 value;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&xv));
                    float cc = comb_coeff[0];
                    value = fma_f32x2_rn_noftz(make_float2(cc, cc), make_float2(rv[0], rv[1]), value);
                    float cc_r_0 = comb_coeff[4];
                    value = fma_f32x2_rn_noftz(make_float2(cc_r_0, cc_r_0), make_float2(rv[2], rv[3]), value);
                    float cc_r_1 = comb_coeff[8];
                    value = fma_f32x2_rn_noftz(make_float2(cc_r_1, cc_r_1), make_float2(rv[4], rv[5]), value);
                    float cc_r_2 = comb_coeff[12];
                    value = fma_f32x2_rn_noftz(make_float2(cc_r_2, cc_r_2), make_float2(rv[6], rv[7]), value);
                    unsigned int word_6 = __as_u32(__float22bfloat162_rn(make_float2(value.x, value.y)));
                    residual_pairs[io_stage * 8192 + offset] = word_6;
                    words[0] = word_6;
                    float2 rounded_pair = make_float2(__uint_as_float(word_6 << 16), __uint_as_float(word_6 & 4294901760u));
                    float2 squares = fma_f32x2_rn_noftz(rounded_pair, rounded_pair, make_float2(sq[0], sq[1]));
                    sq[0] = squares.x;
                    sq[1] = squares.y;
                    float coeff_r_0 = post_coeff[1];
                    float2 _f32x2_mul_a_1 = make_float2(coeff_r_0, coeff_r_0);
                    float2 value_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&xv));
                    float cc_r_r_0 = comb_coeff[1];
                    value_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_0, cc_r_r_0), make_float2(rv[0], rv[1]), value_r_0);
                    float cc_r_r_r_0 = comb_coeff[5];
                    value_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_0, cc_r_r_r_0), make_float2(rv[2], rv[3]), value_r_0);
                    float cc_r_r_r_r_0 = comb_coeff[9];
                    value_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_0, cc_r_r_r_r_0), make_float2(rv[4], rv[5]), value_r_0);
                    float cc_r_r_r_r_r_0 = comb_coeff[13];
                    value_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_0, cc_r_r_r_r_r_0), make_float2(rv[6], rv[7]), value_r_0);
                    unsigned int word_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_0.x, value_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset] = word_r_0;
                    words[4] = word_r_0;
                    float2 rounded_pair_r_0 = make_float2(__uint_as_float(word_r_0 << 16), __uint_as_float(word_r_0 & 4294901760u));
                    float2 squares_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_0, rounded_pair_r_0, make_float2(sq[0], sq[1]));
                    sq[0] = squares_r_0.x;
                    sq[1] = squares_r_0.y;
                    float coeff_r_1 = post_coeff[2];
                    float2 _f32x2_mul_a_2 = make_float2(coeff_r_1, coeff_r_1);
                    float2 value_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&xv));
                    float cc_r_r_1 = comb_coeff[2];
                    value_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_1, cc_r_r_1), make_float2(rv[0], rv[1]), value_r_1);
                    float cc_r_r_r_1 = comb_coeff[6];
                    value_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_1, cc_r_r_r_1), make_float2(rv[2], rv[3]), value_r_1);
                    float cc_r_r_r_r_1 = comb_coeff[10];
                    value_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_1, cc_r_r_r_r_1), make_float2(rv[4], rv[5]), value_r_1);
                    float cc_r_r_r_r_r_1 = comb_coeff[14];
                    value_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_1, cc_r_r_r_r_r_1), make_float2(rv[6], rv[7]), value_r_1);
                    unsigned int word_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_1.x, value_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset] = word_r_1;
                    words[8] = word_r_1;
                    float2 rounded_pair_r_1 = make_float2(__uint_as_float(word_r_1 << 16), __uint_as_float(word_r_1 & 4294901760u));
                    float2 squares_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_1, rounded_pair_r_1, make_float2(sq[0], sq[1]));
                    sq[0] = squares_r_1.x;
                    sq[1] = squares_r_1.y;
                    float coeff_r_2 = post_coeff[3];
                    float2 _f32x2_mul_a_3 = make_float2(coeff_r_2, coeff_r_2);
                    float2 value_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&xv));
                    float cc_r_r_2 = comb_coeff[3];
                    value_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_2, cc_r_r_2), make_float2(rv[0], rv[1]), value_r_2);
                    float cc_r_r_r_2 = comb_coeff[7];
                    value_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_2, cc_r_r_r_2), make_float2(rv[2], rv[3]), value_r_2);
                    float cc_r_r_r_r_2 = comb_coeff[11];
                    value_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_2, cc_r_r_r_r_2), make_float2(rv[4], rv[5]), value_r_2);
                    float cc_r_r_r_r_r_2 = comb_coeff[15];
                    value_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_2, cc_r_r_r_r_r_2), make_float2(rv[6], rv[7]), value_r_2);
                    unsigned int word_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_2.x, value_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset] = word_r_2;
                    words[12] = word_r_2;
                    float2 rounded_pair_r_2 = make_float2(__uint_as_float(word_r_2 << 16), __uint_as_float(word_r_2 & 4294901760u));
                    float2 squares_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_2, rounded_pair_r_2, make_float2(sq[0], sq[1]));
                    sq[0] = squares_r_2.x;
                    sq[1] = squares_r_2.y;
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(sq[0]), "f"(sq[1]));
                    float sub;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(_add_4), "f"(hc_comp[0]));
                    float add;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(hc_acc[0]), "f"(sub));
                    float _sub_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(add), "f"(hc_acc[0]));
                    float _sub_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_sub_6), "f"(sub));
                    hc_comp[0] = _sub_5;
                    hc_acc[0] = add;
                    unsigned int offset_34 = offs[1];
                    unsigned int x_word_35 = x_pairs[io_stage * 2048 + offset_34];
                    float2 xv_36 = make_float2(__uint_as_float(x_word_35 << 16), __uint_as_float(x_word_35 & 4294901760u));
                    float rv_37[8];
                    unsigned int word_38 = residual_pairs[io_stage * 8192 + offset_34];
                    rv_37[0] = __uint_as_float(word_38 << 16);
                    rv_37[1] = __uint_as_float(word_38 & 4294901760u);
                    unsigned int word_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_34];
                    rv_37[2] = __uint_as_float(word_r_r_0 << 16);
                    rv_37[3] = __uint_as_float(word_r_r_0 & 4294901760u);
                    unsigned int word_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_34];
                    rv_37[4] = __uint_as_float(word_r_r_1 << 16);
                    rv_37[5] = __uint_as_float(word_r_r_1 & 4294901760u);
                    unsigned int word_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_34];
                    rv_37[6] = __uint_as_float(word_r_r_2 << 16);
                    rv_37[7] = __uint_as_float(word_r_r_2 & 4294901760u);
                    float sq_42[2];
                    sq_42[0] = 0.0f;
                    sq_42[1] = 0.0f;
                    float coeff_43 = post_coeff[4];
                    float2 _f32x2_mul_a_7 = make_float2(coeff_43, coeff_43);
                    float2 value_44;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&xv_36));
                    float cc_r_r_r_r_r_r_0 = comb_coeff[16];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_0), make_float2(rv_37[0], rv_37[1]), value_44);
                    float cc_r_r_r_r_r_r_1 = comb_coeff[20];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_1), make_float2(rv_37[2], rv_37[3]), value_44);
                    float cc_r_r_r_r_r_r_2 = comb_coeff[24];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_2), make_float2(rv_37[4], rv_37[5]), value_44);
                    float cc_r_r_r_r_r_r_3 = comb_coeff[28];
                    value_44 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_3), make_float2(rv_37[6], rv_37[7]), value_44);
                    unsigned int word_49 = __as_u32(__float22bfloat162_rn(make_float2(value_44.x, value_44.y)));
                    residual_pairs[io_stage * 8192 + offset_34] = word_49;
                    words[2] = word_49;
                    float2 rounded_pair_50 = make_float2(__uint_as_float(word_49 << 16), __uint_as_float(word_49 & 4294901760u));
                    float2 squares_51 = fma_f32x2_rn_noftz(rounded_pair_50, rounded_pair_50, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_51.x;
                    sq_42[1] = squares_51.y;
                    float coeff_r_r_0 = post_coeff[5];
                    float2 _f32x2_mul_a_8 = make_float2(coeff_r_r_0, coeff_r_r_0);
                    float2 value_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&xv_36));
                    float cc_r_r_r_r_r_r_r_0 = comb_coeff[17];
                    value_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_0), make_float2(rv_37[0], rv_37[1]), value_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_0 = comb_coeff[21];
                    value_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_0), make_float2(rv_37[2], rv_37[3]), value_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_0 = comb_coeff[25];
                    value_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_0), make_float2(rv_37[4], rv_37[5]), value_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[29];
                    value_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_37[6], rv_37[7]), value_r_r_0);
                    unsigned int word_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_0.x, value_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_34] = word_r_r_r_0;
                    words[6] = word_r_r_r_0;
                    float2 rounded_pair_r_r_0 = make_float2(__uint_as_float(word_r_r_r_0 << 16), __uint_as_float(word_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_0, rounded_pair_r_r_0, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_r_r_0.x;
                    sq_42[1] = squares_r_r_0.y;
                    float coeff_r_r_1 = post_coeff[6];
                    float2 _f32x2_mul_a_9 = make_float2(coeff_r_r_1, coeff_r_r_1);
                    float2 value_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&xv_36));
                    float cc_r_r_r_r_r_r_r_1 = comb_coeff[18];
                    value_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_1), make_float2(rv_37[0], rv_37[1]), value_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_1 = comb_coeff[22];
                    value_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_1), make_float2(rv_37[2], rv_37[3]), value_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_1 = comb_coeff[26];
                    value_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_1), make_float2(rv_37[4], rv_37[5]), value_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[30];
                    value_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_37[6], rv_37[7]), value_r_r_1);
                    unsigned int word_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_1.x, value_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_34] = word_r_r_r_1;
                    words[10] = word_r_r_r_1;
                    float2 rounded_pair_r_r_1 = make_float2(__uint_as_float(word_r_r_r_1 << 16), __uint_as_float(word_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_1, rounded_pair_r_r_1, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_r_r_1.x;
                    sq_42[1] = squares_r_r_1.y;
                    float coeff_r_r_2 = post_coeff[7];
                    float2 _f32x2_mul_a_10 = make_float2(coeff_r_r_2, coeff_r_r_2);
                    float2 value_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&xv_36));
                    float cc_r_r_r_r_r_r_r_2 = comb_coeff[19];
                    value_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_2), make_float2(rv_37[0], rv_37[1]), value_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_2 = comb_coeff[23];
                    value_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_2), make_float2(rv_37[2], rv_37[3]), value_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_2 = comb_coeff[27];
                    value_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_2), make_float2(rv_37[4], rv_37[5]), value_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[31];
                    value_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_37[6], rv_37[7]), value_r_r_2);
                    unsigned int word_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_2.x, value_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_34] = word_r_r_r_2;
                    words[14] = word_r_r_r_2;
                    float2 rounded_pair_r_r_2 = make_float2(__uint_as_float(word_r_r_r_2 << 16), __uint_as_float(word_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_2, rounded_pair_r_r_2, make_float2(sq_42[0], sq_42[1]));
                    sq_42[0] = squares_r_r_2.x;
                    sq_42[1] = squares_r_r_2.y;
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(sq_42[0]), "f"(sq_42[1]));
                    float sub_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(_add_11), "f"(hc_comp[1]));
                    float add_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(hc_acc[1]), "f"(sub_0));
                    float _sub_13;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(add_0), "f"(hc_acc[1]));
                    float _sub_12;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(_sub_13), "f"(sub_0));
                    hc_comp[1] = _sub_12;
                    hc_acc[1] = add_0;
                    unsigned int offset_79 = offs[2];
                    unsigned int x_word_80 = x_pairs[io_stage * 2048 + offset_79];
                    float2 xv_81 = make_float2(__uint_as_float(x_word_80 << 16), __uint_as_float(x_word_80 & 4294901760u));
                    float rv_82[8];
                    unsigned int word_83 = residual_pairs[io_stage * 8192 + offset_79];
                    rv_82[0] = __uint_as_float(word_83 << 16);
                    rv_82[1] = __uint_as_float(word_83 & 4294901760u);
                    unsigned int word_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_79];
                    rv_82[2] = __uint_as_float(word_r_r_r_r_0 << 16);
                    rv_82[3] = __uint_as_float(word_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_79];
                    rv_82[4] = __uint_as_float(word_r_r_r_r_1 << 16);
                    rv_82[5] = __uint_as_float(word_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_79];
                    rv_82[6] = __uint_as_float(word_r_r_r_r_2 << 16);
                    rv_82[7] = __uint_as_float(word_r_r_r_r_2 & 4294901760u);
                    float sq_87[2];
                    sq_87[0] = 0.0f;
                    sq_87[1] = 0.0f;
                    float coeff_88 = post_coeff[0];
                    float2 _f32x2_mul_a_14 = make_float2(coeff_88, coeff_88);
                    float2 value_89;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&xv_81));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[0];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_82[0], rv_82[1]), value_89);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[4];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_82[2], rv_82[3]), value_89);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[8];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_82[4], rv_82[5]), value_89);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[12];
                    value_89 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_82[6], rv_82[7]), value_89);
                    unsigned int word_94 = __as_u32(__float22bfloat162_rn(make_float2(value_89.x, value_89.y)));
                    residual_pairs[io_stage * 8192 + offset_79] = word_94;
                    words[1] = word_94;
                    float2 rounded_pair_95 = make_float2(__uint_as_float(word_94 << 16), __uint_as_float(word_94 & 4294901760u));
                    float2 squares_96 = fma_f32x2_rn_noftz(rounded_pair_95, rounded_pair_95, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_96.x;
                    sq_87[1] = squares_96.y;
                    float coeff_r_r_r_0 = post_coeff[1];
                    float2 _f32x2_mul_a_15 = make_float2(coeff_r_r_r_0, coeff_r_r_r_0);
                    float2 value_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&xv_81));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[1];
                    value_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_82[0], rv_82[1]), value_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[5];
                    value_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_82[2], rv_82[3]), value_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[9];
                    value_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_82[4], rv_82[5]), value_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[13];
                    value_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_82[6], rv_82[7]), value_r_r_r_0);
                    unsigned int word_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_0.x, value_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_79] = word_r_r_r_r_r_0;
                    words[5] = word_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_0, rounded_pair_r_r_r_0, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_r_r_r_0.x;
                    sq_87[1] = squares_r_r_r_0.y;
                    float coeff_r_r_r_1 = post_coeff[2];
                    float2 _f32x2_mul_a_16 = make_float2(coeff_r_r_r_1, coeff_r_r_r_1);
                    float2 value_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&xv_81));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[2];
                    value_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_82[0], rv_82[1]), value_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[6];
                    value_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_82[2], rv_82[3]), value_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[10];
                    value_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_82[4], rv_82[5]), value_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[14];
                    value_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_82[6], rv_82[7]), value_r_r_r_1);
                    unsigned int word_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_1.x, value_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_79] = word_r_r_r_r_r_1;
                    words[9] = word_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_1, rounded_pair_r_r_r_1, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_r_r_r_1.x;
                    sq_87[1] = squares_r_r_r_1.y;
                    float coeff_r_r_r_2 = post_coeff[3];
                    float2 _f32x2_mul_a_17 = make_float2(coeff_r_r_r_2, coeff_r_r_r_2);
                    float2 value_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&xv_81));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[3];
                    value_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_82[0], rv_82[1]), value_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[7];
                    value_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_82[2], rv_82[3]), value_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[11];
                    value_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_82[4], rv_82[5]), value_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[15];
                    value_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_82[6], rv_82[7]), value_r_r_r_2);
                    unsigned int word_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_2.x, value_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_79] = word_r_r_r_r_r_2;
                    words[13] = word_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_2, rounded_pair_r_r_r_2, make_float2(sq_87[0], sq_87[1]));
                    sq_87[0] = squares_r_r_r_2.x;
                    sq_87[1] = squares_r_r_r_2.y;
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sq_87[0]), "f"(sq_87[1]));
                    float sub_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(_add_18), "f"(hc_comp[0]));
                    float add_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hc_acc[0]), "f"(sub_1));
                    float _sub_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(add_1), "f"(hc_acc[0]));
                    float _sub_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(_sub_20), "f"(sub_1));
                    hc_comp[0] = _sub_19;
                    hc_acc[0] = add_1;
                    unsigned int offset_124 = offs[3];
                    unsigned int x_word_125 = x_pairs[io_stage * 2048 + offset_124];
                    float2 xv_126 = make_float2(__uint_as_float(x_word_125 << 16), __uint_as_float(x_word_125 & 4294901760u));
                    float rv_127[8];
                    unsigned int word_128 = residual_pairs[io_stage * 8192 + offset_124];
                    rv_127[0] = __uint_as_float(word_128 << 16);
                    rv_127[1] = __uint_as_float(word_128 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_124];
                    rv_127[2] = __uint_as_float(word_r_r_r_r_r_r_0 << 16);
                    rv_127[3] = __uint_as_float(word_r_r_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_124];
                    rv_127[4] = __uint_as_float(word_r_r_r_r_r_r_1 << 16);
                    rv_127[5] = __uint_as_float(word_r_r_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_124];
                    rv_127[6] = __uint_as_float(word_r_r_r_r_r_r_2 << 16);
                    rv_127[7] = __uint_as_float(word_r_r_r_r_r_r_2 & 4294901760u);
                    float sq_132[2];
                    sq_132[0] = 0.0f;
                    sq_132[1] = 0.0f;
                    float coeff_133 = post_coeff[4];
                    float2 _f32x2_mul_a_21 = make_float2(coeff_133, coeff_133);
                    float2 value_134;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_134) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&xv_126));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[16];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_127[0], rv_127[1]), value_134);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[20];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_127[2], rv_127[3]), value_134);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[24];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_127[4], rv_127[5]), value_134);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[28];
                    value_134 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_127[6], rv_127[7]), value_134);
                    unsigned int word_139 = __as_u32(__float22bfloat162_rn(make_float2(value_134.x, value_134.y)));
                    residual_pairs[io_stage * 8192 + offset_124] = word_139;
                    words[3] = word_139;
                    float2 rounded_pair_140 = make_float2(__uint_as_float(word_139 << 16), __uint_as_float(word_139 & 4294901760u));
                    float2 squares_141 = fma_f32x2_rn_noftz(rounded_pair_140, rounded_pair_140, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_141.x;
                    sq_132[1] = squares_141.y;
                    float coeff_r_r_r_r_0 = post_coeff[5];
                    float2 _f32x2_mul_a_22 = make_float2(coeff_r_r_r_r_0, coeff_r_r_r_r_0);
                    float2 value_r_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&xv_126));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[17];
                    value_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_127[0], rv_127[1]), value_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[21];
                    value_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_127[2], rv_127[3]), value_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[25];
                    value_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_127[4], rv_127[5]), value_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[29];
                    value_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_127[6], rv_127[7]), value_r_r_r_r_0);
                    unsigned int word_r_r_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_0.x, value_r_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_124] = word_r_r_r_r_r_r_r_0;
                    words[7] = word_r_r_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_0, rounded_pair_r_r_r_r_0, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_r_r_r_r_0.x;
                    sq_132[1] = squares_r_r_r_r_0.y;
                    float coeff_r_r_r_r_1 = post_coeff[6];
                    float2 _f32x2_mul_a_23 = make_float2(coeff_r_r_r_r_1, coeff_r_r_r_r_1);
                    float2 value_r_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&xv_126));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[18];
                    value_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_127[0], rv_127[1]), value_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[22];
                    value_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_127[2], rv_127[3]), value_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[26];
                    value_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_127[4], rv_127[5]), value_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[30];
                    value_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_127[6], rv_127[7]), value_r_r_r_r_1);
                    unsigned int word_r_r_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_1.x, value_r_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_124] = word_r_r_r_r_r_r_r_1;
                    words[11] = word_r_r_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_1, rounded_pair_r_r_r_r_1, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_r_r_r_r_1.x;
                    sq_132[1] = squares_r_r_r_r_1.y;
                    float coeff_r_r_r_r_2 = post_coeff[7];
                    float2 _f32x2_mul_a_24 = make_float2(coeff_r_r_r_r_2, coeff_r_r_r_r_2);
                    float2 value_r_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&xv_126));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[19];
                    value_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_127[0], rv_127[1]), value_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[23];
                    value_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_127[2], rv_127[3]), value_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[27];
                    value_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_127[4], rv_127[5]), value_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[31];
                    value_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_127[6], rv_127[7]), value_r_r_r_r_2);
                    unsigned int word_r_r_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_2.x, value_r_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_124] = word_r_r_r_r_r_r_r_2;
                    words[15] = word_r_r_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_2, rounded_pair_r_r_r_r_2, make_float2(sq_132[0], sq_132[1]));
                    sq_132[0] = squares_r_r_r_r_2.x;
                    sq_132[1] = squares_r_r_r_r_2.y;
                    float _add_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(sq_132[0]), "f"(sq_132[1]));
                    float sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_add_25), "f"(hc_comp[1]));
                    float add_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(hc_acc[1]), "f"(sub_2));
                    float _sub_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(add_2), "f"(hc_acc[1]));
                    float _sub_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(_sub_27), "f"(sub_2));
                    hc_comp[1] = _sub_26;
                    hc_acc[1] = add_2;
                    unsigned int ast = a_seq % 4;
                    unsigned int _stage_empty_a = a_seq % 4;
                    unsigned int _parity_empty_a = a_seq / 4 % 2 ^ 1;
                    mbarrier_wait(empty_a_addr + (_stage_empty_a) * 8, _parity_empty_a);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x4.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"((unsigned int)tmem_a_tmem + ast * 32), "r"(*reinterpret_cast<const uint32_t*>(&words[0])), "r"(*reinterpret_cast<const uint32_t*>(&words[1])), "r"(*reinterpret_cast<const uint32_t*>(&words[2])), "r"(*reinterpret_cast<const uint32_t*>(&words[3])), "r"(*reinterpret_cast<const uint32_t*>(&words[4])), "r"(*reinterpret_cast<const uint32_t*>(&words[5])), "r"(*reinterpret_cast<const uint32_t*>(&words[6])), "r"(*reinterpret_cast<const uint32_t*>(&words[7])), "r"(*reinterpret_cast<const uint32_t*>(&words[8])), "r"(*reinterpret_cast<const uint32_t*>(&words[9])), "r"(*reinterpret_cast<const uint32_t*>(&words[10])), "r"(*reinterpret_cast<const uint32_t*>(&words[11])), "r"(*reinterpret_cast<const uint32_t*>(&words[12])), "r"(*reinterpret_cast<const uint32_t*>(&words[13])), "r"(*reinterpret_cast<const uint32_t*>(&words[14])), "r"(*reinterpret_cast<const uint32_t*>(&words[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(full_a_addr + (ast) * 8);
                    a_seq += 2;
                    unsigned int words_169[16];
                    unsigned int offset_170 = offs[4];
                    unsigned int x_word_171 = x_pairs[io_stage * 2048 + offset_170];
                    float2 xv_172 = make_float2(__uint_as_float(x_word_171 << 16), __uint_as_float(x_word_171 & 4294901760u));
                    float rv_173[8];
                    unsigned int word_174 = residual_pairs[io_stage * 8192 + offset_170];
                    rv_173[0] = __uint_as_float(word_174 << 16);
                    rv_173[1] = __uint_as_float(word_174 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_170];
                    rv_173[2] = __uint_as_float(word_r_r_r_r_r_r_r_r_0 << 16);
                    rv_173[3] = __uint_as_float(word_r_r_r_r_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_170];
                    rv_173[4] = __uint_as_float(word_r_r_r_r_r_r_r_r_1 << 16);
                    rv_173[5] = __uint_as_float(word_r_r_r_r_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_170];
                    rv_173[6] = __uint_as_float(word_r_r_r_r_r_r_r_r_2 << 16);
                    rv_173[7] = __uint_as_float(word_r_r_r_r_r_r_r_r_2 & 4294901760u);
                    float sq_178[2];
                    sq_178[0] = 0.0f;
                    sq_178[1] = 0.0f;
                    float coeff_179 = post_coeff[0];
                    float2 _f32x2_mul_a_28 = make_float2(coeff_179, coeff_179);
                    float2 value_180;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_180) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&xv_172));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[0];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_173[0], rv_173[1]), value_180);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[4];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_173[2], rv_173[3]), value_180);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[8];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_173[4], rv_173[5]), value_180);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[12];
                    value_180 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_173[6], rv_173[7]), value_180);
                    unsigned int word_185 = __as_u32(__float22bfloat162_rn(make_float2(value_180.x, value_180.y)));
                    residual_pairs[io_stage * 8192 + offset_170] = word_185;
                    words_169[0] = word_185;
                    float2 rounded_pair_186 = make_float2(__uint_as_float(word_185 << 16), __uint_as_float(word_185 & 4294901760u));
                    float2 squares_187 = fma_f32x2_rn_noftz(rounded_pair_186, rounded_pair_186, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_187.x;
                    sq_178[1] = squares_187.y;
                    float coeff_r_r_r_r_r_0 = post_coeff[1];
                    float2 _f32x2_mul_a_29 = make_float2(coeff_r_r_r_r_r_0, coeff_r_r_r_r_r_0);
                    float2 value_r_r_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&xv_172));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[1];
                    value_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_173[0], rv_173[1]), value_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[5];
                    value_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_173[2], rv_173[3]), value_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[9];
                    value_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_173[4], rv_173[5]), value_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[13];
                    value_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_173[6], rv_173[7]), value_r_r_r_r_r_0);
                    unsigned int word_r_r_r_r_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_0.x, value_r_r_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_170] = word_r_r_r_r_r_r_r_r_r_0;
                    words_169[4] = word_r_r_r_r_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_0, rounded_pair_r_r_r_r_r_0, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_r_r_r_r_r_0.x;
                    sq_178[1] = squares_r_r_r_r_r_0.y;
                    float coeff_r_r_r_r_r_1 = post_coeff[2];
                    float2 _f32x2_mul_a_30 = make_float2(coeff_r_r_r_r_r_1, coeff_r_r_r_r_r_1);
                    float2 value_r_r_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&xv_172));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[2];
                    value_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_173[0], rv_173[1]), value_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[6];
                    value_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_173[2], rv_173[3]), value_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[10];
                    value_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_173[4], rv_173[5]), value_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[14];
                    value_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_173[6], rv_173[7]), value_r_r_r_r_r_1);
                    unsigned int word_r_r_r_r_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_1.x, value_r_r_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_170] = word_r_r_r_r_r_r_r_r_r_1;
                    words_169[8] = word_r_r_r_r_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_1, rounded_pair_r_r_r_r_r_1, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_r_r_r_r_r_1.x;
                    sq_178[1] = squares_r_r_r_r_r_1.y;
                    float coeff_r_r_r_r_r_2 = post_coeff[3];
                    float2 _f32x2_mul_a_31 = make_float2(coeff_r_r_r_r_r_2, coeff_r_r_r_r_r_2);
                    float2 value_r_r_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&xv_172));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[3];
                    value_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_173[0], rv_173[1]), value_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[7];
                    value_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_173[2], rv_173[3]), value_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[11];
                    value_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_173[4], rv_173[5]), value_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[15];
                    value_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_173[6], rv_173[7]), value_r_r_r_r_r_2);
                    unsigned int word_r_r_r_r_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_2.x, value_r_r_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_170] = word_r_r_r_r_r_r_r_r_r_2;
                    words_169[12] = word_r_r_r_r_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_2, rounded_pair_r_r_r_r_r_2, make_float2(sq_178[0], sq_178[1]));
                    sq_178[0] = squares_r_r_r_r_r_2.x;
                    sq_178[1] = squares_r_r_r_r_r_2.y;
                    float _add_32;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(sq_178[0]), "f"(sq_178[1]));
                    float sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(_add_32), "f"(hc_comp[0]));
                    float add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(hc_acc[0]), "f"(sub_3));
                    float _sub_34;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(add_3), "f"(hc_acc[0]));
                    float _sub_33;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(_sub_34), "f"(sub_3));
                    hc_comp[0] = _sub_33;
                    hc_acc[0] = add_3;
                    unsigned int offset_215 = offs[5];
                    unsigned int x_word_216 = x_pairs[io_stage * 2048 + offset_215];
                    float2 xv_217 = make_float2(__uint_as_float(x_word_216 << 16), __uint_as_float(x_word_216 & 4294901760u));
                    float rv_218[8];
                    unsigned int word_219 = residual_pairs[io_stage * 8192 + offset_215];
                    rv_218[0] = __uint_as_float(word_219 << 16);
                    rv_218[1] = __uint_as_float(word_219 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_215];
                    rv_218[2] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_0 << 16);
                    rv_218[3] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_215];
                    rv_218[4] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_1 << 16);
                    rv_218[5] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_215];
                    rv_218[6] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_2 << 16);
                    rv_218[7] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u);
                    float sq_223[2];
                    sq_223[0] = 0.0f;
                    sq_223[1] = 0.0f;
                    float coeff_224 = post_coeff[4];
                    float2 _f32x2_mul_a_35 = make_float2(coeff_224, coeff_224);
                    float2 value_225;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_225) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&xv_217));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[16];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_218[0], rv_218[1]), value_225);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[20];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_218[2], rv_218[3]), value_225);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[24];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_218[4], rv_218[5]), value_225);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[28];
                    value_225 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_218[6], rv_218[7]), value_225);
                    unsigned int word_230 = __as_u32(__float22bfloat162_rn(make_float2(value_225.x, value_225.y)));
                    residual_pairs[io_stage * 8192 + offset_215] = word_230;
                    words_169[2] = word_230;
                    float2 rounded_pair_231 = make_float2(__uint_as_float(word_230 << 16), __uint_as_float(word_230 & 4294901760u));
                    float2 squares_232 = fma_f32x2_rn_noftz(rounded_pair_231, rounded_pair_231, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_232.x;
                    sq_223[1] = squares_232.y;
                    float coeff_r_r_r_r_r_r_0 = post_coeff[5];
                    float2 _f32x2_mul_a_36 = make_float2(coeff_r_r_r_r_r_r_0, coeff_r_r_r_r_r_r_0);
                    float2 value_r_r_r_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&xv_217));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[17];
                    value_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_218[0], rv_218[1]), value_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[21];
                    value_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_218[2], rv_218[3]), value_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[25];
                    value_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_218[4], rv_218[5]), value_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[29];
                    value_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_218[6], rv_218[7]), value_r_r_r_r_r_r_0);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_0.x, value_r_r_r_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_215] = word_r_r_r_r_r_r_r_r_r_r_r_0;
                    words_169[6] = word_r_r_r_r_r_r_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_0, rounded_pair_r_r_r_r_r_r_0, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_r_r_r_r_r_r_0.x;
                    sq_223[1] = squares_r_r_r_r_r_r_0.y;
                    float coeff_r_r_r_r_r_r_1 = post_coeff[6];
                    float2 _f32x2_mul_a_37 = make_float2(coeff_r_r_r_r_r_r_1, coeff_r_r_r_r_r_r_1);
                    float2 value_r_r_r_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&xv_217));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[18];
                    value_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_218[0], rv_218[1]), value_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[22];
                    value_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_218[2], rv_218[3]), value_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[26];
                    value_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_218[4], rv_218[5]), value_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[30];
                    value_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_218[6], rv_218[7]), value_r_r_r_r_r_r_1);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_1.x, value_r_r_r_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_215] = word_r_r_r_r_r_r_r_r_r_r_r_1;
                    words_169[10] = word_r_r_r_r_r_r_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_1, rounded_pair_r_r_r_r_r_r_1, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_r_r_r_r_r_r_1.x;
                    sq_223[1] = squares_r_r_r_r_r_r_1.y;
                    float coeff_r_r_r_r_r_r_2 = post_coeff[7];
                    float2 _f32x2_mul_a_38 = make_float2(coeff_r_r_r_r_r_r_2, coeff_r_r_r_r_r_r_2);
                    float2 value_r_r_r_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&xv_217));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[19];
                    value_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_218[0], rv_218[1]), value_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[23];
                    value_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_218[2], rv_218[3]), value_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[27];
                    value_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_218[4], rv_218[5]), value_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[31];
                    value_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_218[6], rv_218[7]), value_r_r_r_r_r_r_2);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_2.x, value_r_r_r_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_215] = word_r_r_r_r_r_r_r_r_r_r_r_2;
                    words_169[14] = word_r_r_r_r_r_r_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_2, rounded_pair_r_r_r_r_r_r_2, make_float2(sq_223[0], sq_223[1]));
                    sq_223[0] = squares_r_r_r_r_r_r_2.x;
                    sq_223[1] = squares_r_r_r_r_r_r_2.y;
                    float _add_39;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(sq_223[0]), "f"(sq_223[1]));
                    float sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(_add_39), "f"(hc_comp[1]));
                    float add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(hc_acc[1]), "f"(sub_4));
                    float _sub_41;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(add_4), "f"(hc_acc[1]));
                    float _sub_40;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(_sub_41), "f"(sub_4));
                    hc_comp[1] = _sub_40;
                    hc_acc[1] = add_4;
                    unsigned int offset_260 = offs[6];
                    unsigned int x_word_261 = x_pairs[io_stage * 2048 + offset_260];
                    float2 xv_262 = make_float2(__uint_as_float(x_word_261 << 16), __uint_as_float(x_word_261 & 4294901760u));
                    float rv_263[8];
                    unsigned int word_264 = residual_pairs[io_stage * 8192 + offset_260];
                    rv_263[0] = __uint_as_float(word_264 << 16);
                    rv_263[1] = __uint_as_float(word_264 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_260];
                    rv_263[2] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_0 << 16);
                    rv_263[3] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_260];
                    rv_263[4] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_1 << 16);
                    rv_263[5] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_260];
                    rv_263[6] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_2 << 16);
                    rv_263[7] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u);
                    float sq_268[2];
                    sq_268[0] = 0.0f;
                    sq_268[1] = 0.0f;
                    float coeff_269 = post_coeff[0];
                    float2 _f32x2_mul_a_42 = make_float2(coeff_269, coeff_269);
                    float2 value_270;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_270) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&xv_262));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[0];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_263[0], rv_263[1]), value_270);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[4];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_263[2], rv_263[3]), value_270);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[8];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_263[4], rv_263[5]), value_270);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[12];
                    value_270 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_263[6], rv_263[7]), value_270);
                    unsigned int word_275 = __as_u32(__float22bfloat162_rn(make_float2(value_270.x, value_270.y)));
                    residual_pairs[io_stage * 8192 + offset_260] = word_275;
                    words_169[1] = word_275;
                    float2 rounded_pair_276 = make_float2(__uint_as_float(word_275 << 16), __uint_as_float(word_275 & 4294901760u));
                    float2 squares_277 = fma_f32x2_rn_noftz(rounded_pair_276, rounded_pair_276, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_277.x;
                    sq_268[1] = squares_277.y;
                    float coeff_r_r_r_r_r_r_r_0 = post_coeff[1];
                    float2 _f32x2_mul_a_43 = make_float2(coeff_r_r_r_r_r_r_r_0, coeff_r_r_r_r_r_r_r_0);
                    float2 value_r_r_r_r_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&xv_262));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[1];
                    value_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_263[0], rv_263[1]), value_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[5];
                    value_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_263[2], rv_263[3]), value_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[9];
                    value_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_263[4], rv_263[5]), value_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[13];
                    value_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_263[6], rv_263[7]), value_r_r_r_r_r_r_r_0);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_0.x, value_r_r_r_r_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_260] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_0;
                    words_169[5] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_r_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_0, rounded_pair_r_r_r_r_r_r_r_0, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_r_r_r_r_r_r_r_0.x;
                    sq_268[1] = squares_r_r_r_r_r_r_r_0.y;
                    float coeff_r_r_r_r_r_r_r_1 = post_coeff[2];
                    float2 _f32x2_mul_a_44 = make_float2(coeff_r_r_r_r_r_r_r_1, coeff_r_r_r_r_r_r_r_1);
                    float2 value_r_r_r_r_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&xv_262));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[2];
                    value_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_263[0], rv_263[1]), value_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[6];
                    value_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_263[2], rv_263[3]), value_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[10];
                    value_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_263[4], rv_263[5]), value_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[14];
                    value_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_263[6], rv_263[7]), value_r_r_r_r_r_r_r_1);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_1.x, value_r_r_r_r_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_260] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_1;
                    words_169[9] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_r_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_1, rounded_pair_r_r_r_r_r_r_r_1, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_r_r_r_r_r_r_r_1.x;
                    sq_268[1] = squares_r_r_r_r_r_r_r_1.y;
                    float coeff_r_r_r_r_r_r_r_2 = post_coeff[3];
                    float2 _f32x2_mul_a_45 = make_float2(coeff_r_r_r_r_r_r_r_2, coeff_r_r_r_r_r_r_r_2);
                    float2 value_r_r_r_r_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&xv_262));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[3];
                    value_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_263[0], rv_263[1]), value_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[7];
                    value_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_263[2], rv_263[3]), value_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[11];
                    value_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_263[4], rv_263[5]), value_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[15];
                    value_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_263[6], rv_263[7]), value_r_r_r_r_r_r_r_2);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_2.x, value_r_r_r_r_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_260] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_2;
                    words_169[13] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_r_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_2, rounded_pair_r_r_r_r_r_r_r_2, make_float2(sq_268[0], sq_268[1]));
                    sq_268[0] = squares_r_r_r_r_r_r_r_2.x;
                    sq_268[1] = squares_r_r_r_r_r_r_r_2.y;
                    float _add_46;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(sq_268[0]), "f"(sq_268[1]));
                    float sub_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_5) : "f"(_add_46), "f"(hc_comp[0]));
                    float add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(hc_acc[0]), "f"(sub_5));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(add_5), "f"(hc_acc[0]));
                    float _sub_47;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(_sub_48), "f"(sub_5));
                    hc_comp[0] = _sub_47;
                    hc_acc[0] = add_5;
                    unsigned int offset_305 = offs[7];
                    unsigned int x_word_306 = x_pairs[io_stage * 2048 + offset_305];
                    float2 xv_307 = make_float2(__uint_as_float(x_word_306 << 16), __uint_as_float(x_word_306 & 4294901760u));
                    float rv_308[8];
                    unsigned int word_309 = residual_pairs[io_stage * 8192 + offset_305];
                    rv_308[0] = __uint_as_float(word_309 << 16);
                    rv_308[1] = __uint_as_float(word_309 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = residual_pairs[io_stage * 8192 + 2048 + offset_305];
                    rv_308[2] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 << 16);
                    rv_308[3] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = residual_pairs[io_stage * 8192 + 4096 + offset_305];
                    rv_308[4] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 << 16);
                    rv_308[5] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = residual_pairs[io_stage * 8192 + 6144 + offset_305];
                    rv_308[6] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 << 16);
                    rv_308[7] = __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u);
                    float sq_313[2];
                    sq_313[0] = 0.0f;
                    sq_313[1] = 0.0f;
                    float coeff_314 = post_coeff[4];
                    float2 _f32x2_mul_a_49 = make_float2(coeff_314, coeff_314);
                    float2 value_315;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_315) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&xv_307));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[16];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_308[0], rv_308[1]), value_315);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[20];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_308[2], rv_308[3]), value_315);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[24];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_308[4], rv_308[5]), value_315);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3 = comb_coeff[28];
                    value_315 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_3), make_float2(rv_308[6], rv_308[7]), value_315);
                    unsigned int word_320 = __as_u32(__float22bfloat162_rn(make_float2(value_315.x, value_315.y)));
                    residual_pairs[io_stage * 8192 + offset_305] = word_320;
                    words_169[3] = word_320;
                    float2 rounded_pair_321 = make_float2(__uint_as_float(word_320 << 16), __uint_as_float(word_320 & 4294901760u));
                    float2 squares_322 = fma_f32x2_rn_noftz(rounded_pair_321, rounded_pair_321, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_322.x;
                    sq_313[1] = squares_322.y;
                    float coeff_r_r_r_r_r_r_r_r_0 = post_coeff[5];
                    float2 _f32x2_mul_a_50 = make_float2(coeff_r_r_r_r_r_r_r_r_0, coeff_r_r_r_r_r_r_r_r_0);
                    float2 value_r_r_r_r_r_r_r_r_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&xv_307));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[17];
                    value_r_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_308[0], rv_308[1]), value_r_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[21];
                    value_r_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_308[2], rv_308[3]), value_r_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[25];
                    value_r_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_308[4], rv_308[5]), value_r_r_r_r_r_r_r_r_0);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = comb_coeff[29];
                    value_r_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0), make_float2(rv_308[6], rv_308[7]), value_r_r_r_r_r_r_r_r_0);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_r_0.x, value_r_r_r_r_r_r_r_r_0.y)));
                    residual_pairs[io_stage * 8192 + 2048 + offset_305] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0;
                    words_169[7] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0;
                    float2 rounded_pair_r_r_r_r_r_r_r_r_0 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_0 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_r_0 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_r_0, rounded_pair_r_r_r_r_r_r_r_r_0, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_r_r_r_r_r_r_r_r_0.x;
                    sq_313[1] = squares_r_r_r_r_r_r_r_r_0.y;
                    float coeff_r_r_r_r_r_r_r_r_1 = post_coeff[6];
                    float2 _f32x2_mul_a_51 = make_float2(coeff_r_r_r_r_r_r_r_r_1, coeff_r_r_r_r_r_r_r_r_1);
                    float2 value_r_r_r_r_r_r_r_r_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_r_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&xv_307));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[18];
                    value_r_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_308[0], rv_308[1]), value_r_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[22];
                    value_r_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_308[2], rv_308[3]), value_r_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[26];
                    value_r_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_308[4], rv_308[5]), value_r_r_r_r_r_r_r_r_1);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = comb_coeff[30];
                    value_r_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1), make_float2(rv_308[6], rv_308[7]), value_r_r_r_r_r_r_r_r_1);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_r_1.x, value_r_r_r_r_r_r_r_r_1.y)));
                    residual_pairs[io_stage * 8192 + 4096 + offset_305] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1;
                    words_169[11] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1;
                    float2 rounded_pair_r_r_r_r_r_r_r_r_1 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_1 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_r_1 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_r_1, rounded_pair_r_r_r_r_r_r_r_r_1, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_r_r_r_r_r_r_r_r_1.x;
                    sq_313[1] = squares_r_r_r_r_r_r_r_r_1.y;
                    float coeff_r_r_r_r_r_r_r_r_2 = post_coeff[7];
                    float2 _f32x2_mul_a_52 = make_float2(coeff_r_r_r_r_r_r_r_r_2, coeff_r_r_r_r_r_r_r_r_2);
                    float2 value_r_r_r_r_r_r_r_r_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_r_r_r_r_r_r_r_r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&xv_307));
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[19];
                    value_r_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_308[0], rv_308[1]), value_r_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[23];
                    value_r_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_308[2], rv_308[3]), value_r_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[27];
                    value_r_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_308[4], rv_308[5]), value_r_r_r_r_r_r_r_r_2);
                    float cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = comb_coeff[31];
                    value_r_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(make_float2(cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2, cc_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2), make_float2(rv_308[6], rv_308[7]), value_r_r_r_r_r_r_r_r_2);
                    unsigned int word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 = __as_u32(__float22bfloat162_rn(make_float2(value_r_r_r_r_r_r_r_r_2.x, value_r_r_r_r_r_r_r_r_2.y)));
                    residual_pairs[io_stage * 8192 + 6144 + offset_305] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2;
                    words_169[15] = word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2;
                    float2 rounded_pair_r_r_r_r_r_r_r_r_2 = make_float2(__uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 << 16), __uint_as_float(word_r_r_r_r_r_r_r_r_r_r_r_r_r_r_r_2 & 4294901760u));
                    float2 squares_r_r_r_r_r_r_r_r_2 = fma_f32x2_rn_noftz(rounded_pair_r_r_r_r_r_r_r_r_2, rounded_pair_r_r_r_r_r_r_r_r_2, make_float2(sq_313[0], sq_313[1]));
                    sq_313[0] = squares_r_r_r_r_r_r_r_r_2.x;
                    sq_313[1] = squares_r_r_r_r_r_r_r_r_2.y;
                    float _add_53;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(sq_313[0]), "f"(sq_313[1]));
                    float sub_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_6) : "f"(_add_53), "f"(hc_comp[1]));
                    float add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(hc_acc[1]), "f"(sub_6));
                    float _sub_55;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(add_6), "f"(hc_acc[1]));
                    float _sub_54;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(_sub_55), "f"(sub_6));
                    hc_comp[1] = _sub_54;
                    hc_acc[1] = add_6;
                    unsigned int ast_350 = a_seq % 4;
                    unsigned int _stage_empty_a_1 = a_seq % 4;
                    unsigned int _parity_empty_a_1 = a_seq / 4 % 2 ^ 1;
                    mbarrier_wait(empty_a_addr + (_stage_empty_a_1) * 8, _parity_empty_a_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x4.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"((unsigned int)tmem_a_tmem + ast_350 * 32), "r"(*reinterpret_cast<const uint32_t*>(&words_169[0])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[1])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[2])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[3])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[4])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[5])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[6])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[7])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[8])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[9])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[10])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[11])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[12])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[13])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[14])), "r"(*reinterpret_cast<const uint32_t*>(&words_169[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(full_a_addr + (ast_350) * 8);
                    a_seq += 2;
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(store_ready_addr + (io_stage) * 8);
                    }
                    io_seq_1 += 1;
                }
                mbarrier_wait(empty_stats_addr, _phase_empty_stats_0);
                _phase_empty_stats_0 ^= 1;
                unsigned int row_1_1 = first_row;
                float sub_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_7) : "f"(hc_acc[0]), "f"(hc_comp[0]));
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, sub_7, 2);
                float add_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(sub_7), "f"(_shfl_xor_sync_56));
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, add_7, 1);
                float add_1_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(add_7), "f"(_shfl_xor_sync_57));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[worker * 64 + row_1_1] = add_1_1;
                }
                unsigned int row_4 = first_row + 8;
                float sub_7_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_7_1) : "f"(hc_acc[1]), "f"(hc_comp[1]));
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, sub_7_1, 2);
                float add_7_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7_1) : "f"(sub_7_1), "f"(_shfl_xor_sync_58));
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, add_7_1, 1);
                float add_1_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_0) : "f"(add_7_1), "f"(_shfl_xor_sync_59));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[worker * 64 + row_4] = add_1_0;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(full_stats_addr);
                }
                __syncwarp();
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }
    // ---- Role: workspace ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // workspace_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
            unsigned int seq_1 = 0;
            unsigned int cseq_1 = 0;
            unsigned int first_row_1 = warp % 4 * 16 + lane_in_warp / 4;
            #pragma unroll 1
            for (unsigned int task_4 = (unsigned int)bid; task_4 < (num_tokens + 63) / 64 * num_splits; task_4 += num_ctas) {
                uint32_t fast_div_q_5 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_4), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_4);
                uint32_t fast_div_r_5 = (uint32_t)(task_4) - fast_div_q_5 * (uint32_t)(split_div.divisor);
                if (warp % 4 == 0) {
                    unsigned long long base = (launch_epochs[bid] + 1) * 128;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + fast_div_q_5) * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(base)) >= static_cast<unsigned long long>((unsigned long long)num_splits));
                    }
                }
                float acc[12];
                float comp[12];
                acc[0] = 0.0f;
                comp[0] = 0.0f;
                acc[1] = 0.0f;
                comp[1] = 0.0f;
                acc[2] = 0.0f;
                comp[2] = 0.0f;
                acc[3] = 0.0f;
                comp[3] = 0.0f;
                acc[4] = 0.0f;
                comp[4] = 0.0f;
                acc[5] = 0.0f;
                comp[5] = 0.0f;
                acc[6] = 0.0f;
                comp[6] = 0.0f;
                acc[7] = 0.0f;
                comp[7] = 0.0f;
                acc[8] = 0.0f;
                comp[8] = 0.0f;
                acc[9] = 0.0f;
                comp[9] = 0.0f;
                acc[10] = 0.0f;
                comp[10] = 0.0f;
                acc[11] = 0.0f;
                comp[11] = 0.0f;
                #pragma unroll 1
                for (unsigned int chunk = 0; chunk < (_fast_div_q_0 + ((fast_div_r_5 < _fast_div_r_0) ? 1 : 0)) * 2; chunk++) {
                    unsigned int cstage = cseq_1 % 2;
                    unsigned int _stage_big_full = cseq_1 % 2;
                    unsigned int _parity_big_full = cseq_1 / 2 % 2;
                    mbarrier_wait(big_full_addr + (_stage_big_full) * 8, _parity_big_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    float tmem_load[8];
#else
                    float _tmem_load_0[8];
#endif
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
#if __CUDA_ARCH__ == 1070
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7]))
                        : "r"((unsigned int)tmem_acc_big + cstage * 24));
                    float tmem_load_1[4];
#else
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                        : "r"((unsigned int)tmem_acc_big + cstage * 48));
                    float _tmem_load_1[4];
#endif
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
#if __CUDA_ARCH__ == 1070
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage * 24 + 16)));
#else
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage * 48 + 16)));
                    float _tmem_load_2[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                        : "r"((unsigned int)tmem_acc_big + (cstage * 48 + 24)));
                    float _tmem_load_3[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage * 48 + 24 + 16)));
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(big_empty_addr + (cstage) * 8);
                    }
                    float sub_8;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8) : "f"(tmem_load[0]), "f"(comp[0]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8) : "f"(_tmem_load_0[0]), "f"(comp[0]));
#endif
                    float add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(acc[0]), "f"(sub_8));
                    float _sub_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(add_8), "f"(acc[0]));
                    float _sub_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_sub_1), "f"(sub_8));
                    comp[0] = _sub_0;
                    acc[0] = add_8;
                    float sub_8_1;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8_1) : "f"(tmem_load[1]), "f"(comp[1]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8_1) : "f"(_tmem_load_0[1]), "f"(comp[1]));
#endif
                    float add_8_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8_1) : "f"(acc[1]), "f"(sub_8_1));
                    float _sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(add_8_1), "f"(acc[1]));
                    float _sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_sub_3), "f"(sub_8_1));
                    comp[1] = _sub_2;
                    acc[1] = add_8_1;
                    float sub_9;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_9) : "f"(tmem_load[2]), "f"(comp[2]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_9) : "f"(_tmem_load_0[2]), "f"(comp[2]));
#endif
                    float add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(acc[2]), "f"(sub_9));
                    float _sub_5_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_1) : "f"(add_9), "f"(acc[2]));
                    float _sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_sub_5_1), "f"(sub_9));
                    comp[2] = _sub_4;
                    acc[2] = add_9;
                    float sub_10;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_10) : "f"(tmem_load[3]), "f"(comp[3]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_10) : "f"(_tmem_load_0[3]), "f"(comp[3]));
#endif
                    float add_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(acc[3]), "f"(sub_10));
                    float _sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(add_10), "f"(acc[3]));
                    float _sub_6_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_1) : "f"(_sub_7), "f"(sub_10));
                    comp[3] = _sub_6_1;
                    acc[3] = add_10;
                    float sub_11;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_11) : "f"(tmem_load[4]), "f"(comp[4]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_11) : "f"(_tmem_load_0[4]), "f"(comp[4]));
#endif
                    float add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(acc[4]), "f"(sub_11));
                    float _sub_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(add_11), "f"(acc[4]));
                    float _sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(_sub_9), "f"(sub_11));
                    comp[4] = _sub_8;
                    acc[4] = add_11;
                    float sub_12;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_12) : "f"(tmem_load[5]), "f"(comp[5]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_12) : "f"(_tmem_load_0[5]), "f"(comp[5]));
#endif
                    float add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(acc[5]), "f"(sub_12));
                    float _sub_11;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(add_12), "f"(acc[5]));
                    float _sub_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(_sub_11), "f"(sub_12));
                    comp[5] = _sub_10;
                    acc[5] = add_12;
                    float sub_13;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_13) : "f"(tmem_load[6]), "f"(comp[6]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_13) : "f"(_tmem_load_0[6]), "f"(comp[6]));
#endif
                    float add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(acc[6]), "f"(sub_13));
                    float _sub_13_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_1) : "f"(add_13), "f"(acc[6]));
                    float _sub_12_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_1) : "f"(_sub_13_1), "f"(sub_13));
                    comp[6] = _sub_12_1;
                    acc[6] = add_13;
                    float sub_14;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_14) : "f"(tmem_load[7]), "f"(comp[7]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_14) : "f"(_tmem_load_0[7]), "f"(comp[7]));
#endif
                    float add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(acc[7]), "f"(sub_14));
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(add_14), "f"(acc[7]));
                    float _sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(_sub_15), "f"(sub_14));
                    comp[7] = _sub_14;
                    acc[7] = add_14;
                    float sub_15;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_15) : "f"(tmem_load_1[0]), "f"(comp[8]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_15) : "f"(_tmem_load_1[0]), "f"(comp[8]));
#endif
                    float add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(acc[8]), "f"(sub_15));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(add_15), "f"(acc[8]));
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(_sub_17), "f"(sub_15));
                    comp[8] = _sub_16;
                    acc[8] = add_15;
                    float sub_16;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_16) : "f"(tmem_load_1[1]), "f"(comp[9]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_16) : "f"(_tmem_load_1[1]), "f"(comp[9]));
#endif
                    float add_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(acc[9]), "f"(sub_16));
                    float _sub_19_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_1) : "f"(add_16), "f"(acc[9]));
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(_sub_19_1), "f"(sub_16));
                    comp[9] = _sub_18;
                    acc[9] = add_16;
                    float sub_17;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_17) : "f"(tmem_load_1[2]), "f"(comp[10]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_17) : "f"(_tmem_load_1[2]), "f"(comp[10]));
#endif
                    float add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(acc[10]), "f"(sub_17));
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(add_17), "f"(acc[10]));
                    float _sub_20_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_1) : "f"(_sub_21), "f"(sub_17));
                    comp[10] = _sub_20_1;
                    acc[10] = add_17;
                    float sub_18;
#if __CUDA_ARCH__ == 1070
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_18) : "f"(tmem_load_1[3]), "f"(comp[11]));
#else
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_18) : "f"(_tmem_load_1[3]), "f"(comp[11]));
#endif
                    float add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(acc[11]), "f"(sub_18));
                    float _sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(add_18), "f"(acc[11]));
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(_sub_23), "f"(sub_18));
                    comp[11] = _sub_22;
                    acc[11] = add_18;
#if __CUDA_ARCH__ == 1070
#else
                    float sub_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_19) : "f"(_tmem_load_2[0]), "f"(comp[0]));
                    float add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(acc[0]), "f"(sub_19));
                    float _sub_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(add_19), "f"(acc[0]));
                    float _sub_24;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_sub_25), "f"(sub_19));
                    comp[0] = _sub_24;
                    acc[0] = add_19;
                    float sub_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_20) : "f"(_tmem_load_2[1]), "f"(comp[1]));
                    float add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(acc[1]), "f"(sub_20));
                    float _sub_27_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_1) : "f"(add_20), "f"(acc[1]));
                    float _sub_26_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_1) : "f"(_sub_27_1), "f"(sub_20));
                    comp[1] = _sub_26_1;
                    acc[1] = add_20;
                    float sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_21) : "f"(_tmem_load_2[2]), "f"(comp[2]));
                    float add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(acc[2]), "f"(sub_21));
                    float _sub_29;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(add_21), "f"(acc[2]));
                    float _sub_28;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_sub_29), "f"(sub_21));
                    comp[2] = _sub_28;
                    acc[2] = add_21;
                    float sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_22) : "f"(_tmem_load_2[3]), "f"(comp[3]));
                    float add_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(acc[3]), "f"(sub_22));
                    float _sub_31;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(add_22), "f"(acc[3]));
                    float _sub_30;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_sub_31), "f"(sub_22));
                    comp[3] = _sub_30;
                    acc[3] = add_22;
                    float sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_23) : "f"(_tmem_load_2[4]), "f"(comp[4]));
                    float add_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(acc[4]), "f"(sub_23));
                    float _sub_33_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_1) : "f"(add_23), "f"(acc[4]));
                    float _sub_32;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33_1), "f"(sub_23));
                    comp[4] = _sub_32;
                    acc[4] = add_23;
                    float sub_24;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_24) : "f"(_tmem_load_2[5]), "f"(comp[5]));
                    float add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(acc[5]), "f"(sub_24));
                    float _sub_35;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(add_24), "f"(acc[5]));
                    float _sub_34_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_1) : "f"(_sub_35), "f"(sub_24));
                    comp[5] = _sub_34_1;
                    acc[5] = add_24;
                    float sub_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_25) : "f"(_tmem_load_2[6]), "f"(comp[6]));
                    float add_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_25) : "f"(acc[6]), "f"(sub_25));
                    float _sub_37;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(add_25), "f"(acc[6]));
                    float _sub_36;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_sub_37), "f"(sub_25));
                    comp[6] = _sub_36;
                    acc[6] = add_25;
                    float sub_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_26) : "f"(_tmem_load_2[7]), "f"(comp[7]));
                    float add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_26) : "f"(acc[7]), "f"(sub_26));
                    float _sub_39;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(add_26), "f"(acc[7]));
                    float _sub_38;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_sub_39), "f"(sub_26));
                    comp[7] = _sub_38;
                    acc[7] = add_26;
                    float sub_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_27) : "f"(_tmem_load_3[0]), "f"(comp[8]));
                    float add_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_27) : "f"(acc[8]), "f"(sub_27));
                    float _sub_41_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_1) : "f"(add_27), "f"(acc[8]));
                    float _sub_40_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_1) : "f"(_sub_41_1), "f"(sub_27));
                    comp[8] = _sub_40_1;
                    acc[8] = add_27;
                    float sub_28;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_28) : "f"(_tmem_load_3[1]), "f"(comp[9]));
                    float add_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_28) : "f"(acc[9]), "f"(sub_28));
                    float _sub_43;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(add_28), "f"(acc[9]));
                    float _sub_42;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_sub_43), "f"(sub_28));
                    comp[9] = _sub_42;
                    acc[9] = add_28;
                    float sub_29;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_29) : "f"(_tmem_load_3[2]), "f"(comp[10]));
                    float add_29;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_29) : "f"(acc[10]), "f"(sub_29));
                    float _sub_45;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(add_29), "f"(acc[10]));
                    float _sub_44;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(sub_29));
                    comp[10] = _sub_44;
                    acc[10] = add_29;
                    float sub_30;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_30) : "f"(_tmem_load_3[3]), "f"(comp[11]));
                    float add_30;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30) : "f"(acc[11]), "f"(sub_30));
                    float _sub_47_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_1) : "f"(add_30), "f"(acc[11]));
                    float _sub_46;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47_1), "f"(sub_30));
                    comp[11] = _sub_46;
                    acc[11] = add_30;
#endif
                    cseq_1 += 1;
                }
                unsigned int ustage_1 = seq_1 % 2;
                unsigned int _stage_small_full = seq_1 % 2;
                unsigned int _parity_small_full = seq_1 / 2 % 2;
                mbarrier_wait(small_full_addr + (_stage_small_full) * 8, _parity_small_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                float tmem_load_2[8];
#else
                float tmem_load[8];
#endif
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7]))
#else
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7]))
#endif
                    : "r"((unsigned int)tmem_acc_small + ustage_1 * 24));
#if __CUDA_ARCH__ == 1070
                float tmem_load_1_1[4];
#else
                float tmem_load_1[4];
#endif
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3]))
#else
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3]))
#endif
                    : "r"((unsigned int)tmem_acc_small + (ustage_1 * 24 + 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    mbarrier_arrive(small_empty_addr + (ustage_1) * 8);
                }
#if __CUDA_ARCH__ == 1070
                float sub_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_19) : "f"(tmem_load_2[0]), "f"(comp[0]));
                float add_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(acc[0]), "f"(sub_19));
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(add_19), "f"(acc[0]));
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_sub_25), "f"(sub_19));
                comp[0] = _sub_24;
                acc[0] = add_19;
                float sub_19_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_19_1) : "f"(tmem_load_2[1]), "f"(comp[1]));
                float add_19_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19_1) : "f"(acc[1]), "f"(sub_19_1));
                float _sub_27_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_1) : "f"(add_19_1), "f"(acc[1]));
                float _sub_26_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_1) : "f"(_sub_27_1), "f"(sub_19_1));
                comp[1] = _sub_26_1;
                acc[1] = add_19_1;
                float sub_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_20) : "f"(tmem_load_2[2]), "f"(comp[2]));
                float add_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(acc[2]), "f"(sub_20));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(add_20), "f"(acc[2]));
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_sub_29), "f"(sub_20));
                comp[2] = _sub_28;
                acc[2] = add_20;
                float sub_21;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_21) : "f"(tmem_load_2[3]), "f"(comp[3]));
                float add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(acc[3]), "f"(sub_21));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(add_21), "f"(acc[3]));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_sub_31), "f"(sub_21));
                comp[3] = _sub_30;
                acc[3] = add_21;
                float sub_22;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_22) : "f"(tmem_load_2[4]), "f"(comp[4]));
                float add_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(acc[4]), "f"(sub_22));
                float _sub_33_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_1) : "f"(add_22), "f"(acc[4]));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33_1), "f"(sub_22));
                comp[4] = _sub_32;
                acc[4] = add_22;
                float sub_23;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_23) : "f"(tmem_load_2[5]), "f"(comp[5]));
                float add_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(acc[5]), "f"(sub_23));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(add_23), "f"(acc[5]));
                float _sub_34_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_1) : "f"(_sub_35), "f"(sub_23));
                comp[5] = _sub_34_1;
                acc[5] = add_23;
                float sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_24) : "f"(tmem_load_2[6]), "f"(comp[6]));
                float add_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(acc[6]), "f"(sub_24));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(add_24), "f"(acc[6]));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_sub_37), "f"(sub_24));
                comp[6] = _sub_36;
                acc[6] = add_24;
                float sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_25) : "f"(tmem_load_2[7]), "f"(comp[7]));
                float add_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_25) : "f"(acc[7]), "f"(sub_25));
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(add_25), "f"(acc[7]));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_sub_39), "f"(sub_25));
                comp[7] = _sub_38;
                acc[7] = add_25;
                float sub_26;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_26) : "f"(tmem_load_1_1[0]), "f"(comp[8]));
                float add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_26) : "f"(acc[8]), "f"(sub_26));
                float _sub_41_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_1) : "f"(add_26), "f"(acc[8]));
                float _sub_40_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_1) : "f"(_sub_41_1), "f"(sub_26));
                comp[8] = _sub_40_1;
                acc[8] = add_26;
                float sub_27;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_27) : "f"(tmem_load_1_1[1]), "f"(comp[9]));
                float add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_27) : "f"(acc[9]), "f"(sub_27));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(add_27), "f"(acc[9]));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_sub_43), "f"(sub_27));
                comp[9] = _sub_42;
                acc[9] = add_27;
                float sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_28) : "f"(tmem_load_1_1[2]), "f"(comp[10]));
                float add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_28) : "f"(acc[10]), "f"(sub_28));
                float _sub_45;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(add_28), "f"(acc[10]));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(sub_28));
                comp[10] = _sub_44;
                acc[10] = add_28;
                float sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_29) : "f"(tmem_load_1_1[3]), "f"(comp[11]));
                float add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_29) : "f"(acc[11]), "f"(sub_29));
                float _sub_47_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_1) : "f"(add_29), "f"(acc[11]));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47_1), "f"(sub_29));
                comp[11] = _sub_46;
                acc[11] = add_29;
#else
                float sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_31) : "f"(tmem_load[0]), "f"(comp[0]));
                float add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(acc[0]), "f"(sub_31));
                float _sub_49;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(add_31), "f"(acc[0]));
                float _sub_48_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48_1) : "f"(_sub_49), "f"(sub_31));
                comp[0] = _sub_48_1;
                acc[0] = add_31;
                float sub_31_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_31_1) : "f"(tmem_load[1]), "f"(comp[1]));
                float add_31_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31_1) : "f"(acc[1]), "f"(sub_31_1));
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(add_31_1), "f"(acc[1]));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(_sub_51), "f"(sub_31_1));
                comp[1] = _sub_50;
                acc[1] = add_31_1;
                float sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_32) : "f"(tmem_load[2]), "f"(comp[2]));
                float add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_32) : "f"(acc[2]), "f"(sub_32));
                float _sub_53;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(add_32), "f"(acc[2]));
                float _sub_52;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(_sub_53), "f"(sub_32));
                comp[2] = _sub_52;
                acc[2] = add_32;
                float sub_33;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_33) : "f"(tmem_load[3]), "f"(comp[3]));
                float add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_33) : "f"(acc[3]), "f"(sub_33));
                float _sub_55_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55_1) : "f"(add_33), "f"(acc[3]));
                float _sub_54_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54_1) : "f"(_sub_55_1), "f"(sub_33));
                comp[3] = _sub_54_1;
                acc[3] = add_33;
                float sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_34) : "f"(tmem_load[4]), "f"(comp[4]));
                float add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_34) : "f"(acc[4]), "f"(sub_34));
                float _sub_57;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(add_34), "f"(acc[4]));
                float _sub_56;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(_sub_57), "f"(sub_34));
                comp[4] = _sub_56;
                acc[4] = add_34;
                float sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_35) : "f"(tmem_load[5]), "f"(comp[5]));
                float add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_35) : "f"(acc[5]), "f"(sub_35));
                float _sub_59;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(add_35), "f"(acc[5]));
                float _sub_58;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(_sub_59), "f"(sub_35));
                comp[5] = _sub_58;
                acc[5] = add_35;
                float sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_36) : "f"(tmem_load[6]), "f"(comp[6]));
                float add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_36) : "f"(acc[6]), "f"(sub_36));
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(add_36), "f"(acc[6]));
                float _sub_60;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(_sub_61), "f"(sub_36));
                comp[6] = _sub_60;
                acc[6] = add_36;
                float sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_37) : "f"(tmem_load[7]), "f"(comp[7]));
                float add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_37) : "f"(acc[7]), "f"(sub_37));
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(add_37), "f"(acc[7]));
                float _sub_62;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(_sub_63), "f"(sub_37));
                comp[7] = _sub_62;
                acc[7] = add_37;
                float sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_38) : "f"(tmem_load_1[0]), "f"(comp[8]));
                float add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_38) : "f"(acc[8]), "f"(sub_38));
                float _sub_65;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(add_38), "f"(acc[8]));
                float _sub_64;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(_sub_65), "f"(sub_38));
                comp[8] = _sub_64;
                acc[8] = add_38;
                float sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_39) : "f"(tmem_load_1[1]), "f"(comp[9]));
                float add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_39) : "f"(acc[9]), "f"(sub_39));
                float _sub_67;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(add_39), "f"(acc[9]));
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(_sub_67), "f"(sub_39));
                comp[9] = _sub_66;
                acc[9] = add_39;
                float sub_40;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_40) : "f"(tmem_load_1[2]), "f"(comp[10]));
                float add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_40) : "f"(acc[10]), "f"(sub_40));
                float _sub_69;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(add_40), "f"(acc[10]));
                float _sub_68;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(_sub_69), "f"(sub_40));
                comp[10] = _sub_68;
                acc[10] = add_40;
                float sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_41) : "f"(tmem_load_1[3]), "f"(comp[11]));
                float add_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_41) : "f"(acc[11]), "f"(sub_41));
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(add_41), "f"(acc[11]));
                float _sub_70;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(_sub_71), "f"(sub_41));
                comp[11] = _sub_70;
                acc[11] = add_41;
#endif
                unsigned int row_2_0 = first_row_1;
                float values_1_0[2];
                float total_1_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_0) : "f"(acc[0]), "f"(comp[0]));
                values_1_0[0] = ((fast_div_q_5 * 64 + row_2_0 < num_tokens) ? total_1_0 : 0.0f);
                float total_1_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_1) : "f"(acc[1]), "f"(comp[1]));
                values_1_0[1] = ((fast_div_q_5 * 64 + row_2_0 < num_tokens) ? total_1_1 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_0[0 + 0], values_1_0[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_0 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_2_1 = first_row_1 + 8;
                float values_1_1[2];
                float total_1_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_2) : "f"(acc[2]), "f"(comp[2]));
                values_1_1[0] = ((fast_div_q_5 * 64 + row_2_1 < num_tokens) ? total_1_2 : 0.0f);
                float total_1_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_3) : "f"(acc[3]), "f"(comp[3]));
                values_1_1[1] = ((fast_div_q_5 * 64 + row_2_1 < num_tokens) ? total_1_3 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_1[0 + 0], values_1_1[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_1 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_2_2 = first_row_1;
                float values_1_2[2];
                float total_1_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_4) : "f"(acc[4]), "f"(comp[4]));
                values_1_2[0] = ((fast_div_q_5 * 64 + row_2_2 < num_tokens) ? total_1_4 : 0.0f);
                float total_1_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_5) : "f"(acc[5]), "f"(comp[5]));
                values_1_2[1] = ((fast_div_q_5 * 64 + row_2_2 < num_tokens) ? total_1_5 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_2[0 + 0], values_1_2[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_2 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_2_3 = first_row_1 + 8;
                float values_1_3[2];
                float total_1_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_6) : "f"(acc[6]), "f"(comp[6]));
                values_1_3[0] = ((fast_div_q_5 * 64 + row_2_3 < num_tokens) ? total_1_6 : 0.0f);
                float total_1_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_7) : "f"(acc[7]), "f"(comp[7]));
                values_1_3[1] = ((fast_div_q_5 * 64 + row_2_3 < num_tokens) ? total_1_7 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_3[0 + 0], values_1_3[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_3 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_2_4 = first_row_1;
                float values_1_4[2];
                float total_1_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_8) : "f"(acc[8]), "f"(comp[8]));
                values_1_4[0] = ((fast_div_q_5 * 64 + row_2_4 < num_tokens) ? total_1_8 : 0.0f);
                float total_1_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_9) : "f"(acc[9]), "f"(comp[9]));
                values_1_4[1] = ((fast_div_q_5 * 64 + row_2_4 < num_tokens) ? total_1_9 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_4[0 + 0], values_1_4[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_4 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_2_5 = first_row_1 + 8;
                float values_1_5[2];
                float total_1_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_10) : "f"(acc[10]), "f"(comp[10]));
                values_1_5[0] = ((fast_div_q_5 * 64 + row_2_5 < num_tokens) ? total_1_10 : 0.0f);
                float total_1_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_1_11) : "f"(acc[11]), "f"(comp[11]));
                values_1_5[1] = ((fast_div_q_5 * 64 + row_2_5 < num_tokens) ? total_1_11 : 0.0f);
                {
                    float2 _v2 = make_float2(values_1_5[0 + 0], values_1_5[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_4 * 1536 + (unsigned long long)(row_2_5 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                int _parity_full_stats = (int)((unsigned int)(int)seq_1 % 2);
                mbarrier_wait(full_stats_addr, _parity_full_stats);
                if (lane_in_warp < 16) {
                    unsigned int row_0_1 = warp % 4 * 16 + lane_in_warp;
#if __CUDA_ARCH__ == 1070
                    float _add_48;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(hc_sums[row_0_1]), "f"(hc_sums[64 + row_0_1]));
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)task_4 * 64 + (unsigned long long)row_0_1] = _add_48;
#else
                    float _add_72;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(hc_sums[row_0_1]), "f"(hc_sums[64 + row_0_1]));
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)task_4 * 64 + (unsigned long long)row_0_1] = _add_72;
#endif
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(empty_stats_addr);
                    mbarrier_arrive(mix_arrival_addr);
                }
                if (warp % 4 == 0) {
                    int _parity_mix_arrival = (int)((unsigned int)(int)seq_1 % 2);
                    mbarrier_wait(mix_arrival_addr, _parity_mix_arrival);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + fast_div_q_5) * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                        #elif defined(__CUDA_ARCH__)
                        #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                        #endif
                    }
                }
                seq_1 += 1;
            }
        }
    }
    // ---- Role: mix ----
    if (warp >= 16 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // mix_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (warp == 18) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + lane_in_warp * 16), "l"(rmsnorm_weight + (lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (512 + lane_in_warp * 16)), "l"(rmsnorm_weight + (256 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (1024 + lane_in_warp * 16)), "l"(rmsnorm_weight + (512 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (1536 + lane_in_warp * 16)), "l"(rmsnorm_weight + (768 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (2048 + lane_in_warp * 16)), "l"(rmsnorm_weight + (1024 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (2560 + lane_in_warp * 16)), "l"(rmsnorm_weight + (1280 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (3072 + lane_in_warp * 16)), "l"(rmsnorm_weight + (1536 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (3584 + lane_in_warp * 16)), "l"(rmsnorm_weight + (1792 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (4096 + lane_in_warp * 16)), "l"(rmsnorm_weight + (2048 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (4608 + lane_in_warp * 16)), "l"(rmsnorm_weight + (2304 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (5120 + lane_in_warp * 16)), "l"(rmsnorm_weight + (2560 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (5632 + lane_in_warp * 16)), "l"(rmsnorm_weight + (2816 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (6144 + lane_in_warp * 16)), "l"(rmsnorm_weight + (3072 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (6656 + lane_in_warp * 16)), "l"(rmsnorm_weight + (3328 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (7168 + lane_in_warp * 16)), "l"(rmsnorm_weight + (3584 + lane_in_warp * 8)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(weight_words_addr + (7680 + lane_in_warp * 16)), "l"(rmsnorm_weight + (3840 + lane_in_warp * 8)));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(weight_full_addr) : "memory");
            }
            if (warp == 17) {
                if (elect_sync()) {
                    for (unsigned int mb = (unsigned int)bid; mb < (num_tokens + 63) / 64; mb += num_ctas) {
                        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb) * 16))), "l"(static_cast<unsigned long long>((launch_epochs[bid] + 1) * 128)) : "memory");
                    }
                    if (bid == 0) {
                        unsigned int other_parity = 1 - (unsigned int)(launch_epochs[bid] % 2);
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + ((unsigned long long)other_parity * 16 * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 1) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 2) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 3) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 4) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 5) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 6) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 7) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 8) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 9) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 10) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 11) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 12) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 13) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 14) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)other_parity * 16 + 15) * 16)))), "r"(static_cast<unsigned int>(0)) : "memory");
                    }
                }
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned long long epoch_base = (launch_epochs[bid] + 1) * 128;
            unsigned int first = (unsigned int)bid * 4 + warp % 4;
            unsigned int step = num_ctas * 4;
            unsigned int stop = num_tokens;
            unsigned int first_0 = first + (unsigned int)(warp / 4 - 4) * num_ctas * 4;
            unsigned int step_1 = (((num_tokens + 63) / 64 * num_splits <= 3 * (unsigned int)num_ctas) ? num_ctas * 8 : num_ctas * 4);
            unsigned int stop_2 = (((num_tokens + 63) / 64 * num_splits <= 3 * (unsigned int)num_ctas || warp < 20) ? num_tokens : 0);
#if __CUDA_ARCH__ == 1070
            unsigned int first_3 = (((num_tokens + 63) / 64 * num_splits <= num_ctas) ? (unsigned int)bid + (unsigned int)(warp - 16) * num_ctas : first_0);
#else
            unsigned int first_3 = (((num_tokens + 63) / 64 * num_splits <= num_ctas) ? (unsigned int)bid + ((warp < 20) ? (unsigned int)(warp - 16) : (((num_tokens + 63) / 64 * num_splits <= num_ctas && num_tokens <= num_ctas * 4) ? (unsigned int)(warp - 18) % 4 : (unsigned int)(warp - 16))) * num_ctas : first_0);
#endif
            #pragma unroll 1
            for (unsigned int token = first_3; token < stop_2; token += step_1) {
                unsigned int mb_1 = token / 64;
                float sc = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
#if __CUDA_ARCH__ == 1070
                unsigned long long target = epoch_base + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_1) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target)) >= static_cast<unsigned long long>(1));
#else
                unsigned int flag = ((warp >= 20 && ((num_tokens + 63) / 64 * num_splits <= num_ctas && num_tokens <= num_ctas * 4)) ? 1 : 0);
                if (flag == 0) {
                    unsigned long long target = epoch_base + (unsigned long long)num_splits;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_1) * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target)) >= static_cast<unsigned long long>(1));
                    }
#endif
                }
                float s = 0.0f;
                float q = 0.0f;
#if __CUDA_ARCH__ == 1070
                if ((num_tokens + 63) / 64 * num_splits <= num_ctas) {
                    unsigned int row_5 = token % 64;
                    unsigned int first_task = mb_1 * num_splits;
                    float sums[32];
                    sums[0] = 0.0f;
                    sums[1] = 0.0f;
                    sums[2] = 0.0f;
                    sums[3] = 0.0f;
                    sums[4] = 0.0f;
                    sums[5] = 0.0f;
                    sums[6] = 0.0f;
                    sums[7] = 0.0f;
                    sums[8] = 0.0f;
                    sums[9] = 0.0f;
                    sums[10] = 0.0f;
                    sums[11] = 0.0f;
                    sums[12] = 0.0f;
                    sums[13] = 0.0f;
                    sums[14] = 0.0f;
                    sums[15] = 0.0f;
                    sums[16] = 0.0f;
                    sums[17] = 0.0f;
                    sums[18] = 0.0f;
                    sums[19] = 0.0f;
                    sums[20] = 0.0f;
                    sums[21] = 0.0f;
                    sums[22] = 0.0f;
                    sums[23] = 0.0f;
                    sums[24] = 0.0f;
                    sums[25] = 0.0f;
                    sums[26] = 0.0f;
                    sums[27] = 0.0f;
                    sums[28] = 0.0f;
                    sums[29] = 0.0f;
                    sums[30] = 0.0f;
                    sums[31] = 0.0f;
                    unsigned int split = lane_in_warp * 2;
                    unsigned int safe_split = ((split < num_splits) ? split : 0);
                    unsigned long long record = ((unsigned long long)first_task + (unsigned long long)safe_split) * 1536 + (unsigned long long)(row_5 * 24);
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record + 0);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
#else
                if ((num_tokens + 63) / 64 * num_splits <= num_ctas && num_tokens <= num_ctas * 4) {
                    unsigned int w = ((flag == 1) ? (unsigned int)(warp - 18) : (unsigned int)(warp - 16)) % 4;
                    unsigned int col0 = flag * 12;
                    asm volatile("barrier.sync %0, 64;" :: "r"(4 + w) : "memory");
                    float total = 0.0f;
                    if (num_splits > 22) {
                        unsigned int row_5 = token % 64;
                        unsigned int first_task = mb_1 * num_splits;
                        float sums[16];
                        float errs[16];
                        sums[0] = 0.0f;
                        errs[0] = 0.0f;
                        sums[1] = 0.0f;
                        errs[1] = 0.0f;
                        sums[2] = 0.0f;
                        errs[2] = 0.0f;
                        sums[3] = 0.0f;
                        errs[3] = 0.0f;
                        sums[4] = 0.0f;
                        errs[4] = 0.0f;
                        sums[5] = 0.0f;
                        errs[5] = 0.0f;
                        sums[6] = 0.0f;
                        errs[6] = 0.0f;
                        sums[7] = 0.0f;
                        errs[7] = 0.0f;
                        sums[8] = 0.0f;
                        errs[8] = 0.0f;
                        sums[9] = 0.0f;
                        errs[9] = 0.0f;
                        sums[10] = 0.0f;
                        errs[10] = 0.0f;
                        sums[11] = 0.0f;
                        errs[11] = 0.0f;
                        sums[12] = 0.0f;
                        errs[12] = 0.0f;
                        sums[13] = 0.0f;
                        errs[13] = 0.0f;
                        sums[14] = 0.0f;
                        errs[14] = 0.0f;
                        sums[15] = 0.0f;
                        errs[15] = 0.0f;
                        unsigned int split = lane_in_warp * 2;
                        unsigned int safe_split = ((split < num_splits) ? split : 0);
                        unsigned long long record = ((unsigned long long)first_task + (unsigned long long)safe_split) * 1536 + (unsigned long long)(row_5 * 24);
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)col0) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        float value_11_0 = ((split < num_splits) ? _vec_load_0[0] : 0.0f);
                        sums[0] = value_11_0;
                        float value_11_1 = ((split < num_splits) ? _vec_load_0[1] : 0.0f);
                        sums[8] = value_11_1;
                        float value_11_2 = ((split < num_splits) ? _vec_load_0[2] : 0.0f);
                        sums[4] = value_11_2;
                        float value_11_3 = ((split < num_splits) ? _vec_load_0[3] : 0.0f);
                        sums[12] = value_11_3;
                        float _vec_load_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_1[0 + 0] = _v4.x;
                            _vec_load_1[0 + 1] = _v4.y;
                            _vec_load_1[0 + 2] = _v4.z;
                            _vec_load_1[0 + 3] = _v4.w;
                        }
                        float value_11_4 = ((split < num_splits) ? _vec_load_1[0] : 0.0f);
                        sums[2] = value_11_4;
                        float value_11_5 = ((split < num_splits) ? _vec_load_1[1] : 0.0f);
                        sums[10] = value_11_5;
                        float value_11_6 = ((split < num_splits) ? _vec_load_1[2] : 0.0f);
                        sums[6] = value_11_6;
                        float value_11_7 = ((split < num_splits) ? _vec_load_1[3] : 0.0f);
                        sums[14] = value_11_7;
                        float _vec_load_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_2[0 + 0] = _v4.x;
                            _vec_load_2[0 + 1] = _v4.y;
                            _vec_load_2[0 + 2] = _v4.z;
                            _vec_load_2[0 + 3] = _v4.w;
                        }
                        float value_11_8 = ((split < num_splits) ? _vec_load_2[0] : 0.0f);
                        sums[1] = value_11_8;
                        float value_11_9 = ((split < num_splits) ? _vec_load_2[1] : 0.0f);
                        sums[9] = value_11_9;
                        float value_11_10 = ((split < num_splits) ? _vec_load_2[2] : 0.0f);
                        sums[5] = value_11_10;
                        float value_11_11 = ((split < num_splits) ? _vec_load_2[3] : 0.0f);
                        sums[13] = value_11_11;
                        float q_word = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split * 64) + (unsigned long long)row_5];
                        float q_value = ((split < num_splits) ? q_word : 0.0f);
                        sums[3] = q_value;
                        unsigned int split_11 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_12 = ((split_11 < num_splits) ? split_11 : 0);
                        unsigned long long record_13 = ((unsigned long long)first_task + (unsigned long long)safe_split_12) * 1536 + (unsigned long long)(row_5 * 24);
                        float _vec_load_3[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)col0) + 0);
                            _vec_load_3[0 + 0] = _v4.x;
                            _vec_load_3[0 + 1] = _v4.y;
                            _vec_load_3[0 + 2] = _v4.z;
                            _vec_load_3[0 + 3] = _v4.w;
                        }
                        float value_11_12 = ((split_11 < num_splits) ? _vec_load_3[0] : 0.0f);
                        float s_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(sums[0]), "f"(value_11_12));
                        float bb;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb) : "f"(s_0), "f"(sums[0]));
                        float _sub_5_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_2) : "f"(s_0), "f"(bb));
                        float _sub_4_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4_1) : "f"(sums[0]), "f"(_sub_5_2));
                        float _sub_6_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_2) : "f"(value_11_12), "f"(bb));
                        float e;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_4_1), "f"(_sub_6_2));
                        sums[0] = s_0;
                        errs[0] = e;
                        float value_11_13 = ((split_11 < num_splits) ? _vec_load_3[1] : 0.0f);
                        float s_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(sums[8]), "f"(value_11_13));
                        float bb_0;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_0) : "f"(s_1), "f"(sums[8]));
                        float _sub_8_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8_1) : "f"(s_1), "f"(bb_0));
                        float _sub_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7_1) : "f"(sums[8]), "f"(_sub_8_1));
                        float _sub_9_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9_1) : "f"(value_11_13), "f"(bb_0));
                        float e_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_7_1), "f"(_sub_9_1));
                        sums[8] = s_1;
                        errs[8] = e_0;
                        float value_11_14 = ((split_11 < num_splits) ? _vec_load_3[2] : 0.0f);
                        float s_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(sums[4]), "f"(value_11_14));
                        float bb_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_1) : "f"(s_2), "f"(sums[4]));
                        float _sub_11_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11_1) : "f"(s_2), "f"(bb_1));
                        float _sub_10_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10_1) : "f"(sums[4]), "f"(_sub_11_1));
                        float _sub_12_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_2) : "f"(value_11_14), "f"(bb_1));
                        float e_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_10_1), "f"(_sub_12_2));
                        sums[4] = s_2;
                        errs[4] = e_1;
                        float value_11_15 = ((split_11 < num_splits) ? _vec_load_3[3] : 0.0f);
                        float s_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(sums[12]), "f"(value_11_15));
                        float bb_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_2) : "f"(s_3), "f"(sums[12]));
                        float _sub_14_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_1) : "f"(s_3), "f"(bb_2));
                        float _sub_13_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_2) : "f"(sums[12]), "f"(_sub_14_1));
                        float _sub_15_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_1) : "f"(value_11_15), "f"(bb_2));
                        float e_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_13_2), "f"(_sub_15_1));
                        sums[12] = s_3;
                        errs[12] = e_2;
                        float _vec_load_4[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_4[0 + 0] = _v4.x;
                            _vec_load_4[0 + 1] = _v4.y;
                            _vec_load_4[0 + 2] = _v4.z;
                            _vec_load_4[0 + 3] = _v4.w;
                        }
                        float value_11_16 = ((split_11 < num_splits) ? _vec_load_4[0] : 0.0f);
                        float s_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(sums[2]), "f"(value_11_16));
                        float bb_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_3) : "f"(s_4), "f"(sums[2]));
                        float _sub_18_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_1) : "f"(s_4), "f"(bb_3));
                        float _sub_17_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_1) : "f"(sums[2]), "f"(_sub_18_1));
                        float _sub_19_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_2) : "f"(value_11_16), "f"(bb_3));
                        float e_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_17_1), "f"(_sub_19_2));
                        sums[2] = s_4;
                        errs[2] = e_3;
                        float value_11_17 = ((split_11 < num_splits) ? _vec_load_4[1] : 0.0f);
                        float s_5;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(sums[10]), "f"(value_11_17));
                        float bb_4;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_4) : "f"(s_5), "f"(sums[10]));
                        float _sub_21_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_1) : "f"(s_5), "f"(bb_4));
                        float _sub_20_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_2) : "f"(sums[10]), "f"(_sub_21_1));
                        float _sub_22_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_1) : "f"(value_11_17), "f"(bb_4));
                        float e_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_20_2), "f"(_sub_22_1));
                        sums[10] = s_5;
                        errs[10] = e_4;
                        float value_11_18 = ((split_11 < num_splits) ? _vec_load_4[2] : 0.0f);
                        float s_6;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(sums[6]), "f"(value_11_18));
                        float bb_5;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_5) : "f"(s_6), "f"(sums[6]));
                        float _sub_24_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_1) : "f"(s_6), "f"(bb_5));
                        float _sub_23_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_1) : "f"(sums[6]), "f"(_sub_24_1));
                        float _sub_25_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_1) : "f"(value_11_18), "f"(bb_5));
                        float e_5;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_23_1), "f"(_sub_25_1));
                        sums[6] = s_6;
                        errs[6] = e_5;
                        float value_11_19 = ((split_11 < num_splits) ? _vec_load_4[3] : 0.0f);
                        float s_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(sums[14]), "f"(value_11_19));
                        float bb_6;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_6) : "f"(s_7), "f"(sums[14]));
                        float _sub_27_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_2) : "f"(s_7), "f"(bb_6));
                        float _sub_26_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_2) : "f"(sums[14]), "f"(_sub_27_2));
                        float _sub_28_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_1) : "f"(value_11_19), "f"(bb_6));
                        float e_6;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_26_2), "f"(_sub_28_1));
                        sums[14] = s_7;
                        errs[14] = e_6;
                        float _vec_load_5[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_5[0 + 0] = _v4.x;
                            _vec_load_5[0 + 1] = _v4.y;
                            _vec_load_5[0 + 2] = _v4.z;
                            _vec_load_5[0 + 3] = _v4.w;
                        }
                        float value_11_20 = ((split_11 < num_splits) ? _vec_load_5[0] : 0.0f);
                        float s_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(sums[1]), "f"(value_11_20));
                        float bb_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_7) : "f"(s_8), "f"(sums[1]));
                        float _sub_31_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_1) : "f"(s_8), "f"(bb_7));
                        float _sub_30_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_1) : "f"(sums[1]), "f"(_sub_31_1));
                        float _sub_32_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_1) : "f"(value_11_20), "f"(bb_7));
                        float e_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_30_1), "f"(_sub_32_1));
                        sums[1] = s_8;
                        errs[1] = e_7;
                        float value_11_21 = ((split_11 < num_splits) ? _vec_load_5[1] : 0.0f);
                        float s_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(sums[9]), "f"(value_11_21));
                        float bb_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_8) : "f"(s_9), "f"(sums[9]));
                        float _sub_34_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_2) : "f"(s_9), "f"(bb_8));
                        float _sub_33_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_2) : "f"(sums[9]), "f"(_sub_34_2));
                        float _sub_35_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35_1) : "f"(value_11_21), "f"(bb_8));
                        float e_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_33_2), "f"(_sub_35_1));
                        sums[9] = s_9;
                        errs[9] = e_8;
                        float value_11_22 = ((split_11 < num_splits) ? _vec_load_5[2] : 0.0f);
                        float s_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(sums[5]), "f"(value_11_22));
                        float bb_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_9) : "f"(s_10), "f"(sums[5]));
                        float _sub_37_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37_1) : "f"(s_10), "f"(bb_9));
                        float _sub_36_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36_1) : "f"(sums[5]), "f"(_sub_37_1));
                        float _sub_38_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38_1) : "f"(value_11_22), "f"(bb_9));
                        float e_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_36_1), "f"(_sub_38_1));
                        sums[5] = s_10;
                        errs[5] = e_9;
                        float value_11_23 = ((split_11 < num_splits) ? _vec_load_5[3] : 0.0f);
                        float s_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(sums[13]), "f"(value_11_23));
                        float bb_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_10) : "f"(s_11), "f"(sums[13]));
                        float _sub_40_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_2) : "f"(s_11), "f"(bb_10));
                        float _sub_39_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39_1) : "f"(sums[13]), "f"(_sub_40_2));
                        float _sub_41_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_2) : "f"(value_11_23), "f"(bb_10));
                        float e_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_39_1), "f"(_sub_41_2));
                        sums[13] = s_11;
                        errs[13] = e_10;
                        float q_word_60 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split_12 * 64) + (unsigned long long)row_5];
                        float q_value_61 = ((split_11 < num_splits) ? q_word_60 : 0.0f);
                        float s_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(sums[3]), "f"(q_value_61));
                        float bb_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_11) : "f"(s_12), "f"(sums[3]));
                        float _sub_43_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43_1) : "f"(s_12), "f"(bb_11));
                        float _sub_42_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42_1) : "f"(sums[3]), "f"(_sub_43_1));
                        float _sub_44_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44_1) : "f"(q_value_61), "f"(bb_11));
                        float e_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_42_1), "f"(_sub_44_1));
                        sums[3] = s_12;
                        errs[3] = e_11;
                        float half_s1[8];
                        float half_e1[8];
                        float keep_1_0 = ((lane_in_warp % 2 == 1) ? sums[8] : sums[0]);
                        float send_1_0 = ((lane_in_warp % 2 == 1) ? sums[0] : sums[8]);
                        float keep_err_1_0 = ((lane_in_warp % 2 == 1) ? errs[8] : errs[0]);
                        float send_err_1_0 = ((lane_in_warp % 2 == 1) ? errs[0] : errs[8]);
                        float received_1_0 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
                        float received_err_1_0 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_0, 1);
                        float s_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(keep_1_0), "f"(received_1_0));
                        float bb_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_12) : "f"(s_13), "f"(keep_1_0));
                        float _sub_46_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46_1) : "f"(s_13), "f"(bb_12));
                        float _sub_45_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45_1) : "f"(keep_1_0), "f"(_sub_46_1));
                        float _sub_47_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47_2) : "f"(received_1_0), "f"(bb_12));
                        float e_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_45_1), "f"(_sub_47_2));
                        half_s1[0] = s_13;
                        float _add_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(keep_err_1_0), "f"(received_err_1_0));
                        float _add_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(e_12));
                        half_e1[0] = _add_48;
                        float keep_1_1 = ((lane_in_warp % 2 == 1) ? sums[9] : sums[1]);
                        float send_1_1 = ((lane_in_warp % 2 == 1) ? sums[1] : sums[9]);
                        float keep_err_1_1 = ((lane_in_warp % 2 == 1) ? errs[9] : errs[1]);
                        float send_err_1_1 = ((lane_in_warp % 2 == 1) ? errs[1] : errs[9]);
                        float received_1_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
                        float received_err_1_1 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_1, 1);
                        float s_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(keep_1_1), "f"(received_1_1));
                        float bb_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_13) : "f"(s_14), "f"(keep_1_1));
                        float _sub_51_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51_1) : "f"(s_14), "f"(bb_13));
                        float _sub_50_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50_1) : "f"(keep_1_1), "f"(_sub_51_1));
                        float _sub_52_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52_1) : "f"(received_1_1), "f"(bb_13));
                        float e_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_50_1), "f"(_sub_52_1));
                        half_s1[1] = s_14;
                        float _add_54;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(keep_err_1_1), "f"(received_err_1_1));
                        float _add_53_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(_add_54), "f"(e_13));
                        half_e1[1] = _add_53_1;
                        float keep_1_2 = ((lane_in_warp % 2 == 1) ? sums[10] : sums[2]);
                        float send_1_2 = ((lane_in_warp % 2 == 1) ? sums[2] : sums[10]);
                        float keep_err_1_2 = ((lane_in_warp % 2 == 1) ? errs[10] : errs[2]);
                        float send_err_1_2 = ((lane_in_warp % 2 == 1) ? errs[2] : errs[10]);
                        float received_1_2 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
                        float received_err_1_2 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_2, 1);
                        float s_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(keep_1_2), "f"(received_1_2));
                        float bb_14;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_14) : "f"(s_15), "f"(keep_1_2));
                        float _sub_56_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56_1) : "f"(s_15), "f"(bb_14));
                        float _sub_55_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55_2) : "f"(keep_1_2), "f"(_sub_56_1));
                        float _sub_57_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57_1) : "f"(received_1_2), "f"(bb_14));
                        float e_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_55_2), "f"(_sub_57_1));
                        half_s1[2] = s_15;
                        float _add_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(keep_err_1_2), "f"(received_err_1_2));
                        float _add_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_add_59), "f"(e_14));
                        half_e1[2] = _add_58;
                        float keep_1_3 = ((lane_in_warp % 2 == 1) ? sums[11] : sums[3]);
                        float send_1_3 = ((lane_in_warp % 2 == 1) ? sums[3] : sums[11]);
                        float keep_err_1_3 = ((lane_in_warp % 2 == 1) ? errs[11] : errs[3]);
                        float send_err_1_3 = ((lane_in_warp % 2 == 1) ? errs[3] : errs[11]);
                        float received_1_3 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
                        float received_err_1_3 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_3, 1);
                        float s_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(keep_1_3), "f"(received_1_3));
                        float bb_15;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_15) : "f"(s_16), "f"(keep_1_3));
                        float _sub_61_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61_1) : "f"(s_16), "f"(bb_15));
                        float _sub_60_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60_1) : "f"(keep_1_3), "f"(_sub_61_1));
                        float _sub_62_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62_1) : "f"(received_1_3), "f"(bb_15));
                        float e_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_60_1), "f"(_sub_62_1));
                        half_s1[3] = s_16;
                        float _add_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(keep_err_1_3), "f"(received_err_1_3));
                        float _add_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_64), "f"(e_15));
                        half_e1[3] = _add_63;
                        float keep_1_4 = ((lane_in_warp % 2 == 1) ? sums[12] : sums[4]);
                        float send_1_4 = ((lane_in_warp % 2 == 1) ? sums[4] : sums[12]);
                        float keep_err_1_4 = ((lane_in_warp % 2 == 1) ? errs[12] : errs[4]);
                        float send_err_1_4 = ((lane_in_warp % 2 == 1) ? errs[4] : errs[12]);
                        float received_1_4 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
                        float received_err_1_4 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_4, 1);
                        float s_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(keep_1_4), "f"(received_1_4));
                        float bb_16;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_16) : "f"(s_17), "f"(keep_1_4));
                        float _sub_66_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66_1) : "f"(s_17), "f"(bb_16));
                        float _sub_65_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65_1) : "f"(keep_1_4), "f"(_sub_66_1));
                        float _sub_67_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67_1) : "f"(received_1_4), "f"(bb_16));
                        float e_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_65_1), "f"(_sub_67_1));
                        half_s1[4] = s_17;
                        float _add_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(keep_err_1_4), "f"(received_err_1_4));
                        float _add_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(e_16));
                        half_e1[4] = _add_68;
                        float keep_1_5 = ((lane_in_warp % 2 == 1) ? sums[13] : sums[5]);
                        float send_1_5 = ((lane_in_warp % 2 == 1) ? sums[5] : sums[13]);
                        float keep_err_1_5 = ((lane_in_warp % 2 == 1) ? errs[13] : errs[5]);
                        float send_err_1_5 = ((lane_in_warp % 2 == 1) ? errs[5] : errs[13]);
                        float received_1_5 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
                        float received_err_1_5 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_5, 1);
                        float s_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(keep_1_5), "f"(received_1_5));
                        float bb_17;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_17) : "f"(s_18), "f"(keep_1_5));
                        float _sub_71_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71_1) : "f"(s_18), "f"(bb_17));
                        float _sub_70_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70_1) : "f"(keep_1_5), "f"(_sub_71_1));
                        float _sub_72;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(received_1_5), "f"(bb_17));
                        float e_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_70_1), "f"(_sub_72));
                        half_s1[5] = s_18;
                        float _add_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(keep_err_1_5), "f"(received_err_1_5));
                        float _add_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(e_17));
                        half_e1[5] = _add_73;
                        float keep_1_6 = ((lane_in_warp % 2 == 1) ? sums[14] : sums[6]);
                        float send_1_6 = ((lane_in_warp % 2 == 1) ? sums[6] : sums[14]);
                        float keep_err_1_6 = ((lane_in_warp % 2 == 1) ? errs[14] : errs[6]);
                        float send_err_1_6 = ((lane_in_warp % 2 == 1) ? errs[6] : errs[14]);
                        float received_1_6 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
                        float received_err_1_6 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_6, 1);
                        float s_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(keep_1_6), "f"(received_1_6));
                        float bb_18;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_18) : "f"(s_19), "f"(keep_1_6));
                        float _sub_76;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(s_19), "f"(bb_18));
                        float _sub_75;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(keep_1_6), "f"(_sub_76));
                        float _sub_77;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(received_1_6), "f"(bb_18));
                        float e_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_75), "f"(_sub_77));
                        half_s1[6] = s_19;
                        float _add_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(keep_err_1_6), "f"(received_err_1_6));
                        float _add_78;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_79), "f"(e_18));
                        half_e1[6] = _add_78;
                        float keep_1_7 = ((lane_in_warp % 2 == 1) ? sums[15] : sums[7]);
                        float send_1_7 = ((lane_in_warp % 2 == 1) ? sums[7] : sums[15]);
                        float keep_err_1_7 = ((lane_in_warp % 2 == 1) ? errs[15] : errs[7]);
                        float send_err_1_7 = ((lane_in_warp % 2 == 1) ? errs[7] : errs[15]);
                        float received_1_7 = __shfl_xor_sync(0xFFFFFFFF, send_1_7, 1);
                        float received_err_1_7 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_7, 1);
                        float s_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(keep_1_7), "f"(received_1_7));
                        float bb_19;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_19) : "f"(s_20), "f"(keep_1_7));
                        float _sub_81;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(s_20), "f"(bb_19));
                        float _sub_80;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(keep_1_7), "f"(_sub_81));
                        float _sub_82;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(received_1_7), "f"(bb_19));
                        float e_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_80), "f"(_sub_82));
                        half_s1[7] = s_20;
                        float _add_84;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(keep_err_1_7), "f"(received_err_1_7));
                        float _add_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_84), "f"(e_19));
                        half_e1[7] = _add_83;
                        float half_s2[4];
                        float half_e2[4];
                        float keep_1_8 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[4] : half_s1[0]);
                        float send_1_8 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[0] : half_s1[4]);
                        float keep_err_1_8 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[4] : half_e1[0]);
                        float send_err_1_8 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[0] : half_e1[4]);
                        float received_1_8 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 2);
                        float received_err_1_8 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_8, 2);
                        float s_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(keep_1_8), "f"(received_1_8));
                        float bb_20;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_20) : "f"(s_21), "f"(keep_1_8));
                        float _sub_86;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(s_21), "f"(bb_20));
                        float _sub_85;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(keep_1_8), "f"(_sub_86));
                        float _sub_87;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(received_1_8), "f"(bb_20));
                        float e_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_85), "f"(_sub_87));
                        half_s2[0] = s_21;
                        float _add_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(keep_err_1_8), "f"(received_err_1_8));
                        float _add_88;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(e_20));
                        half_e2[0] = _add_88;
                        float keep_1_9 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[5] : half_s1[1]);
                        float send_1_9 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[1] : half_s1[5]);
                        float keep_err_1_9 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[5] : half_e1[1]);
                        float send_err_1_9 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[1] : half_e1[5]);
                        float received_1_9 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 2);
                        float received_err_1_9 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_9, 2);
                        float s_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(keep_1_9), "f"(received_1_9));
                        float bb_21;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_21) : "f"(s_22), "f"(keep_1_9));
                        float _sub_91;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(s_22), "f"(bb_21));
                        float _sub_90;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(keep_1_9), "f"(_sub_91));
                        float _sub_92;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(received_1_9), "f"(bb_21));
                        float e_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_90), "f"(_sub_92));
                        half_s2[1] = s_22;
                        float _add_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(keep_err_1_9), "f"(received_err_1_9));
                        float _add_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_94), "f"(e_21));
                        half_e2[1] = _add_93;
                        float keep_1_10 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[6] : half_s1[2]);
                        float send_1_10 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[2] : half_s1[6]);
                        float keep_err_1_10 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[6] : half_e1[2]);
                        float send_err_1_10 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[2] : half_e1[6]);
                        float received_1_10 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 2);
                        float received_err_1_10 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_10, 2);
                        float s_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(keep_1_10), "f"(received_1_10));
                        float bb_22;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_22) : "f"(s_23), "f"(keep_1_10));
                        float _sub_96;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(s_23), "f"(bb_22));
                        float _sub_95;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(keep_1_10), "f"(_sub_96));
                        float _sub_97;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(received_1_10), "f"(bb_22));
                        float e_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_22) : "f"(_sub_95), "f"(_sub_97));
                        half_s2[2] = s_23;
                        float _add_99;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(keep_err_1_10), "f"(received_err_1_10));
                        float _add_98;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_add_99), "f"(e_22));
                        half_e2[2] = _add_98;
                        float keep_1_11 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[7] : half_s1[3]);
                        float send_1_11 = ((lane_in_warp / 2 % 2 == 1) ? half_s1[3] : half_s1[7]);
                        float keep_err_1_11 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[7] : half_e1[3]);
                        float send_err_1_11 = ((lane_in_warp / 2 % 2 == 1) ? half_e1[3] : half_e1[7]);
                        float received_1_11 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 2);
                        float received_err_1_11 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_11, 2);
                        float s_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(keep_1_11), "f"(received_1_11));
                        float bb_23;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_23) : "f"(s_24), "f"(keep_1_11));
                        float _sub_101;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(s_24), "f"(bb_23));
                        float _sub_100;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(keep_1_11), "f"(_sub_101));
                        float _sub_102;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(received_1_11), "f"(bb_23));
                        float e_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_23) : "f"(_sub_100), "f"(_sub_102));
                        half_s2[3] = s_24;
                        float _add_104;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(keep_err_1_11), "f"(received_err_1_11));
                        float _add_103;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_104), "f"(e_23));
                        half_e2[3] = _add_103;
                        float half_s3[2];
                        float half_e3[2];
                        float keep_1_12 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[2] : half_s2[0]);
                        float send_1_12 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[0] : half_s2[2]);
                        float keep_err_1_12 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[2] : half_e2[0]);
                        float send_err_1_12 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[0] : half_e2[2]);
                        float received_1_12 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 4);
                        float received_err_1_12 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_12, 4);
                        float s_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(keep_1_12), "f"(received_1_12));
                        float bb_24;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_24) : "f"(s_25), "f"(keep_1_12));
                        float _sub_106;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(s_25), "f"(bb_24));
                        float _sub_105;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(keep_1_12), "f"(_sub_106));
                        float _sub_107;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_107) : "f"(received_1_12), "f"(bb_24));
                        float e_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_24) : "f"(_sub_105), "f"(_sub_107));
                        half_s3[0] = s_25;
                        float _add_109;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(keep_err_1_12), "f"(received_err_1_12));
                        float _add_108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_109), "f"(e_24));
                        half_e3[0] = _add_108;
                        float keep_1_13 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[3] : half_s2[1]);
                        float send_1_13 = ((lane_in_warp / 4 % 2 == 1) ? half_s2[1] : half_s2[3]);
                        float keep_err_1_13 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[3] : half_e2[1]);
                        float send_err_1_13 = ((lane_in_warp / 4 % 2 == 1) ? half_e2[1] : half_e2[3]);
                        float received_1_13 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 4);
                        float received_err_1_13 = __shfl_xor_sync(0xFFFFFFFF, send_err_1_13, 4);
                        float s_26;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(keep_1_13), "f"(received_1_13));
                        float bb_25;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_25) : "f"(s_26), "f"(keep_1_13));
                        float _sub_111;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(s_26), "f"(bb_25));
                        float _sub_110;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_110) : "f"(keep_1_13), "f"(_sub_111));
                        float _sub_112;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(received_1_13), "f"(bb_25));
                        float e_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_25) : "f"(_sub_110), "f"(_sub_112));
                        half_s3[1] = s_26;
                        float _add_114;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(keep_err_1_13), "f"(received_err_1_13));
                        float _add_113;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_114), "f"(e_25));
                        half_e3[1] = _add_113;
                        float half_s4[1];
                        float half_e4[1];
                        float keep_185 = ((lane_in_warp / 8 % 2 == 1) ? half_s3[1] : half_s3[0]);
                        float send_186 = ((lane_in_warp / 8 % 2 == 1) ? half_s3[0] : half_s3[1]);
                        float keep_err_187 = ((lane_in_warp / 8 % 2 == 1) ? half_e3[1] : half_e3[0]);
                        float send_err_188 = ((lane_in_warp / 8 % 2 == 1) ? half_e3[0] : half_e3[1]);
                        float received_189 = __shfl_xor_sync(0xFFFFFFFF, send_186, 8);
                        float received_err_190 = __shfl_xor_sync(0xFFFFFFFF, send_err_188, 8);
                        float s_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(keep_185), "f"(received_189));
                        float bb_26;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_26) : "f"(s_27), "f"(keep_185));
                        float _sub_116;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_116) : "f"(s_27), "f"(bb_26));
                        float _sub_115;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(keep_185), "f"(_sub_116));
                        float _sub_117;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_117) : "f"(received_189), "f"(bb_26));
                        float e_26;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_26) : "f"(_sub_115), "f"(_sub_117));
                        half_s4[0] = s_27;
                        float _add_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(keep_err_187), "f"(received_err_190));
                        float _add_118;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_119), "f"(e_26));
                        half_e4[0] = _add_118;
                        float received5 = __shfl_xor_sync(0xFFFFFFFF, half_s4[0], 16);
                        float received5_err = __shfl_xor_sync(0xFFFFFFFF, half_e4[0], 16);
                        float s_28;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(half_s4[0]), "f"(received5));
                        float bb_27;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_27) : "f"(s_28), "f"(half_s4[0]));
                        float _sub_121;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(s_28), "f"(bb_27));
                        float _sub_120;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(half_s4[0]), "f"(_sub_121));
                        float _sub_122;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(received5), "f"(bb_27));
                        float e_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_27) : "f"(_sub_120), "f"(_sub_122));
                        float _add_123;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(half_e4[0]), "f"(received5_err));
                        float total_err;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_err) : "f"(_add_123), "f"(e_27));
                        float total_197;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_197) : "f"(s_28), "f"(total_err));
                        total = total_197;
                    } else {
                        unsigned int row_8 = token % 64;
                        unsigned int first_task_1 = mb_1 * num_splits;
                        float sums_1[16];
                        sums_1[0] = 0.0f;
                        sums_1[1] = 0.0f;
                        sums_1[2] = 0.0f;
                        sums_1[3] = 0.0f;
                        sums_1[4] = 0.0f;
                        sums_1[5] = 0.0f;
                        sums_1[6] = 0.0f;
                        sums_1[7] = 0.0f;
                        sums_1[8] = 0.0f;
                        sums_1[9] = 0.0f;
                        sums_1[10] = 0.0f;
                        sums_1[11] = 0.0f;
                        sums_1[12] = 0.0f;
                        sums_1[13] = 0.0f;
                        sums_1[14] = 0.0f;
                        sums_1[15] = 0.0f;
                        unsigned int split_1 = lane_in_warp * 2;
                        unsigned int safe_split_1 = ((split_1 < num_splits) ? split_1 : 0);
                        unsigned long long record_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_1) * 1536 + (unsigned long long)(row_8 * 24);
                        float _vec_load_6[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)col0) + 0);
                            _vec_load_6[0 + 0] = _v4.x;
                            _vec_load_6[0 + 1] = _v4.y;
                            _vec_load_6[0 + 2] = _v4.z;
                            _vec_load_6[0 + 3] = _v4.w;
                        }
                        float value_11_24 = ((split_1 < num_splits) ? _vec_load_6[0] : 0.0f);
                        sums_1[0] = value_11_24;
                        float value_11_25 = ((split_1 < num_splits) ? _vec_load_6[1] : 0.0f);
                        sums_1[8] = value_11_25;
                        float value_11_26 = ((split_1 < num_splits) ? _vec_load_6[2] : 0.0f);
                        sums_1[4] = value_11_26;
                        float value_11_27 = ((split_1 < num_splits) ? _vec_load_6[3] : 0.0f);
                        sums_1[12] = value_11_27;
                        float _vec_load_7[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_7[0 + 0] = _v4.x;
                            _vec_load_7[0 + 1] = _v4.y;
                            _vec_load_7[0 + 2] = _v4.z;
                            _vec_load_7[0 + 3] = _v4.w;
                        }
                        float value_11_28 = ((split_1 < num_splits) ? _vec_load_7[0] : 0.0f);
                        sums_1[2] = value_11_28;
                        float value_11_29 = ((split_1 < num_splits) ? _vec_load_7[1] : 0.0f);
                        sums_1[10] = value_11_29;
                        float value_11_30 = ((split_1 < num_splits) ? _vec_load_7[2] : 0.0f);
                        sums_1[6] = value_11_30;
                        float value_11_31 = ((split_1 < num_splits) ? _vec_load_7[3] : 0.0f);
                        sums_1[14] = value_11_31;
                        float _vec_load_8[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_8[0 + 0] = _v4.x;
                            _vec_load_8[0 + 1] = _v4.y;
                            _vec_load_8[0 + 2] = _v4.z;
                            _vec_load_8[0 + 3] = _v4.w;
                        }
                        float value_11_32 = ((split_1 < num_splits) ? _vec_load_8[0] : 0.0f);
                        sums_1[1] = value_11_32;
                        float value_11_33 = ((split_1 < num_splits) ? _vec_load_8[1] : 0.0f);
                        sums_1[9] = value_11_33;
                        float value_11_34 = ((split_1 < num_splits) ? _vec_load_8[2] : 0.0f);
                        sums_1[5] = value_11_34;
                        float value_11_35 = ((split_1 < num_splits) ? _vec_load_8[3] : 0.0f);
                        sums_1[13] = value_11_35;
                        float q_word_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_1 * 64) + (unsigned long long)row_8];
                        float q_value_1 = ((split_1 < num_splits) ? q_word_1 : 0.0f);
                        sums_1[3] = q_value_1;
                        unsigned int split_11_1 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_12_1 = ((split_11_1 < num_splits) ? split_11_1 : 0);
                        unsigned long long record_13_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_12_1) * 1536 + (unsigned long long)(row_8 * 24);
                        float _vec_load_9[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)col0) + 0);
                            _vec_load_9[0 + 0] = _v4.x;
                            _vec_load_9[0 + 1] = _v4.y;
                            _vec_load_9[0 + 2] = _v4.z;
                            _vec_load_9[0 + 3] = _v4.w;
                        }
                        float value_11_36 = ((split_11_1 < num_splits) ? _vec_load_9[0] : 0.0f);
                        float _add_128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(sums_1[0]), "f"(value_11_36));
                        sums_1[0] = _add_128;
                        float value_11_37 = ((split_11_1 < num_splits) ? _vec_load_9[1] : 0.0f);
                        float _add_129;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(sums_1[8]), "f"(value_11_37));
                        sums_1[8] = _add_129;
                        float value_11_38 = ((split_11_1 < num_splits) ? _vec_load_9[2] : 0.0f);
                        float _add_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(sums_1[4]), "f"(value_11_38));
                        sums_1[4] = _add_130;
                        float value_11_39 = ((split_11_1 < num_splits) ? _vec_load_9[3] : 0.0f);
                        float _add_131;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(sums_1[12]), "f"(value_11_39));
                        sums_1[12] = _add_131;
                        float _vec_load_10[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)(col0 + 4)) + 0);
                            _vec_load_10[0 + 0] = _v4.x;
                            _vec_load_10[0 + 1] = _v4.y;
                            _vec_load_10[0 + 2] = _v4.z;
                            _vec_load_10[0 + 3] = _v4.w;
                        }
                        float value_11_40 = ((split_11_1 < num_splits) ? _vec_load_10[0] : 0.0f);
                        float _add_133;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(sums_1[2]), "f"(value_11_40));
                        sums_1[2] = _add_133;
                        float value_11_41 = ((split_11_1 < num_splits) ? _vec_load_10[1] : 0.0f);
                        float _add_134;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(sums_1[10]), "f"(value_11_41));
                        sums_1[10] = _add_134;
                        float value_11_42 = ((split_11_1 < num_splits) ? _vec_load_10[2] : 0.0f);
                        float _add_135;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(sums_1[6]), "f"(value_11_42));
                        sums_1[6] = _add_135;
                        float value_11_43 = ((split_11_1 < num_splits) ? _vec_load_10[3] : 0.0f);
                        float _add_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(sums_1[14]), "f"(value_11_43));
                        sums_1[14] = _add_136;
                        float _vec_load_11[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_13_1 + (unsigned long long)(col0 + 8)) + 0);
                            _vec_load_11[0 + 0] = _v4.x;
                            _vec_load_11[0 + 1] = _v4.y;
                            _vec_load_11[0 + 2] = _v4.z;
                            _vec_load_11[0 + 3] = _v4.w;
                        }
                        float value_11_44 = ((split_11_1 < num_splits) ? _vec_load_11[0] : 0.0f);
                        float _add_138;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(sums_1[1]), "f"(value_11_44));
                        sums_1[1] = _add_138;
                        float value_11_45 = ((split_11_1 < num_splits) ? _vec_load_11[1] : 0.0f);
                        float _add_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(sums_1[9]), "f"(value_11_45));
                        sums_1[9] = _add_139;
                        float value_11_46 = ((split_11_1 < num_splits) ? _vec_load_11[2] : 0.0f);
                        float _add_140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(sums_1[5]), "f"(value_11_46));
                        sums_1[5] = _add_140;
                        float value_11_47 = ((split_11_1 < num_splits) ? _vec_load_11[3] : 0.0f);
                        float _add_141;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(sums_1[13]), "f"(value_11_47));
                        sums_1[13] = _add_141;
                        float q_word_26 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_12_1 * 64) + (unsigned long long)row_8];
                        float q_value_27 = ((split_11_1 < num_splits) ? q_word_26 : 0.0f);
                        float _add_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(sums_1[3]), "f"(q_value_27));
                        sums_1[3] = _add_142;
                        float half_l1[8];
                        float keep_1_14 = ((lane_in_warp % 2 == 1) ? sums_1[8] : sums_1[0]);
                        float send_1_14 = ((lane_in_warp % 2 == 1) ? sums_1[0] : sums_1[8]);
                        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
                        float _add_143;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(keep_1_14), "f"(_shfl_xor_sync_144));
                        half_l1[0] = _add_143;
                        float keep_1_15 = ((lane_in_warp % 2 == 1) ? sums_1[9] : sums_1[1]);
                        float send_1_15 = ((lane_in_warp % 2 == 1) ? sums_1[1] : sums_1[9]);
                        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, send_1_15, 1);
                        float _add_145;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(keep_1_15), "f"(_shfl_xor_sync_146));
                        half_l1[1] = _add_145;
                        float keep_1_16 = ((lane_in_warp % 2 == 1) ? sums_1[10] : sums_1[2]);
                        float send_1_16 = ((lane_in_warp % 2 == 1) ? sums_1[2] : sums_1[10]);
                        float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, send_1_16, 1);
                        float _add_147;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(keep_1_16), "f"(_shfl_xor_sync_148));
                        half_l1[2] = _add_147;
                        float keep_1_17 = ((lane_in_warp % 2 == 1) ? sums_1[11] : sums_1[3]);
                        float send_1_17 = ((lane_in_warp % 2 == 1) ? sums_1[3] : sums_1[11]);
                        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, send_1_17, 1);
                        float _add_149;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(keep_1_17), "f"(_shfl_xor_sync_150));
                        half_l1[3] = _add_149;
                        float keep_1_18 = ((lane_in_warp % 2 == 1) ? sums_1[12] : sums_1[4]);
                        float send_1_18 = ((lane_in_warp % 2 == 1) ? sums_1[4] : sums_1[12]);
                        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, send_1_18, 1);
                        float _add_151;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(keep_1_18), "f"(_shfl_xor_sync_152));
                        half_l1[4] = _add_151;
                        float keep_1_19 = ((lane_in_warp % 2 == 1) ? sums_1[13] : sums_1[5]);
                        float send_1_19 = ((lane_in_warp % 2 == 1) ? sums_1[5] : sums_1[13]);
                        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, send_1_19, 1);
                        float _add_153;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(keep_1_19), "f"(_shfl_xor_sync_154));
                        half_l1[5] = _add_153;
                        float keep_1_20 = ((lane_in_warp % 2 == 1) ? sums_1[14] : sums_1[6]);
                        float send_1_20 = ((lane_in_warp % 2 == 1) ? sums_1[6] : sums_1[14]);
                        float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, send_1_20, 1);
                        float _add_155;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(keep_1_20), "f"(_shfl_xor_sync_156));
                        half_l1[6] = _add_155;
                        float keep_1_21 = ((lane_in_warp % 2 == 1) ? sums_1[15] : sums_1[7]);
                        float send_1_21 = ((lane_in_warp % 2 == 1) ? sums_1[7] : sums_1[15]);
                        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, send_1_21, 1);
                        float _add_157;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(keep_1_21), "f"(_shfl_xor_sync_158));
                        half_l1[7] = _add_157;
                        float half_l2[4];
                        float keep_1_22 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[4] : half_l1[0]);
                        float send_1_22 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[0] : half_l1[4]);
                        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, send_1_22, 2);
                        float _add_159;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(keep_1_22), "f"(_shfl_xor_sync_160));
                        half_l2[0] = _add_159;
                        float keep_1_23 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[5] : half_l1[1]);
                        float send_1_23 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[1] : half_l1[5]);
                        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, send_1_23, 2);
                        float _add_161;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(keep_1_23), "f"(_shfl_xor_sync_162));
                        half_l2[1] = _add_161;
                        float keep_1_24 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[6] : half_l1[2]);
                        float send_1_24 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[2] : half_l1[6]);
                        float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, send_1_24, 2);
                        float _add_163;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(keep_1_24), "f"(_shfl_xor_sync_164));
                        half_l2[2] = _add_163;
                        float keep_1_25 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[7] : half_l1[3]);
                        float send_1_25 = ((lane_in_warp / 2 % 2 == 1) ? half_l1[3] : half_l1[7]);
                        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, send_1_25, 2);
                        float _add_165;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(keep_1_25), "f"(_shfl_xor_sync_166));
                        half_l2[3] = _add_165;
                        float half_l3[2];
                        float keep_1_26 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[2] : half_l2[0]);
                        float send_1_26 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[0] : half_l2[2]);
                        float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, send_1_26, 4);
                        float _add_167;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(keep_1_26), "f"(_shfl_xor_sync_168));
                        half_l3[0] = _add_167;
                        float keep_1_27 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[3] : half_l2[1]);
                        float send_1_27 = ((lane_in_warp / 4 % 2 == 1) ? half_l2[1] : half_l2[3]);
                        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, send_1_27, 4);
                        float _add_169;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(keep_1_27), "f"(_shfl_xor_sync_170));
                        half_l3[1] = _add_169;
                        float half_l4[1];
                        float keep_54 = ((lane_in_warp / 8 % 2 == 1) ? half_l3[1] : half_l3[0]);
                        float send_55 = ((lane_in_warp / 8 % 2 == 1) ? half_l3[0] : half_l3[1]);
                        float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, send_55, 8);
                        float _add_171;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(keep_54), "f"(_shfl_xor_sync_172));
                        half_l4[0] = _add_171;
                        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, half_l4[0], 16);
                        float total_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_0) : "f"(half_l4[0]), "f"(_shfl_xor_sync_173));
                        total = total_0;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_0 = ((split < num_splits) ? _vec_load_0[0] : 0.0f);
                    sums[0] = value_23_0;
                    float value_23_1 = ((split < num_splits) ? _vec_load_0[1] : 0.0f);
                    sums[16] = value_23_1;
                    float value_23_2 = ((split < num_splits) ? _vec_load_0[2] : 0.0f);
                    sums[8] = value_23_2;
                    float value_23_3 = ((split < num_splits) ? _vec_load_0[3] : 0.0f);
                    sums[24] = value_23_3;
                    float _vec_load_1[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 4) + 0);
                        _vec_load_1[0 + 0] = _v4.x;
                        _vec_load_1[0 + 1] = _v4.y;
                        _vec_load_1[0 + 2] = _v4.z;
                        _vec_load_1[0 + 3] = _v4.w;
#else
                    float total_0_1 = total;
                    if (flag == 1 && lane_in_warp < 12) {
                        pair_sums[w * 16 + lane_in_warp] = total_0_1;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_4 = ((split < num_splits) ? _vec_load_1[0] : 0.0f);
                    sums[4] = value_23_4;
                    float value_23_5 = ((split < num_splits) ? _vec_load_1[1] : 0.0f);
                    sums[20] = value_23_5;
                    float value_23_6 = ((split < num_splits) ? _vec_load_1[2] : 0.0f);
                    sums[12] = value_23_6;
                    float value_23_7 = ((split < num_splits) ? _vec_load_1[3] : 0.0f);
                    sums[28] = value_23_7;
                    float _vec_load_2[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 8) + 0);
                        _vec_load_2[0 + 0] = _v4.x;
                        _vec_load_2[0 + 1] = _v4.y;
                        _vec_load_2[0 + 2] = _v4.z;
                        _vec_load_2[0 + 3] = _v4.w;
                    }
                    float value_23_8 = ((split < num_splits) ? _vec_load_2[0] : 0.0f);
                    sums[2] = value_23_8;
                    float value_23_9 = ((split < num_splits) ? _vec_load_2[1] : 0.0f);
                    sums[18] = value_23_9;
                    float value_23_10 = ((split < num_splits) ? _vec_load_2[2] : 0.0f);
                    sums[10] = value_23_10;
                    float value_23_11 = ((split < num_splits) ? _vec_load_2[3] : 0.0f);
                    sums[26] = value_23_11;
                    float _vec_load_3[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 12) + 0);
                        _vec_load_3[0 + 0] = _v4.x;
                        _vec_load_3[0 + 1] = _v4.y;
                        _vec_load_3[0 + 2] = _v4.z;
                        _vec_load_3[0 + 3] = _v4.w;
                    }
                    float value_23_12 = ((split < num_splits) ? _vec_load_3[0] : 0.0f);
                    sums[6] = value_23_12;
                    float value_23_13 = ((split < num_splits) ? _vec_load_3[1] : 0.0f);
                    sums[22] = value_23_13;
                    float value_23_14 = ((split < num_splits) ? _vec_load_3[2] : 0.0f);
                    sums[14] = value_23_14;
                    float value_23_15 = ((split < num_splits) ? _vec_load_3[3] : 0.0f);
                    sums[30] = value_23_15;
                    float _vec_load_4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 16) + 0);
                        _vec_load_4[0 + 0] = _v4.x;
                        _vec_load_4[0 + 1] = _v4.y;
                        _vec_load_4[0 + 2] = _v4.z;
                        _vec_load_4[0 + 3] = _v4.w;
                    }
                    float value_23_16 = ((split < num_splits) ? _vec_load_4[0] : 0.0f);
                    sums[1] = value_23_16;
                    float value_23_17 = ((split < num_splits) ? _vec_load_4[1] : 0.0f);
                    sums[17] = value_23_17;
                    float value_23_18 = ((split < num_splits) ? _vec_load_4[2] : 0.0f);
                    sums[9] = value_23_18;
                    float value_23_19 = ((split < num_splits) ? _vec_load_4[3] : 0.0f);
                    sums[25] = value_23_19;
                    float _vec_load_5[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 20) + 0);
                        _vec_load_5[0 + 0] = _v4.x;
                        _vec_load_5[0 + 1] = _v4.y;
                        _vec_load_5[0 + 2] = _v4.z;
                        _vec_load_5[0 + 3] = _v4.w;
                    }
                    float value_23_20 = ((split < num_splits) ? _vec_load_5[0] : 0.0f);
                    sums[5] = value_23_20;
                    float value_23_21 = ((split < num_splits) ? _vec_load_5[1] : 0.0f);
                    sums[21] = value_23_21;
                    float value_23_22 = ((split < num_splits) ? _vec_load_5[2] : 0.0f);
                    sums[13] = value_23_22;
                    float value_23_23 = ((split < num_splits) ? _vec_load_5[3] : 0.0f);
                    sums[29] = value_23_23;
                    float q_word = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split * 64) + (unsigned long long)row_5];
                    float q_value = ((split < num_splits) ? q_word : 0.0f);
                    sums[3] = q_value;
                    unsigned int split_23 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24 = ((split_23 < num_splits) ? split_23 : 0);
                    unsigned long long record_25 = ((unsigned long long)first_task + (unsigned long long)safe_split_24) * 1536 + (unsigned long long)(row_5 * 24);
                    float _vec_load_6[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25 + 0);
                        _vec_load_6[0 + 0] = _v4.x;
                        _vec_load_6[0 + 1] = _v4.y;
                        _vec_load_6[0 + 2] = _v4.z;
                        _vec_load_6[0 + 3] = _v4.w;
                    }
                    float value_23_24 = ((split_23 < num_splits) ? _vec_load_6[0] : 0.0f);
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(sums[0]), "f"(value_23_24));
                    sums[0] = _add_7;
                    float value_23_25 = ((split_23 < num_splits) ? _vec_load_6[1] : 0.0f);
                    float _add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(sums[16]), "f"(value_23_25));
                    sums[16] = _add_8;
                    float value_23_26 = ((split_23 < num_splits) ? _vec_load_6[2] : 0.0f);
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(sums[8]), "f"(value_23_26));
                    sums[8] = _add_9;
                    float value_23_27 = ((split_23 < num_splits) ? _vec_load_6[3] : 0.0f);
                    float _add_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(sums[24]), "f"(value_23_27));
                    sums[24] = _add_10;
                    float _vec_load_7[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 4) + 0);
                        _vec_load_7[0 + 0] = _v4.x;
                        _vec_load_7[0 + 1] = _v4.y;
                        _vec_load_7[0 + 2] = _v4.z;
                        _vec_load_7[0 + 3] = _v4.w;
                    }
                    float value_23_28 = ((split_23 < num_splits) ? _vec_load_7[0] : 0.0f);
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(sums[4]), "f"(value_23_28));
                    sums[4] = _add_12;
                    float value_23_29 = ((split_23 < num_splits) ? _vec_load_7[1] : 0.0f);
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(sums[20]), "f"(value_23_29));
                    sums[20] = _add_13;
                    float value_23_30 = ((split_23 < num_splits) ? _vec_load_7[2] : 0.0f);
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(sums[12]), "f"(value_23_30));
                    sums[12] = _add_14;
                    float value_23_31 = ((split_23 < num_splits) ? _vec_load_7[3] : 0.0f);
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(sums[28]), "f"(value_23_31));
                    sums[28] = _add_15;
                    float _vec_load_8[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 8) + 0);
                        _vec_load_8[0 + 0] = _v4.x;
                        _vec_load_8[0 + 1] = _v4.y;
                        _vec_load_8[0 + 2] = _v4.z;
                        _vec_load_8[0 + 3] = _v4.w;
                    }
                    float value_23_32 = ((split_23 < num_splits) ? _vec_load_8[0] : 0.0f);
                    float _add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(sums[2]), "f"(value_23_32));
                    sums[2] = _add_17;
                    float value_23_33 = ((split_23 < num_splits) ? _vec_load_8[1] : 0.0f);
                    float _add_18_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_1) : "f"(sums[18]), "f"(value_23_33));
                    sums[18] = _add_18_1;
                    float value_23_34 = ((split_23 < num_splits) ? _vec_load_8[2] : 0.0f);
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(sums[10]), "f"(value_23_34));
                    sums[10] = _add_19;
                    float value_23_35 = ((split_23 < num_splits) ? _vec_load_8[3] : 0.0f);
                    float _add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(sums[26]), "f"(value_23_35));
                    sums[26] = _add_20;
                    float _vec_load_9[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 12) + 0);
                        _vec_load_9[0 + 0] = _v4.x;
                        _vec_load_9[0 + 1] = _v4.y;
                        _vec_load_9[0 + 2] = _v4.z;
                        _vec_load_9[0 + 3] = _v4.w;
                    }
                    float value_23_36 = ((split_23 < num_splits) ? _vec_load_9[0] : 0.0f);
                    float _add_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(sums[6]), "f"(value_23_36));
                    sums[6] = _add_22;
                    float value_23_37 = ((split_23 < num_splits) ? _vec_load_9[1] : 0.0f);
                    float _add_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(sums[22]), "f"(value_23_37));
                    sums[22] = _add_23;
                    float value_23_38 = ((split_23 < num_splits) ? _vec_load_9[2] : 0.0f);
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(sums[14]), "f"(value_23_38));
                    sums[14] = _add_24;
                    float value_23_39 = ((split_23 < num_splits) ? _vec_load_9[3] : 0.0f);
                    float _add_25_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_1) : "f"(sums[30]), "f"(value_23_39));
                    sums[30] = _add_25_1;
                    float _vec_load_10[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 16) + 0);
                        _vec_load_10[0 + 0] = _v4.x;
                        _vec_load_10[0 + 1] = _v4.y;
                        _vec_load_10[0 + 2] = _v4.z;
                        _vec_load_10[0 + 3] = _v4.w;
                    }
                    float value_23_40 = ((split_23 < num_splits) ? _vec_load_10[0] : 0.0f);
                    float _add_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(sums[1]), "f"(value_23_40));
                    sums[1] = _add_27;
                    float value_23_41 = ((split_23 < num_splits) ? _vec_load_10[1] : 0.0f);
                    float _add_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(sums[17]), "f"(value_23_41));
                    sums[17] = _add_28;
                    float value_23_42 = ((split_23 < num_splits) ? _vec_load_10[2] : 0.0f);
                    float _add_29;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(sums[9]), "f"(value_23_42));
                    sums[9] = _add_29;
                    float value_23_43 = ((split_23 < num_splits) ? _vec_load_10[3] : 0.0f);
                    float _add_30;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(sums[25]), "f"(value_23_43));
                    sums[25] = _add_30;
                    float _vec_load_11[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 20) + 0);
                        _vec_load_11[0 + 0] = _v4.x;
                        _vec_load_11[0 + 1] = _v4.y;
                        _vec_load_11[0 + 2] = _v4.z;
                        _vec_load_11[0 + 3] = _v4.w;
                    }
                    float value_23_44 = ((split_23 < num_splits) ? _vec_load_11[0] : 0.0f);
                    float _add_32_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(sums[5]), "f"(value_23_44));
                    sums[5] = _add_32_1;
                    float value_23_45 = ((split_23 < num_splits) ? _vec_load_11[1] : 0.0f);
                    float _add_33;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(sums[21]), "f"(value_23_45));
                    sums[21] = _add_33;
                    float value_23_46 = ((split_23 < num_splits) ? _vec_load_11[2] : 0.0f);
                    float _add_34;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(sums[13]), "f"(value_23_46));
                    sums[13] = _add_34;
                    float value_23_47 = ((split_23 < num_splits) ? _vec_load_11[3] : 0.0f);
                    float _add_35;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(sums[29]), "f"(value_23_47));
                    sums[29] = _add_35;
                    float q_word_50 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split_24 * 64) + (unsigned long long)row_5];
                    float q_value_51 = ((split_23 < num_splits) ? q_word_50 : 0.0f);
                    float _add_36;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(sums[3]), "f"(q_value_51));
                    sums[3] = _add_36;
                    float fold_level1[16];
                    float keep_1_0 = ((lane_in_warp % 2 == 1) ? sums[16] : sums[0]);
                    float send_1_0 = ((lane_in_warp % 2 == 1) ? sums[0] : sums[16]);
                    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
                    float _add_37;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(keep_1_0), "f"(_shfl_xor_sync_38));
                    fold_level1[0] = _add_37;
                    float keep_1_1 = ((lane_in_warp % 2 == 1) ? sums[17] : sums[1]);
                    float send_1_1 = ((lane_in_warp % 2 == 1) ? sums[1] : sums[17]);
                    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
                    float _add_39_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_1) : "f"(keep_1_1), "f"(_shfl_xor_sync_40));
                    fold_level1[1] = _add_39_1;
                    float keep_1_2 = ((lane_in_warp % 2 == 1) ? sums[18] : sums[2]);
                    float send_1_2 = ((lane_in_warp % 2 == 1) ? sums[2] : sums[18]);
                    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
                    float _add_41;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(keep_1_2), "f"(_shfl_xor_sync_42));
                    fold_level1[2] = _add_41;
                    float keep_1_3 = ((lane_in_warp % 2 == 1) ? sums[19] : sums[3]);
                    float send_1_3 = ((lane_in_warp % 2 == 1) ? sums[3] : sums[19]);
                    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
                    float _add_43;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(keep_1_3), "f"(_shfl_xor_sync_44));
                    fold_level1[3] = _add_43;
                    float keep_1_4 = ((lane_in_warp % 2 == 1) ? sums[20] : sums[4]);
                    float send_1_4 = ((lane_in_warp % 2 == 1) ? sums[4] : sums[20]);
                    float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
                    float _add_45;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(keep_1_4), "f"(_shfl_xor_sync_46));
                    fold_level1[4] = _add_45;
                    float keep_1_5 = ((lane_in_warp % 2 == 1) ? sums[21] : sums[5]);
                    float send_1_5 = ((lane_in_warp % 2 == 1) ? sums[5] : sums[21]);
                    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
                    float _add_47;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(keep_1_5), "f"(_shfl_xor_sync_48));
                    fold_level1[5] = _add_47;
                    float keep_1_6 = ((lane_in_warp % 2 == 1) ? sums[22] : sums[6]);
                    float send_1_6 = ((lane_in_warp % 2 == 1) ? sums[6] : sums[22]);
                    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
                    float _add_49;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(keep_1_6), "f"(_shfl_xor_sync_50));
                    fold_level1[6] = _add_49;
                    float keep_1_7 = ((lane_in_warp % 2 == 1) ? sums[23] : sums[7]);
                    float send_1_7 = ((lane_in_warp % 2 == 1) ? sums[7] : sums[23]);
                    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, send_1_7, 1);
                    float _add_51;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(keep_1_7), "f"(_shfl_xor_sync_52));
                    fold_level1[7] = _add_51;
                    float keep_1_8 = ((lane_in_warp % 2 == 1) ? sums[24] : sums[8]);
                    float send_1_8 = ((lane_in_warp % 2 == 1) ? sums[8] : sums[24]);
                    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 1);
                    float _add_53_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(keep_1_8), "f"(_shfl_xor_sync_54));
                    fold_level1[8] = _add_53_1;
                    float keep_1_9 = ((lane_in_warp % 2 == 1) ? sums[25] : sums[9]);
                    float send_1_9 = ((lane_in_warp % 2 == 1) ? sums[9] : sums[25]);
                    float _shfl_xor_sync_56_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 1);
                    float _add_55;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(keep_1_9), "f"(_shfl_xor_sync_56_1));
                    fold_level1[9] = _add_55;
                    float keep_1_10 = ((lane_in_warp % 2 == 1) ? sums[26] : sums[10]);
                    float send_1_10 = ((lane_in_warp % 2 == 1) ? sums[10] : sums[26]);
                    float _shfl_xor_sync_58_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 1);
                    float _add_57;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(keep_1_10), "f"(_shfl_xor_sync_58_1));
                    fold_level1[10] = _add_57;
                    float keep_1_11 = ((lane_in_warp % 2 == 1) ? sums[27] : sums[11]);
                    float send_1_11 = ((lane_in_warp % 2 == 1) ? sums[11] : sums[27]);
                    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 1);
                    float _add_59;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(keep_1_11), "f"(_shfl_xor_sync_60));
                    fold_level1[11] = _add_59;
                    float keep_1_12 = ((lane_in_warp % 2 == 1) ? sums[28] : sums[12]);
                    float send_1_12 = ((lane_in_warp % 2 == 1) ? sums[12] : sums[28]);
                    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 1);
                    float _add_61;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(keep_1_12), "f"(_shfl_xor_sync_62));
                    fold_level1[12] = _add_61;
                    float keep_1_13 = ((lane_in_warp % 2 == 1) ? sums[29] : sums[13]);
                    float send_1_13 = ((lane_in_warp % 2 == 1) ? sums[13] : sums[29]);
                    float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 1);
                    float _add_63;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(keep_1_13), "f"(_shfl_xor_sync_64));
                    fold_level1[13] = _add_63;
                    float keep_1_14 = ((lane_in_warp % 2 == 1) ? sums[30] : sums[14]);
                    float send_1_14 = ((lane_in_warp % 2 == 1) ? sums[14] : sums[30]);
                    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(keep_1_14), "f"(_shfl_xor_sync_66));
                    fold_level1[14] = _add_65;
                    float keep_1_15 = ((lane_in_warp % 2 == 1) ? sums[31] : sums[15]);
                    float send_1_15 = ((lane_in_warp % 2 == 1) ? sums[15] : sums[31]);
                    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, send_1_15, 1);
                    float _add_67;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(keep_1_15), "f"(_shfl_xor_sync_68));
                    fold_level1[15] = _add_67;
                    float fold_level2[8];
                    float keep_1_16 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[8] : fold_level1[0]);
                    float send_1_16 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[0] : fold_level1[8]);
                    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, send_1_16, 2);
                    float _add_69;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(keep_1_16), "f"(_shfl_xor_sync_70));
                    fold_level2[0] = _add_69;
                    float keep_1_17 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[9] : fold_level1[1]);
                    float send_1_17 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[1] : fold_level1[9]);
                    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, send_1_17, 2);
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(keep_1_17), "f"(_shfl_xor_sync_72));
                    fold_level2[1] = _add_71;
                    float keep_1_18 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[10] : fold_level1[2]);
                    float send_1_18 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[2] : fold_level1[10]);
                    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, send_1_18, 2);
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(keep_1_18), "f"(_shfl_xor_sync_74));
                    fold_level2[2] = _add_73;
                    float keep_1_19 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[11] : fold_level1[3]);
                    float send_1_19 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[3] : fold_level1[11]);
                    float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, send_1_19, 2);
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(keep_1_19), "f"(_shfl_xor_sync_76));
                    fold_level2[3] = _add_75;
                    float keep_1_20 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[12] : fold_level1[4]);
                    float send_1_20 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[4] : fold_level1[12]);
                    float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, send_1_20, 2);
                    float _add_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(keep_1_20), "f"(_shfl_xor_sync_78));
                    fold_level2[4] = _add_77;
                    float keep_1_21 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[13] : fold_level1[5]);
                    float send_1_21 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[5] : fold_level1[13]);
                    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, send_1_21, 2);
                    float _add_79;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(keep_1_21), "f"(_shfl_xor_sync_80));
                    fold_level2[5] = _add_79;
                    float keep_1_22 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[14] : fold_level1[6]);
                    float send_1_22 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[6] : fold_level1[14]);
                    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, send_1_22, 2);
                    float _add_81;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(keep_1_22), "f"(_shfl_xor_sync_82));
                    fold_level2[6] = _add_81;
                    float keep_1_23 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[15] : fold_level1[7]);
                    float send_1_23 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[7] : fold_level1[15]);
                    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, send_1_23, 2);
                    float _add_83;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(keep_1_23), "f"(_shfl_xor_sync_84));
                    fold_level2[7] = _add_83;
                    float fold_level3[4];
                    float keep_1_24 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[4] : fold_level2[0]);
                    float send_1_24 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[0] : fold_level2[4]);
                    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, send_1_24, 4);
                    float _add_85;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(keep_1_24), "f"(_shfl_xor_sync_86));
                    fold_level3[0] = _add_85;
                    float keep_1_25 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[5] : fold_level2[1]);
                    float send_1_25 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[1] : fold_level2[5]);
                    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, send_1_25, 4);
                    float _add_87;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(keep_1_25), "f"(_shfl_xor_sync_88));
                    fold_level3[1] = _add_87;
                    float keep_1_26 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[6] : fold_level2[2]);
                    float send_1_26 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[2] : fold_level2[6]);
                    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, send_1_26, 4);
                    float _add_89;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(keep_1_26), "f"(_shfl_xor_sync_90));
                    fold_level3[2] = _add_89;
                    float keep_1_27 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[7] : fold_level2[3]);
                    float send_1_27 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[3] : fold_level2[7]);
                    float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, send_1_27, 4);
                    float _add_91;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(keep_1_27), "f"(_shfl_xor_sync_92));
                    fold_level3[3] = _add_91;
                    float fold_level4[2];
                    float keep_1_28 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[2] : fold_level3[0]);
                    float send_1_28 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[0] : fold_level3[2]);
                    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, send_1_28, 8);
                    float _add_93;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(keep_1_28), "f"(_shfl_xor_sync_94));
                    fold_level4[0] = _add_93;
                    float keep_1_29 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[3] : fold_level3[1]);
                    float send_1_29 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[1] : fold_level3[3]);
                    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, send_1_29, 8);
                    float _add_95;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(keep_1_29), "f"(_shfl_xor_sync_96));
                    fold_level4[1] = _add_95;
                    float fold_level5[1];
                    float keep_110 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[1] : fold_level4[0]);
                    float send_111 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[0] : fold_level4[1]);
                    float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, send_111, 16);
                    float _add_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(keep_110), "f"(_shfl_xor_sync_98));
                    fold_level5[0] = _add_97;
                    float total = fold_level5[0];
                    float s_112 = ((lane_in_warp < 24) ? total : 0.0f);
                    float q_113 = __shfl_sync(0xFFFFFFFF, total, 24);
                    s = s_112;
                    q = q_113;
                } else {
                    unsigned int row_8 = token % 64;
                    unsigned int first_task_1 = mb_1 * num_splits;
                    float sums_1[25];
                    unsigned int split_1 = lane_in_warp * 2;
                    unsigned int safe_split_1 = ((split_1 < num_splits) ? split_1 : 0);
                    unsigned long long record_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_1) * 1536 + (unsigned long long)(row_8 * 24);
#else
                    asm volatile("barrier.sync %0, 64;" :: "r"(4 + w) : "memory");
                    unsigned int pidx = ((lane_in_warp >= 12 && lane_in_warp < 24) ? lane_in_warp - 12 : 0);
                    float partner = pair_sums[w * 16 + pidx];
                    float s_1_1 = ((lane_in_warp < 12) ? total_0_1 : ((lane_in_warp < 24) ? partner : 0.0f));
                    float q_2 = __shfl_sync(0xFFFFFFFF, total_0_1, 12);
                    s = s_1_1;
                    q = q_2;
                } else if ((num_tokens + 63) / 64 * num_splits <= num_ctas && num_splits <= 22) {
                    unsigned int row_11 = token % 64;
                    unsigned int first_task_2 = mb_1 * num_splits;
                    float sums_2[32];
                    sums_2[0] = 0.0f;
                    sums_2[1] = 0.0f;
                    sums_2[2] = 0.0f;
                    sums_2[3] = 0.0f;
                    sums_2[4] = 0.0f;
                    sums_2[5] = 0.0f;
                    sums_2[6] = 0.0f;
                    sums_2[7] = 0.0f;
                    sums_2[8] = 0.0f;
                    sums_2[9] = 0.0f;
                    sums_2[10] = 0.0f;
                    sums_2[11] = 0.0f;
                    sums_2[12] = 0.0f;
                    sums_2[13] = 0.0f;
                    sums_2[14] = 0.0f;
                    sums_2[15] = 0.0f;
                    sums_2[16] = 0.0f;
                    sums_2[17] = 0.0f;
                    sums_2[18] = 0.0f;
                    sums_2[19] = 0.0f;
                    sums_2[20] = 0.0f;
                    sums_2[21] = 0.0f;
                    sums_2[22] = 0.0f;
                    sums_2[23] = 0.0f;
                    sums_2[24] = 0.0f;
                    sums_2[25] = 0.0f;
                    sums_2[26] = 0.0f;
                    sums_2[27] = 0.0f;
                    sums_2[28] = 0.0f;
                    sums_2[29] = 0.0f;
                    sums_2[30] = 0.0f;
                    sums_2[31] = 0.0f;
                    unsigned int split_2 = lane_in_warp * 2;
                    unsigned int safe_split_2 = ((split_2 < num_splits) ? split_2 : 0);
                    unsigned long long record_2 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_2) * 1536 + (unsigned long long)(row_11 * 24);
#endif
                    float _vec_load_12[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_1 + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_2 + 0);
#endif
                        _vec_load_12[0 + 0] = _v4.x;
                        _vec_load_12[0 + 1] = _v4.y;
                        _vec_load_12[0 + 2] = _v4.z;
                        _vec_load_12[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_48 = ((split_1 < num_splits) ? _vec_load_12[0] : 0.0f);
                    sums_1[0] = value_23_48;
                    float value_23_49 = ((split_1 < num_splits) ? _vec_load_12[1] : 0.0f);
                    sums_1[1] = value_23_49;
                    float value_23_50 = ((split_1 < num_splits) ? _vec_load_12[2] : 0.0f);
                    sums_1[2] = value_23_50;
                    float value_23_51 = ((split_1 < num_splits) ? _vec_load_12[3] : 0.0f);
                    sums_1[3] = value_23_51;
#else
                    float value_23_0 = ((split_2 < num_splits) ? _vec_load_12[0] : 0.0f);
                    sums_2[0] = value_23_0;
                    float value_23_1 = ((split_2 < num_splits) ? _vec_load_12[1] : 0.0f);
                    sums_2[16] = value_23_1;
                    float value_23_2 = ((split_2 < num_splits) ? _vec_load_12[2] : 0.0f);
                    sums_2[8] = value_23_2;
                    float value_23_3 = ((split_2 < num_splits) ? _vec_load_12[3] : 0.0f);
                    sums_2[24] = value_23_3;
#endif
                    float _vec_load_13[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 4) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_2 + 4) + 0);
#endif
                        _vec_load_13[0 + 0] = _v4.x;
                        _vec_load_13[0 + 1] = _v4.y;
                        _vec_load_13[0 + 2] = _v4.z;
                        _vec_load_13[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_52 = ((split_1 < num_splits) ? _vec_load_13[0] : 0.0f);
                    sums_1[4] = value_23_52;
                    float value_23_53 = ((split_1 < num_splits) ? _vec_load_13[1] : 0.0f);
                    sums_1[5] = value_23_53;
                    float value_23_54 = ((split_1 < num_splits) ? _vec_load_13[2] : 0.0f);
                    sums_1[6] = value_23_54;
                    float value_23_55 = ((split_1 < num_splits) ? _vec_load_13[3] : 0.0f);
                    sums_1[7] = value_23_55;
#else
                    float value_23_4 = ((split_2 < num_splits) ? _vec_load_13[0] : 0.0f);
                    sums_2[4] = value_23_4;
                    float value_23_5 = ((split_2 < num_splits) ? _vec_load_13[1] : 0.0f);
                    sums_2[20] = value_23_5;
                    float value_23_6 = ((split_2 < num_splits) ? _vec_load_13[2] : 0.0f);
                    sums_2[12] = value_23_6;
                    float value_23_7 = ((split_2 < num_splits) ? _vec_load_13[3] : 0.0f);
                    sums_2[28] = value_23_7;
#endif
                    float _vec_load_14[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 8) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_2 + 8) + 0);
#endif
                        _vec_load_14[0 + 0] = _v4.x;
                        _vec_load_14[0 + 1] = _v4.y;
                        _vec_load_14[0 + 2] = _v4.z;
                        _vec_load_14[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_56 = ((split_1 < num_splits) ? _vec_load_14[0] : 0.0f);
                    sums_1[8] = value_23_56;
                    float value_23_57 = ((split_1 < num_splits) ? _vec_load_14[1] : 0.0f);
                    sums_1[9] = value_23_57;
                    float value_23_58 = ((split_1 < num_splits) ? _vec_load_14[2] : 0.0f);
                    sums_1[10] = value_23_58;
                    float value_23_59 = ((split_1 < num_splits) ? _vec_load_14[3] : 0.0f);
                    sums_1[11] = value_23_59;
#else
                    float value_23_8 = ((split_2 < num_splits) ? _vec_load_14[0] : 0.0f);
                    sums_2[2] = value_23_8;
                    float value_23_9 = ((split_2 < num_splits) ? _vec_load_14[1] : 0.0f);
                    sums_2[18] = value_23_9;
                    float value_23_10 = ((split_2 < num_splits) ? _vec_load_14[2] : 0.0f);
                    sums_2[10] = value_23_10;
                    float value_23_11 = ((split_2 < num_splits) ? _vec_load_14[3] : 0.0f);
                    sums_2[26] = value_23_11;
#endif
                    float _vec_load_15[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 12) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_2 + 12) + 0);
#endif
                        _vec_load_15[0 + 0] = _v4.x;
                        _vec_load_15[0 + 1] = _v4.y;
                        _vec_load_15[0 + 2] = _v4.z;
                        _vec_load_15[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_60 = ((split_1 < num_splits) ? _vec_load_15[0] : 0.0f);
                    sums_1[12] = value_23_60;
                    float value_23_61 = ((split_1 < num_splits) ? _vec_load_15[1] : 0.0f);
                    sums_1[13] = value_23_61;
                    float value_23_62 = ((split_1 < num_splits) ? _vec_load_15[2] : 0.0f);
                    sums_1[14] = value_23_62;
                    float value_23_63 = ((split_1 < num_splits) ? _vec_load_15[3] : 0.0f);
                    sums_1[15] = value_23_63;
#else
                    float value_23_12 = ((split_2 < num_splits) ? _vec_load_15[0] : 0.0f);
                    sums_2[6] = value_23_12;
                    float value_23_13 = ((split_2 < num_splits) ? _vec_load_15[1] : 0.0f);
                    sums_2[22] = value_23_13;
                    float value_23_14 = ((split_2 < num_splits) ? _vec_load_15[2] : 0.0f);
                    sums_2[14] = value_23_14;
                    float value_23_15 = ((split_2 < num_splits) ? _vec_load_15[3] : 0.0f);
                    sums_2[30] = value_23_15;
#endif
                    float _vec_load_16[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 16) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_2 + 16) + 0);
#endif
                        _vec_load_16[0 + 0] = _v4.x;
                        _vec_load_16[0 + 1] = _v4.y;
                        _vec_load_16[0 + 2] = _v4.z;
                        _vec_load_16[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_64 = ((split_1 < num_splits) ? _vec_load_16[0] : 0.0f);
                    sums_1[16] = value_23_64;
                    float value_23_65 = ((split_1 < num_splits) ? _vec_load_16[1] : 0.0f);
                    sums_1[17] = value_23_65;
                    float value_23_66 = ((split_1 < num_splits) ? _vec_load_16[2] : 0.0f);
                    sums_1[18] = value_23_66;
                    float value_23_67 = ((split_1 < num_splits) ? _vec_load_16[3] : 0.0f);
                    sums_1[19] = value_23_67;
#else
                    float value_23_16 = ((split_2 < num_splits) ? _vec_load_16[0] : 0.0f);
                    sums_2[1] = value_23_16;
                    float value_23_17 = ((split_2 < num_splits) ? _vec_load_16[1] : 0.0f);
                    sums_2[17] = value_23_17;
                    float value_23_18 = ((split_2 < num_splits) ? _vec_load_16[2] : 0.0f);
                    sums_2[9] = value_23_18;
                    float value_23_19 = ((split_2 < num_splits) ? _vec_load_16[3] : 0.0f);
                    sums_2[25] = value_23_19;
#endif
                    float _vec_load_17[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_1 + 20) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_2 + 20) + 0);
#endif
                        _vec_load_17[0 + 0] = _v4.x;
                        _vec_load_17[0 + 1] = _v4.y;
                        _vec_load_17[0 + 2] = _v4.z;
                        _vec_load_17[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_68 = ((split_1 < num_splits) ? _vec_load_17[0] : 0.0f);
                    sums_1[20] = value_23_68;
                    float value_23_69 = ((split_1 < num_splits) ? _vec_load_17[1] : 0.0f);
                    sums_1[21] = value_23_69;
                    float value_23_70 = ((split_1 < num_splits) ? _vec_load_17[2] : 0.0f);
                    sums_1[22] = value_23_70;
                    float value_23_71 = ((split_1 < num_splits) ? _vec_load_17[3] : 0.0f);
                    sums_1[23] = value_23_71;
                    float q_word_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_1 * 64) + (unsigned long long)row_8];
                    float q_value_1 = ((split_1 < num_splits) ? q_word_1 : 0.0f);
                    sums_1[24] = q_value_1;
                    unsigned int split_23_1 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24_1 = ((split_23_1 < num_splits) ? split_23_1 : 0);
                    unsigned long long record_25_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_24_1) * 1536 + (unsigned long long)(row_8 * 24);
#else
                    float value_23_20 = ((split_2 < num_splits) ? _vec_load_17[0] : 0.0f);
                    sums_2[5] = value_23_20;
                    float value_23_21 = ((split_2 < num_splits) ? _vec_load_17[1] : 0.0f);
                    sums_2[21] = value_23_21;
                    float value_23_22 = ((split_2 < num_splits) ? _vec_load_17[2] : 0.0f);
                    sums_2[13] = value_23_22;
                    float value_23_23 = ((split_2 < num_splits) ? _vec_load_17[3] : 0.0f);
                    sums_2[29] = value_23_23;
                    float q_word_2 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_2 * 64) + (unsigned long long)row_11];
                    float q_value_2 = ((split_2 < num_splits) ? q_word_2 : 0.0f);
                    sums_2[3] = q_value_2;
                    unsigned int split_23 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_24 = ((split_23 < num_splits) ? split_23 : 0);
                    unsigned long long record_25 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_24) * 1536 + (unsigned long long)(row_11 * 24);
#endif
                    float _vec_load_18[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_1 + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25 + 0);
#endif
                        _vec_load_18[0 + 0] = _v4.x;
                        _vec_load_18[0 + 1] = _v4.y;
                        _vec_load_18[0 + 2] = _v4.z;
                        _vec_load_18[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_72 = ((split_23_1 < num_splits) ? _vec_load_18[0] : 0.0f);
                    float _add_106;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(sums_1[0]), "f"(value_23_72));
                    sums_1[0] = _add_106;
                    float value_23_73 = ((split_23_1 < num_splits) ? _vec_load_18[1] : 0.0f);
                    float _add_107;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(sums_1[1]), "f"(value_23_73));
                    sums_1[1] = _add_107;
                    float value_23_74 = ((split_23_1 < num_splits) ? _vec_load_18[2] : 0.0f);
                    float _add_108;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(sums_1[2]), "f"(value_23_74));
                    sums_1[2] = _add_108;
                    float value_23_75 = ((split_23_1 < num_splits) ? _vec_load_18[3] : 0.0f);
                    float _add_109;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(sums_1[3]), "f"(value_23_75));
                    sums_1[3] = _add_109;
#else
                    float value_23_24 = ((split_23 < num_splits) ? _vec_load_18[0] : 0.0f);
                    float _add_181;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(sums_2[0]), "f"(value_23_24));
                    sums_2[0] = _add_181;
                    float value_23_25 = ((split_23 < num_splits) ? _vec_load_18[1] : 0.0f);
                    float _add_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(sums_2[16]), "f"(value_23_25));
                    sums_2[16] = _add_182;
                    float value_23_26 = ((split_23 < num_splits) ? _vec_load_18[2] : 0.0f);
                    float _add_183;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(sums_2[8]), "f"(value_23_26));
                    sums_2[8] = _add_183;
                    float value_23_27 = ((split_23 < num_splits) ? _vec_load_18[3] : 0.0f);
                    float _add_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(sums_2[24]), "f"(value_23_27));
                    sums_2[24] = _add_184;
#endif
                    float _vec_load_19[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 4) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 4) + 0);
#endif
                        _vec_load_19[0 + 0] = _v4.x;
                        _vec_load_19[0 + 1] = _v4.y;
                        _vec_load_19[0 + 2] = _v4.z;
                        _vec_load_19[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_76 = ((split_23_1 < num_splits) ? _vec_load_19[0] : 0.0f);
                    float _add_111;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(sums_1[4]), "f"(value_23_76));
                    sums_1[4] = _add_111;
                    float value_23_77 = ((split_23_1 < num_splits) ? _vec_load_19[1] : 0.0f);
                    float _add_112;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(sums_1[5]), "f"(value_23_77));
                    sums_1[5] = _add_112;
                    float value_23_78 = ((split_23_1 < num_splits) ? _vec_load_19[2] : 0.0f);
                    float _add_113;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(sums_1[6]), "f"(value_23_78));
                    sums_1[6] = _add_113;
                    float value_23_79 = ((split_23_1 < num_splits) ? _vec_load_19[3] : 0.0f);
                    float _add_114;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(sums_1[7]), "f"(value_23_79));
                    sums_1[7] = _add_114;
#else
                    float value_23_28 = ((split_23 < num_splits) ? _vec_load_19[0] : 0.0f);
                    float _add_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(sums_2[4]), "f"(value_23_28));
                    sums_2[4] = _add_186;
                    float value_23_29 = ((split_23 < num_splits) ? _vec_load_19[1] : 0.0f);
                    float _add_187;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(sums_2[20]), "f"(value_23_29));
                    sums_2[20] = _add_187;
                    float value_23_30 = ((split_23 < num_splits) ? _vec_load_19[2] : 0.0f);
                    float _add_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(sums_2[12]), "f"(value_23_30));
                    sums_2[12] = _add_188;
                    float value_23_31 = ((split_23 < num_splits) ? _vec_load_19[3] : 0.0f);
                    float _add_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(sums_2[28]), "f"(value_23_31));
                    sums_2[28] = _add_189;
#endif
                    float _vec_load_20[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 8) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 8) + 0);
#endif
                        _vec_load_20[0 + 0] = _v4.x;
                        _vec_load_20[0 + 1] = _v4.y;
                        _vec_load_20[0 + 2] = _v4.z;
                        _vec_load_20[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_80 = ((split_23_1 < num_splits) ? _vec_load_20[0] : 0.0f);
                    float _add_116;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(sums_1[8]), "f"(value_23_80));
                    sums_1[8] = _add_116;
                    float value_23_81 = ((split_23_1 < num_splits) ? _vec_load_20[1] : 0.0f);
                    float _add_117;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(sums_1[9]), "f"(value_23_81));
                    sums_1[9] = _add_117;
                    float value_23_82 = ((split_23_1 < num_splits) ? _vec_load_20[2] : 0.0f);
                    float _add_118;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(sums_1[10]), "f"(value_23_82));
                    sums_1[10] = _add_118;
                    float value_23_83 = ((split_23_1 < num_splits) ? _vec_load_20[3] : 0.0f);
                    float _add_119;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(sums_1[11]), "f"(value_23_83));
                    sums_1[11] = _add_119;
#else
                    float value_23_32 = ((split_23 < num_splits) ? _vec_load_20[0] : 0.0f);
                    float _add_191;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(sums_2[2]), "f"(value_23_32));
                    sums_2[2] = _add_191;
                    float value_23_33 = ((split_23 < num_splits) ? _vec_load_20[1] : 0.0f);
                    float _add_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(sums_2[18]), "f"(value_23_33));
                    sums_2[18] = _add_192;
                    float value_23_34 = ((split_23 < num_splits) ? _vec_load_20[2] : 0.0f);
                    float _add_193;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(sums_2[10]), "f"(value_23_34));
                    sums_2[10] = _add_193;
                    float value_23_35 = ((split_23 < num_splits) ? _vec_load_20[3] : 0.0f);
                    float _add_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(sums_2[26]), "f"(value_23_35));
                    sums_2[26] = _add_194;
#endif
                    float _vec_load_21[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 12) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 12) + 0);
#endif
                        _vec_load_21[0 + 0] = _v4.x;
                        _vec_load_21[0 + 1] = _v4.y;
                        _vec_load_21[0 + 2] = _v4.z;
                        _vec_load_21[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_84 = ((split_23_1 < num_splits) ? _vec_load_21[0] : 0.0f);
                    float _add_121;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(sums_1[12]), "f"(value_23_84));
                    sums_1[12] = _add_121;
                    float value_23_85 = ((split_23_1 < num_splits) ? _vec_load_21[1] : 0.0f);
                    float _add_122;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(sums_1[13]), "f"(value_23_85));
                    sums_1[13] = _add_122;
                    float value_23_86 = ((split_23_1 < num_splits) ? _vec_load_21[2] : 0.0f);
                    float _add_123;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(sums_1[14]), "f"(value_23_86));
                    sums_1[14] = _add_123;
                    float value_23_87 = ((split_23_1 < num_splits) ? _vec_load_21[3] : 0.0f);
                    float _add_124;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(sums_1[15]), "f"(value_23_87));
                    sums_1[15] = _add_124;
#else
                    float value_23_36 = ((split_23 < num_splits) ? _vec_load_21[0] : 0.0f);
                    float _add_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(sums_2[6]), "f"(value_23_36));
                    sums_2[6] = _add_196;
                    float value_23_37 = ((split_23 < num_splits) ? _vec_load_21[1] : 0.0f);
                    float _add_197;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(sums_2[22]), "f"(value_23_37));
                    sums_2[22] = _add_197;
                    float value_23_38 = ((split_23 < num_splits) ? _vec_load_21[2] : 0.0f);
                    float _add_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(sums_2[14]), "f"(value_23_38));
                    sums_2[14] = _add_198;
                    float value_23_39 = ((split_23 < num_splits) ? _vec_load_21[3] : 0.0f);
                    float _add_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(sums_2[30]), "f"(value_23_39));
                    sums_2[30] = _add_199;
#endif
                    float _vec_load_22[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 16) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 16) + 0);
#endif
                        _vec_load_22[0 + 0] = _v4.x;
                        _vec_load_22[0 + 1] = _v4.y;
                        _vec_load_22[0 + 2] = _v4.z;
                        _vec_load_22[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_88 = ((split_23_1 < num_splits) ? _vec_load_22[0] : 0.0f);
                    float _add_126;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(sums_1[16]), "f"(value_23_88));
                    sums_1[16] = _add_126;
                    float value_23_89 = ((split_23_1 < num_splits) ? _vec_load_22[1] : 0.0f);
                    float _add_127;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(sums_1[17]), "f"(value_23_89));
                    sums_1[17] = _add_127;
                    float value_23_90 = ((split_23_1 < num_splits) ? _vec_load_22[2] : 0.0f);
                    float _add_128;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(sums_1[18]), "f"(value_23_90));
                    sums_1[18] = _add_128;
                    float value_23_91 = ((split_23_1 < num_splits) ? _vec_load_22[3] : 0.0f);
                    float _add_129;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(sums_1[19]), "f"(value_23_91));
                    sums_1[19] = _add_129;
#else
                    float value_23_40 = ((split_23 < num_splits) ? _vec_load_22[0] : 0.0f);
                    float _add_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(sums_2[1]), "f"(value_23_40));
                    sums_2[1] = _add_201;
                    float value_23_41 = ((split_23 < num_splits) ? _vec_load_22[1] : 0.0f);
                    float _add_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(sums_2[17]), "f"(value_23_41));
                    sums_2[17] = _add_202;
                    float value_23_42 = ((split_23 < num_splits) ? _vec_load_22[2] : 0.0f);
                    float _add_203;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(sums_2[9]), "f"(value_23_42));
                    sums_2[9] = _add_203;
                    float value_23_43 = ((split_23 < num_splits) ? _vec_load_22[3] : 0.0f);
                    float _add_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(sums_2[25]), "f"(value_23_43));
                    sums_2[25] = _add_204;
#endif
                    float _vec_load_23[4];
                    {
#if __CUDA_ARCH__ == 1070
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 20) + 0);
#else
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 20) + 0);
#endif
                        _vec_load_23[0 + 0] = _v4.x;
                        _vec_load_23[0 + 1] = _v4.y;
                        _vec_load_23[0 + 2] = _v4.z;
                        _vec_load_23[0 + 3] = _v4.w;
                    }
#if __CUDA_ARCH__ == 1070
                    float value_23_92 = ((split_23_1 < num_splits) ? _vec_load_23[0] : 0.0f);
                    float _add_131;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(sums_1[20]), "f"(value_23_92));
                    sums_1[20] = _add_131;
                    float value_23_93 = ((split_23_1 < num_splits) ? _vec_load_23[1] : 0.0f);
                    float _add_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(sums_1[21]), "f"(value_23_93));
                    sums_1[21] = _add_132;
                    float value_23_94 = ((split_23_1 < num_splits) ? _vec_load_23[2] : 0.0f);
                    float _add_133;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(sums_1[22]), "f"(value_23_94));
                    sums_1[22] = _add_133;
                    float value_23_95 = ((split_23_1 < num_splits) ? _vec_load_23[3] : 0.0f);
                    float _add_134;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(sums_1[23]), "f"(value_23_95));
                    sums_1[23] = _add_134;
                    float q_word_50_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_24_1 * 64) + (unsigned long long)row_8];
                    float q_value_51_1 = ((split_23_1 < num_splits) ? q_word_50_1 : 0.0f);
                    float _add_135;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(sums_1[24]), "f"(q_value_51_1));
                    sums_1[24] = _add_135;
                    float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 1);
                    float _add_136;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(sums_1[0]), "f"(_shfl_xor_sync_137));
                    sums_1[0] = _add_136;
                    float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 1);
                    float _add_138;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(sums_1[1]), "f"(_shfl_xor_sync_139));
                    sums_1[1] = _add_138;
                    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 1);
                    float _add_140;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(sums_1[2]), "f"(_shfl_xor_sync_141));
                    sums_1[2] = _add_140;
                    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 1);
                    float _add_142;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(sums_1[3]), "f"(_shfl_xor_sync_143));
                    sums_1[3] = _add_142;
                    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 1);
                    float _add_144;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(sums_1[4]), "f"(_shfl_xor_sync_145));
                    sums_1[4] = _add_144;
                    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, sums_1[5], 1);
                    float _add_146;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(sums_1[5]), "f"(_shfl_xor_sync_147));
                    sums_1[5] = _add_146;
                    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, sums_1[6], 1);
                    float _add_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(sums_1[6]), "f"(_shfl_xor_sync_149));
                    sums_1[6] = _add_148;
                    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, sums_1[7], 1);
                    float _add_150;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(sums_1[7]), "f"(_shfl_xor_sync_151));
                    sums_1[7] = _add_150;
                    float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, sums_1[8], 1);
                    float _add_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(sums_1[8]), "f"(_shfl_xor_sync_153));
                    sums_1[8] = _add_152;
                    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, sums_1[9], 1);
                    float _add_154;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(sums_1[9]), "f"(_shfl_xor_sync_155));
                    sums_1[9] = _add_154;
                    float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, sums_1[10], 1);
                    float _add_156;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(sums_1[10]), "f"(_shfl_xor_sync_157));
                    sums_1[10] = _add_156;
                    float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 1);
                    float _add_158;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(sums_1[11]), "f"(_shfl_xor_sync_159));
                    sums_1[11] = _add_158;
                    float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 1);
                    float _add_160;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(sums_1[12]), "f"(_shfl_xor_sync_161));
                    sums_1[12] = _add_160;
                    float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, sums_1[13], 1);
                    float _add_162;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(sums_1[13]), "f"(_shfl_xor_sync_163));
                    sums_1[13] = _add_162;
                    float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, sums_1[14], 1);
                    float _add_164;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(sums_1[14]), "f"(_shfl_xor_sync_165));
                    sums_1[14] = _add_164;
                    float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, sums_1[15], 1);
                    float _add_166;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(sums_1[15]), "f"(_shfl_xor_sync_167));
                    sums_1[15] = _add_166;
                    float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, sums_1[16], 1);
                    float _add_168;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(sums_1[16]), "f"(_shfl_xor_sync_169));
                    sums_1[16] = _add_168;
                    float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, sums_1[17], 1);
                    float _add_170;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(sums_1[17]), "f"(_shfl_xor_sync_171));
                    sums_1[17] = _add_170;
                    float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, sums_1[18], 1);
                    float _add_172;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(sums_1[18]), "f"(_shfl_xor_sync_173));
                    sums_1[18] = _add_172;
                    float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, sums_1[19], 1);
                    float _add_174;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(sums_1[19]), "f"(_shfl_xor_sync_175));
                    sums_1[19] = _add_174;
                    float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, sums_1[20], 1);
                    float _add_176;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(sums_1[20]), "f"(_shfl_xor_sync_177));
                    sums_1[20] = _add_176;
                    float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, sums_1[21], 1);
                    float _add_178;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(sums_1[21]), "f"(_shfl_xor_sync_179));
                    sums_1[21] = _add_178;
                    float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, sums_1[22], 1);
                    float _add_180;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(sums_1[22]), "f"(_shfl_xor_sync_181));
                    sums_1[22] = _add_180;
                    float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, sums_1[23], 1);
                    float _add_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(sums_1[23]), "f"(_shfl_xor_sync_183));
                    sums_1[23] = _add_182;
                    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, sums_1[24], 1);
                    float _add_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(sums_1[24]), "f"(_shfl_xor_sync_185));
                    sums_1[24] = _add_184;
                    float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 2);
                    float _add_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(sums_1[0]), "f"(_shfl_xor_sync_187));
                    sums_1[0] = _add_186;
                    float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 2);
                    float _add_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(sums_1[1]), "f"(_shfl_xor_sync_189));
                    sums_1[1] = _add_188;
                    float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 2);
                    float _add_190;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(sums_1[2]), "f"(_shfl_xor_sync_191));
                    sums_1[2] = _add_190;
                    float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 2);
                    float _add_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(sums_1[3]), "f"(_shfl_xor_sync_193));
                    sums_1[3] = _add_192;
                    float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 2);
                    float _add_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(sums_1[4]), "f"(_shfl_xor_sync_195));
                    sums_1[4] = _add_194;
                    float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, sums_1[5], 2);
                    float _add_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(sums_1[5]), "f"(_shfl_xor_sync_197));
                    sums_1[5] = _add_196;
                    float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, sums_1[6], 2);
                    float _add_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(sums_1[6]), "f"(_shfl_xor_sync_199));
                    sums_1[6] = _add_198;
                    float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, sums_1[7], 2);
                    float _add_200;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(sums_1[7]), "f"(_shfl_xor_sync_201));
                    sums_1[7] = _add_200;
                    float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, sums_1[8], 2);
                    float _add_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(sums_1[8]), "f"(_shfl_xor_sync_203));
                    sums_1[8] = _add_202;
                    float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, sums_1[9], 2);
                    float _add_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(sums_1[9]), "f"(_shfl_xor_sync_205));
                    sums_1[9] = _add_204;
                    float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, sums_1[10], 2);
#else
                    float value_23_44 = ((split_23 < num_splits) ? _vec_load_23[0] : 0.0f);
#endif
                    float _add_206;
#if __CUDA_ARCH__ == 1070
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(sums_1[10]), "f"(_shfl_xor_sync_207));
                    sums_1[10] = _add_206;
                    float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 2);
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(sums_2[5]), "f"(value_23_44));
                    sums_2[5] = _add_206;
                    float value_23_45 = ((split_23 < num_splits) ? _vec_load_23[1] : 0.0f);
                    float _add_207;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(sums_2[21]), "f"(value_23_45));
                    sums_2[21] = _add_207;
                    float value_23_46 = ((split_23 < num_splits) ? _vec_load_23[2] : 0.0f);
#endif
                    float _add_208;
#if __CUDA_ARCH__ == 1070
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(sums_1[11]), "f"(_shfl_xor_sync_209));
                    sums_1[11] = _add_208;
                    float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 2);
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(sums_2[13]), "f"(value_23_46));
                    sums_2[13] = _add_208;
                    float value_23_47 = ((split_23 < num_splits) ? _vec_load_23[3] : 0.0f);
                    float _add_209;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(sums_2[29]), "f"(value_23_47));
                    sums_2[29] = _add_209;
                    float q_word_50 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_24 * 64) + (unsigned long long)row_11];
                    float q_value_51 = ((split_23 < num_splits) ? q_word_50 : 0.0f);
#endif
                    float _add_210;
#if __CUDA_ARCH__ == 1070
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(sums_1[12]), "f"(_shfl_xor_sync_211));
                    sums_1[12] = _add_210;
                    float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, sums_1[13], 2);
                    float _add_212;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(sums_1[13]), "f"(_shfl_xor_sync_213));
                    sums_1[13] = _add_212;
                    float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, sums_1[14], 2);
                    float _add_214;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(sums_1[14]), "f"(_shfl_xor_sync_215));
                    sums_1[14] = _add_214;
                    float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, sums_1[15], 2);
                    float _add_216;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(sums_1[15]), "f"(_shfl_xor_sync_217));
                    sums_1[15] = _add_216;
                    float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, sums_1[16], 2);
                    float _add_218;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(sums_1[16]), "f"(_shfl_xor_sync_219));
                    sums_1[16] = _add_218;
                    float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, sums_1[17], 2);
                    float _add_220;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(sums_1[17]), "f"(_shfl_xor_sync_221));
                    sums_1[17] = _add_220;
                    float _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, sums_1[18], 2);
                    float _add_222;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(sums_1[18]), "f"(_shfl_xor_sync_223));
                    sums_1[18] = _add_222;
                    float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, sums_1[19], 2);
                    float _add_224;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(sums_1[19]), "f"(_shfl_xor_sync_225));
                    sums_1[19] = _add_224;
                    float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, sums_1[20], 2);
                    float _add_226;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(sums_1[20]), "f"(_shfl_xor_sync_227));
                    sums_1[20] = _add_226;
                    float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, sums_1[21], 2);
                    float _add_228;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_228) : "f"(sums_1[21]), "f"(_shfl_xor_sync_229));
                    sums_1[21] = _add_228;
                    float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, sums_1[22], 2);
                    float _add_230;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(sums_1[22]), "f"(_shfl_xor_sync_231));
                    sums_1[22] = _add_230;
                    float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, sums_1[23], 2);
                    float _add_232;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(sums_1[23]), "f"(_shfl_xor_sync_233));
                    sums_1[23] = _add_232;
                    float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, sums_1[24], 2);
                    float _add_234;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(sums_1[24]), "f"(_shfl_xor_sync_235));
                    sums_1[24] = _add_234;
                    float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 4);
                    float _add_236;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(sums_1[0]), "f"(_shfl_xor_sync_237));
                    sums_1[0] = _add_236;
                    float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 4);
                    float _add_238;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(sums_1[1]), "f"(_shfl_xor_sync_239));
                    sums_1[1] = _add_238;
                    float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 4);
                    float _add_240;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(sums_1[2]), "f"(_shfl_xor_sync_241));
                    sums_1[2] = _add_240;
                    float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 4);
                    float _add_242;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(sums_1[3]), "f"(_shfl_xor_sync_243));
                    sums_1[3] = _add_242;
                    float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 4);
                    float _add_244;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(sums_1[4]), "f"(_shfl_xor_sync_245));
                    sums_1[4] = _add_244;
                    float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, sums_1[5], 4);
                    float _add_246;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(sums_1[5]), "f"(_shfl_xor_sync_247));
                    sums_1[5] = _add_246;
                    float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, sums_1[6], 4);
                    float _add_248;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_248) : "f"(sums_1[6]), "f"(_shfl_xor_sync_249));
                    sums_1[6] = _add_248;
                    float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, sums_1[7], 4);
                    float _add_250;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(sums_1[7]), "f"(_shfl_xor_sync_251));
                    sums_1[7] = _add_250;
                    float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, sums_1[8], 4);
                    float _add_252;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(sums_1[8]), "f"(_shfl_xor_sync_253));
                    sums_1[8] = _add_252;
                    float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, sums_1[9], 4);
                    float _add_254;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(sums_1[9]), "f"(_shfl_xor_sync_255));
                    sums_1[9] = _add_254;
                    float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, sums_1[10], 4);
                    float _add_256;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(sums_1[10]), "f"(_shfl_xor_sync_257));
                    sums_1[10] = _add_256;
                    float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 4);
                    float _add_258;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(sums_1[11]), "f"(_shfl_xor_sync_259));
                    sums_1[11] = _add_258;
                    float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 4);
                    float _add_260;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(sums_1[12]), "f"(_shfl_xor_sync_261));
                    sums_1[12] = _add_260;
                    float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, sums_1[13], 4);
                    float _add_262;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(sums_1[13]), "f"(_shfl_xor_sync_263));
                    sums_1[13] = _add_262;
                    float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, sums_1[14], 4);
                    float _add_264;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(sums_1[14]), "f"(_shfl_xor_sync_265));
                    sums_1[14] = _add_264;
                    float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, sums_1[15], 4);
                    float _add_266;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(sums_1[15]), "f"(_shfl_xor_sync_267));
                    sums_1[15] = _add_266;
                    float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, sums_1[16], 4);
                    float _add_268;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(sums_1[16]), "f"(_shfl_xor_sync_269));
                    sums_1[16] = _add_268;
                    float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, sums_1[17], 4);
                    float _add_270;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(sums_1[17]), "f"(_shfl_xor_sync_271));
                    sums_1[17] = _add_270;
                    float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, sums_1[18], 4);
                    float _add_272;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(sums_1[18]), "f"(_shfl_xor_sync_273));
                    sums_1[18] = _add_272;
                    float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, sums_1[19], 4);
                    float _add_274;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(sums_1[19]), "f"(_shfl_xor_sync_275));
                    sums_1[19] = _add_274;
                    float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, sums_1[20], 4);
                    float _add_276;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(sums_1[20]), "f"(_shfl_xor_sync_277));
                    sums_1[20] = _add_276;
                    float _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, sums_1[21], 4);
                    float _add_278;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(sums_1[21]), "f"(_shfl_xor_sync_279));
                    sums_1[21] = _add_278;
                    float _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, sums_1[22], 4);
                    float _add_280;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(sums_1[22]), "f"(_shfl_xor_sync_281));
                    sums_1[22] = _add_280;
                    float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, sums_1[23], 4);
                    float _add_282;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(sums_1[23]), "f"(_shfl_xor_sync_283));
                    sums_1[23] = _add_282;
                    float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, sums_1[24], 4);
                    float _add_284;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(sums_1[24]), "f"(_shfl_xor_sync_285));
                    sums_1[24] = _add_284;
                    float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 8);
                    float _add_286;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(sums_1[0]), "f"(_shfl_xor_sync_287));
                    sums_1[0] = _add_286;
                    float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 8);
                    float _add_288;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(sums_1[1]), "f"(_shfl_xor_sync_289));
                    sums_1[1] = _add_288;
                    float _shfl_xor_sync_291 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 8);
                    float _add_290;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(sums_1[2]), "f"(_shfl_xor_sync_291));
                    sums_1[2] = _add_290;
                    float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 8);
                    float _add_292;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(sums_1[3]), "f"(_shfl_xor_sync_293));
                    sums_1[3] = _add_292;
                    float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 8);
                    float _add_294;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(sums_1[4]), "f"(_shfl_xor_sync_295));
                    sums_1[4] = _add_294;
                    float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, sums_1[5], 8);
                    float _add_296;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_296) : "f"(sums_1[5]), "f"(_shfl_xor_sync_297));
                    sums_1[5] = _add_296;
                    float _shfl_xor_sync_299 = __shfl_xor_sync(0xFFFFFFFF, sums_1[6], 8);
                    float _add_298;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(sums_1[6]), "f"(_shfl_xor_sync_299));
                    sums_1[6] = _add_298;
                    float _shfl_xor_sync_301 = __shfl_xor_sync(0xFFFFFFFF, sums_1[7], 8);
                    float _add_300;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_300) : "f"(sums_1[7]), "f"(_shfl_xor_sync_301));
                    sums_1[7] = _add_300;
                    float _shfl_xor_sync_303 = __shfl_xor_sync(0xFFFFFFFF, sums_1[8], 8);
                    float _add_302;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_302) : "f"(sums_1[8]), "f"(_shfl_xor_sync_303));
                    sums_1[8] = _add_302;
                    float _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, sums_1[9], 8);
                    float _add_304;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_304) : "f"(sums_1[9]), "f"(_shfl_xor_sync_305));
                    sums_1[9] = _add_304;
                    float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, sums_1[10], 8);
                    float _add_306;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_306) : "f"(sums_1[10]), "f"(_shfl_xor_sync_307));
                    sums_1[10] = _add_306;
                    float _shfl_xor_sync_309 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 8);
                    float _add_308;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_308) : "f"(sums_1[11]), "f"(_shfl_xor_sync_309));
                    sums_1[11] = _add_308;
                    float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 8);
                    float _add_310;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_310) : "f"(sums_1[12]), "f"(_shfl_xor_sync_311));
                    sums_1[12] = _add_310;
                    float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, sums_1[13], 8);
                    float _add_312;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(sums_1[13]), "f"(_shfl_xor_sync_313));
                    sums_1[13] = _add_312;
                    float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, sums_1[14], 8);
                    float _add_314;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_314) : "f"(sums_1[14]), "f"(_shfl_xor_sync_315));
                    sums_1[14] = _add_314;
                    float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, sums_1[15], 8);
                    float _add_316;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(sums_1[15]), "f"(_shfl_xor_sync_317));
                    sums_1[15] = _add_316;
                    float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, sums_1[16], 8);
                    float _add_318;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_318) : "f"(sums_1[16]), "f"(_shfl_xor_sync_319));
                    sums_1[16] = _add_318;
                    float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, sums_1[17], 8);
                    float _add_320;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(sums_1[17]), "f"(_shfl_xor_sync_321));
                    sums_1[17] = _add_320;
                    float _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, sums_1[18], 8);
                    float _add_322;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(sums_1[18]), "f"(_shfl_xor_sync_323));
                    sums_1[18] = _add_322;
                    float _shfl_xor_sync_325 = __shfl_xor_sync(0xFFFFFFFF, sums_1[19], 8);
                    float _add_324;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(sums_1[19]), "f"(_shfl_xor_sync_325));
                    sums_1[19] = _add_324;
                    float _shfl_xor_sync_327 = __shfl_xor_sync(0xFFFFFFFF, sums_1[20], 8);
                    float _add_326;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(sums_1[20]), "f"(_shfl_xor_sync_327));
                    sums_1[20] = _add_326;
                    float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, sums_1[21], 8);
                    float _add_328;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(sums_1[21]), "f"(_shfl_xor_sync_329));
                    sums_1[21] = _add_328;
                    float _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, sums_1[22], 8);
                    float _add_330;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(sums_1[22]), "f"(_shfl_xor_sync_331));
                    sums_1[22] = _add_330;
                    float _shfl_xor_sync_333 = __shfl_xor_sync(0xFFFFFFFF, sums_1[23], 8);
                    float _add_332;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(sums_1[23]), "f"(_shfl_xor_sync_333));
                    sums_1[23] = _add_332;
                    float _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, sums_1[24], 8);
                    float _add_334;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(sums_1[24]), "f"(_shfl_xor_sync_335));
                    sums_1[24] = _add_334;
                    float _shfl_xor_sync_337 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 16);
                    float _add_336;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_336) : "f"(sums_1[0]), "f"(_shfl_xor_sync_337));
                    sums_1[0] = _add_336;
                    float _shfl_xor_sync_339 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 16);
                    float _add_338;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_338) : "f"(sums_1[1]), "f"(_shfl_xor_sync_339));
                    sums_1[1] = _add_338;
                    float _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 16);
                    float _add_340;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(sums_1[2]), "f"(_shfl_xor_sync_341));
                    sums_1[2] = _add_340;
                    float _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 16);
                    float _add_342;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_342) : "f"(sums_1[3]), "f"(_shfl_xor_sync_343));
                    sums_1[3] = _add_342;
                    float _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 16);
                    float _add_344;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_344) : "f"(sums_1[4]), "f"(_shfl_xor_sync_345));
                    sums_1[4] = _add_344;
                    float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, sums_1[5], 16);
                    float _add_346;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(sums_1[5]), "f"(_shfl_xor_sync_347));
                    sums_1[5] = _add_346;
                    float _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, sums_1[6], 16);
                    float _add_348;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_348) : "f"(sums_1[6]), "f"(_shfl_xor_sync_349));
                    sums_1[6] = _add_348;
                    float _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, sums_1[7], 16);
                    float _add_350;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(sums_1[7]), "f"(_shfl_xor_sync_351));
                    sums_1[7] = _add_350;
                    float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, sums_1[8], 16);
                    float _add_352;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(sums_1[8]), "f"(_shfl_xor_sync_353));
                    sums_1[8] = _add_352;
                    float _shfl_xor_sync_355 = __shfl_xor_sync(0xFFFFFFFF, sums_1[9], 16);
                    float _add_354;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_354) : "f"(sums_1[9]), "f"(_shfl_xor_sync_355));
                    sums_1[9] = _add_354;
                    float _shfl_xor_sync_357 = __shfl_xor_sync(0xFFFFFFFF, sums_1[10], 16);
                    float _add_356;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_356) : "f"(sums_1[10]), "f"(_shfl_xor_sync_357));
                    sums_1[10] = _add_356;
                    float _shfl_xor_sync_359 = __shfl_xor_sync(0xFFFFFFFF, sums_1[11], 16);
                    float _add_358;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_358) : "f"(sums_1[11]), "f"(_shfl_xor_sync_359));
                    sums_1[11] = _add_358;
                    float _shfl_xor_sync_361 = __shfl_xor_sync(0xFFFFFFFF, sums_1[12], 16);
                    float _add_360;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_360) : "f"(sums_1[12]), "f"(_shfl_xor_sync_361));
                    sums_1[12] = _add_360;
                    float _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, sums_1[13], 16);
                    float _add_362;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_362) : "f"(sums_1[13]), "f"(_shfl_xor_sync_363));
                    sums_1[13] = _add_362;
                    float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, sums_1[14], 16);
                    float _add_364;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(sums_1[14]), "f"(_shfl_xor_sync_365));
                    sums_1[14] = _add_364;
                    float _shfl_xor_sync_367 = __shfl_xor_sync(0xFFFFFFFF, sums_1[15], 16);
                    float _add_366;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(sums_1[15]), "f"(_shfl_xor_sync_367));
                    sums_1[15] = _add_366;
                    float _shfl_xor_sync_369 = __shfl_xor_sync(0xFFFFFFFF, sums_1[16], 16);
                    float _add_368;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(sums_1[16]), "f"(_shfl_xor_sync_369));
                    sums_1[16] = _add_368;
                    float _shfl_xor_sync_371 = __shfl_xor_sync(0xFFFFFFFF, sums_1[17], 16);
                    float _add_370;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_370) : "f"(sums_1[17]), "f"(_shfl_xor_sync_371));
                    sums_1[17] = _add_370;
                    float _shfl_xor_sync_373 = __shfl_xor_sync(0xFFFFFFFF, sums_1[18], 16);
                    float _add_372;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_372) : "f"(sums_1[18]), "f"(_shfl_xor_sync_373));
                    sums_1[18] = _add_372;
                    float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, sums_1[19], 16);
                    float _add_374;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(sums_1[19]), "f"(_shfl_xor_sync_375));
                    sums_1[19] = _add_374;
                    float _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, sums_1[20], 16);
                    float _add_376;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(sums_1[20]), "f"(_shfl_xor_sync_377));
                    sums_1[20] = _add_376;
                    float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, sums_1[21], 16);
                    float _add_378;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(sums_1[21]), "f"(_shfl_xor_sync_379));
                    sums_1[21] = _add_378;
                    float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, sums_1[22], 16);
                    float _add_380;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_380) : "f"(sums_1[22]), "f"(_shfl_xor_sync_381));
                    sums_1[22] = _add_380;
                    float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, sums_1[23], 16);
                    float _add_382;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_382) : "f"(sums_1[23]), "f"(_shfl_xor_sync_383));
                    sums_1[23] = _add_382;
                    float _shfl_xor_sync_385 = __shfl_xor_sync(0xFFFFFFFF, sums_1[24], 16);
                    float _add_384;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_384) : "f"(sums_1[24]), "f"(_shfl_xor_sync_385));
                    sums_1[24] = _add_384;
                    float s_52 = 0.0f;
                    s_52 = ((lane_in_warp == 0) ? sums_1[0] : s_52);
                    s_52 = ((lane_in_warp == 1) ? sums_1[1] : s_52);
                    s_52 = ((lane_in_warp == 2) ? sums_1[2] : s_52);
                    s_52 = ((lane_in_warp == 3) ? sums_1[3] : s_52);
                    s_52 = ((lane_in_warp == 4) ? sums_1[4] : s_52);
                    s_52 = ((lane_in_warp == 5) ? sums_1[5] : s_52);
                    s_52 = ((lane_in_warp == 6) ? sums_1[6] : s_52);
                    s_52 = ((lane_in_warp == 7) ? sums_1[7] : s_52);
                    s_52 = ((lane_in_warp == 8) ? sums_1[8] : s_52);
                    s_52 = ((lane_in_warp == 9) ? sums_1[9] : s_52);
                    s_52 = ((lane_in_warp == 10) ? sums_1[10] : s_52);
                    s_52 = ((lane_in_warp == 11) ? sums_1[11] : s_52);
                    s_52 = ((lane_in_warp == 12) ? sums_1[12] : s_52);
                    s_52 = ((lane_in_warp == 13) ? sums_1[13] : s_52);
                    s_52 = ((lane_in_warp == 14) ? sums_1[14] : s_52);
                    s_52 = ((lane_in_warp == 15) ? sums_1[15] : s_52);
                    s_52 = ((lane_in_warp == 16) ? sums_1[16] : s_52);
                    s_52 = ((lane_in_warp == 17) ? sums_1[17] : s_52);
                    s_52 = ((lane_in_warp == 18) ? sums_1[18] : s_52);
                    s_52 = ((lane_in_warp == 19) ? sums_1[19] : s_52);
                    s_52 = ((lane_in_warp == 20) ? sums_1[20] : s_52);
                    s_52 = ((lane_in_warp == 21) ? sums_1[21] : s_52);
                    s_52 = ((lane_in_warp == 22) ? sums_1[22] : s_52);
                    s_52 = ((lane_in_warp == 23) ? sums_1[23] : s_52);
                    float q_53 = sums_1[24];
                    s = s_52;
                    q = q_53;
                }
                float _fma_387;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(q), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                float _sqrt_386;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_386) : "f"(_fma_387));
                float r = __frcp_rn(_sqrt_386);
                float z;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z) : "f"(r), "f"(s));
                float a_map;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
                float a = ((lane_in_warp < 24) ? a_map : 0.0f);
                float _sub_388;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_388) : "f"(0.0f), "f"(fabsf(a)));
                float exp = __nv_expf(_sub_388);
                float add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30) : "f"(1.0f), "f"(exp));
                float fdiv_rn = __fdiv_rn(1.0f, add_30);
                float fdiv_rn_1 = __fdiv_rn(exp, add_30);
                float sig = ((a >= 0.0f) ? fdiv_rn : fdiv_rn_1);
                float sneg = ((a >= 0.0f) ? fdiv_rn_1 : fdiv_rn);
                float add_30_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30_1) : "f"(sig), "f"(hc_pre_eps));
                float _mul_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(hc_post_scale), "f"(sig));
                float logit = ((lane_in_warp >= 8 && lane_in_warp < 24) ? a : 0.0f);
                float _shfl_xor_sync_390 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
                float _shfl_xor_sync_391 = __shfl_xor_sync(0xFFFFFFFF, logit, 2);
                float _shfl_xor_sync_392 = __shfl_xor_sync(0xFFFFFFFF, logit, 3);
                float _sub_389;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_389) : "f"(logit), "f"(fmaxf(fmaxf(logit, _shfl_xor_sync_390), fmaxf(_shfl_xor_sync_391, _shfl_xor_sync_392))));
                float _exp_1 = __nv_expf(_sub_389);
                float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
                float _add_395;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_395) : "f"(_exp_1), "f"(_shfl_xor_sync_396));
                float _shfl_xor_sync_398 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
                float _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
                float _add_397;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_397) : "f"(_shfl_xor_sync_398), "f"(_shfl_xor_sync_399));
                float _add_394;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_394) : "f"(_add_395), "f"(_add_397));
                float _fdiv_393 = __fdiv_rn(_exp_1, _add_394);
                float C;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(C) : "f"(_fdiv_393), "f"(sinkhorn_eps));
                float _shfl_xor_sync_404 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                float _add_403;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_403) : "f"(C), "f"(_shfl_xor_sync_404));
                float _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, C, 24);
                float _shfl_xor_sync_407 = __shfl_xor_sync(0xFFFFFFFF, C, 28);
                float _add_405;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_405) : "f"(_shfl_xor_sync_406), "f"(_shfl_xor_sync_407));
                float _add_402;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_402) : "f"(_add_403), "f"(_add_405));
                float _add_401;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_401) : "f"(_add_402), "f"(sinkhorn_eps));
                float _fdiv_400 = __fdiv_rn(C, _add_401);
                C = _fdiv_400;
                #pragma unroll 1
                for (unsigned int iteration = 1; iteration < num_sinkhorn_iters; iteration++) {
                    float _shfl_xor_sync_412 = __shfl_xor_sync(0xFFFFFFFF, C, 1);
                    float _add_411;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_411) : "f"(C), "f"(_shfl_xor_sync_412));
                    float _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, C, 2);
                    float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, C, 3);
                    float _add_413;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_413) : "f"(_shfl_xor_sync_414), "f"(_shfl_xor_sync_415));
                    float _add_410;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_410) : "f"(_add_411), "f"(_add_413));
                    float _add_409;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_409) : "f"(_add_410), "f"(sinkhorn_eps));
                    float _fdiv_408 = __fdiv_rn(C, _add_409);
                    C = _fdiv_408;
                    float _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                    float _add_419;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(C), "f"(_shfl_xor_sync_420));
                    float _shfl_xor_sync_422 = __shfl_xor_sync(0xFFFFFFFF, C, 24);
                    float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, C, 28);
                    float _add_421;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_421) : "f"(_shfl_xor_sync_422), "f"(_shfl_xor_sync_423));
                    float _add_418;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(_add_419), "f"(_add_421));
                    float _add_417;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_417) : "f"(_add_418), "f"(sinkhorn_eps));
                    float _fdiv_416 = __fdiv_rn(C, _add_417);
                    C = _fdiv_416;
                }
                if (lane_in_warp >= 8) {
                    if (lane_in_warp < 24) {
                        new_comb_res_mix[(unsigned long long)token * 16 + (unsigned long long)(lane_in_warp - 8)] = C;
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(sums_2[3]), "f"(q_value_51));
                    sums_2[3] = _add_210;
                    float fold_level1[16];
                    float keep_1_28 = ((lane_in_warp % 2 == 1) ? sums_2[16] : sums_2[0]);
                    float send_1_28 = ((lane_in_warp % 2 == 1) ? sums_2[0] : sums_2[16]);
                    float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, send_1_28, 1);
                    float _add_211;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(keep_1_28), "f"(_shfl_xor_sync_212));
                    fold_level1[0] = _add_211;
                    float keep_1_29 = ((lane_in_warp % 2 == 1) ? sums_2[17] : sums_2[1]);
                    float send_1_29 = ((lane_in_warp % 2 == 1) ? sums_2[1] : sums_2[17]);
                    float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, send_1_29, 1);
                    float _add_213;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(keep_1_29), "f"(_shfl_xor_sync_214));
                    fold_level1[1] = _add_213;
                    float keep_1_30 = ((lane_in_warp % 2 == 1) ? sums_2[18] : sums_2[2]);
                    float send_1_30 = ((lane_in_warp % 2 == 1) ? sums_2[2] : sums_2[18]);
                    float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, send_1_30, 1);
                    float _add_215;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(keep_1_30), "f"(_shfl_xor_sync_216));
                    fold_level1[2] = _add_215;
                    float keep_1_31 = ((lane_in_warp % 2 == 1) ? sums_2[19] : sums_2[3]);
                    float send_1_31 = ((lane_in_warp % 2 == 1) ? sums_2[3] : sums_2[19]);
                    float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, send_1_31, 1);
                    float _add_217;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(keep_1_31), "f"(_shfl_xor_sync_218));
                    fold_level1[3] = _add_217;
                    float keep_1_32 = ((lane_in_warp % 2 == 1) ? sums_2[20] : sums_2[4]);
                    float send_1_32 = ((lane_in_warp % 2 == 1) ? sums_2[4] : sums_2[20]);
                    float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, send_1_32, 1);
                    float _add_219;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(keep_1_32), "f"(_shfl_xor_sync_220));
                    fold_level1[4] = _add_219;
                    float keep_1_33 = ((lane_in_warp % 2 == 1) ? sums_2[21] : sums_2[5]);
                    float send_1_33 = ((lane_in_warp % 2 == 1) ? sums_2[5] : sums_2[21]);
                    float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, send_1_33, 1);
                    float _add_221;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(keep_1_33), "f"(_shfl_xor_sync_222));
                    fold_level1[5] = _add_221;
                    float keep_1_34 = ((lane_in_warp % 2 == 1) ? sums_2[22] : sums_2[6]);
                    float send_1_34 = ((lane_in_warp % 2 == 1) ? sums_2[6] : sums_2[22]);
                    float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, send_1_34, 1);
                    float _add_223;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(keep_1_34), "f"(_shfl_xor_sync_224));
                    fold_level1[6] = _add_223;
                    float keep_1_35 = ((lane_in_warp % 2 == 1) ? sums_2[23] : sums_2[7]);
                    float send_1_35 = ((lane_in_warp % 2 == 1) ? sums_2[7] : sums_2[23]);
                    float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, send_1_35, 1);
                    float _add_225;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(keep_1_35), "f"(_shfl_xor_sync_226));
                    fold_level1[7] = _add_225;
                    float keep_1_36 = ((lane_in_warp % 2 == 1) ? sums_2[24] : sums_2[8]);
                    float send_1_36 = ((lane_in_warp % 2 == 1) ? sums_2[8] : sums_2[24]);
                    float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, send_1_36, 1);
                    float _add_227;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(keep_1_36), "f"(_shfl_xor_sync_228));
                    fold_level1[8] = _add_227;
                    float keep_1_37 = ((lane_in_warp % 2 == 1) ? sums_2[25] : sums_2[9]);
                    float send_1_37 = ((lane_in_warp % 2 == 1) ? sums_2[9] : sums_2[25]);
                    float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, send_1_37, 1);
                    float _add_229;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(keep_1_37), "f"(_shfl_xor_sync_230));
                    fold_level1[9] = _add_229;
                    float keep_1_38 = ((lane_in_warp % 2 == 1) ? sums_2[26] : sums_2[10]);
                    float send_1_38 = ((lane_in_warp % 2 == 1) ? sums_2[10] : sums_2[26]);
                    float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, send_1_38, 1);
                    float _add_231;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(keep_1_38), "f"(_shfl_xor_sync_232));
                    fold_level1[10] = _add_231;
                    float keep_1_39 = ((lane_in_warp % 2 == 1) ? sums_2[27] : sums_2[11]);
                    float send_1_39 = ((lane_in_warp % 2 == 1) ? sums_2[11] : sums_2[27]);
                    float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, send_1_39, 1);
                    float _add_233;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(keep_1_39), "f"(_shfl_xor_sync_234));
                    fold_level1[11] = _add_233;
                    float keep_1_40 = ((lane_in_warp % 2 == 1) ? sums_2[28] : sums_2[12]);
                    float send_1_40 = ((lane_in_warp % 2 == 1) ? sums_2[12] : sums_2[28]);
                    float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, send_1_40, 1);
                    float _add_235;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(keep_1_40), "f"(_shfl_xor_sync_236));
                    fold_level1[12] = _add_235;
                    float keep_1_41 = ((lane_in_warp % 2 == 1) ? sums_2[29] : sums_2[13]);
                    float send_1_41 = ((lane_in_warp % 2 == 1) ? sums_2[13] : sums_2[29]);
                    float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, send_1_41, 1);
                    float _add_237;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(keep_1_41), "f"(_shfl_xor_sync_238));
                    fold_level1[13] = _add_237;
                    float keep_1_42 = ((lane_in_warp % 2 == 1) ? sums_2[30] : sums_2[14]);
                    float send_1_42 = ((lane_in_warp % 2 == 1) ? sums_2[14] : sums_2[30]);
                    float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, send_1_42, 1);
                    float _add_239;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(keep_1_42), "f"(_shfl_xor_sync_240));
                    fold_level1[14] = _add_239;
                    float keep_1_43 = ((lane_in_warp % 2 == 1) ? sums_2[31] : sums_2[15]);
                    float send_1_43 = ((lane_in_warp % 2 == 1) ? sums_2[15] : sums_2[31]);
                    float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, send_1_43, 1);
                    float _add_241;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(keep_1_43), "f"(_shfl_xor_sync_242));
                    fold_level1[15] = _add_241;
                    float fold_level2[8];
                    float keep_1_44 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[8] : fold_level1[0]);
                    float send_1_44 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[0] : fold_level1[8]);
                    float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, send_1_44, 2);
                    float _add_243;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(keep_1_44), "f"(_shfl_xor_sync_244));
                    fold_level2[0] = _add_243;
                    float keep_1_45 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[9] : fold_level1[1]);
                    float send_1_45 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[1] : fold_level1[9]);
                    float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, send_1_45, 2);
                    float _add_245;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(keep_1_45), "f"(_shfl_xor_sync_246));
                    fold_level2[1] = _add_245;
                    float keep_1_46 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[10] : fold_level1[2]);
                    float send_1_46 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[2] : fold_level1[10]);
                    float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, send_1_46, 2);
                    float _add_247;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(keep_1_46), "f"(_shfl_xor_sync_248));
                    fold_level2[2] = _add_247;
                    float keep_1_47 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[11] : fold_level1[3]);
                    float send_1_47 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[3] : fold_level1[11]);
                    float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, send_1_47, 2);
                    float _add_249;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(keep_1_47), "f"(_shfl_xor_sync_250));
                    fold_level2[3] = _add_249;
                    float keep_1_48 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[12] : fold_level1[4]);
                    float send_1_48 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[4] : fold_level1[12]);
                    float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, send_1_48, 2);
                    float _add_251;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(keep_1_48), "f"(_shfl_xor_sync_252));
                    fold_level2[4] = _add_251;
                    float keep_1_49 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[13] : fold_level1[5]);
                    float send_1_49 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[5] : fold_level1[13]);
                    float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, send_1_49, 2);
                    float _add_253;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(keep_1_49), "f"(_shfl_xor_sync_254));
                    fold_level2[5] = _add_253;
                    float keep_1_50 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[14] : fold_level1[6]);
                    float send_1_50 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[6] : fold_level1[14]);
                    float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, send_1_50, 2);
                    float _add_255;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(keep_1_50), "f"(_shfl_xor_sync_256));
                    fold_level2[6] = _add_255;
                    float keep_1_51 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[15] : fold_level1[7]);
                    float send_1_51 = ((lane_in_warp / 2 % 2 == 1) ? fold_level1[7] : fold_level1[15]);
                    float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, send_1_51, 2);
                    float _add_257;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(keep_1_51), "f"(_shfl_xor_sync_258));
                    fold_level2[7] = _add_257;
                    float fold_level3[4];
                    float keep_1_52 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[4] : fold_level2[0]);
                    float send_1_52 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[0] : fold_level2[4]);
                    float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, send_1_52, 4);
                    float _add_259;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(keep_1_52), "f"(_shfl_xor_sync_260));
                    fold_level3[0] = _add_259;
                    float keep_1_53 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[5] : fold_level2[1]);
                    float send_1_53 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[1] : fold_level2[5]);
                    float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, send_1_53, 4);
                    float _add_261;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(keep_1_53), "f"(_shfl_xor_sync_262));
                    fold_level3[1] = _add_261;
                    float keep_1_54 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[6] : fold_level2[2]);
                    float send_1_54 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[2] : fold_level2[6]);
                    float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, send_1_54, 4);
                    float _add_263;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(keep_1_54), "f"(_shfl_xor_sync_264));
                    fold_level3[2] = _add_263;
                    float keep_1_55 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[7] : fold_level2[3]);
                    float send_1_55 = ((lane_in_warp / 4 % 2 == 1) ? fold_level2[3] : fold_level2[7]);
                    float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, send_1_55, 4);
                    float _add_265;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(keep_1_55), "f"(_shfl_xor_sync_266));
                    fold_level3[3] = _add_265;
                    float fold_level4[2];
                    float keep_1_56 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[2] : fold_level3[0]);
                    float send_1_56 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[0] : fold_level3[2]);
                    float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, send_1_56, 8);
                    float _add_267;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(keep_1_56), "f"(_shfl_xor_sync_268));
                    fold_level4[0] = _add_267;
                    float keep_1_57 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[3] : fold_level3[1]);
                    float send_1_57 = ((lane_in_warp / 8 % 2 == 1) ? fold_level3[1] : fold_level3[3]);
                    float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, send_1_57, 8);
                    float _add_269;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(keep_1_57), "f"(_shfl_xor_sync_270));
                    fold_level4[1] = _add_269;
                    float fold_level5[1];
                    float keep_110 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[1] : fold_level4[0]);
                    float send_111 = ((lane_in_warp / 16 % 2 == 1) ? fold_level4[0] : fold_level4[1]);
                    float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, send_111, 16);
                    float _add_271;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(keep_110), "f"(_shfl_xor_sync_272));
                    fold_level5[0] = _add_271;
                    float total_1 = fold_level5[0];
                    float s_112 = ((lane_in_warp < 24) ? total_1 : 0.0f);
                    float q_113 = __shfl_sync(0xFFFFFFFF, total_1, 24);
                    s = s_112;
                    q = q_113;
                } else {
                    float s_0_1 = 0.0f;
                    float q_1 = 0.0f;
                    if (num_splits > 22) {
                        unsigned int row_12 = token % 64;
                        unsigned int first_task_3 = mb_1 * num_splits;
                        float sums_3[25];
                        float errs_1[25];
                        unsigned int split_3 = lane_in_warp * 2;
                        unsigned int safe_split_3 = ((split_3 < num_splits) ? split_3 : 0);
                        unsigned long long record_3 = ((unsigned long long)first_task_3 + (unsigned long long)safe_split_3) * 1536 + (unsigned long long)(row_12 * 24);
                        float _vec_load_24[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_3 + 0);
                            _vec_load_24[0 + 0] = _v4.x;
                            _vec_load_24[0 + 1] = _v4.y;
                            _vec_load_24[0 + 2] = _v4.z;
                            _vec_load_24[0 + 3] = _v4.w;
                        }
                        float value_23_48 = ((split_3 < num_splits) ? _vec_load_24[0] : 0.0f);
                        sums_3[0] = value_23_48;
                        float value_23_49 = ((split_3 < num_splits) ? _vec_load_24[1] : 0.0f);
                        sums_3[1] = value_23_49;
                        float value_23_50 = ((split_3 < num_splits) ? _vec_load_24[2] : 0.0f);
                        sums_3[2] = value_23_50;
                        float value_23_51 = ((split_3 < num_splits) ? _vec_load_24[3] : 0.0f);
                        sums_3[3] = value_23_51;
                        float _vec_load_25[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 4) + 0);
                            _vec_load_25[0 + 0] = _v4.x;
                            _vec_load_25[0 + 1] = _v4.y;
                            _vec_load_25[0 + 2] = _v4.z;
                            _vec_load_25[0 + 3] = _v4.w;
                        }
                        float value_23_52 = ((split_3 < num_splits) ? _vec_load_25[0] : 0.0f);
                        sums_3[4] = value_23_52;
                        float value_23_53 = ((split_3 < num_splits) ? _vec_load_25[1] : 0.0f);
                        sums_3[5] = value_23_53;
                        float value_23_54 = ((split_3 < num_splits) ? _vec_load_25[2] : 0.0f);
                        sums_3[6] = value_23_54;
                        float value_23_55 = ((split_3 < num_splits) ? _vec_load_25[3] : 0.0f);
                        sums_3[7] = value_23_55;
                        float _vec_load_26[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 8) + 0);
                            _vec_load_26[0 + 0] = _v4.x;
                            _vec_load_26[0 + 1] = _v4.y;
                            _vec_load_26[0 + 2] = _v4.z;
                            _vec_load_26[0 + 3] = _v4.w;
                        }
                        float value_23_56 = ((split_3 < num_splits) ? _vec_load_26[0] : 0.0f);
                        sums_3[8] = value_23_56;
                        float value_23_57 = ((split_3 < num_splits) ? _vec_load_26[1] : 0.0f);
                        sums_3[9] = value_23_57;
                        float value_23_58 = ((split_3 < num_splits) ? _vec_load_26[2] : 0.0f);
                        sums_3[10] = value_23_58;
                        float value_23_59 = ((split_3 < num_splits) ? _vec_load_26[3] : 0.0f);
                        sums_3[11] = value_23_59;
                        float _vec_load_27[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 12) + 0);
                            _vec_load_27[0 + 0] = _v4.x;
                            _vec_load_27[0 + 1] = _v4.y;
                            _vec_load_27[0 + 2] = _v4.z;
                            _vec_load_27[0 + 3] = _v4.w;
                        }
                        float value_23_60 = ((split_3 < num_splits) ? _vec_load_27[0] : 0.0f);
                        sums_3[12] = value_23_60;
                        float value_23_61 = ((split_3 < num_splits) ? _vec_load_27[1] : 0.0f);
                        sums_3[13] = value_23_61;
                        float value_23_62 = ((split_3 < num_splits) ? _vec_load_27[2] : 0.0f);
                        sums_3[14] = value_23_62;
                        float value_23_63 = ((split_3 < num_splits) ? _vec_load_27[3] : 0.0f);
                        sums_3[15] = value_23_63;
                        float _vec_load_28[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 16) + 0);
                            _vec_load_28[0 + 0] = _v4.x;
                            _vec_load_28[0 + 1] = _v4.y;
                            _vec_load_28[0 + 2] = _v4.z;
                            _vec_load_28[0 + 3] = _v4.w;
                        }
                        float value_23_64 = ((split_3 < num_splits) ? _vec_load_28[0] : 0.0f);
                        sums_3[16] = value_23_64;
                        float value_23_65 = ((split_3 < num_splits) ? _vec_load_28[1] : 0.0f);
                        sums_3[17] = value_23_65;
                        float value_23_66 = ((split_3 < num_splits) ? _vec_load_28[2] : 0.0f);
                        sums_3[18] = value_23_66;
                        float value_23_67 = ((split_3 < num_splits) ? _vec_load_28[3] : 0.0f);
                        sums_3[19] = value_23_67;
                        float _vec_load_29[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_3 + 20) + 0);
                            _vec_load_29[0 + 0] = _v4.x;
                            _vec_load_29[0 + 1] = _v4.y;
                            _vec_load_29[0 + 2] = _v4.z;
                            _vec_load_29[0 + 3] = _v4.w;
                        }
                        float value_23_68 = ((split_3 < num_splits) ? _vec_load_29[0] : 0.0f);
                        sums_3[20] = value_23_68;
                        float value_23_69 = ((split_3 < num_splits) ? _vec_load_29[1] : 0.0f);
                        sums_3[21] = value_23_69;
                        float value_23_70 = ((split_3 < num_splits) ? _vec_load_29[2] : 0.0f);
                        sums_3[22] = value_23_70;
                        float value_23_71 = ((split_3 < num_splits) ? _vec_load_29[3] : 0.0f);
                        sums_3[23] = value_23_71;
                        float q_word_3 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_3 * 64 + (unsigned long long)(safe_split_3 * 64) + (unsigned long long)row_12];
                        float q_value_3 = ((split_3 < num_splits) ? q_word_3 : 0.0f);
                        sums_3[24] = q_value_3;
                        unsigned int split_23_1 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_24_1 = ((split_23_1 < num_splits) ? split_23_1 : 0);
                        unsigned long long record_25_1 = ((unsigned long long)first_task_3 + (unsigned long long)safe_split_24_1) * 1536 + (unsigned long long)(row_12 * 24);
                        float _vec_load_30[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_1 + 0);
                            _vec_load_30[0 + 0] = _v4.x;
                            _vec_load_30[0 + 1] = _v4.y;
                            _vec_load_30[0 + 2] = _v4.z;
                            _vec_load_30[0 + 3] = _v4.w;
                        }
                        float value_23_72 = ((split_23_1 < num_splits) ? _vec_load_30[0] : 0.0f);
                        float s_29;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(sums_3[0]), "f"(value_23_72));
                        float bb_28;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_28) : "f"(s_29), "f"(sums_3[0]));
                        float _sub_281;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_281) : "f"(s_29), "f"(bb_28));
                        float _sub_280;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(sums_3[0]), "f"(_sub_281));
                        float _sub_282;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(value_23_72), "f"(bb_28));
                        float e_28;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_28) : "f"(_sub_280), "f"(_sub_282));
                        sums_3[0] = s_29;
                        errs_1[0] = e_28;
                        float value_23_73 = ((split_23_1 < num_splits) ? _vec_load_30[1] : 0.0f);
                        float s_30;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(sums_3[1]), "f"(value_23_73));
                        float bb_28_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_28_1) : "f"(s_30), "f"(sums_3[1]));
                        float _sub_284;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(s_30), "f"(bb_28_1));
                        float _sub_283;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_283) : "f"(sums_3[1]), "f"(_sub_284));
                        float _sub_285;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_285) : "f"(value_23_73), "f"(bb_28_1));
                        float e_28_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_28_1) : "f"(_sub_283), "f"(_sub_285));
                        sums_3[1] = s_30;
                        errs_1[1] = e_28_1;
                        float value_23_74 = ((split_23_1 < num_splits) ? _vec_load_30[2] : 0.0f);
                        float s_31;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(sums_3[2]), "f"(value_23_74));
                        float bb_29;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_29) : "f"(s_31), "f"(sums_3[2]));
                        float _sub_287;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_287) : "f"(s_31), "f"(bb_29));
                        float _sub_286;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(sums_3[2]), "f"(_sub_287));
                        float _sub_288;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_288) : "f"(value_23_74), "f"(bb_29));
                        float e_29;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_29) : "f"(_sub_286), "f"(_sub_288));
                        sums_3[2] = s_31;
                        errs_1[2] = e_29;
                        float value_23_75 = ((split_23_1 < num_splits) ? _vec_load_30[3] : 0.0f);
                        float s_32;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(sums_3[3]), "f"(value_23_75));
                        float bb_30;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_30) : "f"(s_32), "f"(sums_3[3]));
                        float _sub_290;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_290) : "f"(s_32), "f"(bb_30));
                        float _sub_289;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(sums_3[3]), "f"(_sub_290));
                        float _sub_291;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(value_23_75), "f"(bb_30));
                        float e_30;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_30) : "f"(_sub_289), "f"(_sub_291));
                        sums_3[3] = s_32;
                        errs_1[3] = e_30;
                        float _vec_load_31[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 4) + 0);
                            _vec_load_31[0 + 0] = _v4.x;
                            _vec_load_31[0 + 1] = _v4.y;
                            _vec_load_31[0 + 2] = _v4.z;
                            _vec_load_31[0 + 3] = _v4.w;
                        }
                        float value_23_76 = ((split_23_1 < num_splits) ? _vec_load_31[0] : 0.0f);
                        float s_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(sums_3[4]), "f"(value_23_76));
                        float bb_31;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_31) : "f"(s_33), "f"(sums_3[4]));
                        float _sub_294;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_294) : "f"(s_33), "f"(bb_31));
                        float _sub_293;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(sums_3[4]), "f"(_sub_294));
                        float _sub_295;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(value_23_76), "f"(bb_31));
                        float e_31;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_31) : "f"(_sub_293), "f"(_sub_295));
                        sums_3[4] = s_33;
                        errs_1[4] = e_31;
                        float value_23_77 = ((split_23_1 < num_splits) ? _vec_load_31[1] : 0.0f);
                        float s_34;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(sums_3[5]), "f"(value_23_77));
                        float bb_32;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_32) : "f"(s_34), "f"(sums_3[5]));
                        float _sub_297;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_297) : "f"(s_34), "f"(bb_32));
                        float _sub_296;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_296) : "f"(sums_3[5]), "f"(_sub_297));
                        float _sub_298;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(value_23_77), "f"(bb_32));
                        float e_32;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_32) : "f"(_sub_296), "f"(_sub_298));
                        sums_3[5] = s_34;
                        errs_1[5] = e_32;
                        float value_23_78 = ((split_23_1 < num_splits) ? _vec_load_31[2] : 0.0f);
                        float s_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(sums_3[6]), "f"(value_23_78));
                        float bb_33;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_33) : "f"(s_35), "f"(sums_3[6]));
                        float _sub_300;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(s_35), "f"(bb_33));
                        float _sub_299;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_299) : "f"(sums_3[6]), "f"(_sub_300));
                        float _sub_301;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(value_23_78), "f"(bb_33));
                        float e_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_33) : "f"(_sub_299), "f"(_sub_301));
                        sums_3[6] = s_35;
                        errs_1[6] = e_33;
                        float value_23_79 = ((split_23_1 < num_splits) ? _vec_load_31[3] : 0.0f);
                        float s_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(sums_3[7]), "f"(value_23_79));
                        float bb_34;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_34) : "f"(s_36), "f"(sums_3[7]));
                        float _sub_303;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_303) : "f"(s_36), "f"(bb_34));
                        float _sub_302;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_302) : "f"(sums_3[7]), "f"(_sub_303));
                        float _sub_304;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_304) : "f"(value_23_79), "f"(bb_34));
                        float e_34;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_34) : "f"(_sub_302), "f"(_sub_304));
                        sums_3[7] = s_36;
                        errs_1[7] = e_34;
                        float _vec_load_32[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 8) + 0);
                            _vec_load_32[0 + 0] = _v4.x;
                            _vec_load_32[0 + 1] = _v4.y;
                            _vec_load_32[0 + 2] = _v4.z;
                            _vec_load_32[0 + 3] = _v4.w;
                        }
                        float value_23_80 = ((split_23_1 < num_splits) ? _vec_load_32[0] : 0.0f);
                        float s_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(sums_3[8]), "f"(value_23_80));
                        float bb_35;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_35) : "f"(s_37), "f"(sums_3[8]));
                        float _sub_307;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_307) : "f"(s_37), "f"(bb_35));
                        float _sub_306;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_306) : "f"(sums_3[8]), "f"(_sub_307));
                        float _sub_308;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_308) : "f"(value_23_80), "f"(bb_35));
                        float e_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_35) : "f"(_sub_306), "f"(_sub_308));
                        sums_3[8] = s_37;
                        errs_1[8] = e_35;
                        float value_23_81 = ((split_23_1 < num_splits) ? _vec_load_32[1] : 0.0f);
                        float s_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(sums_3[9]), "f"(value_23_81));
                        float bb_36;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_36) : "f"(s_38), "f"(sums_3[9]));
                        float _sub_310;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_310) : "f"(s_38), "f"(bb_36));
                        float _sub_309;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_309) : "f"(sums_3[9]), "f"(_sub_310));
                        float _sub_311;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_311) : "f"(value_23_81), "f"(bb_36));
                        float e_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_36) : "f"(_sub_309), "f"(_sub_311));
                        sums_3[9] = s_38;
                        errs_1[9] = e_36;
                        float value_23_82 = ((split_23_1 < num_splits) ? _vec_load_32[2] : 0.0f);
                        float s_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(sums_3[10]), "f"(value_23_82));
                        float bb_37;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_37) : "f"(s_39), "f"(sums_3[10]));
                        float _sub_313;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(s_39), "f"(bb_37));
                        float _sub_312;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_312) : "f"(sums_3[10]), "f"(_sub_313));
                        float _sub_314;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_314) : "f"(value_23_82), "f"(bb_37));
                        float e_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_37) : "f"(_sub_312), "f"(_sub_314));
                        sums_3[10] = s_39;
                        errs_1[10] = e_37;
                        float value_23_83 = ((split_23_1 < num_splits) ? _vec_load_32[3] : 0.0f);
                        float s_40;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(sums_3[11]), "f"(value_23_83));
                        float bb_38;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_38) : "f"(s_40), "f"(sums_3[11]));
                        float _sub_316;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_316) : "f"(s_40), "f"(bb_38));
                        float _sub_315;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_315) : "f"(sums_3[11]), "f"(_sub_316));
                        float _sub_317;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_317) : "f"(value_23_83), "f"(bb_38));
                        float e_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_38) : "f"(_sub_315), "f"(_sub_317));
                        sums_3[11] = s_40;
                        errs_1[11] = e_38;
                        float _vec_load_33[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 12) + 0);
                            _vec_load_33[0 + 0] = _v4.x;
                            _vec_load_33[0 + 1] = _v4.y;
                            _vec_load_33[0 + 2] = _v4.z;
                            _vec_load_33[0 + 3] = _v4.w;
                        }
                        float value_23_84 = ((split_23_1 < num_splits) ? _vec_load_33[0] : 0.0f);
                        float s_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(sums_3[12]), "f"(value_23_84));
                        float bb_39;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_39) : "f"(s_41), "f"(sums_3[12]));
                        float _sub_320;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_320) : "f"(s_41), "f"(bb_39));
                        float _sub_319;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(sums_3[12]), "f"(_sub_320));
                        float _sub_321;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_321) : "f"(value_23_84), "f"(bb_39));
                        float e_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_39) : "f"(_sub_319), "f"(_sub_321));
                        sums_3[12] = s_41;
                        errs_1[12] = e_39;
                        float value_23_85 = ((split_23_1 < num_splits) ? _vec_load_33[1] : 0.0f);
                        float s_42;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(sums_3[13]), "f"(value_23_85));
                        float bb_40;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_40) : "f"(s_42), "f"(sums_3[13]));
                        float _sub_323;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(s_42), "f"(bb_40));
                        float _sub_322;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_322) : "f"(sums_3[13]), "f"(_sub_323));
                        float _sub_324;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_324) : "f"(value_23_85), "f"(bb_40));
                        float e_40;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_40) : "f"(_sub_322), "f"(_sub_324));
                        sums_3[13] = s_42;
                        errs_1[13] = e_40;
                        float value_23_86 = ((split_23_1 < num_splits) ? _vec_load_33[2] : 0.0f);
                        float s_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(sums_3[14]), "f"(value_23_86));
                        float bb_41;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_41) : "f"(s_43), "f"(sums_3[14]));
                        float _sub_326;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_326) : "f"(s_43), "f"(bb_41));
                        float _sub_325;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(sums_3[14]), "f"(_sub_326));
                        float _sub_327;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(value_23_86), "f"(bb_41));
                        float e_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_41) : "f"(_sub_325), "f"(_sub_327));
                        sums_3[14] = s_43;
                        errs_1[14] = e_41;
                        float value_23_87 = ((split_23_1 < num_splits) ? _vec_load_33[3] : 0.0f);
                        float s_44;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(sums_3[15]), "f"(value_23_87));
                        float bb_42;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_42) : "f"(s_44), "f"(sums_3[15]));
                        float _sub_329;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(s_44), "f"(bb_42));
                        float _sub_328;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_328) : "f"(sums_3[15]), "f"(_sub_329));
                        float _sub_330;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(value_23_87), "f"(bb_42));
                        float e_42;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_42) : "f"(_sub_328), "f"(_sub_330));
                        sums_3[15] = s_44;
                        errs_1[15] = e_42;
                        float _vec_load_34[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 16) + 0);
                            _vec_load_34[0 + 0] = _v4.x;
                            _vec_load_34[0 + 1] = _v4.y;
                            _vec_load_34[0 + 2] = _v4.z;
                            _vec_load_34[0 + 3] = _v4.w;
                        }
                        float value_23_88 = ((split_23_1 < num_splits) ? _vec_load_34[0] : 0.0f);
                        float s_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(sums_3[16]), "f"(value_23_88));
                        float bb_43;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_43) : "f"(s_45), "f"(sums_3[16]));
                        float _sub_333;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_333) : "f"(s_45), "f"(bb_43));
                        float _sub_332;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_332) : "f"(sums_3[16]), "f"(_sub_333));
                        float _sub_334;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(value_23_88), "f"(bb_43));
                        float e_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_43) : "f"(_sub_332), "f"(_sub_334));
                        sums_3[16] = s_45;
                        errs_1[16] = e_43;
                        float value_23_89 = ((split_23_1 < num_splits) ? _vec_load_34[1] : 0.0f);
                        float s_46;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(sums_3[17]), "f"(value_23_89));
                        float bb_44;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_44) : "f"(s_46), "f"(sums_3[17]));
                        float _sub_336;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(s_46), "f"(bb_44));
                        float _sub_335;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_335) : "f"(sums_3[17]), "f"(_sub_336));
                        float _sub_337;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_337) : "f"(value_23_89), "f"(bb_44));
                        float e_44;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_44) : "f"(_sub_335), "f"(_sub_337));
                        sums_3[17] = s_46;
                        errs_1[17] = e_44;
                        float value_23_90 = ((split_23_1 < num_splits) ? _vec_load_34[2] : 0.0f);
                        float s_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(sums_3[18]), "f"(value_23_90));
                        float bb_45;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_45) : "f"(s_47), "f"(sums_3[18]));
                        float _sub_339;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_339) : "f"(s_47), "f"(bb_45));
                        float _sub_338;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_338) : "f"(sums_3[18]), "f"(_sub_339));
                        float _sub_340;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_340) : "f"(value_23_90), "f"(bb_45));
                        float e_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_45) : "f"(_sub_338), "f"(_sub_340));
                        sums_3[18] = s_47;
                        errs_1[18] = e_45;
                        float value_23_91 = ((split_23_1 < num_splits) ? _vec_load_34[3] : 0.0f);
                        float s_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(sums_3[19]), "f"(value_23_91));
                        float bb_46;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_46) : "f"(s_48), "f"(sums_3[19]));
                        float _sub_342;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_342) : "f"(s_48), "f"(bb_46));
                        float _sub_341;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_341) : "f"(sums_3[19]), "f"(_sub_342));
                        float _sub_343;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_343) : "f"(value_23_91), "f"(bb_46));
                        float e_46;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_46) : "f"(_sub_341), "f"(_sub_343));
                        sums_3[19] = s_48;
                        errs_1[19] = e_46;
                        float _vec_load_35[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_1 + 20) + 0);
                            _vec_load_35[0 + 0] = _v4.x;
                            _vec_load_35[0 + 1] = _v4.y;
                            _vec_load_35[0 + 2] = _v4.z;
                            _vec_load_35[0 + 3] = _v4.w;
                        }
                        float value_23_92 = ((split_23_1 < num_splits) ? _vec_load_35[0] : 0.0f);
                        float s_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(sums_3[20]), "f"(value_23_92));
                        float bb_47;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_47) : "f"(s_49), "f"(sums_3[20]));
                        float _sub_346;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_346) : "f"(s_49), "f"(bb_47));
                        float _sub_345;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_345) : "f"(sums_3[20]), "f"(_sub_346));
                        float _sub_347;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_347) : "f"(value_23_92), "f"(bb_47));
                        float e_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_47) : "f"(_sub_345), "f"(_sub_347));
                        sums_3[20] = s_49;
                        errs_1[20] = e_47;
                        float value_23_93 = ((split_23_1 < num_splits) ? _vec_load_35[1] : 0.0f);
                        float s_50;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(sums_3[21]), "f"(value_23_93));
                        float bb_48;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_48) : "f"(s_50), "f"(sums_3[21]));
                        float _sub_349;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(s_50), "f"(bb_48));
                        float _sub_348;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_348) : "f"(sums_3[21]), "f"(_sub_349));
                        float _sub_350;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_350) : "f"(value_23_93), "f"(bb_48));
                        float e_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_48) : "f"(_sub_348), "f"(_sub_350));
                        sums_3[21] = s_50;
                        errs_1[21] = e_48;
                        float value_23_94 = ((split_23_1 < num_splits) ? _vec_load_35[2] : 0.0f);
                        float s_51;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(sums_3[22]), "f"(value_23_94));
                        float bb_49;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_49) : "f"(s_51), "f"(sums_3[22]));
                        float _sub_352;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_352) : "f"(s_51), "f"(bb_49));
                        float _sub_351;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_351) : "f"(sums_3[22]), "f"(_sub_352));
                        float _sub_353;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_353) : "f"(value_23_94), "f"(bb_49));
                        float e_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_49) : "f"(_sub_351), "f"(_sub_353));
                        sums_3[22] = s_51;
                        errs_1[22] = e_49;
                        float value_23_95 = ((split_23_1 < num_splits) ? _vec_load_35[3] : 0.0f);
                        float s_52;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(sums_3[23]), "f"(value_23_95));
                        float bb_50;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_50) : "f"(s_52), "f"(sums_3[23]));
                        float _sub_355;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_355) : "f"(s_52), "f"(bb_50));
                        float _sub_354;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(sums_3[23]), "f"(_sub_355));
                        float _sub_356;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_356) : "f"(value_23_95), "f"(bb_50));
                        float e_50;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_50) : "f"(_sub_354), "f"(_sub_356));
                        sums_3[23] = s_52;
                        errs_1[23] = e_50;
                        float q_word_120 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_3 * 64 + (unsigned long long)(safe_split_24_1 * 64) + (unsigned long long)row_12];
                        float q_value_121 = ((split_23_1 < num_splits) ? q_word_120 : 0.0f);
                        float s_53;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(sums_3[24]), "f"(q_value_121));
                        float bb_51;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_51) : "f"(s_53), "f"(sums_3[24]));
                        float _sub_358;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_358) : "f"(s_53), "f"(bb_51));
                        float _sub_357;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_357) : "f"(sums_3[24]), "f"(_sub_358));
                        float _sub_359;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_359) : "f"(q_value_121), "f"(bb_51));
                        float e_51;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_51) : "f"(_sub_357), "f"(_sub_359));
                        sums_3[24] = s_53;
                        errs_1[24] = e_51;
                        float partner_1_0 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 1);
                        float partner_err_1_0 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 1);
                        float s_54;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(sums_3[0]), "f"(partner_1_0));
                        float bb_52;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_52) : "f"(s_54), "f"(sums_3[0]));
                        float _sub_361;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_361) : "f"(s_54), "f"(bb_52));
                        float _sub_360;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_360) : "f"(sums_3[0]), "f"(_sub_361));
                        float _sub_362;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_362) : "f"(partner_1_0), "f"(bb_52));
                        float e_52;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_52) : "f"(_sub_360), "f"(_sub_362));
                        sums_3[0] = s_54;
                        float _add_364;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(errs_1[0]), "f"(partner_err_1_0));
                        float _add_363;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_363) : "f"(_add_364), "f"(e_52));
                        errs_1[0] = _add_363;
                        float partner_1_1 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 1);
                        float partner_err_1_1 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 1);
                        float s_55;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(sums_3[1]), "f"(partner_1_1));
                        float bb_53;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_53) : "f"(s_55), "f"(sums_3[1]));
                        float _sub_366;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_366) : "f"(s_55), "f"(bb_53));
                        float _sub_365;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(sums_3[1]), "f"(_sub_366));
                        float _sub_367;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_367) : "f"(partner_1_1), "f"(bb_53));
                        float e_53;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_53) : "f"(_sub_365), "f"(_sub_367));
                        sums_3[1] = s_55;
                        float _add_369;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_369) : "f"(errs_1[1]), "f"(partner_err_1_1));
                        float _add_368;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(_add_369), "f"(e_53));
                        errs_1[1] = _add_368;
                        float partner_1_2 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 1);
                        float partner_err_1_2 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 1);
                        float s_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(sums_3[2]), "f"(partner_1_2));
                        float bb_54;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_54) : "f"(s_56), "f"(sums_3[2]));
                        float _sub_371;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_371) : "f"(s_56), "f"(bb_54));
                        float _sub_370;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_370) : "f"(sums_3[2]), "f"(_sub_371));
                        float _sub_372;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_372) : "f"(partner_1_2), "f"(bb_54));
                        float e_54;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_54) : "f"(_sub_370), "f"(_sub_372));
                        sums_3[2] = s_56;
                        float _add_374;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(errs_1[2]), "f"(partner_err_1_2));
                        float _add_373;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_373) : "f"(_add_374), "f"(e_54));
                        errs_1[2] = _add_373;
                        float partner_1_3 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 1);
                        float partner_err_1_3 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 1);
                        float s_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(sums_3[3]), "f"(partner_1_3));
                        float bb_55;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_55) : "f"(s_57), "f"(sums_3[3]));
                        float _sub_376;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_376) : "f"(s_57), "f"(bb_55));
                        float _sub_375;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_375) : "f"(sums_3[3]), "f"(_sub_376));
                        float _sub_377;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_377) : "f"(partner_1_3), "f"(bb_55));
                        float e_55;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_55) : "f"(_sub_375), "f"(_sub_377));
                        sums_3[3] = s_57;
                        float _add_379;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_379) : "f"(errs_1[3]), "f"(partner_err_1_3));
                        float _add_378;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(_add_379), "f"(e_55));
                        errs_1[3] = _add_378;
                        float partner_1_4 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 1);
                        float partner_err_1_4 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 1);
                        float s_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(sums_3[4]), "f"(partner_1_4));
                        float bb_56;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_56) : "f"(s_58), "f"(sums_3[4]));
                        float _sub_381;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_381) : "f"(s_58), "f"(bb_56));
                        float _sub_380;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_380) : "f"(sums_3[4]), "f"(_sub_381));
                        float _sub_382;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_382) : "f"(partner_1_4), "f"(bb_56));
                        float e_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_56) : "f"(_sub_380), "f"(_sub_382));
                        sums_3[4] = s_58;
                        float _add_384;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_384) : "f"(errs_1[4]), "f"(partner_err_1_4));
                        float _add_383;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_383) : "f"(_add_384), "f"(e_56));
                        errs_1[4] = _add_383;
                        float partner_1_5 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 1);
                        float partner_err_1_5 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 1);
                        float s_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(sums_3[5]), "f"(partner_1_5));
                        float bb_57;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_57) : "f"(s_59), "f"(sums_3[5]));
                        float _sub_386;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_386) : "f"(s_59), "f"(bb_57));
                        float _sub_385;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_385) : "f"(sums_3[5]), "f"(_sub_386));
                        float _sub_387;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_387) : "f"(partner_1_5), "f"(bb_57));
                        float e_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_57) : "f"(_sub_385), "f"(_sub_387));
                        sums_3[5] = s_59;
                        float _add_389;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_389) : "f"(errs_1[5]), "f"(partner_err_1_5));
                        float _add_388;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_388) : "f"(_add_389), "f"(e_57));
                        errs_1[5] = _add_388;
                        float partner_1_6 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 1);
                        float partner_err_1_6 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 1);
                        float s_60;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(sums_3[6]), "f"(partner_1_6));
                        float bb_58;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_58) : "f"(s_60), "f"(sums_3[6]));
                        float _sub_391;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_391) : "f"(s_60), "f"(bb_58));
                        float _sub_390;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_390) : "f"(sums_3[6]), "f"(_sub_391));
                        float _sub_392;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_392) : "f"(partner_1_6), "f"(bb_58));
                        float e_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_58) : "f"(_sub_390), "f"(_sub_392));
                        sums_3[6] = s_60;
                        float _add_394;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_394) : "f"(errs_1[6]), "f"(partner_err_1_6));
                        float _add_393;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_393) : "f"(_add_394), "f"(e_58));
                        errs_1[6] = _add_393;
                        float partner_1_7 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 1);
                        float partner_err_1_7 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 1);
                        float s_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(sums_3[7]), "f"(partner_1_7));
                        float bb_59;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_59) : "f"(s_61), "f"(sums_3[7]));
                        float _sub_396;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_396) : "f"(s_61), "f"(bb_59));
                        float _sub_395;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_395) : "f"(sums_3[7]), "f"(_sub_396));
                        float _sub_397;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_397) : "f"(partner_1_7), "f"(bb_59));
                        float e_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_59) : "f"(_sub_395), "f"(_sub_397));
                        sums_3[7] = s_61;
                        float _add_399;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_399) : "f"(errs_1[7]), "f"(partner_err_1_7));
                        float _add_398;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_398) : "f"(_add_399), "f"(e_59));
                        errs_1[7] = _add_398;
                        float partner_1_8 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 1);
                        float partner_err_1_8 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 1);
                        float s_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(sums_3[8]), "f"(partner_1_8));
                        float bb_60;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_60) : "f"(s_62), "f"(sums_3[8]));
                        float _sub_401;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_401) : "f"(s_62), "f"(bb_60));
                        float _sub_400;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(sums_3[8]), "f"(_sub_401));
                        float _sub_402;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_402) : "f"(partner_1_8), "f"(bb_60));
                        float e_60;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_60) : "f"(_sub_400), "f"(_sub_402));
                        sums_3[8] = s_62;
                        float _add_404;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_404) : "f"(errs_1[8]), "f"(partner_err_1_8));
                        float _add_403;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_403) : "f"(_add_404), "f"(e_60));
                        errs_1[8] = _add_403;
                        float partner_1_9 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 1);
                        float partner_err_1_9 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 1);
                        float s_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(sums_3[9]), "f"(partner_1_9));
                        float bb_61;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_61) : "f"(s_63), "f"(sums_3[9]));
                        float _sub_406;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_406) : "f"(s_63), "f"(bb_61));
                        float _sub_405;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_405) : "f"(sums_3[9]), "f"(_sub_406));
                        float _sub_407;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_407) : "f"(partner_1_9), "f"(bb_61));
                        float e_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_61) : "f"(_sub_405), "f"(_sub_407));
                        sums_3[9] = s_63;
                        float _add_409;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_409) : "f"(errs_1[9]), "f"(partner_err_1_9));
                        float _add_408;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(_add_409), "f"(e_61));
                        errs_1[9] = _add_408;
                        float partner_1_10 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 1);
                        float partner_err_1_10 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 1);
                        float s_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(sums_3[10]), "f"(partner_1_10));
                        float bb_62;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_62) : "f"(s_64), "f"(sums_3[10]));
                        float _sub_411;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_411) : "f"(s_64), "f"(bb_62));
                        float _sub_410;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_410) : "f"(sums_3[10]), "f"(_sub_411));
                        float _sub_412;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_412) : "f"(partner_1_10), "f"(bb_62));
                        float e_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_62) : "f"(_sub_410), "f"(_sub_412));
                        sums_3[10] = s_64;
                        float _add_414;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_414) : "f"(errs_1[10]), "f"(partner_err_1_10));
                        float _add_413;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_413) : "f"(_add_414), "f"(e_62));
                        errs_1[10] = _add_413;
                        float partner_1_11 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 1);
                        float partner_err_1_11 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 1);
                        float s_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(sums_3[11]), "f"(partner_1_11));
                        float bb_63;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_63) : "f"(s_65), "f"(sums_3[11]));
                        float _sub_416;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_416) : "f"(s_65), "f"(bb_63));
                        float _sub_415;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_415) : "f"(sums_3[11]), "f"(_sub_416));
                        float _sub_417;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_417) : "f"(partner_1_11), "f"(bb_63));
                        float e_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_63) : "f"(_sub_415), "f"(_sub_417));
                        sums_3[11] = s_65;
                        float _add_419;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(errs_1[11]), "f"(partner_err_1_11));
                        float _add_418;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(_add_419), "f"(e_63));
                        errs_1[11] = _add_418;
                        float partner_1_12 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 1);
                        float partner_err_1_12 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 1);
                        float s_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(sums_3[12]), "f"(partner_1_12));
                        float bb_64;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_64) : "f"(s_66), "f"(sums_3[12]));
                        float _sub_421;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_421) : "f"(s_66), "f"(bb_64));
                        float _sub_420;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_420) : "f"(sums_3[12]), "f"(_sub_421));
                        float _sub_422;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_422) : "f"(partner_1_12), "f"(bb_64));
                        float e_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_64) : "f"(_sub_420), "f"(_sub_422));
                        sums_3[12] = s_66;
                        float _add_424;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_424) : "f"(errs_1[12]), "f"(partner_err_1_12));
                        float _add_423;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(_add_424), "f"(e_64));
                        errs_1[12] = _add_423;
                        float partner_1_13 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 1);
                        float partner_err_1_13 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 1);
                        float s_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(sums_3[13]), "f"(partner_1_13));
                        float bb_65;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_65) : "f"(s_67), "f"(sums_3[13]));
                        float _sub_426;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_426) : "f"(s_67), "f"(bb_65));
                        float _sub_425;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_425) : "f"(sums_3[13]), "f"(_sub_426));
                        float _sub_427;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_427) : "f"(partner_1_13), "f"(bb_65));
                        float e_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_65) : "f"(_sub_425), "f"(_sub_427));
                        sums_3[13] = s_67;
                        float _add_429;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_429) : "f"(errs_1[13]), "f"(partner_err_1_13));
                        float _add_428;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_428) : "f"(_add_429), "f"(e_65));
                        errs_1[13] = _add_428;
                        float partner_1_14 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 1);
                        float partner_err_1_14 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 1);
                        float s_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(sums_3[14]), "f"(partner_1_14));
                        float bb_66;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_66) : "f"(s_68), "f"(sums_3[14]));
                        float _sub_431;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_431) : "f"(s_68), "f"(bb_66));
                        float _sub_430;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_430) : "f"(sums_3[14]), "f"(_sub_431));
                        float _sub_432;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_432) : "f"(partner_1_14), "f"(bb_66));
                        float e_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_66) : "f"(_sub_430), "f"(_sub_432));
                        sums_3[14] = s_68;
                        float _add_434;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(errs_1[14]), "f"(partner_err_1_14));
                        float _add_433;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(_add_434), "f"(e_66));
                        errs_1[14] = _add_433;
                        float partner_1_15 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 1);
                        float partner_err_1_15 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 1);
                        float s_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(sums_3[15]), "f"(partner_1_15));
                        float bb_67;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_67) : "f"(s_69), "f"(sums_3[15]));
                        float _sub_436;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_436) : "f"(s_69), "f"(bb_67));
                        float _sub_435;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_435) : "f"(sums_3[15]), "f"(_sub_436));
                        float _sub_437;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_437) : "f"(partner_1_15), "f"(bb_67));
                        float e_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_67) : "f"(_sub_435), "f"(_sub_437));
                        sums_3[15] = s_69;
                        float _add_439;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_439) : "f"(errs_1[15]), "f"(partner_err_1_15));
                        float _add_438;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(_add_439), "f"(e_67));
                        errs_1[15] = _add_438;
                        float partner_1_16 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 1);
                        float partner_err_1_16 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 1);
                        float s_70;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_70) : "f"(sums_3[16]), "f"(partner_1_16));
                        float bb_68;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_68) : "f"(s_70), "f"(sums_3[16]));
                        float _sub_441;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_441) : "f"(s_70), "f"(bb_68));
                        float _sub_440;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_440) : "f"(sums_3[16]), "f"(_sub_441));
                        float _sub_442;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_442) : "f"(partner_1_16), "f"(bb_68));
                        float e_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_68) : "f"(_sub_440), "f"(_sub_442));
                        sums_3[16] = s_70;
                        float _add_444;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_444) : "f"(errs_1[16]), "f"(partner_err_1_16));
                        float _add_443;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_443) : "f"(_add_444), "f"(e_68));
                        errs_1[16] = _add_443;
                        float partner_1_17 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 1);
                        float partner_err_1_17 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 1);
                        float s_71;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_71) : "f"(sums_3[17]), "f"(partner_1_17));
                        float bb_69;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_69) : "f"(s_71), "f"(sums_3[17]));
                        float _sub_446;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(s_71), "f"(bb_69));
                        float _sub_445;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_445) : "f"(sums_3[17]), "f"(_sub_446));
                        float _sub_447;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_447) : "f"(partner_1_17), "f"(bb_69));
                        float e_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_69) : "f"(_sub_445), "f"(_sub_447));
                        sums_3[17] = s_71;
                        float _add_449;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(errs_1[17]), "f"(partner_err_1_17));
                        float _add_448;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_448) : "f"(_add_449), "f"(e_69));
                        errs_1[17] = _add_448;
                        float partner_1_18 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 1);
                        float partner_err_1_18 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 1);
                        float s_72;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_72) : "f"(sums_3[18]), "f"(partner_1_18));
                        float bb_70;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_70) : "f"(s_72), "f"(sums_3[18]));
                        float _sub_451;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_451) : "f"(s_72), "f"(bb_70));
                        float _sub_450;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_450) : "f"(sums_3[18]), "f"(_sub_451));
                        float _sub_452;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(partner_1_18), "f"(bb_70));
                        float e_70;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_70) : "f"(_sub_450), "f"(_sub_452));
                        sums_3[18] = s_72;
                        float _add_454;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_454) : "f"(errs_1[18]), "f"(partner_err_1_18));
                        float _add_453;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(_add_454), "f"(e_70));
                        errs_1[18] = _add_453;
                        float partner_1_19 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 1);
                        float partner_err_1_19 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 1);
                        float s_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_73) : "f"(sums_3[19]), "f"(partner_1_19));
                        float bb_71;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_71) : "f"(s_73), "f"(sums_3[19]));
                        float _sub_456;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_456) : "f"(s_73), "f"(bb_71));
                        float _sub_455;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_455) : "f"(sums_3[19]), "f"(_sub_456));
                        float _sub_457;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_457) : "f"(partner_1_19), "f"(bb_71));
                        float e_71;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_71) : "f"(_sub_455), "f"(_sub_457));
                        sums_3[19] = s_73;
                        float _add_459;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_459) : "f"(errs_1[19]), "f"(partner_err_1_19));
                        float _add_458;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_458) : "f"(_add_459), "f"(e_71));
                        errs_1[19] = _add_458;
                        float partner_1_20 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 1);
                        float partner_err_1_20 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 1);
                        float s_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_74) : "f"(sums_3[20]), "f"(partner_1_20));
                        float bb_72;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_72) : "f"(s_74), "f"(sums_3[20]));
                        float _sub_461;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_461) : "f"(s_74), "f"(bb_72));
                        float _sub_460;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_460) : "f"(sums_3[20]), "f"(_sub_461));
                        float _sub_462;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_462) : "f"(partner_1_20), "f"(bb_72));
                        float e_72;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_72) : "f"(_sub_460), "f"(_sub_462));
                        sums_3[20] = s_74;
                        float _add_464;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(errs_1[20]), "f"(partner_err_1_20));
                        float _add_463;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_463) : "f"(_add_464), "f"(e_72));
                        errs_1[20] = _add_463;
                        float partner_1_21 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 1);
                        float partner_err_1_21 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 1);
                        float s_75;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_75) : "f"(sums_3[21]), "f"(partner_1_21));
                        float bb_73;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_73) : "f"(s_75), "f"(sums_3[21]));
                        float _sub_466;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_466) : "f"(s_75), "f"(bb_73));
                        float _sub_465;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_465) : "f"(sums_3[21]), "f"(_sub_466));
                        float _sub_467;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_467) : "f"(partner_1_21), "f"(bb_73));
                        float e_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_73) : "f"(_sub_465), "f"(_sub_467));
                        sums_3[21] = s_75;
                        float _add_469;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_469) : "f"(errs_1[21]), "f"(partner_err_1_21));
                        float _add_468;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(_add_469), "f"(e_73));
                        errs_1[21] = _add_468;
                        float partner_1_22 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 1);
                        float partner_err_1_22 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 1);
                        float s_76;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_76) : "f"(sums_3[22]), "f"(partner_1_22));
                        float bb_74;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_74) : "f"(s_76), "f"(sums_3[22]));
                        float _sub_471;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_471) : "f"(s_76), "f"(bb_74));
                        float _sub_470;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_470) : "f"(sums_3[22]), "f"(_sub_471));
                        float _sub_472;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_472) : "f"(partner_1_22), "f"(bb_74));
                        float e_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_74) : "f"(_sub_470), "f"(_sub_472));
                        sums_3[22] = s_76;
                        float _add_474;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(errs_1[22]), "f"(partner_err_1_22));
                        float _add_473;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_473) : "f"(_add_474), "f"(e_74));
                        errs_1[22] = _add_473;
                        float partner_1_23 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 1);
                        float partner_err_1_23 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 1);
                        float s_77;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_77) : "f"(sums_3[23]), "f"(partner_1_23));
                        float bb_75;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_75) : "f"(s_77), "f"(sums_3[23]));
                        float _sub_476;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_476) : "f"(s_77), "f"(bb_75));
                        float _sub_475;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_475) : "f"(sums_3[23]), "f"(_sub_476));
                        float _sub_477;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_477) : "f"(partner_1_23), "f"(bb_75));
                        float e_75;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_75) : "f"(_sub_475), "f"(_sub_477));
                        sums_3[23] = s_77;
                        float _add_479;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_479) : "f"(errs_1[23]), "f"(partner_err_1_23));
                        float _add_478;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_478) : "f"(_add_479), "f"(e_75));
                        errs_1[23] = _add_478;
                        float partner_1_24 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 1);
                        float partner_err_1_24 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 1);
                        float s_78;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_78) : "f"(sums_3[24]), "f"(partner_1_24));
                        float bb_76;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_76) : "f"(s_78), "f"(sums_3[24]));
                        float _sub_481;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_481) : "f"(s_78), "f"(bb_76));
                        float _sub_480;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_480) : "f"(sums_3[24]), "f"(_sub_481));
                        float _sub_482;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_482) : "f"(partner_1_24), "f"(bb_76));
                        float e_76;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_76) : "f"(_sub_480), "f"(_sub_482));
                        sums_3[24] = s_78;
                        float _add_484;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_484) : "f"(errs_1[24]), "f"(partner_err_1_24));
                        float _add_483;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_483) : "f"(_add_484), "f"(e_76));
                        errs_1[24] = _add_483;
                        float partner_1_25 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 2);
                        float partner_err_1_25 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 2);
                        float s_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_79) : "f"(sums_3[0]), "f"(partner_1_25));
                        float bb_77;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_77) : "f"(s_79), "f"(sums_3[0]));
                        float _sub_486;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_486) : "f"(s_79), "f"(bb_77));
                        float _sub_485;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_485) : "f"(sums_3[0]), "f"(_sub_486));
                        float _sub_487;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_487) : "f"(partner_1_25), "f"(bb_77));
                        float e_77;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_77) : "f"(_sub_485), "f"(_sub_487));
                        sums_3[0] = s_79;
                        float _add_489;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(errs_1[0]), "f"(partner_err_1_25));
                        float _add_488;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_488) : "f"(_add_489), "f"(e_77));
                        errs_1[0] = _add_488;
                        float partner_1_26 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 2);
                        float partner_err_1_26 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 2);
                        float s_80;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_80) : "f"(sums_3[1]), "f"(partner_1_26));
                        float bb_78;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_78) : "f"(s_80), "f"(sums_3[1]));
                        float _sub_491;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_491) : "f"(s_80), "f"(bb_78));
                        float _sub_490;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_490) : "f"(sums_3[1]), "f"(_sub_491));
                        float _sub_492;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_492) : "f"(partner_1_26), "f"(bb_78));
                        float e_78;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_78) : "f"(_sub_490), "f"(_sub_492));
                        sums_3[1] = s_80;
                        float _add_494;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_494) : "f"(errs_1[1]), "f"(partner_err_1_26));
                        float _add_493;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_493) : "f"(_add_494), "f"(e_78));
                        errs_1[1] = _add_493;
                        float partner_1_27 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 2);
                        float partner_err_1_27 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 2);
                        float s_81;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_81) : "f"(sums_3[2]), "f"(partner_1_27));
                        float bb_79;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_79) : "f"(s_81), "f"(sums_3[2]));
                        float _sub_496;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_496) : "f"(s_81), "f"(bb_79));
                        float _sub_495;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_495) : "f"(sums_3[2]), "f"(_sub_496));
                        float _sub_497;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_497) : "f"(partner_1_27), "f"(bb_79));
                        float e_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_79) : "f"(_sub_495), "f"(_sub_497));
                        sums_3[2] = s_81;
                        float _add_499;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_499) : "f"(errs_1[2]), "f"(partner_err_1_27));
                        float _add_498;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_498) : "f"(_add_499), "f"(e_79));
                        errs_1[2] = _add_498;
                        float partner_1_28 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 2);
                        float partner_err_1_28 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 2);
                        float s_82;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_82) : "f"(sums_3[3]), "f"(partner_1_28));
                        float bb_80;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_80) : "f"(s_82), "f"(sums_3[3]));
                        float _sub_501;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_501) : "f"(s_82), "f"(bb_80));
                        float _sub_500;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_500) : "f"(sums_3[3]), "f"(_sub_501));
                        float _sub_502;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_502) : "f"(partner_1_28), "f"(bb_80));
                        float e_80;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_80) : "f"(_sub_500), "f"(_sub_502));
                        sums_3[3] = s_82;
                        float _add_504;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_504) : "f"(errs_1[3]), "f"(partner_err_1_28));
                        float _add_503;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_503) : "f"(_add_504), "f"(e_80));
                        errs_1[3] = _add_503;
                        float partner_1_29 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 2);
                        float partner_err_1_29 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 2);
                        float s_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_83) : "f"(sums_3[4]), "f"(partner_1_29));
                        float bb_81;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_81) : "f"(s_83), "f"(sums_3[4]));
                        float _sub_506;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_506) : "f"(s_83), "f"(bb_81));
                        float _sub_505;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_505) : "f"(sums_3[4]), "f"(_sub_506));
                        float _sub_507;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_507) : "f"(partner_1_29), "f"(bb_81));
                        float e_81;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_81) : "f"(_sub_505), "f"(_sub_507));
                        sums_3[4] = s_83;
                        float _add_509;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_509) : "f"(errs_1[4]), "f"(partner_err_1_29));
                        float _add_508;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_508) : "f"(_add_509), "f"(e_81));
                        errs_1[4] = _add_508;
                        float partner_1_30 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 2);
                        float partner_err_1_30 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 2);
                        float s_84;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_84) : "f"(sums_3[5]), "f"(partner_1_30));
                        float bb_82;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_82) : "f"(s_84), "f"(sums_3[5]));
                        float _sub_511;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_511) : "f"(s_84), "f"(bb_82));
                        float _sub_510;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_510) : "f"(sums_3[5]), "f"(_sub_511));
                        float _sub_512;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_512) : "f"(partner_1_30), "f"(bb_82));
                        float e_82;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_82) : "f"(_sub_510), "f"(_sub_512));
                        sums_3[5] = s_84;
                        float _add_514;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_514) : "f"(errs_1[5]), "f"(partner_err_1_30));
                        float _add_513;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_513) : "f"(_add_514), "f"(e_82));
                        errs_1[5] = _add_513;
                        float partner_1_31 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 2);
                        float partner_err_1_31 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 2);
                        float s_85;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_85) : "f"(sums_3[6]), "f"(partner_1_31));
                        float bb_83;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_83) : "f"(s_85), "f"(sums_3[6]));
                        float _sub_516;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_516) : "f"(s_85), "f"(bb_83));
                        float _sub_515;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_515) : "f"(sums_3[6]), "f"(_sub_516));
                        float _sub_517;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_517) : "f"(partner_1_31), "f"(bb_83));
                        float e_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_83) : "f"(_sub_515), "f"(_sub_517));
                        sums_3[6] = s_85;
                        float _add_519;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_519) : "f"(errs_1[6]), "f"(partner_err_1_31));
                        float _add_518;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_518) : "f"(_add_519), "f"(e_83));
                        errs_1[6] = _add_518;
                        float partner_1_32 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 2);
                        float partner_err_1_32 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 2);
                        float s_86;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_86) : "f"(sums_3[7]), "f"(partner_1_32));
                        float bb_84;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_84) : "f"(s_86), "f"(sums_3[7]));
                        float _sub_521;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_521) : "f"(s_86), "f"(bb_84));
                        float _sub_520;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_520) : "f"(sums_3[7]), "f"(_sub_521));
                        float _sub_522;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_522) : "f"(partner_1_32), "f"(bb_84));
                        float e_84;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_84) : "f"(_sub_520), "f"(_sub_522));
                        sums_3[7] = s_86;
                        float _add_524;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_524) : "f"(errs_1[7]), "f"(partner_err_1_32));
                        float _add_523;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_523) : "f"(_add_524), "f"(e_84));
                        errs_1[7] = _add_523;
                        float partner_1_33 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 2);
                        float partner_err_1_33 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 2);
                        float s_87;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_87) : "f"(sums_3[8]), "f"(partner_1_33));
                        float bb_85;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_85) : "f"(s_87), "f"(sums_3[8]));
                        float _sub_526;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_526) : "f"(s_87), "f"(bb_85));
                        float _sub_525;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_525) : "f"(sums_3[8]), "f"(_sub_526));
                        float _sub_527;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_527) : "f"(partner_1_33), "f"(bb_85));
                        float e_85;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_85) : "f"(_sub_525), "f"(_sub_527));
                        sums_3[8] = s_87;
                        float _add_529;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_529) : "f"(errs_1[8]), "f"(partner_err_1_33));
                        float _add_528;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_528) : "f"(_add_529), "f"(e_85));
                        errs_1[8] = _add_528;
                        float partner_1_34 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 2);
                        float partner_err_1_34 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 2);
                        float s_88;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_88) : "f"(sums_3[9]), "f"(partner_1_34));
                        float bb_86;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_86) : "f"(s_88), "f"(sums_3[9]));
                        float _sub_531;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_531) : "f"(s_88), "f"(bb_86));
                        float _sub_530;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_530) : "f"(sums_3[9]), "f"(_sub_531));
                        float _sub_532;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_532) : "f"(partner_1_34), "f"(bb_86));
                        float e_86;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_86) : "f"(_sub_530), "f"(_sub_532));
                        sums_3[9] = s_88;
                        float _add_534;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_534) : "f"(errs_1[9]), "f"(partner_err_1_34));
                        float _add_533;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_533) : "f"(_add_534), "f"(e_86));
                        errs_1[9] = _add_533;
                        float partner_1_35 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 2);
                        float partner_err_1_35 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 2);
                        float s_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_89) : "f"(sums_3[10]), "f"(partner_1_35));
                        float bb_87;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_87) : "f"(s_89), "f"(sums_3[10]));
                        float _sub_536;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_536) : "f"(s_89), "f"(bb_87));
                        float _sub_535;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_535) : "f"(sums_3[10]), "f"(_sub_536));
                        float _sub_537;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_537) : "f"(partner_1_35), "f"(bb_87));
                        float e_87;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_87) : "f"(_sub_535), "f"(_sub_537));
                        sums_3[10] = s_89;
                        float _add_539;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_539) : "f"(errs_1[10]), "f"(partner_err_1_35));
                        float _add_538;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_538) : "f"(_add_539), "f"(e_87));
                        errs_1[10] = _add_538;
                        float partner_1_36 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 2);
                        float partner_err_1_36 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 2);
                        float s_90;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_90) : "f"(sums_3[11]), "f"(partner_1_36));
                        float bb_88;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_88) : "f"(s_90), "f"(sums_3[11]));
                        float _sub_541;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_541) : "f"(s_90), "f"(bb_88));
                        float _sub_540;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_540) : "f"(sums_3[11]), "f"(_sub_541));
                        float _sub_542;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_542) : "f"(partner_1_36), "f"(bb_88));
                        float e_88;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_88) : "f"(_sub_540), "f"(_sub_542));
                        sums_3[11] = s_90;
                        float _add_544;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_544) : "f"(errs_1[11]), "f"(partner_err_1_36));
                        float _add_543;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_543) : "f"(_add_544), "f"(e_88));
                        errs_1[11] = _add_543;
                        float partner_1_37 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 2);
                        float partner_err_1_37 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 2);
                        float s_91;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_91) : "f"(sums_3[12]), "f"(partner_1_37));
                        float bb_89;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_89) : "f"(s_91), "f"(sums_3[12]));
                        float _sub_546;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_546) : "f"(s_91), "f"(bb_89));
                        float _sub_545;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_545) : "f"(sums_3[12]), "f"(_sub_546));
                        float _sub_547;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_547) : "f"(partner_1_37), "f"(bb_89));
                        float e_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_89) : "f"(_sub_545), "f"(_sub_547));
                        sums_3[12] = s_91;
                        float _add_549;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_549) : "f"(errs_1[12]), "f"(partner_err_1_37));
                        float _add_548;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_548) : "f"(_add_549), "f"(e_89));
                        errs_1[12] = _add_548;
                        float partner_1_38 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 2);
                        float partner_err_1_38 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 2);
                        float s_92;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_92) : "f"(sums_3[13]), "f"(partner_1_38));
                        float bb_90;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_90) : "f"(s_92), "f"(sums_3[13]));
                        float _sub_551;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_551) : "f"(s_92), "f"(bb_90));
                        float _sub_550;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_550) : "f"(sums_3[13]), "f"(_sub_551));
                        float _sub_552;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_552) : "f"(partner_1_38), "f"(bb_90));
                        float e_90;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_90) : "f"(_sub_550), "f"(_sub_552));
                        sums_3[13] = s_92;
                        float _add_554;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_554) : "f"(errs_1[13]), "f"(partner_err_1_38));
                        float _add_553;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_553) : "f"(_add_554), "f"(e_90));
                        errs_1[13] = _add_553;
                        float partner_1_39 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 2);
                        float partner_err_1_39 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 2);
                        float s_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_93) : "f"(sums_3[14]), "f"(partner_1_39));
                        float bb_91;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_91) : "f"(s_93), "f"(sums_3[14]));
                        float _sub_556;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_556) : "f"(s_93), "f"(bb_91));
                        float _sub_555;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_555) : "f"(sums_3[14]), "f"(_sub_556));
                        float _sub_557;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_557) : "f"(partner_1_39), "f"(bb_91));
                        float e_91;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_91) : "f"(_sub_555), "f"(_sub_557));
                        sums_3[14] = s_93;
                        float _add_559;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_559) : "f"(errs_1[14]), "f"(partner_err_1_39));
                        float _add_558;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_558) : "f"(_add_559), "f"(e_91));
                        errs_1[14] = _add_558;
                        float partner_1_40 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 2);
                        float partner_err_1_40 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 2);
                        float s_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_94) : "f"(sums_3[15]), "f"(partner_1_40));
                        float bb_92;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_92) : "f"(s_94), "f"(sums_3[15]));
                        float _sub_561;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_561) : "f"(s_94), "f"(bb_92));
                        float _sub_560;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_560) : "f"(sums_3[15]), "f"(_sub_561));
                        float _sub_562;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_562) : "f"(partner_1_40), "f"(bb_92));
                        float e_92;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_92) : "f"(_sub_560), "f"(_sub_562));
                        sums_3[15] = s_94;
                        float _add_564;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_564) : "f"(errs_1[15]), "f"(partner_err_1_40));
                        float _add_563;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_563) : "f"(_add_564), "f"(e_92));
                        errs_1[15] = _add_563;
                        float partner_1_41 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 2);
                        float partner_err_1_41 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 2);
                        float s_95;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_95) : "f"(sums_3[16]), "f"(partner_1_41));
                        float bb_93;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_93) : "f"(s_95), "f"(sums_3[16]));
                        float _sub_566;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_566) : "f"(s_95), "f"(bb_93));
                        float _sub_565;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(sums_3[16]), "f"(_sub_566));
                        float _sub_567;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_567) : "f"(partner_1_41), "f"(bb_93));
                        float e_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_93) : "f"(_sub_565), "f"(_sub_567));
                        sums_3[16] = s_95;
                        float _add_569;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_569) : "f"(errs_1[16]), "f"(partner_err_1_41));
                        float _add_568;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_568) : "f"(_add_569), "f"(e_93));
                        errs_1[16] = _add_568;
                        float partner_1_42 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 2);
                        float partner_err_1_42 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 2);
                        float s_96;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_96) : "f"(sums_3[17]), "f"(partner_1_42));
                        float bb_94;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_94) : "f"(s_96), "f"(sums_3[17]));
                        float _sub_571;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_571) : "f"(s_96), "f"(bb_94));
                        float _sub_570;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_570) : "f"(sums_3[17]), "f"(_sub_571));
                        float _sub_572;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_572) : "f"(partner_1_42), "f"(bb_94));
                        float e_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_94) : "f"(_sub_570), "f"(_sub_572));
                        sums_3[17] = s_96;
                        float _add_574;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_574) : "f"(errs_1[17]), "f"(partner_err_1_42));
                        float _add_573;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_573) : "f"(_add_574), "f"(e_94));
                        errs_1[17] = _add_573;
                        float partner_1_43 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 2);
                        float partner_err_1_43 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 2);
                        float s_97;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_97) : "f"(sums_3[18]), "f"(partner_1_43));
                        float bb_95;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_95) : "f"(s_97), "f"(sums_3[18]));
                        float _sub_576;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_576) : "f"(s_97), "f"(bb_95));
                        float _sub_575;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_575) : "f"(sums_3[18]), "f"(_sub_576));
                        float _sub_577;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_577) : "f"(partner_1_43), "f"(bb_95));
                        float e_95;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_95) : "f"(_sub_575), "f"(_sub_577));
                        sums_3[18] = s_97;
                        float _add_579;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_579) : "f"(errs_1[18]), "f"(partner_err_1_43));
                        float _add_578;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_578) : "f"(_add_579), "f"(e_95));
                        errs_1[18] = _add_578;
                        float partner_1_44 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 2);
                        float partner_err_1_44 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 2);
                        float s_98;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_98) : "f"(sums_3[19]), "f"(partner_1_44));
                        float bb_96;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_96) : "f"(s_98), "f"(sums_3[19]));
                        float _sub_581;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_581) : "f"(s_98), "f"(bb_96));
                        float _sub_580;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_580) : "f"(sums_3[19]), "f"(_sub_581));
                        float _sub_582;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_582) : "f"(partner_1_44), "f"(bb_96));
                        float e_96;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_96) : "f"(_sub_580), "f"(_sub_582));
                        sums_3[19] = s_98;
                        float _add_584;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_584) : "f"(errs_1[19]), "f"(partner_err_1_44));
                        float _add_583;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_583) : "f"(_add_584), "f"(e_96));
                        errs_1[19] = _add_583;
                        float partner_1_45 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 2);
                        float partner_err_1_45 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 2);
                        float s_99;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_99) : "f"(sums_3[20]), "f"(partner_1_45));
                        float bb_97;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_97) : "f"(s_99), "f"(sums_3[20]));
                        float _sub_586;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_586) : "f"(s_99), "f"(bb_97));
                        float _sub_585;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_585) : "f"(sums_3[20]), "f"(_sub_586));
                        float _sub_587;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_587) : "f"(partner_1_45), "f"(bb_97));
                        float e_97;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_97) : "f"(_sub_585), "f"(_sub_587));
                        sums_3[20] = s_99;
                        float _add_589;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_589) : "f"(errs_1[20]), "f"(partner_err_1_45));
                        float _add_588;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_588) : "f"(_add_589), "f"(e_97));
                        errs_1[20] = _add_588;
                        float partner_1_46 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 2);
                        float partner_err_1_46 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 2);
                        float s_100;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_100) : "f"(sums_3[21]), "f"(partner_1_46));
                        float bb_98;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_98) : "f"(s_100), "f"(sums_3[21]));
                        float _sub_591;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_591) : "f"(s_100), "f"(bb_98));
                        float _sub_590;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_590) : "f"(sums_3[21]), "f"(_sub_591));
                        float _sub_592;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_592) : "f"(partner_1_46), "f"(bb_98));
                        float e_98;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_98) : "f"(_sub_590), "f"(_sub_592));
                        sums_3[21] = s_100;
                        float _add_594;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_594) : "f"(errs_1[21]), "f"(partner_err_1_46));
                        float _add_593;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_593) : "f"(_add_594), "f"(e_98));
                        errs_1[21] = _add_593;
                        float partner_1_47 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 2);
                        float partner_err_1_47 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 2);
                        float s_101;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_101) : "f"(sums_3[22]), "f"(partner_1_47));
                        float bb_99;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_99) : "f"(s_101), "f"(sums_3[22]));
                        float _sub_596;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_596) : "f"(s_101), "f"(bb_99));
                        float _sub_595;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_595) : "f"(sums_3[22]), "f"(_sub_596));
                        float _sub_597;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_597) : "f"(partner_1_47), "f"(bb_99));
                        float e_99;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_99) : "f"(_sub_595), "f"(_sub_597));
                        sums_3[22] = s_101;
                        float _add_599;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_599) : "f"(errs_1[22]), "f"(partner_err_1_47));
                        float _add_598;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_598) : "f"(_add_599), "f"(e_99));
                        errs_1[22] = _add_598;
                        float partner_1_48 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 2);
                        float partner_err_1_48 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 2);
                        float s_102;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_102) : "f"(sums_3[23]), "f"(partner_1_48));
                        float bb_100;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_100) : "f"(s_102), "f"(sums_3[23]));
                        float _sub_601;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_601) : "f"(s_102), "f"(bb_100));
                        float _sub_600;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_600) : "f"(sums_3[23]), "f"(_sub_601));
                        float _sub_602;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_602) : "f"(partner_1_48), "f"(bb_100));
                        float e_100;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_100) : "f"(_sub_600), "f"(_sub_602));
                        sums_3[23] = s_102;
                        float _add_604;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_604) : "f"(errs_1[23]), "f"(partner_err_1_48));
                        float _add_603;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_603) : "f"(_add_604), "f"(e_100));
                        errs_1[23] = _add_603;
                        float partner_1_49 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 2);
                        float partner_err_1_49 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 2);
                        float s_103;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_103) : "f"(sums_3[24]), "f"(partner_1_49));
                        float bb_101;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_101) : "f"(s_103), "f"(sums_3[24]));
                        float _sub_606;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_606) : "f"(s_103), "f"(bb_101));
                        float _sub_605;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_605) : "f"(sums_3[24]), "f"(_sub_606));
                        float _sub_607;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_607) : "f"(partner_1_49), "f"(bb_101));
                        float e_101;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_101) : "f"(_sub_605), "f"(_sub_607));
                        sums_3[24] = s_103;
                        float _add_609;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_609) : "f"(errs_1[24]), "f"(partner_err_1_49));
                        float _add_608;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_608) : "f"(_add_609), "f"(e_101));
                        errs_1[24] = _add_608;
                        float partner_1_50 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 4);
                        float partner_err_1_50 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 4);
                        float s_104;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_104) : "f"(sums_3[0]), "f"(partner_1_50));
                        float bb_102;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_102) : "f"(s_104), "f"(sums_3[0]));
                        float _sub_611;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_611) : "f"(s_104), "f"(bb_102));
                        float _sub_610;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_610) : "f"(sums_3[0]), "f"(_sub_611));
                        float _sub_612;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_612) : "f"(partner_1_50), "f"(bb_102));
                        float e_102;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_102) : "f"(_sub_610), "f"(_sub_612));
                        sums_3[0] = s_104;
                        float _add_614;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(errs_1[0]), "f"(partner_err_1_50));
                        float _add_613;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_613) : "f"(_add_614), "f"(e_102));
                        errs_1[0] = _add_613;
                        float partner_1_51 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 4);
                        float partner_err_1_51 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 4);
                        float s_105;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_105) : "f"(sums_3[1]), "f"(partner_1_51));
                        float bb_103;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_103) : "f"(s_105), "f"(sums_3[1]));
                        float _sub_616;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_616) : "f"(s_105), "f"(bb_103));
                        float _sub_615;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_615) : "f"(sums_3[1]), "f"(_sub_616));
                        float _sub_617;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_617) : "f"(partner_1_51), "f"(bb_103));
                        float e_103;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_103) : "f"(_sub_615), "f"(_sub_617));
                        sums_3[1] = s_105;
                        float _add_619;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_619) : "f"(errs_1[1]), "f"(partner_err_1_51));
                        float _add_618;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_618) : "f"(_add_619), "f"(e_103));
                        errs_1[1] = _add_618;
                        float partner_1_52 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 4);
                        float partner_err_1_52 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 4);
                        float s_106;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_106) : "f"(sums_3[2]), "f"(partner_1_52));
                        float bb_104;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_104) : "f"(s_106), "f"(sums_3[2]));
                        float _sub_621;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_621) : "f"(s_106), "f"(bb_104));
                        float _sub_620;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_620) : "f"(sums_3[2]), "f"(_sub_621));
                        float _sub_622;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_622) : "f"(partner_1_52), "f"(bb_104));
                        float e_104;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_104) : "f"(_sub_620), "f"(_sub_622));
                        sums_3[2] = s_106;
                        float _add_624;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_624) : "f"(errs_1[2]), "f"(partner_err_1_52));
                        float _add_623;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_623) : "f"(_add_624), "f"(e_104));
                        errs_1[2] = _add_623;
                        float partner_1_53 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 4);
                        float partner_err_1_53 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 4);
                        float s_107;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_107) : "f"(sums_3[3]), "f"(partner_1_53));
                        float bb_105;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_105) : "f"(s_107), "f"(sums_3[3]));
                        float _sub_626;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_626) : "f"(s_107), "f"(bb_105));
                        float _sub_625;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_625) : "f"(sums_3[3]), "f"(_sub_626));
                        float _sub_627;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_627) : "f"(partner_1_53), "f"(bb_105));
                        float e_105;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_105) : "f"(_sub_625), "f"(_sub_627));
                        sums_3[3] = s_107;
                        float _add_629;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_629) : "f"(errs_1[3]), "f"(partner_err_1_53));
                        float _add_628;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_628) : "f"(_add_629), "f"(e_105));
                        errs_1[3] = _add_628;
                        float partner_1_54 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 4);
                        float partner_err_1_54 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 4);
                        float s_108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_108) : "f"(sums_3[4]), "f"(partner_1_54));
                        float bb_106;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_106) : "f"(s_108), "f"(sums_3[4]));
                        float _sub_631;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_631) : "f"(s_108), "f"(bb_106));
                        float _sub_630;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_630) : "f"(sums_3[4]), "f"(_sub_631));
                        float _sub_632;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_632) : "f"(partner_1_54), "f"(bb_106));
                        float e_106;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_106) : "f"(_sub_630), "f"(_sub_632));
                        sums_3[4] = s_108;
                        float _add_634;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_634) : "f"(errs_1[4]), "f"(partner_err_1_54));
                        float _add_633;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_633) : "f"(_add_634), "f"(e_106));
                        errs_1[4] = _add_633;
                        float partner_1_55 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 4);
                        float partner_err_1_55 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 4);
                        float s_109;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_109) : "f"(sums_3[5]), "f"(partner_1_55));
                        float bb_107;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_107) : "f"(s_109), "f"(sums_3[5]));
                        float _sub_636;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_636) : "f"(s_109), "f"(bb_107));
                        float _sub_635;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_635) : "f"(sums_3[5]), "f"(_sub_636));
                        float _sub_637;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_637) : "f"(partner_1_55), "f"(bb_107));
                        float e_107;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_107) : "f"(_sub_635), "f"(_sub_637));
                        sums_3[5] = s_109;
                        float _add_639;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_639) : "f"(errs_1[5]), "f"(partner_err_1_55));
                        float _add_638;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_638) : "f"(_add_639), "f"(e_107));
                        errs_1[5] = _add_638;
                        float partner_1_56 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 4);
                        float partner_err_1_56 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 4);
                        float s_110;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_110) : "f"(sums_3[6]), "f"(partner_1_56));
                        float bb_108;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_108) : "f"(s_110), "f"(sums_3[6]));
                        float _sub_641;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_641) : "f"(s_110), "f"(bb_108));
                        float _sub_640;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_640) : "f"(sums_3[6]), "f"(_sub_641));
                        float _sub_642;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_642) : "f"(partner_1_56), "f"(bb_108));
                        float e_108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_108) : "f"(_sub_640), "f"(_sub_642));
                        sums_3[6] = s_110;
                        float _add_644;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(errs_1[6]), "f"(partner_err_1_56));
                        float _add_643;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_643) : "f"(_add_644), "f"(e_108));
                        errs_1[6] = _add_643;
                        float partner_1_57 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 4);
                        float partner_err_1_57 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 4);
                        float s_111;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_111) : "f"(sums_3[7]), "f"(partner_1_57));
                        float bb_109;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_109) : "f"(s_111), "f"(sums_3[7]));
                        float _sub_646;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_646) : "f"(s_111), "f"(bb_109));
                        float _sub_645;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_645) : "f"(sums_3[7]), "f"(_sub_646));
                        float _sub_647;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_647) : "f"(partner_1_57), "f"(bb_109));
                        float e_109;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_109) : "f"(_sub_645), "f"(_sub_647));
                        sums_3[7] = s_111;
                        float _add_649;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_649) : "f"(errs_1[7]), "f"(partner_err_1_57));
                        float _add_648;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_648) : "f"(_add_649), "f"(e_109));
                        errs_1[7] = _add_648;
                        float partner_1_58 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 4);
                        float partner_err_1_58 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 4);
                        float s_112_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_112_1) : "f"(sums_3[8]), "f"(partner_1_58));
                        float bb_110;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_110) : "f"(s_112_1), "f"(sums_3[8]));
                        float _sub_651;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_651) : "f"(s_112_1), "f"(bb_110));
                        float _sub_650;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_650) : "f"(sums_3[8]), "f"(_sub_651));
                        float _sub_652;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_652) : "f"(partner_1_58), "f"(bb_110));
                        float e_110;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_110) : "f"(_sub_650), "f"(_sub_652));
                        sums_3[8] = s_112_1;
                        float _add_654;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(errs_1[8]), "f"(partner_err_1_58));
                        float _add_653;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_653) : "f"(_add_654), "f"(e_110));
                        errs_1[8] = _add_653;
                        float partner_1_59 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 4);
                        float partner_err_1_59 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 4);
                        float s_113;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_113) : "f"(sums_3[9]), "f"(partner_1_59));
                        float bb_111;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_111) : "f"(s_113), "f"(sums_3[9]));
                        float _sub_656;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_656) : "f"(s_113), "f"(bb_111));
                        float _sub_655;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_655) : "f"(sums_3[9]), "f"(_sub_656));
                        float _sub_657;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_657) : "f"(partner_1_59), "f"(bb_111));
                        float e_111;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_111) : "f"(_sub_655), "f"(_sub_657));
                        sums_3[9] = s_113;
                        float _add_659;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_659) : "f"(errs_1[9]), "f"(partner_err_1_59));
                        float _add_658;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(_add_659), "f"(e_111));
                        errs_1[9] = _add_658;
                        float partner_1_60 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 4);
                        float partner_err_1_60 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 4);
                        float s_114;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_114) : "f"(sums_3[10]), "f"(partner_1_60));
                        float bb_112;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_112) : "f"(s_114), "f"(sums_3[10]));
                        float _sub_661;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_661) : "f"(s_114), "f"(bb_112));
                        float _sub_660;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_660) : "f"(sums_3[10]), "f"(_sub_661));
                        float _sub_662;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_662) : "f"(partner_1_60), "f"(bb_112));
                        float e_112;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_112) : "f"(_sub_660), "f"(_sub_662));
                        sums_3[10] = s_114;
                        float _add_664;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_664) : "f"(errs_1[10]), "f"(partner_err_1_60));
                        float _add_663;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_663) : "f"(_add_664), "f"(e_112));
                        errs_1[10] = _add_663;
                        float partner_1_61 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 4);
                        float partner_err_1_61 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 4);
                        float s_115;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_115) : "f"(sums_3[11]), "f"(partner_1_61));
                        float bb_113;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_113) : "f"(s_115), "f"(sums_3[11]));
                        float _sub_666;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_666) : "f"(s_115), "f"(bb_113));
                        float _sub_665;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_665) : "f"(sums_3[11]), "f"(_sub_666));
                        float _sub_667;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_667) : "f"(partner_1_61), "f"(bb_113));
                        float e_113;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_113) : "f"(_sub_665), "f"(_sub_667));
                        sums_3[11] = s_115;
                        float _add_669;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_669) : "f"(errs_1[11]), "f"(partner_err_1_61));
                        float _add_668;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_668) : "f"(_add_669), "f"(e_113));
                        errs_1[11] = _add_668;
                        float partner_1_62 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 4);
                        float partner_err_1_62 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 4);
                        float s_116;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_116) : "f"(sums_3[12]), "f"(partner_1_62));
                        float bb_114;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_114) : "f"(s_116), "f"(sums_3[12]));
                        float _sub_671;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_671) : "f"(s_116), "f"(bb_114));
                        float _sub_670;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_670) : "f"(sums_3[12]), "f"(_sub_671));
                        float _sub_672;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_672) : "f"(partner_1_62), "f"(bb_114));
                        float e_114;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_114) : "f"(_sub_670), "f"(_sub_672));
                        sums_3[12] = s_116;
                        float _add_674;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_674) : "f"(errs_1[12]), "f"(partner_err_1_62));
                        float _add_673;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_673) : "f"(_add_674), "f"(e_114));
                        errs_1[12] = _add_673;
                        float partner_1_63 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 4);
                        float partner_err_1_63 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 4);
                        float s_117;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_117) : "f"(sums_3[13]), "f"(partner_1_63));
                        float bb_115;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_115) : "f"(s_117), "f"(sums_3[13]));
                        float _sub_676;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_676) : "f"(s_117), "f"(bb_115));
                        float _sub_675;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_675) : "f"(sums_3[13]), "f"(_sub_676));
                        float _sub_677;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_677) : "f"(partner_1_63), "f"(bb_115));
                        float e_115;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_115) : "f"(_sub_675), "f"(_sub_677));
                        sums_3[13] = s_117;
                        float _add_679;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_679) : "f"(errs_1[13]), "f"(partner_err_1_63));
                        float _add_678;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_678) : "f"(_add_679), "f"(e_115));
                        errs_1[13] = _add_678;
                        float partner_1_64 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 4);
                        float partner_err_1_64 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 4);
                        float s_118;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_118) : "f"(sums_3[14]), "f"(partner_1_64));
                        float bb_116;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_116) : "f"(s_118), "f"(sums_3[14]));
                        float _sub_681;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_681) : "f"(s_118), "f"(bb_116));
                        float _sub_680;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_680) : "f"(sums_3[14]), "f"(_sub_681));
                        float _sub_682;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_682) : "f"(partner_1_64), "f"(bb_116));
                        float e_116;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_116) : "f"(_sub_680), "f"(_sub_682));
                        sums_3[14] = s_118;
                        float _add_684;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_684) : "f"(errs_1[14]), "f"(partner_err_1_64));
                        float _add_683;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(_add_684), "f"(e_116));
                        errs_1[14] = _add_683;
                        float partner_1_65 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 4);
                        float partner_err_1_65 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 4);
                        float s_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_119) : "f"(sums_3[15]), "f"(partner_1_65));
                        float bb_117;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_117) : "f"(s_119), "f"(sums_3[15]));
                        float _sub_686;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_686) : "f"(s_119), "f"(bb_117));
                        float _sub_685;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_685) : "f"(sums_3[15]), "f"(_sub_686));
                        float _sub_687;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_687) : "f"(partner_1_65), "f"(bb_117));
                        float e_117;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_117) : "f"(_sub_685), "f"(_sub_687));
                        sums_3[15] = s_119;
                        float _add_689;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_689) : "f"(errs_1[15]), "f"(partner_err_1_65));
                        float _add_688;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_688) : "f"(_add_689), "f"(e_117));
                        errs_1[15] = _add_688;
                        float partner_1_66 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 4);
                        float partner_err_1_66 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 4);
                        float s_120;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_120) : "f"(sums_3[16]), "f"(partner_1_66));
                        float bb_118;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_118) : "f"(s_120), "f"(sums_3[16]));
                        float _sub_691;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_691) : "f"(s_120), "f"(bb_118));
                        float _sub_690;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_690) : "f"(sums_3[16]), "f"(_sub_691));
                        float _sub_692;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_692) : "f"(partner_1_66), "f"(bb_118));
                        float e_118;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_118) : "f"(_sub_690), "f"(_sub_692));
                        sums_3[16] = s_120;
                        float _add_694;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_694) : "f"(errs_1[16]), "f"(partner_err_1_66));
                        float _add_693;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_693) : "f"(_add_694), "f"(e_118));
                        errs_1[16] = _add_693;
                        float partner_1_67 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 4);
                        float partner_err_1_67 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 4);
                        float s_121;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_121) : "f"(sums_3[17]), "f"(partner_1_67));
                        float bb_119;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_119) : "f"(s_121), "f"(sums_3[17]));
                        float _sub_696;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_696) : "f"(s_121), "f"(bb_119));
                        float _sub_695;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_695) : "f"(sums_3[17]), "f"(_sub_696));
                        float _sub_697;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_697) : "f"(partner_1_67), "f"(bb_119));
                        float e_119;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_119) : "f"(_sub_695), "f"(_sub_697));
                        sums_3[17] = s_121;
                        float _add_699;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_699) : "f"(errs_1[17]), "f"(partner_err_1_67));
                        float _add_698;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_698) : "f"(_add_699), "f"(e_119));
                        errs_1[17] = _add_698;
                        float partner_1_68 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 4);
                        float partner_err_1_68 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 4);
                        float s_122;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_122) : "f"(sums_3[18]), "f"(partner_1_68));
                        float bb_120;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_120) : "f"(s_122), "f"(sums_3[18]));
                        float _sub_701;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_701) : "f"(s_122), "f"(bb_120));
                        float _sub_700;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_700) : "f"(sums_3[18]), "f"(_sub_701));
                        float _sub_702;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_702) : "f"(partner_1_68), "f"(bb_120));
                        float e_120;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_120) : "f"(_sub_700), "f"(_sub_702));
                        sums_3[18] = s_122;
                        float _add_704;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_704) : "f"(errs_1[18]), "f"(partner_err_1_68));
                        float _add_703;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_703) : "f"(_add_704), "f"(e_120));
                        errs_1[18] = _add_703;
                        float partner_1_69 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 4);
                        float partner_err_1_69 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 4);
                        float s_123;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_123) : "f"(sums_3[19]), "f"(partner_1_69));
                        float bb_121;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_121) : "f"(s_123), "f"(sums_3[19]));
                        float _sub_706;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_706) : "f"(s_123), "f"(bb_121));
                        float _sub_705;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_705) : "f"(sums_3[19]), "f"(_sub_706));
                        float _sub_707;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_707) : "f"(partner_1_69), "f"(bb_121));
                        float e_121;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_121) : "f"(_sub_705), "f"(_sub_707));
                        sums_3[19] = s_123;
                        float _add_709;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_709) : "f"(errs_1[19]), "f"(partner_err_1_69));
                        float _add_708;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_708) : "f"(_add_709), "f"(e_121));
                        errs_1[19] = _add_708;
                        float partner_1_70 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 4);
                        float partner_err_1_70 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 4);
                        float s_124;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_124) : "f"(sums_3[20]), "f"(partner_1_70));
                        float bb_122;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_122) : "f"(s_124), "f"(sums_3[20]));
                        float _sub_711;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_711) : "f"(s_124), "f"(bb_122));
                        float _sub_710;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_710) : "f"(sums_3[20]), "f"(_sub_711));
                        float _sub_712;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_712) : "f"(partner_1_70), "f"(bb_122));
                        float e_122;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_122) : "f"(_sub_710), "f"(_sub_712));
                        sums_3[20] = s_124;
                        float _add_714;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_714) : "f"(errs_1[20]), "f"(partner_err_1_70));
                        float _add_713;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_713) : "f"(_add_714), "f"(e_122));
                        errs_1[20] = _add_713;
                        float partner_1_71 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 4);
                        float partner_err_1_71 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 4);
                        float s_125;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_125) : "f"(sums_3[21]), "f"(partner_1_71));
                        float bb_123;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_123) : "f"(s_125), "f"(sums_3[21]));
                        float _sub_716;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_716) : "f"(s_125), "f"(bb_123));
                        float _sub_715;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_715) : "f"(sums_3[21]), "f"(_sub_716));
                        float _sub_717;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_717) : "f"(partner_1_71), "f"(bb_123));
                        float e_123;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_123) : "f"(_sub_715), "f"(_sub_717));
                        sums_3[21] = s_125;
                        float _add_719;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_719) : "f"(errs_1[21]), "f"(partner_err_1_71));
                        float _add_718;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_718) : "f"(_add_719), "f"(e_123));
                        errs_1[21] = _add_718;
                        float partner_1_72 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 4);
                        float partner_err_1_72 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 4);
                        float s_126;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_126) : "f"(sums_3[22]), "f"(partner_1_72));
                        float bb_124;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_124) : "f"(s_126), "f"(sums_3[22]));
                        float _sub_721;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_721) : "f"(s_126), "f"(bb_124));
                        float _sub_720;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_720) : "f"(sums_3[22]), "f"(_sub_721));
                        float _sub_722;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_722) : "f"(partner_1_72), "f"(bb_124));
                        float e_124;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_124) : "f"(_sub_720), "f"(_sub_722));
                        sums_3[22] = s_126;
                        float _add_724;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_724) : "f"(errs_1[22]), "f"(partner_err_1_72));
                        float _add_723;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_723) : "f"(_add_724), "f"(e_124));
                        errs_1[22] = _add_723;
                        float partner_1_73 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 4);
                        float partner_err_1_73 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 4);
                        float s_127;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_127) : "f"(sums_3[23]), "f"(partner_1_73));
                        float bb_125;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_125) : "f"(s_127), "f"(sums_3[23]));
                        float _sub_726;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_726) : "f"(s_127), "f"(bb_125));
                        float _sub_725;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_725) : "f"(sums_3[23]), "f"(_sub_726));
                        float _sub_727;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_727) : "f"(partner_1_73), "f"(bb_125));
                        float e_125;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_125) : "f"(_sub_725), "f"(_sub_727));
                        sums_3[23] = s_127;
                        float _add_729;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_729) : "f"(errs_1[23]), "f"(partner_err_1_73));
                        float _add_728;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_728) : "f"(_add_729), "f"(e_125));
                        errs_1[23] = _add_728;
                        float partner_1_74 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 4);
                        float partner_err_1_74 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 4);
                        float s_128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_128) : "f"(sums_3[24]), "f"(partner_1_74));
                        float bb_126;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_126) : "f"(s_128), "f"(sums_3[24]));
                        float _sub_731;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_731) : "f"(s_128), "f"(bb_126));
                        float _sub_730;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_730) : "f"(sums_3[24]), "f"(_sub_731));
                        float _sub_732;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_732) : "f"(partner_1_74), "f"(bb_126));
                        float e_126;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_126) : "f"(_sub_730), "f"(_sub_732));
                        sums_3[24] = s_128;
                        float _add_734;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_734) : "f"(errs_1[24]), "f"(partner_err_1_74));
                        float _add_733;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_733) : "f"(_add_734), "f"(e_126));
                        errs_1[24] = _add_733;
                        float partner_1_75 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 8);
                        float partner_err_1_75 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 8);
                        float s_129;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_129) : "f"(sums_3[0]), "f"(partner_1_75));
                        float bb_127;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_127) : "f"(s_129), "f"(sums_3[0]));
                        float _sub_736;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_736) : "f"(s_129), "f"(bb_127));
                        float _sub_735;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_735) : "f"(sums_3[0]), "f"(_sub_736));
                        float _sub_737;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_737) : "f"(partner_1_75), "f"(bb_127));
                        float e_127;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_127) : "f"(_sub_735), "f"(_sub_737));
                        sums_3[0] = s_129;
                        float _add_739;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_739) : "f"(errs_1[0]), "f"(partner_err_1_75));
                        float _add_738;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_738) : "f"(_add_739), "f"(e_127));
                        errs_1[0] = _add_738;
                        float partner_1_76 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 8);
                        float partner_err_1_76 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 8);
                        float s_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_130) : "f"(sums_3[1]), "f"(partner_1_76));
                        float bb_128;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_128) : "f"(s_130), "f"(sums_3[1]));
                        float _sub_741;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_741) : "f"(s_130), "f"(bb_128));
                        float _sub_740;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_740) : "f"(sums_3[1]), "f"(_sub_741));
                        float _sub_742;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_742) : "f"(partner_1_76), "f"(bb_128));
                        float e_128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_128) : "f"(_sub_740), "f"(_sub_742));
                        sums_3[1] = s_130;
                        float _add_744;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_744) : "f"(errs_1[1]), "f"(partner_err_1_76));
                        float _add_743;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_743) : "f"(_add_744), "f"(e_128));
                        errs_1[1] = _add_743;
                        float partner_1_77 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 8);
                        float partner_err_1_77 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 8);
                        float s_131;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_131) : "f"(sums_3[2]), "f"(partner_1_77));
                        float bb_129;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_129) : "f"(s_131), "f"(sums_3[2]));
                        float _sub_746;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_746) : "f"(s_131), "f"(bb_129));
                        float _sub_745;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_745) : "f"(sums_3[2]), "f"(_sub_746));
                        float _sub_747;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_747) : "f"(partner_1_77), "f"(bb_129));
                        float e_129;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_129) : "f"(_sub_745), "f"(_sub_747));
                        sums_3[2] = s_131;
                        float _add_749;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_749) : "f"(errs_1[2]), "f"(partner_err_1_77));
                        float _add_748;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_748) : "f"(_add_749), "f"(e_129));
                        errs_1[2] = _add_748;
                        float partner_1_78 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 8);
                        float partner_err_1_78 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 8);
                        float s_132;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_132) : "f"(sums_3[3]), "f"(partner_1_78));
                        float bb_130;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_130) : "f"(s_132), "f"(sums_3[3]));
                        float _sub_751;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_751) : "f"(s_132), "f"(bb_130));
                        float _sub_750;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_750) : "f"(sums_3[3]), "f"(_sub_751));
                        float _sub_752;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_752) : "f"(partner_1_78), "f"(bb_130));
                        float e_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_130) : "f"(_sub_750), "f"(_sub_752));
                        sums_3[3] = s_132;
                        float _add_754;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_754) : "f"(errs_1[3]), "f"(partner_err_1_78));
                        float _add_753;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_753) : "f"(_add_754), "f"(e_130));
                        errs_1[3] = _add_753;
                        float partner_1_79 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 8);
                        float partner_err_1_79 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 8);
                        float s_133;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_133) : "f"(sums_3[4]), "f"(partner_1_79));
                        float bb_131;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_131) : "f"(s_133), "f"(sums_3[4]));
                        float _sub_756;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_756) : "f"(s_133), "f"(bb_131));
                        float _sub_755;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_755) : "f"(sums_3[4]), "f"(_sub_756));
                        float _sub_757;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_757) : "f"(partner_1_79), "f"(bb_131));
                        float e_131;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_131) : "f"(_sub_755), "f"(_sub_757));
                        sums_3[4] = s_133;
                        float _add_759;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_759) : "f"(errs_1[4]), "f"(partner_err_1_79));
                        float _add_758;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_758) : "f"(_add_759), "f"(e_131));
                        errs_1[4] = _add_758;
                        float partner_1_80 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 8);
                        float partner_err_1_80 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 8);
                        float s_134;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_134) : "f"(sums_3[5]), "f"(partner_1_80));
                        float bb_132;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_132) : "f"(s_134), "f"(sums_3[5]));
                        float _sub_761;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_761) : "f"(s_134), "f"(bb_132));
                        float _sub_760;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_760) : "f"(sums_3[5]), "f"(_sub_761));
                        float _sub_762;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_762) : "f"(partner_1_80), "f"(bb_132));
                        float e_132;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_132) : "f"(_sub_760), "f"(_sub_762));
                        sums_3[5] = s_134;
                        float _add_764;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_764) : "f"(errs_1[5]), "f"(partner_err_1_80));
                        float _add_763;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_763) : "f"(_add_764), "f"(e_132));
                        errs_1[5] = _add_763;
                        float partner_1_81 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 8);
                        float partner_err_1_81 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 8);
                        float s_135;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_135) : "f"(sums_3[6]), "f"(partner_1_81));
                        float bb_133;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_133) : "f"(s_135), "f"(sums_3[6]));
                        float _sub_766;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_766) : "f"(s_135), "f"(bb_133));
                        float _sub_765;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_765) : "f"(sums_3[6]), "f"(_sub_766));
                        float _sub_767;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_767) : "f"(partner_1_81), "f"(bb_133));
                        float e_133;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_133) : "f"(_sub_765), "f"(_sub_767));
                        sums_3[6] = s_135;
                        float _add_769;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_769) : "f"(errs_1[6]), "f"(partner_err_1_81));
                        float _add_768;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_768) : "f"(_add_769), "f"(e_133));
                        errs_1[6] = _add_768;
                        float partner_1_82 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 8);
                        float partner_err_1_82 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 8);
                        float s_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_136) : "f"(sums_3[7]), "f"(partner_1_82));
                        float bb_134;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_134) : "f"(s_136), "f"(sums_3[7]));
                        float _sub_771;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_771) : "f"(s_136), "f"(bb_134));
                        float _sub_770;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_770) : "f"(sums_3[7]), "f"(_sub_771));
                        float _sub_772;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_772) : "f"(partner_1_82), "f"(bb_134));
                        float e_134;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_134) : "f"(_sub_770), "f"(_sub_772));
                        sums_3[7] = s_136;
                        float _add_774;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_774) : "f"(errs_1[7]), "f"(partner_err_1_82));
                        float _add_773;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_773) : "f"(_add_774), "f"(e_134));
                        errs_1[7] = _add_773;
                        float partner_1_83 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 8);
                        float partner_err_1_83 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 8);
                        float s_137;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_137) : "f"(sums_3[8]), "f"(partner_1_83));
                        float bb_135;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_135) : "f"(s_137), "f"(sums_3[8]));
                        float _sub_776;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_776) : "f"(s_137), "f"(bb_135));
                        float _sub_775;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_775) : "f"(sums_3[8]), "f"(_sub_776));
                        float _sub_777;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_777) : "f"(partner_1_83), "f"(bb_135));
                        float e_135;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_135) : "f"(_sub_775), "f"(_sub_777));
                        sums_3[8] = s_137;
                        float _add_779;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_779) : "f"(errs_1[8]), "f"(partner_err_1_83));
                        float _add_778;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_778) : "f"(_add_779), "f"(e_135));
                        errs_1[8] = _add_778;
                        float partner_1_84 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 8);
                        float partner_err_1_84 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 8);
                        float s_138;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_138) : "f"(sums_3[9]), "f"(partner_1_84));
                        float bb_136;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_136) : "f"(s_138), "f"(sums_3[9]));
                        float _sub_781;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_781) : "f"(s_138), "f"(bb_136));
                        float _sub_780;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_780) : "f"(sums_3[9]), "f"(_sub_781));
                        float _sub_782;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_782) : "f"(partner_1_84), "f"(bb_136));
                        float e_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_136) : "f"(_sub_780), "f"(_sub_782));
                        sums_3[9] = s_138;
                        float _add_784;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_784) : "f"(errs_1[9]), "f"(partner_err_1_84));
                        float _add_783;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_783) : "f"(_add_784), "f"(e_136));
                        errs_1[9] = _add_783;
                        float partner_1_85 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 8);
                        float partner_err_1_85 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 8);
                        float s_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_139) : "f"(sums_3[10]), "f"(partner_1_85));
                        float bb_137;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_137) : "f"(s_139), "f"(sums_3[10]));
                        float _sub_786;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_786) : "f"(s_139), "f"(bb_137));
                        float _sub_785;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_785) : "f"(sums_3[10]), "f"(_sub_786));
                        float _sub_787;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_787) : "f"(partner_1_85), "f"(bb_137));
                        float e_137;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_137) : "f"(_sub_785), "f"(_sub_787));
                        sums_3[10] = s_139;
                        float _add_789;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_789) : "f"(errs_1[10]), "f"(partner_err_1_85));
                        float _add_788;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_788) : "f"(_add_789), "f"(e_137));
                        errs_1[10] = _add_788;
                        float partner_1_86 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 8);
                        float partner_err_1_86 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 8);
                        float s_140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_140) : "f"(sums_3[11]), "f"(partner_1_86));
                        float bb_138;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_138) : "f"(s_140), "f"(sums_3[11]));
                        float _sub_791;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_791) : "f"(s_140), "f"(bb_138));
                        float _sub_790;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_790) : "f"(sums_3[11]), "f"(_sub_791));
                        float _sub_792;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_792) : "f"(partner_1_86), "f"(bb_138));
                        float e_138;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_138) : "f"(_sub_790), "f"(_sub_792));
                        sums_3[11] = s_140;
                        float _add_794;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_794) : "f"(errs_1[11]), "f"(partner_err_1_86));
                        float _add_793;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_793) : "f"(_add_794), "f"(e_138));
                        errs_1[11] = _add_793;
                        float partner_1_87 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 8);
                        float partner_err_1_87 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 8);
                        float s_141;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_141) : "f"(sums_3[12]), "f"(partner_1_87));
                        float bb_139;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_139) : "f"(s_141), "f"(sums_3[12]));
                        float _sub_796;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_796) : "f"(s_141), "f"(bb_139));
                        float _sub_795;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_795) : "f"(sums_3[12]), "f"(_sub_796));
                        float _sub_797;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_797) : "f"(partner_1_87), "f"(bb_139));
                        float e_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_139) : "f"(_sub_795), "f"(_sub_797));
                        sums_3[12] = s_141;
                        float _add_799;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_799) : "f"(errs_1[12]), "f"(partner_err_1_87));
                        float _add_798;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_798) : "f"(_add_799), "f"(e_139));
                        errs_1[12] = _add_798;
                        float partner_1_88 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 8);
                        float partner_err_1_88 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 8);
                        float s_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_142) : "f"(sums_3[13]), "f"(partner_1_88));
                        float bb_140;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_140) : "f"(s_142), "f"(sums_3[13]));
                        float _sub_801;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_801) : "f"(s_142), "f"(bb_140));
                        float _sub_800;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_800) : "f"(sums_3[13]), "f"(_sub_801));
                        float _sub_802;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_802) : "f"(partner_1_88), "f"(bb_140));
                        float e_140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_140) : "f"(_sub_800), "f"(_sub_802));
                        sums_3[13] = s_142;
                        float _add_804;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_804) : "f"(errs_1[13]), "f"(partner_err_1_88));
                        float _add_803;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_803) : "f"(_add_804), "f"(e_140));
                        errs_1[13] = _add_803;
                        float partner_1_89 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 8);
                        float partner_err_1_89 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 8);
                        float s_143;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_143) : "f"(sums_3[14]), "f"(partner_1_89));
                        float bb_141;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_141) : "f"(s_143), "f"(sums_3[14]));
                        float _sub_806;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_806) : "f"(s_143), "f"(bb_141));
                        float _sub_805;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_805) : "f"(sums_3[14]), "f"(_sub_806));
                        float _sub_807;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_807) : "f"(partner_1_89), "f"(bb_141));
                        float e_141;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_141) : "f"(_sub_805), "f"(_sub_807));
                        sums_3[14] = s_143;
                        float _add_809;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_809) : "f"(errs_1[14]), "f"(partner_err_1_89));
                        float _add_808;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_808) : "f"(_add_809), "f"(e_141));
                        errs_1[14] = _add_808;
                        float partner_1_90 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 8);
                        float partner_err_1_90 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 8);
                        float s_144;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_144) : "f"(sums_3[15]), "f"(partner_1_90));
                        float bb_142;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_142) : "f"(s_144), "f"(sums_3[15]));
                        float _sub_811;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_811) : "f"(s_144), "f"(bb_142));
                        float _sub_810;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_810) : "f"(sums_3[15]), "f"(_sub_811));
                        float _sub_812;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_812) : "f"(partner_1_90), "f"(bb_142));
                        float e_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_142) : "f"(_sub_810), "f"(_sub_812));
                        sums_3[15] = s_144;
                        float _add_814;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_814) : "f"(errs_1[15]), "f"(partner_err_1_90));
                        float _add_813;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_813) : "f"(_add_814), "f"(e_142));
                        errs_1[15] = _add_813;
                        float partner_1_91 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 8);
                        float partner_err_1_91 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 8);
                        float s_145;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_145) : "f"(sums_3[16]), "f"(partner_1_91));
                        float bb_143;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_143) : "f"(s_145), "f"(sums_3[16]));
                        float _sub_816;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_816) : "f"(s_145), "f"(bb_143));
                        float _sub_815;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_815) : "f"(sums_3[16]), "f"(_sub_816));
                        float _sub_817;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_817) : "f"(partner_1_91), "f"(bb_143));
                        float e_143;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_143) : "f"(_sub_815), "f"(_sub_817));
                        sums_3[16] = s_145;
                        float _add_819;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_819) : "f"(errs_1[16]), "f"(partner_err_1_91));
                        float _add_818;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_818) : "f"(_add_819), "f"(e_143));
                        errs_1[16] = _add_818;
                        float partner_1_92 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 8);
                        float partner_err_1_92 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 8);
                        float s_146;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_146) : "f"(sums_3[17]), "f"(partner_1_92));
                        float bb_144;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_144) : "f"(s_146), "f"(sums_3[17]));
                        float _sub_821;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_821) : "f"(s_146), "f"(bb_144));
                        float _sub_820;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_820) : "f"(sums_3[17]), "f"(_sub_821));
                        float _sub_822;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_822) : "f"(partner_1_92), "f"(bb_144));
                        float e_144;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_144) : "f"(_sub_820), "f"(_sub_822));
                        sums_3[17] = s_146;
                        float _add_824;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_824) : "f"(errs_1[17]), "f"(partner_err_1_92));
                        float _add_823;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_823) : "f"(_add_824), "f"(e_144));
                        errs_1[17] = _add_823;
                        float partner_1_93 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 8);
                        float partner_err_1_93 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 8);
                        float s_147;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_147) : "f"(sums_3[18]), "f"(partner_1_93));
                        float bb_145;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_145) : "f"(s_147), "f"(sums_3[18]));
                        float _sub_826;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_826) : "f"(s_147), "f"(bb_145));
                        float _sub_825;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_825) : "f"(sums_3[18]), "f"(_sub_826));
                        float _sub_827;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_827) : "f"(partner_1_93), "f"(bb_145));
                        float e_145;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_145) : "f"(_sub_825), "f"(_sub_827));
                        sums_3[18] = s_147;
                        float _add_829;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_829) : "f"(errs_1[18]), "f"(partner_err_1_93));
                        float _add_828;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_828) : "f"(_add_829), "f"(e_145));
                        errs_1[18] = _add_828;
                        float partner_1_94 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 8);
                        float partner_err_1_94 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 8);
                        float s_148;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_148) : "f"(sums_3[19]), "f"(partner_1_94));
                        float bb_146;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_146) : "f"(s_148), "f"(sums_3[19]));
                        float _sub_831;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_831) : "f"(s_148), "f"(bb_146));
                        float _sub_830;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_830) : "f"(sums_3[19]), "f"(_sub_831));
                        float _sub_832;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_832) : "f"(partner_1_94), "f"(bb_146));
                        float e_146;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_146) : "f"(_sub_830), "f"(_sub_832));
                        sums_3[19] = s_148;
                        float _add_834;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_834) : "f"(errs_1[19]), "f"(partner_err_1_94));
                        float _add_833;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_833) : "f"(_add_834), "f"(e_146));
                        errs_1[19] = _add_833;
                        float partner_1_95 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 8);
                        float partner_err_1_95 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 8);
                        float s_149;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_149) : "f"(sums_3[20]), "f"(partner_1_95));
                        float bb_147;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_147) : "f"(s_149), "f"(sums_3[20]));
                        float _sub_836;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_836) : "f"(s_149), "f"(bb_147));
                        float _sub_835;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_835) : "f"(sums_3[20]), "f"(_sub_836));
                        float _sub_837;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_837) : "f"(partner_1_95), "f"(bb_147));
                        float e_147;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_147) : "f"(_sub_835), "f"(_sub_837));
                        sums_3[20] = s_149;
                        float _add_839;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_839) : "f"(errs_1[20]), "f"(partner_err_1_95));
                        float _add_838;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_838) : "f"(_add_839), "f"(e_147));
                        errs_1[20] = _add_838;
                        float partner_1_96 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 8);
                        float partner_err_1_96 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 8);
                        float s_150;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_150) : "f"(sums_3[21]), "f"(partner_1_96));
                        float bb_148;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_148) : "f"(s_150), "f"(sums_3[21]));
                        float _sub_841;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_841) : "f"(s_150), "f"(bb_148));
                        float _sub_840;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_840) : "f"(sums_3[21]), "f"(_sub_841));
                        float _sub_842;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_842) : "f"(partner_1_96), "f"(bb_148));
                        float e_148;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_148) : "f"(_sub_840), "f"(_sub_842));
                        sums_3[21] = s_150;
                        float _add_844;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_844) : "f"(errs_1[21]), "f"(partner_err_1_96));
                        float _add_843;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_843) : "f"(_add_844), "f"(e_148));
                        errs_1[21] = _add_843;
                        float partner_1_97 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 8);
                        float partner_err_1_97 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 8);
                        float s_151;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_151) : "f"(sums_3[22]), "f"(partner_1_97));
                        float bb_149;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_149) : "f"(s_151), "f"(sums_3[22]));
                        float _sub_846;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_846) : "f"(s_151), "f"(bb_149));
                        float _sub_845;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_845) : "f"(sums_3[22]), "f"(_sub_846));
                        float _sub_847;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_847) : "f"(partner_1_97), "f"(bb_149));
                        float e_149;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_149) : "f"(_sub_845), "f"(_sub_847));
                        sums_3[22] = s_151;
                        float _add_849;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_849) : "f"(errs_1[22]), "f"(partner_err_1_97));
                        float _add_848;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_848) : "f"(_add_849), "f"(e_149));
                        errs_1[22] = _add_848;
                        float partner_1_98 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 8);
                        float partner_err_1_98 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 8);
                        float s_152;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_152) : "f"(sums_3[23]), "f"(partner_1_98));
                        float bb_150;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_150) : "f"(s_152), "f"(sums_3[23]));
                        float _sub_851;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_851) : "f"(s_152), "f"(bb_150));
                        float _sub_850;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_850) : "f"(sums_3[23]), "f"(_sub_851));
                        float _sub_852;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_852) : "f"(partner_1_98), "f"(bb_150));
                        float e_150;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_150) : "f"(_sub_850), "f"(_sub_852));
                        sums_3[23] = s_152;
                        float _add_854;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_854) : "f"(errs_1[23]), "f"(partner_err_1_98));
                        float _add_853;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_853) : "f"(_add_854), "f"(e_150));
                        errs_1[23] = _add_853;
                        float partner_1_99 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 8);
                        float partner_err_1_99 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 8);
                        float s_153;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_153) : "f"(sums_3[24]), "f"(partner_1_99));
                        float bb_151;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_151) : "f"(s_153), "f"(sums_3[24]));
                        float _sub_856;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_856) : "f"(s_153), "f"(bb_151));
                        float _sub_855;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_855) : "f"(sums_3[24]), "f"(_sub_856));
                        float _sub_857;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_857) : "f"(partner_1_99), "f"(bb_151));
                        float e_151;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_151) : "f"(_sub_855), "f"(_sub_857));
                        sums_3[24] = s_153;
                        float _add_859;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_859) : "f"(errs_1[24]), "f"(partner_err_1_99));
                        float _add_858;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_858) : "f"(_add_859), "f"(e_151));
                        errs_1[24] = _add_858;
                        float partner_1_100 = __shfl_xor_sync(0xFFFFFFFF, sums_3[0], 16);
                        float partner_err_1_100 = __shfl_xor_sync(0xFFFFFFFF, errs_1[0], 16);
                        float s_154;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_154) : "f"(sums_3[0]), "f"(partner_1_100));
                        float bb_152;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_152) : "f"(s_154), "f"(sums_3[0]));
                        float _sub_861;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_861) : "f"(s_154), "f"(bb_152));
                        float _sub_860;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_860) : "f"(sums_3[0]), "f"(_sub_861));
                        float _sub_862;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_862) : "f"(partner_1_100), "f"(bb_152));
                        float e_152;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_152) : "f"(_sub_860), "f"(_sub_862));
                        sums_3[0] = s_154;
                        float _add_864;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_864) : "f"(errs_1[0]), "f"(partner_err_1_100));
                        float _add_863;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_863) : "f"(_add_864), "f"(e_152));
                        errs_1[0] = _add_863;
                        float partner_1_101 = __shfl_xor_sync(0xFFFFFFFF, sums_3[1], 16);
                        float partner_err_1_101 = __shfl_xor_sync(0xFFFFFFFF, errs_1[1], 16);
                        float s_155;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_155) : "f"(sums_3[1]), "f"(partner_1_101));
                        float bb_153;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_153) : "f"(s_155), "f"(sums_3[1]));
                        float _sub_866;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_866) : "f"(s_155), "f"(bb_153));
                        float _sub_865;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_865) : "f"(sums_3[1]), "f"(_sub_866));
                        float _sub_867;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_867) : "f"(partner_1_101), "f"(bb_153));
                        float e_153;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_153) : "f"(_sub_865), "f"(_sub_867));
                        sums_3[1] = s_155;
                        float _add_869;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_869) : "f"(errs_1[1]), "f"(partner_err_1_101));
                        float _add_868;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_868) : "f"(_add_869), "f"(e_153));
                        errs_1[1] = _add_868;
                        float partner_1_102 = __shfl_xor_sync(0xFFFFFFFF, sums_3[2], 16);
                        float partner_err_1_102 = __shfl_xor_sync(0xFFFFFFFF, errs_1[2], 16);
                        float s_156;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_156) : "f"(sums_3[2]), "f"(partner_1_102));
                        float bb_154;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_154) : "f"(s_156), "f"(sums_3[2]));
                        float _sub_871;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_871) : "f"(s_156), "f"(bb_154));
                        float _sub_870;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_870) : "f"(sums_3[2]), "f"(_sub_871));
                        float _sub_872;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_872) : "f"(partner_1_102), "f"(bb_154));
                        float e_154;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_154) : "f"(_sub_870), "f"(_sub_872));
                        sums_3[2] = s_156;
                        float _add_874;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_874) : "f"(errs_1[2]), "f"(partner_err_1_102));
                        float _add_873;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_873) : "f"(_add_874), "f"(e_154));
                        errs_1[2] = _add_873;
                        float partner_1_103 = __shfl_xor_sync(0xFFFFFFFF, sums_3[3], 16);
                        float partner_err_1_103 = __shfl_xor_sync(0xFFFFFFFF, errs_1[3], 16);
                        float s_157;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_157) : "f"(sums_3[3]), "f"(partner_1_103));
                        float bb_155;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_155) : "f"(s_157), "f"(sums_3[3]));
                        float _sub_876;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_876) : "f"(s_157), "f"(bb_155));
                        float _sub_875;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_875) : "f"(sums_3[3]), "f"(_sub_876));
                        float _sub_877;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_877) : "f"(partner_1_103), "f"(bb_155));
                        float e_155;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_155) : "f"(_sub_875), "f"(_sub_877));
                        sums_3[3] = s_157;
                        float _add_879;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_879) : "f"(errs_1[3]), "f"(partner_err_1_103));
                        float _add_878;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_878) : "f"(_add_879), "f"(e_155));
                        errs_1[3] = _add_878;
                        float partner_1_104 = __shfl_xor_sync(0xFFFFFFFF, sums_3[4], 16);
                        float partner_err_1_104 = __shfl_xor_sync(0xFFFFFFFF, errs_1[4], 16);
                        float s_158;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_158) : "f"(sums_3[4]), "f"(partner_1_104));
                        float bb_156;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_156) : "f"(s_158), "f"(sums_3[4]));
                        float _sub_881;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_881) : "f"(s_158), "f"(bb_156));
                        float _sub_880;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_880) : "f"(sums_3[4]), "f"(_sub_881));
                        float _sub_882;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_882) : "f"(partner_1_104), "f"(bb_156));
                        float e_156;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_156) : "f"(_sub_880), "f"(_sub_882));
                        sums_3[4] = s_158;
                        float _add_884;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_884) : "f"(errs_1[4]), "f"(partner_err_1_104));
                        float _add_883;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_883) : "f"(_add_884), "f"(e_156));
                        errs_1[4] = _add_883;
                        float partner_1_105 = __shfl_xor_sync(0xFFFFFFFF, sums_3[5], 16);
                        float partner_err_1_105 = __shfl_xor_sync(0xFFFFFFFF, errs_1[5], 16);
                        float s_159;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_159) : "f"(sums_3[5]), "f"(partner_1_105));
                        float bb_157;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_157) : "f"(s_159), "f"(sums_3[5]));
                        float _sub_886;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_886) : "f"(s_159), "f"(bb_157));
                        float _sub_885;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_885) : "f"(sums_3[5]), "f"(_sub_886));
                        float _sub_887;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_887) : "f"(partner_1_105), "f"(bb_157));
                        float e_157;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_157) : "f"(_sub_885), "f"(_sub_887));
                        sums_3[5] = s_159;
                        float _add_889;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_889) : "f"(errs_1[5]), "f"(partner_err_1_105));
                        float _add_888;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_888) : "f"(_add_889), "f"(e_157));
                        errs_1[5] = _add_888;
                        float partner_1_106 = __shfl_xor_sync(0xFFFFFFFF, sums_3[6], 16);
                        float partner_err_1_106 = __shfl_xor_sync(0xFFFFFFFF, errs_1[6], 16);
                        float s_160;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_160) : "f"(sums_3[6]), "f"(partner_1_106));
                        float bb_158;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_158) : "f"(s_160), "f"(sums_3[6]));
                        float _sub_891;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_891) : "f"(s_160), "f"(bb_158));
                        float _sub_890;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_890) : "f"(sums_3[6]), "f"(_sub_891));
                        float _sub_892;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_892) : "f"(partner_1_106), "f"(bb_158));
                        float e_158;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_158) : "f"(_sub_890), "f"(_sub_892));
                        sums_3[6] = s_160;
                        float _add_894;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_894) : "f"(errs_1[6]), "f"(partner_err_1_106));
                        float _add_893;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_893) : "f"(_add_894), "f"(e_158));
                        errs_1[6] = _add_893;
                        float partner_1_107 = __shfl_xor_sync(0xFFFFFFFF, sums_3[7], 16);
                        float partner_err_1_107 = __shfl_xor_sync(0xFFFFFFFF, errs_1[7], 16);
                        float s_161;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_161) : "f"(sums_3[7]), "f"(partner_1_107));
                        float bb_159;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_159) : "f"(s_161), "f"(sums_3[7]));
                        float _sub_896;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_896) : "f"(s_161), "f"(bb_159));
                        float _sub_895;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_895) : "f"(sums_3[7]), "f"(_sub_896));
                        float _sub_897;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_897) : "f"(partner_1_107), "f"(bb_159));
                        float e_159;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_159) : "f"(_sub_895), "f"(_sub_897));
                        sums_3[7] = s_161;
                        float _add_899;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_899) : "f"(errs_1[7]), "f"(partner_err_1_107));
                        float _add_898;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_898) : "f"(_add_899), "f"(e_159));
                        errs_1[7] = _add_898;
                        float partner_1_108 = __shfl_xor_sync(0xFFFFFFFF, sums_3[8], 16);
                        float partner_err_1_108 = __shfl_xor_sync(0xFFFFFFFF, errs_1[8], 16);
                        float s_162;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_162) : "f"(sums_3[8]), "f"(partner_1_108));
                        float bb_160;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_160) : "f"(s_162), "f"(sums_3[8]));
                        float _sub_901;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_901) : "f"(s_162), "f"(bb_160));
                        float _sub_900;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_900) : "f"(sums_3[8]), "f"(_sub_901));
                        float _sub_902;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_902) : "f"(partner_1_108), "f"(bb_160));
                        float e_160;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_160) : "f"(_sub_900), "f"(_sub_902));
                        sums_3[8] = s_162;
                        float _add_904;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_904) : "f"(errs_1[8]), "f"(partner_err_1_108));
                        float _add_903;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_903) : "f"(_add_904), "f"(e_160));
                        errs_1[8] = _add_903;
                        float partner_1_109 = __shfl_xor_sync(0xFFFFFFFF, sums_3[9], 16);
                        float partner_err_1_109 = __shfl_xor_sync(0xFFFFFFFF, errs_1[9], 16);
                        float s_163;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_163) : "f"(sums_3[9]), "f"(partner_1_109));
                        float bb_161;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_161) : "f"(s_163), "f"(sums_3[9]));
                        float _sub_906;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_906) : "f"(s_163), "f"(bb_161));
                        float _sub_905;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_905) : "f"(sums_3[9]), "f"(_sub_906));
                        float _sub_907;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_907) : "f"(partner_1_109), "f"(bb_161));
                        float e_161;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_161) : "f"(_sub_905), "f"(_sub_907));
                        sums_3[9] = s_163;
                        float _add_909;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_909) : "f"(errs_1[9]), "f"(partner_err_1_109));
                        float _add_908;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_908) : "f"(_add_909), "f"(e_161));
                        errs_1[9] = _add_908;
                        float partner_1_110 = __shfl_xor_sync(0xFFFFFFFF, sums_3[10], 16);
                        float partner_err_1_110 = __shfl_xor_sync(0xFFFFFFFF, errs_1[10], 16);
                        float s_164;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_164) : "f"(sums_3[10]), "f"(partner_1_110));
                        float bb_162;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_162) : "f"(s_164), "f"(sums_3[10]));
                        float _sub_911;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_911) : "f"(s_164), "f"(bb_162));
                        float _sub_910;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_910) : "f"(sums_3[10]), "f"(_sub_911));
                        float _sub_912;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_912) : "f"(partner_1_110), "f"(bb_162));
                        float e_162;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_162) : "f"(_sub_910), "f"(_sub_912));
                        sums_3[10] = s_164;
                        float _add_914;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_914) : "f"(errs_1[10]), "f"(partner_err_1_110));
                        float _add_913;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_913) : "f"(_add_914), "f"(e_162));
                        errs_1[10] = _add_913;
                        float partner_1_111 = __shfl_xor_sync(0xFFFFFFFF, sums_3[11], 16);
                        float partner_err_1_111 = __shfl_xor_sync(0xFFFFFFFF, errs_1[11], 16);
                        float s_165;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_165) : "f"(sums_3[11]), "f"(partner_1_111));
                        float bb_163;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_163) : "f"(s_165), "f"(sums_3[11]));
                        float _sub_916;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_916) : "f"(s_165), "f"(bb_163));
                        float _sub_915;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_915) : "f"(sums_3[11]), "f"(_sub_916));
                        float _sub_917;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_917) : "f"(partner_1_111), "f"(bb_163));
                        float e_163;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_163) : "f"(_sub_915), "f"(_sub_917));
                        sums_3[11] = s_165;
                        float _add_919;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_919) : "f"(errs_1[11]), "f"(partner_err_1_111));
                        float _add_918;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_918) : "f"(_add_919), "f"(e_163));
                        errs_1[11] = _add_918;
                        float partner_1_112 = __shfl_xor_sync(0xFFFFFFFF, sums_3[12], 16);
                        float partner_err_1_112 = __shfl_xor_sync(0xFFFFFFFF, errs_1[12], 16);
                        float s_166;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_166) : "f"(sums_3[12]), "f"(partner_1_112));
                        float bb_164;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_164) : "f"(s_166), "f"(sums_3[12]));
                        float _sub_921;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_921) : "f"(s_166), "f"(bb_164));
                        float _sub_920;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_920) : "f"(sums_3[12]), "f"(_sub_921));
                        float _sub_922;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_922) : "f"(partner_1_112), "f"(bb_164));
                        float e_164;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_164) : "f"(_sub_920), "f"(_sub_922));
                        sums_3[12] = s_166;
                        float _add_924;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_924) : "f"(errs_1[12]), "f"(partner_err_1_112));
                        float _add_923;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_923) : "f"(_add_924), "f"(e_164));
                        errs_1[12] = _add_923;
                        float partner_1_113 = __shfl_xor_sync(0xFFFFFFFF, sums_3[13], 16);
                        float partner_err_1_113 = __shfl_xor_sync(0xFFFFFFFF, errs_1[13], 16);
                        float s_167;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_167) : "f"(sums_3[13]), "f"(partner_1_113));
                        float bb_165;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_165) : "f"(s_167), "f"(sums_3[13]));
                        float _sub_926;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_926) : "f"(s_167), "f"(bb_165));
                        float _sub_925;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_925) : "f"(sums_3[13]), "f"(_sub_926));
                        float _sub_927;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_927) : "f"(partner_1_113), "f"(bb_165));
                        float e_165;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_165) : "f"(_sub_925), "f"(_sub_927));
                        sums_3[13] = s_167;
                        float _add_929;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_929) : "f"(errs_1[13]), "f"(partner_err_1_113));
                        float _add_928;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_928) : "f"(_add_929), "f"(e_165));
                        errs_1[13] = _add_928;
                        float partner_1_114 = __shfl_xor_sync(0xFFFFFFFF, sums_3[14], 16);
                        float partner_err_1_114 = __shfl_xor_sync(0xFFFFFFFF, errs_1[14], 16);
                        float s_168;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_168) : "f"(sums_3[14]), "f"(partner_1_114));
                        float bb_166;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_166) : "f"(s_168), "f"(sums_3[14]));
                        float _sub_931;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_931) : "f"(s_168), "f"(bb_166));
                        float _sub_930;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_930) : "f"(sums_3[14]), "f"(_sub_931));
                        float _sub_932;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_932) : "f"(partner_1_114), "f"(bb_166));
                        float e_166;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_166) : "f"(_sub_930), "f"(_sub_932));
                        sums_3[14] = s_168;
                        float _add_934;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_934) : "f"(errs_1[14]), "f"(partner_err_1_114));
                        float _add_933;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_933) : "f"(_add_934), "f"(e_166));
                        errs_1[14] = _add_933;
                        float partner_1_115 = __shfl_xor_sync(0xFFFFFFFF, sums_3[15], 16);
                        float partner_err_1_115 = __shfl_xor_sync(0xFFFFFFFF, errs_1[15], 16);
                        float s_169;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_169) : "f"(sums_3[15]), "f"(partner_1_115));
                        float bb_167;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_167) : "f"(s_169), "f"(sums_3[15]));
                        float _sub_936;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_936) : "f"(s_169), "f"(bb_167));
                        float _sub_935;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_935) : "f"(sums_3[15]), "f"(_sub_936));
                        float _sub_937;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_937) : "f"(partner_1_115), "f"(bb_167));
                        float e_167;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_167) : "f"(_sub_935), "f"(_sub_937));
                        sums_3[15] = s_169;
                        float _add_939;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_939) : "f"(errs_1[15]), "f"(partner_err_1_115));
                        float _add_938;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_938) : "f"(_add_939), "f"(e_167));
                        errs_1[15] = _add_938;
                        float partner_1_116 = __shfl_xor_sync(0xFFFFFFFF, sums_3[16], 16);
                        float partner_err_1_116 = __shfl_xor_sync(0xFFFFFFFF, errs_1[16], 16);
                        float s_170;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_170) : "f"(sums_3[16]), "f"(partner_1_116));
                        float bb_168;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_168) : "f"(s_170), "f"(sums_3[16]));
                        float _sub_941;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_941) : "f"(s_170), "f"(bb_168));
                        float _sub_940;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_940) : "f"(sums_3[16]), "f"(_sub_941));
                        float _sub_942;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_942) : "f"(partner_1_116), "f"(bb_168));
                        float e_168;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_168) : "f"(_sub_940), "f"(_sub_942));
                        sums_3[16] = s_170;
                        float _add_944;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_944) : "f"(errs_1[16]), "f"(partner_err_1_116));
                        float _add_943;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_943) : "f"(_add_944), "f"(e_168));
                        errs_1[16] = _add_943;
                        float partner_1_117 = __shfl_xor_sync(0xFFFFFFFF, sums_3[17], 16);
                        float partner_err_1_117 = __shfl_xor_sync(0xFFFFFFFF, errs_1[17], 16);
                        float s_171;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_171) : "f"(sums_3[17]), "f"(partner_1_117));
                        float bb_169;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_169) : "f"(s_171), "f"(sums_3[17]));
                        float _sub_946;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_946) : "f"(s_171), "f"(bb_169));
                        float _sub_945;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_945) : "f"(sums_3[17]), "f"(_sub_946));
                        float _sub_947;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_947) : "f"(partner_1_117), "f"(bb_169));
                        float e_169;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_169) : "f"(_sub_945), "f"(_sub_947));
                        sums_3[17] = s_171;
                        float _add_949;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_949) : "f"(errs_1[17]), "f"(partner_err_1_117));
                        float _add_948;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_948) : "f"(_add_949), "f"(e_169));
                        errs_1[17] = _add_948;
                        float partner_1_118 = __shfl_xor_sync(0xFFFFFFFF, sums_3[18], 16);
                        float partner_err_1_118 = __shfl_xor_sync(0xFFFFFFFF, errs_1[18], 16);
                        float s_172;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_172) : "f"(sums_3[18]), "f"(partner_1_118));
                        float bb_170;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_170) : "f"(s_172), "f"(sums_3[18]));
                        float _sub_951;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_951) : "f"(s_172), "f"(bb_170));
                        float _sub_950;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_950) : "f"(sums_3[18]), "f"(_sub_951));
                        float _sub_952;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_952) : "f"(partner_1_118), "f"(bb_170));
                        float e_170;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_170) : "f"(_sub_950), "f"(_sub_952));
                        sums_3[18] = s_172;
                        float _add_954;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_954) : "f"(errs_1[18]), "f"(partner_err_1_118));
                        float _add_953;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_953) : "f"(_add_954), "f"(e_170));
                        errs_1[18] = _add_953;
                        float partner_1_119 = __shfl_xor_sync(0xFFFFFFFF, sums_3[19], 16);
                        float partner_err_1_119 = __shfl_xor_sync(0xFFFFFFFF, errs_1[19], 16);
                        float s_173;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_173) : "f"(sums_3[19]), "f"(partner_1_119));
                        float bb_171;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_171) : "f"(s_173), "f"(sums_3[19]));
                        float _sub_956;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_956) : "f"(s_173), "f"(bb_171));
                        float _sub_955;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_955) : "f"(sums_3[19]), "f"(_sub_956));
                        float _sub_957;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_957) : "f"(partner_1_119), "f"(bb_171));
                        float e_171;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_171) : "f"(_sub_955), "f"(_sub_957));
                        sums_3[19] = s_173;
                        float _add_959;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_959) : "f"(errs_1[19]), "f"(partner_err_1_119));
                        float _add_958;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_958) : "f"(_add_959), "f"(e_171));
                        errs_1[19] = _add_958;
                        float partner_1_120 = __shfl_xor_sync(0xFFFFFFFF, sums_3[20], 16);
                        float partner_err_1_120 = __shfl_xor_sync(0xFFFFFFFF, errs_1[20], 16);
                        float s_174;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_174) : "f"(sums_3[20]), "f"(partner_1_120));
                        float bb_172;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_172) : "f"(s_174), "f"(sums_3[20]));
                        float _sub_961;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_961) : "f"(s_174), "f"(bb_172));
                        float _sub_960;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_960) : "f"(sums_3[20]), "f"(_sub_961));
                        float _sub_962;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_962) : "f"(partner_1_120), "f"(bb_172));
                        float e_172;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_172) : "f"(_sub_960), "f"(_sub_962));
                        sums_3[20] = s_174;
                        float _add_964;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_964) : "f"(errs_1[20]), "f"(partner_err_1_120));
                        float _add_963;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_963) : "f"(_add_964), "f"(e_172));
                        errs_1[20] = _add_963;
                        float partner_1_121 = __shfl_xor_sync(0xFFFFFFFF, sums_3[21], 16);
                        float partner_err_1_121 = __shfl_xor_sync(0xFFFFFFFF, errs_1[21], 16);
                        float s_175;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_175) : "f"(sums_3[21]), "f"(partner_1_121));
                        float bb_173;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_173) : "f"(s_175), "f"(sums_3[21]));
                        float _sub_966;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_966) : "f"(s_175), "f"(bb_173));
                        float _sub_965;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_965) : "f"(sums_3[21]), "f"(_sub_966));
                        float _sub_967;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_967) : "f"(partner_1_121), "f"(bb_173));
                        float e_173;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_173) : "f"(_sub_965), "f"(_sub_967));
                        sums_3[21] = s_175;
                        float _add_969;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_969) : "f"(errs_1[21]), "f"(partner_err_1_121));
                        float _add_968;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_968) : "f"(_add_969), "f"(e_173));
                        errs_1[21] = _add_968;
                        float partner_1_122 = __shfl_xor_sync(0xFFFFFFFF, sums_3[22], 16);
                        float partner_err_1_122 = __shfl_xor_sync(0xFFFFFFFF, errs_1[22], 16);
                        float s_176;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_176) : "f"(sums_3[22]), "f"(partner_1_122));
                        float bb_174;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_174) : "f"(s_176), "f"(sums_3[22]));
                        float _sub_971;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_971) : "f"(s_176), "f"(bb_174));
                        float _sub_970;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_970) : "f"(sums_3[22]), "f"(_sub_971));
                        float _sub_972;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_972) : "f"(partner_1_122), "f"(bb_174));
                        float e_174;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_174) : "f"(_sub_970), "f"(_sub_972));
                        sums_3[22] = s_176;
                        float _add_974;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_974) : "f"(errs_1[22]), "f"(partner_err_1_122));
                        float _add_973;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_973) : "f"(_add_974), "f"(e_174));
                        errs_1[22] = _add_973;
                        float partner_1_123 = __shfl_xor_sync(0xFFFFFFFF, sums_3[23], 16);
                        float partner_err_1_123 = __shfl_xor_sync(0xFFFFFFFF, errs_1[23], 16);
                        float s_177;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_177) : "f"(sums_3[23]), "f"(partner_1_123));
                        float bb_175;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_175) : "f"(s_177), "f"(sums_3[23]));
                        float _sub_976;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_976) : "f"(s_177), "f"(bb_175));
                        float _sub_975;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_975) : "f"(sums_3[23]), "f"(_sub_976));
                        float _sub_977;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_977) : "f"(partner_1_123), "f"(bb_175));
                        float e_175;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_175) : "f"(_sub_975), "f"(_sub_977));
                        sums_3[23] = s_177;
                        float _add_979;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_979) : "f"(errs_1[23]), "f"(partner_err_1_123));
                        float _add_978;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_978) : "f"(_add_979), "f"(e_175));
                        errs_1[23] = _add_978;
                        float partner_1_124 = __shfl_xor_sync(0xFFFFFFFF, sums_3[24], 16);
                        float partner_err_1_124 = __shfl_xor_sync(0xFFFFFFFF, errs_1[24], 16);
                        float s_178;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_178) : "f"(sums_3[24]), "f"(partner_1_124));
                        float bb_176;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_176) : "f"(s_178), "f"(sums_3[24]));
                        float _sub_981;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_981) : "f"(s_178), "f"(bb_176));
                        float _sub_980;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_980) : "f"(sums_3[24]), "f"(_sub_981));
                        float _sub_982;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_982) : "f"(partner_1_124), "f"(bb_176));
                        float e_176;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_176) : "f"(_sub_980), "f"(_sub_982));
                        sums_3[24] = s_178;
                        float _add_984;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_984) : "f"(errs_1[24]), "f"(partner_err_1_124));
                        float _add_983;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_983) : "f"(_add_984), "f"(e_176));
                        errs_1[24] = _add_983;
                        float s_748 = 0.0f;
                        float total_n_1_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_0) : "f"(sums_3[0]), "f"(errs_1[0]));
                        s_748 = ((lane_in_warp == 0) ? total_n_1_0 : s_748);
                        float total_n_1_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_1) : "f"(sums_3[1]), "f"(errs_1[1]));
                        s_748 = ((lane_in_warp == 1) ? total_n_1_1 : s_748);
                        float total_n_1_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_2) : "f"(sums_3[2]), "f"(errs_1[2]));
                        s_748 = ((lane_in_warp == 2) ? total_n_1_2 : s_748);
                        float total_n_1_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_3) : "f"(sums_3[3]), "f"(errs_1[3]));
                        s_748 = ((lane_in_warp == 3) ? total_n_1_3 : s_748);
                        float total_n_1_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_4) : "f"(sums_3[4]), "f"(errs_1[4]));
                        s_748 = ((lane_in_warp == 4) ? total_n_1_4 : s_748);
                        float total_n_1_5;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_5) : "f"(sums_3[5]), "f"(errs_1[5]));
                        s_748 = ((lane_in_warp == 5) ? total_n_1_5 : s_748);
                        float total_n_1_6;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_6) : "f"(sums_3[6]), "f"(errs_1[6]));
                        s_748 = ((lane_in_warp == 6) ? total_n_1_6 : s_748);
                        float total_n_1_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_7) : "f"(sums_3[7]), "f"(errs_1[7]));
                        s_748 = ((lane_in_warp == 7) ? total_n_1_7 : s_748);
                        float total_n_1_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_8) : "f"(sums_3[8]), "f"(errs_1[8]));
                        s_748 = ((lane_in_warp == 8) ? total_n_1_8 : s_748);
                        float total_n_1_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_9) : "f"(sums_3[9]), "f"(errs_1[9]));
                        s_748 = ((lane_in_warp == 9) ? total_n_1_9 : s_748);
                        float total_n_1_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_10) : "f"(sums_3[10]), "f"(errs_1[10]));
                        s_748 = ((lane_in_warp == 10) ? total_n_1_10 : s_748);
                        float total_n_1_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_11) : "f"(sums_3[11]), "f"(errs_1[11]));
                        s_748 = ((lane_in_warp == 11) ? total_n_1_11 : s_748);
                        float total_n_1_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_12) : "f"(sums_3[12]), "f"(errs_1[12]));
                        s_748 = ((lane_in_warp == 12) ? total_n_1_12 : s_748);
                        float total_n_1_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_13) : "f"(sums_3[13]), "f"(errs_1[13]));
                        s_748 = ((lane_in_warp == 13) ? total_n_1_13 : s_748);
                        float total_n_1_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_14) : "f"(sums_3[14]), "f"(errs_1[14]));
                        s_748 = ((lane_in_warp == 14) ? total_n_1_14 : s_748);
                        float total_n_1_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_15) : "f"(sums_3[15]), "f"(errs_1[15]));
                        s_748 = ((lane_in_warp == 15) ? total_n_1_15 : s_748);
                        float total_n_1_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_16) : "f"(sums_3[16]), "f"(errs_1[16]));
                        s_748 = ((lane_in_warp == 16) ? total_n_1_16 : s_748);
                        float total_n_1_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_17) : "f"(sums_3[17]), "f"(errs_1[17]));
                        s_748 = ((lane_in_warp == 17) ? total_n_1_17 : s_748);
                        float total_n_1_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_18) : "f"(sums_3[18]), "f"(errs_1[18]));
                        s_748 = ((lane_in_warp == 18) ? total_n_1_18 : s_748);
                        float total_n_1_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_19) : "f"(sums_3[19]), "f"(errs_1[19]));
                        s_748 = ((lane_in_warp == 19) ? total_n_1_19 : s_748);
                        float total_n_1_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_20) : "f"(sums_3[20]), "f"(errs_1[20]));
                        s_748 = ((lane_in_warp == 20) ? total_n_1_20 : s_748);
                        float total_n_1_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_21) : "f"(sums_3[21]), "f"(errs_1[21]));
                        s_748 = ((lane_in_warp == 21) ? total_n_1_21 : s_748);
                        float total_n_1_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_22) : "f"(sums_3[22]), "f"(errs_1[22]));
                        s_748 = ((lane_in_warp == 22) ? total_n_1_22 : s_748);
                        float total_n_1_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_23) : "f"(sums_3[23]), "f"(errs_1[23]));
                        s_748 = ((lane_in_warp == 23) ? total_n_1_23 : s_748);
                        float q_772;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(q_772) : "f"(sums_3[24]), "f"(errs_1[24]));
                        s_0_1 = s_748;
                        q_1 = q_772;
                    } else {
                        unsigned int row_13 = token % 64;
                        unsigned int first_task_4 = mb_1 * num_splits;
                        float sums_4[25];
                        unsigned int split_4 = lane_in_warp * 2;
                        unsigned int safe_split_4 = ((split_4 < num_splits) ? split_4 : 0);
                        unsigned long long record_4 = ((unsigned long long)first_task_4 + (unsigned long long)safe_split_4) * 1536 + (unsigned long long)(row_13 * 24);
                        float _vec_load_36[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_4 + 0);
                            _vec_load_36[0 + 0] = _v4.x;
                            _vec_load_36[0 + 1] = _v4.y;
                            _vec_load_36[0 + 2] = _v4.z;
                            _vec_load_36[0 + 3] = _v4.w;
                        }
                        float value_23_96 = ((split_4 < num_splits) ? _vec_load_36[0] : 0.0f);
                        sums_4[0] = value_23_96;
                        float value_23_97 = ((split_4 < num_splits) ? _vec_load_36[1] : 0.0f);
                        sums_4[1] = value_23_97;
                        float value_23_98 = ((split_4 < num_splits) ? _vec_load_36[2] : 0.0f);
                        sums_4[2] = value_23_98;
                        float value_23_99 = ((split_4 < num_splits) ? _vec_load_36[3] : 0.0f);
                        sums_4[3] = value_23_99;
                        float _vec_load_37[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 4) + 0);
                            _vec_load_37[0 + 0] = _v4.x;
                            _vec_load_37[0 + 1] = _v4.y;
                            _vec_load_37[0 + 2] = _v4.z;
                            _vec_load_37[0 + 3] = _v4.w;
                        }
                        float value_23_100 = ((split_4 < num_splits) ? _vec_load_37[0] : 0.0f);
                        sums_4[4] = value_23_100;
                        float value_23_101 = ((split_4 < num_splits) ? _vec_load_37[1] : 0.0f);
                        sums_4[5] = value_23_101;
                        float value_23_102 = ((split_4 < num_splits) ? _vec_load_37[2] : 0.0f);
                        sums_4[6] = value_23_102;
                        float value_23_103 = ((split_4 < num_splits) ? _vec_load_37[3] : 0.0f);
                        sums_4[7] = value_23_103;
                        float _vec_load_38[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 8) + 0);
                            _vec_load_38[0 + 0] = _v4.x;
                            _vec_load_38[0 + 1] = _v4.y;
                            _vec_load_38[0 + 2] = _v4.z;
                            _vec_load_38[0 + 3] = _v4.w;
                        }
                        float value_23_104 = ((split_4 < num_splits) ? _vec_load_38[0] : 0.0f);
                        sums_4[8] = value_23_104;
                        float value_23_105 = ((split_4 < num_splits) ? _vec_load_38[1] : 0.0f);
                        sums_4[9] = value_23_105;
                        float value_23_106 = ((split_4 < num_splits) ? _vec_load_38[2] : 0.0f);
                        sums_4[10] = value_23_106;
                        float value_23_107 = ((split_4 < num_splits) ? _vec_load_38[3] : 0.0f);
                        sums_4[11] = value_23_107;
                        float _vec_load_39[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 12) + 0);
                            _vec_load_39[0 + 0] = _v4.x;
                            _vec_load_39[0 + 1] = _v4.y;
                            _vec_load_39[0 + 2] = _v4.z;
                            _vec_load_39[0 + 3] = _v4.w;
                        }
                        float value_23_108 = ((split_4 < num_splits) ? _vec_load_39[0] : 0.0f);
                        sums_4[12] = value_23_108;
                        float value_23_109 = ((split_4 < num_splits) ? _vec_load_39[1] : 0.0f);
                        sums_4[13] = value_23_109;
                        float value_23_110 = ((split_4 < num_splits) ? _vec_load_39[2] : 0.0f);
                        sums_4[14] = value_23_110;
                        float value_23_111 = ((split_4 < num_splits) ? _vec_load_39[3] : 0.0f);
                        sums_4[15] = value_23_111;
                        float _vec_load_40[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 16) + 0);
                            _vec_load_40[0 + 0] = _v4.x;
                            _vec_load_40[0 + 1] = _v4.y;
                            _vec_load_40[0 + 2] = _v4.z;
                            _vec_load_40[0 + 3] = _v4.w;
                        }
                        float value_23_112 = ((split_4 < num_splits) ? _vec_load_40[0] : 0.0f);
                        sums_4[16] = value_23_112;
                        float value_23_113 = ((split_4 < num_splits) ? _vec_load_40[1] : 0.0f);
                        sums_4[17] = value_23_113;
                        float value_23_114 = ((split_4 < num_splits) ? _vec_load_40[2] : 0.0f);
                        sums_4[18] = value_23_114;
                        float value_23_115 = ((split_4 < num_splits) ? _vec_load_40[3] : 0.0f);
                        sums_4[19] = value_23_115;
                        float _vec_load_41[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_4 + 20) + 0);
                            _vec_load_41[0 + 0] = _v4.x;
                            _vec_load_41[0 + 1] = _v4.y;
                            _vec_load_41[0 + 2] = _v4.z;
                            _vec_load_41[0 + 3] = _v4.w;
                        }
                        float value_23_116 = ((split_4 < num_splits) ? _vec_load_41[0] : 0.0f);
                        sums_4[20] = value_23_116;
                        float value_23_117 = ((split_4 < num_splits) ? _vec_load_41[1] : 0.0f);
                        sums_4[21] = value_23_117;
                        float value_23_118 = ((split_4 < num_splits) ? _vec_load_41[2] : 0.0f);
                        sums_4[22] = value_23_118;
                        float value_23_119 = ((split_4 < num_splits) ? _vec_load_41[3] : 0.0f);
                        sums_4[23] = value_23_119;
                        float q_word_4 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_4 * 64 + (unsigned long long)(safe_split_4 * 64) + (unsigned long long)row_13];
                        float q_value_4 = ((split_4 < num_splits) ? q_word_4 : 0.0f);
                        sums_4[24] = q_value_4;
                        unsigned int split_23_2 = lane_in_warp * 2 + 1;
                        unsigned int safe_split_24_2 = ((split_23_2 < num_splits) ? split_23_2 : 0);
                        unsigned long long record_25_2 = ((unsigned long long)first_task_4 + (unsigned long long)safe_split_24_2) * 1536 + (unsigned long long)(row_13 * 24);
                        float _vec_load_42[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25_2 + 0);
                            _vec_load_42[0 + 0] = _v4.x;
                            _vec_load_42[0 + 1] = _v4.y;
                            _vec_load_42[0 + 2] = _v4.z;
                            _vec_load_42[0 + 3] = _v4.w;
                        }
                        float value_23_120 = ((split_23_2 < num_splits) ? _vec_load_42[0] : 0.0f);
                        float _add_992;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_992) : "f"(sums_4[0]), "f"(value_23_120));
                        sums_4[0] = _add_992;
                        float value_23_121 = ((split_23_2 < num_splits) ? _vec_load_42[1] : 0.0f);
                        float _add_993;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_993) : "f"(sums_4[1]), "f"(value_23_121));
                        sums_4[1] = _add_993;
                        float value_23_122 = ((split_23_2 < num_splits) ? _vec_load_42[2] : 0.0f);
                        float _add_994;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_994) : "f"(sums_4[2]), "f"(value_23_122));
                        sums_4[2] = _add_994;
                        float value_23_123 = ((split_23_2 < num_splits) ? _vec_load_42[3] : 0.0f);
                        float _add_995;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_995) : "f"(sums_4[3]), "f"(value_23_123));
                        sums_4[3] = _add_995;
                        float _vec_load_43[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 4) + 0);
                            _vec_load_43[0 + 0] = _v4.x;
                            _vec_load_43[0 + 1] = _v4.y;
                            _vec_load_43[0 + 2] = _v4.z;
                            _vec_load_43[0 + 3] = _v4.w;
                        }
                        float value_23_124 = ((split_23_2 < num_splits) ? _vec_load_43[0] : 0.0f);
                        float _add_997;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_997) : "f"(sums_4[4]), "f"(value_23_124));
                        sums_4[4] = _add_997;
                        float value_23_125 = ((split_23_2 < num_splits) ? _vec_load_43[1] : 0.0f);
                        float _add_998;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_998) : "f"(sums_4[5]), "f"(value_23_125));
                        sums_4[5] = _add_998;
                        float value_23_126 = ((split_23_2 < num_splits) ? _vec_load_43[2] : 0.0f);
                        float _add_999;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_999) : "f"(sums_4[6]), "f"(value_23_126));
                        sums_4[6] = _add_999;
                        float value_23_127 = ((split_23_2 < num_splits) ? _vec_load_43[3] : 0.0f);
                        float _add_1000;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1000) : "f"(sums_4[7]), "f"(value_23_127));
                        sums_4[7] = _add_1000;
                        float _vec_load_44[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 8) + 0);
                            _vec_load_44[0 + 0] = _v4.x;
                            _vec_load_44[0 + 1] = _v4.y;
                            _vec_load_44[0 + 2] = _v4.z;
                            _vec_load_44[0 + 3] = _v4.w;
                        }
                        float value_23_128 = ((split_23_2 < num_splits) ? _vec_load_44[0] : 0.0f);
                        float _add_1002;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1002) : "f"(sums_4[8]), "f"(value_23_128));
                        sums_4[8] = _add_1002;
                        float value_23_129 = ((split_23_2 < num_splits) ? _vec_load_44[1] : 0.0f);
                        float _add_1003;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1003) : "f"(sums_4[9]), "f"(value_23_129));
                        sums_4[9] = _add_1003;
                        float value_23_130 = ((split_23_2 < num_splits) ? _vec_load_44[2] : 0.0f);
                        float _add_1004;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1004) : "f"(sums_4[10]), "f"(value_23_130));
                        sums_4[10] = _add_1004;
                        float value_23_131 = ((split_23_2 < num_splits) ? _vec_load_44[3] : 0.0f);
                        float _add_1005;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1005) : "f"(sums_4[11]), "f"(value_23_131));
                        sums_4[11] = _add_1005;
                        float _vec_load_45[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 12) + 0);
                            _vec_load_45[0 + 0] = _v4.x;
                            _vec_load_45[0 + 1] = _v4.y;
                            _vec_load_45[0 + 2] = _v4.z;
                            _vec_load_45[0 + 3] = _v4.w;
                        }
                        float value_23_132 = ((split_23_2 < num_splits) ? _vec_load_45[0] : 0.0f);
                        float _add_1007;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1007) : "f"(sums_4[12]), "f"(value_23_132));
                        sums_4[12] = _add_1007;
                        float value_23_133 = ((split_23_2 < num_splits) ? _vec_load_45[1] : 0.0f);
                        float _add_1008;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1008) : "f"(sums_4[13]), "f"(value_23_133));
                        sums_4[13] = _add_1008;
                        float value_23_134 = ((split_23_2 < num_splits) ? _vec_load_45[2] : 0.0f);
                        float _add_1009;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1009) : "f"(sums_4[14]), "f"(value_23_134));
                        sums_4[14] = _add_1009;
                        float value_23_135 = ((split_23_2 < num_splits) ? _vec_load_45[3] : 0.0f);
                        float _add_1010;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1010) : "f"(sums_4[15]), "f"(value_23_135));
                        sums_4[15] = _add_1010;
                        float _vec_load_46[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 16) + 0);
                            _vec_load_46[0 + 0] = _v4.x;
                            _vec_load_46[0 + 1] = _v4.y;
                            _vec_load_46[0 + 2] = _v4.z;
                            _vec_load_46[0 + 3] = _v4.w;
                        }
                        float value_23_136 = ((split_23_2 < num_splits) ? _vec_load_46[0] : 0.0f);
                        float _add_1012;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1012) : "f"(sums_4[16]), "f"(value_23_136));
                        sums_4[16] = _add_1012;
                        float value_23_137 = ((split_23_2 < num_splits) ? _vec_load_46[1] : 0.0f);
                        float _add_1013;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1013) : "f"(sums_4[17]), "f"(value_23_137));
                        sums_4[17] = _add_1013;
                        float value_23_138 = ((split_23_2 < num_splits) ? _vec_load_46[2] : 0.0f);
                        float _add_1014;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1014) : "f"(sums_4[18]), "f"(value_23_138));
                        sums_4[18] = _add_1014;
                        float value_23_139 = ((split_23_2 < num_splits) ? _vec_load_46[3] : 0.0f);
                        float _add_1015;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1015) : "f"(sums_4[19]), "f"(value_23_139));
                        sums_4[19] = _add_1015;
                        float _vec_load_47[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25_2 + 20) + 0);
                            _vec_load_47[0 + 0] = _v4.x;
                            _vec_load_47[0 + 1] = _v4.y;
                            _vec_load_47[0 + 2] = _v4.z;
                            _vec_load_47[0 + 3] = _v4.w;
                        }
                        float value_23_140 = ((split_23_2 < num_splits) ? _vec_load_47[0] : 0.0f);
                        float _add_1017;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1017) : "f"(sums_4[20]), "f"(value_23_140));
                        sums_4[20] = _add_1017;
                        float value_23_141 = ((split_23_2 < num_splits) ? _vec_load_47[1] : 0.0f);
                        float _add_1018;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1018) : "f"(sums_4[21]), "f"(value_23_141));
                        sums_4[21] = _add_1018;
                        float value_23_142 = ((split_23_2 < num_splits) ? _vec_load_47[2] : 0.0f);
                        float _add_1019;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1019) : "f"(sums_4[22]), "f"(value_23_142));
                        sums_4[22] = _add_1019;
                        float value_23_143 = ((split_23_2 < num_splits) ? _vec_load_47[3] : 0.0f);
                        float _add_1020;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1020) : "f"(sums_4[23]), "f"(value_23_143));
                        sums_4[23] = _add_1020;
                        float q_word_50_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_4 * 64 + (unsigned long long)(safe_split_24_2 * 64) + (unsigned long long)row_13];
                        float q_value_51_1 = ((split_23_2 < num_splits) ? q_word_50_1 : 0.0f);
                        float _add_1021;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1021) : "f"(sums_4[24]), "f"(q_value_51_1));
                        sums_4[24] = _add_1021;
                        float _shfl_xor_sync_1023 = __shfl_xor_sync(0xFFFFFFFF, sums_4[0], 1);
                        float _add_1022;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1022) : "f"(sums_4[0]), "f"(_shfl_xor_sync_1023));
                        sums_4[0] = _add_1022;
                        float _shfl_xor_sync_1025 = __shfl_xor_sync(0xFFFFFFFF, sums_4[1], 1);
                        float _add_1024;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1024) : "f"(sums_4[1]), "f"(_shfl_xor_sync_1025));
                        sums_4[1] = _add_1024;
                        float _shfl_xor_sync_1027 = __shfl_xor_sync(0xFFFFFFFF, sums_4[2], 1);
                        float _add_1026;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1026) : "f"(sums_4[2]), "f"(_shfl_xor_sync_1027));
                        sums_4[2] = _add_1026;
                        float _shfl_xor_sync_1029 = __shfl_xor_sync(0xFFFFFFFF, sums_4[3], 1);
                        float _add_1028;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1028) : "f"(sums_4[3]), "f"(_shfl_xor_sync_1029));
                        sums_4[3] = _add_1028;
                        float _shfl_xor_sync_1031 = __shfl_xor_sync(0xFFFFFFFF, sums_4[4], 1);
                        float _add_1030;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1030) : "f"(sums_4[4]), "f"(_shfl_xor_sync_1031));
                        sums_4[4] = _add_1030;
                        float _shfl_xor_sync_1033 = __shfl_xor_sync(0xFFFFFFFF, sums_4[5], 1);
                        float _add_1032;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1032) : "f"(sums_4[5]), "f"(_shfl_xor_sync_1033));
                        sums_4[5] = _add_1032;
                        float _shfl_xor_sync_1035 = __shfl_xor_sync(0xFFFFFFFF, sums_4[6], 1);
                        float _add_1034;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1034) : "f"(sums_4[6]), "f"(_shfl_xor_sync_1035));
                        sums_4[6] = _add_1034;
                        float _shfl_xor_sync_1037 = __shfl_xor_sync(0xFFFFFFFF, sums_4[7], 1);
                        float _add_1036;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1036) : "f"(sums_4[7]), "f"(_shfl_xor_sync_1037));
                        sums_4[7] = _add_1036;
                        float _shfl_xor_sync_1039 = __shfl_xor_sync(0xFFFFFFFF, sums_4[8], 1);
                        float _add_1038;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1038) : "f"(sums_4[8]), "f"(_shfl_xor_sync_1039));
                        sums_4[8] = _add_1038;
                        float _shfl_xor_sync_1041 = __shfl_xor_sync(0xFFFFFFFF, sums_4[9], 1);
                        float _add_1040;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1040) : "f"(sums_4[9]), "f"(_shfl_xor_sync_1041));
                        sums_4[9] = _add_1040;
                        float _shfl_xor_sync_1043 = __shfl_xor_sync(0xFFFFFFFF, sums_4[10], 1);
                        float _add_1042;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1042) : "f"(sums_4[10]), "f"(_shfl_xor_sync_1043));
                        sums_4[10] = _add_1042;
                        float _shfl_xor_sync_1045 = __shfl_xor_sync(0xFFFFFFFF, sums_4[11], 1);
                        float _add_1044;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1044) : "f"(sums_4[11]), "f"(_shfl_xor_sync_1045));
                        sums_4[11] = _add_1044;
                        float _shfl_xor_sync_1047 = __shfl_xor_sync(0xFFFFFFFF, sums_4[12], 1);
                        float _add_1046;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1046) : "f"(sums_4[12]), "f"(_shfl_xor_sync_1047));
                        sums_4[12] = _add_1046;
                        float _shfl_xor_sync_1049 = __shfl_xor_sync(0xFFFFFFFF, sums_4[13], 1);
                        float _add_1048;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1048) : "f"(sums_4[13]), "f"(_shfl_xor_sync_1049));
                        sums_4[13] = _add_1048;
                        float _shfl_xor_sync_1051 = __shfl_xor_sync(0xFFFFFFFF, sums_4[14], 1);
                        float _add_1050;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1050) : "f"(sums_4[14]), "f"(_shfl_xor_sync_1051));
                        sums_4[14] = _add_1050;
                        float _shfl_xor_sync_1053 = __shfl_xor_sync(0xFFFFFFFF, sums_4[15], 1);
                        float _add_1052;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1052) : "f"(sums_4[15]), "f"(_shfl_xor_sync_1053));
                        sums_4[15] = _add_1052;
                        float _shfl_xor_sync_1055 = __shfl_xor_sync(0xFFFFFFFF, sums_4[16], 1);
                        float _add_1054;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1054) : "f"(sums_4[16]), "f"(_shfl_xor_sync_1055));
                        sums_4[16] = _add_1054;
                        float _shfl_xor_sync_1057 = __shfl_xor_sync(0xFFFFFFFF, sums_4[17], 1);
                        float _add_1056;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1056) : "f"(sums_4[17]), "f"(_shfl_xor_sync_1057));
                        sums_4[17] = _add_1056;
                        float _shfl_xor_sync_1059 = __shfl_xor_sync(0xFFFFFFFF, sums_4[18], 1);
                        float _add_1058;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1058) : "f"(sums_4[18]), "f"(_shfl_xor_sync_1059));
                        sums_4[18] = _add_1058;
                        float _shfl_xor_sync_1061 = __shfl_xor_sync(0xFFFFFFFF, sums_4[19], 1);
                        float _add_1060;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1060) : "f"(sums_4[19]), "f"(_shfl_xor_sync_1061));
                        sums_4[19] = _add_1060;
                        float _shfl_xor_sync_1063 = __shfl_xor_sync(0xFFFFFFFF, sums_4[20], 1);
                        float _add_1062;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1062) : "f"(sums_4[20]), "f"(_shfl_xor_sync_1063));
                        sums_4[20] = _add_1062;
                        float _shfl_xor_sync_1065 = __shfl_xor_sync(0xFFFFFFFF, sums_4[21], 1);
                        float _add_1064;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1064) : "f"(sums_4[21]), "f"(_shfl_xor_sync_1065));
                        sums_4[21] = _add_1064;
                        float _shfl_xor_sync_1067 = __shfl_xor_sync(0xFFFFFFFF, sums_4[22], 1);
                        float _add_1066;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1066) : "f"(sums_4[22]), "f"(_shfl_xor_sync_1067));
                        sums_4[22] = _add_1066;
                        float _shfl_xor_sync_1069 = __shfl_xor_sync(0xFFFFFFFF, sums_4[23], 1);
                        float _add_1068;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1068) : "f"(sums_4[23]), "f"(_shfl_xor_sync_1069));
                        sums_4[23] = _add_1068;
                        float _shfl_xor_sync_1071 = __shfl_xor_sync(0xFFFFFFFF, sums_4[24], 1);
                        float _add_1070;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1070) : "f"(sums_4[24]), "f"(_shfl_xor_sync_1071));
                        sums_4[24] = _add_1070;
                        float _shfl_xor_sync_1073 = __shfl_xor_sync(0xFFFFFFFF, sums_4[0], 2);
                        float _add_1072;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1072) : "f"(sums_4[0]), "f"(_shfl_xor_sync_1073));
                        sums_4[0] = _add_1072;
                        float _shfl_xor_sync_1075 = __shfl_xor_sync(0xFFFFFFFF, sums_4[1], 2);
                        float _add_1074;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1074) : "f"(sums_4[1]), "f"(_shfl_xor_sync_1075));
                        sums_4[1] = _add_1074;
                        float _shfl_xor_sync_1077 = __shfl_xor_sync(0xFFFFFFFF, sums_4[2], 2);
                        float _add_1076;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1076) : "f"(sums_4[2]), "f"(_shfl_xor_sync_1077));
                        sums_4[2] = _add_1076;
                        float _shfl_xor_sync_1079 = __shfl_xor_sync(0xFFFFFFFF, sums_4[3], 2);
                        float _add_1078;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1078) : "f"(sums_4[3]), "f"(_shfl_xor_sync_1079));
                        sums_4[3] = _add_1078;
                        float _shfl_xor_sync_1081 = __shfl_xor_sync(0xFFFFFFFF, sums_4[4], 2);
                        float _add_1080;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1080) : "f"(sums_4[4]), "f"(_shfl_xor_sync_1081));
                        sums_4[4] = _add_1080;
                        float _shfl_xor_sync_1083 = __shfl_xor_sync(0xFFFFFFFF, sums_4[5], 2);
                        float _add_1082;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1082) : "f"(sums_4[5]), "f"(_shfl_xor_sync_1083));
                        sums_4[5] = _add_1082;
                        float _shfl_xor_sync_1085 = __shfl_xor_sync(0xFFFFFFFF, sums_4[6], 2);
                        float _add_1084;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1084) : "f"(sums_4[6]), "f"(_shfl_xor_sync_1085));
                        sums_4[6] = _add_1084;
                        float _shfl_xor_sync_1087 = __shfl_xor_sync(0xFFFFFFFF, sums_4[7], 2);
                        float _add_1086;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1086) : "f"(sums_4[7]), "f"(_shfl_xor_sync_1087));
                        sums_4[7] = _add_1086;
                        float _shfl_xor_sync_1089 = __shfl_xor_sync(0xFFFFFFFF, sums_4[8], 2);
                        float _add_1088;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1088) : "f"(sums_4[8]), "f"(_shfl_xor_sync_1089));
                        sums_4[8] = _add_1088;
                        float _shfl_xor_sync_1091 = __shfl_xor_sync(0xFFFFFFFF, sums_4[9], 2);
                        float _add_1090;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1090) : "f"(sums_4[9]), "f"(_shfl_xor_sync_1091));
                        sums_4[9] = _add_1090;
                        float _shfl_xor_sync_1093 = __shfl_xor_sync(0xFFFFFFFF, sums_4[10], 2);
                        float _add_1092;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1092) : "f"(sums_4[10]), "f"(_shfl_xor_sync_1093));
                        sums_4[10] = _add_1092;
                        float _shfl_xor_sync_1095 = __shfl_xor_sync(0xFFFFFFFF, sums_4[11], 2);
                        float _add_1094;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1094) : "f"(sums_4[11]), "f"(_shfl_xor_sync_1095));
                        sums_4[11] = _add_1094;
                        float _shfl_xor_sync_1097 = __shfl_xor_sync(0xFFFFFFFF, sums_4[12], 2);
                        float _add_1096;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1096) : "f"(sums_4[12]), "f"(_shfl_xor_sync_1097));
                        sums_4[12] = _add_1096;
                        float _shfl_xor_sync_1099 = __shfl_xor_sync(0xFFFFFFFF, sums_4[13], 2);
                        float _add_1098;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1098) : "f"(sums_4[13]), "f"(_shfl_xor_sync_1099));
                        sums_4[13] = _add_1098;
                        float _shfl_xor_sync_1101 = __shfl_xor_sync(0xFFFFFFFF, sums_4[14], 2);
                        float _add_1100;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1100) : "f"(sums_4[14]), "f"(_shfl_xor_sync_1101));
                        sums_4[14] = _add_1100;
                        float _shfl_xor_sync_1103 = __shfl_xor_sync(0xFFFFFFFF, sums_4[15], 2);
                        float _add_1102;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1102) : "f"(sums_4[15]), "f"(_shfl_xor_sync_1103));
                        sums_4[15] = _add_1102;
                        float _shfl_xor_sync_1105 = __shfl_xor_sync(0xFFFFFFFF, sums_4[16], 2);
                        float _add_1104;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1104) : "f"(sums_4[16]), "f"(_shfl_xor_sync_1105));
                        sums_4[16] = _add_1104;
                        float _shfl_xor_sync_1107 = __shfl_xor_sync(0xFFFFFFFF, sums_4[17], 2);
                        float _add_1106;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1106) : "f"(sums_4[17]), "f"(_shfl_xor_sync_1107));
                        sums_4[17] = _add_1106;
                        float _shfl_xor_sync_1109 = __shfl_xor_sync(0xFFFFFFFF, sums_4[18], 2);
                        float _add_1108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1108) : "f"(sums_4[18]), "f"(_shfl_xor_sync_1109));
                        sums_4[18] = _add_1108;
                        float _shfl_xor_sync_1111 = __shfl_xor_sync(0xFFFFFFFF, sums_4[19], 2);
                        float _add_1110;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1110) : "f"(sums_4[19]), "f"(_shfl_xor_sync_1111));
                        sums_4[19] = _add_1110;
                        float _shfl_xor_sync_1113 = __shfl_xor_sync(0xFFFFFFFF, sums_4[20], 2);
                        float _add_1112;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1112) : "f"(sums_4[20]), "f"(_shfl_xor_sync_1113));
                        sums_4[20] = _add_1112;
                        float _shfl_xor_sync_1115 = __shfl_xor_sync(0xFFFFFFFF, sums_4[21], 2);
                        float _add_1114;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1114) : "f"(sums_4[21]), "f"(_shfl_xor_sync_1115));
                        sums_4[21] = _add_1114;
                        float _shfl_xor_sync_1117 = __shfl_xor_sync(0xFFFFFFFF, sums_4[22], 2);
                        float _add_1116;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1116) : "f"(sums_4[22]), "f"(_shfl_xor_sync_1117));
                        sums_4[22] = _add_1116;
                        float _shfl_xor_sync_1119 = __shfl_xor_sync(0xFFFFFFFF, sums_4[23], 2);
                        float _add_1118;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1118) : "f"(sums_4[23]), "f"(_shfl_xor_sync_1119));
                        sums_4[23] = _add_1118;
                        float _shfl_xor_sync_1121 = __shfl_xor_sync(0xFFFFFFFF, sums_4[24], 2);
                        float _add_1120;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1120) : "f"(sums_4[24]), "f"(_shfl_xor_sync_1121));
                        sums_4[24] = _add_1120;
                        float _shfl_xor_sync_1123 = __shfl_xor_sync(0xFFFFFFFF, sums_4[0], 4);
                        float _add_1122;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1122) : "f"(sums_4[0]), "f"(_shfl_xor_sync_1123));
                        sums_4[0] = _add_1122;
                        float _shfl_xor_sync_1125 = __shfl_xor_sync(0xFFFFFFFF, sums_4[1], 4);
                        float _add_1124;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1124) : "f"(sums_4[1]), "f"(_shfl_xor_sync_1125));
                        sums_4[1] = _add_1124;
                        float _shfl_xor_sync_1127 = __shfl_xor_sync(0xFFFFFFFF, sums_4[2], 4);
                        float _add_1126;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1126) : "f"(sums_4[2]), "f"(_shfl_xor_sync_1127));
                        sums_4[2] = _add_1126;
                        float _shfl_xor_sync_1129 = __shfl_xor_sync(0xFFFFFFFF, sums_4[3], 4);
                        float _add_1128;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1128) : "f"(sums_4[3]), "f"(_shfl_xor_sync_1129));
                        sums_4[3] = _add_1128;
                        float _shfl_xor_sync_1131 = __shfl_xor_sync(0xFFFFFFFF, sums_4[4], 4);
                        float _add_1130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1130) : "f"(sums_4[4]), "f"(_shfl_xor_sync_1131));
                        sums_4[4] = _add_1130;
                        float _shfl_xor_sync_1133 = __shfl_xor_sync(0xFFFFFFFF, sums_4[5], 4);
                        float _add_1132;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1132) : "f"(sums_4[5]), "f"(_shfl_xor_sync_1133));
                        sums_4[5] = _add_1132;
                        float _shfl_xor_sync_1135 = __shfl_xor_sync(0xFFFFFFFF, sums_4[6], 4);
                        float _add_1134;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1134) : "f"(sums_4[6]), "f"(_shfl_xor_sync_1135));
                        sums_4[6] = _add_1134;
                        float _shfl_xor_sync_1137 = __shfl_xor_sync(0xFFFFFFFF, sums_4[7], 4);
                        float _add_1136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1136) : "f"(sums_4[7]), "f"(_shfl_xor_sync_1137));
                        sums_4[7] = _add_1136;
                        float _shfl_xor_sync_1139 = __shfl_xor_sync(0xFFFFFFFF, sums_4[8], 4);
                        float _add_1138;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1138) : "f"(sums_4[8]), "f"(_shfl_xor_sync_1139));
                        sums_4[8] = _add_1138;
                        float _shfl_xor_sync_1141 = __shfl_xor_sync(0xFFFFFFFF, sums_4[9], 4);
                        float _add_1140;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1140) : "f"(sums_4[9]), "f"(_shfl_xor_sync_1141));
                        sums_4[9] = _add_1140;
                        float _shfl_xor_sync_1143 = __shfl_xor_sync(0xFFFFFFFF, sums_4[10], 4);
                        float _add_1142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1142) : "f"(sums_4[10]), "f"(_shfl_xor_sync_1143));
                        sums_4[10] = _add_1142;
                        float _shfl_xor_sync_1145 = __shfl_xor_sync(0xFFFFFFFF, sums_4[11], 4);
                        float _add_1144;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1144) : "f"(sums_4[11]), "f"(_shfl_xor_sync_1145));
                        sums_4[11] = _add_1144;
                        float _shfl_xor_sync_1147 = __shfl_xor_sync(0xFFFFFFFF, sums_4[12], 4);
                        float _add_1146;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1146) : "f"(sums_4[12]), "f"(_shfl_xor_sync_1147));
                        sums_4[12] = _add_1146;
                        float _shfl_xor_sync_1149 = __shfl_xor_sync(0xFFFFFFFF, sums_4[13], 4);
                        float _add_1148;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1148) : "f"(sums_4[13]), "f"(_shfl_xor_sync_1149));
                        sums_4[13] = _add_1148;
                        float _shfl_xor_sync_1151 = __shfl_xor_sync(0xFFFFFFFF, sums_4[14], 4);
                        float _add_1150;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1150) : "f"(sums_4[14]), "f"(_shfl_xor_sync_1151));
                        sums_4[14] = _add_1150;
                        float _shfl_xor_sync_1153 = __shfl_xor_sync(0xFFFFFFFF, sums_4[15], 4);
                        float _add_1152;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1152) : "f"(sums_4[15]), "f"(_shfl_xor_sync_1153));
                        sums_4[15] = _add_1152;
                        float _shfl_xor_sync_1155 = __shfl_xor_sync(0xFFFFFFFF, sums_4[16], 4);
                        float _add_1154;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1154) : "f"(sums_4[16]), "f"(_shfl_xor_sync_1155));
                        sums_4[16] = _add_1154;
                        float _shfl_xor_sync_1157 = __shfl_xor_sync(0xFFFFFFFF, sums_4[17], 4);
                        float _add_1156;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1156) : "f"(sums_4[17]), "f"(_shfl_xor_sync_1157));
                        sums_4[17] = _add_1156;
                        float _shfl_xor_sync_1159 = __shfl_xor_sync(0xFFFFFFFF, sums_4[18], 4);
                        float _add_1158;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1158) : "f"(sums_4[18]), "f"(_shfl_xor_sync_1159));
                        sums_4[18] = _add_1158;
                        float _shfl_xor_sync_1161 = __shfl_xor_sync(0xFFFFFFFF, sums_4[19], 4);
                        float _add_1160;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1160) : "f"(sums_4[19]), "f"(_shfl_xor_sync_1161));
                        sums_4[19] = _add_1160;
                        float _shfl_xor_sync_1163 = __shfl_xor_sync(0xFFFFFFFF, sums_4[20], 4);
                        float _add_1162;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1162) : "f"(sums_4[20]), "f"(_shfl_xor_sync_1163));
                        sums_4[20] = _add_1162;
                        float _shfl_xor_sync_1165 = __shfl_xor_sync(0xFFFFFFFF, sums_4[21], 4);
                        float _add_1164;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1164) : "f"(sums_4[21]), "f"(_shfl_xor_sync_1165));
                        sums_4[21] = _add_1164;
                        float _shfl_xor_sync_1167 = __shfl_xor_sync(0xFFFFFFFF, sums_4[22], 4);
                        float _add_1166;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1166) : "f"(sums_4[22]), "f"(_shfl_xor_sync_1167));
                        sums_4[22] = _add_1166;
                        float _shfl_xor_sync_1169 = __shfl_xor_sync(0xFFFFFFFF, sums_4[23], 4);
                        float _add_1168;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1168) : "f"(sums_4[23]), "f"(_shfl_xor_sync_1169));
                        sums_4[23] = _add_1168;
                        float _shfl_xor_sync_1171 = __shfl_xor_sync(0xFFFFFFFF, sums_4[24], 4);
                        float _add_1170;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1170) : "f"(sums_4[24]), "f"(_shfl_xor_sync_1171));
                        sums_4[24] = _add_1170;
                        float _shfl_xor_sync_1173 = __shfl_xor_sync(0xFFFFFFFF, sums_4[0], 8);
                        float _add_1172;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1172) : "f"(sums_4[0]), "f"(_shfl_xor_sync_1173));
                        sums_4[0] = _add_1172;
                        float _shfl_xor_sync_1175 = __shfl_xor_sync(0xFFFFFFFF, sums_4[1], 8);
                        float _add_1174;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1174) : "f"(sums_4[1]), "f"(_shfl_xor_sync_1175));
                        sums_4[1] = _add_1174;
                        float _shfl_xor_sync_1177 = __shfl_xor_sync(0xFFFFFFFF, sums_4[2], 8);
                        float _add_1176;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1176) : "f"(sums_4[2]), "f"(_shfl_xor_sync_1177));
                        sums_4[2] = _add_1176;
                        float _shfl_xor_sync_1179 = __shfl_xor_sync(0xFFFFFFFF, sums_4[3], 8);
                        float _add_1178;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1178) : "f"(sums_4[3]), "f"(_shfl_xor_sync_1179));
                        sums_4[3] = _add_1178;
                        float _shfl_xor_sync_1181 = __shfl_xor_sync(0xFFFFFFFF, sums_4[4], 8);
                        float _add_1180;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1180) : "f"(sums_4[4]), "f"(_shfl_xor_sync_1181));
                        sums_4[4] = _add_1180;
                        float _shfl_xor_sync_1183 = __shfl_xor_sync(0xFFFFFFFF, sums_4[5], 8);
                        float _add_1182;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1182) : "f"(sums_4[5]), "f"(_shfl_xor_sync_1183));
                        sums_4[5] = _add_1182;
                        float _shfl_xor_sync_1185 = __shfl_xor_sync(0xFFFFFFFF, sums_4[6], 8);
                        float _add_1184;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1184) : "f"(sums_4[6]), "f"(_shfl_xor_sync_1185));
                        sums_4[6] = _add_1184;
                        float _shfl_xor_sync_1187 = __shfl_xor_sync(0xFFFFFFFF, sums_4[7], 8);
                        float _add_1186;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1186) : "f"(sums_4[7]), "f"(_shfl_xor_sync_1187));
                        sums_4[7] = _add_1186;
                        float _shfl_xor_sync_1189 = __shfl_xor_sync(0xFFFFFFFF, sums_4[8], 8);
                        float _add_1188;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1188) : "f"(sums_4[8]), "f"(_shfl_xor_sync_1189));
                        sums_4[8] = _add_1188;
                        float _shfl_xor_sync_1191 = __shfl_xor_sync(0xFFFFFFFF, sums_4[9], 8);
                        float _add_1190;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1190) : "f"(sums_4[9]), "f"(_shfl_xor_sync_1191));
                        sums_4[9] = _add_1190;
                        float _shfl_xor_sync_1193 = __shfl_xor_sync(0xFFFFFFFF, sums_4[10], 8);
                        float _add_1192;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1192) : "f"(sums_4[10]), "f"(_shfl_xor_sync_1193));
                        sums_4[10] = _add_1192;
                        float _shfl_xor_sync_1195 = __shfl_xor_sync(0xFFFFFFFF, sums_4[11], 8);
                        float _add_1194;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1194) : "f"(sums_4[11]), "f"(_shfl_xor_sync_1195));
                        sums_4[11] = _add_1194;
                        float _shfl_xor_sync_1197 = __shfl_xor_sync(0xFFFFFFFF, sums_4[12], 8);
                        float _add_1196;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1196) : "f"(sums_4[12]), "f"(_shfl_xor_sync_1197));
                        sums_4[12] = _add_1196;
                        float _shfl_xor_sync_1199 = __shfl_xor_sync(0xFFFFFFFF, sums_4[13], 8);
                        float _add_1198;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1198) : "f"(sums_4[13]), "f"(_shfl_xor_sync_1199));
                        sums_4[13] = _add_1198;
                        float _shfl_xor_sync_1201 = __shfl_xor_sync(0xFFFFFFFF, sums_4[14], 8);
                        float _add_1200;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1200) : "f"(sums_4[14]), "f"(_shfl_xor_sync_1201));
                        sums_4[14] = _add_1200;
                        float _shfl_xor_sync_1203 = __shfl_xor_sync(0xFFFFFFFF, sums_4[15], 8);
                        float _add_1202;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1202) : "f"(sums_4[15]), "f"(_shfl_xor_sync_1203));
                        sums_4[15] = _add_1202;
                        float _shfl_xor_sync_1205 = __shfl_xor_sync(0xFFFFFFFF, sums_4[16], 8);
                        float _add_1204;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1204) : "f"(sums_4[16]), "f"(_shfl_xor_sync_1205));
                        sums_4[16] = _add_1204;
                        float _shfl_xor_sync_1207 = __shfl_xor_sync(0xFFFFFFFF, sums_4[17], 8);
                        float _add_1206;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1206) : "f"(sums_4[17]), "f"(_shfl_xor_sync_1207));
                        sums_4[17] = _add_1206;
                        float _shfl_xor_sync_1209 = __shfl_xor_sync(0xFFFFFFFF, sums_4[18], 8);
                        float _add_1208;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1208) : "f"(sums_4[18]), "f"(_shfl_xor_sync_1209));
                        sums_4[18] = _add_1208;
                        float _shfl_xor_sync_1211 = __shfl_xor_sync(0xFFFFFFFF, sums_4[19], 8);
                        float _add_1210;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1210) : "f"(sums_4[19]), "f"(_shfl_xor_sync_1211));
                        sums_4[19] = _add_1210;
                        float _shfl_xor_sync_1213 = __shfl_xor_sync(0xFFFFFFFF, sums_4[20], 8);
                        float _add_1212;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1212) : "f"(sums_4[20]), "f"(_shfl_xor_sync_1213));
                        sums_4[20] = _add_1212;
                        float _shfl_xor_sync_1215 = __shfl_xor_sync(0xFFFFFFFF, sums_4[21], 8);
                        float _add_1214;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1214) : "f"(sums_4[21]), "f"(_shfl_xor_sync_1215));
                        sums_4[21] = _add_1214;
                        float _shfl_xor_sync_1217 = __shfl_xor_sync(0xFFFFFFFF, sums_4[22], 8);
                        float _add_1216;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1216) : "f"(sums_4[22]), "f"(_shfl_xor_sync_1217));
                        sums_4[22] = _add_1216;
                        float _shfl_xor_sync_1219 = __shfl_xor_sync(0xFFFFFFFF, sums_4[23], 8);
                        float _add_1218;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1218) : "f"(sums_4[23]), "f"(_shfl_xor_sync_1219));
                        sums_4[23] = _add_1218;
                        float _shfl_xor_sync_1221 = __shfl_xor_sync(0xFFFFFFFF, sums_4[24], 8);
                        float _add_1220;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1220) : "f"(sums_4[24]), "f"(_shfl_xor_sync_1221));
                        sums_4[24] = _add_1220;
                        float _shfl_xor_sync_1223 = __shfl_xor_sync(0xFFFFFFFF, sums_4[0], 16);
                        float _add_1222;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1222) : "f"(sums_4[0]), "f"(_shfl_xor_sync_1223));
                        sums_4[0] = _add_1222;
                        float _shfl_xor_sync_1225 = __shfl_xor_sync(0xFFFFFFFF, sums_4[1], 16);
                        float _add_1224;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1224) : "f"(sums_4[1]), "f"(_shfl_xor_sync_1225));
                        sums_4[1] = _add_1224;
                        float _shfl_xor_sync_1227 = __shfl_xor_sync(0xFFFFFFFF, sums_4[2], 16);
                        float _add_1226;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1226) : "f"(sums_4[2]), "f"(_shfl_xor_sync_1227));
                        sums_4[2] = _add_1226;
                        float _shfl_xor_sync_1229 = __shfl_xor_sync(0xFFFFFFFF, sums_4[3], 16);
                        float _add_1228;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1228) : "f"(sums_4[3]), "f"(_shfl_xor_sync_1229));
                        sums_4[3] = _add_1228;
                        float _shfl_xor_sync_1231 = __shfl_xor_sync(0xFFFFFFFF, sums_4[4], 16);
                        float _add_1230;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1230) : "f"(sums_4[4]), "f"(_shfl_xor_sync_1231));
                        sums_4[4] = _add_1230;
                        float _shfl_xor_sync_1233 = __shfl_xor_sync(0xFFFFFFFF, sums_4[5], 16);
                        float _add_1232;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1232) : "f"(sums_4[5]), "f"(_shfl_xor_sync_1233));
                        sums_4[5] = _add_1232;
                        float _shfl_xor_sync_1235 = __shfl_xor_sync(0xFFFFFFFF, sums_4[6], 16);
                        float _add_1234;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1234) : "f"(sums_4[6]), "f"(_shfl_xor_sync_1235));
                        sums_4[6] = _add_1234;
                        float _shfl_xor_sync_1237 = __shfl_xor_sync(0xFFFFFFFF, sums_4[7], 16);
                        float _add_1236;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1236) : "f"(sums_4[7]), "f"(_shfl_xor_sync_1237));
                        sums_4[7] = _add_1236;
                        float _shfl_xor_sync_1239 = __shfl_xor_sync(0xFFFFFFFF, sums_4[8], 16);
                        float _add_1238;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1238) : "f"(sums_4[8]), "f"(_shfl_xor_sync_1239));
                        sums_4[8] = _add_1238;
                        float _shfl_xor_sync_1241 = __shfl_xor_sync(0xFFFFFFFF, sums_4[9], 16);
                        float _add_1240;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1240) : "f"(sums_4[9]), "f"(_shfl_xor_sync_1241));
                        sums_4[9] = _add_1240;
                        float _shfl_xor_sync_1243 = __shfl_xor_sync(0xFFFFFFFF, sums_4[10], 16);
                        float _add_1242;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1242) : "f"(sums_4[10]), "f"(_shfl_xor_sync_1243));
                        sums_4[10] = _add_1242;
                        float _shfl_xor_sync_1245 = __shfl_xor_sync(0xFFFFFFFF, sums_4[11], 16);
                        float _add_1244;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1244) : "f"(sums_4[11]), "f"(_shfl_xor_sync_1245));
                        sums_4[11] = _add_1244;
                        float _shfl_xor_sync_1247 = __shfl_xor_sync(0xFFFFFFFF, sums_4[12], 16);
                        float _add_1246;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1246) : "f"(sums_4[12]), "f"(_shfl_xor_sync_1247));
                        sums_4[12] = _add_1246;
                        float _shfl_xor_sync_1249 = __shfl_xor_sync(0xFFFFFFFF, sums_4[13], 16);
                        float _add_1248;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1248) : "f"(sums_4[13]), "f"(_shfl_xor_sync_1249));
                        sums_4[13] = _add_1248;
                        float _shfl_xor_sync_1251 = __shfl_xor_sync(0xFFFFFFFF, sums_4[14], 16);
                        float _add_1250;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1250) : "f"(sums_4[14]), "f"(_shfl_xor_sync_1251));
                        sums_4[14] = _add_1250;
                        float _shfl_xor_sync_1253 = __shfl_xor_sync(0xFFFFFFFF, sums_4[15], 16);
                        float _add_1252;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1252) : "f"(sums_4[15]), "f"(_shfl_xor_sync_1253));
                        sums_4[15] = _add_1252;
                        float _shfl_xor_sync_1255 = __shfl_xor_sync(0xFFFFFFFF, sums_4[16], 16);
                        float _add_1254;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1254) : "f"(sums_4[16]), "f"(_shfl_xor_sync_1255));
                        sums_4[16] = _add_1254;
                        float _shfl_xor_sync_1257 = __shfl_xor_sync(0xFFFFFFFF, sums_4[17], 16);
                        float _add_1256;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1256) : "f"(sums_4[17]), "f"(_shfl_xor_sync_1257));
                        sums_4[17] = _add_1256;
                        float _shfl_xor_sync_1259 = __shfl_xor_sync(0xFFFFFFFF, sums_4[18], 16);
                        float _add_1258;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1258) : "f"(sums_4[18]), "f"(_shfl_xor_sync_1259));
                        sums_4[18] = _add_1258;
                        float _shfl_xor_sync_1261 = __shfl_xor_sync(0xFFFFFFFF, sums_4[19], 16);
                        float _add_1260;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1260) : "f"(sums_4[19]), "f"(_shfl_xor_sync_1261));
                        sums_4[19] = _add_1260;
                        float _shfl_xor_sync_1263 = __shfl_xor_sync(0xFFFFFFFF, sums_4[20], 16);
                        float _add_1262;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1262) : "f"(sums_4[20]), "f"(_shfl_xor_sync_1263));
                        sums_4[20] = _add_1262;
                        float _shfl_xor_sync_1265 = __shfl_xor_sync(0xFFFFFFFF, sums_4[21], 16);
                        float _add_1264;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1264) : "f"(sums_4[21]), "f"(_shfl_xor_sync_1265));
                        sums_4[21] = _add_1264;
                        float _shfl_xor_sync_1267 = __shfl_xor_sync(0xFFFFFFFF, sums_4[22], 16);
                        float _add_1266;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1266) : "f"(sums_4[22]), "f"(_shfl_xor_sync_1267));
                        sums_4[22] = _add_1266;
                        float _shfl_xor_sync_1269 = __shfl_xor_sync(0xFFFFFFFF, sums_4[23], 16);
                        float _add_1268;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1268) : "f"(sums_4[23]), "f"(_shfl_xor_sync_1269));
                        sums_4[23] = _add_1268;
                        float _shfl_xor_sync_1271 = __shfl_xor_sync(0xFFFFFFFF, sums_4[24], 16);
                        float _add_1270;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1270) : "f"(sums_4[24]), "f"(_shfl_xor_sync_1271));
                        sums_4[24] = _add_1270;
                        float s_52_1 = 0.0f;
                        s_52_1 = ((lane_in_warp == 0) ? sums_4[0] : s_52_1);
                        s_52_1 = ((lane_in_warp == 1) ? sums_4[1] : s_52_1);
                        s_52_1 = ((lane_in_warp == 2) ? sums_4[2] : s_52_1);
                        s_52_1 = ((lane_in_warp == 3) ? sums_4[3] : s_52_1);
                        s_52_1 = ((lane_in_warp == 4) ? sums_4[4] : s_52_1);
                        s_52_1 = ((lane_in_warp == 5) ? sums_4[5] : s_52_1);
                        s_52_1 = ((lane_in_warp == 6) ? sums_4[6] : s_52_1);
                        s_52_1 = ((lane_in_warp == 7) ? sums_4[7] : s_52_1);
                        s_52_1 = ((lane_in_warp == 8) ? sums_4[8] : s_52_1);
                        s_52_1 = ((lane_in_warp == 9) ? sums_4[9] : s_52_1);
                        s_52_1 = ((lane_in_warp == 10) ? sums_4[10] : s_52_1);
                        s_52_1 = ((lane_in_warp == 11) ? sums_4[11] : s_52_1);
                        s_52_1 = ((lane_in_warp == 12) ? sums_4[12] : s_52_1);
                        s_52_1 = ((lane_in_warp == 13) ? sums_4[13] : s_52_1);
                        s_52_1 = ((lane_in_warp == 14) ? sums_4[14] : s_52_1);
                        s_52_1 = ((lane_in_warp == 15) ? sums_4[15] : s_52_1);
                        s_52_1 = ((lane_in_warp == 16) ? sums_4[16] : s_52_1);
                        s_52_1 = ((lane_in_warp == 17) ? sums_4[17] : s_52_1);
                        s_52_1 = ((lane_in_warp == 18) ? sums_4[18] : s_52_1);
                        s_52_1 = ((lane_in_warp == 19) ? sums_4[19] : s_52_1);
                        s_52_1 = ((lane_in_warp == 20) ? sums_4[20] : s_52_1);
                        s_52_1 = ((lane_in_warp == 21) ? sums_4[21] : s_52_1);
                        s_52_1 = ((lane_in_warp == 22) ? sums_4[22] : s_52_1);
                        s_52_1 = ((lane_in_warp == 23) ? sums_4[23] : s_52_1);
                        float q_53 = sums_4[24];
                        s_0_1 = s_52_1;
                        q_1 = q_53;
#endif
                    }
#if __CUDA_ARCH__ == 1070
                } else if (lane_in_warp >= 4) {
                    new_post_mix[(unsigned long long)token * 4 + (unsigned long long)(lane_in_warp - 4)] = _mul_0;
#else
                    s = s_0_1;
                    q = q_1;
#endif
                }
#if __CUDA_ARCH__ == 1070
                __syncwarp();
#else
                if (flag == 0) {
                    float _fma_1273;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1273) : "f"(q), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                    float _sqrt_1272;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_1272) : "f"(_fma_1273));
                    float r = __frcp_rn(_sqrt_1272);
                    float z;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(z) : "f"(r), "f"(s));
                    float a_map;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
                    float a = ((lane_in_warp < 24) ? a_map : 0.0f);
                    float _sub_1274;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1274) : "f"(0.0f), "f"(fabsf(a)));
                    float _exp_0 = __nv_expf(_sub_1274);
                    float add_42;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_42) : "f"(1.0f), "f"(_exp_0));
                    float _fdiv_rn_0 = __fdiv_rn(1.0f, add_42);
                    float _fdiv_rn_1 = __fdiv_rn(_exp_0, add_42);
                    float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
                    float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
                    float add_1_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(sig), "f"(hc_pre_eps));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(hc_post_scale), "f"(sig));
                    float logit = ((lane_in_warp >= 8 && lane_in_warp < 24) ? a : 0.0f);
                    float _shfl_xor_sync_1276 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
                    float _shfl_xor_sync_1277 = __shfl_xor_sync(0xFFFFFFFF, logit, 2);
                    float _shfl_xor_sync_1278 = __shfl_xor_sync(0xFFFFFFFF, logit, 3);
                    float _sub_1275;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1275) : "f"(logit), "f"(fmaxf(fmaxf(logit, _shfl_xor_sync_1276), fmaxf(_shfl_xor_sync_1277, _shfl_xor_sync_1278))));
                    float _exp_1 = __nv_expf(_sub_1275);
                    float _shfl_xor_sync_1282 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
                    float _add_1281;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1281) : "f"(_exp_1), "f"(_shfl_xor_sync_1282));
                    float _shfl_xor_sync_1284 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
                    float _shfl_xor_sync_1285 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
                    float _add_1283;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1283) : "f"(_shfl_xor_sync_1284), "f"(_shfl_xor_sync_1285));
                    float _add_1280;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1280) : "f"(_add_1281), "f"(_add_1283));
                    float _fdiv_1279 = __fdiv_rn(_exp_1, _add_1280);
                    float C;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(C) : "f"(_fdiv_1279), "f"(sinkhorn_eps));
                    float _shfl_xor_sync_1290 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                    float _add_1289;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1289) : "f"(C), "f"(_shfl_xor_sync_1290));
                    float _shfl_xor_sync_1292 = __shfl_xor_sync(0xFFFFFFFF, C, 24);
                    float _shfl_xor_sync_1293 = __shfl_xor_sync(0xFFFFFFFF, C, 28);
                    float _add_1291;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1291) : "f"(_shfl_xor_sync_1292), "f"(_shfl_xor_sync_1293));
                    float _add_1288;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1288) : "f"(_add_1289), "f"(_add_1291));
                    float _add_1287;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1287) : "f"(_add_1288), "f"(sinkhorn_eps));
                    float _fdiv_1286 = __fdiv_rn(C, _add_1287);
                    C = _fdiv_1286;
                    #pragma unroll 1
                    for (unsigned int iteration = 1; iteration < num_sinkhorn_iters; iteration++) {
                        float _shfl_xor_sync_1298 = __shfl_xor_sync(0xFFFFFFFF, C, 1);
                        float _add_1297;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1297) : "f"(C), "f"(_shfl_xor_sync_1298));
                        float _shfl_xor_sync_1300 = __shfl_xor_sync(0xFFFFFFFF, C, 2);
                        float _shfl_xor_sync_1301 = __shfl_xor_sync(0xFFFFFFFF, C, 3);
                        float _add_1299;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1299) : "f"(_shfl_xor_sync_1300), "f"(_shfl_xor_sync_1301));
                        float _add_1296;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1296) : "f"(_add_1297), "f"(_add_1299));
                        float _add_1295;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1295) : "f"(_add_1296), "f"(sinkhorn_eps));
                        float _fdiv_1294 = __fdiv_rn(C, _add_1295);
                        C = _fdiv_1294;
                        float _shfl_xor_sync_1306 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                        float _add_1305;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1305) : "f"(C), "f"(_shfl_xor_sync_1306));
                        float _shfl_xor_sync_1308 = __shfl_xor_sync(0xFFFFFFFF, C, 24);
                        float _shfl_xor_sync_1309 = __shfl_xor_sync(0xFFFFFFFF, C, 28);
                        float _add_1307;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1307) : "f"(_shfl_xor_sync_1308), "f"(_shfl_xor_sync_1309));
                        float _add_1304;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1304) : "f"(_add_1305), "f"(_add_1307));
                        float _add_1303;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1303) : "f"(_add_1304), "f"(sinkhorn_eps));
                        float _fdiv_1302 = __fdiv_rn(C, _add_1303);
                        C = _fdiv_1302;
                    }
                    if (lane_in_warp >= 8) {
                        if (lane_in_warp < 24) {
                            new_comb_res_mix[(unsigned long long)token * 16 + (unsigned long long)(lane_in_warp - 8)] = C;
                        }
                    } else if (lane_in_warp >= 4) {
                        new_post_mix[(unsigned long long)token * 4 + (unsigned long long)(lane_in_warp - 4)] = _mul_0;
                    }
                    __syncwarp();
                }
#endif
            }
        }
    }

    // Kernel teardown ops
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 80;");
    mbarrier_wait(weight_full_addr, 0);
    unsigned int shard = (unsigned int)bid % 16;
    unsigned int local_tokens = ((num_tokens) < (num_ctas * 3) ? (num_tokens) : (num_ctas * 3));
    bool y_fit = (num_tokens + 63) / 64 <= 64;
    unsigned int ticket = 0;
    unsigned int token_1 = 0;
    #pragma unroll 1
    for (unsigned int attempt = 0; attempt < num_tokens + 1; attempt++) {
        if (warp % 4 == 0) {
            if (lane == 0) {
                unsigned int _atomic_fetch_add_0 = atomicAdd(queue, 1);
                ticket = (unsigned int)bid + _atomic_fetch_add_0 * num_ctas;
                if (ticket >= local_tokens) {
                    ticket = (unsigned int)num_tokens;
#if __CUDA_ARCH__ == 1070
                    unsigned int _atomic_fetch_add_1 = atomicAdd(&exhausted[shard], 0);
#else
                    unsigned int _atomic_fetch_add_1 = atomicAdd(&exhausted[0], 0);
#endif
                    if (_atomic_fetch_add_1 == 0) {
                        unsigned int _atomic_fetch_add_2 = atomicAdd(reinterpret_cast<unsigned int*>(split_barriers + (((unsigned long long)(unsigned int)(launch_epochs[bid] % 2) * 16 + (unsigned long long)shard) * 16)), 1);
                        ticket = local_tokens + _atomic_fetch_add_2 * 16 + shard;
                        if (ticket >= num_tokens) {
#if __CUDA_ARCH__ == 1070
                            atomicMax(&exhausted[shard], 1);
#else
                            atomicMax(&exhausted[0], 1);
#endif
                        }
                    }
                }
            }
            unsigned int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, ticket, 0);
            ticket = _shfl_sync_3;
            token_1 = (unsigned int)ticket;
            if (lane_in_warp == 0) {
                tail_control[24 + warp / 4] = token_1;
            }
            if (token_1 < num_tokens) {
                unsigned long long target_1 = (launch_epochs[bid] + 1) * 128 + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + token_1 / 64) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target_1)) >= static_cast<unsigned long long>(1));
                }
            }
        }
        asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
        token_1 = tail_control[24 + warp / 4];
        if (token_1 < num_tokens) {
            unsigned int early[32];
            unsigned long long early_off = (unsigned long long)token_1 * 16384 + (unsigned long long)(warp % 4 * 4 * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(new_residual + early_off + 0);
                uint4* _vdst_4 = reinterpret_cast<uint4*>(&early[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(new_residual + (early_off + 4096) + 0);
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&early[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(new_residual + (early_off + 8192) + 0);
                uint4* _vdst_6 = reinterpret_cast<uint4*>(&early[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(new_residual + (early_off + 12288) + 0);
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&early[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            unsigned long long early_off_0 = (unsigned long long)token_1 * 16384 + (unsigned long long)((warp % 4 * 4 + 1) * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(new_residual + early_off_0 + 0);
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&early[16]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(new_residual + (early_off_0 + 4096) + 0);
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&early[20]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(new_residual + (early_off_0 + 8192) + 0);
                uint4* _vdst_10 = reinterpret_cast<uint4*>(&early[24]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(new_residual + (early_off_0 + 12288) + 0);
                uint4* _vdst_11 = reinterpret_cast<uint4*>(&early[28]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            unsigned int first_hidden_pack = warp % 4 * 4;
            unsigned int weight_cache[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&weight_cache[0])), "=r"(*reinterpret_cast<uint32_t*>(&weight_cache[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weight_cache[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weight_cache[(0) + 3]))
                : "r"(weight_words_addr + (first_hidden_pack * 512 + lane_in_warp * 16)));
            if (warp % 4 == 0) {
                unsigned int mb_2 = token_1 / 64;
                float sc_1 = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs_1 = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
#if __CUDA_ARCH__ == 1070
                unsigned int row_11 = token_1 % 64;
                unsigned int first_task_2 = mb_2 * num_splits;
                float sums_2[5];
                unsigned int split_2 = lane_in_warp * 2;
                unsigned int safe_split_2 = ((split_2 < num_splits) ? split_2 : 0);
                unsigned long long record_2 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_2) * 1536 + (unsigned long long)(row_11 * 24);
                float vec_load[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_2 + 0);
                    vec_load[0 + 0] = _v4.x;
                    vec_load[0 + 1] = _v4.y;
                    vec_load[0 + 2] = _v4.z;
                    vec_load[0 + 3] = _v4.w;
#else
                float s_179 = 0.0f;
                float q_3 = 0.0f;
                if (num_splits > 22) {
                    unsigned int row_14 = token_1 % 64;
                    unsigned int first_task_5 = mb_2 * num_splits;
                    float sums_5[5];
                    float errs_2[5];
                    unsigned int split_5 = lane_in_warp * 2;
                    unsigned int safe_split_5 = ((split_5 < num_splits) ? split_5 : 0);
                    unsigned long long record_5 = ((unsigned long long)first_task_5 + (unsigned long long)safe_split_5) * 1536 + (unsigned long long)(row_14 * 24);
                    float _vec_load_48[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5 + 0);
                        _vec_load_48[0 + 0] = _v4.x;
                        _vec_load_48[0 + 1] = _v4.y;
                        _vec_load_48[0 + 2] = _v4.z;
                        _vec_load_48[0 + 3] = _v4.w;
                    }
                    float value_3_0 = ((split_5 < num_splits) ? _vec_load_48[0] : 0.0f);
                    sums_5[0] = value_3_0;
                    float value_3_1 = ((split_5 < num_splits) ? _vec_load_48[1] : 0.0f);
                    sums_5[1] = value_3_1;
                    float value_3_2 = ((split_5 < num_splits) ? _vec_load_48[2] : 0.0f);
                    sums_5[2] = value_3_2;
                    float value_3_3 = ((split_5 < num_splits) ? _vec_load_48[3] : 0.0f);
                    sums_5[3] = value_3_3;
                    float q_word_5 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_5 * 64 + (unsigned long long)(safe_split_5 * 64) + (unsigned long long)row_14];
                    float q_value_5 = ((split_5 < num_splits) ? q_word_5 : 0.0f);
                    sums_5[4] = q_value_5;
                    unsigned int split_3_1 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_4_1 = ((split_3_1 < num_splits) ? split_3_1 : 0);
                    unsigned long long record_5_1 = ((unsigned long long)first_task_5 + (unsigned long long)safe_split_4_1) * 1536 + (unsigned long long)(row_14 * 24);
                    float _vec_load_49[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5_1 + 0);
                        _vec_load_49[0 + 0] = _v4.x;
                        _vec_load_49[0 + 1] = _v4.y;
                        _vec_load_49[0 + 2] = _v4.z;
                        _vec_load_49[0 + 3] = _v4.w;
                    }
                    float value_3_4 = ((split_3_1 < num_splits) ? _vec_load_49[0] : 0.0f);
                    float s_179_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_179_1) : "f"(sums_5[0]), "f"(value_3_4));
                    float bb_177;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_177) : "f"(s_179_1), "f"(sums_5[0]));
                    float _sub_15_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_2) : "f"(s_179_1), "f"(bb_177));
                    float _sub_14_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_2) : "f"(sums_5[0]), "f"(_sub_15_2));
                    float _sub_16_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16_1) : "f"(value_3_4), "f"(bb_177));
                    float e_177;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_177) : "f"(_sub_14_2), "f"(_sub_16_1));
                    sums_5[0] = s_179_1;
                    errs_2[0] = e_177;
                    float value_3_5 = ((split_3_1 < num_splits) ? _vec_load_49[1] : 0.0f);
                    float s_180;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_180) : "f"(sums_5[1]), "f"(value_3_5));
                    float bb_177_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_177_1) : "f"(s_180), "f"(sums_5[1]));
                    float _sub_18_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_2) : "f"(s_180), "f"(bb_177_1));
                    float _sub_17_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_2) : "f"(sums_5[1]), "f"(_sub_18_2));
                    float _sub_19_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_3) : "f"(value_3_5), "f"(bb_177_1));
                    float e_177_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_177_1) : "f"(_sub_17_2), "f"(_sub_19_3));
                    sums_5[1] = s_180;
                    errs_2[1] = e_177_1;
                    float value_3_6 = ((split_3_1 < num_splits) ? _vec_load_49[2] : 0.0f);
                    float s_181;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_181) : "f"(sums_5[2]), "f"(value_3_6));
                    float bb_178;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_178) : "f"(s_181), "f"(sums_5[2]));
                    float _sub_21_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_2) : "f"(s_181), "f"(bb_178));
                    float _sub_20_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_3) : "f"(sums_5[2]), "f"(_sub_21_2));
                    float _sub_22_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_2) : "f"(value_3_6), "f"(bb_178));
                    float e_178;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_178) : "f"(_sub_20_3), "f"(_sub_22_2));
                    sums_5[2] = s_181;
                    errs_2[2] = e_178;
                    float value_3_7 = ((split_3_1 < num_splits) ? _vec_load_49[3] : 0.0f);
                    float s_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_182) : "f"(sums_5[3]), "f"(value_3_7));
                    float bb_179;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_179) : "f"(s_182), "f"(sums_5[3]));
                    float _sub_24_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_2) : "f"(s_182), "f"(bb_179));
                    float _sub_23_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_2) : "f"(sums_5[3]), "f"(_sub_24_2));
                    float _sub_25_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_2) : "f"(value_3_7), "f"(bb_179));
                    float e_179;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_179) : "f"(_sub_23_2), "f"(_sub_25_2));
                    sums_5[3] = s_182;
                    errs_2[3] = e_179;
                    float q_word_20 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_5 * 64 + (unsigned long long)(safe_split_4_1 * 64) + (unsigned long long)row_14];
                    float q_value_21 = ((split_3_1 < num_splits) ? q_word_20 : 0.0f);
                    float s_183;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_183) : "f"(sums_5[4]), "f"(q_value_21));
                    float bb_180;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_180) : "f"(s_183), "f"(sums_5[4]));
                    float _sub_27_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_3) : "f"(s_183), "f"(bb_180));
                    float _sub_26_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_3) : "f"(sums_5[4]), "f"(_sub_27_3));
                    float _sub_28_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_2) : "f"(q_value_21), "f"(bb_180));
                    float e_180;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_180) : "f"(_sub_26_3), "f"(_sub_28_2));
                    sums_5[4] = s_183;
                    errs_2[4] = e_180;
                    float partner_1_125 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 1);
                    float partner_err_1_125 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 1);
                    float s_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_184) : "f"(sums_5[0]), "f"(partner_1_125));
                    float bb_181;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_181) : "f"(s_184), "f"(sums_5[0]));
                    float _sub_30_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_2) : "f"(s_184), "f"(bb_181));
                    float _sub_29_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29_1) : "f"(sums_5[0]), "f"(_sub_30_2));
                    float _sub_31_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_2) : "f"(partner_1_125), "f"(bb_181));
                    float e_181;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_181) : "f"(_sub_29_1), "f"(_sub_31_2));
                    sums_5[0] = s_184;
                    float _add_33;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(errs_2[0]), "f"(partner_err_1_125));
                    float _add_32_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(_add_33), "f"(e_181));
                    errs_2[0] = _add_32_1;
                    float partner_1_126 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 1);
                    float partner_err_1_126 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 1);
                    float s_185;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_185) : "f"(sums_5[1]), "f"(partner_1_126));
                    float bb_182;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_182) : "f"(s_185), "f"(sums_5[1]));
                    float _sub_35_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35_2) : "f"(s_185), "f"(bb_182));
                    float _sub_34_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_3) : "f"(sums_5[1]), "f"(_sub_35_2));
                    float _sub_36_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36_2) : "f"(partner_1_126), "f"(bb_182));
                    float e_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_182) : "f"(_sub_34_3), "f"(_sub_36_2));
                    sums_5[1] = s_185;
                    float _add_38;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(errs_2[1]), "f"(partner_err_1_126));
                    float _add_37;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(e_182));
                    errs_2[1] = _add_37;
                    float partner_1_127 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 1);
                    float partner_err_1_127 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 1);
                    float s_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_186) : "f"(sums_5[2]), "f"(partner_1_127));
                    float bb_183;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_183) : "f"(s_186), "f"(sums_5[2]));
                    float _sub_40_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40_3) : "f"(s_186), "f"(bb_183));
                    float _sub_39_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39_2) : "f"(sums_5[2]), "f"(_sub_40_3));
                    float _sub_41_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41_3) : "f"(partner_1_127), "f"(bb_183));
                    float e_183;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_183) : "f"(_sub_39_2), "f"(_sub_41_3));
                    sums_5[2] = s_186;
                    float _add_43;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(errs_2[2]), "f"(partner_err_1_127));
                    float _add_42;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_43), "f"(e_183));
                    errs_2[2] = _add_42;
                    float partner_1_128 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 1);
                    float partner_err_1_128 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 1);
                    float s_187;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_187) : "f"(sums_5[3]), "f"(partner_1_128));
                    float bb_184;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_184) : "f"(s_187), "f"(sums_5[3]));
                    float _sub_45_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45_2) : "f"(s_187), "f"(bb_184));
                    float _sub_44_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44_2) : "f"(sums_5[3]), "f"(_sub_45_2));
                    float _sub_46_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46_2) : "f"(partner_1_128), "f"(bb_184));
                    float e_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_184) : "f"(_sub_44_2), "f"(_sub_46_2));
                    sums_5[3] = s_187;
                    float _add_48_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48_1) : "f"(errs_2[3]), "f"(partner_err_1_128));
                    float _add_47;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_add_48_1), "f"(e_184));
                    errs_2[3] = _add_47;
                    float partner_1_129 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 1);
                    float partner_err_1_129 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 1);
                    float s_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_188) : "f"(sums_5[4]), "f"(partner_1_129));
                    float bb_185;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_185) : "f"(s_188), "f"(sums_5[4]));
                    float _sub_50_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50_2) : "f"(s_188), "f"(bb_185));
                    float _sub_49_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49_1) : "f"(sums_5[4]), "f"(_sub_50_2));
                    float _sub_51_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51_2) : "f"(partner_1_129), "f"(bb_185));
                    float e_185;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_185) : "f"(_sub_49_1), "f"(_sub_51_2));
                    sums_5[4] = s_188;
                    float _add_53_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_2) : "f"(errs_2[4]), "f"(partner_err_1_129));
                    float _add_52;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53_2), "f"(e_185));
                    errs_2[4] = _add_52;
                    float partner_1_130 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 2);
                    float partner_err_1_130 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 2);
                    float s_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_189) : "f"(sums_5[0]), "f"(partner_1_130));
                    float bb_186;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_186) : "f"(s_189), "f"(sums_5[0]));
                    float _sub_55_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55_3) : "f"(s_189), "f"(bb_186));
                    float _sub_54_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54_2) : "f"(sums_5[0]), "f"(_sub_55_3));
                    float _sub_56_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56_2) : "f"(partner_1_130), "f"(bb_186));
                    float e_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_186) : "f"(_sub_54_2), "f"(_sub_56_2));
                    sums_5[0] = s_189;
                    float _add_58_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58_1) : "f"(errs_2[0]), "f"(partner_err_1_130));
                    float _add_57;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_58_1), "f"(e_186));
                    errs_2[0] = _add_57;
                    float partner_1_131 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 2);
                    float partner_err_1_131 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 2);
                    float s_190;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_190) : "f"(sums_5[1]), "f"(partner_1_131));
                    float bb_187;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_187) : "f"(s_190), "f"(sums_5[1]));
                    float _sub_60_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60_2) : "f"(s_190), "f"(bb_187));
                    float _sub_59_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59_1) : "f"(sums_5[1]), "f"(_sub_60_2));
                    float _sub_61_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61_2) : "f"(partner_1_131), "f"(bb_187));
                    float e_187;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_187) : "f"(_sub_59_1), "f"(_sub_61_2));
                    sums_5[1] = s_190;
                    float _add_63_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63_1) : "f"(errs_2[1]), "f"(partner_err_1_131));
                    float _add_62;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_add_63_1), "f"(e_187));
                    errs_2[1] = _add_62;
                    float partner_1_132 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 2);
                    float partner_err_1_132 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 2);
                    float s_191;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_191) : "f"(sums_5[2]), "f"(partner_1_132));
                    float bb_188;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_188) : "f"(s_191), "f"(sums_5[2]));
                    float _sub_65_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65_2) : "f"(s_191), "f"(bb_188));
                    float _sub_64_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64_1) : "f"(sums_5[2]), "f"(_sub_65_2));
                    float _sub_66_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66_2) : "f"(partner_1_132), "f"(bb_188));
                    float e_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_188) : "f"(_sub_64_1), "f"(_sub_66_2));
                    sums_5[2] = s_191;
                    float _add_68_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68_1) : "f"(errs_2[2]), "f"(partner_err_1_132));
                    float _add_67;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_68_1), "f"(e_188));
                    errs_2[2] = _add_67;
                    float partner_1_133 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 2);
                    float partner_err_1_133 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 2);
                    float s_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_192) : "f"(sums_5[3]), "f"(partner_1_133));
                    float bb_189;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_189) : "f"(s_192), "f"(sums_5[3]));
                    float _sub_70_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70_2) : "f"(s_192), "f"(bb_189));
                    float _sub_69_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69_1) : "f"(sums_5[3]), "f"(_sub_70_2));
                    float _sub_71_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71_2) : "f"(partner_1_133), "f"(bb_189));
                    float e_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_189) : "f"(_sub_69_1), "f"(_sub_71_2));
                    sums_5[3] = s_192;
                    float _add_73_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73_1) : "f"(errs_2[3]), "f"(partner_err_1_133));
                    float _add_72_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72_1) : "f"(_add_73_1), "f"(e_189));
                    errs_2[3] = _add_72_1;
                    float partner_1_134 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 2);
                    float partner_err_1_134 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 2);
                    float s_193;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_193) : "f"(sums_5[4]), "f"(partner_1_134));
                    float bb_190;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_190) : "f"(s_193), "f"(sums_5[4]));
                    float _sub_75_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75_1) : "f"(s_193), "f"(bb_190));
                    float _sub_74;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(sums_5[4]), "f"(_sub_75_1));
                    float _sub_76_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76_1) : "f"(partner_1_134), "f"(bb_190));
                    float e_190;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_190) : "f"(_sub_74), "f"(_sub_76_1));
                    sums_5[4] = s_193;
                    float _add_78_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78_1) : "f"(errs_2[4]), "f"(partner_err_1_134));
                    float _add_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78_1), "f"(e_190));
                    errs_2[4] = _add_77;
                    float partner_1_135 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 4);
                    float partner_err_1_135 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 4);
                    float s_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_194) : "f"(sums_5[0]), "f"(partner_1_135));
                    float bb_191;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_191) : "f"(s_194), "f"(sums_5[0]));
                    float _sub_80_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80_1) : "f"(s_194), "f"(bb_191));
                    float _sub_79;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(sums_5[0]), "f"(_sub_80_1));
                    float _sub_81_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81_1) : "f"(partner_1_135), "f"(bb_191));
                    float e_191;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_191) : "f"(_sub_79), "f"(_sub_81_1));
                    sums_5[0] = s_194;
                    float _add_83_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83_1) : "f"(errs_2[0]), "f"(partner_err_1_135));
                    float _add_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_83_1), "f"(e_191));
                    errs_2[0] = _add_82;
                    float partner_1_136 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 4);
                    float partner_err_1_136 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 4);
                    float s_195;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_195) : "f"(sums_5[1]), "f"(partner_1_136));
                    float bb_192;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_192) : "f"(s_195), "f"(sums_5[1]));
                    float _sub_85_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85_1) : "f"(s_195), "f"(bb_192));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(sums_5[1]), "f"(_sub_85_1));
                    float _sub_86_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86_1) : "f"(partner_1_136), "f"(bb_192));
                    float e_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_192) : "f"(_sub_84), "f"(_sub_86_1));
                    sums_5[1] = s_195;
                    float _add_88_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88_1) : "f"(errs_2[1]), "f"(partner_err_1_136));
                    float _add_87;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88_1), "f"(e_192));
                    errs_2[1] = _add_87;
                    float partner_1_137 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 4);
                    float partner_err_1_137 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 4);
                    float s_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_196) : "f"(sums_5[2]), "f"(partner_1_137));
                    float bb_193;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_193) : "f"(s_196), "f"(sums_5[2]));
                    float _sub_90_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90_1) : "f"(s_196), "f"(bb_193));
                    float _sub_89;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(sums_5[2]), "f"(_sub_90_1));
                    float _sub_91_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91_1) : "f"(partner_1_137), "f"(bb_193));
                    float e_193;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_193) : "f"(_sub_89), "f"(_sub_91_1));
                    sums_5[2] = s_196;
                    float _add_93_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93_1) : "f"(errs_2[2]), "f"(partner_err_1_137));
                    float _add_92;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93_1), "f"(e_193));
                    errs_2[2] = _add_92;
                    float partner_1_138 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 4);
                    float partner_err_1_138 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 4);
                    float s_197;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_197) : "f"(sums_5[3]), "f"(partner_1_138));
                    float bb_194;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_194) : "f"(s_197), "f"(sums_5[3]));
                    float _sub_95_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95_1) : "f"(s_197), "f"(bb_194));
                    float _sub_94;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(sums_5[3]), "f"(_sub_95_1));
                    float _sub_96_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96_1) : "f"(partner_1_138), "f"(bb_194));
                    float e_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_194) : "f"(_sub_94), "f"(_sub_96_1));
                    sums_5[3] = s_197;
                    float _add_98_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98_1) : "f"(errs_2[3]), "f"(partner_err_1_138));
                    float _add_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_98_1), "f"(e_194));
                    errs_2[3] = _add_97;
                    float partner_1_139 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 4);
                    float partner_err_1_139 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 4);
                    float s_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_198) : "f"(sums_5[4]), "f"(partner_1_139));
                    float bb_195;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_195) : "f"(s_198), "f"(sums_5[4]));
                    float _sub_100_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100_1) : "f"(s_198), "f"(bb_195));
                    float _sub_99;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(sums_5[4]), "f"(_sub_100_1));
                    float _sub_101_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101_1) : "f"(partner_1_139), "f"(bb_195));
                    float e_195;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_195) : "f"(_sub_99), "f"(_sub_101_1));
                    sums_5[4] = s_198;
                    float _add_103_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103_1) : "f"(errs_2[4]), "f"(partner_err_1_139));
                    float _add_102;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_add_103_1), "f"(e_195));
                    errs_2[4] = _add_102;
                    float partner_1_140 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 8);
                    float partner_err_1_140 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 8);
                    float s_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_199) : "f"(sums_5[0]), "f"(partner_1_140));
                    float bb_196;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_196) : "f"(s_199), "f"(sums_5[0]));
                    float _sub_105_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105_1) : "f"(s_199), "f"(bb_196));
                    float _sub_104;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(sums_5[0]), "f"(_sub_105_1));
                    float _sub_106_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106_1) : "f"(partner_1_140), "f"(bb_196));
                    float e_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_196) : "f"(_sub_104), "f"(_sub_106_1));
                    sums_5[0] = s_199;
                    float _add_108_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108_1) : "f"(errs_2[0]), "f"(partner_err_1_140));
                    float _add_107;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_add_108_1), "f"(e_196));
                    errs_2[0] = _add_107;
                    float partner_1_141 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 8);
                    float partner_err_1_141 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 8);
                    float s_200;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_200) : "f"(sums_5[1]), "f"(partner_1_141));
                    float bb_197;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_197) : "f"(s_200), "f"(sums_5[1]));
                    float _sub_110_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_110_1) : "f"(s_200), "f"(bb_197));
                    float _sub_109;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_109) : "f"(sums_5[1]), "f"(_sub_110_1));
                    float _sub_111_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111_1) : "f"(partner_1_141), "f"(bb_197));
                    float e_197;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_197) : "f"(_sub_109), "f"(_sub_111_1));
                    sums_5[1] = s_200;
                    float _add_113_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113_1) : "f"(errs_2[1]), "f"(partner_err_1_141));
                    float _add_112;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113_1), "f"(e_197));
                    errs_2[1] = _add_112;
                    float partner_1_142 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 8);
                    float partner_err_1_142 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 8);
                    float s_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_201) : "f"(sums_5[2]), "f"(partner_1_142));
                    float bb_198;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_198) : "f"(s_201), "f"(sums_5[2]));
                    float _sub_115_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115_1) : "f"(s_201), "f"(bb_198));
                    float _sub_114;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(sums_5[2]), "f"(_sub_115_1));
                    float _sub_116_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_116_1) : "f"(partner_1_142), "f"(bb_198));
                    float e_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_198) : "f"(_sub_114), "f"(_sub_116_1));
                    sums_5[2] = s_201;
                    float _add_118_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118_1) : "f"(errs_2[2]), "f"(partner_err_1_142));
                    float _add_117;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_add_118_1), "f"(e_198));
                    errs_2[2] = _add_117;
                    float partner_1_143 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 8);
                    float partner_err_1_143 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 8);
                    float s_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_202) : "f"(sums_5[3]), "f"(partner_1_143));
                    float bb_199;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_199) : "f"(s_202), "f"(sums_5[3]));
                    float _sub_120_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120_1) : "f"(s_202), "f"(bb_199));
                    float _sub_119;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(sums_5[3]), "f"(_sub_120_1));
                    float _sub_121_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121_1) : "f"(partner_1_143), "f"(bb_199));
                    float e_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_199) : "f"(_sub_119), "f"(_sub_121_1));
                    sums_5[3] = s_202;
                    float _add_123_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123_1) : "f"(errs_2[3]), "f"(partner_err_1_143));
                    float _add_122;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_add_123_1), "f"(e_199));
                    errs_2[3] = _add_122;
                    float partner_1_144 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 8);
                    float partner_err_1_144 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 8);
                    float s_203;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_203) : "f"(sums_5[4]), "f"(partner_1_144));
                    float bb_200;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_200) : "f"(s_203), "f"(sums_5[4]));
                    float _sub_125;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(s_203), "f"(bb_200));
                    float _sub_124;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_124) : "f"(sums_5[4]), "f"(_sub_125));
                    float _sub_126;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(partner_1_144), "f"(bb_200));
                    float e_200;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_200) : "f"(_sub_124), "f"(_sub_126));
                    sums_5[4] = s_203;
                    float _add_128_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128_1) : "f"(errs_2[4]), "f"(partner_err_1_144));
                    float _add_127;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_add_128_1), "f"(e_200));
                    errs_2[4] = _add_127;
                    float partner_1_145 = __shfl_xor_sync(0xFFFFFFFF, sums_5[0], 16);
                    float partner_err_1_145 = __shfl_xor_sync(0xFFFFFFFF, errs_2[0], 16);
                    float s_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_204) : "f"(sums_5[0]), "f"(partner_1_145));
                    float bb_201;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_201) : "f"(s_204), "f"(sums_5[0]));
                    float _sub_130;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_130) : "f"(s_204), "f"(bb_201));
                    float _sub_129;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(sums_5[0]), "f"(_sub_130));
                    float _sub_131;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(partner_1_145), "f"(bb_201));
                    float e_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_201) : "f"(_sub_129), "f"(_sub_131));
                    sums_5[0] = s_204;
                    float _add_133_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133_1) : "f"(errs_2[0]), "f"(partner_err_1_145));
                    float _add_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_add_133_1), "f"(e_201));
                    errs_2[0] = _add_132;
                    float partner_1_146 = __shfl_xor_sync(0xFFFFFFFF, sums_5[1], 16);
                    float partner_err_1_146 = __shfl_xor_sync(0xFFFFFFFF, errs_2[1], 16);
                    float s_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_205) : "f"(sums_5[1]), "f"(partner_1_146));
                    float bb_202;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_202) : "f"(s_205), "f"(sums_5[1]));
                    float _sub_135;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_135) : "f"(s_205), "f"(bb_202));
                    float _sub_134;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_134) : "f"(sums_5[1]), "f"(_sub_135));
                    float _sub_136;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(partner_1_146), "f"(bb_202));
                    float e_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_202) : "f"(_sub_134), "f"(_sub_136));
                    sums_5[1] = s_205;
                    float _add_138_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138_1) : "f"(errs_2[1]), "f"(partner_err_1_146));
                    float _add_137;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_add_138_1), "f"(e_202));
                    errs_2[1] = _add_137;
                    float partner_1_147 = __shfl_xor_sync(0xFFFFFFFF, sums_5[2], 16);
                    float partner_err_1_147 = __shfl_xor_sync(0xFFFFFFFF, errs_2[2], 16);
                    float s_206;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_206) : "f"(sums_5[2]), "f"(partner_1_147));
                    float bb_203;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_203) : "f"(s_206), "f"(sums_5[2]));
                    float _sub_140;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_140) : "f"(s_206), "f"(bb_203));
                    float _sub_139;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_139) : "f"(sums_5[2]), "f"(_sub_140));
                    float _sub_141;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_141) : "f"(partner_1_147), "f"(bb_203));
                    float e_203;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_203) : "f"(_sub_139), "f"(_sub_141));
                    sums_5[2] = s_206;
                    float _add_143_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143_1) : "f"(errs_2[2]), "f"(partner_err_1_147));
                    float _add_142_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142_1) : "f"(_add_143_1), "f"(e_203));
                    errs_2[2] = _add_142_1;
                    float partner_1_148 = __shfl_xor_sync(0xFFFFFFFF, sums_5[3], 16);
                    float partner_err_1_148 = __shfl_xor_sync(0xFFFFFFFF, errs_2[3], 16);
                    float s_207;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_207) : "f"(sums_5[3]), "f"(partner_1_148));
                    float bb_204;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_204) : "f"(s_207), "f"(sums_5[3]));
                    float _sub_145;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(s_207), "f"(bb_204));
                    float _sub_144;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_144) : "f"(sums_5[3]), "f"(_sub_145));
                    float _sub_146;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_146) : "f"(partner_1_148), "f"(bb_204));
                    float e_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_204) : "f"(_sub_144), "f"(_sub_146));
                    sums_5[3] = s_207;
                    float _add_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(errs_2[3]), "f"(partner_err_1_148));
                    float _add_147_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147_1) : "f"(_add_148), "f"(e_204));
                    errs_2[3] = _add_147_1;
                    float partner_1_149 = __shfl_xor_sync(0xFFFFFFFF, sums_5[4], 16);
                    float partner_err_1_149 = __shfl_xor_sync(0xFFFFFFFF, errs_2[4], 16);
                    float s_208;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_208) : "f"(sums_5[4]), "f"(partner_1_149));
                    float bb_205;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bb_205) : "f"(s_208), "f"(sums_5[4]));
                    float _sub_150;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(s_208), "f"(bb_205));
                    float _sub_149;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(sums_5[4]), "f"(_sub_150));
                    float _sub_151;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_151) : "f"(partner_1_149), "f"(bb_205));
                    float e_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_205) : "f"(_sub_149), "f"(_sub_151));
                    sums_5[4] = s_208;
                    float _add_153_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153_1) : "f"(errs_2[4]), "f"(partner_err_1_149));
                    float _add_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_add_153_1), "f"(e_205));
                    errs_2[4] = _add_152;
                    float s_148_1 = 0.0f;
                    float total_n_1_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_24) : "f"(sums_5[0]), "f"(errs_2[0]));
                    s_148_1 = ((lane_in_warp == 0) ? total_n_1_24 : s_148_1);
                    float total_n_1_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_25) : "f"(sums_5[1]), "f"(errs_2[1]));
                    s_148_1 = ((lane_in_warp == 1) ? total_n_1_25 : s_148_1);
                    float total_n_1_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_26) : "f"(sums_5[2]), "f"(errs_2[2]));
                    s_148_1 = ((lane_in_warp == 2) ? total_n_1_26 : s_148_1);
                    float total_n_1_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(total_n_1_27) : "f"(sums_5[3]), "f"(errs_2[3]));
                    s_148_1 = ((lane_in_warp == 3) ? total_n_1_27 : s_148_1);
                    float q_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(q_152) : "f"(sums_5[4]), "f"(errs_2[4]));
                    s_179 = s_148_1;
                    q_3 = q_152;
                } else {
                    unsigned int row_15 = token_1 % 64;
                    unsigned int first_task_6 = mb_2 * num_splits;
                    float sums_6[5];
                    unsigned int split_6 = lane_in_warp * 2;
                    unsigned int safe_split_6 = ((split_6 < num_splits) ? split_6 : 0);
                    unsigned long long record_6 = ((unsigned long long)first_task_6 + (unsigned long long)safe_split_6) * 1536 + (unsigned long long)(row_15 * 24);
                    float vec_load[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_6 + 0);
                        vec_load[0 + 0] = _v4.x;
                        vec_load[0 + 1] = _v4.y;
                        vec_load[0 + 2] = _v4.z;
                        vec_load[0 + 3] = _v4.w;
                    }
                    float value_3_8 = ((split_6 < num_splits) ? vec_load[0] : 0.0f);
                    sums_6[0] = value_3_8;
                    float value_3_9 = ((split_6 < num_splits) ? vec_load[1] : 0.0f);
                    sums_6[1] = value_3_9;
                    float value_3_10 = ((split_6 < num_splits) ? vec_load[2] : 0.0f);
                    sums_6[2] = value_3_10;
                    float value_3_11 = ((split_6 < num_splits) ? vec_load[3] : 0.0f);
                    sums_6[3] = value_3_11;
                    float q_word_6 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_6 * 64 + (unsigned long long)(safe_split_6 * 64) + (unsigned long long)row_15];
                    float q_value_6 = ((split_6 < num_splits) ? q_word_6 : 0.0f);
                    sums_6[4] = q_value_6;
                    unsigned int split_3_2 = lane_in_warp * 2 + 1;
                    unsigned int safe_split_4_2 = ((split_3_2 < num_splits) ? split_3_2 : 0);
                    unsigned long long record_5_2 = ((unsigned long long)first_task_6 + (unsigned long long)safe_split_4_2) * 1536 + (unsigned long long)(row_15 * 24);
                    float vec_load_1[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5_2 + 0);
                        vec_load_1[0 + 0] = _v4.x;
                        vec_load_1[0 + 1] = _v4.y;
                        vec_load_1[0 + 2] = _v4.z;
                        vec_load_1[0 + 3] = _v4.w;
                    }
                    float value_3_12 = ((split_3_2 < num_splits) ? vec_load_1[0] : 0.0f);
                    float _add_156;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(sums_6[0]), "f"(value_3_12));
                    sums_6[0] = _add_156;
                    float value_3_13 = ((split_3_2 < num_splits) ? vec_load_1[1] : 0.0f);
                    float _add_157_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157_1) : "f"(sums_6[1]), "f"(value_3_13));
                    sums_6[1] = _add_157_1;
                    float value_3_14 = ((split_3_2 < num_splits) ? vec_load_1[2] : 0.0f);
                    float _add_158;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(sums_6[2]), "f"(value_3_14));
                    sums_6[2] = _add_158;
                    float value_3_15 = ((split_3_2 < num_splits) ? vec_load_1[3] : 0.0f);
                    float _add_159_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159_1) : "f"(sums_6[3]), "f"(value_3_15));
                    sums_6[3] = _add_159_1;
                    float q_word_10 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_6 * 64 + (unsigned long long)(safe_split_4_2 * 64) + (unsigned long long)row_15];
                    float q_value_11 = ((split_3_2 < num_splits) ? q_word_10 : 0.0f);
                    float _add_160;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(sums_6[4]), "f"(q_value_11));
                    sums_6[4] = _add_160;
                    float _shfl_xor_sync_162_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 1);
                    float _add_161_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_162_1));
                    sums_6[0] = _add_161_1;
                    float _shfl_xor_sync_164_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 1);
                    float _add_163_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_164_1));
                    sums_6[1] = _add_163_1;
                    float _shfl_xor_sync_166_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 1);
                    float _add_165_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165_1) : "f"(sums_6[2]), "f"(_shfl_xor_sync_166_1));
                    sums_6[2] = _add_165_1;
                    float _shfl_xor_sync_168_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 1);
                    float _add_167_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_168_1));
                    sums_6[3] = _add_167_1;
                    float _shfl_xor_sync_170_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 1);
                    float _add_169_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_170_1));
                    sums_6[4] = _add_169_1;
                    float _shfl_xor_sync_172_1 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 2);
                    float _add_171_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_172_1));
                    sums_6[0] = _add_171_1;
                    float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 2);
                    float _add_173;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(sums_6[1]), "f"(_shfl_xor_sync_174));
                    sums_6[1] = _add_173;
                    float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 2);
                    float _add_175;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(sums_6[2]), "f"(_shfl_xor_sync_176));
                    sums_6[2] = _add_175;
                    float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 2);
                    float _add_177;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(sums_6[3]), "f"(_shfl_xor_sync_178));
                    sums_6[3] = _add_177;
                    float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 2);
                    float _add_179;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(sums_6[4]), "f"(_shfl_xor_sync_180));
                    sums_6[4] = _add_179;
                    float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 4);
                    float _add_181_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_182));
                    sums_6[0] = _add_181_1;
                    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 4);
                    float _add_183_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_184));
                    sums_6[1] = _add_183_1;
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 4);
                    float _add_185;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(sums_6[2]), "f"(_shfl_xor_sync_186));
                    sums_6[2] = _add_185;
                    float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 4);
                    float _add_187_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_188));
                    sums_6[3] = _add_187_1;
                    float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 4);
                    float _add_189_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_190));
                    sums_6[4] = _add_189_1;
                    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 8);
                    float _add_191_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_192));
                    sums_6[0] = _add_191_1;
                    float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 8);
                    float _add_193_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_194));
                    sums_6[1] = _add_193_1;
                    float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 8);
                    float _add_195;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(sums_6[2]), "f"(_shfl_xor_sync_196));
                    sums_6[2] = _add_195;
                    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 8);
                    float _add_197_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_198));
                    sums_6[3] = _add_197_1;
                    float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 8);
                    float _add_199_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_200));
                    sums_6[4] = _add_199_1;
                    float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, sums_6[0], 16);
                    float _add_201_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201_1) : "f"(sums_6[0]), "f"(_shfl_xor_sync_202));
                    sums_6[0] = _add_201_1;
                    float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, sums_6[1], 16);
                    float _add_203_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203_1) : "f"(sums_6[1]), "f"(_shfl_xor_sync_204));
                    sums_6[1] = _add_203_1;
                    float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, sums_6[2], 16);
                    float _add_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(sums_6[2]), "f"(_shfl_xor_sync_206));
                    sums_6[2] = _add_205;
                    float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, sums_6[3], 16);
                    float _add_207_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207_1) : "f"(sums_6[3]), "f"(_shfl_xor_sync_208));
                    sums_6[3] = _add_207_1;
                    float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, sums_6[4], 16);
                    float _add_209_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209_1) : "f"(sums_6[4]), "f"(_shfl_xor_sync_210));
                    sums_6[4] = _add_209_1;
                    float s_209 = 0.0f;
                    s_209 = ((lane_in_warp == 0) ? sums_6[0] : s_209);
                    s_209 = ((lane_in_warp == 1) ? sums_6[1] : s_209);
                    s_209 = ((lane_in_warp == 2) ? sums_6[2] : s_209);
                    s_209 = ((lane_in_warp == 3) ? sums_6[3] : s_209);
                    float q_0 = sums_6[4];
                    s_179 = s_209;
                    q_3 = q_0;
#endif
                }
#if __CUDA_ARCH__ == 1070
                float value_3_0 = ((split_2 < num_splits) ? vec_load[0] : 0.0f);
                sums_2[0] = value_3_0;
                float value_3_1 = ((split_2 < num_splits) ? vec_load[1] : 0.0f);
                sums_2[1] = value_3_1;
                float value_3_2 = ((split_2 < num_splits) ? vec_load[2] : 0.0f);
                sums_2[2] = value_3_2;
                float value_3_3 = ((split_2 < num_splits) ? vec_load[3] : 0.0f);
                sums_2[3] = value_3_3;
                float q_word_2 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_2 * 64) + (unsigned long long)row_11];
                float q_value_2 = ((split_2 < num_splits) ? q_word_2 : 0.0f);
                sums_2[4] = q_value_2;
                unsigned int split_3 = lane_in_warp * 2 + 1;
                unsigned int safe_split_4 = ((split_3 < num_splits) ? split_3 : 0);
                unsigned long long record_5 = ((unsigned long long)first_task_2 + (unsigned long long)safe_split_4) * 1536 + (unsigned long long)(row_11 * 24);
                float vec_load_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5 + 0);
                    vec_load_1[0 + 0] = _v4.x;
                    vec_load_1[0 + 1] = _v4.y;
                    vec_load_1[0 + 2] = _v4.z;
                    vec_load_1[0 + 3] = _v4.w;
                }
                float value_3_4 = ((split_3 < num_splits) ? vec_load_1[0] : 0.0f);
                float _add_14_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14_1) : "f"(sums_2[0]), "f"(value_3_4));
                sums_2[0] = _add_14_1;
                float value_3_5 = ((split_3 < num_splits) ? vec_load_1[1] : 0.0f);
                float _add_15_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15_1) : "f"(sums_2[1]), "f"(value_3_5));
                sums_2[1] = _add_15_1;
                float value_3_6 = ((split_3 < num_splits) ? vec_load_1[2] : 0.0f);
                float _add_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(sums_2[2]), "f"(value_3_6));
                sums_2[2] = _add_16;
                float value_3_7 = ((split_3 < num_splits) ? vec_load_1[3] : 0.0f);
                float _add_17_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_1) : "f"(sums_2[3]), "f"(value_3_7));
                sums_2[3] = _add_17_1;
                float q_word_10 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_2 * 64 + (unsigned long long)(safe_split_4 * 64) + (unsigned long long)row_11];
                float q_value_11 = ((split_3 < num_splits) ? q_word_10 : 0.0f);
                float _add_18_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_2) : "f"(sums_2[4]), "f"(q_value_11));
                sums_2[4] = _add_18_2;
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 1);
                float _add_19_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_20));
                sums_2[0] = _add_19_1;
                float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 1);
                float _add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(sums_2[1]), "f"(_shfl_xor_sync_22));
                sums_2[1] = _add_21;
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 1);
                float _add_23_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_24));
                sums_2[2] = _add_23_1;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 1);
                float _add_25_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_2) : "f"(sums_2[3]), "f"(_shfl_xor_sync_26));
                sums_2[3] = _add_25_2;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 1);
                float _add_27_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_28));
                sums_2[4] = _add_27_1;
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 2);
                float _add_29_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_30));
                sums_2[0] = _add_29_1;
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 2);
                float _add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(sums_2[1]), "f"(_shfl_xor_sync_32));
                sums_2[1] = _add_31;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 2);
                float _add_33_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_34));
                sums_2[2] = _add_33_1;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 2);
                float _add_35_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_36));
                sums_2[3] = _add_35_1;
                float _shfl_xor_sync_38_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 2);
                float _add_37_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_38_1));
                sums_2[4] = _add_37_1;
                float _shfl_xor_sync_40_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 4);
                float _add_39_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_2) : "f"(sums_2[0]), "f"(_shfl_xor_sync_40_1));
                sums_2[0] = _add_39_2;
                float _shfl_xor_sync_42_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 4);
                float _add_41_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41_1) : "f"(sums_2[1]), "f"(_shfl_xor_sync_42_1));
                sums_2[1] = _add_41_1;
                float _shfl_xor_sync_44_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 4);
                float _add_43_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_44_1));
                sums_2[2] = _add_43_1;
                float _shfl_xor_sync_46_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 4);
                float _add_45_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_46_1));
                sums_2[3] = _add_45_1;
                float _shfl_xor_sync_48_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 4);
                float _add_47_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_48_1));
                sums_2[4] = _add_47_1;
                float _shfl_xor_sync_50_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 8);
                float _add_49_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_50_1));
                sums_2[0] = _add_49_1;
                float _shfl_xor_sync_52_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 8);
                float _add_51_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51_1) : "f"(sums_2[1]), "f"(_shfl_xor_sync_52_1));
                sums_2[1] = _add_51_1;
                float _shfl_xor_sync_54_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 8);
                float _add_53_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_2) : "f"(sums_2[2]), "f"(_shfl_xor_sync_54_1));
                sums_2[2] = _add_53_2;
                float _shfl_xor_sync_56_2 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 8);
                float _add_55_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_56_2));
                sums_2[3] = _add_55_1;
                float _shfl_xor_sync_58_2 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 8);
                float _add_57_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_58_2));
                sums_2[4] = _add_57_1;
                float _shfl_xor_sync_60_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[0], 16);
                float _add_59_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59_1) : "f"(sums_2[0]), "f"(_shfl_xor_sync_60_1));
                sums_2[0] = _add_59_1;
                float _shfl_xor_sync_62_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[1], 16);
                float _add_61_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61_1) : "f"(sums_2[1]), "f"(_shfl_xor_sync_62_1));
                sums_2[1] = _add_61_1;
                float _shfl_xor_sync_64_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[2], 16);
                float _add_63_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63_1) : "f"(sums_2[2]), "f"(_shfl_xor_sync_64_1));
                sums_2[2] = _add_63_1;
                float _shfl_xor_sync_66_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[3], 16);
                float _add_65_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65_1) : "f"(sums_2[3]), "f"(_shfl_xor_sync_66_1));
                sums_2[3] = _add_65_1;
                float _shfl_xor_sync_68_1 = __shfl_xor_sync(0xFFFFFFFF, sums_2[4], 16);
                float _add_67_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67_1) : "f"(sums_2[4]), "f"(_shfl_xor_sync_68_1));
                sums_2[4] = _add_67_1;
                float s_1 = 0.0f;
                s_1 = ((lane_in_warp == 0) ? sums_2[0] : s_1);
                s_1 = ((lane_in_warp == 1) ? sums_2[1] : s_1);
                s_1 = ((lane_in_warp == 2) ? sums_2[2] : s_1);
                s_1 = ((lane_in_warp == 3) ? sums_2[3] : s_1);
                float q_1 = sums_2[4];
                float _fma_70;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(q_1), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                float _sqrt_69;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_69) : "f"(_fma_70));
                float r_1 = __frcp_rn(_sqrt_69);
#else
                float _fma_212;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(q_3), "f"(6.1035156e-05f), "f"(hc_norm_eps));
                float _sqrt_211;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_211) : "f"(_fma_212));
                float r_1 = __frcp_rn(_sqrt_211);
#endif
                float z_1;
#if __CUDA_ARCH__ == 1070
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_1));
#else
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_179));
#endif
                float a_map_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map_1) : "f"(z_1), "f"(sc_1), "f"(bs_1));
                float a_1 = ((lane_in_warp < 24) ? a_map_1 : 0.0f);
#if __CUDA_ARCH__ == 1070
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(0.0f), "f"(fabsf(a_1)));
                float exp_1 = __nv_expf(_sub_71);
                float add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(1.0f), "f"(exp_1));
                float fdiv_rn_2 = __fdiv_rn(1.0f, add_31);
                float fdiv_rn_1_1 = __fdiv_rn(exp_1, add_31);
                float sig_1 = ((a_1 >= 0.0f) ? fdiv_rn_2 : fdiv_rn_1_1);
                float sneg_1 = ((a_1 >= 0.0f) ? fdiv_rn_1_1 : fdiv_rn_2);
#else
                float _sub_213;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(0.0f), "f"(fabsf(a_1)));
                float exp = __nv_expf(_sub_213);
                float add_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_43) : "f"(1.0f), "f"(exp));
                float fdiv_rn = __fdiv_rn(1.0f, add_43);
                float fdiv_rn_1 = __fdiv_rn(exp, add_43);
                float sig_1 = ((a_1 >= 0.0f) ? fdiv_rn : fdiv_rn_1);
                float sneg_1 = ((a_1 >= 0.0f) ? fdiv_rn_1 : fdiv_rn);
#endif
                float pre;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pre) : "f"(sig_1), "f"(hc_pre_eps));
                __syncwarp();
                if (lane_in_warp < 4) {
                    tail_scratch[30 + warp / 4 * 4 + lane_in_warp] = pre;
                }
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
            float coeff_1 = ((lane_in_warp < 4) ? tail_scratch[30 + warp / 4 * 4 + lane_in_warp] : 0.0f);
            unsigned int retained[16];
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            #pragma unroll
            for (int pack = 0; pack < 4; pack++) {
                unsigned int hidden_pack = warp % 4 * 4 + (unsigned int)pack;
                unsigned long long off = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack * 256) + (unsigned long long)(lane_in_warp * 8);
                float values[8];
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, coeff_1, 0);
                if (pack < 2) {
                    unsigned int word_3 = early[pack * 16];
                    float lo = __uint_as_float(word_3 << 16);
                    float hi = __uint_as_float(word_3 & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_72 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_73 = make_float2(lo, hi);
#else
                    float2 _f32x2_mul_a_214 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_215 = make_float2(lo, hi);
#endif
                    float2 pair_2;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&_f32x2_mul_b_215));
#endif
                    values[0] = pair_2.x;
                    values[1] = pair_2.y;
                    unsigned int word_0_1 = early[pack * 16 + 1];
                    float lo_1 = __uint_as_float(word_0_1 << 16);
                    float hi_2 = __uint_as_float(word_0_1 & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_74 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_75 = make_float2(lo_1, hi_2);
#else
                    float2 _f32x2_mul_a_216 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_217 = make_float2(lo_1, hi_2);
#endif
                    float2 pair_3;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
#endif
                    values[2] = pair_3.x;
                    values[3] = pair_3.y;
                    unsigned int word_4 = early[pack * 16 + 2];
                    float lo_5 = __uint_as_float(word_4 << 16);
                    float hi_6 = __uint_as_float(word_4 & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_76 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_77 = make_float2(lo_5, hi_6);
#else
                    float2 _f32x2_mul_a_218 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_219 = make_float2(lo_5, hi_6);
#endif
                    float2 pair_7;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
#endif
                    values[4] = pair_7.x;
                    values[5] = pair_7.y;
                    unsigned int word_8 = early[pack * 16 + 3];
                    float lo_9 = __uint_as_float(word_8 << 16);
                    float hi_10 = __uint_as_float(word_8 & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_78 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_79 = make_float2(lo_9, hi_10);
#else
                    float2 _f32x2_mul_a_220 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_221 = make_float2(lo_9, hi_10);
#endif
                    float2 pair_11;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
#endif
                    values[6] = pair_11.x;
                    values[7] = pair_11.y;
                } else {
#if __CUDA_ARCH__ == 1070
                    float _vec_load_26[8];
#else
                    float _vec_load_52[8];
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        const uint4* _vptr_80 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                        uint4 _vld_80[1];
#else
                        const uint4* _vptr_222 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                        uint4 _vld_222[1];
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                                : "=r"(_vld_80[_blk].x), "=r"(_vld_80[_blk].y), "=r"(_vld_80[_blk].z), "=r"(_vld_80[_blk].w) : "l"((const void*)(_vptr_80 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80[_blk]);
#else
                                : "=r"(_vld_222[_blk].x), "=r"(_vld_222[_blk].y), "=r"(_vld_222[_blk].z), "=r"(_vld_222[_blk].w) : "l"((const void*)(_vptr_222 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_222 = reinterpret_cast<uint32_t*>(&_vld_222[_blk]);
#endif
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                                (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                                (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
#else
                                (&_vec_load_52[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_222[_pair]) << 16);
                                (&_vec_load_52[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_222[_pair]) & 0xffff0000u);
#endif
                            }
                        }
                    }
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_81 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_82 = make_float2(_vec_load_26[0], _vec_load_26[1]);
#else
                    float2 _f32x2_mul_a_223 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_224 = make_float2(_vec_load_52[0], _vec_load_52[1]);
#endif
                    float2 pair_4;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&_f32x2_mul_b_224));
#endif
                    values[0] = pair_4.x;
                    values[1] = pair_4.y;
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_83 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_84 = make_float2(_vec_load_26[2], _vec_load_26[3]);
#else
                    float2 _f32x2_mul_a_225 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_226 = make_float2(_vec_load_52[2], _vec_load_52[3]);
#endif
                    float2 pair_0;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&_f32x2_mul_b_226));
#endif
                    values[2] = pair_0.x;
                    values[3] = pair_0.y;
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_85 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_86 = make_float2(_vec_load_26[4], _vec_load_26[5]);
#else
                    float2 _f32x2_mul_a_227 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_228 = make_float2(_vec_load_52[4], _vec_load_52[5]);
#endif
                    float2 pair_1_1;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
#endif
                    values[4] = pair_1_1.x;
                    values[5] = pair_1_1.y;
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_87 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_88 = make_float2(_vec_load_26[6], _vec_load_26[7]);
#else
                    float2 _f32x2_mul_a_229 = make_float2(_shfl_0, _shfl_0);
                    float2 _f32x2_mul_b_230 = make_float2(_vec_load_52[6], _vec_load_52[7]);
#endif
                    float2 pair_2_1;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
#endif
                    values[6] = pair_2_1.x;
                    values[7] = pair_2_1.y;
                }
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, coeff_1, 1);
                if (pack < 2) {
                    unsigned int word_5 = early[pack * 16 + 4];
                    float lo_2 = __uint_as_float(word_5 << 16);
                    float hi_1 = __uint_as_float(word_5 & 4294901760u);
                    float2 pair_6 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(lo_2, hi_1), make_float2(values[0], values[1]));
                    values[0] = pair_6.x;
                    values[1] = pair_6.y;
                    unsigned int word_0_2 = early[pack * 16 + 4 + 1];
                    float lo_1_1 = __uint_as_float(word_0_2 << 16);
                    float hi_2_1 = __uint_as_float(word_0_2 & 4294901760u);
                    float2 pair_3_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(lo_1_1, hi_2_1), make_float2(values[2], values[3]));
                    values[2] = pair_3_1.x;
                    values[3] = pair_3_1.y;
                    unsigned int word_4_1 = early[pack * 16 + 4 + 2];
                    float lo_5_1 = __uint_as_float(word_4_1 << 16);
                    float hi_6_1 = __uint_as_float(word_4_1 & 4294901760u);
                    float2 pair_7_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(lo_5_1, hi_6_1), make_float2(values[4], values[5]));
                    values[4] = pair_7_1.x;
                    values[5] = pair_7_1.y;
                    unsigned int word_8_1 = early[pack * 16 + 4 + 3];
                    float lo_9_1 = __uint_as_float(word_8_1 << 16);
                    float hi_10_1 = __uint_as_float(word_8_1 & 4294901760u);
                    float2 pair_11_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(lo_9_1, hi_10_1), make_float2(values[6], values[7]));
                    values[6] = pair_11_1.x;
                    values[7] = pair_11_1.y;
                } else {
#if __CUDA_ARCH__ == 1070
                    float _vec_load_27[8];
#else
                    float _vec_load_53[8];
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        const uint4* _vptr_89 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                        uint4 _vld_89[1];
#else
                        const uint4* _vptr_231 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                        uint4 _vld_231[1];
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                                : "=r"(_vld_89[_blk].x), "=r"(_vld_89[_blk].y), "=r"(_vld_89[_blk].z), "=r"(_vld_89[_blk].w) : "l"((const void*)(_vptr_89 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
#else
                                : "=r"(_vld_231[_blk].x), "=r"(_vld_231[_blk].y), "=r"(_vld_231[_blk].z), "=r"(_vld_231[_blk].w) : "l"((const void*)(_vptr_231 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_231 = reinterpret_cast<uint32_t*>(&_vld_231[_blk]);
#endif
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                                (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                                (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
#else
                                (&_vec_load_53[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_231[_pair]) << 16);
                                (&_vec_load_53[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_231[_pair]) & 0xffff0000u);
#endif
                            }
                        }
                    }
#if __CUDA_ARCH__ == 1070
                    float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[0], _vec_load_27[1]), make_float2(values[0], values[1]));
#else
                    float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[0], _vec_load_53[1]), make_float2(values[0], values[1]));
#endif
                    values[0] = pair_9.x;
                    values[1] = pair_9.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_0_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[2], _vec_load_27[3]), make_float2(values[2], values[3]));
#else
                    float2 pair_0_1 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[2], _vec_load_53[3]), make_float2(values[2], values[3]));
#endif
                    values[2] = pair_0_1.x;
                    values[3] = pair_0_1.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_1_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[4], _vec_load_27[5]), make_float2(values[4], values[5]));
#else
                    float2 pair_1_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[4], _vec_load_53[5]), make_float2(values[4], values[5]));
#endif
                    values[4] = pair_1_2.x;
                    values[5] = pair_1_2.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_2_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_27[6], _vec_load_27[7]), make_float2(values[6], values[7]));
#else
                    float2 pair_2_2 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(_vec_load_53[6], _vec_load_53[7]), make_float2(values[6], values[7]));
#endif
                    values[6] = pair_2_2.x;
                    values[7] = pair_2_2.y;
                }
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, coeff_1, 2);
                if (pack < 2) {
                    unsigned int word_7 = early[pack * 16 + 8];
                    float lo_3 = __uint_as_float(word_7 << 16);
                    float hi_3 = __uint_as_float(word_7 & 4294901760u);
                    float2 pair_10 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(lo_3, hi_3), make_float2(values[0], values[1]));
                    values[0] = pair_10.x;
                    values[1] = pair_10.y;
                    unsigned int word_0_3 = early[pack * 16 + 8 + 1];
                    float lo_1_2 = __uint_as_float(word_0_3 << 16);
                    float hi_2_2 = __uint_as_float(word_0_3 & 4294901760u);
                    float2 pair_3_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(lo_1_2, hi_2_2), make_float2(values[2], values[3]));
                    values[2] = pair_3_2.x;
                    values[3] = pair_3_2.y;
                    unsigned int word_4_2 = early[pack * 16 + 8 + 2];
                    float lo_5_2 = __uint_as_float(word_4_2 << 16);
                    float hi_6_2 = __uint_as_float(word_4_2 & 4294901760u);
                    float2 pair_7_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(lo_5_2, hi_6_2), make_float2(values[4], values[5]));
                    values[4] = pair_7_2.x;
                    values[5] = pair_7_2.y;
                    unsigned int word_8_2 = early[pack * 16 + 8 + 3];
                    float lo_9_2 = __uint_as_float(word_8_2 << 16);
                    float hi_10_2 = __uint_as_float(word_8_2 & 4294901760u);
                    float2 pair_11_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(lo_9_2, hi_10_2), make_float2(values[6], values[7]));
                    values[6] = pair_11_2.x;
                    values[7] = pair_11_2.y;
                } else {
#if __CUDA_ARCH__ == 1070
                    float _vec_load_28[8];
#else
                    float _vec_load_54[8];
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        const uint4* _vptr_90 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                        uint4 _vld_90[1];
#else
                        const uint4* _vptr_232 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                        uint4 _vld_232[1];
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                                : "=r"(_vld_90[_blk].x), "=r"(_vld_90[_blk].y), "=r"(_vld_90[_blk].z), "=r"(_vld_90[_blk].w) : "l"((const void*)(_vptr_90 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90[_blk]);
#else
                                : "=r"(_vld_232[_blk].x), "=r"(_vld_232[_blk].y), "=r"(_vld_232[_blk].z), "=r"(_vld_232[_blk].w) : "l"((const void*)(_vptr_232 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_232 = reinterpret_cast<uint32_t*>(&_vld_232[_blk]);
#endif
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                                (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                                (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
#else
                                (&_vec_load_54[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_232[_pair]) << 16);
                                (&_vec_load_54[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_232[_pair]) & 0xffff0000u);
#endif
                            }
                        }
                    }
#if __CUDA_ARCH__ == 1070
                    float2 pair_12 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[0], _vec_load_28[1]), make_float2(values[0], values[1]));
#else
                    float2 pair_12 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[0], _vec_load_54[1]), make_float2(values[0], values[1]));
#endif
                    values[0] = pair_12.x;
                    values[1] = pair_12.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_0_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[2], _vec_load_28[3]), make_float2(values[2], values[3]));
#else
                    float2 pair_0_2 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[2], _vec_load_54[3]), make_float2(values[2], values[3]));
#endif
                    values[2] = pair_0_2.x;
                    values[3] = pair_0_2.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_1_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[4], _vec_load_28[5]), make_float2(values[4], values[5]));
#else
                    float2 pair_1_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[4], _vec_load_54[5]), make_float2(values[4], values[5]));
#endif
                    values[4] = pair_1_3.x;
                    values[5] = pair_1_3.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_2_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_28[6], _vec_load_28[7]), make_float2(values[6], values[7]));
#else
                    float2 pair_2_3 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(_vec_load_54[6], _vec_load_54[7]), make_float2(values[6], values[7]));
#endif
                    values[6] = pair_2_3.x;
                    values[7] = pair_2_3.y;
                }
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, coeff_1, 3);
                if (pack < 2) {
                    unsigned int word_9 = early[pack * 16 + 12];
                    float lo_4 = __uint_as_float(word_9 << 16);
                    float hi_4 = __uint_as_float(word_9 & 4294901760u);
                    float2 pair_13 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(lo_4, hi_4), make_float2(values[0], values[1]));
                    values[0] = pair_13.x;
                    values[1] = pair_13.y;
                    unsigned int word_0_4 = early[pack * 16 + 12 + 1];
                    float lo_1_3 = __uint_as_float(word_0_4 << 16);
                    float hi_2_3 = __uint_as_float(word_0_4 & 4294901760u);
                    float2 pair_3_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(lo_1_3, hi_2_3), make_float2(values[2], values[3]));
                    values[2] = pair_3_3.x;
                    values[3] = pair_3_3.y;
                    unsigned int word_4_3 = early[pack * 16 + 12 + 2];
                    float lo_5_3 = __uint_as_float(word_4_3 << 16);
                    float hi_6_3 = __uint_as_float(word_4_3 & 4294901760u);
                    float2 pair_7_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(lo_5_3, hi_6_3), make_float2(values[4], values[5]));
                    values[4] = pair_7_3.x;
                    values[5] = pair_7_3.y;
                    unsigned int word_8_3 = early[pack * 16 + 12 + 3];
                    float lo_9_3 = __uint_as_float(word_8_3 << 16);
                    float hi_10_3 = __uint_as_float(word_8_3 & 4294901760u);
                    float2 pair_11_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(lo_9_3, hi_10_3), make_float2(values[6], values[7]));
                    values[6] = pair_11_3.x;
                    values[7] = pair_11_3.y;
                } else {
#if __CUDA_ARCH__ == 1070
                    float _vec_load_29[8];
#else
                    float _vec_load_55[8];
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        const uint4* _vptr_91 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                        uint4 _vld_91[1];
#else
                        const uint4* _vptr_233 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                        uint4 _vld_233[1];
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                                : "=r"(_vld_91[_blk].x), "=r"(_vld_91[_blk].y), "=r"(_vld_91[_blk].z), "=r"(_vld_91[_blk].w) : "l"((const void*)(_vptr_91 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91[_blk]);
#else
                                : "=r"(_vld_233[_blk].x), "=r"(_vld_233[_blk].y), "=r"(_vld_233[_blk].z), "=r"(_vld_233[_blk].w) : "l"((const void*)(_vptr_233 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_233 = reinterpret_cast<uint32_t*>(&_vld_233[_blk]);
#endif
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1070
                                (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                                (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
#else
                                (&_vec_load_55[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_233[_pair]) << 16);
                                (&_vec_load_55[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_233[_pair]) & 0xffff0000u);
#endif
                            }
                        }
                    }
#if __CUDA_ARCH__ == 1070
                    float2 pair_14 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[0], _vec_load_29[1]), make_float2(values[0], values[1]));
#else
                    float2 pair_14 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[0], _vec_load_55[1]), make_float2(values[0], values[1]));
#endif
                    values[0] = pair_14.x;
                    values[1] = pair_14.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_0_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[2], _vec_load_29[3]), make_float2(values[2], values[3]));
#else
                    float2 pair_0_3 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[2], _vec_load_55[3]), make_float2(values[2], values[3]));
#endif
                    values[2] = pair_0_3.x;
                    values[3] = pair_0_3.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_1_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[4], _vec_load_29[5]), make_float2(values[4], values[5]));
#else
                    float2 pair_1_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[4], _vec_load_55[5]), make_float2(values[4], values[5]));
#endif
                    values[4] = pair_1_4.x;
                    values[5] = pair_1_4.y;
#if __CUDA_ARCH__ == 1070
                    float2 pair_2_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_29[6], _vec_load_29[7]), make_float2(values[6], values[7]));
#else
                    float2 pair_2_4 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(_vec_load_55[6], _vec_load_55[7]), make_float2(values[6], values[7]));
#endif
                    values[6] = pair_2_4.x;
                    values[7] = pair_2_4.y;
                }
                float2 w_pair = make_float2(values[0], values[1]);
                float2 squares_1 = fma_f32x2_rn_noftz(w_pair, w_pair, make_float2(sum0, sum1));
                sum0 = squares_1.x;
                sum1 = squares_1.y;
                retained[pack * 4] = __as_u32(__float22bfloat162_rn(make_float2(values[0], values[1])));
                float2 w_pair_0 = make_float2(values[2], values[3]);
                float2 squares_1_1 = fma_f32x2_rn_noftz(w_pair_0, w_pair_0, make_float2(sum0, sum1));
                sum0 = squares_1_1.x;
                sum1 = squares_1_1.y;
                retained[pack * 4 + 1] = __as_u32(__float22bfloat162_rn(make_float2(values[2], values[3])));
                float2 w_pair_2 = make_float2(values[4], values[5]);
                float2 squares_3 = fma_f32x2_rn_noftz(w_pair_2, w_pair_2, make_float2(sum0, sum1));
                sum0 = squares_3.x;
                sum1 = squares_3.y;
                retained[pack * 4 + 2] = __as_u32(__float22bfloat162_rn(make_float2(values[4], values[5])));
                float2 w_pair_4 = make_float2(values[6], values[7]);
                float2 squares_5 = fma_f32x2_rn_noftz(w_pair_4, w_pair_4, make_float2(sum0, sum1));
                sum0 = squares_5.x;
                sum1 = squares_5.y;
                retained[pack * 4 + 3] = __as_u32(__float22bfloat162_rn(make_float2(values[6], values[7])));
            }
#if __CUDA_ARCH__ == 1070
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(sum0), "f"(sum1));
            float _warp_reduce_0 = _add_92;
#else
            float _add_234;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(sum0), "f"(sum1));
            float _warp_reduce_0 = _add_234;
#endif
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            if (lane_in_warp == 0) {
                tail_scratch[warp] = _warp_reduce_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
            float row_sum = ((lane_in_warp < 4) ? tail_scratch[warp / 4 * 4 + lane_in_warp] : 0.0f);
#if __CUDA_ARCH__ == 1070
            float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            float add_32;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_32) : "f"(row_sum), "f"(_shfl_xor_sync_93));
            float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, add_32, 1);
            float _add_94;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(add_32), "f"(_shfl_xor_sync_95));
            row_sum = _add_94;
            float _shfl_sync_96 = __shfl_sync(0xFFFFFFFF, row_sum, 0);
            row_sum = _shfl_sync_96;
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(row_sum), "f"(0.00024414062f), "f"(rmsnorm_eps));
            float _sqrt_98;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_98) : "f"(_fma_99));
            float _rcp_97 = __frcp_rn(_sqrt_98);
#else
            float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            float add_44;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_44) : "f"(row_sum), "f"(_shfl_xor_sync_235));
            float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, add_44, 1);
            float _add_236;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(add_44), "f"(_shfl_xor_sync_237));
            row_sum = _add_236;
            float _shfl_sync_238 = __shfl_sync(0xFFFFFFFF, row_sum, 0);
            row_sum = _shfl_sync_238;
            float _fma_241;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(row_sum), "f"(0.00024414062f), "f"(rmsnorm_eps));
            float _sqrt_240;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_240) : "f"(_fma_241));
            float _rcp_239 = __frcp_rn(_sqrt_240);
#endif
            float rstd;
#if __CUDA_ARCH__ == 1070
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_97), "f"(rmsnorm_scale));
#else
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_239), "f"(rmsnorm_scale));
#endif
            #pragma unroll
            for (int pack_1 = 0; pack_1 < 4; pack_1++) {
                unsigned int hidden_pack_1 = warp % 4 * 4 + (unsigned int)pack_1;
                unsigned int next_weight[4];
                if (pack_1 + 1 < 4) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&next_weight[0])), "=r"(*reinterpret_cast<uint32_t*>(&next_weight[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&next_weight[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&next_weight[(0) + 3]))
                        : "r"(weight_words_addr + ((hidden_pack_1 + 1) * 512 + lane_in_warp * 16)));
                }
                unsigned int normalized_words[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned int word_10 = retained[pack_1 * 4 + j];
                    float lo_6 = __uint_as_float(word_10 << 16);
                    float hi_5 = __uint_as_float(word_10 & 4294901760u);
                    float weight_lo = __uint_as_float(weight_cache[j] << 16);
                    float weight_hi = __uint_as_float(weight_cache[j] & 4294901760u);
#if __CUDA_ARCH__ == 1070
                    float2 _f32x2_mul_a_100 = make_float2(lo_6, hi_5);
                    float2 _f32x2_mul_b_101 = make_float2(rstd, rstd);
#else
                    float2 _f32x2_mul_a_242 = make_float2(lo_6, hi_5);
                    float2 _f32x2_mul_b_243 = make_float2(rstd, rstd);
#endif
                    float2 scaled;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
                    float2 _f32x2_mul_b_102 = make_float2(weight_lo, weight_hi);
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_242), "l"(*(const unsigned long long*)&_f32x2_mul_b_243));
                    float2 _f32x2_mul_b_244 = make_float2(weight_lo, weight_hi);
#endif
                    float2 weighted;
#if __CUDA_ARCH__ == 1070
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_244));
#endif
                    normalized_words[j] = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                }
                unsigned long long off_1 = (unsigned long long)token_1 * 4096 + (unsigned long long)(hidden_pack_1 * 256) + (unsigned long long)(lane_in_warp * 8);
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %6, 0; @p_ st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5; }" :: "l"(y_bf16 + off_1), "r"(normalized_words[(0) + 0]), "r"(normalized_words[(0) + 1]), "r"(normalized_words[(0) + 2]), "r"(normalized_words[(0) + 3]), "l"(0x12F0000000000000ULL), "r"((unsigned int)(y_fit)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y_bf16 + off_1), "r"(normalized_words[(0) + 0]), "r"(normalized_words[(0) + 1]), "r"(normalized_words[(0) + 2]), "r"(normalized_words[(0) + 3]), "r"((unsigned int)(!y_fit)) : "memory");
                if (pack_1 + 1 < 4) {
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        weight_cache[j_1] = next_weight[j_1];
                    }
                }
            }
        }
        if (token_1 >= num_tokens) {
            break;
        }
    }
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 0, 768;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
            asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(launch_epochs) + (bid))), "l"(static_cast<unsigned long long>(1)) : "memory");
            #elif defined(__CUDA_ARCH__)
            #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
            #endif
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 3) {
#if __CUDA_ARCH__ == 1070
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
#else
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
#endif
    }
}

} // extern "C"
